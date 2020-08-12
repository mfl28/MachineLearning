"""
Models specific to semantic segmentation.
"""
import torch
import torch.nn as nn
import torchvision

class UNetDoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, activation, padding=1):
        super(UNetDoubleConvBlock, self).__init__();
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Dropout2d(p=dropout)
        )

    def forward(self, X):
        return self.conv(X)

class UNetExpansionBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout, activation, padding=1, use_recurrent_block=False):
        super(UNetExpansionBlock, self).__init__()
        if use_recurrent_block:
            self.conv = UNetRRCNNBlock(in_size, out_size, dropout=dropout, padding=padding)
        else:
            self.conv = UNetDoubleConvBlock(in_size, out_size, dropout, activation, padding)

        self.upconv = nn.ConvTranspose2d(out_size, out_size,
                                         kernel_size=2, stride=2)
        self.relu = activation

    def forward(self, X_up, X_down):
        X = torch.cat((X_up, X_down), 1)
        X = self.conv(X)
        X = self.upconv(X)
        X = self.relu(X)
        return X

class UNetAttentionBlock(nn.Module):
    """
    Attention Block

    Based on:
    https://arxiv.org/abs/1804.03999
    """
    def __init__(self, F_g, F_l, F_int):
        super(UNetAttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.Psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        W_g_out = self.W_g(g)
        W_x_out = self.W_x(x)

        out = self.relu(W_g_out + W_x_out)
        out = self.Psi(out)

        return x*out

class UNetRecurrentBlock(nn.Module):
    def __init__(self, n_channels, dropout, rec_depth=3, padding=1):
        super(UNetRecurrentBlock, self).__init__()
        self.rec_depth = rec_depth
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=padding),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, X):
        X_next = self.conv(X)

        for i in range(self.rec_depth - 1):
            X_next = self.conv(X + X_next)

        return X_next

class UNetRRCNNBlock(nn.Module):
    """
    Recurrent Residual Convolutional Block

    Based on:
    https://arxiv.org/pdf/1802.06955.pdf
    """
    def __init__(self, in_size, out_size, dropout, rec_depth=3, padding=1):
        super(UNetRRCNNBlock, self).__init__()
        self.in_conv = nn.Conv2d(in_size, out_size, kernel_size=1)
        self.rec_conv = nn.Sequential(
            UNetRecurrentBlock(out_size, dropout, rec_depth, padding=padding),
            UNetRecurrentBlock(out_size, dropout, rec_depth, padding=padding)
        )

    def forward(self, X):
        X = self.in_conv(X)
        X_next = self.rec_conv(X)

        return X + X_next

class UNet(nn.Module):
    """
    UNet model

    Architecture based on:
    https://arxiv.org/abs/1505.04597 (UNet Paper)

    Implementation inspirations:
    https://stackoverflow.com/questions/52235520/how-to-use-pnasnet5-as-encoder-in-unet-in-pytorch
    https://github.com/jaxony/unet-pytorch/blob/master/model.py

    Hypercolums:
    https://arxiv.org/abs/1411.5752
    """
    def __init__(self, n_classes=1,
                 input_channels=3,
                 adaptiveInputPadding=True,
                 pretrained_encoder=True,
                 freeze_encoder=True,
                 activation=nn.ReLU(inplace=True),
                 dropout=0.2,
                 use_hypercolumns=False,
                 use_attention=False,
                 use_recurrent_decoder_blocks=False):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        encoder_backbone = torchvision.models.resnet34(pretrained=pretrained_encoder)
        encoder_backbone.relu = activation
        self.n_classes = n_classes
        self.adaptiveInputPadding = adaptiveInputPadding
        self.use_hypercolumns = use_hypercolumns
        self.use_attention = use_attention

        d1 = []

        if self.input_channels != 3:
            if pretrained_encoder:
                d1.extend([nn.Conv2d(self.input_channels, 3, 1, bias=False), nn.BatchNorm2d(3)])
            else:
                encoder_backbone.conv1 = nn.Conv2d(self.input_channels,
                                                   encoder_backbone.conv1.out_channels,
                                                   kernel_size=7, stride=2, padding=3, bias=False)


        d1.extend([encoder_backbone.conv1,
                   encoder_backbone.bn1,
                   encoder_backbone.relu,
                   encoder_backbone.layer1])
        d1 = nn.Sequential(*d1)
        d2 = encoder_backbone.layer2
        d3 = encoder_backbone.layer3
        d4 = encoder_backbone.layer4

        self.encoder = nn.ModuleList([d1, d2, d3, d4])
        self.input_size_reduction_factor = 2 ** len(self.encoder)

        bottom_channel_nr = self._calculate_bottom_channel_number()

        self.center = UNetDoubleConvBlock(bottom_channel_nr,
                                          bottom_channel_nr,
                                          dropout, activation)

        if self.use_hypercolumns:
            self.hyper_col_convs = nn.ModuleList([nn.Conv2d(bottom_channel_nr, out_channels=3, kernel_size=3, padding=1)])

        if self.use_attention:
            self.attention_layers = nn.ModuleList([])

        layer_in = bottom_channel_nr * 2
        layer_out = bottom_channel_nr // 2

        self.decoder = nn.ModuleList([])

        for i in range(len(self.encoder)):
            if self.use_attention:
                self.attention_layers.append(UNetAttentionBlock(layer_in // 2, layer_in // 2, layer_in // 2))

            self.decoder.append(UNetExpansionBlock(layer_in, layer_out, dropout, activation,
                                                   use_recurrent_block=use_recurrent_decoder_blocks))

            if i != len(self.encoder) - 1:
                if self.use_hypercolumns:
                    self.hyper_col_convs.append(nn.Conv2d(layer_out, out_channels=3, kernel_size=3, padding=1))
                layer_in //= 2
                layer_out //= 2


        hyper_col_num_channels = len(self.hyper_col_convs) * 3 if self.use_hypercolumns else 0

        self.out_conv = nn.Conv2d(layer_out + hyper_col_num_channels, n_classes, 1)

        if freeze_encoder:
            self.freeze_encoder()

    def forward(self, X):
        d_outputs = []

        if self.adaptiveInputPadding:
            X, padding_lr, padding_tb = self._adaptive_padding(X)

        for d in self.encoder:
            X = d(X)
            d_outputs.append(X)

        if self.use_hypercolumns:
            u_outputs = []

        X = self.center(X)

        for i, (u, d_out) in enumerate(zip(self.decoder, reversed(d_outputs))):
            if self.use_hypercolumns:
                hyper_col_output = F.interpolate(self.hyper_col_convs[i](X),
                                                scale_factor=2**(len(self.decoder) - i),
                                                mode="bilinear",
                                                align_corners=False)
                u_outputs.append(hyper_col_output)

            if self.use_attention:
                d_out = self.attention_layers[i](d_out, X)

            X = u(X, d_out)

        if self.use_hypercolumns:
            u_outputs.append(X)
            X = torch.cat(u_outputs, 1)

        X = self.out_conv(X)

        if self.adaptiveInputPadding:
            if padding_lr != (0,0) or padding_tb != (0,0):
                X = self._center_crop(X, padding_lr, padding_tb)

        return X

    def freeze_encoder(self, last_layer_to_freeze=None):
        if last_layer_to_freeze is None:
            last_layer_to_freeze = len(self.encoder) - 1

        for layer_index in range(last_layer_to_freeze + 1):
            for child in self.encoder[layer_index]:
                for parameter in child.parameters():
                    parameter.requires_grad = False

    def unfreeze_encoder(self, first_layer_to_unfreeze=None):
        if first_layer_to_unfreeze is None:
            first_layer_to_unfreeze = 0

        for layer_index in range(first_layer_to_unfreeze, len(self.encoder)):
            for child in self.encoder[layer_index]:
                for parameter in child.parameters():
                    parameter.requires_grad = True

    def encoder_freeze_status(self):
        frozen_status = [all([all([not param.requires_grad for param in child.parameters()])
                         for child in layer.children()]) for layer in self.encoder]

        return ["frozen" if status else "unfrozen" for status in frozen_status]

    def _adaptive_padding(self, X):
        X_height, X_width = X.shape[2:4]

        if X_width % self.input_size_reduction_factor != 0 or X_height % self.input_size_reduction_factor != 0:
            required_padding_left_right =  self._calculate_required_adaptive_padding(X_width)
            required_padding_top_bottom = self._calculate_required_adaptive_padding(X_height)

            X = nn.ReflectionPad2d(required_padding_left_right + required_padding_top_bottom)(X)
            return X, required_padding_left_right, required_padding_top_bottom

        return X, (0,0), (0,0)

    def _calculate_required_adaptive_padding(self, side_length):
        next_same_output_producing_size = self.input_size_reduction_factor * (side_length // self.input_size_reduction_factor  + 1)
        required_padding_space = next_same_output_producing_size - side_length

        required_padding = (required_padding_space // 2, required_padding_space // 2 + (required_padding_space % 2))

        return required_padding

    def _center_crop(self, X, lr_offset, tb_offset):
        X_height, X_width = X.shape[2:4]
        return X[:, :, tb_offset[0] : (X_height - tb_offset[1]), lr_offset[0] : (X_width - lr_offset[1])]

    def _calculate_bottom_channel_number(self):
        return [l for l in self.encoder[-1][-1].children() if isinstance(l, nn.Conv2d)][-1].out_channels
