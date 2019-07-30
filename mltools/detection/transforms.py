"""
Contains several transform classes which can be passed to a VOCXMLDataset to transform the image and
if necessary the bounding boxes, e.g. to apply random affine transformations for data augmentation.
"""
from math import radians
from random import random

from PIL import Image
from cv2 import UMat, resize, warpAffine
from numpy import abs, array, clip, cos, sin, transpose
from torch import as_tensor, cat, float32, max, min, ones, stack
from torchvision import transforms
from torchvision.transforms.functional import to_tensor


class ImageTransform:
    """Base class for transformations that either only apply to images, or need to use a standard torchvision
    transform's __repr__ function."""

    def __init__(self, image_transform):
        """
        :param image_transform: (torchvision transform) the transform to wrap
        """
        self.image_transform = image_transform

    def __call__(self, image, target):
        return self.image_transform(image), target

    def __repr__(self):
        return self.image_transform.__repr__()


class RandomAffine(ImageTransform):
    """Transform class to apply a random affine transformation to images and corresponding bounding boxes.
    Can be passed to a VOCXMLDataset."""
    __doc__ = transforms.RandomAffine.__doc__

    def __init__(self, degrees, translate=None, scale=None, shear=None, expand_to_fit=True):
        super(RandomAffine, self).__init__(transforms.RandomAffine(degrees, translate, scale, shear))
        self.expand_to_fit = expand_to_fit

    def __call__(self, image, target):
        image_size = (image.shape[2], image.shape[1])
        # 1. Transform image
        # Get the actually chosen parameters used for the affine transformation.
        params = transforms.RandomAffine.get_params(
            self.image_transform.degrees,
            self.image_transform.translate,
            self.image_transform.scale,
            self.image_transform.shear,
            image_size
        )

        center = (image_size[0] / 2, image_size[1] / 2)
        affine_matrix, transformed_image_size = self._create_affine_matrix(center, image_size,
                                                                           *params, self.expand_to_fit)
        # cv2.warpAffine expects the image to be a UMat object.
        image_umat = UMat(transpose(image.numpy(), [1, 2, 0]))
        # Apply the transformation to the image.
        transformed_image = warpAffine(image_umat, affine_matrix, transformed_image_size).get()

        # Resize image to original size and convert back to torch-tensor
        image = to_tensor(resize(transformed_image, image_size))

        # 2. Transform boxes
        # K = number of boxes
        boxes = target["boxes"]  # Shape (K, 4)
        box_coordinate_matrix = self._create_box_coordinate_matrix(boxes)  # Shape (3, 4*K)
        # Convert the affine matrix to torch tensor to be able to perform matrix multiplication.
        affine_matrix = as_tensor(affine_matrix, dtype=float32)  # Shape (2, 3)
        boxes_transformed = affine_matrix.mm(box_coordinate_matrix).transpose(0, 1).reshape(-1, 8)  # Shape (K, 8)
        boxes = self._create_enclosure_boxes(boxes_transformed)  # Shape (K, 4)

        # Scale down boxes.
        self.scale_boxes(boxes, transformed_image_size, image_size)
        # Clip boxes so they lie completely inside the image.
        self.clip_boxes(boxes, image_size)

        target["boxes"] = boxes

        return image, target

    @staticmethod
    def scale_boxes(boxes, old_image_size, new_image_size):
        """
        Rescales bounding boxes when the image size changes.

        :param boxes: (tensor) the bounding boxes
        :param old_image_size: (tuple(int [width], int [height]))
        :param new_image_size: (tuple(int [width], int [height]))
        """
        scale_w = new_image_size[0] / old_image_size[0]
        scale_h = new_image_size[1] / old_image_size[1]

        boxes *= as_tensor([scale_w, scale_h, scale_w, scale_h], dtype=float32)

    @staticmethod
    def clip_boxes(boxes, image_size):
        """
        Clips bounding box coordinates so that the boxes remain within the image.

        :param boxes: (tensor) the bounding boxes
        :param image_size: (tuple(int [width], int [height]))
        """
        boxes[:, [0, 2]] = clip(boxes[:, [0, 2]], 0, image_size[0])
        boxes[:, [1, 3]] = clip(boxes[:, [1, 3]], 0, image_size[1])

    @staticmethod
    def _create_enclosure_boxes(box_coordinate_matrix):
        x = box_coordinate_matrix[:, [0, 2, 4, 6]]
        y = box_coordinate_matrix[:, [1, 3, 5, 7]]

        x_min = min(x, dim=1).values
        y_min = min(y, dim=1).values
        x_max = max(x, dim=1).values
        y_max = max(y, dim=1).values

        return stack([x_min, y_min, x_max, y_max], dim=1)

    @staticmethod
    def _create_box_coordinate_matrix(boxes):
        """ Creates a matrix with the following structure:

        [ x1_1 x1_2 x1_3 x1_4 x2_1 ...]

        [ y1_1 y1_2 y1_3 y1_4 y2_1 ...]

        [ 1    1    1    1    1    ...]

        where (xi_j, yi_j) are the coordinates of the jth corner point
        (counterclockwise order starting from the top left) of the ith box.
        """
        box_coordinates = stack(
            [boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2], boxes[:, 3]],
            dim=1).reshape(-1, 2).transpose(0, 1)

        return cat([box_coordinates, ones((1, box_coordinates.shape[1]), dtype=float32)])

    @staticmethod
    def _create_affine_matrix(center, img_size, angle, translate, scale, shear, expand_to_fit=False):
        angle = radians(angle)
        shear = radians(shear)

        a = scale * cos(angle + shear)
        b = scale * sin(angle + shear)
        d = - scale * sin(angle)
        e = scale * cos(angle)

        if abs(angle) > 0 or abs(shear) > 0 and expand_to_fit:
            (w, h) = img_size
            new_img_size = (int(round(h * abs(b) + w * abs(a))), int(round(h * abs(a) + w * abs(b))))
            if abs(angle) == 0:
                new_img_size = (new_img_size[0], img_size[1])
            new_center = (new_img_size[0] / 2, new_img_size[1] / 2)
        else:
            new_center = center
            new_img_size = img_size

        transform_matrix = \
            array([[a, b, new_center[0] + translate[0] - center[0] * a - center[1] * b],
                   [d, e, new_center[1] + translate[1] - center[0] * d - center[1] * e]])

        return transform_matrix, new_img_size


class RandomHorizontalFlip:
    """Performs a random horizontal flip to images and corresponding bounding boxes.
    Can be passed to a VOCXMLDataset."""

    def __init__(self, prob):
        """
        :param prob: (float) the probability of the flip being applied
        """
        self.prob = prob

    def __call__(self, image, target):
        if random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target

    def __repr__(self):
        return self.__class__.__name__ + '(prob={})'.format(self.prob)


class RandomVerticalFlip:
    """Performs a random vertical flip to images and corresponding bounding boxes.
    Can be passed to a VOCXMLDataset."""

    def __init__(self, prob):
        """
        :param prob: (float) the probability of the flip being applied
        """
        self.prob = prob

    def __call__(self, image, target):
        if random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
        return image, target

    def __repr__(self):
        return self.__class__.__name__ + '(prob={})'.format(self.prob)


class Resize(ImageTransform):
    """Transform to resize images and corresponding bounding boxes. Can be passed to a VOCXMLDataset."""

    def __init__(self, size, interpolation=Image.BILINEAR):
        """
        :param size: (tuple(int, int)) the target image size
        :param interpolation: (cv2 Interpolation, default=Image.BILINEAR) which interpolation to use for resizing
        """
        super(Resize, self).__init__(transforms.Resize(size, interpolation))

    def __call__(self, image, target):
        (old_width, old_height) = image.size

        image = self.image_transform(image)

        (new_width, new_height) = image.size

        scale_w = new_width / old_width
        scale_h = new_height / old_height

        target["boxes"] *= as_tensor([scale_w, scale_h, scale_w, scale_h], dtype=float32)

        return image, target


class Normalize(ImageTransform):
    """Transform to normalize images. Can be passed to a VOCXMLDataset."""

    def __init__(self, mean, std, inplace=False):
        """
        :param mean: (float)
        :param std: (float)
        :param inplace: (boolean, default=False)
        """
        super(Normalize, self).__init__(transforms.Normalize(mean, std, inplace))


class ToTensor(ImageTransform):
    """Transform to turn PIL-images to tensors. Can be passed to a VOCXMLDataset."""

    def __init__(self):
        super(ToTensor, self).__init__(transforms.ToTensor())


class Compose:
    """Aggregates several transforms into a single transform object, calling the contained transforms one
    after the other when applied to images and bounding boxes."""

    def __init__(self, transforms):
        """
        :param transforms: (list) a list of transforms to compose
        """
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '        {0}'.format(t)
        format_string += '\n    )'
        return format_string
