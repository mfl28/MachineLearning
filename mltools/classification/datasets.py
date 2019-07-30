"""
Image classification data-sets all subclassing Pytorch's Dataset.
"""

from PIL import Image
from numpy import uint8
from pandas import read_csv
from torch.utils.data import Dataset


class CSVImageDataset(Dataset):
    """
    A dataset for image classification, where (grayscale) image data is provided
    via a .csv-file. Can be used to hold e.g. MNIST data.
    """

    def __init__(self, csv_file, img_size, transform=None, mapping=None, train=True):
        """
        :param csv_file: The file containing the image data row-wise, i.e. one image per row
                        (first column = image-id, second column = label, subsequent columns
                         contain unrolled pixel values in range 0 - 255).
        :param img_size: The size of the images (e.g.: MNIST: (28,28))
        :param transform: An optional transform to apply to the images (e.g.: for data augmentation)
        :param mapping: An optional index mapping (e.g.: to select certain indices for testing)
        :param train: whether __getitem__ returns image and label (while training) or just image (testing)
        """
        self.data_frame = read_csv(csv_file)
        self.img_size = img_size
        self.transform = transform
        self.mapping = mapping
        self.train = train

    def __len__(self):
        return len(self.mapping) if self.mapping else len(self.data_frame)

    def __getitem__(self, idx):
        im_id = self.mapping[idx] if self.mapping else idx

        if self.train:
            image = Image.fromarray(self.data_frame.iloc[im_id, 1:].values.reshape(self.img_size).astype(uint8))
            label = self.data_frame.iloc[im_id, 0]
        else:
            image = Image.fromarray(self.data_frame.iloc[im_id].values.reshape(self.img_size).astype(uint8))

        if self.transform:
            image = self.transform(image)

        if self.train:
            return image, label
        else:
            return image
