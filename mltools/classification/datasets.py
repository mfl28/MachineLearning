"""
Image classification data-sets all subclassing Pytorch's Dataset.
"""
import random

from PIL import Image
from numpy import asarray, dtype, uint8
from pandas import SparseDtype, read_csv
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class CSVImageDataset(Dataset):
    """
    A dataset for image classification, where (grayscale) image data is provided
    via a .csv-file. Can be used to hold e.g. MNIST data.
    """

    def __init__(self, csv_file, img_size, transform=None, mapping=None, train=True, as_type=None):
        """
        :param csv_file: The file containing the image data row-wise, i.e. one image per row
                        (first column = image-id, second column = label, subsequent columns
                         contain unrolled pixel values in range 0 - 255).
        :param img_size: The size of the images (e.g.: MNIST: (28,28))
        :param transform: An optional transform to apply to the images (e.g.: for data augmentation)
        :param mapping: An optional index mapping (e.g.: to select certain indices for testing)
        :param train: Whether __getitem__ returns image and label (while training) or just image (testing)
        :param as_type: Optionally, cast the data_frame to a new type.
        """
        self.data_frame = read_csv(csv_file)

        if as_type is not None:
            self.data_frame = self.data_frame.astype(as_type)
            self.element_dtype = dtype(as_type._dtype) if isinstance(as_type, SparseDtype) else dtype(as_type)
        else:
            self.element_dtype = uint8

        self.img_size = img_size
        self.transform = transform
        self.mapping = mapping
        self.train = train

    def __len__(self):
        return len(self.mapping) if self.mapping else len(self.data_frame)

    def __getitem__(self, idx):
        im_id = self.mapping[idx] if self.mapping else idx

        if self.train:
            image = Image.fromarray(asarray(self.data_frame.iloc[im_id, 1:].values, dtype=self.element_dtype)
                                    .reshape(self.img_size))
            label = int(self.data_frame.iloc[im_id, 0])
        else:
            image = Image.fromarray(asarray(self.data_frame.iloc[im_id].values, dtype=self.element_dtype)
                                    .reshape(self.img_size))

        if self.transform:
            image = self.transform(image)

        if self.train:
            return image, label
        else:
            return image


class DataframeImageDataset(Dataset):
    """
    A dataset for image classification, where (grayscale) image data is provided
    via a pandas.DataFrame. Can be used to hold e.g. MNIST data.
    """

    def __init__(self, data_frame, img_size, transform=None, mapping=None, train=True, as_type=None):
        """
        :param data_frame: The pandas.DataFrame containing the image data row-wise,
                            i.e. one image per row (first column = image-id, second column = label,
                            subsequent columns contain unrolled pixel values in range 0 - 255).
        :param img_size: The size of the images (e.g.: MNIST: (28,28))
        :param transform: An optional transform to apply to the images (e.g.: for data augmentation)
        :param mapping: An optional index mapping (e.g.: to select certain indices for testing)
        :param train: Whether __getitem__ returns image and label (while training) or just image (testing)
        :param as_type: Optionally, cast the data_frame to a new type.
        """
        self.data_frame = data_frame

        if as_type is not None:
            self.data_frame = self.data_frame.astype(as_type)
            self.element_dtype = dtype(as_type._dtype) if isinstance(as_type, SparseDtype) else dtype(as_type)
        else:
            self.element_dtype = uint8

        self.img_size = img_size
        self.transform = transform
        self.mapping = mapping
        self.train = train

    def __len__(self):
        return len(self.mapping) if self.mapping else len(self.data_frame)

    def __getitem__(self, idx):
        im_id = self.mapping[idx] if self.mapping else idx

        if self.train:
            image = Image.fromarray(asarray(self.data_frame.iloc[im_id, 1:].values, dtype=self.element_dtype)
                                    .reshape(self.img_size))
            label = int(self.data_frame.iloc[im_id, 0])
        else:
            image = Image.fromarray(asarray(self.data_frame.iloc[im_id].values, dtype=self.element_dtype)
                                    .reshape(self.img_size))

        if self.transform:
            image = self.transform(image)

        if self.train:
            return image, label
        else:
            return image


class ImagePairDataset(Dataset):
    def __init__(self, image_folder_dir, transform=None):
        super(ImagePairDataset, self).__init__()
        self.images_dataset = ImageFolder(root=image_folder_dir)
        self.transform = transform

    def __getitem__(self, index):
        first_image_data = self.images_dataset.imgs[index]

        different_class = random.randint(0, 1)
        second_image_data = (random.choice([image for image in self.images_dataset.imgs
                                            if image[1] == first_image_data[1]])
                             if not different_class else
                             random.choice([image for image in self.images_dataset.imgs
                                            if image[1] != first_image_data[1]]))

        first_image = Image.open(first_image_data[0])
        second_image = Image.open(second_image_data[0])

        if self.transform is not None:
            first_image = self.transform(first_image)
            second_image = self.transform(second_image)

        return first_image, second_image, different_class

    def __len__(self):
        return len(self.images_dataset)
