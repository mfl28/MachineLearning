"""
Contains utility functions which can come in handy when working with images.
"""
from concurrent.futures import ThreadPoolExecutor

from cv2 import COLOR_BGR2RGB, INTER_CUBIC, cvtColor, imread, imwrite, resize
from sklearn.model_selection import train_test_split
import pandas as pd


def resize_image(image_path, size, interpolation=INTER_CUBIC, to_rgb=False):
    """
    Resizes a single image.
    :param image_path: (pathlib.Path) the path of the image file
    :param size: (tuple(int [width], int [height])) the target image size
    :param interpolation: (cv2 Interpolation, default=INTER_CUBIC) which interpolation to use for resizing
    :param to_rgb: (boolean, default=False) whether the image needs to be converted to rgb (from bgr)
    :return: the resized image
    """
    img = imread(str(image_path))

    if to_rgb:
        img = cvtColor(img, COLOR_BGR2RGB)

    return resize(img, size, interpolation=interpolation)


def resize_and_save_image(image_path, size, destination_dir, interpolation=INTER_CUBIC):
    """
    Resizes an image and saves the result to a file.
    :param image_path: (pathlib.Path) the path of the image file
    :param size: (tuple(int [width], int [height])) the target image size
    :param destination_dir: (pathlib.Path) the path of the directory where the resized image should be saved
    :param interpolation: (cv2 Interpolation, default=INTER_CUBIC) which interpolation to use for resizing
    """
    imwrite(str(destination_dir / image_path.name), resize_image(image_path, size, interpolation))


def resize_images_from_folder(source_dir, size, destination_dir=None, interpolation=INTER_CUBIC, num_workers=1):
    """
    Resizes all images in a given directory and stores the resized images in a destination folder.
    :param source_dir: (pathlib.Path) the path of the source images folder
    :param size: (int or tuple(int [width], int [height])) the desired output size of the images
    :param destination_dir: (optional, pathlib.Path) the path to a folder where the resized images will be saved to
    :param interpolation: (cv2 Interpolation, default=INTER_CUBIC) the type of interpolation to be used for resizing
    :param num_workers: (int, default=1) the number of threads to use
    """
    filename_iterator = source_dir.iterdir()

    if not filename_iterator:
        print(f"The source directory at {str(source_dir)} is empty.")
        return

    if isinstance(size, int):
        size = (size, size)

    if destination_dir is not None:
        save_folder = destination_dir
        if not destination_dir.exists():
            save_folder.mkdir()
    else:
        save_folder = source_dir.parent / f'{source_dir.name}_{size[0]}x{size[1]}'
        save_folder.mkdir()

    with ThreadPoolExecutor(num_workers) as tpe:
        tpe.map(lambda path: resize_and_save_image(path, size, save_folder, interpolation), filename_iterator)
    print(f"Resized images were saved to {save_folder}.")


def stratified_train_test_split_from_image_folder(image_folder_path, test_size):
    data = [{"file": file, "class_name": file.parent.name} for class_path in image_folder_path.iterdir()
            for file in class_path.iterdir() if file.is_file()]

    df = pd.DataFrame(data)
    df_train, df_test = train_test_split(df, test_size=test_size, shuffle=True, stratify=df["class_name"])

    train_path = image_folder_path / "train"
    test_path = image_folder_path / "test"

    _move_images_by_category(df_train.file, train_path)
    _move_images_by_category(df_test.file, test_path)

    for class_name in df.class_name.unique():
        (image_folder_path / class_name).rmdir()


def _move_images_by_category(files, destination_dir):
    for file in files:
        destination = destination_dir.absolute() / f"{file.parent.name}"

        if not destination.exists():
            destination.mkdir(parents=True)

        file.rename(destination / f"{file.name}")


