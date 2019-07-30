"""
Contains utility functions for object detection tasks.
"""
from xml.etree.ElementTree import parse

from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm, subplots
from numpy import size
from pandas import DataFrame
from pathlib import Path


def display_prediction(image, prediction, classes, color_function=None, figsize=None):
    """
    Displays an image and the bounding box predictions returned by evaluating a model.

    :param image: (PIL image)
    :param prediction: predictions returned when evaluation a model using the image
    :param classes: (list) class names
    :param color_function: a function mapping labels (starting at 1) to colors
    :param figsize: (optional, tuple(int, int)) the size of resulting figure as a tuple of two integers
    """
    fig, ax = subplots(figsize=figsize)
    ax.imshow(image)
    ax.set_title(image.name)

    boxes = prediction[0]["boxes"]
    labels = prediction[0]["labels"]

    if color_function is None:
        color_function = cm.get_cmap("hsv", len(classes))

    if not size(boxes.values) == 0:
        _create_bounding_box_patches(ax, boxes, labels, classes, color_function)
    ax.axis('off')


def parse_voc_annotations(annotations_source):
    """
    Parses .xml annotation files in the Pascal VOC format and creates a pandas DataFrame containing information on
    the images and annotations.

    :param annotations_source: (pathlib.Path or Iterable) a source for the annotation files
    :return: a pandas.DataFrame containing the parsed information
    """
    if isinstance(annotations_source, Path):
        result = [_parse_annotation(annotation_file) for annotation_file in sorted(annotations_source.iterdir())
                  if annotation_file.suffix == ".xml"]
    else:
        result = [_parse_annotation(annotation_file) for annotation_file in annotations_source]

    return DataFrame(result, columns=["filename", "width", "height", "nr_boxes", "classes"])


def create_boxes_per_class_data_frame(df):
    """
    Constructs a DataFrame containing the number of bounding boxes for every class.

    :param df: (pandas.DataFrame) a dataframe returned by parse_voc_annotations()
    :return: a pandas.DataFrame containing the sought information
    """
    classes = set.union(*df.classes)
    count_dict = {class_name: len(df.classes[df.classes.map(lambda box_classes: class_name in box_classes)])
                  for class_name in classes}
    return DataFrame([count_dict], columns=sorted(count_dict), index=["nr_boxes"]).T


def _parse_annotation(annotation_file):
    root = parse(annotation_file).getroot()

    entry = {"filename": root.find("filename").text}

    for size_element in root.find("size"):
        entry[size_element.tag] = int(size_element.text)

    entry["nr_boxes"] = sum(1 for _ in root.iter("object")) + sum(1 for _ in root.iter("part"))
    entry["classes"] = {class_name.text for class_name in root.iter("name")}
    return entry


def _create_bounding_box_patches(ax, boxes, labels, classes, color_function):
    """
    Creates bounding box patches and adds them to a matplotlib.Axes object.
    :param ax: the matplotlib.Axes object to which to add the box patches
    :param boxes: a torch Tensor containing the coordinates [xmin, ymin, xmax, ymax] of the bounding boxes
                  with respect to the image
    :param labels: a list of the class-labels (integers starting at 1)
    :param classes: the list of class-names
    :param color_function: a function mapping class-labels to box colors
    """
    for class_index, (x_min, y_min, x_max, y_max) in enumerate(boxes):
        label = labels[class_index].item()
        color = color_function(label - 1)
        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               linewidth=2, edgecolor=color, fill=False))

        ax.text(x_min, y_min, classes[label - 1], color='black', fontsize=12,
                bbox=dict(facecolor=color, edgecolor=color, pad=0),
                horizontalalignment='left', verticalalignment='bottom')
