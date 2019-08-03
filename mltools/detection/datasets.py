"""
Image object-detection data-sets all subclassing Pytorch's Dataset or VisionDataset.
"""

from xml.etree.ElementTree import parse

from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm, subplots, xticks
from numpy import random, size
from pandas import DataFrame
from torch import Tensor, as_tensor, float32, int64, no_grad, tensor, uint8, zeros
from torchvision import ops
from torchvision.datasets import VisionDataset

from mltools.detection.utils import create_boxes_per_class_data_frame, parse_voc_annotations


class VOCXMLDataset(VisionDataset):
    """A Dataset to hold images and bounding box annotations from .xml files in the Pascal VOC format."""

    def __init__(self,
                 root,
                 image_folder_name="images",
                 annotation_folder_name="annotations",
                 transforms=None):
        """
        Constructs a new VOCXMLDataset. All .xml annotation files in root/annotation_folder_names will be parsed
        to obtain the file-names of the annotated images and additional information on the dataset. It is expected
        that (at least) all annotated images exist in root/image_folder_name and that all annotation files conform
        to the Pascal VOC format.

        :param root: (pathlib.Path) The root folder containing the image and annotation folders.
        :param image_folder_name: (string, default: "images") The name of the folder containing the images.
        :param annotation_folder_name: (string, default: "annotations") The name of the folder containing
                                       the annotations.
        :param transforms: (optional, a transform object from mltools.detection.transforms) A transform to apply to the
                           images and (if necessary) to the bounding boxes.
        """
        super(VOCXMLDataset, self).__init__(root, transforms)

        self.annotation_dir = self.root / annotation_folder_name
        self.image_dir = self.root / image_folder_name

        self.annotations = sorted([annotation_file for annotation_file in self.annotation_dir.iterdir()
                                   if annotation_file.suffix == ".xml"])

        self.info_df = parse_voc_annotations(self.annotations)

        self.images = self.info_df.filename.map(lambda image_file: self.image_dir / image_file).to_list()
        self.classes = sorted(list(set.union(*self.info_df.classes)))

        self.boxes_per_class_df = create_boxes_per_class_data_frame(self.info_df)

        self.class_name_to_label = {class_name: index + 1 for index, class_name in enumerate(self.classes)}
        self.color_function = cm.get_cmap("hsv", len(self.classes))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")

        annotation_image_width, annotation_image_height, classes, boxes = self._parse_annotation(index)

        boxes = as_tensor(boxes, dtype=float32)
        labels = as_tensor([self.class_name_to_label[class_name] for class_name in classes], dtype=uint8)
        image_id = tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # Needed for the torch FasterRCNN, but not used (always set to zero)
        iscrowd = zeros((len(labels),), dtype=int64)

        if annotation_image_width != image.width:
            boxes[:, [0, 2]] *= image.width / annotation_image_width

        if annotation_image_height != image.height:
            boxes[:, [1, 3]] *= image.height / annotation_image_height

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def label_to_class_name(self, label_id):
        """
        Gets the class name of a label id.
        :param label_id: (int) the label id
        :return: (string) the class-name corresponding to the label id
        """
        return self.classes[label_id]

    def extra_repr(self):
        """
        Extra information on the dataset.
        :return: (string) info
        """
        lines = [
            "Number of classes: {}".format(len(self.classes)),
            "Classes: {}".format(self.classes)
        ]

        return '\n'.join(lines)

    def display_sample(self, index, figsize=(10, 8)):
        """
        Displays a sample-image and its annotations.
        :param index: (int) the index of the image in the dataset
        :param figsize: (optional, tuple(int, int)) the desired figure size
        """
        fig, ax = subplots(figsize=figsize)
        self._create_image_with_annotation(index, ax)

    def display_sample_grid(self, rows=3, cols=3, indices=None, figsize=(12, 8)):
        """
        Displays a grid of random samples from the dataset.
        :param rows: (int, default=3) the number of rows
        :param cols: (int, default=3) the number of columns
        :param indices: (optional, list) if provided, random sample-indices will be drawn from this list, otherwise
                        they will be drawn from the indices of the whole dataset
        :param figsize: (tuple(int, int), default=(12, 8)) the desired figure size
        """
        fig, ax = subplots(rows, cols, figsize=figsize, constrained_layout=True)

        indices_choice = random.choice(indices if indices is not None else self.__len__(),
                                       (rows, cols), replace=False)

        if rows == 1 and cols == 1:
            self._create_image_with_annotation(indices_choice[0, 0], ax)
        elif rows == 1:
            for j in range(cols):
                self._create_image_with_annotation(indices_choice[0, j], ax[j])
        elif cols == 1:
            for i in range(rows):
                self._create_image_with_annotation(indices_choice[i, 0], ax[i])
        else:
            for i in range(rows):
                for j in range(cols):
                    self._create_image_with_annotation(indices_choice[i, j], ax[i, j])

    def display_annotation_and_prediction(self, index, model, device="cuda", figsize=(10, 8)):
        """
        Displays the ground truth annotation as well as the prediction returned by a provided model
        evaluated on an image of the dataset.
        :param index: (int) the index of the image
        :param model: (torch.nn.Module) the model to create the prediction
        :param device: ("cuda" or "cpu", default="cuda") the device to perform the prediction on
        :param figsize: (tuple(int, int), default=(10, 8)) the desired figure size
        """
        fig, ax = subplots(figsize=figsize)
        self._create_image_with_annotation_and_prediction(index, model, device, ax)

    def display_annotation_and_prediction_grid(self, model, rows=3, cols=3, indices=None,
                                               device="cuda", figsize=(12, 10)):
        """
        Displays random images including ground truth and predicted annotations in a grid.
        :param model: (torch.nn.Module) the model to create the predictions
        :param rows: (int, default=3) the number of rows
        :param cols: (int, default=3) the number of columns
        :param indices: (optional, list) if provided, random sample-indices will be drawn from this list, otherwise
                        they will be drawn from the indices of the whole dataset
        :param device: ("cuda" or "cpu", default="cuda") the device to perform the predictions on
        :param figsize: (tuple(int, int), default=(12, 10)) the desired figure size
        """
        fig, ax = subplots(rows, cols, figsize=figsize, constrained_layout=True)

        indices_choice = random.choice(indices if indices is not None else self.__len__(),
                                       (rows, cols), replace=False)

        if rows == 1 and cols == 1:
            self._create_image_with_annotation_and_prediction(indices_choice[0, 0], model, device, ax)
        elif rows == 1:
            for j in range(cols):
                self._create_image_with_annotation_and_prediction(indices_choice[0, j], model, device, ax[j])
        elif cols == 1:
            for i in range(rows):
                self._create_image_with_annotation_and_prediction(indices_choice[i, 0], model, device, ax[i])
        else:
            for i in range(rows):
                for j in range(cols):
                    self._create_image_with_annotation_and_prediction(indices_choice[i, j], model, device, ax[i, j])

    def plot_boxes_per_class(self, kind="bar", figsize=None):
        """
        Visualizes the boxes per class counts.
        :param kind: (either "bar" or "pie", default="bar") whether to show the counts as a bar plot or a pie plot
        :param figsize: (optional, tuple(int, int)) the desired figure size
        """
        _, ax = subplots(figsize=figsize)
        ax.set_title("Boxes per Class")

        if kind == "bar":
            ax.bar(self.boxes_per_class_df.index, self.boxes_per_class_df.nr_boxes,
                   color=[self.color_function(i) for i in range(len(self.classes))])
            xticks(rotation='vertical')
        elif kind == "pie":
            boxes_sum = self.boxes_per_class_df.nr_boxes.sum()

            ax.pie(self.boxes_per_class_df.nr_boxes, labels=self.boxes_per_class_df.index,
                   colors=[self.color_function(i) for i in range(len(self.classes))],
                   autopct='%1.1f%%', pctdistance=0.8, labeldistance=1.05)
        else:
            raise ValueError(f"\'kind\' must be either \'bar\' or \'pie\', but got: \'{kind}\'")

    def get_prediction_metrics_df(self, model, indices=None, device="cuda"):
        if indices is None:
            indices = range(self.__len__())

        metrics_list = []
        model.eval()

        for index in indices:
            image, target = self.__getitem__(index)
            with no_grad():
                prediction = model([image.to(device)])[0]

            metrics_list.append({
                "image_id": index,
                "filename": self.images[index].name,
                "classes": [self.classes[label - 1] for label in prediction["labels"]],
                "scores": prediction["scores"].cpu().numpy(),
                "max_ious": self._calculate_max_ious(target["boxes"], prediction["boxes"]).cpu().numpy()
            })

        return DataFrame(metrics_list, columns=["image_id", "filename", "classes", "scores", "max_ious"]).set_index("image_id")

    def _extract_images_and_classes_from_annotations(self):
        images = []
        classes_set = set()

        for annotation_file in self.annotations:
            root = parse(annotation_file).getroot()
            classes_set.update([name_element.text for name_element in root.iter("name")])
            images.append(self.image_dir / root.find("filename").text)

        return images, list(classes_set)

    def _parse_annotation(self, index):
        root = parse(self.annotations[index]).getroot()

        image_size_node = root.find("size")

        image_width = int(image_size_node.find("width").text)
        image_height = int(image_size_node.find("height").text)

        classes = []
        boxes = []

        for object_element in root.findall("object"):
            class_name = object_element.find("name").text

            if class_name in self.classes:
                classes.append(object_element.find("name").text)
                boxes.append(self._parse_bounding_box(object_element))

            for part in object_element.iter("part"):
                if class_name in self.classes:
                    classes.append(part.find("name").text)
                    boxes.append(self._parse_bounding_box(part))

        return image_width, image_height, classes, boxes

    @staticmethod
    def _parse_bounding_box(node):
        box_node = node.find("bndbox")
        x_min = float(box_node.find("xmin").text)
        x_max = float(box_node.find("xmax").text)
        y_min = float(box_node.find("ymin").text)
        y_max = float(box_node.find("ymax").text)

        return [x_min, y_min, x_max, y_max]

    def _create_image_with_annotation(self, index, ax):
        image, target = self.__getitem__(index)
        ax.set_title(f"{index}: {self.images[index].name}")
        ax.axis("off")

        if isinstance(image, Tensor):
            ax.imshow(image.permute(1, 2, 0))
        else:
            ax.imshow(image)
        self._create_annotation_box_patches(target, ax)

    def _create_image_with_annotation_and_prediction(self, index, model, device, ax):
        image, target = self.__getitem__(index)
        ax.set_title(self.images[index].name)
        ax.axis("off")

        if isinstance(image, Tensor):
            ax.imshow(image.permute(1, 2, 0))
        else:
            ax.imshow(image)

        # Display ground truth boxes:
        self._create_annotation_box_patches(target, ax)

        # Get prediction:
        model.eval()
        with no_grad():
            prediction = model([image.to(device)])[0]

        if not size(prediction["boxes"].values) == 0:
            prediction["max_ious"] = self._calculate_max_ious(target["boxes"], prediction["boxes"])

            self._create_prediction_box_patches(prediction, ax)

    @staticmethod
    def _calculate_max_ious(ground_truth_boxes, predicted_boxes):
        ious = ops.box_iou(ground_truth_boxes.to("cpu"), predicted_boxes.to("cpu"))
        return ious.max(dim=1).values

    def _create_annotation_box_patches(self, target, ax):
        for class_index, (x_min, y_min, x_max, y_max) in enumerate(target["boxes"]):
            label = target["labels"][class_index].item()
            color = self.color_function(label - 1)
            ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   linewidth=2, edgecolor=color, fill=False, linestyle='-'))

            ax.text(x_min, y_min, self.classes[label - 1], color='black', fontsize=12,
                    bbox=dict(facecolor=color, edgecolor=color, pad=0),
                    horizontalalignment='left', verticalalignment='bottom')

    def _create_prediction_box_patches(self, prediction, ax):
        for class_index, (x_min, y_min, x_max, y_max) in enumerate(prediction["boxes"]):
            label = prediction["labels"][class_index].item()
            color = self.color_function(label - 1)
            ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   linewidth=2, edgecolor=color, fill=False,
                                   linestyle='--'))

            label_text = self.classes[label - 1] + (" | score: {0:.4f} | max_iou: {1:.4f}"
                                                    .format(prediction["scores"][class_index].item(),
                                                            prediction["max_ious"][class_index].item()))

            ax.text(x_max, y_max, label_text, color='black', fontsize=12,
                    bbox=dict(facecolor=color, edgecolor=color, pad=0),
                    horizontalalignment='right', verticalalignment='top')
