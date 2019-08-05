# Machine Learning
This repo contains a compilation of machine learning projects in the form of Jupyter notebooks. For some notebooks additional data, such as bounding box annotation files are needed, these files can be found in the *data* folder.

## `mltools` Library
This is a small Python library which contains usefull classes and functions for machine learning and data science tasks, such as for example functions helping with feature exploration and Dataset-classes for object detection and classification using Pytorch. 

## Notebooks

### Object Detection

#### VOCXMLDataset Demo ([notebook](https://github.com/mfl28/MachineLearning/blob/master/notebooks/VOCXMLDataset_Demo.ipynb), [nbviewer](https://nbviewer.jupyter.org/github/mfl28/MachineLearning/blob/master/notebooks/VOCXMLDataset_Demo.ipynb))
A notebook showcasing the use of the `VOCXMLDataset` class from `mltools.detection.datasets` using images and annotations from the [VOC2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) for demonstrations. 

#### Humpback Whale Fluke Detection ([notebook](https://github.com/mfl28/MachineLearning/blob/master/notebooks/Humpback_Whale_Fluke_Detection.ipynb), [nbviewer](https://nbviewer.jupyter.org/github/mfl28/MachineLearning/blob/master/notebooks/Humpback_Whale_Fluke_Detection.ipynb), [Google Colab](https://colab.research.google.com/github/mfl28/MachineLearning/blob/master/notebooks/Humpback_Whale_Fluke_Detection.ipynb))
A notebook showing how to perform object detection with a custom dataset using a pre-trained and subsequently fine-tuned neural network. Specifically, the aim is to detect and locate humpback whale flukes in images from the [Kaggle Humpback Whale Identification Challenge](https://www.kaggle.com/c/humpback-whale-identification). The ground truth bounding box labels for a selection of 800 images from the training dataset provided by the challenge were created using [Bounding Box Editor](https://github.com/mfl28/BoundingBoxEditor).
