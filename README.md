# Machine Learning
This repo contains a compilation of machine learning projects in the form of Jupyter notebooks. For some notebooks additional data, such as bounding box annotation files are needed, these files can be found in the *data* folder. [Pytorch](https://pytorch.org/) is used as the underlying library for projects involving deep learning.

## `mltools` Library [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/mfl28/MachineLearning.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/mfl28/MachineLearning/context:python)
This is a small Python library which contains useful classes and functions for machine learning and data science tasks, such as for example functions helping with feature exploration and Dataset-classes for object detection and classification using Pytorch. 

## Notebooks

### Semantic Segmentation

#### Kaggle Competition: Dstl Satellite Imagery Feature Detection ([notebook](https://github.com/mfl28/MachineLearning/blob/master/notebooks/Kaggle_Dstl_Satellite_Imagery_Feature_Detection.ipynb), [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/mfl28/MachineLearning/blob/master/notebooks/Kaggle_Dstl_Satellite_Imagery_Feature_Detection.ipynb), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mfl28/MachineLearning/blob/master/notebooks/Kaggle_Dstl_Satellite_Imagery_Feature_Detection.ipynb))
<p align=left>
<img src="demo-media/satellite_demo1.png" height= "150" />
<img src="demo-media/satellite_demo2.png" height= "150" />
</p>

A notebook showing how to perform semantic segmentation using a fully convolutional neural network. Our aim is to locate buildings in satellite images from the [Kaggle Dstl Satellite Imagery Feature Detection](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection).


### Object Detection

#### Humpback Whale Fluke Detection ([notebook](https://github.com/mfl28/MachineLearning/blob/master/notebooks/Humpback_Whale_Fluke_Detection.ipynb), [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/mfl28/MachineLearning/blob/master/notebooks/Humpback_Whale_Fluke_Detection.ipynb), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mfl28/MachineLearning/blob/master/notebooks/Humpback_Whale_Fluke_Detection.ipynb))
<p align=left>
<img src="demo-media/whale_demo.png" height= "200" />
</p>

A notebook showing how to perform object detection with a custom dataset using a pre-trained and subsequently fine-tuned neural network. Specifically, the aim is to detect and locate humpback whale flukes in images from the [Kaggle Humpback Whale Identification Challenge](https://www.kaggle.com/c/humpback-whale-identification). The ground truth bounding box labels for a selection of 800 images from the training dataset provided by the challenge were created using [Bounding Box Editor](https://github.com/mfl28/BoundingBoxEditor).

#### VOCXMLDataset Demo ([notebook](https://github.com/mfl28/MachineLearning/blob/master/notebooks/VOCXMLDataset_Demo.ipynb), [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/mfl28/MachineLearning/blob/master/notebooks/VOCXMLDataset_Demo.ipynb))
<p align=left>
<img src="demo-media/voc_demo.png" height= "250" />
</p>

A notebook showcasing the use of the `VOCXMLDataset` class from `mltools.detection.datasets` using images and annotations from the [VOC2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) for demonstrations. 

### Classification

#### Kaggle Competition: Humpback Whale Identification ([notebook](https://github.com/mfl28/MachineLearning/blob/master/notebooks/Kaggle_Whale_Identification.ipynb), [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/mfl28/MachineLearning/blob/master/notebooks/Kaggle_Whale_Identification.ipynb), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mfl28/MachineLearning/blob/master/notebooks/Kaggle_Whale_Identification.ipynb))
In this notebook we'll train a classifier to identify humpback whales in images according to the [Kaggle Humpback Whale Identification Challenge](https://www.kaggle.com/c/humpback-whale-identification). We'll use the [fast.ai](https://github.com/fastai/fastai) deep learning library to perform this task. 

#### Kaggle Competition: MNIST Digit Recognizer ([notebook](https://github.com/mfl28/MachineLearning/blob/master/notebooks/Kaggle_Mnist_Digit_Recognizer.ipynb), [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/mfl28/MachineLearning/blob/master/notebooks/Kaggle_Mnist_Digit_Recognizer.ipynb))
<p align=left>
<img src="demo-media/mnist_demo.png" height= "200" />
</p>

A notebook showing how to train a convolutional neural network object classifier for the MNIST Dataset from the [Kaggle MNIST Digit Recognizer competition](https://www.kaggle.com/c/digit-recognizer). The aim is to predict hand-drawn digits in images as accurately as possible.

#### Kaggle Competition: Titanic - Machine Learning from Disaster ([notebook](https://github.com/mfl28/MachineLearning/blob/master/notebooks/Kaggle_Titanic_Machine_Learning_From_Disaster.ipynb), [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/mfl28/MachineLearning/blob/master/notebooks/Kaggle_Titanic_Machine_Learning_From_Disaster.ipynb))
<p align=left>
<img src="demo-media/titanic_demo.jpg" height= "200" />
</p>

The aim of this notebook is to build a model which can predict the survival of passengers of the Titanic. Problem and data come from the [Kaggle Titanic: Machine Learning from Disaster competition](https://www.kaggle.com/c/titanic). We start with an exploration and visualization of the provided features, then proceed to building a feature engineering [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) using [scikit-learn](https://scikit-learn.org/stable/index.html). Finally we'll experiment with several machine learning approaches to solve the prediction problem.

