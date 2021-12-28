# Age-Gender-Ethnicity-Classifier
We provide project report file to understand our methodology.

## Introduction
In this project we aim to explore a dataset of images of people from different backgrounds and build a neural network image classifier to classify the age, gender and ethnicity of people in images. We deal with three different tasks in the project. These are gender classification which is a binary classification problem while age prediction is a regression problem since the output of the network can be any discrete number. Our last problem is that of ethnicity classification which is a multi class classification problem. All these tasks are important and useful in several fields and hence developing an efficient classification model can be helpful in solving the problems. For example, classifying ethnicity is an important task to solve for federal authorities and many companies developing photo editing applications.


## Dataset
We chose the dataset from a kaggle competition on age, gender and ethnicity classification. The dataset is stored as a csv file and all the information including images are present in the csv. The CSV file contains facial images in grayscale that are labeled on the basis of age, gender, and ethnicity. The dataset includes 27305 rows/data-points and 5 columns/features, namely, age, gender, ethnicity, img_name, pixels. We don’t need the img_name for our project since it doesn’t provide any useful information so we first drop this column.
The total size of our dataset is 191MB. Since the dataset is quite small in size, it was possible for us to train it on our local machines and GPU. Each of the image are of dimension 48 * 48. These images are reshaped as 48 * 48 * 1. The extra dimension denotes the color channel which in our case is 1 since the images are grayscale images and is needed by neural network libraries

## Methodology
We use tensorflow keras library to build our neural network architecture. The model architectures and parameters would be discussed in the later section. We build 3 different models, one corresponding to our task of age prediction, gender classification, ethnicity classification. The implementation of the model and training is done in the train.ipynb notebook. We store the best model based on the validation accuracy and the early stopping is used to overcome overfitting.
For the notebook and training to work, a folder named data should be created and the age_gender.csv data file should be copied to this folder.
We also do some exploratory data analysis to understand the data distribution. The relevant code is present in the EDA notebook.
Finally, we perform grid search to tune hyperparameters of the network. The result of training and hyperparameter tuning is present in the later section 4.
For implementation, we use tensorflow keras library for building the models and we use plotly and matplotlib for the visualization.
