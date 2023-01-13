# [NYCU 2022 Fall] Introduction to Machine Learning

* You could download the notebooks & data, put them all in the same folder and `run all` on your local environment or GoogleColab
* You need to unzip the data first
* You might need to download some package if you don't have some of them.

## HW1: Logistic regression and Linear Regression using Gradient Descent

### Goals
In this coding assignment, you need to implement linear regression by using only NumPy, then train your implemented model using Gradient Descent by the provided dataset and test the performance with testing data.

Find the questions at [here](https://docs.google.com/document/d/1kBR0tqYltq1YEoFiwem0Yzn_MYq5wNyzJG9LeURHDYE/edit?usp=sharing)

### Output

* LinearRegression
MSE is 54.2685
Intercept is [-0.4061] and weights are [50.7277]
* LogisticRegression
Cross Entropy Error is 0.1854
Intercept is [4.5803] and weights are [1.5370]


| Method            | Learning Curve | Result |
| ----------------- | -------- | -------- |
| Logistic Regression | ![](Results/logistic_training_loss.jpg) | ![](Results/Logistic_Result.jpg)         |
| Linear Regression | ![](Results/linear_training_loss.jpg)     | ![](Results/Linear_Result.jpg)     |


## HW2: Linear Discriminant Analysis

### Goals
In hw2, you need to implement Fisher’s linear discriminant by using only numpy, then train your implemented model by the provided dataset and test the performance with testing data

Please note that only NUMPY can be used to implement your model, you will get no points by simply calling sklearn.discriminant_analysis.LinearDiscriminantAnalysis

Find the questions at https://docs.google.com/document/d/1T7JLWuDtzOgEQ_OPSgsSiQdp5pd-nS5bKTdU3RR48Z4/edit?usp=sharing

### Projection Result
![](Results/FLD_ProjectionResult.jpg)

* From HW3 to HW5, see the result in the `report.pdf`

## HW3: Decision Tree, AdaBoost and Random Forest

### Goals
In this coding assignment, you need to implement the Decision Tree, AdaBoost and Random Forest algorithm by using only NumPy, then train your implemented model by the provided dataset and test the performance with testing data. Find the sample code and data on the GitHub page.

Please note that only NumPy can be used to implement your model, you will get no points by simply calling sklearn.tree.DecsionTreeClassifier.

Find the questions at this [document](https://docs.google.com/document/d/1ODV5FtIIn6fXjExL6cF8UOsQ-ctu53jObOAjrcSmqfw/edit?usp=sharing)


## HW4: Decision Tree, AdaBoost and Random Forest

### Goals
In this coding assignment, you need to implement the Cross-validation and grid search by using only NumPy, then train the SVM model from scikit-learn by the provided dataset and test the performance with testing data.

Please note that only NumPy can be used to implement cross-validation and grid search, you will get no points by simply calling sklearn.model_selection.GridSearchCV

Find the questions at this [document](https://docs.google.com/document/d/1YvMXHrcyxQrBHbGEZgPZbMVtXesSQuIm/edit?usp=sharing&ouid=106791491758005483971&rtpof=true&sd=true)


## HW5: Simple Captcha Hacker using Neural Network

### Goals
Implement the deep neural network by any deep learning frameworks, e.g., Pytorch, TensorFlow and Keras, and then train DNN model on the provided dataset
Find the Kaggle page [here](https://www.kaggle.com/competitions/captcha-hacker/overview)

### Example
* Train a model to predict all the digits in the image
    * Task 1: Single character in the image
    * Task 2: Two characters in the image (order matters)
    * Task 3: Four characters in the image (order matters)
![](Results/Example.jpg)
* Download the Data [Here](https://www.kaggle.com/competitions/captcha-hacker/data)

## Final Project: Product Failure Prediction

* Download the Data [Here](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/data)
* For more information, read `README.md` in the folder `Final_Project`

