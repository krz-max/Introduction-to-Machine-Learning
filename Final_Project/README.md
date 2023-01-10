# [NYCU 2022 Fall] Introduction to Machine Learning - Final Project

This repository is the official implementation of [Tabular Playground Series - Aug 2022](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/overview). on [Kaggle](https://www.kaggle.com/)

## Requirements
Requirements are documented in the first section of code in the file `Final_Train.ipynb`

## Training
❗❗❗Please place the data, model, and the code in the same folder❗❗❗<br>
To train the model(s), download the notebook:`Train.ipynb` and simply press the `run all` command

## Evaluation
* You could download `Inference.ipynb` and press `run all` to skip the training process
  * You may just use the `Train.ipynb` because the training is actually fast(1~2 mins running all codes in `Train.ipynb`).
To evaluate my model on Kaggle:
1. After the `Training` process or `Inference` process, there should be a new file `submission.csv`
2. Upload the `submission.csv` to the Kaggle playground, click [here](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/overview) to access the website.
3. Click `Late Submission` to upload the `csv` file.
4. And we're done ! 😎😎

## Pre-trained Models

* You can download pretrained models in this repository [here](https://drive.google.com/drive/folders/1cnXAgGk6cMVc0HvN3ZOmyraj9tD7TSfW?usp=sharing)
* or simply use the `Model0.sav` in this GitHub repository

## Results

Our model achieves the following performance on :
### [Tabular Playground Series - Aug 2022](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/submissions)


| Model name         | Private Score  | Public Score |
| ------------------ |---------------- | -------------- |
| Logistic Regression   |     0.59224        |      0.58879       |

The ROC Curve is here:<br>
![](ROC_curve.jpg)


