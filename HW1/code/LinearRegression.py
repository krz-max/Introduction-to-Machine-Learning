import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def CalculateLoss(Error):
    return Error.dot(Error.T) / len(Error)
# throw **all x** into model and calculate total error
# then find the gradient and update it
def LinearRegression(x_data, y_data, learning_rate, x_test, y_test, iter):
    Train_Error = np.zeros(iter)
    Valid_Error = np.zeros(iter)
    weight = np.random.randn(2, 1)
    x_test = np.array(x_test)
    x_test = np.insert(x_test.T, 0, 1, axis=0)
    y_test = y_test[np.newaxis, :]
    x_data = np.array(x_data)
    x_data = np.insert(x_data.T, 0, 1, axis=0)
    y_data = y_data[np.newaxis, :]
    # print('dim of x: {}'.format(x_data.shape))
    # print('dim of w: {}'.format(weight.shape))
    # print('dim of y: {}'.format(y_data.shape))
    # find the predicted value of every x
    # x.shape = (2, 750), weight.shape = (2,1), y.shape = (1, 750)
    # w^T * X = [b0, b1] * [[ 1,  1, ...],
    #                       [x1, x2, ..]]
    # plt.style.use('fivethirtyeight')
    n_iteration = np.resize(np.arange(iter), (iter, 1))
    for iteration in range(len(n_iteration)):
        # gradient of MSE = (2) * sum{ [ (tn - y(xn))*1, (tn - y(xn))*x1,n ] }
        
        # validation data's error
        x_predict = weight.T.dot(x_test)
        Error = (y_test - x_predict)
        Valid_Error[iteration] = Mean_Square_Error(test_data=y_test, pred=x_predict)
        
        # training data's error
        x_predict = weight.T.dot(x_data)
        Error = (y_data - x_predict)
        Train_Error[iteration] = Mean_Square_Error(test_data=y_data, pred=x_predict)
        gradient = np.sum(Error.dot(x_data.T), axis=0)
        # print('dim of gradient: {}'.format(gradient.shape))
        # print('dim of MSE: {}'.format(Error.shape))
        gradient = gradient[:,np.newaxis]
        weight = weight + learning_rate*gradient
    
    # print('dim of iterations: {}'.format(n_iteration.shape))
    # print('dim of Valid_Error: {}'.format(Valid_Error.shape))
    # print('dim of Train_Error: {}'.format(Train_Error.shape))
    plt.title("Learning Curve")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.plot(n_iteration, Valid_Error, color='red', label='Validation Error')
    plt.plot(n_iteration, Train_Error, color='blue', label='Train Error')
    plt.legend(['Validation Error', 'Train Error'])
    plt.show()
    return weight

def myModel(weight, test_in):
    test_in = np.array(test_in)
    test_in = np.insert(test_in.T, 0, 1, axis=0)
    y_pred = weight.T.dot(test_in)
    return y_pred

def Mean_Square_Error(test_data, pred):
    sum = 0
    for idx in range(len(test_data)):
        sum += (test_data[idx]-pred[idx]) ** 2
    sum /= len(test_data)
    return sum[0]

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = np.load('../data/regression_data.npy', allow_pickle=True)
    # print(x_train)
    # print(y_train.shape)
    weight = LinearRegression(x_data=x_train, y_data=y_train, learning_rate=0.0001, x_test=x_test, y_test=y_test, iter=100)
    y_pred = myModel(weight=weight, test_in=x_test)
    y_pred = y_pred.T
    plt.plot(x_train, y_train, '.', label='training set')
    plt.plot(x_test, y_test, '.', label='test data')
    plt.plot(x_test, y_pred, '.', label='prediction')
    plt.legend(['training set', 'test data', 'prediction'])
    plt.title("Regression data") # title
    plt.ylabel("y") # y label
    plt.xlabel("x") # x label
    print('MSE is {}'.format(Mean_Square_Error(test_data=y_test, pred=y_pred)))
    print('Intercept is {} and weights are {}'.format(weight[0], weight[1:]))
    plt.show()
    pass
