import numpy as np 
import matplotlib.pyplot as plt

def AddInterceptWeight(data):
    data = np.array(data)
    data = np.insert(data.T, 0, 1, axis=0)
    return data

def CalculateCrossEntropy(ground_truth, pred):
    sum = np.sum(np.where(ground_truth == 1, np.log(pred), np.log(1-pred)))
    return -sum

def LogisticRegression(x_train, y_train, x_test, y_test, learning_rate, n_iteration, weight):
    Valid_Error = np.zeros(n_iteration)
    Train_Error = np.zeros(n_iteration)
    
    # x_train : (k, 1), weight : (2, 1)
    y_train = y_train[np.newaxis, :]
    # x_train : (2, k), weight : (2, 1), y_train : (1, k)
    n_iteration = np.arange(n_iteration)
    for iteration in range(len(n_iteration)):
        # Validation data's cross entropy
        prob = myModel(weight=weight, x_data=x_test)
        Valid_Error[iteration] = CalculateCrossEntropy(ground_truth=y_test, pred=prob)
        # training data's error
        prob = myModel(weight=weight, x_data=x_train)
        Train_Error[iteration] = CalculateCrossEntropy(ground_truth=y_train, pred=prob)
        Prediction_Error = (y_train - prob)

        # gradient = sum((yn-tn)*phi(xn))
        gradient = np.dot(Prediction_Error, AddInterceptWeight(x_train).T)
        gradient = gradient.T
        # print(gradient.shape)
        
        weight = weight - learning_rate*gradient
    
    plt.title("Learning Curve")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.plot(n_iteration, Valid_Error, color='blue')
    plt.plot(n_iteration, Train_Error, color='red')
    plt.legend(['Validation Error', 'Train Error'])
    plt.show()
    return weight
    

def InitializeWeight(dim):
    return np.random.randn(dim, 1)

# yn = sigmoid(an), an = w^T * phi(xn) for all n
def myModel(weight, x_data):
    x_data = AddInterceptWeight(data=x_data)
    an = np.dot(weight.T, x_data)
    return sigmoid(an)
# if x is 1-d array, do element-wise exponential    
def sigmoid(x):
    return 1 / (1 + np.exp(x))

def ClassToNegativeOne(ground_truth):
    return np.where(ground_truth > 0, 1, -1)

def main():
    # Logistic Using Logistic Regression
    x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)
    # Initialize Weight using Standard Gaussian
    weight = InitializeWeight(dim=2)
    # Start Learning
    weight = LogisticRegression(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, learning_rate=0.0001, n_iteration=100, weight=weight)
    y_pred = myModel(weight=weight, x_data=x_test)
    print('Cross Entropy Error is {}'.format(CalculateCrossEntropy(ground_truth=y_test, pred=y_pred)))
    y_pred = np.where(y_pred.T >= 0.5, 1, 0)
    print('Intercept is {} and weights are {}'.format(weight[0], weight[1:]))
    plt.plot(x_train, y_train, '.')
    plt.plot(x_test, y_test, '.')
    plt.plot(x_test, y_pred, '.')
    plt.legend(['train data', 'test data', 'prediction'])
    plt.show()
    pass
    
if __name__ == "__main__":
    main()