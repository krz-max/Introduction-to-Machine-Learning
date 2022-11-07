from email.headerregistry import Group
from tkinter.messagebox import NO
import numpy as np
from heapq import nsmallest
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def RowtoColumnVector(ndarray):
    # convert from shape (x,) to (x,1)
    return ndarray[:, np.newaxis]

def FindAverage(data, total):
    return np.sum(data, axis=0)/total

def NormalizeVector(vector):
    RowtoColumnVector(ndarray=vector)
    return vector/(np.linalg.norm(vector))

def FisherLinearDiscriminant(x_train, x_test, y_train, y_test):
    x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)
    # print(x_train.shape) # (3750, 2)
    # print(y_train.shape) # (3750,)
    # print(x_test.shape) # (1250, 2)
    # print(y_test.shape) # (1250,)
    y_train = RowtoColumnVector(y_train)
    y_test = RowtoColumnVector(y_test)
    # print(y_train.shape) # (3750,1)
    # print(y_test.shape) # (1250,1)

    # Part1:
    # each data is 2 dimensional, and there are 3750 training data
    # Find the average of C1, the class of each data is specified in y_train
    # m1 and m2 should be a point on 2-d plane
    m1 = x_train.mean(axis=0, where=(y_train==1))
    m2 = x_train.mean(axis=0, where=(y_train==0))
    print(f"mean vector of class 1: {m1}", f"mean vector of class 2: {m2}")

    # Part2:
    # Within-class covariance matrix:
    # for all class:
    #     sum((xn-mk)*(xn-mk).T)
    Group1 = np.empty_like([[0, 0]])
    Group2 = np.empty_like([[0, 0]])
    for x in x_train:
        if y_train[np.where(x_train==x)[0]][0] == [1]:
            Group1 = np.append(Group1, [x], axis=0)
        else:
            Group2 = np.append(Group2, [x], axis=0)
    Group1 = Group1[1:]
    Group2 = Group2[1:]
    sw = np.cov(Group1.T) + np.cov(Group2.T)    
    print(f"Within-class scatter matrix SW:\n{sw}")
    
    # Part3:
    vect = RowtoColumnVector(ndarray=(m2-m1))
    sb = np.dot(vect, vect.T)
    print(f"Between-class scatter matrix SB:\n{sb}")
    
    # Part4:
    w = NormalizeVector(np.dot(np.linalg.inv(sw), vect))
    print(f" Fisher’s linear discriminant:\n{w}")
    
    return m1, m2, sw, sb, w, Group1, Group2

# Part4-1: Prediction
def Predict(x_train, x_test, y_train, y_test):
    x_train_proj = np.dot(x_train, w)
    x_test_proj = np.dot(x_test, w)
    y_pred = np.zeros((len(y_test),1))
    
    for k in range(1, 6):
        for test_proj in x_test_proj:
            Distance = abs(np.subtract(x_train_proj, test_proj))
            MinK = np.sort(Distance, axis=0)[:k]
            Pred = 0
            for x in range(k):
                MinK_at = np.where(Distance==MinK[x])[0]
                Pred += y_train[MinK_at]
            y_pred[np.where(x_test_proj==test_proj)[0]] = (Pred > k/2)
        # Part5:
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy of test-set {acc}")

def PlotResult(Bias, MarkerSize, Alpha, LineWidth, Group1, Group2, weight):
    plt.title(f'Projection Line: w={(weight[1]/weight[0])[0]}, b={Bias}')
    plt.plot([-6*weight[0], 5*weight[0]], [-6*weight[1], 5*weight[1]], 'y-')
    plt.scatter(Group1[:, 0], Group1[:, 1], c='b', s=MarkerSize)
    plt.scatter(Group2[:, 0], Group2[:, 1], c='r', s=MarkerSize)
    r = weight.reshape(2,)
    r2 = np.linalg.norm(r)**2
    for pt in Group1:
        prj = r * r.dot(pt) / r2
        plt.plot([prj[0], pt[0]], [prj[1], pt[1]], marker='.', linestyle='-', color='b', alpha=Alpha, linewidth=LineWidth)
    for pt in Group2:
        prj = r * r.dot(pt) / r2
        plt.plot([prj[0], pt[0]], [prj[1], pt[1]], marker='.', linestyle='-', color='r', alpha=Alpha, linewidth=LineWidth)
    plt.show()
    
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = np.load('classification_data.npy', allow_pickle=True)
    m1, m2, sw, sb, w, Group1, Group2 = FisherLinearDiscriminant(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    
    # Part4-1: Prediction
    Predict(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    
    # Plotting Result
    PlotResult(Bias=0, MarkerSize=4, Alpha=0.2, LineWidth=0.5, Group1=Group1, Group2=Group2, weight=w)
    
    pass