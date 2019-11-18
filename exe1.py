import numpy as np
import pandas as pd
import sys

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
def sigmoid_function(matrix_x, theta):
    prediction = np.dot(matrix_x, theta)
    return 1 / (1 + np.exp(-prediction))


def gradient(matrix_x, vector_of_targets_y, theta):
    prediction = sigmoid_function(matrix_x,theta)
    target_len = vector_of_targets_y.shape[0]
    return np.dot(matrix_x.T, (prediction - vector_of_targets_y)) * (1/target_len)


def gradient_descent(matrix_x, vector_of_targets_y, lr=0.1, max_iterations=1000, threshold=1e-5):
    x_with_bias = np.inset(matrix_x, 0, 1, axis=1)
    theta = np.zeros(vector_of_targets_y.shape[0])

    for i in range(max_iterations):
        updated_theta = gradient(x_with_bias, vector_of_targets_y, theta)*lr
        diff = np.abs(updated_theta - theta)
        root = np.sqrt(diff)
        theta = updated_theta
        if root <= threshold:
            break
    return theta


def to_binary(predictions):
    return np.where(predictions > 0, 1, 0)


def predict(matrix_x, theta):
    x_with_bias = np.inset(matrix_x, 0, 1, axis=1)
    sigmoid_prediction = sigmoid_function(x_with_bias, theta)
    prediction = to_binary(sigmoid_prediction)
    return prediction


def main():
    # filePath = sys.argv[1]
    # print("ex1data1.csv")
    data = pd.read_csv('ex1data1.csv')
    matrix_x = data.drop(columns=['y'])
    y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(matrix_x, y, test_size=0.2)
    theta = gradient_descent(X_train, y_train)
    prediction = predict(X_test, theta)
    accur = accuracy_score(y_test, prediction)
    print(accur)



main()