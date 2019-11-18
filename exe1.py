import pandas as pd
import numpy as np
import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def sigmoid_function(matrix_x, theta):
    prediction = np.dot(matrix_x, theta)
    return 1 / (1 + np.exp(-prediction))


def gradient(matrix_x, vector_of_targets_y, theta):
    prediction = sigmoid_function(matrix_x, theta)
    target_len = vector_of_targets_y.shape[0]
    return np.dot(matrix_x.T, (prediction - vector_of_targets_y)) * (1/target_len)


def root_sum_of_squares(old, new):
    diff_power = np.power(old - new, 2)
    sum_of_diff = diff_power.sum()
    return np.sqrt(sum_of_diff)


def gradient_descent(matrix_x, vector_of_targets_y, lr=0.1, max_iterations=1000, threshold=1e-5):
    x_with_bias = np.insert(matrix_x, 0, 1, axis=1)
    theta = np.zeros((x_with_bias.shape[1], 1))

    for i in range(max_iterations):
        updated_theta = gradient(x_with_bias, vector_of_targets_y, theta)*lr

        if root_sum_of_squares(theta, updated_theta) <= threshold:
             break
    return theta


def to_binary(predictions):
    return np.where(predictions > 0, 1, 0)


def predict(matrix_x, theta):
    x_with_bias = np.insert(matrix_x, 0, 1, axis=1)
    sigmoid_prediction = sigmoid_function(x_with_bias, theta)
    prediction = to_binary(sigmoid_prediction)
    return prediction


def main():
    filePath = sys.argv[1]
    data = pd.read_csv(filePath)
    matrix_x = data.drop(columns=['y'])
    y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(matrix_x, y, test_size=0.2)
    theta = gradient_descent(np.array(X_train), np.array(y_train))
    prediction = predict(np.array(X_test), theta)
    accur = accuracy_score(y_test, prediction)
    print(accur)



main()
