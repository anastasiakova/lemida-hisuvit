import pandas as pd
import numpy as np
import sys

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
        updated_theta = theta - gradient(x_with_bias, vector_of_targets_y, theta)*lr

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

def splitTrainTest(matrix_x, trainSize):
    rows = len(matrix_x) * trainSize
    np.random.shuffle(matrix_x)
    train_data = matrix_x.iloc[:rows]
    test_data = matrix_x.iloc[rows:]
    X_train = train_data.drop(columns=['y'])
    y_train = train_data['y']
    X_test = test_data.drop(columns=['y'])
    y_test = test_data['y']
    return X_train, y_train, X_test, y_test

def accurancyCalc(prediction, trueValues):
    right = 0;
    dataLength = len(prediction)
    for i in range( dataLength - 1):
        if prediction[i] == trueValues[i]:
            right = right + 1
    return right / dataLength

def main():
    # filePath = sys.argv[1]
    filePath ="ex1data1.csv"
    data = pd.read_csv(filePath)
    X_train, y_train, X_test, y_test = splitTrainTest(data, 0.8)
    theta = gradient_descent(np.array(X_train), np.array(y_train))
    prediction = predict(np.array(X_test), theta)
    accur = accurancyCalc(prediction, y_test)
    print(accur)

main()
