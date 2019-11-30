import pandas as pd
import numpy as np
import sys

def sigmoid_function(matrix_x, theta, hypothesisVector):
    for i in range(0, matrix_x.shape[0]):
        value = np.matmul(theta, matrix_x[i])
        hypothesisVector[i] = 1 / (1 + np.exp(-float(value)))

    return hypothesisVector

def hypothesis(theta, matrix_x, numOfFeatures):
    numOfRows = matrix_x.shape[0]
    hypothesisVector = np.ones((numOfRows,1))
    theta = theta.reshape(1, numOfFeatures)
    hypothesisVector = sigmoid_function(matrix_x, theta, hypothesisVector)
    hypothesisVector = hypothesisVector.reshape(numOfRows)
    return hypothesisVector

def gradient(matrix_x, vector_of_targets_y, theta):
    numberOfFeatures = matrix_x.shape[1]
    hypothesisVector = hypothesis(theta, matrix_x, numberOfFeatures)
    for i in range(0, numberOfFeatures):
        gradientVector = sum((hypothesisVector - vector_of_targets_y) * matrix_x.transpose()[i])
        theta[i] = theta[i] - (gradientVector)
    return theta


def root_sum_of_squares(old, new):
    diff_power = np.power(old - new, 2)
    sum_of_diff = diff_power.sum()
    return np.sqrt(sum_of_diff)


def gradient_descent(matrix_x, vector_of_targets_y, lr=0.1, max_iterations=1000, threshold=1e-5):
    numOfFeatures = matrix_x.shape[1] + 1#for bias
    x_with_bias = np.insert(matrix_x, 0, 1, axis=1)
    theta = np.zeros(numOfFeatures)

    for i in range(0, max_iterations):
        oldTheta = theta
        theta = (lr / x_with_bias.shape[0]) * gradient(x_with_bias, vector_of_targets_y, theta)
        if root_sum_of_squares(theta, oldTheta) <= threshold:
             return oldTheta.reshape(1,numOfFeatures)
    theta = theta.reshape(1, numOfFeatures)
    return theta


def splitTrainTest(matrix_x, trainSize):
    rows = int(len(matrix_x) * trainSize)
    matrix_x = matrix_x.sample(frac=1)
    train_data = matrix_x.iloc[:rows]
    test_data = matrix_x.iloc[rows:]
    X_train = train_data.drop(columns=['y'])
    y_train = train_data['y']
    X_test = test_data.drop(columns=['y'])
    y_test = test_data['y']
    return X_train, y_train, X_test, y_test


def accurancyCalc(prediction, trueValues):
    right = 0
    dataLength = len(prediction)
    for i in range(dataLength - 1):
        if prediction[i] == trueValues[i]:
            right = right + 1
    return right / dataLength


def to_binary(predictions, numOfRows):
    # if very small data
    if numOfRows < 30:
        median = predictions.mean()
        return np.where(predictions > median, 1, 0)
    # small data
    if numOfRows < 80:
        return np.where(predictions > 0.4, 1, 0)
    else:
        return np.where(predictions > 0.5, 1, 0)


def predict(matrix_x, theta):
    x_with_bias = np.insert(matrix_x, 0, 1, axis=1)
    prediction = hypothesis(theta, x_with_bias, x_with_bias.shape[1])
    prediction = to_binary(prediction, x_with_bias.shape[0])
    return prediction


def main():
    filePath = sys.argv[1]
    data = pd.read_csv(filePath)
    X_train, y_train, X_test, y_test = splitTrainTest(data, 0.8)
    theta = gradient_descent(np.array(X_train), np.array(y_train))
    prediction = predict(np.array(X_test), theta)
    accur = accurancyCalc(prediction, np.array(y_test))
    print(accur)

main()