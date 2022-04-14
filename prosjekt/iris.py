import numpy as np
from numpy import random
import matplotlib.pyplot as plt


def getMeasurementMatrix(file_array):
    """
    Gets measurement vectors from data_file in matrix form

    :param file_array: array with file names
    :return: measurement matrix
    """
    res_matrix = []
    for file in file_array:
        with open(file) as f:
            # reading each line
            for line in f:
                line_split = line.split(',')
                line_split = [1] + [float(element) for element in line_split]  #
                res_matrix.append(line_split)
    return np.array(res_matrix)


def z(W, x):
    """
    Multiplies weight matrix W with vector x

    :param W: weight matrix
    :param x: feature vector
    :return:
    """
    return W @ x


def sigmoid(z):
    """
    Keeps output between 0 and 1

    :param z:
    :return:
    """
    return 1 / (1 + np.exp(-z))


def g(W, x):
    return sigmoid(z(W, x))


def create_W(classes, features):
    W = np.zeros(shape=(classes, features + 1))
    return W


def MSE_grad_g(W, x, t):
    return g(W, x) - t  # dim(classes,1) - dim(classes,1)


def g_grad_z(W, x):
    return g(W, x) * (1 - g(W, x))  # dim(classes,1) - dim(classes,1)


def z_grad_W(x):
    return np.array([[x_n] for x_n in x])  # dim(1,features+1)


def MSE_grad_W(W, x, t):
    return (MSE_grad_g(W, x, t) * g_grad_z(W, x)) * z_grad_W(x)


def sum_MSE_grad_W(W, x_matrix, t_matrix):
    res_sum = np.transpose(np.zeros(shape=W.shape))
    for x, t in zip(x_matrix, t_matrix):
        res_sum += MSE_grad_W(W, x, t)
    return res_sum


def MSE(W, x_matrix, t_matrix):
    res = 0
    for x, t in zip(x_matrix, t_matrix):
        res += np.transpose(g(W, x) - t) @ (g(W, x) - t)
    return res


def error_rate(W, x_matrix, t_matrix):
    incorrect_guesses = 0
    for x, t in zip(x_matrix, t_matrix):
        predicted_class = np.argmax(g(W, x)) + 1
        actual_class = np.argmax(t) + 1
        # print("g_k:", g(W, x), "t_k", t, "success:", predicted_class == actual_class)
        if (predicted_class != actual_class):
            incorrect_guesses += 1
    return incorrect_guesses


def train_linear_classifier_iteration(W, x_matrix, t_matrix, alpha):
    classes = len(t_matrix[0])
    features = len(x_matrix[0]) - 1

    N = 10
    for k in range(0, len(x_matrix[:90]), N):
        W -= alpha * np.transpose(sum_MSE_grad_W(W, x_matrix[k:k + N], t_matrix[k:k + N]))

    MSE_k = MSE(W, x_matrix[90:], t_matrix[90:])
    error_rate_k = error_rate(W, x_matrix[90:], t_matrix[90:])
    return W, MSE_k, error_rate_k


def shuffle(x_matrix, t_matrix):
    shuffle_matrix = list(zip(x_matrix, t_matrix))
    random.shuffle(shuffle_matrix)
    return zip(*shuffle_matrix)


def main():
    x_matrix = getMeasurementMatrix([f'class_{i}' for i in range(1, 3 + 1)])

    t_1 = [1, 0, 0]
    t_2 = [0, 1, 0]
    t_3 = [0, 0, 1]
    t_class_1 = np.tile(t_1, (50, 1))
    t_class_2 = np.tile(t_2, (50, 1))
    t_class_3 = np.tile(t_3, (50, 1))
    t_matrix = np.concatenate((t_class_1, t_class_2, t_class_3))

    x_matrix, t_matrix = shuffle(x_matrix, t_matrix)

    classes = 3
    features = 4

    W = create_W(classes=classes, features=features)
    num_of_iterations = 1000
    MSEs = []
    error_rates = []
    for i in range(num_of_iterations):
        W, MSE_k, error_rate_k = train_linear_classifier_iteration(W, x_matrix[:150],
                                                                   t_matrix[:150], alpha=0.01)
        MSEs.append(MSE_k)
        error_rates.append(error_rate_k)

    plt.subplot(2, 1, 1)
    plt.title("MSE")
    plt.plot(MSEs)
    plt.subplot(2, 1, 2)
    plt.title("Error rate")
    plt.plot(error_rates)
    plt.show()

if __name__ == '__main__':
    main()