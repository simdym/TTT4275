import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def getFeatureAndLabelMatrix(file_array, class_array):
    """
    Gets feature vectors and label vectors from files in file_array in matrix form

    :param file_array: array with file names
    :param class_array: array with classes for each file
    :return: feature matrix(x_matrix) and label matrix(t_matrix)
    """
    classes = max(class_array) + 1

    feature_matrix = []
    label_matrix = np.empty((0, classes), float)

    for file, file_class in zip(file_array, class_array):
        with open(file) as f:
            length_of_file = 0
            #Feature matrix
            for line in f:
                splitted_line = line.split(',')
                feature_matrix.append([1] + [float(element) for element in splitted_line])
                length_of_file += 1
        t_vec = np.zeros(classes)
        t_vec[file_class] = 1
        file_label_matrix = np.tile(t_vec, (length_of_file, 1))

        label_matrix = np.append(label_matrix, file_label_matrix, axis=0)

    feature_matrix = np.array(feature_matrix)

    return feature_matrix, label_matrix

def z(W, x):
    """
    Multiplies weight matrix W with feature vector x

    :param W: weight matrix
    :param x: feature vector
    :return: product of weight matrix W and feature vector x
    """
    return np.matmul(W, x)


def sigmoid(z):
    """
    Sigmoid function(monotonically increasing function which only outputs values between 0 and 1)

    :param z: numerical value
    :return: value between 0 and 1
    """
    return 1 / (1 + np.exp(-z))


def g(W, x):
    """
    Linear model function

    :param W: weight matrix
    :param x: feature vector
    :return: product of weight matrix W and feature vector x restricted to be between 0 and 1 with sigmoid function
    """
    return sigmoid(z(W, x))


def create_W(classes, features):
    """
    Initializes weight vector with all zeros and shape (classes X features + 1)

    :param classes: amount of classes
    :param features: amount of features
    :return: weight matrix with only zeros
    """
    W = np.zeros(shape=(classes, features + 1))
    return W


def MSE_grad_g(W, x, t):
    """
    Calculates gradient of MSE with respect to g(x)

    :param W: weight matrix
    :param x: feature vector
    :param t: label vector
    :return: gradient of MSE with respect to g(x)
    """
    return g(W, x) - t


def g_grad_z(W, x):
    """
    Calculates gradient of MSE with respect to z(x)

    :param W: weight matrix
    :param x: feature vector
    :return: gradient of MSE with respect to z(x)
    """
    return g(W, x) * (1 - g(W, x))


def z_grad_W(x):
    """
    Calculates gradient of z(x) with respect to W

    :param x: feature vector
    :return: gradient of z(x) with respect to W
    """
    return np.array([[x_n] for x_n in x])


def MSE_grad_W(W, x, t):
    """
    Calculates gradient of MSE with respect to weight matrix W

    :param W: weight matrix
    :param x: feature vector
    :param t: label vector
    :return: gradient of MSE with respect to weight matrix W
    """
    return (MSE_grad_g(W, x, t) * g_grad_z(W, x)) * z_grad_W(x)


def sum_MSE_grad_W(W, x_matrix, t_matrix):
    """
    Calculates sum of gradient of MSE with respect to weight matrix W for several feature and label vectors

    :param W: weight matrix
    :param x_matrix: feature matrix
    :param t_matrix: label matrix
    :return: sum of gradient of MSE with respect to weight matrix W for several feature and label vectors
    """
    res_sum = np.transpose(np.zeros(shape=W.shape))
    for x, t in zip(x_matrix, t_matrix):
        res_sum += MSE_grad_W(W, x, t)
    return res_sum


def MSE(g_matrix, t_matrix):
    return np.sum(np.power(g_matrix - t_matrix, 2))


def getPredictedClass(g, axis=None):
    return np.argmax(g, axis=axis)


def getActualClass(t, axis=None):
    return np.argmax(t, axis=axis)


def isCorrect(g_matrix, t_matrix):
    return getPredictedClass(g_matrix, axis=1) == getActualClass(t_matrix, axis=1)


def isNotCorrect(g_matrix, t_matrix):
    return getPredictedClass(g_matrix, axis=1) != getActualClass(t_matrix, axis=1)


def succes_rate(g_matrix, t_matrix):
    return np.sum(isCorrect(g_matrix, t_matrix))


def error_rate(g_matrix, t_matrix):
    return np.sum(isNotCorrect(g_matrix, t_matrix))


def train_linear_classifier_iteration(W, training_x_matrix, training_t_matrix, alpha):
    N = 10  # Num of data vectors used to calculate MSE gradient
    W_array = []
    for k in range(0, len(training_x_matrix), N):
        W -= alpha * np.transpose(sum_MSE_grad_W(W, training_x_matrix[k:k + N], training_t_matrix[k:k + N]))
        W_array.append(W)
    return W_array


def shuffle(x_matrix, t_matrix):
    """
    Shuffles x_matrix and t_matrix together

    :param x_matrix: feature matrix
    :param t_matrix: label matrix
    :return: shuffled matrices
    """

    shuffle_matrix = list(zip(x_matrix, t_matrix))
    random.shuffle(shuffle_matrix)
    x_matrix, t_matrix = zip(*shuffle_matrix)
    return x_matrix, t_matrix


def shuffle_in_unison(a_matrix, b_matrix):
    rng_state = np.random.get_state()
    np.random.shuffle(a_matrix)
    np.random.set_state(rng_state)
    np.random.shuffle(b_matrix)
    return a_matrix, b_matrix


def plot_incorrect_and_correct(W, x_matrix, t_matrix, g_matrix=None):
    if g_matrix==None:
        g_matrix = np.transpose(g(W, np.transpose(x_matrix)))
    actual_classes = getActualClass(t_matrix, axis=1)
    predicted_classes = getPredictedClass(g_matrix, axis=1)

    colors = ['r', 'g', 'b']
    for x, actual, pred in zip(x_matrix, actual_classes, predicted_classes):
        if actual == pred:
            sign = 'x'
        else:
            sign = "o"

        color = colors[actual]

        plt.subplot(2, 1, 1)
        plt.plot(x[1], x[2], marker=sign, c=color)
        plt.subplot(2, 1, 2)
        plt.plot(x[3], x[4], marker=sign, c=color)
    plt.show()


def get_confusion_matrix(g_matrix, t_matrix):
    actual_classes = getActualClass(t_matrix, axis=1)
    predicted_classes = getPredictedClass(g_matrix, axis=1)
    return confusion_matrix(actual_classes, predicted_classes)


def plot_confusion_matrix(g_matrix, t_matrix):
    confusion_matrix = get_confusion_matrix(g_matrix, t_matrix)

    print(confusion_matrix)

    ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Setosa', 'Versicolor', 'Virginia'])
    ax.yaxis.set_ticklabels(['Setosa', 'Versicolor', 'Virginia'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()


def main():
    file_array = [f'class_{i}' for i in range(1, 3 + 1)]
    class_array = [i for i in range(len(file_array))]
    #class_array = [1,2,3]

    x_matrix, t_matrix = getFeatureAndLabelMatrix(file_array, class_array)

    print("Retrived matricies")

    x_matrix, t_matrix = shuffle_in_unison(x_matrix, t_matrix)

    print("Shuffled matricies")

    training_x_matrix = x_matrix[:90]
    training_t_matrix = t_matrix[:90]

    test_x_matrix = x_matrix[90:]
    test_t_matrix = t_matrix[90:]

    classes = max(class_array) + 1
    features = len(x_matrix[0]) - 1

    num_of_iterations = 10000
    W_array = [create_W(classes=classes, features=features)]
    for i in range(num_of_iterations):
        W_array = np.append(W_array, train_linear_classifier_iteration(W_array[-1],
                                training_x_matrix, training_t_matrix, alpha=0.01), axis=0)

    print("Ws caluclated")

    test_g_matrix = np.transpose(g(W_array[-1], np.transpose(test_x_matrix)))

    plot_confusion_matrix(test_g_matrix, test_t_matrix)

    plot_incorrect_and_correct(W_array[-1], test_x_matrix, test_t_matrix)


    W_array = np.array(W_array)

    mse_array = []
    error_rate_array = []
    for W in W_array:
        test_g_matrix = np.transpose(g(W, np.transpose(test_x_matrix)))
        mse_array.append(MSE(test_g_matrix, test_t_matrix))
        error_rate_array.append(error_rate(test_g_matrix, test_t_matrix))

    print("MSEs caluclated")
    print("Errors calculated")

    plt.subplot(2, 1, 1)
    plt.title("MSE")
    plt.plot(np.arange(len(mse_array))/10, mse_array)
    plt.subplot(2, 1, 2)
    plt.title("Error rate")
    plt.plot(np.arange(len(mse_array))/10, error_rate_array)
    plt.show()


if __name__ == '__main__':
    main()
