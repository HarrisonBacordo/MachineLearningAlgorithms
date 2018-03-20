import random
import numpy as np
import operator as op


class Feature:
    def __init__(self, w, h, row=None, col=None, connected=None):
        """

        :param w:
        :param h:
        :param row:
        :param col:
        :param connected:
        """
        if connected is None:
            self.connected = []
        if col is None:
            self.col = []
        if row is None:
            self.row = []
        for _ in range(4):
            self.row.append(random.randint(1, w - 1))
            self.col.append(random.randint(1, h - 1))
            self.connected.append(random.choice([0, 1]))


def prep_data(file):
    """
    Correctly format the dataset to conform to the machine learning model
    :param file: filename of dataset
    :return: 2d array of pixels as well as appropriate labels
    """
    pixels = []
    labels = []
    data = open(file, 'r')
    data = ''.join(data)
    data = data.replace('#', '')
    data = data.replace('P1', '')
    data = data.replace('10 10', '')
    data = data.replace('\n\n', '\n')
    data = data.splitlines()
    data.remove('')
    current_num = []
    for i in data:
        if i.isalpha():
            if current_num:
                pixels.append(''.join(current_num[0]) + ''.join(current_num[1]))
                current_num = []
            labels.append(i)
        elif i.isnumeric():
            current_num.append(i)
    pixels.append(''.join(current_num[0]) + ''.join(current_num[1]))
    imgs = []
    for i in pixels:
        px = np.array(list(i)).reshape(10, 10)
        imgs.append(px)
    for i in range(len(labels)):
        if labels[i] == "Yes":
            labels[i] = 1
        else:
            labels[i] = -1
    print(labels)

    return imgs, labels


def construct_features(w, h, n):
    """

    :param w: width of picture
    :param h: height of picture
    :param n: number of features to make
    :return: list of features
    """
    feature_list = []
    for _ in range(n):
        feature_list.append(Feature(w, h))
    return feature_list


def feature_values(feats, imgs):
    """
    Convert each img into its appropriate feature values to be used as inputs for
    the perceptron
    :param feats: feature objects
    :param imgs: imgages in binary
    :return: list of all feature values for each img
    """
    img_bools = []
    all_img_bools = []
    for img in imgs:
        for i, f in enumerate(feats):
            total = 0
            for j in range(4):
                if int(img[f.row[j], f.col[j]]) == f.connected[j]:
                    total += 1
            if total >= 3 or i == 50:
                img_bools.append(1)
            else:
                img_bools.append(0)
        all_img_bools.append(img_bools)
        img_bools = []
    all_img_bools = np.array(all_img_bools)
    return all_img_bools


def initialize_weights():
    """
    give weights ar andom value between
    :return:
    """
    return [0.0 for i in range(51)]


# TODO Add necessary args as you progress
def feed_forward(features, weights, activation, epoch_num):
    """
    let data flow through the perceptron and through an activation function to output a prediction.
    :return: the prediction
    """
    total = 0
    for i in range(len(features)):
        total += features[i] * weights[i]
    return 1 if total > 0 else -1


def calculate_cost(label, prediction):
    """
    Calculates the cost function of the prediction. i.e. the how far the prediction was from being fully correct
    :return:
    """
    return label - prediction


def minimize_cost(weights, cost, learn_rate, label):
    """
    Initiate back propagation; Change the weights connected to the perceptron according to stochastic gradient descent
    :return:
    """
    for i in range(len(weights)-1):
        weights[i] = weights[i] + learn_rate * cost * label
    return weights


def main(file):
    imgs, labels = prep_data(file)
    features = construct_features(10, 10, 51)
    imgs = feature_values(features, imgs)
    weights = initialize_weights()
    correct = 0
    epoch = 0
    while correct != 100:
        correct = 0
        epoch += 1
        for i in range(100):
            guess = feed_forward(imgs[i], weights, None, None)
            if guess != labels[i]:
                if guess == 1:
                    weights = weights - imgs[i]
                else:
                    weights = weights + imgs[i]
            else:
                correct += 1
        print("CYCLE: ", epoch, " CORRECT: ", correct, " OUT OF: 100")
    print("\nCONVERGED WEIGHTS: ", weights)
    for i, feature in enumerate(features):
        print("\nFEATURE ", i, "\nROW: ", feature.row, "\nCOL: ", feature.col, "\nBOOLS: ", feature.connected)


if __name__ == '__main__':
    filename = 'ass1-data/part3/image.data'
    main(filename)
