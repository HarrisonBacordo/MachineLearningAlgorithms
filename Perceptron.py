import random


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
            self.row.append(random.randint(1, w))
            self.col.append(random.randint(1, h))
            self.connected.append(random.choice([True, False]))


def prep_data(file):
    """
    Correctly format the dataset to conform to the machine learning model
    :param file: filename of dataset
    :return: None
    """
    training = []
    #     LOGIC HERE
    return training


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


# TODO Add necessary args as you progress
def feed_forward(features, activation, epoch_num):
    """
    let data flow through the perceptron and through an activation function to output a prediction.
    :return: the prediction
    """


def calculate_cost():
    """
    Calculates the cost function of the prediction. i.e. the how far the prediction was from being fully correct
    :return:
    """


def minimize_cost():
    """
    Initiate back propagation; Change the weights connected to the perceptron according to stochastic gradient descent
    :return:
    """


def main(file):
    training_set = prep_data(file)
    features = construct_features(10, 10, 50)
    for feature in features:
        print("COL: ", feature.col, "\nROW: ", feature.row, "\nBOOL", feature.connected, "\n")

#     while cost is still higher than some number
#       feed forward training data
#       calculate cost of prediction
#       minimize cost via gradient descent

if __name__ == '__main__':
    filename = 'ass1-data/part3/image.data'
    main(filename)
