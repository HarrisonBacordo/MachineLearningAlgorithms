def prep_data(file, split, training=None, test=None):
    """
    Correctly format the dataset to conform to the machine learning model
    :param file: filename of dataset
    :param split: how to split dataset b/t training and test
    :param training: list of training data
    :param test: list of test data
    :return: None
    """
    if training is None:
        training = []
    if test is None:
        test = []


# TODO Add necessary args as you progress
def feed_forward():
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


filename = 'ass1-data/part3/image.data'
