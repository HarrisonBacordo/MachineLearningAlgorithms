import csv
import math


def prep_data(training, test):
    """
    Correctly format the dataset to conform to the machine learning model
    :param training: list of training data
    :param test: list of test data
    :return: None
    """
    with open(training, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        training_data = list(lines)
    with open(test, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        test_data = list(lines)
    return training_data, test_data


def dist(v1, v2, length):
    """
    Calculates the euclidean distance between two vectors
    :param v1: first data piece
    :param v2: second data piece
    :param length: amount of numbers in each vector
    :return: distance
    """
    distance = 0
    for x in range(length):
        distance += pow((v1[x] - v2[x]), 2)
    return math.sqrt(distance)


def neighbours(v, data, k):
    """

    :param v: point of concern
    :param data: dataset of all points
    :param k: number of neighbours to calculate
    :return: list of neighbours
    """


def main(train, test):
    training_data, test_data = prep_data(train, test)
    # test for train and test format
    print("TRAIN: ", training_data)
    print("TEST: ", test_data)
    # test for euclidean distance
    x = [4, 4, 4, 4, 'string']
    y = [8, 8, 8, 8, 'string']
    print(dist(x, y, 4))


if __name__ == '__main__':
    train_file = 'ass1-data/part1/iris-training.txt'
    test_file = 'ass1-data/part1/iris-test.txt'
    main(train_file, test_file)

