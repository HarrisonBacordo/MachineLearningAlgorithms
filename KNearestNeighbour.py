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
    if not v1 or not v2:
        return
    for x in range(length):
        distance += pow((float(v1[0][5 * x:5 * x + 3]) - float(v2[0][5 * x:5 * x + 3])), 2)
    return math.sqrt(distance)


def neighbours(v, data, k):
    """
    finds the nearest k neighbours to a given vector based on given data
    :param v: point of concern
    :param data: dataset of all points
    :param k: number of neighbours to calculate
    :return: list of neighbours
    """
    dist_list = []
    for v2 in data:
        distance = dist(v, v2, 4)
        if distance is not None:
            dist_list.append(distance)
    min_list = []
    min_index_list = []
    while len(min_list) <= k:
        i = 0
        min_index = 0
        min_val = 0
        for d in dist_list:
            print(i, d)
            if i == 0:
                min_index = 0
                min_val = d
            elif d <= min_val:
                min_index = i
                min_val = d
            i += 1
        del dist_list[min_index]
        min_index_list.append(min_index)
        min_list.append(min_val)
    #     TODO potential index shift problem when deleting
    return min_index_list, min_list


def main(train, test):
    training_data, test_data = prep_data(train, test)
    # test for train and test format
    print("TRAIN: ", training_data)
    print("TEST: ", test_data)
    # test for euclidean distance
    # test for finding k nearest neighbours
    print(neighbours(test_data[0], training_data, 3))


if __name__ == '__main__':
    train_file = 'ass1-data/part1/iris-training.txt'
    test_file = 'ass1-data/part1/iris-test.txt'
    main(train_file, test_file)
