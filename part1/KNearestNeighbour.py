import csv
import math
import numpy as np


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


def eval_neighbours(v, data, k):
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
    neigh_index = np.argpartition(np.array(dist_list), k)[:k].tolist()
    neighbours = []
    for index in neigh_index:
        neighbours.append(data[index][0])
    return neighbours


def predict(neighbours):
    score = {}
    for n in neighbours:
        label = str(n).rsplit(' ', 1)[1]
        if label in score:
            score[label] += 1
        else:
            score[label] = 1
    return max(score, key=score.get)


def main(train, test):
    training_data, test_data = prep_data(train, test)
    i = 1
    for row in test_data:
        neighbours = eval_neighbours(row, training_data, 5)
        if neighbours:
            print(i, predict(neighbours))
            i += 1


if __name__ == '__main__':
    train_file = '../ass1-data/part1/iris-training.txt'
    test_file = '../ass1-data/part1/iris-test.txt'
    main(train_file, test_file)
