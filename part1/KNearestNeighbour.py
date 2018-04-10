import csv
import math
import numpy as np

ranges = list()


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
        distance += pow((float(v1[0][5 * x:5 * x + 3]) - float(v2[0][5 * x:5 * x + 3])), 2) / pow(ranges[x], 2)
    return math.sqrt(distance)


def eval_neighbours(v, data, k):
    """
    finds the nearest k neighbours to a given vector based on given data
    :param v: point of concern
    :param data: dataset of all points
    :param k: number of neighbours to calculate
    :return: list of neighbours
    """
    dist_list = list()

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
    a1 = list()
    a2 = list()
    a3 = list()
    a4 = list()
    label = list()
    for instance in training_data:
        if instance:
            a1.append(float(instance[0][0:3]))
            a2.append(float(instance[0][5:8]))
            a3.append(float(instance[0][10:13]))
            a4.append(float(instance[0][15:18]))
            label.append(instance[0][20:])
    ranges.append(round(max(a1) - min(a1), 2))
    ranges.append(round(max(a2) - min(a2), 2))
    ranges.append(round(max(a3) - min(a3), 2))
    ranges.append(round(max(a4) - min(a4), 2))

    correct = 0
    k = 3
    for i, row in enumerate(test_data):
        neighbours = eval_neighbours(row, training_data, k)
        if neighbours:
            if predict(neighbours) == label[i]:
                correct += 1
            print(i, predict(neighbours))
    print(f'FOR K = {k}, {correct}/75, {round(correct/75 * 100, 2)}%')


if __name__ == '__main__':
    train_file = '../ass1-data/part1/iris-training.txt'
    test_file = '../ass1-data/part1/iris-test.txt'
    main(train_file, test_file)
