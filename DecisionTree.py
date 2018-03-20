import csv

def prep_data(training, test):
    """
    Correctly format the dataset to conform to the machine learning model
    :param file: filename of dataset
    :param split: how to split dataset b/t training and test
    :param training: list of training data
    :param test: list of test data
    :return: None
    """
    content = [i.strip().split() for i in open(training).readlines()]
    with open("train.csv", 'wt') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for i in range(len(content)):
            writer.writerow(content[i])

    content = [i.strip().split() for i in open(test).readlines()]
    with open("test.csv", 'wt') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for i in range(len(content)):
            writer.writerow(content[i])





def build_tree(instance, attr):
    """
    :param instance:
    :param attr:
    :return:
    """


def main(file1, file2):
    prep_data(file1, file2)

if __name__ == '__main__':
    main('ass1-data/part2/hepatitis-training.dat', 'ass1-data/part2/hepatitis-test.dat')