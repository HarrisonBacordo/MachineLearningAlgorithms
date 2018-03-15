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