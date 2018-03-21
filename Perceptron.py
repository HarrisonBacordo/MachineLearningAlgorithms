import random
import numpy as np


class Feature:
    def __init__(self, w, h, row=None, col=None, connected=None):
        """
        connects to random pixels within the bounds of w and h, then sets a random boolean for each
        random pixel
        :param w: horizontal bounds of feature
        :param h: vertical bounds of feature
        :param row: list of row coordinates
        :param col: list of column coordinates
        :param connected: list of boolean values for each pixel
        """
        if connected is None:
            self.connected = list()
        if col is None:
            self.col = list()
        if row is None:
            self.row = list()
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
    pixels = list()
    labels = list()
    data = open(file, 'r')
    data = ''.join(data)
    # edit data to be easier to read
    data = data.replace('#', '')
    data = data.replace('P1', '')
    data = data.replace('10 10', '')
    data = data.replace('\n\n', '\n')
    data = data.splitlines()
    data.remove('')
    current_num = list()
    for i in data:
        # its a label
        if i.isalpha():
            if current_num:  # if not empty, then merge the numbers and add to pixels list.
                pixels.append(''.join(current_num[0]) + ''.join(current_num[1]))
                current_num = list()
            labels.append(i)
        # its an img binary
        elif i.isnumeric():
            current_num.append(i)
    pixels.append(''.join(current_num[0]) + ''.join(current_num[1]))
    imgs = list()
    for i in pixels:
        px = np.array(list(i)).reshape(10, 10)
        imgs.append(px)
    # convert "yes" and "otherwise" into 1 and -1 respectively
    for i in range(len(labels)):
        if labels[i] == "X":
            labels[i] = 1
        else:
            labels[i] = -1
    print(labels)

    return imgs, labels


def construct_features(w, h, n):
    """
    constructs the random features
    :param w: width of picture
    :param h: height of picture
    :param n: number of features to make
    :return: list of features
    """
    feature_list = list()
    for _ in range(n):
        feature_list.append(Feature(w, h))
    return feature_list


def feature_values(feats, imgs):
    """
    Convert each img into its appropriate feature values to be used as inputs for
    the perceptron
    :param feats: feature objects
    :param imgs: images in binary
    :return: list of all feature values for each img
    """
    img_features = list()
    all_img_features = list()
    for img in imgs:
        for i, f in enumerate(feats):
            total = 0
            for j in range(4):
                if int(img[f.row[j], f.col[j]]) == f.connected[j]:
                    total += 1
            if total >= 3 or i == 50:   # check if dummy variable or if sum > 3
                img_features.append(1)
            else:
                img_features.append(0)
        all_img_features.append(img_features)  # add feature values to list of feature values
        img_features = list()
    all_img_features = np.array(all_img_features)  # convert to numpy array
    return all_img_features


def feed_forward(features, weights):
    """
    let data flow through the perceptron and through an activation function to output a prediction.
    :return: the prediction
    """
    total = 0
    # multiply current feature with appropriate weight. add result to the current total
    for i in range(len(features)):
        total += features[i] * weights[i]
    return 1 if total > 0 else -1   # filter through activation function


def main(file):
    imgs, labels = prep_data(file)
    features = construct_features(10, 10, 51)   # 51 features, 1 for dummy
    imgs = feature_values(features, imgs)   # convert images into appropriate feature values
    weights = [0.0 for _ in range(51)]  # start weights at zero
    # beginning of training session. keep training until weights converge
    correct = 0
    epoch = 0
    while correct != 100:
        correct = 0
        epoch += 1
        for i in range(100):
            guess = feed_forward(imgs[i], weights)
            if guess != labels[i]:  # guess is wrong
                if guess == 1:
                    weights = weights - imgs[i]
                else:
                    weights = weights + imgs[i]
            else:   # guess is correct
                correct += 1
        print("CYCLE: ", epoch, " CORRECT: ", correct, " OUT OF: 100")
    # weights have converged. print resulting values
    print("\nCONVERGED WEIGHTS: ", weights)
    for i, feature in enumerate(features):
        print("\nFEATURE ", i, "\nROW: ", feature.row, "\nCOL: ", feature.col, "\nBOOLS: ", feature.connected)


if __name__ == '__main__':
    filename = 'ass1-data/part3/image.data'
    main(filename)
