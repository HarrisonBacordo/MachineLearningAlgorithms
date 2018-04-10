import csv

global attributes


class Node:

    def __init__(self, attr, left, right):
        """
        Represents a (sub)tree with both a left and a right branch, extending to either a leaf or another node
        :param attr: the attribute this node represents
        :param left: what branches to the left of this node
        :param right: what branches to the right this node
        """
        self.attr = attr
        self.left = left
        self.right = right

    def report(self, indent):
        print(indent, attributes[self.attr], "= TRUE")
        self.left.report(indent.join("  "))
        print(indent, attributes[self.attr], " = FALSE")
        self.right.report(indent + "  ")


class Leaf:
    def __init__(self, _class, chance):
        """
        Represents the end of a branch; part of a tree with neither a left or right branch. Holds a classification and
        a possibility of that classification being true
        :param _class: classification this leaf represents
        :param chance: probability that this classification is correct
        """
        self.clas = _class
        self.chance = chance

    def report(self, indent):
        print(indent, "CLASS: ", self.clas, ", PROB: ", self.chance)


def prep_data(training, test):
    """
    Correctly format the dataset to conform to the machine learning model
    :param training: set of training data
    :param test: set of test data
    :return: list of instances and and attributes for both training and test
    """
    attrnum = list()
    attr = list()
    train_instances = list()
    test_instances = list()
    # remove trailing whitespaces. insert a class column into the dataset for the live/die classification
    content = [j.strip().split() for j in open(training).readlines()]
    content[1].insert(0, "CLASS")
    # convert to csv file for easier read
    with open("train.csv", 'wt') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for i in range(len(content)):
            if i > 0:
                # attribute; add this row's indices to the attribute list
                if i == 1:
                    for j, attribute in enumerate(content[i]):
                        attrnum.append(j)
                        attr.append(attribute)
                else:
                    train_instances.append(content[i])
                writer.writerow(content[i])
    # do same to test dataset
    content = [i.strip().split() for i in open(test).readlines()]
    content[1].insert(0, "CLASS")
    with open("test.csv", 'wt') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for i in range(len(content)):
            if i > 0:
                if i != 1:
                    test_instances.append(content[i])
                writer.writerow(content[i])
    return train_instances, test_instances, attrnum, attr


def classify_instance(instance, tree):
    """
    classifies the given instance using the given decision tree
    :param instance: data piece to be classified
    :param tree: classifier
    :return: 1 for correct guess, 0 for incorrect guess
    """
    if isinstance(tree, Leaf):
        if instance[0] == tree.clas:
            print("CORRECT: ", instance)
            return 1
        else:
            print("INCORRECT: ", instance)
            return 0
    else:
        if instance[tree.attr] == 'true':
            return classify_instance(instance, tree.left)
        elif instance[tree.attr] == 'false':
            return classify_instance(instance, tree.right)


def compute_impurity(instances):
    """
    calculates the purity of the given list of instances
    :param instances: instances to compute purity on
    :return: percent purity of given list
    """
    live, die = split_instances(0, instances)
    if not instances:
        return 0
    ttt = 2 * (len(live) * len(die))/pow(len(live) + len(die), 2)
    return ttt


def split_instances(attr, instances):
    """
    splits the given list of instances based on their classification of the given attribute
    :param attr: attribute to split instances on
    :param instances: instances to split
    :return: 2 lists of separated instances
    """
    true = list()
    false = list()
    for instance in instances:
        if instance[attr] == "true" or instance[attr] == "live":
            true.append(instance)
        elif instance[attr] == "false" or instance[attr] == "die":
            false.append(instance)
    return true, false


def same_class(instances):
    """
    determine if all of the instances in the given list are of the same classification or not
    :param instances: list of instances to check class of
    :return: whether or not they are all the same class
    """
    clas = None
    for instance in instances:
        if not clas:
            clas = instance[0]
        elif clas != instance[0]:
            return False
    return True


def build_tree(instances, attr, baseline):
    """
    :param instances: the set of training instances that have been provided to the node being constructed
    :param attr: the list of attributes that were not used on the path from the root to this node
    :param baseline: the baseline leaf for most probable class over entire dataset
    :return:
    """
    if not instances:
        return baseline
    if same_class(instances):
        return Leaf(instances[0][0], 1)
    if len(attr) == 1:
        live, die = split_instances(0, instances)
        if len(live) > len(die):
            clas = "live"
        else:
            clas = "die"
        return Leaf(clas, compute_impurity(instances))
    # we choose 1 since the classification of the instance counts as a column in the dataset.
    elif len(attr) != 1:
        top_avg = None
        best_attr = None
        best_true = None
        best_false = None
        # cycle through attributes, get their purity for both true or false sides of attribute. determine which
        # is most significant (most pure on average b/t true and false sets)
        for a in attr:
            if a != 0:
                true_set, false_set = split_instances(a, instances)
                weighttrue = len(true_set) / len(instances)
                weightfalse = len(false_set) / len(instances)
                avg = 1 - (((compute_impurity(true_set) * weighttrue) + (compute_impurity(false_set) * weightfalse))/2)

                if not best_true or not top_avg or avg > top_avg:
                    top_avg = avg
                    best_attr = a
                    best_true = true_set
                    best_false = false_set
        # remove best attribute from attr list, constuct a node with best attribute. continue recursion of build_tree
        attr.remove(best_attr)
        left = build_tree(best_true, attr[:], baseline)
        right = build_tree(best_false, attr[:], baseline)
        return Node(best_attr, left, right)


def main(file1, file2):
    global attributes
    train_instances, test_instances, attrnum, attributes = prep_data(file1, file2)
    # determine baseline and chance of baseline of dataset
    live, die = split_instances(0, train_instances)
    if len(live) > len(die):
        baseline = "live"
    else:
        baseline = "die"
    chance = compute_impurity(train_instances)
    # construct the tree with the training set. Print it afterward
    tree = build_tree(train_instances, attrnum, Leaf(baseline, chance))
    tree.report("")
    # test and get score of accuracy against test instances.
    i = 0
    for instance in test_instances:
        i += classify_instance(instance, tree)
    percentage = round(i / len(test_instances) * 100, 2)
    print(i, "/", len(test_instances))
    print(percentage)


if __name__ == '__main__':
    main('../ass1-data/part2/hepatitis-training.dat', '../ass1-data/part2/hepatitis-test.dat')
    for i in range(1, 10):
        if i == 10:
            print("\n\nSET", i)
            train = f'../ass1-data/part2/hepatitis-training-run{i}.dat'
            test = f'../ass1-data/part2/hepatitis-test-run{i}.dat'
            main(train, test)
        else:
            print("\n\nSET", i)
            train = f'../ass1-data/part2/hepatitis-training-run0{i}.dat'
            test = f'../ass1-data/part2/hepatitis-test-run0{i}.dat'
            main(train, test)
