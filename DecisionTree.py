import csv


class Node:

    def __init__(self, attr, left, right):
        self.attr = attr
        self.left = left
        self.right = right

    def report(self, indent):
        print(indent, self.attr, "= TRUE")
        self.left.report(indent.join("  "))
        print(indent, self.attr, " = FALSE")
        self.right.report(indent + "  ")


class Leaf:
    def __init__(self, clas, chance):
        self.clas = clas
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
    attr = list()
    train_instances = list()
    test_instances = list()
    content = [i.strip().split() for i in open(training).readlines()]
    content[1].insert(0, "CLASS")
    with open("train.csv", 'wt') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for i in range(len(content)):
            if i > 0:
                if i == 1:
                    for j in range(len(content[i])):
                        attr.append(j)
                else:
                    train_instances.append(content[i])
                writer.writerow(content[i])

    content = [i.strip().split() for i in open(test).readlines()]
    content[1].insert(0, "CLASS")
    with open("test.csv", 'wt') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for i in range(len(content)):
            if i > 0:
                if i != 1:
                    test_instances.append(content[i])
                writer.writerow(content[i])
    return train_instances, test_instances, attr


def run_test(instance, tree):
    if isinstance(tree, Leaf):
        if instance[0] == tree.clas:
            return 1
        else:
            return 0
    else:
        if instance[tree.attr] == 'true':
            return run_test(instance, tree.left)
        elif instance[tree.attr] == 'false':
            return run_test(instance, tree.right)


def compute_purity(instances):
    live, die = split_instances(0, instances)
    if not instances:
        return 0
    if len(live) > len(die):
        return len(live)/len(instances)
    return len(die)/len(instances)


def split_instances(attr, instances):
    true = list()
    false = list()
    for instance in instances:
        if instance[attr] == "true" or instance[attr] == "live":
            true.append(instance)
        elif instance[attr] == "false" or instance[attr] == "die":
            false.append(instance)
    return true, false


def same_class(instances):
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
        return Leaf(clas, compute_purity(instances))
    elif len(attr) != 1:
        top_avg = None
        best_attr = None
        best_true = None
        best_false = None
        for a in attr:
            if a != 0:
                true_set, false_set = split_instances(a, instances)
                avg = (compute_purity(true_set) + compute_purity(false_set)) / 2
                if not best_true or not top_avg or avg > top_avg:
                    top_avg = avg
                    best_attr = a
                    best_true = true_set
                    best_false = false_set
        if best_attr:
            attr.remove(best_attr)
            left = build_tree(best_true, attr, baseline)
            right = build_tree(best_false, attr, baseline)
            return Node(best_attr, left, right)


def main(file1, file2):
    train_instances, test_instances, attr = prep_data(file1, file2)
    live, die = split_instances(0, train_instances)
    if len(live) > len(die):
        baseline = "live"
    else:
        baseline = "die"
    chance = compute_purity(train_instances)
    tree = build_tree(train_instances, attr, Leaf(baseline, chance))
    tree.report("  ")
    i = 0
    for instance in test_instances:
        i += run_test(instance, tree)
    percentage = i / len(test_instances)
    print(i, "/", len(test_instances))
    print(percentage)


if __name__ == '__main__':
    main('ass1-data/part2/hepatitis-training.dat', 'ass1-data/part2/hepatitis-test.dat')