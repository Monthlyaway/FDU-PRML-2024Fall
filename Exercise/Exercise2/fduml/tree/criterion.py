"""
criterion
"""

import math


def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """Count the number of labels of nodes"""
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels


def __entropy(labels):
    total_count = sum(labels.values())
    result = .0

    for count in labels.values():
        if count > 0:
            p = count / total_count
            result += -p * math.log2(p)

    return result


def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain if splitting y into      #
    # l_y and r_y                                                             #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    entropy_parent = __entropy(all_labels)
    entropy_left = __entropy(left_labels)
    entropy_right = __entropy(right_labels)

    num_parent = sum(all_labels.values())
    num_left = sum(left_labels.values())
    num_right = sum(right_labels.values())
    info_gain = entropy_parent - num_left / num_parent * \
        entropy_left - num_right / num_parent * entropy_right
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return info_gain


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain ratio if splitting y     #
    # into l_y and r_y                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_parent = len(y.reshape(-1))
    num_left = len(l_y.reshape(-1))
    num_right = len(r_y.reshape(-1))

    intrinsic_value = - num_left / num_parent * \
        math.log2(num_left / num_parent) - num_right / \
        num_parent * math.log2(num_right / num_parent)

    info_gain = info_gain / intrinsic_value

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return info_gain


def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the gini index value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Get label distributions for parent, left, and right nodes
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    # Helper function to calculate Gini for a given set of labels
    def gini(labels):
        total = sum(labels.values())
        gini_value = 1.0
        for count in labels.values():
            p = count / total
            gini_value -= p ** 2
        return gini_value

    # Calculate Gini for parent, left, and right nodes
    before = gini(all_labels)
    gini_left = gini(left_labels)
    gini_right = gini(right_labels)

    # Number of samples in parent and child nodes
    num_parent = sum(all_labels.values())
    num_left = sum(left_labels.values())
    num_right = sum(right_labels.values())

    after = (num_left / num_parent) * gini_left + \
        (num_right / num_parent) * gini_right

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after


def __error_rate(y, l_y, r_y):
    """Calculate the error rate"""
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the error rate value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    def error_rate(labels):
        total = sum(labels.values())
        if total == 0:
            return 0
        majority = max(labels.values())
        return 1 - majority / total

    before = error_rate(all_labels)
    left_rate = error_rate(left_labels)
    right_rate = error_rate(right_labels)

    num_parent = sum(all_labels.values())
    num_left = sum(left_labels.values())
    num_right = sum(right_labels.values())
    after = num_left / num_parent * left_rate + num_right / num_parent * right_rate

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after
