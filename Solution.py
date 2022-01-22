import numpy as np
from scipy import stats
import pandas as pd
import math
import pprint
eps = np.finfo(float).eps
from numpy import log2 as log
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# To avoid a pandas related warning
pd.options.mode.chained_assignment = None  # default='warn'

# To keep track of the continous variables
continous_attributes = ['A2','A3','A8','A11','A14','A15']

# Preprocess
# 1. Change the continous variables to float
# 2. Take care of missing variables : Mode for discrete variables, mean for continous variables
# 3. Split data from labels
def preprocess(data):
    continous_index = [1, 2, 7, 10, 13, 14]
    # continous_index = [1]
    for i in range(len(data[0])):
        if (i in continous_index):
            sum_cont_mean = 0
            count = 0
            mean_cont = 0
            missing_indices = []
            for j in range(len(data)):
                if (data[j][i] == '?'):
                    missing_indices.append((j, i))
                    data[j][i] = 0.0
                else:
                    data[j][i] = float(data[j][i])
                    sum_cont_mean += float(data[j][i])
                    count += 1
            if (count != 0):
                mean_cont = sum_cont_mean / count
            for elements in missing_indices:
                data[elements[0]][elements[1]] = mean_cont
        else:
            count = 0
            missing_indices = []
            not_missing_indices = []
            for j in range(len(data)):
                if (data[j][i] == '?'):
                    missing_indices.append((j, i))
                    count += 1
                else:
                    not_missing_indices.append(data[j][i])
            if (count != 0):
                mode = stats.mode(not_missing_indices)[0]
                for elements in missing_indices:
                    data[elements[0]][elements[1]] = mode[0]
    return (data)

# Function to calculate the info gain at all splits of a sorted continous attribute and target
def continous_info_gain_df(sorted_list, entropy_target):
    info_gain_arr = []
    unzip_list = [[i for i, j in sorted_list],
                  [j for i, j in sorted_list]]
    sorted_arr = unzip_list[0]
    y_sorted = unzip_list[1]

    splits = []
    # Find where all array changes, (Potential points of split)
    for i in range(len(sorted_arr) - 1):
        if (y_sorted[i] != y_sorted[i + 1]):
            splits.append(i)

    for split in splits:
        H_lesser = y_sorted[0:split + 1]
        H_greater = y_sorted[split + 1:]

        H_lesser_num = len(H_lesser)
        H_greater_num = len(H_greater)

        H_lesser_num_ratio = float(H_lesser_num) / (H_lesser_num + H_greater_num)
        H_greater_num_ratio = float(H_greater_num) / (H_lesser_num + H_greater_num)

        H_lesser_pos = [i for i in H_lesser if i == '+']
        H_lesser_neg = [i for i in H_lesser if i == '-']

        H_lesser_pos_num = len(H_lesser_pos)
        H_lesser_neg_num = len(H_lesser_neg)

        H_lesser_pos_num_ratio = float(H_lesser_pos_num) / (H_lesser_pos_num + H_lesser_neg_num)
        H_lesser_neg_num_ratio = float(H_lesser_neg_num) / (H_lesser_pos_num + H_lesser_neg_num)

        H_greater_pos = [i for i in H_greater if i == '+']
        H_greater_neg = [i for i in H_greater if i == '-']

        H_greater_pos_num = len(H_greater_pos)
        H_greater_neg_num = len(H_greater_neg)

        H_greater_pos_num_ratio = float(H_greater_pos_num) / (H_greater_pos_num + H_greater_neg_num)
        H_greater_neg_num_ratio = float(H_greater_neg_num) / (H_greater_pos_num + H_greater_neg_num)

        H_expected_entropy_lesser_pos = 0
        H_expected_entropy_lesser_neg = 0
        if (H_lesser_neg_num_ratio > 0):
            H_expected_entropy_lesser_neg = math.log2(H_lesser_neg_num_ratio)
        else:
            H_expected_entropy_lesser_neg = 0
        if (H_lesser_pos_num_ratio > 0):
            H_expected_entropy_lesser_pos = math.log2(H_lesser_pos_num_ratio)
        else:
            H_expected_entropy_lesser_pos = 0

        H_expected_entropy_greater_pos = 0
        H_expected_entropy_greater_neg = 0
        if (H_greater_pos_num_ratio > 0):
            H_expected_entropy_greater_pos = math.log2(H_greater_pos_num_ratio)
        else:
            H_expected_entropy_greater_pos = 0
        if (H_greater_neg_num_ratio > 0):
            H_expected_entropy_greater_neg = math.log2(H_greater_neg_num_ratio)
        else:
            H_expected_entropy_greater_neg = 0

        H_expected_entropy_lesser = H_lesser_num_ratio * (
                -H_lesser_pos_num_ratio * H_expected_entropy_lesser_pos - H_lesser_neg_num_ratio * H_expected_entropy_lesser_neg)
        H_expected_entropy_greater = H_greater_num_ratio * (
                -H_greater_pos_num_ratio * H_expected_entropy_greater_pos - H_greater_neg_num_ratio * H_expected_entropy_greater_neg)

        H_expected_entropy = H_expected_entropy_lesser + H_expected_entropy_greater

        H_information_gain = entropy_target - H_expected_entropy

        info_gain_arr.append(H_information_gain)

    max_info_gain = 0
    max_index = 0
    for index, info_gain in enumerate(info_gain_arr):
        if (info_gain > max_info_gain):
            max_info_gain = info_gain
            max_index = index
    split_point = float(sorted_arr[splits[max_index]] + sorted_arr[splits[max_index] + 1]) / 2.0
    return ((split_point, max_info_gain))

# Convert all continous attributes to discrete
def continous_to_discrete_df(data, attributes):
    continous_index = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
    for i in continous_index:
        if (i in attributes):
            to_sort_list_cont = []
            to_sort_list_target = []
            num_pos = 0
            num_neg = 0
            for j in range(len(data)):
                to_sort_list_cont.append(data[i][j])
                to_sort_list_target.append(data['target'][j])
                if (data['target'][j] == '+'):
                    num_pos += 1
                else:
                    num_neg += 1
            to_sort_cont = zip(to_sort_list_cont, to_sort_list_target)
            to_sort_cont = list(to_sort_cont)
            sorted_cont = sorted(to_sort_cont, key=lambda x: x[0])
            pos_ratio = num_pos / (num_neg + num_pos + eps)
            neg_ratio = num_neg / (num_neg + num_pos + eps)
            entropy_target = -pos_ratio * math.log2(pos_ratio + eps) - neg_ratio * math.log2(neg_ratio + eps)
            split_point, max_info_gain = continous_info_gain_df(sorted_cont, entropy_target)
            for j in range(len(data)):
                if (data[i][j] <= split_point):
                    data[i][j] = 'less' + str(split_point)
                else:
                    data[i][j] = 'grea' + str(split_point)
    return (data)

#Find entropy of the target
def find_entropy(df):
    Class = df.keys()[-1]
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value] / len(df[Class])
        entropy += -fraction * np.log2(fraction)
    return entropy

#Find entropy of a given attribute
def find_entropy_attribute(df, attribute):
    Class = df.keys()[-1]
    target_variables = df[Class].unique()
    variables = df[attribute].unique()
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            fraction = num / (den + eps)
            entropy += -fraction * log(fraction + eps)
        fraction2 = den / len(df)
        entropy2 += -fraction2 * entropy
    return abs(entropy2)

# Find the attribute with the maximum information gain
def find_winner(df, attributes):
    Entropy_att = []
    IG = []
    df = continous_to_discrete_df(df.copy(deep=True), attributes)
    # print(df)
    for key in df.keys()[:-1]:
        if (key not in attributes):
            IG.append(0)
        else:
            IG.append(find_entropy(df) - find_entropy_attribute(df, key))
    return (df.keys()[:-1][np.argmax(IG)], df)

#Return subtable based on the value of the attribute used
def get_subtable(df, node, value):
    if(node not in continous_attributes):
        return df[df[node] == value].reset_index(drop=True)
    else:
        if('less' in value):
            split_point = float(value[4:])
            return df[df[node] <= split_point].reset_index(drop=True)
        if ('grea' in value):
            split_point = float(value[4:])
            return df[df[node] > split_point].reset_index(drop=True)


def buildTree(df, attributes, tree=None):
    clValue, counts = np.unique(df['target'], return_counts=True)
    if len(counts) == 1:  # Checking purity of subset
        return(clValue[0])
    if (not attributes):
        if (counts[0] > counts[1]):
            return (clValue[0])
        else:
            return (clValue[1])
    split_attribute, df_step = find_winner(df.copy(deep=True), attributes)
    attValue = np.unique(df_step[split_attribute])
    attributes.remove(split_attribute)
    if tree is None:
        tree = {}
        tree[split_attribute] = {}

    for value in attValue:
        if(split_attribute not in continous_attributes):
            subtable = get_subtable(df.copy(deep=True), split_attribute, value)
            tree[split_attribute][value] = buildTree(subtable.copy(deep=True),
                                                     attributes.copy())  # Calling the function recursively

        else:
            subtable_step = get_subtable(df, split_attribute, value)
            tree[split_attribute][value] = buildTree(subtable_step.copy(deep=True),
                                                     attributes.copy())  # Calling the function recursively

    return tree


def recursiveParse(decisionTree, inputData, max_depth, df_training_accuracy):
    # import pdb; pdb.set_trace()
    if(max_depth == 0):
        clValue, counts = np.unique(df_training_accuracy['target'], return_counts=True)
        if(len(counts) == 1):
            return(clValue[0])
        if(counts[0] > counts[1]):
            return(clValue[0])
        else:
            return(clValue[1])
    if type(decisionTree) == type(""):
        return decisionTree
    key = [*decisionTree]
    values = decisionTree[key[0]]
    for v in values:
        if "grea" in v or "less" in v:
            v1 = float(v[4:])
            if float(inputData[key[0]]) > v1:
                return recursiveParse(decisionTree[key[0]][v], inputData, max_depth, df_training_accuracy[df_training_accuracy[key[0]] > v1].reset_index(drop=True))
            else:
                return recursiveParse(decisionTree[key[0]]["less" + str(v1)], inputData, max_depth-1, df_training_accuracy[df_training_accuracy[key[0]] <= v1].reset_index(drop=True))
        if inputData[key[0]] == v:
            return recursiveParse(decisionTree[key[0]][v], inputData, max_depth-1, get_subtable(df_training_accuracy.copy(deep=True), key[0], v))


def predict(decision_tree, inputData, max_depth, df_training_accuracy):
    key = [*decision_tree]
    values = decision_tree[key[0]]
    for v in values:
        if inputData[key[0]] == v:
            return recursiveParse(decision_tree[key[0]][v], inputData, max_depth-1, get_subtable(df_training_accuracy.copy(deep=True), key[0], v))

# Build and return results for tree with best max depth (4 in my case)
def DecisionTree(maxDepth=4):
    # Define the output variables
    training_labels = []
    training_preds = []
    test_labels = []
    test_preds = []

    # Load the training data - (490, 16)
    training_data = np.loadtxt("train.txt", dtype=str).tolist()

    # Preprocess the data - Take care of missing values
    processed_training_data = preprocess(training_data)

    # Creating a dataframe ofthe training data for ease of coding
    df_training = pd.DataFrame(processed_training_data,
                               columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12',
                                        'A13', 'A14', 'A15', 'target'])

    # All the names of the attributes in the data
    attributes = {'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12',
                                        'A13', 'A14', 'A15'}

    # Create the decision tree
    decision_tree = buildTree(df_training, attributes)

    df_training_accuracy = pd.DataFrame(processed_training_data,
                                        columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11',
                                                 'A12',
                                                 'A13', 'A14', 'A15', 'target'])

    columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12',
               'A13', 'A14', 'A15', 'target']
    for row in range(len(df_training_accuracy)):
        inputData = {}
        for items in columns:
            inputData[items] = df_training_accuracy[items][row]
        decision_tree_accuracy = decision_tree.copy()
        training_labels.append(df_training_accuracy['target'][row])
        training_preds.append(predict(decision_tree_accuracy, inputData, maxDepth, df_training_accuracy.copy(deep=True)))

    # Load the test data and use decision tree to predict its outputs
    test_data = np.loadtxt("test.txt", dtype=str).tolist()
    processed_test_data = preprocess(test_data)
    df_test_accuracy = pd.DataFrame(processed_test_data,
                                    columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11',
                                             'A12',
                                             'A13', 'A14', 'A15', 'target'])
    for row in range(len(df_test_accuracy)):
        inputData = {}
        for items in columns:
            inputData[items] = df_test_accuracy[items][row]
        # print(inputData)
        decision_tree_accuracy = decision_tree.copy()
        test_labels.append(df_test_accuracy['target'][row])
        test_preds.append(predict(decision_tree_accuracy, inputData, maxDepth, df_test_accuracy.copy(deep=True)))

    return ([training_labels, training_preds, test_labels, test_preds])



# To find the performance of the decision tree with pruning to a certain maxDepth

def DecisionTreeBounded(maxDepth):
    DecisionTree(maxDepth)

# I have used these functions below to find the optimal depth value. I have found it and applied
# that depth to the previous functions. The Check.py only calls the previous function in
# which gives the optimal model.

# eval:
#   Input: original training labels list, predicted training labels list,
#	       original testing labels list, predicted testing labels list.
#   Output: print out training and testing accuracy
def eval(o_train, p_train, o_validation, p_validation, o_test, p_test):
    # print('\nTraining Result!')
    train = accuracy(o_train, p_train)
    # print(o_train)
    # print(p_train)
    # print(train)
    # print('\nValidation Result!')
    validation = accuracy(o_validation, p_validation)
    # print(o_validation)
    # print(p_validation)
    # print(validation)
    # print('\nTesting Result!')
    test = accuracy(o_test, p_test)
    # print(o_test)
    # print(p_test)
    # print(test)
    return ((train, validation, test))


# accuracy:
#   Input: original labels list, predicted labels list
#   Output: print out accuracy
def accuracy(orig, pred):
    num = len(orig)
    if (num != len(pred)):
        print('Error!! Num of labels are not equal.')
        return
    match = 0
    for i in range(len(orig)):
        o_label = orig[i]
        p_label = pred[i]
        if (o_label == p_label):
            match += 1
    return (match / num)

def DecisionTreeDepthFind():
    # Define the output variables
    training_labels = []
    training_preds = []
    validation_labels = []
    validation_preds = []
    test_labels = []
    test_preds = []

    # Load the training data - (490, 16)
    training_data = np.loadtxt("train.txt", dtype=str).tolist()
    validation_data = np.loadtxt("validation.txt", dtype=str).tolist()
    test_data = np.loadtxt("test.txt", dtype=str).tolist()

    # Preprocess the data - Take care of missing values
    processed_training_data = preprocess(training_data)
    processed_validation_data = preprocess(validation_data)
    processed_test_data = preprocess(test_data)

    # Creating a dataframe ofthe training data for ease of coding
    df_training = pd.DataFrame(processed_training_data,
                               columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12',
                                        'A13', 'A14', 'A15', 'target'])

    # All the names of the attributes in the data
    attributes = {'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12',
                                        'A13', 'A14', 'A15'}

    # Create the decision tree
    decision_tree = buildTree(df_training, attributes)

    #print(decision_tree)
    training_accuracy_depth = []
    validation_accuracy_depth = []
    test_accuracy_depth = []


    columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12',
               'A13', 'A14', 'A15', 'target']

    # Data for classification of training data
    df_training_accuracy = pd.DataFrame(processed_training_data, columns=columns)
    df_validation_accuracy = pd.DataFrame(processed_validation_data, columns=columns)
    df_test_accuracy = pd.DataFrame(processed_test_data, columns=columns)

    for depth in range(1,16):

        training_labels = []
        training_preds = []
        validation_labels = []
        validation_preds = []
        test_labels = []
        test_preds = []

        for row in range(len(df_training_accuracy)):
            inputData = {}
            for items in columns:
                inputData[items] = df_training_accuracy[items][row]
            training_labels.append(df_training_accuracy['target'][row])
            training_preds.append(predict(decision_tree.copy(), inputData, depth, df_training_accuracy.copy(deep=True)))

        for row in range(len(df_validation_accuracy)):
            inputData = {}
            for items in columns:
                inputData[items] = df_validation_accuracy[items][row]
            validation_labels.append(df_validation_accuracy['target'][row])
            validation_preds.append(predict(decision_tree.copy(), inputData, depth, df_validation_accuracy.copy(deep=True)))

        for row in range(len(df_test_accuracy)):
            inputData = {}
            for items in columns:
                inputData[items] = df_test_accuracy[items][row]
            test_labels.append(df_test_accuracy['target'][row])
            test_preds.append(predict(decision_tree.copy(), inputData, depth, df_test_accuracy.copy(deep=True)))

        # print(len(validation_labels), len(validation_preds))
        # print(len(test_labels), len(test_preds))
        # print(len(training_labels), len(training_preds))
        training_accuracy, validation_accuracy, test_accuracy = eval(training_labels, training_preds, validation_labels, validation_preds, test_labels, test_preds)

        training_accuracy_depth.append(training_accuracy)
        validation_accuracy_depth.append(validation_accuracy)
        test_accuracy_depth.append(test_accuracy)

    print('Training accuracies:')
    print(training_accuracy_depth)
    print("Validation acccuracies:")
    print(validation_accuracy_depth)
    print("Test accuracies")
    print(test_accuracy_depth)

    x_axis = [x for x in range(1, 16)]
    fig = plt.figure(figsize=(16, 12), dpi=80)
    ax = plt.axes()
    ax.set(xlabel='Depth', ylabel='Accuracy',
           title='Train, Validation, Test accuracy vs Depth of Decision Tree');

    plt.plot(x_axis, training_accuracy_depth, label="Training")
    plt.plot(x_axis, validation_accuracy_depth, label="Validation")
    plt.plot(x_axis, test_accuracy_depth, label="Test");
    plt.legend()

# Used to output the accuracies of train, test, validation at all maxDepthValues
#DecisionTreeDepthFind()