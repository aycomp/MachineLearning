from pprint import pprint
import numpy as np
import pandas as pd
import pydot

train_file_name = "datasets/dt_dataset/train.txt"
test_file_name = "datasets/dt_dataset/test.txt"

def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]

    data_below = data[split_column_values == split_value]
    data_above = data[split_column_values != split_value]

    return data_below, data_above

def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):  # excluding the last column which is the label
        values = data[:, column_index]
        unique_values = np.unique(values)

        if len(unique_values) > 1:
            potential_splits[column_index] = unique_values

    return potential_splits

def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy

def calculate_overall_entropy(data_below, data_above):
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy = (p_data_below * calculate_entropy(data_below)
                       + p_data_above * calculate_entropy(data_above))

    return overall_entropy

def determine_best_split(data, potential_splits):
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value

def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=10):
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df

    # base cases
    if (len(data) < min_samples) or (counter == max_depth):
        label_column = data[:, -1]
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

        index = counts_unique_classes.argmax()
        classification = unique_classes[index]

        return classification


    # recursive part
    else:
        counter += 1

        # helper functions
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)

        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        question = "{} = {}".format(feature_name, split_value)

        # instantiate sub-tree
        sub_tree = {question: []}

        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter)
        no_answer = decision_tree_algorithm(data_above, counter)

        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree

def classify_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if str(example[feature_name]) == value:
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer

    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)

def calculate_accuracy(df, tree):

    df["classification"] = df.apply(classify_example, args=(tree,), axis=1)
    df["classification_correct"] = df["classification"] == df["label"]

    accuracy = df["classification_correct"].mean()

    return accuracy

def draw(parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name)
    graph.add_edge(edge)

def visit(node, parent=None):
    for k,v in node.items():
        if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                draw(parent, k)
            visit(v, k)
        else:
            draw(parent, k)
            # drawing the label using a distinct name
            draw(k, str(k) + "_" + str(v))

if __name__ == '__main__':
    print('program started...')

    #reading training file
    df_train = pd.read_csv(train_file_name, sep=" ", header=None)
    df_train = df_train.rename(columns={0: "buying", 1: "maint", 2: "doors", 3: "persons", 4: "lug_boot", 5: "safety", 6: "label"})


    #ds_train = df_train.values
    #print(ds_train[:-1])

    #reading test file
    df_test = pd.read_csv(test_file_name, sep=" ", header=None)
    df_test = df_test.rename(columns={0: "buying", 1: "maint", 2: "doors", 3: "persons", 4: "lug_boot", 5: "safety", 6: "label"})

    tree = decision_tree_algorithm(df_train, max_depth=10)
    print(tree)
    print('program finished...')