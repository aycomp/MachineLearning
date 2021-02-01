import pandas as pd
import numpy as np
import pprint

train_file_path = "datasets/dt_dataset/train.txt"
test_file_path = "datasets/dt_dataset/test.txt"

def entropy(df):
    label = df.keys()[-1]
    _, counts = np.unique(df[label], return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy

def entropy_attr(fd, attr):
    label = df.keys()[-1]
    _, counts_label = np.unique(df[label], return_counts=True)
    count_label = counts_label.sum()

    _, counts_attr = np.unique(df[attr], return_counts=True)

    probabilities = counts_attr / counts_attr.sum()

    entropy_attr = probabilities * -np.log2(probabilities)
    entropy_total = sum((counts_attr / count_label) * entropy_attr)

    return entropy_total

#returns minimum entropy, it means maximum information gain
def winner(df):
    attrs = df.keys()[:-1]
    dict_entropy = {}
    for attr in attrs: #'buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'
        e = entropy_attr(df, attr)
        dict_entropy[attr] = e
    minimum_entropy = min(dict_entropy, key=dict_entropy.get)
    return minimum_entropy

def buildTree(df, tree=None):
    node = winner(df)
    att_value = np.unique(df[node])

    if tree is None:
        tree={}
        tree[node] = {}

    for value in att_value:
        sub_table = df[df[node] == value].reset_index(drop=True)
        clValue, counts = np.unique(sub_table['label'], return_counts=True)

        if len(counts) == 1:  # Checking purity of subset
            tree[node][value] = clValue[0]
        else:
            tree[node][value] = buildTree(sub_table)  # Calling the function recursively

    return tree

if __name__ == "__main__":
    # read file with pandas
    df = pd.read_csv(train_file_path, delimiter=" ", header=None)
    df = df.rename(columns={0: "buying", 1: "maint", 2: "doors", 3: "persons", 4: "lug_boot", 5: "safety", 6: "label"})

    #convert pandas, into numpy
    #data = df.values

    #build tree
    pprint.pprint(buildTree(df))
