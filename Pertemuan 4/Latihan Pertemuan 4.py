import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# Implementasi Decision Tree dari scratch
# -----------------------------

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def information_gain(X_column, y, threshold):
    parent_entropy = entropy(y)

    left_idx = X_column <= threshold
    right_idx = X_column > threshold

    if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
        return 0

    n = len(y)
    n_left, n_right = len(y[left_idx]), len(y[right_idx])
    e_left, e_right = entropy(y[left_idx]), entropy(y[right_idx])

    child_entropy = (n_left/n) * e_left + (n_right/n) * e_right
    return parent_entropy - child_entropy

def best_split(X, y):
    best_gain = -1
    split_idx, split_threshold = None, None

    for feature_idx in range(X.shape[1]):
        X_column = X[:, feature_idx]
        thresholds = np.unique(X_column)

        for t in thresholds:
            gain = information_gain(X_column, y, t)
            if gain > best_gain:
                best_gain = gain
                split_idx = feature_idx
                split_threshold = t

    return split_idx, split_threshold

def build_tree(X, y, depth=0, max_depth=3):
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) == 1:
        return Node(value=classes[0])
    if depth >= max_depth:
        return Node(value=classes[np.argmax(counts)])

    feature_idx, threshold = best_split(X, y)
    if feature_idx is None:
        return Node(value=classes[np.argmax(counts)])

    left_idx = X[:, feature_idx] <= threshold
    right_idx = X[:, feature_idx] > threshold

    left = build_tree(X[left_idx], y[left_idx], depth+1, max_depth)
    right = build_tree(X[right_idx], y[right_idx], depth+1, max_depth)
    return Node(feature=feature_idx, threshold=threshold, left=left, right=right)

def predict_one(x, tree):
    if tree.value is not None:
        return tree.value
    if x[tree.feature] <= tree.threshold:
        return predict_one(x, tree.left)
    else:
        return predict_one(x, tree.right)

def predict(X, tree):
    return [predict_one(x, tree) for x in X]

# -----------------------------
# Dataset Iris
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Training dari scratch
tree = build_tree(X_train, y_train, max_depth=3)
y_pred_scratch = predict(X_test, tree)
acc_scratch = accuracy_score(y_test, y_pred_scratch)

print("Akurasi Decision Tree (scratch):", acc_scratch)

# -----------------------------
# Bandingkan dengan scikit-learn
# -----------------------------
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred_sklearn = dt.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

print("Akurasi Decision Tree (scikit-learn):", acc_sklearn)
