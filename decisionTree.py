import numpy as np

class Node:     #creating a class for the node of the tree
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
       self.feature_index = feature_index
       self.threshold = threshold
       self.left = left
       self.right = right
       self.value = value  # only set for leaf nodes, not for decision nodes: others are filled for decision nodes and value is left empty

    def is_leaf(self):
        return self.value is not None

def gini(y):
    #computing the gini function
    if len(y) == 0:
        return 0
    _,counts = np.unique(y, return_counts=True)
    prob = counts / len(y)
    return 1-np.sum(prob**2)

def best_split(X, y, max_features=None):
    n_samples, n_features = X.shape        
    parent_gini = gini(y)       # compare every

    best_gain = -1              # defining base values, so we can compare and find the best gain, feature, threshold
    best_feature = None         
    best_threshold = None
    if max_features is None:
        feature_indices = range(n_features)
    else:
        feature_indices = np.random.choice(n_features, size=max_features, replace=False)

    
    for feature_index in feature_indices:       # looping through all the columns(features) of the dataset
        thresholds = np.unique(X[:, feature_index]) #defining thresholds as the unique values of the feature column
        # midpoints between consecutive sorted unique values
        candidate_thresholds = (thresholds[:-1] + thresholds[1:]) / 2       # [1,2,3,4] -> [1,2,3] + [2,3,4] /2 = [1.5,2.5,3.5] -> midpoints between consecutive sorted unique values
        for threshold in candidate_thresholds:      #looping through midpoint values
            left_mask = X[:, feature_index] <= threshold    # keeping all the values that are less than or equal to that particular midpoint in the given column
            right_mask = ~left_mask         # inverting left_mask to get greater values

            y_left, y_right = y[left_mask], y[right_mask]       #left side me values with true, and right side me values with false

            if len(y_left) == 0 or len(y_right) == 0:
                continue  # skip splits that don't actually separate anything

            weighted_gini = (
                (len(y_left) / n_samples) * gini(y_left)
                + (len(y_right) / n_samples) * gini(y_right)
            )

            gain = parent_gini - weighted_gini          # only if weighted gini is less than parent gini, we have a gain(choosing the best gini)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold, best_gain

# from sklearn.datasets import load_iris
# X,y = load_iris(return_X_y=True)
# print("X", X)
# print("\n y is: ", y)
# print(best_split(X,y))

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Stopping conditions
        if n_classes == 1:
            return Node(value=y[0])  # pure node, no need to split further

        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=self._majority_class(y))

        if n_samples < self.min_samples_split:
            return Node(value=self._majority_class(y))

        # Try to find a split
        feature_index, threshold, gain = best_split(X, y, max_features=self.max_features)

        if feature_index is None or gain <= 0:
            return Node(value=self._majority_class(y))

        # Partition data and recurse
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_index=feature_index, threshold=threshold,
                    left=left_child, right=right_child)


    def _majority_class(self, y):
        """Return the most common label in y — used for leaf predictions."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

if __name__ == "__main__":  #helps us to use this file as a package or module in other files, using __main__ allows us to test this code without ruining its usage as a package
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tree = DecisionTree(max_depth=3)
    tree.fit(X_train, y_train)

    train_acc = tree.score(X_train, y_train)
    test_acc = tree.score(X_test, y_test)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")


