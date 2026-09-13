# Decision Tree and Random Forest from Scratch

Implementation of a decision tree classifier built from scratch (no `sklearn.tree` internals
used), plus a random forest built on top of it using bagging and random feature subsets.

# Project structure


DecisionTreeScratch/
├── decisionTree.ipynb     # Main notebook — decision tree implementation, results, and experiments
├── decisionTree.py         # Same decision tree code as the notebook, exported as a plain .py
│                           # file purely so randomForest.ipynb can import DecisionTree from it
├── randomForest.ipynb      # Random forest implementation, built on top of decisionTree.py
└── writeup.md               # Write-up covering theory, results, and discussion


# Note on the `.py` file

`decisionTree.py` is not a separate implementation — it's the same code as `decisionTree.ipynb`,
exported to a plain Python file. This exists only so that `randomForest.ipynb` can do
`from decisionTree import DecisionTree` and reuse the tree implementation directly, since
importing a class from one notebook into another isn't straightforward in plain Python.

**All actual results, outputs, and experiments should be checked in the `.ipynb` files** —
that's where the tree is trained, tested, and printed with real numbers.

# Where to find each deliverable

| Item | Description | Where to check |
|------|-------------|-----------------|
| 4.1 | Decision tree with Gini impurity | `decisionTree.ipynb` |
| 4.2 | Classification-only justification | `decisionTree.ipynb` / `writeup.md` |
| 4.3 | Overfitting demonstration | `decisionTree.ipynb` |
| 4.4 | Regularization (max_depth) | `decisionTree.ipynb` |
| 4.5 | Random forest (bagging + feature subsets) | `randomForest.ipynb` |
| 4.6 | Comparison against sklearn's DecisionTreeClassifier | `decisionTree.ipynb` |
| 4.7 | Single tree vs random forest comparison | `randomForest.ipynb` |
| 4.8 | Why bagging reduces variance (discussion) | `writeup.md`, Section 9 |

# How to run

Open `decisionTree.ipynb` in Jupyter (or VS Code) and run all cells top to bottom — this covers
4.1 through 4.4 and 4.6.

Then open `randomForest.ipynb` and run all cells — this covers 4.5 and 4.7. It imports
`DecisionTree` from `decisionTree.py`, so make sure that file is present in the same folder.

# Dataset

Using sklearn's Iris dataset for the core implementation and correctness checks, and a
synthetic noisy dataset (via `sklearn.datasets.make_classification`, with `flip_y` label
noise) to demonstrate overfitting and the benefit of random forest's variance reduction.

# Requirements

bash
pip install numpy scikit-learn
