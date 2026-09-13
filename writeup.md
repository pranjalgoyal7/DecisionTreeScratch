Write-up: Decision Tree and Random Forest from Scratch

1. What I built
I built a decision tree classifier from scratch using Gini impurity, trained on the Iris dataset. I focused on classification only, not regression.
Why not regression? Gini/entropy measure how mixed class labels are — they don't apply to continuous targets. Regression trees need a different splitting rule (variance/MSE) and a different leaf prediction (mean instead of majority class). Rather than build both shallowly, I chose to implement classification properly.


2. Impurity measure and split selection
I used Gini impurity:
Gini(node) = 1 - sum(p_i)²
where p_i is the proportion of each class in a node. Pure node: Gini = 0; evenly mixed: higher Gini. I picked Gini over entropy for simplicity (no log, and it's sklearn's default) — in practice it rarely changes which split gets picked anyway.
To find the best split, I try every feature and every midpoint between sorted unique values as a candidate threshold, split the data left/right, and compute the weighted Gini of the split. The split with the highest information gain (parent_gini - weighted_gini) wins.

3. Recursive tree building and stopping conditions
The tree builds recursively, starting from the full dataset at depth 0. At each step, it stops and makes a leaf if: the node is pure, max_depth is reached, too few samples remain, or no split improves purity — otherwise it finds the best split, partitions the data, and recurses on each half.
Prediction follows the same recursive pattern: check the feature against the threshold, go left or right, repeat until hitting a leaf.
Result: with max_depth=3 on Iris, the tree hit 95.8% train / 100% test accuracy, and the printed tree split first on petal length then petal width — matching real biological intuition about Iris species.

4. Overfitting demonstration (Item 4.3)
I trained the tree with max_depth=None (fully grown, no restriction) on a noisy synthetic dataset generated with sklearn's make_classification (with flip_y injecting label noise). The unrestricted tree hit 100% train accuracy but only 63.3% test accuracy — a large gap, showing the tree memorized noise in the training data rather than learning patterns that generalize. This is the expected signature of overfitting: near-perfect performance on data it has seen, much weaker performance on unseen data.


5. Regularization (Item 4.4)
To check the importance of setting a stopping condition, I checked the model on the same noisy dataset 
once with max_depth set to None and in one with max_depth set to 3. When we check the gap between the test
and train data, to compare consistency, we can see that the model is more consistent in the second case 
when we have set the max_depth to 3 ( gap reduces to 0.20 from 0.36). This shows that regularization reduces
overfitting.

6. Random forest (Item 4.5) 
I extended best_split() and DecisionTree to accept a max_features parameter, so instead of searching every feature at each split, it randomly picks a subset — this decorrelates trees from each other, since they don't all latch onto the same strongest feature every time. The RandomForest class trains multiple DecisionTrees, each on its own bootstrap sample (rows sampled with replacement from the training set), and combines their individual predictions with a majority vote at prediction time. Both sources of randomness (bootstrapped rows, random feature subsets) work together to make the trees make different, largely independent mistakes

7. Comparison against sklearn (Item 4.6) 
I compared my single DecisionTree against sklearn.tree.DecisionTreeClassifier, using matching settings (max_depth=3, Gini criterion) on the Iris dataset. Both achieved 100% test accuracy, and a sample-by-sample comparison showed 100% prediction agreement — every single test sample got the same predicted class from both implementations. This strongly confirms my tree's splitting and prediction logic is correct.

8. Single tree vs random forest (Item 4.7) 
I compared a single tree against my random forest under two conditions: the clean Iris dataset and the noisy make_classification dataset used in Section 4. On Iris, the forest scored 68.3% test accuracy vs. 66.7% for the single tree. On the noisy dataset, the gap was more pronounced: the forest achieved 75% test accuracy compared to the single tree's 63.3%, showing the forest is noticeably more robust to noisy/overfitting-prone data. Across multiple random seeds, the forest's accuracy also varied less run-to-run than the single tree's, consistent with the variance-reduction argument in Section 9.

9. Why bagging reduces variance (Item 4.8) 
A single tree is very sensitive to its exact training data — small changes can produce a very different tree (high variance). Bagging trains many trees on different bootstrap samples and different random feature subsets, so each tree makes different, largely independent mistakes. Averaging/voting across trees cancels out these independent errors while reinforcing the genuine pattern all trees pick up on, giving a more stable overall prediction.

10. Mistakes made and how I fixed them
    1. Typo'd pip install skicit-learn instead of scikit-learn — fixed by correcting the package name.
    2. Initially assumed Item 4.6 ("compare against sklearn.tree.DecisionTreeClassifier") meant comparing the random forest, since it came after building the forest. Re-reading the item's exact wording clarified it meant the single tree, not the forest.
    3. Core algorithm (Gini, split search, recursion, prediction) worked correctly on first implementation, with no bugs to fix — a result of working through the math by hand before coding it.

11. What I'd do with more time
Probably try out the stretch task. Deepen my knowledge in decisionTrees and try out regression tree