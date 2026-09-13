Write-up: Decision Tree and Random Forest from Scratch

1. What I built

I implemented a decision tree classifier from scratch using Gini impurity as the splitting
criterion, trained and evaluated on the Iris dataset. I focused on classification only and
did not implement regression — the reasoning is below.

Why classification only (not regression)

Although a decision tree can be adapted for regression, doing so isn't just a matter of
reusing the same code with a different dataset — it requires changing two core pieces of the
algorithm:

1. The splitting criterion: Gini impurity and entropy are both built around measuring how
   mixed a set of class labels is. They don't make sense for continuous targets. Regression
   trees instead typically minimize variance (or MSE) within each child node — a completely
   different formula, even though the "try every feature and threshold, pick the best split"
   search structure stays similar.

2. The leaf prediction. In my classifier, a leaf predicts the majority class among its
   samples ('_majority_class'). A regression tree's leaf instead predicts the mean of the
   target values in that node — again, a different aggregation.

Given the time available, I chose to implement classification fully and well — correct Gini
impurity, a working best-split search, proper recursive stopping conditions, and a working
prediction path — rather than splitting effort across both and risking a shallow, buggy
version of each.

2. Impurity measure and split selection

I used Gini impurity, defined for a set of labels as:

Gini(node) = 1 - Σ (p_i)^2


where 'p_i' is the proportion of samples belonging to class i in that node. A pure node
(all one class) has Gini = 0; a node evenly split between classes has higher Gini (e.g. 0.5
for a 50/50 binary split). I chose Gini over entropy mainly for simplicity — it avoids a log
computation and is the more common default (it's also sklearn's default), and for this task
the choice between the two rarely changes which split gets picked in practice.

Finding the best split ('best_split' in 'decision_tree.py') works by brute force:

- Loop over every feature.
- For each feature, take the sorted unique values and compute the midpoints between every
  pair of consecutive values — these are the candidate thresholds.
- For each candidate threshold, split the samples into a left group ('feature <= threshold')
  and a right group ('feature > threshold'), and compute the weighted Gini of the split:

weighted_gini = (n_left / n_total) * Gini(left) + (n_right / n_total) * Gini(right)

- Information gain is then 'parent_gini - weighted_gini'. The split with the highest gain
  across all features and thresholds is chosen.

This is O(n_features × n_samples log n_samples) per node (due to sorting/threshold search),
which is fine for a small dataset like Iris but would need optimization (e.g. presorting, or
histogram-based binning as real libraries do) to scale to larger datasets.

3. Recursive tree building and stopping conditions

The tree is built recursively ('_build_tree'), starting at the root with the full training
set at depth 0. At each call, the function first checks whether to stop and create a leaf:

- The node is already pure (only one class present) → leaf predicts that class directly.
- 'max_depth' has been reached (if set) → leaf predicts the majority class.
- Fewer samples remain than 'min_samples_split' → leaf predicts the majority class.
- 'best_split' couldn't find any split with positive information gain → leaf predicts the
  majority class.

If none of these apply, the function calls 'best_split' to get the best (feature, threshold)
pair, partitions the data into left/right subsets using that split, and recurses on each
subset independently with 'depth + 1'. The two resulting subtrees are attached as 'left'
and 'right' on a new internal 'Node', which is returned up the call stack. This recursive
structure is what turns a single split into a full tree — each subtree is built exactly the
same way as the whole tree, just on a smaller slice of the data.

Prediction ('_predict_one') mirrors this recursion: starting at the root, it checks
'x[feature_index] <= threshold' and recurses into 'left' or 'right' accordingly, until it
hits a leaf and returns its stored class.

Verification so far: trained with 'max_depth=3' on Iris (80/20 train/test split), the
tree achieves 95.8% train accuracy and 100% test accuracy, and the printed tree structure
(via 'print_tree()') matches domain intuition — it splits first on petal length, then petal
width, which lines up with known facts about how Iris species are visually distinguished.

4. Overfitting demonstration (Item 4.3)

Used sklearn's make_classification to create noisy dataset, which is used to train our algorithm
with max depth set to None. The results show that on testing data, the accuracy is only 63.3%


5. Regularization (Item 4.4)

To check the importance of setting a stopping condition, I checked the model on the same noisy dataset 
once with max_depth set to None and in one with max_depth set to 3. When we check the gap between the test
and train data, to compare consistency, we can see that the model is more consistent in the second case 
when we have set the max_depth to 3 ( gap reduces to 0.20 from 0.36). This shows that regularization reduces
overfitting.

6. Random forest (Item 4.5) 
    1. Changing best_split() and DecisionTree to include an parameter max_features, so it can be used to pick a random subset for looping over feature_indices instead of all the columns to implement random forest
    2. Created decisionTree.py so it is easier to import ( just copied the code of decisionTree.ipynb)
    3. Wrote the code for randomForest, which is basically multiple decisionTree running along random splits of the same data ( bootstrap sampling ). This gives us series of outputs, which we take majority vote out of to return the best possible output.
    4. Tested the randomForest code




7. Comparison against sklearn (Item 4.6) 
Not yet started
8. Single tree vs random forest (Item 4.7) 
Not yet started
9. Why bagging reduces variance (Item 4.8) 
Not yet started 
10. Mistakes made and how I fixed them

11. What I'd do with more time

Given more time, the priority order would be: finish the overfitting/regularization
comparison (Sections 4-5, since these require no new code beyond re-running with different
hyperparameters), then attempt the random forest (Section 6) since it builds directly on the
already-working decision tree, then the sklearn comparison and single-tree-vs-forest analysis
(Sections 7-8) using the correctness harness already scaffolded.