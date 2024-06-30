# `D:\src\scipysrc\scikit-learn\examples\inspection\plot_permutation_importance_multicollinear.py`

```
"""
=================================================================
Permutation Importance with Multicollinear or Correlated Features
=================================================================

In this example, we compute the
:func:`~sklearn.inspection.permutation_importance` of the features to a trained
:class:`~sklearn.ensemble.RandomForestClassifier` using the
:ref:`breast_cancer_dataset`. The model can easily get about 97% accuracy on a
test dataset. Because this dataset contains multicollinear features, the
permutation importance shows that none of the features are important, in
contradiction with the high test accuracy.

We demo a possible approach to handling multicollinearity, which consists of
hierarchical clustering on the features' Spearman rank-order correlations,
picking a threshold, and keeping a single feature from each cluster.

.. note::
    See also
    :ref:`sphx_glr_auto_examples_inspection_plot_permutation_importance.py`

"""

# %%
# Random Forest Feature Importance on Breast Cancer Data
# ------------------------------------------------------
#
# First, we define a function to ease the plotting:
from sklearn.inspection import permutation_importance


def plot_permutation_importance(clf, X, y, ax):
    # Calculate permutation importance using the RandomForestClassifier 'clf'
    # on dataset X with labels y, repeating 10 times, using 2 parallel jobs
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=2)
    # Sort features by their mean importance across permutations
    perm_sorted_idx = result.importances_mean.argsort()

    # Create a boxplot of the permutation importances
    ax.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,  # Plot horizontal boxplot
        labels=X.columns[perm_sorted_idx],  # Label features on y-axis
    )
    # Add a dashed line at x=0 for reference
    ax.axvline(x=0, color="k", linestyle="--")
    return ax


# %%
# We then train a :class:`~sklearn.ensemble.RandomForestClassifier` on the
# :ref:`breast_cancer_dataset` and evaluate its accuracy on a test set:
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load breast cancer dataset and split into training and test sets
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Initialize a RandomForestClassifier with 100 trees
clf = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the classifier on the training data
clf.fit(X_train, y_train)
# Print the baseline accuracy of the classifier on the test set
print(f"Baseline accuracy on test data: {clf.score(X_test, y_test):.2}")

# %%
# Next, we plot the tree based feature importance and the permutation
# importance. The permutation importance is calculated on the training set to
# show how much the model relies on each feature during training.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Compute Mean Decrease in Impurity (MDI) based feature importances
mdi_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
# Sort features based on their MDI importance
tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
# Generate indices for plotting
tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5

# Create a figure with two subplots for feature importance comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
# Plot bar chart of MDI importances
mdi_importances.sort_values().plot.barh(ax=ax1)
ax1.set_xlabel("Gini importance")  # Set x-axis label

# Plot permutation importances using the defined function
plot_permutation_importance(clf, X_train, y_train, ax2)
ax2.set_xlabel("Decrease in accuracy score")  # Set x-axis label

# Set the figure title
fig.suptitle(
    "Comparing Feature Importances",
)
    "Impurity-based vs. permutation importances on multicollinear features (train set)"



# 分析基于不纯度和排列重要性在多重共线特征上的表现（训练集）
# 这行代码是一个字符串文字，描述了进行的分析内容和数据集。
# 在代码中没有实际功能作用，仅作为一个描述性的文本。
# %%
# The plot on the left shows the Gini importance of the model. As the
# scikit-learn implementation of
# :class:`~sklearn.ensemble.RandomForestClassifier` uses a random subsets of
# :math:`\sqrt{n_\text{features}}` features at each split, it is able to dilute
# the dominance of any single correlated feature. As a result, the individual
# feature importance may be distributed more evenly among the correlated
# features. Since the features have large cardinality and the classifier is
# non-overfitted, we can relatively trust those values.
#
# The permutation importance on the right plot shows that permuting a feature
# drops the accuracy by at most `0.012`, which would suggest that none of the
# features are important. This is in contradiction with the high test accuracy
# computed as baseline: some feature must be important.
#
# Similarly, the change in accuracy score computed on the test set appears to be
# driven by chance:
fig, ax = plt.subplots(figsize=(7, 6))
plot_permutation_importance(clf, X_test, y_test, ax)
ax.set_title("Permutation Importances on multicollinear features\n(test set)")
ax.set_xlabel("Decrease in accuracy score")
_ = ax.figure.tight_layout()

# %%
# Nevertheless, one can still compute a meaningful permutation importance in the
# presence of correlated features, as demonstrated in the following section.
#
# Handling Multicollinear Features
# --------------------------------
# When features are collinear, permuting one feature has little effect on the
# models performance because it can get the same information from a correlated
# feature. Note that this is not the case for all predictive models and depends
# on their underlying implementation.
#
# One way to handle multicollinear features is by performing hierarchical
# clustering on the Spearman rank-order correlations, picking a threshold, and
# keeping a single feature from each cluster. First, we plot a heatmap of the
# correlated features:
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
# Calculate Spearman correlation matrix for features X
corr = spearmanr(X).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# Convert correlation matrix to a distance matrix for hierarchical clustering
distance_matrix = 1 - np.abs(corr)
# Perform hierarchical clustering using Ward's linkage
dist_linkage = hierarchy.ward(squareform(distance_matrix))
# Plot dendrogram of the hierarchical clustering
dendro = hierarchy.dendrogram(
    dist_linkage, labels=X.columns.to_list(), ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

# Plot heatmap of the correlated features
ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
_ = fig.tight_layout()

# %%
# Next, we manually pick a threshold by visual inspection of the dendrogram to
# 根据层次聚类的结果将特征分组，并从每个组中选择一个特征保留，然后从数据集中选择这些特征，训练一个新的随机森林模型。
# 新随机森林模型在测试数据上的准确率与在完整数据集上训练的随机森林模型相比变化不大。
from collections import defaultdict

# 使用层次聚类将特征分组，并返回每个特征组对应的特征索引列表
cluster_ids = hierarchy.fcluster(dist_linkage, 1, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)

# 从每个特征组中选择第一个特征的索引，形成选定的特征索引列表
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
# 根据选定的特征索引列表获取对应的特征名称
selected_features_names = X.columns[selected_features]

# 根据选定的特征名称从训练集和测试集中提取相应的数据子集
X_train_sel = X_train[selected_features_names]
X_test_sel = X_test[selected_features_names]

# 使用随机森林分类器进行模型训练
clf_sel = RandomForestClassifier(n_estimators=100, random_state=42)
clf_sel.fit(X_train_sel, y_train)

# 打印在移除特征后的测试数据上的基准准确率
print(
    "Baseline accuracy on test data with features removed:"
    f" {clf_sel.score(X_test_sel, y_test):.2}"
)

# %%
# 最后，我们可以探索选定特征子集的排列重要性：
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 6))
# 绘制选定特征子集在测试集上的排列重要性
plot_permutation_importance(clf_sel, X_test_sel, y_test, ax)
ax.set_title("Permutation Importances on selected subset of features\n(test set)")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
plt.show()
```