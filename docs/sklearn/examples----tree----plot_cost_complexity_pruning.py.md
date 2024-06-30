# `D:\src\scipysrc\scikit-learn\examples\tree\plot_cost_complexity_pruning.py`

```
"""
========================================================
Post pruning decision trees with cost complexity pruning
========================================================

.. currentmodule:: sklearn.tree

The :class:`DecisionTreeClassifier` provides parameters such as
``min_samples_leaf`` and ``max_depth`` to prevent a tree from overfiting. Cost
complexity pruning provides another option to control the size of a tree. In
:class:`DecisionTreeClassifier`, this pruning technique is parameterized by the
cost complexity parameter, ``ccp_alpha``. Greater values of ``ccp_alpha``
increase the number of nodes pruned. Here we only show the effect of
``ccp_alpha`` on regularizing the trees and how to choose a ``ccp_alpha``
based on validation scores.

See also :ref:`minimal_cost_complexity_pruning` for details on pruning.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 库，用于绘图

from sklearn.datasets import load_breast_cancer  # 导入乳腺癌数据集加载函数
from sklearn.model_selection import train_test_split  # 导入数据集划分函数
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器类

# %%
# Total impurity of leaves vs effective alphas of pruned tree
# ---------------------------------------------------------------
# Minimal cost complexity pruning recursively finds the node with the "weakest
# link". The weakest link is characterized by an effective alpha, where the
# nodes with the smallest effective alpha are pruned first. To get an idea of
# what values of ``ccp_alpha`` could be appropriate, scikit-learn provides
# :func:`DecisionTreeClassifier.cost_complexity_pruning_path` that returns the
# effective alphas and the corresponding total leaf impurities at each step of
# the pruning process. As alpha increases, more of the tree is pruned, which
# increases the total impurity of its leaves.
X, y = load_breast_cancer(return_X_y=True)  # 加载乳腺癌数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)  # 划分数据集为训练集和测试集

clf = DecisionTreeClassifier(random_state=0)  # 初始化决策树分类器对象
path = clf.cost_complexity_pruning_path(X_train, y_train)  # 计算训练集上的成本复杂度剪枝路径
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# %%
# In the following plot, the maximum effective alpha value is removed, because
# it is the trivial tree with only one node.
fig, ax = plt.subplots()  # 创建一个新的图形和子图
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")  # 绘制曲线图，展示不同 alpha 对应的总不纯度
ax.set_xlabel("effective alpha")  # 设置 x 轴标签
ax.set_ylabel("total impurity of leaves")  # 设置 y 轴标签
ax.set_title("Total Impurity vs effective alpha for training set")  # 设置图标题

# %%
# Next, we train a decision tree using the effective alphas. The last value
# in ``ccp_alphas`` is the alpha value that prunes the whole tree,
# leaving the tree, ``clfs[-1]``, with one node.
clfs = []  # 初始化一个空列表，用于存储不同 alpha 值下的决策树分类器
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)  # 使用当前 alpha 值初始化决策树分类器
    clf.fit(X_train, y_train)  # 在训练集上训练分类器
    clfs.append(clf)  # 将训练好的分类器加入列表
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)

# %%
# For the remainder of this example, we remove the last element in
# 截取列表 ``clfs`` 和 ``ccp_alphas`` 中的所有元素，除了最后一个元素
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

# 获取每个决策树分类器（DecisionTreeClassifier）的节点数
node_counts = [clf.tree_.node_count for clf in clfs]
# 获取每个决策树分类器的最大深度
depth = [clf.tree_.max_depth for clf in clfs]

# 创建包含两个子图的图表对象，用于绘制节点数和 alpha 的关系图
fig, ax = plt.subplots(2, 1)
# 绘制节点数随 alpha 变化的图像
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")

# 绘制树深度随 alpha 变化的图像
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")

# 调整子图布局，使其紧凑显示
fig.tight_layout()

# %%
# 训练集和测试集的准确率随 alpha 变化的图像
# ----------------------------------------------------
# 当 ``ccp_alpha`` 设置为零，并保持 :class:`DecisionTreeClassifier` 的其他默认参数时，
# 决策树会过拟合，导致训练准确率达到100%，测试准确率为88%。随着 alpha 的增加，
# 决策树的剪枝增多，生成的决策树更好地泛化。在本例中，设置 ``ccp_alpha=0.015`` 可以
# 最大化测试准确率。
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

# 创建图表对象
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")

# 绘制训练集和测试集准确率随 alpha 变化的图像
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()

# 显示图表
plt.show()
```