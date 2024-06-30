# `D:\src\scipysrc\scikit-learn\examples\tree\plot_unveil_tree_structure.py`

```
"""
=========================================
Understanding the decision tree structure
=========================================

The decision tree structure can be analysed to gain further insight on the
relation between the features and the target to predict. In this example, we
show how to retrieve:

- the binary tree structure;
- the depth of each node and whether or not it's a leaf;
- the nodes that were reached by a sample using the ``decision_path`` method;
- the leaf that was reached by a sample using the apply method;
- the rules that were used to predict a sample;
- the decision path shared by a group of samples.

"""

import numpy as np
from matplotlib import pyplot as plt

from sklearn import tree  # 导入决策树模块
from sklearn.datasets import load_iris  # 导入鸢尾花数据集
from sklearn.model_selection import train_test_split  # 导入数据集分割函数
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器模型

##############################################################################
# Train tree classifier
# ---------------------
# First, we fit a :class:`~sklearn.tree.DecisionTreeClassifier` using the
# :func:`~sklearn.datasets.load_iris` dataset.

iris = load_iris()  # 载入鸢尾花数据集
X = iris.data  # 提取特征
y = iris.target  # 提取目标变量
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)  # 将数据集划分为训练集和测试集

clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)  # 初始化决策树分类器模型，限制最大叶子节点数为3
clf.fit(X_train, y_train)  # 在训练集上拟合决策树分类器模型

##############################################################################
# Tree structure
# --------------
#
# The decision classifier has an attribute called ``tree_`` which allows access
# to low level attributes such as ``node_count``, the total number of nodes,
# and ``max_depth``, the maximal depth of the tree. The
# ``tree_.compute_node_depths()`` method computes the depth of each node in the
# tree. `tree_` also stores the entire binary tree structure, represented as a
# number of parallel arrays. The i-th element of each array holds information
# about the node ``i``. Node 0 is the tree's root. Some of the arrays only
# apply to either leaves or split nodes. In this case the values of the nodes
# of the other type is arbitrary. For example, the arrays ``feature`` and
# ``threshold`` only apply to split nodes. The values for leaf nodes in these
# arrays are therefore arbitrary.
#
# Among these arrays, we have:
#
#   - ``children_left[i]``: id of the left child of node ``i`` or -1 if leaf
#     node
#   - ``children_right[i]``: id of the right child of node ``i`` or -1 if leaf
#     node
#   - ``feature[i]``: feature used for splitting node ``i``
#   - ``threshold[i]``: threshold value at node ``i``
#   - ``n_node_samples[i]``: the number of training samples reaching node
#     ``i``
#   - ``impurity[i]``: the impurity at node ``i``
#   - ``weighted_n_node_samples[i]``: the weighted number of training samples
#     reaching node ``i``
#   - ``value[i, j, k]``: the summary of the training samples that reached node i for
#     output j and class k (for regression tree, class is set to 1).
#
# 获得决策树模型中节点的总数
n_nodes = clf.tree_.node_count
# 获取每个节点的左子节点数组
children_left = clf.tree_.children_left
# 获取每个节点的右子节点数组
children_right = clf.tree_.children_right
# 获取每个节点的分裂特征数组，即决策树节点对应的特征索引
feature = clf.tree_.feature
# 获取每个节点的分裂阈值数组
threshold = clf.tree_.threshold
# 获取每个节点的值数组，表示节点中各类别样本的数量
values = clf.tree_.value

# 初始化节点深度数组，全为零，长度为节点总数
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
# 初始化节点是否为叶子节点的布尔数组，全为False，长度为节点总数
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
# 使用栈来迭代地遍历树结构，初始根节点为(0, 0)，深度为0
stack = [(0, 0)]

# 迭代遍历树结构直到栈为空
while len(stack) > 0:
    # 弹出栈顶节点，并获取节点ID和深度
    node_id, depth = stack.pop()
    # 将节点深度记录在node_depth数组中
    node_depth[node_id] = depth

    # 判断当前节点是否为分裂节点
    is_split_node = children_left[node_id] != children_right[node_id]
    # 如果是分裂节点，则将左右子节点和深度压入栈中
    if is_split_node:
        stack.append((children_left[node_id], depth + 1))
        stack.append((children_right[node_id], depth + 1))
    else:
        # 如果是叶子节点，则将is_leaves数组对应位置置为True
        is_leaves[node_id] = True

# 打印树结构信息的总节点数
print(
    "The binary tree structure has {n} nodes and has "
    "the following tree structure:\n".format(n=n_nodes)
)
# 遍历每个节点，根据节点类型打印节点信息
for i in range(n_nodes):
    if is_leaves[i]:
        # 如果是叶子节点，打印叶子节点的信息
        print(
            "{space}node={node} is a leaf node with value={value}.".format(
                space=node_depth[i] * "\t", node=i, value=values[i]
            )
        )
    else:
        # 如果是分裂节点，打印分裂节点的信息
        print(
            "{space}node={node} is a split node with value={value}: "
            "go to node {left} if X[:, {feature}] <= {threshold} "
            "else to node {right}.".format(
                space=node_depth[i] * "\t",
                node=i,
                left=children_left[i],
                feature=feature[i],
                threshold=threshold[i],
                right=children_right[i],
                value=values[i],
            )
        )
##############################################################################
# We can compare the above output to the plot of the decision tree.
# 将上述输出与决策树的绘图进行比较。

tree.plot_tree(clf)
plt.show()

##############################################################################
# Decision path
# -------------
#
# We can also retrieve the decision path of samples of interest. The
# ``decision_path`` method outputs an indicator matrix that allows us to
# retrieve the nodes the samples of interest traverse through. A non zero
# element in the indicator matrix at position ``(i, j)`` indicates that
# the sample ``i`` goes through the node ``j``. Or, for one sample ``i``, the
# positions of the non zero elements in row ``i`` of the indicator matrix
# designate the ids of the nodes that sample goes through.
#
# The leaf ids reached by samples of interest can be obtained with the
# ``apply`` method. This returns an array of the node ids of the leaves
# reached by each sample of interest. Using the leaf ids and the
# ``decision_path`` we can obtain the splitting conditions that were used to
# predict a sample or a group of samples. First, let's do it for one sample.
# Note that ``node_index`` is a sparse matrix.
#
# 获取样本感兴趣的决策路径。
# ``decision_path`` 方法输出一个指示器矩阵，允许我们获取样本穿过的节点。
# 指示器矩阵中在 ``(i, j)`` 位置的非零元素表示样本 ``i`` 经过节点 ``j``。
# 或者对于一个样本 ``i``，指示器矩阵第 ``i`` 行中非零元素的位置指示样本经过的节点 ids。
#
# 样本感兴趣达到的叶节点 ids 可以通过 ``apply`` 方法获得。
# 这返回一个数组，包含每个感兴趣样本达到的叶节点 ids。
# 利用叶节点 ids 和 ``decision_path`` 方法，我们可以获取用于预测一个样本或一组样本的分裂条件。
# 首先，我们来处理一个样本。注意，``node_index`` 是一个稀疏矩阵。

node_indicator = clf.decision_path(X_test)
leaf_id = clf.apply(X_test)

sample_id = 0
# obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
node_index = node_indicator.indices[
    node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
]

print("Rules used to predict sample {id}:\n".format(id=sample_id))
for node_id in node_index:
    # continue to the next node if it is a leaf node
    # 如果是叶节点，则继续到下一个节点
    if leaf_id[sample_id] == node_id:
        continue

    # check if value of the split feature for sample 0 is below threshold
    # 检查样本 0 的分裂特征值是否低于阈值
    if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
        threshold_sign = "<="
    else:
        threshold_sign = ">"

    print(
        "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
        "{inequality} {threshold})".format(
            node=node_id,
            sample=sample_id,
            feature=feature[node_id],
            value=X_test[sample_id, feature[node_id]],
            inequality=threshold_sign,
            threshold=threshold[node_id],
        )
    )

##############################################################################
# For a group of samples, we can determine the common nodes the samples go
# through.
#
# 对于一组样本，我们可以确定这些样本共同经过的节点。

sample_ids = [0, 1]
# boolean array indicating the nodes both samples go through
# 布尔数组指示两个样本都经过的节点
common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids)
# obtain node ids using position in array
# 使用数组中的位置获取节点 ids
common_node_id = np.arange(n_nodes)[common_nodes]

print(
    "\nThe following samples {samples} share the node(s) {nodes} in the tree.".format(
        samples=sample_ids, nodes=common_node_id
    )
)
print("This is {prop}% of all nodes.".format(prop=100 * len(common_node_id) / n_nodes))
```