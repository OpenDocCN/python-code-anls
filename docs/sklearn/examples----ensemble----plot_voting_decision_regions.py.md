# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_voting_decision_regions.py`

```
"""
==================================================
Plot the decision boundaries of a VotingClassifier
==================================================

.. currentmodule:: sklearn

Plot the decision boundaries of a :class:`~ensemble.VotingClassifier` for two
features of the Iris dataset.

Plot the class probabilities of the first sample in a toy dataset predicted by
three different classifiers and averaged by the
:class:`~ensemble.VotingClassifier`.

First, three exemplary classifiers are initialized
(:class:`~tree.DecisionTreeClassifier`,
:class:`~neighbors.KNeighborsClassifier`, and :class:`~svm.SVC`) and used to
initialize a soft-voting :class:`~ensemble.VotingClassifier` with weights `[2,
1, 2]`, which means that the predicted probabilities of the
:class:`~tree.DecisionTreeClassifier` and :class:`~svm.SVC` each count 2 times
as much as the weights of the :class:`~neighbors.KNeighborsClassifier`
classifier when the averaged probability is calculated.

"""

from itertools import product  # 导入 itertools 库中的 product 函数，用于生成笛卡尔积

import matplotlib.pyplot as plt  # 导入 matplotlib 库的 pyplot 模块，并简写为 plt

from sklearn import datasets  # 导入 sklearn 库中的 datasets 模块
from sklearn.ensemble import VotingClassifier  # 导入 sklearn 库中的 ensemble 模块中的 VotingClassifier 类
from sklearn.inspection import DecisionBoundaryDisplay  # 导入 sklearn 库中的 inspection 模块中的 DecisionBoundaryDisplay 类
from sklearn.neighbors import KNeighborsClassifier  # 导入 sklearn 库中的 neighbors 模块中的 KNeighborsClassifier 类
from sklearn.svm import SVC  # 导入 sklearn 库中的 svm 模块中的 SVC 类
from sklearn.tree import DecisionTreeClassifier  # 导入 sklearn 库中的 tree 模块中的 DecisionTreeClassifier 类

# Loading some example data
iris = datasets.load_iris()  # 加载鸢尾花数据集
X = iris.data[:, [0, 2]]  # 选择数据集中的第一列和第三列作为特征数据
y = iris.target  # 将数据集中的标签赋值给 y

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)  # 初始化决策树分类器，限定最大深度为4
clf2 = KNeighborsClassifier(n_neighbors=7)  # 初始化 K 近邻分类器，设置邻居数为7
clf3 = SVC(gamma=0.1, kernel="rbf", probability=True)  # 初始化支持向量机分类器，设置 gamma 为0.1，核函数为径向基函数，启用概率估计
eclf = VotingClassifier(
    estimators=[("dt", clf1), ("knn", clf2), ("svc", clf3)],  # 构建投票分类器，传入三个基分类器，命名为 dt、knn、svc
    voting="soft",  # 使用软投票策略
    weights=[2, 1, 2],  # 指定每个分类器的权重，决策树和支持向量机的权重为2，K 近邻的权重为1
)

clf1.fit(X, y)  # 使用决策树分类器进行数据拟合
clf2.fit(X, y)  # 使用 K 近邻分类器进行数据拟合
clf3.fit(X, y)  # 使用支持向量机分类器进行数据拟合
eclf.fit(X, y)  # 使用投票分类器进行数据拟合

# Plotting decision regions
f, axarr = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(10, 8))  # 创建一个包含4个子图的画布，共享 x 轴和 y 轴，设置尺寸为10x8
for idx, clf, tt in zip(
    product([0, 1], [0, 1]),  # 遍历笛卡尔积的坐标索引
    [clf1, clf2, clf3, eclf],  # 遍历包含四个分类器的列表
    ["Decision Tree (depth=4)", "KNN (k=7)", "Kernel SVM", "Soft Voting"],  # 每个子图的标题
):
    DecisionBoundaryDisplay.from_estimator(
        clf, X, alpha=0.4, ax=axarr[idx[0], idx[1]], response_method="predict"
    )  # 使用 DecisionBoundaryDisplay 类绘制分类器的决策边界和数据点，透明度为0.4
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")  # 绘制散点图，显示数据点的分布和类别
    axarr[idx[0], idx[1]].set_title(tt)  # 设置子图的标题

plt.show()  # 显示图形
```