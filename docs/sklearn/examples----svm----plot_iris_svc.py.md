# `D:\src\scipysrc\scikit-learn\examples\svm\plot_iris_svc.py`

```
"""
==================================================
Plot different SVM classifiers in the iris dataset
==================================================

Comparison of different linear SVM classifiers on a 2D projection of the iris
dataset. We only consider the first 2 features of this dataset:

- Sepal length
- Sepal width

This example shows how to plot the decision surface for four SVM classifiers
with different kernels.

The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
different decision boundaries. This can be a consequence of the following
differences:

- ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
  regular hinge loss.

- ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
  reduction while ``SVC`` uses the One-vs-One multiclass reduction.

Both linear models have linear decision boundaries (intersecting hyperplanes)
while the non-linear kernel models (polynomial or Gaussian RBF) have more
flexible non-linear decision boundaries with shapes that depend on the kind of
kernel and its parameters.

.. NOTE:: while plotting the decision function of classifiers for toy 2D
   datasets can help get an intuitive understanding of their respective
   expressive power, be aware that those intuitions don't always generalize to
   more realistic high-dimensional problems.

"""

import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图

from sklearn import datasets, svm  # 导入 sklearn 中的数据集和支持向量机模型
from sklearn.inspection import DecisionBoundaryDisplay  # 导入决策边界显示模块

# import some data to play with
iris = datasets.load_iris()  # 载入鸢尾花数据集
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]  # 取数据集的前两个特征作为 X 数据
y = iris.target  # 标签 y 是鸢尾花的目标分类

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM 正则化参数
models = (
    svm.SVC(kernel="linear", C=C),  # 创建线性核 SVM 模型
    svm.LinearSVC(C=C, max_iter=10000),  # 创建线性核的 LinearSVC 模型
    svm.SVC(kernel="rbf", gamma=0.7, C=C),  # 创建 RBF 核 SVM 模型
    svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),  # 创建三次多项式核 SVM 模型
)
models = (clf.fit(X, y) for clf in models)  # 对每个模型进行数据拟合

# title for the plots
titles = (
    "SVC with linear kernel",  # 每个子图的标题，对应不同的 SVM 模型
    "LinearSVC (linear kernel)",
    "SVC with RBF kernel",
    "SVC with polynomial (degree 3) kernel",
)

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)  # 创建 2x2 的子图布局
plt.subplots_adjust(wspace=0.4, hspace=0.4)  # 调整子图间的水平和垂直间距

X0, X1 = X[:, 0], X[:, 1]  # 将数据集 X 的两个特征分别赋值给 X0 和 X1

for clf, title, ax in zip(models, titles, sub.flatten()):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel=iris.feature_names[0],
        ylabel=iris.feature_names[1],
    )
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")  # 绘制散点图
    ax.set_xticks(())  # 不显示 x 轴刻度
    ax.set_yticks(())  # 不显示 y 轴刻度
    ax.set_title(title)  # 设置子图标题

plt.show()  # 显示图形
```