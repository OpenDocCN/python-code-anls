# `D:\src\scipysrc\scikit-learn\examples\svm\plot_linearsvc_support_vectors.py`

```
"""
=====================================
Plot the support vectors in LinearSVC
=====================================

Unlike SVC (based on LIBSVM), LinearSVC (based on LIBLINEAR) does not provide
the support vectors. This example demonstrates how to obtain the support
vectors in LinearSVC.

"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

from sklearn.datasets import make_blobs  # 导入 make_blobs 数据集生成器，用于生成样本数据
from sklearn.inspection import DecisionBoundaryDisplay  # 导入 DecisionBoundaryDisplay 类，用于绘制决策边界
from sklearn.svm import LinearSVC  # 导入 LinearSVC 类，用于支持向量机分类

X, y = make_blobs(n_samples=40, centers=2, random_state=0)  # 生成含有两个簇的样本数据 X 和对应的标签 y

plt.figure(figsize=(10, 5))  # 创建图形窗口，设置大小为 10x5

for i, C in enumerate([1, 100]):
    # "hinge" is the standard SVM loss
    clf = LinearSVC(C=C, loss="hinge", random_state=42).fit(X, y)
    # obtain the support vectors through the decision function
    decision_function = clf.decision_function(X)
    # we can also calculate the decision function manually
    # decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
    # The support vectors are the samples that lie within the margin
    # boundaries, whose size is conventionally constrained to 1
    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    support_vectors = X[support_vector_indices]

    plt.subplot(1, 2, i + 1)  # 创建子图，1 行 2 列，当前是第 i+1 个子图
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)  # 绘制散点图表示样本点
    ax = plt.gca()  # 获取当前子图的坐标轴
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        ax=ax,
        grid_resolution=50,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )
    plt.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )  # 标记支持向量，以不填充的黑色圆圈表示
    plt.title("C=" + str(C))  # 设置子图标题，显示当前 C 值
plt.tight_layout()  # 调整子图布局，防止重叠
plt.show()  # 显示图形
```