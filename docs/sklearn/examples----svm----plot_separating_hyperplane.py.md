# `D:\src\scipysrc\scikit-learn\examples\svm\plot_separating_hyperplane.py`

```
"""
=========================================
SVM: Maximum margin separating hyperplane
=========================================

Plot the maximum margin separating hyperplane within a two-class
separable dataset using a Support Vector Machine classifier with
linear kernel.

"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 库用于绘图

from sklearn import svm  # 导入 SVM 模型
from sklearn.datasets import make_blobs  # 导入 make_blobs 数据生成器
from sklearn.inspection import DecisionBoundaryDisplay  # 导入 DecisionBoundaryDisplay 用于绘制决策边界

# 生成包含 40 个可分离点的数据集
X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# 创建 SVM 分类器对象，使用线性核函数，C 参数设置为 1000，不进行正则化以示例为目的
clf = svm.SVC(kernel="linear", C=1000)
clf.fit(X, y)  # 对数据进行拟合

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)  # 绘制散点图表示数据集

# 绘制决策函数
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
    ax=ax,
)

# 绘制支持向量
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.show()  # 显示图形
```