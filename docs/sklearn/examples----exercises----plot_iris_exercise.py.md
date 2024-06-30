# `D:\src\scipysrc\scikit-learn\examples\exercises\plot_iris_exercise.py`

```
"""
================================
SVM Exercise
================================

A tutorial exercise for using different SVM kernels.

This exercise is used in the :ref:`using_kernels_tut` part of the
:ref:`supervised_learning_tut` section of the :ref:`stat_learn_tut_index`.

"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库
import numpy as np

# 导入 datasets 和 svm 模块
from sklearn import datasets, svm

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data  # 特征数据
y = iris.target  # 类别标签

# 只选取两类数据进行二分类任务
X = X[y != 0, :2]
y = y[y != 0]

n_sample = len(X)  # 样本数

np.random.seed(0)
# 打乱数据顺序
order = np.random.permutation(n_sample)
X = X[order]
y = y[order].astype(float)

# 划分训练集和测试集
X_train = X[: int(0.9 * n_sample)]
y_train = y[: int(0.9 * n_sample)]
X_test = X[int(0.9 * n_sample) :]
y_test = y[int(0.9 * n_sample) :]

# 用不同的核函数训练模型
for kernel in ("linear", "rbf", "poly"):
    # 创建 SVM 分类器对象
    clf = svm.SVC(kernel=kernel, gamma=10)
    # 在训练集上训练模型
    clf.fit(X_train, y_train)

    # 创建新的图形窗口
    plt.figure()
    # 清除当前图形所有轴
    plt.clf()
    # 绘制训练数据的散点图
    plt.scatter(
        X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired, edgecolor="k", s=20
    )

    # 标出测试数据
    plt.scatter(
        X_test[:, 0], X_test[:, 1], s=80, facecolors="none", zorder=10, edgecolor="k"
    )

    plt.axis("tight")
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    # 生成坐标网格
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    # 计算决策函数的值
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # 将结果绘制为颜色图
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    # 绘制决策边界
    plt.contour(
        XX,
        YY,
        Z,
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
        levels=[-0.5, 0, 0.5],
    )

    plt.title(kernel)  # 设置图的标题为当前核函数的名称

plt.show()  # 展示所有图形
```