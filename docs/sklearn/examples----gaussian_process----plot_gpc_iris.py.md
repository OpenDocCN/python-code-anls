# `D:\src\scipysrc\scikit-learn\examples\gaussian_process\plot_gpc_iris.py`

```
"""
=====================================================
Gaussian process classification (GPC) on iris dataset
=====================================================

This example illustrates the predicted probability of GPC for an isotropic
and anisotropic RBF kernel on a two-dimensional version for the iris-dataset.
The anisotropic RBF kernel obtains slightly higher log-marginal-likelihood by
assigning different length-scales to the two feature dimensions.

"""

# 导入必要的库
import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图
import numpy as np  # 导入numpy库用于数值计算

from sklearn import datasets  # 导入sklearn中的datasets模块，用于加载数据集
from sklearn.gaussian_process import GaussianProcessClassifier  # 导入高斯过程分类器
from sklearn.gaussian_process.kernels import RBF  # 导入RBF核函数

# 导入一些用于玩耍的数据
iris = datasets.load_iris()  # 加载鸢尾花数据集
X = iris.data[:, :2]  # 只使用前两个特征进行训练和预测
y = np.array(iris.target, dtype=int)  # 将标签转换为整数类型数组

h = 0.02  # 网格步长

# 使用isotropic RBF核函数创建高斯过程分类器，并进行拟合
kernel = 1.0 * RBF([1.0])
gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel).fit(X, y)

# 使用anisotropic RBF核函数创建高斯过程分类器，并进行拟合
kernel = 1.0 * RBF([1.0, 1.0])
gpc_rbf_anisotropic = GaussianProcessClassifier(kernel=kernel).fit(X, y)

# 创建一个网格用于绘图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 计算x轴的边界
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 计算y轴的边界
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # 创建网格坐标

titles = ["Isotropic RBF", "Anisotropic RBF"]  # 设置子图标题

plt.figure(figsize=(10, 5))  # 创建绘图窗口
for i, clf in enumerate((gpc_rbf_isotropic, gpc_rbf_anisotropic)):
    # 绘制预测的概率。为此，我们将为网格中的每个点分配一种颜色
    plt.subplot(1, 2, i + 1)

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    # 将结果绘制成颜色图
    Z = Z.reshape((xx.shape[0], xx.shape[1], 3))
    plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")

    # 绘制训练点
    plt.scatter(X[:, 0], X[:, 1], c=np.array(["r", "g", "b"])[y], edgecolors=(0, 0, 0))
    plt.xlabel("Sepal length")  # 设置x轴标签
    plt.ylabel("Sepal width")  # 设置y轴标签
    plt.xlim(xx.min(), xx.max())  # 设置x轴范围
    plt.ylim(yy.min(), yy.max())  # 设置y轴范围
    plt.xticks(())  # 隐藏x轴刻度
    plt.yticks(())  # 隐藏y轴刻度
    # 设置子图标题，显示模型类型和对数边缘似然
    plt.title("%s, LML: %.3f" % (titles[i], clf.log_marginal_likelihood(clf.kernel_.theta)))

plt.tight_layout()  # 调整子图布局
plt.show()  # 显示图形
```