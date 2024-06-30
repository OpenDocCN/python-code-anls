# `D:\src\scipysrc\scikit-learn\examples\svm\plot_custom_kernel.py`

```
"""
======================
SVM with custom kernel
======================

Simple usage of Support Vector Machines to classify a sample. It will
plot the decision surface and the support vectors.

"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库用于绘图
import numpy as np  # 导入 numpy 库用于数值计算

from sklearn import datasets, svm  # 导入 sklearn 中的 datasets 和 svm 模块
from sklearn.inspection import DecisionBoundaryDisplay  # 导入 DecisionBoundaryDisplay 用于绘制决策边界的显示

# import some data to play with
# 导入鸢尾花数据集
iris = datasets.load_iris()
X = iris.data[:, :2]  # 仅使用前两个特征进行训练。可以通过使用二维数据集来避免这种切片
Y = iris.target  # 鸢尾花数据集的目标值


def my_kernel(X, Y):
    """
    We create a custom kernel:

                 (2  0)
    k(X, Y) = X  (    ) Y.T
                 (0  1)
    """
    M = np.array([[2, 0], [0, 1.0]])  # 定义一个自定义的核矩阵 M
    return np.dot(np.dot(X, M), Y.T)  # 计算自定义核函数的结果


h = 0.02  # 网格中的步长大小

# 创建 SVM 的一个实例并拟合我们的数据
clf = svm.SVC(kernel=my_kernel)  # 使用自定义核函数创建一个 SVM 分类器
clf.fit(X, Y)  # 拟合数据

ax = plt.gca()  # 获取当前的 Axes 对象用于绘图

# 使用 DecisionBoundaryDisplay 从估算器创建决策边界显示
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    cmap=plt.cm.Paired,  # 配色方案
    ax=ax,  # 指定绘图的 Axes 对象
    response_method="predict",  # 响应方法为预测
    plot_method="pcolormesh",  # 绘制方法为填充网格
    shading="auto",  # 自动选择阴影
)

# 绘制训练数据点
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors="k")  # 绘制散点图
plt.title("3-Class classification using Support Vector Machine with custom kernel")  # 设置图表标题
plt.axis("tight")  # 自动调整坐标轴范围以适应数据点
plt.show()  # 显示图形
```