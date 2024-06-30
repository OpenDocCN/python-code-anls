# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_sgd_weighted_samples.py`

```
"""
=====================
SGD: Weighted samples
=====================

Plot decision function of a weighted dataset, where the size of points
is proportional to its weight.

"""

import matplotlib.pyplot as plt  # 导入绘图库matplotlib
import numpy as np  # 导入数值计算库numpy

from sklearn import linear_model  # 导入线性模型类

# we create 20 points
np.random.seed(0)  # 设定随机种子，保证结果可复现
X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]  # 生成随机数据点，总共20个点，其中前10个点均值偏移为[1, 1]
y = [1] * 10 + [-1] * 10  # 设置标签，前10个点标签为1，后10个点标签为-1
sample_weight = 100 * np.abs(np.random.randn(20))  # 根据正态分布生成样本权重，权重为绝对值后乘以100

# and assign a bigger weight to the last 10 samples
sample_weight[:10] *= 10  # 将前10个样本的权重乘以10，使其比后10个样本的权重更大

# plot the weighted data points
xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))  # 生成用于绘制等高线图的网格点
fig, ax = plt.subplots()  # 创建图形和子图对象
ax.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    s=sample_weight,
    alpha=0.9,
    cmap=plt.cm.bone,
    edgecolor="black",
)  # 绘制带权重的散点图

# fit the unweighted model
clf = linear_model.SGDClassifier(alpha=0.01, max_iter=100)  # 创建SGD分类器对象，未使用样本权重
clf.fit(X, y)  # 使用未加权的数据拟合模型
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  # 计算决策函数值
Z = Z.reshape(xx.shape)  # 调整决策函数值的形状以匹配网格形状
no_weights = ax.contour(xx, yy, Z, levels=[0], linestyles=["solid"])  # 绘制决策边界等高线图，不加权

# fit the weighted model
clf = linear_model.SGDClassifier(alpha=0.01, max_iter=100)  # 创建SGD分类器对象，使用样本权重
clf.fit(X, y, sample_weight=sample_weight)  # 使用加权数据拟合模型
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  # 计算决策函数值
Z = Z.reshape(xx.shape)  # 调整决策函数值的形状以匹配网格形状
samples_weights = ax.contour(xx, yy, Z, levels=[0], linestyles=["dashed"])  # 绘制决策边界等高线图，加权

no_weights_handles, _ = no_weights.legend_elements()  # 获取不加权图例句柄
weights_handles, _ = samples_weights.legend_elements()  # 获取加权图例句柄
ax.legend(
    [no_weights_handles[0], weights_handles[0]],
    ["no weights", "with weights"],
    loc="lower left",
)  # 添加图例，显示不加权和加权模型的标签

ax.set(xticks=(), yticks=())  # 设置坐标轴刻度为空
plt.show()  # 显示图形
```