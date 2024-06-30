# `D:\src\scipysrc\scikit-learn\examples\neighbors\plot_nca_illustration.py`

```
"""
=============================================
Neighborhood Components Analysis Illustration
=============================================

This example illustrates a learned distance metric that maximizes
the nearest neighbors classification accuracy. It provides a visual
representation of this metric compared to the original point
space. Please refer to the :ref:`User Guide <nca>` for more information.

"""

# SPDX-License-Identifier: BSD-3-Clause

# 导入必要的库
import matplotlib.pyplot as plt  # 导入绘图库matplotlib
import numpy as np  # 导入数值计算库numpy
from matplotlib import cm  # 导入颜色映射模块
from scipy.special import logsumexp  # 导入scipy库中的logsumexp函数

from sklearn.datasets import make_classification  # 导入make_classification生成分类数据的函数
from sklearn.neighbors import NeighborhoodComponentsAnalysis  # 导入NeighborhoodComponentsAnalysis类

# %%
# Original points
# ---------------
# 首先创建一个包含3类的数据集，总共有9个样本，并在原始空间中绘制这些点。
# 本示例关注第3号点的分类。连接第3号点与其他点的连线粗细与它们的距离成正比。

X, y = make_classification(
    n_samples=9,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=1.0,
    random_state=0,
)

plt.figure(1)
ax = plt.gca()  # 获取当前图形的轴
for i in range(X.shape[0]):
    ax.text(X[i, 0], X[i, 1], str(i), va="center", ha="center")  # 在点的位置标注点号
    ax.scatter(X[i, 0], X[i, 1], s=300, c=cm.Set1(y[[i]]), alpha=0.4)  # 绘制散点图，每个点的颜色根据类别y确定

ax.set_title("Original points")  # 设置图标题为"Original points"
ax.axes.get_xaxis().set_visible(False)  # 隐藏x轴
ax.axes.get_yaxis().set_visible(False)  # 隐藏y轴
ax.axis("equal")  # 设置坐标轴等比例显示，确保边界显示正确的圆形

# 定义函数link_thickness_i和relate_point略过，已在示例中有详细解释

i = 3
relate_point(X, i, ax)  # 绘制连接第i号点与其他点的连线
plt.show()

# %%
# Learning an embedding
# ---------------------
# 使用NeighborhoodComponentsAnalysis学习嵌入，并在转换后绘制点。
# 然后获取嵌入并找到最近的邻居。

nca = NeighborhoodComponentsAnalysis(max_iter=30, random_state=0)  # 创建NCA对象，设置最大迭代次数和随机种子
nca = nca.fit(X, y)  # 使用数据X和标签y拟合NCA模型

plt.figure(2)
ax2 = plt.gca()  # 获取当前图形的轴
X_embedded = nca.transform(X)  # 将原始数据X转换为嵌入空间
relate_point(X_embedded, i, ax2)  # 绘制连接第i号点与其他点的连线

for i in range(len(X)):
    ax2.text(X_embedded[i, 0], X_embedded[i, 1], str(i), va="center", ha="center")  # 在嵌入空间中标注点号
    ax2.scatter(X_embedded[i, 0], X_embedded[i, 1], s=300, c=cm.Set1(y[[i]]), alpha=0.4)  # 绘制散点图，每个点的颜色根据类别y确定

ax2.set_title("NCA embedding")  # 设置图标题为"NCA embedding"
ax2.axes.get_xaxis().set_visible(False)  # 隐藏x轴
ax2.axes.get_yaxis().set_visible(False)  # 隐藏y轴
# 设置图形的纵横比为相等，以便正确显示圆形图等。这里的 "equal" 参数确保轴的纵横比相等。
ax2.axis("equal")
# 展示当前的图形。这会在屏幕上显示绘制的图形。
plt.show()
```