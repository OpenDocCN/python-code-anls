# `D:\src\scipysrc\seaborn\examples\layered_bivariate_plot.py`

```
"""
Bivariate plot with multiple elements
=====================================

This script generates a bivariate plot with a combination of scatterplot,
histogram, and density contours using seaborn and matplotlib.

"""

import numpy as np                  # 导入NumPy库，用于数值计算
import seaborn as sns               # 导入Seaborn库，用于统计数据可视化
import matplotlib.pyplot as plt     # 导入Matplotlib库，用于绘图
sns.set_theme(style="dark")         # 设置Seaborn的绘图风格为暗色系列

# Simulate data from a bivariate Gaussian
n = 10000                           # 设置样本数量
mean = [0, 0]                       # 设置高斯分布的均值
cov = [(2, .4), (.4, .2)]            # 设置高斯分布的协方差矩阵
rng = np.random.RandomState(0)      # 设置随机数种子，确保结果可重复
x, y = rng.multivariate_normal(mean, cov, n).T   # 生成二维高斯分布的随机样本

# Draw a combo histogram and scatterplot with density contours
f, ax = plt.subplots(figsize=(6, 6))    # 创建一个6x6英寸大小的绘图窗口
sns.scatterplot(x=x, y=y, s=5, color=".15")   # 绘制散点图，设置点的大小和颜色
sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")   # 绘制联合直方图，设置直方图的参数
sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)   # 绘制二维核密度估计图，设置等高线的层数和线宽
```