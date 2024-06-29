# `D:\src\scipysrc\matplotlib\galleries\examples\scales\logit_demo.py`

```py
"""
================
Logit Demo
================

Examples of plots with logit axes.
"""

import math  # 导入数学库

import matplotlib.pyplot as plt  # 导入matplotlib库
import numpy as np  # 导入numpy库

xmax = 10  # 定义变量xmax为10
x = np.linspace(-xmax, xmax, 10000)  # 在区间[-xmax, xmax]上生成10000个均匀间隔的点
cdf_norm = [math.erf(w / np.sqrt(2)) / 2 + 1 / 2 for w in x]  # 计算标准正态分布的累积分布函数
cdf_laplacian = np.where(x < 0, 1 / 2 * np.exp(x), 1 - 1 / 2 * np.exp(-x))  # 计算拉普拉斯分布的累积分布函数
cdf_cauchy = np.arctan(x) / np.pi + 1 / 2  # 计算柯西分布的累积分布函数

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6.4, 8.5))  # 创建一个3行2列的图形布局

# Common part, for the example, we will do the same plots on all graphs
# 共同部分，例子中我们将在所有图上做相同的绘图
for i in range(3):
    for j in range(2):
        axs[i, j].plot(x, cdf_norm, label=r"$\mathcal{N}$")  # 绘制标准正态分布的累积分布函数曲线
        axs[i, j].plot(x, cdf_laplacian, label=r"$\mathcal{L}$")  # 绘制拉普拉斯分布的累积分布函数曲线
        axs[i, j].plot(x, cdf_cauchy, label="Cauchy")  # 绘制柯西分布的累积分布函数曲线
        axs[i, j].legend()  # 添加图例
        axs[i, j].grid()  # 添加网格线

# First line, logitscale, with standard notation
# 第一行，使用标准记法的logit刻度
axs[0, 0].set(title="logit scale")  # 设置子图标题为"logit scale"
axs[0, 0].set_yscale("logit")  # 设置y轴为logit刻度
axs[0, 0].set_ylim(1e-5, 1 - 1e-5)  # 设置y轴范围

axs[0, 1].set(title="logit scale")  # 设置子图标题为"logit scale"
axs[0, 1].set_yscale("logit")  # 设置y轴为logit刻度
axs[0, 1].set_xlim(0, xmax)  # 设置x轴范围
axs[0, 1].set_ylim(0.8, 1 - 5e-3)  # 设置y轴范围

# Second line, logitscale, with survival notation (with `use_overline`), and
# other format display 1/2
# 第二行，logit刻度，使用生存函数的表示法（带有`use_overline`），并且
# 其它格式显示1/2
axs[1, 0].set(title="logit scale")  # 设置子图标题为"logit scale"
axs[1, 0].set_yscale("logit", one_half="1/2", use_overline=True)  # 设置y轴为logit刻度，使用生存函数表示法，显示1/2，并使用上划线
axs[1, 0].set_ylim(1e-5, 1 - 1e-5)  # 设置y轴范围

axs[1, 1].set(title="logit scale")  # 设置子图标题为"logit scale"
axs[1, 1].set_yscale("logit", one_half="1/2", use_overline=True)  # 设置y轴为logit刻度，使用生存函数表示法，显示1/2，并使用上划线
axs[1, 1].set_xlim(0, xmax)  # 设置x轴范围
axs[1, 1].set_ylim(0.8, 1 - 5e-3)  # 设置y轴范围

# Third line, linear scale
# 第三行，线性刻度
axs[2, 0].set(title="linear scale")  # 设置子图标题为"linear scale"
axs[2, 0].set_ylim(0, 1)  # 设置y轴范围为0到1

axs[2, 1].set(title="linear scale")  # 设置子图标题为"linear scale"
axs[2, 1].set_xlim(0, xmax)  # 设置x轴范围
axs[2, 1].set_ylim(0.8, 1)  # 设置y轴范围为0.8到1

fig.tight_layout()  # 调整子图布局
plt.show()  # 显示图形
```