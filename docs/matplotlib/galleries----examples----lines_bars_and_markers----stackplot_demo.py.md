# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\stackplot_demo.py`

```
"""
===========================
Stackplots and streamgraphs
===========================
"""

# %%
# Stackplots
# ----------
#
# Stackplots draw multiple datasets as vertically stacked areas. This is
# useful when the individual data values and additionally their cumulative
# value are of interest.

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

# data from United Nations World Population Prospects (Revision 2019)
# https://population.un.org/wpp/, license: CC BY 3.0 IGO
# 年份数据
year = [1950, 1960, 1970, 1980, 1990, 2000, 2010, 2018]
# 各大洲的人口数据
population_by_continent = {
    'Africa': [.228, .284, .365, .477, .631, .814, 1.044, 1.275],
    'the Americas': [.340, .425, .519, .619, .727, .840, .943, 1.006],
    'Asia': [1.394, 1.686, 2.120, 2.625, 3.202, 3.714, 4.169, 4.560],
    'Europe': [.220, .253, .276, .295, .310, .303, .294, .293],
    'Oceania': [.012, .015, .019, .022, .026, .031, .036, .039],
}

# 创建一个新的图形和轴对象
fig, ax = plt.subplots()
# 使用 stackplot 方法绘制堆叠面积图
ax.stackplot(year, population_by_continent.values(),
             labels=population_by_continent.keys(), alpha=0.8)
# 添加图例，位置在左上角，反转顺序
ax.legend(loc='upper left', reverse=True)
# 设置图表标题
ax.set_title('World population')
# 设置 X 轴标签
ax.set_xlabel('Year')
# 设置 Y 轴标签
ax.set_ylabel('Number of people (billions)')
# 在每2亿人口处添加一个刻度
ax.yaxis.set_minor_locator(mticker.MultipleLocator(.2))

plt.show()

# %%
# Streamgraphs
# ------------
#
# Using the *baseline* parameter, you can turn an ordinary stacked area plot
# with baseline 0 into a stream graph.

# Fixing random state for reproducibility
np.random.seed(19680801)

# 定义一个高斯混合函数，生成在给定位置 x 上的 n 个高斯分布的随机混合
def gaussian_mixture(x, n=5):
    """Return a random mixture of *n* Gaussians, evaluated at positions *x*."""
    # 定义添加随机高斯分布的内部函数
    def add_random_gaussian(a):
        # 随机生成高斯分布的振幅
        amplitude = 1 / (.1 + np.random.random())
        # 计算 x 的范围
        dx = x[-1] - x[0]
        # 在范围内随机生成 x0
        x0 = (2 * np.random.random() - .5) * dx
        # 随机生成 z
        z = 10 / (.1 + np.random.random()) / dx
        # 向数组 a 中添加高斯分布
        a += amplitude * np.exp(-(z * (x - x0))**2)
    
    # 初始化一个与 x 等长的零数组
    a = np.zeros_like(x)
    # 多次调用内部函数，生成多个高斯分布的叠加
    for j in range(n):
        add_random_gaussian(a)
    return a

# 在 0 到 100 之间生成 101 个均匀分布的点作为 x 值
x = np.linspace(0, 100, 101)
# 生成 3 个随机高斯混合数据集
ys = [gaussian_mixture(x) for _ in range(3)]

# 创建一个新的图形和轴对象
fig, ax = plt.subplots()
# 使用 stackplot 方法绘制堆叠面积图，baseline 参数设置为 'wiggle'
ax.stackplot(x, ys, baseline='wiggle')
plt.show()
```