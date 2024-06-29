# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\histogram_multihist.py`

```py
"""
=====================================================
The histogram (hist) function with multiple data sets
=====================================================

Plot histogram with multiple sample sets and demonstrate:

* Use of legend with multiple sample sets
* Stacked bars
* Step curve with no fill
* Data sets of different sample sizes

Selecting different bin counts and sizes can significantly affect the
shape of a histogram. The Astropy docs have a great section on how to
select these parameters:
http://docs.astropy.org/en/stable/visualization/histogram.html
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，并设置随机种子
import numpy as np

np.random.seed(19680801)

# 设置直方图的箱数
n_bins = 10
# 生成三组服从标准正态分布的随机数据
x = np.random.randn(1000, 3)

# 创建一个 2x2 的子图布局
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

# 设置三组数据对应的颜色
colors = ['red', 'tan', 'lime']
# 绘制堆叠的直方图，并添加图例
ax0.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
ax0.legend(prop={'size': 10})
ax0.set_title('bars with legend')

# 绘制堆叠的直方图
ax1.hist(x, n_bins, density=True, histtype='bar', stacked=True)
ax1.set_title('stacked bar')

# 绘制步进线的直方图（不填充）
ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)
ax2.set_title('stack step (unfilled)')

# 绘制不同样本大小的多组数据的直方图
x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
ax3.hist(x_multi, n_bins, histtype='bar')
ax3.set_title('different sample sizes')

# 调整子图布局，使得图形显示更加紧凑
fig.tight_layout()
# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.hist` / `matplotlib.pyplot.hist`
```