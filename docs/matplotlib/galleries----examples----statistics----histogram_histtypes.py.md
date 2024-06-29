# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\histogram_histtypes.py`

```py
"""
================================================================
Demo of the histogram function's different ``histtype`` settings
================================================================

* Histogram with step curve that has a color fill.
* Histogram with step curve with no fill.
* Histogram with custom and unequal bin widths.
* Two histograms with stacked bars.

Selecting different bin counts and sizes can significantly affect the
shape of a histogram. The Astropy docs have a great section on how to
select these parameters:
http://docs.astropy.org/en/stable/visualization/histogram.html
"""

# 导入 matplotlib.pyplot 库，并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简写为 np
import numpy as np

# 设置随机数种子，以便结果可重现
np.random.seed(19680801)

# 生成一组服从正态分布的随机数 x
mu_x = 200
sigma_x = 25
x = np.random.normal(mu_x, sigma_x, size=100)

# 生成另一组服从正态分布的随机数 w
mu_w = 200
sigma_w = 10
w = np.random.normal(mu_w, sigma_w, size=100)

# 创建一个 2x2 的图形布局
fig, axs = plt.subplots(nrows=2, ncols=2)

# 绘制第一个子图：stepfilled 类型的直方图
axs[0, 0].hist(x, 20, density=True, histtype='stepfilled', facecolor='g',
               alpha=0.75)
axs[0, 0].set_title('stepfilled')

# 绘制第二个子图：step 类型的直方图
axs[0, 1].hist(x, 20, density=True, histtype='step', facecolor='g',
               alpha=0.75)
axs[0, 1].set_title('step')

# 绘制第三个子图：两个数据集的堆叠条形直方图
axs[1, 0].hist(x, density=True, histtype='barstacked', rwidth=0.8)
axs[1, 0].hist(w, density=True, histtype='barstacked', rwidth=0.8)
axs[1, 0].set_title('barstacked')

# 绘制第四个子图：使用不均匀的边界创建条形直方图
bins = [100, 150, 180, 195, 205, 220, 250, 300]
axs[1, 1].hist(x, bins, density=True, histtype='bar', rwidth=0.8)
axs[1, 1].set_title('bar, unequal bins')

# 调整子图之间的布局
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