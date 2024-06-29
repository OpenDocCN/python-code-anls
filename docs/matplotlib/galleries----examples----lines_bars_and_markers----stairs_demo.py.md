# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\stairs_demo.py`

```
"""
===========
Stairs Demo
===========

This example demonstrates the use of `~.matplotlib.pyplot.stairs` for stepwise
constant functions. A common use case is histogram and histogram-like data
visualization.

"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

from matplotlib.patches import StepPatch  # 从matplotlib.patches模块导入StepPatch类，用于创建步阶图形

np.random.seed(0)  # 设置随机种子，以确保结果可重复
h, edges = np.histogram(np.random.normal(5, 3, 5000),  # 使用正态分布生成5000个随机数，计算直方图
                        bins=np.linspace(0, 10, 20))  # 设置直方图的边界

fig, axs = plt.subplots(3, 1, figsize=(7, 15))  # 创建包含3个子图的画布，设置尺寸

# 在第一个子图上绘制步阶直方图，并添加标签说明
axs[0].stairs(h, edges, label='Simple histogram')
axs[0].stairs(h, edges + 5, baseline=50, label='Modified baseline')
axs[0].stairs(h, edges + 10, baseline=None, label='No edges')
axs[0].set_title("Step Histograms")  # 设置子图标题

# 在第二个子图上绘制填充的步阶直方图，并添加标签说明
axs[1].stairs(np.arange(1, 6, 1), fill=True,
              label='Filled histogram\nw/ automatic edges')
axs[1].stairs(np.arange(1, 6, 1)*0.3, np.arange(2, 8, 1),
              orientation='horizontal', hatch='//',
              label='Hatched histogram\nw/ horizontal orientation')
axs[1].set_title("Filled histogram")  # 设置子图标题

# 创建StepPatch对象，并添加到第三个子图上，设置坐标轴范围和标题
patch = StepPatch(values=[1, 2, 3, 2, 1],
                  edges=range(1, 7),
                  label=('Patch derived underlying object\n'
                         'with default edge/facecolor behaviour'))
axs[2].add_patch(patch)
axs[2].set_xlim(0, 7)
axs[2].set_ylim(-1, 5)
axs[2].set_title("StepPatch artist")  # 设置子图标题

for ax in axs:
    ax.legend()  # 在每个子图上添加图例
plt.show()  # 显示图形

# %%
# *baseline* can take an array to allow for stacked histogram plots
A = [[0, 0, 0],
     [1, 2, 3],
     [2, 4, 6],
     [3, 6, 9]]

for i in range(len(A) - 1):
    plt.stairs(A[i+1], baseline=A[i], fill=True)

# %%
# Comparison of `.pyplot.step` and `.pyplot.stairs`
# -------------------------------------------------
#
# `.pyplot.step` defines the positions of the steps as single values. The steps
# extend left/right/both ways from these reference values depending on the
# parameter *where*. The number of *x* and *y* values is the same.
#
# In contrast, `.pyplot.stairs` defines the positions of the steps via their
# bounds *edges*, which is one element longer than the step values.

bins = np.arange(14)
centers = bins[:-1] + np.diff(bins) / 2
y = np.sin(centers / 2)

plt.step(bins[:-1], y, where='post', label='step(where="post")')
plt.plot(bins[:-1], y, 'o--', color='grey', alpha=0.3)

plt.stairs(y - 1, bins, baseline=None, label='stairs()')
plt.plot(centers, y - 1, 'o--', color='grey', alpha=0.3)
plt.plot(np.repeat(bins, 2), np.hstack([y[0], np.repeat(y, 2), y[-1]]) - 1,
         'o', color='red', alpha=0.2)

plt.legend()  # 添加图例
plt.title('step() vs. stairs()')  # 设置图形标题
plt.show()  # 显示图形

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.stairs` / `matplotlib.pyplot.stairs`
#    - `matplotlib.patches.StepPatch`
```