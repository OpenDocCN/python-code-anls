# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\colorbar_tick_labelling_demo.py`

```
"""
=======================
Colorbar Tick Labelling
=======================

Vertical colorbars have ticks, tick labels, and labels visible on the *y* axis,
horizontal colorbars on the *x* axis. The ``ticks`` parameter can be used to
set the ticks and the ``format`` parameter can be used to format the tick labels
of the visible colorbar Axes. For further adjustments, the ``yaxis`` or
``xaxis`` Axes of the colorbar can be retrieved using its ``ax`` property.
"""
# 导入 matplotlib 库中的 pyplot 模块并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库并重命名为 np
import numpy as np
# 导入 matplotlib 中的 ticker 模块并重命名为 mticker
import matplotlib.ticker as mticker

# Fixing random state for reproducibility
# 使用指定的种子创建随机数生成器对象
rng = np.random.default_rng(seed=19680801)

# %%
# Make plot with vertical (default) colorbar

# 创建一个图形窗口和一个子图对象
fig, ax = plt.subplots()

# 生成一个随机的 250x250 的正态分布数据
data = rng.standard_normal((250, 250))

# 在子图上显示数据，设置颜色映射为 'coolwarm'，指定最小值和最大值
cax = ax.imshow(data, vmin=-1, vmax=1, cmap='coolwarm')
# 设置子图的标题
ax.set_title('Gaussian noise with vertical colorbar')

# Add colorbar, make sure to specify tick locations to match desired ticklabels
# 添加垂直方向的颜色条，设置 ticks 参数为 [-1, 0, 1]，format 参数为指定的标签格式
cbar = fig.colorbar(cax,
                    ticks=[-1, 0, 1],
                    format=mticker.FixedFormatter(['< -1', '0', '> 1']),
                    extend='both'
                    )
# 获取颜色条的刻度标签
labels = cbar.ax.get_yticklabels()
# 调整第一个和最后一个标签的垂直对齐方式
labels[0].set_verticalalignment('top')
labels[-1].set_verticalalignment('bottom')

# %%
# Make plot with horizontal colorbar

# 创建一个新的图形窗口和一个子图对象
fig, ax = plt.subplots()

# 将数据限制在 [-1, 1] 范围内
data = np.clip(data, -1, 1)

# 在子图上显示数据，设置颜色映射为 'afmhot'
cax = ax.imshow(data, cmap='afmhot')
# 设置子图的标题
ax.set_title('Gaussian noise with horizontal colorbar')

# Add colorbar and adjust ticks afterwards
# 添加水平方向的颜色条，设置 ticks 参数为 [-1, 0, 1]，labels 参数为指定的标签
cbar = fig.colorbar(cax, orientation='horizontal')
cbar.set_ticks(ticks=[-1, 0, 1], labels=['Low', 'Medium', 'High'])

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.colorbar.Colorbar.set_ticks`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
```