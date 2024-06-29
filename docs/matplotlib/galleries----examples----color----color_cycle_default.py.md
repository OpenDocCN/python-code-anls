# `D:\src\scipysrc\matplotlib\galleries\examples\color\color_cycle_default.py`

```
"""
====================================
Colors in the default property cycle
====================================

Display the colors from the default prop_cycle, which is obtained from the
:ref:`rc parameters<customizing>`.
"""
# 导入 matplotlib 库
import matplotlib.pyplot as plt
import numpy as np

# 获取当前的图表参数中的默认属性循环
prop_cycle = plt.rcParams['axes.prop_cycle']
# 从属性循环中获取颜色列表
colors = prop_cycle.by_key()['color']

# 获取默认线条宽度
lwbase = plt.rcParams['lines.linewidth']
# 计算细线和粗线的宽度
thin = lwbase / 2
thick = lwbase * 3

# 创建一个包含4个子图的图表
fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
# 遍历子图列
for icol in range(2):
    # 根据列的索引确定线条的宽度
    if icol == 0:
        lwx, lwy = thin, lwbase
    else:
        lwx, lwy = lwbase, thick
    # 遍历子图行
    for irow in range(2):
        # 遍历颜色列表，并在每个子图中绘制水平和垂直线条
        for i, color in enumerate(colors):
            axs[irow, icol].axhline(i, color=color, lw=lwx)
            axs[irow, icol].axvline(i, color=color, lw=lwy)

    # 设置第二行子图的背景颜色为黑色，设置其 x 轴刻度
    axs[1, icol].set_facecolor('k')
    axs[1, icol].xaxis.set_ticks(np.arange(0, 10, 2))
    # 设置第一行子图的标题，显示当前线条宽度
    axs[0, icol].set_title(f'line widths (pts): {lwx:g}, {lwy:g}',
                           fontsize='medium')

# 设置第一列子图的 y 轴刻度
for irow in range(2):
    axs[irow, 0].yaxis.set_ticks(np.arange(0, 10, 2))

# 设置整个图表的总标题
fig.suptitle('Colors in the default prop_cycle', fontsize='large')

# 显示图表
plt.show()
```