# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\tick_xlabel_top.py`

```
"""
==================================
Move x-axis tick labels to the top
==================================

`~.axes.Axes.tick_params` can be used to configure the ticks. *top* and
*labeltop* control the visibility of tick lines and labels at the top x-axis.
To move x-axis ticks from bottom to top, we have to activate the top ticks
and deactivate the bottom ticks::

    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

.. note::

    If the change should be made for all future plots and not only the current
    Axes, you can adapt the respective config parameters

    - :rc:`xtick.top`
    - :rc:`xtick.labeltop`
    - :rc:`xtick.bottom`
    - :rc:`xtick.labelbottom`

"""

# 导入 matplotlib 的 pyplot 模块
import matplotlib.pyplot as plt

# 创建一个新的图形和坐标系
fig, ax = plt.subplots()

# 在坐标系上绘制一个简单的折线图（这里是0到9的数据）
ax.plot(range(10))

# 调整坐标轴的刻度参数，将 x 轴的刻度标签移动到顶部并隐藏底部的刻度线和标签
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

# 设置图形的标题
ax.set_title('x-ticks moved to the top')

# 显示图形
plt.show()
```