# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\tick_labels_from_values.py`

```
"""
=========================================
Setting tick labels from a list of values
=========================================

Using `.Axes.set_xticks` causes the tick labels to be set on the currently
chosen ticks. However, you may want to allow matplotlib to dynamically
choose the number of ticks and their spacing.

In this case it may be better to determine the tick label from the
value at the tick. The following example shows how to do this.

NB: The `.ticker.MaxNLocator` is used here to ensure that the tick values
take integer values.

"""

# 导入 matplotlib.pyplot 模块，简称为 plt
import matplotlib.pyplot as plt

# 导入 MaxNLocator 类从 matplotlib.ticker 模块
from matplotlib.ticker import MaxNLocator

# 创建一个图形和轴对象
fig, ax = plt.subplots()

# 创建一组 x 和 y 数据
xs = range(26)
ys = range(26)

# 创建一个包含小写字母 'a' 到 'z' 的列表作为标签
labels = list('abcdefghijklmnopqrstuvwxyz')

# 定义一个格式化函数，根据 tick_val 的整数部分返回对应的字母标签或空字符串
def format_fn(tick_val, tick_pos):
    if int(tick_val) in xs:
        return labels[int(tick_val)]
    else:
        return ''

# 设置 x 轴的主要格式化器为上面定义的 format_fn 函数
ax.xaxis.set_major_formatter(format_fn)

# 设置 x 轴的主要定位器为 MaxNLocator，确保刻度值为整数
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# 绘制图形，使用之前定义的 xs 和 ys 数据
ax.plot(xs, ys)

# 显示图形
plt.show()


# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.pyplot.subplots`
#    - `matplotlib.axis.Axis.set_major_formatter`
#    - `matplotlib.axis.Axis.set_major_locator`
#    - `matplotlib.ticker.FuncFormatter`
#    - `matplotlib.ticker.MaxNLocator`
```