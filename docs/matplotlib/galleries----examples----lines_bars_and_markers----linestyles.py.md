# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\linestyles.py`

```
"""
==========
Linestyles
==========

Simple linestyles can be defined using the strings "solid", "dotted", "dashed"
or "dashdot". More refined control can be achieved by providing a dash tuple
``(offset, (on_off_seq))``. For example, ``(0, (3, 10, 1, 15))`` means
(3pt line, 10pt space, 1pt line, 15pt space) with no offset, while
``(5, (10, 3))``, means (10pt line, 3pt space), but skip the first 5pt line.
See also `.Line2D.set_linestyle`.

*Note*: The dash style can also be configured via `.Line2D.set_dashes`
as shown in :doc:`/gallery/lines_bars_and_markers/line_demo_dash_control`
and passing a list of dash sequences using the keyword *dashes* to the
cycler in :ref:`property_cycle <color_cycle>`.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 模块，并简称为 np

# 定义预设的线条样式和对应的字符串表示
linestyle_str = [
     ('solid', 'solid'),      # 等同于 (0, ()) 或者 '-'
     ('dotted', 'dotted'),    # 等同于 (0, (1, 1)) 或者 ':'
     ('dashed', 'dashed'),    # 等同于 '--'
     ('dashdot', 'dashdot')]  # 等同于 '-.'

# 定义具体参数化的线条样式及其说明
linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),
     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

def plot_linestyles(ax, linestyles, title):
    # 生成 X 和 Y 数据
    X, Y = np.linspace(0, 100, 10), np.zeros(10)
    yticklabels = []

    # 遍历每种线条样式及其名称
    for i, (name, linestyle) in enumerate(linestyles):
        # 在指定的坐标系 ax 上绘制线条，设置线条样式、线宽和颜色
        ax.plot(X, Y+i, linestyle=linestyle, linewidth=1.5, color='black')
        yticklabels.append(name)

    # 设置图表标题和 y 轴的标签及刻度
    ax.set_title(title)
    ax.set(ylim=(-0.5, len(linestyles)-0.5),
           yticks=np.arange(len(linestyles)),
           yticklabels=yticklabels)
    ax.tick_params(left=False, bottom=False, labelbottom=False)
    ax.spines[:].set_visible(False)

    # 对每种线条样式，添加带有偏移的文本注释
    for i, (name, linestyle) in enumerate(linestyles):
        ax.annotate(repr(linestyle),
                    xy=(0.0, i), xycoords=ax.get_yaxis_transform(),
                    xytext=(-6, -12), textcoords='offset points',
                    color="blue", fontsize=8, ha="right", family="monospace")

# 创建包含两个子图的图表对象
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1, 3])

# 分别绘制命名的线条样式和参数化的线条样式
plot_linestyles(ax0, linestyle_str[::-1], title='Named linestyles')
plot_linestyles(ax1, linestyle_tuple[::-1], title='Parametrized linestyles')

# 调整布局，使图表更加紧凑
plt.tight_layout()

# 显示图表
plt.show()
```