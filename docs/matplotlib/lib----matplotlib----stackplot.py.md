# `D:\src\scipysrc\matplotlib\lib\matplotlib\stackplot.py`

```py
"""
Stacked area plot for 1D arrays inspired by Douglas Y'barbo's stackoverflow
answer:
https://stackoverflow.com/q/2225995/

(https://stackoverflow.com/users/66549/doug)
"""

import itertools  # 导入 itertools 模块，用于生成迭代器的工具函数

import numpy as np  # 导入 NumPy 库，用于处理数组和数值计算

from matplotlib import _api  # 从 matplotlib 中导入 _api 模块

__all__ = ['stackplot']  # 定义模块的公开接口，包括 stackplot 函数

def stackplot(axes, x, *args,
              labels=(), colors=None, hatch=None, baseline='zero',
              **kwargs):
    """
    Draw a stacked area plot or a streamgraph.

    Parameters
    ----------
    x : (N,) array-like
        X 轴数据，一维数组

    *args : (M, N) array-like
        不堆叠的数据。下列调用都合法：

            stackplot(x, y)           # 其中 y 的形状为 (M, N)
            stackplot(x, y1, y2, y3)  # 其中 y1, y2, y3, y4 的长度为 N

    baseline : {'zero', 'sym', 'wiggle', 'weighted_wiggle'}
        计算基线的方法：

        - ``'zero'``: 固定零基线，即简单的堆叠图。
        - ``'sym'``: 围绕零对称，有时称为 'ThemeRiver'。
        - ``'wiggle'``: 最小化斜率平方和。
        - ``'weighted_wiggle'``: 同样最小化斜率平方和，但考虑每层的大小权重。有时称为 'Streamgraph' 布局。

    labels : list of str, optional
        每个数据系列的标签序列。如果未指定，则不会为图形添加标签。

    colors : list of :mpltype:`color`, optional
        堆叠区域的填充颜色序列。序列的长度不需要与提供的 *y* 的数量完全相同，颜色会从头开始重复。

        如果未指定，则使用 Axes 属性循环中的颜色。

    hatch : list of str, default: None
        填充堆叠区域的阴影样式序列。参见：/gallery/shapes_and_collections/hatch_style_reference。
        样式序列将从底部到顶部循环使用。

        如果未指定或为字符串，则重复使用一种样式。

    data : indexable object, optional
        DATA_PARAMETER_PLACEHOLDER

    **kwargs
        所有其他关键字参数将传递给 `.Axes.fill_between`。

    Returns
    -------
    list of `.PolyCollection`
        堆叠区域图的 `.PolyCollection` 实例列表，每个实例对应一个堆叠区域元素。
    """

    y = np.vstack(args)  # 堆叠输入的数据，得到堆叠后的 y 数据

    labels = iter(labels)  # 创建标签的迭代器
    if colors is not None:
        colors = itertools.cycle(colors)  # 如果颜色序列不为空，创建颜色的循环迭代器
    else:
        colors = (axes._get_lines.get_next_color() for _ in y)  # 否则使用 Axes 的默认颜色循环

    if hatch is None or isinstance(hatch, str):
        hatch = itertools.cycle([hatch])  # 如果阴影样式为空或者是字符串，则创建阴影样式的循环迭代器
    else:
        hatch = itertools.cycle(hatch)  # 否则使用提供的阴影样式序列创建循环迭代器

    # 假设传入的数据尚未堆叠，因此在这里进行堆叠处理。
    # 我们将需要一个浮点数缓冲区来进行接下来的计算。
    stack = np.cumsum(y, axis=0, dtype=np.promote_types(y.dtype, np.float32))

    # 检查基线参数是否在允许的列表中，包括 'zero', 'sym', 'wiggle', 'weighted_wiggle'
    _api.check_in_list(['zero', 'sym', 'wiggle', 'weighted_wiggle'],
                       baseline=baseline)

    # 如果基线是 'zero'，则将第一行设为0.0
    if baseline == 'zero':
        first_line = 0.

    # 如果基线是 'sym'，计算第一行为所有列的和的负值的一半，并将其加到堆栈中
    elif baseline == 'sym':
        first_line = -np.sum(y, 0) * 0.5
        stack += first_line[None, :]

    # 如果基线是 'wiggle'，计算第一行为权重调整后的值，并将其加到堆栈中
    elif baseline == 'wiggle':
        m = y.shape[0]
        first_line = (y * (m - 0.5 - np.arange(m)[:, None])).sum(0)
        first_line /= -m
        stack += first_line

    # 如果基线是 'weighted_wiggle'，根据权重调整计算中心，并将其加到堆栈中
    elif baseline == 'weighted_wiggle':
        total = np.sum(y, 0)
        inv_total = np.zeros_like(total)
        mask = total > 0
        inv_total[mask] = 1.0 / total[mask]
        increase = np.hstack((y[:, 0:1], np.diff(y)))
        below_size = total - stack
        below_size += 0.5 * y
        move_up = below_size * inv_total
        move_up[:, 0] = 0.5
        center = (move_up - 0.5) * increase
        center = np.cumsum(center.sum(0))
        first_line = center - 0.5 * total
        stack += first_line

    # 在 x = 0 和第一个数组之间填充颜色
    coll = axes.fill_between(x, first_line, stack[0, :],
                             facecolor=next(colors),
                             hatch=next(hatch),
                             label=next(labels, None),
                             **kwargs)
    # 设置填充的边缘粘性
    coll.sticky_edges.y[:] = [0]
    r = [coll]

    # 在每两个数组之间填充颜色
    for i in range(len(y) - 1):
        r.append(axes.fill_between(x, stack[i, :], stack[i + 1, :],
                                   facecolor=next(colors),
                                   hatch=next(hatch),
                                   label=next(labels, None),
                                   **kwargs))
    # 返回填充对象的列表
    return r
```