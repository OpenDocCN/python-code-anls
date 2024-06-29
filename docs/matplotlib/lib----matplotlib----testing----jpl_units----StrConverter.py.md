# `D:\src\scipysrc\matplotlib\lib\matplotlib\testing\jpl_units\StrConverter.py`

```py
"""StrConverter module containing class StrConverter."""

import numpy as np  # 导入 NumPy 库

import matplotlib.units as units  # 导入 Matplotlib 的 units 模块

__all__ = ['StrConverter']  # 模块中公开的类列表


class StrConverter(units.ConversionInterface):
    """
    A Matplotlib converter class for string data values.

    Valid units for string are:
    - 'indexed' : Values are indexed as they are specified for plotting.
    - 'sorted'  : Values are sorted alphanumerically.
    - 'inverted' : Values are inverted so that the first value is on top.
    - 'sorted-inverted' :  A combination of 'sorted' and 'inverted'
    """

    @staticmethod
    def axisinfo(unit, axis):
        # docstring inherited
        return None  # 返回空值

    @staticmethod
    def convert(value, unit, axis):
        # docstring inherited

        if value == []:
            return []  # 如果值为空列表，则返回空列表

        # we delay loading to make matplotlib happy
        ax = axis.axes  # 获取轴对象
        if axis is ax.xaxis:
            isXAxis = True  # 判断是否为 X 轴
        else:
            isXAxis = False

        axis.get_major_ticks()  # 获取主刻度
        ticks = axis.get_ticklocs()  # 获取刻度位置
        labels = axis.get_ticklabels()  # 获取刻度标签

        labels = [l.get_text() for l in labels if l.get_text()]  # 获取刻度标签的文本内容

        if not labels:
            ticks = []
            labels = []

        if not np.iterable(value):
            value = [value]  # 如果 value 不可迭代，转为列表

        newValues = []
        for v in value:
            if v not in labels and v not in newValues:
                newValues.append(v)  # 将不在标签中的新值添加到 newValues 中

        labels.extend(newValues)  # 将新值添加到标签列表中

        # DISABLED: This is disabled because matplotlib bar plots do not
        # DISABLED: recalculate the unit conversion of the data values
        # DISABLED: this is due to design and is not really a bug.
        # DISABLED: If this gets changed, then we can activate the following
        # DISABLED: block of code.  Note that this works for line plots.
        # DISABLED if unit:
        # DISABLED     if unit.find("sorted") > -1:
        # DISABLED         labels.sort()
        # DISABLED     if unit.find("inverted") > -1:
        # DISABLED         labels = labels[::-1]

        # add padding (so they do not appear on the axes themselves)
        labels = [''] + labels + ['']  # 在标签列表两端添加空字符串
        ticks = list(range(len(labels)))  # 创建与标签列表长度相同的刻度位置列表
        ticks[0] = 0.5
        ticks[-1] = ticks[-1] - 0.5

        axis.set_ticks(ticks)  # 设置轴的刻度位置
        axis.set_ticklabels(labels)  # 设置轴的刻度标签
        # we have to do the following lines to make ax.autoscale_view work
        loc = axis.get_major_locator()  # 获取主定位器
        loc.set_bounds(ticks[0], ticks[-1])  # 设置定位器的边界

        if isXAxis:
            ax.set_xlim(ticks[0], ticks[-1])  # 设置 X 轴的显示范围
        else:
            ax.set_ylim(ticks[0], ticks[-1])  # 设置 Y 轴的显示范围

        result = [ticks[labels.index(v)] for v in value]  # 根据值在标签中的索引生成结果列表

        ax.viewLim.ignore(-1)  # 忽略特定的视图限制
        return result  # 返回结果列表

    @staticmethod
    def default_units(value, axis):
        # docstring inherited
        # The default behavior for string indexing.
        return "indexed"  # 返回默认的单位 "indexed"
```