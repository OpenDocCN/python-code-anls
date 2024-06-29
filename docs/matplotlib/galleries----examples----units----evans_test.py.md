# `D:\src\scipysrc\matplotlib\galleries\examples\units\evans_test.py`

```
"""
==========
Evans test
==========

A mockup "Foo" units class which supports conversion and different tick
formatting depending on the "unit".  Here the "unit" is just a scalar
conversion factor, but this example shows that Matplotlib is entirely agnostic
to what kind of units client packages use.
"""

import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图
import numpy as np  # 导入numpy模块，用于数值计算

import matplotlib.ticker as ticker  # 导入matplotlib的ticker模块，用于刻度格式化
import matplotlib.units as units  # 导入matplotlib的units模块，用于单位处理


class Foo:
    def __init__(self, val, unit=1.0):
        self.unit = unit  # 初始化单位
        self._val = val * unit  # 计算实际值，考虑单位因素

    def value(self, unit):
        """返回使用指定单位的值。如果单位为None，则使用对象的默认单位。"""
        if unit is None:
            unit = self.unit
        return self._val / unit


class FooConverter(units.ConversionInterface):
    @staticmethod
    def axisinfo(unit, axis):
        """返回Foo的AxisInfo对象。根据单位选择合适的刻度定位器和格式化器。"""
        if unit == 1.0 or unit == 2.0:
            return units.AxisInfo(
                majloc=ticker.IndexLocator(8, 0),  # 主要刻度定位器
                majfmt=ticker.FormatStrFormatter("VAL: %s"),  # 主要刻度格式化器
                label='foo',  # 轴标签
                )
        else:
            return None

    @staticmethod
    def convert(obj, unit, axis):
        """
        使用指定单位unit来转换obj。

        如果obj是一个序列，则返回转换后的序列。
        """
        if np.iterable(obj):
            return [o.value(unit) for o in obj]  # 对象是一个序列，返回序列中每个对象使用指定单位后的值
        else:
            return obj.value(unit)  # 对象是单个实例，返回使用指定单位后的值

    @staticmethod
    def default_units(x, axis):
        """返回x的默认单位，或者None。"""
        if np.iterable(x):
            for thisx in x:
                return thisx.unit  # 返回序列中第一个对象的默认单位
        else:
            return x.unit  # 返回单个对象的默认单位


units.registry[Foo] = FooConverter()  # 将Foo类注册到Matplotlib的单位注册表中

# 创建一些Foo对象
x = [Foo(val, 1.0) for val in range(0, 50, 2)]
# 创建一些任意的y数据
y = [i for i in range(len(x))]

fig, (ax1, ax2) = plt.subplots(1, 2)  # 创建包含两个子图的Figure对象和Axes对象
fig.suptitle("Custom units")  # 设置图的总标题
fig.subplots_adjust(bottom=0.2)  # 调整子图的布局，使得底部空间更大

# 在指定单位下绘制图像
ax2.plot(x, y, 'o', xunits=2.0)  # 在ax2上绘制以2.0为单位的x数据
ax2.set_title("xunits = 2.0")  # 设置子图ax2的标题
plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')  # 设置x轴刻度标签的旋转和对齐方式

# 在默认单位下绘制图像；将使用axisinfo中的None分支
ax1.plot(x, y)  # 使用默认单位绘制图像
ax1.set_title('default units')  # 设置子图ax1的标题
plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')  # 设置x轴刻度标签的旋转和对齐方式

plt.show()  # 显示绘制的图形
```