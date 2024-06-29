# `D:\src\scipysrc\matplotlib\galleries\users_explain\artists\color_cycle.py`

```
"""
.. redirect-from:: /tutorials/intermediate/color_cycle

.. _color_cycle:

===================
Styling with cycler
===================

Demo of custom property-cycle settings to control colors and other style
properties for multi-line plots.

.. note::

    More complete documentation of the ``cycler`` API can be found
    `here <https://matplotlib.org/cycler/>`_.

This example demonstrates two different APIs:

1. Setting the rc parameter specifying the default property cycle.
   This affects all subsequent Axes (but not Axes already created).
2. Setting the property cycle for a single pair of Axes.

"""
from cycler import cycler  # 导入 cycler 模块

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块
import numpy as np  # 导入 numpy 模块

# %%
# 首先生成一些示例数据，这里生成四条偏移的正弦曲线。
x = np.linspace(0, 2 * np.pi, 50)  # 生成从 0 到 2π 的50个点的等间隔数组
offsets = np.linspace(0, 2 * np.pi, 4, endpoint=False)  # 生成四个偏移量数组
yy = np.transpose([np.sin(x + phi) for phi in offsets])  # 生成四条偏移正弦曲线组成的矩阵

# %%
# 现在 ``yy`` 的形状为
print(yy.shape)

# %%
# 因此，``yy[:, i]`` 将给出第 ``i`` 条偏移的正弦曲线。接下来使用 :func:`matplotlib.pyplot.rc` 设置默认的 ``prop_cycle``。
# 我们将通过将两个 ``cycler`` 相加（``+``）来结合颜色和线型的循环器。
# 关于如何结合不同的循环器，可以参考本教程末尾的说明。
default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
                  cycler(linestyle=['-', '--', ':', '-.']))

plt.rc('lines', linewidth=4)  # 设置线宽度
plt.rc('axes', prop_cycle=default_cycler)  # 设置坐标轴的属性循环器

# %%
# 现在我们将生成一个包含两个 Axes 的图形，一个在另一个上面。
# 在第一个轴上，我们将使用默认的循环器绘制图形。
# 在第二个轴上，我们将使用 :func:`matplotlib.axes.Axes.set_prop_cycle` 来设置 ``prop_cycle``，
# 这只会影响当前的 :mod:`matplotlib.axes.Axes` 实例。
# 我们将使用另一个 ``cycler`` 结合颜色和线宽的循环器。
custom_cycler = (cycler(color=['c', 'm', 'y', 'k']) +
                 cycler(lw=[1, 2, 3, 4]))

fig, (ax0, ax1) = plt.subplots(nrows=2)  # 创建包含两个子图的图形
ax0.plot(yy)  # 在第一个轴上绘制 yy 的曲线
ax0.set_title('Set default color cycle to rgby')  # 设置第一个轴的标题
ax1.set_prop_cycle(custom_cycler)  # 设置第二个轴的属性循环器
ax1.plot(yy)  # 在第二个轴上绘制 yy 的曲线
ax1.set_title('Set axes color cycle to cmyk')  # 设置第二个轴的标题

# 增加两个子图之间的间距
fig.subplots_adjust(hspace=0.3)
plt.show()

# %%
# 在 :file:`matplotlibrc` 文件或样式文件中设置 ``prop_cycle``
# -------------------------------------------------------------
#
# 记住，可以在 :file:`matplotlibrc` 文件或样式文件（如 :file:`style.mplstyle`）中设置自定义的循环器，
# 在 ``axes.prop_cycle`` 下：
#
# .. code-block:: python
#
#    axes.prop_cycle : cycler(color='bgrcmyk')
#
# 循环多个属性
# -----------------------------------
#
# 可以添加循环器：
#
# .. code-block:: python
#
#    from cycler import cycler
#    cc = (cycler(color=list('rgb')) +
#          cycler(linestyle=['-', '--', '-.']))
#    for d in cc:
#        print(d)
#
# 结果为：
#
# .. code-block:: python
#
# 创建一个循环器对象 `cycler`，定义了三种颜色和三种线条样式的组合
cc = (cycler(color=list('rgb')) *
      cycler(linestyle=['-', '--', '-.']))

# 使用循环遍历组合后的循环器 `cc`，打印每个组合的结果
for d in cc:
    print(d)
```