# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\errorbar_limits_simple.py`

```
"""
========================
Errorbar limit selection
========================

Illustration of selectively drawing lower and/or upper limit symbols on
errorbars using the parameters ``uplims``, ``lolims`` of `~.pyplot.errorbar`.

Alternatively, you can use 2xN values to draw errorbars in only one direction.
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于数值计算
import numpy as np

# 创建一个图形对象
fig = plt.figure()
# 生成一个包含 0 到 9 的数组作为 x 轴数据
x = np.arange(10)
# 计算 y 轴数据，通过正弦函数
y = 2.5 * np.sin(x / 20 * np.pi)
# 创建一个长度为 10 的数组，作为 y 轴方向的误差范围
yerr = np.linspace(0.05, 0.2, 10)

# 绘制带有默认上下限符号的误差条图形
plt.errorbar(x, y + 3, yerr=yerr, label='both limits (default)')

# 绘制只有上限符号的误差条图形
plt.errorbar(x, y + 2, yerr=yerr, uplims=True, label='uplims=True')

# 绘制既有上限符号又有下限符号的误差条图形
plt.errorbar(x, y + 1, yerr=yerr, uplims=True, lolims=True,
             label='uplims=True, lolims=True')

# 创建一个包含 True 和 False 的数组，作为上限符号的选择
upperlimits = [True, False] * 5
# 创建一个包含 True 和 False 的数组，作为下限符号的选择
lowerlimits = [False, True] * 5
# 绘制只显示部分上下限符号的误差条图形
plt.errorbar(x, y, yerr=yerr, uplims=upperlimits, lolims=lowerlimits,
             label='subsets of uplims and lolims')

# 在图形的右下角添加图例
plt.legend(loc='lower right')


# %%
# 类似地，``xuplims`` 和 ``xlolims`` 可以用于水平的 ``xerr`` 误差条。

# 创建另一个图形对象
fig = plt.figure()
# 生成一个包含 0 到 0.9 的数组，作为新的 x 轴数据
x = np.arange(10) / 10
# 计算 y 轴数据，通过平方函数
y = (x + 0.1)**2

# 绘制带有下限符号的 x 方向误差条图形
plt.errorbar(x, y, xerr=0.1, xlolims=True, label='xlolims=True')
# 计算另一组 y 轴数据，通过立方函数
y = (x + 0.1)**3

# 绘制只显示部分上下限符号的 x 方向误差条图形
plt.errorbar(x + 0.6, y, xerr=0.1, xuplims=upperlimits, xlolims=lowerlimits,
             label='subsets of xuplims and xlolims')

# 计算另一组 y 轴数据，通过四次方函数
y = (x + 0.1)**4
# 绘制只显示上限符号的 x 方向误差条图形
plt.errorbar(x + 1.2, y, xerr=0.1, xuplims=True, label='xuplims=True')

# 在图形中添加图例
plt.legend()
# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.errorbar` / `matplotlib.pyplot.errorbar`
```