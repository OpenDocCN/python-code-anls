# `D:\src\scipysrc\matplotlib\galleries\examples\pie_and_polar_charts\polar_error_caps.py`

```
"""
=================================
Error bar rendering on polar axis
=================================

Demo of error bar plot in polar coordinates.
Theta error bars are curved lines ended with caps oriented towards the
center.
Radius error bars are straight lines oriented towards center with
perpendicular caps.
"""
# 导入 matplotlib 的 pyplot 模块，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于数值计算
import numpy as np

# 创建角度数组，从 0 到 2π，步长为 π/4
theta = np.arange(0, 2 * np.pi, np.pi / 4)
# 创建半径数组，根据角度计算半径值
r = theta / np.pi / 2 + 0.5

# 创建 10x10 英寸大小的图像对象
fig = plt.figure(figsize=(10, 10))
# 添加极坐标子图
ax = fig.add_subplot(projection='polar')
# 绘制带有误差棒的散点图到极坐标子图上
ax.errorbar(theta, r, xerr=0.25, yerr=0.1, capsize=7, fmt="o", c="seagreen")
# 设置子图标题
ax.set_title("Pretty polar error bars")
# 显示图像
plt.show()

# %%
# 提示：大的角度误差棒会发生重叠，可能降低输出图像的可读性。请参考以下示例图：

# 创建 10x10 英寸大小的图像对象
fig = plt.figure(figsize=(10, 10))
# 添加极坐标子图
ax = fig.add_subplot(projection='polar')
# 绘制带有大角度误差棒的散点图到极坐标子图上
ax.errorbar(theta, r, xerr=5.25, yerr=0.1, capsize=7, fmt="o", c="darkred")
# 设置子图标题
ax.set_title("Overlapping theta error bars")
# 显示图像
plt.show()

# %%
# 另一方面，大的半径误差棒不会重叠，但会导致数据范围不理想，降低显示范围。

# 创建 10x10 英寸大小的图像对象
fig = plt.figure(figsize=(10, 10))
# 添加极坐标子图
ax = fig.add_subplot(projection='polar')
# 绘制带有大半径误差棒的散点图到极坐标子图上
ax.errorbar(theta, r, xerr=0.25, yerr=10.1, capsize=7, fmt="o", c="orangered")
# 设置子图标题
ax.set_title("Large radius error bars")
# 显示图像
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.errorbar` / `matplotlib.pyplot.errorbar`
#    - `matplotlib.projections.polar`
```