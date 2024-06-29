# `D:\src\scipysrc\matplotlib\galleries\examples\ticks\fig_axes_customize_simple.py`

```py
# 导入 matplotlib.pyplot 模块作为 plt 别名
import matplotlib.pyplot as plt

# %%
# 创建一个新的图形实例，返回 matplotlib.figure.Figure 对象
fig = plt.figure()
# 获取图形的矩形背景（patch）并设置其背景颜色为 'lightgoldenrodyellow'
rect = fig.patch  # a rectangle instance
rect.set_facecolor('lightgoldenrodyellow')

# 在图形上添加一个新的坐标轴 ax1，指定其位置和大小
ax1 = fig.add_axes([0.1, 0.3, 0.4, 0.4])
# 获取坐标轴的矩形背景（patch）并设置其背景颜色为 'lightslategray'
rect = ax1.patch
rect.set_facecolor('lightslategray')

# 设置坐标轴 ax1 的 x 轴刻度参数：标签颜色为 'tab:red'，标签旋转角度为 45 度，标签大小为 16
ax1.tick_params(axis='x', labelcolor='tab:red', labelrotation=45, labelsize=16)
# 设置坐标轴 ax1 的 y 轴刻度参数：刻度线颜色为 'tab:green'，刻度线长度为 25，刻度线宽度为 3
ax1.tick_params(axis='y', color='tab:green', size=25, width=3)

# 显示绘制的图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axis.Axis.get_ticklabels`
#    - `matplotlib.axis.Axis.get_ticklines`
#    - `matplotlib.text.Text.set_rotation`
#    - `matplotlib.text.Text.set_fontsize`
#    - `matplotlib.text.Text.set_color`
#    - `matplotlib.lines.Line2D`
#    - `matplotlib.lines.Line2D.set_markeredgecolor`
#    - `matplotlib.lines.Line2D.set_markersize`
#    - `matplotlib.lines.Line2D.set_markeredgewidth`
#    - `matplotlib.patches.Patch.set_facecolor`
```