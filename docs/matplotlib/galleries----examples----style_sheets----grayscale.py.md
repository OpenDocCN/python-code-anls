# `D:\src\scipysrc\matplotlib\galleries\examples\style_sheets\grayscale.py`

```py
"""
=====================
Grayscale style sheet
=====================

This example demonstrates the "grayscale" style sheet, which changes all colors
that are defined as `.rcParams` to grayscale. Note, however, that not all
plot elements respect `.rcParams`.
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于生成数据和数组操作
import numpy as np

# 设定随机种子以便结果可重现
np.random.seed(19680801)

# 定义函数 color_cycle_example，接受一个 Axes 对象 ax 作为参数
def color_cycle_example(ax):
    # 生成从 0 到 L 的等间隔数字，共 L+1 个点
    L = 6
    x = np.linspace(0, L)
    # 获取当前设置中的颜色循环中的颜色数目
    ncolors = len(plt.rcParams['axes.prop_cycle'])
    # 生成一个等间距的数组，用于每个曲线的偏移量
    shift = np.linspace(0, L, ncolors, endpoint=False)
    # 对每个偏移量 s，绘制 sin(x + s) 的曲线
    for s in shift:
        ax.plot(x, np.sin(x + s), 'o-')

# 定义函数 image_and_patch_example，接受一个 Axes 对象 ax 作为参数
def image_and_patch_example(ax):
    # 在 Axes 对象上显示一个随机生成的 20x20 的灰度图像
    ax.imshow(np.random.random(size=(20, 20)), interpolation='none')
    # 在 Axes 对象上添加一个圆形补丁对象
    c = plt.Circle((5, 5), radius=5, label='patch')
    ax.add_patch(c)

# 设定当前图形的样式为 'grayscale'，以使用灰度风格
plt.style.use('grayscale')

# 创建一个包含两个子图的图形对象 fig，返回每个子图对应的 Axes 对象 ax1 和 ax2
fig, (ax1, ax2) = plt.subplots(ncols=2)
# 设置整个图形的标题
fig.suptitle("'grayscale' style sheet")

# 在第一个子图 ax1 上演示颜色循环的例子
color_cycle_example(ax1)
# 在第二个子图 ax2 上演示图像和补丁的例子
image_and_patch_example(ax2)

# 显示绘制的图形
plt.show()
```