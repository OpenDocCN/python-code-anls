# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\barb_demo.py`

```py
"""
==========
Wind Barbs
==========

Demonstration of wind barb plots.
"""
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，并重命名为 np
import numpy as np

# 生成一个从 -5 到 5 的包含 5 个元素的等间隔数组
x = np.linspace(-5, 5, 5)
# 创建二维网格，X 和 Y 分别是以 x 为坐标轴的网格
X, Y = np.meshgrid(x, x)
# 根据网格坐标计算风向矢量的分量
U, V = 12 * X, 12 * Y

# 定义风向矢量的数据，包括位置 (x, y) 和矢量分量 (u, v)
data = [(-1.5, .5, -6, -6),
        (1, -1, -46, 46),
        (-3, -1, 11, -11),
        (1, 1.5, 80, 80),
        (0.5, 0.25, 25, 15),
        (-1.5, -0.5, -5, 40)]

# 将数据转换为 numpy 数组，指定数据类型为浮点数
data = np.array(data, dtype=[('x', np.float32), ('y', np.float32),
                             ('u', np.float32), ('v', np.float32)])

# 创建包含子图的 Figure 对象和 Axes 对象数组
fig1, axs1 = plt.subplots(nrows=2, ncols=2)

# 在第一个子图中绘制默认参数下的风羽图
axs1[0, 0].barbs(X, Y, U, V)

# 在第二个子图中绘制指定参数的风羽图，包括风矢长度和旋转中心
axs1[0, 1].barbs(
    data['x'], data['y'], data['u'], data['v'], length=8, pivot='middle')

# 在第三个子图中展示使用均匀网格的颜色映射效果，包括空风羽的填充和大小参数
axs1[1, 0].barbs(
    X, Y, U, V, np.sqrt(U ** 2 + V ** 2), fill_empty=True, rounding=False,
    sizes=dict(emptybarb=0.25, spacing=0.2, height=0.3))

# 在第四个子图中改变风羽的颜色和增量设置
axs1[1, 1].barbs(data['x'], data['y'], data['u'], data['v'], flagcolor='r',
                 barbcolor=['b', 'g'], flip_barb=True,
                 barb_increments=dict(half=10, full=20, flag=100))

# 创建一个支持掩码数组的风羽图，用于处理数据中的异常值
masked_u = np.ma.masked_array(data['u'])
masked_u[4] = 1000  # 将第五个值标记为异常值，不显示在图中
masked_u[4] = np.ma.masked

# 创建第二个图形，与第一个图的第二个子图相同，但点 (0.5, 0.25) 被掩码隐藏
fig2, ax2 = plt.subplots()
ax2.barbs(data['x'], data['y'], masked_u, data['v'], length=8, pivot='middle')

# 显示绘图结果
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.barbs` / `matplotlib.pyplot.barbs`
```