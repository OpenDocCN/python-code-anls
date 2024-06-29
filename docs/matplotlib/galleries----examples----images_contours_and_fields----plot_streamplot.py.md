# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\plot_streamplot.py`

```
"""
==========
Streamplot
==========

A stream plot, or streamline plot, is used to display 2D vector fields. This
example shows a few features of the `~.axes.Axes.streamplot` function:

* Varying the color along a streamline.
* Varying the density of streamlines.
* Varying the line width along a streamline.
* Controlling the starting points of streamlines.
* Streamlines skipping masked regions and NaN values.
* Unbroken streamlines even when exceeding the limit of lines within a single
  grid cell.
"""
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库
import numpy as np  # 导入 numpy 库

w = 3  # 设置 w 的值为 3
Y, X = np.mgrid[-w:w:100j, -w:w:100j]  # 生成一个网格，Y 和 X 均为 100x100 的数组
U = -1 - X**2 + Y  # 计算向量场的 U 分量
V = 1 + X - Y**2  # 计算向量场的 V 分量
speed = np.sqrt(U**2 + V**2)  # 计算速度大小

fig, axs = plt.subplots(3, 2, figsize=(7, 9), height_ratios=[1, 1, 2])  # 创建一个 3x2 的图像布局
axs = axs.flat  # 展开子图数组为一个迭代器

# Varying density along a streamline
axs[0].streamplot(X, Y, U, V, density=[0.5, 1])  # 绘制密度变化的流线图
axs[0].set_title('Varying Density')  # 设置子图标题

# Varying color along a streamline
strm = axs[1].streamplot(X, Y, U, V, color=U, linewidth=2, cmap='autumn')  # 绘制颜色变化的流线图
fig.colorbar(strm.lines)  # 添加颜色条
axs[1].set_title('Varying Color')  # 设置子图标题

# Varying line width along a streamline
lw = 5*speed / speed.max()  # 根据速度大小计算线宽
axs[2].streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)  # 绘制线宽变化的流线图
axs[2].set_title('Varying Line Width')  # 设置子图标题

# Controlling the starting points of the streamlines
seed_points = np.array([[-2, -1, 0, 1, 2, -1], [-2, -1,  0, 1, 2, 2]])  # 设置流线的起始点

strm = axs[3].streamplot(X, Y, U, V, color=U, linewidth=2,
                         cmap='autumn', start_points=seed_points.T)  # 绘制控制起始点的流线图
fig.colorbar(strm.lines)  # 添加颜色条
axs[3].set_title('Controlling Starting Points')  # 设置子图标题

# Displaying the starting points with blue symbols.
axs[3].plot(seed_points[0], seed_points[1], 'bo')  # 在子图中显示起始点的蓝色符号
axs[3].set(xlim=(-w, w), ylim=(-w, w))  # 设置子图的 x 和 y 范围

# Create a mask
mask = np.zeros(U.shape, dtype=bool)  # 创建一个与 U 形状相同的布尔类型数组
mask[40:60, 40:60] = True  # 在 mask 中设置一个区域为 True
U[:20, :20] = np.nan  # 将部分 U 的值设为 NaN
U = np.ma.array(U, mask=mask)  # 创建一个掩码数组

axs[4].streamplot(X, Y, U, V, color='r')  # 绘制使用掩码的流线图
axs[4].set_title('Streamplot with Masking')  # 设置子图标题

axs[4].imshow(~mask, extent=(-w, w, -w, w), alpha=0.5, cmap='gray',
              aspect='auto')  # 在子图中显示掩码的区域
axs[4].set_aspect('equal')  # 设置子图的纵横比

axs[5].streamplot(X, Y, U, V, broken_streamlines=False)  # 绘制不允许断裂的流线图
axs[5].set_title('Streamplot with unbroken streamlines')  # 设置子图标题

plt.tight_layout()  # 调整布局使子图适应画布
plt.show()  # 显示图像
# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.streamplot` / `matplotlib.pyplot.streamplot`
#    - `matplotlib.gridspec.GridSpec`
```