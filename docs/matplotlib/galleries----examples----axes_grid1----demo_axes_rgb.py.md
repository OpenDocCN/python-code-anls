# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\demo_axes_rgb.py`

```
"""
==================================
Showing RGB channels using RGBAxes
==================================

`~.axes_grid1.axes_rgb.RGBAxes` creates a layout of 4 Axes for displaying RGB
channels: one large Axes for the RGB image and 3 smaller Axes for the R, G, B
channels.
"""

# 导入 matplotlib 的 pyplot 模块，并命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 模块，并命名为 np
import numpy as np
# 导入 matplotlib 的 cbook 模块
from matplotlib import cbook
# 从 mpl_toolkits.axes_grid1.axes_rgb 模块导入 RGBAxes 和 make_rgb_axes 函数
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes, make_rgb_axes


# 定义函数 get_rgb，用于获取 RGB 通道数据
def get_rgb():
    # 从样本数据中获取 "axes_grid/bivariate_normal.npy" 文件的数据
    Z = cbook.get_sample_data("axes_grid/bivariate_normal.npy")
    # 将小于 0 的值设为 0
    Z[Z < 0] = 0.
    # 数据归一化到 [0, 1] 范围
    Z = Z / Z.max()

    # 分别提取 R、G、B 通道数据
    R = Z[:13, :13]
    G = Z[2:, 2:]
    B = Z[:13, 2:]

    return R, G, B


# 定义函数 make_cube，用于生成 RGB 立方体数据
def make_cube(r, g, b):
    # 获取数据的维度大小
    ny, nx = r.shape
    # 初始化全零数组 R、G、B
    R = np.zeros((ny, nx, 3))
    # 将 r 数据填充到 R 的红色通道
    R[:, :, 0] = r
    # 初始化全零数组 G，将 g 数据填充到其绿色通道
    G = np.zeros_like(R)
    G[:, :, 1] = g
    # 初始化全零数组 B，将 b 数据填充到其蓝色通道
    B = np.zeros_like(R)
    B[:, :, 2] = b

    # 合并 R、G、B 通道数据形成 RGB 图像数据
    RGB = R + G + B

    return R, G, B, RGB


# 定义函数 demo_rgb1，展示使用 RGBAxes 绘制 RGB 通道
def demo_rgb1():
    # 创建一个新的图形窗口
    fig = plt.figure()
    # 在图形窗口中创建 RGBAxes 对象，设置位置和填充
    ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8], pad=0.0)
    # 获取 R、G、B 通道数据
    r, g, b = get_rgb()
    # 在 RGBAxes 对象中显示 RGB 图像
    ax.imshow_rgb(r, g, b)


# 定义函数 demo_rgb2，展示使用 make_rgb_axes 绘制 RGB 通道
def demo_rgb2():
    # 创建一个新的图形窗口和主轴对象 ax
    fig, ax = plt.subplots()
    # 使用 make_rgb_axes 在主轴对象 ax 上创建 R、G、B 通道的子图对象
    ax_r, ax_g, ax_b = make_rgb_axes(ax, pad=0.02)

    # 获取 R、G、B 通道数据
    r, g, b = get_rgb()
    # 创建 R、G、B 通道立方体数据及合并后的 RGB 图像数据
    im_r, im_g, im_b, im_rgb = make_cube(r, g, b)
    # 在主轴对象 ax 上分别显示 RGB 合并图像及其各个通道的子图对象上显示的图像
    ax.imshow(im_rgb)
    ax_r.imshow(im_r)
    ax_g.imshow(im_g)
    ax_b.imshow(im_b)

    # 遍历图形对象的所有子图对象，设置刻度参数和边框颜色为白色
    for ax in fig.axes:
        ax.tick_params(direction='in', color='w')
        ax.spines[:].set_color("w")


# 调用 demo_rgb1 和 demo_rgb2 函数展示 RGB 通道图像
demo_rgb1()
demo_rgb2()

# 显示图形
plt.show()
```