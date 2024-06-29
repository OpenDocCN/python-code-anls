# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\image_demo.py`

```
"""
========================
Many ways to plot images
========================

The most common way to plot images in Matplotlib is with
`~.axes.Axes.imshow`. The following examples demonstrate much of the
functionality of imshow and the many images you can create.
"""

# 导入需要的库和模块
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 NumPy 库

import matplotlib.cbook as cbook  # 导入 matplotlib 的 cbook 模块
import matplotlib.cm as cm  # 导入 matplotlib 的 cm 模块，用于颜色映射
from matplotlib.patches import PathPatch  # 导入 matplotlib 的 PathPatch 类
from matplotlib.path import Path  # 导入 matplotlib 的 Path 类

# 设置随机种子，以便结果可重复
np.random.seed(19680801)

# %%
# 首先生成一个简单的双变量正态分布。

delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)  # 生成均匀间隔的数组
X, Y = np.meshgrid(x, y)  # 生成坐标网格
Z1 = np.exp(-X**2 - Y**2)  # 第一个正态分布
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)  # 第二个正态分布
Z = (Z1 - Z2) * 2  # 合并两个分布

fig, ax = plt.subplots()  # 创建画布和坐标轴对象
im = ax.imshow(Z, interpolation='bilinear', cmap=cm.RdYlGn,  # 显示 Z 数据的图像，使用双线性插值，颜色映射为 RdYlGn
               origin='lower', extent=[-3, 3, -3, 3],  # 设置图像的原点在左下角，坐标范围为 [-3, 3, -3, 3]
               vmax=abs(Z).max(), vmin=-abs(Z).max())  # 设置最大和最小值来规范颜色映射

plt.show()


# %%
# 还可以显示图片。

# 一个样本图像
with cbook.get_sample_data('grace_hopper.jpg') as image_file:  # 获取样本图片数据
    image = plt.imread(image_file)  # 读取图像数据

# 另一个图像，使用 256x256 的 16 位整数
w, h = 256, 256
with cbook.get_sample_data('s1045.ima.gz') as datafile:  # 获取样本数据文件
    s = datafile.read()  # 读取数据文件内容
A = np.frombuffer(s, np.uint16).astype(float).reshape((w, h))  # 将数据转换为 NumPy 数组，设置数据类型和形状
extent = (0, 25, 0, 25)  # 设置图像的显示范围

fig, ax = plt.subplot_mosaic([
    ['hopper', 'mri']  # 创建包含子图 'hopper' 和 'mri' 的 subplot 布局
], figsize=(7, 3.5))  # 设置画布大小

ax['hopper'].imshow(image)  # 在 'hopper' 子图上显示图像
ax['hopper'].axis('off')  # 清除 x 轴和 y 轴

im = ax['mri'].imshow(A, cmap=plt.cm.hot, origin='upper', extent=extent)  # 在 'mri' 子图上显示 A 数组，使用热量图，设置数据原点和显示范围

markers = [(15.9, 14.5), (16.8, 15)]  # 设置标记的位置
x, y = zip(*markers)
ax['mri'].plot(x, y, 'o')  # 在 'mri' 子图上绘制圆形标记

ax['mri'].set_title('MRI')  # 设置 'mri' 子图的标题

plt.show()


# %%
# 插值图像
# --------------------
#
# 在显示图像之前，也可以对图像进行插值。注意，这可能会改变数据的外观，但有助于实现所需的视觉效果。下面我们将显示同一个（小）数组，
# 使用三种不同的插值方法进行插值。
#
# 如果使用 interpolation='nearest'，则在 (i, j) 和 (i+1, j+1) 边界的区域将具有相同的颜色。如果使用插值，
# 则像素中心的颜色与最近邻的像素相同，但其他像素将在相邻像素之间进行插值。
#
# 在进行插值时，为了防止边缘效应，Matplotlib 在输入数组的边缘填充相同的像素：例如，对于如下的 5x5 数组，
# 如果颜色为 a-y：
#
#   a b c d e
#   f g h i j
#   k l m n o
#   p q r s t
#   u v w x y
#
# Matplotlib 在填充的数组上计算插值和调整大小：
#
#   a a b c d e e
#   a a b c d e e
#   f f g h i j j
#   k k l m n o o
#   p p q r s t t
#   o u v w x y y
#   o u v w x y y
#
# 然后提取结果的中心区域。（极其老版本
# 创建一个 5x5 的随机数组 A，用于显示在图像中
A = np.random.rand(5, 5)

# 创建一个包含 1 行 3 列的图像布局，每列使用不同的插值方法进行图像显示
fig, axs = plt.subplots(1, 3, figsize=(10, 3))

# 遍历 axs 数组和对应的插值方法列表
for ax, interp in zip(axs, ['nearest', 'bilinear', 'bicubic']):
    # 在当前轴上显示数组 A，使用指定的插值方法
    ax.imshow(A, interpolation=interp)
    # 设置当前轴的标题，首字母大写显示插值方法名称
    ax.set_title(interp.capitalize())
    # 打开当前轴的网格显示
    ax.grid(True)

# 显示图像
plt.show()


# %%
# 可以通过 origin 参数指定图像的起始点位置：左上角（upper）或者左下角（lower）。
# 还可以通过设置 image.origin 在 matplotlibrc 文件中控制默认设置。
# 更多关于这个主题的信息，请参阅原点和范围的完整指南。

# 创建一个 10x12 的数组 x
x = np.arange(120).reshape((10, 12))

# 设置插值方法为 'bilinear'
interp = 'bilinear'

# 创建包含 2 行的图像布局，共享 x 轴，指定图像尺寸为 (3, 5)
fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(3, 5))

# 设置第一个子图的标题
axs[0].set_title('blue should be up')

# 在第一个子图中显示数组 x，设置起始点为上方，使用 'bilinear' 插值方法
axs[0].imshow(x, origin='upper', interpolation=interp)

# 设置第二个子图的标题
axs[1].set_title('blue should be down')

# 在第二个子图中显示数组 x，设置起始点为下方，使用 'bilinear' 插值方法
axs[1].imshow(x, origin='lower', interpolation=interp)

# 显示图像
plt.show()


# %%
# 最后，我们将展示使用裁剪路径显示图像。

# 定义 delta 和 x、y 坐标数组
delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)

# 计算 Z1 和 Z2
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)

# 计算 Z，通过 Z1 和 Z2 的差值乘以 2
Z = (Z1 - Z2) * 2

# 定义路径对象 path，表示一个正方形路径
path = Path([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 1]])

# 创建路径补丁对象 patch，设定其属性为无填充色
patch = PathPatch(path, facecolor='none')

# 创建图像和轴对象
fig, ax = plt.subplots()

# 将路径补丁添加到轴上
ax.add_patch(patch)

# 在轴上显示 Z 数组的灰度图像，使用 'bilinear' 插值方法和灰度色彩映射
# 设置起始点为下方，图像范围为 [-3, 3, -3, 3]，裁剪路径为 patch 对象
im = ax.imshow(Z, interpolation='bilinear', cmap=cm.gray,
               origin='lower', extent=[-3, 3, -3, 3],
               clip_path=patch, clip_on=True)

# 设置图像的裁剪路径为 patch 对象
im.set_clip_path(patch)

# 显示图像
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
#    - `matplotlib.artist.Artist.set_clip_path`
#    - `matplotlib.patches.PathPatch`
```