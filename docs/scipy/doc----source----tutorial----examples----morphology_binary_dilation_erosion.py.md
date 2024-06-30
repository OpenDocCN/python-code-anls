# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\morphology_binary_dilation_erosion.py`

```
# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，并使用别名 np
import numpy as np
# 导入 scipy.ndimage 库，用于图像处理
import scipy.ndimage

# 定义球形结构元素的函数 ball，创建一个球形结构元素
# radius: 球形结构元素的半径
# dtype: 结果数组的数据类型，默认为 np.uint8
def ball(radius, dtype=np.uint8):
    # 计算结构元素的尺寸
    n = 2 * radius + 1
    # 创建网格坐标，表示球形结构元素的形状
    Z, Y, X = np.mgrid[
        -radius: radius: n * 1j,
        -radius: radius: n * 1j,
        -radius: radius: n * 1j
    ]
    # 计算球形结构元素内部点的距离平方和
    s = X ** 2 + Y ** 2 + Z ** 2
    # 返回一个布尔数组，表示结构元素内部点的位置
    return np.array(s <= radius * radius, dtype=dtype)


# 定义绘制体素数据的函数 plot_voxels
# varray: 体素数据数组
# ax: 绘图对象
# title: 图像标题
def plot_voxels(varray, ax, title):
    # 设置视角
    ax.view_init(20, 200)
    # 绘制体素数据的立方体表示
    ax.voxels(varray, edgecolor="k")
    # 设置子图标题
    ax.set_title(title, fontsize=30)


# 创建一个形状为 (11, 11, 11) 的全零数组，表示体素数据
voxelarray = np.full((11, 11, 11), 0)
# 设置指定位置的体素为 1，表示体素数据的改变
voxelarray[5, 3, 5] = 1
voxelarray[5, 7, 5] = 1

# 对体素数据进行二值膨胀操作，使用球形结构元素，半径为 3
img_morphed = scipy.ndimage.binary_dilation(voxelarray, ball(3))
# 对上一步结果进行二值腐蚀操作，使用球形结构元素，半径为 2
img_morphed2 = scipy.ndimage.binary_erosion(img_morphed, ball(2))

# 创建一个新的绘图窗口
fig = plt.figure(figsize=(16, 9))
# 添加三个子图，每个子图使用 3D 投影
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax3 = fig.add_subplot(1, 3, 3, projection="3d")

# 绘制原始体素数据的立方体表示，子图标题为 "a) Original"
plot_voxels(voxelarray, ax1, title="a) Original")
# 绘制经二值膨胀处理后的体素数据的立方体表示，子图标题为 "b) binary_dilation \nwith ball, radius 3"
plot_voxels(img_morphed, ax2, title="b) binary_dilation \nwith ball, radius 3")
# 绘制经二值腐蚀处理后的体素数据的立方体表示，子图标题为 "c) binary_erosion of b \nwith ball, radius 2"
plot_voxels(img_morphed2, ax3,
            title="c) binary_erosion of b \nwith ball, radius 2")

# 调整子图布局，使得子图间的间距合适
plt.tight_layout()
# 显示绘制的图像
plt.show()
```