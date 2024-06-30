# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\gaussian_filter_plot1.py`

```
# 导入 matplotlib 中的 pyplot 模块，并命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并命名为 np
import numpy as np
# 从 scipy.ndimage 模块中导入 gaussian_filter 函数
from scipy.ndimage import gaussian_filter

# 定义三维数组的网格数
grids = 2
# 定义每个小立方体的边长
boxs = 5

# 创建一个形状为 (boxs * grids, boxs * grids, boxs * grids) 的全零数组
voxelarray = np.zeros((boxs * grids, boxs * grids, boxs * grids))

# 初始化计数器 i
i = 1
# 三重循环遍历三维数组的每个小立方体
for xi in range(0, 2):
    for yi in range(0, 2):
        for zi in range(0, 2):
            # 将当前小立方体的区域赋值为当前计数器的值 i
            voxelarray[
                xi * boxs: xi * boxs + boxs,
                yi * boxs: yi * boxs + boxs,
                zi * boxs: zi * boxs + boxs,
            ] = i
            # 计数器递增
            i += 1

# 将三维数组的值缩放到 0 到 255 的整数范围，并转换为无符号整数类型
voxelarray = np.uint8(voxelarray * 255 / 8)

# 获取名为 "YlGnBu" 的颜色映射
cmap = plt.get_cmap("YlGnBu")

# 定义一个函数 plot_voxels，用于绘制体素图
def plot_voxels(varray, ax, title):
    # 根据体素数组 varray 生成对应的颜色数组
    colors = cmap(varray)
    # 设置三维图的视角
    ax.view_init(30, 200)
    # 关闭坐标轴显示
    ax.axis("off")
    # 使用颜色数组绘制体素图
    ax.voxels(varray, facecolors=colors, edgecolor="#000000", linewidth=0.1)
    # 设置子图标题
    ax.set_title(title, fontsize=30)

# 创建一个大小为 16x9 的新图像对象
fig = plt.figure(figsize=(16, 9))
# 在图像对象中添加三个子图，分别为 1 行 3 列的布局，各自使用三维投影
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax3 = fig.add_subplot(1, 3, 3, projection="3d")

# 绘制原始体素图到第一个子图中
plot_voxels(voxelarray, ax1, title="Original")
# 对原始体素数组进行高斯滤波（sigma=1），并绘制到第二个子图中
voxelarray2 = gaussian_filter(voxelarray, sigma=1)
plot_voxels(voxelarray2, ax2, title="gaussian_filter \n sigma=1")
# 对原始体素数组进行高斯滤波（sigma=3），并绘制到第三个子图中
voxelarray3 = gaussian_filter(voxelarray, sigma=3)
plot_voxels(voxelarray3, ax3, title="gaussian_filter \n sigma=3")

# 调整子图布局，使之紧凑显示
plt.tight_layout()
# 显示图像
plt.show()
```