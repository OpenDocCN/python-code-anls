# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\ndimage\3D_binary_structure.py`

```
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import scipy.ndimage  # 导入 scipy.ndimage 库，用于处理图像数据


def plot_voxels(varray, ax, title):
    ax.view_init(20, 200)  # 设置视角
    ax.voxels(varray, edgecolor="k")  # 在坐标轴上绘制体素图
    ax.set_title(title, fontsize=30)  # 设置图表标题


fig = plt.figure(figsize=(16, 9))  # 创建一个大小为 16x9 的图形对象

for i in [1, 2, 3]:
    ax = fig.add_subplot(1, 3, i, projection="3d")  # 添加一个三维子图到图形对象中
    arrray = scipy.ndimage.generate_binary_structure(3, i)  # 生成一个指定连接度的三维二进制结构
    plot_voxels(arrray, ax, title=f"rank=3 \n connectivity={i}")  # 调用函数绘制体素图，并设置标题

plt.tight_layout()  # 调整子图的布局，使其紧凑显示
plt.show()  # 显示绘制的图形
```