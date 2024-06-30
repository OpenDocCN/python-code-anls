# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\optimize_global_2.py`

```
# 导入NumPy库并重命名为np，导入Matplotlib的pyplot模块并重命名为plt
import numpy as np
import matplotlib.pyplot as plt

# 定义一个函数eggholder，接受一个包含两个元素的数组x作为输入，计算并返回特定函数的值
def eggholder(x):
    # 计算eggholder函数的值
    return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
            - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

# 定义二维边界范围列表
bounds = [(-512, 512), (-512, 512)]

# 生成从-512到512的数组x和y
x = np.arange(-512, 513)
y = np.arange(-512, 513)

# 生成x和y的网格坐标矩阵
xgrid, ygrid = np.meshgrid(x, y)

# 将xgrid和ygrid堆叠成一个三维数组xy，其中xy[0]是xgrid，xy[1]是ygrid
xy = np.stack([xgrid, ygrid])

# 创建一个新的Matplotlib图形对象，设置尺寸为6x4英寸
fig = plt.figure(figsize=(6, 4))

# 在图形上添加一个3D子图，投影方式为3D
ax = fig.add_subplot(111, projection='3d')

# 设置3D图的视角，初始角度为视角仰角45度，方位角-45度
ax.view_init(45, -45)

# 绘制3D表面图，x轴为xgrid，y轴为ygrid，z轴为eggholder函数在xy点集上的值，使用'terrain'颜色映射
ax.plot_surface(xgrid, ygrid, eggholder(xy), cmap='terrain')

# 设置x轴标签
ax.set_xlabel('x')

# 设置y轴标签
ax.set_ylabel('y')

# 设置z轴标签
ax.set_zlabel('eggholder(x, y)')

# 调整图形布局使其紧凑
fig.tight_layout()

# 显示图形
plt.show()
```