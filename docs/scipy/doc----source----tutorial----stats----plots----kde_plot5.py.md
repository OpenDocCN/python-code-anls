# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\kde_plot5.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from scipy import stats  # 导入 SciPy 库中的统计模块
import matplotlib.pyplot as plt  # 导入 Matplotlib 库的绘图模块


def measure(n):
    """测量模型，返回两个关联的测量值。"""
    m1 = np.random.normal(size=n)  # 生成服从标准正态分布的随机数数组 m1
    m2 = np.random.normal(scale=0.5, size=n)  # 生成标准差为 0.5 的正态分布随机数数组 m2
    return m1+m2, m1-m2  # 返回 m1 + m2 和 m1 - m2


m1, m2 = measure(2000)  # 调用 measure 函数生成 2000 个测量值 m1 和 m2
xmin = m1.min()  # 计算 m1 数组中的最小值
xmax = m1.max()  # 计算 m1 数组中的最大值
ymin = m2.min()  # 计算 m2 数组中的最小值
ymax = m2.max()  # 计算 m2 数组中的最大值

# 生成二维网格 X 和 Y，用于绘制密度图
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])  # 将 X 和 Y 的网格点展开成一维数组
values = np.vstack([m1, m2])  # 将 m1 和 m2 数组堆叠成二维数组
kernel = stats.gaussian_kde(values)  # 使用 m1 和 m2 构建高斯核密度估计对象
Z = np.reshape(kernel.evaluate(positions).T, X.shape)  # 计算核密度估计并重塑成与 X 形状相同的数组

fig = plt.figure(figsize=(8, 6))  # 创建一个大小为 8x6 的新图形
ax = fig.add_subplot(111)  # 在图形中添加一个子图

# 在子图中显示密度估计的热图，使用反转后的 Z 值，并设置颜色映射和坐标范围
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])

# 绘制 m1 和 m2 的散点图，颜色为黑色，点的大小为 2
ax.plot(m1, m2, 'k.', markersize=2)

# 设置子图的 x 和 y 轴限制范围为 xmin 到 xmax，ymin 到 ymax
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

plt.show()  # 显示绘制的图形
```