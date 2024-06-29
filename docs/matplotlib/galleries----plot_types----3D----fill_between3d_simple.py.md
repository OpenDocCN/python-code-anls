# `D:\src\scipysrc\matplotlib\galleries\plot_types\3D\fill_between3d_simple.py`

```py
"""
====================================
fill_between(x1, y1, z1, x2, y2, z2)
====================================

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.fill_between`.
"""
# 导入 Matplotlib 库中的 pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 NumPy 库，并重命名为 np
import numpy as np

# 使用 '_mpl-gallery' 风格样式
plt.style.use('_mpl-gallery')

# 创建一个双螺旋线的数据
n = 50
theta = np.linspace(0, 2*np.pi, n)
x1 = np.cos(theta)
y1 = np.sin(theta)
z1 = np.linspace(0, 1, n)
x2 = np.cos(theta + np.pi)
y2 = np.sin(theta + np.pi)
z2 = z1

# 创建一个新的 3D 图形和轴
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# 在两个曲线 x1, y1, z1 和 x2, y2, z2 之间填充区域，设置透明度为 0.5
ax.fill_between(x1, y1, z1, x2, y2, z2, alpha=0.5)
# 绘制第一个螺旋线的曲线
ax.plot(x1, y1, z1, linewidth=2, color='C0')
# 绘制第二个螺旋线的曲线
ax.plot(x2, y2, z2, linewidth=2, color='C0')

# 设置坐标轴标签为空列表，隐藏坐标轴刻度标签
ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

# 显示图形
plt.show()
```