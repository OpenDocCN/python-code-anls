# `D:\src\scipysrc\matplotlib\galleries\plot_types\3D\bar3d_simple.py`

```
"""
==========================
bar3d(x, y, z, dx, dy, dz)
==========================

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.bar3d`.
"""

# 导入 matplotlib.pyplot 库，并简称为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简称为 np
import numpy as np

# 使用 '_mpl-gallery' 风格
plt.style.use('_mpl-gallery')

# 定义 3D 柱状图的参数
x = [1, 1, 2, 2]
y = [1, 2, 1, 2]
z = [0, 0, 0, 0]
dx = np.ones_like(x) * 0.5
dy = np.ones_like(x) * 0.5
dz = [2, 3, 1, 4]

# 创建图形和坐标轴
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# 绘制 3D 柱状图
ax.bar3d(x, y, z, dx, dy, dz)

# 设置坐标轴的刻度标签为空列表，隐藏刻度
ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

# 显示图形
plt.show()
```