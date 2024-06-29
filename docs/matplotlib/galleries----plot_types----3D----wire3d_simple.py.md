# `D:\src\scipysrc\matplotlib\galleries\plot_types\3D\wire3d_simple.py`

```
"""
=======================
plot_wireframe(X, Y, Z)
=======================

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_wireframe`.
"""
# 导入 matplotlib 的 pyplot 模块，并简写为 plt
import matplotlib.pyplot as plt

# 从 mpl_toolkits.mplot3d 模块中导入 axes3d 子模块
from mpl_toolkits.mplot3d import axes3d

# 使用自定义的 matplotlib 风格 '_mpl-gallery'
plt.style.use('_mpl-gallery')

# 使用 axes3d 模块提供的函数生成测试数据 X, Y, Z
X, Y, Z = axes3d.get_test_data(0.05)

# 创建图形窗口和 3D 坐标轴
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# 在 3D 坐标轴上绘制线框图
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

# 设置坐标轴的刻度标签为空列表，即不显示刻度标签
ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

# 显示绘制的图形
plt.show()
```