# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\wire3d_zero_stride.py`

```py
"""
===================================
3D wireframe plots in one direction
===================================

Demonstrates that setting *rstride* or *cstride* to 0 causes wires to not be
generated in the corresponding direction.
"""

# 导入 matplotlib 库中的 pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt

# 从 mpl_toolkits.mplot3d 模块中导入 axes3d 类
from mpl_toolkits.mplot3d import axes3d

# 创建一个 2x1 的图形窗口，包含两个子图，fig 是整个图形对象，ax1 和 ax2 是两个子图对象
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 12), subplot_kw={'projection': '3d'})

# 获取测试数据 X, Y, Z，这些数据用于绘制 3D 图形
X, Y, Z = axes3d.get_test_data(0.05)

# 在第一个子图上绘制线框图，设置 x 方向步长为 10，y 方向步长为 0，表示只沿着 x 方向绘制线框
ax1.plot_wireframe(X, Y, Z, rstride=10, cstride=0)
ax1.set_title("Column (x) stride set to 0")  # 设置第一个子图的标题

# 在第二个子图上绘制线框图，设置 x 方向步长为 0，y 方向步长为 10，表示只沿着 y 方向绘制线框
ax2.plot_wireframe(X, Y, Z, rstride=0, cstride=10)
ax2.set_title("Row (y) stride set to 0")  # 设置第二个子图的标题

# 调整子图之间的布局，使它们之间的空白最小化
plt.tight_layout()

# 显示绘制的图形
plt.show()
```