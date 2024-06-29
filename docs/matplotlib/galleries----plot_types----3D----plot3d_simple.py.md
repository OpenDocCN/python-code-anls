# `D:\src\scipysrc\matplotlib\galleries\plot_types\3D\plot3d_simple.py`

```py
"""
================
plot(xs, ys, zs)
================

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot`.
"""

# 导入 matplotlib.pyplot 模块，用于绘图操作
import matplotlib.pyplot as plt
# 导入 numpy 模块，用于数值计算
import numpy as np

# 使用指定的风格样式 '_mpl-gallery'
plt.style.use('_mpl-gallery')

# 创建数据
n = 100
# 在 [0, 1] 区间生成包含 n 个元素的等间距数组作为 x 坐标
xs = np.linspace(0, 1, n)
# 计算 ys 数组，其值为 xs 数组每个元素乘以 6π 后的正弦值
ys = np.sin(xs * 6 * np.pi)
# 计算 zs 数组，其值为 xs 数组每个元素乘以 6π 后的余弦值
zs = np.cos(xs * 6 * np.pi)

# 创建图形和坐标轴对象，指定为 3D 投影
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# 在 3D 坐标轴上绘制曲线，xs 对应 x 坐标，ys 对应 y 坐标，zs 对应 z 坐标
ax.plot(xs, ys, zs)

# 设置坐标轴的刻度标签为空列表，即不显示刻度标签
ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

# 显示图形
plt.show()
```