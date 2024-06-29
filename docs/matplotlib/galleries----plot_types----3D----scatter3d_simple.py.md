# `D:\src\scipysrc\matplotlib\galleries\plot_types\3D\scatter3d_simple.py`

```
"""
===================
scatter(xs, ys, zs)
===================

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.scatter`.
"""
# 导入 matplotlib.pyplot 库，简称 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，简称 np
import numpy as np

# 使用 '_mpl-gallery' 风格样式
plt.style.use('_mpl-gallery')

# 生成数据
np.random.seed(19680801)  # 设定随机种子以便复现随机结果
n = 100  # 数据点个数
rng = np.random.default_rng()  # 创建一个随机数生成器实例
xs = rng.uniform(23, 32, n)  # 在区间 [23, 32) 内生成 n 个均匀分布的随机数作为 x 坐标
ys = rng.uniform(0, 100, n)  # 在区间 [0, 100) 内生成 n 个均匀分布的随机数作为 y 坐标
zs = rng.uniform(-50, -25, n)  # 在区间 [-50, -25) 内生成 n 个均匀分布的随机数作为 z 坐标

# 绘制散点图
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})  # 创建一个 3D 图形和坐标轴对象
ax.scatter(xs, ys, zs)  # 在 3D 坐标轴上绘制散点图，传入 x, y, z 坐标数据

# 设置坐标轴标签为空列表，即不显示刻度标签
ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()  # 显示图形
```