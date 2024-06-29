# `D:\src\scipysrc\matplotlib\galleries\examples\misc\zorder_demo.py`

```py
# 导入 matplotlib.pyplot 库，用于绘图操作
import matplotlib.pyplot as plt
# 导入 numpy 库，用于生成数据
import numpy as np

# 生成极坐标系下的数据点
r = np.linspace(0.3, 1, 30)
theta = np.linspace(0, 4*np.pi, 30)
x = r * np.sin(theta)
y = r * np.cos(theta)

# 创建一个 1x2 的图表布局，并得到两个子图对象 ax1 和 ax2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3.2))

# 在第一个子图 ax1 上绘制线条，使用颜色 'C3'，线宽为 3
ax1.plot(x, y, 'C3', lw=3)
# 在第一个子图 ax1 上绘制散点图，点大小为 120
ax1.scatter(x, y, s=120)
# 设置第一个子图 ax1 的标题为 'Lines on top of dots'
ax1.set_title('Lines on top of dots')

# 在第二个子图 ax2 上绘制线条，使用颜色 'C3'，线宽为 3
ax2.plot(x, y, 'C3', lw=3)
# 在第二个子图 ax2 上绘制散点图，点大小为 120，并设置 zorder 为 2.5，将点置于线条之上
ax2.scatter(x, y, s=120, zorder=2.5)
# 设置第二个子图 ax2 的标题为 'Dots on top of lines'
ax2.set_title('Dots on top of lines')

# 自动调整子图的布局
plt.tight_layout()

# 创建新的图表
plt.figure()
# 设置全局的线条宽度参数为 5
plt.rcParams['lines.linewidth'] = 5
# 绘制两条正弦曲线，分别设置它们的 zorder 参数为 2 和 3
plt.plot(x, np.sin(x), label='zorder=2', zorder=2)  # 底层
plt.plot(x, np.sin(x+0.5), label='zorder=3',  zorder=3)
# 添加一条水平参考线，设置其 zorder 参数为 2.5，使其在两条曲线之间
plt.axhline(0, label='zorder=2.5', color='lightgrey', zorder=2.5)
# 设置图表的标题为 'Custom order of elements'
plt.title('Custom order of elements')
# 创建并设置图例的位置为 'upper right'
l = plt.legend(loc='upper right')
# 设置图例的 zorder 参数为 2.5，使其位于蓝色和橙色曲线之间
l.set_zorder(2.5)
# 显示图表
plt.show()
```