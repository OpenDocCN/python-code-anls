# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\figlegend_demo.py`

```
# 导入matplotlib.pyplot库，用于绘图
import matplotlib.pyplot as plt
# 导入numpy库，用于生成数据
import numpy as np

# 创建一个包含两个子图的图像对象fig和子图数组axs
fig, axs = plt.subplots(1, 2)

# 生成数据
x = np.arange(0.0, 2.0, 0.02)
y1 = np.sin(2 * np.pi * x)
y2 = np.exp(-x)

# 在第一个子图axs[0]上绘制两条曲线，返回曲线对象l1和l2
l1, = axs[0].plot(x, y1)
l2, = axs[0].plot(x, y2, marker='o')

# 生成更多数据
y3 = np.sin(4 * np.pi * x)
y4 = np.exp(-2 * x)

# 在第二个子图axs[1]上绘制两条曲线，返回曲线对象l3和l4
l3, = axs[1].plot(x, y3, color='tab:green')
l4, = axs[1].plot(x, y4, color='tab:red', marker='^')

# 在图像fig上创建图例，将l1和l2的标签设置为'Line 1'和'Line 2'，位置为左上角
fig.legend((l1, l2), ('Line 1', 'Line 2'), loc='upper left')

# 在图像fig上创建另一个图例，将l3和l4的标签设置为'Line 3'和'Line 4'，位置为右上角
fig.legend((l3, l4), ('Line 3', 'Line 4'), loc='upper right')

# 调整子图布局使其紧凑显示
plt.tight_layout()

# 显示图像
plt.show()

# %%
# 有时我们不希望图例重叠在子图上。如果使用了“constrained layout”，可以指定“outside right upper”，
# “constrained layout”会为图例腾出空间。

# 创建一个包含两个子图的图像对象fig和子图数组axs，并使用约束布局'constrained'
fig, axs = plt.subplots(1, 2, layout='constrained')

# 生成数据
x = np.arange(0.0, 2.0, 0.02)
y1 = np.sin(2 * np.pi * x)
y2 = np.exp(-x)

# 在第一个子图axs[0]上绘制两条曲线，返回曲线对象l1和l2
l1, = axs[0].plot(x, y1)
l2, = axs[0].plot(x, y2, marker='o')

# 生成更多数据
y3 = np.sin(4 * np.pi * x)
y4 = np.exp(-2 * x)

# 在第二个子图axs[1]上绘制两条曲线，返回曲线对象l3和l4
l3, = axs[1].plot(x, y3, color='tab:green')
l4, = axs[1].plot(x, y4, color='tab:red', marker='^')

# 在图像fig上创建图例，将l1和l2的标签设置为'Line 1'和'Line 2'，位置为左上角
fig.legend((l1, l2), ('Line 1', 'Line 2'), loc='upper left')

# 在图像fig上创建另一个图例，将l3和l4的标签设置为'Line 3'和'Line 4'，位置为外部右上角
fig.legend((l3, l4), ('Line 3', 'Line 4'), loc='outside right upper')

# 显示图像
plt.show()
```