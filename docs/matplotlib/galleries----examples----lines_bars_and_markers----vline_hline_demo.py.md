# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\vline_hline_demo.py`

```py
"""
=================
hlines and vlines
=================

This example showcases the functions hlines and vlines.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

# 设置随机种子以便结果可重复
np.random.seed(19680801)

t = np.arange(0.0, 5.0, 0.1)  # 生成从0到5（不包括）的间隔为0.1的数组
s = np.exp(-t) + np.sin(2 * np.pi * t) + 1  # 计算指数衰减加正弦波的复合函数
nse = np.random.normal(0.0, 0.3, t.shape) * s  # 生成符合正态分布的噪声数据，乘以s数组

fig, (vax, hax) = plt.subplots(1, 2, figsize=(12, 6))  # 创建一个包含两个子图的图像对象

vax.plot(t, s + nse, '^')  # 在vax子图中绘制散点图，'^'表示使用三角形标记
vax.vlines(t, [0], s)  # 在vax子图中绘制垂直线，覆盖从0到s的高度
# 通过设置``transform=vax.get_xaxis_transform()``，y坐标按比例缩放，0映射到Axes底部，1映射到顶部。
vax.vlines([1, 2], 0, 1, transform=vax.get_xaxis_transform(), colors='r')  # 在vax子图中绘制红色垂直线
vax.set_xlabel('time (s)')  # 设置x轴标签
vax.set_title('Vertical lines demo')  # 设置子图标题

hax.plot(s + nse, t, '^')  # 在hax子图中绘制散点图，'^'表示使用三角形标记
hax.hlines(t, [0], s, lw=2)  # 在hax子图中绘制水平线，覆盖从0到s的宽度，线宽为2
hax.set_xlabel('time (s)')  # 设置x轴标签
hax.set_title('Horizontal lines demo')  # 设置子图标题

plt.show()  # 显示图像
```