# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\mgc_plot1.py`

```
# 导入 NumPy 库，简称为 np，用于数值计算
import numpy as np
# 导入 Matplotlib 库中的 pyplot 模块，简称为 plt，用于绘图
import matplotlib.pyplot as plt

# 定义函数 mgc_plot，用于绘制相似性和 MGC 图
def mgc_plot(x, y, sim_name):
    """Plot sim and MGC-plot"""
    # 创建一个 8x8 英寸大小的新图
    plt.figure(figsize=(8, 8))
    # 获取当前图的坐标轴
    ax = plt.gca()
    # 设置图的标题，包括传入的 sim_name 参数，设置字体大小为 20
    ax.set_title(sim_name + " Simulation", fontsize=20)
    # 绘制散点图，x 为横坐标，y 为纵坐标
    ax.scatter(x, y)
    # 设置 x 轴标签为 'X'，字体大小为 15
    ax.set_xlabel('X', fontsize=15)
    # 设置 y 轴标签为 'Y'，字体大小为 15
    ax.set_ylabel('Y', fontsize=15)
    # 设置坐标轴的等比例缩放
    ax.axis('equal')
    # 设置 x 轴刻度标签的大小为 15
    ax.tick_params(axis="x", labelsize=15)
    # 设置 y 轴刻度标签的大小为 15
    ax.tick_params(axis="y", labelsize=15)
    # 显示绘制的图形
    plt.show()

# 创建一个新的随机数生成器实例
rng = np.random.default_rng()
# 生成一个包含 100 个元素的数组，元素范围在 -1 到 1 之间
x = np.linspace(-1, 1, num=100)
# 根据 x 数组生成对应的 y 数组，加入一定的随机扰动
y = x + 0.3 * rng.random(x.size)

# 调用 mgc_plot 函数，传入生成的 x 和 y 数组，以及字符串 "Linear" 作为 sim_name 参数
mgc_plot(x, y, "Linear")
```