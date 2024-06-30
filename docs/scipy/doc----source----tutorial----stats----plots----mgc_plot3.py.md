# `D:\src\scipysrc\scipy\doc\source\tutorial\stats\plots\mgc_plot3.py`

```
# 导入 numpy 库，并用 np 作为别名
import numpy as np
# 导入 matplotlib.pyplot 库，并用 plt 作为别名
import matplotlib.pyplot as plt

# 定义函数 mgc_plot，用于绘制 sim 和 MGC 图
def mgc_plot(x, y, sim_name):
    """Plot sim and MGC-plot"""
    # 创建一个大小为 8x8 的新图像
    plt.figure(figsize=(8, 8))
    # 获取当前的坐标轴
    ax = plt.gca()
    # 设置图像标题，包括 sim_name 和 "Simulation"
    ax.set_title(sim_name + " Simulation", fontsize=20)
    # 绘制散点图，x 为横坐标，y 为纵坐标
    ax.scatter(x, y)
    # 设置 x 轴标签为 'X'，字体大小为 15
    ax.set_xlabel('X', fontsize=15)
    # 设置 y 轴标签为 'Y'，字体大小为 15
    ax.set_ylabel('Y', fontsize=15)
    # 设置坐标轴等比例显示
    ax.axis('equal')
    # 设置 x 轴刻度标签的字体大小为 15
    ax.tick_params(axis="x", labelsize=15)
    # 设置 y 轴刻度标签的字体大小为 15
    ax.tick_params(axis="y", labelsize=15)
    # 显示图像
    plt.show()

# 使用 numpy 的默认随机数生成器创建一个随机数生成器对象 rng
rng = np.random.default_rng()
# 生成一个包含 100 个从 0 到 5 均匀分布的随机数的数组
unif = np.array(rng.uniform(0, 5, size=100))
# 根据生成的随机数数组，计算 x 坐标
x = unif * np.cos(np.pi * unif)
# 根据生成的随机数数组，计算 y 坐标，并添加一些随机噪声
y = unif * np.sin(np.pi * unif) + 0.4 * rng.random(x.size)

# 调用 mgc_plot 函数，绘制 x 和 y 的散点图，设置图像标题为 "Spiral" Simulation
mgc_plot(x, y, "Spiral")
```