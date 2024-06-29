# `D:\src\scipysrc\matplotlib\galleries\examples\animation\animate_decay.py`

```py
"""
=====
Decay
=====

This example showcases:

- using a generator to drive an animation,
- changing axes limits during an animation.

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

# 导入必要的库
import itertools  # 导入 itertools 库，用于生成迭代器
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

import matplotlib.animation as animation  # 导入 matplotlib.animation 库，用于动画制作


def data_gen():
    # 生成器函数，每次迭代产生一个时间步 t 和对应的函数值 y
    for cnt in itertools.count():
        t = cnt / 10
        yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)


def init():
    # 初始化函数，设置初始图形状态
    ax.set_ylim(-1.1, 1.1)  # 设置 y 轴的范围
    ax.set_xlim(0, 1)  # 设置 x 轴的范围
    del xdata[:]  # 清空 xdata 列表
    del ydata[:]  # 清空 ydata 列表
    line.set_data(xdata, ydata)  # 设置图形对象 line 的数据
    return line,  # 返回更新后的图形对象 line


fig, ax = plt.subplots()  # 创建图形窗口和坐标系
line, = ax.plot([], [], lw=2)  # 在坐标系上绘制空的线条对象
ax.grid()  # 显示坐标系的网格线
xdata, ydata = [], []  # 初始化存储数据的列表


def run(data):
    # 动画更新函数，根据新的数据更新图形
    t, y = data
    xdata.append(t)  # 添加新的 t 值到 xdata 列表
    ydata.append(y)  # 添加新的 y 值到 ydata 列表
    xmin, xmax = ax.get_xlim()  # 获取当前 x 轴的范围

    if t >= xmax:  # 如果 t 超过当前 x 轴的最大值
        ax.set_xlim(xmin, 2*xmax)  # 扩展 x 轴的范围
        ax.figure.canvas.draw()  # 重新绘制图形
    line.set_data(xdata, ydata)  # 更新图形对象 line 的数据

    return line,  # 返回更新后的图形对象 line


# 创建动画对象，每 100 毫秒更新一次，初始函数为 init，数据来源为 data_gen 生成器
ani = animation.FuncAnimation(fig, run, data_gen, interval=100, init_func=init,
                              save_count=100)

plt.show()  # 显示动画
```