# `D:\src\scipysrc\matplotlib\galleries\examples\animation\strip_chart.py`

```py
"""
============
Oscilloscope
============

Emulates an oscilloscope.

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入numpy数值计算库

import matplotlib.animation as animation  # 导入matplotlib动画模块
from matplotlib.lines import Line2D  # 导入Line2D类用于绘制折线图


class Scope:
    def __init__(self, ax, maxt=2, dt=0.02):
        self.ax = ax  # 设置图形的坐标轴
        self.dt = dt  # 时间步长
        self.maxt = maxt  # 最大时间范围
        self.tdata = [0]  # 时间数据列表，初始为0
        self.ydata = [0]  # 数据值列表，初始为0
        self.line = Line2D(self.tdata, self.ydata)  # 创建Line2D对象用于绘制折线图
        self.ax.add_line(self.line)  # 将折线图对象添加到坐标轴上
        self.ax.set_ylim(-.1, 1.1)  # 设置y轴范围
        self.ax.set_xlim(0, self.maxt)  # 设置x轴范围为0到最大时间范围

    def update(self, y):
        lastt = self.tdata[-1]  # 获取当前时间数据的最后一个值
        if lastt >= self.tdata[0] + self.maxt:  # 如果最后一个时间超过了显示范围
            self.tdata = [self.tdata[-1]]  # 重置时间数据列表
            self.ydata = [self.ydata[-1]]  # 重置数据值列表
            self.ax.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)  # 更新x轴范围
            self.ax.figure.canvas.draw()  # 更新绘图

        # 计算新的时间值，避免浮点数精度问题
        t = self.tdata[0] + len(self.tdata) * self.dt

        self.tdata.append(t)  # 添加新的时间值到时间数据列表
        self.ydata.append(y)  # 添加新的数据值到数据值列表
        self.line.set_data(self.tdata, self.ydata)  # 设置折线图的数据
        return self.line,  # 返回更新后的折线图对象


def emitter(p=0.1):
    """Return a random value in [0, 1) with probability p, else 0."""
    while True:
        v = np.random.rand()  # 生成一个随机数v，范围[0, 1)
        if v > p:
            yield 0.  # 如果v大于概率p，则产生数据值0
        else:
            yield np.random.rand()  # 否则产生一个新的随机数作为数据值


# 设置随机种子以便结果可重现性
np.random.seed(19680801 // 10)


fig, ax = plt.subplots()  # 创建图形和坐标轴对象
scope = Scope(ax)  # 创建Scope类的实例对象

# 将生成器函数"emitter"传递给FuncAnimation函数，用于产生更新数据
ani = animation.FuncAnimation(fig, scope.update, emitter, interval=50,
                              blit=True, save_count=100)

plt.show()  # 显示动画
```