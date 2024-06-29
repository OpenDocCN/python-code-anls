# `D:\src\scipysrc\matplotlib\galleries\examples\animation\bayes_update.py`

```py
"""
================
The Bayes update
================

This animation displays the posterior estimate updates as it is refitted when
new data arrives.
The vertical line represents the theoretical value to which the plotted
distribution should converge.

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

# 导入需要的库
import math  # 导入数学库，用于计算数学函数

import matplotlib.pyplot as plt  # 导入 matplotlib 库中的 pyplot 模块
import numpy as np  # 导入 numpy 数值计算库

from matplotlib.animation import FuncAnimation  # 从 matplotlib 库的 animation 模块中导入 FuncAnimation 类


def beta_pdf(x, a, b):
    # 定义 beta 分布的概率密度函数
    return (x**(a-1) * (1-x)**(b-1) * math.gamma(a + b)
            / (math.gamma(a) * math.gamma(b)))


class UpdateDist:
    def __init__(self, ax, prob=0.5):
        self.success = 0  # 成功次数初始化为 0
        self.prob = prob  # 成功概率设定为传入的 prob 参数
        self.line, = ax.plot([], [], 'k-')  # 创建空的线条对象，并与 ax 绑定
        self.x = np.linspace(0, 1, 200)  # 在 [0, 1] 区间上生成 200 个均匀分布的点作为 x 值
        self.ax = ax  # 将传入的轴对象保存为实例变量

        # 设置绘图参数
        self.ax.set_xlim(0, 1)  # 设置 x 轴范围为 [0, 1]
        self.ax.set_ylim(0, 10)  # 设置 y 轴范围为 [0, 10]
        self.ax.grid(True)  # 显示网格线

        # 这条垂直线代表理论值，即绘制的分布应该收敛到的值
        self.ax.axvline(prob, linestyle='--', color='black')

    def start(self):
        # 用于 FuncAnimation 的 init_func 参数，初始化动画时调用此方法
        return self.line,

    def __call__(self, i):
        # 实现 __call__ 方法，使得对象可调用；每次更新动画时调用此方法

        if i == 0:
            self.success = 0  # 如果是第一帧，重置成功次数为 0
            self.line.set_data([], [])  # 清空线条的数据
            return self.line,

        # 根据成功概率确定成功与否
        if np.random.rand() < self.prob:
            self.success += 1  # 如果随机数小于成功概率，则成功次数加一

        # 计算 Beta 分布的概率密度函数，并更新线条的数据
        y = beta_pdf(self.x, self.success + 1, (i - self.success) + 1)
        self.line.set_data(self.x, y)

        return self.line,

# 为了可重现性，固定随机种子
np.random.seed(19680801)

# 创建 matplotlib 图形和轴对象
fig, ax = plt.subplots()

# 创建 UpdateDist 的实例对象 ud，并设置成功概率为 0.7
ud = UpdateDist(ax, prob=0.7)

# 创建动画对象 anim，使用 FuncAnimation 运行 UpdateDist 的实例对象 ud
anim = FuncAnimation(fig, ud, init_func=ud.start, frames=100, interval=100, blit=True)

# 显示动画
plt.show()
```