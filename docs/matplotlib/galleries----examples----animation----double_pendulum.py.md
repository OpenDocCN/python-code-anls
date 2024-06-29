# `D:\src\scipysrc\matplotlib\galleries\examples\animation\double_pendulum.py`

```py
"""
===========================
The double pendulum problem
===========================

This animation illustrates the double pendulum problem.

Double pendulum formula translated from the C code at
http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

Output generated via `matplotlib.animation.Animation.to_jshtml`.
"""

# 导入所需的库
import matplotlib.pyplot as plt  # 导入 matplotlib 库中的 pyplot 模块
import numpy as np  # 导入 numpy 库并将其重命名为 np
from numpy import cos, sin  # 从 numpy 中导入 cos 和 sin 函数

import matplotlib.animation as animation  # 导入 matplotlib 库中的 animation 模块

# 定义物理常数和参数
G = 9.8  # 重力加速度，单位是 m/s^2
L1 = 1.0  # 第一摆杆的长度，单位是 m
L2 = 1.0  # 第二摆杆的长度，单位是 m
L = L1 + L2  # 组合摆杆的最大长度
M1 = 1.0  # 第一摆杆的质量，单位是 kg
M2 = 1.0  # 第二摆杆的质量，单位是 kg
t_stop = 2.5  # 模拟时间长度，单位是秒
history_len = 500  # 轨迹显示的点数

def derivs(t, state):
    # 初始化状态导数数组
    dydx = np.zeros_like(state)

    # 计算第一个摆杆的角度导数
    dydx[0] = state[1]

    # 计算角度差
    delta = state[2] - state[0]
    # 计算第一个摆杆角速度的导数
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)

    # 计算第二个摆杆的角度导数
    dydx[2] = state[3]

    # 计算第二个摆杆角速度的导数
    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)

    return dydx

# 创建时间数组，步长为 0.01 秒
dt = 0.01
t = np.arange(0, t_stop, dt)

# 初始角度（以度为单位）和初始角速度（每秒度）
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0

# 初始状态，转换为弧度制
state = np.radians([th1, w1, th2, w2])

# 使用欧拉方法积分微分方程
y = np.empty((len(t), 4))
y[0] = state
for i in range(1, len(t)):
    y[i] = y[i - 1] + derivs(t[i - 1], y[i - 1]) * dt

# 可以使用 scipy 进行更精确的积分估计：
# y = scipy.integrate.solve_ivp(derivs, t[[0, -1]], state, t_eval=t).y.T

# 计算摆杆端点的坐标
x1 = L1*sin(y[:, 0])
y1 = -L1*cos(y[:, 0])
x2 = L2*sin(y[:, 2]) + x1
y2 = -L2*cos(y[:, 2]) + y1

# 创建动画图形
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)  # 绘制摆杆的线条
trace, = ax.plot([], [], '.-', lw=1, ms=2)  # 绘制轨迹点
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)  # 显示时间的文本

# 动画函数
def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    history_x = x2[:i]
    history_y = y2[:i]

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return line, trace, time_text

# 创建动画对象并显示
ani = animation.FuncAnimation(
    fig, animate, len(y), interval=dt*1000, blit=True)
plt.show()
```