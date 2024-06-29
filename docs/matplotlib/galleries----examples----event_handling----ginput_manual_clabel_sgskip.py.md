# `D:\src\scipysrc\matplotlib\galleries\examples\event_handling\ginput_manual_clabel_sgskip.py`

```py
"""
=====================
Interactive functions
=====================

This provides examples of uses of interactive functions, such as ginput,
waitforbuttonpress and manual clabel placement.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""

import time  # 导入时间模块

import matplotlib.pyplot as plt  # 导入 Matplotlib 绘图库
import numpy as np  # 导入 NumPy 数学库


def tellme(s):
    """显示信息s，并设置标题为s"""
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()

# %%
# Define a triangle by clicking three points

plt.figure()  # 创建一个新的图形窗口
plt.xlim(0, 1)  # 设置x轴范围
plt.ylim(0, 1)  # 设置y轴范围

tellme('You will define a triangle, click to begin')  # 显示提示信息

plt.waitforbuttonpress()  # 等待用户点击鼠标按钮

while True:
    pts = []
    while len(pts) < 3:
        tellme('Select 3 corners with mouse')  # 提示用户用鼠标选择三个角点
        pts = np.asarray(plt.ginput(3, timeout=-1))  # 获取用户输入的三个点坐标
        if len(pts) < 3:
            tellme('Too few points, starting over')  # 如果选择点少于3个，重新开始
            time.sleep(1)  # 等待1秒钟

    ph = plt.fill(pts[:, 0], pts[:, 1], 'r', lw=2)  # 用红色填充三角形

    tellme('Happy? Key click for yes, mouse click for no')  # 提示用户确认选择

    if plt.waitforbuttonpress():  # 等待用户按键确认
        break

    # Get rid of fill
    for p in ph:
        p.remove()  # 移除填充区域

# %%
# Now contour according to distance from triangle
# corners - just an example

# Define a nice function of distance from individual pts
def f(x, y, pts):
    """计算每个点(x, y)到三角形顶点pts的距离倒数之和"""
    z = np.zeros_like(x)
    for p in pts:
        z = z + 1/(np.sqrt((x - p[0])**2 + (y - p[1])**2))
    return 1/z  # 返回距离的倒数

X, Y = np.meshgrid(np.linspace(-1, 1, 51), np.linspace(-1, 1, 51))  # 创建网格坐标
Z = f(X, Y, pts)  # 计算网格上每个点到三角形顶点的距离倒数之和

CS = plt.contour(X, Y, Z, 20)  # 绘制等高线图

tellme('Use mouse to select contour label locations, middle button to finish')  # 提示用户用鼠标选择等高线标签位置
CL = plt.clabel(CS, manual=True)  # 手动添加等高线标签

# %%
# Now do a zoom

tellme('Now do a nested zoom, click to begin')  # 提示用户开始缩放操作
plt.waitforbuttonpress()  # 等待用户点击鼠标按钮

while True:
    tellme('Select two corners of zoom, middle mouse button to finish')  # 提示用户选择缩放区域的两个角
    pts = plt.ginput(2, timeout=-1)  # 获取用户选择的两个点
    if len(pts) < 2:
        break
    (x0, y0), (x1, y1) = pts
    xmin, xmax = sorted([x0, x1])
    ymin, ymax = sorted([y0, y1])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

tellme('All Done!')  # 提示用户操作完成
plt.show()  # 显示图形
```