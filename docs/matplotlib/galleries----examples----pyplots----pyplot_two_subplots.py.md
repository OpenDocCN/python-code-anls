# `D:\src\scipysrc\matplotlib\galleries\examples\pyplots\pyplot_two_subplots.py`

```py
"""
=========================
Two subplots using pyplot
=========================

Create a figure with two subplots using `.pyplot.subplot`.
"""

# 导入 matplotlib.pyplot 库，并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简写为 np
import numpy as np

# 定义一个函数 f(t)，返回 np.exp(-t) * np.cos(2*np.pi*t)
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

# 生成 t1 数组，从 0.0 到 5.0，步长为 0.1
t1 = np.arange(0.0, 5.0, 0.1)
# 生成 t2 数组，从 0.0 到 5.0，步长为 0.02
t2 = np.arange(0.0, 5.0, 0.02)

# 创建一个新的图形窗口
plt.figure()

# 在图形窗口中创建第一个子图，2行1列的布局，第1个子图
plt.subplot(211)
# 绘制 t1 和 f(t1) 的图像，使用蓝色标记为圆圈
plt.plot(t1, f(t1), color='tab:blue', marker='o')
# 绘制 t2 和 f(t2) 的图像，使用黑色实线
plt.plot(t2, f(t2), color='black')

# 在图形窗口中创建第二个子图，2行1列的布局，第2个子图
plt.subplot(212)
# 绘制 t2 和 np.cos(2*np.pi*t2) 的图像，使用橙色虚线
plt.plot(t2, np.cos(2*np.pi*t2), color='tab:orange', linestyle='--')

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.pyplot.figure`
#    - `matplotlib.pyplot.subplot`
```