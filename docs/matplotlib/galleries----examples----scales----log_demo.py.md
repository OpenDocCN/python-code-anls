# `D:\src\scipysrc\matplotlib\galleries\examples\scales\log_demo.py`

```py
"""
========
Log Demo
========

Examples of plots with logarithmic axes.
"""

# 导入 matplotlib 库并简称为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库并简称为 np
import numpy as np

# 准备绘图所需的数据
t = np.arange(0.01, 20.0, 0.01)

# 创建图形对象和子图对象
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# 在第一个子图上绘制对数 y 轴图像
ax1.semilogy(t, np.exp(-t / 5.0))
ax1.set(title='semilogy')  # 设置子图标题
ax1.grid()  # 显示网格线

# 在第二个子图上绘制对数 x 轴图像
ax2.semilogx(t, np.sin(2 * np.pi * t))
ax2.set(title='semilogx')  # 设置子图标题
ax2.grid()  # 显示网格线

# 在第三个子图上绘制对数 x 和 y 轴图像（双对数坐标）
ax3.loglog(t, 20 * np.exp(-t / 10.0))
ax3.set_xscale('log', base=2)  # 设置 x 轴的对数尺度为以 2 为底
ax3.set(title='loglog base 2 on x')  # 设置子图标题
ax3.grid()  # 显示网格线

# 在第四个子图上绘制带误差条的对数坐标图像，并处理非正值
x = 10.0**np.linspace(0.0, 2.0, 20)
y = x**2.0

ax4.set_xscale("log", nonpositive='clip')  # 设置 x 轴的对数尺度，负值和零被截断
ax4.set_yscale("log", nonpositive='clip')  # 设置 y 轴的对数尺度，负值和零被截断
ax4.set(title='Errorbars go negative')  # 设置子图标题
ax4.errorbar(x, y, xerr=0.1 * x, yerr=5.0 + 0.75 * y)  # 绘制带有误差条的图像
ax4.set_ylim(bottom=0.1)  # 设置 y 轴的最小值为 0.1，用于自动调整误差条的显示范围

# 调整图形布局使子图之间不重叠，并显示图形
fig.tight_layout()
plt.show()
```