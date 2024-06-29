# `D:\src\scipysrc\matplotlib\galleries\examples\misc\fill_spiral.py`

```py
"""
===========
Fill Spiral
===========

"""
# 导入 matplotlib.pyplot 作为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库并将其重命名为 np
import numpy as np

# 创建角度数组 theta，范围从 0 到 8π，步长为 0.1
theta = np.arange(0, 8*np.pi, 0.1)
# 设置参数 a 和 b 的值
a = 1
b = .2

# 在 0 到 2π 的范围内，以步长 π/2 遍历
for dt in np.arange(0, 2*np.pi, np.pi/2.0):

    # 计算第一组坐标点 x 和 y
    x = a*np.cos(theta + dt)*np.exp(b*theta)
    y = a*np.sin(theta + dt)*np.exp(b*theta)

    # 增加 dt 的值
    dt = dt + np.pi/4.0

    # 计算第二组坐标点 x2 和 y2
    x2 = a*np.cos(theta + dt)*np.exp(b*theta)
    y2 = a*np.sin(theta + dt)*np.exp(b*theta)

    # 合并两组 x 和 x2，反向排列后拼接成新的 x 坐标数组 xf
    xf = np.concatenate((x, x2[::-1]))
    # 合并两组 y 和 y2，反向排列后拼接成新的 y 坐标数组 yf
    yf = np.concatenate((y, y2[::-1]))

    # 使用 plt.fill 函数填充 xf 和 yf 所表示的多边形
    p1 = plt.fill(xf, yf)

# 显示绘制的图形
plt.show()
```