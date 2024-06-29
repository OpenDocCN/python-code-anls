# `D:\src\scipysrc\matplotlib\galleries\plot_types\arrays\imshow.py`

```py
"""
=========
imshow(Z)
=========
Display data as an image, i.e., on a 2D regular raster.

See `~matplotlib.axes.Axes.imshow`.
"""

# 导入matplotlib.pyplot模块，并命名为plt
import matplotlib.pyplot as plt
# 导入numpy模块，并命名为np
import numpy as np

# 使用指定的样式风格 '_mpl-gallery-nogrid'
plt.style.use('_mpl-gallery-nogrid')

# 生成数据
# 创建一个网格，X和Y分别是在[-3, 3]范围内均匀分布的16个点
X, Y = np.meshgrid(np.linspace(-3, 3, 16), np.linspace(-3, 3, 16))
# 计算二维函数Z的值
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)

# 绘图
# 创建一个图形窗口和一个子图
fig, ax = plt.subplots()
# 在子图ax上显示二维数组Z，origin='lower'表示原点在左下角
ax.imshow(Z, origin='lower')

# 显示图形
plt.show()
```