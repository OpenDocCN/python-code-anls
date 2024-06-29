# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\figimage_demo.py`

```py
"""
=============
Figimage Demo
=============

This illustrates placing images directly in the figure, with no Axes objects.

"""
# 导入 matplotlib.pyplot 库，并简称为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简称为 np
import numpy as np

# 创建一个新的图形对象
fig = plt.figure()

# 创建一个 100x100 的二维数组 Z，包含从 0 到 9999 的整数
Z = np.arange(10000).reshape((100, 100))
# 将 Z 的右半部分（从第50列到最后一列）的值设为1
Z[:, 50:] = 1

# 在图形上添加第一个图像，使用 Z 作为图像数据，xo=50, yo=0 表示偏移量，origin='lower' 表示原点在左下角
im1 = fig.figimage(Z, xo=50, yo=0, origin='lower')
# 在图形上添加第二个图像，使用 Z 作为图像数据，xo=100, yo=100 表示另一种偏移量，alpha=.8 表示透明度，origin='lower' 表示原点在左下角
im2 = fig.figimage(Z, xo=100, yo=100, alpha=.8, origin='lower')

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure`
#    - `matplotlib.figure.Figure.figimage` / `matplotlib.pyplot.figimage`
```