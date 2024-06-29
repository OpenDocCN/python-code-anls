# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\matshow.py`

```
"""
===============================
Visualize matrices with matshow
===============================

`~.axes.Axes.matshow` visualizes a 2D matrix or array as color-coded image.
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于数组操作
import numpy as np

# 创建一个对角线上逐渐增加的二维数组
a = np.diag(range(15))

# 使用 matshow 函数将数组 a 可视化为彩色图像并显示出来
plt.matshow(a)

# 显示绘制的图像
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
```