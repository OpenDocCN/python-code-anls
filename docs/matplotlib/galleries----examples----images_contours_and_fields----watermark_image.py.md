# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\watermark_image.py`

```py
"""
===============
Watermark image
===============

Overlay an image on a plot by moving it to the front (``zorder=3``) and making it
semi-transparent (``alpha=0.7``).
"""

# 导入必要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 库

import matplotlib.cbook as cbook  # 导入 matplotlib 的 cbook 模块，用于处理示例数据
import matplotlib.image as image  # 导入 matplotlib 的 image 模块，用于处理图像数据

# 使用 cbook 提供的示例数据 'logo2.png'，并读取为图像对象
with cbook.get_sample_data('logo2.png') as file:
    im = image.imread(file)  # 读取文件内容并转换为图像对象

# 创建一个图形窗口和一个坐标轴
fig, ax = plt.subplots()

# 生成一些随机数据用于柱状图的绘制
np.random.seed(19680801)
x = np.arange(30)
y = x + np.random.randn(30)

# 绘制柱状图，并设置柱子的颜色为 '#6bbc6b'
ax.bar(x, y, color='#6bbc6b')

# 添加网格线到坐标轴
ax.grid()

# 在图形上插入图像作为水印，位置为 (25, 25)，设置图像层级为 3，透明度为 0.7
fig.figimage(im, 25, 25, zorder=3, alpha=.7)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.image`
#    - `matplotlib.image.imread` / `matplotlib.pyplot.imread`
#    - `matplotlib.figure.Figure.figimage`
```