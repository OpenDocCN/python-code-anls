# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\multi_image.py`

```
"""
===============
Multiple images
===============

Make a set of images with a single colormap, norm, and colorbar.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 NumPy 库，用于数值计算

from matplotlib import colors  # 从 matplotlib 中导入 colors 模块，用于颜色处理

np.random.seed(19680801)  # 设置随机数种子，保证随机数据可复现
Nr = 3  # 设定图像行数为 3
Nc = 2  # 设定图像列数为 2

# 创建一个包含多个子图的图形对象和子图数组
fig, axs = plt.subplots(Nr, Nc)
fig.suptitle('Multiple images')  # 设置图形对象的总标题为 'Multiple images'

images = []  # 初始化一个空列表，用于存储图像对象
for i in range(Nr):
    for j in range(Nc):
        # 生成数据，数据范围因每个子图而异
        data = ((1 + i + j) / 10) * np.random.rand(10, 20)
        # 在当前子图位置绘制图像，并将图像对象添加到列表中
        images.append(axs[i, j].imshow(data))
        axs[i, j].label_outer()  # 在外侧标记坐标轴刻度

# 计算所有图像中颜色值的最小和最大值，用于设置颜色刻度
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)  # 创建归一化对象，用于颜色映射
for im in images:
    im.set_norm(norm)  # 设置图像对象的归一化对象

# 在图形对象上添加水平方向的颜色条
fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)


# 定义函数，使图像对象响应其他图像对象的颜色映射和颜色范围的变化
def update(changed_image):
    for im in images:
        if (changed_image.get_cmap() != im.get_cmap()
                or changed_image.get_clim() != im.get_clim()):
            im.set_cmap(changed_image.get_cmap())  # 更新颜色映射
            im.set_clim(changed_image.get_clim())  # 更新颜色范围


for im in images:
    im.callbacks.connect('changed', update)  # 将更新函数连接到图像对象的 'changed' 事件

plt.show()  # 显示绘制的图形

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.colors.Normalize`
#    - `matplotlib.cm.ScalarMappable.set_cmap`
#    - `matplotlib.cm.ScalarMappable.set_norm`
#    - `matplotlib.cm.ScalarMappable.set_clim`
#    - `matplotlib.cbook.CallbackRegistry.connect`
```