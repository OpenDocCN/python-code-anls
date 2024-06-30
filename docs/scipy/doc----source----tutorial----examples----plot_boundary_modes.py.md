# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\plot_boundary_modes.py`

```
# 导入必要的库：numpy用于数值计算，matplotlib.pyplot用于绘图
import numpy as np
import matplotlib.pyplot as plt

# 从scipy库中导入ndimage模块，用于图像处理
from scipy import ndimage

# 创建一个包含浮点数的numpy数组，表示图像中的像素值
img = np.array([-2, -1, 0, 1, 2], float)

# 在指定范围内生成一组等间距的数值，用作图像坐标
x = np.linspace(-2, 6, num=1000)

# 定义不同的插值模式，用于ndimage.map_coordinates函数
modes = ['constant', 'grid-constant', 'nearest', 'reflect', 'mirror', 'wrap',
         'grid-wrap']

# 创建一个图形窗口和子图网格，每种模式对应三个子图
fig, axes = plt.subplots(len(modes), 3, figsize=(11, 8), sharex=True,
                         sharey=True)

# 遍历每种插值模式，并对应每种模式的三个子图进行绘制
for mode, (ax0, ax1, ax2) in zip(modes, axes):

    # 使用ndimage.map_coordinates函数对图像进行插值计算，order=0
    y = ndimage.map_coordinates(img, [x], order=0, mode=mode)
    ax0.scatter(np.arange(img.size), img)  # 绘制散点图
    ax0.plot(x, y, '-')  # 绘制线性图
    ax0.set_title(f'mode={mode}, order=0')  # 设置子图标题

    # 使用ndimage.map_coordinates函数对图像进行插值计算，order=1
    y2 = ndimage.map_coordinates(img, [x], order=1, mode=mode)
    ax1.scatter(np.arange(img.size), img)  # 绘制散点图
    ax1.plot(x, y2, '-')  # 绘制线性图
    ax1.set_title(f'mode={mode}, order=1')  # 设置子图标题

    # 使用ndimage.map_coordinates函数对图像进行插值计算，order=3
    y3 = ndimage.map_coordinates(img, [x], order=3, mode=mode)
    ax2.scatter(np.arange(img.size), img)  # 绘制散点图
    ax2.plot(x, y3, '-')  # 绘制线性图
    ax2.set_title(f'mode={mode}, order=3')  # 设置子图标题

    sz = len(img)
    # 根据不同的模式绘制额外的辅助线
    for ax in (ax0, ax1, ax2):
        if mode in ['grid-wrap', 'reflect']:
            ax.plot([-0.5, -0.5], [-2.5, 2.5], 'k--')  # 绘制虚线
            ax.plot([sz - 0.5, sz - 0.5], [-2.5, 2.5], 'k--')  # 绘制虚线
        elif mode in ['wrap', 'mirror']:
            ax.plot([0, 0], [-2.5, 2.5], 'k--')  # 绘制虚线
            ax.plot([sz - 1, sz - 1], [-2.5, 2.5], 'k--')  # 绘制虚线

    # 如果模式不是'constant'，则在图中标记出超出边界的点
    if mode != 'constant':
        for xx in range(int(x[0]), int(x[-1] + 1)):
            if (xx < 0) or (xx > img.size - 1):
                idx = np.argmin(np.abs(x - xx))  # 找到最接近xx的索引

                # 在三个子图中标记出超出边界的点
                for y_vals, ax in zip((y, y2, y3), (ax0, ax1, ax2)):
                    ax.scatter(
                        [x[idx]], [y_vals[idx]], facecolors='none',
                        edgecolor='#0343df', marker='o'
                    )

# 调整子图之间的布局，使得图形更加紧凑
plt.tight_layout()

# 显示绘制的图形
plt.show()
```