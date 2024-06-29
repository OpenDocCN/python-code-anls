# `D:\src\scipysrc\matplotlib\galleries\examples\color\colormap_reference.py`

```
"""
==================
Colormap reference
==================

Reference for colormaps included with Matplotlib.

A reversed version of each of these colormaps is available by appending
``_r`` to the name, as shown in :ref:`reverse-cmap`.

See :ref:`colormaps` for an in-depth discussion about
colormaps, including colorblind-friendliness, and
:ref:`colormap-manipulation` for a guide to creating
colormaps.
"""

# 导入 Matplotlib 库
import matplotlib.pyplot as plt
import numpy as np

# 定义不同类别的 colormap 列表
cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar'])]

# 创建一个线性渐变数组
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

# 定义函数用于绘制颜色渐变图
def plot_color_gradients(cmap_category, cmap_list):
    # 计算图的行数
    nrows = len(cmap_list)
    # 计算图形的高度，根据行数动态调整
    figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
    # 创建一个子图，并调整图形布局
    fig, axs = plt.subplots(nrows=nrows, figsize=(6.4, figh))
    fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)

    # 设置第一个子图的标题
    axs[0].set_title(f"{cmap_category} colormaps", fontsize=14)

    # 遍历颜色映射列表，为每个子图绘制颜色渐变图，并添加 colormap 名称的文本
    for ax, cmap_name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=cmap_name)
        ax.text(-.01, .5, cmap_name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # 关闭所有子图的刻度线和轴线
    for ax in axs:
        ax.set_axis_off()

# 对每个类别的 colormap 进行循环，调用绘制函数
for cmap_category, cmap_list in cmaps:
    plot_color_gradients(cmap_category, cmap_list)


# %%
# .. _reverse-cmap:
#
# Reversed colormaps
# ------------------
#
# Append ``_r`` to the name of any built-in colormap to get the reversed
# version:

# 绘制原始和反转后的 colormap 图像
plot_color_gradients("Original and reversed ", ['viridis', 'viridis_r'])

# %%
# The built-in reversed colormaps are generated using `.Colormap.reversed`.
# For an example, see :ref:`reversing-colormap`

# %%
#
# 创建一个新的图形窗口
fig, ax = plt.subplots()

# 在图形窗口中显示图像，使用'gray'色彩映射
ax.imshow(image, cmap='gray')

# 在图形窗口中添加文本标签，显示在指定位置和指定文本
ax.text(0.5, 1.08, 'Example image', ha='center', va='center', transform=ax.transAxes)

# 将图形窗口的坐标轴关闭
ax.set_axis_off()
```