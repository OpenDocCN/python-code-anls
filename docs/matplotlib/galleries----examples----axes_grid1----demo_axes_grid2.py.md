# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\demo_axes_grid2.py`

```py
"""
==========
Axes Grid2
==========

Grid of images with shared xaxis and yaxis.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 库

# 导入 matplotlib 的 cbook 模块，用于获取示例数据
from matplotlib import cbook
# 从 mpl_toolkits.axes_grid1 中导入 ImageGrid 类
from mpl_toolkits.axes_grid1 import ImageGrid


# 定义一个函数，用于在图像中添加内部标题
def add_inner_title(ax, title, loc, **kwargs):
    from matplotlib.offsetbox import AnchoredText  # 导入 AnchoredText 类
    from matplotlib.patheffects import withStroke  # 导入 withStroke 函数
    # 设置文本属性，包括描边效果和字体大小
    prop = dict(path_effects=[withStroke(foreground='w', linewidth=3)],
                size=plt.rcParams['legend.fontsize'])
    # 创建 AnchoredText 对象，用于添加标题
    at = AnchoredText(title, loc=loc, prop=prop,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    # 在 Axes 对象中添加 AnchoredText 对象
    ax.add_artist(at)
    return at


# 创建一个新的 Figure 对象
fig = plt.figure(figsize=(6, 6))

# 准备图像数据
Z = cbook.get_sample_data("axes_grid/bivariate_normal.npy")  # 获取示例数据
extent = (-3, 4, -4, 3)  # 设置图像的坐标范围
ZS = [Z[i::3, :] for i in range(3)]  # 按行分割示例数据，生成一个包含三个数组的列表
extent = extent[0], extent[1]/3., extent[2], extent[3]  # 调整坐标范围的格式


# *** Demo 1: colorbar at each Axes ***
# 创建 ImageGrid 对象，包含三个 Axes 对象
grid = ImageGrid(
    fig, 211, nrows_ncols=(1, 3), axes_pad=0.05, label_mode="1", share_all=True,
    cbar_location="top", cbar_mode="each", cbar_size="7%", cbar_pad="1%")
grid[0].set(xticks=[-2, 0], yticks=[-2, 0, 2])  # 设置第一个 Axes 的刻度

# 遍历每个 Axes 对象和对应的图像数据
for i, (ax, z) in enumerate(zip(grid, ZS)):
    im = ax.imshow(z, origin="lower", extent=extent)  # 在 Axes 中显示图像
    cb = ax.cax.colorbar(im)  # 在当前 Axes 的 colorbar 上添加颜色条
    # 修改颜色条的刻度
    if i in [1, 2]:
        cb.set_ticks([-1, 0, 1])

# 在每个 Axes 上添加内部标题
for ax, im_title in zip(grid, ["Image 1", "Image 2", "Image 3"]):
    add_inner_title(ax, im_title, loc='lower left')


# *** Demo 2: shared colorbar ***
# 创建另一个 ImageGrid 对象，包含三个 Axes 对象
grid2 = ImageGrid(
    fig, 212, nrows_ncols=(1, 3), axes_pad=0.05, label_mode="1", share_all=True,
    cbar_location="right", cbar_mode="single", cbar_size="10%", cbar_pad=0.05)
grid2[0].set(xlabel="X", ylabel="Y", xticks=[-2, 0], yticks=[-2, 0, 2])  # 设置第一个 Axes 的标签和刻度

clim = (np.min(ZS), np.max(ZS))  # 设置色彩限制
# 遍历每个 Axes 对象和对应的图像数据
for ax, z in zip(grid2, ZS):
    im = ax.imshow(z, clim=clim, origin="lower", extent=extent)  # 在 Axes 中显示图像

# 使用 cbar_mode="single"，设置所有 Axes 对象共享一个 colorbar
ax.cax.colorbar(im)

# 在每个 Axes 上添加内部标题
for ax, im_title in zip(grid2, ["(a)", "(b)", "(c)"]):
    add_inner_title(ax, im_title, loc='upper left')

# 显示图形
plt.show()
```