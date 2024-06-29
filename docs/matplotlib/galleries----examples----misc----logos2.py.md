# `D:\src\scipysrc\matplotlib\galleries\examples\misc\logos2.py`

```
"""
===============
Matplotlib logo
===============

This example generates the current matplotlib logo.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 numpy 模块

import matplotlib.cm as cm  # 导入 matplotlib 的 colormap 模块
import matplotlib.font_manager  # 导入 matplotlib 的字体管理模块
from matplotlib.patches import PathPatch, Rectangle  # 从 matplotlib 的 patches 模块导入 PathPatch 和 Rectangle 类
from matplotlib.text import TextPath  # 从 matplotlib 的 text 模块导入 TextPath 类
import matplotlib.transforms as mtrans  # 导入 matplotlib 的 transforms 模块

MPL_BLUE = '#11557c'  # 定义常量 MPL_BLUE 为颜色码 '#11557c'


def get_font_properties():
    # 检查是否安装了 Calibri 字体，若未安装则回退到 Carlito 字体，两者在度量上等效
    if 'Calibri' in matplotlib.font_manager.findfont('Calibri:bold'):
        return matplotlib.font_manager.FontProperties(family='Calibri',
                                                      weight='bold')
    if 'Carlito' in matplotlib.font_manager.findfont('Carlito:bold'):
        print('Original font not found. Falling back to Carlito. '
              'The logo text will not be in the correct font.')
        return matplotlib.font_manager.FontProperties(family='Carlito',
                                                      weight='bold')
    print('Original font not found. '
          'The logo text will not be in the correct font.')
    return None


def create_icon_axes(fig, ax_position, lw_bars, lw_grid, lw_border, rgrid):
    """
    Create a polar Axes containing the matplotlib radar plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to draw into.
    ax_position : (float, float, float, float)
        The position of the created Axes in figure coordinates as
        (x, y, width, height).
    lw_bars : float
        The linewidth of the bars.
    lw_grid : float
        The linewidth of the grid.
    lw_border : float
        The linewidth of the Axes border.
    rgrid : array-like
        Positions of the radial grid.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The created Axes.
    """
    # 使用指定的 matplotlib 配置上下文设置，设置轴的边缘颜色和宽度
    with plt.rc_context({'axes.edgecolor': MPL_BLUE,
                         'axes.linewidth': lw_border}):
        # 在图形上添加一个极坐标轴
        ax = fig.add_axes(ax_position, projection='polar')
        # 将网格绘制在数据之下
        ax.set_axisbelow(True)

        # 定义柱状图的参数
        N = 7
        arc = 2. * np.pi
        theta = np.arange(0.0, arc, arc / N)
        radii = np.array([2, 6, 8, 7, 4, 5, 8])
        width = np.pi / 4 * np.array([0.4, 0.4, 0.6, 0.8, 0.2, 0.5, 0.3])
        # 创建柱状图对象
        bars = ax.bar(theta, radii, width=width, bottom=0.0, align='edge',
                      edgecolor='0.3', lw=lw_bars)
        # 遍历每个柱状图对象，设置其颜色
        for r, bar in zip(radii, bars):
            color = *cm.jet(r / 10.)[:3], 0.6  # 从 jet 色图中获取颜色，设置透明度为 0.6
            bar.set_facecolor(color)

        # 设置坐标轴的刻度参数，隐藏标签
        ax.tick_params(labelbottom=False, labeltop=False,
                       labelleft=False, labelright=False)

        # 绘制极坐标网格
        ax.grid(lw=lw_grid, color='0.9')
        # 设置极坐标轴的最大半径和刻度
        ax.set_rmax(9)
        ax.set_yticks(rgrid)

        # 添加一个矩形补丁作为可见的背景，稍微超出轴的范围
        ax.add_patch(Rectangle((0, 0), arc, 9.58,
                               facecolor='white', zorder=0,
                               clip_on=False, in_layout=False))
        # 返回绘制好的极坐标轴对象
        return ax
# %%
# 创建一个包含 Matplotlib 图标的完整图形。
# 
# 参数：
# - height_px：图形的高度（以像素为单位）。
# - lw_bars：条形边框的线宽度。
# - lw_grid：网格的线宽度。
# - lw_border：图标边框的线宽度。
# - rgrid：径向网格的位置序列。
# - with_text：是否绘制仅图标或包括“matplotlib”文本。
def make_logo(height_px, lw_bars, lw_grid, lw_border, rgrid, with_text=False):
    """
    创建一个包含 Matplotlib 图标的完整图形。

    Parameters
    ----------
    height_px : int
        图形的高度（以像素为单位）。
    lw_bars : float
        条形边框的线宽度。
    lw_grid : float
        网格的线宽度。
    lw_border : float
        图标边框的线宽度。
    rgrid : sequence of float
        径向网格的位置序列。
    with_text : bool
        是否绘制仅图标或包括“matplotlib”文本。
    """
    dpi = 100
    height = height_px / dpi
    figsize = (5 * height, height) if with_text else (height, height)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_alpha(0)

    if with_text:
        # 如果需要包括文本，则在图形中创建包含“matplotlib”文本的 Axes。
        create_text_axes(fig, height_px)
    
    # 根据是否包括文本选择不同的 Axes 位置和大小。
    ax_pos = (0.535, 0.12, .17, 0.75) if with_text else (0.03, 0.03, .94, .94)
    # 创建图标的 Axes，并返回它。
    ax = create_icon_axes(fig, ax_pos, lw_bars, lw_grid, lw_border, rgrid)

    return fig, ax

# %%
# 创建一个大尺寸的 Matplotlib 图标。
make_logo(height_px=110, lw_bars=0.7, lw_grid=0.5, lw_border=1,
          rgrid=[1, 3, 5, 7])

# %%
# 创建一个小尺寸（32像素）的 Matplotlib 图标。
make_logo(height_px=32, lw_bars=0.3, lw_grid=0.3, lw_border=0.3, rgrid=[5])

# %%
# 创建一个包括文本的大尺寸 Matplotlib 图标，通常用于 Matplotlib 网站上。
make_logo(height_px=110, lw_bars=0.7, lw_grid=0.5, lw_border=1,
          rgrid=[1, 3, 5, 7], with_text=True)
# 展示图形。
plt.show()
```