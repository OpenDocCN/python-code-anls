# `D:\src\scipysrc\matplotlib\galleries\users_explain\artists\imshow_extent.py`

```
def set_extent_None_text(ax):
    """
    Add text to the plot when extent is None.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to which the text is added.
    """
    # Define text properties
    text = 'extent is None'
    x = 0.5
    y = 0.5
    ha = 'center'
    va = 'center'

    # Add text to the plot
    ax.text(x, y, text, ha=ha, va=va, transform=ax.transAxes)
    # 在图形 ax 上添加文本，内容为 'equals\nextent=None'，位于坐标 (3, 2.5)
    # 文本的大小为 'large'，水平对齐方式为居中 ('ha='center')，垂直对齐方式为居中 ('va='center')，
    # 文本颜色为白色 ('color='w')
    ax.text(3, 2.5, 'equals\nextent=None', size='large',
            ha='center', va='center', color='w')
def plot_imshow_with_labels(ax, data, extent, origin, xlim, ylim):
    """Actually run ``imshow()`` and add extent and index labels."""
    # 使用给定数据绘制图像，并设置图像的起始位置和范围
    im = ax.imshow(data, origin=origin, extent=extent)

    # 获取图像的范围 (left, right, bottom, top)
    left, right, bottom, top = im.get_extent()
    
    # 根据 xlim 和 ylim 设置上下标签的位置字符串
    if xlim is None or top > bottom:
        upper_string, lower_string = 'top', 'bottom'
    else:
        upper_string, lower_string = 'bottom', 'top'
    
    # 根据 xlim 和 ylim 设置左右标签的位置字符串，以及是否翻转 x 轴索引
    if ylim is None or left < right:
        port_string, starboard_string = 'left', 'right'
        inverted_xindex = False
    else:
        port_string, starboard_string = 'right', 'left'
        inverted_xindex = True
    
    # 设置标签的样式和属性
    bbox_kwargs = {'fc': 'w', 'alpha': .75, 'boxstyle': "round4"}
    ann_kwargs = {'xycoords': 'axes fraction',
                  'textcoords': 'offset points',
                  'bbox': bbox_kwargs}
    
    # 添加上下左右标签
    ax.annotate(upper_string, xy=(.5, 1), xytext=(0, -1),
                ha='center', va='top', **ann_kwargs)
    ax.annotate(lower_string, xy=(.5, 0), xytext=(0, 1),
                ha='center', va='bottom', **ann_kwargs)
    ax.annotate(port_string, xy=(0, .5), xytext=(1, 0),
                ha='left', va='center', rotation=90,
                **ann_kwargs)
    ax.annotate(starboard_string, xy=(1, .5), xytext=(-1, 0),
                ha='right', va='center', rotation=-90,
                **ann_kwargs)
    
    # 设置图像标题，显示起始位置的信息
    ax.set_title(f'origin: {origin}')

    # 添加索引标签
    for index in ["[0, 0]", "[0, N']", "[M', 0]", "[M', N']"]:
        # 获取索引标签的位置和对齐方式
        tx, ty, halign = get_index_label_pos(index, extent, origin,
                                             inverted_xindex)
        # 获取标签的颜色
        facecolor = get_color(index, data, im.get_cmap())
        # 添加索引标签
        ax.text(tx, ty, index, color='white', ha=halign, va='center',
                bbox={'boxstyle': 'square', 'facecolor': facecolor})
    
    # 如果设置了 xlim，应用到当前轴
    if xlim:
        ax.set_xlim(*xlim)
    
    # 如果设置了 ylim，应用到当前轴
    if ylim:
        ax.set_ylim(*ylim)


def generate_imshow_demo_grid(extents, xlim=None, ylim=None):
    # 计算格子的行数
    N = len(extents)
    
    # 创建一个新的图形对象
    fig = plt.figure(tight_layout=True)
    # 设置图形的尺寸
    fig.set_size_inches(6, N * (11.25) / 5)
    # 创建一个网格布局
    gs = GridSpec(N, 5, figure=fig)

    # 创建列的字典，包括标签、上部和下部的子图
    columns = {'label': [fig.add_subplot(gs[j, 0]) for j in range(N)],
               'upper': [fig.add_subplot(gs[j, 1:3]) for j in range(N)],
               'lower': [fig.add_subplot(gs[j, 3:5]) for j in range(N)]}
    
    # 创建一个示例数据
    x, y = np.ogrid[0:6, 0:7]
    data = x + y

    # 对于每个起始位置，绘制图像并添加标签
    for origin in ['upper', 'lower']:
        for ax, extent in zip(columns[origin], extents):
            plot_imshow_with_labels(ax, data, extent, origin, xlim, ylim)

    # 设置第一列的标签标题
    columns['label'][0].set_title('extent=')
    
    # 对于每个标签，添加其范围信息或显示 "None"
    for ax, extent in zip(columns['label'], extents):
        if extent is None:
            text = 'None'
        else:
            left, right, bottom, top = extent
            text = (f'left: {left:0.1f}\nright: {right:0.1f}\n'
                    f'bottom: {bottom:0.1f}\ntop: {top:0.1f}\n')
        ax.text(1., .5, text, transform=ax.transAxes, ha='right', va='center')
        ax.axis('off')
    # 返回函数中的变量 columns，即函数执行的结果
    return columns
# %%
#
# Default extent
# --------------
#
# First, let's have a look at the default ``extent=None``
generate_imshow_demo_grid(extents=[None])

# %%
#
# Generally, for an array of shape (M, N), the first index runs along the
# vertical, the second index runs along the horizontal.
# The pixel centers are at integer positions ranging from 0 to ``N' = N - 1``
# horizontally and from 0 to ``M' = M - 1`` vertically.
# *origin* determines how the data is filled in the bounding box.
#
# For ``origin='lower'``:
#
# - [0, 0] is at (left, bottom)
# - [M', 0] is at (left, top)
# - [0, N'] is at (right, bottom)
# - [M', N'] is at (right, top)
#
# ``origin='upper'`` reverses the vertical axes direction and filling:
#
# - [0, 0] is at (left, top)
# - [M', 0] is at (left, bottom)
# - [0, N'] is at (right, top)
# - [M', N'] is at (right, bottom)
#
# In summary, the position of the [0, 0] index as well as the extent are
# influenced by *origin*:
#
# ======  ===============  ==========================================
# origin  [0, 0] position  extent
# ======  ===============  ==========================================
# upper   top left         ``(-0.5, numcols-0.5, numrows-0.5, -0.5)``
# lower   bottom left      ``(-0.5, numcols-0.5, -0.5, numrows-0.5)``
# ======  ===============  ==========================================
#
# The default value of *origin* is set by :rc:`image.origin` which defaults
# to ``'upper'`` to match the matrix indexing conventions in math and
# computer graphics image indexing conventions.
#
#
# Explicit extent
# ---------------
#
# By setting *extent* we define the coordinates of the image area. The
# underlying image data is interpolated/resampled to fill that area.
#
# If the Axes is set to autoscale, then the view limits of the Axes are set
# to match the *extent* which ensures that the coordinate set by
# ``(left, bottom)`` is at the bottom left of the Axes!  However, this
# may invert the axis so they do not increase in the 'natural' direction.
#

# Define a list of extents to demonstrate different coordinate configurations
extents = [(-0.5, 6.5, -0.5, 5.5),
           (-0.5, 6.5, 5.5, -0.5),
           (6.5, -0.5, -0.5, 5.5),
           (6.5, -0.5, 5.5, -0.5)]

# Generate a grid of images using the specified extents
columns = generate_imshow_demo_grid(extents)

# Set text indicating that extents are set to None for upper and lower grids
set_extent_None_text(columns['upper'][1])
set_extent_None_text(columns['lower'][0])

# %%
#
# Explicit extent and axes limits
# -------------------------------
#
# If we fix the axes limits by explicitly setting `~.axes.Axes.set_xlim` /
# `~.axes.Axes.set_ylim`, we force a certain size and orientation of the Axes.
# This can decouple the 'left-right' and 'top-bottom' sense of the image from
# the orientation on the screen.
#
# In the example below we have chosen the limits slightly larger than the
# extent (note the white areas within the Axes).
#
# While we keep the extents as in the examples before, the coordinate (0, 0)
# is now explicitly put at the bottom left and values increase to up and to
# the right (from the viewer's point of view).
# We can see that:
#
# - The coordinate ``(left, bottom)`` anchors the image which then fills the
#   box going towards the ``(right, top)`` point in data space.
# - The first column is always closest to the 'left'.
# - *origin* controls if the first row is closest to 'top' or 'bottom'.
# - The image may be inverted along either direction.
# - The 'left-right' and 'top-bottom' sense of the image may be uncoupled from
#   the orientation on the screen.

# 调用函数 `generate_imshow_demo_grid` 来生成演示图表，指定图表的数据范围 `extents` 和坐标轴的界限 `xlim`, `ylim`。
generate_imshow_demo_grid(extents=[None] + extents,
                          xlim=(-2, 8), ylim=(-1, 6))

# 显示生成的图表
plt.show()
```