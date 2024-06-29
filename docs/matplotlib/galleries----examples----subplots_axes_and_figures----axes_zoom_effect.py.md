# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\axes_zoom_effect.py`

```
"""
================
Axes Zoom Effect
================

"""

# 导入 Matplotlib 的 pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt

# 导入需要的变换和插入定位器类
from matplotlib.transforms import (Bbox, TransformedBbox,
                                   blended_transform_factory)
from mpl_toolkits.axes_grid1.inset_locator import (BboxConnector,
                                                   BboxConnectorPatch,
                                                   BboxPatch)


# 定义函数 connect_bbox，用于连接两个边界框，并绘制连接线和填充矩形
def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_lines, prop_patches=None):
    # 如果未提供 prop_patches 参数，则设置默认值
    if prop_patches is None:
        prop_patches = {
            **prop_lines,
            "alpha": prop_lines.get("alpha", 1) * 0.2,
            "clip_on": False,
        }

    # 创建两个边界框连接线对象
    c1 = BboxConnector(
        bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **prop_lines)
    c2 = BboxConnector(
        bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **prop_lines)

    # 创建两个边界框填充矩形对象
    bbox_patch1 = BboxPatch(bbox1, **prop_patches)
    bbox_patch2 = BboxPatch(bbox2, **prop_patches)

    # 创建连接边界框的填充矩形补丁对象
    p = BboxConnectorPatch(bbox1, bbox2,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           clip_on=False,
                           **prop_patches)

    return c1, c2, bbox_patch1, bbox_patch2, p


# 定义函数 zoom_effect01，实现在两个坐标轴之间创建缩放效果
def zoom_effect01(ax1, ax2, xmin, xmax, **kwargs):
    """
    Connect *ax1* and *ax2*. The *xmin*-to-*xmax* range in both Axes will
    be marked.

    Parameters
    ----------
    ax1
        The main Axes.
    ax2
        The zoomed Axes.
    xmin, xmax
        The limits of the colored area in both plot Axes.
    **kwargs
        Arguments passed to the patch constructor.
    """

    # 根据给定的 xmin 和 xmax 创建边界框对象
    bbox = Bbox.from_extents(xmin, 0, xmax, 1)

    # 根据 ax1 和 ax2 的坐标轴变换创建转换后的边界框对象
    mybbox1 = TransformedBbox(bbox, ax1.get_xaxis_transform())
    mybbox2 = TransformedBbox(bbox, ax2.get_xaxis_transform())

    # 设置填充矩形的属性
    prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}

    # 连接两个边界框，创建连接线和填充矩形
    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        mybbox1, mybbox2,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines=kwargs, prop_patches=prop_patches)

    # 将填充矩形和连接线添加到 ax1 和 ax2 上
    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


# 定义函数 zoom_effect02，与 zoom_effect01 类似，从 ax1.viewLim 中获取 xmin 和 xmax
def zoom_effect02(ax1, ax2, **kwargs):
    """
    ax1 : the main Axes
    ax1 : the zoomed Axes

    Similar to zoom_effect01.  The xmin & xmax will be taken from the
    ax1.viewLim.
    """

    # 创建混合转换对象，用于将 ax1 和 ax2 的坐标轴进行混合转换
    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(ax2.transData, tt)

    # 获取 ax1 的边界框对象
    mybbox1 = ax1.bbox
    # 根据 ax1.viewLim 和混合转换创建 ax2 的转换后的边界框对象
    mybbox2 = TransformedBbox(ax1.viewLim, trans)

    # 设置填充矩形的属性
    prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}

    # 连接两个边界框，创建连接线和填充矩形
    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        mybbox1, mybbox2,
        loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        prop_lines=kwargs, prop_patches=prop_patches)

    # 将填充矩形和连接线添加到 ax1 和 ax2 上
    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    # 将图形对象 p 添加到子图 ax2 中
    ax2.add_patch(p)
    
    # 返回多个对象 c1, c2, bbox_patch1, bbox_patch2, p，这些对象可能代表某种计算或图形元素
    return c1, c2, bbox_patch1, bbox_patch2, p
# 创建一个新的图形对象，并使用subplot_mosaic方法定义子图布局
axs = plt.figure().subplot_mosaic([
    ["zoom1", "zoom2"],  # 创建一个布局，包含两行两列，定义了四个子图位置
    ["main", "main"],
])

# 在子图"main"上设置x轴范围为0到5
axs["main"].set(xlim=(0, 5))

# 调用zoom_effect01函数，在子图"zoom1"上实现缩放效果，将"zoom1"和"main"作为参数传递
zoom_effect01(axs["zoom1"], axs["main"], 0.2, 0.8)

# 在子图"zoom2"上设置x轴范围为2到3
axs["zoom2"].set(xlim=(2, 3))

# 调用zoom_effect02函数，在子图"zoom2"上实现缩放效果，将"zoom2"和"main"作为参数传递
zoom_effect02(axs["zoom2"], axs["main"])

# 显示绘制的图形
plt.show()
```