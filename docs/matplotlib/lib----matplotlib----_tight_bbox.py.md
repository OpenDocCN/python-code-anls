# `D:\src\scipysrc\matplotlib\lib\matplotlib\_tight_bbox.py`

```
"""
Helper module for the *bbox_inches* parameter in `.Figure.savefig`.
"""

from matplotlib.transforms import Bbox, TransformedBbox, Affine2D


def adjust_bbox(fig, bbox_inches, fixed_dpi=None):
    """
    Temporarily adjust the figure so that only the specified area
    (bbox_inches) is saved.

    It modifies fig.bbox, fig.bbox_inches,
    fig.transFigure._boxout, and fig.patch.  While the figure size
    changes, the scale of the original figure is conserved.  A
    function which restores the original values are returned.
    """
    # 保存原始的图形边界框和英寸边界框
    origBbox = fig.bbox
    origBboxInches = fig.bbox_inches
    _boxout = fig.transFigure._boxout

    old_aspect = []
    locator_list = []
    sentinel = object()
    # 遍历所有子图
    for ax in fig.axes:
        # 获取当前子图的定位器
        locator = ax.get_axes_locator()
        if locator is not None:
            # 应用定位器以调整子图位置
            ax.apply_aspect(locator(ax, None))
        locator_list.append(locator)
        # 冻结当前位置作为原始位置
        current_pos = ax.get_position(original=False).frozen()
        # 设置子图的定位器为冻结的当前位置
        ax.set_axes_locator(lambda a, r, _pos=current_pos: _pos)
        # 覆盖强制应用于子图的纵横比的方法
        if 'apply_aspect' in ax.__dict__:
            old_aspect.append(ax.apply_aspect)
        else:
            old_aspect.append(sentinel)
        # 设置子图的纵横比应用为无操作
        ax.apply_aspect = lambda pos=None: None

    def restore_bbox():
        # 还原子图的定位器和纵横比设置
        for ax, loc, aspect in zip(fig.axes, locator_list, old_aspect):
            ax.set_axes_locator(loc)
            if aspect is sentinel:
                # 删除无操作函数，以恢复原始方法
                del ax.apply_aspect
            else:
                ax.apply_aspect = aspect

        # 还原图形的边界框和英寸边界框
        fig.bbox = origBbox
        fig.bbox_inches = origBboxInches
        fig.transFigure._boxout = _boxout
        fig.transFigure.invalidate()
        fig.patch.set_bounds(0, 0, 1, 1)

    if fixed_dpi is None:
        fixed_dpi = fig.dpi
    # 创建一个仿射变换对象，用于缩放
    tr = Affine2D().scale(fixed_dpi)
    dpi_scale = fixed_dpi / fig.dpi

    # 设置图形的英寸边界框
    fig.bbox_inches = Bbox.from_bounds(0, 0, *bbox_inches.size)
    x0, y0 = tr.transform(bbox_inches.p0)
    w1, h1 = fig.bbox.size * dpi_scale
    # 设置图形变换的外部框
    fig.transFigure._boxout = Bbox.from_bounds(-x0, -y0, w1, h1)
    fig.transFigure.invalidate()

    # 使用仿射变换对象和英寸边界框来重新设置图形的边界框
    fig.bbox = TransformedBbox(fig.bbox_inches, tr)

    # 设置图形补丁的边界
    fig.patch.set_bounds(x0 / w1, y0 / h1,
                         fig.bbox.width / w1, fig.bbox.height / h1)

    # 返回还原函数
    return restore_bbox


def process_figure_for_rasterizing(fig, bbox_inches_restore, fixed_dpi=None):
    """
    A function that needs to be called when figure dpi changes during the
    drawing (e.g., rasterizing).  It recovers the bbox and re-adjust it with
    the new dpi.
    """
    # 还原图形的边界框和执行调整边界框函数
    bbox_inches, restore_bbox = bbox_inches_restore
    restore_bbox()
    r = adjust_bbox(fig, bbox_inches, fixed_dpi)

    return bbox_inches, r
```