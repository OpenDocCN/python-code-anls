# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axisartist\axis_artist.py`

```
# FIXME :
# angles are given in data coordinate - need to convert it to canvas coordinate

# 导入模块和函数
from operator import methodcaller

# 导入 NumPy 库，并使用 np 别名
import numpy as np

# 导入 Matplotlib 库，使用 mpl 别名
import matplotlib as mpl
# 从 Matplotlib 中导入 _api 和 cbook 模块
from matplotlib import _api, cbook
# 导入 Matplotlib 的 artist 模块，并使用 martist 别名
import matplotlib.artist as martist
# 导入 Matplotlib 的 colors 模块，并使用 mcolors 别名
import matplotlib.colors as mcolors
# 导入 Matplotlib 的 text 模块，并使用 mtext 别名
import matplotlib.text as mtext
# 从 Matplotlib 中导入 LineCollection 类
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import (
    Affine2D, Bbox, IdentityTransform, ScaledTranslation)

from .axisline_style import AxislineStyle

class AttributeCopier:
    def get_ref_artist(self):
        """
        Return the underlying artist that actually defines some properties
        (e.g., color) of this artist.
        """
        # 返回定义此艺术家某些属性（如颜色）的基础艺术家对象
        raise RuntimeError("get_ref_artist must overridden")

    def get_attribute_from_ref_artist(self, attr_name):
        # 从基础艺术家对象获取属性值
        getter = methodcaller("get_" + attr_name)
        prop = getter(super())
        return getter(self.get_ref_artist()) if prop == "auto" else prop

class Ticks(AttributeCopier, Line2D):
    """
    Ticks are derived from `.Line2D`, and note that ticks themselves
    are markers. Thus, you should use set_mec, set_mew, etc.

    To change the tick size (length), you need to use
    `set_ticksize`. To change the direction of the ticks (ticks are
    in opposite direction of ticklabels by default), use
    ``set_tick_out(False)``
    """
    
    def __init__(self, ticksize, tick_out=False, *, axis=None, **kwargs):
        # 初始化函数，设置刻度的大小和位置
        self._ticksize = ticksize
        self.locs_angles_labels = []

        # 设置刻度的位置（内部或外部）
        self.set_tick_out(tick_out)

        # 设置刻度线的轴
        self._axis = axis
        if self._axis is not None:
            # 如果未指定颜色，则设置为自动获取
            if "color" not in kwargs:
                kwargs["color"] = "auto"
            # 如果未指定标记边缘宽度，则设置为自动获取
            if "mew" not in kwargs and "markeredgewidth" not in kwargs:
                kwargs["markeredgewidth"] = "auto"

        # 调用父类 Line2D 的初始化函数
        Line2D.__init__(self, [0.], [0.], **kwargs)
        # 设置刻度自动吸附
        self.set_snap(True)

    def get_ref_artist(self):
        # 返回基础艺术家对象，继承自 AttributeCopier 类
        return self._axis.majorTicks[0].tick1line

    def set_color(self, color):
        # 设置刻度线的颜色
        if not cbook._str_equal(color, "auto"):
            mcolors._check_color_like(color=color)
        self._color = color
        self.stale = True

    def get_color(self):
        # 获取刻度线的颜色
        return self.get_attribute_from_ref_artist("color")

    def get_markeredgecolor(self):
        # 获取标记边缘的颜色
        return self.get_attribute_from_ref_artist("markeredgecolor")

    def get_markeredgewidth(self):
        # 获取标记边缘的宽度
        return self.get_attribute_from_ref_artist("markeredgewidth")

    def set_tick_out(self, b):
        """Set whether ticks are drawn inside or outside the axes."""
        # 设置刻度线是否在坐标轴内或外
        self._tick_out = b

    def get_tick_out(self):
        """Return whether ticks are drawn inside or outside the axes."""
        # 返回刻度线是否在坐标轴内或外
        return self._tick_out

    def set_ticksize(self, ticksize):
        """Set length of the ticks in points."""
        # 设置刻度线的长度（单位为点）
        self._ticksize = ticksize

    def get_ticksize(self):
        """Return length of the ticks in points."""
        # 返回刻度线的长度（单位为点）
        return self._ticksize

    def set_locs_angles(self, locs_angles):
        # 设置刻度线的位置和角度
        self.locs_angles = locs_angles

    _tickvert_path = Path([[0., 0.], [1., 0.]])
    # 如果对象不可见，则不进行绘制
    def draw(self, renderer):
        if not self.get_visible():
            return

        # 创建新的图形上下文对象
        gc = renderer.new_gc()
        # 设置标记边缘的颜色
        gc.set_foreground(self.get_markeredgecolor())
        # 设置标记边缘的线宽
        gc.set_linewidth(self.get_markeredgewidth())
        # 设置透明度
        gc.set_alpha(self._alpha)

        # 获取路径的变换信息
        path_trans = self.get_transform()
        # 创建标记的变换，基于点到像素的转换
        marker_transform = (Affine2D()
                            .scale(renderer.points_to_pixels(self._ticksize)))
        # 如果标记向外，则旋转180度
        if self.get_tick_out():
            marker_transform.rotate_deg(180)

        # 遍历位置和角度的列表
        for loc, angle in self.locs_angles:
            # 将位置变换为路径的仿射变换后的坐标
            locs = path_trans.transform_non_affine(np.array([loc]))
            # 如果存在坐标轴并且坐标不在视图范围内，则跳过
            if self.axes and not self.axes.viewLim.contains(*locs[0]):
                continue
            # 绘制标记点
            renderer.draw_markers(
                gc, self._tickvert_path,
                # 应用标记变换和角度的仿射变换
                marker_transform + Affine2D().rotate_deg(angle),
                # 路径使用路径变换的仿射变换
                Path(locs), path_trans.get_affine())

        # 恢复图形上下文状态
        gc.restore()
class LabelBase(mtext.Text):
    """
    A base class for `.AxisLabel` and `.TickLabels`. The position and
    angle of the text are calculated by the offset_ref_angle,
    text_ref_angle, and offset_radius attributes.
    """

    def __init__(self, *args, **kwargs):
        # 初始化位置、角度和标签信息列表为空
        self.locs_angles_labels = []
        # 参考角度初始化为0
        self._ref_angle = 0
        # 偏移半径初始化为0.0
        self._offset_radius = 0.

        # 调用父类构造函数
        super().__init__(*args, **kwargs)

        # 设置文本旋转模式为"anchor"
        self.set_rotation_mode("anchor")
        # 设置文本角度是否跟随参考角度
        self._text_follow_ref_angle = True

    @property
    def _text_ref_angle(self):
        # 如果文本角度跟随参考角度，返回参考角度加90度
        # 否则返回0度
        if self._text_follow_ref_angle:
            return self._ref_angle + 90
        else:
            return 0

    @property
    def _offset_ref_angle(self):
        # 返回参考角度
        return self._ref_angle

    # 获取相反方向的映射字典
    _get_opposite_direction = {"left": "right",
                               "right": "left",
                               "top": "bottom",
                               "bottom": "top"}.__getitem__

    def draw(self, renderer):
        if not self.get_visible():
            return

        # 保存原始属性并调整一些属性
        tr = self.get_transform()
        angle_orig = self.get_rotation()
        theta = np.deg2rad(self._offset_ref_angle)
        dd = self._offset_radius
        dx, dy = dd * np.cos(theta), dd * np.sin(theta)

        # 根据偏移角度和半径平移文本
        self.set_transform(tr + Affine2D().translate(dx, dy))
        # 设置文本旋转角度
        self.set_rotation(self._text_ref_angle + angle_orig)
        # 绘制文本
        super().draw(renderer)
        # 恢复原始属性
        self.set_transform(tr)
        self.set_rotation(angle_orig)

    def get_window_extent(self, renderer=None):
        if renderer is None:
            renderer = self.figure._get_renderer()

        # 保存原始属性并调整一些属性
        tr = self.get_transform()
        angle_orig = self.get_rotation()
        theta = np.deg2rad(self._offset_ref_angle)
        dd = self._offset_radius
        dx, dy = dd * np.cos(theta), dd * np.sin(theta)

        # 根据偏移角度和半径平移文本
        self.set_transform(tr + Affine2D().translate(dx, dy))
        # 设置文本旋转角度
        self.set_rotation(self._text_ref_angle + angle_orig)
        # 获取文本在窗口中的边界框
        bbox = super().get_window_extent(renderer).frozen()
        # 恢复原始属性
        self.set_transform(tr)
        self.set_rotation(angle_orig)

        return bbox


class AxisLabel(AttributeCopier, LabelBase):
    """
    Axis label. Derived from `.Text`. The position of the text is updated
    in the fly, so changing text position has no effect. Otherwise, the
    properties can be changed as a normal `.Text`.

    To change the pad between tick labels and axis label, use `set_pad`.
    """

    def __init__(self, *args, axis_direction="bottom", axis=None, **kwargs):
        # 设置轴对象和默认填充值
        self._axis = axis
        self._pad = 5
        self._external_pad = 0  # in pixels
        # 调用LabelBase类的构造函数
        LabelBase.__init__(self, *args, **kwargs)
        # 设置轴方向
        self.set_axis_direction(axis_direction)
    def set_pad(self, pad):
        """
        Set the internal pad in points.

        The actual pad will be the sum of the internal pad and the
        external pad (the latter is set automatically by the `.AxisArtist`).

        Parameters
        ----------
        pad : float
            The internal pad in points.
        """
        self._pad = pad



    def get_pad(self):
        """
        Return the internal pad in points.

        See `.set_pad` for more details.
        """
        return self._pad



    def get_ref_artist(self):
        # docstring inherited
        return self._axis.get_label()



    def get_text(self):
        # docstring inherited
        t = super().get_text()
        if t == "__from_axes__":
            return self._axis.get_label().get_text()
        return self._text



    _default_alignments = dict(left=("bottom", "center"),
                               right=("top", "center"),
                               bottom=("top", "center"),
                               top=("bottom", "center"))

    def set_default_alignment(self, d):
        """
        Set the default alignment. See `set_axis_direction` for details.

        Parameters
        ----------
        d : {"left", "bottom", "right", "top"}
        """
        va, ha = _api.check_getitem(self._default_alignments, d=d)
        self.set_va(va)
        self.set_ha(ha)



    _default_angles = dict(left=180,
                           right=0,
                           bottom=0,
                           top=180)

    def set_default_angle(self, d):
        """
        Set the default angle. See `set_axis_direction` for details.

        Parameters
        ----------
        d : {"left", "bottom", "right", "top"}
        """
        self.set_rotation(_api.check_getitem(self._default_angles, d=d))



    def set_axis_direction(self, d):
        """
        Adjust the text angle and text alignment of axis label
        according to the matplotlib convention.

        =====================    ========== ========= ========== ==========
        Property                 left       bottom    right      top
        =====================    ========== ========= ========== ==========
        axislabel angle          180        0         0          180
        axislabel va             center     top       center     bottom
        axislabel ha             right      center    right      center
        =====================    ========== ========= ========== ==========

        Note that the text angles are actually relative to (90 + angle
        of the direction to the ticklabel), which gives 0 for bottom
        axis.

        Parameters
        ----------
        d : {"left", "bottom", "right", "top"}
        """
        self.set_default_alignment(d)
        self.set_default_angle(d)



    def get_color(self):
        return self.get_attribute_from_ref_artist("color")
    # 绘制方法，接受一个渲染器对象作为参数
    def draw(self, renderer):
        # 如果图形对象不可见，则直接返回
        if not self.get_visible():
            return
        
        # 计算偏移半径，包括外部填充和内部填充的像素转换值
        self._offset_radius = \
            self._external_pad + renderer.points_to_pixels(self.get_pad())
        
        # 调用父类的绘制方法进行实际的绘制操作
        super().draw(renderer)

    # 获取对象在窗口中的范围，可以接受一个渲染器对象作为参数
    def get_window_extent(self, renderer=None):
        # 如果没有提供渲染器对象，则使用图形对象关联的渲染器
        if renderer is None:
            renderer = self.figure._get_renderer()
        
        # 如果图形对象不可见，则直接返回
        if not self.get_visible():
            return
        
        # 计算偏移半径，包括外部填充和内部填充的像素转换值
        r = self._external_pad + renderer.points_to_pixels(self.get_pad())
        self._offset_radius = r
        
        # 调用父类的方法获取对象在窗口中的边界框
        bb = super().get_window_extent(renderer)
        
        # 返回获取的边界框对象
        return bb
class TickLabels(AxisLabel):  # TickLabels 类继承自 AxisLabel 类，用于绘制刻度标签

    """
    Tick labels. While derived from `.Text`, this single artist draws all
    ticklabels. As in `.AxisLabel`, the position of the text is updated
    in the fly, so changing text position has no effect. Otherwise,
    the properties can be changed as a normal `.Text`. Unlike the
    ticklabels of the mainline Matplotlib, properties of a single
    ticklabel alone cannot be modified.

    To change the pad between ticks and ticklabels, use `~.AxisLabel.set_pad`.
    """

    def __init__(self, *, axis_direction="bottom", **kwargs):
        # TickLabels 类的构造函数，初始化刻度标签对象
        super().__init__(**kwargs)
        # 设置刻度标签的轴方向
        self.set_axis_direction(axis_direction)
        # 初始化轴标签的间距为0
        self._axislabel_pad = 0

    def get_ref_artist(self):
        # docstring inherited
        # 返回与本对象相关的参考艺术家对象，这里是轴对象的第一个刻度标签
        return self._axis.get_ticklabels()[0]

    def set_axis_direction(self, label_direction):
        """
        Adjust the text angle and text alignment of ticklabels
        according to the Matplotlib convention.

        The *label_direction* must be one of [left, right, bottom, top].

        =====================    ========== ========= ========== ==========
        Property                 left       bottom    right      top
        =====================    ========== ========= ========== ==========
        ticklabel angle          90         0         -90        180
        ticklabel va             center     baseline  center     baseline
        ticklabel ha             right      center    right      center
        =====================    ========== ========= ========== ==========

        Note that the text angles are actually relative to (90 + angle
        of the direction to the ticklabel), which gives 0 for bottom
        axis.

        Parameters
        ----------
        label_direction : {"left", "bottom", "right", "top"}

        """
        # 根据 Matplotlib 的约定调整刻度标签的文本角度和对齐方式
        self.set_default_alignment(label_direction)
        # 设置刻度标签的默认角度
        self.set_default_angle(label_direction)
        # 设置对象的轴方向属性
        self._axis_direction = label_direction

    def invert_axis_direction(self):
        # 反转轴方向
        label_direction = self._get_opposite_direction(self._axis_direction)
        self.set_axis_direction(label_direction)
    def _get_ticklabels_offsets(self, renderer, label_direction):
        """
        Calculate the ticklabel offsets from the tick and their total heights.

        The offset only takes account the offset due to the vertical alignment
        of the ticklabels: if axis direction is bottom and va is 'top', it will
        return 0; if va is 'baseline', it will return (height-descent).
        """
        # 获得文本宽度、高度和下降量的列表
        whd_list = self.get_texts_widths_heights_descents(renderer)

        # 如果列表为空，则返回默认值
        if not whd_list:
            return 0, 0

        # 初始化偏移量和垂直对齐方式（va）、水平对齐方式（ha）
        r = 0
        va, ha = self.get_va(), self.get_ha()

        # 根据标签方向进行不同的处理
        if label_direction == "left":
            # 计算最大的宽度作为边距
            pad = max(w for w, h, d in whd_list)
            if ha == "left":
                r = pad
            elif ha == "center":
                r = .5 * pad
        elif label_direction == "right":
            # 计算最大的宽度作为边距
            pad = max(w for w, h, d in whd_list)
            if ha == "right":
                r = pad
            elif ha == "center":
                r = .5 * pad
        elif label_direction == "bottom":
            # 计算最大的高度作为边距
            pad = max(h for w, h, d in whd_list)
            if va == "bottom":
                r = pad
            elif va == "center":
                r = .5 * pad
            elif va == "baseline":
                # 计算最大的上升和下降量，以及总的边距
                max_ascent = max(h - d for w, h, d in whd_list)
                max_descent = max(d for w, h, d in whd_list)
                r = max_ascent
                pad = max_ascent + max_descent
        elif label_direction == "top":
            # 计算最大的高度作为边距
            pad = max(h for w, h, d in whd_list)
            if va == "top":
                r = pad
            elif va == "center":
                r = .5 * pad
            elif va == "baseline":
                # 计算最大的上升和下降量，以及总的边距
                max_ascent = max(h - d for w, h, d in whd_list)
                max_descent = max(d for w, h, d in whd_list)
                r = max_descent
                pad = max_ascent + max_descent

        # r : 偏移量
        # pad : ticklabels 的总高度。这将用于计算 axislabel 的边距。
        return r, pad

    # 默认的对齐方式字典
    _default_alignments = dict(left=("center", "right"),
                               right=("center", "left"),
                               bottom=("baseline", "center"),
                               top=("baseline", "center"))

    # 默认的旋转角度字典
    _default_angles = dict(left=90,
                           right=-90,
                           bottom=0,
                           top=180)
    # 如果对象不可见，则设置轴标签的内部填充为外部填充，并返回
    def draw(self, renderer):
        if not self.get_visible():
            self._axislabel_pad = self._external_pad
            return

        # 调用方法获取刻度标签的偏移量和总宽度
        r, total_width = self._get_ticklabels_offsets(renderer,
                                                      self._axis_direction)

        # 计算外部填充加上用户定义的填充的总和，并赋给 pad
        pad = self._external_pad + renderer.points_to_pixels(self.get_pad())
        self._offset_radius = r + pad

        # 遍历每个位置、角度和标签的元组
        for (x, y), a, l in self._locs_angles_labels:
            # 如果标签内容为空白，则跳过不处理
            if not l.strip():
                continue
            self._ref_angle = a
            self.set_x(x)
            self.set_y(y)
            self.set_text(l)
            # 调用 LabelBase 类的 draw 方法进行绘制
            LabelBase.draw(self, renderer)

        # 保存此值以后用于绘制轴标签
        self._axislabel_pad = total_width + pad

    # 设置位置、角度和标签的元组列表
    def set_locs_angles_labels(self, locs_angles_labels):
        self._locs_angles_labels = locs_angles_labels

    # 获取窗口边界的列表，用于每个标签的边界框
    def get_window_extents(self, renderer=None):
        if renderer is None:
            renderer = self.figure._get_renderer()

        # 如果对象不可见，则设置轴标签的内部填充为外部填充，并返回空列表
        if not self.get_visible():
            self._axislabel_pad = self._external_pad
            return []

        # 获取刻度标签的偏移量和总宽度
        r, total_width = self._get_ticklabels_offsets(renderer,
                                                      self._axis_direction)

        # 计算外部填充加上用户定义的填充的总和，并赋给 pad
        pad = self._external_pad + renderer.points_to_pixels(self.get_pad())
        self._offset_radius = r + pad

        # 遍历每个位置、角度和标签的元组
        bboxes = []
        for (x, y), a, l in self._locs_angles_labels:
            self._ref_angle = a
            self.set_x(x)
            self.set_y(y)
            self.set_text(l)
            # 调用 LabelBase 类的 get_window_extent 方法获取边界框，并添加到 bboxes 列表中
            bb = LabelBase.get_window_extent(self, renderer)
            bboxes.append(bb)

        # 保存此值以后用于绘制轴标签
        self._axislabel_pad = total_width + pad

        return bboxes

    # 获取每个标签文本的宽度、高度和下降值的列表
    def get_texts_widths_heights_descents(self, renderer):
        """
        返回刻度标签的 ``(宽度, 高度, 下降值)`` 元组列表。

        空标签将被排除在外。
        """
        whd_list = []
        for _loc, _angle, label in self._locs_angles_labels:
            # 如果标签内容为空白，则跳过不处理
            if not label.strip():
                continue
            # 预处理标签内容，确定是否包含数学公式
            clean_line, ismath = self._preprocess_math(label)
            # 调用 renderer 的方法获取文本的宽度、高度和下降值，并添加到 whd_list 中
            whd = renderer.get_text_width_height_descent(
                clean_line, self._fontproperties, ismath=ismath)
            whd_list.append(whd)
        return whd_list
class GridlinesCollection(LineCollection):
    """
    A collection of grid lines for plotting.

    Parameters
    ----------
    which : {"major", "minor"}
        Specifies major or minor grid lines.
    axis : {"both", "x", "y"}
        Specifies which axis to apply grid lines to.
    *args, **kwargs
        Additional arguments passed to `.LineCollection`.
    """

    def __init__(self, *args, which="major", axis="both", **kwargs):
        # Initialize with specified grid type and axis
        self._which = which
        self._axis = axis
        super().__init__(*args, **kwargs)
        # Initialize grid helper to None initially
        self.set_grid_helper(None)

    def set_which(self, which):
        """
        Set whether to use major or minor grid lines.

        Parameters
        ----------
        which : {"major", "minor"}
        """
        self._which = which

    def set_axis(self, axis):
        """
        Set which axis to apply grid lines to.

        Parameters
        ----------
        axis : {"both", "x", "y"}
        """
        self._axis = axis

    def set_grid_helper(self, grid_helper):
        """
        Set the grid helper object.

        Parameters
        ----------
        grid_helper : `.GridHelperBase` subclass
        """
        self._grid_helper = grid_helper

    def draw(self, renderer):
        """
        Draw the grid lines using the specified renderer.

        Parameters
        ----------
        renderer : RendererBase subclass
        """
        if self._grid_helper is not None:
            # Update the limits using the associated axes
            self._grid_helper.update_lim(self.axes)
            # Retrieve grid lines based on current settings
            gl = self._grid_helper.get_gridlines(self._which, self._axis)
            # Set the segments for LineCollection from grid lines
            self.set_segments([np.transpose(l) for l in gl])
        super().draw(renderer)


class AxisArtist(martist.Artist):
    """
    An artist which draws axis lines, ticks, tick labels, and axis labels.
    """

    zorder = 2.5

    @property
    def LABELPAD(self):
        """
        Get the padding of the axis label.
        """
        return self.label.get_pad()

    @LABELPAD.setter
    def LABELPAD(self, v):
        """
        Set the padding of the axis label.

        Parameters
        ----------
        v : float
            Padding value to set.
        """
        self.label.set_pad(v)
    def __init__(self, axes,
                 helper,
                 offset=None,
                 axis_direction="bottom",
                 **kwargs):
        """
        Parameters
        ----------
        axes : `mpl_toolkits.axisartist.axislines.Axes`
            绘图中使用的坐标轴对象
        helper : `~mpl_toolkits.axisartist.axislines.AxisArtistHelper`
            坐标轴辅助对象，用于绘制坐标轴的辅助元素
        """
        # axes is also used to follow the axis attribute (tick color, etc).
        # axes 也用于跟踪轴属性（如刻度颜色等）

        super().__init__(**kwargs)

        self.axes = axes

        self._axis_artist_helper = helper

        if offset is None:
            offset = (0, 0)
        self.offset_transform = ScaledTranslation(
            *offset,
            Affine2D().scale(1 / 72)  # points to inches.
            + self.axes.figure.dpi_scale_trans)
        # 创建偏移变换对象，将偏移量从点转换为英寸，并考虑 DPI 缩放

        if axis_direction in ["left", "right"]:
            self.axis = axes.yaxis
        else:
            self.axis = axes.xaxis
        # 根据指定的轴方向确定要操作的轴对象

        self._axisline_style = None
        self._axis_direction = axis_direction

        self._init_line()
        self._init_ticks(**kwargs)
        self._init_offsetText(axis_direction)
        self._init_label()

        # axis direction
        self._ticklabel_add_angle = 0.
        self._axislabel_add_angle = 0.
        self.set_axis_direction(axis_direction)

    # axis direction
    # 设置轴的方向相关属性
    def set_axis_direction(self, axis_direction):
        """
        Adjust the direction, text angle, and text alignment of tick labels
        and axis labels following the Matplotlib convention for the rectangle
        axes.

        The *axis_direction* must be one of [left, right, bottom, top].

        =====================    ========== ========= ========== ==========
        Property                 left       bottom    right      top
        =====================    ========== ========= ========== ==========
        ticklabel direction      "-"        "+"       "+"        "-"
        axislabel direction      "-"        "+"       "+"        "-"
        ticklabel angle          90         0         -90        180
        ticklabel va             center     baseline  center     baseline
        ticklabel ha             right      center    right      center
        axislabel angle          180        0         0          180
        axislabel va             center     top       center     bottom
        axislabel ha             right      center    right      center
        =====================    ========== ========= ========== ==========

        Note that the direction "+" and "-" are relative to the direction of
        the increasing coordinate. Also, the text angles are actually
        relative to (90 + angle of the direction to the ticklabel),
        which gives 0 for bottom axis.

        Parameters
        ----------
        axis_direction : {"left", "bottom", "right", "top"}
            Direction of the axis, specifying how tick labels and axis labels should be oriented.
        """
        # Set axis direction for major tick labels
        self.major_ticklabels.set_axis_direction(axis_direction)
        # Set axis direction for axis label
        self.label.set_axis_direction(axis_direction)
        # Store the current axis direction internally
        self._axis_direction = axis_direction
        # Set tick label and axis label directions based on axis_direction
        if axis_direction in ["left", "top"]:
            self.set_ticklabel_direction("-")
            self.set_axislabel_direction("-")
        else:
            self.set_ticklabel_direction("+")
            self.set_axislabel_direction("+")

    def set_ticklabel_direction(self, tick_direction):
        r"""
        Adjust the direction of the tick labels.

        Note that the *tick_direction*\s '+' and '-' are relative to the
        direction of the increasing coordinate.

        Parameters
        ----------
        tick_direction : {"+", "-"}
            Direction of the tick labels relative to the axis.
        """
        # Set additional angle for tick labels based on tick_direction
        self._ticklabel_add_angle = _api.check_getitem(
            {"+": 0, "-": 180}, tick_direction=tick_direction)

    def invert_ticklabel_direction(self):
        """
        Invert the direction of tick labels.

        This method flips the direction of tick labels by adding 180 degrees
        to the current tick label angle and adjusting tick label direction.
        """
        # Increase tick label angle by 180 degrees and wrap around within 360 degrees
        self._ticklabel_add_angle = (self._ticklabel_add_angle + 180) % 360
        # Invert axis direction for major tick labels
        self.major_ticklabels.invert_axis_direction()
        # Invert axis direction for minor tick labels
        self.minor_ticklabels.invert_axis_direction()
    def set_axislabel_direction(self, label_direction):
        r"""
        Adjust the direction of the axis label.

        Note that the *label_direction*\s '+' and '-' are relative to the
        direction of the increasing coordinate.

        Parameters
        ----------
        label_direction : {"+", "-"}
            Direction indicator for axis label orientation.
        """
        # 设置私有属性 _axislabel_add_angle，根据 label_direction 设置角度
        self._axislabel_add_angle = _api.check_getitem(
            {"+": 0, "-": 180}, label_direction=label_direction)

    def get_transform(self):
        """
        Return the transformation combining transAxes and offset_transform.
        """
        return self.axes.transAxes + self.offset_transform

    def get_helper(self):
        """
        Return axis artist helper instance.
        """
        return self._axis_artist_helper

    def set_axisline_style(self, axisline_style=None, **kwargs):
        """
        Set the axisline style.

        The new style is completely defined by the passed attributes. Existing
        style attributes are forgotten.

        Parameters
        ----------
        axisline_style : str or None
            The line style, e.g. '->', optionally followed by a comma-separated
            list of attributes. Alternatively, the attributes can be provided
            as keywords.

            If *None* this returns a string containing the available styles.

        Examples
        --------
        The following two commands are equal:

        >>> set_axisline_style("->,size=1.5")
        >>> set_axisline_style("->", size=1.5)
        """
        # 如果 axisline_style 是 None，则打印可用的样式并返回
        if axisline_style is None:
            return AxislineStyle.pprint_styles()

        # 如果 axisline_style 是 AxislineStyle._Base 的实例，直接使用它
        if isinstance(axisline_style, AxislineStyle._Base):
            self._axisline_style = axisline_style
        else:
            # 否则，创建一个新的 AxislineStyle 对象并初始化
            self._axisline_style = AxislineStyle(axisline_style, **kwargs)

        # 初始化线条
        self._init_line()

    def get_axisline_style(self):
        """Return the current axisline style."""
        return self._axisline_style

    def _init_line(self):
        """
        Initialize the *line* artist that is responsible to draw the axis line.
        """
        # 获取线条的变换
        tran = (self._axis_artist_helper.get_line_transform(self.axes)
                + self.offset_transform)

        # 获取当前的轴线样式
        axisline_style = self.get_axisline_style()
        if axisline_style is None:
            # 如果样式为空，使用默认参数创建 PathPatch 对象
            self.line = PathPatch(
                self._axis_artist_helper.get_line(self.axes),
                color=mpl.rcParams['axes.edgecolor'],
                fill=False,
                linewidth=mpl.rcParams['axes.linewidth'],
                capstyle=mpl.rcParams['lines.solid_capstyle'],
                joinstyle=mpl.rcParams['lines.solid_joinstyle'],
                transform=tran)
        else:
            # 否则，使用当前样式创建线条对象
            self.line = axisline_style(self, transform=tran)

    def _draw_line(self, renderer):
        """
        Draw the axis line using the given renderer.

        Parameters
        ----------
        renderer : RendererBase subclass
            Renderer object used for drawing.
        """
        # 设置线条路径
        self.line.set_path(self._axis_artist_helper.get_line(self.axes))
        # 如果轴线样式不为空，设置线条的变异比例
        if self.get_axisline_style() is not None:
            self.line.set_line_mutation_scale(self.major_ticklabels.get_size())
        # 绘制线条
        self.line.draw(renderer)
    # 初始化 ticks 的方法，用于设置主要和次要刻度线及其标签的样式和位置
    def _init_ticks(self, **kwargs):
        # 获取轴的名称（如 'x' 或 'y'）
        axis_name = self.axis.axis_name

        # 计算刻度线的变换方式，包括轴辅助对象的刻度变换和偏移变换
        trans = (self._axis_artist_helper.get_tick_transform(self.axes)
                 + self.offset_transform)

        # 创建主要刻度线对象，设置刻度大小为参数中的值或默认大小
        self.major_ticks = Ticks(
            kwargs.get(
                "major_tick_size",
                mpl.rcParams[f"{axis_name}tick.major.size"]),
            axis=self.axis, transform=trans)
        
        # 创建次要刻度线对象，设置刻度大小为参数中的值或默认大小
        self.minor_ticks = Ticks(
            kwargs.get(
                "minor_tick_size",
                mpl.rcParams[f"{axis_name}tick.minor.size"]),
            axis=self.axis, transform=trans)

        # 设置主要刻度标签的样式，包括字体大小、位置偏移等
        size = mpl.rcParams[f"{axis_name}tick.labelsize"]
        self.major_ticklabels = TickLabels(
            axis=self.axis,
            axis_direction=self._axis_direction,
            figure=self.axes.figure,
            transform=trans,
            fontsize=size,
            pad=kwargs.get(
                "major_tick_pad", mpl.rcParams[f"{axis_name}tick.major.pad"]),
        )
        
        # 设置次要刻度标签的样式，包括字体大小、位置偏移等
        self.minor_ticklabels = TickLabels(
            axis=self.axis,
            axis_direction=self._axis_direction,
            figure=self.axes.figure,
            transform=trans,
            fontsize=size,
            pad=kwargs.get(
                "minor_tick_pad", mpl.rcParams[f"{axis_name}tick.minor.pad"]),
        )

    # 获取刻度信息的方法，返回两个列表：
    # - 包含刻度位置和角度的列表
    # - 包含刻度标签位置、角度和标签内容的列表
    def _get_tick_info(self, tick_iter):
        """
        Return a pair of:

        - list of locs and angles for ticks
        - list of locs, angles and labels for ticklabels.
        """
        ticks_loc_angle = []  # 存储刻度位置和角度的列表
        ticklabels_loc_angle_label = []  # 存储刻度标签位置、角度和标签内容的列表

        ticklabel_add_angle = self._ticklabel_add_angle

        # 遍历提供的刻度迭代器，获取每个刻度的位置、角度和标签
        for loc, angle_normal, angle_tangent, label in tick_iter:
            angle_label = angle_tangent - 90 + ticklabel_add_angle
            angle_tick = (angle_normal
                          if 90 <= (angle_label - angle_normal) % 360 <= 270
                          else angle_normal + 180)
            ticks_loc_angle.append([loc, angle_tick])  # 添加刻度位置和角度
            ticklabels_loc_angle_label.append([loc, angle_label, label])  # 添加刻度标签位置、角度和标签内容

        return ticks_loc_angle, ticklabels_loc_angle_label  # 返回刻度和刻度标签信息的元组
    # 更新刻度标签的外部填充，使主刻度和次刻度标签使用主刻度的刻度尺寸。最佳实践不明确。
    def _update_ticks(self, renderer=None):
        # 如果未提供渲染器，则使用图形对象的渲染器
        if renderer is None:
            renderer = self.figure._get_renderer()

        # 将点转换为像素，用于计算 DPI 校正
        dpi_cor = renderer.points_to_pixels(1.)
        
        # 如果主刻度可见且刻度线超出轴边界
        if self.major_ticks.get_visible() and self.major_ticks.get_tick_out():
            # 计算刻度标签的外部填充
            ticklabel_pad = self.major_ticks._ticksize * dpi_cor
            # 设置主刻度标签的外部填充
            self.major_ticklabels._external_pad = ticklabel_pad
            # 设置次刻度标签的外部填充
            self.minor_ticklabels._external_pad = ticklabel_pad
        else:
            # 否则将外部填充设为零
            self.major_ticklabels._external_pad = 0
            self.minor_ticklabels._external_pad = 0

        # 获取主刻度和次刻度的迭代器
        majortick_iter, minortick_iter = \
            self._axis_artist_helper.get_tick_iterators(self.axes)

        # 获取刻度位置和角度信息以及刻度标签位置、角度和标签信息
        tick_loc_angle, ticklabel_loc_angle_label = \
            self._get_tick_info(majortick_iter)
        # 设置主刻度的位置和角度
        self.major_ticks.set_locs_angles(tick_loc_angle)
        # 设置主刻度标签的位置、角度和标签
        self.major_ticklabels.set_locs_angles_labels(ticklabel_loc_angle_label)

        # 获取次刻度的位置和角度信息以及刻度标签位置、角度和标签信息
        tick_loc_angle, ticklabel_loc_angle_label = \
            self._get_tick_info(minortick_iter)
        # 设置次刻度的位置和角度
        self.minor_ticks.set_locs_angles(tick_loc_angle)
        # 设置次刻度标签的位置、角度和标签
        self.minor_ticklabels.set_locs_angles_labels(ticklabel_loc_angle_label)

    # 绘制刻度
    def _draw_ticks(self, renderer):
        # 更新刻度
        self._update_ticks(renderer)
        # 绘制主刻度
        self.major_ticks.draw(renderer)
        # 绘制主刻度标签
        self.major_ticklabels.draw(renderer)
        # 绘制次刻度
        self.minor_ticks.draw(renderer)
        # 绘制次刻度标签
        self.minor_ticklabels.draw(renderer)
        # 如果主刻度标签或次刻度标签可见，则绘制偏移文本
        if (self.major_ticklabels.get_visible()
                or self.minor_ticklabels.get_visible()):
            self._draw_offsetText(renderer)

    # 偏移文本的位置映射
    _offsetText_pos = dict(left=(0, 1, "bottom", "right"),
                           right=(1, 1, "bottom", "left"),
                           bottom=(1, 0, "top", "right"),
                           top=(1, 1, "bottom", "right"))

    # 初始化偏移文本
    def _init_offsetText(self, direction):
        # 根据方向选择位置参数
        x, y, va, ha = self._offsetText_pos[direction]
        # 创建注释对象作为偏移文本
        self.offsetText = mtext.Annotation(
            "",
            xy=(x, y), xycoords="axes fraction",
            xytext=(0, 0), textcoords="offset points",
            color=mpl.rcParams['xtick.color'],
            horizontalalignment=ha, verticalalignment=va,
        )
        # 设置变换为单位矩阵变换
        self.offsetText.set_transform(IdentityTransform())
        # 设置艺术家属性
        self.axes._set_artist_props(self.offsetText)

    # 更新偏移文本内容和位置
    def _update_offsetText(self):
        # 设置偏移文本的文本内容为轴的主要格式化程序的偏移量
        self.offsetText.set_text(self.axis.major.formatter.get_offset())
        # 设置偏移文本的大小为主刻度标签的大小
        self.offsetText.set_size(self.major_ticklabels.get_size())
        # 计算偏移量
        offset = (self.major_ticklabels.get_pad()
                  + self.major_ticklabels.get_size()
                  + 2)
        # 设置偏移文本的位置
        self.offsetText.xyann = (0, offset)

    # 绘制偏移文本
    def _draw_offsetText(self, renderer):
        # 更新偏移文本
        self._update_offsetText()
        # 绘制偏移文本
        self.offsetText.draw(renderer)
    # 初始化标签，根据传入的关键字参数设置标签属性
    def _init_label(self, **kwargs):
        # 获取标签的转换信息，包括坐标轴的转换和偏移量的转换
        tr = (self._axis_artist_helper.get_axislabel_transform(self.axes)
              + self.offset_transform)
        # 创建轴标签对象
        self.label = AxisLabel(
            0, 0, "__from_axes__",
            color="auto",
            fontsize=kwargs.get("labelsize", mpl.rcParams['axes.labelsize']),
            fontweight=mpl.rcParams['axes.labelweight'],
            axis=self.axis,
            transform=tr,
            axis_direction=self._axis_direction,
        )
        # 将轴标签关联到所属的图形对象上
        self.label.set_figure(self.axes.figure)
        # 设置标签的填充量
        labelpad = kwargs.get("labelpad", 5)
        self.label.set_pad(labelpad)

    # 更新标签的显示
    def _update_label(self, renderer):
        # 如果标签不可见，则直接返回
        if not self.label.get_visible():
            return

        # 根据条件设置轴标签的外部填充量
        if self._ticklabel_add_angle != self._axislabel_add_angle:
            if ((self.major_ticks.get_visible()
                 and not self.major_ticks.get_tick_out())
                or (self.minor_ticks.get_visible()
                    and not self.major_ticks.get_tick_out())):
                axislabel_pad = self.major_ticks._ticksize
            else:
                axislabel_pad = 0
        else:
            axislabel_pad = max(self.major_ticklabels._axislabel_pad,
                                self.minor_ticklabels._axislabel_pad)

        # 设置标签的外部填充量
        self.label._external_pad = axislabel_pad

        # 获取标签的位置和角度信息
        xy, angle_tangent = \
            self._axis_artist_helper.get_axislabel_pos_angle(self.axes)
        # 如果位置信息为空，则直接返回
        if xy is None:
            return

        # 计算标签的角度
        angle_label = angle_tangent - 90

        # 设置标签的位置和角度
        x, y = xy
        self.label._ref_angle = angle_label + self._axislabel_add_angle
        self.label.set(x=x, y=y)

    # 绘制标签
    def _draw_label(self, renderer):
        # 更新标签的显示
        self._update_label(renderer)
        # 绘制标签
        self.label.draw(renderer)

    # 设置标签文本内容
    def set_label(self, s):
        # 继承的文档字符串
        self.label.set_text(s)

    # 获取紧凑的包围盒
    def get_tightbbox(self, renderer=None):
        # 如果标签不可见，则返回空
        if not self.get_visible():
            return
        # 更新轴的限制信息
        self._axis_artist_helper.update_lim(self.axes)
        # 更新刻度信息
        self._update_ticks(renderer)
        # 更新标签信息
        self._update_label(renderer)

        # 设置轴线路径
        self.line.set_path(self._axis_artist_helper.get_line(self.axes))
        # 如果轴线风格不为空，则设置线条变异比例
        if self.get_axisline_style() is not None:
            self.line.set_line_mutation_scale(self.major_ticklabels.get_size())

        # 获取所有元素的窗口扩展信息
        bb = [
            *self.major_ticklabels.get_window_extents(renderer),
            *self.minor_ticklabels.get_window_extents(renderer),
            self.label.get_window_extent(renderer),
            self.offsetText.get_window_extent(renderer),
            self.line.get_window_extent(renderer),
        ]
        # 过滤掉宽度或高度为0的元素
        bb = [b for b in bb if b and (b.width != 0 or b.height != 0)]
        # 如果存在有效的包围盒，则返回合并后的包围盒；否则返回空
        if bb:
            _bbox = Bbox.union(bb)
            return _bbox
        else:
            return None

    # 允许栅格化操作
    @martist.allow_rasterization
    def draw(self, renderer):
        # 如果对象不可见，则直接返回，不进行绘制操作
        if not self.get_visible():
            return
        # 打开一个新的渲染组，使用当前对象的 gid 作为组的唯一标识符
        renderer.open_group(__name__, gid=self.get_gid())
        # 更新轴辅助对象的限制范围，以匹配当前轴对象的设置
        self._axis_artist_helper.update_lim(self.axes)
        # 绘制轴上的刻度线
        self._draw_ticks(renderer)
        # 绘制轴线
        self._draw_line(renderer)
        # 绘制轴标签
        self._draw_label(renderer)
        # 关闭之前打开的渲染组
        renderer.close_group(__name__)

    def toggle(self, all=None, ticks=None, ticklabels=None, label=None):
        """
        切换刻度、刻度标签和轴标签的可见性。
        若要全部关闭，使用::

          axis.toggle(all=False)

        若要关闭全部但打开刻度线，使用::

          axis.toggle(all=False, ticks=True)

        若要打开全部但关闭轴标签，使用::

          axis.toggle(all=True, label=False)

        """
        # 根据传入参数设置刻度、刻度标签和轴标签的可见性
        if all:
            _ticks, _ticklabels, _label = True, True, True
        elif all is not None:
            _ticks, _ticklabels, _label = False, False, False
        else:
            _ticks, _ticklabels, _label = None, None, None

        # 根据 ticks 参数设置刻度的可见性
        if ticks is not None:
            _ticks = ticks
        # 根据 ticklabels 参数设置刻度标签的可见性
        if ticklabels is not None:
            _ticklabels = ticklabels
        # 根据 label 参数设置轴标签的可见性
        if label is not None:
            _label = label

        # 根据上述设置更新轴对象的可见性
        if _ticks is not None:
            self.major_ticks.set_visible(_ticks)
            self.minor_ticks.set_visible(_ticks)
        if _ticklabels is not None:
            self.major_ticklabels.set_visible(_ticklabels)
            self.minor_ticklabels.set_visible(_ticklabels)
        if _label is not None:
            self.label.set_visible(_label)
```