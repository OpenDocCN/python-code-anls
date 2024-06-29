# `D:\src\scipysrc\matplotlib\lib\matplotlib\text.py`

```py
"""
Classes for including text in a figure.
"""

# 导入所需模块和库
import functools  # 导入 functools 模块
import logging  # 导入 logging 模块
import math  # 导入 math 模块
from numbers import Real  # 从 numbers 模块导入 Real 类型
import weakref  # 导入 weakref 模块

import numpy as np  # 导入 numpy 库

import matplotlib as mpl  # 导入 matplotlib 库并简写为 mpl
from . import _api, artist, cbook, _docstring  # 导入本地模块和资源
from .artist import Artist  # 从本地模块导入 Artist 类
from .font_manager import FontProperties  # 从本地模块导入 FontProperties 类
from .patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # 从本地模块导入特定的类
from .textpath import TextPath, TextToPath  # noqa # 从本地模块导入 TextPath 和 TextToPath 类，禁止 pylint 检查
from .transforms import (  # 从本地模块导入多个类
    Affine2D, Bbox, BboxBase, BboxTransformTo, IdentityTransform, Transform)

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


def _get_textbox(text, renderer):
    """
    Calculate the bounding box of the text.

    The bbox position takes text rotation into account, but the width and
    height are those of the unrotated box (unlike `.Text.get_window_extent`).
    """
    # 计算文本的边界框
    # 该函数可能会作为 Text 类的一个方法进行移动，因为在 Text._get_layout() 调用时，
    # 可能需要 _get_textbox 函数中的信息。因此最好将此函数作为一个方法，并对 _get_layout 方法进行重构。

    projected_xs = []  # 初始化存储投影 X 值的列表
    projected_ys = []  # 初始化存储投影 Y 值的列表

    theta = np.deg2rad(text.get_rotation())  # 获取文本旋转角度并转换为弧度
    tr = Affine2D().rotate(-theta)  # 创建一个旋转矩阵对象，逆时针旋转角度 theta

    _, parts, d = text._get_layout(renderer)  # 调用文本对象的 _get_layout 方法获取布局信息

    for t, wh, x, y in parts:
        w, h = wh  # 获取文本块的宽度和高度

        xt1, yt1 = tr.transform((x, y))  # 应用旋转矩阵获取旋转后的坐标
        yt1 -= d  # 调整 Y 坐标以考虑下降距离
        xt2, yt2 = xt1 + w, yt1 + h  # 计算文本块的对角坐标

        projected_xs.extend([xt1, xt2])  # 将计算得到的 X 坐标添加到列表中
        projected_ys.extend([yt1, yt2])  # 将计算得到的 Y 坐标添加到列表中

    xt_box, yt_box = min(projected_xs), min(projected_ys)  # 计算边界框的左上角坐标
    w_box, h_box = max(projected_xs) - xt_box, max(projected_ys) - yt_box  # 计算边界框的宽度和高度

    x_box, y_box = Affine2D().rotate(theta).transform((xt_box, yt_box))  # 应用反向旋转矩阵计算边界框的位置

    return x_box, y_box, w_box, h_box  # 返回边界框的位置和尺寸


def _get_text_metrics_with_cache(renderer, text, fontprop, ismath, dpi):
    """Call ``renderer.get_text_width_height_descent``, caching the results."""
    # 使用缓存调用 renderer.get_text_width_height_descent 方法获取文本度量信息
    # 使用 fontprop 的副本进行缓存，以防止后续对传入参数的就地修改影响缓存
    return _get_text_metrics_with_cache_impl(
        weakref.ref(renderer), text, fontprop.copy(), ismath, dpi)


@functools.lru_cache(4096)
def _get_text_metrics_with_cache_impl(
        renderer_ref, text, fontprop, ismath, dpi):
    # 使用缓存调用 renderer.get_text_width_height_descent 方法获取文本度量信息的实现
    # dpi 参数未使用，但通过 renderer 参与缓存失效
    return renderer_ref().get_text_width_height_descent(text, fontprop, ismath)


@_docstring.interpd
@_api.define_aliases({
    "color": ["c"],
    "fontproperties": ["font", "font_properties"],
    "fontfamily": ["family"],
    "fontname": ["name"],
    "fontsize": ["size"],
    "fontstretch": ["stretch"],
    "fontstyle": ["style"],
    "fontvariant": ["variant"],
    "fontweight": ["weight"],
    "horizontalalignment": ["ha"],
    "verticalalignment": ["va"],
    "multialignment": ["ma"],
})
class Text(Artist):
    # 文本类，用于在图中包含文本
    # 处理文本在窗口或数据坐标中的存储和绘制。
    
    zorder = 3
    # 文本对象的绘制顺序，默认为3
    _charsize_cache = dict()
    # 字符大小的缓存字典
    
    def __repr__(self):
        return f"Text({self._x}, {self._y}, {self._text!r})"
    # 返回文本对象的可打印表示，包括其位置和文本内容
    
    def __init__(self,
                 x=0, y=0, text='', *,
                 color=None,           # 默认为rc参数
                 verticalalignment='baseline',
                 horizontalalignment='left',
                 multialignment=None,
                 fontproperties=None,  # 默认为FontProperties()
                 rotation=None,
                 linespacing=None,
                 rotation_mode=None,
                 usetex=None,          # 默认为rcParams['text.usetex']
                 wrap=False,
                 transform_rotates_text=False,
                 parse_math=None,    # 默认为rcParams['text.parse_math']
                 antialiased=None,  # 默认为rcParams['text.antialiased']
                 **kwargs
                 ):
        """
        在位置(x, y)创建一个Text实例，文本内容为text。
    
        文本相对锚点(x, y)按照horizontalalignment（默认为'left'）和
        verticalalignment（默认为'baseline'）对齐。参见
        :doc:`/gallery/text_labels_and_annotations/text_alignment`。
    
        虽然Text接受'label'关键字参数，但默认情况下不会添加到图例的句柄中。
    
        有效的关键字参数包括：
    
        %(Text:kwdoc)s
        """
        super().__init__()
        self._x, self._y = x, y
        self._text = ''
        # 初始化文本内容为空字符串
        self._reset_visual_defaults(
            text=text,
            color=color,
            fontproperties=fontproperties,
            usetex=usetex,
            parse_math=parse_math,
            wrap=wrap,
            verticalalignment=verticalalignment,
            horizontalalignment=horizontalalignment,
            multialignment=multialignment,
            rotation=rotation,
            transform_rotates_text=transform_rotates_text,
            linespacing=linespacing,
            rotation_mode=rotation_mode,
            antialiased=antialiased
        )
        self.update(kwargs)
    
    def _reset_visual_defaults(
        self,
        text='',
        color=None,
        fontproperties=None,
        usetex=None,
        parse_math=None,
        wrap=False,
        verticalalignment='baseline',
        horizontalalignment='left',
        multialignment=None,
        rotation=None,
        transform_rotates_text=False,
        linespacing=None,
        rotation_mode=None,
        antialiased=None
    ):
        # 重置文本对象的视觉默认设置，包括文本内容和各种样式参数
    ):
        # 设置文本内容
        self.set_text(text)
        # 设置文本颜色，使用 mpl._val_or_rc 获取默认配置或者用户自定义配置
        self.set_color(mpl._val_or_rc(color, "text.color"))
        # 设置字体属性
        self.set_fontproperties(fontproperties)
        # 设置是否使用 LaTeX 渲染
        self.set_usetex(usetex)
        # 设置是否解析数学公式
        self.set_parse_math(mpl._val_or_rc(parse_math, 'text.parse_math'))
        # 设置文本是否换行
        self.set_wrap(wrap)
        # 设置垂直对齐方式
        self.set_verticalalignment(verticalalignment)
        # 设置水平对齐方式
        self.set_horizontalalignment(horizontalalignment)
        # 设置多行文本的对齐方式
        self._multialignment = multialignment
        # 设置文本旋转角度
        self.set_rotation(rotation)
        # 设置文本旋转时是否变换
        self._transform_rotates_text = transform_rotates_text
        # 初始化为 None，用于存储 FancyBboxPatch 实例
        self._bbox_patch = None  # a FancyBboxPatch instance
        # 初始化为 None，用于存储渲染器实例
        self._renderer = None
        # 如果未指定行间距，使用默认值 1.2，可能会在后续使用 rcParam 覆盖
        if linespacing is None:
            linespacing = 1.2  # Maybe use rcParam later.
        # 设置行间距
        self.set_linespacing(linespacing)
        # 设置文本旋转模式
        self.set_rotation_mode(rotation_mode)
        # 设置是否反锯齿，如果未指定则使用 rcParams 中的默认设置
        self.set_antialiased(antialiased if antialiased is not None else
                             mpl.rcParams['text.antialiased'])

    def update(self, kwargs):
        # docstring inherited
        # 初始化空列表用于存储返回值
        ret = []
        # 标准化 kwargs 参数
        kwargs = cbook.normalize_kwargs(kwargs, Text)
        # 使用对象作为哨兵值，以处理 bbox 可能为 None 的情况
        sentinel = object()  # bbox can be None, so use another sentinel.
        # 首先更新字体属性，因为它具有最低优先级
        fontproperties = kwargs.pop("fontproperties", sentinel)
        if fontproperties is not sentinel:
            ret.append(self.set_fontproperties(fontproperties))
        # 最后更新 bbox，因为它依赖于字体属性
        bbox = kwargs.pop("bbox", sentinel)
        # 调用父类的 update 方法，并将结果扩展到 ret 列表中
        ret.extend(super().update(kwargs))
        if bbox is not sentinel:
            # 如果 bbox 已更新，则设置新的 bbox
            ret.append(self.set_bbox(bbox))
        return ret

    def __getstate__(self):
        # 获取父类的状态字典
        d = super().__getstate__()
        # 移除缓存的 _renderer（如果存在）
        d['_renderer'] = None
        return d

    def contains(self, mouseevent):
        """
        Return whether the mouse event occurred inside the axis-aligned
        bounding-box of the text.
        """
        # 如果鼠标事件发生在不同的画布上、文本不可见或者 _renderer 为 None，则返回 False 和空字典
        if (self._different_canvas(mouseevent) or not self.get_visible()
                or self._renderer is None):
            return False, {}
        # 显式使用 Text.get_window_extent(self)，而不是 self.get_window_extent()，以避免误覆盖整个注释框
        # 获取文本的窗口范围（bounding box）
        bbox = Text.get_window_extent(self)
        # 检查鼠标事件是否在文本的边界框内
        inside = (bbox.x0 <= mouseevent.x <= bbox.x1
                  and bbox.y0 <= mouseevent.y <= bbox.y1)
        cattr = {}
        # 如果文本周围有一个补丁（patch），也检查其包含情况，并将结果与文本的结果合并
        if self._bbox_patch:
            patch_inside, patch_cattr = self._bbox_patch.contains(mouseevent)
            inside = inside or patch_inside
            cattr["bbox_patch"] = patch_cattr
        return inside, cattr
    def _get_xy_display(self):
        """
        Get the (possibly unit converted) transformed x, y in display coords.
        """
        # 获取当前对象的无单位位置坐标
        x, y = self.get_unitless_position()
        # 使用当前对象的变换获取转换后的 x, y 坐标（可能已进行单位转换）
        return self.get_transform().transform((x, y))

    def _get_multialignment(self):
        if self._multialignment is not None:
            return self._multialignment
        else:
            return self._horizontalalignment

    def _char_index_at(self, x):
        """
        Calculate the index closest to the coordinate x in display space.

        The position of text[index] is assumed to be the sum of the widths
        of all preceding characters text[:index].

        This works only on single line texts.
        """
        if not self._text:
            return 0

        # 获取文本内容
        text = self._text

        # 获取字体属性并将其转换为字符串格式
        fontproperties = str(self._fontproperties)
        # 如果缓存中没有当前字体属性的字符尺寸信息，则创建一个新的缓存字典
        if fontproperties not in Text._charsize_cache:
            Text._charsize_cache[fontproperties] = dict()

        # 从缓存中获取字符尺寸信息
        charsize_cache = Text._charsize_cache[fontproperties]
        # 对文本中的每个字符进行遍历，如果字符尺寸信息不在缓存中，则将其添加到缓存中
        for char in set(text):
            if char not in charsize_cache:
                self.set_text(char)
                bb = self.get_window_extent()
                charsize_cache[char] = bb.x1 - bb.x0

        # 设置文本内容为原始的文本内容
        self.set_text(text)
        # 获取文本在窗口中的边界框
        bb = self.get_window_extent()

        # 计算文本每个字符累积宽度
        size_accum = np.cumsum([0] + [charsize_cache[x] for x in text])
        # 标准化 x 坐标
        std_x = x - bb.x0
        # 返回最接近标准化 x 坐标的字符索引
        return (np.abs(size_accum - std_x)).argmin()

    def get_rotation(self):
        """Return the text angle in degrees between 0 and 360."""
        # 如果文本对象的变换会旋转文本方向，则计算旋转后的角度
        if self.get_transform_rotates_text():
            return self.get_transform().transform_angles(
                [self._rotation], [self.get_unitless_position()]).item(0)
        else:
            # 否则返回文本对象的旋转角度
            return self._rotation

    def get_transform_rotates_text(self):
        """
        Return whether rotations of the transform affect the text direction.
        """
        # 返回文本对象的变换是否影响文本方向的布尔值
        return self._transform_rotates_text

    def set_rotation_mode(self, m):
        """
        Set text rotation mode.

        Parameters
        ----------
        m : {None, 'default', 'anchor'}
            If ``"default"``, the text will be first rotated, then aligned according
            to their horizontal and vertical alignments.  If ``"anchor"``, then
            alignment occurs before rotation. Passing ``None`` will set the rotation
            mode to ``"default"``.
        """
        # 如果输入参数 m 为 None，则设置为默认模式 "default"
        if m is None:
            m = "default"
        else:
            # 否则检查输入参数 m 是否在允许的列表中
            _api.check_in_list(("anchor", "default"), rotation_mode=m)
        # 设置对象的旋转模式为 m
        self._rotation_mode = m
        # 设置对象为过时状态，需要重新绘制
        self.stale = True

    def get_rotation_mode(self):
        """Return the text rotation mode."""
        # 返回对象的旋转模式
        return self._rotation_mode
    def set_antialiased(self, antialiased):
        """
        设置是否使用抗锯齿渲染。

        Parameters
        ----------
        antialiased : bool
            是否使用抗锯齿渲染的布尔值。

        Notes
        -----
        抗锯齿渲染将由 :rc:`text.antialiased` 决定，
        如果文本中包含数学表达式，则参数 *antialiased* 将不起作用。
        """
        # 将输入的抗锯齿设置值赋给对象的属性
        self._antialiased = antialiased
        # 将对象的状态标记为过期
        self.stale = True

    def get_antialiased(self):
        """
        返回是否使用抗锯齿渲染。
        """
        return self._antialiased

    def update_from(self, other):
        """
        从另一个对象中更新当前对象的属性。

        Parameters
        ----------
        other : object
            另一个对象，从该对象复制属性。

        Notes
        -----
        该方法继承自父类，并复制另一个对象的各种属性到当前对象。
        """
        # 调用父类方法，从另一个对象复制继承的属性
        super().update_from(other)
        # 从另一个对象复制具体属性值到当前对象
        self._color = other._color
        self._multialignment = other._multialignment
        self._verticalalignment = other._verticalalignment
        self._horizontalalignment = other._horizontalalignment
        self._fontproperties = other._fontproperties.copy()
        self._usetex = other._usetex
        self._rotation = other._rotation
        self._transform_rotates_text = other._transform_rotates_text
        self._picker = other._picker
        self._linespacing = other._linespacing
        self._antialiased = other._antialiased
        # 将对象的状态标记为过期
        self.stale = True

    def set_bbox(self, rectprops):
        """
        绘制一个边界框围绕当前对象。

        Parameters
        ----------
        rectprops : dict
            包含 `.patches.FancyBboxPatch` 的属性的字典。
            默认的 boxstyle 是 'square'。`.patches.FancyBboxPatch` 的
            mutation scale 被设置为字体大小。

        Examples
        --------
        ::

            t.set_bbox(dict(facecolor='red', alpha=0.5))
        """

        if rectprops is not None:
            # 复制给定的边界框属性
            props = rectprops.copy()
            # 从属性中提取 boxstyle 和 pad
            boxstyle = props.pop("boxstyle", None)
            pad = props.pop("pad", None)
            if boxstyle is None:
                boxstyle = "square"
                if pad is None:
                    pad = 4  # 点
                # 将 pad 转换为相对于字体大小的分数
                pad /= self.get_size()
            else:
                if pad is None:
                    pad = 0.3
            # 如果 boxstyle 是字符串且未包含 'pad'，则添加 pad 属性
            if isinstance(boxstyle, str) and "pad" not in boxstyle:
                boxstyle += ",pad=%0.2f" % pad
            # 创建 FancyBboxPatch 对象，并设置属性
            self._bbox_patch = FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle=boxstyle, transform=IdentityTransform(), **props)
        else:
            # 如果 rectprops 为 None，则将 _bbox_patch 设置为 None
            self._bbox_patch = None

        # 更新剪辑属性
        self._update_clip_properties()

    def get_bbox_patch(self):
        """
        返回边界框 Patch，如果 `.patches.FancyBboxPatch` 未创建则返回 None。
        """
        return self._bbox_patch
    # 更新边界框（bbox）的位置和大小
    def update_bbox_position_size(self, renderer):
        """
        Update the location and the size of the bbox.

        This method should be used when the position and size of the bbox needs
        to be updated before actually drawing the bbox.
        """
        if self._bbox_patch:
            # 不要在此处使用 self.get_unitless_position，该方法是针对文本在 Text 类中的位置
            posx = float(self.convert_xunits(self._x))  # 将 x 坐标转换为绘图单位，并转换为浮点数
            posy = float(self.convert_yunits(self._y))  # 将 y 坐标转换为绘图单位，并转换为浮点数
            posx, posy = self.get_transform().transform((posx, posy))  # 使用当前的转换将坐标转换为绘图坐标系中的位置

            x_box, y_box, w_box, h_box = _get_textbox(self, renderer)  # 获取文本框的位置和大小
            self._bbox_patch.set_bounds(0., 0., w_box, h_box)  # 设置边界框的边界（左下角坐标及宽高）
            self._bbox_patch.set_transform(
                Affine2D()
                .rotate_deg(self.get_rotation())  # 根据文本的旋转角度进行旋转变换
                .translate(posx + x_box, posy + y_box))  # 将边界框平移到指定位置

            fontsize_in_pixel = renderer.points_to_pixels(self.get_size())  # 将文本的字体大小转换为像素大小
            self._bbox_patch.set_mutation_scale(fontsize_in_pixel)  # 设置边界框的变换比例

    # 更新剪切属性
    def _update_clip_properties(self):
        if self._bbox_patch:
            clipprops = dict(clip_box=self.clipbox,  # 剪切框的定义
                             clip_path=self._clippath,  # 剪切路径对象
                             clip_on=self._clipon)  # 是否开启剪切
            self._bbox_patch.update(clipprops)  # 更新边界框的剪切属性

    # 设置剪切框
    def set_clip_box(self, clipbox):
        # 继承的文档字符串。
        super().set_clip_box(clipbox)  # 调用父类的方法设置剪切框
        self._update_clip_properties()  # 更新剪切属性

    # 设置剪切路径
    def set_clip_path(self, path, transform=None):
        # 继承的文档字符串。
        super().set_clip_path(path, transform)  # 调用父类的方法设置剪切路径
        self._update_clip_properties()  # 更新剪切属性

    # 设置是否开启剪切
    def set_clip_on(self, b):
        # 继承的文档字符串。
        super().set_clip_on(b)  # 调用父类的方法设置是否开启剪切
        self._update_clip_properties()  # 更新剪切属性

    # 获取文本是否可以换行
    def get_wrap(self):
        """Return whether the text can be wrapped."""
        return self._wrap

    # 设置文本是否可以换行
    def set_wrap(self, wrap):
        """
        Set whether the text can be wrapped.

        Wrapping makes sure the text is confined to the (sub)figure box. It
        does not take into account any other artists.

        Parameters
        ----------
        wrap : bool

        Notes
        -----
        Wrapping does not work together with
        ``savefig(..., bbox_inches='tight')`` (which is also used internally
        by ``%matplotlib inline`` in IPython/Jupyter). The 'tight' setting
        rescales the canvas to accommodate all content and happens before
        wrapping.
        """
        self._wrap = wrap  # 设置文本是否可以换行的属性
    def _get_wrap_line_width(self):
        """
        Return the maximum line width for wrapping text based on the current
        orientation.
        """
        # 获取当前对象的位置并进行坐标转换
        x0, y0 = self.get_transform().transform(self.get_position())
        # 获取整个图形的边界框
        figure_box = self.get_figure().get_window_extent()

        # 根据文本对齐方式计算可用的宽度
        alignment = self.get_horizontalalignment()
        self.set_rotation_mode('anchor')
        rotation = self.get_rotation()

        # 计算相对于图形边界框的距离
        left = self._get_dist_to_box(rotation, x0, y0, figure_box)
        right = self._get_dist_to_box(
            (180 + rotation) % 360, x0, y0, figure_box)

        # 根据对齐方式选择线宽
        if alignment == 'left':
            line_width = left
        elif alignment == 'right':
            line_width = right
        else:
            line_width = 2 * min(left, right)

        return line_width

    def _get_dist_to_box(self, rotation, x0, y0, figure_box):
        """
        Return the distance from the given points to the boundaries of a
        rotated box, in pixels.
        """
        # 根据旋转角度分四个象限计算到旋转框边界的距离
        if rotation > 270:
            quad = rotation - 270
            h1 = (y0 - figure_box.y0) / math.cos(math.radians(quad))
            h2 = (figure_box.x1 - x0) / math.cos(math.radians(90 - quad))
        elif rotation > 180:
            quad = rotation - 180
            h1 = (x0 - figure_box.x0) / math.cos(math.radians(quad))
            h2 = (y0 - figure_box.y0) / math.cos(math.radians(90 - quad))
        elif rotation > 90:
            quad = rotation - 90
            h1 = (figure_box.y1 - y0) / math.cos(math.radians(quad))
            h2 = (x0 - figure_box.x0) / math.cos(math.radians(90 - quad))
        else:
            h1 = (figure_box.x1 - x0) / math.cos(math.radians(rotation))
            h2 = (figure_box.y1 - y0) / math.cos(math.radians(90 - rotation))

        return min(h1, h2)

    def _get_rendered_text_width(self, text):
        """
        Return the width of a given text string, in pixels.
        """
        # 使用渲染器获取文本的宽度、高度和下降值
        w, h, d = self._renderer.get_text_width_height_descent(
            text,
            self.get_fontproperties(),
            cbook.is_math_text(text))
        return math.ceil(w)
    # 返回一个包含换行符的文本副本，以便根据父图形进行文本换行（如果 get_wrap 返回 True）
    def _get_wrapped_text(self):
        if not self.get_wrap():  # 如果 get_wrap 方法返回 False，直接返回原始文本
            return self.get_text()

        # 对于 LaTeX 语法，暂时无法正确处理分割，因此忽略 LaTeX
        if self.get_usetex():
            return self.get_text()

        # 获取文本行的宽度限制
        line_width = self._get_wrap_line_width()
        wrapped_lines = []  # 存储包装后的文本行

        # 按换行符拆分用户文本
        unwrapped_lines = self.get_text().split('\n')

        # 对每一行进行文本包装
        for unwrapped_line in unwrapped_lines:
            sub_words = unwrapped_line.split(' ')  # 将每一行进一步拆分为单词
            while len(sub_words) > 0:  # 当仍有单词未处理时继续
                if len(sub_words) == 1:
                    # 只有一个单词，直接添加到结果中
                    wrapped_lines.append(sub_words.pop(0))
                    continue

                for i in range(2, len(sub_words) + 1):
                    # 获取直到当前位置的所有单词的宽度
                    line = ' '.join(sub_words[:i])
                    current_width = self._get_rendered_text_width(line)

                    # 如果这些单词的宽度超过了行宽度，则添加前面的单词到结果中
                    if current_width > line_width:
                        wrapped_lines.append(' '.join(sub_words[:i - 1]))
                        sub_words = sub_words[i - 1:]
                        break

                    # 否则如果所有单词都适合行宽度，则全部添加到结果中
                    elif i == len(sub_words):
                        wrapped_lines.append(' '.join(sub_words[:i]))
                        sub_words = []
                        break

        # 返回包装后的文本，使用换行符连接所有行
        return '\n'.join(wrapped_lines)

    @artist.allow_rasterization
    def draw(self, renderer):
        # 绘制文本到指定的渲染器上

        # 如果渲染器不为None，则设置当前对象的渲染器为指定的渲染器
        if renderer is not None:
            self._renderer = renderer
        
        # 如果对象不可见，则返回
        if not self.get_visible():
            return
        
        # 如果文本内容为空字符串，则返回
        if self.get_text() == '':
            return

        # 在渲染器中打开一个名为'text'，ID为当前对象ID的分组
        renderer.open_group('text', self.get_gid())

        # 使用文本属性设置当前文本内容的上下文管理器，获取布局信息和下降值
        with self._cm_set(text=self._get_wrapped_text()):
            bbox, info, descent = self._get_layout(renderer)
            trans = self.get_transform()

            # 不使用self.get_position，因为它是Text中文本位置的引用
            posx = float(self.convert_xunits(self._x))
            posy = float(self.convert_yunits(self._y))
            posx, posy = trans.transform((posx, posy))

            # 检查posx和posy是否为有限值，如果不是则记录警告并返回
            if not np.isfinite(posx) or not np.isfinite(posy):
                _log.warning("posx and posy should be finite values")
                return
            
            # 获取渲染器的画布宽度和高度
            canvasw, canvash = renderer.get_canvas_width_height()

            # 如果存在bbox_patch，则更新其位置和大小，并绘制它
            if self._bbox_patch:
                self.update_bbox_position_size(renderer)
                self._bbox_patch.draw(renderer)

            # 创建新的图形上下文
            gc = renderer.new_gc()
            gc.set_foreground(self.get_color())
            gc.set_alpha(self.get_alpha())
            gc.set_url(self._url)
            gc.set_antialiased(self._antialiased)
            self._set_gc_clip(gc)

            # 获取文本的旋转角度
            angle = self.get_rotation()

            # 遍历每一行文本的信息
            for line, wh, x, y in info:
                # 如果只有一行文本，将mtext设置为self，否则设为None
                mtext = self if len(info) == 1 else None
                x = x + posx
                y = y + posy
                
                # 如果渲染器是垂直翻转的，则更新y坐标
                if renderer.flipy():
                    y = canvash - y
                
                # 预处理数学文本，获取干净的行和数学状态
                clean_line, ismath = self._preprocess_math(line)

                # 如果存在路径效果，则使用PathEffectRenderer进行文本渲染，否则使用普通的renderer
                if self.get_path_effects():
                    from matplotlib.patheffects import PathEffectRenderer
                    textrenderer = PathEffectRenderer(
                        self.get_path_effects(), renderer)
                else:
                    textrenderer = renderer

                # 根据usetex属性选择绘制TeX文本或普通文本
                if self.get_usetex():
                    textrenderer.draw_tex(gc, x, y, clean_line,
                                          self._fontproperties, angle,
                                          mtext=mtext)
                else:
                    textrenderer.draw_text(gc, x, y, clean_line,
                                           self._fontproperties, angle,
                                           ismath=ismath, mtext=mtext)

        # 恢复图形上下文的状态
        gc.restore()
        
        # 在渲染器中关闭名为'text'的分组
        renderer.close_group('text')
        
        # 将对象的stale标志设置为False，表示对象不需要更新
        self.stale = False

    def get_color(self):
        """返回文本的颜色。"""
        return self._color

    def get_fontproperties(self):
        """返回文本的字体属性对象`.font_manager.FontProperties`。"""
        return self._fontproperties
    def get_fontfamily(self):
        """
        Return the list of font families used for font lookup.

        See Also
        --------
        .font_manager.FontProperties.get_family
        """
        # 调用 _fontproperties 对象的 get_family 方法，返回字体系列列表
        return self._fontproperties.get_family()

    def get_fontname(self):
        """
        Return the font name as a string.

        See Also
        --------
        .font_manager.FontProperties.get_name
        """
        # 调用 _fontproperties 对象的 get_name 方法，返回字体名称字符串
        return self._fontproperties.get_name()

    def get_fontstyle(self):
        """
        Return the font style as a string.

        See Also
        --------
        .font_manager.FontProperties.get_style
        """
        # 调用 _fontproperties 对象的 get_style 方法，返回字体样式字符串
        return self._fontproperties.get_style()

    def get_fontsize(self):
        """
        Return the font size as an integer.

        See Also
        --------
        .font_manager.FontProperties.get_size_in_points
        """
        # 调用 _fontproperties 对象的 get_size_in_points 方法，返回字体大小整数
        return self._fontproperties.get_size_in_points()

    def get_fontvariant(self):
        """
        Return the font variant as a string.

        See Also
        --------
        .font_manager.FontProperties.get_variant
        """
        # 调用 _fontproperties 对象的 get_variant 方法，返回字体变体字符串
        return self._fontproperties.get_variant()

    def get_fontweight(self):
        """
        Return the font weight as a string or a number.

        See Also
        --------
        .font_manager.FontProperties.get_weight
        """
        # 调用 _fontproperties 对象的 get_weight 方法，返回字体粗细字符串或数字
        return self._fontproperties.get_weight()

    def get_stretch(self):
        """
        Return the font stretch as a string or a number.

        See Also
        --------
        .font_manager.FontProperties.get_stretch
        """
        # 调用 _fontproperties 对象的 get_stretch 方法，返回字体拉伸字符串或数字
        return self._fontproperties.get_stretch()

    def get_horizontalalignment(self):
        """
        Return the horizontal alignment as a string.  Will be one of
        'left', 'center' or 'right'.
        """
        # 返回对象的水平对齐方式，可能是 'left'、'center' 或 'right' 中的一个字符串
        return self._horizontalalignment

    def get_unitless_position(self):
        """Return the (x, y) unitless position of the text."""
        # 返回文本的无单位位置坐标 (x, y)
        # 这里将 _x 和 _y 转换为浮点数，去除单位信息后返回
        x = float(self.convert_xunits(self._x))
        y = float(self.convert_yunits(self._y))
        return x, y

    def get_position(self):
        """Return the (x, y) position of the text."""
        # 返回文本的位置坐标 (x, y)
        # 直接返回对象内部的 _x 和 _y 属性值
        return self._x, self._y

    def get_text(self):
        """Return the text string."""
        # 返回文本字符串
        return self._text

    def get_verticalalignment(self):
        """
        Return the vertical alignment as a string.  Will be one of
        'top', 'center', 'bottom', 'baseline' or 'center_baseline'.
        """
        # 返回对象的垂直对齐方式，可能是 'top'、'center'、'bottom'、'baseline' 或 'center_baseline' 中的一个字符串
        return self._verticalalignment
    def get_window_extent(self, renderer=None, dpi=None):
        """
        Return the `.Bbox` bounding the text, in display units.

        In addition to being used internally, this is useful for specifying
        clickable regions in a png file on a web page.

        Parameters
        ----------
        renderer : Renderer, optional
            A renderer is needed to compute the bounding box.  If the artist
            has already been drawn, the renderer is cached; thus, it is only
            necessary to pass this argument when calling `get_window_extent`
            before the first draw.  In practice, it is usually easier to
            trigger a draw first, e.g. by calling
            `~.Figure.draw_without_rendering` or ``plt.show()``.

        dpi : float, optional
            The dpi value for computing the bbox, defaults to
            ``self.figure.dpi`` (*not* the renderer dpi); should be set e.g. if
            to match regions with a figure saved with a custom dpi value.
        """
        # 如果文本不可见，则返回一个单位 Bbox
        if not self.get_visible():
            return Bbox.unit()
        
        # 如果未指定 dpi，则使用 self.figure 的 dpi
        if dpi is None:
            dpi = self.figure.dpi
        
        # 如果文本内容为空字符串，则返回一个 Bbox，位置由 _get_xy_display() 提供
        if self.get_text() == '':
            with cbook._setattr_cm(self.figure, dpi=dpi):
                tx, ty = self._get_xy_display()
                return Bbox.from_bounds(tx, ty, 0, 0)

        # 如果 renderer 不为 None，则设置当前对象的 renderer
        if renderer is not None:
            self._renderer = renderer
        
        # 如果当前对象的 renderer 仍为 None，则尝试从 self.figure 获取 renderer
        if self._renderer is None:
            self._renderer = self.figure._get_renderer()
        
        # 如果获取的 renderer 仍为 None，则抛出 RuntimeError
        if self._renderer is None:
            raise RuntimeError(
                "Cannot get window extent of text w/o renderer. You likely "
                "want to call 'figure.draw_without_rendering()' first.")

        # 使用 with 语句设置 self.figure 的 dpi，并计算文本的布局和边界框
        with cbook._setattr_cm(self.figure, dpi=dpi):
            bbox, info, descent = self._get_layout(self._renderer)
            x, y = self.get_unitless_position()
            x, y = self.get_transform().transform((x, y))
            bbox = bbox.translated(x, y)
            return bbox

    def set_backgroundcolor(self, color):
        """
        Set the background color of the text by updating the bbox.

        Parameters
        ----------
        color : :mpltype:`color`

        See Also
        --------
        .set_bbox : To change the position of the bounding box
        """
        # 如果 _bbox_patch 为空，则通过 set_bbox 方法设置新的 bbox
        if self._bbox_patch is None:
            self.set_bbox(dict(facecolor=color, edgecolor=color))
        else:
            # 否则，直接更新 _bbox_patch 的 facecolor
            self._bbox_patch.update(dict(facecolor=color))

        # 更新剪切属性
        self._update_clip_properties()
        # 将 stale 属性设置为 True，表示需要重新绘制
        self.stale = True
    def set_color(self, color):
        """
        Set the foreground color of the text

        Parameters
        ----------
        color : :mpltype:`color`
            The color to set for the text. Can be any color format supported
            by Matplotlib.

        """
        # "auto" is only supported by axisartist, but we can just let it error
        # out at draw time for simplicity.
        # 检查颜色参数是否为字符串"auto"，如果不是，则验证其是否为合法的颜色格式
        if not cbook._str_equal(color, "auto"):
            mpl.colors._check_color_like(color=color)
        
        # 设置对象的文本颜色属性为指定的颜色值
        self._color = color
        
        # 将对象标记为需要重新绘制，以便颜色变更能够生效
        self.stale = True

    def set_horizontalalignment(self, align):
        """
        Set the horizontal alignment relative to the anchor point.

        See also :doc:`/gallery/text_labels_and_annotations/text_alignment`.

        Parameters
        ----------
        align : {'left', 'center', 'right'}
            The horizontal alignment to set for the text.
        """
        # 检查对齐参数是否属于预定义的对齐方式之一
        _api.check_in_list(['center', 'right', 'left'], align=align)
        
        # 设置对象的水平对齐属性为指定的对齐方式
        self._horizontalalignment = align
        
        # 标记对象为需要重新绘制，以应用新的对齐设置
        self.stale = True

    def set_multialignment(self, align):
        """
        Set the text alignment for multiline texts.

        The layout of the bounding box of all the lines is determined by the
        horizontalalignment and verticalalignment properties. This property
        controls the alignment of the text lines within that box.

        Parameters
        ----------
        align : {'left', 'right', 'center'}
            The multiline alignment to set for the text.
        """
        # 检查多行文本对齐参数是否属于预定义的对齐方式之一
        _api.check_in_list(['center', 'right', 'left'], align=align)
        
        # 设置对象的多行文本对齐属性为指定的对齐方式
        self._multialignment = align
        
        # 标记对象为需要重新绘制，以应用新的多行对齐设置
        self.stale = True

    def set_linespacing(self, spacing):
        """
        Set the line spacing as a multiple of the font size.

        The default line spacing is 1.2.

        Parameters
        ----------
        spacing : float (multiple of font size)
            The line spacing factor to set for the text.
        """
        # 检查行间距参数是否为实数类型
        _api.check_isinstance(Real, spacing=spacing)
        
        # 设置对象的行间距属性为指定的倍数
        self._linespacing = spacing
        
        # 标记对象为需要重新绘制，以应用新的行间距设置
        self.stale = True

    def set_fontfamily(self, fontname):
        """
        Set the font family.  Can be either a single string, or a list of
        strings in decreasing priority.  Each string may be either a real font
        name or a generic font class name.  If the latter, the specific font
        names will be looked up in the corresponding rcParams.

        If a `Text` instance is constructed with ``fontfamily=None``, then the
        font is set to :rc:`font.family`, and the
        same is done when `set_fontfamily()` is called on an existing
        `Text` instance.

        Parameters
        ----------
        fontname : {FONTNAME, 'serif', 'sans-serif', 'cursive', 'fantasy', \
                   'monospace', 'DejaVu Sans', ...}
            The font family name(s) to set for the text. This can be a single
            font name, a list of font names in priority order, or one of the
            generic font class names supported by Matplotlib.
        """
        # 检查字体族参数是否为预定义的字体族或字体名称
        # 这些参数将会在相关的 rcParams 中查找
        # 如果参数为 None，则会使用默认字体
        _api.check_in_list(
            ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace', 
             'DejaVu Sans', ...], fontname=fontname
        )
        
        # 设置对象的字体族属性为指定的字体族或字体名称
        self._fontfamily = fontname
        
        # 标记对象为需要重新绘制，以应用新的字体设置
        self.stale = True
    def set_family(self, fontname):
        """
        Set the font family.

        Parameters
        ----------
        fontname : str
            The name of the font family.

        See Also
        --------
        .font_manager.FontProperties.set_family
        """
        self._fontproperties.set_family(fontname)
        self.stale = True

    def set_variant(self, variant):
        """
        Set the font variant.

        Parameters
        ----------
        variant : {'normal', 'small-caps'}

        See Also
        --------
        .font_manager.FontProperties.set_variant
        """
        self._fontproperties.set_variant(variant)
        self.stale = True

    def set_style(self, fontstyle):
        """
        Set the font style.

        Parameters
        ----------
        fontstyle : {'normal', 'italic', 'oblique'}

        See Also
        --------
        .font_manager.FontProperties.set_style
        """
        self._fontproperties.set_style(fontstyle)
        self.stale = True

    def set_size(self, fontsize):
        """
        Set the font size.

        Parameters
        ----------
        fontsize : float or {'xx-small', 'x-small', 'small', 'medium', \
'large', 'x-large', 'xx-large'}
            If a float, the fontsize in points. The string values denote sizes
            relative to the default font size.

        See Also
        --------
        .font_manager.FontProperties.set_size
        """
        self._fontproperties.set_size(fontsize)
        self.stale = True

    def get_math_fontfamily(self):
        """
        Return the font family name for math text rendered by Matplotlib.

        The default value is :rc:`mathtext.fontset`.

        See Also
        --------
        set_math_fontfamily
        """
        return self._fontproperties.get_math_fontfamily()

    def set_math_fontfamily(self, fontfamily):
        """
        Set the font family for math text rendered by Matplotlib.

        This does only affect Matplotlib's own math renderer. It has no effect
        when rendering with TeX (``usetex=True``).

        Parameters
        ----------
        fontfamily : str
            The name of the font family.

            Available font families are defined in the
            :ref:`default matplotlibrc file
            <customizing-with-matplotlibrc-files>`.

        See Also
        --------
        get_math_fontfamily
        """
        self._fontproperties.set_math_fontfamily(fontfamily)

    def set_weight(self, weight):
        """
        Set the font weight.

        Parameters
        ----------
        weight : {a numeric value in range 0-1000, 'ultralight', 'light', \
'normal', 'regular', 'book', 'medium', 'roman', 'semibold', 'demibold', \
'demi', 'bold', 'heavy', 'extra bold', 'black'}

        See Also
        --------
        .font_manager.FontProperties.set_weight
        """
        self._fontproperties.set_weight(weight)
        self.stale = True
    # 定义方法用于设置字体的水平拉伸（收缩或扩展）

        """
        # 方法的文档字符串，解释了这个方法的作用和用法

        Parameters
        ----------
        # 参数部分开始

        stretch : {a numeric value in range 0-1000, 'ultra-condensed', \
        # stretch 参数可以是一个数值在0到1000之间，或者字符串 'ultra-condensed'
    def set_stretch(self, stretch):
        """
        Set the stretch factor for the font.

        Parameters
        ----------
        stretch : float
            Stretch factor for the font. Valid values are:
            {'extra-condensed', 'condensed', 'semi-condensed', 'normal',
             'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'}

        See Also
        --------
        .font_manager.FontProperties.set_stretch
        """
        self._fontproperties.set_stretch(stretch)
        self.stale = True

    def set_position(self, xy):
        """
        Set the (*x*, *y*) position of the text.

        Parameters
        ----------
        xy : (float, float)
            Tuple containing the x and y coordinates.
        """
        self.set_x(xy[0])
        self.set_y(xy[1])

    def set_x(self, x):
        """
        Set the *x* position of the text.

        Parameters
        ----------
        x : float
            X-coordinate for positioning the text.
        """
        self._x = x
        self.stale = True

    def set_y(self, y):
        """
        Set the *y* position of the text.

        Parameters
        ----------
        y : float
            Y-coordinate for positioning the text.
        """
        self._y = y
        self.stale = True

    def set_rotation(self, s):
        """
        Set the rotation angle of the text.

        Parameters
        ----------
        s : float or {'vertical', 'horizontal'}
            The rotation angle in degrees in mathematically positive direction
            (counterclockwise). 'horizontal' equals 0, 'vertical' equals 90.
        """
        if isinstance(s, Real):
            self._rotation = float(s) % 360
        elif cbook._str_equal(s, 'horizontal') or s is None:
            self._rotation = 0.
        elif cbook._str_equal(s, 'vertical'):
            self._rotation = 90.
        else:
            raise ValueError("rotation must be 'vertical', 'horizontal' or "
                             f"a number, not {s}")
        self.stale = True

    def set_transform_rotates_text(self, t):
        """
        Set whether rotations of the transform affect the text direction.

        Parameters
        ----------
        t : bool
            True if text direction should be affected by transform rotations,
            False otherwise.
        """
        self._transform_rotates_text = t
        self.stale = True

    def set_verticalalignment(self, align):
        """
        Set the vertical alignment of the text relative to the anchor point.

        Parameters
        ----------
        align : {'baseline', 'bottom', 'center', 'center_baseline', 'top'}
            Vertical alignment options for the text.
        """
        _api.check_in_list(
            ['top', 'bottom', 'center', 'baseline', 'center_baseline'],
            align=align)
        self._verticalalignment = align
        self.stale = True

    def set_text(self, s):
        r"""
        Set the text string for the object.

        Parameters
        ----------
        s : object
            Object to be converted to string representation.
        """
        s = '' if s is None else str(s)
        if s != self._text:
            self._text = s
            self.stale = True
    def _preprocess_math(self, s):
        """
        Return the string *s* after mathtext preprocessing, and the kind of
        mathtext support needed.

        - If *self* is configured to use TeX, return *s* unchanged except that
          a single space gets escaped, and the flag "TeX".
        - Otherwise, if *s* is mathtext (has an even number of unescaped dollar
          signs) and ``parse_math`` is not set to False, return *s* and the
          flag True.
        - Otherwise, return *s* with dollar signs unescaped, and the flag
          False.
        """
        # 如果配置为使用 TeX，则对空格进行转义处理后返回字符串 *s* 和 "TeX" 标志
        if self.get_usetex():
            if s == " ":
                s = r"\ "
            return s, "TeX"
        # 如果未配置使用 TeX 且 parse_math 为 True，且 *s* 是数学文本（含有偶数个未转义的美元符号），返回 *s* 和 True 标志
        elif not self.get_parse_math():
            return s, False
        # 如果以上条件不满足，将 *s* 中的美元符号进行转义处理后返回 *s* 和 False 标志
        elif cbook.is_math_text(s):
            return s, True
        else:
            return s.replace(r"\$", "$"), False

    def set_fontproperties(self, fp):
        """
        Set the font properties that control the text.

        Parameters
        ----------
        fp : `.font_manager.FontProperties` or `str` or `pathlib.Path`
            If a `str`, it is interpreted as a fontconfig pattern parsed by
            `.FontProperties`.  If a `pathlib.Path`, it is interpreted as the
            absolute path to a font file.
        """
        # 设置控制文本的字体属性，根据参数 fp 的类型进行解析，并赋值给 self._fontproperties
        self._fontproperties = FontProperties._from_any(fp).copy()
        self.stale = True

    @_docstring.kwarg_doc("bool, default: :rc:`text.usetex`")
    def set_usetex(self, usetex):
        """
        Parameters
        ----------
        usetex : bool or None
            Whether to render using TeX, ``None`` means to use
            :rc:`text.usetex`.
        """
        # 设置是否使用 TeX 渲染文本，根据参数 usetex 的值更新 self._usetex，并设置 self.stale 为 True
        if usetex is None:
            self._usetex = mpl.rcParams['text.usetex']
        else:
            self._usetex = bool(usetex)
        self.stale = True

    def get_usetex(self):
        """Return whether this `Text` object uses TeX for rendering."""
        # 返回当前文本对象是否使用 TeX 渲染的状态，即返回 self._usetex
        return self._usetex

    def set_parse_math(self, parse_math):
        """
        Override switch to disable any mathtext parsing for this `Text`.

        Parameters
        ----------
        parse_math : bool
            If False, this `Text` will never use mathtext.  If True, mathtext
            will be used if there is an even number of unescaped dollar signs.
        """
        # 设置是否解析数学文本的开关，根据参数 parse_math 的值更新 self._parse_math
        self._parse_math = bool(parse_math)

    def get_parse_math(self):
        """Return whether mathtext parsing is considered for this `Text`."""
        # 返回当前文本对象是否考虑解析数学文本的状态，即返回 self._parse_math
        return self._parse_math

    def set_fontname(self, fontname):
        """
        Alias for `set_fontfamily`.

        One-way alias only: the getter differs.

        Parameters
        ----------
        fontname : {FONTNAME, 'serif', 'sans-serif', 'cursive', 'fantasy', \
class OffsetFrom:
    """Callable helper class for working with `Annotation`."""

    def __init__(self, artist, ref_coord, unit="points"):
        """
        Parameters
        ----------
        artist : `~matplotlib.artist.Artist` or `.BboxBase` or `.Transform`
            The object to compute the offset from.

        ref_coord : (float, float)
            If *artist* is an `.Artist` or `.BboxBase`, this values is
            the location to of the offset origin in fractions of the
            *artist* bounding box.

            If *artist* is a transform, the offset origin is the
            transform applied to this value.

        unit : {'points, 'pixels'}, default: 'points'
            The screen units to use (pixels or points) for the offset input.
        """
        self._artist = artist
        x, y = ref_coord  # Make copy when ref_coord is an array (and check the shape).
        self._ref_coord = x, y
        self.set_unit(unit)  # 调用设置单位的方法，设定输入到 transform 的单位

    def set_unit(self, unit):
        """
        Set the unit for input to the transform used by ``__call__``.

        Parameters
        ----------
        unit : {'points', 'pixels'}
        """
        _api.check_in_list(["points", "pixels"], unit=unit)  # 检查单位是否合法
        self._unit = unit  # 设置输入到 transform 的单位

    def get_unit(self):
        """Return the unit for input to the transform used by ``__call__``."""
        return self._unit  # 返回用于 transform 的输入单位

    def __call__(self, renderer):
        """
        Return the offset transform.

        Parameters
        ----------
        renderer : `RendererBase`
            The renderer to use to compute the offset

        Returns
        -------
        `Transform`
            Maps (x, y) in pixel or point units to screen units
            relative to the given artist.
        """
        if isinstance(self._artist, Artist):
            bbox = self._artist.get_window_extent(renderer)  # 获取艺术对象的窗口范围
            xf, yf = self._ref_coord
            x = bbox.x0 + bbox.width * xf  # 计算 x 偏移量
            y = bbox.y0 + bbox.height * yf  # 计算 y 偏移量
        elif isinstance(self._artist, BboxBase):
            bbox = self._artist
            xf, yf = self._ref_coord
            x = bbox.x0 + bbox.width * xf  # 计算 x 偏移量
            y = bbox.y0 + bbox.height * yf  # 计算 y 偏移量
        elif isinstance(self._artist, Transform):
            x, y = self._artist.transform(self._ref_coord)  # 使用 transform 计算偏移量
        else:
            _api.check_isinstance((Artist, BboxBase, Transform), artist=self._artist)  # 检查艺术对象的类型
        scale = 1 if self._unit == "pixels" else renderer.points_to_pixels(1)
        return Affine2D().scale(scale).translate(x, y)  # 返回偏移变换
    # 定义初始化方法，用于创建一个 Annotation 实例
    def __init__(self,
                 xy,
                 xycoords='data',
                 annotation_clip=None):
        # 解构元组 xy，确保对数组进行复制并检查其形状
        x, y = xy  # Make copy when xy is an array (and check the shape).
        # 将解构后的 x, y 存储为对象的实例变量
        self.xy = x, y
        # 设置注解的坐标系统，如果未指定则默认为 'data'
        self.xycoords = xycoords
        # 设置注解是否被剪切的属性，默认为 None
        self.set_annotation_clip(annotation_clip)

        # 初始化时，可拖动的标志置为 None
        self._draggable = None

    # 定义内部方法，用于获取转换后的注解坐标
    def _get_xy(self, renderer, xy, coords):
        # 解构 xy 元组
        x, y = xy
        # 将 coords 转换为元组形式，如果本身不是元组则使用相同的坐标类型
        xcoord, ycoord = coords if isinstance(coords, tuple) else (coords, coords)
        # 如果 x 坐标类型是 'data'，则进行单位转换
        if xcoord == 'data':
            x = float(self.convert_xunits(x))
        # 如果 y 坐标类型是 'data'，则进行单位转换
        if ycoord == 'data':
            y = float(self.convert_yunits(y))
        # 调用内部方法获取坐标变换后的结果，并返回
        return self._get_xy_transform(renderer, coords).transform((x, y))
    # 定义一个方法，用于获取坐标变换对象
    def _get_xy_transform(self, renderer, coords):
        # 如果 coords 是一个元组
        if isinstance(coords, tuple):
            # 将元组拆分为 xcoord 和 ycoord
            xcoord, ycoord = coords
            # 导入混合变换工厂
            from matplotlib.transforms import blended_transform_factory
            # 递归调用 _get_xy_transform 方法获取 xcoord 和 ycoord 的变换对象
            tr1 = self._get_xy_transform(renderer, xcoord)
            tr2 = self._get_xy_transform(renderer, ycoord)
            # 返回混合变换工厂创建的混合变换对象
            return blended_transform_factory(tr1, tr2)
        # 如果 coords 是一个可调用对象
        elif callable(coords):
            # 调用 coords 获取变换对象 tr
            tr = coords(renderer)
            # 如果 tr 是 BboxBase 类型，则返回到该 Bbox 的变换
            if isinstance(tr, BboxBase):
                return BboxTransformTo(tr)
            # 如果 tr 是 Transform 类型，则直接返回 tr
            elif isinstance(tr, Transform):
                return tr
            # 否则抛出类型错误异常
            else:
                raise TypeError(
                    f"xycoords callable must return a BboxBase or Transform, not a "
                    f"{type(tr).__name__}")
        # 如果 coords 是一个 Artist 对象
        elif isinstance(coords, Artist):
            # 获取该 Artist 的窗口范围并返回到其 Bbox 的变换
            bbox = coords.get_window_extent(renderer)
            return BboxTransformTo(bbox)
        # 如果 coords 是一个 BboxBase 对象，则返回到该 Bbox 的变换
        elif isinstance(coords, BboxBase):
            return BboxTransformTo(coords)
        # 如果 coords 是一个 Transform 对象，则直接返回 coords
        elif isinstance(coords, Transform):
            return coords
        # 如果 coords 不是字符串，则抛出类型错误异常
        elif not isinstance(coords, str):
            raise TypeError(
                f"'xycoords' must be an instance of str, tuple[str, str], Artist, "
                f"Transform, or Callable, not a {type(coords).__name__}")

        # 如果 coords 等于 'data'，返回当前坐标系的数据变换
        if coords == 'data':
            return self.axes.transData
        # 如果 coords 等于 'polar'
        elif coords == 'polar':
            # 导入极坐标投影
            from matplotlib.projections import PolarAxes
            # 创建一个不应用极角变换的极坐标变换对象 tr
            tr = PolarAxes.PolarTransform(apply_theta_transforms=False)
            # 将 tr 与当前坐标系的数据变换相结合，并返回结果
            trans = tr + self.axes.transData
            return trans

        # 尝试将 coords 拆分为 bbox_name 和 unit
        try:
            bbox_name, unit = coords.split()
        except ValueError:  # 如果拆分失败，即 len(coords.split()) != 2
            raise ValueError(f"{coords!r} is not a valid coordinate") from None

        bbox0, xy0 = None, None

        # 如果 unit 是类似偏移量的单位
        if bbox_name == "figure":
            bbox0 = self.figure.figbbox
        elif bbox_name == "subfigure":
            bbox0 = self.figure.bbox
        elif bbox_name == "axes":
            bbox0 = self.axes.bbox

        # 如果存在有效的 bbox0，则获取其左下角的坐标作为 xy0
        if bbox0 is not None:
            xy0 = bbox0.p0
        # 如果 bbox_name 是 'offset'，则调用 _get_position_xy 方法获取当前位置的坐标
        elif bbox_name == "offset":
            xy0 = self._get_position_xy(renderer)
        else:
            # 否则抛出值错误异常，说明 coords 不是有效的坐标
            raise ValueError(f"{coords!r} is not a valid coordinate")

        # 根据 unit 不同的单位进行不同的坐标变换处理
        if unit == "points":
            tr = Affine2D().scale(self.figure.dpi / 72)  # dpi/72 每点的像素数
        elif unit == "pixels":
            tr = Affine2D()
        elif unit == "fontsize":
            tr = Affine2D().scale(self.get_size() * self.figure.dpi / 72)
        elif unit == "fraction":
            tr = Affine2D().scale(*bbox0.size)
        else:
            # 如果 unit 不是已知单位，则抛出值错误异常
            raise ValueError(f"{unit!r} is not a recognized unit")

        # 将 tr 应用平移操作，平移到 xy0 所表示的位置
        return tr.translate(*xy0)
    # 设置注释的剪裁行为。
    def set_annotation_clip(self, b):
        """
        Set the annotation's clipping behavior.

        Parameters
        ----------
        b : bool or None
            - True: 当 `self.xy` 在 Axes 外部时，注释将被剪裁。
            - False: 注释始终可见。
            - None: 当 `self.xy` 在 Axes 外部且 `self.xycoords == "data"` 时，注释将被剪裁。
        """
        self._annotation_clip = b

    # 返回注释的剪裁行为。
    def get_annotation_clip(self):
        """
        Return the annotation's clipping behavior.

        See `set_annotation_clip` for the meaning of return values.
        """
        return self._annotation_clip

    # 返回被注释点的像素位置。
    def _get_position_xy(self, renderer):
        """Return the pixel position of the annotated point."""
        return self._get_xy(renderer, self.xy, self.xycoords)

    # 检查是否应该绘制位于 *xy_pixel* 处的注释。
    def _check_xy(self, renderer=None):
        """Check whether the annotation at *xy_pixel* should be drawn."""
        if renderer is None:
            renderer = self.figure._get_renderer()
        b = self.get_annotation_clip()
        if b or (b is None and self.xycoords == "data"):
            # 检查 self.xy 是否位于 Axes 内部。
            xy_pixel = self._get_position_xy(renderer)
            return self.axes.contains_point(xy_pixel)
        return True

    # 设置注释是否可通过鼠标拖动。
    def draggable(self, state=None, use_blit=False):
        """
        Set whether the annotation is draggable with the mouse.

        Parameters
        ----------
        state : bool or None
            - True or False: 设置是否可拖动。
            - None: 切换拖动状态。
        use_blit : bool, default: False
            使用 blit 提高图像组合速度。详细信息参见 :ref:`func-animation`。

        Returns
        -------
        DraggableAnnotation or None
            如果注释可拖动，则返回相应的 `.DraggableAnnotation` 辅助对象。
        """
        from matplotlib.offsetbox import DraggableAnnotation
        is_draggable = self._draggable is not None

        # 如果 state 为 None，则切换拖动状态
        if state is None:
            state = not is_draggable

        if state:
            if self._draggable is None:
                self._draggable = DraggableAnnotation(self, use_blit)
        else:
            if self._draggable is not None:
                self._draggable.disconnect()
            self._draggable = None

        return self._draggable
# 继承自 Text 和 _AnnotationBase 的 Annotation 类，表示可以引用特定位置 xy 的文本对象。
class Annotation(Text, _AnnotationBase):
    """
    An `.Annotation` is a `.Text` that can refer to a specific position *xy*.
    Optionally an arrow pointing from the text to *xy* can be drawn.

    Attributes
    ----------
    xy
        The annotated position.
    xycoords
        The coordinate system for *xy*.
    arrow_patch
        A `.FancyArrowPatch` to point from *xytext* to *xy*.
    """

    # 返回 Annotation 对象的字符串表示形式，包括 xy 的坐标和文本 _text 的表示。
    def __str__(self):
        return f"Annotation({self.xy[0]:g}, {self.xy[1]:g}, {self._text!r})"

    # 初始化 Annotation 对象，用给定的 text 和 xy 注解点，可选地使用 xytext 和 arrowprops 参数进行定位和箭头绘制。
    def __init__(self, text, xy,
                 xytext=None,
                 xycoords='data',
                 textcoords=None,
                 arrowprops=None,
                 annotation_clip=None,
                 **kwargs):
        """
        Annotate the point *xy* with text *text*.

        In the simplest form, the text is placed at *xy*.

        Optionally, the text can be displayed in another position *xytext*.
        An arrow pointing from the text to the annotated point *xy* can then
        be added by defining *arrowprops*.

        Parameters
        ----------
        text : str
            The text of the annotation.

        xy : (float, float)
            The point *(x, y)* to annotate. The coordinate system is determined
            by *xycoords*.

        xytext : (float, float), default: *xy*
            The position *(x, y)* to place the text at. The coordinate system
            is determined by *textcoords*.

        xycoords : single or two-tuple of str or `.Artist` or `.Transform` or \
# 可调用对象，默认为 'data'
#   xy 参数所使用的坐标系。支持以下类型的值：

#   - 下列字符串之一：

#     ==================== ============================================
#     值                    描述
#     ==================== ============================================
#     'figure points'      图形左下角起点的点数
#     'figure pixels'      图形左下角起点的像素
#     'figure fraction'    相对于图形左下角的分数
#     'subfigure points'   子图左下角起点的点数
#     'subfigure pixels'   子图左下角起点的像素
#     'subfigure fraction' 相对于子图左下角的分数
#     'axes points'        坐标轴左下角起点的点数
#     'axes pixels'        坐标轴左下角起点的像素
#     'axes fraction'      相对于坐标轴左下角的分数
#     'data'               使用被标注对象的坐标系（默认）
#     'polar'              若非本地 'data' 坐标系，则为 *(theta, r)*
#     ==================== ============================================

#     注意，对于父图而言，'subfigure pixels' 和 'figure pixels' 是相同的，
#     因此希望在子图中可用的用户可以使用 'subfigure pixels'。

#   - 一个 `.Artist`：xy 被解释为艺术家的 `~matplotlib.transforms.Bbox` 的分数。
#     例如，*(0, 0)* 将是边界框的左下角，*(0.5, 1)* 将是边界框的中上角。

#   - 一个 `.Transform`，用于将 xy 转换为屏幕坐标。

#   - 一个具有以下签名之一的函数：

#       def transform(renderer) -> Bbox
#       def transform(renderer) -> Transform

#     其中 renderer 是 `.RendererBase` 的子类。
#     函数的结果与 `.Artist` 和 `.Transform` 情况类似。

#   - 一个元组 *(xcoords, ycoords)*，指定分别用于 x 和 y 的单独坐标系。
#     xcoords 和 ycoords 必须是上述描述类型之一。

#   详细信息请参阅 :ref:`plotting-guide-annotation`。

textcoords : single or two-tuple of str or `.Artist` or `.Transform` \
@api.rename_parameter("3.8", "event", "mouseevent")
    # 判断鼠标事件是否发生在当前对象的画布上
    def contains(self, mouseevent):
        # 如果鼠标事件发生在不同的画布上，则返回 False 和空字典
        if self._different_canvas(mouseevent):
            return False, {}
        # 调用 Text 类的 contains 方法，获取鼠标事件是否发生在文本对象内以及相关信息
        contains, tinfo = Text.contains(self, mouseevent)
        # 如果箭头补丁对象存在
        if self.arrow_patch is not None:
            # 判断鼠标事件是否发生在箭头补丁对象内
            in_patch, _ = self.arrow_patch.contains(mouseevent)
            contains = contains or in_patch
        # 返回包含信息和相关信息
        return contains, tinfo

    # 返回 xycoords 属性的值
    @property
    def xycoords(self):
        return self._xycoords

    # 设置 xycoords 属性的值
    @xycoords.setter
    def xycoords(self, xycoords):
        # 检查传入的 xycoords 是否为字符串元组，并且是否包含偏移量坐标
        def is_offset(s):
            return isinstance(s, str) and s.startswith("offset")

        # 如果 xycoords 是偏移量坐标或者包含偏移量坐标的元组，则引发 ValueError 异常
        if (isinstance(xycoords, tuple) and any(map(is_offset, xycoords))
                or is_offset(xycoords)):
            raise ValueError("xycoords cannot be an offset coordinate")
        # 设置对象的 _xycoords 属性为传入的 xycoords 值
        self._xycoords = xycoords

    # 返回 xyann 属性的值，即文本的位置，通过调用 get_position 方法获取
    @property
    def xyann(self):
        """
        The text position.

        See also *xytext* in `.Annotation`.
        """
        return self.get_position()

    # 设置 xyann 属性的值，即设置文本的位置，通过调用 set_position 方法设置
    @xyann.setter
    def xyann(self, xytext):
        self.set_position(xytext)

    # 返回使用于 `.Annotation.xyann` 的坐标系，通过调用 get_anncoords 方法获取
    def get_anncoords(self):
        """
        Return the coordinate system to use for `.Annotation.xyann`.

        See also *xycoords* in `.Annotation`.
        """
        return self._textcoords

    # 设置使用于 `.Annotation.xyann` 的坐标系，通过调用 set_anncoords 方法设置
    def set_anncoords(self, coords):
        """
        Set the coordinate system to use for `.Annotation.xyann`.

        See also *xycoords* in `.Annotation`.
        """
        self._textcoords = coords

    # 定义 anncoords 属性，用于获取和设置 `.Annotation.xyann` 的坐标系，具有相应的文档字符串
    anncoords = property(get_anncoords, set_anncoords, doc="""
        The coordinate system to use for `.Annotation.xyann`.""")

    # 设置对象所属的图形对象，继承自 Artist 类的方法
    def set_figure(self, fig):
        # 继承的方法文档字符串
        # 如果箭头补丁对象存在，则设置其所属的图形对象为传入的 fig
        if self.arrow_patch is not None:
            self.arrow_patch.set_figure(fig)
        # 调用父类 Artist 的 set_figure 方法，设置对象自身的图形对象为传入的 fig
        Artist.set_figure(self, fig)
    def update_positions(self, renderer):
        """
        Update the pixel positions of the annotation text and the arrow patch.
        """
        # 生成变换矩阵，用于定位注释文本和箭头的像素位置
        self.set_transform(self._get_xy_transform(renderer, self.anncoords))

        arrowprops = self.arrowprops
        if arrowprops is None:
            return

        bbox = Text.get_window_extent(self, renderer)

        # 获取注释点的位置坐标
        arrow_end = x1, y1 = self._get_position_xy(renderer)  # Annotated pos.

        ms = arrowprops.get("mutation_scale", self.get_size())
        self.arrow_patch.set_mutation_scale(ms)

        if "arrowstyle" not in arrowprops:
            # 模拟YAArrow风格的箭头
            shrink = arrowprops.get('shrink', 0.0)
            width = arrowprops.get('width', 4)
            headwidth = arrowprops.get('headwidth', 12)
            headlength = arrowprops.get('headlength', 12)

            # 注意：ms的单位为磅（points）
            stylekw = dict(head_length=headlength / ms,
                           head_width=headwidth / ms,
                           tail_width=width / ms)

            self.arrow_patch.set_arrowstyle('simple', **stylekw)

            # 使用YAArrow风格：
            # 选择文本框最靠近注释点的角落。
            xpos = [(bbox.x0, 0), ((bbox.x0 + bbox.x1) / 2, 0.5), (bbox.x1, 1)]
            ypos = [(bbox.y0, 0), ((bbox.y0 + bbox.y1) / 2, 0.5), (bbox.y1, 1)]
            x, relposx = min(xpos, key=lambda v: abs(v[0] - x1))
            y, relposy = min(ypos, key=lambda v: abs(v[0] - y1))
            self._arrow_relpos = (relposx, relposy)
            r = np.hypot(y - y1, x - x1)
            shrink_pts = shrink * r / renderer.points_to_pixels(1)
            self.arrow_patch.shrinkA = self.arrow_patch.shrinkB = shrink_pts

        # 调整箭头的起始点，相对于文本框的位置。
        # TODO : 需要考虑旋转的影响。
        arrow_begin = bbox.p0 + bbox.size * self._arrow_relpos
        # 箭头从arrow_begin到arrow_end绘制。首先通过patchA和patchB进行裁剪。
        # 然后通过shrinkA和shrinkB（以点为单位）进行缩小。
        self.arrow_patch.set_positions(arrow_begin, arrow_end)

        if "patchA" in arrowprops:
            patchA = arrowprops["patchA"]
        elif self._bbox_patch:
            patchA = self._bbox_patch
        elif self.get_text() == "":
            patchA = None
        else:
            pad = renderer.points_to_pixels(4)
            patchA = Rectangle(
                xy=(bbox.x0 - pad / 2, bbox.y0 - pad / 2),
                width=bbox.width + pad, height=bbox.height + pad,
                transform=IdentityTransform(), clip_on=False)
        self.arrow_patch.set_patchA(patchA)

    @artist.allow_rasterization
    def draw(self, renderer):
        # 如果渲染器不为None，则设置对象的渲染器为传入的渲染器
        if renderer is not None:
            self._renderer = renderer
        # 如果对象不可见或者位置不正确，则直接返回
        if not self.get_visible() or not self._check_xy(renderer):
            return
        # 在调用Text.draw之前更新文本位置，确保FancyArrowPatch定位正确
        self.update_positions(renderer)
        # 更新文本框的位置和大小
        self.update_bbox_position_size(renderer)
        # 如果存在箭头补丁（FancyArrowPatch）
        if self.arrow_patch is not None:
            # 如果箭头补丁的图形属性为空且对象的图形属性不为空，则将箭头补丁的图形属性设置为对象的图形属性
            if self.arrow_patch.figure is None and self.figure is not None:
                self.arrow_patch.figure = self.figure
            # 绘制箭头补丁
            self.arrow_patch.draw(renderer)
        # 绘制文本，包括FancyBboxPatch，在绘制FancyArrowPatch之后
        # 否则，楔形箭头样式可能会部分覆盖Bbox
        Text.draw(self, renderer)

    def get_window_extent(self, renderer=None):
        # 如果对象不可见或者位置不正确，则返回单位Bbox
        if not self.get_visible() or not self._check_xy(renderer):
            return Bbox.unit()
        # 如果传入了渲染器，则设置对象的渲染器为传入的渲染器
        if renderer is not None:
            self._renderer = renderer
        # 如果对象的渲染器为空，则从对象所属的图形对象获取渲染器
        if self._renderer is None:
            self._renderer = self.figure._get_renderer()
        # 如果仍然没有渲染器，则抛出运行时错误
        if self._renderer is None:
            raise RuntimeError('Cannot get window extent without renderer')

        # 更新文本位置
        self.update_positions(self._renderer)

        # 获取文本的Bbox
        text_bbox = Text.get_window_extent(self)
        bboxes = [text_bbox]

        # 如果存在箭头补丁，则加入箭头补丁的Bbox
        if self.arrow_patch is not None:
            bboxes.append(self.arrow_patch.get_window_extent())

        # 返回所有Bbox的并集
        return Bbox.union(bboxes)

    def get_tightbbox(self, renderer=None):
        # 如果位置不正确，则返回空Bbox
        if not self._check_xy(renderer):
            return Bbox.null()
        # 调用父类的get_tightbbox方法，并返回结果
        return super().get_tightbbox(renderer)
# 使用_docstring对象的interpd属性，更新Annotation类的构造函数__init__的文档字符串
_docstring.interpd.update(Annotation=Annotation.__init__.__doc__)
```