# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_cairo.py`

```
"""
A Cairo backend for Matplotlib
==============================
:Author: Steve Chaplin and others

This backend depends on cairocffi or pycairo.
"""

# 导入必要的模块
import functools  # 导入 functools 模块
import gzip  # 导入 gzip 模块
import math  # 导入 math 模块

import numpy as np  # 导入 NumPy 库

# 尝试导入 cairo 模块，并检查版本要求
try:
    import cairo
    if cairo.version_info < (1, 14, 0):  # 如果版本低于要求的最低版本 (1.14.0)，则引发 ImportError
        raise ImportError(f"Cairo backend requires cairo>=1.14.0, "
                          f"but only {cairo.version_info} is available")
except ImportError:
    try:
        import cairocffi as cairo  # 如果无法导入 cairo，则尝试导入 cairocffi
    except ImportError as err:
        raise ImportError(
            "cairo backend requires that pycairo>=1.14.0 or cairocffi "
            "is installed") from err  # 如果都无法导入，则抛出错误

# 导入 matplotlib 相关模块
from .. import _api, cbook, font_manager  # 导入上级包中的 _api, cbook, font_manager
from matplotlib.backend_bases import (  # 导入 matplotlib 中的基础后端类
    _Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase,
    RendererBase)
from matplotlib.font_manager import ttfFontProperty  # 导入 TrueType 字体属性
from matplotlib.path import Path  # 导入路径对象 Path
from matplotlib.transforms import Affine2D  # 导入仿射变换类 Affine2D


def _set_rgba(ctx, color, alpha, forced_alpha):
    """
    设置 Cairo 上下文的颜色和透明度。

    Parameters
    ----------
    ctx : cairo.Context
        Cairo 上下文对象。
    color : tuple
        RGB 颜色元组。
    alpha : float
        透明度。
    forced_alpha : bool
        是否强制使用给定的透明度。

    Returns
    -------
    None
    """
    if len(color) == 3 or forced_alpha:
        ctx.set_source_rgba(*color[:3], alpha)
    else:
        ctx.set_source_rgba(*color)


def _append_path(ctx, path, transform, clip=None):
    """
    将 matplotlib 路径对象添加到 Cairo 上下文中。

    Parameters
    ----------
    ctx : cairo.Context
        Cairo 上下文对象。
    path : matplotlib.path.Path
        要添加的路径对象。
    transform : matplotlib.transforms.Affine2D
        仿射变换对象，用于转换路径坐标。
    clip : bool, optional
        是否裁剪路径。

    Returns
    -------
    None
    """
    for points, code in path.iter_segments(
            transform, remove_nans=True, clip=clip):
        if code == Path.MOVETO:
            ctx.move_to(*points)
        elif code == Path.CLOSEPOLY:
            ctx.close_path()
        elif code == Path.LINETO:
            ctx.line_to(*points)
        elif code == Path.CURVE3:
            cur = np.asarray(ctx.get_current_point())
            a = points[:2]
            b = points[-2:]
            ctx.curve_to(*(cur / 3 + a * 2 / 3), *(a * 2 / 3 + b / 3), *b)
        elif code == Path.CURVE4:
            ctx.curve_to(*points)


def _cairo_font_args_from_font_prop(prop):
    """
    根据 matplotlib 的字体属性创建用于 Cairo 的字体参数。

    Parameters
    ----------
    prop : matplotlib.font_manager.FontProperties
        字体属性对象或字体条目对象。

    Returns
    -------
    tuple
        适用于 Cairo 的字体参数 (name, slant, weight)。
    """
    def attr(field):
        try:
            return getattr(prop, f"get_{field}")()
        except AttributeError:
            return getattr(prop, field)

    name = attr("name")
    slant = getattr(cairo, f"FONT_SLANT_{attr('style').upper()}")
    weight = attr("weight")
    weight = (cairo.FONT_WEIGHT_NORMAL
              if font_manager.weight_dict.get(weight, weight) < 550
              else cairo.FONT_WEIGHT_BOLD)
    return name, slant, weight


class RendererCairo(RendererBase):
    """
    Matplotlib 的 Cairo 渲染器类。

    Parameters
    ----------
    dpi : float
        渲染时的分辨率（每英寸点数）。

    Attributes
    ----------
    dpi : float
        渲染时的分辨率。
    gc : GraphicsContextCairo
        Cairo 图形上下文对象。
    width : int or None
        渲染区域的宽度。
    height : int or None
        渲染区域的高度。
    text_ctx : cairo.Context
        用于文本渲染的 Cairo 上下文对象。
    """

    def __init__(self, dpi):
        self.dpi = dpi
        self.gc = GraphicsContextCairo(renderer=self)
        self.width = None
        self.height = None
        self.text_ctx = cairo.Context(
           cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1))
        super().__init__()
    # 设置上下文环境，获取绘图目标表面
    surface = ctx.get_target()
    # 检查表面是否具有 get_width 和 get_height 方法，获取表面尺寸
    if hasattr(surface, "get_width") and hasattr(surface, "get_height"):
        size = surface.get_width(), surface.get_height()
    # 如果表面具有 get_extents 方法（适用于 GTK4 RecordingSurface）
    elif hasattr(surface, "get_extents"):
        # 获取表面的范围并获取其宽度和高度
        ext = surface.get_extents()
        size = ext.width, ext.height
    else:
        # 对于矢量表面的情况
        ctx.save()
        ctx.reset_clip()
        # 复制当前剪切矩形列表，并获取其尺寸信息
        rect, *rest = ctx.copy_clip_rectangle_list()
        if rest:
            # 如果存在多余的剪切矩形，抛出类型错误
            raise TypeError("Cannot infer surface size")
        # 解包矩形以获取尺寸信息
        _, _, *size = rect
        ctx.restore()
    # 将上下文环境和表面尺寸保存到实例变量中
    self.gc.ctx = ctx
    self.width, self.height = size

@staticmethod
def _fill_and_stroke(ctx, fill_c, alpha, alpha_overrides):
    # 如果填充颜色不为空
    if fill_c is not None:
        # 保存当前绘图状态
        ctx.save()
        # 设置颜色和透明度，填充路径并保留路径
        _set_rgba(ctx, fill_c, alpha, alpha_overrides)
        ctx.fill_preserve()
        # 恢复之前的绘图状态
        ctx.restore()
    # 描边路径
    ctx.stroke()

def draw_path(self, gc, path, transform, rgbFace=None):
    # 继承的文档字符串
    ctx = gc.ctx
    # 如果没有填充颜色并且没有填充图案，剪切路径至实际渲染范围
    clip = (ctx.clip_extents()
            if rgbFace is None and gc.get_hatch() is None
            else None)
    # 对变换进行处理，包括缩放和平移
    transform = (transform
                 + Affine2D().scale(1, -1).translate(0, self.height))
    # 开始新路径绘制
    ctx.new_path()
    # 添加路径到上下文中，应用变换和剪切
    _append_path(ctx, path, transform, clip)
    # 如果有填充颜色
    if rgbFace is not None:
        # 保存当前绘图状态
        ctx.save()
        # 设置填充颜色和透明度，填充路径并保留路径
        _set_rgba(ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())
        ctx.fill_preserve()
        # 恢复之前的绘图状态
        ctx.restore()
    # 获取填充图案路径
    hatch_path = gc.get_hatch_path()
    if hatch_path:
        # 获取 DPI 并创建相似表面作为填充图案的绘制表面
        dpi = int(self.dpi)
        hatch_surface = ctx.get_target().create_similar(
            cairo.Content.COLOR_ALPHA, dpi, dpi)
        hatch_ctx = cairo.Context(hatch_surface)
        # 添加填充图案路径到填充表面上下文中，应用变换和剪切
        _append_path(hatch_ctx, hatch_path,
                     Affine2D().scale(dpi, -dpi).translate(0, dpi),
                     None)
        # 设置填充线宽和颜色，并填充路径并描边
        hatch_ctx.set_line_width(self.points_to_pixels(gc.get_hatch_linewidth()))
        hatch_ctx.set_source_rgba(*gc.get_hatch_color())
        hatch_ctx.fill_preserve()
        hatch_ctx.stroke()
        # 创建填充图案的表面模式
        hatch_pattern = cairo.SurfacePattern(hatch_surface)
        hatch_pattern.set_extend(cairo.Extend.REPEAT)
        # 保存当前绘图状态
        ctx.save()
        # 设置填充图案的表面模式为当前上下文的源
        ctx.set_source(hatch_pattern)
        ctx.fill_preserve()
        # 恢复之前的绘图状态
        ctx.restore()
    # 描边路径
    ctx.stroke()
    # docstring inherited

    # 获取绘图上下文对象
    ctx = gc.ctx
    # 创建新路径
    ctx.new_path()
    # 根据 marker_path 和 marker_trans 创建标记路径，并进行 Y 轴翻转处理
    _append_path(ctx, marker_path, marker_trans + Affine2D().scale(1, -1))
    # 将标记路径的平面副本保存到 marker_path 中
    marker_path = ctx.copy_path_flat()

    # 判断路径是否有填充
    x1, y1, x2, y2 = ctx.fill_extents()
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        # 没有填充，将 filled 设为 False，并清除 rgbFace
        filled = False
        rgbFace = None
    else:
        # 有填充，将 filled 设为 True
        filled = True

    # 更新 transform，进行 Y 轴翻转和高度调整
    transform = (transform
                 + Affine2D().scale(1, -1).translate(0, self.height))

    # 创建新路径
    ctx.new_path()
    # 遍历 path 的所有线段和代码段
    for i, (vertices, codes) in enumerate(
            path.iter_segments(transform, simplify=False)):
        if len(vertices):
            # 获取最后两个顶点坐标
            x, y = vertices[-2:]
            ctx.save()

            # 平移并应用标记路径
            ctx.translate(x, y)
            ctx.append_path(marker_path)

            ctx.restore()

            # 如果有填充或者遍历次数是 1000 的倍数，调用 _fill_and_stroke 方法
            if filled or i % 1000 == 0:
                self._fill_and_stroke(
                    ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())

    # 如果没有填充，调用 _fill_and_stroke 方法
    if not filled:
        self._fill_and_stroke(
            ctx, rgbFace, gc.get_alpha(), gc.get_forced_alpha())
    # 绘制文本到指定位置 (x, y)，使用给定的绘图上下文 gc，文本内容为 s，字体属性为 prop，角度为 angle
    # 如果 ismath 为 True，使用数学文本绘制方法 _draw_mathtext 进行绘制
    # 否则，使用 Cairo 绘图上下文 ctx 直接绘制文本
    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # docstring inherited
        # 注意：(x, y) 是设备/显示坐标，不是用户坐标，与其他 draw_* 方法不同

        if ismath:
            # 如果是数学文本，调用 _draw_mathtext 方法进行绘制
            self._draw_mathtext(gc, x, y, s, prop, angle)

        else:
            # 否则，使用 Cairo 绘图上下文 ctx 绘制文本
            ctx = gc.ctx
            ctx.new_path()
            ctx.move_to(x, y)

            ctx.save()
            # 选择字体并设置字体大小
            ctx.select_font_face(*_cairo_font_args_from_font_prop(prop))
            ctx.set_font_size(self.points_to_pixels(prop.get_size_in_points()))
            opts = cairo.FontOptions()
            opts.set_antialias(gc.get_antialiased())
            ctx.set_font_options(opts)
            # 如果指定了角度，旋转文本
            if angle:
                ctx.rotate(np.deg2rad(-angle))
            # 显示文本
            ctx.show_text(s)
            ctx.restore()

    # 使用数学文本方式绘制文本到指定位置 (x, y)，使用给定的绘图上下文 gc，文本内容为 s，字体属性为 prop，角度为 angle
    def _draw_mathtext(self, gc, x, y, s, prop, angle):
        ctx = gc.ctx
        # 解析数学文本，获取字形、矩形等信息
        width, height, descent, glyphs, rects = \
            self._text2path.mathtext_parser.parse(s, self.dpi, prop)

        ctx.save()
        ctx.translate(x, y)
        # 如果指定了角度，旋转文本
        if angle:
            ctx.rotate(np.deg2rad(-angle))

        # 遍历所有字形，绘制每个字形
        for font, fontsize, idx, ox, oy in glyphs:
            ctx.new_path()
            ctx.move_to(ox, -oy)
            ctx.select_font_face(
                *_cairo_font_args_from_font_prop(ttfFontProperty(font)))
            ctx.set_font_size(self.points_to_pixels(fontsize))
            ctx.show_text(chr(idx))

        # 遍历所有矩形，绘制每个矩形
        for ox, oy, w, h in rects:
            ctx.new_path()
            ctx.rectangle(ox, -oy, w, -h)
            ctx.set_source_rgb(0, 0, 0)
            ctx.fill_preserve()

        ctx.restore()

    # 获取画布的宽度和高度
    def get_canvas_width_height(self):
        # docstring inherited
        return self.width, self.height

    # 获取文本的宽度、高度和下行高度，参数 s 为文本内容，prop 为字体属性，ismath 指示是否为数学文本
    def get_text_width_height_descent(self, s, prop, ismath):
        # docstring inherited

        if ismath == 'TeX':
            # 如果是 TeX 格式的数学文本，调用父类的方法获取其宽度、高度和下行高度
            return super().get_text_width_height_descent(s, prop, ismath)

        if ismath:
            # 如果是数学文本，解析数学文本，获取宽度、高度和下行高度
            width, height, descent, *_ = \
                self._text2path.mathtext_parser.parse(s, self.dpi, prop)
            return width, height, descent

        # 否则，使用 Cairo 的文本上下文绘制普通文本，避免因字体设置问题导致程序崩溃
        ctx = self.text_ctx
        ctx.save()
        ctx.select_font_face(*_cairo_font_args_from_font_prop(prop))
        ctx.set_font_size(self.points_to_pixels(prop.get_size_in_points()))

        # 获取文本的 y_bearing、宽度 w 和高度 h
        y_bearing, w, h = ctx.text_extents(s)[1:4]
        ctx.restore()

        return w, h, h + y_bearing
    # 创建一个新的图形上下文对象，并保存当前状态
    def new_gc(self):
        # 继承的文档字符串
        self.gc.ctx.save()
        # FIXME: 以下代码没有正确实现类似堆栈的行为，
        # 而是依赖于（不保证的）艺术家永远不会依赖于嵌套的gc状态，
        # 因此直接重置属性（即单层堆栈）足够。
        self.gc._alpha = 1
        self.gc._forced_alpha = False  # 如果为True，则_alpha会覆盖RGBA中的A值
        self.gc._hatch = None
        return self.gc

    # 将点数转换为像素数
    def points_to_pixels(self, points):
        # 继承的文档字符串
        return points / 72 * self.dpi
# Cairo 绘图上下文类，继承自 GraphicsContextBase 类
class GraphicsContextCairo(GraphicsContextBase):
    # 定义线段连接风格的映射关系
    _joind = {
        'bevel':  cairo.LINE_JOIN_BEVEL,
        'miter':  cairo.LINE_JOIN_MITER,
        'round':  cairo.LINE_JOIN_ROUND,
    }

    # 定义线段端点风格的映射关系
    _capd = {
        'butt':        cairo.LINE_CAP_BUTT,
        'projecting':  cairo.LINE_CAP_SQUARE,
        'round':       cairo.LINE_CAP_ROUND,
    }

    # 初始化方法，接受一个渲染器对象作为参数
    def __init__(self, renderer):
        super().__init__()  # 调用父类的初始化方法
        self.renderer = renderer  # 设置渲染器对象属性

    # 恢复绘图状态
    def restore(self):
        self.ctx.restore()  # 使用 Cairo 上下文对象恢复状态

    # 设置透明度
    def set_alpha(self, alpha):
        super().set_alpha(alpha)  # 调用父类的设置透明度方法
        _set_rgba(
            self.ctx, self._rgb, self.get_alpha(), self.get_forced_alpha())  # 设置 RGBA 值

    # 设置是否抗锯齿
    def set_antialiased(self, b):
        self.ctx.set_antialias(
            cairo.ANTIALIAS_DEFAULT if b else cairo.ANTIALIAS_NONE)  # 根据参数设置抗锯齿属性

    # 获取当前是否抗锯齿
    def get_antialiased(self):
        return self.ctx.get_antialias()  # 返回当前的抗锯齿设置

    # 设置线段端点风格
    def set_capstyle(self, cs):
        self.ctx.set_line_cap(_api.check_getitem(self._capd, capstyle=cs))  # 设置线段端点风格
        self._capstyle = cs  # 记录当前设置的端点风格

    # 设置裁剪矩形
    def set_clip_rectangle(self, rectangle):
        if not rectangle:
            return  # 如果矩形为空，则直接返回
        x, y, w, h = np.round(rectangle.bounds)  # 获取矩形的边界值并四舍五入
        ctx = self.ctx
        ctx.new_path()  # 创建新路径
        ctx.rectangle(x, self.renderer.height - h - y, w, h)  # 绘制矩形
        ctx.clip()  # 执行裁剪操作

    # 设置裁剪路径
    def set_clip_path(self, path):
        if not path:
            return  # 如果路径为空，则直接返回
        tpath, affine = path.get_transformed_path_and_affine()  # 获取变换后的路径和仿射变换矩阵
        ctx = self.ctx
        ctx.new_path()  # 创建新路径
        affine = (affine
                  + Affine2D().scale(1, -1).translate(0, self.renderer.height))  # 仿射变换矩阵处理
        _append_path(ctx, tpath, affine)  # 添加路径到 Cairo 上下文
        ctx.clip()  # 执行裁剪操作

    # 设置虚线样式
    def set_dashes(self, offset, dashes):
        self._dashes = offset, dashes  # 记录当前的虚线样式
        if dashes is None:
            self.ctx.set_dash([], 0)  # 关闭虚线效果
        else:
            self.ctx.set_dash(
                list(self.renderer.points_to_pixels(np.asarray(dashes))),  # 设置虚线样式，将点转换为像素
                offset)

    # 设置前景色
    def set_foreground(self, fg, isRGBA=None):
        super().set_foreground(fg, isRGBA)  # 调用父类的设置前景色方法
        if len(self._rgb) == 3:
            self.ctx.set_source_rgb(*self._rgb)  # 设置 RGB 颜色
        else:
            self.ctx.set_source_rgba(*self._rgb)  # 设置 RGBA 颜色

    # 获取当前前景色的 RGB 值
    def get_rgb(self):
        return self.ctx.get_source().get_rgba()[:3]  # 获取当前 Cairo 上下文的 RGB 值

    # 设置线段连接风格
    def set_joinstyle(self, js):
        self.ctx.set_line_join(_api.check_getitem(self._joind, joinstyle=js))  # 设置线段连接风格
        self._joinstyle = js  # 记录当前设置的连接风格

    # 设置线段宽度
    def set_linewidth(self, w):
        self._linewidth = float(w)  # 记录当前线段宽度
        self.ctx.set_line_width(self.renderer.points_to_pixels(w))  # 设置线段宽度，将点转换为像素


# Cairo 区域类，用于存储切片和数据
class _CairoRegion:
    def __init__(self, slices, data):
        self._slices = slices  # 初始化切片数据
        self._data = data  # 初始化数据
    def _renderer(self):
        # 理论上，_renderer 应在 __init__ 中设置，但 GUI 画布的子类（例如 FigureCanvasFooCairo）
        # 与多重继承不兼容（FigureCanvasFoo 初始化但不进行超类初始化 FigureCanvasCairo），
        # 因此在 getter 方法中进行初始化。
        
        # 检查是否存在 _cached_renderer 属性，如果不存在则创建 RendererCairo 实例并缓存
        if not hasattr(self, "_cached_renderer"):
            self._cached_renderer = RendererCairo(self.figure.dpi)
        # 返回缓存的渲染器实例
        return self._cached_renderer

    def get_renderer(self):
        # 返回当前对象的渲染器，调用 _renderer 方法获取
        return self._renderer

    def copy_from_bbox(self, bbox):
        # 获取渲染器关联的 Cairo 图像表面
        surface = self._renderer.gc.ctx.get_target()
        # 如果图像表面不是 cairo.ImageSurface 类型，则抛出运行时错误
        if not isinstance(surface, cairo.ImageSurface):
            raise RuntimeError(
                "copy_from_bbox 只在渲染到 ImageSurface 时有效")
        # 获取图像表面的宽度和高度
        sw = surface.get_width()
        sh = surface.get_height()
        # 计算裁剪框的四个边界
        x0 = math.ceil(bbox.x0)
        x1 = math.floor(bbox.x1)
        y0 = math.ceil(sh - bbox.y1)
        y1 = math.floor(sh - bbox.y0)
        # 检查裁剪框和图像表面的有效性
        if not (0 <= x0 and x1 <= sw and bbox.x0 <= bbox.x1
                and 0 <= y0 and y1 <= sh and bbox.y0 <= bbox.y1):
            raise ValueError("Invalid bbox")
        # 根据裁剪框在图像表面上进行裁剪，返回裁剪后的数据
        sls = slice(y0, y0 + max(y1 - y0, 0)), slice(x0, x0 + max(x1 - x0, 0))
        data = (np.frombuffer(surface.get_data(), np.uint32)
                .reshape((sh, sw))[sls].copy())
        return _CairoRegion(sls, data)

    def restore_region(self, region):
        # 获取渲染器关联的 Cairo 图像表面
        surface = self._renderer.gc.ctx.get_target()
        # 如果图像表面不是 cairo.ImageSurface 类型，则抛出运行时错误
        if not isinstance(surface, cairo.ImageSurface):
            raise RuntimeError(
                "restore_region 只在渲染到 ImageSurface 时有效")
        # 刷新图像表面确保最新数据
        surface.flush()
        # 获取图像表面的宽度和高度
        sw = surface.get_width()
        sh = surface.get_height()
        # 从区域对象中获取裁剪位置
        sly, slx = region._slices
        # 将区域数据恢复到图像表面对应位置
        (np.frombuffer(surface.get_data(), np.uint32)
         .reshape((sh, sw))[sly, slx]) = region._data
        # 标记恢复区域的矩形为脏区域，以便后续更新
        surface.mark_dirty_rectangle(
            slx.start, sly.start, slx.stop - slx.start, sly.stop - sly.start)

    def print_png(self, fobj):
        # 将渲染后的图像表面输出为 PNG 格式到文件对象 fobj
        self._get_printed_image_surface().write_to_png(fobj)

    def print_rgba(self, fobj):
        # 获取图像的宽度和高度
        width, height = self.get_width_height()
        # 获取渲染后的图像表面数据
        buf = self._get_printed_image_surface().get_data()
        # 将数据转换为 RGBA8888 格式并写入文件对象 fobj
        fobj.write(cbook._premultiplied_argb32_to_unmultiplied_rgba8888(
            np.asarray(buf).reshape((width, height, 4))))

    print_raw = print_rgba

    def _get_printed_image_surface(self):
        # 设置渲染器 DPI 为图形对象的 DPI
        self._renderer.dpi = self.figure.dpi
        # 获取图形对象的宽度和高度
        width, height = self.get_width_height()
        # 创建指定大小的 Cairo 图像表面
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        # 将图像表面与渲染器的上下文关联
        self._renderer.set_context(cairo.Context(surface))
        # 绘制图形对象到渲染器
        self.figure.draw(self._renderer)
        # 返回渲染后的图像表面
        return surface
    # 定义一个私有方法 `_save`，用于保存图形为 PDF/PS/SVG 格式的文件
    def _save(self, fmt, fobj, *, orientation='portrait'):
        # 设置图形的 DPI 为 72
        dpi = 72
        self.figure.dpi = dpi
        # 获取图形的尺寸，单位为英寸
        w_in, h_in = self.figure.get_size_inches()
        # 计算图形宽度和高度的像素值
        width_in_points, height_in_points = w_in * dpi, h_in * dpi

        # 根据方向调整宽度和高度的像素值，如果是横向则交换宽度和高度
        if orientation == 'landscape':
            width_in_points, height_in_points = (
                height_in_points, width_in_points)

        # 根据格式选择不同的 Cairo Surface 类型，并创建对应的表面对象
        if fmt == 'ps':
            # 检查 cairo 是否支持 PS 格式，若不支持则抛出运行时错误
            if not hasattr(cairo, 'PSSurface'):
                raise RuntimeError('cairo has not been compiled with PS '
                                   'support enabled')
            surface = cairo.PSSurface(fobj, width_in_points, height_in_points)
        elif fmt == 'pdf':
            # 检查 cairo 是否支持 PDF 格式，若不支持则抛出运行时错误
            if not hasattr(cairo, 'PDFSurface'):
                raise RuntimeError('cairo has not been compiled with PDF '
                                   'support enabled')
            surface = cairo.PDFSurface(fobj, width_in_points, height_in_points)
        elif fmt in ('svg', 'svgz'):
            # 检查 cairo 是否支持 SVG 格式，若不支持则抛出运行时错误
            if not hasattr(cairo, 'SVGSurface'):
                raise RuntimeError('cairo has not been compiled with SVG '
                                   'support enabled')
            # 如果是 svgz 格式，则根据 fobj 的类型选择是否压缩文件
            if fmt == 'svgz':
                if isinstance(fobj, str):
                    fobj = gzip.GzipFile(fobj, 'wb')
                else:
                    fobj = gzip.GzipFile(None, 'wb', fileobj=fobj)
            surface = cairo.SVGSurface(fobj, width_in_points, height_in_points)
        else:
            # 若格式未知，则抛出值错误异常
            raise ValueError(f"Unknown format: {fmt!r}")

        # 设置渲染器的 DPI，并将其与表面对象关联的 Cairo 上下文设置为当前上下文
        self._renderer.dpi = self.figure.dpi
        self._renderer.set_context(cairo.Context(surface))
        ctx = self._renderer.gc.ctx

        # 如果是横向打印，则旋转上下文并调整坐标原点
        if orientation == 'landscape':
            ctx.rotate(np.pi / 2)
            ctx.translate(0, -height_in_points)
            # 可能添加一个 '%%Orientation: Landscape' 的注释？

        # 绘制图形到渲染器
        self.figure.draw(self._renderer)

        # 结束页面并完成表面绘制
        ctx.show_page()
        surface.finish()
        # 如果是 svgz 格式，则关闭文件对象
        if fmt == 'svgz':
            fobj.close()

    # 定义一系列部分应用方法，用于分别打印为 PDF/PS/SVG/SVGZ 格式
    print_pdf = functools.partialmethod(_save, "pdf")
    print_ps = functools.partialmethod(_save, "ps")
    print_svg = functools.partialmethod(_save, "svg")
    print_svgz = functools.partialmethod(_save, "svgz")
# 将 _BackendCairo 类导出为模块的一部分
@_Backend.export
# 定义 _BackendCairo 类，继承自 _Backend 类
class _BackendCairo(_Backend):
    # 设置后端版本为 Cairo 库的版本号
    backend_version = cairo.version
    # 将 FigureCanvas 类指定为 FigureCanvasCairo
    FigureCanvas = FigureCanvasCairo
    # 将 FigureManager 类指定为 FigureManagerBase
    FigureManager = FigureManagerBase
```