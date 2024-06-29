# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_agg.py`

```py
"""
An `Anti-Grain Geometry`_ (AGG) backend.

Features that are implemented:

* capstyles and join styles
* dashes
* linewidth
* lines, rectangles, ellipses
* clipping to a rectangle
* output to RGBA and Pillow-supported image formats
* alpha blending
* DPI scaling properly - everything scales properly (dashes, linewidths, etc)
* draw polygon
* freetype2 w/ ft2font

Still TODO:

* integrate screen dpi w/ ppi and text

.. _Anti-Grain Geometry: http://agg.sourceforge.net/antigrain.com
"""

from contextlib import nullcontext  # 导入nullcontext，用于创建一个无操作的上下文管理器
from math import radians, cos, sin  # 导入数学函数：弧度转换、余弦和正弦

import numpy as np  # 导入NumPy库

import matplotlib as mpl  # 导入Matplotlib库
from matplotlib import _api, cbook  # 导入Matplotlib的内部API和cbook模块
from matplotlib.backend_bases import (  # 从Matplotlib的backend_bases模块导入以下类
    _Backend, FigureCanvasBase, FigureManagerBase, RendererBase)
from matplotlib.font_manager import fontManager as _fontManager, get_font  # 导入字体管理器和获取字体函数
from matplotlib.ft2font import (  # 从Matplotlib的ft2font模块导入以下常量
    LOAD_FORCE_AUTOHINT, LOAD_NO_HINTING, LOAD_DEFAULT, LOAD_NO_AUTOHINT)
from matplotlib.mathtext import MathTextParser  # 导入Matplotlib的数学文本解析器
from matplotlib.path import Path  # 导入Matplotlib的路径对象
from matplotlib.transforms import Bbox, BboxBase  # 导入Matplotlib的边界框和基础边界框
from matplotlib.backends._backend_agg import RendererAgg as _RendererAgg  # 导入AGG渲染器

def get_hinting_flag():
    """
    根据当前Matplotlib配置返回文本渲染时的hinting标志

    返回：
        hinting标志值
    """
    mapping = {
        'default': LOAD_DEFAULT,
        'no_autohint': LOAD_NO_AUTOHINT,
        'force_autohint': LOAD_FORCE_AUTOHINT,
        'no_hinting': LOAD_NO_HINTING,
        True: LOAD_FORCE_AUTOHINT,
        False: LOAD_NO_HINTING,
        'either': LOAD_DEFAULT,
        'native': LOAD_NO_AUTOHINT,
        'auto': LOAD_FORCE_AUTOHINT,
        'none': LOAD_NO_HINTING,
    }
    return mapping[mpl.rcParams['text.hinting']]

class RendererAgg(RendererBase):
    """
    AGG渲染器类，处理所有绘图原语的渲染，使用图形上下文实例来控制颜色/样式
    """

    def __init__(self, width, height, dpi):
        """
        初始化AGG渲染器对象

        参数：
            width (float)：渲染区域宽度
            height (float)：渲染区域高度
            dpi (float)：渲染分辨率
        """
        super().__init__()

        self.dpi = dpi  # 设置渲染分辨率
        self.width = width  # 设置渲染区域宽度
        self.height = height  # 设置渲染区域高度
        self._renderer = _RendererAgg(int(width), int(height), dpi)  # 创建AGG渲染器实例

        self._filter_renderers = []  # 初始化过滤渲染器列表

        self._update_methods()  # 更新渲染方法
        self.mathtext_parser = MathTextParser('agg')  # 创建用于数学文本解析的解析器实例

        self.bbox = Bbox.from_bounds(0, 0, self.width, self.height)  # 根据渲染区域设置边界框

    def __getstate__(self):
        """
        获取渲染器对象的状态

        返回：
            dict：渲染器对象的关键状态信息
        """
        # 仅保留渲染器初始化参数作为状态信息，其它可以重新创建
        return {'width': self.width, 'height': self.height, 'dpi': self.dpi}

    def __setstate__(self, state):
        """
        设置渲染器对象的状态

        参数：
            state (dict)：包含渲染器初始化参数的状态信息
        """
        self.__init__(state['width'], state['height'], state['dpi'])  # 根据状态信息重新初始化渲染器对象

    def _update_methods(self):
        """
        更新渲染器对象的方法
        """
        # 将渲染器对象的绘制方法更新为内部渲染器的对应方法
        self.draw_gouraud_triangles = self._renderer.draw_gouraud_triangles
        self.draw_image = self._renderer.draw_image
        self.draw_markers = self._renderer.draw_markers
        self.draw_path_collection = self._renderer.draw_path_collection
        self.draw_quad_mesh = self._renderer.draw_quad_mesh
        self.copy_from_bbox = self._renderer.copy_from_bbox
    def draw_mathtext(self, gc, x, y, s, prop, angle):
        """Draw mathtext using :mod:`matplotlib.mathtext`."""
        # 解析 mathtext 字符串 s，获取绘制所需的位置、大小等信息
        ox, oy, width, height, descent, font_image = \
            self.mathtext_parser.parse(s, self.dpi, prop,
                                       antialiased=gc.get_antialiased())

        # 计算偏移量，考虑文本角度对应的水平和垂直偏移
        xd = descent * sin(radians(angle))
        yd = descent * cos(radians(angle))

        # 计算最终的绘制位置坐标
        x = round(x + ox + xd)
        y = round(y - oy + yd)

        # 调用渲染器的方法绘制文本图像
        self._renderer.draw_text_image(font_image, x, y + 1, angle, gc)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # docstring inherited
        # 如果是数学公式，调用 draw_mathtext 方法绘制
        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)

        # 否则，准备字体属性并绘制普通文本
        font = self._prepare_font(prop)

        # 在这里传递 '0' 作为角度，因为文本将在后续的 draw_text_image 调用中旋转
        font.set_text(s, 0, flags=get_hinting_flag())

        # 将字形绘制到位图中，考虑抗锯齿设置
        font.draw_glyphs_to_bitmap(
            antialiased=gc.get_antialiased())

        # 获取字体的下降值并进行调整，以考虑角度影响
        d = font.get_descent() / 64.0
        xo, yo = font.get_bitmap_offset()
        xo /= 64.0
        yo /= 64.0
        xd = d * sin(radians(angle))
        yd = d * cos(radians(angle))

        # 计算最终的绘制位置坐标
        x = round(x + xo + xd)
        y = round(y + yo + yd)

        # 调用渲染器的方法绘制文本图像
        self._renderer.draw_text_image(font, x, y + 1, angle, gc)

    def get_text_width_height_descent(self, s, prop, ismath):
        # docstring inherited

        # 检查 ismath 参数是否有效
        _api.check_in_list(["TeX", True, False], ismath=ismath)

        # 如果是 TeX 数学模式，则调用父类方法获取宽度、高度和下降值
        if ismath == "TeX":
            return super().get_text_width_height_descent(s, prop, ismath)

        # 如果是数学模式，解析 mathtext 字符串 s，获取宽度、高度和下降值
        if ismath:
            ox, oy, width, height, descent, font_image = \
                self.mathtext_parser.parse(s, self.dpi, prop)
            return width, height, descent

        # 否则，准备字体属性并设置文本，获取宽度、高度和下降值
        font = self._prepare_font(prop)
        font.set_text(s, 0.0, flags=get_hinting_flag())
        w, h = font.get_width_height()  # 获取未旋转字符串的宽度和高度
        d = font.get_descent()
        w /= 64.0  # 转换为子像素
        h /= 64.0
        d /= 64.0
        return w, h, d

    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        # docstring inherited
        # todo, handle props, angle, origins

        # 获取字体大小（以点为单位）
        size = prop.get_size_in_points()

        # 获取 TeX 渲染器
        texmanager = self.get_texmanager()

        # 获取 TeX 字符串的灰度图像，并将其转换为 numpy 数组
        Z = texmanager.get_grey(s, size, self.dpi)
        Z = np.array(Z * 255.0, np.uint8)

        # 获取文本的宽度、高度和下降值
        w, h, d = self.get_text_width_height_descent(s, prop, ismath="TeX")

        # 考虑角度对下降值的影响，计算最终的绘制位置坐标
        xd = d * sin(radians(angle))
        yd = d * cos(radians(angle))
        x = round(x + xd)
        y = round(y + yd)

        # 调用渲染器的方法绘制文本图像
        self._renderer.draw_text_image(Z, x, y, angle, gc)

    def get_canvas_width_height(self):
        # docstring inherited
        # 返回画布的宽度和高度
        return self.width, self.height
    def _prepare_font(self, font_prop):
        """
        Get the `.FT2Font` for *font_prop*, clear its buffer, and set its size.
        """
        # 根据字体属性获取 `.FT2Font` 对象，并清空其缓冲区，设置字体大小
        font = get_font(_fontManager._find_fonts_by_props(font_prop))
        font.clear()
        size = font_prop.get_size_in_points()
        font.set_size(size, self.dpi)
        return font

    def points_to_pixels(self, points):
        """
        Convert points to pixels using the current DPI setting.
        """
        # 继承的文档字符串，将点数转换为像素数，使用当前的 DPI 设置
        return points * self.dpi / 72

    def buffer_rgba(self):
        """
        Return a memory view of the renderer's RGBA buffer.
        """
        # 返回渲染器的 RGBA 缓冲区的内存视图
        return memoryview(self._renderer)

    def tostring_argb(self):
        """
        Convert the renderer's RGBA buffer to ARGB format and return as bytes.
        """
        # 将渲染器的 RGBA 缓冲区转换为 ARGB 格式，并返回字节串
        return np.asarray(self._renderer).take([3, 0, 1, 2], axis=2).tobytes()

    @_api.deprecated("3.8", alternative="buffer_rgba")
    def tostring_rgb(self):
        """
        (Deprecated) Convert the renderer's RGBA buffer to RGB format and return as bytes.
        Use `buffer_rgba` instead.
        """
        # (已弃用) 将渲染器的 RGBA 缓冲区转换为 RGB 格式，并返回字节串
        # 推荐使用 `buffer_rgba` 替代
        return np.asarray(self._renderer).take([0, 1, 2], axis=2).tobytes()

    def clear(self):
        """
        Clear the renderer's content.
        """
        # 清空渲染器的内容
        self._renderer.clear()

    def option_image_nocomposite(self):
        """
        Return True to indicate no compositing is needed for image rendering.
        """
        # 继承的文档字符串，返回 True 表示图像渲染不需要合成
        # 在 Agg 后端下，直接将每个图像直接合成到 Figure 通常更快，且不会减少文件大小
        return True

    def option_scale_image(self):
        """
        Return False to indicate images should not be scaled automatically.
        """
        # 继承的文档字符串，返回 False 表示图像不应自动缩放
        return False

    def restore_region(self, region, bbox=None, xy=None):
        """
        Restore the saved region of the renderer.

        If bbox (instance of BboxBase, or its extents) is given, only the region specified
        by the bbox will be restored. *xy* (a pair of floats) optionally specifies the new
        position where the region will be restored.
        """
        # 恢复渲染器中保存的区域

        # 如果提供了 bbox 或 xy 参数，则根据参数指定的区域来恢复
        # bbox 是 BboxBase 的实例或其边界的表示方式
        # xy 是一个浮点数对，指定新的恢复位置，其LLC是原始区域的LLC，而不是bbox的LLC
        if bbox is not None or xy is not None:
            if bbox is None:
                x1, y1, x2, y2 = region.get_extents()
            elif isinstance(bbox, BboxBase):
                x1, y1, x2, y2 = bbox.extents
            else:
                x1, y1, x2, y2 = bbox

            if xy is None:
                ox, oy = x1, y1
            else:
                ox, oy = xy

            # 传入的数据是浮点数，但 _renderer 的类型检查需要整数
            self._renderer.restore_region(region, int(x1), int(y1),
                                          int(x2), int(y2), int(ox), int(oy))
        else:
            self._renderer.restore_region(region)

    def start_filter(self):
        """
        Start a new filtering process by creating a new canvas and saving the current one.
        """
        # 启动新的过滤器过程，简单地创建一个新画布并保存当前的画布
        self._filter_renderers.append(self._renderer)
        self._renderer = _RendererAgg(int(self.width), int(self.height),
                                      self.dpi)
        self._update_methods()
    def stop_filter(self, post_processing):
        """
        Save the current canvas as an image and apply post processing.

        The *post_processing* function::

           def post_processing(image, dpi):
             # ny, nx, depth = image.shape
             # image (numpy array) has RGBA channels and has a depth of 4.
             ...
             # create a new_image (numpy array of 4 channels, size can be
             # different). The resulting image may have offsets from
             # lower-left corner of the original image
             return new_image, offset_x, offset_y

        The saved renderer is restored and the returned image from
        post_processing is plotted (using draw_image) on it.
        """
        # 将当前画布转换为 RGBA 格式的 numpy 数组
        orig_img = np.asarray(self.buffer_rgba())
        # 获取不含全透明像素的裁剪图像的切片
        slice_y, slice_x = cbook._get_nonzero_slices(orig_img[..., 3])
        cropped_img = orig_img[slice_y, slice_x]

        # 弹出最近保存的渲染器，恢复原始渲染器，并更新相关方法
        self._renderer = self._filter_renderers.pop()
        self._update_methods()

        # 如果裁剪后的图像不为空
        if cropped_img.size:
            # 调用外部提供的后处理函数 post_processing
            img, ox, oy = post_processing(cropped_img / 255, self.dpi)
            gc = self.new_gc()
            # 如果处理后的图像数据类型为浮点型，将其转换为无符号整型（0-255）
            if img.dtype.kind == 'f':
                img = np.asarray(img * 255., np.uint8)
            # 在画布上绘制处理后的图像
            self._renderer.draw_image(
                gc, slice_x.start + ox, int(self.height) - slice_y.stop + oy,
                img[::-1])
class FigureCanvasAgg(FigureCanvasBase):
    # FigureCanvasBase 类的文档字符串已继承

    _lastKey = None  # 每个实例在第一次绘制时会被重新赋值。

    def copy_from_bbox(self, bbox):
        # 获取渲染器对象并调用其方法复制指定边界框区域的内容
        renderer = self.get_renderer()
        return renderer.copy_from_bbox(bbox)

    def restore_region(self, region, bbox=None, xy=None):
        # 获取渲染器对象并调用其方法恢复指定区域的内容
        renderer = self.get_renderer()
        return renderer.restore_region(region, bbox, xy)

    def draw(self):
        # 继承的文档字符串
        self.renderer = self.get_renderer()
        self.renderer.clear()
        # 获取工具栏对象，如果存在则在绘制期间使用等待光标
        with (self.toolbar._wait_cursor_for_draw_cm() if self.toolbar
              else nullcontext()):
            # 绘制图形对象到渲染器
            self.figure.draw(self.renderer)
            # 如果存在 GUI 类，则可能需要使用此绘制方法更新窗口，因此别忘了调用超类方法。
            super().draw()

    def get_renderer(self):
        # 获取图形对象的尺寸和 DPI，根据是否重用上次的渲染器对象来创建新的或重用旧的渲染器对象
        w, h = self.figure.bbox.size
        key = w, h, self.figure.dpi
        reuse_renderer = (self._lastKey == key)
        if not reuse_renderer:
            self.renderer = RendererAgg(w, h, self.figure.dpi)
            self._lastKey = key
        return self.renderer

    @_api.deprecated("3.8", alternative="buffer_rgba")
    def tostring_rgb(self):
        """
        返回 RGB 格式的图像数据 `bytes`。

        必须至少调用一次 `draw` 方法才能使用此函数，并且对图形对象进行任何后续更改时也要更新渲染器。
        """
        return self.renderer.tostring_rgb()

    def tostring_argb(self):
        """
        返回 ARGB 格式的图像数据 `bytes`。

        必须至少调用一次 `draw` 方法才能使用此函数，并且对图形对象进行任何后续更改时也要更新渲染器。
        """
        return self.renderer.tostring_argb()

    def buffer_rgba(self):
        """
        返回渲染器缓冲区的 `memoryview` 对象，表示图像数据。

        必须至少调用一次 `draw` 方法才能使用此函数，并且对图形对象进行任何后续更改时也要更新渲染器。
        """
        return self.renderer.buffer_rgba()

    def print_raw(self, filename_or_obj, *, metadata=None):
        # 如果传入的元数据不为 None，则引发值错误异常，因为原始或 RGBA 输出不支持元数据
        if metadata is not None:
            raise ValueError("metadata not supported for raw/rgba")
        # 调用 FigureCanvasAgg.draw(self) 方法来绘制图形
        FigureCanvasAgg.draw(self)
        # 获取渲染器对象，并将其 RGBA 缓冲区内容写入到指定的文件名或对象中
        renderer = self.get_renderer()
        with cbook.open_file_cm(filename_or_obj, "wb") as fh:
            fh.write(renderer.buffer_rgba())

    print_rgba = print_raw

    def _print_pil(self, filename_or_obj, fmt, pil_kwargs, metadata=None):
        """
        绘制画布，然后使用 `.image.imsave` 方法保存图像（将 *pil_kwargs* 和 *metadata* 转发给它）。
        """
        # 绘制图形对象
        FigureCanvasAgg.draw(self)
        # 使用 `mpl.image.imsave` 方法保存图像，以指定的格式和参数
        mpl.image.imsave(
            filename_or_obj, self.buffer_rgba(), format=fmt, origin="upper",
            dpi=self.figure.dpi, metadata=metadata, pil_kwargs=pil_kwargs)
    def print_png(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        """
        将图形保存为 PNG 文件。

        Parameters
        ----------
        filename_or_obj : str or path-like or file-like
            要写入的文件路径或文件对象。

        metadata : dict, optional
            PNG 文件中的元数据，以字节或 latin-1 可编码字符串的键值对形式提供。
            根据 PNG 规范，键名必须少于79个字符。

            `PNG 规范`_ 定义了一些常见的关键字，可以根据需要使用：

            - Title: 图像的简短标题或说明。
            - Author: 图像创建者的名称。
            - Description: 图像的描述（可能很长）。
            - Copyright: 版权声明。
            - Creation Time: 原始图像创建时间（通常为 RFC 1123 格式）。
            - Software: 创建图像的软件名称。
            - Disclaimer: 法律声明。
            - Warning: 内容性质的警告。
            - Source: 创建图像的设备。
            - Comment: 其他格式转换来的杂注。

            可以为其他目的发明其他关键字。

            如果未提供 'Software'，Matplotlib 将自动生成一个值。可以通过将其设置为 *None* 来移除。

            更多细节请参阅 `PNG 规范`_。

            .. _PNG 规范: \
                https://www.w3.org/TR/2003/REC-PNG-20031110/#11keywords

        pil_kwargs : dict, optional
            传递给 `PIL.Image.Image.save` 的关键字参数。

            如果包含 'pnginfo' 键，它会完全覆盖 *metadata*，包括默认的 'Software' 键。
        """
        self._print_pil(filename_or_obj, "png", pil_kwargs, metadata)

    def print_to_buffer(self):
        """
        绘制图形并将其渲染为 RGBA 缓冲区。

        Returns
        -------
        bytes
            RGBA 缓冲区的字节表示。
        tuple
            包含渲染器宽度和高度的元组。
        """
        FigureCanvasAgg.draw(self)
        renderer = self.get_renderer()
        return (bytes(renderer.buffer_rgba()),
                (int(renderer.width), int(renderer.height)))

    # 注意，通常应通过 savefig() 和 print_figure() 调用这些方法，
    # 后者确保 `self.figure.dpi` 已经与 dpi 关键字匹配（如果有的话）。

    def print_jpg(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        """
        将图形保存为 JPEG 文件。

        Parameters
        ----------
        filename_or_obj : str or path-like or file-like
            要写入的文件路径或文件对象。

        metadata : dict, optional
            JPEG 文件中的元数据，以字节或 latin-1 可编码字符串的键值对形式提供。

        pil_kwargs : dict, optional
            传递给 `PIL.Image.Image.save` 的关键字参数。

        Notes
        -----
        savefig() 已经应用了 savefig.facecolor；现在将其设置为白色，以便使半透明图形
        在假定的白色背景上混合。
        """
        with mpl.rc_context({"savefig.facecolor": "white"}):
            self._print_pil(filename_or_obj, "jpeg", pil_kwargs, metadata)

    print_jpeg = print_jpg

    def print_tif(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        """
        将图形保存为 TIFF 文件。

        Parameters
        ----------
        filename_or_obj : str or path-like or file-like
            要写入的文件路径或文件对象。

        metadata : dict, optional
            TIFF 文件中的元数据，以字节或 latin-1 可编码字符串的键值对形式提供。

        pil_kwargs : dict, optional
            传递给 `PIL.Image.Image.save` 的关键字参数。
        """
        self._print_pil(filename_or_obj, "tiff", pil_kwargs, metadata)

    print_tiff = print_tif
    # 定义一个打印 WebP 格式文件的方法，调用内部通用的图像打印方法 _print_pil
    def print_webp(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        # 调用 _print_pil 方法，传入文件名或对象、"webp" 字符串表示图像格式、pil_kwargs 和 metadata 参数
        self._print_pil(filename_or_obj, "webp", pil_kwargs, metadata)

    # 将 print_jpg、print_tif、print_webp 方法的文档字符串设置为统一格式的模板字符串
    print_jpg.__doc__, print_tif.__doc__, print_webp.__doc__ = map(
        """
        Write the figure to a {} file.

        Parameters
        ----------
        filename_or_obj : str or path-like or file-like
            The file to write to.
        pil_kwargs : dict, optional
            Additional keyword arguments that are passed to
            `PIL.Image.Image.save` when saving the figure.
        """.format, ["JPEG", "TIFF", "WebP"])
# 将 _BackendAgg 类标记为导出，使其可被外部调用
@_Backend.export
# 定义 _BackendAgg 类，继承自 _Backend 类
class _BackendAgg(_Backend):
    # 设置后端版本号为 'v2.2'
    backend_version = 'v2.2'
    # 将 FigureCanvas 属性指向 FigureCanvasAgg 类
    FigureCanvas = FigureCanvasAgg
    # 将 FigureManager 属性指向 FigureManagerBase 类
    FigureManager = FigureManagerBase
```