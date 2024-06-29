# `D:\src\scipysrc\matplotlib\lib\matplotlib\backend_bases.py`

```
"""
Abstract base classes define the primitives that renderers and
graphics contexts must implement to serve as a Matplotlib backend.

`RendererBase`
    An abstract base class to handle drawing/rendering operations.

`FigureCanvasBase`
    The abstraction layer that separates the `.Figure` from the backend
    specific details like a user interface drawing area.

`GraphicsContextBase`
    An abstract base class that provides color, line styles, etc.

`Event`
    The base class for all of the Matplotlib event handling.  Derived classes
    such as `KeyEvent` and `MouseEvent` store the meta data like keys and
    buttons pressed, x and y locations in pixel and `~.axes.Axes` coordinates.

`ShowBase`
    The base class for the ``Show`` class of each interactive backend; the
    'show' callable is then set to ``Show.__call__``.

`ToolContainerBase`
    The base class for the Toolbar class of each interactive backend.
"""

from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import pathlib
import signal
import socket
import sys
import time
import weakref
from weakref import WeakKeyDictionary

import numpy as np

# Importing necessary components from Matplotlib and its submodules
import matplotlib as mpl
from matplotlib import (
    _api, backend_tools as tools, cbook, colors, _docstring, text,
    _tight_bbox, transforms, widgets, is_interactive, rcParams)
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle

# Setting up logging for the current module
_log = logging.getLogger(__name__)

# Default file types with corresponding descriptions
_default_filetypes = {
    'eps': 'Encapsulated Postscript',
    'jpg': 'Joint Photographic Experts Group',
    'jpeg': 'Joint Photographic Experts Group',
    'pdf': 'Portable Document Format',
    'pgf': 'PGF code for LaTeX',
    'png': 'Portable Network Graphics',
    'ps': 'Postscript',
    'raw': 'Raw RGBA bitmap',
    'rgba': 'Raw RGBA bitmap',
    'svg': 'Scalable Vector Graphics',
    'svgz': 'Scalable Vector Graphics',
    'tif': 'Tagged Image File Format',
    'tiff': 'Tagged Image File Format',
    'webp': 'WebP Image Format',
}

# Default backends for each file type
_default_backends = {
    'eps': 'matplotlib.backends.backend_ps',
    'jpg': 'matplotlib.backends.backend_agg',
    'jpeg': 'matplotlib.backends.backend_agg',
    'pdf': 'matplotlib.backends.backend_pdf',
    'pgf': 'matplotlib.backends.backend_pgf',
    'png': 'matplotlib.backends.backend_agg',
    'ps': 'matplotlib.backends.backend_ps',
    'raw': 'matplotlib.backends.backend_agg',
    'rgba': 'matplotlib.backends.backend_agg',
    'svg': 'matplotlib.backends.backend_svg',
    'svgz': 'matplotlib.backends.backend_svg',
}
    'tif': 'matplotlib.backends.backend_agg',  # 将 'tif' 文件扩展名映射到 matplotlib 的后端模块 backend_agg
    'tiff': 'matplotlib.backends.backend_agg',  # 将 'tiff' 文件扩展名映射到 matplotlib 的后端模块 backend_agg
    'webp': 'matplotlib.backends.backend_agg',  # 将 'webp' 文件扩展名映射到 matplotlib 的后端模块 backend_agg
    # 注册一个后端，用于保存特定文件格式
    """
    Register a backend for saving to a given file format.

    Parameters
    ----------
    format : str
        File extension
    backend : module string or canvas class
        Backend for handling file output
    description : str, default: ""
        Description of the file type.
    """
    if description is None:
        description = ''
    # 将后端与文件格式关联起来
    _default_backends[format] = backend
    # 设置文件格式的描述信息
    _default_filetypes[format] = description


def get_registered_canvas_class(format):
    """
    Return the registered default canvas for given file format.
    Handles deferred import of required backend.
    """
    if format not in _default_backends:
        return None
    # 获取注册的后端类
    backend_class = _default_backends[format]
    # 如果后端类是字符串，则导入相应模块获取其FigureCanvas类
    if isinstance(backend_class, str):
        backend_class = importlib.import_module(backend_class).FigureCanvas
        # 更新默认后端为实际的后端类
        _default_backends[format] = backend_class
    # 返回获取到的后端类
    return backend_class


class RendererBase:
    """
    An abstract base class to handle drawing/rendering operations.

    The following methods must be implemented in the backend for full
    functionality (though just implementing `draw_path` alone would give a
    highly capable backend):

    * `draw_path`
    * `draw_image`
    * `draw_gouraud_triangles`

    The following methods *should* be implemented in the backend for
    optimization reasons:

    * `draw_text`
    * `draw_markers`
    * `draw_path_collection`
    * `draw_quad_mesh`
    """
    def __init__(self):
        super().__init__()
        # 纹理管理器对象
        self._texmanager = None
        # 文本到路径转换器对象
        self._text2path = text.TextToPath()
        # 光栅深度
        self._raster_depth = 0
        # 是否正在进行光栅化
        self._rasterizing = False

    def open_group(self, s, gid=None):
        """
        Open a grouping element with label *s* and *gid* (if set) as id.

        Only used by the SVG renderer.
        """
        # 打开一个带有标签*s*的分组元素，如果设置了*gid*，则作为其id
        pass

    def close_group(self, s):
        """
        Close a grouping element with label *s*.

        Only used by the SVG renderer.
        """
        # 关闭标签为*s*的分组元素
        pass

    def draw_path(self, gc, path, transform, rgbFace=None):
        """Draw a `~.path.Path` instance using the given affine transform."""
        # 使用给定的仿射变换绘制`~.path.Path`实例
        raise NotImplementedError
    def draw_markers(self, gc, marker_path, marker_trans, path,
                     trans, rgbFace=None):
        """
        Draw a marker at each of *path*'s vertices (excluding control points).

        The base (fallback) implementation makes multiple calls to `draw_path`.
        Backends may want to override this method in order to draw the marker
        only once and reuse it multiple times.

        Parameters
        ----------
        gc : `.GraphicsContextBase`
            The graphics context.
        marker_path : `~matplotlib.path.Path`
            The path for the marker.
        marker_trans : `~matplotlib.transforms.Transform`
            An affine transform applied to the marker.
        path : `~matplotlib.path.Path`
            The locations to draw the markers.
        trans : `~matplotlib.transforms.Transform`
            An affine transform applied to the path.
        rgbFace : :mpltype:`color`, optional
            The color to fill the marker with.

        """
        # 遍历路径的所有线段和代码
        for vertices, codes in path.iter_segments(trans, simplify=False):
            # 如果有顶点存在
            if len(vertices):
                # 取最后两个顶点作为绘制点的位置
                x, y = vertices[-2:]
                # 在指定位置绘制路径，使用给定的变换和颜色
                self.draw_path(gc, marker_path,
                               marker_trans +
                               transforms.Affine2D().translate(x, y),
                               rgbFace)
    def draw_path_collection(self, gc, master_transform, paths, all_transforms,
                             offsets, offset_trans, facecolors, edgecolors,
                             linewidths, linestyles, antialiaseds, urls,
                             offset_position):
        """
        Draw a collection of *paths*.

        Each path is first transformed by the corresponding entry
        in *all_transforms* (a list of (3, 3) matrices) and then by
        *master_transform*.  They are then translated by the corresponding
        entry in *offsets*, which has been first transformed by *offset_trans*.

        *facecolors*, *edgecolors*, *linewidths*, *linestyles*, and
        *antialiased* are lists that set the corresponding properties.

        *offset_position* is unused now, but the argument is kept for
        backwards compatibility.

        The base (fallback) implementation makes multiple calls to `draw_path`.
        Backends may want to override this in order to render each set of
        path data only once, and then reference that path multiple times with
        the different offsets, colors, styles etc.  The generator methods
        `_iter_collection_raw_paths` and `_iter_collection` are provided to
        help with (and standardize) the implementation across backends.  It
        is highly recommended to use those generators, so that changes to the
        behavior of `draw_path_collection` can be made globally.
        """
        # 通过 `_iter_collection_raw_paths` 方法获取路径的标识符列表
        path_ids = self._iter_collection_raw_paths(master_transform,
                                                   paths, all_transforms)

        # 遍历路径标识符列表和其他参数来绘制集合中的每条路径
        for xo, yo, path_id, gc0, rgbFace in self._iter_collection(
                gc, list(path_ids), offsets, offset_trans,
                facecolors, edgecolors, linewidths, linestyles,
                antialiaseds, urls, offset_position):
            # 解包路径和变换信息
            path, transform = path_id

            # 如果有偏移量，则对变换进行额外的平移操作
            if xo != 0 or yo != 0:
                # 因为 translate 是就地操作，所以在应用平移之前需要使用 .frozen() 方法复制变换
                transform = transform.frozen()
                transform.translate(xo, yo)

            # 调用 draw_path 方法来绘制路径
            self.draw_path(gc0, path, transform, rgbFace)
    def draw_quad_mesh(self, gc, master_transform, meshWidth, meshHeight,
                       coordinates, offsets, offsetTrans, facecolors,
                       antialiased, edgecolors):
        """
        Draw a quadmesh.

        The base (fallback) implementation converts the quadmesh to paths and
        then calls `draw_path_collection`.
        """

        # 导入QuadMesh类
        from matplotlib.collections import QuadMesh
        # 将四边形网格转换为路径对象
        paths = QuadMesh._convert_mesh_to_paths(coordinates)

        # 如果未指定边缘颜色，则使用面颜色作为边缘颜色
        if edgecolors is None:
            edgecolors = facecolors
        # 创建包含当前线宽的NumPy数组
        linewidths = np.array([gc.get_linewidth()], float)

        # 调用draw_path_collection方法绘制路径集合，并返回结果
        return self.draw_path_collection(
            gc, master_transform, paths, [], offsets, offsetTrans, facecolors,
            edgecolors, linewidths, [], [antialiased], [None], 'screen')

    def draw_gouraud_triangles(self, gc, triangles_array, colors_array,
                               transform):
        """
        Draw a series of Gouraud triangles.

        Parameters
        ----------
        gc : `.GraphicsContextBase`
            The graphics context.
        triangles_array : (N, 3, 2) array-like
            Array of *N* (x, y) points for the triangles.
        colors_array : (N, 3, 4) array-like
            Array of *N* RGBA colors for each point of the triangles.
        transform : `~matplotlib.transforms.Transform`
            An affine transform to apply to the points.
        """
        # 抛出未实现错误，表明该方法需要在子类中实现
        raise NotImplementedError

    def _iter_collection_raw_paths(self, master_transform, paths,
                                   all_transforms):
        """
        Helper method (along with `_iter_collection`) to implement
        `draw_path_collection` in a memory-efficient manner.

        This method yields all of the base path/transform combinations, given a
        master transform, a list of paths and list of transforms.

        The arguments should be exactly what is passed in to
        `draw_path_collection`.

        The backend should take each yielded path and transform and create an
        object that can be referenced (reused) later.
        """
        # 计算路径数和变换数
        Npaths = len(paths)
        Ntransforms = len(all_transforms)
        N = max(Npaths, Ntransforms)

        # 如果路径数为0，直接返回
        if Npaths == 0:
            return

        # 初始化默认变换为单位变换
        transform = transforms.IdentityTransform()
        # 循环遍历路径和变换组合，并使用生成器进行逐个返回
        for i in range(N):
            path = paths[i % Npaths]
            if Ntransforms:
                transform = Affine2D(all_transforms[i % Ntransforms])
            yield path, transform + master_transform
    def _iter_collection_uses_per_path(self, paths, all_transforms,
                                       offsets, facecolors, edgecolors):
        """
        计算调用 `_iter_collection` 时，每个由 `_iter_collection_raw_paths` 返回的原始路径对象将被使用的次数。
        这用于后端决定在使用路径内联和存储路径之间的权衡。如果使用次数不同，向上取整。
        """
        Npaths = len(paths)
        if Npaths == 0 or len(facecolors) == len(edgecolors) == 0:
            return 0
        Npath_ids = max(Npaths, len(all_transforms))
        N = max(Npath_ids, len(offsets))
        return (N + Npath_ids - 1) // Npath_ids

    def get_image_magnification(self):
        """
        获取传递给 `draw_image` 的图像放大因子。
        允许后端具有与其他艺术家不同分辨率的图像。
        """
        return 1.0

    def draw_image(self, gc, x, y, im, transform=None):
        """
        绘制一个RGBA图像。

        Parameters
        ----------
        gc : `.GraphicsContextBase`
            带有剪切信息的图形上下文。

        x : scalar
            距离画布左侧的物理单位（即点或像素）的距离。

        y : scalar
            距离画布底部的物理单位（即点或像素）的距离。

        im : (N, M, 4) array of `numpy.uint8`
            RGBA像素数组。

        transform : `~matplotlib.transforms.Affine2DBase`
            只有当具体的后端被编写为 `option_scale_image` 返回 ``True`` 时，
            可以传递仿射变换（即 `.Affine2DBase` ）到 `draw_image`。
            变换的平移向量以物理单位（即点或像素）给出。
            注意，变换不会覆盖 *x* 和 *y*，必须在将结果通过 *x* 和 *y* 平移之前应用变换
            （可以通过将 *x* 和 *y* 添加到由 *transform* 定义的平移向量来实现）。
        """
        raise NotImplementedError

    def option_image_nocomposite(self):
        """
        返回是否应跳过Matplotlib的图像合成。

        光栅后端通常应返回False（让C级别的光栅化器负责图像合成）；
        矢量后端通常应返回 ``not rcParams["image.composite_image"]``。
        """
        return False

    def option_scale_image(self):
        """
        返回 `draw_image` 中是否支持任意仿射变换（大多数矢量后端为True）。
        """
        return False
    # 绘制一个 TeX 实例。
    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        """
        Draw a TeX instance.

        Parameters
        ----------
        gc : `.GraphicsContextBase`
            绘图上下文。
        x : float
            文本在显示坐标中的 x 位置。
        y : float
            文本基线在显示坐标中的 y 位置。
        s : str
            TeX 文本字符串。
        prop : `~matplotlib.font_manager.FontProperties`
            字体属性。
        angle : float
            逆时针旋转角度（度数）。
        mtext : `~matplotlib.text.Text`
            待渲染的原始文本对象。
        """
        self._draw_text_as_path(gc, x, y, s, prop, angle, ismath="TeX")

    # 绘制一个文本实例。
    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        """
        Draw a text instance.

        Parameters
        ----------
        gc : `.GraphicsContextBase`
            绘图上下文。
        x : float
            文本在显示坐标中的 x 位置。
        y : float
            文本基线在显示坐标中的 y 位置。
        s : str
            文本字符串。
        prop : `~matplotlib.font_manager.FontProperties`
            字体属性。
        angle : float
            逆时针旋转角度（度数）。
        ismath : bool or "TeX"
            如果为 True，使用数学文本解析器。
        mtext : `~matplotlib.text.Text`
            待渲染的原始文本对象。

        Notes
        -----
        **Notes for backend implementers:**

        `.RendererBase.draw_text` 也支持将 "TeX" 传递给 *ismath* 参数以使用 TeX 渲染，
        但实际的渲染后端不要求支持这一特性，事实上，许多内置的后端并不支持。
        反而，TeX 渲染由 `~.RendererBase.draw_tex` 提供。
        """
        self._draw_text_as_path(gc, x, y, s, prop, angle, ismath)
    def _draw_text_as_path(self, gc, x, y, s, prop, angle, ismath):
        """
        Draw the text by converting them to paths using `.TextToPath`.

        This private helper supports the same parameters as
        `~.RendererBase.draw_text`; setting *ismath* to "TeX" triggers TeX
        rendering.
        """
        # 获取文本转换为路径的工具对象
        text2path = self._text2path
        # 将字体大小从点转换为像素
        fontsize = self.points_to_pixels(prop.get_size_in_points())
        # 使用文本转换为路径工具对象将文本转换为路径的顶点和代码
        verts, codes = text2path.get_text_path(prop, s, ismath=ismath)
        # 创建路径对象
        path = Path(verts, codes)
        
        # 根据是否需要垂直翻转设置变换矩阵
        if self.flipy():
            width, height = self.get_canvas_width_height()
            transform = (Affine2D()
                         .scale(fontsize / text2path.FONT_SCALE)
                         .rotate_deg(angle)
                         .translate(x, height - y))
        else:
            transform = (Affine2D()
                         .scale(fontsize / text2path.FONT_SCALE)
                         .rotate_deg(angle)
                         .translate(x, y))
        
        # 获取绘制颜色并设置线宽为0
        color = gc.get_rgb()
        gc.set_linewidth(0.0)
        # 使用给定的变换和颜色绘制路径
        self.draw_path(gc, path, transform, rgbFace=color)

    def get_text_width_height_descent(self, s, prop, ismath):
        """
        Get the width, height, and descent (offset from the bottom to the baseline), in
        display coords, of the string *s* with `.FontProperties` *prop*.

        Whitespace at the start and the end of *s* is included in the reported width.
        """
        # 获取字体大小（以点为单位）
        fontsize = prop.get_size_in_points()

        if ismath == 'TeX':
            # 处理 TeX 渲染的情况，返回文本的宽度、高度和下降量
            return self.get_texmanager().get_text_width_height_descent(
                s, fontsize, renderer=self)

        # 将点转换为像素
        dpi = self.points_to_pixels(72)
        if ismath:
            # 使用数学文本解析器解析数学文本并返回宽度、高度和下降量
            dims = self._text2path.mathtext_parser.parse(s, dpi, prop)
            return dims[0:3]  # 返回宽度、高度和下降量

        # 获取文本转换为路径的工具对象的提示标志
        flags = self._text2path._get_hinting_flag()
        # 获取字体对象并设置字体大小和 DPI
        font = self._text2path._get_font(prop)
        font.set_size(fontsize, dpi)
        # 设置文本并获取其宽度和高度
        font.set_text(s, 0.0, flags=flags)
        w, h = font.get_width_height()
        # 获取文本的下降量
        d = font.get_descent()
        # 将宽度、高度和下降量从子像素转换为像素
        w /= 64.0
        h /= 64.0
        d /= 64.0
        return w, h, d

    def flipy(self):
        """
        Return whether y values increase from top to bottom.

        Note that this only affects drawing of texts.
        """
        # 返回是否需要垂直翻转坐标系以绘制文本
        return True

    def get_canvas_width_height(self):
        """Return the canvas width and height in display coords."""
        # 返回画布的宽度和高度，以显示坐标为单位
        return 1, 1

    def get_texmanager(self):
        """Return the `.TexManager` instance."""
        # 返回 TeX 管理器的实例
        if self._texmanager is None:
            self._texmanager = TexManager()
        return self._texmanager

    def new_gc(self):
        """Return an instance of a `.GraphicsContextBase`."""
        # 返回一个 `.GraphicsContextBase` 实例
        return GraphicsContextBase()
    def points_to_pixels(self, points):
        """
        Convert points to display units.

        You need to override this function (unless your backend
        doesn't have a dpi, e.g., postscript or svg).  Some imaging
        systems assume some value for pixels per inch::

            points to pixels = points * pixels_per_inch/72 * dpi/72

        Parameters
        ----------
        points : float or array-like

        Returns
        -------
        Points converted to pixels
        """
        return points



    def start_rasterizing(self):
        """
        Switch to the raster renderer.

        Used by `.MixedModeRenderer`.
        """



    def stop_rasterizing(self):
        """
        Switch back to the vector renderer and draw the contents of the raster
        renderer as an image on the vector renderer.

        Used by `.MixedModeRenderer`.
        """



    def start_filter(self):
        """
        Switch to a temporary renderer for image filtering effects.

        Currently only supported by the agg renderer.
        """



    def stop_filter(self, filter_func):
        """
        Switch back to the original renderer.  The contents of the temporary
        renderer is processed with the *filter_func* and is drawn on the
        original renderer as an image.

        Currently only supported by the agg renderer.
        """



    def _draw_disabled(self):
        """
        Context manager to temporary disable drawing.

        This is used for getting the drawn size of Artists.  This lets us
        run the draw process to update any Python state but does not pay the
        cost of the draw_XYZ calls on the canvas.
        """
        # Create a dictionary of no-op methods for temporarily disabling drawing operations
        no_ops = {
            meth_name: lambda *args, **kwargs: None
            for meth_name in dir(RendererBase)  # Iterate over all methods of RendererBase
            if (meth_name.startswith("draw_")   # Include methods starting with 'draw_'
                or meth_name in ["open_group", "close_group"])  # Also include 'open_group' and 'close_group'
        }

        # Return a context manager that sets all relevant methods of self to no-op functions
        return _setattr_cm(self, **no_ops)
# 抽象基类，提供图形上下文的颜色、线条样式等属性
class GraphicsContextBase:
    """An abstract base class that provides color, line styles, etc."""

    def __init__(self):
        # 设置不透明度，默认为完全不透明
        self._alpha = 1.0
        # 是否强制使用 _alpha 覆盖 RGBA 中的 A 值
        self._forced_alpha = False  # if True, _alpha overrides A from RGBA
        # 抗锯齿设置，使用 0 和 1 代表扩展代码中的 True 和 False
        self._antialiased = 1  # use 0, 1 not True, False for extension code
        # 设置线段端点样式，默认为 'butt'
        self._capstyle = CapStyle('butt')
        # 设置裁剪矩形区域，默认为 None
        self._cliprect = None
        # 设置裁剪路径，默认为 None
        self._clippath = None
        # 设置虚线样式，默认为 (0, None)
        self._dashes = 0, None
        # 设置连接线段样式，默认为 'round'
        self._joinstyle = JoinStyle('round')
        # 设置线条样式，默认为 'solid'
        self._linestyle = 'solid'
        # 设置线条宽度，默认为 1
        self._linewidth = 1
        # 设置颜色，默认为黑色不透明
        self._rgb = (0.0, 0.0, 0.0, 1.0)
        # 设置填充图案，默认为 None
        self._hatch = None
        # 设置填充图案颜色，默认为 rcParams['hatch.color'] 的 RGBA 值
        self._hatch_color = colors.to_rgba(rcParams['hatch.color'])
        # 设置填充图案线宽，默认为 rcParams['hatch.linewidth']
        self._hatch_linewidth = rcParams['hatch.linewidth']
        # 设置 URL，用于关联对象的 URL，默认为 None
        self._url = None
        # 设置图形 ID，默认为 None
        self._gid = None
        # 设置吸附行为，默认为 None
        self._snap = None
        # 设置草图样式，默认为 None
        self._sketch = None

    def copy_properties(self, gc):
        """Copy properties from *gc* to self."""
        # 复制属性值从给定的 GraphicsContextBase 实例 *gc* 到当前实例 self
        self._alpha = gc._alpha
        self._forced_alpha = gc._forced_alpha
        self._antialiased = gc._antialiased
        self._capstyle = gc._capstyle
        self._cliprect = gc._cliprect
        self._clippath = gc._clippath
        self._dashes = gc._dashes
        self._joinstyle = gc._joinstyle
        self._linestyle = gc._linestyle
        self._linewidth = gc._linewidth
        self._rgb = gc._rgb
        self._hatch = gc._hatch
        self._hatch_color = gc._hatch_color
        self._hatch_linewidth = gc._hatch_linewidth
        self._url = gc._url
        self._gid = gc._gid
        self._snap = gc._snap
        self._sketch = gc._sketch

    def restore(self):
        """
        Restore the graphics context from the stack - needed only
        for backends that save graphics contexts on a stack.
        """
        # 从堆栈中恢复图形上下文，仅适用于保存图形上下文的后端

    def get_alpha(self):
        """
        Return the alpha value used for blending - not supported on all
        backends.
        """
        # 返回用于混合的 alpha 值，不是所有后端都支持该功能
        return self._alpha

    def get_antialiased(self):
        """Return whether the object should try to do antialiased rendering."""
        # 返回对象是否尝试进行抗锯齿渲染
        return self._antialiased

    def get_capstyle(self):
        """Return the `.CapStyle`."""
        # 返回线段端点样式的名称
        return self._capstyle.name

    def get_clip_rectangle(self):
        """
        Return the clip rectangle as a `~matplotlib.transforms.Bbox` instance.
        """
        # 返回裁剪矩形作为 `~matplotlib.transforms.Bbox` 实例
        return self._cliprect

    def get_clip_path(self):
        """
        Return the clip path in the form (path, transform), where path
        is a `~.path.Path` instance, and transform is
        an affine transform to apply to the path before clipping.
        """
        # 返回裁剪路径，返回形式为 (path, transform)，其中 path 是 `~.path.Path` 实例，transform 是应用于路径的仿射变换
        if self._clippath is not None:
            tpath, tr = self._clippath.get_transformed_path_and_affine()
            if np.all(np.isfinite(tpath.vertices)):
                return tpath, tr
            else:
                _log.warning("Ill-defined clip_path detected. Returning None.")
                return None, None
        return None, None
    def get_dashes(self):
        """
        Return the dash style as an (offset, dash-list) pair.

        See `.set_dashes` for details.

        Default value is (None, None).
        """
        # 返回当前对象的_dash属性，表示虚线样式的偏移量和虚线列表
        return self._dashes

    def get_forced_alpha(self):
        """
        Return whether the value given by get_alpha() should be used to
        override any other alpha-channel values.
        """
        # 返回当前对象的_forced_alpha属性，表示是否应使用get_alpha()的值覆盖任何其他alpha通道值
        return self._forced_alpha

    def get_joinstyle(self):
        """Return the `.JoinStyle`."""
        # 返回当前对象的_joinstyle属性的名称
        return self._joinstyle.name

    def get_linewidth(self):
        """Return the line width in points."""
        # 返回当前对象的_linewidth属性，表示线条宽度（单位为点）
        return self._linewidth

    def get_rgb(self):
        """Return a tuple of three or four floats from 0-1."""
        # 返回当前对象的_rgb属性，表示颜色的RGB或RGBA值，每个值范围在0到1之间
        return self._rgb

    def get_url(self):
        """Return a url if one is set, None otherwise."""
        # 返回当前对象的_url属性，表示如果设置了URL，则返回URL，否则返回None
        return self._url

    def get_gid(self):
        """Return the object identifier if one is set, None otherwise."""
        # 返回当前对象的_gid属性，表示如果设置了对象标识符，则返回标识符，否则返回None
        return self._gid

    def get_snap(self):
        """
        Return the snap setting, which can be:

        * True: snap vertices to the nearest pixel center
        * False: leave vertices as-is
        * None: (auto) If the path contains only rectilinear line segments,
          round to the nearest pixel center
        """
        # 返回当前对象的_snap属性，表示顶点捕捉设置，可以是True（捕捉到最近的像素中心）、False（保持顶点不变）、None（自动，如果路径仅包含直线段，则四舍五入到最近的像素中心）
        return self._snap

    def set_alpha(self, alpha):
        """
        Set the alpha value used for blending - not supported on all backends.

        If ``alpha=None`` (the default), the alpha components of the
        foreground and fill colors will be used to set their respective
        transparencies (where applicable); otherwise, ``alpha`` will override
        them.
        """
        # 设置用于混合的alpha值，如果alpha为None（默认），则前景色和填充色的alpha分量将用于设置它们各自的透明度；否则，alpha将覆盖它们
        if alpha is not None:
            self._alpha = alpha
            self._forced_alpha = True
        else:
            self._alpha = 1.0
            self._forced_alpha = False
        self.set_foreground(self._rgb, isRGBA=True)

    def set_antialiased(self, b):
        """Set whether object should be drawn with antialiased rendering."""
        # 设置对象是否使用抗锯齿渲染绘制，参数b为布尔值，转换为整数后存储在_antialiased属性中
        self._antialiased = int(bool(b))

    @_docstring.interpd
    def set_capstyle(self, cs):
        """
        Set how to draw endpoints of lines.

        Parameters
        ----------
        cs : `.CapStyle` or %(CapStyle)s
        """
        # 设置如何绘制线段的端点样式，参数cs可以是`.CapStyle`类型或字符串，存储在_capstyle属性中
        self._capstyle = CapStyle(cs)

    def set_clip_rectangle(self, rectangle):
        """Set the clip rectangle to a `.Bbox` or None."""
        # 设置剪辑矩形区域，参数rectangle为`.Bbox`对象或None，存储在_cliprect属性中
        self._cliprect = rectangle

    def set_clip_path(self, path):
        """Set the clip path to a `.TransformedPath` or None."""
        # 设置剪辑路径，参数path为`.TransformedPath`对象或None，存储在_clippath属性中
        _api.check_isinstance((transforms.TransformedPath, None), path=path)
        self._clippath = path
    def set_dashes(self, dash_offset, dash_list):
        """
        Set the dash style for the gc.

        Parameters
        ----------
        dash_offset : float
            Distance, in points, into the dash pattern at which to
            start the pattern. It is usually set to 0.
        dash_list : array-like or None
            The on-off sequence as points.  None specifies a solid line. All
            values must otherwise be non-negative (:math:`\\ge 0`).

        Notes
        -----
        See p. 666 of the PostScript
        `Language Reference
        <https://www.adobe.com/jp/print/postscript/pdfs/PLRM.pdf>`_
        for more info.
        """
        # 如果 dash_list 不为 None，则将其转换为 NumPy 数组
        if dash_list is not None:
            dl = np.asarray(dash_list)
            # 如果 dash_list 中有任何负值，抛出 ValueError 异常
            if np.any(dl < 0.0):
                raise ValueError(
                    "All values in the dash list must be non-negative")
            # 如果 dash_list 非空且没有任何正值，抛出 ValueError 异常
            if dl.size and not np.any(dl > 0.0):
                raise ValueError(
                    'At least one value in the dash list must be positive')
        # 将 dash_offset 和 dash_list 存储到对象属性 _dashes 中
        self._dashes = dash_offset, dash_list

    def set_foreground(self, fg, isRGBA=False):
        """
        Set the foreground color.

        Parameters
        ----------
        fg : :mpltype:`color`
        isRGBA : bool
            If *fg* is known to be an ``(r, g, b, a)`` tuple, *isRGBA* can be
            set to True to improve performance.
        """
        # 如果 _forced_alpha 为 True 并且 isRGBA 为 True，则直接设置 _rgb
        if self._forced_alpha and isRGBA:
            self._rgb = fg[:3] + (self._alpha,)
        # 如果 _forced_alpha 为 True，则通过 colors.to_rgba 转换 fg，并设置 _rgb
        elif self._forced_alpha:
            self._rgb = colors.to_rgba(fg, self._alpha)
        # 如果 isRGBA 为 True，则直接设置 _rgb
        elif isRGBA:
            self._rgb = fg
        # 否则，通过 colors.to_rgba 转换 fg，并设置 _rgb
        else:
            self._rgb = colors.to_rgba(fg)

    @_docstring.interpd
    def set_joinstyle(self, js):
        """
        Set how to draw connections between line segments.

        Parameters
        ----------
        js : `.JoinStyle` or %(JoinStyle)s
        """
        # 将连接样式对象 JoinStyle(js) 存储到对象属性 _joinstyle 中
        self._joinstyle = JoinStyle(js)

    def set_linewidth(self, w):
        """Set the linewidth in points."""
        # 将线宽度值 w 存储到对象属性 _linewidth 中
        self._linewidth = float(w)

    def set_url(self, url):
        """Set the url for links in compatible backends."""
        # 将 URL 字符串 url 存储到对象属性 _url 中
        self._url = url

    def set_gid(self, id):
        """Set the id."""
        # 将标识符 id 存储到对象属性 _gid 中
        self._gid = id

    def set_snap(self, snap):
        """
        Set the snap setting which may be:

        * True: snap vertices to the nearest pixel center
        * False: leave vertices as-is
        * None: (auto) If the path contains only rectilinear line segments,
          round to the nearest pixel center
        """
        # 将 snap 设置存储到对象属性 _snap 中
        self._snap = snap

    def set_hatch(self, hatch):
        """Set the hatch style (for fills)."""
        # 将填充样式 hatch 存储到对象属性 _hatch 中
        self._hatch = hatch

    def get_hatch(self):
        """Get the current hatch style."""
        # 返回当前填充样式 _hatch
        return self._hatch
    def get_hatch_path(self, density=6.0):
        """
        Return a `.Path` for the current hatch.

        Parameters
        ----------
        density : float, optional
            Density of the hatch lines.

        Returns
        -------
        `.Path` or `None`
            The Path object representing the hatch, or None if no hatch is set.
        """
        hatch = self.get_hatch()  # 获取当前设定的图案填充方式
        if hatch is None:  # 如果没有设定图案填充方式，则返回 None
            return None
        return Path.hatch(hatch, density)  # 根据图案填充方式和密度创建路径对象

    def get_hatch_color(self):
        """
        Get the hatch color.

        Returns
        -------
        any
            The current hatch color.
        """
        return self._hatch_color  # 返回当前图案填充的颜色

    def set_hatch_color(self, hatch_color):
        """
        Set the hatch color.

        Parameters
        ----------
        hatch_color : any
            The color to set for hatch filling.
        """
        self._hatch_color = hatch_color  # 设置图案填充的颜色

    def get_hatch_linewidth(self):
        """
        Get the hatch linewidth.

        Returns
        -------
        float
            The current linewidth for hatch filling.
        """
        return self._hatch_linewidth  # 返回当前图案填充的线宽

    def get_sketch_params(self):
        """
        Return the sketch parameters for the artist.

        Returns
        -------
        tuple or `None`
            Sketch parameters consisting of (scale, length, randomness),
            or None if no sketch parameters were set.
        """
        return self._sketch  # 返回当前设定的素描参数

    def set_sketch_params(self, scale=None, length=None, randomness=None):
        """
        Set the sketch parameters.

        Parameters
        ----------
        scale : float, optional
            The amplitude of the wiggle perpendicular to the source line.
        length : float, optional, default: 128
            The length of the wiggle along the line.
        randomness : float, optional, default: 16
            The scale factor by which the length is shrunken or expanded.
        """
        self._sketch = (
            None if scale is None
            else (scale, length or 128., randomness or 16.))  # 设置素描参数，如果参数为 None 则不设定相应的素描特效
    def _timer_set_single_shot(self):
        """
        Set the timer to operate in single shot mode.

        This method should be overridden in subclasses that support single
        shot timers. If not overridden, the TimerBase class will store the
        single_shot flag and the `_on_timer` method should handle single shot
        behavior accordingly.
        """
        pass

    def _timer_set_interval(self, interval):
        """
        Set the interval for the timer.

        Parameters
        ----------
        interval : int
            Timer interval in milliseconds.
        """
        pass

    def _on_timer(self):
        """
        Internal function to handle timer events.

        This method should be overridden in subclasses to execute the
        callbacks associated with timer events.
        """
        pass
    def interval(self, interval):
        # 强制将间隔转换为整数，因为后端不支持分数毫秒，
        # 并且一些后端会报错或警告。
        # 一些后端在 interval == 0 时也会失败，因此确保至少为 1 毫秒
        interval = max(int(interval), 1)
        # 设置对象的间隔时间
        self._interval = interval
        # 调用方法来设置底层定时器的间隔时间
        self._timer_set_interval()

    @property
    def single_shot(self):
        """是否应在单次运行后停止该定时器。"""
        # 返回单次运行标志位
        return self._single

    @single_shot.setter
    def single_shot(self, ss):
        # 设置单次运行标志位
        self._single = ss
        # 调用方法来设置底层定时器的单次运行设置
        self._timer_set_single_shot()

    def add_callback(self, func, *args, **kwargs):
        """
        注册一个函数 *func* 在定时器触发时调用。可以传递额外的参数给 *func*。

        返回 *func*，这使得可以将其用作装饰器。
        """
        # 添加回调函数及其参数到回调列表
        self.callbacks.append((func, args, kwargs))
        return func

    def remove_callback(self, func, *args, **kwargs):
        """
        从回调列表中移除函数 *func*。

        *args* 和 *kwargs* 是可选的，用于区分注册了不同参数的同一个函数。
        这种行为已被弃用。未来版本中，`*args, **kwargs` 将不再考虑；
        若要保持特定回调函数可移除性，请将其作为 `functools.partial` 对象传递给 `add_callback`。
        """
        if args or kwargs:
            # 如果存在 *args 或 **kwargs，则发出警告提示
            _api.warn_deprecated(
                "3.1", message="在将来的版本中，Timer.remove_callback 将不再接受 *args, **kwargs，"
                "而是会移除所有与可调用对象匹配的回调；要保持特定回调函数的可移除性，请将其作为 functools.partial 对象传递给 add_callback。")
            # 从回调列表中移除带有特定参数的函数
            self.callbacks.remove((func, args, kwargs))
        else:
            # 否则，只移除与指定函数匹配的回调函数
            funcs = [c[0] for c in self.callbacks]
            if func in funcs:
                self.callbacks.pop(funcs.index(func))

    def _timer_set_interval(self):
        """用于设置底层定时器对象的间隔时间。"""

    def _timer_set_single_shot(self):
        """用于设置底层定时器对象的单次运行模式。"""
    def _on_timer(self):
        """
        运行所有已注册为回调函数的函数。如果函数返回 False（或 0），则表示它们不应再被调用。
        如果没有回调函数，定时器会自动停止。
        """
        # 遍历回调函数列表，每个元素包含函数、参数和关键字参数
        for func, args, kwargs in self.callbacks:
            # 调用函数，并接收返回值
            ret = func(*args, **kwargs)
            # 上面的文档字符串解释了为什么我们在这里使用 `if ret == 0`，而不是 `if not ret`。
            # 这样可以捕捉到 `ret == False`，因为 `False == 0`，但不会触发 linters 的警告
            # 参考：https://docs.python.org/3/library/stdtypes.html#boolean-values
            if ret == 0:
                # 如果返回值为 0，则从回调函数列表中移除该回调函数及其参数
                self.callbacks.remove((func, args, kwargs))

        # 如果回调函数列表为空，则停止定时器
        if len(self.callbacks) == 0:
            self.stop()
class Event:
    """
    A Matplotlib event.

    The following attributes are defined and shown with their default values.
    Subclasses may define additional attributes.

    Attributes
    ----------
    name : str
        The event name.
    canvas : `FigureCanvasBase`
        The backend-specific canvas instance generating the event.
    guiEvent
        The GUI event that triggered the Matplotlib event.
    """

    def __init__(self, name, canvas, guiEvent=None):
        # 初始化事件对象，设置事件名称、画布实例以及可选的 GUI 事件
        self.name = name
        self.canvas = canvas
        self._guiEvent = guiEvent
        self._guiEvent_deleted = False

    def _process(self):
        """Process this event on ``self.canvas``, then unset ``guiEvent``."""
        # 处理事件，调用画布的回调函数来处理当前事件，并标记 GUI 事件已被处理
        self.canvas.callbacks.process(self.name, self)
        self._guiEvent_deleted = True

    @property
    def guiEvent(self):
        # 获取 GUI 事件属性；在过渡期结束后，将删除 _guiEvent_deleted，使 guiEvent 成为一个普通属性
        if self._guiEvent_deleted:
            _api.warn_deprecated(
                "3.8", message="Accessing guiEvent outside of the original GUI event "
                "handler is unsafe and deprecated since %(since)s; in the future, the "
                "attribute will be set to None after quitting the event handler.  You "
                "may separately record the value of the guiEvent attribute at your own "
                "risk.")
        return self._guiEvent


class DrawEvent(Event):
    """
    An event triggered by a draw operation on the canvas.

    In most backends, callbacks subscribed to this event will be fired after
    the rendering is complete but before the screen is updated. Any extra
    artists drawn to the canvas's renderer will be reflected without an
    explicit call to ``blit``.

    .. warning::

       Calling ``canvas.draw`` and ``canvas.blit`` in these callbacks may
       not be safe with all backends and may cause infinite recursion.

    A DrawEvent has a number of special attributes in addition to those defined
    by the parent `Event` class.

    Attributes
    ----------
    renderer : `RendererBase`
        The renderer for the draw event.
    """
    def __init__(self, name, canvas, renderer):
        # 初始化绘制事件，设置事件名称、画布实例以及绘制渲染器
        super().__init__(name, canvas)
        self.renderer = renderer


class ResizeEvent(Event):
    """
    An event triggered by a canvas resize.

    A ResizeEvent has a number of special attributes in addition to those
    defined by the parent `Event` class.

    Attributes
    ----------
    width : int
        Width of the canvas in pixels.
    height : int
        Height of the canvas in pixels.
    """

    def __init__(self, name, canvas):
        # 初始化大小调整事件，设置事件名称和画布实例，并获取画布的宽度和高度
        super().__init__(name, canvas)
        self.width, self.height = canvas.get_width_height()


class CloseEvent(Event):
    """An event triggered by a figure being closed."""


class LocationEvent(Event):
    """
    An event that has a screen location.
    """
    # 定义 LocationEvent 类，它继承自 Event 类，并具有额外的特殊属性。
    
    # 属性：
    # x, y : int or None
    #     事件在画布上的像素位置，从左下角计算。
    # inaxes : `~matplotlib.axes.Axes` or None
    #     鼠标所在的 `~.axes.Axes` 实例，如果有的话。
    # xdata, ydata : float or None
    #     鼠标在 *inaxes* 中的数据坐标，如果鼠标不在 Axes 上则为 *None*。
    # modifiers : frozenset
    #     当前按下的键盘修饰键集合（KeyEvent 除外）。
    
    # 完全删除所有对 lastevent 的引用，自去除后将不再使用。
    _lastevent = None
    
    # 用于返回 _lastevent 属性值的类属性，已在 3.8 版本被弃用。
    lastevent = _api.deprecated("3.8")(
        _api.classproperty(lambda cls: cls._lastevent))
    
    # 用于存储最后 Axes 的引用，初始值为 None。
    _last_axes_ref = None
    
    def __init__(self, name, canvas, x, y, guiEvent=None, *, modifiers=None):
        # 调用父类 Event 的初始化方法
        super().__init__(name, canvas, guiEvent=guiEvent)
        # x 位置 - 从画布左侧的像素位置
        self.x = int(x) if x is not None else x
        # y 位置 - 从画布底部的像素位置
        self.y = int(y) if y is not None else y
        self.inaxes = None  # 鼠标所在的 Axes 实例
        self.xdata = None   # 鼠标在数据坐标中的 x 坐标
        self.ydata = None   # 鼠标在数据坐标中的 y 坐标
        # 当前按下的键盘修饰键集合，如果没有则为空集合
        self.modifiers = frozenset(modifiers if modifiers is not None else [])
    
        if x is None or y is None:
            # 如果没有 (x, y) 信息，则无法确定事件是否在 Axes 上
            return
    
        # 设置鼠标所在的 Axes 实例
        self._set_inaxes(self.canvas.inaxes((x, y))
                         if self.canvas.mouse_grabber is None else
                         self.canvas.mouse_grabber,
                         (x, y))
    
    def _set_inaxes(self, inaxes, xy=None):
        # 设置鼠标所在的 Axes 实例
        self.inaxes = inaxes
        if inaxes is not None:
            try:
                # 尝试将像素坐标 (xy 或者 self.x, self.y) 转换为数据坐标
                self.xdata, self.ydata = inaxes.transData.inverted().transform(
                    xy if xy is not None else (self.x, self.y))
            except ValueError:
                pass
# 定义了一个枚举类 `MouseButton`，用于表示鼠标按键的整数值
class MouseButton(IntEnum):
    LEFT = 1
    MIDDLE = 2
    RIGHT = 3
    BACK = 8
    FORWARD = 9

# 定义了一个鼠标事件类 `MouseEvent`，继承自 `LocationEvent` 类
class MouseEvent(LocationEvent):
    """
    A mouse event ('button_press_event', 'button_release_event', \
'scroll_event', 'motion_notify_event').

    A MouseEvent has a number of special attributes in addition to those
    defined by the parent `Event` and `LocationEvent` classes.

    Attributes
    ----------
    button : None or `MouseButton` or {'up', 'down'}
        The button pressed. 'up' and 'down' are used for scroll events.

        Note that LEFT and RIGHT actually refer to the "primary" and
        "secondary" buttons, i.e. if the user inverts their left and right
        buttons ("left-handed setting") then the LEFT button will be the one
        physically on the right.

        If this is unset, *name* is "scroll_event", and *step* is nonzero, then
        this will be set to "up" or "down" depending on the sign of *step*.

    key : None or str
        The key pressed when the mouse event triggered, e.g. 'shift'.
        See `KeyEvent`.

        .. warning::
           This key is currently obtained from the last 'key_press_event' or
           'key_release_event' that occurred within the canvas.  Thus, if the
           last change of keyboard state occurred while the canvas did not have
           focus, this attribute will be wrong.  On the other hand, the
           ``modifiers`` attribute should always be correct, but it can only
           report on modifier keys.

    step : float
        The number of scroll steps (positive for 'up', negative for 'down').
        This applies only to 'scroll_event' and defaults to 0 otherwise.

    dblclick : bool
        Whether the event is a double-click. This applies only to
        'button_press_event' and is False otherwise. In particular, it's
        not used in 'button_release_event'.

    Examples
    --------
    ::

        def on_press(event):
            print('you pressed', event.button, event.xdata, event.ydata)

        cid = fig.canvas.mpl_connect('button_press_event', on_press)
    """

    def __init__(self, name, canvas, x, y, button=None, key=None,
                 step=0, dblclick=False, guiEvent=None, *, modifiers=None):
        # 调用父类 `LocationEvent` 的初始化方法
        super().__init__(
            name, canvas, x, y, guiEvent=guiEvent, modifiers=modifiers)
        # 如果 `button` 是 `MouseButton` 枚举类的一个值，则转换成枚举类型
        if button in MouseButton.__members__.values():
            button = MouseButton(button)
        # 如果事件名称为 "scroll_event" 且 `button` 为 None，则根据 `step` 的正负设置 `button` 为 "up" 或 "down"
        if name == "scroll_event" and button is None:
            if step > 0:
                button = "up"
            elif step < 0:
                button = "down"
        # 设置对象的属性
        self.button = button
        self.key = key
        self.step = step
        self.dblclick = dblclick

    def __str__(self):
        # 返回对象的字符串表示，包括事件名称、坐标、按键状态等信息
        return (f"{self.name}: "
                f"xy=({self.x}, {self.y}) xydata=({self.xdata}, {self.ydata}) "
                f"button={self.button} dblclick={self.dblclick} "
                f"inaxes={self.inaxes}")


class PickEvent(Event):
    """
    Placeholder class for PickEvent. Derived classes should specify their
    own constructors and methods as needed.
    """
    """
    A pick event.

    This event is fired when the user picks a location on the canvas
    sufficiently close to an artist that has been made pickable with
    `.Artist.set_picker`.

    A PickEvent has a number of special attributes in addition to those defined
    by the parent `Event` class.

    Attributes
    ----------
    mouseevent : `MouseEvent`
        The mouse event that generated the pick.
    artist : `~matplotlib.artist.Artist`
        The picked artist.  Note that artists are not pickable by default
        (see `.Artist.set_picker`).
    other
        Additional attributes may be present depending on the type of the
        picked object; e.g., a `.Line2D` pick may define different extra
        attributes than a `.PatchCollection` pick.

    Examples
    --------
    Bind a function ``on_pick()`` to pick events, that prints the coordinates
    of the picked data point::

        ax.plot(np.rand(100), 'o', picker=5)  # 5 points tolerance

        def on_pick(event):
            line = event.artist
            xdata, ydata = line.get_data()
            ind = event.ind
            print(f'on pick line: {xdata[ind]:.3f}, {ydata[ind]:.3f}')

        cid = fig.canvas.mpl_connect('pick_event', on_pick)
    """

    # 初始化 PickEvent 对象
    def __init__(self, name, canvas, mouseevent, artist,
                 guiEvent=None, **kwargs):
        # 如果未提供 guiEvent，则使用 mouseevent 的 guiEvent
        if guiEvent is None:
            guiEvent = mouseevent.guiEvent
        # 调用父类 Event 的构造函数，传入 name, canvas, guiEvent
        super().__init__(name, canvas, guiEvent)
        # 设置 PickEvent 特有的属性：mouseevent 和 artist
        self.mouseevent = mouseevent
        self.artist = artist
        # 更新 PickEvent 对象的其他属性（如果有）
        self.__dict__.update(kwargs)
# KeyEvent 类继承自 LocationEvent 类，表示键盘事件（按键按下或释放）。
class KeyEvent(LocationEvent):
    """
    A key event (key press, key release).

    A KeyEvent has a number of special attributes in addition to those defined
    by the parent `Event` and `LocationEvent` classes.

    Attributes
    ----------
    key : None or str
        The key(s) pressed. Could be *None*, a single case sensitive Unicode
        character ("g", "G", "#", etc.), a special key ("control", "shift",
        "f1", "up", etc.) or a combination of the above (e.g., "ctrl+alt+g",
        "ctrl+alt+G").

    Notes
    -----
    Modifier keys will be prefixed to the pressed key and will be in the order
    "ctrl", "alt", "super". The exception to this rule is when the pressed key
    is itself a modifier key, therefore "ctrl+alt" and "alt+control" can both
    be valid key values.

    Examples
    --------
    ::

        def on_key(event):
            print('you pressed', event.key, event.xdata, event.ydata)

        cid = fig.canvas.mpl_connect('key_press_event', on_key)
    """

    # KeyEvent 的构造函数，初始化键盘事件对象。
    def __init__(self, name, canvas, key, x=0, y=0, guiEvent=None):
        # 调用父类 LocationEvent 的构造函数进行基本初始化
        super().__init__(name, canvas, x, y, guiEvent=guiEvent)
        # 设置 KeyEvent 特有的 key 属性，表示按下的键或键组合
        self.key = key


# 默认的键盘事件回调函数。
def _key_handler(event):
    # 对按键事件的处理逻辑。
    # 如果事件是按键按下事件
    if event.name == "key_press_event":
        # 将按下的键赋值给 event.canvas._key
        event.canvas._key = event.key
    # 如果事件是按键释放事件
    elif event.name == "key_release_event":
        # 将 event.canvas._key 设为 None
        event.canvas._key = None


# 默认的鼠标事件回调函数。
def _mouse_handler(event):
    # 对鼠标事件的处理逻辑。
    # 如果事件是鼠标按下事件
    if event.name == "button_press_event":
        # 将按下的按钮赋值给 event.canvas._button
        event.canvas._button = event.button
    # 如果事件是鼠标释放事件
    elif event.name == "button_release_event":
        # 将 event.canvas._button 设为 None
        event.canvas._button = None
    # 如果事件是鼠标移动事件且没有按键按下
    elif event.name == "motion_notify_event" and event.button is None:
        # 将 event.canvas._button 的值赋给 event.button
        event.button = event.canvas._button
    # 如果事件中没有键按下
    if event.key is None:
        # 将 event.canvas._key 的值赋给 event.key
        event.key = event.canvas._key
    # 发出 axes_enter/axes_leave 事件。
    # 如果事件类型为鼠标移动事件
    if event.name == "motion_notify_event":
        # 获取上一个参考的Axes对象
        last_ref = LocationEvent._last_axes_ref
        # 如果引用存在，则获取其实际对象；否则为None
        last_axes = last_ref() if last_ref else None
        # 如果上一个Axes对象不等于当前事件所在的Axes对象
        if last_axes != event.inaxes:
            # 如果上一个Axes对象不为None，创建一个合成的Axes离开事件LocationEvent
            if last_axes is not None:
                # 创建合成的Axes离开事件，需要手动设置其inaxes属性为None，因为此时光标已经离开该Axes
                # 使用内部方法_set_inaxes确保xdata和ydata属性也正确设置
                try:
                    leave_event = LocationEvent(
                        "axes_leave_event", last_axes.figure.canvas,
                        event.x, event.y, event.guiEvent,
                        modifiers=event.modifiers)
                    leave_event._set_inaxes(last_axes)
                    # 触发Axes离开事件的处理
                    last_axes.figure.canvas.callbacks.process(
                        "axes_leave_event", leave_event)
                except Exception:
                    pass  # 可能最后的画布已经被销毁
            # 如果当前事件在一个有效的Axes对象中
            if event.inaxes is not None:
                # 触发Axes进入事件的处理
                event.canvas.callbacks.process("axes_enter_event", event)
        # 记录当前事件所在的Axes对象的弱引用，用于下一次参考
        LocationEvent._last_axes_ref = (
            weakref.ref(event.inaxes) if event.inaxes else None)
        # 记录最后一个事件，如果是"figure_leave_event"则设置为None
        LocationEvent._lastevent = (
            None if event.name == "figure_leave_event" else event)
# 获取用于保存 `.Figure` 的渲染器
def _get_renderer(figure, print_method=None):
    """
    Get the renderer that would be used to save a `.Figure`.

    If you need a renderer without any active draw methods use
    renderer._draw_disabled to temporary patch them out at your call site.
    """
    # 定义一个异常类，用于提前结束 Figure.draw() 的执行
    class Done(Exception):
        pass

    # 定义一个内部函数 _draw，用于触发绘图操作并抛出 Done 异常
    def _draw(renderer): raise Done(renderer)

    # 使用 cbook._setattr_cm 上下文管理器设置 figure 对象的 draw 属性为 _draw 函数
    with cbook._setattr_cm(figure, draw=_draw), ExitStack() as stack:
        # 如果 print_method 未指定，则使用 figure.canvas 的默认文件类型来获取打印方法
        if print_method is None:
            fmt = figure.canvas.get_default_filetype()
            # 即使是 canvas 的默认输出类型，可能也需要进行 canvas 切换
            print_method = stack.enter_context(
                figure.canvas._switch_canvas_and_return_print_method(fmt))
        try:
            # 调用 print_method 方法，传入一个空的 BytesIO 对象
            print_method(io.BytesIO())
        except Done as exc:
            # 捕获 Done 异常，获取 renderer 对象并返回
            renderer, = exc.args
            return renderer
        else:
            # 如果 print_method 没有调用 Figure.draw，则抛出运行时错误
            raise RuntimeError(f"{print_method} did not call Figure.draw, so "
                               f"no renderer is available")


# 将 _no_output_draw 升级到 figure 层级，但保留此函数以防有人调用它
def _no_output_draw(figure):
    figure.draw_without_rendering()


# 判断当前环境是否为非交互式的终端 IPython
def _is_non_interactive_terminal_ipython(ip):
    """
    Return whether we are in a terminal IPython, but non interactive.

    When in _terminal_ IPython, ip.parent will have and `interact` attribute,
    if this attribute is False we do not setup eventloop integration as the
    user will _not_ interact with IPython. In all other case (ZMQKernel, or is
    interactive), we do.
    """
    return (hasattr(ip, 'parent')
            and (ip.parent is not None)
            and getattr(ip.parent, 'interact', None) is False)


# 允许通过发送 SIGINT 信号来终止绘图的上下文管理器
@contextmanager
def _allow_interrupt(prepare_notifier, handle_sigint):
    """
    A context manager that allows terminating a plot by sending a SIGINT.  It
    is necessary because the running backend prevents the Python interpreter
    from running and processing signals (i.e., to raise a KeyboardInterrupt).
    To solve this, one needs to somehow wake up the interpreter and make it
    close the plot window.  We do this by using the signal.set_wakeup_fd()
    function which organizes a write of the signal number into a socketpair.
    A backend-specific function, *prepare_notifier*, arranges to listen to
    the pair's read socket while the event loop is running.  (If it returns a
    notifier object, that object is kept alive while the context manager runs.)

    If SIGINT was indeed caught, after exiting the on_signal() function the
    interpreter reacts to the signal according to the handler function which
    had been set up by a signal.signal() call; here, we arrange to call the
    backend-specific *handle_sigint* function.  Finally, we call the old SIGINT
    """

    # 一个上下文管理器，允许通过发送 SIGINT 信号来终止绘图
    """
    A context manager that allows terminating a plot by sending a SIGINT.  It
    is necessary because the running backend prevents the Python interpreter
    from running and processing signals (i.e., to raise a KeyboardInterrupt).
    To solve this, one needs to somehow wake up the interpreter and make it
    close the plot window.  We do this by using the signal.set_wakeup_fd()
    function which organizes a write of the signal number into a socketpair.
    A backend-specific function, *prepare_notifier*, arranges to listen to
    the pair's read socket while the event loop is running.  (If it returns a
    notifier object, that object is kept alive while the context manager runs.)

    If SIGINT was indeed caught, after exiting the on_signal() function the
    interpreter reacts to the signal according to the handler function which
    had been set up by a signal.signal() call; here, we arrange to call the
    backend-specific *handle_sigint* function.  Finally, we call the old SIGINT
    """
    # 获取当前的 SIGINT 信号处理函数
    old_sigint_handler = signal.getsignal(signal.SIGINT)
    # 如果旧的 SIGINT 处理函数是 None、SIG_IGN 或者 SIG_DFL，则直接返回，不做处理
    if old_sigint_handler in (None, signal.SIG_IGN, signal.SIG_DFL):
        yield
        return

    # 创建一个非阻塞的 socket 对象对，用于信号处理
    wsock, rsock = socket.socketpair()
    wsock.setblocking(False)
    rsock.setblocking(False)
    # 设置信号唤醒文件描述符，用于接收信号
    old_wakeup_fd = signal.set_wakeup_fd(wsock.fileno())
    # 准备通知器，根据 rsock 创建
    notifier = prepare_notifier(rsock)

    # 定义一个处理函数，保存参数并处理 SIGINT 信号
    def save_args_and_handle_sigint(*args):
        nonlocal handler_args
        handler_args = args
        handle_sigint()

    # 设置新的 SIGINT 处理函数为 save_args_and_handle_sigint
    signal.signal(signal.SIGINT, save_args_and_handle_sigint)
    try:
        # 执行 yield，暂停执行直到 finally 块完成
        yield
    finally:
        # 关闭 socket 对
        wsock.close()
        rsock.close()
        # 恢复原来的信号唤醒文件描述符
        signal.set_wakeup_fd(old_wakeup_fd)
        # 恢复原来的 SIGINT 处理函数
        signal.signal(signal.SIGINT, old_sigint_handler)
        # 如果存在保存的处理函数参数，则调用原来的 SIGINT 处理函数
        if handler_args is not None:
            old_sigint_handler(*handler_args)
# FigureCanvasBase 类定义了用于绘制图形的画布基类。

class FigureCanvasBase:
    """
    The canvas the figure renders into.

    Attributes
    ----------
    figure : `~matplotlib.figure.Figure`
        A high-level figure instance.
    """

    # 如果需要交互式框架，设置为 {"qt", "gtk3", "gtk4", "wx", "tk", "macosx"} 中的一个，否则为 None。
    required_interactive_framework = None

    # 由 new_manager 实例化的管理器类。
    # （这是一个类属性，因为管理器类当前在画布类之后定义，但也可以在定义两个类之后分配
    # ``FigureCanvasBase.manager_class = FigureManagerBase``。）
    manager_class = _api.classproperty(lambda cls: FigureManagerBase)

    # 支持的事件列表
    events = [
        'resize_event',
        'draw_event',
        'key_press_event',
        'key_release_event',
        'button_press_event',
        'button_release_event',
        'scroll_event',
        'motion_notify_event',
        'pick_event',
        'figure_enter_event',
        'figure_leave_event',
        'axes_enter_event',
        'axes_leave_event',
        'close_event'
    ]

    # 固定 DPI 设置为 None
    fixed_dpi = None

    # 文件类型由 _default_filetypes 决定
    filetypes = _default_filetypes

    # 支持 blit 的 Canvas 子类则返回 True
    @_api.classproperty
    def supports_blit(cls):
        """If this Canvas sub-class supports blitting."""
        return (hasattr(cls, "copy_from_bbox")
                and hasattr(cls, "restore_region"))

    # 初始化方法，figure 参数为 matplotlib.figure.Figure 实例
    def __init__(self, figure=None):
        from matplotlib.figure import Figure
        # 修复 IPython 后端到 GUI 的问题
        self._fix_ipython_backend2gui()
        self._is_idle_drawing = True  # 是否空闲绘制
        self._is_saving = False  # 是否正在保存
        if figure is None:
            figure = Figure()  # 创建一个新的 Figure 对象
        figure.set_canvas(self)  # 将画布对象设置给 figure
        self.figure = figure  # 设置 figure 实例
        self.manager = None  # 管理器对象，初始化为 None
        self.widgetlock = widgets.LockDraw()  # 绘图锁定
        self._button = None  # 当前按下的按钮
        self._key = None  # 当前按下的键
        self.mouse_grabber = None  # 当前抓取鼠标的 Axes 对象
        self.toolbar = None  # NavigationToolbar2 将设置这个属性
        self._is_idle_drawing = False  # 空闲绘制状态
        # 不希望多次缩放图形 DPI
        figure._original_dpi = figure.dpi
        self._device_pixel_ratio = 1  # 设备像素比率，默认为 1
        super().__init__()  # 调用父类初始化方法（通常是 GUI 小部件的初始化，如果有的话）

    # 回调函数属性，返回 figure 的 canvas 回调列表
    callbacks = property(lambda self: self.figure._canvas_callbacks)
    # 按钮选取 ID 属性，返回 figure 的按钮选取 ID
    button_pick_id = property(lambda self: self.figure._button_pick_id)
    # 滚动选取 ID 属性，返回 figure 的滚动选取 ID
    scroll_pick_id = property(lambda self: self.figure._scroll_pick_id)
    def _fix_ipython_backend2gui(cls):
        # 修复 IPython 中硬编码的模块到工具包的映射（用于 `ipython --auto`）。
        # 由于导入顺序问题，不能在导入时完成此操作，因此在创建画布时进行，
        # 每个类只需执行一次（因此使用 `cache`）。

        # 当 Python 3.12 达到终止支持状态（预计在 2028 年末），IPython < 8.24 将不再受支持时，
        # 此函数将不再需要。届时可以将此函数设置为无操作并弃用。
        mod_ipython = sys.modules.get("IPython")
        if mod_ipython is None or mod_ipython.version_info[:2] >= (8, 24):
            # 对于 IPython >= 8.24，不再需要使用 backend2gui，因为该功能已移至 Matplotlib。
            return

        import IPython
        ip = IPython.get_ipython()
        if not ip:
            return
        from IPython.core import pylabtools as pt
        if (not hasattr(pt, "backend2gui")
                or not hasattr(ip, "enable_matplotlib")):
            # 如果我们将补丁移动到 IPython 并删除这些 API，不要在我们这边中断。
            return
        # 根据所需的交互式框架，选择合适的 backend2gui 映射
        backend2gui_rif = {
            "qt": "qt",
            "gtk3": "gtk3",
            "gtk4": "gtk4",
            "wx": "wx",
            "macosx": "osx",
        }.get(cls.required_interactive_framework)
        if backend2gui_rif:
            # 如果当前 IPython 是非交互式终端，启用相应的 GUI
            if _is_non_interactive_terminal_ipython(ip):
                ip.enable_gui(backend2gui_rif)

    @classmethod
    def new_manager(cls, figure, num):
        """
        Create a new figure manager for *figure*, using this canvas class.

        Notes
        -----
        This method should not be reimplemented in subclasses.  If
        custom manager creation logic is needed, please reimplement
        ``FigureManager.create_with_canvas``.
        """
        return cls.manager_class.create_with_canvas(cls, figure, num)

    @contextmanager
    def _idle_draw_cntx(self):
        # 进入空闲绘制上下文，设置空闲绘制标志为 True
        self._is_idle_drawing = True
        try:
            yield
        finally:
            # 离开空闲绘制上下文，将空闲绘制标志重置为 False
            self._is_idle_drawing = False

    def is_saving(self):
        """
        Return whether the renderer is in the process of saving
        to a file, rather than rendering for an on-screen buffer.
        """
        # 返回渲染器是否正在保存到文件而不是为屏幕缓冲区渲染
        return self._is_saving

    def blit(self, bbox=None):
        """Blit the canvas in bbox (default entire canvas)."""
        # 在给定的 bbox 中传输画布（默认为整个画布）
    def inaxes(self, xy):
        """
        返回包含点 *xy* 的最顶层可见 `~.axes.Axes` 对象。

        Parameters
        ----------
        xy : (float, float)
            从画布左/底部开始的像素位置 (x, y)。

        Returns
        -------
        `~matplotlib.axes.Axes` 或 None
            包含该点的最顶层可见 Axes 对象，如果该点没有 Axes，则返回 None。
        """
        # 获取包含点 xy 并且可见的所有 Axes 对象列表
        axes_list = [a for a in self.figure.get_axes()
                     if a.patch.contains_point(xy) and a.get_visible()]
        # 如果存在满足条件的 Axes，则找到最顶层的 Artist 并返回其所属的 Axes
        if axes_list:
            axes = cbook._topmost_artist(axes_list)
        else:
            axes = None

        return axes

    def grab_mouse(self, ax):
        """
        设置正在捕获鼠标事件的子 `~.axes.Axes` 对象。

        通常由小部件自身调用。如果鼠标已被其他 Axes 对象捕获，则调用此方法会报错。
        """
        # 如果已经有其他 Axes 捕获了鼠标事件，则抛出运行时错误
        if self.mouse_grabber not in (None, ax):
            raise RuntimeError("Another Axes already grabs mouse input")
        # 将当前 Axes 对象设置为捕获鼠标事件的对象
        self.mouse_grabber = ax

    def release_mouse(self, ax):
        """
        释放由 `~.axes.Axes` *ax* 捕获的鼠标事件。

        通常由小部件调用。即使 *ax* 当前没有捕获鼠标事件，调用此方法也是可以的。
        """
        # 如果当前 Axes 对象正捕获鼠标事件，则将其释放
        if self.mouse_grabber is ax:
            self.mouse_grabber = None

    def set_cursor(self, cursor):
        """
        设置当前的鼠标指针。

        如果后端不显示任何内容，则此操作可能无效。

        如果需要，此方法应在设置鼠标指针后触发后端事件循环的更新，
        因为在执行长时间任务之前调用此方法可能导致 GUI 不更新。

        Parameters
        ----------
        cursor : `.Cursors`
            要在画布上显示的鼠标指针。注意：某些后端可能会更改整个窗口的鼠标指针。
        """

    def draw(self, *args, **kwargs):
        """
        渲染 `.Figure`。

        此方法必须遍历 Artist 树，即使不生成任何输出，
        因为它触发了延迟的工作，用户可能希望在将输出保存到磁盘之前访问这些工作。
        例如计算限制、自动限制和刻度值。
        """

    def draw_idle(self, *args, **kwargs):
        """
        在控制返回到 GUI 事件循环后，请求重新绘制小部件。

        即使在控制返回到 GUI 事件循环之前多次调用 `draw_idle`，图形也只会渲染一次。

        Notes
        -----
        后端可以选择重写此方法并实现自己的策略以防止多次渲染。
        """
        # 如果当前不是空闲绘制状态，则在空闲绘制上下文中执行绘制
        if not self._is_idle_drawing:
            with self._idle_draw_cntx():
                self.draw(*args, **kwargs)

    @property
    def device_pixel_ratio(self):
        """
        返回屏幕上用于画布的物理像素与逻辑像素的比率。

        默认情况下，该值为1，表示物理像素和逻辑像素大小相同。支持高DPI屏幕的子类可以设置
        该属性，以指示该比率不同。除非直接与画布交互，否则所有的Matplotlib交互仍然使用逻辑像素。
        """
        return self._device_pixel_ratio

    def _set_device_pixel_ratio(self, ratio):
        """
        设置用于画布的物理像素与逻辑像素的比率。

        支持高DPI屏幕的子类可以设置该属性，以指示该比率不同。画布本身将以物理大小创建，而客户端
        端将使用逻辑大小。因此，图形的DPI将按比例缩放。支持高DPI屏幕的实现应该使用物理像素
        处理事件，以确保转换回坐标轴空间的正确性。

        默认情况下，该值为1，表示物理像素和逻辑像素大小相同。

        Parameters
        ----------
        ratio : float
            用于画布的逻辑像素与物理像素的比率。

        Returns
        -------
        bool
            比率是否已更改。后端可以将此解释为调整窗口大小、重绘画布或更改任何其他相关属性的信号。
        """
        if self._device_pixel_ratio == ratio:
            return False
        # 在混合分辨率显示的情况下，如果设备像素比率发生变化，需要谨慎处理 -
        # 在这种情况下，我们需要相应地调整画布大小。某些后端提供指示DPI变化的事件，
        # 但那些没有的后端将在绘制之前更新此值。
        dpi = ratio * self.figure._original_dpi
        self.figure._set_dpi(dpi, forward=False)
        self._device_pixel_ratio = ratio
        return True
    def get_width_height(self, *, physical=False):
        """
        返回图形的宽度和高度，单位为整数点或像素。

        当图形用于高 DPI 屏幕时（并且后端支持），在设备像素比例缩放后截断为整数。

        Parameters
        ----------
        physical : bool, default: False
            是否返回真实的物理像素或逻辑像素。物理像素可能会被支持 HiDPI 的后端使用，
            但仍然使用实际大小配置画布。

        Returns
        -------
        width, height : int
            图形的尺寸，单位为点或像素，取决于后端。
        """
        return tuple(int(size / (1 if physical else self.device_pixel_ratio))
                     for size in self.figure.bbox.max)

    @classmethod
    def get_supported_filetypes(cls):
        """
        返回该后端支持的保存图形文件格式的字典。
        """
        return cls.filetypes

    @classmethod
    def get_supported_filetypes_grouped(cls):
        """
        返回该后端支持的保存图形文件格式的分组字典，
        其中键为文件类型名称（如 'Joint Photographic Experts Group'），
        值为用于该文件类型的文件扩展名列表（例如 ['jpg', 'jpeg']）。
        """
        groupings = {}
        for ext, name in cls.filetypes.items():
            groupings.setdefault(name, []).append(ext)
            groupings[name].sort()
        return groupings

    @contextmanager
    @classmethod
    def get_default_filetype(cls):
        """
        返回在 :rc:`savefig.format` 中指定的默认保存图形文件格式。

        返回的字符串不包含句点。此方法在仅支持单一文件类型的后端中被覆盖。
        """
        return rcParams['savefig.format']

    def get_default_filename(self):
        """
        返回一个适当的默认文件名，包括扩展名。
        """
        default_basename = (
            self.manager.get_window_title()
            if self.manager is not None
            else ''
        )
        default_basename = default_basename or 'image'
        # NT 路径中需避免的字符：
        # https://msdn.microsoft.com/en-us/library/windows/desktop/aa365247(v=vs.85).aspx#naming_conventions
        # 加上 ' '
        removed_chars = r'<>:"/\|?*\0 '
        default_basename = default_basename.translate(
            {ord(c): "_" for c in removed_chars})
        default_filetype = self.get_default_filetype()
        return f'{default_basename}.{default_filetype}'

    @_api.deprecated("3.8")
    # 实例化一个 FigureCanvasClass 的实例

    def switch_backends(self, FigureCanvasClass):
        """
        Instantiate an instance of FigureCanvasClass

        This is used for backend switching, e.g., to instantiate a
        FigureCanvasPS from a FigureCanvasGTK.  Note, deep copying is
        not done, so any changes to one of the instances (e.g., setting
        figure size or line props), will be reflected in the other
        """
        newCanvas = FigureCanvasClass(self.figure)
        # 将当前对象的 _is_saving 属性传递给新创建的画布对象
        newCanvas._is_saving = self._is_saving
        return newCanvas

    def mpl_connect(self, s, func):
        """
        Bind function *func* to event *s*.

        Parameters
        ----------
        s : str
            One of the following events ids:

            - 'button_press_event'
            - 'button_release_event'
            - 'draw_event'
            - 'key_press_event'
            - 'key_release_event'
            - 'motion_notify_event'
            - 'pick_event'
            - 'resize_event'
            - 'scroll_event'
            - 'figure_enter_event',
            - 'figure_leave_event',
            - 'axes_enter_event',
            - 'axes_leave_event'
            - 'close_event'.

        func : callable
            The callback function to be executed, which must have the
            signature::

                def func(event: Event) -> Any

            For the location events (button and key press/release), if the
            mouse is over the Axes, the ``inaxes`` attribute of the event will
            be set to the `~matplotlib.axes.Axes` the event occurs is over, and
            additionally, the variables ``xdata`` and ``ydata`` attributes will
            be set to the mouse location in data coordinates.  See `.KeyEvent`
            and `.MouseEvent` for more info.

            .. note::

                If func is a method, this only stores a weak reference to the
                method. Thus, the figure does not influence the lifetime of
                the associated object. Usually, you want to make sure that the
                object is kept alive throughout the lifetime of the figure by
                holding a reference to it.

        Returns
        -------
        cid
            A connection id that can be used with
            `.FigureCanvasBase.mpl_disconnect`.

        Examples
        --------
        ::

            def on_press(event):
                print('you pressed', event.button, event.xdata, event.ydata)

            cid = canvas.mpl_connect('button_press_event', on_press)
        """
        # 将事件 s 和函数 func 绑定到回调函数中
        return self.callbacks.connect(s, func)

    def mpl_disconnect(self, cid):
        """
        Disconnect the callback with id *cid*.

        Examples
        --------
        ::

            cid = canvas.mpl_connect('button_press_event', on_press)
            # ... later
            canvas.mpl_disconnect(cid)
        """
        # 断开指定 cid 的回调函数连接
        self.callbacks.disconnect(cid)

    # Internal subclasses can override _timer_cls instead of new_timer, though
    # 定义一个非公开的 API 类，用于内部子类的实例化。
    _timer_cls = TimerBase

    def new_timer(self, interval=None, callbacks=None):
        """
        创建一个新的特定于后端的 `.Timer` 子类实例。

        这对于通过后端的本机事件循环获取周期性事件非常有用。仅适用于具有 GUI 的后端。

        Parameters
        ----------
        interval : int
            定时器间隔，单位为毫秒。

        callbacks : list[tuple[callable, tuple, dict]]
            元组序列 (func, args, kwargs)，其中 ``func(*args, **kwargs)`` 将会每隔 *interval* 被定时器执行一次。

            返回 ``False`` 或 ``0`` 的回调将从定时器中移除。

        Examples
        --------
        >>> timer = fig.canvas.new_timer(callbacks=[(f1, (1,), {'a': 3})])
        """
        return self._timer_cls(interval=interval, callbacks=callbacks)

    def flush_events(self):
        """
        刷新图形界面事件队列，针对特定的图形。

        交互式后端需要重新实现此方法。
        """

    def start_event_loop(self, timeout=0):
        """
        启动一个阻塞的事件循环。

        这样的事件循环被交互式函数使用，例如 `~.Figure.ginput` 和 `~.Figure.waitforbuttonpress`，来等待事件。

        事件循环会阻塞，直到一个回调函数触发 `stop_event_loop`，或者达到 *timeout*。

        如果 *timeout* 是 0 或负数，则永不超时。

        只有交互式后端需要重新实现此方法，并且它依赖于正确实现的 `flush_events`。

        交互式后端应该以更本地化的方式实现这一点。
        """
        if timeout <= 0:
            timeout = np.inf
        timestep = 0.01
        counter = 0
        self._looping = True
        while self._looping and counter * timestep < timeout:
            self.flush_events()
            time.sleep(timestep)
            counter += 1

    def stop_event_loop(self):
        """
        停止当前的阻塞事件循环。

        交互式后端需要重新实现此方法，以匹配 `start_event_loop`。
        """
        self._looping = False
def key_press_handler(event, canvas=None, toolbar=None):
    """
    Implement the default Matplotlib key bindings for the canvas and toolbar
    described at :ref:`key-event-handling`.

    Parameters
    ----------
    event : `KeyEvent`
        A key press/release event.
    canvas : `FigureCanvasBase`, default: ``event.canvas``
        The backend-specific canvas instance.  This parameter is kept for
        back-compatibility, but, if set, should always be equal to
        ``event.canvas``.
    toolbar : `NavigationToolbar2`, default: ``event.canvas.toolbar``
        The navigation cursor toolbar.  This parameter is kept for
        back-compatibility, but, if set, should always be equal to
        ``event.canvas.toolbar``.
    """
    # 如果事件中的按键为 None，则直接返回，不做任何操作
    if event.key is None:
        return
    
    # 如果未指定 canvas，则使用事件中的 canvas
    if canvas is None:
        canvas = event.canvas
    
    # 如果未指定 toolbar，则使用 canvas 对象中的 toolbar
    if toolbar is None:
        toolbar = canvas.toolbar

    # 切换全屏模式（默认键为 'f', 'ctrl + f'）
    if event.key in rcParams['keymap.fullscreen']:
        try:
            canvas.manager.full_screen_toggle()
        except AttributeError:
            pass

    # 关闭当前图形（默认键为 'ctrl+w'）
    if event.key in rcParams['keymap.quit']:
        Gcf.destroy_fig(canvas.figure)
    
    # 关闭所有图形（默认键为 'ctrl+q'）
    if event.key in rcParams['keymap.quit_all']:
        Gcf.destroy_all()

    # 如果存在 toolbar 对象
    if toolbar is not None:
        # 回到起始位置（默认键为 'h', 'home' 和 'r'）
        if event.key in rcParams['keymap.home']:
            toolbar.home()
        # 向后导航（默认键为 'left', 'backspace' 和 'c'）
        elif event.key in rcParams['keymap.back']:
            toolbar.back()
        # 向前导航（默认键为 'right' 和 'v'）
        elif event.key in rcParams['keymap.forward']:
            toolbar.forward()
        # 平移（默认键为 'p'）
        elif event.key in rcParams['keymap.pan']:
            toolbar.pan()
            toolbar._update_cursor(event)
        # 缩放（默认键为 'o'）
        elif event.key in rcParams['keymap.zoom']:
            toolbar.zoom()
            toolbar._update_cursor(event)
        # 保存当前图形（默认键为 's'）
        elif event.key in rcParams['keymap.save']:
            toolbar.save_figure()

    # 如果事件中没有指定坐标轴，则直接返回
    if event.inaxes is None:
        return

    # 获取当前坐标轴对象
    ax = event.inaxes

    # 切换当前坐标轴的主要网格线（默认键为 'g'）
    # 在此处和以下（对于 'G'），如果任何网格线（主要或次要，横向或纵向）不处于统一状态，
    # 则不执行任何操作，以避免干扰用户
    def _get_uniform_gridstate(ticks):
        # 返回 True 或 False，表示所有网格线是否全部显示或全部隐藏，
        # 返回 None 表示网格线状态不一致。
        return (True if all(tick.gridline.get_visible() for tick in ticks) else
                False if not any(tick.gridline.get_visible() for tick in ticks) else
                None)
    # customization.
    # 检查事件是否匹配自定义键映射中的网格切换键
    if (event.key in rcParams['keymap.grid']
            # 排除非统一状态的次要网格
            and None not in [_get_uniform_gridstate(ax.xaxis.minorTicks),
                             _get_uniform_gridstate(ax.yaxis.minorTicks)]):
        # 获取当前主要x轴和y轴的统一网格状态
        x_state = _get_uniform_gridstate(ax.xaxis.majorTicks)
        y_state = _get_uniform_gridstate(ax.yaxis.majorTicks)
        cycle = [(False, False), (True, False), (True, True), (False, True)]
        try:
            # 循环切换到下一个网格状态
            x_state, y_state = (
                cycle[(cycle.index((x_state, y_state)) + 1) % len(cycle)])
        except ValueError:
            # 排除非统一状态的主要网格
            pass
        else:
            # 如果关闭主要网格，同时关闭次要网格
            ax.grid(x_state, which="major" if x_state else "both", axis="x")
            ax.grid(y_state, which="major" if y_state else "both", axis="y")
            canvas.draw_idle()
    
    # toggle major and minor grids in current Axes (default key 'G')
    # 检查事件是否匹配网格切换键映射中的次要网格切换键
    elif (event.key in rcParams['keymap.grid_minor']
            # 排除非统一状态的主要网格
            and None not in [_get_uniform_gridstate(ax.xaxis.majorTicks),
                             _get_uniform_gridstate(ax.yaxis.majorTicks)]):
        # 获取当前次要x轴和y轴的统一网格状态
        x_state = _get_uniform_gridstate(ax.xaxis.minorTicks)
        y_state = _get_uniform_gridstate(ax.yaxis.minorTicks)
        cycle = [(False, False), (True, False), (True, True), (False, True)]
        try:
            # 循环切换到下一个网格状态
            x_state, y_state = (
                cycle[(cycle.index((x_state, y_state)) + 1) % len(cycle)])
        except ValueError:
            # 排除非统一状态的次要网格
            pass
        else:
            # 切换当前Axes中x轴和y轴的主要和次要网格
            ax.grid(x_state, which="both", axis="x")
            ax.grid(y_state, which="both", axis="y")
            canvas.draw_idle()
    
    # toggle scaling of y-axes between 'log and 'linear' (default key 'l')
    # 检查事件是否匹配y轴缩放切换键映射中的键
    elif event.key in rcParams['keymap.yscale']:
        # 获取当前y轴的缩放类型
        scale = ax.get_yscale()
        if scale == 'log':
            ax.set_yscale('linear')
            ax.figure.canvas.draw_idle()
        elif scale == 'linear':
            try:
                ax.set_yscale('log')
            except ValueError as exc:
                _log.warning(str(exc))
                ax.set_yscale('linear')
            ax.figure.canvas.draw_idle()
    
    # toggle scaling of x-axes between 'log and 'linear' (default key 'k')
    # 检查事件是否匹配x轴缩放切换键映射中的键
    elif event.key in rcParams['keymap.xscale']:
        # 获取当前x轴的缩放类型
        scalex = ax.get_xscale()
        if scalex == 'log':
            ax.set_xscale('linear')
            ax.figure.canvas.draw_idle()
        elif scalex == 'linear':
            try:
                ax.set_xscale('log')
            except ValueError as exc:
                _log.warning(str(exc))
                ax.set_xscale('linear')
            ax.figure.canvas.draw_idle()
# 定义按钮按下事件处理函数，处理Matplotlib中的额外鼠标按钮操作
def button_press_handler(event, canvas=None, toolbar=None):
    """
    The default Matplotlib button actions for extra mouse buttons.

    Parameters are as for `key_press_handler`, except that *event* is a
    `MouseEvent`.
    """
    # 如果未提供canvas参数，则使用事件对象的canvas属性
    if canvas is None:
        canvas = event.canvas
    # 如果未提供toolbar参数，则使用canvas对象的toolbar属性
    if toolbar is None:
        toolbar = canvas.toolbar
    # 如果toolbar不为None，则根据事件的按钮名称执行对应的操作
    if toolbar is not None:
        button_name = str(MouseButton(event.button))
        # 如果按钮名称在rcParams['keymap.back']中，则执行toolbar的后退操作
        if button_name in rcParams['keymap.back']:
            toolbar.back()
        # 如果按钮名称在rcParams['keymap.forward']中，则执行toolbar的前进操作
        elif button_name in rcParams['keymap.forward']:
            toolbar.forward()


# 定义一个自定义异常类，用于在非GUI后端尝试显示图形时引发异常
class NonGuiException(Exception):
    """Raised when trying show a figure in a non-GUI backend."""
    pass


# 定义一个图形管理基类，提供对图形窗口和控制的抽象接口
class FigureManagerBase:
    """
    A backend-independent abstraction of a figure container and controller.

    The figure manager is used by pyplot to interact with the window in a
    backend-independent way. It's an adapter for the real (GUI) framework that
    represents the visual figure on screen.

    The figure manager is connected to a specific canvas instance, which in turn
    is connected to a specific figure instance. To access a figure manager for
    a given figure in user code, you typically use ``fig.canvas.manager``.

    GUI backends derive from this class to translate common operations such
    as *show* or *resize* to the GUI-specific code. Non-GUI backends do not
    support these operations and can just use the base class.

    This following basic operations are accessible:

    **Window operations**

    - `~.FigureManagerBase.show`
    - `~.FigureManagerBase.destroy`
    - `~.FigureManagerBase.full_screen_toggle`
    - `~.FigureManagerBase.resize`
    - `~.FigureManagerBase.get_window_title`
    - `~.FigureManagerBase.set_window_title`

    **Key and mouse button press handling**

    The figure manager sets up default key and mouse button press handling by
    hooking up the `.key_press_handler` to the matplotlib event system. This
    ensures the same shortcuts and mouse actions across backends.

    **Other operations**

    Subclasses will have additional attributes and functions to access
    additional functionality. This is of course backend-specific. For example,
    most GUI backends have ``window`` and ``toolbar`` attributes that give
    access to the native GUI widgets of the respective framework.

    Attributes
    ----------
    canvas : `FigureCanvasBase`
        The backend-specific canvas instance.

    num : int or str
        The figure number.

    key_press_handler_id : int
        The default key handler cid, when using the toolmanager.
        To disable the default key press handling use::

            figure.canvas.mpl_disconnect(
                figure.canvas.manager.key_press_handler_id)
    """
    pass  # 基类只提供接口定义，没有实际操作
    """
    button_press_handler_id : int
        The default mouse button handler cid, when using the toolmanager.
        To disable the default button press handling use::

            figure.canvas.mpl_disconnect(
                figure.canvas.manager.button_press_handler_id)
    """
    
    _toolbar2_class = None
    _toolmanager_toolbar_class = None

    def __init__(self, canvas, num):
        # 初始化 FigureManagerBase 实例
        self.canvas = canvas
        canvas.manager = self  # 存储对父对象的指针
        self.num = num
        self.set_window_title(f"Figure {num:d}")  # 设置窗口标题

        self.key_press_handler_id = None
        self.button_press_handler_id = None
        # 如果不是使用 toolmanager 工具栏，则连接键盘按键和鼠标按钮事件的处理器
        if rcParams['toolbar'] != 'toolmanager':
            self.key_press_handler_id = self.canvas.mpl_connect(
                'key_press_event', key_press_handler)
            self.button_press_handler_id = self.canvas.mpl_connect(
                'button_press_event', button_press_handler)

        self.toolmanager = (ToolManager(canvas.figure)
                            if mpl.rcParams['toolbar'] == 'toolmanager'
                            else None)
        # 根据 rcParams 中的 toolbar 设置，选择不同的工具栏类实例化
        if (mpl.rcParams["toolbar"] == "toolbar2"
                and self._toolbar2_class):
            self.toolbar = self._toolbar2_class(self.canvas)
        elif (mpl.rcParams["toolbar"] == "toolmanager"
                and self._toolmanager_toolbar_class):
            self.toolbar = self._toolmanager_toolbar_class(self.toolmanager)
        else:
            self.toolbar = None

        # 如果存在 toolmanager，则向其添加工具
        if self.toolmanager:
            tools.add_tools_to_manager(self.toolmanager)
            if self.toolbar:
                tools.add_tools_to_container(self.toolbar)

        # 将函数作为 axes observer 添加到 canvas.figure，当当前 Axes 改变时调用
        @self.canvas.figure.add_axobserver
        def notify_axes_change(fig):
            # 当 toolmanager 不存在但 toolbar 存在时，更新工具栏
            if self.toolmanager is None and self.toolbar is not None:
                self.toolbar.update()

    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num):
        """
        Create a manager for a given *figure* using a specific *canvas_class*.

        Backends should override this method if they have specific needs for
        setting up the canvas or the manager.
        """
        # 使用指定的 canvas_class 创建一个管理器实例
        return cls(canvas_class(figure), num)

    @classmethod
    def start_main_loop(cls):
        """
        Start the main event loop.

        This method is called by `.FigureManagerBase.pyplot_show`, which is the
        implementation of `.pyplot.show`.  To customize the behavior of
        `.pyplot.show`, interactive backends should usually override
        `~.FigureManagerBase.start_main_loop`; if more customized logic is
        necessary, `~.FigureManagerBase.pyplot_show` can also be overridden.
        """
    def pyplot_show(cls, *, block=None):
        """
        Show all figures.  This method is the implementation of `.pyplot.show`.

        To customize the behavior of `.pyplot.show`, interactive backends
        should usually override `~.FigureManagerBase.start_main_loop`; if more
        customized logic is necessary, `~.FigureManagerBase.pyplot_show` can
        also be overridden.

        Parameters
        ----------
        block : bool, optional
            Whether to block by calling ``start_main_loop``.  The default,
            None, means to block if we are neither in IPython's ``%pylab`` mode
            nor in ``interactive`` mode.
        """
        # 获取所有图形管理器
        managers = Gcf.get_all_fig_managers()
        # 如果没有图形管理器，直接返回
        if not managers:
            return
        # 遍历每个图形管理器
        for manager in managers:
            try:
                # 显示当前图形管理器管理的图形窗口，对非交互式后端发出警告
                manager.show()  # Emits a warning for non-interactive backend.
            except NonGuiException as exc:
                # 如果发生非 GUI 异常，以警告形式输出异常信息
                _api.warn_external(str(exc))
        # 如果 block 参数为 None
        if block is None:
            # Hack: 我们是否处于 IPython 的 %pylab 模式？
            # 在 pylab 模式下，IPython（>= 0.10）会将 _needmain 属性附加到 pyplot.show 上（始终设置为 False）。
            pyplot_show = getattr(sys.modules.get("matplotlib.pyplot"), "show", None)
            ipython_pylab = hasattr(pyplot_show, "_needmain")
            # 如果不是 IPython 的 %pylab 模式并且不处于交互模式，则设置 block 为 True
            block = not ipython_pylab and not is_interactive()
        # 如果需要阻塞，则调用类的 start_main_loop 方法
        if block:
            cls.start_main_loop()

    def show(self):
        """
        For GUI backends, show the figure window and redraw.
        For non-GUI backends, raise an exception, unless running headless (i.e.
        on Linux with an unset DISPLAY); this exception is converted to a
        warning in `.Figure.show`.
        """
        # 这应该在 GUI 后端中被重写。
        if sys.platform == "linux" and not os.environ.get("DISPLAY"):
            # 如果在 Linux 上并且 DISPLAY 环境变量未设置，则直接返回
            return
        # 否则，抛出非 GUI 异常，指示无法显示图形
        raise NonGuiException(
            f"{type(self.canvas).__name__} is non-interactive, and thus cannot be "
            f"shown")

    def destroy(self):
        pass

    def full_screen_toggle(self):
        pass

    def resize(self, w, h):
        """For GUI backends, resize the window (in physical pixels)."""

    def get_window_title(self):
        """
        Return the title text of the window containing the figure, or None
        if there is no window (e.g., a PS backend).
        """
        return 'image'

    def set_window_title(self, title):
        """
        Set the title text of the window containing the figure.

        This has no effect for non-GUI (e.g., PS) backends.

        Examples
        --------
        >>> fig = plt.figure()
        >>> fig.canvas.manager.set_window_title('My figure')
        """
cursors = tools.cursors

# 导入工具模块中的游标对象

class _Mode(str, Enum):
    """
    枚举类_Mode，继承自str类型，定义了三种模式：NONE, PAN, ZOOM
    """

    NONE = ""
    PAN = "pan/zoom"
    ZOOM = "zoom rect"

    def __str__(self):
        """
        返回枚举对象的值（即模式名称）
        """
        return self.value

    @property
    def _navigate_mode(self):
        """
        返回当前枚举对象的名称（如果不是_NONE模式的话）
        """
        return self.name if self is not _Mode.NONE else None


class NavigationToolbar2:
    """
    导航工具栏的基类，版本2。

    后端必须实现一个处理'button_press_event'和'button_release_event'的画布。
    参见:meth:`FigureCanvasBase.mpl_connect`了解更多信息。

    他们还必须定义

    :meth:`save_figure`
        保存当前图形。

    :meth:`draw_rubberband` (可选)
        绘制缩放到矩形的"橡皮筋"矩形。

    :meth:`set_message` (可选)
        显示消息。

    :meth:`set_history_buttons` (可选)
        您可以更改历史后退/前进按钮以指示禁用/启用状态。

    并且必须重写``__init__``来设置工具栏--不要忘记调用基类init。
    典型地，``__init__`` 需要设置与`home`、`back`、`forward`、`pan`、`zoom`和
    `save_figure`方法连接的工具栏按钮，并使用数据路径的“images”子目录中的标准图标。

    就是这样，剩下的我们来做！
    """

    toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),
        ('Pan',
         'Left button pans, Right button zooms\n'
         'x/y fixes axis, CTRL fixes aspect',
         'move', 'pan'),
        ('Zoom', 'Zoom to rectangle\nx/y fixes axis', 'zoom_to_rect', 'zoom'),
        ('Subplots', 'Configure subplots', 'subplots', 'configure_subplots'),
        (None, None, None, None),
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
    )
    # 初始化方法，接收一个 canvas 参数
    def __init__(self, canvas):
        # 将 canvas 参数赋值给实例变量 self.canvas
        self.canvas = canvas
        # 将当前 toolbar 对象赋值给 canvas 的 toolbar 属性
        canvas.toolbar = self
        # 创建一个 _Stack 对象作为导航堆栈，用于记录视图状态历史
        self._nav_stack = cbook._Stack()
        # 初始化鼠标光标为 POINTER，在初始绘制后会被设置
        self._last_cursor = tools.Cursors.POINTER

        # 绑定鼠标按下事件到 _zoom_pan_handler 方法
        self._id_press = self.canvas.mpl_connect(
            'button_press_event', self._zoom_pan_handler)
        # 绑定鼠标释放事件到 _zoom_pan_handler 方法
        self._id_release = self.canvas.mpl_connect(
            'button_release_event', self._zoom_pan_handler)
        # 绑定鼠标移动事件到 mouse_move 方法
        self._id_drag = self.canvas.mpl_connect(
            'motion_notify_event', self.mouse_move)
        # 初始化 _pan_info 和 _zoom_info 为 None，用于记录平移和缩放信息
        self._pan_info = None
        self._zoom_info = None

        # 初始化 mode 为 _Mode.NONE，用于在状态栏显示当前模式
        self.mode = _Mode.NONE  # a mode string for the status bar
        # 设置历史按钮的初始状态
        self.set_history_buttons()

    # 显示工具栏或状态栏上的消息
    def set_message(self, s):
        """Display a message on toolbar or in status bar."""

    # 绘制用于指示缩放限制的矩形橡皮筋
    def draw_rubberband(self, event, x0, y0, x1, y1):
        """
        Draw a rectangle rubberband to indicate zoom limits.

        Note that it is not guaranteed that ``x0 <= x1`` and ``y0 <= y1``.
        """

    # 移除橡皮筋
    def remove_rubberband(self):
        """Remove the rubberband."""

    # 恢复原始视图
    def home(self, *args):
        """
        Restore the original view.

        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
        # 调用导航堆栈的 home 方法，恢复原始视图状态
        self._nav_stack.home()
        # 设置历史按钮的状态
        self.set_history_buttons()
        # 更新视图
        self._update_view()

    # 后退到上一个视图状态
    def back(self, *args):
        """
        Move back up the view lim stack.

        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
        # 调用导航堆栈的 back 方法，后退到上一个视图状态
        self._nav_stack.back()
        # 设置历史按钮的状态
        self.set_history_buttons()
        # 更新视图
        self._update_view()

    # 前进到下一个视图状态
    def forward(self, *args):
        """
        Move forward in the view lim stack.

        For convenience of being directly connected as a GUI callback, which
        often get passed additional parameters, this method accepts arbitrary
        parameters, but does not use them.
        """
        # 调用导航堆栈的 forward 方法，前进到下一个视图状态
        self._nav_stack.forward()
        # 设置历史按钮的状态
        self.set_history_buttons()
        # 更新视图
        self._update_view()
    def _update_cursor(self, event):
        """
        Update the cursor after a mouse move event or a tool (de)activation.
        """
        # 检查当前模式和事件发生在坐标轴中且可导航
        if self.mode and event.inaxes and event.inaxes.get_navigate():
            # 如果当前模式为ZOOM且上次光标不是SELECT_REGION，则设置光标为SELECT_REGION
            if (self.mode == _Mode.ZOOM
                    and self._last_cursor != tools.Cursors.SELECT_REGION):
                self.canvas.set_cursor(tools.Cursors.SELECT_REGION)
                self._last_cursor = tools.Cursors.SELECT_REGION
            # 如果当前模式为PAN且上次光标不是MOVE，则设置光标为MOVE
            elif (self.mode == _Mode.PAN
                  and self._last_cursor != tools.Cursors.MOVE):
                self.canvas.set_cursor(tools.Cursors.MOVE)
                self._last_cursor = tools.Cursors.MOVE
        # 如果上述条件不满足且上次光标不是POINTER，则设置光标为POINTER
        elif self._last_cursor != tools.Cursors.POINTER:
            self.canvas.set_cursor(tools.Cursors.POINTER)
            self._last_cursor = tools.Cursors.POINTER

    @contextmanager
    def _wait_cursor_for_draw_cm(self):
        """
        Set the cursor to a wait cursor when drawing the canvas.

        In order to avoid constantly changing the cursor when the canvas
        changes frequently, do nothing if this context was triggered during the
        last second.  (Optimally we'd prefer only setting the wait cursor if
        the *current* draw takes too long, but the current draw blocks the GUI
        thread).
        """
        # 记录当前时间和上次绘制时间
        self._draw_time, last_draw_time = (
            time.time(), getattr(self, "_draw_time", -np.inf))
        # 如果距离上次绘制超过1秒，则设置光标为WAIT，并在此期间保持该设置
        if self._draw_time - last_draw_time > 1:
            try:
                self.canvas.set_cursor(tools.Cursors.WAIT)
                yield
            finally:
                # 恢复到之前的光标状态
                self.canvas.set_cursor(self._last_cursor)
        else:
            yield

    @staticmethod
    def _mouse_event_to_message(event):
        # 如果事件发生在坐标轴中且可导航
        if event.inaxes and event.inaxes.get_navigate():
            try:
                # 格式化鼠标事件的坐标信息
                s = event.inaxes.format_coord(event.xdata, event.ydata)
            except (ValueError, OverflowError):
                pass
            else:
                s = s.rstrip()
                # 获取鼠标悬停的艺术家对象
                artists = [a for a in event.inaxes._mouseover_set
                           if a.contains(event)[0] and a.get_visible()]
                if artists:
                    # 选择最上层的艺术家对象
                    a = cbook._topmost_artist(artists)
                    if a is not event.inaxes.patch:
                        # 获取艺术家对象的光标数据
                        data = a.get_cursor_data(event)
                        if data is not None:
                            # 格式化光标数据并加入到坐标信息中
                            data_str = a.format_cursor_data(data).rstrip()
                            if data_str:
                                s = s + '\n' + data_str
                return s
        return ""

    def mouse_move(self, event):
        # 更新光标
        self._update_cursor(event)
        # 设置消息内容为鼠标事件的详细信息
        self.set_message(self._mouse_event_to_message(event))
    # 处理缩放和平移操作的事件处理器
    def _zoom_pan_handler(self, event):
        # 如果当前模式为平移
        if self.mode == _Mode.PAN:
            # 如果事件类型为鼠标按下事件，调用平移处理方法
            if event.name == "button_press_event":
                self.press_pan(event)
            # 如果事件类型为鼠标释放事件，调用释放平移处理方法
            elif event.name == "button_release_event":
                self.release_pan(event)
        # 如果当前模式为缩放
        if self.mode == _Mode.ZOOM:
            # 如果事件类型为鼠标按下事件，调用缩放处理方法
            if event.name == "button_press_event":
                self.press_zoom(event)
            # 如果事件类型为鼠标释放事件，调用释放缩放处理方法
            elif event.name == "button_release_event":
                self.release_zoom(event)

    # 开始处理与事件相关的坐标轴交互
    def _start_event_axes_interaction(self, event, *, method):
        
        # 过滤器函数，确定事件涉及的坐标轴是否需要交互
        def _ax_filter(ax):
            return (ax.in_axes(event) and  # 判断事件是否在坐标轴内
                    ax.get_navigate() and  # 获取坐标轴导航状态
                    getattr(ax, f"can_{method}")()  # 判断坐标轴是否能进行指定方法的交互
                    )

        # 事件捕获函数，确定是否捕获事件
        def _capture_events(ax):
            f = ax.get_forward_navigation_events()
            if f == "auto":  # 根据patch的可见性确定是否捕获事件
                f = not ax.patch.get_visible()
            return not f

        # 获取所有相关的坐标轴
        axes = list(filter(_ax_filter, self.canvas.figure.get_axes()))

        if len(axes) == 0:
            return []

        # 如果导航栈为空，则设置当前视图为主视图
        if self._nav_stack() is None:
            self.push_current()  # 设置主页按钮为当前视图

        # 按zorder分组坐标轴（反向以后面的坐标轴先触发）
        grps = dict()
        for ax in reversed(axes):
            grps.setdefault(ax.get_zorder(), []).append(ax)

        axes_to_trigger = []
        # 逆序遍历zorder，直到找到捕获事件的坐标轴
        for zorder in sorted(grps, reverse=True):
            for ax in grps[zorder]:
                axes_to_trigger.append(ax)
                # 注意：共享坐标轴会自动触发，但孪生坐标轴不会！
                axes_to_trigger.extend(ax._twinned_axes.get_siblings(ax))

                if _capture_events(ax):
                    break  # 找到捕获事件的坐标轴后退出循环
            else:
                # 如果内层循环没有显示退出（即没有找到捕获事件的坐标轴），继续外层循环查找下一个zorder
                continue

            # 如果内层循环被显式退出，则外层循环也要退出
            break

        # 去除重复的触发坐标轴（但保持列表顺序）
        axes_to_trigger = list(dict.fromkeys(axes_to_trigger))

        return axes_to_trigger

    # 平移方法，切换平移/缩放工具
    def pan(self, *args):
        """
        切换平移/缩放工具。

        使用左键进行平移，使用右键进行缩放。
        """
        if not self.canvas.widgetlock.available(self):
            self.set_message("pan unavailable")  # 如果平移不可用，则设置消息并返回
            return
        if self.mode == _Mode.PAN:
            self.mode = _Mode.NONE
            self.canvas.widgetlock.release(self)  # 释放平移模式
        else:
            self.mode = _Mode.PAN
            self.canvas.widgetlock(self)  # 进入平移模式
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self.mode._navigate_mode)  # 设置所有坐标轴的导航模式为当前模式
    # 定义一个命名元组_PanInfo，用于保存与平移操作相关的信息：按钮、轴对象列表和连接标识符
    _PanInfo = namedtuple("_PanInfo", "button axes cid")

    def press_pan(self, event):
        """Callback for mouse button press in pan/zoom mode."""
        # 如果按下的按钮不是左键或右键，或者事件的 x 或 y 坐标为 None，则直接返回
        if (event.button not in [MouseButton.LEFT, MouseButton.RIGHT]
                or event.x is None or event.y is None):
            return

        # 调用 _start_event_axes_interaction 方法以开始与事件相关的轴交互，并指定方法为平移
        axes = self._start_event_axes_interaction(event, method="pan")
        if not axes:
            return

        # 在所有相关的轴对象上调用 ax.start_pan 方法，启动平移操作
        for ax in axes:
            ax.start_pan(event.x, event.y, event.button)

        # 断开之前的拖动事件连接，建立新的连接来处理平移过程中的鼠标拖动
        self.canvas.mpl_disconnect(self._id_drag)
        id_drag = self.canvas.mpl_connect("motion_notify_event", self.drag_pan)

        # 记录当前平移操作的信息，包括按钮、轴对象列表和拖动事件的连接标识符
        self._pan_info = self._PanInfo(
            button=event.button, axes=axes, cid=id_drag)

    def drag_pan(self, event):
        """Callback for dragging in pan/zoom mode."""
        # 对每个轴对象进行循环，执行平移操作
        for ax in self._pan_info.axes:
            # 使用按下时记录的按钮更安全，因为在移动过程中可能会按下多个按钮
            ax.drag_pan(self._pan_info.button, event.key, event.x, event.y)
        # 更新画布显示
        self.canvas.draw_idle()

    def release_pan(self, event):
        """Callback for mouse button release in pan/zoom mode."""
        # 如果没有记录到平移信息，则直接返回
        if self._pan_info is None:
            return
        # 断开当前平移操作的拖动事件连接
        self.canvas.mpl_disconnect(self._pan_info.cid)
        # 建立新的连接来处理鼠标移动事件
        self._id_drag = self.canvas.mpl_connect(
            'motion_notify_event', self.mouse_move)
        # 对每个轴对象执行结束平移操作
        for ax in self._pan_info.axes:
            ax.end_pan()
        # 更新画布显示
        self.canvas.draw_idle()
        # 清空平移信息
        self._pan_info = None
        # 将当前状态推入历史记录
        self.push_current()

    def zoom(self, *args):
        # 如果画布的 widgetlock 对象不可用，则设置消息并返回
        if not self.canvas.widgetlock.available(self):
            self.set_message("zoom unavailable")
            return
        """Toggle zoom to rect mode."""
        # 切换到或关闭矩形缩放模式
        if self.mode == _Mode.ZOOM:
            self.mode = _Mode.NONE
            self.canvas.widgetlock.release(self)
        else:
            self.mode = _Mode.ZOOM
            self.canvas.widgetlock(self)
        # 设置所有轴对象的导航模式为当前操作模式
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self.mode._navigate_mode)

    # 定义一个命名元组_ZoomInfo，用于保存与缩放操作相关的信息：方向、起始坐标、轴对象列表、连接标识符和颜色条对象
    _ZoomInfo = namedtuple("_ZoomInfo", "direction start_xy axes cid cbar")
    def press_zoom(self, event):
        """Callback for mouse button press in zoom to rect mode."""
        # 检查事件的按钮是否为左键或右键，以及鼠标坐标是否存在
        if (event.button not in [MouseButton.LEFT, MouseButton.RIGHT]
                or event.x is None or event.y is None):
            return

        # 获取与事件交互的 Axes 对象，并启动缩放操作
        axes = self._start_event_axes_interaction(event, method="zoom")
        if not axes:
            return

        # 将缩放操作与鼠标移动事件连接起来，获取连接标识符
        id_zoom = self.canvas.mpl_connect(
            "motion_notify_event", self.drag_zoom)

        # 如果存在 colorbar，则存储其方向以便后续使用
        parent_ax = axes[0]
        if hasattr(parent_ax, "_colorbar"):
            cbar = parent_ax._colorbar.orientation
        else:
            cbar = None

        # 存储缩放相关信息
        self._zoom_info = self._ZoomInfo(
            direction="in" if event.button == 1 else "out",
            start_xy=(event.x, event.y), axes=axes, cid=id_zoom, cbar=cbar)

    def drag_zoom(self, event):
        """Callback for dragging in zoom mode."""
        # 获取起始坐标和对应的 Axes 对象
        start_xy = self._zoom_info.start_xy
        ax = self._zoom_info.axes[0]
        # 对当前事件的坐标进行裁剪，确保在 Axes 边界内
        (x1, y1), (x2, y2) = np.clip(
            [start_xy, [event.x, event.y]], ax.bbox.min, ax.bbox.max)
        key = event.key

        # 如果存在 colorbar，并且其方向是水平或垂直，则强制扩展矩形框以适应短轴边界
        if self._zoom_info.cbar == "horizontal":
            key = "x"
        elif self._zoom_info.cbar == "vertical":
            key = "y"

        # 根据键值调整矩形框的边界
        if key == "x":
            y1, y2 = ax.bbox.intervaly
        elif key == "y":
            x1, x2 = ax.bbox.intervalx

        # 绘制橡皮筋效果
        self.draw_rubberband(event, x1, y1, x2, y2)
    def release_zoom(self, event):
        """Callback for mouse button release in zoom to rect mode."""
        # 如果没有启动放大操作，则直接返回
        if self._zoom_info is None:
            return

        # 不检查事件按钮类型，以便通过释放其他鼠标按钮来取消放大操作
        self.canvas.mpl_disconnect(self._zoom_info.cid)
        self.remove_rubberband()

        start_x, start_y = self._zoom_info.start_xy
        key = event.key
        # 如果是水平颜色条，则强制将按键设置为 "x"，以忽略在短轴上的放大取消操作
        if self._zoom_info.cbar == "horizontal":
            key = "x"
        elif self._zoom_info.cbar == "vertical":
            key = "y"
        # 忽略单击事件：5像素是一个阈值，允许用户通过小于5像素的放大来取消放大操作
        if ((abs(event.x - start_x) < 5 and key != "y") or
                (abs(event.y - start_y) < 5 and key != "x")):
            self.canvas.draw_idle()
            self._zoom_info = None
            return

        for i, ax in enumerate(self._zoom_info.axes):
            # 检测此 Axes 是否与之前的 Axes 共享轴，避免重复放大
            twinx = any(ax.get_shared_x_axes().joined(ax, prev)
                        for prev in self._zoom_info.axes[:i])
            twiny = any(ax.get_shared_y_axes().joined(ax, prev)
                        for prev in self._zoom_info.axes[:i])
            ax._set_view_from_bbox(
                (start_x, start_y, event.x, event.y),
                self._zoom_info.direction, key, twinx, twiny)

        self.canvas.draw_idle()
        self._zoom_info = None
        self.push_current()

    def push_current(self):
        """Push the current view limits and position onto the stack."""
        self._nav_stack.push(
            WeakKeyDictionary(
                {ax: (ax._get_view(),
                      # 存储原始和修改后的位置
                      (ax.get_position(True).frozen(),
                       ax.get_position().frozen()))
                 for ax in self.canvas.figure.axes}))
        self.set_history_buttons()

    def _update_view(self):
        """
        Update the viewlim and position from the view and position stack for
        each Axes.
        """
        nav_info = self._nav_stack()
        if nav_info is None:
            return
        # 一次性检索所有项，以避免在下面的循环中间发生 GC 删除 Axes 对象的风险
        items = list(nav_info.items())
        for ax, (view, (pos_orig, pos_active)) in items:
            ax._set_view(view)
            # 恢复原始和活动状态下的位置
            ax._set_position(pos_orig, 'original')
            ax._set_position(pos_active, 'active')
        self.canvas.draw_idle()
    def configure_subplots(self, *args):
        # 检查是否存在 subplot_tool 属性，如果存在，则显示当前图形的管理器并返回
        if hasattr(self, "subplot_tool"):
            self.subplot_tool.figure.canvas.manager.show()
            return
        
        # 由于存在循环导入，所以这里需要在此处进行导入
        from matplotlib.figure import Figure
        
        # 使用 mpl.rc_context 来配置参数，这里设置工具图形的工具栏为无
        with mpl.rc_context({"toolbar": "none"}):  # No navbar for the toolfig.
            # 创建一个新的图形管理器，并设置其大小为 (6, 3)
            manager = type(self.canvas).new_manager(Figure(figsize=(6, 3)), -1)
        
        # 设置管理器的窗口标题为 "Subplot configuration tool"
        manager.set_window_title("Subplot configuration tool")
        
        # 调整工具图形的子图布局，使得顶部留出一定空间
        tool_fig = manager.canvas.figure
        tool_fig.subplots_adjust(top=0.9)
        
        # 创建一个 SubplotTool 对象，连接到当前画布的图形上
        self.subplot_tool = widgets.SubplotTool(self.canvas.figure, tool_fig)
        
        # 绑定关闭事件处理函数，当工具图形关闭时销毁连接
        cid = self.canvas.mpl_connect("close_event", lambda e: manager.destroy())

        def on_tool_fig_close(e):
            # 断开与画布的连接，并删除 subplot_tool 属性
            self.canvas.mpl_disconnect(cid)
            del self.subplot_tool

        # 绑定工具图形的关闭事件处理函数
        tool_fig.canvas.mpl_connect("close_event", on_tool_fig_close)
        
        # 显示管理器
        manager.show()
        
        # 返回 subplot_tool 属性
        return self.subplot_tool

    def save_figure(self, *args):
        """Save the current figure."""
        # 保存当前图形，但是该方法未实现
        raise NotImplementedError

    def update(self):
        """Reset the Axes stack."""
        # 清空导航栈
        self._nav_stack.clear()
        # 设置历史按钮
        self.set_history_buttons()

    def set_history_buttons(self):
        """Enable or disable the back/forward button."""
        # 启用或禁用后退/前进按钮
class ToolContainerBase:
    """
    Base class for all tool containers, e.g. toolbars.

    Attributes
    ----------
    toolmanager : `.ToolManager`
        The tools with which this `ToolContainer` wants to communicate.
    """

    _icon_extension = '.png'
    """
    Toolcontainer button icon image format extension

    **String**: Image extension
    """

    def __init__(self, toolmanager):
        # 初始化方法，设置工具管理器实例变量
        self.toolmanager = toolmanager
        # 连接工具管理器的事件，当工具消息事件发生时调用 set_message 方法
        toolmanager.toolmanager_connect(
            'tool_message_event',
            lambda event: self.set_message(event.message))
        # 连接工具管理器的事件，当工具移除事件发生时调用 remove_toolitem 方法
        toolmanager.toolmanager_connect(
            'tool_removed_event',
            lambda event: self.remove_toolitem(event.tool.name))

    def _tool_toggled_cbk(self, event):
        """
        Capture the 'tool_trigger_[name]'

        This only gets used for toggled tools.
        """
        # 工具切换回调函数，根据事件中的工具名称调用 toggle_toolitem 方法
        self.toggle_toolitem(event.tool.name, event.tool.toggled)

    def add_tool(self, tool, group, position=-1):
        """
        Add a tool to this container.

        Parameters
        ----------
        tool : tool_like
            The tool to add, see `.ToolManager.get_tool`.
        group : str
            The name of the group to add this tool to.
        position : int, default: -1
            The position within the group to place this tool.
        """
        # 获取工具实例
        tool = self.toolmanager.get_tool(tool)
        # 获取工具对应的图像文件名
        image = self._get_image_filename(tool)
        # 检查工具是否具有切换状态，并设置 toggle 变量
        toggle = getattr(tool, 'toggled', None) is not None
        # 添加工具项到容器中，包括名称、分组、位置、图像、描述和切换状态
        self.add_toolitem(tool.name, group, position,
                          image, tool.description, toggle)
        # 如果工具具有切换状态，则连接工具切换事件，并根据初始状态调用 toggle_toolitem 方法
        if toggle:
            self.toolmanager.toolmanager_connect('tool_trigger_%s' % tool.name,
                                                 self._tool_toggled_cbk)
            # 如果工具初始为切换状态
            if tool.toggled:
                self.toggle_toolitem(tool.name, True)
    def _get_image_filename(self, tool):
        """Resolve a tool icon's filename."""
        # 如果工具没有设置图标，则返回 None
        if not tool.image:
            return None
        # 如果工具图标路径是绝对路径，则直接使用该路径
        if os.path.isabs(tool.image):
            filename = tool.image
        else:
            # 如果工具的图标路径不是绝对路径，则尝试根据其类型及其父类查找图标文件
            # 首先检查工具实例的字典中是否包含 'image' 属性
            if "image" in getattr(tool, "__dict__", {}):
                raise ValueError("If 'tool.image' is an instance variable, "
                                 "it must be an absolute path")
            # 遍历工具类的方法解析顺序（Method Resolution Order, MRO）
            for cls in type(tool).__mro__:
                # 查找类属性中是否定义了 'image' 属性
                if "image" in vars(cls):
                    try:
                        # 获取定义 'image' 属性的类的源文件路径
                        src = inspect.getfile(cls)
                        break
                    except (OSError, TypeError):
                        raise ValueError("Failed to locate source file "
                                         "where 'tool.image' is defined") from None
            else:
                # 如果未能找到定义 'tool.image' 的父类，则抛出异常
                raise ValueError("Failed to find parent class defining 'tool.image'")
            # 构建图标文件的完整路径
            filename = str(pathlib.Path(src).parent / tool.image)
        
        # 检查生成的文件路径及其加上扩展名后的路径是否存在，返回第一个存在的绝对路径
        for filename in [filename, filename + self._icon_extension]:
            if os.path.isfile(filename):
                return os.path.abspath(filename)
        
        # 如果以上路径都不存在，则尝试一组后备路径
        for fname in [
            tool.image,
            tool.image + self._icon_extension,
            cbook._get_data_path("images", tool.image),
            cbook._get_data_path("images", tool.image + self._icon_extension),
        ]:
            if os.path.isfile(fname):
                # 发出弃用警告，说明使用了已经弃用的路径解析方法
                _api.warn_deprecated(
                    "3.9", message=f"Loading icon {tool.image!r} from the current "
                    "directory or from Matplotlib's image directory.  This behavior "
                    "is deprecated since %(since)s and will be removed %(removal)s; "
                    "Tool.image should be set to a path relative to the Tool's source "
                    "file, or to an absolute path.")
                return os.path.abspath(fname)

    def trigger_tool(self, name):
        """
        Trigger the tool.

        Parameters
        ----------
        name : str
            Name (id) of the tool triggered from within the container.
        """
        # 调用工具管理器触发指定名称的工具
        self.toolmanager.trigger_tool(name, sender=self)
    # 定义一个方法用于向工具栏容器添加工具项
    def add_toolitem(self, name, group, position, image, description, toggle):
        """
        A hook to add a toolitem to the container.

        This hook must be implemented in each backend and contains the
        backend-specific code to add an element to the toolbar.

        .. warning::
            This is part of the backend implementation and should
            not be called by end-users.  They should instead call
            `.ToolContainerBase.add_tool`.

        The callback associated with the button click event
        must be *exactly* ``self.trigger_tool(name)``.

        Parameters
        ----------
        name : str
            Name of the tool to add, this gets used as the tool's ID and as the
            default label of the buttons.
        group : str
            Name of the group that this tool belongs to.
        position : int
            Position of the tool within its group, if -1 it goes at the end.
        image : str
            Filename of the image for the button or `None`.
        description : str
            Description of the tool, used for the tooltips.
        toggle : bool
            * `True` : The button is a toggle (change the pressed/unpressed
              state between consecutive clicks).
            * `False` : The button is a normal button (returns to unpressed
              state after release).
        """
        # 抛出未实现错误，具体实现由每个后端自行完成
        raise NotImplementedError

    # 定义一个方法用于在不触发事件的情况下切换工具项状态
    def toggle_toolitem(self, name, toggled):
        """
        A hook to toggle a toolitem without firing an event.

        This hook must be implemented in each backend and contains the
        backend-specific code to silently toggle a toolbar element.

        .. warning::
            This is part of the backend implementation and should
            not be called by end-users.  They should instead call
            `.ToolManager.trigger_tool` or `.ToolContainerBase.trigger_tool`
            (which are equivalent).

        Parameters
        ----------
        name : str
            Id of the tool to toggle.
        toggled : bool
            Whether to set this tool as toggled or not.
        """
        # 抛出未实现错误，具体实现由每个后端自行完成
        raise NotImplementedError
    # 抛出未实现错误，提示子类需要实现这个方法
    def remove_toolitem(self, name):
        """
        A hook to remove a toolitem from the container.

        This hook must be implemented in each backend and contains the
        backend-specific code to remove an element from the toolbar; it is
        called when `.ToolManager` emits a `tool_removed_event`.

        Because some tools are present only on the `.ToolManager` but not on
        the `ToolContainer`, this method must be a no-op when called on a tool
        absent from the container.

        .. warning::
            This is part of the backend implementation and should
            not be called by end-users.  They should instead call
            `.ToolManager.remove_tool`.

        Parameters
        ----------
        name : str
            Name of the tool to remove.
        """
        raise NotImplementedError


```    
    # 抛出未实现错误，提示子类需要实现这个方法
    def set_message(self, s):
        """
        Display a message on the toolbar.

        Parameters
        ----------
        s : str
            Message text.
        """
        raise NotImplementedError
class _Backend:
    # 一个后端可以通过以下模式来定义：
    #
    # @_Backend.export
    # class FooBackend(_Backend):
    #     # 重写下面文档中描述的属性和方法。

    # `backend_version` 可能会被子类覆盖。
    backend_version = "unknown"

    # `FigureCanvas` 类必须被定义。
    FigureCanvas = None

    # 对于交互式后端，必须重写 `FigureManager` 类。
    FigureManager = FigureManagerBase

    # 对于交互式后端，`mainloop` 应该是一个不带参数的函数，用于启动后端的主循环。
    # 对于非交互式后端，应该将其保留为 None。
    mainloop = None

    # 以下方法将被自动定义和导出，但可以被重写。

    @classmethod
    def new_figure_manager(cls, num, *args, **kwargs):
        """创建一个新的图形管理器实例。"""
        # 由于循环引用，此处需要进行导入。
        from matplotlib.figure import Figure
        fig_cls = kwargs.pop('FigureClass', Figure)
        fig = fig_cls(*args, **kwargs)
        return cls.new_figure_manager_given_figure(num, fig)

    @classmethod
    def new_figure_manager_given_figure(cls, num, figure):
        """为给定的图形创建一个新的图形管理器实例。"""
        return cls.FigureCanvas.new_manager(figure, num)

    @classmethod
    def draw_if_interactive(cls):
        manager_class = cls.FigureCanvas.manager_class
        # 交互式后端重新实现 start_main_loop 或 pyplot_show。
        backend_is_interactive = (
            manager_class.start_main_loop != FigureManagerBase.start_main_loop
            or manager_class.pyplot_show != FigureManagerBase.pyplot_show)
        if backend_is_interactive and is_interactive():
            manager = Gcf.get_active()
            if manager:
                manager.canvas.draw_idle()

    @classmethod
    # 定义一个静态方法，用于显示所有图形。
    def show(cls, *, block=None):
        """
        Show all figures.

        `show` blocks by calling `mainloop` if *block* is ``True``, or if it is
        ``None`` and we are not in `interactive` mode and if IPython's
        ``%matplotlib`` integration has not been activated.
        """
        # 获取所有图形管理器
        managers = Gcf.get_all_fig_managers()
        if not managers:
            return  # 如果没有图形管理器，则直接返回

        # 遍历每个图形管理器并显示图形
        for manager in managers:
            try:
                manager.show()  # 显示图形，对于非交互式后端会发出警告
            except NonGuiException as exc:
                _api.warn_external(str(exc))  # 如果出现 NonGuiException，警告外部系统

        # 如果没有主循环可用，则直接返回
        if cls.mainloop is None:
            return
        
        # 如果 block 为 None，则检查 IPython 的 %matplotlib 集成是否已激活
        if block is None:
            # IPython 的 activate_matplotlib (>= 0.10) 会在 pyplot.show 上添加 _needmain 属性
            pyplot_show = getattr(sys.modules.get("matplotlib.pyplot"), "show", None)
            ipython_pylab = hasattr(pyplot_show, "_needmain")
            # 如果不是 IPython 集成且不是交互模式，则 block 为 True
            block = not ipython_pylab and not is_interactive()

        # 如果 block 为 True，则执行主循环
        if block:
            cls.mainloop()

    # 此方法实际上是导出所需方法的方法。
    @staticmethod
    def export(cls):
        # 遍历要导出的方法列表，并将其设置到当前模块中
        for name in [
                "backend_version",
                "FigureCanvas",
                "FigureManager",
                "new_figure_manager",
                "new_figure_manager_given_figure",
                "draw_if_interactive",
                "show",
        ]:
            setattr(sys.modules[cls.__module__], name, getattr(cls, name))

        # 为了向后兼容性，生成一个名为 `Show` 的类，作为 `ShowBase` 的补充
        class Show(ShowBase):
            # 定义一个 mainloop 方法，返回 cls.mainloop() 的结果
            def mainloop(self):
                return cls.mainloop()

        # 将新定义的 `Show` 类设置到当前模块中
        setattr(sys.modules[cls.__module__], "Show", Show)
        return cls
class ShowBase(_Backend):
    """
    Simple base class to generate a ``show()`` function in backends.

    Subclass must override ``mainloop()`` method.
    """

    # 定义类的调用运算符重载，使得实例可以像函数一样调用
    def __call__(self, block=None):
        # 调用实例的 show 方法，并返回其结果
        return self.show(block=block)
```