# `D:\src\scipysrc\matplotlib\lib\matplotlib\markers.py`

```
r"""
Functions to handle markers; used by the marker functionality of
`~matplotlib.axes.Axes.plot`, `~matplotlib.axes.Axes.scatter`, and
`~matplotlib.axes.Axes.errorbar`.

All possible markers are defined here:

============================== ====== =========================================
marker                         symbol description
============================== ====== =========================================
``"."``                        |m00|  point
``","``                        |m01|  pixel
``"o"``                        |m02|  circle
``"v"``                        |m03|  triangle_down
``"^"``                        |m04|  triangle_up
``"<"``                        |m05|  triangle_left
``">"``                        |m06|  triangle_right
``"1"``                        |m07|  tri_down
``"2"``                        |m08|  tri_up
``"3"``                        |m09|  tri_left
``"4"``                        |m10|  tri_right
``"8"``                        |m11|  octagon
``"s"``                        |m12|  square
``"p"``                        |m13|  pentagon
``"P"``                        |m23|  plus (filled)
``"*"``                        |m14|  star
``"h"``                        |m15|  hexagon1
``"H"``                        |m16|  hexagon2
``"+"``                        |m17|  plus
``"x"``                        |m18|  x
``"X"``                        |m24|  x (filled)
``"D"``                        |m19|  diamond
``"d"``                        |m20|  thin_diamond
``"|"``                        |m21|  vline
``"_"``                        |m22|  hline
``0`` (``TICKLEFT``)           |m25|  tickleft
``1`` (``TICKRIGHT``)          |m26|  tickright
``2`` (``TICKUP``)             |m27|  tickup
``3`` (``TICKDOWN``)           |m28|  tickdown
``4`` (``CARETLEFT``)          |m29|  caretleft
``5`` (``CARETRIGHT``)         |m30|  caretright
``6`` (``CARETUP``)            |m31|  caretup
``7`` (``CARETDOWN``)          |m32|  caretdown
``8`` (``CARETLEFTBASE``)      |m33|  caretleft (centered at base)
``9`` (``CARETRIGHTBASE``)     |m34|  caretright (centered at base)
``10`` (``CARETUPBASE``)       |m35|  caretup (centered at base)
``11`` (``CARETDOWNBASE``)     |m36|  caretdown (centered at base)
``"none"`` or ``"None"``              nothing
``" "`` or  ``""``                    nothing
``"$...$"``                    |m37|  Render the string using mathtext.
                                      E.g ``"$f$"`` for marker showing the
                                      letter ``f``.
``verts``                             A list of (x, y) pairs used for Path
                                      vertices. The center of the marker is
                                      located at (0, 0) and the size is
                                      normalized, such that the created path
                                      is encapsulated inside the unit cell.
``path``                              A `~matplotlib.path.Path` instance.

"""
(numsides, 0, angle)              # 创建一个有 numsides 条边的正多边形，旋转角度为 angle。
(numsides, 1, angle)              # 创建一个星形符号，具有 numsides 条边，旋转角度为 angle。
(numsides, 2, angle)              # 创建一个星号符号，具有 numsides 条边，旋转角度为 angle。
============================== ====== =========================================

Note that special symbols can be defined via the
:ref:`STIX math font <mathtext>`,
e.g. ``"$\u266B$"``. For an overview over the STIX font symbols refer to the
`STIX font table <http://www.stixfonts.org/allGlyphs.html>`_.
Also see the :doc:`/gallery/text_labels_and_annotations/stix_fonts_demo`.

Integer numbers from ``0`` to ``11`` create lines and triangles. Those are
equally accessible via capitalized variables, like ``CARETDOWNBASE``.
Hence the following are equivalent::

    plt.plot([1, 2, 3], marker=11)
    plt.plot([1, 2, 3], marker=matplotlib.markers.CARETDOWNBASE)

Markers join and cap styles can be customized by creating a new instance of
MarkerStyle.
A MarkerStyle can also have a custom `~matplotlib.transforms.Transform`
allowing it to be arbitrarily rotated or offset.

Examples showing the use of markers:

* :doc:`/gallery/lines_bars_and_markers/marker_reference`
* :doc:`/gallery/lines_bars_and_markers/scatter_star_poly`
* :doc:`/gallery/lines_bars_and_markers/multivariate_marker_plot`

.. |m00| image:: /_static/markers/m00.png
.. |m01| image:: /_static/markers/m01.png
.. |m02| image:: /_static/markers/m02.png
.. |m03| image:: /_static/markers/m03.png
.. |m04| image:: /_static/markers/m04.png
.. |m05| image:: /_static/markers/m05.png
.. |m06| image:: /_static/markers/m06.png
.. |m07| image:: /_static/markers/m07.png
.. |m08| image:: /_static/markers/m08.png
.. |m09| image:: /_static/markers/m09.png
.. |m10| image:: /_static/markers/m10.png
.. |m11| image:: /_static/markers/m11.png
.. |m12| image:: /_static/markers/m12.png
.. |m13| image:: /_static/markers/m13.png
.. |m14| image:: /_static/markers/m14.png
.. |m15| image:: /_static/markers/m15.png
.. |m16| image:: /_static/markers/m16.png
.. |m17| image:: /_static/markers/m17.png
.. |m18| image:: /_static/markers/m18.png
.. |m19| image:: /_static/markers/m19.png
.. |m20| image:: /_static/markers/m20.png
.. |m21| image:: /_static/markers/m21.png
.. |m22| image:: /_static/markers/m22.png
.. |m23| image:: /_static/markers/m23.png
.. |m24| image:: /_static/markers/m24.png
.. |m25| image:: /_static/markers/m25.png
.. |m26| image:: /_static/markers/m26.png
.. |m27| image:: /_static/markers/m27.png
.. |m28| image:: /_static/markers/m28.png
.. |m29| image:: /_static/markers/m29.png
.. |m30| image:: /_static/markers/m30.png
.. |m31| image:: /_static/markers/m31.png
.. |m32| image:: /_static/markers/m32.png
.. |m33| image:: /_static/markers/m33.png
.. |m34| image:: /_static/markers/m34.png
# 导入必要的模块和库
import copy  # 导入copy模块，用于复制对象

from collections.abc import Sized  # 从collections.abc模块导入Sized类

import numpy as np  # 导入NumPy库，并重命名为np

import matplotlib as mpl  # 导入Matplotlib库，并重命名为mpl
from . import _api, cbook  # 从当前包中导入_api和cbook模块
from .path import Path  # 从当前包中导入Path类
from .transforms import IdentityTransform, Affine2D  # 从当前包中导入IdentityTransform和Affine2D类
from ._enums import JoinStyle, CapStyle  # 从当前包中导入JoinStyle和CapStyle枚举类型

# 特殊用途的标记符号:
(TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN,
 CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN,
 CARETLEFTBASE, CARETRIGHTBASE, CARETUPBASE, CARETDOWNBASE) = range(12)

_empty_path = Path(np.empty((0, 2)))  # 创建一个空路径对象

class MarkerStyle:
    """
    A class representing marker types.

    Instances are immutable. If you need to change anything, create a new
    instance.

    Attributes
    ----------
    markers : dict
        All known markers.
    filled_markers : tuple
        All known filled markers. This is a subset of *markers*.
    fillstyles : tuple
        The supported fillstyles.
    """

    markers = {
        '.': 'point',
        ',': 'pixel',
        'o': 'circle',
        'v': 'triangle_down',
        '^': 'triangle_up',
        '<': 'triangle_left',
        '>': 'triangle_right',
        '1': 'tri_down',
        '2': 'tri_up',
        '3': 'tri_left',
        '4': 'tri_right',
        '8': 'octagon',
        's': 'square',
        'p': 'pentagon',
        '*': 'star',
        'h': 'hexagon1',
        'H': 'hexagon2',
        '+': 'plus',
        'x': 'x',
        'D': 'diamond',
        'd': 'thin_diamond',
        '|': 'vline',
        '_': 'hline',
        'P': 'plus_filled',
        'X': 'x_filled',
        TICKLEFT: 'tickleft',
        TICKRIGHT: 'tickright',
        TICKUP: 'tickup',
        TICKDOWN: 'tickdown',
        CARETLEFT: 'caretleft',
        CARETRIGHT: 'caretright',
        CARETUP: 'caretup',
        CARETDOWN: 'caretdown',
        CARETLEFTBASE: 'caretleftbase',
        CARETRIGHTBASE: 'caretrightbase',
        CARETUPBASE: 'caretupbase',
        CARETDOWNBASE: 'caretdownbase',
        "None": 'nothing',
        "none": 'nothing',
        ' ': 'nothing',
        '': 'nothing'
    }
    
    # 用于信息目的。is_filled()
    # 在_set_*函数中计算。
    filled_markers = (
        '.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd',
        'P', 'X')  # 包含所有已知的填充标记的元组

    fillstyles = ('full', 'left', 'right', 'bottom', 'top', 'none')  # 支持的填充样式的元组
    _half_fillstyles = ('left', 'right', 'bottom', 'top')  # 部分填充样式的元组
    def __init__(self, marker,
                 fillstyle=None, transform=None, capstyle=None, joinstyle=None):
        """
        Parameters
        ----------
        marker : str, array-like, Path, MarkerStyle
            - Another instance of `MarkerStyle` copies the details of that *marker*.
            - For other possible marker values, see the module docstring
              `matplotlib.markers`.

        fillstyle : str, default: :rc:`markers.fillstyle`
            One of 'full', 'left', 'right', 'bottom', 'top', 'none'.

        transform : `~matplotlib.transforms.Transform`, optional
            Transform that will be combined with the native transform of the
            marker.

        capstyle : `.CapStyle` or %(CapStyle)s, optional
            Cap style that will override the default cap style of the marker.

        joinstyle : `.JoinStyle` or %(JoinStyle)s, optional
            Join style that will override the default join style of the marker.
        """
        # 初始化函数，设置各种样式和属性
        self._marker_function = None  # 初始化标记函数为 None
        self._user_transform = transform  # 用户自定义的变换
        self._user_capstyle = CapStyle(capstyle) if capstyle is not None else None  # 用户自定义的端点样式
        self._user_joinstyle = JoinStyle(joinstyle) if joinstyle is not None else None  # 用户自定义的连接样式
        self._set_fillstyle(fillstyle)  # 设置填充样式
        self._set_marker(marker)  # 设置标记样式

    def _recache(self):
        # 如果标记函数为空，直接返回
        if self._marker_function is None:
            return
        self._path = _empty_path  # 路径为空路径
        self._transform = IdentityTransform()  # 变换为单位变换
        self._alt_path = None  # 备用路径为空
        self._alt_transform = None  # 备用变换为空
        self._snap_threshold = None  # 捕捉阈值为空
        self._joinstyle = JoinStyle.round  # 默认连接样式为圆形
        self._capstyle = self._user_capstyle or CapStyle.butt  # 端点样式为用户自定义的或者默认的矩形
        # 初始假设：假设标记是填充的，除非填充样式设置为 'none'。标记函数将根据实际情况覆盖这一设置。
        self._filled = self._fillstyle != 'none'  # 填充状态根据填充样式决定是否填充
        self._marker_function()  # 调用标记函数进行进一步设置

    def __bool__(self):
        # 返回路径顶点数是否为真
        return bool(len(self._path.vertices))

    def is_filled(self):
        # 返回填充状态
        return self._filled

    def get_fillstyle(self):
        # 返回填充样式
        return self._fillstyle

    def _set_fillstyle(self, fillstyle):
        """
        Set the fillstyle.

        Parameters
        ----------
        fillstyle : {'full', 'left', 'right', 'bottom', 'top', 'none'}
            The part of the marker surface that is colored with
            markerfacecolor.
        """
        # 设置填充样式
        if fillstyle is None:
            fillstyle = mpl.rcParams['markers.fillstyle']  # 如果未指定填充样式，使用默认配置
        _api.check_in_list(self.fillstyles, fillstyle=fillstyle)  # 检查填充样式是否有效
        self._fillstyle = fillstyle  # 设置填充样式

    def get_joinstyle(self):
        # 返回连接样式的名称
        return self._joinstyle.name

    def get_capstyle(self):
        # 返回端点样式的名称
        return self._capstyle.name

    def get_marker(self):
        # 返回标记样式
        return self._marker
    def _set_marker(self, marker):
        """
        Set the marker.

        Parameters
        ----------
        marker : str, array-like, Path, MarkerStyle
            - Another instance of `MarkerStyle` copies the details of that *marker*.
            - For other possible marker values see the module docstring
              `matplotlib.markers`.
        """
        # 如果 marker 是字符串并且是数学文本，则使用 _set_mathtext_path 方法设置标记函数
        if isinstance(marker, str) and cbook.is_math_text(marker):
            self._marker_function = self._set_mathtext_path
        # 如果 marker 是整数或者字符串并且在已知的标记集合中，则根据标记设置相应的标记函数
        elif isinstance(marker, (int, str)) and marker in self.markers:
            self._marker_function = getattr(self, '_set_' + self.markers[marker])
        # 如果 marker 是二维数组，则使用 _set_vertices 方法设置标记函数
        elif (isinstance(marker, np.ndarray) and marker.ndim == 2 and
                marker.shape[1] == 2):
            self._marker_function = self._set_vertices
        # 如果 marker 是 Path 对象，则使用 _set_path_marker 方法设置标记函数
        elif isinstance(marker, Path):
            self._marker_function = self._set_path_marker
        # 如果 marker 是长度为 2 或 3 的可迭代对象且第二个元素在 (0, 1, 2) 中，则使用 _set_tuple_marker 方法设置标记函数
        elif (isinstance(marker, Sized) and len(marker) in (2, 3) and
                marker[1] in (0, 1, 2)):
            self._marker_function = self._set_tuple_marker
        # 如果 marker 是 MarkerStyle 对象，则深拷贝其属性到当前对象
        elif isinstance(marker, MarkerStyle):
            self.__dict__ = copy.deepcopy(marker.__dict__)
        else:
            try:
                # 尝试将 marker 转换为 Path 对象，如果失败则抛出 ValueError
                Path(marker)
                self._marker_function = self._set_vertices
            except ValueError as err:
                # 如果无法识别 marker 的样式，则抛出 ValueError 异常
                raise ValueError(
                    f'Unrecognized marker style {marker!r}') from err

        # 如果 marker 不是 MarkerStyle 对象，则设置当前对象的 marker 属性并重新计算相关缓存
        if not isinstance(marker, MarkerStyle):
            self._marker = marker
            self._recache()

    def get_path(self):
        """
        Return a `.Path` for the primary part of the marker.

        For unfilled markers this is the whole marker, for filled markers,
        this is the area to be drawn with *markerfacecolor*.
        """
        # 返回主要部分的路径对象 _path
        return self._path

    def get_transform(self):
        """
        Return the transform to be applied to the `.Path` from
        `MarkerStyle.get_path()`.
        """
        # 返回应用于 _path 的转换
        if self._user_transform is None:
            return self._transform.frozen()
        else:
            return (self._transform + self._user_transform).frozen()

    def get_alt_path(self):
        """
        Return a `.Path` for the alternate part of the marker.

        For unfilled markers, this is *None*; for filled markers, this is the
        area to be drawn with *markerfacecoloralt*.
        """
        # 返回替代部分的路径对象 _alt_path
        return self._alt_path

    def get_alt_transform(self):
        """
        Return the transform to be applied to the `.Path` from
        `MarkerStyle.get_alt_path()`.
        """
        # 返回应用于 _alt_path 的转换
        if self._user_transform is None:
            return self._alt_transform.frozen()
        else:
            return (self._alt_transform + self._user_transform).frozen()

    def get_snap_threshold(self):
        # 返回 snap 阈值 _snap_threshold
        return self._snap_threshold
    # 返回用户提供的标记变换的一部分。
    def get_user_transform(self):
        if self._user_transform is not None:
            return self._user_transform.frozen()

    # 返回应用了给定变换的新标记版本。
    #
    # Parameters
    # ----------
    # transform : `~matplotlib.transforms.Affine2D`
    #     将与当前用户提供的变换组合的变换。
    def transformed(self, transform):
        new_marker = MarkerStyle(self)
        if new_marker._user_transform is not None:
            new_marker._user_transform += transform
        else:
            new_marker._user_transform = transform
        return new_marker

    # 返回指定角度旋转的新标记版本。
    #
    # Parameters
    # ----------
    # deg : float, optional
    #     旋转角度（单位：度）。
    #
    # rad : float, optional
    #     旋转角度（单位：弧度）。
    #
    # Raises
    # ------
    # ValueError
    #     如果未指定 deg 或 rad，或者同时指定了 deg 和 rad。
    def rotated(self, *, deg=None, rad=None):
        if deg is None and rad is None:
            raise ValueError('One of deg or rad is required')
        if deg is not None and rad is not None:
            raise ValueError('Only one of deg and rad can be supplied')
        new_marker = MarkerStyle(self)
        if new_marker._user_transform is None:
            new_marker._user_transform = Affine2D()

        if deg is not None:
            new_marker._user_transform.rotate_deg(deg)
        if rad is not None:
            new_marker._user_transform.rotate(rad)

        return new_marker

    # 返回按指定比例因子缩放的新标记。
    #
    # Parameters
    # ----------
    # sx : float
    #     X 方向的缩放因子。
    #
    # sy : float, optional
    #     Y 方向的缩放因子。
    def scaled(self, sx, sy=None):
        if sy is None:
            sy = sx

        new_marker = MarkerStyle(self)
        _transform = new_marker._user_transform or Affine2D()
        new_marker._user_transform = _transform.scale(sx, sy)
        return new_marker

    # 将 _filled 设置为 False。
    def _set_nothing(self):
        self._filled = False

    # 设置自定义标记的变换，并根据路径调整比例。
    #
    # Parameters
    # ----------
    # path : Path
    #     自定义标记的路径。
    def _set_custom_marker(self, path):
        rescale = np.max(np.abs(path.vertices))  # x 和 y 的最大值。
        self._transform = Affine2D().scale(0.5 / rescale)
        self._path = path

    # 设置路径标记的自定义标记。
    def _set_path_marker(self):
        self._set_custom_marker(self._marker)

    # 设置顶点标记的自定义标记。
    def _set_vertices(self):
        self._set_custom_marker(Path(self._marker))
    # 设置元组标记的形状和样式
    def _set_tuple_marker(self):
        # 获取标记
        marker = self._marker
        # 如果标记长度为2，设置边数和旋转角度为默认值
        if len(marker) == 2:
            numsides, rotation = marker[0], 0.0
        # 如果标记长度为3，设置边数和旋转角度为标记中的值
        elif len(marker) == 3:
            numsides, rotation = marker[0], marker[2]
        # 获取样式类型
        symstyle = marker[1]
        # 根据样式类型设置路径对象和连接样式
        if symstyle == 0:
            self._path = Path.unit_regular_polygon(numsides)
            self._joinstyle = self._user_joinstyle or JoinStyle.miter
        elif symstyle == 1:
            self._path = Path.unit_regular_star(numsides)
            self._joinstyle = self._user_joinstyle or JoinStyle.bevel
        elif symstyle == 2:
            self._path = Path.unit_regular_asterisk(numsides)
            self._filled = False
            self._joinstyle = self._user_joinstyle or JoinStyle.bevel
        else:
            # 抛出异常，如果样式类型不在预期的范围内
            raise ValueError(f"Unexpected tuple marker: {marker}")
        # 设置变换矩阵，缩放并旋转路径对象
        self._transform = Affine2D().scale(0.5).rotate_deg(rotation)

    # 设置数学文本路径对象
    def _set_mathtext_path(self):
        """
        使用 `.TextPath` 对象绘制数学文本标记 '$...$'。
        
        由 tcb 提交
        """
        from matplotlib.text import TextPath
        
        # 创建文本路径对象，使用 LaTeX 渲染标记文本
        text = TextPath(xy=(0, 0), s=self.get_marker(),
                        usetex=mpl.rcParams['text.usetex'])
        # 如果文本路径的顶点数为0，返回
        if len(text.vertices) == 0:
            return
        
        # 获取文本路径对象的边界框
        bbox = text.get_extents()
        # 计算边界框的最大维度
        max_dim = max(bbox.width, bbox.height)
        # 设置变换矩阵，平移和缩放文本路径对象
        self._transform = (
            Affine2D()
            .translate(-bbox.xmin + 0.5 * -bbox.width, -bbox.ymin + 0.5 * -bbox.height)
            .scale(1.0 / max_dim))
        # 设置路径对象为文本路径
        self._path = text
        # 禁用路径对象的捕捉功能
        self._snap = False

    # 检查是否填充了一半
    def _half_fill(self):
        return self.get_fillstyle() in self._half_fillstyles

    # 设置圆形路径对象
    def _set_circle(self, size=1.0):
        # 设置缩放变换矩阵
        self._transform = Affine2D().scale(0.5 * size)
        # 设置捕捉阈值为无穷大
        self._snap_threshold = np.inf
        # 如果不是半填充状态，设置路径对象为单位圆
        if not self._half_fill():
            self._path = Path.unit_circle()
        else:
            # 如果是半填充状态，设置路径对象为右半单位圆，并设置备用路径对象和变换矩阵
            self._path = self._alt_path = Path.unit_circle_righthalf()
            fs = self.get_fillstyle()
            self._transform.rotate_deg(
                {'right': 0, 'top': 90, 'left': 180, 'bottom': 270}[fs])
            self._alt_transform = self._transform.frozen().rotate_deg(180.)

    # 设置点路径对象
    def _set_point(self):
        # 调用设置圆形路径对象，设置尺寸为0.5
        self._set_circle(size=0.5)
    def _set_pixel(self):
        self._path = Path.unit_rectangle()
        # Ideally, you'd want -0.5, -0.5 here, but then the snapping
        # algorithm in the Agg backend will round this to a 2x2
        # rectangle from (-1, -1) to (1, 1).  By offsetting it
        # slightly, we can force it to be (0, 0) to (1, 1), which both
        # makes it only be a single pixel and places it correctly
        # aligned to 1-width stroking (i.e. the ticks).  This hack is
        # the best of a number of bad alternatives, mainly because the
        # backends are not aware of what marker is actually being used
        # beyond just its path data.
        # 设置路径为单位矩形路径，用于表示一个像素大小的图形
        self._transform = Affine2D().translate(-0.49999, -0.49999)
        # 设置变换矩阵，通过微小偏移确保图形边界在(0, 0)到(1, 1)之间，以便正确对齐到1像素宽度的绘制线条
        self._snap_threshold = None
        # 设定对齐阈值为None，即无需对齐阈值

    _triangle_path = Path._create_closed([[0, 1], [-1, -1], [1, -1]])
    # 创建一个闭合路径表示一个等边三角形，顶点(0, 1)，左下角(-1, -1)，右下角(1, -1)
    # Going down halfway looks to small.  Golden ratio is too far.
    _triangle_path_u = Path._create_closed([[0, 1], [-3/5, -1/5], [3/5, -1/5]])
    # 创建一个闭合路径表示一个朝上的等边三角形，顶点(0, 1)，左下角(-3/5, -1/5)，右下角(3/5, -1/5)
    _triangle_path_d = Path._create_closed(
        [[-3/5, -1/5], [3/5, -1/5], [1, -1], [-1, -1]])
    # 创建一个闭合路径表示一个朝下的等边三角形，左下角(-3/5, -1/5)，右下角(3/5, -1/5)，右上角(1, -1)，左上角(-1, -1)
    _triangle_path_l = Path._create_closed([[0, 1], [0, -1], [-1, -1]])
    # 创建一个闭合路径表示一个朝左的等边三角形，顶点(0, 1)，底边从左到右(-1, -1)到(0, -1)
    _triangle_path_r = Path._create_closed([[0, 1], [0, -1], [1, -1]])
    # 创建一个闭合路径表示一个朝右的等边三角形，顶点(0, 1)，底边从左到右(0, -1)到(1, -1)

    def _set_triangle(self, rot, skip):
        self._transform = Affine2D().scale(0.5).rotate_deg(rot)
        # 设置变换矩阵，先缩放0.5倍，再按照给定的角度rot进行旋转
        self._snap_threshold = 5.0
        # 设定对齐阈值为5.0

        if not self._half_fill():
            self._path = self._triangle_path
            # 如果不需要半填充，则使用默认的等边三角形路径
        else:
            mpaths = [self._triangle_path_u,
                      self._triangle_path_l,
                      self._triangle_path_d,
                      self._triangle_path_r]

            fs = self.get_fillstyle()
            if fs == 'top':
                self._path = mpaths[(0 + skip) % 4]
                self._alt_path = mpaths[(2 + skip) % 4]
            elif fs == 'bottom':
                self._path = mpaths[(2 + skip) % 4]
                self._alt_path = mpaths[(0 + skip) % 4]
            elif fs == 'left':
                self._path = mpaths[(1 + skip) % 4]
                self._alt_path = mpaths[(3 + skip) % 4]
            else:
                self._path = mpaths[(3 + skip) % 4]
                self._alt_path = mpaths[(1 + skip) % 4]

            self._alt_transform = self._transform
            # 根据填充风格fs选择主要路径和备用路径，并设置备用变换矩阵

        self._joinstyle = self._user_joinstyle or JoinStyle.miter
        # 设置连接风格为用户定义的连接风格或默认的尖角连接
    # 设置形状为正方形
    def _set_square(self):
        # 创建一个平移了 (-0.5, -0.5) 的仿射变换对象
        self._transform = Affine2D().translate(-0.5, -0.5)
        # 设置捕捉阈值为 2.0
        self._snap_threshold = 2.0
        # 如果不是半填充状态，则使用单位矩形作为路径
        if not self._half_fill():
            self._path = Path.unit_rectangle()
        else:
            # 构建一个底部填充的正方形，使用两个矩形组成，其中一个被填充
            self._path = Path([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5],
                               [0.0, 0.5], [0.0, 0.0]])
            self._alt_path = Path([[0.0, 0.5], [1.0, 0.5], [1.0, 1.0],
                                   [0.0, 1.0], [0.0, 0.5]])
            # 根据填充样式确定旋转角度
            fs = self.get_fillstyle()
            rotate = {'bottom': 0, 'right': 90, 'top': 180, 'left': 270}[fs]
            self._transform.rotate_deg(rotate)
            self._alt_transform = self._transform

        # 设置连接样式为用户指定的连接样式或默认的尖角连接
        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    # 设置形状为菱形
    def _set_diamond(self):
        # 创建一个平移了 (-0.5, -0.5) 并旋转了 45 度的仿射变换对象
        self._transform = Affine2D().translate(-0.5, -0.5).rotate_deg(45)
        # 设置捕捉阈值为 5.0
        self._snap_threshold = 5.0
        # 如果不是半填充状态，则使用单位矩形作为路径
        if not self._half_fill():
            self._path = Path.unit_rectangle()
        else:
            # 使用菱形的顶点设置路径
            self._path = Path([[0, 0], [1, 0], [1, 1], [0, 0]])
            self._alt_path = Path([[0, 0], [0, 1], [1, 1], [0, 0]])
            # 根据填充样式确定旋转角度
            fs = self.get_fillstyle()
            rotate = {'right': 0, 'top': 90, 'left': 180, 'bottom': 270}[fs]
            self._transform.rotate_deg(rotate)
            self._alt_transform = self._transform
        # 设置连接样式为用户指定的连接样式或默认的尖角连接
        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    # 设置形状为细菱形
    def _set_thin_diamond(self):
        # 先调用 _set_diamond() 方法设置形状为菱形
        self._set_diamond()
        # 在当前变换基础上进行缩放，使得形状变为细菱形
        self._transform.scale(0.6, 1.0)

    # 设置形状为五边形
    def _set_pentagon(self):
        # 创建一个缩放了 0.5 倍的仿射变换对象
        self._transform = Affine2D().scale(0.5)
        # 设置捕捉阈值为 5.0
        self._snap_threshold = 5.0

        # 生成一个单位正五边形的路径对象
        polypath = Path.unit_regular_polygon(5)

        # 如果不是半填充状态，则使用正五边形作为路径
        if not self._half_fill():
            self._path = polypath
        else:
            # 根据填充样式选择不同的填充方式
            verts = polypath.vertices
            y = (1 + np.sqrt(5)) / 4.
            top = Path(verts[[0, 1, 4, 0]])
            bottom = Path(verts[[1, 2, 3, 4, 1]])
            left = Path([verts[0], verts[1], verts[2], [0, -y], verts[0]])
            right = Path([verts[0], verts[4], verts[3], [0, -y], verts[0]])
            self._path, self._alt_path = {
                'top': (top, bottom), 'bottom': (bottom, top),
                'left': (left, right), 'right': (right, left),
            }[self.get_fillstyle()]
            self._alt_transform = self._transform

        # 设置连接样式为用户指定的连接样式或默认的尖角连接
        self._joinstyle = self._user_joinstyle or JoinStyle.miter
    # 设置星形路径的变换和吸附阈值
    def _set_star(self):
        # 使用 Affine2D 对象创建缩放比例为 0.5的仿射变换
        self._transform = Affine2D().scale(0.5)
        # 设置吸附阈值为 5.0
        self._snap_threshold = 5.0

        # 创建一个具有内径比为 0.381966的正规五角星形路径
        polypath = Path.unit_regular_star(5, innerCircle=0.381966)

        # 如果不是半填充状态，则使用完整的星形路径
        if not self._half_fill():
            self._path = polypath
        else:
            # 如果是半填充状态，则根据填充样式选择主路径和备选路径
            verts = polypath.vertices
            top = Path(np.concatenate([verts[0:4], verts[7:10], verts[0:1]]))
            bottom = Path(np.concatenate([verts[3:8], verts[3:4]]))
            left = Path(np.concatenate([verts[0:6], verts[0:1]]))
            right = Path(np.concatenate([verts[0:1], verts[5:10], verts[0:1]]))
            self._path, self._alt_path = {
                'top': (top, bottom), 'bottom': (bottom, top),
                'left': (left, right), 'right': (right, left),
            }[self.get_fillstyle()]  # 根据填充样式选择主路径和备选路径
            self._alt_transform = self._transform

        # 设置连接样式为用户指定的连接样式或默认的斜角连接
        self._joinstyle = self._user_joinstyle or JoinStyle.bevel

    # 设置第一个六边形路径的变换和吸附阈值
    def _set_hexagon1(self):
        # 使用 Affine2D 对象创建缩放比例为 0.5的仿射变换
        self._transform = Affine2D().scale(0.5)
        # 清空吸附阈值
        self._snap_threshold = None

        # 创建一个正规六边形路径
        polypath = Path.unit_regular_polygon(6)

        # 如果不是半填充状态，则使用完整的六边形路径
        if not self._half_fill():
            self._path = polypath
        else:
            # 如果是半填充状态，则根据填充样式选择主路径和备选路径
            verts = polypath.vertices
            # 在内部不绘制线条
            x = np.abs(np.cos(5 * np.pi / 6.))
            top = Path(np.concatenate([[(-x, 0)], verts[[1, 0, 5]], [(x, 0)]]))
            bottom = Path(np.concatenate([[(-x, 0)], verts[2:5], [(x, 0)]]))
            left = Path(verts[0:4])
            right = Path(verts[[0, 5, 4, 3]])
            self._path, self._alt_path = {
                'top': (top, bottom), 'bottom': (bottom, top),
                'left': (left, right), 'right': (right, left),
            }[self.get_fillstyle()]  # 根据填充样式选择主路径和备选路径
            self._alt_transform = self._transform

        # 设置连接样式为用户指定的连接样式或默认的直角连接
        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    # 设置第二个六边形路径的变换和吸附阈值
    def _set_hexagon2(self):
        # 使用 Affine2D 对象创建缩放比例为 0.5，并旋转30度的仿射变换
        self._transform = Affine2D().scale(0.5).rotate_deg(30)
        # 清空吸附阈值
        self._snap_threshold = None

        # 创建一个正规六边形路径
        polypath = Path.unit_regular_polygon(6)

        # 如果不是半填充状态，则使用完整的六边形路径
        if not self._half_fill():
            self._path = polypath
        else:
            # 如果是半填充状态，则根据填充样式选择主路径和备选路径
            verts = polypath.vertices
            # 在内部不绘制线条
            x, y = np.sqrt(3) / 4, 3 / 4.
            top = Path(verts[[1, 0, 5, 4, 1]])
            bottom = Path(verts[1:5])
            left = Path(np.concatenate([
                [(x, y)], verts[:3], [(-x, -y), (x, y)]]))
            right = Path(np.concatenate([
                [(x, y)], verts[5:2:-1], [(-x, -y)]]))
            self._path, self._alt_path = {
                'top': (top, bottom), 'bottom': (bottom, top),
                'left': (left, right), 'right': (right, left),
            }[self.get_fillstyle()]  # 根据填充样式选择主路径和备选路径
            self._alt_transform = self._transform

        # 设置连接样式为用户指定的连接样式或默认的直角连接
        self._joinstyle = self._user_joinstyle or JoinStyle.miter
    # 设置八边形的形状变换
    def _set_octagon(self):
        # 设置仿射变换，将图形缩放为原来的一半
        self._transform = Affine2D().scale(0.5)
        # 设置吸附阈值为5.0
        self._snap_threshold = 5.0

        # 创建一个八边形路径对象
        polypath = Path.unit_regular_polygon(8)

        # 如果不是半填充状态
        if not self._half_fill():
            # 将图形旋转22.5度
            self._transform.rotate_deg(22.5)
            # 设置路径为八边形路径
            self._path = polypath
        else:
            # 计算特定的坐标值
            x = np.sqrt(2.) / 4.
            # 设置路径为备用路径，并保存为备用路径
            self._path = self._alt_path = Path(
                [[0, -1], [0, 1], [-x, 1], [-1, x],
                 [-1, -x], [-x, -1], [0, -1]])
            # 获取填充样式
            fs = self.get_fillstyle()
            # 根据填充样式旋转图形
            self._transform.rotate_deg(
                {'left': 0, 'bottom': 90, 'right': 180, 'top': 270}[fs])
            # 设置备用变换为当前变换的冻结状态，旋转180度
            self._alt_transform = self._transform.frozen().rotate_deg(180.0)

        # 设置连接样式为用户指定的连接样式或默认的斜接连接
        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    # 直线标记路径对象
    _line_marker_path = Path([[0.0, -1.0], [0.0, 1.0]])

    # 设置垂直线形状
    def _set_vline(self):
        # 设置仿射变换，将图形缩放为原来的一半
        self._transform = Affine2D().scale(0.5)
        # 设置吸附阈值为1.0
        self._snap_threshold = 1.0
        # 设置填充状态为False
        self._filled = False
        # 设置路径为直线标记路径对象
        self._path = self._line_marker_path

    # 设置水平线形状
    def _set_hline(self):
        # 调用设置垂直线形状的方法
        self._set_vline()
        # 将当前变换旋转90度
        self._transform = self._transform.rotate_deg(90)

    # 水平刻度路径对象
    _tickhoriz_path = Path([[0.0, 0.0], [1.0, 0.0]])

    # 设置左侧刻度
    def _set_tickleft(self):
        # 设置仿射变换，将图形横向翻转
        self._transform = Affine2D().scale(-1.0, 1.0)
        # 设置吸附阈值为1.0
        self._snap_threshold = 1.0
        # 设置填充状态为False
        self._filled = False
        # 设置路径为水平刻度路径对象
        self._path = self._tickhoriz_path

    # 设置右侧刻度
    def _set_tickright(self):
        # 设置仿射变换，将图形按原样放置
        self._transform = Affine2D().scale(1.0, 1.0)
        # 设置吸附阈值为1.0
        self._snap_threshold = 1.0
        # 设置填充状态为False
        self._filled = False
        # 设置路径为水平刻度路径对象
        self._path = self._tickhoriz_path

    # 垂直刻度路径对象
    _tickvert_path = Path([[-0.0, 0.0], [-0.0, 1.0]])

    # 设置向上的刻度
    def _set_tickup(self):
        # 设置仿射变换，将图形按原样放置
        self._transform = Affine2D().scale(1.0, 1.0)
        # 设置吸附阈值为1.0
        self._snap_threshold = 1.0
        # 设置填充状态为False
        self._filled = False
        # 设置路径为垂直刻度路径对象
        self._path = self._tickvert_path

    # 设置向下的刻度
    def _set_tickdown(self):
        # 设置仿射变换，将图形纵向翻转
        self._transform = Affine2D().scale(1.0, -1.0)
        # 设置吸附阈值为1.0
        self._snap_threshold = 1.0
        # 设置填充状态为False
        self._filled = False
        # 设置路径为垂直刻度路径对象
        self._path = self._tickvert_path

    # 三角形路径对象
    _tri_path = Path([[0.0, 0.0], [0.0, -1.0],
                      [0.0, 0.0], [0.8, 0.5],
                      [0.0, 0.0], [-0.8, 0.5]],
                     [Path.MOVETO, Path.LINETO,
                      Path.MOVETO, Path.LINETO,
                      Path.MOVETO, Path.LINETO])

    # 设置朝下的三角形
    def _set_tri_down(self):
        # 设置仿射变换，将图形缩放为原来的一半
        self._transform = Affine2D().scale(0.5)
        # 设置吸附阈值为5.0
        self._snap_threshold = 5.0
        # 设置填充状态为False
        self._filled = False
        # 设置路径为三角形路径对象
        self._path = self._tri_path

    # 设置朝上的三角形
    def _set_tri_up(self):
        # 调用设置朝下的三角形的方法
        self._set_tri_down()
        # 将当前变换旋转180度
        self._transform = self._transform.rotate_deg(180)

    # 设置朝左的三角形
    def _set_tri_left(self):
        # 调用设置朝下的三角形的方法
        self._set_tri_down()
        # 将当前变换旋转270度
        self._transform = self._transform.rotate_deg(270)

    # 设置朝右的三角形
    def _set_tri_right(self):
        # 调用设置朝下的三角形的方法
        self._set_tri_down()
        # 将当前变换旋转90度
        self._transform = self._transform.rotate_deg(90)

    # 插入符号路径对象
    _caret_path = Path([[-1.0, 1.5], [0.0, 0.0], [1.0, 1.5]])
    # 设置下箭头形状的私有方法
    def _set_caretdown(self):
        # 创建一个仿射变换对象，并按比例缩放为原始的一半大小
        self._transform = Affine2D().scale(0.5)
        # 设置箭头捕捉的阈值为3.0
        self._snap_threshold = 3.0
        # 设定箭头是否填充的标志为False
        self._filled = False
        # 设置路径为下箭头的路径对象
        self._path = self._caret_path
        # 设置连接风格为用户指定的或默认为斜接
        self._joinstyle = self._user_joinstyle or JoinStyle.miter

    # 设置上箭头形状的私有方法
    def _set_caretup(self):
        # 调用下箭头设置方法
        self._set_caretdown()
        # 将当前仿射变换对象旋转180度，得到上箭头形状
        self._transform = self._transform.rotate_deg(180)

    # 设置左箭头形状的私有方法
    def _set_caretleft(self):
        # 调用下箭头设置方法
        self._set_caretdown()
        # 将当前仿射变换对象旋转270度，得到左箭头形状
        self._transform = self._transform.rotate_deg(270)

    # 设置右箭头形状的私有方法
    def _set_caretright(self):
        # 调用下箭头设置方法
        self._set_caretdown()
        # 将当前仿射变换对象旋转90度，得到右箭头形状
        self._transform = self._transform.rotate_deg(90)

    # 定义基础下箭头路径的静态成员变量
    _caret_path_base = Path([[-1.0, 0.0], [0.0, -1.5], [1.0, 0]])

    # 设置基础下箭头形状的私有方法
    def _set_caretdownbase(self):
        # 调用下箭头设置方法
        self._set_caretdown()
        # 将路径设置为基础下箭头路径
        self._path = self._caret_path_base

    # 设置基础上箭头形状的私有方法
    def _set_caretupbase(self):
        # 调用基础下箭头设置方法
        self._set_caretdownbase()
        # 将当前仿射变换对象旋转180度，得到基础上箭头形状
        self._transform = self._transform.rotate_deg(180)

    # 设置基础左箭头形状的私有方法
    def _set_caretleftbase(self):
        # 调用基础下箭头设置方法
        self._set_caretdownbase()
        # 将当前仿射变换对象旋转270度，得到基础左箭头形状
        self._transform = self._transform.rotate_deg(270)

    # 设置基础右箭头形状的私有方法
    def _set_caretrightbase(self):
        # 调用基础下箭头设置方法
        self._set_caretdownbase()
        # 将当前仿射变换对象旋转90度，得到基础右箭头形状
        self._transform = self._transform.rotate_deg(90)

    # 定义加号路径的静态成员变量
    _plus_path = Path([[-1.0, 0.0], [1.0, 0.0],
                       [0.0, -1.0], [0.0, 1.0]],
                      [Path.MOVETO, Path.LINETO,
                       Path.MOVETO, Path.LINETO])

    # 设置加号形状的私有方法
    def _set_plus(self):
        # 创建一个仿射变换对象，并按比例缩放为原始的一半大小
        self._transform = Affine2D().scale(0.5)
        # 设置加号捕捉的阈值为1.0
        self._snap_threshold = 1.0
        # 设定加号是否填充的标志为False
        self._filled = False
        # 设置路径为加号的路径对象
        self._path = self._plus_path

    # 定义叉号路径的静态成员变量
    _x_path = Path([[-1.0, -1.0], [1.0, 1.0],
                    [-1.0, 1.0], [1.0, -1.0]],
                   [Path.MOVETO, Path.LINETO,
                    Path.MOVETO, Path.LINETO])

    # 设置叉号形状的私有方法
    def _set_x(self):
        # 创建一个仿射变换对象，并按比例缩放为原始的一半大小
        self._transform = Affine2D().scale(0.5)
        # 设置叉号捕捉的阈值为3.0
        self._snap_threshold = 3.0
        # 设定叉号是否填充的标志为False
        self._filled = False
        # 设置路径为叉号的路径对象
        self._path = self._x_path

    # 定义填充加号路径的静态成员变量
    _plus_filled_path = Path._create_closed(np.array([
        (-1, -3), (+1, -3), (+1, -1), (+3, -1), (+3, +1), (+1, +1),
        (+1, +3), (-1, +3), (-1, +1), (-3, +1), (-3, -1), (-1, -1)]) / 6)
    _plus_filled_path_t = Path._create_closed(np.array([
        (+3, 0), (+3, +1), (+1, +1), (+1, +3),
        (-1, +3), (-1, +1), (-3, +1), (-3, 0)]) / 6)

    # 设置填充加号形状的私有方法
    def _set_plus_filled(self):
        # 创建一个空的仿射变换对象
        self._transform = Affine2D()
        # 设置填充加号捕捉的阈值为5.0
        self._snap_threshold = 5.0
        # 设定连接风格为用户指定的或默认为斜接
        self._joinstyle = self._user_joinstyle or JoinStyle.miter
        # 如果没有半填充成功，则使用填充加号路径
        if not self._half_fill():
            self._path = self._plus_filled_path
        else:
            # 否则，将路径设置为另一个填充加号路径
            self._path = self._alt_path = self._plus_filled_path_t
            # 获取填充样式并根据不同位置旋转路径以支持所有分区
            fs = self.get_fillstyle()
            self._transform.rotate_deg(
                {'top': 0, 'left': 90, 'bottom': 180, 'right': 270}[fs])
            # 冻结旋转后的仿射变换对象，并再次旋转180度
            self._alt_transform = self._transform.frozen().rotate_deg(180)
    # 定义一个闭合路径，表示完整的填充区域，使用相对坐标构建
    _x_filled_path = Path._create_closed(np.array([
        (-1, -2), (0, -1), (+1, -2), (+2, -1), (+1, 0), (+2, +1),
        (+1, +2), (0, +1), (-1, +2), (-2, +1), (-1, 0), (-2, -1)]) / 4)
    
    # 定义一个闭合路径，表示用于半填充区域的路径，使用相对坐标构建
    _x_filled_path_t = Path._create_closed(np.array([
        (+1, 0), (+2, +1), (+1, +2), (0, +1),
        (-1, +2), (-2, +1), (-1, 0)]) / 4)
    
    # 定义一个方法 _set_x_filled，用于设置填充样式为 'x' 形状
    def _set_x_filled(self):
        # 初始化仿射变换对象
        self._transform = Affine2D()
        # 设置用于捕捉的阈值
        self._snap_threshold = 5.0
        # 设置连接风格为用户指定的风格，或默认为 Miter 风格
        self._joinstyle = self._user_joinstyle or JoinStyle.miter
        
        # 如果不是半填充模式
        if not self._half_fill():
            # 使用完整填充路径
            self._path = self._x_filled_path
        else:
            # 否则，使用备用路径支持所有分区的顶部路径
            self._path = self._alt_path = self._x_filled_path_t
            # 获取填充样式并根据其旋转顶部路径
            fs = self.get_fillstyle()
            self._transform.rotate_deg(
                {'top': 0, 'left': 90, 'bottom': 180, 'right': 270}[fs])
            # 设置备用仿射变换对象，对应旋转180度
            self._alt_transform = self._transform.frozen().rotate_deg(180)
```