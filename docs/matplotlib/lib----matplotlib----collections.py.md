# `D:\src\scipysrc\matplotlib\lib\matplotlib\collections.py`

```py
"""
Classes for the efficient drawing of large collections of objects that
share most properties, e.g., a large number of line segments or
polygons.

The classes are not meant to be as flexible as their single element
counterparts (e.g., you may not be able to select all line styles) but
they are meant to be fast for common use cases (e.g., a large set of solid
line segments).
"""

# 导入所需的模块和库
import itertools  # 导入 itertools 模块，用于高效迭代工具
import math  # 导入 math 模块，提供数学运算函数
from numbers import Number, Real  # 从 numbers 模块导入 Number 和 Real 类型
import warnings  # 导入 warnings 模块，用于警告处理

import numpy as np  # 导入 NumPy 库，并将其命名为 np

import matplotlib as mpl  # 导入 Matplotlib 库，并将其命名为 mpl
from . import (_api, _path, artist, cbook, cm, colors as mcolors, _docstring,
               hatch as mhatch, lines as mlines, path as mpath, transforms)  # 从当前包中导入多个子模块和函数
from ._enums import JoinStyle, CapStyle  # 从当前包的 _enums 模块导入 JoinStyle 和 CapStyle 枚举类型


# "color" is excluded; it is a compound setter, and its docstring differs
# in LineCollection.
@_api.define_aliases({
    "antialiased": ["antialiaseds", "aa"],  # 定义属性别名，将 "antialiased" 别名扩展为 "antialiaseds" 和 "aa"
    "edgecolor": ["edgecolors", "ec"],  # 将 "edgecolor" 别名扩展为 "edgecolors" 和 "ec"
    "facecolor": ["facecolors", "fc"],  # 将 "facecolor" 别名扩展为 "facecolors" 和 "fc"
    "linestyle": ["linestyles", "dashes", "ls"],  # 将 "linestyle" 别名扩展为 "linestyles"、"dashes" 和 "ls"
    "linewidth": ["linewidths", "lw"],  # 将 "linewidth" 别名扩展为 "linewidths" 和 "lw"
    "offset_transform": ["transOffset"],  # 将 "offset_transform" 别名扩展为 "transOffset"
})
class Collection(artist.Artist, cm.ScalarMappable):
    r"""
    Base class for Collections. Must be subclassed to be usable.

    A Collection represents a sequence of `.Patch`\es that can be drawn
    more efficiently together than individually. For example, when a single
    path is being drawn repeatedly at different offsets, the renderer can
    typically execute a ``draw_marker()`` call much more efficiently than a
    series of repeated calls to ``draw_path()`` with the offsets put in
    one-by-one.

    Most properties of a collection can be configured per-element. Therefore,
    Collections have "plural" versions of many of the properties of a `.Patch`
    (e.g. `.Collection.get_paths` instead of `.Patch.get_path`). Exceptions are
    the *zorder*, *hatch*, *pickradius*, *capstyle* and *joinstyle* properties,
    which can only be set globally for the whole collection.

    Besides these exceptions, all properties can be specified as single values
    (applying to all elements) or sequences of values. The property of the
    ``i``\th element of the collection is::

      prop[i % len(prop)]

    Each Collection can optionally be used as its own `.ScalarMappable` by
    passing the *norm* and *cmap* parameters to its constructor. If the
    Collection's `.ScalarMappable` matrix ``_A`` has been set (via a call
    to `.Collection.set_array`), then at draw time this internal scalar
    mappable will be used to set the ``facecolors`` and ``edgecolors``,
    ignoring those that were manually passed in.
    """
    #: Either a list of 3x3 arrays or an Nx3x3 array (representing N
    #: transforms), suitable for the `all_transforms` argument to
    #: `~matplotlib.backend_bases.RendererBase.draw_path_collection`;
    #: each 3x3 array is used to initialize an
    #: `~matplotlib.transforms.Affine2D` object.
    #: Each kind of collection defines this based on its arguments.
    # 创建一个空的 NumPy 数组，形状为 (0, 3, 3)，用于存储变换矩阵
    _transforms = np.empty((0, 3, 3))
    
    # 默认情况下是否绘制边缘。这个属性在每个子类中会有所设置。
    _edge_default = False
    
    # 用于插值文档字符串的装饰器函数，返回存储在 self._paths 中的路径对象
    @_docstring.interpd
    def get_paths(self):
        return self._paths
    
    # 设置存储在 self._paths 中的路径对象
    def set_paths(self, paths):
        self._paths = paths
        self.stale = True
    
    # 返回存储在 self._transforms 中的变换矩阵数组
    def get_transforms(self):
        return self._transforms
    
    # 返回此艺术家偏移所使用的 `.Transform` 实例
    def get_offset_transform(self):
        if self._offset_transform is None:
            self._offset_transform = transforms.IdentityTransform()
        elif (not isinstance(self._offset_transform, transforms.Transform)
              and hasattr(self._offset_transform, '_as_mpl_transform')):
            self._offset_transform = \
                self._offset_transform._as_mpl_transform(self.axes)
        return self._offset_transform
    
    # 设置艺术家的偏移变换
    def set_offset_transform(self, offset_transform):
        """
        设置艺术家的偏移变换。
    
        Parameters
        ----------
        offset_transform : `.Transform`
        """
        self._offset_transform = offset_transform
    
    # 返回渲染器中艺术家的窗口范围
    def get_window_extent(self, renderer=None):
        # TODO: 检查确保这不会在除了散点图图例之外的其他情况下失败
        return self.get_datalim(transforms.IdentityTransform())
    
    # 辅助函数，用于准备绘制和命中测试
    def _prepare_points(self):
        transform = self.get_transform()
        offset_trf = self.get_offset_transform()
        offsets = self.get_offsets()
        paths = self.get_paths()
    
        # 如果艺术家使用单位，则将路径对象转换为使用相应单位的路径对象
        if self.have_units():
            paths = []
            for path in self.get_paths():
                vertices = path.vertices
                xs, ys = vertices[:, 0], vertices[:, 1]
                xs = self.convert_xunits(xs)
                ys = self.convert_yunits(ys)
                paths.append(mpath.Path(np.column_stack([xs, ys]), path.codes))
            xs = self.convert_xunits(offsets[:, 0])
            ys = self.convert_yunits(offsets[:, 1])
            offsets = np.ma.column_stack([xs, ys])
    
        # 如果 transform 不是仿射变换，则将 paths 中的路径对象转换为仿射变换
        if not transform.is_affine:
            paths = [transform.transform_path_non_affine(path)
                     for path in paths]
            transform = transform.get_affine()
    
        # 如果 offset_trf 不是仿射变换，则将 offsets 转换为仿射变换
        if not offset_trf.is_affine:
            offsets = offset_trf.transform_non_affine(offsets)
            offset_trf = offset_trf.get_affine()
    
        # 如果 offsets 是掩码数组，则将其填充为 NaN 值的 ndarray
        if isinstance(offsets, np.ma.MaskedArray):
            offsets = offsets.filled(np.nan)
    
        return transform, offset_trf, offsets, paths
    
    # 允许艺术家进行光栅化，通过 decorator 设置
    @artist.allow_rasterization
    # 设置用于包含测试的拾取半径。

    def set_pickradius(self, pickradius):
        """
        Set the pick radius used for containment tests.

        Parameters
        ----------
        pickradius : float
            Pick radius, in points.
        """
        # 如果pickradius不是实数，则引发值错误异常
        if not isinstance(pickradius, Real):
            raise ValueError(
                f"pickradius must be a real-valued number, not {pickradius!r}")
        # 将pickradius设置为对象的_pickradius属性
        self._pickradius = pickradius

    def get_pickradius(self):
        # 返回对象的_pickradius属性
        return self._pickradius

    def contains(self, mouseevent):
        """
        Test whether the mouse event occurred in the collection.

        Returns ``bool, dict(ind=itemlist)``, where every item in itemlist
        contains the event.
        """
        # 如果鼠标事件不在相同的画布上，或者集合不可见，则返回False和空字典
        if self._different_canvas(mouseevent) or not self.get_visible():
            return False, {}
        # 确定拾取半径
        pickradius = (
            float(self._picker)
            if isinstance(self._picker, Number) and
               self._picker is not True  # 只有在_picker是布尔值True时才会是True，而不仅仅是非零或1
            else self._pickradius)
        # 如果存在坐标轴，则更新视图限制
        if self.axes:
            self.axes._unstale_viewLim()
        # 准备转换、偏移转换、偏移量和路径
        transform, offset_trf, offsets, paths = self._prepare_points()
        # 检查鼠标事件点是否包含在路径集合中的任何路径中
        ind = _path.point_in_path_collection(
            mouseevent.x, mouseevent.y, pickradius,
            transform.frozen(), paths, self.get_transforms(),
            offsets, offset_trf, pickradius <= 0)
        # 返回结果，指示是否有路径包含鼠标事件点，以及包含路径的字典
        return len(ind) > 0, dict(ind=ind)

    def set_urls(self, urls):
        """
        Parameters
        ----------
        urls : list of str or None

        Notes
        -----
        URLs are currently only implemented by the SVG backend. They are
        ignored by all other backends.
        """
        # 设置对象的_urls属性为传入的urls列表，如果urls为None，则设置为包含一个None的列表
        self._urls = urls if urls is not None else [None]
        # 将stale属性设置为True，表示对象需要更新

    def get_urls(self):
        """
        Return a list of URLs, one for each element of the collection.

        The list contains *None* for elements without a URL. See
        :doc:`/gallery/misc/hyperlinks_sgskip` for an example.
        """
        # 返回对象的_urls属性，该属性是一个包含每个集合元素对应URL的列表
        return self._urls
    def set_hatch(self, hatch):
        r"""
        Set the hatching pattern

        *hatch* can be one of::

          /   - diagonal hatching
          \   - back diagonal
          |   - vertical
          -   - horizontal
          +   - crossed
          x   - crossed diagonal
          o   - small circle
          O   - large circle
          .   - dots
          *   - stars

        Letters can be combined, in which case all the specified
        hatchings are done.  If same letter repeats, it increases the
        density of hatching of that pattern.

        Unlike other properties such as linewidth and colors, hatching
        can only be specified for the collection as a whole, not separately
        for each member.

        Parameters
        ----------
        hatch : {'/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
            The hatching pattern to be set.
        """
        # Use validate_hatch(list) after deprecation.
        # 调用验证函数，检查并设置合法的填充图案
        mhatch._validate_hatch_pattern(hatch)
        # 将传入的填充图案设置为对象的填充属性
        self._hatch = hatch
        # 设定对象状态为过期，需要重新绘制
        self.stale = True

    def get_hatch(self):
        """
        Return the current hatching pattern.

        Returns
        -------
        str
            The current hatching pattern.
        """
        # 返回当前对象的填充图案属性
        return self._hatch

    def set_offsets(self, offsets):
        """
        Set the offsets for the collection.

        Parameters
        ----------
        offsets : (N, 2) or (2,) array-like
            The offsets to be set.
        """
        # 将输入的偏移量转换为NumPy数组
        offsets = np.asanyarray(offsets)
        # 如果偏移量的形状是 (2,)，则将其广播为 (1, 2)，否则保持不变
        if offsets.shape == (2,):  # Broadcast (2,) -> (1, 2) but nothing else.
            offsets = offsets[None, :]
        # 选择合适的列堆叠函数，处理掩码数组或普通数组
        cstack = (np.ma.column_stack if isinstance(offsets, np.ma.MaskedArray)
                  else np.column_stack)
        # 转换并设置 x 和 y 单位后的偏移量数组
        self._offsets = cstack(
            (np.asanyarray(self.convert_xunits(offsets[:, 0]), float),
             np.asanyarray(self.convert_yunits(offsets[:, 1]), float)))
        # 设定对象状态为过期，需要重新绘制
        self.stale = True

    def get_offsets(self):
        """
        Return the offsets for the collection.

        Returns
        -------
        ndarray
            The current offsets for the collection.
        """
        # 如果偏移量为空（None），返回一个 (1, 2) 的零数组；否则返回当前偏移量数组
        return np.zeros((1, 2)) if self._offsets is None else self._offsets

    def _get_default_linewidth(self):
        """
        Return the default linewidth for the collection.

        Returns
        -------
        float
            The default linewidth.
        """
        # 返回默认线宽，可以在子类中被覆盖
        return mpl.rcParams['patch.linewidth']  # validated as float

    def set_linewidth(self, lw):
        """
        Set the linewidth(s) for the collection.

        Parameters
        ----------
        lw : float or list of floats
            The linewidth(s) to be set.
        """
        # 如果线宽为 None，则使用默认线宽
        if lw is None:
            lw = self._get_default_linewidth()
        # 将输入的线宽转换为至少一维数组
        self._us_lw = np.atleast_1d(lw)

        # 缩放所有的虚线模式
        self._linewidths, self._linestyles = self._bcast_lwls(
            self._us_lw, self._us_linestyles)
        # 设定对象状态为过期，需要重新绘制
        self.stale = True
    # 定义方法，用于设置集合的线条样式
    def set_linestyle(self, ls):
        """
        Set the linestyle(s) for the collection.

        ===========================   =================
        linestyle                     description
        ===========================   =================
        ``'-'`` or ``'solid'``        实线
        ``'--'`` or  ``'dashed'``     虚线
        ``'-.'`` or  ``'dashdot'``    点划线
        ``':'`` or ``'dotted'``       点线
        ===========================   =================

        或者可以提供以下形式的破折号元组::

            (offset, onoffseq),

        其中 ``onoffseq`` 是一对偶数长度的元组，表示点线的各段长度（以点为单位）。

        Parameters
        ----------
        ls : str or tuple or list thereof
            可接受的值包括 {'-', '--', '-.', ':', '', (offset, on-off-seq)}。
            参见 `.Line2D.set_linestyle` 以获取完整的描述。
        """
        try:
            # 尝试获取线条样式对应的线型模式
            dashes = [mlines._get_dash_pattern(ls)]
        except ValueError:
            try:
                # 尝试获取列表中每个元素对应的线型模式
                dashes = [mlines._get_dash_pattern(x) for x in ls]
            except ValueError as err:
                emsg = f'Do not know how to convert {ls!r} to dashes'
                raise ValueError(emsg) from err

        # 获取未缩放的原始线型模式列表
        self._us_linestyles = dashes

        # 广播和缩放线宽和线型模式
        self._linewidths, self._linestyles = self._bcast_lwls(
            self._us_lw, self._us_linestyles)

    @_docstring.interpd
    # 设置集合的端点样式
    def set_capstyle(self, cs):
        """
        Set the `.CapStyle` for the collection (for all its elements).

        Parameters
        ----------
        cs : `.CapStyle` or %(CapStyle)s
        """
        self._capstyle = CapStyle(cs)

    @_docstring.interpd
    # 获取集合的端点样式
    def get_capstyle(self):
        """
        Return the cap style for the collection (for all its elements).

        Returns
        -------
        %(CapStyle)s or None
        """
        return self._capstyle.name if self._capstyle else None

    @_docstring.interpd
    # 设置集合的连接样式
    def set_joinstyle(self, js):
        """
        Set the `.JoinStyle` for the collection (for all its elements).

        Parameters
        ----------
        js : `.JoinStyle` or %(JoinStyle)s
        """
        self._joinstyle = JoinStyle(js)

    @_docstring.interpd
    # 获取集合的连接样式
    def get_joinstyle(self):
        """
        Return the join style for the collection (for all its elements).

        Returns
        -------
        %(JoinStyle)s or None
        """
        return self._joinstyle.name if self._joinstyle else None

    @staticmethod
    def _bcast_lwls(linewidths, dashes):
        """
        Internal helper function to broadcast + scale ls/lw
        
        In the collection drawing code, the linewidth and linestyle are cycled
        through as circular buffers (via ``v[i % len(v)]``).  Thus, if we are
        going to scale the dash pattern at set time (not draw time) we need to
        do the broadcasting now and expand both lists to be the same length.
        
        Parameters
        ----------
        linewidths : list
            line widths of collection
        dashes : list
            dash specification (offset, (dash pattern tuple))
        
        Returns
        -------
        linewidths, dashes : list
            Will be the same length, dashes are scaled by paired linewidth
        """
        # 如果处于经典模式，则直接返回原始的linewidths和dashes
        if mpl.rcParams['_internal.classic_mode']:
            return linewidths, dashes
        
        # 确保linewidths和dashes长度相同，以便可以进行配对操作
        if len(dashes) != len(linewidths):
            l_dashes = len(dashes)
            l_lw = len(linewidths)
            gcd = math.gcd(l_dashes, l_lw)
            dashes = list(dashes) * (l_lw // gcd)
            linewidths = list(linewidths) * (l_dashes // gcd)
        
        # 缩放虚线模式
        dashes = [mlines._scale_dashes(o, d, lw)
                  for (o, d), lw in zip(dashes, linewidths)]
        
        return linewidths, dashes

    def get_antialiased(self):
        """
        Get the antialiasing state for rendering.
        
        Returns
        -------
        array of bools
        """
        return self._antialiaseds

    def set_antialiased(self, aa):
        """
        Set the antialiasing state for rendering.
        
        Parameters
        ----------
        aa : bool or list of bools
        """
        # 如果aa为None，则使用默认的抗锯齿设置
        if aa is None:
            aa = self._get_default_antialiased()
        # 将aa转换为至少是一维的布尔数组，并赋值给_antialiaseds属性
        self._antialiaseds = np.atleast_1d(np.asarray(aa, bool))
        self.stale = True

    def _get_default_antialiased(self):
        # 可能在子类中被覆盖
        return mpl.rcParams['patch.antialiased']

    def set_color(self, c):
        """
        Set both the edgecolor and the facecolor.
        
        Parameters
        ----------
        c : :mpltype:`color` or list of RGBA tuples
        
        See Also
        --------
        Collection.set_facecolor, Collection.set_edgecolor
            For setting the edge or face color individually.
        """
        # 同时设置边缘颜色和填充颜色
        self.set_facecolor(c)
        self.set_edgecolor(c)

    def _get_default_facecolor(self):
        # 可能在子类中被覆盖
        return mpl.rcParams['patch.facecolor']

    def _set_facecolor(self, c):
        """
        Set the facecolor.

        Parameters
        ----------
        c : :mpltype:`color` or None

        If None, the default face color is used.

        """
        # 如果c为None，则使用默认的填充颜色
        if c is None:
            c = self._get_default_facecolor()

        # 将c转换为RGBA数组，并赋值给_facecolors属性
        self._facecolors = mcolors.to_rgba_array(c, self._alpha)
        self.stale = True
    # 设置集合对象的填充颜色。*c* 可以是一个颜色（所有图元都具有相同的颜色），或者是颜色序列；
    # 如果是序列，则图元将按序列循环显示颜色。
    # 如果 *c* 是 'none'，则图元将不被填充。
    def set_facecolor(self, c):
        if isinstance(c, str) and c.lower() in ("none", "face"):
            c = c.lower()
        # 存储原始的填充颜色设置
        self._original_facecolor = c
        # 调用内部方法设置填充颜色
        self._set_facecolor(c)

    # 获取集合对象的填充颜色
    def get_facecolor(self):
        return self._facecolors

    # 获取集合对象的边框颜色
    def get_edgecolor(self):
        # 如果边框颜色为 'face'，则返回填充颜色
        if cbook._str_equal(self._edgecolors, 'face'):
            return self.get_facecolor()
        else:
            return self._edgecolors

    # 获取默认的边框颜色
    def _get_default_edgecolor(self):
        # 可能会在子类中被覆盖
        return mpl.rcParams['patch.edgecolor']

    # 设置集合对象的边框颜色
    def _set_edgecolor(self, c):
        set_hatch_color = True
        if c is None:
            # 如果边框颜色为 None，根据配置或原始填充颜色决定是否使用默认边框颜色或者不设置边框
            if (mpl.rcParams['patch.force_edgecolor']
                    or self._edge_default
                    or cbook._str_equal(self._original_facecolor, 'none')):
                c = self._get_default_edgecolor()
            else:
                c = 'none'
                set_hatch_color = False
        # 如果边框颜色为 'face'，则设置边框颜色与填充颜色相同
        if cbook._str_lower_equal(c, 'face'):
            self._edgecolors = 'face'
            self.stale = True
            return
        # 将颜色转换为 RGBA 数组，并根据 alpha 值设置边框颜色
        self._edgecolors = mcolors.to_rgba_array(c, self._alpha)
        # 如果需要设置斜纹填充的颜色，并且边框颜色数组不为空，则将斜纹填充颜色设置为第一个边框颜色
        if set_hatch_color and len(self._edgecolors):
            self._hatch_color = tuple(self._edgecolors[0])
        self.stale = True

    # 设置集合对象的边框颜色
    def set_edgecolor(self, c):
        # 传递默认值用于 LineCollection。
        # 这允许我们在 _original_edgecolor 中保持 None 作为默认指示器。
        if isinstance(c, str) and c.lower() in ("none", "face"):
            c = c.lower()
        # 存储原始的边框颜色设置
        self._original_edgecolor = c
        # 调用内部方法设置边框颜色
        self._set_edgecolor(c)

    # 设置集合对象的透明度
    def set_alpha(self, alpha):
        # 调用父类方法设置数组的透明度
        artist.Artist._set_alpha_for_array(self, alpha)
        # 根据原始填充颜色重新设置填充颜色
        self._set_facecolor(self._original_facecolor)
        # 根据原始边框颜色重新设置边框颜色
        self._set_edgecolor(self._original_edgecolor)
    # 将 set_alpha 方法的文档字符串设置为 artist.Artist._set_alpha_for_array 方法的文档字符串
    set_alpha.__doc__ = artist.Artist._set_alpha_for_array.__doc__

    # 获取当前对象的线条宽度信息并返回
    def get_linewidth(self):
        return self._linewidths

    # 获取当前对象的线条样式信息并返回
    def get_linestyle(self):
        return self._linestyles

    # 设置映射标志位，用于确定边缘和/或面部是否进行了颜色映射
    def _set_mappable_flags(self):
        """
        确定边缘和/或面部是否进行了颜色映射。

        这是 update_scalarmappable 的辅助函数。
        设置布尔标志 '_edge_is_mapped' 和 '_face_is_mapped'。

        Returns
        -------
        mapping_change : bool
            如果任一标志为 True 或者标志已更改，则为 True。
        """
        # 将标志初始化为 None，以确保第一次调用时返回 True。
        edge0 = self._edge_is_mapped
        face0 = self._face_is_mapped
        # 在返回后，标志必须是布尔值，而不是 None。
        self._edge_is_mapped = False
        self._face_is_mapped = False
        if self._A is not None:
            if not cbook._str_equal(self._original_facecolor, 'none'):
                self._face_is_mapped = True
                if cbook._str_equal(self._original_edgecolor, 'face'):
                    self._edge_is_mapped = True
            else:
                if self._original_edgecolor is None:
                    self._edge_is_mapped = True

        mapped = self._face_is_mapped or self._edge_is_mapped
        changed = (edge0 is None or face0 is None
                   or self._edge_is_mapped != edge0
                   or self._face_is_mapped != face0)
        return mapped or changed
    def update_scalarmappable(self):
        """
        Update colors from the scalar mappable array, if any.

        Assign colors to edges and faces based on the array and/or
        colors that were directly set, as appropriate.
        """
        # 检查是否设置了标量映射标志，如果没有设置则返回
        if not self._set_mappable_flags():
            return
        
        # 如果存在数据数组 _A
        if self._A is not None:
            # 对于 QuadMesh 类型的对象，可以映射二维数组（但 pcolormesh 提供一维数组）
            if self._A.ndim > 1 and not isinstance(self, _MeshData):
                raise ValueError('Collections can only map rank 1 arrays')
            
            # 如果 _alpha 是可迭代的且大小与 _A 的大小不匹配，抛出 ValueError
            if np.iterable(self._alpha):
                if self._alpha.size != self._A.size:
                    raise ValueError(
                        f'Data array shape, {self._A.shape} '
                        'is incompatible with alpha array shape, '
                        f'{self._alpha.shape}. '
                        'This can occur with the deprecated '
                        'behavior of the "flat" shading option, '
                        'in which a row and/or column of the data '
                        'array is dropped.')
                
                # 对于 pcolormesh、scatter 等操作可能会使 _A 变平坦，重新调整 _alpha 的形状
                self._alpha = self._alpha.reshape(self._A.shape)
            
            # 根据 _A 和 _alpha 转换为 RGBA 颜色数组，并存储在 _mapped_colors 中
            self._mapped_colors = self.to_rgba(self._A, self._alpha)

        # 如果需要映射面的颜色，则将 _mapped_colors 赋值给 _facecolors
        if self._face_is_mapped:
            self._facecolors = self._mapped_colors
        else:
            # 否则恢复到初始的面颜色 _original_facecolor
            self._set_facecolor(self._original_facecolor)
        
        # 如果需要映射边的颜色，则将 _mapped_colors 赋值给 _edgecolors
        if self._edge_is_mapped:
            self._edgecolors = self._mapped_colors
        else:
            # 否则恢复到初始的边颜色 _original_edgecolor
            self._set_edgecolor(self._original_edgecolor)
        
        # 将 stale 属性设置为 True，表示对象已更新
        self.stale = True

    def get_fill(self):
        """Return whether face is colored."""
        # 返回面是否被填充色（即判断 _original_facecolor 是否为 "none"）
        return not cbook._str_lower_equal(self._original_facecolor, "none")

    def update_from(self, other):
        """Copy properties from other to self."""

        # 调用父类 artist.Artist 的 update_from 方法，复制其他属性到当前对象
        artist.Artist.update_from(self, other)
        
        # 复制特定属性值
        self._antialiaseds = other._antialiaseds
        self._mapped_colors = other._mapped_colors
        self._edge_is_mapped = other._edge_is_mapped
        self._original_edgecolor = other._original_edgecolor
        self._edgecolors = other._edgecolors
        self._face_is_mapped = other._face_is_mapped
        self._original_facecolor = other._original_facecolor
        self._facecolors = other._facecolors
        self._linewidths = other._linewidths
        self._linestyles = other._linestyles
        self._us_linestyles = other._us_linestyles
        self._pickradius = other._pickradius
        self._hatch = other._hatch
        
        # 复制用于标量映射的属性
        self._A = other._A
        self.norm = other.norm
        self.cmap = other.cmap
        self.stale = True
class _CollectionWithSizes(Collection):
    """
    Base class for collections that have an array of sizes.
    """
    _factor = 1.0  # 定义默认缩放因子为1.0

    def get_sizes(self):
        """
        Return the sizes ('areas') of the elements in the collection.

        Returns
        -------
        array
            The 'area' of each element.
        """
        return self._sizes  # 返回集合中各元素的大小（面积）

    def set_sizes(self, sizes, dpi=72.0):
        """
        Set the sizes of each member of the collection.

        Parameters
        ----------
        sizes : `numpy.ndarray` or None
            The size to set for each element of the collection.  The
            value is the 'area' of the element.
        dpi : float, default: 72
            The dpi of the canvas.
        """
        if sizes is None:
            self._sizes = np.array([])  # 如果sizes为None，则设置空数组作为大小
            self._transforms = np.empty((0, 3, 3))  # 设置空的变换数组
        else:
            self._sizes = np.asarray(sizes)  # 将sizes转换为numpy数组
            self._transforms = np.zeros((len(self._sizes), 3, 3))  # 初始化变换数组为全零
            scale = np.sqrt(self._sizes) * dpi / 72.0 * self._factor  # 计算缩放比例
            self._transforms[:, 0, 0] = scale  # 在变换数组中设置X方向的缩放比例
            self._transforms[:, 1, 1] = scale  # 在变换数组中设置Y方向的缩放比例
            self._transforms[:, 2, 2] = 1.0  # 在变换数组中设置Z方向的缩放比例为1.0（不变）
        self.stale = True  # 标记为需要重新绘制

    @artist.allow_rasterization
    def draw(self, renderer):
        self.set_sizes(self._sizes, self.figure.dpi)  # 根据当前尺寸和dpi设置大小
        super().draw(renderer)  # 调用父类的绘制方法


class PathCollection(_CollectionWithSizes):
    r"""
    A collection of `~.path.Path`\s, as created by e.g. `~.Axes.scatter`.
    """

    def __init__(self, paths, sizes=None, **kwargs):
        """
        Parameters
        ----------
        paths : list of `.path.Path`
            The paths that will make up the `.Collection`.
        sizes : array-like
            The factor by which to scale each drawn `~.path.Path`. One unit
            squared in the Path's data space is scaled to be ``sizes**2``
            points when rendered.
        **kwargs
            Forwarded to `.Collection`.
        """

        super().__init__(**kwargs)  # 调用父类的初始化方法
        self.set_paths(paths)  # 设置路径集合
        self.set_sizes(sizes)  # 设置大小
        self.stale = True  # 标记为需要重新绘制

    def get_paths(self):
        return self._paths  # 返回路径集合


class PolyCollection(_CollectionWithSizes):
    """
    A collection of polygons with the same color and transparency.
    """
    def __init__(self, verts, sizes=None, *, closed=True, **kwargs):
        """
        Parameters
        ----------
        verts : list of array-like
            多边形的顶点序列 [*verts0*, *verts1*, ...]，其中每个元素 *verts_i* 定义了多边形 *i* 的顶点，是一个形状为 (M, 2) 的二维数组。
        sizes : array-like, default: None
            多边形的缩放因子的平方值。每个多边形 *verts_i* 的坐标将会乘以对应 *sizes* 中条目的平方根（即 *sizes* 指定了面积的缩放）。这个缩放是在应用 Artist 主变换之前进行的。
        closed : bool, default: True
            指定多边形是否应该闭合，通过在末尾添加 CLOSEPOLY 连接。
        **kwargs
            转发到 `.Collection`。
        """
        super().__init__(**kwargs)
        self.set_sizes(sizes)  # 调用 set_sizes 方法设置多边形的缩放因子
        self.set_verts(verts, closed)  # 调用 set_verts 方法设置多边形的顶点
        self.stale = True  # 将 stale 属性设置为 True，表示需要更新

    def set_verts(self, verts, closed=True):
        """
        设置多边形的顶点。

        Parameters
        ----------
        verts : list of array-like
            多边形的顶点序列 [*verts0*, *verts1*, ...]，其中每个元素 *verts_i* 定义了多边形 *i* 的顶点，是一个形状为 (M, 2) 的二维数组。
        closed : bool, default: True
            指定多边形是否应该闭合，通过在末尾添加 CLOSEPOLY 连接。
        """
        self.stale = True  # 将 stale 属性设置为 True，表示需要更新
        if isinstance(verts, np.ma.MaskedArray):
            verts = verts.astype(float).filled(np.nan)

        # 如果路径不需要闭合，则不需要执行复杂的操作。
        if not closed:
            self._paths = [mpath.Path(xy) for xy in verts]  # 创建未闭合路径的 Path 对象列表
            return

        # 对于数组的快速处理路径
        if isinstance(verts, np.ndarray) and len(verts.shape) == 3:
            verts_pad = np.concatenate((verts, verts[:, :1]), axis=1)
            # 一次性创建代码比每次传递 closed=True 给 Path 更快。
            codes = np.empty(verts_pad.shape[1], dtype=mpath.Path.code_type)
            codes[:] = mpath.Path.LINETO
            codes[0] = mpath.Path.MOVETO
            codes[-1] = mpath.Path.CLOSEPOLY
            self._paths = [mpath.Path(xy, codes) for xy in verts_pad]  # 创建闭合路径的 Path 对象列表
            return

        self._paths = []
        for xy in verts:
            if len(xy):
                self._paths.append(mpath.Path._create_closed(xy))  # 创建闭合路径的 Path 对象列表
            else:
                self._paths.append(mpath.Path(xy))  # 创建空路径的 Path 对象列表

    set_paths = set_verts  # 将 set_paths 方法指向 set_verts 方法，两者功能相同
    # 定义方法 `set_verts_and_codes`，用于设置路径顶点及其路径代码
    def set_verts_and_codes(self, verts, codes):
        """Initialize vertices with path codes."""
        # 检查顶点和代码长度是否一致，如果不一致则抛出数值错误异常
        if len(verts) != len(codes):
            raise ValueError("'codes' must be a 1D list or array "
                             "with the same length of 'verts'")
        # 使用列表推导式创建路径对象 `_paths`，每个路径对象由顶点 `verts` 和对应的代码 `codes` 组成
        self._paths = [mpath.Path(xy, cds) if len(xy) else mpath.Path(xy)
                       for xy, cds in zip(verts, codes)]
        # 将 `stale` 属性设置为 `True`，表示数据已过时
        self.stale = True
class RegularPolyCollection(_CollectionWithSizes):
    """A collection of n-sided regular polygons."""

    # 使用 Matplotlib 提供的路径生成器创建正多边形路径
    _path_generator = mpath.Path.unit_regular_polygon
    # 缩放因子，用于计算面积
    _factor = np.pi ** (-1/2)

    def __init__(self,
                 numsides,
                 *,
                 rotation=0,
                 sizes=(1,),
                 **kwargs):
        """
        Parameters
        ----------
        numsides : int
            多边形的边数。
        rotation : float
            以弧度表示的多边形旋转角度。
        sizes : tuple of float
            以点数的平方表示的多边形外接圆的面积。
        **kwargs
            传递给 `.Collection` 的其他参数。

        Examples
        --------
        完整示例请参考 :doc:`/gallery/event_handling/lasso_demo`::

            offsets = np.random.rand(20, 2)
            facecolors = [cm.jet(x) for x in np.random.rand(20)]

            collection = RegularPolyCollection(
                numsides=5, # 五边形
                rotation=0, sizes=(50,),
                facecolors=facecolors,
                edgecolors=("black",),
                linewidths=(1,),
                offsets=offsets,
                offset_transform=ax.transData,
                )
        """
        super().__init__(**kwargs)
        self.set_sizes(sizes)
        self._numsides = numsides
        # 根据给定边数生成正多边形的路径
        self._paths = [self._path_generator(numsides)]
        self._rotation = rotation
        self.set_transform(transforms.IdentityTransform())

    def get_numsides(self):
        return self._numsides

    def get_rotation(self):
        return self._rotation

    @artist.allow_rasterization
    def draw(self, renderer):
        # 根据指定的尺寸和 DPI 设置多边形大小
        self.set_sizes(self._sizes, self.figure.dpi)
        # 根据变换列表更新变换矩阵
        self._transforms = [
            transforms.Affine2D(x).rotate(-self._rotation).get_matrix()
            for x in self._transforms
        ]
        # 明确指定不调用父类的 draw 方法，因为 set_sizes 必须在更新 self._transforms 之前调用
        Collection.draw(self, renderer)


class StarPolygonCollection(RegularPolyCollection):
    """Draw a collection of regular stars with *numsides* points."""
    # 使用 Matplotlib 提供的路径生成器创建正星形路径
    _path_generator = mpath.Path.unit_regular_star


class AsteriskPolygonCollection(RegularPolyCollection):
    """Draw a collection of regular asterisks with *numsides* points."""
    # 使用 Matplotlib 提供的路径生成器创建正星号路径
    _path_generator = mpath.Path.unit_regular_asterisk


class LineCollection(Collection):
    r"""
    Represents a sequence of `.Line2D`\s that should be drawn together.

    This class extends `.Collection` to represent a sequence of
    `.Line2D`\s instead of just a sequence of `.Patch`\s.
    Just as in `.Collection`, each property of a *LineCollection* may be either
    a single value or a list of values. This list is then used cyclically for
    each element of the LineCollection, so the property of the ``i``\th element
    of the collection is::

      prop[i % len(prop)]
    """
    """
    The properties of each member of a *LineCollection* default to their values
    in :rc:`lines.*` instead of :rc:`patch.*`, and the property *colors* is
    added in place of *edgecolors*.
    """
    
    # 默认情况下，每个 *LineCollection* 成员的属性值使用 :rc:`lines.*` 中的设定而不是 :rc:`patch.*`，并且 *colors* 属性替代了 *edgecolors*。
    
    _edge_default = True
    
    def __init__(self, segments,  # Can be None.
                 *,
                 zorder=2,        # Collection.zorder is 1
                 **kwargs
                 ):
        """
        Parameters
        ----------
        segments : list of array-like
            A sequence (*line0*, *line1*, *line2*) of lines, where each line is a list
            of points::

                lineN = [(x0, y0), (x1, y1), ... (xm, ym)]

            or the equivalent Mx2 numpy array with two columns. Each line
            can have a different number of segments.
        linewidths : float or list of float, default: :rc:`lines.linewidth`
            The width of each line in points.
        colors : :mpltype:`color` or list of color, default: :rc:`lines.color`
            A sequence of RGBA tuples (e.g., arbitrary color strings, etc, not
            allowed).
        antialiaseds : bool or list of bool, default: :rc:`lines.antialiased`
            Whether to use antialiasing for each line.
        zorder : float, default: 2
            zorder of the lines once drawn.

        facecolors : :mpltype:`color` or list of :mpltype:`color`, default: 'none'
            When setting *facecolors*, each line is interpreted as a boundary
            for an area, implicitly closing the path from the last point to the
            first point. The enclosed area is filled with *facecolor*.
            In order to manually specify what should count as the "interior" of
            each line, please use `.PathCollection` instead, where the
            "interior" can be specified by appropriate usage of
            `~.path.Path.CLOSEPOLY`.

        **kwargs
            Forwarded to `.Collection`.
        """
        
        # 无奈地，mplot3d 需要显式设置 'facecolors'。
        kwargs.setdefault('facecolors', 'none')
        # 调用父类的初始化方法，设置 zorder 和其他传入的关键字参数
        super().__init__(
            zorder=zorder,
            **kwargs)
        # 设置线段集合
        self.set_segments(segments)

    def set_segments(self, segments):
        # 如果 segments 为 None，则直接返回
        if segments is None:
            return
        
        # 将每个线段转换为对应的 Path 对象存储在 self._paths 中
        self._paths = [mpath.Path(seg) if isinstance(seg, np.ma.MaskedArray)
                       else mpath.Path(np.asarray(seg, float))
                       for seg in segments]
        # 设置 stale 属性为 True，表示需要更新
        self.stale = True

    # 为了与 PolyCollection 兼容而设置的别名方法
    set_verts = set_segments  # for compatibility with PolyCollection
    set_paths = set_segments
    def get_segments(self):
        """
        Returns
        -------
        list
            List of segments in the LineCollection. Each list item contains an
            array of vertices.
        """
        # 初始化空列表以存储所有线段的顶点数据
        segments = []

        # 遍历所有路径对象
        for path in self._paths:
            # 从路径对象中迭代获取线段的顶点数据，不进行简化处理
            vertices = [
                vertex
                for vertex, _  # 忽略第二个元素，只关注顶点数据
                # 在这里不进行简化处理，因为需要保留数据空间的值
                # 没有办法确定“正确”的简化阈值，所以不尝试简化
                in path.iter_segments(simplify=False)
            ]
            # 将顶点数据转换为 NumPy 数组并添加到 segments 列表中
            vertices = np.asarray(vertices)
            segments.append(vertices)

        return segments

    def _get_default_linewidth(self):
        # 返回默认线条宽度，从 matplotlib 全局配置中获取
        return mpl.rcParams['lines.linewidth']

    def _get_default_antialiased(self):
        # 返回默认是否开启抗锯齿效果，从 matplotlib 全局配置中获取
        return mpl.rcParams['lines.antialiased']

    def _get_default_edgecolor(self):
        # 返回默认边缘颜色，从 matplotlib 全局配置中获取
        return mpl.rcParams['lines.color']

    def _get_default_facecolor(self):
        # 返回默认填充颜色，这里设为 'none' 表示无填充
        return 'none'

    def set_alpha(self, alpha):
        # 继承文档字符串，设置对象的透明度
        super().set_alpha(alpha)
        # 如果存在间隔颜色，则恢复原始的间隔颜色
        if self._gapcolor is not None:
            self.set_gapcolor(self._original_gapcolor)

    def set_color(self, c):
        """
        Set the edgecolor(s) of the LineCollection.

        Parameters
        ----------
        c : :mpltype:`color` or list of :mpltype:`color`
            Single color (all lines have same color), or a
            sequence of RGBA tuples; if it is a sequence the lines will
            cycle through the sequence.
        """
        # 调用 set_edgecolor 方法设置 LineCollection 的边缘颜色
        self.set_edgecolor(c)

    set_colors = set_color

    def get_color(self):
        # 返回 LineCollection 的边缘颜色
        return self._edgecolors

    get_colors = get_color  # for compatibility with old versions

    def set_gapcolor(self, gapcolor):
        """
        Set a color to fill the gaps in the dashed line style.

        .. note::

            Striped lines are created by drawing two interleaved dashed lines.
            There can be overlaps between those two, which may result in
            artifacts when using transparency.

            This functionality is experimental and may change.

        Parameters
        ----------
        gapcolor : :mpltype:`color` or list of :mpltype:`color` or None
            The color with which to fill the gaps. If None, the gaps are
            unfilled.
        """
        # 存储原始的间隔颜色
        self._original_gapcolor = gapcolor
        # 调用 _set_gapcolor 方法设置填充间隔的颜色
        self._set_gapcolor(gapcolor)

    def _set_gapcolor(self, gapcolor):
        # 如果 gapcolor 不为 None，则将其转换为 RGBA 数组
        if gapcolor is not None:
            gapcolor = mcolors.to_rgba_array(gapcolor, self._alpha)
        # 更新 LineCollection 的间隔颜色属性
        self._gapcolor = gapcolor
        # 标记对象为过时，需要重新绘制
        self.stale = True

    def get_gapcolor(self):
        # 返回 LineCollection 的间隔颜色
        return self._gapcolor
    def _get_inverse_paths_linestyles(self):
        """
        Returns the path and pattern for the gaps in the non-solid lines.

        This path and pattern is the inverse of the path and pattern used to
        construct the non-solid lines. For solid lines, we set the inverse path
        to nans to prevent drawing an inverse line.
        """
        # 生成包含路径和线型模式的列表，用于非实线中的间隙
        path_patterns = [
            # 如果线型为实线 (0, None)，则将路径设置为全为 NaN 的路径，以防止绘制反向线条
            (mpath.Path(np.full((1, 2), np.nan)), ls)
            if ls == (0, None) else
            # 否则，使用当前路径和其反向虚线模式
            (path, mlines._get_inverse_dash_pattern(*ls))
            for (path, ls) in
            # 将当前对象的路径和循环的线型样式一一对应起来
            zip(self._paths, itertools.cycle(self._linestyles))]

        # 返回路径和模式的列表
        return zip(*path_patterns)
# EventCollection 类，继承自 LineCollection 类
class EventCollection(LineCollection):
    """
    一个包含沿单一轴上事件发生位置的集合。

    事件由一维数组给出，它们没有幅度，并且显示为平行线。
    """

    # 默认边缘值为 True
    _edge_default = True

    def __init__(self,
                 positions,  # 位置数组，不能为空
                 orientation='horizontal',  # 方向，默认水平
                 *,
                 lineoffset=0,  # 线偏移，默认为0
                 linelength=1,  # 线长度，默认为1
                 linewidth=None,  # 线宽度，可以是单个值或列表，默认取自全局配置
                 color=None,  # 线颜色，可以是单个值或列表，默认取自全局配置
                 linestyle='solid',  # 线样式，默认实线
                 antialiased=None,  # 是否使用抗锯齿，可以是单个值或列表，默认取自全局配置
                 **kwargs
                 ):
        """
        参数
        ----------
        positions : 一维类数组
            每个值都是一个事件。
        orientation : {'horizontal', 'vertical'}，默认：'horizontal'
            事件序列沿此方向绘制。单个事件的标记线沿正交方向。
        lineoffset : 浮点数，默认：0
            标记的中心点在与 *orientation* 正交的方向上的偏移量。
        linelength : 浮点数，默认：1
            标记的总高度（即标记从 ``lineoffset - linelength/2`` 到 ``lineoffset + linelength/2``）。
        linewidth : 浮点数或其列表，默认：全局配置中的 :rc:`lines.linewidth`
            事件线的线宽，以点为单位。
        color : :mpltype:`color` 或其列表中的 :mpltype:`color`，默认：全局配置中的 :rc:`lines.color`
            事件线的颜色。
        linestyle : 字符串或其元组或列表，默认：'solid'
            有效字符串包括 ['solid', 'dashed', 'dashdot', 'dotted',
            '-', '--', '-.', ':']。虚线元组应该是形式为::

                (offset, onoffseq),

            其中 *onoffseq* 是点中的奇数长度元组，表示开和关的墨水。
        antialiased : 布尔值或其列表，默认：全局配置中的 :rc:`lines.antialiased`
            是否使用反走样来绘制线条。
        **kwargs
            转发到 `.LineCollection`。

        示例
        --------
        .. plot:: gallery/lines_bars_and_markers/eventcollection_demo.py
        """
        # 调用父类的构造函数，传入空列表作为初始参数
        super().__init__([],
                         linewidths=linewidth, linestyles=linestyle,
                         colors=color, antialiaseds=antialiased,
                         **kwargs)
        self._is_horizontal = True  # 初始值，下面可能会修改
        self._linelength = linelength
        self._lineoffset = lineoffset
        self.set_orientation(orientation)  # 设置方向
        self.set_positions(positions)  # 设置事件位置

    def get_positions(self):
        """
        返回一个包含位置浮点值的数组。
        """
        pos = 0 if self.is_horizontal() else 1  # 如果是水平的，取第一列；否则取第二列
        return [segment[0, pos] for segment in self.get_segments()]  # 返回每个线段的第一个点的位置值
    def set_positions(self, positions):
        """
        设置事件的位置。

        Parameters
        ----------
        positions : array-like
            事件的位置坐标。

        Raises
        ------
        ValueError
            如果 positions 不是一维数组。

        """
        # 如果 positions 是 None，则将其设置为空列表
        if positions is None:
            positions = []
        # 检查 positions 是否是一维数组，如果不是则抛出 ValueError 异常
        if np.ndim(positions) != 1:
            raise ValueError('positions must be one-dimensional')
        # 获取线的偏移量和长度
        lineoffset = self.get_lineoffset()
        linelength = self.get_linelength()
        # 根据事件线的方向确定位置的索引
        pos_idx = 0 if self.is_horizontal() else 1
        # 创建一个空数组用于存储线段的坐标
        segments = np.empty((len(positions), 2, 2))
        # 将 positions 中的坐标进行排序并赋值给 segments
        segments[:, :, pos_idx] = np.sort(positions)[:, None]
        # 设置线段的另一个坐标，使其在指定方向上的位置为线的中心
        segments[:, 0, 1 - pos_idx] = lineoffset + linelength / 2
        segments[:, 1, 1 - pos_idx] = lineoffset - linelength / 2
        # 调用 set_segments 方法设置事件线的线段
        self.set_segments(segments)

    def add_positions(self, position):
        """
        在指定的位置添加一个或多个事件。

        Parameters
        ----------
        position : array-like
            要添加的事件的位置坐标。

        """
        # 如果 position 是 None 或者是空数组，则直接返回
        if position is None or (hasattr(position, 'len') and len(position) == 0):
            return
        # 获取当前的位置
        positions = self.get_positions()
        # 将 position 转换成数组并与当前位置合并
        positions = np.hstack([positions, np.asanyarray(position)])
        # 调用 set_positions 方法设置事件的位置
        self.set_positions(positions)
    extend_positions = append_positions = add_positions

    def is_horizontal(self):
        """
        如果事件线是水平的返回 True，否则返回 False。
        """
        return self._is_horizontal

    def get_orientation(self):
        """
        返回事件线的方向 ('horizontal' 或 'vertical')。
        """
        return 'horizontal' if self.is_horizontal() else 'vertical'

    def switch_orientation(self):
        """
        切换事件线的方向，从垂直到水平或者从水平到垂直。
        """
        # 获取当前线段的坐标
        segments = self.get_segments()
        # 遍历并反转每个线段的坐标
        for i, segment in enumerate(segments):
            segments[i] = np.fliplr(segment)
        # 调用 set_segments 方法设置反转后的线段坐标
        self.set_segments(segments)
        # 更新事件线的方向
        self._is_horizontal = not self.is_horizontal()
        # 设置 stale 标志为 True，表示需要更新
        self.stale = True

    def set_orientation(self, orientation):
        """
        设置事件线的方向。

        Parameters
        ----------
        orientation : {'horizontal', 'vertical'}
            事件线的方向，可以是 'horizontal' 或 'vertical'。

        """
        # 检查方向参数，确定是否需要切换方向
        is_horizontal = _api.check_getitem(
            {"horizontal": True, "vertical": False},
            orientation=orientation)
        if is_horizontal == self.is_horizontal():
            return
        # 调用 switch_orientation 方法切换事件线的方向
        self.switch_orientation()

    def get_linelength(self):
        """
        返回用于标记每个事件的线的长度。
        """
        return self._linelength

    def set_linelength(self, linelength):
        """
        设置用于标记每个事件的线的长度。

        Parameters
        ----------
        linelength : float
            线的长度。

        """
        # 如果新长度与当前长度相同，则直接返回
        if linelength == self.get_linelength():
            return
        # 获取线的偏移量和线段的坐标
        lineoffset = self.get_lineoffset()
        segments = self.get_segments()
        # 确定坐标数组中的位置索引
        pos = 1 if self.is_horizontal() else 0
        # 更新每个线段的坐标，使其长度为指定的 linelength
        for segment in segments:
            segment[0, pos] = lineoffset + linelength / 2.
            segment[1, pos] = lineoffset - linelength / 2.
        # 调用 set_segments 方法设置更新后的线段坐标
        self.set_segments(segments)
        # 更新事件线的长度属性
        self._linelength = linelength
    # 返回用于标记每个事件的行的偏移量
    def get_lineoffset(self):
        return self._lineoffset

    # 设置用于标记每个事件的行的偏移量
    def set_lineoffset(self, lineoffset):
        # 如果传入的偏移量与当前偏移量相同，则直接返回，不进行修改
        if lineoffset == self.get_lineoffset():
            return
        
        # 获取线的长度和分段信息
        linelength = self.get_linelength()
        segments = self.get_segments()
        
        # 确定位置索引，如果是水平方向则为1，否则为0
        pos = 1 if self.is_horizontal() else 0
        
        # 更新每个分段的起始和结束位置，使其相对于新的偏移量居中
        for segment in segments:
            segment[0, pos] = lineoffset + linelength / 2.
            segment[1, pos] = lineoffset - linelength / 2.
        
        # 更新分段信息
        self.set_segments(segments)
        # 设置新的偏移量
        self._lineoffset = lineoffset

    # 获取用于标记每个事件的线的宽度
    def get_linewidth(self):
        return super().get_linewidth()[0]

    # 获取用于标记每个事件的线的宽度，以列表形式返回
    def get_linewidths(self):
        return super().get_linewidth()

    # 返回用于标记每个事件的线的颜色
    def get_color(self):
        return self.get_colors()[0]
class CircleCollection(_CollectionWithSizes):
    """A collection of circles, drawn using splines."""

    _factor = np.pi ** (-1/2)  # 计算系数，用于后续的计算

    def __init__(self, sizes, **kwargs):
        """
        Parameters
        ----------
        sizes : float or array-like
            The area of each circle in points^2.
        **kwargs
            Forwarded to `.Collection`.
        """
        super().__init__(**kwargs)  # 调用父类的初始化方法
        self.set_sizes(sizes)  # 设置每个圆的面积大小
        self.set_transform(transforms.IdentityTransform())  # 设置变换为单位变换
        self._paths = [mpath.Path.unit_circle()]  # 初始化路径为单位圆的路径对象


class EllipseCollection(Collection):
    """A collection of ellipses, drawn using splines."""

    def __init__(self, widths, heights, angles, *, units='points', **kwargs):
        """
        Parameters
        ----------
        widths : array-like
            The lengths of the first axes (e.g., major axis lengths).
        heights : array-like
            The lengths of second axes.
        angles : array-like
            The angles of the first axes, degrees CCW from the x-axis.
        units : {'points', 'inches', 'dots', 'width', 'height', 'x', 'y', 'xy'}
            The units in which majors and minors are given; 'width' and
            'height' refer to the dimensions of the axes, while 'x' and 'y'
            refer to the *offsets* data units. 'xy' differs from all others in
            that the angle as plotted varies with the aspect ratio, and equals
            the specified angle only when the aspect ratio is unity.  Hence
            it behaves the same as the `~.patches.Ellipse` with
            ``axes.transData`` as its transform.
        **kwargs
            Forwarded to `Collection`.
        """
        super().__init__(**kwargs)  # 调用父类的初始化方法
        self.set_widths(widths)  # 设置每个椭圆的宽度
        self.set_heights(heights)  # 设置每个椭圆的高度
        self.set_angles(angles)  # 设置每个椭圆的角度
        self._units = units  # 设置单位类型
        self.set_transform(transforms.IdentityTransform())  # 设置变换为单位变换
        self._transforms = np.empty((0, 3, 3))  # 初始化变换矩阵为空数组
        self._paths = [mpath.Path.unit_circle()]  # 初始化路径为单位圆的路径对象
    def _set_transforms(self):
        """Calculate transforms immediately before drawing."""

        # 获取当前对象的坐标轴和图形
        ax = self.axes
        fig = self.figure

        # 根据设置的单位计算比例尺 sc
        if self._units == 'xy':
            sc = 1
        elif self._units == 'x':
            sc = ax.bbox.width / ax.viewLim.width
        elif self._units == 'y':
            sc = ax.bbox.height / ax.viewLim.height
        elif self._units == 'inches':
            sc = fig.dpi
        elif self._units == 'points':
            sc = fig.dpi / 72.0
        elif self._units == 'width':
            sc = ax.bbox.width
        elif self._units == 'height':
            sc = ax.bbox.height
        elif self._units == 'dots':
            sc = 1.0
        else:
            # 如果单位不被识别，抛出异常
            raise ValueError(f'Unrecognized units: {self._units!r}')

        # 初始化变换矩阵
        self._transforms = np.zeros((len(self._widths), 3, 3))
        # 根据比例尺调整宽度和高度
        widths = self._widths * sc
        heights = self._heights * sc
        sin_angle = np.sin(self._angles)
        cos_angle = np.cos(self._angles)
        # 计算变换矩阵的各个元素
        self._transforms[:, 0, 0] = widths * cos_angle
        self._transforms[:, 0, 1] = heights * -sin_angle
        self._transforms[:, 1, 0] = widths * sin_angle
        self._transforms[:, 1, 1] = heights * cos_angle
        self._transforms[:, 2, 2] = 1.0

        # 获取 Affine2D 类型的变换对象
        _affine = transforms.Affine2D
        # 如果单位是 'xy'，则获取当前数据坐标变换矩阵并清零平移部分，然后设置变换
        if self._units == 'xy':
            m = ax.transData.get_affine().get_matrix().copy()
            m[:2, 2:] = 0
            self.set_transform(_affine(m))

    def set_widths(self, widths):
        """Set the lengths of the first axes (e.g., major axis)."""
        # 设置宽度，并标记为需要更新
        self._widths = 0.5 * np.asarray(widths).ravel()
        self.stale = True

    def set_heights(self, heights):
        """Set the lengths of second axes (e.g., minor axes)."""
        # 设置高度，并标记为需要更新
        self._heights = 0.5 * np.asarray(heights).ravel()
        self.stale = True

    def set_angles(self, angles):
        """Set the angles of the first axes, degrees CCW from the x-axis."""
        # 设置角度，并标记为需要更新
        self._angles = np.deg2rad(angles).ravel()
        self.stale = True

    def get_widths(self):
        """Get the lengths of the first axes (e.g., major axis)."""
        # 返回宽度的两倍
        return self._widths * 2

    def get_heights(self):
        """Set the lengths of second axes (e.g., minor axes)."""
        # 返回高度的两倍
        return self._heights * 2

    def get_angles(self):
        """Get the angles of the first axes, degrees CCW from the x-axis."""
        # 返回角度，转换为角度制
        return np.rad2deg(self._angles)

    @artist.allow_rasterization
    def draw(self, renderer):
        # 调用 _set_transforms 方法更新变换，然后调用父类的 draw 方法进行绘制
        self._set_transforms()
        super().draw(renderer)
# 定义一个继承自 Collection 的 PatchCollection 类，用于管理图形补丁的集合
class PatchCollection(Collection):
    """
    A generic collection of patches.

    PatchCollection draws faster than a large number of equivalent individual
    Patches. It also makes it easier to assign a colormap to a heterogeneous
    collection of patches.
    """

    # 初始化方法，接受一个补丁对象的列表 patches 和其他可选参数
    def __init__(self, patches, *, match_original=False, **kwargs):
        """
        Parameters
        ----------
        patches : list of `.Patch`
            A sequence of Patch objects.  This list may include
            a heterogeneous assortment of different patch types.

        match_original : bool, default: False
            If True, use the colors and linewidths of the original
            patches.  If False, new colors may be assigned by
            providing the standard collection arguments, facecolor,
            edgecolor, linewidths, norm or cmap.

        **kwargs
            All other parameters are forwarded to `.Collection`.

            If any of *edgecolors*, *facecolors*, *linewidths*, *antialiaseds*
            are None, they default to their `.rcParams` patch setting, in
            sequence form.

        Notes
        -----
        The use of `~matplotlib.cm.ScalarMappable` functionality is optional.
        If the `~matplotlib.cm.ScalarMappable` matrix ``_A`` has been set (via
        a call to `~.ScalarMappable.set_array`), at draw time a call to scalar
        mappable will be made to set the face colors.
        """

        # 如果 match_original 为 True，则根据原始补丁的颜色和线宽设置参数
        if match_original:
            # 内部函数，确定每个补丁的 facecolor
            def determine_facecolor(patch):
                if patch.get_fill():
                    return patch.get_facecolor()
                return [0, 0, 0, 0]

            # 根据 patches 中各补丁的属性设置 kwargs 中的颜色、线宽等参数
            kwargs['facecolors'] = [determine_facecolor(p) for p in patches]
            kwargs['edgecolors'] = [p.get_edgecolor() for p in patches]
            kwargs['linewidths'] = [p.get_linewidth() for p in patches]
            kwargs['linestyles'] = [p.get_linestyle() for p in patches]
            kwargs['antialiaseds'] = [p.get_antialiased() for p in patches]

        # 调用父类 Collection 的初始化方法，传递所有的 kwargs
        super().__init__(**kwargs)

        # 设置 Collection 的路径信息为 patches 的路径信息
        self.set_paths(patches)

    # 设置 Collection 的路径信息为 patches 的路径信息
    def set_paths(self, patches):
        paths = [p.get_transform().transform_path(p.get_path())
                 for p in patches]
        self._paths = paths


# TriMesh 类，继承自 Collection，用于以 Gouraud 着色方式绘制三角网格
class TriMesh(Collection):
    """
    Class for the efficient drawing of a triangular mesh using Gouraud shading.

    A triangular mesh is a `~matplotlib.tri.Triangulation` object.
    """
    # 初始化方法，接受一个 Triangulation 对象 triangulation 和其他参数
    def __init__(self, triangulation, **kwargs):
        super().__init__(**kwargs)
        self._triangulation = triangulation
        self._shading = 'gouraud'

        # 创建单位边界框
        self._bbox = transforms.Bbox.unit()

        # 由于 Triangulation 对象的限制，需要进行 xy 数据的复制
        xy = np.hstack((triangulation.x.reshape(-1, 1),
                        triangulation.y.reshape(-1, 1)))
        self._bbox.update_from_data_xy(xy)
    # 如果路径属性尚未初始化，则调用set_paths()方法初始化
    def get_paths(self):
        if self._paths is None:
            self.set_paths()
        # 返回当前对象的路径属性
        return self._paths

    # 将三角剖分转换为路径并设置为对象的路径属性
    def set_paths(self):
        self._paths = self.convert_mesh_to_paths(self._triangulation)

    @staticmethod
    # 将给定的三角剖分转换为`.Path`对象的序列
    def convert_mesh_to_paths(tri):
        """
        Convert a given mesh into a sequence of `.Path` objects.

        This function is primarily of use to implementers of backends that do
        not directly support meshes.
        """
        triangles = tri.get_masked_triangles()  # 获取掩模化后的三角形
        verts = np.stack((tri.x[triangles], tri.y[triangles]), axis=-1)  # 提取三角形的顶点坐标
        return [mpath.Path(x) for x in verts]  # 返回顶点坐标转换为路径对象的列表

    @artist.allow_rasterization
    # 在渲染器上绘制对象
    def draw(self, renderer):
        if not self.get_visible():  # 如果对象不可见，则直接返回
            return
        renderer.open_group(self.__class__.__name__, gid=self.get_gid())  # 在渲染器上打开一个组

        transform = self.get_transform()  # 获取对象的变换信息

        # 获取三角剖分和每个顶点的颜色
        tri = self._triangulation
        triangles = tri.get_masked_triangles()  # 获取掩模化后的三角形

        verts = np.stack((tri.x[triangles], tri.y[triangles]), axis=-1)  # 提取三角形的顶点坐标

        self.update_scalarmappable()  # 更新标量映射信息
        colors = self._facecolors[triangles]  # 获取三角形的颜色信息

        gc = renderer.new_gc()  # 创建一个新的渲染上下文
        self._set_gc_clip(gc)  # 设置渲染上下文的剪辑区域
        gc.set_linewidth(self.get_linewidth()[0])  # 设置渲染上下文的线宽
        renderer.draw_gouraud_triangles(gc, verts, colors, transform.frozen())  # 在渲染器上绘制高洛德三角形
        gc.restore()  # 恢复渲染上下文的状态
        renderer.close_group(self.__class__.__name__)  # 在渲染器上关闭当前组
class _MeshData:
    r"""
    Class for managing the two dimensional coordinates of Quadrilateral meshes
    and the associated data with them. This class is a mixin and is intended to
    be used with another collection that will implement the draw separately.

    A quadrilateral mesh is a grid of M by N adjacent quadrilaterals that are
    defined via a (M+1, N+1) grid of vertices. The quadrilateral (m, n) is
    defined by the vertices ::

               (m+1, n) ----------- (m+1, n+1)
                  /                   /
                 /                 /
                /               /
            (m, n) -------- (m, n+1)

    The mesh need not be regular and the polygons need not be convex.

    Parameters
    ----------
    coordinates : (M+1, N+1, 2) array-like
        The vertices. ``coordinates[m, n]`` specifies the (x, y) coordinates
        of vertex (m, n).

    shading : {'flat', 'gouraud'}, default: 'flat'
        Specifies the shading model for the mesh.

    Attributes
    ----------
    _coordinates : array-like
        Holds the vertex coordinates of the mesh.

    _shading : str
        Holds the shading model for the mesh ('flat' or 'gouraud').
    """
    
    def __init__(self, coordinates, *, shading='flat'):
        # 检查传入的 coordinates 形状是否为 (None, None, 2)，即 (M+1, N+1, 2)
        _api.check_shape((None, None, 2), coordinates=coordinates)
        # 初始化 _coordinates 为传入的 vertices
        self._coordinates = coordinates
        # 初始化 _shading 为传入的 shading 模式
        self._shading = shading

    def set_array(self, A):
        """
        Set the data values.

        Parameters
        ----------
        A : array-like
            The mesh data. Supported array shapes are:

            - (M, N) or (M*N,): a mesh with scalar data. The values are mapped
              to colors using normalization and a colormap. See parameters
              *norm*, *cmap*, *vmin*, *vmax*.
            - (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
            - (M, N, 4): an image with RGBA values (0-1 float or 0-255 int),
              i.e. including transparency.

            If the values are provided as a 2D grid, the shape must match the
            coordinates grid. If the values are 1D, they are reshaped to 2D.
            M, N follow from the coordinates grid, where the coordinates grid
            shape is (M, N) for 'gouraud' *shading* and (M+1, N+1) for 'flat'
            shading.
        """
        # 获取网格的高度和宽度
        height, width = self._coordinates.shape[0:-1]
        # 根据 shading 模式设置网格的高度和宽度
        if self._shading == 'flat':
            h, w = height - 1, width - 1
        else:
            h, w = height, width
        # 支持的数据形状列表
        ok_shapes = [(h, w, 3), (h, w, 4), (h, w), (h * w,)]
        # 如果 A 不为 None，则检查其形状是否在支持的形状列表中
        if A is not None:
            shape = np.shape(A)
            if shape not in ok_shapes:
                raise ValueError(
                    f"For X ({width}) and Y ({height}) with {self._shading} "
                    f"shading, A should have shape "
                    f"{' or '.join(map(str, ok_shapes))}, not {A.shape}")
        # 调用父类的 set_array 方法，传入 A
        return super().set_array(A)
    def get_coordinates(self):
        """
        Return the vertices of the mesh as an (M+1, N+1, 2) array.

        M, N are the number of quadrilaterals in the rows / columns of the
        mesh, corresponding to (M+1, N+1) vertices.
        The last dimension specifies the components (x, y).
        """
        # 返回网格顶点坐标数组，形状为 (M+1, N+1, 2)
        return self._coordinates

    def get_edgecolor(self):
        """
        Return the edge colors as a flattened RGBA array.

        Note that we want to return an array of shape (N*M, 4),
        a flattened RGBA collection.
        """
        # 返回边缘颜色数组，形状为 (N*M, 4)，表示为展平的 RGBA 集合
        return super().get_edgecolor().reshape(-1, 4)

    def get_facecolor(self):
        """
        Return the face colors as a flattened RGBA array.

        Note that we want to return an array of shape (N*M, 4),
        a flattened RGBA collection.
        """
        # 返回面颜色数组，形状为 (N*M, 4)，表示为展平的 RGBA 集合
        return super().get_facecolor().reshape(-1, 4)

    @staticmethod
    def _convert_mesh_to_paths(coordinates):
        """
        Convert a given mesh into a sequence of `.Path` objects.

        This function is primarily of use to implementers of backends that do
        not directly support quadmeshes.
        """
        # 将给定的网格转换为一系列 `.Path` 对象
        if isinstance(coordinates, np.ma.MaskedArray):
            c = coordinates.data
        else:
            c = coordinates
        points = np.concatenate([
            c[:-1, :-1],
            c[:-1, 1:],
            c[1:, 1:],
            c[1:, :-1],
            c[:-1, :-1]
        ], axis=2).reshape((-1, 5, 2))
        return [mpath.Path(x) for x in points]

    def _convert_mesh_to_triangles(self, coordinates):
        """
        Convert a given mesh into a sequence of triangles, each point
        with its own color.  The result can be used to construct a call to
        `~.RendererBase.draw_gouraud_triangles`.
        """
        # 将给定的网格转换为一系列三角形，每个点都有自己的颜色
        # 结果可以用于构建调用 `~.RendererBase.draw_gouraud_triangles` 的参数
        if isinstance(coordinates, np.ma.MaskedArray):
            p = coordinates.data
        else:
            p = coordinates

        p_a = p[:-1, :-1]
        p_b = p[:-1, 1:]
        p_c = p[1:, 1:]
        p_d = p[1:, :-1]
        p_center = (p_a + p_b + p_c + p_d) / 4.0
        triangles = np.concatenate([
            p_a, p_b, p_center,
            p_b, p_c, p_center,
            p_c, p_d, p_center,
            p_d, p_a, p_center,
        ], axis=2).reshape((-1, 3, 2))

        c = self.get_facecolor().reshape((*coordinates.shape[:2], 4))
        z = self.get_array()
        mask = z.mask if np.ma.is_masked(z) else None
        if mask is not None:
            c[mask, 3] = np.nan
        c_a = c[:-1, :-1]
        c_b = c[:-1, 1:]
        c_c = c[1:, 1:]
        c_d = c[1:, :-1]
        c_center = (c_a + c_b + c_c + c_d) / 4.0
        colors = np.concatenate([
            c_a, c_b, c_center,
            c_b, c_c, c_center,
            c_c, c_d, c_center,
            c_d, c_a, c_center,
        ], axis=2).reshape((-1, 3, 4))
        tmask = np.isnan(colors[..., 2, 3])
        return triangles[~tmask], colors[~tmask]
class QuadMesh(_MeshData, Collection):
    r"""
    Class for the efficient drawing of a quadrilateral mesh.

    A quadrilateral mesh is a grid of M by N adjacent quadrilaterals that are
    defined via a (M+1, N+1) grid of vertices. The quadrilateral (m, n) is
    defined by the vertices ::

               (m+1, n) ----------- (m+1, n+1)
                  /                   /
                 /                 /
                /               /
            (m, n) -------- (m, n+1)

    The mesh need not be regular and the polygons need not be convex.

    Parameters
    ----------
    coordinates : (M+1, N+1, 2) array-like
        The vertices. ``coordinates[m, n]`` specifies the (x, y) coordinates
        of vertex (m, n).

    antialiased : bool, default: True
        Flag indicating whether to use antialiased rendering.

    shading : {'flat', 'gouraud'}, default: 'flat'
        Type of shading for the mesh.

    Notes
    -----
    Unlike other `.Collection`\s, the default *pickradius* of `.QuadMesh` is 0,
    i.e. `~.Artist.contains` checks whether the test point is within any of the
    mesh quadrilaterals.

    """

    def __init__(self, coordinates, *, antialiased=True, shading='flat',
                 **kwargs):
        kwargs.setdefault("pickradius", 0)
        # 设置默认的 pickradius 为 0
        super().__init__(coordinates=coordinates, shading=shading)
        # 调用父类的初始化方法，传递顶点坐标和着色方式参数
        Collection.__init__(self, **kwargs)
        # 调用 Collection 类的初始化方法，传递额外的参数

        self._antialiased = antialiased
        # 设置抗锯齿标志位
        self._bbox = transforms.Bbox.unit()
        # 创建单位边界框对象
        self._bbox.update_from_data_xy(self._coordinates.reshape(-1, 2))
        # 从顶点坐标更新边界框
        self.set_mouseover(False)
        # 设置鼠标悬停效果为 False

    def get_paths(self):
        if self._paths is None:
            self.set_paths()
        return self._paths
        # 获取路径信息，如果路径为空则调用设置路径方法后返回

    def set_paths(self):
        self._paths = self._convert_mesh_to_paths(self._coordinates)
        # 将网格转换为路径对象
        self.stale = True
        # 设置对象为过时状态

    def get_datalim(self, transData):
        return (self.get_transform() - transData).transform_bbox(self._bbox)
        # 获取数据限制边界框的变换结果

    @artist.allow_rasterization
    # 允许栅格化绘制
    # 绘制对象到指定的渲染器上
    def draw(self, renderer):
        # 如果对象不可见，则直接返回
        if not self.get_visible():
            return
        
        # 在渲染器中打开一个新的绘制组，使用对象的类名和组 ID
        renderer.open_group(self.__class__.__name__, self.get_gid())
        
        # 获取对象的变换
        transform = self.get_transform()
        
        # 获取对象的偏移变换
        offset_trf = self.get_offset_transform()
        
        # 获取对象的偏移量
        offsets = self.get_offsets()
        
        # 如果对象使用了单位制（如英寸、厘米等），则进行单位转换
        if self.have_units():
            xs = self.convert_xunits(offsets[:, 0])
            ys = self.convert_yunits(offsets[:, 1])
            offsets = np.column_stack([xs, ys])
        
        # 更新标量映射对象
        self.update_scalarmappable()
        
        # 如果变换不是仿射变换，则对坐标进行变换
        if not transform.is_affine:
            coordinates = self._coordinates.reshape((-1, 2))
            coordinates = transform.transform(coordinates)
            coordinates = coordinates.reshape(self._coordinates.shape)
            transform = transforms.IdentityTransform()
        else:
            coordinates = self._coordinates
        
        # 如果偏移变换不是仿射变换，则对偏移量进行非仿射变换
        if not offset_trf.is_affine:
            offsets = offset_trf.transform_non_affine(offsets)
            offset_trf = offset_trf.get_affine()
        
        # 创建新的图形上下文对象
        gc = renderer.new_gc()
        
        # 设置图形上下文的对齐方式
        gc.set_snap(self.get_snap())
        
        # 设置图形上下文的裁剪区域
        self._set_gc_clip(gc)
        
        # 设置图形上下文的线宽
        gc.set_linewidth(self.get_linewidth()[0])
        
        # 如果使用了 'gouraud' 阴影模式，则绘制高拉德三角形
        if self._shading == 'gouraud':
            triangles, colors = self._convert_mesh_to_triangles(coordinates)
            renderer.draw_gouraud_triangles(
                gc, triangles, colors, transform.frozen())
        else:
            # 否则，绘制四边形网格
            renderer.draw_quad_mesh(
                gc, transform.frozen(),
                coordinates.shape[1] - 1, coordinates.shape[0] - 1,
                coordinates, offsets, offset_trf,
                # 后端期望展平的 rgba 数组 (n*m, 4) 用于填充颜色和边框颜色
                self.get_facecolor().reshape((-1, 4)),
                self._antialiased, self.get_edgecolors().reshape((-1, 4)))
        
        # 恢复图形上下文的状态
        gc.restore()
        
        # 在渲染器中关闭当前绘制组
        renderer.close_group(self.__class__.__name__)
        
        # 将对象的 stale 标志设为 False，表示绘制状态为最新
        self.stale = False

    # 获取与光标事件相关的数据
    def get_cursor_data(self, event):
        # 判断事件是否在对象内部，并返回是否包含信息
        contained, info = self.contains(event)
        
        # 如果事件在对象内部且对象有数组数据，则返回数组数据中对应索引的值
        if contained and self.get_array() is not None:
            return self.get_array().ravel()[info["ind"]]
        
        # 否则返回 None
        return None
# 定义一个名为 PolyQuadMesh 的类，继承自 _MeshData 和 PolyCollection 类
class PolyQuadMesh(_MeshData, PolyCollection):
    """
    Class for drawing a quadrilateral mesh as individual Polygons.

    A quadrilateral mesh is a grid of M by N adjacent quadrilaterals that are
    defined via a (M+1, N+1) grid of vertices. The quadrilateral (m, n) is
    defined by the vertices ::

               (m+1, n) ----------- (m+1, n+1)
                  /                   /
                 /                 /
                /               /
            (m, n) -------- (m, n+1)

    The mesh need not be regular and the polygons need not be convex.

    Parameters
    ----------
    coordinates : (M+1, N+1, 2) array-like
        The vertices. ``coordinates[m, n]`` specifies the (x, y) coordinates
        of vertex (m, n).

    Notes
    -----
    Unlike `.QuadMesh`, this class will draw each cell as an individual Polygon.
    This is significantly slower, but allows for more flexibility when wanting
    to add additional properties to the cells, such as hatching.

    Another difference from `.QuadMesh` is that if any of the vertices or data
    of a cell are masked, that Polygon will **not** be drawn and it won't be in
    the list of paths returned.
    """

    # 初始化方法，接受 coordinates 参数和其他关键字参数
    def __init__(self, coordinates, **kwargs):
        # 用于跟踪是否正在使用已弃用的压缩方式
        # 在初始化器之后进行更新
        self._deprecated_compression = False
        # 调用父类 _MeshData 的初始化方法
        super().__init__(coordinates=coordinates)
        # 调用 PolyCollection 的初始化方法，传入空的顶点列表和其他关键字参数
        PolyCollection.__init__(self, verts=[], **kwargs)
        # 在压缩已弃用期间存储此属性
        self._original_mask = ~self._get_unmasked_polys()
        # 检查是否有任何已屏蔽的多边形
        self._deprecated_compression = np.any(self._original_mask)
        # 设置顶点以更新 PolyCollection 的路径
        # 此操作在初始化器之后调用，确保所有关键字参数已被处理，并可用于屏蔽计算
        self._set_unmasked_verts()
    # 获取未遮罩的多边形区域，使用坐标和数组来确定
    def _get_unmasked_polys(self):
        """Get the unmasked regions using the coordinates and array"""
        # 获取坐标数组的掩码（mask），并进行逻辑或操作
        mask = np.any(np.ma.getmaskarray(self._coordinates), axis=-1)

        # 确定多边形的形状，即每个 X/Y 数组的角点
        mask = (mask[0:-1, 0:-1] | mask[1:, 1:] | mask[0:-1, 1:] | mask[1:, 0:-1])

        # 如果存在废弃的压缩属性，并且存在原始的掩码数组，返回未遮罩的区域
        if (getattr(self, "_deprecated_compression", False) and
                np.any(self._original_mask)):
            return ~(mask | self._original_mask)
        
        # 考虑数组数据，暂时避免压缩警告，并在调用后重置该变量
        with cbook._setattr_cm(self, _deprecated_compression=False):
            arr = self.get_array()
        
        # 如果数组不为空，则根据数组的维度进行处理
        if arr is not None:
            arr = np.ma.getmaskarray(arr)
            if arr.ndim == 3:
                # RGB(A) 情况
                mask |= np.any(arr, axis=-1)
            elif arr.ndim == 2:
                mask |= arr
            else:
                mask |= arr.reshape(self._coordinates[:-1, :-1, :].shape[:2])
        
        # 返回未遮罩的区域
        return ~mask

    # 设置未遮罩的顶点
    def _set_unmasked_verts(self):
        X = self._coordinates[..., 0]
        Y = self._coordinates[..., 1]

        # 获取未遮罩的多边形
        unmask = self._get_unmasked_polys()
        X1 = np.ma.filled(X[:-1, :-1])[unmask]
        Y1 = np.ma.filled(Y[:-1, :-1])[unmask]
        X2 = np.ma.filled(X[1:, :-1])[unmask]
        Y2 = np.ma.filled(Y[1:, :-1])[unmask]
        X3 = np.ma.filled(X[1:, 1:])[unmask]
        Y3 = np.ma.filled(Y[1:, 1:])[unmask]
        X4 = np.ma.filled(X[:-1, 1:])[unmask]
        Y4 = np.ma.filled(Y[:-1, 1:])[unmask]
        npoly = len(X1)

        # 组装顶点数组
        xy = np.ma.stack([X1, Y1, X2, Y2, X3, Y3, X4, Y4, X1, Y1], axis=-1)
        verts = xy.reshape((npoly, 5, 2))
        
        # 设置图形对象的顶点
        self.set_verts(verts)

    # 获取边缘颜色
    def get_edgecolor(self):
        # 继承的文档字符串
        # 只返回已绘制多边形的边缘颜色
        ec = super().get_edgecolor()
        unmasked_polys = self._get_unmasked_polys().ravel()
        
        # 如果边缘颜色数组长度与未遮罩多边形数量不匹配，则返回原数组
        if len(ec) != len(unmasked_polys):
            # 映射有误
            return ec
        
        # 返回未遮罩多边形的边缘颜色
        return ec[unmasked_polys, :]

    # 获取填充颜色
    def get_facecolor(self):
        # 继承的文档字符串
        # 只返回已绘制多边形的填充颜色
        fc = super().get_facecolor()
        unmasked_polys = self._get_unmasked_polys().ravel()
        
        # 如果填充颜色数组长度与未遮罩多边形数量不匹配，则返回原数组
        if len(fc) != len(unmasked_polys):
            # 映射有误
            return fc
        
        # 返回未遮罩多边形的填充颜色
        return fc[unmasked_polys, :]
    # 设置多边形网格的数组数据，继承自父类的文档字符串
    prev_unmask = self._get_unmasked_polys()
    # 如果 MPL 版本低于 3.8，且输入数组 A 是一维的，需要处理压缩后的情况
    # 在过渡期内，只有在存在掩码元素且发生压缩时才警告
    if self._deprecated_compression and np.ndim(A) == 1:
        _api.warn_deprecated("3.8", message="Setting a PolyQuadMesh array using "
                             "the compressed values is deprecated. "
                             "Pass the full 2D shape of the original array "
                             f"{prev_unmask.shape} including the masked elements.")
        # 创建一个与原始掩码形状相同的空数组 Afull
        Afull = np.empty(self._original_mask.shape)
        # 将未掩码的部分填充为输入的数组 A 的值
        Afull[~self._original_mask] = A
        # 更新掩码以包含可能的新掩码元素，但不更新原始数据的压缩部分
        mask = self._original_mask.copy()
        mask[~self._original_mask] |= np.ma.getmask(A)
        # 使用新的掩码创建一个带有掩码的多维数组 A
        A = np.ma.array(Afull, mask=mask)
        # 调用父类的 set_array 方法来设置数组 A
        return super().set_array(A)
    # 将标记为不推荐使用压缩的标志设置为 False
    self._deprecated_compression = False
    # 调用父类的 set_array 方法设置数组 A
    super().set_array(A)
    # 如果掩码发生了变化，则需要更新绘制的多边形集合
    if not np.array_equal(prev_unmask, self._get_unmasked_polys()):
        self._set_unmasked_verts()

# 获取多边形网格的数组数据，继承自父类的文档字符串
def get_array(self):
    A = super().get_array()
    if A is None:
        return
    # 如果标记为不推荐使用压缩，并且数组 A 中存在掩码元素，发出警告
    if self._deprecated_compression and np.any(np.ma.getmask(A)):
        _api.warn_deprecated("3.8", message=(
            "Getting the array from a PolyQuadMesh will return the full "
            "array in the future (uncompressed). To get this behavior now "
            "set the PolyQuadMesh with a 2D array .set_array(data2d)."))
        # 返回压缩后的数组 A 的数据部分
        return np.ma.compressed(A)
    # 否则返回原始的数组 A
    return A
```