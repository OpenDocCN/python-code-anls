# `D:\src\scipysrc\matplotlib\lib\matplotlib\path.py`

```py
"""
A module for dealing with the polylines used throughout Matplotlib.
Matplotlib 中用于处理折线的模块。

The primary class for polyline handling in Matplotlib is `Path`.  Almost all
vector drawing makes use of `Path`\s somewhere in the drawing pipeline.
Matplotlib 中处理折线的主要类是 `Path`。几乎所有的矢量绘图在绘图流程中都会使用 `Path`。

Whilst a `Path` instance itself cannot be drawn, some `.Artist` subclasses,
such as `.PathPatch` and `.PathCollection`, can be used for convenient `Path`
visualisation.
虽然 `Path` 实例本身不能被绘制，但一些 `.Artist` 的子类，比如 `.PathPatch` 和 `.PathCollection`，可以用于方便地可视化 `Path`。

"""

import copy  # 导入深拷贝模块
from functools import lru_cache  # 导入 lru_cache 装饰器
from weakref import WeakValueDictionary  # 导入弱引用字典模块

import numpy as np  # 导入 NumPy 库

import matplotlib as mpl  # 导入 Matplotlib 库
from . import _api, _path  # 导入本地模块 _api 和 _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation  # 从 cbook 模块中导入函数
from .bezier import BezierSegment  # 从 bezier 模块中导入 BezierSegment 类


class Path:
    """
    A series of possibly disconnected, possibly closed, line and curve
    segments.

    The underlying storage is made up of two parallel numpy arrays:

    - *vertices*: an (N, 2) float array of vertices
        顶点数组，存储 (N, 2) 大小的浮点型数组

    - *codes*: an N-length `numpy.uint8` array of path codes, or None
        代码数组，长度为 N 的 `numpy.uint8` 数组，或者为 None

    These two arrays always have the same length in the first
    dimension.  For example, to represent a cubic curve, you must
    provide three vertices and three `CURVE4` codes.
    这两个数组在第一个维度上总是具有相同的长度。例如，要表示一个三次曲线，必须提供三个顶点和三个 `CURVE4` 代码。

    The code types are:

    - `STOP`   :  1 vertex (ignored)
        停止标记，当前未使用

    - `MOVETO` :  1 vertex
        移动到指定顶点，抬起画笔并移动到给定顶点。

    - `LINETO` :  1 vertex
        从当前位置画一条直线到给定顶点。

    - `CURVE3` :  1 control point, 1 endpoint
        从当前位置画一条二次贝塞尔曲线，给定控制点和终点。

    - `CURVE4` :  2 control points, 1 endpoint
        从当前位置画一条三次贝塞尔曲线，给定两个控制点和终点。

    - `CLOSEPOLY` : 1 vertex (ignored)
        画一条线段到当前折线的起始点，当前未使用。

    If *codes* is None, it is interpreted as a `MOVETO` followed by a series
    of `LINETO`.
    如果 *codes* 为 None，则被解释为一个 `MOVETO` 后面跟着一系列 `LINETO`。

    Users of Path objects should not access the vertices and codes arrays
    directly.  Instead, they should use `iter_segments` or `cleaned` to get the
    vertex/code pairs.  This helps, in particular, to consistently handle the
    case of *codes* being None.
    Path 对象的用户不应直接访问顶点和代码数组。而是应该使用 `iter_segments` 或 `cleaned` 来获取顶点/代码对。这可以帮助一致地处理 *codes* 为 None 的情况。

    Some behavior of Path objects can be controlled by rcParams. See the
    rcParams whose keys start with 'path.'.
    Path 对象的某些行为可以通过 rcParams 控制。参见以 'path.' 开头的 rcParams 键。

    .. note::
        注意：顶点和代码数组应被视为不可变的 —— 构造函数中会做一些优化和假设。

    """

    code_type = np.uint8

    # Path codes
    STOP = code_type(0)         # 1 vertex
    MOVETO = code_type(1)       # 1 vertex
    LINETO = code_type(2)       # 1 vertex
    CURVE3 = code_type(3)       # 2 vertices
    CURVE4 = code_type(4)       # 3 vertices
    CLOSEPOLY = code_type(79)   # 1 vertex
    #: A dictionary mapping Path codes to the number of vertices that the
    #: code expects.
    NUM_VERTICES_FOR_CODE = {STOP: 1,
                             MOVETO: 1,
                             LINETO: 1,
                             CURVE3: 2,
                             CURVE4: 3,
                             CLOSEPOLY: 1}

    @classmethod
    def _fast_from_codes_and_verts(cls, verts, codes, internals_from=None):
        """
        Create a Path instance without the expense of calling the constructor.

        Parameters
        ----------
        verts : array-like
            Array-like object containing vertex coordinates.
        codes : array
            Array of path codes defining different types of path segments.
        internals_from : Path or None
            If not None, another `Path` from which specific attributes
            will be copied.

            - ``should_simplify``: Whether the path should be simplified.
            - ``simplify_threshold``: Threshold for simplifying the path.
            - ``interpolation_steps``: Number of steps for path interpolation.

            Note: ``readonly`` attribute is always set to ``False``.
        """
        # Create a new instance of the Path class without calling its constructor
        pth = cls.__new__(cls)
        # Convert verts to an unmasked float array and assign to _vertices attribute
        pth._vertices = _to_unmasked_float_array(verts)
        # Assign codes array to _codes attribute
        pth._codes = codes
        # Set _readonly attribute to False as this is a new instance
        pth._readonly = False
        # Copy attributes from internals_from if it's not None
        if internals_from is not None:
            pth._should_simplify = internals_from._should_simplify
            pth._simplify_threshold = internals_from._simplify_threshold
            pth._interpolation_steps = internals_from._interpolation_steps
        else:
            # Set default values if internals_from is None
            pth._should_simplify = True
            pth._simplify_threshold = mpl.rcParams['path.simplify_threshold']
            pth._interpolation_steps = 1
        return pth

    @classmethod
    def _create_closed(cls, vertices):
        """
        Create a closed polygonal path going through *vertices*.

        Unlike ``Path(..., closed=True)``, *vertices* should **not** end with
        an entry for the CLOSEPATH; this entry is added by `._create_closed`.
        """
        # Convert vertices to unmasked float array
        v = _to_unmasked_float_array(vertices)
        # Concatenate vertices with the first vertex to create a closed path
        return cls(np.concatenate([v, v[:1]]), closed=True)

    def _update_values(self):
        """
        Update internal values based on current vertex data and matplotlib
        configuration settings.
        """
        # Update simplify threshold based on matplotlib configuration
        self._simplify_threshold = mpl.rcParams['path.simplify_threshold']
        # Determine whether path should be simplified based on conditions
        self._should_simplify = (
            self._simplify_threshold > 0 and
            mpl.rcParams['path.simplify'] and
            len(self._vertices) >= 128 and
            (self._codes is None or np.all(self._codes <= Path.LINETO))
        )

    @property
    def vertices(self):
        """
        The vertices of the `Path` as an (N, 2) array.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        """
        Setter for vertices property. Sets new vertices and updates internal
        values accordingly.

        Parameters
        ----------
        vertices : array-like
            New vertices to set for the `Path`.
        """
        # Check if path is readonly, raise exception if it is
        if self._readonly:
            raise AttributeError("Can't set vertices on a readonly Path")
        # Set new vertices
        self._vertices = vertices
        # Update internal values based on new vertices
        self._update_values()

    @property
    def codes(self):
        """
        The list of codes in the `Path` as a 1D array.

        Each code is one of `STOP`, `MOVETO`, `LINETO`, `CURVE3`, `CURVE4` or
        `CLOSEPOLY`.  For codes that correspond to more than one vertex
        (`CURVE3` and `CURVE4`), that code will be repeated so that the length
        of `vertices` and `codes` is always the same.
        """
        # 返回路径中的代码列表作为一维数组
        return self._codes

    @codes.setter
    def codes(self, codes):
        # 如果路径是只读的，抛出属性错误异常
        if self._readonly:
            raise AttributeError("Can't set codes on a readonly Path")
        # 设置路径的代码
        self._codes = codes
        # 更新路径的值
        self._update_values()

    @property
    def simplify_threshold(self):
        """
        The fraction of a pixel difference below which vertices will
        be simplified out.
        """
        # 返回简化阈值，即顶点之间的像素差异分数
        return self._simplify_threshold

    @simplify_threshold.setter
    def simplify_threshold(self, threshold):
        # 设置简化阈值
        self._simplify_threshold = threshold

    @property
    def should_simplify(self):
        """
        `True` if the vertices array should be simplified.
        """
        # 如果应简化顶点数组，则返回 True
        return self._should_simplify

    @should_simplify.setter
    def should_simplify(self, should_simplify):
        # 设置是否应简化顶点数组
        self._should_simplify = should_simplify

    @property
    def readonly(self):
        """
        `True` if the `Path` is read-only.
        """
        # 如果路径是只读的，则返回 True
        return self._readonly

    def copy(self):
        """
        Return a shallow copy of the `Path`, which will share the
        vertices and codes with the source `Path`.
        """
        # 返回路径的浅拷贝，该拷贝将与源路径共享顶点和代码
        return copy.copy(self)

    def __deepcopy__(self, memo=None):
        """
        Return a deepcopy of the `Path`.  The `Path` will not be
        readonly, even if the source `Path` is.
        """
        # 深度拷贝路径。即使源路径是只读的，新路径也不会是只读的。
        # 深度拷贝数组（vertices, codes）会去除 writeable=False 标志。
        p = copy.deepcopy(super(), memo)
        p._readonly = False
        return p

    deepcopy = __deepcopy__

    @classmethod
    def make_compound_path_from_polys(cls, XY):
        """
        Make a compound `Path` object to draw a number of polygons with equal
        numbers of sides.

        .. plot:: gallery/misc/histogram_path.py

        Parameters
        ----------
        XY : (numpolys, numsides, 2) array
        """
        # 创建复合路径对象，用于绘制多个具有相同边数的多边形

        # 获取输入数组的形状信息
        numpolys, numsides, two = XY.shape
        if two != 2:
            raise ValueError("The third dimension of 'XY' must be 2")
        
        # 计算步长和顶点数量
        stride = numsides + 1
        nverts = numpolys * stride
        
        # 初始化顶点数组
        verts = np.zeros((nverts, 2))
        
        # 初始化代码数组，并设置起始点和闭合点的代码
        codes = np.full(nverts, cls.LINETO, dtype=cls.code_type)
        codes[0::stride] = cls.MOVETO
        codes[numsides::stride] = cls.CLOSEPOLY
        
        # 将多边形的顶点复制到顶点数组中
        for i in range(numsides):
            verts[i::stride] = XY[:, i]
        
        # 返回创建的路径对象
        return cls(verts, codes)

    @classmethod
    # 定义一个类方法，用于将多个 Path 对象连接成一个单一的 Path 对象，并移除所有 STOP 标记。
    def make_compound_path(cls, *args):
        """
        Concatenate a list of `Path`\s into a single `Path`, removing all `STOP`\s.
        """
        # 如果没有传入参数，则返回一个空的 Path 对象
        if not args:
            return Path(np.empty([0, 2], dtype=np.float32))
        
        # 从所有传入的 Path 对象中提取顶点数据，并拼接成一个数组
        vertices = np.concatenate([path.vertices for path in args])
        
        # 创建一个空的代码数组，用于存储路径指令
        codes = np.empty(len(vertices), dtype=cls.code_type)
        i = 0
        
        # 遍历每个传入的 Path 对象
        for path in args:
            size = len(path.vertices)
            # 如果当前 Path 对象没有指令代码，则默认为 MOVETO 和 LINETO
            if path.codes is None:
                if size:
                    codes[i] = cls.MOVETO
                    codes[i+1:i+size] = cls.LINETO
            else:
                # 否则，将当前 Path 对象的代码复制到整体代码数组中
                codes[i:i+size] = path.codes
            i += size
        
        # 创建一个掩码数组，标记所有不是 STOP 的指令
        not_stop_mask = codes != cls.STOP  # Remove STOPs, as internal STOPs are a bug.
        
        # 根据掩码数组过滤顶点和指令数组，创建一个新的 Path 对象并返回
        return cls(vertices[not_stop_mask], codes[not_stop_mask])

    # 返回 Path 对象的字符串表示形式，包括顶点和代码数组的信息
    def __repr__(self):
        return f"Path({self.vertices!r}, {self.codes!r})"

    # 返回 Path 对象的顶点数组的长度
    def __len__(self):
        return len(self.vertices)
    def iter_bezier(self, **kwargs):
        """
        Iterate over each Bézier curve (lines included) in a `Path`.

        Parameters
        ----------
        **kwargs
            Forwarded to `.iter_segments`.

        Yields
        ------
        B : `~matplotlib.bezier.BezierSegment`
            The Bézier curves that make up the current path. Note in particular
            that freestanding points are Bézier curves of order 0, and lines
            are Bézier curves of order 1 (with two control points).
        code : `~matplotlib.path.Path.code_type`
            The code describing what kind of curve is being returned.
            `MOVETO`, `LINETO`, `CURVE3`, and `CURVE4` correspond to
            Bézier curves with 1, 2, 3, and 4 control points (respectively).
            `CLOSEPOLY` is a `LINETO` with the control points correctly
            chosen based on the start/end points of the current stroke.
        """
        first_vert = None  # 初始化首个顶点为空
        prev_vert = None  # 初始化前一个顶点为空
        for verts, code in self.iter_segments(**kwargs):  # 遍历路径中的段落和代码类型
            if first_vert is None:
                if code != Path.MOVETO:
                    raise ValueError("Malformed path, must start with MOVETO.")
            if code == Path.MOVETO:  # 如果是 MOVETO 类型的路径段
                first_vert = verts  # 记录首个顶点
                yield BezierSegment(np.array([first_vert])), code  # 生成 Bézier 曲线段并返回
            elif code == Path.LINETO:  # 如果是 LINETO 类型的路径段
                yield BezierSegment(np.array([prev_vert, verts])), code  # 生成 Bézier 曲线段并返回
            elif code == Path.CURVE3:  # 如果是 CURVE3 类型的路径段
                yield BezierSegment(np.array([prev_vert, verts[:2],
                                              verts[2:]])), code  # 生成 Bézier 曲线段并返回
            elif code == Path.CURVE4:  # 如果是 CURVE4 类型的路径段
                yield BezierSegment(np.array([prev_vert, verts[:2],
                                              verts[2:4], verts[4:]])), code  # 生成 Bézier 曲线段并返回
            elif code == Path.CLOSEPOLY:  # 如果是 CLOSEPOLY 类型的路径段
                yield BezierSegment(np.array([prev_vert, first_vert])), code  # 生成 Bézier 曲线段并返回
            elif code == Path.STOP:  # 如果是 STOP 类型的路径段
                return  # 结束迭代
            else:
                raise ValueError(f"Invalid Path.code_type: {code}")  # 抛出异常，无效的路径段类型
            prev_vert = verts[-2:]  # 更新前一个顶点

    def _iter_connected_components(self):
        """Return subpaths split at MOVETOs."""
        if self.codes is None:  # 如果路径的代码为空
            yield self  # 返回当前路径对象
        else:
            idxs = np.append((self.codes == Path.MOVETO).nonzero()[0], len(self.codes))  # 找到 MOVETO 的索引位置
            for sl in map(slice, idxs, idxs[1:]):  # 遍历索引范围
                yield Path._fast_from_codes_and_verts(
                    self.vertices[sl], self.codes[sl], self)  # 返回由代码和顶点快速生成的路径对象
    # 返回一个新的 `Path` 对象，其中顶点和代码根据给定的参数进行清理和转换。
    # 如果 `transform` 不为 None，则使用指定的变换进行变换。
    # 如果 `remove_nans` 为 True，则移除顶点中的 NaN 值。
    # 如果 `clip` 不为 None，则裁剪路径。
    # `simplify` 和 `curves` 控制是否简化路径和是否保留曲线。
    # `stroke_width` 控制路径的笔画宽度。
    # 如果 `snap` 为 True，则路径顶点会被吸附到网格上。
    # `sketch` 是一个可选参数，用于指定草图风格。
    def cleaned(self, transform=None, remove_nans=False, clip=None,
                *, simplify=False, curves=False,
                stroke_width=1.0, snap=False, sketch=None):
        """
        Return a new `Path` with vertices and codes cleaned according to the
        parameters.

        See Also
        --------
        Path.iter_segments : for details of the keyword arguments.
        """
        # 调用内部函数 `_path.cleanup_path` 进行路径清理和转换，返回清理后的顶点和代码。
        vertices, codes = _path.cleanup_path(
            self, transform, remove_nans, clip, snap, stroke_width, simplify,
            curves, sketch)
        # 使用类方法 `_fast_from_codes_and_verts` 创建一个新的 `Path` 对象。
        pth = Path._fast_from_codes_and_verts(vertices, codes, self)
        # 如果 `simplify` 参数为 False，则禁用简化选项。
        if not simplify:
            pth._should_simplify = False
        # 返回处理后的 `Path` 对象。
        return pth

    # 返回一个路径的变换副本。
    # 使用给定的 `transform` 对路径的顶点进行变换。
    def transformed(self, transform):
        """
        Return a transformed copy of the path.

        See Also
        --------
        matplotlib.transforms.TransformedPath
            A specialized path class that will cache the transformed result and
            automatically update when the transform changes.
        """
        # 使用给定的 `transform` 对路径的顶点进行变换，并返回新的 `Path` 对象。
        return Path(transform.transform(self.vertices), self.codes,
                    self._interpolation_steps)
    def contains_point(self, point, transform=None, radius=0.0):
        """
        Return whether the area enclosed by the path contains the given point.

        The path is always treated as closed; i.e. if the last code is not
        `CLOSEPOLY` an implicit segment connecting the last vertex to the first
        vertex is assumed.

        Parameters
        ----------
        point : (float, float)
            The point (x, y) to check.
        transform : `~matplotlib.transforms.Transform`, optional
            If not ``None``, *point* will be compared to ``self`` transformed
            by *transform*; i.e. for a correct check, *transform* should
            transform the path into the coordinate system of *point*.
        radius : float, default: 0
            Additional margin on the path in coordinates of *point*.
            The path is extended tangentially by *radius/2*; i.e. if you would
            draw the path with a linewidth of *radius*, all points on the line
            would still be considered to be contained in the area. Conversely,
            negative values shrink the area: Points on the imaginary line
            will be considered outside the area.

        Returns
        -------
        bool

        Notes
        -----
        The current algorithm has some limitations:

        - The result is undefined for points exactly at the boundary
          (i.e. at the path shifted by *radius/2*).
        - The result is undefined if there is no enclosed area, i.e. all
          vertices are on a straight line.
        - If bounding lines start to cross each other due to *radius* shift,
          the result is not guaranteed to be correct.
        """
        # 如果提供了 transform 参数，则先冻结 transform
        if transform is not None:
            transform = transform.frozen()
        
        # 如果 transform 存在且不是仿射变换，则手动进行路径变换
        # 如果 transform 是仿射变换，则让 `point_in_path` 处理变换以避免额外的缓冲区分配
        if transform and not transform.is_affine:
            # 手动变换路径
            self = transform.transform_path(self)
            transform = None
        
        # 调用底层函数 `_path.point_in_path` 判断点是否在路径内部
        return _path.point_in_path(point[0], point[1], radius, self, transform)
    def contains_points(self, points, transform=None, radius=0.0):
        """
        Return whether the area enclosed by the path contains the given points.

        The path is always treated as closed; i.e. if the last code is not
        `CLOSEPOLY` an implicit segment connecting the last vertex to the first
        vertex is assumed.

        Parameters
        ----------
        points : (N, 2) array
            The points to check. Columns contain x and y values.
        transform : `~matplotlib.transforms.Transform`, optional
            If not ``None``, *points* will be compared to ``self`` transformed
            by *transform*; i.e. for a correct check, *transform* should
            transform the path into the coordinate system of *points*.
        radius : float, default: 0
            Additional margin on the path in coordinates of *points*.
            The path is extended tangentially by *radius/2*; i.e. if you would
            draw the path with a linewidth of *radius*, all points on the line
            would still be considered to be contained in the area. Conversely,
            negative values shrink the area: Points on the imaginary line
            will be considered outside the area.

        Returns
        -------
        length-N bool array

        Notes
        -----
        The current algorithm has some limitations:

        - The result is undefined for points exactly at the boundary
          (i.e. at the path shifted by *radius/2*).
        - The result is undefined if there is no enclosed area, i.e. all
          vertices are on a straight line.
        - If bounding lines start to cross each other due to *radius* shift,
          the result is not guaranteed to be correct.
        """
        # 如果提供了 transform 参数，确保其为不可变状态
        if transform is not None:
            transform = transform.frozen()
        # 调用底层方法 _path.points_in_path，检查 points 是否在路径中
        result = _path.points_in_path(points, radius, self, transform)
        # 将结果转换为布尔类型数组并返回
        return result.astype('bool')

    def contains_path(self, path, transform=None):
        """
        Return whether this (closed) path completely contains the given path.

        If *transform* is not ``None``, the path will be transformed before
        checking for containment.
        """
        # 如果提供了 transform 参数，确保其为不可变状态
        if transform is not None:
            transform = transform.frozen()
        # 调用底层方法 _path.path_in_path，检查当前路径是否完全包含给定路径 path
        return _path.path_in_path(self, None, path, transform)
    def get_extents(self, transform=None, **kwargs):
        """
        Get Bbox of the path.

        Parameters
        ----------
        transform : `~matplotlib.transforms.Transform`, optional
            Transform to apply to path before computing extents, if any.
        **kwargs
            Forwarded to `.iter_bezier`.

        Returns
        -------
        matplotlib.transforms.Bbox
            The extents of the path Bbox([[xmin, ymin], [xmax, ymax]])
        """
        # 导入 Bbox 类
        from .transforms import Bbox
        # 如果给定了 transform，将当前路径对象按照该 transform 进行变换
        if transform is not None:
            self = transform.transform_path(self)
        # 如果路径对象的代码（codes）为空，则直接使用顶点（vertices）
        if self.codes is None:
            xys = self.vertices
        # 如果路径对象包含曲线的代码（codes），且没有包含三次或四次贝塞尔曲线的代码
        elif len(np.intersect1d(self.codes, [Path.CURVE3, Path.CURVE4])) == 0:
            # 对于直线情况的优化处理
            # 不迭代每个曲线，而是考虑每条线段的端点
            # （忽略 STOP 和 CLOSEPOLY 顶点）
            xys = self.vertices[np.isin(self.codes,
                                        [Path.MOVETO, Path.LINETO])]
        else:
            xys = []
            # 迭代路径对象中的贝塞尔曲线，计算可能的极值点
            for curve, code in self.iter_bezier(**kwargs):
                # 曲线导数为零的位置可能是极值点
                _, dzeros = curve.axis_aligned_extrema()
                # 曲线的端点也可能是极值点
                xys.append(curve([0, *dzeros, 1]))
            xys = np.concatenate(xys)
        # 如果存在顶点，则返回其最小和最大边界框
        if len(xys):
            return Bbox([xys.min(axis=0), xys.max(axis=0)])
        else:
            # 如果顶点不存在，则返回空边界框
            return Bbox.null()

    def intersects_path(self, other, filled=True):
        """
        Return whether if this path intersects another given path.

        If *filled* is True, then this also returns True if one path completely
        encloses the other (i.e., the paths are treated as filled).
        """
        # 调用 C 扩展的函数来检查路径是否相交
        return _path.path_intersects_path(self, other, filled)

    def intersects_bbox(self, bbox, filled=True):
        """
        Return whether this path intersects a given `~.transforms.Bbox`.

        If *filled* is True, then this also returns True if the path completely
        encloses the `.Bbox` (i.e., the path is treated as filled).

        The bounding box is always considered filled.
        """
        # 调用 C 扩展的函数来检查路径是否与给定的边界框相交
        return _path.path_intersects_rectangle(
            self, bbox.x0, bbox.y0, bbox.x1, bbox.y1, filled)
    def interpolated(self, steps):
        """
        Return a new path resampled to length N x *steps*.

        Codes other than `LINETO` are not handled correctly.
        """
        # 如果 steps 等于 1，则直接返回原路径对象 self
        if steps == 1:
            return self

        # 使用 simple_linear_interpolation 函数对路径上的顶点进行线性插值
        vertices = simple_linear_interpolation(self.vertices, steps)
        codes = self.codes

        # 如果路径有指令码（codes），则处理指令码的插值
        if codes is not None:
            # 创建一个新的指令码数组 new_codes，长度为 (len(codes) - 1) * steps + 1
            # 并且初始值全部设为 Path.LINETO，数据类型为 self.code_type
            new_codes = np.full((len(codes) - 1) * steps + 1, Path.LINETO,
                                dtype=self.code_type)
            # 将原始的指令码 codes 按照步长 steps 进行插值赋值给 new_codes
            new_codes[0::steps] = codes
        else:
            new_codes = None

        # 返回一个新的 Path 对象，使用插值后的顶点 vertices 和新的指令码 new_codes
        return Path(vertices, new_codes)

    def to_polygons(self, transform=None, width=0, height=0, closed_only=True):
        """
        Convert this path to a list of polygons or polylines.  Each
        polygon/polyline is an (N, 2) array of vertices.  In other words,
        each polygon has no `MOVETO` instructions or curves.  This
        is useful for displaying in backends that do not support
        compound paths or Bézier curves.

        If *width* and *height* are both non-zero then the lines will
        be simplified so that vertices outside of (0, 0), (width,
        height) will be clipped.

        The resulting polygons will be simplified if the
        :attr:`Path.should_simplify` attribute of the path is `True`.

        If *closed_only* is `True` (default), only closed polygons,
        with the last point being the same as the first point, will be
        returned.  Any unclosed polylines in the path will be
        explicitly closed.  If *closed_only* is `False`, any unclosed
        polygons in the path will be returned as unclosed polygons,
        and the closed polygons will be returned explicitly closed by
        setting the last point to the same as the first point.
        """
        # 如果顶点数为零，则返回空列表
        if len(self.vertices) == 0:
            return []

        # 如果有 transform 参数，则冻结该 transform
        if transform is not None:
            transform = transform.frozen()

        # 如果路径没有指令码并且 width 或 height 为 0，则直接返回顶点列表
        if self.codes is None and (width == 0 or height == 0):
            vertices = self.vertices
            # 如果 closed_only 为 True，并且顶点数小于 3 或者第一个顶点不等于最后一个顶点，则返回空列表
            if closed_only:
                if len(vertices) < 3:
                    return []
                elif np.any(vertices[0] != vertices[-1]):
                    vertices = [*vertices, vertices[0]]

            # 如果 transform 为 None，则直接返回顶点列表的列表
            if transform is None:
                return [vertices]
            else:
                return [transform.transform(vertices)]

        # 处理包含曲线和/或多个子路径的情况，调用 _path.convert_path_to_polygons 方法处理
        return _path.convert_path_to_polygons(
            self, transform, width, height, closed_only)

    _unit_rectangle = None

    @classmethod
    def unit_rectangle(cls):
        """
        Return a `Path` instance of the unit rectangle from (0, 0) to (1, 1).
        """
        # 如果类属性 `_unit_rectangle` 为空，创建一个表示单位矩形的 Path 实例并存储在 `_unit_rectangle` 中
        if cls._unit_rectangle is None:
            cls._unit_rectangle = cls([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
                                      closed=True, readonly=True)
        # 返回存储在 `_unit_rectangle` 中的单位矩形 Path 实例
        return cls._unit_rectangle

    _unit_regular_polygons = WeakValueDictionary()

    @classmethod
    def unit_regular_polygon(cls, numVertices):
        """
        Return a :class:`Path` instance for a unit regular polygon with the
        given *numVertices* such that the circumscribing circle has radius 1.0,
        centered at (0, 0).
        """
        # 如果 numVertices 小于等于 16，尝试从 `_unit_regular_polygons` 中获取已存储的 Path 实例
        if numVertices <= 16:
            path = cls._unit_regular_polygons.get(numVertices)
        else:
            path = None
        # 如果未找到已存储的 Path 实例，则创建一个新的单位正多边形 Path 实例并存储在 `_unit_regular_polygons` 中
        if path is None:
            theta = ((2 * np.pi / numVertices) * np.arange(numVertices + 1)
                     # 这个初始旋转确保多边形始终“指向上方”
                     + np.pi / 2)
            verts = np.column_stack((np.cos(theta), np.sin(theta)))
            path = cls(verts, closed=True, readonly=True)
            if numVertices <= 16:
                cls._unit_regular_polygons[numVertices] = path
        # 返回创建或获取的单位正多边形 Path 实例
        return path

    _unit_regular_stars = WeakValueDictionary()

    @classmethod
    def unit_regular_star(cls, numVertices, innerCircle=0.5):
        """
        Return a :class:`Path` for a unit regular star with the given
        numVertices and radius of 1.0, centered at (0, 0).
        """
        # 如果 numVertices 小于等于 16，尝试从 `_unit_regular_stars` 中获取已存储的 Path 实例
        if numVertices <= 16:
            path = cls._unit_regular_stars.get((numVertices, innerCircle))
        else:
            path = None
        # 如果未找到已存储的 Path 实例，则创建一个新的单位正星形 Path 实例并存储在 `_unit_regular_stars` 中
        if path is None:
            ns2 = numVertices * 2
            theta = (2*np.pi/ns2 * np.arange(ns2 + 1))
            # 这个初始旋转确保星形始终“指向上方”
            theta += np.pi / 2.0
            r = np.ones(ns2 + 1)
            r[1::2] = innerCircle
            verts = (r * np.vstack((np.cos(theta), np.sin(theta)))).T
            path = cls(verts, closed=True, readonly=True)
            if numVertices <= 16:
                cls._unit_regular_stars[(numVertices, innerCircle)] = path
        # 返回创建或获取的单位正星形 Path 实例
        return path

    @classmethod
    def unit_regular_asterisk(cls, numVertices):
        """
        Return a :class:`Path` for a unit regular asterisk with the given
        numVertices and radius of 1.0, centered at (0, 0).
        """
        # 调用 `unit_regular_star` 方法创建一个单位正星形 Path 实例
        return cls.unit_regular_star(numVertices, 0.0)

    _unit_circle = None

    @classmethod
    def unit_circle(cls):
        """
        Return the readonly :class:`Path` of the unit circle.

        For most cases, :func:`Path.circle` will be what you want.
        """
        # 如果 `_unit_circle` 为空，调用 `circle` 方法创建一个表示单位圆的 Path 实例并存储在 `_unit_circle` 中
        if cls._unit_circle is None:
            cls._unit_circle = cls.circle(center=(0, 0), radius=1,
                                          readonly=True)
        # 返回存储在 `_unit_circle` 中的单位圆 Path 实例
        return cls._unit_circle
    @classmethod
    # 定义一个类方法，用于创建一个表示具有给定半径和中心的圆的路径对象

    def circle(cls, center=(0., 0.), radius=1., readonly=False):
        """
        Return a `Path` representing a circle of a given radius and center.

        Parameters
        ----------
        center : (float, float), default: (0, 0)
            The center of the circle.
        radius : float, default: 1
            The radius of the circle.
        readonly : bool
            Whether the created path should have the "readonly" argument
            set when creating the Path instance.

        Notes
        -----
        The circle is approximated using 8 cubic Bézier curves, as described in

          Lancaster, Don.  `Approximating a Circle or an Ellipse Using Four
          Bezier Cubic Splines <https://www.tinaja.com/glib/ellipse4.pdf>`_.
        """
        MAGIC = 0.2652031
        SQRTHALF = np.sqrt(0.5)
        MAGIC45 = SQRTHALF * MAGIC

        vertices = np.array([[0.0, -1.0],
                             [MAGIC, -1.0],
                             [SQRTHALF-MAGIC45, -SQRTHALF-MAGIC45],
                             [SQRTHALF, -SQRTHALF],
                             [SQRTHALF+MAGIC45, -SQRTHALF+MAGIC45],
                             [1.0, -MAGIC],
                             [1.0, 0.0],
                             [1.0, MAGIC],
                             [SQRTHALF+MAGIC45, SQRTHALF-MAGIC45],
                             [SQRTHALF, SQRTHALF],
                             [SQRTHALF-MAGIC45, SQRTHALF+MAGIC45],
                             [MAGIC, 1.0],
                             [0.0, 1.0],
                             [-MAGIC, 1.0],
                             [-SQRTHALF+MAGIC45, SQRTHALF+MAGIC45],
                             [-SQRTHALF, SQRTHALF],
                             [-SQRTHALF-MAGIC45, SQRTHALF-MAGIC45],
                             [-1.0, MAGIC],
                             [-1.0, 0.0],
                             [-1.0, -MAGIC],
                             [-SQRTHALF-MAGIC45, -SQRTHALF+MAGIC45],
                             [-SQRTHALF, -SQRTHALF],
                             [-SQRTHALF+MAGIC45, -SQRTHALF-MAGIC45],
                             [-MAGIC, -1.0],
                             [0.0, -1.0],
                             [0.0, -1.0]],
                            dtype=float)
        # 定义圆的轮廓顶点数组

        codes = [cls.CURVE4] * 26
        # 创建长度为 26 的曲线代码数组，每个元素都是曲线类型 CURVE4

        codes[0] = cls.MOVETO
        # 将第一个代码设置为 MOVETO，表示移动到指定点开始曲线绘制
        codes[-1] = cls.CLOSEPOLY
        # 将最后一个代码设置为 CLOSEPOLY，表示闭合路径

        return Path(vertices * radius + center, codes, readonly=readonly)
        # 返回一个 Path 对象，其中顶点根据给定半径和中心进行缩放和偏移，使用指定的代码和 readonly 参数创建路径

    _unit_circle_righthalf = None

    @classmethod
    def unit_circle_righthalf(cls):
        """
        Return a `Path` of the right half of a unit circle.

        See `Path.circle` for the reference on the approximation used.
        """
        # 检查是否已经计算过右半单位圆路径，避免重复计算
        if cls._unit_circle_righthalf is None:
            MAGIC = 0.2652031
            SQRTHALF = np.sqrt(0.5)
            MAGIC45 = SQRTHALF * MAGIC

            # 定义右半单位圆的顶点数组
            vertices = np.array(
                [[0.0, -1.0],                     # 底部中心点
                 [MAGIC, -1.0],                   # 右下角边缘
                 [SQRTHALF-MAGIC45, -SQRTHALF-MAGIC45],  # 右下角曲线连接点
                 [SQRTHALF, -SQRTHALF],           # 右侧中心点

                 [SQRTHALF+MAGIC45, -SQRTHALF+MAGIC45],  # 右上角曲线连接点
                 [1.0, -MAGIC],                   # 右上角边缘
                 [1.0, 0.0],                      # 顶部中心点

                 [1.0, MAGIC],                    # 右上角边缘
                 [SQRTHALF+MAGIC45, SQRTHALF-MAGIC45],  # 右上角曲线连接点
                 [SQRTHALF, SQRTHALF],            # 右侧中心点

                 [SQRTHALF-MAGIC45, SQRTHALF+MAGIC45],  # 右下角曲线连接点
                 [MAGIC, 1.0],                    # 右下角边缘
                 [0.0, 1.0],                      # 底部中心点

                 [0.0, -1.0]],                    # 返回底部中心点以闭合路径

                float)

            # 定义路径指令，使其成为闭合的曲线
            codes = np.full(14, cls.CURVE4, dtype=cls.code_type)
            codes[0] = cls.MOVETO  # 将起始点设置为MOVETO
            codes[-1] = cls.CLOSEPOLY  # 将路径闭合设置为CLOSEPOLY

            # 使用定义的顶点和路径指令创建一个 `Path` 对象，设置为只读
            cls._unit_circle_righthalf = cls(vertices, codes, readonly=True)
        return cls._unit_circle_righthalf

    @classmethod
    @classmethod
    def arc(cls, theta1, theta2, n=None, is_wedge=False):
        """
        Return a `Path` for the unit circle arc from angles *theta1* to
        *theta2* (in degrees).

        *theta2* is unwrapped to produce the shortest arc within 360 degrees.
        That is, if *theta2* > *theta1* + 360, the arc will be from *theta1* to
        *theta2* - 360 and not a full circle plus some extra overlap.

        If *n* is provided, it is the number of spline segments to make.
        If *n* is not provided, the number of spline segments is
        determined based on the delta between *theta1* and *theta2*.

           Masionobe, L.  2003.  `Drawing an elliptical arc using
           polylines, quadratic or cubic Bezier curves
           <https://web.archive.org/web/20190318044212/http://www.spaceroots.org/documents/ellipse/index.html>`_.
        """
        halfpi = np.pi * 0.5

        # Adjust angles to ensure they are within a 360-degree range
        eta1 = theta1
        eta2 = theta2 - 360 * np.floor((theta2 - theta1) / 360)
        if theta2 != theta1 and eta2 <= eta1:
            eta2 += 360
        # Convert angles from degrees to radians
        eta1, eta2 = np.deg2rad([eta1, eta2])

        # Determine number of spline segments if not provided
        if n is None:
            n = int(2 ** np.ceil((eta2 - eta1) / halfpi))
        # Ensure at least one segment is used
        if n < 1:
            raise ValueError("n must be >= 1 or None")

        # Calculate angular step
        deta = (eta2 - eta1) / n
        # Compute alpha for Bezier curve approximation
        t = np.tan(0.5 * deta)
        alpha = np.sin(deta) * (np.sqrt(4.0 + 3.0 * t * t) - 1) / 3.0

        # Generate points along the arc using cosine and sine of angular steps
        steps = np.linspace(eta1, eta2, n + 1, True)
        cos_eta = np.cos(steps)
        sin_eta = np.sin(steps)

        # Initialize arrays for control points of the Bezier curves
        xA = cos_eta[:-1]
        yA = sin_eta[:-1]
        xA_dot = -yA
        yA_dot = xA

        xB = cos_eta[1:]
        yB = sin_eta[1:]
        xB_dot = -yB
        yB_dot = xB

        # Construct vertices and codes for the Path object
        if is_wedge:
            # For wedge, prepare longer array for vertex and codes
            length = n * 3 + 4
            vertices = np.zeros((length, 2), float)
            codes = np.full(length, cls.CURVE4, dtype=cls.code_type)
            # Set initial move and line-to points
            vertices[1] = [xA[0], yA[0]]
            codes[0:2] = [cls.MOVETO, cls.LINETO]
            # Set closing line-to and close polygon codes
            codes[-2:] = [cls.LINETO, cls.CLOSEPOLY]
            vertex_offset = 2
            end = length - 2
        else:
            # For non-wedge, prepare shorter array
            length = n * 3 + 1
            vertices = np.empty((length, 2), float)
            codes = np.full(length, cls.CURVE4, dtype=cls.code_type)
            # Set initial move-to point
            vertices[0] = [xA[0], yA[0]]
            codes[0] = cls.MOVETO
            vertex_offset = 1
            end = length

        # Assign computed Bezier control points to vertices array
        vertices[vertex_offset:end:3, 0] = xA + alpha * xA_dot
        vertices[vertex_offset:end:3, 1] = yA + alpha * yA_dot
        vertices[vertex_offset+1:end:3, 0] = xB - alpha * xB_dot
        vertices[vertex_offset+1:end:3, 1] = yB - alpha * yB_dot
        vertices[vertex_offset+2:end:3, 0] = xB
        vertices[vertex_offset+2:end:3, 1] = yB

        # Return a Path object initialized with computed vertices and codes
        return cls(vertices, codes, readonly=True)
    def wedge(cls, theta1, theta2, n=None):
        """
        返回一个 `Path` 对象，代表从角度 *theta1* 到 *theta2* 的单位圆扇形。

        *theta2* 会被展开以保证在 360 度内得到最短的扇形。
        也就是说，如果 *theta2* > *theta1* + 360，扇形将从 *theta1* 到 *theta2* - 360，而不是一个完整的圆再加上额外的重叠部分。

        如果提供了 *n*，则生成 *n* 个样条段的扇形。
        如果没有提供 *n*，则基于 *theta1* 和 *theta2* 的角度差确定样条段的数量。

        参见 `Path.arc` 查看所使用的近似方法。
        """
        return cls.arc(theta1, theta2, n, True)

    @staticmethod
    @lru_cache(8)
    def hatch(hatchpattern, density=6):
        """
        给定填充样式 *hatchpattern*，生成一个可用于重复填充的 `Path` 对象。
        *density* 是每单位正方形中线条的数量。
        """
        from matplotlib.hatch import get_path
        return (get_path(hatchpattern, density)
                if hatchpattern is not None else None)

    def clip_to_bbox(self, bbox, inside=True):
        """
        将路径剪裁到给定的边界框 *bbox*。

        路径必须由一个或多个闭合多边形组成。这个算法对于非闭合路径不会产生正确的结果。

        如果 *inside* 是 `True`，则剪裁到边界框内部；否则剪裁到边界框外部。
        """
        verts = _path.clip_path_to_rect(self, bbox, inside)
        paths = [Path(poly) for poly in verts]
        return self.make_compound_path(*paths)
def get_path_collection_extents(
        master_transform, paths, transforms, offsets, offset_transform):
    r"""
    获取 `.PathCollection` 内部对象的边界框。

    即，给定一个 `Path` 序列，`.Transform` 对象，以及偏移量，如 `.PathCollection` 中所发现的，
    返回一个包围它们所有的边界框。

    Parameters
    ----------
    master_transform : `~matplotlib.transforms.Transform`
        应用于所有路径的全局变换。
    paths : list of `Path`
        包含要计算边界框的路径列表。
    transforms : list of `~matplotlib.transforms.Affine2DBase`
        如果非空，将覆盖 *master_transform*。
        每个路径对应的变换对象。
    offsets : (N, 2) array-like
        每个路径的偏移量列表。
    offset_transform : `~matplotlib.transforms.Affine2DBase`
        应用于偏移量的变换对象。

    Notes
    -----
    *paths*, *transforms* 和 *offsets* 的组合方式与集合的方式相同：
    每个对象都会独立迭代，因此如果有3个路径 (A, B, C)，2个变换 (α, β) 和 1个偏移量 (O)，它们的组合如下：

    - (A, α, O)
    - (B, β, O)
    - (C, α, O)
    """
    from .transforms import Bbox
    # 如果没有提供路径，则抛出 ValueError 异常
    if len(paths) == 0:
        raise ValueError("No paths provided")
    # 如果没有提供偏移量，则发出警告
    if len(offsets) == 0:
        _api.warn_deprecated(
            "3.8", message="Calling get_path_collection_extents() with an"
            " empty offsets list is deprecated since %(since)s. Support will"
            " be removed %(removal)s.")
    # 调用内部方法计算路径集合的边界框
    extents, minpos = _path.get_path_collection_extents(
        master_transform, paths, np.atleast_3d(transforms),
        offsets, offset_transform)
    # 从计算得到的边界框参数构建 Bbox 对象，并返回
    return Bbox.from_extents(*extents, minpos=minpos)
```