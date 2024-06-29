# `D:\src\scipysrc\matplotlib\lib\matplotlib\tri\_triangulation.py`

```
# 导入系统模块 sys
import sys

# 导入 numpy 库并使用 np 别名
import numpy as np

# 从 matplotlib 库中导入 _api 模块
from matplotlib import _api

# 定义 Triangulation 类，用于表示一个无结构的三角网格
class Triangulation:
    """
    An unstructured triangular grid consisting of npoints points and
    ntri triangles.  The triangles can either be specified by the user
    or automatically generated using a Delaunay triangulation.

    Parameters
    ----------
    x, y : (npoints,) array-like
        Coordinates of grid points.
    triangles : (ntri, 3) array-like of int, optional
        For each triangle, the indices of the three points that make
        up the triangle, ordered in an anticlockwise manner.  If not
        specified, the Delaunay triangulation is calculated.
    mask : (ntri,) array-like of bool, optional
        Which triangles are masked out.

    Attributes
    ----------
    triangles : (ntri, 3) array of int
        For each triangle, the indices of the three points that make
        up the triangle, ordered in an anticlockwise manner. If you want to
        take the *mask* into account, use `get_masked_triangles` instead.
    mask : (ntri, 3) array of bool or None
        Masked out triangles.
    is_delaunay : bool
        Whether the Triangulation is a calculated Delaunay
        triangulation (where *triangles* was not specified) or not.

    Notes
    -----
    For a Triangulation to be valid it must not have duplicate points,
    triangles formed from colinear points, or overlapping triangles.
    """
    def __init__(self, x, y, triangles=None, mask=None):
        # 导入必要的模块
        from matplotlib import _qhull

        # 将输入的 x 和 y 转换为 numpy 的 float64 数组
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)

        # 检查 x 和 y 的形状，必须是相同长度的一维数组
        if self.x.shape != self.y.shape or self.x.ndim != 1:
            raise ValueError("x and y must be equal-length 1D arrays, but "
                             f"found shapes {self.x.shape!r} and "
                             f"{self.y.shape!r}")

        # 初始化 mask、_edges 和 _neighbors
        self.mask = None
        self._edges = None
        self._neighbors = None
        self.is_delaunay = False

        # 如果 triangles 参数为 None，则使用 matplotlib._qhull 获取 Delaunay 三角剖分
        if triangles is None:
            # 使用 _qhull.delaunay 函数获取三角剖分和邻居信息
            self.triangles, self._neighbors = _qhull.delaunay(x, y, sys.flags.verbose)
            self.is_delaunay = True
        else:
            # 如果 triangles 参数不为 None，则复制并检查三角形数组的形状和值范围
            try:
                self.triangles = np.array(triangles, dtype=np.int32, order='C')
            except ValueError as e:
                raise ValueError('triangles must be a (N, 3) int array, not '
                                 f'{triangles!r}') from e
            if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
                raise ValueError(
                    'triangles must be a (N, 3) int array, but found shape '
                    f'{self.triangles.shape!r}')
            if self.triangles.max() >= len(self.x):
                raise ValueError(
                    'triangles are indices into the points and must be in the '
                    f'range 0 <= i < {len(self.x)} but found value '
                    f'{self.triangles.max()}')
            if self.triangles.min() < 0:
                raise ValueError(
                    'triangles are indices into the points and must be in the '
                    f'range 0 <= i < {len(self.x)} but found value '
                    f'{self.triangles.min()}')

        # C++ 对象在首次需要时才会创建
        self._cpp_triangulation = None

        # 默认的 TriFinder 在需要时才会创建
        self._trifinder = None

        # 设置 mask 属性
        self.set_mask(mask)

    def calculate_plane_coefficients(self, z):
        """
        Calculate plane equation coefficients for all unmasked triangles from
        the point (x, y) coordinates and specified z-array of shape (npoints).
        The returned array has shape (npoints, 3) and allows z-value at (x, y)
        position in triangle tri to be calculated using
        ``z = array[tri, 0] * x  + array[tri, 1] * y + array[tri, 2]``.
        """
        # 使用 C++ 三角剖分对象计算所有未屏蔽三角形的平面方程系数
        return self.get_cpp_triangulation().calculate_plane_coefficients(z)

    @property
    def edges(self):
        """
        Return integer array of shape (nedges, 2) containing all edges of
        non-masked triangles.

        Each row defines an edge by its start point index and end point
        index.  Each edge appears only once, i.e. for an edge between points
        *i*  and *j*, there will only be either *(i, j)* or *(j, i)*.
        """
        # 如果尚未计算边，则调用 get_cpp_triangulation() 方法获取三角剖分的边
        if self._edges is None:
            self._edges = self.get_cpp_triangulation().get_edges()
        # 返回存储边的数组
        return self._edges

    def get_cpp_triangulation(self):
        """
        Return the underlying C++ Triangulation object, creating it
        if necessary.
        """
        from matplotlib import _tri
        # 如果尚未创建 C++ Triangulation 对象，则创建并返回它
        if self._cpp_triangulation is None:
            self._cpp_triangulation = _tri.Triangulation(
                # 如果 mask 不为空，则传入 mask；否则传入空元组
                self.x, self.y, self.triangles,
                self.mask if self.mask is not None else (),
                # 如果 _edges 不为 None，则传入 _edges；否则传入空元组
                self._edges if self._edges is not None else (),
                # 如果 _neighbors 不为 None，则传入 _neighbors；否则传入空元组
                self._neighbors if self._neighbors is not None else (),
                # 是否为非 Delaunay 三角剖分
                not self.is_delaunay)
        # 返回 C++ Triangulation 对象
        return self._cpp_triangulation

    def get_masked_triangles(self):
        """
        Return an array of triangles taking the mask into account.
        """
        # 如果存在 mask，则返回剔除 mask 的 triangles；否则直接返回 triangles
        if self.mask is not None:
            return self.triangles[~self.mask]
        else:
            return self.triangles

    @staticmethod
    def get_from_args_and_kwargs(*args, **kwargs):
        """
        Return a Triangulation object from the args and kwargs, and
        the remaining args and kwargs with the consumed values removed.

        There are two alternatives: either the first argument is a
        Triangulation object, in which case it is returned, or the args
        and kwargs are sufficient to create a new Triangulation to
        return.  In the latter case, see Triangulation.__init__ for
        the possible args and kwargs.
        """
        # 如果第一个参数是 Triangulation 对象，则返回该对象；
        # 否则使用参数和关键字参数创建一个新的 Triangulation 对象并返回
        if isinstance(args[0], Triangulation):
            triangulation, *args = args
            # 如果 kwargs 中包含 'triangles' 关键字，发出警告
            if 'triangles' in kwargs:
                _api.warn_external(
                    "Passing the keyword 'triangles' has no effect when also "
                    "passing a Triangulation")
            # 如果 kwargs 中包含 'mask' 关键字，发出警告
            if 'mask' in kwargs:
                _api.warn_external(
                    "Passing the keyword 'mask' has no effect when also "
                    "passing a Triangulation")
        else:
            # 从参数和关键字参数中提取 Triangulation 对象的构造参数
            x, y, triangles, mask, args, kwargs = \
                Triangulation._extract_triangulation_params(args, kwargs)
            # 使用提取的参数和关键字参数创建 Triangulation 对象
            triangulation = Triangulation(x, y, triangles, mask)
        # 返回创建的 Triangulation 对象以及剩余的参数和关键字参数
        return triangulation, args, kwargs
    def _extract_triangulation_params(args, kwargs):
        x, y, *args = args  # 解构参数 args 中的 x 和 y，剩余的参数存放在 args 列表中
        # 从 kwargs 中取出 triangles，如果不存在则为 None
        triangles = kwargs.pop('triangles', None)
        from_args = False
        # 如果 triangles 为 None，并且 args 不为空，则将 args 的第一个元素赋给 triangles，并设置 from_args 为 True
        if triangles is None and args:
            triangles = args[0]
            from_args = True
        # 如果 triangles 不为 None，则尝试将其转换为 np.int32 类型的 NumPy 数组，如果转换失败则将 triangles 设为 None
        if triangles is not None:
            try:
                triangles = np.asarray(triangles, dtype=np.int32)
            except ValueError:
                triangles = None
        # 如果 triangles 不为 None 并且其维度不为 2，或者第二维的大小不为 3，则将 triangles 设为 None
        if triangles is not None and (triangles.ndim != 2 or
                                      triangles.shape[1] != 3):
            triangles = None
        # 如果 triangles 不为 None 并且是从 args 中获取的，则从 args 中移除第一个元素
        if triangles is not None and from_args:
            args = args[1:]  # 消耗了 args 中的第一个元素
        # 从 kwargs 中取出 mask，如果不存在则为 None
        mask = kwargs.pop('mask', None)
        # 返回 x, y, triangles, mask, args, kwargs 这些参数
        return x, y, triangles, mask, args, kwargs

    def get_trifinder(self):
        """
        Return the default `matplotlib.tri.TriFinder` of this
        triangulation, creating it if necessary.  This allows the same
        TriFinder object to be easily shared.
        """
        if self._trifinder is None:
            # 导入默认的 TriFinder 类
            from matplotlib.tri._trifinder import TrapezoidMapTriFinder
            # 创建 TriFinder 对象并将其赋值给 self._trifinder
            self._trifinder = TrapezoidMapTriFinder(self)
        # 返回 self._trifinder
        return self._trifinder

    @property
    def neighbors(self):
        """
        Return integer array of shape (ntri, 3) containing neighbor triangles.

        For each triangle, the indices of the three triangles that
        share the same edges, or -1 if there is no such neighboring
        triangle.  ``neighbors[i, j]`` is the triangle that is the neighbor
        to the edge from point index ``triangles[i, j]`` to point index
        ``triangles[i, (j+1)%3]``.
        """
        if self._neighbors is None:
            # 获取 C++ Triangulation 中的邻居信息并将其赋值给 self._neighbors
            self._neighbors = self.get_cpp_triangulation().get_neighbors()
        # 返回 self._neighbors
        return self._neighbors

    def set_mask(self, mask):
        """
        Set or clear the mask array.

        Parameters
        ----------
        mask : None or bool array of length ntri
        """
        # 如果 mask 为 None，则将 self.mask 设为 None
        if mask is None:
            self.mask = None
        else:
            # 否则将 mask 转换为 np.bool 类型的 NumPy 数组，并检查其形状是否与 triangles 数组的行数相同
            self.mask = np.asarray(mask, dtype=bool)
            if self.mask.shape != (self.triangles.shape[0],):
                raise ValueError('mask array must have same length as '
                                 'triangles array')

        # 如果 self._cpp_triangulation 不为 None，则将 mask 设置到 C++ Triangulation 中
        if self._cpp_triangulation is not None:
            self._cpp_triangulation.set_mask(
                self.mask if self.mask is not None else ())

        # 清空计算得到的衍生字段，以便在需要时重新计算
        self._edges = None
        self._neighbors = None

        # 如果 self._trifinder 不为 None，则重新初始化 TriFinder 对象
        if self._trifinder is not None:
            self._trifinder._initialize()
```