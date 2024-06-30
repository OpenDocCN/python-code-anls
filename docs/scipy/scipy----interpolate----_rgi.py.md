# `D:\src\scipysrc\scipy\scipy\interpolate\_rgi.py`

```
# 定义一个公开的变量列表，列出模块中可以公开访问的类和函数名
__all__ = ['RegularGridInterpolator', 'interpn']

# 导入 itertools 库，用于生成迭代器
import itertools

# 导入 numpy 库，并将其重命名为 np，用于支持多维数组和数学函数
import numpy as np

# 导入 scipy.sparse.linalg 库，将其重命名为 ssl，用于稀疏矩阵运算
import scipy.sparse.linalg as ssl

# 导入当前包中的模块 interpnd，引入 _ndim_coords_from_arrays 函数
from .interpnd import _ndim_coords_from_arrays

# 导入当前包中的 _cubic 模块，引入 PchipInterpolator 类
from ._cubic import PchipInterpolator

# 导入当前包中的 _rgi_cython 模块，引入 evaluate_linear_2d 和 find_indices 函数
from ._rgi_cython import evaluate_linear_2d, find_indices

# 导入当前包中的 _bsplines 模块，引入 make_interp_spline 函数
from ._bsplines import make_interp_spline

# 导入当前包中的 _fitpack2 模块，引入 RectBivariateSpline 类
from ._fitpack2 import RectBivariateSpline

# 导入当前包中的 _ndbspline 模块，引入 make_ndbspl 函数
from ._ndbspline import make_ndbspl


def _check_points(points):
    # 用于检查给定点集合的维度和顺序，并做必要的调整，以确保点的顺序是升序或降序的
    descending_dimensions = []
    grid = []
    for i, p in enumerate(points):
        # 将每个点集转换为浮点类型的 numpy 数组
        p = np.asarray(p, dtype=float)
        # 检查点集是否严格升序或严格降序，如果不是则抛出 ValueError 异常
        if not np.all(p[1:] > p[:-1]):
            if np.all(p[1:] < p[:-1]):
                # 如果是降序，则将其翻转为升序
                descending_dimensions.append(i)
                p = np.flip(p)
            else:
                raise ValueError(
                    "The points in dimension %d must be strictly "
                    "ascending or descending" % i)
        # 将点集转换为连续的内存布局，以优化计算
        p = np.ascontiguousarray(p)
        grid.append(p)
    # 返回调整后的点集合和降序维度列表
    return tuple(grid), tuple(descending_dimensions)


def _check_dimensionality(points, values):
    # 检查给定点集合和值的维度是否匹配
    if len(points) > values.ndim:
        # 如果点集合的数量大于值的维度，则抛出 ValueError 异常
        raise ValueError("There are %d point arrays, but values has %d "
                         "dimensions" % (len(points), values.ndim))
    for i, p in enumerate(points):
        # 检查每个维度的点集是否为一维数组
        if not np.asarray(p).ndim == 1:
            raise ValueError("The points in dimension %d must be "
                             "1-dimensional" % i)
        # 检查每个维度的点集数量是否与值的对应维度长度相同
        if not values.shape[i] == len(p):
            raise ValueError("There are %d points and %d values in "
                             "dimension %d" % (len(p), values.shape[i], i))


class RegularGridInterpolator:
    """
    多维正则网格插值器。

    在任意维度上的正则或矩形网格上进行插值。
    数据必须在一个矩形网格上定义，即具有均匀或不均匀的间距。
    支持线性、最近邻、样条插值。在设置插值器对象后，可以选择每次评估的插值方法。

    Parameters
    ----------
    points : 元组，其中每个元素是形状为 (m1, ), ..., (mn, ) 的 float 类型的 ndarray
        定义 n 维正则网格的点。每个维度上的点（即 points 元组的每个元素）必须是严格升序或降序排列。

    values : array_like，形状为 (m1, ..., mn, ...)
        n 维正则网格上的数据。接受复杂数据。

    method : str，可选
        要执行的插值方法。支持 "linear"、"nearest"、"slinear"、"cubic"、"quintic" 和 "pchip"。
        此参数将成为对象的 ``__call__`` 方法的默认值。默认为 "linear"。
    """
    
    def __init__(self, points, values, method='linear'):
        pass  # 在初始化时不执行任何操作，具体的初始化在子类中实现
    bounds_error : bool, optional
        # 控制是否在请求超出输入数据域的插值值时引发 ValueError
        # True 表示引发 ValueError，False 表示使用 fill_value
        Default is True.

    fill_value : float or None, optional
        # 插值域外的点所使用的值
        # 如果为 None，则对域外的值进行外推
        Default is ``np.nan``.

    solver : callable, optional
        # 仅适用于方法 "slinear", "cubic" 和 "quintic"
        # 用于 NdBSpline 实例构造的稀疏线性代数求解器
        # 默认为迭代求解器 `scipy.sparse.linalg.gcrotmk`
        .. versionadded:: 1.13

    solver_args: dict, optional
        # 传递给 `solver` 的额外参数
        .. versionadded:: 1.13

    Methods
    -------
    __call__

    Attributes
    ----------
    grid : tuple of ndarrays
        # 定义 n 维正则网格的点
        # 通过 `np.meshgrid(*grid, indexing='ij')` 定义完整网格
    values : ndarray
        # 网格上的数据值
    method : str
        # 插值方法
    fill_value : float or ``None``
        # 用于 `__call__` 的域外参数的值
    bounds_error : bool
        # 如果为 ``True``，则超出边界的参数会引发 ``ValueError``

    Notes
    -----
    # 与 `LinearNDInterpolator` 和 `NearestNDInterpolator` 相反，此类
    # 通过利用正则网格结构避免了输入数据的昂贵三角化。
    # 换句话说，此类假设数据在*矩形*网格上定义。

    .. versionadded:: 0.14

    # 'slinear'(k=1), 'cubic'(k=3), 和 'quintic'(k=5) 方法是
    # 张量积样条插值器，其中 `k` 是样条度数，
    # 如果任何维度的点少于 `k` + 1，将会引发错误。

    .. versionadded:: 1.9

    # 如果输入数据的维度具有不相容的单位并且相差多个数量级，
    # 插值器可能会产生数值人为现象。考虑在插值之前对数据进行重新缩放。

    **Choosing a solver for spline methods**

    # Spline 方法 "slinear", "cubic" 和 "quintic" 在实例化时涉及解决
    # 大型稀疏线性系统。默认求解器可能或可能不足够。当不足时，您可以
    # 尝试使用可选的 `solver` 参数，您可以在直接求解器 (`scipy.sparse.linalg.spsolve`)
    # 或 `scipy.sparse.linalg` 的迭代求解器之间选择。您可能需要通过
    # 可选的 `solver_args` 参数提供额外的参数（例如，您可以提供起始值或目标容差）。
    # 查看 `scipy.sparse.linalg` 文档以获取所有可用选项的完整列表。

    Alternatively, you may instead use the legacy methods, "slinear_legacy",
    """
    Evaluate a function on a points of a 3-D grid and interpolate using RegularGridInterpolator.
    
    Examples
    --------
    **Evaluate a function on the points of a 3-D grid**
    
    As a first example, we evaluate a simple example function on the points of
    a 3-D grid:
    
    >>> from scipy.interpolate import RegularGridInterpolator
    >>> import numpy as np
    >>> def f(x, y, z):
    ...     return 2 * x**3 + 3 * y**2 - z
    >>> x = np.linspace(1, 4, 11)
    >>> y = np.linspace(4, 7, 22)
    >>> z = np.linspace(7, 9, 33)
    >>> xg, yg ,zg = np.meshgrid(x, y, z, indexing='ij', sparse=True)
    >>> data = f(xg, yg, zg)
    
    ``data`` is now a 3-D array with ``data[i, j, k] = f(x[i], y[j], z[k])``.
    Next, define an interpolating function from this data:
    
    >>> interp = RegularGridInterpolator((x, y, z), data)
    
    Evaluate the interpolating function at the two points
    ``(x,y,z) = (2.1, 6.2, 8.3)`` and ``(3.3, 5.2, 7.1)``:
    
    >>> pts = np.array([[2.1, 6.2, 8.3],
    ...                 [3.3, 5.2, 7.1]])
    >>> interp(pts)
    array([ 125.80469388,  146.30069388])
    
    which is indeed a close approximation to
    
    >>> f(2.1, 6.2, 8.3), f(3.3, 5.2, 7.1)
    (125.54200000000002, 145.894)
    
    **Interpolate and extrapolate a 2D dataset**
    
    As a second example, we interpolate and extrapolate a 2D data set:
    
    >>> x, y = np.array([-2, 0, 4]), np.array([-2, 0, 2, 5])
    >>> def ff(x, y):
    ...     return x**2 + y**2
    
    >>> xg, yg = np.meshgrid(x, y, indexing='ij')
    >>> data = ff(xg, yg)
    >>> interp = RegularGridInterpolator((x, y), data,
    ...                                  bounds_error=False, fill_value=None)
    
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(projection='3d')
    >>> ax.scatter(xg.ravel(), yg.ravel(), data.ravel(),
    ...            s=60, c='k', label='data')
    
    Evaluate and plot the interpolator on a finer grid
    
    >>> xx = np.linspace(-4, 9, 31)
    >>> yy = np.linspace(-4, 9, 31)
    >>> X, Y = np.meshgrid(xx, yy, indexing='ij')
    
    >>> # interpolator
    >>> ax.plot_wireframe(X, Y, interp((X, Y)), rstride=3, cstride=3,
    ...                   alpha=0.4, color='m', label='linear interp')
    
    >>> # ground truth
    >>> ax.plot_wireframe(X, Y, ff(X, Y), rstride=3, cstride=3,
    ...                   alpha=0.4, label='ground truth')
    >>> plt.legend()
    >>> plt.show()
    
    Other examples are given
    :ref:`in the tutorial <tutorial-interpolate_regular_grid_interpolator>`.
    
    See Also
    --------
    NearestNDInterpolator : Nearest neighbor interpolator on *unstructured*
                            data in N dimensions
    
    LinearNDInterpolator : Piecewise linear interpolator on *unstructured* data
                           in N dimensions
    
    interpn : a convenience function which wraps `RegularGridInterpolator`
    """
    """
    scipy.ndimage.map_coordinates : interpolation on grids with equal spacing
                                    (suitable for e.g., N-D image resampling)

    References
    ----------
    .. [1] Python package *regulargrid* by Johannes Buchner, see
           https://pypi.python.org/pypi/regulargrid/
    .. [2] Wikipedia, "Trilinear interpolation",
           https://en.wikipedia.org/wiki/Trilinear_interpolation
    .. [3] Weiser, Alan, and Sergio E. Zarantonello. "A note on piecewise linear
           and multilinear table interpolation in many dimensions." MATH.
           COMPUT. 50.181 (1988): 189-196.
           https://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf
           :doi:`10.1090/S0025-5718-1988-0917826-0`

    """
    # this class is based on code originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid

    # 映射不同插值方法到对应的样条度数
    _SPLINE_DEGREE_MAP = {"slinear": 1, "cubic": 3, "quintic": 5, 'pchip': 3,
                          "slinear_legacy": 1, "cubic_legacy": 3, "quintic_legacy": 5,}
    # 递归插值方法集合
    _SPLINE_METHODS_recursive = {"slinear_legacy", "cubic_legacy",
                                "quintic_legacy", "pchip"}
    # 非递归插值方法集合
    _SPLINE_METHODS_ndbspl = {"slinear", "cubic", "quintic"}
    # 所有插值方法的列表
    _SPLINE_METHODS = list(_SPLINE_DEGREE_MAP.keys())
    # 所有支持的插值方法列表
    _ALL_METHODS = ["linear", "nearest"] + _SPLINE_METHODS
    # 初始化函数，用于创建 RegularGridInterpolator 对象
    def __init__(self, points, values, method="linear", bounds_error=True,
                 fill_value=np.nan, *, solver=None, solver_args=None):
        # 检查所选的插值方法是否合法，如果不合法则引发 ValueError 异常
        if method not in self._ALL_METHODS:
            raise ValueError("Method '%s' is not defined" % method)
        # 如果方法属于需要检查网格维度的插值方法，则调用 _validate_grid_dimensions 方法进行检查
        elif method in self._SPLINE_METHODS:
            self._validate_grid_dimensions(points, method)
        
        # 将传入的参数赋值给对象属性
        self.method = method
        self.bounds_error = bounds_error
        # 调用 _check_points 函数，检查并返回处理后的网格点集合及其降序维度信息
        self.grid, self._descending_dimensions = _check_points(points)
        # 调用 _check_values 函数，检查并返回处理后的值集合
        self.values = self._check_values(values)
        # 调用 _check_dimensionality 函数，检查网格和值的维度是否匹配
        self._check_dimensionality(self.grid, self.values)
        # 调用 _check_fill_value 函数，检查并返回处理后的填充值
        self.fill_value = self._check_fill_value(self.values, fill_value)
        
        # 如果存在降序维度，则对值进行翻转操作以匹配 RegularGridInterpolator 的要求
        if self._descending_dimensions:
            self.values = np.flip(values, axis=self._descending_dimensions)
        
        # 如果选择的方法是 "pchip"，且值集合包含复数，则引发 ValueError 异常
        if self.method == "pchip" and np.iscomplexobj(self.values):
            msg = ("`PchipInterpolator` only works with real values. If you are trying "
                   "to use the real components of the passed array, use `np.real` on "
                   "the array before passing to `RegularGridInterpolator`.")
            raise ValueError(msg)
        
        # 如果方法属于需要特定求解器的插值方法，则调用 _construct_spline 方法创建对应的样条对象
        if method in self._SPLINE_METHODS_ndbspl:
            if solver_args is None:
                solver_args = {}
            self._spline = self._construct_spline(method, solver, **solver_args)
        else:
            # 对于不需要求解器的方法，若传入了 solver 参数或 solver_args 参数，则引发 ValueError 异常
            if solver is not None or solver_args:
                raise ValueError(
                    f"{method =} does not accept the 'solver' argument. Got "
                    f" {solver = } and with arguments {solver_args}."
                )

    # 创建样条插值对象的私有方法
    def _construct_spline(self, method, solver=None, **solver_args):
        # 如果未指定 solver 参数，则默认使用 ssl.gcrotmk
        if solver is None:
            solver = ssl.gcrotmk
        # 调用 make_ndbspl 函数创建样条插值对象 spl
        spl = make_ndbspl(
                self.grid, self.values, self._SPLINE_DEGREE_MAP[method],
                solver=solver, **solver_args
              )
        return spl

    # 检查网格和值的维度是否匹配的私有方法
    def _check_dimensionality(self, grid, values):
        _check_dimensionality(grid, values)

    # 检查网格点集合的私有方法
    def _check_points(self, points):
        return _check_points(points)

    # 检查值集合的私有方法
    def _check_values(self, values):
        # 如果 values 对象缺少 ndim 属性，则将其转换为合理的 numpy 数组
        if not hasattr(values, 'ndim'):
            # 允许合理的鸭子类型值
            values = np.asarray(values)

        # 如果 values 对象同时具有 dtype 和 astype 方法，则检查其 dtype 是否为浮点类型，若不是则转换为浮点类型
        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        return values

    # 检查填充值的私有方法
    def _check_fill_value(self, values, fill_value):
        # 如果 fill_value 不为 None，则检查其 dtype 是否与 values 兼容，若不兼容则引发 ValueError 异常
        if fill_value is not None:
            fill_value_dtype = np.asarray(fill_value).dtype
            if (hasattr(values, 'dtype') and not
                    np.can_cast(fill_value_dtype, values.dtype,
                                casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")
        return fill_value
    def _prepare_xi(self, xi):
        # 计算网格的维度
        ndim = len(self.grid)
        # 将输入的 xi 转换为与网格维度相同的坐标数组
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        # 检查 xi 的最后一个维度是否与网格的维度相同，若不同则抛出数值错误异常
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             f"{xi.shape[-1]} but this "
                             f"RegularGridInterpolator has dimension {ndim}")

        # 记录原始的 xi 形状，并将 xi 重新塑造为二维数组
        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])
        xi = np.asarray(xi, dtype=float)

        # 在输入的 xi 中查找 NaN 值
        nans = np.any(np.isnan(xi), axis=-1)

        # 如果 bounds_error 为 True，则检查是否所有的 xi 都在网格范围内
        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    raise ValueError("One of the requested xi is out of bounds "
                                     "in dimension %d" % i)
            out_of_bounds = None
        else:
            # 否则，使用 _find_out_of_bounds 方法查找超出边界的 xi
            out_of_bounds = self._find_out_of_bounds(xi.T)

        # 返回处理后的 xi、其形状、网格的维度、NaN 值的索引以及超出边界的 xi
        return xi, xi_shape, ndim, nans, out_of_bounds

    def _evaluate_linear(self, indices, norm_distances):
        # 对 self.values 的尾随维度进行广播切片
        vslice = (slice(None),) + (None,)*(self.values.ndim - len(indices))

        # 预先计算偏移量，以便在组合所有值之前进行
        shift_norm_distances = [1 - yi for yi in norm_distances]
        shift_indices = [i + 1 for i in indices]

        # 线性插值的公式在二维情况下为:
        # values = self.values[(i0, i1)] * (1 - y0) * (1 - y1) + \
        #          self.values[(i0, i1 + 1)] * (1 - y0) * y1 + \
        #          self.values[(i0 + 1, i1)] * y0 * (1 - y1) + \
        #          self.values[(i0 + 1, i1 + 1)] * y0 * y1
        # 我们将 i 与 1 - yi（zipped1）配对，并将 i + 1 与 yi（zipped2）配对
        zipped1 = zip(indices, shift_norm_distances)
        zipped2 = zip(shift_indices, norm_distances)

        # 取 zipped1 和 zipped2 的所有乘积，并迭代获取上述公式中的各项
        hypercube = itertools.product(*zip(zipped1, zipped2))
        value = np.array([0.])
        for h in hypercube:
            edge_indices, weights = zip(*h)
            weight = np.array([1.])
            for w in weights:
                weight = weight * w
            term = np.asarray(self.values[edge_indices]) * weight[vslice]
            value = value + term   # 不能使用 += 因为需要进行广播
        return value

    def _evaluate_nearest(self, indices, norm_distances):
        # 根据 norm_distances 中的值选择最近的索引
        idx_res = [np.where(yi <= .5, i, i + 1)
                   for i, yi in zip(indices, norm_distances)]
        # 返回最近索引对应的 self.values 的值
        return self.values[tuple(idx_res)]
    # 验证网格维度是否符合要求，根据指定方法确定每个维度的阶数
    def _validate_grid_dimensions(self, points, method):
        k = self._SPLINE_DEGREE_MAP[method]
        # 遍历所有点，确保每个点的维度大于指定方法所需的最小维度
        for i, point in enumerate(points):
            ndim = len(np.atleast_1d(point))
            if ndim <= k:
                raise ValueError(f"There are {ndim} points in dimension {i},"
                                 f" but method {method} requires at least "
                                 f" {k+1} points per dimension.")

    # 对样点进行样条插值计算
    def _evaluate_spline(self, xi, method):
        # 确保 xi 是一个二维列表，每个元素是要评估的点（`m` 是点的数量，`n` 是插值维度的数量，
        # ``n == len(self.grid)``）
        if xi.ndim == 1:
            xi = xi.reshape((1, xi.size))
        m, n = xi.shape

        # 重新排序轴：n 维度的过程沿着插值轴从最后一个轴向下迭代：例如，对于一个四维网格，
        # 轴的顺序是 3, 2, 1, 0。每个一维插值沿着其参数数组的第 0 轴工作（对于一维例程，它是其 ``y`` 数组）。
        # 因此，重新排列 `values` 的插值轴 *并保持尾部维度不变*。
        axes = tuple(range(self.values.ndim))
        axx = axes[:n][::-1] + axes[n:]
        values = self.values.transpose(axx)

        if method == 'pchip':
            _eval_func = self._do_pchip
        else:
            _eval_func = self._do_spline_fit
        k = self._SPLINE_DEGREE_MAP[method]

        # 非定态过程：难以完全向量化这部分到 numpy 操作层。不幸的是，这需要显式循环遍历 xi 中的每个点。

        # 至少可以向量化对 xi 中所有点的第一个通道的第一次传递。
        last_dim = n - 1
        first_values = _eval_func(self.grid[last_dim],
                                  values,
                                  xi[:, last_dim],
                                  k)

        # 其余维度必须基于每个 xi 中的点进行处理
        shape = (m, *self.values.shape[n:])
        result = np.empty(shape, dtype=self.values.dtype)
        for j in range(m):
            # 主要过程：依次在每个维度上应用一维插值，从最后一个维度开始。
            # 然后将这些折叠到下一个维度中。
            folded_values = first_values[j, ...]
            for i in range(last_dim-1, -1, -1):
                # 对每个一维从最后维度进行插值。这将每个一维序列折叠成一个标量。
                folded_values = _eval_func(self.grid[i],
                                           folded_values,
                                           xi[j, i],
                                           k)
            result[j, ...] = folded_values

        return result
    # 执行样条插值拟合，使用给定的 x 和 y 数据进行插值
    def _do_spline_fit(x, y, pt, k):
        # 创建一个样条插值对象，使用指定的次数 k
        local_interp = make_interp_spline(x, y, k=k, axis=0)
        # 对指定的 pt 进行插值计算
        values = local_interp(pt)
        return values

    # 执行 PCHIP 插值拟合，使用给定的 x 和 y 数据进行插值
    @staticmethod
    def _do_pchip(x, y, pt, k):
        # 创建一个 PCHIP 插值对象
        local_interp = PchipInterpolator(x, y, axis=0)
        # 对指定的 pt 进行插值计算
        values = local_interp(pt)
        return values

    # 在网格中查找给定 xi 对应的索引
    def _find_indices(self, xi):
        return find_indices(self.grid, xi)

    # 查找超出边界的 xi
    def _find_out_of_bounds(self, xi):
        # 创建一个布尔数组，用于标记是否超出边界
        out_of_bounds = np.zeros((xi.shape[1]), dtype=bool)
        # 遍历每个维度的 xi 和对应的网格
        for x, grid in zip(xi, self.grid):
            # 检查每个维度的 x 是否小于网格的最小值或大于网格的最大值
            out_of_bounds += x < grid[0]
            out_of_bounds += x > grid[-1]
        return out_of_bounds
# 定义多维插值函数，用于在规则或矩形网格上进行插值

def interpn(points, values, xi, method="linear", bounds_error=True,
            fill_value=np.nan):
    """
    Multidimensional interpolation on regular or rectilinear grids.

    Strictly speaking, not all regular grids are supported - this function
    works on *rectilinear* grids, that is, a rectangular grid with even or
    uneven spacing.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        定义 n 维规则网格的点。每个维度上的点（即 points 元组的每个元素）必须是严格升序或降序排列。

    values : array_like, shape (m1, ..., mn, ...)
        n 维规则网格上的数据。可以接受复杂数据。

        .. deprecated:: 1.13.0
            使用 ``method="pchip"`` 时，复杂数据已弃用，并将在 SciPy 1.15.0 中引发错误。
            这是因为 ``PchipInterpolator`` 仅适用于实数值。如果要使用传递数组的实部，请在 ``values`` 上使用 ``np.real``。

    xi : ndarray of shape (..., ndim)
        在其上采样格网数据的坐标点集

    method : str, optional
        要执行的插值方法。支持的方法有 "linear", "nearest", "slinear", "cubic", "quintic", "pchip" 和 "splinef2d"。
        "splinef2d" 仅支持二维数据。

    bounds_error : bool, optional
        如果为 True，则当请求插值值超出输入数据的定义域时，引发 ValueError。
        如果为 False，则使用 `fill_value`。

    fill_value : number, optional
        如果提供，则用于插值域外的点的值。如果为 None，则对域外的值进行外推。
        "splinef2d" 方法不支持外推。

    Returns
    -------
    values_x : ndarray, shape xi.shape[:-1] + values.shape[ndim:]
        在 `xi` 上的插值值。当 ``xi.ndim == 1`` 时，请参阅备注中的行为。

    See Also
    --------
    NearestNDInterpolator : 在 N 维非结构化数据上进行最近邻插值
    LinearNDInterpolator : 在 N 维非结构化数据上进行分段线性插值
    RegularGridInterpolator : 在任意维度上的规则或矩形网格上进行插值（`interpn` 封装了这个类）。
    RectBivariateSpline : 矩形网格上的双变量样条近似
    scipy.ndimage.map_coordinates : 在等间距网格上进行插值（适用于例如 N 维图像重采样）

    Notes
    -----

    .. versionadded:: 0.14

    如果 ``xi.ndim == 1``，则在返回数组 values_x 的 0 位置插入一个新轴，使其形状为
    """
    # 检查 'method' 参数是否合法
    if method not in ["linear", "nearest", "cubic", "quintic", "pchip",
                      "splinef2d", "slinear",
                      "slinear_legacy", "cubic_legacy", "quintic_legacy"]:
        raise ValueError("interpn 只支持 'linear', 'nearest', 'slinear', "
                         "'cubic', 'quintic', 'pchip' 和 'splinef2d' 方法。"
                         f"你提供的方法为 {method}。")

    # 如果 values 没有 'ndim' 属性，则将其转换为 NumPy 数组
    if not hasattr(values, 'ndim'):
        values = np.asarray(values)

    # 检查数据维度 ndim
    ndim = values.ndim

    # 如果 ndim 大于 2 并且方法为 "splinef2d"，则抛出异常
    if ndim > 2 and method == "splinef2d":
        raise ValueError("方法 'splinef2d' 仅适用于二维输入数据")

    # 如果 bounds_error 为 False，fill_value 为 None，并且方法为 "splinef2d"，则抛出异常
    if not bounds_error and fill_value is None and method == "splinef2d":
        raise ValueError("方法 'splinef2d' 不支持外推（extrapolation）")

    # 检查输入维度的一致性
    if len(points) > ndim:
        raise ValueError("有 %d 个点数组，但是 values 有 %d 维度" % (len(points), ndim))

    # 如果点数组的数量不等于数据维度，并且方法为 'splinef2d'，则抛出异常
    if len(points) != ndim and method == 'splinef2d':
        raise ValueError("方法 'splinef2d' 仅适用于每个坐标一个标量数据")

    # 检查网格和降序维度
    grid, descending_dimensions = _check_points(points)

    # 检查网格和 values 的维度
    _check_dimensionality(grid, values)

    # 检查请求的 xi 坐标
    xi = _ndim_coords_from_arrays(xi, ndim=len(grid))
    if xi.shape[-1] != len(grid):
        raise ValueError("请求的样本点 xi 的维度为 %d，但此 RegularGridInterpolator "
                         "的维度为 %d" % (xi.shape[-1], len(grid)))

    # 如果 bounds_error 为 True，则检查 xi 是否在边界内
    if bounds_error:
        for i, p in enumerate(xi.T):
            if not np.logical_and(np.all(grid[i][0] <= p),
                                  np.all(p <= grid[i][-1])):
                raise ValueError("请求的 xi 中的第 %d 维超出了边界" % i)

    # 执行插值操作
    # 检查插值方法是否在支持的方法列表中
    if method in RegularGridInterpolator._ALL_METHODS:
        # 使用 RegularGridInterpolator 进行多维网格插值
        interp = RegularGridInterpolator(points, values, method=method,
                                         bounds_error=bounds_error,
                                         fill_value=fill_value)
        # 对给定的点集 xi 进行插值并返回结果
        return interp(xi)
    # 如果插值方法是 "splinef2d"
    elif method == "splinef2d":
        # 记录原始 xi 的形状
        xi_shape = xi.shape
        # 将 xi 展平为二维数组
        xi = xi.reshape(-1, xi.shape[-1])

        # 检查 xi 是否在网格的有效范围内
        idx_valid = np.all((grid[0][0] <= xi[:, 0], xi[:, 0] <= grid[0][-1],
                            grid[1][0] <= xi[:, 1], xi[:, 1] <= grid[1][-1]),
                           axis=0)
        # 创建一个与 xi[:, 0] 大小相同的空数组
        result = np.empty_like(xi[:, 0])

        # 使用 RectBivariateSpline 进行二维矩形样条插值
        interp = RectBivariateSpline(points[0], points[1], values[:])
        # 对有效的 xi 进行插值，并将结果保存在 result 中
        result[idx_valid] = interp.ev(xi[idx_valid, 0], xi[idx_valid, 1])
        # 对非有效的 xi 设置为 fill_value
        result[np.logical_not(idx_valid)] = fill_value

        # 将 result 重新调整为原始 xi 的形状，除去最后一个维度
        return result.reshape(xi_shape[:-1])
    else:
        # 如果方法不在支持列表中，则引发 ValueError 异常
        raise ValueError(f"unknown {method = }")
```