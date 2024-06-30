# `D:\src\scipysrc\scipy\scipy\interpolate\_ndgriddata.py`

```
"""
Convenience interface to N-D interpolation

.. versionadded:: 0.9

"""
# 导入必要的库和模块
import numpy as np
from .interpnd import LinearNDInterpolator, NDInterpolatorBase, \
     CloughTocher2DInterpolator, _ndim_coords_from_arrays
from scipy.spatial import cKDTree

# 定义公开的接口列表
__all__ = ['griddata', 'NearestNDInterpolator', 'LinearNDInterpolator',
           'CloughTocher2DInterpolator']

#------------------------------------------------------------------------------
# Nearest-neighbor interpolation
#------------------------------------------------------------------------------

# NearestNDInterpolator 类，继承自 NDInterpolatorBase 类
class NearestNDInterpolator(NDInterpolatorBase):
    """NearestNDInterpolator(x, y).

    Nearest-neighbor interpolator in N > 1 dimensions.

    .. versionadded:: 0.9

    Methods
    -------
    __call__

    Parameters
    ----------
    x : (npoints, ndims) 2-D ndarray of floats
        Data point coordinates.
    y : (npoints, ) 1-D ndarray of float or complex
        Data values.
    rescale : boolean, optional
        Rescale points to unit cube before performing interpolation.
        This is useful if some of the input dimensions have
        incommensurable units and differ by many orders of magnitude.

        .. versionadded:: 0.14.0
    tree_options : dict, optional
        Options passed to the underlying ``cKDTree``.

        .. versionadded:: 0.17.0

    See Also
    --------
    griddata :
        Interpolate unstructured D-D data.
    LinearNDInterpolator :
        Piecewise linear interpolator in N dimensions.
    CloughTocher2DInterpolator :
        Piecewise cubic, C1 smooth, curvature-minimizing interpolator in 2D.
    interpn : Interpolation on a regular grid or rectilinear grid.
    RegularGridInterpolator : Interpolator on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).

    Notes
    -----
    Uses ``scipy.spatial.cKDTree``

    .. note:: For data on a regular grid use `interpn` instead.

    Examples
    --------
    We can interpolate values on a 2D plane:

    >>> from scipy.interpolate import NearestNDInterpolator
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.random(10) - 0.5
    >>> y = rng.random(10) - 0.5
    >>> z = np.hypot(x, y)
    >>> X = np.linspace(min(x), max(x))
    >>> Y = np.linspace(min(y), max(y))
    >>> X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    >>> interp = NearestNDInterpolator(list(zip(x, y)), z)
    >>> Z = interp(X, Y)
    >>> plt.pcolormesh(X, Y, Z, shading='auto')
    >>> plt.plot(x, y, "ok", label="input point")
    >>> plt.legend()
    >>> plt.colorbar()
    >>> plt.axis("equal")
    >>> plt.show()

    """
    # 定义初始化方法，接受参数 x, y 和可选参数 rescale 和 tree_options
    def __init__(self, x, y, rescale=False, tree_options=None):
        # 调用父类 NDInterpolatorBase 的初始化方法，传入 x, y，设置 rescale 标志位，
        # 设置 need_contiguous 和 need_values 为 False
        NDInterpolatorBase.__init__(self, x, y, rescale=rescale,
                                    need_contiguous=False,
                                    need_values=False)
        # 如果 tree_options 参数为 None，则将其设为一个空字典
        if tree_options is None:
            tree_options = dict()
        # 使用给定的点集 x 创建一个 cKDTree 对象，并传入 tree_options 中的参数
        self.tree = cKDTree(self.points, **tree_options)
        # 将 y 转换为 NumPy 数组并赋值给 self.values
        self.values = np.asarray(y)
    def __call__(self, *args, **query_options):
        """
        Evaluate interpolator at given points.

        Parameters
        ----------
        x1, x2, ... xn : array-like of float
            Points where to interpolate data at.
            x1, x2, ... xn can be array-like of float with broadcastable shape.
            or x1 can be array-like of float with shape ``(..., ndim)``
        **query_options
            This allows ``eps``, ``p``, ``distance_upper_bound``, and ``workers``
            being passed to the cKDTree's query function to be explicitly set.
            See `scipy.spatial.cKDTree.query` for an overview of the different options.

            .. versionadded:: 1.12.0

        """
        # For the sake of enabling subclassing, NDInterpolatorBase._set_xi performs
        # some operations which are not required by NearestNDInterpolator.__call__, 
        # hence here we operate on xi directly, without calling a parent class function.
        # 为了支持子类化，NDInterpolatorBase._set_xi 执行了一些不被 NearestNDInterpolator.__call__ 需要的操作，
        # 因此我们在这里直接操作 xi，而不调用父类函数。

        # Convert input arguments to ndim-dimensional coordinates
        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        # Ensure xi has the correct shape for calling _scale_x
        xi = self._check_call_shape(xi)
        # Scale xi coordinates based on the interpolation model's internal scaling
        xi = self._scale_x(xi)

        # We need to handle two important cases:
        # (1) the case where xi has trailing dimensions (..., ndim), and
        # (2) the case where y has trailing dimensions
        # We will first flatten xi to deal with case (1),
        # do the computation in flattened array while retaining y's dimensionality,
        # and then reshape the interpolated values back to match xi's shape.
        
        # Flatten xi for the query
        xi_flat = xi.reshape(-1, xi.shape[-1])
        original_shape = xi.shape
        flattened_shape = xi_flat.shape

        # if distance_upper_bound is set to not be infinite,
        # then we need to consider the case where cKDtree
        # does not find any points within distance_upper_bound to return.
        # It marks those points as having infinte distance, which is what will be used
        # below to mask the array and return only the points that were deemed
        # to have a close enough neighbor to return something useful.
        # 如果 distance_upper_bound 被设置为非无穷大，
        # 那么我们需要考虑 cKDtree 找不到任何在 distance_upper_bound 范围内的点的情况。
        # 它会将这些点标记为具有无限距离，这将被用来掩盖数组，并返回仅被认为足够接近邻居的点，
        # 从而返回有用的内容。

        # Perform a nearest neighbor query on xi_flat using cKDTree
        dist, i = self.tree.query(xi_flat, **query_options)
        # Create a boolean mask where dist is finite (i.e., valid points)
        valid_mask = np.isfinite(dist)

        # create a holder interp_values array and fill with nans.
        # 创建一个 interp_values 数组作为容器，并用 NaN 填充。

        # Determine shape of interp_values based on whether self.values is multi-dimensional
        if self.values.ndim > 1:
            interp_shape = flattened_shape[:-1] + self.values.shape[1:]
        else:
            interp_shape = flattened_shape[:-1]

        # Initialize interp_values array with NaN values of appropriate dtype
        if np.issubdtype(self.values.dtype, np.complexfloating):
            interp_values = np.full(interp_shape, np.nan, dtype=self.values.dtype)
        else:
            interp_values = np.full(interp_shape, np.nan)

        # Assign values from self.values to interp_values where valid_mask is True
        interp_values[valid_mask] = self.values[i[valid_mask], ...]

        # Reshape interp_values back to the original shape of xi
        if self.values.ndim > 1:
            new_shape = original_shape[:-1] + self.values.shape[1:]
        else:
            new_shape = original_shape[:-1]
        interp_values = interp_values.reshape(new_shape)

        # Return the interpolated values
        return interp_values
#------------------------------------------------------------------------------
# Convenience interface function
#------------------------------------------------------------------------------

# 定义一个便捷的接口函数用于数据插值

def griddata(points, values, xi, method='linear', fill_value=np.nan,
             rescale=False):
    """
    Interpolate unstructured D-D data.

    Parameters
    ----------
    points : 2-D ndarray of floats with shape (n, D), or length D tuple of 1-D ndarrays with shape (n,).
        Data point coordinates.
        数据点的坐标，可以是形状为 (n, D) 的二维浮点数组，或者长度为 D 的元组，其中包含形状为 (n,) 的一维数组。
    values : ndarray of float or complex, shape (n,)
        Data values.
        数据的值，形状为 (n,) 的浮点数或复数数组。
    xi : 2-D ndarray of floats with shape (m, D), or length D tuple of ndarrays broadcastable to the same shape.
        Points at which to interpolate data.
        用于插值数据的点的坐标，可以是形状为 (m, D) 的二维浮点数组，或者长度为 D 的元组，包含可以广播为相同形状的数组。
    method : {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation. One of
        插值方法。可选值为 'linear'、'nearest'、'cubic' 之一。

        ``nearest``
          return the value at the data point closest to
          the point of interpolation. See `NearestNDInterpolator` for
          more details.
          返回距离插值点最近的数据点的值。详见 `NearestNDInterpolator`。

        ``linear``
          tessellate the input point set to N-D
          simplices, and interpolate linearly on each simplex. See
          `LinearNDInterpolator` for more details.
          将输入点集镶嵌成 N-D 三角形，对每个三角形进行线性插值。详见 `LinearNDInterpolator`。

        ``cubic`` (1-D)
          return the value determined from a cubic
          spline.

        ``cubic`` (2-D)
          return the value determined from a
          piecewise cubic, continuously differentiable (C1), and
          approximately curvature-minimizing polynomial surface. See
          `CloughTocher2DInterpolator` for more details.
          返回通过分段三次连续可微（C1）和近似曲率最小化多项式曲面确定的值。详见 `CloughTocher2DInterpolator`。
    fill_value : float, optional
        Value used to fill in for requested points outside of the
        convex hull of the input points. If not provided, then the
        default is ``nan``. This option has no effect for the
        'nearest' method.
        用于填充请求点的值，这些点在输入点的凸包外部。如果未提供，则默认为 ``nan``。对 'nearest' 方法无效。
    rescale : bool, optional
        Rescale points to unit cube before performing interpolation.
        This is useful if some of the input dimensions have
        incommensurable units and differ by many orders of magnitude.
        在执行插值之前将点重新缩放到单位立方体。如果输入的一些维度具有不可比较的单位，并且相差多个数量级，则这很有用。

        .. versionadded:: 0.14.0

    Returns
    -------
    ndarray
        Array of interpolated values.
        插值值的数组。

    See Also
    --------
    LinearNDInterpolator :
        Piecewise linear interpolator in N dimensions.
        N 维分段线性插值器。
    NearestNDInterpolator :
        Nearest-neighbor interpolator in N dimensions.
        N 维最近邻插值器。
    CloughTocher2DInterpolator :
        Piecewise cubic, C1 smooth, curvature-minimizing interpolator in 2D.
        二维分段三次连续平滑、曲率最小化插值器。
    interpn : Interpolation on a regular grid or rectilinear grid.
             在规则网格或矩形网格上的插值。
    RegularGridInterpolator : Interpolator on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).
                              在任意维度的规则或矩形网格上的插值器（`interpn` 包装了这个类）。

    Notes
    -----

    .. versionadded:: 0.9

    .. note:: For data on a regular grid use `interpn` instead.
              对于规则网格上的数据，请改用 `interpn`。

    Examples
    --------

    Suppose we want to interpolate the 2-D function

    >>> import numpy as np
    >>> def func(x, y):
    """
    Interpolate unstructured D-D data.
    插值非结构化的 D-D 数据。
    # 定义一个二维函数，该函数在区间[0, 1]x[0, 1]上以 x 和 y 为变量
    # 返回的是表达式 x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2 的值
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

    # 在区间[0, 1]x[0, 1]上创建一个网格，其中 grid_x 和 grid_y 是网格的 x 和 y 坐标
    >>> grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

    # 生成 1000 个随机数据点，每个点有两个坐标，存储在 points 数组中
    >>> rng = np.random.default_rng()
    >>> points = rng.random((1000, 2))
    # 使用之前定义的函数 func 计算这些随机点的函数值，并存储在 values 数组中
    >>> values = func(points[:,0], points[:,1])

    # 使用 griddata 函数对上述数据进行插值，生成三种不同方法（nearest, linear, cubic）的插值结果
    >>> from scipy.interpolate import griddata
    >>> grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
    >>> grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
    >>> grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

    # 绘制四幅子图，展示原始函数和三种插值方法的结果
    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(221)
    >>> plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
    >>> plt.plot(points[:,0], points[:,1], 'k.', ms=1)
    >>> plt.title('Original')
    >>> plt.subplot(222)
    >>> plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
    >>> plt.title('Nearest')
    >>> plt.subplot(223)
    >>> plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
    >>> plt.title('Linear')
    >>> plt.subplot(224)
    >>> plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
    >>> plt.title('Cubic')
    >>> plt.gcf().set_size_inches(6, 6)
    >>> plt.show()

    """ # numpy/numpydoc#87  # noqa: E501

    # 将 points 数组转换为标准化的坐标形式，确保每个坐标点的形状正确
    points = _ndim_coords_from_arrays(points)

    # 确定 points 数组的维度
    if points.ndim < 2:
        ndim = points.ndim
    else:
        ndim = points.shape[-1]

    # 如果 points 是一维数组，并且插值方法为 nearest, linear 或 cubic
    if ndim == 1 and method in ('nearest', 'linear', 'cubic'):
        # 从 _interpolate 模块导入 interp1d 函数
        from ._interpolate import interp1d
        # 将 points 数组展平为一维数组
        points = points.ravel()
        # 如果 xi 是一个元组且长度不为 1，则抛出 ValueError
        if isinstance(xi, tuple):
            if len(xi) != 1:
                raise ValueError("invalid number of dimensions in xi")
            xi, = xi
        # 按照 points 和 values 的顺序排序，这对 interp1d 函数的输入是必需的
        idx = np.argsort(points)
        points = points[idx]
        values = values[idx]
        # 如果使用 nearest 方法，则设置 fill_value 为 'extrapolate'
        if method == 'nearest':
            fill_value = 'extrapolate'
        # 调用 interp1d 函数进行插值，并返回结果
        ip = interp1d(points, values, kind=method, axis=0, bounds_error=False,
                      fill_value=fill_value)
        return ip(xi)
    # 如果插值方法为 nearest，并且 points 的维度大于 1
    elif method == 'nearest':
        # 使用 NearestNDInterpolator 进行多维插值，并返回结果
        ip = NearestNDInterpolator(points, values, rescale=rescale)
        return ip(xi)
    # 如果插值方法为 linear
    elif method == 'linear':
        # 使用 LinearNDInterpolator 进行多维插值，并返回结果
        ip = LinearNDInterpolator(points, values, fill_value=fill_value,
                                  rescale=rescale)
        return ip(xi)
    # 如果插值方法为 cubic，并且 points 的维度为 2
    elif method == 'cubic' and ndim == 2:
        # 使用 CloughTocher2DInterpolator 进行二维插值，并返回结果
        ip = CloughTocher2DInterpolator(points, values, fill_value=fill_value,
                                        rescale=rescale)
        return ip(xi)
    else:
        # 如果不是已知的插值方法，则抛出值错误异常，指明未知的插值方法和数据维度
        raise ValueError("Unknown interpolation method %r for "
                         "%d dimensional data" % (method, ndim))
```