# `D:\src\scipysrc\scipy\scipy\interpolate\interpnd.pyx`

```
# cython: cpow=True
"""
Simple N-D interpolation

.. versionadded:: 0.9

"""
#
# Copyright (C)  Pauli Virtanen, 2010.
#
# Distributed under the same BSD license as Scipy.
#

#
# Note: this file should be run through the Mako template engine before
#       feeding it to Cython.
#
#       Run ``generate_qhull.py`` to regenerate the ``qhull.c`` file
#

# 引入 Cython 的声明
cimport cython

# 从 C 标准库中引入双精度浮点数的极小值和数学函数 fabs, sqrt
from libc.float cimport DBL_EPSILON
from libc.math cimport fabs, sqrt

# 引入 NumPy 库
import numpy as np

# 引入 scipy.spatial._qhull 中的 qhull 模块
import scipy.spatial._qhull as qhull
cimport scipy.spatial._qhull as qhull

# 引入警告模块
import warnings

#------------------------------------------------------------------------------
# Numpy etc.
#------------------------------------------------------------------------------

# 从 NumPy 的头文件中引入 NPY_MAXDIMS 常量
cdef extern from "numpy/ndarrayobject.h":
    cdef enum:
        NPY_MAXDIMS

# 定义一个融合类型 double_or_complex，包括 double 和 double complex
ctypedef fused double_or_complex:
    double
    double complex


#------------------------------------------------------------------------------
# Interpolator base class
#------------------------------------------------------------------------------

# 定义一个多维插值器的基类 NDInterpolatorBase
class NDInterpolatorBase:
    """
    Common routines for interpolators.

    .. versionadded:: 0.9

    """

    # 初始化方法，接受 points, values, fill_value 等参数
    def __init__(self, points, values, fill_value=np.nan, ndim=None,
                 rescale=False, need_contiguous=True, need_values=True):
        """
        Check shape of points and values arrays, and reshape values to
        (npoints, nvalues).  Ensure the `points` and values arrays are
        C-contiguous, and of correct type.
        """

        # 如果 points 是 qhull.Delaunay 类型的对象，则使用预先计算的三角剖分
        if isinstance(points, qhull.Delaunay):
            # 抛出错误，因为不支持 rescale
            if rescale:
                raise ValueError("Rescaling is not supported when passing "
                                 "a Delaunay triangulation as ``points``.")
            # 设置三角剖分对象为传入的 points
            self.tri = points
            # 将 points 设置为 points.points
            points = points.points
        else:
            self.tri = None

        # 调用 _ndim_coords_from_arrays 函数处理 points
        points = _ndim_coords_from_arrays(points)

        # 如果需要连续存储，则将 points 转换为 C 连续存储的 np.float64 类型数组
        if need_contiguous:
            points = np.ascontiguousarray(points, dtype=np.float64)

        # 如果不需要 rescale，则直接设置 points，否则进行尺度调整
        if not rescale:
            self.scale = None
            self.points = points
        else:
            # 将 points 缩放到以 0 为中心的单位立方体
            self.offset = np.mean(points, axis=0)
            self.points = points - self.offset
            self.scale = np.ptp(points, axis=0)
            self.scale[~(self.scale > 0)] = 1.0  # 避免除以 0
            self.points /= self.scale

        # 如果没有传入三角剖分对象，则计算三角剖分
        if self.tri is None:
            self._calculate_triangulation(self.points)

        # 如果需要 values 或者 values 不为 None，则设置 values
        if need_values or values is not None:
            self._set_values(values, fill_value, need_contiguous, ndim)
        else:
            self.values = None

    # 计算三角剖分的方法，暂时为空实现
    def _calculate_triangulation(self, points):
        pass
    # 将传入的值转换为 NumPy 数组
    def _set_values(self, values, fill_value=np.nan, need_contiguous=True, ndim=None):
        values = np.asarray(values)
        # 检查形状是否符合预期
        _check_init_shape(self.points, values, ndim=ndim)

        # 记录值的形状（不包括第一个维度）
        self.values_shape = values.shape[1:]

        # 根据值的维度进行不同的处理
        if values.ndim == 1:
            self.values = values[:, None]  # 将一维数组转换为列向量
        elif values.ndim == 2:
            self.values = values  # 直接使用二维数组
        else:
            # 将多维数组压缩为二维数组
            self.values = values.reshape(values.shape[0], np.prod(values.shape[1:]))

        # 判断数组类型是否为复数
        self.is_complex = np.issubdtype(self.values.dtype, np.complexfloating)
        if self.is_complex:
            if need_contiguous:
                # 将复数数组转换为连续存储的复数数组
                self.values = np.ascontiguousarray(self.values, dtype=np.complex128)
            # 设置填充值为复数类型
            self.fill_value = complex(fill_value)
        else:
            if need_contiguous:
                # 将实数数组转换为连续存储的双精度浮点数数组
                self.values = np.ascontiguousarray(self.values, dtype=np.float64)
            # 设置填充值为浮点数类型
            self.fill_value = float(fill_value)

    # 将输入数组转换为 NumPy 数组
    def _check_call_shape(self, xi):
        xi = np.asanyarray(xi)
        # 检查最后一个维度是否与插值点的维度匹配
        if xi.shape[-1] != self.points.shape[1]:
            raise ValueError("number of dimensions in xi does not match x")
        return xi

    # 如果设置了比例，则对输入数组进行比例缩放
    def _scale_x(self, xi):
        if self.scale is None:
            return xi
        else:
            # 对输入数组进行比例缩放，并进行偏移
            return (xi - self.offset) / self.scale

    # 对输入参数进行预处理，转换为合适的数组形式
    def _preprocess_xi(self, *args):
        # 将输入参数转换为合适维度的坐标数组
        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        # 检查坐标数组的形状是否正确
        xi = self._check_call_shape(xi)
        # 记录插值点的形状
        interpolation_points_shape = xi.shape
        # 将坐标数组转换为二维数组
        xi = xi.reshape(-1, xi.shape[-1])
        # 将坐标数组转换为连续存储的双精度浮点数数组
        xi = np.ascontiguousarray(xi, dtype=np.float64)
        # 对坐标数组进行比例缩放，并返回缩放后的数组和插值点的形状
        return self._scale_x(xi), interpolation_points_shape

    # 调用函数，进行插值计算
    def __call__(self, *args):
        """
        interpolator(xi)

        在给定点进行插值计算。

        Parameters
        ----------
        x1, x2, ... xn: array-like of float
            插值的点。
            x1, x2, ... xn 可以是浮点数的数组，形状可以广播。
            或者 x1 可以是形状为 ``(..., ndim)`` 的浮点数数组。
        """
        # 预处理输入的坐标数组
        xi, interpolation_points_shape = self._preprocess_xi(*args)

        # 判断值数组的类型（复数或实数），并进行相应的插值计算
        if self.is_complex:
            r = self._evaluate_complex(xi)
        else:
            r = self._evaluate_double(xi)

        # 返回插值结果的数组，形状与插值点形状和值数组形状一致
        return np.asarray(r).reshape(interpolation_points_shape[:-1] + self.values_shape)
# 定义一个 CPython 可调用的函数，用于从坐标数组中生成 (..., ndim) 形状的数组
cpdef _ndim_coords_from_arrays(points, ndim=None):
    """
    Convert a tuple of coordinate arrays to a (..., ndim)-shaped array.
    将坐标数组元组转换为形状为 (..., ndim) 的数组。

    """
    # 声明 ssize_t 类型的变量 j 和 n
    cdef ssize_t j, n

    # 如果 points 是元组且长度为 1，则处理为单一的参数数组
    if isinstance(points, tuple) and len(points) == 1:
        points = points[0]
    
    # 如果 points 是元组
    if isinstance(points, tuple):
        # 广播所有坐标数组，返回广播后的数组 p
        p = np.broadcast_arrays(*points)
        # 获取广播后数组的长度
        n = len(p)
        # 遍历数组，检查是否每个元素的形状都相同
        for j in range(1, n):
            if p[j].shape != p[0].shape:
                raise ValueError("coordinate arrays do not have the same shape")
        # 创建一个空数组，形状为 p[0].shape + (len(points),)，数据类型为 float
        points = np.empty(p[0].shape + (len(points),), dtype=float)
        # 将广播后的数组中的数据填充到 points 数组中对应的位置
        for j, item in enumerate(p):
            points[..., j] = item
    else:
        # 将 points 转换为任意数组
        points = np.asanyarray(points)
        # 如果 points 是一维数组
        if points.ndim == 1:
            # 如果未指定 ndim，则将 points 重塑为二维数组（-1 行，1 列），否则重塑为指定维度 ndim
            if ndim is None:
                points = points.reshape(-1, 1)
            else:
                points = points.reshape(-1, ndim)
    
    # 返回处理后的 points 数组
    return points


def _check_init_shape(points, values, ndim=None):
    """
    Check shape of points and values arrays
    检查 points 和 values 数组的形状。

    """
    # 如果 values 数组的第一个维度长度与 points 数组的第一个维度长度不同，引发 ValueError
    if values.shape[0] != points.shape[0]:
        raise ValueError("different number of values and points")
    # 如果 points 数组的维度不为 2，引发 ValueError
    if points.ndim != 2:
        raise ValueError("invalid shape for input data points")
    # 如果 points 数组的第二个维度小于 2，引发 ValueError
    if points.shape[1] < 2:
        raise ValueError("input data must be at least 2-D")
    # 如果指定了 ndim 且 points 数组的第二个维度与 ndim 不同，引发 ValueError
    if ndim is not None and points.shape[1] != ndim:
        raise ValueError("this mode of interpolation available only for "
                         "%d-D data" % ndim)


#------------------------------------------------------------------------------
# Linear interpolation in N-D
#------------------------------------------------------------------------------

class LinearNDInterpolator(NDInterpolatorBase):
    """
    LinearNDInterpolator(points, values, fill_value=np.nan, rescale=False)

    Piecewise linear interpolator in N > 1 dimensions.

    .. versionadded:: 0.9

    Methods
    -------
    __call__

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndims); or Delaunay
        2-D array of data point coordinates, or a precomputed Delaunay triangulation.
        数据点坐标的二维数组，或预先计算的 Delaunay 三角剖分。
    values : ndarray of float or complex, shape (npoints, ...), optional
        N-D array of data values at `points`.  The length of `values` along the
        first axis must be equal to the length of `points`. Unlike some
        interpolators, the interpolation axis cannot be changed.
        在 points 处的数据值的 N 维数组。values 沿第一个轴的长度必须等于 points 的长度。
        与某些插值器不同，无法更改插值轴。
    fill_value : float, optional
        Value used to fill in for requested points outside of the
        convex hull of the input points.  If not provided, then
        the default is ``nan``.
        用于在请求点位于输入点凸包之外时填充的值。如果未提供，则默认为“nan”。
    rescale : bool, optional
        Rescale points to unit cube before performing interpolation.
        This is useful if some of the input dimensions have
        incommensurable units and differ by many orders of magnitude.
        在执行插值之前，将 points 重新缩放到单位立方体中。如果某些输入维度具有不可比较的单位并且数量级差异很大，则这很有用。

    Notes
    -----
    The interpolant is constructed by triangulating the input data
    with Qhull [1]_, and on each triangle performing linear
    barycentric interpolation.
    插值器通过使用 Qhull 对输入数据进行三角剖分构建，然后在每个三角形上执行线性重心插值。

    .. [1] Qhull: http://www.qhull.org/
    """

    # LinearNDInterpolator 类实现了 NDInterpolatorBase 的线性插值功能，支持 N > 1 维度的数据插值。
    pass
    """
    .. note:: For data on a regular grid use `interpn` instead.

    Examples
    --------
    We can interpolate values on a 2D plane:

    >>> from scipy.interpolate import LinearNDInterpolator
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.random(10) - 0.5
    >>> y = rng.random(10) - 0.5
    >>> z = np.hypot(x, y)
    >>> X = np.linspace(min(x), max(x))
    >>> Y = np.linspace(min(y), max(y))
    >>> X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    >>> interp = LinearNDInterpolator(list(zip(x, y)), z)
    >>> Z = interp(X, Y)
    >>> plt.pcolormesh(X, Y, Z, shading='auto')
    >>> plt.plot(x, y, "ok", label="input point")
    >>> plt.legend()
    >>> plt.colorbar()
    >>> plt.axis("equal")
    >>> plt.show()

    See also
    --------
    griddata :
        Interpolate unstructured D-D data.
    NearestNDInterpolator :
        Nearest-neighbor interpolator in N dimensions.
    CloughTocher2DInterpolator :
        Piecewise cubic, C1 smooth, curvature-minimizing interpolator in 2D.
    interpn : Interpolation on a regular grid or rectilinear grid.
    RegularGridInterpolator : Interpolator on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).

    References
    ----------
    .. [1] http://www.qhull.org/

    """

    # 初始化函数，用于创建 LinearNDInterpolator 的实例
    def __init__(self, points, values, fill_value=np.nan, rescale=False):
        # 调用父类 NDInterpolatorBase 的初始化方法，传入数据点和对应的值
        NDInterpolatorBase.__init__(self, points, values, fill_value=fill_value,
                rescale=rescale)

    # 计算三角剖分的内部方法，接收点集作为参数
    def _calculate_triangulation(self, points):
        # 使用 Qhull 库进行 Delaunay 三角剖分
        self.tri = qhull.Delaunay(points)

    # 对复数值进行评估的内部方法，接收评估点 xi 作为参数
    def _evaluate_complex(self, xi):
        # 调用 _do_evaluate 方法，传入评估点 xi 和复数单位 1.0j
        return self._do_evaluate(xi, 1.0j)

    # 对双精度值进行评估的内部方法，接收评估点 xi 作为参数
    def _evaluate_double(self, xi):
        # 调用 _do_evaluate 方法，传入评估点 xi 和浮点数 1.0
        return self._do_evaluate(xi, 1.0)

    # 使用 Cython 的装饰器指令，关闭边界检查
    @cython.boundscheck(False)
    # 使用 Cython 的装饰器指令，关闭数组索引的包装
    @cython.wraparound(False)
    # 定义一个名为 `_do_evaluate` 的方法，接受两个参数：
    # - `xi`，一个二维的常量数组（指针），其元素类型为 `double`
    # - `dummy`，一个常量 `double` 或者 `complex` 类型的参数，未使用
    def _do_evaluate(self, const double[:,::1] xi, const double_or_complex dummy):
        # 从 `self.values` 中获取一个常量的二维数组（指针），其元素类型为 `double` 或者 `complex`
        cdef const double_or_complex[:,::1] values = self.values
        # 声明一个二维数组（指针），用于存储计算结果，元素类型为 `double` 或者 `complex`
        cdef double_or_complex[:,::1] out
        # 从 `self.tri.simplices` 中获取一个常量的二维整数数组（指针）
        cdef const int[:,::1] simplices = self.tri.simplices
        # 声明一个变量 `fill_value`，其类型为 `double` 或者 `complex`
        cdef double_or_complex fill_value
        # 声明一个一维数组 `c`，其元素类型为 `double`
        cdef double c[NPY_MAXDIMS]
        # 声明一些整数变量：`i`, `j`, `k`, `m`, `ndim`, `isimplex`, `start`, `nvalues`
        cdef int i, j, k, m, ndim, isimplex, start, nvalues
        # 声明一个 `qhull.DelaunayInfo_t` 类型的变量 `info`
        cdef qhull.DelaunayInfo_t info
        # 声明两个 `double` 类型的变量 `eps`, `eps_broad`
        cdef double eps, eps_broad
        
        # 获取 `xi` 的第二维度大小，赋给 `ndim`
        ndim = xi.shape[1]
        # 将 `self.fill_value` 的值赋给 `fill_value`
        fill_value = self.fill_value

        # 调用 `qhull._get_delaunay_info` 函数，获取 `info` 对象的信息
        qhull._get_delaunay_info(&info, self.tri, 1, 0, 0)

        # 创建一个空的数组 `out`，形状为 `(xi.shape[0], self.values.shape[1])`，数据类型与 `self.values` 相同
        out = np.empty((xi.shape[0], self.values.shape[1]),
                       dtype=self.values.dtype)
        # 获取 `out` 的第二维度大小，赋给 `nvalues`
        nvalues = out.shape[1]

        # 初始化 `start` 为 `0`
        start = 0
        # 设置 `eps` 为 `100 * DBL_EPSILON`
        eps = 100 * DBL_EPSILON
        # 计算 `eps_broad` 为 `sqrt(DBL_EPSILON)`
        eps_broad = sqrt(DBL_EPSILON)

        # 使用 `nogil` 上下文，对 `xi` 的每一行进行迭代
        with nogil:
            for i in range(xi.shape[0]):

                # 1) 查找所在的单纯形（simplex）

                # 调用 `qhull._find_simplex` 函数，找到 `xi` 的第 `i` 行所在的单纯形，并将结果赋给 `isimplex`
                isimplex = qhull._find_simplex(&info, c,
                                               &xi[0,0] + i*ndim,
                                               &start, eps, eps_broad)

                # 2) 线性重心插值

                # 如果找不到单纯形（isimplex == -1）
                if isimplex == -1:
                    # 不进行外推，将 `fill_value` 填充到 `out` 的第 `i` 行
                    for k in range(nvalues):
                        out[i,k] = fill_value
                    # 继续下一次迭代
                    continue

                # 将 `out` 的第 `i` 行所有元素初始化为 `0`
                for k in range(nvalues):
                    out[i,k] = 0

                # 对于单纯形中的每个顶点（`ndim + 1` 个）
                for j in range(ndim+1):
                    # 对 `out` 的第 `i` 行的每个元素进行加权累加
                    for k in range(nvalues):
                        m = simplices[isimplex,j]
                        out[i,k] = out[i,k] + c[j] * values[m,k]

        # 返回计算结果数组 `out`
        return out
#------------------------------------------------------------------------------
# Gradient estimation in 2D
#------------------------------------------------------------------------------

class GradientEstimationWarning(Warning):
    pass

@cython.cdivision(True)
cdef int _estimate_gradients_2d_global(const qhull.DelaunayInfo_t *d,
                                       const double *data,
                                       int maxiter,
                                       double tol,
                                       double *y) noexcept nogil:
    """
    Estimate gradients of a function at the vertices of a 2d triangulation.

    Parameters
    ----------
    d : const qhull.DelaunayInfo_t *
        Triangulation information in 2D
    data : const double *
        Function values at the vertices
    maxiter : int
        Maximum number of Gauss-Seidel iterations
    tol : double
        Absolute / relative stop tolerance
    y : double *
        Output array for derivatives [F_x, F_y] at the vertices

    Returns
    -------
    int
        Number of iterations if converged, 0 if maxiter reached
        without convergence

    Notes
    -----
    This routine uses a re-implementation of the global approximate
    curvature minimization algorithm described in [Nielson83] and [Renka84].

    References
    ----------
    .. [Nielson83] G. Nielson,
       ''A method for interpolating scattered data based upon a minimum norm
       network''.
       Math. Comp., 40, 253 (1983).
    .. [Renka84] R. J. Renka and A. K. Cline.
       ''A Triangle-based C1 interpolation method.'',
       Rocky Mountain J. Math., 14, 223 (1984).

    """
    cdef double Q[2*2]  # Matrix for computations
    cdef double s[2]    # Auxiliary vector
    cdef double r[2]    # Another auxiliary vector
    cdef int ipoint, iiter, k, ipoint2, jpoint2  # Loop indices and counters
    cdef double f1, f2, df2, ex, ey, L, L3, det, err, change  # Variables for calculations

    # initialize
    for ipoint in range(2*d.npoints):
        y[ipoint] = 0

    #
    # Main point:
    #
    #    Z = sum_T sum_{E in T} int_E |W''|^2 = min!
    #
    # where W'' is the second derivative of the Clough-Tocher
    # interpolant to the direction of the edge E in triangle T.
    #
    # The minimization is done iteratively: for each vertex V,
    # the sum
    #
    #    Z_V = sum_{E connected to V} int_E |W''|^2
    #
    # is minimized separately, using existing values at other V.
    #
    # Since the interpolant can be written as
    #
    #     W(x) = f(x) + w(x)^T y
    #
    # where y = [ F_x(V); F_y(V) ], it is clear that the solution to
    # the local problem is given as a solution of the 2x2 matrix
    # equation.
    #
    # Here, we use the Clough-Tocher interpolant, which restricted to
    # a single edge is
    #
    #     w(x) = (1 - x)**3   * f1
    #          + x*(1 - x)**2 * (df1 + 3*f1)
    #          + x**2*(1 - x) * (df2 + 3*f2)
    #          + x**3         * f2
    #
    # where f1, f2 are values at the vertices, and df1 and df2 are
    # derivatives along the edge (away from the vertices).
    #
    # Gauss-Seidel 迭代方法求解线性方程组
    for iiter in range(maxiter):
        # 初始化误差
        err = 0
    
        # 遍历每个点
        for ipoint in range(d.npoints):
            # 初始化 Q 矩阵和 s 向量
            for k in range(2*2):
                Q[k] = 0
            for k in range(2):
                s[k] = 0
    
            # 遍历当前点的邻居点
            for jpoint2 in range(d.vertex_neighbors_indptr[ipoint],
                                  d.vertex_neighbors_indptr[ipoint+1]):
                ipoint2 = d.vertex_neighbors_indices[jpoint2]
    
                # 计算边的向量和长度
                ex = d.points[2*ipoint2 + 0] - d.points[2*ipoint + 0]
                ey = d.points[2*ipoint2 + 1] - d.points[2*ipoint + 1]
                L = sqrt(ex**2 + ey**2)
                L3 = L * L * L
    
                # 获取两个点的数据
                f1 = data[ipoint]
                f2 = data[ipoint2]
    
                # 计算在边上的梯度投影
                df2 = -ex * y[2*ipoint2 + 0] - ey * y[2*ipoint2 + 1]
    
                # 更新 Q 矩阵和 s 向量的元素
                Q[0] += 4 * ex * ex / L3
                Q[1] += 4 * ex * ey / L3
                Q[3] += 4 * ey * ey / L3
    
                s[0] += (6 * (f1 - f2) - 2 * df2) * ex / L3
                s[1] += (6 * (f1 - f2) - 2 * df2) * ey / L3
    
            # 设置 Q 矩阵的第二行第一列元素
            Q[2] = Q[1]
    
            # 求解线性方程组
            det = Q[0] * Q[3] - Q[1] * Q[2]
            r[0] = (Q[3] * s[0] - Q[1] * s[1]) / det
            r[1] = (-Q[2] * s[0] + Q[0] * s[1]) / det
    
            # 计算解的变化
            change = max(fabs(y[2*ipoint + 0] + r[0]),
                         fabs(y[2*ipoint + 1] + r[1]))
    
            # 更新 y 向量的值
            y[2*ipoint + 0] = -r[0]
            y[2*ipoint + 1] = -r[1]
    
            # 计算相对/绝对误差
            change /= max(1.0, max(fabs(r[0]), fabs(r[1])))
            err = max(err, change)
    
        # 如果误差小于设定的容差值，则迭代结束，返回迭代次数
        if err < tol:
            return iiter + 1
    
    # 如果达到最大迭代次数仍未收敛，则返回 0 表示未收敛
    # Gauss-Seidel 迭代未收敛
    return 0
@cython.boundscheck(False)
@cython.wraparound(False)
# 使用 Cython 的装饰器设置数组边界检查和循环包装检查的关闭

cpdef estimate_gradients_2d_global(tri, y, int maxiter=400, double tol=1e-6):
    # 定义 Cython 编译的 C 函数的原型，用于全局二维梯度估计
    cdef const double[:,::1] data  # 常量双精度浮点数二维数组
    cdef double[:,:,::1] grad  # 可变双精度浮点数三维数组
    cdef qhull.DelaunayInfo_t info  # qhull 库中的 Delaunay 三角剖分信息结构体
    cdef int k, ret, nvalues  # 定义整型变量

    y = np.asanyarray(y)  # 将输入的 y 转换为 NumPy 数组

    if y.shape[0] != tri.npoints:
        raise ValueError("'y' has a wrong number of items")  # 如果 y 的长度不等于 tri 的点数，则引发 ValueError

    if np.issubdtype(y.dtype, np.complexfloating):
        # 如果 y 的数据类型是复数类型，分别估计实部和虚部的梯度，并返回复数类型的结果
        rg = estimate_gradients_2d_global(tri, y.real, maxiter=maxiter, tol=tol)
        ig = estimate_gradients_2d_global(tri, y.imag, maxiter=maxiter, tol=tol)
        r = np.zeros(rg.shape, dtype=complex)
        r.real = rg
        r.imag = ig
        return r

    y_shape = y.shape  # 记录原始 y 的形状

    if y.ndim == 1:
        y = y[:,None]  # 如果 y 是一维数组，转换为列向量

    y = y.reshape(tri.npoints, -1).T  # 将 y 转换为与三角剖分点数相符的形状，并转置
    y = np.ascontiguousarray(y, dtype=np.float64)  # 将 y 转换为连续的双精度浮点数数组
    yi = np.empty((y.shape[0], y.shape[1], 2))  # 创建一个空的三维数组，用于存储梯度信息

    data = y  # 将 y 赋值给 data
    grad = yi  # 将 yi 赋值给 grad

    qhull._get_delaunay_info(&info, tri, 0, 0, 1)  # 调用 qhull 库的函数获取 Delaunay 信息
    nvalues = data.shape[0]  # 获取 data 的第一维长度

    for k in range(nvalues):
        with nogil:  # 进入无全局解锁区域
            ret = _estimate_gradients_2d_global(
                &info,
                &data[k,0],  # 第 k 行的起始地址
                maxiter,
                tol,
                &grad[k,0,0]  # 第 k 行的梯度起始地址
            )

        if ret == 0:
            warnings.warn("Gradient estimation did not converge, "
                          "the results may be inaccurate",
                          GradientEstimationWarning)  # 如果梯度估计未收敛，发出警告

    return yi.transpose(1, 0, 2).reshape(y_shape + (2,))
    # 将 yi 的维度进行转置和重塑，返回原始 y 的形状增加一个维度为 2 的数组

#------------------------------------------------------------------------------
# 在二维空间进行立方插值
#------------------------------------------------------------------------------


@cython.cdivision(True)
# 使用 Cython 装饰器启用 C 语言的除法
cdef double_or_complex _clough_tocher_2d_single(const qhull.DelaunayInfo_t *d,
                                                int isimplex,
                                                double *b,
                                                double_or_complex *f,
                                                double_or_complex *df) noexcept nogil:
    """
    在二维三角形上评估 Clough-Tocher 插值器。

    Parameters
    ----------
    d :
        Delaunay 信息
    isimplex : int
        要评估的三角形
    b : shape (3,)
        三角形上点的重心坐标
    f : shape (3,)
        顶点处的函数值
    df : shape (3, 2)
        顶点处的梯度值

    Returns
    -------
    w :
        给定点的插值器值

    References
    ----------
    .. [CT] 参见，
       P. Alfeld,
       ''A trivariate Clough-Tocher scheme for tetrahedral data''.
       Computer Aided Geometric Design, 1, 169 (1984);
       G. Farin,
       ''Triangular Bernstein-Bezier patches''.
       Computer Aided Geometric Design, 3, 83 (1986).

    """
    cdef double_or_complex \
         c3000, c0300, c0030, c0003, \
         c2100, c2010, c2001, c0210, c0201, c0021, \
         c1200, c1020, c1002, c0120, c0102, c0012, \
         c1101, c1011, c0111
    cdef double_or_complex \
         f1, f2, f3, df12, df13, df21, df23, df31, df32
    cdef double g[3]
    cdef double e12x, e12y, e23x, e23y, e31x, e31y
    cdef double_or_complex w
    cdef double minval
    cdef double b1, b2, b3, b4
    cdef int k, itri
    cdef double c[3]
    cdef double y[2]

    # XXX: optimize + refactor this!

    # 计算三角形边的长度差值
    e12x = (+ d.points[0 + 2*d.simplices[3*isimplex + 1]]
            - d.points[0 + 2*d.simplices[3*isimplex + 0]])
    e12y = (+ d.points[1 + 2*d.simplices[3*isimplex + 1]]
            - d.points[1 + 2*d.simplices[3*isimplex + 0]])

    e23x = (+ d.points[0 + 2*d.simplices[3*isimplex + 2]]
            - d.points[0 + 2*d.simplices[3*isimplex + 1]])
    e23y = (+ d.points[1 + 2*d.simplices[3*isimplex + 2]]
            - d.points[1 + 2*d.simplices[3*isimplex + 1]])

    e31x = (+ d.points[0 + 2*d.simplices[3*isimplex + 0]]
            - d.points[0 + 2*d.simplices[3*isimplex + 2]])
    e31y = (+ d.points[1 + 2*d.simplices[3*isimplex + 0]]
            - d.points[1 + 2*d.simplices[3*isimplex + 2]])

    # 设置 f1, f2, f3 为给定 f 的前三个元素
    f1 = f[0]
    f2 = f[1]
    f3 = f[2]

    # 计算 df12, df21, df23, df32, df31, df13
    df12 = +(df[2*0+0]*e12x + df[2*0+1]*e12y)
    df21 = -(df[2*1+0]*e12x + df[2*1+1]*e12y)
    df23 = +(df[2*1+0]*e23x + df[2*1+1]*e23y)
    df32 = -(df[2*2+0]*e23x + df[2*2+1]*e23y)
    df31 = +(df[2*2+0]*e31x + df[2*2+1]*e31y)
    df13 = -(df[2*0+0]*e31x + df[2*0+1]*e31y)

    # 设置 c3000, c2100, ..., c0021, c2001 为给定的线性组合值
    c3000 = f1
    c2100 = (df12 + 3*c3000)/3
    c2010 = (df13 + 3*c3000)/3
    c0300 = f2
    c1200 = (df21 + 3*c0300)/3
    c0210 = (df23 + 3*c0300)/3
    c0030 = f3
    c1020 = (df31 + 3*c0030)/3
    c0120 = (df32 + 3*c0030)/3

    c2001 = (c2100 + c2010 + c3000)/3
    c0201 = (c1200 + c0300 + c0210)/3
    c0021 = (c1020 + c0120 + c0030)/3

    #
    # 现在，我们需要强加条件，使得样条的梯度在某个方向 `w` 上是沿着边的线性函数。
    #
    # 只要两个相邻三角形在方向 `w` 的选择上达成一致，这保证了全局的 C1 可微性。
    # 否则，方向的选择是任意的（当然，它不应指向沿着边的方向）。
    #
    # 在 [CT]_ 中建议选择 `w` 为边的法线。这个选择由以下公式给出
    #
    #    w_12 = E_24 + g[0] * E_23
    #    w_23 = E_34 + g[1] * E_31
    #    w_31 = E_14 + g[2] * E_12
    #
    #    g[0] = -(e24x*e23x + e24y*e23y) / (e23x**2 + e23y**2)
    #    g[1] = -(e34x*e31x + e34y*e31y) / (e31x**2 + e31y**2)
    #    g[2] = -(e14x*e12x + e14y*e12y) / (e12x**2 + e12y**2)
    #
    # 然而，这个选择得到的插值函数在仿射变换下并不保持不变。这带来了一些问题：
    # 对于非常窄的三角形，样条可能会产生
    # 对于输入数据进行局部插值时，需要选择合适的 affine 不变量 g[k]。
    # 我们选择 w = V_4' - V_4，其中 V_4 是当前三角形的重心，V_4' 是相邻三角形的重心。
    # 这个选择在仿射变换下类似于梯度，因此插值结果是 affine-invariant 的。
    # 此外，两个相邻三角形对于选择 w 是一致的（符号不重要），因此这个选择也使得插值 C1 连续。
    # 这种方法的缺点是性能惩罚，因为需要查看相邻的三角形。

    for k in range(3):
        # 获取当前简单形状的第 k 个邻居的索引
        itri = d.neighbors[3*isimplex + k]

        if itri == -1:
            # 如果没有邻居，计算朝重心方向 (e_12 + e_13)/2 的导数
            g[k] = -1./2
            continue

        # 计算邻居的重心，在局部重心坐标系中
        y[0] = (+ d.points[0 + 2*d.simplices[3*itri + 0]]
                + d.points[0 + 2*d.simplices[3*itri + 1]]
                + d.points[0 + 2*d.simplices[3*itri + 2]]) / 3

        y[1] = (+ d.points[1 + 2*d.simplices[3*itri + 0]]
                + d.points[1 + 2*d.simplices[3*itri + 1]]
                + d.points[1 + 2*d.simplices[3*itri + 2]]) / 3

        # 计算邻居的重心在当前局部坐标系的重心坐标
        qhull._barycentric_coordinates(2, d.transform + isimplex*2*3, y, c)

        # 根据局部坐标系中的 barycentric 坐标重写 V_4'-V_4 = const*[(V_4-V_2) + g_i*(V_3 - V_2)]
        # 结果可以写成 barycentric 坐标的形式。Barycentric 坐标在仿射变换下不变，因此可以直接得出以下选择是 affine-invariant 的。

        if k == 0:
            g[k] = (2*c[2] + c[1] - 1) / (2 - 3*c[2] - 3*c[1])
        elif k == 1:
            g[k] = (2*c[0] + c[2] - 1) / (2 - 3*c[0] - 3*c[2])
        elif k == 2:
            g[k] = (2*c[1] + c[0] - 1) / (2 - 3*c[1] - 3*c[0])

    # 计算插值函数的边界条件
    c0111 = (g[0]*(-c0300 + 3*c0210 - 3*c0120 + c0030)
             + (-c0300 + 2*c0210 - c0120 + c0021 + c0201))/2
    c1011 = (g[1]*(-c0030 + 3*c1020 - 3*c2010 + c3000)
             + (-c0030 + 2*c1020 - c2010 + c2001 + c0021))/2
    c1101 = (g[2]*(-c3000 + 3*c2100 - 3*c1200 + c0300)
             + (-c3000 + 2*c2100 - c1200 + c2001 + c0201))/2

    # 计算插值函数的边界条件的平均值
    c1002 = (c1101 + c1011 + c2001)/3
    c0102 = (c1101 + c0111 + c0201)/3
    c0012 = (c1011 + c0111 + c0021)/3

    # 计算扩展 barycentric 坐标
    c0003 = (c1002 + c0102 + c0012)/3

    # 找到 barycentric 坐标中的最小值
    minval = b[0]
    for k in range(3):
        if b[k] < minval:
            minval = b[k]

    # 重新计算 barycentric 坐标
    b1 = b[0] - minval
    b2 = b[1] - minval
    b3 = b[2] - minval
    # 计算 b4 的值，这里 b4 是最小值 minval 的三倍
    b4 = 3*minval

    # 计算多项式的值 —— 使用繁琐且不优雅的方式进行计算，
    # 其中四个坐标中有一个实际上为零
    w = (b1**3*c3000 + 3*b1**2*b2*c2100 + 3*b1**2*b3*c2010 +
         3*b1**2*b4*c2001 + 3*b1*b2**2*c1200 +
         6*b1*b2*b4*c1101 + 3*b1*b3**2*c1020 + 6*b1*b3*b4*c1011 +
         3*b1*b4**2*c1002 + b2**3*c0300 + 3*b2**2*b3*c0210 +
         3*b2**2*b4*c0201 + 3*b2*b3**2*c0120 + 6*b2*b3*b4*c0111 +
         3*b2*b4**2*c0102 + b3**3*c0030 + 3*b3**2*b4*c0021 +
         3*b3*b4**2*c0012 + b4**3*c0003)

    # 返回计算得到的多项式值 w
    return w
# 定义 CloughTocher2DInterpolator 类，继承自 NDInterpolatorBase 类
class CloughTocher2DInterpolator(NDInterpolatorBase):
    """CloughTocher2DInterpolator(points, values, tol=1e-6).

    Piecewise cubic, C1 smooth, curvature-minimizing interpolator in 2D.

    .. versionadded:: 0.9

    Methods
    -------
    __call__

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndims); or Delaunay
        2-D array of data point coordinates, or a precomputed Delaunay triangulation.
    values : ndarray of float or complex, shape (npoints, ...)
        N-D array of data values at `points`. The length of `values` along the
        first axis must be equal to the length of `points`. Unlike some
        interpolators, the interpolation axis cannot be changed.
    fill_value : float, optional
        Value used to fill in for requested points outside of the
        convex hull of the input points.  If not provided, then
        the default is ``nan``.
    tol : float, optional
        Absolute/relative tolerance for gradient estimation.
    maxiter : int, optional
        Maximum number of iterations in gradient estimation.
    rescale : bool, optional
        Rescale points to unit cube before performing interpolation.
        This is useful if some of the input dimensions have
        incommensurable units and differ by many orders of magnitude.

    Notes
    -----
    The interpolant is constructed by triangulating the input data
    with Qhull [1]_, and constructing a piecewise cubic
    interpolating Bezier polynomial on each triangle, using a
    Clough-Tocher scheme [CT]_.  The interpolant is guaranteed to be
    continuously differentiable.

    The gradients of the interpolant are chosen so that the curvature
    of the interpolating surface is approximatively minimized. The
    gradients necessary for this are estimated using the global
    algorithm described in [Nielson83]_ and [Renka84]_.

    .. note:: For data on a regular grid use `interpn` instead.

    Examples
    --------
    We can interpolate values on a 2D plane:

    >>> from scipy.interpolate import CloughTocher2DInterpolator
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.random(10) - 0.5
    >>> y = rng.random(10) - 0.5
    >>> z = np.hypot(x, y)
    >>> X = np.linspace(min(x), max(x))
    >>> Y = np.linspace(min(y), max(y))
    >>> X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    >>> interp = CloughTocher2DInterpolator(list(zip(x, y)), z)
    >>> Z = interp(X, Y)
    >>> plt.pcolormesh(X, Y, Z, shading='auto')
    >>> plt.plot(x, y, "ok", label="input point")
    >>> plt.legend()
    >>> plt.colorbar()
    >>> plt.axis("equal")
    >>> plt.show()

    See also
    --------
    griddata :
        Interpolate unstructured D-D data.
    LinearNDInterpolator :
        Piecewise linear interpolator in N > 1 dimensions.
    NearestNDInterpolator :
        Nearest-neighbor interpolator in N > 1 dimensions.
    """
    
    # __init__ 方法用于初始化对象，接受数据点和对应的值，以及其他可选参数
    def __init__(self, points, values, fill_value=np.nan, tol=1e-6, maxiter=None, rescale=False):
        super().__init__(points, values)
        self.fill_value = fill_value
        self.tol = tol
        self.maxiter = maxiter
        self.rescale = rescale

    # __call__ 方法使对象实例可以像函数一样被调用，实现插值计算
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Clough-Tocher interpolation is not callable. Use interp(X, Y) instead.")
    interpn : Interpolation on a regular grid or rectilinear grid.
    RegularGridInterpolator : Interpolator on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).

    References
    ----------
    .. [1] http://www.qhull.org/

    .. [CT] See, for example,
       P. Alfeld,
       ''A trivariate Clough-Tocher scheme for tetrahedral data''.
       Computer Aided Geometric Design, 1, 169 (1984);
       G. Farin,
       ''Triangular Bernstein-Bezier patches''.
       Computer Aided Geometric Design, 3, 83 (1986).

    .. [Nielson83] G. Nielson,
       ''A method for interpolating scattered data based upon a minimum norm
       network''.
       Math. Comp., 40, 253 (1983).

    .. [Renka84] R. J. Renka and A. K. Cline.
       ''A Triangle-based C1 interpolation method.'',
       Rocky Mountain J. Math., 14, 223 (1984).

    """

    # 初始化函数，用于创建 InterpolatedUnivariateSpline 对象
    def __init__(self, points, values, fill_value=np.nan,
                 tol=1e-6, maxiter=400, rescale=False):
        self._tol = tol
        self._maxiter = maxiter
        # 调用父类 NDInterpolatorBase 的初始化方法，设置基本参数和值
        NDInterpolatorBase.__init__(self, points, values, ndim=2,
                                    fill_value=fill_value, rescale=rescale,
                                    need_values=False)
    
    # 设置插值点的数值
    def _set_values(self, values, fill_value=np.nan, need_contiguous=True, ndim=None):
        """
        Sets the values of the interpolation points.

        Parameters
        ----------
        values : ndarray of float or complex, shape (npoints, ...)
            Data values.
        """
        # 调用父类 NDInterpolatorBase 的 _set_values 方法，设置插值点的数值
        NDInterpolatorBase._set_values(self, values, fill_value=fill_value, need_contiguous=need_contiguous, ndim=ndim)
        # 如果数值不为 None，则估算二维全局梯度
        if self.values is not None:
            self.grad = estimate_gradients_2d_global(self.tri, self.values,
                                                    tol=self._tol, maxiter=self._maxiter)
    
    # 计算三角化数据点
    def _calculate_triangulation(self, points):
        # 使用 qhull 库进行 Delaunay 三角化
        self.tri = qhull.Delaunay(points)

    # 对复数进行评估
    def _evaluate_double(self, xi):
        return self._do_evaluate(xi, 1.0)

    # 对复杂评估进行评估
    def _evaluate_complex(self, xi):
        return self._do_evaluate(xi, 1.0j)

    # 在 Cython 中禁用边界检查和包装
    @cython.boundscheck(False)
    @cython.wraparound(False)
    # 定义一个私有方法 `_do_evaluate`，接受两个参数：xi 是一个常量双精度二维数组，dummy 是一个常量双精度或复数。
    def _do_evaluate(self, const double[:,::1] xi, const double_or_complex dummy):
        # 将 self.values 赋给 values，这是一个常量双精度或复数二维数组
        cdef const double_or_complex[:,::1] values = self.values
        # 将 self.grad 赋给 grad，这是一个常量双精度或复数三维数组
        cdef const double_or_complex[:,:,:] grad = self.grad
        # 定义一个双精度或复数二维数组 out
        cdef double_or_complex[:,::1] out
        # 将 self.tri.simplices 赋给 simplices，这是一个整数二维数组
        cdef const int[:,::1] simplices = self.tri.simplices
        # 定义一个双精度一维数组 c，长度为 NPY_MAXDIMS
        cdef double c[NPY_MAXDIMS]
        # 定义一个双精度或复数一维数组 f，长度为 NPY_MAXDIMS+1
        cdef double_or_complex f[NPY_MAXDIMS+1]
        # 定义一个双精度或复数一维数组 df，长度为 2*NPY_MAXDIMS+2
        cdef double_or_complex df[2*NPY_MAXDIMS+2]
        # 定义一个双精度或复数 w
        cdef double_or_complex w
        # 定义一个双精度或复数 fill_value
        cdef double_or_complex fill_value
        # 定义整数 i, j, k, ndim, isimplex, start, nvalues
        cdef int i, j, k, ndim, isimplex, start, nvalues
        # 定义 qhull.DelaunayInfo_t 结构体变量 info
        cdef qhull.DelaunayInfo_t info
        # 定义双精度变量 eps, eps_broad

        # 获取 xi 的第二维度大小，即 xi 的列数，赋给 ndim
        ndim = xi.shape[1]
        # 将 self.fill_value 赋给 fill_value
        fill_value = self.fill_value

        # 调用 qhull._get_delaunay_info 函数，填充 info 结构体
        qhull._get_delaunay_info(&info, self.tri, 1, 1, 0)

        # 创建一个全零数组 out，形状为 (xi 的行数, self.values 的第二维度大小)，数据类型与 self.values 相同
        out = np.zeros((xi.shape[0], self.values.shape[1]),
                       dtype=self.values.dtype)
        # 获取 out 的第二维度大小，赋给 nvalues
        nvalues = out.shape[1]

        # 初始化 start 为 0
        start = 0
        # 设置 eps 为 100 倍的双精度机器精度常数
        eps = 100 * DBL_EPSILON
        # 计算 eps_broad 为 eps 的平方根
        eps_broad = sqrt(eps)

        # 使用 nogil 上下文，不需要全局解释器锁 GIL
        with nogil:
            # 遍历 xi 的行数
            for i in range(xi.shape[0]):
                # 1) 寻找简单形式（simplex）

                # 调用 qhull._find_simplex 函数，查找包含 xi[i,:] 的简单形式，返回 isimplex
                isimplex = qhull._find_simplex(&info, c,
                                               &xi[i,0],
                                               &start, eps, eps_broad)
                
                # 2) Clough-Tocher 插值

                # 如果未找到简单形式（isimplex == -1）
                if isimplex == -1:
                    # 对于每个 k，将 fill_value 赋给 out[i,k]
                    for k in range(nvalues):
                        out[i,k] = fill_value
                    # 继续下一个 i 的迭代
                    continue

                # 找到简单形式后，进行 Clough-Tocher 二维单点插值
                for k in range(nvalues):
                    # 遍历简单形式中的每个顶点 j
                    for j in range(ndim+1):
                        # 将 values[simplices[isimplex,j],k] 赋给 f[j]
                        f[j] = values[simplices[isimplex,j],k]
                        # 将 grad[simplices[isimplex,j],k,0] 赋给 df[2*j]
                        df[2*j] = grad[simplices[isimplex,j],k,0]
                        # 将 grad[simplices[isimplex,j],k,1] 赋给 df[2*j+1]
                        df[2*j+1] = grad[simplices[isimplex,j],k,1]

                    # 调用 _clough_tocher_2d_single 函数进行二维 Clough-Tocher 单点插值，将结果赋给 w
                    w = _clough_tocher_2d_single(&info, isimplex, c, f, df)
                    # 将插值结果 w 赋给 out[i,k]
                    out[i,k] = w

        # 返回数组 out
        return out
```