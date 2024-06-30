# `D:\src\scipysrc\scipy\scipy\interpolate\_ndbspline.py`

```
import itertools
import functools
import operator
import numpy as np

from math import prod  # 导入 math 模块中的 prod 函数

from . import _bspl  # type: ignore  # 导入当前包中的 _bspl 模块，忽略类型检查

import scipy.sparse.linalg as ssl  # 导入 scipy.sparse.linalg 中的 ssl 模块
from scipy.sparse import csr_array  # 从 scipy.sparse 中导入 csr_array 类

from ._bsplines import _not_a_knot  # 从当前包中的 _bsplines 模块导入 _not_a_knot 函数

__all__ = ["NdBSpline"]  # 定义在当前模块中可以被外部访问的公共接口

def _get_dtype(dtype):
    """Return np.complex128 for complex dtypes, np.float64 otherwise."""
    # 如果传入的 dtype 是复数类型，则返回 np.complex128，否则返回 np.float64
    if np.issubdtype(dtype, np.complexfloating):
        return np.complex128
    else:
        return np.float64

class NdBSpline:
    """Tensor product spline object.

    The value at point ``xp = (x1, x2, ..., xN)`` is evaluated as a linear
    combination of products of one-dimensional b-splines in each of the ``N``
    dimensions::

       c[i1, i2, ..., iN] * B(x1; i1, t1) * B(x2; i2, t2) * ... * B(xN; iN, tN)

    Here ``B(x; i, t)`` is the ``i``-th b-spline defined by the knot vector
    ``t`` evaluated at ``x``.

    Parameters
    ----------
    t : tuple of 1D ndarrays
        knot vectors in directions 1, 2, ... N,
        ``len(t[i]) == n[i] + k + 1``
    c : ndarray, shape (n1, n2, ..., nN, ...)
        b-spline coefficients
    k : int or length-d tuple of integers
        spline degrees.
        A single integer is interpreted as having this degree for
        all dimensions.
    extrapolate : bool, optional
        Whether to extrapolate out-of-bounds inputs, or return `nan`.
        Default is to extrapolate.

    Attributes
    ----------
    t : tuple of ndarrays
        Knots vectors.
    c : ndarray
        Coefficients of the tensor-product spline.
    k : tuple of integers
        Degrees for each dimension.
    extrapolate : bool, optional
        Whether to extrapolate or return nans for out-of-bounds inputs.
        Defaults to true.

    Methods
    -------
    __call__
    design_matrix

    See Also
    --------
    BSpline : a one-dimensional B-spline object
    NdPPoly : an N-dimensional piecewise tensor product polynomial

    """
    def __init__(self, t, c, k, *, extrapolate=None):
        # 计算维度数量
        ndim = len(t)

        # 如果 k 不是可迭代对象，将其转换为元组
        try:
            len(k)
        except TypeError:
            k = (k,)*ndim

        # 检查 k 和 t 的维度是否一致
        if len(k) != ndim:
            raise ValueError(f"{len(t) = } != {len(k) = }.")

        # 将 k 转换为元组，并确保其中每个元素是整数索引
        self.k = tuple(operator.index(ki) for ki in k)
        # 将 t 中的每个元素转换为浮点数的 numpy 数组
        self.t = tuple(np.ascontiguousarray(ti, dtype=float) for ti in t)
        # 将 c 转换为 numpy 数组
        self.c = np.asarray(c)

        # 设置 extrapolate 标志，默认为 True
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = bool(extrapolate)

        # 再次确保 c 是 numpy 数组
        self.c = np.asarray(c)

        # 对每个维度进行验证
        for d in range(ndim):
            td = self.t[d]
            kd = self.k[d]
            n = td.shape[0] - kd - 1

            # 检查样条阶数 kd 是否为非负数
            if kd < 0:
                raise ValueError(f"Spline degree in dimension {d} cannot be"
                                 f" negative.")
            # 检查结点向量 td 是否为一维数组
            if td.ndim != 1:
                raise ValueError(f"Knot vector in dimension {d} must be"
                                 f" one-dimensional.")
            # 检查至少需要的结点数是否满足
            if n < kd + 1:
                raise ValueError(f"Need at least {2*kd + 2} knots for degree"
                                 f" {kd} in dimension {d}.")
            # 检查结点向量 td 是否严格递增
            if (np.diff(td) < 0).any():
                raise ValueError(f"Knots in dimension {d} must be in a"
                                 f" non-decreasing order.")
            # 检查结点向量 td 中的唯一内部结点是否至少有两个
            if len(np.unique(td[kd:n + 1])) < 2:
                raise ValueError(f"Need at least two internal knots in"
                                 f" dimension {d}.")
            # 检查结点向量 td 是否不含 NaN 或 Inf
            if not np.isfinite(td).all():
                raise ValueError(f"Knots in dimension {d} should not have"
                                 f" nans or infs.")
            # 检查系数数组 c 的维度是否至少为 ndim
            if self.c.ndim < ndim:
                raise ValueError(f"Coefficients must be at least"
                                 f" {d}-dimensional.")
            # 检查系数数组 c 在当前维度 d 上的形状是否与结点数 n 一致
            if self.c.shape[d] != n:
                raise ValueError(f"Knots, coefficients and degree in dimension"
                                 f" {d} are inconsistent:"
                                 f" got {self.c.shape[d]} coefficients for"
                                 f" {len(td)} knots, need at least {n} for"
                                 f" k={k}.")

        # 获取适当的数据类型并确保系数数组 c 是连续的内存布局
        dt = _get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dt)
    def design_matrix(cls, xvals, t, k, extrapolate=True):
        """Construct the design matrix as a CSR format sparse array.

        Parameters
        ----------
        xvals :  ndarray, shape(npts, ndim)
            Data points. ``xvals[j, :]`` gives the ``j``-th data point as an
            ``ndim``-dimensional array.
        t : tuple of 1D ndarrays, length-ndim
            Knot vectors in directions 1, 2, ... ndim,
        k : int
            B-spline degree.
        extrapolate : bool, optional
            Whether to extrapolate out-of-bounds values or raise a `ValueError`

        Returns
        -------
        design_matrix : a CSR array
            Each row of the design matrix corresponds to a value in `xvals` and
            contains values of B-spline basis elements which are non-zero
            at this value.

        """
        # 将输入的数据点转换为浮点类型的 NumPy 数组
        xvals = np.asarray(xvals, dtype=float)
        # 获取数据点的维度数
        ndim = xvals.shape[-1]
        # 检查结点向量的长度与数据维度数是否一致，若不一致则抛出异常
        if len(t) != ndim:
            raise ValueError(
                f"Data and knots are inconsistent: len(t) = {len(t)} for "
                f"{ndim = }."
            )
        try:
            # 尝试检查 k 是否可以被长度为 ndim 的元组接受，若不能则将 k 转换为长度为 ndim 的元组
            len(k)
        except TypeError:
            k = (k,)*ndim  # 将 k 转换为长度为 ndim 的元组

        # 将 k 转换为 int32 类型的 NumPy 数组
        kk = np.asarray(k, dtype=np.int32)
        # 调用底层函数 _colloc_nd，返回 B-spline 插值的数据、索引和指针
        data, indices, indptr = _bspl._colloc_nd(xvals, t, kk)
        # 返回 CSR 格式的稀疏数组，构成设计矩阵
        return csr_array((data, indices, indptr))
def make_ndbspl(points, values, k=3, *, solver=ssl.gcrotmk, **solver_args):
    """Construct an interpolating NdBspline.

    Parameters
    ----------
    points : tuple of ndarrays of float, with shapes (m1,), ... (mN,)
        The points defining the regular grid in N dimensions. The points in
        each dimension (i.e. every element of the `points` tuple) must be
        strictly ascending or descending.      
    values : ndarray of float, shape (m1, ..., mN, ...)
        The data on the regular grid in n dimensions.
    k : int, optional
        The spline degree. Must be odd. Default is cubic, k=3
    solver : a `scipy.sparse.linalg` solver (iterative or direct), optional.
        An iterative solver from `scipy.sparse.linalg` or a direct one,
        `sparse.sparse.linalg.spsolve`.
        Used to solve the sparse linear system
        ``design_matrix @ coefficients = rhs`` for the coefficients.
        Default is `scipy.sparse.linalg.gcrotmk`
    solver_args : dict, optional
        Additional arguments for the solver. The call signature is
        ``solver(csr_array, rhs_vector, **solver_args)``

    Returns
    -------
    spl : NdBSpline object

    Notes
    -----
    Boundary conditions are not-a-knot in all dimensions.
    """
    ndim = len(points)
    xi_shape = tuple(len(x) for x in points)

    try:
        len(k)
    except TypeError:
        # make k a tuple if it's not already
        k = (k,)*ndim

    # Validate each dimension has enough points for the given spline order
    for d, point in enumerate(points):
        numpts = len(np.atleast_1d(point))
        if numpts <= k[d]:
            raise ValueError(f"There are {numpts} points in dimension {d},"
                             f" but order {k[d]} requires at least "
                             f" {k[d]+1} points per dimension.")

    # Determine the not-a-knot values for each dimension
    t = tuple(_not_a_knot(np.asarray(points[d], dtype=float), k[d])
              for d in range(ndim))

    # Create a grid of points for the colocation matrix
    xvals = np.asarray([xv for xv in itertools.product(*points)], dtype=float)

    # Construct the colocation matrix
    matr = NdBSpline.design_matrix(xvals, t, k)

    # Solve for the coefficients given `values`.
    # 获取输入数组 values 的形状
    v_shape = values.shape
    # 计算将 values 堆叠成二维数组的形状，前 ndim 维度为数据维度，后面是批处理维度
    vals_shape = (prod(v_shape[:ndim]), prod(v_shape[ndim:]))
    # 将 values 重新整形为 vals_shape 形状的数组
    vals = values.reshape(vals_shape)

    # 如果 solver 不是 ssl.spsolve 函数，则使用 functools.partial 创建一个包装函数 _iter_solve
    if solver != ssl.spsolve:
        solver = functools.partial(_iter_solve, solver=solver)
        # 如果 solver_args 中没有 "atol" 参数，则设置默认的 "atol" 为 1e-6，以避免 DeprecationWarning
        if "atol" not in solver_args:
            solver_args["atol"] = 1e-6

    # 使用指定的 solver 求解方程 matr * coef = vals，solver_args 是额外的求解参数
    coef = solver(matr, vals, **solver_args)
    # 将 coef 重新整形为 xi_shape + v_shape[ndim:] 形状的数组
    coef = coef.reshape(xi_shape + v_shape[ndim:])
    # 返回一个 NdBSpline 对象，使用时间点 t 和系数 coef 还有阶数 k 初始化
    return NdBSpline(t, coef, k)
```