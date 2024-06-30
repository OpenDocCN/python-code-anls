# `D:\src\scipysrc\scipy\scipy\interpolate\_bsplines.py`

```
import operator  # 导入 operator 模块，用于函数操作符的支持
from math import prod  # 导入 math 模块中的 prod 函数，用于计算可迭代对象的乘积

import numpy as np  # 导入 NumPy 库，并简写为 np
from scipy._lib._util import normalize_axis_index  # 导入 scipy 库中的 normalize_axis_index 函数
from scipy.linalg import (get_lapack_funcs, LinAlgError,  # 导入 scipy 库中的多个线性代数函数和异常类
                          cholesky_banded, cho_solve_banded,
                          solve, solve_banded)
from scipy.optimize import minimize_scalar  # 导入 scipy 库中的 minimize_scalar 函数
from . import _bspl  # 导入当前包中的 _bspl 模块
from . import _fitpack_impl  # 导入当前包中的 _fitpack_impl 模块
from scipy.sparse import csr_array  # 导入 scipy 库中的 csr_array 类
from scipy.special import poch  # 导入 scipy 库中的 poch 函数
from itertools import combinations  # 导入 itertools 库中的 combinations 函数

__all__ = ["BSpline", "make_interp_spline", "make_lsq_spline",  # 定义当前模块中公开的接口列表
           "make_smoothing_spline"]


def _get_dtype(dtype):
    """Return np.complex128 for complex dtypes, np.float64 otherwise."""
    if np.issubdtype(dtype, np.complexfloating):
        return np.complex128  # 如果输入 dtype 是复数类型，则返回 np.complex128
    else:
        return np.float64  # 否则返回 np.float64


def _as_float_array(x, check_finite=False):
    """Convert the input into a C contiguous float array.

    NB: Upcasts half- and single-precision floats to double precision.
    """
    x = np.ascontiguousarray(x)  # 将输入转换为 C 连续的数组
    dtyp = _get_dtype(x.dtype)  # 获取数组的数据类型
    x = x.astype(dtyp, copy=False)  # 将数组类型转换为 dtyp，并确保不复制数据
    if check_finite and not np.isfinite(x).all():
        raise ValueError("Array must not contain infs or nans.")  # 如果数组包含无穷大或 NaN，则抛出异常
    return x  # 返回转换后的数组


def _dual_poly(j, k, t, y):
    """
    Dual polynomial of the B-spline B_{j,k,t} -
    polynomial which is associated with B_{j,k,t}:
    $p_{j,k}(y) = (y - t_{j+1})(y - t_{j+2})...(y - t_{j+k})$
    """
    if k == 0:
        return 1  # 如果 k 等于 0，则返回 1
    return np.prod([(y - t[j + i]) for i in range(1, k + 1)])  # 计算 B-spline B_{j,k,t} 的对偶多项式


def _diff_dual_poly(j, k, y, d, t):
    """
    d-th derivative of the dual polynomial $p_{j,k}(y)$
    """
    if d == 0:
        return _dual_poly(j, k, t, y)  # 如果 d 等于 0，则返回对偶多项式 $p_{j,k}(y)$
    if d == k:
        return poch(1, k)  # 如果 d 等于 k，则返回 pochhammer 符号 (1)_k
    comb = list(combinations(range(j + 1, j + k + 1), d))  # 生成组合列表
    res = 0
    for i in range(len(comb) * len(comb[0])):
        res += np.prod([(y - t[j + p]) for p in range(1, k + 1)
                        if (j + p) not in comb[i//d]])
    return res  # 返回对偶多项式 $p_{j,k}(y)$ 的 d 阶导数


class BSpline:
    r"""Univariate spline in the B-spline basis.

    .. math::

        S(x) = \sum_{j=0}^{n-1} c_j  B_{j, k; t}(x)

    where :math:`B_{j, k; t}` are B-spline basis functions of degree `k`
    and knots `t`.

    Parameters
    ----------
    t : ndarray, shape (n+k+1,)
        knots
    c : ndarray, shape (>=n, ...)
        spline coefficients
    k : int
        B-spline degree
    extrapolate : bool or 'periodic', optional
        whether to extrapolate beyond the base interval, ``t[k] .. t[n]``,
        or to return nans.
        If True, extrapolates the first and last polynomial pieces of b-spline
        functions active on the base interval.
        If 'periodic', periodic extrapolation is used.
        Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    t : ndarray
        knot vector
    c : ndarray
        spline coefficients
    k : int
        spline degree
    """
    extrapolate : bool
        是否对 B-spline 函数在基础区间上活动的第一个和最后一个多项式片段进行外推。
        如果为 True，则对基础区间上活动的第一个和最后一个多项式片段进行外推。
    axis : int
        插值轴。
        指定插值操作的轴。
    tck : tuple
        一个只读的 ``(self.t, self.c, self.k)`` 的等效表示。

    Methods
    -------
    __call__
        调用对象以计算 B-spline 的值。
    basis_element
        计算 B-spline 的基函数元素。
    derivative
        计算 B-spline 的导数。
    antiderivative
        计算 B-spline 的反导数。
    integrate
        计算 B-spline 的积分。
    insert_knot
        插入节点到 B-spline 中。
    construct_fast
        快速构建 B-spline。
    design_matrix
        计算 B-spline 的设计矩阵。
    from_power_basis
        从幂基础构建 B-spline。

    Notes
    -----
    B-spline 的基函数通过以下方式定义：

    .. math::

        B_{i, 0}(x) = 1, \textrm{如果 $t_i \le x < t_{i+1}$，否则为 $0$}

        B_{i, k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
                 + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)

    **实现细节**

    - 对于度为 `k` 的样条，至少需要 `k+1` 个系数，因此应满足 `n >= k+1`。额外的系数 `c[j]`，其中 `j > n`，将被忽略。

    - 度为 `k` 的 B-spline 基函数在基础区间 `t[k] <= x <= t[n]` 上形成了一个单位划分。

    Examples
    --------

    将 B-spline 的递归定义转换为 Python 代码，可以得到以下示例：

    >>> def B(x, k, i, t):
    ...    if k == 0:
    ...       return 1.0 if t[i] <= x < t[i+1] else 0.0
    ...    if t[i+k] == t[i]:
    ...       c1 = 0.0
    ...    else:
    ...       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    ...    if t[i+k+1] == t[i+1]:
    ...       c2 = 0.0
    ...    else:
    ...       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    ...    return c1 + c2

    >>> def bspline(x, t, c, k):
    ...    n = len(t) - k - 1
    ...    assert (n >= k+1) and (len(c) >= n)
    ...    return sum(c[i] * B(x, k, i, t) for i in range(n))

    注意，这是评估 B-spline 的一种低效（但直观）的方式 --- 该样条类以等效但更高效的方式执行相同操作。

    在基础区间 ``2 <= x <= 4`` 上构建一个二次样条函数，并与评估样条的直接方式进行比较：

    >>> from scipy.interpolate import BSpline
    >>> k = 2
    >>> t = [0, 1, 2, 3, 4, 5, 6]
    >>> c = [-1, 2, 0, -1]
    >>> spl = BSpline(t, c, k)
    >>> spl(2.5)
    array(1.375)
    >>> bspline(2.5, t, c, k)
    1.375

    注意，在基础区间之外，结果可能会有所不同。这是因为 `BSpline` 对 B-spline 函数在基础区间上活动的第一个和最后一个多项式片段进行了外推。

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig, ax = plt.subplots()
    >>> xx = np.linspace(1.5, 4.5, 50)
    >>> ax.plot(xx, [bspline(x, t, c ,k) for x in xx], 'r-', lw=3, label='naive')
    >>> ax.plot(xx, spl(xx), 'b-', lw=4, alpha=0.7, label='BSpline')
    >>> ax.grid(True)
    >>> ax.legend(loc='best')
    >>> plt.show()


    References
    ----------
    @classmethod
    def construct_fast(cls, t, c, k, extrapolate=True, axis=0):
        """Construct a spline without making checks.

        Accepts same parameters as the regular constructor. Input arrays
        `t` and `c` must of correct shape and dtype.
        """
        # 使用类方法构建快速创建样条的实例，跳过输入参数的验证步骤
        self = object.__new__(cls)
        # 设置实例的 t, c, k 属性
        self.t, self.c, self.k = t, c, k
        # 设置是否外推和轴向
        self.extrapolate = extrapolate
        self.axis = axis
        # 返回构建的实例
        return self

    @property
    def tck(self):
        """Equivalent to ``(self.t, self.c, self.k)`` (read-only).
        """
        # 返回元组 (self.t, self.c, self.k)，作为只读属性 tck
        return self.t, self.c, self.k
    def basis_element(cls, t, extrapolate=True):
        """Return a B-spline basis element ``B(x | t[0], ..., t[k+1])``.

        Parameters
        ----------
        t : ndarray, shape (k+2,)
            internal knots
            内部结点数组，长度为 k+2
        extrapolate : bool or 'periodic', optional
            whether to extrapolate beyond the base interval, ``t[0] .. t[k+1]``,
            or to return nans.
            If 'periodic', periodic extrapolation is used.
            Default is True.
            是否对基础区间 ``t[0] .. t[k+1]`` 进行外推，如果选择 'periodic'，则进行周期外推。
            默认为 True。

        Returns
        -------
        basis_element : callable
            A callable representing a B-spline basis element for the knot
            vector `t`.
            返回一个可调用对象，代表给定结点向量 `t` 的 B-样条基函数。

        Notes
        -----
        The degree of the B-spline, `k`, is inferred from the length of `t` as
        ``len(t)-2``. The knot vector is constructed by appending and prepending
        ``k+1`` elements to internal knots `t`.
        B-样条的阶数 `k` 通过 `t` 的长度推断为 ``len(t)-2``。结点向量通过在内部结点 `t` 前后各添加 ``k+1`` 个元素来构建。

        Examples
        --------

        Construct a cubic B-spline:

        >>> import numpy as np
        >>> from scipy.interpolate import BSpline
        >>> b = BSpline.basis_element([0, 1, 2, 3, 4])
        >>> k = b.k
        >>> b.t[k:-k]
        array([ 0.,  1.,  2.,  3.,  4.])
        >>> k
        3

        Construct a quadratic B-spline on ``[0, 1, 1, 2]``, and compare
        to its explicit form:

        >>> t = [0, 1, 1, 2]
        >>> b = BSpline.basis_element(t)
        >>> def f(x):
        ...     return np.where(x < 1, x*x, (2. - x)**2)

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> x = np.linspace(0, 2, 51)
        >>> ax.plot(x, b(x), 'g', lw=3)
        >>> ax.plot(x, f(x), 'r', lw=8, alpha=0.4)
        >>> ax.grid(True)
        >>> plt.show()

        """
        k = len(t) - 2
        t = _as_float_array(t)
        t = np.r_[(t[0]-1,) * k, t, (t[-1]+1,) * k]
        c = np.zeros_like(t)
        c[k] = 1.
        return cls.construct_fast(t, c, k, extrapolate)

    @classmethod
    def __call__(self, x, nu=0, extrapolate=None):
        """
        Evaluate a spline function.

        Parameters
        ----------
        x : array_like
            points to evaluate the spline at.
        nu : int, optional
            derivative to evaluate (default is 0).
        extrapolate : bool or 'periodic', optional
            whether to extrapolate based on the first and last intervals
            or return nans. If 'periodic', periodic extrapolation is used.
            Default is `self.extrapolate`.

        Returns
        -------
        y : array_like
            Shape is determined by replacing the interpolation axis
            in the coefficient array with the shape of `x`.

        """
        if extrapolate is None:
            extrapolate = self.extrapolate  # 如果没有指定 extrapolate 参数，则使用默认值 self.extrapolate
        x = np.asarray(x)  # 将输入 x 转换为 NumPy 数组
        x_shape, x_ndim = x.shape, x.ndim  # 记录输入 x 的原始形状和维度数
        x = np.ascontiguousarray(x.ravel(), dtype=np.float64)  # 将输入 x 摊平并转换为 C 连续的 float64 类型数组

        # 如果 extrapolate 参数为 'periodic'，则进行周期性外推
        if extrapolate == 'periodic':
            n = self.t.size - self.k - 1
            x = self.t[self.k] + (x - self.t[self.k]) % (self.t[n] - self.t[self.k])
            extrapolate = False  # 将 extrapolate 参数重置为 False，关闭周期性外推功能

        out = np.empty((len(x), prod(self.c.shape[1:])), dtype=self.c.dtype)  # 创建一个空的输出数组，形状由输入 x 和系数数组的后续维度决定
        self._ensure_c_contiguous()  # 确保系数数组 self.c 是 C 连续的
        self._evaluate(x, nu, extrapolate, out)  # 调用 _evaluate 方法计算样条插值
        out = out.reshape(x_shape + self.c.shape[1:])  # 将计算得到的输出重新调整为原始 x 的形状加上系数数组的后续维度形状

        if self.axis != 0:
            # 将计算得到的值移动到指定的插值轴
            l = list(range(out.ndim))
            l = l[x_ndim:x_ndim+self.axis] + l[:x_ndim] + l[x_ndim+self.axis:]
            out = out.transpose(l)  # 对输出数组进行轴变换

        return out

    def _evaluate(self, xp, nu, extrapolate, out):
        _bspl.evaluate_spline(self.t, self.c.reshape(self.c.shape[0], -1),
                              self.k, xp, nu, extrapolate, out)
        # 调用 Cython 实现的 evaluate_spline 函数来执行实际的样条插值计算

    def _ensure_c_contiguous(self):
        """
        c and t may be modified by the user. The Cython code expects
        that they are C contiguous.

        """
        if not self.t.flags.c_contiguous:
            self.t = self.t.copy()  # 如果 t 不是 C 连续的，则复制一份使之成为 C 连续的
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()  # 如果 c 不是 C 连续的，则复制一份使之成为 C 连续的
    def derivative(self, nu=1):
        """Return a B-spline representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Derivative order.
            Default is 1.

        Returns
        -------
        b : BSpline object
            A new instance representing the derivative.

        See Also
        --------
        splder, splantider

        """
        c = self.c
        # 如果需要，对 c 数组进行填充以匹配 t 数组的长度
        ct = len(self.t) - len(c)
        if ct > 0:
            c = np.r_[c, np.zeros((ct,) + c.shape[1:])]
        # 调用底层函数计算导数的 B-spline 对象
        tck = _fitpack_impl.splder((self.t, c, self.k), nu)
        # 使用计算得到的参数构造新的 B-spline 对象并返回
        return self.construct_fast(*tck, extrapolate=self.extrapolate,
                                   axis=self.axis)

    def antiderivative(self, nu=1):
        """Return a B-spline representing the antiderivative.

        Parameters
        ----------
        nu : int, optional
            Antiderivative order. Default is 1.

        Returns
        -------
        b : BSpline object
            A new instance representing the antiderivative.

        Notes
        -----
        如果计算反导数且 self.extrapolate='periodic'，
        返回的实例的 extrapolate 属性将被设为 False。
        这是因为反导数不再是周期性的，而在初始给定的 x 区间外正确评估它是困难的。

        See Also
        --------
        splder, splantider

        """
        c = self.c
        # 如果需要，对 c 数组进行填充以匹配 t 数组的长度
        ct = len(self.t) - len(c)
        if ct > 0:
            c = np.r_[c, np.zeros((ct,) + c.shape[1:])]
        # 调用底层函数计算反导数的 B-spline 对象
        tck = _fitpack_impl.splantider((self.t, c, self.k), nu)

        # 根据 self.extrapolate 的值设置 extrapolate 变量
        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        # 使用计算得到的参数构造新的 B-spline 对象并返回
        return self.construct_fast(*tck, extrapolate=extrapolate,
                                   axis=self.axis)

    @classmethod
    def insert_knot(self, x, m=1):
        """
        Insert a new knot at `x` of multiplicity `m`.

        Given the knots and coefficients of a B-spline representation, create a
        new B-spline with a knot inserted `m` times at point `x`.

        Parameters
        ----------
        x : float
            The position of the new knot
        m : int, optional
            The number of times to insert the given knot (its multiplicity).
            Default is 1.

        Returns
        -------
        spl : BSpline object
            A new BSpline object with the new knot inserted.

        Notes
        -----
        Based on algorithms from [1]_ and [2]_.

        In case of a periodic spline (``self.extrapolate == "periodic"``)
        there must be either at least k interior knots t(j) satisfying
        ``t(k+1)<t(j)<=x`` or at least k interior knots t(j) satisfying
        ``x<=t(j)<t(n-k)``.

        This routine is functionally equivalent to `scipy.interpolate.insert`.

        .. versionadded:: 1.13

        References
        ----------
        .. [1] W. Boehm, "Inserting new knots into b-spline curves.",
            Computer Aided Design, 12, p.199-201, 1980.
            :doi:`10.1016/0010-4485(80)90154-2`.
        .. [2] P. Dierckx, "Curve and surface fitting with splines, Monographs on
            Numerical Analysis", Oxford University Press, 1993.

        See Also
        --------
        scipy.interpolate.insert

        Examples
        --------
        You can insert knots into a B-spline:

        >>> import numpy as np
        >>> from scipy.interpolate import BSpline, make_interp_spline
        >>> x = np.linspace(0, 10, 5)
        >>> y = np.sin(x)
        >>> spl = make_interp_spline(x, y, k=3)
        >>> spl.t
        array([ 0.,  0.,  0.,  0.,  5., 10., 10., 10., 10.])

        Insert a single knot

        >>> spl_1 = spl.insert_knot(3)
        >>> spl_1.t
        array([ 0.,  0.,  0.,  0.,  3.,  5., 10., 10., 10., 10.])

        Insert a multiple knot

        >>> spl_2 = spl.insert_knot(8, m=3)
        >>> spl_2.t
        array([ 0.,  0.,  0.,  0.,  5.,  8.,  8.,  8., 10., 10., 10., 10.])

        """

        # Check if x is within the allowable range for knot insertion
        if x < self.t[self.k] or x > self.t[-self.k-1]:
            raise ValueError(f"Cannot insert a knot at {x}.")

        # Check if m is a positive integer
        if m <= 0:
            raise ValueError(f"`m` must be positive, got {m = }.")

        # Extract extra dimensions from coefficient array
        extradim = self.c.shape[1:]
        num_extra = prod(extradim)

        # Make copies of knot vector and coefficient array
        tt = self.t.copy()
        cc = self.c.copy()

        # Reshape coefficient array for manipulation
        cc = cc.reshape(-1, num_extra)

        # Insert the knot `m` times using low-level B-spline insertion routine
        for _ in range(m):
            tt, cc = _bspl.insert(x, tt, cc, self.k, self.extrapolate == "periodic")

        # Construct a new B-spline object with the modified knots and coefficients
        return self.construct_fast(
            tt, cc.reshape((-1,) + extradim), self.k, self.extrapolate, self.axis
        )
def _not_a_knot(x, k):
    """Given data x, construct the knot vector w/ not-a-knot BC.
    cf de Boor, XIII(12)."""
    # 将输入数据 x 转换为 NumPy 数组
    x = np.asarray(x)
    # 检查插值阶数 k 是否为奇数，若不是则抛出异常
    if k % 2 != 1:
        raise ValueError("Odd degree for now only. Got %s." % k)

    # 计算 m，即左右两侧不插值点的数量
    m = (k - 1) // 2
    # 生成不完全节点向量，使用“not-a-knot”边界条件
    t = x[m+1:-m-1]
    t = np.r_[(x[0],)*(k+1), t, (x[-1],)*(k+1)]
    return t


def _augknt(x, k):
    """Construct a knot vector appropriate for the order-k interpolation."""
    # 生成用于 k 阶插值的节点向量
    return np.r_[(x[0],)*k, x, (x[-1],)*k]


def _convert_string_aliases(deriv, target_shape):
    # 将字符串别名转换为对应的导数边界条件
    if isinstance(deriv, str):
        if deriv == "clamped":
            deriv = [(1, np.zeros(target_shape))]
        elif deriv == "natural":
            deriv = [(2, np.zeros(target_shape))]
        else:
            raise ValueError("Unknown boundary condition : %s" % deriv)
    return deriv


def _process_deriv_spec(deriv):
    # 处理导数边界条件的规范化
    if deriv is not None:
        try:
            ords, vals = zip(*deriv)
        except TypeError as e:
            msg = ("Derivatives, `bc_type`, should be specified as a pair of "
                   "iterables of pairs of (order, value).")
            raise ValueError(msg) from e
    else:
        ords, vals = [], []
    return np.atleast_1d(ords, vals)


def _woodbury_algorithm(A, ur, ll, b, k):
    '''
    Solve a cyclic banded linear system with upper right
    and lower blocks of size ``(k-1) / 2`` using
    the Woodbury formula

    Parameters
    ----------
    A : 2-D array, shape(k, n)
        Matrix of diagonals of original matrix (see
        ``solve_banded`` documentation).
    ur : 2-D array, shape(bs, bs)
        Upper right block matrix.
    ll : 2-D array, shape(bs, bs)
        Lower left block matrix.
    b : 1-D array, shape(n,)
        Vector of constant terms of the system of linear equations.
    k : int
        B-spline degree.

    Returns
    -------
    c : 1-D array, shape(n,)
        Solution of the original system of linear equations.

    Notes
    -----
    This algorithm works only for systems with banded matrix A plus
    a correction term U @ V.T, where the matrix U @ V.T gives upper right
    and lower left block of A
    The system is solved with the following steps:
        1.  New systems of linear equations are constructed:
            A @ z_i = u_i,
            u_i - column vector of U,
            i = 1, ..., k - 1
        2.  Matrix Z is formed from vectors z_i:
            Z = [ z_1 | z_2 | ... | z_{k - 1} ]
        3.  Matrix H = (1 + V.T @ Z)^{-1}
        4.  The system A' @ y = b is solved
        5.  x = y - Z @ (H @ V.T @ y)
    Also, ``n`` should be greater than ``k``, otherwise corner block
    elements will intersect with diagonals.

    Examples
    --------
    Consider the case of n = 8, k = 5 (size of blocks - 2 x 2).

    '''
    # 使用 Woodbury 公式解决带有循环带状结构的线性系统
    # 其中包括了上右和下左大小为 ``(k-1) / 2`` 的块矩阵
    pass
    # 计算 k_mod，即 k 除以 2 的余数，用于确定 U 矩阵的大小
    k_mod = k - k % 2
    # 计算 bs，即 (k - 1) / 2 的整数部分加上 (k + 1) 除以 2 的余数，用于确定 U 矩阵的大小
    bs = int((k - 1) / 2) + (k + 1) % 2
    
    # 获取 A 的列数加一，作为 U 矩阵的行数
    n = A.shape[1] + 1
    # 初始化 U 矩阵，形状为 (n-1, k_mod)，用零填充
    U = np.zeros((n - 1, k_mod))
    # 初始化 VT 矩阵，形状为 (k_mod, n-1)，用零填充，表示 V 的转置
    VT = np.zeros((k_mod, n - 1))
    
    # 设置 U 的右上角块
    U[:bs, :bs] = ur
    # 设置 VT 的对角线上的块为 1
    VT[np.arange(bs), np.arange(bs) - bs] = 1
    
    # 设置 U 的左下角块
    U[-bs:, -bs:] = ll
    # 设置 VT 的对角线下的块为 1
    VT[np.arange(bs) - bs, np.arange(bs)] = 1
    
    # 解线性带状方程组 (bs, bs)，得到 Z
    Z = solve_banded((bs, bs), A, U)
    
    # 计算 H 矩阵，解线性方程组 (I + VT @ Z) = I，其中 I 是单位矩阵
    H = solve(np.identity(k_mod) + VT @ Z, np.identity(k_mod))
    
    # 解线性带状方程组 (bs, bs)，得到 y
    y = solve_banded((bs, bs), A, b)
    # 计算 c，通过 Z @ (H @ (VT @ y)) 计算修正后的 y
    c = y - Z @ (H @ (VT @ y))
    
    # 返回计算结果 c
    return c
# 定义一个函数 `_periodic_knots`，用于生成在圆上的节点向量
def _periodic_knots(x, k):
    # 复制输入的 x 数组
    xc = np.copy(x)
    # 获取节点数量 n
    n = len(xc)
    # 如果 k 是偶数
    if k % 2 == 0:
        # 计算相邻节点之间的差值
        dx = np.diff(xc)
        # 调整 xc 中间节点的位置，使得节点布局更均匀
        xc[1: -1] -= dx[:-1] / 2
    # 重新计算节点之间的差值
    dx = np.diff(xc)
    # 初始化节点数组 t，长度为 n + 2*k
    t = np.zeros(n + 2 * k)
    # 将 xc 的值复制到 t 的中间部分
    t[k: -k] = xc
    # 填充前 k 个节点，以递减顺序排列
    for i in range(0, k):
        t[k - i - 1] = t[k - i] - dx[-(i % (n - 1)) - 1]
    # 填充后 k 个节点，以递增顺序排列
    for i in range(0, k):
        t[-k + i] = t[-k + i - 1] + dx[i % (n - 1)]
    # 返回生成的节点向量 t
    return t


# 定义一个函数 `_make_interp_per_full_matr`，用于计算具有周期边界条件的 B-样条插值的系数
def _make_interp_per_full_matr(x, y, t, k):
    '''
    Returns a solution of a system for B-spline interpolation with periodic
    boundary conditions. First ``k - 1`` rows of matrix are conditions of
    periodicity (continuity of ``k - 1`` derivatives at the boundary points).
    Last ``n`` rows are interpolation conditions.
    RHS is ``k - 1`` zeros and ``n`` ordinates in this case.

    Parameters
    ----------
    x : 1-D array, shape (n,)
        Values of x - coordinate of a given set of points.
    y : 1-D array, shape (n,)
        Values of y - coordinate of a given set of points.
    t : 1-D array, shape(n+2*k,)
        Vector of knots.
    k : int
        The maximum degree of spline

    Returns
    -------
    c : 1-D array, shape (n+k-1,)
        B-spline coefficients

    Notes
    -----
    ``t`` is supposed to be taken on circle.

    '''

    # 将输入的 x, y, t 转换为 numpy 数组
    x, y, t = map(np.asarray, (x, y, t))

    # 获取点的数量 n
    n = x.size
    # 初始化一个 n+k-1 行 n+k-1 列的全零矩阵 matr
    matr = np.zeros((n + k - 1, n + k - 1))

    # 处理边界点上导数的条件
    for i in range(k - 1):
        # 计算左边界点处的 B-样条基函数值
        bb = _bspl.evaluate_all_bspl(t, k, x[0], k, nu=i + 1)
        matr[i, : k + 1] += bb
        # 计算右边界点处的 B-样条基函数值
        bb = _bspl.evaluate_all_bspl(t, k, x[-1], n + k - 1, nu=i + 1)[:-1]
        matr[i, -k:] -= bb

    # 填充插值条件的矩阵部分
    for i in range(n):
        xval = x[i]
        # 查找 xval 所在的区间
        if xval == t[k]:
            left = k
        else:
            left = np.searchsorted(t, xval) - 1
        # 计算该行对应的 B-样条基函数值
        bb = _bspl.evaluate_all_bspl(t, k, xval, left)
        matr[i + k - 1, left-k:left+1] = bb

    # 初始化右侧向量 b，前 k-1 个元素为零，后面是 y 的值
    b = np.r_[[0] * (k - 1), y]

    # 解线性方程组，得到 B-样条系数向量 c
    c = solve(matr, b)
    # 返回系数向量 c
    return c


# 定义一个函数 `_make_periodic_spline`，用于计算具有周期边界条件的 B-样条插值
def _make_periodic_spline(x, y, t, k, axis):
    '''
    Compute the (coefficients of) interpolating B-spline with periodic
    boundary conditions.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas.
    y : array_like, shape (n,)
        Ordinates.
    k : int
        B-spline degree.
    t : array_like, shape (n + 2 * k,).
        Knots taken on a circle, ``k`` on the left and ``k`` on the right
        of the vector ``x``.

    Returns
    -------
    b : a BSpline object of the degree ``k`` and with knots ``t``.

    Notes
    -----
    The original system is formed by ``n + k - 1`` equations where the first
    ``k - 1`` of them stand for the ``k - 1`` derivatives continuity on the

    '''

    # 参数转换为 numpy 数组
    x, y, t = map(np.asarray, (x, y, t))

    # 计算点的数量 n
    n = x.size
    # 初始化一个 n+k-1 行 n+k-1 列的全零矩阵 matr
    matr = np.zeros((n + k - 1, n + k - 1))

    # 处理边界点上导数的条件
    for i in range(k - 1):
        # 计算左边界点处的 B-样条基函数值
        bb = _bspl.evaluate_all_bspl(t, k, x[0], k, nu=i + 1)
        matr[i, : k + 1] += bb
        # 计算右边界点处的 B-样条基函数值
        bb = _bspl.evaluate_all_bspl(t, k, x[-1], n + k - 1, nu=i + 1)[:-1]
        matr[i, -k:] -= bb

    # 填充插值条件的矩阵部分
    for i in range(n):
        xval = x[i]
        # 查找 xval 所在的区间
        if xval == t[k]:
            left = k
        else:
            left = np.searchsorted(t, xval) - 1
        # 计算该行对应的 B-样条基函数值
        bb = _bspl.evaluate_all_bspl(t, k, xval, left)
        matr[i + k - 1, left-k:left+1] = bb

    # 初始化右侧向量 b，前 k-1 个元素为零，后面是 y 的值
    b = np.r_[[0] * (k - 1), y]

    # 解线性方程组，得到 B-样条系数向量 c
    c = solve(matr, b)
    # 返回 B-样条对象 b，其中包括度数 k 和节点 t
    return b
    '''
    This code implements a B-spline interpolation algorithm that optimizes computations using the Woodbury formula.
    It handles cases where the number of input points is less than or equal to the spline degree.
    The algorithm involves cyclic shift of values of B-splines due to equality of unknown coefficients.
    It constructs a matrix system with upper right and lower left blocks to optimize computations.
    
    Parameters:
    - y: Input data points
    - x: Input data points
    - t: Knot vector
    - k: Spline degree
    - axis: Axis along which the spline should be constructed
    
    Returns:
    - BSpline object constructed using the input data points and parameters
    
    '''
    
    n = y.shape[0]  # Get the number of input data points
    
    extradim = prod(y.shape[1:])  # Calculate the product of dimensions excluding the first one
    y_new = y.reshape(n, extradim)  # Reshape the input data points
    c = np.zeros((n + k - 1, extradim))  # Initialize the coefficient matrix
    
    # Handle the case where the number of input points is less than or equal to the spline degree
    if n <= k:
        for i in range(extradim):
            c[:, i] = _make_interp_per_full_matr(x, y_new[:, i], t, k)  # Interpolate using full matrix
        c = np.ascontiguousarray(c.reshape((n + k - 1,) + y.shape[1:]))  # Reshape the coefficient matrix
        return BSpline.construct_fast(t, c, k, extrapolate='periodic', axis=axis)  # Construct BSpline object
    
    nt = len(t) - k - 1  # Calculate the size of block elements
    
    kul = int(k / 2)  # Calculate the size of upper left block
    
    ab = np.zeros((3 * k + 1, nt), dtype=np.float64, order='F')  # Initialize the matrix for block elements
    
    ur = np.zeros((kul, kul))  # Initialize upper right block
    ll = np.zeros_like(ur)  # Initialize lower left block
    
    # Shift all non-zero elements to the end of the matrix
    _bspl._colloc(x, t, k, ab, offset=k)
    
    ab = ab[-k - (k + 1) % 2:, :]  # Remove zeros before the matrix
    
    # Extract diagonals of block matrices
    for i in range(kul):
        ur += np.diag(ab[-i - 1, i: kul], k=i)  # Extract upper right block diagonals
        ll += np.diag(ab[i, -kul - (k % 2): n - 1 + 2 * kul - i], k=-i)  # Extract lower left block diagonals
    
    A = ab[:, kul: -k + kul]  # Extract elements excluding the last point
    
    # Apply Woodbury algorithm to optimize computations
    for i in range(extradim):
        cc = _woodbury_algorithm(A, ur, ll, y_new[:, i][:-1], k)  # Apply Woodbury algorithm
        c[:, i] = np.concatenate((cc[-kul:], cc, cc[:kul + k % 2]))  # Concatenate the results
    c = np.ascontiguousarray(c.reshape((n + k - 1,) + y.shape[1:]))  # Reshape the coefficient matrix
    return BSpline.construct_fast(t, c, k, extrapolate='periodic', axis=axis)  # Construct BSpline object
# 计算插值 B-spline 的系数

def make_interp_spline(x, y, k=3, t=None, bc_type=None, axis=0,
                       check_finite=True):
    """Compute the (coefficients of) interpolating B-spline.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas. 横坐标数组，长度为 n。
    y : array_like, shape (n, ...)
        Ordinates. 纵坐标数组，形状为 (n, ...)。
    k : int, optional
        B-spline degree. Default is cubic, ``k = 3``. B-spline 的次数，默认为三次（立方）。
    t : array_like, shape (nt + k + 1,), optional.
        Knots. 节点数组，长度为 nt + k + 1。
        The number of knots needs to agree with the number of data points and
        the number of derivatives at the edges. Specifically, ``nt - n`` must
        equal ``len(deriv_l) + len(deriv_r)``.
        节点数量必须与数据点数量及边缘导数的数量相符。具体而言，``nt - n`` 必须等于 ``len(deriv_l) + len(deriv_r)``。
    bc_type : 2-tuple or None
        Boundary conditions. 边界条件。默认为 None，即自动选择边界条件。
        Otherwise, it must be a length-two tuple where the first
        element (``deriv_l``) sets the boundary conditions at ``x[0]`` and
        the second element (``deriv_r``) sets the boundary conditions at
        ``x[-1]``. Each of these must be an iterable of pairs
        ``(order, value)`` which gives the values of derivatives of specified
        orders at the given edge of the interpolation interval.
        另外，也可以使用以下字符串别名：

        * ``"clamped"``: 边缘的一阶导数为零。等同于 ``bc_type=([(1, 0.0)], [(1, 0.0)])``。
        * ``"natural"``: 边缘的二阶导数为零。等同于 ``bc_type=([(2, 0.0)], [(2, 0.0)])``。
        * ``"not-a-knot"`` (默认): 前两段是同一个多项式。等同于 ``bc_type=None``。
        * ``"periodic"``: 边缘处的值和前 ``k-1`` 阶导数相等。

    axis : int, optional
        Interpolation axis. Default is 0. 插值的轴，默认为 0。
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default is True. 是否检查输入数组只包含有限数字。禁用此选项可能会提升性能，但如果输入包含无穷大或 NaN，可能会导致问题（崩溃、无法终止）。

    Returns
    -------
    b : a BSpline object of the degree ``k`` and with knots ``t``.
    返回一个 B-spline 对象，其次数为 ``k``，节点为 ``t``。

    See Also
    --------
    BSpline : base class representing the B-spline objects
    CubicSpline : a cubic spline in the polynomial basis
    make_lsq_spline : a similar factory function for spline fitting
    UnivariateSpline : a wrapper over FITPACK spline fitting routines
    splrep : a wrapper over FITPACK spline fitting routines

    Examples
    --------

    Use cubic interpolation on Chebyshev nodes:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> def cheb_nodes(N):
    ...     jj = 2.*np.arange(N) + 1
    ...     x = np.cos(np.pi * jj / 2 / N)[::-1]
    ...     return x

    >>> x = cheb_nodes(20)
    >>> y = np.sqrt(1 - x**2)
    # 导入需要的函数和类
    >>> from scipy.interpolate import BSpline, make_interp_spline
    >>> b = make_interp_spline(x, y)
    # 使用 make_interp_spline 函数创建 B 样条插值对象 b，使用给定的 x 和 y 数据进行插值

    >>> np.allclose(b(x), y)
    # 检查插值对象 b 在给定的 x 点上的计算值是否与原始 y 值非常接近
    True

    Note that the default is a cubic spline with a not-a-knot boundary condition
    # 注意，默认情况下是一个三次样条插值，并且使用 not-a-knot 边界条件

    >>> b.k
    # 获取 B 样条插值对象 b 的阶数 k
    3

    Here we use a 'natural' spline, with zero 2nd derivatives at edges:
    # 这里使用 'natural' 自然样条插值，边界处的二阶导数为零：

    >>> l, r = [(2, 0.0)], [(2, 0.0)]
    >>> b_n = make_interp_spline(x, y, bc_type=(l, r))  # or, bc_type="natural"
    # 使用自定义的边界条件 (l, r) 或者字符串 "natural" 创建自然样条插值对象 b_n
    >>> np.allclose(b_n(x), y)
    # 检查自然样条插值对象 b_n 在给定的 x 点上的计算值是否与原始 y 值非常接近
    True

    >>> x0, x1 = x[0], x[-1]
    >>> np.allclose([b_n(x0, 2), b_n(x1, 2)], [0, 0])
    # 检查自然样条插值对象 b_n 在给定的两个边界点 x0 和 x1 处的二阶导数是否为零
    True

    Interpolation of parametric curves is also supported. As an example, we
    compute a discretization of a snail curve in polar coordinates
    # 还支持参数曲线的插值。例如，我们计算极坐标中蜗牛曲线的离散化：

    >>> phi = np.linspace(0, 2.*np.pi, 40)
    >>> r = 0.3 + np.cos(phi)
    >>> x, y = r*np.cos(phi), r*np.sin(phi)  # convert to Cartesian coordinates
    # 计算极坐标中蜗牛曲线的离散化，然后转换为笛卡尔坐标系

    Build an interpolating curve, parameterizing it by the angle
    # 构建一个插值曲线，通过角度进行参数化

    >>> spl = make_interp_spline(phi, np.c_[x, y])
    # 使用 make_interp_spline 函数创建参数化曲线插值对象 spl

    Evaluate the interpolant on a finer grid (note that we transpose the result
    to unpack it into a pair of x- and y-arrays)
    # 在更细的网格上评估插值器（注意我们转置结果以将其解压缩为 x 和 y 数组对）

    >>> phi_new = np.linspace(0, 2.*np.pi, 100)
    >>> x_new, y_new = spl(phi_new).T
    # 在新的 phi_new 网格上评估插值器 spl，然后将结果转置以得到 x_new 和 y_new 数组

    Plot the result
    # 绘制结果

    >>> plt.plot(x, y, 'o')
    >>> plt.plot(x_new, y_new, '-')
    >>> plt.show()

    Build a B-spline curve with 2 dimensional y
    # 使用二维 y 构建一个 B 样条曲线

    >>> x = np.linspace(0, 2*np.pi, 10)
    >>> y = np.array([np.sin(x), np.cos(x)])

    Periodic condition is satisfied because y coordinates of points on the ends
    are equivalent
    # 周期条件得到满足，因为端点上的 y 坐标是等价的

    >>> ax = plt.axes(projection='3d')
    >>> xx = np.linspace(0, 2*np.pi, 100)
    >>> bspl = make_interp_spline(x, y, k=5, bc_type='periodic', axis=1)
    # 使用周期性边界条件和轴索引 1 创建一个阶数为 5 的 B 样条插值对象 bspl
    >>> ax.plot3D(xx, *bspl(xx))
    # 在 3D 图上绘制 bspl 在 xx 上的结果
    >>> ax.scatter3D(x, *y, color='red')
    # 在 3D 图上绘制原始数据点
    >>> plt.show()

    """
    # 将边界条件 bc_type 转换为字符串别名
    if bc_type is None or bc_type == 'not-a-knot' or bc_type == 'periodic':
        deriv_l, deriv_r = None, None
    elif isinstance(bc_type, str):
        deriv_l, deriv_r = bc_type, bc_type
    else:
        try:
            deriv_l, deriv_r = bc_type
        except TypeError as e:
            raise ValueError("Unknown boundary condition: %s" % bc_type) from e

    y = np.asarray(y)

    axis = normalize_axis_index(axis, y.ndim)

    x = _as_float_array(x, check_finite)
    y = _as_float_array(y, check_finite)

    y = np.moveaxis(y, axis, 0)    # 现在内部插值轴是零

    # 对输入进行检查
    if bc_type == 'periodic' and not np.allclose(y[0], y[-1], atol=1e-15):
        raise ValueError("First and last points does not match while "
                         "periodic case expected")
    if x.size != y.shape[0]:
        raise ValueError(f'Shapes of x {x.shape} and y {y.shape} are incompatible')
    if np.any(x[1:] == x[:-1]):
        raise ValueError("Expect x to not have duplicates")
    # 检查输入数组 x 是否为一维且严格递增，若不是则抛出数值错误异常
    if x.ndim != 1 or np.any(x[1:] < x[:-1]):
        raise ValueError("Expect x to be a 1D strictly increasing sequence.")

    # 对 k=0 的特殊情况进行快速处理
    if k == 0:
        # 如果 t、deriv_l、deriv_r 中有任何一个不为 None，则抛出数值错误异常
        if any(_ is not None for _ in (t, deriv_l, deriv_r)):
            raise ValueError("Too much info for k=0: t and bc_type can only "
                             "be None.")
        # 构造新的节点序列 t，将 x 和 x 的最后一个元素连接起来
        t = np.r_[x, x[-1]]
        # 将 y 转换为数组 c，并确保其连续性，同时根据数据类型获取相应的 dtype
        c = np.asarray(y)
        c = np.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
        # 调用 BSpline.construct_fast 方法构建 B 样条曲线并返回结果
        return BSpline.construct_fast(t, c, k, axis=axis)

    # 对 k=1 的特殊情况进行处理（例如 Lyche 和 Morken，Eq.(2.16)）
    if k == 1 and t is None:
        # 如果 deriv_l 和 deriv_r 不同时为 None，则抛出数值错误异常
        if not (deriv_l is None and deriv_r is None):
            raise ValueError("Too much info for k=1: bc_type can only be None.")
        # 构造新的节点序列 t，将 x 的第一个元素、x、x 的最后一个元素连接起来
        t = np.r_[x[0], x, x[-1]]
        # 将 y 转换为数组 c，并确保其连续性，同时根据数据类型获取相应的 dtype
        c = np.asarray(y)
        c = np.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
        # 调用 BSpline.construct_fast 方法构建 B 样条曲线并返回结果
        return BSpline.construct_fast(t, c, k, axis=axis)

    # 将 k 转换为整数类型
    k = operator.index(k)

    # 如果 bc_type 为 'periodic' 且 t 不为 None，则抛出未实现的错误异常
    if bc_type == 'periodic' and t is not None:
        raise NotImplementedError("For periodic case t is constructed "
                                  "automatically and can not be passed "
                                  "manually")

    # 如果未提供节点序列 t，则根据情况生成一个合理的节点序列
    if t is None:
        # 如果 deriv_l 和 deriv_r 都为 None，则根据不同的 bc_type 生成节点序列 t
        if deriv_l is None and deriv_r is None:
            if bc_type == 'periodic':
                # 对于周期性，自动生成节点序列 t
                t = _periodic_knots(x, k)
            elif k == 2:
                # 对于 k=2，使用一种 ad hoc 方法生成节点序列 t，类似于 not-a-knot 方法
                t = (x[1:] + x[:-1]) / 2.
                t = np.r_[(x[0],)*(k+1),
                          t[1:-1],
                          (x[-1],)*(k+1)]
            else:
                # 对于其他情况，使用 not-a-knot 方法生成节点序列 t
                t = _not_a_knot(x, k)
        else:
            # 如果 deriv_l 或 deriv_r 不为 None，则使用增广节点方法生成节点序列 t
            t = _augknt(x, k)

    # 将节点序列 t 转换为浮点数数组，并根据需要检查其有限性
    t = _as_float_array(t, check_finite)

    # 检查 k 是否为非负数，若不是则抛出数值错误异常
    if k < 0:
        raise ValueError("Expect non-negative k.")
    
    # 检查节点序列 t 是否为一维且已排序数组，若不是则抛出数值错误异常
    if t.ndim != 1 or np.any(t[1:] < t[:-1]):
        raise ValueError("Expect t to be a 1-D sorted array_like.")
    
    # 检查节点序列 t 的长度是否足够支持给定的数据点和阶数 k，若不足则抛出数值错误异常
    if t.size < x.size + k + 1:
        raise ValueError('Got %d knots, need at least %d.' %
                         (t.size, x.size + k + 1))
    
    # 检查数据点 x 是否在节点序列 t 的边界内，若不在则抛出数值错误异常
    if (x[0] < t[k]) or (x[-1] > t[-k]):
        raise ValueError('Out of bounds w/ x = %s.' % x)

    # 如果 bc_type 为 'periodic'，则调用 _make_periodic_spline 方法生成周期样条曲线并返回结果
    if bc_type == 'periodic':
        return _make_periodic_spline(x, y, t, k, axis)

    # 在此处：deriv_l, r = [(nu, value), ...]
    # 将 deriv_l 转换为字符串别名，并根据 y 的形状生成相应的导数边界条件
    deriv_l = _convert_string_aliases(deriv_l, y.shape[1:])
    deriv_l_ords, deriv_l_vals = _process_deriv_spec(deriv_l)
    nleft = deriv_l_ords.shape[0]

    # 将 deriv_r 转换为字符串别名，并根据 y 的形状生成相应的导数边界条件
    deriv_r = _convert_string_aliases(deriv_r, y.shape[1:])
    deriv_r_ords, deriv_r_vals = _process_deriv_spec(deriv_r)
    nright = deriv_r_ords.shape[0]

    # 检查左侧导数边界条件的阶数是否在有效范围内，若不在则抛出数值错误异常
    if not all(0 <= i <= k for i in deriv_l_ords):
        raise ValueError(f"Bad boundary conditions at {x[0]}.")

    # 检查右侧导数边界条件的阶数是否在有效范围内，若不在则抛出数值错误异常
    if not all(0 <= i <= k for i in deriv_r_ords):
        raise ValueError(f"Bad boundary conditions at {x[-1]}.")
    # 计算 `n` 个条件对应的 `nt-n` 个导数
    n = x.size
    nt = t.size - k - 1

    # 检查边界导数的数量是否与预期相符
    if nt - n != nleft + nright:
        raise ValueError("The number of derivatives at boundaries does not "
                         f"match: expected {nt-n}, got {nleft}+{nright}")

    # 如果 `y` 数组的大小为零，返回一个零数组
    if y.size == 0:
        c = np.zeros((nt,) + y.shape[1:], dtype=float)
        return BSpline.construct_fast(t, c, k, axis=axis)

    # 设置左手边矩阵（LHS）：插值矩阵 + 边界处的导数
    kl = ku = k
    ab = np.zeros((2*kl + ku + 1, nt), dtype=np.float64, order='F')
    _bspl._colloc(x, t, k, ab, offset=nleft)
    # 如果有左边的导数，处理左边界处的导数
    if nleft > 0:
        _bspl._handle_lhs_derivatives(t, k, x[0], ab, kl, ku,
                                      deriv_l_ords.astype(np.dtype("long")))
    # 如果有右边的导数，处理右边界处的导数
    if nright > 0:
        _bspl._handle_lhs_derivatives(t, k, x[-1], ab, kl, ku,
                                      deriv_r_ords.astype(np.dtype("long")),
                                      offset=nt-nright)

    # 设置右手边矩阵（RHS）：需要插值的值（+ 如果有的话的导数值）
    extradim = prod(y.shape[1:])
    rhs = np.empty((nt, extradim), dtype=y.dtype)
    # 如果有左边的导数，设置左边界处的导数值
    if nleft > 0:
        rhs[:nleft] = deriv_l_vals.reshape(-1, extradim)
    # 设置插值值的部分
    rhs[nleft:nt - nright] = y.reshape(-1, extradim)
    # 如果有右边的导数，设置右边界处的导数值
    if nright > 0:
        rhs[nt - nright:] = deriv_r_vals.reshape(-1, extradim)

    # 解线性方程组 Ab @ x = rhs；这是 linalg.solve_banded 的相关部分
    # 如果需要检查有限性，使用 np.asarray_chkfinite 检查 ab 和 rhs
    if check_finite:
        ab, rhs = map(np.asarray_chkfinite, (ab, rhs))
    # 调用 LAPACK 函数 gbsv 解方程
    gbsv, = get_lapack_funcs(('gbsv',), (ab, rhs))
    lu, piv, c, info = gbsv(kl, ku, ab, rhs,
                            overwrite_ab=True, overwrite_b=True)

    # 检查解的信息
    if info > 0:
        raise LinAlgError("Collocation matrix is singular.")
    elif info < 0:
        raise ValueError('illegal value in %d-th argument of internal gbsv' % -info)

    # 重新整理结果数组，并返回 BSpline 对象
    c = np.ascontiguousarray(c.reshape((nt,) + y.shape[1:]))
    return BSpline.construct_fast(t, c, k, axis=axis)
# 定义一个函数，用于计算基于最小二乘法（LSQ）拟合的 B-spline 的系数
def make_lsq_spline(x, y, t, k=3, w=None, axis=0, check_finite=True):
    r"""Compute the (coefficients of) an LSQ (Least SQuared) based
    fitting B-spline.

    The result is a linear combination

    .. math::

            S(x) = \sum_j c_j B_j(x; t)

    of the B-spline basis elements, :math:`B_j(x; t)`, which minimizes

    .. math::

        \sum_{j} \left( w_j \times (S(x_j) - y_j) \right)^2

    Parameters
    ----------
    x : array_like, shape (m,)
        Abscissas.
    y : array_like, shape (m, ...)
        Ordinates.
    t : array_like, shape (n + k + 1,).
        Knots.
        Knots and data points must satisfy Schoenberg-Whitney conditions.
    k : int, optional
        B-spline degree. Default is cubic, ``k = 3``.
    w : array_like, shape (m,), optional
        Weights for spline fitting. Must be positive. If ``None``,
        then weights are all equal.
        Default is ``None``.
    axis : int, optional
        Interpolation axis. Default is zero.
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default is True.

    Returns
    -------
    b : a BSpline object of the degree ``k`` with knots ``t``.

    See Also
    --------
    BSpline : base class representing the B-spline objects
    make_interp_spline : a similar factory function for interpolating splines
    LSQUnivariateSpline : a FITPACK-based spline fitting routine
    splrep : a FITPACK-based fitting routine

    Notes
    -----
    The number of data points must be larger than the spline degree ``k``.

    Knots ``t`` must satisfy the Schoenberg-Whitney conditions,
    i.e., there must be a subset of data points ``x[j]`` such that
    ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.

    Examples
    --------
    Generate some noisy data:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)

    Now fit a smoothing cubic spline with a pre-defined internal knots.
    Here we make the knot vector (k+1)-regular by adding boundary knots:

    >>> from scipy.interpolate import make_lsq_spline, BSpline
    >>> t = [-1, 0, 1]
    >>> k = 3
    >>> t = np.r_[(x[0],)*(k+1),
    ...           t,
    ...           (x[-1],)*(k+1)]
    >>> spl = make_lsq_spline(x, y, t, k)

    For comparison, we also construct an interpolating spline for the same
    set of data:

    >>> from scipy.interpolate import make_interp_spline
    >>> spl_i = make_interp_spline(x, y)

    Plot both:

    >>> xs = np.linspace(-3, 3, 100)
    >>> plt.plot(x, y, 'ro', ms=5)
    >>> plt.plot(xs, spl(xs), 'g-', lw=3, label='LSQ spline')
    >>> plt.plot(xs, spl_i(xs), 'b-', lw=3, alpha=0.7, label='interp spline')
    # 在图表上添加图例，并选择最佳位置进行显示
    >>> plt.legend(loc='best')
    # 展示当前图表
    >>> plt.show()

    # NaN 处理：如果输入数组包含 NaN 值，由于底层样条拟合例程无法处理 NaN，因此结果是无用的。
    # 一种解决方法是对非数字数据点使用零权重：
    >>> y[8] = np.nan
    # 创建一个布尔数组，标记出 NaN 的位置
    >>> w = np.isnan(y)
    # 将所有 NaN 替换为 0，对应的权重为零
    >>> y[w] = 0.
    # 使用修改后的权重数组 w（反转后）创建最小二乘样条拟合对象
    >>> tck = make_lsq_spline(x, y, t, w=~w)

    # 注意：需要将 NaN 替换为数值（精确值并不重要，只要对应的权重为零即可）。

    """
    # 将输入的 x、y、t 转换为浮点数数组，并可选地检查其有限性
    x = _as_float_array(x, check_finite)
    y = _as_float_array(y, check_finite)
    t = _as_float_array(t, check_finite)
    # 如果给定了权重 w，则也转换为浮点数数组；否则初始化为与 x 相同形状的全 1 数组
    if w is not None:
        w = _as_float_array(w, check_finite)
    else:
        w = np.ones_like(x)
    # 将操作数 k 转换为整数
    k = operator.index(k)

    # 根据轴的索引值将 y 的轴移动到第 0 位置，以便内部插值轴是零
    y = np.moveaxis(y, axis, 0)

    # 检查输入的 x 是否为 1-D 并且已排序
    if x.ndim != 1 or np.any(x[1:] - x[:-1] <= 0):
        raise ValueError("Expect x to be a 1-D sorted array_like.")
    # 检查 x 的长度是否大于 k+1
    if x.shape[0] < k+1:
        raise ValueError("Need more x points.")
    # 检查 k 是否为非负数
    if k < 0:
        raise ValueError("Expect non-negative k.")
    # 检查输入的 t 是否为 1-D 并且已排序
    if t.ndim != 1 or np.any(t[1:] - t[:-1] < 0):
        raise ValueError("Expect t to be a 1-D sorted array_like.")
    # 检查 x 和 y 的形状是否兼容
    if x.size != y.shape[0]:
        raise ValueError(f'Shapes of x {x.shape} and y {y.shape} are incompatible')
    # 如果 k > 0，则检查 x 是否在指定的 t 区间内
    if k > 0 and np.any((x < t[k]) | (x > t[-k])):
        raise ValueError('Out of bounds w/ x = %s.' % x)
    # 检查 x 和 w 的形状是否兼容
    if x.size != w.size:
        raise ValueError(f'Shapes of x {x.shape} and w {w.shape} are incompatible')

    # 系数的数量
    n = t.size - k - 1

    # 构建 A.T @ A 和 rhs，其中 A 是配点矩阵，rhs = A.T @ y 用于解决最小二乘问题 ``A.T @ A @ c = A.T @ y``
    lower = True
    extradim = np.prod(y.shape[1:])
    ab = np.zeros((k+1, n), dtype=np.float64, order='F')
    rhs = np.zeros((n, extradim), dtype=y.dtype, order='F')
    _bspl._norm_eq_lsq(x, t, k,
                       y.reshape(-1, extradim),
                       w,
                       ab, rhs)
    rhs = rhs.reshape((n,) + y.shape[1:])

    # 有观测矩阵和 rhs，可以解决最小二乘问题
    cho_decomp = cholesky_banded(ab, overwrite_ab=True, lower=lower,
                                 check_finite=check_finite)
    c = cho_solve_banded((cho_decomp, lower), rhs, overwrite_b=True,
                         check_finite=check_finite)

    # 将 c 转换为连续的数组
    c = np.ascontiguousarray(c)
    # 使用结果构造并返回一个快速构造的 B 样条对象
    return BSpline.construct_fast(t, c, k, axis=axis)
#############################
#  Smoothing spline helpers #
#############################

# 计算最优的广义交叉验证（GCV）参数
def _compute_optimal_gcv_parameter(X, wE, y, w):
    """
    Returns an optimal regularization parameter from the GCV criteria [1].

    Parameters
    ----------
    X : array, shape (5, n)
        5 bands of the design matrix ``X`` stored in LAPACK banded storage.
        设计矩阵 ``X`` 的5个带区存储在LAPACK带状存储中。
    wE : array, shape (5, n)
        5 bands of the penalty matrix :math:`W^{-1} E` stored in LAPACK banded
        storage.
        惩罚矩阵 :math:`W^{-1} E` 的5个带区存储在LAPACK带状存储中。
    y : array, shape (n,)
        Ordinates.
        纵坐标。
    w : array, shape (n,)
        Vector of weights.
        权重向量。

    Returns
    -------
    lam : float
        An optimal from the GCV criteria point of view regularization
        parameter.
        根据GCV标准返回的最优正则化参数。

    Notes
    -----
    No checks are performed.
    没有进行检查。

    References
    ----------
    .. [1] G. Wahba, "Estimating the smoothing parameter" in Spline models
        for observational data, Philadelphia, Pennsylvania: Society for
        Industrial and Applied Mathematics, 1990, pp. 45-65.
        :doi:`10.1137/1.9781611970128`

    """

    # 计算带状对称矩阵的乘积 X^T W Y
    def compute_banded_symmetric_XT_W_Y(X, w, Y):
        """
        Assuming that the product :math:`X^T W Y` is symmetric and both ``X``
        and ``Y`` are 5-banded, compute the unique bands of the product.

        Parameters
        ----------
        X : array, shape (5, n)
            5 bands of the matrix ``X`` stored in LAPACK banded storage.
            存储在LAPACK带状存储中的矩阵 ``X`` 的5个带区。
        w : array, shape (n,)
            Array of weights
            权重数组。
        Y : array, shape (5, n)
            5 bands of the matrix ``Y`` stored in LAPACK banded storage.
            存储在LAPACK带状存储中的矩阵 ``Y`` 的5个带区。

        Returns
        -------
        res : array, shape (4, n)
            The result of the product :math:`X^T Y` stored in the banded way.
            以带状方式存储的乘积 :math:`X^T Y` 的结果。

        Notes
        -----
        As far as the matrices ``X`` and ``Y`` are 5-banded, their product
        :math:`X^T W Y` is 7-banded. It is also symmetric, so we can store only
        unique diagonals.
        只要矩阵 ``X`` 和 ``Y`` 是5带区的，它们的乘积 :math:`X^T W Y` 就是7带区的。
        它也是对称的，因此我们只需存储唯一的对角线元素。

        """
        # 计算 W Y
        W_Y = np.copy(Y)

        # 对称性调整
        W_Y[2] *= w
        for i in range(2):
            W_Y[i, 2 - i:] *= w[:-2 + i]
            W_Y[3 + i, :-1 - i] *= w[1 + i:]

        n = X.shape[1]
        res = np.zeros((4, n))
        for i in range(n):
            for j in range(min(n-i, 4)):
                res[-j-1, i + j] = sum(X[j:, i] * W_Y[:5-j, i + j])
        return res
    def compute_b_inv(A):
        """
        Inverse 3 central bands of matrix :math:`A=U^T D^{-1} U` assuming that
        ``U`` is a unit upper triangular banded matrix using an algorithm
        proposed in [1].

        Parameters
        ----------
        A : array, shape (4, n)
            Matrix to inverse, stored in LAPACK banded storage.

        Returns
        -------
        B : array, shape (4, n)
            3 unique bands of the symmetric matrix that is an inverse to ``A``.
            The first row is filled with zeros.

        Notes
        -----
        The algorithm is based on the cholesky decomposition and, therefore,
        in case matrix ``A`` is close to not positive defined, the function
        raises LinalgError.

        Both matrices ``A`` and ``B`` are stored in LAPACK banded storage.

        References
        ----------
        .. [1] M. F. Hutchinson and F. R. de Hoog, "Smoothing noisy data with
            spline functions," Numerische Mathematik, vol. 47, no. 1,
            pp. 99-106, 1985.
            :doi:`10.1007/BF01389878`

        """

        def find_b_inv_elem(i, j, U, D, B):
            rng = min(3, n - i - 1)  # Determine the range of elements to process based on bandwidth and matrix dimensions
            rng_sum = 0.  # Initialize the sum of elements in the specified range
            if j == 0:
                # Compute element using the 2nd formula from reference [1]
                for k in range(1, rng + 1):
                    rng_sum -= U[-k - 1, i + k] * B[-k - 1, i + k]  # Accumulate terms based on matrix indices
                rng_sum += D[i]  # Add diagonal element from matrix D
                B[-1, i] = rng_sum  # Store computed value in B matrix
            else:
                # Compute element using the 1st formula from reference [1]
                for k in range(1, rng + 1):
                    diag = abs(k - j)  # Calculate diagonal offset for indexing
                    ind = i + min(k, j)  # Calculate index in U and B matrices
                    rng_sum -= U[-k - 1, i + k] * B[-diag - 1, ind + diag]  # Accumulate terms based on matrix indices
                B[-j - 1, i + j] = rng_sum  # Store computed value in B matrix

        U = cholesky_banded(A)  # Perform Cholesky decomposition on matrix A
        for i in range(2, 5):
            U[-i, i-1:] /= U[-1, :-i+1]  # Normalize elements in U matrix
        D = 1. / (U[-1])**2  # Compute diagonal elements for matrix D
        U[-1] /= U[-1]  # Normalize elements in U matrix

        n = U.shape[1]  # Determine the size of the matrix based on its shape

        B = np.zeros(shape=(4, n))  # Initialize matrix B with zeros
        for i in range(n - 1, -1, -1):
            for j in range(min(3, n - i - 1), -1, -1):
                find_b_inv_elem(i, j, U, D, B)  # Compute elements of matrix B using helper function
        # the first row contains garbage and should be removed
        B[0] = [0.] * n  # Replace the first row of B with zeros
        return B  # Return the computed inverse matrix B
    def _gcv(lam, X, XtWX, wE, XtE):
        r"""
        Computes the generalized cross-validation criteria [1].

        Parameters
        ----------
        lam : float, (:math:`\lambda \geq 0`)
            Regularization parameter.
        X : array, shape (5, n)
            Matrix is stored in LAPACK banded storage.
        XtWX : array, shape (4, n)
            Product :math:`X^T W X` stored in LAPACK banded storage.
        wE : array, shape (5, n)
            Matrix :math:`W^{-1} E` stored in LAPACK banded storage.
        XtE : array, shape (4, n)
            Product :math:`X^T E` stored in LAPACK banded storage.

        Returns
        -------
        res : float
            Value of the GCV criteria with the regularization parameter
            :math:`\lambda`.

        Notes
        -----
        Criteria is computed from the formula (1.3.2) [3]:

        .. math:

        GCV(\lambda) = \dfrac{1}{n} \sum\limits_{k = 1}^{n} \dfrac{ \left(
        y_k - f_{\lambda}(x_k) \right)^2}{\left( 1 - \Tr{A}/n\right)^2}$.
        The criteria is discussed in section 1.3 [3].

        The numerator is computed using (2.2.4) [3] and the denominator is
        computed using an algorithm from [2] (see in the ``compute_b_inv``
        function).

        References
        ----------
        .. [1] G. Wahba, "Estimating the smoothing parameter" in Spline models
            for observational data, Philadelphia, Pennsylvania: Society for
            Industrial and Applied Mathematics, 1990, pp. 45-65.
            :doi:`10.1137/1.9781611970128`
        .. [2] M. F. Hutchinson and F. R. de Hoog, "Smoothing noisy data with
            spline functions," Numerische Mathematik, vol. 47, no. 1,
            pp. 99-106, 1985.
            :doi:`10.1007/BF01389878`
        .. [3] E. Zemlyanoy, "Generalized cross-validation smoothing splines",
            BSc thesis, 2022. Might be available (in Russian)
            `here <https://www.hse.ru/ba/am/students/diplomas/620910604>`_

        """
        # Compute the numerator from (2.2.4) [3]
        n = X.shape[1]  # 获取矩阵 X 的列数，即数据点的数量
        c = solve_banded((2, 2), X + lam * wE, y)  # 解线性方程组 X + λ * W^-1 * E * c = y
        res = np.zeros(n)  # 创建一个长度为 n 的零向量 res
        # 计算 W^-1 * E * c，考虑到 E 的带状存储结构
        tmp = wE * c
        for i in range(n):
            for j in range(max(0, i - n + 3), min(5, i + 3)):
                res[i] += tmp[j, i + 2 - j]  # 累加 W^-1 * E * c 的相应元素到 res[i]
        numer = np.linalg.norm(lam * res)**2 / n  # 计算分子部分的平方范数除以 n

        # 计算分母部分
        lhs = XtWX + lam * XtE  # 构造左手侧矩阵 XtWX + λ * XtE
        try:
            b_banded = compute_b_inv(lhs)  # 计算 lhs 的逆矩阵，使用带状存储
            # 计算 b_banded @ XtWX 的迹
            tr = b_banded * XtWX
            tr[:-1] *= 2  # 调整迹的计算
            # 找到分母部分
            denom = (1 - sum(sum(tr)) / n)**2
        except LinAlgError:
            # 如果无法进行 cholesky 分解
            raise ValueError('Seems like the problem is ill-posed')

        res = numer / denom  # 计算最终的 GCV 准则值

        return res

    n = X.shape[1]  # 获取矩阵 X 的列数
    # 使用给定的函数计算带状对称矩阵乘积 XtWX
    XtWX = compute_banded_symmetric_XT_W_Y(X, w, X)
    # 使用给定的函数计算带状对称矩阵乘积 XtE
    XtE = compute_banded_symmetric_XT_W_Y(X, w, wE)

    # 定义一个函数 fun，该函数接受参数 lam，返回 _gcv 函数在 lam 上的结果
    def fun(lam):
        return _gcv(lam, X, XtWX, wE, XtE)

    # 使用 minimize_scalar 函数找到函数 fun 在区间 [0, n] 上的最小值
    gcv_est = minimize_scalar(fun, bounds=(0, n), method='Bounded')
    # 如果最小化成功，返回找到的最小值 gcv_est.x
    if gcv_est.success:
        return gcv_est.x
    # 如果最小化失败，引发 ValueError 异常，显示错误消息
    raise ValueError(f"Unable to find minimum of the GCV "
                     f"function: {gcv_est.message}")
def _coeff_of_divided_diff(x):
    """
    Returns the coefficients of the divided difference.

    Parameters
    ----------
    x : array, shape (n,)
        Array which is used for the computation of divided difference.

    Returns
    -------
    res : array_like, shape (n,)
        Coefficients of the divided difference.

    Notes
    -----
    Vector ``x`` should have unique elements, otherwise an error division by
    zero might be raised.

    No checks are performed.

    """
    # 获取数组 x 的长度
    n = x.shape[0]
    # 初始化一个全零数组 res，用于存储计算得到的系数
    res = np.zeros(n)
    # 循环计算每个点的 divided difference 系数
    for i in range(n):
        pp = 1.
        # 计算每个点的 divided difference，除了当前点 x[i] 以外的其他点 x[k]
        for k in range(n):
            if k != i:
                pp *= (x[i] - x[k])
        # 将计算得到的系数存入结果数组 res 中
        res[i] = 1. / pp
    # 返回计算得到的系数数组
    return res


def make_smoothing_spline(x, y, w=None, lam=None):
    r"""
    Compute the (coefficients of) smoothing cubic spline function using
    ``lam`` to control the tradeoff between the amount of smoothness of the
    curve and its proximity to the data. In case ``lam`` is None, using the
    GCV criteria [1] to find it.

    A smoothing spline is found as a solution to the regularized weighted
    linear regression problem:

    .. math::

        \sum\limits_{i=1}^n w_i\lvert y_i - f(x_i) \rvert^2 +
        \lambda\int\limits_{x_1}^{x_n} (f^{(2)}(u))^2 d u

    where :math:`f` is a spline function, :math:`w` is a vector of weights and
    :math:`\lambda` is a regularization parameter.

    If ``lam`` is None, we use the GCV criteria to find an optimal
    regularization parameter, otherwise we solve the regularized weighted
    linear regression problem with given parameter. The parameter controls
    the tradeoff in the following way: the larger the parameter becomes, the
    smoother the function gets.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas. `n` must be at least 5.
    y : array_like, shape (n,)
        Ordinates. `n` must be at least 5.
    w : array_like, shape (n,), optional
        Vector of weights. Default is ``np.ones_like(x)``.
    lam : float, (:math:`\lambda \geq 0`), optional
        Regularization parameter. If ``lam`` is None, then it is found from
        the GCV criteria. Default is None.

    Returns
    -------
    func : a BSpline object.
        A callable representing a spline in the B-spline basis
        as a solution of the problem of smoothing splines using
        the GCV criteria [1] in case ``lam`` is None, otherwise using the
        given parameter ``lam``.

    Notes
    -----
    This algorithm is a clean room reimplementation of the algorithm
    introduced by Woltring in FORTRAN [2]. The original version cannot be used
    in SciPy source code because of the license issues. The details of the
    reimplementation are discussed here (available only in Russian) [4].

    If the vector of weights ``w`` is None, we assume that all the points are
    equal in terms of weights, and vector of weights is vector of ones.

    Note that in weighted residual sum of squares, weights are not squared:
    """
    # 函数主体已在文档字符串中详细说明
    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)


# 将输入的 x 和 y 转换为连续的内存数组，并指定数据类型为 float
x = np.ascontiguousarray(x, dtype=float)
y = np.ascontiguousarray(y, dtype=float)



    if any(x[1:] - x[:-1] <= 0):
        raise ValueError('``x`` should be an ascending array')


# 检查数组 x 是否为升序数组，如果不是则抛出 ValueError 异常
if any(x[1:] - x[:-1] <= 0):
    raise ValueError('``x`` should be an ascending array')



    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError('``x`` and ``y`` should be one dimensional and the'
                         ' same size')


# 检查 x 和 y 是否均为一维数组，并且长度相同，否则抛出 ValueError 异常
if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
    raise ValueError('``x`` and ``y`` should be one dimensional and the'
                     ' same size')



    if w is None:
        w = np.ones(len(x))
    else:
        w = np.ascontiguousarray(w)
        if any(w <= 0):
            raise ValueError('Invalid vector of weights')


# 如果权重 w 为 None，则设定为长度与 x 相同的全为 1 的数组；否则，将 w 转换为连续的内存数组
# 并检查是否存在非正数的权重值，若有则抛出 ValueError 异常
if w is None:
    w = np.ones(len(x))
else:
    w = np.ascontiguousarray(w)
    if any(w <= 0):
        raise ValueError('Invalid vector of weights')



    t = np.r_[[x[0]] * 3, x, [x[-1]] * 3]
    n = x.shape[0]


# 构造扩展后的节点数组 t，包括 x 的前三个元素、全部 x 元素、以及 x 的最后三个元素
# 记录 x 数组的长度
t = np.r_[[x[0]] * 3, x, [x[-1]] * 3]
n = x.shape[0]



    if n <= 4:
        raise ValueError('``x`` and ``y`` length must be at least 5')


# 如果数组 x 的长度小于等于 4，则抛出 ValueError 异常，要求长度至少为 5
if n <= 4:
    raise ValueError('``x`` and ``y`` length must be at least 5')



    # It is known that the solution to the stated minimization problem exists
    # and is a natural cubic spline with vector of knots equal to the unique
    # elements of ``x`` [3], so we will solve the problem in the basis of
    # natural splines.


# 已知所述最小化问题的解存在，并且是一个自然三次样条，其节点向量等于 x 的唯一元素 [3]，
# 因此我们将在自然样条的基础上解决这个问题。
    # 在 B-样条基础上创建设计矩阵
    X_bspl = BSpline.design_matrix(x, t, 3)

    # 从 B-样条基础转换到自然样条基础的方程式 (2.1.7) [4]
    # 中心元素
    X = np.zeros((5, n))
    for i in range(1, 4):
        X[i, 2: -2] = X_bspl[i: i - 4, 3: -3][np.diag_indices(n - 4)]

    # 第一个元素
    X[1, 1] = X_bspl[0, 0]
    X[2, :2] = ((x[2] + x[1] - 2 * x[0]) * X_bspl[0, 0],
                X_bspl[1, 1] + X_bspl[1, 2])
    X[3, :2] = ((x[2] - x[0]) * X_bspl[1, 1], X_bspl[2, 2])

    # 最后的元素
    X[1, -2:] = (X_bspl[-3, -3], (x[-1] - x[-3]) * X_bspl[-2, -2])
    X[2, -2:] = (X_bspl[-2, -3] + X_bspl[-2, -2],
                 (2 * x[-1] - x[-2] - x[-3]) * X_bspl[-1, -1])
    X[3, -2] = X_bspl[-1, -1]

    # 创建惩罚矩阵并将其除以权重向量：W^{-1} E
    wE = np.zeros((5, n))
    wE[2:, 0] = _coeff_of_divided_diff(x[:3]) / w[:3]
    wE[1:, 1] = _coeff_of_divided_diff(x[:4]) / w[:4]
    for j in range(2, n - 2):
        wE[:, j] = (x[j+2] - x[j-2]) * _coeff_of_divided_diff(x[j-2:j+3]) \
                   / w[j-2: j+3]

    wE[:-1, -2] = -_coeff_of_divided_diff(x[-4:]) / w[-4:]
    wE[:-2, -1] = _coeff_of_divided_diff(x[-3:]) / w[-3:]
    wE *= 6

    # 如果未指定 lam，则计算最优的 GCV 参数
    if lam is None:
        lam = _compute_optimal_gcv_parameter(X, wE, y, w)
    # 如果 lam 小于 0，则引发数值错误
    elif lam < 0.:
        raise ValueError('Regularization parameter should be non-negative')

    # 解决基于自然样条基础的初始问题
    c = solve_banded((2, 2), X + lam * wE, y)

    # 使用方程式 (2.2.10) [4] 将结果移回 B-样条基础
    c_ = np.r_[c[0] * (t[5] + t[4] - 2 * t[3]) + c[1],
               c[0] * (t[5] - t[3]) + c[1],
               c[1: -1],
               c[-1] * (t[-4] - t[-6]) + c[-2],
               c[-1] * (2 * t[-4] - t[-5] - t[-6]) + c[-2]]

    # 使用解构造快速 BSpline 并返回
    return BSpline.construct_fast(t, c_, 3)
########################
#  FITPACK look-alikes #
########################

# 检查数据向量 `x` 和结点向量 `t` 的一致性
# 如果输入一致返回 None，否则引发 ValueError 异常
def fpcheck(x, t, k):
    """ Check consistency of the data vector `x` and the knot vector `t`.

    Return None if inputs are consistent, raises a ValueError otherwise.
    """
    # 这个例程是 `fpchec` Fortran 例程的克隆
    # https://github.com/scipy/scipy/blob/main/scipy/interpolate/fitpack/fpchec.f
    # 下面是原始例程的注释：
    #
    # subroutine fpchec verifies the number and the position of the knots
    #  t(j),j=1,2,...,n of a spline of degree k, in relation to the number
    #  and the position of the data points x(i),i=1,2,...,m. if all of the
    #  following conditions are fulfilled, the error parameter ier is set
    #  to zero. if one of the conditions is violated ier is set to ten.
    #      1) k+1 <= n-k-1 <= m
    #      2) t(1) <= t(2) <= ... <= t(k+1)
    #         t(n-k) <= t(n-k+1) <= ... <= t(n)
    #      3) t(k+1) < t(k+2) < ... < t(n-k)
    #      4) t(k+1) <= x(i) <= t(n-k)
    #      5) the conditions specified by schoenberg and whitney must hold
    #         for at least one subset of data points, i.e. there must be a
    #         subset of data points y(j) such that
    #             t(j) < y(j) < t(j+k+1), j=1,2,...,n-k-1
    
    x = np.asarray(x)
    t = np.asarray(t)

    # 检查 `x` 和 `t` 是否是一维序列
    if x.ndim != 1 or t.ndim != 1:
        raise ValueError(f"Expect `x` and `t` be 1D sequences. Got {x = } and {t = }")

    m = x.shape[0]  # 数据向量 `x` 的长度
    n = t.shape[0]  # 结点向量 `t` 的长度
    nk1 = n - k - 1  # k+1

    # 检查条件 1
    # c      1) k+1 <= n-k-1 <= m
    if not (k + 1 <= nk1 <= m):
        raise ValueError(f"Need k+1 <= n-k-1 <= m. Got {m = }, {n = } and {k = }.")

    # 检查条件 2
    # c      2) t(1) <= t(2) <= ... <= t(k+1)
    # c         t(n-k) <= t(n-k+1) <= ... <= t(n)
    if (t[:k+1] > t[1:k+2]).any():
        raise ValueError(f"First k knots must be ordered; got {t = }.")

    if (t[nk1:] < t[nk1-1:-1]).any():
        raise ValueError(f"Last k knots must be ordered; got {t = }.")

    # 检查条件 3
    # c      3) t(k+1) < t(k+2) < ... < t(n-k)
    if (t[k+1:n-k] <= t[k:n-k-1]).any():
        raise ValueError(f"Internal knots must be distinct. Got {t = }.")

    # 检查条件 4
    # c      4) t(k+1) <= x(i) <= t(n-k)
    # 注意：FITPACK 的 fpchec 只检查 x[0] 和 x[-1]，所以我们遵循这个。
    if (x[0] < t[k]) or (x[-1] > t[n-k-1]):
        raise ValueError(f"Out of bounds: {x = } and {t = }.")

    # 检查条件 5
    # c      5) the conditions specified by schoenberg and whitney must hold
    # c         for at least one subset of data points, i.e. there must be a
    # c         subset of data points y(j) such that
    # c             t(j) < y(j) < t(j+k+1), j=1,2,...,n-k-1
    mesg = f"Schoenberg-Whitney condition is violated with {t = } and {x =}."

    if (x[0] >= t[k+1]) or (x[-1] <= t[n-k-2]):
        raise ValueError(mesg)

    m = x.shape[0]  # 更新数据向量 `x` 的长度
    l = k+1  # l = k+1
    nk3 = n - k - 3  # n-k-3
    # 如果 nk3 小于 2，则直接返回，不执行后续代码
    if nk3 < 2:
        return
    # 遍历范围从 1 到 nk3+1 的所有整数 j
    for j in range(1, nk3+1):
        # 获取 t[j] 的值并赋给 tj
        tj = t[j]
        # 将 l 的值加 1，表示 l 的下一个索引位置
        l += 1
        # 获取 t[l] 的值并赋给 tl
        tl = t[l]
        # 在数组 x 中找到第一个大于 tj 的元素的索引 i
        i = np.argmax(x > tj)
        # 如果 i 大于等于 m-1，则抛出 ValueError 异常，使用 mesg 作为错误信息
        if i >= m-1:
            raise ValueError(mesg)
        # 如果 x[i] 大于等于 tl，则抛出 ValueError 异常，使用 mesg 作为错误信息
        if x[i] >= tl:
            raise ValueError(mesg)
    # 循环结束后返回
    return
```