# `D:\src\scipysrc\scipy\scipy\interpolate\_fitpack_py.py`

```
__all__ = ['splrep', 'splprep', 'splev', 'splint', 'sproot', 'spalde',
           'bisplrep', 'bisplev', 'insert', 'splder', 'splantider']

# 导入NumPy库，用于科学计算和数组操作
import numpy as np

# These are in the API for fitpack even if not used in fitpack.py itself.
# 导入_fitpack_impl模块中的bisplrep, bisplev, dblint函数，尽管在fitpack.py文件中未直接使用
from ._fitpack_impl import bisplrep, bisplev, dblint  # noqa: F401

# 导入_fitpack_impl模块作为_impl别名，用于后续可能的调用
from . import _fitpack_impl as _impl

# 导入_BSplines模块中的BSpline类，用于B样条曲线的操作
from ._bsplines import BSpline


def splprep(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None, t=None,
            full_output=0, nest=None, per=0, quiet=1):
    """
    Find the B-spline representation of an N-D curve.

    Given a list of N rank-1 arrays, `x`, which represent a curve in
    N-dimensional space parametrized by `u`, find a smooth approximating
    spline curve g(`u`). Uses the FORTRAN routine parcur from FITPACK.

    Parameters
    ----------
    x : array_like
        A list of sample vector arrays representing the curve.
    w : array_like, optional
        Strictly positive rank-1 array of weights the same length as `x[0]`.
        The weights are used in computing the weighted least-squares spline
        fit. If the errors in the `x` values have standard-deviation given by
        the vector d, then `w` should be 1/d. Default is ``ones(len(x[0]))``.
    u : array_like, optional
        An array of parameter values. If not given, these values are
        calculated automatically as ``M = len(x[0])``, where

            v[0] = 0

            v[i] = v[i-1] + distance(`x[i]`, `x[i-1]`)

            u[i] = v[i] / v[M-1]

    ub, ue : int, optional
        The end-points of the parameters interval.  Defaults to
        u[0] and u[-1].
    k : int, optional
        Degree of the spline. Cubic splines are recommended.
        Even values of `k` should be avoided especially with a small s-value.
        ``1 <= k <= 5``, default is 3.
    task : int, optional
        If task==0 (default), find t and c for a given smoothing factor, s.
        If task==1, find t and c for another value of the smoothing factor, s.
        There must have been a previous call with task=0 or task=1
        for the same set of data.
        If task=-1 find the weighted least square spline for a given set of
        knots, t.
    s : float, optional
        A smoothing condition.  The amount of smoothness is determined by
        satisfying the conditions: ``sum((w * (y - g))**2,axis=0) <= s``,
        where g(x) is the smoothed interpolation of (x,y).  The user can
        use `s` to control the trade-off between closeness and smoothness
        of fit.  Larger `s` means more smoothing while smaller values of `s`
        indicate less smoothing. Recommended values of `s` depend on the
        weights, w.  If the weights represent the inverse of the
        standard-deviation of y, then a good `s` value should be found in
        the range ``(m-sqrt(2*m),m+sqrt(2*m))``, where m is the number of
        data points in x, y, and w.
    """

    # 该函数的详细参数解释见上述文档字符串

    # 函数内部实现依赖于FORTRAN库FITPACK中的parcur例程
    # 用于在N维空间中找到N维曲线的B样条表示
    # 返回平滑的逼近样条曲线g(u)

    pass
    t : array, optional
        # 参数t：数组，可选。用于`task=-1`时的节点。
        # 至少需要有`2*k+2`个节点。
    full_output : int, optional
        # 参数full_output：整数，可选。如果非零，则返回可选的输出。
    nest : int, optional
        # 参数nest：整数，可选。对样条的总节点数进行过估计，
        # 以帮助确定存储空间。默认为`m/2`。
        # 总是足够大，即`nest=m+k+1`。
    per : int, optional
        # 参数per：整数，可选。如果非零，则数据点被视为具有周期性，
        # 周期为`x[m-1] - x[0]`，并返回平滑的周期样条逼近。
        # `y[m-1]`和`w[m-1]`的值不会被使用。
    quiet : int, optional
        # 参数quiet：整数，可选。非零表示抑制消息输出。

    Returns
    -------
    tck : tuple
        # 返回一个元组`(t, c, k)`，包含节点向量、B样条系数和样条的阶数。
    u : array
        # 返回参数的值的数组。
    fp : float
        # 样条逼近的加权残差平方和。
    ier : int
        # 有关splrep成功的整数标志。如果`ier<=0`，表示成功。
        # 如果`ier`在[1,2,3]中，表示发生错误但未引发错误。
        # 否则会引发错误。
    msg : str
        # 与整数标志`ier`对应的消息。
        
    See Also
    --------
    splrep, splev, sproot, spalde, splint,
    bisplrep, bisplev
    UnivariateSpline, BivariateSpline
    BSpline
    make_interp_spline

    Notes
    -----
    # 查看`splev`以评估样条及其导数。
    # 维数N必须小于11。

    # `c`数组中的系数数量为`k+1`，比节点数组`t`的长度`len(t)`少`k+1`。
    # 这与`splrep`相反，后者将系数数组填充为与节点数组相同的长度。
    # 评估例程`splev`和`BSpline`将忽略这些额外的系数。

    References
    ----------
    .. [1] P. Dierckx, "Algorithms for smoothing data with periodic and
        parametric splines, Computer Graphics and Image Processing",
        20 (1982) 171-184.
    .. [2] P. Dierckx, "Algorithms for smoothing data with periodic and
        parametric splines", report tw55, Dept. Computer Science,
        K.U.Leuven, 1981.
    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs on
        Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    # 在极坐标中生成limacon曲线的离散化：

    >>> import numpy as np
    >>> phi = np.linspace(0, 2.*np.pi, 40)
    >>> r = 0.5 + np.cos(phi)         # 极坐标
    >>> x, y = r * np.cos(phi), r * np.sin(phi)    # 转换为笛卡尔坐标系

    # 进行插值：

    >>> from scipy.interpolate import splprep, splev
    >>> tck, u = splprep([x, y], s=0)
    >>> new_points = splev(u, tck)

    # 注意，我们通过使用`s=0`来强制插值。
    """
    调用_splprep函数进行样条插值处理。

    res = _impl.splprep(x, w, u, ub, ue, k, task, s, t, full_output, nest, per,
                        quiet)
    # 调用_splprep函数，进行样条插值计算
    # x: 输入的数据点的坐标
    # w: 数据点的权重
    # u: 样条插值的参数化参数（自动生成）
    # ub, ue: 参数u的边界
    # k: 样条插值的阶数
    # task: 任务类型，控制插值过程的不同阶段
    # s, t: 平滑因子和收缩因子
    # full_output: 是否返回完整输出信息
    # nest: 估计的B样条结点的数量
    # per: 是否对周期数据进行处理
    # quiet: 是否安静模式，即禁止输出警告消息

    return res
    # 返回样条插值计算的结果
    """
# 定义函数splrep，用于找到一维曲线的 B-spline 插值表示

def splrep(x, y, w=None, xb=None, xe=None, k=3, task=0, s=None, t=None,
           full_output=0, per=0, quiet=1):
    """
    Find the B-spline representation of a 1-D curve.

    Given the set of data points ``(x[i], y[i])`` determine a smooth spline
    approximation of degree k on the interval ``xb <= x <= xe``.

    Parameters
    ----------
    x, y : array_like
        The data points defining a curve ``y = f(x)``.
    w : array_like, optional
        Strictly positive rank-1 array of weights the same length as `x` and `y`.
        The weights are used in computing the weighted least-squares spline
        fit. If the errors in the `y` values have standard-deviation given by the
        vector ``d``, then `w` should be ``1/d``. Default is ``ones(len(x))``.
    xb, xe : float, optional
        The interval to fit.  If None, these default to ``x[0]`` and ``x[-1]``
        respectively.
    k : int, optional
        The degree of the spline fit. It is recommended to use cubic splines.
        Even values of `k` should be avoided especially with small `s` values.
        ``1 <= k <= 5``.
    task : {1, 0, -1}, optional
        If ``task==0``, find ``t`` and ``c`` for a given smoothing factor, `s`.

        If ``task==1`` find ``t`` and ``c`` for another value of the smoothing factor,
        `s`. There must have been a previous call with ``task=0`` or ``task=1`` for
        the same set of data (``t`` will be stored an used internally)

        If ``task=-1`` find the weighted least square spline for a given set of
        knots, ``t``. These should be interior knots as knots on the ends will be
        added automatically.
    s : float, optional
        A smoothing condition. The amount of smoothness is determined by
        satisfying the conditions: ``sum((w * (y - g))**2,axis=0) <= s`` where ``g(x)``
        is the smoothed interpolation of ``(x,y)``. The user can use `s` to control
        the tradeoff between closeness and smoothness of fit. Larger `s` means
        more smoothing while smaller values of `s` indicate less smoothing.
        Recommended values of `s` depend on the weights, `w`. If the weights
        represent the inverse of the standard-deviation of `y`, then a good `s`
        value should be found in the range ``(m-sqrt(2*m),m+sqrt(2*m))`` where ``m`` is
        the number of datapoints in `x`, `y`, and `w`. default : ``s=m-sqrt(2*m)`` if
        weights are supplied. ``s = 0.0`` (interpolating) if no weights are
        supplied.
    t : array_like, optional
        The knots needed for ``task=-1``. If given then task is automatically set
        to ``-1``.
    full_output : bool, optional
        If non-zero, then return optional outputs.
    """
    per : bool, optional
        如果非零，则数据点被视为周期性，周期为 ``x[m-1]`` - ``x[0]``，并返回平滑的周期样条近似。不使用 ``y[m-1]`` 和 ``w[m-1]`` 的值。
        默认值为零，对应边界条件为 'not-a-knot'。
    quiet : bool, optional
        非零以抑制消息输出。

    Returns
    -------
    tck : tuple
        包含结点向量、B-样条系数和样条的阶数的元组 ``(t,c,k)``。
    fp : array, optional
        样条逼近的加权残差平方和。
    ier : int, optional
        有关 splrep 成功的整数标志。如果 ``ier<=0`` 表示成功。如果 ``ier in [1,2,3]``，表示发生了错误但未引发异常。否则会引发错误。
    msg : str, optional
        与整数标志 `ier` 相对应的消息。

    See Also
    --------
    UnivariateSpline, BivariateSpline
    splprep, splev, sproot, spalde, splint
    bisplrep, bisplev
    BSpline
    make_interp_spline

    Notes
    -----
    查看 `splev` 以评估样条及其导数。使用来自 FITPACK 的 FORTRAN 程序 ``curfit``。

    用户有责任确保 `x` 的值是唯一的。否则，`splrep` 将无法返回合理的结果。

    如果提供了结点 `t`，则必须满足 Schoenberg-Whitney 条件，即必须存在数据点的子集 ``x[j]`` 满足 ``t[j] < x[j] < t[j+k+1]``，其中 ``j=0, 1,...,n-k-2``。

    此例程将系数数组 ``c`` 零填充至与结点数组 ``t`` 相同的长度（评估例程 `splev` 和 `BSpline` 忽略末尾的 ``k + 1`` 个系数）。这与 `splprep` 相反，后者不会对系数进行零填充。

    默认边界条件为 'not-a-knot'，即曲线端点处的第一段和第二段是相同的多项式。在 `CubicSpline` 中提供更多边界条件选项。

    References
    ----------
    基于以下文献中描述的算法 [1]_, [2]_, [3]_ 和 [4]_：

    .. [1] P. Dierckx, "An algorithm for smoothing, differentiation and
       integration of experimental data using spline functions",
       J.Comp.Appl.Maths 1 (1975) 165-184.
    .. [2] P. Dierckx, "A fast algorithm for smoothing data on a rectangular
       grid while using spline functions", SIAM J.Numer.Anal. 19 (1982)
       1286-1304.
    .. [3] P. Dierckx, "An improved algorithm for curve fitting with spline
       functions", report tw54, Dept. Computer Science,K.U. Leuven, 1981.
    .. [4] P. Dierckx, "Curve and surface fitting with splines", Monographs on
       Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    可以用 B-样条曲线插值 1-D 点。
    更多示例请参阅
    :ref:`本教程中的示例 <tutorial-interpolate_splXXX>`。
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import splev, splrep
    >>> x = np.linspace(0, 10, 10)
    >>> y = np.sin(x)
    >>> spl = splrep(x, y)
    >>> x2 = np.linspace(0, 10, 200)
    >>> y2 = splev(x2, spl)
    >>> plt.plot(x, y, 'o', x2, y2)
    >>> plt.show()
    
    """
    使用 SciPy 的 splrep 函数进行样条插值拟合。
    
    参数解释:
    - x: 输入数据的 x 坐标数组
    - y: 输入数据的 y 坐标数组
    - w: 可选参数，数据点的权重
    - xb, xe: 可选参数，指定边界条件
    - k: 可选参数，插值样条的次数
    - task: 可选参数，定义执行的任务类型
    - s: 可选参数，平滑因子的大小
    - t: 可选参数，指定节点
    - full_output: 可选参数，控制是否输出详细信息
    - per: 可选参数，是否周期性处理
    - quiet: 可选参数，控制输出警告
    
    返回值:
    - res: 插值样条的表示形式，可以用于计算新的插值点
    """
    res = _impl.splrep(x, y, w, xb, xe, k, task, s, t, full_output, per, quiet)
    return res
# 根据给定的 B-spline 和其导数次数，在指定点 x 处计算 B-spline 或其导数的值。

def splev(x, tck, der=0, ext=0):
    """
    Evaluate a B-spline or its derivatives.

    Given the knots and coefficients of a B-spline representation, evaluate
    the value of the smoothing polynomial and its derivatives. This is a
    wrapper around the FORTRAN routines splev and splder of FITPACK.

    Parameters
    ----------
    x : array_like
        An array of points at which to return the value of the smoothed
        spline or its derivatives. If `tck` was returned from `splprep`,
        then the parameter values, u should be given.
    tck : BSpline instance or tuple
        If a tuple, then it should be a sequence of length 3 returned by
        `splrep` or `splprep` containing the knots, coefficients, and degree
        of the spline. (Also see Notes.)
    der : int, optional
        The order of derivative of the spline to compute (must be less than
        or equal to k, the degree of the spline).
    ext : int, optional
        Controls the value returned for elements of ``x`` not in the
        interval defined by the knot sequence.

        * if ext=0, return the extrapolated value.
        * if ext=1, return 0
        * if ext=2, raise a ValueError
        * if ext=3, return the boundary value.

        The default value is 0.

    Returns
    -------
    y : ndarray or list of ndarrays
        An array of values representing the spline function evaluated at
        the points in `x`.  If `tck` was returned from `splprep`, then this
        is a list of arrays representing the curve in an N-D space.

    See Also
    --------
    splprep, splrep, sproot, spalde, splint
    bisplrep, bisplev
    BSpline

    Notes
    -----
    Manipulating the tck-tuples directly is not recommended. In new code,
    prefer using `BSpline` objects.

    References
    ----------
    .. [1] C. de Boor, "On calculating with b-splines", J. Approximation
        Theory, 6, p.50-62, 1972.
    .. [2] M. G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths
        Applics, 10, p.134-149, 1972.
    .. [3] P. Dierckx, "Curve and surface fitting with splines", Monographs
        on Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    Examples are given :ref:`in the tutorial <tutorial-interpolate_splXXX>`.

    A comparison between `splev`, `splder` and `spalde` to compute the derivatives of a 
    B-spline can be found in the `spalde` examples section.

    """
    # 检查 tck 是否为 BSpline 对象
    if isinstance(tck, BSpline):
        # 如果 tck 是 BSpline 对象并且其系数的维度大于 1，则抛出异常
        if tck.c.ndim > 1:
            mesg = ("Calling splev() with BSpline objects with c.ndim > 1 is "
                    "not allowed. Use BSpline.__call__(x) instead.")
            raise ValueError(mesg)

        # 根据 ext 参数选择如何处理超出边界的情况
        try:
            extrapolate = {0: True, }[ext]
        except KeyError as e:
            raise ValueError("Extrapolation mode %s is not supported "
                             "by BSpline." % ext) from e

        # 调用 BSpline 对象的 __call__ 方法计算 B-spline 或其导数的值
        return tck(x, der, extrapolate=extrapolate)
    else:
        # 如果前面的条件不满足，则执行这里的代码
        # 调用 _impl.splev 函数，对输入参数进行样条插值计算
        # x: 插值点的数组或值
        # tck: 样条插值的节点和系数
        # der: 求导数的阶数
        # ext: 超出节点范围时的处理方式
        return _impl.splev(x, tck, der, ext)
def splint(a, b, tck, full_output=0):
    """
    Evaluate the definite integral of a B-spline between two given points.

    Parameters
    ----------
    a, b : float
        The end-points of the integration interval.
    tck : tuple or a BSpline instance
        If a tuple, then it should be a sequence of length 3, containing the
        vector of knots, the B-spline coefficients, and the degree of the
        spline (see `splev`).
    full_output : int, optional
        Non-zero to return optional output.

    Returns
    -------
    integral : float
        The resulting integral.
    wrk : ndarray
        An array containing the integrals of the normalized B-splines
        defined on the set of knots.
        (Only returned if `full_output` is non-zero)

    See Also
    --------
    splprep, splrep, sproot, spalde, splev
    bisplrep, bisplev
    BSpline

    Notes
    -----
    `splint` silently assumes that the spline function is zero outside the data
    interval (`a`, `b`).

    Manipulating the tck-tuples directly is not recommended. In new code,
    prefer using the `BSpline` objects.

    References
    ----------
    .. [1] P.W. Gaffney, The calculation of indefinite integrals of b-splines",
        J. Inst. Maths Applics, 17, p.37-41, 1976.
    .. [2] P. Dierckx, "Curve and surface fitting with splines", Monographs
        on Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    Examples are given :ref:`in the tutorial <tutorial-interpolate_splXXX>`.

    """
    # 如果 tck 是 BSpline 实例
    if isinstance(tck, BSpline):
        # 检查 BSpline 对象的系数维度是否大于1，如果是，则抛出异常
        if tck.c.ndim > 1:
            mesg = ("Calling splint() with BSpline objects with c.ndim > 1 is "
                    "not allowed. Use BSpline.integrate() instead.")
            raise ValueError(mesg)
        
        # 如果 full_output 非零，给出警告信息，然后以 extrapolate=False 调用 BSpline 的 integrate 方法
        if full_output != 0:
            mesg = ("full_output = %s is not supported. Proceeding as if "
                    "full_output = 0" % full_output)
        
        # 返回 BSpline 对象的 integrate 方法计算的积分结果
        return tck.integrate(a, b, extrapolate=False)
    else:
        # 否则，调用 _impl.splint 函数处理积分计算
        return _impl.splint(a, b, tck, full_output)


def sproot(tck, mest=10):
    """
    Find the roots of a cubic B-spline.

    Given the knots (>=8) and coefficients of a cubic B-spline return the
    roots of the spline.

    Parameters
    ----------
    tck : tuple or a BSpline object
        If a tuple, then it should be a sequence of length 3, containing the
        vector of knots, the B-spline coefficients, and the degree of the
        spline.
        The number of knots must be >= 8, and the degree must be 3.
        The knots must be a montonically increasing sequence.
    mest : int, optional
        An estimate of the number of zeros (Default is 10).

    Returns
    -------
    zeros : ndarray
        An array giving the roots of the spline.

    See Also
    --------
    splprep, splrep, splint, spalde, splev
    bisplrep, bisplev
    BSpline

    Notes
    -----
    Manipulating the tck-tuples directly is not recommended. In new code,
    ```
    """
    If the input `tck` is an instance of `BSpline`, check its dimensional consistency and transpose if necessary before calling `_impl.sproot`.

    Parameters
    ----------
    tck : tuple or BSpline object
        A tuple representing the spline (t, c, k) or a BSpline object.
    mest : bool
        A boolean flag indicating whether the mesg should be displayed or not.

    Returns
    -------
    ndarray
        An array containing the roots of the spline.

    Raises
    ------
    ValueError
        If `tck` is a BSpline object and its coefficient array `c` has more than one dimension.

    Notes
    -----
    This function handles both tuple representation of splines and BSpline objects. It checks the dimensionality of `c` if `tck` is a BSpline object and raises an error if `c` has more than one dimension.

    References
    ----------
    [1] C. de Boor, "On calculating with b-splines", J. Approximation
        Theory, 6, p.50-62, 1972.
    [2] M. G. Cox, "The numerical evaluation of b-splines", J. Inst. Maths
        Applics, 10, p.134-149, 1972.
    [3] P. Dierckx, "Curve and surface fitting with splines", Monographs
        on Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    For usage examples, see the docstring at the beginning of the file.
    """
    # Check if tck is an instance of BSpline
    if isinstance(tck, BSpline):
        # Check if the coefficient array c has more than one dimension
        if tck.c.ndim > 1:
            # If so, raise an error
            mesg = ("Calling sproot() with BSpline objects with c.ndim > 1 is "
                    "not allowed.")
            raise ValueError(mesg)

        # Unpack tck into t, c, k
        t, c, k = tck.tck

        # Transpose c if necessary, ensuring interpolation axis is last
        sh = tuple(range(c.ndim))
        c = c.transpose(sh[1:] + (0,))

        # Call _impl.sproot with (t, c, k) tuple and mest flag
        return _impl.sproot((t, c, k), mest)
    else:
        # Call _impl.sproot with tck and mest flag
        return _impl.sproot(tck, mest)
# 定义一个函数 spalde，用于计算 B-spline 及其在一个或多个点上的所有导数，直到阶数 k（样条的度数）为止，0阶是指样条本身。

def spalde(x, tck):
    """
    Evaluate a B-spline and all its derivatives at one point (or set of points) up
    to order k (the degree of the spline), being 0 the spline itself.

    Parameters
    ----------
    x : array_like
        A point or a set of points at which to evaluate the derivatives.
        Note that ``t(k) <= x <= t(n-k+1)`` must hold for each `x`.
    tck : tuple
        A tuple (t,c,k) containing the vector of knots,
        the B-spline coefficients, and the degree of the spline whose 
        derivatives to compute.
    """
    
    # 从输入的参数元组 tck 中解包出节点向量 t、B-spline 系数 c 和样条的度数 k
    t, c, k = tck
    
    # 计算样条的阶数
    order = k
    
    # 如果输入的 x 不是数组，转换成数组以便后续处理
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    
    # 获取 x 的形状
    xshape = x.shape
    
    # 将 x 展平为一维数组，方便在循环中处理每个点
    x = x.ravel()
    
    # 初始化一个数组，用于存储每个点 x 处的所有导数值
    results = [np.empty((len(x),)) for m in range(order+1)]
    
    # 迭代每个点 x，计算其处的样条及其所有导数
    for ii in range(len(x)):
        # 调用底层的 Fortran 函数计算样条及其导数，结果存储在 results 中
        results[:,ii] = _fitpack._bspleval(x[ii], order, t, c, k)
    
    # 根据输入 x 的形状，将结果重新调整为原始形状
    results = [res.reshape(xshape) for res in results]
    
    # 如果只有一个点，返回一个数组而不是列表
    if len(results) == 1:
        results = results[0]
    
    # 返回结果，包含样条及其所有导数的数组或数组列表
    return results
    # 如果 tck 是 BSpline 类型的实例，则抛出类型错误异常
    if isinstance(tck, BSpline):
        # 抛出异常，提示 spalde 不接受 BSpline 实例作为参数
        raise TypeError("spalde does not accept BSpline instances.")
    else:
        # 否则，调用内部函数 _impl.spalde()，并返回结果
        return _impl.spalde(x, tck)
# 定义插入节点的函数，用于 B-spline 曲线的节点插入操作
def insert(x, tck, m=1, per=0):
    """
    Insert knots into a B-spline.

    Given the knots and coefficients of a B-spline representation, create a
    new B-spline with a knot inserted `m` times at point `x`.
    This is a wrapper around the FORTRAN routine insert of FITPACK.

    Parameters
    ----------
    x (u) : float
        A knot value at which to insert a new knot.  If `tck` was returned
        from ``splprep``, then the parameter values, u should be given.
    tck : a `BSpline` instance or a tuple
        If tuple, then it is expected to be a tuple (t,c,k) containing
        the vector of knots, the B-spline coefficients, and the degree of
        the spline.
    m : int, optional
        The number of times to insert the given knot (its multiplicity).
        Default is 1.
    per : int, optional
        If non-zero, the input spline is considered periodic.

    Returns
    -------
    BSpline instance or a tuple
        A new B-spline with knots t, coefficients c, and degree k.
        ``t(k+1) <= x <= t(n-k)``, where k is the degree of the spline.
        In case of a periodic spline (``per != 0``) there must be
        either at least k interior knots t(j) satisfying ``t(k+1)<t(j)<=x``
        or at least k interior knots t(j) satisfying ``x<=t(j)<t(n-k)``.
        A tuple is returned iff the input argument `tck` is a tuple, otherwise
        a BSpline object is constructed and returned.

    Notes
    -----
    Based on algorithms from [1]_ and [2]_.

    Manipulating the tck-tuples directly is not recommended. In new code,
    prefer using the `BSpline` objects, in particular `BSpline.insert_knot`
    method.

    See Also
    --------
    BSpline.insert_knot

    References
    ----------
    .. [1] W. Boehm, "Inserting new knots into b-spline curves.",
        Computer Aided Design, 12, p.199-201, 1980.
    .. [2] P. Dierckx, "Curve and surface fitting with splines, Monographs on
        Numerical Analysis", Oxford University Press, 1993.

    Examples
    --------
    You can insert knots into a B-spline.

    >>> from scipy.interpolate import splrep, insert
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 5)
    >>> y = np.sin(x)
    >>> tck = splrep(x, y)
    >>> tck[0]
    array([ 0.,  0.,  0.,  0.,  5., 10., 10., 10., 10.])

    A knot is inserted:

    >>> tck_inserted = insert(3, tck)
    >>> tck_inserted[0]
    array([ 0.,  0.,  0.,  0.,  3.,  5., 10., 10., 10., 10.])

    Some knots are inserted:

    >>> tck_inserted2 = insert(8, tck, m=3)
    >>> tck_inserted2[0]
    array([ 0.,  0.,  0.,  0.,  5.,  8.,  8.,  8., 10., 10., 10., 10.])

    """
    if isinstance(tck, BSpline):
        # 如果 tck 是 BSpline 类型的对象

        t, c, k = tck.tck
        # 从 tck 对象中解包出节点 t, 控制点 c 和阶数 k

        # FITPACK 需要插值轴在最后，因此将其滚动到最后
        # 注意：如果 c 数组是一维的，transpose 操作不会产生变化
        sh = tuple(range(c.ndim))
        # 创建一个元组，其中包含 c 的所有维度的索引
        c = c.transpose(sh[1:] + (0,))
        # 将 c 数组的最后一个轴移到第一个位置

        t_, c_, k_ = _impl.insert(x, (t, c, k), m, per)
        # 调用 _impl.insert 函数，将 x, (t, c, k), m, per 作为参数传递，并获取返回的 t_, c_, k_

        # 将最后一个轴滚动回原来的位置
        c_ = np.asarray(c_)
        c_ = c_.transpose((sh[-1],) + sh[:-1])
        # 将 c_ 数组的第一个轴移动回最后一个位置

        return BSpline(t_, c_, k_)
        # 返回一个新的 BSpline 对象，使用 t_, c_, k_

    else:
        # 如果 tck 不是 BSpline 对象

        return _impl.insert(x, tck, m, per)
        # 直接调用 _impl.insert 函数，将 x, tck, m, per 作为参数传递，并返回其结果
# 计算给定样条曲线的导数的样条表示

def splder(tck, n=1):
    """
    计算给定样条曲线的导数的样条表示

    Parameters
    ----------
    tck : BSpline instance or tuple
        BSpline 实例或元组 (t,c,k)，包含结点向量、B-样条系数和要计算导数的样条的阶数
    n : int, optional
        要计算的导数阶数。默认值：1

    Returns
    -------
    `BSpline` 实例或元组
        表示输入样条曲线的导数的阶数为 k2=k-n 的样条。
        如果输入参数 `tck` 是元组，则返回一个元组；否则构造并返回一个 BSpline 对象。

    See Also
    --------
    splantider, splev, spalde
    BSpline

    Notes
    -----
    `splder` 的版本添加：0.13.0

    Examples
    --------
    用于找到曲线的极值点的示例：

    >>> from scipy.interpolate import splrep, splder, sproot
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 70)
    >>> y = np.sin(x)
    >>> spl = splrep(x, y, k=4)

    现在，对样条进行求导，并找到导数的零点。（注意：`sproot` 仅适用于阶数为 3 的样条，所以我们
    拟合一个阶数为 4 的样条）：

    >>> dspl = splder(spl)
    >>> sproot(dspl) / np.pi
    array([ 0.50000001,  1.5       ,  2.49999998])

    这与根据 :math:`\\cos(x) = \\sin'(x)` 推导出的 :math:`\\pi/2 + n\\pi` 的根非常吻合。

    可在 `spalde` 示例部分找到用于计算 B-样条导数的 `splev`、`splder` 和 `spalde` 的比较。

    """
    if isinstance(tck, BSpline):
        return tck.derivative(n)
    else:
        return _impl.splder(tck, n)
    # 使用样条插值的求值函数 splev 对给定的 spl 对象进行求值，同时也对 splantider(spl) 的结果进行求值
    >>> splev(1.7, spl), splev(1.7, splder(splantider(spl)))
    # 返回值分别为 1.7 处的样条插值结果和 splantider(spl) 的导数样条插值结果
    (array(2.1565429877197317), array(2.1565429877201865))

    # 使用 splantider 函数获取反导函数，可以用于计算定积分：
    >>> ispl = splantider(spl)
    # 计算从 0 到 π/2 的定积分值，这是通过 ispl 函数得到的
    >>> splev(np.pi/2, ispl) - splev(0, ispl)
    # 结果大约为 2.2572053588768486

    # 这实际上是完全椭圆积分的近似：
    # :math:`K(m) = \int_0^{\\pi/2} [1 - m\\sin^2 x]^{-1/2} dx`
    >>> from scipy.special import ellipk
    # 计算参数为 0.8 的完全椭圆积分 K(m)
    >>> ellipk(0.8)
    # 结果大约为 2.2572053268208538

    """
    # 如果 tck 是 BSpline 对象，则调用其反导函数 antiderivative
    if isinstance(tck, BSpline):
        return tck.antiderivative(n)
    # 否则调用 _impl 模块中的 splantider 函数，计算其 n 次导函数
    else:
        return _impl.splantider(tck, n)
```