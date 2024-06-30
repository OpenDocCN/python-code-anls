# `D:\src\scipysrc\scipy\scipy\interpolate\_rbfinterp.py`

```
# 导入警告模块，用于可能的警告处理
import warnings
# 导入用于组合生成的函数
from itertools import combinations_with_replacement

# 导入 NumPy 库
import numpy as np
# 导入线性代数错误处理模块
from numpy.linalg import LinAlgError
# 导入 KD 树数据结构
from scipy.spatial import KDTree
# 导入组合数计算函数
from scipy.special import comb
# 导入 LAPACK 函数库中的 dgesv 函数
from scipy.linalg.lapack import dgesv  # type: ignore[attr-defined]

# 导入 Pythran 优化后的 RBF 插值模块中的相关函数
from ._rbfinterp_pythran import (_build_system,
                                 _build_evaluation_coefficients,
                                 _polynomial_matrix)


# 指定模块中公开的类和函数
__all__ = ["RBFInterpolator"]

# 已实现的 RBF 插值方法集合
_AVAILABLE = {
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian"
    }

# 对于这些 RBF，不需要指定形状参数
_SCALE_INVARIANT = {"linear", "thin_plate_spline", "cubic", "quintic"}

# 对于条件正定的 RBF，插值多项式的最小阶数定义
# 这些值来源于 Fasshauer 的《用 MATLAB 实现无网格逼近方法》第 8 章
_NAME_TO_MIN_DEGREE = {
    "multiquadric": 0,
    "linear": 0,
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2
    }

def _monomial_powers(ndim, degree):
    """生成多项式中每个单项的幂次。

    Parameters
    ----------
    ndim : int
        多项式中的变量数。
    degree : int
        多项式的次数。

    Returns
    -------
    (nmonos, ndim) int ndarray
        每个单项中每个变量的幂次组成的数组。

    """
    # 计算多项式的总数
    nmonos = comb(degree + ndim, ndim, exact=True)
    # 初始化存储幂次的数组
    out = np.zeros((nmonos, ndim), dtype=np.dtype("long"))
    count = 0
    # 循环生成每个次数下的单项幂次
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            # `mono` 是当前单项中的变量索引的元组，其重复次数表示幂次
            for var in mono:
                out[count, var] += 1

            count += 1

    return out

def _build_and_solve_system(y, d, smoothing, kernel, epsilon, powers):
    """构建并求解 RBF 插值系统方程组。

    Parameters
    ----------
    y : (P, N) float ndarray
        数据点的坐标。
    d : (P, S) float ndarray
        在 `y` 处的数据值。
    smoothing : (P,) float ndarray
        每个数据点的平滑参数。
    kernel : str
        RBF 的名称。
    epsilon : float
        形状参数。
    powers : (R, N) int ndarray
        多项式中每个单项的幂次。

    Returns
    -------
    coeffs : (P + R, S) float ndarray
        每个 RBF 和多项式的系数。
    shift : (N,) float ndarray
        用于创建多项式矩阵的域偏移量。

    """
    scale : (N,) float ndarray
        Domain scaling used to create the polynomial matrix.

    """
    # 调用 _build_system 函数，生成线性方程组的左手边矩阵 lhs、右手边向量 rhs、平移量 shift 和缩放量 scale
    lhs, rhs, shift, scale = _build_system(
        y, d, smoothing, kernel, epsilon, powers
        )
    # 使用 LAPACK 中的 dgesv 函数求解线性方程组 lhs * coeffs = rhs，coeffs 为未知数向量，同时覆盖原有的 lhs 和 rhs
    _, _, coeffs, info = dgesv(lhs, rhs, overwrite_a=True, overwrite_b=True)
    # 如果 info < 0，表示 dgesv 函数的参数之一非法，抛出 ValueError 异常
    if info < 0:
        raise ValueError(f"The {-info}-th argument had an illegal value.")
    # 如果 info > 0，表示发现奇异矩阵，抛出 LinAlgError 异常
    elif info > 0:
        msg = "Singular matrix."
        # 计算多项式矩阵的行数
        nmonos = powers.shape[0]
        # 如果存在多项式
        if nmonos > 0:
            # 构建多项式矩阵 pmat，将数据点的坐标标准化后除以 scale
            pmat = _polynomial_matrix((y - shift)/scale, powers)
            # 计算多项式矩阵的秩
            rank = np.linalg.matrix_rank(pmat)
            # 如果多项式矩阵的秩小于多项式的数量，更新错误消息
            if rank < nmonos:
                msg = (
                    "Singular matrix. The matrix of monomials evaluated at "
                    "the data point coordinates does not have full column "
                    f"rank ({rank}/{nmonos})."
                    )
        # 抛出 LinAlgError 异常，显示具体的错误消息
        raise LinAlgError(msg)

    # 返回平移量 shift、缩放量 scale 和系数 coeffs
    return shift, scale, coeffs
class RBFInterpolator:
    """Radial basis function (RBF) interpolation in N dimensions.
    
    Parameters
    ----------
    y : (npoints, ndims) array_like
        2-D array of data point coordinates.
    d : (npoints, ...) array_like
        N-D array of data values at `y`. The length of `d` along the first
        axis must be equal to the length of `y`. Unlike some interpolators, the
        interpolation axis cannot be changed.
    neighbors : int, optional
        If specified, the value of the interpolant at each evaluation point
        will be computed using only this many nearest data points. All the data
        points are used by default.
    smoothing : float or (npoints, ) array_like, optional
        Smoothing parameter. The interpolant perfectly fits the data when this
        is set to 0. For large values, the interpolant approaches a least
        squares fit of a polynomial with the specified degree. Default is 0.
    kernel : str, optional
        Type of RBF. This should be one of

            - 'linear'               : ``-r``
            - 'thin_plate_spline'    : ``r**2 * log(r)``
            - 'cubic'                : ``r**3``
            - 'quintic'              : ``-r**5``
            - 'multiquadric'         : ``-sqrt(1 + r**2)``
            - 'inverse_multiquadric' : ``1/sqrt(1 + r**2)``
            - 'inverse_quadratic'    : ``1/(1 + r**2)``
            - 'gaussian'             : ``exp(-r**2)``

        Default is 'thin_plate_spline'.
    epsilon : float, optional
        Shape parameter that scales the input to the RBF. If `kernel` is
        'linear', 'thin_plate_spline', 'cubic', or 'quintic', this defaults to
        1 and can be ignored because it has the same effect as scaling the
        smoothing parameter. Otherwise, this must be specified.
    degree : int, optional
        Degree of the added polynomial. For some RBFs the interpolant may not
        be well-posed if the polynomial degree is too small. Those RBFs and
        their corresponding minimum degrees are

            - 'multiquadric'      : 0
            - 'linear'            : 0
            - 'thin_plate_spline' : 1
            - 'cubic'             : 1
            - 'quintic'           : 2

        The default value is the minimum degree for `kernel` or 0 if there is
        no minimum degree. Set this to -1 for no added polynomial.

    Notes
    -----
    An RBF is a scalar valued function in N-dimensional space whose value at
    :math:`x` can be expressed in terms of :math:`r=||x - c||`, where :math:`c`
    is the center of the RBF.

    An RBF interpolant for the vector of data values :math:`d`, which are from
    locations :math:`y`, is a linear combination of RBFs centered at :math:`y`
    plus a polynomial with a specified degree. The RBF interpolant is written
    as

    .. math::
        f(x) = K(x, y) a + P(x) b,

    where :math:`K(x, y)` is a matrix of RBFs with centers at :math:`y`
    """

    def __init__(self, y, d, neighbors=None, smoothing=0, kernel='thin_plate_spline', epsilon=None, degree=None):
        """
        Initialize RBFInterpolator object with given parameters.

        Parameters
        ----------
        y : (npoints, ndims) array_like
            Coordinates of data points.
        d : (npoints, ...) array_like
            Values of data points.
        neighbors : int, optional
            Number of nearest data points to use in interpolation.
        smoothing : float or (npoints, ) array_like, optional
            Smoothing parameter for interpolation.
        kernel : str, optional
            Type of radial basis function (RBF).
        epsilon : float, optional
            Shape parameter scaling the input to the RBF.
        degree : int, optional
            Degree of added polynomial.
        """
        # Initialize the object with given parameters
        pass
    evaluated at the points :math:`x`, and :math:`P(x)` is a matrix of
    monomials, which span polynomials with the specified degree, evaluated at
    :math:`x`. The coefficients :math:`a` and :math:`b` are the solution to the
    linear equations

    .. math::
        (K(y, y) + \\lambda I) a + P(y) b = d

    and

    .. math::
        P(y)^T a = 0,

    where :math:`\\lambda` is a non-negative smoothing parameter that controls
    how well we want to fit the data. The data are fit exactly when the
    smoothing parameter is 0.

    The above system is uniquely solvable if the following requirements are
    met:

        - :math:`P(y)` must have full column rank. :math:`P(y)` always has full
          column rank when `degree` is -1 or 0. When `degree` is 1,
          :math:`P(y)` has full column rank if the data point locations are not
          all collinear (N=2), coplanar (N=3), etc.
        - If `kernel` is 'multiquadric', 'linear', 'thin_plate_spline',
          'cubic', or 'quintic', then `degree` must not be lower than the
          minimum value listed above.
        - If `smoothing` is 0, then each data point location must be distinct.

    When using an RBF that is not scale invariant ('multiquadric',
    'inverse_multiquadric', 'inverse_quadratic', or 'gaussian'), an appropriate
    shape parameter must be chosen (e.g., through cross validation). Smaller
    values for the shape parameter correspond to wider RBFs. The problem can
    become ill-conditioned or singular when the shape parameter is too small.

    The memory required to solve for the RBF interpolation coefficients
    increases quadratically with the number of data points, which can become
    impractical when interpolating more than about a thousand data points.
    To overcome memory limitations for large interpolation problems, the
    `neighbors` argument can be specified to compute an RBF interpolant for
    each evaluation point using only the nearest data points.

    .. versionadded:: 1.7.0

    See Also
    --------
    NearestNDInterpolator
    LinearNDInterpolator
    CloughTocher2DInterpolator

    References
    ----------
    .. [1] Fasshauer, G., 2007. Meshfree Approximation Methods with Matlab.
        World Scientific Publishing Co.

    .. [2] http://amadeus.math.iit.edu/~fass/603_ch3.pdf

    .. [3] Wahba, G., 1990. Spline Models for Observational Data. SIAM.

    .. [4] http://pages.stat.wisc.edu/~wahba/stat860public/lect/lect8/lect8.pdf

    Examples
    --------
    Demonstrate interpolating scattered data to a grid in 2-D.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import RBFInterpolator
    >>> from scipy.stats.qmc import Halton

    >>> rng = np.random.default_rng()
    >>> xobs = 2*Halton(2, seed=rng).random(100) - 1
    >>> yobs = np.sum(xobs, axis=1)*np.exp(-6*np.sum(xobs**2, axis=1))

    >>> xgrid = np.mgrid[-1:1:50j, -1:1:50j]
    >>> xflat = xgrid.reshape(2, -1).T
    >>> yflat = RBFInterpolator(xobs, yobs)(xflat)
    >>> ygrid = yflat.reshape(50, 50)


    # 使用 RBF 插值器对观测点进行插值，得到平坦化后的结果
    yflat = RBFInterpolator(xobs, yobs)(xflat)
    # 将平坦化后的结果重新整形成网格形式
    ygrid = yflat.reshape(50, 50)



    >>> fig, ax = plt.subplots()
    >>> ax.pcolormesh(*xgrid, ygrid, vmin=-0.25, vmax=0.25, shading='gouraud')
    >>> p = ax.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)
    >>> fig.colorbar(p)
    >>> plt.show()


    # 创建图形和轴对象
    fig, ax = plt.subplots()
    # 使用伪颜色网格绘制插值结果
    ax.pcolormesh(*xgrid, ygrid, vmin=-0.25, vmax=0.25, shading='gouraud')
    # 绘制观测点的散点图
    p = ax.scatter(*xobs.T, c=yobs, s=50, ec='k', vmin=-0.25, vmax=0.25)
    # 添加颜色条
    fig.colorbar(p)
    # 显示图形
    plt.show()



    """


    # 下面是 _chunk_evaluator 方法的定义，用于控制内存消耗进行插值评估
    def _chunk_evaluator(
            self,
            x,
            y,
            shift,
            scale,
            coeffs,
            memory_budget=1000000
    ):



        """
        Evaluate the interpolation while controlling memory consumption.
        We chunk the input if we need more memory than specified.

        Parameters
        ----------
        x : (Q, N) float ndarray
            array of points on which to evaluate
        y: (P, N) float ndarray
            array of points on which we know function values
        shift: (N, ) ndarray
            Domain shift used to create the polynomial matrix.
        scale : (N,) float ndarray
            Domain scaling used to create the polynomial matrix.
        coeffs: (P+R, S) float ndarray
            Coefficients in front of basis functions
        memory_budget: int
            Total amount of memory (in units of sizeof(float)) we wish
            to devote for storing the array of coefficients for
            interpolated points. If we need more memory than that, we
            chunk the input.

        Returns
        -------
        (Q, S) float ndarray
            Interpolated array
        """


        # 获取输入 x 的形状信息
        nx, ndim = x.shape
        # 确定邻居点数目
        if self.neighbors is None:
            nnei = len(y)
        else:
            nnei = self.neighbors
        # 计算每个数据块的大小，确保内存消耗不超过预算
        chunksize = memory_budget // (self.powers.shape[0] + nnei) + 1
        # 如果每个数据块大小小于等于数据点数，采用分块方法进行评估
        if chunksize <= nx:
            out = np.empty((nx, self.d.shape[1]), dtype=float)
            for i in range(0, nx, chunksize):
                # 构建评估系数向量
                vec = _build_evaluation_coefficients(
                    x[i:i + chunksize, :],
                    y,
                    self.kernel,
                    self.epsilon,
                    self.powers,
                    shift,
                    scale)
                # 计算评估结果并存储到输出数组中
                out[i:i + chunksize, :] = np.dot(vec, coeffs)
        else:
            # 如果内存足够，直接计算评估结果
            vec = _build_evaluation_coefficients(
                x,
                y,
                self.kernel,
                self.epsilon,
                self.powers,
                shift,
                scale)
            out = np.dot(vec, coeffs)
        # 返回插值评估结果
        return out
```