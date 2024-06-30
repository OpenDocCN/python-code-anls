# `D:\src\scipysrc\scipy\scipy\optimize\_lsq\least_squares.py`

```
"""Generic interface for least-squares minimization."""
# 从警告模块中导入 warn 函数
from warnings import warn

# 导入 NumPy 库，并别名为 np
import numpy as np
# 从 NumPy 线性代数模块中导入 norm 函数
from numpy.linalg import norm

# 导入 SciPy 稀疏矩阵模块中的相关函数和类
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator

# 导入 SciPy 优化模块中的部分函数和类
from scipy.optimize import _minpack, OptimizeResult
# 导入 SciPy 优化模块中的数值微分函数和列分组函数
from scipy.optimize._numdiff import approx_derivative, group_columns
# 导入 SciPy 优化模块中的 _minimize 模块中的 Bounds 类
from scipy.optimize._minimize import Bounds

# 从当前目录中导入 trf、dogbox 和 common 模块
from .trf import trf
from .dogbox import dogbox
from .common import EPS, in_bounds, make_strictly_feasible

# 定义优化过程中的终止信息字典
TERMINATION_MESSAGES = {
    -1: "Improper input parameters status returned from `leastsq`",
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied."
}

# 定义从 MINPACK 状态码到通用状态码的映射
FROM_MINPACK_TO_COMMON = {
    0: -1,  # Improper input parameters from MINPACK.
    1: 2,
    2: 3,
    3: 4,
    4: 1,
    5: 0
    # 还有 6, 7, 8 对应过小的容差参数，
    # 但我们通过预先检查 ftol、xtol、gtol 防范这种情况。
}


# 定义调用 MINPACK 最小化函数的函数
def call_minpack(fun, x0, jac, ftol, xtol, gtol, max_nfev, x_scale, diff_step):
    n = x0.size

    # 如果未指定数值微分步长，使用默认的 EPS
    if diff_step is None:
        epsfcn = EPS
    else:
        epsfcn = diff_step**2

    # 计算 MINPACK 的 `diag`，它是我们的 `x_scale` 的倒数，
    # 当 ``x_scale='jac'`` 时对应 ``diag=None``。
    if isinstance(x_scale, str) and x_scale == 'jac':
        diag = None
    else:
        diag = 1 / x_scale

    full_output = True
    col_deriv = False
    factor = 100.0

    # 如果未提供雅可比矩阵，根据默认策略计算最大函数评估次数
    if jac is None:
        if max_nfev is None:
            # 要考虑雅可比矩阵评估的 n 的平方倍。
            max_nfev = 100 * n * (n + 1)
        x, info, status = _minpack._lmdif(
            fun, x0, (), full_output, ftol, xtol, gtol,
            max_nfev, epsfcn, factor, diag)
    else:
        # 否则，根据提供的雅可比矩阵计算最大函数评估次数
        if max_nfev is None:
            max_nfev = 100 * n
        x, info, status = _minpack._lmder(
            fun, jac, x0, (), full_output, col_deriv,
            ftol, xtol, gtol, max_nfev, factor, diag)

    # 计算最小二乘问题的残差向量 f
    f = info['fvec']

    # 如果 jac 是可调用函数，则计算雅可比矩阵 J
    if callable(jac):
        J = jac(x)
    else:
        J = np.atleast_2d(approx_derivative(fun, x))

    # 计算代价函数值
    cost = 0.5 * np.dot(f, f)
    # 计算梯度
    g = J.T.dot(f)
    # 计算梯度的无穷范数
    g_norm = norm(g, ord=np.inf)

    # 获取函数评估次数和雅可比矩阵评估次数
    nfev = info['nfev']
    njev = info.get('njev', None)

    # 将 MINPACK 状态码映射到通用状态码
    status = FROM_MINPACK_TO_COMMON[status]

    # 创建优化结果对象并返回
    active_mask = np.zeros_like(x0, dtype=int)
    return OptimizeResult(
        x=x, cost=cost, fun=f, jac=J, grad=g, optimality=g_norm,
        active_mask=active_mask, nfev=nfev, njev=njev, status=status)


# 准备边界条件函数，将边界转换为 NumPy 数组的形式
def prepare_bounds(bounds, n):
    lb, ub = (np.asarray(b, dtype=float) for b in bounds)
    if lb.ndim == 0:
        lb = np.resize(lb, n)

    if ub.ndim == 0:
        ub = np.resize(ub, n)

    return lb, ub


# 检查容差函数，用于确保容差参数的有效性
def check_tolerance(ftol, xtol, gtol, method):
    # 检查和调整容差值，确保它们大于等于零
    def check(tol, name):
        # 如果容差值为 None，则设置为 0
        if tol is None:
            tol = 0
        # 如果容差值小于机器精度 EPS，则发出警告并设置容差值为 EPS
        elif tol < EPS:
            warn(f"Setting `{name}` below the machine epsilon ({EPS:.2e}) effectively "
                 f"disables the corresponding termination condition.",
                 stacklevel=3)
        # 返回调整后的容差值
        return tol

    # 对于给定的容差值，调用 check 函数进行检查和调整
    ftol = check(ftol, "ftol")
    xtol = check(xtol, "xtol")
    gtol = check(gtol, "gtol")

    # 如果使用 Levenberg-Marquardt 方法且任何一个容差值小于 EPS，则抛出值错误异常
    if method == "lm" and (ftol < EPS or xtol < EPS or gtol < EPS):
        raise ValueError("All tolerances must be higher than machine epsilon "
                         f"({EPS:.2e}) for method 'lm'.")
    # 如果所有容差值均小于 EPS，则抛出值错误异常
    elif ftol < EPS and xtol < EPS and gtol < EPS:
        raise ValueError("At least one of the tolerances must be higher than "
                         f"machine epsilon ({EPS:.2e}).")

    # 返回调整后的容差值 ftol, xtol, gtol
    return ftol, xtol, gtol
def least_squares(
        fun, x0, jac='2-point', bounds=(-np.inf, np.inf), method='trf',
        ftol=1e-8, xtol=1e-8, gtol=1e-8, x_scale=1.0, loss='linear',
        f_scale=1.0, diff_step=None, tr_solver=None, tr_options={},
        jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs={}):
    """
    Define a least squares optimization function with various parameters.

    Parameters:
    - fun: Callable function to minimize.
    - x0: Initial guess for the solution.
    - jac: Method for computing the Jacobian ('2-point' or callable).
    - bounds: Tuple defining the lower and upper bounds for variables.
    - method: Optimization method ('trf' by default).
    - ftol, xtol, gtol: Tolerances for convergence criteria.
    - x_scale: Scaling factor for variables.
    - loss: Loss function type or callable.
    - f_scale: Scaling factor for the residual (cost function).
    - diff_step: Step size for numerical differentiation.
    - tr_solver: Trust-region solver for nonlinear least squares.
    - tr_options: Options for the trust-region solver.
    - jac_sparsity: Sparsity structure of the Jacobian matrix.
    - max_nfev: Maximum number of function evaluations.
    - verbose: Verbosity level.
    - args, kwargs: Additional arguments and keyword arguments for `fun`.

    Returns:
    - Callable `least_squares` function with defined optimization settings.
    """

    # Check and potentially resize x_scale to match the shape of x0
    if isinstance(x_scale, str) and x_scale == 'jac':
        return x_scale

    try:
        x_scale = np.asarray(x_scale, dtype=float)
        valid = np.all(np.isfinite(x_scale)) and np.all(x_scale > 0)
    except (ValueError, TypeError):
        valid = False

    if not valid:
        raise ValueError("`x_scale` must be 'jac' or array_like with "
                         "positive numbers.")

    if x_scale.ndim == 0:
        x_scale = np.resize(x_scale, x0.shape)

    if x_scale.shape != x0.shape:
        raise ValueError("Inconsistent shapes between `x_scale` and `x0`.")

    return x_scale
    """Solve a nonlinear least-squares problem with bounds on the variables.

    Given the residuals f(x) (an m-D real function of n real
    variables) and the loss function rho(s) (a scalar function), `least_squares`
    finds a local minimum of the cost function F(x)::

        minimize F(x) = 0.5 * sum(rho(f_i(x)**2), i = 0, ..., m - 1)
        subject to lb <= x <= ub

    The purpose of the loss function rho(s) is to reduce the influence of
    outliers on the solution.

    Parameters
    ----------
    fun : callable
        Function which computes the vector of residuals, with the signature
        ``fun(x, *args, **kwargs)``, i.e., the minimization proceeds with
        respect to its first argument. The argument ``x`` passed to this
        function is an ndarray of shape (n,) (never a scalar, even for n=1).
        It must allocate and return a 1-D array_like of shape (m,) or a scalar.
        If the argument ``x`` is complex or the function ``fun`` returns
        complex residuals, it must be wrapped in a real function of real
        arguments, as shown at the end of the Examples section.
    x0 : array_like with shape (n,) or float
        Initial guess on independent variables. If float, it will be treated
        as a 1-D array with one element. When `method` is 'trf', the initial
        guess might be slightly adjusted to lie sufficiently within the given
        `bounds`.
    jac : {'2-point', '3-point', 'cs', callable}, optional
        Method of computing the Jacobian matrix (an m-by-n matrix, where
        element (i, j) is the partial derivative of f[i] with respect to
        x[j]). The keywords select a finite difference scheme for numerical
        estimation. The scheme '3-point' is more accurate, but requires
        twice as many operations as '2-point' (default). The scheme 'cs'
        uses complex steps, and while potentially the most accurate, it is
        applicable only when `fun` correctly handles complex inputs and
        can be analytically continued to the complex plane. Method 'lm'
        always uses the '2-point' scheme. If callable, it is used as
        ``jac(x, *args, **kwargs)`` and should return a good approximation
        (or the exact value) for the Jacobian as an array_like (np.atleast_2d
        is applied), a sparse matrix (csr_matrix preferred for performance) or
        a `scipy.sparse.linalg.LinearOperator`.
    bounds : 2-tuple of array_like or `Bounds`, optional
        There are two ways to specify bounds:

            1. Instance of `Bounds` class
            2. Lower and upper bounds on independent variables. Defaults to no
               bounds. Each array must match the size of `x0` or be a scalar,
               in the latter case a bound will be the same for all variables.
               Use ``np.inf`` with an appropriate sign to disable bounds on all
               or some variables.
    """
    method : {'trf', 'dogbox', 'lm'}, optional
        算法选择，用于执行最小化操作。

            * 'trf' : 反射信任区域算法，特别适用于具有边界的大型稀疏问题。通常是一种稳健的方法。
            * 'dogbox' : 矩形信任区域的狗腿算法，典型应用于有边界的小问题。不推荐用于具有秩亏雅可比矩阵的问题。
            * 'lm' : MINPACK 中实现的 Levenberg-Marquardt 算法。不处理边界和稀疏雅可比矩阵。通常是小型无约束问题中最有效的方法。

        默认为 'trf'。详细信息请参阅备注。
    ftol : float or None, optional
        成本函数变化的终止容差。默认为 1e-8。当 ``dF < ftol * F`` 且在上一步中局部二次模型与真实模型有足够一致时，优化过程停止。

        如果为 None 并且 'method' 不是 'lm'，则不通过此条件终止。如果 'method' 是 'lm'，则此容差必须高于机器精度。
    xtol : float or None, optional
        自变量变化的终止容差。默认为 1e-8。确切条件取决于使用的 `method`：

            * 对于 'trf' 和 'dogbox' ：``norm(dx) < xtol * (xtol + norm(x))``。
            * 对于 'lm' ：``Delta < xtol * norm(xs)``，其中 ``Delta`` 是信任区域半径，``xs`` 是根据 `x_scale` 参数缩放后的 `x` 的值（见下文）。

        如果为 None 并且 'method' 不是 'lm'，则不通过此条件终止。如果 'method' 是 'lm'，则此容差必须高于机器精度。
    gtol : float or None, optional
        梯度范数终止容差。默认为 1e-8。确切条件取决于使用的 `method`：

            * 对于 'trf' ：``norm(g_scaled, ord=np.inf) < gtol``，其中 ``g_scaled`` 是考虑边界存在而缩放的梯度值 [STIR]_。
            * 对于 'dogbox' ：``norm(g_free, ord=np.inf) < gtol``，其中 ``g_free`` 是相对于边界上非最优状态的变量的梯度。
            * 对于 'lm' ：雅可比矩阵的列与残差向量之间夹角的最大绝对余弦值小于 `gtol`，或者残差向量为零。

        如果为 None 并且 'method' 不是 'lm'，则不通过此条件终止。如果 'method' 是 'lm'，则此容差必须高于机器精度。
    x_scale : array_like or 'jac', optional
        # 变量的特征尺度。设置 `x_scale` 相当于用缩放变量 ``xs = x / x_scale`` 重新定义问题。
        # 另一个角度是，每个变量的信赖域大小与 `x_scale[j]` 成正比。
        # 设置 `x_scale` 可以改善收敛性，使得在缩放变量上的一定大小步长对成本函数产生类似的影响。
        # 如果设为 'jac'，则通过迭代更新使用雅可比矩阵列的逆范数来自适应地调整尺度（如 [JJMore]_ 描述）。

    loss : str or callable, optional
        # 确定损失函数。以下关键字值可选：

            * 'linear'（默认）: ``rho(z) = z``。标准的最小二乘问题。
            * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``。l1（绝对值）损失的平滑近似，通常用于鲁棒最小二乘。
            * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``。类似于 'soft_l1'。
            * 'cauchy' : ``rho(z) = ln(1 + z)``。大大减弱离群值的影响，但可能在优化过程中造成困难。
            * 'arctan' : ``rho(z) = arctan(z)``。限制单个残差的最大损失，具有类似 'cauchy' 的性质。

        # 如果是可调用对象，必须接受一个1-D ndarray ``z=f**2``，并返回一个形状为 (3, m) 的 array_like，
        # 其中第 0 行包含函数值，第 1 行包含一阶导数，第 2 行包含二阶导数。方法 'lm' 仅支持 'linear' 损失。

    f_scale : float, optional
        # 内点和外点残差之间的软边界值，默认为 1.0。
        # 损失函数的评估方式为 ``rho_(f**2) = C**2 * rho(f**2 / C**2)``,
        # 其中 ``C`` 是 `f_scale`，而 `rho` 取决于 `loss` 参数。
        # 对于 ``loss='linear'``，此参数无效，但对于其他 `loss` 值则至关重要。

    max_nfev : None or int, optional
        # 终止前的最大函数评估次数。如果为 None（默认），则自动选择：

            * 对于 'trf' 和 'dogbox'：100 * n。
            * 对于 'lm'：如果 `jac` 是可调用的，则为 100 * n；否则为 100 * n * (n + 1)（因为 'lm' 在雅可比估计中计算函数调用次数）。

    diff_step : None or array_like, optional
        # 确定有限差分近似雅可比矩阵的相对步长。实际步长计算为 ``x * diff_step``。
        # 如果为 None（默认），则 `diff_step` 被取为用于有限差分方案的传统“最佳”机器epsilon的幂次 [NR]_。
    tr_solver : {None, 'exact', 'lsmr'}, optional
        # 定义用于解决信任区域子问题的方法，仅适用于 'trf' 和 'dogbox' 方法。
        # 
        # * 'exact' 适用于具有稠密雅可比矩阵的不太大的问题。每次迭代的计算复杂度与雅可比矩阵的奇异值分解相当。
        # * 'lsmr' 适用于具有稀疏和大型雅可比矩阵的问题。它使用迭代过程 `scipy.sparse.linalg.lsmr` 来找到线性最小二乘问题的解，并且仅需要矩阵-向量乘积的评估。
        #
        # 如果为 None（默认），则根据第一次迭代返回的雅可比矩阵的类型选择求解器。
    tr_options : dict, optional
        # 传递给信任区域求解器的关键字选项。
        #
        # * ``tr_solver='exact'``: `tr_options` 将被忽略。
        # * ``tr_solver='lsmr'``: 适用于 `scipy.sparse.linalg.lsmr` 的选项。
        #   此外，``method='trf'`` 支持 'regularize' 选项（布尔值，默认为 True），它向正规方程添加正则化项，如果雅可比矩阵是秩缺乏的 [Byrd]_ (eq. 3.4) 将提高收敛性。
    jac_sparsity : {None, array_like, sparse matrix}, optional
        # 定义有限差分估计雅可比矩阵的稀疏结构，其形状必须为 (m, n)。
        # 如果每行的雅可比矩阵只有少数非零元素，则提供稀疏结构将极大加快计算速度 [Curtis]_。
        # 零条目表示雅可比矩阵中对应元素完全为零。如果提供，则强制使用 'lsmr' 信任区域求解器。
        # 如果为 None（默认），则将使用密集差分。对于 'lm' 方法无效。
    verbose : {0, 1, 2}, optional
        # 算法的详细程度：
        #
        # * 0（默认）：静默工作。
        # * 1：显示终止报告。
        # * 2：显示迭代过程中的进展（'lm' 方法不支持）。
    args, kwargs : tuple and dict, optional
        # 传递给 `fun` 和 `jac` 的额外参数。默认均为空。
        # 调用签名为 ``fun(x, *args, **kwargs)``，`jac` 的签名相同。
    result : OptimizeResult
        # `result` 是一个 OptimizeResult 对象，包含以下字段：

            x : ndarray, shape (n,)
                # 找到的解决方案，形状为 (n,) 的数组。
            cost : float
                # 解决方案处的成本函数值。
            fun : ndarray, shape (m,)
                # 解决方案处的残差向量，形状为 (m,) 的数组。
            jac : ndarray, sparse matrix or LinearOperator, shape (m, n)
                # 解决方案处的修改后的雅可比矩阵，表示 J^T J 是成本函数海森矩阵的 Gauss-Newton 近似。
                # 类型与算法使用的类型相同。
            grad : ndarray, shape (m,)
                # 解决方案处的成本函数梯度。
            optimality : float
                # 一阶最优性度量。在无约束问题中，它始终是梯度的均匀范数。
                # 在约束问题中，它是与 `gtol` 在迭代期间进行比较的数量。
            active_mask : ndarray of int, shape (n,)
                # 每个分量显示相应约束是否活动（即变量是否在边界上）：

                    *  0 : 约束未激活。
                    * -1 : 下界激活。
                    *  1 : 上界激活。

                # 对于 'trf' 方法可能有些随意，因为它生成严格可行的迭代序列，并且 `active_mask`
                # 在容差阈值内确定。
            nfev : int
                # 完成的函数评估次数。'trf' 和 'dogbox' 方法不计算数值雅可比逼近的函数调用，
                # 而 'lm' 方法则不同。
            njev : int or None
                # 完成的雅可比评估次数。如果 'lm' 方法使用数值雅可比逼近，则设置为 None。
            status : int
                # 算法终止的原因：

                    * -1 : MINPACK 返回的不合适的输入参数状态。
                    *  0 : 超过最大函数评估次数。
                    *  1 : 满足 `gtol` 终止条件。
                    *  2 : 满足 `ftol` 终止条件。
                    *  3 : 满足 `xtol` 终止条件。
                    *  4 : 同时满足 `ftol` 和 `xtol` 终止条件。

            message : str
                # 终止原因的口头描述。
            success : bool
                # 如果满足收敛标准之一，则为 True（`status` > 0）。
    # 方法 'lm' (Levenberg-Marquardt) 调用了 MINPACK 中实现的最小二乘算法的包装器 (lmder, lmdif)。
    # 它运行了 Levenberg-Marquardt 算法，该算法被表述为一种信任域类型的算法。
    # 实现基于文献 [JJMore]_，非常稳健和高效，具有许多智能技巧。
    # 对于无约束问题，这应该是首选算法。注意，它不支持边界约束。
    # 当 m < n 时，它无法工作。

    # 方法 'trf' (Trust Region Reflective) 的动机源于解决一组方程，这些方程构成了边界约束最小化问题的一阶最优性条件，如 [STIR]_ 中所述。
    # 该算法通过迭代解决信任域子问题，特别增加了一个特殊的对角二次项，并且信任域的形状由边界距离和梯度方向确定。
    # 这些增强措施有助于避免直接朝边界步进，并有效地探索变量空间的整体。
    # 为了进一步改善收敛性，该算法考虑从边界反射的搜索方向。
    # 为了遵守理论要求，算法保持迭代严格可行。
    # 对于稠密雅可比矩阵，信任域子问题通过一种类似于 [JJMore]_ 描述的精确方法解决（MINPACK 中实现的方法）。
    # 与 MINPACK 实现的不同之处在于，每次迭代只进行一次雅可比矩阵的奇异值分解，而不是 QR 分解和一系列 Givens 旋转消除。
    # 对于大型稀疏雅可比矩阵，使用二维子空间方法来解决信任域子问题，该方法由 `scipy.sparse.linalg.lsmr` 提供一个缩放梯度和近似 Gauss-Newton 解决方案组成。
    # 在不添加约束时，该算法与 MINPACK 非常相似，并且通常具有可比较的性能。
    # 该算法在无界和有界问题中表现相当稳健，因此被选择为默认算法。

    # 方法 'dogbox' 在信任域框架内操作，但考虑的是矩形信任域，而不是传统的椭圆体 [Voglis]_。
    # 当前信任域与初始边界的交集再次是矩形的，因此每次迭代都通过 Powell 的 dogleg 方法 [NumOpt]_ 解决约束条件下的二次最小化问题。
    # 对于稠密雅可比矩阵，所需的 Gauss-Newton 步骤可以精确计算；对于大型稀疏雅可比矩阵，则可以通过 `scipy.sparse.linalg.lsmr` 进行近似计算。
    # 当雅可比矩阵的秩小于变量数时，该算法可能表现出缓慢的收敛性。
    # 在具有少量变量的约束问题中，该算法通常比 'trf' 表现更优。

    # 实现了鲁棒的损失函数，如 [BA]_ 中所述的方法。
    is to modify a residual vector and a Jacobian matrix on each iteration
    such that computed gradient and Gauss-Newton Hessian approximation match
    the true gradient and Hessian approximation of the cost function. Then
    the algorithm proceeds in a normal way, i.e., robust loss functions are
    implemented as a simple wrapper over standard least-squares algorithms.

    .. versionadded:: 0.17.0

    References
    ----------
    .. [STIR] M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior,
              and Conjugate Gradient Method for Large-Scale Bound-Constrained
              Minimization Problems," SIAM Journal on Scientific Computing,
              Vol. 21, Number 1, pp 1-23, 1999.
    .. [NR] William H. Press et. al., "Numerical Recipes. The Art of Scientific
            Computing. 3rd edition", Sec. 5.7.
    .. [Byrd] R. H. Byrd, R. B. Schnabel and G. A. Shultz, "Approximate
              solution of the trust region problem by minimization over
              two-dimensional subspaces", Math. Programming, 40, pp. 247-263,
              1988.
    .. [Curtis] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
                sparse Jacobian matrices", Journal of the Institute of
                Mathematics and its Applications, 13, pp. 117-120, 1974.
    .. [JJMore] J. J. More, "The Levenberg-Marquardt Algorithm: Implementation
                and Theory," Numerical Analysis, ed. G. A. Watson, Lecture
                Notes in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
    .. [Voglis] C. Voglis and I. E. Lagaris, "A Rectangular Trust Region
                Dogleg Approach for Unconstrained and Bound Constrained
                Nonlinear Optimization", WSEAS International Conference on
                Applied Mathematics, Corfu, Greece, 2004.
    .. [NumOpt] J. Nocedal and S. J. Wright, "Numerical optimization,
                2nd edition", Chapter 4.
    .. [BA] B. Triggs et. al., "Bundle Adjustment - A Modern Synthesis",
            Proceedings of the International Workshop on Vision Algorithms:
            Theory and Practice, pp. 298-372, 1999.

    Examples
    --------
    In this example we find a minimum of the Rosenbrock function without bounds
    on independent variables.

    >>> import numpy as np
    >>> def fun_rosenbrock(x):
    ...     return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])

    Notice that we only provide the vector of the residuals. The algorithm
    constructs the cost function as a sum of squares of the residuals, which
    gives the Rosenbrock function. The exact minimum is at ``x = [1.0, 1.0]``.

    >>> from scipy.optimize import least_squares
    >>> x0_rosenbrock = np.array([2, 2])
    >>> res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)
    >>> res_1.x
    array([ 1.,  1.])
    >>> res_1.cost
    9.8669242910846867e-30
    >>> res_1.optimality
    8.8928864934219529e-14

    We now constrain the variables, in such a way that the previous solution
    becomes infeasible. Specifically, we require that ``x[1] >= 1.5``, and
    ``x[0]`` left unconstrained. To this end, we specify the `bounds` parameter
    to `least_squares` in the form ``bounds=([-np.inf, 1.5], np.inf)``.



    We also provide the analytic Jacobian:

    >>> def jac_rosenbrock(x):
    ...     return np.array([
    ...         [-20 * x[0], 10],
    ...         [-1, 0]])



    Putting this all together, we see that the new solution lies on the bound:

    >>> res_2 = least_squares(fun_rosenbrock, x0_rosenbrock, jac_rosenbrock,
    ...                       bounds=([-np.inf, 1.5], np.inf))
    >>> res_2.x
    array([ 1.22437075,  1.5       ])
    >>> res_2.cost
    0.025213093946805685
    >>> res_2.optimality
    1.5885401433157753e-07



    Now we solve a system of equations (i.e., the cost function should be zero
    at a minimum) for a Broyden tridiagonal vector-valued function of 100000
    variables:

    >>> def fun_broyden(x):
    ...     f = (3 - x) * x + 1
    ...     f[1:] -= x[:-1]
    ...     f[:-1] -= 2 * x[1:]
    ...     return f



    The corresponding Jacobian matrix is sparse. We tell the algorithm to
    estimate it by finite differences and provide the sparsity structure of
    Jacobian to significantly speed up this process.

    >>> from scipy.sparse import lil_matrix
    >>> def sparsity_broyden(n):
    ...     sparsity = lil_matrix((n, n), dtype=int)
    ...     i = np.arange(n)
    ...     sparsity[i, i] = 1
    ...     i = np.arange(1, n)
    ...     sparsity[i, i - 1] = 1
    ...     i = np.arange(n - 1)
    ...     sparsity[i, i + 1] = 1
    ...     return sparsity
    ...
    >>> n = 100000
    >>> x0_broyden = -np.ones(n)
    ...
    >>> res_3 = least_squares(fun_broyden, x0_broyden,
    ...                       jac_sparsity=sparsity_broyden(n))
    >>> res_3.cost
    4.5687069299604613e-23
    >>> res_3.optimality
    1.1650454296851518e-11



    Let's also solve a curve fitting problem using robust loss function to
    take care of outliers in the data. Define the model function as
    ``y = a + b * exp(c * t)``, where t is a predictor variable, y is an
    observation and a, b, c are parameters to estimate.

    First, define the function which generates the data with noise and
    outliers, define the model parameters, and generate data:

    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> def gen_data(t, a, b, c, noise=0., n_outliers=0, seed=None):
    ...     rng = default_rng(seed)
    ...
    ...     y = a + b * np.exp(t * c)
    ...
    ...     error = noise * rng.standard_normal(t.size)
    ...     outliers = rng.integers(0, t.size, n_outliers)
    ...     error[outliers] *= 10
    ...
    ...     return y + error
    ...
    >>> a = 0.5
    >>> b = 2.0
    >>> c = -1
    >>> t_min = 0
    >>> t_max = 10
    >>> n_points = 15
    ...
    >>> t_train = np.linspace(t_min, t_max, n_points)
    >>> y_train = gen_data(t_train, a, b, c, noise=0.1, n_outliers=3)
    如果 `method` 不在 ['trf', 'dogbox', 'lm'] 中，则抛出 ValueError 异常。
    这些是 `least_squares` 函数中可选的优化方法。

    如果 `jac` 不在 ['2-point', '3-point', 'cs'] 中，并且也不是一个可调用的函数，则抛出 ValueError 异常。
    这些是 `least_squares` 函数中用于计算雅可比矩阵的方法选项。
    # 检查 `tr_solver` 参数是否在预期的取值范围内，如果不在则抛出数值错误
    if tr_solver not in [None, 'exact', 'lsmr']:
        raise ValueError("`tr_solver` must be None, 'exact' or 'lsmr'.")

    # 检查 `loss` 参数是否在已实现的损失函数中或者是可调用的函数，否则抛出数值错误
    if loss not in IMPLEMENTED_LOSSES and not callable(loss):
        raise ValueError("`loss` must be one of {} or a callable."
                         .format(IMPLEMENTED_LOSSES.keys()))

    # 如果 `method` 参数为 'lm'，则检查 `loss` 是否为 'linear'，否则抛出数值错误
    if method == 'lm' and loss != 'linear':
        raise ValueError("method='lm' supports only 'linear' loss function.")

    # 检查 `verbose` 参数是否在 [0, 1, 2] 中，如果不在则抛出数值错误
    if verbose not in [0, 1, 2]:
        raise ValueError("`verbose` must be in [0, 1, 2].")

    # 检查 `max_nfev` 参数是否为正整数或者 None，如果不是则抛出数值错误
    if max_nfev is not None and max_nfev <= 0:
        raise ValueError("`max_nfev` must be None or positive integer.")

    # 检查 `x0` 参数是否为实数，如果不是则抛出数值错误
    if np.iscomplexobj(x0):
        raise ValueError("`x0` must be real.")

    # 将 `x0` 参数转换为至少是 1 维的浮点数数组
    x0 = np.atleast_1d(x0).astype(float)

    # 检查 `x0` 参数的维度是否不超过 1，如果超过则抛出数值错误
    if x0.ndim > 1:
        raise ValueError("`x0` must have at most 1 dimension.")

    # 如果 `bounds` 是 Bounds 类型，则从中获取上下界；否则，根据长度准备上下界
    if isinstance(bounds, Bounds):
        lb, ub = bounds.lb, bounds.ub
        bounds = (lb, ub)
    else:
        # 如果 `bounds` 不是长度为 2 的元组，则抛出数值错误
        if len(bounds) == 2:
            lb, ub = prepare_bounds(bounds, x0.shape[0])
        else:
            raise ValueError("`bounds` must contain 2 elements.")

    # 如果 `method` 为 'lm'，则检查上下界是否全部为无穷大，如果是则抛出数值错误
    if method == 'lm' and not np.all((lb == -np.inf) & (ub == np.inf)):
        raise ValueError("Method 'lm' doesn't support bounds.")

    # 检查上下界 `lb` 和 `ub` 与 `x0` 参数的形状是否一致，如果不一致则抛出数值错误
    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("Inconsistent shapes between bounds and `x0`.")

    # 检查每个下界是否严格小于对应的上界，如果不是则抛出数值错误
    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be strictly less than each "
                         "upper bound.")

    # 检查 `x0` 是否在给定的上下界内，如果不在则抛出数值错误
    if not in_bounds(x0, lb, ub):
        raise ValueError("`x0` is infeasible.")

    # 检查 `x_scale` 参数，确保其符合要求，并返回检查后的 `x_scale`
    x_scale = check_x_scale(x_scale, x0)

    # 检查容差参数 `ftol`, `xtol`, `gtol` 是否符合要求，并返回检查后的值
    ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol, method)

    # 如果 `method` 为 'trf'，则调用函数使 `x0` 变得严格可行
    if method == 'trf':
        x0 = make_strictly_feasible(x0, lb, ub)

    # 定义一个包装函数 `fun_wrapped`，确保 `fun` 返回至少是 1 维的数组
    def fun_wrapped(x):
        return np.atleast_1d(fun(x, *args, **kwargs))

    # 计算初始点 `x0` 处的函数值 `f0`
    f0 = fun_wrapped(x0)

    # 检查 `f0` 的维度是否不超过 1，如果超过则抛出数值错误
    if f0.ndim != 1:
        raise ValueError("`fun` must return at most 1-d array_like. "
                         f"f0.shape: {f0.shape}")

    # 检查初始点处的残差是否均为有限值，如果有非有限值则抛出数值错误
    if not np.all(np.isfinite(f0)):
        raise ValueError("Residuals are not finite in the initial point.")

    # 获取 `x0` 和 `f0` 的大小信息
    n = x0.size
    m = f0.size

    # 如果 `method` 为 'lm'，则检查残差数量是否小于变量数量，如果是则抛出数值错误
    if method == 'lm' and m < n:
        raise ValueError("Method 'lm' doesn't work when the number of "
                         "residuals is less than the number of variables.")

    # 构造损失函数，根据 `loss` 参数的不同选择不同的方法，并计算初始成本
    loss_function = construct_loss_function(m, loss, f_scale)
    if callable(loss):
        rho = loss_function(f0)
        # 检查损失函数返回值的形状是否正确，如果不正确则抛出数值错误
        if rho.shape != (3, m):
            raise ValueError("The return value of `loss` callable has wrong "
                             "shape.")
        initial_cost = 0.5 * np.sum(rho[0])
    elif loss_function is not None:
        initial_cost = loss_function(f0, cost_only=True)
    else:
        initial_cost = 0.5 * np.dot(f0, f0)
    # 如果输入的 jac 是可调用对象，则计算初始点 x0 处的雅可比矩阵 J0
    if callable(jac):
        J0 = jac(x0, *args, **kwargs)

        # 如果 J0 是稀疏矩阵，则转换为 CSR 格式
        if issparse(J0):
            J0 = J0.tocsr()

            # 定义一个封装函数 jac_wrapped，用于返回稀疏的雅可比矩阵
            def jac_wrapped(x, _=None):
                return jac(x, *args, **kwargs).tocsr()

        # 如果 J0 是 LinearOperator 类型的对象
        elif isinstance(J0, LinearOperator):
            # 定义一个封装函数 jac_wrapped，直接返回雅可比矩阵
            def jac_wrapped(x, _=None):
                return jac(x, *args, **kwargs)

        # 如果 J0 是 ndarray 类型，则至少确保其为二维数组
        else:
            J0 = np.atleast_2d(J0)

            # 定义一个封装函数 jac_wrapped，确保返回至少是二维的 ndarray
            def jac_wrapped(x, _=None):
                return np.atleast_2d(jac(x, *args, **kwargs))

    else:  # 如果没有提供可调用的 jac 函数，则通过有限差分法估算雅可比矩阵。
        if method == 'lm':
            # 对于 Levenberg-Marquardt 方法，不支持 jac_sparsity 参数
            if jac_sparsity is not None:
                raise ValueError("method='lm' does not support "
                                 "`jac_sparsity`.")

            # 如果 jac 不是 '2-point'，则发出警告，因为 'lm' 方法默认使用 '2-point'
            if jac != '2-point':
                warn(f"jac='{jac}' works equivalently to '2-point' for method='lm'.",
                     stacklevel=2)

            # 将 J0 和 jac_wrapped 设为 None，表示不使用预定义的雅可比矩阵
            J0 = jac_wrapped = None
        else:
            # 如果 jac_sparsity 不为 None 并且 tr_solver 是 'exact'，则抛出错误
            if jac_sparsity is not None and tr_solver == 'exact':
                raise ValueError("tr_solver='exact' is incompatible "
                                 "with `jac_sparsity`.")

            # 检查并获取有效的 jac_sparsity，确保其与问题的 m 和 n 匹配
            jac_sparsity = check_jac_sparsity(jac_sparsity, m, n)

            # 定义一个封装函数 jac_wrapped，用于计算近似的雅可比矩阵
            def jac_wrapped(x, f):
                J = approx_derivative(fun, x, rel_step=diff_step, method=jac,
                                      f0=f, bounds=bounds, args=args,
                                      kwargs=kwargs, sparsity=jac_sparsity)
                # 如果 J 的维度不是二维，则至少将其转换为二维数组
                if J.ndim != 2:  # J is guaranteed not sparse.
                    J = np.atleast_2d(J)

                return J

            # 计算初始点 x0, f0 处的雅可比矩阵 J0
            J0 = jac_wrapped(x0, f0)

    # 如果成功计算出 J0（即 J0 不为 None）
    if J0 is not None:
        # 检查 J0 的形状是否与期望的 (m, n) 一致
        if J0.shape != (m, n):
            raise ValueError(
                f"The return value of `jac` has wrong shape: expected {(m, n)}, "
                f"actual {J0.shape}."
            )

        # 如果 J0 不是 ndarray 类型，则根据 method 和 tr_solver 抛出错误
        if not isinstance(J0, np.ndarray):
            if method == 'lm':
                raise ValueError("method='lm' works only with dense "
                                 "Jacobian matrices.")

            if tr_solver == 'exact':
                raise ValueError(
                    "tr_solver='exact' works only with dense "
                    "Jacobian matrices.")

        # 检查是否需要按照 Jacobi 缩放 x_scale
        jac_scale = isinstance(x_scale, str) and x_scale == 'jac'

        # 如果 J0 是 LinearOperator 类型且要求按照 Jacobi 缩放，则抛出错误
        if isinstance(J0, LinearOperator) and jac_scale:
            raise ValueError("x_scale='jac' can't be used when `jac` "
                             "returns LinearOperator.")

        # 如果未指定 tr_solver，则根据 J0 的类型自动选择 tr_solver
        if tr_solver is None:
            if isinstance(J0, np.ndarray):
                tr_solver = 'exact'
            else:
                tr_solver = 'lsmr'

    # 如果使用 Levenberg-Marquardt 方法，则调用 call_minpack 函数进行优化
    if method == 'lm':
        result = call_minpack(fun_wrapped, x0, jac_wrapped, ftol, xtol, gtol,
                              max_nfev, x_scale, diff_step)
    # 如果优化方法为 'trf'，则调用 trust-region reflective 方法进行优化
    elif method == 'trf':
        # 调用 trf 函数进行优化，并将结果赋给 result
        result = trf(fun_wrapped, jac_wrapped, x0, f0, J0, lb, ub, ftol, xtol,
                     gtol, max_nfev, x_scale, loss_function, tr_solver,
                     tr_options.copy(), verbose)

    # 如果优化方法为 'dogbox'，则调用 dogleg trust-region 方法进行优化
    elif method == 'dogbox':
        # 如果使用 'lsmr' 作为 tr_solver 并且 tr_options 中包含 'regularize' 关键字
        if tr_solver == 'lsmr' and 'regularize' in tr_options:
            # 发出警告，因为 'regularize' 对于 'dogbox' 方法不相关
            warn("The keyword 'regularize' in `tr_options` is not relevant "
                 "for 'dogbox' method.",
                 stacklevel=2)
            # 复制 tr_options，删除 'regularize' 关键字后更新 tr_options
            tr_options = tr_options.copy()
            del tr_options['regularize']

        # 调用 dogbox 函数进行优化，并将结果赋给 result
        result = dogbox(fun_wrapped, jac_wrapped, x0, f0, J0, lb, ub, ftol,
                        xtol, gtol, max_nfev, x_scale, loss_function,
                        tr_solver, tr_options, verbose)

    # 将优化结果的状态码转换为对应的消息
    result.message = TERMINATION_MESSAGES[result.status]
    # 根据状态码判断优化是否成功，并将结果赋给 result.success
    result.success = result.status > 0

    # 如果 verbose 大于等于 1，则输出优化的消息及其它详细信息
    if verbose >= 1:
        print(result.message)
        print("Function evaluations {}, initial cost {:.4e}, final cost "
              "{:.4e}, first-order optimality {:.2e}."
              .format(result.nfev, initial_cost, result.cost,
                      result.optimality))

    # 返回优化结果对象 result
    return result
```