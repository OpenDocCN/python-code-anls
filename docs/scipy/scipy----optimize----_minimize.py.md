# `D:\src\scipysrc\scipy\scipy\optimize\_minimize.py`

```
"""
Unified interfaces to minimization algorithms.

Functions
---------
- minimize : minimization of a function of several variables.
- minimize_scalar : minimization of a function of one variable.
"""

__all__ = ['minimize', 'minimize_scalar']

# 导入警告模块，用于可能的警告信息
from warnings import warn

# 导入 NumPy 库，用于数值计算
import numpy as np

# 导入无约束最小化方法的具体实现
from ._optimize import (_minimize_neldermead, _minimize_powell, _minimize_cg,
                        _minimize_bfgs, _minimize_newtoncg,
                        _minimize_scalar_brent, _minimize_scalar_bounded,
                        _minimize_scalar_golden, MemoizeJac, OptimizeResult,
                        _wrap_callback, _recover_from_bracket_error)

# 导入信赖域方法的具体实现
from ._trustregion_dogleg import _minimize_dogleg
from ._trustregion_ncg import _minimize_trust_ncg
from ._trustregion_krylov import _minimize_trust_krylov
from ._trustregion_exact import _minimize_trustregion_exact
from ._trustregion_constr import _minimize_trustregion_constr

# 导入有约束最小化方法的具体实现
from ._lbfgsb_py import _minimize_lbfgsb
from ._tnc import _minimize_tnc
from ._cobyla_py import _minimize_cobyla
from ._cobyqa_py import _minimize_cobyqa
from ._slsqp_py import _minimize_slsqp

# 导入约束相关的辅助函数和类
from ._constraints import (old_bound_to_new, new_bounds_to_old,
                           old_constraint_to_new, new_constraint_to_old,
                           NonlinearConstraint, LinearConstraint, Bounds,
                           PreparedConstraint)

# 导入支持的有限差分方法
from ._differentiable_functions import FD_METHODS

# 定义支持的无约束最小化方法列表
MINIMIZE_METHODS = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                    'l-bfgs-b', 'tnc', 'cobyla', 'cobyqa', 'slsqp',
                    'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact',
                    'trust-krylov']

# 支持新回调接口的方法列表（传递 OptimizeResult）
MINIMIZE_METHODS_NEW_CB = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                           'l-bfgs-b', 'trust-constr', 'dogleg', 'trust-ncg',
                           'trust-exact', 'trust-krylov', 'cobyqa']

# 定义支持的标量最小化方法列表
MINIMIZE_SCALAR_METHODS = ['brent', 'bounded', 'golden']

def minimize(fun, x0, args=(), method=None, jac=None, hess=None,
             hessp=None, bounds=None, constraints=(), tol=None,
             callback=None, options=None):
    """Minimization of scalar function of one or more variables.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is a 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where ``n`` is the number of independent variables.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` and `hess` functions).

    """
    method : str or callable, optional
        # 定义优化器的类型，可以是字符串或者可调用对象，默认为以下之一：
        # 'Nelder-Mead'：见文档链接
        # 'Powell'：见文档链接
        # 'CG'：见文档链接
        # 'BFGS'：见文档链接
        # 'Newton-CG'：见文档链接
        # 'L-BFGS-B'：见文档链接
        # 'TNC'：见文档链接
        # 'COBYLA'：见文档链接
        # 'COBYQA'：见文档链接
        # 'SLSQP'：见文档链接
        # 'trust-constr'：见文档链接
        # 'dogleg'：见文档链接
        # 'trust-ncg'：见文档链接
        # 'trust-exact'：见文档链接
        # 'trust-krylov'：见文档链接
        # 或者是自定义的可调用对象。详情见下方描述。
        # 如果未提供，则根据问题是否有约束或边界选择 'BFGS'、'L-BFGS-B' 或 'SLSQP' 中的一个。
    jac : {callable,  '2-point', '3-point', 'cs', bool}, optional
        # 计算梯度向量的方法。仅适用于 CG、BFGS、Newton-CG、L-BFGS-B、TNC、SLSQP、dogleg、trust-ncg、trust-krylov、trust-exact 和 trust-constr 方法。
        # 如果是可调用对象，应为返回梯度向量的函数：
        #     ``jac(x, *args) -> array_like, shape (n,)``
        # 其中 ``x`` 是形状为 (n,) 的数组，``args`` 是包含固定参数的元组。如果 `jac` 是布尔值且为 True，则假定 `fun` 返回一个元组 ``(f, g)``，包含目标函数和梯度。
        # 方法 'Newton-CG'、'trust-ncg'、'dogleg'、'trust-exact' 和 'trust-krylov' 要求要么提供一个可调用对象，要么 `fun` 返回目标函数和梯度。
        # 如果为 None 或 False，则使用绝对步长进行二点有限差分估计梯度。
        # 另外，关键字 {'2-point', '3-point', 'cs'} 可用于选择有界限制条件下的数值梯度估计的有限差分方案。
    # hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy}, optional
    #     计算 Hessian 矩阵的方法。仅适用于 Newton-CG、dogleg、trust-ncg、trust-krylov、trust-exact 和 trust-constr 方法。
    #     如果是可调用对象，应返回 Hessian 矩阵：
    #     
    #         ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``
    #     
    #     这里 ``x`` 是一个 (n,) 的 ndarray，``args`` 是一个包含固定参数的元组。
    #     关键字 {'2-point', '3-point', 'cs'} 也可以用来选择数值估计 Hessian 的有限差分方案。
    #     另外，实现了 `HessianUpdateStrategy` 接口的对象也可以用来近似 Hessian。实现该接口的拟牛顿方法有：
    #     
    #         - `BFGS`;
    #         - `SR1`.
    #     
    #     并非所有选项都适用于每种方法；详细信息请参考注释部分。
    hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy}, optional

    # hessp : callable, optional
    #     目标函数的 Hessian 矩阵乘以任意向量 p。仅适用于 Newton-CG、trust-ncg、trust-krylov、trust-constr 方法。
    #     `hessp` 或 `hess` 中只需提供一个。如果提供了 `hess`，则 `hessp` 将被忽略。`hessp` 必须计算 Hessian 乘以任意向量的结果：
    #     
    #         ``hessp(x, p, *args) ->  ndarray shape (n,)``
    #     
    #     这里 ``x`` 是一个 (n,) 的 ndarray，``p`` 是一个维度为 (n,) 的任意向量，``args`` 是一个包含固定参数的元组。
    hessp : callable, optional

    # bounds : sequence or `Bounds`, optional
    #     变量的边界条件，适用于 Nelder-Mead、L-BFGS-B、TNC、SLSQP、Powell、trust-constr、COBYLA 和 COBYQA 方法。
    #     可以通过以下两种方式指定边界条件：
    #     
    #         1. `Bounds` 类的实例。
    #         2. 对 `x` 中每个元素使用 ``(min, max)`` 对形式的序列。使用 `None` 表示无边界条件。
    bounds : sequence or `Bounds`, optional
    # 约束条件定义，可接受多种形式的输入：
    # - 单个约束对象或字典（对于COBYLA, COBYQA, SLSQP和trust-constr方法）
    # - 约束对象列表（对于trust-constr和cobyqa方法）
    # - 字典列表（对于COBYLA和SLSQP方法）
    constraints : {Constraint, dict} or List of {Constraint, dict}, optional

        # 'trust-constr' 和 'cobyqa' 方法的约束定义为单个对象或对象列表，指定优化问题的约束条件。
        # 可用的约束条件包括：
        # - `LinearConstraint`
        # - `NonlinearConstraint`

        # COBYLA 和 SLSQP 方法的约束定义为字典列表。
        # 每个字典包含以下字段：
        # type : str
        #     约束类型: 'eq' 表示等式约束，'ineq' 表示不等式约束。
        # fun : callable
        #     定义约束的函数。
        # jac : callable, optional
        #     `fun` 的雅可比矩阵（仅对于SLSQP方法有效）。
        # args : sequence, optional
        #     传递给函数和雅可比矩阵的额外参数。

        # 等式约束意味着约束函数结果应为零，而不等式约束则要求其非负。
        # 注意，COBYLA 方法仅支持不等式约束。
        Constraints for 'trust-constr' and 'cobyqa' are defined as a single object
        or a list of objects specifying constraints to the optimization problem.
        Available constraints are:

            - `LinearConstraint`
            - `NonlinearConstraint`

        Constraints for COBYLA, SLSQP are defined as a list of dictionaries.
        Each dictionary with fields:

            type : str
                Constraint type: 'eq' for equality, 'ineq' for inequality.
            fun : callable
                The function defining the constraint.
            jac : callable, optional
                The Jacobian of `fun` (only for SLSQP).
            args : sequence, optional
                Extra arguments to be passed to the function and Jacobian.

        Equality constraint means that the constraint function result is to
        be zero whereas inequality means that it is to be non-negative.
        Note that COBYLA only supports inequality constraints.

    # 终止容差，指定优化算法的终止容差。
    tol : float, optional

        # 当指定 `tol` 时，所选的最小化算法会设置一些相关的特定于求解器的容差等于 `tol`。
        # 若要详细控制，请使用特定于求解器的选项。
        Tolerance for termination. When `tol` is specified, the selected
        minimization algorithm sets some relevant solver-specific tolerance(s)
        equal to `tol`. For detailed control, use solver-specific
        options.

    # 求解器选项字典，所有方法除了 `TNC` 方法都支持以下通用选项：
    options : dict, optional

        # 求解器选项的字典形式。除了 `TNC` 方法外，所有方法都支持以下通用选项：
        # maxiter : int
        #     最大迭代次数。每次迭代可能需要多次函数评估。
        #     对于 `TNC` 方法，请使用 `maxfun` 代替 `maxiter`。
        # disp : bool
        #     设为 True 以打印收敛消息。
        #
        # 对于特定于方法的选项，请参阅 :func:`show_options()`。
        A dictionary of solver options. All methods except `TNC` accept the
        following generic options:

            maxiter : int
                Maximum number of iterations to perform. Depending on the
                method each iteration may use several function evaluations.

                For `TNC` use `maxfun` instead of `maxiter`.
            disp : bool
                Set to True to print convergence messages.

        For method-specific options, see :func:`show_options()`.

    # 迭代回调函数，可选。
    callback : callable, optional

        # 每次迭代后调用的可调用对象。
        #
        # 除了 TNC, SLSQP 和 COBYLA 方法支持带有如下签名的可调用对象：
        #
        # ``callback(intermediate_result: OptimizeResult)``
        #
        # 其中 ``intermediate_result`` 是包含参数向量和目标函数当前值的 `OptimizeResult` 的关键参数。
        # 注意，回调函数的参数名必须为 ``intermediate_result`` 才能传递 `OptimizeResult`。
        # 如果回调函数引发 ``StopIteration``，这些方法也将终止。
        #
        # 除了 trust-constr 方法，还支持如下签名：
        #
        # ``callback(xk)``
        #
        # 其中 ``xk`` 是当前参数向量。
        #
        # 使用内省确定要调用哪个以上签名的方法。
        A callable called after each iteration.

        All methods except TNC, SLSQP, and COBYLA support a callable with
        the signature:

            ``callback(intermediate_result: OptimizeResult)``

        where ``intermediate_result`` is a keyword parameter containing an
        `OptimizeResult` with attributes ``x`` and ``fun``, the present values
        of the parameter vector and objective function. Note that the name
        of the parameter must be ``intermediate_result`` for the callback
        to be passed an `OptimizeResult`. These methods will also terminate if
        the callback raises ``StopIteration``.

        All methods except trust-constr (also) support a signature like:

            ``callback(xk)``

        where ``xk`` is the current parameter vector.

        Introspection is used to determine which of the signatures above to
        invoke.

    # 返回值说明
    Returns
    -------
    res : OptimizeResult
        # 定义变量 res，表示优化结果，类型为 OptimizeResult 对象
        The optimization result represented as a ``OptimizeResult`` object.
        # 以 OptimizeResult 对象的形式表示优化结果
        Important attributes are: ``x`` the solution array, ``success`` a
        # 主要属性包括：``x`` 表示解的数组，``success`` 表示优化器是否成功退出
        Boolean flag indicating if the optimizer exited successfully and
        # 布尔标志，指示优化器是否成功退出
        ``message`` which describes the cause of the termination. See
        # ``message`` 描述优化终止的原因。详见
        `OptimizeResult` for a description of other attributes.
        # `OptimizeResult` 获取其他属性的描述信息。

    See also
    --------
    minimize_scalar : Interface to minimization algorithms for scalar
        univariate functions
        # 参见 minimize_scalar：用于标量单变量函数的最小化算法接口
    show_options : Additional options accepted by the solvers
        # show_options：求解器接受的额外选项

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter. The default method is *BFGS*.
    # 此部分描述可通过 'method' 参数选择的可用求解器。默认方法为 *BFGS*。

    **Unconstrained minimization**

    Method :ref:`CG <optimize.minimize-cg>` uses a nonlinear conjugate
    gradient algorithm by Polak and Ribiere, a variant of the
    Fletcher-Reeves method described in [5]_ pp.120-122. Only the
    first derivatives are used.
    # Method :ref:`CG <optimize.minimize-cg>` 使用 Polak 和 Ribiere 的非线性共轭梯度算法，
    # 是 Fletcher-Reeves 方法的变体，详见 [5]_ pp.120-122。仅使用一阶导数。

    Method :ref:`BFGS <optimize.minimize-bfgs>` uses the quasi-Newton
    method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS) [5]_
    pp. 136. It uses the first derivatives only. BFGS has proven good
    performance even for non-smooth optimizations. This method also
    returns an approximation of the Hessian inverse, stored as
    `hess_inv` in the OptimizeResult object.
    # Method :ref:`BFGS <optimize.minimize-bfgs>` 使用 Broyden、Fletcher、Goldfarb 和 Shanno 的拟牛顿方法（BFGS）
    # [5]_ pp. 136。仅使用一阶导数。BFGS 已经证明在非光滑优化中具有良好的性能。此方法还返回 Hessian 逆的近似值，
    # 存储在 OptimizeResult 对象的 `hess_inv` 中。

    Method :ref:`Newton-CG <optimize.minimize-newtoncg>` uses a
    Newton-CG algorithm [5]_ pp. 168 (also known as the truncated
    Newton method). It uses a CG method to the compute the search
    direction. See also *TNC* method for a box-constrained
    minimization with a similar algorithm. Suitable for large-scale
    problems.
    # Method :ref:`Newton-CG <optimize.minimize-newtoncg>` 使用 Newton-CG 算法 [5]_ pp. 168
    # （也称为截断牛顿法）。它使用 CG 方法来计算搜索方向。详见 *TNC* 方法，用于带约束箱的最小化问题，使用类似算法。
    # 适用于大规模问题。

    Method :ref:`dogleg <optimize.minimize-dogleg>` uses the dog-leg
    trust-region algorithm [5]_ for unconstrained minimization. This
    algorithm requires the gradient and Hessian; furthermore the
    Hessian is required to be positive definite.
    # Method :ref:`dogleg <optimize.minimize-dogleg>` 使用狗腿信赖域算法 [5]_ 进行无约束最小化。
    # 此算法需要梯度和 Hessian；此外，Hessian 必须是正定的。

    Method :ref:`trust-ncg <optimize.minimize-trustncg>` uses the
    Newton conjugate gradient trust-region algorithm [5]_ for
    unconstrained minimization. This algorithm requires the gradient
    and either the Hessian or a function that computes the product of
    the Hessian with a given vector. Suitable for large-scale problems.
    # Method :ref:`trust-ncg <optimize.minimize-trustncg>` 使用牛顿共轭梯度信赖域算法 [5]_ 进行无约束最小化。
    # 此算法需要梯度和 Hessian，或者计算给定向量与 Hessian 乘积的函数。适用于大规模问题。

    Method :ref:`trust-krylov <optimize.minimize-trustkrylov>` uses
    the Newton GLTR trust-region algorithm [14]_, [15]_ for unconstrained
    minimization. This algorithm requires the gradient
    and either the Hessian or a function that computes the product of
    the Hessian with a given vector. Suitable for large-scale problems.
    On indefinite problems it requires usually less iterations than the
    `trust-ncg` method and is recommended for medium and large-scale problems.
    # Method :ref:`trust-krylov <optimize.minimize-trustkrylov>` 使用 Newton GLTR 信赖域算法 [14]_, [15]_ 进行无约束最小化。
    # 此算法需要梯度和 Hessian，或者计算给定向量与 Hessian 乘积的函数。适用于大规模问题。
    # 对于不定问题，通常比 `trust-ncg` 方法需要更少的迭代次数，建议用于中等和大规模问题。

    Method :ref:`trust-exact <optimize.minimize-trustexact>`
    is a trust-region method for unconstrained minimization in which
    quadratic subproblems are solved almost exactly [13]_. This
    # Method :ref:`trust-exact <optimize.minimize-trustexact>` 是一种用于无约束最小化的信赖域方法，
    # 几乎完全解决了二次子问题 [13]_。
    # 算法要求梯度和Hessian矩阵（不要求正定）。在许多情况下，牛顿法能够在较少迭代次数内收敛，并且对于小型和中型问题是最推荐的方法。

    # 边界约束最小化

    # 方法 :ref:`Nelder-Mead <optimize.minimize-neldermead>` 使用Simplex算法 [1]_, [2]_。这种算法在许多应用中都表现出很好的鲁棒性。然而，如果可以信任导数的数值计算，可能会更倾向于使用使用一阶和/或二阶导数信息的其他算法，因为它们在一般情况下性能更好。

    # 方法 :ref:`L-BFGS-B <optimize.minimize-lbfgsb>` 使用L-BFGS-B算法 [6]_, [7]_ 进行边界约束最小化。

    # 方法 :ref:`Powell <optimize.minimize-powell>` 是Powell方法的修改 [3]_, [4]_，它是一种共轭方向法。它沿着每个方向集合（`options`和`info`中的`direc`字段）执行顺序一维最小化，该集合在主最小化循环的每次迭代中更新。函数不必可微，也不需要导数。如果没有提供边界，则将使用无边界线搜索。如果提供了边界，并且初始猜测在边界内，则最小化过程中的每次函数评估都将在边界内。如果提供了边界，初始猜测在边界之外，并且`direc`具有完全秩（默认具有完全秩），则在第一次迭代期间可能有些函数评估在边界之外，但在第一次迭代后的每次函数评估都将在边界内。如果`direc`不是完全秩，则某些参数可能无法优化，并且不能保证解在边界内。

    # 方法 :ref:`TNC <optimize.minimize-tnc>` 使用截断牛顿算法 [5]_, [8]_ 来最小化受限变量的函数。该算法使用梯度信息；它也被称为牛顿共轭梯度法。它与上述的*Newton-CG*方法不同，因为它封装了一个C实现，并允许每个变量给定上限和下限。

    # 受限制最小化

    # 方法 :ref:`COBYLA <optimize.minimize-cobyla>` 使用基于线性逼近的受限制优化BY Linear Approximation (COBYLA)方法 [9]_, [10]_, [11]_。该算法基于目标函数和每个约束的线性逼近。该方法封装了该算法的FORTRAN实现。约束函数'fun'可以返回单个数字或数字数组或列表。

    # 方法 :ref:`COBYQA <optimize.minimize-cobyqa>` 使用基于二次逼近的受限制优化BY Quadratic Approximations (COBYQA)方法 [18]_。该算法是一种基于二次逼近的无导数信赖域SQP方法，基于二次
    # 对于这段文档，描述了SciPy中用于优化的两种不同方法：SLSQP和trust-constr。
    # SLSQP方法使用顺序最小二乘规划来最小化多变量函数，支持边界、相等和不等式约束。
    # 这个方法是由Dieter Kraft实现的SLSQP优化子程序的包装器。
    # 特别地，处理边界中的无穷值，将它们转换为大浮点值。

    # trust-constr方法是一种受信任域算法，用于受约束的优化。根据问题定义，它会切换两种实现方式。
    # 对于等式约束问题，它实现了Byrd-Omojokun信赖域SQP方法。对于同时存在不等式约束的问题，
    # 则切换到信赖域内点方法。后者通过引入松弛变量解决不等式约束，并解决一系列逐渐减小的障碍参数值下的等式约束障碍问题。
    # 前述的等式约束SQP方法用于解决随着迭代接近解决方案而逐渐增加准确性的子问题。

    # **有限差分选项**

    # 对于trust-constr方法，可以使用三种有限差分方案来近似梯度和Hessian矩阵：{'2-point', '3-point', 'cs'}。
    # 'cs'方案可能是最准确的，但需要函数能正确处理复杂输入并在复平面上可微分。
    # '3-point'方案比'2-point'更准确，但需要两倍的操作。如果通过有限差分估计梯度，则必须使用一种拟牛顿策略来估计Hessian矩阵。

    # **特定于** `hess` **关键字的方法选项**

    # +--------------+------+----------+-------------------------+-----+
    # | 方法/Hess     | None | callable | '2-point/'3-point'/'cs' | HUS |
    # +==============+======+==========+=========================+=====+
    # | Newton-CG    | x    | (n, n)   | x                       | x   |
    # |              |      | LO       |                         |     |
    # +--------------+------+----------+-------------------------+-----+
    # | dogleg       |      | (n, n)   |                         |     |
    # +--------------+------+----------+-------------------------+-----+
    # trust-ncg    |      | (n, n)   | x                       | x   |
    # trust-krylov |      | (n, n)   | x                       | x   |
    # trust-exact  |      | (n, n)   |                         |     |
    # trust-constr | x    | (n, n)   |  x                      | x   |
    #              |      | LO       |                         |     |
    #              |      | sp       |                         |     |

    # 定义了不同的优化算法和它们的特性，如是否支持 Hessian 矩阵乘法 (n, n)，是否支持约束 (x)，
    # 以及是否依赖 LinearOperator (LO) 或 Sparse matrix (sp) 等。

    # where LO=LinearOperator, sp=Sparse matrix, HUS=HessianUpdateStrategy

    # 自定义最小化器
    # 当使用像 `scipy.optimize.basinhopping` 或其他库的前端方法时，传递自定义的最小化方法可能很有用。
    # 可以简单地将可调用对象作为 `method` 参数传递。

    # 这个可调用对象被调用为 `method(fun, x0, args, **kwargs, **options)`
    # 其中 `kwargs` 对应于传递给 `minimize` 的其他参数（如 `callback`, `hess` 等），
    # 除了 `options` 字典，它的内容也被逐对地传递为 `method` 的参数。

    # 如果 `jac` 被传递为布尔类型，`jac` 和 `fun` 会被修改，使得 `fun` 只返回函数值，`jac` 被转换为返回雅可比矩阵的函数。
    # 方法应返回一个 `OptimizeResult` 对象。

    # 提供的 `method` 可调用对象必须能够接受（并可能忽略）任意参数；`minimize` 可接受的参数集可能在未来版本中扩展，
    # 然后这些参数将被传递给该方法。可以在 scipy.optimize 教程中找到示例。

    # 参考文献
    # ----------
    # .. [1] Nelder, J A, and R Mead. 1965. A Simplex Method for Function
    #        Minimization. The Computer Journal 7: 308-13.
    # .. [2] Wright M H. 1996. Direct search methods: Once scorned, now
    #        respectable, in Numerical Analysis 1995: Proceedings of the 1995
    #        Dundee Biennial Conference in Numerical Analysis (Eds. D F
    #        Griffiths and G A Watson). Addison Wesley Longman, Harlow, UK.
    #        191-208.
    # .. [3] Powell, M J D. 1964. An efficient method for finding the minimum of
    #        a function of several variables without calculating derivatives. The
    #        Computer Journal 7: 155-162.
    # .. [4] Press W, S A Teukolsky, W T Vetterling and B P Flannery.
    #        Numerical Recipes (any edition), Cambridge University Press.
    # .. [5] Nocedal, J, and S J Wright. 2006. Numerical Optimization.
    #        Springer New York.
    # .. [6] Byrd, R H and P Lu and J. Nocedal. 1995. A Limited Memory
    #        Algorithm for Bound Constrained Optimization. SIAM Journal on
    #        Scientific and Statistical Computing 16 (5): 1190-1208.
    # 文献引用和示例开始，涉及优化算法的相关文献和示例应用
    
    # 文献引用 [7]，介绍了 L-BFGS-B 算法，一种用于大规模边界约束优化问题的算法，
    # 原始版本是用 FORTRAN 编写的，由 Zhu, Byrd, 和 Nocedal 在 ACM Transactions on Mathematical Software 上发表。
    
    # 文献引用 [8]，描述了使用 Lanczos 方法的 Newton 类型最小化方法，由 S.G. Nash 在 1984 年在 SIAM Journal of Numerical Analysis 上发表。
    
    # 文献引用 [9]，介绍了 Powell 提出的直接搜索优化方法，该方法通过线性插值模拟目标和约束函数，
    # 1994 年发表在 Advances in Optimization and Numerical Analysis。
    
    # 文献引用 [10]，详细阐述了 Powell 发展的直接搜索算法用于优化计算，发表在 Acta Numerica 1998 年版上。
    
    # 文献引用 [11]，讨论了用于无导数优化算法的多种算法视角，是剑桥大学 DAMTP 2007 年的技术报告。
    
    # 文献引用 [12]，介绍了 Kraft 开发的用于顺序二次规划的软件包，1988 年在德国航空航天中心的技术报告中发布。
    
    # 文献引用 [13]，详细描述了 Conn, Gould 和 Toint 开发的信任域方法，2000 年发表在 SIAM 上，页码为 169-200。
    
    # 文献引用 [14]，描述了 Lenders, Kirches 和 Potschka 实现的 trlib，这是一种用于解决信任域问题的 GLTR 方法的矢量自由实现，
    # 详细内容可参考 arXiv 上的论文号 1611.04718。
    
    # 文献引用 [15]，介绍了 Gould, Lucidi, Roma 和 Toint 开发的使用 Lanczos 方法求解信任域子问题的方法，
    # 发表在 SIAM J. Optim. 1999 年的第 9 卷第 2 期，页码为 504-525。
    
    # 文献引用 [16]，描述了 Byrd, Hribar 和 Nocedal 开发的大规模非线性规划的内点算法，
    # 发表在 SIAM Journal on Optimization 1999 年的第 9 卷第 4 期，页码为 877-900。
    
    # 文献引用 [17]，详细讨论了 Lalee, Nocedal 和 Plantega 实现的用于大规模等式约束优化的算法，
    # 发表在 SIAM Journal on Optimization 1998 年的第 8 卷第 3 期，页码为 682-706。
    
    # 文献引用 [18]，介绍了 Ragonneau 的博士论文，该论文研究基于模型的无导数优化方法及其软件的开发，
    # 2022 年在香港理工大学应用数学系完成，论文链接：https://theses.lib.polyu.edu.hk/handle/200/12294。
    
    # 示例开始，展示了如何使用 SciPy 中的优化模块最小化 Rosenbrock 函数的问题。
    
    # 导入必要的模块和函数
    from scipy.optimize import minimize, rosen, rosen_der
    
    # 初始化起始点
    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    
    # 使用 Nelder-Mead 方法进行简单的 Rosenbrock 函数最小化
    res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)
    
    # 输出最优解的 x 值
    res.x
    
    # 继续使用 BFGS 算法，指定了使用 Rosenbrock 函数的一阶导数和一些选项进行优化
    res = minimize(rosen, x0, method='BFGS', jac=rosen_der,
    x0 = np.atleast_1d(np.asarray(x0))

将输入的初始点 `x0` 转换为至少是一维的 numpy 数组。


    if x0.ndim != 1:
        raise ValueError("'x0' must only have one dimension.")

如果 `x0` 不是一维数组，则引发值错误，要求 `x0` 必须是一维的。


    if x0.dtype.kind in np.typecodes["AllInteger"]:
        x0 = np.asarray(x0, dtype=float)

如果 `x0` 的数据类型是任何整数类型，则将其转换为浮点数类型的 numpy 数组。


    if not isinstance(args, tuple):
        args = (args,)

如果 `args` 不是元组类型，则将其转换为包含一个元素的元组。


    if method is None:
        # Select automatically
        if constraints:
            method = 'SLSQP'
        elif bounds is not None:
            method = 'L-BFGS-B'
        else:
            method = 'BFGS'

如果未指定优化方法 `method`，根据情况自动选择：
- 如果存在约束条件 `constraints`，则选择 'SLSQP' 方法。
- 如果存在变量范围 `bounds`，则选择 'L-BFGS-B' 方法。
- 否则，默认选择 'BFGS' 方法。


    if callable(method):
        meth = "_custom"
    else:
        meth = method.lower()

如果 `method` 是可调用对象，则将 `meth` 设为 "_custom"；否则，将 `method` 的小写形式赋给 `meth`。


    if options is None:
        options = {}

如果未提供选项 `options`，则将其设为一个空字典。


    # check if optional parameters are supported by the selected method
    # - jac
    if meth in ('nelder-mead', 'powell', 'cobyla', 'cobyqa') and bool(jac):
        warn('Method %s does not use gradient information (jac).' % method,
             RuntimeWarning, stacklevel=2)

检查所选方法 `meth` 是否支持可选参数：
- `jac` 是指定的梯度函数。如果所选方法不使用梯度信息，并且 `jac` 为真，则发出警告。


    # - hess
    if meth not in ('newton-cg', 'dogleg', 'trust-ncg', 'trust-constr',
                    'trust-krylov', 'trust-exact', '_custom') and hess is not None:
        warn('Method %s does not use Hessian information (hess).' % method,
             RuntimeWarning, stacklevel=2)

检查所选方法 `meth` 是否支持 Hessian 矩阵信息：
- 如果所选方法不使用 Hessian 信息，并且提供了 `hess` 参数，则发出相应的警告。


    # - hessp

这一行注释是一个占位符，可能预留用于检查其他可选参数。
    # 如果方法不在指定的方法列表中，并且 hessp 参数不为 None，则发出警告
    if meth not in ('newton-cg', 'trust-ncg', 'trust-constr',
                    'trust-krylov', '_custom') \
       and hessp is not None:
        warn('Method %s does not use Hessian-vector product '
             'information (hessp).' % method,
             RuntimeWarning, stacklevel=2)

    # 如果方法不在指定的方法列表中，并且 constraints 中有任何约束存在，则发出警告
    if (meth not in ('cobyla', 'cobyqa', 'slsqp', 'trust-constr', '_custom') and
            np.any(constraints)):
        warn('Method %s cannot handle constraints.' % method,
             RuntimeWarning, stacklevel=2)

    # 如果方法不在指定的方法列表中，并且 bounds 参数不为 None，则发出警告
    if meth not in (
            'nelder-mead', 'powell', 'l-bfgs-b', 'cobyla', 'cobyqa', 'slsqp',
            'tnc', 'trust-constr', '_custom') and bounds is not None:
        warn('Method %s cannot handle bounds.' % method,
             RuntimeWarning, stacklevel=2)

    # 如果方法在指定的方法列表中，并且 options 中设置了 return_all 为 True，则发出警告
    if (meth in ('l-bfgs-b', 'tnc', 'cobyla', 'cobyqa', 'slsqp') and
            options.get('return_all', False)):
        warn('Method %s does not support the return_all option.' % method,
             RuntimeWarning, stacklevel=2)

    # 检查 jac 参数的设置
    if callable(jac):
        pass
    elif jac is True:
        # 如果 jac 参数为 True，则使用 MemoizeJac 类来缓存函数的导数
        fun = MemoizeJac(fun)
        jac = fun.derivative
    elif (jac in FD_METHODS and
          meth in ['trust-constr', 'bfgs', 'cg', 'l-bfgs-b', 'tnc', 'slsqp']):
        # 如果 jac 参数在有限差分方法列表中，并且方法在支持有限差分的方法中，则保持不变
        pass
    elif meth in ['trust-constr']:
        # 如果方法是 'trust-constr'，则使用默认的 jac 计算方式 '2-point'
        jac = '2-point'
    elif jac is None or bool(jac) is False:
        # 如果 jac 参数为 None 或者布尔值为 False，则将 jac 设置为 None
        jac = None
    else:
        # 对于未知的 jac 参数设置，默认将 jac 设置为 None
        jac = None

    # 设置默认的容差值
    if tol is not None:
        options = dict(options)
        if meth == 'nelder-mead':
            # 如果方法是 'nelder-mead'，则设置 xatol 和 fatol 的容差值为 tol
            options.setdefault('xatol', tol)
            options.setdefault('fatol', tol)
        if meth in ('newton-cg', 'powell', 'tnc'):
            # 如果方法是 'newton-cg', 'powell', 'tnc' 中的一种，则设置 xtol 的容差值为 tol
            options.setdefault('xtol', tol)
        if meth in ('powell', 'l-bfgs-b', 'tnc', 'slsqp'):
            # 如果方法是 'powell', 'l-bfgs-b', 'tnc', 'slsqp' 中的一种，则设置 ftol 的容差值为 tol
            options.setdefault('ftol', tol)
        if meth in ('bfgs', 'cg', 'l-bfgs-b', 'tnc', 'dogleg',
                    'trust-ncg', 'trust-exact', 'trust-krylov'):
            # 如果方法是支持 gtol 容差值设置的方法，则设置 gtol 的容差值为 tol
            options.setdefault('gtol', tol)
        if meth in ('cobyla', '_custom'):
            # 如果方法是 'cobyla' 或者 '_custom'，则设置 tol 的容差值为 tol
            options.setdefault('tol', tol)
        if meth == 'cobyqa':
            # 如果方法是 'cobyqa'，则设置 final_tr_radius 的容差值为 tol
            options.setdefault('final_tr_radius', tol)
        if meth == 'trust-constr':
            # 如果方法是 'trust-constr'，则设置 xtol、gtol 和 barrier_tol 的容差值为 tol
            options.setdefault('xtol', tol)
            options.setdefault('gtol', tol)
            options.setdefault('barrier_tol', tol)
    # 如果方法是 '_custom'，则执行以下代码块
    if meth == '_custom':
        # 在边界和约束被标准化之前调用自定义方法
        # 自定义方法应该能够接受任何提供给它的边界和约束
        # 返回自定义方法的执行结果
        return method(fun, x0, args=args, jac=jac, hess=hess, hessp=hessp,
                      bounds=bounds, constraints=constraints,
                      callback=callback, **options)

    # 使用标准化函数对约束条件进行标准化处理
    constraints = standardize_constraints(constraints, x0, meth)

    # 设置一个标志，表示未来是否需要删除变量
    remove_vars = False
    if bounds is not None:
        # 转换为新样式的边界，这样我们只需要考虑一种情况
        bounds = standardize_bounds(bounds, x0, 'new')
        # 验证边界，并根据优化方法进一步处理
        bounds = _validate_bounds(bounds, x0, meth)

        if meth in {"tnc", "slsqp", "l-bfgs-b"}:
            # 这些方法在某些变量被边界固定时无法使用需要的有限差分导数。为避免这个问题，
            # 从问题中移除被固定的变量。
            # 注意: 如果增加了方法，请确保更新相关的测试和 test_optimize.eb_data。还要考虑
            # 是否需要更新默认的 OptimizeResult。

            # 确定是否有任何变量被固定
            i_fixed = (bounds.lb == bounds.ub)

            if np.all(i_fixed):
                # 所有参数都被固定，优化器无法进行任何操作
                return _optimize_result_for_equal_bounds(
                    fun, bounds, meth, args=args, constraints=constraints
                )

            # 确定是否需要使用有限差分来计算梯度/jacobian
            fd_needed = (not callable(jac))
            for con in constraints:
                if not callable(con.get('jac', None)):
                    fd_needed = True

            # 如果有需要使用有限差分，移除所有被固定的变量
            # 对于 TNC 方法始终移除固定变量；参见 gh-14565
            remove_vars = i_fixed.any() and (fd_needed or meth == "tnc")
            if remove_vars:
                x_fixed = (bounds.lb)[i_fixed]
                x0 = x0[~i_fixed]
                bounds = _remove_from_bounds(bounds, i_fixed)
                fun = _remove_from_func(fun, i_fixed, x_fixed)
                if callable(callback):
                    callback = _remove_from_func(callback, i_fixed, x_fixed)
                if callable(jac):
                    jac = _remove_from_func(jac, i_fixed, x_fixed, remove=1)

                # 深拷贝约束，以防用户的版本被改变（浅拷贝即可）
                constraints = [con.copy() for con in constraints]
                for con in constraints:
                    con['fun'] = _remove_from_func(con['fun'], i_fixed,
                                                   x_fixed, min_dim=1,
                                                   remove=0)
                    if callable(con.get('jac', None)):
                        con['jac'] = _remove_from_func(con['jac'], i_fixed,
                                                       x_fixed, min_dim=2,
                                                       remove=1)
        # 标准化边界以确保与当前方法的要求兼容
        bounds = standardize_bounds(bounds, x0, meth)

    # 包装回调函数以符合特定优化方法的要求
    callback = _wrap_callback(callback, meth)

    if meth == 'nelder-mead':
        # 对于 Nelder-Mead 方法的优化
        res = _minimize_neldermead(fun, x0, args, callback, bounds=bounds,
                                   **options)
    # 如果优化方法为 'powell'，则调用 Powell 方法进行最小化优化
    elif meth == 'powell':
        res = _minimize_powell(fun, x0, args, callback, bounds, **options)
    
    # 如果优化方法为 'cg'，则调用 Conjugate Gradient 方法进行最小化优化
    elif meth == 'cg':
        res = _minimize_cg(fun, x0, args, jac, callback, **options)
    
    # 如果优化方法为 'bfgs'，则调用 BFGS 方法进行最小化优化
    elif meth == 'bfgs':
        res = _minimize_bfgs(fun, x0, args, jac, callback, **options)
    
    # 如果优化方法为 'newton-cg'，则调用 Newton-CG 方法进行最小化优化
    elif meth == 'newton-cg':
        res = _minimize_newtoncg(fun, x0, args, jac, hess, hessp, callback,
                                 **options)
    
    # 如果优化方法为 'l-bfgs-b'，则调用 L-BFGS-B 方法进行最小化优化
    elif meth == 'l-bfgs-b':
        res = _minimize_lbfgsb(fun, x0, args, jac, bounds,
                               callback=callback, **options)
    
    # 如果优化方法为 'tnc'，则调用 TNC 方法进行最小化优化
    elif meth == 'tnc':
        res = _minimize_tnc(fun, x0, args, jac, bounds, callback=callback,
                            **options)
    
    # 如果优化方法为 'cobyla'，则调用 COBYLA 方法进行最小化优化
    elif meth == 'cobyla':
        res = _minimize_cobyla(fun, x0, args, constraints, callback=callback,
                               bounds=bounds, **options)
    
    # 如果优化方法为 'cobyqa'，则调用 COBYQA 方法进行最小化优化
    elif meth == 'cobyqa':
        res = _minimize_cobyqa(fun, x0, args, bounds, constraints, callback,
                               **options)
    
    # 如果优化方法为 'slsqp'，则调用 SLSQP 方法进行最小化优化
    elif meth == 'slsqp':
        res = _minimize_slsqp(fun, x0, args, jac, bounds,
                              constraints, callback=callback, **options)
    
    # 如果优化方法为 'trust-constr'，则调用 Trust Region Constrained 方法进行最小化优化
    elif meth == 'trust-constr':
        res = _minimize_trustregion_constr(fun, x0, args, jac, hess, hessp,
                                           bounds, constraints,
                                           callback=callback, **options)
    
    # 如果优化方法为 'dogleg'，则调用 Dogleg 方法进行最小化优化
    elif meth == 'dogleg':
        res = _minimize_dogleg(fun, x0, args, jac, hess,
                               callback=callback, **options)
    
    # 如果优化方法为 'trust-ncg'，则调用 Trust Newton-CG 方法进行最小化优化
    elif meth == 'trust-ncg':
        res = _minimize_trust_ncg(fun, x0, args, jac, hess, hessp,
                                  callback=callback, **options)
    
    # 如果优化方法为 'trust-krylov'，则调用 Trust Krylov 方法进行最小化优化
    elif meth == 'trust-krylov':
        res = _minimize_trust_krylov(fun, x0, args, jac, hess, hessp,
                                     callback=callback, **options)
    
    # 如果优化方法为 'trust-exact'，则调用 Trust Region Exact 方法进行最小化优化
    elif meth == 'trust-exact':
        res = _minimize_trustregion_exact(fun, x0, args, jac, hess,
                                          callback=callback, **options)
    
    # 如果未知优化方法，则抛出 ValueError 异常
    else:
        raise ValueError('Unknown solver %s' % method)

    # 如果 remove_vars 为真，则进行后处理
    if remove_vars:
        # 将固定的变量添加到结果中
        res.x = _add_to_array(res.x, i_fixed, x_fixed)
        # 将固定的变量的雅可比矩阵值设置为 NaN
        res.jac = _add_to_array(res.jac, i_fixed, np.nan)
        # 如果结果中包含 "hess_inv" 属性，则将其设置为 None，表示未知
        if "hess_inv" in res:
            res.hess_inv = None  # unknown

    # 如果回调函数的 'stop_iteration' 属性为真，则设置部分结果属性以指示异常情况
    if getattr(callback, 'stop_iteration', False):
        res.success = False
        res.status = 99
        res.message = "`callback` raised `StopIteration`."

    # 返回优化结果
    return res
# 定义函数 `minimize_scalar`，用于局部最小化单变量标量函数。

def minimize_scalar(fun, bracket=None, bounds=None, args=(),
                    method=None, tol=None, options=None):
    """Local minimization of scalar function of one variable.
    单变量标量函数的局部最小化。

    Parameters
    ----------
    fun : callable
        Objective function.
        目标函数，必须返回一个标量。

    bracket : sequence, optional
        For methods 'brent' and 'golden', `bracket` defines the bracketing
        interval and is required.
        对于方法 'brent' 和 'golden'，`bracket` 定义了区间并且是必需的。
        可以是三元组 ``(xa, xb, xc)``，满足条件 ``xa < xb < xc`` 和
        ``func(xb) < func(xa) and  func(xb) < func(xc)``，
        或者是一对 ``(xa, xb)`` 作为下坡区间搜索的初始点
        (参见 `scipy.optimize.bracket`)。
        结果的最小值 `res.x` 不一定满足 ``xa <= res.x <= xb``。

    bounds : sequence, optional
        For method 'bounded', `bounds` is mandatory and must have two finite
        items corresponding to the optimization bounds.
        对于方法 'bounded'，`bounds` 是必须的，必须有两个有限项，
        对应于优化的边界。

    args : tuple, optional
        Extra arguments passed to the objective function.
        传递给目标函数的额外参数。

    method : str or callable, optional
        Type of solver.  Should be one of:
        解决器类型。应为以下之一：

            - :ref:`Brent <optimize.minimize_scalar-brent>`
            - :ref:`Bounded <optimize.minimize_scalar-bounded>`
            - :ref:`Golden <optimize.minimize_scalar-golden>`
            - custom - a callable object (added in version 0.14.0), see below
            自定义 - 可调用对象（自版本 0.14.0 添加），见下文。

        Default is "Bounded" if bounds are provided and "Brent" otherwise.
        如果提供了 `bounds`，默认为 "Bounded"；否则默认为 "Brent"。
        有关每个解算器的详细信息，请参阅“Notes”部分。

    tol : float, optional
        Tolerance for termination. For detailed control, use solver-specific
        options.
        终止的容差。如需详细控制，请使用特定于解算器的选项。

    options : dict, optional
        A dictionary of solver options.
        解算器选项的字典。

            maxiter : int
                Maximum number of iterations to perform.
                执行的最大迭代次数。
            disp : bool
                Set to True to print convergence messages.
                设置为 True 以打印收敛消息。

        See :func:`show_options()` for solver-specific options.
        有关特定于解算器的选项，请参阅 `show_options()`。

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        优化结果表示为 ``OptimizeResult`` 对象。
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.
        重要属性包括：``x`` 解数组，``success`` 一个布尔标志，指示优化器是否成功退出，
        以及 ``message`` 描述终止原因。有关其他属性的描述，请参阅 `OptimizeResult`。

    See also
    --------
    minimize : Interface to minimization algorithms for scalar multivariate
        functions
    show_options : Additional options accepted by the solvers
    另请参阅：
    minimize：标量多元函数的最小化算法接口
    show_options：解算器接受的额外选项

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter. The default method is the ``"Bounded"`` Brent method if
    `bounds` are passed and unbounded ``"Brent"`` otherwise.
    本节描述了可以通过 'method' 参数选择的可用解算器。
    默认方法是如果传递了 `bounds` 则为 ``"Bounded"`` Brent 方法，
    否则为无界 ``"Brent"`` 方法。

    Method :ref:`Brent <optimize.minimize_scalar-brent>` uses Brent's
    algorithm [1]_ to find a local minimum.  The algorithm uses inverse
    parabolic interpolation when possible to speed up convergence of
    使用 Brent 的算法 [1]_ 寻找局部最小值。
    算法在可能时使用反向抛物线插值来加速收敛。
    """
    # 如果参数 `args` 不是元组，则将其转换为元组
    if not isinstance(args, tuple):
        args = (args,)

    # 如果 `method` 是可调用对象，则选择自定义方法 "_custom"
    if callable(method):
        meth = "_custom"
    # 如果方法参数为 None，则根据 bounds 是否为 None 来选择默认的优化方法
    elif method is None:
        meth = 'brent' if bounds is None else 'bounded'
    else:
        # 将方法名转换为小写
        meth = method.lower()
    
    # 如果选项参数为 None，则设为空字典
    if options is None:
        options = {}

    # 如果 bounds 参数不为 None，并且方法为 'brent' 或 'golden'，则抛出异常
    if bounds is not None and meth in {'brent', 'golden'}:
        message = f"Use of `bounds` is incompatible with 'method={method}'."
        raise ValueError(message)

    # 如果设定了 tol 参数
    if tol is not None:
        # 将 options 转换为字典类型
        options = dict(options)
        # 如果方法为 'bounded' 并且 'xatol' 不在 options 中，则发出警告并设定 'xatol' 参数
        if meth == 'bounded' and 'xatol' not in options:
            warn("Method 'bounded' does not support relative tolerance in x; "
                 "defaulting to absolute tolerance.",
                 RuntimeWarning, stacklevel=2)
            options['xatol'] = tol
        # 如果方法为 '_custom'，则设定 'tol' 参数为 tol 的默认值
        elif meth == '_custom':
            options.setdefault('tol', tol)
        else:
            # 对于其他方法，默认设定 'xtol' 参数为 tol 的默认值
            options.setdefault('xtol', tol)

    # 如果 options 中指定了布尔类型的 disp 参数，则替换为对应的整数值
    disp = options.get('disp')
    if isinstance(disp, bool):
        options['disp'] = 2 * int(disp)

    # 根据不同的优化方法调用相应的函数进行标量最小化计算
    if meth == '_custom':
        res = method(fun, args=args, bracket=bracket, bounds=bounds, **options)
    elif meth == 'brent':
        res = _recover_from_bracket_error(_minimize_scalar_brent,
                                          fun, bracket, args, **options)
    elif meth == 'bounded':
        # 如果没有指定 bounds 参数，则抛出异常
        if bounds is None:
            raise ValueError('The `bounds` parameter is mandatory for '
                             'method `bounded`.')
        # 使用指定的边界 bounds 进行标量最小化计算
        res = _minimize_scalar_bounded(fun, bounds, args, **options)
    elif meth == 'golden':
        res = _recover_from_bracket_error(_minimize_scalar_golden,
                                          fun, bracket, args, **options)
    else:
        # 如果方法名未知，则抛出异常
        raise ValueError('Unknown solver %s' % method)

    # 修复 gh-16196 报告的问题，确保 `res.x` 的输出形状与 `res.fun` 一致
    res.fun = np.asarray(res.fun)[()]
    res.x = np.reshape(res.x, res.fun.shape)[()]
    return res
# 从 `Bounds` 实例中移除固定的变量
def _remove_from_bounds(bounds, i_fixed):
    # 从下界数组中移除固定变量
    lb = bounds.lb[~i_fixed]
    # 从上界数组中移除固定变量
    ub = bounds.ub[~i_fixed]
    # 返回一个新的 `Bounds` 对象，不改变原始的 `Bounds` 对象
    return Bounds(lb, ub)


# 包装一个函数，使得固定变量不需要在调用时传入
def _remove_from_func(fun_in, i_fixed, x_fixed, min_dim=None, remove=0):
    def fun_out(x_in, *args, **kwargs):
        # 创建一个与 `i_fixed` 大小相同的零数组，与输入类型一致
        x_out = np.zeros_like(i_fixed, dtype=x_in.dtype)
        # 将固定的变量复制到输出数组中
        x_out[i_fixed] = x_fixed
        # 将未固定的变量复制到输出数组中
        x_out[~i_fixed] = x_in
        # 调用输入函数 `fun_in`，计算输出结果
        y_out = fun_in(x_out, *args, **kwargs)
        # 将输出结果转换为数组
        y_out = np.array(y_out)

        # 根据 `min_dim` 的值，至少将输出结果转换为一维数组
        if min_dim == 1:
            y_out = np.atleast_1d(y_out)
        # 根据 `min_dim` 的值，至少将输出结果转换为二维数组
        elif min_dim == 2:
            y_out = np.atleast_2d(y_out)

        # 根据 `remove` 的值，可能从输出结果中移除固定的变量
        if remove == 1:
            y_out = y_out[..., ~i_fixed]
        elif remove == 2:
            y_out = y_out[~i_fixed, ~i_fixed]

        # 返回处理后的输出结果
        return y_out
    return fun_out


# 将固定的变量添加回数组中
def _add_to_array(x_in, i_fixed, x_fixed):
    # 计算未固定的变量索引
    i_free = ~i_fixed
    # 如果输入数组是二维的，则更新未固定的变量索引
    if x_in.ndim == 2:
        i_free = i_free[:, None] @ i_free[None, :]
    # 创建一个与输入数组类型相同的零数组
    x_out = np.zeros_like(i_free, dtype=x_in.dtype)
    # 将固定的变量添加回输出数组中
    x_out[~i_free] = x_fixed
    # 将未固定的变量添加回输出数组中
    x_out[i_free] = x_in.ravel()
    # 返回处理后的输出数组
    return x_out


# 验证边界是否有效
def _validate_bounds(bounds, x0, meth):
    # 如果任意上界小于对应的下界，则抛出值错误异常
    msg = "An upper bound is less than the corresponding lower bound."
    if np.any(bounds.ub < bounds.lb):
        raise ValueError(msg)

    # 如果边界数量与 `x0` 的长度不兼容，则抛出值错误异常
    msg = "The number of bounds is not compatible with the length of `x0`."
    try:
        # 将下界广播到与 `x0` 形状相同
        bounds.lb = np.broadcast_to(bounds.lb, x0.shape)
        # 将上界广播到与 `x0` 形状相同
        bounds.ub = np.broadcast_to(bounds.ub, x0.shape)
    except Exception as e:
        raise ValueError(msg) from e

    # 返回验证后的边界对象
    return bounds


# 将边界转换为求解器所需的形式
def standardize_bounds(bounds, x0, meth):
    # 如果 `meth` 在指定的方法集合中
    if meth in {'trust-constr', 'powell', 'nelder-mead', 'cobyla', 'cobyqa', 'new'}:
        # 如果 `bounds` 不是 `Bounds` 实例，则将其转换为新的边界形式
        if not isinstance(bounds, Bounds):
            lb, ub = old_bound_to_new(bounds)
            bounds = Bounds(lb, ub)
    # 如果 `meth` 在另一个方法集合中
    elif meth in ('l-bfgs-b', 'tnc', 'slsqp', 'old'):
        # 如果 `bounds` 是 `Bounds` 实例，则将其转换为旧的边界形式
        if isinstance(bounds, Bounds):
            bounds = new_bounds_to_old(bounds.lb, bounds.ub, x0.shape[0])
    # 返回标准化后的边界对象
    return bounds


# 将约束条件转换为求解器所需的形式
def standardize_constraints(constraints, x0, meth):
    # 所有的约束条件类型
    all_constraint_types = (NonlinearConstraint, LinearConstraint, dict)
    # 新的约束条件类型，排除最后一个类型
    new_constraint_types = all_constraint_types[:-1]
    
    # 如果约束条件为空，则将其设为空列表
    if constraints is None:
        constraints = []
    # 如果约束条件是其中的一种类型，则转换为列表形式
    elif isinstance(constraints, all_constraint_types):
        constraints = [constraints]
    else:
        # 确保约束条件是可变序列
        constraints = list(constraints)

    # 如果 `meth` 在指定的方法集合中
    if meth in ['trust-constr', 'cobyqa', 'new']:
        # 遍历约束条件列表，将非新约束类型的约束条件转换为新的约束类型
        for i, con in enumerate(constraints):
            if not isinstance(con, new_constraint_types):
                constraints[i] = old_constraint_to_new(i, con)

# 结束函数定义
    else:
        # 遍历复制的约束条件列表，修改原始列表
        for i, con in enumerate(list(constraints)):
            # 如果约束条件属于新的约束类型之一
            if isinstance(con, new_constraint_types):
                # 将新约束转换为旧约束
                old_constraints = new_constraint_to_old(con, x0)
                # 替换原始列表中的新约束为旧约束的第一个元素
                constraints[i] = old_constraints[0]
                # 将剩余的旧约束添加到原始约束列表末尾（如果有多个）
                constraints.extend(old_constraints[1:])  # 如果有多个旧约束，则逐一添加到列表中

    # 返回修改后的约束列表
    return constraints
def _optimize_result_for_equal_bounds(
        fun, bounds, method, args=(), constraints=()
):
    """
    Provides a default OptimizeResult for when a bounded minimization method
    has (lb == ub).all().

    Parameters
    ----------
    fun: callable
        目标函数，用于计算给定参数的目标值。
    bounds: Bounds
        包含最小和最大边界的对象，用于约束参数。
    method: str
        使用的优化方法的名称或标识符。
    args: tuple, optional
        传递给目标函数 `fun` 的额外参数。
    constraints: Constraint
        用于描述参数约束条件的对象。
    """
    success = True
    message = 'All independent variables were fixed by bounds.'
    
    # bounds is new-style
    x0 = bounds.lb  # 获取边界对象的下限作为初始参数
    
    if constraints:
        message = ("All independent variables were fixed by bounds at values"
                   " that satisfy the constraints.")
        constraints = standardize_constraints(constraints, x0, 'new')
    
    maxcv = 0
    for c in constraints:
        pc = PreparedConstraint(c, x0)
        violation = pc.violation(x0)
        if np.sum(violation):
            maxcv = max(maxcv, np.max(violation))
            success = False
            message = (f"All independent variables were fixed by bounds, but "
                       f"the independent variables do not satisfy the "
                       f"constraints exactly. (Maximum violation: {maxcv}).")
    
    return OptimizeResult(
        x=x0,  # 最优化参数的值
        fun=fun(x0, *args),  # 计算目标函数在最优化参数下的值
        success=success,  # 优化是否成功的标志
        message=message,  # 关于优化结果的描述性消息
        nfev=1,  # 函数评估的次数
        njev=0,  # 雅可比矩阵评估的次数
        nhev=0,  # 黑塞矩阵评估的次数
    )
```