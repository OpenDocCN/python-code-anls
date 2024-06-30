# `D:\src\scipysrc\scipy\scipy\optimize\_qap.py`

```
import numpy as np  # 导入NumPy库，用于处理数组和矩阵的操作
import operator  # 导入operator模块，用于支持一些操作符的函数形式
from . import (linear_sum_assignment, OptimizeResult)  # 从当前包中导入linear_sum_assignment和OptimizeResult类
from ._optimize import _check_unknown_options  # 从当前包的_optimize模块导入_check_unknown_options函数

from scipy._lib._util import check_random_state  # 从SciPy的内部工具库中导入check_random_state函数
import itertools  # 导入itertools模块，用于高效的迭代工具

QUADRATIC_ASSIGNMENT_METHODS = ['faq', '2opt']  # 定义全局变量QUADRATIC_ASSIGNMENT_METHODS，包含两种方法名：'faq'和'2opt'

def quadratic_assignment(A, B, method="faq", options=None):
    r"""
    Approximates solution to the quadratic assignment problem and
    the graph matching problem.

    Quadratic assignment solves problems of the following form:

    .. math::

        \min_P & \ {\ \text{trace}(A^T P B P^T)}\\
        \mbox{s.t. } & {P \ \epsilon \ \mathcal{P}}\\

    where :math:`\mathcal{P}` is the set of all permutation matrices,
    and :math:`A` and :math:`B` are square matrices.

    Graph matching tries to *maximize* the same objective function.
    This algorithm can be thought of as finding the alignment of the
    nodes of two graphs that minimizes the number of induced edge
    disagreements, or, in the case of weighted graphs, the sum of squared
    edge weight differences.

    Note that the quadratic assignment problem is NP-hard. The results given
    here are approximations and are not guaranteed to be optimal.


    Parameters
    ----------
    A : 2-D array, square
        The square matrix :math:`A` in the objective function above.

    B : 2-D array, square
        The square matrix :math:`B` in the objective function above.

    method :  str in {'faq', '2opt'} (default: 'faq')
        The algorithm used to solve the problem.
        :ref:`'faq' <optimize.qap-faq>` (default) and
        :ref:`'2opt' <optimize.qap-2opt>` are available.

    options : dict, optional
        A dictionary of solver options. All solvers support the following:

        maximize : bool (default: False)
            Maximizes the objective function if ``True``.

        partial_match : 2-D array of integers, optional (default: None)
            Fixes part of the matching. Also known as a "seed" [2]_.

            Each row of `partial_match` specifies a pair of matched nodes:
            node ``partial_match[i, 0]`` of `A` is matched to node
            ``partial_match[i, 1]`` of `B`. The array has shape ``(m, 2)``,
            where ``m`` is not greater than the number of nodes, :math:`n`.

        rng : {None, int, `numpy.random.Generator`,
               `numpy.random.RandomState`}, optional

            If `seed` is None (or `np.random`), the `numpy.random.RandomState`
            singleton is used.
            If `seed` is an int, a new ``RandomState`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance then
            that instance is used.

        For method-specific options, see
        :func:`show_options('quadratic_assignment') <show_options>`.

    Returns
    -------
    res : OptimizeResult
        # 定义一个变量 res，类型为 OptimizeResult，表示优化结果对象，包含以下字段。

        col_ind : 1-D array
            # 一维数组，表示找到的节点 B 的最佳排列对应的列索引。
            Column indices corresponding to the best permutation found of the
            nodes of `B`.
        
        fun : float
            # 浮点数，表示解的目标函数值。
            The objective value of the solution.
        
        nit : int
            # 整数，表示优化过程中执行的迭代次数。
            The number of iterations performed during optimization.

    Notes
    -----
    # 说明部分开始
    The default method :ref:`'faq' <optimize.qap-faq>` uses the Fast
    Approximate QAP algorithm [1]_; it typically offers the best combination of
    speed and accuracy.
    # 默认方法使用快速近似 QAP 算法，通常提供速度和精度的最佳组合。

    Method :ref:`'2opt' <optimize.qap-2opt>` can be computationally expensive,
    but may be a useful alternative, or it can be used to refine the solution
    returned by another method.
    # 方法 '2opt' 可能计算成本高昂，但可以作为一个有用的替代方法，或者用于改进另一种方法返回的解。

    References
    ----------
    .. [1] J.T. Vogelstein, J.M. Conroy, V. Lyzinski, L.J. Podrazik,
           S.G. Kratzer, E.T. Harley, D.E. Fishkind, R.J. Vogelstein, and
           C.E. Priebe, "Fast approximate quadratic programming for graph
           matching," PLOS one, vol. 10, no. 4, p. e0121002, 2015,
           :doi:`10.1371/journal.pone.0121002`

    .. [2] D. Fishkind, S. Adali, H. Patsolic, L. Meng, D. Singh, V. Lyzinski,
           C. Priebe, "Seeded graph matching", Pattern Recognit. 87 (2019):
           203-215, :doi:`10.1016/j.patcog.2018.09.014`

    .. [3] "2-opt," Wikipedia.
           https://en.wikipedia.org/wiki/2-opt
    # 引用部分结束

    Examples
    --------
    # 示例部分开始
    >>> import numpy as np
    >>> from scipy.optimize import quadratic_assignment
    >>> A = np.array([[0, 80, 150, 170], [80, 0, 130, 100],
    ...               [150, 130, 0, 120], [170, 100, 120, 0]])
    >>> B = np.array([[0, 5, 2, 7], [0, 0, 3, 8],
    ...               [0, 0, 0, 3], [0, 0, 0, 0]])
    >>> res = quadratic_assignment(A, B)
    # 调用 quadratic_assignment 函数进行优化，返回结果保存在 res 变量中。
    >>> print(res)
         fun: 3260
     col_ind: [0 3 2 1]
         nit: 9

    The see the relationship between the returned ``col_ind`` and ``fun``,
    use ``col_ind`` to form the best permutation matrix found, then evaluate
    the objective function :math:`f(P) = trace(A^T P B P^T )`.
    # 使用 col_ind 构造找到的最佳排列矩阵 P，然后评估目标函数 f(P) = trace(A^T P B P^T)。

    >>> perm = res['col_ind']
    >>> P = np.eye(len(A), dtype=int)[perm]
    >>> fun = np.trace(A.T @ P @ B @ P.T)
    >>> print(fun)
    3260

    Alternatively, to avoid constructing the permutation matrix explicitly,
    directly permute the rows and columns of the distance matrix.
    # 或者，为了避免显式构造排列矩阵，直接对距离矩阵的行和列进行排列。

    >>> fun = np.trace(A.T @ B[perm][:, perm])
    >>> print(fun)
    3260

    Although not guaranteed in general, ``quadratic_assignment`` happens to
    have found the globally optimal solution.
    # 虽然不能一般保证，但 ``quadratic_assignment`` 偶然找到了全局最优解。

    >>> from itertools import permutations
    >>> perm_opt, fun_opt = None, np.inf
    >>> for perm in permutations([0, 1, 2, 3]):
    ...     perm = np.array(perm)
    ...     fun = np.trace(A.T @ B[perm][:, perm])
    ...     if fun < fun_opt:
    ...         fun_opt, perm_opt = fun, perm
    >>> print(np.array_equal(perm_opt, res['col_ind']))
    True
    # 使用排列的方式遍历可能的组合，找到最优解并进行比较。

    Here is an example for which the default method,
    # 这里是一个示例，展示了默认方法的使用。
    # 如果选项为 None，则将其设为空字典
    if options is None:
        options = {}

    # 将方法名转换为小写
    method = method.lower()
    # 定义支持的方法及其对应的函数
    methods = {"faq": _quadratic_assignment_faq,
               "2opt": _quadratic_assignment_2opt}
    # 如果指定的方法不在支持的方法列表中，抛出数值错误异常
    if method not in methods:
        raise ValueError(f"method {method} must be in {methods}.")
    # 根据选择的方法调用相应的函数，并传入参数 A, B 和选项
    res = methods[method](A, B, **options)
    # 返回计算结果
    return res
    # 计算得分的函数，避免使用矩阵乘法
    return np.sum(A * B[perm][:, perm])


def _common_input_validation(A, B, partial_match):
    # 将 A 和 B 至少转换为二维数组
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    # 如果 partial_match 为 None，则初始化为空的二维数组
    if partial_match is None:
        partial_match = np.array([[], []]).T
    partial_match = np.atleast_2d(partial_match).astype(int)

    msg = None
    # 检查 A 是否为方阵
    if A.shape[0] != A.shape[1]:
        msg = "`A` 必须是方阵"
    # 检查 B 是否为方阵
    elif B.shape[0] != B.shape[1]:
        msg = "`B` 必须是方阵"
    # 检查 A 和 B 是否都是二维数组
    elif A.ndim != 2 or B.ndim != 2:
        msg = "`A` 和 `B` 必须确切地是二维数组"
    # 检查 A 和 B 的形状是否相同
    elif A.shape != B.shape:
        msg = "`A` 和 `B` 矩阵必须具有相同的大小"
    # 检查 partial_match 的行数是否不超过 A 的行数
    elif partial_match.shape[0] > A.shape[0]:
        msg = "`partial_match` 的种子数不能超过节点数"
    # 检查 partial_match 是否有两列
    elif partial_match.shape[1] != 2:
        msg = "`partial_match` 必须有两列"
    # 检查 partial_match 是否确切为二维数组
    elif partial_match.ndim != 2:
        msg = "`partial_match` 必须确切地是二维数组"
    # 检查 partial_match 是否只包含正整数索引
    elif (partial_match < 0).any():
        msg = "`partial_match` 必须只包含正整数索引"
    # 检查 partial_match 是否所有的索引都小于 A 的长度
    elif (partial_match >= len(A)).any():
        msg = "`partial_match` 条目必须小于节点数"
    # 检查 partial_match 的列条目是否唯一
    elif (not len(set(partial_match[:, 0])) == len(partial_match[:, 0]) or
          not len(set(partial_match[:, 1])) == len(partial_match[:, 1])):
        msg = "`partial_match` 列条目必须是唯一的"

    # 如果有错误消息，则抛出值错误异常
    if msg is not None:
        raise ValueError(msg)

    # 返回验证后的 A、B 和 partial_match
    return A, B, partial_match


def _quadratic_assignment_faq(A, B,
                              maximize=False, partial_match=None, rng=None,
                              P0="barycenter", shuffle_input=False, maxiter=30,
                              tol=0.03, **unknown_options):
    r"""解决二次分配问题（近似）。

    此函数使用快速近似二次分配算法（FAQ）[1]_ 解决二次分配问题（QAP）和图匹配问题（GMP）。

    二次分配解决如下形式的问题：

    .. math::

        \min_P & \ {\ \text{trace}(A^T P B P^T)}\\
        \mbox{s.t. } & {P \ \epsilon \ \mathcal{P}}\\

    其中 :math:`\mathcal{P}` 是所有排列矩阵的集合，:math:`A` 和 :math:`B` 是方阵。

    图匹配试图 *最大化* 相同的目标函数。此算法可以被认为是找到两个图的节点对齐，
    使得诱导边分歧的数量最小，或者在加权图的情况下，边权重差的平方和最小。

    注意，二次分配问题是 NP 难题。这里给出的结果是近似值，不能保证是最优的。

    Parameters
    ----------
    A : 二维数组，方阵
        上述目标函数中的方阵 :math:`A`。
    # B 是一个二维数组，表示正方形矩阵，用作上述目标函数中的矩阵 B。
    B : 2-D array, square
    # method 是一个字符串，可以取 {'faq', '2opt'} 中的一个值（默认为 'faq'）。
    # 这是用于解决问题的算法。对于 'faq' 方法，这是特定于方法的文档。
    # 也可以参考 :ref:`'2opt' <optimize.qap-2opt>`。
    method : str in {'faq', '2opt'} (default: 'faq')

    # maximize 是一个布尔值（默认为 False）。
    # 如果为 True，则最大化目标函数。
    maximize : bool (default: False)

    # partial_match 是一个可选的二维整数数组（默认为 None）。
    # 用于固定部分匹配，也称为 "种子"。
    # partial_match 的每一行指定一对匹配的节点：
    # A 的节点 partial_match[i, 0] 匹配到 B 的节点 partial_match[i, 1]。
    # 数组的形状为 (m, 2)，其中 m 不大于节点数 n。
    partial_match : 2-D array of integers, optional (default: None)

    # rng 是一个可选参数，可以是 None、整数、numpy.random.Generator 或 numpy.random.RandomState 类的实例。
    # 当 seed 为 None（或 np.random）时，使用 numpy.random.RandomState 单例。
    # 当 seed 是一个整数时，使用一个新的 RandomState 实例，种子为 seed。
    # 当 seed 已经是 Generator 或 RandomState 实例时，则直接使用该实例。
    rng : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional

    # P0 是一个可选的参数，可以是二维数组、"barycenter" 或 "randomized"（默认为 "barycenter"）。
    # 初始位置。必须是一个双随机矩阵。
    # 如果初始位置是一个数组，它必须是一个大小为 m' x m' 的双随机矩阵，其中 m' = n - m。
    # 如果是 "barycenter"（默认），初始位置是 Birkhoff 多面体的重心（双随机矩阵空间）。
    # 这是一个 m' x m' 的矩阵，所有条目均相等于 1 / m'。
    # 如果是 "randomized"，则初始搜索位置是 P0 = (J + K) / 2，其中 J 是重心，K 是一个随机双随机矩阵。
    P0 : 2-D array, "barycenter", or "randomized" (default: "barycenter")

    # shuffle_input 是一个布尔值（默认为 False）。
    # 设置为 True 以随机解决退化梯度问题。对于非退化梯度，此选项无效。
    shuffle_input : bool (default: False)

    # maxiter 是一个正整数（默认为 30）。
    # 指定执行的 Frank-Wolfe 迭代的最大次数。
    maxiter : int, positive (default: 30)

    # tol 是一个浮点数（默认为 0.03）。
    # 终止的容差。Frank-Wolfe 迭代在满足条件时终止：
    # ||P_{i}-P_{i+1}||_F / sqrt(m') <= tol，
    # 其中 i 是迭代次数，m' 是大小为 m' x m' 的矩阵。
    tol : float (default: 0.03)

    # 返回一个 OptimizeResult 对象。
    # 包含以下字段：
    # col_ind: 1-D 数组，找到的最佳排列节点 B 的列索引。
    # fun: 浮点数，解的目标函数值。
    # nit: 整数，执行的 Frank-Wolfe 迭代次数。
    Returns
    -------
    res : OptimizeResult
        `OptimizeResult` 包含以下字段。
        col_ind : 1-D array
            最佳排列的节点 B 的列索引。
        fun : float
            解的目标函数值。
        nit : int
            执行的 Frank-Wolfe 迭代次数。
    """
    Check for unknown options and raise an error if found.

    Parameters
    ----------
    unknown_options : dict
        Dictionary containing options that are not recognized.

    Raises
    ------
    ValueError
        If unknown options are found.

    """
    _check_unknown_options(unknown_options)

    # Convert maxiter to an integer
    maxiter = operator.index(maxiter)

    # Validate input matrices A, B and partial_match
    A, B, partial_match = _common_input_validation(A, B, partial_match)

    msg = None
    # Validate P0 parameter
    if isinstance(P0, str) and P0 not in {'barycenter', 'randomized'}:
        msg = "Invalid 'P0' parameter string"
    # Check maxiter value
    elif maxiter <= 0:
        msg = "'maxiter' must be a positive integer"
    # Check tol value
    elif tol <= 0:
        msg = "'tol' must be a positive float"
    
    # Raise ValueError if any validation failed
    if msg is not None:
        raise ValueError(msg)

    # Ensure rng is a valid random state
    rng = check_random_state(rng)
    n = len(A)  # number of vertices in graphs
    n_seeds = len(partial_match)  # number of seeds
    n_unseed = n - n_seeds

    # [1] Algorithm 1 Line 1 - choose initialization
    # This comment indicates the start of a specific step in the algorithm,
    # referring to the process of choosing an initialization method.
    # It's a reference to a specific line in the algorithm described in the
    # accompanying documentation or literature.
    #
    # The code execution follows from here based on the initialized parameters.
    #
    # 如果 P0 不是字符串，则将其转换为至少二维数组
    if not isinstance(P0, str):
        P0 = np.atleast_2d(P0)
        # 检查 P0 的形状是否为 (n_unseed, n_unseed)
        if P0.shape != (n_unseed, n_unseed):
            msg = "`P0` matrix must have shape m' x m', where m'=n-m"
        # 检查 P0 是否为双随机矩阵：所有元素非负，行和列的和均为1
        elif ((P0 < 0).any() or not np.allclose(np.sum(P0, axis=0), 1)
              or not np.allclose(np.sum(P0, axis=1), 1)):
            msg = "`P0` matrix must be doubly stochastic"
        # 如果 msg 不为 None，则抛出 ValueError 异常
        if msg is not None:
            raise ValueError(msg)
    # 如果 P0 是字符串 'barycenter'，则生成一个均匀分布的双随机矩阵
    elif P0 == 'barycenter':
        P0 = np.ones((n_unseed, n_unseed)) / n_unseed
    # 如果 P0 是字符串 'randomized'，则生成一个随机双随机矩阵
    elif P0 == 'randomized':
        J = np.ones((n_unseed, n_unseed)) / n_unseed
        # 生成一个 n_unseed x n_unseed 的随机矩阵，每个元素均匀分布在 [0, 1] 之间
        K = _doubly_stochastic(rng.uniform(size=(n_unseed, n_unseed)))
        # 计算 J 和 K 的平均值作为 P0
        P0 = (J + K) / 2

    # 检查是否为平凡情况
    if n == 0 or n_seeds == n:
        # 计算初始匹配的得分并返回优化结果
        score = _calc_score(A, B, partial_match[:, 1])
        res = {"col_ind": partial_match[:, 1], "fun": score, "nit": 0}
        return OptimizeResult(res)

    # 设置目标函数的标量值，默认为 1
    obj_func_scalar = 1
    # 如果 maximize 为 True，则将目标函数标量值设为 -1
    if maximize:
        obj_func_scalar = -1

    # 计算非种子节点 B 的索引
    nonseed_B = np.setdiff1d(range(n), partial_match[:, 1])
    # 如果 shuffle_input 为 True，则对非种子节点 B 进行随机排列
    if shuffle_input:
        nonseed_B = rng.permutation(nonseed_B)

    # 计算非种子节点 A 的索引
    nonseed_A = np.setdiff1d(range(n), partial_match[:, 0])
    # 构造重新排列后的节点索引数组
    perm_A = np.concatenate([partial_match[:, 0], nonseed_A])
    perm_B = np.concatenate([partial_match[:, 1], nonseed_B])

    # 根据 Seeded Graph Matching [2] 定义的矩阵分割方式，将 A 和 B 分割为四个子矩阵
    A11, A12, A21, A22 = _split_matrix(A[perm_A][:, perm_A], n_seeds)
    B11, B12, B21, B22 = _split_matrix(B[perm_B][:, perm_B], n_seeds)
    # 计算常数项 A21 @ B21.T + A12.T @ B12
    const_sum = A21 @ B21.T + A12.T @ B12

    # 将 P 初始化为 P0
    P = P0
    # [1] 算法 1 第 2 行 - 在未满足停止条件时循环
    for n_iter in range(1, maxiter+1):
        # [1] Algorithm 1 Line 3 - 计算梯度，f(P) = -tr(APB^tP^t)
        grad_fp = (const_sum + A22 @ P @ B22.T + A22.T @ P @ B22)
        # [1] Algorithm 1 Line 4 - 解方程 8 获取方向 Q
        _, cols = linear_sum_assignment(grad_fp, maximize=maximize)
        Q = np.eye(n_unseed)[cols]

        # [1] Algorithm 1 Line 5 - 计算步长 alpha
        # 注意，例如 trace(Ax) = trace(A)*x，展开和重新收集术语为 ax**2 + bx + c。
        # c 不影响最小值位置，可以忽略。同时，注意 trace(A@B) = (A.T*B).sum()，在可能的情况下应用以提高效率。
        R = P - Q
        b21 = ((R.T @ A21) * B21).sum()
        b12 = ((R.T @ A12.T) * B12.T).sum()
        AR22 = A22.T @ R
        BR22 = B22 @ R.T
        b22a = (AR22 * B22.T[cols]).sum()
        b22b = (A22 * BR22[cols]).sum()
        a = (AR22.T * BR22).sum()
        b = b21 + b12 + b22a + b22b
        # ax^2 + bx + c 的临界点在 x = -d/(2*e)
        # 如果 a * obj_func_scalar > 0，则是一个最小值
        # 如果最小值不在 [0, 1] 内，只需要考虑端点
        if a * obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = np.argmin([0, (b + a) * obj_func_scalar])

        # [1] Algorithm 1 Line 6 - 更新 P
        P_i1 = alpha * P + (1 - alpha) * Q
        if np.linalg.norm(P - P_i1) / np.sqrt(n_unseed) < tol:
            P = P_i1
            break
        P = P_i1
    # [1] Algorithm 1 Line 7 - 结束主循环

    # [1] Algorithm 1 Line 8 - 投影到置换矩阵集合上
    _, col = linear_sum_assignment(P, maximize=True)
    perm = np.concatenate((np.arange(n_seeds), col + n_seeds))

    unshuffled_perm = np.zeros(n, dtype=int)
    unshuffled_perm[perm_A] = perm_B[perm]

    score = _calc_score(A, B, unshuffled_perm)
    res = {"col_ind": unshuffled_perm, "fun": score, "nit": n_iter}
    return OptimizeResult(res)
def _split_matrix(X, n):
    # 根据 Seeded Graph Matching [2] 的定义，将矩阵 X 分割成上下两部分
    upper, lower = X[:n], X[n:]
    return upper[:, :n], upper[:, n:], lower[:, :n], lower[:, n:]


def _doubly_stochastic(P, tol=1e-3):
    # 改编自 @btaba 的实现
    # https://github.com/btaba/sinkhorn_knopp
    # Sinkhorn-Knopp 算法的实现
    # https://projecteuclid.org/euclid.pjm/1102992505

    max_iter = 1000
    # 计算列的倒数之和，用于初始化 c
    c = 1 / P.sum(axis=0)
    # 计算行的倒数之和，用于初始化 r
    r = 1 / (P @ c)
    P_eps = P

    for it in range(max_iter):
        # 如果所有行和列的和接近 1（在容差 tol 内）
        if ((np.abs(P_eps.sum(axis=1) - 1) < tol).all() and
                (np.abs(P_eps.sum(axis=0) - 1) < tol).all()):
            # 已达到稳定状态，退出循环
            break

        # 更新列比例因子 c
        c = 1 / (r @ P)
        # 更新行比例因子 r
        r = 1 / (P @ c)
        # 更新 P_eps 为新的 doubly stochastic 矩阵
        P_eps = r[:, None] * P * c

    return P_eps


def _quadratic_assignment_2opt(A, B, maximize=False, rng=None,
                               partial_match=None,
                               partial_guess=None,
                               **unknown_options):
    r"""解决二次分配问题（QAP）的近似方法。

    该函数使用 2-opt 算法 [1]_ 解决二次分配问题（QAP）和图匹配问题（GMP）。

    二次分配问题解决以下形式的问题：

    .. math::

        \min_P & \ {\ \text{trace}(A^T P B P^T)}\\
        \mbox{s.t. } & {P \ \epsilon \ \mathcal{P}}

    其中 :math:`\mathcal{P}` 是所有置换矩阵的集合，:math:`A` 和 :math:`B` 是方阵。

    图匹配试图*最大化*相同的目标函数。该算法可视为找到最小化引起的边不一致的对齐方式，
    或者在加权图的情况下，最小化平方边权重差异的节点。

    注意，二次分配问题是 NP-难的。这里给出的结果是近似值，不能保证是最优的。

    Parameters
    ----------
    A : 2-D array, square
        上述目标函数中的方阵 :math:`A`。
    B : 2-D array, square
        上述目标函数中的方阵 :math:`B`。
    method :  str in {'faq', '2opt'} (default: 'faq')
        用于解决问题的算法。这是“2opt”方法的特定于方法的文档。
        参见 :ref:`'faq' <optimize.qap-faq>` 也可以。

    Options
    -------
    maximize : bool (default: False)
        如果为 ``True``，最大化目标函数。
    rng : {None, int, `numpy.random.Generator`,
           `numpy.random.RandomState`}, optional

        如果 `seed` 是 None（或 `np.random`），则使用 `numpy.random.RandomState` 单例。
        如果 `seed` 是整数，则使用带有 `seed` 的新 ``RandomState`` 实例。
        如果 `seed` 已经是 ``Generator`` 或 ``RandomState`` 实例，则使用该实例。
    # 检查未知选项，确保没有未知选项被传入
    _check_unknown_options(unknown_options)
    # 检查随机数生成器是否有效，如果有效则使用，否则创建一个新的
    rng = check_random_state(rng)
    # 对输入进行验证和处理，确保输入的一致性和有效性
    A, B, partial_match = _common_input_validation(A, B, partial_match)

    # 计算节点数目 N
    N = len(A)
    # 检查特殊情况：当节点数为0或partial_match的行数等于N时
    if N == 0 or partial_match.shape[0] == N:
        # 计算当前匹配的得分
        score = _calc_score(A, B, partial_match[:, 1])
        # 构造结果字典，包含最佳排列的列索引，得分和迭代次数
        res = {"col_ind": partial_match[:, 1], "fun": score, "nit": 0}
        # 返回优化结果对象
        return OptimizeResult(res)

    # 如果未提供partial_guess，则创建一个空的数组
    if partial_guess is None:
        partial_guess = np.array([[], []]).T
    # 将partial_guess转换为至少二维的整数数组
    partial_guess = np.atleast_2d(partial_guess).astype(int)

    # 检查partial_guess的合法性并生成相应的错误消息
    msg = None
    if partial_guess.shape[0] > A.shape[0]:
        msg = ("`partial_guess` can have only as "
               "many entries as there are nodes")
    elif partial_guess.shape[1] != 2:
        msg = "`partial_guess` must have two columns"
    elif partial_guess.ndim != 2:
        msg = "`partial_guess` must have exactly two dimensions"
    # 如果 `partial_guess` 中有负数，则抛出异常信息
    elif (partial_guess < 0).any():
        msg = "`partial_guess` must contain only positive indices"
    # 如果 `partial_guess` 中有大于等于 `A` 的长度的数，则抛出异常信息
    elif (partial_guess >= len(A)).any():
        msg = "`partial_guess` entries must be less than number of nodes"
    # 如果 `partial_guess` 的第一列或第二列有重复值，则抛出异常信息
    elif (not len(set(partial_guess[:, 0])) == len(partial_guess[:, 0]) or
          not len(set(partial_guess[:, 1])) == len(partial_guess[:, 1])):
        msg = "`partial_guess` column entries must be unique"
    # 如果异常信息不为空，则抛出值错误异常
    if msg is not None:
        raise ValueError(msg)

    # 初始化 `fixed_rows` 为 None
    fixed_rows = None
    # 如果 `partial_match` 或 `partial_guess` 非空
    if partial_match.size or partial_guess.size:
        # 使用 `partial_guess` 和 `partial_match` 进行初始排列，其余部分随机排列
        guess_rows = np.zeros(N, dtype=bool)
        guess_cols = np.zeros(N, dtype=bool)
        fixed_rows = np.zeros(N, dtype=bool)
        fixed_cols = np.zeros(N, dtype=bool)
        perm = np.zeros(N, dtype=int)

        # 从 `partial_guess` 中提取行和列
        rg, cg = partial_guess.T
        guess_rows[rg] = True
        guess_cols[cg] = True
        perm[guess_rows] = cg

        # `partial_match` 覆盖 `partial_guess` 的结果
        rf, cf = partial_match.T
        fixed_rows[rf] = True
        fixed_cols[cf] = True
        perm[fixed_rows] = cf

        # 随机选择未固定的行和列进行排列
        random_rows = ~fixed_rows & ~guess_rows
        random_cols = ~fixed_cols & ~guess_cols
        perm[random_rows] = rng.permutation(np.arange(N)[random_cols])
    else:
        # 否则，完全随机排列
        perm = rng.permutation(np.arange(N))

    # 计算使用当前排列 `perm` 的最佳得分
    best_score = _calc_score(A, B, perm)

    # 初始化自由索引 `i_free` 为所有可能的索引
    i_free = np.arange(N)
    # 如果 `fixed_rows` 不为空，则筛选出未固定的索引
    if fixed_rows is not None:
        i_free = i_free[~fixed_rows]

    # 根据 `maximize` 决定比较函数 `better`
    better = operator.gt if maximize else operator.lt
    # 初始化迭代计数器和结束标志
    n_iter = 0
    done = False
    while not done:
        # 等效于嵌套的 for 循环，i 和 j 的组合，i 和 j 在范围内进行选择
        for i, j in itertools.combinations_with_replacement(i_free, 2):
            n_iter += 1
            # 交换排列 `perm` 中的两个位置
            perm[i], perm[j] = perm[j], perm[i]
            # 计算交换后的排列的得分
            score = _calc_score(A, B, perm)
            # 如果得分更优，则更新最佳得分，并终止当前循环
            if better(score, best_score):
                best_score = score
                break
            # 否则，恢复原始排列，以便下一次交换
            perm[i], perm[j] = perm[j], perm[i]
        else:  # 如果内层循环没有进行交换
            done = True

    # 构建优化结果的字典
    res = {"col_ind": perm, "fun": best_score, "nit": n_iter}
    # 返回优化结果对象
    return OptimizeResult(res)
```