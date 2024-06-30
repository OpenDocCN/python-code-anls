# `D:\src\scipysrc\sympy\sympy\solvers\simplex.py`

```
# 导入必要的模块和函数
from sympy.core import sympify  # 导入 sympify 函数，用于将输入转换为 SymPy 表达式
from sympy.core.exprtools import factor_terms  # 导入 factor_terms 函数，用于因式分解表达式
from sympy.core.relational import Le, Ge, Eq  # 导入符号表示 ≤、≥、= 的类
from sympy.core.singleton import S  # 导入符号 S，表示 SymPy 的单例类
from sympy.core.symbol import Dummy  # 导入 Dummy 符号，用于生成无用的符号变量
from sympy.core.sorting import ordered  # 导入 ordered 函数，用于对对象进行排序
from sympy.functions.elementary.complexes import sign  # 导入 sign 函数，用于计算数的符号
from sympy.matrices.dense import Matrix, zeros  # 导入 Matrix 和 zeros 类，用于处理矩阵
from sympy.solvers.solveset import linear_eq_to_matrix  # 导入 linear_eq_to_matrix 函数，用于将线性方程转换为矩阵形式
from sympy.utilities.iterables import numbered_symbols  # 导入 numbered_symbols 函数，用于生成带编号的符号
from sympy.utilities.misc import filldedent  # 导入 filldedent 函数，用于消除多余的缩进

# 自定义异常类，表示线性规划问题无界
class UnboundedLPError(Exception):
    """
    A linear programing problem is said to be unbounded if its objective
    function can assume arbitrarily large values.

    Example
    =======

    Suppose you want to maximize
        2x
    subject to
        x >= 0

    There's no upper limit that 2x can take.
    """
    pass

# 自定义异常类，表示线性规划问题无解
class InfeasibleLPError(Exception):
    """
    An error is raised if there is no possible solution:

    >>> lpmin(x,[x<=1,x>=2])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.InfeasibleLPError:
    Inconsistent/False constraint
    """
    pass
    A linear programing problem is considered infeasible if its
    constraint set is empty. That is, if the set of all vectors
    satisfying the contraints is empty, then the problem is infeasible.

    Example
    =======

    Suppose you want to maximize
        x
    subject to
        x >= 10
        x <= 9

    No x can satisfy those constraints.
    """
def _pivot(M, i, j):
    """
    The pivot element `M[i, j]` is inverted and the rest of the matrix
    modified and returned as a new matrix; original is left unmodified.

    Example
    =======

    >>> from sympy.matrices.dense import Matrix
    >>> from sympy.solvers.simplex import _pivot
    >>> from sympy import var
    >>> Matrix(3, 3, var('a:i'))
    Matrix([
    [a, b, c],
    [d, e, f],
    [g, h, i]])
    >>> _pivot(_, 1, 0)
    Matrix([
    [-a/d, -a*e/d + b, -a*f/d + c],
    [ 1/d,        e/d,        f/d],
    [-g/d,  h - e*g/d,  i - f*g/d]])
    """
    # Extract rows and columns related to pivot
    Mi, Mj, Mij = M[i, :], M[:, j], M[i, j]
    # Check if the pivot element is zero, which is not allowed
    if Mij == 0:
        raise ZeroDivisionError(
            "Tried to pivot about zero-valued entry.")
    # Compute the new matrix after pivoting
    A = M - Mj * (Mi / Mij)
    A[i, :] = Mi / Mij   # Normalize the pivot row
    A[:, j] = -Mj / Mij  # Normalize the pivot column
    A[i, j] = 1 / Mij    # Set the pivot element to its inverse
    return A


def _choose_pivot_row(A, B, candidate_rows, pivot_col, Y):
    """
    Choose the row with the smallest ratio B[i] / A[i, pivot_col],
    and in case of ties, use Bland's rule.

    Parameters:
    ===========
    A : Matrix
        Coefficient matrix for the linear constraints.
    B : Matrix
        Right-hand side vector of the linear constraints.
    candidate_rows : list
        List of candidate row indices to choose from.
    pivot_col : int
        Column index of the pivot element.
    Y : list
        List representing additional criteria for tie-breaking.

    Returns:
    ========
    int
        Index of the chosen pivot row.
    """
    # Use min function with a lambda to find the index of the minimum ratio
    return min(candidate_rows, key=lambda i: (B[i] / A[i, pivot_col], Y[i]))


def _simplex(A, B, C, D=None, dual=False):
    """
    Perform the two-phase simplex method to solve linear programming problems.

    Parameters:
    ===========
    A : Matrix
        Coefficient matrix for the linear constraints.
    B : Matrix
        Right-hand side vector of the linear constraints.
    C : Matrix
        Coefficients of the objective function to be minimized or maximized.
    D : Matrix, optional
        Constant vector in the objective function (default is None).
    dual : bool, optional
        If True, solve the dual problem (default is False).

    Returns:
    ========
    tuple
        Depending on `dual`:
        If dual is False:
            (o, x, y)
            o: Minimum value of the objective function C*x - D under constraints Ax <= B and x >= 0.
            x: Solution vector.
            y: None.
        If dual is True:
            (o, y, x)
            o: Maximum value of the dual objective function y^T * B - D under constraints A^T * y >= C^T and y >= 0.
            y: Solution vector.
            x: None.
    """
    # Implementation details and examples are omitted for brevity
    pass
    """
    Since `_simplex` will do a minimization for constraints given as
    ``A*x <= B``, the signs of ``A`` and ``B`` must be negated since
    the currently correspond to a greater-than inequality:
    """

    A, B, C, D = [Matrix(i) for i in (A, B, C, D or [0])]
    # 将输入的矩阵转换为SymPy的Matrix对象，并处理缺失的D参数为默认值0

    if dual:
        # 如果执行对偶模式
        _o, d, p = _simplex(-A.T, C.T, B.T, -D)
        # 对偶模式下调用_simplex函数，需传入转置后的矩阵以及相反的C和D
        return -_o, d, p

    if A and B:
        # 如果A和B均存在，表示有约束条件
        M = Matrix([[A, B], [C, D]])
        # 构造对应的矩阵M，包含A、B、C、D
    else:
        if A or B:
            # 如果只有A或者B其中之一存在，抛出数值错误异常
            raise ValueError("must give A and B")
        # 如果没有约束条件给出
        # M仅包含C和D
        M = Matrix([[C, D]])

    n = M.cols - 1
    # 矩阵M的列数减1，得到n
    m = M.rows - 1
    # 矩阵M的行数减1，得到m
    ```
    # 检查矩阵 M 中是否所有元素都是 Float 或 Rational 类型
    if not all(i.is_Float or i.is_Rational for i in M):
        # 如果 M 中有不是 Float 或 Rational 的元素，则抛出类型错误异常
        raise TypeError(filldedent("""
            Only rationals and floats are allowed.
            """
            )
        )

    # x 变量在 Bland's 规则中具有优先权，因为 False < True
    X = [(False, j) for j in range(n)]
    Y = [(True, i) for i in range(m)]

    # 第一阶段：找到一个可行解或确定不存在
    ## 记录最后一个主元素所在的行和列
    last = None

    while True:
        B = M[:-1, -1]
        A = M[:-1, :-1]
        # 检查是否所有 B 中的元素都大于等于 0
        if all(B[i] >= 0 for i in range(B.rows)):
            # 找到了一个可行解
            break

        # 找到第一个右侧最后一个元素为负数的行 k
        for k in range(B.rows):
            if B[k] < 0:
                break  # 使用下面的 k 值

        # 选择主元素列 c
        piv_cols = [_ for _ in range(A.cols) if A[k, _] < 0]
        if not piv_cols:
            # 如果不存在主元素列，则抛出线性规划不可行异常
            raise InfeasibleLPError(filldedent("""
                The constraint set is empty!"""))
        # 使用 Bland's 规则选择最小的主元素列
        _, c = min((X[i], i) for i in piv_cols)

        # 选择主元素行 r
        piv_rows = [_ for _ in range(A.rows) if A[_, c] > 0 and B[_] > 0]
        piv_rows.append(k)
        r = _choose_pivot_row(A, B, piv_rows, c, Y)

        # 检查是否存在振荡
        if (r, c) == last:
            # 如果存在振荡，记录并跳出循环
            last = True
            break
        last = r, c

        # 执行主元素调整操作
        M = _pivot(M, r, c)
        # 更新 X 和 Y 列表中的元素
        X[c], Y[r] = Y[r], X[c]

    # 第二阶段：从可行解开始，执行主元素调整以达到最优解
    # 进入主循环，执行单纯形算法的迭代过程，直至找到最优解或发现无界面问题
    while True:
        # 提取出矩阵 M 的 B 列（除最后一行）作为向量 B
        B = M[:-1, -1]
        # 提取出矩阵 M 的除最后一行和最后一列的部分作为子矩阵 A
        A = M[:-1, :-1]
        # 提取出矩阵 M 的最后一行除最后一个元素外的部分作为向量 C
        C = M[-1, :-1]

        # 选择一个主元列 c
        piv_cols = [_ for _ in range(n) if C[_] < 0]
        if not piv_cols:
            break
        # 使用 Bland 规则选择主元列 c
        _, c = min((X[i], i) for i in piv_cols)  # Bland's rule

        # 选择一个主元行 r
        piv_rows = [_ for _ in range(m) if A[_, c] > 0]
        if not piv_rows:
            raise UnboundedLPError(filldedent("""
                Objective function can assume
                arbitrarily large values!"""))
        # 使用辅助函数选择主元行 r
        r = _choose_pivot_row(A, B, piv_rows, c, Y)

        # 进行主元素调整操作，得到新的矩阵 M
        M = _pivot(M, r, c)
        # 更新变量 X 和 Y 的顺序
        X[c], Y[r] = Y[r], X[c]

    # 初始化最优解变量和对偶最优解变量
    argmax = [None] * n
    argmin_dual = [None] * m

    # 处理最优解变量 X
    for i, (v, n) in enumerate(X):
        if v == False:
            argmax[n] = 0
        else:
            argmin_dual[n] = M[-1, i]

    # 处理对偶最优解变量 Y
    for i, (v, n) in enumerate(Y):
        if v == True:
            argmin_dual[n] = 0
        else:
            argmax[n] = M[i, -1]

    # 检查是否存在无效解，如果存在则抛出异常
    if last and not all(i >= 0 for i in argmax + argmin_dual):
        raise InfeasibleLPError(filldedent("""
            Oscillating system led to invalid solution.
            If you believe there was a valid solution, please
            report this as a bug."""))

    # 返回最优解值以及最优解变量和对偶最优解变量
    return -M[-1, -1], argmax, argmin_dual
## routines that use _simplex or support those that do

# 定义函数 _abcd，从矩阵 M 中返回部分矩阵或列表
def _abcd(M, list=False):
    """return parts of M as matrices or lists

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.solvers.simplex import _abcd

    >>> m = Matrix(3, 3, range(9)); m
    Matrix([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]])
    >>> a, b, c, d = _abcd(m)
    >>> a
    Matrix([
    [0, 1],
    [3, 4]])
    >>> b
    Matrix([
    [2],
    [5]])
    >>> c
    Matrix([[6, 7]])
    >>> d
    Matrix([[8]])

    The matrices can be returned as compact lists, too:

    >>> L = a, b, c, d = _abcd(m, list=True); L
    ([[0, 1], [3, 4]], [2, 5], [[6, 7]], [8])
    """

    # 定义内部函数 aslist，将矩阵转换为列表形式
    def aslist(i):
        l = i.tolist()
        if len(l[0]) == 1:  # 如果是列向量
            return [i[0] for i in l]
        return l

    # 按照指定的参数 list，返回矩阵 M 的部分
    m = M[:-1, :-1], M[:-1, -1], M[-1, :-1], M[-1:, -1:]
    if not list:
        return m
    return tuple([aslist(i) for i in m])


# 定义函数 _m，根据给定的矩阵或列表返回 Matrix([[a, b], [c, d]])
def _m(a, b, c, d=None):
    """return Matrix([[a, b], [c, d]]) from matrices
    in Matrix or list form.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.solvers.simplex import _abcd, _m
    >>> m = Matrix(3, 3, range(9))
    >>> L = _abcd(m, list=True); L
    ([[0, 1], [3, 4]], [2, 5], [[6, 7]], [8])
    >>> _abcd(m)
    (Matrix([
    [0, 1],
    [3, 4]]), Matrix([
    [2],
    [5]]), Matrix([[6, 7]]), Matrix([[8]]))
    >>> assert m == _m(*L) == _m(*_)
    """
    # 将参数转换为 Matrix 对象
    a, b, c, d = [Matrix(i) for i in (a, b, c, d or [0])]
    return Matrix([[a, b], [c, d]])


# 定义函数 _primal_dual，根据矩阵 M 返回原始问题和对偶问题的函数和约束条件
def _primal_dual(M, factor=True):
    """return primal and dual function and constraints
    assuming that ``M = Matrix([[A, b], [c, d]])`` and the
    function ``c*x - d`` is being minimized with ``Ax >= b``
    for nonnegative values of ``x``. The dual and its
    constraints will be for maximizing `b.T*y - d` subject
    to ``A.T*y <= c.T``.

    Examples
    ========

    >>> from sympy.solvers.simplex import _primal_dual, lpmin, lpmax
    >>> from sympy import Matrix

    The following matrix represents the primal task of
    minimizing x + y + 7 for y >= x + 1 and y >= -2*x + 3.
    The dual task seeks to maximize x + 3*y + 7 with
    2*y - x <= 1 and and x + y <= 1:

    >>> M = Matrix([
    ...     [-1, 1,  1],
    ...     [ 2, 1,  3],
    ...     [ 1, 1, -7]])
    >>> p, d = _primal_dual(M)

    The minimum of the primal and maximum of the dual are the same
    (though they occur at different points):

    >>> lpmin(*p)
    (28/3, {x1: 2/3, x2: 5/3})
    >>> lpmax(*d)
    (28/3, {y1: 1/3, y2: 2/3})

    If the equivalent (but canonical) inequalities are
    desired, leave `factor=True`, otherwise the unmodified
    inequalities for M will be returned.

    >>> m = Matrix([
    ... [-3, -2,  4, -2],
    ... [ 2,  0,  0, -2],
    ... [ 0,  1, -3,  0]])

    >>> _primal_dual(m, False)  # last condition is 2*x1 >= -2
    ((x2 - 3*x3,
        [-3*x1 - 2*x2 + 4*x3 >= -2, 2*x1 >= -2]),

"""
    if not M:
        return (None, []), (None, [])
    if not hasattr(M, "shape"):
        if len(M) not in (3, 4):
            raise ValueError("expecting Matrix or 3 or 4 lists")
        # 将 M 转换成合适的矩阵格式
        M = _m(*M)
    m, n = [i - 1 for i in M.shape]
    # 获取线性规划问题的系数矩阵 A, b, c, d
    A, b, c, d = _abcd(M)
    d = d[0]
    _ = lambda x: numbered_symbols(x, start=1)
    # 创建变量 x 和 y，分别表示原始问题的变量和对偶问题的变量
    x = Matrix([i for i, j in zip(_("x"), range(n))])
    yT = Matrix([i for i, j in zip(_("y"), range(m))]).T

    def ineq(L, r, op):
        # 检查不等式约束是否成立，返回符合条件的约束列表
        rv = []
        for r in (op(i, j) for i, j in zip(L, r)):
            if r == True:
                continue
            elif r == False:
                return [False]
            if factor:
                # 因式分解处理
                f = factor_terms(r)
                if f.lhs.is_Mul and f.rhs % f.lhs.args[0] == 0:
                    assert len(f.lhs.args) == 2, f.lhs
                    k = f.lhs.args[0]
                    r = r.func(sign(k) * f.lhs.args[1], f.rhs // abs(k))
            rv.append(r)
        return rv

    # 定义等式表达式的函数
    eq = lambda x, d: x[0] - d if x else -d
    # 计算原始问题的目标函数 F 和约束条件
    F = eq(c * x, d)
    # 计算对偶问题的目标函数 f 和约束条件
    f = eq(yT * b, d)
    # 返回原始问题和对偶问题的结果元组
    return (F, ineq(A * x, b, Ge)), (f, ineq(yT * A, c, Le))
    r = {}  # 用于处理变量替换的字典，用于处理变量的变换
    np = []  # 存储非正表达式的列表
    aux = []  # 添加的辅助符号列表
    ui = numbered_symbols("z", start=1, cls=Dummy)  # 创建一个以 z 开头的无数符号生成器，用作辅助变量
    univariate = {}  # 用于存储单变量约束条件的字典，格式为 {x: 区间}
    unbound = []  # 指定为无约束的符号列表
    syms = set(syms)  # 将给定的符号集合转换为集合形式，以确保唯一性
    # 对于每个符号变量 syms 中的每个变量 x，执行以下操作
    for x in syms:
        # 从 univariate 字典中获取变量 x 对应的值 i，如果不存在则返回 True
        i = univariate.get(x, True)
        # 如果 i 为假值，则返回空，表示无法找到解决方案
        if not i:
            return None  # no solution possible
        # 如果 i 为 True，则表示变量 x 没有界限
        if i == True:
            # 将变量 x 加入到 unbound 列表中，表示未绑定的变量
            unbound.append(x)
            # 继续处理下一个变量 x
            continue
        # 从 i 中解析出下界 a 和上界 b
        a, b = i.inf, i.sup
        # 如果下界 a 是无穷大
        if a.is_infinite:
            # 获取下一个可用的唯一标识符 u
            u = next(ui)
            # 计算变量 x 对应的值为 b 减去 u
            r[x] = b - u
            # 将 u 添加到辅助列表 aux 中
            aux.append(u)
        # 如果上界 b 是无穷大
        elif b.is_infinite:
            # 如果存在下界 a
            if a:
                # 获取下一个可用的唯一标识符 u
                u = next(ui)
                # 计算变量 x 对应的值为 a 加上 u
                r[x] = a + u
                # 将 u 添加到辅助列表 aux 中
                aux.append(u)
            else:
                # 标准的非负关系，无需特殊处理
                pass
        else:
            # 获取下一个可用的唯一标识符 u
            u = next(ui)
            # 将 u 添加到辅助列表 aux 中
            aux.append(u)
            # 计算变量 x 对应的值为 u 加上 a
            r[x] = u + a
            # 添加约束条件 u <= b - a，确保 x 的上限为 b
            # 因为当 u = b - a 时，x = u + a = b
            np.append(u - (b - a))

    # 对于未绑定的变量进行变量的更改
    for x in unbound:
        # 获取下一个可用的唯一标识符 u
        u = next(ui)
        # 重新利用变量 x，计算其对应的值为 u 减去 x
        r[x] = u - x
        # 将 u 添加到辅助列表 aux 中
        aux.append(u)

    # 返回结果数组 np，结果字典 r，以及辅助列表 aux
    return np, r, aux
# 返回线性规划问题的系数矩阵和相关变量
def _lp_matrices(objective, constraints):
    """return A, B, C, D, r, x+X, X for maximizing
    objective = Cx - D with constraints Ax <= B, introducing
    introducing auxilliary variables, X, as necessary to make
    replacements of symbols as given in r, {xi: expression with Xj},
    so all variables in x+X will take on nonnegative values.

    Every univariate condition creates a semi-infinite
    condition, e.g. a single ``x <= 3`` creates the
    interval ``[-oo, 3]`` while ``x <= 3`` and ``x >= 2``
    create an interval ``[2, 3]``. Variables not in a univariate
    expression will take on nonnegative values.
    """

    # 将目标函数和约束条件转换为符号表达式并收集自由符号
    F = sympify(objective)
    np = [sympify(i) for i in constraints]
    syms = set.union(*[i.free_symbols for i in [F] + np], set())

    # 将形如 Eq(x, y) 的等式转换为 x - y <= 0 和 y - x <= 0 的形式
    for i in range(len(np)):
        if isinstance(np[i], Eq):
            np[i] = np[i].lhs - np[i].rhs <= 0
            np.append(-np[i].lhs <= 0)

    # 将约束条件转换为非正表达式
    _ = _rel_as_nonpos(np, syms)
    if _ is None:
        raise InfeasibleLPError(filldedent("""
            Inconsistent/False constraint"""))
    np, r, aux = _

    # 进行变量替换
    F = F.xreplace(r)
    np = [i.xreplace(r) for i in np]

    # 转换为矩阵形式
    xx = list(ordered(syms)) + aux
    A, B = linear_eq_to_matrix(np, xx)
    C, D = linear_eq_to_matrix([F], xx)
    return A, B, C, D, r, xx, aux


# 执行线性规划（LP）优化，返回优化结果和最优解
def _lp(min_max, f, constr):
    """Return the optimization (min or max) of ``f`` with the given
    constraints. All variables are unbounded unless constrained.

    If `min_max` is 'max' then the results corresponding to the
    maximization of ``f`` will be returned, else the minimization.
    The constraints can be given as Le, Ge or Eq expressions.

    Examples
    ========

    >>> from sympy.solvers.simplex import _lp as lp
    >>> from sympy import Eq
    >>> from sympy.abc import x, y, z
    >>> f = x + y - 2*z
    >>> c = [7*x + 4*y - 7*z <= 3, 3*x - y + 10*z <= 6]
    >>> c += [i >= 0 for i in (x, y, z)]
    >>> lp(min, f, c)
    (-6/5, {x: 0, y: 0, z: 3/5})

    By passing max, the maximum value for f under the constraints
    is returned (if possible):

    >>> lp(max, f, c)
    (3/4, {x: 0, y: 3/4, z: 0})

    Constraints that are equalities will require that the solution
    also satisfy them:

    >>> lp(max, f, c + [Eq(y - 9*x, 1)])
    (5/7, {x: 0, y: 1, z: 1/7})

    All symbols are reported, even if they are not in the objective
    function:

    >>> lp(min, x, [y + x >= 3, x >= 0])
    (0, {x: 0, y: 3})
    """
    # 获取表示仅非负变量的系统的矩阵组成部分
    A, B, C, D, r, xx, aux = _lp_matrices(f, constr)

    how = str(min_max).lower()
    if "max" in how:
        # 如果目标是最大化，则使用 _simplex 函数进行最小化求解
        # 因为 _simplex 函数针对 Ax <= B 进行最小化求解，需要改变函数的符号
        # 同时对返回的最优值进行取反以得到最大化的结果
        _o, p, d = _simplex(A, B, -C, -D)
        o = -_o
    elif "min" in how:
        # 如果目标是最小化，则直接使用 _simplex 函数进行求解
        o, p, d = _simplex(A, B, C, D)
    else:
        # 如果既不是最大化也不是最小化，则抛出 ValueError
        raise ValueError("expecting min or max")

    # 恢复原始变量并从 p 中移除辅助变量
    p = dict(zip(xx, p))
    if r:  # p 包含原始符号和辅助符号
        # 如果 r 中含有 x: x - z1，则使用 p 中的值更新
        r = {k: v.xreplace(p) for k, v in r.items()}
        # 然后在 p 中使用 x 的实际值（即 x - z1）
        p.update(r)
        # 不显示辅助变量 aux
        p = {k: p[k] for k in ordered(p) if k not in aux}

    # 不返回对偶问题，因为变量可能有有限的边界约束
    return o, p
# 返回线性方程 f 的最小值，受限于用 Ge、Le 或 Eq 表示的线性约束条件
def lpmin(f, constr):
    """return minimum of linear equation ``f`` under
    linear constraints expressed using Ge, Le or Eq.

    All variables are unbounded unless constrained.

    Examples
    ========

    >>> from sympy.solvers.simplex import lpmin
    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> lpmin(x, [2*x - 3*y >= -1, Eq(x + 3*y, 2), x <= 2*y])
    (1/3, {x: 1/3, y: 5/9})

    Negative values for variables are permitted unless explicitly
    exluding, so minimizing ``x`` for ``x <= 3`` is an
    unbounded problem while the following has a bounded solution:

    >>> lpmin(x, [x >= 0, x <= 3])
    (0, {x: 0})

    Without indicating that ``x`` is nonnegative, there
    is no minimum for this objective:

    >>> lpmin(x, [x <= 3])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.UnboundedLPError:
    Objective function can assume arbitrarily large values!

    See Also
    ========
    linprog, lpmax
    """
    # 调用内部函数 _lp，传入 min 函数、目标函数 f 和约束条件 constr
    return _lp(min, f, constr)


def lpmax(f, constr):
    """return maximum of linear equation ``f`` under
    linear constraints expressed using Ge, Le or Eq.

    All variables are unbounded unless constrained.

    Examples
    ========

    >>> from sympy.solvers.simplex import lpmax
    >>> from sympy import Eq
    >>> from sympy.abc import x, y
    >>> lpmax(x, [2*x - 3*y >= -1, Eq(x+ 3*y,2), x <= 2*y])
    (4/5, {x: 4/5, y: 2/5})

    Negative values for variables are permitted unless explicitly
    exluding:

    >>> lpmax(x, [x <= -1])
    (-1, {x: -1})

    If a non-negative constraint is added for x, there is no
    possible solution:

    >>> lpmax(x, [x <= -1, x >= 0])
    Traceback (most recent call last):
    ...
    sympy.solvers.simplex.InfeasibleLPError: inconsistent/False constraint

    See Also
    ========
    linprog, lpmin
    """
    # 调用内部函数 _lp，传入 max 函数、目标函数 f 和约束条件 constr
    return _lp(max, f, constr)


def _handle_bounds(bounds):
    # introduce auxilliary variables as needed for univariate
    # inequalities

    # 初始化一个空列表 unbound，用于存放无约束的辅助变量
    unbound = []
    # 初始化一个全零列表 R，长度与 bounds 相同
    R = [0] * len(bounds)  # a (growing) row of zeros

    # 定义一个函数 n，返回 R 列表的长度减一
    def n():
        return len(R) - 1

    # 定义函数 Arow，用于向 R 列表末尾添加指定数量的零，并返回该列表的副本
    def Arow(inc=1):
        R.extend([0] * inc)
        return R[:]

    # 初始化一个空列表 row，用于存放辅助变量
    row = []
    for x, (a, b) in enumerate(bounds):
        if a is None and b is None:
            unbound.append(x)
        elif a is None:
            # 如果下界为 None，即 r[x] = b - u
            A = Arow()
            A[x] = 1  # 设置 A 的第 x 个元素为 1
            A[n()] = 1  # 设置 A 的最后一个元素为 1
            B = [b]  # 设置约束 B 为 [b]
            row.append((A, B))  # 将 A 和 B 添加到 row 中
            A = [0] * len(A)  # 创建与 A 同样长度的零数组
            A[x] = -1  # 设置 A 的第 x 个元素为 -1
            A[n()] = -1  # 设置 A 的最后一个元素为 -1
            B = [-b]  # 设置约束 B 为 [-b]
            row.append((A, B))  # 将 A 和 B 添加到 row 中
        elif b is None:
            if a:
                # 如果上界为 None 且下界不为 None，即 r[x] = a + u
                A = Arow()
                A[x] = 1  # 设置 A 的第 x 个元素为 1
                A[n()] = -1  # 设置 A 的最后一个元素为 -1
                B = [a]  # 设置约束 B 为 [a]
                row.append((A, B))  # 将 A 和 B 添加到 row 中
                A = [0] * len(A)  # 创建与 A 同样长度的零数组
                A[x] = -1  # 设置 A 的第 x 个元素为 -1
                A[n()] = 1  # 设置 A 的最后一个元素为 1
                B = [-a]  # 设置约束 B 为 [-a]
                row.append((A, B))  # 将 A 和 B 添加到 row 中
            else:
                # 如果上界和下界都为 None，则标准非负关系
                pass
        else:
            # 如果上界和下界都不为 None，即 r[x] = u + a
            A = Arow()
            A[x] = 1  # 设置 A 的第 x 个元素为 1
            A[n()] = -1  # 设置 A 的最后一个元素为 -1
            B = [a]  # 设置约束 B 为 [a]
            row.append((A, B))  # 将 A 和 B 添加到 row 中
            A = [0] * len(A)  # 创建与 A 同样长度的零数组
            A[x] = -1  # 设置 A 的第 x 个元素为 -1
            A[n()] = 1  # 设置 A 的最后一个元素为 1
            B = [-a]  # 设置约束 B 为 [-a]
            row.append((A, B))  # 将 A 和 B 添加到 row 中
            # 添加额外的约束条件 u <= b - a
            A = [0] * len(A)  # 创建与 A 同样长度的零数组
            A[x] = 0  # 设置 A 的第 x 个元素为 0
            A[n()] = 1  # 设置 A 的最后一个元素为 1
            B = [b - a]  # 设置约束 B 为 [b - a]
            row.append((A, B))  # 将 A 和 B 添加到 row 中

    # 对于未限定的变量，进行变量变换
    for x in unbound:
        # 变量变换，使得 r[x] = u - v
        A = Arow(2)  # 创建长度为 2 的 A
        B = [0]  # 设置约束 B 为 [0]
        A[x] = 1  # 设置 A 的第 x 个元素为 1
        A[n()] = 1  # 设置 A 的最后一个元素为 1
        A[n() - 1] = -1  # 设置 A 的倒数第二个元素为 -1
        row.append((A, B))  # 将 A 和 B 添加到 row 中
        A = [0] * len(A)  # 创建与 A 同样长度的零数组
        A[x] = -1  # 设置 A 的第 x 个元素为 -1
        A[n()] = -1  # 设置 A 的最后一个元素为 -1
        A[n() - 1] = 1  # 设置 A 的倒数第二个元素为 1
        row.append((A, B))  # 将 A 和 B 添加到 row 中

    return Matrix([r + [0] * (len(R) - len(r)) for r, _ in row]), Matrix([i[1] for i in row])
# 定义线性规划函数，用于求解目标函数 c*x 的最小化问题，同时考虑线性约束条件
def linprog(c, A=None, b=None, A_eq=None, b_eq=None, bounds=None):
    """Return the minimization of ``c*x`` with the given
    constraints ``A*x <= b`` and ``A_eq*x = b_eq``. Unless bounds
    are given, variables will have nonnegative values in the solution.

    If ``A`` is not given, then the dimension of the system will
    be determined by the length of ``C``.

    By default, all variables will be nonnegative. If ``bounds``
    is given as a single tuple, ``(lo, hi)``, then all variables
    will be constrained to be between ``lo`` and ``hi``. Use
    None for a ``lo`` or ``hi`` if it is unconstrained in the
    negative or positive direction, respectively, e.g.
    ``(None, 0)`` indicates nonpositive values. To set
    individual ranges, pass a list with length equal to the
    number of columns in ``A``, each element being a tuple; if
    only a few variables take on non-default values they can be
    passed as a dictionary with keys giving the corresponding
    column to which the variable is assigned, e.g. ``bounds={2:
    (1, 4)}`` would limit the 3rd variable to have a value in
    range ``[1, 4]``.

    Examples
    ========

    >>> from sympy.solvers.simplex import linprog
    >>> from sympy import symbols, Eq, linear_eq_to_matrix as M, Matrix
    >>> x = x1, x2, x3, x4 = symbols('x1:5')
    >>> X = Matrix(x)
    >>> c, d = M(5*x2 + x3 + 4*x4 - x1, x)
    >>> a, b = M([5*x2 + 2*x3 + 5*x4 - (x1 + 5)], x)
    >>> aeq, beq = M([Eq(3*x2 + x4, 2), Eq(-x1 + x3 + 2*x4, 1)], x)
    >>> constr = [i <= j for i,j in zip(a*X, b)]
    >>> constr += [Eq(i, j) for i,j in zip(aeq*X, beq)]
    >>> linprog(c, a, b, aeq, beq)
    (9/2, [0, 1/2, 0, 1/2])
    >>> assert all(i.subs(dict(zip(x, _[1]))) for i in constr)

    See Also
    ========
    lpmin, lpmax
    """

    # 目标函数矩阵化
    C = Matrix(c)
    if C.rows != 1 and C.cols == 1:
        C = C.T
    if C.rows != 1:
        raise ValueError("C must be a single row.")

    # 不等式约束
    if not A:
        if b:
            raise ValueError("A and b must both be given")
        # 如果未提供 A 和 b，则设定简单约束为零
        A, b = zeros(0, C.cols), zeros(C.cols, 1)
    else:
        A, b = [Matrix(i) for i in (A, b)]

    if A.cols != C.cols:
        raise ValueError("number of columns in A and C must match")

    # 等式约束
    if A_eq is None:
        if not b_eq is None:
            raise ValueError("A_eq and b_eq must both be given")
    else:
        A_eq, b_eq = [Matrix(i) for i in (A_eq, b_eq)]
        # 添加等式约束及其对应的反向约束
        A = A.col_join(A_eq)
        A = A.col_join(-A_eq)
        b = b.col_join(b_eq)
        b = b.col_join(-b_eq)
    # 检查边界是否为None、空字典或(0, None)，如果不是则进行解释
    if not (bounds is None or bounds == {} or bounds == (0, None)):
        ## 边界被解释
        # 如果边界是元组且长度为2，则复制为A的列数个元组
        if type(bounds) is tuple and len(bounds) == 2:
            bounds = [bounds] * A.cols
        # 如果边界长度等于A的列数，并且所有元素都是长度为2的元组
        elif len(bounds) == A.cols and all(
                type(i) is tuple and len(i) == 2 for i in bounds):
            pass  # 单独的边界
        # 如果边界是字典，并且所有值都是长度为2的元组
        elif type(bounds) is dict and all(
                type(i) is tuple and len(i) == 2
                for i in bounds.values()):
            # 稀疏边界
            db = bounds
            bounds = [(0, None)] * A.cols
            # 将字典db中的项逐一设置到bounds中，如果索引超出范围则引发IndexError
            while db:
                i, j = db.popitem()
                bounds[i] = j
        else:
            # 抛出异常，表示边界不符合预期
            raise ValueError("unexpected bounds %s" % bounds)
        # 处理边界，返回处理后的矩阵A_和向量b_
        A_, b_ = _handle_bounds(bounds)
        # 计算辅助变量的数量
        aux = A_.cols - A.cols
        # 如果A存在，则将A扩展并连接到新的A_，b扩展并连接到新的b_
        if A:
            A = Matrix([[A, zeros(A.rows, aux)], [A_]])
            b = b.col_join(b_)
        else:
            A = A_
            b = b_
        # 将C扩展并连接到新的C，增加aux列
        C = C.row_join(zeros(1, aux))
    else:
        # 设置辅助变量的数量为-A的列数，以便在后续使用
        aux = -A.cols  # 设置为负值以便于获取所有列下标

    # 调用_simplex函数进行线性规划计算，返回最优值o、解向量p和d
    o, p, d = _simplex(A, b, C)
    # 返回最优值o和解向量p的前-p个元素，不包括辅助变量
    return o, p[:-aux]  # 不包含辅助变量的值
# 定义线性规划问题的显示函数，给定目标系数和约束条件，返回符号表达式表示的目标函数和约束条件列表
def show_linprog(c, A=None, b=None, A_eq=None, b_eq=None, bounds=None):
    # 导入 sympy 库中的 symbols 函数
    from sympy import symbols
    
    # 创建目标系数的符号矩阵 C
    C = Matrix(c)
    # 如果 C 不是行向量而是列向量，则转置为行向量
    if C.rows != 1 and C.cols == 1:
        C = C.T
    # 如果 C 不是行向量，则引发 ValueError 异常
    if C.rows != 1:
        raise ValueError("C must be a single row.")

    # 处理不等式约束
    if not A:
        # 如果 A 为空，则 b 也必须为空；此时没有不等式约束，仅有简单变量约束
        A, b = zeros(0, C.cols), zeros(C.cols, 1)
    else:
        # 如果 A 不为空，则将 A 和 b 转换为 sympy 的矩阵对象
        A, b = [Matrix(i) for i in (A, b)]

    # 检查 A 的列数与 C 的列数是否匹配，如果不匹配则引发 ValueError 异常
    if A.cols != C.cols:
        raise ValueError("number of columns in A and C must match")

    # 处理等式约束
    if A_eq is None:
        # 如果 A_eq 为空，则 b_eq 也必须为空；此时没有等式约束
        if not b_eq is None:
            raise ValueError("A_eq and b_eq must both be given")
    else:
        # 如果 A_eq 不为空，则将 A_eq 和 b_eq 转换为 sympy 的矩阵对象
        A_eq, b_eq = [Matrix(i) for i in (A_eq, b_eq)]

    # 处理变量的上下界约束
    if not (bounds is None or bounds == {} or bounds == (0, None)):
        # 解释和处理变量的上下界约束
        if type(bounds) is tuple and len(bounds) == 2:
            # 如果 bounds 是元组且长度为 2，则将其扩展为与变量数目相匹配的列表
            bounds = [bounds] * A.cols
        elif len(bounds) == A.cols and all(
                type(i) is tuple and len(i) == 2 for i in bounds):
            pass  # 每个变量有单独的上下界约束
        elif type(bounds) is dict and all(
                type(i) is tuple and len(i) == 2
                for i in bounds.values()):
            # 如果 bounds 是字典，则将其转换为列表形式的上下界约束
            db = bounds
            bounds = [(0, None)] * A.cols
            while db:
                i, j = db.popitem()
                bounds[i] = j  # 如果索引超出边界，引发 IndexError
        else:
            raise ValueError("unexpected bounds %s" % bounds)

    # 创建变量向量 x，其中包含 A.cols 个符号变量
    x = Matrix(symbols('x1:%s' % (A.cols + 1)))
    # 创建目标函数表达式 f 和约束条件列表 c
    f, c = (C * x)[0], [i <= j for i, j in zip(A * x, b)] + [Eq(i, j) for i, j in zip(A_eq * x, b_eq)]
    
    # 遍历变量的上下界约束，将其加入约束条件列表 c
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None:
            c.append(x[i] >= lo)
        if hi is not None:
            c.append(x[i] <= hi)
    
    # 返回目标函数表达式 f 和约束条件列表 c
    return f, c
```