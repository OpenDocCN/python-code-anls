# `D:\src\scipysrc\sympy\sympy\logic\algorithms\lra_theory.py`

```
"""Implements "A Fast Linear-Arithmetic Solver for DPLL(T)"

The LRASolver class defined in this file can be used
in conjunction with a SAT solver to check the
satisfiability of formulas involving inequalities.

Here's an example of how that would work:

    Suppose you want to check the satisfiability of
    the following formula:

    >>> from sympy.core.relational import Eq
    >>> from sympy.abc import x, y
    >>> f = ((x > 0) | (x < 0)) & (Eq(x, 0) | Eq(y, 1)) & (~Eq(y, 1) | Eq(1, 2))

    First a preprocessing step should be done on f. During preprocessing,
    f should be checked for any predicates such as `Q.prime` that can't be
    handled. Also unequality like `~Eq(y, 1)` should be split.

    I should mention that the paper says to split both equalities and
    unequality, but this implementation only requires that unequality
    be split.

    >>> f = ((x > 0) | (x < 0)) & (Eq(x, 0) | Eq(y, 1)) & ((y < 1) | (y > 1) | Eq(1, 2))

    Then an LRASolver instance needs to be initialized with this formula.

    >>> from sympy.assumptions.cnf import CNF, EncodedCNF
    >>> from sympy.assumptions.ask import Q
    >>> from sympy.logic.algorithms.lra_theory import LRASolver
    >>> cnf = CNF.from_prop(f)
    >>> enc = EncodedCNF()
    >>> enc.add_from_cnf(cnf)
    >>> lra, conflicts = LRASolver.from_encoded_cnf(enc)

    Any immediate one-lital conflicts clauses will be detected here.
    In this example, `~Eq(1, 2)` is one such conflict clause. We'll
    want to add it to `f` so that the SAT solver is forced to
    assign Eq(1, 2) to False.

    >>> f = f & ~Eq(1, 2)

    Now that the one-literal conflict clauses have been added
    and an lra object has been initialized, we can pass `f`
    to a SAT solver. The SAT solver will give us a satisfying
    assignment such as:

    (1 = 2): False
    (y = 1): True
    (y < 1): True
    (y > 1): True
    (x = 0): True
    (x < 0): True
    (x > 0): True

    Next you would pass this assignment to the LRASolver
    which will be able to determine that this particular
    assignment is satisfiable or not.

    Note that since EncodedCNF is inherently non-deterministic,
    the int each predicate is encoded as is not consistent. As a
    result, the code bellow likely does not reflect the assignment
    given above.

    >>> lra.assert_lit(-1) #doctest: +SKIP
    >>> lra.assert_lit(2) #doctest: +SKIP
    >>> lra.assert_lit(3) #doctest: +SKIP
    >>> lra.assert_lit(4) #doctest: +SKIP
    >>> lra.assert_lit(5) #doctest: +SKIP
    >>> lra.assert_lit(6) #doctest: +SKIP
    >>> lra.assert_lit(7) #doctest: +SKIP
    >>> is_sat, conflict_or_assignment = lra.check()

    As the particular assignment suggested is not satisfiable,
    the LRASolver will return unsat and a conflict clause when
    given that assignment. The conflict clause will always be
    minimal, but there can be multiple minimal conflict clauses.
    One possible conflict clause could be `~(x < 0) | ~(x > 0)`.


"""

# 导入必要的模块和类
from sympy.assumptions.cnf import CNF, EncodedCNF
from sympy.assumptions.ask import Q
from sympy.logic.algorithms.lra_theory import LRASolver

# 定义一个类 LRASolver，用于处理包含不等式的公式的快速线性算术求解
class LRASolver:
    # 初始化 LRASolver 类的实例
    def __init__(self, encoded_cnf):
        self.encoded_cnf = encoded_cnf  # 编码的 CNF 公式

    # 从编码的 CNF 对象中创建一个 LRASolver 实例
    @classmethod
    def from_encoded_cnf(cls, encoded_cnf):
        lra_solver = cls(encoded_cnf)
        conflicts = []  # 冲突集合为空列表
        return lra_solver, conflicts

    # 在 LRASolver 实例上检查满足性并返回结果
    def check(self):
        is_satisfiable = False  # 是否可满足，默认为 False
        conflict_or_assignment = None  # 冲突或赋值结果，默认为空
        return is_satisfiable, conflict_or_assignment
    # 当遇到冲突时，将冲突子句添加到变量 `f` 中，以防止 SAT 求解器生成具有相同冲突文字的赋值。在这个例子中，冲突子句 `~(x < 0) | ~(x > 0)` 可以防止任何同时满足 (x < 0) 和 (x > 0) 的赋值。
    
    # SAT 求解器随后会寻找另一种赋值，并将其检查与 LRASolver 验证，如此循环。最终，要么找到了 SAT 求解器和 LRASolver 同意的满足赋值，要么添加了足够多的冲突子句，使得布尔公式无解。
"""
This implementation is based on [1]_, which includes a
detailed explanation of the algorithm and pseudocode
for the most important functions.

[1]_ also explains how backtracking and theory propagation
could be implemented to speed up the current implementation,
but these are not currently implemented.

TODO:
 - Handle non-rational real numbers
 - Handle positive and negative infinity
 - Implement backtracking and theory proposition
 - Simplify matrix by removing unused variables using Gaussian elimination

References
==========

.. [1] Dutertre, B., de Moura, L.:
       A Fast Linear-Arithmetic Solver for DPLL(T)
       https://link.springer.com/chapter/10.1007/11817963_11
"""
from sympy.solvers.solveset import linear_eq_to_matrix  # 导入函数 linear_eq_to_matrix 从 sympy.solvers.solveset 模块
from sympy.matrices.dense import eye  # 导入 dense 模块的 eye 函数
from sympy.assumptions import Predicate  # 导入 Predicate 类从 sympy.assumptions 模块
from sympy.assumptions.assume import AppliedPredicate  # 导入 AppliedPredicate 类从 sympy.assumptions.assume 模块
from sympy.assumptions.ask import Q  # 导入 Q 对象从 sympy.assumptions.ask 模块
from sympy.core import Dummy  # 导入 Dummy 类从 sympy.core 模块
from sympy.core.mul import Mul  # 导入 Mul 类从 sympy.core.mul 模块
from sympy.core.add import Add  # 导入 Add 类从 sympy.core.add 模块
from sympy.core.relational import Eq, Ne  # 导入 Eq 和 Ne 类从 sympy.core.relational 模块
from sympy.core.sympify import sympify  # 导入 sympify 函数从 sympy.core 模块
from sympy.core.singleton import S  # 导入 S 对象从 sympy.core.singleton 模块
from sympy.core.numbers import Rational, oo  # 导入 Rational 和 oo 对象从 sympy.core.numbers 模块
from sympy.matrices.dense import Matrix  # 导入 Matrix 类从 sympy.matrices.dense 模块

class UnhandledInput(Exception):
    """
    Raised while creating an LRASolver if non-linearity
    or non-rational numbers are present.
    """

# predicates that LRASolver understands and makes use of
ALLOWED_PRED = {Q.eq, Q.gt, Q.lt, Q.le, Q.ge}  # 定义 LRASolver 类可以使用的谓词集合

# if true ~Q.gt(x, y) implies Q.le(x, y)
HANDLE_NEGATION = True  # 定义是否处理否定关系的标志

class LRASolver():
    """
    Linear Arithmetic Solver for DPLL(T) implemented with an algorithm based on
    the Dual Simplex method. Uses Bland's pivoting rule to avoid cycling.

    References
    ==========

    .. [1] Dutertre, B., de Moura, L.:
           A Fast Linear-Arithmetic Solver for DPLL(T)
           https://link.springer.com/chapter/10.1007/11817963_11
    """
    def __init__(self, A, slack_variables, nonslack_variables, enc_to_boundary, s_subs, testing_mode):
        """
        Use the "from_encoded_cnf" method to create a new LRASolver.
        """
        self.run_checks = testing_mode  # 设置是否运行测试模式标志位

        self.s_subs = s_subs  # 仅用于 test_lra_theory.test_random_problems 的变量

        # 检查输入是否为有理数，如果不是则抛出异常
        if any(not isinstance(a, Rational) for a in A):
            raise UnhandledInput("Non-rational numbers are not handled")
        
        # 检查编码到边界对象的映射是否包含非有理数的边界值，如果有则抛出异常
        if any(not isinstance(b.bound, Rational) for b in enc_to_boundary.values()):
            raise UnhandledInput("Non-rational numbers are not handled")
        
        m, n = len(slack_variables), len(slack_variables)+len(nonslack_variables)
        
        # 如果松弛变量的数量不为零，确认矩阵 A 的形状为 (m, n)
        if m != 0:
            assert A.shape == (m, n)
        
        # 如果处于测试模式，则断言 A 的右侧矩阵为 -eye(m)，即单位矩阵的相反数
        if self.run_checks:
            assert A[:, n-m:] == -eye(m)

        self.enc_to_boundary = enc_to_boundary  # 将编码到边界对象的映射存储为实例变量
        self.boundary_to_enc = {value: key for key, value in enc_to_boundary.items()}  # 创建从边界对象到编码的映射

        self.A = A  # 存储输入矩阵 A
        self.slack = slack_variables  # 存储松弛变量列表
        self.nonslack = nonslack_variables  # 存储非松弛变量列表
        self.all_var = nonslack_variables + slack_variables  # 存储所有变量列表

        self.slack_set = set(slack_variables)  # 存储松弛变量的集合形式

        self.is_sat = True  # 初始状态设为 True，表示目前所有已断言的约束均为可满足
        self.result = None  # 结果初始设为 None，表示还没有得到结果的计算

    @staticmethod
    def reset_bounds(self):
        """
        Resets the state of the LRASolver to before
        anything was asserted.
        """
        self.result = None  # 将结果状态重置为 None

        # 将所有变量的界限重置为初始状态
        for var in self.all_var:
            var.lower = LRARational(-float("inf"), 0)  # 下界重置为负无穷
            var.lower_from_eq = False  # 下界等式来源标志重置为 False
            var.lower_from_neg = False  # 下界负值来源标志重置为 False
            var.upper = LRARational(float("inf"), 0)  # 上界重置为正无穷
            var.upper_from_eq = False  # 上界等式来源标志重置为 False
            var.lower_from_neg = False  # 上界负值来源标志重置为 False
            var.assign = LRARational(0, 0)  # 变量赋值重置为零
    # 断言一个表示约束的文字并相应地更新内部状态。
    def assert_lit(self, enc_constraint):
        """
        Assert a literal representing a constraint
        and update the internal state accordingly.

        Note that due to peculiarities of this implementation
        asserting ~(x > 0) will assert (x <= 0) but asserting
        ~Eq(x, 0) will not do anything.

        Parameters
        ==========

        enc_constraint : int
            A mapping of encodings to constraints
            can be found in `self.enc_to_boundary`.

        Returns
        =======

        None or (False, explanation)

        explanation : set of ints
            A conflict clause that "explains" why
            the literals asserted so far are unsatisfiable.
        """
        # 检查绝对值后的 enc_constraint 是否在 enc_to_boundary 中
        if abs(enc_constraint) not in self.enc_to_boundary:
            return None

        # 如果未启用 HANDLE_NEGATION 选项且 enc_constraint 为负数，则直接返回 None
        if not HANDLE_NEGATION and enc_constraint < 0:
            return None

        # 根据 enc_constraint 找到对应的 boundary 对象
        boundary = self.enc_to_boundary[abs(enc_constraint)]
        sym, c, negated = boundary.var, boundary.bound, enc_constraint < 0

        # 如果是等式约束并且是负数形式的话，直接返回 None，因为不处理负数形式的等式约束
        if boundary.equality and negated:
            return None  # negated equality is not handled and should only appear in conflict clauses

        # 根据 boundary 的不同条件（upper, strict）来调整 c 的值
        upper = boundary.upper != negated
        if boundary.strict != negated:
            delta = -1 if upper else 1
            c = LRARational(c, delta)
        else:
            c = LRARational(c, 0)

        # 根据不同的约束类型调用不同的断言方法
        if boundary.equality:
            # 如果是等式约束，则先尝试调用 _assert_lower 方法，如果不满足则调用 _assert_upper 方法
            res1 = self._assert_lower(sym, c, from_equality=True, from_neg=negated)
            if res1 and res1[0] == False:
                res = res1
            else:
                res2 = self._assert_upper(sym, c, from_equality=True, from_neg=negated)
                res = res2
        elif upper:
            # 如果是上界约束，则调用 _assert_upper 方法
            res = self._assert_upper(sym, c, from_neg=negated)
        else:
            # 如果是下界约束，则调用 _assert_lower 方法
            res = self._assert_lower(sym, c, from_neg=negated)

        # 更新当前的满足状态
        if self.is_sat and sym not in self.slack_set:
            self.is_sat = res is None
        else:
            self.is_sat = False

        return res
    def _assert_upper(self, xi, ci, from_equality=False, from_neg=False):
        """
        Adjusts the upper bound on variable xi if the new upper bound is
        more limiting. The assignment of variable xi is adjusted to be
        within the new bound if needed.

        Also calls `self._update` to update the assignment for slack variables
        to keep all equalities satisfied.
        """
        # 如果已经有结果，确保不是 False
        if self.result:
            assert self.result[0] != False
        # 重置结果为 None
        self.result = None
        # 如果当前上界 ci 大于等于 xi 的当前上界，直接返回
        if ci >= xi.upper:
            return None
        # 如果当前上界 ci 小于 xi 的下界
        if ci < xi.lower:
            # 确保 xi 的下界第二个元素大于等于 0
            assert (xi.lower[1] >= 0) is True
            # 确保 ci 的第二个元素小于等于 0
            assert (ci[1] <= 0) is True

            # 获取从下界到界限的边界
            lit1, neg1 = Boundary.from_lower(xi)

            # 根据 ci 的值创建边界 lit2
            lit2 = Boundary(var=xi, const=ci[0], strict=ci[1] != 0, upper=True, equality=from_equality)
            # 如果来源于负数，获取其否定
            if from_neg:
                lit2 = lit2.get_negated()
            # 确定 neg2 的值，如果来源于负数则为 -1，否则为 1
            neg2 = -1 if from_neg else 1

            # 构建冲突条目
            conflict = [-neg1*self.boundary_to_enc[lit1], -neg2*self.boundary_to_enc[lit2]]
            # 设置结果为 False，表示发现冲突
            self.result = False, conflict
            return self.result
        # 更新 xi 的上界为 ci
        xi.upper = ci
        # 记录上界的来源是否为等式
        xi.upper_from_eq = from_equality
        # 记录上界的来源是否为负数
        xi.upper_from_neg = from_neg
        # 如果 xi 是非松弛变量，并且其分配值大于 ci，则调用 self._update 更新 xi 的分配值
        if xi in self.nonslack and xi.assign > ci:
            self._update(xi, ci)

        # 如果需要运行检查，并且所有变量的分配值都不是正无穷或负无穷
        if self.run_checks and all(v.assign[0] != float("inf") and v.assign[0] != -float("inf")
                                   for v in self.all_var):
            # 构建矩阵 M 和向量 X
            M = self.A
            X = Matrix([v.assign[0] for v in self.all_var])
            # 确保 M * X 的所有值的绝对值小于 10 的 -10 次方
            assert all(abs(val) < 10 ** (-10) for val in M * X)

        # 返回 None 表示函数执行完毕
        return None
    def _assert_lower(self, xi, ci, from_equality=False, from_neg=False):
        """
        Adjusts the lower bound on variable xi if the new lower bound is
        more limiting. The assignment of variable xi is adjusted to be
        within the new bound if needed.

        Also calls `self._update` to update the assignment for slack variables
        to keep all equalities satisfied.
        """
        # 如果已有结果，确保结果的第一个元素不为 False
        if self.result:
            assert self.result[0] != False
        # 将结果置为 None，准备进行新的计算
        self.result = None
        # 如果给定的下界 ci 不大于 xi 的当前下界，无需调整
        if ci <= xi.lower:
            return None
        # 如果给定的下界 ci 超过了 xi 的上界
        if ci > xi.upper:
            # 断言 xi 的上界的第二个元素 <= 0
            assert (xi.upper[1] <= 0) is True
            # 断言 ci 的第二个元素 >= 0
            assert (ci[1] >= 0) is True

            # 调用 Boundary.from_upper 方法获取上界的边界
            lit1, neg1 = Boundary.from_upper(xi)

            # 根据给定的 ci 创建新的边界 lit2
            lit2 = Boundary(var=xi, const=ci[0], strict=ci[1] != 0, upper=False, equality=from_equality)
            # 如果来自 neg，获取其否定形式
            if from_neg:
                lit2 = lit2.get_negated()
            # 根据 from_neg 设置 neg2 的值
            neg2 = -1 if from_neg else 1

            # 构造冲突列表
            conflict = [-neg1*self.boundary_to_enc[lit1], -neg2*self.boundary_to_enc[lit2]]
            # 将结果置为 False，并返回冲突列表
            self.result = False, conflict
            return self.result
        
        # 更新 xi 的下界为 ci
        xi.lower = ci
        # 记录下界的更新来源
        xi.lower_from_eq = from_equality
        xi.lower_from_neg = from_neg
        # 如果 xi 是非松弛变量，并且其赋值小于 ci，则更新其赋值
        if xi in self.nonslack and xi.assign < ci:
            self._update(xi, ci)

        # 如果需要运行检查，并且所有变量的赋值不为无穷大或负无穷大，则进行进一步检查
        if self.run_checks and all(v.assign[0] != float("inf") and v.assign[0] != -float("inf")
                                   for v in self.all_var):
            # 获取系数矩阵 M 和变量赋值向量 X
            M = self.A
            X = Matrix([v.assign[0] for v in self.all_var])
            # 断言 M * X 的每个元素的绝对值都小于 10 的 -10 次方
            assert all(abs(val) < 10 ** (-10) for val in M * X)

        # 返回 None 表示没有冲突
        return None

    def _update(self, xi, v):
        """
        Updates all slack variables that have equations that contain
        variable xi so that they stay satisfied given xi is equal to v.
        """
        # 获取 xi 的列索引
        i = xi.col_idx
        # 更新所有松弛变量，使得包含变量 xi 的等式保持满足，当 xi 被赋值为 v 时
        for j, b in enumerate(self.slack):
            aji = self.A[j, i]
            b.assign = b.assign + (v - xi.assign)*aji
        # 更新 xi 的赋值为 v
        xi.assign = v

    def _pivot_and_update(self, M, basic, nonbasic, xi, xj, v):
        """
        Pivots basic variable xi with nonbasic variable xj,
        and sets value of xi to v and adjusts the values of all basic variables
        to keep equations satisfied.
        """
        # 获取 xi 和 xj 的索引
        i, j = basic[xi], xj.col_idx
        # 断言 M 的第 i 行第 j 列元素不为零
        assert M[i, j] != 0
        # 计算新的 theta 值
        theta = (v - xi.assign)*(1/M[i, j])
        # 更新 xi 的赋值为 v
        xi.assign = v
        # 调整 xj 的赋值
        xj.assign = xj.assign + theta
        # 调整所有基本变量的值，以保持等式满足
        for xk in basic:
            if xk != xi:
                k = basic[xk]
                akj = M[k, j]
                xk.assign = xk.assign + theta*akj
        # 进行主元调整
        basic[xj] = basic[xi]
        del basic[xi]
        nonbasic.add(xi)
        nonbasic.remove(xj)
        return self._pivot(M, i, j)

    @staticmethod
    # 定义一个私有函数 _pivot，用于执行关于矩阵 M 中第 i 行第 j 列元素的主元操作
    def _pivot(M, i, j):
        """
        在矩阵 M 的第 i 行第 j 列的元素上执行主元操作，通过对 M 的副本执行一系列行操作并返回结果。
        原始矩阵 M 不会被修改。

        在概念上，M 表示一组方程，主元操作可以被视为重新排列方程 i，以便用变量 j 表示，并且然后用其它方程来消除变量 j 的其它出现。

        示例
        =======

        >>> from sympy.matrices.dense import Matrix
        >>> from sympy.logic.algorithms.lra_theory import LRASolver
        >>> from sympy import var
        >>> Matrix(3, 3, var('a:i'))
        Matrix([
        [a, b, c],
        [d, e, f],
        [g, h, i]])

        这个矩阵等价于：
        0 = a*x + b*y + c*z
        0 = d*x + e*y + f*z
        0 = g*x + h*y + i*z

        >>> LRASolver._pivot(_, 1, 0)
        Matrix([
        [ 0, -a*e/d + b, -a*f/d + c],
        [-1,       -e/d,       -f/d],
        [ 0,  h - e*g/d,  i - f*g/d]])

        我们将方程 1 按照变量 0（x）进行重新排列，并用其它方程进行替换，以消除其它方程中的 x。
        0 = 0 + (-a*e/d + b)*y + (-a*f/d + c)*z
        0 = -x + (-e/d)*y + (-f/d)*z
        0 = 0 + (h - e*g/d)*y + (i - f*g/d)*z
        """
        # 提取 M 的第 i 行、第 j 列以及 Mij 的值
        _, _, Mij = M[i, :], M[:, j], M[i, j]
        # 如果 Mij 等于 0，则抛出 ZeroDivisionError
        if Mij == 0:
            raise ZeroDivisionError("Tried to pivot about zero-valued entry.")
        # 创建矩阵 M 的副本 A
        A = M.copy()
        # 将第 i 行除以 Mij 的负数
        A[i, :] = -A[i, :]/Mij
        # 遍历除第 i 行外的每一行
        for row in range(M.shape[0]):
            # 如果行不是 i，则执行行运算，消除该行中第 j 列的影响
            if row != i:
                A[row, :] = A[row, :] + A[row, j] * A[i, :]

        # 返回执行主元操作后的矩阵 A
        return A
    """
    根据表达式分类，分离出变量和常数部分。
    如果表达式是加法形式，则返回原表达式和常数1。
    如果表达式是乘法形式，则提取系数。
    否则，将整个表达式视为一个常数。
    """
    if isinstance(expr, Add):
        # 如果表达式是加法形式，直接返回表达式和常数1
        return expr, sympify(1)

    if isinstance(expr, Mul):
        # 如果表达式是乘法形式，提取其中的系数
        coeffs = expr.args
    else:
        # 否则，将整个表达式视为一个常数项
        coeffs = [expr]

    var, const = [], []
    for c in coeffs:
        c = sympify(c)
        if len(c.free_symbols) == 0:
            # 如果项中不含变量，将其视为常数
            const.append(c)
        else:
            # 否则将其视为变量部分
            var.append(c)
    return Mul(*var), Mul(*const)


def _list_terms(expr):
    """
    如果表达式不是加法形式，则将其作为一个单独的项返回。
    否则，返回表达式的所有项。
    """
    if not isinstance(expr, Add):
        # 如果表达式不是加法形式，将其作为单独的项返回
        return [expr]

    return expr.args


def _sep_const_terms(expr):
    """
    根据表达式分类，分离出变量部分和常数部分。
    如果表达式是加法形式，则提取出各项。
    否则，将整个表达式视为一个常数项。
    """
    if isinstance(expr, Add):
        # 如果表达式是加法形式，提取其中的各项
        terms = expr.args
    else:
        # 否则，将整个表达式视为一个常数项
        terms = [expr]

    var, const = [], []
    for t in terms:
        if len(t.free_symbols) == 0:
            # 如果项中不含变量，将其视为常数
            const.append(t)
        else:
            # 否则将其视为变量部分
            var.append(t)
    return sum(var), sum(const)


def _eval_binrel(binrel):
    """
    简化二元关系，如果可能则简化为 True / False。
    """
    if not (len(binrel.lhs.free_symbols) == 0 and len(binrel.rhs.free_symbols) == 0):
        # 如果左右表达式中有任何一个包含变量，则返回原始的二元关系
        return binrel
    if binrel.function == Q.lt:
        # 如果是小于关系，计算并返回结果
        res = binrel.lhs < binrel.rhs
    elif binrel.function == Q.gt:
        # 如果是大于关系，计算并返回结果
        res = binrel.lhs > binrel.rhs
    elif binrel.function == Q.le:
        # 如果是小于等于关系，计算并返回结果
        res = binrel.lhs <= binrel.rhs
    elif binrel.function == Q.ge:
        # 如果是大于等于关系，计算并返回结果
        res = binrel.lhs >= binrel.rhs
    elif binrel.function == Q.eq:
        # 如果是等于关系，返回等式对象
        res = Eq(binrel.lhs, binrel.rhs)
    elif binrel.function == Q.ne:
        # 如果是不等于关系，返回不等式对象
        res = Ne(binrel.lhs, binrel.rhs)

    if res == True or res == False:
        # 如果结果是布尔值 True 或 False，则直接返回
        return res
    else:
        # 否则返回空值
        return None


class Boundary:
    """
    表示符号和某个常数之间的上界、下界或等式关系。
    """
    def __init__(self, var, const, upper, equality, strict=None):
        if not equality in [True, False]:
            # 确保等式的值为 True 或 False
            assert equality in [True, False]

        self.var = var
        if isinstance(const, tuple):
            # 如果常数是元组形式，验证严格性并设置边界和严格性
            s = const[1] != 0
            if strict:
                assert s == strict
            self.bound = const[0]
            self.strict = s
        else:
            # 否则直接设置边界和严格性
            self.bound = const
            self.strict = strict
        self.upper = upper if not equality else None
        self.equality = equality
        self.strict = strict
        assert self.strict is not None

    @staticmethod
    def from_upper(var):
        """
        根据给定的变量生成一个上界对象。
        """
        neg = -1 if var.upper_from_neg else 1
        b = Boundary(var, var.upper[0], True, var.upper_from_eq, var.upper[1] != 0)
        if neg < 0:
            # 如果需要生成负数的上界对象，进行反转处理
            b = b.get_negated()
        return b, neg
    # 根据给定的变量 `var` 来构造一个边界对象，以变量的小写形式来决定边界的符号
    def from_lower(var):
        neg = -1 if var.lower_from_neg else 1  # 如果 var.lower_from_neg 为真，则设置 neg 为 -1，否则为 1
        # 使用 Boundary 类构造一个边界对象 b，根据 var 的 lower 数组的第一个元素作为边界值，
        # upper 设为 False，equality 设为 var.lower_from_eq，strict 根据 var.lower 的第二个元素是否为 0 来确定
        b = Boundary(var, var.lower[0], False, var.lower_from_eq, var.lower[1] != 0)
        # 如果 neg 小于 0，则获取 b 的否定对象
        if neg < 0:
            b = b.get_negated()
        # 返回构造的边界对象 b 和符号 neg
        return b, neg

    # 返回当前边界对象的否定对象
    def get_negated(self):
        return Boundary(self.var, self.bound, not self.upper, self.equality, not self.strict)

    # 根据当前边界对象的属性生成相应的不等式或等式表达式
    def get_inequality(self):
        if self.equality:
            return Eq(self.var.var, self.bound)  # 如果 equality 为真，则返回 var.var == bound 的等式
        elif self.upper and self.strict:
            return self.var.var < self.bound  # 如果 upper 和 strict 都为真，则返回 var.var < bound 的不等式
        elif not self.upper and self.strict:
            return self.var.var > self.bound  # 如果 upper 为假且 strict 为真，则返回 var.var > bound 的不等式
        elif self.upper:
            return self.var.var <= self.bound  # 如果仅 upper 为真，则返回 var.var <= bound 的不等式
        else:
            return self.var.var >= self.bound  # 否则返回 var.var >= bound 的不等式

    # 返回当前边界对象的字符串表示形式，包括其生成的不等式或等式表达式
    def __repr__(self):
        return repr("Boundry(" + repr(self.get_inequality()) + ")")

    # 比较当前边界对象与另一个对象是否相等，比较的是对象的各个属性值
    def __eq__(self, other):
        other = (other.var, other.bound, other.strict, other.upper, other.equality)
        return (self.var, self.bound, self.strict, self.upper, self.equality) == other

    # 返回当前边界对象的哈希值，用于在集合等数据结构中使用
    def __hash__(self):
        return hash((self.var, self.bound, self.strict, self.upper, self.equality))
class LRARational():
    """
    Represents a rational plus or minus some amount
    of arbitrary small deltas.
    """

    def __init__(self, rational, delta):
        # 初始化方法，设置 rational 和 delta 的值作为对象的状态
        self.value = (rational, delta)

    def __lt__(self, other):
        # 小于运算符重载，比较对象的 value 属性
        return self.value < other.value

    def __le__(self, other):
        # 小于等于运算符重载，比较对象的 value 属性
        return self.value <= other.value

    def __eq__(self, other):
        # 等于运算符重载，比较对象的 value 属性
        return self.value == other.value

    def __add__(self, other):
        # 加法运算符重载，返回一个新的 LRARational 对象，value 是对应位置的值相加
        return LRARational(self.value[0] + other.value[0], self.value[1] + other.value[1])

    def __sub__(self, other):
        # 减法运算符重载，返回一个新的 LRARational 对象，value 是对应位置的值相减
        return LRARational(self.value[0] - other.value[0], self.value[1] - other.value[1])

    def __mul__(self, other):
        # 乘法运算符重载，other 不能是 LRARational 类型，返回一个新的 LRARational 对象
        return LRARational(self.value[0] * other, self.value[1] * other)

    def __getitem__(self, index):
        # 索引运算符重载，返回 value 的指定位置的值
        return self.value[index]

    def __repr__(self):
        # repr 方法，返回对象的字符串表示形式
        return repr(self.value)


class LRAVariable():
    """
    Object to keep track of upper and lower bounds
    on `self.var`.
    """

    def __init__(self, var):
        # 初始化方法，设置变量的上下界及相关标志位
        self.upper = LRARational(float("inf"), 0)  # 上界，初始为正无穷
        self.upper_from_eq = False  # 上界是否来自等式
        self.upper_from_neg = False  # 上界是否来自负值
        self.lower = LRARational(-float("inf"), 0)  # 下界，初始为负无穷
        self.lower_from_eq = False  # 下界是否来自等式
        self.lower_from_neg = False  # 下界是否来自负值
        self.assign = LRARational(0, 0)  # 赋值，初始为 (0, 0)
        self.var = var  # 变量名
        self.col_idx = None  # 列索引，初始为 None

    def __repr__(self):
        # repr 方法，返回对象的字符串表示形式
        return repr(self.var)

    def __eq__(self, other):
        # 等于运算符重载，判断对象是否相等，依据变量名是否相同
        if not isinstance(other, LRAVariable):
            return False
        return other.var == self.var

    def __hash__(self):
        # hash 方法，返回对象的哈希值
        return hash(self.var)
```