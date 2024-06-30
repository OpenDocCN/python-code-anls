# `D:\src\scipysrc\sympy\sympy\solvers\diophantine\diophantine.py`

```
# 从 __future__ 模块导入 annotations 特性，使得类型注解中可以使用字符串形式的类型名称
from __future__ import annotations

# 从 sympy.core.add 模块导入 Add 类，用于代表 SymPy 表达式中的加法操作
from sympy.core.add import Add

# 从 sympy.core.assumptions 模块导入 check_assumptions 函数，用于检查符号表达式的假设条件
from sympy.core.assumptions import check_assumptions

# 从 sympy.core.containers 模块导入 Tuple 类，用于表示 SymPy 中的元组
from sympy.core.containers import Tuple

# 从 sympy.core.exprtools 模块导入 factor_terms 函数，用于因式分解符号表达式
from sympy.core.exprtools import factor_terms

# 从 sympy.core.function 模块导入 _mexpand 函数，用于执行 SymPy 表达式的乘法展开
from sympy.core.function import _mexpand

# 从 sympy.core.mul 模块导入 Mul 类，用于代表 SymPy 表达式中的乘法操作
from sympy.core.mul import Mul

# 从 sympy.core.numbers 模块导入 Rational 和 int_valued 类，用于处理有理数和整数值
from sympy.core.numbers import Rational, int_valued

# 从 sympy.core.intfunc 模块导入 igcdex, ilcm, igcd, integer_nthroot, isqrt 函数，
# 用于整数的扩展最大公约数、最小公倍数、最大公约数、整数的 n 次根和整数的平方根计算
from sympy.core.intfunc import igcdex, ilcm, igcd, integer_nthroot, isqrt

# 从 sympy.core.relational 模块导入 Eq 类，用于表示 SymPy 中的等式关系
from sympy.core.relational import Eq

# 从 sympy.core.singleton 模块导入 S 对象，用于表示 SymPy 中的单例对象
from sympy.core.singleton import S

# 从 sympy.core.sorting 模块导入 default_sort_key 和 ordered 函数，
# 用于排序和顺序化 SymPy 表达式
from sympy.core.sorting import default_sort_key, ordered

# 从 sympy.core.symbol 模块导入 Symbol 和 symbols 函数，用于表示 SymPy 中的符号和符号的集合
from sympy.core.symbol import Symbol, symbols

# 从 sympy.core.sympify 模块导入 _sympify 函数，用于将输入转换为 SymPy 符号表达式
from sympy.core.sympify import _sympify

# 从 sympy.external.gmpy 模块导入 jacobi, remove, invert, iroot 函数，
# 用于计算雅可比符号、移除元素、求逆元素和整数根
from sympy.external.gmpy import jacobi, remove, invert, iroot

# 从 sympy.functions.elementary.complexes 模块导入 sign 函数，用于计算符号函数的值
from sympy.functions.elementary.complexes import sign

# 从 sympy.functions.elementary.integers 模块导入 floor 函数，用于计算向下取整
from sympy.functions.elementary.integers import floor

# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt 函数，用于计算平方根
from sympy.functions.elementary.miscellaneous import sqrt

# 从 sympy.matrices.dense 模块导入 Matrix 类，用于表示 SymPy 中的密集矩阵
from sympy.matrices.dense import MutableDenseMatrix as Matrix

# 从 sympy.ntheory.factor_ 模块导入 divisors, factorint, perfect_power 函数，
# 用于计算因子、因数分解和完全幂
from sympy.ntheory.factor_ import divisors, factorint, perfect_power

# 从 sympy.ntheory.generate 模块导入 nextprime 函数，用于生成下一个素数
from sympy.ntheory.generate import nextprime

# 从 sympy.ntheory.primetest 模块导入 is_square 和 isprime 函数，用于检测平方数和素数
from sympy.ntheory.primetest import is_square, isprime

# 从 sympy.ntheory.modular 模块导入 symmetric_residue 函数，用于计算对称剩余
from sympy.ntheory.modular import symmetric_residue

# 从 sympy.ntheory.residue_ntheory 模块导入 sqrt_mod 和 sqrt_mod_iter 函数，
# 用于计算模平方根和模平方根的迭代
from sympy.ntheory.residue_ntheory import sqrt_mod, sqrt_mod_iter

# 从 sympy.polys.polyerrors 模块导入 GeneratorsNeeded 异常，用于表示多项式操作需要生成器的错误
from sympy.polys.polyerrors import GeneratorsNeeded

# 从 sympy.polys.polytools 模块导入 Poly 和 factor_list 函数，
# 用于多项式和多项式的因式分解
from sympy.polys.polytools import Poly, factor_list

# 从 sympy.simplify.simplify 模块导入 signsimp 函数，用于简化符号表达式
from sympy.simplify.simplify import signsimp

# 从 sympy.solvers.solveset 模块导入 solveset_real 函数，用于解实数解集合
from sympy.solvers.solveset import solveset_real

# 从 sympy.utilities 模块导入 numbered_symbols 函数，用于生成编号的符号
from sympy.utilities import numbered_symbols

# 从 sympy.utilities.misc 模块导入 as_int 和 filldedent 函数，
# 用于将输入转换为整数和填充缩进
from sympy.utilities.misc import as_int, filldedent

# 从 sympy.utilities.iterables 模块导入 is_sequence, subsets, permute_signs,
# signed_permutations, ordered_partitions 函数，
# 用于检测序列、子集、符号排列、带符号排列和有序分区
from sympy.utilities.iterables import (
    is_sequence, subsets, permute_signs,
    signed_permutations, ordered_partitions
)

# 这些是从 sympy.solvers.diophantine 模块中导入的函数和类
# 使用 '*' 通配符导入所有符号、分类等
__all__ = ['diophantine', 'classify_diop']


class DiophantineSolutionSet(set):
    """
    Container for a set of solutions to a particular diophantine equation.

    The base representation is a set of tuples representing each of the solutions.

    Parameters
    ==========

    symbols : list
        List of free symbols in the original equation.
    parameters: list
        List of parameters to be used in the solution.

    Examples
    ========

    Adding solutions:

        >>> from sympy.solvers.diophantine.diophantine import DiophantineSolutionSet
        >>> from sympy.abc import x, y, t, u
        >>> s1 = DiophantineSolutionSet([x, y], [t, u])
        >>> s1
        set()
        >>> s1.add((2, 3))
        >>> s1.add((-1, u))
        >>> s1
        {(-1, u), (2, 3)}
        >>> s2 = DiophantineSolutionSet([x, y], [t, u])
        >>> s2.add((3, 4))
        >>> s1.update(*s2)
        >>> s1
        {(-1, u), (2, 3), (3, 4)}

    Conversion of solutions into dicts:

        >>> list(s1.dict_iterator())
        [{x: -1, y: u}, {x: 2, y: 3}, {x: 3, y: 4}]
    """
    # 定义一个用于解二次方程的解集的类 DiophantineSolutionSet
    class DiophantineSolutionSet:

        # 初始化方法，接受符号序列和参数序列作为输入
        def __init__(self, symbols_seq, parameters):
            # 调用父类的初始化方法
            super().__init__()

            # 检查符号序列是否为序列类型，如果不是则引发 ValueError 异常
            if not is_sequence(symbols_seq):
                raise ValueError("Symbols must be given as a sequence.")

            # 检查参数序列是否为序列类型，如果不是则引发 ValueError 异常
            if not is_sequence(parameters):
                raise ValueError("Parameters must be given as a sequence.")

            # 将符号序列转换为元组并赋值给 self.symbols
            self.symbols = tuple(symbols_seq)
            # 将参数序列转换为元组并赋值给 self.parameters
            self.parameters = tuple(parameters)

        # 添加解到解集中的方法
        def add(self, solution):
            # 如果解的长度与符号序列的长度不匹配，则引发 ValueError 异常
            if len(solution) != len(self.symbols):
                raise ValueError("Solution should have a length of %s, not %s" % (len(self.symbols), len(solution)))

            # 使解在符号的正负号上是规范化的（即没有 -x，除非 x 也同时出现在解中）
            args = set(solution)
            for i in range(len(solution)):
                x = solution[i]
                # 如果 x 不是整数且其相反数不是符号，并且其相反数不在 args 中，则将解中的 -x 替换为 x
                if not type(x) is int and (-x).is_Symbol and -x not in args:
                    solution = [_.subs(-x, x) for _ in solution]

            # 调用父类的 add 方法，将规范化后的解添加到解集中
            super().add(Tuple(*solution))

        # 更新解集的方法，接受多个解作为参数
        def update(self, *solutions):
            # 对于每个解调用 add 方法，逐一添加到解集中
            for solution in solutions:
                self.add(solution)

        # 返回解集中的解的字典迭代器
        def dict_iterator(self):
            # 使用 ordered 方法对解集中的解进行排序后，逐一生成对应的符号和解的字典
            for solution in ordered(self):
                yield dict(zip(self.symbols, solution))

        # 对解集中的每个解应用指定的替换，返回一个新的解集对象
        def subs(self, *args, **kwargs):
            # 创建一个新的 DiophantineSolutionSet 对象，使用当前的符号序列和参数序列
            result = DiophantineSolutionSet(self.symbols, self.parameters)
            # 对当前解集中的每个解，应用替换 args 和 kwargs，然后添加到新的解集对象中
            for solution in self:
                result.add(solution.subs(*args, **kwargs))
            return result

        # 对解集对象进行调用时的方法，用于对解集中的解进行求值
        def __call__(self, *args):
            # 如果传入的参数个数超过了参数序列的长度，则引发 ValueError 异常
            if len(args) > len(self.parameters):
                raise ValueError("Evaluation should have at most %s values, not %s" % (len(self.parameters), len(args)))

            # 创建一个字典 rep，用参数序列和传入的参数 args 组成
            rep = {p: v for p, v in zip(self.parameters, args) if v is not None}
            # 对当前解集应用替换 rep，并返回结果
            return self.subs(rep)
class DiophantineEquationType:
    """
    Internal representation of a particular diophantine equation type.

    Parameters
    ==========

    equation :
        The diophantine equation that is being solved.
    free_symbols : list (optional)
        The symbols being solved for.

    Attributes
    ==========

    total_degree :
        The maximum of the degrees of all terms in the equation
    homogeneous :
        Does the equation contain a term of degree 0
    homogeneous_order :
        Does the equation contain any coefficient that is in the symbols being solved for
    dimension :
        The number of symbols being solved for
    """
    name = None  # type: str  # 类型注释：该类的名称，初始为None

    def __init__(self, equation, free_symbols=None):
        self.equation = _sympify(equation).expand(force=True)  # 将传入的方程式转换为SymPy表示，并展开

        if free_symbols is not None:
            self.free_symbols = free_symbols  # 如果提供了自由符号列表，则使用它
        else:
            self.free_symbols = list(self.equation.free_symbols)  # 否则从方程式中提取自由符号
            self.free_symbols.sort(key=default_sort_key)  # 根据默认排序键对符号进行排序

        if not self.free_symbols:
            raise ValueError('equation should have 1 or more free symbols')  # 如果没有自由符号，则抛出错误

        self.coeff = self.equation.as_coefficients_dict()  # 将方程式转换为系数字典
        if not all(int_valued(c) for c in self.coeff.values()):
            raise TypeError("Coefficients should be Integers")  # 确保所有系数都是整数类型

        self.total_degree = Poly(self.equation).total_degree()  # 计算方程式的总次数
        self.homogeneous = 1 not in self.coeff  # 检查方程式是否为齐次的（是否缺少常数项）
        self.homogeneous_order = not (set(self.coeff) & set(self.free_symbols))  # 检查方程式是否为同次的（是否所有系数都不在自由符号中）
        self.dimension = len(self.free_symbols)  # 计算自由符号的数量
        self._parameters = None  # 参数初始化为None

    def matches(self):
        """
        Determine whether the given equation can be matched to the particular equation type.
        """
        return False  # 默认返回False，表示不匹配任何特定的方程类型

    @property
    def n_parameters(self):
        return self.dimension  # 返回参数的数量，即自由符号的数量

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = symbols('t_:%i' % (self.n_parameters,), integer=True)  # 如果参数尚未设置，则生成参数符号列表
        return self._parameters  # 返回参数符号列表

    def solve(self, parameters=None, limit=None) -> DiophantineSolutionSet:
        raise NotImplementedError('No solver has been written for %s.' % self.name)  # 抛出未实现错误信息

    def pre_solve(self, parameters=None):
        if not self.matches():
            raise ValueError("This equation does not match the %s equation type." % self.name)  # 如果不匹配特定方程类型，则抛出值错误

        if parameters is not None:
            if len(parameters) != self.n_parameters:
                raise ValueError("Expected %s parameter(s) but got %s" % (self.n_parameters, len(parameters)))  # 如果参数数量不匹配，则抛出值错误

        self._parameters = parameters  # 设置参数
    # 导入 sympy 库中的 x 符号
    >>> from sympy.abc import x
    # 解方程 (x - 2)*(x - 3)**2 == 0，返回其解集合
    >>> Univariate((x - 2)*(x - 3)**2).solve()
    # 解集合包含了方程的根 (2,) 和 (3,)
    {(2,), (3,)}

    """

    # 定义类名为 'univariate'
    name = 'univariate'

    # 定义类方法 matches，用于检查维度是否为 1
    def matches(self):
        return self.dimension == 1

    # 定义 solve 方法，用于解方程
    def solve(self, parameters=None, limit=None):
        # 调用 pre_solve 方法，准备解方程的参数
        self.pre_solve(parameters)

        # 创建 DiophantineSolutionSet 对象，初始化结果集合
        result = DiophantineSolutionSet(self.free_symbols, parameters=self.parameters)
        
        # 遍历 solveset_real 方法得到的实数解，且这些解是整数
        for i in solveset_real(self.equation, self.free_symbols[0]).intersect(S.Integers):
            # 将解添加到结果集合中
            result.add((i,))
        
        # 返回最终的结果集合
        return result
```python`
class Linear(DiophantineEquationType):
    """
    Representation of a linear diophantine equation.

    A linear diophantine equation is an equation of the form `a_{1}x_{1} +
    a_{2}x_{2} + .. + a_{n}x_{n} = 0` where `a_{1}, a_{2}, ..a_{n}` are
    integer constants and `x_{1}, x_{2}, ..x_{n}` are integer variables.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import Linear
    >>> from sympy.abc import x, y, z
    >>> l1 = Linear(2*x - 3*y - 5)
    >>> l1.matches() # is this equation linear
    True
    >>> l1.solve() # solves equation 2*x - 3*y - 5 == 0
    {(3*t_0 - 5, 2*t_0 - 5)}

    Here x = -3*t_0 - 5 and y = -2*t_0 - 5

    >>> Linear(2*x - 3*y - 4*z -3).solve()
    {(t_0, 2*t_0 + 4*t_1 + 3, -t_0 - 3*t_1 - 3)}

    """

    name = 'linear'  # 定义线性方程的名称

    def matches(self):
        return self.total_degree == 1  # 检查方程的总次数是否为1，判断其是否为线性方程

class BinaryQuadratic(DiophantineEquationType):
    """
    Representation of a binary quadratic diophantine equation.

    A binary quadratic diophantine equation is an equation of the
    form `Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0`, where `A, B, C, D, E,
    F` are integer constants and `x` and `y` are integer variables.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.solvers.diophantine.diophantine import BinaryQuadratic
    >>> b1 = BinaryQuadratic(x**3 + y**2 + 1)
    >>> b1.matches()
    False
    >>> b2 = BinaryQuadratic(x**2 + y**2 + 2*x + 2*y + 2)
    >>> b2.matches()
    True
    >>> b2.solve()
    {(-1, -1)}

    References
    ==========

    .. [1] Methods to solve Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0, [online],
          Available: https://www.alpertron.com.ar/METHODS.HTM
    .. [2] Solving the equation ax^2+ bxy + cy^2 + dx + ey + f= 0, [online],
          Available: https://web.archive.org/web/20160323033111/http://www.jpr2718.org/ax2p.pdf

    """

    name = 'binary_quadratic'  # 定义二次二元方程的名称

    def matches(self):
        return self.total_degree == 2 and self.dimension == 2  # 检查方程的总次数是否为2，并且维度是否为2

class InhomogeneousTernaryQuadratic(DiophantineEquationType):
    """

    Representation of an inhomogeneous ternary quadratic.

    No solver is currently implemented for this equation type.

    """

    name = 'inhomogeneous_ternary_quadratic'  # 定义非齐次三次二次方程的名称

    def matches(self):
        if not (self.total_degree == 2 and self.dimension == 3):
            return False  # 检查方程的总次数是否为2，并且维度是否为3，不满足条件返回False
        if not self.homogeneous:
            return False  # 检查方程是否为齐次方程，不是齐次方程返回False
        return not self.homogeneous_order  # 检查齐次方程的顺序，若不满足条件返回False

class HomogeneousTernaryQuadraticNormal(DiophantineEquationType):
    """
    Representation of a homogeneous ternary quadratic normal diophantine equation.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import HomogeneousTernaryQuadraticNormal
    >>> HomogeneousTernaryQuadraticNormal(4*x**2 - 5*y**2 + z**2).solve()
    {(1, 2, 4)}

    """

    name = 'homogeneous_ternary_quadratic_normal'  # 定义齐次三次二次方程的名称
    # 判断是否符合条件：总次数为2且维度为3
    def matches(self):
        if not (self.total_degree == 2 and self.dimension == 3):
            return False
        # 如果不是齐次的，返回 False
        if not self.homogeneous:
            return False
        # 如果没有齐次次数，返回 False
        if not self.homogeneous_order:
            return False

        # 找出非零系数对应的键值列表
        nonzero = [k for k in self.coeff if self.coeff[k]]
        # 如果非零系数的个数不为3或者所有自由符号的平方都在非零系数中，则返回 True
        return len(nonzero) == 3 and all(i**2 in nonzero for i in self.free_symbols)

    # 解 Diophantine 方程组，返回 DiophantineSolutionSet 对象
    def solve(self, parameters=None, limit=None) -> DiophantineSolutionSet:
        # 预处理参数
        self.pre_solve(parameters)

        # 提取自由符号和系数
        var = self.free_symbols
        coeff = self.coeff

        # 将自由符号分配给 x, y, z
        x, y, z = var

        # 分别获取系数 a, b, c
        a = coeff[x**2]
        b = coeff[y**2]
        c = coeff[z**2]

        # 进行平方因式分解规范化，得到三个因式的平方、系数对应的一组值、另一组值
        (sqf_of_a, sqf_of_b, sqf_of_c), (a_1, b_1, c_1), (a_2, b_2, c_2) = \
            sqf_normal(a, b, c, steps=True)

        # 计算 A 和 B
        A = -a_2*c_2
        B = -b_2*c_2

        # 创建 DiophantineSolutionSet 对象
        result = DiophantineSolutionSet(var, parameters=self.parameters)

        # 如果 A 和 B 都小于 0，则返回结果对象
        if A < 0 and B < 0:
            return result

        # 如果无法取平方根模，也返回结果对象
        if (
            sqrt_mod(-b_2*c_2, a_2) is None or
            sqrt_mod(-c_2*a_2, b_2) is None or
            sqrt_mod(-a_2*b_2, c_2) is None):
            return result

        # 利用下降法求解初始解 (z_0, x_0, y_0)
        z_0, x_0, y_0 = descent(A, B)

        # 计算有理数比 q 和 z_0
        z_0, q = _rational_pq(z_0, abs(c_2))
        x_0 *= q
        y_0 *= q

        # 去除 x_0, y_0, z_0 的最大公因数
        x_0, y_0, z_0 = _remove_gcd(x_0, y_0, z_0)

        # 根据符号执行 Holzer 缩减
        if sign(a) == sign(b):
            x_0, y_0, z_0 = holzer(x_0, y_0, z_0, abs(a_2), abs(b_2), abs(c_2))
        elif sign(a) == sign(c):
            x_0, z_0, y_0 = holzer(x_0, z_0, y_0, abs(a_2), abs(c_2), abs(b_2))
        else:
            y_0, z_0, x_0 = holzer(y_0, z_0, x_0, abs(b_2), abs(c_2), abs(a_2))

        # 重构 x_0, y_0, z_0
        x_0 = reconstruct(b_1, c_1, x_0)
        y_0 = reconstruct(a_1, c_1, y_0)
        z_0 = reconstruct(a_1, b_1, z_0)

        # 计算平方的最小公倍数
        sq_lcm = ilcm(sqf_of_a, sqf_of_b, sqf_of_c)

        # 根据公式计算 x_0, y_0, z_0 的值
        x_0 = abs(x_0*sq_lcm // sqf_of_a)
        y_0 = abs(y_0*sq_lcm // sqf_of_b)
        z_0 = abs(z_0*sq_lcm // sqf_of_c)

        # 添加处理过的 x_0, y_0, z_0 到结果集
        result.add(_remove_gcd(x_0, y_0, z_0))
        return result
class HomogeneousTernaryQuadratic(DiophantineEquationType):
    """
    Representation of a homogeneous ternary quadratic diophantine equation.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import HomogeneousTernaryQuadratic
    >>> HomogeneousTernaryQuadratic(x**2 + y**2 - 3*z**2 + x*y).solve()
    {(-1, 2, 1)}
    >>> HomogeneousTernaryQuadratic(3*x**2 + y**2 - 3*z**2 + 5*x*y + y*z).solve()
    {(3, 12, 13)}

    """

    name = 'homogeneous_ternary_quadratic'

    def matches(self):
        # 检查方程的总次数和维度是否符合条件
        if not (self.total_degree == 2 and self.dimension == 3):
            return False
        # 检查方程是否是齐次的
        if not self.homogeneous:
            return False
        # 检查齐次方程的阶数是否存在
        if not self.homogeneous_order:
            return False

        # 找出非零系数对应的变量，并检查是否为自由符号的平方
        nonzero = [k for k in self.coeff if self.coeff[k]]
        return not (len(nonzero) == 3 and all(i**2 in nonzero for i in self.free_symbols))


class InhomogeneousGeneralQuadratic(DiophantineEquationType):
    """
    Representation of an inhomogeneous general quadratic.

    No solver is currently implemented for this equation type.

    """

    name = 'inhomogeneous_general_quadratic'

    def matches(self):
        # 检查方程的总次数和维度是否符合条件
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        # 检查非齐次方程的阶数是否存在
        if not self.homogeneous_order:
            return True
        # 检查方程是否包含乘积项
        return any(k.is_Mul for k in self.coeff) and not self.homogeneous


class HomogeneousGeneralQuadratic(DiophantineEquationType):
    """
    Representation of a homogeneous general quadratic.

    No solver is currently implemented for this equation type.

    """

    name = 'homogeneous_general_quadratic'

    def matches(self):
        # 检查方程的总次数和维度是否符合条件
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        # 检查齐次方程的阶数是否存在
        if not self.homogeneous_order:
            return False
        # 检查方程是否包含乘积项
        return any(k.is_Mul for k in self.coeff) and self.homogeneous


class GeneralSumOfSquares(DiophantineEquationType):
    r"""
    Representation of the diophantine equation

    `x_{1}^2 + x_{2}^2 + . . . + x_{n}^2 - k = 0`.

    Details
    =======

    When `n = 3` if `k = 4^a(8m + 7)` for some `a, m \in Z` then there will be
    no solutions. Refer [1]_ for more details.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import GeneralSumOfSquares
    >>> from sympy.abc import a, b, c, d, e
    >>> GeneralSumOfSquares(a**2 + b**2 + c**2 + d**2 + e**2 - 2345).solve()
    {(15, 22, 22, 24, 24)}

    By default only 1 solution is returned. Use the `limit` keyword for more:

    >>> sorted(GeneralSumOfSquares(a**2 + b**2 + c**2 + d**2 + e**2 - 2345).solve(limit=3))
    [(15, 22, 22, 24, 24), (16, 19, 24, 24, 24), (16, 20, 22, 23, 26)]

    References
    ==========
    # 定义一个名为 'general_sum_of_squares' 的类

    def matches(self):
        # 检查总次数是否为 2 并且维度是否大于等于 3
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        # 检查是否为齐次方程
        if not self.homogeneous_order:
            return False
        # 检查是否存在乘法项
        if any(k.is_Mul for k in self.coeff):
            return False
        # 检查所有系数是否为 1，除了键为 1 的项
        return all(self.coeff[k] == 1 for k in self.coeff if k != 1)

    def solve(self, parameters=None, limit=1):
        # 预处理求解参数
        self.pre_solve(parameters)

        # 自由符号集合
        var = self.free_symbols
        # 系数 -k
        k = -int(self.coeff[1])
        # 维度 n
        n = self.dimension

        # 创建一个 DiophantineSolutionSet 对象作为结果集合
        result = DiophantineSolutionSet(var, parameters=self.parameters)

        # 如果 k 小于 0 或者 limit 小于 1，直接返回结果集合
        if k < 0 or limit < 1:
            return result

        # 确定符号列表，根据自由符号的非正性确定取值为 -1 或者 1
        signs = [-1 if x.is_nonpositive else 1 for x in var]
        # 是否有负数解
        negs = signs.count(-1) != 0

        # 记录已处理的解的数量
        took = 0
        # 遍历求解 k = -k, 维度为 n 的平方和解
        for t in sum_of_squares(k, n, zeros=True):
            # 如果有负数解，按符号调整解
            if negs:
                result.add([signs[i]*j for i, j in enumerate(t)])
            else:
                result.add(t)
            took += 1
            # 达到限制的解的数量，停止添加解
            if took == limit:
                break
        # 返回结果集合
        return result
class GeneralPythagorean(DiophantineEquationType):
    """
    Representation of the general pythagorean equation,
    `a_{1}^2x_{1}^2 + a_{2}^2x_{2}^2 + . . . + a_{n}^2x_{n}^2 - a_{n + 1}^2x_{n + 1}^2 = 0`.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import GeneralPythagorean
    >>> from sympy.abc import a, b, c, d, e, x, y, z, t
    >>> GeneralPythagorean(a**2 + b**2 + c**2 - d**2).solve()
    {(t_0**2 + t_1**2 - t_2**2, 2*t_0*t_2, 2*t_1*t_2, t_0**2 + t_1**2 + t_2**2)}
    >>> GeneralPythagorean(9*a**2 - 4*b**2 + 16*c**2 + 25*d**2 + e**2).solve(parameters=[x, y, z, t])
    {(-10*t**2 + 10*x**2 + 10*y**2 + 10*z**2, 15*t**2 + 15*x**2 + 15*y**2 + 15*z**2, 15*t*x, 12*t*y, 60*t*z)}
    """

    name = 'general_pythagorean'

    # 检查当前方程是否符合一般的勾股数方程形式
    def matches(self):
        # 检查总次数为2且维度至少为3
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        # 检查是否为齐次方程
        if not self.homogeneous_order:
            return False
        # 检查系数中是否包含乘积形式
        if any(k.is_Mul for k in self.coeff):
            return False
        # 检查除了1之外的系数是否都是1
        if all(self.coeff[k] == 1 for k in self.coeff if k != 1):
            return False
        # 检查所有系数的绝对值是否是完全平方数
        if not all(is_square(abs(self.coeff[k])) for k in self.coeff):
            return False
        # 检查除了一个系数之外，所有系数的符号是否相同
        # 例如，4*x**2 + y**2 - 4*z**2
        return abs(sum(sign(self.coeff[k]) for k in self.coeff)) == self.dimension - 2

    @property
    # 返回方程的参数数量
    def n_parameters(self):
        return self.dimension - 1

    # 解决方程
    def solve(self, parameters=None, limit=1):
        self.pre_solve(parameters)

        coeff = self.coeff  # 获取方程的系数
        var = self.free_symbols  # 获取自由符号（变量）
        n = self.dimension  # 获取维度

        # 如果方程中某些二次项系数为负数，则取相反数
        if sign(coeff[var[0] ** 2]) + sign(coeff[var[1] ** 2]) + sign(coeff[var[2] ** 2]) < 0:
            for key in coeff.keys():
                coeff[key] = -coeff[key]

        result = DiophantineSolutionSet(var, parameters=self.parameters)  # 创建解集对象

        index = 0

        # 找出二次项系数为负数的变量索引
        for i, v in enumerate(var):
            if sign(coeff[v ** 2]) == -1:
                index = i

        m = result.parameters  # 获取解集的参数

        ith = sum(m_i ** 2 for m_i in m)  # 计算 m 的平方和
        L = [ith - 2 * m[n - 2] ** 2]  # 计算列表 L 的第一个元素
        L.extend([2 * m[i] * m[n - 2] for i in range(n - 2)])  # 扩展列表 L

        sol = L[:index] + [ith] + L[index:]  # 创建解向量 sol

        lcm = 1

        # 计算最小公倍数
        for i, v in enumerate(var):
            if i == index or (index > 0 and i == 0) or (index == 0 and i == 1):
                lcm = ilcm(lcm, sqrt(abs(coeff[v ** 2])))
            else:
                s = sqrt(coeff[v ** 2])
                lcm = ilcm(lcm, s if _odd(s) else s // 2)

        # 调整解向量的每个元素
        for i, v in enumerate(var):
            sol[i] = (lcm * sol[i]) / sqrt(abs(coeff[v ** 2]))

        result.add(sol)  # 将解向量添加到解集中
        return result


class CubicThue(DiophantineEquationType):
    """
    Representation of a cubic Thue diophantine equation.

    A cubic Thue diophantine equation is a polynomial of the form
    `f(x, y) = r` of degree 3, where `x` and `y` are integers
    and `r` is a rational number.

    No solver is currently implemented for this equation type.
    """

    # 立方图方程的表示，暂未实现解法
    Examples
    ========

    >>> from sympy.abc import x, y         # 导入 sympy 库中的符号 x 和 y
    >>> from sympy.solvers.diophantine.diophantine import CubicThue  # 导入 CubicThue 类
    >>> c1 = CubicThue(x**3 + y**2 + 1)    # 创建一个 CubicThue 对象，传入 x^3 + y^2 + 1 作为参数
    >>> c1.matches()                       # 调用 CubicThue 对象的 matches 方法
    True                                   # 返回匹配结果为 True

    """

    name = 'cubic_thue'                    # 定义一个字符串变量 name，赋值为 'cubic_thue'

    def matches(self):
        return self.total_degree == 3 and self.dimension == 2  # 检查对象的 total_degree 是否为 3，dimension 是否为 2
class GeneralSumOfEvenPowers(DiophantineEquationType):
    """
    Representation of the diophantine equation

    `x_{1}^e + x_{2}^e + . . . + x_{n}^e - k = 0`

    where `e` is an even, integer power.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import GeneralSumOfEvenPowers
    >>> from sympy.abc import a, b
    >>> GeneralSumOfEvenPowers(a**4 + b**4 - (2**4 + 3**4)).solve()
    {(2, 3)}

    """

    name = 'general_sum_of_even_powers'  # 定义此类的名称

    def matches(self):
        if not self.total_degree > 3:  # 若总次数不大于3，则返回False
            return False
        if self.total_degree % 2 != 0:  # 若总次数不是偶数，则返回False
            return False
        if not all(k.is_Pow and k.exp == self.total_degree for k in self.coeff if k != 1):  # 如果系数不是1的幂次项不匹配总次数，则返回False
            return False
        return all(self.coeff[k] == 1 for k in self.coeff if k != 1)  # 如果系数为1的幂次项满足条件，则返回True

    def solve(self, parameters=None, limit=1):
        self.pre_solve(parameters)  # 预处理参数

        var = self.free_symbols  # 获取自由符号
        coeff = self.coeff  # 获取系数

        p = None
        for q in coeff.keys():  # 遍历系数的键
            if q.is_Pow and coeff[q]:  # 如果键是幂次项且系数不为零
                p = q.exp  # 获取幂次

        k = len(var)  # 变量数
        n = -coeff[1]  # 常数项的负值

        result = DiophantineSolutionSet(var, parameters=self.parameters)  # 创建解集对象

        if n < 0 or limit < 1:  # 如果常数项小于零或限制小于1
            return result  # 返回结果集

        sign = [-1 if x.is_nonpositive else 1 for x in var]  # 符号列表，根据自由符号的非正性确定符号
        negs = sign.count(-1) != 0  # 是否存在负数

        took = 0
        for t in power_representation(n, p, k):  # 遍历幂表示的生成器
            if negs:
                result.add([sign[i]*j for i, j in enumerate(t)])  # 将解加入结果集，根据符号确定符号
            else:
                result.add(t)  # 将解加入结果集
            took += 1
            if took == limit:  # 如果达到限制数
                break
        return result  # 返回结果集


# these types are known (but not necessarily handled)
# note that order is important here (in the current solver state)
all_diop_classes = [
    Linear,
    Univariate,
    BinaryQuadratic,
    InhomogeneousTernaryQuadratic,
    HomogeneousTernaryQuadraticNormal,
    HomogeneousTernaryQuadratic,
    InhomogeneousGeneralQuadratic,
    HomogeneousGeneralQuadratic,
    GeneralSumOfSquares,
    GeneralPythagorean,
    CubicThue,
    GeneralSumOfEvenPowers,  # 已知的解类型列表，包括偶数次幂求和类型

]

diop_known = {diop_class.name for diop_class in all_diop_classes}  # 已知的解类型名称集合


def _remove_gcd(*x):
    try:
        g = igcd(*x)  # 计算参数的最大公约数
    except ValueError:
        fx = list(filter(None, x))
        if len(fx) < 2:
            return x
        g = igcd(*[i.as_content_primitive()[0] for i in fx])  # 计算非空参数的最大公约数
    except TypeError:
        raise TypeError('_remove_gcd(a,b,c) or _remove_gcd(*container)')  # 类型错误时抛出异常
    if g == 1:
        return x
    return tuple([i//g for i in x])  # 返回去除最大公约数后的参数元组


def _rational_pq(a, b):
    # return `(numer, denom)` for a/b; sign in numer and gcd removed
    return _remove_gcd(sign(b)*a, abs(b))  # 返回 a/b 的有理数元组，移除符号和最大公约数


def _nint_or_floor(p, q):
    # return nearest int to p/q; in case of tie return floor(p/q)
    w, r = divmod(p, q)  # 计算 p/q 的商和余数
    if abs(r) <= abs(q)//2:  # 如果余数的绝对值小于等于分母的一半
        return w  # 返回商
    return w + 1  # 返回商加一


def _odd(i):
    return i % 2 != 0  # 判断是否为奇数


def _even(i):
    return i % 2 == 0  # 判断是否为偶数
# 定义解二次丢番图方程的函数 diophantine，接受以下参数：
# - eq：要解决的丢番图方程
# - param：可选参数，默认为整数符号"t"，在解过程中使用
# - syms：可选参数，符号列表，决定返回的元组中变量的顺序
# - permute：布尔值参数，控制是否返回基本解的排列组合和符号的排列组合

"""
简化解丢番图方程 `eq` 的过程，将其转换为一系列乘积项，这些项的乘积应该为零。

解释
====

例如，对于解 `x^2 - y^2 = 0`，将其看作 `(x + y)(x - y) = 0`，然后分别求解 `x + y = 0` 和 `x - y = 0`，最后将结果组合起来。每个项通过调用 `diop_solve()` 来解决。（虽然可以直接调用 `diop_solve()`，但必须小心传递正确形式的方程并正确解释输出；`diophantine()` 是一般情况下要使用的公共函数。）

`diophantine()` 的输出是一组元组。元组的元素是方程中每个变量的解，按照变量名称的字母顺序排列。
例如，对于包含两个变量 `a` 和 `b` 的方程，元组的第一个元素是 `a` 的解，第二个元素是 `b` 的解。

用法
====

`diophantine(eq, t, syms)`: 解丢番图方程 `eq`。
`t` 是可选参数，用于 `diop_solve()`。
`syms` 是一个可选的符号列表，决定返回的元组中元素的顺序。

默认情况下，仅返回基本解。如果将 `permute` 设置为 True，则在适用时返回基本解的排列组合和/或值的符号排列组合。

详细说明
========

`eq` 应该是一个假定为零的表达式。
`t` 是在解中要使用的参数。

示例
====

>>> from sympy import diophantine
>>> from sympy.abc import a, b
>>> eq = a**4 + b**4 - (2**4 + 3**4)
>>> diophantine(eq)
{(2, 3)}
>>> diophantine(eq, permute=True)
{(-3, -2), (-3, 2), (-2, -3), (-2, 3), (2, -3), (2, 3), (3, -2), (3, 2)}

>>> from sympy.abc import x, y, z
>>> diophantine(x**2 - y**2)
{(t_0, -t_0), (t_0, t_0)}

>>> diophantine(x*(2*x + 3*y - z))
{(0, n1, n2), (t_0, t_1, 2*t_0 + 3*t_1)}
>>> diophantine(x**2 + 3*x*y + 4*x)
{(0, n1), (-3*t_0 - 4, t_0)}

参见
====

diop_solve
sympy.utilities.iterables.permute_signs
sympy.utilities.iterables.signed_permutations
"""

eq = _sympify(eq)

if isinstance(eq, Eq):
    eq = eq.lhs - eq.rhs
    try:
        # 将表达式中的变量展开，并找出自由符号
        var = list(eq.expand(force=True).free_symbols)
        # 对变量进行排序
        var.sort(key=default_sort_key)
        
        # 如果给定了syms参数，确保其为序列类型（如列表）
        if syms:
            if not is_sequence(syms):
                # 如果syms不是序列类型，则抛出类型错误
                raise TypeError(
                    'syms should be given as a sequence, e.g. a list')
            # 筛选出在变量列表中存在的符号
            syms = [i for i in syms if i in var]
            # 如果syms与var不相等，则创建符号到索引的映射字典
            if syms != var:
                dict_sym_index = dict(zip(syms, range(len(syms))))
                # 调用diophantine函数，返回符合条件的解集合
                return {tuple([t[dict_sym_index[i]] for i in var])
                            for t in diophantine(eq, param, permute=permute)}
        
        # 将方程eq分解为分子n和分母d
        n, d = eq.as_numer_denom()
        
        # 如果n为数值，则返回空集合
        if n.is_number:
            return set()
        
        # 如果d不是数值，则求解d的整数解dsol，并找出n的整数解good
        if not d.is_number:
            dsol = diophantine(d)
            good = diophantine(n) - dsol
            return {s for s in good if _mexpand(d.subs(zip(var, s)))}
        
        # 对n进行因式分解
        eq = factor_terms(n)
        # 断言eq不是数值
        assert not eq.is_number
        # 将eq分解为独立的部分，并取第二部分作为eq
        eq = eq.as_independent(*var, as_Add=False)[1]
        # 将eq转换为多项式p
        p = Poly(eq)
        # 断言p中的所有生成元都不是数值
        assert not any(g.is_number for g in p.gens)
        # 将eq重新赋值为p的表达式形式
        eq = p.as_expr()
        # 断言eq是多项式
        assert eq.is_polynomial()
    
    except (GeneratorsNeeded, AssertionError):
        # 捕获GeneratorsNeeded或AssertionError异常，抛出类型错误，说明方程应为有理系数的多项式
        raise TypeError(filldedent('''
    Equation should be a polynomial with Rational coefficients.'''))

    # 是否只对符号进行置换
    do_permute_signs = False
    # 是否对符号和值进行置换
    do_permute_signs_var = False
    # 是否对少量符号进行置换
    permute_few_signs = False
    
    except (TypeError, NotImplementedError):
        # 捕获TypeError或NotImplementedError异常
        # 对方程进行因式分解，获取因式列表
        fl = factor_list(eq)
        # 如果第一个因式是有理数且不等于1，则递归调用diophantine函数，排除该因子后求解
        if fl[0].is_Rational and fl[0] != 1:
            return diophantine(eq/fl[0], param=param, syms=syms, permute=permute)
        # 取出其他因式列表项
        terms = fl[1]

    # 初始化解集合
    sols = set()

    # 遍历每个因式项
    for term in terms:
        # 提取基础部分和指数
        base, _ = term
        # 分类处理基础部分，确定变量类型和方程类型
        var_t, _, eq_type = classify_diop(base, _dict=False)
        # 简化处理基础部分的符号，并提取其系数
        _, base = signsimp(base, evaluate=False).as_coeff_Mul()
        # 解方程base，获取解集合solution
        solution = diop_solve(base, param)

        # 根据方程类型，将解集合合并到sols中
        if eq_type in [
                Linear.name,
                HomogeneousTernaryQuadratic.name,
                HomogeneousTernaryQuadraticNormal.name,
                GeneralPythagorean.name]:
            sols.add(merge_solution(var, var_t, solution))

        elif eq_type in [
                BinaryQuadratic.name,
                GeneralSumOfSquares.name,
                GeneralSumOfEvenPowers.name,
                Univariate.name]:
            sols.update(merge_solution(var, var_t, sol) for sol in solution)

        else:
            # 如果方程类型未被处理，则抛出NotImplementedError
            raise NotImplementedError('unhandled type: %s' % eq_type)

    # 移除空解结果
    if () in sols:
        sols.remove(())
    # 创建全零解
    null = tuple([0]*len(var))
    # 如果没有解且eq在var取全零时为零，则添加全零解到sols中
    if not sols and eq.subs(zip(var, null)).is_zero:
        sols.add(null)
    # 初始化最终解集合
    final_soln = set()
    # 遍历解集合 sols 中的每一个解 sol
    for sol in sols:
        # 检查解 sol 中的所有元素是否都是整数值
        if all(int_valued(s) for s in sol):
            # 如果需要对符号进行排列组合
            if do_permute_signs:
                # 对解 sol 进行符号排列组合，并将结果添加到最终解集合 final_soln 中
                permuted_sign = set(permute_signs(sol))
                final_soln.update(permuted_sign)
            # 如果需要仅排列少量符号
            elif permute_few_signs:
                # 对解 sol 进行符号排列组合，然后过滤出符合条件的结果，并将结果添加到最终解集合 final_soln 中
                lst = list(permute_signs(sol))
                lst = list(filter(lambda x: x[0]*x[1] == sol[1]*sol[0], lst))
                permuted_sign = set(lst)
                final_soln.update(permuted_sign)
            # 如果需要对符号和变量同时进行排列组合
            elif do_permute_signs_var:
                # 对解 sol 进行符号和变量的排列组合，并将结果添加到最终解集合 final_soln 中
                permuted_sign_var = set(signed_permutations(sol))
                final_soln.update(permuted_sign_var)
            else:
                # 如果无需进行符号排列组合，直接将解 sol 添加到最终解集合 final_soln 中
                final_soln.add(sol)
        else:
            # 如果解 sol 中存在非整数值的元素，直接将解 sol 添加到最终解集合 final_soln 中
            final_soln.add(sol)
    # 返回最终解集合 final_soln
    return final_soln
# 定义一个函数，用于将子方程的解合并成原始方程的完整解
def merge_solution(var, var_t, solution):
    """
    This is used to construct the full solution from the solutions of sub
    equations.

    Explanation
    ===========

    For example when solving the equation `(x - y)(x^2 + y^2 - z^2) = 0`,
    solutions for each of the equations `x - y = 0` and `x^2 + y^2 - z^2` are
    found independently. Solutions for `x - y = 0` are `(x, y) = (t, t)`. But
    we should introduce a value for z when we output the solution for the
    original equation. This function converts `(t, t)` into `(t, t, n_{1})`
    where `n_{1}` is an integer parameter.
    """
    # 初始化一个空列表用于存放解
    sol = []

    # 如果解中有None，直接返回空元组
    if None in solution:
        return ()

    # 使用迭代器处理解
    solution = iter(solution)
    # 创建一个从'n_1'开始的整数参数符号生成器
    params = numbered_symbols("n", integer=True, start=1)
    # 遍历变量列表
    for v in var:
        # 如果变量在var_t中，则直接取解中的值
        if v in var_t:
            sol.append(next(solution))
        else:
            # 否则取下一个整数参数
            sol.append(next(params))

    # 检查解的每个值是否满足其变量的假设条件
    for val, symb in zip(sol, var):
        if check_assumptions(val, **symb.assumptions0) is False:
            return ()

    # 返回转换后的解作为元组
    return tuple(sol)


def _diop_solve(eq, params=None):
    # 遍历所有的Diophantine方程类型
    for diop_type in all_diop_classes:
        # 如果方程类型匹配当前类型，则解方程并返回解
        if diop_type(eq).matches():
            return diop_type(eq).solve(parameters=params)


def diop_solve(eq, param=symbols("t", integer=True)):
    """
    Solves the diophantine equation ``eq``.

    Explanation
    ===========

    Unlike ``diophantine()``, factoring of ``eq`` is not attempted. Uses
    ``classify_diop()`` to determine the type of the equation and calls
    the appropriate solver function.

    Use of ``diophantine()`` is recommended over other helper functions.
    ``diop_solve()`` can return either a set or a tuple depending on the
    nature of the equation.

    Usage
    =====

    ``diop_solve(eq, t)``: Solve diophantine equation, ``eq`` using ``t``
    as a parameter if needed.

    Details
    =======

    ``eq`` should be an expression which is assumed to be zero.
    ``t`` is a parameter to be used in the solution.

    Examples
    ========

    >>> from sympy.solvers.diophantine import diop_solve
    >>> from sympy.abc import x, y, z, w
    >>> diop_solve(2*x + 3*y - 5)
    (3*t_0 - 5, 5 - 2*t_0)
    >>> diop_solve(4*x + 3*y - 4*z + 5)
    (t_0, 8*t_0 + 4*t_1 + 5, 7*t_0 + 3*t_1 + 5)
    >>> diop_solve(x + 3*y - 4*z + w - 6)
    (t_0, t_0 + t_1, 6*t_0 + 5*t_1 + 4*t_2 - 6, 5*t_0 + 4*t_1 + 3*t_2 - 6)
    >>> diop_solve(x**2 + y**2 - 5)
    {(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)}


    See Also
    ========

    diophantine()
    """
    # 使用classify_diop()分类方程，获取变量、系数和方程类型
    var, coeff, eq_type = classify_diop(eq, _dict=False)

    # 根据方程类型选择合适的求解器并返回解
    if eq_type == Linear.name:
        return diop_linear(eq, param)

    elif eq_type == BinaryQuadratic.name:
        return diop_quadratic(eq, param)

    elif eq_type == HomogeneousTernaryQuadratic.name:
        return diop_ternary_quadratic(eq, parameterize=True)

    elif eq_type == HomogeneousTernaryQuadraticNormal.name:
        return diop_ternary_quadratic_normal(eq, parameterize=True)
    # 如果等式类型为一般毕达哥拉斯类型，则调用对应的求解函数
    elif eq_type == GeneralPythagorean.name:
        return diop_general_pythagorean(eq, param)

    # 如果等式类型为一元方程类型，则调用对应的求解函数
    elif eq_type == Univariate.name:
        return diop_univariate(eq)

    # 如果等式类型为一般平方和类型，则调用对应的求解函数，限制为无穷大
    elif eq_type == GeneralSumOfSquares.name:
        return diop_general_sum_of_squares(eq, limit=S.Infinity)

    # 如果等式类型为一般偶数次幂和类型，则调用对应的求解函数，限制为无穷大
    elif eq_type == GeneralSumOfEvenPowers.name:
        return diop_general_sum_of_even_powers(eq, limit=S.Infinity)

    # 如果等式类型不为空且不在已知的类型列表中，则引发值错误异常
    if eq_type is not None and eq_type not in diop_known:
            raise ValueError(filldedent('''
    虽然识别到此类型的方程，但尚未处理。应在本文件顶部的 `diop_known` 列表中列出。
    开发人员应查看 `classify_diop` 末尾的注释。
            '''))  # pragma: no cover
    else:
        # 否则，引发未实现错误，说明没有为此类型的方程编写求解器
        raise NotImplementedError(
            'No solver has been written for %s.' % eq_type)
# 定义一个函数用于对二次或更高次的丢番图方程进行分类和处理
def classify_diop(eq, _dict=True):
    # 初始匹配状态为假
    matched = False
    # 初始类型为空
    diop_type = None
    # 遍历所有的丢番图方程类型类
    for diop_class in all_diop_classes:
        # 使用当前类型类对方程进行分类
        diop_type = diop_class(eq)
        # 如果匹配成功
        if diop_type.matches():
            # 更新匹配状态为真
            matched = True
            # 退出循环
            break

    # 如果匹配成功
    if matched:
        # 返回自由符号、系数（作为字典或默认字典）、类型名称的元组
        return diop_type.free_symbols, dict(diop_type.coeff) if _dict else diop_type.coeff, diop_type.name

    # 如果未匹配成功，抛出未实现错误，指示方程未能被分类
    raise NotImplementedError(filldedent('''
        This equation is not yet recognized or else has not been
        simplified sufficiently to put it in a form recognized by
        diop_classify().'''))


# 将函数的文档字符串设置为外部提供的字符串
classify_diop.func_doc = (
    '''
    Helper routine used by diop_solve() to find information about ``eq``.

    Explanation
    ===========

    Returns a tuple containing the type of the diophantine equation
    along with the variables (free symbols) and their coefficients.
    Variables are returned as a list and coefficients are returned
    as a dict with the key being the respective term and the constant
    term is keyed to 1. The type is one of the following:

    * %s

    Usage
    =====

    ``classify_diop(eq)``: Return variables, coefficients and type of the
    ``eq``.

    Details
    =======

    ``eq`` should be an expression which is assumed to be zero.
    ``_dict`` is for internal use: when True (default) a dict is returned,
    otherwise a defaultdict which supplies 0 for missing keys is returned.

    Examples
    ========

    >>> from sympy.solvers.diophantine import classify_diop
    >>> from sympy.abc import x, y, z, w, t
    >>> classify_diop(4*x + 6*y - 4)
    ([x, y], {1: -4, x: 4, y: 6}, 'linear')
    >>> classify_diop(x + 3*y -4*z + 5)
    ([x, y, z], {1: 5, x: 1, y: 3, z: -4}, 'linear')
    >>> classify_diop(x**2 + y**2 - x*y + x + 5)
    ([x, y], {1: 5, x: 1, x**2: 1, y**2: 1, x*y: -1}, 'binary_quadratic')
    ''' % ('\n    * '.join(sorted(diop_known))))
    # 根据给定的线性二次方程或线性多次方程分类变量、系数和类型
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    # 如果方程是线性方程
    if diop_type == Linear.name:
        # 参数初始化为 None
        parameters = None
        # 如果 param 不为空，则创建相应数量的整数符号参数
        if param is not None:
            parameters = symbols('%s_0:%i' % (param, len(var)), integer=True)

        # 解决线性方程
        result = Linear(eq).solve(parameters=parameters)

        # 如果 param 为空，则将结果初始化为全零参数
        if param is None:
            result = result(*[0]*len(result.parameters))

        # 如果结果列表长度大于零，则返回结果列表的第一个元素
        if len(result) > 0:
            return list(result)[0]
        else:
            # 否则返回一个包含和参数数量相同的 None 的元组
            return tuple([None]*len(result.parameters))
def base_solution_linear(c, a, b, t=None):
    """
    Return the base solution for the linear equation, `ax + by = c`.

    Explanation
    ===========

    Used by ``diop_linear()`` to find the base solution of a linear
    Diophantine equation. If ``t`` is given then the parametrized solution is
    returned.

    Usage
    =====

    ``base_solution_linear(c, a, b, t)``: ``a``, ``b``, ``c`` are coefficients
    in `ax + by = c` and ``t`` is the parameter to be used in the solution.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import base_solution_linear
    >>> from sympy.abc import t
    >>> base_solution_linear(5, 2, 3) # equation 2*x + 3*y = 5
    (-5, 5)
    >>> base_solution_linear(0, 5, 7) # equation 5*x + 7*y = 0
    (0, 0)
    >>> base_solution_linear(5, 2, 3, t) # equation 2*x + 3*y = 5
    (3*t - 5, 5 - 2*t)
    >>> base_solution_linear(0, 5, 7, t) # equation 5*x + 7*y = 0
    (7*t, -5*t)
    """
    # 确保 a, b, c 没有公因子
    a, b, c = _remove_gcd(a, b, c)

    # 如果 c 为 0，返回特殊解
    if c == 0:
        if t is None:
            return (0, 0)
        if b < 0:
            t = -t
        return (b*t, -a*t)

    # 计算 ax + by = gcd(a, b) 的基础解 (x0, y0)
    x0, y0, d = igcdex(abs(a), abs(b))
    x0 *= sign(a)
    y0 *= sign(b)

    # 如果 c 不是 gcd(a, b) 的倍数，则无解
    if c % d:
        return (None, None)

    # 返回特解或通解，取决于是否有参数 t
    if t is None:
        return (c*x0, c*y0)
    if b < 0:
        t = -t
    return (c*x0 + b*t, c*y0 - a*t)


def diop_univariate(eq):
    """
    Solves a univariate diophantine equations.

    Explanation
    ===========

    A univariate diophantine equation is an equation of the form
    `a_{0} + a_{1}x + a_{2}x^2 + .. + a_{n}x^n = 0` where `a_{1}, a_{2}, ..a_{n}` are
    integer constants and `x` is an integer variable.

    Usage
    =====

    ``diop_univariate(eq)``: Returns a set containing solutions to the
    diophantine equation ``eq``.

    Details
    =======

    ``eq`` is a univariate diophantine equation which is assumed to be zero.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_univariate
    >>> from sympy.abc import x
    >>> diop_univariate((x - 2)*(x - 3)**2) # solves equation (x - 2)*(x - 3)**2 == 0
    {(2,), (3,)}

    """
    # 分类单变量二次方程
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    # 如果是单变量二次方程
    if diop_type == Univariate.name:
        # 解方程并返回整数解的集合
        return {(int(i),) for i in solveset_real(
            eq, var[0]).intersect(S.Integers)}


def divisible(a, b):
    """
    Returns `True` if ``a`` is divisible by ``b`` and `False` otherwise.
    """
    # 检查 a 是否能被 b 整除
    return not a % b


def diop_quadratic(eq, param=symbols("t", integer=True)):
    """
    Solves quadratic diophantine equations.

    i.e. equations of the form `Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0`. Returns a
    set containing the tuples `(x, y)` which contains the solutions. If there
    are no solutions then `(None, None)` is returned.

    Usage
    =====

    ``diop_quadratic(eq, param)``: ``eq`` is a quadratic binary diophantine
    equation. ``param`` is used to indicate the parameter to be used in the
    solution.

    Details
    =======
    """
    # 根据给定的二次二元方程，分类并确定其类型、系数和变量
    var, coeff, diop_type = classify_diop(eq, _dict=False)
    
    # 检查方程的类型是否为二元二次方程
    if diop_type == BinaryQuadratic.name:
        # 如果有指定参数，创建参数列表，包括给定参数和一个整数符号"u"
        if param is not None:
            parameters = [param, Symbol("u", integer=True)]
        else:
            # 如果没有指定参数，则参数列表为None
            parameters = None
        # 使用二元二次方程对象解决方程，返回解的集合
        return set(BinaryQuadratic(eq).solve(parameters=parameters))
# 检查给定的 (u, v) 是否是二次二元整数二次方程的解，方程的变量列表为 ``var``，系数字典为 ``coeff``
def is_solution_quad(var, coeff, u, v):
    """
    Check whether `(u, v)` is solution to the quadratic binary diophantine
    equation with the variable list ``var`` and coefficient dictionary
    ``coeff``.

    Not intended for use by normal users.
    """
    # 创建变量到对应值的映射字典
    reps = dict(zip(var, (u, v)))
    # 计算二次方程的表达式，替换变量为对应的值后求和
    eq = Add(*[j*i.xreplace(reps) for i, j in coeff.items()])
    # 扩展和简化表达式，并检查是否等于零
    return _mexpand(eq) == 0


# 解决形如 `x^2 - Dy^2 = N` 的二次形二元整数方程
def diop_DN(D, N, t=symbols("t", integer=True)):
    """
    Solves the equation `x^2 - Dy^2 = N`.

    Explanation
    ===========

    Mainly concerned with the case `D > 0, D` is not a perfect square,
    which is the same as the generalized Pell equation. The LMM
    algorithm [1]_ is used to solve this equation.

    Returns one solution tuple, (`x, y)` for each class of the solutions.
    Other solutions of the class can be constructed according to the
    values of ``D`` and ``N``.

    Usage
    =====

    ``diop_DN(D, N, t)``: D and N are integers as in `x^2 - Dy^2 = N` and
    ``t`` is the parameter to be used in the solutions.

    Details
    =======

    ``D`` and ``N`` correspond to D and N in the equation.
    ``t`` is the parameter to be used in the solutions.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_DN
    >>> diop_DN(13, -4) # Solves equation x**2 - 13*y**2 = -4
    [(3, 1), (393, 109), (36, 10)]

    The output can be interpreted as follows: There are three fundamental
    solutions to the equation `x^2 - 13y^2 = -4` given by (3, 1), (393, 109)
    and (36, 10). Each tuple is in the form (x, y), i.e. solution (3, 1) means
    that `x = 3` and `y = 1`.

    >>> diop_DN(986, 1) # Solves equation x**2 - 986*y**2 = 1
    [(49299, 1570)]

    See Also
    ========

    find_DN(), diop_bf_DN()

    References
    ==========

    .. [1] Solving the generalized Pell equation x**2 - D*y**2 = N, John P.
        Robertson, July 31, 2004, Pages 16 - 17. [online], Available:
        https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf
    """
    # 处理 D < 0 的情况
    if D < 0:
        if N == 0:
            return [(0, 0)]
        if N < 0:
            return []
        # N > 0:
        sol = []
        # 对于 N 的所有因子 d，使用 cornacchia 算法寻找解 (x, y)
        for d in divisors(square_factor(N), generator=True):
            for x, y in cornacchia(1, int(-D), int(N // d**2)):
                sol.append((d*x, d*y))
                if D == -1:
                    sol.append((d*y, d*x))  # 若 D 为 -1，则交换 x 和 y 的顺序也是解
        return sol

    # 处理 D == 0 的情况
    if D == 0:
        if N < 0:
            return []
        if N == 0:
            return [(0, t)]  # 返回 (0, t)，其中 t 是给定的整数参数
        sN, _exact = integer_nthroot(N, 2)
        if _exact:
            return [(sN, t)]  # 返回 (sN, t)，其中 sN 是 N 的平方根
        return []

    # 处理 D > 0 的情况
    sD, _exact = integer_nthroot(D, 2)
    # 如果 _exact 为真
    if _exact:
        # 如果 N 等于 0，直接返回一个包含一个元组的列表
        if N == 0:
            return [(sD*t, t)]

        # 初始化一个空列表 sol 来存储解
        sol = []

        # 遍历范围为 floor(sign(N)*(N - 1)/(2*sD)) + 1 的整数 y
        for y in range(floor(sign(N)*(N - 1)/(2*sD)) + 1):
            try:
                # 尝试计算 D*y**2 + N 的平方根 sq，并确保确切性 _exact 为真
                sq, _exact = integer_nthroot(D*y**2 + N, 2)
            except ValueError:
                # 如果计算平方根时出现错误，将 _exact 设为假
                _exact = False

            # 如果 _exact 为真，将解 (sq, y) 添加到 sol 列表中
            if _exact:
                sol.append((sq, y))

        # 返回找到的解列表 sol
        return sol

    # 如果 1 < N**2 < D
    if 1 < N**2 < D:
        # 更快地调用 _special_diop_DN 函数
        return _special_diop_DN(D, N)

    # 如果 N 等于 0，返回一个包含单个元组 (0, 0) 的列表
    if N == 0:
        return [(0, 0)]

    # 初始化一个空列表 sol 来存储解
    sol = []

    # 如果 abs(N) 等于 1
    if abs(N) == 1:
        # 使用 PQa(0, 1, D) 来生成 Pell 方程解的迭代器
        pqa = PQa(0, 1, D)
        *_, prev_B, prev_G = next(pqa)

        # 遍历 PQa 迭代器，直到找到符合条件的解
        for j, (*_, a, _, _B, _G) in enumerate(pqa):
            if a == 2*sD:
                break
            prev_B, prev_G = _B, _G
        
        # 如果 j 是奇数
        if j % 2:
            # 如果 N 等于 1，将解 (prev_G, prev_B) 添加到 sol 中
            if N == 1:
                sol.append((prev_G, prev_B))
            return sol
        
        # 如果 N 等于 -1，返回解 [(prev_G, prev_B)]
        if N == -1:
            return [(prev_G, prev_B)]
        
        # 再次遍历 PQa 迭代器，直到找到符合条件的解
        for _ in range(j):
            *_, _B, _G = next(pqa)
        
        # 返回找到的解列表 sol
        return [(_G, _B)]

    # 对于除数 f 在 square_factor(N) 生成器中产生的每个值
    for f in divisors(square_factor(N), generator=True):
        # 计算 m 作为 N 除以 f 的平方
        m = N // f**2
        am = abs(m)

        # 对于 sqrt_mod(D, am, all_roots=True) 生成的每个平方根 sqm
        for sqm in sqrt_mod(D, am, all_roots=True):
            # 计算 z 作为 symmetric_residue(sqm, am) 的结果
            z = symmetric_residue(sqm, am)

            # 使用 PQa(z, am, D) 来生成 Pell 方程解的迭代器
            pqa = PQa(z, am, D)
            *_, prev_B, prev_G = next(pqa)

            # 遍历 PQa 迭代器，直到找到符合条件的解
            for _ in range(length(z, am, D) - 1):
                _, q, *_, _B, _G = next(pqa)

                # 如果 q 的绝对值为 1
                if abs(q) == 1:
                    # 如果满足 Pell 方程条件 prev_G**2 - D*prev_B**2 == m
                    if prev_G**2 - D*prev_B**2 == m:
                        sol.append((f*prev_G, f*prev_B))
                    # 如果 a 是 diop_DN(D, -1) 的结果
                    elif a := diop_DN(D, -1):
                        sol.append((f*(prev_G*a[0][0] + prev_B*D*a[0][1]),
                                    f*(prev_G*a[0][1] + prev_B*a[0][0])))
                    break
                prev_B, prev_G = _B, _G

    # 返回找到的解列表 sol
    return sol
def _special_diop_DN(D, N):
    """
    Solves the equation `x^2 - Dy^2 = N` for the special case where
    `1 < N**2 < D` and `D` is not a perfect square.
    It is better to call `diop_DN` rather than this function, as
    the former checks the condition `1 < N**2 < D`, and calls the latter only
    if appropriate.

    Usage
    =====

    WARNING: Internal method. Do not call directly!

    ``_special_diop_DN(D, N)``: D and N are integers as in `x^2 - Dy^2 = N`.

    Details
    =======

    ``D`` and ``N`` correspond to D and N in the equation.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import _special_diop_DN
    >>> _special_diop_DN(13, -3) # Solves equation x**2 - 13*y**2 = -3
    [(7, 2), (137, 38)]

    The output can be interpreted as follows: There are two fundamental
    solutions to the equation `x^2 - 13y**2 = -3` given by (7, 2) and
    (137, 38). Each tuple is in the form (x, y), i.e. solution (7, 2) means
    that `x = 7` and `y = 2`.

    >>> _special_diop_DN(2445, -20) # Solves equation x**2 - 2445*y**2 = -20
    [(445, 9), (17625560, 356454), (698095554475, 14118073569)]

    See Also
    ========

    diop_DN()

    References
    ==========

    .. [1] Section 4.4.4 of the following book:
        Quadratic Diophantine Equations, T. Andreescu and D. Andrica,
        Springer, 2015.
    """

    # 计算 D 的平方根
    sqrt_D = isqrt(D)
    # 计算 N 的平方和其绝对值的平方因子
    F = {N // f**2: f for f in divisors(square_factor(abs(N)), generator=True)}
    # 初始化变量
    P = 0
    Q = 1
    G0, G1 = 0, 1
    B0, B1 = 1, 0

    # 存储解的列表
    solutions = []
    # 开始求解循环
    while True:
        # 迭代两次
        for _ in range(2):
            # 计算 a，这里的 a 是 P+sqrt_D 除以 Q 的整数部分
            a = (P + sqrt_D) // Q
            # 更新 P 和 Q
            P = a*Q - P
            Q = (D - P**2) // Q
            # 更新 G0, G1 和 B0, B1
            G0, G1 = G1, a*G1 + G0
            B0, B1 = B1, a*B1 + B0
            # 如果 G1^2 - D*B1^2 在 F 中，添加解
            if (s := G1**2 - D*B1**2) in F:
                f = F[s]
                solutions.append((f*G1, f*B1))
        # 如果 Q 变为 1，则退出循环
        if Q == 1:
            break
    # 返回所有解的列表
    return solutions
    # Assume gcd(a, b) = gcd(a, m) = 1 and a, b > 0 but no error checking
    # 假设 gcd(a, b) = gcd(a, m) = 1，并且 a, b > 0，但没有错误检查
    sols = set()
    # 初始化解集合
    for t in sqrt_mod_iter(-b*invert(a, m), m):
        # 对于每个满足条件的 t，使用模平方根迭代器
        if t < m // 2:
            # 如果 t 小于 m 的一半，则继续下一个迭代
            continue
        u, r = m, t
        # 初始化 u 和 r
        while (m1 := m - a*r**2) <= 0:
            # 当 m1 <= 0 时，不断更新 u 和 r
            u, r = r, u % r
        m1, _r = divmod(m1, b)
        # 对 m1 进行除法，得到商 m1 和余数 _r
        if _r:
            # 如果有余数，则继续下一个迭代
            continue
        s, _exact = iroot(m1, 2)
        # 计算 m1 的平方根 s，及其是否精确的 _exact 标志
        if _exact:
            # 如果平方根精确，则根据条件添加解到解集合
            if a == b and r < s:
                r, s = s, r
            sols.add((int(r), int(s)))
    # 返回解集合
    return sols
def diop_bf_DN(D, N, t=symbols("t", integer=True)):
    r"""
    Uses brute force to solve the equation, `x^2 - Dy^2 = N`.

    Explanation
    ===========

    Mainly concerned with the generalized Pell equation which is the case when
    `D > 0, D` is not a perfect square. For more information on the case refer
    [1]_. Let `(t, u)` be the minimal positive solution of the equation
    `x^2 - Dy^2 = 1`. Then this method requires
    `\sqrt{\\frac{\mid N \mid (t \pm 1)}{2D}}` to be small.

    Usage
    =====

    ``diop_bf_DN(D, N, t)``: ``D`` and ``N`` are coefficients in
    `x^2 - Dy^2 = N` and ``t`` is the parameter to be used in the solutions.

    Details
    =======

    ``D`` and ``N`` correspond to D and N in the equation.
    ``t`` is the parameter to be used in the solutions.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_bf_DN
    >>> diop_bf_DN(13, -4)
    [(3, 1), (-3, 1), (36, 10)]
    >>> diop_bf_DN(986, 1)
    [(49299, 1570)]

    See Also
    ========

    diop_DN()

    References
    ==========

    .. [1] Solving the generalized Pell equation x**2 - D*y**2 = N, John P.
        Robertson, July 31, 2004, Page 15. https://web.archive.org/web/20160323033128/http://www.jpr2718.org/pell.pdf
    """
    # 将 D 和 N 转换为整数
    D = as_int(D)
    N = as_int(N)

    # 初始化解集合
    sol = []

    # 调用 diop_DN 函数，获取基础解 (t, u)
    a = diop_DN(D, 1)
    u = a[0][0]
    # 如果 N 等于 0
    if N == 0:
        # 如果 D 小于 0，返回 [(0, 0)]
        if D < 0:
            return [(0, 0)]
        # 如果 D 等于 0，返回 [(0, t)]
        if D == 0:
            return [(0, t)]
        # 计算 D 的平方根 sD，以及是否完全平方 _exact
        sD, _exact = integer_nthroot(D, 2)
        # 如果 D 是完全平方数，返回 [(sD*t, t), (-sD*t, t)]
        if _exact:
            return [(sD*t, t), (-sD*t, t)]
        # 否则返回 [(0, 0)]
        return [(0, 0)]

    # 如果 N 的绝对值为 1，调用 diop_DN 函数处理
    if abs(N) == 1:
        return diop_DN(D, N)

    # 如果 N 大于 1
    if N > 1:
        # 初始化 L1 和 L2
        L1 = 0
        # 计算 L2，integer_nthroot(int(N*(u - 1)/(2*D)), 2)[0] + 1
        L2 = integer_nthroot(int(N*(u - 1)/(2*D)), 2)[0] + 1
    else: # N < -1
        # 计算 L1，_exact 是否为完全平方根
        L1, _exact = integer_nthroot(-int(N/D), 2)
        # 如果不是完全平方根，L1 加一
        if not _exact:
            L1 += 1
        # 计算 L2，integer_nthroot(-int(N*(u + 1)/(2*D)), 2)[0] + 1
        L2 = integer_nthroot(-int(N*(u + 1)/(2*D)), 2)[0] + 1

    # 在范围 L1 到 L2 中迭代 y
    for y in range(L1, L2):
        try:
            # 计算 x，_exact 是否为完全平方根
            x, _exact = integer_nthroot(N + D*y**2, 2)
        except ValueError:
            _exact = False
        # 如果是完全平方根，将解 (x, y) 加入 sol 中
        if _exact:
            sol.append((x, y))
            # 如果 (x, y) 和 (-x, y) 不等效，则将 (-x, y) 也加入 sol 中
            if not equivalent(x, y, -x, y, D, N):
                sol.append((-x, y))

    # 返回所有解集合 sol
    return sol
def transformation_to_DN(eq):
    """
    This function transforms general quadratic,
    `ax^2 + bxy + cy^2 + dx + ey + f = 0`
    to more easy to deal with `X^2 - DY^2 = N` form.

    Explanation
    ===========

    This is used to solve the general quadratic equation by transforming it to
    the latter form. Refer to [1]_ for more detailed information on the
    transformation. This function returns a tuple (A, B) where A is a 2 X 2
    matrix and B is a 2 X 1 matrix such that,

    Transpose([x y]) =  A * Transpose([X Y]) + B

    Usage
    =====

    ``transformation_to_DN(eq)``: `eq` represents a general quadratic equation
    in the form `ax^2 + bxy + cy^2 + dx + ey + f = 0`, and this function 
    returns a tuple (A, B).

    References
    ==========

    .. [1] Reference to the detailed transformation explanation.

    """
    # 根据给定的二次方程进行变换到 DN 形式
    def transformation_to_DN(eq):
        # 对二次方程进行分类，确定其类型和系数
        var, coeff, diop_type = classify_diop(eq, _dict=False)
        # 如果方程类型是二次二元对角线方程
        if diop_type == BinaryQuadratic.name:
            # 调用具体的函数进行变换到 DN 形式并返回结果
            return _transformation_to_DN(var, coeff)
# 定义一个函数 `_find_DN`，用于将二次二元方程的系数转换成形如 `X^2 - DY^2 = N` 的简化形式
def _find_DN(var, coeff):

    # 解包变量 `var`，分别赋值给 `x` 和 `y`
    x, y = var

    # 从系数字典 `coeff` 中获取对应的值赋给变量 `a`, `b`, `c`, `d`, `e`, `f`
    a = coeff[x**2]
    b = coeff[x*y]
    c = coeff[y**2]
    d = coeff[x]
    e = coeff[y]
    f = coeff[1]

    # 将系数列表中的每个元素应用 `as_int` 函数，然后通过 `_remove_gcd` 函数移除它们的最大公约数
    a, b, c, d, e, f = [as_int(i) for i in _remove_gcd(a, b, c, d, e, f)]

    # 定义符号 `X, Y`，声明它们为整数
    X, Y = symbols("X, Y", integer=True)

    # 如果变量 `b` 不为零，则执行以下逻辑
    if b:
        # 计算 `B` 和 `C`，并且通过 `_rational_pq` 函数得到它们的比率
        B, C = _rational_pq(2*a, b)
        A, T = _rational_pq(a, B**2)

        # 构建二次形式的系数字典 `coeff`
        coeff = {X**2: A*B, X*Y: 0, Y**2: B*(c*T - A*C**2), X: d*T, Y: B*e*T - d*T*C, 1: f*T*B}
        
        # 调用 `_transformation_to_DN` 函数，并返回变换后的矩阵 `A_0` 和 `B_0`
        A_0, B_0 = _transformation_to_DN([X, Y], coeff)
        
        # 返回矩阵变换结果
        return Matrix(2, 2, [S.One/B, -S(C)/B, 0, 1])*A_0, Matrix(2, 2, [S.One/B, -S(C)/B, 0, 1])*B_0

    # 如果变量 `d` 不为零，则执行以下逻辑
    if d:
        # 计算 `B` 和 `C`，并且通过 `_rational_pq` 函数得到它们的比率
        B, C = _rational_pq(2*a, d)
        A, T = _rational_pq(a, B**2)

        # 构建二次形式的系数字典 `coeff`
        coeff = {X**2: A, X*Y: 0, Y**2: c*T, X: 0, Y: e*T, 1: f*T - A*C**2}
        
        # 调用 `_transformation_to_DN` 函数，并返回变换后的矩阵 `A_0` 和 `B_0`
        A_0, B_0 = _transformation_to_DN([X, Y], coeff)
        
        # 返回矩阵变换结果
        return Matrix(2, 2, [S.One/B, 0, 0, 1])*A_0, Matrix(2, 2, [S.One/B, 0, 0, 1])*B_0 + Matrix([-S(C)/B, 0])

    # 如果变量 `e` 不为零，则执行以下逻辑
    if e:
        # 计算 `B` 和 `C`，并且通过 `_rational_pq` 函数得到它们的比率
        B, C = _rational_pq(2*c, e)
        A, T = _rational_pq(c, B**2)

        # 构建二次形式的系数字典 `coeff`
        coeff = {X**2: a*T, X*Y: 0, Y**2: A, X: 0, Y: 0, 1: f*T - A*C**2}
        
        # 调用 `_transformation_to_DN` 函数，并返回变换后的矩阵 `A_0` 和 `B_0`
        A_0, B_0 = _transformation_to_DN([X, Y], coeff)
        
        # 返回矩阵变换结果
        return Matrix(2, 2, [1, 0, 0, S.One/B])*A_0, Matrix(2, 2, [1, 0, 0, S.One/B])*B_0 + Matrix([0, -S(C)/B])

    # 如果上述条件都不满足，则执行以下逻辑
    # TODO: 预处理简化：不是必须的，但可能会简化方程
    return Matrix(2, 2, [S.One/a, 0, 0, 1]), Matrix([0, 0])
    # 定义符号变量 X 和 Y，并声明其为整数
    X, Y = symbols("X, Y", integer=True)
    # 使用 _transformation_to_DN 函数将 var 和 coeff 转换为 A 和 B
    A, B = _transformation_to_DN(var, coeff)
    
    # 计算变换后的坐标 u 和 v
    u = (A*Matrix([X, Y]) + B)[0]
    v = (A*Matrix([X, Y]) + B)[1]
    
    # 构建方程 eq，包含 x 和 y 的多项式系数
    eq = x**2*coeff[x**2] + x*y*coeff[x*y] + y**2*coeff[y**2] + x*coeff[x] + y*coeff[y] + coeff[1]
    
    # 将方程 eq 中的 x, y 替换为 u, v，然后进行化简
    simplified = _mexpand(eq.subs(zip((x, y), (u, v))))
    
    # 将化简后的表达式 simplified 转换为系数字典
    coeff = simplified.as_coefficients_dict()
    
    # 返回计算得到的两个比值
    return -coeff[Y**2]/coeff[X**2], -coeff[1]/coeff[X**2]
# 定义一个函数，用于检查参数 `x`, `y`, `a` 是否符合特定条件，并返回相应的参数化表示或者 None
def check_param(x, y, a, params):
    """
    If there is a number modulo ``a`` such that ``x`` and ``y`` are both
    integers, then return a parametric representation for ``x`` and ``y``
    else return (None, None).

    Here ``x`` and ``y`` are functions of ``t``.
    """
    # 导入清除系数的函数
    from sympy.simplify.simplify import clear_coefficients

    # 如果 x 是数值但不是整数，则返回一个参数化的二次方程解集合，其中包含 x, y 和参数 params
    if x.is_number and not x.is_Integer:
        return DiophantineSolutionSet([x, y], parameters=params)

    # 如果 y 是数值但不是整数，则返回一个参数化的二次方程解集合，其中包含 x, y 和参数 params
    if y.is_number and not y.is_Integer:
        return DiophantineSolutionSet([x, y], parameters=params)

    # 定义整数符号 m, n
    m, n = symbols("m, n", integer=True)
    # 将 m*x + n*y 的内容拆分成基本内容和系数
    c, p = (m*x + n*y).as_content_primitive()
    # 如果 a 不能整除 c 的分母，则返回一个参数化的二次方程解集合，其中包含 x, y 和参数 params
    if a % c.q:
        return DiophantineSolutionSet([x, y], parameters=params)

    # 通过清除 x, y 的系数，形成一个新的等式 eq
    # clear_coefficients(mx + b, R)[1] -> (R - b)/m
    eq = clear_coefficients(x, m)[1] - clear_coefficients(y, n)[1]
    # 将等式 eq 变为其基本内容和系数
    junk, eq = eq.as_content_primitive()

    # 返回使用 eq 和参数 params 进行解的结果
    return _diop_solve(eq, params=params)


# 定义一个函数，用于解决一般的三元二次方程 `ax^2 + by^2 + cz^2 + fxy + gyz + hxz = 0`
def diop_ternary_quadratic(eq, parameterize=False):
    """
    Solves the general quadratic ternary form,
    `ax^2 + by^2 + cz^2 + fxy + gyz + hxz = 0`.

    Returns a tuple `(x, y, z)` which is a base solution for the above
    equation. If there are no solutions, `(None, None, None)` is returned.

    Usage
    =====

    ``diop_ternary_quadratic(eq)``: Return a tuple containing a basic solution
    to ``eq``.

    Details
    =======

    ``eq`` should be an homogeneous expression of degree two in three variables
    and it is assumed to be zero.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import diop_ternary_quadratic
    >>> diop_ternary_quadratic(x**2 + 3*y**2 - z**2)
    (1, 0, 1)
    >>> diop_ternary_quadratic(4*x**2 + 5*y**2 - z**2)
    (1, 0, 2)
    >>> diop_ternary_quadratic(45*x**2 - 7*y**2 - 8*x*y - z**2)
    (28, 45, 105)
    >>> diop_ternary_quadratic(x**2 - 49*y**2 - z**2 + 13*z*y -8*x*y)
    (9, 1, 5)
    """
    # 分类三元二次方程，并提取变量、系数和类型信息
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    # 如果方程属于某种特定类型的三元二次方程
    if diop_type in (
            HomogeneousTernaryQuadratic.name,
            HomogeneousTernaryQuadraticNormal.name):
        # 解方程并获取基本解
        sol = _diop_ternary_quadratic(var, coeff)
        if len(sol) > 0:
            x_0, y_0, z_0 = list(sol)[0]
        else:
            x_0, y_0, z_0 = None, None, None

        # 如果需要参数化解，则返回参数化的三元二次方程解
        if parameterize:
            return _parametrize_ternary_quadratic(
                (x_0, y_0, z_0), var, coeff)
        return x_0, y_0, z_0


# 定义一个私有函数，用于解决一般的三元二次方程
def _diop_ternary_quadratic(_var, coeff):
    # 计算方程的总和
    eq = sum(i*coeff[i] for i in coeff)
    # 如果方程是一个齐次三元二次方程的匹配模式，则求解
    if HomogeneousTernaryQuadratic(eq).matches():
        return HomogeneousTernaryQuadratic(eq, free_symbols=_var).solve()
    # 如果方程是一个正常的齐次三元二次方程的匹配模式，则求解
    elif HomogeneousTernaryQuadraticNormal(eq).matches():
        return HomogeneousTernaryQuadraticNormal(eq, free_symbols=_var).solve()


# 定义一个函数，用于将一般三元二次方程转换为正常形式
def transformation_to_normal(eq):
    """
    Returns the transformation Matrix that converts a general ternary
    quadratic equation ``eq`` (`ax^2 + by^2 + cz^2 + dxy + eyz + fxz`)
    """
    """
    将二次型方程组转换为没有交叉项的标准形式：`ax^2 + by^2 + cz^2 = 0`。
    这种转换在解决三元二次方程时不使用；仅仅是为了完整性而实现的。
    """
    # 根据方程分类，确定变量、系数和方程类型
    var, coeff, diop_type = classify_diop(eq, _dict=False)
    
    # 如果方程类型是“齐次三元二次方程”或“齐次三元二次方程的标准形式”
    if diop_type in (
            "homogeneous_ternary_quadratic",
            "homogeneous_ternary_quadratic_normal"):
        # 调用函数将方程转换为标准形式
        return _transformation_to_normal(var, coeff)
def _transformation_to_normal(var, coeff):
    _var = list(var)  # 复制变量列表，以便修改而不影响原始变量
    x, y, z = var  # 将变量解包为 x, y, z
    
    # 检查是否存在非零的二次项系数，根据系数确定变换矩阵
    if not any(coeff[i**2] for i in var):
        # 如果没有二次项系数不为零的情况，按照特定变换方式构建变换矩阵 T
        a = coeff[x*y]
        b = coeff[y*z]
        c = coeff[x*z]
        swap = False
        if not a:
            swap = True
            a, b = b, a
        T = Matrix(((1, 1, -b/a), (1, -1, -c/a), (0, 0, 1)))
        if swap:
            T.row_swap(0, 1)
            T.col_swap(0, 1)
        return T

    if coeff[x**2] == 0:
        # 如果 x 的系数为零，根据 y 和 z 的系数情况改变变量顺序，然后递归调用 _transformation_to_normal
        if coeff[y**2] == 0:
            _var[0], _var[2] = var[2], var[0]
            T = _transformation_to_normal(_var, coeff)
            T.row_swap(0, 2)
            T.col_swap(0, 2)
            return T

        _var[0], _var[1] = var[1], var[0]
        T = _transformation_to_normal(_var, coeff)
        T.row_swap(0, 1)
        T.col_swap(0, 1)
        return T

    # 对于非零的 x 的二次项系数，应用特定的线性变换公式
    if coeff[x*y] != 0 or coeff[x*z] != 0:
        A = coeff[x**2]
        B = coeff[x*y]
        C = coeff[x*z]
        D = coeff[y**2]
        E = coeff[y*z]
        F = coeff[z**2]

        _coeff = {}

        # 计算线性变换的新系数
        _coeff[x**2] = 4*A**2
        _coeff[y**2] = 4*A*D - B**2
        _coeff[z**2] = 4*A*F - C**2
        _coeff[y*z] = 4*A*E - 2*B*C
        _coeff[x*y] = 0
        _coeff[x*z] = 0

        # 对变换后的系数递归调用 _transformation_to_normal，并将结果与特定矩阵相乘
        T_0 = _transformation_to_normal(_var, _coeff)
        return Matrix(3, 3, [1, S(-B)/(2*A), S(-C)/(2*A), 0, 1, 0, 0, 0, 1])*T_0

    elif coeff[y*z] != 0:
        if coeff[y**2] == 0:
            if coeff[z**2] == 0:
                # 对应形式为 A*x**2 + E*yz = 0 的情况，进行特定变量变换
                return Matrix(3, 3, [1, 0, 0, 0, 1, 1, 0, 1, -1])

            # 对应形式为 Ax**2 + E*y*z + F*z**2  = 0 的情况，调整变量顺序后递归调用
            _var[0], _var[2] = var[2], var[0]
            T = _transformation_to_normal(_var, coeff)
            T.row_swap(0, 2)
            T.col_swap(0, 2)
            return T

        # 对应形式为 A*x**2 + D*y**2 + E*y*z + F*z**2 = 0 的情况，调整变量顺序后递归调用
        _var[0], _var[1] = var[1], var[0]
        T = _transformation_to_normal(_var, coeff)
        T.row_swap(0, 1)
        T.col_swap(0, 1)
        return T

    # 如果以上条件都不满足，则返回单位矩阵
    return Matrix.eye(3)
    # 分类和识别二次三元齐次方程的类型及其系数
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    # 如果识别到的方程类型是齐次三元二次方程或正规化的齐次三元二次方程
    if diop_type in (
            "homogeneous_ternary_quadratic",
            "homogeneous_ternary_quadratic_normal"):
        # 从解集中取出第一个解 x_0, y_0, z_0
        x_0, y_0, z_0 = list(_diop_ternary_quadratic(var, coeff))[0]
        # 调用 _parametrize_ternary_quadratic 函数进行参数化
        return _parametrize_ternary_quadratic(
            (x_0, y_0, z_0), var, coeff)
# 解决三元二次二次齐次二次方程的参数化问题
def _parametrize_ternary_quadratic(solution, _var, coeff):
    # 断言排除了系数中是否有1，确保是三元二次方程
    assert 1 not in coeff

    # 从解中提取出三个变量的初始值
    x_0, y_0, z_0 = solution

    # 复制变量列表
    v = list(_var)  # copy

    # 如果 x_0 是 None，则返回三个 None
    if x_0 is None:
        return (None, None, None)

    # 如果解中有两个以上的零，方程简化为 k*X**2 == 0，其中 X 是 x, y 或 z，因此 X 也必须是零，只有平凡解
    if solution.count(0) >= 2:
        return (None, None, None)

    # 如果 x_0 为零，交换变量顺序，递归调用参数化函数并返回结果
    if x_0 == 0:
        v[0], v[1] = v[1], v[0]
        y_p, x_p, z_p = _parametrize_ternary_quadratic(
            (y_0, x_0, z_0), v, coeff)
        return x_p, y_p, z_p

    # 获取变量 x, y, z
    x, y, z = v
    # 声明符号变量 r, p, q
    r, p, q = symbols("r, p, q", integer=True)

    # 构造方程 eq = a*x**2 + b*y**2 + c*z**2 + d*x*y + e*y*z + f*x*z
    eq = sum(k*v for k, v in coeff.items())

    # 替换变量 x, y, z 为 r*x_0, r*y_0 + p, r*z_0 + q，展开并将其分解为 A*r + B
    eq_1 = _mexpand(eq.subs(zip(
        (x, y, z), (r*x_0, r*y_0 + p, r*z_0 + q))))
    A, B = eq_1.as_independent(r, as_Add=True)

    # 计算解 x, y, z
    x = A*x_0
    y = (A*y_0 - _mexpand(B/r*p))
    z = (A*z_0 - _mexpand(B/r*q))

    # 移除最大公约数并返回结果
    return _remove_gcd(x, y, z)


# 解决一般形式的三元二次齐次方程
def diop_ternary_quadratic_normal(eq, parameterize=False):
    """
    解决形如 `ax^2 + by^2 + cz^2 = 0` 的三元二次齐次方程。

    说明
    ===========
    这里的系数 `a`, `b`, `c` 应该是非零的。否则方程将是一个二元或一元二次方程。如果有整数解，
    返回满足给定方程的元组 `(x, y, z)`。如果方程没有整数解，则返回 `(None, None, None)`。

    使用方法
    =====

    ``diop_ternary_quadratic_normal(eq)``: 其中 ``eq`` 是形如 `ax^2 + by^2 + cz^2 = 0` 的方程。

    示例
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import diop_ternary_quadratic_normal
    >>> diop_ternary_quadratic_normal(x**2 + 3*y**2 - z**2)
    (1, 0, 1)
    >>> diop_ternary_quadratic_normal(4*x**2 + 5*y**2 - z**2)
    (1, 0, 2)
    >>> diop_ternary_quadratic_normal(34*x**2 - 3*y**2 - 301*z**2)
    (4, 9, 1)
    """
    # 根据方程类型分类，获取变量、系数和方程类型
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    # 如果是三元二次齐次方程，解方程并提取解的第一个元组
    if diop_type == HomogeneousTernaryQuadraticNormal.name:
        sol = _diop_ternary_quadratic_normal(var, coeff)
        if len(sol) > 0:
            x_0, y_0, z_0 = list(sol)[0]
        else:
            x_0, y_0, z_0 = None, None, None
        
        # 如果需要参数化，则调用参数化函数并返回结果
        if parameterize:
            return _parametrize_ternary_quadratic(
                (x_0, y_0, z_0), var, coeff)
        
        # 否则直接返回解
        return x_0, y_0, z_0


# 解决三元二次齐次方程的特解
def _diop_ternary_quadratic_normal(var, coeff):
    # 计算方程的表达式并返回其解
    eq = sum(i * coeff[i] for i in coeff)
    return HomogeneousTernaryQuadraticNormal(eq, free_symbols=var).solve()


# 返回 `ax^2 + by^2 + cz^2 = 0` 方程的平方自由正则形式的系数
def sqf_normal(a, b, c, steps=False):
    """
    返回 `ax^2 + by^2 + cz^2 = 0` 方程的平方自由正则形式的系数，
    其中 `a', b', c'` 是两两互质的。如果 `steps` 为 True，则还返回三个元组：
    """
    # 将 a, b, c 的最大公约数移除后，计算并存储每个数的平方因子
    ABC = _remove_gcd(a, b, c)
    sq = tuple(square_factor(i) for i in ABC)
    
    # 计算并存储每个数在移除最大公约数和平方因子后的值
    sqf = A, B, C = tuple([i//j**2 for i,j in zip(ABC, sq)])
    
    # 计算 A 和 B 的最大公约数，并归一化 A
    pc = igcd(A, B)
    A /= pc
    B /= pc
    
    # 计算 B 和 C 的最大公约数，并归一化 B
    pa = igcd(B, C)
    B /= pa
    C /= pa
    
    # 计算 A 和 C 的最大公约数，并归一化 A 和 C
    pb = igcd(A, C)
    A /= pb
    B /= pb
    
    # 根据之前归一化的因子重新计算 A, B, C
    A *= pa
    B *= pb
    C *= pc
    
    # 如果需要详细步骤，则返回包含 sq, sqf 和 (A, B, C) 的元组
    if steps:
        return (sq, sqf, (A, B, C))
    else:
        # 否则只返回归一化后的 A, B, C
        return A, B, C
def square_factor(a):
    r"""
    Returns an integer `c` s.t. `a = c^2k, \ c,k \in Z`. Here `k` is square
    free. `a` can be given as an integer or a dictionary of factors.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import square_factor
    >>> square_factor(24)
    2
    >>> square_factor(-36*3)
    6
    >>> square_factor(1)
    1
    >>> square_factor({3: 2, 2: 1, -1: 1})  # -18
    3

    See Also
    ========
    sympy.ntheory.factor_.core
    """
    # 若输入参数 `a` 是字典，则直接使用，否则通过 `factorint` 函数获取其因子
    f = a if isinstance(a, dict) else factorint(a)
    # 计算每个素因子的一半幂，然后返回它们的乘积
    return Mul(*[p**(e//2) for p, e in f.items()])


def reconstruct(A, B, z):
    """
    Reconstruct the `z` value of an equivalent solution of `ax^2 + by^2 + cz^2`
    from the `z` value of a solution of the square-free normal form of the
    equation, `a'*x^2 + b'*y^2 + c'*z^2`, where `a'`, `b'` and `c'` are square
    free and `gcd(a', b', c') == 1`.
    """
    # 计算 `A` 和 `B` 的最大公因数的素因子分解
    f = factorint(igcd(A, B))
    # 如果某个素因子的指数不为1，则抛出异常，因为 `A` 和 `B` 应该是无平方因子的
    for p, e in f.items():
        if e != 1:
            raise ValueError('a and b should be square-free')
        # 更新 `z` 的值为 `z` 乘以所有素因子 `p`
        z *= p
    return z


def ldescent(A, B):
    """
    Return a non-trivial solution to `w^2 = Ax^2 + By^2` using
    Lagrange's method; return None if there is no such solution.

    Parameters
    ==========

    A : Integer
    B : Integer
        non-zero integer

    Returns
    =======

    (int, int, int) | None : a tuple `(w_0, x_0, y_0)` which is a solution to the above equation.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import ldescent
    >>> ldescent(1, 1) # w^2 = x^2 + y^2
    (1, 1, 0)
    >>> ldescent(4, -7) # w^2 = 4x^2 - 7y^2
    (2, -1, 0)

    This means that `x = -1, y = 0` and `w = 2` is a solution to the equation
    `w^2 = 4x^2 - 7y^2`

    >>> ldescent(5, -1) # w^2 = 5x^2 - y^2
    (2, 1, -1)

    References
    ==========

    .. [1] The algorithmic resolution of Diophantine equations, Nigel P. Smart,
           London Mathematical Society Student Texts 41, Cambridge University
           Press, Cambridge, 1998.
    .. [2] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    """
    # 如果 `A` 或 `B` 为零，则抛出异常
    if A == 0 or B == 0:
        raise ValueError("A and B must be non-zero integers")
    # 如果 `A` 的绝对值大于 `B` 的绝对值，则递归调用 `ldescent` 函数
    if abs(A) > abs(B):
        w, y, x = ldescent(B, A)
        return w, x, y
    # 如果 `A` 为1，返回特定的解
    if A == 1:
        return (1, 1, 0)
    # 如果 `B` 为1，返回特定的解
    if B == 1:
        return (1, 0, 1)
    # 如果 `A` 和 `B` 都为 -1，则直接返回，没有解
    if B == -1:  # and A == -1
        return

    # 计算 `A` 和 `B` 的平方根模 `B` 的值
    r = sqrt_mod(A, B)
    if r is None:
        return
    # 计算 `Q` 值
    Q = (r**2 - A) // B
    if Q == 0:
        return r, -1, 0
    # 遍历 `Q` 的所有除数
    for i in divisors(Q):
        # 如果 `Q` 除以 `i` 的整数平方根存在
        d, _exact = integer_nthroot(abs(Q) // i, 2)
        if _exact:
            # 计算 `B_0` 的值
            B_0 = sign(Q)*i
            # 调用 `ldescent` 函数，计算 `W`, `X`, `Y` 的值，并返回去除最大公因数后的结果
            W, X, Y = ldescent(A, B_0)
            return _remove_gcd(-A*X + r*W, r*X - W, Y*B_0*d)


def descent(A, B):
    """
    Returns a non-trivial solution, (x, y, z), to `x^2 = Ay^2 + Bz^2`
    """
    """
    使用拉格朗日下降法结合格点约简来解决二次二元对角线方程。假设 `A` 和 `B` 是合适的，以使解存在。

    这比普通的拉格朗日下降算法更快，因为使用了高斯约简。

    示例
    ========

    >>> from sympy.solvers.diophantine.diophantine import descent
    >>> descent(3, 1) # x**2 = 3*y**2 + z**2
    (1, 0, 1)

    `(x, y, z) = (1, 0, 1)` 是上述方程的一个解。

    >>> descent(41, -113)
    (-16, -3, 1)

    参考文献
    ==========

    .. [1] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    """
    # 如果 A 的绝对值大于 B 的绝对值，则交换参数位置再调用递归
    if abs(A) > abs(B):
        x, y, z = descent(B, A)
        return x, z, y

    # 特殊情况：当 B 为 1 时的解
    if B == 1:
        return (1, 0, 1)
    # 特殊情况：当 A 为 1 时的解
    if A == 1:
        return (1, 1, 0)
    # 特殊情况：当 B 等于 -A 时的解
    if B == -A:
        return (0, 1, 1)
    # 特殊情况：当 B 等于 A 时，交换参数并调用递归
    if B == A:
        x, z, y = descent(-1, A)
        return (A*y, z, x)

    # 计算平方根模 B
    w = sqrt_mod(A, B)
    # 高斯约简处理
    x_0, z_0 = gaussian_reduce(w, A, B)

    # 计算 t = (x_0**2 - A*z_0**2) / B
    t = (x_0**2 - A*z_0**2) // B
    # t 的平方因子
    t_2 = square_factor(t)
    # t 的其余部分
    t_1 = t // t_2**2

    # 递归调用解决 t_1 的方程
    x_1, z_1, y_1 = descent(A, t_1)

    # 返回解，并移除最大公因数影响
    return _remove_gcd(x_0*x_1 + A*z_0*z_1, z_0*x_1 + x_0*z_1, t_1*t_2*y_1)
def gaussian_reduce(w:int, a:int, b:int) -> tuple[int, int]:
    r"""
    Returns a reduced solution `(x, z)` to the congruence
    `X^2 - aZ^2 \equiv 0 \pmod{b}` so that `x^2 + |a|z^2` is as small as possible.
    Here ``w`` is a solution of the congruence `x^2 \equiv a \pmod{b}`.

    This function is intended to be used only for ``descent()``.

    Explanation
    ===========

    The Gaussian reduction can find the shortest vector for any norm.
    So we define the special norm for the vectors `u = (u_1, u_2)` and `v = (v_1, v_2)` as follows.

    .. math ::
        u \cdot v := (wu_1 + bu_2)(wv_1 + bv_2) + |a|u_1v_1

    Note that, given the mapping `f: (u_1, u_2) \to (wu_1 + bu_2, u_1)`,
    `f((u_1,u_2))` is the solution to `X^2 - aZ^2 \equiv 0 \pmod{b}`.
    In other words, finding the shortest vector in this norm will yield a solution with smaller `X^2 + |a|Z^2`.
    The algorithm starts from basis vectors `(0, 1)` and `(1, 0)`
    (corresponding to solutions `(b, 0)` and `(w, 1)`, respectively) and finds the shortest vector.
    The shortest vector does not necessarily correspond to the smallest solution,
    but since ``descent()`` only wants the smallest possible solution, it is sufficient.

    Parameters
    ==========

    w : int
        ``w`` s.t. `w^2 \equiv a \pmod{b}`
    a : int
        square-free nonzero integer
    b : int
        square-free nonzero integer

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import gaussian_reduce
    >>> from sympy.ntheory.residue_ntheory import sqrt_mod
    >>> a, b = 19, 101
    >>> gaussian_reduce(sqrt_mod(a, b), a, b) # 1**2 - 19*(-4)**2 = -303
    (1, -4)
    >>> a, b = 11, 14
    >>> x, z = gaussian_reduce(sqrt_mod(a, b), a, b)
    >>> (x**2 - a*z**2) % b == 0
    True

    It does not always return the smallest solution.

    >>> a, b = 6, 95
    >>> min_x, min_z = 1, 4
    >>> x, z = gaussian_reduce(sqrt_mod(a, b), a, b)
    >>> (x**2 - a*z**2) % b == 0 and (min_x**2 - a*min_z**2) % b == 0
    True
    >>> min_x**2 + abs(a)*min_z**2 < x**2 + abs(a)*z**2
    True

    References
    ==========

    .. [1] Gaussian lattice Reduction [online]. Available:
           https://web.archive.org/web/20201021115213/http://home.ie.cuhk.edu.hk/~wkshum/wordpress/?p=404
    .. [2] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    """
    a = abs(a)  # Ensure `a` is positive

    # Define inner function to compute dot product
    def _dot(u, v):
        return u[0]*v[0] + a*u[1]*v[1]

    u = (b, 0)  # Initialize basis vector u = (b, 0)
    v = (w, 1) if b*w >= 0 else (-w, -1)  # Initialize basis vector v based on sign of bw

    # Ensure that u dot v is non-negative
    # i.e., _dot(u, v) >= 0
    if b**2 < w**2 + a:
        u, v = v, u  # Swap u and v if norm(u) >= norm(v)

    # Perform Gaussian reduction algorithm
    while _dot(u, u) > (dv := _dot(v, v)):
        k = _dot(u, v) // dv
        u, v = v, (u[0] - k*v[0], u[1] - k*v[1])
    
    c = (v[0] - u[0], v[1] - u[1])  # Compute the reduced solution c

    # Check condition for returning the reduced solution
    if _dot(c, c) <= _dot(u, u) <= 2*_dot(u, v):
        return c  # Return the reduced solution (x, z)
    return u


注释：


    # 返回变量 u 的值
    return u


这行代码简单地返回变量 `u` 的值。在程序的上下文中，可能有一些先前的代码定义了变量 `u`，并且这里的返回语句将这个值返回给调用方或者用于后续的操作。
def holzer(x, y, z, a, b, c):
    r"""
    Simplify the solution `(x, y, z)` of the equation
    `ax^2 + by^2 = cz^2` with `a, b, c > 0` and `z^2 \geq \mid ab \mid` to
    a new reduced solution `(x', y', z')` such that `z'^2 \leq \mid ab \mid`.

    The algorithm is an interpretation of Mordell's reduction as described
    on page 8 of Cremona and Rusin's paper [1]_ and the work of Mordell in
    reference [2]_.

    References
    ==========

    .. [1] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    .. [2] Diophantine Equations, L. J. Mordell, page 48.

    """

    # 根据奇偶性判断变量k的值
    if _odd(c):
        k = 2*c
    else:
        k = c//2

    # 计算参数small
    small = a*b*c
    step = 0
    while True:
        # 计算t1, t2, t3
        t1, t2, t3 = a*x**2, b*y**2, c*z**2
        # 检查是否为解
        if t1 + t2 != t3:
            if step == 0:
                raise ValueError('bad starting solution')
            break
        # 保存当前解
        x_0, y_0, z_0 = x, y, z
        # 检查是否满足Holzer条件
        if max(t1, t2, t3) <= small:
            break

        # 计算基础线性解
        uv = u, v = base_solution_linear(k, y_0, -x_0)
        if None in uv:
            break

        # 计算比例r
        p, q = -(a*u*x_0 + b*v*y_0), c*z_0
        r = Rational(p, q)
        # 根据c的奇偶性确定w的值
        if _even(c):
            w = _nint_or_floor(p, q)
            assert abs(w - r) <= S.Half
        else:
            w = p//q  # floor
            if _odd(a*u + b*v + c*w):
                w += 1
            assert abs(w - r) <= S.One

        # 更新A和B的值
        A = (a*u**2 + b*v**2 + c*w**2)
        B = (a*u*x_0 + b*v*y_0 + c*w*z_0)
        # 更新(x, y, z)的值
        x = Rational(x_0*A - 2*u*B, k)
        y = Rational(y_0*A - 2*v*B, k)
        z = Rational(z_0*A - 2*w*B, k)
        # 检查(x, y, z)是否为整数解
        assert all(i.is_Integer for i in (x, y, z))
        step += 1

    # 返回整数化的解(x_0, y_0, z_0)
    return tuple([int(i) for i in (x_0, y_0, z_0)])


def diop_general_pythagorean(eq, param=symbols("m", integer=True)):
    """
    Solves the general pythagorean equation,
    `a_{1}^2x_{1}^2 + a_{2}^2x_{2}^2 + . . . + a_{n}^2x_{n}^2 - a_{n + 1}^2x_{n + 1}^2 = 0`.

    Returns a tuple which contains a parametrized solution to the equation,
    sorted in the same order as the input variables.

    Usage
    =====

    ``diop_general_pythagorean(eq, param)``: where ``eq`` is a general
    pythagorean equation which is assumed to be zero and ``param`` is the base
    parameter used to construct other parameters by subscripting.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_general_pythagorean
    >>> from sympy.abc import a, b, c, d, e
    >>> diop_general_pythagorean(a**2 + b**2 + c**2 - d**2)
    (m1**2 + m2**2 - m3**2, 2*m1*m3, 2*m2*m3, m1**2 + m2**2 + m3**2)
    >>> diop_general_pythagorean(9*a**2 - 4*b**2 + 16*c**2 + 25*d**2 + e**2)
    (10*m1**2  + 10*m2**2  + 10*m3**2 - 10*m4**2, 15*m1**2  + 15*m2**2  + 15*m3**2  + 15*m4**2, 15*m1*m4, 12*m2*m4, 60*m3*m4)
    """
    # 调用 classify_diop 函数对方程 eq 进行分类，并获取返回的变量、系数和类型
    var, coeff, diop_type  = classify_diop(eq, _dict=False)
    
    # 检查方程的分类结果是否为 GeneralPythagorean.name
    if diop_type == GeneralPythagorean.name:
        # 如果参数 param 为 None，则将 params 设置为 None
        if param is None:
            params = None
        else:
            # 如果 param 不为 None，则创建一组整数符号作为参数，参数的数量与变量 var 的数量相同
            params = symbols('%s1:%i' % (param, len(var)), integer=True)
        
        # 使用 GeneralPythagorean 类解决方程 eq，传递参数 params，并获取第一个解
        # 注意：假设 solve() 方法返回的是一个生成器，通过 list(...) 获取第一个解
        return list(GeneralPythagorean(eq).solve(parameters=params))[0]
# 解决形如 `x_{1}^2 + x_{2}^2 + . . . + x_{n}^2 - k = 0` 的方程
def diop_general_sum_of_squares(eq, limit=1):
    # 使用 classify_diop 函数对方程进行分类，返回变量、系数和方程类型
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    # 如果方程类型是 GeneralSumOfSquares.name
    if diop_type == GeneralSumOfSquares.name:
        # 返回使用 GeneralSumOfSquares 类解决方程的解集，最多返回 limit 个解
        return set(GeneralSumOfSquares(eq).solve(limit=limit))


# 解决形如 `x_{1}^e + x_{2}^e + . . . + x_{n}^e - k = 0` 的方程，其中 `e` 是偶数幂
def diop_general_sum_of_even_powers(eq, limit=1):
    # 使用 classify_diop 函数对方程进行分类，返回变量、系数和方程类型
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    # 如果方程类型是 GeneralSumOfEvenPowers.name
    if diop_type == GeneralSumOfEvenPowers.name:
        # 返回使用 GeneralSumOfEvenPowers 类解决方程的解集，最多返回 limit 个解
        return set(GeneralSumOfEvenPowers(eq).solve(limit=limit))


## 以下的函数可以更合适地归类到加法数论模块而不是丢番图方程模块下。


# 生成整数 `n` 的所有分割方式的生成器
def partition(n, k=None, zeros=False):
    """
    返回一个生成器，用于生成整数 `n` 的所有分割方式。

    解释
    ====

    `n` 的一个分割是一组加起来等于 `n` 的正整数。例如，3 的分割有 3, 1 + 2, 1 + 1 + 1。分割以元组形式返回。
    如果 `k` 为 None，则返回所有可能的分割，无论其大小。否则，只返回大小为 `k` 的分割。如果 `zero` 参数为 True，
    则会在每个小于 `k` 大小的分割末尾添加适当数量的零。

    `zero` 参数只有在 `k` 不为 None 时才考虑。
    """
    if not zeros or k is None:
        # 如果 `zeros` 参数为 False 或者 `k` 参数为 None，则执行以下代码块
        for i in ordered_partitions(n, k):
            # 使用 ordered_partitions 函数生成的迭代器 i，逐个生成其元组并 yield 出来
            yield tuple(i)
    else:
        # 如果 `zeros` 参数为 True 且 `k` 参数不为 None，则执行以下代码块
        for m in range(1, k + 1):
            # 循环遍历 m 从 1 到 k
            for i in ordered_partitions(n, m):
                # 使用 ordered_partitions 函数生成的迭代器 i，转换成元组
                i = tuple(i)
                # 生成一个新的元组，前面补充 (0,) * (k - len(i)) 个零元素，然后接上原元组 i
                yield (0,) * (k - len(i)) + i
# 定义函数 prime_as_sum_of_two_squares，表示将素数 p 表示为两个平方数之和的唯一方式；仅当 p ≡ 1 (mod 4) 时有效。
def prime_as_sum_of_two_squares(p):
    """
    Represent a prime `p` as a unique sum of two squares; this can
    only be done if the prime is congruent to 1 mod 4.

    Parameters
    ==========

    p : Integer
        A prime that is congruent to 1 mod 4

    Returns
    =======

    (int, int) | None : Pair of positive integers ``(x, y)`` satisfying ``x**2 + y**2 = p``.
                        None if ``p`` is not congruent to 1 mod 4.

    Raises
    ======

    ValueError
        If ``p`` is not prime number

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import prime_as_sum_of_two_squares
    >>> prime_as_sum_of_two_squares(7)  # can't be done
    >>> prime_as_sum_of_two_squares(5)
    (1, 2)

    Reference
    =========

    .. [1] Representing a number as a sum of four squares, [online],
           Available: https://schorn.ch/lagrange.html

    See Also
    ========

    sum_of_squares

    """
    # 将 p 转换为整数
    p = as_int(p)
    # 如果 p % 4 不等于 1，则返回 None
    if p % 4 != 1:
        return
    # 如果 p 不是素数，则引发 ValueError 异常
    if not isprime(p):
        raise ValueError("p should be a prime number")

    # 根据 p % 8 的不同值选择不同的 b 值，用于后续计算
    if p % 8 == 5:
        # Legendre 符号 (2/p) == -1，如果 p % 8 属于 [3, 5]
        b = 2
    elif p % 12 == 5:
        # Legendre 符号 (3/p) == -1，如果 p % 12 属于 [5, 7]
        b = 3
    elif p % 5 in [2, 3]:
        # Legendre 符号 (5/p) == -1，如果 p % 5 属于 [2, 3]
        b = 5
    else:
        # 否则设置 b = 7，并且要求 jacobi(b, p) == 1
        b = 7
        while jacobi(b, p) == 1:
            b = nextprime(b)

    # 计算 b^((p >> 2) % p)，用于后续计算
    b = pow(b, p >> 2, p)
    a = p
    # 当 b^2 > p 时，进行迭代更新 a, b
    while b**2 > p:
        a, b = b, a % b
    return (int(a % b), int(b))  # convert from long


# 定义函数 sum_of_three_squares，表示将整数 n 表示为三个平方数之和的方式
def sum_of_three_squares(n):
    r"""
    Returns a 3-tuple $(a, b, c)$ such that $a^2 + b^2 + c^2 = n$ and
    $a, b, c \geq 0$.

    Returns None if $n = 4^a(8m + 7)$ for some `a, m \in \mathbb{Z}`. See
    [1]_ for more details.

    Parameters
    ==========

    n : Integer
        non-negative integer

    Returns
    =======

    (int, int, int) | None : 3-tuple non-negative integers ``(a, b, c)`` satisfying ``a**2 + b**2 + c**2 = n``.
                             a,b,c are sorted in ascending order. ``None`` if no such ``(a,b,c)``.

    Raises
    ======

    ValueError
        If ``n`` is a negative integer

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sum_of_three_squares
    >>> sum_of_three_squares(44542)
    (18, 37, 207)

    References
    ==========

    .. [1] Representing a number as a sum of three squares, [online],
        Available: https://schorn.ch/lagrange.html

    See Also
    ========

    power_representation :
        ``sum_of_three_squares(n)`` is one of the solutions output by ``power_representation(n, 2, 3, zeros=True)``

    """
    # https://math.stackexchange.com/questions/483101/rabin-and-shallit-algorithm/651425#651425
    # discusses these numbers (except for 1, 2, 3) as the exceptions of H&L's conjecture that
    # Every sufficiently large number n is either a square or the sum of a prime and a square.
    # 定义一个特殊情况的字典，键为整数，值为三元组
    special = {1: (0, 0, 1), 2: (0, 1, 1), 3: (1, 1, 1), 10: (0, 1, 3), 34: (3, 3, 4),
               58: (0, 3, 7), 85: (0, 6, 7), 130: (0, 3, 11), 214: (3, 6, 13), 226: (8, 9, 9),
               370: (8, 9, 15), 526: (6, 7, 21), 706: (15, 15, 16), 730: (0, 1, 27),
               1414: (6, 17, 33), 1906: (13, 21, 36), 2986: (21, 32, 39), 9634: (56, 57, 57)}
    # 将参数 n 转换为整数
    n = as_int(n)
    # 如果 n 小于 0，则抛出值错误异常
    if n < 0:
        raise ValueError("n should be a non-negative integer")
    # 如果 n 等于 0，则直接返回全零元组 (0, 0, 0)
    if n == 0:
        return (0, 0, 0)
    # 调用 remove 函数，移除 n 的 4 位，并返回移除后的 n 和移除的值 v
    n, v = remove(n, 4)
    # 计算 v 的二进制左移 v 位后的值
    v = 1 << v
    # 如果 n 除以 8 的余数为 7，则返回空值
    if n % 8 == 7:
        return
    # 如果 n 在特殊情况字典中，则返回特殊情况对应的三元组，每个元素乘以 v
    if n in special:
        return tuple([v*i for i in special[n]])

    # 计算 n 的平方根 s，及是否为精确整数 _exact
    s, _exact = integer_nthroot(n, 2)
    # 如果 s 为精确整数，则返回 (0, 0, v*s)
    if _exact:
        return (0, 0, v*s)
    # 如果 n 除以 8 的余数为 3
    if n % 8 == 3:
        # 如果 s 为偶数，则减去 1
        if not s % 2:
            s -= 1
        # 从 s 开始到 0，每次减 2 的范围内寻找符合条件的 x
        for x in range(s, -1, -2):
            # 计算 N 的值
            N = (n - x**2) // 2
            # 如果 N 是素数，则计算 y 和 z 为两个平方数的和与差
            if isprime(N):
                # 返回排序后的三元组，每个元素乘以 v
                y, z = prime_as_sum_of_two_squares(N)
                return tuple(sorted([v*x, v*(y + z), v*abs(y - z)]))
        # 如果没有找到符合条件的 x，则断言错误，程序不应该执行到这里
        assert False

    # 断言 n 除以 4 的余数为 1 或 2
    # assert n % 4 in [1, 2]
    # 如果 n 与 s 的奇偶性相同，则减去 1
    if not((n % 2) ^ (s % 2)):
        s -= 1
    # 从 s 开始到 0，每次减 2 的范围内寻找符合条件的 x
    for x in range(s, -1, -2):
        # 计算 N 的值
        N = n - x**2
        # 如果 N 是素数，则计算 y 和 z 为两个平方数的和与差
        if isprime(N):
            # 返回排序后的三元组，每个元素乘以 v
            y, z = prime_as_sum_of_two_squares(N)
            return tuple(sorted([v*x, v*y, v*z]))
    # 如果没有找到符合条件的 x，则断言错误，程序不应该执行到这里
    assert False
def sum_of_four_squares(n):
    r"""
    Returns a 4-tuple `(a, b, c, d)` such that `a^2 + b^2 + c^2 + d^2 = n`.
    Here `a, b, c, d \geq 0`.

    Parameters
    ==========

    n : Integer
        non-negative integer

    Returns
    =======

    (int, int, int, int) : 4-tuple non-negative integers ``(a, b, c, d)`` satisfying ``a**2 + b**2 + c**2 + d**2 = n``.
                           a,b,c,d are sorted in ascending order.

    Raises
    ======

    ValueError
        If ``n`` is a negative integer

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sum_of_four_squares
    >>> sum_of_four_squares(3456)
    (8, 8, 32, 48)
    >>> sum_of_four_squares(1294585930293)
    (0, 1234, 2161, 1137796)

    References
    ==========

    .. [1] Representing a number as a sum of four squares, [online],
        Available: https://schorn.ch/lagrange.html

    See Also
    ========

    power_representation :
        ``sum_of_four_squares(n)`` is one of the solutions output by ``power_representation(n, 2, 4, zeros=True)``

    """
    # Convert n to integer (if it's not already)
    n = as_int(n)
    # Raise ValueError if n is negative
    if n < 0:
        raise ValueError("n should be a non-negative integer")
    # Return (0, 0, 0, 0) if n is 0
    if n == 0:
        return (0, 0, 0, 0)
    
    # Remove factors of 4 from n and record the number of 2s removed
    n, v = remove(n, 4)
    # v is multiplied by 2 and stored in v (v = 1 << v)
    v = 1 << v
    
    # Determine d based on the remainder of n modulo 8
    if n % 8 == 7:
        d = 2
        n = n - 4
    elif n % 8 in (2, 6):
        d = 1
        n = n - 1
    else:
        d = 0
    
    # Call sum_of_three_squares(n) and unpack the result into x, y, z
    x, y, z = sum_of_three_squares(n)  # sorted
    # Return a sorted tuple of values [v*d, v*x, v*y, v*z]
    return tuple(sorted([v*d, v*x, v*y, v*z]))


def power_representation(n, p, k, zeros=False):
    r"""
    Returns a generator for finding k-tuples of integers,
    `(n_{1}, n_{2}, . . . n_{k})`, such that
    `n = n_{1}^p + n_{2}^p + . . . n_{k}^p`.

    Usage
    =====

    ``power_representation(n, p, k, zeros)``: Represent non-negative number
    ``n`` as a sum of ``k`` ``p``\ th powers. If ``zeros`` is true, then the
    solutions is allowed to contain zeros.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import power_representation

    Represent 1729 as a sum of two cubes:

    >>> f = power_representation(1729, 3, 2)
    >>> next(f)
    (9, 10)
    >>> next(f)
    (1, 12)

    If the flag `zeros` is True, the solution may contain tuples with
    zeros; any such solutions will be generated after the solutions
    without zeros:

    >>> list(power_representation(125, 2, 3, zeros=True))
    [(5, 6, 8), (3, 4, 10), (0, 5, 10), (0, 2, 11)]

    For even `p` the `permute_sign` function can be used to get all
    signed values:

    >>> from sympy.utilities.iterables import permute_signs
    >>> list(permute_signs((1, 12)))
    [(1, 12), (-1, 12), (1, -12), (-1, -12)]

    All possible signed permutations can also be obtained:

    >>> from sympy.utilities.iterables import signed_permutations
    """
    # 将输入参数 n, p, k 转换为整数类型
    n, p, k = [as_int(i) for i in (n, p, k)]

    # 如果 n 小于 0，则根据奇偶性判断是否生成负号全排列
    if n < 0:
        if p % 2:
            # 生成 -n 的 p 次幂表达式的全排列
            for t in power_representation(-n, p, k, zeros):
                yield tuple(-i for i in t)
        return

    # 如果 p 或 k 不是正整数，则抛出 ValueError 异常
    if p < 1 or k < 1:
        raise ValueError(filldedent('''
    Expecting positive integers for `(p, k)`, but got `(%s, %s)`'''
    % (p, k)))

    # 如果 n 等于 0，则根据 zeros 参数生成全为零的 k 元组
    if n == 0:
        if zeros:
            yield (0,)*k
        return

    # 如果 k 等于 1，则根据 p 的不同情况生成对应的元组
    if k == 1:
        if p == 1:
            yield (n,)
        elif n == 1:
            yield (1,)
        else:
            # 判断 n 是否为完全幂，如果是，则生成对应的幂次元组
            be = perfect_power(n)
            if be:
                b, e = be
                d, r = divmod(e, p)
                if not r:
                    yield (b**d,)
        return

    # 如果 p 等于 1，则根据 k 的不同情况使用 partition 函数生成元组
    if p == 1:
        yield from partition(n, k, zeros=zeros)
        return

    # 如果 p 等于 2，则根据 k 的不同情况生成元组，考虑特定条件下的生成规则
    if p == 2:
        if k == 3:
            # 尝试移除 n 中的 4，同时生成与特定值 v 相乘的元组
            n, v = remove(n, 4)
            if v:
                v = 1 << v
                for t in power_representation(n, p, k, zeros):
                    yield tuple(i*v for i in t)
                return
        # 检查是否可以表示为 k 个平方数的和
        feasible = _can_do_sum_of_squares(n, k)
        if not feasible:
            return
        # 如果不允许零元素且满足特定条件，则返回
        if not zeros:
            if n > 33 and k >= 5 and k <= n and n - k in (
                13, 10, 7, 5, 4, 2, 1):
                '''Todd G. Will, "When Is n^2 a Sum of k Squares?", [online].
                Available: https://www.maa.org/sites/default/files/Will-MMz-201037918.pdf'''
                return
            # 快速测试，包括零可能性的可行性测试
            if k == 4 and (n in (1, 3, 5, 9, 11, 17, 29, 41) or remove(n, 4)[0] in (2, 6, 14)):
                # A000534
                return
            if k == 3 and n in (1, 2, 5, 10, 13, 25, 37, 58, 85, 130):  # or n = some number >= 5*10**10
                # A051952
                return
        # 如果 feasible 不等于 True，则表示 n 是质数且 k 等于 2
        if feasible is not True:
            yield prime_as_sum_of_two_squares(n)
            return

    # 如果 k 等于 2 且 p 大于 2，则根据 n 是否是 p 的幂次来判断是否返回
    if k == 2 and p > 2:
        be = perfect_power(n)
        if be and be[1] % p == 0:
            return  # Fermat: a**n + b**n = c**n has no solution for n > 2

    # 如果 n 大于等于 k，则计算 a 的值并递归生成 k 元组
    if n >= k:
        a = integer_nthroot(n - (k - 1), p)[0]
        for t in pow_rep_recursive(a, k, n, [], p):
            yield tuple(reversed(t))

    # 如果 zeros 为 True，则生成特定条件下的 k 元组
    if zeros:
        a = integer_nthroot(n, p)[0]
        for i in range(1, k):
            for t in pow_rep_recursive(a, i, n, [], p):
                yield tuple(reversed(t + (0,)*(k - i)))
# 将全局变量 sum_of_powers 初始化为 power_representation 函数的引用
sum_of_powers = power_representation

# 定义一个递归生成器函数，用于找出满足幂表示的 k 元组
def pow_rep_recursive(n_i, k, n_remaining, terms, p):
    # 如果 n_i 或 k 小于等于 0，则参数无效，直接返回
    if n_i <= 0 or k <= 0:
        return

    # 如果 n_remaining 小于 k，则不存在解
    if n_remaining < k:
        return
    
    # 如果 k * pow(n_i, p) 小于 n_remaining，则不存在解
    if k * pow(n_i, p) < n_remaining:
        return

    # 当 k 为 0 且 n_remaining 为 0 时，生成一个满足条件的 k 元组
    if k == 0 and n_remaining == 0:
        yield tuple(terms)

    # 当 k 为 1 时
    elif k == 1:
        # 下一个项的 p 次幂必须等于 n_remaining
        next_term, exact = integer_nthroot(n_remaining, p)
        if exact and next_term <= n_i:
            yield tuple(terms + [next_term])
        return

    # 当 k 大于 1 时
    else:
        # TODO: 当 k 等于 2 时，使用 diop_DN 进行处理
        if n_i >= 1 and k > 0:
            # 遍历可能的下一个项
            for next_term in range(1, n_i + 1):
                residual = n_remaining - pow(next_term, p)
                if residual < 0:
                    break
                # 递归调用 pow_rep_recursive 函数，继续查找下一个项
                yield from pow_rep_recursive(next_term, k - 1, residual, terms + [next_term], p)


# 定义一个生成器函数 sum_of_squares，返回满足条件的 k 元组
def sum_of_squares(n, k, zeros=False):
    """Return a generator that yields the k-tuples of nonnegative
    values, the squares of which sum to n. If zeros is False (default)
    then the solution will not contain zeros. The nonnegative
    elements of a tuple are sorted.

    * If k == 1 and n is square, (n,) is returned.

    * If k == 2 then n can only be written as a sum of squares if
      every prime in the factorization of n that has the form
      4*k + 3 has an even multiplicity. If n is prime then
      it can only be written as a sum of two squares if it is
      in the form 4*k + 1.

    * if k == 3 then n can be written as a sum of squares if it does
      not have the form 4**m*(8*k + 7).

    * all integers can be written as the sum of 4 squares.

    * if k > 4 then n can be partitioned and each partition can
      be written as a sum of 4 squares; if n is not evenly divisible
      by 4 then n can be written as a sum of squares only if the
      an additional partition can be written as sum of squares.
      For example, if k = 6 then n is partitioned into two parts,
      the first being written as a sum of 4 squares and the second
      being written as a sum of 2 squares -- which can only be
      done if the condition above for k = 2 can be met, so this will
      automatically reject certain partitions of n.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sum_of_squares
    >>> list(sum_of_squares(25, 2))
    [(3, 4)]
    >>> list(sum_of_squares(25, 2, True))
    [(3, 4), (0, 5)]
    >>> list(sum_of_squares(25, 4))
    [(1, 2, 2, 4)]

    See Also
    ========

    sympy.utilities.iterables.signed_permutations
    """
    # 使用 power_representation 函数生成满足条件的 k 元组并返回
    yield from power_representation(n, 2, k, zeros)


# 内部函数，检查 n 是否可以表示为 k 个平方数之和
def _can_do_sum_of_squares(n, k):
    """Return True if n can be written as the sum of k squares,
    False if it cannot, or 1 if ``k == 2`` and ``n`` is prime (in which
    case it *can* be written as a sum of two squares). A False
    is returned only if it cannot be written as ``k``-squares, even
    """
    # 此函数的实现在文档字符串中有详细描述
    # 如果 k 小于 1，返回 False，因为 k 不能小于 1
    if k < 1:
        return False
    # 如果 n 小于 0，返回 False，因为 n 不能是负数
    if n < 0:
        return False
    # 如果 n 等于 0，返回 True，因为 0 是一个完全平方数
    if n == 0:
        return True
    # 如果 k 等于 1，调用 is_square 函数检查 n 是否是完全平方数，并返回结果
    if k == 1:
        return is_square(n)
    # 如果 k 等于 2
    if k == 2:
        # 如果 n 是 1 或 2，返回 True
        if n in (1, 2):
            return True
        # 如果 n 是素数
        if isprime(n):
            # 如果 n 除以 4 的余数是 1，返回 1，表示 n 是素数
            if n % 4 == 1:
                return 1  # signal that it was prime
            # 否则返回 False
            return False
        # n 是合数，需要检查是否所有形如 4*k + 3 的质因数具有偶数次幂
        # 返回表达式的结果，判断条件是所有质因数的余数不为 3 或者幂为偶数
        return all(p % 4 !=3 or m % 2 == 0 for p, m in factorint(n).items())
    # 如果 k 等于 3，移除 n 中的 4，检查结果的第一个元素除以 8 的余数是否为 7，返回结果
    if k == 3:
        return remove(n, 4)[0] % 8 != 7
    # 对于 k 大于 4 的情况，每个数都可以写成 4 个平方数的和，返回 True
    # 对于 k 大于 4 的分区可以是 0
    return True
```