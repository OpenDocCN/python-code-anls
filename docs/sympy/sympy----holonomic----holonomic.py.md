# `D:\src\scipysrc\sympy\sympy\holonomic\holonomic.py`

```
"""
This module implements Holonomic Functions and
various operations on them.
"""

# 导入 sympy 库的各个模块和函数
from sympy.core import Add, Mul, Pow
from sympy.core.numbers import (NaN, Infinity, NegativeInfinity, Float, I, pi,
        equal_valued, int_valued)
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.elementary.exponential import exp_polar, exp, log
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.functions.special.error_functions import (Ci, Shi, Si, erf, erfc, erfi)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.integrals import meijerint
from sympy.matrices import Matrix
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement
from sympy.polys.domains import QQ, RR
from sympy.polys.polyclasses import DMF
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly
from sympy.polys.matrices import DomainMatrix
from sympy.printing import sstr
from sympy.series.limits import limit
from sympy.series.order import Order
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import nsimplify
from sympy.solvers.solvers import solve

# 导入 HolonomicSequence、RecurrenceOperator 和 RecurrenceOperators 类以及相关错误处理类
from .recurrence import HolonomicSequence, RecurrenceOperator, RecurrenceOperators
from .holonomicerrors import (NotPowerSeriesError, NotHyperSeriesError,
    SingularityError, NotHolonomicError)

# 定义函数 _find_nonzero_solution，用于查找非零解
def _find_nonzero_solution(r, homosys):
    ones = lambda shape: DomainMatrix.ones(shape, r.domain)
    # 解非齐次线性方程组，返回特解和齐次解空间
    particular, nullspace = r._solve(homosys)
    nullity = nullspace.shape[0]
    nullpart = ones((1, nullity)) * nullspace
    # 构造非零解
    sol = (particular + nullpart).transpose()
    return sol


# 定义 DifferentialOperators 函数，用于创建微分算子代数
def DifferentialOperators(base, generator):
    r"""
    This function is used to create annihilators using ``Dx``.

    Explanation
    ===========

    Returns an Algebra of Differential Operators also called Weyl Algebra
    and the operator for differentiation i.e. the ``Dx`` operator.

    Parameters
    ==========

    base:
        Base polynomial ring for the algebra.
        The base polynomial ring is the ring of polynomials in :math:`x` that
        will appear as coefficients in the operators.
    generator:
        Generator of the algebra which can
        be either a noncommutative ``Symbol`` or a string. e.g. "Dx" or "D".

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.abc import x
    >>> from sympy.holonomic.holonomic import DifferentialOperators
    >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    >>> R
    Univariate Differential Operator Algebra in intermediate Dx over the base ring ZZ[x]
    """
    # 返回微分算子代数和微分算子 generator
    return (
        Univariate Differential Operator Algebra in intermediate Dx over the base ring ZZ[x]
    )
    >>> Dx*x
    # 执行运算 Dx * x，这表示对 x 进行微分操作
    (1) + (x)*Dx
    # 结果是 1 加上 x 乘以 Dx，这是微分操作的代数表达式
    """

    # 创建一个微分算子代数环对象，使用给定的基和生成元参数
    ring = DifferentialOperatorAlgebra(base, generator)
    # 返回一个元组，包含创建的代数环对象和其微分算子
    return (ring, ring.derivative_operator)
class DifferentialOperatorAlgebra:
    r"""
    An Ore Algebra is a set of noncommutative polynomials in the
    intermediate ``Dx`` and coefficients in a base polynomial ring :math:`A`.
    It follows the commutation rule:

    .. math ::
       Dxa = \sigma(a)Dx + \delta(a)

    for :math:`a \subset A`.

    Where :math:`\sigma: A \Rightarrow A` is an endomorphism and :math:`\delta: A \rightarrow A`
    is a skew-derivation i.e. :math:`\delta(ab) = \delta(a) b + \sigma(a) \delta(b)`.

    If one takes the sigma as identity map and delta as the standard derivation
    then it becomes the algebra of Differential Operators also called
    a Weyl Algebra i.e. an algebra whose elements are Differential Operators.

    This class represents a Weyl Algebra and serves as the parent ring for
    Differential Operators.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> from sympy.holonomic.holonomic import DifferentialOperators
    >>> x = symbols('x')
    >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    >>> R
    Univariate Differential Operator Algebra in intermediate Dx over the base ring
    ZZ[x]

    See Also
    ========

    DifferentialOperator
    """

    def __init__(self, base, generator):
        # the base polynomial ring for the algebra
        self.base = base
        # the operator representing differentiation i.e. `Dx`
        self.derivative_operator = DifferentialOperator(
            [base.zero, base.one], self)

        if generator is None:
            # default generator symbol for the algebra
            self.gen_symbol = Symbol('Dx', commutative=False)
        else:
            if isinstance(generator, str):
                # use the provided string as the generator symbol
                self.gen_symbol = Symbol(generator, commutative=False)
            elif isinstance(generator, Symbol):
                # use the provided Symbol instance as the generator symbol
                self.gen_symbol = generator

    def __str__(self):
        # return a string representation of the algebra
        string = 'Univariate Differential Operator Algebra in intermediate ' \
                 + sstr(self.gen_symbol) + ' over the base ring ' + \
                 (self.base).__str__()
        return string

    __repr__ = __str__

    def __eq__(self, other):
        # check equality between two DifferentialOperatorAlgebra instances
        return self.base == other.base and \
               self.gen_symbol == other.gen_symbol


class DifferentialOperator:
    """
    Differential Operators are elements of Weyl Algebra. The Operators
    are defined by a list of polynomials in the base ring and the
    parent ring of the Operator i.e. the algebra it belongs to.

    Explanation
    ===========

    Takes a list of polynomials for each power of ``Dx`` and the
    parent ring which must be an instance of DifferentialOperatorAlgebra.

    A Differential Operator can be created easily using
    the operator ``Dx``. See examples below.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import DifferentialOperator, DifferentialOperators
    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> x = symbols('x')
    >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')
    """
    _op_priority = 20
    # 设置操作符优先级为 20
    
    def __init__(self, list_of_poly, parent):
        """
        Parameters
        ==========
    
        list_of_poly:
            属于代数基环的多项式列表。
        parent:
            操作符的父代数。
    
        """
    
        # 运算符的父环
        # 必须是 DifferentialOperatorAlgebra 对象
        self.parent = parent
        base = self.parent.base
        # 如果第一个生成元是符号，则设置为 self.x，否则为 base.gens[0][0]
        self.x = base.gens[0] if isinstance(base.gens[0], Symbol) else base.gens[0][0]
    
        # 对于每个多项式，如果不是 base.dtype 类型，则转换为 base.from_sympy(sympify(j)) 的结果
        for i, j in enumerate(list_of_poly):
            if not isinstance(j, base.dtype):
                list_of_poly[i] = base.from_sympy(sympify(j))
            else:
                list_of_poly[i] = base.from_sympy(base.to_sympy(j))
    
        self.listofpoly = list_of_poly
        # Dx 的最高次数
        self.order = len(self.listofpoly) - 1
    
    def __mul__(self, other):
        """
        将两个 DifferentialOperator 相乘，并根据交换规则返回另一个 DifferentialOperator 实例
        Dx*a = a*Dx + a'
        """
    
        listofself = self.listofpoly
        if isinstance(other, DifferentialOperator):
            listofother = other.listofpoly
        elif isinstance(other, self.parent.base.dtype):
            listofother = [other]
        else:
            listofother = [self.parent.base.from_sympy(sympify(other))]
    
        # 将多项式 `b` 乘以多项式列表
        def _mul_dmp_diffop(b, listofother):
            if isinstance(listofother, list):
                return [i * b for i in listofother]
            return [b * listofother]
    
        sol = _mul_dmp_diffop(listofself[0], listofother)
    
        # 计算 Dx^i * b
        def _mul_Dxi_b(b):
            sol1 = [self.parent.base.zero]
            sol2 = []
    
            if isinstance(b, list):
                for i in b:
                    sol1.append(i)
                    sol2.append(i.diff())
            else:
                sol1.append(self.parent.base.from_sympy(b))
                sol2.append(self.parent.base.from_sympy(b).diff())
    
            return _add_lists(sol1, sol2)
    
        for i in range(1, len(listofself)):
            # 在第 i 次迭代中找到 Dx^i * b
            listofother = _mul_Dxi_b(listofother)
            # 解 = 解 + listofself[i] * (Dx^i * b)
            sol = _add_lists(sol, _mul_dmp_diffop(listofself[i], listofother))
    
        return DifferentialOperator(sol, self.parent)
    # 定义 DifferentialOperator 类的右乘方法，用于实现 self * other 操作
    def __rmul__(self, other):
        # 如果 other 不是 DifferentialOperator 类型
        if not isinstance(other, DifferentialOperator):
            # 如果 other 不是 self.parent.base.dtype 类型，则将其转换为 self.parent.base 类型
            if not isinstance(other, self.parent.base.dtype):
                other = (self.parent.base).from_sympy(sympify(other))

            # 对 self.listofpoly 中的每个多项式与 other 相乘，生成结果列表 sol
            sol = [other * j for j in self.listofpoly]
            # 返回新的 DifferentialOperator 对象，包含结果 sol 和当前的 self.parent
            return DifferentialOperator(sol, self.parent)

    # 定义 DifferentialOperator 类的加法方法，用于实现 self + other 操作
    def __add__(self, other):
        # 如果 other 是 DifferentialOperator 类型
        if isinstance(other, DifferentialOperator):
            # 调用 _add_lists 函数将 self.listofpoly 和 other.listofpoly 对应位置相加，生成结果列表 sol
            sol = _add_lists(self.listofpoly, other.listofpoly)
            # 返回新的 DifferentialOperator 对象，包含结果 sol 和当前的 self.parent
            return DifferentialOperator(sol, self.parent)

        # 如果 other 不是 DifferentialOperator 类型
        list_self = self.listofpoly
        # 如果 other 不是 self.parent.base.dtype 类型，则将其转换为 self.parent.base 类型
        if not isinstance(other, self.parent.base.dtype):
            list_other = [((self.parent).base).from_sympy(sympify(other))]
        else:
            list_other = [other]

        # 将 self.listofpoly 的第一个多项式与 list_other 的第一个多项式相加，生成结果列表 sol
        sol = [list_self[0] + list_other[0]] + list_self[1:]
        # 返回新的 DifferentialOperator 对象，包含结果 sol 和当前的 self.parent
        return DifferentialOperator(sol, self.parent)

    # 将 __radd__ 方法设置为 __add__ 方法的别名，用于实现 other + self 操作
    __radd__ = __add__

    # 定义 DifferentialOperator 类的减法方法，用于实现 self - other 操作
    def __sub__(self, other):
        # 返回 self + (-1) * other 的结果
        return self + (-1) * other

    # 定义 DifferentialOperator 类的右减法方法，用于实现 other - self 操作
    def __rsub__(self, other):
        # 返回 (-1) * self + other 的结果
        return (-1) * self + other

    # 定义 DifferentialOperator 类的负号方法，用于实现 -self 操作
    def __neg__(self):
        # 返回 -1 * self 的结果
        return -1 * self

    # 定义 DifferentialOperator 类的除法方法，用于实现 self / other 操作
    def __truediv__(self, other):
        # 返回 self * (S.One / other) 的结果，其中 S.One 是 sympy 中的标量 1
        return self * (S.One / other)

    # 定义 DifferentialOperator 类的乘幂方法，用于实现 self ** n 操作
    def __pow__(self, n):
        # 如果 n 等于 1，则返回 self 本身
        if n == 1:
            return self

        # 创建一个包含 self.parent.base.one 的 DifferentialOperator 对象 result
        result = DifferentialOperator([self.parent.base.one], self.parent)

        # 如果 n 等于 0，则返回 result
        if n == 0:
            return result

        # 如果 self 的 listofpoly 等于 self.parent.derivative_operator.listofpoly
        if self.listofpoly == self.parent.derivative_operator.listofpoly:
            # 创建一个特定的 DifferentialOperator 对象 sol，包含 n 个 self.parent.base.zero 和一个 self.parent.base.one
            sol = [self.parent.base.zero]*n + [self.parent.base.one]
            return DifferentialOperator(sol, self.parent)

        # 初始化 x 为 self
        x = self
        while True:
            # 如果 n 是奇数，则 result *= x
            if n % 2:
                result *= x
            # 将 n 右移一位
            n >>= 1
            # 如果 n 变为 0，则跳出循环
            if not n:
                break
            # x *= x
            x *= x

        # 返回结果 DifferentialOperator 对象 result
        return result

    # 定义 DifferentialOperator 类的字符串表示方法，用于生成可打印的字符串
    def __str__(self):
        # 获取 self.listofpoly 列表
        listofpoly = self.listofpoly
        # 初始化打印字符串 print_str
        print_str = ''

        # 遍历 listofpoly 列表的索引 i 和元素 j
        for i, j in enumerate(listofpoly):
            # 如果 j 等于 self.parent.base.zero，则跳过当前循环
            if j == self.parent.base.zero:
                continue

            # 将 j 转换为 sympy 格式
            j = self.parent.base.to_sympy(j)

            # 如果 i 等于 0，则将 j 格式化为字符串后加入 print_str
            if i == 0:
                print_str += '(' + sstr(j) + ')'
                continue

            # 如果 print_str 非空，则在其末尾加上 ' + '
            if print_str:
                print_str += ' + '

            # 如果 i 等于 1，则将 j 格式化为字符串后加入 print_str，后面加上乘以 self.parent.gen_symbol
            if i == 1:
                print_str += '(' + sstr(j) + ')*%s' %(self.parent.gen_symbol)
                continue

            # 将 j 格式化为字符串后加入 print_str，后面加上 '*self.parent.gen_symbol**' + i 的字符串形式
            print_str += '(' + sstr(j) + ')' + '*%s**' %(self.parent.gen_symbol) + sstr(i)

        # 返回打印字符串 print_str
        return print_str

    # 将 __repr__ 方法设置为 __str__ 方法的别名，用于返回对象的字符串表示形式
    __repr__ = __str__

    # 定义 DifferentialOperator 类的相等比较方法，用于判断对象是否相等
    def __eq__(self, other):
        # 如果 other 是 DifferentialOperator 类型，则比较 self.listofpoly 和 other.listofpoly 以及 self.parent 和 other.parent 是否相等
        if isinstance(other, DifferentialOperator):
            return self.listofpoly == other.listofpoly and \
                   self.parent == other.parent
        # 如果 other 不是 DifferentialOperator 类型，则比较 self.listofpoly 的第一个元素是否等于 other，并且后续元素是否全部为 self.parent.base.zero
        return self.listofpoly[0] == other and \
            all(i is self.parent.base.zero for i in self.listofpoly[1:])

    # 定义 DifferentialOperator 类的 is_singular 方法，用于检查在 x0 处微分方程是否奇异
    def is_singular(self, x0):
        """
        Checks if the differential equation is singular at x0.
        """

        # 获取 self.parent.base 的引用
        base = self.parent.base
        # 将 self.listofpoly 的最后一个多项式转换为 sympy 格式，并检查 x0 是否为其根
        return x0 in roots(base.to_sympy(self.listofpoly[-1]), self.x)
# 定义一个类 HolonomicFunction，表示具有多项式系数的线性齐次常微分方程的解
# 这些方程可以用湮灭算子 L 来表示，即 L.f = 0。为确保函数的唯一性，初始条件可以与湮灭算子一起提供。
class HolonomicFunction:
    """
    A Holonomic Function is a solution to a linear homogeneous ordinary
    differential equation with polynomial coefficients. This differential
    equation can also be represented by an annihilator i.e. a Differential
    Operator ``L`` such that :math:`L.f = 0`. For uniqueness of these functions,
    initial conditions can also be provided along with the annihilator.

    Explanation
    ===========

    Holonomic functions have closure properties and thus forms a ring.
    Given two Holonomic Functions f and g, their sum, product,
    integral and derivative is also a Holonomic Function.

    For ordinary points initial condition should be a vector of values of
    the derivatives i.e. :math:`[y(x_0), y'(x_0), y''(x_0) ... ]`.

    For regular singular points initial conditions can also be provided in this
    format:
    :math:`{s0: [C_0, C_1, ...], s1: [C^1_0, C^1_1, ...], ...}`
    where s0, s1, ... are the roots of indicial equation and vectors
    :math:`[C_0, C_1, ...], [C^0_0, C^0_1, ...], ...` are the corresponding initial
    terms of the associated power series. See Examples below.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
    >>> from sympy import QQ
    >>> from sympy import symbols, S
    >>> x = symbols('x')
    >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')

    >>> p = HolonomicFunction(Dx - 1, x, 0, [1])  # e^x
    >>> q = HolonomicFunction(Dx**2 + 1, x, 0, [0, 1])  # sin(x)

    >>> p + q  # annihilator of e^x + sin(x)
    HolonomicFunction((-1) + (1)*Dx + (-1)*Dx**2 + (1)*Dx**3, x, 0, [1, 2, 1])

    >>> p * q  # annihilator of e^x * sin(x)
    HolonomicFunction((2) + (-2)*Dx + (1)*Dx**2, x, 0, [0, 1])

    An example of initial conditions for regular singular points,
    the indicial equation has only one root `1/2`.

    >>> HolonomicFunction(-S(1)/2 + x*Dx, x, 0, {S(1)/2: [1]})
    HolonomicFunction((-1/2) + (x)*Dx, x, 0, {1/2: [1]})

    >>> HolonomicFunction(-S(1)/2 + x*Dx, x, 0, {S(1)/2: [1]}).to_expr()
    sqrt(x)

    To plot a Holonomic Function, one can use `.evalf()` for numerical
    computation. Here's an example on `sin(x)**2/x` using numpy and matplotlib.

    >>> import sympy.holonomic # doctest: +SKIP
    >>> from sympy import var, sin # doctest: +SKIP
    >>> import matplotlib.pyplot as plt # doctest: +SKIP
    >>> import numpy as np # doctest: +SKIP
    >>> var("x") # doctest: +SKIP
    >>> r = np.linspace(1, 5, 100) # doctest: +SKIP
    >>> y = sympy.holonomic.expr_to_holonomic(sin(x)**2/x, x0=1).evalf(r) # doctest: +SKIP
    >>> plt.plot(r, y, label="holonomic function") # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP
    """

    # 定义类属性 _op_priority，表示 HolonomicFunction 对象的运算优先级
    _op_priority = 20
    def __init__(self, annihilator, x, x0=0, y0=None):
        """
        初始化方法，用于创建 HolonomicFunction 对象。

        Parameters
        ==========
        annihilator:
            Holonomic Function 的湮灭算子，使用 DifferentialOperator 对象表示。
        x:
            函数的自变量。
        x0:
            存储初始条件的点。通常是一个整数，默认为零。
        y0:
            初始条件。初始条件的正确格式在类文档字符串中有描述。为了使函数唯一，
            y0 向量的长度应大于或等于微分方程的阶数。
        """

        # 初始条件
        self.y0 = y0
        # 初始条件的点，默认为零。
        self.x0 = x0
        # 湮灭算子 L，使得 L.f = 0
        self.annihilator = annihilator
        self.x = x

    def __str__(self):
        """
        返回对象的字符串表示形式。
        如果存在初始条件，则返回完整的 HolonomicFunction 描述。
        如果不存在初始条件，则只返回 HolonomicFunction 描述。
        """

        if self._have_init_cond():
            str_sol = 'HolonomicFunction(%s, %s, %s, %s)' % (str(self.annihilator),\
                sstr(self.x), sstr(self.x0), sstr(self.y0))
        else:
            str_sol = 'HolonomicFunction(%s, %s)' % (str(self.annihilator),\
                sstr(self.x))

        return str_sol

    __repr__ = __str__

    def unify(self, other):
        """
        统一给定两个 Holonomic 函数的基多项式环。
        """

        R1 = self.annihilator.parent.base
        R2 = other.annihilator.parent.base

        dom1 = R1.dom
        dom2 = R2.dom

        if R1 == R2:
            return (self, other)

        R = (dom1.unify(dom2)).old_poly_ring(self.x)

        newparent, _ = DifferentialOperators(R, str(self.annihilator.parent.gen_symbol))

        sol1 = [R1.to_sympy(i) for i in self.annihilator.listofpoly]
        sol2 = [R2.to_sympy(i) for i in other.annihilator.listofpoly]

        sol1 = DifferentialOperator(sol1, newparent)
        sol2 = DifferentialOperator(sol2, newparent)

        sol1 = HolonomicFunction(sol1, self.x, self.x0, self.y0)
        sol2 = HolonomicFunction(sol2, other.x, other.x0, other.y0)

        return (sol1, sol2)

    def is_singularics(self):
        """
        如果函数具有字典格式的奇异初始条件，则返回 True。
        如果函数具有列表格式的普通初始条件，则返回 False。
        对于所有其他情况，返回 None。
        """

        if isinstance(self.y0, dict):
            return True
        elif isinstance(self.y0, list):
            return False

    def _have_init_cond(self):
        """
        检查函数是否具有初始条件。
        """
        return bool(self.y0)
    def _singularics_to_ord(self):
        """
        Converts a singular initial condition to ordinary if possible.
        """
        # 取第一个初始条件的键
        a = list(self.y0)[0]
        # 取第一个初始条件的值
        b = self.y0[a]

        # 如果只有一个初始条件且该条件为正整数
        if len(self.y0) == 1 and a == int(a) and a > 0:
            # 将 a 转换为整数
            a = int(a)
            # 构造一个新的初始条件列表，包含 0 和从 b 中计算得到的项
            y0 = [S.Zero] * a
            y0 += [j * factorial(a + i) for i, j in enumerate(b)]

            # 返回一个新的 HolonomicFunction 对象，用于表示常微分方程的解
            return HolonomicFunction(self.annihilator, self.x, self.x0, y0)

    def diff(self, *args, **kwargs):
        r"""
        Differentiation of the given Holonomic function.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import ZZ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx**2 + 1, x, 0, [0, 1]).diff().to_expr()
        cos(x)
        >>> HolonomicFunction(Dx - 2, x, 0, [1]).diff().to_expr()
        2*exp(2*x)

        See Also
        ========

        integrate
        """
        # 设置默认的关键字参数 evaluate 为 True
        kwargs.setdefault('evaluate', True)
        # 如果有传入参数
        if args:
            # 如果参数列表中的第一个参数不等于 self.x，返回零
            if args[0] != self.x:
                return S.Zero
            # 如果参数列表长度为 2
            elif len(args) == 2:
                sol = self
                # 对 self 进行 args[1] 次求导
                for i in range(args[1]):
                    sol = sol.diff(args[0])
                return sol

        # 获取湮灭算子
        ann = self.annihilator

        # 如果方程是常数函数
        if ann.listofpoly[0] == ann.parent.base.zero and ann.order == 1:
            return S.Zero

        # 如果微分方程中 y 的系数为零，进行转换计算
        elif ann.listofpoly[0] == ann.parent.base.zero:

            sol = DifferentialOperator(ann.listofpoly[1:], ann.parent)

            if self._have_init_cond():
                # 如果是普通的初始条件
                if self.is_singularics() == False:
                    return HolonomicFunction(sol, self.x, self.x0, self.y0[1:])
                # TODO: 支持奇异初始条件的情况
                return HolonomicFunction(sol, self.x)
            else:
                return HolonomicFunction(sol, self.x)

        # 通用算法
        R = ann.parent.base
        K = R.get_field()

        seq_dmf = [K.new(i.to_list()) for i in ann.listofpoly]

        # 计算右手边的表达式，即 -y = a1*y'/a0 + a2*y''/a0 ... + an*y^n/a0
        rhs = [i / seq_dmf[0] for i in seq_dmf[1:]]
        rhs.insert(0, K.zero)

        # 对左右两侧同时求导
        sol = _derivate_diff_eq(rhs, K)

        # 在 lhs 中加入 y' 项到 rhs 中
        sol = _add_lists(sol, [K.zero, K.one])

        # 对 sol 进行归一化处理，确保系数正确
        sol = _normalize(sol[1:], self.annihilator.parent, negative=False)

        # 如果没有初始条件或者是奇异初始条件
        if not self._have_init_cond() or self.is_singularics() == True:
            return HolonomicFunction(sol, self.x)

        # 扩展初始条件列表 y0 至 sol.order + 1
        y0 = _extend_y0(self, sol.order + 1)[1:]
        return HolonomicFunction(sol, self.x, self.x0, y0)
    # 定义相等运算符重载方法，比较两个对象是否相等
    def __eq__(self, other):
        # 检查两个对象的湮灭子和 x 值是否相同
        if self.annihilator != other.annihilator or self.x != other.x:
            return False
        # 如果两个对象都有初始条件，则比较初始条件 x0 和 y0 是否相同
        if self._have_init_cond() and other._have_init_cond():
            return self.x0 == other.x0 and self.y0 == other.y0
        # 否则返回相等
        return True

    # 右乘运算符重载方法与乘法运算符重载方法相同
    __rmul__ = __mul__

    # 定义减法运算符重载方法
    def __sub__(self, other):
        # 返回自身与 other 的相反数相加的结果
        return self + other * -1

    # 定义反向减法运算符重载方法
    def __rsub__(self, other):
        # 返回 other 与自身的相反数相加的结果
        return self * -1 + other

    # 定义取负运算符重载方法
    def __neg__(self):
        # 返回自身的相反数
        return -1 * self

    # 定义真除运算符重载方法
    def __truediv__(self, other):
        # 返回自身乘以 1/other 的结果
        return self * (S.One / other)

    # 定义乘方运算符重载方法
    def __pow__(self, n):
        # 如果湮灭子的阶数小于等于1，则处理特殊情况
        if self.annihilator.order <= 1:
            ann = self.annihilator
            parent = ann.parent

            # 如果 y0 为 None，则 y0 也为 None；否则计算 y0 的乘方
            if self.y0 is None:
                y0 = None
            else:
                y0 = [list(self.y0)[0] ** n]

            # 获取湮灭子的第一个和第二个多项式
            p0 = ann.listofpoly[0]
            p1 = ann.listofpoly[1]

            # 计算 p0 的乘以 n 后的新多项式
            p0 = (Poly.new(p0, self.x) * n).rep

            # 转换为 sympy 格式的解
            sol = [parent.base.to_sympy(i) for i in [p0, p1]]
            dd = DifferentialOperator(sol, parent)
            # 返回一个新的 HolonomicFunction 对象
            return HolonomicFunction(dd, self.x, self.x0, y0)
        
        # 如果 n 小于 0，则抛出异常
        if n < 0:
            raise NotHolonomicError("Negative Power on a Holonomic Function")
        
        # 获取导数操作符 Dx
        Dx = self.annihilator.parent.derivative_operator
        # 创建一个零阶 HolonomicFunction 对象
        result = HolonomicFunction(Dx, self.x, S.Zero, [S.One])
        
        # 如果 n 等于 0，则直接返回 result
        if n == 0:
            return result
        
        # 使用快速幂算法计算自身的 n 次幂
        x = self
        while True:
            if n % 2:
                result *= x
            n >>= 1
            if not n:
                break
            x *= x
        return result

    # 返回湮灭子的多项式中 x 的最高次数
    def degree(self):
        """
        Returns the highest power of `x` in the annihilator.
        """
        return max(i.degree() for i in self.annihilator.listofpoly)
    # 返回经过一个保持性函数和一个代数函数组合后的函数。
    # 该方法无法自行计算结果函数的初始条件，因此也可以手动提供。

    R = self.annihilator.parent  # 获取湮没算子的父环
    a = self.annihilator.order  # 获取湮没算子的阶数
    diff = expr.diff(self.x)  # 计算表达式关于自变量 x 的导数
    listofpoly = self.annihilator.listofpoly  # 获取湮没算子的多项式列表

    # 将多项式列表中的每个多项式转换为 Sympy 表达式
    for i, j in enumerate(listofpoly):
        if isinstance(j, self.annihilator.parent.base.dtype):
            listofpoly[i] = self.annihilator.parent.base.to_sympy(j)

    r = listofpoly[a].subs({self.x:expr})  # 将表达式代入到多项式列表的第 a 项中
    subs = [-listofpoly[i].subs({self.x:expr}) / r for i in range (a)]  # 计算替换项的列表

    coeffs = [S.Zero for i in range(a)]  # coeffs[i] == (D^i f)(a) 在 D^k (f(a)) 中的系数
    coeffs[0] = S.One  # 初始化第一个系数为 1
    system = [coeffs]  # 初始化线性方程组

    homogeneous = Matrix([[S.Zero for i in range(a)]]).transpose()  # 初始化零向量的列矩阵
    while True:
        coeffs_next = [p.diff(self.x) for p in coeffs]  # 计算下一个系数的导数
        for i in range(a - 1):
            coeffs_next[i + 1] += (coeffs[i] * diff)
        for i in range(a):
            coeffs_next[i] += (coeffs[-1] * subs[i] * diff)
        coeffs = coeffs_next  # 更新系数列表

        # 检查线性关系
        system.append(coeffs)
        sol, taus = (Matrix(system).transpose()
            ).gauss_jordan_solve(homogeneous)  # 使用 Gauss-Jordan 方法解线性方程组
        if sol.is_zero_matrix is not True:  # 如果找到非零解则跳出循环
            break

    tau = list(taus)[0]  # 获取第一个解
    sol = sol.subs(tau, 1)  # 将解中的 tau 替换为 1
    sol = _normalize(sol[0:], R, negative=False)  # 标准化解

    # 如果为结果函数提供了初始条件
    if args:
        return HolonomicFunction(sol, self.x, args[0], args[1])  # 返回结果函数及其初始条件
    return HolonomicFunction(sol, self.x)  # 返回结果函数
    def series(self, n=6, coefficient=False, order=True, _recur=None):
        r"""
        Finds the power series expansion of given holonomic function about :math:`x_0`.

        Explanation
        ===========

        A list of series might be returned if :math:`x_0` is a regular point with
        multiple roots of the indicial equation.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx - 1, x, 0, [1]).series()  # e^x
        1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + O(x**6)
        >>> HolonomicFunction(Dx**2 + 1, x, 0, [0, 1]).series(n=8)  # sin(x)
        x - x**3/6 + x**5/120 - x**7/5040 + O(x**8)

        See Also
        ========

        HolonomicFunction.to_sequence
        """

        # If no recursion provided, convert self to sequence
        if _recur is None:
            recurrence = self.to_sequence()
        else:
            recurrence = _recur

        # Handling different forms of recurrence relations
        if isinstance(recurrence, tuple) and len(recurrence) == 2:
            recurrence = recurrence[0]
            constantpower = 0
        elif isinstance(recurrence, tuple) and len(recurrence) == 3:
            constantpower = recurrence[1]
            recurrence = recurrence[0]
        elif len(recurrence) == 1 and len(recurrence[0]) == 2:
            recurrence = recurrence[0][0]
            constantpower = 0
        elif len(recurrence) == 1 and len(recurrence[0]) == 3:
            constantpower = recurrence[0][1]
            recurrence = recurrence[0][0]
        else:
            # Recursively call series for each element in recurrence
            return [self.series(_recur=i) for i in recurrence]

        # Adjust n based on the constant power in the recurrence relation
        n = n - int(constantpower)
        l = len(recurrence.u0) - 1  # Length of initial conditions
        k = recurrence.recurrence.order  # Order of the recurrence relation
        x = self.x  # Variable symbol
        x0 = self.x0  # Expansion point
        seq_dmp = recurrence.recurrence.listofpoly  # Sequence of polynomials
        R = recurrence.recurrence.parent.base  # Base ring of recurrence
        K = R.get_field()  # Field associated with the base ring
        seq = [K.new(j.to_list()) for j in seq_dmp]  # Convert polynomials to field elements
        sub = [-seq[i] / seq[k] for i in range(k)]  # Substitutions for recurrence relation
        sol = list(recurrence.u0)  # Initial conditions as a list

        # Compute additional terms beyond initial conditions if necessary
        if l + 1 < n:
            for i in range(l + 1 - k, n - k):
                # Compute coefficient using the substitution and initial conditions
                coeff = sum((DMFsubs(sub[j], i) * sol[i + j]
                            for j in range(k) if i + j >= 0), start=S.Zero)
                sol.append(coeff)

        # Return coefficients if coefficient=True
        if coefficient:
            return sol

        # Compute the series expansion
        ser = sum((x**(i + constantpower) * j for i, j in enumerate(sol)),
                  start=S.Zero)

        # Include the order term if order=True
        if order:
            ser += Order(x**(n + int(constantpower)), x)

        # Adjust series if expansion point x0 is not zero
        if x0 != 0:
            return ser.subs(x, x - x0)

        return ser
    # 定义一个方法 _indicial，用于计算 Indicial 方程的根

    # 检查初始条件，如果 self.x0 不为零，则进行 x0 的偏移后递归调用 _indicial()
    if self.x0 != 0:
        return self.shift_x(self.x0)._indicial()

    # 获取消子多项式列表
    list_coeff = self.annihilator.listofpoly
    # 获取基环域 R
    R = self.annihilator.parent.base
    # 获取变量 x
    x = self.x
    # 初始化 s 为零
    s = R.zero
    # 初始化 y 为单位元
    y = R.one

    # 定义内部函数 _pole_degree，用于计算多项式的极点度数
    def _pole_degree(poly):
        # 将多项式转换为 SymPy 格式，并计算其根
        root_all = roots(R.to_sympy(poly), x, filter='Z')
        # 如果有根为零，则返回该根的值，否则返回零
        if 0 in root_all.keys():
            return root_all[0]
        else:
            return 0

    # 计算消子多项式的最高次数
    degree = max(j.degree() for j in list_coeff)
    # 设置无穷大的值作为初始值
    inf = 10 * (max(1, degree) + max(1, self.annihilator.order))

    # 定义函数 deg 用于计算多项式的极点度数
    deg = lambda q: inf if q.is_zero else _pole_degree(q)
    # 计算偏移量 b
    b = min(deg(q) - j for j, q in enumerate(list_coeff))

    # 遍历消子多项式列表
    for i, j in enumerate(list_coeff):
        # 获取当前多项式的系数列表
        listofdmp = j.all_coeffs()
        # 计算当前多项式的最高次数
        degree = len(listofdmp) - 1
        # 如果偏移量 i + b 在有效范围内，则累加对应系数乘以 y 到 s 中
        if 0 <= i + b <= degree:
            s = s + listofdmp[degree - i - b] * y
        # 更新 y 为 R 格式的多项式 (x - i)
        y *= R.from_sympy(x - i)

    # 返回 s 的根作为 Indicial 方程的解集
    return roots(R.to_sympy(s), x)
    def evalf(self, points, method='RK4', h=0.05, derivatives=False):
        r"""
        Finds numerical value of a holonomic function using numerical methods.
        (RK4 by default). A set of points (real or complex) must be provided
        which will be the path for the numerical integration.

        Explanation
        ===========

        The path should be given as a list :math:`[x_1, x_2, \dots x_n]`. The numerical
        values will be computed at each point in this order
        :math:`x_1 \rightarrow x_2 \rightarrow x_3 \dots \rightarrow x_n`.

        Returns values of the function at :math:`x_1, x_2, \dots x_n` in a list.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')

        A straight line on the real axis from (0 to 1)

        >>> r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        Runge-Kutta 4th order on e^x from 0.1 to 1.
        Exact solution at 1 is 2.71828182845905

        >>> HolonomicFunction(Dx - 1, x, 0, [1]).evalf(r)
        [1.10517083333333, 1.22140257085069, 1.34985849706254, 1.49182424008069,
        1.64872063859684, 1.82211796209193, 2.01375162659678, 2.22553956329232,
        2.45960141378007, 2.71827974413517]

        Euler's method for the same

        >>> HolonomicFunction(Dx - 1, x, 0, [1]).evalf(r, method='Euler')
        [1.1, 1.21, 1.331, 1.4641, 1.61051, 1.771561, 1.9487171, 2.14358881,
        2.357947691, 2.5937424601]

        One can also observe that the value obtained using Runge-Kutta 4th order
        is much more accurate than Euler's method.
        """

        # 导入私有函数 `_evalf` 用于计算数值结果
        from sympy.holonomic.numerical import _evalf
        
        # 初始化 lp 为 False
        lp = False

        # 如果 points 不可迭代，说明只有一个点 b
        if not hasattr(points, "__iter__"):
            lp = True
            # 将 points 转换为 SymPy 对象
            b = S(points)
            # 如果初始点 self.x0 等于目标点 b，则直接返回最后一个值的数值结果
            if self.x0 == b:
                return _evalf(self, [b], method=method, derivatives=derivatives)[-1]

            # 如果目标点 b 不是数值，则抛出 NotImplementedError
            if not b.is_Number:
                raise NotImplementedError

            # 设置起始点 a 为 self.x0
            a = self.x0
            # 如果 a 大于 b，则反向步长为负数
            if a > b:
                h = -h
            # 计算点的数量 n
            n = int((b - a) / h)
            # 初始化 points 列表，并生成等差数列
            points = [a + h]
            for i in range(n - 1):
                points.append(points[-1] + h)

        # 遍历根据 annihilator 列表生成的多项式的根
        for i in roots(self.annihilator.parent.base.to_sympy(self.annihilator.listofpoly[-1]), self.x):
            # 如果根 i 等于初始点 self.x0 或者在给定点 points 中，则抛出 SingularityError
            if i == self.x0 or i in points:
                raise SingularityError(self, i)

        # 如果 lp 为 True，则返回数值结果列表的最后一个值
        if lp:
            return _evalf(self, points, method=method, derivatives=derivatives)[-1]
        # 否则返回数值结果列表
        return _evalf(self, points, method=method, derivatives=derivatives)
    def change_x(self, z):
        """
        Changes only the variable of Holonomic Function, for internal
        purposes. For composition use HolonomicFunction.composition()
        """

        # 获取当前 Holonomic 函数的定义域
        dom = self.annihilator.parent.base.dom
        # 使用给定的变量 z 构建旧的多项式环
        R = dom.old_poly_ring(z)
        # 创建微分算子环的父环和微分算子
        parent, _ = DifferentialOperators(R, 'Dx')
        # 将自身的多项式列表转换为给定环的解
        sol = [R(j.to_list()) for j in self.annihilator.listofpoly]
        # 构建新的微分算子
        sol = DifferentialOperator(sol, parent)
        # 返回使用新变量 z 构建的 HolonomicFunction 对象
        return HolonomicFunction(sol, z, self.x0, self.y0)

    def shift_x(self, a):
        """
        Substitute `x + a` for `x`.
        """

        # 获取当前 Holonomic 函数的变量 x
        x = self.x
        # 获取当前 Holonomic 函数的多项式列表
        listaftershift = self.annihilator.listofpoly
        # 获取当前 Holonomic 函数的基环
        base = self.annihilator.parent.base

        # 将多项式列表中的每个多项式中的 x 替换为 x + a
        sol = [base.from_sympy(base.to_sympy(i).subs(x, x + a)) for i in listaftershift]
        # 构建新的微分算子
        sol = DifferentialOperator(sol, self.annihilator.parent)
        # 更新新的初始点 x0
        x0 = self.x0 - a
        # 如果没有初始条件，则返回使用新变量 x 构建的 HolonomicFunction 对象
        if not self._have_init_cond():
            return HolonomicFunction(sol, x)
        # 否则返回使用新变量 x 和更新的初始点 x0 构建的 HolonomicFunction 对象
        return HolonomicFunction(sol, x, x0, self.y0)

    def to_expr(self):
        """
        Converts a Holonomic Function back to elementary functions.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import ZZ
        >>> from sympy import symbols, S
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(x**2*Dx**2 + x*Dx + (x**2 - 1), x, 0, [0, S(1)/2]).to_expr()
        besselj(1, x)
        >>> HolonomicFunction((1 + x)*Dx**3 + Dx**2, x, 0, [1, 1, 1]).to_expr()
        x*log(x + 1) + log(x + 1) + 1

        """

        # 将 Holonomic Function 转换为基本函数表达式
        return hyperexpand(self.to_hyper()).simplify()

    def change_ics(self, b, lenics=None):
        """
        Changes the point `x0` to ``b`` for initial conditions.

        Examples
        ========

        >>> from sympy.holonomic import expr_to_holonomic
        >>> from sympy import symbols, sin, exp
        >>> x = symbols('x')

        >>> expr_to_holonomic(sin(x)).change_ics(1)
        HolonomicFunction((1) + (1)*Dx**2, x, 1, [sin(1), cos(1)])

        >>> expr_to_holonomic(exp(x)).change_ics(2)
        HolonomicFunction((-1) + (1)*Dx, x, 2, [exp(2)])
        """

        # 初始化符号变量为 True
        symbolic = True

        # 如果初始条件的长度未指定且 y0 的长度大于微分算子的阶数，则使用 y0 的长度
        if lenics is None and len(self.y0) > self.annihilator.order:
            lenics = len(self.y0)
        # 获取当前 Holonomic 函数的基环
        dom = self.annihilator.parent.base.domain

        try:
            # 尝试将表达式转换为 HolonomicFunction，指定新的初始点 b 和长度 lenics
            sol = expr_to_holonomic(self.to_expr(), x=self.x, x0=b, lenics=lenics, domain=dom)
        except (NotPowerSeriesError, NotHyperSeriesError):
            # 如果转换失败，将符号变量置为 False
            symbolic = False

        # 如果转换成功且新的初始点与当前 HolonomicFunction 的初始点相同，则直接返回结果
        if symbolic and sol.x0 == b:
            return sol

        # 否则，重新计算在新初始点 b 处的函数值 y0，并返回新的 HolonomicFunction 对象
        y0 = self.evalf(b, derivatives=True)
        return HolonomicFunction(self.annihilator, self.x, b, y0)
    # 将当前对象转换为 Meijer G 函数的线性组合

    # 首先将当前对象转换为超几何函数的表达式列表
    rep = self.to_hyper(as_list=True)
    
    # 初始化结果为零
    sol = S.Zero

    # 遍历超几何函数的表达式列表
    for i in rep:
        # 如果表达式长度为1，直接加到结果中
        if len(i) == 1:
            sol += i[0]

        # 如果表达式长度为2，将其转换为 Meijer G 函数并加到结果中
        elif len(i) == 2:
            sol += i[0] * _hyper_to_meijerg(i[1])

    # 返回最终的 Meijer G 函数的线性组合结果
    return sol
# 将超几何函数转换为霍洛莫尼克函数
def from_hyper(func, x0=0, evalf=False):
    r"""
    Converts a hypergeometric function to holonomic.
    ``func`` is the Hypergeometric Function and ``x0`` is the point at
    which initial conditions are required.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import from_hyper
    >>> from sympy import symbols, hyper, S
    >>> x = symbols('x')
    >>> from_hyper(hyper([], [S(3)/2], x**2/4))
    HolonomicFunction((-x) + (2)*Dx + (x)*Dx**2, x, 1, [sinh(1), -sinh(1) + cosh(1)])
    """

    # 获取超几何函数的参数 ap 和 bq
    a = func.ap
    b = func.bq
    # 获取超几何函数的第三个参数，即自变量
    z = func.args[2]
    # 从自变量中找到唯一的符号变量 x
    x = z.atoms(Symbol).pop()
    # 创建有理系数环 QQ 上的微分算子环 R 和 Dx
    R, Dx = DifferentialOperators(QQ.old_poly_ring(x), 'Dx')

    # 构建广义超几何微分方程的左手边 x*Dx
    xDx = x*Dx
    r1 = 1
    # 计算超几何函数的微分方程的左手边 r1
    for ai in a:
        r1 *= xDx + ai
    # 构建广义超几何微分方程的右手边 x*Dx - 1
    xDx_1 = xDx - 1
    r2 = Dx
    # 计算超几何函数的微分方程的右手边 r2
    for bi in b:
        r2 *= xDx_1 + bi
    # 求解超几何函数的微分方程得到的解
    sol = r1 - r2

    # 简化超几何函数，得到简化后的表达式 simp
    simp = hyperexpand(func)

    # 如果简化后的表达式为无穷大或负无穷大，则返回合成后的霍洛莫尼克函数
    if simp in (Infinity, NegativeInfinity):
        return HolonomicFunction(sol, x).composition(z)

    # 定义内部函数 _find_conditions 用于寻找初始条件
    def _find_conditions(simp, x, x0, order, evalf=False):
        y0 = []
        for i in range(order):
            if evalf:
                val = simp.subs(x, x0).evalf()
            else:
                val = simp.subs(x, x0)
            # 如果值为无穷或 NaN，则返回 None
            if val.is_finite is False or isinstance(val, NaN):
                return None
            y0.append(val)
            simp = simp.diff(x)
        return y0

    # 如果简化后的函数不是超几何函数，而是已知的符号函数
    if not isinstance(simp, hyper):
        # 寻找初始条件 y0
        y0 = _find_conditions(simp, x, x0, sol.order)
        while not y0:
            # 如果在 x0 处不存在值，则尝试在 x0+1 处寻找初始条件
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order)

        # 返回合成后的霍洛莫尼克函数
        return HolonomicFunction(sol, x).composition(z, x0, y0)

    # 如果简化后的函数仍然是超几何函数
    if isinstance(simp, hyper):
        x0 = 1
        # 尝试在 x0 处寻找初始条件 y0
        y0 = _find_conditions(simp, x, x0, sol.order, evalf)
        while not y0:
            # 如果在 x0 处不存在值，则尝试在 x0+1 处寻找初始条件
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order, evalf)
        # 返回合成后的霍洛莫尼克函数
        return HolonomicFunction(sol, x).composition(z, x0, y0)

    # 如果简化后的函数类型未知，则返回合成后的霍洛莫尼克函数
    return HolonomicFunction(sol, x).composition(z)
    # 从 func 对象中获取属性 bq
    b = func.bq
    # 获取 func.an 列表的长度
    n = len(func.an)
    # 获取 func.bm 列表的长度
    m = len(func.bm)
    # 获取列表 a 的长度
    p = len(a)
    # 从 func.args 中获取第三个元素
    z = func.args[2]
    # 从 z 中获取一个 Symbol 类型的原子，并将其从 z 中弹出
    x = z.atoms(Symbol).pop()
    # 使用 domain.old_poly_ring(x) 创建 DifferentialOperators 对象，并分别赋值给 R 和 Dx
    R, Dx = DifferentialOperators(domain.old_poly_ring(x), 'Dx')

    # 计算满足 Meijer G-函数的微分方程
    xDx = x*Dx
    xDx1 = xDx + 1
    r1 = x*(-1)**(m + n - p)
    # 遍历列表 a 中的每个元素 ai
    for ai in a:  # XXX gives sympify error if args given in list
        # 将 r1 乘以 xDx1 - ai
        r1 *= xDx1 - ai
    # 使用列表 b 中的每个元素 bi 来计算 r2
    r2 = 1
    for bi in b:
        r2 *= xDx - bi
    # 构造微分方程的解 sol
    sol = r1 - r2

    # 如果没有初始条件 initcond，则返回 HolonomicFunction(sol, x) 组合 z 后的结果
    if not initcond:
        return HolonomicFunction(sol, x).composition(z)

    # 对 func 进行超级展开
    simp = hyperexpand(func)

    # 如果超级展开结果 simp 是 Infinity 或 NegativeInfinity，则返回 HolonomicFunction(sol, x) 组合 z 后的结果
    if simp in (Infinity, NegativeInfinity):
        return HolonomicFunction(sol, x).composition(z)

    # 定义内部函数 _find_conditions，用于找到初始条件
    def _find_conditions(simp, x, x0, order, evalf=False):
        y0 = []
        # 对于每个阶数，计算初始条件
        for i in range(order):
            if evalf:
                val = simp.subs(x, x0).evalf()
            else:
                val = simp.subs(x, x0)
            # 如果值不是有限的或者是 NaN 类型，则返回 None
            if val.is_finite is False or isinstance(val, NaN):
                return None
            y0.append(val)
            simp = simp.diff(x)
        return y0

    # 计算初始条件
    if not isinstance(simp, meijerg):
        # 获取初始条件 y0
        y0 = _find_conditions(simp, x, x0, sol.order)
        # 当 y0 为空时，增加 x0 直到找到非空的 y0
        while not y0:
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order)

        # 返回 HolonomicFunction(sol, x) 组合 z，并带有初始条件 x0 和 y0
        return HolonomicFunction(sol, x).composition(z, x0, y0)

    # 如果 simp 是 meijerg 类型
    if isinstance(simp, meijerg):
        x0 = 1
        # 获取初始条件 y0
        y0 = _find_conditions(simp, x, x0, sol.order, evalf)
        # 当 y0 为空时，增加 x0 直到找到非空的 y0
        while not y0:
            x0 += 1
            y0 = _find_conditions(simp, x, x0, sol.order, evalf)

        # 返回 HolonomicFunction(sol, x) 组合 z，并带有初始条件 x0 和 y0
        return HolonomicFunction(sol, x).composition(z, x0, y0)

    # 默认情况下，返回 HolonomicFunction(sol, x) 组合 z
    return HolonomicFunction(sol, x).composition(z)
x_1 = Dummy('x_1')
_lookup_table = None
domain_for_table = None
from sympy.integrals.meijerint import _mytype

# 将表达式转换为霍洛莫尼克函数
def expr_to_holonomic(func, x=None, x0=0, y0=None, lenics=None, domain=None, initcond=True):
    """
    Converts a function or an expression to a holonomic function.

    Parameters
    ==========

    func:
        The expression to be converted.
    x:
        variable for the function.
    x0:
        point at which initial condition must be computed.
    y0:
        One can optionally provide initial condition if the method
        is not able to do it automatically.
    lenics:
        Number of terms in the initial condition. By default it is
        equal to the order of the annihilator.
    domain:
        Ground domain for the polynomials in ``x`` appearing as coefficients
        in the annihilator.
    initcond:
        Set it false if you do not want the initial conditions to be computed.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import expr_to_holonomic
    >>> from sympy import sin, exp, symbols
    >>> x = symbols('x')
    >>> expr_to_holonomic(sin(x))
    HolonomicFunction((1) + (1)*Dx**2, x, 0, [0, 1])
    >>> expr_to_holonomic(exp(x))
    HolonomicFunction((-1) + (1)*Dx, x, 0, [1])

    See Also
    ========

    sympy.integrals.meijerint._rewrite1, _convert_poly_rat_alg, _create_table
    """
    func = sympify(func)  # 将输入的 func 转换为 SymPy 表达式
    syms = func.free_symbols  # 获取 func 中的自由符号集合

    if not x:
        if len(syms) == 1:
            x = syms.pop()  # 如果未提供 x，则从 func 的自由符号中弹出一个作为 x
        else:
            raise ValueError("Specify the variable for the function")  # 如果无法确定 x，则引发异常
    elif x in syms:
        syms.remove(x)  # 如果提供了 x，并且它在 func 的自由符号中，则将其移除

    extra_syms = list(syms)

    if domain is None:
        if func.has(Float):
            domain = RR  # 如果 func 中包含浮点数，则使用实数域 RR
        else:
            domain = QQ  # 否则使用有理数域 QQ
        if len(extra_syms) != 0:
            domain = domain[extra_syms].get_field()  # 如果有额外的符号，则基于这些符号获取域

    # 尝试将 func 转换为多项式或有理函数的解
    solpoly = _convert_poly_rat_alg(func, x, x0=x0, y0=y0, lenics=lenics, domain=domain, initcond=initcond)
    if solpoly:
        return solpoly  # 如果成功转换，则返回解

    # 创建查找表
    global _lookup_table, domain_for_table
    if not _lookup_table:
        domain_for_table = domain
        _lookup_table = {}
        _create_table(_lookup_table, domain=domain)  # 如果查找表不存在，则创建并初始化之
    elif domain != domain_for_table:
        domain_for_table = domain
        _lookup_table = {}
        _create_table(_lookup_table, domain=domain)  # 如果域发生变化，则重新创建查找表

    # 使用查找表直接转换为霍洛莫尼克函数
    # 如果 func 是一个函数对象
    if func.is_Function:
        # 在点 x=x_1 处对 func 进行代换，得到新的函数 f
        f = func.subs(x, x_1)
        # 根据函数 f 和点 x_1，确定函数类型 t
        t = _mytype(f, x_1)
        # 如果 t 在预先定义的查找表 _lookup_table 中
        if t in _lookup_table:
            # 从查找表中获取相关信息 l，并在点 x 处更改函数参数
            l = _lookup_table[t]
            sol = l[0][1].change_x(x)
        else:
            # 如果 t 不在查找表中，尝试进行 Meijer G 函数的积分转换
            sol = _convert_meijerint(func, x, initcond=False, domain=domain)
            # 如果无法成功转换，抛出 NotImplementedError 异常
            if not sol:
                raise NotImplementedError
            # 如果提供了初始值 y0，则将其赋给解 sol 的初始值
            if y0:
                sol.y0 = y0
            # 如果提供了初始值 y0 或者没有初始条件 initcond，设置解的初始点 x0，并返回解
            if y0 or not initcond:
                sol.x0 = x0
                return sol
            # 如果未指定 lenics（初始条件的长度），则根据解的湮灭算子的阶数设定 lenics
            if not lenics:
                lenics = sol.annihilator.order
            # 在初始点 x0 处找到初始条件 _y0
            _y0 = _find_conditions(func, x, x0, lenics)
            # 如果未找到初始条件，则逐步增加 x0 直至找到合适的初始条件
            while not _y0:
                x0 += 1
                _y0 = _find_conditions(func, x, x0, lenics)
            # 返回一个带有特定初始条件的 HolonomicFunction 对象
            return HolonomicFunction(sol.annihilator, x, x0, _y0)

        # 如果提供了初始值 y0 或者没有初始条件 initcond
        if y0 or not initcond:
            # 将解 sol 与 func 的第一个参数进行组合
            sol = sol.composition(func.args[0])
            # 如果提供了初始值 y0，则将其赋给解 sol 的初始值
            if y0:
                sol.y0 = y0
            # 设置解的初始点 x0，并返回解
            sol.x0 = x0
            return sol
        # 如果未指定 lenics（初始条件的长度），则根据解的湮灭算子的阶数设定 lenics
        if not lenics:
            lenics = sol.annihilator.order

        # 在初始点 x0 处找到初始条件 _y0
        _y0 = _find_conditions(func, x, x0, lenics)
        # 如果未找到初始条件，则逐步增加 x0 直至找到合适的初始条件
        while not _y0:
            x0 += 1
            _y0 = _find_conditions(func, x, x0, lenics)
        # 返回一个带有特定初始条件的 HolonomicFunction 对象
        return sol.composition(func.args[0], x0, _y0)

    # 如果不是函数对象，递归地迭代表达式中的每个参数
    args = func.args
    # 获取 func 的函数部分 f 和参数部分 sol
    f = func.func
    # 将表达式转换为相应的 HolonomicFunction 对象
    sol = expr_to_holonomic(args[0], x=x, initcond=False, domain=domain)

    # 如果 f 是加法运算
    if f is Add:
        # 对表达式的每个参数，逐个转换为 HolonomicFunction 对象并相加
        for i in range(1, len(args)):
            sol += expr_to_holonomic(args[i], x=x, initcond=False, domain=domain)

    # 如果 f 是乘法运算
    elif f is Mul:
        # 对表达式的每个参数，逐个转换为 HolonomicFunction 对象并相乘
        for i in range(1, len(args)):
            sol *= expr_to_holonomic(args[i], x=x, initcond=False, domain=domain)

    # 如果 f 是指数运算
    elif f is Pow:
        # 将表达式转换为 HolonomicFunction 对象并求其指数次幂
        sol = sol**args[1]

    # 设置解的初始点 x0
    sol.x0 = x0
    # 如果解为空，则抛出 NotImplementedError 异常
    if not sol:
        raise NotImplementedError
    # 如果提供了初始值 y0，则将其赋给解 sol 的初始值
    if y0:
        sol.y0 = y0
    # 如果提供了初始值 y0 或者没有初始条件 initcond，则返回解
    if y0 or not initcond:
        return sol
    # 如果解具有初始值 y0，则返回解
    if sol.y0:
        return sol
    # 如果未指定 lenics（初始条件的长度），则根据解的湮灭算子的阶数设定 lenics
    if not lenics:
        lenics = sol.annihilator.order
    # 如果解的湮灭算子在 x0 处是奇异的
    if sol.annihilator.is_singular(x0):
        # 计算解的指标
        r = sol._indicial()
        l = list(r)
        # 如果只有一个指标且其对应的值为 1
        if len(r) == 1 and r[l[0]] == S.One:
            r = l[0]
            # 构造新的函数 g，用于计算奇异初始条件
            g = func / (x - x0)**r
            # 寻找奇异初始条件并进行归一化处理
            singular_ics = _find_conditions(g, x, x0, lenics)
            singular_ics = [j / factorial(i) for i, j in enumerate(singular_ics)]
            y0 = {r:singular_ics}
            # 返回一个带有特定奇异初始条件的 HolonomicFunction 对象
            return HolonomicFunction(sol.annihilator, x, x0, y0)

    # 在初始点 x0 处找到初始条件 _y0
    _y0 = _find_conditions(func, x, x0, lenics)
    # 如果未找到初始条件，则逐步增加 x0 直至找到合适的初始条件
    while not _y0:
        x0 += 1
        _y0 = _find_conditions(func, x, x0, lenics)

    # 返回一个带有特定初始条件的 HolonomicFunction 对象
    return HolonomicFunction(sol.annihilator, x, x0, _y0)
## Some helper functions ##

# 标准化给定的湮灭器列表
def _normalize(list_of, parent, negative=True):
    """
    Normalize a given annihilator
    给定一个湮灭算子，进行标准化
    """

    num = []  # 存储分子的列表
    denom = []  # 存储分母的列表
    base = parent.base  # 获取父类的基础信息
    K = base.get_field()  # 获取关联的域
    lcm_denom = base.from_sympy(S.One)  # 初始化最小公倍数的分母
    list_of_coeff = []  # 存储系数列表

    # 将多项式转换为关联分式域中的元素
    for i, j in enumerate(list_of):
        if isinstance(j, base.dtype):
            list_of_coeff.append(K.new(j.to_list()))
        elif not isinstance(j, K.dtype):
            list_of_coeff.append(K.from_sympy(sympify(j)))
        else:
            list_of_coeff.append(j)

        # 相应的分子
        num.append(list_of_coeff[i].numer())

        # 相应的分母
        denom.append(list_of_coeff[i].denom())

    # 计算系数中分母的最小公倍数
    for i in denom:
        lcm_denom = i.lcm(lcm_denom)

    if negative:
        lcm_denom = -lcm_denom

    lcm_denom = K.new(lcm_denom.to_list())

    # 将系数乘以最小公倍数
    for i, j in enumerate(list_of_coeff):
        list_of_coeff[i] = j * lcm_denom

    gcd_numer = base((list_of_coeff[-1].numer() / list_of_coeff[-1].denom()).to_list())

    # 计算系数中分子的最大公约数
    for i in num:
        gcd_numer = i.gcd(gcd_numer)

    gcd_numer = K.new(gcd_numer.to_list())

    # 将所有系数除以最大公约数
    for i, j in enumerate(list_of_coeff):
        frac_ans = j / gcd_numer
        list_of_coeff[i] = base((frac_ans.numer() / frac_ans.denom()).to_list())

    return DifferentialOperator(list_of_coeff, parent)


# 导数方程的导数
def _derivate_diff_eq(listofpoly, K):
    """
    Let a differential equation a0(x)y(x) + a1(x)y'(x) + ... = 0
    where a0, a1,... are polynomials or rational functions. The function
    returns b0, b1, b2... such that the differential equation
    b0(x)y(x) + b1(x)y'(x) +... = 0 is formed after differentiating the
    former equation.
    给定微分方程 a0(x)y(x) + a1(x)y'(x) + ... = 0，其中 a0, a1,... 是多项式或有理函数。
    函数返回 b0, b1, b2...，使得在对前述方程求导后形成微分方程
    b0(x)y(x) + b1(x)y'(x) +... = 0。
    """

    sol = []  # 存储解的列表
    a = len(listofpoly) - 1
    sol.append(DMFdiff(listofpoly[0], K))  # 计算第一个解

    for i, j in enumerate(listofpoly[1:]):
        sol.append(DMFdiff(j, K) + listofpoly[i])  # 计算后续的解

    sol.append(listofpoly[a])
    return sol


# 将超几何函数转换为 MeijerG 函数
def _hyper_to_meijerg(func):
    """
    Converts a `hyper` to meijerg.
    将超几何函数转换为 MeijerG 函数。
    """
    ap = func.ap
    bq = func.bq

    if any(i <= 0 and int(i) == i for i in ap):
        return hyperexpand(func)

    z = func.args[2]

    # `meijerg` 函数的参数
    an = (1 - i for i in ap)
    anp = ()
    bm = (S.Zero, )
    bmq = (1 - i for i in bq)

    k = S.One

    for i in bq:
        k = k * gamma(i)

    for i in ap:
        k = k / gamma(i)

    return k * meijerg(an, anp, bm, bmq, -z)


# 添加两个列表的多项式序列
def _add_lists(list1, list2):
    """Takes polynomial sequences of two annihilators a and b and returns
    the list of polynomials of sum of a and b.
    接受两个湮灭算子 a 和 b 的多项式序列，并返回其和的多项式列表。
    """
    if len(list1) <= len(list2):
        sol = [a + b for a, b in zip(list1, list2)] + list2[len(list1):]
    else:
        # 使用列表推导式对 list1 和 list2 中对应位置的元素进行相加，生成新的列表 sol
        sol = [a + b for a, b in zip(list1, list2)]
        # 将 list1 中剩余未加的元素添加到 sol 中
        sol += list1[len(list2):]
    # 返回计算结果 sol
    return sol
# 将给定的函数 `func` 转换为其在 `x` 上的多项式环中的表示。
def _convert_poly_rat_alg(func, x, x0=0, y0=None, lenics=None, domain=QQ, initcond=True):
    """
    Converts polynomials, rationals and algebraic functions to holonomic.
    """

    # 检查函数是否为多项式
    ispoly = func.is_polynomial()
    if not ispoly:
        # 检查函数是否为有理函数
        israt = func.is_rational_function()
    else:
        israt = True

    # 如果函数既不是多项式也不是有理函数，则尝试将其表示为基数和指数的形式
    if not (ispoly or israt):
        basepoly, ratexp = func.as_base_exp()
        # 如果基数是多项式且指数是数值类型
        if basepoly.is_polynomial() and ratexp.is_Number:
            # 处理指数为浮点数的情况，尝试简化为分数形式
            if isinstance(ratexp, Float):
                ratexp = nsimplify(ratexp)
            m, n = ratexp.p, ratexp.q
            is_alg = True  # 函数是代数函数
        else:
            is_alg = False  # 函数不是代数函数
    else:
        is_alg = True  # 函数是代数函数

    # 如果函数既不是多项式、有理函数，也不是代数函数，则返回空值
    if not (ispoly or israt or is_alg):
        return None

    # 使用给定的域 `domain` 创建关于变量 `x` 的多项式环
    R = domain.old_poly_ring(x)
    # 获取微分算子
    _, Dx = DifferentialOperators(R, 'Dx')

    # 如果函数是常数函数
    if not func.has(x):
        # 返回具有指定初值的全纯函数对象
        return HolonomicFunction(Dx, x, 0, [func])

    if ispoly:
        # 如果函数是多项式，则求解满足的微分方程
        sol = func * Dx - func.diff(x)
        # 对解进行标准化处理，确保多项式系数正确
        sol = _normalize(sol.listofpoly, sol.parent, negative=False)
        # 检查初始点是否是奇异点
        is_singular = sol.is_singular(x0)

        # 尝试计算奇异点的条件
        if y0 is None and x0 == 0 and is_singular:
            # 将函数转换为 SymPy 格式，并转换为列表形式
            rep = R.from_sympy(func).to_list()
            for i, j in enumerate(reversed(rep)):
                if j == 0:
                    continue
                coeff = list(reversed(rep))[i:]
                indicial = i
                break
            for i, j in enumerate(coeff):
                if isinstance(j, (PolyElement, FracElement)):
                    # 将多项式元素或分数元素转换为表达式
                    coeff[i] = j.as_expr()
            # 构建初值字典
            y0 = {indicial: S(coeff)}

    elif israt:
        # 如果函数是有理函数，则求解满足的微分方程
        p, q = func.as_numer_denom()
        sol = p * q * Dx + p * q.diff(x) - q * p.diff(x)
        sol = _normalize(sol.listofpoly, sol.parent, negative=False)

    elif is_alg:
        # 如果函数是代数函数，则求解满足的微分方程
        sol = n * (x / m) * Dx - 1
        sol = HolonomicFunction(sol, x).composition(basepoly).annihilator
        is_singular = sol.is_singular(x0)

        # 尝试计算奇异点的条件
        if y0 is None and x0 == 0 and is_singular and \
            (lenics is None or lenics <= 1):
            # 将基础多项式转换为 SymPy 格式，并转换为列表形式
            rep = R.from_sympy(basepoly).to_list()
            for i, j in enumerate(reversed(rep)):
                if j == 0:
                    continue
                if isinstance(j, (PolyElement, FracElement)):
                    j = j.as_expr()

                coeff = S(j)**ratexp
                indicial = S(i) * ratexp
                break
            if isinstance(coeff, (PolyElement, FracElement)):
                # 将多项式元素或分数元素转换为表达式
                coeff = coeff.as_expr()
            # 构建初值字典
            y0 = {indicial: S([coeff])}

    if y0 or not initcond:
        # 如果存在初值字典或者不需要初始条件，则返回全纯函数对象
        return HolonomicFunction(sol, x, x0, y0)

    if not lenics:
        # 如果没有指定初始条件长度，则使用解的阶数
        lenics = sol.order

    if sol.is_singular(x0):
        # 如果解在 x0 处是奇异的
        r = HolonomicFunction(sol, x, x0)._indicial()
        l = list(r)
        if len(r) == 1 and r[l[0]] == S.One:
            r = l[0]
            # 计算奇异初始条件
            g = func / (x - x0)**r
            singular_ics = _find_conditions(g, x, x0, lenics)
            singular_ics = [j / factorial(i) for i, j in enumerate(singular_ics)]
            y0 = {r: singular_ics}
            return HolonomicFunction(sol, x, x0, y0)

    # 计算通常的初始条件
    y0 = _find_conditions(func, x, x0, lenics)
    while not y0:
        x0 += 1
        y0 = _find_conditions(func, x, x0, lenics)

    return HolonomicFunction(sol, x, x0, y0)
# 转换梅耶尔函数积分表达式为泛函方程
def _convert_meijerint(func, x, initcond=True, domain=QQ):
    # 对输入的梅耶尔函数进行重写处理，得到参数
    args = meijerint._rewrite1(func, x)

    # 如果参数存在，则解构参数
    if args:
        fac, po, g, _ = args
    else:
        return None

    # 用于存储梅耶尔函数求和的列表
    fac_list = [fac * i[0] for i in g]
    # 获取参数的基本表达式
    t = po.as_base_exp()
    # 如果基本表达式的第一个元素是 x，则取第二个元素，否则取零
    s = t[1] if t[0] == x else S.Zero
    # 构建梅耶尔函数的指数列表
    po_list = [s + i[1] for i in g]
    # 获取梅耶尔函数列表
    G_list = [i[2] for i in g]

    # 定义内部函数，用于计算带移位的梅耶尔函数表示
    def _shift(func, s):
        z = func.args[-1]
        # 如果 z 中包含 I，则替换为 exp_polar 中的 exp
        if z.has(I):
            z = z.subs(exp_polar, exp)

        # 收集 z 中关于 x 的项
        d = z.collect(x, evaluate=False)
        b = list(d)[0]
        a = d[b]

        # 获取 b 的基本表达式
        t = b.as_base_exp()
        # 如果基本表达式的第一个元素是 x，则取第二个元素，否则取零
        b = t[1] if t[0] == x else S.Zero
        # 计算移位系数
        r = s / b
        # 更新梅耶尔函数参数
        an = (i + r for i in func.args[0][0])
        ap = (i + r for i in func.args[0][1])
        bm = (i + r for i in func.args[1][0])
        bq = (i + r for i in func.args[1][1])

        return a**-r, meijerg((an, ap), (bm, bq), z)

    # 计算初始解的系数和梅耶尔函数表示
    coeff, m = _shift(G_list[0], po_list[0])
    sol = fac_list[0] * coeff * from_meijerg(m, initcond=initcond, domain=domain)

    # 循环添加所有梅耶尔函数，并转换为保形函数
    for i in range(1, len(G_list)):
        coeff, m = _shift(G_list[i], po_list[i])
        sol += fac_list[i] * coeff * from_meijerg(m, initcond=initcond, domain=domain)

    return sol


# 创建查找表格的函数，使用给定的定义域
def _create_table(table, domain=QQ):
    """
    Creates the look-up table. For a similar implementation
    see meijerint._create_lookup_table.
    """

    def add(formula, annihilator, arg, x0=0, y0=()):
        """
        Adds a formula in the dictionary
        """
        table.setdefault(_mytype(formula, x_1), []).append((formula,
            HolonomicFunction(annihilator, arg, x0, y0)))

    # 在给定的环境中创建环和微分算子
    R = domain.old_poly_ring(x_1)
    _, Dx = DifferentialOperators(R, 'Dx')

    # 添加基本函数到查找表中
    add(sin(x_1), Dx**2 + 1, x_1, 0, [0, 1])
    add(cos(x_1), Dx**2 + 1, x_1, 0, [1, 0])
    add(exp(x_1), Dx - 1, x_1, 0, 1)
    add(log(x_1), Dx + x_1*Dx**2, x_1, 1, [0, 1])

    add(erf(x_1), 2*x_1*Dx + Dx**2, x_1, 0, [0, 2/sqrt(pi)])
    add(erfc(x_1), 2*x_1*Dx + Dx**2, x_1, 0, [1, -2/sqrt(pi)])
    add(erfi(x_1), -2*x_1*Dx + Dx**2, x_1, 0, [0, 2/sqrt(pi)])

    add(sinh(x_1), Dx**2 - 1, x_1, 0, [0, 1])
    add(cosh(x_1), Dx**2 - 1, x_1, 0, [1, 0])

    add(sinc(x_1), x_1 + 2*Dx + x_1*Dx**2, x_1)

    add(Si(x_1), x_1*Dx + 2*Dx**2 + x_1*Dx**3, x_1)
    add(Ci(x_1), x_1*Dx + 2*Dx**2 + x_1*Dx**3, x_1)

    add(Shi(x_1), -x_1*Dx + 2*Dx**2 + x_1*Dx**3, x_1)


# 查找条件，以确定给定函数的初始条件
def _find_conditions(func, x, x0, order):
    y0 = []
    for i in range(order):
        # 在给定点 x0 处计算函数值
        val = func.subs(x, x0)
        # 如果值为 NaN，则取极限
        if isinstance(val, NaN):
            val = limit(func, x, x0)
        # 如果值无限大或者为 NaN，则返回空
        if val.is_finite is False or isinstance(val, NaN):
            return None
        # 添加计算得到的初始条件
        y0.append(val)
        func = func.diff(x)
    return y0
```