# `D:\src\scipysrc\sympy\sympy\integrals\manualintegrate.py`

```
"""Integration method that emulates by-hand techniques.

This module also provides functionality to get the steps used to evaluate a
particular integral, in the ``integral_steps`` function. This will return
nested ``Rule`` s representing the integration rules used.

Each ``Rule`` class represents a (maybe parametrized) integration rule, e.g.
``SinRule`` for integrating ``sin(x)`` and ``ReciprocalSqrtQuadraticRule``
for integrating ``1/sqrt(a+b*x+c*x**2)``. The ``eval`` method returns the
integration result.

The ``manualintegrate`` function computes the integral by calling ``eval``
on the rule returned by ``integral_steps``.

The integrator can be extended with new heuristics and evaluation
techniques. To do so, extend the ``Rule`` class, implement ``eval`` method,
then write a function that accepts an ``IntegralInfo`` object and returns
either a ``Rule`` instance or ``None``. If the new technique requires a new
match, add the key and call to the antiderivative function to integral_steps.
To enable simple substitutions, add the match to find_substitutions.

"""

from __future__ import annotations  # 允许在类型提示中使用类自身
from typing import NamedTuple, Type, Callable, Sequence  # 引入类型提示需要的模块
from abc import ABC, abstractmethod  # 引入抽象基类相关的模块
from dataclasses import dataclass  # 引入数据类相关的模块
from collections import defaultdict  # 引入默认字典相关的模块
from collections.abc import Mapping  # 引入映射类型相关的模块

from sympy.core.add import Add  # 导入 SymPy 中的加法运算
from sympy.core.cache import cacheit  # 导入 SymPy 中的缓存功能
from sympy.core.containers import Dict  # 导入 SymPy 中的字典容器
from sympy.core.expr import Expr  # 导入 SymPy 中的表达式类型
from sympy.core.function import Derivative  # 导入 SymPy 中的导数函数
from sympy.core.logic import fuzzy_not  # 导入 SymPy 中的模糊逻辑函数
from sympy.core.mul import Mul  # 导入 SymPy 中的乘法运算
from sympy.core.numbers import Integer, Number, E  # 导入 SymPy 中的整数和常数 e
from sympy.core.power import Pow  # 导入 SymPy 中的幂运算
from sympy.core.relational import Eq, Ne, Boolean  # 导入 SymPy 中的关系运算
from sympy.core.singleton import S  # 导入 SymPy 中的单例对象
from sympy.core.symbol import Dummy, Symbol, Wild  # 导入 SymPy 中的符号和通配符
from sympy.functions.elementary.complexes import Abs  # 导入 SymPy 中的绝对值函数
from sympy.functions.elementary.exponential import exp, log  # 导入 SymPy 中的指数和对数函数
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction, csch,
    cosh, coth, sech, sinh, tanh, asinh)  # 导入 SymPy 中的双曲函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 SymPy 中的平方根函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入 SymPy 中的分段函数
from sympy.functions.elementary.trigonometric import (TrigonometricFunction,
    cos, sin, tan, cot, csc, sec, acos, asin, atan, acot, acsc, asec)  # 导入 SymPy 中的三角函数
from sympy.functions.special.delta_functions import Heaviside, DiracDelta  # 导入 SymPy 中的 delta 函数
from sympy.functions.special.error_functions import (erf, erfi, fresnelc,
    fresnels, Ci, Chi, Si, Shi, Ei, li)  # 导入 SymPy 中的特殊误差函数
from sympy.functions.special.gamma_functions import uppergamma  # 导入 SymPy 中的 gamma 函数
from sympy.functions.special.elliptic_integrals import elliptic_e, elliptic_f  # 导入 SymPy 中的椭圆积分函数
from sympy.functions.special.polynomials import (chebyshevt, chebyshevu,
    legendre, hermite, laguerre, assoc_laguerre, gegenbauer, jacobi,
    OrthogonalPolynomial)  # 导入 SymPy 中的多项式函数
from sympy.functions.special.zeta_functions import polylog  # 导入 SymPy 中的 polylog 函数
from .integrals import Integral  # 导入本地的 integrals 模块中的 Integral 类
from sympy.logic.boolalg import And  # 导入 SymPy 中的布尔代数操作
from sympy.ntheory.factor_ import primefactors  # 导入 SymPy 中的质因数分解函数
from sympy.polys.polytools import degree, lcm_list, gcd_list, Poly  # 导入多项式操作函数和类
from sympy.simplify.radsimp import fraction  # 导入有理化简函数
from sympy.simplify.simplify import simplify  # 导入表达式简化函数
from sympy.solvers.solvers import solve  # 导入求解方程函数
from sympy.strategies.core import switch, do_one, null_safe, condition  # 导入策略函数
from sympy.utilities.iterables import iterable  # 导入可迭代工具函数
from sympy.utilities.misc import debug  # 导入调试函数

@dataclass
class Rule(ABC):
    integrand: Expr  # 规则的被积函数表达式
    variable: Symbol  # 积分变量

    @abstractmethod
    def eval(self) -> Expr:
        pass  # 抽象方法，规则求值

    @abstractmethod
    def contains_dont_know(self) -> bool:
        pass  # 抽象方法，判断规则是否包含未知项

@dataclass
class AtomicRule(Rule, ABC):
    """A simple rule that does not depend on other rules"""
    def contains_dont_know(self) -> bool:
        return False  # 简单规则不包含未知项

@dataclass
class ConstantRule(AtomicRule):
    """integrate(a, x)  ->  a*x"""
    def eval(self) -> Expr:
        return self.integrand * self.variable  # 常数积分规则的求值

@dataclass
class ConstantTimesRule(Rule):
    """integrate(a*f(x), x)  ->  a*integrate(f(x), x)"""
    constant: Expr  # 常数
    other: Expr  # 函数f(x)
    substep: Rule  # 子规则

    def eval(self) -> Expr:
        return self.constant * self.substep.eval()  # 常数乘以子规则的求值

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()  # 判断子规则是否包含未知项

@dataclass
class PowerRule(AtomicRule):
    """integrate(x**a, x)"""
    base: Expr  # 底数x
    exp: Expr  # 指数a

    def eval(self) -> Expr:
        return Piecewise(
            ((self.base**(self.exp + 1))/(self.exp + 1), Ne(self.exp, -1)),  # 幂函数积分规则的求值
            (log(self.base), True),
        )

@dataclass
class NestedPowRule(AtomicRule):
    """integrate((x**a)**b, x)"""
    base: Expr  # 底数x
    exp: Expr  # 指数a

    def eval(self) -> Expr:
        m = self.base * self.integrand
        return Piecewise((m / (self.exp + 1), Ne(self.exp, -1)),  # 嵌套幂函数积分规则的求值
                         (m * log(self.base), True))

@dataclass
class AddRule(Rule):
    """integrate(f(x) + g(x), x) -> integrate(f(x), x) + integrate(g(x), x)"""
    substeps: list[Rule]  # 子规则列表

    def eval(self) -> Expr:
        return Add(*(substep.eval() for substep in self.substeps))  # 加法规则的求值

    def contains_dont_know(self) -> bool:
        return any(substep.contains_dont_know() for substep in self.substeps)  # 判断子规则中是否有未知项

@dataclass
class URule(Rule):
    """integrate(f(g(x))*g'(x), x) -> integrate(f(u), u), u = g(x)"""
    u_var: Symbol  # 替换变量u
    u_func: Expr  # 函数g(x)
    substep: Rule  # 子规则

    def eval(self) -> Expr:
        result = self.substep.eval()
        if self.u_func.is_Pow:
            base, exp_ = self.u_func.as_base_exp()
            if exp_ == -1:
                # avoid needless -log(1/x) from substitution
                result = result.subs(log(self.u_var), -log(base))  # 对结果进行变量替换处理
        return result.subs(self.u_var, self.u_func)  # 对结果进行最终的变量替换

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()  # 判断子规则是否包含未知项

@dataclass
class PartsRule(Rule):
    """integrate(u(x)*v'(x), x) -> u(x)*v(x) - integrate(u'(x)*v(x), x)"""
    u: Symbol  # 函数u(x)
    dv: Expr  # 函数v'(x)
    v_step: Rule  # 子规则
    second_step: Rule | None  # 第二步规则（或为空，用于循环部分规则的子规则）
    # 计算表达式的值并返回结果表达式
    def eval(self) -> Expr:
        # 断言第二步骤不为空
        assert self.second_step is not None
        # 计算 v_step 表达式的值
        v = self.v_step.eval()
        # 返回计算结果表达式 u * v - second_step 的值
        return self.u * v - self.second_step.eval()
    
    # 检查表达式中是否包含未知的部分
    def contains_dont_know(self) -> bool:
        # 检查 v_step 表达式是否包含未知的部分，或者第二步骤不为空且包含未知的部分
        return self.v_step.contains_dont_know() or (
            self.second_step is not None and self.second_step.contains_dont_know())
@dataclass
class CyclicPartsRule(Rule):
    """Apply PartsRule multiple times to integrate exp(x)*sin(x)"""
    # 定义循环部分规则，用于对 exp(x)*sin(x) 进行多次积分
    parts_rules: list[PartsRule]  # 包含多个 PartsRule 对象的列表
    coefficient: Expr  # 表达式系数

    def eval(self) -> Expr:
        # 计算积分结果
        result = []
        sign = 1
        for rule in self.parts_rules:
            # 依次计算每个部分规则的积分结果，并按照正负号交替累加
            result.append(sign * rule.u * rule.v_step.eval())
            sign *= -1
        return Add(*result) / (1 - self.coefficient)  # 返回最终结果除以系数

    def contains_dont_know(self) -> bool:
        # 检查是否存在不明确情况，即其中是否有部分规则包含不明确内容
        return any(substep.contains_dont_know() for substep in self.parts_rules)


@dataclass
class TrigRule(AtomicRule, ABC):
    pass  # 三角函数规则的基类


@dataclass
class SinRule(TrigRule):
    """integrate(sin(x), x) -> -cos(x)"""
    def eval(self) -> Expr:
        # 计算 sin(x) 的积分结果
        return -cos(self.variable)


@dataclass
class CosRule(TrigRule):
    """integrate(cos(x), x) -> sin(x)"""
    def eval(self) -> Expr:
        # 计算 cos(x) 的积分结果
        return sin(self.variable)


@dataclass
class SecTanRule(TrigRule):
    """integrate(sec(x)*tan(x), x) -> sec(x)"""
    def eval(self) -> Expr:
        # 计算 sec(x)*tan(x) 的积分结果
        return sec(self.variable)


@dataclass
class CscCotRule(TrigRule):
    """integrate(csc(x)*cot(x), x) -> -csc(x)"""
    def eval(self) -> Expr:
        # 计算 csc(x)*cot(x) 的积分结果
        return -csc(self.variable)


@dataclass
class Sec2Rule(TrigRule):
    """integrate(sec(x)**2, x) -> tan(x)"""
    def eval(self) -> Expr:
        # 计算 sec(x)**2 的积分结果
        return tan(self.variable)


@dataclass
class Csc2Rule(TrigRule):
    """integrate(csc(x)**2, x) -> -cot(x)"""
    def eval(self) -> Expr:
        # 计算 csc(x)**2 的积分结果
        return -cot(self.variable)


@dataclass
class HyperbolicRule(AtomicRule, ABC):
    pass  # 双曲函数规则的基类


@dataclass
class SinhRule(HyperbolicRule):
    """integrate(sinh(x), x) -> cosh(x)"""
    def eval(self) -> Expr:
        # 计算 sinh(x) 的积分结果
        return cosh(self.variable)


@dataclass
class CoshRule(HyperbolicRule):
    """integrate(cosh(x), x) -> sinh(x)"""
    def eval(self):
        # 计算 cosh(x) 的积分结果
        return sinh(self.variable)


@dataclass
class ExpRule(AtomicRule):
    """integrate(a**x, x) -> a**x/ln(a)"""
    base: Expr
    exp: Expr

    def eval(self) -> Expr:
        # 计算 a**x 的积分结果
        return self.integrand / log(self.base)


@dataclass
class ReciprocalRule(AtomicRule):
    """integrate(1/x, x) -> ln(x)"""
    base: Expr

    def eval(self) -> Expr:
        # 计算 1/x 的积分结果
        return log(self.base)


@dataclass
class ArcsinRule(AtomicRule):
    """integrate(1/sqrt(1-x**2), x) -> asin(x)"""
    def eval(self) -> Expr:
        # 计算 1/sqrt(1-x**2) 的积分结果
        return asin(self.variable)


@dataclass
class ArcsinhRule(AtomicRule):
    """integrate(1/sqrt(1+x**2), x) -> asin(x)"""
    def eval(self) -> Expr:
        # 计算 1/sqrt(1+x**2) 的积分结果
        return asinh(self.variable)


@dataclass
class ReciprocalSqrtQuadraticRule(AtomicRule):
    """integrate(1/sqrt(a+b*x+c*x**2), x) -> log(2*sqrt(c)*sqrt(a+b*x+c*x**2)+b+2*c*x)/sqrt(c)"""
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        # 计算 1/sqrt(a+b*x+c*x**2) 的积分结果
        a, b, c, x = self.a, self.b, self.c, self.variable
        return log(2*sqrt(c)*sqrt(a+b*x+c*x**2)+b+2*c*x)/sqrt(c)


@dataclass
class SqrtQuadraticDenomRule(AtomicRule):
    """integrate(poly(x)/sqrt(a+b*x+c*x**2), x)"""
    a: Expr
    b: Expr
    c: Expr
    coeffs: list[Expr]
    def eval(self) -> Expr:
        # 拷贝实例属性到局部变量
        a, b, c, coeffs, x = self.a, self.b, self.c, self.coeffs.copy(), self.variable
        # 使用递归积分poly/sqrt(a+b*x+c*x**2)。
        # coeffs 是多项式的系数。
        # 设 I_n = x**n/sqrt(a+b*x+c*x**2)，则有
        # I_n = A * x**(n-1)*sqrt(a+b*x+c*x**2) - B * I_{n-1} - C * I_{n-2}
        # 其中 A = 1/(n*c), B = (2*n-1)*b/(2*n*c), C = (n-1)*a/(n*c)
        # 详见 https://github.com/sympy/sympy/pull/23608 进行证明。
        result_coeffs = []  # 初始化结果系数列表
        coeffs = coeffs.copy()  # 复制系数列表以防止修改原始值
        for i in range(len(coeffs)-2):
            n = len(coeffs)-1-i
            coeff = coeffs[i]/(c*n)
            result_coeffs.append(coeff)
            coeffs[i+1] -= (2*n-1)*b/2*coeff
            coeffs[i+2] -= (n-1)*a*coeff
        d, e = coeffs[-1], coeffs[-2]
        s = sqrt(a+b*x+c*x**2)  # 计算 sqrt(a+b*x+c*x**2)
        constant = d-b*e/(2*c)
        if constant == 0:
            I0 = 0  # 如果常数项为零，则积分值为零
        else:
            step = inverse_trig_rule(IntegralInfo(1/s, x), degenerate=False)
            I0 = constant*step.eval()  # 使用反三角函数规则计算积分值
        # 返回表达式结果
        return Add(*(result_coeffs[i]*x**(len(coeffs)-2-i)
                     for i in range(len(result_coeffs))), e/c)*s + I0
@dataclass
class SqrtQuadraticRule(AtomicRule):
    """A rule for integrating sqrt(a+b*x+c*x**2) with respect to x."""
    a: Expr  # Coefficient of x^0
    b: Expr  # Coefficient of x^1
    c: Expr  # Coefficient of x^2

    def eval(self) -> Expr:
        # Invoke specific rule for integrating sqrt(a+b*x+c*x**2)
        step = sqrt_quadratic_rule(IntegralInfo(self.integrand, self.variable), degenerate=False)
        return step.eval()


@dataclass
class AlternativeRule(Rule):
    """A rule that provides multiple methods for integration."""
    alternatives: list[Rule]  # List of alternative integration rules

    def eval(self) -> Expr:
        # Evaluate using the first alternative
        return self.alternatives[0].eval()

    def contains_dont_know(self) -> bool:
        # Check if any alternative contains 'don't know' scenario
        return any(substep.contains_dont_know() for substep in self.alternatives)


@dataclass
class DontKnowRule(Rule):
    """A rule that leaves the integral unchanged."""
    def eval(self) -> Expr:
        # Return the integral as it is
        return Integral(self.integrand, self.variable)

    def contains_dont_know(self) -> bool:
        # Always indicates that it contains 'don't know'
        return True


@dataclass
class DerivativeRule(AtomicRule):
    """A rule for integrating the derivative f'(x) with respect to x."""
    def eval(self) -> Expr:
        assert isinstance(self.integrand, Derivative)
        variable_count = list(self.integrand.variable_count)
        for i, (var, count) in enumerate(variable_count):
            if var == self.variable:
                variable_count[i] = (var, count - 1)
                break
        return Derivative(self.integrand.expr, *variable_count)


@dataclass
class RewriteRule(Rule):
    """A rule that rewrites the integrand to a simpler form."""
    rewritten: Expr  # The rewritten form of the integrand
    substep: Rule    # The substep rule used for evaluation

    def eval(self) -> Expr:
        # Evaluate using the substep rule
        return self.substep.eval()

    def contains_dont_know(self) -> bool:
        # Check if the substep contains 'don't know'
        return self.substep.contains_dont_know()


@dataclass
class CompleteSquareRule(RewriteRule):
    """A rule that completes the square for integrands of the form a+b*x+c*x**2."""
    pass


@dataclass
class PiecewiseRule(Rule):
    """A rule for integrating piecewise functions."""
    subfunctions: Sequence[tuple[Rule, bool | Boolean]]  # List of (substep, condition) pairs

    def eval(self) -> Expr:
        # Evaluate using piecewise function based on conditions
        return Piecewise(*[(substep.eval(), cond)
                           for substep, cond in self.subfunctions])

    def contains_dont_know(self) -> bool:
        # Check if any substep contains 'don't know'
        return any(substep.contains_dont_know() for substep, _ in self.subfunctions)


@dataclass
class HeavisideRule(Rule):
    """A rule for integrating functions involving the Heaviside function."""
    harg: Expr    # Argument of the Heaviside function
    ibnd: Expr    # Integration bound
    substep: Rule  # Substep rule for evaluation

    def eval(self) -> Expr:
        # Adjust the result for integration involving Heaviside function
        result = self.substep.eval()
        return Heaviside(self.harg) * (result - result.subs(self.variable, self.ibnd))

    def contains_dont_know(self) -> bool:
        # Check if the substep contains 'don't know'
        return self.substep.contains_dont_know()


@dataclass
class DiracDeltaRule(AtomicRule):
    """A rule for integrating functions involving the Dirac delta function."""
    n: Expr   # Order of the Dirac delta function
    a: Expr   # Constant term
    b: Expr   # Coefficient of x

    def eval(self) -> Expr:
        n, a, b, x = self.n, self.a, self.b, self.variable
        if n == 0:
            return Heaviside(a+b*x)/b
        return DiracDelta(a+b*x, n-1)/b


@dataclass
class TrigSubstitutionRule(Rule):
    """A rule for integrating using trigonometric substitutions."""
    theta: Expr   # Trigonometric substitution variable
    func: Expr    # Trigonometric function to substitute
    # 用于存储重写后的表达式
    rewritten: Expr
    # 子步骤，类型为 Rule
    substep: Rule
    # 表示是否有限制，可以是布尔值或者 Boolean 类型

    # 计算方法的评估函数，返回一个表达式对象
    def eval(self) -> Expr:
        # 从对象的属性中获取 theta, func, x
        theta, func, x = self.theta, self.func, self.variable
        # 将函数 func 中的 sec(theta) 替换为 1/cos(theta)
        func = func.subs(sec(theta), 1/cos(theta))
        # 将函数 func 中的 csc(theta) 替换为 1/sin(theta)
        func = func.subs(csc(theta), 1/sin(theta))
        # 将函数 func 中的 cot(theta) 替换为 1/tan(theta)
        func = func.subs(cot(theta), 1/tan(theta))

        # 找到 func 中的三角函数表达式，并确保只有一个
        trig_function = list(func.find(TrigonometricFunction))
        assert len(trig_function) == 1
        trig_function = trig_function[0]
        # 解方程 x - func = 0，得到与 trig_function 相关的关系式
        relation = solve(x - func, trig_function)
        assert len(relation) == 1
        # 将关系式化为分子和分母
        numer, denom = fraction(relation[0])

        # 根据 trig_function 的类型进行不同的处理
        if isinstance(trig_function, sin):
            opposite = numer
            hypotenuse = denom
            adjacent = sqrt(denom**2 - numer**2)
            inverse = asin(relation[0])
        elif isinstance(trig_function, cos):
            adjacent = numer
            hypotenuse = denom
            opposite = sqrt(denom**2 - numer**2)
            inverse = acos(relation[0])
        else:  # tan
            opposite = numer
            adjacent = denom
            hypotenuse = sqrt(denom**2 + numer**2)
            inverse = atan(relation[0])

        # 替换 theta 和三角函数，形成一个替换列表
        substitution = [
            (sin(theta), opposite/hypotenuse),
            (cos(theta), adjacent/hypotenuse),
            (tan(theta), opposite/adjacent),
            (theta, inverse)
        ]
        # 返回 Piecewise 对象，使用 substep 的评估结果进行替换和简化，根据 restriction 条件选择
        return Piecewise(
            (self.substep.eval().subs(substitution).trigsimp(), self.restriction)
        )

    # 检查 substep 是否包含 "dont_know"，返回布尔值
    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()
@dataclass
class ArctanRule(AtomicRule):
    """Represents a rule for integrating a/(b*x**2+c) with respect to x."""
    a: Expr  # Coefficient 'a' in the expression
    b: Expr  # Coefficient 'b' in the expression
    c: Expr  # Coefficient 'c' in the expression

    def eval(self) -> Expr:
        """Evaluate the integral of a/(b*x**2+c) with respect to x."""
        a, b, c, x = self.a, self.b, self.c, self.variable
        # Compute the result using the arctangent rule formula
        return a/b / sqrt(c/b) * atan(x/sqrt(c/b))


@dataclass
class OrthogonalPolyRule(AtomicRule, ABC):
    """Abstract base class for rules involving orthogonal polynomials."""
    n: Expr  # Polynomial degree


@dataclass
class JacobiRule(OrthogonalPolyRule):
    """Represents a Jacobi polynomial rule."""
    a: Expr  # Parameter 'a' in Jacobi polynomials
    b: Expr  # Parameter 'b' in Jacobi polynomials

    def eval(self) -> Expr:
        """Evaluate the Jacobi polynomial of degree 'n'."""
        n, a, b, x = self.n, self.a, self.b, self.variable
        # Return the evaluated Jacobi polynomial using Piecewise conditions
        return Piecewise(
            (2*jacobi(n + 1, a - 1, b - 1, x)/(n + a + b), Ne(n + a + b, 0)),
            (x, Eq(n, 0)),
            ((a + b + 2)*x**2/4 + (a - b)*x/2, Eq(n, 1)))


@dataclass
class GegenbauerRule(OrthogonalPolyRule):
    """Represents a Gegenbauer polynomial rule."""
    a: Expr  # Parameter 'a' in Gegenbauer polynomials

    def eval(self) -> Expr:
        """Evaluate the Gegenbauer polynomial of degree 'n'."""
        n, a, x = self.n, self.a, self.variable
        # Return the evaluated Gegenbauer polynomial using Piecewise conditions
        return Piecewise(
            (gegenbauer(n + 1, a - 1, x)/(2*(a - 1)), Ne(a, 1)),
            (chebyshevt(n + 1, x)/(n + 1), Ne(n, -1)),
            (S.Zero, True))


@dataclass
class ChebyshevTRule(OrthogonalPolyRule):
    """Represents a Chebyshev T polynomial rule."""
    def eval(self) -> Expr:
        """Evaluate the Chebyshev T polynomial of degree 'n'."""
        n, x = self.n, self.variable
        # Return the evaluated Chebyshev T polynomial using Piecewise conditions
        return Piecewise(
            ((chebyshevt(n + 1, x)/(n + 1) -
              chebyshevt(n - 1, x)/(n - 1))/2, Ne(Abs(n), 1)),
            (x**2/2, True))


@dataclass
class ChebyshevURule(OrthogonalPolyRule):
    """Represents a Chebyshev U polynomial rule."""
    def eval(self) -> Expr:
        """Evaluate the Chebyshev U polynomial of degree 'n'."""
        n, x = self.n, self.variable
        # Return the evaluated Chebyshev U polynomial using Piecewise conditions
        return Piecewise(
            (chebyshevt(n + 1, x)/(n + 1), Ne(n, -1)),
            (S.Zero, True))


@dataclass
class LegendreRule(OrthogonalPolyRule):
    """Represents a Legendre polynomial rule."""
    def eval(self) -> Expr:
        """Evaluate the Legendre polynomial of degree 'n'."""
        n, x = self.n, self.variable
        # Return the evaluated Legendre polynomial
        return (legendre(n + 1, x) - legendre(n - 1, x))/(2*n + 1)


@dataclass
class HermiteRule(OrthogonalPolyRule):
    """Represents a Hermite polynomial rule."""
    def eval(self) -> Expr:
        """Evaluate the Hermite polynomial of degree 'n'."""
        n, x = self.n, self.variable
        # Return the evaluated Hermite polynomial
        return hermite(n + 1, x)/(2*(n + 1))


@dataclass
class LaguerreRule(OrthogonalPolyRule):
    """Represents a Laguerre polynomial rule."""
    def eval(self) -> Expr:
        """Evaluate the Laguerre polynomial of degree 'n'."""
        n, x = self.n, self.variable
        # Return the evaluated Laguerre polynomial
        return laguerre(n, x) - laguerre(n + 1, x)


@dataclass
class AssocLaguerreRule(OrthogonalPolyRule):
    """Represents an associated Laguerre polynomial rule."""
    a: Expr  # Parameter 'a' in associated Laguerre polynomials

    def eval(self) -> Expr:
        """Evaluate the associated Laguerre polynomial of degree 'n'."""
        return -assoc_laguerre(self.n + 1, self.a - 1, self.variable)


@dataclass
class IRule(AtomicRule, ABC):
    """Abstract base class for rules involving I functions."""
    a: Expr
    b: Expr


@dataclass
class CiRule(IRule):
    """Represents a Ci function rule."""
    def eval(self) -> Expr:
        """Evaluate the Ci function."""
        a, b, x = self.a, self.b, self.variable
        # Return the evaluated Ci function using cosine and sine integrals
        return cos(b)*Ci(a*x) - sin(b)*Si(a*x)


@dataclass
class ChiRule(IRule):
    """Represents a Chi function rule."""
    def eval(self) -> Expr:
        """Evaluate the Chi function."""
        a, b, x = self.a, self.b, self.variable
        # Return the evaluated Chi function using hyperbolic cosine and sine integrals
        return cosh(b)*Chi(a*x) + sinh(b)*Shi(a*x)


@dataclass
class EiRule(IRule):
    """Represents an Ei function rule."""
    def eval(self) -> Expr:
        """Evaluate the Ei function."""
        a, b, x = self.a, self.b, self.variable
        # Return the evaluated Ei function using exponential integral
        return exp(b)*Ei(a*x)


@dataclass
class SiRule(IRule):
    """Represents a Si function rule."""
    def eval(self) -> Expr:
        """Evaluate the Si function."""
        a, b, x = self.a, self.b, self.variable
        # Return the evaluated Si function using sine and cosine integrals
        return sin(b)*Ci(a*x) + cos(b)*Si(a*x)


@dataclass
class ShiRule(IRule):
    """Represents a Shi function rule."""
    # 定义一个方法 eval，返回类型为 Expr（表达式）
    def eval(self) -> Expr:
        # 从当前对象中获取属性 a, b, variable，并分别赋值给变量 a, b, x
        a, b, x = self.a, self.b, self.variable
        # 返回表达式 sinh(b)*Chi(a*x) + cosh(b)*Shi(a*x)，其中 sinh、Chi、cosh 和 Shi 是数学函数或对象
        return sinh(b)*Chi(a*x) + cosh(b)*Shi(a*x)
@dataclass
class LiRule(IRule):
    # Li 求导规则的实现，继承自 IRule 接口
    def eval(self) -> Expr:
        # 解构赋值获取参数 a, b, x
        a, b, x = self.a, self.b, self.variable
        # 返回 Li 函数的导数表达式
        return li(a*x + b)/a


@dataclass
class ErfRule(AtomicRule):
    # 带误差函数 Erf 的求导规则，继承自 AtomicRule
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        # 解构赋值获取参数 a, b, c, x
        a, b, c, x = self.a, self.b, self.c, self.variable
        # 如果 a 是扩展实数，返回分段函数
        if a.is_extended_real:
            return Piecewise(
                (sqrt(S.Pi)/sqrt(-a)/2 * exp(c - b**2/(4*a)) *
                    erf((-2*a*x - b)/(2*sqrt(-a))), a < 0),
                (sqrt(S.Pi)/sqrt(a)/2 * exp(c - b**2/(4*a)) *
                    erfi((2*a*x + b)/(2*sqrt(a))), True))
        # 否则返回一般情况下的表达式
        return sqrt(S.Pi)/sqrt(a)/2 * exp(c - b**2/(4*a)) * \
                erfi((2*a*x + b)/(2*sqrt(a)))


@dataclass
class FresnelCRule(AtomicRule):
    # Fresnel C 积分的求导规则，继承自 AtomicRule
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        # 解构赋值获取参数 a, b, c, x
        a, b, c, x = self.a, self.b, self.c, self.variable
        # 返回 Fresnel C 积分的导数表达式
        return sqrt(S.Pi)/sqrt(2*a) * (
            cos(b**2/(4*a) - c)*fresnelc((2*a*x + b)/sqrt(2*a*S.Pi)) +
            sin(b**2/(4*a) - c)*fresnels((2*a*x + b)/sqrt(2*a*S.Pi)))


@dataclass
class FresnelSRule(AtomicRule):
    # Fresnel S 积分的求导规则，继承自 AtomicRule
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        # 解构赋值获取参数 a, b, c, x
        a, b, c, x = self.a, self.b, self.c, self.variable
        # 返回 Fresnel S 积分的导数表达式
        return sqrt(S.Pi)/sqrt(2*a) * (
            cos(b**2/(4*a) - c)*fresnels((2*a*x + b)/sqrt(2*a*S.Pi)) -
            sin(b**2/(4*a) - c)*fresnelc((2*a*x + b)/sqrt(2*a*S.Pi)))


@dataclass
class PolylogRule(AtomicRule):
    # Polylog 函数的求导规则，继承自 AtomicRule
    a: Expr
    b: Expr

    def eval(self) -> Expr:
        # 返回 Polylog 函数的导数表达式
        return polylog(self.b + 1, self.a * self.variable)


@dataclass
class UpperGammaRule(AtomicRule):
    # Upper Gamma 函数的求导规则，继承自 AtomicRule
    a: Expr
    e: Expr

    def eval(self) -> Expr:
        # 解构赋值获取参数 a, e, x
        a, e, x = self.a, self.e, self.variable
        # 返回 Upper Gamma 函数的导数表达式
        return x**e * (-a*x)**(-e) * uppergamma(e + 1, -a*x)/a


@dataclass
class EllipticFRule(AtomicRule):
    # 椭圆积分 F 的求导规则，继承自 AtomicRule
    a: Expr
    d: Expr

    def eval(self) -> Expr:
        # 返回椭圆积分 F 的导数表达式
        return elliptic_f(self.variable, self.d/self.a)/sqrt(self.a)


@dataclass
class EllipticERule(AtomicRule):
    # 椭圆积分 E 的求导规则，继承自 AtomicRule
    a: Expr
    d: Expr

    def eval(self) -> Expr:
        # 返回椭圆积分 E 的导数表达式
        return elliptic_e(self.variable, self.d/self.a)*sqrt(self.a)


class IntegralInfo(NamedTuple):
    # 积分信息的命名元组，包含被积函数和符号
    integrand: Expr
    symbol: Symbol


def manual_diff(f, symbol):
    """Derivative of f in form expected by find_substitutions

    SymPy's derivatives for some trig functions (like cot) are not in a form
    that works well with finding substitutions; this replaces the
    derivatives for those particular forms with something that works better.

    """
    # 如果函数 f 有参数
    if f.args:
        # 取第一个参数
        arg = f.args[0]
        # 如果 f 是 tan 函数的实例
        if isinstance(f, tan):
            # 返回 tan 函数的导数乘以 sec(arg) 的平方
            return arg.diff(symbol) * sec(arg)**2
        # 如果 f 是 cot 函数的实例
        elif isinstance(f, cot):
            # 返回 cot 函数的导数乘以 csc(arg) 的平方的负数
            return -arg.diff(symbol) * csc(arg)**2
        # 如果 f 是 sec 函数的实例
        elif isinstance(f, sec):
            # 返回 sec 函数的导数乘以 sec(arg) 乘以 tan(arg)
            return arg.diff(symbol) * sec(arg) * tan(arg)
        # 如果 f 是 csc 函数的实例
        elif isinstance(f, csc):
            # 返回 csc 函数的导数乘以 csc(arg) 乘以 cot(arg)的负数
            return -arg.diff(symbol) * csc(arg) * cot(arg)
        # 如果 f 是 Add 类的实例
        elif isinstance(f, Add):
            # 返回所有参数的手动求导之和
            return sum(manual_diff(arg, symbol) for arg in f.args)
        # 如果 f 是 Mul 类的实例，并且参数有两个且第一个参数是数字
        elif isinstance(f, Mul):
            if len(f.args) == 2 and isinstance(f.args[0], Number):
                # 返回第一个参数乘以第二个参数的手动求导
                return f.args[0] * manual_diff(f.args[1], symbol)
    # 如果以上条件均不满足，则返回 f 关于 symbol 的导数
    return f.diff(symbol)
# 对表达式进行手动替换操作，支持可逆函数的特殊逻辑
def manual_subs(expr, *args):
    # 如果参数个数为1，将其视为一个序列
    if len(args) == 1:
        sequence = args[0]
        # 如果序列是字典或者映射，将其转换为键值对列表
        if isinstance(sequence, (Dict, Mapping)):
            sequence = sequence.items()
        # 如果不是可迭代对象，抛出数值错误
        elif not iterable(sequence):
            raise ValueError("Expected an iterable of (old, new) pairs")
    # 如果参数个数为2，将其作为一个元组放入序列中
    elif len(args) == 2:
        sequence = [args]
    # 如果参数个数不是1或2，抛出数值错误
    else:
        raise ValueError("subs accepts either 1 or 2 arguments")

    new_subs = []
    # 遍历序列中的每对旧值和新值
    for old, new in sequence:
        # 如果旧值是对数函数
        if isinstance(old, log):
            # 如果 log(x) = y，则 exp(a*log(x)) = exp(a*y)
            # 即 x**a = exp(a*y)。在 subs 转换这些非平凡幂次前替换它们为 `exp(y)**a`，
            # 但避免直接替换 x 本身，以免出现 `log(exp(y))`。
            x0 = old.args[0]
            expr = expr.replace(lambda x: x.is_Pow and x.base == x0,
                lambda x: exp(x.exp*new))
            new_subs.append((x0, exp(new)))

    return expr.subs(list(sequence) + new_subs)

# 基于 "Symbolic Integration: The Stormy Decade" 中 SIN 的方法

inverse_trig_functions = (atan, asin, acos, acot, acsc, asec)


def find_substitutions(integrand, symbol, u_var):
    results = []

    def test_subterm(u, u_diff):
        # 如果 u_diff 为零，返回 False
        if u_diff == 0:
            return False
        # 计算被替换后的表达式
        substituted = integrand / u_diff
        debug("substituted: {}, u: {}, u_var: {}".format(substituted, u, u_var))
        # 手动进行替换操作并取消表达式
        substituted = manual_subs(substituted, u, u_var).cancel()

        # 如果被替换后的表达式中包含自由符号 symbol，返回 False
        if substituted.has_free(symbol):
            return False
        # 避免增加有理函数的次数
        if integrand.is_rational_function(symbol) and substituted.is_rational_function(u_var):
            deg_before = max(degree(t, symbol) for t in integrand.as_numer_denom())
            deg_after = max(degree(t, u_var) for t in substituted.as_numer_denom())
            if deg_after > deg_before:
                return False
        # 返回被独立化的 u_var 的表达式部分，不作为 Add 形式返回
        return substituted.as_independent(u_var, as_Add=False)

    def exp_subterms(term: Expr):
        linear_coeffs = []
        terms = []
        n = Wild('n', properties=[lambda n: n.is_Integer])
        # 查找并处理表达式中的指数项
        for exp_ in term.find(exp):
            arg = exp_.args[0]
            # 如果 arg 中不包含符号 symbol，继续下一个迭代
            if symbol not in arg.free_symbols:
                continue
            # 尝试匹配 n*symbol 的形式
            match = arg.match(n*symbol)
            if match:
                linear_coeffs.append(match[n])
            else:
                terms.append(exp_)
        # 如果存在线性系数，加入 exp(gcd_list(linear_coeffs)*symbol)
        if linear_coeffs:
            terms.append(exp(gcd_list(linear_coeffs)*symbol))
        return terms
    # 定义一个函数，用于确定给定数学表达式中的可能子项
    def possible_subterms(term):
        # 如果 term 是以下类型之一：三角函数、双曲函数、逆三角函数、指数函数、对数函数、Heaviside函数，则返回其第一个参数作为可能的子项
        if isinstance(term, (TrigonometricFunction, HyperbolicFunction,
                             *inverse_trig_functions,
                             exp, log, Heaviside)):
            return [term.args[0]]
        # 如果 term 是切比雪夫多项式、勒让德多项式、厄米多项式、拉盖尔多项式中的一种，则返回其第二个参数作为可能的子项
        elif isinstance(term, (chebyshevt, chebyshevu,
                        legendre, hermite, laguerre)):
            return [term.args[1]]
        # 如果 term 是基尔霍夫多项式或关联勒让德多项式，则返回其第三个参数作为可能的子项
        elif isinstance(term, (gegenbauer, assoc_laguerre)):
            return [term.args[2]]
        # 如果 term 是雅可比多项式，则返回其第四个参数作为可能的子项
        elif isinstance(term, jacobi):
            return [term.args[3]]
        # 如果 term 是乘法表达式，则递归地找出所有子项
        elif isinstance(term, Mul):
            r = []
            for u in term.args:
                r.append(u)
                r.extend(possible_subterms(u))
            return r
        # 如果 term 是幂次表达式，则找出包含给定符号的子项
        elif isinstance(term, Pow):
            r = [arg for arg in term.args if arg.has(symbol)]
            # 如果指数是整数，则添加其底数的素因数幂作为子项
            if term.exp.is_Integer:
                r.extend([term.base**d for d in primefactors(term.exp)
                    if 1 < d < abs(term.args[1])])
                # 如果底数是加法表达式，则找出其中的幂次表达式作为子项
                if term.base.is_Add:
                    r.extend([t for t in possible_subterms(term.base)
                        if t.is_Pow])
            return r
        # 如果 term 是加法表达式，则递归地找出所有子项
        elif isinstance(term, Add):
            r = []
            for arg in term.args:
                r.append(arg)
                r.extend(possible_subterms(arg))
            return r
        # 如果 term 不属于上述任何类型，则返回空列表，表示没有找到可能的子项
        return []

    # 对可能的子项列表去重，并遍历每个子项
    for u in list(dict.fromkeys(possible_subterms(integrand) + exp_subterms(integrand))):
        # 如果子项等于符号本身，则跳过
        if u == symbol:
            continue
        # 对当前子项 u 求关于 symbol 的手动微分
        u_diff = manual_diff(u, symbol)
        # 测试子项 u 是否能替换成新的积分被积函数部分
        new_integrand = test_subterm(u, u_diff)
        # 如果测试结果不为 False
        if new_integrand is not False:
            # 解析出常数项和新的积分被积函数部分
            constant, new_integrand = new_integrand
            # 如果新的积分被积函数部分等于将 symbol 替换为 u_var 后的原积分被积函数部分，则跳过
            if new_integrand == integrand.subs(symbol, u_var):
                continue
            # 构建替换元组 (u, constant, new_integrand)，如果该替换元组不在结果列表中，则添加进去
            substitution = (u, constant, new_integrand)
            if substitution not in results:
                results.append(substitution)

    # 返回结果列表
    return results
# 定义一个函数 rewriter，用于根据特定条件重写被积函数
def rewriter(condition, rewrite):
    """Strategy that rewrites an integrand."""
    # 定义内部函数 _rewriter，接受一个积分元组 integral 作为参数
    def _rewriter(integral):
        # 解析积分元组，分别获取积分被积函数和符号
        integrand, symbol = integral
        # 输出调试信息，显示被积函数 integrand 被 rewrite 重写，作用于符号 symbol
        debug("Integral: {} is rewritten with {} on symbol: {}".format(integrand, rewrite, symbol))
        # 如果条件 condition 函数对当前积分元组 integral 返回 True
        if condition(*integral):
            # 利用 rewrite 函数重写积分元组 integral
            rewritten = rewrite(*integral)
            # 如果重写后的结果 rewritten 不等于原始被积函数 integrand
            if rewritten != integrand:
                # 计算重写后的子步骤 substep
                substep = integral_steps(rewritten, symbol)
                # 如果 substep 不是 DontKnowRule 类型且不为空
                if not isinstance(substep, DontKnowRule) and substep:
                    # 返回一个 RewriteRule 对象，表示积分重写的规则
                    return RewriteRule(integrand, symbol, rewritten, substep)
    return _rewriter

# 定义一个函数 proxy_rewriter，根据其他条件重写积分元组
def proxy_rewriter(condition, rewrite):
    """Strategy that rewrites an integrand based on some other criteria."""
    # 定义内部函数 _proxy_rewriter，接受一个 criteria 元组作为参数
    def _proxy_rewriter(criteria):
        # 解析 criteria 元组，分别获取条件 criteria 和积分元组 integral
        criteria, integral = criteria
        # 解析积分元组，分别获取被积函数 integrand 和符号 symbol
        integrand, symbol = integral
        # 输出调试信息，显示被积函数 integrand 被 rewrite 重写，作用于符号 symbol，并使用 criteria
        debug("Integral: {} is rewritten with {} on symbol: {} and criteria: {}".format(integrand, rewrite, symbol, criteria))
        # 将 criteria 和 integral 的内容作为参数传递给 condition 函数
        args = criteria + list(integral)
        # 如果条件 condition 返回 True
        if condition(*args):
            # 利用 rewrite 函数重写积分元组 integral
            rewritten = rewrite(*args)
            # 如果重写后的结果 rewritten 不等于原始被积函数 integrand
            if rewritten != integrand:
                # 返回一个 RewriteRule 对象，表示积分重写的规则
                return RewriteRule(integrand, symbol, rewritten, integral_steps(rewritten, symbol))
    return _proxy_rewriter

# 定义一个函数 multiplexer，根据条件应用相应的规则
def multiplexer(conditions):
    """Apply the rule that matches the condition, else None"""
    # 定义内部函数 multiplexer_rl，接受一个表达式 expr 作为参数
    def multiplexer_rl(expr):
        # 遍历 conditions 字典的键值对
        for key, rule in conditions.items():
            # 如果 key 函数对 expr 返回 True
            if key(expr):
                # 应用对应的 rule 函数，并返回结果
                return rule(expr)
    return multiplexer_rl

# 定义一个函数 alternatives，将多个规则组合成一个 AlternativeRule
def alternatives(*rules):
    """Strategy that makes an AlternativeRule out of multiple possible results."""
    # 定义内部函数 _alternatives，接受一个积分元组 integral 作为参数
    def _alternatives(integral):
        # 初始化一个空列表 alts，用于存储备选规则
        alts = []
        # 初始化计数器 count
        count = 0
        # 输出调试信息，显示备选规则列表
        debug("List of Alternative Rules")
        # 遍历 rules 中的每一个 rule
        for rule in rules:
            # 计数器加一
            count = count + 1
            # 输出调试信息，显示规则编号和具体规则内容
            debug("Rule {}: {}".format(count, rule))

            # 应用当前规则 rule 到积分元组 integral 上，获取结果
            result = rule(integral)
            # 如果 result 存在且不是 DontKnowRule 类型，且与 integral 不相同，并且不在 alts 列表中
            if (result and not isinstance(result, DontKnowRule) and
                result != integral and result not in alts):
                # 将结果 result 添加到 alts 列表中
                alts.append(result)
        # 如果 alts 列表长度为 1，返回唯一的备选规则
        if len(alts) == 1:
            return alts[0]
        # 如果 alts 列表不为空
        elif alts:
            # 过滤掉包含 DontKnowRule 的规则
            doable = [rule for rule in alts if not rule.contains_dont_know()]
            # 如果有可行的备选规则
            if doable:
                # 返回一个 AlternativeRule 对象，表示多个可行备选规则
                return AlternativeRule(*integral, doable)
            else:
                # 返回一个 AlternativeRule 对象，表示所有备选规则
                return AlternativeRule(*integral, alts)
    return _alternatives

# 定义一个函数 constant_rule，返回一个 ConstantRule 对象
def constant_rule(integral):
    return ConstantRule(*integral)

# 定义一个函数 power_rule，处理幂函数的积分规则
def power_rule(integral):
    # 解析积分元组，分别获取被积函数 integrand 和符号 symbol
    integrand, symbol = integral
    # 分解被积函数 integrand 为底数 base 和指数 expt
    base, expt = integrand.as_base_exp()

    # 如果符号 symbol 不在指数 expt 的自由符号中，并且底数 base 是符号类型
    if symbol not in expt.free_symbols and isinstance(base, Symbol):
        # 如果简化后的指数 expt + 1 等于 0
        if simplify(expt + 1) == 0:
            # 返回一个 ReciprocalRule 对象，表示倒数规则
            return ReciprocalRule(integrand, symbol, base)
        # 返回一个 PowerRule 对象，表示幂函数积分规则
        return PowerRule(integrand, symbol, base, expt)
    # 如果符号不在基本表达式的自由符号中，并且指数是符号类型
    elif symbol not in base.free_symbols and isinstance(expt, Symbol):
        # 创建一个指数规则对象
        rule = ExpRule(integrand, symbol, base, expt)

        # 如果 log(base) 不为零的模糊非操作结果为真
        if fuzzy_not(log(base).is_zero):
            # 返回指数规则对象
            return rule
        # 否则，如果 log(base) 为零
        elif log(base).is_zero:
            # 返回常数规则对象（1，符号）
            return ConstantRule(1, symbol)

        # 返回分段规则对象，根据条件判断选择返回指数规则或者常数规则
        return PiecewiseRule(integrand, symbol, [
            (rule, Ne(log(base), 0)),
            (ConstantRule(1, symbol), True)
        ])
# 定义一个函数用于处理指数函数的积分规则
def exp_rule(integral):
    integrand, symbol = integral
    # 如果被积函数的第一个参数是符号，则应用指数函数的积分规则
    if isinstance(integrand.args[0], Symbol):
        return ExpRule(integrand, symbol, E, integrand.args[0])


# 定义一个函数用于处理正交多项式的积分规则
def orthogonal_poly_rule(integral):
    # 定义不同正交多项式类对应的积分规则类
    orthogonal_poly_classes = {
        jacobi: JacobiRule,
        gegenbauer: GegenbauerRule,
        chebyshevt: ChebyshevTRule,
        chebyshevu: ChebyshevURule,
        legendre: LegendreRule,
        hermite: HermiteRule,
        laguerre: LaguerreRule,
        assoc_laguerre: AssocLaguerreRule
    }
    # 定义需要特定变量索引的正交多项式类
    orthogonal_poly_var_index = {
        jacobi: 3,
        gegenbauer: 2,
        assoc_laguerre: 2
    }
    integrand, symbol = integral
    # 遍历所有正交多项式类，查找匹配的类别
    for klass in orthogonal_poly_classes:
        if isinstance(integrand, klass):
            var_index = orthogonal_poly_var_index.get(klass, 1)
            # 如果被积函数的特定参数是符号，并且前面的参数中不包含该符号，则应用对应的积分规则
            if (integrand.args[var_index] is symbol and not
                any(v.has(symbol) for v in integrand.args[:var_index])):
                    return orthogonal_poly_classes[klass](integrand, symbol, *integrand.args[:var_index])


# 定义一个空列表，用于存储特殊函数模式的元组
_special_function_patterns: list[tuple[Type, Expr, Callable | None, tuple]] = []
# 定义一个空列表，用于存储通配符（未指定的变量）
_wilds = []
# 定义一个虚拟符号 'x'
_symbol = Dummy('x')


# 定义一个函数用于处理特殊函数的积分规则
def special_function_rule(integral):
    integrand, symbol = integral
    # 如果特殊函数模式列表为空，则定义一些通配符模式并添加到列表中
    if not _special_function_patterns:
        # 定义通配符并设置排除条件和属性
        a = Wild('a', exclude=[_symbol], properties=[lambda x: not x.is_zero])
        b = Wild('b', exclude=[_symbol])
        c = Wild('c', exclude=[_symbol])
        d = Wild('d', exclude=[_symbol], properties=[lambda x: not x.is_zero])
        e = Wild('e', exclude=[_symbol], properties=[
            lambda x: not (x.is_nonnegative and x.is_integer)])
        # 将通配符添加到通配符列表
        _wilds.extend((a, b, c, d, e))

        # 定义线性模式和二次模式
        linear_pattern = a*_symbol + b
        quadratic_pattern = a*_symbol**2 + b*_symbol + c

        # 将特殊函数模式添加到特殊函数模式列表
        _special_function_patterns.extend((
            (Mul, exp(linear_pattern, evaluate=False)/_symbol, None, EiRule),
            (Mul, cos(linear_pattern, evaluate=False)/_symbol, None, CiRule),
            (Mul, cosh(linear_pattern, evaluate=False)/_symbol, None, ChiRule),
            (Mul, sin(linear_pattern, evaluate=False)/_symbol, None, SiRule),
            (Mul, sinh(linear_pattern, evaluate=False)/_symbol, None, ShiRule),
            (Pow, 1/log(linear_pattern, evaluate=False), None, LiRule),
            (exp, exp(quadratic_pattern, evaluate=False), None, ErfRule),
            (sin, sin(quadratic_pattern, evaluate=False), None, FresnelSRule),
            (cos, cos(quadratic_pattern, evaluate=False), None, FresnelCRule),
            (Mul, _symbol**e*exp(a*_symbol, evaluate=False), None, UpperGammaRule),
            (Mul, polylog(b, a*_symbol, evaluate=False)/_symbol, None, PolylogRule),
            (Pow, 1/sqrt(a - d*sin(_symbol, evaluate=False)**2),
                lambda a, d: a != d, EllipticFRule),
            (Pow, sqrt(a - d*sin(_symbol, evaluate=False)**2),
                lambda a, d: a != d, EllipticERule),
        ))

    # 使用通配符列表对积分被积函数进行匹配
    _integrand = integrand.subs(symbol, _symbol)

    # 遍历特殊函数模式列表，寻找匹配的模式并应用对应的规则
    for type_, pattern, constraint, rule in _special_function_patterns:
        if isinstance(_integrand, type_):
            match = _integrand.match(pattern)
            if match:
                # 提取匹配到的通配符的值
                wild_vals = tuple(match.get(w) for w in _wilds
                                  if match.get(w) is not None)
                # 如果有约束条件，并且不满足约束条件，则跳过当前模式
                if constraint is None or constraint(*wild_vals):
                    # 应用规则并返回结果
                    return rule(integrand, symbol, *wild_vals)
def _add_degenerate_step(generic_cond, generic_step: Rule, degenerate_step: Rule | None) -> Rule:
    # 如果 degenerate_step 为 None，则返回 generic_step
    if degenerate_step is None:
        return generic_step
    # 如果 generic_step 是 PiecewiseRule 类型
    if isinstance(generic_step, PiecewiseRule):
        # 获取其子函数列表，并将条件与 generic_cond 简化后的逻辑与结果配对
        subfunctions = [(substep, (cond & generic_cond).simplify())
                        for substep, cond in generic_step.subfunctions]
    else:
        # 否则创建一个只包含 generic_step 和 generic_cond 的子函数列表
        subfunctions = [(generic_step, generic_cond)]
    # 如果 degenerate_step 是 PiecewiseRule 类型
    if isinstance(degenerate_step, PiecewiseRule):
        # 将其子函数列表添加到 subfunctions 中
        subfunctions += degenerate_step.subfunctions
    else:
        # 否则将 degenerate_step 与 S.true 添加到 subfunctions 中
        subfunctions.append((degenerate_step, S.true))
    # 返回一个新的 PiecewiseRule 对象，包含 generic_step 的被积函数、变量及 subfunctions
    return PiecewiseRule(generic_step.integrand, generic_step.variable, subfunctions)


def nested_pow_rule(integral: IntegralInfo):
    # nested (c*(a+b*x)**d)**e 的处理函数
    integrand, x = integral

    # 定义通配符，用于匹配 a+b*x 的模式
    a_ = Wild('a', exclude=[x])
    b_ = Wild('b', exclude=[x, 0])
    pattern = a_+b_*x
    generic_cond = S.true  # 初始化通用条件为真

    class NoMatch(Exception):
        pass

    def _get_base_exp(expr: Expr) -> tuple[Expr, Expr]:
        # 如果表达式不含 x 自由变量，则基数为 1，指数为 0
        if not expr.has_free(x):
            return S.One, S.Zero
        # 如果表达式是乘法
        if expr.is_Mul:
            # 提取出系数和乘法项
            _, terms = expr.as_coeff_mul()
            if not terms:
                return S.One, S.Zero
            # 递归处理每个乘法项，并收集结果
            results = [_get_base_exp(term) for term in terms]
            bases = {b for b, _ in results}
            bases.discard(S.One)
            # 如果所有乘法项的基数相同，则返回这个基数和所有指数的和
            if len(bases) == 1:
                return bases.pop(), Add(*(e for _, e in results))
            raise NoMatch
        # 如果表达式是幂次
        if expr.is_Pow:
            b, e = expr.base, expr.exp  # type: ignore
            # 如果指数 e 含有 x 自由变量，则抛出 NoMatch 异常
            if e.has_free(x):
                raise NoMatch
            base_, sub_exp = _get_base_exp(b)
            return base_, sub_exp * e
        # 尝试匹配表达式与模式 pattern
        match = expr.match(pattern)
        if match:
            a, b = match[a_], match[b_]
            base_ = x + a/b  # 计算基数的表达式
            nonlocal generic_cond
            generic_cond = Ne(b, 0)  # 更新通用条件，排除 b 为 0 的情况
            return base_, S.One
        raise NoMatch

    try:
        base, exp_ = _get_base_exp(integrand)  # 尝试获取基数和指数
    except NoMatch:
        return  # 如果无法匹配，则返回

    if generic_cond is S.true:
        degenerate_step = None  # 如果通用条件为真，则 degenerate_step 为 None
    else:
        # 否则创建一个 ConstantRule，处理在 x=0 处的常数情况
        degenerate_step = ConstantRule(integrand.subs(x, 0), x)

    # 创建 NestedPowRule 对象，处理 nested (c*(a+b*x)**d)**e 结构
    generic_step = NestedPowRule(integrand, x, base, exp_)
    return _add_degenerate_step(generic_cond, generic_step, degenerate_step)


def inverse_trig_rule(integral: IntegralInfo, degenerate=True):
    """
    Set degenerate=False on recursive call where coefficient of quadratic term
    is assumed non-zero.
    """
    integrand, symbol = integral
    base, exp = integrand.as_base_exp()

    # 定义通配符，匹配 a + b*symbol + c*symbol**2 的模式
    a = Wild('a', exclude=[symbol])
    b = Wild('b', exclude=[symbol])
    c = Wild('c', exclude=[symbol, 0])
    match = base.match(a + b*symbol + c*symbol**2)

    if not match:
        return  # 如果无法匹配模式，则返回

    # 如果匹配成功，可以处理反三角函数的积分规则
    def make_inverse_trig(RuleClass, a, sign_a, c, sign_c, h) -> Rule:
        u_var = Dummy("u")
        # 重写表达式为倒数形式
        rewritten = 1/sqrt(sign_a*a + sign_c*c*(symbol-h)**2)  # a>0, c>0
        # 计算二次型的基本形式
        quadratic_base = sqrt(c/a)*(symbol-h)
        # 计算常数项
        constant = 1/sqrt(c)
        u_func = None
        # 如果二次型的基本形式不是 symbol，则将其视为函数 u_func
        if quadratic_base is not symbol:
            u_func = quadratic_base
            quadratic_base = u_var
        # 计算标准形式
        standard_form = 1/sqrt(sign_a + sign_c*quadratic_base**2)
        # 创建子步骤 Rule 对象
        substep = RuleClass(standard_form, quadratic_base)
        # 如果常数项不为 1，则引入常数乘法规则
        if constant != 1:
            substep = ConstantTimesRule(constant*standard_form, symbol, constant, standard_form, substep)
        # 如果存在 u_func，则引入变换规则 URule
        if u_func is not None:
            substep = URule(rewritten, symbol, u_var, u_func, substep)
        # 如果 h 不等于 0，则引入完全平方规则 CompleteSquareRule
        if h != 0:
            substep = CompleteSquareRule(integrand, symbol, rewritten, substep)
        # 返回子步骤对象
        return substep

    # 获取 a, b, c 的值，若未找到则置为 0
    a, b, c = [match.get(i, S.Zero) for i in (a, b, c)]
    # 创建通用条件判断 Ne(c, 0)
    generic_cond = Ne(c, 0)
    # 如果不是退化情况或者 c 不为 0
    if not degenerate or generic_cond is S.true:
        degenerate_step = None
    # 如果 b 为 0，则退化步骤为常数规则 ConstantRule
    elif b.is_zero:
        degenerate_step = ConstantRule(a ** exp, symbol)
    # 否则，退化步骤为 sqrt_linear_rule 函数的返回结果
    else:
        degenerate_step = sqrt_linear_rule(IntegralInfo((a + b * symbol) ** exp, symbol))

    # 如果简化后的表达式为 0
    if simplify(2*exp + 1) == 0:
        # 计算 h, k 的值，重写基础表达式为 k + c*(symbol-h)**2
        h, k = -b/(2*c), a - b**2/(4*c)
        # 创建非二次项条件 Ne(k, 0)
        non_square_cond = Ne(k, 0)
        square_step = None
        # 如果非二次项条件为真，则引入嵌套幂规则 NestedPowRule
        if non_square_cond is S.true:
            square_step = NestedPowRule(1/sqrt(c*(symbol-h)**2), symbol, symbol-h, S.NegativeOne)
        # 如果非二次项条件为假，则直接返回 square_step
        if non_square_cond is S.false:
            return square_step
        # 否则，引入倒数平方根二次规则 ReciprocalSqrtQuadraticRule
        generic_step = ReciprocalSqrtQuadraticRule(integrand, symbol, a, b, c)
        # 最终步骤为添加退化步骤后的结果 _add_degenerate_step
        step = _add_degenerate_step(non_square_cond, generic_step, square_step)
        # 如果 k 和 c 均为实数
        if k.is_real and c.is_real:
            # 创建规则列表 rules
            rules = []
            # 遍历规则和条件的组合
            for args, cond in (
                ((ArcsinRule, k, 1, -c, -1, h), And(k > 0, c < 0)),  # 1-x**2
                ((ArcsinhRule, k, 1, c, 1, h), And(k > 0, c > 0)),  # 1+x**2
            ):
                # 如果条件为真，则返回 make_inverse_trig 函数的结果
                if cond is S.true:
                    return make_inverse_trig(*args)
                # 如果条件不为假，则将结果添加到 rules 列表中
                if cond is not S.false:
                    rules.append((make_inverse_trig(*args), cond))
            # 如果 rules 列表不为空
            if rules:
                # 如果 k 不是正数，则添加通用步骤
                if not k.is_positive:
                    rules.append((generic_step, S.true))
                # 最终步骤为 PiecewiseRule
                step = PiecewiseRule(integrand, symbol, rules)
            else:
                step = generic_step
        # 返回添加退化步骤后的结果
        return _add_degenerate_step(generic_cond, step, degenerate_step)
    # 如果指数为 Half
    if exp == S.Half:
        # 创建平方根二次规则 SqrtQuadraticRule
        step = SqrtQuadraticRule(integrand, symbol, a, b, c)
        # 返回添加退化步骤后的结果
        return _add_degenerate_step(generic_cond, step, degenerate_step)
# 定义一个函数 add_rule，接收一个积分信息的元组 integral
def add_rule(integral):
    # 将积分被积函数和符号分离出来
    integrand, symbol = integral
    # 对被积函数中按次序排列的每一项，分别进行积分步骤的获取
    results = [integral_steps(g, symbol)
               for g in integrand.as_ordered_terms()]
    # 如果结果中包含 None，则返回 None，否则返回 AddRule 对象
    return None if None in results else AddRule(integrand, symbol, results)


# 定义一个函数 mul_rule，接收一个 IntegralInfo 类型的积分信息对象 integral
def mul_rule(integral: IntegralInfo):
    # 将积分被积函数和符号分离出来
    integrand, symbol = integral

    # 处理常数乘以函数的情况
    coeff, f = integrand.as_independent(symbol)
    if coeff != 1:
        # 获取下一步的积分步骤
        next_step = integral_steps(f, symbol)
        # 如果下一步不为 None，则返回 ConstantTimesRule 对象
        if next_step is not None:
            return ConstantTimesRule(integrand, symbol, coeff, f, next_step)


# 定义一个函数 _parts_rule，接收被积函数 integrand 和符号 symbol，返回一个元组或 None
def _parts_rule(integrand, symbol) -> tuple[Expr, Expr, Expr, Expr, Rule] | None:
    # LIATE 规则：
    # log, inverse trig, algebraic, trigonometric, exponential

    # 从被积函数中抽取代数部分
    def pull_out_algebraic(integrand):
        # 化简和合并被积函数
        integrand = integrand.cancel().together()
        # 如果被积函数是 Piecewise 或不是乘法表达式，则返回空列表；否则返回代数表达式的列表
        algebraic = ([] if isinstance(integrand, Piecewise) or not integrand.is_Mul
                     else [arg for arg in integrand.args if arg.is_algebraic_expr(symbol)])
        if algebraic:
            # 构建代数部分乘积
            u = Mul(*algebraic)
            # 计算余下部分 dv
            dv = (integrand / u).cancel()
            return u, dv

    # 定义一个函数 pull_out_u，接收多个函数并返回一个函数，用于抽取特定函数类型的部分
    def pull_out_u(*functions) -> Callable[[Expr], tuple[Expr, Expr] | None]:
        def pull_out_u_rl(integrand: Expr) -> tuple[Expr, Expr] | None:
            # 如果被积函数中包含指定函数类型，则抽取对应的部分
            if any(integrand.has(f) for f in functions):
                args = [arg for arg in integrand.args
                        if any(isinstance(arg, cls) for cls in functions)]
                if args:
                    # 构建函数类型部分的乘积
                    u = Mul(*args)
                    # 计算余下部分 dv
                    dv = integrand / u
                    return u, dv
            return None

        return pull_out_u_rl

    # LIATE 规则的具体应用，定义抽取函数的列表
    liate_rules = [pull_out_u(log), pull_out_u(*inverse_trig_functions),
                   pull_out_algebraic, pull_out_u(sin, cos),
                   pull_out_u(exp)]

    # 创建一个虚拟符号对象，用于处理 log(x) 和 atan(x) 的特殊情况
    dummy = Dummy("temporary")
    # 如果被积函数是 log(x) 或 inverse_trig_functions 中的函数，则乘以虚拟符号
    if isinstance(integrand, (log, *inverse_trig_functions)):
        integrand = dummy * integrand
    # 遍历 liate_rules 列表中的规则，同时追踪索引和当前规则对象
    for index, rule in enumerate(liate_rules):
        # 对当前积分被积函数应用规则，得到规则的结果
        result = rule(integrand)

        # 如果结果存在
        if result:
            # 解构结果得到 u 和 dv
            u, dv = result

            # 如果 u 不是常数，且不是包含积分变量的常数，则返回 None
            if symbol not in u.free_symbols and not u.has(dummy):
                return None

            # 将 u 中的 dummy 符号替换为 1
            u = u.subs(dummy, 1)
            # 将 dv 中的 dummy 符号替换为 1
            dv = dv.subs(dummy, 1)

            # 如果选取的非多项式代数式作为被积函数 dv，则返回 None
            if rule == pull_out_algebraic and not u.is_polynomial(symbol):
                return None

            # 如果 u 是对数函数
            if isinstance(u, log):
                # 计算 dv 的倒数
                rec_dv = 1/dv
                # 如果倒数是多项式且次数为1，则返回 None
                if (rec_dv.is_polynomial(symbol) and
                    degree(rec_dv, symbol) == 1):
                    return None

            # 如果当前规则是 pull_out_algebraic，且 dv 满足一定条件
            if rule == pull_out_algebraic:
                # 如果 dv 是导数、三角函数或正交多项式
                if dv.is_Derivative or dv.has(TrigonometricFunction) or \
                        isinstance(dv, OrthogonalPolynomial):
                    # 对 dv 进行积分步骤分析，检查是否包含未知量
                    v_step = integral_steps(dv, symbol)
                    if v_step.contains_dont_know():
                        return None
                    else:
                        # 计算 u 的导数
                        du = u.diff(symbol)
                        # 计算 v 的值
                        v = v_step.eval()
                        return u, dv, v, du, v_step

            # 判断 dv 是否适合积分
            accept = False
            # 前两个规则通常尝试对数和反三角函数
            if index < 2:
                accept = True
            # 如果当前规则是 pull_out_algebraic，且 dv 的参数都是 sin、cos、exp 函数
            elif (rule == pull_out_algebraic and dv.args and
                  all(isinstance(a, (sin, cos, exp))
                      for a in dv.args)):
                accept = True
            else:
                # 检查后续规则，看是否有规则的 u 和 dv 相等
                for lrule in liate_rules[index + 1:]:
                    r = lrule(integrand)
                    if r and r[0].subs(dummy, 1).equals(dv):
                        accept = True
                        break

            # 如果接受当前 dv，则继续处理
            if accept:
                # 计算 u 的导数
                du = u.diff(symbol)
                # 简化 dv 并进行积分步骤分析
                v_step = integral_steps(simplify(dv), symbol)
                # 如果步骤分析不包含未知量，则计算 v 的值并返回结果
                if not v_step.contains_dont_know():
                    v = v_step.eval()
                    return u, dv, v, du, v_step

    # 若未找到符合条件的 u, dv 组合，则返回 None
    return None
# 分部积分规则的实现函数，接受一个积分元组作为参数
def parts_rule(integral):
    # 解构积分元组，获取被积函数和积分变量
    integrand, symbol = integral
    # 将被积函数分解为常数和另一部分
    constant, integrand = integrand.as_coeff_Mul()

    # 调用实际的分部积分规则计算函数
    result = _parts_rule(integrand, symbol)

    # 存储每一步的详细计算过程
    steps = []

    # 如果有计算结果
    if result:
        # 解构分部积分的结果，获取各个部分
        u, dv, v, du, v_step = result
        # 输出调试信息
        debug("u : {}, dv : {}, v : {}, du : {}, v_step: {}".format(u, dv, v, du, v_step))
        # 将结果添加到步骤列表中
        steps.append(result)

        # 如果 v 是积分对象，则直接返回
        if isinstance(v, Integral):
            return

        # 对于一些特定的 u 函数（如 sin、cos、exp、sinh、cosh），设置 u 被使用的次数限制
        if isinstance(u, (sin, cos, exp, sinh, cosh)):
            cachekey = u.xreplace({symbol: _cache_dummy})
            # 如果超过次数限制，则直接返回
            if _parts_u_cache[cachekey] > 2:
                return
            _parts_u_cache[cachekey] += 1

        # 尝试进行几次循环的分部积分
        for _ in range(4):
            # 输出调试信息
            debug("Cyclic integration {} with v: {}, du: {}, integrand: {}".format(_, v, du, integrand))
            # 计算系数
            coefficient = ((v * du) / integrand).cancel()
            # 如果系数为 1，则中断循环
            if coefficient == 1:
                break
            # 如果积分变量不在系数的自由符号中，则应用循环分部积分规则
            if symbol not in coefficient.free_symbols:
                rule = CyclicPartsRule(integrand, symbol,
                    [PartsRule(None, None, u, dv, v_step, None)
                     for (u, dv, v, du, v_step) in steps],
                    (-1) ** len(steps) * coefficient)
                # 如果常数不为 1 并且规则存在，则将常数合并到规则中
                if (constant != 1) and rule:
                    rule = ConstantTimesRule(constant * integrand, symbol, constant, integrand, rule)
                return rule

            # 对常数敏感的 _parts_rule 函数，将常数分解出来
            next_constant, next_integrand = (v * du).as_coeff_Mul()
            # 再次调用分部积分规则计算函数
            result = _parts_rule(next_integrand, symbol)

            # 如果有计算结果
            if result:
                # 解构分部积分的结果，包括乘以常数的部分
                u, dv, v, du, v_step = result
                u *= next_constant
                du *= next_constant
                steps.append((u, dv, v, du, v_step))
            else:
                break

    # 定义第二步的计算函数，递归构建分部积分规则
    def make_second_step(steps, integrand):
        if steps:
            u, dv, v, du, v_step = steps[0]
            return PartsRule(integrand, symbol, u, dv, v_step, make_second_step(steps[1:], v * du))
        return integral_steps(integrand, symbol)

    # 如果有步骤记录，则调用第二步计算函数
    if steps:
        u, dv, v, du, v_step = steps[0]
        rule = PartsRule(integrand, symbol, u, dv, v_step, make_second_step(steps[1:], v * du))
        # 如果常数不为 1 并且规则存在，则将常数合并到规则中
        if (constant != 1) and rule:
            rule = ConstantTimesRule(constant * integrand, symbol, constant, integrand, rule)
        return rule


# 三角函数积分规则函数，根据被积函数返回相应的积分规则对象
def trig_rule(integral):
    # 解构积分元组，获取被积函数和积分变量
    integrand, symbol = integral
    # 如果被积函数是 sin(symbol)，返回对应的正弦积分规则对象
    if integrand == sin(symbol):
        return SinRule(integrand, symbol)
    # 如果被积函数是 cos(symbol)，返回对应的余弦积分规则对象
    if integrand == cos(symbol):
        return CosRule(integrand, symbol)
    # 如果被积函数是 sec(symbol)^2，返回对应的 sec^2 积分规则对象
    if integrand == sec(symbol)**2:
        return Sec2Rule(integrand, symbol)
    # 如果被积函数是 csc(symbol)^2，返回对应的 csc^2 积分规则对象
    if integrand == csc(symbol)**2:
        return Csc2Rule(integrand, symbol)

    # 如果被积函数是 tan 函数的实例，则将其重写为 sin/cos 形式
    if isinstance(integrand, tan):
        rewritten = sin(*integrand.args) / cos(*integrand.args)
    # 如果被积函数是 cot 函数的实例，则按照 cot 函数的积分规则进行重写
    elif isinstance(integrand, cot):
        # 将 cot 函数重写为 cos 函数与 sin 函数的比值
        rewritten = cos(*integrand.args) / sin(*integrand.args)
    
    # 如果被积函数是 sec 函数的实例，则按照 sec 函数的积分规则进行重写
    elif isinstance(integrand, sec):
        # 获取 sec 函数的参数
        arg = integrand.args[0]
        # 将 sec 函数重写为其积分规则的表达式
        rewritten = ((sec(arg)**2 + tan(arg) * sec(arg)) /
                     (sec(arg) + tan(arg)))
    
    # 如果被积函数是 csc 函数的实例，则按照 csc 函数的积分规则进行重写
    elif isinstance(integrand, csc):
        # 获取 csc 函数的参数
        arg = integrand.args[0]
        # 将 csc 函数重写为其积分规则的表达式
        rewritten = ((csc(arg)**2 + cot(arg) * csc(arg)) /
                     (csc(arg) + cot(arg)))
    
    # 如果被积函数不属于上述特定的三种函数类型，则直接返回，不做处理
    else:
        return

    # 返回应用重写规则后的对象，使用 RewriteRule 类将原始函数、积分变量、重写后的函数及积分步骤整合为一个对象
    return RewriteRule(integrand, symbol, rewritten, integral_steps(rewritten, symbol))
# 根据积分信息对象进行三角函数乘积规则的应用
def trig_product_rule(integral: IntegralInfo):
    # 从积分信息对象中获取被积函数和符号变量
    integrand, symbol = integral
    # 如果被积函数是 sec(symbol) * tan(symbol)，则应用 SecTanRule
    if integrand == sec(symbol) * tan(symbol):
        return SecTanRule(integrand, symbol)
    # 如果被积函数是 csc(symbol) * cot(symbol)，则应用 CscCotRule
    if integrand == csc(symbol) * cot(symbol):
        return CscCotRule(integrand, symbol)


# 对于二次分母的积分规则处理
def quadratic_denom_rule(integral):
    # 从积分对象中获取被积函数和符号变量
    integrand, symbol = integral
    # 定义通配符，排除符号变量
    a = Wild('a', exclude=[symbol])
    b = Wild('b', exclude=[symbol])
    c = Wild('c', exclude=[symbol])

    # 匹配被积函数是否符合 a / (b * symbol ** 2 + c) 的形式
    match = integrand.match(a / (b * symbol ** 2 + c))

    if match:
        # 提取匹配的 a, b, c 值
        a, b, c = match[a], match[b], match[c]
        # 创建通用规则对象 ArctanRule
        general_rule = ArctanRule(integrand, symbol, a, b, c)
        
        # 如果 b 和 c 是扩展实数
        if b.is_extended_real and c.is_extended_real:
            # 判断 c/b 是否大于 0
            positive_cond = c/b > 0
            # 如果为真，返回通用规则对象
            if positive_cond is S.true:
                return general_rule
            # 计算系数和常数
            coeff = a/(2*sqrt(-c)*sqrt(b))
            constant = sqrt(-c/b)
            r1 = 1/(symbol-constant)
            r2 = 1/(symbol+constant)
            # 构建对数步骤列表
            log_steps = [ReciprocalRule(r1, symbol, symbol-constant),
                         ConstantTimesRule(-r2, symbol, -1, r2, ReciprocalRule(r2, symbol, symbol+constant))]
            rewritten = sub = r1 - r2
            # 构建负数步骤
            negative_step = AddRule(sub, symbol, log_steps)
            # 如果系数不为 1，则重写为 Mul(coeff, sub, evaluate=False)
            if coeff != 1:
                rewritten = Mul(coeff, sub, evaluate=False)
                negative_step = ConstantTimesRule(rewritten, symbol, coeff, sub, negative_step)
            # 使用重写规则 RewriteRule 处理被积函数
            negative_step = RewriteRule(integrand, symbol, rewritten, negative_step)
            # 如果 positive_cond 为假，返回负数步骤
            if positive_cond is S.false:
                return negative_step
            # 返回分段规则 PiecewiseRule
            return PiecewiseRule(integrand, symbol, [(general_rule, positive_cond), (negative_step, S.true)])

        # 如果不满足上述条件，应用幂规则 PowerRule
        power = PowerRule(integrand, symbol, symbol, -2)
        # 如果 b 不等于 1，则使用常数乘积规则 ConstantTimesRule
        if b != 1:
            power = ConstantTimesRule(integrand, symbol, 1/b, symbol**-2, power)

        # 返回分段规则 PiecewiseRule
        return PiecewiseRule(integrand, symbol, [(general_rule, Ne(c, 0)), (power, True)])

    # 处理形如 a / (b * symbol ** 2 + c * symbol + d) 的情况
    d = Wild('d', exclude=[symbol])
    match2 = integrand.match(a / (b * symbol ** 2 + c * symbol + d))
    if match2:
        b, c =  match2[b], match2[c]
        # 如果 b 等于零，返回空
        if b.is_zero:
            return
        # 创建虚拟符号 u
        u = Dummy('u')
        u_func = symbol + c/(2*b)
        # 替换符号变量 symbol，得到 integrand2
        integrand2 = integrand.subs(symbol, u - c / (2*b))
        # 进行下一步的积分处理
        next_step = integral_steps(integrand2, u)
        if next_step:
            # 返回 URule 规则对象
            return URule(integrand2, symbol, u, u_func, next_step)
        else:
            return
    # 处理形如 (a* symbol + b) / (c * symbol ** 2 + d * symbol + e) 的情况
    e = Wild('e', exclude=[symbol])
    match3 = integrand.match((a* symbol + b) / (c * symbol ** 2 + d * symbol + e))
    # 如果 match3 不为 None，则继续执行下面的逻辑
    if match3:
        # 从 match3 中获取 a, b, c, d, e 的值
        a, b, c, d, e = match3[a], match3[b], match3[c], match3[d], match3[e]
        # 如果 c 是零，则直接返回，中止函数执行
        if c.is_zero:
            return
        # 计算分母，这里使用 c*symbol**2 + d*symbol + e
        denominator = c * symbol**2 + d * symbol + e
        # 计算常数 const，这里是 a / (2*c)
        const = a / (2*c)
        # 计算 numer1，这里是 2*c*symbol + d
        numer1 = (2*c*symbol + d)
        # 计算 numer2，这里是 -const*d + b
        numer2 = -const * d + b
        # 创建一个符号变量 u
        u = Dummy('u')
        # 执行第一步积分规则 URule
        step1 = URule(integrand, symbol, u, denominator, integral_steps(u**(-1), u))
        # 如果 const 不等于 1，则执行常数乘法规则 ConstantTimesRule
        if const != 1:
            step1 = ConstantTimesRule(const * numer1 / denominator, symbol,
                                      const, numer1 / denominator, step1)
        # 如果 numer2 是零，则直接返回 step1
        if numer2.is_zero:
            return step1
        # 执行第二步积分规则 integral_steps
        step2 = integral_steps(numer2 / denominator, symbol)
        # 将 step1 和 step2 组合成一个加法规则 AddRule 的子步骤
        substeps = AddRule(integrand, symbol, [step1, step2])
        # 计算 rewriten，这里是 const*numer1/denominator + numer2/denominator
        rewriten = const * numer1 / denominator + numer2 / denominator
        # 返回重写规则 RewriteRule 的结果，包括最终表达式和子步骤
        return RewriteRule(integrand, symbol, rewriten, substeps)

    # 如果 match3 是 None，则直接返回，中止函数执行
    return
def sqrt_linear_rule(integral: IntegralInfo):
    """
    Substitute common (a+b*x)**(1/n)
    """
    integrand, x = integral
    a = Wild('a', exclude=[x])  # 定义通配符 a，排除 x 变量
    b = Wild('b', exclude=[x, 0])  # 定义通配符 b，排除 x 变量和常数项
    a0 = b0 = 0  # 初始化 a0 和 b0 为 0
    bases, qs, bs = [], [], []  # 初始化空列表用于存储基数、分母、系数
    for pow_ in integrand.find(Pow):  # 遍历积分被积函数中所有的 Pow 类型的对象
        base, exp_ = pow_.base, pow_.exp  # 提取基数和指数
        if exp_.is_Integer or x not in base.free_symbols:  # 如果指数是整数或基数不包含 x，则跳过
            continue
        if not exp_.is_Rational:  # 如果指数不是有理数，则跳过
            return  # 返回空，结束函数
        match = base.match(a+b*x)  # 尝试匹配基数是否符合形式 a+b*x
        if not match:  # 如果匹配失败，则跳过
            continue
        a1, b1 = match[a], match[b]  # 提取匹配结果中的 a 和 b
        if a0*b1 != a1*b0 or not (b0/b1).is_nonnegative:  # 如果 a0*b1 不等于 a1*b0 或者 b0/b1 不是非负数
            return  # 返回空，结束函数
        if b0 == 0 or (b0/b1 > 1) is S.true:  # 如果 b0 等于 0 或者 b0/b1 大于 1
            a0, b0 = a1, b1  # 更新 a0 和 b0
        bases.append(base)  # 将符合条件的基数添加到列表
        bs.append(b1)  # 将 b1 添加到系数列表
        qs.append(exp_.q)  # 将指数的分母添加到分母列表
    if b0 == 0:  # 如果未找到符合条件的模式
        return  # 返回空，结束函数
    q0: Integer = lcm_list(qs)  # 计算分母列表的最小公倍数
    u_x = (a0 + b0*x)**(1/q0)  # 计算代换公式中的 u_x
    u = Dummy("u")  # 创建一个虚拟变量 u
    substituted = integrand.subs({base**(S.One/q): (b/b0)**(S.One/q)*u**(q0/q)
                                  for base, b, q in zip(bases, bs, qs)}).subs(x, (u**q0-a0)/b0)
    # 用代换公式替换被积函数中的表达式，并将 x 替换为 (u**q0-a0)/b0
    substep = integral_steps(substituted*u**(q0-1)*q0/b0, u)
    # 调用 integral_steps 函数，进行积分步骤的计算
    if not substep.contains_dont_know():  # 如果 substep 不包含 don't know
        step: Rule = URule(integrand, x, u, u_x, substep)  # 创建 URule 规则对象
        generic_cond = Ne(b0, 0)  # 创建通用条件，检查 b0 是否不等于 0
        if generic_cond is not S.true:  # 如果通用条件不为真，可能是退化情况
            simplified = integrand.subs(dict.fromkeys(bs, 0))  # 将系数 bs 替换为 0
            degenerate_step = integral_steps(simplified, x)  # 计算退化情况的积分步骤
            step = PiecewiseRule(integrand, x, [(step, generic_cond), (degenerate_step, S.true)])
            # 创建 PiecewiseRule 规则对象，处理退化和非退化情况
        return step  # 返回积分步骤规则对象


def sqrt_quadratic_rule(integral: IntegralInfo, degenerate=True):
    integrand, x = integral  # 提取被积函数和变量 x
    a = Wild('a', exclude=[x])  # 定义通配符 a，排除 x 变量
    b = Wild('b', exclude=[x])  # 定义通配符 b，排除 x 变量
    c = Wild('c', exclude=[x, 0])  # 定义通配符 c，排除 x 变量和常数项
    f = Wild('f')  # 定义通配符 f
    n = Wild('n', properties=[lambda n: n.is_Integer and n.is_odd])  # 定义通配符 n，要求为奇数整数
    match = integrand.match(f*sqrt(a+b*x+c*x**2)**n)  # 尝试匹配被积函数是否符合形式 f*sqrt(a+b*x+c*x**2)**n
    if not match:  # 如果匹配失败，则返回空，结束函数
        return
    a, b, c, f, n = match[a], match[b], match[c], match[f], match[n]  # 提取匹配结果中的变量
    f_poly = f.as_poly(x)  # 将 f 表达式转化为关于 x 的多项式
    if f_poly is None:  # 如果转化失败，则返回空，结束函数
        return

    generic_cond = Ne(c, 0)  # 创建通用条件，检查 c 是否不等于 0
    if not degenerate or generic_cond is S.true:  # 如果不考虑退化情况或者通用条件为真
        degenerate_step = None  # 不需要进行退化情况的处理
    elif b.is_zero:  # 如果 b 为 0
        degenerate_step = integral_steps(f*sqrt(a)**n, x)  # 计算退化情况的积分步骤
    else:
        degenerate_step = sqrt_linear_rule(IntegralInfo(f*sqrt(a+b*x)**n, x))  # 调用 sqrt_linear_rule 处理退化情况
    def sqrt_quadratic_denom_rule(numer_poly: Poly, integrand: Expr):
        # 定义分母为 sqrt(a+b*x+c*x**2)
        denom = sqrt(a+b*x+c*x**2)
        # 计算数值多项式的次数
        deg = numer_poly.degree()
        if deg <= 1:
            # 当数值多项式的次数小于等于1时，重写被积函数为 (d+e*x)/sqrt(a+b*x+c*x**2)
            e, d = numer_poly.all_coeffs() if deg == 1 else (S.Zero, numer_poly.as_expr())
            # 将分子重写为 A*(2*c*x+b) + B 的形式
            A = e/(2*c)
            B = d-A*b
            pre_substitute = (2*c*x+b)/denom
            constant_step: Rule | None = None
            linear_step: Rule | None = None
            if A != 0:
                # 创建变量 u 并应用幂规则
                u = Dummy("u")
                pow_rule = PowerRule(1/sqrt(u), u, u, -S.Half)
                linear_step = URule(pre_substitute, x, u, a+b*x+c*x**2, pow_rule)
                if A != 1:
                    # 如果 A 不为1，应用常数乘法规则
                    linear_step = ConstantTimesRule(A*pre_substitute, x, A, pre_substitute, linear_step)
            if B != 0:
                # 应用反三角函数积分规则
                constant_step = inverse_trig_rule(IntegralInfo(1/denom, x), degenerate=False)
                if B != 1:
                    # 如果 B 不为1，应用常数乘法规则
                    constant_step = ConstantTimesRule(B/denom, x, B, 1/denom, constant_step)  # type: ignore
            if linear_step and constant_step:
                # 如果存在线性步骤和常数步骤，将它们组合成一个重写规则
                add = Add(A*pre_substitute, B/denom, evaluate=False)
                step: Rule | None = RewriteRule(integrand, x, add, AddRule(add, x, [linear_step, constant_step]))
            else:
                # 否则，选择线性步骤或常数步骤作为积分规则
                step = linear_step or constant_step
        else:
            # 当数值多项式的次数大于1时，使用特定的平方根二次分母规则处理
            coeffs = numer_poly.all_coeffs()
            step = SqrtQuadraticDenomRule(integrand, x, a, b, c, coeffs)
        return step

    if n > 0:  # rewrite poly * sqrt(s)**(2*k-1) to poly*s**k / sqrt(s)
        # 对多项式 f_poly 乘以 (a+b*x+c*x**2)**((n+1)/2)，并重写为 f_poly*s**k / sqrt(s)
        numer_poly = f_poly * (a+b*x+c*x**2)**((n+1)/2)
        rewritten = numer_poly.as_expr()/sqrt(a+b*x+c*x**2)
        # 应用平方根二次分母规则处理重写后的被积函数
        substep = sqrt_quadratic_denom_rule(numer_poly, rewritten)
        # 创建重写规则
        generic_step = RewriteRule(integrand, x, rewritten, substep)
    elif n == -1:
        # 当 n 等于 -1 时，直接应用平方根二次分母规则处理 f_poly
        generic_step = sqrt_quadratic_denom_rule(f_poly, integrand)
    else:
        return  # todo: handle n < -1 case
    # 返回处理后的积分步骤
    return _add_degenerate_step(generic_cond, generic_step, degenerate_step)
# 定义一个函数，用于处理双曲函数积分规则的选择和重写
def hyperbolic_rule(integral: tuple[Expr, Symbol]):
    # 解析输入的积分被积函数和积分变量
    integrand, symbol = integral
    # 如果被积函数是双曲正弦函数，并且其参数是积分变量
    if isinstance(integrand, HyperbolicFunction) and integrand.args[0] == symbol:
        # 返回对应的正弦积分规则对象
        if integrand.func == sinh:
            return SinhRule(integrand, symbol)
        # 返回对应的余弦积分规则对象
        if integrand.func == cosh:
            return CoshRule(integrand, symbol)
        # 如果是双曲正切函数
        u = Dummy('u')
        if integrand.func == tanh:
            # 用正弦和余弦的商重写，并返回重写规则对象
            rewritten = sinh(symbol)/cosh(symbol)
            return RewriteRule(integrand, symbol, rewritten,
                   URule(rewritten, symbol, u, cosh(symbol), ReciprocalRule(1/u, u, u)))
        # 如果是双曲余切函数
        if integrand.func == coth:
            # 用余弦和正弦的商重写，并返回重写规则对象
            rewritten = cosh(symbol)/sinh(symbol)
            return RewriteRule(integrand, symbol, rewritten,
                   URule(rewritten, symbol, u, sinh(symbol), ReciprocalRule(1/u, u, u)))
        else:
            # 对于其它双曲函数，尝试用双曲正切重写，并根据具体函数类型返回对应的重写规则对象
            rewritten = integrand.rewrite(tanh)
            if integrand.func == sech:
                return RewriteRule(integrand, symbol, rewritten,
                       URule(rewritten, symbol, u, tanh(symbol/2),
                       ArctanRule(2/(u**2 + 1), u, S(2), S.One, S.One)))
            if integrand.func == csch:
                return RewriteRule(integrand, symbol, rewritten,
                       URule(rewritten, symbol, u, tanh(symbol/2),
                       ReciprocalRule(1/u, u, u)))

# 缓存化函数，根据给定符号创建 Wild 对象
@cacheit
def make_wilds(symbol):
    # 创建四个 Wild 对象，并指定排除的符号以及额外的属性
    a = Wild('a', exclude=[symbol])
    b = Wild('b', exclude=[symbol])
    m = Wild('m', exclude=[symbol], properties=[lambda n: isinstance(n, Integer)])
    n = Wild('n', exclude=[symbol], properties=[lambda n: isinstance(n, Integer)])
    # 返回创建的 Wild 对象组成的元组
    return a, b, m, n

# 缓存化函数，生成 sin 和 cos 模式的匹配模式及相关 Wild 对象
@cacheit
def sincos_pattern(symbol):
    # 调用 make_wilds 函数创建 Wild 对象
    a, b, m, n = make_wilds(symbol)
    # 创建匹配模式，包括 sin(a*symbol)**m * cos(b*symbol)**n 和相应的 Wild 对象
    pattern = sin(a*symbol)**m * cos(b*symbol)**n
    # 返回匹配模式及相关 Wild 对象
    return pattern, a, b, m, n

# 缓存化函数，生成 tan 和 sec 模式的匹配模式及相关 Wild 对象
@cacheit
def tansec_pattern(symbol):
    # 调用 make_wilds 函数创建 Wild 对象
    a, b, m, n = make_wilds(symbol)
    # 创建匹配模式，包括 tan(a*symbol)**m * sec(b*symbol)**n 和相应的 Wild 对象
    pattern = tan(a*symbol)**m * sec(b*symbol)**n
    # 返回匹配模式及相关 Wild 对象
    return pattern, a, b, m, n

# 缓存化函数，生成 cot 和 csc 模式的匹配模式及相关 Wild 对象
@cacheit
def cotcsc_pattern(symbol):
    # 调用 make_wilds 函数创建 Wild 对象
    a, b, m, n = make_wilds(symbol)
    # 创建匹配模式，包括 cot(a*symbol)**m * csc(b*symbol)**n 和相应的 Wild 对象
    pattern = cot(a*symbol)**m * csc(b*symbol)**n
    # 返回匹配模式及相关 Wild 对象
    return pattern, a, b, m, n

# 缓存化函数，生成 Heaviside 函数模式的匹配模式及相关 Wild 对象
@cacheit
def heaviside_pattern(symbol):
    # 创建两个 Wild 对象，其中一个包含特定属性
    m = Wild('m', exclude=[symbol])
    b = Wild('b', exclude=[symbol])
    g = Wild('g')
    # 创建匹配模式，包括 Heaviside(m*symbol + b) * g 和相关 Wild 对象
    pattern = Heaviside(m*symbol + b) * g
    # 返回匹配模式及相关 Wild 对象
    return pattern, m, b, g

# 定义一个函数，用于将单参数函数转换为接受元组参数的函数
def uncurry(func):
    # 定义内部函数，将接收的参数转发给原始函数
    def uncurry_rl(args):
        return func(*args)
    return uncurry_rl

# 定义一个函数，生成针对三角函数重写的函数
def trig_rewriter(rewrite):
    # 定义内部函数，接收元组参数，并提取其中的参数
    def trig_rewriter_rl(args):
        a, b, m, n, integrand, symbol = args
        # 使用给定的重写函数处理输入的参数，并生成重写规则对象
        rewritten = rewrite(a, b, m, n, integrand, symbol)
        # 如果重写后的结果不等于原始积分被积函数，则返回对应的重写规则对象
        if rewritten != integrand:
            return RewriteRule(integrand, symbol, rewritten, integral_steps(rewritten, symbol))
    return trig_rewriter_rl

# 定义一个函数，检查 sin 和 cos 模式是否为偶数幂次的条件
sincos_botheven_condition = uncurry(
    lambda a, b, m, n, i, s: m.is_even and n.is_even and
    m.is_nonnegative and n.is_nonnegative)

# 定义一个函数，针对 sin 和 cos 模式进行重写
sincos_botheven = trig_rewriter(
    # 定义了一个 lambda 函数，接受六个参数 a, b, m, n, i, symbol，并返回一个计算结果
    lambda a, b, m, n, i, symbol: ( (((1 - cos(2*a*symbol)) / 2) ** (m / 2)) *
                                    (((1 + cos(2*b*symbol)) / 2) ** (n / 2)) ))
sincos_sinodd_condition = uncurry(lambda a, b, m, n, i, s: m.is_odd and m >= 3)


# 定义一个条件函数，判断参数 m 是否为奇数且大于等于 3
sincos_sinodd_condition = uncurry(lambda a, b, m, n, i, s: m.is_odd and m >= 3)




sincos_sinodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 - cos(a*symbol)**2)**((m - 1) / 2) *
                                    sin(a*symbol) *
                                    cos(b*symbol) ** n))


# 定义一个三角函数重写器，处理满足特定条件的积分表达式
sincos_sinodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 - cos(a*symbol)**2)**((m - 1) / 2) *
                                    sin(a*symbol) *
                                    cos(b*symbol) ** n))




sincos_cosodd_condition = uncurry(lambda a, b, m, n, i, s: n.is_odd and n >= 3)


# 定义一个条件函数，判断参数 n 是否为奇数且大于等于 3
sincos_cosodd_condition = uncurry(lambda a, b, m, n, i, s: n.is_odd and n >= 3)




sincos_cosodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 - sin(b*symbol)**2)**((n - 1) / 2) *
                                    cos(b*symbol) *
                                    sin(a*symbol) ** m))


# 定义一个三角函数重写器，处理满足特定条件的积分表达式
sincos_cosodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 - sin(b*symbol)**2)**((n - 1) / 2) *
                                    cos(b*symbol) *
                                    sin(a*symbol) ** m))




tansec_seceven_condition = uncurry(lambda a, b, m, n, i, s: n.is_even and n >= 4)


# 定义一个条件函数，判断参数 n 是否为偶数且大于等于 4
tansec_seceven_condition = uncurry(lambda a, b, m, n, i, s: n.is_even and n >= 4)




tansec_seceven = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 + tan(b*symbol)**2) ** (n/2 - 1) *
                                    sec(b*symbol)**2 *
                                    tan(a*symbol) ** m ))


# 定义一个三角函数重写器，处理满足特定条件的积分表达式
tansec_seceven = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 + tan(b*symbol)**2) ** (n/2 - 1) *
                                    sec(b*symbol)**2 *
                                    tan(a*symbol) ** m ))




tansec_tanodd_condition = uncurry(lambda a, b, m, n, i, s: m.is_odd)


# 定义一个条件函数，判断参数 m 是否为奇数
tansec_tanodd_condition = uncurry(lambda a, b, m, n, i, s: m.is_odd)




tansec_tanodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (sec(a*symbol)**2 - 1) ** ((m - 1) / 2) *
                                     tan(a*symbol) *
                                     sec(b*symbol) ** n ))


# 定义一个三角函数重写器，处理满足特定条件的积分表达式
tansec_tanodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (sec(a*symbol)**2 - 1) ** ((m - 1) / 2) *
                                     tan(a*symbol) *
                                     sec(b*symbol) ** n ))




tan_tansquared_condition = uncurry(lambda a, b, m, n, i, s: m == 2 and n == 0)


# 定义一个条件函数，判断参数 m 是否为 2 且参数 n 是否为 0
tan_tansquared_condition = uncurry(lambda a, b, m, n, i, s: m == 2 and n == 0)




tan_tansquared = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( sec(a*symbol)**2 - 1))


# 定义一个三角函数重写器，处理满足特定条件的积分表达式
tan_tansquared = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( sec(a*symbol)**2 - 1))




cotcsc_csceven_condition = uncurry(lambda a, b, m, n, i, s: n.is_even and n >= 4)


# 定义一个条件函数，判断参数 n 是否为偶数且大于等于 4
cotcsc_csceven_condition = uncurry(lambda a, b, m, n, i, s: n.is_even and n >= 4)




cotcsc_csceven = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 + cot(b*symbol)**2) ** (n/2 - 1) *
                                    csc(b*symbol)**2 *
                                    cot(a*symbol) ** m ))


# 定义一个三角函数重写器，处理满足特定条件的积分表达式
cotcsc_csceven = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (1 + cot(b*symbol)**2) ** (n/2 - 1) *
                                    csc(b*symbol)**2 *
                                    cot(a*symbol) ** m ))




cotcsc_cotodd_condition = uncurry(lambda a, b, m, n, i, s: m.is_odd)


# 定义一个条件函数，判断参数 m 是否为奇数
cotcsc_cotodd_condition = uncurry(lambda a, b, m, n, i, s: m.is_odd)




cotcsc_cotodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (csc(a*symbol)**2 - 1) ** ((m - 1) / 2) *
                                    cot(a*symbol) *
                                    csc(b*symbol) ** n ))


# 定义一个三角函数重写器，处理满足特定条件的积分表达式
cotcsc_cotodd = trig_rewriter(
    lambda a, b, m, n, i, symbol: ( (csc(a*symbol)**2 - 1) ** ((m - 1) / 2) *
                                    cot(a*symbol) *
                                    csc(b*symbol) ** n ))




def trig_sincos_rule(integral):
    integrand, symbol = integral

    if any(integrand.has(f) for f in (sin, cos)):
        pattern, a, b, m, n = sincos_pattern(symbol)
        match = integrand.match(pattern)
        if not match:
            return

        return multiplexer({
            sincos_botheven_condition: sincos_botheven,
            sincos_sinodd_condition: sincos_sinodd,
            sincos_cosodd_condition: sincos_cosodd
        })(tuple(
            [match.get(i, S.Zero) for i in (a, b, m, n)] +
            [integrand, symbol]))


# 定义一个函数 trig_sincos_rule，用于处理含有 sin 或 cos 的积分表达式
def trig_sincos_rule(integral):
    integrand, symbol = integral

    # 如果积分被函数 sin 或 cos 中的任意一个包含，则执行以下逻辑
    if any(integrand.has(f) for f in (sin, cos)):
        # 根据符号变量 symbol 获取 sincos_pattern
        pattern, a, b, m, n = sincos_pattern(symbol)
        # 尝试将积分表达式与 pattern
    # 检查是否存在函数 tan 或 sec 在被积函数 integrand 中
    if any(integrand.has(f) for f in (tan, sec)):
        # 获取符号 symbol 的 tansec 模式及相关参数
        pattern, a, b, m, n = tansec_pattern(symbol)
        # 尝试在被积函数 integrand 中匹配 tansec 模式
        match = integrand.match(pattern)
        # 如果匹配失败，则返回 None
        if not match:
            return

        # 返回一个函数，根据条件选择并应用不同的积分方法
        return multiplexer({
            tansec_tanodd_condition: tansec_tanodd,
            tansec_seceven_condition: tansec_seceven,
            tan_tansquared_condition: tan_tansquared
        })(tuple(
            # 构建参数元组，包含从 match 中获取的 a, b, m, n 的值，以及 integrand 和 symbol
            [match.get(i, S.Zero) for i in (a, b, m, n)] +
            [integrand, symbol]))
# 对三角函数的积分应用余割和余切的规则
def trig_cotcsc_rule(integral):
    # 解包积分表达式和符号
    integrand, symbol = integral
    # 将积分中的 1/sin 替换为 csc，1/tan 替换为 cot，cos/sin 替换为 cot
    integrand = integrand.subs({
        1 / sin(symbol): csc(symbol),
        1 / tan(symbol): cot(symbol),
        cos(symbol) / tan(symbol): cot(symbol)
    })

    # 如果积分中包含余切或余割函数
    if any(integrand.has(f) for f in (cot, csc)):
        # 获取余切和余割的模式以及相关参数
        pattern, a, b, m, n = cotcsc_pattern(symbol)
        # 尝试匹配积分表达式与模式
        match = integrand.match(pattern)
        # 如果匹配失败，则返回 None
        if not match:
            return

        # 根据匹配的结果，应用不同的条件函数处理积分
        return multiplexer({
            cotcsc_cotodd_condition: cotcsc_cotodd,
            cotcsc_csceven_condition: cotcsc_csceven
        })(tuple(
            # 提取匹配结果中的参数，如果没有则使用零
            [match.get(i, S.Zero) for i in (a, b, m, n)] +
            # 将积分表达式和符号也作为参数传递
            [integrand, symbol]))

# 对 sin(2*symbol) 的积分应用倍角公式
def trig_sindouble_rule(integral):
    # 解包积分表达式和符号
    integrand, symbol = integral
    # 定义一个通配符，排除 sin(2*symbol)，尝试匹配 sin(2*symbol)*a 的形式
    a = Wild('a', exclude=[sin(2*symbol)])
    match = integrand.match(sin(2*symbol)*a)
    # 如果匹配成功
    if match:
        # 计算 sin(2*symbol) 的倍角公式
        sin_double = 2*sin(symbol)*cos(symbol)/sin(2*symbol)
        # 返回将倍角公式应用到积分表达式的结果
        return integral_steps(integrand * sin_double, symbol)

# 对三角函数的幂次和乘积的积分应用规则
def trig_powers_products_rule(integral):
    # 对 trig_sincos_rule, trig_tansec_rule, trig_cotcsc_rule, trig_sindouble_rule 中的任何一个应用
    return do_one(null_safe(trig_sincos_rule),
                  null_safe(trig_tansec_rule),
                  null_safe(trig_cotcsc_rule),
                  null_safe(trig_sindouble_rule))(integral)

# 对三角函数的替换积分规则
def trig_substitution_rule(integral):
    # 解包积分表达式和符号
    integrand, symbol = integral
    # 定义通配符 A 和 B，排除 0 和 symbol，创建虚拟变量 theta
    A = Wild('a', exclude=[0, symbol])
    B = Wild('b', exclude=[0, symbol])
    theta = Dummy("theta")
    # 定义目标模式 A + B*symbol**2
    target_pattern = A + B*symbol**2

    # 在积分表达式中查找目标模式的所有匹配
    matches = integrand.find(target_pattern)
    # 对于每个匹配表达式，从目标模式中提取匹配项
    match = expr.match(target_pattern)
    # 获取匹配项中的变量 A 的值，如果没有则默认为零
    a = match.get(A, S.Zero)
    # 获取匹配项中的变量 B 的值，如果没有则默认为零
    b = match.get(B, S.Zero)

    # 判断 a 是否为正数或正数表达式
    a_positive = ((a.is_number and a > 0) or a.is_positive)
    # 判断 b 是否为正数或正数表达式
    b_positive = ((b.is_number and b > 0) or b.is_positive)
    # 判断 a 是否为负数或负数表达式
    a_negative = ((a.is_number and a < 0) or a.is_negative)
    # 判断 b 是否为负数或负数表达式
    b_negative = ((b.is_number and b < 0) or b.is_negative)
    
    # 初始化 x_func 变量
    x_func = None
    # 根据不同的条件分支进行处理
    if a_positive and b_positive:
        # 当 a > 0 且 b > 0 时，计算表达式 sqrt(a)/sqrt(b) * tan(theta)
        x_func = (sqrt(a)/sqrt(b)) * tan(theta)
        # 设置 domain 限制为 True，即不限制 x 的取值范围
        restriction = True
    elif a_positive and b_negative:
        # 当 a > 0 且 b < 0 时，计算表达式 sqrt(a)/sqrt(-b) * sin(theta)
        constant = sqrt(a)/sqrt(-b)
        x_func = constant * sin(theta)
        # 设置 domain 限制为 And(symbol > -constant, symbol < constant)
        restriction = And(symbol > -constant, symbol < constant)
    elif a_negative and b_positive:
        # 当 a < 0 且 b > 0 时，计算表达式 sqrt(-a)/sqrt(b) * sec(theta)
        constant = sqrt(-a)/sqrt(b)
        x_func = constant * sec(theta)
        # 设置 domain 限制为 And(symbol > -constant, symbol < constant)
        restriction = And(symbol > -constant, symbol < constant)
    
    # 如果 x_func 不为 None，则进行进一步处理
    if x_func:
        # 手动简化 sqrt(trig(theta)**2) 到 trig(theta)
        substitutions = {}
        # 遍历 sin, cos, tan, sec, csc, cot 函数，应用特定的替换规则
        for f in [sin, cos, tan,
                  sec, csc, cot]:
            substitutions[sqrt(f(theta)**2)] = f(theta)
            substitutions[sqrt(f(theta)**(-2))] = 1/f(theta)

        # 将 integrand 中的 symbol 替换为 x_func，并进行三角函数简化
        replaced = integrand.subs(symbol, x_func).trigsimp()
        # 使用手动替换函数进行额外的替换操作
        replaced = manual_subs(replaced, substitutions)
        
        # 如果替换后的表达式不再包含 symbol，则进行下一步处理
        if not replaced.has(symbol):
            # 乘以关于 theta 的导数
            replaced *= manual_diff(x_func, theta)
            # 再次进行三角函数简化
            replaced = replaced.trigsimp()
            # 查找并替换 sec(theta) 函数
            secants = replaced.find(1/cos(theta))
            if secants:
                replaced = replaced.xreplace({
                    1/cos(theta): sec(theta)
                })

            # 生成积分步骤对象 substep
            substep = integral_steps(replaced, theta)
            # 如果 substep 不包含未知项，则返回 TrigSubstitutionRule 对象
            if not substep.contains_dont_know():
                return TrigSubstitutionRule(integrand, symbol,
                    theta, x_func, replaced, substep, restriction)
# 定义 Heaviside 积分规则函数
def heaviside_rule(integral):
    # 解构积分表达式和符号
    integrand, symbol = integral
    # 调用 heaviside_pattern 函数获取模式、参数
    pattern, m, b, g = heaviside_pattern(symbol)
    # 尝试匹配积分表达式与模式
    match = integrand.match(pattern)
    # 如果匹配成功且 g 不为零，则执行以下操作
    if match and 0 != match[g]:
        # 调用 integral_steps 函数获取子步骤
        substep = integral_steps(match[g], symbol)
        # 提取 m 和 b 的值
        m, b = match[m], match[b]
        # 返回 HeavisideRule 对象
        return HeavisideRule(integrand, symbol, m*symbol + b, -b/m, substep)


# 定义 Dirac Delta 积分规则函数
def dirac_delta_rule(integral: IntegralInfo):
    # 解构积分表达式和自变量 x
    integrand, x = integral
    # 如果积分表达式参数个数为 1，则 n 赋值为零
    if len(integrand.args) == 1:
        n = S.Zero
    else:
        n = integrand.args[1]
    # 如果 n 不是整数或小于零，则返回空
    if not n.is_Integer or n < 0:
        return
    # 定义 Wildcards a 和 b，用于匹配积分表达式
    a, b = Wild('a', exclude=[x]), Wild('b', exclude=[x, 0])
    # 尝试匹配积分表达式的第一个参数 a + b*x
    match = integrand.args[0].match(a+b*x)
    # 如果匹配失败，则返回空
    if not match:
        return
    # 提取 a 和 b 的值
    a, b = match[a], match[b]
    # 定义通用条件 Ne(b, 0)
    generic_cond = Ne(b, 0)
    # 如果通用条件为真，则 degenerate_step 为 None
    if generic_cond is S.true:
        degenerate_step = None
    else:
        # 否则，使用 DiracDeltaRule 函数生成 generic_step
        degenerate_step = ConstantRule(DiracDelta(a, n), x)
    # 返回添加 degenerate_step 后的结果
    return _add_degenerate_step(generic_cond, generic_step, degenerate_step)


# 定义替换规则函数
def substitution_rule(integral):
    # 解构积分表达式和符号
    integrand, symbol = integral

    # 创建一个虚拟变量 u_var
    u_var = Dummy("u")
    # 查找适用于积分表达式和符号的替换
    substitutions = find_substitutions(integrand, symbol, u_var)
    # 计数器初始化为 0
    count = 0
    # 如果存在替换规则，则执行以下操作
    if substitutions:
        # 输出调试信息
        debug("List of Substitution Rules")
        # 初始化方式列表
        ways = []
        # 遍历每一种替换方式
        for u_func, c, substituted in substitutions:
            # 调用 integral_steps 函数获取替换后的子步骤
            subrule = integral_steps(substituted, u_var)
            # 计数器加一
            count = count + 1
            # 输出调试信息
            debug("Rule {}: {}".format(count, subrule))

            # 如果 subrule 包含不明确的部分，则继续下一次循环
            if subrule.contains_dont_know():
                continue

            # 如果简化后的常数项 c - 1 不为零
            if simplify(c - 1) != 0:
                # 分离 c 的分子和分母
                _, denom = c.as_numer_denom()
                # 如果分母包含自由符号
                if denom.free_symbols:
                    # 初始化 piecewise 列表和 could_be_zero 列表
                    piecewise = []
                    could_be_zero = []

                    # 如果 denom 是乘积表达式，则逐个处理其因子
                    if isinstance(denom, Mul):
                        could_be_zero = denom.args
                    else:
                        could_be_zero.append(denom)

                    # 遍历每一个可能为零的表达式
                    for expr in could_be_zero:
                        # 如果表达式可以为零，则执行手动替换后获取子步骤
                        if not fuzzy_not(expr.is_zero):
                            substep = integral_steps(manual_subs(integrand, expr, 0), symbol)

                            # 如果子步骤存在，则添加到 piecewise 列表
                            if substep:
                                piecewise.append((
                                    substep,
                                    Eq(expr, 0)
                                ))
                    # 将原始 subrule 添加到 piecewise 列表末尾
                    piecewise.append((subrule, True))
                    # 使用 PiecewiseRule 函数创建替换后的 subrule
                    subrule = PiecewiseRule(substituted, symbol, piecewise)

            # 将每个 u_func、c、subrule 组合为 URule 对象，并添加到 ways 列表
            ways.append(URule(integrand, symbol, u_var, u_func, subrule))

        # 如果 ways 列表长度大于 1，则返回 AlternativeRule 对象
        if len(ways) > 1:
            return AlternativeRule(integrand, symbol, ways)
        # 如果 ways 列表非空，则返回第一个 URule 对象
        elif ways:
            return ways[0]


# 定义部分分式分解规则函数
partial_fractions_rule = rewriter(
    lambda integrand, symbol: integrand.is_rational_function(),
    lambda integrand, symbol: integrand.apart(symbol))
cancel_rule = rewriter(
    # 规则函数1：始终返回True，取消对积分项的条件限制
    lambda integrand, symbol: True,
    # 规则函数2：取消积分项
    lambda integrand, symbol: integrand.cancel())

distribute_expand_rule = rewriter(
    # 规则函数1：如果积分项是幂次或乘积，则返回True
    lambda integrand, symbol: (
        isinstance(integrand, (Pow, Mul)) or all(arg.is_Pow or arg.is_polynomial(symbol) for arg in integrand.args)),
    # 规则函数2：展开积分项
    lambda integrand, symbol: integrand.expand())

trig_expand_rule = rewriter(
    # 如果存在不同参数的三角函数，则展开它们
    lambda integrand, symbol: (
        len({a.args[0] for a in integrand.atoms(TrigonometricFunction)}) > 1),
    # 规则函数2：展开三角函数积分项
    lambda integrand, symbol: integrand.expand(trig=True))

def derivative_rule(integral):
    integrand = integral[0]
    diff_variables = integrand.variables
    undifferentiated_function = integrand.expr
    integrand_variables = undifferentiated_function.free_symbols

    if integral.symbol in integrand_variables:
        if integral.symbol in diff_variables:
            return DerivativeRule(*integral)
        else:
            return DontKnowRule(integrand, integral.symbol)
    else:
        return ConstantRule(*integral)

def rewrites_rule(integral):
    integrand, symbol = integral

    if integrand.match(1/cos(symbol)):
        rewritten = integrand.subs(1/cos(symbol), sec(symbol))
        return RewriteRule(integrand, symbol, rewritten, integral_steps(rewritten, symbol))

def fallback_rule(integral):
    return DontKnowRule(*integral)

# 缓存用于打破循环积分。
# 需要在缓存表达式中使用相同的虚拟变量以进行匹配。
# 还记录积分部分的“u”，以避免无限重复。
_integral_cache: dict[Expr, Expr | None] = {}
_parts_u_cache: dict[Expr, int] = defaultdict(int)
_cache_dummy = Dummy("z")

def integral_steps(integrand, symbol, **options):
    """Returns the steps needed to compute an integral.

    Explanation
    ===========

    This function attempts to mirror what a student would do by hand as
    closely as possible.

    SymPy Gamma uses this to provide a step-by-step explanation of an
    integral. The code it uses to format the results of this function can be
    found at
    https://github.com/sympy/sympy_gamma/blob/master/app/logic/intsteps.py.

    Examples
    ========

    >>> from sympy import exp, sin
    >>> from sympy.integrals.manualintegrate import integral_steps
    >>> from sympy.abc import x
    >>> print(repr(integral_steps(exp(x) / (1 + exp(2 * x)), x))) \
    # doctest: +NORMALIZE_WHITESPACE
    URule(integrand=exp(x)/(exp(2*x) + 1), variable=x, u_var=_u, u_func=exp(x),
    substep=ArctanRule(integrand=1/(_u**2 + 1), variable=_u, a=1, b=1, c=1))
    >>> print(repr(integral_steps(sin(x), x))) \
    # doctest: +NORMALIZE_WHITESPACE
    SinRule(integrand=sin(x), variable=x)
    >>> print(repr(integral_steps((x**2 + 3)**2, x))) \
    # doctest: +NORMALIZE_WHITESPACE
    RewriteRule(integrand=(x**2 + 3)**2, variable=x, rewritten=x**4 + 6*x**2 + 9,
    substep=AddRule(integrand=x**4 + 6*x**2 + 9, variable=x,
    substeps=[PowerRule(integrand=x**4, variable=x, base=x, exp=4),
    ConstantTimesRule(integrand=6*x**2, variable=x, constant=6, other=x**2,
    substep=PowerRule(integrand=x**2, variable=x, base=x, exp=2)),
    ConstantRule(integrand=9, variable=x)]))


# 定义 RewriteRule 规则，处理 (x**2 + 3)**2 的积分重写为 x**4 + 6*x**2 + 9，
# 并包含一个 AddRule 子规则，以及其余的子规则列表。
RewriteRule(integrand=(x**2 + 3)**2, variable=x, rewritten=x**4 + 6*x**2 + 9,
            substep=AddRule(integrand=x**4 + 6*x**2 + 9, variable=x,
                            substeps=[PowerRule(integrand=x**4, variable=x, base=x, exp=4),
                                      ConstantTimesRule(integrand=6*x**2, variable=x, constant=6, other=x**2,
                                                        substep=PowerRule(integrand=x**2, variable=x, base=x, exp=2)),
                                      ConstantRule(integrand=9, variable=x)]))

Returns
=======

rule : Rule
    The first step; most rules have substeps that must also be
    considered. These substeps can be evaluated using ``manualintegrate``
    to obtain a result.
"""
cachekey = integrand.xreplace({symbol: _cache_dummy})
# 检查缓存中是否存在当前积分的关键字
if cachekey in _integral_cache:
    if _integral_cache[cachekey] is None:
        # 如果缓存中的值为 None，表示尝试积分会导致循环，返回 DontKnowRule
        return DontKnowRule(integrand, symbol)
    else:
        # TODO: 这是未来开发的标记，因为当前 _integral_cache 除了 None 没有其他值
        return (_integral_cache[cachekey].xreplace(_cache_dummy, symbol),
                symbol)
else:
    # 将缓存中当前积分的关键字设为 None
    _integral_cache[cachekey] = None

integral = IntegralInfo(integrand, symbol)

# 定义一个函数 key，用于确定积分的类型
def key(integral):
    integrand = integral.integrand

    if symbol not in integrand.free_symbols:
        return Number
    for cls in (Symbol, TrigonometricFunction, OrthogonalPolynomial):
        if isinstance(integrand, cls):
            return cls
    return type(integrand)

# 定义一个函数 integral_is_subclass，用于检查积分是否是指定类别的子类
def integral_is_subclass(*klasses):
    def _integral_is_subclass(integral):
        k = key(integral)
        return k and issubclass(k, klasses)
    return _integral_is_subclass
    result = do_one(  # 调用 do_one 函数，选择其中一个规则处理积分表达式
        null_safe(special_function_rule),  # 处理特殊函数的规则（如果存在）
        null_safe(switch(key, {  # 根据 key 进行分支选择
            Pow: do_one(null_safe(power_rule), null_safe(inverse_trig_rule),  # 如果 key 是 Pow，依次应用多个规则
                        null_safe(sqrt_linear_rule),
                        null_safe(quadratic_denom_rule)),
            Symbol: power_rule,  # 如果 key 是 Symbol，则应用 power_rule
            exp: exp_rule,  # 如果 key 是 exp，则应用 exp_rule
            Add: add_rule,  # 如果 key 是 Add，则应用 add_rule
            Mul: do_one(null_safe(mul_rule), null_safe(trig_product_rule),  # 如果 key 是 Mul，依次应用多个规则
                        null_safe(heaviside_rule), null_safe(quadratic_denom_rule),
                        null_safe(sqrt_linear_rule),
                        null_safe(sqrt_quadratic_rule)),
            Derivative: derivative_rule,  # 如果 key 是 Derivative，则应用 derivative_rule
            TrigonometricFunction: trig_rule,  # 如果 key 是 TrigonometricFunction，则应用 trig_rule
            Heaviside: heaviside_rule,  # 如果 key 是 Heaviside，则应用 heaviside_rule
            DiracDelta: dirac_delta_rule,  # 如果 key 是 DiracDelta，则应用 dirac_delta_rule
            OrthogonalPolynomial: orthogonal_poly_rule,  # 如果 key 是 OrthogonalPolynomial，则应用 orthogonal_poly_rule
            Number: constant_rule  # 如果 key 是 Number，则应用 constant_rule
        })),
        do_one(  # 第二个 do_one 调用，选择其中一个规则处理积分表达式
            null_safe(trig_rule),  # 处理三角函数的规则（如果存在）
            null_safe(hyperbolic_rule),  # 处理双曲函数的规则（如果存在）
            null_safe(alternatives(  # 处理多种替代规则的组合
                rewrites_rule,  # 重写规则
                substitution_rule,  # 替换规则
                condition(  # 条件规则：当积分是 Mul 类型和 Pow 类型的子类时
                    integral_is_subclass(Mul, Pow),
                    partial_fractions_rule),  # 部分分式分解规则
                condition(  # 条件规则：当积分是 Mul 类型和 Pow 类型的子类时
                    integral_is_subclass(Mul, Pow),
                    cancel_rule),  # 取消规则
                condition(  # 条件规则：当积分是 Mul 类型和 log 类型以及其逆三角函数的子类时
                    integral_is_subclass(Mul, log,
                    *inverse_trig_functions),
                    parts_rule),  # 分部积分规则
                condition(  # 条件规则：当积分是 Mul 类型和 Pow 类型的子类时
                    integral_is_subclass(Mul, Pow),
                    distribute_expand_rule),  # 分布展开规则
                trig_powers_products_rule,  # 处理三角函数幂次和乘积的规则
                trig_expand_rule  # 处理三角函数展开的规则
            )),
            null_safe(condition(integral_is_subclass(Mul, Pow), nested_pow_rule)),  # 处理嵌套幂次的规则（如果存在）
            null_safe(trig_substitution_rule)  # 处理三角函数替换的规则（如果存在）
        ),
        fallback_rule)(integral)  # 调用 fallback_rule 处理积分表达式作为最后的备用规则
    del _integral_cache[cachekey]  # 从缓存中删除已处理的积分表达式
    return result  # 返回处理后的积分表达式结果
# 定义一个函数，计算单变量的不定积分，采用类似学生手算的算法
def manualintegrate(f, var):
    """manualintegrate(f, var)

    Explanation
    ===========

    Compute indefinite integral of a single variable using an algorithm that
    resembles what a student would do by hand.

    Unlike :func:`~.integrate`, var can only be a single symbol.

    Examples
    ========

    >>> from sympy import sin, cos, tan, exp, log, integrate
    >>> from sympy.integrals.manualintegrate import manualintegrate
    >>> from sympy.abc import x
    >>> manualintegrate(1 / x, x)
    log(x)
    >>> integrate(1/x)
    log(x)
    >>> manualintegrate(log(x), x)
    x*log(x) - x
    >>> integrate(log(x))
    x*log(x) - x
    >>> manualintegrate(exp(x) / (1 + exp(2 * x)), x)
    atan(exp(x))
    >>> integrate(exp(x) / (1 + exp(2 * x)))
    RootSum(4*_z**2 + 1, Lambda(_i, _i*log(2*_i + exp(x))))
    >>> manualintegrate(cos(x)**4 * sin(x), x)
    -cos(x)**5/5
    >>> integrate(cos(x)**4 * sin(x), x)
    -cos(x)**5/5
    >>> manualintegrate(cos(x)**4 * sin(x)**3, x)
    cos(x)**7/7 - cos(x)**5/5
    >>> integrate(cos(x)**4 * sin(x)**3, x)
    cos(x)**7/7 - cos(x)**5/5
    >>> manualintegrate(tan(x), x)
    -log(cos(x))
    >>> integrate(tan(x), x)
    -log(cos(x))

    See Also
    ========

    sympy.integrals.integrals.integrate
    sympy.integrals.integrals.Integral.doit
    sympy.integrals.integrals.Integral
    """
    # 调用 integral_steps 函数计算积分步骤，并返回结果
    result = integral_steps(f, var).eval()
    # 清空 u-parts 的缓存
    _parts_u_cache.clear()
    # 如果结果是带有两个部分的 Piecewise 对象，则重新排序确保泛化部分在前
    if isinstance(result, Piecewise) and len(result.args) == 2:
        cond = result.args[0][1]
        # 如果条件是等式类型，并且第二部分是真值
        if isinstance(cond, Eq) and result.args[1][1] == True:
            # 重新构造 Piecewise 对象，确保泛化条件在前
            result = result.func(
                (result.args[1][0], Ne(*cond.args)),
                (result.args[0][0], True))
    # 返回计算得到的积分结果
    return result
```