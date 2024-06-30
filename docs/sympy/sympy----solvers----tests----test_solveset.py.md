# `D:\src\scipysrc\sympy\sympy\solvers\tests\test_solveset.py`

```
# 导入数学库中的 isclose 函数
from math import isclose

# 从 sympy.calculus.util 模块中导入 stationary_points 函数
from sympy.calculus.util import stationary_points

# 从 sympy.core.containers 模块中导入 Tuple 类
from sympy.core.containers import Tuple

# 从 sympy.core.function 模块中导入多个类和函数
from sympy.core.function import (Function, Lambda, nfloat, diff)

# 从 sympy.core.mod 模块中导入 Mod 类
from sympy.core.mod import Mod

# 从 sympy.core.numbers 模块中导入多个常数和函数
from sympy.core.numbers import (E, I, Rational, oo, pi, Integer, all_close)

# 从 sympy.core.relational 模块中导入多个关系运算符
from sympy.core.relational import (Eq, Gt, Ne, Ge)

# 从 sympy.core.singleton 模块中导入 S 常数
from sympy.core.singleton import S

# 从 sympy.core.sorting 模块中导入 ordered 函数
from sympy.core.sorting import ordered

# 从 sympy.core.symbol 模块中导入 Dummy, Symbol, symbols 函数
from sympy.core.symbol import (Dummy, Symbol, symbols)

# 从 sympy.core.sympify 模块中导入 sympify 函数
from sympy.core.sympify import sympify

# 从 sympy.functions.elementary.complexes 模块中导入多个复数函数
from sympy.functions.elementary.complexes import (Abs, arg, im, re, sign, conjugate)

# 从 sympy.functions.elementary.exponential 模块中导入多个指数函数
from sympy.functions.elementary.exponential import (LambertW, exp, log)

# 从 sympy.functions.elementary.hyperbolic 模块中导入多个双曲函数
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction,
    sinh, cosh, tanh, coth, sech, csch, asinh, acosh, atanh, acoth, asech, acsch)

# 从 sympy.functions.elementary.miscellaneous 模块中导入 sqrt, Min, Max 函数
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max

# 从 sympy.functions.elementary.piecewise 模块中导入 Piecewise 函数
from sympy.functions.elementary.piecewise import Piecewise

# 从 sympy.functions.elementary.trigonometric 模块中导入多个三角函数
from sympy.functions.elementary.trigonometric import (
    TrigonometricFunction, acos, acot, acsc, asec, asin, atan, atan2,
    cos, cot, csc, sec, sin, tan)

# 从 sympy.functions.special.error_functions 模块中导入多个误差函数
from sympy.functions.special.error_functions import (erf, erfc,
    erfcinv, erfinv)

# 从 sympy.logic.boolalg 模块中导入 And 函数
from sympy.logic.boolalg import And

# 从 sympy.matrices.dense 模块中导入 Matrix 类
from sympy.matrices.dense import MutableDenseMatrix as Matrix

# 从 sympy.matrices.immutable 模块中导入 ImmutableDenseMatrix 类
from sympy.matrices.immutable import ImmutableDenseMatrix

# 从 sympy.polys.polytools 模块中导入 Poly 类
from sympy.polys.polytools import Poly

# 从 sympy.polys.rootoftools 模块中导入 CRootOf 类
from sympy.polys.rootoftools import CRootOf

# 从 sympy.sets.contains 模块中导入 Contains 类
from sympy.sets.contains import Contains

# 从 sympy.sets.conditionset 模块中导入 ConditionSet 类
from sympy.sets.conditionset import ConditionSet

# 从 sympy.sets.fancysets 模块中导入 ImageSet, Range 类
from sympy.sets.fancysets import ImageSet, Range

# 从 sympy.sets.sets 模块中导入多个集合操作类和函数
from sympy.sets.sets import (Complement, FiniteSet,
    Intersection, Interval, Union, imageset, ProductSet)

# 从 sympy.simplify 模块中导入 simplify 函数
from sympy.simplify import simplify

# 从 sympy.tensor.indexed 模块中导入 Indexed 类
from sympy.tensor.indexed import Indexed

# 从 sympy.utilities.iterables 模块中导入 numbered_symbols 函数
from sympy.utilities.iterables import numbered_symbols

# 从 sympy.testing.pytest 模块中导入多个测试相关函数和装饰器
from sympy.testing.pytest import (XFAIL, raises, skip, slow, SKIP, _both_exp_pow)

# 从 sympy.core.random 模块中导入 verify_numerically 函数并重命名为 tn
from sympy.core.random import verify_numerically as tn

# 从 sympy.physics.units 模块中导入 cm 单位
from sympy.physics.units import cm

# 从 sympy.solvers 模块中导入 solve 函数
from sympy.solvers import solve

# 从 sympy.solvers.solveset 模块中导入多个解集相关函数
from sympy.solvers.solveset import (
    solveset_real, domain_check, solveset_complex, linear_eq_to_matrix,
    linsolve, _is_function_class_equation, invert_real, invert_complex,
    _invert_trig_hyp_real, solveset, solve_decomposition, substitution,
    nonlinsolve, solvify,
    _is_finite_with_finite_vars, _transolve, _is_exponential,
    _solve_exponential, _is_logarithmic, _is_lambert,
    _solve_logarithm, _term_factors, _is_modular, NonlinearError)

# 从 sympy.abc 模块中导入多个变量
from sympy.abc import (a, b, c, d, e, f, g, h, i, j, k, l, m, n, q, r,
    t, w, x, y, z)


def dumeq(i, j):
    # 如果 i 是 list 或 tuple 类型，则递归比较对应元素是否相等
    if type(i) in (list, tuple):
        return all(dumeq(i, j) for i, j in zip(i, j))
    # 否则比较 i 和 j 是否相等，或者它们是否满足符号相等条件
    return i == j or i.dummy_eq(j)


def assert_close_ss(sol1, sol2):
    """测试 solveset 返回的浮点数解是否接近"""
    # 将 sol1 和 sol2 转换为 sympy 表达式
    sol1 = sympify(sol1)
    sol2 = sympify(sol2)
    # 断言 sol1 和 sol2 是 FiniteSet 类型的对象
    assert isinstance(sol1, FiniteSet)
    assert isinstance(sol2, FiniteSet)
    # 断言 sol1 和 sol2 的长度相等
    assert len(sol1) == len(sol2)
    # 断言语句，用于检查两个可迭代对象 sol1 和 sol2 中的每对元素是否满足近似相等的条件
    assert all(isclose(v1, v2) for v1, v2 in zip(sol1, sol2))
# 定义一个函数，用于检验通过非线性求解得到的浮点数解是否接近
def assert_close_nl(sol1, sol2):
    """Test solutions with floats from nonlinsolve are close"""
    # 将输入的解析表达式转换为符号表达式
    sol1 = sympify(sol1)
    sol2 = sympify(sol2)
    # 确保sol1和sol2都是FiniteSet类型的对象
    assert isinstance(sol1, FiniteSet)
    assert isinstance(sol2, FiniteSet)
    # 检查sol1和sol2中的解的数量是否相等
    assert len(sol1) == len(sol2)
    # 逐个比较sol1和sol2中对应位置的解是否接近
    for s1, s2 in zip(sol1, sol2):
        assert len(s1) == len(s2)
        # 使用isclose函数检查两个浮点数是否接近
        assert all(isclose(v1, v2) for v1, v2 in zip(s1, s2))


# 装饰器函数，用于测试invert_real函数的行为
@_both_exp_pow
def test_invert_real():
    # 声明一个实数符号变量x
    x = Symbol('x', real=True)

    # 内部函数ireal，用于计算两个集合的交集
    def ireal(x, s=S.Reals):
        return Intersection(s, x)

    # 测试invert_real函数对exp(x)的反函数计算是否正确
    assert invert_real(exp(x), z, x) == (x, ireal(FiniteSet(log(z))))

    # 声明一个正数符号变量y和一个实数符号变量n
    y = Symbol('y', positive=True)
    n = Symbol('n', real=True)
    # 测试invert_real函数对x + 3的反函数计算是否正确
    assert invert_real(x + 3, y, x) == (x, FiniteSet(y - 3))
    # 测试invert_real函数对x*3的反函数计算是否正确
    assert invert_real(x*3, y, x) == (x, FiniteSet(y / 3))

    # 测试invert_real函数对exp(x)的反函数计算是否正确
    assert invert_real(exp(x), y, x) == (x, FiniteSet(log(y)))
    # 测试invert_real函数对exp(3*x)的反函数计算是否正确
    assert invert_real(exp(3*x), y, x) == (x, FiniteSet(log(y) / 3))
    # 测试invert_real函数对exp(x + 3)的反函数计算是否正确
    assert invert_real(exp(x + 3), y, x) == (x, FiniteSet(log(y) - 3))

    # 测试invert_real函数对exp(x) + 3的反函数计算是否正确
    assert invert_real(exp(x) + 3, y, x) == (x, ireal(FiniteSet(log(y - 3))))
    # 测试invert_real函数对exp(x)*3的反函数计算是否正确
    assert invert_real(exp(x)*3, y, x) == (x, FiniteSet(log(y / 3)))

    # 测试invert_real函数对log(x)的反函数计算是否正确
    assert invert_real(log(x), y, x) == (x, FiniteSet(exp(y)))
    # 测试invert_real函数对log(3*x)的反函数计算是否正确
    assert invert_real(log(3*x), y, x) == (x, FiniteSet(exp(y) / 3))
    # 测试invert_real函数对log(x + 3)的反函数计算是否正确
    assert invert_real(log(x + 3), y, x) == (x, FiniteSet(exp(y) - 3))

    # 测试invert_real函数对Abs(x)的反函数计算是否正确
    assert invert_real(Abs(x), y, x) == (x, FiniteSet(y, -y))

    # 测试invert_real函数对2**x的反函数计算是否正确
    assert invert_real(2**x, y, x) == (x, FiniteSet(log(y)/log(2)))
    # 测试invert_real函数对2**exp(x)的反函数计算是否正确
    assert invert_real(2**exp(x), y, x) == (x, ireal(FiniteSet(log(log(y)/log(2)))))

    # 测试invert_real函数对x**2的反函数计算是否正确
    assert invert_real(x**2, y, x) == (x, FiniteSet(sqrt(y), -sqrt(y)))
    # 测试invert_real函数对x**S.Half的反函数计算是否正确
    assert invert_real(x**S.Half, y, x) == (x, FiniteSet(y**2))

    # 测试invert_real函数对x的反函数计算是否引发异常
    raises(ValueError, lambda: invert_real(x, x, x))

    # 测试invert_real函数对x**pi的反函数计算是否正确
    assert invert_real(x**pi, y, x) == (x, FiniteSet(y**(1/pi)))
    # 测试invert_real函数对x**pi的反函数计算是否得到空集
    assert invert_real(x**pi, -E, x) == (x, S.EmptySet)
    # 测试invert_real函数对x**(3/2)的反函数计算是否正确
    assert invert_real(x**Rational(3/2), 1000, x) == (x, FiniteSet(100))
    # 测试invert_real函数对x**1.0的反函数计算是否正确
    assert invert_real(x**1.0, 1, x) == (x**1.0, FiniteSet(1))

    # 测试invert_real函数对S.One的反函数计算是否引发异常
    raises(ValueError, lambda: invert_real(S.One, y, x))

    # 测试invert_real函数对x**31 + x的反函数计算是否正确
    assert invert_real(x**31 + x, y, x) == (x**31 + x, FiniteSet(y))

    # 将表达式lhs赋给变量lhs
    lhs = x**31 + x
    # 声明一个基础值集合base_values
    base_values =  FiniteSet(y - 1, -y - 1)
    # 测试invert_real函数对Abs(x**31 + x + 1)的反函数计算是否正确
    assert invert_real(Abs(x**31 + x + 1), y, x) == (lhs, base_values)

    # 测试invert_real函数对sin(x)的反函数计算是否符合指定条件
    assert dumeq(invert_real(sin(x), y, x), (x,
        ConditionSet(x, (S(-1) <= y) & (y <= S(1)), Union(
            ImageSet(Lambda(n, 2*n*pi + asin(y)), S.Integers),
            ImageSet(Lambda(n, pi*2*n + pi - asin(y)), S.Integers)))))

    # 测试invert_real函数对sin(exp(x))的反函数计算是否符合指定条件
    assert dumeq(invert_real(sin(exp(x)), y, x), (x,
        ConditionSet(x, (S(-1) <= y) & (y <= S(1)), Union(
            ImageSet(Lambda(n, log(2*n*pi + asin(y))), S.Integers),
            ImageSet(Lambda(n, log(pi*2*n + pi - asin(y))), S.Integers)))))
    # 断言语句，验证 invert_real 函数对 csc(x) 的逆运算结果是否与期望值相等
    assert dumeq(invert_real(csc(x), y, x), (x,
        # 确定条件集合，限制 x 的取值范围在特定条件下
        ConditionSet(x, ((S(1) <= y) & (y < oo)) | ((-oo < y) & (y <= S(-1))),
            # 合并两个图像集，表示 csc(y) 的可能取值
            Union(ImageSet(Lambda(n, 2*n*pi + acsc(y)), S.Integers),
                ImageSet(Lambda(n, 2*n*pi - acsc(y) + pi), S.Integers)))))

    # 断言语句，验证 invert_real 函数对 csc(exp(x)) 的逆运算结果是否与期望值相等
    assert dumeq(invert_real(csc(exp(x)), y, x), (x,
        # 确定条件集合，限制 x 的取值范围在特定条件下
        ConditionSet(x, ((S(1) <= y) & (y < oo)) | ((-oo < y) & (y <= S(-1))),
            # 合并两个图像集，表示 csc(exp(y)) 的可能取值
            Union(ImageSet(Lambda(n, log(2*n*pi + acsc(y))), S.Integers),
                ImageSet(Lambda(n, log(2*n*pi - acsc(y) + pi)), S.Integers)))))

    # 断言语句，验证 invert_real 函数对 cos(x) 的逆运算结果是否与期望值相等
    assert dumeq(invert_real(cos(x), y, x), (x,
        # 确定条件集合，限制 x 的取值范围在特定条件下
        ConditionSet(x, (S(-1) <= y) & (y <= S(1)), Union(
            # 合并两个图像集，表示 cos(y) 的可能取值
            ImageSet(Lambda(n, 2*n*pi + acos(y)), S.Integers),
            ImageSet(Lambda(n, 2*n*pi - acos(y)), S.Integers)))))

    # 断言语句，验证 invert_real 函数对 cos(exp(x)) 的逆运算结果是否与期望值相等
    assert dumeq(invert_real(cos(exp(x)), y, x), (x,
        # 确定条件集合，限制 x 的取值范围在特定条件下
        ConditionSet(x, (S(-1) <= y) & (y <= S(1)), Union(
            # 合并两个图像集，表示 cos(exp(y)) 的可能取值
            ImageSet(Lambda(n, log(2*n*pi + acos(y))), S.Integers),
            ImageSet(Lambda(n, log(2*n*pi - acos(y))), S.Integers)))))

    # 断言语句，验证 invert_real 函数对 sec(x) 的逆运算结果是否与期望值相等
    assert dumeq(invert_real(sec(x), y, x), (x,
        # 确定条件集合，限制 x 的取值范围在特定条件下
        ConditionSet(x, ((S(1) <= y) & (y < oo)) | ((-oo < y) & (y <= S(-1))),
            # 合并两个图像集，表示 sec(y) 的可能取值
            Union(ImageSet(Lambda(n, 2*n*pi + asec(y)), S.Integers),
                ImageSet(Lambda(n, 2*n*pi - asec(y)), S.Integers)))))

    # 断言语句，验证 invert_real 函数对 sec(exp(x)) 的逆运算结果是否与期望值相等
    assert dumeq(invert_real(sec(exp(x)), y, x), (x,
        # 确定条件集合，限制 x 的取值范围在特定条件下
        ConditionSet(x, ((S(1) <= y) & (y < oo)) | ((-oo < y) & (y <= S(-1))),
            # 合并两个图像集，表示 sec(exp(y)) 的可能取值
            Union(ImageSet(Lambda(n, log(2*n*pi - asec(y))), S.Integers),
                ImageSet(Lambda(n, log(2*n*pi + asec(y))), S.Integers)))))

    # 断言语句，验证 invert_real 函数对 tan(x) 的逆运算结果是否与期望值相等
    assert dumeq(invert_real(tan(x), y, x), (x,
        # 确定条件集合，限制 x 的取值范围在特定条件下
        ConditionSet(x, (-oo < y) & (y < oo),
            # 表示 tan(y) 的可能取值
            ImageSet(Lambda(n, n*pi + atan(y)), S.Integers))))

    # 断言语句，验证 invert_real 函数对 tan(exp(x)) 的逆运算结果是否与期望值相等
    assert dumeq(invert_real(tan(exp(x)), y, x), (x,
        # 确定条件集合，限制 x 的取值范围在特定条件下
        ConditionSet(x, (-oo < y) & (y < oo),
            # 表示 tan(exp(y)) 的可能取值
            ImageSet(Lambda(n, log(n*pi + atan(y))), S.Integers))))

    # 断言语句，验证 invert_real 函数对 cot(x) 的逆运算结果是否与期望值相等
    assert dumeq(invert_real(cot(x), y, x), (x,
        # 确定条件集合，限制 x 的取值范围在特定条件下
        ConditionSet(x, (-oo < y) & (y < oo),
            # 表示 cot(y) 的可能取值
            ImageSet(Lambda(n, n*pi + acot(y)), S.Integers))))

    # 断言语句，验证 invert_real 函数对 cot(exp(x)) 的逆运算结果是否与期望值相等
    assert dumeq(invert_real(cot(exp(x)), y, x), (x,
        # 确定条件集合，限制 x 的取值范围在特定条件下
        ConditionSet(x, (-oo < y) & (y < oo),
            # 表示 cot(exp(y)) 的可能取值
            ImageSet(Lambda(n, log(n*pi + acot(y))), S.Integers))))

    # 断言语句，验证 invert_real 函数对 tan(tan(x)) 的逆运算结果是否与期望值相等
    assert dumeq(invert_real(tan(tan(x)), y, x),
        (x, ConditionSet(x, Eq(tan(tan(x)), y), S.Reals)))
        # 这里的注释是与先前结果略有不同，即 (tan(x), imageset(Lambda(n, n*pi + atan(y)), S.Integers)))

    # 定义一个正数符号 x
    x = Symbol('x', positive=True)
    # 断言语句，验证 invert_real 函数对 x**pi 的逆运算结果是否与期望值相等
    assert invert_real(x**pi, y, x) == (x, FiniteSet(y**(1/pi)))

    # 定义一个实数符号 r
    r = Symbol('r', real=True)
    # 定义一个正数符号 p
    p = Symbol('p', positive=True)
    # 断言语句，验证 invert_real 函数对 sinh(x) 的逆运算结果是否与期望值相等
    assert invert_real(sinh(x), r, x) == (x, FiniteSet(asinh(r)))
    # 断言语句，验证 invert_real 函数对 sinh(log(x)) 的逆运算结果是否与期望值相等
    assert invert_real(sinh(log(x)), p, x) == (x, FiniteSet(exp(asinh(p))))

    # 断言语句，验证 invert_real 函数对 cosh(x) 的逆运算结果是否与期望值相等
    assert invert_real(cosh(x), r, x) == (x, Intersection(
        # 表示 cosh(r) 的可能取值，交集限制在实数范围内
        FiniteSet(-acosh(r), acosh(r)), S.Reals))
    # 断言：计算反函数 invert_real(cosh(x), p + 1, x) 的结果是否符合预期
    assert invert_real(cosh(x), p + 1, x) == (x,
        FiniteSet(-acosh(p + 1), acosh(p + 1)))

    # 断言：计算反函数 invert_real(tanh(x), r, x) 的结果是否符合预期
    assert invert_real(tanh(x), r, x) == (x, Intersection(FiniteSet(atanh(r)), S.Reals))

    # 断言：计算反函数 invert_real(coth(x), p+1, x) 的结果是否符合预期
    assert invert_real(coth(x), p+1, x) == (x, FiniteSet(acoth(p+1)))

    # 断言：计算反函数 invert_real(sech(x), r, x) 的结果是否符合预期
    assert invert_real(sech(x), r, x) == (x, Intersection(
        FiniteSet(-asech(r), asech(r)), S.Reals))

    # 断言：计算反函数 invert_real(csch(x), p, x) 的结果是否符合预期
    assert invert_real(csch(x), p, x) == (x, FiniteSet(acsch(p)))

    # 断言：计算复合函数 invert_real(tanh(sin(x)), r, x) 的结果是否符合预期
    assert dumeq(invert_real(tanh(sin(x)), r, x), (x,
        ConditionSet(x, (S(-1) <= atanh(r)) & (atanh(r) <= S(1)), Union(
            ImageSet(Lambda(n, 2*n*pi + asin(atanh(r))), S.Integers),
            ImageSet(Lambda(n, 2*n*pi - asin(atanh(r)) + pi), S.Integers)))))
def test_invert_trig_hyp_real():
    # 检查一些不容易通过其他方式到达的代码路径
    n = Dummy('n')
    # 断言反三角和双曲反函数在指定范围内的计算结果是否正确
    assert _invert_trig_hyp_real(cosh(x), Range(-5, 10, 1), x)[1].dummy_eq(Union(
        ImageSet(Lambda(n, -acosh(n)), Range(1, 10, 1)),
        ImageSet(Lambda(n, acosh(n)), Range(1, 10, 1))))
    # 断言反三角和双曲反函数在指定区间内的计算结果是否正确
    assert _invert_trig_hyp_real(coth(x), Interval(-3, 2), x) == (x, Union(
        Interval(-oo, -acoth(3)), Interval(acoth(2), oo)))
    # 断言反三角和双曲反函数在指定区间内的计算结果是否正确
    assert _invert_trig_hyp_real(tanh(x), Interval(-S.Half, 1), x) == (x,
        Interval(-atanh(S.Half), oo))
    # 断言反三角和双曲反函数在指定图像集和域的计算结果是否正确
    assert _invert_trig_hyp_real(sech(x), imageset(n, S.Half + n/3, S.Naturals0), x) == \
        (x, FiniteSet(-asech(S(1)/2), asech(S(1)/2), -asech(S(5)/6), asech(S(5)/6)))
    # 断言反三角和双曲反函数在实数集合中的计算结果是否正确
    assert _invert_trig_hyp_real(csch(x), S.Reals, x) == (x,
        Union(Interval.open(-oo, 0), Interval.open(0, oo)))


def test_invert_complex():
    # 断言复数函数的反函数计算结果是否正确
    assert invert_complex(x + 3, y, x) == (x, FiniteSet(y - 3))
    # 断言复数函数的反函数计算结果是否正确
    assert invert_complex(x*3, y, x) == (x, FiniteSet(y / 3))
    # 断言复数函数的反函数计算结果是否正确
    assert invert_complex((x - 1)**3, 0, x) == (x, FiniteSet(1))

    # 断言复数函数的反函数计算结果是否正确
    assert dumeq(invert_complex(exp(x), y, x),
        (x, imageset(Lambda(n, I*(2*pi*n + arg(y)) + log(Abs(y))), S.Integers)))

    # 断言复数函数的反函数计算结果是否正确
    assert invert_complex(log(x), y, x) == (x, FiniteSet(exp(y)))

    # 断言复数函数的反函数在特定条件下会引发值错误
    raises(ValueError, lambda: invert_real(1, y, x))
    raises(ValueError, lambda: invert_complex(x, x, x))
    raises(ValueError, lambda: invert_complex(x, x, 1))

    # 断言复数函数的反函数计算结果是否正确
    assert dumeq(invert_complex(sin(x), I, x), (x, Union(
        ImageSet(Lambda(n, 2*n*pi + I*log(1 + sqrt(2))), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + pi - I*log(1 + sqrt(2))), S.Integers))))
    # 断言复数函数的反函数计算结果是否正确
    assert dumeq(invert_complex(cos(x), 1+I, x), (x, Union(
        ImageSet(Lambda(n, 2*n*pi - acos(1 + I)), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + acos(1 + I)), S.Integers))))
    # 断言复数函数的反函数计算结果是否正确
    assert dumeq(invert_complex(tan(2*x), 1, x), (x,
        ImageSet(Lambda(n, n*pi/2 + pi/8), S.Integers)))
    # 断言复数函数的反函数计算结果是否正确
    assert dumeq(invert_complex(cot(x), 2*I, x), (x,
        ImageSet(Lambda(n, n*pi - I*acoth(2)), S.Integers)))

    # 断言复数函数的反函数计算结果是否正确
    assert dumeq(invert_complex(sinh(x), 0, x), (x, Union(
        ImageSet(Lambda(n, 2*n*I*pi), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi + I*pi), S.Integers))))
    # 断言复数函数的反函数计算结果是否正确
    assert dumeq(invert_complex(cosh(x), 0, x), (x, Union(
        ImageSet(Lambda(n, 2*n*I*pi + I*pi/2), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi + 3*I*pi/2), S.Integers))))
    # 断言复数函数的反函数计算结果是否正确
    assert invert_complex(tanh(x), 1, x) == (x, S.EmptySet)
    # 断言复数函数的反函数计算结果是否正确
    assert dumeq(invert_complex(tanh(x), a, x), (x,
        ConditionSet(x, Ne(a, -1) & Ne(a, 1),
        ImageSet(Lambda(n, n*I*pi + atanh(a)), S.Integers))))
    # 断言复数函数的反函数计算结果是否正确
    assert invert_complex(coth(x), 1, x) == (x, S.EmptySet)
    # 断言复数函数的反函数计算结果是否正确
    assert dumeq(invert_complex(coth(x), a, x), (x,
        ConditionSet(x, Ne(a, -1) & Ne(a, 1),
        ImageSet(Lambda(n, n*I*pi + acoth(a)), S.Integers))))
    # 使用断言来验证 invert_complex(sech(x), 2, x) 函数的输出是否等于给定的值。
    assert dumeq(invert_complex(sech(x), 2, x), (x, Union(
        # 第一个可能的输出是 Lambda 表达式生成的整数集合的图像集，每个元素为 2*n*I*pi + I*pi/3
        ImageSet(Lambda(n, 2*n*I*pi + I*pi/3), S.Integers),
        # 第二个可能的输出是 Lambda 表达式生成的整数集合的图像集，每个元素为 2*n*I*pi + 5*I*pi/3
        ImageSet(Lambda(n, 2*n*I*pi + 5*I*pi/3), S.Integers))))
# 测试函数，用于检查 domain_check 函数的各种情况
def test_domain_check():
    # 检查当 x 不满足条件时，domain_check 返回 False
    assert domain_check(1/(1 + (1/(x+1))**2), x, -1) is False
    # 检查当 x 满足条件时，domain_check 返回 True
    assert domain_check(x**2, x, 0) is True
    # 检查当 x 为无穷大时，domain_check 返回 False
    assert domain_check(x, x, oo) is False
    # 检查当函数为常数 0 时，domain_check 返回 False
    assert domain_check(0, x, oo) is False


# 测试函数，用于检查 solveset 函数在解决特定问题时的返回结果
def test_issue_11536():
    # 解决方程 0**x - 100 = 0，在实数范围内无解
    assert solveset(0**x - 100, x, S.Reals) == S.EmptySet
    # 解决方程 0**x - 1 = 0，在实数范围内的解为 {0}
    assert solveset(0**x - 1, x, S.Reals) == FiniteSet(0)


# 测试函数，用于检查 solveset 函数在解决特定问题时的返回结果
def test_issue_17479():
    # 定义复杂的方程 f，并求取其对 x, y, z 的偏导数
    f = (x**2 + y**2)**2 + (x**2 + z**2)**2 - 2*(2*x**2 + y**2 + z**2)
    fx = f.diff(x)
    fy = f.diff(y)
    fz = f.diff(z)
    # 解决非线性方程组 [fx, fy, fz]，返回一组解 sol
    sol = nonlinsolve([fx, fy, fz], [x, y, z])
    # 断言 sol 的解的数量在 4 到 20 之间
    assert len(sol) >= 4 and len(sol) <= 20
    # 提示信息：nonlinsolve 可能会因内部变化而返回不同数量的解
    # （最初是 18，后来是 20，现在是 19）。不是所有解都有效，有些是冗余的。
    # 原始问题是引发了异常，第一个测试只检查 nonlinsolve 返回一个“合理”的解集。
    # 下一个测试检查结果的正确性。


# 标记为 XFAIL 的测试函数，用于检查 solveset 在解决特定问题时的返回结果
@XFAIL
def test_issue_18449():
    x, y, z = symbols("x, y, z")
    # 定义复杂的方程 f，并求取其对 x, y, z 的偏导数
    f = (x**2 + y**2)**2 + (x**2 + z**2)**2 - 2*(2*x**2 + y**2 + z**2)
    fx = diff(f, x)
    fy = diff(f, y)
    fz = diff(f, z)
    # 解决非线性方程组 [fx, fy, fz]，返回一组解 sol
    sol = nonlinsolve([fx, fy, fz], [x, y, z])
    # 对于 sol 中的每组解 (xs, ys, zs)，将其代入方程的结果应为 (0, 0, 0)
    for (xs, ys, zs) in sol:
        d = {x: xs, y: ys, z: zs}
        assert tuple(_.subs(d).simplify() for _ in (fx, fy, fz)) == (0, 0, 0)
    # 简化并去除重复元素后，sol 应该只剩下 4 个参数化解
    # simplifiedsolutions = FiniteSet((sqrt(1 - z**2), z, z),
    #                                 (-sqrt(1 - z**2), z, z),
    #                                 (sqrt(1 - z**2), -z, z),
    #                                 (-sqrt(1 - z**2), -z, z))
    # TODO: 上述解集是否确实完整？


# 测试函数，用于检查 solveset 函数在解决特定问题时的返回结果
def test_issue_21047():
    # 定义复杂的方程 f，并求解其在实数范围内的解
    f = (2 - x)**2 + (sqrt(x - 1) - 1)**6
    assert solveset(f, x, S.Reals) == FiniteSet(2)

    # 定义复杂的方程 f，并求解其在实数范围内的解
    f = (sqrt(x)-1)**2 + (sqrt(x)+1)**2 -2*x**2 + sqrt(2)
    assert solveset(f, x, S.Reals) == FiniteSet(
        S.Half - sqrt(2*sqrt(2) + 5)/2, S.Half + sqrt(2*sqrt(2) + 5)/2)


# 测试函数，用于检查 _is_function_class_equation 函数在特定情况下的返回结果
def test_is_function_class_equation():
    # 检查是否满足三角函数类方程 tan(x) == x
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x), x) is True
    # 检查是否满足三角函数类方程 tan(x) - 1 == x
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x) - 1, x) is True
    # 检查是否满足三角函数类方程 tan(x) + sin(x) == x
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x) + sin(x), x) is True
    # 检查是否满足三角函数类方程 tan(x) + sin(x) - a == x
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x) + sin(x) - a, x) is True
    # 检查是否满足三角函数类方程 sin(x)*tan(x) + sin(x) == x
    assert _is_function_class_equation(TrigonometricFunction,
                                       sin(x)*tan(x) + sin(x), x) is True
    # 检查是否满足三角函数类方程 sin(x)*tan(x + a) + sin(x) == x
    assert _is_function_class_equation(TrigonometricFunction,
                                       sin(x)*tan(x + a) + sin(x), x) is True
    # 断言：检查 _is_function_class_equation 函数对于给定的三角函数类方程是否返回 True
    assert _is_function_class_equation(TrigonometricFunction,
                                       sin(x)*tan(x*a) + sin(x), x) is True
    # 断言：检查 _is_function_class_equation 函数对于给定的三角函数类方程是否返回 True
    assert _is_function_class_equation(TrigonometricFunction,
                                       a*tan(x) - 1, x) is True
    # 断言：检查 _is_function_class_equation 函数对于给定的三角函数类方程是否返回 True
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x)**2 + sin(x) - 1, x) is True
    # 断言：检查 _is_function_class_equation 函数对于给定的三角函数类方程是否返回 False
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x) + x, x) is False
    # 断言：检查 _is_function_class_equation 函数对于给定的三角函数类方程是否返回 False
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x**2), x) is False
    # 断言：检查 _is_function_class_equation 函数对于给定的三角函数类方程是否返回 False
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x**2) + sin(x), x) is False
    # 断言：检查 _is_function_class_equation 函数对于给定的三角函数类方程是否返回 False
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x)**sin(x), x) is False
    # 断言：检查 _is_function_class_equation 函数对于给定的三角函数类方程是否返回 False
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(sin(x)) + sin(x), x) is False
    # 断言：检查 _is_function_class_equation 函数对于给定的双曲函数类方程是否返回 True
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x), x) is True
    # 断言：检查 _is_function_class_equation 函数对于给定的双曲函数类方程是否返回 True
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x) - 1, x) is True
    # 断言：检查 _is_function_class_equation 函数对于给定的双曲函数类方程是否返回 True
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x) + sinh(x), x) is True
    # 断言：检查 _is_function_class_equation 函数对于给定的双曲函数类方程是否返回 True
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x) + sinh(x) - a, x) is True
    # 断言：检查 _is_function_class_equation 函数对于给定的双曲函数类方程是否返回 True
    assert _is_function_class_equation(HyperbolicFunction,
                                       sinh(x)*tanh(x) + sinh(x), x) is True
    # 断言：检查 _is_function_class_equation 函数对于给定的双曲函数类方程是否返回 True
    assert _is_function_class_equation(HyperbolicFunction,
                                       sinh(x)*tanh(x + a) + sinh(x), x) is True
    # 断言：检查 _is_function_class_equation 函数对于给定的双曲函数类方程是否返回 True
    assert _is_function_class_equation(HyperbolicFunction,
                                       sinh(x)*tanh(x*a) + sinh(x), x) is True
    # 断言：检查 _is_function_class_equation 函数对于给定的双曲函数类方程是否返回 True
    assert _is_function_class_equation(HyperbolicFunction,
                                       a*tanh(x) - 1, x) is True
    # 断言：检查 _is_function_class_equation 函数对于给定的双曲函数类方程是否返回 True
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x)**2 + sinh(x) - 1, x) is True
    # 断言：检查 _is_function_class_equation 函数对于给定的双曲函数类方程是否返回 False
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x) + x, x) is False
    # 断言：检查 _is_function_class_equation 函数对于给定的双曲函数类方程是否返回 False
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x**2), x) is False
    # 断言：检查 _is_function_class_equation 函数对于给定的双曲函数类方程是否返回 False
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x**2) + sinh(x), x) is False
    # 断言：检查 _is_function_class_equation 函数对于给定的双曲函数类方程是否返回 False
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x)**sinh(x), x) is False
    # 断言：检查 _is_function_class_equation 函数对于给定的双曲函数类方程是否返回 False
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(sinh(x)) + sinh(x), x) is False
# 定义测试函数 test_garbage_input
def test_garbage_input():
    # 测试异常情况，期望抛出 ValueError 异常，lambda 函数调用 solveset_real([y], y)
    raises(ValueError, lambda: solveset_real([y], y))
    
    # 创建实数符号 x
    x = Symbol('x', real=True)
    
    # 断言解实数方程 solveset_real(x, 1) 为空集
    assert solveset_real(x, 1) == S.EmptySet
    
    # 断言解实数方程 solveset_real(x - 1, 1) 等于有限集 {x}
    assert solveset_real(x - 1, 1) == FiniteSet(x)
    
    # 断言解实数方程 solveset_real(x, pi) 为空集
    assert solveset_real(x, pi) == S.EmptySet
    
    # 断言解实数方程 solveset_real(x, x**2) 为空集
    assert solveset_real(x, x**2) == S.EmptySet
    
    # 测试异常情况，期望抛出 ValueError 异常，lambda 函数调用 solveset_complex([x], x)
    raises(ValueError, lambda: solveset_complex([x], x))
    
    # 断言解复数方程 solveset_complex(x, pi) 为空集
    assert solveset_complex(x, pi) == S.EmptySet
    
    # 测试异常情况，期望抛出 ValueError 异常，调用 solveset((x, y), x)
    raises(ValueError, lambda: solveset((x, y), x))
    
    # 测试异常情况，期望抛出 ValueError 异常，调用 solveset(x + 1, S.Reals)
    raises(ValueError, lambda: solveset(x + 1, S.Reals))
    
    # 测试异常情况，期望抛出 ValueError 异常，调用 solveset(x + 1, x, 2)
    raises(ValueError, lambda: solveset(x + 1, x, 2))


# 定义测试函数 test_solve_mul
def test_solve_mul():
    # 断言解实数方程 solveset_real((a*x + b)*(exp(x) - 3), x) 等于 Union({log(3)}, Intersection({-b/a}, S.Reals))
    assert solveset_real((a*x + b)*(exp(x) - 3), x) == \
        Union({log(3)}, Intersection({-b/a}, S.Reals))
    
    # 创建非零符号 anz
    anz = Symbol('anz', nonzero=True)
    
    # 创建实数符号 bb
    bb = Symbol('bb', real=True)
    
    # 断言解实数方程 solveset_real((anz*x + bb)*(exp(x) - 3), x) 等于有限集 {-bb/anz, log(3)}
    assert solveset_real((anz*x + bb)*(exp(x) - 3), x) == \
        FiniteSet(-bb/anz, log(3))
    
    # 断言解实数方程 solveset_real((2*x + 8)*(8 + exp(x)), x) 等于有限集 {-4}
    assert solveset_real((2*x + 8)*(8 + exp(x)), x) == FiniteSet(S(-4))
    
    # 断言解实数方程 solveset_real(x/log(x), x) 为空集
    assert solveset_real(x/log(x), x) is S.EmptySet


# 定义测试函数 test_solve_invert
def test_solve_invert():
    # 断言解实数方程 solveset_real(exp(x) - 3, x) 等于有限集 {log(3)}
    assert solveset_real(exp(x) - 3, x) == FiniteSet(log(3))
    
    # 断言解实数方程 solveset_real(log(x) - 3, x) 等于有限集 {exp(3)}
    assert solveset_real(log(x) - 3, x) == FiniteSet(exp(3))
    
    # 断言解实数方程 solveset_real(3**(x + 2), x) 为空集
    assert solveset_real(3**(x + 2), x) == FiniteSet()
    
    # 断言解实数方程 solveset_real(3**(2 - x), x) 为空集
    assert solveset_real(3**(2 - x), x) == FiniteSet()
    
    # 断言解实数方程 solveset_real(y - b*exp(a/x), x) 等于交集(S.Reals, {a/log(y/b)})
    assert solveset_real(y - b*exp(a/x), x) == Intersection(
        S.Reals, FiniteSet(a/log(y/b)))
    
    # issue 4504 的断言，解实数方程 solveset_real(2**x - 10, x) 等于有限集 {1 + log(5)/log(2)}
    assert solveset_real(2**x - 10, x) == FiniteSet(1 + log(5)/log(2))


# 定义测试函数 test_issue_25768
def test_issue_25768():
    # 断言解实数方程 solveset_real(sin(x) - S.Half, x) 等于 Union(ImageSet(Lambda(n, pi*2*n + pi/6), S.Integers), ImageSet(Lambda(n, pi*2*n + pi*5/6), S.Integers))
    assert dumeq(solveset_real(sin(x) - S.Half, x), Union(
        ImageSet(Lambda(n, pi*2*n + pi/6), S.Integers),
        ImageSet(Lambda(n, pi*2*n + pi*5/6), S.Integers)))
    
    # 调用 solveset_real(sin(x) - 0.5, x).n(5)，将结果存入 n1
    n1 = solveset_real(sin(x) - 0.5, x).n(5)
    
    # 调用 solveset_real(sin(x) - S.Half, x).n(5)，将结果存入 n2
    n2 = solveset_real(sin(x) - S.Half, x).n(5)
    
    # 对 n1 和 n2 的浮点数结果进行处理，将其中浮点数转换为有限精度有理数
    eq = [i.replace(
        lambda x:x.is_Float,
        lambda x:Rational(x).limit_denominator(1000)) for i in (n1, n2)]
    
    # 断言处理后的结果相等
    assert dumeq(*eq), (n1, n2)


# 定义测试函数 test_errorinverses
def test_errorinverses():
    # 断言解实数方程 solveset_real(erf(x) - S.Half, x) 等于有限集 {erfinv(S.Half)}
    assert solveset_real(erf(x) - S.Half, x) == \
        FiniteSet(erfinv(S.Half))
    
    # 断言解实数方程 solveset_real(erfinv(x) - 2, x) 等于有限集 {erf(2)}
    assert solveset_real(erfinv(x) - 2, x) == \
        FiniteSet(erf(2))
    
    # 断言解实数方程 solveset_real(erfc(x) - S.One, x) 等于有限集 {erfcinv(S.One)}
    assert solveset_real(erfc(x) - S.One, x) == \
        FiniteSet(erfcinv(S.One))
    
    # 断言解实数方程 solveset_real(erfcinv(x) - 2, x) 等于有限集 {erfc(2)}
    assert solveset_real(erfcinv(x) - 2, x) == FiniteSet(erfc(2))


# 定义测试函数 test_solve_polynomial
def test_solve_polynomial():
    # 创建实数符号 x
    x = Symbol('x', real=True)
    
    # 创建实数符号 y
    y = Symbol('y', real=True)
    
    # 断言解实数方程 solveset_real(3*x - 2, x) 等于有限集 {Rational(2, 3)}
    assert solveset_real(3*x - 2, x) == FiniteSet(Rational(2, 3))
    
    # 断言解实数方程 solveset_real(x**2 - 1, x) 等于有限集 {-1, 1}
    assert solveset_real(x**2 - 1, x) == FiniteSet(-S.One, S.One)
    
    # 断言解实数方程 solveset_real(x - y**3, x) 等于有限集 {y ** 3}
    assert solveset_real(x - y**3, x) == FiniteSet(y ** 3)
    # 断言：验证求解实数解的结果集合长度大于零
    assert len(solveset_real(-2*x**3 + 4*x**2 - 2*x + 6, x)) > 0
    
    # 断言：验证求解 x^6 + x^4 + i = 0 的实数解是否为空集
    assert solveset_real(x**6 + x**4 + I, x) is S.EmptySet
# 定义一个测试函数，用于验证 solveset_complex 的返回结果是否正确
def test_return_root_of():
    # 定义一个五次多项式
    f = x**5 - 15*x**3 - 5*x**2 + 10*x + 20
    # 解五次方程复数解集合，并转换为列表
    s = list(solveset_complex(f, x))
    # 验证每个解是否为 CRootOf 类型
    for root in s:
        assert root.func == CRootOf

    # 验证对于具有 CRootOf 解的多项式，使用 nfloat 处理求解过程不会失败
    assert nfloat(list(solveset_complex(x**5 + 3*x**3 + 7, x))[0],
                  exponent=False) == CRootOf(x**5 + 3*x**3 + 7, 0).n()

    # 解六次方程的复数解集合，并转换为列表
    sol = list(solveset_complex(x**6 - 2*x + 2, x))
    # 验证所有解是否均为 CRootOf 类型，并且解的数量为 6
    assert all(isinstance(i, CRootOf) for i in sol) and len(sol) == 6

    # 重新定义五次多项式
    f = x**5 - 15*x**3 - 5*x**2 + 10*x + 20
    # 解五次方程复数解集合，并转换为列表
    s = list(solveset_complex(f, x))
    # 验证每个解是否为 CRootOf 类型
    for root in s:
        assert root.func == CRootOf

    # 定义一个多项式表达式
    s = x**5 + 4*x**3 + 3*x**2 + Rational(7, 4)
    # 验证 solveset_complex 函数返回的解是否与 Poly 函数获取所有根的结果相等
    assert solveset_complex(s, x) == \
        FiniteSet(*Poly(s*4, domain='ZZ').all_roots())

    # 引用问题 #7876
    # 定义一个复杂的多项式方程
    eq = x*(x - 1)**2*(x + 1)*(x**6 - x + 1)
    # 验证 solveset_complex 函数返回的解是否与多项式所有根的结果相等
    assert solveset_complex(eq, x) == \
        FiniteSet(-1, 0, 1, CRootOf(x**6 - x + 1, 0),
                       CRootOf(x**6 - x + 1, 1),
                       CRootOf(x**6 - x + 1, 2),
                       CRootOf(x**6 - x + 1, 3),
                       CRootOf(x**6 - x + 1, 4),
                       CRootOf(x**6 - x + 1, 5))


# 定义一个测试函数，用于验证 solveset_real 处理平方根方程的解是否正确
def test_solveset_sqrt_1():
    # 验证解方程 sqrt(5*x + 6) - 2 - x 的实数解是否为 {-1, 2}
    assert solveset_real(sqrt(5*x + 6) - 2 - x, x) == \
        FiniteSet(-S.One, S(2))
    # 验证解方程 sqrt(x - 1) - x + 7 的实数解是否为 {10}
    assert solveset_real(sqrt(x - 1) - x + 7, x) == FiniteSet(10)
    # 验证解方程 sqrt(x - 2) - 5 的实数解是否为 {27}
    assert solveset_real(sqrt(x - 2) - 5, x) == FiniteSet(27)
    # 验证解方程 sqrt(x) - 2 - 5 的实数解是否为 {49}
    assert solveset_real(sqrt(x) - 2 - 5, x) == FiniteSet(49)
    # 验证解方程 sqrt(x**3) 的实数解是否为 {0}
    assert solveset_real(sqrt(x**3), x) == FiniteSet(0)
    # 验证解方程 sqrt(x - 1) 的实数解是否为 {1}
    assert solveset_real(sqrt(x - 1), x) == FiniteSet(1)
    # 验证解方程 sqrt((x-3)/x) 的实数解是否为 {3}
    assert solveset_real(sqrt((x-3)/x), x) == FiniteSet(3)
    # 验证解方程 sqrt((x-3)/x)-Rational(1, 2) 的实数解是否为 {4}


# 定义一个测试函数，用于验证 solveset_real 处理复杂平方根方程的解是否正确
def test_solveset_sqrt_2():
    # 定义符号 x 和 y 为实数
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    # 验证解方程 sqrt(2*x - 1) - sqrt(x - 4) - 2 的实数解是否为 {5, 13}
    assert solveset_real(sqrt(2*x - 1) - sqrt(x - 4) - 2, x) == \
        FiniteSet(S(5), S(13))
    # 验证解方程 sqrt(x + 7) + 2 - sqrt(3 - x) 的实数解是否为 {-6}
    assert solveset_real(sqrt(x + 7) + 2 - sqrt(3 - x), x) == \
        FiniteSet(-6)

    # 验证解方程 sqrt(17*x - sqrt(x**2 - 5)) - 7 的实数解是否为 {3}
    assert solveset_real(sqrt(17*x - sqrt(x**2 - 5)) - 7, x) == \
        FiniteSet(3)

    # 定义一个复杂的方程
    eq = x + 1 - (x**4 + 4*x**3 - x)**Rational(1, 4)
    # 验证 solveset_real 函数返回的解是否为 {-1/2, -1/3}
    assert solveset_real(eq, x) == FiniteSet(Rational(-1, 2), Rational(-1, 3))

    # 定义一个复杂的方程
    eq = sqrt(2*x + 9) - sqrt(x + 1) - sqrt(x + 4)
    # 验证 solveset_real 函数返回的解是否为 {0}
    assert solveset_real(eq, x) == FiniteSet(0)

    # 定义一个复杂的方程
    eq = sqrt(x + 4) + sqrt(2*x - 1) - 3*sqrt(x - 1)
    # 确定解集，验证方程解是否等于有限集合中的5
    assert solveset_real(eq, x) == FiniteSet(5)
    
    # 定义方程，验证解是否等于有限集合中的16
    eq = sqrt(x)*sqrt(x - 7) - 12
    assert solveset_real(eq, x) == FiniteSet(16)
    
    # 设定方程，验证解是否等于有限集合中的4
    eq = sqrt(x - 3) + sqrt(x) - 3
    assert solveset_real(eq, x) == FiniteSet(4)
    
    # 定义方程，验证解是否等于有限集合中的 -8 和 2
    eq = sqrt(2*x**2 - 7) - (3 - x)
    assert solveset_real(eq, x) == FiniteSet(-S(8), S(2))
    
    # 其他方程
    eq = sqrt(9*x**2 + 4) - (3*x + 2)
    assert solveset_real(eq, x) == FiniteSet(0)
    
    # 验证解是否为空集
    assert solveset_real(sqrt(x - 3) - sqrt(x) - 3, x) == FiniteSet()
    
    # 定义方程，验证解是否等于有限集合中的16
    eq = (2*x - 5)**Rational(1, 3) - 3
    assert solveset_real(eq, x) == FiniteSet(16)
    
    # 验证解是否等于有限集合中的特定值
    assert solveset_real(sqrt(x) + sqrt(sqrt(x)) - 4, x) == \
        FiniteSet((Rational(-1, 2) + sqrt(17)/2)**4)
    
    # 定义方程，验证解是否为空集
    eq = sqrt(x) - sqrt(x - 1) + sqrt(sqrt(x))
    assert solveset_real(eq, x) == FiniteSet()
    
    # 定义方程，验证解是否等于有限集合中的 -4 和 4
    eq = (x - 4)**2 + (sqrt(x) - 2)**4
    assert solveset_real(eq, x) == FiniteSet(-4, 4)
    
    # 定义方程，验证解是否满足特定的数值条件
    eq = (sqrt(x) + sqrt(x + 1) + sqrt(1 - x) - 6*sqrt(5)/5)
    ans = solveset_real(eq, x)
    ra = S('''-1484/375 - 4*(-S(1)/2 + sqrt(3)*I/2)*(-12459439/52734375 +
    114*sqrt(12657)/78125)**(S(1)/3) - 172564/(140625*(-S(1)/2 +
    sqrt(3)*I/2)*(-12459439/52734375 + 114*sqrt(12657)/78125)**(S(1)/3))''')
    rb = Rational(4, 5)
    assert all(abs(eq.subs(x, i).n()) < 1e-10 for i in (ra, rb)) and \
        len(ans) == 2 and \
        {i.n(chop=True) for i in ans} == \
        {i.n(chop=True) for i in (ra, rb)}
    
    # 定义方程，验证解是否等于有限集合中的0
    assert solveset_real(sqrt(x) + x**Rational(1, 3) +
                                 x**Rational(1, 4), x) == FiniteSet(0)
    
    # 定义方程，验证解是否等于有限集合中的0
    assert solveset_real(x/sqrt(x**2 + 1), x) == FiniteSet(0)
    
    # 定义方程，验证解是否等于有限集合中的 y**3
    eq = (x - y**3)/((y**2)*sqrt(1 - y**2))
    assert solveset_real(eq, x) == FiniteSet(y**3)
    
    # issue 4497
    # 定义方程，验证解是否等于有限集合中的 Rational(-295244, 59049)
    assert solveset_real(1/(5 + x)**Rational(1, 5) - 9, x) == \
        FiniteSet(Rational(-295244, 59049))
@XFAIL
# 标记此测试函数为预期失败（XFAIL），因为 solveset_real(eq.subs(x, Rational(1, 3))) 的检查方式与 checksol 的方式不匹配
def test_solve_sqrt_fail():
    # 创建方程式 eq，并断言其解集合为有理数 1/3 的有限集
    eq = (x**3 - 3*x**2)**Rational(1, 3) + 1 - x
    assert solveset_real(eq, x) == FiniteSet(Rational(1, 3))


@slow
# 标记此测试函数为慢速执行（slow）
def test_solve_sqrt_3():
    # 创建符号 R
    R = Symbol('R')
    # 创建方程式 eq
    eq = sqrt(2)*R*sqrt(1/(R + 1)) + (R + 1)*(sqrt(2)*sqrt(1/(R + 1)) - 1)
    # 求解复数解集 sol
    sol = solveset_complex(eq, R)
    # 创建一个包含复数解的列表 fset
    fset = [Rational(5, 3) + 4*sqrt(10)*cos(atan(3*sqrt(111)/251)/3)/3,
            -sqrt(10)*cos(atan(3*sqrt(111)/251)/3)/3 +
            40*re(1/((Rational(-1, 2) - sqrt(3)*I/2)*(Rational(251, 27) + sqrt(111)*I/9)**Rational(1, 3)))/9 +
            sqrt(30)*sin(atan(3*sqrt(111)/251)/3)/3 + Rational(5, 3) +
            I*(-sqrt(30)*cos(atan(3*sqrt(111)/251)/3)/3 -
               sqrt(10)*sin(atan(3*sqrt(111)/251)/3)/3 +
               40*im(1/((Rational(-1, 2) - sqrt(3)*I/2)*(Rational(251, 27) + sqrt(111)*I/9)**Rational(1, 3)))/9)]
    # 创建一个包含复数条件解集的列表 cset
    cset = [40*re(1/((Rational(-1, 2) + sqrt(3)*I/2)*(Rational(251, 27) + sqrt(111)*I/9)**Rational(1, 3)))/9 -
            sqrt(10)*cos(atan(3*sqrt(111)/251)/3)/3 - sqrt(30)*sin(atan(3*sqrt(111)/251)/3)/3 +
            Rational(5, 3) +
            I*(40*im(1/((Rational(-1, 2) + sqrt(3)*I/2)*(Rational(251, 27) + sqrt(111)*I/9)**Rational(1, 3)))/9 -
               sqrt(10)*sin(atan(3*sqrt(111)/251)/3)/3 +
               sqrt(30)*cos(atan(3*sqrt(111)/251)/3)/3)]

    # 创建有限集 fs 和条件集 cs
    fs = FiniteSet(*fset)
    cs = ConditionSet(R, Eq(eq, 0), FiniteSet(*cset))
    # 断言 sol 等于 fs 和 cs 的并集减去元素 -1
    assert sol == (fs - {-1}) | (cs - {-1})

    # 创建方程式 eq2，描述其实数根数量取决于 m 的值：当 m=1 时有 4 个根，当 m=-1 时没有根
    eq2 = -sqrt((m - q)**2 + (-m/(2*q) + S.Half)**2) + sqrt((-m**2/2 - sqrt(
        4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2 + (m**2/2 - m - sqrt(
            4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2)
    # 创建未解决的对象 unsolved_object
    unsolved_object = ConditionSet(q, Eq(sqrt((m - q)**2 + (-m/(2*q) + S.Half)**2) -
        sqrt((-m**2/2 - sqrt(4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2 + (m**2/2 - m -
        sqrt(4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2), 0), S.Reals)
    # 断言 solveset_real(eq2, q) 等于 unsolved_object
    assert solveset_real(eq2, q) == unsolved_object


# 测试解多项式符号参数
def test_solve_polynomial_symbolic_param():
    # 断言求解复数解集 (x**2 - 1)**2 - a 关于 x 的解集合
    assert solveset_complex((x**2 - 1)**2 - a, x) == \
        FiniteSet(sqrt(1 + sqrt(a)), -sqrt(1 + sqrt(a)),
                  sqrt(1 - sqrt(a)), -sqrt(1 - sqrt(a)))

    # issue 4507 的问题
    # 断言求解复数解集 y - b/(1 + a*x) 关于 x 的解集合
    assert solveset_complex(y - b/(1 + a*x), x) == \
        FiniteSet((b/y - 1)/a) - FiniteSet(-1/a)

    # issue 4508 的问题
    # 断言求解复数解集 y - b*x/(a + x) 关于 x 的解集合
    assert solveset_complex(y - b*x/(a + x), x) == \
        FiniteSet(-a*y/(y - b)) - FiniteSet(-a)


# 测试解有理数方程
def test_solve_rational():
    # 断言求解实数解集 1/x + 1 关于 x 的解集合
    assert solveset_real(1/x + 1, x) == FiniteSet(-S.One)
    # 断言求解实数解集 1/exp(x) - 1 关于 x 的解集合
    assert solveset_real(1/exp(x) - 1, x) == FiniteSet(0)
    # 断言求解实数解集 x*(1 - 5/x) 关于 x 的解集合
    assert solveset_real(x*(1 - 5/x), x) == FiniteSet(5)
    # 断言求解实数解集 2*x/(x + 2) - 1 关于 x 的解集合
    assert solveset_real(2*x/(x + 2) - 1, x) == FiniteSet(2)
    # 使用 SymPy 解方程 (x**2/(7 - x)) 的导数，求解实数解集合
    assert solveset_real((x**2/(7 - x)).diff(x), x) == \
        FiniteSet(S.Zero, S(14))
`
def test_solveset_real_gen_is_pow():
    # 测试 solveset_real 函数求解 sqrt(1) + 1 的结果，期望返回空集合
    assert solveset_real(sqrt(1) + 1, x) is S.EmptySet


def test_no_sol():
    # 测试求解 1 - oo*x 的结果，期望返回空集合
    assert solveset(1 - oo*x) is S.EmptySet
    # 测试求解 oo*x 的结果，期望返回空集合
    assert solveset(oo*x, x) is S.EmptySet
    # 测试求解 oo*x - oo 的结果，期望返回空集合
    assert solveset(oo*x - oo, x) is S.EmptySet
    # 测试求解常数 4 的结果，期望返回空集合
    assert solveset_real(4, x) is S.EmptySet
    # 测试求解 exp(x) 的结果，期望返回空集合
    assert solveset_real(exp(x), x) is S.EmptySet
    # 测试求解 x^2 + 1 的结果，期望返回空集合
    assert solveset_real(x**2 + 1, x) is S.EmptySet
    # 测试求解 -3*a/sqrt(x) 的结果，期望返回空集合
    assert solveset_real(-3*a/sqrt(x), x) is S.EmptySet
    # 测试求解 1/x 的结果，期望返回空集合
    assert solveset_real(1/x, x) is S.EmptySet
    # 测试复杂表达式 -(1 + x)/(2 + x)**2 + 1/(2 + x) 的求解结果，期望返回空集合
    assert solveset_real(-(1 + x)/(2 + x)**2 + 1/(2 + x), x) is S.EmptySet


def test_sol_zero_real():
    # 测试求解 0 的实数解，期望返回所有实数
    assert solveset_real(0, x) == S.Reals
    # 测试在区间 [1, 2] 上求解 0 的结果，期望返回区间 [1, 2]
    assert solveset(0, x, Interval(1, 2)) == Interval(1, 2)
    # 测试求解 -x^2 - 2*x + (x + 1)^2 - 1 的实数解，期望返回所有实数
    assert solveset_real(-x**2 - 2*x + (x + 1)**2 - 1, x) == S.Reals


def test_no_sol_rational_extragenous():
    # 测试求解 (x/(x + 1) + 3)^(-2) 的实数解，期望返回空集合
    assert solveset_real((x/(x + 1) + 3)**(-2), x) is S.EmptySet
    # 测试求解 (x - 1)/(1 + 1/(x - 1)) 的实数解，期望返回空集合
    assert solveset_real((x - 1)/(1 + 1/(x - 1)), x) is S.EmptySet


def test_solve_polynomial_cv_1a():
    """
    测试通过变量变换 y -> x**Rational(p, q) 将方程转化为多项式方程的求解
    """
    # 测试求解 sqrt(x) - 1 的实数解，期望结果为 {1}
    assert solveset_real(sqrt(x) - 1, x) == FiniteSet(1)
    # 测试求解 sqrt(x) - 2 的实数解，期望结果为 {4}
    assert solveset_real(sqrt(x) - 2, x) == FiniteSet(4)
    # 测试求解 x**Rational(1, 4) - 2 的实数解，期望结果为 {16}
    assert solveset_real(x**Rational(1, 4) - 2, x) == FiniteSet(16)
    # 测试求解 x**Rational(1, 3) - 3 的实数解，期望结果为 {27}
    assert solveset_real(x**Rational(1, 3) - 3, x) == FiniteSet(27)
    # 测试求解 x*(x**(S.One / 3) - 3) 的实数解，期望结果为 {0, 27}
    assert solveset_real(x*(x**(S.One / 3) - 3), x) == FiniteSet(S.Zero, S(27))


def test_solveset_real_rational():
    """测试 solveset_real 函数对有理函数的求解"""
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    # 测试求解 (x - y**3) / (y**2 * sqrt(1 - y**2)) 的实数解，期望结果为 {y**3}
    assert solveset_real((x - y**3) / ((y**2)*sqrt(1 - y**2)), x) == FiniteSet(y**3)
    # 测试 issue 4486，求解 2*x/(x + 2) - 1 的实数解，期望结果为 {2}
    assert solveset_real(2*x/(x + 2) - 1, x) == FiniteSet(2)


def test_solveset_real_log():
    # 测试求解 log((x-1)*(x+1)) 的实数解，期望结果为 {sqrt(2), -sqrt(2)}
    assert solveset_real(log((x-1)*(x+1)), x) == FiniteSet(sqrt(2), -sqrt(2))


def test_poly_gens():
    # 测试求解 4**(2*(x**2) + 2*x) - 8 的实数解，期望结果为 {-3/2, 1/2}
    assert solveset_real(4**(2*(x**2) + 2*x) - 8, x) == FiniteSet(Rational(-3, 2), S.Half)


def test_solve_abs():
    n = Dummy('n')
    # 测试求解 Abs(x) - 1 的实数解，期望抛出 ValueError 异常
    raises(ValueError, lambda: solveset(Abs(x) - 1, x))
    # 测试求解 Abs(x) - n 的实数解，期望结果为 ConditionSet(x, Contains(n, Interval(0, oo)), {-n, n})
    assert solveset(Abs(x) - n, x, S.Reals).dummy_eq(
        ConditionSet(x, Contains(n, Interval(0, oo)), {-n, n}))
    # 测试求解 Abs(x) - 2 的实数解，期望结果为 {-2, 2}
    assert solveset_real(Abs(x) - 2, x) == FiniteSet(-2, 2)
    # 测试求解 Abs(x) + 2 的实数解，期望返回空集合
    assert solveset_real(Abs(x) + 2, x) is S.EmptySet
    # 测试求解 Abs(x + 3) - 2*Abs(x - 3) 的实数解，期望结果为 {1, 9}
    assert solveset_real(Abs(x + 3) - 2*Abs(x - 3), x) == FiniteSet(1, 9)
    # 测试求解 2*Abs(x) - Abs(x - 1) 的实数解，期望结果为 {-1, 1/3}
    assert solveset_real(2*Abs(x) - Abs(x - 1), x) == FiniteSet(-1, Rational(1, 3))

    # 定义条件集 sol
    sol = ConditionSet(
            x,
            And(
                Contains(b, Interval(0, oo)),
                Contains(a + b, Interval(0, oo)),
                Contains(a - b, Interval(0, oo))),
            FiniteSet(-a - b - 3, -a + b - 3, a - b - 3, a + b - 3))
    # 定义方程 eq
    eq = Abs(Abs(x + 3) - a) - b
    # 测试 invert_real 函数的结果，期望结果为 sol
    assert invert_real(eq, 0, x)[1] == sol
    # 定义字典替换 {a: 3, b: 1}
    reps = {a: 3, b: 1}
    # 将 eq 中的 a 和 b 替换为对应的值
    eqab = eq.subs(reps)
    # 对 sol 中的每个元素 si 执行替换操作，使用 reps 中的替换规则
    for si in sol.subs(reps):
        # 断言在替换后的表达式 eqab 中，对 x 使用 si 后为 False
        assert not eqab.subs(x, si)
    
    # 断言解集 solveset(Eq(sin(Abs(x)), 1), x, domain=S.Reals) 等于以下联合集合
    assert dumeq(solveset(Eq(sin(Abs(x)), 1), x, domain=S.Reals), Union(
        # 包含以下两部分的交集
        Intersection(Interval(0, oo), Union(
            # 第一部分：ImageSet(Lambda(n, 2*n*pi + 3*pi/2), S.Integers) 与 Interval(-oo, 0) 的交集
            Intersection(ImageSet(Lambda(n, 2*n*pi + 3*pi/2), S.Integers),
                         Interval(-oo, 0)),
            # 第二部分：ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers) 与 Interval(0, oo) 的交集
            Intersection(ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers),
                         Interval(0, oo)))))
def test_issue_9824():
    # 检查解 sin(x)^2 - 2*sin(x) + 1 = ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers) 是否相等
    assert dumeq(solveset(sin(x)**2 - 2*sin(x) + 1, x), ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers))
    # 检查解 cos(x)^2 - 2*cos(x) + 1 = ImageSet(Lambda(n, 2*n*pi), S.Integers) 是否相等
    assert dumeq(solveset(cos(x)**2 - 2*cos(x) + 1, x), ImageSet(Lambda(n, 2*n*pi), S.Integers))


def test_issue_9565():
    # 解绝对值不等式 Abs((x - 1)/(x - 5)) <= 1/3，应为闭区间 [-1, 2]
    assert solveset_real(Abs((x - 1)/(x - 5)) <= Rational(1, 3), x) == Interval(-1, 2)


def test_issue_10069():
    # 解不等式 abs(1/(x - 1)) - 1 > 0，结果应为两个开区间的并集 (0, 1) ∪ (1, 2)
    eq = abs(1/(x - 1)) - 1 > 0
    assert solveset_real(eq, x) == Union(
        Interval.open(0, 1), Interval.open(1, 2))


def test_real_imag_splitting():
    # 测试实数解，sqrt(a**2 - b**2) - 3 = 0 的解为 {-sqrt(b**2 + 9), sqrt(b**2 + 9)}
    a, b = symbols('a b', real=True)
    assert solveset_real(sqrt(a**2 - b**2) - 3, a) == \
        FiniteSet(-sqrt(b**2 + 9), sqrt(b**2 + 9))
    # 测试实数解，sqrt(a**2 + b**2) - 3 = 0 的解不应为空集
    assert solveset_real(sqrt(a**2 + b**2) - 3, a) != \
        S.EmptySet


def test_units():
    # 解方程 1/x - 1/(2*cm) = 0，结果应为 {2*cm}
    assert solveset_real(1/x - 1/(2*cm), x) == FiniteSet(2*cm)


def test_solve_only_exp_1():
    # 解指数方程 exp(x) - y = 0，结果应为 {log(y)}
    y = Symbol('y', positive=True)
    assert solveset_real(exp(x) - y, x) == FiniteSet(log(y))
    # 解指数方程 exp(x) + exp(-x) - 4 = 0，结果应为 {log(-sqrt(3) + 2), log(sqrt(3) + 2)}
    assert solveset_real(exp(x) + exp(-x) - 4, x) == \
        FiniteSet(log(-sqrt(3) + 2), log(sqrt(3) + 2))
    # 解指数方程 exp(x) + exp(-x) - y = 0 的解不应为空集
    assert solveset_real(exp(x) + exp(-x) - y, x) != S.EmptySet


def test_atan2():
    # 解反正切函数方程 atan2(x, 2) - pi/3 = 0，结果应为 {2*sqrt(3)}
    assert solveset_real(atan2(x, 2) - pi/3, x) == FiniteSet(2*sqrt(3))


def test_piecewise_solveset():
    # 解分段函数方程 Piecewise((x - 2, x > 2), (2 - x, True)) - 3 = 0，结果应为 {-1, 5}
    eq = Piecewise((x - 2, Gt(x, 2)), (2 - x, True)) - 3
    assert set(solveset_real(eq, x)) == set(FiniteSet(-1, 5))

    # 解分段函数方程 Piecewise((x - 3, 0 <= x - 3), (3 - x, 0 > x - 3)) - y = 0，结果应为 {-y + 3, y + 3}
    absxm3 = Piecewise(
        (x - 3, 0 <= x - 3),
        (3 - x, 0 > x - 3))
    y = Symbol('y', positive=True)
    assert solveset_real(absxm3 - y, x) == FiniteSet(-y + 3, y + 3)

    # 解分段函数 f(x) = Piecewise(((x - 2)**2, x >= 0), (0, True))，在实数域上的解应为 {2} ∪ (-oo, 0)
    f = Piecewise(((x - 2)**2, x >= 0), (0, True))
    assert solveset(f, x, domain=S.Reals) == Union(FiniteSet(2), Interval(-oo, 0, True, True))

    # 解分段函数 Piecewise((x + 1, x > 0), (I, True)) - I = 0，结果应为 (-oo, 0)
    assert solveset(Piecewise((x + 1, x > 0), (I, True)) - I, x, S.Reals) == Interval(-oo, 0)

    # 解分段函数 Piecewise((x - 1, Ne(x, I)), (x, True)) = 0，结果应为 {1}
    assert solveset(Piecewise((x - 1, Ne(x, I)), (x, True)), x) == FiniteSet(1)

    # issue 19718
    # 解分段函数 Piecewise((1, x > 10), (0, True)) > 0，结果应为开区间 (10, oo)
    g = Piecewise((1, x > 10), (0, True))
    assert solveset(g > 0, x, S.Reals) == Interval.open(10, oo)

    # 解布尔值函数 f = BooleanTrue()，在区间 [-3, 10] 上的解应为闭区间 [-3, 10]
    from sympy.logic.boolalg import BooleanTrue
    f = BooleanTrue()
    assert solveset(f, x, domain=Interval(-3, 10)) == Interval(-3, 10)

    # issue 20552
    # 解分段函数 Piecewise((0, Eq(x, 0)), (x**2/Abs(x), True))，在实数域上的解应为 {0}
    f = Piecewise((0, Eq(x, 0)), (x**2/Abs(x), True))
    g = Piecewise((0, Eq(x, pi)), ((x - pi)/sin(x), True))
    assert solveset(f, x, domain=S.Reals) == FiniteSet(0)
    # 解分段函数 Piecewise((0, Eq(x, pi)), ((x - pi)/sin(x), True)) 的解应为 {pi}
    assert solveset(g) == FiniteSet(pi)


def test_solveset_complex_polynomial():
    # 解复数多项式方程 a*x**2 + b*x + c = 0，结果应为 {-b/(2*a) - sqrt(-4*a*c + b**2)/(2*a), -b/(2*a) + sqrt(-4*a*c + b**2)/(2*a)}
    assert solveset_complex(a*x**2 + b*x + c, x) == \
        FiniteSet(-b/(2*a) - sqrt(-4*a*c + b**2)/(2*a),
                  -b/(2*a) + sqrt(-4*a*c + b**2)/(2*a))

    # 解复数多项式方程 x - y**3 = 0，结果应为 {(-x**Rational(1, 3))/2 + I*sqrt(3)*x**Rational(1, 3)/2, x**Rational(1, 3), (-x**Rational(1, 3))/2 - I*sqrt(3)*x**Rational(1, 3)/2}
    assert solveset_complex(x - y**3, y) == FiniteSet(
        (-x**Rational(1, 3))/2 + I*sqrt(3)*x**Rational(1, 3)/2,
        x**Rational(1, 3),
        (-x**Rational(1, 3))/2 - I*sqrt(3)*x**Rational(1, 3)/2)
    # 使用断言来验证解集合是否等于指定的有限集合
    assert solveset_complex(x + 1/x - 1, x) == \
        FiniteSet(S.Half + I*sqrt(3)/2, S.Half - I*sqrt(3)/2)
# 定义测试函数，用于测试 solveset_complex 函数处理边界情况是否正确
def test_sol_zero_complex():
    # 断言 solveset_complex(0, x) 返回 S.Complexes
    assert solveset_complex(0, x) is S.Complexes


# 定义测试函数，测试 solveset_complex 函数处理有理表达式的情况
def test_solveset_complex_rational():
    # 断言 solveset_complex((x - 1)*(x - I)/(x - 3), x) 返回 {1, I}
    assert solveset_complex((x - 1)*(x - I)/(x - 3), x) == \
        FiniteSet(1, I)

    # 断言 solveset_complex((x - y**3)/((y**2)*sqrt(1 - y**2)), x) 返回 {y**3}
    assert solveset_complex((x - y**3)/((y**2)*sqrt(1 - y**2)), x) == \
        FiniteSet(y**3)

    # 断言 solveset_complex(-x**2 - I, x) 返回 {-sqrt(2)/2 + sqrt(2)*I/2, sqrt(2)/2 - sqrt(2)*I/2}
    assert solveset_complex(-x**2 - I, x) == \
        FiniteSet(-sqrt(2)/2 + sqrt(2)*I/2, sqrt(2)/2 - sqrt(2)*I/2)


# 定义测试函数，测试 solveset_complex 函数处理五次方程的情况（跳过这个测试，因为执行太慢）
def test_solve_quintics():
    skip("This test is too slow")
    # 定义五次方程 f(x) = x**5 - 110*x**3 - 55*x**2 + 2310*x + 979
    f = x**5 - 110*x**3 - 55*x**2 + 2310*x + 979
    # 求解 f(x) 的复数解集合 s
    s = solveset_complex(f, x)
    # 对于 s 中的每个根 root，验证 f(root) 的近似值是否接近于零
    for root in s:
        res = f.subs(x, root.n()).n()
        assert tn(res, 0)

    # 定义另一个五次方程 f(x) = x**5 + 15*x + 12
    f = x**5 + 15*x + 12
    # 求解 f(x) 的复数解集合 s
    s = solveset_complex(f, x)
    # 对于 s 中的每个根 root，验证 f(root) 的近似值是否接近于零
    for root in s:
        res = f.subs(x, root.n()).n()
        assert tn(res, 0)


# 定义测试函数，测试 solveset_complex 函数处理指数函数的情况
def test_solveset_complex_exp():
    # 断言 solveset_complex(exp(x) - 1, x) 返回 imageset(Lambda(n, I*2*n*pi), S.Integers)
    assert dumeq(solveset_complex(exp(x) - 1, x),
        imageset(Lambda(n, I*2*n*pi), S.Integers))
    
    # 断言 solveset_complex(exp(x) - I, x) 返回 imageset(Lambda(n, I*(2*n*pi + pi/2)), S.Integers)
    assert dumeq(solveset_complex(exp(x) - I, x),
        imageset(Lambda(n, I*(2*n*pi + pi/2)), S.Integers))
    
    # 断言 solveset_complex(1/exp(x), x) 返回 S.EmptySet
    assert solveset_complex(1/exp(x), x) == S.EmptySet
    
    # 断言 solveset_complex(sinh(x).rewrite(exp), x) 返回 imageset(Lambda(n, n*pi*I), S.Integers)
    assert dumeq(solveset_complex(sinh(x).rewrite(exp), x),
        imageset(Lambda(n, n*pi*I), S.Integers))


# 定义测试函数，测试 solveset_real 函数处理指数函数的实数解情况
def test_solveset_real_exp():
    # 断言 solveset(Eq((-2)**x, 4), x, S.Reals) 返回 {2}
    assert solveset(Eq((-2)**x, 4), x, S.Reals) == FiniteSet(2)
    
    # 断言 solveset(Eq(-2**x, 4), x, S.Reals) 返回 S.EmptySet
    assert solveset(Eq(-2**x, 4), x, S.Reals) == S.EmptySet
    
    # 断言 solveset(Eq((-3)**x, 27), x, S.Reals) 返回 S.EmptySet
    assert solveset(Eq((-3)**x, 27), x, S.Reals) == S.EmptySet
    
    # 断言 solveset(Eq((-5)**(x+1), 625), x, S.Reals) 返回 {3}
    assert solveset(Eq((-5)**(x+1), 625), x, S.Reals) == FiniteSet(3)
    
    # 断言 solveset(Eq(2**(x-3), -16), x, S.Reals) 返回 S.EmptySet
    assert solveset(Eq(2**(x-3), -16), x, S.Reals) == S.EmptySet
    
    # 断言 solveset(Eq((-3)**(x - 3), -3**39), x, S.Reals) 返回 {42}
    assert solveset(Eq((-3)**(x - 3), -3**39), x, S.Reals) == FiniteSet(42)
    
    # 断言 solveset(Eq(2**x, y), x, S.Reals) 返回 Reals 与 {log(y)/log(2)} 的交集
    assert solveset(Eq(2**x, y), x, S.Reals) == Intersection(S.Reals, FiniteSet(log(y)/log(2)))
    
    # 断言 invert_real((-2)**(2*x) - 16, 0, x) 返回 (x, {2})
    assert invert_real((-2)**(2*x) - 16, 0, x) == (x, FiniteSet(2))


# 定义测试函数，测试 solveset_complex 函数处理对数函数的情况
def test_solve_complex_log():
    # 断言 solveset_complex(log(x), x) 返回 {1}
    assert solveset_complex(log(x), x) == FiniteSet(1)
    
    # 断言 solveset_complex(1 - log(a + 4*x**2), x) 返回 {-sqrt(-a + E)/2, sqrt(-a + E)/2}
    assert solveset_complex(1 - log(a + 4*x**2), x) == \
        FiniteSet(-sqrt(-a + E)/2, sqrt(-a + E)/2)


# 定义测试函数，测试 solveset_complex 函数处理平方根函数的情况
def test_solve_complex_sqrt():
    # 断言 solveset_complex(sqrt(5*x + 6) - 2 - x, x) 返回 {-1, 2}
    assert solveset_complex(sqrt(5*x + 6) - 2 - x, x) == \
        FiniteSet(-S.One, S(2))
    
    # 断言 solveset_complex(sqrt(5*x + 6) - (2 + 2*I) - x, x) 返回 {-2, 3 - 4*I}
    assert solveset_complex(sqrt(5*x + 6) - (2 + 2*I) - x, x) == \
        FiniteSet(-S(2), 3 - 4*I)
    
    # 断言 solveset_complex(4*x*(1 - a * sqrt(x)), x) 返回 {0, 1 / a ** 2}
    assert solveset_complex(4*x*(1 - a * sqrt(x)), x) == \
        FiniteSet(S.Zero, 1 / a ** 2)


# 定义测试函数，测试 solveset_complex 函数处理正切函数的情况
def test_solveset_complex_tan():
    # 求解 tan(x) = 0 的复数解集合 s
    s = solveset_complex(tan(x).rewrite(exp), x)
    # 断言 s 等于 imageset(Lambda(n, pi*n), S.Integers) 减去 imageset(Lambda(n, pi*n + pi/2), S.Integers)
    assert dumeq(s, imageset(Lambda(n, pi*n), S.Integers) - \
        imageset(Lambda(n, pi*n + pi/2), S.Integers))


# 使用 _both_exp_pow 修饰的测试函数，测试 solve_real 函数处理三角函数的情况
@_both_exp_pow
def test_solve_trig():
    # 断言 solveset_real(sin(x), x) 等于 2*pi*n 或
    # 断言：求解 sin(x) + cos(x) = 0 的实数解集，与给定的并集相等
    assert dumeq(solveset_real(sin(x) + cos(x), x),
        Union(imageset(Lambda(n, 2*n*pi + pi*Rational(3, 4)), S.Integers),
              imageset(Lambda(n, 2*n*pi + pi*Rational(7, 4)), S.Integers)))

    # 断言：求解 sin(x)**2 + cos(x)**2 = 0 的实数解集，结果为空集
    assert solveset_real(sin(x)**2 + cos(x)**2, x) == S.EmptySet

    # 断言：求解 cos(x) - 1/2 = 0 的复数解集，与给定的并集相等
    assert dumeq(solveset_complex(cos(x) - S.Half, x),
        Union(imageset(Lambda(n, 2*n*pi + pi*Rational(5, 3)), S.Integers),
              imageset(Lambda(n, 2*n*pi + pi/3), S.Integers)))

    # 断言：求解 sin(y + a) - sin(y) = 0 的实数解集，满足给定的条件集合和并集
    assert dumeq(solveset(sin(y + a) - sin(y), a, domain=S.Reals),
        ConditionSet(a, (S(-1) <= sin(y)) & (sin(y) <= S(1)), Union(
            ImageSet(Lambda(n, 2*n*pi - y + asin(sin(y))), S.Integers),
            ImageSet(Lambda(n, 2*n*pi - y - asin(sin(y)) + pi), S.Integers))))

    # 断言：求解 sin(2*x)*cos(x) + cos(2*x)*sin(x) - 1 = 0 的实数解集
    assert dumeq(solveset_real(sin(2*x)*cos(x) + cos(2*x)*sin(x)-1, x),
        ImageSet(Lambda(n, n*pi*Rational(2, 3) + pi/6), S.Integers))

    # 断言：求解 2*tan(x)*sin(x) + 1 = 0 的实数解集，与给定的并集相等
    assert dumeq(solveset_real(2*tan(x)*sin(x) + 1, x), Union(
        ImageSet(Lambda(n, 2*n*pi + atan(sqrt(2)*sqrt(-1 + sqrt(17))/
            (1 - sqrt(17))) + pi), S.Integers),
        ImageSet(Lambda(n, 2*n*pi - atan(sqrt(2)*sqrt(-1 + sqrt(17))/
            (1 - sqrt(17))) + pi), S.Integers)))

    # 断言：求解 cos(2*x)*cos(4*x) - 1 = 0 的实数解集，与给定的映射集相等
    assert dumeq(solveset_real(cos(2*x)*cos(4*x) - 1, x),
                            ImageSet(Lambda(n, n*pi), S.Integers))

    # 断言：求解 sin(x/10) + 3/4 = 0 的实数解集，与给定的并集相等
    assert dumeq(solveset(sin(x/10) + Rational(3, 4)), Union(
        ImageSet(Lambda(n, 20*n*pi - 10*asin(S(3)/4) + 20*pi), S.Integers),
        ImageSet(Lambda(n, 20*n*pi + 10*asin(S(3)/4) + 10*pi), S.Integers)))

    # 断言：求解 cos(x/15) + cos(x/5) = 0 的实数解集，与给定的并集相等
    assert dumeq(solveset(cos(x/15) + cos(x/5)), Union(
        ImageSet(Lambda(n, 30*n*pi + 15*pi/2), S.Integers),
        ImageSet(Lambda(n, 30*n*pi + 45*pi/2), S.Integers),
        ImageSet(Lambda(n, 30*n*pi + 75*pi/4), S.Integers),
        ImageSet(Lambda(n, 30*n*pi + 45*pi/4), S.Integers),
        ImageSet(Lambda(n, 30*n*pi + 105*pi/4), S.Integers),
        ImageSet(Lambda(n, 30*n*pi + 15*pi/4), S.Integers)))

    # 断言：求解 sec(sqrt(2)*x/3) + 5 = 0 的实数解集，与给定的并集相等
    assert dumeq(solveset(sec(sqrt(2)*x/3) + 5), Union(
        ImageSet(Lambda(n, 3*sqrt(2)*(2*n*pi - asec(-5))/2), S.Integers),
        ImageSet(Lambda(n, 3*sqrt(2)*(2*n*pi + asec(-5))/2), S.Integers)))

    # 断言：求解 tan(pi*x) - cot(pi/2*x) = 0 的简化解集，与给定的并集相等
    assert dumeq(simplify(solveset(tan(pi*x) - cot(pi/2*x))), Union(
        ImageSet(Lambda(n, 4*n + 1), S.Integers),
        ImageSet(Lambda(n, 4*n + 3), S.Integers),
        ImageSet(Lambda(n, 4*n + Rational(7, 3)), S.Integers),
        ImageSet(Lambda(n, 4*n + Rational(5, 3)), S.Integers),
        ImageSet(Lambda(n, 4*n + Rational(11, 3)), S.Integers),
        ImageSet(Lambda(n, 4*n + Rational(1, 3)), S.Integers)))

    # 断言：求解 cos(9*x) = 0 的实数解集，与给定的并集相等
    assert dumeq(solveset(cos(9*x)), Union(
        ImageSet(Lambda(n, 2*n*pi/9 + pi/18), S.Integers),
        ImageSet(Lambda(n, 2*n*pi/9 + pi/6), S.Integers)))
    # 检查解集是否与给定的 Union 对象相等，用于求解 sin(8*x) + cot(12*x) = 0 在实数集合中的解
    assert dumeq(solveset(sin(8*x) + cot(12*x), x, S.Reals), Union(
        # Lambda 表达式生成的 ImageSet，表示解集中的一部分，格式为 n*pi/2 + pi/8，其中 n 属于整数集合
        ImageSet(Lambda(n, n*pi/2 + pi/8), S.Integers),
        ImageSet(Lambda(n, n*pi/2 + 3*pi/8), S.Integers),
        ImageSet(Lambda(n, n*pi/2 + 5*pi/16), S.Integers),
        ImageSet(Lambda(n, n*pi/2 + 3*pi/16), S.Integers),
        ImageSet(Lambda(n, n*pi/2 + 7*pi/16), S.Integers),
        ImageSet(Lambda(n, n*pi/2 + pi/16), S.Integers)))
    
    # 这是唯一仍需通过 _solve_trig2() 解决的 solveset 测试，所有其他情况均由改进后的 _solve_trig1 处理
    assert dumeq(solveset_real(2*cos(x)*cos(2*x) - 1, x),
        Union(
            # Lambda 表达式生成的 ImageSet，表示解集中的一部分，包含复杂的数学表达式
            ImageSet(Lambda(n, 2*n*pi + 2*atan(sqrt(-2*2**Rational(1, 3)*(67 +
                      9*sqrt(57))**Rational(2, 3) + 8*2**Rational(2, 3) + 11*(67 +
                      9*sqrt(57))**Rational(1, 3))/(3*(67 + 9*sqrt(57))**Rational(1, 6)))), S.Integers),
            # Lambda 表达式生成的 ImageSet，表示解集中的一部分，包含复杂的数学表达式加上 2*pi
            ImageSet(Lambda(n, 2*n*pi - 2*atan(sqrt(-2*2**Rational(1, 3)*(67 +
                      9*sqrt(57))**Rational(2, 3) + 8*2**Rational(2, 3) + 11*(67 +
                      9*sqrt(57))**Rational(1, 3))/(3*(67 + 9*sqrt(57))**Rational(1, 6))) +
                      2*pi), S.Integers)))
    
    # issue #16870 的测试，简化解集中 sin(x/180*pi) - 1/2 = 0 在实数集合中的解
    assert dumeq(simplify(solveset(sin(x/180*pi) - S.Half, x, S.Reals)), Union(
        # Lambda 表达式生成的 ImageSet，表示解集中的一部分，格式为 360*n + 150，其中 n 属于整数集合
        ImageSet(Lambda(n, 360*n + 150), S.Integers),
        # Lambda 表达式生成的 ImageSet，表示解集中的一部分，格式为 360*n + 30，其中 n 属于整数集合
        ImageSet(Lambda(n, 360*n + 30), S.Integers)))
def test_solve_trig_hyp_by_inversion():
    n = Dummy('n')
    # 测试解决 sin(2*x + 3) - 1/2 = 0 的实数解
    assert solveset_real(sin(2*x + 3) - S(1)/2, x).dummy_eq(Union(
        # 使用整数 n 构建的集合，其中 n 可取任意整数
        ImageSet(Lambda(n, n*pi - S(3)/2 + 13*pi/12), S.Integers),
        ImageSet(Lambda(n, n*pi - S(3)/2 + 17*pi/12), S.Integers)))
    # 测试解决 sin(2*x + 3) - 1/2 = 0 的复数解
    assert solveset_complex(sin(2*x + 3) - S(1)/2, x).dummy_eq(Union(
        # 使用整数 n 构建的集合，其中 n 可取任意整数
        ImageSet(Lambda(n, n*pi - S(3)/2 + 13*pi/12), S.Integers),
        ImageSet(Lambda(n, n*pi - S(3)/2 + 17*pi/12), S.Integers)))
    # 测试解决 tan(x) - tan(pi/10) = 0 的实数解
    assert solveset_real(tan(x) - tan(pi/10), x).dummy_eq(
        # 使用整数 n 构建的集合，其中 n 可取任意整数
        ImageSet(Lambda(n, n*pi + pi/10), S.Integers))
    # 测试解决 tan(x) - tan(pi/10) = 0 的复数解
    assert solveset_complex(tan(x) - tan(pi/10), x).dummy_eq(
        # 使用整数 n 构建的集合，其中 n 可取任意整数
        ImageSet(Lambda(n, n*pi + pi/10), S.Integers))

    # 测试解决 3*cosh(2*x) - 5 = 0 的实数解
    assert solveset_real(3*cosh(2*x) - 5, x) == FiniteSet(
        # 实数解的有限集合
        -acosh(S(5)/3)/2, acosh(S(5)/3)/2)
    # 测试解决 3*cosh(2*x) - 5 = 0 的复数解
    assert solveset_complex(3*cosh(2*x) - 5, x).dummy_eq(Union(
        # 使用整数 n 和虚数单位构建的集合，其中 n 可取任意整数
        ImageSet(Lambda(n, n*I*pi - acosh(S(5)/3)/2), S.Integers),
        ImageSet(Lambda(n, n*I*pi + acosh(S(5)/3)/2), S.Integers)))
    # 测试解决 sinh(x - 3) - 2 = 0 的实数解
    assert solveset_real(sinh(x - 3) - 2, x) == FiniteSet(
        # 实数解的有限集合
        asinh(2) + 3)
    # 测试解决 sinh(x - 3) - 2 = 0 的复数解
    assert solveset_complex(sinh(x - 3) - 2, x).dummy_eq(Union(
        # 使用整数 n 和虚数单位构建的集合，其中 n 可取任意整数
        ImageSet(Lambda(n, 2*n*I*pi + asinh(2) + 3), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi - asinh(2) + 3 + I*pi), S.Integers)))

    # 测试解决 cos(sinh(x)) - cos(pi/12) = 0 的实数解
    assert solveset_real(cos(sinh(x))-cos(pi/12), x).dummy_eq(Union(
        # 使用整数 n 构建的集合，其中 n 可取任意整数
        ImageSet(Lambda(n, asinh(2*n*pi + pi/12)), S.Integers),
        ImageSet(Lambda(n, asinh(2*n*pi + 23*pi/12)), S.Integers)))
    # 在区间 [2, 3] 上测试解决 cos(sinh(x)) - cos(pi/12) = 0 的实数解
    assert solveset(cos(sinh(x))-cos(pi/12), x, Interval(2,3)) == \
        # 实数解的有限集合
        FiniteSet(asinh(23*pi/12), asinh(25*pi/12))
    # 测试解决 cosh(x**2-1) - 2 = 0 的实数解
    assert solveset_real(cosh(x**2-1)-2, x) == FiniteSet(
        # 实数解的有限集合
        -sqrt(1 + acosh(2)), sqrt(1 + acosh(2)))

    # 下面是一系列无解的测试
    assert solveset_real(sin(x) - 2, x) == S.EmptySet   # issue #17334
    assert solveset_real(cos(x) + 2, x) == S.EmptySet
    assert solveset_real(sec(x), x) == S.EmptySet
    assert solveset_real(csc(x), x) == S.EmptySet
    assert solveset_real(cosh(x) + 1, x) == S.EmptySet
    assert solveset_real(coth(x), x) == S.EmptySet
    assert solveset_real(sech(x) - 2, x) == S.EmptySet
    assert solveset_real(sech(x), x) == S.EmptySet
    assert solveset_real(tanh(x) + 1, x) == S.EmptySet
    assert solveset_complex(tanh(x), 1) == S.EmptySet
    assert solveset_complex(coth(x), -1) == S.EmptySet
    assert solveset_complex(sech(x), 0) == S.EmptySet
    assert solveset_complex(csch(x), 0) == S.EmptySet

    # 测试解决 abs(csch(x)) - 3 = 0 的实数解
    assert solveset_real(abs(csch(x)) - 3, x) == FiniteSet(-acsch(3), acsch(3))

    # 测试解决 tanh(x**2 - 1) - exp(-9) = 0 的实数解
    assert solveset_real(tanh(x**2 - 1) - exp(-9), x) == FiniteSet(
        # 实数解的有限集合
        -sqrt(atanh(exp(-9)) + 1), sqrt(atanh(exp(-9)) + 1))

    # 测试解决 coth(log(x)) + 2 = 0 的实数解
    assert solveset_real(coth(log(x)) + 2, x) == FiniteSet(exp(-acoth(2)))
    # 测试解决 coth(exp(x)) + 2 = 0 的实数解
    assert solveset_real(coth(exp(x)) + 2, x) == S.EmptySet

    # 测试解决 sinh(x) - I/2 = 0 的复数解
    assert solveset_complex(sinh(x) - I/2, x).dummy_eq(Union(
        # 使用整数 n 和虚数单位构建的集合，其中 n 可取任意整数
        ImageSet(Lambda(n, 2*I*pi*n + 5*I*pi/6), S.Integers),
        ImageSet(Lambda(n, 2*I*pi*n + I*pi/6), S.Integers)))
    # 断言：解决复数解集的求解，对 sinh(x/10) + Rational(3, 4) 进行求解并验证是否与给定的并集相等
    assert solveset_complex(sinh(x/10) + Rational(3, 4), x).dummy_eq(Union(
        # 第一个 ImageSet 表示 lambda 表达式生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, 20*n*I*pi - 10*asinh(S(3)/4)), S.Integers),
        # 第二个 ImageSet 表示 lambda 表达式生成的整数序列乘以特定复数常数加上复数常数
        ImageSet(Lambda(n, 20*n*I*pi + 10*asinh(S(3)/4) + 10*I*pi), S.Integers)))

    # 断言：解决复数解集的求解，对 sech(sqrt(2)*x/3) + 5 进行求解并验证是否与给定的并集相等
    assert solveset_complex(sech(sqrt(2)*x/3) + 5, x).dummy_eq(Union(
        # 第一个 ImageSet 表示 lambda 表达式生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, 3*sqrt(2)*(2*n*I*pi - asech(-5))/2), S.Integers),
        # 第二个 ImageSet 表示 lambda 表达式生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, 3*sqrt(2)*(2*n*I*pi + asech(-5))/2), S.Integers)))

    # 断言：解决复数解集的求解，对 cosh(9*x) 进行求解并验证是否与给定的并集相等
    assert solveset_complex(cosh(9*x), x).dummy_eq(Union(
        # 第一个 ImageSet 表示 lambda 表达式生成的整数序列乘以特定复数常数加上复数常数
        ImageSet(Lambda(n, 2*n*I*pi/9 + I*pi/18), S.Integers),
        # 第二个 ImageSet 表示 lambda 表达式生成的整数序列乘以特定复数常数加上复数常数
        ImageSet(Lambda(n, 2*n*I*pi/9 + I*pi/6), S.Integers)))

    # 将 x**5 - 4*x + 1 中的 x 替换为 coth(z)，并将其作为等式 eq
    eq = (x**5 - 4*x + 1).subs(x, coth(z))
    # 断言：解决 z 的复数解集，对等式 eq 进行求解并验证是否与给定的并集相等
    assert solveset(eq, z, S.Complexes).dummy_eq(Union(
        # 使用 CRootOf(x**5 - 4*x + 1, 0) 的反双曲余切函数生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, n*I*pi + acoth(CRootOf(x**5 - 4*x + 1, 0))), S.Integers),
        # 使用 CRootOf(x**5 - 4*x + 1, 1) 的反双曲余切函数生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, n*I*pi + acoth(CRootOf(x**5 - 4*x + 1, 1))), S.Integers),
        # 使用 CRootOf(x**5 - 4*x + 1, 2) 的反双曲余切函数生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, n*I*pi + acoth(CRootOf(x**5 - 4*x + 1, 2))), S.Integers),
        # 使用 CRootOf(x**5 - 4*x + 1, 3) 的反双曲余切函数生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, n*I*pi + acoth(CRootOf(x**5 - 4*x + 1, 3))), S.Integers),
        # 使用 CRootOf(x**5 - 4*x + 1, 4) 的反双曲余切函数生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, n*I*pi + acoth(CRootOf(x**5 - 4*x + 1, 4))), S.Integers)))
    # 断言：解决 z 的实数解集，对等式 eq 进行求解并验证是否与给定的有限集相等
    assert solveset(eq, z, S.Reals) == FiniteSet(
        acoth(CRootOf(x**5 - 4*x + 1, 0)), acoth(CRootOf(x**5 - 4*x + 1, 2)))

    # 将 ((x - sqrt(3)/2)*(x + 2)).expand() 中的 x 替换为 cos(x)，并将其作为等式 eq
    eq = ((x - sqrt(3)/2)*(x + 2)).expand().subs(x, cos(x))
    # 断言：解决 x 的复数解集，对等式 eq 进行求解并验证是否与给定的并集相等
    assert solveset(eq, x, S.Complexes).dummy_eq(Union(
        # 使用 acos(-2) 的 lambda 表达式生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, 2*n*pi - acos(-2)), S.Integers),
        # 使用 acos(-2) 的 lambda 表达式生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, 2*n*pi + acos(-2)), S.Integers),
        # 使用 pi/6 的 lambda 表达式生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, 2*n*pi + pi/6), S.Integers),
        # 使用 11*pi/6 的 lambda 表达式生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, 2*n*pi + 11*pi/6), S.Integers)))
    # 断言：解决 x 的实数解集，对等式 eq 进行求解并验证是否与给定的并集相等
    assert solveset(eq, x, S.Reals).dummy_eq(Union(
        # 使用 pi/6 的 lambda 表达式生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, 2*n*pi + pi/6), S.Integers),
        # 使用 11*pi/6 的 lambda 表达式生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, 2*n*pi + 11*pi/6), S.Integers)))

    # 断言：解决 (1+sec(sqrt(3)*x+4)**2)/(1-sec(sqrt(3)*x+4)) 的解集，并验证是否与给定的并集相等
    assert solveset((1+sec(sqrt(3)*x+4)**2)/(1-sec(sqrt(3)*x+4))).dummy_eq(Union(
        # 使用 asec(I) 的 lambda 表达式生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, sqrt(3)*(2*n*pi - 4 - asec(I))/3), S.Integers),
        # 使用 asec(I) 的 lambda 表达式生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, sqrt(3)*(2*n*pi - 4 + asec(I))/3), S.Integers),
        # 使用 asec(-I) 的 lambda 表达式生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, sqrt(3)*(2*n*pi - 4 - asec(-I))/3), S.Integers),
        # 使用 asec(-I) 的 lambda 表达式生成的整数序列乘以特定复数常数
        ImageSet(Lambda(n, sqrt(3)*(2*n*pi - 4 + asec(-I))/3), S.Integers)))

    # 断言：使用 tan(3.14*x)**(S(3)/2) - 5.678 在区间 [0, 3] 上的解集，并验证是否与给定的有限集相等
    assert all_close(solveset(tan(3.14*x)**(
def test_old_trig_issues():
    # issues #9606 / #9531:
    # 断言求解 sinh(x) 在实数域中的解为 {0}
    assert solveset(sinh(x), x, S.Reals) == FiniteSet(0)
    
    # 断言求解 sinh(x) 在复数域中的解为 Union(...)，使用 dummy_eq 进行近似比较
    assert solveset(sinh(x), x, S.Complexes).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*I*pi), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi + I*pi), S.Integers)))

    # issues #11218 / #18427
    # 断言求解 sin(pi*x) 在实数域中的解为 Union(...)
    assert solveset(sin(pi*x), x, S.Reals).dummy_eq(Union(
        ImageSet(Lambda(n, (2*n*pi + pi)/pi), S.Integers),
        ImageSet(Lambda(n, 2*n), S.Integers)))
    
    # 断言求解 sin(pi*x) 的解为 Union(...)，不指定域
    assert solveset(sin(pi*x), x).dummy_eq(Union(
        ImageSet(Lambda(n, (2*n*pi + pi)/pi), S.Integers),
        ImageSet(Lambda(n, 2*n), S.Integers)))

    # issue #17543
    # 断言求解 I*cot(8*x - 8*E) 的解为 ImageSet(...)
    assert solveset(I*cot(8*x - 8*E), x).dummy_eq(
        ImageSet(Lambda(n, pi*n/8 - 13*pi/16 + E), S.Integers))

    # issue #20798
    # 断言 solveset(cos(2*x) - 0.5, x, Interval(0, 2*pi)) 的结果近似等于 FiniteSet(...)
    assert all_close(solveset(cos(2*x) - 0.5, x, Interval(0, 2*pi)), FiniteSet(
        0.523598775598299, -0.523598775598299 + pi,
        -0.523598775598299 + 2*pi, 0.523598775598299 + pi))
    
    # 计算解集合 sol，并替换 ret 中的 Dummy 符号 n 为常规 Symbol n，以便进行 all_close 比较
    sol = Union(ImageSet(Lambda(n, n*pi - 0.523598775598299), S.Integers),
                ImageSet(Lambda(n, n*pi + 0.523598775598299), S.Integers))
    ret = solveset(cos(2*x) - 0.5, x, S.Reals)
    ret = ret.subs(ret.atoms(Dummy).pop(), n)
    assert all_close(ret, sol)
    
    # 求解 cos(2*x) - 0.5 在复数域中的解，并进行替换 Dummy 符号 n
    ret = solveset(cos(2*x) - 0.5, x, S.Complexes)
    ret = ret.subs(ret.atoms(Dummy).pop(), n)
    assert all_close(ret, sol)

    # issue #21296 / #17667
    # 断言 solveset(tan(x)-sqrt(2), x, Interval(0, pi/2)) 的结果为 FiniteSet(...)
    assert solveset(tan(x)-sqrt(2), x, Interval(0, pi/2)) == FiniteSet(atan(sqrt(2)))
    
    # 断言 solveset(tan(x)-pi, x, Interval(0, pi/2)) 的结果为 FiniteSet(...)
    assert solveset(tan(x)-pi, x, Interval(0, pi/2)) == FiniteSet(atan(pi))

    # issue #17667
    # 尚未正常工作：
    # solveset(cos(x)-y, x, Interval(0, pi))
    # 断言 solveset(cos(x)-y, x, S.Reals) 的结果为 Union(...)
    assert solveset(cos(x)-y, x, S.Reals).dummy_eq(
        ConditionSet(x,(S(-1) <= y) & (y <= S(1)), Union(
            ImageSet(Lambda(n, 2*n*pi - acos(y)), S.Integers),
            ImageSet(Lambda(n, 2*n*pi + acos(y)), S.Integers))))

    # issue #17579
    # 有效结果，但交集可能被简化
    # 断言 solveset(sin(log(x)), x, Interval(0,1, True, False)) 的结果为 Union(...)
    assert solveset(sin(log(x)), x, Interval(0,1, True, False)).dummy_eq(
        Union(Intersection(ImageSet(Lambda(n, exp(2*n*pi)), S.Integers), Interval.Lopen(0, 1)),
              Intersection(ImageSet(Lambda(n, exp(2*n*pi + pi)), S.Integers), Interval.Lopen(0, 1))))

    # issue #17334
    # 断言 solveset(sin(x) - sin(1), x, S.Reals) 的结果为 Union(...)
    assert solveset(sin(x) - sin(1), x, S.Reals).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*pi + 1), S.Integers),
        ImageSet(Lambda(n, 2*n*pi - 1 + pi), S.Integers)))
    
    # 断言 solveset(sin(x) - sqrt(5)/3, x, S.Reals) 的结果为 Union(...)
    assert solveset(sin(x) - sqrt(5)/3, x, S.Reals).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*pi + asin(sqrt(5)/3)), S.Integers),
        ImageSet(Lambda(n, 2*n*pi - asin(sqrt(5)/3) + pi), S.Integers)))
    
    # 断言 solveset(sinh(x)-cosh(2), x, S.Reals) 的结果为 FiniteSet(...)
    assert solveset(sinh(x)-cosh(2), x, S.Reals) == FiniteSet(asinh(cosh(2)))

    # issue 9825
    # 使用 SymPy 解方程 tan(x) = y 在实数域上，检查解是否等价于给定的条件集合
    assert solveset(Eq(tan(x), y), x, domain=S.Reals).dummy_eq(
        ConditionSet(x, (-oo < y) & (y < oo),
                     ImageSet(Lambda(n, n*pi + atan(y)), S.Integers)))
    
    # 创建一个实数符号 r，并使用 SymPy 解方程 tan(x) = r 在实数域上，检查解是否等价于给定的整数集合
    r = Symbol('r', real=True)
    assert solveset(Eq(tan(x), r), x, domain=S.Reals).dummy_eq(
        ImageSet(Lambda(n, n*pi + atan(r)), S.Integers))
# 定义用于解决双曲函数方程的测试函数
def test_solve_hyperbolic():
    # 使用 Dummy 符号创建一个虚拟变量 n
    n = Dummy('n')
    # 验证 sinh(x) + cosh(x) 的解为空集
    assert solveset(sinh(x) + cosh(x), x) == S.EmptySet
    # 验证 sinh(x) + cos(x) 的解为复数域中满足 cos(x) + sinh(x) == 0 的 x 的集合
    assert solveset(sinh(x) + cos(x), x) == ConditionSet(x,
        Eq(cos(x) + sinh(x), 0), S.Complexes)
    # 验证 sinh(x) + sech(x) 的实数解为对数形式的有限集合
    assert solveset_real(sinh(x) + sech(x), x) == FiniteSet(
        log(sqrt(sqrt(5) - 2)))
    # 验证 cosh(2*x) + 2*sinh(x) - 5 的实数解为对数形式的有限集合
    assert solveset_real(cosh(2*x) + 2*sinh(x) - 5, x) == FiniteSet(
        log(-2 + sqrt(5)), log(1 + sqrt(2)))
    # 验证 (coth(x) + sinh(2*x))/cosh(x) - 3 的实数解为对数形式的有限集合
    assert solveset_real((coth(x) + sinh(2*x))/cosh(x) - 3, x) == FiniteSet(
        log(S.Half + sqrt(5)/2), log(1 + sqrt(2)))
    # 验证 cosh(x)*sinh(x) - 2 的实数解为对数形式的有限集合
    assert solveset_real(cosh(x)*sinh(x) - 2, x) == FiniteSet(
        log(4 + sqrt(17))/2)
    # 验证 sinh(x) + tanh(x) - 1 的实数解为对数形式的有限集合
    assert solveset_real(sinh(x) + tanh(x) - 1, x) == FiniteSet(
        log(sqrt(2)/2 + sqrt(-S(1)/2 + sqrt(2))))

    # 验证解复数域中 sinh(x) + sech(x) 的解集等于多个 Lambda 函数的并集
    assert dumeq(solveset_complex(sinh(x) + sech(x), x), Union(
        ImageSet(Lambda(n, 2*n*I*pi + log(sqrt(-2 + sqrt(5)))), S.Integers),
        ImageSet(Lambda(n, I*(2*n*pi + pi/2) + log(sqrt(2 + sqrt(5)))), S.Integers),
        ImageSet(Lambda(n, I*(2*n*pi + pi) + log(sqrt(-2 + sqrt(5)))), S.Integers),
        ImageSet(Lambda(n, I*(2*n*pi - pi/2) + log(sqrt(2 + sqrt(5)))), S.Integers)))

    # 验证解 cosh(x/15) + cosh(x/5) 的解集等于多个 Lambda 函数的并集
    assert dumeq(solveset(cosh(x/15) + cosh(x/5)), Union(
        ImageSet(Lambda(n, 15*I*(2*n*pi + pi/2)), S.Integers),
        ImageSet(Lambda(n, 15*I*(2*n*pi - pi/2)), S.Integers),
        ImageSet(Lambda(n, 15*I*(2*n*pi - 3*pi/4)), S.Integers),
        ImageSet(Lambda(n, 15*I*(2*n*pi + 3*pi/4)), S.Integers),
        ImageSet(Lambda(n, 15*I*(2*n*pi - pi/4)), S.Integers),
        ImageSet(Lambda(n, 15*I*(2*n*pi + pi/4)), S.Integers)))

    # 验证解 tanh(pi*x) - coth(pi/2*x) 的解集等于多个 Lambda 函数的并集
    assert dumeq(solveset(tanh(pi*x) - coth(pi/2*x)), Union(
        ImageSet(Lambda(n, 2*I*(2*n*pi + pi/2)/pi), S.Integers),
        ImageSet(Lambda(n, 2*I*(2*n*pi - pi/2)/pi), S.Integers)))

    # 验证 cosh(x) + cosh(3*x) - cosh(5*x) 的解集等于条件集合
    assert solveset(cosh(x) + cosh(3*x) - cosh(5*x), x, S.Reals
        ).dummy_eq(ConditionSet(x,
        Eq(cosh(x) + cosh(3*x) - cosh(5*x), 0), S.Reals))
    # 验证 sinh(8*x) + coth(12*x) 的解集等于条件集合
    assert solveset(sinh(8*x) + coth(12*x)).dummy_eq(
        ConditionSet(x, Eq(sinh(8*x) + coth(12*x), 0), S.Complexes))


# 定义用于解决三角函数与双曲函数符号方程的测试函数
def test_solve_trig_hyp_symbolic():
    # 验证 sin(a*x) 的解集等于条件集合
    assert dumeq(solveset(sin(a*x), x), ConditionSet(x, Ne(a, 0), Union(
        ImageSet(Lambda(n, (2*n*pi + pi)/a), S.Integers),
        ImageSet(Lambda(n, 2*n*pi/a), S.Integers))))

    # 验证 cosh(x/a) 的解集等于条件集合
    assert dumeq(solveset(cosh(x/a), x), ConditionSet(x, Ne(a, 0), Union(
        ImageSet(Lambda(n, a*(2*n*I*pi + I*pi/2)), S.Integers),
        ImageSet(Lambda(n, a*(2*n*I*pi + 3*I*pi/2)), S.Integers))))
    # 断言：验证解集 solveset(sin(2*sqrt(3)/3*a**2/(b*pi)*x) + cos(4*sqrt(3)/3*a**2/(b*pi)*x), x) 的值
    assert dumeq(solveset(sin(2*sqrt(3)/3*a**2/(b*pi)*x)
        + cos(4*sqrt(3)/3*a**2/(b*pi)*x), x),
       ConditionSet(x, Ne(b, 0) & Ne(a**2, 0), Union(
           # 图像集：Lambda 函数定义了 n 的表达式，该表达式由 sqrt(3)*pi*b*(2*n*pi + pi/2)/(2*a**2) 组成
           ImageSet(Lambda(n, sqrt(3)*pi*b*(2*n*pi + pi/2)/(2*a**2)), S.Integers),
           # 图像集：Lambda 函数定义了 n 的表达式，该表达式由 sqrt(3)*pi*b*(2*n*pi - 5*pi/6)/(2*a**2) 组成
           ImageSet(Lambda(n, sqrt(3)*pi*b*(2*n*pi - 5*pi/6)/(2*a**2)), S.Integers),
           # 图像集：Lambda 函数定义了 n 的表达式，该表达式由 sqrt(3)*pi*b*(2*n*pi - pi/6)/(2*a**2) 组成
           ImageSet(Lambda(n, sqrt(3)*pi*b*(2*n*pi - pi/6)/(2*a**2)), S.Integers))))

    # 断言：验证解集 solveset(cosh((a**2 + 1)*x) - 3, x) 的值
    assert dumeq(solveset(cosh((a**2 + 1)*x) - 3, x), ConditionSet(
        x, Ne(a**2 + 1, 0), Union(
            # 图像集：Lambda 函数定义了 n 的表达式，该表达式由 (2*n*I*pi - acosh(3))/(a**2 + 1) 组成
            ImageSet(Lambda(n, (2*n*I*pi - acosh(3))/(a**2 + 1)), S.Integers),
            # 图像集：Lambda 函数定义了 n 的表达式，该表达式由 (2*n*I*pi + acosh(3))/(a**2 + 1) 组成
            ImageSet(Lambda(n, (2*n*I*pi + acosh(3))/(a**2 + 1)), S.Integers))))

    # 符号 ar 定义为实数
    ar = Symbol('ar', real=True)
    # 断言：验证求解 cosh((ar**2 + 1)*x) - 2 = 0 的结果
    assert solveset(cosh((ar**2 + 1)*x) - 2, x, S.Reals) == FiniteSet(
        # 有限集：解集包括 -acosh(2)/(ar**2 + 1) 和 acosh(2)/(ar**2 + 1)
        -acosh(2)/(ar**2 + 1), acosh(2)/(ar**2 + 1))

    # 断言：验证简化后的 solveset(cot((1 + I)*x) - cot((3 + 3*I)*x), x) 的值
    assert dumeq(simplify(solveset(cot((1 + I)*x) - cot((3 + 3*I)*x), x)), Union(
        # 图像集：Lambda 函数定义了 n 的表达式，该表达式由 pi*(1 - I)*(4*n + 1)/4 组成
        ImageSet(Lambda(n, pi*(1 - I)*(4*n + 1)/4), S.Integers),
        # 图像集：Lambda 函数定义了 n 的表达式，该表达式由 pi*(1 - I)*(4*n - 1)/4 组成
        ImageSet(Lambda(n, pi*(1 - I)*(4*n - 1)/4), S.Integers)))
# 定义测试函数 test_issue_9616，用于验证数学方程解的正确性
def test_issue_9616():
    # 断言解决 sinh(x) + tanh(x) - 1 = 0 方程，返回的解集合应与预期的并集相等
    assert dumeq(solveset(sinh(x) + tanh(x) - 1, x), Union(
        # 第一个解集合：表示为 Lambda 表达式的整数集合乘以 2*pi*i，并加上对数部分
        ImageSet(Lambda(n, 2*n*I*pi + log(sqrt(2)/2 + sqrt(-S.Half + sqrt(2)))), S.Integers),
        # 第二个解集合：Lambda 表达式的整数集合乘以 2*pi*i，加上 atan 和对数部分
        ImageSet(Lambda(n, I*(2*n*pi - atan(sqrt(2)*sqrt(S.Half + sqrt(2))) + pi)
            + log(sqrt(1 + sqrt(2)))), S.Integers),
        # 第三个解集合：Lambda 表达式的整数集合乘以 2*pi*i，加上对数部分
        ImageSet(Lambda(n, I*(2*n*pi + pi) + log(-sqrt(2)/2 + sqrt(-S.Half + sqrt(2)))), S.Integers),
        # 第四个解集合：Lambda 表达式的整数集合乘以 2*pi*i，加上 atan 和对数部分
        ImageSet(Lambda(n, I*(2*n*pi - pi + atan(sqrt(2)*sqrt(S.Half + sqrt(2))))
            + log(sqrt(1 + sqrt(2)))), S.Integers)))
    
    # 对 sinh(x) 和 tanh(x) 分别应用 rewrite 方法得到 f1 和 f2
    f1 = (sinh(x)).rewrite(exp)
    f2 = (tanh(x)).rewrite(exp)
    
    # 断言解决 f1 + f2 - 1 = 0 方程，返回的解集合应与预期的并集相等
    assert dumeq(solveset(f1 + f2 - 1, x), Union(
        # 第一个解集合的补集：Lambda 表达式的整数集合乘以 2*pi*i，加上对数部分的补集
        Complement(ImageSet(
            Lambda(n, I*(2*n*pi + pi) + log(-sqrt(2)/2 + sqrt(-S.Half + sqrt(2)))), S.Integers),
            ImageSet(Lambda(n, I*(2*n*pi + pi)/2), S.Integers)),
        # 第二个解集合的补集：Lambda 表达式的整数集合乘以 2*pi*i，加上 atan 和对数部分的补集
        Complement(ImageSet(Lambda(n, I*(2*n*pi - pi + atan(sqrt(2)*sqrt(S.Half + sqrt(2))))
                + log(sqrt(1 + sqrt(2)))), S.Integers),
            ImageSet(Lambda(n, I*(2*n*pi + pi)/2), S.Integers)),
        # 第三个解集合的补集：Lambda 表达式的整数集合乘以 2*pi*i，加上对数部分的补集
        Complement(ImageSet(Lambda(n, I*(2*n*pi - atan(sqrt(2)*sqrt(S.Half + sqrt(2))) + pi)
                + log(sqrt(1 + sqrt(2)))), S.Integers),
            ImageSet(Lambda(n, I*(2*n*pi + pi)/2), S.Integers)),
        # 第四个解集合的补集：Lambda 表达式的整数集合乘以 2*pi*i，加上对数部分的补集
        Complement(
            ImageSet(Lambda(n, 2*n*I*pi + log(sqrt(2)/2 + sqrt(-S.Half + sqrt(2)))), S.Integers),
            ImageSet(Lambda(n, I*(2*n*pi + pi)/2), S.Integers))))

# 定义测试函数 test_solve_invalid_sol，用于验证特定数学方程的解不存在的情况
def test_solve_invalid_sol():
    # 断言 sin(x)/x = 0 方程不存在解集合中包含 0 的实数解
    assert 0 not in solveset_real(sin(x)/x, x)
    # 断言 (exp(x) - 1)/x = 0 方程不存在解集合中包含 0 的复数解

# 标记的测试函数，test_solve_trig_simplified，标志为预期失败
@XFAIL
def test_solve_trig_simplified():
    # 使用 Lambda 函数验证 sin(x) = 0 方程的解集合，应为整数倍数乘以 pi
    n = Dummy('n')
    assert dumeq(solveset_real(sin(x), x),
        imageset(Lambda(n, n*pi), S.Integers))

    # 使用 Lambda 函数验证 cos(x) = 0 方程的解集合，应为整数倍数乘以 pi 加上 pi/2
    assert dumeq(solveset_real(cos(x), x),
        imageset(Lambda(n, n*pi + pi/2), S.Integers))

    # 使用 Lambda 函数验证 cos(x) + sin(x) = 0 方程的解集合，应为整数倍数乘以 pi 减去 pi/4
    assert dumeq(solveset_real(cos(x) + sin(x), x),
        imageset(Lambda(n, n*pi - pi/4), S.Integers))

# 标记的测试函数，test_solve_lambert，标志为预期失败
@XFAIL
def test_solve_lambert():
    # 验证 x*exp(x) - 1 = 0 方程的实数解应为 LambertW(1)
    assert solveset_real(x*exp(x) - 1, x) == FiniteSet(LambertW(1))
    # 验证 exp(x) + x = 0 方程的实数解应为 -LambertW(1)
    assert solveset_real(exp(x) + x, x) == FiniteSet(-LambertW(1))
    # 验证 x + 2**x = 0 方程的实数解应为 -LambertW(log(2))/log(2)

    # issue 4739
    # 验证 3*x + 5 + 2**(-5*x + 3) 方程的实数解
    ans = solveset_real(3*x + 5 + 2**(-5*x + 3), x)
    assert ans == FiniteSet(Rational(-5, 3) +
                            LambertW(-10240*2**Rational(1, 3)*log(2)/3)/(5*log(2)))

    # 验证复杂表达式的实数解
    eq = 2*(3*x + 4)**5 - 6*7**(3*x + 9)
    result = solveset_real(eq, x)
    ans = FiniteSet((log(2401) +
                     5*LambertW(-log(7**(7*3**Rational(1, 5)/5))))/(3*log(7))/-1)
    assert result == ans
    # 验证化简后的实数解与原始结果相同
    assert solveset_real(eq.expand(), x) == result

    # 验证复杂表达式的实数解
    assert solveset_real(5*x - 1 + 3*exp(2 - 7*x), x) == \
        FiniteSet(Rational(1, 5) + LambertW(-21*exp(Rational(3, 5))/5)/7)

    # 验证复杂表达式的实数解
    assert solveset_real(2*x + 5 + log(3*x - 2), x) == \
        FiniteSet(Rational(2, 3) + LambertW(2*exp(Rational(-19, 3))/3)/2)
    # 断言语句，验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    assert solveset_real(3*x + log(4*x), x) == \
        FiniteSet(LambertW(Rational(3, 4))/3)

    # 断言语句，验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    assert solveset_real(x**x - 2) == FiniteSet(exp(LambertW(log(2))))

    # 创建符号变量 a
    a = Symbol('a')
    # 断言语句，验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    assert solveset_real(-a*x + 2*x*log(x), x) == FiniteSet(exp(a/2))
    # 重新定义符号变量 a，并指定其为实数
    a = Symbol('a', real=True)
    # 断言语句，验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    assert solveset_real(a/x + exp(x/2), x) == \
        FiniteSet(2*LambertW(-a/2))
    # 断言语句，验证 solveset_real 函数对于给定表达式的导数的求解结果是否符合预期
    assert solveset_real((a/x + exp(x/2)).diff(x), x) == \
        FiniteSet(4*LambertW(sqrt(2)*sqrt(a)/4))

    # 覆盖测试，验证 solveset_real 函数对于给定表达式的求解结果是否为空集
    assert solveset_real(tanh(x + 3)*tanh(x - 3) - 1, x) is S.EmptySet

    # 断言语句，验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    assert solveset_real((x**2 - 2*x + 1).subs(x, log(x) + 3*x), x) == \
        FiniteSet(LambertW(3*S.Exp1)/3)
    # 断言语句，验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    assert solveset_real((x**2 - 2*x + 1).subs(x, (log(x) + 3*x)**2 - 1), x) == \
        FiniteSet(LambertW(3*exp(-sqrt(2)))/3, LambertW(3*exp(sqrt(2)))/3)
    # 断言语句，验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    assert solveset_real((x**2 - 2*x - 2).subs(x, log(x) + 3*x), x) == \
        FiniteSet(LambertW(3*exp(1 + sqrt(3)))/3, LambertW(3*exp(-sqrt(3) + 1))/3)
    # 断言语句，验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    assert solveset_real(x*log(x) + 3*x + 1, x) == \
        FiniteSet(exp(-3 + LambertW(-exp(3))))
    # 创建等式 eq，并验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    eq = (x*exp(x) - 3).subs(x, x*exp(x))
    assert solveset_real(eq, x) == \
        FiniteSet(LambertW(3*exp(-LambertW(3))))

    # 断言语句，验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    assert solveset_real(3*log(a**(3*x + 5)) + a**(3*x + 5), x) == \
        FiniteSet(-((log(a**5) + LambertW(Rational(1, 3)))/(3*log(a))))
    # 创建正数符号变量 p，并验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    p = symbols('p', positive=True)
    assert solveset_real(3*log(p**(3*x + 5)) + p**(3*x + 5), x) == \
        FiniteSet(
        log((-3**Rational(1, 3) - 3**Rational(5, 6)*I)*LambertW(Rational(1, 3))**Rational(1, 3)/(2*p**Rational(5, 3)))/log(p),
        log((-3**Rational(1, 3) + 3**Rational(5, 6)*I)*LambertW(Rational(1, 3))**Rational(1, 3)/(2*p**Rational(5, 3)))/log(p),
        log((3*LambertW(Rational(1, 3))/p**5)**(1/(3*log(p)))),)  # checked numerically
    # 检查集合的收集，验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    b = Symbol('b')
    eq = 3*log(a**(3*x + 5)) + b*log(a**(3*x + 5)) + a**(3*x + 5)
    assert solveset_real(eq, x) == FiniteSet(
        -((log(a**5) + LambertW(1/(b + 3)))/(3*log(a))))

    # issue 4271 的问题验证，验证 solveset_real 函数对于给定表达式的二阶导数的求解结果是否符合预期
    assert solveset_real((a/x + exp(x/2)).diff(x, 2), x) == FiniteSet(
        6*LambertW((-1)**Rational(1, 3)*a**Rational(1, 3)/3))

    # 断言语句，验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    assert solveset_real(x**3 - 3**x, x) == \
        FiniteSet(-3/log(3)*LambertW(-log(3)/3))
    # 断言语句，验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    assert solveset_real(3**cos(x) - cos(x)**3) == FiniteSet(
        acos(-3*LambertW(-log(3)/3)/log(3)))

    # 断言语句，验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    assert solveset_real(x**2 - 2**x, x) == \
        solveset_real(-x**2 + 2**x, x)

    # 断言语句，验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    assert solveset_real(3*log(x) - x*log(3)) == FiniteSet(
        -3*LambertW(-log(3)/3)/log(3),
        -3*LambertW(-log(3)/3, -1)/log(3))

    # 断言语句，验证 solveset_real 函数对于给定表达式的求解结果是否符合预期
    assert solveset_real(LambertW(2*x) - y) == FiniteSet(
        y*exp(y)/2)
@XFAIL
# 标记测试函数为预期失败的测试
def test_other_lambert():
    # 创建有理数对象 a = 6/5
    a = Rational(6, 5)
    # 断言解集为有限集，包含特定解
    assert solveset_real(x**a - a**x, x) == FiniteSet(
        a, -a*LambertW(-log(a)/a)/log(a))


@_both_exp_pow
# 标记测试函数为同时支持指数和幂的测试
def test_solveset():
    # 定义函数 f
    f = Function('f')
    # 引发值错误，lambda函数解 x + y 的集合时
    raises(ValueError, lambda: solveset(x + y))
    # 断言解 x = 1 时为空集
    assert solveset(x, 1) == S.EmptySet
    # 断言解 f(1)**2 + y + 1 = 0 时的解集
    assert solveset(f(1)**2 + y + 1, f(1)
        ) == FiniteSet(-sqrt(-y - 1), sqrt(-y - 1))
    # 断言解 f(1)**2 - 1 = 0 时的解集，限定在实数域
    assert solveset(f(1)**2 - 1, f(1), S.Reals) == FiniteSet(-1, 1)
    # 断言解 f(1)**2 + 1 = 0 时的解集
    assert solveset(f(1)**2 + 1, f(1)) == FiniteSet(-I, I)
    # 断言解 x - 1 = 0 时的解集
    assert solveset(x - 1, 1) == FiniteSet(x)
    # 断言解 sin(x) - cos(x) = sin(x) 时的解集
    assert solveset(sin(x) - cos(x), sin(x)) == FiniteSet(cos(x))

    # 断言解 0 = 0 时的解集，限定在实数域
    assert solveset(0, domain=S.Reals) == S.Reals
    # 断言解 1 = 0 时的解集
    assert solveset(1) == S.EmptySet
    # 断言解 True = 0 时的解集，限定在实数域，问题10197
    assert solveset(True, domain=S.Reals) == S.Reals  # issue 10197
    # 断言解 False = 0 时的解集，限定在实数域
    assert solveset(False, domain=S.Reals) == S.EmptySet

    # 断言解 exp(x) - 1 = 0 时的解集，限定在实数域
    assert solveset(exp(x) - 1, domain=S.Reals) == FiniteSet(0)
    # 断言解 exp(x) - 1 = 0 时的解集，限定变量为 x，限定在实数域
    assert solveset(exp(x) - 1, x, S.Reals) == FiniteSet(0)
    # 断言解 exp(x) = 1 时的解集，限定变量为 x，限定在实数域
    assert solveset(Eq(exp(x), 1), x, S.Reals) == FiniteSet(0)
    # 断言解 exp(x) - 1 = 1 时的解集，限定变量为 exp(x)，限定在实数域
    assert solveset(exp(x) - 1, exp(x), S.Reals) == FiniteSet(1)
    # 定义索引为 x 的 Indexed 对象 A
    A = Indexed('A', x)
    # 断言解 A - 1 = 1 时的解集，限定变量为 A，限定在实数域
    assert solveset(A - 1, A, S.Reals) == FiniteSet(1)

    # 断言解 x - 1 >= 0 时的解集，限定变量为 x，限定在实数域
    assert solveset(x - 1 >= 0, x, S.Reals) == Interval(1, oo)
    # 断言解 exp(x) - 1 >= 0 时的解集，限定变量为 x，限定在实数域
    assert solveset(exp(x) - 1 >= 0, x, S.Reals) == Interval(0, oo)

    # 断言解 exp(x) - 1 的解集等于 imageset(Lambda(n, 2*I*pi*n), S.Integers)
    assert dumeq(solveset(exp(x) - 1, x), imageset(Lambda(n, 2*I*pi*n), S.Integers))
    # 断言解 exp(x) = 1 的解集等于 imageset(Lambda(n, 2*I*pi*n), S.Integers)
    assert dumeq(solveset(Eq(exp(x), 1), x), imageset(Lambda(n, 2*I*pi*n),
                                                  S.Integers))
    # issue 13825
    # 断言解 x**2 + f(0) + 1 = 0 时的解集
    assert solveset(x**2 + f(0) + 1, x) == {-sqrt(-f(0) - 1), sqrt(-f(0) - 1)}

    # issue 19977
    # 断言解 atan(log(x)) > 0 时的解集，限定变量为 x，限定域为开区间(0, oo)
    assert solveset(atan(log(x)) > 0, x, domain=Interval.open(0, oo)) == Interval.open(1, oo)


@_both_exp_pow
# 标记测试函数为同时支持指数和幂的测试
def test_multi_exp():
    # 定义符号 k1, k2, k3
    k1, k2, k3 = symbols('k1, k2, k3')
    # 断言解 exp(exp(x)) - 5 = 0 时的解集
    assert dumeq(solveset(exp(exp(x)) - 5, x),\
         imageset(Lambda(((k1, n),), I*(2*k1*pi + arg(2*n*I*pi + log(5))) + log(Abs(2*n*I*pi + log(5)))),\
             ProductSet(S.Integers, S.Integers)))
    # 断言解 (d*exp(exp(a*x + b)) + c) = 0 时的解集
    assert dumeq(solveset((d*exp(exp(a*x + b)) + c), x),\
        imageset(Lambda(x, (-b + x)/a), ImageSet(Lambda(((k1, n),), \
            I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))) + log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d))))), \
                ProductSet(S.Integers, S.Integers))))

    # 断言解 (d*exp(exp(exp(a*x + b))) + c) = 0 时的解集
    assert dumeq(solveset((d*exp(exp(exp(a*x + b))) + c), x),\
        imageset(Lambda(x, (-b + x)/a), ImageSet(Lambda(((k2, k1, n),), \
            I*(2*k2*pi + arg(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))) + \
                log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))))) + log(Abs(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + \
                    log(Abs(c/d)))) + log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d))))))), \
                        ProductSet(S.Integers, S.Integers, S.Integers))))
    # 断言语句，用于验证 solveset 函数的返回结果是否与 ImageSet 对象相等
    assert dumeq(solveset((d*exp(exp(exp(exp(a*x + b)))) + c), x),\
        ImageSet(Lambda(x, (-b + x)/a), ImageSet(Lambda(((k3, k2, k1, n),), \
            I*(2*k3*pi + arg(I*(2*k2*pi + arg(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))) + \
                log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))))) + log(Abs(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + \
                    log(Abs(c/d)))) + log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))))))) + log(Abs(I*(2*k2*pi + \
                        arg(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))) + log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))))) + \
                            log(Abs(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))) + log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d))))))))), \
             ProductSet(S.Integers, S.Integers, S.Integers, S.Integers))))
def test__solveset_multi():
    # 导入 _solveset_multi 函数和 Reals 类
    from sympy.solvers.solveset import _solveset_multi
    from sympy.sets import Reals

    # 基本的一元情况:
    assert _solveset_multi([x**2-1], [x], [S.Reals]) == FiniteSet((1,), (-1,))

    # 二元线性系统
    assert _solveset_multi([x+y, x+1], [x, y], [Reals, Reals]) == FiniteSet((-1, 1))
    assert _solveset_multi([x+y, x+1], [y, x], [Reals, Reals]) == FiniteSet((1, -1))
    assert _solveset_multi([x+y, x-y-1], [x, y], [Reals, Reals]) == FiniteSet((S(1)/2, -S(1)/2))
    assert _solveset_multi([x-1, y-2], [x, y], [Reals, Reals]) == FiniteSet((1, 2))
    # dumeq 函数验证多重解集是否相等
    assert dumeq(_solveset_multi([x+y], [x, y], [Reals, Reals]), Union(
            ImageSet(Lambda(((x,),), (x, -x)), ProductSet(Reals)),
            ImageSet(Lambda(((y,),), (-y, y)), ProductSet(Reals))))
    assert _solveset_multi([x+y, x+y+1], [x, y], [Reals, Reals]) == S.EmptySet
    assert _solveset_multi([x+y, x-y, x-1], [x, y], [Reals, Reals]) == S.EmptySet
    assert _solveset_multi([x+y, x-y, x-1], [y, x], [Reals, Reals]) == S.EmptySet

    # 三元系统
    assert _solveset_multi([x+y+z-1, x+y-z-2, x-y-z-3], [x, y, z], [Reals,
        Reals, Reals]) == FiniteSet((2, -S.Half, -S.Half))

    # 非线性系统
    from sympy.abc import theta
    assert _solveset_multi([x**2+y**2-2, x+y], [x, y], [Reals, Reals]) == FiniteSet((-1, 1), (1, -1))
    assert _solveset_multi([x**2-1, y], [x, y], [Reals, Reals]) == FiniteSet((1, 0), (-1, 0))
    assert dumeq(_solveset_multi([x**2-y**2], [x, y], [Reals, Reals]), Union(
            ImageSet(Lambda(((x,),), (x, -Abs(x))), ProductSet(Reals)),
            ImageSet(Lambda(((x,),), (x, Abs(x))), ProductSet(Reals)),
            ImageSet(Lambda(((y,),), (-Abs(y), y)), ProductSet(Reals)),
            ImageSet(Lambda(((y,),), (Abs(y), y)), ProductSet(Reals))))
    assert _solveset_multi([r*cos(theta)-1, r*sin(theta)], [theta, r],
            [Interval(0, pi), Interval(-1, 1)]) == FiniteSet((0, 1), (pi, -1))
    assert _solveset_multi([r*cos(theta)-1, r*sin(theta)], [r, theta],
            [Interval(0, 1), Interval(0, pi)]) == FiniteSet((1, 0))
    assert _solveset_multi([r*cos(theta)-r, r*sin(theta)], [r, theta],
           [Interval(0, 1), Interval(0, pi)]) == Union(
           ImageSet(Lambda(((r,),), (r, 0)),
           ImageSet(Lambda(r, (r,)), Interval(0, 1))),
           ImageSet(Lambda(((theta,),), (0, theta)),
           ImageSet(Lambda(theta, (theta,)), Interval(0, pi))))


def test_conditionset():
    # 测试 solveset 函数是否正确处理方程
    assert solveset(Eq(sin(x)**2 + cos(x)**2, 1), x, domain=S.Reals
        ) is S.Reals
    # 使用 SymPy 解方程 x**2 + x*sin(x) = 1，在实数域内求解，并断言结果与给定条件集合相等
    assert solveset(Eq(x**2 + x*sin(x), 1), x, domain=S.Reals
        ).dummy_eq(ConditionSet(x, Eq(x**2 + x*sin(x) - 1, 0), S.Reals))

    # 使用 SymPy 解方程 -I*(exp(I*x) - exp(-I*x))/2 = 1，在复数域内求解，并断言结果与给定条件集合相等
    assert dumeq(solveset(Eq(-I*(exp(I*x) - exp(-I*x))/2, 1), x
        ), imageset(Lambda(n, 2*n*pi + pi/2), S.Integers))

    # 使用 SymPy 解不等式 x + sin(x) > 1，在实数域内求解，并断言结果与给定条件集合相等
    assert solveset(x + sin(x) > 1, x, domain=S.Reals
        ).dummy_eq(ConditionSet(x, x + sin(x) > 1, S.Reals))

    # 使用 SymPy 解方程 sin(Abs(x)) = x，在实数域内求解，并断言结果与给定条件集合相等
    assert solveset(Eq(sin(Abs(x)), x), x, domain=S.Reals
        ).dummy_eq(ConditionSet(x, Eq(-x + sin(Abs(x)), 0), S.Reals))

    # 使用 SymPy 解方程 y**x - z = 0，在实数域内求解，并断言结果与给定条件集合相等
    assert solveset(y**x-z, x, S.Reals
        ).dummy_eq(ConditionSet(x, Eq(y**x - z, 0), S.Reals))
@XFAIL
# 标记为测试失败的装饰器，用于测试条件集相等性
def test_conditionset_equality():
    ''' Checking equality of different representations of ConditionSet'''
    # 断言解集函数对给定的方程求解得到的结果等于条件集的表示
    assert solveset(Eq(tan(x), y), x) == ConditionSet(x, Eq(tan(x), y), S.Complexes)


def test_solveset_domain():
    # 断言解方程 x**2 - x - 6 = 0 在区间 [0, 无穷) 中的解集为 {3}
    assert solveset(x**2 - x - 6, x, Interval(0, oo)) == FiniteSet(3)
    # 断言解方程 x**2 - 1 = 0 在区间 [0, 无穷) 中的解集为 {1}
    assert solveset(x**2 - 1, x, Interval(0, oo)) == FiniteSet(1)
    # 断言解方程 x**4 - 16 = 0 在区间 [0, 10] 中的解集为 {2}
    assert solveset(x**4 - 16, x, Interval(0, 10)) == FiniteSet(2)


def test_improve_coverage():
    # 使用 solveset 求解 exp(x) + sin(x) = 0 在实数域上的解
    solution = solveset(exp(x) + sin(x), x, S.Reals)
    # 创建一个未解决的条件集对象
    unsolved_object = ConditionSet(x, Eq(exp(x) + sin(x), 0), S.Reals)
    # 断言解 solution 和 unsolved_object 是相等的
    assert solution.dummy_eq(unsolved_object)


def test_issue_9522():
    # 创建两个方程对象
    expr1 = Eq(1/(x**2 - 4) + x, 1/(x**2 - 4) + 2)
    expr2 = Eq(1/x + x, 1/x)

    # 断言 solveset 对 expr1 在实数域上的解集为空集
    assert solveset(expr1, x, S.Reals) is S.EmptySet
    # 断言 solveset 对 expr2 在实数域上的解集为空集
    assert solveset(expr2, x, S.Reals) is S.EmptySet


def test_solvify():
    # 断言 solvify 对 x**2 + 10 = 0 的解集为空列表
    assert solvify(x**2 + 10, x, S.Reals) == []
    # 断言 solvify 对 x**3 + 1 = 0 在复数域上的解集为 [-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2]
    assert solvify(x**3 + 1, x, S.Complexes) == [-1, S.Half - sqrt(3)*I/2, S.Half + sqrt(3)*I/2]
    # 断言 solvify 对 log(x) = 0 在实数域上的解集为 [1]
    assert solvify(log(x), x, S.Reals) == [1]
    # 断言 solvify 对 cos(x) = 0 在实数域上的解集为 [pi/2, 3*pi/2]
    assert solvify(cos(x), x, S.Reals) == [pi/2, pi*Rational(3, 2)]
    # 断言 solvify 对 sin(x) + 1 = 0 在实数域上的解集为 [3*pi/2]
    assert solvify(sin(x) + 1, x, S.Reals) == [pi*Rational(3, 2)]
    # 使用 lambda 函数来断言 solvify 对 sin(exp(x)) 抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: solvify(sin(exp(x)), x, S.Complexes))


def test_solvify_piecewise():
    # 创建四个分段函数对象
    p1 = Piecewise((0, x < -1), (x**2, x <= 1), (log(x), True))
    p2 = Piecewise((0, x < -10), (x**2 + 5*x - 6, x >= -9))
    p3 = Piecewise((0, Eq(x, 0)), (x**2/Abs(x), True))
    p4 = Piecewise((0, Eq(x, pi)), ((x - pi)/sin(x), True))

    # 断言 solvify 对 p1 在实数域上的解集为 [0]
    assert solvify(p1, x, S.Reals) == [0]
    # 断言 solvify 对 p2 在实数域上的解集为 [-6, 1]
    assert solvify(p2, x, S.Reals) == [-6, 1]
    # 断言 solvify 对 p3 在实数域上的解集为 [0]
    assert solvify(p3, x, S.Reals) == [0]
    # 断言 solvify 对 p4 在实数域上的解集为 [pi]
    assert solvify(p4, x, S.Reals) == [pi]


def test_abs_invert_solvify():
    # 对于正数 x，断言 solvify 对 sin(|x|) 在实数域上的解集为 [0, pi]
    x = Symbol('x', positive=True)
    assert solvify(sin(Abs(x)), x, S.Reals) == [0, pi]
    # 对于一般的 x，断言 solvify 对 sin(|x|) 在实数域上的解集为空
    x = Symbol('x')
    assert solvify(sin(Abs(x)), x, S.Reals) is None


def test_linear_eq_to_matrix():
    # 断言 linear_eq_to_matrix 对 0 = 0 的转换结果为 (Matrix([[0]]), Matrix([[0]]))
    assert linear_eq_to_matrix(0, x) == (Matrix([[0]]), Matrix([[0]]))
    # 断言 linear_eq_to_matrix 对 x = 1 的转换结果为 (Matrix([[0]]), Matrix([[-1]]))
    assert linear_eq_to_matrix(1, x) == (Matrix([[0]]), Matrix([[-1]]))

    # 整数系数的方程组
    eqns1 = [2*x + y - 2*z - 3, x - y - z, x + y + 3*z - 12]
    eqns2 = [Eq(3*x + 2*y - z, 1), Eq(2*x - 2*y + 4*z, -2), -2*x + y - 2*z]

    # 断言 linear_eq_to_matrix 对 eqns1 的转换结果为特定的矩阵形式
    A, B = linear_eq_to_matrix(eqns1, x, y, z)
    assert A == Matrix([[2, 1, -2], [1, -1, -1], [1, 1, 3]])
    assert B == Matrix([[3], [0], [12]])

    # 断言 linear_eq_to_matrix 对 eqns2 的转换结果为特定的矩阵形式
    A, B = linear_eq_to_matrix(eqns2, x, y, z)
    assert A == Matrix([[3, 2, -1], [2, -2, 4], [-2, 1, -2]])
    assert B == Matrix([[1], [-2], [0]])

    # 纯符号系数的方程组
    eqns3 = [a*b*x + b*y + c*z - d, e*x + d*x + f*y + g*z - h, i*x + j*y + k*z - l]
    # 断言 linear_eq_to_matrix 对 eqns3 的转换结果为特定的矩阵形式
    A, B = linear_eq_to_matrix(eqns3, x, y, z)
    assert A == Matrix([[a*b, b, c], [d + e, f, g], [i, j, k]])
    assert B == Matrix([[d], [h], [l]])

    # 如果没有给定符号，则应该引发错误
    # 调用 linear_eq_to_matrix 函数，并预期引发 ValueError 异常
    raises(ValueError, lambda: linear_eq_to_matrix(eqns3))
    
    # 传递重复的符号列表，预期引发 ValueError 异常
    raises(ValueError, lambda: linear_eq_to_matrix(eqns3, [x, x, y]))
    
    # 在原始表达式中检测到非线性项，预期引发 NonlinearError 异常
    raises(NonlinearError, lambda: linear_eq_to_matrix(Eq(1/x + x, 1/x), [x]))
    raises(NonlinearError, lambda: linear_eq_to_matrix([x**2], [x]))
    raises(NonlinearError, lambda: linear_eq_to_matrix([x*y], [x, y]))
    
    # 使用 Eq 表示方程自动评估（应使用未评估的 Eq）
    raises(ValueError, lambda: linear_eq_to_matrix(Eq(x, x), x))
    raises(ValueError, lambda: linear_eq_to_matrix(Eq(x, x + 1), x))
    
    
    # 如果传递了非符号参数，用户需要负责解释
    assert linear_eq_to_matrix([x], [1/x]) == (Matrix([[0]]), Matrix([[-x]]))
    
    # 处理 issue 15195
    assert linear_eq_to_matrix(x + y*(z*(3*x + 2) + 3), x) == (
        Matrix([[3*y*z + 1]]), Matrix([[-y*(2*z + 3)]]))
    
    # 处理 issue 15312
    assert linear_eq_to_matrix(Matrix(
        [[a*x + b*y - 7], [5*x + 6*y - c]]), x, y) == (
        Matrix([[a, b], [5, 6]]), Matrix([[7], [c]]))
    
    # 处理 issue 25423
    raises(TypeError, lambda: linear_eq_to_matrix([], {x, y}))
    raises(TypeError, lambda: linear_eq_to_matrix([x + y], {x, y}))
    raises(ValueError, lambda: linear_eq_to_matrix({x + y}, (x, y)))
def test_issue_16577():
    assert linear_eq_to_matrix(Eq(a*(2*x + 3*y) + 4*y, 5), x, y) == (
        Matrix([[2*a, 3*a + 4]]), Matrix([[5]]))

def test_issue_10085():
    assert invert_real(exp(x),0,x) == (x, S.EmptySet)

def test_linsolve():
    x1, x2, x3, x4 = symbols('x1, x2, x3, x4')

    # Test for different input forms

    M = Matrix([[1, 2, 1, 1, 7], [1, 2, 2, -1, 12], [2, 4, 0, 6, 4]])
    system1 = A, B = M[:, :-1], M[:, -1]
    Eqns = [x1 + 2*x2 + x3 + x4 - 7, x1 + 2*x2 + 2*x3 - x4 - 12,
            2*x1 + 4*x2 + 6*x4 - 4]

    sol = FiniteSet((-2*x2 - 3*x4 + 2, x2, 2*x4 + 5, x4))
    assert linsolve(Eqns, (x1, x2, x3, x4)) == sol
    assert linsolve(Eqns, *(x1, x2, x3, x4)) == sol
    assert linsolve(system1, (x1, x2, x3, x4)) == sol
    assert linsolve(system1, *(x1, x2, x3, x4)) == sol
    # issue 9667 - symbols can be Dummy symbols
    x1, x2, x3, x4 = symbols('x:4', cls=Dummy)
    assert linsolve(system1, x1, x2, x3, x4) == FiniteSet(
        (-2*x2 - 3*x4 + 2, x2, 2*x4 + 5, x4))

    # raise ValueError for garbage value
    raises(ValueError, lambda: linsolve(Eqns))
    raises(ValueError, lambda: linsolve(x1))
    raises(ValueError, lambda: linsolve(x1, x2))
    raises(ValueError, lambda: linsolve((A,), x1, x2))
    raises(ValueError, lambda: linsolve(A, B, x1, x2))
    raises(ValueError, lambda: linsolve([x1], x1, x1))
    raises(ValueError, lambda: linsolve([x1], (i for i in (x1, x1))))

    # raise ValueError if equations are non-linear in given variables
    raises(NonlinearError, lambda: linsolve([x + y - 1, x ** 2 + y - 3], [x, y]))
    raises(NonlinearError, lambda: linsolve([cos(x) + y, x + y], [x, y]))
    assert linsolve([x + z - 1, x ** 2 + y - 3], [z, y]) == {(-x + 1, -x**2 + 3)}

    # Fully symbolic test
    A = Matrix([[a, b], [c, d]])
    B = Matrix([[e], [g]])
    system2 = (A, B)
    sol = FiniteSet(((-b*g + d*e)/(a*d - b*c), (a*g - c*e)/(a*d - b*c)))
    assert linsolve(system2, [x, y]) == sol

    # No solution
    A = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    B = Matrix([0, 0, 1])
    assert linsolve((A, B), (x, y, z)) is S.EmptySet

    # Issue #10056
    A, B, J1, J2 = symbols('A B J1 J2')
    Augmatrix = Matrix([
        [2*I*J1, 2*I*J2, -2/J1],
        [-2*I*J2, -2*I*J1, 2/J2],
        [0, 2, 2*I/(J1*J2)],
        [2, 0,  0],
    ])

    assert linsolve(Augmatrix, A, B) == FiniteSet((0, I/(J1*J2)))

    # Issue #10121 - Assignment of free variables
    Augmatrix = Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    assert linsolve(Augmatrix, a, b, c, d, e) == FiniteSet((a, 0, c, 0, e))
    # raises(IndexError, lambda: linsolve(Augmatrix, a, b, c))  # This line is commented out in the original code

    x0, x1, x2, _x0 = symbols('tau0 tau1 tau2 _tau0')
    assert linsolve(Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])
        ) == FiniteSet((x0, 0, x1, _x0, x2))
    x0, x1, x2, _x0 = symbols('tau00 tau01 tau02 tau0')
    assert linsolve(Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])
        ) == FiniteSet((x0, 0, x1, _x0, x2))
    # 定义符号变量 x0, x1, x2, _x0，并将其赋值给符号对象
    x0, x1, x2, _x0 = symbols('tau00 tau01 tau02 tau1')
    # 使用 linsolve 函数解线性方程组，验证是否等于给定的有限集合
    assert linsolve(Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])
        ) == FiniteSet((x0, 0, x1, _x0, x2))
    
    # 符号可以作为生成器使用
    # 定义符号变量 x0, x2, x4，并将其赋值给符号对象
    x0, x2, x4 = symbols('x0, x2, x4')
    # 使用 linsolve 函数解增广矩阵方程组，验证是否等于给定的有限集合
    assert linsolve(Augmatrix, numbered_symbols('x')
        ) == FiniteSet((x0, 0, x2, 0, x4))
    
    # 修改增广矩阵的最后一个元素为 x0
    Augmatrix[-1, -1] = x0
    
    # 使用 Dummy 来避免符号冲突；名称可能会冲突，但符号不会
    # 修改增广矩阵的最后一个元素为 _x0
    Augmatrix[-1, -1] = symbols('_x0')
    # 使用 linsolve 函数解增广矩阵方程组，验证自由符号的数量是否为 4
    assert len(linsolve(
        Augmatrix, numbered_symbols('x', cls=Dummy)).free_symbols) == 4
    
    # Issue #12604
    # 定义一个函数 f
    f = Function('f')
    # 使用 linsolve 函数解方程 f(x) - 5 = 0，验证是否等于给定的有限集合
    assert linsolve([f(x) - 5], f(x)) == FiniteSet((5,))
    
    # Issue #14860
    # 导入物理单位和符号，定义 kN 为 kilo*newton
    from sympy.physics.units import meter, newton, kilo
    kN = kilo*newton
    # 定义方程组 Eqns
    Eqns = [8*kN + x + y, 28*kN*meter + 3*x*meter]
    # 使用 linsolve 函数解方程组 Eqns，验证是否等于给定的集合
    assert linsolve(Eqns, x, y) == {
            (kilo*newton*Rational(-28, 3), kN*Rational(4, 3))}
    
    # linsolve 不允许扩展（实际或实现），以消除奇异性，但会消除线性项
    # 验证 linsolve 函数解线性方程组是否等于给定的集合
    assert linsolve([Eq(x, x + y)], [x, y]) == {(x, 0)}
    # 验证 linsolve 函数解非线性方程组是否会抛出 NonlinearError 异常
    raises(NonlinearError, lambda:
        linsolve([Eq(x**2, x**2 + y)], [x, y]))
    
    # corner cases
    #
    # XXX: 下面的情况应该与 [0] 的情况相同
    # assert linsolve([], [x]) == {(x,)}
    # 验证 linsolve 函数解空方程组是否是空集
    assert linsolve([], [x]) is S.EmptySet
    # 验证 linsolve 函数解方程组 [0]，验证是否等于给定的集合
    assert linsolve([0], [x]) == {(x,)}
    # 验证 linsolve 函数解方程组 [x]，验证是否等于给定的集合
    assert linsolve([x], [x, y]) == {(0, y)}
    # 验证 linsolve 函数解方程组 [x, 0]，验证是否等于给定的集合
    assert linsolve([x, 0], [x, y]) == {(0, y)}
def test_linsolve_large_sparse():
    #
    # This is mainly a performance test
    #
    
    # 定义一个内部函数用于生成方程组和解
    def _mk_eqs_sol(n):
        # 创建符号变量列表 xs 和 ys
        xs = symbols('x:{}'.format(n))
        ys = symbols('y:{}'.format(n))
        # 合并符号变量为一个列表 syms
        syms = xs + ys
        # 初始化方程列表 eqs 和解 sol
        eqs = []
        sol = (-S.Half,) * n + (S.Half,) * n
        # 构建方程组，每对 xi 和 yi 生成两个方程
        for xi, yi in zip(xs, ys):
            eqs.extend([xi + yi, xi - yi + 1])
        # 返回方程组 eqs，符号列表 syms 和解 sol 的 FiniteSet
        return eqs, syms, FiniteSet(sol)

    # 设置方程组的大小
    n = 500
    # 生成方程组 eqs, 符号列表 syms 和解 sol
    eqs, syms, sol = _mk_eqs_sol(n)
    # 断言求解线性方程组 eqs 关于符号列表 syms 的结果与预期解 sol 相等
    assert linsolve(eqs, syms) == sol


def test_linsolve_immutable():
    # 创建不可变的稠密矩阵 A 和 B
    A = ImmutableDenseMatrix([[1, 1, 2], [0, 1, 2], [0, 0, 1]])
    B = ImmutableDenseMatrix([2, 1, -1])
    # 断言求解线性方程组 [A, B] 关于变量 (x, y, z) 的结果与预期解 (1, 3, -1) 相等
    assert linsolve([A, B], (x, y, z)) == FiniteSet((1, 3, -1))

    # 修改矩阵 A 的值
    A = ImmutableDenseMatrix([[1, 1, 7], [1, -1, 3]])
    # 断言求解线性方程组 A 的结果与预期解 (5, 2) 相等
    assert linsolve(A) == FiniteSet((5, 2))


def test_solve_decomposition():
    # 创建虚拟符号 n
    n = Dummy('n')

    # 定义多个函数 f1 到 f7
    f1 = exp(3*x) - 6*exp(2*x) + 11*exp(x) - 6
    f2 = sin(x)**2 - 2*sin(x) + 1
    f3 = sin(x)**2 - sin(x)
    f4 = sin(x + 1)
    f5 = exp(x + 2) - 1
    f6 = 1/log(x)
    f7 = 1/x

    # 定义多个解集 s1 到 s5
    s1 = ImageSet(Lambda(n, 2*n*pi), S.Integers)
    s2 = ImageSet(Lambda(n, 2*n*pi + pi), S.Integers)
    s3 = ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers)
    s4 = ImageSet(Lambda(n, 2*n*pi - 1), S.Integers)
    s5 = ImageSet(Lambda(n, 2*n*pi - 1 + pi), S.Integers)

    # 断言求解函数 f1 关于变量 x 在实数域上的结果与预期解 {0, log(2), log(3)} 相等
    assert solve_decomposition(f1, x, S.Reals) == FiniteSet(0, log(2), log(3))
    # 断言求解函数 f2 关于变量 x 在实数域上的结果与预期解 s3 相等
    assert dumeq(solve_decomposition(f2, x, S.Reals), s3)
    # 断言求解函数 f3 关于变量 x 在实数域上的结果与预期解 Union(s1, s2, s3) 相等
    assert dumeq(solve_decomposition(f3, x, S.Reals), Union(s1, s2, s3))
    # 断言求解函数 f4 关于变量 x 在实数域上的结果与预期解 Union(s4, s5) 相等
    assert dumeq(solve_decomposition(f4, x, S.Reals), Union(s4, s5))
    # 断言求解函数 f5 关于变量 x 在实数域上的结果与预期解 {-2} 相等
    assert solve_decomposition(f5, x, S.Reals) == FiniteSet(-2)
    # 断言求解函数 f6 关于变量 x 在实数域上的结果为空集
    assert solve_decomposition(f6, x, S.Reals) == S.EmptySet
    # 断言求解函数 f7 关于变量 x 在实数域上的结果为空集
    assert solve_decomposition(f7, x, S.Reals) == S.EmptySet
    # 断言求解函数 x 关于变量 x 在区间 [1, 2] 上的结果为空集
    assert solve_decomposition(x, x, Interval(1, 2)) == S.EmptySet


# nonlinsolve testcases
def test_nonlinsolve_basic():
    # 断言对于空的非线性方程组，结果为空集
    assert nonlinsolve([],[]) == S.EmptySet
    # 断言对于只有变量 x 的非线性方程组，结果为空集
    assert nonlinsolve([],[x, y]) == S.EmptySet

    # 创建一个包含一个方程的非线性方程组 system
    system = [x, y - x - 5]
    # 断言对于只有一个变量 x 的非线性方程组，结果为 {(0, y)}
    assert nonlinsolve([x],[x, y]) == FiniteSet((0, y))
    # 断言对于包含两个方程的非线性方程组 system 关于变量 y 的结果为空集
    assert nonlinsolve(system, [y]) == S.EmptySet
    # 断言求解非线性方程 sin(x) - 1 关于变量 x 的结果与预期解 s3 相等
    soln = (ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers),)
    assert dumeq(nonlinsolve([sin(x) - 1], [x]), FiniteSet(tuple(soln)))
    # 断言求解非线性方程组 [sin(x), y - 1] 关于变量 x 和 y 的结果与预期解 soln 相等
    soln = ((ImageSet(Lambda(n, 2*n*pi + pi), S.Integers), 1),
            (ImageSet(Lambda(n, 2*n*pi), S.Integers), 1))
    assert dumeq(nonlinsolve([sin(x), y - 1], [x, y]), FiniteSet(*soln))
    # 断言求解非线性方程 x**2 - 1 关于变量 x 的结果为 {(-1,), (1,)}
    assert nonlinsolve([x**2 - 1], [x]) == FiniteSet((-1,), (1,))

    # 断言求解非线性方程组 [x - y, 0] 关于变量 x 和 y 的结果与预期解 soln 相等
    soln = FiniteSet((y, y))
    assert nonlinsolve([x - y, 0], x, y) == soln
    assert nonlinsolve([0, x - y], x, y) == soln
    assert nonlinsolve([x - y, x - y], x, y) == soln
    # 断言求解非线性方程组 [x, 0] 关于变量 x 和 y 的结果为 {(0, y)}
    assert nonlinsolve([x, 0], x, y) == FiniteSet((0, y))
    # 断言求解非线性方程组 [f(x), 0] 关于函数 f(x) 和 y 的结果为 {(0, y)}
    f = Function('f')
    assert nonlinsolve([f(x), 0], f(x), y) == FiniteSet((0, y))
    # 断言求解非线性方程组 [f(x), 0] 关于函数 f(x) 和 f(y) 的结果为 {(0, f(y))}
    assert nonlinsolve([f(x), 0], f
    # 使用非线性方程组求解器求解方程组 x^2 - 1 = 0, sin(x) = 0 的交集，预期结果是空集的有限集
    assert nonlinsolve([x**2 - 1], [sin(x)]) == FiniteSet((S.EmptySet,))
    
    # 使用非线性方程组求解器求解方程组 x^2 - 1 = 0, sin(x) = 0 的交集，预期结果是空集的有限集
    # 第二个参数 sin(x) 传递方式有误，应该是一个列表或元组
    assert nonlinsolve([x**2 - 1], sin(x)) == FiniteSet((S.EmptySet,))
    
    # 使用非线性方程组求解器求解方程组 x^2 - 1 = 0, 常数 1 = 0 的交集，预期结果是 x^2 的有限集
    assert nonlinsolve([x**2 - 1], 1) == FiniteSet((x**2,))
    
    # 使用非线性方程组求解器求解方程组 x^2 - 1 = 0, x + y = 0 的交集，预期结果是空集的有限集
    assert nonlinsolve([x**2 - 1], x + y) == FiniteSet((S.EmptySet,))
    
    # 使用非线性方程组求解器求解方程组 1 = x + y, 1 = -x + y - 1, 1 = -x + y - 1 的交集，预期结果是 (-1/2, 3/2) 的有限集
    assert nonlinsolve([Eq(1, x + y), Eq(1, -x + y - 1), Eq(1, -x + y - 1)], x, y) == FiniteSet((-S.Half, 3*S.Half))
def test_nonlinsolve_abs():
    # 创建包含两个解的有限集合，每个解是一个元组 (y, y) 或 (-y, y)
    soln = FiniteSet((y, y), (-y, y))
    # 断言非线性方程组解的正确性
    assert nonlinsolve([Abs(x) - y], x, y) == soln


def test_raise_exception_nonlinsolve():
    # 断言调用 nonlinsolve 函数时会抛出 IndexError 异常，因为参数列表为空
    raises(IndexError, lambda: nonlinsolve([x**2 -1], []))
    # 断言调用 nonlinsolve 函数时会抛出 ValueError 异常，因为缺少一个参数
    raises(ValueError, lambda: nonlinsolve([x**2 -1]))


def test_trig_system():
    # 断言解三角方程组 sin(x) - 1 和 cos(x) - 1 会得到空集
    assert nonlinsolve([sin(x) - 1, cos(x) -1 ], x) == S.EmptySet
    # 定义预期的解集合，包含 Lambda 表达式和整数集合
    soln1 = (ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers),)
    soln = FiniteSet(soln1)
    # 断言非线性方程组的解与预期的解集合相等
    assert dumeq(nonlinsolve([sin(x) - 1, cos(x)], x), soln)


@XFAIL
def test_trig_system_fail():
    # 由于 solveset 的三角函数求解器不够智能，此测试失败
    sys = [x + y - pi/2, sin(x) + sin(y) - 1]
    # solveset 返回 sin(x) + sin(y) - 1 的条件集合
    soln_1 = (ImageSet(Lambda(n, n*pi + pi/2), S.Integers),
              ImageSet(Lambda(n, n*pi), S.Integers))
    soln_1 = FiniteSet(soln_1)
    soln_2 = (ImageSet(Lambda(n, n*pi), S.Integers),
              ImageSet(Lambda(n, n*pi+ pi/2), S.Integers))
    soln_2 = FiniteSet(soln_2)
    soln = soln_1 + soln_2
    # 断言非线性方程组的解与预期的解集合相等
    assert dumeq(nonlinsolve(sys, [x, y]), soln)

    # 添加更多的测试用例，参考 http://www.vitutor.com/geometry/trigonometry/equations_systems.html#uno
    sys = [sin(x) + sin(y) - (sqrt(3)+1)/2, sin(x) - sin(y) - (sqrt(3) - 1)/2]
    soln_x = Union(ImageSet(Lambda(n, 2*n*pi + pi/3), S.Integers),
                   ImageSet(Lambda(n, 2*n*pi + pi*Rational(2, 3)), S.Integers))
    soln_y = Union(ImageSet(Lambda(n, 2*n*pi + pi/6), S.Integers),
                   ImageSet(Lambda(n, 2*n*pi + pi*Rational(5, 6)), S.Integers))
    # 断言非线性方程组的解与预期的解集合相等
    assert dumeq(nonlinsolve(sys, [x, y]), FiniteSet((soln_x, soln_y)))


def test_nonlinsolve_positive_dimensional():
    x, y, a, b, c, d = symbols('x, y, a, b, c, d', extended_real=True)
    # 断言非线性方程组的解与预期的解集合相等
    assert nonlinsolve([x*y, x*y - x], [x, y]) == FiniteSet((0, y))

    system = [a**2 + a*c, a - b]
    # 断言非线性方程组的解与预期的解集合相等
    assert nonlinsolve(system, [a, b]) == FiniteSet((0, 0), (-c, -c))
    # 当 symbols = [a, b, c] 时，只打印独立的解 (a : -c ,b : -c)

    eq1 =  a + b + c + d
    eq2 = a*b + b*c + c*d + d*a
    eq3 = a*b*c + b*c*d + c*d*a + d*a*b
    eq4 = a*b*c*d - 1
    system = [eq1, eq2, eq3, eq4]
    sol1 = (-1/d, -d, 1/d, FiniteSet(d) - FiniteSet(0))
    sol2 = (1/d, -d, -1/d, FiniteSet(d) - FiniteSet(0))
    soln = FiniteSet(sol1, sol2)
    # 断言非线性方程组的解与预期的解集合相等
    assert nonlinsolve(system, [a, b, c, d]) == soln

    # 断言非线性方程组的解与预期的解集合相等
    assert nonlinsolve([x**4 - 3*x**2 + y*x, x*z**2, y*z - 1], [x, y, z]) == \
           {(0, 1/z, z)}


def test_nonlinsolve_polysys():
    x, y, z = symbols('x, y, z', real=True)
    # 断言解非线性多项式方程组 x**2 + y - 2 和 x**2 + y 为空集
    assert nonlinsolve([x**2 + y - 2, x**2 + y], [x, y]) == S.EmptySet

    s = (-y + 2, y)
    # 断言非线性方程组的解与预期的解集合相等
    assert nonlinsolve([(x + y)**2 - 4, x + y - 2], [x, y]) == FiniteSet(s)

    system = [x**2 - y**2]
    soln_real = FiniteSet((-y, y), (y, y))
    soln_complex = FiniteSet((-Abs(y), y), (Abs(y), y))
    soln = soln_real + soln_complex
    # 断言非线性方程组的解与预期的解集合相等
    assert nonlinsolve(system, [x, y]) == soln
    # 确保非线性方程组求解函数返回的结果与预期解相等
    assert nonlinsolve(system, [x, y]) == soln
    
    # 定义一个包含单个方程 x^2 - y^2 的方程组
    system = [x**2 - y**2]
    # 定义实数解集合
    soln_real = FiniteSet((y, -y), (y, y))
    # 定义复数解集合
    soln_complex = FiniteSet((y, -Abs(y)), (y, Abs(y)))
    # 合并实数和复数解集合
    soln = soln_real + soln_complex
    # 确保按照 [y, x] 顺序求解方程组返回的结果与预期解相等
    assert nonlinsolve(system, [y, x]) == soln
    
    # 定义一个包含两个方程的方程组
    system = [x**2 + y - 3, x - y - 4]
    # 确保按照 (x, y) 和 (y, x) 两种顺序求解方程组得到的结果不相等
    assert nonlinsolve(system, (x, y)) != nonlinsolve(system, (y, x))
    
    # 确保对于给定的方程组，求解结果为空集
    assert nonlinsolve([-x**2 - y**2 + z, -2*x, -2*y, S.One], [x, y, z]) == S.EmptySet
    
    # 确保对于给定的方程组，求解结果为空集
    assert nonlinsolve([x + y + z, S.One, S.One, S.One], [x, y, z]) == S.EmptySet
    
    # 定义一个包含四个方程的方程组
    system = [-x**2*z**2 + x*y*z + y**4, -2*x*z**2 + y*z, x*z + 4*y**3, -2*x**2*z + x*y]
    # 确保求解该方程组返回的结果包含特定的解集合
    assert nonlinsolve(system, [x, y, z]) == FiniteSet((0, 0, z), (x, 0, 0))
# 定义测试函数，用于测试非线性方程组的求解，使用代换方法
def test_nonlinsolve_using_substitution():
    # 定义符号变量，其中 x, y, z, n 均为实数
    x, y, z, n = symbols('x, y, z, n', real=True)
    # 定义第一个方程组
    system = [(x + y)*n - y**2 + 2]
    # 计算 x 的解
    s_x = (n*y - y**2 + 2)/n
    # 计算解集，包含一个解 (-s_x, y)
    soln = (-s_x, y)
    # 断言非线性方程组的解等于解集 FiniteSet(soln)
    assert nonlinsolve(system, [x, y]) == FiniteSet(soln)

    # 定义第二个方程组
    system = [z**2*x**2 - z**2*y**2/exp(x)]
    # 定义实数解集
    soln_real_1 = (y, x, 0)
    soln_real_2 = (-exp(x/2)*Abs(x), x, z)
    soln_real_3 = (exp(x/2)*Abs(x), x, z)
    # 定义复数解集
    soln_complex_1 = (-x*exp(x/2), x, z)
    soln_complex_2 = (x*exp(x/2), x, z)
    # 符号变量列表
    syms = [y, x, z]
    # 计算解集，包含多个解
    soln = FiniteSet(soln_real_1, soln_complex_1, soln_complex_2,
                     soln_real_2, soln_real_3)
    # 断言非线性方程组的解等于解集 soln
    assert nonlinsolve(system, syms) == soln


# 定义测试函数，用于测试包含复数解的非线性方程组
def test_nonlinsolve_complex():
    # 定义虚拟变量 n
    n = Dummy('n')
    # 断言求解给定方程组的结果满足约束
    assert dumeq(nonlinsolve([exp(x) - sin(y), 1/y - 3], [x, y]), {
        (ImageSet(Lambda(n, 2*n*I*pi + log(sin(Rational(1, 3)))), S.Integers), Rational(1, 3))})

    # 定义多个方程组，包含复数解
    system = [exp(x) - sin(y), 1/exp(y) - 3]
    # 断言求解给定方程组的结果满足约束
    assert dumeq(nonlinsolve(system, [x, y]), {
        (ImageSet(Lambda(n, I*(2*n*pi + pi)
                         + log(sin(log(3)))), S.Integers), -log(3)),
        (ImageSet(Lambda(n, I*(2*n*pi + arg(sin(2*n*I*pi - log(3))))
                         + log(Abs(sin(2*n*I*pi - log(3))))), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi - log(3)), S.Integers))})

    # 定义多个方程组，包含复数解
    system = [exp(x) - sin(y), y**2 - 4]
    # 断言求解给定方程组的结果满足约束
    assert dumeq(nonlinsolve(system, [x, y]), {
        (ImageSet(Lambda(n, I*(2*n*pi + pi) + log(sin(2))), S.Integers), -2),
        (ImageSet(Lambda(n, 2*n*I*pi + log(sin(2))), S.Integers), 2)})

    # 定义多个方程组，包含复数解
    system = [exp(x) - 2, y ** 2 - 2]
    # 断言求解给定方程组的结果满足约束
    assert dumeq(nonlinsolve(system, [x, y]), {
        (log(2), -sqrt(2)), (log(2), sqrt(2)),
        (ImageSet(Lambda(n, 2*n*I*pi + log(2)), S.Integers), -sqrt(2)),
        (ImageSet(Lambda(n, 2 * n * I * pi + log(2)), S.Integers), sqrt(2))})


# 定义测试函数，用于测试包含根式解的非线性方程组
def test_nonlinsolve_radical():
    # 断言求解给定方程组的结果等于集合 {(1 - z, 1, z)}
    assert nonlinsolve([sqrt(y) - x - z, y - 1], [x, y, z]) == {(1 - z, 1, z)}


# 定义测试函数，用于测试包含非精确解的非线性方程组
def test_nonlinsolve_inexact():
    # 定义预期的解集
    sol = [(-1.625, -1.375), (1.625, 1.375)]
    # 求解给定方程组，结果与预期解 sol 比较
    res = nonlinsolve([(x + y)**2 - 9, x**2 - y**2 - 0.75], [x, y])
    # 断言所有解的误差小于 1e-9
    assert all(abs(res.args[i][j]-sol[i][j]) < 1e-9
               for i in range(2) for j in range(2))

    # 断言求解结果为空集
    assert nonlinsolve([(x + y)**2 - 9, (x + y)**2 - 0.75], [x, y]) == S.EmptySet

    # 断言求解结果为空集
    assert nonlinsolve([y**2 + (x - 0.5)**2 - 0.0625, 2*x - 1.0, 2*y], [x, y]) == \
           S.EmptySet

    # 求解给定方程组
    res = nonlinsolve([x**2 + y - 0.5, (x + y)**2, log(z)], [x, y, z])
    # 预期的解集
    sol = [(-0.366025403784439, 0.366025403784439, 1),
           (-0.366025403784439, 0.366025403784439, 1),
           (1.36602540378444, -1.36602540378444, 1)]
    # 断言所有解的误差小于 1e-9
    assert all(abs(res.args[i][j]-sol[i][j]) < 1e-9
               for i in range(3) for j in range(3))

    # 求解给定方程组
    res = nonlinsolve([y - x**2, x**5 - x + 1.0], [x, y])
    # 定义一个包含五个复数对的列表，每个复数对由两个元素组成，表示复平面上的点坐标
    sol = [(-1.16730397826142, 1.36259857766493),
           (-0.181232444469876 - 1.08395410131771j, -1.14211129483496 + 0.392895302949911j),
           (-0.181232444469876 + 1.08395410131771j, -1.14211129483496 - 0.392895302949911j),
           (0.764884433600585 - 0.352471546031726j, 0.460812006002492 - 0.539199997693599j),
           (0.764884433600585 + 0.352471546031726j, 0.460812006002492 + 0.539199997693599j)]
    
    # 使用断言检查所有复数对的每个元素的差的绝对值是否小于1e-9
    # 这里的断言确保计算结果与预期解 sol 的每个元素非常接近
    assert all(abs(res.args[i][j] - sol[i][j]) < 1e-9
               for i in range(5) for j in range(2))
@XFAIL
def test_solve_nonlinear_trans():
    # 在解超越方程后，以下测试将会生效
    x, y = symbols('x, y', real=True)
    # 第一个解集，包含 LambertW 函数的应用
    soln1 = FiniteSet((2*LambertW(y/2), y))
    # 第二个解集，包含负号和平方根的组合
    soln2 = FiniteSet((-x*sqrt(exp(x)), y), (x*sqrt(exp(x)), y))
    # 第三个解集，包含指数函数的应用
    soln3 = FiniteSet((x*exp(x/2), x))
    # 第四个解集，包含 LambertW 函数的应用
    soln4 = FiniteSet(2*LambertW(y/2), y)
    # 断言解决非线性方程组的结果与预期解集 soln1 相等
    assert nonlinsolve([x**2 - y**2/exp(x)], [x, y]) == soln1
    # 断言解决非线性方程组的结果与预期解集 soln2 相等
    assert nonlinsolve([x**2 - y**2/exp(x)], [y, x]) == soln2
    # 断言解决非线性方程组的结果与预期解集 soln3 相等
    assert nonlinsolve([x**2 - y**2/exp(x)], [y, x]) == soln3
    # 断言解决非线性方程组的结果与预期解集 soln4 相等
    assert nonlinsolve([x**2 - y**2/exp(x)], [x, y]) == soln4


def test_nonlinsolve_issue_25182():
    # 定义符号变量
    a1, b1, c1, ca, cb, cg = symbols('a1, b1, c1, ca, cb, cg')
    # 定义非线性方程组的各个方程
    eq1 = a1*a1 + b1*b1 - 2.*a1*b1*cg - c1*c1
    eq2 = a1*a1 + c1*c1 - 2.*a1*c1*cb - b1*b1
    eq3 = b1*b1 + c1*c1 - 2.*b1*c1*ca - a1*a1
    # 断言解决非线性方程组的结果与预期解集相等
    assert nonlinsolve([eq1, eq2, eq3], [c1, cb, cg]) == FiniteSet(
        # 第一个解集
        (1.0*b1*ca - 1.0*sqrt(a1**2 + b1**2*ca**2 - b1**2),
         -1.0*sqrt(a1**2 + b1**2*ca**2 - b1**2)/a1,
         -1.0*b1*(ca - 1)*(ca + 1)/a1 + 1.0*ca*sqrt(a1**2 + b1**2*ca**2 - b1**2)/a1),
        # 第二个解集
        (1.0*b1*ca + 1.0*sqrt(a1**2 + b1**2*ca**2 - b1**2),
         1.0*sqrt(a1**2 + b1**2*ca**2 - b1**2)/a1,
         -1.0*b1*(ca - 1)*(ca + 1)/a1 - 1.0*ca*sqrt(a1**2 + b1**2*ca**2 - b1**2)/a1))


def test_issue_14642():
    # 定义符号变量
    x = Symbol('x')
    # 定义多项式表达式
    n1 = 0.5*x**3+x**2+0.5+I  # 在多项式中添加虚数单位 I
    # 解多项式方程
    solution = solveset(n1, x)
    # 断言解的精度
    assert abs(solution.args[0] - (-2.28267560928153 - 0.312325580497716*I)) <= 1e-9
    assert abs(solution.args[1] - (-0.297354141679308 + 1.01904778618762*I)) <= 1e-9
    assert abs(solution.args[2] - (0.580029750960839 - 0.706722205689907*I)) <= 1e-9

    # 在符号表达式中使用 SymPy 的常量 S.Half 和虚数单位 I
    n1 = S.Half*x**3+x**2+S.Half+I
    # 断言解多项式方程的结果与预期结果相等
    assert solveset(n1, x) == res


def test_issue_13961():
    # 定义符号变量列表
    V = (ax, bx, cx, gx, jx, lx, mx, nx, q) = symbols('ax bx cx gx jx lx mx nx q')
    # 定义非线性方程组
    S = (ax*q - lx*q - mx, ax - gx*q - lx, bx*q**2 + cx*q - jx*q - nx, q*(-ax*q + lx*q + mx), q*(-ax + gx*q + lx))
    # 预期的解集
    sol = FiniteSet((lx + mx/q, (-cx*q + jx*q + nx)/q**2, cx, mx/q**2, jx, lx, mx, nx, Complement({q}, {0})),
                    (lx + mx/q, (cx*q - jx*q - nx)/q**2*-1, cx, mx/q**2, jx, lx, mx, nx, Complement({q}, {0})))
    # 断言解决非线性方程组的结果与预期解集相等
    assert nonlinsolve(S, *V) == sol
    # 由于两个解实际上是相同的，因此最好只返回一个解


def test_issue_14541():
    # 解平方根方程的结果
    solutions = solveset(sqrt(-x**2 - 2.0), x)
    # 断言解的精度
    assert abs(solutions.args[0]+1.4142135623731*I) <= 1e-9
    assert abs(solutions.args[1]-1.4142135623731*I) <= 1e-9


def test_issue_13396():
    # 定义符号表达式
    expr = -2*y*exp(-x**2 - y**2)*Abs(x)
    # 预期的解集
    sol = FiniteSet(0)

    # 断言解决符号表达式的结果与预期解集相等，限定 y 属于实数域
    assert solveset(expr, y, domain=S.Reals) == sol

    # 在此解决相关类型的方程
    assert solveset(atan(x**2 - y**2)-pi/2, y, S.Reals) is S.EmptySet


def test_issue_12032():
    # 未完成的测试用例，留空
    pass
    # 解方程 x**4 + x - 1 = 0，并将解集合赋给 sol
    sol = FiniteSet(
        # 第一个解
        -sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
              2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))/2 +
        sqrt(Abs(-2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)) +
                 2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                 2/sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                        2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))))/2,

        # 第二个解
        -sqrt(Abs(-2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)) +
                  2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                  2/sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                         2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))))/2 -
        sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
             2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))/2,

        # 第三个解
        sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
             2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))/2 -
        I*sqrt(Abs(-2/sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                          2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) -
                  2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)) +
                  2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))))/2,

        # 第四个解
        sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
             2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))/2 +
        I*sqrt(Abs(-2/sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                          2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) -
                  2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)) +
                  2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))))/2
    )

    # 断言解集合与预期的 sol 相等
    assert solveset(x**4 + x - 1, x) == sol
def test_issue_10876():
    # 检查 solveset 是否正确处理给定的表达式
    assert solveset(1/sqrt(x), x) == S.EmptySet


def test_issue_19050():
    # 检查 nonlinsolve 是否正确解决问题 19050，并移除 TypeError
    assert dumeq(nonlinsolve([x + y, sin(y)], [x, y]),
        FiniteSet((ImageSet(Lambda(n, -2*n*pi), S.Integers), ImageSet(Lambda(n, 2*n*pi), S.Integers)),
             (ImageSet(Lambda(n, -2*n*pi - pi), S.Integers), ImageSet(Lambda(n, 2*n*pi + pi), S.Integers))))
    assert dumeq(nonlinsolve([x + y, sin(y) + cos(y)], [x, y]),
        FiniteSet((ImageSet(Lambda(n, -2*n*pi - 3*pi/4), S.Integers), ImageSet(Lambda(n, 2*n*pi + 3*pi/4), S.Integers)),
            (ImageSet(Lambda(n, -2*n*pi - 7*pi/4), S.Integers), ImageSet(Lambda(n, 2*n*pi + 7*pi/4), S.Integers))))


def test_issue_16618():
    # 检查 nonlinsolve 是否正确解决问题 16618，并验证解是否合理
    eqn = [sin(x)*sin(y), cos(x)*cos(y) - 1]
    # 提供的答案 ans 看起来可疑，因为它只包含三个不同的 Dummys，而不是四个（两个 'x' 的 ImageSets 共享相同的 Dummy）
    ans = FiniteSet((ImageSet(Lambda(n, 2*n*pi), S.Integers), ImageSet(Lambda(n, 2*n*pi), S.Integers)),
        (ImageSet(Lambda(n, 2*n*pi + pi), S.Integers), ImageSet(Lambda(n, 2*n*pi + pi), S.Integers)))
    sol = nonlinsolve(eqn, [x, y])

    for i0, j0 in zip(ordered(sol), ordered(ans)):
        assert len(i0) == len(j0) == 2
        assert all(a.dummy_eq(b) for a, b in zip(i0, j0))
    assert len(sol) == len(ans)


def test_issue_17566():
    # 检查 nonlinsolve 是否正确解决问题 17566
    assert nonlinsolve([32*(2**x)/2**(-y) - 4**y, 27*(3**x) - S(1)/3**y], x, y) ==\
        FiniteSet((-log(81)/log(3), 1))


def test_issue_16643():
    # 检查 solveset 是否正确处理 x**2*sin(x) 这个表达式
    n = Dummy('n')
    assert solveset(x**2*sin(x), x).dummy_eq(Union(ImageSet(Lambda(n, 2*n*pi + pi), S.Integers),
                                                   ImageSet(Lambda(n, 2*n*pi), S.Integers)))


def test_issue_19587():
    # 检查 nonlinsolve 是否正确解决问题 19587
    n,m = symbols('n m')
    assert nonlinsolve([32*2**m*2**n - 4**n, 27*3**m - 3**(-n)], m, n) ==\
        FiniteSet((-log(81)/log(3), 1))


def test_issue_5132_1():
    # 检查 nonlinsolve 是否正确解决问题 5132_1
    system = [sqrt(x**2 + y**2) - sqrt(10), x + y - 4]
    assert nonlinsolve(system, [x, y]) == FiniteSet((1, 3), (3, 1))

    n = Dummy('n')
    eqs = [exp(x)**2 - sin(y) + z**2, 1/exp(y) - 3]
    s_real_y = -log(3)
    s_real_z = sqrt(-exp(2*x) - sin(log(3)))
    soln_real = FiniteSet((s_real_y, s_real_z), (s_real_y, -s_real_z))
    lam = Lambda(n, 2*n*I*pi + -log(3))
    s_complex_y = ImageSet(lam, S.Integers)
    lam = Lambda(n, sqrt(-exp(2*x) + sin(2*n*I*pi + -log(3))))
    s_complex_z_1 = ImageSet(lam, S.Integers)
    lam = Lambda(n, -sqrt(-exp(2*x) + sin(2*n*I*pi + -log(3))))
    s_complex_z_2 = ImageSet(lam, S.Integers)
    soln_complex = FiniteSet(
                                            (s_complex_y, s_complex_z_1),
                                            (s_complex_y, s_complex_z_2)
                                        )
    soln = soln_real + soln_complex
    assert dumeq(nonlinsolve(eqs, [y, z]), soln)


def test_issue_5132_2():
    # 检查 nonlinsolve 是否正确解决问题 5132_2
    x, y = symbols('x, y', real=True)
    eqs = [exp(x)**2 - sin(y) + z**2]
    # 创建一个名为 n 的符号变量对象 Dummy
    n = Dummy('n')
    # 定义实数解 soln_real，包含一个复杂表达式
    soln_real = (log(-z**2 + sin(y))/2, z)
    # 创建一个 Lambda 函数 lam，接受 n 作为参数，返回一个复杂的表达式
    lam = Lambda(n, I*(2*n*pi + arg(-z**2 + sin(y)))/2 + log(Abs(z**2 - sin(y)))/2)
    # 使用 Lambda 函数 lam 创建一个 ImageSet 对象 img，限定参数为整数集合
    img = ImageSet(lam, S.Integers)
    # 创建一个包含复数解的元组 soln_complex
    soln_complex = (img, z)
    # 将实数解 soln_real 和复数解 soln_complex 放入一个 FiniteSet 对象 soln 中
    soln = FiniteSet(soln_real, soln_complex)
    # 使用非线性方程求解器验证非线性方程组 eqs 在变量 [x, z] 上的解是否等于 soln
    assert dumeq(nonlinsolve(eqs, [x, z]), soln)

    # 定义一个包含两个方程的系统 system
    system = [r - x**2 - y**2, tan(t) - y/x]
    # 计算 s_x 和 s_y 分别作为 x 和 y 的解
    s_x = sqrt(r/(tan(t)**2 + 1))
    s_y = sqrt(r/(tan(t)**2 + 1)) * tan(t)
    # 创建一个包含两个解的 FiniteSet 对象 soln
    soln = FiniteSet((s_x, s_y), (-s_x, -s_y))
    # 使用非线性方程求解器验证非线性方程组 system 在变量 [x, y] 上的解是否等于 soln
    assert nonlinsolve(system, [x, y]) == soln
def test_issue_6752():
    # 定义实数符号 a 和 b
    a, b = symbols('a, b', real=True)
    # 使用非线性求解器解方程组 {a^2 + a, a - b}，期望得到解集 {(-1, -1), (0, 0)}
    assert nonlinsolve([a**2 + a, a - b], [a, b]) == {(-1, -1), (0, 0)}


@SKIP("slow")
def test_issue_5114_solveset():
    # 慢速测试用例
    from sympy.abc import o, p

    # 方程组中没有 'a'，这是问题最初的形式
    syms = [a, b, c, f, h, k, n]
    eqs = [
        b + r/d - c/d,
        c*(1/d + 1/e + 1/g) - f/g - r/d,
        f*(1/g + 1/i + 1/j) - c/g - h/i,
        h*(1/i + 1/l + 1/m) - f/i - k/m,
        k*(1/m + 1/o + 1/p) - h/m - n/p,
        n*(1/p + 1/q) - k/p
    ]
    # 使用非线性求解器求解方程组，期望得到单一解
    assert len(nonlinsolve(eqs, syms)) == 1


@SKIP("Hangs")
def _test_issue_5335():
    # 无法检查零维系统
    # is_zero_dimensional Hangs
    lam, a0, conc = symbols('lam a0 conc')
    eqs = [
        lam + 2*y - a0*(1 - x/2)*x - 0.005*x/2*x,
        a0*(1 - x/2)*x - 1*y - 0.743436700916726*y,
        x + y - conc
    ]
    sym = [x, y, a0]
    # 有4个解，但只有两个是有效的
    assert len(nonlinsolve(eqs, sym)) == 2
    # 浮点数
    eqs = [
        lam + 2*y - a0*(1 - x/2)*x - 0.005*x/2*x,
        a0*(1 - x/2)*x - 1*y - 0.743436700916726*y,
        x + y - conc
    ]
    sym = [x, y, a0]
    assert len(nonlinsolve(eqs, sym)) == 2


def test_issue_2777():
    # 方程表示两个圆
    x, y = symbols('x y', real=True)
    e1, e2 = sqrt(x**2 + y**2) - 10, sqrt(y**2 + (-x + 10)**2) - 3
    a, b = Rational(191, 20), 3*sqrt(391)/20
    ans = {(a, -b), (a, b)}
    # 使用非线性求解器求解方程组 {e1, e2}，期望得到解集 {(a, -b), (a, b)}
    assert nonlinsolve((e1, e2), (x, y)) == ans
    assert nonlinsolve((e1, e2/(x - a)), (x, y)) == S.EmptySet
    # 使第二个圆的半径为 -3
    e2 += 6
    assert nonlinsolve((e1, e2), (x, y)) == S.EmptySet


def test_issue_8828():
    x1 = 0
    y1 = -620
    r1 = 920
    x2 = 126
    y2 = 276
    x3 = 51
    y3 = 205
    r3 = 104
    v = [x, y, z]

    f1 = (x - x1)**2 + (y - y1)**2 - (r1 - z)**2
    f2 = (x2 - x)**2 + (y2 - y)**2 - z**2
    f3 = (x - x3)**2 + (y - y3)**2 - (r3 - z)**2
    F = [f1, f2, f3]

    g1 = sqrt((x - x1)**2 + (y - y1)**2) + z - r1
    g2 = f2
    g3 = sqrt((x - x3)**2 + (y - y3)**2) + z - r3
    G = [g1, g2, g3]

    # 两个解相同
    A = nonlinsolve(F, v)
    B = nonlinsolve(G, v)
    assert A == B


def test_nonlinsolve_conditionset():
    # 当 solveset 无法解决所有方程时，返回 conditionset
    f = Function('f')
    f1 = f(x) - pi/2
    f2 = f(y) - pi*Rational(3, 2)
    intermediate_system = Eq(2*f(x) - pi, 0) & Eq(2*f(y) - 3*pi, 0)
    syms = Tuple(x, y)
    soln = ConditionSet(
        syms,
        intermediate_system,
        S.Complexes**2)
    # 使用非线性求解器求解方程组 {f1, f2}，期望得到 soln
    assert nonlinsolve([f1, f2], [x, y]) == soln


def test_substitution_basic():
    assert substitution([], [x, y]) == S.EmptySet
    assert substitution([], []) == S.EmptySet
    system = [2*x**2 + 3*y**2 - 30, 3*x**2 - 2*y**2 - 19]
    soln = FiniteSet((-3, -2), (-3, 2), (3, -2), (3, 2))
    # 使用替换方法求解方程组 {system}，期望得到 soln
    assert substitution(system, [x, y]) == soln

    soln = FiniteSet((-1, 1))
    # 断言：调用 substitution 函数，并检查其返回结果是否与 soln 相等
    assert substitution([x + y],            # 使用表达式 [x + y]
                        [x],                 # 替换变量 x
                        [{y: 1}],            # 使用映射 {y: 1} 替换 y
                        [y],                 # 替换变量 y
                        set(),               # 使用空集合替换 free_variables
                        [x, y])              # 使用 [x, y] 替换原变量顺序
           == soln                         # 断言结果应等于 soln
    
    # 断言：再次调用 substitution 函数，并检查其返回结果是否等于 S.EmptySet
    assert substitution(
            [x + y],                       # 使用表达式 [x + y]
            [x],                            # 替换变量 x
            [{y: 1}],                       # 使用映射 {y: 1} 替换 y
            [y],                            # 替换变量 y
            {x + 1},                        # 使用集合 {x + 1} 替换 free_variables
            [y, x])                         # 使用 [y, x] 替换原变量顺序
           == S.EmptySet                   # 断言结果应等于 S.EmptySet
def test_substitution_incorrect():
    # 测试替换函数对不正确的解的处理

    # 第一个测试，预期结果应该是{(1, 1, f)}
    assert substitution([h - 1, k - 1, f - 2, f - 4, -2 * k],
                        [h, k, f]) == {(1, 1, f)}

    # 第二个测试，预期结果应该是{(-y - z, y, z)}
    assert substitution([x + y + z, S.One, S.One, S.One], [x, y, z]) == \
                        {(-y - z, y, z)}

    # 第三个测试，预期结果应该是{(d, -d, -d, d)}
    assert substitution([a - d, b + d, c + d, d**2 + 1], [a, b, c, d]) == \
                        {(d, -d, -d, d)}

    # 第四个测试，预期结果应该是{(0, b)}
    assert substitution([a*(a - log(b)), a*(b - 2)], [a, b]) == \
           {(0, b)}

    # 第五个测试，预期结果应该是{z}
    assert substitution([-k*y + 6*x - 4*y, -81*k + 49*y**2 - 270,
                         -3*k*z + k + z**3, k**2 - 2*k + 4],
                        [x, y, z, k]).free_symbols == {z}


def test_substitution_redundant():
    # 测试替换函数对冗余解的处理

    # 第一个测试，预期结果应该是{(-y, 1), (y, 1), (-sqrt(y**2), 1), (sqrt(y**2), 1)}
    assert substitution([x**2 - y**2, z - 1], [x, z]) == \
           {(-y, 1), (y, 1), (-sqrt(y**2), 1), (sqrt(y**2), 1)}

    # 第二个测试，预期结果长度应该是5
    res = substitution([x - y, y**3 - 3*y**2 + 1], [x, y])
    assert len(res) == 5


def test_issue_5132_substitution():
    # 测试替换函数对具体问题 #5132 的处理

    x, y, z, r, t = symbols('x, y, z, r, t', real=True)
    system = [r - x**2 - y**2, tan(t) - y/x]

    # 计算出的解集
    s_x_1 = Complement(FiniteSet(-sqrt(r/(tan(t)**2 + 1))), FiniteSet(0))
    s_x_2 = Complement(FiniteSet(sqrt(r/(tan(t)**2 + 1))), FiniteSet(0))
    s_y = sqrt(r/(tan(t)**2 + 1))*tan(t)
    soln = FiniteSet((s_x_2, s_y)) + FiniteSet((s_x_1, -s_y))
    assert substitution(system, [x, y]) == soln

    # 另一个方案的解集
    n = Dummy('n')
    eqs = [exp(x)**2 - sin(y) + z**2, 1/exp(y) - 3]
    s_real_y = -log(3)
    s_real_z = sqrt(-exp(2*x) - sin(log(3)))
    soln_real = FiniteSet((s_real_y, s_real_z), (s_real_y, -s_real_z))
    lam = Lambda(n, 2*n*I*pi + -log(3))
    s_complex_y = ImageSet(lam, S.Integers)
    lam = Lambda(n, sqrt(-exp(2*x) + sin(2*n*I*pi + -log(3))))
    s_complex_z_1 = ImageSet(lam, S.Integers)
    lam = Lambda(n, -sqrt(-exp(2*x) + sin(2*n*I*pi + -log(3))))
    s_complex_z_2 = ImageSet(lam, S.Integers)
    soln_complex = FiniteSet(
        (s_complex_y, s_complex_z_1),
        (s_complex_y, s_complex_z_2))
    soln = soln_real + soln_complex
    assert dumeq(substitution(eqs, [y, z]), soln)


def test_raises_substitution():
    # 测试替换函数对异常情况的处理

    # 应该引发 ValueError 异常，因为变量列表为空
    raises(ValueError, lambda: substitution([x**2 -1], []))

    # 应该引发 TypeError 异常，因为缺少替换列表
    raises(TypeError, lambda: substitution([x**2 -1]))

    # 应该引发 ValueError 异常，因为替换列表中的元素不是符号
    raises(ValueError, lambda: substitution([x**2 -1], [sin(x)]))

    # 应该引发 TypeError 异常，因为替换列表不是可迭代对象
    raises(TypeError, lambda: substitution([x**2 -1], x))

    # 应该引发 TypeError 异常，因为替换列表中的元素不是符号
    raises(TypeError, lambda: substitution([x**2 -1], 1))


def test_issue_21022():
    # 测试问题 #21022 的解决方案
    # 导入 sympy 库中的 sympify 函数，用于将字符串转换为符号表达式
    from sympy.core.sympify import sympify

    # 定义包含符号表达式字符串的列表
    eqs = [
    'k-16',
    'p-8',
    'y*y+z*z-x*x',
    'd - x + p',
    'd*d+k*k-y*y',
    'z*z-p*p-k*k',
    'abc-efg',
    ]
    # 创建符号变量 efg
    efg = Symbol('efg')
    # 将字符串列表中的每个元素转换为 sympy 的符号表达式对象
    eqs = [sympify(x) for x in eqs]

    # 找出所有符号表达式中使用的符号变量，并保持其顺序
    syb = list(ordered(set.union(*[x.free_symbols for x in eqs])))
    # 解非线性方程组 eqs，得到所有解的集合 res
    res = nonlinsolve(eqs, syb)

    # 预期的解集合 ans，包含多个元组，每个元组表示一组解
    ans = FiniteSet(
    (efg, 32, efg, 16, 8, 40, -16*sqrt(5), -8*sqrt(5)),
    (efg, 32, efg, 16, 8, 40, -16*sqrt(5), 8*sqrt(5)),
    (efg, 32, efg, 16, 8, 40, 16*sqrt(5), -8*sqrt(5)),
    (efg, 32, efg, 16, 8, 40, 16*sqrt(5), 8*sqrt(5)),
    )
    # 断言解集合 res 和预期解集合 ans 的长度均为 4
    assert len(res) == len(ans) == 4
    # 断言解集合 res 等于预期解集合 ans
    assert res == ans
    # 对于解集合 res 中的每个解，断言其长度为 8
    for result in res.args:
        assert len(result) == 8
def test_issue_17940():
    # 创建一个名为 n 的虚拟符号对象
    n = Dummy('n')
    # 创建一个名为 k1 的虚拟符号对象
    k1 = Dummy('k1')
    # 定义一个包含 Lambda 表达式和 ProductSet 的解集对象 sol
    sol = ImageSet(
        # Lambda 表达式：接受参数 (k1, n)，返回表达式的复杂数值
        Lambda(((k1, n),), I*(2*k1*pi + arg(2*n*I*pi + log(5)))
              + log(Abs(2*n*I*pi + log(5)))),
        # ProductSet：包含整数集合 S.Integers 的乘积
        ProductSet(S.Integers, S.Integers))
    # 断言解出 exp(exp(x)) - 5 等式的解集是否等同于预期的解集 sol
    assert solveset(exp(exp(x)) - 5, x).dummy_eq(sol)


def test_issue_17906():
    # 断言解出 7**(x**2 - 80) - 49**x 等式的解集是否等同于预期的有限解集 {-8, 10}
    assert solveset(7**(x**2 - 80) - 49**x, x) == FiniteSet(-8, 10)


@XFAIL
def test_issue_17933():
    # 定义一系列非线性方程 eq1, eq2, eq3, eq4
    eq1 = x*sin(45) - y*cos(q)
    eq2 = x*cos(45) - y*sin(q)
    eq3 = 9*x*sin(45)/10 + y*cos(q)
    eq4 = 9*x*cos(45)/10 + y*sin(z) - z
    # 断言使用 nonlinsolve 解这些方程是否等同于预期的解集 {(0, 0, 0, q)}
    assert nonlinsolve([eq1, eq2, eq3, eq4], x, y, z, q) == \
        FiniteSet((0, 0, 0, q))


def test_issue_17933_bis():
    # 'nonlinsolve' 的结果取决于未知数的 'default_sort_key' 排序。
    eq1 = x*sin(45) - y*cos(q)
    eq2 = x*cos(45) - y*sin(q)
    eq3 = 9*x*sin(45)/10 + y*cos(q)
    eq4 = 9*x*cos(45)/10 + y*sin(z) - z
    zz = Symbol('zz')
    # 将方程组中的符号 q 替换为 zz，并用 nonlinsolve 求解
    eqs = [e.subs(q, zz) for e in (eq1, eq2, eq3, eq4)]
    # 断言解出方程组 eqs 是否等同于预期的解集 {(0, 0, 0, zz)}
    assert nonlinsolve(eqs, x, y, z, zz) == FiniteSet((0, 0, 0, zz))


def test_issue_14565():
    # 移除冗余代码后，断言 nonlinsolve 解方程组 [k + m, k + m*exp(-2*pi*k)], [k, m] 是否等同于预期的解集 {(-n*I, ImageSet(Lambda(n, n*I), S.Integers))}
    assert dumeq(nonlinsolve([k + m, k + m*exp(-2*pi*k)], [k, m]),
        FiniteSet((-n*I, ImageSet(Lambda(n, n*I), S.Integers))))


# 非线性方程组 'nonlinsolve' 的测试结束


def test_issue_9556():
    # 创建一个正数符号 b
    b = Symbol('b', positive=True)
    # 断言解出 Abs(x) + 1 等式的解集是否等同于预期的空集 S.EmptySet
    assert solveset(Abs(x) + 1, x, S.Reals) is S.EmptySet
    # 断言解出 Abs(x) + b 等式的解集是否等同于预期的空集 S.EmptySet
    assert solveset(Abs(x) + b, x, S.Reals) is S.EmptySet
    # 断言解出方程式 Eq(b, -1) 的解集是否等同于预期的空集 S.EmptySet
    assert solveset(Eq(b, -1), b, S.Reals) is S.EmptySet


def test_issue_9611():
    # 断言解出方程式 Eq(x - x + a, a) 的解集是否等同于预期的实数集 S.Reals
    assert solveset(Eq(x - x + a, a), x, S.Reals) == S.Reals
    # 断言解出方程式 Eq(y - y + a, a) 的解集是否等同于预期的复数集 S.Complexes


def test_issue_9557():
    # 断言解出方程式 x**2 + a 的解集是否等同于预期的交集 Intersection(S.Reals, FiniteSet(-sqrt(-a), sqrt(-a)))
    assert solveset(x**2 + a, x, S.Reals) == Intersection(S.Reals,
        FiniteSet(-sqrt(-a), sqrt(-a)))


def test_issue_9778():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    # 断言解出方程式 x**3 + 1 的解集是否等同于预期的有限集 FiniteSet(-1)
    assert solveset(x**3 + 1, x, S.Reals) == FiniteSet(-1)
    # 断言解出方程式 x**Rational(3, 5) + 1 的解集是否等同于预期的空集 S.EmptySet
    assert solveset(x**Rational(3, 5) + 1, x, S.Reals) == S.EmptySet
    # 断言解出方程式 x**3 + y 的解集是否等同于预期的有限集 FiniteSet(-Abs(y)**Rational(1, 3)*sign(y))


def test_issue_10214():
    # 断言解出方程式 x**Rational(3, 2) + 4 的解集是否等同于预期的空集 S.EmptySet
    assert solveset(x**Rational(3, 2) + 4, x, S.Reals) == S.EmptySet
    # 断言解出方程式 x**(Rational(-3, 2)) + 4 的解集是否等同于预期的空集 S.EmptySet

    ans = FiniteSet(-2**Rational(2, 3))
    # 断言解出方程式 x**(S(3)) + 4 的解集是否等同于预期的有限集 ans
    assert solveset(x**(S(3)) + 4, x, S.Reals) == ans
    # 替换 x 为 ans 中的值，并验证结果
    assert (x**(S(3)) + 4).subs(x,list(ans)[0]) == 0 
    assert (x**(S(3)) + 4).subs(x,-(-2)**Rational(2, 3)) == 0


def test_issue_9849():
    # 断言解出方程式 Abs(sin(x)) + 1 的解集是否等同于预期的空集 S.EmptySet


def test_issue_9953():
    # 断言解出方程式 linsolve([ ], x) 的解集是否等同于预期的空集 S.EmptySet


def test_issue_9913():
    # 断言解出方程式 2*x + 1/(x - 10)**2 的解集是否等同于预期的有限集 FiniteSet(-(3*sqrt(24081)/4 + Rational(4027, 4))**Rational(1, 3)/3 - 100/
                (3*(3*sqrt(24081)/4 + Rational(4027, 4))**Rational(1, 3)) + Rational(20, 3))


def test_issue_10397():
    pass
    # 断言：确保求解 sqrt(x) = 0 的解集为复数域上的有限集 {0}
    assert solveset(sqrt(x), x, S.Complexes) == FiniteSet(0)
# 测试函数，用于验证 linear_eq_to_matrix 函数在特定输入下是否引发 ValueError 异常
def test_issue_14987():
    # 验证当传入单个方程 x**2 时，linear_eq_to_matrix 是否会引发 ValueError 异常
    raises(ValueError, lambda: linear_eq_to_matrix([x**2], x))
    # 验证当传入方程组 [x*(-3/x + 1) + 2*y - a] 时，linear_eq_to_matrix 是否会引发 ValueError 异常
    raises(ValueError, lambda: linear_eq_to_matrix([x*(-3/x + 1) + 2*y - a], [x, y]))
    # 验证当传入方程 (x**2 - 3*x)/(x - 3) - 3 时，linear_eq_to_matrix 是否会引发 ValueError 异常
    raises(ValueError, lambda: linear_eq_to_matrix([(x**2 - 3*x)/(x - 3) - 3], x))
    # 验证当传入方程 (x + 1)**3 - x**3 - 3*x**2 + 7 时，linear_eq_to_matrix 是否会引发 ValueError 异常
    raises(ValueError, lambda: linear_eq_to_matrix([(x + 1)**3 - x**3 - 3*x**2 + 7], x))
    # 验证当传入方程组 [x*(1/x + 1) + y] 时，linear_eq_to_matrix 是否会引发 ValueError 异常
    raises(ValueError, lambda: linear_eq_to_matrix([x*(1/x + 1) + y], [x, y]))
    # 验证当传入方程 (x + 1)*y 时，linear_eq_to_matrix 是否会引发 ValueError 异常
    raises(ValueError, lambda: linear_eq_to_matrix([(x + 1)*y], [x, y]))
    # 验证当传入方程组 [Eq(1/x, 1/x + y)] 时，linear_eq_to_matrix 是否会引发 ValueError 异常
    raises(ValueError, lambda: linear_eq_to_matrix([Eq(1/x, 1/x + y)], [x, y]))
    # 验证当传入方程组 [Eq(y/x, y/x + y)] 时，linear_eq_to_matrix 是否会引发 ValueError 异常
    raises(ValueError, lambda: linear_eq_to_matrix([Eq(y/x, y/x + y)], [x, y]))
    # 验证当传入方程组 [Eq(x*(x + 1), x**2 + y)] 时，linear_eq_to_matrix 是否会引发 ValueError 异常
    raises(ValueError, lambda: linear_eq_to_matrix([Eq(x*(x + 1), x**2 + y)], [x, y]))


# 测试函数，用于验证 solveset 函数的一些特定案例
def test_simplification():
    # 定义一个方程
    eq = x + (a - b)/(-2*a + 2*b)
    # 验证 solveset 函数解方程 eq 时的结果是否为 FiniteSet(S.Half)
    assert solveset(eq, x) == FiniteSet(S.Half)
    # 验证 solveset 函数在求解方程 eq 时，限定解集为实数域时的结果
    assert solveset(eq, x, S.Reals) == Intersection({-((a - b)/(-2*a + 2*b))}, S.Reals)
    # 再定义一个带有正负号限制的方程
    ap = Symbol('ap', positive=True)
    bn = Symbol('bn', negative=True)
    eq = x + (ap - bn)/(-2*ap + 2*bn)
    # 验证 solveset 函数解带有正负号限制的方程 eq 时的结果是否为 FiniteSet(S.Half)
    assert solveset(eq, x) == FiniteSet(S.Half)
    # 验证 solveset 函数在求解方程 eq 时，限定解集为实数域时的结果是否为 FiniteSet(S.Half)


# 测试函数，用于验证 solveset 函数对整数域关系的处理
def test_integer_domain_relational():
    # 定义多个不等式
    eq1 = 2*x + 3 > 0
    eq2 = x**2 + 3*x - 2 >= 0
    eq3 = x + 1/x > -2 + 1/x
    eq4 = x + sqrt(x**2 - 5) > 0
    eq = x + 1/x > -2 + 1/x
    eq5 = eq.subs(x, log(x))
    eq6 = log(x)/x <= 0
    eq7 = log(x)/x < 0
    eq8 = x/(x-3) < 3
    eq9 = x/(x**2-3) < 3

    # 验证 solveset 函数求解不等式 eq1 在整数集合上的结果是否为 Range(-1, oo, 1)
    assert solveset(eq1, x, S.Integers) == Range(-1, oo, 1)
    # 验证 solveset 函数求解不等式 eq2 在整数集合上的结果是否为 Union(Range(-oo, -3, 1), Range(1, oo, 1))
    assert solveset(eq2, x, S.Integers) == Union(Range(-oo, -3, 1), Range(1, oo, 1))
    # 验证 solveset 函数求解不等式 eq3 在整数集合上的结果是否为 Union(Range(-1, 0, 1), Range(1, oo, 1))
    assert solveset(eq3, x, S.Integers) == Union(Range(-1, 0, 1), Range(1, oo, 1))
    # 验证 solveset 函数求解不等式 eq4 在整数集合上的结果是否为 Range(3, oo, 1)
    assert solveset(eq4, x, S.Integers) == Range(3, oo, 1)
    # 验证 solveset 函数求解不等式 eq5 在整数集合上的结果是否为 Range(2, oo, 1)
    assert solveset(eq5, x, S.Integers) == Range(2, oo, 1)
    # 验证 solveset 函数求解不等式 eq6 在整数集合上的结果是否为 Range(1, 2, 1)
    assert solveset(eq6, x, S.Integers) == Range(1, 2, 1)
    # 验证 solveset 函数求解不等式 eq7 在整数集合上的结果是否为空集 S.EmptySet
    assert solveset(eq7, x, S.Integers) == S.EmptySet
    # 验证 solveset 函数求解不等式 eq8 在 x 取值范围为 Range(0,5) 时的结果是否为 Range(0, 3, 1)
    assert solveset(eq8, x, domain=Range(0,5)) == Range(0, 3, 1)
    # 验证 solveset 函数求解不等式 eq9 在 x 取值范围为 Range(0,5) 时的结果是否为 Union(Range(0, 2, 1), Range(2, 5, 1))

    # test_issue_19794
    # 验证 solveset 函数求解不等式 x + 2 < 0 在整数集合上的结果是否为 Range(-oo, -2, 1)


# 测试函数，用于验证 solveset 函数对特定问题的处理
def test_issue_10555():
    # 定义函数 f 和 g
    f = Function('f')
    g = Function('g')
    # 验证 solveset 函数解方程 f(x) - pi/2 在实数域上的结果是否与 ConditionSet(x, Eq(f(x) - pi/2, 0), S.Reals) 相同
    assert solveset(f(x) - pi/2, x, S.Reals).dummy_eq(ConditionSet(x, Eq(f(x) - pi/2, 0), S.Reals))
    # 验证 solveset 函数解方程 f(g(x)) - pi/2 在实数域上的结果是否与 ConditionSet(g(x), Eq(f(g(x)) - pi/2, 0), S.Reals) 相同


# 测试函数，用于验证 solveset 函数对特定问题的处理
def test_issue_8715():
    # 定义方程 eq
    eq = x + 1/x > -2 + 1/x
    # 验证 solveset 函数解方程 eq 在实数域上的结果是否为 Interval.open(-2, oo) - FiniteSet(0)
    # 使用 SymPy 的 solveset 函数解方程 eq 对变量 x，解集应为 soln
    assert solveset(eq, x, S.Reals) == soln

    # 定义方程 eq
    eq = sqrt(r)*Abs(tan(t))/sqrt(tan(t)**2 + 1) + x*tan(t)
    # 计算 s 的值
    s = -sqrt(r)*Abs(tan(t))/(sqrt(tan(t)**2 + 1)*tan(t))
    # 计算方程 Intersection(S.Reals, FiniteSet(s)) 的解集
    soln = Intersection(S.Reals, FiniteSet(s))
    # 使用 SymPy 的 solveset 函数解方程 eq 对变量 x，解集应为 soln
    assert solveset(eq, x, S.Reals) == soln
def test_issue_11534():
    # 定义符号变量 x 和 y，限定为实数
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    # 定义两个方程 eq1 和 eq2
    eq1 = -y + x/sqrt(-x**2 + 1)
    eq2 = -y**2 + x**2/(-x**2 + 1)

    # 创建一个包含唯一解的有限集 s1 和 s2
    s1, s2 = FiniteSet(-y/sqrt(y**2 + 1)), FiniteSet(y/sqrt(y**2 + 1))
    # 创建一个条件集 cset，用于存储符合 eq1 的解集合
    cset = ConditionSet(x, Eq(eq1, 0), s1)
    # 计算两个解集合 sol1 和 sol2
    sol1 = (s2 - {-1, 1}) | (cset - {-1, 1})
    sol2 = (s1 | s2) - {-1, 1}

    # 断言解集合 solveset(eq1, x, S.Reals) 应等于 sol1
    assert solveset(eq1, x, S.Reals) == sol1
    # 断言解集合 solveset(eq2, x, S.Reals) 应等于 sol2
    assert solveset(eq2, x, S.Reals) == sol2


def test_issue_10477():
    # 断言解集合 solveset((x**2 + 4*x - 3)/x < 2, x, S.Reals) 应等于指定的并集
    assert solveset((x**2 + 4*x - 3)/x < 2, x, S.Reals) == \
        Union(Interval.open(-oo, -3), Interval.open(0, 1))


def test_issue_10671():
    # 断言解集合 solveset(sin(y), y, Interval(0, pi)) 应等于指定的有限集
    assert solveset(sin(y), y, Interval(0, pi)) == FiniteSet(0, pi)
    # 创建区间对象 i
    i = Interval(1, 10)
    # 断言解集合 solveset((1/x).diff(x) < 0, x, i) 应等于区间 i
    assert solveset((1/x).diff(x) < 0, x, i) == i


def test_issue_11064():
    # 定义方程 eq
    eq = x + sqrt(x**2 - 5)
    # 断言解集合 solveset(eq > 0, x, S.Reals) 应等于指定的区间
    assert solveset(eq > 0, x, S.Reals) == \
        Interval(sqrt(5), oo)
    # 断言解集合 solveset(eq < 0, x, S.Reals) 应等于指定的区间
    assert solveset(eq < 0, x, S.Reals) == \
        Interval(-oo, -sqrt(5))
    # 断言解集合 solveset(eq > sqrt(5), x, S.Reals) 应等于指定的区间
    assert solveset(eq > sqrt(5), x, S.Reals) == \
        Interval.Lopen(sqrt(5), oo)


def test_issue_12478():
    # 定义方程 eq
    eq = sqrt(x - 2) + 2
    # 断言实数解集合 solveset(eq > 0, x, S.Reals) 应为空集
    assert solveset(eq > 0, x, S.Reals) is S.EmptySet
    # 断言实数解集合 solveset(eq < 0, x, S.Reals) 应为空集
    assert solveset(eq < 0, x, S.Reals) is S.EmptySet
    # 断言实数解集合 solveset(eq > 0, x, S.Reals) 应等于指定的区间
    assert solveset(eq > 0, x, S.Reals) == Interval(2, oo)


def test_issue_12429():
    # 求解 log(x)/x <= 0 的解集合
    eq = solveset(log(x)/x <= 0, x, S.Reals)
    # 定义期望的解集合
    sol = Interval.Lopen(0, 1)
    # 断言解集合 eq 应等于 sol
    assert eq == sol


def test_issue_19506():
    # 求解 arg(x + I) 的解集合
    eq = arg(x + I)
    # 定义虚拟变量 C
    C = Dummy('C')
    # 断言 solveset(eq) 应等于指定的交集
    assert solveset(eq).dummy_eq(Intersection(ConditionSet(C, Eq(im(C) + 1, 0), S.Complexes),
                                             ConditionSet(C, re(C) > 0, S.Complexes)))


def test_solveset_arg():
    # 断言解集合 solveset(arg(x), x, S.Reals) 应等于指定的开区间
    assert solveset(arg(x), x, S.Reals)  == Interval.open(0, oo)
    # 断言解集合 solveset(arg(4*x -3), x, S.Reals) 应等于指定的开区间
    assert solveset(arg(4*x -3), x, S.Reals) == Interval.open(Rational(3, 4), oo)


def test__is_finite_with_finite_vars():
    f = _is_finite_with_finite_vars
    # 断言对于不同类型的虚拟变量 x，_is_finite_with_finite_vars 函数的返回值
    assert all(f(1/x) is None for x in (
        Dummy(), Dummy(real=True), Dummy(complex=True)))
    # 断言对于具有限定条件的虚拟变量，_is_finite_with_finite_vars 函数的返回值
    assert f(1/Dummy(real=False)) is True  # b/c it's finite but not 0


def test_issue_13550():
    # 解方程 x**2 - 2*x - 15 在指定域的解集合
    assert solveset(x**2 - 2*x - 15, symbol=x, domain=Interval(-oo, 0)) == FiniteSet(-3)


def test_issue_13849():
    pass  # 暂无代码，保留函数结构
    # 使用 assert 断言来验证 nonlinsolve((t*(sqrt(5) + sqrt(2)) - sqrt(2), t), t) 的结果是否为空集 S.EmptySet
    assert nonlinsolve((t*(sqrt(5) + sqrt(2)) - sqrt(2), t), t) is S.EmptySet
# 定义一个测试函数，用于验证解决包含绝对值、最小值、最大值和复杂函数的方程的问题
def test_issue_14223():
    # 验证解决绝对值和最小值函数组合的方程在实数域中的解
    assert solveset((Abs(x + Min(x, 2)) - 2).rewrite(Piecewise), x, S.Reals) == FiniteSet(-1, 1)
    # 验证解决绝对值和最小值函数组合的方程在区间[0, 2]中的解
    assert solveset((Abs(x + Min(x, 2)) - 2).rewrite(Piecewise), x, Interval(0, 2)) == FiniteSet(1)
    # 验证解决线性方程 x = 0 在有限集 {1, 2} 中的解为空集
    assert solveset(x, x, FiniteSet(1, 2)) is S.EmptySet


# 定义一个测试函数，用于验证解决包含最大值、最小值、绝对值和复杂函数的方程的问题
def test_issue_10158():
    # 设置定义域为实数集合
    dom = S.Reals
    # 验证解决包含 x*Max(x, 15) - 10 的方程的解为有理数集合 {2/3}
    assert solveset(x*Max(x, 15) - 10, x, dom) == FiniteSet(Rational(2, 3))
    # 验证解决包含 x*Min(x, 15) - 10 的方程的解为 {-sqrt(10), sqrt(10)}
    assert solveset(x*Min(x, 15) - 10, x, dom) == FiniteSet(-sqrt(10), sqrt(10))
    # 验证解决包含 Max(Abs(x - 3) - 1, x + 2) - 3 的方程的解为有限集 {-1, 1}
    assert solveset(Max(Abs(x - 3) - 1, x + 2) - 3, x, dom) == FiniteSet(-1, 1)
    # 验证解决包含 Abs(x - 1) - Abs(y) 的方程的解为有限集 {-Abs(y) + 1, Abs(y) + 1}
    assert solveset(Abs(x - 1) - Abs(y), x, dom) == FiniteSet(-Abs(y) + 1, Abs(y) + 1)
    # 验证解决包含 Abs(x + 4*Abs(x + 1)) 的方程的解为有理数集合 {-4/3, -4/5}
    assert solveset(Abs(x + 4*Abs(x + 1)), x, dom) == FiniteSet(Rational(-4, 3), Rational(-4, 5))
    # 验证解决包含 2*Abs(x + Abs(x + Max(3, x))) - 2 的方程的解为有限集 {-1, -2}
    assert solveset(2*Abs(x + Abs(x + Max(3, x))) - 2, x, S.Reals) == FiniteSet(-1, -2)
    # 设置定义域为复数集合
    dom = S.Complexes
    # 验证解决包含 x*Max(x, 15) - 10 的方程在复数域中会引发 ValueError 异常
    raises(ValueError, lambda: solveset(x*Max(x, 15) - 10, x, dom))
    # 验证解决包含 x*Min(x, 15) - 10 的方程在复数域中会引发 ValueError 异常
    raises(ValueError, lambda: solveset(x*Min(x, 15) - 10, x, dom))
    # 验证解决包含 Max(Abs(x - 3) - 1, x + 2) - 3 的方程在复数域中会引发 ValueError 异常
    raises(ValueError, lambda: solveset(Max(Abs(x - 3) - 1, x + 2) - 3, x, dom))
    # 验证解决包含 Abs(x - 1) - Abs(y) 的方程在复数域中会引发 ValueError 异常
    raises(ValueError, lambda: solveset(Abs(x - 1) - Abs(y), x, dom))
    # 验证解决包含 Abs(x + 4*Abs(x + 1)) 的方程在复数域中会引发 ValueError 异常
    raises(ValueError, lambda: solveset(Abs(x + 4*Abs(x + 1)), x, dom))


# 定义一个测试函数，用于验证解决包含指数函数的方程的问题
def test_issue_14300():
    # 定义方程和预期结果
    f = 1 - exp(-18000000*x) - y
    a1 = FiniteSet(-log(-y + 1)/18000000)

    # 验证解决方程 f 在实数域中的解为实数集合和 a1 的交集
    assert solveset(f, x, S.Reals) == Intersection(S.Reals, a1)
    # 验证解决方程 f 的解为映射集合，其中 Lambda 表达式表示解的形式
    assert dumeq(solveset(f, x),
        ImageSet(Lambda(n, -I*(2*n*pi + arg(-y + 1))/18000000 -
            log(Abs(y - 1))/18000000), S.Integers))


# 定义一个测试函数，用于验证解决包含根的方程的问题
def test_issue_14454():
    # 定义一个根的对象
    number = CRootOf(x**4 + x - 1, 2)
    # 验证尝试对 invert_real 函数应用根对象和 x**2 的方程会引发 ValueError 异常
    raises(ValueError, lambda: invert_real(number, 0, x))
    # 验证解决 x**2 的方程没有错误地执行
    assert invert_real(x**2, number, x)  # no error


# 定义一个测试函数，用于验证解决包含复杂方程的问题
def test_issue_17882():
    # 验证解决复杂方程的解在复数集合中为有限集 {sqrt(3), -sqrt(3)}
    assert solveset(-8*x**2/(9*(x**2 - 1)**(S(4)/3)) + 4/(3*(x**2 - 1)**(S(1)/3)), x, S.Complexes) == \
        FiniteSet(sqrt(3), -sqrt(3))


# 定义一个测试函数，用于验证解决包含指数项的方程的问题
def test_term_factors():
    # 验证对于给定表达式 3**x - 2，_term_factors 函数正确返回 [-2, 3**x]
    assert list(_term_factors(3**x - 2)) == [-2, 3**x]
    # 验证对于给定复杂表达式 expr，_term_factors 函数正确返回一组因子的集合
    expr = 4**(x + 1) + 4**(x + 2) + 4**(x - 1) - 3**(x + 2) - 3**(x + 3)
    assert set(_term_factors(expr)) == {
        3**(x + 2), 4**(x + 2), 3**(x + 3), 4**(x - 1), -1, 4**(x + 1)}


# 定义一个测试函数，用于验证 transolve 函数及其辅助函数的问题
def test_transolve():

    # 验证对于指数函数 3**x，_transolve 函数在实数域中返回空集
    assert _transolve(3**x, x, S.Reals) == S.EmptySet
    # 验证对于方程 3**x - 9**(x + 5)，_transolve 函数在实数域中返回有限集 {-10}
    assert _transolve(3**x - 9**(x + 5), x, S.Reals) == FiniteSet(-10)


# 定义一个测试函数，用于验证解决包含指数函数的实数方程的问题
def test_issue_21276():
    # 定义一个复杂的方程 eq
    eq = (2*x*(y - z) - y
    # 定义表达式 e10
    e10 = 29*2**(x + 1)*615**(x) - 123*2726**(x)

    # 使用 solveset 求解 e1 关于 x 的实数解，并断言结果为有限集合
    assert solveset(e1, x, S.Reals) == FiniteSet(
        -3*log(2)/(-2*log(3) + log(2)))

    # 使用 solveset 求解 e2 关于 x 的实数解，并断言结果为有限集合
    assert solveset(e2, x, S.Reals) == FiniteSet(Rational(4, 15))

    # 使用 solveset 求解 e3 关于 x 的实数解，并断言结果为空集
    assert solveset(e3, x, S.Reals) == S.EmptySet

    # 使用 solveset 求解 e4 关于 x 的实数解，并断言结果为有限集合包含 0
    assert solveset(e4, x, S.Reals) == FiniteSet(0)

    # 使用 solveset 求解 e5 关于 x 的实数解，并断言结果为实数与一个特定函数的交集
    assert solveset(e5, x, S.Reals) == Intersection(
        S.Reals, FiniteSet(y*log(2*exp(z/y))))

    # 使用 solveset 求解 e6 关于 x 的实数解，并断言结果为有限集合包含 0
    assert solveset(e6, x, S.Reals) == FiniteSet(0)

    # 使用 solveset 求解 e7 关于 x 的实数解，并断言结果为有限集合包含 2
    assert solveset(e7, x, S.Reals) == FiniteSet(2)

    # 使用 solveset 求解 e8 关于 x 的实数解，并断言结果为有限集合
    assert solveset(e8, x, S.Reals) == FiniteSet(-2*log(2)/5 + 2*log(3)/5 + Rational(4, 5))

    # 使用 solveset 求解 e9 关于 x 的实数解，并断言结果为有限集合包含 2
    assert solveset(e9, x, S.Reals) == FiniteSet(2)

    # 使用 solveset 求解 e10 关于 x 的实数解，并断言结果为有限集合，计算复杂的数学表达式
    assert solveset(e10, x, S.Reals) == FiniteSet((-log(29) - log(2) + log(123))/(-log(2726) + log(2) + log(615)))

    # 使用 solveset_real 求解给定的实数方程，断言结果为有限集合
    assert solveset_real(-9*exp(-2*x + 5) + 2**(x + 1), x) == FiniteSet(
        -((-5 - 2*log(3) + log(2))/(log(2) + 2)))

    # 使用 solveset_real 求解给定的实数方程，断言结果为有限集合
    assert solveset_real(4**(x/2) - 2**(x/3), x) == FiniteSet(0)

    # 计算常数 b
    b = sqrt(6)*sqrt(log(2))/sqrt(log(5))

    # 使用 solveset_real 求解给定的实数方程，断言结果为有限集合
    assert solveset_real(5**(x/2) - 2**(3/x), x) == FiniteSet(-b, b)

    # 对 solveset_real 的覆盖测试
    C1, C2 = symbols('C1 C2')
    f = Function('f')

    # 使用 solveset_real 求解给定的实数方程，断言结果为实数与一个特定函数的交集
    assert solveset_real(C1 + C2/x**2 - exp(-f(x)), f(x)) == Intersection(
        S.Reals, FiniteSet(-log(C1 + C2/x**2)))

    # 使用 solveset_real 求解给定的实数方程，断言结果为实数与一个特定函数的交集
    y = symbols('y', positive=True)
    assert solveset_real(x**2 - y**2/exp(x), y) == Intersection(
        S.Reals, FiniteSet(-sqrt(x**2*exp(x)), sqrt(x**2*exp(x))))

    # 使用 solveset 求解给定的方程，断言结果为条件集合
    p = Symbol('p', positive=True)
    assert solveset_real((1/p + 1)**(p + 1), p).dummy_eq(
        ConditionSet(x, Eq((1 + 1/x)**(x + 1), 0), S.Reals))

    # 使用 solveset 求解给定的方程，断言结果为集合包含特定的值
    assert solveset(2**x - 4**x + 12, x, S.Reals) == {2}

    # 使用 solveset 求解给定的方程，断言结果为集合包含特定的值
    assert solveset(2**x - 2**(2*x) + 12, x, S.Reals) == {2}
@XFAIL
# 定义一个测试函数，用于测试指数复杂条件下的解决方案
def test_exponential_complex():
    # 定义一个虚拟符号 'n'
    n = Dummy('n')

    # 断言解决方程 2**x + 4**x 的复数解集
    assert dumeq(solveset_complex(2**x + 4**x, x), imageset(
        Lambda(n, I*(2*n*pi + pi)/log(2)), S.Integers))
    
    # 断言解决方程 x**z * y**z - 2 的复数解集
    assert solveset_complex(x**z*y**z - 2, z) == FiniteSet(
        log(2)/(log(x) + log(y)))
    
    # 断言解决方程 4**(x/2) - 2**(x/3) 的复数解集
    assert dumeq(solveset_complex(4**(x/2) - 2**(x/3), x), imageset(
        Lambda(n, 3*n*I*pi/log(2)), S.Integers))
    
    # 断言解决方程 2**x + 32 的复数解集
    assert dumeq(solveset(2**x + 32, x), imageset(
        Lambda(n, (I*(2*n*pi + pi) + 5*log(2))/log(2)), S.Integers))

    # 定义一个复杂的方程 eq
    eq = (2**exp(y**2/x) + 2)/(x**2 + 15)
    # 定义变量 a
    a = sqrt(x)*sqrt(-log(log(2)) + log(log(2) + 2*n*I*pi))
    # 断言解决方程 eq 关于 y 的复数解集
    assert solveset_complex(eq, y) == FiniteSet(-a, a)

    # 定义 union1 和 union2
    union1 = imageset(Lambda(n, I*(2*n*pi - pi*Rational(2, 3))/log(2)), S.Integers)
    union2 = imageset(Lambda(n, I*(2*n*pi + pi*Rational(2, 3))/log(2)), S.Integers)
    # 断言解决方程 2**x + 4**x + 8**x 的复数解集
    assert dumeq(solveset(2**x + 4**x + 8**x, x), Union(union1, union2))

    # 定义复杂方程 eq
    eq = 4**(x + 1) + 4**(x + 2) + 4**(x - 1) - 3**(x + 2) - 3**(x + 3)
    # 求解方程 eq 关于 x 的解集
    res = solveset(eq, x)
    # 定义 num 和 den
    num = 2*n*I*pi - 4*log(2) + 2*log(3)
    den = -2*log(2) + log(3)
    # 定义 ans
    ans = imageset(Lambda(n, num/den), S.Integers)
    # 断言 res 和 ans 的相等性
    assert dumeq(res, ans)


# 定义一个测试函数，用于测试指数条件集
def test_expo_conditionset():

    # 定义多个函数 f1 到 f5
    f1 = (exp(x) + 1)**x - 2
    f2 = (x + 2)**y*x - 3
    f3 = 2**x - exp(x) - 3
    f4 = log(x) - exp(x)
    f5 = 2**x + 3**x - 5**x

    # 断言解决方程 f1 关于 x 的实数解集
    assert solveset(f1, x, S.Reals).dummy_eq(ConditionSet(
        x, Eq((exp(x) + 1)**x - 2, 0), S.Reals))
    
    # 断言解决方程 f2 关于 x 的实数解集
    assert solveset(f2, x, S.Reals).dummy_eq(ConditionSet(
        x, Eq(x*(x + 2)**y - 3, 0), S.Reals))
    
    # 断言解决方程 f3 关于 x 的实数解集
    assert solveset(f3, x, S.Reals).dummy_eq(ConditionSet(
        x, Eq(2**x - exp(x) - 3, 0), S.Reals))
    
    # 断言解决方程 f4 关于 x 的实数解集
    assert solveset(f4, x, S.Reals).dummy_eq(ConditionSet(
        x, Eq(-exp(x) + log(x), 0), S.Reals))
    
    # 断言解决方程 f5 关于 x 的实数解集
    assert solveset(f5, x, S.Reals).dummy_eq(ConditionSet(
        x, Eq(2**x + 3**x - 5**x, 0), S.Reals))


# 定义一个测试函数，用于测试指数符号
def test_exponential_symbols():
    # 定义符号 x, y, z 为正数
    x, y, z = symbols('x y z', positive=True)
    # 定义符号 xr, zr 为实数
    xr, zr = symbols('xr, zr', real=True)

    # 断言解决方程 z**x - y 关于 x 的实数解集
    assert solveset(z**x - y, x, S.Reals) == Intersection(
        S.Reals, FiniteSet(log(y)/log(z)))

    # 定义函数 f1 和 f2
    f1 = 2*x**w - 4*y**w
    f2 = (x/y)**w - 2
    # 定义解集 sol1 和 sol2
    sol1 = Intersection({log(2)/(log(x) - log(y))}, S.Reals)
    sol2 = Intersection({log(2)/log(x/y)}, S.Reals)
    # 断言解决方程 f1 关于 w 的实数解集
    assert solveset(f1, w, S.Reals) == sol1, solveset(f1, w, S.Reals)
    # 断言解决方程 f2 关于 w 的实数解集
    assert solveset(f2, w, S.Reals) == sol2, solveset(f2, w, S.Reals)

    # 断言解决方程 x**x 关于 x 的开区间 (0, oo) 内的解集
    assert solveset(x**x, x, Interval.Lopen(0,oo)).dummy_eq(
        ConditionSet(w, Eq(w**w, 0), Interval.open(0, oo)))
    
    # 断言解决方程 exp(x/y)*exp(-z/y) - 2 关于 y 的实数解集
    assert solveset(exp(x/y)*exp(-z/y) - 2, y, S.Reals) == \
    Complement(ConditionSet(y, Eq(im(x)/y, 0) & Eq(im(z)/y, 0), \
    Complement(Intersection(FiniteSet((x - z)/log(2)), S.Reals), FiniteSet(0))), FiniteSet(0))
    
    # 断言解决方程 exp(xr/y)*exp(-zr/y) - 2 关于 y 的实数解集
    assert solveset(exp(xr/y)*exp(-zr/y) - 2, y, S.Reals) == \
        Complement(FiniteSet((xr - zr)/log(2)), FiniteSet(0))
    # 使用符号计算库解决方程 a**x - b**x = 0，并断言其结果与给定的条件集合相等
    assert solveset(a**x - b**x, x).dummy_eq(ConditionSet(
        w, Ne(a, 0) & Ne(b, 0), FiniteSet(0)))
# 测试函数，检查解集求解器是否忽略假设条件
def test_ignore_assumptions():
    # 定义正数符号 x_pos 和一般符号 x
    xpos = symbols('x', positive=True)
    x = symbols('x')
    # 断言解决复数解集函数 solveset_complex 对两个方程的结果是否相同
    assert solveset_complex(xpos**2 - 4, xpos) == solveset_complex(x**2 - 4, x)


# 期望失败的测试用例，检查 issue 10864
@XFAIL
def test_issue_10864():
    # 断言解决方程 x**(y*z) - x = 0 在实数域上的解集是否等于 {1}
    assert solveset(x**(y*z) - x, x, S.Reals) == FiniteSet(1)


# 期望失败的测试用例，检查只解决指数型方程的函数
@XFAIL
def test_solve_only_exp_2():
    # 断言解决实数域上的方程 sqrt(exp(x)) + sqrt(exp(-x)) - 4 的解集是否等于 {2*log(-sqrt(3) + 2), 2*log(sqrt(3) + 2)}
    assert solveset_real(sqrt(exp(x)) + sqrt(exp(-x)) - 4, x) == \
        FiniteSet(2*log(-sqrt(3) + 2), 2*log(sqrt(3) + 2))


# 测试函数，检查是否为指数函数
def test_is_exponential():
    # 断言 _is_exponential 函数对不同输入的返回值是否正确
    assert _is_exponential(y, x) is False
    assert _is_exponential(3**x - 2, x) is True
    assert _is_exponential(5**x - 7**(2 - x), x) is True
    assert _is_exponential(sin(2**x) - 4*x, x) is False
    assert _is_exponential(x**y - z, y) is True
    assert _is_exponential(x**y - z, x) is False
    assert _is_exponential(2**x + 4**x - 1, x) is True
    assert _is_exponential(x**(y*z) - x, x) is False
    assert _is_exponential(x**(2*x) - 3**x, x) is False
    assert _is_exponential(x**y - y*z, y) is False
    assert _is_exponential(x**y - x*z, y) is True


# 测试函数，检查指数函数的解析求解器
def test_solve_exponential():
    # 断言 _solve_exponential 函数对不同方程和参数的返回值是否正确
    assert _solve_exponential(3**(2*x) - 2**(x + 3), 0, x, S.Reals) == \
        FiniteSet(-3*log(2)/(-2*log(3) + log(2)))
    assert _solve_exponential(2**y + 4**y, 1, y, S.Reals) == \
        FiniteSet(log(Rational(-1, 2) + sqrt(5)/2)/log(2))
    assert _solve_exponential(2**y + 4**y, 0, y, S.Reals) == \
        S.EmptySet
    assert _solve_exponential(2**x + 3**x - 5**x, 0, x, S.Reals) == \
        ConditionSet(x, Eq(2**x + 3**x - 5**x, 0), S.Reals)

# 指数函数测试结束


# 对数函数测试
def test_logarithmic():
    # 断言 solveset_real 函数对不同对数方程的解集是否正确
    assert solveset_real(log(x - 3) + log(x + 3), x) == FiniteSet(
        -sqrt(10), sqrt(10))
    assert solveset_real(log(x + 1) - log(2*x - 1), x) == FiniteSet(2)
    assert solveset_real(log(x + 3) + log(1 + 3/x) - 3, x) == FiniteSet(
        -3 + sqrt(-12 + exp(3))*exp(Rational(3, 2))/2 + exp(3)/2,
        -sqrt(-12 + exp(3))*exp(Rational(3, 2))/2 - 3 + exp(3)/2)

    eq = z - log(x) + log(y/(x*(-1 + y**2/x**2)))
    assert solveset_real(eq, x) == \
        Intersection(S.Reals, FiniteSet(-sqrt(y**2 - y*exp(z)),
            sqrt(y**2 - y*exp(z)))) - \
        Intersection(S.Reals, FiniteSet(-sqrt(y**2), sqrt(y**2)))
    assert solveset_real(
        log(3*x) - log(-x + 1) - log(4*x + 1), x) == FiniteSet(Rational(-1, 2), S.Half)
    assert solveset(log(x**y) - y*log(x), x, S.Reals) == S.Reals

# 期望失败的测试用例，检查 log 合并的情况
@XFAIL
def test_uselogcombine_2():
    eq = log(exp(2*x) + 1) + log(-tanh(x) + 1) - log(2)
    assert solveset_real(eq, x) is S.EmptySet
    eq = log(8*x) - log(sqrt(x) + 1) - 2
    assert solveset_real(eq, x) is S.EmptySet


# 测试函数，检查是否为对数函数
def test_is_logarithmic():
    # 断言 _is_logarithmic 函数对不同输入的返回值是否正确
    assert _is_logarithmic(y, x) is False
    assert _is_logarithmic(log(x), x) is True
    assert _is_logarithmic(log(x) - 3, x) is True
    assert _is_logarithmic(log(x)*log(y), x) is True
    assert _is_logarithmic(log(x)**2, x) is False
    # 调用 _is_logarithmic 函数，检查 log(x - 3) + log(x + 3) 是否满足对 x 的对数特性
    assert _is_logarithmic(log(x - 3) + log(x + 3), x) is True
    
    # 调用 _is_logarithmic 函数，检查 log(x**y) - y*log(x) 是否满足对 x 的对数特性
    assert _is_logarithmic(log(x**y) - y*log(x), x) is True
    
    # 调用 _is_logarithmic 函数，检查 sin(log(x)) 是否满足对 x 的对数特性
    assert _is_logarithmic(sin(log(x)), x) is False
    
    # 调用 _is_logarithmic 函数，检查 x + y 是否满足对 x 的对数特性
    assert _is_logarithmic(x + y, x) is False
    
    # 调用 _is_logarithmic 函数，检查 log(3*x) - log(1 - x) + 4 是否满足对 x 的对数特性
    assert _is_logarithmic(log(3*x) - log(1 - x) + 4, x) is True
    
    # 调用 _is_logarithmic 函数，检查 log(x) + log(y) + x 是否满足对 x 的对数特性
    assert _is_logarithmic(log(x) + log(y) + x, x) is False
    
    # 调用 _is_logarithmic 函数，检查 log(log(x - 3)) + log(x - 3) 是否满足对 x 的对数特性
    assert _is_logarithmic(log(log(x - 3)) + log(x - 3), x) is True
    
    # 调用 _is_logarithmic 函数，检查 log(log(3) + x) + log(x) 是否满足对 x 的对数特性
    assert _is_logarithmic(log(log(3) + x) + log(x), x) is True
    
    # 调用 _is_logarithmic 函数，检查 log(x)*(y + 3) + log(x) 是否满足对 y 的对数特性
    assert _is_logarithmic(log(x)*(y + 3) + log(x), y) is False
# 定义一个测试函数，用于测试解对数的函数
def test_solve_logarithm():
    # 创建符号变量 y
    y = Symbol('y')
    # 断言求解 log(x**y) - y*log(x) = 0 对 x 在实数域上的解是整个实数集
    assert _solve_logarithm(log(x**y) - y*log(x), 0, x, S.Reals) == S.Reals
    # 设置 y 为正数
    y = Symbol('y', positive=True)
    # 断言求解 log(x)*log(y) = 0 对 x 在实数域上的解是有限集 {1}
    assert _solve_logarithm(log(x)*log(y), 0, x, S.Reals) == FiniteSet(1)

# logarithmic tests 的结束


# lambert tests
def test_is_lambert():
    # 定义符号变量 a, b, c
    a, b, c = symbols('a,b,c')
    # 断言检查 x**2 是否 Lambertain
    assert _is_lambert(x**2, x) is False
    # 断言检查 a**x**2 + b*x + c 是否 Lambertain
    assert _is_lambert(a**x**2+b*x+c, x) is True
    # 断言检查 E**2 是否 Lambertain
    assert _is_lambert(E**2, x) is False
    # 断言检查 x*E**2 是否 Lambertain
    assert _is_lambert(x*E**2, x) is False
    # 断言检查 3*log(x) - x*log(3) 是否 Lambertain
    assert _is_lambert(3*log(x) - x*log(3), x) is True
    # 断言检查 log(log(x - 3)) + log(x-3) 是否 Lambertain
    assert _is_lambert(log(log(x - 3)) + log(x-3), x) is True
    # 断言检查 5*x - 1 + 3*exp(2 - 7*x) 是否 Lambertain
    assert _is_lambert(5*x - 1 + 3*exp(2 - 7*x), x) is True
    # 断言检查 (a/x + exp(x/2)).diff(x, 2) 是否 Lambertain
    assert _is_lambert((a/x + exp(x/2)).diff(x, 2), x) is True
    # 断言检查 (x**2 - 2*x + 1).subs(x, (log(x) + 3*x)**2 - 1) 是否 Lambertain
    assert _is_lambert((x**2 - 2*x + 1).subs(x, (log(x) + 3*x)**2 - 1), x) is True
    # 断言检查 x*sinh(x) - 1 是否 Lambertain
    assert _is_lambert(x*sinh(x) - 1, x) is True
    # 断言检查 x*cos(x) - 5 是否 Lambertain
    assert _is_lambert(x*cos(x) - 5, x) is True
    # 断言检查 tanh(x) - 5*x 是否 Lambertain
    assert _is_lambert(tanh(x) - 5*x, x) is True
    # 断言检查 cosh(x) - sinh(x) 是否 Lambertain
    assert _is_lambert(cosh(x) - sinh(x), x) is False

# lambert tests 的结束


# 定义测试线性系数的函数
def test_linear_coeffs():
    # 导入线性系数函数
    from sympy.solvers.solveset import linear_coeffs
    # 断言检查 linear_coeffs(0, x) 返回 [0, 0]
    assert linear_coeffs(0, x) == [0, 0]
    # 断言检查 linear_coeffs(0, x) 的所有元素都是 S.Zero
    assert all(i is S.Zero for i in linear_coeffs(0, x))
    # 断言检查 linear_coeffs(x + 2*y + 3, x, y) 返回 [1, 2, 3]
    assert linear_coeffs(x + 2*y + 3, x, y) == [1, 2, 3]
    # 断言检查 linear_coeffs(x + 2*y + 3, y, x) 返回 [2, 1, 3]
    assert linear_coeffs(x + 2*y + 3, y, x) == [2, 1, 3]
    # 断言检查 linear_coeffs(x + 2*x**2 + 3, x, x**2) 返回 [1, 2, 3]
    assert linear_coeffs(x + 2*x**2 + 3, x, x**2) == [1, 2, 3]
    # 断言检查 linear_coeffs(x + 2*x**2 + x**3, x, x**2) 引发 ValueError 异常
    raises(ValueError, lambda:
        linear_coeffs(x + 2*x**2 + x**3, x, x**2))
    # 断言检查 linear_coeffs(1/x*(x - 1) + 1/x, x) 引发 ValueError 异常
    raises(ValueError, lambda:
        linear_coeffs(1/x*(x - 1) + 1/x, x))
    # 断言检查 linear_coeffs(x, x, x) 引发 ValueError 异常
    raises(ValueError, lambda:
        linear_coeffs(x, x, x))
    # 断言检查 linear_coeffs(a*(x + y), x, y) 返回 [a, a, 0]
    assert linear_coeffs(a*(x + y), x, y) == [a, a, 0]
    # 断言检查 linear_coeffs(1.0, x, y) 返回 [0, 0, 1.0]
    assert linear_coeffs(1.0, x, y) == [0, 0, 1.0]
    # 断言检查 linear_coeffs(Eq(x, x + y), x, y, dict=True) 返回 {y: -1}
    assert linear_coeffs(Eq(x, x + y), x, y, dict=True) == {y: -1}
    # 断言检查 linear_coeffs(0, x, y, dict=True) 返回 {}
    assert linear_coeffs(0, x, y, dict=True) == {}

# end of linear_coeffs tests 的结束


# 定义测试模块化函数
def test_is_modular():
    # 断言检查 _is_modular(y, x) 返回 False
    assert _is_modular(y, x) is False
    # 断言检查 _is_modular(Mod(x, 3) - 1, x) 返回 True
    assert _is_modular(Mod(x, 3) - 1, x) is True
    # 断言检查 _is_modular(Mod(x**3 - 3*x**2 - x + 1, 3) - 1, x) 返回 True
    assert _is_modular(Mod(x**3 - 3*x**2 - x + 1, 3) - 1, x) is True
    # 断言检查 _is_modular(Mod(exp(x + y), 3) - 2, x) 返回 True
    assert _is_modular(Mod(exp(x + y), 3) - 2, x) is True
    # 断言检查 _is_modular(Mod(exp(x + y), 3) - log(x), x) 返回 True
    assert _is_modular(Mod(exp(x + y), 3) - log(x), x) is True
    # 断言检查 _is_modular(Mod(x, 3) - 1, y) 返回 False
    assert _is_modular(Mod(x, 3) - 1, y) is False
    # 断言检查 _is_modular(Mod(x, 3)**2 - 5, x) 返回 False
    assert _is_modular(Mod(x, 3)**2 - 5, x) is False
    # 断言检查 _is_modular(Mod(x, 3)**2 - y, x) 返回 False
    assert _is_modular(Mod(x, 3)**2 - y, x) is False
    # 断言检查 _is_modular(exp(Mod(x, 3)) - 1, x) 返回 False
    assert _is_modular(exp(Mod(x, 3)) - 1, x) is False
    # 断言检查 _is_modular(Mod(3, y) - 1, y) 返回 False
    assert _is_modular(Mod(3, y) - 1, y) is False

# end of modular tests 的结束


# 定义测试模块化反函数
def test_invert_modular():
    # 定义整数符号变量 n
    # 检查模反演函数对于给定的模数、余数和表达式的正确性
    assert dumeq(invert_modular(Mod(x, 7), S(5), n, x),
                (x, ImageSet(Lambda(n, 7*n + 5), S.Integers)))
    # 对表达式进行检查，确认是否为加法表达式
    assert dumeq(invert_modular(Mod(x + 8, 7), S(5), n, x),
                (x, ImageSet(Lambda(n, 7*n + 4), S.Integers)))
    # 检查模反演函数对于给定的二次多项式表达式的正确性
    assert invert_modular(Mod(x**2 + x, 7), S(5), n, x) == \
                (Mod(x**2 + x, 7), 5)
    # 对表达式进行检查，确认是否为乘法表达式
    assert dumeq(invert_modular(Mod(3*x, 7), S(5), n, x),
                (x, ImageSet(Lambda(n, 7*n + 4), S.Integers)))
    # 检查模反演函数对于给定的幂次表达式的正确性
    assert invert_modular(Mod((x + 1)*(x + 2), 7), S(5), n, x) == \
                (Mod((x + 1)*(x + 2), 7), 5)
    # 检查模反演函数对于给定的四次幂次表达式的正确性
    assert invert_modular(Mod(x**4, 7), S(5), n, x) == \
                (x, S.EmptySet)
    # 检查模反演函数对于给定的指数表达式的正确性
    assert dumeq(invert_modular(Mod(3**x, 4), S(3), n, x),
                (x, ImageSet(Lambda(n, 2*n + 1), S.Naturals0)))
    # 检查模反演函数对于给定的指数幂次表达式的正确性
    assert dumeq(invert_modular(Mod(2**(x**2 + x + 1), 7), S(2), n, x),
                (x**2 + x + 1, ImageSet(Lambda(n, 3*n + 1), S.Naturals0)))
    # 检查模反演函数对于给定的三角函数的幂次表达式的正确性
    assert invert_modular(Mod(sin(x)**4, 7), S(5), n, x) == (x, S.EmptySet)
def test_solve_modular():
    # 创建一个名为 n 的虚拟变量，限定为整数
    n = Dummy('n', integer=True)
    # 解决模方程 Mod(x, 4) - x = 0，期望得到整数解集
    assert solveset(Mod(x, 4) - x, x, S.Integers
        ).dummy_eq(
            ConditionSet(x, Eq(-x + Mod(x, 4), 0),
            S.Integers))
    # 当 _invert_modular 无法反转时
    assert solveset(3 - Mod(sin(x), 7), x, S.Integers
        ).dummy_eq(
            ConditionSet(x, Eq(Mod(sin(x), 7) - 3, 0), S.Integers))
    # 解决模方程 3 - Mod(log(x), 7) = 0，期望得到整数解集
    assert solveset(3 - Mod(log(x), 7), x, S.Integers
        ).dummy_eq(
            ConditionSet(x, Eq(Mod(log(x), 7) - 3, 0), S.Integers))
    # 解决模方程 3 - Mod(exp(x), 7) = 0，期望得到整数解集
    assert solveset(3 - Mod(exp(x), 7), x, S.Integers
        ).dummy_eq(ConditionSet(x, Eq(Mod(exp(x), 7) - 3, 0),
        S.Integers))
    # 解决模方程 7 - Mod(x, 5) = 0，期望得到空解集
    assert solveset(7 - Mod(x, 5), x, S.Integers) is S.EmptySet
    # 解决模方程 5 - Mod(x, 5) = 0，期望得到空解集
    assert solveset(5 - Mod(x, 5), x, S.Integers) is S.EmptySet
    # 负 m 的情况
    assert dumeq(solveset(2 + Mod(x, -3), x, S.Integers),
            ImageSet(Lambda(n, -3*n - 2), S.Integers))
    # 解决线性表达式的模方程 3 - Mod(x, 5) = 0，期望得到整数解集
    assert dumeq(solveset(3 - Mod(x, 5), x, S.Integers),
        ImageSet(Lambda(n, 5*n + 3), S.Integers))
    # 解决线性表达式的模方程 3 - Mod(5*x - 8, 7) = 0，期望得到整数解集
    assert dumeq(solveset(3 - Mod(5*x - 8, 7), x, S.Integers),
                ImageSet(Lambda(n, 7*n + 5), S.Integers))
    # 解决线性表达式的模方程 3 - Mod(5*x, 7) = 0，期望得到整数解集
    assert dumeq(solveset(3 - Mod(5*x, 7), x, S.Integers),
                ImageSet(Lambda(n, 7*n + 2), S.Integers))
    # 解决高次表达式的模方程 Mod(x**2, 160) - 9 = 0，期望得到整数解集的并集
    assert dumeq(solveset(Mod(x**2, 160) - 9, x, S.Integers),
            Union(ImageSet(Lambda(n, 160*n + 3), S.Integers),
            ImageSet(Lambda(n, 160*n + 13), S.Integers),
            ImageSet(Lambda(n, 160*n + 67), S.Integers),
            ImageSet(Lambda(n, 160*n + 77), S.Integers),
            ImageSet(Lambda(n, 160*n + 83), S.Integers),
            ImageSet(Lambda(n, 160*n + 93), S.Integers),
            ImageSet(Lambda(n, 160*n + 147), S.Integers),
            ImageSet(Lambda(n, 160*n + 157), S.Integers)))
    # 解决高次表达式的模方程 3 - Mod(x**4, 7) = 0，期望得到空解集
    assert solveset(3 - Mod(x**4, 7), x, S.Integers) is S.EmptySet
    # 解决高次表达式的模方程 Mod(x**4, 17) - 13 = 0，期望得到整数解集的并集
    assert dumeq(solveset(Mod(x**4, 17) - 13, x, S.Integers),
            Union(ImageSet(Lambda(n, 17*n + 3), S.Integers),
            ImageSet(Lambda(n, 17*n + 5), S.Integers),
            ImageSet(Lambda(n, 17*n + 12), S.Integers),
            ImageSet(Lambda(n, 17*n + 14), S.Integers)))
    # 解决 a.is_Pow 测试
    assert dumeq(solveset(Mod(7**x, 41) - 15, x, S.Integers),
            ImageSet(Lambda(n, 40*n + 3), S.Naturals0))
    assert dumeq(solveset(Mod(12**x, 21) - 18, x, S.Integers),
            ImageSet(Lambda(n, 6*n + 2), S.Naturals0))
    assert dumeq(solveset(Mod(3**x, 4) - 3, x, S.Integers),
            ImageSet(Lambda(n, 2*n + 1), S.Naturals0))
    assert dumeq(solveset(Mod(2**x, 7) - 2 , x, S.Integers),
            ImageSet(Lambda(n, 3*n + 1), S.Naturals0))
    # 断言：解决模方程 Mod(3**(3**x), 4) - 3 = 0 在整数集合下的解集与交集的比较
    assert dumeq(solveset(Mod(3**(3**x), 4) - 3, x, S.Integers),
            Intersection(ImageSet(Lambda(n, Intersection({log(2*n + 1)/log(3)},
            S.Integers)), S.Naturals0), S.Integers))
    
    # 断言：解决模方程 Mod(x**3, 7) - 2 = 0 在整数集合下的解集为空集
    assert solveset(Mod(x**3, 7) - 2, x, S.Integers) is S.EmptySet
    
    # 断言：解决模方程 Mod(x**3, 8) - 1 = 0 在整数集合下的解集与映射集的比较
    assert dumeq(solveset(Mod(x**3, 8) - 1, x, S.Integers),
            ImageSet(Lambda(n, 8*n + 1), S.Integers))
    
    # 断言：解决模方程 Mod(x**4, 9) - 4 = 0 在整数集合下的解集与两个映射集的并集的比较
    assert dumeq(solveset(Mod(x**4, 9) - 4, x, S.Integers),
            Union(ImageSet(Lambda(n, 9*n + 4), S.Integers),
            ImageSet(Lambda(n, 9*n + 5), S.Integers)))
    
    # 断言：解决模方程 3 - Mod(5*x - 8, 7) = 0 在自然数集合下的解集与交集的比较
    assert dumeq(solveset(3 - Mod(5*x - 8, 7), x, S.Naturals0),
            Intersection(ImageSet(Lambda(n, 7*n + 5), S.Integers), S.Naturals0))
    
    # 断言：解决模方程 Mod(x, 3) - I = 0 在整数集合下的解集为空集
    assert solveset(Mod(x, 3) - I, x, S.Integers) == \
            S.EmptySet
    
    # 断言：解决模方程 Mod(I*x, 3) - 2 = 0 在整数集合下的解集与条件集的比较
    assert solveset(Mod(I*x, 3) - 2, x, S.Integers
        ).dummy_eq(
            ConditionSet(x, Eq(Mod(I*x, 3) - 2, 0), S.Integers))
    
    # 断言：解决模方程 Mod(I + x, 3) - 2 = 0 在整数集合下的解集与条件集的比较
    assert solveset(Mod(I + x, 3) - 2, x, S.Integers
        ).dummy_eq(
            ConditionSet(x, Eq(Mod(x + I, 3) - 2, 0), S.Integers))
    
    # 断言：解决模方程 Mod(x**4, 14) - 11 = 0 在整数集合下的解集与两个映射集的并集的比较
    assert dumeq(solveset(Mod(x**4, 14) - 11, x, S.Integers),
            Union(ImageSet(Lambda(n, 14*n + 3), S.Integers),
            ImageSet(Lambda(n, 14*n + 11), S.Integers)))
    
    # 断言：解决模方程 Mod(x**31, 74) - 43 = 0 在整数集合下的解集与映射集的比较
    assert dumeq(solveset(Mod(x**31, 74) - 43, x, S.Integers),
            ImageSet(Lambda(n, 74*n + 31), S.Integers))
    
    # 断言：解决模方程 c - Mod(a**n*b, m) = 0 在整数集合下的解集与映射集的比较
    assert dumeq(solveset(c - Mod(a**n*b, m), n, S.Integers),
            ImageSet(Lambda(n, 2147483646*n + 100), S.Naturals0))
    
    # 断言：解决模方程 c - Mod(a**n*b, m) = 0 在自然数集合下的解集与交集的比较
    assert dumeq(solveset(c - Mod(a**n*b, m), n, S.Naturals0),
            Intersection(ImageSet(Lambda(n, 2147483646*n + 100), S.Naturals0),
            S.Naturals0))
    
    # 断言：解决模方程 c - Mod(a**(2*n)*b, m) = 0 在整数集合下的解集与交集的比较
    assert dumeq(solveset(c - Mod(a**(2*n)*b, m), n, S.Integers),
            Intersection(ImageSet(Lambda(n, 1073741823*n + 50), S.Naturals0),
            S.Integers))
    
    # 断言：解决模方程 c - Mod(a**(2*n + 7)*b, m) = 0 在整数集合下的解集为空集
    assert solveset(c - Mod(a**(2*n + 7)*b, m), n, S.Integers) is S.EmptySet
    
    # 断言：解决模方程 c - Mod(a**(n - 4)*b, m) = 0 在整数集合下的解集与交集的比较
    assert dumeq(solveset(c - Mod(a**(n - 4)*b, m), n, S.Integers),
            Intersection(ImageSet(Lambda(n, 2147483646*n + 104), S.Naturals0),
            S.Integers))
# end of modular tests
# 模块测试结束

def test_issue_17276():
    # 测试解决非线性方程组的函数，验证是否得到期望的解
    assert nonlinsolve([Eq(x, 5**(S(1)/5)), Eq(x*y, 25*sqrt(5))], x, y) == \
     FiniteSet((5**(S(1)/5), 25*5**(S(3)/10)))


def test_issue_10426():
    # 测试解决方程 sin(x + a) - sin(x) = 0，验证是否得到期望的解
    x = Dummy('x')
    a = Symbol('a')
    n = Dummy('n')
    assert (solveset(sin(x + a) - sin(x), a)).dummy_eq(Dummy('x')) == (Union(
        ImageSet(Lambda(n, 2*n*pi), S.Integers),
        Intersection(S.Complexes, ImageSet(Lambda(n, -I*(I*(2*n*pi + arg(-exp(-2*I*x))) + 2*im(x))),
        S.Integers)))).dummy_eq(Dummy('x,n'))


def test_solveset_conjugate():
    """Test solveset for simple conjugate functions"""
    # 测试解决共轭函数的方程，验证是否得到期望的解
    assert solveset(conjugate(x) -3 + I) == FiniteSet(3 + I)


def test_issue_18208():
    # 测试解决线性方程组的不同方法
    variables = symbols('x0:16') + symbols('y0:12')
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15,\
    y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11 = variables

    # 给定的线性方程组
    eqs = [x0 + x1 + x2 + x3 - 51,
           x0 + x1 + x4 + x5 - 46,
           x2 + x3 + x6 + x7 - 39,
           x0 + x3 + x4 + x7 - 50,
           x1 + x2 + x5 + x6 - 35,
           x4 + x5 + x6 + x7 - 34,
           x4 + x5 + x8 + x9 - 46,
           x10 + x11 + x6 + x7 - 23,
           x11 + x4 + x7 + x8 - 25,
           x10 + x5 + x6 + x9 - 44,
           x10 + x11 + x8 + x9 - 35,
           x12 + x13 + x8 + x9 - 35,
           x10 + x11 + x14 + x15 - 29,
           x11 + x12 + x15 + x8 - 35,
           x10 + x13 + x14 + x9 - 29,
           x12 + x13 + x14 + x15 - 29,
           y0 + y1 + y2 + y3 - 55,
           y0 + y1 + y4 + y5 - 53,
           y2 + y3 + y6 + y7 - 56,
           y0 + y3 + y4 + y7 - 57,
           y1 + y2 + y5 + y6 - 52,
           y4 + y5 + y6 + y7 - 54,
           y4 + y5 + y8 + y9 - 48,
           y10 + y11 + y6 + y7 - 60,
           y11 + y4 + y7 + y8 - 51,
           y10 + y5 + y6 + y9 - 57,
           y10 + y11 + y8 + y9 - 54,
           x10 - 2,
           x11 - 5,
           x12 - 1,
           x13 - 6,
           x14 - 1,
           x15 - 21,
           y0 - 12,
           y1 - 20]

    # 期望的结果
    expected = [38 - x3, x3 - 10, 23 - x3, x3, 12 - x7, x7 + 6, 16 - x7, x7,
                8, 20, 2, 5, 1, 6, 1, 21, 12, 20, -y11 + y9 + 2, y11 - y9 + 21,
                -y11 - y7 + y9 + 24, y11 + y7 - y9 - 3, 33 - y7, y7, 27 - y9, y9,
                27 - y11, y11]

    # 将线性方程转化为矩阵形式
    A, b = linear_eq_to_matrix(eqs, variables)

    # 解决线性方程组并验证结果
    # solve 函数的期望结果
    solve_expected = {v:eq for v, eq in zip(variables, expected) if v != eq}
    assert solve(eqs, variables) == solve_expected

    # linsolve 函数的期望结果
    linsolve_expected = FiniteSet(Tuple(*expected))
    assert linsolve(eqs, variables) == linsolve_expected
    assert linsolve((A, b), variables) == linsolve_expected

    # gauss_jordan_solve 函数的期望结果
    gj_solve, new_vars = A.gauss_jordan_solve(b)
    gj_solve = list(gj_solve)
    gj_expected = linsolve_expected.subs(zip([x3, x7, y7, y9, y11], new_vars))
    assert FiniteSet(Tuple(*gj_solve)) == gj_expected

    # nonlinsolve
    # 非线性方程组的解集当前与线性解集等效且正确。然而，我们希望在可能的情况下使用与参数解决超定系统相同的符号。
    # 我们希望的解决方案不仅等效而且以相同的形式给出。如果非线性解决方案以这种方式修改，此测试可能会更改。
    
    nonlinsolve_expected = FiniteSet((38 - x3, x3 - 10, 23 - x3, x3, 12 - x7, x7 + 6,
                                      16 - x7, x7, 8, 20, 2, 5, 1, 6, 1, 21, 12, 20,
                                      -y5 + y7 - 1, y5 - y7 + 24, 21 - y5, y5, 33 - y7,
                                      y7, 27 - y9, y9, -y5 + y7 - y9 + 24, y5 - y7 + y9 + 3))
    # 设置预期的非线性解决方案集合，包含一系列代表变量的表达式。
    
    assert nonlinsolve(eqs, variables) == nonlinsolve_expected
    # 断言实际解决方案与预期的非线性解决方案集合相等。
# 定义测试函数，用于验证对于不可行解的替换
def test_substitution_with_infeasible_solution():
    # 定义符号变量
    a00, a01, a10, a11, l0, l1, l2, l3, m0, m1, m2, m3, m4, m5, m6, m7, c00, c01, c10, c11, p00, p01, p10, p11 = symbols(
        'a00, a01, a10, a11, l0, l1, l2, l3, m0, m1, m2, m3, m4, m5, m6, m7, c00, c01, c10, c11, p00, p01, p10, p11'
    )
    # 定义要解决的变量列表
    solvefor = [p00, p01, p10, p11, c00, c01, c10, c11, m0, m1, m3, l0, l1, l2, l3]
    # 定义非线性系统方程
    system = [
        -l0 * c00 - l1 * c01 + m0 + c00 + c01,
        -l0 * c10 - l1 * c11 + m1,
        -l2 * c00 - l3 * c01 + c00 + c01,
        -l2 * c10 - l3 * c11 + m3,
        -l0 * p00 - l2 * p10 + p00 + p10,
        -l1 * p00 - l3 * p10 + p00 + p10,
        -l0 * p01 - l2 * p11,
        -l1 * p01 - l3 * p11,
        -a00 + c00 * p00 + c10 * p01,
        -a01 + c01 * p00 + c11 * p01,
        -a10 + c00 * p10 + c10 * p11,
        -a11 + c01 * p10 + c11 * p11,
        -m0 * p00,
        -m1 * p01,
        -m2 * p10,
        -m3 * p11,
        -m4 * c00,
        -m5 * c01,
        -m6 * c10,
        -m7 * c11,
        m2,
        m4,
        m5,
        m6,
        m7
    ]
    # 解非线性方程组
    sol = FiniteSet(
        (0, Complement(FiniteSet(p01), FiniteSet(0)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, l2, l3),
        (p00, Complement(FiniteSet(p01), FiniteSet(0)), 0, p11, 0, 0, 0, 0, 0, 0, 0, 1, 1, -p01/p11, -p01/p11),
        (0, Complement(FiniteSet(p01), FiniteSet(0)), 0, p11, 0, 0, 0, 0, 0, 0, 0, 1, -l3*p11/p01, -p01/p11, l3),
        (0, Complement(FiniteSet(p01), FiniteSet(0)), 0, p11, 0, 0, 0, 0, 0, 0, 0, -l2*p11/p01, -l3*p11/p01, l2, l3),
    )
    # 断言求解结果与非线性方程组的不相等性
    assert sol != nonlinsolve(system, solvefor)


# 验证解集为空集的问题
def test_issue_20097():
    assert solveset(1/sqrt(x)) is S.EmptySet


# 验证对求导结果集的问题
def test_issue_15350():
    assert solveset(diff(sqrt(1/x+x))) == FiniteSet(-1, 1)


# 验证 Piecewise 函数求解问题
def test_issue_18359():
    c1 = Piecewise((0, x < 0), (Min(1, x)/2 - Min(2, x)/2 + Min(3, x)/2, True))
    c2 = Piecewise((Piecewise((0, x < 0), (Min(1, x)/2 - Min(2, x)/2 + Min(3, x)/2, True)), x >= 0), (0, True))
    correct_result = Interval(1, 2)
    result1 = solveset(c1 - Rational(1, 2), x, Interval(0, 3))
    result2 = solveset(c2 - Rational(1, 2), x, Interval(0, 3))
    assert result1 == correct_result
    assert result2 == correct_result


# 验证指数函数求解问题
def test_issue_17604():
    lhs = -2**(3*x/11)*exp(x/11) + pi**(x/11)
    assert _is_exponential(lhs, x)
    assert _solve_exponential(lhs, 0, x, S.Complexes) == FiniteSet(0)


# 验证分式函数求解问题
def test_issue_17580():
    assert solveset(1/(1 - x**3)**2, x, S.Reals) is S.EmptySet


# 验证非线性方程组求解问题
def test_issue_17566_actual():
    sys = [2**x + 2**y - 3, 4**x + 9**y - 5]
    # 不确定这是正确的结果，但至少没有递归错误
    assert nonlinsolve(sys, x, y) == FiniteSet((log(3 - 2**y)/log(2), y))


# 验证不等式求解问题
def test_issue_17565():
    eq = Ge(2*(x - 2)**2/(3*(x + 1)**(Integer(1)/3)) + 2*(x - 2)*(x + 1)**(Integer(2)/3), 0)
    res = Union(Interval.Lopen(-1, -Rational(1, 4)), Interval(2, oo))
    assert solveset(eq, x, S.Reals) == res


# 验证函数求解问题
def test_issue_15024():
    function = (x + 5)/sqrt(-x**2 - 10*x)
    # 使用断言来验证解集求解器的结果是否等于有限集 {Integer(-5)}
    assert solveset(function, x, S.Reals) == FiniteSet(Integer(-5))
def test_issue_16877():
    assert dumeq(nonlinsolve([x - 1, sin(y)], x, y),
                 FiniteSet((1, ImageSet(Lambda(n, 2*n*pi), S.Integers)),
                           (1, ImageSet(Lambda(n, 2*n*pi + pi), S.Integers))))
    # 检验非线性方程组的解是否符合预期
    # 对于给定的方程组 [x - 1, sin(y)], 求解变量 x 和 y
    # 预期的解集应为 {(1, ImageSet(Lambda(n, 2*n*pi), S.Integers)),
    #              (1, ImageSet(Lambda(n, 2*n*pi + pi), S.Integers))}


def test_issue_16876():
    assert dumeq(nonlinsolve([sin(x), 2*x - 4*y], x, y),
                 FiniteSet((ImageSet(Lambda(n, 2*n*pi), S.Integers),
                            ImageSet(Lambda(n, n*pi), S.Integers)),
                           (ImageSet(Lambda(n, 2*n*pi + pi), S.Integers),
                            ImageSet(Lambda(n, n*pi + pi/2), S.Integers))))
    # 检验非线性方程组的解是否符合预期
    # 对于给定的方程组 [sin(x), 2*x - 4*y], 求解变量 x 和 y
    # 预期的解集应为 {(ImageSet(Lambda(n, 2*n*pi), S.Integers),
    #              ImageSet(Lambda(n, n*pi), S.Integers)),
    #              (ImageSet(Lambda(n, 2*n*pi + pi), S.Integers),
    #              ImageSet(Lambda(n, n*pi + pi/2), S.Integers))}


def test_issue_21236():
    x, z = symbols("x z")
    y = symbols('y', rational=True)
    assert solveset(x**y - z, x, S.Reals) == ConditionSet(x, Eq(x**y - z, 0), S.Reals)
    # 检验解集是否符合预期
    # 对于方程 x**y - z = 0，求解变量 x
    # 预期的解集应为 ConditionSet(x, Eq(x**y - z, 0), S.Reals)


def test_issue_21908():
    assert nonlinsolve([(x**2 + 2*x - y**2)*exp(x), -2*y*exp(x)], x, y
                      ) == {(-2, 0), (0, 0)}
    # 检验非线性方程组的解是否符合预期
    # 对于给定的方程组 [(x**2 + 2*x - y**2)*exp(x), -2*y*exp(x)], 求解变量 x 和 y
    # 预期的解集应为 {(-2, 0), (0, 0)}


def test_issue_19144():
    # test case 1
    expr1 = [x + y - 1, y**2 + 1]
    eq1 = [Eq(i, 0) for i in expr1]
    soln1 = {(1 - I, I), (1 + I, -I)}
    soln_expr1 = nonlinsolve(expr1, [x, y])
    soln_eq1 = nonlinsolve(eq1, [x, y])
    assert soln_eq1 == soln_expr1 == soln1
    # 检验非线性方程组的解是否符合预期
    # 测试用例 1：
    # 对于方程组 [x + y - 1, y**2 + 1]，分别检验直接方程和等式形式的求解结果是否与预期一致

    # test case 2 - with denoms
    expr2 = [x/y - 1, y**2 + 1]
    eq2 = [Eq(i, 0) for i in expr2]
    soln2 = {(-I, -I), (I, I)}
    soln_expr2 = nonlinsolve(expr2, [x, y])
    soln_eq2 = nonlinsolve(eq2, [x, y])
    assert soln_eq2 == soln_expr2 == soln2
    # 检验非线性方程组的解是否符合预期
    # 测试用例 2 - 包含分母：
    # 对于方程组 [x/y - 1, y**2 + 1]，分别检验直接方程和等式形式的求解结果是否与预期一致

    # denominators that cancel in expression
    assert nonlinsolve([Eq(x + 1/x, 1/x)], [x]) == FiniteSet((S.EmptySet,))
    # 检验非线性方程组的解是否符合预期
    # 对于方程组 [Eq(x + 1/x, 1/x)]，求解变量 x
    # 预期的解集应为 {EmptySet}


def test_issue_22413():
    res =  nonlinsolve((4*y*(2*x + 2*exp(y) + 1)*exp(2*x),
                         4*x*exp(2*x) + 4*y*exp(2*x + y) + 4*exp(2*x + y) + 1),
                        x, y)
    # 检验非线性方程组的解是否符合预期
    # 对于给定的方程组 (4*y*(2*x + 2*exp(y) + 1)*exp(2*x),
    #                  4*x*exp(2*x) + 4*y*exp(2*x + y) + 4*exp(2*x + y) + 1)，求解变量 x 和 y
    # 预期的解集应为 {(x, 0), (-exp(y) - S.Half, y)}


def test_issue_23318():
    eqs_eq = [
        Eq(53.5780461486929, x * log(y / (5.0 - y) + 1) / y),
        Eq(x, 0.0015 * z),
        Eq(0.0015, 7845.32 * y / z),
    ]
    eqs_expr = [eq.lhs - eq.rhs for eq in eqs_eq]

    sol = {(266.97755814852, 0.0340301680681629, 177985.03876568)}

    assert_close_nl(nonlinsolve(eqs_eq, [x, y, z]), sol)
    # 检验非线性方程组的解是否在数值上接近预期的解集
    # 对于给定的方程组 eqs_eq，求解变量 x, y, z
    # 预期的数值解集应接近 {(266.97755814852, 0.0340301680681629, 177985.03876568)}

    assert_close_nl(nonlinsolve(eqs_expr, [x, y, z]), sol)
    # 检验非线性方程组的解是否在数值上接近预期的解集
    # 对于给定的方程组 eqs_expr（化简为差分形式的方程组），求解变量 x, y, z
    # 预期的数值解集应接近 {(266.97755814852, 0.0340301680681629, 177985.03876568)}

    logterm = log(1.91196789933362e-7*z/(5.0 - 1.91196789933362e-7*z) + 1)
    eq = -0.0015*z*logterm + 1.02439504345316e-5*z
    assert_close_ss(solveset(eq, z), {0, 177985.038765679})
    # 检验解集是否在数值上接近预期的解集
    # 对于方程 solveset(eq, z)，求解变量 z
    # 预期的数值解集应为 {0, 177985.038765679}


def test_issue_19814():
    # 断言语句，用于验证非线性方程组的解
    assert nonlinsolve([2**m - 2**(2*n), 4*2**m - 2**(4*n)], m, n) == FiniteSet((log(2**(2*n))/log(2), S.Complexes))
    # 上述方程组的解应该是一个由元组构成的集合，包含形如 (log(2**(2*n))/log(2), S.Complexes) 的解
def test_issue_22058():
    # 解决方程 -sqrt(t)*x**2 + 2*x + sqrt(t) 关于变量 x 的实数解集
    sol = solveset(-sqrt(t)*x**2 + 2*x + sqrt(t), x, S.Reals)
    # 验证替换 t 为 1 后的解是否等于集合 {1 - sqrt(2), 1 + sqrt(2)}
    assert sol.xreplace({t: 1}) == {1 - sqrt(2), 1 + sqrt(2)}, sol.xreplace({t: 1})


def test_issue_11184():
    # 验证方程 20*sqrt(y**2 + (sqrt(-(y - 10)*(y + 10)) + 10)**2) - 60 是否为空集
    assert solveset(20*sqrt(y**2 + (sqrt(-(y - 10)*(y + 10)) + 10)**2) - 60, y, S.Reals) is S.EmptySet


def test_issue_21890():
    # 初始化符号表达式 S(2)/3
    e = S(2)/3
    # 验证非线性方程组的解集是否等于给定的集合
    assert nonlinsolve([4*x**3*y**4 - 2*y, 4*x**4*y**3 - 2*x], x, y) == {
        (2**e/(2*y), y), ((-2**e/4 - 2**e*sqrt(3)*I/4)/y, y),
        ((-2**e/4 + 2**e*sqrt(3)*I/4)/y, y)}
    # 验证非线性方程组的解集是否等于给定的集合
    assert nonlinsolve([(1 - 4*x**2)*exp(-2*x**2 - 2*y**2),
        -4*x*y*exp(-2*x**2)*exp(-2*y**2)], x, y) == {(-S(1)/2, 0), (S(1)/2, 0)}
    # 声明实数符号变量 rx, ry
    rx, ry = symbols('x y', real=True)
    # 求解非线性方程组的解集
    sol = nonlinsolve([4*rx**3*ry**4 - 2*ry, 4*rx**4*ry**3 - 2*rx], rx, ry)
    # 期望的解集
    ans = {(2**(S(2)/3)/(2*ry), ry),
        ((-2**(S(2)/3)/4 - 2**(S(2)/3)*sqrt(3)*I/4)/ry, ry),
        ((-2**(S(2)/3)/4 + 2**(S(2)/3)*sqrt(3)*I/4)/ry, ry)}
    # 验证解是否等于期望的解集
    assert sol == ans


def test_issue_22628():
    # 验证非线性方程组的解集是否为空集
    assert nonlinsolve([h - 1, k - 1, f - 2, f - 4, -2*k], h, k, f) == S.EmptySet
    # 验证非线性方程组的解集是否为空集
    assert nonlinsolve([x**3 - 1, x + y, x**2 - 4], [x, y]) == S.EmptySet


def test_issue_25781():
    # 解方程 sqrt(x/2) - x 的解集
    assert solve(sqrt(x/2) - x) == [0, S.Half]


def test_issue_26077():
    # 声明符号变量 _n
    _n = Symbol('_n')
    # 定义函数 x*cot(5*x)
    function = x*cot(5*x)
    # 找到函数的稳定点
    critical_points = stationary_points(function, x, S.Reals)
    # 排除点的集合
    excluded_points = Union(
        ImageSet(Lambda(_n, 2*_n*pi/5), S.Integers),
        ImageSet(Lambda(_n, 2*_n*pi/5 + pi/5), S.Integers)
    )
    # 条件集合的解
    solution = ConditionSet(x,
        Eq(x*(-5*cot(5*x)**2 - 5) + cot(5*x), 0),
        Complement(S.Reals, excluded_points)
    )
    # 验证条件集合的标识化是否等于稳定点的标识化
    assert solution.as_dummy() == critical_points.as_dummy()
```