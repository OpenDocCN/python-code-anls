# `D:\src\scipysrc\sympy\sympy\integrals\tests\test_risch.py`

```
# 导入 Sympy 中的各种模块和函数，用于符号计算
"""Most of these tests come from the examples in Bronstein's book."""
from sympy.core.function import (Function, Lambda, diff, expand_log)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, tan)
from sympy.polys.polytools import (Poly, cancel, factor)
from sympy.integrals.risch import (gcdex_diophantine, frac_in, as_poly_1t,
    derivation, splitfactor, splitfactor_sqf, canonical_representation,
    hermite_reduce, polynomial_reduce, residue_reduce, residue_reduce_to_basic,
    integrate_primitive, integrate_hyperexponential_polynomial,
    integrate_hyperexponential, integrate_hypertangent_polynomial,
    integrate_nonlinear_no_specials, integer_powers, DifferentialExtension,
    risch_integrate, DecrementLevel, NonElementaryIntegral, recognize_log_derivative,
    recognize_derivative, laurent_series)
from sympy.testing.pytest import raises

# 导入 Sympy 中的符号变量 x, t, nu, z, a, y
from sympy.abc import x, t, nu, z, a, y

# 创建符号变量 t0, t1, t2，并赋予它们 t:3 的符号列表
t0, t1, t2 = symbols('t:3')

# 创建符号变量 i，并赋予它 'i' 的符号
i = Symbol('i')

# 定义测试函数 test_gcdex_diophantine，用于测试 gcdex_diophantine 函数
def test_gcdex_diophantine():
    # 断言 gcdex_diophantine 的计算结果与预期相符
    assert gcdex_diophantine(Poly(x**4 - 2*x**3 - 6*x**2 + 12*x + 15),
    Poly(x**3 + x**2 - 4*x - 4), Poly(x**2 - 1)) == \
        (Poly((-x**2 + 4*x - 3)/5), Poly((x**3 - 7*x**2 + 16*x - 10)/5))
    assert gcdex_diophantine(Poly(x**3 + 6*x + 7), Poly(x**2 + 3*x + 2), Poly(x + 1)) == \
        (Poly(1/13, x, domain='QQ'), Poly(-1/13*x + 3/13, x, domain='QQ'))

# 定义测试函数 test_frac_in，用于测试 frac_in 函数
def test_frac_in():
    # 断言 frac_in 函数对给定的多项式进行分数分解，并验证结果
    assert frac_in(Poly((x + 1)/x*t, t), x) == \
        (Poly(t*x + t, x), Poly(x, x))
    assert frac_in((x + 1)/x*t, x) == \
        (Poly(t*x + t, x), Poly(x, x))
    assert frac_in((Poly((x + 1)/x*t, t), Poly(t + 1, t)), x) == \
        (Poly(t*x + t, x), Poly((1 + t)*x, x))
    # 验证 frac_in 函数在遇到无法处理的情况时引发 ValueError 异常
    raises(ValueError, lambda: frac_in((x + 1)/log(x)*t, x))
    assert frac_in(Poly((2 + 2*x + x*(1 + x))/(1 + x)**2, t), x, cancel=True) == \
        (Poly(x + 2, x), Poly(x + 1, x))

# 定义测试函数 test_as_poly_1t，用于测试 as_poly_1t 函数
def test_as_poly_1t():
    # 断言 as_poly_1t 函数对给定的表达式进行 t 到 z 的多项式转换，并验证结果
    assert as_poly_1t(2/t + t, t, z) in [
        Poly(t + 2*z, t, z), Poly(t + 2*z, z, t)]
    assert as_poly_1t(2/t + 3/t**2, t, z) in [
        Poly(2*z + 3*z**2, t, z), Poly(2*z + 3*z**2, z, t)]
    assert as_poly_1t(2/((exp(2) + 1)*t), t, z) in [
        Poly(2/(exp(2) + 1)*z, t, z), Poly(2/(exp(2) + 1)*z, z, t)]
    assert as_poly_1t(2/((exp(2) + 1)*t) + t, t, z) in [
        Poly(t + 2/(exp(2) + 1)*z, t, z), Poly(t + 2/(exp(2) + 1)*z, z, t)]
    assert as_poly_1t(S.Zero, t, z) == Poly(0, t, z)

# 定义测试函数 test_derivation，用于测试 derivation 函数
def test_derivation():
    # 创建多项式 p，并使用 DifferentialExtension 创建 DifferentialExtension 对象 DE
    p = Poly(4*x**4*t**5 + (-4*x**3 - 4*x**4)*t**4 + (-3*x**2 + 2*x**3)*t**3 +
        (2*x + 7*x**2 + 2*x**3)*t**2 + (1 - 4*x - 4*x**2)*t - 1 + 2*x, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t**2 - 3/(2*x)*t + 1/(2*x), t)]})
    # 断言：使用 derivation 函数验证给定的表达式与期望结果的相等性
    assert derivation(p, DE) == Poly(-20*x**4*t**6 + (2*x**3 + 16*x**4)*t**5 +
        (21*x**2 + 12*x**3)*t**4 + (x*Rational(7, 2) - 25*x**2 - 12*x**3)*t**3 +
        (-5 - x*Rational(15, 2) + 7*x**2)*t**2 - (3 - 8*x - 10*x**2 - 4*x**3)/(2*x)*t +
        (1 - 4*x**2)/(2*x), t)
    # 断言：使用 derivation 函数验证对于常数多项式，结果应为多项式 0
    assert derivation(Poly(1, t), DE) == Poly(0, t)
    # 断言：使用 derivation 函数验证对于 t 多项式，结果应为 DE.d
    assert derivation(Poly(t, t), DE) == DE.d
    # 断言：使用 derivation 函数验证给定复杂多项式表达式的导数计算结果
    assert derivation(Poly(t**2 + 1/x*t + (1 - 2*x)/(4*x**2), t), DE) == \
        Poly(-2*t**3 - 4/x*t**2 - (5 - 2*x)/(2*x**2)*t - (1 - 2*x)/(2*x**3), t, domain='ZZ(x)')
    # 设置新的 DifferentialExtension 对象 DE，包含不同的扩展
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t1), Poly(t, t)]})
    # 断言：使用 derivation 函数验证给定表达式的导数计算结果
    assert derivation(Poly(x*t*t1, t), DE) == Poly(t*t1 + x*t*t1 + t, t)
    # 断言：使用 derivation 函数验证给定表达式的导数计算结果，同时考虑系数 D
    assert derivation(Poly(x*t*t1, t), DE, coefficientD=True) == \
        Poly((1 + t1)*t, t)
    # 设置新的 DifferentialExtension 对象 DE，包含不同的扩展
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    # 断言：使用 derivation 函数验证给定表达式的导数计算结果
    assert derivation(Poly(x, x), DE) == Poly(1, x)
    # 断言：使用 derivation 函数验证基本选项下给定表达式的导数计算结果
    assert derivation((x + 1)/(x - 1), DE, basic=True) == -2/(1 - 2*x + x**2)
    # 设置新的 DifferentialExtension 对象 DE，包含不同的扩展
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    # 断言：使用 derivation 函数验证基本选项下给定表达式的导数计算结果
    assert derivation((t + 1)/(t - 1), DE, basic=True) == -2*t/(1 - 2*t + t**2)
    # 断言：使用 derivation 函数验证基本选项下给定表达式的导数计算结果
    assert derivation(t + 1, DE, basic=True) == t
def test_splitfactor():
    # 创建多项式 p，包含多个项的线性组合，定义在有限域上
    p = Poly(4*x**4*t**5 + (-4*x**3 - 4*x**4)*t**4 + (-3*x**2 + 2*x**3)*t**3 +
        (2*x + 7*x**2 + 2*x**3)*t**2 + (1 - 4*x - 4*x**2)*t - 1 + 2*x, t, field=True)
    
    # 创建微分扩展 DE，其扩展中包含了两个多项式
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t**2 - 3/(2*x)*t + 1/(2*x), t)]})
    
    # 断言 splitfactor 函数对 p 在 DE 上的分解结果，应为两个多项式的元组
    assert splitfactor(p, DE) == (Poly(4*x**4*t**3 + (-8*x**3 - 4*x**4)*t**2 +
        (4*x**2 + 8*x**3)*t - 4*x**2, t, domain='ZZ(x)'),
        Poly(t**2 + 1/x*t + (1 - 2*x)/(4*x**2), t, domain='ZZ(x)'))
    
    # 断言 splitfactor 函数对 Poly(x, t) 在 DE 上的分解结果，应为本身和常数多项式的元组
    assert splitfactor(Poly(x, t), DE) == (Poly(x, t), Poly(1, t))
    
    # 创建多项式 r，定义在变量 t 上
    r = Poly(-4*x**4*z**2 + 4*x**6*z**2 - z*x**3 - 4*x**5*z**3 + 4*x**3*z**3 + x**4 + z*x**5 - x**6, t)
    
    # 创建另一个微分扩展 DE，其扩展中包含了两个不同的多项式
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    
    # 断言 splitfactor 函数对 r 在 DE 上的分解结果，应为两个多项式的元组，且包含 D 系数
    assert splitfactor(r, DE, coefficientD=True) == \
        (Poly(x*z - x**2 - z*x**3 + x**4, t), Poly(-x**2 + 4*x**2*z**2, t))
    
    # 断言 splitfactor_sqf 函数对 r 在 DE 上的平方因子分解结果，应为两个元组的元组，每个元组包含一个多项式和其幂次数
    assert splitfactor_sqf(r, DE, coefficientD=True) == \
        (((Poly(x*z - x**2 - z*x**3 + x**4, t), 1),), ((Poly(-x**2 + 4*x**2*z**2, t), 1),))
    
    # 断言 splitfactor 函数对 Poly(0, t) 在 DE 上的分解结果，应为多项式本身和常数多项式的元组
    assert splitfactor(Poly(0, t), DE) == (Poly(0, t), Poly(1, t))
    
    # 断言 splitfactor_sqf 函数对 Poly(0, t) 在 DE 上的平方因子分解结果，应为空的元组的元组
    assert splitfactor_sqf(Poly(0, t), DE) == (((Poly(0, t), 1),), ())


def test_canonical_representation():
    # 创建微分扩展 DE，其扩展中包含了两个不同的多项式
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2, t)]})
    
    # 断言 canonical_representation 函数对 Poly(x - t, t) 和 Poly(t**2, t) 在 DE 上的标准表示结果
    assert canonical_representation(Poly(x - t, t), Poly(t**2, t), DE) == \
        (Poly(0, t, domain='ZZ[x]'), (Poly(0, t, domain='QQ[x]'),
        Poly(1, t, domain='ZZ')), (Poly(-t + x, t, domain='QQ[x]'),
        Poly(t**2, t)))
    
    # 创建另一个微分扩展 DE，其扩展中包含了两个不同的多项式
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    
    # 断言 canonical_representation 函数对 Poly(t**5 + t**3 + x**2*t + 1, t) 和 Poly((t**2 + 1)**3, t) 在 DE 上的标准表示结果
    assert canonical_representation(Poly(t**5 + t**3 + x**2*t + 1, t),
    Poly((t**2 + 1)**3, t), DE) == \
        (Poly(0, t, domain='ZZ[x]'), (Poly(t**5 + t**3 + x**2*t + 1, t, domain='QQ[x]'),
         Poly(t**6 + 3*t**4 + 3*t**2 + 1, t, domain='QQ')),
        (Poly(0, t, domain='QQ[x]'), Poly(1, t, domain='QQ')))


def test_hermite_reduce():
    # 创建微分扩展 DE，其扩展中包含了两个不同的多项式
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    
    # 断言 hermite_reduce 函数对 Poly(x - t, t) 和 Poly(t**2, t) 在 DE 上的 Hermite 归约结果
    assert hermite_reduce(Poly(x - t, t), Poly(t**2, t), DE) == \
        ((Poly(-x, t, domain='QQ[x]'), Poly(t, t, domain='QQ[x]')),
         (Poly(0, t, domain='QQ[x]'), Poly(1, t, domain='QQ[x]')),
         (Poly(-x, t, domain='QQ[x]'), Poly(1, t, domain='QQ[x]')))
    
    # 创建另一个微分扩展 DE，其扩展中包含了两个不同的多项式，包含了变量 nu
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t**2 - t/x - (1 - nu**2/x**2), t)]})
    
    # 断言 hermite_reduce 函数对复杂多项式在 DE 上的 Hermite 归约结果
    assert hermite_reduce(
            Poly(x**2*t**5 + x*t**4 - nu**2*t**3 - x*(x**2 + 1)*t**2 - (x**2 - nu**2)*t - x**5/4, t),
            Poly(x**2*t**4 + x**2*(x**2 + 2)*t**2 + x**2 + x**4 + x**6/4, t), DE) == \
        ((Poly(-x**2 - 4, t, domain='ZZ(x,nu)'), Poly(4*t**2 + 2*x**2 + 4, t, domain='ZZ(x,nu)')),
         (Poly((-2*nu**2 - x**4)*t - (2*x**3 + 2*x), t, domain='ZZ(x,nu)'),
          Poly(2*x**2*t**2 + x**4 + 2*x**2, t, domain='ZZ(x,nu)')),
         (Poly(x*t + 1, t, domain='ZZ(x,nu)'), Poly(x, t, domain='ZZ(x,nu)')))
    # 创建 DifferentialExtension 对象 DE，指定扩展为 {'D': [Poly(1, x), Poly(1/x, t)]}
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})

    # 创建多项式 a，使用 Poly 类在 t 上定义多项式 (-2 + 3*x)*t**3 + (-1 + x)*t**2 + (-4*x + 2*x**2)*t + x**2
    a = Poly((-2 + 3*x)*t**3 + (-1 + x)*t**2 + (-4*x + 2*x**2)*t + x**2, t)
    
    # 创建多项式 d，使用 Poly 类在 t 上定义多项式 x*t**6 - 4*x**2*t**5 + 6*x**3*t**4 - 4*x**4*t**3 + x**5*t**2
    d = Poly(x*t**6 - 4*x**2*t**5 + 6*x**3*t**4 - 4*x**4*t**3 + x**5*t**2, t)
    
    # 使用 hermite_reduce 函数验证 a 和 d 是否满足 DifferentialExtension DE 的 Hermite 归约条件
    assert hermite_reduce(a, d, DE) == \
        # 返回值是一个三元组，每个元素都是包含 Poly 对象的元组，表示 Hermite 归约后的结果
        ((Poly(3*t**2 + t + 3*x, t, domain='ZZ(x)'),
          Poly(3*t**4 - 9*x*t**3 + 9*x**2*t**2 - 3*x**3*t, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')))

    # 使用 hermite_reduce 函数验证指定的两个多项式是否满足 DifferentialExtension DE 的 Hermite 归约条件
    assert hermite_reduce(
            Poly(-t**2 + 2*t + 2, t, domain='ZZ(x)'),
            Poly(-x*t**2 + 2*x*t - x, t, domain='ZZ(x)'), DE) == \
        # 返回值是一个三元组，每个元素都是包含 Poly 对象的元组，表示 Hermite 归约后的结果
        ((Poly(3, t, domain='ZZ(x)'), Poly(t - 1, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')),
         (Poly(1, t, domain='ZZ(x)'), Poly(x, t, domain='ZZ(x)')))

    # 使用 hermite_reduce 函数验证指定的两个多项式是否满足 DifferentialExtension DE 的 Hermite 归约条件
    assert hermite_reduce(
            Poly(-x**2*t**6 + (-1 - 2*x**3 + x**4)*t**3 + (-3 - 3*x**4)*t**2 -
                2*x*t - x - 3*x**2, t, domain='ZZ(x)'),
            Poly(x**4*t**6 - 2*x**2*t**3 + 1, t, domain='ZZ(x)'), DE) == \
        # 返回值是一个三元组，每个元素都是包含 Poly 对象的元组，表示 Hermite 归约后的结果
        ((Poly(x**3*t + x**4 + 1, t, domain='ZZ(x)'), Poly(x**3*t**3 - x, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')),
         (Poly(-1, t, domain='ZZ(x)'), Poly(x**2, t, domain='ZZ(x)')))

    # 使用 hermite_reduce 函数验证 a 和 d 是否满足 DifferentialExtension DE 的 Hermite 归约条件
    assert hermite_reduce(
            Poly((-2 + 3*x)*t**3 + (-1 + x)*t**2 + (-4*x + 2*x**2)*t + x**2, t),
            Poly(x*t**6 - 4*x**2*t**5 + 6*x**3*t**4 - 4*x**4*t**3 + x**5*t**2, t), DE) == \
        # 返回值是一个三元组，每个元素都是包含 Poly 对象的元组，表示 Hermite 归约后的结果
        ((Poly(3*t**2 + t + 3*x, t, domain='ZZ(x)'),
          Poly(3*t**4 - 9*x*t**3 + 9*x**2*t**2 - 3*x**3*t, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')),
         (Poly(0, t, domain='ZZ(x)'), Poly(1, t, domain='ZZ(x)')))
def test_polynomial_reduce():
    # 创建一个差分扩展对象，使用给定的扩展字典
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2, t)]})
    # 断言对于给定的多项式和差分扩展，调用 polynomial_reduce 函数返回期望的元组结果
    assert polynomial_reduce(Poly(1 + x*t + t**2, t), DE) == \
        (Poly(t, t), Poly(x*t, t))
    # 断言对于零多项式和差分扩展，调用 polynomial_reduce 函数返回期望的零多项式元组结果
    assert polynomial_reduce(Poly(0, t), DE) == \
        (Poly(0, t), Poly(0, t))


def test_laurent_series():
    # 创建一个差分扩展对象，使用给定的扩展字典
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1, t)]})
    # 定义多项式 a 和 d
    a = Poly(36, t)
    d = Poly((t - 2)*(t**2 - 1)**2, t)
    F = Poly(t**2 - 1, t)
    n = 2
    # 断言 laurent_series 函数对于给定的参数和差分扩展返回期望的元组结果
    assert laurent_series(a, d, F, n, DE) == \
        (Poly(-3*t**3 + 3*t**2 - 6*t - 8, t), Poly(t**5 + t**4 - 2*t**3 - 2*t**2 + t + 1, t),
        [Poly(-3*t**3 - 6*t**2, t, domain='QQ'), Poly(2*t**6 + 6*t**5 - 8*t**3, t, domain='QQ')])


def test_recognize_derivative():
    # 创建一个差分扩展对象，使用给定的扩展字典
    DE = DifferentialExtension(extension={'D': [Poly(1, t)]})
    # 定义多项式 a 和 d，然后断言 recognize_derivative 函数的返回值为 False
    a = Poly(36, t)
    d = Poly((t - 2)*(t**2 - 1)**2, t)
    assert recognize_derivative(a, d, DE) == False
    # 更新差分扩展对象，定义新的多项式 a 和 d，然后断言 recognize_derivative 函数的返回值为 False
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    a = Poly(2, t)
    d = Poly(t**2 - 1, t)
    assert recognize_derivative(a, d, DE) == False
    # 断言 recognize_derivative 函数对于给定的参数返回期望的 True 和 False 结果
    assert recognize_derivative(Poly(x*t, t), Poly(1, t), DE) == True
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert recognize_derivative(Poly(t, t), Poly(1, t), DE) == True


def test_recognize_log_derivative():
    # 定义多项式 a 和 d，创建一个差分扩展对象，使用给定的扩展字典
    a = Poly(2*x**2 + 4*x*t - 2*t - x**2*t, t)
    d = Poly((2*x + t)*(t + x**2), t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    # 断言 recognize_log_derivative 函数对于给定的参数和差分扩展返回期望的 True 结果
    assert recognize_log_derivative(a, d, DE, z) == True
    # 更新差分扩展对象，使用给定的扩展字典，然后断言 recognize_log_derivative 函数对于给定的参数返回期望的 True 结果
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    assert recognize_log_derivative(Poly(t + 1, t), Poly(t + x, t), DE) == True
    assert recognize_log_derivative(Poly(2, t), Poly(t**2 - 1, t), DE) == True
    # 更新差分扩展对象，使用给定的扩展字典，然后断言 recognize_log_derivative 函数对于给定的参数返回期望的 True 和 False 结果
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    assert recognize_log_derivative(Poly(1, x), Poly(x**2 - 2, x), DE) == False
    assert recognize_log_derivative(Poly(1, x), Poly(x**2 + x, x), DE) == True
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    assert recognize_log_derivative(Poly(1, t), Poly(t**2 - 2, t), DE) == False
    assert recognize_log_derivative(Poly(1, t), Poly(t**2 + t, t), DE) == False


def test_residue_reduce():
    # 定义多项式 a 和 d，创建一个差分扩展对象，使用给定的扩展字典和函数
    a = Poly(2*t**2 - t - x**2, t)
    d = Poly(t**3 - x**2*t, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)], 'Tfuncs': [log]})
    # 断言 residue_reduce 函数对于给定的参数返回期望的元组结果
    assert residue_reduce(a, d, DE, z, invert=False) == \
        ([(Poly(z**2 - Rational(1, 4), z, domain='ZZ(x)'),
          Poly((1 + 3*x*z - 6*z**2 - 2*x**2 + 4*x**2*z**2)*t - x*z + x**2 +
              2*x**2*z**2 - 2*z*x**3, t, domain='ZZ(z, x)'))], False)
    assert residue_reduce(a, d, DE, z, invert=True) == \
        ([(Poly(z**2 - Rational(1, 4), z, domain='ZZ(x)'), Poly(t + 2*x*z, t))], False)
    # 断言：使用 residue_reduce 函数验证给定的多项式和微分方程的残差减少结果是否符合预期
    assert residue_reduce(Poly(-2/x, t), Poly(t**2 - 1, t,), DE, z, invert=False) == \
        ([(Poly(z**2 - 1, z, domain='QQ'), Poly(-2*z*t/x - 2/x, t, domain='ZZ(z,x)'))], True)

    # 使用 residue_reduce 函数计算残差减少结果，并将结果赋给 ans 变量
    ans = residue_reduce(Poly(-2/x, t), Poly(t**2 - 1, t), DE, z, invert=True)

    # 断言：验证 ans 的值是否符合预期
    assert ans == ([(Poly(z**2 - 1, z, domain='QQ'), Poly(t + z, t))], True)

    # 断言：验证 residue_reduce_to_basic 函数的计算结果是否符合预期
    assert residue_reduce_to_basic(ans[0], DE, z) == -log(-1 + log(x)) + log(1 + log(x))

    # 设置微分方程 DE 的新值，含有多项式的扩展
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t**2 - t/x - (1 - nu**2/x**2), t)]})

    # TODO: 跳过或者优化计算速度
    # 断言：验证 residue_reduce 函数的计算结果是否符合预期
    assert residue_reduce(Poly((-2*nu**2 - x**4)/(2*x**2)*t - (1 + x**2)/x, t),
        Poly(t**2 + 1 + x**2/2, t), DE, z) == \
        ([(Poly(z + S.Half, z, domain='QQ'), Poly(t**2 + 1 + x**2/2, t,
            domain='ZZ(x,nu)'))], True)

    # 设置微分方程 DE 的新值，含有多项式的扩展
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2, t)]})

    # 断言：验证 residue_reduce 函数的计算结果是否符合预期
    assert residue_reduce(Poly(-2*x*t + 1 - x**2, t),
        Poly(t**2 + 2*x*t + 1 + x**2, t), DE, z) == \
        ([(Poly(z**2 + Rational(1, 4), z), Poly(t + x + 2*z, t))], True)

    # 设置微分方程 DE 的新值，含有多项式的扩展
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})

    # 断言：验证 residue_reduce 函数的计算结果是否符合预期
    assert residue_reduce(Poly(t, t), Poly(t + sqrt(2), t), DE, z) == \
        ([(Poly(z - 1, z, domain='QQ'), Poly(t + sqrt(2), t))], True)
def test_integrate_hyperexponential():
    # TODO: Add tests for integrate_hyperexponential() from the book

    # 构造多项式 a 和微分 d
    a = Poly((1 + 2*t1 + t1**2 + 2*t1**3)*t**2 + (1 + t1**2)*t + 1 + t1**2, t)
    d = Poly(1, t)
    
    # 创建微分扩展 DE
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t1**2, t1),
        Poly(t*(1 + t1**2), t)], 'Tfuncs': [tan, Lambda(i, exp(tan(i)))]})
    
    # 断言 integrate_hyperexponential(a, d, DE) 的返回结果
    assert integrate_hyperexponential(a, d, DE) == \
        (exp(2*tan(x))*tan(x) + exp(tan(x)), 1 + t1**2, True)

    # 更改多项式 a 的定义
    a = Poly((t1**3 + (x + 1)*t1**2 + t1 + x + 2)*t, t)
    
    # 再次断言 integrate_hyperexponential(a, d, DE) 的返回结果
    assert integrate_hyperexponential(a, d, DE) == \
        ((x + tan(x))*exp(tan(x)), 0, True)

    # 重新定义 a 和 d
    a = Poly(t, t)
    d = Poly(1, t)
    
    # 更新微分扩展 DE
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(2*x*t, t)],
        'Tfuncs': [Lambda(i, exp(x**2))]})

    # 断言 integrate_hyperexponential(a, d, DE) 的返回结果
    assert integrate_hyperexponential(a, d, DE) == \
        (0, NonElementaryIntegral(exp(x**2), x), False)

    # 更新微分扩展 DE
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)], 'Tfuncs': [exp]})
    
    # 断言 integrate_hyperexponential(a, d, DE) 的返回结果
    assert integrate_hyperexponential(a, d, DE) == (exp(x), 0, True)

    # 更改多项式 a 和微分 d
    a = Poly(25*t**6 - 10*t**5 + 7*t**4 - 8*t**3 + 13*t**2 + 2*t - 1, t)
    d = Poly(25*t**6 + 35*t**4 + 11*t**2 + 1, t)
    
    # 断言 integrate_hyperexponential(a, d, DE) 的返回结果
    assert integrate_hyperexponential(a, d, DE) == \
        (-(11 - 10*exp(x))/(5 + 25*exp(2*x)) + log(1 + exp(2*x)), -1, True)
    
    # 更新微分扩展 DE
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t0, t0), Poly(t0*t, t)],
        'Tfuncs': [exp, Lambda(i, exp(exp(i)))]})
    
    # 断言 integrate_hyperexponential(Poly(2*t0*t**2, t), Poly(1, t), DE) 的返回结果
    assert integrate_hyperexponential(Poly(2*t0*t**2, t), Poly(1, t), DE) == (exp(2*exp(x)), 0, True)

    # 更新微分扩展 DE
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t0, t0), Poly(-t0*t, t)],
        'Tfuncs': [exp, Lambda(i, exp(-exp(i)))]})
    
    # 断言 integrate_hyperexponential 的返回结果
    assert integrate_hyperexponential(Poly(-27*exp(9) - 162*t0*exp(9) +
    27*x*t0*exp(9), t), Poly((36*exp(18) + x**2*exp(18) - 12*x*exp(18))*t, t), DE) == \
        (27*exp(exp(x))/(-6*exp(9) + x*exp(9)), 0, True)

    # 更新微分扩展 DE
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)], 'Tfuncs': [exp]})
    
    # 断言 integrate_hyperexponential(Poly(x**2/2*t, t), Poly(1, t), DE) 的返回结果
    assert integrate_hyperexponential(Poly(x**2/2*t, t), Poly(1, t), DE) == \
        ((2 - 2*x + x**2)*exp(x)/2, 0, True)
    
    # 断言 integrate_hyperexponential(Poly(1 + t, t), Poly(t, t), DE) 的返回结果
    assert integrate_hyperexponential(Poly(1 + t, t), Poly(t, t), DE) == \
        (-exp(-x), 1, True)  # x - exp(-x)
    
    # 断言 integrate_hyperexponential(Poly(x, t), Poly(t + 1, t), DE) 的返回结果
    assert integrate_hyperexponential(Poly(x, t), Poly(t + 1, t), DE) == \
        (0, NonElementaryIntegral(x/(1 + exp(x)), x), False)

    # 更新微分扩展 DE
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t0), Poly(2*x*t1, t1)],
        'Tfuncs': [log, Lambda(i, exp(i**2))]})
    # 解构赋值，将 integrate_hyperexponential 返回的元组中的三个元素分别赋值给 elem, nonelem, b
    elem, nonelem, b = integrate_hyperexponential(
        Poly((8*x**7 - 12*x**5 + 6*x**3 - x)*t1**4 +
            (8*t0*x**7 - 8*t0*x**6 - 4*t0*x**5 + 2*t0*x**3 + 2*t0*x**2 - t0*x +
            24*x**8 - 36*x**6 - 4*x**5 + 22*x**4 + 4*x**3 - 7*x**2 - x + 1)*t1**3
            + (8*t0*x**8 - 4*t0*x**6 - 16*t0*x**5 - 2*t0*x**4 + 12*t0*x**3 +
            t0*x**2 - 2*t0*x + 24*x**9 - 36*x**7 - 8*x**6 + 22*x**5 + 12*x**4 -
            7*x**3 - 6*x**2 + x + 1)*t1**2 + (8*t0*x**8 - 8*t0*x**6 - 16*t0*x**5 +
            6*t0*x**4 + 10*t0*x**3 - 2*t0*x**2 - t0*x + 8*x**10 - 12*x**8 - 4*x**7
            + 2*x**6 + 12*x**5 + 3*x**4 - 9*x**3 - x**2 + 2*x)*t1 + 8*t0*x**7 -
            12*t0*x**6 - 4*t0*x**5 + 8*t0*x**4 - t0*x**2 - 4*x**7 + 4*x**6 +
            4*x**5 - 4*x**4 - x**3 + x**2, t1),
        Poly((8*x**7 - 12*x**5 + 6*x**3 - x)*t1**4 + (24*x**8 + 8*x**7 - 36*x**6 - 12*x**5 + 18*x**4 + 6*x**3 -
            3*x**2 - x)*t1**3 + (24*x**9 + 24*x**8 - 36*x**7 - 36*x**6 + 18*x**5 +
            18*x**4 - 3*x**3 - 3*x**2)*t1**2 + (8*x**10 + 24*x**9 - 12*x**8 -
            36*x**7 + 6*x**6 + 18*x**5 - x**4 - 3*x**3)*t1 + 8*x**10 - 12*x**8 +
            6*x**6 - x**4, t1),
        DE)

    # 使用 assert 断言来验证 factor(elem) 的返回值是否等于 -((x - 1)*log(x)/((x + exp(x**2))*(2*x**2 - 1)))
    assert factor(elem) == -((x - 1)*log(x)/((x + exp(x**2))*(2*x**2 - 1)))
    
    # 使用 assert 断言来验证 (nonelem, b) 是否等于 (NonElementaryIntegral(exp(x**2)/(exp(x**2) + 1), x), False)
    assert (nonelem, b) == (NonElementaryIntegral(exp(x**2)/(exp(x**2) + 1), x), False)
# 定义测试函数，用于集成超指数多项式
def test_integrate_hyperexponential_polynomial():
    # 使用超指数多项式定义 p
    p = Poly((-28*x**11*t0 - 6*x**8*t0 + 6*x**9*t0 - 15*x**8*t0**2 +
        15*x**7*t0**2 + 84*x**10*t0**2 - 140*x**9*t0**3 - 20*x**6*t0**3 +
        20*x**7*t0**3 - 15*x**6*t0**4 + 15*x**5*t0**4 + 140*x**8*t0**4 -
        84*x**7*t0**5 - 6*x**4*t0**5 + 6*x**5*t0**5 + x**3*t0**6 - x**4*t0**6 +
        28*x**6*t0**6 - 4*x**5*t0**7 + x**9 - x**10 + 4*x**12)/(-8*x**11*t0 +
        28*x**10*t0**2 - 56*x**9*t0**3 + 70*x**8*t0**4 - 56*x**7*t0**5 +
        28*x**6*t0**6 - 8*x**5*t0**7 + x**4*t0**8 + x**12)*t1**2 +
        (-28*x**11*t0 - 12*x**8*t0 + 12*x**9*t0 - 30*x**8*t0**2 +
        30*x**7*t0**2 + 84*x**10*t0**2 - 140*x**9*t0**3 - 40*x**6*t0**3 +
        40*x**7*t0**3 - 30*x**6*t0**4 + 30*x**5*t0**4 + 140*x**8*t0**4 -
        84*x**7*t0**5 - 12*x**4*t0**5 + 12*x**5*t0**5 - 2*x**4*t0**6 +
        2*x**3*t0**6 + 28*x**6*t0**6 - 4*x**5*t0**7 + 2*x**9 - 2*x**10 +
        4*x**12)/(-8*x**11*t0 + 28*x**10*t0**2 - 56*x**9*t0**3 +
        70*x**8*t0**4 - 56*x**7*t0**5 + 28*x**6*t0**6 - 8*x**5*t0**7 +
        x**4*t0**8 + x**12)*t1 + (-2*x**2*t0 + 2*x**3*t0 + x*t0**2 -
        x**2*t0**2 + x**3 - x**4)/(-4*x**5*t0 + 6*x**4*t0**2 - 4*x**3*t0**3 +
        x**2*t0**4 + x**6), t1, z, expand=False)
    
    # 定义微分扩展 DE
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t0), Poly(2*x*t1, t1)]})
    
    # 断言集成超指数多项式函数的返回值
    assert integrate_hyperexponential_polynomial(p, DE, z) == (
        Poly((x - t0)*t1**2 + (-2*t0 + 2*x)*t1, t1), Poly(-2*x*t0 + x**2 +
        t0**2, t1), True)

    # 重新定义微分扩展 DE
    DE = DifferentialExtension(extension={'D':[Poly(1, x), Poly(t0, t0)]})
    
    # 断言集成超指数多项式函数的返回值
    assert integrate_hyperexponential_polynomial(Poly(0, t0), DE, z) == (
        Poly(0, t0), Poly(1, t0), True)


# 定义测试函数，用于测试集成超指数函数返回分段函数的情况
def test_integrate_hyperexponential_returns_piecewise():
    # 定义符号变量 a, b
    a, b = symbols('a b')
    
    # 定义微分扩展 DE，使用指数函数 a**x
    DE = DifferentialExtension(a**x, x)
    
    # 断言集成超指数函数的返回值
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        (exp(x*log(a))/log(a), Ne(log(a), 0)), (x, True)), 0, True)
    
    # 定义微分扩展 DE，使用指数函数 a**(b*x)
    DE = DifferentialExtension(a**(b*x), x)
    
    # 断言集成超指数函数的返回值
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        (exp(b*x*log(a))/(b*log(a)), Ne(b*log(a), 0)), (x, True)), 0, True)
    
    # 定义微分扩展 DE，使用指数函数 exp(a*x)
    DE = DifferentialExtension(exp(a*x), x)
    
    # 断言集成超指数函数的返回值
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        (exp(a*x)/a, Ne(a, 0)), (x, True)), 0, True)
    
    # 定义微分扩展 DE，使用指数函数 x*exp(a*x)
    DE = DifferentialExtension(x*exp(a*x), x)
    
    # 断言集成超指数函数的返回值
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        ((a*x - 1)*exp(a*x)/a**2, Ne(a**2, 0)), (x**2/2, True)), 0, True)
    
    # 定义微分扩展 DE，使用指数函数 x**2*exp(a*x)
    DE = DifferentialExtension(x**2*exp(a*x), x)
    
    # 断言集成超指数函数的返回值
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        ((x**2*a**2 - 2*a*x + 2)*exp(a*x)/a**3, Ne(a**3, 0)),
        (x**3/3, True)), 0, True)
    
    # 定义微分扩展 DE，使用多项式 x**y + z
    DE = DifferentialExtension(x**y + z, y)
    # 使用超指数函数积分来计算，断言结果与给定的 Piecewise 对象相等
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        # 当 log(x) 不等于 0 时，返回 exp(log(x)*y)/log(x)
        (exp(log(x)*y)/log(x), Ne(log(x), 0)),
        # 否则返回 y
        (y, True)), z, True)
    
    # 创建 DifferentialExtension 对象 DE，包含表达式 x**y + z + x**(2*y)
    DE = DifferentialExtension(x**y + z + x**(2*y), y)
    
    # 使用超指数函数积分来计算，断言结果与给定的 Piecewise 对象相等
    assert integrate_hyperexponential(DE.fa, DE.fd, DE) == (Piecewise(
        # 当 2*log(x)**2 不等于 0 时，返回 (exp(2*log(x)*y)*log(x) + 2*exp(log(x)*y)*log(x))/(2*log(x)**2)
        ((exp(2*log(x)*y)*log(x) + 2*exp(log(x)*y)*log(x))/(2*log(x)**2), Ne(2*log(x)**2, 0)),
        # 否则返回 2*y
        (2*y, True),
    ), z, True)
    
    # TODO: Add a test where two different parts of the extension use a
    # Piecewise, like y**x + z**x.
# 定义一个测试函数，用于测试问题编号为13947的特定功能
def test_issue_13947():
    # 定义符号变量a, t, s
    a, t, s = symbols('a t s')
    # 断言：对于给定表达式，使用risch_integrate函数计算结果并进行断言
    assert risch_integrate(2**(-pi)/(2**t + 1), t) == \
        2**(-pi)*t - 2**(-pi)*log(2**t + 1)/log(2)
    # 断言：对于给定表达式，使用risch_integrate函数计算结果并进行断言
    assert risch_integrate(a**(t - s)/(a**t + 1), t) == \
        exp(-s*log(a))*log(a**t + 1)/log(a)


# 定义一个测试函数，用于测试积分原始函数的不同情况
def test_integrate_primitive():
    # 定义DifferentialExtension对象DE，指定其扩展
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)],
        'Tfuncs': [log]})
    # 断言：调用integrate_primitive函数计算结果，并进行断言
    assert integrate_primitive(Poly(t, t), Poly(1, t), DE) == (x*log(x), -1, True)
    # 断言：调用integrate_primitive函数计算结果，并进行断言
    assert integrate_primitive(Poly(x, t), Poly(t, t), DE) == (0, NonElementaryIntegral(x/log(x), x), False)

    # 定义DifferentialExtension对象DE，指定其扩展
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t1), Poly(1/(x + 1), t2)],
        'Tfuncs': [log, Lambda(i, log(i + 1))]})
    # 断言：调用integrate_primitive函数计算结果，并进行断言
    assert integrate_primitive(Poly(t1, t2), Poly(t2, t2), DE) == \
        (0, NonElementaryIntegral(log(x)/log(1 + x), x), False)

    # 定义DifferentialExtension对象DE，指定其扩展
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t1), Poly(1/(x*t1), t2)],
        'Tfuncs': [log, Lambda(i, log(log(i)))]})
    # 断言：调用integrate_primitive函数计算结果，并进行断言
    assert integrate_primitive(Poly(t2, t2), Poly(t1, t2), DE) == \
        (0, NonElementaryIntegral(log(log(x))/log(x), x), False)

    # 定义DifferentialExtension对象DE，指定其扩展
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t0)],
        'Tfuncs': [log]})
    # 断言：调用integrate_primitive函数计算结果，并进行断言
    assert integrate_primitive(Poly(x**2*t0**3 + (3*x**2 + x)*t0**2 + (3*x**2
    + 2*x)*t0 + x**2 + x, t0), Poly(x**2*t0**4 + 4*x**2*t0**3 + 6*x**2*t0**2 +
    4*x**2*t0 + x**2, t0), DE) == \
        (-1/(log(x) + 1), NonElementaryIntegral(1/(log(x) + 1), x), False)


# 定义一个测试函数，用于测试超越正切多项式的积分
def test_integrate_hypertangent_polynomial():
    # 定义DifferentialExtension对象DE，指定其扩展
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**2 + 1, t)]})
    # 断言：调用integrate_hypertangent_polynomial函数计算结果，并进行断言
    assert integrate_hypertangent_polynomial(Poly(t**2 + x*t + 1, t), DE) == \
        (Poly(t, t), Poly(x/2, t))
    # 定义DifferentialExtension对象DE，指定其扩展
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(a*(t**2 + 1), t)]})
    # 断言：调用integrate_hypertangent_polynomial函数计算结果，并进行断言
    assert integrate_hypertangent_polynomial(Poly(t**5, t), DE) == \
        (Poly(1/(4*a)*t**4 - 1/(2*a)*t**2, t), Poly(1/(2*a), t))


# 定义一个测试函数，用于测试非线性无特殊项的积分
def test_integrate_nonlinear_no_specials():
    # 定义Poly对象a和d
    a, d, = Poly(x**2*t**5 + x*t**4 - nu**2*t**3 - x*(x**2 + 1)*t**2 - (x**2 -
    nu**2)*t - x**5/4, t), Poly(x**2*t**4 + x**2*(x**2 + 2)*t**2 + x**2 + x**4 + x**6/4, t)
    # 定义函数f为phi_nu，其为J_v贝塞尔函数的对数导数，不包含特殊项
    f = Function('phi_nu')
    # 定义DifferentialExtension对象DE，指定其扩展
    DE = DifferentialExtension(extension={'D': [Poly(1, x),
        Poly(-t**2 - t/x - (1 - nu**2/x**2), t)], 'Tfuncs': [f]})
    # 断言：调用integrate_nonlinear_no_specials函数计算结果，并进行断言
    assert integrate_nonlinear_no_specials(a, d, DE) == \
        (-log(1 + f(x)**2 + x**2/2)/2 + (- 4 - x**2)/(4 + 2*x**2 + 4*f(x)**2), True)
    # 断言：调用integrate_nonlinear_no_specials函数计算结果，并进行断言
    assert integrate_nonlinear_no_specials(Poly(t, t), Poly(1, t), DE) == \
        (0, False)


# 定义一个测试函数，用于测试整数幂
def test_integer_powers():
    # 断言：调用integer_powers函数计算结果，并进行断言
    assert integer_powers([x, x/2, x**2 + 1, x*Rational(2, 3)]) == [
            (x/6, [(x, 6), (x/2, 3), (x*Rational(2, 3), 4)]),
            (1 + x**2, [(1 + x**2, 1)])]


# 定义一个测试函数，但其定义部分被省略
def test_DifferentialExtension_exp():
    # 断言语句，验证 DifferentialExtension 类的重要属性是否符合预期值
    
    assert DifferentialExtension(exp(x) + exp(x**2), x)._important_attrs == \
        (Poly(t1 + t0, t1),                    # 第一个重要属性：表示为多项式 t1 + t0 的 Poly 对象
         Poly(1, t1),                          # 第二个重要属性：表示为常数多项式 1 的 Poly 对象
         [Poly(1, x,), Poly(t0, t0),           # 第三个重要属性：列表包含多个 Poly 对象，依次为 Poly(1, x), Poly(t0, t0)
          Poly(2*x*t1, t1)],                   # 继续的 Poly 对象为 Poly(2*x*t1, t1)
         [x, t0, t1],                          # 第四个重要属性：表示变量的列表，依次为 x, t0, t1
         [Lambda(i, exp(i)),                   # 第五个重要属性：表示为 Lambda 函数对象，Lambda(i, exp(i))
          Lambda(i, exp(i**2))],               # 继续的 Lambda 函数对象为 Lambda(i, exp(i**2))
         [],                                   # 第六个重要属性：空列表
         [None, 'exp', 'exp'],                 # 第七个重要属性：包含 None 和字符串 'exp' 的列表
         [None, x, x**2])                       # 最后一个重要属性：包含 None、x 和 x**2 的列表
    
    # 后续断言语句与上述注释相似，验证不同输入条件下 DifferentialExtension 类的重要属性
    assert DifferentialExtension(exp(x) + exp(2*x), x)._important_attrs == \
        (Poly(t0**2 + t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0, t0)], [x, t0],
         [Lambda(i, exp(i))], [], [None, 'exp'], [None, x])
    
    assert DifferentialExtension(exp(x) + exp(x/2), x)._important_attrs == \
        (Poly(t0**2 + t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0/2, t0)], [x, t0],
         [Lambda(i, exp(i/2))], [], [None, 'exp'], [None, x/2])
    
    assert DifferentialExtension(exp(x) + exp(x**2) + exp(x + x**2), x)._important_attrs == \
        (Poly((1 + t0)*t1 + t0, t1), Poly(1, t1), [Poly(1, x), Poly(t0, t0),
         Poly(2*x*t1, t1)], [x, t0, t1], [Lambda(i, exp(i)), Lambda(i, exp(i**2))],
         [], [None, 'exp', 'exp'], [None, x, x**2])
    
    assert DifferentialExtension(exp(x) + exp(x**2) + exp(x + x**2 + 1), x)._important_attrs == \
        (Poly((1 + S.Exp1*t0)*t1 + t0, t1), Poly(1, t1), [Poly(1, x), Poly(t0, t0),
         Poly(2*x*t1, t1)], [x, t0, t1], [Lambda(i, exp(i)), Lambda(i, exp(i**2))],
         [], [None, 'exp', 'exp'], [None, x, x**2])
    
    assert DifferentialExtension(exp(x) + exp(x**2) + exp(x/2 + x**2), x)._important_attrs == \
        (Poly((t0 + 1)*t1 + t0**2, t1), Poly(1, t1), [Poly(1, x), Poly(t0/2, t0),
         Poly(2*x*t1, t1)], [x, t0, t1], [Lambda(i, exp(i/2)), Lambda(i, exp(i**2))],
         [(exp(x/2), sqrt(exp(x)))], [None, 'exp', 'exp'], [None, x/2, x**2])
    
    assert DifferentialExtension(exp(x) + exp(x**2) + exp(x/2 + x**2 + 3), x)._important_attrs == \
        (Poly((t0*exp(3) + 1)*t1 + t0**2, t1), Poly(1, t1), [Poly(1, x), Poly(t0/2, t0),
         Poly(2*x*t1, t1)], [x, t0, t1], [Lambda(i, exp(i/2)), Lambda(i, exp(i**2))],
         [(exp(x/2), sqrt(exp(x)))], [None, 'exp', 'exp'], [None, x/2, x**2])
    
    assert DifferentialExtension(sqrt(exp(x)), x)._important_attrs == \
        (Poly(t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0/2, t0)], [x, t0],
         [Lambda(i, exp(i/2))], [(exp(x/2), sqrt(exp(x)))], [None, 'exp'], [None, x/2])
    
    assert DifferentialExtension(exp(x/2), x)._important_attrs == \
        (Poly(t0, t0), Poly(1, t0), [Poly(1, x), Poly(t0/2, t0)], [x, t0],
         [Lambda(i, exp(i/2))], [], [None, 'exp'], [None, x/2])
def test_DifferentialExtension_log():
    # 测试 DifferentialExtension 类的 log 方法

    # 断言表达式，验证 DifferentialExtension 对象的 _important_attrs 属性
    assert DifferentialExtension(log(x)*log(x + 1)*log(2*x**2 + 2*x), x)._important_attrs == \
        # 返回元组，包含多个元素，每个元素对应一个属性
        (Poly(t0*t1**2 + (t0*log(2) + t0**2)*t1, t1),  # 多项式对象 t1
        Poly(1, t1),  # 多项式对象 t1
        [Poly(1, x), Poly(1/x, t0),  # 多项式对象 x 和 t0
        Poly(1/(x + 1), t1, expand=False)],  # 多项式对象 t1，设置为不展开
        [x, t0, t1],  # 符号变量列表
        [Lambda(i, log(i)), Lambda(i, log(i + 1))],  # Lambda 函数列表
        [],  # 空列表
        [None, 'log', 'log'],  # 字符串列表
        [None, x, x + 1])  # 其他对象列表

    assert DifferentialExtension(x**x*log(x), x)._important_attrs == \
        # 返回元组，包含多个元素，每个元素对应一个属性
        (Poly(t0*t1, t1),  # 多项式对象 t1
        Poly(1, t1),  # 多项式对象 t1
        [Poly(1, x), Poly(1/x, t0),  # 多项式对象 x 和 t0
        Poly((1 + t0)*t1, t1)],  # 多项式对象 t1
        [x, t0, t1],  # 符号变量列表
        [Lambda(i, log(i)), Lambda(i, exp(t0*i))],  # Lambda 函数列表
        [(exp(x*log(x)), x**x)],  # 元组列表
        [None, 'log', 'exp'],  # 字符串列表
        [None, x, t0*x])  # 其他对象列表


def test_DifferentialExtension_symlog():
    # 测试 DifferentialExtension 类的 symlog 方法

    # 断言表达式，验证 DifferentialExtension 对象的 _important_attrs 属性
    assert DifferentialExtension(log(x**x), x)._important_attrs == \
        # 返回元组，包含多个元素，每个元素对应一个属性
        (Poly(t0*x, t1),  # 多项式对象 t1
        Poly(1, t1),  # 多项式对象 t1
        [Poly(1, x), Poly(1/x, t0), Poly((t0 + 1)*t1, t1)],  # 多项式对象 x、t0 和 t1
        [x, t0, t1],  # 符号变量列表
        [Lambda(i, log(i)), Lambda(i, exp(i*t0))],  # Lambda 函数列表
        [(exp(x*log(x)), x**x)],  # 元组列表
        [None, 'log', 'exp'],  # 字符串列表
        [None, x, t0*x])  # 其他对象列表

    assert DifferentialExtension(log(x**y), x)._important_attrs == \
        # 返回元组，包含多个元素，每个元素对应一个属性
        (Poly(y*t0, t0),  # 多项式对象 t0
        Poly(1, t0),  # 多项式对象 t0
        [Poly(1, x), Poly(1/x, t0)],  # 多项式对象 x 和 t0
        [x, t0],  # 符号变量列表
        [Lambda(i, log(i))],  # Lambda 函数列表
        [(y*log(x), log(x**y))],  # 元组列表
        [None, 'log'],  # 字符串列表
        [None, x])  # 其他对象列表

    assert DifferentialExtension(log(sqrt(x)), x)._important_attrs == \
        # 返回元组，包含多个元素，每个元素对应一个属性
        (Poly(t0, t0),  # 多项式对象 t0
        Poly(2, t0),  # 多项式对象 t0
        [Poly(1, x), Poly(1/x, t0)],  # 多项式对象 x 和 t0
        [x, t0],  # 符号变量列表
        [Lambda(i, log(i))],  # Lambda 函数列表
        [(log(x)/2, log(sqrt(x)))],  # 元组列表
        [None, 'log'],  # 字符串列表
        [None, x])  # 其他对象列表


def test_DifferentialExtension_handle_first():
    # 测试 DifferentialExtension 类的 handle_first 参数

    assert DifferentialExtension(exp(x)*log(x), x, handle_first='log')._important_attrs == \
        # 返回元组，包含多个元素，每个元素对应一个属性
        (Poly(t0*t1, t1),  # 多项式对象 t1
        Poly(1, t1),  # 多项式对象 t1
        [Poly(1, x), Poly(1/x, t0), Poly(t1, t1)],  # 多项式对象 x、t0 和 t1
        [x, t0, t1],  # 符号变量列表
        [Lambda(i, log(i)), Lambda(i, exp(i))],  # Lambda 函数列表
        [],  # 空列表
        [None, 'log', 'exp'],  # 字符串列表
        [None, x, x])  # 其他对象列表

    assert DifferentialExtension(exp(x)*log(x), x, handle_first='exp')._important_attrs == \
        # 返回元组，包含多个元素，每个元素对应一个属性
        (Poly(t0*t1, t1),  # 多项式对象 t1
        Poly(1, t1),  # 多项式对象 t1
        [Poly(1, x), Poly(t0, t0), Poly(1/x, t1)],  # 多项式对象 x、t0 和 t1
        [x, t0, t1],  # 符号变量列表
        [Lambda(i, exp(i)), Lambda(i, log(i))],  # Lambda 函数列表
        [],  # 空列表
        [None, 'exp', 'log'],  # 字符串列表
        [None, x, x])  # 其他对象列表

    # This one must have the log first, regardless of what we set it to
    # (because the log is inside of the exponential: x**x == exp(x*log(x)))
    assert DifferentialExtension(-x**x*log(x)**2 + x**x - x**x/x, x,
        handle_first='exp')._important_attrs == \
        DifferentialExtension(-x**x*log(x)**2 + x**x - x**x/x, x,
        handle_first='log')._important_attrs == \
        # 返回元组，包含多个元素，每个元素对应一个属性
        (Poly((-1 + x - x*t0**2)*t1, t1),  # 多项式对象 t1
        Poly(x, t1),  # 多项式对象 t1
        [Poly(1, x), Poly(1/x, t0), Poly((1 + t0)*t1, t1)],  # 多项式对象 x、t0 和 t1
        [x, t0, t1],  # 符号变量列表
        [Lambda(i, log(i)), Lambda(i, exp(t0*i))],  # Lambda 函数列表
        [(exp(x*log(x)), x**x)],  # 元组列表
        [None, 'log', 'exp'],  # 字符串列表
        [None, x, t0*x])  # 其他对象列表


def test_DifferentialExtension_all_attrs():
    # 测试 DifferentialExtension 类的所有属性

    # Test 'unimportant' attributes
    DE = DifferentialExtension(exp(x)*log(x), x, handle_first='exp')
    # 断言DE对象的f属性等于exp(x)*log(x)
    assert DE.f == exp(x)*log(x)
    # 断言DE对象的newf属性等于t0*t1
    assert DE.newf == t0*t1
    # 断言DE对象的x属性等于x
    assert DE.x == x
    # 断言DE对象的cases属性等于['base', 'exp', 'primitive']
    assert DE.cases == ['base', 'exp', 'primitive']
    # 断言DE对象的case属性等于'primitive'
    assert DE.case == 'primitive'

    # 断言DE对象的level属性等于-1
    assert DE.level == -1
    # 断言DE对象的t属性和t1属性等于DE对象的T属性中level为-1的元素
    assert DE.t == t1 == DE.T[DE.level]
    # 断言DE对象的d属性等于Poly(1/x, t1)，Poly是多项式对象
    assert DE.d == Poly(1/x, t1) == DE.D[DE.level]
    # 使用lambda函数检查调用DE对象的increment_level()方法是否引发ValueError异常
    raises(ValueError, lambda: DE.increment_level())
    # 调用DE对象的decrement_level()方法
    DE.decrement_level()
    # 断言DE对象的level属性等于-2
    assert DE.level == -2
    # 断言DE对象的t属性和t0属性等于DE对象的T属性中level为-2的元素
    assert DE.t == t0 == DE.T[DE.level]
    # 断言DE对象的d属性等于Poly(t0, t0)，Poly是多项式对象
    assert DE.d == Poly(t0, t0) == DE.D[DE.level]
    # 断言DE对象的case属性等于'exp'
    assert DE.case == 'exp'
    # 调用DE对象的decrement_level()方法
    DE.decrement_level()
    # 断言DE对象的level属性等于-3
    assert DE.level == -3
    # 断言DE对象的t属性和x属性等于DE对象的T属性中level为-3的元素，即x
    assert DE.t == x == DE.T[DE.level] == DE.x
    # 断言DE对象的d属性等于Poly(1, x)，Poly是多项式对象
    assert DE.d == Poly(1, x) == DE.D[DE.level]
    # 断言DE对象的case属性等于'base'
    assert DE.case == 'base'
    # 使用lambda函数检查调用DE对象的decrement_level()方法是否引发ValueError异常
    raises(ValueError, lambda: DE.decrement_level())
    # 调用DE对象的increment_level()方法两次
    DE.increment_level()
    DE.increment_level()
    # 断言DE对象的level属性等于-1
    assert DE.level == -1
    # 断言DE对象的t属性和t1属性等于DE对象的T属性中level为-1的元素
    assert DE.t == t1 == DE.T[DE.level]
    # 断言DE对象的d属性等于Poly(1/x, t1)，Poly是多项式对象
    assert DE.d == Poly(1/x, t1) == DE.D[DE.level]
    # 断言DE对象的case属性等于'primitive'

    # 测试方法
    # 断言DE对象调用indices('log')方法返回[2]
    assert DE.indices('log') == [2]
    # 断言DE对象调用indices('exp')方法返回[1]
    assert DE.indices('exp') == [1]
def test_DifferentialExtension_extension_flag():
    # 测试 DifferentialExtension 的扩展标志错误引发 ValueError
    raises(ValueError, lambda: DifferentialExtension(extension={'T': [x, t]}))
    
    # 创建 DifferentialExtension 对象 DE，使用给定的扩展信息
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    
    # 断言 DE 的重要属性 _important_attrs 符合预期值
    assert DE._important_attrs == (None, None, [Poly(1, x), Poly(t, t)], [x, t],
        None, None, None, None)
    
    # 断言 DE 的微分属性 d 符合预期值
    assert DE.d == Poly(t, t)
    
    # 断言 DE 的参数 t 符合预期值
    assert DE.t == t
    
    # 断言 DE 的层级 level 符合预期值
    assert DE.level == -1
    
    # 断言 DE 的情况 cases 符合预期值
    assert DE.cases == ['base', 'exp']
    
    # 断言 DE 的变量 x 符合预期值
    assert DE.x == x
    
    # 断言 DE 的情况 case 符合预期值
    assert DE.case == 'exp'
    
    # 创建带有更多扩展信息的 DifferentialExtension 对象 DE
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)],
        'exts': [None, 'exp'], 'extargs': [None, x]})
    
    # 断言 DE 的重要属性 _important_attrs 符合预期值
    assert DE._important_attrs == (None, None, [Poly(1, x), Poly(t, t)], [x, t],
        None, None, [None, 'exp'], [None, x])
    
    # 测试 DifferentialExtension 调用时未提供扩展信息引发 ValueError
    raises(ValueError, lambda: DifferentialExtension())


def test_DifferentialExtension_misc():
    # 测试 DifferentialExtension 处理奇怪情况
    assert DifferentialExtension(sin(y)*exp(x), x)._important_attrs == \
        (Poly(sin(y)*t0, t0, domain='ZZ[sin(y)]'), Poly(1, t0, domain='ZZ'),
        [Poly(1, x, domain='ZZ'), Poly(t0, t0, domain='ZZ')], [x, t0],
        [Lambda(i, exp(i))], [], [None, 'exp'], [None, x])
    
    # 测试 DifferentialExtension 处理未实现的函数引发 NotImplementedError
    raises(NotImplementedError, lambda: DifferentialExtension(sin(x), x))
    
    # 测试 DifferentialExtension 处理指数函数
    assert DifferentialExtension(10**x, x)._important_attrs == \
        (Poly(t0, t0), Poly(1, t0), [Poly(1, x), Poly(log(10)*t0, t0)], [x, t0],
        [Lambda(i, exp(i*log(10)))], [(exp(x*log(10)), 10**x)], [None, 'exp'],
        [None, x*log(10)])
    
    # 测试 DifferentialExtension 处理对数和幂函数的情况
    assert DifferentialExtension(log(x) + log(x**2), x)._important_attrs in [
        (Poly(3*t0, t0), Poly(2, t0), [Poly(1, x), Poly(2/x, t0)], [x, t0],
        [Lambda(i, log(i**2))], [], [None, ], [], [1], [x**2]),
        (Poly(3*t0, t0), Poly(1, t0), [Poly(1, x), Poly(1/x, t0)], [x, t0],
        [Lambda(i, log(i))], [], [None, 'log'], [None, x])]
    
    # 测试 DifferentialExtension 处理零函数的情况
    assert DifferentialExtension(S.Zero, x)._important_attrs == \
        (Poly(0, x), Poly(1, x), [Poly(1, x)], [x], [], [], [None], [None])
    
    # 测试 DifferentialExtension 处理 tan 和 atan 函数的情况
    assert DifferentialExtension(tan(atan(x).rewrite(log)), x)._important_attrs == \
        (Poly(x, x), Poly(1, x), [Poly(1, x)], [x], [], [], [None], [None])


def test_DifferentialExtension_Rothstein():
    # 测试 DifferentialExtension 处理 Rothstein 积分
    f = (2581284541*exp(x) + 1757211400)/(39916800*exp(3*x) +
    119750400*exp(x)**2 + 119750400*exp(x) + 39916800)*exp(1/(exp(x) + 1) - 10*x)
    assert DifferentialExtension(f, x)._important_attrs == \
        (Poly((1757211400 + 2581284541*t0)*t1, t1), Poly(39916800 +
        119750400*t0 + 119750400*t0**2 + 39916800*t0**3, t1),
        [Poly(1, x), Poly(t0, t0), Poly(-(10 + 21*t0 + 10*t0**2)/(1 + 2*t0 +
        t0**2)*t1, t1, domain='ZZ(t0)')], [x, t0, t1],
        [Lambda(i, exp(i)), Lambda(i, exp(1/(t0 + 1) - 10*i))], [],
        [None, 'exp', 'exp'], [None, x, 1/(t0 + 1) - 10*x])


class _TestingException(Exception):
    """用于测试的虚拟异常类。"""
    pass


def test_DecrementLevel():
    # 测试 DecrementLevel 函数
    DE = DifferentialExtension(x*log(exp(x) + 1), x)
    
    # 断言 DE 的层级 level 符合预期值
    assert DE.level == -1
    
    # 断言 DE 的参数 t 符合预期值
    assert DE.t == t1
    # 断言DE对象的属性d等于Poly对象，使用t0/(t0 + 1)初始化
    assert DE.d == Poly(t0/(t0 + 1), t1)
    # 断言DE对象的属性case为'primitive'

    assert DE.case == 'primitive'

    # 使用DecrementLevel类对DE进行上下文管理
    with DecrementLevel(DE):
        # 断言DE对象的level属性为-2
        assert DE.level == -2
        # 断言DE对象的t属性为t0
        assert DE.t == t0
        # 断言DE对象的d属性为Poly对象，使用t0初始化
        assert DE.d == Poly(t0, t0)
        # 断言DE对象的case属性为'exp'

        with DecrementLevel(DE):
            # 在更深层级的上下文管理中，断言DE对象的level属性为-3
            assert DE.level == -3
            # 断言DE对象的t属性为x
            assert DE.t == x
            # 断言DE对象的d属性为Poly对象，使用1初始化
            assert DE.d == Poly(1, x)
            # 断言DE对象的case属性为'base'

        # 退出内部的上下文管理后，再次断言DE对象的level属性为-2
        assert DE.level == -2
        # 再次断言DE对象的t属性为t0
        assert DE.t == t0
        # 再次断言DE对象的d属性为Poly对象，使用t0初始化
        assert DE.d == Poly(t0, t0)
        # 再次断言DE对象的case属性为'exp'

    # 最外层上下文管理结束后，断言DE对象的level属性为-1
    assert DE.level == -1
    # 断言DE对象的t属性为t1
    assert DE.t == t1
    # 断言DE对象的d属性为Poly对象，使用t0/(t0 + 1)初始化
    assert DE.d == Poly(t0/(t0 + 1), t1)
    # 断言DE对象的case属性为'primitive'

    # 测试异常处理是否正确，验证__exit__方法在异常后被正确调用
    try:
        with DecrementLevel(DE):
            # 抛出_TestingException异常
            raise _TestingException
    except _TestingException:
        # 捕获_TestingException异常
        pass
    else:
        # 如果没有抛出异常，则抛出AssertionError异常
        raise AssertionError("Did not raise.")

    # 最终断言DE对象的level属性为-1
    assert DE.level == -1
    # 断言DE对象的t属性为t1
    assert DE.t == t1
    # 断言DE对象的d属性为Poly对象，使用t0/(t0 + 1)初始化
    assert DE.d == Poly(t0/(t0 + 1), t1)
    # 断言DE对象的case属性为'primitive'
# 定义测试函数 test_risch_integrate，用于测试 risch_integrate 函数的不同输入条件下的输出结果
def test_risch_integrate():
    # 断言：对 t0 * exp(x) 进行不定积分应该得到 t0 * exp(x)
    assert risch_integrate(t0*exp(x), x) == t0*exp(x)
    
    # 断言：对 sin(x) 进行复数域重写后的不定积分应该得到 -exp(I*x)/2 - exp(-I*x)/2
    assert risch_integrate(sin(x), x, rewrite_complex=True) == -exp(I*x)/2 - exp(-I*x)/2

    # 断言：对复杂表达式 (1 + 2*x**2 + x**4 + 2*x**3*exp(2*x**2)) / (x**4*exp(x**2) + 2*x**2*exp(x**2) + exp(x**2)) 进行不定积分应该得到 NonElementaryIntegral(exp(-x**2), x) + exp(x**2)/(1 + x**2)
    assert risch_integrate((1 + 2*x**2 + x**4 + 2*x**3*exp(2*x**2)) /
                           (x**4*exp(x**2) + 2*x**2*exp(x**2) + exp(x**2)), x) == \
        NonElementaryIntegral(exp(-x**2), x) + exp(x**2)/(1 + x**2)

    # 断言：对常数 0 进行不定积分应该得到 0
    assert risch_integrate(0, x) == 0

    # e1 表达式测试，也测试 prde_cancel() 函数
    e1 = log(x/exp(x) + 1)
    ans1 = risch_integrate(e1, x)
    # 断言：ans1 的导数减去 e1 应该得到 0
    assert cancel(diff(ans1, x) - e1) == 0

    # e2 表达式测试，也测试 issue #10798
    e2 = (log(-1/y)/2 - log(1/y)/2)/y - (log(1 - 1/y)/2 - log(1 + 1/y)/2)/y
    ans2 = risch_integrate(e2, y)
    # 断言：ans2 的导数减去 e2 应该得到 0
    assert expand_log(cancel(diff(ans2, y) - e2), force=True) == 0

    # 断言：对 log(x**x) 进行不定积分应该得到 x**2*log(x)/2 - x**2/4
    assert risch_integrate(log(x**x), x) == x**2*log(x)/2 - x**2/4

    # 断言：对 log(x**y) 进行不定积分应该得到 x*log(x**y) - x*y
    assert risch_integrate(log(x**y), x) == x*log(x**y) - x*y

    # 断言：对 log(sqrt(x)) 进行不定积分应该得到 x*log(sqrt(x)) - x/2
    assert risch_integrate(log(sqrt(x)), x) == x*log(sqrt(x)) - x/2


# 定义测试函数 test_risch_integrate_float，测试 risch_integrate 函数对浮点数输入的处理
def test_risch_integrate_float():
    # 断言：对 (-60*exp(x) - 19.2*exp(4*x))*exp(4*x) 进行不定积分应该得到 -2.4*exp(8*x) - 12.0*exp(5*x)
    assert risch_integrate((-60*exp(x) - 19.2*exp(4*x))*exp(4*x), x) == -2.4*exp(8*x) - 12.0*exp(5*x)


# 定义测试函数 test_NonElementaryIntegral，测试 risch_integrate 函数对特定函数的输出是否是 NonElementaryIntegral 类型
def test_NonElementaryIntegral():
    # 断言：对 exp(x**2) 进行不定积分应该得到 NonElementaryIntegral 类型的结果
    assert isinstance(risch_integrate(exp(x**2), x), NonElementaryIntegral)
    # 断言：对 x**x * log(x) 进行不定积分应该得到 NonElementaryIntegral 类型的结果
    assert isinstance(risch_integrate(x**x*log(x), x), NonElementaryIntegral)
    # 断言：确保 Integral 类的方法返回的仍然是 NonElementaryIntegral 类型
    assert isinstance(NonElementaryIntegral(x**x*t0, x).subs(t0, log(x)), NonElementaryIntegral)


# 定义测试函数 test_xtothex，测试 risch_integrate 函数对 x**x 的处理
def test_xtothex():
    a = risch_integrate(x**x, x)
    # 断言：对 x**x 进行不定积分应该得到 NonElementaryIntegral(x**x, x)
    assert a == NonElementaryIntegral(x**x, x)
    # 断言：a 应该是 NonElementaryIntegral 类型的实例
    assert isinstance(a, NonElementaryIntegral)


# 定义测试函数 test_DifferentialExtension_equality，测试 DifferentialExtension 对象的相等性
def test_DifferentialExtension_equality():
    DE1 = DE2 = DifferentialExtension(log(x), x)
    # 断言：DE1 和 DE2 应该相等
    assert DE1 == DE2


# 定义测试函数 test_DifferentialExtension_printing，测试 DifferentialExtension 对象的打印输出
def test_DifferentialExtension_printing():
    DE = DifferentialExtension(exp(2*x**2) + log(exp(x**2) + 1), x)
    # 使用断言检查对象 DE 的字符串表示是否与预期相同
    assert repr(DE) == ("DifferentialExtension(dict([('f', exp(2*x**2) + log(exp(x**2) + 1)), "
        "('x', x), ('T', [x, t0, t1]), ('D', [Poly(1, x, domain='ZZ'), Poly(2*x*t0, t0, domain='ZZ[x]'), "
        "Poly(2*t0*x/(t0 + 1), t1, domain='ZZ(x,t0)')]), ('fa', Poly(t1 + t0**2, t1, domain='ZZ[t0]')), "
        "('fd', Poly(1, t1, domain='ZZ')), ('Tfuncs', [Lambda(i, exp(i**2)), Lambda(i, log(t0 + 1))]), "
        "('backsubs', []), ('exts', [None, 'exp', 'log']), ('extargs', [None, x**2, t0 + 1]), "
        "('cases', ['base', 'exp', 'primitive']), ('case', 'primitive'), ('t', t1), "
        "('d', Poly(2*t0*x/(t0 + 1), t1, domain='ZZ(x,t0)')), ('newf', t0**2 + t1), ('level', -1), "
        "('dummy', False)]))")
    
    # 使用断言检查对象 DE 的字符串表示是否与预期相同
    assert str(DE) == ("DifferentialExtension({fa=Poly(t1 + t0**2, t1, domain='ZZ[t0]'), "
        "fd=Poly(1, t1, domain='ZZ'), D=[Poly(1, x, domain='ZZ'), Poly(2*x*t0, t0, domain='ZZ[x]'), "
        "Poly(2*t0*x/(t0 + 1), t1, domain='ZZ(x,t0)')]})")
# 定义一个函数，用于测试 issue 23948 的数学表达式
def test_issue_23948():
    # 定义函数 f(x)，包含复杂的数学表达式
    f = (
        ( (-2*x**5 + 28*x**4 - 144*x**3 + 324*x**2 - 270*x)*log(x)**2
         +(-4*x**6 + 56*x**5 - 288*x**4 + 648*x**3 - 540*x**2)*log(x)
         +(2*x**5 - 28*x**4 + 144*x**3 - 324*x**2 + 270*x)*exp(x)
         +(2*x**5 - 28*x**4 + 144*x**3 - 324*x**2 + 270*x)*log(5)
         -2*x**7 + 26*x**6 - 116*x**5 + 180*x**4 + 54*x**3 - 270*x**2
        )*log(-log(x)**2 - 2*x*log(x) + exp(x) + log(5) - x**2 - x)**2
       +( (4*x**5 - 44*x**4 + 168*x**3 - 216*x**2 - 108*x + 324)*log(x)
         +(-2*x**5 + 24*x**4 - 108*x**3 + 216*x**2 - 162*x)*exp(x)
         +4*x**6 - 42*x**5 + 144*x**4 - 108*x**3 - 324*x**2 + 486*x
        )*log(-log(x)**2 - 2*x*log(x) + exp(x) + log(5) - x**2 - x)
    )/(x*exp(x)**2*log(x)**2 + 2*x**2*exp(x)**2*log(x) - x*exp(x)**3
       +(-x*log(5) + x**3 + x**2)*exp(x)**2)

    # 定义函数 F(x)，作为 f(x) 的预期积分结果
    F = ((x**4 - 12*x**3 + 54*x**2 - 108*x + 81)*exp(-2*x)
        *log(-x**2 - 2*x*log(x) - x + exp(x) - log(x)**2 + log(5))**2)

    # 使用 risch_integrate 函数计算 f(x) 的不定积分，并断言其结果应为 F(x)
    assert risch_integrate(f, x) == F
```