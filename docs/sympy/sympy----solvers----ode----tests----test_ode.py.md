# `D:\src\scipysrc\sympy\sympy\solvers\ode\tests\test_ode.py`

```
# 导入 SymPy 库中不同模块的特定函数和类
from sympy.core.function import (Derivative, Function, Subs, diff)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import acosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan2, cos, sin, tan)
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import Poly
from sympy.series.order import O
from sympy.simplify.radsimp import collect

# 导入 SymPy 库中特定模块的函数和类，以及各种辅助工具
from sympy.solvers.ode import (classify_ode,
    homogeneous_order, dsolve)

from sympy.solvers.ode.subscheck import checkodesol
from sympy.solvers.ode.ode import (classify_sysode,
    constant_renumber, constantsimp, get_numbered_constants, solve_ics)

from sympy.solvers.ode.nonhomogeneous import _undetermined_coefficients_match
from sympy.solvers.ode.single import LinearCoefficients
from sympy.solvers.deutils import ode_order
from sympy.testing.pytest import XFAIL, raises, slow, SKIP
from sympy.utilities.misc import filldedent

# 定义符号常量和符号变量
C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10 = symbols('C0:11')
u, x, y, z = symbols('u,x:z', real=True)
f = Function('f')
g = Function('g')
h = Function('h')

# Note: Examples which were specifically testing Single ODE solver are moved to test_single.py
# and all the system of ode examples are moved to test_systems.py
# Note: the tests below may fail (but still be correct) if ODE solver,
# the integral engine, solve(), or even simplify() changes. Also, in
# differently formatted solutions, the arbitrary constants might not be
# equal.  Using specific hints in tests can help to avoid this.

# Tests of order higher than 1 should run the solutions through
# constant_renumber because it will normalize it (constant_renumber causes
# dsolve() to return different results on different machines)

# 测试函数：验证 get_numbered_constants 函数是否会引发 ValueError 异常
def test_get_numbered_constants():
    with raises(ValueError):
        get_numbered_constants(None)

# 测试函数：使用不同的提示方式对常微分方程进行求解，验证 dsolve 函数返回的结果
def test_dsolve_all_hint():
    eq = f(x).diff(x)
    output = dsolve(eq, hint='all')

    # 匹配 Dummy 变量：
    sol1 = output['separable_Integral']
    _y = sol1.lhs.args[1][0]
    sol1 = output['1st_homogeneous_coeff_subs_dep_div_indep_Integral']
    _u1 = sol1.rhs.args[1].args[1][0]
    # 定义预期的输出字典，包含不同积分类型对应的解析解表达式
    expected = {
        'Bernoulli_Integral': Eq(f(x), C1 + Integral(0, x)),
        '1st_homogeneous_coeff_best': Eq(f(x), C1),
        'Bernoulli': Eq(f(x), C1),
        'nth_algebraic': Eq(f(x), C1),
        'nth_linear_euler_eq_homogeneous': Eq(f(x), C1),
        'nth_linear_constant_coeff_homogeneous': Eq(f(x), C1),
        'separable': Eq(f(x), C1),
        '1st_homogeneous_coeff_subs_indep_div_dep': Eq(f(x), C1),
        'nth_algebraic_Integral': Eq(f(x), C1),
        '1st_linear': Eq(f(x), C1),
        '1st_linear_Integral': Eq(f(x), C1 + Integral(0, x)),
        '1st_exact': Eq(f(x), C1),
        '1st_exact_Integral': Eq(Subs(Integral(0, x) + Integral(1, _y), _y, f(x)), C1),
        'lie_group': Eq(f(x), C1),
        '1st_homogeneous_coeff_subs_dep_div_indep': Eq(f(x), C1),
        '1st_homogeneous_coeff_subs_dep_div_indep_Integral': Eq(log(x), C1 + Integral(-1/_u1, (_u1, f(x)/x))),
        '1st_power_series': Eq(f(x), C1),
        'separable_Integral': Eq(Integral(1, (_y, f(x))), C1 + Integral(0, x)),
        '1st_homogeneous_coeff_subs_indep_div_dep_Integral': Eq(f(x), C1),
        'best': Eq(f(x), C1),
        'best_hint': 'nth_algebraic',  # 设置最佳提示的预期值为 'nth_algebraic'
        'default': 'nth_algebraic',    # 设置默认提示的预期值为 'nth_algebraic'
        'order': 1                      # 设置预期的阶数为 1
    }
    
    # 断言输出与预期的输出字典相等
    assert output == expected

    # 断言使用最佳提示 'best' 求解微分方程 eq 的结果
    assert dsolve(eq, hint='best') == Eq(f(x), C1)
# 定义测试函数 test_dsolve_ics，用于测试 dsolve 函数在给定条件下的行为
def test_dsolve_ics():
    # 使用 pytest 的 raises 断言，验证在特定条件下是否会引发 NotImplementedError 异常
    with raises(NotImplementedError):
        dsolve(f(x).diff(x) - sqrt(f(x)), ics={f(1):1})

# 使用 @slow 装饰器标记的测试函数 test_dsolve_options，用于测试 dsolve 函数的不同选项
@slow
def test_dsolve_options():
    # 定义微分方程 eq
    eq = x*f(x).diff(x) + f(x)
    # 调用 dsolve 函数，使用不同的 hint 参数来求解微分方程
    a = dsolve(eq, hint='all')
    b = dsolve(eq, hint='all', simplify=False)
    c = dsolve(eq, hint='all_Integral')
    # 预期的解的键列表
    keys = ['1st_exact', '1st_exact_Integral', '1st_homogeneous_coeff_best',
        '1st_homogeneous_coeff_subs_dep_div_indep',
        '1st_homogeneous_coeff_subs_dep_div_indep_Integral',
        '1st_homogeneous_coeff_subs_indep_div_dep',
        '1st_homogeneous_coeff_subs_indep_div_dep_Integral', '1st_linear',
        '1st_linear_Integral', 'Bernoulli', 'Bernoulli_Integral',
        'almost_linear', 'almost_linear_Integral', 'best', 'best_hint',
        'default', 'factorable', 'lie_group',
        'nth_linear_euler_eq_homogeneous', 'order',
        'separable', 'separable_Integral']
    # 断言 a 的键列表是否和预期的 keys 相等
    assert sorted(a.keys()) == keys
    # 断言 a 中 'order' 键对应的值是否正确
    assert a['order'] == ode_order(eq, f(x))
    # 断言 a 中 'best' 键对应的值是否正确
    assert a['best'] == Eq(f(x), C1/x)
    # 断言 dsolve(eq, hint='best') 的结果是否正确
    assert dsolve(eq, hint='best') == Eq(f(x), C1/x)
    # 断言 a 中 'default' 键对应的值是否为 'factorable'
    assert a['default'] == 'factorable'
    # 断言 a 中 'best_hint' 键对应的值是否为 'factorable'
    assert a['best_hint'] == 'factorable'
    # 断言 a 中 '1st_exact' 的解是否不包含 Integral 符号
    assert not a['1st_exact'].has(Integral)
    # 断言 a 中 'separable' 的解是否不包含 Integral 符号
    assert not a['separable'].has(Integral)
    # 断言 a 中 '1st_homogeneous_coeff_best' 的解是否不包含 Integral 符号
    assert not a['1st_homogeneous_coeff_best'].has(Integral)
    # 断言 a 中 '1st_homogeneous_coeff_subs_dep_div_indep' 的解是否不包含 Integral 符号
    assert not a['1st_homogeneous_coeff_subs_dep_div_indep'].has(Integral)
    # 断言 a 中 '1st_homogeneous_coeff_subs_indep_div_dep' 的解是否不包含 Integral 符号
    assert not a['1st_homogeneous_coeff_subs_indep_div_dep'].has(Integral)
    # 断言 a 中 '1st_linear' 的解是否不包含 Integral 符号
    assert not a['1st_linear'].has(Integral)
    # 断言 a 中 '1st_linear_Integral' 的解是否包含 Integral 符号
    assert a['1st_linear_Integral'].has(Integral)
    # 断言 a 中 '1st_exact_Integral' 的解是否包含 Integral 符号
    assert a['1st_exact_Integral'].has(Integral)
    # 断言 a 中 '1st_homogeneous_coeff_subs_dep_div_indep_Integral' 的解是否包含 Integral 符号
    assert a['1st_homogeneous_coeff_subs_dep_div_indep_Integral'].has(Integral)
    # 断言 a 中 '1st_homogeneous_coeff_subs_indep_div_dep_Integral' 的解是否包含 Integral 符号
    assert a['1st_homogeneous_coeff_subs_indep_div_dep_Integral'].has(Integral)
    # 断言 a 中 'separable_Integral' 的解是否包含 Integral 符号
    assert a['separable_Integral'].has(Integral)
    # 断言 b 的键列表是否和预期的 keys 相等
    assert sorted(b.keys()) == keys
    # 断言 b 中 'order' 键对应的值是否正确
    assert b['order'] == ode_order(eq, f(x))
    # 断言 b 中 'best' 键对应的值是否正确
    assert b['best'] == Eq(f(x), C1/x)
    # 断言 dsolve(eq, hint='best', simplify=False) 的结果是否正确
    assert dsolve(eq, hint='best', simplify=False) == Eq(f(x), C1/x)
    # 断言 b 中 'default' 键对应的值是否为 'factorable'
    assert b['default'] == 'factorable'
    # 断言 b 中 'best_hint' 键对应的值是否为 'factorable'
    assert b['best_hint'] == 'factorable'
    # 断言 a 中 'separable' 和 b 中 'separable' 的结果不同
    assert a['separable'] != b['separable']
    # 断言 a 中 '1st_homogeneous_coeff_subs_dep_div_indep' 和 b 中 '1st_homogeneous_coeff_subs_dep_div_indep' 的结果不同
    assert a['1st_homogeneous_coeff_subs_dep_div_indep'] != \
        b['1st_homogeneous_coeff_subs_dep_div_indep']
    # 断言 a 中 '1st_homogeneous_coeff_subs_indep_div_dep' 和 b 中 '1st_homogeneous_coeff_subs_indep_div_dep' 的结果不同
    assert a['1st_homogeneous_coeff_subs_indep_div_dep'] != \
        b['1st_homogeneous_coeff_subs_indep_div_dep']
    # 断言 b 中 '1st_exact' 的解是否不包含 Integral 符号
    assert not b['1st_exact'].has(Integral)
    # 断言 b 中 'separable' 的解是否不包含 Integral 符号
    assert not b['separable'].has(Integral)
    # 断言 b 中 '1st_homogeneous_coeff_best' 的解是否不包含 Integral 符号
    assert not b['1st_homogeneous_coeff_best'].has(Integral)
    # 断言检查 '1st_homogeneous_coeff_subs_dep_div_indep' 的表达式不含不定积分
    assert not b['1st_homogeneous_coeff_subs_dep_div_indep'].has(Integral)
    # 断言检查 '1st_homogeneous_coeff_subs_indep_div_dep' 的表达式不含不定积分
    assert not b['1st_homogeneous_coeff_subs_indep_div_dep'].has(Integral)
    # 断言检查 '1st_linear' 的表达式不含不定积分
    assert not b['1st_linear'].has(Integral)
    # 断言检查 '1st_linear_Integral' 的表达式含有不定积分
    assert b['1st_linear_Integral'].has(Integral)
    # 断言检查 '1st_exact_Integral' 的表达式含有不定积分
    assert b['1st_exact_Integral'].has(Integral)
    # 断言检查 '1st_homogeneous_coeff_subs_dep_div_indep_Integral' 的表达式含有不定积分
    assert b['1st_homogeneous_coeff_subs_dep_div_indep_Integral'].has(Integral)
    # 断言检查 '1st_homogeneous_coeff_subs_indep_div_dep_Integral' 的表达式含有不定积分
    assert b['1st_homogeneous_coeff_subs_indep_div_dep_Integral'].has(Integral)
    # 断言检查 'separable_Integral' 的表达式含有不定积分
    assert b['separable_Integral'].has(Integral)
    # 断言检查 c 字典的键按照 Integral_keys 排序后是否相等
    assert sorted(c.keys()) == Integral_keys
    # 使用 raises 函数测试 dsolve 函数在 hint 参数为 'notarealhint' 时是否抛出 ValueError
    raises(ValueError, lambda: dsolve(eq, hint='notarealhint'))
    # 使用 raises 函数测试 dsolve 函数在 hint 参数为 'Liouville' 时是否抛出 ValueError
    raises(ValueError, lambda: dsolve(eq, hint='Liouville'))
    # 断言检查使用 hint 参数为 'all' 和 'best' 时得到的解是否相同
    assert dsolve(f(x).diff(x) - 1/f(x)**2, hint='all')['best'] == \
        dsolve(f(x).diff(x) - 1/f(x)**2, hint='best')
    # 断言检查使用 hint 参数为 '1st_linear_Integral' 时得到的解是否符合预期
    assert dsolve(f(x) + f(x).diff(x) + sin(x).diff(x) + 1, f(x),
                  hint="1st_linear_Integral") == \
        Eq(f(x), (C1 + Integral((-sin(x).diff(x) - 1)*
                exp(Integral(1, x)), x))*exp(-Integral(1, x)))
# 定义一个测试函数，用于测试 classify_ode 函数的不同输入情况
def test_classify_ode():
    # 断言：对于 f(x).diff(x, 2) 的二阶导数等于 f(x)，分类结果应为以下元组
    assert classify_ode(f(x).diff(x, 2), f(x)) == \
        (
        'nth_algebraic',  # 代数方程
        'nth_linear_constant_coeff_homogeneous',  # 常系数齐次线性微分方程
        'nth_linear_euler_eq_homogeneous',  # Euler型齐次线性微分方程
        'Liouville',  # Liouville方程
        '2nd_power_series_ordinary',  # 二阶常规幂级数微分方程
        'nth_algebraic_Integral',  # 代数方程的积分形式
        'Liouville_Integral',  # Liouville方程的积分形式
        )
    # 断言：对于 f(x) 等于 f(x)，分类结果应为以下元组
    assert classify_ode(f(x), f(x)) == ('nth_algebraic', 'nth_algebraic_Integral')
    # 断言：对于 f(x).diff(x) 等于 0，分类结果应为以下元组
    assert classify_ode(Eq(f(x).diff(x), 0), f(x)) == (
        'nth_algebraic',  # 代数方程
        'separable',  # 可分离变量
        '1st_exact',  # 一阶精确微分方程
        '1st_linear',  # 一阶线性微分方程
        'Bernoulli',  # Bernoulli方程
        '1st_homogeneous_coeff_best',  # 最佳一阶齐次系数微分方程
        '1st_homogeneous_coeff_subs_indep_div_dep',  # 齐次系数微分方程，独立变量分离依赖
        '1st_homogeneous_coeff_subs_dep_div_indep',  # 齐次系数微分方程，依赖变量分离独立
        '1st_power_series',  # 一阶幂级数微分方程
        'lie_group',  # Lie群方程
        'nth_linear_constant_coeff_homogeneous',  # 常系数齐次线性微分方程
        'nth_linear_euler_eq_homogeneous',  # Euler型齐次线性微分方程
        'nth_algebraic_Integral',  # 代数方程的积分形式
        'separable_Integral',  # 可分离变量的积分形式
        '1st_exact_Integral',  # 一阶精确微分方程的积分形式
        '1st_linear_Integral',  # 一阶线性微分方程的积分形式
        'Bernoulli_Integral',  # Bernoulli方程的积分形式
        '1st_homogeneous_coeff_subs_indep_div_dep_Integral',  # 齐次系数微分方程，独立变量分离依赖的积分形式
        '1st_homogeneous_coeff_subs_dep_div_indep_Integral',  # 齐次系数微分方程，依赖变量分离独立的积分形式
        )
    # 断言：对于 f(x).diff(x)**2，分类结果应为以下元组
    assert classify_ode(f(x).diff(x)**2, f(x)) == ('factorable',
         'nth_algebraic',  # 代数方程
         'separable',  # 可分离变量
         '1st_exact',  # 一阶精确微分方程
         '1st_linear',  # 一阶线性微分方程
         'Bernoulli',  # Bernoulli方程
         '1st_homogeneous_coeff_best',  # 最佳一阶齐次系数微分方程
         '1st_homogeneous_coeff_subs_indep_div_dep',  # 齐次系数微分方程，独立变量分离依赖
         '1st_homogeneous_coeff_subs_dep_div_indep',  # 齐次系数微分方程，依赖变量分离独立
         '1st_power_series',  # 一阶幂级数微分方程
         'lie_group',  # Lie群方程
         'nth_linear_euler_eq_homogeneous',  # Euler型齐次线性微分方程
         'nth_algebraic_Integral',  # 代数方程的积分形式
         'separable_Integral',  # 可分离变量的积分形式
         '1st_exact_Integral',  # 一阶精确微分方程的积分形式
         '1st_linear_Integral',  # 一阶线性微分方程的积分形式
         'Bernoulli_Integral',  # Bernoulli方程的积分形式
         '1st_homogeneous_coeff_subs_indep_div_dep_Integral',  # 齐次系数微分方程，独立变量分离依赖的积分形式
         '1st_homogeneous_coeff_subs_dep_div_indep_Integral',  # 齐次系数微分方程，依赖变量分离独立的积分形式
         )
    # issue 4749: f(x) 在分类之前应该从最高导数中清除
    # 对三种不同的微分方程进行分类，并存储结果到变量a、b、c中
    a = classify_ode(Eq(f(x).diff(x) + f(x), x), f(x))
    b = classify_ode(f(x).diff(x)*f(x) + f(x)*f(x) - x*f(x), f(x))
    c = classify_ode(f(x).diff(x)/f(x) + f(x)/f(x) - x/f(x), f(x))
    # 断言：变量a的分类结果应为以下元组
    assert a == ('1st_exact',
        '1st_linear',
        'Bernoulli',
        'almost_linear',
        '1st_power_series', "lie_group",
        'nth_linear_constant_coeff_undetermined_coefficients',
        'nth_linear_constant_coeff_variation_of_parameters',
        '1st_exact_Integral',
        '1st_linear_Integral',
        'Bernoulli_Integral',
        'almost_linear_Integral',
        'nth_linear_constant_coeff_variation_of_parameters_Integral')
    # 断言：变量b的分类结果应为以下元组
    assert b == ('factorable',
         '1st_linear',
         'Bernoulli',
         '1st_power_series',
         'lie_group',
         'nth_linear_constant_coeff_undetermined_coefficients',
         'nth_linear_constant_coeff_variation_of_parameters',
         '1st_linear_Integral',
         'Bernoulli_Integral',
         'nth_linear_constant_coeff_variation_of_parameters_Integral')
    # 断言：检查分类ODE结果是否与给定的元组相等
    assert c == ('factorable',
         '1st_linear',
         'Bernoulli',
         '1st_power_series',
         'lie_group',
         'nth_linear_constant_coeff_undetermined_coefficients',
         'nth_linear_constant_coeff_variation_of_parameters',
         '1st_linear_Integral',
         'Bernoulli_Integral',
         'nth_linear_constant_coeff_variation_of_parameters_Integral')

    # 断言：检查对ODE进行分类的结果是否与预期的元组相等
    assert classify_ode(
        2*x*f(x)*f(x).diff(x) + (1 + x)*f(x)**2 - exp(x), f(x)
    ) == ('factorable', '1st_exact', 'Bernoulli', 'almost_linear', 'lie_group',
        '1st_exact_Integral', 'Bernoulli_Integral', 'almost_linear_Integral')

    # 断言：检查特定字符串是否在分类ODE的结果中
    assert 'Riccati_special_minus2' in \
        classify_ode(2*f(x).diff(x) + f(x)**2 - f(x)/x + 3*x**(-2), f(x))

    # 引发值错误异常，检查对非法ODE进行分类时是否引发异常
    raises(ValueError, lambda: classify_ode(x + f(x, y).diff(x).diff(
        y), f(x, y)))

    # 断言：检查对ODE进行分类的结果是否与给定的元组相等
    # issue 5176
    k = Symbol('k')
    assert classify_ode(f(x).diff(x)/(k*f(x) + k*x*f(x)) + 2*f(x)/(k*f(x) +
        k*x*f(x)) + x*f(x).diff(x)/(k*f(x) + k*x*f(x)) + z, f(x)) == \
        ('factorable', 'separable', '1st_exact', '1st_linear', 'Bernoulli',
        '1st_power_series', 'lie_group', 'separable_Integral', '1st_exact_Integral',
        '1st_linear_Integral', 'Bernoulli_Integral')

    # 断言：检查对ODE进行预处理后的分类结果是否与给定的元组相等
    # preprocessing
    ans = ('factorable', 'nth_algebraic', 'separable', '1st_exact', '1st_linear', 'Bernoulli',
        '1st_homogeneous_coeff_best',
        '1st_homogeneous_coeff_subs_indep_div_dep',
        '1st_homogeneous_coeff_subs_dep_div_indep',
        '1st_power_series', 'lie_group',
        'nth_linear_constant_coeff_undetermined_coefficients',
        'nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients',
        'nth_linear_constant_coeff_variation_of_parameters',
        'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters',
        'nth_algebraic_Integral',
        'separable_Integral', '1st_exact_Integral',
        '1st_linear_Integral',
        'Bernoulli_Integral',
        '1st_homogeneous_coeff_subs_indep_div_dep_Integral',
        '1st_homogeneous_coeff_subs_dep_div_indep_Integral',
        'nth_linear_constant_coeff_variation_of_parameters_Integral',
        'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters_Integral')
    assert classify_ode(diff(f(x) + x, x) + diff(f(x), x)) == ans

    # 断言：检查对ODE进行预处理且指定函数 f(x) 后的分类结果是否与给定的元组相等
    assert classify_ode(diff(f(x) + x, x) + diff(f(x), x), f(x),
                        prep=True) == ans

    # 断言：检查对ODE进行分类的结果是否与给定的元组相等
    assert classify_ode(Eq(2*x**3*f(x).diff(x), 0), f(x)) == \
        ('factorable', 'nth_algebraic', 'separable', '1st_exact',
         '1st_linear', 'Bernoulli', '1st_power_series',
         'lie_group', 'nth_linear_euler_eq_homogeneous',
         'nth_algebraic_Integral', 'separable_Integral', '1st_exact_Integral',
         '1st_linear_Integral', 'Bernoulli_Integral')
    # 断言语句：验证 classify_ode 函数对给定的微分方程进行分类并返回预期的结果
    assert classify_ode(Eq(2*f(x)**3*f(x).diff(x), 0), f(x)) == \
        ('factorable', 'nth_algebraic', 'separable', '1st_exact', '1st_linear',
         'Bernoulli', '1st_power_series', 'lie_group', 'nth_algebraic_Integral',
         'separable_Integral', '1st_exact_Integral', '1st_linear_Integral',
         'Bernoulli_Integral')
    
    # 测试问题编号 13864
    assert classify_ode(Eq(diff(f(x), x) - f(x)**x, 0), f(x)) == \
        ('1st_power_series', 'lie_group')
    
    # 断言语句：验证 classify_ode 函数在给定的微分方程上返回的结果类型为字典
    assert isinstance(classify_ode(Eq(f(x), 5), f(x), dict=True), dict)
    
    # 注释：这是用于测试 classify_ode 函数在内部调用时的新行为，默认情况下应返回第一个匹配的提示，
    # 因此 'ordered_hints' 键不会存在。
    assert sorted(classify_ode(Eq(f(x).diff(x), 0), f(x), dict=True).keys()) == \
        ['default', 'nth_linear_constant_coeff_homogeneous', 'order']
    
    # 变量赋值：调用 classify_ode 函数，并指定 Bernoulli 提示
    a = classify_ode(2*x*f(x)*f(x).diff(x) + (1 + x)*f(x)**2 - exp(x), f(x), dict=True, hint='Bernoulli')
    # 断言语句：验证返回的结果字典的键按字母顺序排列
    assert sorted(a.keys()) == ['Bernoulli', 'Bernoulli_Integral', 'default', 'order', 'ordered_hints']
    
    # 测试问题编号 22155
    a = classify_ode(f(x).diff(x) - exp(f(x) - x), f(x))
    # 断言语句：验证 classify_ode 函数对给定的微分方程返回预期的结果元组
    assert a == ('separable',
        '1st_exact', '1st_power_series',
        'lie_group', 'separable_Integral',
        '1st_exact_Integral')
def test_classify_ode_ics():
    # 定义一个简单的微分方程，二阶导数等于函数本身
    eq = f(x).diff(x, x) - f(x)

    # 给定初始条件字典，但没有包含 f(0) 或者 f'(0)
    ics = {x: 1}
    # 断言会抛出 ValueError 异常，因为缺少必要的初始条件
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))


    ############################
    # f(0) 类型 (AppliedUndef) #
    ############################


    # 错误的函数作为键值
    ics = {g(0): 1}
    # 断言会抛出 ValueError 异常，因为初始条件的键必须是 f(0) 的形式
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # 包含变量 x
    ics = {f(x): 1}
    # 断言会抛出 ValueError 异常，因为初始条件的键必须是 f(0) 的形式
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # 参数太多
    ics = {f(0, 0): 1}
    # 断言会抛出 ValueError 异常，因为初始条件的键必须是 f(0) 的形式
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # 初始条件包含变量 x
    ics = {f(0): f(x)}
    # 断言会抛出 ValueError 异常，因为初始条件的值不能包含变量 x
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # 不会抛出异常
    ics = {f(0): f(0)}
    classify_ode(eq, f(x), ics=ics)

    # 不会抛出异常
    ics = {f(0): 1}
    classify_ode(eq, f(x), ics=ics)


    #####################
    # f'(0) 类型 (Subs) #
    #####################

    # 错误的函数作为键值
    ics = {g(x).diff(x).subs(x, 0): 1}
    # 断言会抛出 ValueError 异常，因为初始条件的键必须是 f'(0) 的形式
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # 包含变量 x
    ics = {f(y).diff(y).subs(y, x): 1}
    # 断言会抛出 ValueError 异常，因为初始条件的键必须是 f'(0) 的形式
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # 错误的变量
    ics = {f(y).diff(y).subs(y, 0): 1}
    # 断言会抛出 ValueError 异常，因为初始条件的键必须是 f'(0) 的形式
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # 参数太多
    ics = {f(x, y).diff(x).subs(x, 0): 1}
    # 断言会抛出 ValueError 异常，因为初始条件的键必须是 f'(0) 的形式
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # 对错误变量求导
    ics = {Derivative(f(x), x, y).subs(x, 0): 1}
    # 断言会抛出 ValueError 异常，因为初始条件的键必须是 f'(0) 的形式
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # 初始条件包含变量 x
    ics = {f(x).diff(x).subs(x, 0): f(x)}
    # 断言会抛出 ValueError 异常，因为初始条件的值不能包含变量 x
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # 不会抛出异常
    ics = {f(x).diff(x).subs(x, 0): f(x).diff(x).subs(x, 0)}
    classify_ode(eq, f(x), ics=ics)

    # 不会抛出异常
    ics = {f(x).diff(x).subs(x, 0): 1}
    classify_ode(eq, f(x), ics=ics)

    ###########################
    # f'(y) 类型 (Derivative) #
    ###########################

    # 错误的函数作为键值
    ics = {g(x).diff(x).subs(x, y): 1}
    # 断言会抛出 ValueError 异常，因为初始条件的键必须是 f'(y) 的形式
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # 包含变量 x
    ics = {f(y).diff(y).subs(y, x): 1}
    # 断言会抛出 ValueError 异常，因为初始条件的键必须是 f'(y) 的形式
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # 参数太多
    ics = {f(x, y).diff(x).subs(x, y): 1}
    # 断言会抛出 ValueError 异常，因为初始条件的键必须是 f'(y) 的形式
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # 对错误变量求导
    ics = {Derivative(f(x), x, z).subs(x, y): 1}
    # 断言会抛出 ValueError 异常，因为初始条件的键必须是 f'(y) 的形式
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # 初始条件包含变量 x
    ics = {f(x).diff(x).subs(x, y): f(x)}
    # 断言会抛出 ValueError 异常，因为初始条件的值不能包含变量 x
    raises(ValueError, lambda: classify_ode(eq, f(x), ics=ics))

    # 不会抛出异常
    ics = {f(x).diff(x).subs(x, 0): f(0)}
    classify_ode(eq, f(x), ics=ics)

    # 不会抛出异常
    ics = {f(x).diff(x).subs(x, y): 1}
    classify_ode(eq, f(x), ics=ics)
    # 定义符号变量 k, l, m, n，均为整数类型
    k, l, m, n = symbols('k, l, m, n', Integer=True)
    # 定义符号变量 k1, k2, k3, l1, l2, l3, m1, m2, m3，均为整数类型
    k1, k2, k3, l1, l2, l3, m1, m2, m3 = symbols('k1, k2, k3, l1, l2, l3, m1, m2, m3', Integer=True)
    # 定义符号变量 P, Q, R, p, q, r，均为函数类型
    P, Q, R, p, q, r = symbols('P, Q, R, p, q, r', cls=Function)
    # 定义符号变量 P1, P2, P3, Q1, Q2, R1, R2，均为函数类型
    P1, P2, P3, Q1, Q2, R1, R2 = symbols('P1, P2, P3, Q1, Q2, R1, R2', cls=Function)
    # 定义符号变量 x, y, z，均为函数类型
    x, y, z = symbols('x, y, z', cls=Function)
    # 定义符号变量 t，表示时间
    t = symbols('t')
    # 计算 x1 和 y1 分别为 x(t) 和 y(t) 的时间导数
    x1 = diff(x(t),t) ; y1 = diff(y(t),t) ;

    # 定义方程 eq6，表示系统微分方程组
    eq6 = (Eq(x1, exp(k*x(t))*P(x(t),y(t))), Eq(y1,r(y(t))*P(x(t),y(t))))
    # 定义 sol6，表示对应的分类系统微分方程组的解
    sol6 = {'no_of_equation': 2, 'func_coeff': {(0, x(t), 0): 0, (1, x(t), 1): 0, (0, x(t), 1): 1, (1, y(t), 0): 0, \
    (1, x(t), 0): 0, (0, y(t), 1): 0, (0, y(t), 0): 0, (1, y(t), 1): 1}, 'type_of_equation': 'type2', 'func': \
    [x(t), y(t)], 'is_linear': False, 'eq': [-P(x(t), y(t))*exp(k*x(t)) + Derivative(x(t), t), -P(x(t), \
    y(t))*r(y(t)) + Derivative(y(t), t)], 'order': {y(t): 1, x(t): 1}}
    # 断言分类函数对 eq6 的计算结果与 sol6 相等
    assert classify_sysode(eq6) == sol6

    # 定义方程 eq7，表示系统微分方程组
    eq7 = (Eq(x1, x(t)**2+y(t)/x(t)), Eq(y1, x(t)/y(t)))
    # 定义 sol7，表示对应的分类系统微分方程组的解
    sol7 = {'no_of_equation': 2, 'func_coeff': {(0, x(t), 0): 0, (1, x(t), 1): 0, (0, x(t), 1): 1, (1, y(t), 0): 0, \
    (1, x(t), 0): -1/y(t), (0, y(t), 1): 0, (0, y(t), 0): -1/x(t), (1, y(t), 1): 1}, 'type_of_equation': 'type3', \
    'func': [x(t), y(t)], 'is_linear': False, 'eq': [-x(t)**2 + Derivative(x(t), t) - y(t)/x(t), -x(t)/y(t) + \
    Derivative(y(t), t)], 'order': {y(t): 1, x(t): 1}}
    # 断言分类函数对 eq7 的计算结果与 sol7 相等
    assert classify_sysode(eq7) == sol7

    # 定义方程 eq8，表示系统微分方程组
    eq8 = (Eq(x1, P1(x(t))*Q1(y(t))*R(x(t),y(t),t)), Eq(y1, P1(x(t))*Q1(y(t))*R(x(t),y(t),t)))
    # 定义 sol8，表示对应的分类系统微分方程组的解
    sol8 = {'func': [x(t), y(t)], 'is_linear': False, 'type_of_equation': 'type4', 'eq': \
    [-P1(x(t))*Q1(y(t))*R(x(t), y(t), t) + Derivative(x(t), t), -P1(x(t))*Q1(y(t))*R(x(t), y(t), t) + \
    Derivative(y(t), t)], 'func_coeff': {(0, y(t), 1): 0, (1, y(t), 1): 1, (1, x(t), 1): 0, (0, y(t), 0): 0, \
    (1, x(t), 0): 0, (0, x(t), 0): 0, (1, y(t), 0): 0, (0, x(t), 1): 1}, 'order': {y(t): 1, x(t): 1}, 'no_of_equation': 2}
    # 断言分类函数对 eq8 的计算结果与 sol8 相等
    assert classify_sysode(eq8) == sol8

    # 定义方程 eq11，表示系统微分方程组
    eq11 = (Eq(x1,x(t)*y(t)**3), Eq(y1,y(t)**5))
    # 定义 sol11，表示对应的分类系统微分方程组的解
    sol11 = {'no_of_equation': 2, 'func_coeff': {(0, x(t), 0): -y(t)**3, (1, x(t), 1): 0, (0, x(t), 1): 1, \
    (1, y(t), 0): 0, (1, x(t), 0): 0, (0, y(t), 1): 0, (0, y(t), 0): 0, (1, y(t), 1): 1}, 'type_of_equation': \
    'type1', 'func': [x(t), y(t)], 'is_linear': False, 'eq': [-x(t)*y(t)**3 + Derivative(x(t), t), \
    -y(t)**5 + Derivative(y(t), t)], 'order': {y(t): 1, x(t): 1}}
    # 断言分类函数对 eq11 的计算结果与 sol11 相等
    assert classify_sysode(eq11) == sol11

    # 定义方程 eq13，表示系统微分方程组
    eq13 = (Eq(x1,x(t)*y(t)*sin(t)**2), Eq(y1,y(t)**2*sin(t)**2))
    # 定义 sol13，表示对应的分类系统微分方程组的解
    sol13 = {'no_of_equation': 2, 'func_coeff': {(0, x(t), 0): -y(t)*sin(t)**2, (1, x(t), 1): 0, (0, x(t), 1): 1, \
    (1, y(t), 0): 0, (1, x(t), 0): 0, (0, y(t), 1): 0, (0, y(t), 0): -x(t)*sin(t)**2, (1, y(t), 1): 1}, \
    'type_of_equation': 'type4', 'func': [x(t), y(t)], 'is_linear': False, 'eq': [-x(t)*y(t)*sin(t)**2 + \
    Derivative(x(t), t), -y(t)**2*sin(t)**2 + Derivative(y(t), t)], 'order': {y(t): 1, x(t): 1}}
    # 断言分类函数对 eq13 的计算结果与 sol13 相等
    assert classify_sysode(eq13) == sol13
    Derivative(x(t), t), -y(t)**2*sin(t)**2 + Derivative(y(t), t)], 'order': {y(t): 1, x(t): 1}}
    assert classify_sysode(eq13) == sol13



# 定义一个包含微分方程和顺序信息的字典
{'eq': [Derivative(x(t), t), -y(t)**2*sin(t)**2 + Derivative(y(t), t)], 'order': {y(t): 1, x(t): 1}}
# 使用函数 classify_sysode 对 eq13 进行分类，并断言其分类结果等于 sol13
assert classify_sysode(eq13) == sol13


这段代码假设存在 `eq13` 和 `sol13` 变量，其中 `eq13` 是一个包含微分方程和其阶数信息的字典，`sol13` 则是对 `eq13` 进行分类后的预期结果。代码通过 `assert` 语句来验证 `classify_sysode(eq13)` 返回的结果是否与 `sol13` 相等，用于确保分类函数的正确性。
def test_solve_ics():
    # Basic tests that things work from dsolve.
    # 检查 dsolve 是否能正确处理基本测试案例
    assert dsolve(f(x).diff(x) - 1/f(x), f(x), ics={f(1): 2}) == \
        Eq(f(x), sqrt(2 * x + 2))
    # 解微分方程 f'(x) - f(x) = 0, 初始条件 f(1) = 2，验证结果是否为 f(x) = sqrt(2 * x + 2)
    assert dsolve(f(x).diff(x) - f(x), f(x), ics={f(0): 1}) == Eq(f(x), exp(x))
    # 解微分方程 f'(x) - f(x) = 0, 初始条件 f(0) = 1，验证结果是否为 f(x) = exp(x)
    assert dsolve(f(x).diff(x) - f(x), f(x), ics={f(x).diff(x).subs(x, 0): 1}) == Eq(f(x), exp(x))
    # 解微分方程 f''(x) + f(x) = 0, 初始条件 f(0) = 1, f'(0) = 1，验证结果是否为 f(x) = sin(x) + cos(x)
    assert dsolve([f(x).diff(x) - f(x) + g(x), g(x).diff(x) - g(x) - f(x)],
        [f(x), g(x)], ics={f(0): 1, f(x).diff(x).subs(x, 0): 1}) == Eq(f(x), sin(x) + cos(x))

    # Test cases where dsolve returns two solutions.
    # 测试 dsolve 返回两个解的情况
    eq = (x**2*f(x)**2 - x).diff(x)
    assert dsolve(eq, f(x), ics={f(1): 0}) == [Eq(f(x),
        -sqrt(x - 1)/x), Eq(f(x), sqrt(x - 1)/x)]
    # 解微分方程 (x**2*f(x)**2 - x)' = 0, 初始条件 f(1) = 0，验证结果是否为 f(x) = -sqrt(x - 1)/x 或 sqrt(x - 1)/x
    assert dsolve(eq, f(x), ics={f(x).diff(x).subs(x, 1): 0}) == [Eq(f(x),
        -sqrt(x - S.Half)/x), Eq(f(x), sqrt(x - S.Half)/x)]

    eq = cos(f(x)) - (x*sin(f(x)) - f(x)**2)*f(x).diff(x)
    assert dsolve(eq, f(x),
        ics={f(0):1}, hint='1st_exact', simplify=False) == Eq(x*cos(f(x)) + f(x)**3/3, Rational(1, 3))
    # 解特定的一阶非线性方程，初始条件 f(0) = 1，验证结果是否为 x*cos(f(x)) + f(x)**3/3 = 1/3
    assert dsolve(eq, f(x),
        ics={f(0):1}, hint='1st_exact', simplify=True) == Eq(x*cos(f(x)) + f(x)**3/3, Rational(1, 3))

    assert solve_ics([Eq(f(x), C1*exp(x))], [f(x)], [C1], {f(0): 1}) == {C1: 1}
    # 解初始条件方程 Eq(f(x), C1*exp(x)), 初始条件 f(0) = 1，验证结果为 {C1: 1}

    assert solve_ics([Eq(f(x), C1*sin(x) + C2*cos(x))], [f(x)], [C1, C2],
        {f(0): 1, f(pi/2): 1}) == {C1: 1, C2: 1}
    # 解初始条件方程 Eq(f(x), C1*sin(x) + C2*cos(x)), 初始条件 f(0) = 1, f(pi/2) = 1，验证结果为 {C1: 1, C2: 1}

    assert solve_ics([Eq(f(x), C1*sin(x) + C2*cos(x))], [f(x)], [C1, C2],
        {f(0): 1, f(x).diff(x).subs(x, 0): 1}) == {C1: 1, C2: 1}
    # 解初始条件方程 Eq(f(x), C1*sin(x) + C2*cos(x)), 初始条件 f(0) = 1, f'(0) = 1，验证结果为 {C1: 1, C2: 1}

    assert solve_ics([Eq(f(x), C1*sin(x) + C2*cos(x))], [f(x)], [C1, C2], {f(0): 1}) == \
        {C2: 1}
    # 解初始条件方程 Eq(f(x), C1*sin(x) + C2*cos(x)), 初始条件 f(0) = 1，验证结果为 {C2: 1}

    # Some more complicated tests Refer to PR #16098

    assert set(dsolve(f(x).diff(x)*(f(x).diff(x, 2)-x), ics={f(0):0, f(x).diff(x).subs(x, 1):0})) == \
        {Eq(f(x), 0), Eq(f(x), x ** 3 / 6 - x / 2)}
    # 解微分方程 f'(x)*(f''(x) - x) = 0, 初始条件 f(0) = 0, f'(1) = 0，验证结果是否为 {Eq(f(x), 0), Eq(f(x), x ** 3 / 6 - x / 2)}
    assert set(dsolve(f(x).diff(x)*(f(x).diff(x, 2)-x), ics={f(0):0})) == \
        {Eq(f(x), 0), Eq(f(x), C2*x + x**3/6)}
    # 解微分方程 f'(x)*(f''(x) - x) = 0, 初始条件 f(0) = 0，验证结果是否为 {Eq(f(x), 0), Eq(f(x), C2*x + x**3/6)}

    K, r, f0 = symbols('K r f0')
    sol = Eq(f(x), K*f0*exp(r*x)/((-K + f0)*(f0*exp(r*x)/(-K + f0) - 1)))
    assert (dsolve(Eq(f(x).diff(x), r * f(x) * (1 - f(x) / K)), f(x), ics={f(0): f0})) == sol
    # 解微分方程 f'(x) = r * f(x) * (1 - f(x) / K), 初始条件 f(0) = f0，验证结果是否为 sol

    # Order dependent issues Refer to PR #16098
    # 解微分方程 f'(x)*(f''(x) - x), 初始条件 f'(0) = 0, f(0) = 0，验证结果是否为 {Eq(f(x), 0), Eq(f(x), x ** 3 / 6)}
    assert set(dsolve(f(x).diff(x)*(f(x).diff(x, 2)-x), ics={f(x).diff(x).subs(x,0):0, f(0):0})) == \
        {Eq(f(x), 0), Eq(f(x), x ** 3 / 6)}
    # 解微分方程 f'(x)*(f''(x) - x), 初始条件 f(0) = 0, f'(0) = 0，验证结果是否为 {Eq(f(x), 0), Eq(f(x), x ** 3 / 6)}

    # XXX: Ought to be ValueError
    # 检查特定情况下是否引发 ValueError 异常
    raises(ValueError, lambda: solve_ics([Eq(f(x), C1*sin(x) + C2*cos(x))], [f(x)], [C1, C2], {f(0): 1, f(pi): 1}))

    # Degenerate case. f'(0) is identically 0.
    # 检查特定情况下是否引发 ValueError 异常
    raises(ValueError, lambda: solve_ics([Eq(f(x), sqrt(C1 - x**2))], [f(x)], [C1], {f(x).diff(x).subs(x, 0): 0}))

    EI, q, L = symbols('EI q L')
    # 定义方程：EI*diff(f(x), x, 4) = q，得到方程 Eq(EI*diff(f(x), x, 4), q)
    sols = [Eq(f(x), C1 + C2*x + C3*x**2 + C4*x**3 + q*x**4/(24*EI))]
    # 将 f(x) 添加到函数列表中
    funcs = [f(x)]
    # 定义常数列表
    constants = [C1, C2, C3, C4]
    
    # 第一种情况下的初始条件字典
    ics1 = {f(0): 0,
            # 计算 f(x).diff(x) 在 x=0 处的值，并设为0
            f(x).diff(x).subs(x, 0): 0,
            # 计算 f(L) 关于 L 的二阶导数，并设为0
            f(L).diff(L, 2): 0,
            # 计算 f(L) 关于 L 的三阶导数，并设为0
            f(L).diff(L, 3): 0}
    
    # 第二种情况下的初始条件字典
    ics2 = {f(0): 0,
            # 计算 f(x).diff(x) 在 x=0 处的值，并设为0
            f(x).diff(x).subs(x, 0): 0,
            # 计算 f(x).diff(x, 2) 在 x=L 处的值，并设为0
            Subs(f(x).diff(x, 2), x, L): 0,
            # 计算 f(x).diff(x, 3) 在 x=L 处的值，并设为0
            Subs(f(x).diff(x, 3), x, L): 0}

    # 解决第一组初始条件，得到解的常数
    solved_constants1 = solve_ics(sols, funcs, constants, ics1)
    # 解决第二组初始条件，得到解的常数
    solved_constants2 = solve_ics(sols, funcs, constants, ics2)
    # 断言：两组解的常数应该相等
    assert solved_constants1 == solved_constants2 == {
        C1: 0,
        C2: 0,
        # 计算 C3 的值
        C3: L**2*q/(4*EI),
        # 计算 C4 的值
        C4: -L*q/(6*EI)}

    # 允许初始条件字典中引用 f(x)
    ics = {f(0): f(0)}
    # 断言：求解微分方程 f(x).diff(x) - f(x)，应该得到方程 Eq(f(x), f(0)*exp(x))
    assert dsolve(f(x).diff(x) - f(x), f(x), ics=ics) == Eq(f(x), f(0)*exp(x))

    # 定义另一组初始条件字典
    ics = {f(x).diff(x).subs(x, 0): f(x).diff(x).subs(x, 0),
           f(0): f(0)}
    # 断言：求解微分方程 f(x).diff(x, x) + f(x)，应该得到方程 Eq(f(x), f(0)*cos(x) + f(x).diff(x).subs(x, 0)*sin(x))
    assert dsolve(f(x).diff(x, x) + f(x), f(x), ics=ics) == \
        Eq(f(x), f(0)*cos(x) + f(x).diff(x).subs(x, 0)*sin(x))
# 定义一个测试函数，用于测试常微分方程的阶数计算
def test_ode_order():
    # 定义函数 f 和 g
    f = Function('f')
    g = Function('g')
    # 定义符号 x
    x = Symbol('x')
    
    # 测试常微分方程的阶数计算，验证结果是否为预期值
    assert ode_order(3*x*exp(f(x)), f(x)) == 0
    assert ode_order(x*diff(f(x), x) + 3*x*f(x) - sin(x)/x, f(x)) == 1
    assert ode_order(x**2*f(x).diff(x, x) + x*diff(f(x), x) - f(x), f(x)) == 2
    assert ode_order(diff(x*exp(f(x)), x, x), f(x)) == 2
    assert ode_order(diff(x*diff(x*exp(f(x)), x, x), x), f(x)) == 3
    assert ode_order(diff(f(x), x, x), g(x)) == 0
    assert ode_order(diff(f(x), x, x)*diff(g(x), x), f(x)) == 2
    assert ode_order(diff(f(x), x, x)*diff(g(x), x), g(x)) == 1
    assert ode_order(diff(x*diff(x*exp(f(x)), x, x), x), g(x)) == 0
    
    # issue 5835: ode_order 在不使用 doit() 的情况下也必须处理未求值的导数
    assert ode_order(Derivative(x*f(x), x), f(x)) == 1
    assert ode_order(x*sin(Derivative(x*f(x)**2, x, x)), f(x)) == 2
    assert ode_order(Derivative(x*Derivative(x*exp(f(x)), x, x), x), g(x)) == 0
    assert ode_order(Derivative(f(x), x, x), g(x)) == 0
    assert ode_order(Derivative(x*exp(f(x)), x, x), f(x)) == 2
    assert ode_order(Derivative(f(x), x, x)*Derivative(g(x), x), g(x)) == 1
    assert ode_order(Derivative(x*Derivative(f(x), x, x), x), f(x)) == 3
    assert ode_order(x*sin(Derivative(x*Derivative(f(x), x)**2, x, x)), f(x)) == 3


# 定义测试函数，用于测试齐次方程的阶数计算
def test_homogeneous_order():
    # 验证齐次方程的阶数计算是否为预期值
    assert homogeneous_order(exp(y/x) + tan(y/x), x, y) == 0
    assert homogeneous_order(x**2 + sin(x)*cos(y), x, y) is None
    assert homogeneous_order(x - y - x*sin(y/x), x, y) == 1
    assert homogeneous_order((x*y + sqrt(x**4 + y**4) + x**2*(log(x) - log(y)))/
        (pi*x**Rational(2, 3)*sqrt(y)**3), x, y) == Rational(-1, 6)
    assert homogeneous_order(y/x*cos(y/x) - x/y*sin(y/x) + cos(y/x), x, y) == 0
    assert homogeneous_order(f(x), x, f(x)) == 1
    assert homogeneous_order(f(x)**2, x, f(x)) == 2
    assert homogeneous_order(x*y*z, x, y) == 2
    assert homogeneous_order(x*y*z, x, y, z) == 3
    assert homogeneous_order(x**2*f(x)/sqrt(x**2 + f(x)**2), f(x)) is None
    assert homogeneous_order(f(x, y)**2, x, f(x, y), y) == 2
    assert homogeneous_order(f(x, y)**2, x, f(x), y) is None
    assert homogeneous_order(f(x, y)**2, x, f(x, y)) is None
    assert homogeneous_order(f(y, x)**2, x, y, f(x, y)) is None
    assert homogeneous_order(f(y), f(x), x) is None
    assert homogeneous_order(-f(x)/x + 1/sin(f(x)/ x), f(x), x) == 0
    assert homogeneous_order(log(1/y) + log(x**2), x, y) is None
    assert homogeneous_order(log(1/y) + log(x), x, y) == 0
    assert homogeneous_order(log(x/y), x, y) == 0
    assert homogeneous_order(2*log(1/y) + 2*log(x), x, y) == 0
    a = Symbol('a')
    assert homogeneous_order(a*log(1/y) + a*log(x), x, y) == 0
    assert homogeneous_order(f(x).diff(x), x, y) is None
    assert homogeneous_order(-f(x).diff(x) + x, x, y) is None
    assert homogeneous_order(O(x), x, y) is None
    assert homogeneous_order(x + O(x**2), x, y) is None
    # 断言：验证 x**pi 关于 x 的齐次次数为 pi
    assert homogeneous_order(x**pi, x) == pi
    
    # 断言：验证 x**x 关于 x 的齐次次数为 None
    assert homogeneous_order(x**x, x) is None
    
    # 使用 raises 函数验证：homogeneous_order 函数对于不带足够参数的调用会引发 ValueError 异常
    raises(ValueError, lambda: homogeneous_order(x*y))
@XFAIL
# 标记该测试函数为预期失败的测试，不会影响测试结果
def test_noncircularized_real_imaginary_parts():
    # 如果此测试通过，则应该移除 sympy/solvers/ode.py 文件中第3878-3882行的代码
    # 这段代码测试计算平方根和其实部、虚部是否包含 atan2 函数
    y = sqrt(1+x)
    i, r = im(y), re(y)
    # 断言检查是否实部和虚部中没有 atan2 函数
    assert not (i.has(atan2) and r.has(atan2))


def test_collect_respecting_exponentials():
    # 如果此测试通过，则应该移除 sympy/solvers/ode.py 文件中第1306-1311行的代码
    # 这段代码测试 collect 函数对指数函数的处理是否正确
    sol = 1 + exp(x/2)
    # 断言检查 collect 函数是否正确收集 sol 中的指数函数
    assert sol == collect(sol, exp(x/3))


def test_undetermined_coefficients_match():
    # 检查 _undetermined_coefficients_match 函数的不同输入情况下的返回值是否正确
    assert _undetermined_coefficients_match(g(x), x) == {'test': False}
    assert _undetermined_coefficients_match(sin(2*x + sqrt(5)), x) == \
        {'test': True, 'trialset':
            {cos(2*x + sqrt(5)), sin(2*x + sqrt(5))}}
    assert _undetermined_coefficients_match(sin(x)*cos(x), x) == \
        {'test': False}
    s = {cos(x), x*cos(x), x**2*cos(x), x**2*sin(x), x*sin(x), sin(x)}
    assert _undetermined_coefficients_match(sin(x)*(x**2 + x + 1), x) == \
        {'test': True, 'trialset': s}
    assert _undetermined_coefficients_match(
        sin(x)*x**2 + sin(x)*x + sin(x), x) == {'test': True, 'trialset': s}
    assert _undetermined_coefficients_match(
        exp(2*x)*sin(x)*(x**2 + x + 1), x
    ) == {
        'test': True, 'trialset': {exp(2*x)*sin(x), x**2*exp(2*x)*sin(x),
        cos(x)*exp(2*x), x**2*cos(x)*exp(2*x), x*cos(x)*exp(2*x),
        x*exp(2*x)*sin(x)}}
    assert _undetermined_coefficients_match(1/sin(x), x) == {'test': False}
    assert _undetermined_coefficients_match(log(x), x) == {'test': False}
    assert _undetermined_coefficients_match(2**(x)*(x**2 + x + 1), x) == \
        {'test': True, 'trialset': {2**x, x*2**x, x**2*2**x}}
    assert _undetermined_coefficients_match(x**y, x) == {'test': False}
    assert _undetermined_coefficients_match(exp(x)*exp(2*x + 1), x) == \
        {'test': True, 'trialset': {exp(1 + 3*x)}}
    assert _undetermined_coefficients_match(sin(x)*(x**2 + x + 1), x) == \
        {'test': True, 'trialset': {x*cos(x), x*sin(x), x**2*cos(x),
        x**2*sin(x), cos(x), sin(x)}}
    assert _undetermined_coefficients_match(sin(x)*(x + sin(x)), x) == \
        {'test': False}
    assert _undetermined_coefficients_match(sin(x)*(x + sin(2*x)), x) == \
        {'test': False}
    assert _undetermined_coefficients_match(sin(x)*tan(x), x) == \
        {'test': False}
    assert _undetermined_coefficients_match(
        x**2*sin(x)*exp(x) + x*sin(x) + x, x
    ) == {
        'test': True, 'trialset': {x**2*cos(x)*exp(x), x, cos(x), S.One,
        exp(x)*sin(x), sin(x), x*exp(x)*sin(x), x*cos(x), x*cos(x)*exp(x),
        x*sin(x), cos(x)*exp(x), x**2*exp(x)*sin(x)}}
    assert _undetermined_coefficients_match(4*x*sin(x - 2), x) == {
        'trialset': {x*cos(x - 2), x*sin(x - 2), cos(x - 2), sin(x - 2)},
        'test': True,
    }
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(2**x*x, x) == \
        {'test': True, 'trialset': {2**x, x*2**x}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(2**x*exp(2*x), x) == \
        {'test': True, 'trialset': {2**x*exp(2*x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(exp(-x)/x, x) == \
        {'test': False}
    # 下面的测试来自于普通微分方程，Tenenbaum 和 Pollard 的第 231 页
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(S(4), x) == \
        {'test': True, 'trialset': {S.One}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(12*exp(x), x) == \
        {'test': True, 'trialset': {exp(x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(exp(I*x), x) == \
        {'test': True, 'trialset': {exp(I*x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(sin(x), x) == \
        {'test': True, 'trialset': {cos(x), sin(x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(cos(x), x) == \
        {'test': True, 'trialset': {cos(x), sin(x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(8 + 6*exp(x) + 2*sin(x), x) == \
        {'test': True, 'trialset': {S.One, cos(x), sin(x), exp(x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(x**2, x) == \
        {'test': True, 'trialset': {S.One, x, x**2}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(9*x*exp(x) + exp(-x), x) == \
        {'test': True, 'trialset': {x*exp(x), exp(x), exp(-x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(2*exp(2*x)*sin(x), x) == \
        {'test': True, 'trialset': {exp(2*x)*sin(x), cos(x)*exp(2*x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(x - sin(x), x) == \
        {'test': True, 'trialset': {S.One, x, cos(x), sin(x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(x**2 + 2*x, x) == \
        {'test': True, 'trialset': {S.One, x, x**2}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(4*x*sin(x), x) == \
        {'test': True, 'trialset': {x*cos(x), x*sin(x), cos(x), sin(x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(x*sin(2*x), x) == \
        {'test': True, 'trialset':
            {x*cos(2*x), x*sin(2*x), cos(2*x), sin(2*x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(x**2*exp(-x), x) == \
        {'test': True, 'trialset': {x*exp(-x), x**2*exp(-x), exp(-x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(2*exp(-x) - x**2*exp(-x), x) == \
        {'test': True, 'trialset': {x*exp(-x), x**2*exp(-x), exp(-x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(exp(-2*x) + x**2, x) == \
        {'test': True, 'trialset': {S.One, x, x**2, exp(-2*x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(x*exp(-x), x) == \
        {'test': True, 'trialset': {x*exp(-x), exp(-x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(x + exp(2*x), x) == \
        {'test': True, 'trialset': {S.One, x, exp(2*x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(sin(x) + exp(-x), x) == \
        {'test': True, 'trialset': {cos(x), sin(x), exp(-x)}}
    # 断言测试调用 _undetermined_coefficients_match 函数，验证其对给定表达式的返回结果是否符合预期
    assert _undetermined_coefficients_match(exp(x), x) == \
        {'test': True, 'trialset': {exp(x)}}
    # 从 sin(x)**2 转换而来
    # 断言：验证 _undetermined_coefficients_match 函数对 S.Half - cos(2*x)/2 的处理结果是否符合预期
    assert _undetermined_coefficients_match(S.Half - cos(2*x)/2, x) == \
        {'test': True, 'trialset': {S.One, cos(2*x), sin(2*x)}}
    
    # 断言：验证 _undetermined_coefficients_match 函数对 exp(2*x)*(S.Half + cos(2*x)/2) 的处理结果是否符合预期
    # 转换自 exp(2*x)*sin(x)**2
    assert _undetermined_coefficients_match(
        exp(2*x)*(S.Half + cos(2*x)/2), x
    ) == {
        'test': True,
        'trialset': {exp(2*x)*sin(2*x), cos(2*x)*exp(2*x), exp(2*x)}
    }
    
    # 断言：验证 _undetermined_coefficients_match 函数对 2*x + sin(x) + cos(x) 的处理结果是否符合预期
    assert _undetermined_coefficients_match(2*x + sin(x) + cos(x), x) == \
        {'test': True, 'trialset': {S.One, x, cos(x), sin(x)}}
    
    # 断言：验证 _undetermined_coefficients_match 函数对 cos(x)/2 - cos(3*x)/2 的处理结果是否符合预期
    # 转换自 sin(2*x)*sin(x)
    assert _undetermined_coefficients_match(cos(x)/2 - cos(3*x)/2, x) == \
        {'test': True, 'trialset': {cos(x), cos(3*x), sin(x), sin(3*x)}}
    
    # 断言：验证 _undetermined_coefficients_match 函数对 cos(x**2) 的处理结果是否符合预期
    assert _undetermined_coefficients_match(cos(x**2), x) == {'test': False}
    
    # 断言：验证 _undetermined_coefficients_match 函数对 2**(x**2) 的处理结果是否符合预期
    assert _undetermined_coefficients_match(2**(x**2), x) == {'test': False}
def test_issue_4785_22462():
    # 导入符号 A
    from sympy.abc import A
    # 定义一个微分方程
    eq = x + A*(x + diff(f(x), x) + f(x)) + diff(f(x), x) + f(x) + 2
    # 断言分类函数对微分方程的分类结果
    assert classify_ode(eq, f(x)) == ('factorable', '1st_exact', '1st_linear',
        'Bernoulli', 'almost_linear', '1st_power_series', 'lie_group',
        'nth_linear_constant_coeff_undetermined_coefficients',
        'nth_linear_constant_coeff_variation_of_parameters',
        '1st_exact_Integral', '1st_linear_Integral', 'Bernoulli_Integral',
        'almost_linear_Integral',
        'nth_linear_constant_coeff_variation_of_parameters_Integral')
    
    # issue 4864
    # 定义另一个微分方程
    eq = (x**2 + f(x)**2)*f(x).diff(x) - 2*x*f(x)
    # 断言分类函数对第二个微分方程的分类结果
    assert classify_ode(eq, f(x)) == ('factorable', '1st_exact',
        '1st_homogeneous_coeff_best',
        '1st_homogeneous_coeff_subs_indep_div_dep',
        '1st_homogeneous_coeff_subs_dep_div_indep',
        '1st_power_series',
        'lie_group', '1st_exact_Integral',
        '1st_homogeneous_coeff_subs_indep_div_dep_Integral',
        '1st_homogeneous_coeff_subs_dep_div_indep_Integral')


def test_issue_4825():
    # 断言在解微分方程时引发 ValueError 异常
    raises(ValueError, lambda: dsolve(f(x, y).diff(x) - y*f(x, y), f(x)))
    # 断言分类函数对给定微分方程的分类结果，返回字典形式
    assert classify_ode(f(x, y).diff(x) - y*f(x, y), f(x), dict=True) == \
        {'order': 0, 'default': None, 'ordered_hints': ()}
    # 查看 issue 3793, test Z13.
    # 断言在解微分方程时引发 ValueError 异常
    raises(ValueError, lambda: dsolve(f(x).diff(x), f(y)))
    # 断言分类函数对给定微分方程的分类结果，返回字典形式
    assert classify_ode(f(x).diff(x), f(y), dict=True) == \
        {'order': 0, 'default': None, 'ordered_hints': ()}


def test_constant_renumber_order_issue_5308():
    # 导入 variations 函数
    from sympy.utilities.iterables import variations

    # 断言常数重新编号后的结果
    assert constant_renumber(C1*x + C2*y) == \
        constant_renumber(C1*y + C2*x) == \
        C1*x + C2*y
    # 定义一个表达式
    e = C1*(C2 + x)*(C3 + y)
    # 使用 variations 函数遍历常数组合
    for a, b, c in variations([C1, C2, C3], 3):
        # 断言常数重新编号后的表达式与预期相等
        assert constant_renumber(a*(b + x)*(c + y)) == e


def test_constant_renumber():
    # 定义符号和表达式
    e1, e2, x, y = symbols("e1:3 x y")
    exprs = [e2*x, e1*x + e2*y]

    # 断言常数重新编号后的结果
    assert constant_renumber(exprs[0]) == e2*x
    # 断言常数重新编号后的结果，指定变量为 [x]
    assert constant_renumber(exprs[0], variables=[x]) == C1*x
    # 断言常数重新编号后的结果，指定变量为 [x]，新常数为 [C2]
    assert constant_renumber(exprs[0], variables=[x], newconstants=[C2]) == C2*x
    # 断言常数重新编号后的结果，指定变量为 [x, y]
    assert constant_renumber(exprs, variables=[x, y]) == [C1*x, C1*y + C2*x]
    # 断言常数重新编号后的结果，指定变量为 [x, y]，新常数为 [C3, C4]
    assert constant_renumber(exprs, variables=[x, y], newconstants=symbols("C3:5")) == [C3*x, C3*y + C4*x]


def test_issue_5770():
    # 定义符号和函数
    k = Symbol("k", real=True)
    t = Symbol('t')
    w = Function('w')
    # 求解微分方程
    sol = dsolve(w(t).diff(t, 6) - k**6*w(t), w(t))
    # 断言解中自由符号以 'C' 开头的数量为 6
    assert len([s for s in sol.free_symbols if s.name.startswith('C')]) == 6
    # 断言常数简化函数的结果
    assert constantsimp((C1*cos(x) + C2*cos(x))*exp(x), {C1, C2}) == \
        C1*cos(x)*exp(x)
    # 断言常数简化函数的结果
    assert constantsimp(C1*cos(x) + C2*cos(x) + C3*sin(x), {C1, C2, C3}) == \
        C1*cos(x) + C3*sin(x)
    # 断言常数简化函数的结果
    assert constantsimp(exp(C1 + x), {C1}) == C1*exp(x)
    # 断言常数简化函数的结果
    assert constantsimp(x + C1 + y, {C1, y}) == C1 + x
    # 断言常数简化函数的结果
    assert constantsimp(x + C1 + Integral(x, (x, 1, 2)), {C1}) == C1 + x


def test_issue_5112_5430():
    pass
    # 断言检查表达式 -log(x) + acosh(x) 在变量 x 上的齐次性阶数是否为 None
    assert homogeneous_order(-log(x) + acosh(x), x) is None
    # 断言检查表达式 y - log(x) 在变量 x 和 y 上的齐次性阶数是否为 None
    assert homogeneous_order(y - log(x), x, y) is None
# 定义一个测试函数，用于验证 issue 5095
def test_issue_5095():
    # 创建一个符号函数 f(x)
    f = Function('f')
    # 使用 lambda 函数来检查是否引发 ValueError 异常
    raises(ValueError, lambda: dsolve(f(x).diff(x)**2, f(x), 'fdsjf'))


# 定义一个测试函数，用于验证齐次方程
def test_homogeneous_function():
    # 创建一个符号函数 f(x)
    f = Function('f')
    # 定义多个包含 f(x) 的方程式
    eq1 = tan(x + f(x))
    eq2 = sin((3*x)/(4*f(x)))
    eq3 = cos(x*f(x)*Rational(3, 4))
    eq4 = log((3*x + 4*f(x))/(5*f(x) + 7*x))
    eq5 = exp((2*x**2)/(3*f(x)**2))
    eq6 = log((3*x + 4*f(x))/(5*f(x) + 7*x) + exp((2*x**2)/(3*f(x)**2)))
    eq7 = sin((3*x)/(5*f(x) + x**2))
    # 断言每个方程的齐次阶数（返回值为 None 或 0）
    assert homogeneous_order(eq1, x, f(x)) == None
    assert homogeneous_order(eq2, x, f(x)) == 0
    assert homogeneous_order(eq3, x, f(x)) == None
    assert homogeneous_order(eq4, x, f(x)) == 0
    assert homogeneous_order(eq5, x, f(x)) == 0
    assert homogeneous_order(eq6, x, f(x)) == 0
    assert homogeneous_order(eq7, x, f(x)) == None


# 定义一个测试函数，用于验证线性系数匹配
def test_linear_coeff_match():
    # 定义表达式 n 和 d
    n, d = z*(2*x + 3*f(x) + 5), z*(7*x + 9*f(x) + 11)
    # 计算比例 rat
    rat = n/d
    # 创建包含 rat 的多个表达式
    eq1 = sin(rat) + cos(rat.expand())
    eq2 = rat
    eq3 = log(sin(rat))
    # 创建 LinearCoefficients 对象 obj1, obj2, obj3
    obj1 = LinearCoefficients(eq1)
    obj2 = LinearCoefficients(eq2)
    obj3 = LinearCoefficients(eq3)
    # 预期的线性系数匹配结果 ans
    ans = (4, Rational(-13, 3))
    # 断言每个对象的 _linear_coeff_match 方法返回预期结果
    assert obj1._linear_coeff_match(eq1, f(x)) == ans
    assert obj2._linear_coeff_match(eq2, f(x)) == ans
    assert obj3._linear_coeff_match(eq3, f(x)) == ans

    # 针对无法匹配的情况进行额外断言
    eq4 = (3*x)/f(x)
    obj4 = LinearCoefficients(eq4)
    eq5 = (3*x + 2)/x
    obj5 = LinearCoefficients(eq5)
    eq6 = (3*x + 2*f(x) + 1)/(3*x + 2*f(x) + 5)
    obj6 = LinearCoefficients(eq6)
    eq7 = (3*x + 2*f(x) + sqrt(2))/(3*x + 2*f(x) + 5)
    obj7 = LinearCoefficients(eq7)
    # 断言这些情况下 _linear_coeff_match 返回 None
    assert obj4._linear_coeff_match(eq4, f(x)) is None
    assert obj5._linear_coeff_match(eq5, f(x)) is None
    assert obj6._linear_coeff_match(eq6, f(x)) is None
    assert obj7._linear_coeff_match(eq7, f(x)) is None


# 定义一个测试函数，用于验证常数简化
def test_constantsimp_take_problem():
    # 创建表达式 c
    c = exp(C1) + 2
    # 断言 Poly 对象的生成器数量为 2
    assert len(Poly(constantsimp(exp(C1) + c + c*x, [C1])).gens) == 2


# 定义一个测试函数，用于验证幂级数
def test_series():
    # 创建符号 C1
    C1 = Symbol("C1")
    # 定义不同的微分方程 eq 和对应的解 sol
    eq = f(x).diff(x) - f(x)
    sol = Eq(f(x), C1 + C1*x + C1*x**2/2 + C1*x**3/6 + C1*x**4/24 +
            C1*x**5/120 + O(x**6))
    # 断言使用 1st_power_series 方法求解微分方程的结果 sol
    assert dsolve(eq, hint='1st_power_series') == sol
    # 断言检查是否是给定阶数的解
    assert checkodesol(eq, sol, order=1)[0]

    eq = f(x).diff(x) - x*f(x)
    sol = Eq(f(x), C1*x**4/8 + C1*x**2/2 + C1 + O(x**6))
    assert dsolve(eq, hint='1st_power_series') == sol
    assert checkodesol(eq, sol, order=1)[0]

    eq = f(x).diff(x) - sin(x*f(x))
    sol = Eq(f(x), (x - 2)**2*(1+ sin(4))*cos(4) + (x - 2)*sin(4) + 2 + O(x**3))
    assert dsolve(eq, hint='1st_power_series', ics={f(2): 2}, n=3) == sol
    # FIXME: The solution here should be O((x-2)**3) so is incorrect
    #assert checkodesol(eq, sol, order=1)[0]


# 定义一个测试函数，用于验证二阶幂级数普通方程
@slow
def test_2nd_power_series_ordinary():
    # 创建符号 C1 和 C2
    C1, C2 = symbols("C1 C2")

    # 定义微分方程 eq
    eq = f(x).diff(x, 2) - x*f(x)
    # 断言 classify_ode 返回的类型
    assert classify_ode(eq) == ('2nd_linear_airy', '2nd_power_series_ordinary')
    # 使用方程 Eq 定义解 sol，假设 f(x) 的解为 C2*(x**3/6 + 1) + C1*x*(x**3/12 + 1) + O(x**6)
    sol = Eq(f(x), C2*(x**3/6 + 1) + C1*x*(x**3/12 + 1) + O(x**6))
    # 断言解 sol 是方程 eq 的解
    assert dsolve(eq, hint='2nd_power_series_ordinary') == sol
    # 断言检查方程 eq 的解 sol 是否满足
    assert checkodesol(eq, sol) == (True, 0)

    # 使用方程 Eq 定义解 sol，假设 f(x) 的解为 C2*((x + 2)**4/6 + (x + 2)**3/6 - (x + 2)**2 + 1)
    # 加上 C1*(x + (x + 2)**4/12 - (x + 2)**3/3 + S(2)) + O(x**6)
    sol = Eq(f(x), C2*((x + 2)**4/6 + (x + 2)**3/6 - (x + 2)**2 + 1)
        + C1*(x + (x + 2)**4/12 - (x + 2)**3/3 + S(2))
        + O(x**6))
    # 断言解 sol 是方程 eq 在 x0=-2 处的解
    assert dsolve(eq, hint='2nd_power_series_ordinary', x0=-2) == sol
    # FIXME: Solution should be O((x+2)**6)
    # assert checkodesol(eq, sol) == (True, 0)

    # 使用方程 Eq 定义解 sol，假设 f(x) 的解为 C2*x + C1 + O(x**2)
    sol = Eq(f(x), C2*x + C1 + O(x**2))
    # 断言解 sol 是方程 eq 的解，且计算到 n=2 的阶数
    assert dsolve(eq, hint='2nd_power_series_ordinary', n=2) == sol
    # 断言检查方程 eq 的解 sol 是否满足
    assert checkodesol(eq, sol) == (True, 0)

    # 定义方程 eq
    eq = (1 + x**2)*(f(x).diff(x, 2)) + 2*x*(f(x).diff(x)) -2*f(x)
    # 断言方程 eq 的分类结果
    assert classify_ode(eq) == ('factorable', '2nd_hypergeometric', '2nd_hypergeometric_Integral',
    '2nd_power_series_ordinary')

    # 使用方程 Eq 定义解 sol，假设 f(x) 的解为 C2*(-x**4/3 + x**2 + 1) + C1*x + O(x**6)
    sol = Eq(f(x), C2*(-x**4/3 + x**2 + 1) + C1*x + O(x**6))
    # 断言解 sol 是方程 eq 的解
    assert dsolve(eq, hint='2nd_power_series_ordinary') == sol
    # 断言检查方程 eq 的解 sol 是否满足
    assert checkodesol(eq, sol) == (True, 0)

    # 定义方程 eq
    eq = f(x).diff(x, 2) + x*(f(x).diff(x)) + f(x)
    # 断言方程 eq 的分类结果
    assert classify_ode(eq) == ('factorable', '2nd_power_series_ordinary',)
    # 使用方程 Eq 定义解 sol，假设 f(x) 的解为 C2*(x**4/8 - x**2/2 + 1) + C1*x*(-x**2/3 + 1) + O(x**6)
    sol = Eq(f(x), C2*(x**4/8 - x**2/2 + 1) + C1*x*(-x**2/3 + 1) + O(x**6))
    # 断言解 sol 是方程 eq 的解
    assert dsolve(eq) == sol
    # FIXME: checkodesol fails for this solution...
    # assert checkodesol(eq, sol) == (True, 0)

    # 定义方程 eq
    eq = f(x).diff(x, 2) + f(x).diff(x) - x*f(x)
    # 断言方程 eq 的分类结果
    assert classify_ode(eq) == ('2nd_power_series_ordinary',)
    # 使用方程 Eq 定义解 sol，假设 f(x) 的解为 C2*(-x**4/24 + x**3/6 + 1)
    # + C1*x*(x**3/24 + x**2/6 - x/2 + 1) + O(x**6)
    sol = Eq(f(x), C2*(-x**4/24 + x**3/6 + 1)
            + C1*x*(x**3/24 + x**2/6 - x/2 + 1) + O(x**6))
    # 断言解 sol 是方程 eq 的解
    assert dsolve(eq) == sol
    # FIXME: checkodesol fails for this solution...
    # assert checkodesol(eq, sol) == (True, 0)

    # 定义方程 eq
    eq = f(x).diff(x, 2) + x*f(x)
    # 断言方程 eq 的分类结果
    assert classify_ode(eq) == ('2nd_linear_airy', '2nd_power_series_ordinary')
    # 使用方程 Eq 定义解 sol，假设 f(x) 的解为 C2*(x**6/180 - x**3/6 + 1) + C1*x*(-x**3/12 + 1) + O(x**7)
    sol = Eq(f(x), C2*(x**6/180 - x**3/6 + 1) + C1*x*(-x**3/12 + 1) + O(x**7))
    # 断言解 sol 是方程 eq 的解，且计算到 n=7 的阶数
    assert dsolve(eq, hint='2nd_power_series_ordinary', n=7) == sol
    # 断言检查方程 eq 的解 sol 是否满足
    assert checkodesol(eq, sol) == (True, 0)
# 定义测试函数 test_2nd_power_series_regular，用于测试二阶幂级数正则形式的微分方程求解
def test_2nd_power_series_regular():
    # 定义符号变量 C1, C2, a
    C1, C2, a = symbols("C1 C2 a")
    # 定义第一个微分方程
    eq = x**2*(f(x).diff(x, 2)) - 3*x*(f(x).diff(x)) + (4*x + 4)*f(x)
    # 定义第一个微分方程的解
    sol = Eq(f(x), C1*x**2*(-16*x**3/9 + 4*x**2 - 4*x + 1) + O(x**6))
    # 断言求解微分方程的结果等于预期的解
    assert dsolve(eq, hint='2nd_power_series_regular') == sol
    # 断言检查方程解的正确性
    assert checkodesol(eq, sol) == (True, 0)

    # 定义第二个微分方程
    eq = 4*x**2*(f(x).diff(x, 2)) - 8*x**2*(f(x).diff(x)) + (4*x**2 + 1)*f(x)
    # 定义第二个微分方程的解
    sol = Eq(f(x), C1*sqrt(x)*(x**4/24 + x**3/6 + x**2/2 + x + 1) + O(x**6))
    # 断言求解微分方程的结果等于预期的解
    assert dsolve(eq, hint='2nd_power_series_regular') == sol
    # 断言检查方程解的正确性
    assert checkodesol(eq, sol) == (True, 0)

    # 定义第三个微分方程
    eq = x**2*(f(x).diff(x, 2)) - x**2*(f(x).diff(x)) + (x**2 - 2)*f(x)
    # 定义第三个微分方程的解
    sol = Eq(f(x), C1*(-x**6/720 - 3*x**5/80 - x**4/8 + x**2/2 + x/2 + 1)/x +
            C2*x**2*(-x**3/60 + x**2/20 + x/2 + 1) + O(x**6))
    # 断言求解微分方程的结果等于预期的解
    assert dsolve(eq) == sol
    # 断言检查方程解的正确性
    assert checkodesol(eq, sol) == (True, 0)

    # 定义第四个微分方程
    eq = x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (x**2 - Rational(1, 4))*f(x)
    # 定义第四个微分方程的解
    sol = Eq(f(x), C1*(x**4/24 - x**2/2 + 1)/sqrt(x) +
        C2*sqrt(x)*(x**4/120 - x**2/6 + 1) + O(x**6))
    # 断言求解微分方程的结果等于预期的解
    assert dsolve(eq, hint='2nd_power_series_regular') == sol
    # 断言检查方程解的正确性
    assert checkodesol(eq, sol) == (True, 0)

    # 定义第五个微分方程
    eq = x*f(x).diff(x, 2) + f(x).diff(x) - a*x*f(x)
    # 定义第五个微分方程的解
    sol = Eq(f(x), C1*(a**2*x**4/64 + a*x**2/4 + 1) + O(x**6))
    # 断言求解微分方程的结果等于预期的解
    assert dsolve(eq, f(x), hint="2nd_power_series_regular") == sol
    # 断言检查方程解的正确性
    assert checkodesol(eq, sol) == (True, 0)

    # 定义第六个微分方程
    eq = f(x).diff(x, 2) + ((1 - x)/x)*f(x).diff(x) + (a/x)*f(x)
    # 定义第六个微分方程的解
    sol = Eq(f(x), C1*(-a*x**5*(a - 4)*(a - 3)*(a - 2)*(a - 1)/14400 + \
        a*x**4*(a - 3)*(a - 2)*(a - 1)/576 - a*x**3*(a - 2)*(a - 1)/36 + \
        a*x**2*(a - 1)/4 - a*x + 1) + O(x**6))
    # 断言求解微分方程的结果等于预期的解
    assert dsolve(eq, f(x), hint="2nd_power_series_regular") == sol
    # 断言检查方程解的正确性
    assert checkodesol(eq, sol) == (True, 0)


# 定义测试函数 test_issue_15056，用于测试特定问题
def test_issue_15056():
    # 定义符号变量 t 和 C3
    t = Symbol('t')
    C3 = Symbol('C3')
    # 断言获取带编号的常数项函数的符号
    assert get_numbered_constants(Symbol('C1') * Function('C2')(t)) == C3


# 定义测试函数 test_issue_15913，用于测试特定问题
def test_issue_15913():
    # 定义微分方程 eq
    eq = -C1/x - 2*x*f(x) - f(x) + Derivative(f(x), x)
    # 定义微分方程的解 sol
    sol = C2*exp(x**2 + x) + exp(x**2 + x)*Integral(C1*exp(-x**2 - x)/x, x)
    # 断言检查微分方程解的正确性
    assert checkodesol(eq, sol) == (True, 0)
    # 重新定义微分方程的解 sol
    sol = C1 + C2*exp(-x*y)
    # 重新定义微分方程 eq
    eq = Derivative(y*f(x), x) + f(x).diff(x, 2)
    # 断言检查微分方程解的正确性
    assert checkodesol(eq, sol, f(x)) == (True, 0)


# 定义测试函数 test_issue_16146，用于测试特定问题
def test_issue_16146():
    # 断言检查异常抛出情况，lambda 函数用于调用 dsolve
    raises(ValueError, lambda: dsolve([f(x).diff(x), g(x).diff(x)], [f(x), g(x), h(x)]))
    # 断言检查异常抛出情况，lambda 函数用于调用 dsolve
    raises(ValueError, lambda: dsolve([f(x).diff(x), g(x).diff(x)], [f(x)]))


# 定义测试函数 test_dsolve_remove_redundant_solutions，用于测试特定问题
def test_dsolve_remove_redundant_solutions():
    # 定义微分方程 eq
    eq = (f(x)-2)*f(x).diff(x)
    # 定义微分方程的解 sol
    sol = Eq(f(x), C1)
    # 断言求解微分方程的结果等于预期的解
    assert dsolve(eq) == sol

    # 定义微分方程 eq
    eq = (f(x)-sin(x))*(f(x).diff(x, 2))
    # 定义微分方程的解 sol
    sol = {Eq(f(x), C1 + C2*x), Eq(f(x), sin(x))}
    # 断言求解微分方程的结果等于预期的解（使用 set 进行顺序无关的比较）
    assert set(dsolve(eq)) == sol

    # 定义微分方程 eq
    eq = (f(x)**2-2*f(x)+1)*f(x).diff(x, 3)
    # 定义微分方程的解 sol
    sol = Eq(f(x), C1 + C2*x + C3*x**2)
    # 断言求解微分方程的结果等于预期的解
    assert dsolve(eq) == sol


# 定义测试函数 test_issue_13060，用于测试特定问题
def test_issue_13060():
    # 定
    # 使用 dsolve 函数求解微分方程 eq，返回解析解 sol
    sol = dsolve(eq)
    # 断言检查求解的结果是否符合预期，即方程 eq 的解 sol 经过检验应为 (True, [0, 0])
    assert checkodesol(eq, sol) == (True, [0, 0])
def test_issue_22523():
    # 定义符号变量 N 和 s
    N, s = symbols('N s')
    # 定义 rho 为符号函数
    rho = Function('rho')
    # 创建微分方程，使用 4.0 来确认 nfloat 存在的问题
    eqn = 4.0*N*sqrt(N - 1)*rho(s) + (4*s**2*(N - 1) + (N - 2*s*(N - 1))**2
        )*Derivative(rho(s), (s, 2))
    # 对微分方程进行分类，返回匹配结果的字典
    match = classify_ode(eqn, dict=True, hint='all')
    # 断言二阶幂级数普通微分方程中的项数为 5
    assert match['2nd_power_series_ordinary']['terms'] == 5
    # 定义符号变量 C1 和 C2
    C1, C2 = symbols('C1,C2')
    # 求解微分方程，采用二阶幂级数普通微分方程的方法
    sol = dsolve(eqn, hint='2nd_power_series_ordinary')
    # 断言结果中不存在 r(2.0)
    assert filldedent(sol) == filldedent(str('''
        Eq(rho(s), C2*(1 - 4.0*s**4*sqrt(N - 1.0)/N + 0.666666666666667*s**4/N
        - 2.66666666666667*s**3*sqrt(N - 1.0)/N - 2.0*s**2*sqrt(N - 1.0)/N +
        9.33333333333333*s**4*sqrt(N - 1.0)/N**2 - 0.666666666666667*s**4/N**2
        + 2.66666666666667*s**3*sqrt(N - 1.0)/N**2 -
        5.33333333333333*s**4*sqrt(N - 1.0)/N**3) + C1*s*(1.0 -
        1.33333333333333*s**3*sqrt(N - 1.0)/N - 0.666666666666667*s**2*sqrt(N
        - 1.0)/N + 1.33333333333333*s**3*sqrt(N - 1.0)/N**2) + O(s**6))'''))


def test_issue_22604():
    # 定义符号函数 x1 和 x2
    x1, x2 = symbols('x1, x2', cls = Function)
    # 定义符号变量 t, k1, k2, m1, m2
    t, k1, k2, m1, m2 = symbols('t k1 k2 m1 m2', real = True)
    # 初始化常数值
    k1, k2, m1, m2 = 1, 1, 1, 1
    # 定义两个微分方程
    eq1 = Eq(m1*diff(x1(t), t, 2) + k1*x1(t) - k2*(x2(t) - x1(t)), 0)
    eq2 = Eq(m2*diff(x2(t), t, 2) + k2*(x2(t) - x1(t)), 0)
    # 将微分方程放入列表中
    eqs = [eq1, eq2]
    # 求解微分方程组，带有初始条件
    [x1sol, x2sol] = dsolve(eqs, [x1(t), x2(t)], ics = {x1(0):0, x1(t).diff().subs(t,0):0, \
                                                        x2(0):1, x2(t).diff().subs(t,0):0})
    # 断言解 x1sol 和 x2sol 的正确性
    assert x1sol == Eq(x1(t), sqrt(3 - sqrt(5))*(sqrt(10) + 5*sqrt(2))*cos(sqrt(2)*t*sqrt(3 - sqrt(5))/2)/20 + \
                       (-5*sqrt(2) + sqrt(10))*sqrt(sqrt(5) + 3)*cos(sqrt(2)*t*sqrt(sqrt(5) + 3)/2)/20)
    assert x2sol == Eq(x2(t), (sqrt(5) + 5)*cos(sqrt(2)*t*sqrt(3 - sqrt(5))/2)/10 + (5 - sqrt(5))*cos(sqrt(2)*t*sqrt(sqrt(5) + 3)/2)/10)


def test_issue_22462():
    # 对每个微分方程进行循环
    for de in [
            Eq(f(x).diff(x), -20*f(x)**2 - 500*f(x)/7200),
            Eq(f(x).diff(x), -2*f(x)**2 - 5*f(x)/7)]:
        # 断言 Bernoulli 函数在分类结果中
        assert 'Bernoulli' in classify_ode(de, f(x))


def test_issue_23425():
    # 定义符号变量 x 和 y
    x = symbols('x')
    y = Function('y')
    # 定义微分方程
    eq = Eq(-E**x*y(x).diff().diff() + y(x).diff(), 0)
    # 断言微分方程的分类结果
    assert classify_ode(eq) == \
        ('Liouville', 'nth_order_reducible', \
        '2nd_power_series_ordinary', 'Liouville_Integral')


@SKIP("too slow for @slow")
def test_issue_25820():
    # 定义符号变量 x 和 y
    x = Symbol('x')
    y = Function('y')
    # 定义微分方程
    eq = y(x)**3*y(x).diff(x, 2) + 49
    # 断言求解微分方程不会引发异常
    assert dsolve(eq, y(x)) is not None  # doesn't raise
```