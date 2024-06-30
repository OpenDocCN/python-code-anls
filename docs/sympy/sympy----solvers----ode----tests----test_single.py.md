# `D:\src\scipysrc\sympy\sympy\solvers\ode\tests\test_single.py`

```
# 以下是对单独提示（ODE）解决方案的代码中主要测试的位置，目前位于 sympy/solvers/tests/test_ode.py 文件中
# 这个文件包含了用于解决常微分方程的单个提示的测试函数。
r"""
This File contains test functions for the individual hints used for solving ODEs.

Examples of each solver will be returned by _get_examples_ode_sol_name_of_solver.

Examples should have a key 'XFAIL' which stores the list of hints if they are
expected to fail for that hint.

Functions that are for internal use:

1) _ode_solver_test(ode_examples) - It takes a dictionary of examples returned by
   _get_examples method and tests them with their respective hints.

2) _test_particular_example(our_hint, example_name) - It tests the ODE example corresponding
   to the hint provided.

3) _test_all_hints(runxfail=False) - It is used to test all the examples with all the hints
  currently implemented. It calls _test_all_examples_for_one_hint() which outputs whether the
  given hint functions properly if it classifies the ODE example.
  If runxfail flag is set to True then it will only test the examples which are expected to fail.

  Everytime the ODE of a particular solver is added, _test_all_hints() is to be executed to find
  the possible failures of different solver hints.

4) _test_all_examples_for_one_hint(our_hint, all_examples) - It takes hint as argument and checks
   this hint against all the ODE examples and gives output as the number of ODEs matched, number
   of ODEs which were solved correctly, list of ODEs which gives incorrect solution and list of
   ODEs which raises exception.

"""
from sympy.core.function import (Derivative, diff)  # 导入函数、导数
from sympy.core.mul import Mul  # 导入乘法操作
from sympy.core.numbers import (E, I, Rational, pi)  # 导入常用数学常数
from sympy.core.relational import (Eq, Ne)  # 导入关系运算
from sympy.core.singleton import S  # 导入单例类
from sympy.core.symbol import (Dummy, symbols)  # 导入符号和虚拟符号
from sympy.functions.elementary.complexes import (im, re)  # 导入复数函数
from sympy.functions.elementary.exponential import (LambertW, exp, log)  # 导入指数和对数函数
from sympy.functions.elementary.hyperbolic import (asinh, cosh, sinh, tanh)  # 导入双曲函数
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)  # 导入其他基础函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sec, sin, tan)  # 导入三角函数
from sympy.functions.special.error_functions import (Ei, erfi)  # 导入误差函数
from sympy.functions.special.hyper import hyper  # 导入超几何函数
from sympy.integrals.integrals import (Integral, integrate)  # 导入积分和积分计算
from sympy.polys.rootoftools import rootof  # 导入多项式根的工具函数

from sympy.core import Function, Symbol  # 导入函数和符号
from sympy.functions import airyai, airybi, besselj, bessely, lowergamma  # 导入特殊函数
from sympy.integrals.risch import NonElementaryIntegral  # 导入非元积分
from sympy.solvers.ode import classify_ode, dsolve  # 导入ODE分类和求解函数
from sympy.solvers.ode.ode import allhints, _remove_redundant_solutions  # 导入所有提示和移除冗余解决方案函数
from sympy.solvers.ode.single import (FirstLinear, ODEMatchError,  # 导入单个ODE求解器类和异常
    SingleODEProblem, SingleODESolver, NthOrderReducible)

from sympy.solvers.ode.subscheck import checkodesol  # 导入ODE解的检查函数

from sympy.testing.pytest import raises, slow  # 导入测试框架的断言和性能标记
import traceback  # 导入异常跟踪功能

x = Symbol('x')  # 创建符号对象 x
u = Symbol('u')  # 创建符号对象 u
# 创建一个名为 _u 的虚拟对象 'u'
_u = Dummy('u')
# 创建一个符号变量 'y'
y = Symbol('y')
# 创建函数 f
f = Function('f')
# 创建函数 g
g = Function('g')
# 创建多个符号变量 C1 到 C10
C1, C2, C3, C4, C5, C6, C7, C8, C9, C10  = symbols('C1:11')
# 创建符号变量 a, b, c
a, b, c = symbols('a b c')

# 定义一个多行字符串，包含未匹配示例的提示信息模板
hint_message = """\
Hint did not match the example {example}.

The ODE is:
{eq}.

The expected hint was
{our_hint}\
"""

# 定义一个多行字符串，包含期望的解决方案与 dsolve 返回的解决方案不匹配的信息模板
expected_sol_message = """\
Different solution found from dsolve for example {example}.

The ODE is:
{eq}

The expected solution was
{sol}

What dsolve returned is:
{dsolve_sol}\
"""

# 定义一个多行字符串，包含找到的解决方案不正确的信息模板
checkodesol_msg = """\
solution found is not correct for example {example}.

The ODE is:
{eq}\
"""

# 定义一个多行字符串，包含使用特定提示时 dsolve 返回的解决方案不正确的信息模板
dsol_incorrect_msg = """\
solution returned by dsolve is incorrect when using {hint}.

The ODE is:
{eq}

The expected solution was
{sol}

what dsolve returned is:
{dsolve_sol}

You can test this with:

eq = {eq}
sol = dsolve(eq, hint='{hint}')
print(sol)
print(checkodesol(eq, sol))

"""

# 定义一个多行字符串，包含 dsolve 引发异常时的信息模板
exception_msg = """\
dsolve raised exception : {e}

when using {hint} for the example {example}

You can test this with:

from sympy.solvers.ode.tests.test_single import _test_an_example

_test_an_example('{hint}', example_name = '{example}')

The ODE is:
{eq}

\
"""

# 定义一个多行字符串，包含测试使用的提示信息的统计结果模板
check_hint_msg = """\
Tested hint was : {hint}

Total of {matched} examples matched with this hint.

Out of which {solve} gave correct results.

Examples which gave incorrect results are {unsolve}.

Examples which raised exceptions are {exceptions}
\
"""


# 定义一个装饰器函数，用于向内部函数添加示例键
def _add_example_keys(func):
    def inner():
        solver=func()
        examples=[]
        for example in solver['examples']:
            temp={
                'eq': solver['examples'][example]['eq'],
                'sol': solver['examples'][example]['sol'],
                'XFAIL': solver['examples'][example].get('XFAIL', []),
                'func': solver['examples'][example].get('func',solver['func']),
                'example_name': example,
                'slow': solver['examples'][example].get('slow', False),
                'simplify_flag':solver['examples'][example].get('simplify_flag',True),
                'checkodesol_XFAIL': solver['examples'][example].get('checkodesol_XFAIL', False),
                'dsolve_too_slow':solver['examples'][example].get('dsolve_too_slow',False),
                'checkodesol_too_slow':solver['examples'][example].get('checkodesol_too_slow',False),
                'hint': solver['hint']
            }
            examples.append(temp)
        return examples
    return inner()


# 定义一个函数，用于测试常微分方程求解器的函数
def _ode_solver_test(ode_examples, run_slow_test=False):
    for example in ode_examples:
        if ((not run_slow_test) and example['slow']) or (run_slow_test and (not example['slow'])):
            continue

        result = _test_particular_example(example['hint'], example, solver_flag=True)
        if result['xpass_msg'] != "":
            print(result['xpass_msg'])


# 定义一个函数，用于测试所有提示的效果
def _test_all_hints(runxfail=False):
    all_hints = list(allhints)+["default"]
    all_examples = _get_all_examples()
    # 遍历给定的所有提示列表
    for our_hint in all_hints:
        # 检查当前提示是否以'_Integral'结尾或者包含'series'
        if our_hint.endswith('_Integral') or 'series' in our_hint:
            # 如果条件满足，则跳过当前循环，继续下一个提示
            continue
        # 对当前提示调用指定函数，测试所有示例
        _test_all_examples_for_one_hint(our_hint, all_examples, runxfail)
# 判断给定的解是否符合预期解，如果 dsolve_sol 是列表，则检查其中任意一个子解是否符合预期解；否则直接比较解是否相等
def _test_dummy_sol(expected_sol, dsolve_sol):
    if type(dsolve_sol) == list:
        return any(expected_sol.dummy_eq(sub_dsol) for sub_dsol in dsolve_sol)
    else:
        return expected_sol.dummy_eq(dsolve_sol)

# 测试特定示例，根据示例名从所有示例中找到对应的示例并进行测试
def _test_an_example(our_hint, example_name):
    all_examples = _get_all_examples()
    for example in all_examples:
        if example['example_name'] == example_name:
            _test_particular_example(our_hint, example)

# 对特定的 ODE 示例进行测试
def _test_particular_example(our_hint, ode_example, solver_flag=False):
    # 从示例中获取方程、预期解、示例名称、函数等信息
    eq = ode_example['eq']
    expected_sol = ode_example['sol']
    example = ode_example['example_name']
    xfail = our_hint in ode_example['XFAIL']
    func = ode_example['func']
    result = {'msg': '', 'xpass_msg': ''}
    simplify_flag = ode_example['simplify_flag']
    checkodesol_XFAIL = ode_example['checkodesol_XFAIL']
    dsolve_too_slow = ode_example['dsolve_too_slow']
    checkodesol_too_slow = ode_example['checkodesol_too_slow']
    xpass = True
    # 如果 solver_flag 为 True，则检查我们的提示是否在方程的分类中，如果不在则抛出断言错误
    if solver_flag:
        if our_hint not in classify_ode(eq, func):
            message = hint_message.format(example=example, eq=eq, our_hint=our_hint)
            raise AssertionError(message)
    # 如果我们的提示在对给定方程分类后存在
    if our_hint in classify_ode(eq, func):
        # 将结果列表中的'match_list'键设置为示例
        result['match_list'] = example
        try:
            # 如果求解不是太慢，使用dsolve函数求解方程
            if not (dsolve_too_slow):
                dsolve_sol = dsolve(eq, func, simplify=simplify_flag, hint=our_hint)
            else:
                # 如果期望的解只有一个，则直接使用这个解
                if len(expected_sol) == 1:
                    dsolve_sol = expected_sol[0]
                else:
                    dsolve_sol = expected_sol

        except Exception as e:
            # 如果出现异常，将dsolve_sol置为空列表，记录异常示例
            dsolve_sol = []
            result['exception_list'] = example
            # 如果不是求解器标志，打印异常的跟踪信息
            if not solver_flag:
                traceback.print_exc()
            # 设置结果消息，包含异常信息、提示、示例、方程等信息
            result['msg'] = exception_msg.format(e=str(e), hint=our_hint, example=example, eq=eq)
            # 如果是求解器标志并且不是xfail，打印结果消息并抛出异常
            if solver_flag and not xfail:
                print(result['msg'])
                raise
            # xpass标志设为False，表示不通过
            xpass = False

        # 如果是求解器标志并且dsolve_sol不为空列表
        if solver_flag and dsolve_sol != []:
            expect_sol_check = False
            # 如果dsolve_sol是列表类型，逐个检查期望解是否包含在dsolve_sol中
            if type(dsolve_sol) == list:
                for sub_sol in expected_sol:
                    if sub_sol.has(Dummy):
                        expect_sol_check = not _test_dummy_sol(sub_sol, dsolve_sol)
                    else:
                        expect_sol_check = sub_sol not in dsolve_sol
                    if expect_sol_check:
                        break
            else:
                # 如果dsolve_sol不是列表类型，直接检查期望解是否不在dsolve_sol中
                expect_sol_check = dsolve_sol not in expected_sol
                for sub_sol in expected_sol:
                    if sub_sol.has(Dummy):
                        expect_sol_check = not _test_dummy_sol(sub_sol, dsolve_sol)

            # 如果期望解检查失败，生成错误消息并抛出断言错误
            if expect_sol_check:
                message = expected_sol_message.format(example=example, eq=eq, sol=expected_sol, dsolve_sol=dsolve_sol)
                raise AssertionError(message)

            # 如果期望的解通过了检查，设置期望的检查odesol结果为True
            expected_checkodesol = [(True, 0) for i in range(len(expected_sol))]
            if len(expected_sol) == 1:
                expected_checkodesol = (True, 0)

            # 如果不是太慢的odesol检查，并且不是XFAIL状态下的odesol检查
            if not checkodesol_too_slow:
                if not checkodesol_XFAIL:
                    # 检查odesol并与期望的检查结果进行比较
                    if checkodesol(eq, dsolve_sol, func, solve_for_func=False) != expected_checkodesol:
                        result['unsolve_list'] = example
                        xpass = False
                        # 生成错误消息，表示dsolve_sol的odesol检查不正确
                        message = dsol_incorrect_msg.format(hint=our_hint, eq=eq, sol=expected_sol, dsolve_sol=dsolve_sol)
                        if solver_flag:
                            message = checkodesol_msg.format(example=example, eq=eq)
                            raise AssertionError(message)
                        else:
                            result['msg'] = 'AssertionError: ' + message

        # 如果xpass为True并且处于xfail状态下，设置xpass_msg
        if xpass and xfail:
            result['xpass_msg'] = example + "is now passing for the hint" + our_hint
    # 返回结果字典
    return result
# 定义函数 _test_all_examples_for_one_hint，用于测试给定提示在所有示例中的表现
def _test_all_examples_for_one_hint(our_hint, all_examples=[], runxfail=None):
    # 如果未提供所有示例的列表，则获取所有示例列表
    if all_examples == []:
        all_examples = _get_all_examples()
    # 初始化匹配列表、未解决列表和异常列表
    match_list, unsolve_list, exception_list = [], [], []
    # 遍历所有示例
    for ode_example in all_examples:
        # 检查当前示例中是否标记为 XFAIL
        xfail = our_hint in ode_example['XFAIL']
        # 如果设置了 runxfail 且当前示例不是 XFAIL，则跳过
        if runxfail and not xfail:
            continue
        # 如果当前示例是 XFAIL，则直接跳过
        if xfail:
            continue
        # 测试特定提示在当前示例中的表现
        result = _test_particular_example(our_hint, ode_example)
        # 将匹配列表、未解决列表和异常列表进行累加
        match_list += result.get('match_list', [])
        unsolve_list += result.get('unsolve_list', [])
        exception_list += result.get('exception_list', [])
        # 如果设置了 runxfail，则输出特定结果消息
        if runxfail is not None:
            msg = result['msg']
            if msg != '':
                print(result['msg'])
            # 打印可选的 xpass_msg，如果有的话

    # 如果未设置 runxfail，则生成检查提示消息
    if runxfail is None:
        # 计算匹配数量
        match_count = len(match_list)
        # 计算成功解决的数量
        solved = len(match_list) - len(unsolve_list) - len(exception_list)
        # 格式化并打印检查提示消息
        msg = check_hint_msg.format(hint=our_hint, matched=match_count, solve=solved, unsolve=unsolve_list, exceptions=exception_list)
        print(msg)


# 定义函数 test_SingleODESolver，用于测试 SingleODESolver 类
def test_SingleODESolver():
    # 测试未实现方法是否引发 NotImplementedError
    # 子类应覆盖这些方法
    problem = SingleODEProblem(f(x).diff(x), f(x), x)
    solver = SingleODESolver(problem)
    raises(NotImplementedError, lambda: solver.matches())
    raises(NotImplementedError, lambda: solver.get_general_solution())
    raises(NotImplementedError, lambda: solver._matches())
    raises(NotImplementedError, lambda: solver._get_general_solution())

    # 测试 FirstLinear solver 无法解决的情况
    # 这里测试它不匹配，并且请求一般解引发 ODEMatchError
    problem = SingleODEProblem(f(x).diff(x) + f(x)*f(x), f(x), x)
    solver = FirstLinear(problem)
    raises(ODEMatchError, lambda: solver.get_general_solution())

    solver = FirstLinear(problem)
    assert solver.matches() is False

    # 这些只是用于测试 ODE 的顺序

    # 测试一阶 ODE 的顺序
    problem = SingleODEProblem(f(x).diff(x) + f(x), f(x), x)
    assert problem.order == 1

    # 测试四阶 ODE 的顺序
    problem = SingleODEProblem(f(x).diff(x, 4) + f(x).diff(x, 2) - f(x).diff(x, 3), f(x), x)
    assert problem.order == 4

    # 测试是否为自主 ODE
    problem = SingleODEProblem(f(x).diff(x, 3) + f(x).diff(x, 2) - f(x)**2, f(x), x)
    assert problem.is_autonomous == True

    # 测试非自主 ODE
    problem = SingleODEProblem(f(x).diff(x, 3) + x*f(x).diff(x, 2) - f(x)**2, f(x), x)
    assert problem.is_autonomous == False


# 定义函数 test_linear_coefficients，用于测试线性系数的 ODE 解算器
def test_linear_coefficients():
    _ode_solver_test(_get_examples_ode_sol_linear_coefficients)


# 定义函数 test_1st_homogeneous_coeff_ode，用于测试一阶齐次系数 ODE
@slow
def test_1st_homogeneous_coeff_ode():
    # 这些标记为 test_1st_homogeneous_coeff_corner_case
    eq1 = f(x).diff(x) - f(x)/x
    c1 = classify_ode(eq1, f(x))
    eq2 = x*f(x).diff(x) - f(x)
    c2 = classify_ode(eq2, f(x))
    sdi = "1st_homogeneous_coeff_subs_dep_div_indep"
    sid = "1st_homogeneous_coeff_subs_indep_div_dep"
    assert sid not in c1 and sdi not in c1
    # 断言确保 sid 和 sdi 不在 c2 中，否则引发异常
    assert sid not in c2 and sdi not in c2
    # 测试 _get_examples_ode_sol_1st_homogeneous_coeff_subs_dep_div_indep 函数的求解器
    _ode_solver_test(_get_examples_ode_sol_1st_homogeneous_coeff_subs_dep_div_indep)
    # 测试 _get_examples_ode_sol_1st_homogeneous_coeff_best 函数的求解器
    _ode_solver_test(_get_examples_ode_sol_1st_homogeneous_coeff_best)
# 用于测试慢速示例的第一个同质系数一阶常系数ODE求解器
@slow
def test_slow_examples_1st_homogeneous_coeff_ode():
    # 调用 _ode_solver_test 函数，传入第一个同质系数一阶常系数ODE的求解器函数，并指定运行慢速测试模式
    _ode_solver_test(_get_examples_ode_sol_1st_homogeneous_coeff_subs_dep_div_indep, run_slow_test=True)
    # 调用 _ode_solver_test 函数，传入第一个同质系数一阶常系数ODE的最佳求解器函数，并指定运行慢速测试模式
    _ode_solver_test(_get_examples_ode_sol_1st_homogeneous_coeff_best, run_slow_test=True)


# 测试n次线性恒定系数齐次ODE求解器
@slow
def test_nth_linear_constant_coeff_homogeneous():
    # 调用 _ode_solver_test 函数，传入n次线性恒定系数齐次ODE的求解器函数


# 用于测试慢速示例的n次线性恒定系数齐次ODE求解器
@slow
def test_slow_examples_nth_linear_constant_coeff_homogeneous():
    # 调用 _ode_solver_test 函数，传入n次线性恒定系数齐次ODE的求解器函数，并指定运行慢速测试模式
    _ode_solver_test(_get_examples_ode_sol_nth_linear_constant_coeff_homogeneous, run_slow_test=True)


# Airy方程的测试
def test_Airy_equation():
    # 调用 _ode_solver_test 函数，传入Airy方程的求解器函数
    _ode_solver_test(_get_examples_ode_sol_2nd_linear_airy)


# 测试Lie群ODE求解器
@slow
def test_lie_group():
    # 调用 _ode_solver_test 函数，传入Lie群ODE的求解器函数
    _ode_solver_test(_get_examples_ode_sol_lie_group)


# 测试可分离约化ODE求解器
@slow
def test_separable_reduced():
    # 计算f(x)关于x的导数
    df = f(x).diff(x)
    # 计算等式
    eq = (x / f(x))*df  + tan(x**2*f(x) / (x**2*f(x) - 1))
    # 断言ODE的分类结果
    assert classify_ode(eq) == ('factorable', 'separable_reduced', 'lie_group',
        'separable_reduced_Integral')
    # 调用 _ode_solver_test 函数，传入可分离约化ODE的求解器函数


# 用于测试慢速示例的可分离约化ODE求解器
@slow
def test_slow_examples_separable_reduced():
    # 调用 _ode_solver_test 函数，传入可分离约化ODE的求解器函数，并指定运行慢速测试模式
    _ode_solver_test(_get_examples_ode_sol_separable_reduced, run_slow_test=True)


# 测试第二类2F1超几何ODE求解器
@slow
def test_2nd_2F1_hypergeometric():
    # 调用 _ode_solver_test 函数，传入第二类2F1超几何ODE的求解器函数
    _ode_solver_test(_get_examples_ode_sol_2nd_2F1_hypergeometric)


# 测试第二类2F1超几何ODE的积分解
def test_2nd_2F1_hypergeometric_integral():
    # 构造ODE等式
    eq = x*(x-1)*f(x).diff(x, 2) + (-1+ S(7)/2*x)*f(x).diff(x) + f(x)
    # 构造期望解
    sol = Eq(f(x), (C1 + C2*Integral(exp(Integral((1 - x/2)/(x*(x - 1)), x))/(1 -
          x/2)**2, x))*exp(Integral(1/(x - 1), x)/4)*exp(-Integral(7/(x -
          1), x)/4)*hyper((S(1)/2, -1), (1,), x))
    # 断言ODE的dsolve结果
    assert sol == dsolve(eq, hint='2nd_hypergeometric_Integral')
    # 断言ODE解的有效性
    assert checkodesol(eq, sol) == (True, 0)


# 用于测试第二类非线性自治守恒ODE求解器
@slow
def test_2nd_nonlinear_autonomous_conserved():
    # 调用 _ode_solver_test 函数，传入第二类非线性自治守恒ODE的求解器函数
    _ode_solver_test(_get_examples_ode_sol_2nd_nonlinear_autonomous_conserved)


# 测试第二类非线性自治守恒ODE的积分解
def test_2nd_nonlinear_autonomous_conserved_integral():
    # 构造ODE等式
    eq = f(x).diff(x, 2) + asin(f(x))
    # 构造实际解列表
    actual = [Eq(Integral(1/sqrt(C1 - 2*Integral(asin(_u), _u)), (_u, f(x))), C2 + x),
    Eq(Integral(1/sqrt(C1 - 2*Integral(asin(_u), _u)), (_u, f(x))), C2 - x)]
    # 求解ODE，指定使用积分解法
    solved = dsolve(eq, hint='2nd_nonlinear_autonomous_conserved_Integral', simplify=False)
    # 断言实际解与求解结果的相等性
    for a,s in zip(actual, solved):
        assert a.dummy_eq(s)
    # checkodesol 无法简化带有f(x)的积分方程解
    assert checkodesol(eq, [s.doit() for s in solved]) == [(True, 0), (True, 0)]


# 测试第二类线性贝塞尔ODE求解器
@slow
def test_2nd_linear_bessel_equation():
    # 调用 _ode_solver_test 函数，传入第二类线性贝塞尔ODE的求解器函数
    _ode_solver_test(_get_examples_ode_sol_2nd_linear_bessel)


# 测试n次代数ODE求解器
@slow
def test_nth_algebraic():
    # 构造f(x) + f(x)*f(x).diff(x)的等式
    eqn = f(x) + f(x)*f(x).diff(x)
    # 构造解列表
    solns = [Eq(f(x), exp(x)),
             Eq(f(x), C1*exp(C2*x))]
    # 移除多余的解，保留最终解列表
    solns_final =  _remove_redundant_solutions(eqn, solns, 2, x)
    # 断言最终解列表
    assert solns_final == [Eq(f(x), C1*exp(C2*x))]
    # 调用 _ode_solver_test 函数，传入n次代数ODE的求解器函数


# 用于测试慢速示例的n次线性恒定系数参数变化ODE求解器
@slow
def test_slow_examples_nth_linear_constant_coeff_var_of_parameters():
    # 调用 _ode_solver_test 函数，传入n次线性恒定系数参数变化ODE的求解器函数
    # 调用函数 _ode_solver_test，并传入 _get_examples_ode_sol_nth_linear_var_of_parameters 函数作为参数
    # 参数 run_slow_test 被设置为 True，表示运行较慢的测试
    _ode_solver_test(_get_examples_ode_sol_nth_linear_var_of_parameters, run_slow_test=True)
def test_nth_linear_constant_coeff_var_of_parameters():
    # 调用 _ode_solver_test 函数来测试 nth 线性常系数变参数方程的求解
    _ode_solver_test(_get_examples_ode_sol_nth_linear_var_of_parameters)


@slow
def test_nth_linear_constant_coeff_variation_of_parameters__integral():
    # solve_variation_of_parameters 如果 simplify=False，不应尝试简化 Wronskian。
    # 如果 wronskian() 自身可以简化结果，这个测试可能会失败。
    
    # 定义微分方程
    eq = f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x) - 2*x - exp(I*x)
    
    # 分别使用简化和不简化两种方式求解微分方程
    sol_simp = dsolve(eq, f(x), hint='nth_linear_constant_coeff_variation_of_parameters_Integral', simplify=True)
    sol_nsimp = dsolve(eq, f(x), hint='nth_linear_constant_coeff_variation_of_parameters_Integral', simplify=False)
    
    # 确保简化和不简化的解不相等
    assert sol_simp != sol_nsimp
    
    # 验证简化解的正确性
    assert checkodesol(eq, sol_simp, order=5, solve_for_func=False) == (True, 0)


@slow
def test_slow_examples_1st_exact():
    # 调用 _ode_solver_test 函数来测试一阶完全可解方程的求解
    _ode_solver_test(_get_examples_ode_sol_1st_exact, run_slow_test=True)


@slow
def test_1st_exact():
    # 调用 _ode_solver_test 函数来测试一阶完全可解方程的求解
    _ode_solver_test(_get_examples_ode_sol_1st_exact)


def test_1st_exact_integral():
    # 定义微分方程
    eq = cos(f(x)) - (x*sin(f(x)) - f(x)**2)*f(x).diff(x)
    
    # 使用积分作为 hint 求解微分方程
    sol_1 = dsolve(eq, f(x), hint='1st_exact_Integral', simplify=False)
    
    # 验证解的正确性
    assert checkodesol(eq, sol_1, order=1, solve_for_func=False)


@slow
def test_slow_examples_nth_order_reducible():
    # 调用 _ode_solver_test 函数来测试可降阶的高阶方程的求解
    _ode_solver_test(_get_examples_ode_sol_nth_order_reducible, run_slow_test=True)


@slow
def test_slow_examples_nth_linear_constant_coeff_undetermined_coefficients():
    # 调用 _ode_solver_test 函数来测试 nth 线性常系数未定系数法的求解
    _ode_solver_test(_get_examples_ode_sol_nth_linear_undetermined_coefficients, run_slow_test=True)


@slow
def test_slow_examples_separable():
    # 调用 _ode_solver_test 函数来测试可分离方程的求解
    _ode_solver_test(_get_examples_ode_sol_separable, run_slow_test=True)


@slow
def test_nth_linear_constant_coeff_undetermined_coefficients():
    # 这个测试案例展示了在 nth 线性常系数未定系数法中虚数常数的分类。
    
    # 定义微分方程
    eq = Eq(diff(f(x), x), I*f(x) + S.Half - I)
    
    # 预期的解的 hint
    our_hint = 'nth_linear_constant_coeff_undetermined_coefficients'
    
    # 确保微分方程被正确分类
    assert our_hint in classify_ode(eq)
    
    # 调用 _ode_solver_test 函数来测试 nth 线性常系数未定系数法的求解
    _ode_solver_test(_get_examples_ode_sol_nth_linear_undetermined_coefficients)


def test_nth_order_reducible():
    # 定义函数 F，用于检测高阶方程是否可降阶
    F = lambda eq: NthOrderReducible(SingleODEProblem(eq, f(x), x))._matches()
    D = Derivative
    
    # 一系列测试案例来验证高阶方程是否可降阶
    assert F(D(y*f(x), x, y) + D(f(x), x)) == False
    assert F(D(y*f(y), y, y) + D(f(y), y)) == False
    assert F(f(x)*D(f(x), x) + D(f(x), x, 2))== False
    assert F(D(x*f(y), y, 2) + D(u*y*f(x), x, 3)) == False  # 按设计不进行简化
    assert F(D(f(y), y, 2) + D(f(y), y, 3) + D(f(x), x, 4)) == False
    assert F(D(f(x), x, 2) + D(f(x), x, 3)) == True
    
    # 调用 _ode_solver_test 函数来测试可降阶的高阶方程的求解
    _ode_solver_test(_get_examples_ode_sol_nth_order_reducible)


@slow
def test_separable():
    # 调用 _ode_solver_test 函数来测试可分离方程的求解
    _ode_solver_test(_get_examples_ode_sol_separable)


@slow
def test_factorable():
    # 此测试尚未完整添加，可能是为了测试因式化的方程解的情况
    pass
    # 确保对于给定的表达式，积分结果与其反正弦函数应用后的负积分相等
    assert integrate(-asin(f(2*x)+pi), x) == -Integral(asin(pi + f(2*x)), x)
    
    # 执行ODE求解器的测试函数，使用可因式分解的ODE示例进行测试
    _ode_solver_test(_get_examples_ode_sol_factorable)
# 标记一个慢速测试的装饰器，用于测试可分解的示例
@slow
# 测试慢速可分解的例子
def test_slow_examples_factorable():
    # 调用内部函数测试常微分方程求解，使用可分解ODE的解作为输入，执行慢速测试
    _ode_solver_test(_get_examples_ode_sol_factorable, run_slow_test=True)


# 测试 Riccati 特殊情况下的常微分方程求解
def test_Riccati_special_minus2():
    # 调用内部函数测试常微分方程求解，使用 Riccati 方程的特殊情况作为输入
    _ode_solver_test(_get_examples_ode_sol_riccati)


# 标记一个慢速测试的装饰器，测试第一类有理 Riccati 方程的常微分方程求解
@slow
def test_1st_rational_riccati():
    # 调用内部函数测试常微分方程求解，使用第一类有理 Riccati 方程的解作为输入，执行慢速测试
    _ode_solver_test(_get_examples_ode_sol_1st_rational_riccati)


# 测试 Bernoulli 方程的常微分方程求解
def test_Bernoulli():
    # 调用内部函数测试常微分方程求解，使用 Bernoulli 方程的解作为输入
    _ode_solver_test(_get_examples_ode_sol_bernoulli)


# 测试一阶线性方程的常微分方程求解
def test_1st_linear():
    # 调用内部函数测试常微分方程求解，使用一阶线性方程的解作为输入
    _ode_solver_test(_get_examples_ode_sol_1st_linear)


# 测试几乎线性方程的常微分方程求解
def test_almost_linear():
    # 调用内部函数测试常微分方程求解，使用几乎线性方程的解作为输入
    _ode_solver_test(_get_examples_ode_sol_almost_linear)


# 标记一个慢速测试的装饰器，测试 Liouville ODE 的常微分方程求解
@slow
def test_Liouville_ODE():
    # 指定 Liouville ODE 的提示信息
    hint = 'Liouville'
    # 使用 classify_ode 函数对两个不是 Liouville ODE 的方程进行分类
    not_Liouville1 = classify_ode(diff(f(x), x)/x + f(x)*diff(f(x), x, x)/2 -
        diff(f(x), x)**2/2, f(x))
    not_Liouville2 = classify_ode(diff(f(x), x)/x + diff(f(x), x, x)/2 -
        x*diff(f(x), x)**2/2, f(x))
    # 断言 Liouville 不在分类结果中
    assert hint not in not_Liouville1
    assert hint not in not_Liouville2
    assert hint + '_Integral' not in not_Liouville1
    assert hint + '_Integral' not in not_Liouville2

    # 调用内部函数测试常微分方程求解，使用 Liouville ODE 的解作为输入，执行慢速测试
    _ode_solver_test(_get_examples_ode_sol_liouville)


# 测试高阶线性 Euler 方程的齐次情况的常微分方程求解
def test_nth_order_linear_euler_eq_homogeneous():
    # 定义符号变量
    x, t, a, b, c = symbols('x t a b c')
    # 指定 Euler 方程的提示信息
    our_hint = "nth_linear_euler_eq_homogeneous"

    # 第一个方程的 Euler 形式
    eq = diff(f(t), t, 4)*t**4 - 13*diff(f(t), t, 2)*t**2 + 36*f(t)
    # 断言 Euler 提示信息在方程分类结果中
    assert our_hint in classify_ode(eq)

    # 第二个方程的 Euler 形式
    eq = a*y(t) + b*t*diff(y(t), t) + c*t**2*diff(y(t), t, 2)
    # 断言 Euler 提示信息在方程分类结果中
    assert our_hint in classify_ode(eq)

    # 调用内部函数测试常微分方程求解，使用 Euler 方程的齐次解作为输入
    _ode_solver_test(_get_examples_ode_sol_euler_homogeneous)


# 测试高阶线性 Euler 方程的非齐次情况，使用未定系数法的常微分方程求解
def test_nth_order_linear_euler_eq_nonhomogeneous_undetermined_coefficients():
    # 定义符号变量
    x, t = symbols('x t')
    a, b, c, d = symbols('a b c d', integer=True)
    # 指定 Euler 方程的提示信息
    our_hint = "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients"

    # 第一个方程的 Euler 形式
    eq = x**4*diff(f(x), x, 4) - 13*x**2*diff(f(x), x, 2) + 36*f(x) + x
    # 断言 Euler 提示信息在方程分类结果中
    assert our_hint in classify_ode(eq, f(x))

    # 第二个方程的 Euler 形式
    eq = a*x**2*diff(f(x), x, 2) + b*x*diff(f(x), x) + c*f(x) + d*log(x)
    # 断言 Euler 提示信息在方程分类结果中
    assert our_hint in classify_ode(eq, f(x))

    # 调用内部函数测试常微分方程求解，使用 Euler 方程的非齐次解和未定系数法作为输入
    _ode_solver_test(_get_examples_ode_sol_euler_undetermined_coeff)


# 标记一个慢速测试的装饰器，测试高阶线性 Euler 方程的非齐次情况，使用参数变化法的常微分方程求解
@slow
def test_nth_order_linear_euler_eq_nonhomogeneous_variation_of_parameters():
    # 定义符号变量
    x, t = symbols('x, t')
    a, b, c, d = symbols('a, b, c, d', integer=True)
    # 指定 Euler 方程的提示信息
    our_hint = "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters"

    # 第一个方程的 Euler 形式
    eq = Eq(x**2*diff(f(x),x,2) - 8*x*diff(f(x),x) + 12*f(x), x**2)
    # 断言 Euler 提示信息在方程分类结果中
    assert our_hint in classify_ode(eq, f(x))

    # 第二个方程的 Euler 形式
    eq = Eq(a*x**3*diff(f(x),x,3) + b*x**2*diff(f(x),x,2) + c*x*diff(f(x),x) + d*f(x), x*log(x))
    # 断言 Euler 提示信息在方程分类结果中
    assert our_hint in classify_ode(eq, f(x))

    # 调用内部函数测试常微分方程求解，使用 Euler 方程的非齐次解和参数变化法作为输入，执行慢速测试
    _ode_solver_test(_get_examples_ode_sol_euler_var_para)


# 为 Euler 齐次方程提供的示例解函数，添加了例子的键
@_add_example_keys
def _get_examples_ode_sol_euler_homogeneous():
    # 计算 Euler 方程的根并返回示例字典
    r1, r2, r3, r4, r5 = [rootof(x**5 - 14*x**4 + 71*x**3 - 154*x**2 + 120*x - 1, n) for n in range(5)]
    return {
            'hint': "nth_linear_euler_eq_homogeneous",
            'func': f(x),
            'examples':{
    'euler_hom_01': {
        'eq': Eq(-3*diff(f(x), x)*x + 2*x**2*diff(f(x), x, x), 0),
        'sol': [Eq(f(x), C1 + C2*x**Rational(5, 2))],
    },

    'euler_hom_02': {
        'eq': Eq(3*f(x) - 5*diff(f(x), x)*x + 2*x**2*diff(f(x), x, x), 0),
        'sol': [Eq(f(x), C1*sqrt(x) + C2*x**3)]
    },

    'euler_hom_03': {
        'eq': Eq(4*f(x) + 5*diff(f(x), x)*x + x**2*diff(f(x), x, x), 0),
        'sol': [Eq(f(x), (C1 + C2*log(x))/x**2)]
    },

    'euler_hom_04': {
        'eq': Eq(6*f(x) - 6*diff(f(x), x)*x + 1*x**2*diff(f(x), x, x) + x**3*diff(f(x), x, x, x), 0),
        'sol': [Eq(f(x), C1/x**2 + C2*x + C3*x**3)]
    },

    'euler_hom_05': {
        'eq': Eq(-125*f(x) + 61*diff(f(x), x)*x - 12*x**2*diff(f(x), x, x) + x**3*diff(f(x), x, x, x), 0),
        'sol': [Eq(f(x), x**5*(C1 + C2*log(x) + C3*log(x)**2))]
    },

    'euler_hom_06': {
        'eq': x**2*diff(f(x), x, 2) + x*diff(f(x), x) - 9*f(x),
        'sol': [Eq(f(x), C1*x**-3 + C2*x**3)]
    },

    'euler_hom_07': {
        'eq': sin(x)*x**2*f(x).diff(x, 2) + sin(x)*x*f(x).diff(x) + sin(x)*f(x),
        'sol': [Eq(f(x), C1*sin(log(x)) + C2*cos(log(x)))],
        'XFAIL': ['2nd_power_series_regular','nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients']
    },

    'euler_hom_08': {
        'eq': x**6 * f(x).diff(x, 6) - x*f(x).diff(x) + f(x),
        'sol': [Eq(f(x), C1*x + C2*x**r1 + C3*x**r2 + C4*x**r3 + C5*x**r4 + C6*x**r5)],
        'checkodesol_XFAIL':True
    },

    # This example is from issue: https://github.com/sympy/sympy/issues/15237
    'euler_hom_09': {
        'eq': Derivative(x*f(x), x, x, x),
        'sol': [Eq(f(x), C1 + C2/x + C3*x)],
    },
}
# 为函数添加示例键，用于ODE求解器的例子
@_add_example_keys
def _get_examples_ode_sol_euler_undetermined_coeff():
    # 返回包含提示、函数及示例字典的字典
    return {
        'hint': "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients",
        'func': f(x),
        'examples': {
            # 第一个例子：方程和解的表达式
            'euler_undet_01': {
                'eq': Eq(x**2*diff(f(x), x, x) + x*diff(f(x), x), 1),
                'sol': [Eq(f(x), C1 + C2*log(x) + log(x)**2/2)]
            },
            
            'euler_undet_02': {
                'eq': Eq(x**2*diff(f(x), x, x) - 2*x*diff(f(x), x) + 2*f(x), x**3),
                'sol': [Eq(f(x), x*(C1 + C2*x + Rational(1, 2)*x**2))]
            },
            
            'euler_undet_03': {
                'eq': Eq(x**2*diff(f(x), x, x) - x*diff(f(x), x) - 3*f(x), log(x)/x),
                'sol': [Eq(f(x), (C1 + C2*x**4 - log(x)**2/8 - log(x)/16)/x)]
            },
            
            'euler_undet_04': {
                'eq': Eq(x**2*diff(f(x), x, x) + 3*x*diff(f(x), x) - 8*f(x), log(x)**3 - log(x)),
                'sol': [Eq(f(x), C1/x**4 + C2*x**2 - Rational(1,8)*log(x)**3 - Rational(3,32)*log(x)**2 - Rational(1,64)*log(x) - Rational(7, 256))]
            },
            
            'euler_undet_05': {
                'eq': Eq(x**3*diff(f(x), x, x, x) - 3*x**2*diff(f(x), x, x) + 6*x*diff(f(x), x) - 6*f(x), log(x)),
                'sol': [Eq(f(x), C1*x + C2*x**2 + C3*x**3 - Rational(1, 6)*log(x) - Rational(11, 36))]
            },
            
            # 下面的例子是为了解决问题：https://github.com/sympy/sympy/issues/5096
            'euler_undet_06': {
                'eq': 2*x**2*f(x).diff(x, 2) + f(x) + sqrt(2*x)*sin(log(2*x)/2),
                'sol': [Eq(f(x), sqrt(x)*(C1*sin(log(x)/2) + C2*cos(log(x)/2) + sqrt(2)*log(x)*cos(log(2*x)/2)/2))]
            },
            
            'euler_undet_07': {
                'eq': 2*x**2*f(x).diff(x, 2) + f(x) + sin(log(2*x)/2),
                'sol': [Eq(f(x), C1*sqrt(x)*sin(log(x)/2) + C2*sqrt(x)*cos(log(x)/2) - 2*sin(log(2*x)/2)/5 - 4*cos(log(2*x)/2)/5)]
            },
        }
    }


# 为函数添加示例键，用于ODE求解器的例子
@_add_example_keys
def _get_examples_ode_sol_euler_var_para():
    # 返回包含提示、函数及示例字典的字典
    return {
        'hint': "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters",
        'func': f(x),
        'examples': {
            'euler_var_01': {
                'eq': Eq(x**2*Derivative(f(x), x, x) - 2*x*Derivative(f(x), x) + 2*f(x), x**4),
                'sol': [Eq(f(x), x*(C1 + C2*x + x**3/6))]
            },
            
            'euler_var_02': {
                'eq': Eq(3*x**2*diff(f(x), x, x) + 6*x*diff(f(x), x) - 6*f(x), x**3*exp(x)),
                'sol': [Eq(f(x), C1/x**2 + C2*x + x*exp(x)/3 - 4*exp(x)/3 + 8*exp(x)/(3*x) - 8*exp(x)/(3*x**2))]
            },
            
            'euler_var_03': {
                'eq': Eq(x**2*Derivative(f(x), x, x) - 2*x*Derivative(f(x), x) + 2*f(x), x**4*exp(x)),
                'sol':  [Eq(f(x), x*(C1 + C2*x + x*exp(x) - 2*exp(x)))]
            },
            
            'euler_var_04': {
                'eq': x**2*Derivative(f(x), x, x) - 2*x*Derivative(f(x), x) + 2*f(x) - log(x),
                'sol': [Eq(f(x), C1*x + C2*x**2 + log(x)/2 + Rational(3, 4))]
            },
            
            'euler_var_05': {
                'eq': -exp(x) + (x*Derivative(f(x), (x, 2)) + Derivative(f(x), x))/x,
                'sol': [Eq(f(x), C1 + C2*log(x) + exp(x) - Ei(x))]
            },
        }
    }
    'euler_var_06': {
        # 定义 Euler 变分方程的第六个变量，包含方程和解的信息
        'eq': x**2 * f(x).diff(x, 2) + x * f(x).diff(x) + 4 * f(x) - 1/x,
        # 方程的解列表，包含一个解表达式
        'sol': [Eq(f(x), C1*sin(2*log(x)) + C2*cos(2*log(x)) + 1/(5*x))]
    },
    }
@_add_example_keys
def _get_examples_ode_sol_bernoulli():
    # 定义函数 _get_examples_ode_sol_bernoulli，用于返回 Bernoulli 类型的ODE示例
    # Type: Bernoulli, f'(x) + p(x)*f(x) == q(x)*f(x)**n
    return {
            'hint': "Bernoulli",
            'func': f(x),
            'examples':{
    'bernoulli_01': {
        # Bernoulli ODE 示例 1
        'eq': Eq(x*f(x).diff(x) + f(x) - f(x)**2, 0),
        'sol': [Eq(f(x), 1/(C1*x + 1))],
        'XFAIL': ['separable_reduced']
    },

    'bernoulli_02': {
        # Bernoulli ODE 示例 2
        'eq': f(x).diff(x) - y*f(x),
        'sol': [Eq(f(x), C1*exp(x*y))]
    },

    'bernoulli_03': {
        # Bernoulli ODE 示例 3
        'eq': f(x)*f(x).diff(x) - 1,
        'sol': [Eq(f(x), -sqrt(C1 + 2*x)), Eq(f(x), sqrt(C1 + 2*x))]
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_riccati():
    # 定义函数 _get_examples_ode_sol_riccati，用于返回 Riccati 特定类型的ODE示例
    # Type: Riccati special alpha = -2, a*dy/dx + b*y**2 + c*y/x +d/x**2
    return {
            'hint': "Riccati_special_minus2",
            'func': f(x),
            'examples':{
    'riccati_01': {
        # Riccati ODE 特例 1
        'eq': 2*f(x).diff(x) + f(x)**2 - f(x)/x + 3*x**(-2),
        'sol': [Eq(f(x), (-sqrt(3)*tan(C1 + sqrt(3)*log(x)/4) + 3)/(2*x))],
    },
    },
    }


@_add_example_keys
def _get_examples_ode_sol_1st_rational_riccati():
    # 定义函数 _get_examples_ode_sol_1st_rational_riccati，用于返回一阶有理 Riccati 类型的ODE示例
    # Type: 1st Order Rational Riccati, dy/dx = a + b*y + c*y**2,
    # a, b, c are rational functions of x
    return {
            'hint': "1st_rational_riccati",
            'func': f(x),
            'examples':{
    # a(x) is a constant
    "rational_riccati_01": {
        # 有理 Riccati ODE 示例 1
        "eq": Eq(f(x).diff(x) + f(x)**2 - 2, 0),
        "sol": [Eq(f(x), sqrt(2)*(-C1 - exp(2*sqrt(2)*x))/(C1 - exp(2*sqrt(2)*x)))]
    },
    # a(x) is a constant
    "rational_riccati_02": {
        # 有理 Riccati ODE 示例 2
        "eq": f(x)**2 + Derivative(f(x), x) + 4*f(x)/x + 2/x**2,
        "sol": [Eq(f(x), (-2*C1 - x)/(x*(C1 + x)))]
    },
    # a(x) is a constant
    "rational_riccati_03": {
        # 有理 Riccati ODE 示例 3
        "eq": 2*x**2*Derivative(f(x), x) - x*(4*f(x) + Derivative(f(x), x) - 4) + (f(x) - 1)*f(x),
        "sol": [Eq(f(x), (C1 + 2*x**2)/(C1 + x))]
    },
    # Constant coefficients
    "rational_riccati_04": {
        # 有理 Riccati ODE 示例 4
        "eq": f(x).diff(x) - 6 - 5*f(x) - f(x)**2,
        "sol": [Eq(f(x), (-2*C1 + 3*exp(x))/(C1 - exp(x)))]
    },
    # One pole of multiplicity 2
    "rational_riccati_05": {
        # 有理 Riccati ODE 示例 5
        "eq": x**2 - (2*x + 1/x)*f(x) + f(x)**2 + Derivative(f(x), x),
        "sol": [Eq(f(x), x*(C1 + x**2 + 1)/(C1 + x**2 - 1))]
    },
    # One pole of multiplicity 2
    "rational_riccati_06": {
        # 有理 Riccati ODE 示例 6
        "eq": x**4*Derivative(f(x), x) + x**2 - x*(2*f(x)**2 + Derivative(f(x), x)) + f(x),
        "sol": [Eq(f(x), x*(C1*x - x + 1)/(C1 + x**2 - 1))]
    },
    # Multiple poles of multiplicity 2
    "rational_riccati_07": {
        # 有理 Riccati ODE 示例 7
        "eq": -f(x)**2 + Derivative(f(x), x) + (15*x**2 - 20*x + 7)/((x - 1)**2*(2*x \
            - 1)**2),
        "sol": [Eq(f(x), (9*C1*x - 6*C1 - 15*x**5 + 60*x**4 - 94*x**3 + 72*x**2 - \
            33*x + 8)/(6*C1*x**2 - 9*C1*x + 3*C1 + 6*x**6 - 29*x**5 + 57*x**4 - \
            58*x**3 + 28*x**2 - 3*x - 1))]
    },
    # Imaginary poles
    }
    }
    # 定义名为 "rational_riccati_08" 的方程和解的字典条目
    "rational_riccati_08": {
        # 方程：f(x) 的导数 + (3*x**2 + 1)*f(x)**2/x + (6*x**2 - x + 3)*f(x)/(x*(x - 1)) + (3*x**2 - 2*x + 2)/(x*(x - 1)**2)
        "eq": Derivative(f(x), x) + (3*x**2 + 1)*f(x)**2/x + (6*x**2 - x + 3)*f(x)/(x*(x \
            - 1)) + (3*x**2 - 2*x + 2)/(x*(x - 1)**2),
        # 解：f(x) = (-C1 - x**3 + x**2 - 2*x + 1)/(C1*x - C1 + x**4 - x**3 + x**2 - 2*x + 1)
        "sol": [Eq(f(x), (-C1 - x**3 + x**2 - 2*x + 1)/(C1*x - C1 + x**4 - x**3 + x**2 - \
            2*x + 1))],
    },
    # 定义名为 "rational_riccati_09" 的方程和解的字典条目
    "rational_riccati_09": {
        # 方程：f(x) 的导数 - 2*I*(f(x)**2 + 1)/x
        "eq": Derivative(f(x), x) - 2*I*(f(x)**2 + 1)/x,
        # 解：f(x) = (-I*C1 + I*x**4 + I)/(C1 + x**4 - 1)
        "sol": [Eq(f(x), (-I*C1 + I*x**4 + I)/(C1 + x**4 - 1))]
    },
    # 定义名为 "rational_riccati_10" 的方程和解的字典条目
    "rational_riccati_10": {
        # 方程：f(x) 的导数等于 x*f(x)/(S(3)/2 - 2*x) + (x/2 - S(1)/3)*f(x)**2/(2*x/3 - S(1)/2) - S(5)/4 + (281*x**2 - 1260*x + 756)/(16*x**3 - 12*x**2)
        "eq": Eq(Derivative(f(x), x), x*f(x)/(S(3)/2 - 2*x) + (x/2 - S(1)/3)*f(x)**2/\
            (2*x/3 - S(1)/2) - S(5)/4 + (281*x**2 - 1260*x + 756)/(16*x**3 - 12*x**2)),
        # 解：f(x) = (40*C1*x**14 + 28*C1*x**13 + 420*C1*x**12 + 2940*C1*x**11 + \
        #        18480*C1*x**10 + 103950*C1*x**9 + 519750*C1*x**8 + 2286900*C1*x**7 + \
        #        8731800*C1*x**6 + 28378350*C1*x**5 + 76403250*C1*x**4 + 163721250*C1*x**3 \
        #        + 261954000*C1*x**2 + 278326125*C1*x + 147349125*C1 + x*exp(2*x) - 9*exp(2*x) \
        #        )/(x*(24*C1*x**13 + 140*C1*x**12 + 840*C1*x**11 + 4620*C1*x**10 + 23100*C1*x**9 \
        #        + 103950*C1*x**8 + 415800*C1*x**7 + 1455300*C1*x**6 + 4365900*C1*x**5 + \
        #        10914750*C1*x**4 + 21829500*C1*x**3 + 32744250*C1*x**2 + 32744250*C1*x + \
        #        16372125*C1 - exp(2*x)))
        "sol": [Eq(f(x), (40*C1*x**14 + 28*C1*x**13 + 420*C1*x**12 + 2940*C1*x**11 + \
            18480*C1*x**10 + 103950*C1*x**9 + 519750*C1*x**8 + 2286900*C1*x**7 + \
            8731800*C1*x**6 + 28378350*C1*x**5 + 76403250*C1*x**4 + 163721250*C1*x**3 \
            + 261954000*C1*x**2 + 278326125*C1*x + 147349125*C1 + x*exp(2*x) - 9*exp(2*x) \
            )/(x*(24*C1*x**13 + 140*C1*x**12 + 840*C1*x**11 + 4620*C1*x**10 + 23100*C1*x**9 \
            + 103950*C1*x**8 + 415800*C1*x**7 + 1455300*C1*x**6 + 4365900*C1*x**5 + \
            10914750*C1*x**4 + 21829500*C1*x**3 + 32744250*C1*x**2 + 32744250*C1*x + \
            16372125*C1 - exp(2*x))))]
    }
@_add_example_keys
# 定义一个函数用于获取一阶线性ODE的解例子
def _get_examples_ode_sol_1st_linear():
    # 返回一个字典，包含一阶线性ODE解的示例
    return {
            'hint': "1st_linear",  # 提示信息，指出这是一阶线性ODE
            'func': f(x),  # 待求解函数 f(x)
            'examples':{
    'linear_01': {
        'eq': Eq(f(x).diff(x) + x*f(x), x**2),  # 方程示例 f'(x) + x*f(x) = x^2
        'sol': [Eq(f(x), (C1 + x*exp(x**2/2)- sqrt(2)*sqrt(pi)*erfi(sqrt(2)*x/2)/2)*exp(-x**2/2))],  # 解
    },
    },
    }


@_add_example_keys
# 定义一个函数用于获取可分解ODE的解例子
def _get_examples_ode_sol_factorable():
    """ some hints are marked as xfail for examples because they missed additional algebraic solution
    which could be found by Factorable hint. Fact_01 raise exception for
    nth_linear_constant_coeff_undetermined_coefficients"""
    
    y = Dummy('y')  # 创建一个虚拟变量 y
    a0,a1,a2,a3,a4 = symbols('a0, a1, a2, a3, a4')  # 定义符号变量 a0, a1, a2, a3, a4
    # 返回一个字典，包含可分解ODE解的示例
    return {
            'hint': "factorable",  # 提示信息，指出这是可分解ODE
            'func': f(x),  # 待求解函数 f(x)
            'examples':{
    'fact_01': {
        'eq': f(x) + f(x)*f(x).diff(x),  # 方程示例 f(x) + f(x)*f'(x)
        'sol': [Eq(f(x), 0), Eq(f(x), C1 - x)],  # 解
        'XFAIL': ['separable', '1st_exact', '1st_linear', 'Bernoulli', '1st_homogeneous_coeff_best',
        '1st_homogeneous_coeff_subs_indep_div_dep', '1st_homogeneous_coeff_subs_dep_div_indep',
        'lie_group', 'nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients',
        'nth_linear_constant_coeff_variation_of_parameters',
        'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters',
        'nth_linear_constant_coeff_undetermined_coefficients']  # 标记解的XFAIL，即不能通过这些提示得到解
    },

    'fact_02': {
        'eq': f(x)*(f(x).diff(x)+f(x)*x+2),  # 方程示例 f(x)*(f'(x) + f(x)*x + 2)
        'sol': [Eq(f(x), (C1 - sqrt(2)*sqrt(pi)*erfi(sqrt(2)*x/2))*exp(-x**2/2)), Eq(f(x), 0)],  # 解
        'XFAIL': ['Bernoulli', '1st_linear', 'lie_group']  # 标记解的XFAIL，即不能通过这些提示得到解
    },

    'fact_03': {
        'eq': (f(x).diff(x)+f(x)*x**2)*(f(x).diff(x, 2) + x*f(x)),  # 方程示例 (f'(x) + f(x)*x^2)*(f''(x) + x*f(x))
        'sol':  [Eq(f(x), C1*airyai(-x) + C2*airybi(-x)),Eq(f(x), C1*exp(-x**3/3))]  # 解
    },

    'fact_04': {
        'eq': (f(x).diff(x)+f(x)*x**2)*(f(x).diff(x, 2) + f(x)),  # 方程示例 (f'(x) + f(x)*x^2)*(f''(x) + f(x))
        'sol': [Eq(f(x), C1*exp(-x**3/3)), Eq(f(x), C1*sin(x) + C2*cos(x))]  # 解
    },

    'fact_05': {
        'eq': (f(x).diff(x)**2-1)*(f(x).diff(x)**2-4),  # 方程示例 (f'(x)^2 - 1)*(f'(x)^2 - 4)
        'sol': [Eq(f(x), C1 - x), Eq(f(x), C1 + x), Eq(f(x), C1 + 2*x), Eq(f(x), C1 - 2*x)]  # 解
    },

    'fact_06': {
        'eq': (f(x).diff(x, 2)-exp(f(x)))*f(x).diff(x),  # 方程示例 (f''(x) - exp(f(x)))*f'(x)
        'sol': [
            Eq(f(x), log(-C1/(cos(sqrt(-C1)*(C2 + x)) + 1))),  # 解
            Eq(f(x), log(-C1/(cos(sqrt(-C1)*(C2 - x)) + 1))),  # 解
            Eq(f(x), C1)  # 解
        ],
        'slow': True,  # 标记这个示例比较慢
    },

    'fact_07': {
        'eq': (f(x).diff(x)**2-1)*(f(x)*f(x).diff(x)-1),  # 方程示例 (f'(x)^2 - 1)*(f(x)*f'(x) - 1)
        'sol': [Eq(f(x), C1 - x), Eq(f(x), -sqrt(C1 + 2*x)),Eq(f(x), sqrt(C1 + 2*x)), Eq(f(x), C1 + x)]  # 解
    },

    'fact_08': {
        'eq': Derivative(f(x), x)**4 - 2*Derivative(f(x), x)**2 + 1,  # 方程示例 f'(x)^4 - 2*f'(x)^2 + 1
        'sol': [Eq(f(x), C1 - x), Eq(f(x), C1 + x)]  # 解
    },
    # 定义名为 'fact_09' 的字典，包含方程和解的信息
    'fact_09': {
        # 方程：f(x)**2*Derivative(f(x), x)**6 - 2*f(x)**2*Derivative(f(x), x)**4 + ...
        'eq': f(x)**2*Derivative(f(x), x)**6 - 2*f(x)**2*Derivative(f(x), x)**4 + ...
             # 这里由于长表达式，换行继续
             f(x)**2*Derivative(f(x), x)**2 - 2*f(x)*Derivative(f(x), x)**5 + ...
             # 继续换行
             4*f(x)*Derivative(f(x), x)**3 - 2*f(x)*Derivative(f(x), x) + ...
             # 继续换行
             Derivative(f(x), x)**4 - 2*Derivative(f(x), x)**2 + 1,

        # 解列表
        'sol': [
            Eq(f(x), C1 - x),  # f(x) = C1 - x
            Eq(f(x), -sqrt(C1 + 2*x)),  # f(x) = -sqrt(C1 + 2*x)
            Eq(f(x), sqrt(C1 + 2*x)),   # f(x) = sqrt(C1 + 2*x)
            Eq(f(x), C1 + x)           # f(x) = C1 + x
        ]
    },

    # 定义名为 'fact_10' 的字典，包含方程、解和一个额外的标记 'slow'
    'fact_10': {
        # 方程：x**4*f(x)**2 + 2*x**4*f(x)*Derivative(f(x), (x, 2)) + ...
        'eq': x**4*f(x)**2 + 2*x**4*f(x)*Derivative(f(x), (x, 2)) + ...
             # 继续换行
             x**4*Derivative(f(x), (x, 2))**2 + 2*x**3*f(x)*Derivative(f(x), x) + ...
             # 继续换行
             2*x**3*Derivative(f(x), x)*Derivative(f(x), (x, 2)) - 7*x**2*f(x)**2 + ...
             # 继续换行
             x**2*Derivative(f(x), x)**2 - 7*x*f(x)*Derivative(f(x), x) + 12*f(x)**2,

        # 解列表
        'sol': [
            Eq(f(x), C1*besselj(2, x) + C2*bessely(2, x)),  # f(x) = C1*besselj(2, x) + C2*bessely(2, x)
            Eq(f(x), C1*besselj(sqrt(3), x) + C2*bessely(sqrt(3), x))  # f(x) = C1*besselj(sqrt(3), x) + C2*bessely(sqrt(3), x)
        ],

        # 标记，表示这个问题处理速度较慢
        'slow': True,
    },

    # 定义名为 'fact_11' 的字典，包含方程、解和一个标记 'dsolve_too_slow'
    'fact_11': {
        # 方程：(f(x).diff(x, 2)-exp(f(x)))*(f(x).diff(x, 2)+exp(f(x)))
        'eq': (f(x).diff(x, 2)-exp(f(x)))*(f(x).diff(x, 2)+exp(f(x))),

        # 解列表
        'sol': [
            Eq(f(x), log(C1/(cos(C1*sqrt(-1/C1)*(C2 + x)) - 1))),  # f(x) = log(C1/(cos(C1*sqrt(-1/C1)*(C2 + x)) - 1))
            Eq(f(x), log(C1/(cos(C1*sqrt(-1/C1)*(C2 - x)) - 1))),  # f(x) = log(C1/(cos(C1*sqrt(-1/C1)*(C2 - x)) - 1))
            Eq(f(x), log(C1/(1 - cos(C1*sqrt(-1/C1)*(C2 + x))))),  # f(x) = log(C1/(1 - cos(C1*sqrt(-1/C1)*(C2 + x))))
            Eq(f(x), log(C1/(1 - cos(C1*sqrt(-1/C1)*(C2 - x)))))   # f(x) = log(C1/(1 - cos(C1*sqrt(-1/C1)*(C2 - x))))
        ],

        # 标记，表示求解速度过慢
        'dsolve_too_slow': True,
    },

    # 定义名为 'fact_12' 的字典，包含方程、解和一个标记 'XFAIL'
    'fact_12': {
        # 方程：exp(f(x).diff(x))-f(x)**2
        'eq': exp(f(x).diff(x))-f(x)**2,

        # 解列表
        'sol': [Eq(NonElementaryIntegral(1/log(y**2), (y, f(x))), C1 + x)],

        # 标记，表示在 'lie_group' 问题上未实现
        'XFAIL': ['lie_group']  # It shows not implemented error for lie_group.
    },

    # 定义名为 'fact_13' 的字典，包含方程、解和一个标记 'XFAIL'
    'fact_13': {
        # 方程：f(x).diff(x)**2 - f(x)**3
        'eq': f(x).diff(x)**2 - f(x)**3,

        # 解列表
        'sol': [Eq(f(x), 4/(C1**2 - 2*C1*x + x**2))],

        # 标记，表示在 'lie_group' 问题上未实现
        'XFAIL': ['lie_group']  # It shows not implemented error for lie_group.
    },

    # 定义名为 'fact_14' 的字典，包含方程和解
    'fact_14': {
        # 方程：f(x).diff(x)**2 - f(x)
        'eq': f(x).diff(x)**2 - f(x),

        # 解列表
        'sol': [Eq(f(x), C1**2/4 - C1*x/2 + x**2/4)]
    },

    # 定义名为 'fact_15' 的字典，包含方程和解
    'fact_15': {
        # 方程：f(x).diff(x)**2 - f(x)**2
        'eq': f(x).diff(x)**2 - f(x)**2,

        # 解列表
        'sol': [Eq(f(x), C1*exp(x)), Eq(f(x), C1*exp(-x))]
    },

    # 定义名为 'fact_16' 的字典，包含方程和解
    'fact_16': {
        # 方程：f(x).diff(x)**2 - f(x)**3
        'eq': f(x).diff(x)**2 - f(x)**3,

        # 解列表
        'sol': [Eq(f(x), 4/(C1**2 - 2*C1*x + x**2))],
    },

    # 定义名为 'fact_17' 的字典，包含方程、解和一个标记 'slow'
    'fact_17': {
        # 方程：f(x).diff(x)-(a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0)**(-1/2)
        'eq': f(x).diff(x)-(a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0)**(-1/2),

        # 解列表
        'sol': [Eq(f(x), C1 + Integral(1/sqrt(a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4), x))],

        # 标记，表示这个问题处理速度较慢
        'slow': True
    },

    # 定义名为 'fact_18' 的字典，包含方程、解和一个标记 'checkodesol_XFAIL'
    'fact_18': {
        # 方程：Eq(f(2 * x), sin(Derivative(f(x))))
        'eq': Eq(f(2 * x), sin(Derivative(f(x)))),

        # 解列表
        'sol': [Eq(f(x), C1 + Integral(pi - asin(f(2*x)), x)),
                Eq(f(x), C1 + Integral(asin(f(2*x)), x))],

        # 标记，表示检查 'odesol' 时会失败
        'checkodesol_XFAIL': True
    },
    'fact_19': {
        'eq': Derivative(f(x), x)**2 - x**3,
        'sol': [Eq(f(x), C1 - 2*x**Rational(5,2)/5), Eq(f(x), C1 + 2*x**Rational(5,2)/5)],
    },

    'fact_20': {
        'eq': x*f(x).diff(x, 2) - x*f(x),
        'sol': [Eq(f(x), C1*exp(-x) + C2*exp(x))],
    },
    }



# 表示“fact_19”的方程和解集合
'fact_19': {
    # 方程为二阶导数平方减去 x 的立方
    'eq': Derivative(f(x), x)**2 - x**3,
    # 解集合包含两个方程：f(x) = C1 - 2*x**(5/2)/5 和 f(x) = C1 + 2*x**(5/2)/5
    'sol': [Eq(f(x), C1 - 2*x**Rational(5,2)/5), Eq(f(x), C1 + 2*x**Rational(5,2)/5)],
},

# 表示“fact_20”的方程和解集合
'fact_20': {
    # 方程为 x 乘以 f(x) 二阶导数减去 x 乘以 f(x)
    'eq': x*f(x).diff(x, 2) - x*f(x),
    # 解集合包含一个方程：f(x) = C1*exp(-x) + C2*exp(x)
    'sol': [Eq(f(x), C1*exp(-x) + C2*exp(x))],
},
# 添加示例关键字到函数 `_get_examples_ode_sol_almost_linear` 的装饰器
@_add_example_keys
# 定义一个函数 `_get_examples_ode_sol_almost_linear`，返回一个包含示例的字典
def _get_examples_ode_sol_almost_linear():
    # 从 sympy 函数库中导入特殊误差函数 Ei
    from sympy.functions.special.error_functions import Ei
    # 定义一个正数符号 A
    A = Symbol('A', positive=True)
    # 定义一个未知函数 f
    f = Function('f')
    # 对 f(x) 求导数，并赋值给 d
    d = f(x).diff(x)

    # 返回一个包含多个示例的字典
    return {
        'hint': "almost_linear",  # 提示信息指示几乎线性的特性
        'func': f(x),  # 函数表达式 f(x)
        'examples':{
            'almost_lin_01': {  # 第一个几乎线性示例
                'eq': x**2*f(x)**2*d + f(x)**3 + 1,  # 方程
                'sol': [  # 解的列表
                    Eq(f(x), (C1*exp(3/x) - 1)**Rational(1, 3)),  # 解1
                    Eq(f(x), (-1 - sqrt(3)*I)*(C1*exp(3/x) - 1)**Rational(1, 3)/2),  # 解2
                    Eq(f(x), (-1 + sqrt(3)*I)*(C1*exp(3/x) - 1)**Rational(1, 3)/2)  # 解3
                ],
            },

            'almost_lin_02': {  # 第二个几乎线性示例
                'eq': x*f(x)*d + 2*x*f(x)**2 + 1,  # 方程
                'sol': [  # 解的列表
                    Eq(f(x), -sqrt((C1 - 2*Ei(4*x))*exp(-4*x))),  # 解1
                    Eq(f(x), sqrt((C1 - 2*Ei(4*x))*exp(-4*x)))  # 解2
                ]
            },

            'almost_lin_03': {  # 第三个几乎线性示例
                'eq': x*d + x*f(x) + 1,  # 方程
                'sol': [  # 解的列表
                    Eq(f(x), (C1 - Ei(x))*exp(-x))  # 解1
                ]
            },

            'almost_lin_04': {  # 第四个几乎线性示例
                'eq': x*exp(f(x))*d + exp(f(x)) + 3*x,  # 方程
                'sol': [  # 解的列表
                    Eq(f(x), log(C1/x - x*Rational(3, 2)))  # 解1
                ],
            },

            'almost_lin_05': {  # 第五个几乎线性示例
                'eq': x + A*(x + diff(f(x), x) + f(x)) + diff(f(x), x) + f(x) + 2,  # 方程
                'sol': [  # 解的列表
                    Eq(f(x), (C1 + Piecewise(  # 解1
                        (x, Eq(A + 1, 0)),
                        ((-A*x + A - x - 1)*exp(x)/(A + 1), True)
                    ))*exp(-x))
                ],
            },
        }
    }
    # 返回一个包含多个代数方程示例的字典
    return {
        'hint': "nth_algebraic",  # 提示信息，指出这些例子属于“nth_algebraic”类型
        'func': f(x),  # 使用函数 f(x)
        'examples':{  # 代数方程示例开始
            'algeb_01': {
                'eq': f(x) * f(x).diff(x) * f(x).diff(x, x) * (f(x) - 1) * (f(x).diff(x) - x),
                'sol': [Eq(f(x), C1 + x**2/2), Eq(f(x), C1 + C2*x)]
            },

            'algeb_02': {
                'eq': f(x) * f(x).diff(x) * f(x).diff(x, x) * (f(x) - 1),
                'sol': [Eq(f(x), C1 + C2*x)]
            },

            'algeb_03': {
                'eq': f(x) * f(x).diff(x) * f(x).diff(x, x),
                'sol': [Eq(f(x), C1 + C2*x)]
            },

            'algeb_04': {
                'eq': Eq(-M * phi(t).diff(t),
                         Rational(3, 2) * m * r**2 * phi(t).diff(t) * phi(t).diff(t,t)),
                'sol': [Eq(phi(t), C1), Eq(phi(t), C1 + C2*t - M*t**2/(3*m*r**2))],
                'func': phi(t)
            },

            'algeb_05': {
                'eq': (1 - sin(f(x))) * f(x).diff(x),
                'sol': [Eq(f(x), C1)],
                'XFAIL': ['separable']  # 如果出现异常，会打印出来
            },

            'algeb_06': {
                'eq': (diff(f(x)) - x)*(diff(f(x)) + x),
                'sol': [Eq(f(x), C1 - x**2/2), Eq(f(x), C1 + x**2/2)]
            },

            'algeb_07': {
                'eq': Eq(Derivative(f(x), x), Derivative(g(x), x)),
                'sol': [Eq(f(x), C1 + g(x))],
            },

            'algeb_08': {
                'eq': f(x).diff(x) - C1,   # 这个例子来自问题 15999
                'sol': [Eq(f(x), C1*x + C2)],
            },

            'algeb_09': {
                'eq': f(x)*f(x).diff(x),
                'sol': [Eq(f(x), C1)],
            },

            'algeb_10': {
                'eq': (diff(f(x)) - x)*(diff(f(x)) + x),
                'sol': [Eq(f(x), C1 - x**2/2), Eq(f(x), C1 + x**2/2)],
            },

            'algeb_11': {
                'eq': f(x) + f(x)*f(x).diff(x),
                'sol': [Eq(f(x), 0), Eq(f(x), C1 - x)],
                'XFAIL': ['separable', '1st_exact', '1st_linear', 'Bernoulli', '1st_homogeneous_coeff_best',
                          '1st_homogeneous_coeff_subs_indep_div_dep', '1st_homogeneous_coeff_subs_dep_div_indep',
                          'lie_group', 'nth_linear_constant_coeff_undetermined_coefficients',
                          'nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients',
                          'nth_linear_constant_coeff_variation_of_parameters',
                          'nth_linear_euler_eq_nonhomogeneous_variation_of_parameters']
                          # 当出现 nth_linear_constant_coeff_undetermined_coefficients 异常时，其他所有情况都不会显示出来
            },

            'algeb_12': {
                'eq': Derivative(x*f(x), x, x, x),
                'sol': [Eq(f(x), (C1 + C2*x + C3*x**2) / x)],
                'XFAIL': ['nth_algebraic']  # 只有在 dsolve 设置 prep=False 时才会通过
            },

            'algeb_13': {
                'eq': Eq(Derivative(x*Derivative(f(x), x), x)/x, exp(x)),
                'sol': [Eq(f(x), C1 + C2*log(x) + exp(x) - Ei(x))],
                'XFAIL': ['nth_algebraic']  # 只有在 dsolve 设置 prep=False 时才会通过
            },
        }  # 代数方程示例结束
    }
    'algeb_14': {
        'eq': Eq(f(x).diff(x), 0),  # 定义微分方程 f'(x) = 0
        'sol': [Eq(f(x), C1)],  # 解为 f(x) = C1
    },

    'algeb_15': {
        'eq': Eq(3*f(x).diff(x) - 5, 0),  # 定义微分方程 3*f'(x) - 5 = 0
        'sol': [Eq(f(x), C1 + x*Rational(5, 3))],  # 解为 f(x) = C1 + 5*x/3
    },

    'algeb_16': {
        'eq': Eq(3*f(x).diff(x), 5),  # 定义微分方程 3*f'(x) = 5
        'sol': [Eq(f(x), C1 + x*Rational(5, 3))],  # 解为 f(x) = C1 + 5*x/3
    },

    # Type: 2nd order, constant coefficients (two complex roots)
    'algeb_17': {
        'eq': Eq(3*f(x).diff(x) - 1, 0),  # 定义微分方程 3*f'(x) - 1 = 0
        'sol': [Eq(f(x), C1 + x/3)],  # 解为 f(x) = C1 + x/3
    },

    'algeb_18': {
        'eq': Eq(x*f(x).diff(x) - 1, 0),  # 定义微分方程 x*f'(x) - 1 = 0
        'sol': [Eq(f(x), C1 + log(x))],  # 解为 f(x) = C1 + log(x)
    },

    # https://github.com/sympy/sympy/issues/6989
    'algeb_19': {
        'eq': f(x).diff(x) - x*exp(-k*x),  # 定义微分方程 f'(x) - x*exp(-k*x) = 0
        'sol': [Eq(f(x), C1 + Piecewise(((-k*x - 1)*exp(-k*x)/k**2, Ne(k**2, 0)),(x**2/2, True)))],  # 解为复杂的分段函数
    },

    'algeb_20': {
        'eq': -f(x).diff(x) + x*exp(-k*x),  # 定义微分方程 -f'(x) + x*exp(-k*x) = 0
        'sol': [Eq(f(x), C1 + Piecewise(((-k*x - 1)*exp(-k*x)/k**2, Ne(k**2, 0)),(x**2/2, True)))],  # 解为复杂的分段函数
    },

    # https://github.com/sympy/sympy/issues/10867
    'algeb_21': {
        'eq': Eq(g(x).diff(x).diff(x), (x-2)**2 + (x-3)**3),  # 定义二阶微分方程 g''(x) = (x-2)^2 + (x-3)^3
        'sol': [Eq(g(x), C1 + C2*x + x**5/20 - 2*x**4/3 + 23*x**3/6 - 23*x**2/2)],  # 解为 g(x) 的复杂多项式
        'func': g(x),  # 函数 g(x)
    },

    # https://github.com/sympy/sympy/issues/13691
    'algeb_22': {
        'eq': f(x).diff(x) - C1*g(x).diff(x),  # 定义微分方程 f'(x) - C1*g'(x) = 0
        'sol': [Eq(f(x), C2 + C1*g(x))],  # 解为 f(x) = C2 + C1*g(x)
        'func': f(x),  # 函数 f(x)
    },

    # https://github.com/sympy/sympy/issues/4838
    'algeb_23': {
        'eq': f(x).diff(x) - 3*C1 - 3*x**2,  # 定义微分方程 f'(x) - 3*C1 - 3*x**2 = 0
        'sol': [Eq(f(x), C2 + 3*C1*x + x**3)],  # 解为 f(x) = C2 + 3*C1*x + x**3
    },
}
# 添加示例键的装饰器，用于函数 _get_examples_ode_sol_nth_order_reducible
@_add_example_keys
# 定义函数，返回包含示例的字典，用于求解可降阶的高阶常微分方程
def _get_examples_ode_sol_nth_order_reducible():
    # 返回包含多个示例的字典
    return {
        'hint': "nth_order_reducible",  # 提示信息，指出这些示例是关于可降阶的方程
        'func': f(x),  # 函数 f(x) 的表达式
        'examples': {
            'reducible_01': {
                'eq': Eq(x*Derivative(f(x), x)**2 + Derivative(f(x), x, 2), 0),
                'sol': [Eq(f(x),C1 - sqrt(-1/C2)*log(-C2*sqrt(-1/C2) + x) +
                sqrt(-1/C2)*log(C2*sqrt(-1/C2) + x))],
                'slow': True,  # 解决此方程的速度较慢
            },
            'reducible_02': {
                'eq': -exp(x) + (x*Derivative(f(x), (x, 2)) + Derivative(f(x), x))/x,
                'sol': [Eq(f(x), C1 + C2*log(x) + exp(x) - Ei(x))],
                'slow': True,  # 解决此方程的速度较慢
            },
            'reducible_03': {
                'eq': Eq(sqrt(2) * f(x).diff(x,x,x) + f(x).diff(x), 0),
                'sol': [Eq(f(x), C1 + C2*sin(2**Rational(3, 4)*x/2) + C3*cos(2**Rational(3, 4)*x/2))],
                'slow': True,  # 解决此方程的速度较慢
            },
            'reducible_04': {
                'eq': f(x).diff(x, 2) + 2*f(x).diff(x),
                'sol': [Eq(f(x), C1 + C2*exp(-2*x))],
            },
            'reducible_05': {
                'eq': f(x).diff(x, 3) + f(x).diff(x, 2) - 6*f(x).diff(x),
                'sol': [Eq(f(x), C1 + C2*exp(-3*x) + C3*exp(2*x))],
                'slow': True,  # 解决此方程的速度较慢
            },
            'reducible_06': {
                'eq': f(x).diff(x, 4) - f(x).diff(x, 3) - 4*f(x).diff(x, 2) + \
                4*f(x).diff(x),
                'sol': [Eq(f(x), C1 + C2*exp(-2*x) + C3*exp(x) + C4*exp(2*x))],
                'slow': True,  # 解决此方程的速度较慢
            },
            'reducible_07': {
                'eq': f(x).diff(x, 4) + 3*f(x).diff(x, 3),
                'sol': [Eq(f(x), C1 + C2*x + C3*x**2 + C4*exp(-3*x))],
                'slow': True,  # 解决此方程的速度较慢
            },
            'reducible_08': {
                'eq': f(x).diff(x, 4) - 2*f(x).diff(x, 2),
                'sol': [Eq(f(x), C1 + C2*x + C3*exp(-sqrt(2)*x) + C4*exp(sqrt(2)*x))],
                'slow': True,  # 解决此方程的速度较慢
            },
            'reducible_09': {
                'eq': f(x).diff(x, 4) + 4*f(x).diff(x, 2),
                'sol': [Eq(f(x), C1 + C2*x + C3*sin(2*x) + C4*cos(2*x))],
                'slow': True,  # 解决此方程的速度较慢
            },
            'reducible_10': {
                'eq': f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x),
                'sol': [Eq(f(x), C1 + C2*x*sin(x) + C2*cos(x) - C3*x*cos(x) + C3*sin(x) + C4*sin(x) + C5*cos(x))],
                'slow': True,  # 解决此方程的速度较慢
            },
            'reducible_11': {
                'eq': f(x).diff(x, 2) - f(x).diff(x)**3,
                'sol': [Eq(f(x), C1 - sqrt(2)*sqrt(-1/(C2 + x))*(C2 + x)),
                Eq(f(x), C1 + sqrt(2)*sqrt(-1/(C2 + x))*(C2 + x))],
                'slow': True,  # 解决此方程的速度较慢
            },
            'reducible_12': {
                'eq': Derivative(x*f(x), x, x, x) + Derivative(f(x), x, x, x),
                'sol': [Eq(f(x), C1 + C3/Mul(2, (x**2 + 2*x + 1), evaluate=False) +
                x*(C2 + C3/Mul(2, (x**2 + 2*x + 1), evaluate=False)))], # 2-arg Mul!
                'slow': True,  # 解决此方程的速度较慢
            },
        }
    }



# 添加示例键的装饰器，用于函数 _get_examples_ode_sol_nth_linear_undetermined_coefficients
@_add_example_keys
# 定义函数，从普通微分方程中获取具有未定系数的线性方程的解
def _get_examples_ode_sol_nth_linear_undetermined_coefficients():
    # 以下示例来源于《普通微分方程》，Tenenbaum 和 Pollard，第 231 页
    g = exp(-x)  # 变量 g 表示 e^(-x)
    f2 = f(x).diff(x, 2)  # 变量 f2 表示 f(x) 的二阶导数
    c = 3*f(x).diff(x, 3) + 5*f2 + f(x).diff(x) - f(x) - x  # 变量 c 表示给定的线性方程的右侧
    # 创建符号变量 t
    t = symbols("t")
    # 创建符号变量 u，其类型为 Function
    u = symbols("u", cls=Function)
    # 创建多个正数符号变量 R, L, C, E_0, alpha 和一个未指定类型的符号变量 omega
    R, L, C, E_0, alpha = symbols("R L C E_0 alpha", positive=True)
    omega = Symbol('omega')
    # 返回一个字典，包含一些键值对，用于提示、函数、示例等信息
    return {
            'hint': "nth_linear_constant_coeff_undetermined_coefficients",
            # 键为 'func'，对应的值为函数 f(x)
            'func': f(x),
            'examples':{
    'undet_01': {
        # 示例 'undet_01' 的方程
        'eq': c - x*g,
        # 'undet_01' 的解
        'sol': [Eq(f(x), C3*exp(x/3) - x + (C1 + x*(C2 - x**2/24 - 3*x/32))*exp(-x) - 1)],
        # 'undet_01' 的解是否较慢
        'slow': True,
    },

    'undet_02': {
        # 示例 'undet_02' 的方程
        'eq': c - g,
        # 'undet_02' 的解
        'sol': [Eq(f(x), C3*exp(x/3) - x + (C1 + x*(C2 - x/8))*exp(-x) - 1)],
        # 'undet_02' 的解是否较慢
        'slow': True,
    },

    'undet_03': {
        # 示例 'undet_03' 的方程
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - 4,
        # 'undet_03' 的解
        'sol': [Eq(f(x), C1*exp(-2*x) + C2*exp(-x) + 2)],
        # 'undet_03' 的解是否较慢
        'slow': True,
    },

    'undet_04': {
        # 示例 'undet_04' 的方程
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - 12*exp(x),
        # 'undet_04' 的解
        'sol': [Eq(f(x), C1*exp(-2*x) + C2*exp(-x) + 2*exp(x))],
        # 'undet_04' 的解是否较慢
        'slow': True,
    },

    'undet_05': {
        # 示例 'undet_05' 的方程
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - exp(I*x),
        # 'undet_05' 的解
        'sol': [Eq(f(x), (S(3)/10 + I/10)*(C1*exp(-2*x) + C2*exp(-x) - I*exp(I*x)))],
        # 'undet_05' 的解是否较慢
        'slow': True,
    },

    'undet_06': {
        # 示例 'undet_06' 的方程
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - sin(x),
        # 'undet_06' 的解
        'sol': [Eq(f(x), C1*exp(-2*x) + C2*exp(-x) + sin(x)/10 - 3*cos(x)/10)],
        # 'undet_06' 的解是否较慢
        'slow': True,
    },

    'undet_07': {
        # 示例 'undet_07' 的方程
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - cos(x),
        # 'undet_07' 的解
        'sol': [Eq(f(x), C1*exp(-2*x) + C2*exp(-x) + 3*sin(x)/10 + cos(x)/10)],
        # 'undet_07' 的解是否较慢
        'slow': True,
    },

    'undet_08': {
        # 示例 'undet_08' 的方程
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - (8 + 6*exp(x) + 2*sin(x)),
        # 'undet_08' 的解
        'sol': [Eq(f(x), C1*exp(-2*x) + C2*exp(-x) + exp(x) + sin(x)/5 - 3*cos(x)/5 + 4)],
        # 'undet_08' 的解是否较慢
        'slow': True,
    },

    'undet_09': {
        # 示例 'undet_09' 的方程
        'eq': f2 + f(x).diff(x) + f(x) - x**2,
        # 'undet_09' 的解
        'sol': [Eq(f(x), -2*x + x**2 + (C1*sin(x*sqrt(3)/2) + C2*cos(x*sqrt(3)/2))*exp(-x/2))],
        # 'undet_09' 的解是否较慢
        'slow': True,
    },

    'undet_10': {
        # 示例 'undet_10' 的方程
        'eq': f2 - 2*f(x).diff(x) - 8*f(x) - 9*x*exp(x) - 10*exp(-x),
        # 'undet_10' 的解
        'sol': [Eq(f(x), -x*exp(x) - 2*exp(-x) + C1*exp(-2*x) + C2*exp(4*x))],
        # 'undet_10' 的解是否较慢
        'slow': True,
    },

    'undet_11': {
        # 示例 'undet_11' 的方程
        'eq': f2 - 3*f(x).diff(x) - 2*exp(2*x)*sin(x),
        # 'undet_11' 的解
        'sol': [Eq(f(x), C1 + C2*exp(3*x) - 3*exp(2*x)*sin(x)/5 - exp(2*x)*cos(x)/5)],
        # 'undet_11' 的解是否较慢
        'slow': True,
    },

    'undet_12': {
        # 示例 'undet_12' 的方程
        'eq': f(x).diff(x, 4) - 2*f2 + f(x) - x + sin(x),
        # 'undet_12' 的解
        'sol': [Eq(f(x), x - sin(x)/4 + (C1 + C2*x)*exp(-x) + (C3 + C4*x)*exp(x))],
        # 'undet_12' 的解是否较慢
        'slow': True,
    },

    'undet_13': {
        # 示例 'undet_13' 的方程
        'eq': f2 + f(x).diff(x) - x**2 - 2*x,
        # 'undet_13' 的解
        'sol': [Eq(f(x), C1 + x**3/3 + C2*exp(-x))],
        # 'undet_13' 的解是否较慢
        'slow': True,
    },

    'undet_14': {
        # 示例 'undet_14' 的方程
        'eq': f2 + f(x).diff(x) - x - sin(2*x),
        # 'undet_14' 的解
        'sol': [Eq(f(x), C1 - x - sin(2*x)/5 - cos(2*x)/10 + x**2/2 + C2*exp(-x))],
        # 'undet_14' 的解是否较慢
        'slow': True,
    },

    'undet_15': {
        # 示例 'undet_15' 的方程
        'eq': f2 + f(x) - 4*x*sin(x),
        # 'undet_15' 的解
        'sol': [Eq(f(x), (C1 - x**2)*cos(x) + (C2 + x)*sin(x))],
        # 'undet_15' 的解是否较慢
        'slow': True,
    },
    'undet_16': {
        'eq': f2 + 4*f(x) - x*sin(2*x),
        'sol': [Eq(f(x), (C1 - x**2/8)*cos(2*x) + (C2 + x/16)*sin(2*x))],
        'slow': True,
    },

    # 微分方程 undet_16 的形式
    'undet_17': {
        'eq': f2 + 2*f(x).diff(x) + f(x) - x**2*exp(-x),
        'sol': [Eq(f(x), (C1 + x*(C2 + x**3/12))*exp(-x))],
        'slow': True,
    },

    # 微分方程 undet_17 的形式
    'undet_18': {
        'eq': f(x).diff(x, 3) + 3*f2 + 3*f(x).diff(x) + f(x) - 2*exp(-x) + \
        x**2*exp(-x),
        'sol': [Eq(f(x), (C1 + x*(C2 + x*(C3 - x**3/60 + x/3)))*exp(-x))],
        'slow': True,
    },

    # 微分方程 undet_18 的形式
    'undet_19': {
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - exp(-2*x) - x**2,
        'sol': [Eq(f(x), C2*exp(-x) + x**2/2 - x*Rational(3,2) + (C1 - x)*exp(-2*x) + Rational(7,4))],
        'slow': True,
    },

    # 微分方程 undet_19 的形式
    'undet_20': {
        'eq': f2 - 3*f(x).diff(x) + 2*f(x) - x*exp(-x),
        'sol': [Eq(f(x), C1*exp(x) + C2*exp(2*x) + (6*x + 5)*exp(-x)/36)],
        'slow': True,
    },

    # 微分方程 undet_20 的形式
    'undet_21': {
        'eq': f2 + f(x).diff(x) - 6*f(x) - x - exp(2*x),
        'sol': [Eq(f(x), Rational(-1, 36) - x/6 + C2*exp(-3*x) + (C1 + x/5)*exp(2*x))],
        'slow': True,
    },

    # 微分方程 undet_21 的形式
    'undet_22': {
        'eq': f2 + f(x) - sin(x) - exp(-x),
        'sol': [Eq(f(x), C2*sin(x) + (C1 - x/2)*cos(x) + exp(-x)/2)],
        'slow': True,
    },

    # 微分方程 undet_22 的形式
    'undet_23': {
        'eq': f(x).diff(x, 3) - 3*f2 + 3*f(x).diff(x) - f(x) - exp(x),
        'sol': [Eq(f(x), (C1 + x*(C2 + x*(C3 + x/6)))*exp(x))],
        'slow': True,
    },

    # 微分方程 undet_23 的形式
    'undet_24': {
        'eq': f2 + f(x) - S.Half - cos(2*x)/2,
        'sol': [Eq(f(x), S.Half - cos(2*x)/6 + C1*sin(x) + C2*cos(x))],
        'slow': True,
    },

    # 微分方程 undet_24 的形式
    'undet_25': {
        'eq': f(x).diff(x, 3) - f(x).diff(x) - exp(2*x)*(S.Half - cos(2*x)/2),
        'sol': [Eq(f(x), C1 + C2*exp(-x) + C3*exp(x) + (-21*sin(2*x) + 27*cos(2*x) + 130)*exp(2*x)/1560)],
        'slow': True,
    },

    # 微分方程 undet_25 的形式
    # 注意：'undet_26' 在 'undet_37' 中被引用
    'undet_26': {
        'eq': (f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x) - 2*x -
        sin(x) - cos(x)),
        'sol': [Eq(f(x), C1 + x**2 + (C2 + x*(C3 - x/8))*sin(x) + (C4 + x*(C5 + x/8))*cos(x))],
        'slow': True,
    },

    # 微分方程 undet_26 的形式
    'undet_27': {
        'eq': f2 + f(x) - cos(x)/2 + cos(3*x)/2,
        'sol': [Eq(f(x), cos(3*x)/16 + C2*cos(x) + (C1 + x/4)*sin(x))],
        'slow': True,
    },

    # 微分方程 undet_27 的形式
    'undet_28': {
        'eq': f(x).diff(x) - 1,
        'sol': [Eq(f(x), C1 + x)],
        'slow': True,
    },

    # 微分方程 undet_28 的形式
    # https://github.com/sympy/sympy/issues/19358
    'undet_29': {
        'eq': f2 + f(x).diff(x) + exp(x-C1),
        'sol': [Eq(f(x), C2 + C3*exp(-x) - exp(-C1 + x)/2)],
        'slow': True,
    },

    # 微分方程 undet_29 的形式
    # https://github.com/sympy/sympy/issues/18408
    'undet_30': {
        'eq': f(x).diff(x, 3) - f(x).diff(x) - sinh(x),
        'sol': [Eq(f(x), C1 + C2*exp(-x) + C3*exp(x) + x*sinh(x)/2)],
    },

    # 微分方程 undet_30 的形式
    'undet_31': {
        'eq': f(x).diff(x, 2) - 49*f(x) - sinh(3*x),
        'sol': [Eq(f(x), C1*exp(-7*x) + C2*exp(7*x) - sinh(3*x)/40)],
    },

    'undet_32': {
        'eq': f(x).diff(x, 3) - f(x).diff(x) - sinh(x) - exp(x),
        'sol': [Eq(f(x), C1 + C3*exp(-x) + x*sinh(x)/2 + (C2 + x/2)*exp(x))],
    },

    # https://github.com/sympy/sympy/issues/5096
    'undet_33': {
        'eq': f(x).diff(x, x) + f(x) - x*sin(x - 2),
        'sol': [Eq(f(x), C1*sin(x) + C2*cos(x) - x**2*cos(x - 2)/4 + x*sin(x - 2)/4)],
    },

    'undet_34': {
        'eq': f(x).diff(x, 2) + f(x) - x**4*sin(x-1),
        'sol': [ Eq(f(x), C1*sin(x) + C2*cos(x) - x**5*cos(x - 1)/10 + x**4*sin(x - 1)/4 + x**3*cos(x - 1)/2 - 3*x**2*sin(x - 1)/4 - 3*x*cos(x - 1)/4)],
    },

    'undet_35': {
        'eq': f(x).diff(x, 2) - f(x) - exp(x - 1),
        'sol': [Eq(f(x), C2*exp(-x) + (C1 + x*exp(-1)/2)*exp(x))],
    },

    'undet_36': {
        'eq': f(x).diff(x, 2)+f(x)-(sin(x-2)+1),
        'sol': [Eq(f(x), C1*sin(x) + C2*cos(x) - x*cos(x - 2)/2 + 1)],
    },

    # Equivalent to example_name 'undet_26'.
    # This previously failed because the algorithm for undetermined coefficients
    # didn't know to multiply exp(I*x) by sufficient x because it is linearly
    # dependent on sin(x) and cos(x).
    'undet_37': {
        'eq': f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x) - 2*x - exp(I*x),
        'sol': [Eq(f(x), C1 + x**2*(I*exp(I*x)/8 + 1) + (C2 + C3*x)*sin(x) + (C4 + C5*x)*cos(x))],
    },

    # https://github.com/sympy/sympy/issues/12623
    'undet_38': {
        'eq': Eq( u(t).diff(t,t) + R /L*u(t).diff(t) + 1/(L*C)*u(t), alpha),
        'sol': [Eq(u(t), C*L*alpha + C2*exp(-t*(R + sqrt(C*R**2 - 4*L)/sqrt(C))/(2*L))
        + C1*exp(t*(-R + sqrt(C*R**2 - 4*L)/sqrt(C))/(2*L)))],
        'func': u(t)
    },

    'undet_39': {
        'eq': Eq( L*C*u(t).diff(t,t) + R*C*u(t).diff(t) + u(t), E_0*exp(I*omega*t) ),
        'sol': [Eq(u(t), C2*exp(-t*(R + sqrt(C*R**2 - 4*L)/sqrt(C))/(2*L))
        + C1*exp(t*(-R + sqrt(C*R**2 - 4*L)/sqrt(C))/(2*L))
        - E_0*exp(I*omega*t)/(C*L*omega**2 - I*C*R*omega - 1))],
        'func': u(t),
    },

    # https://github.com/sympy/sympy/issues/6879
    'undet_40': {
        'eq': Eq(Derivative(f(x), x, 2) - 2*Derivative(f(x), x) + f(x), sin(x)),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(x) + cos(x)/2)],
    },
}
@_add_example_keys
# 定义一个函数用于获取可分离的常微分方程示例
def _get_examples_ode_sol_separable():
    # 符号变量的声明
    t, a = symbols('a, t')
    # 常数定义
    m = 96
    g = 9.8
    k = .2
    # 计算力的公式
    f1 = g * m
    # 定义一个函数符号
    v = Function('v')
    # 返回包含示例常微分方程及其解的字典
    return {
        'hint': "separable",
        'func': f(x),  # 此处应为函数表达式的字符串，而不是变量 f(x)
        'examples': {
            'separable_01': {
                'eq': f(x).diff(x) - f(x),  # 第一个可分离微分方程
                'sol': [Eq(f(x), C1*exp(x))],  # 对应的解
            },

            'separable_02': {
                'eq': x*f(x).diff(x) - f(x),  # 第二个可分离微分方程
                'sol': [Eq(f(x), C1*x)],  # 对应的解
            },

            'separable_03': {
                'eq': f(x).diff(x) + sin(x),  # 第三个可分离微分方程
                'sol': [Eq(f(x), C1 + cos(x))],  # 对应的解
            },

            'separable_04': {
                'eq': f(x)**2 + 1 - (x**2 + 1)*f(x).diff(x),  # 第四个可分离微分方程
                'sol': [Eq(f(x), tan(C1 + atan(x)))],  # 对应的解
            },

            'separable_05': {
                'eq': f(x).diff(x)/tan(x) - f(x) - 2,  # 第五个可分离微分方程
                'sol': [Eq(f(x), C1/cos(x) - 2)],  # 对应的解
            },

            'separable_06': {
                'eq': f(x).diff(x) * (1 - sin(f(x))) - 1,  # 第六个可分离微分方程
                'sol': [Eq(-x + f(x) + cos(f(x)), C1)],  # 对应的解
            },

            'separable_07': {
                'eq': f(x)*x**2*f(x).diff(x) - f(x)**3 - 2*x**2*f(x).diff(x),  # 第七个可分离微分方程
                'sol': [  # 对应的解，包括两个不同的解
                    Eq(f(x), (-x - sqrt(x*(4*C1*x + x - 4)))/(C1*x - 1)/2),
                    Eq(f(x), (-x + sqrt(x*(4*C1*x + x - 4)))/(C1*x - 1)/2)
                ],
                'slow': True,
            },

            'separable_08': {
                'eq': f(x)**2 - 1 - (2*f(x) + x*f(x))*f(x).diff(x),  # 第八个可分离微分方程
                'sol': [  # 对应的解，包括两个不同的解
                    Eq(f(x), -sqrt(C1*x**2 + 4*C1*x + 4*C1 + 1)),
                    Eq(f(x), sqrt(C1*x**2 + 4*C1*x + 4*C1 + 1))
                ],
                'slow': True,
            },

            'separable_09': {
                'eq': x*log(x)*f(x).diff(x) + sqrt(1 + f(x)**2),  # 第九个可分离微分方程
                'sol': [Eq(f(x), sinh(C1 - log(log(x))))],  # 对应的解
                'slow': True,
                'checkodesol_XFAIL': True,
            },

            'separable_10': {
                'eq': exp(x + 1)*tan(f(x)) + cos(f(x))*f(x).diff(x),  # 第十个可分离微分方程
                'sol': [  # 对应的解
                    Eq(E*exp(x) + log(cos(f(x)) - 1)/2 - log(cos(f(x)) + 1)/2 + cos(f(x)), C1)
                ],
                'slow': True,
            },

            'separable_11': {
                'eq': (x*cos(f(x)) + x**2*sin(f(x))*f(x).diff(x) - a**2*sin(f(x))*f(x).diff(x)),  # 第十一个可分离微分方程
                'sol': [  # 对应的解，包括两个不同的解
                    Eq(f(x), -acos(C1*sqrt(-a**2 + x**2)) + 2*pi),
                    Eq(f(x), acos(C1*sqrt(-a**2 + x**2)))
                ],
                'slow': True,
            },

            'separable_12': {
                'eq': f(x).diff(x) - f(x)*tan(x),  # 第十二个可分离微分方程
                'sol': [Eq(f(x), C1/cos(x))],  # 对应的解
            },

            'separable_13': {
                'eq': (x - 1)*cos(f(x))*f(x).diff(x) - 2*x*sin(f(x)),  # 第十三个可分离微分方程
                'sol': [  # 对应的解，包括两个不同的解
                    Eq(f(x), pi - asin(C1*(x**2 - 2*x + 1)*exp(2*x))),
                    Eq(f(x), asin(C1*(x**2 - 2*x + 1)*exp(2*x)))
                ],
            },

            'separable_14': {
                'eq': f(x).diff(x) - f(x)*log(f(x))/tan(x),  # 第十四个可分离微分方程
                'sol': [Eq(f(x), exp(C1*sin(x)))],  # 对应的解
            },

            'separable_15': {
                'eq': x*f(x).diff(x) + (1 + f(x)**2)*atan(f(x)),  # 第十五个可分离微分方程
                'sol': [Eq(f(x), tan(C1/x))],  # 对应的解
                'slow': True,
                'checkodesol_XFAIL': True,
            },
        },
    }
    'separable_16': {
        'eq': f(x).diff(x) + x*(f(x) + 1),
        'sol': [Eq(f(x), -1 + C1*exp(-x**2/2))],
    },

    'separable_17': {
        'eq': exp(f(x)**2)*(x**2 + 2*x + 1) + (x*f(x) + f(x))*f(x).diff(x),
        'sol': [
            Eq(f(x), -sqrt(log(1/(C1 + x**2 + 2*x)))),
            Eq(f(x), sqrt(log(1/(C1 + x**2 + 2*x))))
        ],
    },

    'separable_18': {
        'eq': f(x).diff(x) + f(x),
        'sol': [Eq(f(x), C1*exp(-x))],
    },

    'separable_19': {
        'eq': sin(x)*cos(2*f(x)) + cos(x)*sin(2*f(x))*f(x).diff(x),
        'sol': [Eq(f(x), pi - acos(C1/cos(x)**2)/2), Eq(f(x), acos(C1/cos(x)**2)/2)],
    },

    'separable_20': {
        'eq': (1 - x)*f(x).diff(x) - x*(f(x) + 1),
        'sol': [Eq(f(x), (C1*exp(-x) - x + 1)/(x - 1))],
    },

    'separable_21': {
        'eq': f(x)*diff(f(x), x) + x - 3*x*f(x)**2,
        'sol': [Eq(f(x), -sqrt(3)*sqrt(C1*exp(3*x**2) + 1)/3),
        Eq(f(x), sqrt(3)*sqrt(C1*exp(3*x**2) + 1)/3)],
    },

    'separable_22': {
        'eq': f(x).diff(x) - exp(x + f(x)),
        'sol': [Eq(f(x), log(-1/(C1 + exp(x))))],
        'XFAIL': ['lie_group'] # It shows 'NoneType' object is not subscriptable for lie_group.
    },

    # https://github.com/sympy/sympy/issues/7081
    'separable_23': {
        'eq': x*(f(x).diff(x)) + 1 - f(x)**2,
        'sol': [Eq(f(x), (-C1 - x**2)/(-C1 + x**2))],
    },

    # https://github.com/sympy/sympy/issues/10379
    'separable_24': {
        'eq': f(t).diff(t)-(1-51.05*y*f(t)),
        'sol': [Eq(f(t), (0.019588638589618023*exp(y*(C1 - 51.049999999999997*t)) + 0.019588638589618023)/y)],
        'func': f(t),
    },

    # https://github.com/sympy/sympy/issues/15999
    'separable_25': {
        'eq': f(x).diff(x) - C1*f(x),
        'sol': [Eq(f(x), C2*exp(C1*x))],
    },

    'separable_26': {
        'eq': f1 - k * (v(t) ** 2) - m * Derivative(v(t)),
        'sol': [Eq(v(t), -68.585712797928991/tanh(C1 - 0.14288690166235204*t))],
        'func': v(t),
        'checkodesol_XFAIL': True,
    },

    # https://github.com/sympy/sympy/issues/22155
    'separable_27': {
        'eq': f(x).diff(x) - exp(f(x) - x),
        'sol': [Eq(f(x), log(-exp(x)/(C1*exp(x) - 1)))],
    }
@_add_example_keys
# 定义一个函数，用于获取一阶精确微分方程的示例解法
def _get_examples_ode_sol_1st_exact():
    '''
    Example 7 is an exact equation that fails under the exact engine. It is caught
    by first order homogeneous albeit with a much contorted solution.  The
    exact engine fails because of a poorly simplified integral of q(0,y)dy,
    where q is the function multiplying f'.  The solutions should be
    Eq(sqrt(x**2+f(x)**2)**3+y**3, C1).  The equation below is
    equivalent, but it is so complex that checkodesol fails, and takes a long
    time to do so.
    '''
    # 返回一个字典，包含示例的解法和相关信息
    return {
            'hint': "1st_exact",  # 提示信息，指示这是一阶精确微分方程的示例
            'func': f(x),  # 函数的表达式 f(x)，这里可能是一个占位符或符号表达式
            'examples':{
    '1st_exact_01': {
        'eq': sin(x)*cos(f(x)) + cos(x)*sin(f(x))*f(x).diff(x),
        'sol': [Eq(f(x), -acos(C1/cos(x)) + 2*pi), Eq(f(x), acos(C1/cos(x)))],
        'slow': True,  # 解法比较慢，可能需要较长时间计算
    },

    '1st_exact_02': {
        'eq': (2*x*f(x) + 1)/f(x) + (f(x) - x)/f(x)**2*f(x).diff(x),
        'sol': [Eq(f(x), exp(C1 - x**2 + LambertW(-x*exp(-C1 + x**2))))],
        'XFAIL': ['lie_group'],  # 在 lie_group 情况下失败，dsolve 抛出异常
        'slow': True,
        'checkodesol_XFAIL':True  # checkodesol 运行失败
    },

    '1st_exact_03': {
        'eq': 2*x + f(x)*cos(x) + (2*f(x) + sin(x) - sin(f(x)))*f(x).diff(x),
        'sol': [Eq(f(x)*sin(x) + cos(f(x)) + x**2 + f(x)**2, C1)],
        'XFAIL': ['lie_group'],  # 在 lie_group 情况下进入无限循环
        'slow': True,
    },

    '1st_exact_04': {
        'eq': cos(f(x)) - (x*sin(f(x)) - f(x)**2)*f(x).diff(x),
        'sol': [Eq(x*cos(f(x)) + f(x)**3/3, C1)],
        'slow': True,
    },

    '1st_exact_05': {
        'eq': 2*x*f(x) + (x**2 + f(x)**2)*f(x).diff(x),
        'sol': [Eq(x**2*f(x) + f(x)**3/3, C1)],
        'slow': True,
        'simplify_flag':False  # 不进行简化
    },

    # This was from issue: https://github.com/sympy/sympy/issues/11290
    '1st_exact_06': {
        'eq': cos(f(x)) - (x*sin(f(x)) - f(x)**2)*f(x).diff(x),
        'sol': [Eq(x*cos(f(x)) + f(x)**3/3, C1)],
        'simplify_flag':False  # 不进行简化
    },

    '1st_exact_07': {
        'eq': x*sqrt(x**2 + f(x)**2) - (x**2*f(x)/(f(x) - sqrt(x**2 + f(x)**2)))*f(x).diff(x),
        'sol': [Eq(log(x),
        C1 - 9*sqrt(1 + f(x)**2/x**2)*asinh(f(x)/x)/(-27*f(x)/x +
        27*sqrt(1 + f(x)**2/x**2)) - 9*sqrt(1 + f(x)**2/x**2)*
        log(1 - sqrt(1 + f(x)**2/x**2)*f(x)/x + 2*f(x)**2/x**2)/
        (-27*f(x)/x + 27*sqrt(1 + f(x)**2/x**2)) +
        9*asinh(f(x)/x)*f(x)/(x*(-27*f(x)/x + 27*sqrt(1 + f(x)**2/x**2))) +
        9*f(x)*log(1 - sqrt(1 + f(x)**2/x**2)*f(x)/x + 2*f(x)**2/x**2)/
        (x*(-27*f(x)/x + 27*sqrt(1 + f(x)**2/x**2))))],
        'slow': True,
        'dsolve_too_slow':True  # dsolve 运行速度过慢
    },

    # Type: a(x)f'(x)+b(x)*f(x)+c(x)=0
    '1st_exact_08': {
        'eq': Eq(x**2*f(x).diff(x) + 3*x*f(x) - sin(x)/x, 0),
        'sol': [Eq(f(x), (C1 - cos(x))/x**3)],
    },
    # 这些例子来自 test_exact_enhancement 测试
    '1st_exact_09': {
        # 方程表达式：f(x)/x**2 + ((f(x)*x - 1)/x)*f(x).diff(x)
        'eq': f(x)/x**2 + ((f(x)*x - 1)/x)*f(x).diff(x),
        # 方程的解：[Eq(f(x), (i*sqrt(C1*x**2 + 1) + 1)/x) for i in (-1, 1)]
        'sol': [Eq(f(x), (i*sqrt(C1*x**2 + 1) + 1)/x) for i in (-1, 1)],
    },

    '1st_exact_10': {
        # 方程表达式：(x*f(x) - 1) + f(x).diff(x)*(x**2 - x*f(x))
        'eq': (x*f(x) - 1) + f(x).diff(x)*(x**2 - x*f(x)),
        # 方程的解：[Eq(f(x), x - sqrt(C1 + x**2 - 2*log(x))), Eq(f(x), x + sqrt(C1 + x**2 - 2*log(x)))]
        'sol': [Eq(f(x), x - sqrt(C1 + x**2 - 2*log(x))), Eq(f(x), x + sqrt(C1 + x**2 - 2*log(x)))],
    },

    '1st_exact_11': {
        # 方程表达式：(x + 2)*sin(f(x)) + f(x).diff(x)*x*cos(f(x))
        'eq': (x + 2)*sin(f(x)) + f(x).diff(x)*x*cos(f(x)),
        # 方程的解：[Eq(f(x), -asin(C1*exp(-x)/x**2) + pi), Eq(f(x), asin(C1*exp(-x)/x**2))]
        'sol': [Eq(f(x), -asin(C1*exp(-x)/x**2) + pi), Eq(f(x), asin(C1*exp(-x)/x**2))],
    },
}
@_add_example_keys
def _get_examples_ode_sol_nth_linear_var_of_parameters():
    # 定义变量 g，表示 exp(-x)
    g = exp(-x)
    # 计算 f(x) 关于 x 的二阶导数
    f2 = f(x).diff(x, 2)
    # 构建常系数线性变分参数法的方程 c
    c = 3*f(x).diff(x, 3) + 5*f2 + f(x).diff(x) - f(x) - x
    # 返回包含多个示例的字典
    return {
            'hint': "nth_linear_constant_coeff_variation_of_parameters",
            'func': f(x),
            'examples':{
    'var_of_parameters_01': {
        # 第一个示例的方程
        'eq': c - x*g,
        # 第一个示例的解
        'sol': [Eq(f(x), C3*exp(x/3) - x + (C1 + x*(C2 - x**2/24 - 3*x/32))*exp(-x) - 1)],
        # 表示这个示例解的计算速度较慢
        'slow': True,
    },

    'var_of_parameters_02': {
        # 第二个示例的方程
        'eq': c - g,
        # 第二个示例的解
        'sol': [Eq(f(x), C3*exp(x/3) - x + (C1 + x*(C2 - x/8))*exp(-x) - 1)],
        # 表示这个示例解的计算速度较慢
        'slow': True,
    },

    'var_of_parameters_03': {
        # 第三个示例的方程
        'eq': f(x).diff(x) - 1,
        # 第三个示例的解
        'sol': [Eq(f(x), C1 + x)],
        # 表示这个示例解的计算速度较慢
        'slow': True,
    },

    'var_of_parameters_04': {
        # 第四个示例的方程
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - 4,
        # 第四个示例的解
        'sol': [Eq(f(x), C1*exp(-2*x) + C2*exp(-x) + 2)],
        # 表示这个示例解的计算速度较慢
        'slow': True,
    },

    'var_of_parameters_05': {
        # 第五个示例的方程
        'eq': f2 + 3*f(x).diff(x) + 2*f(x) - 12*exp(x),
        # 第五个示例的解
        'sol': [Eq(f(x), C1*exp(-2*x) + C2*exp(-x) + 2*exp(x))],
        # 表示这个示例解的计算速度较慢
        'slow': True,
    },

    'var_of_parameters_06': {
        # 第六个示例的方程
        'eq': f2 - 2*f(x).diff(x) - 8*f(x) - 9*x*exp(x) - 10*exp(-x),
        # 第六个示例的解
        'sol': [Eq(f(x), -x*exp(x) - 2*exp(-x) + C1*exp(-2*x) + C2*exp(4*x))],
        # 表示这个示例解的计算速度较慢
        'slow': True,
    },

    'var_of_parameters_07': {
        # 第七个示例的方程
        'eq': f2 + 2*f(x).diff(x) + f(x) - x**2*exp(-x),
        # 第七个示例的解
        'sol': [Eq(f(x), (C1 + x*(C2 + x**3/12))*exp(-x))],
        # 表示这个示例解的计算速度较慢
        'slow': True,
    },

    'var_of_parameters_08': {
        # 第八个示例的方程
        'eq': f2 - 3*f(x).diff(x) + 2*f(x) - x*exp(-x),
        # 第八个示例的解
        'sol': [Eq(f(x), C1*exp(x) + C2*exp(2*x) + (6*x + 5)*exp(-x)/36)],
        # 表示这个示例解的计算速度较慢
        'slow': True,
    },

    'var_of_parameters_09': {
        # 第九个示例的方程
        'eq': f(x).diff(x, 3) - 3*f2 + 3*f(x).diff(x) - f(x) - exp(x),
        # 第九个示例的解
        'sol': [Eq(f(x), (C1 + x*(C2 + x*(C3 + x/6)))*exp(x))],
        # 表示这个示例解的计算速度较慢
        'slow': True,
    },

    'var_of_parameters_10': {
        # 第十个示例的方程
        'eq': f2 + 2*f(x).diff(x) + f(x) - exp(-x)/x,
        # 第十个示例的解
        'sol': [Eq(f(x), (C1 + x*(C2 + log(x)))*exp(-x))],
        # 表示这个示例解的计算速度较慢
        'slow': True,
    },

    'var_of_parameters_11': {
        # 第十一个示例的方程
        'eq': f2 + f(x) - 1/sin(x)*1/cos(x),
        # 第十一个示例的解
        'sol': [Eq(f(x), (C1 + log(sin(x) - 1)/2 - log(sin(x) + 1)/2
        )*cos(x) + (C2 + log(cos(x) - 1)/2 - log(cos(x) + 1)/2)*sin(x))],
        # 表示这个示例解的计算速度较慢
        'slow': True,
    },

    'var_of_parameters_12': {
        # 第十二个示例的方程
        'eq': f(x).diff(x, 4) - 1/x,
        # 第十二个示例的解
        'sol': [Eq(f(x), C1 + C2*x + C3*x**2 + x**3*(C4 + log(x)/6))],
        # 表示这个示例解的计算速度较慢
        'slow': True,
    },

    # 这些示例来源于问题：https://github.com/sympy/sympy/issues/15996
    'var_of_parameters_13': {
        # 第十三个示例的方程
        'eq': f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x) - 2*x - exp(I*x),
        # 第十三个示例的解
        'sol': [Eq(f(x), C1 + x**2 + (C2 + x*(C3 - x/8 + 3*exp(I*x)/2 + 3*exp(-I*x)/2) + 5*exp(2*I*x)/16 + 2*I*exp(I*x) - 2*I*exp(-I*x))*sin(x) + (C4 + x*(C5 + I*x/8 + 3*I*exp(I*x)/2 - 3*I*exp(-I*x)/2)
        + 5*I*exp(2*I*x)/16 - 2*exp(I*x) - 2*exp(-I*x))*cos(x) - I*exp(I*x))],
    },
    # 定义一个名为 'var_of_parameters_14' 的字典，包含方程和解的信息
    'var_of_parameters_14': {
        # 方程: f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x) - exp(I*x)
        'eq': f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x) - exp(I*x),
        # 解: [Eq(f(x), C1 + (C2 + x*(C3 - x/8) + 5*exp(2*I*x)/16)*sin(x) + (C4 + x*(C5 + I*x/8) + 5*I*exp(2*I*x)/16)*cos(x) - I*exp(I*x))]
        'sol': [Eq(f(x), C1 + (C2 + x*(C3 - x/8) + 5*exp(2*I*x)/16)*sin(x) + (C4 + x*(C5 + I*x/8) + 5*I*exp(2*I*x)/16)*cos(x) - I*exp(I*x))],
    },

    # https://github.com/sympy/sympy/issues/14395
    # 定义一个名为 'var_of_parameters_15' 的字典，包含方程、解和慢速标志的信息
    'var_of_parameters_15': {
        # 方程: Derivative(f(x), x, x) + 9*f(x) - sec(x)
        'eq': Derivative(f(x), x, x) + 9*f(x) - sec(x),
        # 解: [Eq(f(x), (C1 - x/3 + sin(2*x)/3)*sin(3*x) + (C2 + log(cos(x)) - 2*log(cos(x)**2)/3 + 2*cos(x)**2/3)*cos(3*x))]
        'sol': [Eq(f(x), (C1 - x/3 + sin(2*x)/3)*sin(3*x) + (C2 + log(cos(x)) - 2*log(cos(x)**2)/3 + 2*cos(x)**2/3)*cos(3*x))],
        # 慢速标志: True
        'slow': True,
    }
@_add_example_keys
# 添加示例关键字的装饰器函数，用于给特定函数返回的字典添加额外的示例信息
def _get_examples_ode_sol_2nd_linear_bessel():
    return {
            'hint': "2nd_linear_bessel",
            'func': f(x),
            'examples':{
    '2nd_lin_bessel_01': {
        'eq': x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (x**2 - 4)*f(x),
        'sol': [Eq(f(x), C1*besselj(2, x) + C2*bessely(2, x))],
    },

    '2nd_lin_bessel_02': {
        'eq': x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (x**2 +25)*f(x),
        'sol': [Eq(f(x), C1*besselj(5*I, x) + C2*bessely(5*I, x))],
    },

    '2nd_lin_bessel_03': {
        'eq': x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (x**2)*f(x),
        'sol': [Eq(f(x), C1*besselj(0, x) + C2*bessely(0, x))],
    },

    '2nd_lin_bessel_04': {
        'eq': x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (81*x**2 -S(1)/9)*f(x),
        'sol': [Eq(f(x), C1*besselj(S(1)/3, 9*x) + C2*bessely(S(1)/3, 9*x))],
    },

    '2nd_lin_bessel_05': {
        'eq': x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (x**4 - 4)*f(x),
        'sol': [Eq(f(x), C1*besselj(1, x**2/2) + C2*bessely(1, x**2/2))],
    },

    '2nd_lin_bessel_06': {
        'eq': x**2*(f(x).diff(x, 2)) + 2*x*(f(x).diff(x)) + (x**4 - 4)*f(x),
        'sol': [Eq(f(x), (C1*besselj(sqrt(17)/4, x**2/2) + C2*bessely(sqrt(17)/4, x**2/2))/sqrt(x))],
    },

    '2nd_lin_bessel_07': {
        'eq': x**2*(f(x).diff(x, 2)) + x*(f(x).diff(x)) + (x**2 - S(1)/4)*f(x),
        'sol': [Eq(f(x), C1*besselj(S(1)/2, x) + C2*bessely(S(1)/2, x))],
    },

    '2nd_lin_bessel_08': {
        'eq': x**2*(f(x).diff(x, 2)) - 3*x*(f(x).diff(x)) + (4*x + 4)*f(x),
        'sol': [Eq(f(x), x**2*(C1*besselj(0, 4*sqrt(x)) + C2*bessely(0, 4*sqrt(x))))],
    },

    '2nd_lin_bessel_09': {
        'eq': x*(f(x).diff(x, 2)) - f(x).diff(x) + 4*x**3*f(x),
        'sol': [Eq(f(x), x*(C1*besselj(S(1)/2, x**2) + C2*bessely(S(1)/2, x**2)))],
    },

    '2nd_lin_bessel_10': {
        'eq': (x-2)**2*(f(x).diff(x, 2)) - (x-2)*f(x).diff(x) + 4*(x-2)**2*f(x),
        'sol': [Eq(f(x), (x - 2)*(C1*besselj(1, 2*x - 4) + C2*bessely(1, 2*x - 4)))],
    },

    # https://github.com/sympy/sympy/issues/4414
    '2nd_lin_bessel_11': {
        'eq': f(x).diff(x, x) + 2/x*f(x).diff(x) + f(x),
        'sol': [Eq(f(x), (C1*besselj(S(1)/2, x) + C2*bessely(S(1)/2, x))/sqrt(x))],
    },
    '2nd_lin_bessel_12': {
        'eq': x**2*f(x).diff(x, 2) + x*f(x).diff(x) + (a**2*x**2/c**2 - b**2)*f(x),
        'sol': [Eq(f(x), C1*besselj(sqrt(b**2), x*sqrt(a**2/c**2)) + C2*bessely(sqrt(b**2), x*sqrt(a**2/c**2)))],
    },
    }
    }


@_add_example_keys
# 添加示例关键字的装饰器函数，用于给特定函数返回的字典添加额外的示例信息
def _get_examples_ode_sol_2nd_2F1_hypergeometric():
    return {
            'hint': "2nd_hypergeometric",
            'func': f(x),
            'examples':{
    '2nd_2F1_hyper_01': {
        'eq': x*(x-1)*f(x).diff(x, 2) + (S(3)/2 -2*x)*f(x).diff(x) + 2*f(x),
        'sol': [Eq(f(x), C1*x**(S(5)/2)*hyper((S(3)/2, S(1)/2), (S(7)/2,), x) + C2*hyper((-1, -2), (-S(3)/2,), x))],
    },
    '2nd_2F1_hyper_02': {
        # 定义微分方程
        'eq': x*(x-1)*f(x).diff(x, 2) + (S(7)/2*x)*f(x).diff(x) + f(x),
        # 给出解
        'sol': [Eq(f(x), (C1*(1 - x)**(S(5)/2)*hyper((S(1)/2, 2), (S(7)/2,), 1 - x) +
          C2*hyper((-S(1)/2, -2), (-S(3)/2,), 1 - x))/(x - 1)**(S(5)/2))],
    },

    '2nd_2F1_hyper_03': {
        # 定义微分方程
        'eq': x*(x-1)*f(x).diff(x, 2) + (S(3)+ S(7)/2*x)*f(x).diff(x) + f(x),
        # 给出解
        'sol': [Eq(f(x), (C1*(1 - x)**(S(11)/2)*hyper((S(1)/2, 2), (S(13)/2,), 1 - x) +
          C2*hyper((-S(7)/2, -5), (-S(9)/2,), 1 - x))/(x - 1)**(S(11)/2))],
    },

    '2nd_2F1_hyper_04': {
        # 定义微分方程
        'eq': -x**(S(5)/7)*(-416*x**(S(9)/7)/9 - 2385*x**(S(5)/7)/49 + S(298)*x/3)*f(x)/(196*(-x**(S(6)/7) +
         x)**2*(x**(S(6)/7) + x)**2) + Derivative(f(x), (x, 2)),
        # 给出解
        'sol': [Eq(f(x), x**(S(45)/98)*(C1*x**(S(4)/49)*hyper((S(1)/3, -S(1)/2), (S(9)/7,), x**(S(2)/7)) +
          C2*hyper((S(1)/21, -S(11)/14), (S(5)/7,), x**(S(2)/7)))/(x**(S(2)/7) - 1)**(S(19)/84))],
        # 指示解不符合常微分方程的检查
        'checkodesol_XFAIL':True,
    },
}
# 添加示例关键字的装饰器
@_add_example_keys
# 定义函数 _get_examples_ode_sol_2nd_nonlinear_autonomous_conserved，返回示例字典
def _get_examples_ode_sol_2nd_nonlinear_autonomous_conserved():
    # 返回包含提示、函数和示例的字典
    return {
        'hint': "2nd_nonlinear_autonomous_conserved",  # 提示信息
        'func': f(x),  # 函数表达式
        'examples': {
            # 第一个示例
            '2nd_nonlinear_autonomous_conserved_01': {
                'eq': f(x).diff(x, 2) + exp(f(x)) + log(f(x)),  # 微分方程
                'sol': [
                    # 解的列表
                    Eq(Integral(1/sqrt(C1 - 2*_u*log(_u) + 2*_u - 2*exp(_u)), (_u, f(x))), C2 + x),
                    Eq(Integral(1/sqrt(C1 - 2*_u*log(_u) + 2*_u - 2*exp(_u)), (_u, f(x))), C2 - x)
                ],
                'simplify_flag': False,  # 简化标志
            },
            # 第二个示例
            '2nd_nonlinear_autonomous_conserved_02': {
                'eq': f(x).diff(x, 2) + cbrt(f(x)) + 1/f(x),  # 微分方程
                'sol': [
                    # 解的列表
                    Eq(sqrt(2)*Integral(1/sqrt(2*C1 - 3*_u**Rational(4, 3) - 4*log(_u)), (_u, f(x))), C2 + x),
                    Eq(sqrt(2)*Integral(1/sqrt(2*C1 - 3*_u**Rational(4, 3) - 4*log(_u)), (_u, f(x))), C2 - x)
                ],
                'simplify_flag': False,  # 简化标志
            },
            # 第三个示例
            '2nd_nonlinear_autonomous_conserved_03': {
                'eq': f(x).diff(x, 2) + sin(f(x)),  # 微分方程
                'sol': [
                    # 解的列表
                    Eq(Integral(1/sqrt(C1 + 2*cos(_u)), (_u, f(x))), C2 + x),
                    Eq(Integral(1/sqrt(C1 + 2*cos(_u)), (_u, f(x))), C2 - x)
                ],
                'simplify_flag': False,  # 简化标志
            },
            # 第四个示例
            '2nd_nonlinear_autonomous_conserved_04': {
                'eq': f(x).diff(x, 2) + cosh(f(x)),  # 微分方程
                'sol': [
                    # 解的列表
                    Eq(Integral(1/sqrt(C1 - 2*sinh(_u)), (_u, f(x))), C2 + x),
                    Eq(Integral(1/sqrt(C1 - 2*sinh(_u)), (_u, f(x))), C2 - x)
                ],
                'simplify_flag': False,  # 简化标志
            },
            # 第五个示例
            '2nd_nonlinear_autonomous_conserved_05': {
                'eq': f(x).diff(x, 2) + asin(f(x)),  # 微分方程
                'sol': [
                    # 解的列表
                    Eq(Integral(1/sqrt(C1 - 2*_u*asin(_u) - 2*sqrt(1 - _u**2)), (_u, f(x))), C2 + x),
                    Eq(Integral(1/sqrt(C1 - 2*_u*asin(_u) - 2*sqrt(1 - _u**2)), (_u, f(x))), C2 - x)
                ],
                'simplify_flag': False,  # 简化标志
                'XFAIL': ['2nd_nonlinear_autonomous_conserved_Integral']  # 额外的失败信息
            }
        }
    }


# 添加示例关键字的装饰器
@_add_example_keys
# 定义函数 _get_examples_ode_sol_separable_reduced，返回示例字典
def _get_examples_ode_sol_separable_reduced():
    # 计算函数的导数
    df = f(x).diff(x)
    # 返回包含提示、函数和示例的字典
    return {
        'hint': "separable_reduced",  # 提示信息
        'func': f(x),  # 函数表达式
        'examples': {
            # 第一个示例
            'separable_reduced_01': {
                'eq': x * df + f(x) * (1 / (x**2 * f(x) - 1)),  # 微分方程
                'sol': [Eq(log(x**2 * f(x)) / 3 + log(x**2 * f(x) - Rational(3, 2)) / 6, C1 + log(x))],  # 解的列表
                'simplify_flag': False,  # 简化标志
                'XFAIL': ['lie_group'],  # 额外的失败信息
            },
            # 第二个示例，注释指出被第一个例子引用
            'separable_reduced_02': {
                'eq': f(x).diff(x) + (f(x) / (x**4 * f(x) - x)),  # 微分方程
                'sol': [Eq(log(x**3 * f(x)) / 4 + log(x**3 * f(x) - Rational(4, 3)) / 12, C1 + log(x))],  # 解的列表
                'simplify_flag': False,  # 简化标志
                'checkodesol_XFAIL': True,  # 额外的失败信息
            },
            # 第三个示例
            'separable_reduced_03': {
                'eq': x * df + f(x) * (x**2 * f(x)),  # 微分方程
                'sol': [Eq(log(x**2 * f(x)) / 2 - log(x**2 * f(x) - 2) / 2, C1 + log(x))],  # 解的列表
                'simplify_flag': False,  # 简化标志
            }
        }
    }
    # 定义一个名为 'separable_reduced_04' 的字典项，包含以下内容：
    'separable_reduced_04': {
        # 方程式：f(x) 的导数加上 f(x)/x * (1 + (x**(S(2)/3)*f(x))**2) 等于 0
        'eq': Eq(f(x).diff(x) + f(x)/x * (1 + (x**(S(2)/3)*f(x))**2), 0),
        # 解的列表
        'sol': [Eq(-3*log(x**(S(2)/3)*f(x)) + 3*log(3*x**(S(4)/3)*f(x)**2 + 1)/2, C1 + log(x))],
        # 简化标志
        'simplify_flag': False,
    },

    # 定义一个名为 'separable_reduced_05' 的字典项，包含以下内容：
    'separable_reduced_05': {
        # 方程式：f(x) 的导数加上 f(x)/x * (1 + (x*f(x))**2) 等于 0
        'eq': Eq(f(x).diff(x) + f(x)/x * (1 + (x*f(x))**2), 0),
        # 解的列表
        'sol': [Eq(f(x), -sqrt(2)*sqrt(1/(C1 + log(x)))/(2*x)), Eq(f(x), sqrt(2)*sqrt(1/(C1 + log(x)))/(2*x))],
    },

    # 定义一个名为 'separable_reduced_06' 的字典项，包含以下内容：
    'separable_reduced_06': {
        # 方程式：f(x) 的导数加上 (x**4*f(x)**2 + x**2*f(x))*f(x)/(x*(x**6*f(x)**3 + x**4*f(x)**2)) 等于 0
        'eq': Eq(f(x).diff(x) + (x**4*f(x)**2 + x**2*f(x))*f(x)/(x*(x**6*f(x)**3 + x**4*f(x)**2)), 0),
        # 解的列表
        'sol': [Eq(f(x), C1 + 1/(2*x**2))],
    },

    # 定义一个名为 'separable_reduced_07' 的字典项，包含以下内容：
    'separable_reduced_07': {
        # 方程式：f(x) 的导数加上 (f(x)**2)*f(x)/(x) 等于 0
        'eq': Eq(f(x).diff(x) + (f(x)**2)*f(x)/(x), 0),
        # 解的列表
        'sol': [
            Eq(f(x), -sqrt(2)*sqrt(1/(C1 + log(x)))/2),
            Eq(f(x), sqrt(2)*sqrt(1/(C1 + log(x)))/2)
        ],
    },

    # 定义一个名为 'separable_reduced_08' 的字典项，包含以下内容：
    'separable_reduced_08': {
        # 方程式：f(x) 的导数加上 (f(x)+3)*f(x)/(x*(f(x)+2)) 等于 0
        'eq': Eq(f(x).diff(x) + (f(x)+3)*f(x)/(x*(f(x)+2)), 0),
        # 解的列表
        'sol': [Eq(-log(f(x) + 3)/3 - 2*log(f(x))/3, C1 + log(x))],
        # 简化标志
        'simplify_flag': False,
        # XFAIL 标志
        'XFAIL': ['lie_group'], # It hangs.
    },

    # 定义一个名为 'separable_reduced_09' 的字典项，包含以下内容：
    'separable_reduced_09': {
        # 方程式：f(x) 的导数加上 (f(x)+3)*f(x)/x 等于 0
        'eq': Eq(f(x).diff(x) + (f(x)+3)*f(x)/x, 0),
        # 解的列表
        'sol': [Eq(f(x), 3/(C1*x**3 - 1))],
    },

    # 定义一个名为 'separable_reduced_10' 的字典项，包含以下内容：
    'separable_reduced_10': {
        # 方程式：f(x) 的导数加上 (f(x)**2+f(x))*f(x)/(x) 等于 0
        'eq': Eq(f(x).diff(x) + (f(x)**2+f(x))*f(x)/(x), 0),
        # 解的列表
        'sol': [Eq(- log(x) - log(f(x) + 1) + log(f(x)) + 1/f(x), C1)],
        # XFAIL 标志
        'XFAIL': ['lie_group'], # No algorithms are implemented to solve equation -C1 + x*(_y + 1)*exp(-1/_y)/_y
    },

    # 定义一个名为 'separable_reduced_11' 的字典项，包含以下内容：
    'separable_reduced_11': {
        # 方程式：f(x) 的导数加上 f(x) / (x**4*f(x) - x)
        'eq': f(x).diff(x) + (f(x) / (x**4*f(x) - x)),
        # 解的列表（未完整展示）
        'sol': [Eq(f(x), -sqrt(2)*sqrt(3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
# 以下是一个包含多个数学表达式的列表，用于表达函数 f(x) 的表达式
[
    # 第一个表达式
    Eq(f(x), 
       -3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) 
       + 2/x**6)/6,

    # 第二个表达式
    Eq(f(x), 
       -sqrt(2)*sqrt(-3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
       + 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) 
       + 4/x**6 - 4*sqrt(2)/(x**9*sqrt(3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
       - 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) + 2/x**6)))/6 
       + 1/(3*x**3)),

    # 第三个表达式
    Eq(f(x), 
       sqrt(2)*sqrt(3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
       - 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) 
       + 2/x**6)/6 
       - sqrt(2)*sqrt(-3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
       + 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) 
       + 4/x**6 + 4*sqrt(2)/(x**9*sqrt(3*3**Rational(1,3)*(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3)
       - 3*3**Rational(2,3)*exp(12*C1)/(sqrt((3*exp(12*C1) + x**(-12))*exp(24*C1)) - exp(12*C1)/x**6)**Rational(1,3) + 2/x**6)))/6 
       + 1/(3*x**3))
],

{
    # 包含关于此问题的一些说明和特性的字典
    'checkodesol_XFAIL': True,  # 这个测试在此处挂起
    'slow': True  # 此问题被标记为慢速处理
}
    'separable_reduced_12': {  # 定义字典中的键名为'separable_reduced_12'
        'eq': x**2*f(x)**2 + x*Derivative(f(x), x),  # 键名为'eq'的值是一个表达式 x**2*f(x)**2 + x*f'(x)，表示一个微分方程
        'sol': [Eq(f(x), 2*C1/(C1*x**2 - 1))],  # 键名为'sol'的值是包含一个方程Eq(f(x), 2*C1/(C1*x**2 - 1))的列表，表示方程的解
    },
    }
@_add_example_keys
def _get_examples_ode_sol_lie_group():
    # 定义符号变量 a, b, c
    a, b, c = symbols("a b c")
    # 返回一个字典，包含 'hint', 'func', 'examples' 三个键
    return {
            'hint': "lie_group",
            'func': f(x),
            'examples':{
    #Example 1-4 and 19-20 were from issue: https://github.com/sympy/sympy/issues/17322

    # 'lie_group_01' 示例
    'lie_group_01': {
        'eq': x*f(x).diff(x)*(f(x)+4) + (f(x)**2) -2*f(x)-2*x,
        'sol': [],
        'dsolve_too_slow': True,
        'checkodesol_too_slow': True,
    },

    # 'lie_group_02' 示例
    'lie_group_02': {
        'eq': x*f(x).diff(x)*(f(x)+4) + (f(x)**2) -2*f(x)-2*x,
        'sol': [],
        'dsolve_too_slow': True,
    },

    # 'lie_group_03' 示例
    'lie_group_03': {
        'eq': Eq(x**7*Derivative(f(x), x) + 5*x**3*f(x)**2 - (2*x**2 + 2)*f(x)**3, 0),
        'sol': [],
        'dsolve_too_slow': True,
    },

    # 'lie_group_04' 示例
    'lie_group_04': {
        'eq': f(x).diff(x) - (f(x) - x*log(x))**2/x**2 + log(x),
        'sol': [],
        'XFAIL': ['lie_group'],
    },

    # 'lie_group_05' 示例
    'lie_group_05': {
        'eq': f(x).diff(x)**2,
        'sol': [Eq(f(x), C1)],
        'XFAIL': ['factorable'],  #It raises Not Implemented error
    },

    # 'lie_group_06' 示例
    'lie_group_06': {
        'eq': Eq(f(x).diff(x), x**2*f(x)),
        'sol': [Eq(f(x), C1*exp(x**3)**Rational(1, 3))],
    },

    # 'lie_group_07' 示例
    'lie_group_07': {
        'eq': f(x).diff(x) + a*f(x) - c*exp(b*x),
        'sol': [Eq(f(x), Piecewise(((-C1*(a + b) + c*exp(x*(a + b)))*exp(-a*x)/(a + b),\
        Ne(a, -b)), ((-C1 + c*x)*exp(-a*x), True)))],
    },

    # 'lie_group_08' 示例
    'lie_group_08': {
        'eq': f(x).diff(x) + 2*x*f(x) - x*exp(-x**2),
        'sol': [Eq(f(x), (C1 + x**2/2)*exp(-x**2))],
    },

    # 'lie_group_09' 示例
    'lie_group_09': {
        'eq': (1 + 2*x)*(f(x).diff(x)) + 2 - 4*exp(-f(x)),
        'sol': [Eq(f(x), log(C1/(2*x + 1) + 2))],
    },

    # 'lie_group_10' 示例
    'lie_group_10': {
        'eq': x**2*(f(x).diff(x)) - f(x) + x**2*exp(x - (1/x)),
        'sol': [Eq(f(x), (C1 - exp(x))*exp(-1/x))],
        'XFAIL': ['factorable'], #It raises Recursion Error (maixmum depth exceeded)
    },

    # 'lie_group_11' 示例
    'lie_group_11': {
        'eq': x**2*f(x)**2 + x*Derivative(f(x), x),
        'sol': [Eq(f(x), 2/(C1 + x**2))],
    },

    # 'lie_group_12' 示例
    'lie_group_12': {
        'eq': diff(f(x),x) + 2*x*f(x) - x*exp(-x**2),
        'sol': [Eq(f(x), exp(-x**2)*(C1 + x**2/2))],
    },

    # 'lie_group_13' 示例
    'lie_group_13': {
        'eq': diff(f(x),x) + f(x)*cos(x) - exp(2*x),
        'sol': [Eq(f(x), exp(-sin(x))*(C1 + Integral(exp(2*x)*exp(sin(x)), x)))],
    },

    # 'lie_group_14' 示例
    'lie_group_14': {
        'eq': diff(f(x),x) + f(x)*cos(x) - sin(2*x)/2,
        'sol': [Eq(f(x), C1*exp(-sin(x)) + sin(x) - 1)],
    },

    # 'lie_group_15' 示例
    'lie_group_15': {
        'eq': x*diff(f(x),x) + f(x) - x*sin(x),
        'sol': [Eq(f(x), (C1 - x*cos(x) + sin(x))/x)],
    },

    # 'lie_group_16' 示例
    'lie_group_16': {
        'eq': x*diff(f(x),x) - f(x) - x/log(x),
        'sol': [Eq(f(x), x*(C1 + log(log(x))))],
    },

    # 'lie_group_17' 示例
    'lie_group_17': {
        'eq': (f(x).diff(x)-f(x)) * (f(x).diff(x)+f(x)),
        'sol': [Eq(f(x), C1*exp(x)), Eq(f(x), C1*exp(-x))],
    },
    {
        'lie_group_18': {
            'eq': f(x).diff(x) * (f(x).diff(x) - f(x)),
            'sol': [Eq(f(x), C1*exp(x)), Eq(f(x), C1)],
        },
    
        'lie_group_19': {
            'eq': (f(x).diff(x)-f(x)) * (f(x).diff(x)+f(x)),
            'sol': [Eq(f(x), C1*exp(-x)), Eq(f(x), C1*exp(x))],
        },
    
        'lie_group_20': {
            'eq': f(x).diff(x)*(f(x).diff(x)+f(x)),
            'sol': [Eq(f(x), C1), Eq(f(x), C1*exp(-x))],
        },
    }
    
    
    
    # 第一个分组的Lie群微分方程和解
    'lie_group_18': {
        # 微分方程: f(x)的导数乘以(f(x)的导数减去f(x)本身
        'eq': f(x).diff(x) * (f(x).diff(x) - f(x)),
        # 解集: 包括f(x)的解的表达式
        'sol': [Eq(f(x), C1*exp(x)), Eq(f(x), C1)],
    },
    
    # 第二个分组的Lie群微分方程和解
    'lie_group_19': {
        # 微分方程: (f(x)的导数减去f(x))乘以(f(x)的导数加上f(x))
        'eq': (f(x).diff(x)-f(x)) * (f(x).diff(x)+f(x)),
        # 解集: 包括f(x)的解的表达式
        'sol': [Eq(f(x), C1*exp(-x)), Eq(f(x), C1*exp(x))],
    },
    
    # 第三个分组的Lie群微分方程和解
    'lie_group_20': {
        # 微分方程: f(x)的导数乘以(f(x)的导数加上f(x))
        'eq': f(x).diff(x)*(f(x).diff(x)+f(x)),
        # 解集: 包括f(x)的解的表达式
        'sol': [Eq(f(x), C1), Eq(f(x), C1*exp(-x))],
    },
    
    
    这段代码定义了三组Lie群的微分方程和相应的解集，每组包含一个微分方程和一组解的表达式。
@_add_example_keys
def _get_examples_ode_sol_2nd_linear_airy():
    # 返回一个包含 2nd_linear_airy 示例的字典，包括提示、函数和示例
    return {
            'hint': "2nd_linear_airy",
            'func': f(x),  # 函数 f(x)
            'examples':{
    '2nd_lin_airy_01': {
        'eq': f(x).diff(x, 2) - x*f(x),  # 给定的微分方程
        'sol': [Eq(f(x), C1*airyai(x) + C2*airybi(x))],  # 方程的解
    },

    '2nd_lin_airy_02': {
        'eq': f(x).diff(x, 2) + 2*x*f(x),  # 给定的微分方程
        'sol': [Eq(f(x), C1*airyai(-2**(S(1)/3)*x) + C2*airybi(-2**(S(1)/3)*x))],  # 方程的解
    },
    }
    }


@_add_example_keys
def _get_examples_ode_sol_nth_linear_constant_coeff_homogeneous():
    # 从《Ordinary Differential Equations, Tenenbaum and Pollard, pg. 220》中的 Exercise 20 获取示例
    a = Symbol('a', positive=True)
    k = Symbol('k', real=True)
    # 计算不同方程的根
    r1, r2, r3, r4, r5 = [rootof(x**5 + 11*x - 2, n) for n in range(5)]
    r6, r7, r8, r9, r10 = [rootof(x**5 - 3*x + 1, n) for n in range(5)]
    r11, r12, r13, r14, r15 = [rootof(x**5 - 100*x**3 + 1000*x + 1, n) for n in range(5)]
    r16, r17, r18, r19, r20 = [rootof(x**5 - x**4 + 10, n) for n in range(5)]
    r21, r22, r23, r24, r25 = [rootof(x**5 - x + 1, n) for n in range(5)]
    E = exp(1)
    # 返回一个包含 nth_linear_constant_coeff_homogeneous 示例的字典，包括提示、函数和示例
    return {
            'hint': "nth_linear_constant_coeff_homogeneous",
            'func': f(x),  # 函数 f(x)
            'examples':{
    'lin_const_coeff_hom_01': {
        'eq': f(x).diff(x, 2) + 2*f(x).diff(x),  # 给定的微分方程
        'sol': [Eq(f(x), C1 + C2*exp(-2*x))],  # 方程的解
    },

    'lin_const_coeff_hom_02': {
        'eq': f(x).diff(x, 2) - 3*f(x).diff(x) + 2*f(x),  # 给定的微分方程
        'sol': [Eq(f(x), (C1 + C2*exp(x))*exp(x))],  # 方程的解
    },

    'lin_const_coeff_hom_03': {
        'eq': f(x).diff(x, 2) - f(x),  # 给定的微分方程
        'sol': [Eq(f(x), C1*exp(-x) + C2*exp(x))],  # 方程的解
    },

    'lin_const_coeff_hom_04': {
        'eq': f(x).diff(x, 3) + f(x).diff(x, 2) - 6*f(x).diff(x),  # 给定的微分方程
        'sol': [Eq(f(x), C1 + C2*exp(-3*x) + C3*exp(2*x))],  # 方程的解
        'slow': True,  # 解求解缓慢的标记
    },

    'lin_const_coeff_hom_05': {
        'eq': 6*f(x).diff(x, 2) - 11*f(x).diff(x) + 4*f(x),  # 给定的微分方程
        'sol': [Eq(f(x), C1*exp(x/2) + C2*exp(x*Rational(4, 3)))],  # 方程的解
        'slow': True,  # 解求解缓慢的标记
    },

    'lin_const_coeff_hom_06': {
        'eq': Eq(f(x).diff(x, 2) + 2*f(x).diff(x) - f(x), 0),  # 给定的微分方程
        'sol': [Eq(f(x), C1*exp(x*(-1 + sqrt(2))) + C2*exp(-x*(sqrt(2) + 1)))],  # 方程的解
        'slow': True,  # 解求解缓慢的标记
    },

    'lin_const_coeff_hom_07': {
        'eq': diff(f(x), x, 3) + diff(f(x), x, 2) - 10*diff(f(x), x) - 6*f(x),  # 给定的微分方程
        'sol': [Eq(f(x), C1*exp(3*x) + C3*exp(-x*(2 + sqrt(2))) + C2*exp(x*(-2 + sqrt(2))))],  # 方程的解
        'slow': True,  # 解求解缓慢的标记
    },

    'lin_const_coeff_hom_08': {
        'eq': f(x).diff(x, 4) - f(x).diff(x, 3) - 4*f(x).diff(x, 2) + \
        4*f(x).diff(x),  # 给定的微分方程
        'sol': [Eq(f(x), C1 + C2*exp(-2*x) + C3*exp(x) + C4*exp(2*x))],  # 方程的解
        'slow': True,  # 解求解缓慢的标记
    },

    'lin_const_coeff_hom_09': {
        'eq': f(x).diff(x, 4) + 4*f(x).diff(x, 3) + f(x).diff(x, 2) - \
        4*f(x).diff(x) - 2*f(x),  # 给定的微分方程
        'sol': [Eq(f(x), C3*exp(-x) + C4*exp(x) + (C1*exp(-sqrt(2)*x) + C2*exp(sqrt(2)*x))*exp(-2*x))],  # 方程的解
        'slow': True,  # 解求解缓慢的标记
    },
    'lin_const_coeff_hom_10': {
        'eq': f(x).diff(x, 4) - a**2*f(x),
        'sol': [Eq(f(x), C1*exp(-sqrt(a)*x) + C2*exp(sqrt(a)*x) + C3*sin(sqrt(a)*x) + C4*cos(sqrt(a)*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_11': {
        'eq': f(x).diff(x, 2) - 2*k*f(x).diff(x) - 2*f(x),
        'sol': [Eq(f(x), C1*exp(x*(k - sqrt(k**2 + 2))) + C2*exp(x*(k + sqrt(k**2 + 2))))],
        'slow': True,
    },

    'lin_const_coeff_hom_12': {
        'eq': f(x).diff(x, 2) + 4*k*f(x).diff(x) - 12*k**2*f(x),
        'sol': [Eq(f(x), C1*exp(-6*k*x) + C2*exp(2*k*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_13': {
        'eq': f(x).diff(x, 4),
        'sol': [Eq(f(x), C1 + C2*x + C3*x**2 + C4*x**3)],
        'slow': True,
    },

    'lin_const_coeff_hom_14': {
        'eq': f(x).diff(x, 2) + 4*f(x).diff(x) + 4*f(x),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(-2*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_15': {
        'eq': 3*f(x).diff(x, 3) + 5*f(x).diff(x, 2) + f(x).diff(x) - f(x),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(-x) + C3*exp(x/3))],
        'slow': True,
    },

    'lin_const_coeff_hom_16': {
        'eq': f(x).diff(x, 3) - 6*f(x).diff(x, 2) + 12*f(x).diff(x) - 8*f(x),
        'sol': [Eq(f(x), (C1 + x*(C2 + C3*x))*exp(2*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_17': {
        'eq': f(x).diff(x, 2) - 2*a*f(x).diff(x) + a**2*f(x),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(a*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_18': {
        'eq': f(x).diff(x, 4) + 3*f(x).diff(x, 3),
        'sol': [Eq(f(x), C1 + C2*x + C3*x**2 + C4*exp(-3*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_19': {
        'eq': f(x).diff(x, 4) - 2*f(x).diff(x, 2),
        'sol': [Eq(f(x), C1 + C2*x + C3*exp(-sqrt(2)*x) + C4*exp(sqrt(2)*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_20': {
        'eq': f(x).diff(x, 4) + 2*f(x).diff(x, 3) - 11*f(x).diff(x, 2) - \
        12*f(x).diff(x) + 36*f(x),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(-3*x) + (C3 + C4*x)*exp(2*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_21': {
        'eq': 36*f(x).diff(x, 4) - 37*f(x).diff(x, 2) + 4*f(x).diff(x) + 5*f(x),
        'sol': [Eq(f(x), C1*exp(-x) + C2*exp(-x/3) + C3*exp(x/2) + C4*exp(x*Rational(5, 6)))],
        'slow': True,
    },

    'lin_const_coeff_hom_22': {
        'eq': f(x).diff(x, 4) - 8*f(x).diff(x, 2) + 16*f(x),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(-2*x) + (C3 + C4*x)*exp(2*x))],
        'slow': True,
    },

    'lin_const_coeff_hom_23': {
        'eq': f(x).diff(x, 2) - 2*f(x).diff(x) + 5*f(x),
        'sol': [Eq(f(x), (C1*sin(2*x) + C2*cos(2*x))*exp(x))],
        'slow': True,
    },

    'lin_const_coeff_hom_24': {
        'eq': f(x).diff(x, 2) - f(x).diff(x) + f(x),
        'sol': [Eq(f(x), (C1*sin(x*sqrt(3)/2) + C2*cos(x*sqrt(3)/2))*exp(x/2))],
        'slow': True,
    },
    # 线性常系数齐次微分方程示例 25
    'lin_const_coeff_hom_25': {
        # 方程：f(x).diff(x, 4) + 5*f(x).diff(x, 2) + 6*f(x)
        'eq': f(x).diff(x, 4) + 5*f(x).diff(x, 2) + 6*f(x),
        # 解：[Eq(f(x), C1*sin(sqrt(2)*x) + C2*sin(sqrt(3)*x) + C3*cos(sqrt(2)*x) + C4*cos(sqrt(3)*x))]
        'sol': [Eq(f(x), C1*sin(sqrt(2)*x) + C2*sin(sqrt(3)*x) + C3*cos(sqrt(2)*x) + C4*cos(sqrt(3)*x))],
        # 慢速解求解标志
        'slow': True,
    },

    # 线性常系数齐次微分方程示例 26
    'lin_const_coeff_hom_26': {
        # 方程：f(x).diff(x, 2) - 4*f(x).diff(x) + 20*f(x)
        'eq': f(x).diff(x, 2) - 4*f(x).diff(x) + 20*f(x),
        # 解：[Eq(f(x), (C1*sin(4*x) + C2*cos(4*x))*exp(2*x))]
        'sol': [Eq(f(x), (C1*sin(4*x) + C2*cos(4*x))*exp(2*x))],
        'slow': True,
    },

    # 线性常系数齐次微分方程示例 27
    'lin_const_coeff_hom_27': {
        # 方程：f(x).diff(x, 4) + 4*f(x).diff(x, 2) + 4*f(x)
        'eq': f(x).diff(x, 4) + 4*f(x).diff(x, 2) + 4*f(x),
        # 解：[Eq(f(x), (C1 + C2*x)*sin(x*sqrt(2)) + (C3 + C4*x)*cos(x*sqrt(2)))]
        'sol': [Eq(f(x), (C1 + C2*x)*sin(x*sqrt(2)) + (C3 + C4*x)*cos(x*sqrt(2)))],
        'slow': True,
    },

    # 线性常系数齐次微分方程示例 28
    'lin_const_coeff_hom_28': {
        # 方程：f(x).diff(x, 3) + 8*f(x)
        'eq': f(x).diff(x, 3) + 8*f(x),
        # 解：[Eq(f(x), (C1*sin(x*sqrt(3)) + C2*cos(x*sqrt(3)))*exp(x) + C3*exp(-2*x))]
        'sol': [Eq(f(x), (C1*sin(x*sqrt(3)) + C2*cos(x*sqrt(3)))*exp(x) + C3*exp(-2*x))],
        'slow': True,
    },

    # 线性常系数齐次微分方程示例 29
    'lin_const_coeff_hom_29': {
        # 方程：f(x).diff(x, 4) + 4*f(x).diff(x, 2)
        'eq': f(x).diff(x, 4) + 4*f(x).diff(x, 2),
        # 解：[Eq(f(x), C1 + C2*x + C3*sin(2*x) + C4*cos(2*x))]
        'sol': [Eq(f(x), C1 + C2*x + C3*sin(2*x) + C4*cos(2*x))],
        'slow': True,
    },

    # 线性常系数齐次微分方程示例 30
    'lin_const_coeff_hom_30': {
        # 方程：f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x)
        'eq': f(x).diff(x, 5) + 2*f(x).diff(x, 3) + f(x).diff(x),
        # 解：[Eq(f(x), C1 + (C2 + C3*x)*sin(x) + (C4 + C5*x)*cos(x))]
        'sol': [Eq(f(x), C1 + (C2 + C3*x)*sin(x) + (C4 + C5*x)*cos(x))],
        'slow': True,
    },

    # 线性常系数齐次微分方程示例 31
    'lin_const_coeff_hom_31': {
        # 方程：f(x).diff(x, 4) + f(x).diff(x, 2) + f(x)
        'eq': f(x).diff(x, 4) + f(x).diff(x, 2) + f(x),
        # 解：[Eq(f(x), (C1*sin(sqrt(3)*x/2) + C2*cos(sqrt(3)*x/2))*exp(-x/2)
        #      + (C3*sin(sqrt(3)*x/2) + C4*cos(sqrt(3)*x/2))*exp(x/2))]
        'sol': [Eq(f(x), (C1*sin(sqrt(3)*x/2) + C2*cos(sqrt(3)*x/2))*exp(-x/2)
               + (C3*sin(sqrt(3)*x/2) + C4*cos(sqrt(3)*x/2))*exp(x/2))],
        'slow': True,
    },

    # 线性常系数齐次微分方程示例 32
    'lin_const_coeff_hom_32': {
        # 方程：f(x).diff(x, 4) + 4*f(x).diff(x, 2) + f(x)
        'eq': f(x).diff(x, 4) + 4*f(x).diff(x, 2) + f(x),
        # 解：[Eq(f(x), C1*sin(x*sqrt(-sqrt(3) + 2)) + C2*sin(x*sqrt(sqrt(3) + 2))
        #      + C3*cos(x*sqrt(-sqrt(3) + 2)) + C4*cos(x*sqrt(sqrt(3) + 2)))]
        'sol': [Eq(f(x), C1*sin(x*sqrt(-sqrt(3) + 2)) + C2*sin(x*sqrt(sqrt(3) + 2))
               + C3*cos(x*sqrt(-sqrt(3) + 2)) + C4*cos(x*sqrt(sqrt(3) + 2)))],
        'slow': True,
    },

    # 一个实根，两个共轭复数对
    'lin_const_coeff_hom_33': {
        # 方程：f(x).diff(x, 5) + 11*f(x).diff(x) - 2*f(x)
        'eq': f(x).diff(x, 5) + 11*f(x).diff(x) - 2*f(x),
        # 解：[Eq(f(x), C5*exp(r1*x) + exp(re(r2)*x) * (C1*sin(im(r2)*x) + C2*cos(im(r2)*x))
        #      + exp(re(r4)*x) * (C3*sin(im(r4)*x) + C4*cos(im(r4)*x)))]
        'sol': [Eq(f(x), C5*exp(r1*x) + exp(re(r2)*x) * (C1*sin(im(r2)*x) + C2*cos(im(r2)*x))
               + exp(re(r4)*x) * (C3*sin(im(r4)*x) + C4*cos(im(r4)*x)))],
        # 解法检查失败标志
        'checkodesol_XFAIL': True,
    },

    # 三个实根，一个共轭复数对
    'lin_const_coeff_hom_34': {
        # 方程：f(x).diff(x, 5) - 3*f(x).diff(x) + f(x)
        'eq': f(x).diff(x, 5) - 3*f(x).diff(x) + f(x),
        # 解：[Eq(f(x), C3*exp(r6*x) + C4*exp(r7*x) + C5*exp(r8*x)
        #      + exp(re(r9)*x) * (C1*sin(im(r9)*x) + C2*cos(im(r9)*x)))]
        'sol': [Eq(f(x), C3*exp(r6*x) + C4*exp(r7*x) + C5*exp(r8*x)
               + exp(re(r9)*x) * (C1*sin(im(r9)*x) + C2*cos(im(r9)*x)))],
        # 解法检查失败标志
        'checkodesol_XFAIL': True,
    },

    # 五个不同的实根
    'lin
    'lin_const_coeff_hom_36': {
        'eq': f(x).diff(x, 6) - 6*f(x).diff(x, 5) + 5*f(x).diff(x, 4) + 10*f(x).diff(x) - 50 * f(x),
        'sol': [Eq(f(x),
        C5*exp(5*x)
        + C6*exp(x*r16)
        + exp(re(r17)*x) * (C1*sin(im(r17)*x) + C2*cos(im(r17)*x))
        + exp(re(r19)*x) * (C3*sin(im(r19)*x) + C4*cos(im(r19)*x)))],
        'checkodesol_XFAIL':True, #It Hangs
    },

    # Five double roots (this is (x**5 - x + 1)**2)
    'lin_const_coeff_hom_37': {
        'eq': f(x).diff(x, 10) - 2*f(x).diff(x, 6) + 2*f(x).diff(x, 5)
        + f(x).diff(x, 2) - 2*f(x).diff(x, 1) + f(x),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(x*r21) + (-((C3 + C4*x)*sin(x*im(r22)))
        + (C5 + C6*x)*cos(x*im(r22)))*exp(x*re(r22)) + (-((C7 + C8*x)*sin(x*im(r24)))
        + (C10*x + C9)*cos(x*im(r24)))*exp(x*re(r24)))],
        'checkodesol_XFAIL':True, #It Hangs
    },

    'lin_const_coeff_hom_38': {
        'eq': Eq(sqrt(2) * f(x).diff(x,x,x) + f(x).diff(x), 0),
        'sol': [Eq(f(x), C1 + C2*sin(2**Rational(3, 4)*x/2) + C3*cos(2**Rational(3, 4)*x/2))],
    },

    'lin_const_coeff_hom_39': {
        'eq': Eq(E * f(x).diff(x,x,x) + f(x).diff(x), 0),
        'sol': [Eq(f(x), C1 + C2*sin(x/sqrt(E)) + C3*cos(x/sqrt(E)))],
    },

    'lin_const_coeff_hom_40': {
        'eq': Eq(pi * f(x).diff(x,x,x) + f(x).diff(x), 0),
        'sol': [Eq(f(x), C1 + C2*sin(x/sqrt(pi)) + C3*cos(x/sqrt(pi)))],
    },

    'lin_const_coeff_hom_41': {
        'eq': Eq(I * f(x).diff(x,x,x) + f(x).diff(x), 0),
        'sol': [Eq(f(x), C1 + C2*exp(-sqrt(I)*x) + C3*exp(sqrt(I)*x))],
    },

    'lin_const_coeff_hom_42': {
        'eq': f(x).diff(x, x) + y*f(x),
        'sol': [Eq(f(x), C1*exp(-x*sqrt(-y)) + C2*exp(x*sqrt(-y)))],
    },

    'lin_const_coeff_hom_43': {
        'eq': Eq(9*f(x).diff(x, x) + f(x), 0),
        'sol': [Eq(f(x), C1*sin(x/3) + C2*cos(x/3))],
    },

    'lin_const_coeff_hom_44': {
        'eq': Eq(9*f(x).diff(x, x), f(x)),
        'sol': [Eq(f(x), C1*exp(-x/3) + C2*exp(x/3))],
    },

    'lin_const_coeff_hom_45': {
        'eq': Eq(f(x).diff(x, x) - 3*diff(f(x), x) + 2*f(x), 0),
        'sol': [Eq(f(x), (C1 + C2*exp(x))*exp(x))],
    },

    'lin_const_coeff_hom_46': {
        'eq': Eq(f(x).diff(x, x) - 4*diff(f(x), x) + 4*f(x), 0),
        'sol': [Eq(f(x), (C1 + C2*x)*exp(2*x))],
    },

    # Type: 2nd order, constant coefficients (two real equal roots)
    'lin_const_coeff_hom_47': {
        'eq': Eq(f(x).diff(x, x) + 2*diff(f(x), x) + 3*f(x), 0),
        'sol': [Eq(f(x), (C1*sin(x*sqrt(2)) + C2*cos(x*sqrt(2)))*exp(-x))],
    },

    #These were from issue: https://github.com/sympy/sympy/issues/6247
    'lin_const_coeff_hom_48': {
        'eq': f(x).diff(x, x) + 4*f(x),
        'sol': [Eq(f(x), C1*sin(2*x) + C2*cos(2*x))],
    },
}
# 装饰器函数，用于添加示例键到函数返回的字典中
@_add_example_keys
# 返回一个包含ODE解示例的字典
def _get_examples_ode_sol_1st_homogeneous_coeff_subs_dep_div_indep():
    return {
            # 提示信息，指示这些示例是关于1st_homogeneous_coeff_subs_dep_div_indep的
            'hint': "1st_homogeneous_coeff_subs_dep_div_indep",
            # 函数变量，但这里的'f(x)'可能是占位符，因为它没有定义
            'func': f(x),
            # 示例字典包含多个具体的ODE例子
            'examples':{
    'dep_div_indep_01': {
        # 方程示例，包含f(x)和其导数的复杂表达式
        'eq': f(x)/x*cos(f(x)/x) - (x/f(x)*sin(f(x)/x) + cos(f(x)/x))*f(x).diff(x),
        # 方程的解，使用常数C1和对数log(x)来表示
        'sol': [Eq(log(x), C1 - log(f(x)*sin(f(x)/x)/x))],
        # 标记这个示例运行速度慢
        'slow': True
    },

    # indep_div_dep有一个更简单的解法，但是运行速度太慢了。
    'dep_div_indep_02': {
        # 包含f(x)和其导数的线性方程
        'eq': x*f(x).diff(x) - f(x) - x*sin(f(x)/x),
        # 方程的解，使用对数log(x)的复杂表达式
        'sol': [Eq(log(x), log(C1) + log(cos(f(x)/x) - 1)/2 - log(cos(f(x)/x) + 1)/2)],
        # 简化标记为假
        'simplify_flag': False,
    },

    'dep_div_indep_03': {
        # 包含f(x)和其导数的指数和三角函数组合方程
        'eq': x*exp(f(x)/x) - f(x)*sin(f(x)/x) + x*sin(f(x)/x)*f(x).diff(x),
        # 方程的解，包含指数和三角函数的复杂表达式
        'sol': [Eq(log(x), C1 + exp(-f(x)/x)*sin(f(x)/x)/2 + exp(-f(x)/x)*cos(f(x)/x)/2)],
        # 标记这个示例运行速度慢
        'slow': True
    },

    'dep_div_indep_04': {
        # 包含f(x)和其导数的复杂非线性方程
        'eq': f(x).diff(x) - f(x)/x + 1/sin(f(x)/x),
        # 方程的解，使用常数C1和对数log(x)的表达式
        'sol': [Eq(f(x), x*(-acos(C1 + log(x)) + 2*pi)), Eq(f(x), x*acos(C1 + log(x)))],
        # 标记这个示例运行速度慢
        'slow': True
    },

    # 前面的代码使用这些其他解法进行测试：
    # example5_solb = Eq(f(x), log(log(C1/x)**(-x)))
    'dep_div_indep_05': {
        # 包含f(x)和其导数的指数方程，具有对数和分数幂的复杂表达式
        'eq': x*exp(f(x)/x) + f(x) - x*f(x).diff(x),
        # 方程的解，使用对数和分数幂的复杂表达式
        'sol': [Eq(f(x), log((1/(C1 - log(x)))**x))],
        # 由于 **x 的存在，标记这个示例不符合预期（XFAIL）
        'checkodesol_XFAIL': True,
    },
    }
    }

# 装饰器函数，用于添加示例键到函数返回的字典中
@_add_example_keys
# 返回一个包含ODE解示例的字典
def _get_examples_ode_sol_linear_coefficients():
    return {
            # 提示信息，指示这些示例是关于linear_coefficients的
            'hint': "linear_coefficients",
            # 函数变量，但这里的'f(x)'可能是占位符，因为它没有定义
            'func': f(x),
            # 示例字典包含一个线性系数的ODE例子
            'examples':{
    'linear_coeff_01': {
        # 包含f(x)和其导数的线性方程
        'eq': f(x).diff(x) + (3 + 2*f(x))/(x + 3),
        # 方程的解，使用常数C1和有理数Rational(3, 2)的表达式
        'sol': [Eq(f(x), C1/(x**2 + 6*x + 9) - Rational(3, 2))],
    },
    }
    }

# 装饰器函数，用于添加示例键到函数返回的字典中
@_add_example_keys
# 返回一个包含ODE解示例的字典
def _get_examples_ode_sol_1st_homogeneous_coeff_best():
    return {
            # 提示信息，指示这些示例是关于1st_homogeneous_coeff_best的
            'hint': "1st_homogeneous_coeff_best",
            # 函数变量，但这里的'f(x)'可能是占位符，因为它没有定义
            'func': f(x),
            # 示例字典包含多个最佳1st_homogeneous_coeff的ODE例子
            'examples':{
    # 前面的代码使用这个其他解法进行测试：
    # example1_solb = Eq(-f(x)/(1 + log(x/f(x))), C1)
    '1st_homogeneous_coeff_best_01': {
        # 包含f(x)和其导数的复杂方程，包含Lambert W函数的表达式
        'eq': f(x) + (x*log(f(x)/x) - 2*x)*diff(f(x), x),
        # 方程的解，使用常数C1和Lambert W函数的表达式
        'sol': [Eq(f(x), -exp(C1)*LambertW(-x*exp(-C1 + 1)))],
        # 由于Lambert W函数的存在，标记这个示例不符合预期（XFAIL）
        'checkodesol_XFAIL': True,
    },

    '1st_homogeneous_coeff_best_02': {
        # 包含f(x)和其导数的复杂方程，包含指数和对数的表达式
        'eq': 2*f(x)*exp(x/f(x)) + f(x)*f(x).diff(x) - 2*x*exp(x/f(x))*f(x).diff(x),
        # 方程的解，使用对数的表达式
        'sol': [Eq(log(f(x)), C1 - 2*exp(x/f(x)))],
    },

    # 前面的代码使用这个其他解法进行测试：
    # example3_solb = Eq(log(C1*x*sqrt(1/x)*sqrt(f(x))) + x**2/(2*f(x)**2), 0)
    '1st_homogeneous_coeff_best_03': {
        # 包含f(x)和其导数的复杂方程，包含Lambert W函数的表达式
        'eq': 2*x**2*f(x) + f(x)**3 + (x*f(x)**2 - 2*x**3)*f(x).diff(x),
        # 方程的解，使用常数C1和Lambert W函数的表达式
        'sol': [Eq(f(x), exp(2*C1 + LambertW(-2*x**4*exp(-4*C1))/2)/x)],
        # 由于Lambert W函数的存在，标记这个示例不符合预期（XFAIL）
        'checkodesol_XFAIL': True,
    },
    # '1st_homogeneous_coeff_best_04' 的注释
    '1st_homogeneous_coeff_best_04': {
        # 方程式，表达式为 (x + sqrt(f(x)**2 - x*f(x)))*f(x).diff(x) - f(x)
        'eq': (x + sqrt(f(x)**2 - x*f(x)))*f(x).diff(x) - f(x),
        # 解的列表，包含方程的解 Eq(log(f(x)), C1 - 2*sqrt(-x/f(x) + 1))
        'sol': [Eq(log(f(x)), C1 - 2*sqrt(-x/f(x) + 1))],
        # 标记，表示解的推导较慢
        'slow': True,
    },

    # '1st_homogeneous_coeff_best_05' 的注释
    '1st_homogeneous_coeff_best_05': {
        # 方程式，表达式为 x + f(x) - (x - f(x))*f(x).diff(x)
        'eq': x + f(x) - (x - f(x))*f(x).diff(x),
        # 解的列表，包含方程的解 Eq(log(x), C1 - log(sqrt(1 + f(x)**2/x**2)) + atan(f(x)/x))
        'sol': [Eq(log(x), C1 - log(sqrt(1 + f(x)**2/x**2)) + atan(f(x)/x))],
    },

    # '1st_homogeneous_coeff_best_06' 的注释
    '1st_homogeneous_coeff_best_06': {
        # 方程式，表达式为 x*f(x).diff(x) - f(x) - x*sin(f(x)/x)
        'eq': x*f(x).diff(x) - f(x) - x*sin(f(x)/x),
        # 解的列表，包含方程的解 Eq(f(x), 2*x*atan(C1*x))
        'sol': [Eq(f(x), 2*x*atan(C1*x))],
    },

    # '1st_homogeneous_coeff_best_07' 的注释
    '1st_homogeneous_coeff_best_07': {
        # 方程式，表达式为 x**2 + f(x)**2 - 2*x*f(x)*f(x).diff(x)
        'eq': x**2 + f(x)**2 - 2*x*f(x)*f(x).diff(x),
        # 解的列表，包含方程的解 Eq(f(x), -sqrt(x*(C1 + x))) 和 Eq(f(x), sqrt(x*(C1 + x)))
        'sol': [Eq(f(x), -sqrt(x*(C1 + x))), Eq(f(x), sqrt(x*(C1 + x)))],
    },

    # '1st_homogeneous_coeff_best_08' 的注释
    '1st_homogeneous_coeff_best_08': {
        # 方程式，表达式为 f(x)**2 + (x*sqrt(f(x)**2 - x**2) - x*f(x))*f(x).diff(x)
        'eq': f(x)**2 + (x*sqrt(f(x)**2 - x**2) - x*f(x))*f(x).diff(x),
        # 解的列表，包含方程的解 Eq(f(x), -C1*sqrt(-x/(x - 2*C1))) 和 Eq(f(x), C1*sqrt(-x/(x - 2*C1)))
        'sol': [Eq(f(x), -C1*sqrt(-x/(x - 2*C1))), Eq(f(x), C1*sqrt(-x/(x - 2*C1)))],
        # 标记，表示解在一定范围内是有效的
        'checkodesol_XFAIL': True  # solutions are valid in a range
    },
}
# 定义函数 _get_all_examples()，用于获取所有的示例列表
def _get_all_examples():
    # 将各种类型的微分方程求解示例列表连接成一个总列表
    all_examples = _get_examples_ode_sol_euler_homogeneous + \
    _get_examples_ode_sol_euler_undetermined_coeff + \
    _get_examples_ode_sol_euler_var_para + \
    _get_examples_ode_sol_factorable + \
    _get_examples_ode_sol_bernoulli + \
    _get_examples_ode_sol_nth_algebraic + \
    _get_examples_ode_sol_riccati + \
    _get_examples_ode_sol_1st_linear + \
    _get_examples_ode_sol_1st_exact + \
    _get_examples_ode_sol_almost_linear + \
    _get_examples_ode_sol_nth_order_reducible + \
    _get_examples_ode_sol_nth_linear_undetermined_coefficients + \
    _get_examples_ode_sol_liouville + \
    _get_examples_ode_sol_separable + \
    _get_examples_ode_sol_1st_rational_riccati + \
    _get_examples_ode_sol_nth_linear_var_of_parameters + \
    _get_examples_ode_sol_2nd_linear_bessel + \
    _get_examples_ode_sol_2nd_2F1_hypergeometric + \
    _get_examples_ode_sol_2nd_nonlinear_autonomous_conserved + \
    _get_examples_ode_sol_separable_reduced + \
    _get_examples_ode_sol_lie_group + \
    _get_examples_ode_sol_2nd_linear_airy + \
    _get_examples_ode_sol_nth_linear_constant_coeff_homogeneous +\
    _get_examples_ode_sol_1st_homogeneous_coeff_best +\
    _get_examples_ode_sol_1st_homogeneous_coeff_subs_dep_div_indep +\
    _get_examples_ode_sol_linear_coefficients

    # 返回合并后的所有示例列表
    return all_examples
```