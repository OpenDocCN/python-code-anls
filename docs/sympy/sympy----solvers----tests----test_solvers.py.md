# `D:\src\scipysrc\sympy\sympy\solvers\tests\test_solvers.py`

```
from sympy.assumptions.ask import (Q, ask)  # 导入 Q 和 ask 符号，用于符号推理
from sympy.core.add import Add  # 导入加法表达式类
from sympy.core.containers import Tuple  # 导入元组类
from sympy.core.function import (Derivative, Function, diff)  # 导入导数、函数和求导函数
from sympy.core.mod import Mod  # 导入取模运算类
from sympy.core.mul import Mul  # 导入乘法表达式类
from sympy.core import (GoldenRatio, TribonacciConstant)  # 导入黄金比例和 Tribonacci 常数
from sympy.core.numbers import (E, Float, I, Rational, oo, pi)  # 导入常见数学常数
from sympy.core.relational import (Eq, Gt, Lt, Ne)  # 导入关系运算符类
from sympy.core.singleton import S  # 导入单例符号
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)  # 导入符号类和符号生成函数
from sympy.core.sympify import sympify  # 导入 sympify 函数，用于符号转换
from sympy.functions.combinatorial.factorials import binomial  # 导入二项式系数函数
from sympy.functions.elementary.complexes import (Abs, arg, conjugate, im, re)  # 导入复数相关函数
from sympy.functions.elementary.exponential import (LambertW, exp, log)  # 导入指数和对数函数
from sympy.functions.elementary.hyperbolic import (atanh, cosh, sinh, tanh)  # 导入双曲函数
from sympy.functions.elementary.integers import floor  # 导入向下取整函数
from sympy.functions.elementary.miscellaneous import (cbrt, root, sqrt)  # 导入开方和立方根函数
from sympy.functions.elementary.piecewise import Piecewise  # 导入分段函数
from sympy.functions.elementary.trigonometric import (acos, asin, atan, atan2, cos, sec, sin, tan)  # 导入三角函数
from sympy.functions.special.error_functions import (erf, erfc, erfcinv, erfinv)  # 导入误差函数
from sympy.integrals.integrals import Integral  # 导入积分函数
from sympy.logic.boolalg import (And, Or)  # 导入布尔逻辑运算类
from sympy.matrices.dense import Matrix  # 导入密集矩阵类
from sympy.matrices import SparseMatrix  # 导入稀疏矩阵类
from sympy.polys.polytools import Poly  # 导入多项式操作类
from sympy.printing.str import sstr  # 导入打印字符串函数
from sympy.simplify.radsimp import denom  # 导入分母简化函数
from sympy.solvers.solvers import (nsolve, solve, solve_linear)  # 导入数值求解函数和线性方程求解函数

from sympy.core.function import nfloat  # 导入浮点数函数
from sympy.solvers import solve_linear_system, solve_linear_system_LU, \
    solve_undetermined_coeffs  # 导入线性系统求解函数和未定系数求解函数
from sympy.solvers.bivariate import _filtered_gens, _solve_lambert, _lambert  # 导入双变量求解和 Lambert W 函数
from sympy.solvers.solvers import _invert, unrad, checksol, posify, _ispow, \
    det_quick, det_perm, det_minor, _simple_dens, denoms  # 导入求解辅助函数

from sympy.physics.units import cm  # 导入厘米单位
from sympy.polys.rootoftools import CRootOf  # 导入根式工具类

from sympy.testing.pytest import slow, XFAIL, SKIP, raises  # 导入测试相关函数和装饰器
from sympy.core.random import verify_numerically as tn  # 导入数值验证函数

from sympy.abc import a, b, c, d, e, k, h, p, x, y, z, t, q, m, R  # 导入符号变量

# 定义 NS 函数，用于将表达式转换为指定精度的数值字符串
def NS(e, n=15, **options):
    return sstr(sympify(e).evalf(n, **options), full_prec=True)

# 定义测试函数 test_swap_back，用于测试解方程组的交换和替换行为
def test_swap_back():
    f, g = map(Function, 'fg')  # 创建函数符号 f 和 g
    fx, gx = f(x), g(x)  # 定义 f(x) 和 g(x)
    # 验证解方程组 fx + y = 2, fx - gx - 5 的解
    assert solve([fx + y - 2, fx - gx - 5], fx, y, gx) == \
        {fx: gx + 5, y: -gx - 3}
    # 验证解方程 fx + gx*x - 2 的解
    assert solve(fx + gx*x - 2, [fx, gx], dict=True) == [{fx: 2, gx: 0}]
    # 验证解方程 fx + gx**2*x - y 的解
    assert solve(fx + gx**2*x - y, [fx, gx], dict=True) == [{fx: y, gx: 0}]
    # 验证解方程组 f(1) - 2, x + 2 的解
    assert solve([f(1) - 2, x + 2], dict=True) == [{x: -2, f(1): 2}]

# 定义猜测解策略函数 guess_solve_strategy，用于判断解方程的策略
def guess_solve_strategy(eq, symbol):
    try:
        solve(eq, symbol)  # 尝试求解方程 eq 关于 symbol 的解
        return True
    except (TypeError, NotImplementedError):
        return False

# 定义测试函数 test_guess_poly，用于测试多项式方程的解策略猜测
def test_guess_poly():
    # 验证多项式方程 S(4) 是否被正确猜测为多项式解策略
    assert guess_solve_strategy( S(4), x )  # == GS_POLY
    # 验证方程 x 是否被正确猜测为多项式解策略
    assert guess_solve_strategy( x, x )  # == GS_POLY
    assert guess_solve_strategy( x + a, x )  # 调用 guess_solve_strategy 函数，判断解决策略是否为多项式类型
    assert guess_solve_strategy( 2*x, x )  # 调用 guess_solve_strategy 函数，判断解决策略是否为多项式类型
    assert guess_solve_strategy( x + sqrt(2), x)  # 调用 guess_solve_strategy 函数，判断解决策略是否为多项式类型
    assert guess_solve_strategy( x + 2**Rational(1, 4), x)  # 调用 guess_solve_strategy 函数，判断解决策略是否为多项式类型
    assert guess_solve_strategy( x**2 + 1, x )  # 调用 guess_solve_strategy 函数，判断解决策略是否为多项式类型
    assert guess_solve_strategy( x**2 - 1, x )  # 调用 guess_solve_strategy 函数，判断解决策略是否为多项式类型
    assert guess_solve_strategy( x*y + y, x )  # 调用 guess_solve_strategy 函数，判断解决策略是否为多项式类型
    assert guess_solve_strategy( x*exp(y) + y, x)  # 调用 guess_solve_strategy 函数，判断解决策略是否为多项式类型
    assert guess_solve_strategy(
        (x - y**3)/(y**2*sqrt(1 - y**2)), x)  # 调用 guess_solve_strategy 函数，判断解决策略是否为多项式类型
def test_guess_poly_cv():
    # polynomial equations via a change of variable
    assert guess_solve_strategy( sqrt(x) + 1, x )  # 判断解策略是否为多项式变量转换类型1
    assert guess_solve_strategy(
        x**Rational(1, 3) + sqrt(x) + 1, x )  # 判断解策略是否为多项式变量转换类型1
    assert guess_solve_strategy( 4*x*(1 - sqrt(x)), x )  # 判断解策略是否为多项式变量转换类型1

    # polynomial equation multiplying both sides by x**n
    assert guess_solve_strategy( x + 1/x + y, x )  # 判断解策略是否为多项式变量转换类型2


def test_guess_rational_cv():
    # rational functions
    assert guess_solve_strategy( (x + 1)/(x**2 + 2), x)  # 判断解策略是否为有理函数类型
    assert guess_solve_strategy(
        (x - y**3)/(y**2*sqrt(1 - y**2)), y)  # 判断解策略是否为有理函数变量转换类型1

    # rational functions via the change of variable y -> x**n
    assert guess_solve_strategy( (sqrt(x) + 1)/(x**Rational(1, 3) + sqrt(x) + 1), x ) \
        # 判断解策略是否为有理函数变量转换类型1


def test_guess_transcendental():
    # transcendental functions
    assert guess_solve_strategy( exp(x) + 1, x )  # 判断解策略是否为超越函数类型
    assert guess_solve_strategy( 2*cos(x) - y, x )  # 判断解策略是否为超越函数类型
    assert guess_solve_strategy(
        exp(x) + exp(-x) - y, x )  # 判断解策略是否为超越函数类型
    assert guess_solve_strategy(3**x - 10, x)  # 判断解策略是否为超越函数类型
    assert guess_solve_strategy(-3**x + 10, x)  # 判断解策略是否为超越函数类型

    assert guess_solve_strategy(a*x**b - y, x)  # 判断解策略是否为超越函数类型


def test_solve_args():
    # equation container, issue 5113
    ans = {x: -3, y: 1}
    eqs = (x + 5*y - 2, -3*x + 6*y - 15)
    assert all(solve(container(eqs), x, y) == ans for container in
        (tuple, list, set, frozenset))  # 解方程组，使用不同容器类型作为输入
    assert solve(Tuple(*eqs), x, y) == ans
    # implicit symbol to solve for
    assert set(solve(x**2 - 4)) == {S(2), -S(2)}  # 解二次方程，返回解集合
    assert solve([x + y - 3, x - y - 5]) == {x: 4, y: -1}  # 解线性方程组，返回解字典
    assert solve(x - exp(x), x, implicit=True) == [exp(x)]  # 解非线性方程，返回隐式符号解
    # no symbol to solve for
    assert solve(42) == solve(42, x) == []  # 没有符号进行求解，返回空列表
    assert solve([1, 2]) == []  # 没有方程进行求解，返回空列表
    assert solve([sqrt(2)],[x]) == []  # 没有符号进行求解，返回空列表
    # duplicate symbols raises
    raises(ValueError, lambda: solve((x - 3, y + 2), x, y, x))  # 解方程时重复符号会引发错误
    raises(ValueError, lambda: solve(x, x, x))  # 解方程时重复符号会引发错误
    # no error in exclude
    assert solve(x, x, exclude=[y, y]) == [0]  # 解方程时排除指定符号，返回解列表
    # unordered symbols
    # only 1
    assert solve(y - 3, {y}) == [3]  # 解方程，返回解列表
    # more than 1
    assert solve(y - 3, {x, y}) == [{y: 3}]  # 解方程，返回解字典
    # multiple symbols: take the first linear solution+
    # - return as tuple with values for all requested symbols
    assert solve(x + y - 3, [x, y]) == [(3 - y, y)]  # 解线性方程组，返回解元组
    # - unless dict is True
    assert solve(x + y - 3, [x, y], dict=True) == [{x: 3 - y}]  # 解线性方程组，返回解字典
    # - or no symbols are given
    assert solve(x + y - 3) == [{x: 3 - y}]  # 解线性方程组，返回解字典
    # multiple symbols might represent an undetermined coefficients system
    # 使用 solve 函数解决方程 a + b*x - 2 = 0 关于变量 a 和 b 的方程组，期望解为 {a: 2, b: 0}
    assert solve(a + b*x - 2, [a, b]) == {a: 2, b: 0}
    
    # 使用 solve 函数解决方程 (a + b)*x + b - c = 0 关于变量 a 和 b 的方程组，期望解为 {a: -c, b: c}
    assert solve((a + b)*x + b - c, [a, b]) == {a: -c, b: c}
    
    # 构造复杂的等式 eq
    eq = a*x**2 + b*x + c - ((x - h)**2 + 4*p*k)/4/p
    
    # 使用 solve 函数解决方程 eq 关于变量 h, p, k 的方程组，排除变量 a, b, c，期望解为 sol
    sol = solve(eq, [h, p, k], exclude=[a, b, c])
    assert sol == {h: -b/(2*a), k: (4*a*c - b**2)/(4*a), p: 1/(4*a)}
    
    # 使用 solve 函数解决方程 eq 关于变量 h, p, k 的方程组，返回字典形式的解，期望解为 [sol]
    assert solve(eq, [h, p, k], dict=True) == [sol]
    
    # 使用 solve 函数解决方程 eq 关于变量 h, p, k 的方程组，返回集合形式的解，期望解为 ([h, p, k], {(...)})
    assert solve(eq, [h, p, k], set=True) == ([h, p, k], {(-b/(2*a), 1/(4*a), (4*a*c - b**2)/(4*a))})
    
    # issue 23889 - 多项式系统未简化
    # 使用 solve 函数解决方程 eq 关于变量 h, p, k 的方程组，排除变量 a, b, c，不进行简化，期望解为 {h: ..., k: ..., p: ...}
    assert solve(eq, [h, p, k], exclude=[a, b, c], simplify=False) == {h: -b/(2*a), k: (4*a*c - b**2)/(4*a), p: 1/(4*a)}
    
    # 构造元组 args
    args = (a + b)*x - b**2 + 2, a, b
    
    # 使用 solve 函数解决 args[0] = 0 关于变量 a, b 的方程组，期望解为 [((b**2 - b*x - 2)/x, b)]
    assert solve(*args) == [((b**2 - b*x - 2)/x, b)]
    
    # 使用 solve 函数解决方程 a*x + b**2/(x + 4) - 3*x - 4/x = 0 关于变量 a, b 的方程组，返回字典形式的解，期望解为 [{a: ...}]
    assert solve(a*x + b**2/(x + 4) - 3*x - 4/x, a, b, dict=True) == [{a: (-b**2*x + 3*x**3 + 12*x**2 + 4*x + 16)/(x**2*(x + 4))}]
    
    # 失败的单一方程
    # 使用 solve 函数解决方程 1/(1/x - y + exp(y)) = 0，期望解为 []
    assert solve(1/(1/x - y + exp(y))) == []
    
    # 使用 solve 函数验证对于 exp(x) + sin(x) + exp(y) + sin(y) 的解决方法未实现的情况
    raises(NotImplementedError, lambda: solve(exp(x) + sin(x) + exp(y) + sin(y)))
    
    # 失败的系统
    # -- 当未给出符号时，一个方程组失败
    assert solve([y, exp(x) + x]) == [{x: -LambertW(1), y: 0}]
    
    # -- 当给出符号时，两个方程组都失败
    assert solve((exp(x) - x, exp(y) - y)) == [{x: -LambertW(-1), y: -LambertW(-1)}]
    
    # -- 当给出符号时，解决方程组 [y, exp(x) + x] 关于变量 x, y 的方程组，期望解为 [(-LambertW(1), 0)]
    assert solve([y, exp(x) + x], x, y) == [(-LambertW(1), 0)]
    
    # 符号为数字
    # 使用 solve 函数解决方程 x**2 - pi = 0 关于变量 pi 的方程，期望解为 [x**2]
    assert solve(x**2 - pi, pi) == [x**2]
    
    # 没有方程
    # 使用 solve 函数解决空方程组 [] 关于变量 x 的方程组，期望解为 []
    assert solve([], [x]) == []
    
    # 非线性系统
    # 使用 solve 函数解决方程组 (x**2 - 4, y - 2) 关于变量 x, y 的方程组，期望解为 [(-2, 2), (2, 2)]
    assert solve((x**2 - 4, y - 2), x, y) == [(-2, 2), (2, 2)]
    
    # 使用 solve 函数解决方程组 (x**2 - 4, y - 2) 关于变量 y, x 的方程组，期望解为 [(2, -2), (2, 2)]
    assert solve((x**2 - 4, y - 2), y, x) == [(2, -2), (2, 2)]
    
    # 使用 solve 函数解决方程组 (x**2 - 4 + z, y - 2 - z) 关于变量 a, z, y, x 的方程组，返回集合形式的解，期望解为 ([a, z, y, x], {(...)})
    assert solve((x**2 - 4 + z, y - 2 - z), a, z, y, x, set=True) == ([a, z, y, x], {(a, z, z + 2, -sqrt(4 - z)), (a, z, z + 2, sqrt(4 - z))})
    
    # 过度确定的系统
    # - 非线性
    # 使用 solve 函数解决方程组 [(x + y)**2 - 4, x + y - 2]，期望解为 [{x: -y + 2}]
    assert solve([(x + y)**2 - 4, x + y - 2]) == [{x: -y + 2}]
    
    # - 线性
    # 使用 solve 函数解决方程组 (x + y - 2, 2*x + 2*y - 4)，期望解为 {x: -y + 2}
    assert solve((x + y - 2, 2*x + 2*y - 4)) == {x: -y + 2}
    
    # 当一个或多个参数为布尔值时
    # 使用 solve 函数解决方程 Eq(x**2, 0.0)，期望解为 [0.0]
    assert solve(Eq(x**2, 0.0)) == [0.0]
    
    # 使用 solve 函数解决方程组 [True, Eq(x, 0)] 关于变量 x 的方程组，返回字典形式的解，期望解为 [{x: 0}]
    assert solve([True, Eq(x, 0)], [x], dict=True) == [{x: 0}]
    
    # 使用 solve 函数解决方程组 [Eq(x, x), Eq(x, 0), Eq(x, x+1)] 关于变量 x 的方程组，期望解为 []
    assert solve([Eq(x, x), Eq(x, 0), Eq(x, x+1)], [x], dict=True) == []
    
    # 使用 solve 函数解决方程组 [Eq(x, x+1), x < 2] 关于变量 x 的方程组，期望结果为 False
    assert not solve([
    # 确保 solve 函数按预期处理第一个示例，期望结果是 ([y, x], set())
    assert solve([x - 1, x], (y, x), set=True) == ([y, x], set())
    
    # 确保 solve 函数按预期处理第二个示例，期望结果是 ([x, y], set())
    assert solve([x - 1, x], {y, x}, set=True) == ([x, y], set())
def test_solve_polynomial1():
    # 测试解决一次多项式方程 3*x - 2 = 0，期望返回解 [Rational(2, 3)]
    assert solve(3*x - 2, x) == [Rational(2, 3)]
    
    # 测试解决一次多项式方程 3*x = 2，期望返回解 [Rational(2, 3)]
    assert solve(Eq(3*x, 2), x) == [Rational(2, 3)]
    
    # 测试解决二次多项式方程 x**2 - 1 = 0，期望返回解集 {-1, 1}
    assert set(solve(x**2 - 1, x)) == {-S.One, S.One}
    
    # 测试解决二次多项式方程 x**2 = 1，期望返回解集 {-1, 1}
    assert set(solve(Eq(x**2, 1), x)) == {-S.One, S.One}
    
    # 测试解决一次多项式方程 x - y**3 = 0，期望返回解集 [y**3]
    assert solve(x - y**3, x) == [y**3]
    
    # 计算 x 的三次方根
    rx = root(x, 3)
    
    # 测试解决方程 x - y**3 = 0 关于 y 的解集
    assert solve(x - y**3, y) == [
        rx, -rx/2 - sqrt(3)*I*rx/2, -rx/2 + sqrt(3)*I*rx/2]
    
    # 定义符号变量
    a11, a12, a21, a22, b1, b2 = symbols('a11,a12,a21,a22,b1,b2')
    
    # 测试解决线性方程组 [a11*x + a12*y - b1, a21*x + a22*y - b2] 关于 x 和 y 的解
    assert solve([a11*x + a12*y - b1, a21*x + a22*y - b2], x, y) == \
        {
            x: (a22*b1 - a12*b2)/(a11*a22 - a12*a21),
            y: (a11*b2 - a21*b1)/(a11*a22 - a12*a21),
        }
    
    # 初始化解为 {x: 0, y: 0}
    solution = {x: S.Zero, y: S.Zero}
    
    # 测试解决方程组 (x - y, x + y) 关于 x 和 y 的解，期望解为 {x: 0, y: 0}
    assert solve((x - y, x + y), x, y ) == solution
    assert solve((x - y, x + y), (x, y)) == solution
    assert solve((x - y, x + y), [x, y]) == solution
    
    # 测试解决三次多项式方程 x**3 - 15*x - 4 = 0，期望返回解集
    assert set(solve(x**3 - 15*x - 4, x)) == {
        -2 + 3**S.Half,
        S(4),
        -2 - 3**S.Half
    }
    
    # 测试解决四次多项式方程 (x**2 - 1)**2 - a = 0，期望返回解集
    assert set(solve((x**2 - 1)**2 - a, x)) == \
        {sqrt(1 + sqrt(a)), -sqrt(1 + sqrt(a)),
             sqrt(1 - sqrt(a)), -sqrt(1 - sqrt(a))}


def test_solve_polynomial2():
    # 测试解决常数方程 4 = 0，期望返回空列表
    assert solve(4, x) == []


def test_solve_polynomial_cv_1a():
    """
    Test for solving on equations that can be converted to a polynomial equation
    using the change of variable y -> x**Rational(p, q)
    """
    # 测试解决方程 sqrt(x) - 1 = 0，期望返回解 [1]
    assert solve( sqrt(x) - 1, x) == [1]
    
    # 测试解决方程 sqrt(x) - 2 = 0，期望返回解 [4]
    assert solve( sqrt(x) - 2, x) == [4]
    
    # 测试解决方程 x**(1/4) - 2 = 0，期望返回解 [16]
    assert solve( x**Rational(1, 4) - 2, x) == [16]
    
    # 测试解决方程 x**(1/3) - 3 = 0，期望返回解 [27]
    assert solve( x**Rational(1, 3) - 3, x) == [27]
    
    # 测试解决方程 sqrt(x) + x**(1/3) + x**(1/4) = 0，期望返回解 [0]
    assert solve(sqrt(x) + x**Rational(1, 3) + x**Rational(1, 4), x) == [0]


def test_solve_polynomial_cv_1b():
    # 测试解决方程 4*x*(1 - a*sqrt(x)) = 0，期望返回解集 {0, 1/a**2}
    assert set(solve(4*x*(1 - a*sqrt(x)), x)) == {S.Zero, 1/a**2}
    
    # 测试解决方程 x*(root(x, 3) - 3) = 0，期望返回解集 {0, 27}
    assert set(solve(x*(root(x, 3) - 3), x)) == {S.Zero, S(27)}


def test_solve_polynomial_cv_2():
    """
    Test for solving on equations that can be converted to a polynomial equation
    multiplying both sides of the equation by x**m
    """
    # 测试解决方程 x + 1/x - 1 = 0，期望返回解集
    assert solve(x + 1/x - 1, x) in \
        [[ S.Half + I*sqrt(3)/2, S.Half - I*sqrt(3)/2],
         [ S.Half - I*sqrt(3)/2, S.Half + I*sqrt(3)/2]]


def test_quintics_1():
    f = x**5 - 110*x**3 - 55*x**2 + 2310*x + 979
    s = solve(f, check=False)
    for r in s:
        res = f.subs(x, r.n()).n()
        # 断言每个根 r 满足方程的数值近似为 0
        assert tn(res, 0)

    f = x**5 - 15*x**3 - 5*x**2 + 10*x + 20
    s = solve(f)
    for r in s:
        # 断言每个根 r 是 CRootOf 对象
        assert r.func == CRootOf

    # 如果使用 solve 来获取具有 CRootOf 解的多项式的根，确保在解过程中使用 nfloat 不会失败。
    # 注意：如果需要多项式的数值解，使用 nroots 比仅解决方程并数值评估 RootOf 解更快。
    # 因此，对于方程 eq = x**5 + 3*x + 7，建议使用 Poly(eq).nroots() 而不是 [i.n() for i in solve(eq)] 来获取数值根。
    # 使用 sympy 求解方程 x^5 + 3*x^3 + 7 的根，并取第一个根的数值近似
    assert nfloat(solve(x**5 + 3*x**3 + 7)[0], exponent=False) == \
        # 使用 sympy 中 CRootOf 类来获取方程 x^5 + 3*x^3 + 7 的第一个根，并将其数值化
        CRootOf(x**5 + 3*x**3 + 7, 0).n()
def test_quintics_2():
    # 定义一个五次多项式
    f = x**5 + 15*x + 12
    # 求解多项式的根，忽略检查
    s = solve(f, check=False)
    # 对每个根进行验证
    for r in s:
        # 计算多项式在根处的值，并转换为浮点数
        res = f.subs(x, r.n()).n()
        # 断言结果接近零
        assert tn(res, 0)

    # 定义另一个五次多项式
    f = x**5 - 15*x**3 - 5*x**2 + 10*x + 20
    # 求解多项式的根
    s = solve(f)
    # 对每个根进行断言，应为 CRootOf 类型
    for r in s:
        assert r.func == CRootOf

    # 断言求解特定五次多项式的根
    assert solve(x**5 - 6*x**3 - 6*x**2 + x - 6) == [
        CRootOf(x**5 - 6*x**3 - 6*x**2 + x - 6, 0),
        CRootOf(x**5 - 6*x**3 - 6*x**2 + x - 6, 1),
        CRootOf(x**5 - 6*x**3 - 6*x**2 + x - 6, 2),
        CRootOf(x**5 - 6*x**3 - 6*x**2 + x - 6, 3),
        CRootOf(x**5 - 6*x**3 - 6*x**2 + x - 6, 4)]


def test_quintics_3():
    # 定义一个五次多项式
    y = x**5 + x**3 - 2**Rational(1, 3)
    # 断言求解多项式得到空列表
    assert solve(y) == solve(-y) == []


def test_highorder_poly():
    # 测试解六次多项式
    sol = solve(x**6 - 2*x + 2)
    # 断言所有解为 CRootOf 类型，并且解的数量为 6
    assert all(isinstance(i, CRootOf) for i in sol) and len(sol) == 6


def test_solve_rational():
    """Test solve for rational functions"""
    # 断言解有理函数的结果
    assert solve((x - y**3) / ((y**2) * sqrt(1 - y**2)), x) == [y**3]


def test_solve_conjugate():
    """Test solve for simple conjugate functions"""
    # 断言解共轭函数的结果
    assert solve(conjugate(x) - 3 + I) == [3 + I]


def test_solve_nonlinear():
    # 断言解非线性方程的结果
    assert solve(x**2 - y**2, x, y, dict=True) == [{x: -y}, {x: y}]
    assert solve(x**2 - y**2/exp(x), y, x, dict=True) == [{y: -x*sqrt(exp(x))},
                                                          {y: x*sqrt(exp(x))}]


def test_issue_8666():
    x = symbols('x')
    # 断言解方程的结果
    assert solve(Eq(x**2 - 1/(x**2 - 4), 4 - 1/(x**2 - 4)), x) == []
    assert solve(Eq(x + 1/x, 1/x), x) == []


def test_issue_7228():
    # 断言解方程的结果
    assert solve(4**(2*(x**2) + 2*x) - 8, x) == [Rational(-3, 2), S.Half]


def test_issue_7190():
    # 断言解方程的结果
    assert solve(log(x-3) + log(x+3), x) == [sqrt(10)]


def test_issue_21004():
    x = symbols('x')
    # 定义一个函数和其导数
    f = x / sqrt(x**2 + 1)
    f_diff = f.diff(x)
    # 断言解导数方程的结果
    assert solve(f_diff, x) == []


def test_issue_24650():
    x = symbols('x')
    # 解分段函数方程的结果
    r = solve(Eq(Piecewise((x, Eq(x, 0) | (x > 1))), 0))
    assert r == [0]
    # 验证解是否满足方程
    r = checksol(Eq(Piecewise((x, Eq(x, 0) | (x > 1))), 0), x, sol=0)
    assert r is True


def test_linear_system():
    x, y, z, t, n = symbols('x, y, z, t, n')

    # 解线性方程组的结果，期望为空列表
    assert solve([x - 1, x - y, x - 2*y, y - 1], [x, y]) == []

    # 解线性方程组的结果，期望为空列表
    assert solve([x - 1, x - 1, x - y, x - 2*y], [x, y]) == []

    # 解线性方程组的结果，期望为特定解
    assert solve([x + 5*y - 2, -3*x + 6*y - 15], x, y) == {x: -3, y: 1}

    M = Matrix([[0, 0, n*(n + 1), (n + 1)**2, 0],
                [n + 1, n + 1, -2*n - 1, -(n + 1), 0],
                [-1, 0, 1, 0, 0]])

    # 解线性方程组的结果，期望为字典形式
    assert solve_linear_system(M, x, y, z, t) == \
        {x: t*(-n-1)/n, y: 0, z: t*(-n-1)/n}

    # 解线性方程组的结果，期望为字典形式
    assert solve([x + y + z + t, -z - t], x, y, z, t) == {x: -y, z: -t}


@XFAIL
def test_linear_system_xfail():
    # 测试未解决的线性方程组问题
    M = Matrix([[0,    15.0, 10.0, 700.0],
                [1,    1,    1,    100.0],
                [0,    10.0, 5.0,  200.0],
                [-5.0, 0,    0,    0    ]])
    # 调用 solve_linear_system 函数，并断言其返回结果与预期的字典相等
    assert solve_linear_system(M, x, y, z) == {x: 0, y: -60.0, z: 160.0}
# 定义一个测试函数，用于测试线性系统的函数求解
def test_linear_system_function():
    # 创建一个函数符号对象 'a'
    a = Function('a')
    # 断言求解给定的线性系统方程组，并验证结果是否符合预期
    assert solve([a(0, 0) + a(0, 1) + a(1, 0) + a(1, 1), -a(1, 0) - a(1, 1)],
                 a(0, 0), a(0, 1), a(1, 0), a(1, 1)) == {a(1, 0): -a(1, 1), a(0, 0): -a(0, 1)}

# 定义一个测试函数，用于验证符号表达式求解的性能
def test_linear_system_symbols_doesnt_hang_1():
    
    def _mk_eqs(wy):
        # 生成一个适用于拟合 wy*2 - 1 阶多项式的方程组
        # 在端点处，导数直到 wy - 1 阶是已知的
        order = 2*wy - 1
        # 定义实数符号变量 x, x0, x1
        x, x0, x1 = symbols('x, x0, x1', real=True)
        # 定义实数符号变量 y0s 和 y1s，用于存储端点处的值
        y0s = symbols('y0_:{}'.format(wy), real=True)
        y1s = symbols('y1_:{}'.format(wy), real=True)
        # 定义实数符号变量 c，用于存储多项式系数
        c = symbols('c_:{}'.format(order+1), real=True)

        # 构建多项式表达式
        expr = sum(coeff*x**o for o, coeff in enumerate(c))
        eqs = []
        # 生成方程组
        for i in range(wy):
            eqs.append(expr.diff(x, i).subs({x: x0}) - y0s[i])
            eqs.append(expr.diff(x, i).subs({x: x1}) - y1s[i])
        return eqs, c

    #
    # 此测试的目的是确保这些调用不会导致程序挂起。返回的表达式较为复杂，因此未在此处包含。
    # 测试其正确性需要的时间比求解系统方程组本身更长。
    #

    # 对 n 从 1 到 7 进行迭代
    for n in range(1, 7+1):
        # 生成方程组和系数列表
        eqs, c = _mk_eqs(n)
        # 解方程组
        solve(eqs, c)


def test_linear_system_symbols_doesnt_hang_2():
    # 此函数尚未实现，保留以后扩展
    # 创建一个 19x19 的矩阵 M，包含了整数类型的数据
    M = Matrix([
        [66, 24, 39, 50, 88, 40, 37, 96, 16, 65, 31, 11, 37, 72, 16, 19, 55, 37, 28, 76],
        [10, 93, 34, 98, 59, 44, 67, 74, 74, 94, 71, 61, 60, 23,  6,  2, 57,  8, 29, 78],
        [19, 91, 57, 13, 64, 65, 24, 53, 77, 34, 85, 58, 87, 39, 39,  7, 36, 67, 91,  3],
        [74, 70, 15, 53, 68, 43, 86, 83, 81, 72, 25, 46, 67, 17, 59, 25, 78, 39, 63,  6],
        [69, 40, 67, 21, 67, 40, 17, 13, 93, 44, 46, 89, 62, 31, 30, 38, 18, 20, 12, 81],
        [50, 22, 74, 76, 34, 45, 19, 76, 28, 28, 11, 99, 97, 82,  8, 46, 99, 57, 68, 35],
        [58, 18, 45, 88, 10, 64,  9, 34, 90, 82, 17, 41, 43, 81, 45, 83, 22, 88, 24, 39],
        [42, 21, 70, 68,  6, 33, 64, 81, 83, 15, 86, 75, 86, 17, 77, 34, 62, 72, 20, 24],
        [ 7,  8,  2, 72, 71, 52, 96,  5, 32, 51, 31, 36, 79, 88, 25, 77, 29, 26, 33, 13],
        [19, 31, 30, 85, 81, 39, 63, 28, 19, 12, 16, 49, 37, 66, 38, 13,  3, 71, 61, 51],
        [29, 82, 80, 49, 26, 85,  1, 37,  2, 74, 54, 82, 26, 47, 54,  9, 35,  0, 99, 40],
        [15, 49, 82, 91, 93, 57, 45, 25, 45, 97, 15, 98, 48, 52, 66, 24, 62, 54, 97, 37],
        [62, 23, 73, 53, 52, 86, 28, 38,  0, 74, 92, 38, 97, 70, 71, 29, 26, 90, 67, 45],
        [ 2, 32, 23, 24, 71, 37, 25, 71,  5, 41, 97, 65, 93, 13, 65, 45, 25, 88, 69, 50],
        [40, 56,  1, 29, 79, 98, 79, 62, 37, 28, 45, 47,  3,  1, 32, 74, 98, 35, 84, 32],
        [33, 15, 87, 79, 65,  9, 14, 63, 24, 19, 46, 28, 74, 20, 29, 96, 84, 91, 93,  1],
        [97, 18, 12, 52,  1,  2, 50, 14, 52, 76, 19, 82, 41, 73, 51, 79, 13,  3, 82, 96],
        [40, 28, 52, 10, 10, 71, 56, 78, 82,  5, 29, 48,  1, 26, 16, 18, 50, 76, 86, 52],
        [38, 89, 83, 43, 29, 52, 90, 77, 57,  0, 67, 20, 81, 88, 48, 96, 88, 58, 14,  3]])
    
    # 创建符号变量列表 syms，包含 19 个符号变量 x0 到 x18
    syms = x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18 = symbols('x:19')
    sol = {
        x0:  -S(1967374186044955317099186851240896179)/3166636564687820453598895768302256588,
        x1:  -S(84268280268757263347292368432053826)/791659141171955113399723942075564147,
        x2:  -S(229962957341664730974463872411844965)/1583318282343910226799447884151128294,
        x3:   S(990156781744251750886760432229180537)/6333273129375640907197791536604513176,
        x4:  -S(2169830351210066092046760299593096265)/18999819388126922721593374609813539528,
        x5:   S(4680868883477577389628494526618745355)/9499909694063461360796687304906769764,
        x6:  -S(1590820774344371990683178396480879213)/3166636564687820453598895768302256588,
        x7:  -S(54104723404825537735226491634383072)/339282489073695048599881689460956063,
        x8:   S(3182076494196560075964847771774733847)/6333273129375640907197791536604513176,
        x9:  -S(10870817431029210431989147852497539675)/18999819388126922721593374609813539528,
        x10: -S(13118019242576506476316318268573312603)/18999819388126922721593374609813539528,
        x11: -S(5173852969886775824855781403820641259)/4749954847031730680398343652453384882,
        x12:  S(4261112042731942783763341580651820563)/4749954847031730680398343652453384882,
        x13: -S(821833082694661608993818117038209051)/6333273129375640907197791536604513176,
        x14:  S(906881575107250690508618713632090559)/904753304196520129599684505229216168,
        x15: -S(732162528717458388995329317371283987)/6333273129375640907197791536604513176,
        x16:  S(4524215476705983545537087360959896817)/9499909694063461360796687304906769764,
        x17: -S(3898571347562055611881270844646055217)/6333273129375640907197791536604513176,
        x18:  S(7513502486176995632751685137907442269)/18999819388126922721593374609813539528
    }


    # 定义一个包含符号表达式的字典，表示解的集合
    sol = {
        x0:  -S(1967374186044955317099186851240896179)/3166636564687820453598895768302256588,
        x1:  -S(84268280268757263347292368432053826)/791659141171955113399723942075564147,
        x2:  -S(229962957341664730974463872411844965)/1583318282343910226799447884151128294,
        x3:   S(990156781744251750886760432229180537)/6333273129375640907197791536604513176,
        x4:  -S(2169830351210066092046760299593096265)/18999819388126922721593374609813539528,
        x5:   S(4680868883477577389628494526618745355)/9499909694063461360796687304906769764,
        x6:  -S(1590820774344371990683178396480879213)/3166636564687820453598895768302256588,
        x7:  -S(54104723404825537735226491634383072)/339282489073695048599881689460956063,
        x8:   S(3182076494196560075964847771774733847)/6333273129375640907197791536604513176,
        x9:  -S(10870817431029210431989147852497539675)/18999819388126922721593374609813539528,
        x10: -S(13118019242576506476316318268573312603)/18999819388126922721593374609813539528,
        x11: -S(5173852969886775824855781403820641259)/4749954847031730680398343652453384882,
        x12:  S(4261112042731942783763341580651820563)/4749954847031730680398343652453384882,
        x13: -S(821833082694661608993818117038209051)/6333273129375640907197791536604513176,
        x14:  S(906881575107250690508618713632090559)/904753304196520129599684505229216168,
        x15: -S(732162528717458388995329317371283987)/6333273129375640907197791536604513176,
        x16:  S(4524215476705983545537087360959896817)/9499909694063461360796687304906769764,
        x17: -S(3898571347562055611881270844646055217)/6333273129375640907197791536604513176,
        x18:  S(7513502486176995632751685137907442269)/18999819388126922721593374609813539528
    }


    eqs = list(M * Matrix(syms + (1,)))
    # 构造包含矩阵乘法的方程列表，将符号和常数 1 添加到矩阵中
    assert solve(eqs, syms) == sol


    y = Symbol('y')
    # 创建一个符号 'y'
    eqs = list(y * M * Matrix(syms + (1,)))
    # 构造包含符号 'y'、矩阵乘法的方程列表，将符号和常数 1 添加到矩阵中
    assert solve(eqs, syms) == sol
# 定义一个测试函数，用于测试解线性系统的LU分解求解器
def test_linear_systemLU():
    # 导入符号 n
    n = Symbol('n')

    # 创建一个 3x4 的符号矩阵 M
    M = Matrix([[1, 2, 0, 1], [1, 3, 2*n, 1], [4, -1, n**2, 1]])

    # 断言：使用 LU 分解求解线性系统 M * [x, y, z]^T = [0, 0, 0]^T，期望得到解字典
    assert solve_linear_system_LU(M, [x, y, z]) == {z: -3/(n**2 + 18*n),
                                                  x: 1 - 12*n/(n**2 + 18*n),
                                                  y: 6*n/(n**2 + 18*n)}

# 注意：对于某些方程，可能存在多个解，因此如果求解器的实现方式改变，可能会选择不同的解分支而导致测试失败

# 标记为慢速测试的装饰器函数，用于测试解超越方程
@slow
def test_solve_transcendental():
    # 从 sympy.abc 导入符号 a 和 b
    from sympy.abc import a, b

    # 断言：解方程 exp(x) - 3 = 0，期望得到 [log(3)]
    assert solve(exp(x) - 3, x) == [log(3)]

    # 断言：解方程 (a*x + b)*(exp(x) - 3) = 0，期望得到 {-b/a, log(3)}
    assert set(solve((a*x + b)*(exp(x) - 3), x)) == {-b/a, log(3)}

    # 断言：解方程 cos(x) - y = 0，期望得到 [-acos(y) + 2*pi, acos(y)]
    assert solve(cos(x) - y, x) == [-acos(y) + 2*pi, acos(y)]

    # 断言：解方程 2*cos(x) - y = 0，期望得到 [-acos(y/2) + 2*pi, acos(y/2)]
    assert solve(2*cos(x) - y, x) == [-acos(y/2) + 2*pi, acos(y/2)]

    # 断言：解方程 cos(x) = sin(x)，期望得到 [pi/4]
    assert solve(Eq(cos(x), sin(x)), x) == [pi/4]

    # 断言：解方程 exp(x) + exp(-x) - y = 0，期望得到不定集合，有三种可能的结果
    assert set(solve(exp(x) + exp(-x) - y, x)) in [{
        log(y/2 - sqrt(y**2 - 4)/2),
        log(y/2 + sqrt(y**2 - 4)/2),
    }, {
        log(y - sqrt(y**2 - 4)) - log(2),
        log(y + sqrt(y**2 - 4)) - log(2)},
    {
        log(y/2 - sqrt((y - 2)*(y + 2))/2),
        log(y/2 + sqrt((y - 2)*(y + 2))/2)}]

    # 其余断言省略，依次验证解超越方程的各种情况
    # 第一个assert语句，解决方程 z*cos(sin(x)) - y = 0 关于 x 的根
    assert solve(z*cos(sin(x)) - y, x) == [
        pi - asin(acos(y/z)), asin(acos(y/z) - 2*pi) + pi,
        -asin(acos(y/z) - 2*pi), asin(acos(y/z))]
    
    # 第二个assert语句，解决方程 z*cos(x) = 0 关于 x 的根
    assert solve(z*cos(x), x) == [pi/2, pi*Rational(3, 2)]
    
    # issue 4508
    # 解决方程 y - b*x/(a + x) = 0 关于 x 的根
    assert solve(y - b*x/(a + x), x) in [[-a*y/(y - b)], [a*y/(b - y)]]
    
    # 解决方程 y - b*exp(a/x) = 0 关于 x 的根
    assert solve(y - b*exp(a/x), x) == [a/log(y/b)]
    
    # issue 4507
    # 解决方程 y - b/(1 + a*x) = 0 关于 x 的根
    assert solve(y - b/(1 + a*x), x) in [[(b - y)/(a*y)], [-((y - b)/(a*y))]]
    
    # issue 4506
    # 解决方程 y - a*x**b = 0 关于 x 的根
    assert solve(y - a*x**b, x) == [(y/a)**(1/b)]
    
    # issue 4505
    # 解决方程 z**x - y = 0 关于 x 的根
    assert solve(z**x - y, x) == [log(y)/log(z)]
    
    # issue 4504
    # 解决方程 2**x - 10 = 0 关于 x 的根
    assert solve(2**x - 10, x) == [1 + log(5)/log(2)]
    
    # issue 6744
    # 解决方程 x*y = 0 关于 x 和 y 的根
    assert solve(x*y) == [{x: 0}, {y: 0}]
    assert solve([x*y]) == [{x: 0}, {y: 0}]
    
    # 解决方程 x**y - 1 = 0 关于 x 和 y 的根
    assert solve(x**y - 1) == [{x: 1}, {y: 0}]
    assert solve([x**y - 1]) == [{x: 1}, {y: 0}]
    
    # 解决方程 x*y*(x**2 - y**2) = 0 关于 x 和 y 的根
    assert solve(x*y*(x**2 - y**2)) == [{x: 0}, {x: -y}, {x: y}, {y: 0}]
    assert solve([x*y*(x**2 - y**2)]) == [{x: 0}, {x: -y}, {x: y}, {y: 0}]
    
    # issue 4739
    # 解决方程 exp(log(5)*x) - 2**x = 0 关于 x 的根
    assert solve(exp(log(5)*x) - 2**x, x) == [0]
    
    # issue 14791
    # 解决方程 exp(log(5)*x) - exp(log(2)*x) = 0 关于 x 的根
    assert solve(exp(log(5)*x) - exp(log(2)*x), x) == [0]
    
    # 解决方程 y*f(log(5)*x) - y*f(log(2)*x) = 0 关于 x 的根
    assert solve(y*f(log(5)*x) - y*f(log(2)*x), x) == [0]
    
    # 解决方程 f(x) - f(0) = 0 关于 x 的根
    assert solve(f(x) - f(0), x) == [0]
    
    # 解决方程 f(x) - f(2 - x) = 0 关于 x 的根
    assert solve(f(x) - f(2 - x), x) == [1]
    
    # 检查函数调用时引发的异常
    raises(NotImplementedError, lambda: solve(f(x, y) - f(1, 2), x))
    raises(NotImplementedError, lambda: solve(f(x, y) - f(2 - x, 2), x))
    raises(ValueError, lambda: solve(f(x, y) - f(1 - x), x))
    raises(ValueError, lambda: solve(f(x, y) - f(1), x))
    
    # misc
    # issue 7245
    # 解决方程 sin(sqrt(x)) = 0 关于 x 的根
    assert solve(sin(sqrt(x))) == [0, pi**2]
    
    # issue 7602
    # 解决方程 Eq(a, 0.5 - cos(pi*b)/2) 关于 b 的根
    a, b = symbols('a, b', real=True, negative=False)
    assert str(solve(Eq(a, 0.5 - cos(pi*b)/2), b)) == \
        '[2.0 - 0.318309886183791*acos(1.0 - 2.0*a), 0.318309886183791*acos(1.0 - 2.0*a)]'
    
    # issue 15325
    # 解决方程 y**(1/x) - z = 0 关于 x 的根
    assert solve(y**(1/x) - z, x) == [log(y)/log(z)]
    
    # issue 25685
    # 使用不同的 yi 解决方程组 [cos(x) - S(3)/5, yi - y] 关于 x 和 y 的根
    for yi in [cos(2*x),sin(2*x),cos(x - pi/3)]:
        sol = solve([cos(x) - S(3)/5, yi - y])
        assert (sol[0][y] + sol[1][y]).is_Rational, (yi,sol)
    
    # issue 26008
    # 解决方程 sin(x + pi/6) = 0 关于 x 的根
    assert solve(sin(x + pi/6)) == [-pi/6, 5*pi/6]
# 定义测试函数，用于求解函数的导数
def test_solve_for_functions_derivatives():
    # 定义符号变量 t
    t = Symbol('t')
    # 定义函数 x(t) 和 y(t)
    x = Function('x')(t)
    y = Function('y')(t)
    # 定义符号变量 a11, a12, a21, a22, b1, b2
    a11, a12, a21, a22, b1, b2 = symbols('a11,a12,a21,a22,b1,b2')

    # 解方程组并验证解
    soln = solve([a11*x + a12*y - b1, a21*x + a22*y - b2], x, y)
    assert soln == {
        x: (a22*b1 - a12*b2)/(a11*a22 - a12*a21),
        y: (a11*b2 - a21*b1)/(a11*a22 - a12*a21),
    }

    # 验证单变量方程的解
    assert solve(x - 1, x) == [1]
    assert solve(3*x - 2, x) == [Rational(2, 3)]

    # 解导数方程组并验证解
    soln = solve([a11*x.diff(t) + a12*y.diff(t) - b1, a21*x.diff(t) +
            a22*y.diff(t) - b2], x.diff(t), y.diff(t))
    assert soln == { y.diff(t): (a11*b2 - a21*b1)/(a11*a22 - a12*a21),
            x.diff(t): (a22*b1 - a12*b2)/(a11*a22 - a12*a21) }

    # 验证导数的单变量方程解
    assert solve(x.diff(t) - 1, x.diff(t)) == [1]
    assert solve(3*x.diff(t) - 2, x.diff(t)) == [Rational(2, 3)]

    # 解集合方程组并验证解
    eqns = {3*x - 1, 2*y - 4}
    assert solve(eqns, {x, y}) == { x: Rational(1, 3), y: 2 }

    # 定义符号变量 x 和函数 f(x)
    x = Symbol('x')
    f = Function('f')
    F = x**2 + f(x)**2 - 4*x - 1
    # 解微分方程并验证解
    assert solve(F.diff(x), diff(f(x), x)) == [(-x + 2)/f(x)]

    # 混合大小写的符号和函数
    x = Symbol('x')
    y = Function('y')(t)
    # 解混合微分方程组并验证解
    soln = solve([a11*x + a12*y.diff(t) - b1, a21*x +
            a22*y.diff(t) - b2], x, y.diff(t))
    assert soln == { y.diff(t): (a11*b2 - a21*b1)/(a11*a22 - a12*a21),
            x: (a22*b1 - a12*b2)/(a11*a22 - a12*a21) }

    # 解特定的微分方程组（issue 13263）
    x = Symbol('x')
    f = Function('f')
    soln = solve([f(x).diff(x) + f(x).diff(x, 2) - 1, f(x).diff(x) - f(x).diff(x, 2)],
            f(x).diff(x), f(x).diff(x, 2))
    assert soln == { f(x).diff(x, 2): S(1)/2, f(x).diff(x): S(1)/2 }

    # 解特定的微分方程组（issue 13263 的另一种形式）
    soln = solve([f(x).diff(x, 2) + f(x).diff(x, 3) - 1, 1 - f(x).diff(x, 2) -
            f(x).diff(x, 3), 1 - f(x).diff(x,3)], f(x).diff(x, 2), f(x).diff(x, 3))
    assert soln == { f(x).diff(x, 2): 0, f(x).diff(x, 3): 1 }


def test_issue_3725():
    # 定义函数 f(x)
    f = Function('f')
    F = x**2 + f(x)**2 - 4*x - 1
    e = F.diff(x)
    # 解特定的微分方程（issue 3725）
    assert solve(e, f(x).diff(x)) in [[(2 - x)/f(x)], [-((x - 2)/f(x))]]


def test_issue_3870():
    # 定义符号变量 a, b, c, d
    a, b, c, d = symbols('a b c d')
    A = Matrix(2, 2, [a, b, c, d])
    B = Matrix(2, 2, [0, 2, -3, 0])
    C = Matrix(2, 2, [1, 2, 3, 4])

    # 解线性方程组并验证解
    assert solve(A*B - C, [a, b, c, d]) == {a: 1, b: Rational(-1, 3), c: 2, d: -1}
    assert solve([A*B - C], [a, b, c, d]) == {a: 1, b: Rational(-1, 3), c: 2, d: -1}
    assert solve(Eq(A*B, C), [a, b, c, d]) == {a: 1, b: Rational(-1, 3), c: 2, d: -1}

    assert solve([A*B - B*A], [a, b, c, d]) == {a: d, b: Rational(-2, 3)*c}
    assert solve([A*C - C*A], [a, b, c, d]) == {a: d - c, b: Rational(2, 3)*c}
    assert solve([A*B - B*A, A*C - C*A], [a, b, c, d]) == {a: d, b: 0, c: 0}

    assert solve([Eq(A*B, B*A)], [a, b, c, d]) == {a: d, b: Rational(-2, 3)*c}
    assert solve([Eq(A*C, C*A)], [a, b, c, d]) == {a: d - c, b: Rational(2, 3)*c}
    assert solve([Eq(A*B, B*A), Eq(A*C, C*A)], [a, b, c, d]) == {a: d, b: 0, c: 0}


def test_solve_linear():
    # 定义 Wild 变量 w
    w = Wild('w')
    # 断言：解线性方程 solve_linear(x, x)，期望结果是 (0, 1)
    assert solve_linear(x, x) == (0, 1)
    
    # 断言：解线性方程 solve_linear(x, exclude=[x])，期望结果是 (0, 1)
    assert solve_linear(x, exclude=[x]) == (0, 1)
    
    # 断言：解线性方程 solve_linear(x, symbols=[w])，期望结果是 (0, 1)
    assert solve_linear(x, symbols=[w]) == (0, 1)
    
    # 断言：解线性方程 solve_linear(x, y - 2*x)，期望结果是 [(x, y/3), (y, 3*x)] 中的一个
    assert solve_linear(x, y - 2*x) in [(x, y/3), (y, 3*x)]
    
    # 断言：解线性方程 solve_linear(x, y - 2*x, exclude=[x])，期望结果是 (y, 3*x)
    assert solve_linear(x, y - 2*x, exclude=[x]) == (y, 3*x)
    
    # 断言：解线性方程 solve_linear(3*x - y, 0)，期望结果是 [(x, y/3), (y, 3*x)] 中的一个
    assert solve_linear(3*x - y, 0) in [(x, y/3), (y, 3*x)]
    
    # 断言：解线性方程 solve_linear(3*x - y, 0, [x])，期望结果是 (x, y/3)
    assert solve_linear(3*x - y, 0, [x]) == (x, y/3)
    
    # 断言：解线性方程 solve_linear(3*x - y, 0, [y])，期望结果是 (y, 3*x)
    assert solve_linear(3*x - y, 0, [y]) == (y, 3*x)
    
    # 断言：解线性方程 solve_linear(x**2/y, 1)，期望结果是 (y, x**2)
    assert solve_linear(x**2/y, 1) == (y, x**2)
    
    # 断言：解线性方程 solve_linear(w, x)，期望结果是 [(w, x), (x, w)] 中的一个
    assert solve_linear(w, x) in [(w, x), (x, w)]
    
    # 断言：解线性方程 solve_linear(cos(x)**2 + sin(x)**2 + 2 + y)，期望结果是 (y, -2 - cos(x)**2 - sin(x)**2)
    assert solve_linear(cos(x)**2 + sin(x)**2 + 2 + y) == (y, -2 - cos(x)**2 - sin(x)**2)
    
    # 断言：解线性方程 solve_linear(cos(x)**2 + sin(x)**2 + 2 + y, symbols=[x])，期望结果是 (0, 1)
    assert solve_linear(cos(x)**2 + sin(x)**2 + 2 + y, symbols=[x]) == (0, 1)
    
    # 断言：解线性方程 solve_linear(Eq(x, 3))，期望结果是 (x, 3)
    assert solve_linear(Eq(x, 3)) == (x, 3)
    
    # 断言：解线性方程 solve_linear(1/(1/x - 2))，期望结果是 (0, 0)
    assert solve_linear(1/(1/x - 2)) == (0, 0)
    
    # 断言：解线性方程 solve_linear((x + 1)*exp(-x), symbols=[x])，期望结果是 (x, -1)
    assert solve_linear((x + 1)*exp(-x), symbols=[x]) == (x, -1)
    
    # 断言：解线性方程 solve_linear((x + 1)*exp(x), symbols=[x])，期望结果是 ((x + 1)*exp(x), 1)
    assert solve_linear((x + 1)*exp(x), symbols=[x]) == ((x + 1)*exp(x), 1)
    
    # 断言：解线性方程 solve_linear(x*exp(-x**2), symbols=[x])，期望结果是 (x, 0)
    assert solve_linear(x*exp(-x**2), symbols=[x]) == (x, 0)
    
    # 断言：解线性方程 solve_linear(0**x - 1)，期望结果是 (0**x - 1, 1)
    assert solve_linear(0**x - 1) == (0**x - 1, 1)
    
    # 定义方程式 eq = y*cos(x)**2 + y*sin(x)**2 - y，该式化简后为 0
    eq = y*cos(x)**2 + y*sin(x)**2 - y
    # 断言：解线性方程 solve_linear(eq)，期望结果是 (0, 1)
    assert solve_linear(eq) == (0, 1)
    
    # 定义方程式 eq = cos(x)**2 + sin(x)**2，该式化简后为 1
    eq = cos(x)**2 + sin(x)**2
    # 断言：解线性方程 solve_linear(eq)，期望结果是 (0, 1)
    assert solve_linear(eq) == (0, 1)
    
    # 断言：调用 solve_linear(Eq(x, 3), 3) 应引发 ValueError 异常
    raises(ValueError, lambda: solve_linear(Eq(x, 3), 3))
def test_solve_undetermined_coeffs():
    # 测试未定系数求解函数 solve_undetermined_coeffs
    assert solve_undetermined_coeffs(
        a*x**2 + b*x**2 + b*x + 2*c*x + c + 1, [a, b, c], x
        ) == {a: -2, b: 2, c: -1}
    # 测试有理函数的求解
    assert solve_undetermined_coeffs(a/x + b/(x + 1)
        - (2*x + 1)/(x**2 + x), [a, b], x) == {a: 1, b: 1}
    # 测试有理函数中的约分
    assert solve_undetermined_coeffs(
        ((c + 1)*a*x**2 + (c + 1)*b*x**2 +
        (c + 1)*b*x + (c + 1)*2*c*x + (c + 1)**2)/(c + 1),
        [a, b, c], x) == \
        {a: -2, b: 2, c: -1}
    # 多变量情况
    X, Y, Z = y, x**y, y*x**y
    eq = a*X + b*Y + c*Z - X - 2*Y - 3*Z
    coeffs = a, b, c
    syms = x, y
    assert solve_undetermined_coeffs(eq, coeffs) == {
        a: 1, b: 2, c: 3}
    assert solve_undetermined_coeffs(eq, coeffs, syms) == {
        a: 1, b: 2, c: 3}
    assert solve_undetermined_coeffs(eq, coeffs, *syms) == {
        a: 1, b: 2, c: 3}
    # 检查输出格式
    assert solve_undetermined_coeffs(a*x + a - 2, [a]) == []
    assert solve_undetermined_coeffs(a**2*x - 4*x, [a]) == [
        {a: -2}, {a: 2}]
    assert solve_undetermined_coeffs(0, [a]) == []
    assert solve_undetermined_coeffs(0, [a], dict=True) == []
    assert solve_undetermined_coeffs(0, [a], set=True) == ([], {})
    assert solve_undetermined_coeffs(1, [a]) == []
    abeq = a*x - 2*x + b - 3
    s = {b, a}
    assert solve_undetermined_coeffs(abeq, s, x) == {a: 2, b: 3}
    assert solve_undetermined_coeffs(abeq, s, x, set=True) == ([a, b], {(2, 3)})
    assert solve_undetermined_coeffs(sin(a*x) - sin(2*x), (a,)) is None
    assert solve_undetermined_coeffs(a*x + b*x - 2*x, (a, b)) == {a: 2 - b}


def test_solve_inequalities():
    # 定义符号 x
    x = Symbol('x')
    sol = And(S.Zero < x, x < oo)
    # 解 x + 1 > 1 的不等式
    assert solve(x + 1 > 1) == sol
    assert solve([x + 1 > 1]) == sol
    assert solve([x + 1 > 1], x) == sol
    assert solve([x + 1 > 1], [x]) == sol

    system = [Lt(x**2 - 2, 0), Gt(x**2 - 1, 0)]
    # 解不等式系统
    assert solve(system) == \
        And(Or(And(Lt(-sqrt(2), x), Lt(x, -1)),
               And(Lt(1, x), Lt(x, sqrt(2)))), Eq(0, 0))

    x = Symbol('x', real=True)
    system = [Lt(x**2 - 2, 0), Gt(x**2 - 1, 0)]
    # 解实数域下的不等式系统
    assert solve(system) == \
        Or(And(Lt(-sqrt(2), x), Lt(x, -1)), And(Lt(1, x), Lt(x, sqrt(2))))

    # 问题 6627, 3448
    assert solve((x - 3)/(x - 2) < 0, x) == And(Lt(2, x), Lt(x, 3))
    assert solve(x/(x + 1) > 1, x) == And(Lt(-oo, x), Lt(x, -1))

    assert solve(sin(x) > S.Half) == And(pi/6 < x, x < pi*Rational(5, 6))

    assert solve(Eq(False, x < 1)) == (S.One <= x) & (x < oo)
    assert solve(Eq(True, x < 1)) == (-oo < x) & (x < 1)
    assert solve(Eq(x < 1, False)) == (S.One <= x) & (x < oo)
    assert solve(Eq(x < 1, True)) == (-oo < x) & (x < 1)

    assert solve(Eq(False, x)) == False
    assert solve(Eq(0, x)) == [0]
    assert solve(Eq(True, x)) == True
    assert solve(Eq(1, x)) == [1]
    assert solve(Eq(False, ~x)) == True
    assert solve(Eq(True, ~x)) == False
    # 断言：验证 solve(Ne(True, x)) 的返回值是否为 False
    assert solve(Ne(True, x)) == False
    
    # 断言：验证 solve(Ne(1, x)) 的返回值是否为 (x > -oo) & (x < oo) & Ne(x, 1)
    assert solve(Ne(1, x)) == (x > -oo) & (x < oo) & Ne(x, 1)
def test_issue_4793():
    # 检查 solve 函数在处理各种方程时的正确性

    # 确保对于 1/x 这样的无解方程，solve 返回空列表
    assert solve(1/x) == []

    # 确保 solve 能够正确找到 x*(1 - 5/x) = 0 的解 [5]
    assert solve(x*(1 - 5/x)) == [5]

    # 确保 solve 能够正确找到 x + sqrt(x) - 2 = 0 的解 [1]
    assert solve(x + sqrt(x) - 2) == [1]

    # 确保对于复杂的表达式 -(1 + x)/(2 + x)**2 + 1/(2 + x) = 0，solve 返回空列表
    assert solve(-(1 + x)/(2 + x)**2 + 1/(2 + x)) == []

    # 确保 solve 能够正确找到 -x**2 - 2*x + (x + 1)**2 - 1 = 0 的解 []
    assert solve(-x**2 - 2*x + (x + 1)**2 - 1) == []

    # 确保对于 (x/(x + 1) + 3)**(-2) = 0，solve 返回空列表
    assert solve((x/(x + 1) + 3)**(-2)) == []

    # 确保 solve 能够正确找到 x/sqrt(x**2 + 1) = 0 的解 [0]
    assert solve(x/sqrt(x**2 + 1), x) == [0]

    # 确保 solve 能够正确找到 exp(x) - y = 0 的解 [log(y)]
    assert solve(exp(x) - y, x) == [log(y)]

    # 确保 solve 对于 exp(x) = 0 这样的无解方程返回空列表
    assert solve(exp(x)) == []

    # 确保 solve 能够找到 x**2 + x + sin(y)**2 + cos(y)**2 - 1 = 0 的解 [0, -1] 或 [-1, 0]
    assert solve(x**2 + x + sin(y)**2 + cos(y)**2 - 1, x) in [[0, -1], [-1, 0]]

    # 确保对于复杂的方程 4*3**(5*x + 2) - 7 = 0，solve 能够返回符合条件的解集合
    eq = 4*3**(5*x + 2) - 7
    ans = solve(eq, x)
    assert len(ans) == 5 and all(eq.subs(x, a).n(chop=True) == 0 for a in ans)

    # 确保 solve 能够正确处理 log(x**2) - y**2/exp(x) = 0，并返回符合条件的解集合
    assert solve(log(x**2) - y**2/exp(x), x, y, set=True) == (
        [x, y],
        {(x, sqrt(exp(x) * log(x ** 2))), (x, -sqrt(exp(x) * log(x ** 2)))})

    # 确保 solve 能够正确找到 x**2*z**2 - z**2*y**2 = 0 的解 [{x: -y}, {x: y}, {z: 0}]
    assert solve(x**2*z**2 - z**2*y**2) == [{x: -y}, {x: y}, {z: 0}]

    # 确保 solve 能够正确处理 (x - 1)/(1 + 1/(x - 1)) = 0 的无解情况，返回空列表
    assert solve((x - 1)/(1 + 1/(x - 1))) == []

    # 确保 solve 能够正确找到 x**(y*z) - x = 0 的解 [1]
    assert solve(x**(y*z) - x, x) == [1]

    # 确保 solve 能够正确处理 raise(NotImplementedError) 异常
    raises(NotImplementedError, lambda: solve(log(x) - exp(x), x))
    raises(NotImplementedError, lambda: solve(2**x - exp(x) - 3))


def test_PR1964():
    # issue 5171
    # 确保 solve 能够正确处理 sqrt(x) = 0 和 sqrt(x**3) = 0，返回解 [0]
    assert solve(sqrt(x)) == solve(sqrt(x**3)) == [0]

    # 确保 solve 能够正确找到 sqrt(x - 1) = 0 的解 [1]
    assert solve(sqrt(x - 1)) == [1]

    # issue 4462
    a = Symbol('a')
    # 确保 solve 对于 -3*a/sqrt(x) = 0 的无解情况，返回空列表
    assert solve(-3*a/sqrt(x), x) == []

    # issue 4486
    # 确保 solve 能够正确找到 2*x/(x + 2) - 1 = 0 的解 [2]
    assert solve(2*x/(x + 2) - 1, x) == [2]

    # issue 4496
    # 确保 solve 能够正确找到 (x**2/(7 - x)).diff(x) = 0 的解集合 {0, 14}
    assert set(solve((x**2/(7 - x)).diff(x))) == {S.Zero, S(14)}

    # issue 4695
    f = Function('f')
    # 确保 solve 能够正确找到 (3 - 5*x/f(x))*f(x) = 0 的解 [x*Rational(5, 3)]
    assert solve((3 - 5*x/f(x))*f(x), f(x)) == [x*Rational(5, 3)]

    # issue 4497
    # 确保 solve 能够正确找到 1/root(5 + x, 5) - 9 = 0 的解 [Rational(-295244, 59049)]
    assert solve(1/root(5 + x, 5) - 9, x) == [Rational(-295244, 59049)]

    # 确保 solve 能够正确处理 sqrt(x) + sqrt(sqrt(x)) - 4 = 0 的解 [(Rational(-1, 2) + sqrt(17)/2)**4]
    assert solve(sqrt(x) + sqrt(sqrt(x)) - 4) == [(Rational(-1, 2) + sqrt(17)/2)**4]

    # 确保 solve 能够正确处理 Poly(sqrt(exp(x)) + sqrt(exp(-x)) - 4) = 0 的解集合
    assert set(solve(Poly(sqrt(exp(x)) + sqrt(exp(-x)) - 4))) in \
        [
            {log((-sqrt(3) + 2)**2), log((sqrt(3) + 2)**2)},
            {2*log(-sqrt(3) + 2), 2*log(sqrt(3) + 2)},
            {log(-4*sqrt(3) + 7), log(4*sqrt(3) + 7)},
        ]

    # 确保 solve 能够正确处理 Poly(exp(x) + exp(-x) - 4) = 0 的解集合
    assert set(solve(Poly(exp(x) + exp(-x) - 4))) == \
        {log(-sqrt(3) + 2), log(sqrt(3) + 2)}

    # 确保 solve 能够正确处理 x**y + x**(2*y) - 1 = 0，并返回符合条件的解集合
    assert set(solve(x**y + x**(2*y) - 1, x)) == \
        {(Rational(-1, 2) + sqrt(5)/2)**(1/y), (Rational(-1, 2) - sqrt(5)/2)**(1/y)}

    # 确保 solve 能够正确处理 exp(x/y)*exp(-z/y) - 2 = 0，并返回符合条件的解 [ (x - z)/log(2) ]
    assert solve(exp(x/y)*exp(-z/y) - 2, y) == [(x - z)/log(2)]

    # 确保 solve 能够正确处理 x**z*y**z - 2 = 0，并返回符合条件的解集合
    assert solve(x**z*y**z - 2, z) in [[log(2)/(log(x) + log(y))], [log(2)/(log(x*y))]]

    # 确保 solve 能够正确处理 exp(3*x) - exp(3) =
    # 使用 solve 函数解决方程 (n - 1)*(n + 2)*(2*n - 1) = 0，期望的解是 [1]
    assert solve((n - 1)*(n + 2)*(2*n - 1), n) == [1]
    
    # 创建一个符号变量 x，并指定它是正数
    x = Symbol('x', positive=True)
    
    # 创建一个符号变量 y，未指定正负性
    y = Symbol('y')
    
    # 使用 solve 函数解决方程组 [x + 5*y - 2, -3*x + 6*y - 15]，期望的解是空列表 []，
    # 因为该方程组没有满足 x 为正数的解
    assert solve([x + 5*y - 2, -3*x + 6*y - 15], x, y) == []
                 # not {x: -3, y: 1} b/c x is positive
    
    # 解决方程组 [(x + y), 2 - y**2]，期望的解是 [(sqrt(2), -sqrt(2))]，
    # 因为 x 和 y 没有正数限制，所以 (-sqrt(2), sqrt(2)) 不是解
    assert solve([(x + y), 2 - y**2], x, y) == [(sqrt(2), -sqrt(2))]
    
    # 创建一个符号变量 y，并指定它是正数
    y = Symbol('y', positive=True)
    
    # 解决方程 x**2 - y**2/exp(x)，期望的解是 [{y: x*exp(x/2)}]，
    # 因为 y 是正数，所以解中不应包含 {y: -x*exp(x/2)}
    assert solve(x**2 - y**2/exp(x), y, x, dict=True) == [{y: x*exp(x/2)}]
    
    # 创建三个符号变量 x, y, z，并指定它们都是正数
    x, y, z = symbols('x y z', positive=True)
    
    # 解决方程 z**2*x**2 - z**2*y**2/exp(x)，期望的解是 [{y: x*exp(x/2)}]，
    # 因为 x, y, z 都是正数，所以解中不应包含 {y: -x*exp(x/2)}
    assert solve(z**2*x**2 - z**2*y**2/exp(x), y, x, z, dict=True) == [{y: x*exp(x/2)}]
def test_checking():
    # 断言解对表达式 x*(x - y/x) 求解结果的集合应该包含 {sqrt(y), S.Zero, -sqrt(y)}
    assert set(solve(x*(x - y/x), x, check=False)) == {sqrt(y), S.Zero, -sqrt(y)}
    # 断言解对表达式 x*(x - y/x) 求解结果的集合应该等于 {sqrt(y), -sqrt(y)}
    assert set(solve(x*(x - y/x), x, check=True)) == {sqrt(y), -sqrt(y)}
    # {x: 0, y: 4} 会使得 1/x 的分母为零，因此系统应该返回空列表
    assert solve((1/(1/x + 2), 1/(y - 3) - 1)) == []
    # 当 x 等于 0 时，1/x 的分母为零，因此返回 None
    assert solve(1/(1/x + 2)) == []


def test_issue_4671_4463_4467():
    # 断言解对表达式 sqrt(x**2 - 1) - 2 应该是 [sqrt(5), -sqrt(5)] 或者 [-sqrt(5), sqrt(5)]
    assert solve(sqrt(x**2 - 1) - 2) in ([sqrt(5), -sqrt(5)], [-sqrt(5), sqrt(5)])
    # 断言解对表达式 (2**exp(y**2/x) + 2)/(x**2 + 15) 关于 y 的解应该是 [-sqrt(x*log(1 + I*pi/log(2))), sqrt(x*log(1 + I*pi/log(2)))]
    assert solve((2**exp(y**2/x) + 2)/(x**2 + 15), y) == [-sqrt(x*log(1 + I*pi/log(2))), sqrt(x*log(1 + I*pi/log(2)))]
    
    C1, C2 = symbols('C1 C2')
    f = Function('f')
    # 断言解对表达式 C1 + C2/x**2 - exp(-f(x)) 关于 f(x) 的解应该是 [log(x**2/(C1*x**2 + C2))]
    assert solve(C1 + C2/x**2 - exp(-f(x)), f(x)) == [log(x**2/(C1*x**2 + C2))]
    
    a = Symbol('a')
    E = S.Exp1
    # 断言解对表达式 1 - log(a + 4*x**2) 关于 x 的解应该是 [-sqrt(-a + E)/2, sqrt(-a + E)/2] 或者 [sqrt(-a + E)/2, -sqrt(-a + E)/2]
    assert solve(1 - log(a + 4*x**2), x) in ([-sqrt(-a + E)/2, sqrt(-a + E)/2], [sqrt(-a + E)/2, -sqrt(-a + E)/2])
    # 断言解对表达式 log(a**(-3) - x**2)/a 关于 x 的解应该是 [-sqrt(-1 + a**(-3)), sqrt(-1 + a**(-3))] 或者 [sqrt(-1 + a**(-3)), -sqrt(-1 + a**(-3))]
    assert solve(log(a**(-3) - x**2)/a, x) in ([-sqrt(-1 + a**(-3)), sqrt(-1 + a**(-3))], [sqrt(-1 + a**(-3)), -sqrt(-1 + a**(-3))])
    # 断言解对表达式 1 - log(a + 4*x**2) 关于 x 的解应该是 [-sqrt(-a + E)/2, sqrt(-a + E)/2] 或者 [sqrt(-a + E)/2, -sqrt(-a + E)/2]
    assert solve(1 - log(a + 4*x**2), x) in ([-sqrt(-a + E)/2, sqrt(-a + E)/2], [sqrt(-a + E)/2, -sqrt(-a + E)/2])
    # 断言解对表达式 (a**2 + 1)*(sin(a*x) + cos(a*x)) 关于 x 的解应该是 [-pi/(4*a)]
    assert solve((a**2 + 1)*(sin(a*x) + cos(a*x)), x) == [-pi/(4*a)]
    # 断言解对表达式 3 - (sinh(a*x) + cosh(a*x)) 关于 x 的解应该是 [log(3)/a]
    assert solve(3 - (sinh(a*x) + cosh(a*x)), x) == [log(3)/a]
    # 断言解对表达式 3 - (sinh(a*x) + cosh(a*x)**2) 关于 x 的解应该是 {log(-2 + sqrt(5))/a, log(-sqrt(2) + 1)/a, log(-sqrt(5) - 2)/a, log(1 + sqrt(2))/a}
    assert set(solve(3 - (sinh(a*x) + cosh(a*x)**2), x)) == {log(-2 + sqrt(5))/a, log(-sqrt(2) + 1)/a, log(-sqrt(5) - 2)/a, log(1 + sqrt(2))/a}
    # 断言解对表达式 atan(x) - 1 关于 x 的解应该是 [tan(1)]
    assert solve(atan(x) - 1) == [tan(1)]


def test_issue_5132():
    r, t = symbols('r,t')
    # 断言解对方程组 [r - x**2 - y**2, tan(t) - y/x] 关于 [x, y] 的解应该是 {(-sqrt(r*cos(t)**2), -1*sqrt(r*cos(t)**2)*tan(t)), (sqrt(r*cos(t)**2), sqrt(r*cos(t)**2)*tan(t))}
    assert set(solve([r - x**2 - y**2, tan(t) - y/x], [x, y])) == {(-sqrt(r*cos(t)**2), -sqrt(r*cos(t)**2)*tan(t)), (sqrt(r*cos(t)**2), sqrt(r*cos(t)**2)*tan(t))}
    # 断言解对方程组 [exp(x) - sin(y), 1/y - 3] 关于 [x, y] 的解应该是 [(log(sin(Rational(1, 3))), Rational(1, 3))]
    assert solve([exp(x) - sin(y), 1/y - 3], [x, y]) == [(log(sin(Rational(1, 3))), Rational(1, 3))]
    # 断言解对方程组 [exp(x) - sin(y), 1/exp(y) - 3] 关于 [x, y] 的解应该是 [(log(-sin(log(3))), -log(3))]
    assert solve([exp(x) - sin(y), 1/exp(y) - 3], [x, y]) == [(log(-sin(log(3))), -log(3))]
    # 断言解对方程组 [exp(x) - sin(y), y**2 - 4] 关于 [x, y] 的解应该是 {(log(-sin(2)), -S(2)), (log(sin(2)), S(2))}
    assert set(solve([exp(x) - sin(y), y**2 - 4], [x, y])) == {(log(-sin(2)), -S(2)), (log(sin(2)), S(2))}
    # 断言解对方程组 [exp(x)**2 - sin(y) + z**2, 1/exp(y) - 3] 关于 [y, z] 的解应该是 ([y, z], {(-log(3), sqrt(-exp(2*x) - sin(log(3)))), (-log(3), -sqrt(-exp(2*x) - sin(log(3))))})
    assert solve([exp(x)**2 - sin(y) + z**2, 1/exp(y) - 3], set=True) == ([y, z], {(-log(3), sqrt(-exp(2*x) - sin(log(3)))), (-log(3), -sqrt(-exp(2*x) - sin(log(3))))})
    # 断言解对方程组 [exp(x)**2 - sin(y) + z, 1/exp(y) - 3] 关于 [x, z] 的解应该是 ([x, z], {(x, sqrt(-exp(2*x) + sin(y))), (x, -sqrt(-exp(2*x) + sin(y)))})
    assert solve([exp(x)**2 - sin(y) + z, 1/exp(y) - 3], x, z, set=True) == ([x, z], {(x, sqrt(-exp(2*x) + sin(y))), (x, -sqrt(-exp(2*x
    # 断言：解方程组 `eqs`，期望返回结果为 ([y, z], {(-log(3), -exp(2*x) - sin(log(3)))})
    assert solve(eqs, set=True) == ([y, z], {(-log(3), -exp(2*x) - sin(log(3)))})
    
    # 断言：解方程组 `eqs`，限定返回结果仅包括变量 x 和 z，期望返回 ([x, z], {(x, -exp(2*x) + sin(y))})
    assert solve(eqs, x, z, set=True) == ([x, z], {(x, -exp(2*x) + sin(y))})
    
    # 断言：解方程组 `eqs`，求解 x 和 y，期望结果是一个集合，包含两个元组
    # 每个元组表示一组解 (log(-sqrt(-z - sin(log(3)))), -log(3)) 和 (log(-z - sin(log(3)))/2, -log(3))
    assert set(solve(eqs, x, y)) == {
            (log(-sqrt(-z - sin(log(3)))), -log(3)),
            (log(-z - sin(log(3)))/2, -log(3))}
    
    # 断言：解方程组 `eqs`，求解 z 和 y，期望返回一个列表，包含一组解 (-exp(2*x) - sin(log(3)), -log(3))
    assert solve(eqs, z, y) == \
        [(-exp(2*x) - sin(log(3)), -log(3))]
    
    # 断言：解方程组 `(sqrt(x**2 + y**2) - sqrt(10), x + y - 4)`，期望返回结果为一个集合，
    # 包含两个元组 (S.One, S(3)) 和 (S(3), S.One)
    assert solve((sqrt(x**2 + y**2) - sqrt(10), x + y - 4), set=True) == (
        [x, y], {(S.One, S(3)), (S(3), S.One)})
    
    # 断言：解方程组 `(sqrt(x**2 + y**2) - sqrt(10), x + y - 4)`，求解 x 和 y，期望返回一个集合，
    # 包含两个元组 (S.One, S(3)) 和 (S(3), S.One)
    assert set(solve((sqrt(x**2 + y**2) - sqrt(10), x + y - 4), x, y)) == \
        {(S.One, S(3)), (S(3), S.One)}
def test_issue_5335():
    # 定义符号变量 lam, a0, conc
    lam, a0, conc = symbols('lam a0 conc')
    # 初始化常量 a 和 b
    a = 0.005
    b = 0.743436700916726
    # 定义方程组
    eqs = [lam + 2*y - a0*(1 - x/2)*x - a*x/2*x,
           a0*(1 - x/2)*x - 1*y - b*y,
           x + y - conc]
    # 定义符号变量列表
    sym = [x, y, a0]
    # 手动解方程组，仅保留最小解的个数为 2
    assert len(solve(eqs, sym, manual=True, minimal=True)) == 2
    # 自动解方程组，预期解的个数为 2，不使用有理数解
    assert len(solve(eqs, sym)) == 2  # cf below with rational=False


@SKIP("Hangs")
def _test_issue_5335_float():
    # 符号变量 lam, a0, conc
    lam, a0, conc = symbols('lam a0 conc')
    # 初始化常量 a 和 b
    a = 0.005
    b = 0.743436700916726
    # 定义方程组
    eqs = [lam + 2*y - a0*(1 - x/2)*x - a*x/2*x,
           a0*(1 - x/2)*x - 1*y - b*y,
           x + y - conc]
    # 定义符号变量列表
    sym = [x, y, a0]
    # 解方程组，预期解的个数为 2，不使用有理数解
    assert len(solve(eqs, sym, rational=False)) == 2


def test_issue_5767():
    # 解方程 x^2 + y + 4 = 0，解集为 {(-sqrt(-y - 4),), (sqrt(-y - 4),)}
    assert set(solve([x**2 + y + 4], [x])) == \
        {(-sqrt(-y - 4),), (sqrt(-y - 4),)}


def _make_example_24609():
    # 定义实数和正数符号变量 D, R, H, B_g, V, D_c
    D, R, H, B_g, V, D_c = symbols("D, R, H, B_g, V, D_c", real=True, positive=True)
    # 定义实数符号变量 Sigma_f, Sigma_a, nu
    Sigma_f, Sigma_a, nu = symbols("Sigma_f, Sigma_a, nu", real=True, positive=True)
    # 定义实数和正数符号变量 x
    x = symbols("x", real=True, positive=True)
    # 定义方程 eq
    eq = (
        2**(S(2)/3)*pi**(S(2)/3)*D_c*(S(231361)/10000 + pi**2/x**2)
        /(6*V**(S(2)/3)*x**(S(1)/3))
      - 2**(S(2)/3)*pi**(S(8)/3)*D_c/(2*V**(S(2)/3)*x**(S(7)/3))
    )
    # 定义预期值 expected
    expected = 100*sqrt(2)*pi/481
    return eq, expected, x


def test_issue_24609():
    # 解方程 eq 关于 x，简化后的解为 [expected]
    eq, expected, x = _make_example_24609()
    assert solve(eq, x, simplify=True) == [expected]
    # 数值近似解 solapprox
    [solapprox] = solve(eq.n(), x)
    assert abs(solapprox - expected.n()) < 1e-14


@XFAIL
def test_issue_24609_xfail():
    #
    # 这里返回 5 个解，但应该只有 1 个（x 为正数时）。
    # 简化后显示所有解是等价的。预期 solve 在某些情况下可能返回冗余解，
    # 但该方程核心是一个简单的二次方程，可以轻松解决而不引入多余的解：
    #
    #     >>> print(factor_terms(eq.as_numer_denom()[0]))
    #     2**(2/3)*pi**(2/3)*D_c*V**(2/3)*x**(7/3)*(231361*x**2 - 20000*pi**2)
    #
    eq, expected, x = _make_example_24609()
    assert len(solve(eq, x)) == [expected]
    #
    # 我们不希望仅通过使用 simplify 而通过此测试，
    # 如果以上通过，请取消下面附加测试的注释：
    #
    # assert len(solve(eq, x, simplify=False)) == 1


def test_polysys():
    # 解方程组 {x**2 + 2/y - 2, x + y - 3} 关于 {x, y}，解集为 {(1, 2), (1 + sqrt(5), 2 - sqrt(5)), (1 - sqrt(5), 2 + sqrt(5))}
    assert set(solve([x**2 + 2/y - 2, x + y - 3], [x, y])) == \
        {(S.One, S(2)), (1 + sqrt(5), 2 - sqrt(5)),
        (1 - sqrt(5), 2 + sqrt(5))}
    # 解方程组 {x**2 + y - 2, x**2 + y} 无解
    assert solve([x**2 + y - 2, x**2 + y]) == []
    # 结果顺序应与用户请求的顺序一致
    assert solve([x**2 + y - 3, x - y - 4], (x, y)) != solve([x**2 +
                 y - 3, x - y - 4], (y, x))


@slow
def test_unrad1():
    # 调用 raises 函数来测试未实现错误是否被正确抛出
    raises(NotImplementedError, lambda:
        unrad(sqrt(x) + sqrt(x + 1) + sqrt(1 - sqrt(x)) + 3))
    raises(NotImplementedError, lambda:
        unrad(sqrt(x) + (x + 1)**Rational(1, 3) + 2*sqrt(y)))

    # 创建一个符号变量 s，作为 Dummy 对象
    s = symbols('s', cls=Dummy)

    # 定义一个检查函数 check，用于检查结果是否满足特定条件
    # 处理结果可能存在符号变化的情况（参考问题 5203）
    def check(rv, ans):
        # 断言结果 rv[1] 和 ans[1] 的布尔值相同
        assert bool(rv[1]) == bool(ans[1])
        # 如果结果 ans[1] 为真，则调用 s_check 函数进一步检查
        if ans[1]:
            return s_check(rv, ans)
        # 展开 rv[0] 和 ans[0]，并比较它们是否相等或相反
        e = rv[0].expand()
        a = ans[0].expand()
        return e in [a, -a] and rv[1] == ans[1]

    # 定义 s_check 函数，用于检查结果中的 Dummy 对象替换
    def s_check(rv, ans):
        # 将 rv 中的 Dummy 对象替换为 s
        rv = list(rv)
        d = rv[0].atoms(Dummy)
        reps = list(zip(d, [s]*len(d)))
        rv = (rv[0].subs(reps).expand(), [rv[1][0].subs(reps), rv[1][1].subs(reps)])
        ans = (ans[0].subs(reps).expand(), [ans[1][0].subs(reps), ans[1][1].subs(reps)])
        # 检查 rv[0] 是否在 [ans[0], -ans[0]] 中，且 rv[1] 是否等于 ans[1]
        return str(rv[0]) in [str(ans[0]), str(-ans[0])] and \
            str(rv[1]) == str(ans[1])

    # 断言 unrad(1) 返回 None
    assert unrad(1) is None
    # 断言 check 函数对 unrad(sqrt(x)) 的返回结果符合预期 (x, [])
    assert check(unrad(sqrt(x)),
        (x, []))
    # 断言 check 函数对 unrad(sqrt(x) + 1) 的返回结果符合预期 (x - 1, [])
    assert check(unrad(sqrt(x) + 1),
        (x - 1, []))
    # 断言 check 函数对 unrad(sqrt(x) + root(x, 3) + 2*sqrt(y)) 的返回结果符合预期 (s**3 + s**2 + 2, [s, s**6 - x])
    assert check(unrad(sqrt(x) + root(x, 3) + 2*sqrt(y)),
        (s**3 + s**2 + 2, [s, s**6 - x]))
    # 断言 check 函数对 unrad(sqrt(x)*root(x, 3) + 2) 的返回结果符合预期 (x**5 - 64, [])
    assert check(unrad(sqrt(x)*root(x, 3) + 2),
        (x**5 - 64, []))
    # 断言 check 函数对 unrad(sqrt(x) + (x + 1)**Rational(1, 3)) 的返回结果符合预期 (x**3 - (x + 1)**2, [])
    assert check(unrad(sqrt(x) + (x + 1)**Rational(1, 3)),
        (x**3 - (x + 1)**2, []))
    # 断言 check 函数对 unrad(sqrt(x) + sqrt(x + 1) + sqrt(2*x)) 的返回结果符合预期 (-2*sqrt(2)*x - 2*x + 1, [])
    assert check(unrad(sqrt(x) + sqrt(x + 1) + sqrt(2*x)),
        (-2*sqrt(2)*x - 2*x + 1, []))
    # 断言 check 函数对 unrad(sqrt(x) + sqrt(x + 1) + 2) 的返回结果符合预期 (16*x - 9, [])
    assert check(unrad(sqrt(x) + sqrt(x + 1) + 2),
        (16*x - 9, []))
    # 断言 check 函数对 unrad(sqrt(x) + sqrt(x + 1) + sqrt(1 - x)) 的返回结果符合预期 (5*x**2 - 4*x, [])
    assert check(unrad(sqrt(x) + sqrt(x + 1) + sqrt(1 - x)),
        (5*x**2 - 4*x, []))
    # 断言 check 函数对 unrad(a*sqrt(x) + b*sqrt(x) + c*sqrt(y) + d*sqrt(y)) 的返回结果符合预期 ((a*sqrt(x) + b*sqrt(x))**2 - (c*sqrt(y) + d*sqrt(y))**2, [])
    assert check(unrad(a*sqrt(x) + b*sqrt(x) + c*sqrt(y) + d*sqrt(y)),
        ((a*sqrt(x) + b*sqrt(x))**2 - (c*sqrt(y) + d*sqrt(y))**2, []))
    # 断言 check 函数对 unrad(sqrt(x) + sqrt(1 - x)) 的返回结果符合预期 (2*x - 1, [])
    assert check(unrad(sqrt(x) + sqrt(1 - x)),
        (2*x - 1, []))
    # 断言 check 函数对 unrad(sqrt(x) + sqrt(1 - x) - 3) 的返回结果符合预期 (x**2 - x + 16, [])
    assert check(unrad(sqrt(x) + sqrt(1 - x) - 3),
        (x**2 - x + 16, []))
    # 断言 check 函数对 unrad(sqrt(x) + sqrt(1 - x) + sqrt(2 + x)) 的返回结果符合预期 (5*x**2 - 2*x + 1, [])
    assert check(unrad(sqrt(x) + sqrt(1 - x) + sqrt(2 + x)),
        (5*x**2 - 2*x + 1, []))
    # 断言 unrad(sqrt(x) + sqrt(1 - x) + sqrt(2 + x) - 3) 的返回结果在给定的可能值列表中
    assert unrad(sqrt(x) + sqrt(1 - x) + sqrt(2 + x) - 3) in [
        (25*x**4 + 376*x**3 + 1256*x**2 - 2272*x + 784, []),
        (25*x**8 - 476*x**6 + 2534*x**4 - 1468*x**2 + 169, [])]
    # 断言 unrad(sqrt(x) + sqrt(1 - x) + sqrt(2 + x) - sqrt(1 - 2*x)) 的返回结果符合预期 (41*x**4 + 40*x**3 + 232*x**2 - 160*x + 16, [])
    assert unrad(sqrt(x) + sqrt(1 - x) + sqrt(2 + x) - sqrt(1 - 2*x)) == \
        (41*x**4 + 40*x**3 + 232*x**2 - 160*x + 16, [])  # orig root at 0.487
    # 断言 check 函数对 unrad(sqrt(x) + sqrt(x + 1)) 的返回结果符合预期 (1, [])
    assert check(unrad(sqrt(x) + sqrt(x + 1)), (S.One, []))

    # 定义等式 eq
    eq = sqrt(x) + sqrt(x + 1) + sqrt(1 - sqrt(x))
    # 断言 check 函数对 unrad(eq) 的返回结果符合预期 (16*x**2 - 9*x, [])
    assert check(unrad(eq),
        (16*x**2 - 9*x, []))
    # 断言 solve 函数对 eq 的结果集合包含 {S.Zero, Rational(9, 16)}
    assert set(solve(eq, check=False)) == {S.Zero, Rational(9, 16)}
    # 断言 solve 函数对 eq 的结果为空列表
    assert solve(eq) == []
    # 断言 solve 函数对 sqrt(x) - sqrt(x + 1) + sqrt(1 - sqrt(x)) 的结果集合包含 {S.Zero, Rational(9, 16)}
    assert set(solve(sqrt(x) - sqrt(x + 1) + sqrt(1 - sqrt(x)))) == \
        {
    # 使用自定义的 check 函数验证解是否正确
    assert check(unrad(sqrt(x/(1 - x)) + (x + 1)**Rational(1, 3)),
        (x**5 - x**4 - x**3 + 2*x**2 + x - 1, []))
    # 使用自定义的 check 函数验证解是否正确
    assert check(unrad(sqrt(x/(1 - x)) + 2*sqrt(y), y),
        (4*x*y + x - 4*y, []))
    # 使用自定义的 check 函数验证解是否正确
    assert check(unrad(sqrt(x)*sqrt(1 - x) + 2, x),
        (x**2 - x + 4, []))

    # 使用 solve 函数解方程 Eq(x, sqrt(x + 6))，期望得到解 [3]
    assert solve(Eq(x, sqrt(x + 6))) == [3]
    # 使用 solve 函数解方程 Eq(x + sqrt(x - 4), 4)，期望得到解 [4]
    assert solve(Eq(x + sqrt(x - 4), 4)) == [4]
    # 使用 solve 函数解方程 Eq(1, x + sqrt(2*x - 3))，期望得到空列表，即无解
    assert solve(Eq(1, x + sqrt(2*x - 3))) == []
    # 使用 solve 函数解方程 Eq(sqrt(5*x + 6) - 2, x)，期望得到解集 {-1, 2}
    assert set(solve(Eq(sqrt(5*x + 6) - 2, x))) == {-S.One, S(2)}
    # 使用 solve 函数解方程 Eq(sqrt(2*x - 1) - sqrt(x - 4), 2)，期望得到解集 {5, 13}
    assert set(solve(Eq(sqrt(2*x - 1) - sqrt(x - 4), 2))) == {S(5), S(13)}
    # 使用 solve 函数解方程 Eq(sqrt(x + 7) + 2, sqrt(3 - x))，期望得到解 [-6]
    assert solve(Eq(sqrt(x + 7) + 2, sqrt(3 - x))) == [-6]
    # 使用 solve 函数解方程 (2*x - 5)**Rational(1, 3) - 3，期望得到解 [16]
    assert solve((2*x - 5)**Rational(1, 3) - 3) == [16]
    # 使用 solve 函数解方程 x + 1 - root(x**4 + 4*x**3 - x, 4)，期望得到解集 {-1/2, -1/3}
    assert set(solve(x + 1 - root(x**4 + 4*x**3 - x, 4))) == \
        {Rational(-1, 2), Rational(-1, 3)}
    # 使用 solve 函数解方程 sqrt(2*x**2 - 7) - (3 - x)，期望得到解集 {-8, 2}
    assert set(solve(sqrt(2*x**2 - 7) - (3 - x))) == {-S(8), S(2)}
    # 使用 solve 函数解方程 sqrt(x + 4) + sqrt(2*x - 1) - 3*sqrt(x - 1)，期望得到解 [0]
    assert solve(sqrt(x + 4) + sqrt(2*x - 1) - 3*sqrt(x - 1)) == [0]
    # 使用 solve 函数解方程 sqrt(x)*sqrt(x - 7) - 12，期望得到解 [16]
    assert solve(sqrt(x)*sqrt(x - 7) - 12) == [16]
    # 使用 solve 函数解方程 sqrt(x - 3) + sqrt(x) - 3，期望得到解 [4]
    assert solve(sqrt(x - 3) + sqrt(x) - 3) == [4]
    # 使用 solve 函数解方程 sqrt(9*x**2 + 4) - (3*x + 2)，期望得到解 [0]
    assert solve(sqrt(9*x**2 + 4) - (3*x + 2)) == [0]
    # 使用 solve 函数解方程 sqrt(x) - 2 - 5，期望得到解 [49]
    assert solve(sqrt(x) - 2 - 5) == [49]
    # 使用 solve 函数解方程 sqrt(x - 3) - sqrt(x) - 3，期望得到空列表，即无解
    assert solve(sqrt(x - 3) - sqrt(x) - 3) == []
    # 使用 solve 函数解方程 sqrt(x - 1) - x + 7，期望得到解 [10]
    assert solve(sqrt(x - 1) - x + 7) == [10]
    # 使用 solve 函数解方程 sqrt(x - 2) - 5，期望得到解 [27]
    assert solve(sqrt(x - 2) - 5) == [27]
    # 使用 solve 函数解方程 sqrt(17*x - sqrt(x**2 - 5)) - 7，期望得到解 [3]
    assert solve(sqrt(17*x - sqrt(x**2 - 5)) - 7) == [3]
    # 使用 solve 函数解方程 sqrt(x) - sqrt(x - 1) + sqrt(sqrt(x))，期望得到空列表，即无解
    assert solve(sqrt(x) - sqrt(x - 1) + sqrt(sqrt(x))) == []

    # 不要对 unrad 中的表达式进行 posify，并使用 _mexpand
    z = sqrt(2*x + 1)/sqrt(x) - sqrt(2 + 1/x)
    p = posify(z)[0]
    # 使用 solve 函数解方程 p，期望得到空列表，即无解
    assert solve(p) == []
    # 使用 solve 函数解方程 z，期望得到空列表，即无解
    assert solve(z) == []
    # 使用 solve 函数解方程 z + 6*I，期望得到解 [-1/11]
    assert solve(z + 6*I) == [Rational(-1, 11)]
    # 使用 solve 函数解方程 p + 6*I，期望得到空列表，即无解
    assert solve(p + 6*I) == []

    # issue 8622 的特殊情况
    assert unrad(root(x + 1, 5) - root(x, 3)) == (
        -(x**5 - x**3 - 3*x**2 - 3*x - 1), [])
    # issue #8679 的特殊情况
    assert check(unrad(x + root(x, 3) + root(x, 3)**2 + sqrt(y), x),
        (s**3 + s**2 + s + sqrt(y), [s, s**3 - x]))

    # 用于覆盖率测试的情况
    assert check(unrad(sqrt(x) + root(x, 3) + y),
        (s**3 + s**2 + y, [s, s**6 - x]))
    # 使用 solve 函数解方程 sqrt(x) + root(x, 3) - 2，期望得到解 [1]
    assert solve(sqrt(x) + root(x, 3) - 2) == [1]
    # 抛出 NotImplementedError 异常，尝试解方程 sqrt(x) + root(x, 3) + root(x + 1, 5) - 2
    raises(NotImplementedError, lambda:
        solve(sqrt(x) + root(x, 3) + root(x + 1, 5) - 2))
    # 通过不同的代码路径失败
    raises(NotImplementedError, lambda: solve(-sqrt(2) + cosh(x)/x))
    # unrad 一些特殊情况
    assert solve(sqrt(x + root(x, 3))+root(x - y, 5), y) == [
        x + (x**Rational(1, 3) + x)**Rational(5, 2)]
    # 使用自定义的 check 函数验证解是否正确
    assert check(unrad(sqrt(x) - root(x + 1, 3)*sqrt(x + 2) + 2),
        (s**10 + 8*s**8 + 24*s**6 - 12*s**5 - 22*s**4 - 160*s**3 - 212*s**2 -
        192*s - 56, [s, s**2 - x]))
    # 定义 e 为表达式 root(x + 1, 3) + root(x, 3)
    e = root(x + 1, 3) + root(x, 3)
    # 使用 unrad
    # 检查 unrad 函数处理给定方程时的结果是否符合预期
    assert check(unrad(eq),
        (15625*x**4 + 173000*x**3 + 355600*x**2 - 817920*x + 331776, []))

    # 检查 unrad 函数处理根式表达式的结果是否符合预期
    assert check(unrad(root(x, 4) + root(x, 4)**3 - 1),
        (s**3 + s - 1, [s, s**4 - x]))

    # 检查 unrad 函数处理根式表达式的结果是否符合预期
    assert check(unrad(root(x, 2) + root(x, 2)**3 - 1),
        (x**3 + 2*x**2 + x - 1, []))

    # 检查 unrad 函数处理平方根的情况下返回 None 是否符合预期
    assert unrad(x**0.5) is None

    # 检查 unrad 函数处理包含两个变量的根式表达式的结果是否符合预期
    assert check(unrad(t + root(x + y, 5) + root(x + y, 5)**3),
        (s**3 + s + t, [s, s**5 - x - y]))

    # 检查 unrad 函数处理包含特定变量的根式表达式的结果是否符合预期
    assert check(unrad(x + root(x + y, 5) + root(x + y, 5)**3, y),
        (s**3 + s + x, [s, s**5 - x - y]))

    # 检查 unrad 函数处理包含特定变量的根式表达式的结果是否符合预期
    assert check(unrad(x + root(x + y, 5) + root(x + y, 5)**3, x),
        (s**5 + s**3 + s - y, [s, s**5 - x - y]))

    # 检查 unrad 函数处理复杂根式表达式的结果是否符合预期
    assert check(unrad(root(x - 1, 3) + root(x + 1, 5) + root(2, 5)),
        (s**5 + 5*2**Rational(1, 5)*s**4 + s**3 + 10*2**Rational(2, 5)*s**3 +
        10*2**Rational(3, 5)*s**2 + 5*2**Rational(4, 5)*s + 4, [s, s**3 - x + 1]))

    # 检查 unrad 函数处理未实现的情况是否会引发 NotImplementedError
    raises(NotImplementedError, lambda:
        unrad((root(x, 2) + root(x, 3) + root(x, 4)).subs(x, x**5 - x + 1)))

    # 检查 unrad 函数在 simplify 标志未设置为 False 时的结果是否符合预期
    assert solve(root(x, 3) + root(x, 5) - 2) == [1]

    # 检查 unrad 函数处理复杂方程的结果是否符合预期
    eq = (sqrt(x) + sqrt(x + 1) + sqrt(1 - x) - 6*sqrt(5)/5)
    assert check(unrad(eq),
        ((5*x - 4)*(3125*x**3 + 37100*x**2 + 100800*x - 82944), []))

    # 检查 solve 函数处理复杂方程的结果是否符合预期
    ans = S('''
        [4/5, -1484/375 + 172564/(140625*(114*sqrt(12657)/78125 +
        12459439/52734375)**(1/3)) +
        4*(114*sqrt(12657)/78125 + 12459439/52734375)**(1/3)]''')
    assert solve(eq) == ans

    # 检查 unrad 函数处理包含重复根式的表达式的结果是否符合预期
    assert check(unrad(sqrt(x + root(x + 1, 3)) - root(x + 1, 3) - 2),
        (s**3 - s**2 - 3*s - 5, [s, s**3 - x - 1]))

    # 检查 unrad 函数处理复杂根式表达式的结果是否符合预期
    e = root(x**2 + 1, 3) - root(x**2 - 1, 5) - 2
    assert check(unrad(e),
        (s**5 - 10*s**4 + 39*s**3 - 80*s**2 + 80*s - 30,
        [s, s**3 - x**2 - 1]))

    # 检查 unrad 函数处理复杂根式表达式的结果是否符合预期
    e = sqrt(x + root(x + 1, 2)) - root(x + 1, 3) - 2
    assert check(unrad(e),
        (s**6 - 2*s**5 - 7*s**4 - 3*s**3 + 26*s**2 + 40*s + 25,
        [s, s**3 - x - 1]))

    # 检查 unrad 函数处理复杂根式表达式的结果是否符合预期（反向处理）
    assert check(unrad(e, _reverse=True),
        (s**6 - 14*s**5 + 73*s**4 - 187*s**3 + 276*s**2 - 228*s + 89,
        [s, s**2 - x - sqrt(x + 1)]))

    # 检查 unrad 函数处理包含特定根式的表达式的结果是否符合预期
    assert check(unrad(sqrt(x + sqrt(root(x, 3) - 1)) - root(x, 6) - 2),
        (s**12 - 2*s**8 - 8*s**7 - 8*s**6 + s**4 + 8*s**3 + 23*s**2 +
        32*s + 17, [s, s**6 - x]))

    # 检查 unrad 函数处理复杂表达式的结果是否符合预期
    assert unrad(root(cosh(x), 3)/x*root(x + 1, 5) - 1) == (
        -(x**15 - x**3*cosh(x)**5 - 3*x**2*cosh(x)**5 - 3*x*cosh(x)**5
        - cosh(x)**5), [])

    # 检查 unrad 函数处理复杂表达式的结果是否符合预期（预期失败的情况）
    #assert unrad(sqrt(cosh(x)/x) + root(x + 1, 3)*sqrt(x) - 1) == (
    #    -s**6 + 6*s**5 - 15*s**4 + 20*s**3 - 15*s**2 + 6*s + x**5 +
    #    2*x**4 + x**3 - 1, [s, s**2 - cosh(x)/x])

    # 检查 unrad 函数处理包含指数中包含符号的表达式的情况是否会返回 None
    assert unrad(S('(x+y)**(2*y/3) + (x+y)**(1/3) + 1')) is None
    # 使用 assert 语句检查表达式的正确性，确保 unrad 函数应用于 '(x+y)**(2*y/3) + (x+y)**(1/3) + 1' 结果与期望的解相符
    assert check(unrad(S('(x+y)**(2*y/3) + (x+y)**(1/3) + 1'), x),
        (s**(2*y) + s + 1, [s, s**3 - x - y]))
    
    # 这个断言检查 unrad 函数是否正确处理 x**(S.Half/y) + y 表达式，返回的结果应该是 (x**(1/y) - y**2, [])
    assert unrad(x**(S.Half/y) + y, x) == (x**(1/y) - y**2, [])

    # 此处的注释解释了这个断言的目的：测试 solve 函数对 sqrt(y)*x + x**3 - 1 表达式的求解，预期结果长度为 3
    assert len(solve(sqrt(y)*x + x**3 - 1, x)) == 3
    
    # 这个断言测试 solve 函数对 -512*y**3 + 1344*(x + 2)**Rational(1, 3)*y**2 - 1176*(x + 2)**Rational(2, 3)*y - 169*x + 686 表达式的求解，关闭 _unrad 选项
    assert len(solve(-512*y**3 + 1344*(x + 2)**Rational(1, 3)*y**2 -
        1176*(x + 2)**Rational(2, 3)*y - 169*x + 686, y, _unrad=False)) == 3

    # 创建一个方程对象 eq，用于测试 solve 函数解决 '-x + (7*y/8 - (27*x/2 + 27*sqrt(x**2)/2)**(1/3)/3)**3 - 1' 表达式
    eq = S('-x + (7*y/8 - (27*x/2 + 27*sqrt(x**2)/2)**(1/3)/3)**3 - 1')
    assert solve(eq, y) == [
        2**(S(2)/3)*(27*x + 27*sqrt(x**2))**(S(1)/3)*S(4)/21 + (512*x/343 +
        S(512)/343)**(S(1)/3)*(-S(1)/2 - sqrt(3)*I/2), 2**(S(2)/3)*(27*x +
        27*sqrt(x**2))**(S(1)/3)*S(4)/21 + (512*x/343 +
        S(512)/343)**(S(1)/3)*(-S(1)/2 + sqrt(3)*I/2), 2**(S(2)/3)*(27*x +
        27*sqrt(x**2))**(S(1)/3)*S(4)/21 + (512*x/343 + S(512)/343)**(S(1)/3)]

    # 创建方程对象 eq，用于测试 unrad 函数对 root(x + 1, 3) - (root(x, 3) + root(x, 5)) 表达式的处理
    eq = root(x + 1, 3) - (root(x, 3) + root(x, 5))
    assert check(unrad(eq),
        (3*s**13 + 3*s**11 + s**9 - 1, [s, s**15 - x]))
    
    # 创建方程对象 eq，测试 unrad 函数对 eq - 2 表达式的处理
    assert check(unrad(eq - 2),
        (3*s**13 + 3*s**11 + 6*s**10 + s**9 + 12*s**8 + 6*s**6 + 12*s**5 +
        12*s**3 + 7, [s, s**15 - x]))
    
    # 创建方程对象 eq，测试 unrad 函数对 root(x, 3) - root(x + 1, 4)/2 + root(x + 2, 3) 表达式的处理
    assert check(unrad(root(x, 3) - root(x + 1, 4)/2 + root(x + 2, 3)),
        (s*(4096*s**9 + 960*s**8 + 48*s**7 - s**6 - 1728),
        [s, s**4 - x - 1]))  # orig expr has two real roots: -1, -.389
    
    # 创建方程对象 eq，测试 unrad 函数对 root(x, 3) + root(x + 1, 4) - root(x + 2, 3)/2 表达式的处理
    assert check(unrad(root(x, 3) + root(x + 1, 4) - root(x + 2, 3)/2),
        (343*s**13 + 2904*s**12 + 1344*s**11 + 512*s**10 - 1323*s**9 -
        3024*s**8 - 1728*s**7 + 1701*s**5 + 216*s**4 - 729*s, [s, s**4 - x -
        1]))  # orig expr has one real root: -0.048
    
    # 创建方程对象 eq，测试 unrad 函数对 root(x, 3)/2 - root(x + 1, 4) + root(x + 2, 3) 表达式的处理
    assert check(unrad(root(x, 3)/2 - root(x + 1, 4) + root(x + 2, 3)),
        (729*s**13 - 216*s**12 + 1728*s**11 - 512*s**10 + 1701*s**9 -
        3024*s**8 + 1344*s**7 + 1323*s**5 - 2904*s**4 + 343*s, [s, s**4 - x -
        1]))  # orig expr has 2 real roots: -0.91, -0.15
    
    # 创建方程对象 eq，测试 unrad 函数对 root(x, 3)/2 - root(x + 1, 4) + root(x + 2, 3) - 2 表达式的处理
    assert check(unrad(root(x, 3)/2 - root(x + 1, 4) + root(x + 2, 3) - 2),
        (729*s**13 + 1242*s**12 + 18496*s**10 + 129701*s**9 + 388602*s**8 +
        453312*s**7 - 612864*s**6 - 3337173*s**5 - 6332418*s**4 - 7134912*s**3
        - 5064768*s**2 - 2111913*s - 398034, [s, s**4 - x - 1]))
        # orig expr has 1 real root: 19.53
    
    # 解决 sqrt(x) + sqrt(x + 1) - sqrt(1 - x) - sqrt(2 + x) 优化问题
    ans = solve(sqrt(x) + sqrt(x + 1) -
                sqrt(1 - x) - sqrt(2 + x))
    # 断言解的数量为1，并且数值表示近似为 '0.73'
    assert len(ans) == 1 and NS(ans[0])[:4] == '0.73'
    
    # 创建符号 F，用于解决方程 F - (2*x + 2*y + sqrt(x**2 + y**2)) 的优化问题
    F = Symbol('F')
    eq = F - (2*x + 2*y + sqrt(x**2 + y**2))
    ans = F*Rational(2, 7) - sqrt(2)*F/14
    # 解方程 eq 关于 x，但不进行检查（check=False）
    X = solve(eq, x, check=False)
    for xi in reversed(X):  # 逆序遍历 X 列表中的元素
        # 使用 xi 替换 x，对 (x*y).diff(y) 求导，并简化计算
        Y = solve((x*y).subs(x, xi).diff(y), y, simplify=False, check=False)
        # 如果 Y 中的任何表达式与 ans 相同（经展开后为零），则跳出循环
        if any((a - ans).expand().is_zero for a in Y):
            break
    else:
        assert None  # 如果没有找到答案，则断言错误

    assert solve(sqrt(x + 1) + root(x, 3) - 2) == S('''
        [(-11/(9*(47/54 + sqrt(93)/6)**(1/3)) + 1/3 + (47/54 +
        sqrt(93)/6)**(1/3))**3]''')  # 断言求解结果等于给定的 SymPy 表达式

    assert solve(sqrt(sqrt(x + 1)) + x**Rational(1, 3) - 2) == S('''
        [(-sqrt(-2*(-1/16 + sqrt(6913)/16)**(1/3) + 6/(-1/16 +
        sqrt(6913)/16)**(1/3) + 17/2 + 121/(4*sqrt(-6/(-1/16 +
        sqrt(6913)/16)**(1/3) + 2*(-1/16 + sqrt(6913)/16)**(1/3) + 17/4)))/2 +
        sqrt(-6/(-1/16 + sqrt(6913)/16)**(1/3) + 2*(-1/16 +
        sqrt(6913)/16)**(1/3) + 17/4)/2 + 9/4)**3]''')  # 断言求解结果等于给定的 SymPy 表达式

    assert solve(sqrt(x) + root(sqrt(x) + 1, 3) - 2) == S('''
        [(-(81/2 + 3*sqrt(741)/2)**(1/3)/3 + (81/2 + 3*sqrt(741)/2)**(-1/3) +
        2)**2]''')  # 断言求解结果等于给定的 SymPy 表达式

    eq = S('''
        -x + (1/2 - sqrt(3)*I/2)*(3*x**3/2 - x*(3*x**2 - 34)/2 + sqrt((-3*x**3
        + x*(3*x**2 - 34) + 90)**2/4 - 39304/27) - 45)**(1/3) + 34/(3*(1/2 -
        sqrt(3)*I/2)*(3*x**3/2 - x*(3*x**2 - 34)/2 + sqrt((-3*x**3 + x*(3*x**2
        - 34) + 90)**2/4 - 39304/27) - 45)**(1/3))''')
    assert check(unrad(eq),
        (s*-(-s**6 + sqrt(3)*s**6*I - 153*2**Rational(2, 3)*3**Rational(1, 3)*s**4 +
        51*12**Rational(1, 3)*s**4 - 102*2**Rational(2, 3)*3**Rational(5, 6)*s**4*I - 1620*s**3 +
        1620*sqrt(3)*s**3*I + 13872*18**Rational(1, 3)*s**2 - 471648 +
        471648*sqrt(3)*I), [s, s**3 - 306*x - sqrt(3)*sqrt(31212*x**2 -
        165240*x + 61484) + 810]))  # 断言使用 unrad 函数检查等式的解

    assert solve(eq) == []  # 断言解方程 eq 后结果为空列表，即没有其它代码错误

    eq = root(x, 3) - root(y, 3) + root(x, 5)
    assert check(unrad(eq),
           (s**15 + 3*s**13 + 3*s**11 + s**9 - y, [s, s**15 - x]))  # 断言使用 unrad 函数检查等式的解

    eq = root(x, 3) + root(y, 3) + root(x*y, 4)
    assert check(unrad(eq),
                 (s*y*(-s**12 - 3*s**11*y - 3*s**10*y**2 - s**9*y**3 -
                       3*s**8*y**2 + 21*s**7*y**3 - 3*s**6*y**4 - 3*s**4*y**4 -
                       3*s**3*y**5 - y**6), [s, s**4 - x*y]))  # 断言使用 unrad 函数检查等式的解

    raises(NotImplementedError,
           lambda: unrad(root(x, 3) + root(y, 3) + root(x*y, 5)))  # 断言调用 unrad 函数会抛出 NotImplementedError 异常

    # 测试带有等式的 unrad 函数
    eq = Eq(-x**(S(1)/5) + x**(S(1)/3), -3**(S(1)/3) - (-1)**(S(3)/5)*3**(S(1)/5))
    assert check(unrad(eq),
        (-s**5 + s**3 - 3**(S(1)/3) - (-1)**(S(3)/5)*3**(S(1)/5), [s, s**15 - x]))  # 断言使用 unrad 函数检查等式的解

    # 确保隐藏的根号被暴露出来
    s = sqrt(x) - 1
    assert unrad(s**2 - s**3) == (x**3 - 6*x**2 + 9*x - 4, [])  # 断言 unrad 函数暴露了隐藏的根号

    # 确保已经是多项式的分子被拒绝
    assert unrad((x/(x + 1) + 3)**(-2), x) is None  # 断言 unrad 函数拒绝了已经是多项式的分子

    # https://github.com/sympy/sympy/issues/23707
    eq = sqrt(x - y)*exp(t*sqrt(x - y)) - exp(t*sqrt(x - y))
    assert solve(eq, y) == [x - 1]  # 断言求解等式 eq 后的结果为 [x - 1]
    assert unrad(eq) is None  # 断言 unrad 函数不适用于该等式
@slow
def test_unrad_slow():
    # 根据多重根的情况，这里的方程求解应该不会得到重复的根
    eq = (sqrt(1 + sqrt(1 - 4*x**2)) - x*(1 + sqrt(1 + 2*sqrt(1 - 4*x**2))))
    # 断言求解方程的结果应该是 [S.Half]
    assert solve(eq) == [S.Half]


@XFAIL
def test_unrad_fail():
    # 这个测试只有在检查 real_root(eq.subs(x, Rational(1, 3))) 的情况下才有效
    # 但 checksol 不能像这样工作
    assert solve(root(x**3 - 3*x**2, 3) + 1 - x) == [Rational(1, 3)]
    assert solve(root(x + 1, 3) + root(x**2 - 2, 5) + 1) == [
        -1, -1 + CRootOf(x**5 + x**4 + 5*x**3 + 8*x**2 + 10*x + 5, 0)**3]


def test_checksol():
    x, y, r, t = symbols('x, y, r, t')
    eq = r - x**2 - y**2
    # 检查符号变量的解是否符合方程
    dict_var_soln = {y: - sqrt(r) / sqrt(tan(t)**2 + 1),
        x: -sqrt(r)*tan(t)/sqrt(tan(t)**2 + 1)}
    assert checksol(eq, dict_var_soln) == True
    # 检查特定的逻辑表达式是否为 True
    assert checksol(Eq(x, False), {x: False}) is True
    assert checksol(Ne(x, False), {x: False}) is False
    assert checksol(Eq(x < 1, True), {x: 0}) is True
    assert checksol(Eq(x < 1, True), {x: 1}) is False
    assert checksol(Eq(x < 1, False), {x: 1}) is True
    assert checksol(Eq(x < 1, False), {x: 0}) is False
    assert checksol(Eq(x + 1, x**2 + 1), {x: 1}) is True
    assert checksol([x - 1, x**2 - 1], x, 1) is True
    assert checksol([x - 1, x**2 - 2], x, 1) is False
    assert checksol(Poly(x**2 - 1), x, 1) is True
    assert checksol(0, {}) is True
    assert checksol([1e-10, x - 2], x, 2) is False
    assert checksol([0.5, 0, x], x, 0) is False
    assert checksol(y, x, 2) is False
    assert checksol(x+1e-10, x, 0, numerical=True) is True
    assert checksol(x+1e-10, x, 0, numerical=False) is False
    assert checksol(exp(92*x), {x: log(sqrt(2)/2)}) is False
    assert checksol(exp(92*x), {x: log(sqrt(2)/2) + I*pi}) is False
    assert checksol(1/x**5, x, 1000) is False
    raises(ValueError, lambda: checksol(x, 1))
    raises(ValueError, lambda: checksol([], x, 1))


def test__invert():
    # 测试反函数求解
    assert _invert(x - 2) == (2, x)
    assert _invert(2) == (2, 0)
    assert _invert(exp(1/x) - 3, x) == (1/log(3), x)
    assert _invert(exp(1/x + a/x) - 3, x) == ((a + 1)/log(3), x)
    assert _invert(a, x) == (a, 0)


def test_issue_4463():
    # 测试解决特定问题
    assert solve(-a*x + 2*x*log(x), x) == [exp(a/2)]
    assert solve(x**x) == []
    assert solve(x**x - 2) == [exp(LambertW(log(2)))]
    assert solve(((x - 3)*(x - 2))**((x - 3)*(x - 4))) == [2]


@slow
def test_issue_5114_solvers():
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r = symbols('a:r')

    # 在方程组中没有 'a'，但问题最初是这样提出的
    syms = a, b, c, f, h, k, n
    eqs = [b + r/d - c/d,
        c*(1/d + 1/e + 1/g) - f/g - r/d,
        f*(1/g + 1/i + 1/j) - c/g - h/i,
        h*(1/i + 1/l + 1/m) - f/i - k/m,
        k*(1/m + 1/o + 1/p) - h/m - n/p,
        n*(1/p + 1/q) - k/p]
    # 断言解决方程组后返回的结果长度为 1
    assert len(solve(eqs, syms, manual=True, check=False, simplify=False)) == 1


def test_issue_5849():
    # 待添加
    pass
    # XXX: This system does not have a solution for most values of the
    # parameters. Generally solve returns the empty set for systems that are
    # generically inconsistent.
    #
    # 定义符号变量 I1 到 I6 和 dI1, dI4, dQ2, dQ4, Q2, Q4
    I1, I2, I3, I4, I5, I6 = symbols('I1:7')
    dI1, dI4, dQ2, dQ4, Q2, Q4 = symbols('dI1,dI4,dQ2,dQ4,Q2,Q4')

    # 定义方程组 e，表示系统的一组方程
    e = (
        I1 - I2 - I3,
        I3 - I4 - I5,
        I4 + I5 - I6,
        -I1 + I2 + I6,
        -2*I1 - 2*I3 - 2*I5 - 3*I6 - dI1/2 + 12,
        -I4 + dQ4,
        -I2 + dQ2,
        2*I3 + 2*I5 + 3*I6 - Q2,
        I4 - 2*I5 + 2*Q4 + dI4
    )

    # 定义方程组的一个解 ans，是一个包含字典的列表
    ans = [{
        I1: I2 + I3,
        dI1: -4*I2 - 8*I3 - 4*I5 - 6*I6 + 24,
        I4: I3 - I5,
        dQ4: I3 - I5,
        Q4: -I3/2 + 3*I5/2 - dI4/2,
        dQ2: I2,
        Q2: 2*I3 + 2*I5 + 3*I6
    }]

    # 定义变量 v，包含 I1 到 dQ4 这些符号变量
    v = I1, I4, Q2, Q4, dI1, dI4, dQ2, dQ4

    # 断言求解方程组 e，使用符号变量 v，手动模式，不检查，返回字典形式的解等于 ans
    assert solve(e, *v, manual=True, check=False, dict=True) == ans

    # 断言求解方程组 e，使用符号变量 v，手动模式，不检查，返回每个变量的解列表等于 ans
    assert solve(e, *v, manual=True, check=False) == [
        tuple([a.get(i, i) for i in v]) for a in ans]

    # 断言求解方程组 e，使用符号变量 v，手动模式，返回空列表（即没有解）
    assert solve(e, *v, manual=True) == []

    # 断言求解方程组 e，使用符号变量 v，返回空列表（即没有解）
    assert solve(e, *v) == []

    # 对于每个方程 ei 在 ans[0] 中代入，检查结果是否为零
    assert [ei.subs(ans[0]) for ei in e] == [0, 0, I3 - I6, -I3 + I6, 0, 0, 0, 0, 0]

    # 矩阵求解器不喜欢这种情况，因为它在矩阵中产生了一个零行。这与问题 4551 有关吗？
def test_issue_5849_matrix():
    '''Same as test_issue_5849 but solved with the matrix solver.

    A solution only exists if I3 == I6 which is not generically true,
    but `solve` does not return conditions under which the solution is
    valid, only a solution that is canonical and consistent with the input.
    '''
    # 定义符号变量 I1 到 I6 和 dI1 到 dQ4
    I1, I2, I3, I4, I5, I6 = symbols('I1:7')
    dI1, dI4, dQ2, dQ4, Q2, Q4 = symbols('dI1,dI4,dQ2,dQ4,Q2,Q4')

    # 定义线性方程组 e
    e = (
        I1 - I2 - I3,                   # 方程1
        I3 - I4 - I5,                   # 方程2
        I4 + I5 - I6,                   # 方程3
        -I1 + I2 + I6,                  # 方程4
        -2*I1 - 2*I3 - 2*I5 - 3*I6 - dI1/2 + 12,  # 方程5
        -I4 + dQ4,                      # 方程6
        -I2 + dQ2,                      # 方程7
        2*I3 + 2*I5 + 3*I6 - Q2,         # 方程8
        I4 - 2*I5 + 2*Q4 + dI4          # 方程9
    )
    # 解方程组 e，期望得到空列表作为解
    assert solve(e, I1, I4, Q2, Q4, dI1, dI4, dQ2, dQ4) == []


def test_issue_21882():

    a, b, c, d, f, g, k = unknowns = symbols('a, b, c, d, f, g, k')

    # 定义方程组 equations
    equations = [
        -k*a + b + 5*f/6 + 2*c/9 + 5*d/6 + 4*a/3,   # 方程1
        -k*f + 4*f/3 + d/2,                        # 方程2
        -k*d + f/6 + d,                            # 方程3
        13*b/18 + 13*c/18 + 13*a/18,                # 方程4
        -k*c + b/2 + 20*c/9 + a,                    # 方程5
        -k*b + b + c/18 + a/6,                      # 方程6
        5*b/3 + c/3 + a,                            # 方程7
        2*b/3 + 2*c + 4*a/3,                        # 方程8
        -g                                          # 方程9
    ]

    # 定义预期的答案 answer
    answer = [
        {a: 0, f: 0, b: 0, d: 0, c: 0, g: 0},         # 答案1
        {a: 0, f: -d, b: 0, k: S(5)/6, c: 0, g: 0},   # 答案2
        {a: -2*c, f: 0, b: c, d: 0, k: S(13)/18, g: 0}  # 答案3
    ]
    # 断言 solve 函数返回的结果与预期答案相等
    assert solve(equations, unknowns, dict=True) == answer, (got, answer)


def test_issue_5901():
    f, g, h = map(Function, 'fgh')
    a = Symbol('a')
    D = Derivative(f(x), x)
    G = Derivative(g(a), a)

    # 解方程 f(x) + f'(x) = 0，预期结果为 [-D]
    assert solve(f(x) + f(x).diff(x), f(x)) == [-D]
    # 解方程 f(x) - 3 = 0，预期结果为 [3]
    assert solve(f(x) - 3, f(x)) == [3]
    # 解方程 f(x) - 3*f'(x) = 0，预期结果为 [3*D]
    assert solve(f(x) - 3*f(x).diff(x), f(x)) == [3*D]
    # 解方程组 [f(x) - 3*f'(x)]，预期结果为 {f(x): 3*D}
    assert solve([f(x) - 3*f(x).diff(x)], f(x)) == {f(x): 3*D}
    # 解方程组 [f(x) - 3*f'(x), f(x)^2 - y + 4]，预期结果为 [(3*D, 9*D**2 + 4)]
    assert solve([f(x) - 3*f(x).diff(x), f(x)**2 - y + 4], f(x), y) == [(3*D, 9*D**2 + 4)]
    # 解方程组 [-f(a)^2*g(a)^2 + f(a)^2*h(a)^2 + g(a).diff(a), h(a), g(a)]，预期结果为 ([h(a), g(a)], {...})
    assert solve(-f(a)**2*g(a)**2 + f(a)**2*h(a)**2 + g(a).diff(a), h(a), g(a), set=True) == \
        ([h(a), g(a)], {
        (-sqrt(f(a)**2*g(a)**2 - G)/f(a), g(a)),
        (sqrt(f(a)**2*g(a)**2 - G)/f(a), g(a))})
    # 解方程组 [f(x).diff(x, 2)*(f(x) + g(x)), 2 - g(x)^2]，预期结果为 {(-sqrt(2), sqrt(2)), (sqrt(2), -sqrt(2))}
    args = [[f(x).diff(x, 2)*(f(x) + g(x)), 2 - g(x)**2], f(x), g(x)]
    assert solve(*args, set=True)[1] == {(-sqrt(2), sqrt(2)), (sqrt(2), -sqrt(2))}
    # 解方程组 [f(x)^2 + g(x) - 2*f(x).diff(x), g(x)^2 - 4]，预期结果为 ([f(x), g(x)], {...})
    eqs = [f(x)**2 + g(x) - 2*f(x).diff(x), g(x)**2 - 4]
    assert solve(eqs, f(x), g(x), set=True) == \
        ([f(x), g(x)], {
        (-sqrt(2*D - 2), S(2)),
        (sqrt(2*D - 2), S(2)),
        (-sqrt(2*D + 2), -S(2)),
        (sqrt(2*D + 2), -S(2))})
    # 调用 `raises` 函数，验证解析 `solve_linear(f(x) + f(x).diff(x), symbols=[f(x)])` 是否引发 ValueError 异常
    raises(
        ValueError, lambda: solve_linear(f(x) + f(x).diff(x), symbols=[f(x)]))
    
    # 断言 `solve_linear(f(x) + f(x).diff(x), symbols=[x])` 的返回结果是否为 `(f(x) + Derivative(f(x), x), 1)`
    assert solve_linear(f(x) + f(x).diff(x), symbols=[x]) == \
        (f(x) + Derivative(f(x), x), 1)
    
    # 断言 `solve_linear(f(x) + Integral(x, (x, y)), symbols=[x])` 的返回结果是否为 `(f(x) + Integral(x, (x, y)), 1)`
    assert solve_linear(f(x) + Integral(x, (x, y)), symbols=[x]) == \
        (f(x) + Integral(x, (x, y)), 1)
    
    # 断言 `solve_linear(f(x) + Integral(x, (x, y)) + x, symbols=[x])` 的返回结果是否为 `(x + f(x) + Integral(x, (x, y)), 1)`
    assert solve_linear(f(x) + Integral(x, (x, y)) + x, symbols=[x]) == \
        (x + f(x) + Integral(x, (x, y)), 1)
    
    # 断言 `solve_linear(f(y) + Integral(x, (x, y)) + x, symbols=[x])` 的返回结果是否为 `(x, -f(y) - Integral(x, (x, y)))`
    assert solve_linear(f(y) + Integral(x, (x, y)) + x, symbols=[x]) == \
        (x, -f(y) - Integral(x, (x, y)))
    
    # 断言 `solve_linear(x - f(x)/a + (f(x) - 1)/a, symbols=[x])` 的返回结果是否为 `(x, 1/a)`
    assert solve_linear(x - f(x)/a + (f(x) - 1)/a, symbols=[x]) == \
        (x, 1/a)
    
    # 断言 `solve_linear(x + Derivative(2*x, x))` 的返回结果是否为 `(x, -2)`
    assert solve_linear(x + Derivative(2*x, x)) == \
        (x, -2)
    
    # 断言 `solve_linear(x + Integral(x, y), symbols=[x])` 的返回结果是否为 `(x, 0)`
    assert solve_linear(x + Integral(x, y), symbols=[x]) == \
        (x, 0)
    
    # 断言 `solve_linear(x + Integral(x, y) - 2, symbols=[x])` 的返回结果是否为 `(x, 2/(y + 1))`
    assert solve_linear(x + Integral(x, y) - 2, symbols=[x]) == \
        (x, 2/(y + 1))
    
    # 断言 `set(solve(x + exp(x)**2, exp(x)))` 的返回结果是否为 `{-sqrt(-x), sqrt(-x)}`
    assert set(solve(x + exp(x)**2, exp(x))) == \
        {-sqrt(-x), sqrt(-x)}
    
    # 断言 `solve(x + exp(x), x, implicit=True)` 的返回结果是否为 `[-exp(x)]`
    assert solve(x + exp(x), x, implicit=True) == \
        [-exp(x)]
    
    # 断言 `solve(cos(x) - sin(x), x, implicit=True)` 的返回结果是否为空列表 `[]`
    assert solve(cos(x) - sin(x), x, implicit=True) == []
    
    # 断言 `solve(x - sin(x), x, implicit=True)` 的返回结果是否为 `[sin(x)]`
    assert solve(x - sin(x), x, implicit=True) == \
        [sin(x)]
    
    # 断言 `solve(x**2 + x - 3, x, implicit=True)` 的返回结果是否为 `[-x**2 + 3]`
    assert solve(x**2 + x - 3, x, implicit=True) == \
        [-x**2 + 3]
    
    # 断言 `solve(x**2 + x - 3, x**2, implicit=True)` 的返回结果是否为 `[-x + 3]`
    assert solve(x**2 + x - 3, x**2, implicit=True) == \
        [-x + 3]
# 定义一个函数用于测试解决 x^2 - x - 0.1 的方程的解
def test_issue_5912():
    # 断言使用 rational=True 参数解决方程时的结果是指定的集合
    assert set(solve(x**2 - x - 0.1, rational=True)) == \
        {S.Half + sqrt(35)/10, -sqrt(35)/10 + S.Half}
    # 使用 rational=False 参数解决方程，确保结果有两个数值型解
    ans = solve(x**2 - x - 0.1, rational=False)
    assert len(ans) == 2 and all(a.is_Number for a in ans)
    # 使用默认参数解决方程，同样确保结果有两个数值型解
    ans = solve(x**2 - x - 0.1)
    assert len(ans) == 2 and all(a.is_Number for a in ans)


# 定义一个测试函数，用于检查浮点数处理
def test_float_handling():
    # 定义一个内部函数用于比较两个表达式中浮点数的数量
    def test(e1, e2):
        return len(e1.atoms(Float)) == len(e2.atoms(Float))
    # 断言使用 rational=True 参数解决 x - 0.5 的方程时，得到的结果是有理数
    assert solve(x - 0.5, rational=True)[0].is_Rational
    # 断言使用 rational=False 参数解决 x - 0.5 的方程时，得到的结果是浮点数
    assert solve(x - 0.5, rational=False)[0].is_Float
    # 断言使用 rational=False 参数解决 x - S.Half 的方程时，得到的结果是有理数
    assert solve(x - S.Half, rational=False)[0].is_Rational
    # 断言使用 rational=None 参数解决 x - 0.5 的方程时，得到的结果是浮点数
    assert solve(x - 0.5, rational=None)[0].is_Float
    # 断言使用 rational=None 参数解决 x - S.Half 的方程时，得到的结果是有理数
    assert solve(x - S.Half, rational=None)[0].is_Rational
    # 断言处理包含浮点数的表达式 nfloat(1 + 2*x) 的结果，确保浮点数的正确性
    assert test(nfloat(1 + 2*x), 1.0 + 2.0*x)
    # 对于不同的容器类型，检查 nfloat(contain([1 + 2*x])) 的结果是否正确
    for contain in [list, tuple, set]:
        ans = nfloat(contain([1 + 2*x]))
        assert type(ans) is contain and test(list(ans)[0], 1.0 + 2.0*x)
    # 检查 nfloat({2*x: [1 + 2*x]}) 的结果，确保浮点数的正确性
    k, v = list(nfloat({2*x: [1 + 2*x]}).items())[0]
    assert test(k, 2*x) and test(v[0], 1.0 + 2.0*x)
    # 检查 nfloat(cos(2*x)) 的结果，确保浮点数的正确性
    assert test(nfloat(cos(2*x)), cos(2.0*x))
    # 检查 nfloat(3*x**2) 的结果，确保浮点数的正确性
    assert test(nfloat(3*x**2), 3.0*x**2)
    # 检查 nfloat(3*x**2, exponent=True) 的结果，确保浮点数的正确性
    assert test(nfloat(3*x**2, exponent=True), 3.0*x**2.0)
    # 检查 nfloat(exp(2*x)) 的结果，确保浮点数的正确性
    assert test(nfloat(exp(2*x)), exp(2.0*x))
    # 检查 nfloat(x/3) 的结果，确保浮点数的正确性
    assert test(nfloat(x/3), x/3.0)
    # 检查 nfloat(x**4 + 2*x + cos(Rational(1, 3)) + 1) 的结果，确保浮点数的正确性
    assert test(nfloat(x**4 + 2*x + cos(Rational(1, 3)) + 1),
            x**4 + 2.0*x + 1.94495694631474)
    # 断言在无解的情况下不调用 nfloat
    tot = 100 + c + z + t
    assert solve(((.7 + c)/tot - .6, (.2 + z)/tot - .3, t/tot - .1)) == []


# 定义一个函数测试符号 x 为正数的情况下解方程 x**2 - 1
def test_check_assumptions():
    x = symbols('x', positive=True)
    assert solve(x**2 - 1) == [1]


# 定义一个函数用于测试解决 tanh 方程的相关问题
def test_issue_6056():
    # 断言解 tanh(x + 3)*tanh(x - 3) - 1 的结果为空列表
    assert solve(tanh(x + 3)*tanh(x - 3) - 1) == []
    # 断言解 tanh(x - 1)*tanh(x + 1) + 1 的结果符合预期
    assert solve(tanh(x - 1)*tanh(x + 1) + 1) == \
            [I*pi*Rational(-3, 4), -I*pi/4, I*pi/4, I*pi*Rational(3, 4)]
    # 断言解 (tanh(x + 3)*tanh(x - 3) + 1)**2 的结果符合预期
    assert solve((tanh(x + 3)*tanh(x - 3) + 1)**2) == \
            [I*pi*Rational(-3, 4), -I*pi/4, I*pi/4, I*pi*Rational(3, 4)]


# 定义一个函数用于测试解决复杂方程的问题
def test_issue_5673():
    # 定义一个复杂方程
    eq = -x + exp(exp(LambertW(log(x)))*LambertW(log(x)))
    # 断言对于方程 eq 在 x = 2 时解的验证为真
    assert checksol(eq, x, 2) is True
    # 断言对于方程 eq 在 x = 2 时的解没有数值解
    assert checksol(eq, x, 2, numerical=False) is None


# 定义一个函数测试解决包含符号和特定排除条件的方程组的问题
def test_exclude():
    # 定义符号变量
    R, C, Ri, Vout, V1, Vminus, Vplus, s = \
        symbols('R, C, Ri, Vout, V1, Vminus, Vplus, s')
    Rf = symbols('Rf', positive=True)  # 用于排除 Rf = 0 的解
    # 定义方程组
    eqs = [C*V1*s + Vplus*(-2*C*s - 1/R),
           Vminus*(-1/Ri - 1/Rf) + Vout/Rf,
           C*Vplus*s + V1*(-C*s - 1/R) + Vout/R,
           -Vminus + Vplus]
    # 断言解方程组 eqs，并排除 s*C*R 的情况
    assert solve(eqs, exclude=s*C*R) == [
        {
            Rf: Ri*(C*R*s + 1)**2/(C*R*s),
            Vminus: Vplus,
            V1: 2*Vplus + Vplus/(C*R*s),
            Vout: C*R*Vplus*s + 3*Vplus + Vplus/(C*R*s)},
        {
            Vplus: 0,
            Vminus: 0,
            V1: 0,
            Vout: 0},
    ]

    # TODO: 调查为何当前情况下解 [0] 优先于解 [1]。
    # 使用断言验证 solve 函数的返回结果是否在以下两个列表中的其中一个：
    # 第一个列表包含两个字典，每个字典表示一个解决方案：
    #   - 第一个字典包含键值对: Vminus: Vplus, V1: ..., R: ..., Rf: ...
    #   - 第二个字典包含键值对: Vminus: Vplus, V1: ..., R: ..., Rf: ...
    # 第二个列表包含一个字典，表示另一个解决方案：
    #   - 字典包含键值对: Vminus: Vplus, Vout: ..., Rf: ..., R: ...
    assert solve(eqs, exclude=[Vplus, s, C]) in [[{
        Vminus: Vplus,
        V1: Vout/2 + Vplus/2 + sqrt((Vout - 5*Vplus)*(Vout - Vplus))/2,
        R: (Vout - 3*Vplus - sqrt(Vout**2 - 6*Vout*Vplus + 5*Vplus**2))/(2*C*Vplus*s),
        Rf: Ri*(Vout - Vplus)/Vplus,
    }, {
        Vminus: Vplus,
        V1: Vout/2 + Vplus/2 - sqrt((Vout - 5*Vplus)*(Vout - Vplus))/2,
        R: (Vout - 3*Vplus + sqrt(Vout**2 - 6*Vout*Vplus + 5*Vplus**2))/(2*C*Vplus*s),
        Rf: Ri*(Vout - Vplus)/Vplus,
    }], [{
        Vminus: Vplus,
        Vout: (V1**2 - V1*Vplus - Vplus**2)/(V1 - 2*Vplus),
        Rf: Ri*(V1 - Vplus)**2/(Vplus*(V1 - 2*Vplus)),
        R: Vplus/(C*s*(V1 - 2*Vplus)),
    }]]
# 定义一个函数，用于测试高阶根的解法
def test_high_order_roots():
    # 构造一个高阶方程
    s = x**5 + 4*x**3 + 3*x**2 + Rational(7, 4)
    # 断言求解该方程得到的解集合与将其转为整数系数的多项式的所有根集合相等
    assert set(solve(s)) == set(Poly(s*4, domain='ZZ').all_roots())


# 定义一个函数，用于测试线性方程组的解法
def test_minsolve_linear_system():
    # 定义两种解法策略
    pqt = {"quick": True, "particular": True}
    pqf = {"quick": False, "particular": True}
    # 断言使用不同解法策略解同一个线性方程组能得到相同的解
    assert solve([x + y - 5, 2*x - y - 1], **pqt) == {x: 2, y: 3}
    assert solve([x + y - 5, 2*x - y - 1], **pqf) == {x: 2, y: 3}
    
    # 定义一个局部函数，用于计算字典中值为0的数量
    def count(dic):
        return len([x for x in dic.values() if x == 0])
    
    # 断言使用不同解法策略解不同的线性方程组时，返回的解中值为0的数量
    assert count(solve([x + y + z, y + z + a + t], **pqt)) == 3
    assert count(solve([x + y + z, y + z + a + t], **pqf)) == 3
    assert count(solve([x + y + z, y + z + a], **pqt)) == 1
    assert count(solve([x + y + z, y + z + a], **pqf)) == 2
    
    # 以下测试解一个特定的矩阵方程组是否为空解集合
    A = Matrix([
        [ 1,  1,  1,  0,  1,  1,  0,  1,  0,  0,  1,  1,  1,  0],
        [ 1,  1,  0,  1,  1,  0,  1,  0,  1,  0, -1, -1,  0,  0],
        [-1, -1,  0,  0, -1,  0,  0,  0,  0,  0,  1,  1,  0,  1],
        [ 1,  0,  1,  1,  0,  1,  1,  0,  0,  1, -1,  0, -1,  0],
        [-1,  0, -1,  0,  0, -1,  0,  0,  0,  0,  1,  0,  1,  1],
        [-1,  0,  0, -1,  0,  0, -1,  0,  0,  0, -1,  0,  0, -1],
        [ 0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0, -1, -1,  0],
        [ 0, -1, -1,  0,  0,  0,  0, -1,  0,  0,  0,  1,  1,  1],
        [ 0, -1,  0, -1,  0,  0,  0,  0, -1,  0,  0, -1,  0, -1],
        [ 0,  0, -1, -1,  0,  0,  ```python
# 0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0]])
    v = Matrix(symbols("v:14", integer=True))
    B = Matrix([[2], [-2], [0], [0], [0], [0], [0], [0], [0],
        [0], [0], [0]])
    eqs = A@v-B
    # 断言解该矩阵方程组得到的解是空列表
    assert solve(eqs) == []
    # 断言假设存在特解的情况下解该矩阵方程组的解同样是空列表（假设不成立）
    assert solve(eqs, particular=True) == []  # assumption violated
    # 断言解一个带特定约束的线性方程组中所有变量是否有非零解
    assert all(v for v in solve([x + y + z, y + z + a]).values())
    for _q in (True, False):
        # 断言使用不同的解法策略解具有特定约束的线性方程组是否存在非零解
        assert not all(v for v in solve(
            [x + y + z, y + z + a], quick=_q,
            particular=True).values())
        # 如果使用快速求解但未指定特定解，则应该引发 ValueError 错误
        raises(ValueError, lambda: solve([x + 1], quick=_q))
        raises(ValueError, lambda: solve([x + 1], quick=_q, particular=False))
    # 如果尝试在单个方程中使用特定解会引发 ValueError 错误，需要提供清晰的错误信息
    raises(ValueError, lambda: solve(x + 1, particular=True))


# 定义一个函数，用于测试实数根的求解
def test_real_roots():
    # 定义一个实数变量 x
    x = Symbol('x', real=True)
    # 断言求解多项式方程 x**5 + x**3 + 1 的根的个数为 1
    assert len(solve(x**5 + x**3 + 1)) == 1


# 定义一个函数，用于测试解决特定问题的方程组
def test_issue_6528():
    eqs = [
        327600995*x**2 - 37869137*x + 1809975124*y**2 - 9998905626,
        895613949*x**2 - 273830224*x*y + 530506983*y**2 - 10000000000]
    # 由于两个表达式的操作数超过 1400 个，如果解算过程出现问题，可能是因为过度简化导致
    assert len(solve(eqs, y, x, check=False)) == 4


# 定义一个函数，用于测试过度确定的方程组
def test_overdetermined():
    # 定义一个实数变量 x
    x = symbols('x', real=True)
    # 定义一个包含绝对值的方程组
    eqs = [Abs(4*x - 7) - 5, Abs(3 - 8*x) - 1]
    # 断言：解决方程组 eqs，期望返回 [(S.Half,)]
    assert solve(eqs, x) == [(S.Half,)]
    
    # 断言：解决方程组 eqs，期望返回 [(S.Half,)]，使用手动模式求解
    assert solve(eqs, x, manual=True) == [(S.Half,)]
    
    # 断言：解决方程组 eqs，期望返回 [(S.Half,), (S(3),)]，使用手动模式求解，不进行结果验证
    assert solve(eqs, x, manual=True, check=False) == [(S.Half,), (S(3),)]
def test_issue_6605():
    # 定义符号变量 x
    x = symbols('x')
    # 解方程 4^(x/2) - 2^(x/3)，预期解为 [0, 3*I*pi/log(2)]
    assert solve(4**(x/2) - 2**(x/3)) == [0, 3*I*pi/log(2)]
    
    # 第一个测试通过，但是第二个测试失败
    # 重新定义符号变量 x，并指定其为实数
    x = symbols('x', real=True)
    # 解方程 5^(x/2) - 2^(x/3)，预期解为 [0]
    assert solve(5**(x/2) - 2**(x/3)) == [0]
    
    # 计算常量 b，其值为 sqrt(6)*sqrt(log(2))/sqrt(log(5))
    b = sqrt(6)*sqrt(log(2))/sqrt(log(5))
    # 解方程 5^(x/2) - 2^(3/x)，预期解为 [-b, b]
    assert solve(5**(x/2) - 2**(3/x)) == [-b, b]


def test__ispow():
    # 断言 _ispow(x**2) 返回 True
    assert _ispow(x**2)
    # 断言 _ispow(x) 返回 False
    assert not _ispow(x)
    # 断言 _ispow(True) 返回 False
    assert not _ispow(True)


def test_issue_6644():
    # 定义复杂的方程 eq
    eq = -sqrt((m - q)**2 + (-m/(2*q) + S.Half)**2) + sqrt((-m**2/2 - sqrt(
        4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2 + (m**2/2 - m - sqrt(
        4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2)
    # 解方程 eq 关于 q，预期解的长度为 5
    sol = solve(eq, q, simplify=False, check=False)
    assert len(sol) == 5


def test_issue_6752():
    # 解方程组 [a**2 + a, a - b] 关于 [a, b]，预期解为 [(-1, -1), (0, 0)]
    assert solve([a**2 + a, a - b], [a, b]) == [(-1, -1), (0, 0)]
    # 解方程组 [a**2 + a*c, a - b] 关于 [a, b]，预期解为 [(0, 0), (-c, -c)]
    assert solve([a**2 + a*c, a - b], [a, b]) == [(0, 0), (-c, -c)]


def test_issue_6792():
    # 解方程 x*(x - 1)**2*(x + 1)*(x**6 - x + 1)，预期解为所有根的列表
    assert solve(x*(x - 1)**2*(x + 1)*(x**6 - x + 1)) == [
        -1, 0, 1, CRootOf(x**6 - x + 1, 0), CRootOf(x**6 - x + 1, 1),
         CRootOf(x**6 - x + 1, 2), CRootOf(x**6 - x + 1, 3),
         CRootOf(x**6 - x + 1, 4), CRootOf(x**6 - x + 1, 5)]


def test_issues_6819_6820_6821_6248_8692_25777_25779():
    # issue 6821
    # 解绝对值方程 abs(x + 3) - 2*abs(x - 3)，预期解为 [1, 9]
    assert solve(abs(x + 3) - 2*abs(x - 3)) == [1, 9]
    # 解方程组 [abs(x) - 2, arg(x) - pi] 关于 x，预期解为 [(-2,)]
    assert solve([abs(x) - 2, arg(x) - pi], x) == [(-2,)]
    # 解绝对值方程 abs(x - 7) - 8，预期解为 {-1, 15}
    assert set(solve(abs(x - 7) - 8)) == {-S.One, S(15)}

    # issue 8692
    # 解方程 Abs(x + 1) + Abs(x**2 - 7) - 9，预期解为 [(-1/2 + sqrt(61)/2), (-sqrt(69)/2 + 1/2)]
    assert solve(Eq(Abs(x + 1) + Abs(x**2 - 7), 9), x) == [
        Rational(-1, 2) + sqrt(61)/2, -sqrt(69)/2 + S.Half]

    # issue 7145
    # 解绝对值方程 2*abs(x) - abs(x - 1)，预期解为 [-1, 1/3]
    assert solve(2*abs(x) - abs(x - 1)) == [-1, Rational(1, 3)]

    # 25777
    # 解绝对值方程 abs(x**3 + x + 2)/(x + 1)，预期解为 []
    assert solve(abs(x**3 + x + 2)/(x + 1)) == []

    # 25779
    # 解绝对值方程 abs(x)，预期解为 [0]
    assert solve(abs(x)) == [0]
    # 解方程 Abs(x**2 - 2*x) - 4，预期解为 [1 - sqrt(5), 1 + sqrt(5)]
    assert solve(Eq(abs(x**2 - 2*x), 4), x) == [
        1 - sqrt(5), 1 + sqrt(5)]
    # 解绝对值方程 abs(sqrt(nn))，预期解为 [0]
    nn = symbols('nn', nonnegative=True)
    assert solve(abs(sqrt(nn))) == [0]
    # 解绝对值方程 Abs(4 + 1 / (4*nz))，预期解为 [-1/16]
    nz = symbols('nz', nonzero=True)
    assert solve(Eq(Abs(4 + 1 / (4*nz)), 0)) == [-Rational(1, 16)]

    x = symbols('x')
    # 解方程组 [re(x) - 1, im(x) - 2] 关于 x，预期解为 [{x: 1 + 2*I, re(x): 1, im(x): 2}]
    assert solve([re(x) - 1, im(x) - 2], x) == [
        {x: 1 + 2*I, re(x): 1, im(x): 2}]

    # 检查对解的 'dict' 处理
    # 定义复杂方程 eq
    eq = sqrt(re(x)**2 + im(x)**2) - 3
    # 解方程 eq 关于 x，预期解与 solve(eq) 相同
    assert solve(eq) == solve(eq, x)

    # 处理虚数 i
    i = symbols('i', imaginary=True)
    # 解绝对值方程 abs(i) - 3，预期解为 [-3*I, 3*I]
    assert solve(abs(i) - 3) == [-3*I, 3*I]
    # 断言解决方程 abs(x) - 3 会引发 NotImplementedError
    raises(NotImplementedError, lambda: solve(abs(x) - 3))

    # 处理整数 w
    w = symbols('w', integer=True)
    # 解方程 2*x**w - 4*y**w 关于 w，预期解为 (x/y)**w - 2
    assert solve(2*x**w - 4*y**w, w) == solve((x/y)**w - 2, w)

    x, y = symbols('x y', real=True)
    # 解方程 x + y*I + 3 关于 x 和 y，预期解为 {y: 0, x: -3}
    assert solve(x + y*I + 3) == {y: 0, x: -3}

    # issue 2642
    # 解方程 x*(1 + I)，预期解为 [0]
    assert solve(x*(1 + I)) == [0]

    x, y = symbols('x y', imaginary=True)
    # 解方程 x + y*I + 3 + 2*I，预期解为 {x: -2*I, y: 3*I}
    # 断言语句，用于在程序中检查条件是否满足，若条件为 False，则抛出 AssertionError 异常。
    # 此处断言的条件为 solve(2**x + 4**x) 等于 [I*pi/log(2)]，即求解 2**x + 4**x 的结果是否为 [I*pi/log(2)]。
    assert solve(2**x + 4**x) == [I*pi/log(2)]
def test_issue_17638():
    # 检验解决方案是否符合条件
    assert solve(((2-exp(2*x))*exp(x))/(exp(2*x)+2)**2 > 0, x) == (-oo < x) & (x < log(2)/2)
    assert solve(((2-exp(2*x)+2)*exp(x+2))/(exp(x)+2)**2 > 0, x) == (-oo < x) & (x < log(4)/2)
    assert solve((exp(x)+2+x**2)*exp(2*x+2)/(exp(x)+2)**2 > 0, x) == (-oo < x) & (x < oo)


def test_issue_14607():
    # issue 14607
    # 定义符号变量
    s, tau_c, tau_1, tau_2, phi, K = symbols('s, tau_c, tau_1, tau_2, phi, K')

    # 定义目标表达式
    target = (s**2*tau_1*tau_2 + s*tau_1 + s*tau_2 + 1)/(K*s*(-phi + tau_c))

    # 定义PID控制器表达式
    K_C, tau_I, tau_D = symbols('K_C, tau_I, tau_D', positive=True, nonzero=True)
    PID = K_C*(1 + 1/(tau_I*s) + tau_D*s)

    # 合并目标和PID表达式并化简
    eq = (target - PID).together()
    eq *= denom(eq).simplify()

    # 将表达式转化为多项式
    eq = Poly(eq, s)
    c = eq.coeffs()

    # 解多项式方程得到符号变量的值
    vars = [K_C, tau_I, tau_D]
    s = solve(c, vars, dict=True)

    # 断言解的唯一性
    assert len(s) == 1

    # 已知的解决方案
    knownsolution = {K_C: -(tau_1 + tau_2)/(K*(phi - tau_c)),
                     tau_I: tau_1 + tau_2,
                     tau_D: tau_1*tau_2/(tau_1 + tau_2)}

    # 检验计算得到的解是否与已知解一致
    for var in vars:
        assert s[0][var].simplify() == knownsolution[var].simplify()


def test_lambert_multivariate():
    # 导入必要的符号变量
    from sympy.abc import x, y

    # 断言过滤后的生成器集合
    assert _filtered_gens(Poly(x + 1/x + exp(x) + y), x) == {x, exp(x)}

    # 断言 Lambet 函数的应用
    assert _lambert(x, x) == []

    # 解方程替换后的结果
    assert solve((x**2 - 2*x + 1).subs(x, log(x) + 3*x)) == [LambertW(3*S.Exp1)/3]

    # 解方程替换后的结果
    assert solve((x**2 - 2*x + 1).subs(x, (log(x) + 3*x)**2 - 1)) == \
          [LambertW(3*exp(-sqrt(2)))/3, LambertW(3*exp(sqrt(2)))/3]

    # 解方程替换后的结果
    assert solve((x**2 - 2*x - 2).subs(x, log(x) + 3*x)) == \
          [LambertW(3*exp(1 - sqrt(3)))/3, LambertW(3*exp(1 + sqrt(3)))/3]

    # 解方程替换后的结果
    eq = (x*exp(x) - 3).subs(x, x*exp(x))
    assert solve(eq) == [LambertW(3*exp(-LambertW(3)))]

    # 覆盖测试
    raises(NotImplementedError, lambda: solve(x - sin(x)*log(y - x), x))

    # 解方程替换后的结果
    ans = [3, -3*LambertW(-log(3)/3)/log(3)]  # 3 and 2.478...
    assert solve(x**3 - 3**x, x) == ans

    # 解方程替换后的结果
    assert set(solve(3*log(x) - x*log(3))) == set(ans)

    # 解方程替换后的结果
    assert solve(LambertW(2*x) - y, x) == [y*exp(y)/2]


@XFAIL
def test_other_lambert():
    # 断言另一种 Lambet 函数的应用
    assert solve(3*sin(x) - x*sin(3), x) == [3]

    # 断言解的集合
    assert set(solve(x**a - a**x), x) == {
        a, -a*LambertW(-log(a)/a)/log(a)}


@slow
def test_lambert_bivariate():
    # 测试通过当前实现
    # 解方程替换后的结果
    assert solve((x**2 + x)*exp(x**2 + x) - 1) == [
        Rational(-1, 2) + sqrt(1 + 4*LambertW(1))/2,
        Rational(-1, 2) - sqrt(1 + 4*LambertW(1))/2]

    # 解方程替换后的结果
    assert solve((x**2 + x)*exp((x**2 + x)*2) - 1) == [
        Rational(-1, 2) + sqrt(1 + 2*LambertW(2))/2,
        Rational(-1, 2) - sqrt(1 + 2*LambertW(2))/2]

    # 解方程替换后的结果
    assert solve(a/x + exp(x/2), x) == [2*LambertW(-a/2)]

    # 解方程替换后的结果
    assert solve((a/x + exp(x/2)).diff(x), x) == \
            [4*LambertW(-sqrt(2)*sqrt(a)/4), 4*LambertW(sqrt(2)*sqrt(a)/4)]
    # 确定解析表达式的符号
    assert solve((1/x + exp(x/2)).diff(x), x) == \
        [4*LambertW(-sqrt(2)/4),  # 解析表达式的第一个解
        4*LambertW(sqrt(2)/4),   # 解析表达式的第二个解，通过数学化简为公式
        4*LambertW(-sqrt(2)/4, -1)]  # 解析表达式的第三个解，通过指定分支索引

    # 求解给定的代数方程式
    assert solve(x*log(x) + 3*x + 1, x) == \
            [exp(-3 + LambertW(-exp(3)))]  # 解析表达式的解

    # 求解给定的代数方程式
    assert solve(-x**2 + 2**x, x) == [2, 4, -2*LambertW(log(2)/2)/log(2)]  # 解析表达式的解

    # 求解给定的代数方程式
    assert solve(x**2 - 2**x, x) == [2, 4, -2*LambertW(log(2)/2)/log(2)]  # 解析表达式的解

    # 求解给定的代数方程式
    ans = solve(3*x + 5 + 2**(-5*x + 3), x)
    assert len(ans) == 1 and ans[0].expand() == \
        Rational(-5, 3) + LambertW(-10240*root(2, 3)*log(2)/3)/(5*log(2))  # 解析表达式的解

    # 求解给定的代数方程式
    assert solve(5*x - 1 + 3*exp(2 - 7*x), x) == \
        [Rational(1, 5) + LambertW(-21*exp(Rational(3, 5))/5)/7]  # 解析表达式的解

    # 求解给定的代数方程式，代入变量后求解
    assert solve((log(x) + x).subs(x, x**2 + 1)) == [
        -I*sqrt(-LambertW(1) + 1), sqrt(-1 + LambertW(1))]  # 解析表达式的解

    # 检查收集到的解
    ax = a**(3*x + 5)
    ans = solve(3*log(ax) + b*log(ax) + ax, x)
    x0 = 1/log(a)
    x1 = sqrt(3)*I
    x2 = b + 3
    x3 = x2*LambertW(1/x2)/a**5
    x4 = x3**Rational(1, 3)/2
    assert ans == [
        x0*log(x4*(-x1 - 1)),  # 解析表达式的第一个解
        x0*log(x4*(x1 - 1)),   # 解析表达式的第二个解
        x0*log(x3)/3]          # 解析表达式的第三个解

    # 求解给定的代数方程式
    x1 = LambertW(Rational(1, 3))
    x2 = a**(-5)
    x3 = -3**Rational(1, 3)
    x4 = 3**Rational(5, 6)*I
    x5 = x1**Rational(1, 3)*x2**Rational(1, 3)/2
    ans = solve(3*log(ax) + ax, x)
    assert ans == [
        x0*log(3*x1*x2)/3,       # 解析表达式的第一个解
        x0*log(x5*(x3 - x4)),    # 解析表达式的第二个解
        x0*log(x5*(x3 + x4))]    # 解析表达式的第三个解

    # 求解给定的代数方程式
    p = symbols('p', positive=True)
    eq = 4*2**(2*p + 3) - 2*p - 3
    assert _solve_lambert(eq, p, _filtered_gens(Poly(eq), p)) == [
        Rational(-3, 2) - LambertW(-4*log(2))/(2*log(2))]  # 解析表达式的解

    # 求解给定的代数方程式，集合形式
    assert set(solve(3**cos(x) - cos(x)**3)) == {
        acos(3), acos(-3*LambertW(-log(3)/3)/log(3))}  # 解析表达式的解

    # 求解给定的代数方程式
    assert solve(2*log(x) - 2*log(z) + log(z + log(x) + log(z)), x) == [
        exp(-z + LambertW(2*z**4*exp(2*z))/2)/z]  # 解析表达式的解

    # 求解给定的代数方程式，考虑 p != S.One 的情况
    # issue 4271
    ans = solve((a/x + exp(x/2)).diff(x, 2), x)
    x0 = (-a)**Rational(1, 3)
    x1 = sqrt(3)*I
    x2 = x0/6
    assert ans == [
        6*LambertW(x0/3),         # 解析表达式的第一个解
        6*LambertW(x2*(-x1 - 1)), # 解析表达式的第二个解
        6*LambertW(x2*(x1 - 1))]  # 解析表达式的第三个解

    # 求解给定的代数方程式
    assert solve((1/x + exp(x/2)).diff(x, 2), x) == \
                [6*LambertW(Rational(-1, 3)), 6*LambertW(Rational(1, 6) - sqrt(3)*I/6), \
                6*LambertW(Rational(1, 6) + sqrt(3)*I/6), 6*LambertW(Rational(-1, 3), -1)]  # 解析表达式的解

    # 求解给定的代数方程式，同时解 x 和 y
    assert solve(x**2 - y**2/exp(x), x, y, dict=True) == \
                [{x: 2*LambertW(-y/2)}, {x: 2*LambertW(y/2)}]  # 解析表达式的解

    # 求解给定的代数方程式
    assert solve((x**3)**(x/2) + pi/2, x) == [
        exp(LambertW(-2*log(2)/3 + 2*log(pi)/3 + I*pi*Rational(2, 3)))]  # 解析表达式的解

    # issue 23253
    # 求解给定的代数方程式
    assert solve((1/log(sqrt(x) + 2)**2 - 1/x)) == [
        (LambertW(-exp(-2), -1) + 2)**2]  # 解析表达式的解

    # issue 23253
    # 求解给定的代数方程式
    assert solve((1/log(1/sqrt(x) + 2)**2 - x)) == [
        (LambertW(-exp(-2), -1) + 2)**-2]  # 解析表达式的解
    # 断言语句，用于验证 solve 函数对给定表达式的输出是否符合预期结果
    assert solve((1/log(x**2 + 2)**2 - x**-4)) == [
        # 复数解：-I*sqrt(2 - LambertW(exp(2)))
        -I*sqrt(2 - LambertW(exp(2))),
        # 复数解：-I*sqrt(LambertW(-exp(-2)) + 2)
        -I*sqrt(LambertW(-exp(-2)) + 2),
        # 实数解：sqrt(-2 - LambertW(-exp(-2)))
        sqrt(-2 - LambertW(-exp(-2))),
        # 实数解：sqrt(-2 + LambertW(exp(2)))
        sqrt(-2 + LambertW(exp(2))),
        # 复数解：-sqrt(-2 - LambertW(-exp(-2), -1))
        -sqrt(-2 - LambertW(-exp(-2), -1)),
        # 复数解：sqrt(-2 - LambertW(-exp(-2), -1))
        sqrt(-2 - LambertW(-exp(-2), -1))
    ]
def test_rewrite_trig():
    # 测试解决 sin(x) + tan(x) 的方程，预期结果为 [0, -pi, pi, 2*pi]
    assert solve(sin(x) + tan(x)) == [0, -pi, pi, 2*pi]
    # 测试解决 sin(x) + sec(x) 的方程，预期结果为一个包含四个复杂表达式的列表
    assert solve(sin(x) + sec(x)) == [
        -2*atan(Rational(-1, 2) + sqrt(2)*sqrt(1 - sqrt(3)*I)/2 + sqrt(3)*I/2),
        2*atan(S.Half - sqrt(2)*sqrt(1 + sqrt(3)*I)/2 + sqrt(3)*I/2), 2*atan(S.Half
        + sqrt(2)*sqrt(1 + sqrt(3)*I)/2 + sqrt(3)*I/2), 2*atan(S.Half -
        sqrt(3)*I/2 + sqrt(2)*sqrt(1 - sqrt(3)*I)/2)]
    # 测试解决 sinh(x) + tanh(x) 的方程，预期结果为 [0, I*pi]
    assert solve(sinh(x) + tanh(x)) == [0, I*pi]

    # issue 6157
    # 测试解决 2*sin(x) - cos(x) = 0 的方程，预期结果为 [atan(S.Half)]
    assert solve(2*sin(x) - cos(x), x) == [atan(S.Half)]


@XFAIL
def test_rewrite_trigh():
    # 如果导入成功，则测试解决 sinh(x) + sech(x) 的方程，预期结果为一个包含四个复杂表达式的列表
    from sympy.functions.elementary.hyperbolic import sech
    assert solve(sinh(x) + sech(x)) == [
        2*atanh(Rational(-1, 2) + sqrt(5)/2 - sqrt(-2*sqrt(5) + 2)/2),
        2*atanh(Rational(-1, 2) + sqrt(5)/2 + sqrt(-2*sqrt(5) + 2)/2),
        2*atanh(-sqrt(5)/2 - S.Half + sqrt(2 + 2*sqrt(5))/2),
        2*atanh(-sqrt(2 + 2*sqrt(5))/2 - sqrt(5)/2 - S.Half)]


def test_uselogcombine():
    # 创建方程对象 eq
    eq = z - log(x) + log(y/(x*(-1 + y**2/x**2)))
    # 解决方程 eq，解 x 的值，强制模式为 True，预期结果为两个值的列表
    assert solve(eq, x, force=True) == [-sqrt(y*(y - exp(z))), sqrt(y*(y - exp(z)))]
    # 解决 log(x + 3) + log(1 + 3/x) - 3 = 0 的方程，预期结果为两个可能的解的列表
    assert solve(log(x + 3) + log(1 + 3/x) - 3) in [
        [-3 + sqrt(-12 + exp(3))*exp(Rational(3, 2))/2 + exp(3)/2,
        -sqrt(-12 + exp(3))*exp(Rational(3, 2))/2 - 3 + exp(3)/2],
        [-3 + sqrt(-36 + (-exp(3) + 6)**2)/2 + exp(3)/2,
        -3 - sqrt(-36 + (-exp(3) + 6)**2)/2 + exp(3)/2],
        ]
    # 解决 log(exp(2*x) + 1) + log(-tanh(x) + 1) - log(2) = 0 的方程，预期结果为空列表
    assert solve(log(exp(2*x) + 1) + log(-tanh(x) + 1) - log(2)) == []


def test_atan2():
    # 解决 atan2(x, 2) - pi/3 = 0 的方程，解 x 的值，预期结果为 [2*sqrt(3)]
    assert solve(atan2(x, 2) - pi/3, x) == [2*sqrt(3)]


def test_errorinverses():
    # 解决 erf(x) - y = 0 的方程，解 x 的值，预期结果为 [erfinv(y)]
    assert solve(erf(x) - y, x) == [erfinv(y)]
    # 解决 erfinv(x) - y = 0 的方程，解 x 的值，预期结果为 [erf(y)]
    assert solve(erfinv(x) - y, x) == [erf(y)]
    # 解决 erfc(x) - y = 0 的方程，解 x 的值，预期结果为 [erfcinv(y)]
    assert solve(erfc(x) - y, x) == [erfcinv(y)]
    # 解决 erfcinv(x) - y = 0 的方程，解 x 的值，预期结果为 [erfc(y)]
    assert solve(erfcinv(x) - y, x) == [erfc(y)]


def test_issue_2725():
    # 创建符号变量 R
    R = Symbol('R')
    # 创建方程对象 eq
    eq = sqrt(2)*R*sqrt(1/(R + 1)) + (R + 1)*(sqrt(2)*sqrt(1/(R + 1)) - 1)
    # 解决方程 eq，解 R 的值，预期结果为一个包含两个可能解的集合
    sol = solve(eq, R, set=True)[1]
    assert sol == {(Rational(5, 3) + (Rational(-1, 2) - sqrt(3)*I/2)*(Rational(251, 27) +
        sqrt(111)*I/9)**Rational(1, 3) + 40/(9*((Rational(-1, 2) - sqrt(3)*I/2)*(Rational(251, 27) +
        sqrt(111)*I/9)**Rational(1, 3))),), (Rational(5, 3) + 40/(9*(Rational(251, 27) +
        sqrt(111)*I/9)**Rational(1, 3)) + (Rational(251, 27) + sqrt(111)*I/9)**Rational(1, 3),)}


def test_issue_5114_6611():
    # 确保解决此系统不会导致程序挂起；此问题大约需要 2 秒钟解决。
    # 同时检查解的大小是否合理。
    # 注意：在 issue 6611 中的系统解决大约需要 5 秒钟，并且有 138336 个操作数（不使用简化选项）。
    b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r = symbols('b:r')
    # 创建方程组对象 eqs
    eqs = Matrix([
        [b - c/d + r/d], [c*(1/g + 1/e + 1/d) - f/g - r/d],
        [-c/g + f*(1/j + 1/i + 1/g) - h/i], [-f/i + h*(1/m + 1/l + 1/i) - k/m],
        [-h/m + k*(1/p + 1/o + 1/m) - n/p], [-k/p + n*(1/q + 1/p)]])
    # 创建符号向量 v
    v = Matrix([f, h, k, n, b, c])
    # 调用 solve 函数解决方程组，并将结果存储在 ans 中
    ans = solve(list(eqs), list(v), simplify=False)
    # 断言：检查所有解的操作数总和是否不超过 3270
    # 如果简化选项为 False，则下面的 2617 会变成 1168，并且运行时间约为 50 秒而不是 2 秒。
    assert sum(s.count_ops() for s in ans.values()) <= 3270
# 定义一个测试函数，用于测试 det_quick 函数的各种情况
def test_det_quick():
    # 创建一个 3x3 的符号矩阵 m
    m = Matrix(3, 3, symbols('a:9'))
    # 断言 m 的行列式等于 det_quick(m) 的返回值，调用 det_perm 函数
    assert m.det() == det_quick(m)  # calls det_perm
    # 修改矩阵 m 的元素 m[0, 0] 为 1
    m[0, 0] = 1
    # 再次断言 m 的行列式等于 det_quick(m) 的返回值，调用 det_minor 函数
    assert m.det() == det_quick(m)  # calls det_minor
    # 创建一个 3x3 的数值矩阵 m
    m = Matrix(3, 3, list(range(9)))
    # 断言 m 的行列式等于 det_quick(m) 的返回值，默认调用 .det() 函数
    assert m.det() == det_quick(m)  # defaults to .det()
    # 确保 SparseMatrix 也能正常使用
    s = SparseMatrix(2, 2, (1, 2, 1, 4))
    # 断言 det_perm(s) 等于 det_minor(s)，且它们都等于 s 的行列式
    assert det_perm(s) == det_minor(s) == s.det()


# 测试实部和虚部分离函数
def test_real_imag_splitting():
    # 定义符号变量 a, b，限定为实数
    a, b = symbols('a b', real=True)
    # 解方程 sqrt(a**2 + b**2) - 3 = 0，对变量 a 求解
    assert solve(sqrt(a**2 + b**2) - 3, a) == \
        [-sqrt(-b**2 + 9), sqrt(-b**2 + 9)]
    # 定义符号变量 a, b，限定为虚数
    a, b = symbols('a b', imaginary=True)
    # 解方程 sqrt(a**2 + b**2) - 3 = 0，对变量 a 求解，预期返回空列表
    assert solve(sqrt(a**2 + b**2) - 3, a) == []


# 测试 issue 7110
def test_issue_7110():
    # 定义一个关于 x 的多项式
    y = -2*x**3 + 4*x**2 - 2*x + 5
    # 对多项式的解进行询问，确保至少有一个解是实数
    assert any(ask(Q.real(i)) for i in solve(y))


# 测试单位转换
def test_units():
    # 解方程 1/x - 1/(2*cm) = 0，预期解为 [2*cm]
    assert solve(1/x - 1/(2*cm)) == [2*cm]


# 测试 issue 7547
def test_issue_7547():
    # 定义符号变量 A, B, V
    A, B, V = symbols('A,B,V')
    # 定义三个方程
    eq1 = Eq(630.26*(V - 39.0)*V*(V + 39) - A + B, 0)
    eq2 = Eq(B, 1.36*10**8*(V - 39))
    eq3 = Eq(A, 5.75*10**5*V*(V + 39.0))
    # 使用数值求解符号方程组的解，并将解转换为矩阵形式
    sol = Matrix(nsolve(Tuple(eq1, eq2, eq3), [A, B, V], (0, 0, 0)))
    # 断言数值解的字符串形式等于预期的矩阵形式
    assert str(sol) == str(Matrix(
        [['4442890172.68209'],
         ['4289299466.1432'],
         ['70.5389666628177']]))


# 测试 issue 7895
def test_issue_7895():
    # 定义实数符号变量 r
    r = symbols('r', real=True)
    # 解方程 sqrt(r) - 2 = 0，预期解为 [4]
    assert solve(sqrt(r) - 2) == [4]


# 测试 issue 2777
def test_issue_2777():
    # 定义实数符号变量 x, y
    x, y = symbols('x y', real=True)
    # 定义两个圆的方程 e1, e2
    e1, e2 = sqrt(x**2 + y**2) - 10, sqrt(y**2 + (-x + 10)**2) - 3
    # 定义预期的解答
    a, b = Rational(191, 20), 3*sqrt(391)/20
    ans = [(a, -b), (a, b)]
    # 解方程组 (e1, e2)，预期解为 ans
    assert solve((e1, e2), (x, y)) == ans
    # 解方程组 (e1, e2/(x - a))，预期解为空列表
    assert solve((e1, e2/(x - a)), (x, y)) == []
    # 修改第二个圆的半径为 -3
    e2 += 6
    # 解方程组 (e1, e2)，预期解为空列表
    assert solve((e1, e2), (x, y)) == []
    # 解方程组 (e1, e2)，关闭检查，预期解为 ans
    assert solve((e1, e2), (x, y), check=False) == ans


# 测试 issue 7322
def test_issue_7322():
    # 定义一个很小的数值
    number = 5.62527e-35
    # 解方程 x - number = 0，预期解为 [number]
    assert solve(x - number, x)[0] == number


# 测试数值求解函数 nsolve
def test_nsolve():
    # 检查当输入为 x 时引发 ValueError
    raises(ValueError, lambda: nsolve(x, (-1, 1), method='bisect'))
    # 检查当输入为 (x - y + 3,x + y,z - y) 时引发 TypeError
    raises(TypeError, lambda: nsolve((x - y + 3,x + y,z - y),(x,y,z),(-50,50)))
    # 检查当输入为 (x + y, x - y) 时引发 TypeError
    raises(TypeError, lambda: nsolve((x + y, x - y), (0, 1)))
    # 检查当输入为 x < 0.5 时引发 TypeError
    raises(TypeError, lambda: nsolve(x < 0.5, x, 1))


# 测试高阶多变量方程求解
@slow
def test_high_order_multivariate():
    # 解方程 a*x**3 - x + 1 = 0，预期有三个解
    assert len(solve(a*x**3 - x + 1, x)) == 3
    # 解方程 a*x**4 - x + 1 = 0，预期有四个解
    assert len(solve(a*x**4 - x + 1, x)) == 4
    # 解方程 a*x**5 - x + 1 = 0，预期无解，允许不完全解
    assert solve(a*x**5 - x + 1, x) == []  # incomplete solution allowed
    # 解方程 a*x**5 - x + 1 = 0，关闭不完全解，预期引发 NotImplementedError
    raises(NotImplementedError, lambda:
        solve(a*x**5 - x + 1, x, incomplete=False))

    # 检查结果时必须考虑分母和 CRootOf
    d = x**5 - x + 1
    assert solve(d*(1 + 1/d)) == [CRootOf(d + 1, i) for i in range(5)]
    d = x - 1
    assert solve(d*(2 + 1/d)) == [S.Half]


# 测试基本的 0 次幂和 0 次指数
def test_base_0_exp_0():
    # 解方程 0**x - 1 = 0，预期解为 [0]
    assert solve(0**x - 1) == [0]
    # 解方程 0**(x - 2
    # 断言：验证 _simple_dens 函数对 1/x**y 表达式的处理是否正确，期望返回一个集合，包含 x**y 这个表达式
    assert _simple_dens(1/x**y, [x]) == {x**y}
    
    # 断言：验证 _simple_dens 函数对 1/root(x, 3) 表达式的处理是否正确，期望返回一个集合，包含 x 这个表达式
    assert _simple_dens(1/root(x, 3), [x]) == {x}
def test_issue_8755():
    # This tests two things: that if full unrad is attempted and fails
    # the solution should still be found; also it tests the use of
    # keyword `composite`.
    # 断言语句，验证解决方案的数量是否为3
    assert len(solve(sqrt(y)*x + x**3 - 1, x)) == 3
    # 断言语句，验证解决方案的数量是否为3，且不使用unrad选项
    assert len(solve(-512*y**3 + 1344*(x + 2)**Rational(1, 3)*y**2 -
        1176*(x + 2)**Rational(2, 3)*y - 169*x + 686, y, _unrad=False)) == 3


@slow
def test_issue_8828():
    # 定义几何参数
    x1 = 0
    y1 = -620
    r1 = 920
    x2 = 126
    y2 = 276
    x3 = 51
    y3 = 205
    r3 = 104
    v = x, y, z

    # 定义方程组 F
    f1 = (x - x1)**2 + (y - y1)**2 - (r1 - z)**2
    f2 = (x - x2)**2 + (y - y2)**2 - z**2
    f3 = (x - x3)**2 + (y - y3)**2 - (r3 - z)**2
    F = f1, f2, f3

    # 定义方程组 G
    g1 = sqrt((x - x1)**2 + (y - y1)**2) + z - r1
    g2 = f2
    g3 = sqrt((x - x3)**2 + (y - y3)**2) + z - r3
    G = g1, g2, g3

    # 求解方程组 F 和 G
    A = solve(F, v)
    B = solve(G, v)
    C = solve(G, v, manual=True)

    # 断言语句，验证解的唯一性
    p, q, r = [{tuple(i.evalf(2) for i in j) for j in R} for R in [A, B, C]]
    assert p == q == r


def test_issue_2840_8155():
    # 对于无参数解（即没有 `n` 的情况），希望避免过多的周期解
    assert solve(sin(3*x) + sin(6*x)) == [0, -2*pi/9, 2*pi/9]
    assert solve(sin(300*x) + sin(600*x)) == [0, -pi/450, pi/450]
    assert solve(2*sin(x) - 2*sin(2*x)) == [0, -pi/3, pi/3]


def test_issue_9567():
    # 断言语句，验证解是否为 [0]
    assert solve(1 + 1/(x - 1)) == [0]


def test_issue_11538():
    # 断言语句，验证解是否正确
    assert solve(x + E) == [-E]
    assert solve(x**2 + E) == [-I*sqrt(E), I*sqrt(E)]
    assert solve(x**3 + 2*E) == [
        -cbrt(2 * E),
        cbrt(2)*cbrt(E)/2 - cbrt(2)*sqrt(3)*I*cbrt(E)/2,
        cbrt(2)*cbrt(E)/2 + cbrt(2)*sqrt(3)*I*cbrt(E)/2]
    assert solve([x + 4, y + E], x, y) == {x: -4, y: -E}
    assert solve([x**2 + 4, y + E], x, y) == [
        (-2*I, -E), (2*I, -E)]

    # 定义两个方程，并验证解的数量
    e1 = x - y**3 + 4
    e2 = x + y + 4 + 4 * E
    assert len(solve([e1, e2], x, y)) == 3


@slow
def test_issue_12114():
    a, b, c, d, e, f, g = symbols('a,b,c,d,e,f,g')
    terms = [1 + a*b + d*e, 1 + a*c + d*f, 1 + b*c + e*f,
             g - a**2 - d**2, g - b**2 - e**2, g - c**2 - f**2]
    sol = solve(terms, [a, b, c, d, e, f, g], dict=True)
    s = sqrt(-f**2 - 1)
    s2 = sqrt(2 - f**2)
    s3 = sqrt(6 - 3*f**2)
    s4 = sqrt(3)*f
    s5 = sqrt(3)*s2
    # 断言语句，验证解是否正确
    assert sol == [
        {a: -s, b: -s, c: -s, d: f, e: f, g: -1},
        {a: s, b: s, c: s, d: f, e: f, g: -1},
        {a: -s4/2 - s2/2, b: s4/2 - s2/2, c: s2,
            d: -f/2 + s3/2, e: -f/2 - s5/2, g: 2},
        {a: -s4/2 + s2/2, b: s4/2 + s2/2, c: -s2,
            d: -f/2 - s3/2, e: -f/2 + s5/2, g: 2},
        {a: s4/2 - s2/2, b: -s4/2 - s2/2, c: s2,
            d: -f/2 - s3/2, e: -f/2 + s5/2, g: 2},
        {a: s4/2 + s2/2, b: -s4/2 + s2/2, c: -s2,
            d: -f/2 + s3/2, e: -f/2 - s5/2, g: 2}]


def test_inf():
    # 断言语句，验证解是否为空列表
    assert solve(1 - oo*x) == []
    assert solve(oo*x, x) == []
    assert solve(oo*x - oo, x) == []


def test_issue_12448():
    f = Function('f')
    fun = [f(i) for i in range(15)]
    # 定义符号变量 'x', 'y', 'z' 直到 'x:15'
    sym = symbols('x:15')
    
    # 将函数 'fun' 与符号变量一一对应，创建替换字典
    reps = dict(zip(fun, sym))
    
    # 将前三个符号变量分配给 (x, y, z)，将剩余的符号变量分配给 'c'
    (x, y, z), c = sym[:3], sym[3:]
    
    # 解决线性方程组，使用解集合 'ssym' 解决 c 的线性组合方程
    ssym = solve([c[4*i]*x + c[4*i + 1]*y + c[4*i + 2]*z + c[4*i + 3]
        for i in range(3)], (x, y, z))
    
    # 将前三个函数 'fun' 分配给 (x, y, z)，将剩余的函数分配给 'c'
    (x, y, z), c = fun[:3], fun[3:]
    
    # 解决线性方程组，使用解集合 'sfun' 解决 c 的线性组合方程
    sfun = solve([c[4*i]*x + c[4*i + 1]*y + c[4*i + 2]*z + c[4*i + 3]
        for i in range(3)], (x, y, z))
    
    # 断言函数 'sfun' 中的第一个元素 'fun[0]' 经过替换后的操作数数目
    # 等于解集合 'ssym' 中的第一个元素 'sym[0]' 的操作数数目
    assert sfun[fun[0]].xreplace(reps).count_ops() == \
        ssym[sym[0]].count_ops()
# 定义测试函数 test_denoms()
def test_denoms():
    # 断言调用 denoms 函数对表达式 x/2 + 1/y 进行求分母操作，期望得到集合 {2, y}
    assert denoms(x/2 + 1/y) == {2, y}
    # 断言调用 denoms 函数对表达式 x/2 + 1/y 进行求分母操作，限定分母为 y，期望得到集合 {y}
    assert denoms(x/2 + 1/y, y) == {y}
    # 断言调用 denoms 函数对表达式 x/2 + 1/y 进行求分母操作，限定分母为 [y]，期望得到集合 {y}
    assert denoms(x/2 + 1/y, [y]) == {y}
    # 断言调用 denoms 函数对表达式 1/x + 1/y + 1/z 进行求分母操作，限定分母为 [x, y]，期望得到集合 {x, y}
    assert denoms(1/x + 1/y + 1/z, [x, y]) == {x, y}
    # 断言调用 denoms 函数对表达式 1/x + 1/y + 1/z 进行求分母操作，限定分母为 x, y，期望得到集合 {x, y}
    assert denoms(1/x + 1/y + 1/z, x, y) == {x, y}
    # 断言调用 denoms 函数对表达式 1/x + 1/y + 1/z 进行求分母操作，限定分母为 {x, y}，期望得到集合 {x, y}
    assert denoms(1/x + 1/y + 1/z, {x, y}) == {x, y}


# 定义测试函数 test_issue_12476()
def test_issue_12476():
    # 定义符号变量 x0, x1, x2, x3, x4, x5
    x0, x1, x2, x3, x4, x5 = symbols('x0 x1 x2 x3 x4 x5')
    # 定义方程列表 eqns
    eqns = [
        x0**2 - x0,
        x0*x1 - x1,
        x0*x2 - x2,
        x0*x3 - x3,
        x0*x4 - x4,
        x0*x5 - x5,
        x0*x1 - x1,
        -x0/3 + x1**2 - 2*x2/3,
        x1*x2 - x1/3 - x2/3 - x3/3,
        x1*x3 - x2/3 - x3/3 - x4/3,
        x1*x4 - 2*x3/3 - x5/3,
        x1*x5 - x4,
        x0*x2 - x2,
        x1*x2 - x1/3 - x2/3 - x3/3,
        -x0/6 - x1/6 + x2**2 - x2/6 - x3/3 - x4/6,
        -x1/6 + x2*x3 - x2/3 - x3/6 - x4/6 - x5/6,
        x2*x4 - x2/3 - x3/3 - x4/3,
        x2*x5 - x3,
        x0*x3 - x3,
        x1*x3 - x2/3 - x3/3 - x4/3,
        -x1/6 + x2*x3 - x2/3 - x3/6 - x4/6 - x5/6,
        -x0/6 - x1/6 - x2/6 + x3**2 - x3/3 - x4/6,
        -x1/3 - x2/3 + x3*x4 - x3/3,
        -x2 + x3*x5,
        x0*x4 - x4,
        x1*x4 - 2*x3/3 - x5/3,
        x2*x4 - x2/3 - x3/3 - x4/3,
        -x1/3 - x2/3 + x3*x4 - x3/3,
        -x0/3 - 2*x2/3 + x4**2,
        -x1 + x4*x5,
        x0*x5 - x5,
        x1*x5 - x4,
        x2*x5 - x3,
        -x2 + x3*x5,
        -x1 + x4*x5,
        -x0 + x5**2,
        x0 - 1
    ]
    # 定义解集 sols
    sols = [
        {x0: 1, x3: Rational(1, 6), x2: Rational(1, 6), x4: Rational(-2, 3), x1: Rational(-2, 3), x5: 1},
        {x0: 1, x3: S.Half, x2: Rational(-1, 2), x4: 0, x1: 0, x5: -1},
        {x0: 1, x3: Rational(-1, 3), x2: Rational(-1, 3), x4: Rational(1, 3), x1: Rational(1, 3), x5: 1},
        {x0: 1, x3: 1, x2: 1, x4: 1, x1: 1, x5: 1},
        {x0: 1, x3: Rational(-1, 3), x2: Rational(1, 3), x4: sqrt(5)/3, x1: -sqrt(5)/3, x5: -1},
        {x0: 1, x3: Rational(-1, 3), x2: Rational(1, 3), x4: -sqrt(5)/3, x1: sqrt(5)/3, x5: -1}
    ]
    # 断言调用 solve 函数解决方程组 eqns，期望得到解集 sols
    assert solve(eqns) == sols


# 定义测试函数 test_issue_13849()
def test_issue_13849():
    # 定义符号变量 t
    t = symbols('t')
    # 断言调用 solve 函数解决方程组 (t*(sqrt(5) + sqrt(2)) - sqrt(2), t)，期望得到空列表 []
    assert solve((t*(sqrt(5) + sqrt(2)) - sqrt(2), t), t) == []


# 定义测试函数 test_issue_14860()
def test_issue_14860():
    # 导入 sympy.physics.units 中的 newton 和 kilo
    from sympy.physics.units import newton, kilo
    # 断言调用 solve 函数解决方程组 (8*kilo*newton + x + y, x)，期望得到列表 [-8000*newton - y]
    assert solve(8*kilo*newton + x + y, x) == [-8000*newton - y]


# 定义测试函数 test_issue_14721()
def test_issue_14721():
    # 定义符号变量 k, h, a, b
    k, h, a, b = symbols(':4')
    # 断言调用 solve 函数解决方程组列表，期望得到解集列表
    assert solve([
        -1 + (-k + 1)**2/b**2 + (-h - 1)**2/a**2,
        -1 + (-k + 1)**2/b**2 + (-h + 1)**2/a**2,
        h, k + 2], h, k, a, b) == [
        (0, -2, -b*sqrt(1/(b**2 - 9)), b),
        (0, -2, b*sqrt(1/(b**2 - 9)), b)]
    # 断言调用 solve 函数解决方程组列表，期望得到解集列表
    assert solve([
        h, h/a + 1/b**2 - 2, -h/2 + 1/b**2 - 2], a, h, b) == [
        (a, 0, -sqrt(2)/2), (a, 0, sqrt(2)/2)]
    # 断言调用 solve 函数解决方程组 (a + b**2 - 1, a + b**2 - 2)，期望得到空列表 []
    assert solve((
    # 使用 solve 函数验证给定表达式的解
    assert solve((y - 2, Add(x + 4, x - 2, evaluate=False))) == \
        {x: -1, y: 2}
    
    # 定义第一个方程式，12513*x + 2*y - 219093 = -5726*x - y
    eq1 = Eq(12513*x + 2*y - 219093, -5726*x - y)
    
    # 定义第二个方程式，-2*x + 8 = 2*x - 40
    eq2 = Eq(-2*x + 8, 2*x - 40)
    
    # 使用 solve 函数解方程组 [eq1, eq2]，并验证其解
    assert solve([eq1, eq2]) == {x:12, y:75}
def test_issue_15415():
    # 测试 solve 函数解决 x - 3 = 0 方程时返回 [3]
    assert solve(x - 3, x) == [3]
    # 测试 solve 函数解决 [x - 3] = {x: 3} 方程时返回 {x: 3}
    assert solve([x - 3], x) == {x: 3}
    # 测试 solve 函数解决 y + 3*x**2/2 = y + 3*x 方程时返回空列表 []
    assert solve(Eq(y + 3*x**2/2, y + 3*x), y) == []
    # 测试 solve 函数解决 [y + 3*x**2/2 = y + 3*x] = y 方程时返回空列表 []
    assert solve([Eq(y + 3*x**2/2, y + 3*x)], y) == []
    # 测试 solve 函数解决 [y + 3*x**2/2 = y + 3*x, x = 1] = y 方程时返回空列表 []
    assert solve([Eq(y + 3*x**2/2, y + 3*x), Eq(x, 1)], y) == []


@slow
def test_issue_15731():
    # 测试 solve 函数解决 (x**2 - 7*x + 11)**(x**2 - 13*x + 42) = 1 方程时返回 [2, 3, 4, 5, 6, 7]
    assert solve(Eq((x**2 - 7*x + 11)**(x**2 - 13*x + 42), 1)) == [2, 3, 4, 5, 6, 7]
    # 测试 solve 函数解决 x**(x + 4) - 4 = 0 方程时返回 [-2]
    assert solve(x**(x + 4) - 4) == [-2]
    # 测试 solve 函数解决 (-x)**(-x + 4) - 4 = 0 方程时返回 [2]
    assert solve((-x)**(-x + 4) - 4) == [2]
    # 测试 solve 函数解决 (x**2 - 6)**(x**2 - 2) - 4 = 0 方程时返回 [-2, 2]
    assert solve((x**2 - 6)**(x**2 - 2) - 4) == [-2, 2]
    # 测试 solve 函数解决 (x**2 - 2*x - 1)**(x**2 - 3) - 1/(1 - 2*sqrt(2)) = 0 方程时返回 [sqrt(2)]
    assert solve((x**2 - 2*x - 1)**(x**2 - 3) - 1/(1 - 2*sqrt(2))) == [sqrt(2)]
    # 测试 solve 函数解决 x**(x + S.Half) - 4*sqrt(2) = 0 方程时返回 [2]
    assert solve(x**(x + S.Half) - 4*sqrt(2)) == [S(2)]
    # 测试 solve 函数解决 (x**2 + 1)**x - 25 = 0 方程时返回 [2]
    assert solve((x**2 + 1)**x - 25) == [2]
    # 测试 solve 函数解决 x**(2/x) - 2 = 0 方程时返回 [2, 4]
    assert solve(x**(2/x) - 2) == [2, 4]
    # 测试 solve 函数解决 (x/2)**(2/x) - sqrt(2) = 0 方程时返回 [4, 8]
    assert solve((x/2)**(2/x) - sqrt(2)) == [4, 8]
    # 测试 solve 函数解决 x**(x + S.Half) - 9/4 = 0 方程时返回 [3/2]
    assert solve(x**(x + S.Half) - Rational(9, 4)) == [Rational(3, 2)]
    # 测试 solve 函数解决 (-sqrt(sqrt(2)))**x - 2 = 0 方程时返回 [4, log(2)/(log(2**Rational(1, 4)) + I*pi)]
    assert solve((-sqrt(sqrt(2)))**x - 2) == [4, log(2)/(log(2**Rational(1, 4)) + I*pi)]
    # 测试 solve 函数解决 (sqrt(2))**x - sqrt(sqrt(2)) = 0 方程时返回 [1/2]
    assert solve((sqrt(2))**x - sqrt(sqrt(2))) == [S.Half]
    # 测试 solve 函数解决 (-sqrt(2))**x + 2*(sqrt(2)) = 0 方程时返回 [3, (3*log(2)**2 + 4*pi**2 - 4*I*pi*log(2))/(log(2)**2 + 4*pi**2)]
    assert solve((-sqrt(2))**x + 2*(sqrt(2))) == [3, (3*log(2)**2 + 4*pi**2 - 4*I*pi*log(2))/(log(2)**2 + 4*pi**2)]
    # 测试 solve 函数解决 (sqrt(2))**x - 2*(sqrt(2)) = 0 方程时返回 [3]
    assert solve((sqrt(2))**x - 2*(sqrt(2))) == [3]
    # 测试 solve 函数解决 I**x + 1 = 0 方程时返回 [2]
    assert solve(I**x + 1) == [2]
    # 测试 solve 函数解决 (1 + I)**x - 2*I = 0 方程时返回 [2]
    assert solve((1 + I)**x - 2*I) == [2]
    # 测试 solve 函数解决 (sqrt(2) + sqrt(3))**x - (2*sqrt(6) + 5)**(1/3) = 0 方程时返回 [2/3]
    assert solve((sqrt(2) + sqrt(3))**x - (2*sqrt(6) + 5)**Rational(1, 3)) == [Rational(2, 3)]
    # 测试 solve 函数解决 b**x - b**2 = 0 方程时返回 [2]
    b = Symbol('b')
    assert solve(b**x - b**2, x) == [2]
    # 测试 solve 函数解决 b**x - 1/b = 0 方程时返回 [-1]
    assert solve(b**x - 1/b, x) == [-1]
    # 测试 solve 函数解决 b**x - b = 0 方程时返回 [1]
    assert solve(b**x - b, x) == [1]
    # 测试 solve 函数解决 b**x - b**2 = 0 方程时返回 [2]
    b = Symbol('b', positive=True)
    assert solve(b**x - b**2, x) == [2]
    # 测试 solve 函数解决 b**x - 1/b = 0 方程时返回 [-1]
    assert solve(b**x - 1/b, x) == [-1]


def test_issue_10933():
    # 测试 solve 函数解决 x**4 + y*(x + 0.1) = 0 方程时不出错
    assert solve(x**4 + y*(x + 0.1), x)
    # 测试 solve 函数解决 I*x**4 + x**3 + x**2 + 1. = 0 方程时不出错
    assert solve(I*x**4 + x**3 + x**2 + 1.)


def test_Abs_handling():
    x = symbols('x', real=True)
    # 测试 solve 函数解决 abs(x/y) = 0 方程时返回 [0]
    assert solve(abs(x/y), x) == [0]


def test_issue_7982():
    x = Symbol('x')
    # 测试 solve 函数解决 [2*x**2 + 5*x + 20 <= 0, x >= 1.5] = False 方程时返回 S.false
    assert solve([2*x**2 + 5*x + 20 <= 0, x >= 1.5], x) is S.false
    # 测试 solve 函数解决 [x**3 - 8.08*x**2 - 56.48*x/5 - 106 >= 0, x - 1 <= 0] = False 方程时返回 S.false
    assert solve([x**3 - 8.08*x**2 - 56.48*x/5 - 106
    # 计算给定的表达式 eq
    eq = -8*x**2/(9*(x**2 - 1)**(S(4)/3)) + 4/(3*(x**2 - 1)**(S(1)/3))
    
    # 使用 assert 语句来验证 unrad(eq) 返回 None
    assert unrad(eq) is None
def test_issue_17949():
    # Assert that the solutions for exp(+x+x**2) in terms of x are an empty list
    assert solve(exp(+x+x**2), x) == []
    # Assert that the solutions for exp(-x+x**2) in terms of x are an empty list
    assert solve(exp(-x+x**2), x) == []
    # Assert that the solutions for exp(+x-x**2) in terms of x are an empty list
    assert solve(exp(+x-x**2), x) == []
    # Assert that the solutions for exp(-x-x**2) in terms of x are an empty list
    assert solve(exp(-x-x**2), x) == []


def test_issue_10993():
    # Assert that solutions for equations involving binomial and power functions are as expected
    assert solve(Eq(binomial(x, 2), 3)) == [-2, 3]
    assert solve(Eq(pow(x, 2) + binomial(x, 3), x)) == [-4, 0, 1]
    assert solve(Eq(binomial(x, 2), 0)) == [0, 1]
    assert solve(a+binomial(x, 3), a) == [-binomial(x, 3)]
    assert solve(x-binomial(a, 3) + binomial(y, 2) + sin(a), x) == [-sin(a) + binomial(a, 3) - binomial(y, 2)]
    assert solve((x+1)-binomial(x+1, 3), x) == [-2, -1, 3]


def test_issue_11553():
    # Define equations
    eq1 = x + y + 1
    eq2 = x + GoldenRatio
    # Assert that simultaneous solutions for eq1 and eq2 in terms of x and y are as expected
    assert solve([eq1, eq2], x, y) == {x: -GoldenRatio, y: -1 + GoldenRatio}
    # Define another equation
    eq3 = x + 2 + TribonacciConstant
    # Assert that simultaneous solutions for eq1 and eq3 in terms of x and y are as expected
    assert solve([eq1, eq3], x, y) == {x: -2 - TribonacciConstant, y: 1 + TribonacciConstant}


def test_issue_19113_19102():
    # Define equations involving trigonometric and inverse trigonometric functions
    t = S(1)/3
    solve(cos(x)**5-sin(x)**5)  # Solve equation involving powers of trigonometric functions
    # Assert that solutions for a trigonometric equation are as expected
    assert solve(4*cos(x)**3 - 2*sin(x)**3) == [
        atan(2**(t)), -atan(2**(t)*(1 - sqrt(3)*I)/2),
        -atan(2**(t)*(1 + sqrt(3)*I)/2)]
    h = S.Half
    # Assert that solutions for another trigonometric equation are as expected
    assert solve(cos(x)**2 + sin(x)) == [
        2*atan(-h + sqrt(5)/2 + sqrt(2)*sqrt(1 - sqrt(5))/2),
        -2*atan(h + sqrt(5)/2 + sqrt(2)*sqrt(1 + sqrt(5))/2),
        -2*atan(-sqrt(5)/2 + h + sqrt(2)*sqrt(1 - sqrt(5))/2),
        -2*atan(-sqrt(2)*sqrt(1 + sqrt(5))/2 + h + sqrt(5)/2)]
    # Assert that solutions for a trigonometric equation involving constants are as expected
    assert solve(3*cos(x) - sin(x)) == [atan(3)]


def test_issue_19509():
    # Define constants and assert solutions for a complex rational equation
    a = S(3)/4
    b = S(5)/8
    c = sqrt(5)/8
    d = sqrt(5)/4
    assert solve(1/(x -1)**5 - 1) == [2,
        -d + a - sqrt(-b + c),
        -d + a + sqrt(-b + c),
        d + a - sqrt(-b - c),
        d + a + sqrt(-b - c)]


def test_issue_20747():
    # Define symbols and equations involving exponential and logarithmic functions
    THT, HT, DBH, dib, c0, c1, c2, c3, c4  = symbols('THT HT DBH dib c0 c1 c2 c3 c4')
    f = DBH*c3 + THT*c4 + c2
    rhs = 1 - ((HT - 1)/(THT - 1))**c1*(1 - exp(c0/f))
    eq = dib - DBH*(c0 - f*log(rhs))
    term = ((1 - exp((DBH*c0 - dib)/(DBH*(DBH*c3 + THT*c4 + c2))))
            / (1 - exp(c0/(DBH*c3 + THT*c4 + c2))))
    sol = [THT*term**(1/c1) - term**(1/c1) + 1]
    # Assert that solutions for eq in terms of HT are as expected
    assert solve(eq, HT) == sol


def test_issue_20902():
    # Define symbols and assert solutions for a function involving a parameter substitution and differentiation
    f = (t / ((1 + t) ** 2))
    assert solve(f.subs({t: 3 * x + 2}).diff(x) > 0, x) == (S(-1) < x) & (x < S(-1)/3)
    assert solve(f.subs({t: 3 * x + 3}).diff(x) > 0, x) == (S(-4)/3 < x) & (x < S(-2)/3)
    assert solve(f.subs({t: 3 * x + 4}).diff(x) > 0, x) == (S(-5)/3 < x) & (x < S(-1))
    assert solve(f.subs({t: 3 * x + 2}).diff(x) > 0, x) == (S(-1) < x) & (x < S(-1)/3)


def test_issue_21034():
    # Define symbols and assert solutions for a system of equations involving hyperbolic functions
    a = symbols('a', real=True)
    system = [x - cosh(cos(4)), y - sinh(cos(a)), z - tanh(x)]
    # Constants inside hyperbolic functions should not be rewritten in terms of exp
    assert solve(system, x, y, z) == [(cosh(cos(4)), sinh(cos(a)), tanh(cosh(cos(4))))]
    # If the variable of interest is present in a hyperbolic function, then rewrite in terms of exp and solve further
    # 定义一个包含一个方程的列表，方程为 exp(x) - exp(-x) - tanh(x)*(exp(x) + exp(-x)) + x - 5
    newsystem = [(exp(x) - exp(-x)) - tanh(x)*(exp(x) + exp(-x)) + x - 5]
    
    # 使用 solve 函数求解上述方程组，期望得到一个字典形式的解，解中 x 的值为 5
    assert solve(newsystem, x) == {x: 5}
def test_issue_4886():
    # Calculate z using the given formula involving constants a, R, b, and c
    z = a*sqrt(R**2*a**2 + R**2*b**2 - c**2)/(a**2 + b**2)
    # Calculate t using constants b, c, and a
    t = b*c/(a**2 + b**2)
    # Compute solutions for the equations using solve function and compare with expected solution sol
    sol = [((b*(t - z) - c)/(-a), t - z), ((b*(t + z) - c)/(-a), t + z)]
    # Assert that solve function returns expected solution sol
    assert solve([x**2 + y**2 - R**2, a*x + b*y - c], x, y) == sol


def test_issue_6819():
    # Declare symbols a, b, c, d as positive
    a, b, c, d = symbols('a b c d', positive=True)
    # Assert that solve function returns the expected solution
    assert solve(a*b**x - c*d**x, x) == [log(c/a)/log(b/d)]


def test_issue_17454():
    # Declare symbol x
    x = Symbol('x')
    # Assert that solve function returns the expected solution
    assert solve((1 - x - I)**4, x) == [1 - I]


def test_issue_21852():
    # Define the expected solution
    solution = [21 - 21*sqrt(2)/2]
    # Assert that solve function returns the expected solution
    assert solve(2*x + sqrt(2*x**2) - 21) == solution


def test_issue_21942():
    # Define equation eq
    eq = -d + (a*c**(1 - e) + b**(1 - e)*(1 - a))**(1/(1 - e))
    # Solve eq for variable c, with simplify and check set to False
    sol = solve(eq, c, simplify=False, check=False)
    # Assert that solve function returns the expected solution sol
    assert sol == [((a*b**(1 - e) - b**(1 - e) +
        d**(1 - e))/a)**(1/(1 - e))]


def test_solver_flags():
    # Solve equation x**5 + x**2 - x - 1 for root with cubics=False
    root = solve(x**5 + x**2 - x - 1, cubics=False)
    # Solve equation x**5 + x**2 - x - 1 for root with cubics=True
    rad = solve(x**5 + x**2 - x - 1, cubics=True)
    # Assert that root and rad solutions are not equal
    assert root != rad


def test_issue_22768():
    # Define equation eq
    eq = 2*x**3 - 16*(y - 1)**6*z**3
    # Solve expanded equation eq for variable x, with simplify set to False
    assert solve(eq.expand(), x, simplify=False
        ) == [2*z*(y - 1)**2, z*(-1 + sqrt(3)*I)*(y - 1)**2,
        -z*(1 + sqrt(3)*I)*(y - 1)**2]


def test_issue_22717():
    # Assert that solve function returns expected solutions for the given equations
    assert solve((-y**2 + log(y**2/x) + 2, -2*x*y + 2*x/y)) == [
        {y: -1, x: E}, {y: 1, x: E}]


def test_issue_25176():
    # Define equation eq
    eq = (x - 5)**-8 - 3
    # Solve equation eq
    sol = solve(eq)
    # Assert that none of the solutions for eq substituted into eq itself yield True
    assert not any(eq.subs(x, i) for i in sol)


def test_issue_10169():
    # Define a large symbolic equation eq involving variables a, b, c, d, e, k and x
    eq = S(-8*a - x**5*(a + b + c + e) - x**4*(4*a - 2**Rational(3,4)*c + 4*c +
        d + 2**Rational(3,4)*e + 4*e + k) - x**3*(-4*2**Rational(3,4)*c + sqrt(2)*c -
        2**Rational(3,4)*d + 4*d + sqrt(2)*e + 4*2**Rational(3,4)*e + 2**Rational(3,4)*k + 4*k) -
        x**2*(4*sqrt(2)*c - 4*2**Rational(3,4)*d + sqrt(2)*d + 4*sqrt(2)*e +
        sqrt(2)*k + 4*2**Rational(3,4)*k) - x*(2*a + 2*b + 4*sqrt(2)*d +
        4*sqrt(2)*k) + 5)
    # Assert that solve_undetermined_coeffs function returns expected solutions for eq
    assert solve_undetermined_coeffs(eq, [a, b, c, d, e, k], x) == {
        a: Rational(5,8),
        b: Rational(-5,1032),
        c: Rational(-40,129) - 5*2**Rational(3,4)/129 + 5*2**Rational(1,4)/1032,
        d: -20*2**Rational(3,4)/129 - 10*sqrt(2)/129 - 5*2**Rational(1,4)/258,
        e: Rational(-40,129) - 5*2**Rational(1,4)/1032 + 5*2**Rational(3,4)/129,
        k: -10*sqrt(2)/129 + 5*2**Rational(1,4)/258 + 20*2**Rational(3,4)/129
    }


def test_solve_undetermined_coeffs_issue_23927():
    # Declare symbols A, B, r, phi
    A, B, r, phi = symbols('A, B, r, phi')
    # Define equation e
    e = Eq(A*sin(t) + B*cos(t), r*sin(t - phi))
    # Expand equation e and solve for undetermined coefficients r and phi in terms of t
    eq = (e.lhs - e.rhs).expand(trig=True)
    # Solve equation eq using solve_undetermined_coeffs function
    soln = solve_undetermined_coeffs(eq, (r, phi), t)
    # Assert that solve_undetermined_coeffs function returns expected solutions soln
    assert soln == [{
        phi: 2*atan((A - sqrt(A**2 + B**2))/B),
        r: (-A**2 + A*sqrt(A**2 + B**2) - B**2)/(A - sqrt(A**2 + B**2))
        }, {
        phi: 2*atan((A + sqrt(A**2 + B**2))/B),
        r: (A**2 + A*sqrt(A**2 + B**2) + B**2)/(A + sqrt(A**2 + B**2))/-1
        }]
    # 抛出 NotImplementedError 异常，确保不会因为 RuntimeError 失败
    raises(NotImplementedError, lambda: solve(Mod(x**2, 49), x))
    # 创建一个整数类型的符号变量 s2，要求其为正数
    s2 = Symbol('s2', integer=True, positive=True)
    # 计算 s2/2 - 1/2 的向下取整值，并赋给变量 f
    f = floor(s2/2 - S(1)/2)
    # 抛出 NotImplementedError 异常，处理复杂的方程求解
    raises(NotImplementedError, lambda: solve((Mod(f**2/(f + 1) + 2*f/(f + 1) + 1/(f + 1), 1))*f + Mod(f**2/(f + 1) + 2*f/(f + 1) + 1/(f + 1), 1), s2))
# 定义测试函数 test_solve_Piecewise，用于测试 solve 函数解决 Piecewise 表达式的结果是否符合预期
def test_solve_Piecewise():
    # 使用断言检查解的列表是否等于 [S(10)/3]
    assert [S(10)/3] == solve(
        # 求解 Piecewise 函数，包含多个条件和对应的表达式
        3 * Piecewise(
            # 第一个条件：当 x <= 0 时返回 S.NaN
            (S.NaN, x <= 0),
            # 第二个条件：当 x >= 0 且 x >= 2 且 x >= 4 且 x >= 6 且 x < 10 时，返回 20*x - 3*(x - 6)**2/2 - 176
            (20*x - 3*(x - 6)**2/2 - 176, (x >= 0) & (x >= 2) & (x >= 4) & (x >= 6) & (x < 10)),
            # 第三个条件：当 x >= 0 且 x >= 2 且 x >= 4 且 x < 10 时，返回 100 - 26*x
            (100 - 26*x, (x >= 0) & (x >= 2) & (x >= 4) & (x < 10)),
            # 第四个条件：当 x >= 2 且 x >= 4 且 x >= 6 且 x < 10 时，返回 16*x - 3*(x - 6)**2/2 - 176
            (16*x - 3*(x - 6)**2/2 - 176, (x >= 2) & (x >= 4) & (x >= 6) & (x < 10)),
            # 第五个条件：当 x >= 2 且 x >= 4 且 x < 10 时，返回 100 - 30*x
            (100 - 30*x, (x >= 2) & (x >= 4) & (x < 10)),
            # 第六个条件：当 x >= 0 且 x >= 4 且 x >= 6 且 x < 10 时，返回 30*x - 3*(x - 6)**2/2 - 196
            (30*x - 3*(x - 6)**2/2 - 196, (x >= 0) & (x >= 4) & (x >= 6) & (x < 10)),
            # 第七个条件：当 x >= 0 且 x >= 4 且 x < 10 时，返回 80 - 16*x
            (80 - 16*x, (x >= 0) & (x >= 4) & (x < 10)),
            # 第八个条件：当 x >= 4 且 x >= 6 且 x < 10 时，返回 26*x - 3*(x - 6)**2/2 - 196
            (26*x - 3*(x - 6)**2/2 - 196, (x >= 4) & (x >= 6) & (x < 10)),
            # 第九个条件：当 x >= 4 且 x < 10 时，返回 80 - 20*x
            (80 - 20*x, (x >= 4) & (x < 10)),
            # 第十个条件：当 x >= 0 且 x >= 2 且 x >= 6 且 x < 10 时，返回 40*x - 3*(x - 6)**2/2 - 256
            (40*x - 3*(x - 6)**2/2 - 256, (x >= 0) & (x >= 2) & (x >= 6) & (x < 10)),
            # 第十一个条件：当 x >= 0 且 x >= 2 且 x < 10 时，返回 20 - 6*x
            (20 - 6*x, (x >= 0) & (x >= 2) & (x < 10)),
            # 第十二个条件：当 x >= 2 且 x >= 6 且 x < 10 时，返回 36*x - 3*(x - 6)**2/2 - 256
            (36*x - 3*(x - 6)**2/2 - 256, (x >= 2) & (x >= 6) & (x < 10)),
            # 第十三个条件：当 x >= 2 且 x < 10 时，返回 20 - 10*x
            (20 - 10*x, (x >= 2) & (x < 10)),
            # 第十四个条件：当 x >= 0 且 x >= 6 且 x < 10 时，返回 50*x - 3*(x - 6)**2/2 - 276
            (50*x - 3*(x - 6)**2/2 - 276, (x >= 0) & (x >= 6) & (x < 10)),
            # 第十五个条件：当 x >= 0 且 x < 10 时，返回 4*x
            (4*x, (x >= 0) & (x < 10)),
            # 第十六个条件：当 x >= 6 且 x < 10 时，返回 46*x - 3*(x - 6)**2/2 - 276
            (46*x - 3*(x - 6)**2/2 - 276, (x >= 6) & (x < 10)),
            # 第十七个条件：当 x < 10 时，返回 0
            (0, x < 10),
            # 默认条件：其余情况返回 S.NaN
            (S.NaN, True)
        )
    )
```