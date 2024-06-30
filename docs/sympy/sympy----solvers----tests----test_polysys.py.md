# `D:\src\scipysrc\sympy\sympy\solvers\tests\test_polysys.py`

```
"""Tests for solvers of systems of polynomial equations. """

# 导入所需的 SymPy 模块和函数
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polyerrors import UnsolvableFactorError
from sympy.polys.polyoptions import Options
from sympy.polys.polytools import Poly
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import flatten
from sympy.abc import x, y, z
from sympy.polys import PolynomialError
from sympy.solvers.polysys import (solve_poly_system,
                                   solve_triangulated,
                                   solve_biquadratic, SolveFailed,
                                   solve_generic)
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.testing.pytest import raises

# 定义测试函数 test_solve_poly_system
def test_solve_poly_system():
    # 测试1：解一个简单的一元线性方程
    assert solve_poly_system([x - 1], x) == [(S.One,)]

    # 测试2：解一个无解的线性方程组
    assert solve_poly_system([y - x, y - x - 1], x, y) is None

    # 测试3：解一个二元非线性方程组
    assert solve_poly_system([y - x**2, y + x**2], x, y) == [(S.Zero, S.Zero)]

    # 测试4：解一个三元线性方程组
    assert solve_poly_system([2*x - 3, y*Rational(3, 2) - 2*x, z - 5*y], x, y, z) == \
        [(Rational(3, 2), Integer(2), Integer(10))]

    # 测试5：解一个二元非线性方程组
    assert solve_poly_system([x*y - 2*y, 2*y**2 - x**2], x, y) == \
        [(0, 0), (2, -sqrt(2)), (2, sqrt(2))]

    # 测试6：解一个包含复数解的二元非线性方程组
    assert solve_poly_system([y - x**2, y + x**2 + 1], x, y) == \
        [(-I*sqrt(S.Half), Rational(-1, 2)), (I*sqrt(S.Half), Rational(-1, 2))]

    # 测试7：解一个包含多个方程的三元非线性方程组
    f_1 = x**2 + y + z - 1
    f_2 = x + y**2 + z - 1
    f_3 = x + y + z**2 - 1
    a, b = sqrt(2) - 1, -sqrt(2) - 1
    assert solve_poly_system([f_1, f_2, f_3], x, y, z) == \
        [(0, 0, 1), (0, 1, 0), (1, 0, 0), (a, a, a), (b, b, b)]

    # 测试8：解一个包含多项式对象的方程组
    solution = [(1, -1), (1, 1)]
    assert solve_poly_system([Poly(x**2 - y**2), Poly(x - 1)]) == solution
    assert solve_poly_system([x**2 - y**2, x - 1], x, y) == solution
    assert solve_poly_system([x**2 - y**2, x - 1]) == solution

    # 测试9：解一个包含分式的方程组
    assert solve_poly_system(
        [x + x*y - 3, y + x*y - 4], x, y) == [(-3, -2), (1, 2)]

    # 测试10：抛出未实现错误
    raises(NotImplementedError, lambda: solve_poly_system([x**3 - y**3], x, y))

    # 测试11：抛出多项式错误
    raises(PolynomialError, lambda: solve_poly_system([1/x], x))

    # 测试12：抛出未实现错误（使用元组作为符号变量）
    raises(NotImplementedError, lambda: solve_poly_system([x-1,], (x, y)))
    raises(NotImplementedError, lambda: solve_poly_system([y-1,], (x, y)))

    # solve_poly_system 在以下四个测试中最好使用 CRootOf 构造解
    # 测试13：使用 strict=False 不严格求解
    assert solve_poly_system([x**5 - x + 1], [x], strict=False) == []

    # 测试14：使用 strict=True 严格求解，应抛出 UnsolvableFactorError
    raises(UnsolvableFactorError, lambda: solve_poly_system(
        [x**5 - x + 1], [x], strict=True))

    # 测试15：解一个包含多项式和方程的方程组
    assert solve_poly_system([(x - 1)*(x**5 - x + 1), y**2 - 1], [x, y],
                             strict=False) == [(1, -1), (1, 1)]
    # 在解决多项式系统时，期望引发 UnsolvableFactorError 异常，
    # 通过 lambda 表达式传入的函数调用 solve_poly_system 来尝试解决包含的方程组
    raises(UnsolvableFactorError,
           lambda: solve_poly_system([(x - 1)*(x**5 - x + 1), y**2-1],
                                     [x, y], strict=True))
# 定义一个测试函数，用于测试 solve_generic 函数的功能
def test_solve_generic():
    # 创建一个 Options 对象 NewOption，指定元组 (x, y) 作为变量，domain 设为 'ZZ'
    NewOption = Options((x, y), {'domain': 'ZZ'})
    # 断言 solve_generic 函数对于给定的方程组返回期望的解
    assert solve_generic([x**2 - 2*y**2, y**2 - y + 1], NewOption) == \
           [(-sqrt(-1 - sqrt(3)*I), Rational(1, 2) - sqrt(3)*I/2),
            (sqrt(-1 - sqrt(3)*I), Rational(1, 2) - sqrt(3)*I/2),
            (-sqrt(-1 + sqrt(3)*I), Rational(1, 2) + sqrt(3)*I/2),
            (sqrt(-1 + sqrt(3)*I), Rational(1, 2) + sqrt(3)*I/2)]

    # solve_generic 函数在以下两个测试中应当使用 CRootOf 构造解
    assert solve_generic(
        [2*x - y, (y - 1)*(y**5 - y + 1)], NewOption, strict=False) == \
        [(Rational(1, 2), 1)]
    # 使用 lambda 函数检查是否引发 UnsolvableFactorError 异常
    raises(UnsolvableFactorError, lambda: solve_generic(
        [2*x - y, (y - 1)*(y**5 - y + 1)], NewOption, strict=True))


# 定义一个测试函数，用于测试 solve_biquadratic 函数的功能
def test_solve_biquadratic():
    # 定义符号变量
    x0, y0, x1, y1, r = symbols('x0 y0 x1 y1 r')

    # 定义两个圆的方程 f_1 和 f_2
    f_1 = (x - 1)**2 + (y - 1)**2 - r**2
    f_2 = (x - 2)**2 + (y - 2)**2 - r**2
    # 计算预期的解
    s = sqrt(2*r**2 - 1)
    a = (3 - s)/2
    b = (3 + s)/2
    # 断言 solve_poly_system 函数返回期望的解
    assert solve_poly_system([f_1, f_2], x, y) == [(a, b), (b, a)]

    # 重新定义 f_1 和 f_2
    f_1 = (x - 1)**2 + (y - 2)**2 - r**2
    f_2 = (x - 1)**2 + (y - 1)**2 - r**2
    # 计算预期的解
    assert solve_poly_system([f_1, f_2], x, y) == \
        [(1 - sqrt((2*r - 1)*(2*r + 1))/2, Rational(3, 2)),
         (1 + sqrt((2*r - 1)*(2*r + 1))/2, Rational(3, 2))]

    # 定义一个查询 lambda 函数，用于检查解中是否包含 S.Half 的幂表达式
    query = lambda expr: expr.is_Pow and expr.exp is S.Half

    # 重新定义 f_1 和 f_2
    f_1 = (x - 1)**2 + (y - 2)**2 - r**2
    f_2 = (x - x1)**2 + (y - 1)**2 - r**2
    # 计算预期的解
    result = solve_poly_system([f_1, f_2], x, y)
    # 断言解的长度为 2，每个解含有一个 S.Half 的幂表达式
    assert len(result) == 2 and all(len(r) == 2 for r in result)
    assert all(r.count(query) == 1 for r in flatten(result))

    # 重新定义 f_1 和 f_2
    f_1 = (x - x0)**2 + (y - y0)**2 - r**2
    f_2 = (x - x1)**2 + (y - y1)**2 - r**2
    # 计算预期的解
    result = solve_poly_system([f_1, f_2], x, y)
    # 断言解的长度为 2，每个解含有一个 S.Half 的幂表达式
    assert len(result) == 2 and all(len(r) == 2 for r in result)
    assert all(len(r.find(query)) == 1 for r in flatten(result))

    # 定义两个方程组
    s1 = (x*y - y, x**2 - x)
    s2 = (x*y - x, y**2 - y)
    # 断言 solve 函数返回预期的解集
    assert solve(s1) == [{x: 1}, {x: 0, y: 0}]
    assert solve(s2) == [{y: 1}, {x: 0, y: 0}]
    # 定义变量 gens 为元组 (x, y)
    gens = (x, y)
    # 迭代处理方程组 s1 和 s2
    for seq in (s1, s2):
        # 使用 parallel_poly_from_expr 函数获取 (f, g) 和 opt
        (f, g), opt = parallel_poly_from_expr(seq, *gens)
        # 使用 lambda 函数检查是否引发 SolveFailed 异常
        raises(SolveFailed, lambda: solve_biquadratic(f, g, opt))
    # 定义方程组 seq
    seq = (x**2 + y**2 - 2, y**2 - 1)
    # 使用 parallel_poly_from_expr 函数获取 (f, g) 和 opt
    (f, g), opt = parallel_poly_from_expr(seq, *gens)
    # 断言 solve_biquadratic 函数返回预期的解集
    assert solve_biquadratic(f, g, opt) == [
        (-1, -1), (-1, 1), (1, -1), (1, 1)]
    # 定义预期的解集 ans
    ans = [(0, -1), (0, 1)]
    # 重新定义方程组 seq
    seq = (x**2 + y**2 - 1, y**2 - 1)
    # 使用 parallel_poly_from_expr 函数获取 (f, g) 和 opt
    (f, g), opt = parallel_poly_from_expr(seq, *gens)
    # 断言 solve_biquadratic 函数返回预期的解集 ans
    assert solve_biquadratic(f, g, opt) == ans
    # 重新定义方程组 seq
    seq = (x**2 + y**2 - 1, x**2 - x + y**2 - 1)
    # 使用 parallel_poly_from_expr 函数获取 (f, g) 和 opt
    (f, g), opt = parallel_poly_from_expr(seq, *gens)
    # 断言 solve_biquadratic 函数返回预期的解集 ans
    assert solve_biquadratic(f, g, opt) == ans


# 定义一个测试函数，用于测试 solve_triangulated 函数的功能
def test_solve_triangulated():
    # 定义三个平面的方程 f_1, f_2, f_3
    f_1 = x**2 + y + z - 1
    f_2 = x + y**2 + z - 1
    f_3 = x + y + z**2 - 1
    # 定义预期的解集
    a, b = sqrt(2) - 1, -sqrt(2) - 1
    # 断言 solve_triangulated 函数返回预期的解集
    assert solve_triangulated([f_1, f_2, f_3], x, y, z) == \
        [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    # 使用 QQ.algebraic_field(sqrt(2)) 创建一个域对象 dom，其中 sqrt(2) 是代数域的一部分
    dom = QQ.algebraic_field(sqrt(2))
    
    # 调用 solve_triangulated 函数，解决由 [f_1, f_2, f_3] 组成的三角化系统，通过变量 x, y, z 进行参数化
    # 使用指定的域对象 dom 进行计算，期望结果是 [(0, 0, 1), (0, 1, 0), (1, 0, 0), (a, a, a), (b, b, b)]
    assert solve_triangulated([f_1, f_2, f_3], x, y, z, domain=dom) == \
        [(0, 0, 1), (0, 1, 0), (1, 0, 0), (a, a, a), (b, b, b)]
# 定义解决问题 3686 的测试函数
def test_solve_issue_3686():
    # 求解多项式系统，返回根
    roots = solve_poly_system([((x - 5)**2/250000 + (y - Rational(5, 10))**2/250000) - 1, x], x, y)
    # 断言根与预期相符
    assert roots == [(0, S.Half - 15*sqrt(1111)), (0, S.Half + 15*sqrt(1111))]

    # 重新计算多项式系统的根
    roots = solve_poly_system([((x - 5)**2/250000 + (y - 5.0/10)**2/250000) - 1, x], x, y)
    # TODO: 这真的需要这么复杂吗？！
    # 断言根的数量为 2
    assert len(roots) == 2
    # 断言第一个根的第一个元素为 0
    assert roots[0][0] == 0
    # 断言第一个根的第二个元素在一定误差范围内等于 -499.474999374969
    assert roots[0][1].epsilon_eq(-499.474999374969, 1e12)
    # 断言第二个根的第一个元素为 0
    assert roots[1][0] == 0
    # 断言第二个根的第二个元素在一定误差范围内等于 500.474999374969
    assert roots[1][1].epsilon_eq(500.474999374969, 1e12)
```