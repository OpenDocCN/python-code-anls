# `D:\src\scipysrc\sympy\sympy\integrals\tests\test_prde.py`

```
"""Most of these tests come from the examples in Bronstein's book."""
# 导入必要的函数和类：DifferentialExtension, derivation, prde_normal_denom, prde_special_denom, 等等
from sympy.integrals.risch import DifferentialExtension, derivation
from sympy.integrals.prde import (prde_normal_denom, prde_special_denom,
    prde_linear_constraints, constant_system, prde_spde, prde_no_cancel_b_large,
    prde_no_cancel_b_small, limited_integrate_reduce, limited_integrate,
    is_deriv_k, is_log_deriv_k_t_radical, parametric_log_deriv_heu,
    is_log_deriv_k_t_radical_in_field, param_poly_rischDE, param_rischDE,
    prde_cancel_liouvillian)

# 导入并重命名 PolyMatrix 为 Matrix
from sympy.polys.polymatrix import PolyMatrix as Matrix

# 导入必要的符号和对象：Rational, S, symbols, QQ, Poly, x, t, n
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polytools import Poly
from sympy.abc import x, t, n

# 定义符号变量
t0, t1, t2, t3, k = symbols('t:4 k')

# 定义测试函数 test_prde_normal_denom
def test_prde_normal_denom():
    # 创建 DifferentialExtension 对象 DE，指定其扩展部分为 {'D': [Poly(1, x), Poly(1 + t**2, t)]}
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2, t)]})
    # 定义多项式对象 fa 和 fd
    fa = Poly(1, t)
    fd = Poly(x, t)
    # 定义 G 列表
    G = [(Poly(t, t), Poly(1 + t**2, t)), (Poly(1, t), Poly(x + x*t**2, t))]
    # 断言 prde_normal_denom 的返回值与预期相等
    assert prde_normal_denom(fa, fd, G, DE) == \
        (Poly(x, t, domain='ZZ(x)'), (Poly(1, t, domain='ZZ(x)'), Poly(1, t,
            domain='ZZ(x)')), [(Poly(x*t, t, domain='ZZ(x)'),
         Poly(t**2 + 1, t, domain='ZZ(x)')), (Poly(1, t, domain='ZZ(x)'),
             Poly(t**2 + 1, t, domain='ZZ(x)'))], Poly(1, t, domain='ZZ(x)'))
    
    # 更新 G 列表
    G = [(Poly(t, t), Poly(t**2 + 2*t + 1, t)), (Poly(x*t, t),
        Poly(t**2 + 2*t + 1, t)), (Poly(x*t**2, t), Poly(t**2 + 2*t + 1, t))]
    # 更新 DE 对象
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    # 断言 prde_normal_denom 的返回值与预期相等
    assert prde_normal_denom(Poly(x, t), Poly(1, t), G, DE) == \
        (Poly(t + 1, t), (Poly((-1 + x)*t + x, t), Poly(1, t, domain='ZZ[x]')), [(Poly(t, t),
        Poly(1, t)), (Poly(x*t, t), Poly(1, t, domain='ZZ[x]')), (Poly(x*t**2, t),
        Poly(1, t, domain='ZZ[x]'))], Poly(t + 1, t))

# 定义测试函数 test_prde_special_denom
def test_prde_special_denom():
    # 定义多项式对象 a, ba, bd
    a = Poly(t + 1, t)
    ba = Poly(t**2, t)
    bd = Poly(1, t)
    # 定义 G 列表
    G = [(Poly(t, t), Poly(1, t)), (Poly(t**2, t), Poly(1, t)), (Poly(t**3, t), Poly(1, t))]
    # 创建 DifferentialExtension 对象 DE，指定其扩展部分为 {'D': [Poly(1, x), Poly(t, t)]}
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    # 断言 prde_special_denom 的返回值与预期相等
    assert prde_special_denom(a, ba, bd, G, DE) == \
        (Poly(t + 1, t), Poly(t**2, t), [(Poly(t, t), Poly(1, t)),
        (Poly(t**2, t), Poly(1, t)), (Poly(t**3, t), Poly(1, t))], Poly(1, t))
    
    # 更新 G 列表
    G = [(Poly(t, t), Poly(1, t)), (Poly(1, t), Poly(t, t))]
    # 断言 prde_special_denom 的返回值与预期相等
    assert prde_special_denom(Poly(1, t), Poly(t**2, t), Poly(1, t), G, DE) == \
        (Poly(1, t), Poly(t**2 - 1, t), [(Poly(t**2, t), Poly(1, t)),
        (Poly(1, t), Poly(1, t))], Poly(t, t))
    
    # 更新 DE 对象
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-2*x*t0, t0)]})
    # 降低 DifferentialExtension 对象 DE 的级别
    DE.decrement_level()
    # 更新 G 列表
    G = [(Poly(t, t), Poly(t**2, t)), (Poly(2*t, t), Poly(t, t))]
    # 断言，验证 prde_special_denom 函数的返回结果是否符合预期
    assert prde_special_denom(Poly(5*x*t + 1, t), Poly(t**2 + 2*x**3*t, t), Poly(t**3 + 2, t), G, DE) == \
        (Poly(5*x*t + 1, t), Poly(0, t, domain='ZZ[x]'), [(Poly(t, t), Poly(t**2, t)),
        (Poly(2*t, t), Poly(t, t))], Poly(1, x))
    
    # 初始化 DifferentialExtension 对象 DE，指定其扩展项为两个多项式
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly((t**2 + 1)*2*x, t)]})
    
    # 初始化列表 G，包含两个元组，每个元组中都有两个多项式
    G = [(Poly(t + x, t), Poly(t*x, t)), (Poly(2*t, t), Poly(x**2, x))]
    
    # 断言，验证 prde_special_denom 函数的返回结果是否符合预期
    assert prde_special_denom(Poly(5*x*t + 1, t), Poly(t**2 + 2*x**3*t, t), Poly(t**3, t), G, DE) == \
        (Poly(5*x*t + 1, t), Poly(0, t, domain='ZZ[x]'), [(Poly(t + x, t), Poly(x*t, t)),
        (Poly(2*t, t, x), Poly(x**2, t, x))], Poly(1, t))
    
    # 断言，验证 prde_special_denom 函数的返回结果是否符合预期
    assert prde_special_denom(Poly(t + 1, t), Poly(t**2, t), Poly(t**3, t), G, DE) == \
        (Poly(t + 1, t), Poly(0, t, domain='ZZ[x]'), [(Poly(t + x, t), Poly(x*t, t)), (Poly(2*t, t, x),
        Poly(x**2, t, x))], Poly(1, t))
# 定义测试函数 `test_prde_linear_constraints`，用于测试 `prde_linear_constraints` 函数
def test_prde_linear_constraints():
    # 创建差分扩展对象 DE，其扩展包括导数 'D' 对应的多项式 Poly(1, x)
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    
    # 定义约束列表 G，包含多个元组，每个元组中是两个多项式的对，形式为 (Poly(a, x), Poly(b, x))
    G = [(Poly(2*x**3 + 3*x + 1, x), Poly(x**2 - 1, x)),
         (Poly(1, x), Poly(x - 1, x)),
         (Poly(1, x), Poly(x + 1, x))]
    
    # 使用 assert 语句验证 prde_linear_constraints 函数的返回结果是否符合预期
    assert prde_linear_constraints(Poly(1, x), Poly(0, x), G, DE) == \
        ((Poly(2*x, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(0, x, domain='QQ')),
         Matrix([[1, 1, -1], [5, 1, 1]], x))
    
    # 修改约束列表 G，包含多个元组，每个元组中是两个多项式的对，形式为 (Poly(t, t), Poly(1, t))
    G = [(Poly(t, t), Poly(1, t)),
         (Poly(t**2, t), Poly(1, t)),
         (Poly(t**3, t), Poly(1, t))]
    
    # 更新差分扩展对象 DE，其扩展包括导数 'D' 对应的多项式 Poly(1, x) 和 Poly(t, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    
    # 使用 assert 语句验证 prde_linear_constraints 函数的返回结果是否符合预期
    assert prde_linear_constraints(Poly(t + 1, t), Poly(t**2, t), G, DE) == \
        ((Poly(t, t, domain='QQ'), Poly(t**2, t, domain='QQ'), Poly(t**3, t, domain='QQ')),
         Matrix(0, 3, [], t))
    
    # 修改约束列表 G，包含多个元组，每个元组中是两个多项式的对，形式为 (Poly(2*x, t), Poly(t, t)) 和 (Poly(-x, t), Poly(t, t))
    G = [(Poly(2*x, t), Poly(t, t)),
         (Poly(-x, t), Poly(t, t))]
    
    # 更新差分扩展对象 DE，其扩展包括导数 'D' 对应的多项式 Poly(1, x) 和 Poly(1/x, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    
    # 使用 assert 语句验证 prde_linear_constraints 函数的返回结果是否符合预期
    assert prde_linear_constraints(Poly(1, t), Poly(0, t), G, DE) == \
        ((Poly(0, t, domain='QQ[x]'), Poly(0, t, domain='QQ[x]')),
         Matrix([[2*x, -x]], t))


# 定义测试函数 `test_constant_system`，用于测试 `constant_system` 函数
def test_constant_system():
    # 创建矩阵 A，其中元素是多项式表达式，定义在环 QQ.frac_field(x)[t] 上
    A = Matrix([[-(x + 3)/(x - 1), (x + 1)/(x - 1), 1],
                [-x - 3, x + 1, x - 1],
                [2*(x + 3)/(x - 1), 0, 0]], t)
    
    # 创建向量 u，其中元素是多项式表达式，定义在环 QQ.frac_field(x)[t] 上
    u = Matrix([[(x + 1)/(x - 1)], [x + 1], [0]], t)
    
    # 创建差分扩展对象 DE，其扩展包括导数 'D' 对应的多项式 Poly(1, x)
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    
    # 创建环 R，表示矩阵元素的环，定义为 QQ.frac_field(x)[t]
    R = QQ.frac_field(x)[t]
    
    # 使用 assert 语句验证 constant_system 函数的返回结果是否符合预期
    assert constant_system(A, u, DE) == \
        (Matrix([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0],
                 [0, 0, 1]], ring=R), Matrix([0, 1, 0, 0], ring=R))


# 定义测试函数 `test_prde_spde`，用于测试 `prde_spde` 函数
def test_prde_spde():
    # 创建差分表达式列表 D，其中每个元素是多项式 Poly(x, t) 或 Poly(-x*t, t)
    D = [Poly(x, t), Poly(-x*t, t)]
    
    # 创建差分扩展对象 DE，其扩展包括导数 'D' 对应的多项式 Poly(1, x) 和 Poly(1/x, t)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    
    # 使用 assert 语句验证 prde_spde 函数的返回结果是否符合预期
    # TODO: when bound_degree() can handle this, test degree bound from that too
    assert prde_spde(Poly(t, t), Poly(-1/x, t), D, n, DE) == \
        (Poly(t, t), Poly(0, t, domain='ZZ(x)'),
         [Poly(2*x, t, domain='ZZ(x)'), Poly(-x, t, domain='ZZ(x)')],
         [Poly(-x**2, t, domain='ZZ(x)'), Poly(0, t, domain='ZZ(x)')], n - 1)


# 定义测试函数 `test_prde_no_cancel`，用于测试 `prde_no_cancel_b_large` 函数
def test_prde_no_cancel():
    # 创建差分扩展对象 DE，其扩展包括导数 'D' 对应的多项式 Poly(1, x)
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    
    # 使用 assert 语句验证 prde_no_cancel_b_large 函数的返回结果是否符合预期
    assert prde_no_cancel_b_large(Poly(1, x), [Poly(x**2, x), Poly(1, x)], 2, DE) == \
        ([Poly(x**2 - 2*x + 2, x), Poly(1, x)], Matrix([[1, 0, -1, 0],
                                                       [0, 1, 0, -1]], x))
    
    # 使用 assert 语句验证 prde_no_cancel_b_large 函数的返回结果是否符合预期
    assert prde_no_cancel_b_large(Poly(1, x), [Poly(x**3, x), Poly(1, x)], 3, DE) == \
        ([Poly(x**3 - 3*x**2 + 6*x - 6, x), Poly(1, x)], Matrix([[1, 0, -1, 0],
                                                                [0, 1, 0, -1]], x))
    assert prde_no_cancel_b_large(Poly(x, x), [Poly(x**2, x), Poly(1, x)], 1, DE) == \
        ([Poly(x, x, domain='ZZ'), Poly(0, x, domain='ZZ')], Matrix([[1, -1,  0,  0],
                                                                    [1,  0, -1,  0],
                                                                    [0,  1,  0, -1]], x))
    # 对大型 b 的断言测试
    # XXX: 是否有一个 D.degree() > 2 的更好的单项式的例子？
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t**3 + 1, t)]})

    # 我原来的 q 是 t**4 + t + 1，但这个解意味着 q == t**4 (c1 = 4)，并且原始 q 的一些 ci 等于 0。
    G = [Poly(t**6, t), Poly(x*t**5, t), Poly(t**3, t), Poly(x*t**2, t), Poly(1 + x, t)]
    R = QQ.frac_field(x)[t]
    assert prde_no_cancel_b_small(Poly(x*t, t), G, 4, DE) == \
        ([Poly(t**4/4 - x/12*t**3 + x**2/24*t**2 + (Rational(-11, 12) - x**3/24)*t + x/24, t),
        Poly(x/3*t**3 - x**2/6*t**2 + (Rational(-1, 3) + x**3/6)*t - x/6, t), Poly(t, t),
        Poly(0, t), Poly(0, t)], Matrix([[1, 0,              -1, 0, 0,  0,  0,  0,  0,  0],
                                         [0, 1, Rational(-1, 4), 0, 0,  0,  0,  0,  0,  0],
                                         [0, 0,               0, 0, 0,  0,  0,  0,  0,  0],
                                         [0, 0,               0, 1, 0,  0,  0,  0,  0,  0],
                                         [0, 0,               0, 0, 1,  0,  0,  0,  0,  0],
                                         [1, 0,               0, 0, 0, -1,  0,  0,  0,  0],
                                         [0, 1,               0, 0, 0,  0, -1,  0,  0,  0],
                                         [0, 0,               1, 0, 0,  0,  0, -1,  0,  0],
                                         [0, 0,               0, 1, 0,  0,  0,  0, -1,  0],
                                         [0, 0,               0, 0, 1,  0,  0,  0,  0, -1]], ring=R))

    # TODO: 添加 deg(b) <= 0 且 b 较小的测试
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1 + t**2, t)]})
    b = Poly(-1/x**2, t, field=True)  # deg(b) == 0
    q = [Poly(x**i*t**j, t, field=True) for i in range(2) for j in range(3)]
    h, A = prde_no_cancel_b_small(b, q, 3, DE)
    V = A.nullspace()
    R = QQ.frac_field(x)[t]
    assert len(V) == 1
    assert V[0] == Matrix([Rational(-1, 2), 0, 0, 1, 0, 0]*3, ring=R)
    assert (Matrix([h])*V[0][6:, :])[0] == Poly(x**2/2, t, domain='QQ(x)')
    assert (Matrix([q])*V[0][:6, :])[0] == Poly(x - S.Half, t, domain='QQ(x)')
def test_prde_cancel_liouvillian():
    ### 1. case == 'primitive'
    # 当集成 f = log(x) - log(x - 1) 时使用
    # 不是从 'the' book 中获取的
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    # 创建多项式 p0
    p0 = Poly(0, t, field=True)
    # 创建多项式 p1
    p1 = Poly((x - 1)*t, t, domain='ZZ(x)')
    # 创建多项式 p2
    p2 = Poly(x - 1, t, domain='ZZ(x)')
    # 创建多项式 p3
    p3 = Poly(-x**2 + x, t, domain='ZZ(x)')
    # 计算 prde_cancel_liouvillian 的结果 h 和 A
    h, A = prde_cancel_liouvillian(Poly(-1/(x - 1), t), [Poly(-x + 1, t), Poly(1, t)], 1, DE)
    # 计算 A 的零空间
    V = A.nullspace()
    # 断言 h 的值
    assert h == [p0, p0, p1, p0, p0, p0, p0, p0, p0, p0, p2, p3, p0, p0, p0, p0]
    # 断言 A 的秩
    assert A.rank() == 16
    # 断言 (Matrix([h])*V[0][:16, :]) 的结果
    assert (Matrix([h])*V[0][:16, :]) == Matrix([[Poly(0, t, domain='QQ(x)')]])

    ### 2. case == 'exp'
    # 当集成 log(x/exp(x) + 1) 时使用
    # 不是从 book 中获取的
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t, t)]})
    # 断言 prde_cancel_liouvillian 的结果
    assert prde_cancel_liouvillian(Poly(0, t, domain='QQ[x]'), [Poly(1, t, domain='QQ(x)')], 0, DE) == \
            ([Poly(1, t, domain='QQ'), Poly(x, t, domain='ZZ(x)')], Matrix([[-1, 0, 1]], DE.t))


def test_param_poly_rischDE():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    # 创建多项式 a
    a = Poly(x**2 - x, x, field=True)
    # 创建多项式 b
    b = Poly(1, x, field=True)
    # 创建多项式列表 q
    q = [Poly(x, x, field=True), Poly(x**2, x, field=True)]
    # 计算 param_poly_rischDE 的结果 h 和 A
    h, A = param_poly_rischDE(a, b, q, 3, DE)

    # 断言 A 的零空间
    assert A.nullspace() == [Matrix([0, 1, 1, 1], DE.t)]  # c1, c2, d1, d2
    # 断言 h[0] + h[1] 的结果
    assert h[0] + h[1] == Poly(x, x, domain='QQ')
    # 断言 a*derivation(p, DE) + b*p 的结果
    assert a*derivation(p, DE) + b*p == Poly(x**2 - 5*x + 3, x, domain='QQ')


def test_param_rischDE():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    # 创建多项式 p1 和 px
    p1, px = Poly(1, x, field=True), Poly(x, x, field=True)
    # 创建列表 G
    G = [(p1, px), (p1, p1), (px, p1)]  # [1/x, 1, x]
    # 计算 param_rischDE 的结果 h 和 A
    h, A = param_rischDE(-p1, Poly(x**2, x, field=True), G, DE)
    # 断言 h 的长度
    assert len(h) == 3
    # 计算 p
    p = [hi[0].as_expr()/hi[1].as_expr() for hi in h]
    # 计算 A 的零空间
    V = A.nullspace()
    # 断言 V[0] 的值
    assert V[0] == Matrix([-1, 1, 0, -1, 1, 0], DE.t)
    # 计算 y
    y = -p[0] + p[1] + 0*p[2]  # x
    # 断言 Dy + f*y 的结果
    assert y.diff(x) - y/x**2 == 1 - 1/x  # Dy + f*y == -G0 + G1 + 0*G2

    # 计算 f = log(log(x + exp(x)))
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t, t)]})
    # 创建列表 G
    G = [(Poly(t + x, t, domain='ZZ(x)'), Poly(1, t, domain='QQ')), (Poly(0, t, domain='QQ'), Poly(1, t, domain='QQ'))]
    # 计算 param_rischDE 的结果 h 和 A
    h, A = param_rischDE(Poly(-t - 1, t, field=True), Poly(t + x, t, field=True), G, DE)
    # 断言 h 的长度
    assert len(h) == 5
    # 创建列表 p，包含 hi[0].as_expr() / hi[1].as_expr() 的结果，其中 hi 是 h 中的每个元素
    p = [hi[0].as_expr()/hi[1].as_expr() for hi in h]
    # 计算矩阵 A 的零空间
    V = A.nullspace()
    # 断言 V 的长度为 3
    assert len(V) == 3
    # 断言 V 的第一个元素等于给定的 Matrix 对象
    assert V[0] == Matrix([0, 0, 0, 0, 1, 0, 0], DE.t)
    # 计算 y，为 p[2] 乘以 1，其它项乘以 0
    y = 0*p[0] + 0*p[1] + 1*p[2] + 0*p[3] + 0*p[4]
    # 断言 y 对 t 的导数减去 y 除以 (t + x) 的结果等于 0
    assert y.diff(t) - y/(t + x) == 0   # Dy + f*y = 0*G0 + 0*G1
# 定义一个测试函数，用于测试 limited_integrate_reduce 函数的功能
def test_limited_integrate_reduce():
    # 创建一个微分扩展对象 DE，包含扩展 {'D': [Poly(1, x), Poly(1/x, t)]}
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    # 断言调用 limited_integrate_reduce 函数返回的结果符合预期
    assert limited_integrate_reduce(
        Poly(x, t),  # 第一个参数是 Poly(x, t)
        Poly(t**2, t),  # 第二个参数是 Poly(t**2, t)
        [(Poly(x, t), Poly(t, t))],  # 第三个参数是一个包含元组的列表
        DE  # 微分扩展对象 DE
    ) == (
        Poly(t, t),  # 返回结果的第一个元素是 Poly(t, t)
        Poly(-1/x, t),  # 返回结果的第二个元素是 Poly(-1/x, t)
        Poly(t, t),  # 返回结果的第三个元素是 Poly(t, t)
        1,  # 返回结果的第四个元素是整数 1
        (Poly(x, t), Poly(1, t, domain='ZZ[x]')),  # 返回结果的第五个元素是一个元组
        [(Poly(-x*t, t), Poly(1, t, domain='ZZ[x]'))]  # 返回结果的最后一个元素是列表
    )


# 定义一个测试函数，用于测试 limited_integrate 函数的功能
def test_limited_integrate():
    # 创建一个微分扩展对象 DE，包含扩展 {'D': [Poly(1, x)]}
    DE = DifferentialExtension(extension={'D': [Poly(1, x)]})
    # 初始化 G 变量为 [(Poly(x, x), Poly(x + 1, x))]
    G = [(Poly(x, x), Poly(x + 1, x))]
    # 断言调用 limited_integrate 函数返回的结果符合预期
    assert limited_integrate(
        Poly(-(1 + x + 5*x**2 - 3*x**3), x),  # 第一个参数是 Poly(-(1 + x + 5*x**2 - 3*x**3), x)
        Poly(1 - x - x**2 + x**3, x),  # 第二个参数是 Poly(1 - x - x**2 + x**3, x)
        G,  # 第三个参数是变量 G
        DE  # 微分扩展对象 DE
    ) == (
        (Poly(x**2 - x + 2, x), Poly(x - 1, x, domain='QQ')),  # 返回结果的第一个元素是一个元组
        [2]  # 返回结果的第二个元素是列表
    )

    # 将 G 重新赋值为 [(Poly(1, x), Poly(x, x))]
    G = [(Poly(1, x), Poly(x, x))]
    # 断言调用 limited_integrate 函数返回的结果符合预期
    assert limited_integrate(
        Poly(5*x**2, x),  # 第一个参数是 Poly(5*x**2, x)
        Poly(3, x),  # 第二个参数是 Poly(3, x)
        G,  # 第三个参数是变量 G
        DE  # 微分扩展对象 DE
    ) == (
        (Poly(5*x**3/9, x), Poly(1, x, domain='QQ')),  # 返回结果的第一个元素是一个元组
        [0]  # 返回结果的第二个元素是列表
    )


# 定义一个测试函数，用于测试 is_log_deriv_k_t_radical 函数的功能
def test_is_log_deriv_k_t_radical():
    # 创建一个微分扩展对象 DE，包含扩展 {'D': [Poly(1, x)], 'exts': [None], 'extargs': [None]}
    DE = DifferentialExtension(extension={'D': [Poly(1, x)], 'exts': [None], 'extargs': [None]})
    # 断言调用 is_log_deriv_k_t_radical 函数返回的结果符合预期
    assert is_log_deriv_k_t_radical(
        Poly(2*x, x),  # 第一个参数是 Poly(2*x, x)
        Poly(1, x),  # 第二个参数是 Poly(1, x)
        DE  # 微分扩展对象 DE
    ) is None

    # 创建一个微分扩展对象 DE，包含扩展 {'D': [Poly(1, x), Poly(2*t1, t1), Poly(1/x, t2)], 'exts': [None, 'exp', 'log'], 'extargs': [None, 2*x, x]}
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(2*t1, t1), Poly(1/x, t2)], 'exts': [None, 'exp', 'log'], 'extargs': [None, 2*x, x]})
    # 断言调用 is_log_deriv_k_t_radical 函数返回的结果符合预期
    assert is_log_deriv_k_t_radical(
        Poly(x + t2/2, t2),  # 第一个参数是 Poly(x + t2/2, t2)
        Poly(1, t2),  # 第二个参数是 Poly(1, t2)
        DE  # 微分扩展对象 DE
    ) == (
        [(t1, 1), (x, 1)],  # 返回结果的第一个元素是一个列表
        t1*x,  # 返回结果的第二个元素是 t1*x
        2,  # 返回结果的第三个元素是整数 2
        0  # 返回结果的第四个元素是整数 0
    )

    # 创建一个微分扩展对象 DE，包含扩展 {'D': [Poly(1, x), Poly(t0, t0), Poly(1/x, t)], 'exts': [None, 'exp', 'log'], 'extargs': [None, x, x]}
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t0, t0), Poly(1/x, t)], 'exts': [None, 'exp', 'log'], 'extargs': [None, x, x]})
    # 断言调用 is_log_deriv_k_t_radical 函数返回的结果符合预期
    assert is_log_deriv_k_t_radical(
        Poly(x + t/2 + 3, t),  # 第一个参数是 Poly(x + t/2 + 3, t)
        Poly(1, t),  # 第二个参数是 Poly(1, t)
        DE  # 微分扩展对象 DE
    ) == (
        [(t0, 2), (x, 1)],  # 返回结果的第一个元素是一个列表
        x*t0**2,  # 返回结果的第二个元素是 x*t0**2
        2,  # 返回结果的第三个元素是整数 2
        3  # 返回结果的第四个元素是整数 3
    )


# 定义一个测试函数，用于测试 is_deriv_k 函数的功能
def test_is_deriv_k():
    # 创建一个微分扩展对象 DE，包含扩展 {'D': [Poly(1, x), Poly(1/x, t1), Poly(1/(x + 1), t2)], 'exts': [None, 'log', 'log'], 'extargs': [None, x, x + 1]}
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t1), Poly(1/(x + 1), t2)], 'exts': [None, 'log', 'log'], 'extargs': [None, x, x + 1]})
    # 断言调用 is_deriv_k 函数返回的结果符合预期
    assert is_deriv_k(
        Poly(2*x**2 + 2*x, t2),  # 第一个参数是 Poly(2*x**2 + 2*x, t2)
        Poly(1, t2),  # 第二个参数是 Poly(1, t2)
        DE  # 微分扩展对象 DE
    ) == (
        [(t1, 1), (t2, 1)],  # 返回结果的第一个元素是一个列表
        t1 + t2,  # 返回结果的第二个元素是 t1 + t2
        2  # 返回结果的第三个元素是整数 2
    )

    # 创建一个微分扩展对象 DE，包
    # 断言语句，用于检查 is_deriv_k 函数的返回结果是否符合预期
    assert is_deriv_k(Poly(1, t), Poly(x, t), DE) == ([(t, 1)], t, 1)
# 定义测试函数，用于测试是否在给定域内的对数导数是 k*t 根式的
def test_is_log_deriv_k_t_radical_in_field():
    # 定义一个微分扩展对象 DE，包含 D 操作符
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    # 断言函数 is_log_deriv_k_t_radical_in_field 的返回结果是否为预期值
    assert is_log_deriv_k_t_radical_in_field(Poly(5*t + 1, t), Poly(2*t*x, t), DE) == \
        (2, t*x**5)
    # 断言函数 is_log_deriv_k_t_radical_in_field 的返回结果是否为预期值
    assert is_log_deriv_k_t_radical_in_field(Poly(2 + 3*t, t), Poly(5*x*t, t), DE) == \
        (5, x**3*t**2)

    # 重新定义微分扩展对象 DE，包含不同的 D 操作符
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(-t/x**2, t)]})
    # 断言函数 is_log_deriv_k_t_radical_in_field 的返回结果是否为预期值
    assert is_log_deriv_k_t_radical_in_field(Poly(-(1 + 2*t), t),
    Poly(2*x**2 + 2*x**2*t, t), DE) == \
        (2, t + t**2)
    # 断言函数 is_log_deriv_k_t_radical_in_field 的返回结果是否为预期值
    assert is_log_deriv_k_t_radical_in_field(Poly(-1, t), Poly(x**2, t), DE) == \
        (1, t)
    # 断言函数 is_log_deriv_k_t_radical_in_field 的返回结果是否为预期值
    assert is_log_deriv_k_t_radical_in_field(Poly(1, t), Poly(2*x**2, t), DE) == \
        (2, 1/t)


# 定义测试函数，用于测试参数化对数导数的推测函数
def test_parametric_log_deriv():
    # 定义一个微分扩展对象 DE，包含 D 操作符
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(1/x, t)]})
    # 断言函数 parametric_log_deriv_heu 的返回结果是否为预期值
    assert parametric_log_deriv_heu(Poly(5*t**2 + t - 6, t), Poly(2*x*t**2, t),
    Poly(-1, t), Poly(x*t**2, t), DE) == \
        (2, 6, t*x**5)
```