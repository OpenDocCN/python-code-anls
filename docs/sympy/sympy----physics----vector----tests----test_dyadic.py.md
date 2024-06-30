# `D:\src\scipysrc\sympy\sympy\physics\vector\tests\test_dyadic.py`

```
from sympy.core.numbers import (Float, pi)  # 导入必要的模块和函数
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, outer
from sympy.physics.vector.dyadic import _check_dyadic
from sympy.testing.pytest import raises

A = ReferenceFrame('A')  # 创建一个参考框架对象 A


def test_dyadic():  # 定义测试函数 test_dyadic
    d1 = A.x | A.x  # 创建 dyadic 对象 d1，表示 A.x 在 A.x 方向上的外积
    d2 = A.y | A.y  # 创建 dyadic 对象 d2，表示 A.y 在 A.y 方向上的外积
    d3 = A.x | A.y  # 创建 dyadic 对象 d3，表示 A.x 在 A.y 方向上的外积

    assert d1 * 0 == 0  # 断言：dyadic 对象 d1 乘以 0 等于 0
    assert d1 != 0  # 断言：dyadic 对象 d1 不等于 0
    assert d1 * 2 == 2 * A.x | A.x  # 断言：dyadic 对象 d1 乘以 2 等于 2 乘以 A.x 在 A.x 方向上的外积
    assert d1 / 2. == 0.5 * d1  # 断言：dyadic 对象 d1 除以 2. 等于 0.5 倍的 d1
    assert d1 & (0 * d1) == 0  # 断言：dyadic 对象 d1 与 (0 倍 d1) 的内积等于 0
    assert d1 & d2 == 0  # 断言：dyadic 对象 d1 与 d2 的内积等于 0
    assert d1 & A.x == A.x  # 断言：dyadic 对象 d1 与 A.x 的内积等于 A.x
    assert d1 ^ A.x == 0  # 断言：dyadic 对象 d1 与 A.x 的外积等于 0
    assert d1 ^ A.y == A.x | A.z  # 断言：dyadic 对象 d1 与 A.y 的外积等于 A.x 在 A.z 方向上的外积
    assert d1 ^ A.z == - A.x | A.y  # 断言：dyadic 对象 d1 与 A.z 的外积等于 -A.x 在 A.y 方向上的外积
    assert d2 ^ A.x == - A.y | A.z  # 断言：dyadic 对象 d2 与 A.x 的外积等于 -A.y 在 A.z 方向上的外积
    assert A.x ^ d1 == 0  # 断言：A.x 与 dyadic 对象 d1 的外积等于 0
    assert A.y ^ d1 == - A.z | A.x  # 断言：A.y 与 dyadic 对象 d1 的外积等于 -A.z 在 A.x 方向上的外积
    assert A.z ^ d1 == A.y | A.x  # 断言：A.z 与 dyadic 对象 d1 的外积等于 A.y 在 A.x 方向上的外积
    assert A.x & d1 == A.x  # 断言：A.x 与 dyadic 对象 d1 的内积等于 A.x
    assert A.y & d1 == 0  # 断言：A.y 与 dyadic 对象 d1 的内积等于 0
    assert A.y & d2 == A.y  # 断言：A.y 与 dyadic 对象 d2 的内积等于 A.y
    assert d1 & d3 == A.x | A.y  # 断言：dyadic 对象 d1 与 dyadic 对象 d3 的内积等于 A.x 在 A.y 方向上的外积
    assert d3 & d1 == 0  # 断言：dyadic 对象 d3 与 dyadic 对象 d1 的内积等于 0
    assert d1.dt(A) == 0  # 断言：dyadic 对象 d1 关于参考框架 A 的时间导数等于 0

    q = dynamicsymbols('q')  # 创建动力学符号 q
    qd = dynamicsymbols('q', 1)  # 创建动力学符号 q 的一阶导数
    B = A.orientnew('B', 'Axis', [q, A.z])  # 在参考框架 A 上创建一个新的参考框架 B，绕 A.z 轴旋转 q 角度

    assert d1.express(B) == d1.express(B, B)  # 断言：dyadic 对象 d1 在参考框架 B 上的表达式等于在 B 参考框架自身上的表达式
    assert d1.express(B) == ((cos(q)**2) * (B.x | B.x) + (-sin(q) * cos(q)) *
            (B.x | B.y) + (-sin(q) * cos(q)) * (B.y | B.x) + (sin(q)**2) *
            (B.y | B.y))  # 断言：dyadic 对象 d1 在参考框架 B 上的表达式

    assert d1.express(B, A) == (cos(q)) * (B.x | A.x) + (-sin(q)) * (B.y | A.x)  # 断言：dyadic 对象 d1 在 B 到 A 的表达式
    assert d1.express(A, B) == (cos(q)) * (A.x | B.x) + (-sin(q)) * (A.x | B.y)  # 断言：dyadic 对象 d1 在 A 到 B 的表达式

    assert d1.dt(B) == (-qd) * (A.y | A.x) + (-qd) * (A.x | A.y)  # 断言：dyadic 对象 d1 关于参考框架 B 的时间导数

    assert d1.to_matrix(A) == Matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]])  # 断言：dyadic 对象 d1 在参考框架 A 下的矩阵表示
    assert d1.to_matrix(A, B) == Matrix([[cos(q), -sin(q), 0],
                                         [0, 0, 0],
                                         [0, 0, 0]])  # 断言：dyadic 对象 d1 在参考框架 A 到 B 的矩阵表示

    assert d3.to_matrix(A) == Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])  # 断言：dyadic 对象 d3 在参考框架 A 下的矩阵表示

    a, b, c, d, e, f = symbols('a, b, c, d, e, f')  # 创建符号变量 a, b, c, d, e, f
    v1 = a * A.x + b * A.y + c * A.z  # 创建向量 v1
    v2 = d * A.x + e * A.y + f * A.z  # 创建向量 v2
    d4 = v1.outer(v2)  # 创建 v1 和 v2 的外积 dyadic 对象 d4

    assert d4.to_matrix(A) == Matrix([[a * d, a * e, a * f],
                                      [b * d, b * e, b * f],
                                      [c * d, c * e, c * f]])  # 断言：dyadic 对象 d4 在参考框架 A 下的矩阵表示

    d5 = v1.outer(v1)  # 创建 v1 和 v1 的外积 dyadic 对象 d5
    C = A.orientnew('C', 'Axis', [q, A.x])  # 在参考框架 A 上创建一个新的参考框架 C，绕 A.x 轴旋转 q 角度

    for expected, actual in zip(C.dcm(A) * d5.to_matrix(A) * C.dcm(A).T,
                                d5.to_matrix(C)):
        assert (expected - actual).
    # 断言检查公式是否成立，左右两侧的表达式应当相等
    assert (N.x & test2 & N.x) == (A**2 * s**4 / (4 * pi * k * m**3))
    
    # 计算数学表达式 test3，并简化其结果
    test3 = ((4 + 4 * x - 2 * (2 + 2 * x)) / (2 + 2 * x)) * dy
    test3 = test3.simplify()
    # 断言检查结果是否为零
    assert (N.x & test3 & N.x) == 0
    
    # 计算数学表达式 test4，并简化其结果
    test4 = ((-4 * x * y**2 - 2 * y**3 - 2 * x**2 * y) / (x + y)**2) * dy
    test4 = test4.simplify()
    # 断言检查结果是否为 -2 * y
    assert (N.x & test4 & N.x) == -2 * y
# 定义一个测试函数，用于测试 dyadic_subs 函数
def test_dyadic_subs():
    # 创建一个参考框架 N
    N = ReferenceFrame('N')
    # 创建一个符号 s
    s = symbols('s')
    # 创建一个 dyadic 表达式 a，使用 s 乘以 N.x 与 N.x 的外积
    a = s*(N.x | N.x)
    # 断言当 s 替换为 2 时，a 的值为 2 乘以 (N.x | N.x)
    assert a.subs({s: 2}) == 2*(N.x | N.x)


# 定义一个测试函数，用于测试 check_dyadic 函数是否会引发 TypeError 异常
def test_check_dyadic():
    # 断言调用 _check_dyadic(0) 会引发 TypeError 异常
    raises(TypeError, lambda: _check_dyadic(0))


# 定义一个测试函数，用于测试 dyadic_evalf 函数
def test_dyadic_evalf():
    # 创建一个参考框架 N
    N = ReferenceFrame('N')
    # 创建一个 dyadic 表达式 a，使用 pi 乘以 N.x 与 N.x 的外积
    a = pi * (N.x | N.x)
    # 断言对 a 执行 evalf(3) 的结果与 Float('3.1416', 3) 乘以 (N.x | N.x) 相等
    assert a.evalf(3) == Float('3.1416', 3) * (N.x | N.x)
    # 创建一个符号 s
    s = symbols('s')
    # 创建一个 dyadic 表达式 a，使用 5、s 和 pi 乘以 N.x 与 N.x 的外积
    a = 5 * s * pi * (N.x | N.x)
    # 断言对 a 执行 evalf(2) 的结果与 Float('5', 2)、Float('3.1416', 2)、s、(N.x | N.x) 的乘积相等
    assert a.evalf(2) == Float('5', 2) * Float('3.1416', 2) * s * (N.x | N.x)
    # 断言对 a 执行 evalf(9, subs={s: 5.124}) 的结果与 Float('80.48760378', 9) 乘以 (N.x | N.x) 相等
    assert a.evalf(9, subs={s: 5.124}) == Float('80.48760378', 9) * (N.x | N.x)


# 定义一个测试函数，用于测试 dyadic_xreplace 函数
def test_dyadic_xreplace():
    # 创建符号变量 x, y, z
    x, y, z = symbols('x y z')
    # 创建一个参考框架 N
    N = ReferenceFrame('N')
    # 创建一个 dyadic 表达式 D，使用 N.x 与 N.x 的外积
    D = outer(N.x, N.x)
    # 创建一个向量 v，使用 x*y 乘以 D
    v = x*y * D
    # 断言对 v 执行 xreplace({x: cos(x)}) 的结果与 cos(x)*y 乘以 D 相等
    assert v.xreplace({x : cos(x)}) == cos(x)*y * D
    # 断言对 v 执行 xreplace({x*y: pi}) 的结果与 pi 乘以 D 相等
    assert v.xreplace({x*y : pi}) == pi * D
    # 创建一个 dyadic 表达式 v，使用 (x*y)**z 乘以 D
    v = (x*y)**z * D
    # 断言对 v 执行 xreplace({(x*y)**z: 1}) 的结果与 D 相等
    assert v.xreplace({(x*y)**z : 1}) == D
    # 断言对 v 执行 xreplace({x: 1, z: 0}) 的结果与 D 相等
    assert v.xreplace({x:1, z:0}) == D
    # 断言调用 v.xreplace() 会引发 TypeError 异常
    raises(TypeError, lambda: v.xreplace())
    # 断言调用 v.xreplace([x, y]) 会引发 TypeError 异常
    raises(TypeError, lambda: v.xreplace([x, y]))
```