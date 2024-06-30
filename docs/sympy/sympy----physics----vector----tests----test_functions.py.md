# `D:\src\scipysrc\sympy\sympy\physics\vector\tests\test_functions.py`

```
# 导入必要的符号和函数库
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.physics.vector import Dyadic, Point, ReferenceFrame, Vector
from sympy.physics.vector.functions import (cross, dot, express,
                                            time_derivative,
                                            kinematic_equations, outer,
                                            partial_velocity,
                                            get_motion_params, dynamicsymbols)
from sympy.simplify import trigsimp
from sympy.testing.pytest import raises

# 定义符号变量
q1, q2, q3, q4, q5 = symbols('q1 q2 q3 q4 q5')

# 定义参考框架 N
N = ReferenceFrame('N')

# 在 N 框架上定义 A 框架，绕 z 轴旋转 q1 角度
A = N.orientnew('A', 'Axis', [q1, N.z])

# 在 A 框架上定义 B 框架，绕 A 框架的 x 轴旋转 q2 角度
B = A.orientnew('B', 'Axis', [q2, A.x])

# 在 B 框架上定义 C 框架，绕 B 框架的 y 轴旋转 q3 角度
C = B.orientnew('C', 'Axis', [q3, B.y])


# 定义测试函数 test_dot，测试点积函数 dot 的几种情况
def test_dot():
    assert dot(A.x, A.x) == 1  # 测试 A.x 与 A.x 的点积结果是否为 1
    assert dot(A.x, A.y) == 0  # 测试 A.x 与 A.y 的点积结果是否为 0
    assert dot(A.x, A.z) == 0  # 测试 A.x 与 A.z 的点积结果是否为 0

    assert dot(A.y, A.x) == 0  # 测试 A.y 与 A.x 的点积结果是否为 0
    assert dot(A.y, A.y) == 1  # 测试 A.y 与 A.y 的点积结果是否为 1
    assert dot(A.y, A.z) == 0  # 测试 A.y 与 A.z 的点积结果是否为 0

    assert dot(A.z, A.x) == 0  # 测试 A.z 与 A.x 的点积结果是否为 0
    assert dot(A.z, A.y) == 0  # 测试 A.z 与 A.y 的点积结果是否为 0
    assert dot(A.z, A.z) == 1  # 测试 A.z 与 A.z 的点积结果是否为 1


# 定义测试函数 test_dot_different_frames，测试不同框架之间的点积结果
def test_dot_different_frames():
    assert dot(N.x, A.x) == cos(q1)        # 测试 N.x 与 A.x 的点积结果是否为 cos(q1)
    assert dot(N.x, A.y) == -sin(q1)       # 测试 N.x 与 A.y 的点积结果是否为 -sin(q1)
    assert dot(N.x, A.z) == 0              # 测试 N.x 与 A.z 的点积结果是否为 0
    assert dot(N.y, A.x) == sin(q1)        # 测试 N.y 与 A.x 的点积结果是否为 sin(q1)
    assert dot(N.y, A.y) == cos(q1)        # 测试 N.y 与 A.y 的点积结果是否为 cos(q1)
    assert dot(N.y, A.z) == 0              # 测试 N.y 与 A.z 的点积结果是否为 0
    assert dot(N.z, A.x) == 0              # 测试 N.z 与 A.x 的点积结果是否为 0
    assert dot(N.z, A.y) == 0              # 测试 N.z 与 A.y 的点积结果是否为 0
    assert dot(N.z, A.z) == 1              # 测试 N.z 与 A.z 的点积结果是否为 1

    assert trigsimp(dot(N.x, A.x + A.y)) == sqrt(2)*cos(q1 + pi/4)   # 测试复合表达式的点积结果经过化简
    assert trigsimp(dot(A.x + A.y, N.x)) == trigsimp(dot(N.x, A.x + A.y))  # 测试点积的交换律

    assert dot(A.x, C.x) == cos(q3)        # 测试 A.x 与 C.x 的点积结果是否为 cos(q3)
    assert dot(A.x, C.y) == 0              # 测试 A.x 与 C.y 的点积结果是否为 0
    assert dot(A.x, C.z) == sin(q3)        # 测试 A.x 与 C.z 的点积结果是否为 sin(q3)
    assert dot(A.y, C.x) == sin(q2)*sin(q3)  # 测试 A.y 与 C.x 的点积结果是否为 sin(q2)*sin(q3)
    assert dot(A.y, C.y) == cos(q2)        # 测试 A.y 与 C.y 的点积结果是否为 cos(q2)
    assert dot(A.y, C.z) == -sin(q2)*cos(q3)  # 测试 A.y 与 C.z 的点积结果是否为 -sin(q2)*cos(q3)
    assert dot(A.z, C.x) == -cos(q2)*sin(q3)  # 测试 A.z 与 C.x 的点积结果是否为 -cos(q2)*sin(q3)
    assert dot(A.z, C.y) == sin(q2)        # 测试 A.z 与 C.y 的点积结果是否为 sin(q2)
    assert dot(A.z, C.z) == cos(q2)*cos(q3)  # 测试 A.z 与 C.z 的点积结果是否为 cos(q2)*cos(q3)


# 定义测试函数 test_cross，测试叉积函数 cross 的几种情况
def test_cross():
    assert cross(A.x, A.x) == 0            # 测试 A.x 叉乘 A.x 的结果是否为 0
    assert cross(A.x, A.y) == A.z          # 测试 A.x 叉乘 A.y 的结果是否为 A.z
    assert cross(A.x, A.z) == -A.y         # 测试 A.x 叉乘 A.z 的结果是否为 -A.y

    assert cross(A.y, A.x) == -A.z         # 测试 A.y 叉乘 A.x 的结果是否为 -A.z
    assert cross(A.y, A.y) == 0            # 测试 A.y 叉乘 A.y 的结果是否为 0
    assert cross(A.y, A.z) == A.x          # 测试 A.y 叉乘 A.z 的结果是否为 A.x

    assert cross(A.z, A.x) == A.y          # 测试 A.z 叉乘 A.x 的结果是否为 A.y
    assert cross(A.z, A.y) == -A.x         # 测试 A.z 叉乘 A.y 的结果是否为 -A.x
    assert cross(A.z, A.z) == 0            # 测试 A.z 叉乘 A.z 的结果是否为 0


# 定义测试函数 test_cross_different_frames，测试不同框架之间的叉积结果
def test_cross_different_frames():
    assert cross(N.x, A.x) == sin(q1)*A.z  # 测试 N.x 叉乘 A.x 的结果是否为 sin(q1)*A.z
    assert cross(N.x, A.y) == cos(q1)*A.z  # 测试 N.x 叉乘 A.y 的结果是否为 cos(q1)*A.z
    assert cross(N.x, A.z) == -sin(q1)*A.x - cos(q1)*A.y  # 测试 N.x 叉乘 A.z 的结果是否为 -sin(q1)*A.x - cos(q1)*A.y
    assert cross(N.y, A.x) == -cos(q1)*A.z  # 测试 N.y 叉乘 A.x 的结果是否为 -cos(q1)*A.z
    assert cross(N.y, A.y) == sin(q1)*A.z   # 测试 N.y 叉乘 A.y 的结果是否为 sin(q1)*A.z
    assert cross(N.y, A.z) == cos(q1)*A.x - sin(q1)*A.y  # 测试 N
    # 断言：计算矢量 A.x + A.y 与矢量 N.x 的叉乘结果，应为 -sin(q1)*A.z - cos(q1)*A.z
    assert cross(A.x + A.y, N.x) == -sin(q1)*A.z - cos(q1)*A.z

    # 断言：计算矢量 A.x 与矢量 C.x 的叉乘结果，应为 sin(q3)*C.y
    assert cross(A.x, C.x) == sin(q3)*C.y
    # 断言：计算矢量 A.x 与矢量 C.y 的叉乘结果，应为 -sin(q3)*C.x + cos(q3)*C.z
    assert cross(A.x, C.y) == -sin(q3)*C.x + cos(q3)*C.z
    # 断言：计算矢量 A.x 与矢量 C.z 的叉乘结果，应为 -cos(q3)*C.y
    assert cross(A.x, C.z) == -cos(q3)*C.y
    # 断言：计算矢量 C.x 与矢量 A.x 的叉乘结果，应为 -sin(q3)*C.y
    assert cross(C.x, A.x) == -sin(q3)*C.y
    # 断言：计算矢量 C.y 与矢量 A.x 的叉乘结果，并在 C 变量下进行简化，应为 sin(q3)*C.x - cos(q3)*C.z
    assert cross(C.y, A.x).express(C).simplify() == sin(q3)*C.x - cos(q3)*C.z
    # 断言：计算矢量 C.z 与矢量 A.x 的叉乘结果，应为 cos(q3)*C.y
    assert cross(C.z, A.x) == cos(q3)*C.y
# 定义测试函数，用于验证 dot、cross、outer 函数与运算符的一致性
def test_operator_match():
    """Test that the output of dot, cross, outer functions match
    operator behavior.
    """
    # 创建参考坐标系 A
    A = ReferenceFrame('A')
    # 定义向量 v 为 A 坐标系中的 x 和 y 方向之和
    v = A.x + A.y
    # 计算 v 与 v 的点乘结果
    d = v | v
    # 创建一个零向量 zerov
    zerov = Vector(0)
    # 创建一个零偶张量 zerod
    zerod = Dyadic(0)

    # dot product 的测试
    assert d & d == dot(d, d)  # 验证 d 与 d 的点乘
    assert d & zerod == dot(d, zerod)  # 验证 d 与 zerod 的点乘
    assert zerod & d == dot(zerod, d)  # 验证 zerod 与 d 的点乘
    assert d & v == dot(d, v)  # 验证 d 与 v 的点乘
    assert v & d == dot(v, d)  # 验证 v 与 d 的点乘
    assert d & zerov == dot(d, zerov)  # 验证 d 与 zerov 的点乘
    assert zerov & d == dot(zerov, d)  # 验证 zerov 与 d 的点乘
    raises(TypeError, lambda: dot(d, S.Zero))  # 验证 dot 函数对 S.Zero 的异常处理
    raises(TypeError, lambda: dot(S.Zero, d))  # 验证 dot 函数对 S.Zero 的异常处理
    raises(TypeError, lambda: dot(d, 0))  # 验证 dot 函数对整数 0 的异常处理
    raises(TypeError, lambda: dot(0, d))  # 验证 dot 函数对整数 0 的异常处理
    assert v & v == dot(v, v)  # 验证 v 与 v 的点乘
    assert v & zerov == dot(v, zerov)  # 验证 v 与 zerov 的点乘
    assert zerov & v == dot(zerov, v)  # 验证 zerov 与 v 的点乘
    raises(TypeError, lambda: dot(v, S.Zero))  # 验证 dot 函数对 S.Zero 的异常处理
    raises(TypeError, lambda: dot(S.Zero, v))  # 验证 dot 函数对 S.Zero 的异常处理
    raises(TypeError, lambda: dot(v, 0))  # 验证 dot 函数对整数 0 的异常处理
    raises(TypeError, lambda: dot(0, v))  # 验证 dot 函数对整数 0 的异常处理

    # cross product 的测试
    raises(TypeError, lambda: cross(d, d))  # 验证 cross 函数对 d 与 d 的异常处理
    raises(TypeError, lambda: cross(d, zerod))  # 验证 cross 函数对 d 与 zerod 的异常处理
    raises(TypeError, lambda: cross(zerod, d))  # 验证 cross 函数对 zerod 与 d 的异常处理
    assert d ^ v == cross(d, v)  # 验证 d 与 v 的叉乘
    assert v ^ d == cross(v, d)  # 验证 v 与 d 的叉乘
    assert d ^ zerov == cross(d, zerov)  # 验证 d 与 zerov 的叉乘
    assert zerov ^ d == cross(zerov, d)  # 验证 zerov 与 d 的叉乘
    assert zerov ^ d == cross(zerov, d)  # 验证 zerov 与 d 的叉乘（重复注释）
    raises(TypeError, lambda: cross(d, S.Zero))  # 验证 cross 函数对 S.Zero 的异常处理
    raises(TypeError, lambda: cross(S.Zero, d))  # 验证 cross 函数对 S.Zero 的异常处理
    raises(TypeError, lambda: cross(d, 0))  # 验证 cross 函数对整数 0 的异常处理
    raises(TypeError, lambda: cross(0, d))  # 验证 cross 函数对整数 0 的异常处理
    assert v ^ v == cross(v, v)  # 验证 v 与 v 的叉乘
    assert v ^ zerov == cross(v, zerov)  # 验证 v 与 zerov 的叉乘
    assert zerov ^ v == cross(zerov, v)  # 验证 zerov 与 v 的叉乘
    raises(TypeError, lambda: cross(v, S.Zero))  # 验证 cross 函数对 S.Zero 的异常处理
    raises(TypeError, lambda: cross(S.Zero, v))  # 验证 cross 函数对 S.Zero 的异常处理
    raises(TypeError, lambda: cross(v, 0))  # 验证 cross 函数对整数 0 的异常处理
    raises(TypeError, lambda: cross(0, v))  # 验证 cross 函数对整数 0 的异常处理

    # outer product 的测试
    raises(TypeError, lambda: outer(d, d))  # 验证 outer 函数对 d 与 d 的异常处理
    raises(TypeError, lambda: outer(d, zerod))  # 验证 outer 函数对 d 与 zerod 的异常处理
    raises(TypeError, lambda: outer(zerod, d))  # 验证 outer 函数对 zerod 与 d 的异常处理
    raises(TypeError, lambda: outer(d, v))  # 验证 outer 函数对 d 与 v 的异常处理
    raises(TypeError, lambda: outer(v, d))  # 验证 outer 函数对 v 与 d 的异常处理
    raises(TypeError, lambda: outer(d, zerov))  # 验证 outer 函数对 d 与 zerov 的异常处理
    raises(TypeError, lambda: outer(zerov, d))  # 验证 outer 函数对 zerov 与 d 的异常处理
    raises(TypeError, lambda: outer(zerov, d))  # 验证 outer 函数对 zerov 与 d 的异常处理（重复注释）
    raises(TypeError, lambda: outer(d, S.Zero))  # 验证 outer 函数对 S.Zero 的异常处理
    raises(TypeError, lambda: outer(S.Zero, d))  # 验证 outer 函数对 S.Zero 的异常处理
    raises(TypeError, lambda: outer(d, 0))  # 验证 outer 函数对整数 0 的异常处理
    raises(TypeError, lambda: outer(0, d))  # 验证 outer 函数对整数 0 的异常处理
    assert v | v == outer(v, v)  # 验证 v 与 v 的外积
    assert v | zerov == outer(v, zerov)  # 验证 v 与 zerov 的外积
    assert zerov | v == outer(zerov, v)  # 验证 zerov 与 v 的外积
    raises(TypeError, lambda: outer(v, S.Zero))  # 验证 outer 函数对 S.Zero 的异常处理
    raises(TypeError, lambda: outer(S.Zero, v))  # 验证 outer 函数对 S.Zero 的异常处理
    raises(TypeError, lambda: outer(v, 0))  # 验证 outer 函数对整数 0 的异常处理
    raises(TypeError, lambda: outer(0, v))  # 验证 outer 函数对整数 0 的异常处理


def test_express():
    assert express(Vector(0), N) == Vector(0)  # 验证 express 函数对零向量的处理
    assert express(S.Zero, N) is S.Zero  # 验证 express 函数对 S.Zero 的处理
    assert express(A.x, C) == cos(q3)*C.x + sin(q3)*C.z  # 验证 express 函数对 A.x 的处理
    assert express(A.y, C) == sin(q2)*sin(q3)*C.x + cos(q2)*C.y - \
        sin(q2)*cos(q3)*C.z  # 验证 express 函数对 A.y 的处理
    assert express(A.z, C) == -sin(q3)*cos(q2)*C.x + sin(q2)*C.y + \
        cos(q2)*cos(q3)*C.z  # 验证 express 函数对 A.z 的处理
    # 断言表达式，验证 A 点在 N 坐标系中的表达式是否正确
    assert express(A.x, N) == cos(q1)*N.x + sin(q1)*N.y
    assert express(A.y, N) == -sin(q1)*N.x + cos(q1)*N.y
    assert express(A.z, N) == N.z
    assert express(A.x, A) == A.x
    assert express(A.y, A) == A.y
    assert express(A.z, A) == A.z
    assert express(A.x, B) == B.x
    assert express(A.y, B) == cos(q2)*B.y - sin(q2)*B.z
    assert express(A.z, B) == sin(q2)*B.y + cos(q2)*B.z
    assert express(A.x, C) == cos(q3)*C.x + sin(q3)*C.z
    assert express(A.y, C) == sin(q2)*sin(q3)*C.x + cos(q2)*C.y - \
        sin(q2)*cos(q3)*C.z
    assert express(A.z, C) == -sin(q3)*cos(q2)*C.x + sin(q2)*C.y + \
        cos(q2)*cos(q3)*C.z
    # 验证 UnitVectors 在转换时的正确性
    assert express(N.x, N) == N.x
    assert express(N.y, N) == N.y
    assert express(N.z, N) == N.z
    assert express(N.x, A) == (cos(q1)*A.x - sin(q1)*A.y)
    assert express(N.y, A) == (sin(q1)*A.x + cos(q1)*A.y)
    assert express(N.z, A) == A.z
    assert express(N.x, B) == (cos(q1)*B.x - sin(q1)*cos(q2)*B.y +
            sin(q1)*sin(q2)*B.z)
    assert express(N.y, B) == (sin(q1)*B.x + cos(q1)*cos(q2)*B.y -
            sin(q2)*cos(q1)*B.z)
    assert express(N.z, B) == (sin(q2)*B.y + cos(q2)*B.z)
    assert express(N.x, C) == (
        (cos(q1)*cos(q3) - sin(q1)*sin(q2)*sin(q3))*C.x -
        sin(q1)*cos(q2)*C.y +
        (sin(q3)*cos(q1) + sin(q1)*sin(q2)*cos(q3))*C.z)
    assert express(N.y, C) == (
        (sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*C.x +
        cos(q1)*cos(q2)*C.y +
        (sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*C.z)
    assert express(N.z, C) == (-sin(q3)*cos(q2)*C.x + sin(q2)*C.y +
            cos(q2)*cos(q3)*C.z)

    # 再次验证 A 点在 N 坐标系中的表达式是否正确
    assert express(A.x, N) == (cos(q1)*N.x + sin(q1)*N.y)
    assert express(A.y, N) == (-sin(q1)*N.x + cos(q1)*N.y)
    assert express(A.z, N) == N.z
    assert express(A.x, A) == A.x
    assert express(A.y, A) == A.y
    assert express(A.z, A) == A.z
    assert express(A.x, B) == B.x
    assert express(A.y, B) == (cos(q2)*B.y - sin(q2)*B.z)
    assert express(A.z, B) == (sin(q2)*B.y + cos(q2)*B.z)
    assert express(A.x, C) == (cos(q3)*C.x + sin(q3)*C.z)
    assert express(A.y, C) == (sin(q2)*sin(q3)*C.x + cos(q2)*C.y -
            sin(q2)*cos(q3)*C.z)
    assert express(A.z, C) == (-sin(q3)*cos(q2)*C.x + sin(q2)*C.y +
            cos(q2)*cos(q3)*C.z)

    # 验证 B 点在 N 坐标系中的表达式是否正确
    assert express(B.x, N) == (cos(q1)*N.x + sin(q1)*N.y)
    assert express(B.y, N) == (-sin(q1)*cos(q2)*N.x +
            cos(q1)*cos(q2)*N.y + sin(q2)*N.z)
    assert express(B.z, N) == (sin(q1)*sin(q2)*N.x -
            sin(q2)*cos(q1)*N.y + cos(q2)*N.z)
    assert express(B.x, A) == A.x
    assert express(B.y, A) == (cos(q2)*A.y + sin(q2)*A.z)
    assert express(B.z, A) == (-sin(q2)*A.y + cos(q2)*A.z)
    assert express(B.x, B) == B.x
    assert express(B.y, B) == B.y
    assert express(B.z, B) == B.z
    assert express(B.x, C) == (cos(q3)*C.x + sin(q3)*C.z)
    assert express(B.y, C) == C.y
    assert express(B.z, C) == (-sin(q3)*C.x + cos(q3)*C.z)
    # 断言表达式，验证向量 C.x 在 N 坐标系中的表达式是否正确
    assert express(C.x, N) == (
        (cos(q1)*cos(q3) - sin(q1)*sin(q2)*sin(q3))*N.x +
        (sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*N.y -
        sin(q3)*cos(q2)*N.z)
    # 断言表达式，验证向量 C.y 在 N 坐标系中的表达式是否正确
    assert express(C.y, N) == (
        -sin(q1)*cos(q2)*N.x + cos(q1)*cos(q2)*N.y + sin(q2)*N.z)
    # 断言表达式，验证向量 C.z 在 N 坐标系中的表达式是否正确
    assert express(C.z, N) == (
        (sin(q3)*cos(q1) + sin(q1)*sin(q2)*cos(q3))*N.x +
        (sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*N.y +
        cos(q2)*cos(q3)*N.z)
    # 断言表达式，验证向量 C.x 在 A 坐标系中的表达式是否正确
    assert express(C.x, A) == (cos(q3)*A.x + sin(q2)*sin(q3)*A.y -
            sin(q3)*cos(q2)*A.z)
    # 断言表达式，验证向量 C.y 在 A 坐标系中的表达式是否正确
    assert express(C.y, A) == (cos(q2)*A.y + sin(q2)*A.z)
    # 断言表达式，验证向量 C.z 在 A 坐标系中的表达式是否正确
    assert express(C.z, A) == (sin(q3)*A.x - sin(q2)*cos(q3)*A.y +
            cos(q2)*cos(q3)*A.z)
    # 断言表达式，验证向量 C.x 在 B 坐标系中的表达式是否正确
    assert express(C.x, B) == (cos(q3)*B.x - sin(q3)*B.z)
    # 断言表达式，验证向量 C.y 在 B 坐标系中的表达式是否正确
    assert express(C.y, B) == B.y
    # 断言表达式，验证向量 C.z 在 B 坐标系中的表达式是否正确
    assert express(C.z, B) == (sin(q3)*B.x + cos(q3)*B.z)
    # 断言表达式，验证向量 C.x 在 C 坐标系中的表达式是否正确
    assert express(C.x, C) == C.x
    # 断言表达式，验证向量 C.y 在 C 坐标系中的表达式是否正确
    assert express(C.y, C) == C.y
    # 断言表达式，验证向量 C.z 在 C 坐标系中的表达式是否正确
    assert express(C.z, C) == C.z == (C.z)

    # 验证向量 N.x 是否正确地被转换回单位向量形式
    assert N.x == express((cos(q1)*A.x - sin(q1)*A.y), N).simplify()
    # 验证向量 N.y 是否正确地被转换回单位向量形式
    assert N.y == express((sin(q1)*A.x + cos(q1)*A.y), N).simplify()
    # 验证向量 N.x 是否正确地被转换回单位向量形式
    assert N.x == express((cos(q1)*B.x - sin(q1)*cos(q2)*B.y +
            sin(q1)*sin(q2)*B.z), N).simplify()
    # 验证向量 N.y 是否正确地被转换回单位向量形式
    assert N.y == express((sin(q1)*B.x + cos(q1)*cos(q2)*B.y -
        sin(q2)*cos(q1)*B.z), N).simplify()
    # 验证向量 N.z 是否正确地被转换回单位向量形式
    assert N.z == express((sin(q2)*B.y + cos(q2)*B.z), N).simplify()

    """
    这些断言不是在测试我们的代码，而是在测试 SymPy 的自动简化（或未简化）功能。
    assert N.x == express((
            (cos(q1)*cos(q3)-sin(q1)*sin(q2)*sin(q3))*C.x -
            sin(q1)*cos(q2)*C.y +
            (sin(q3)*cos(q1)+sin(q1)*sin(q2)*cos(q3))*C.z), N)
    assert N.y == express((
            (sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*C.x +
            cos(q1)*cos(q2)*C.y +
            (sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*C.z), N)
    assert N.z == express((-sin(q3)*cos(q2)*C.x + sin(q2)*C.y +
            cos(q2)*cos(q3)*C.z), N)
    """

    # 验证向量 A.x 是否正确地被转换回单位向量形式
    assert A.x == express((cos(q1)*N.x + sin(q1)*N.y), A).simplify()
    # 验证向量 A.y 是否正确地被转换回单位向量形式
    assert A.y == express((-sin(q1)*N.x + cos(q1)*N.y), A).simplify()

    # 验证向量 A.y 是否正确地被转换回单位向量形式
    assert A.y == express((cos(q2)*B.y - sin(q2)*B.z), A).simplify()
    # 验证向量 A.z 是否正确地被转换回单位向量形式
    assert A.z == express((sin(q2)*B.y + cos(q2)*B.z), A).simplify()

    # 验证向量 A.x 是否正确地被转换回单位向量形式
    assert A.x == express((cos(q3)*C.x + sin(q3)*C.z), A).simplify()

    # Tripsimp 在这里也出问题了。
    #print express((sin(q2)*sin(q3)*C.x + cos(q2)*C.y -
    #        sin(q2)*cos(q3)*C.z), A)
    # 验证向量 A.y 是否正确地被转换回单位向量形式
    assert A.y == express((sin(q2)*sin(q3)*C.x + cos(q2)*C.y -
            sin(q2)*cos(q3)*C.z), A).simplify()

    # 验证向量 A.z 是否正确地被转换回单位向量形式
    assert A.z == express((-sin(q3)*cos(q2)*C.x + sin(q2)*C.y +
            cos(q2)*cos(q3)*C.z), A).simplify()
    # 验证向量 B.x 是否正确地被转换回单位向量形式
    assert B.x == express((cos(q1)*N.x + sin(q1)*N.y), B).simplify()
    # 验证向量 B.y 是否正确地被转换回单位向量形式
    assert B.y == express((-sin(q1)*cos(q2)*N.x +
            cos(q1)*cos(q2)*N.y + sin(q2)*N.z), B).simplify()
    # 断言：验证表达式 B.z 是否等于根据给定表达式 sin(q1)*sin(q2)*N.x - sin(q2)*cos(q1)*N.y + cos(q2)*N.z 在参考系 B 下化简的结果
    assert B.z == express((sin(q1)*sin(q2)*N.x -
            sin(q2)*cos(q1)*N.y + cos(q2)*N.z), B).simplify()

    # 断言：验证表达式 B.y 是否等于根据给定表达式 cos(q2)*A.y + sin(q2)*A.z 在参考系 B 下化简的结果
    assert B.y == express((cos(q2)*A.y + sin(q2)*A.z), B).simplify()
    
    # 断言：验证表达式 B.z 是否等于根据给定表达式 -sin(q2)*A.y + cos(q2)*A.z 在参考系 B 下化简的结果
    assert B.z == express((-sin(q2)*A.y + cos(q2)*A.z), B).simplify()
    
    # 断言：验证表达式 B.x 是否等于根据给定表达式 cos(q3)*C.x + sin(q3)*C.z 在参考系 B 下化简的结果
    assert B.x == express((cos(q3)*C.x + sin(q3)*C.z), B).simplify()
    
    # 断言：验证表达式 B.z 是否等于根据给定表达式 -sin(q3)*C.x + cos(q3)*C.z 在参考系 B 下化简的结果
    assert B.z == express((-sin(q3)*C.x + cos(q3)*C.z), B).simplify()

    """
    # 以下注释部分被注释掉的代码块

    # 断言：验证表达式 C.x 是否等于根据给定表达式 (cos(q1)*cos(q3)-sin(q1)*sin(q2)*sin(q3))*N.x + (sin(q1)*cos(q3)+sin(q2)*sin(q3)*cos(q1))*N.y - sin(q3)*cos(q2)*N.z 在参考系 C 下的表达式
    assert C.x == express((
            (cos(q1)*cos(q3)-sin(q1)*sin(q2)*sin(q3))*N.x +
            (sin(q1)*cos(q3)+sin(q2)*sin(q3)*cos(q1))*N.y -
            sin(q3)*cos(q2)*N.z), C)

    # 断言：验证表达式 C.y 是否等于根据给定表达式 -sin(q1)*cos(q2)*N.x + cos(q1)*cos(q2)*N.y + sin(q2)*N.z 在参考系 C 下的表达式
    assert C.y == express((
            -sin(q1)*cos(q2)*N.x + cos(q1)*cos(q2)*N.y + sin(q2)*N.z), C)

    # 断言：验证表达式 C.z 是否等于根据给定表达式 (sin(q3)*cos(q1)+sin(q1)*sin(q2)*cos(q3))*N.x + (sin(q1)*sin(q3)-sin(q2)*cos(q1)*cos(q3))*N.y + cos(q2)*cos(q3)*N.z 在参考系 C 下的表达式
    assert C.z == express((
            (sin(q3)*cos(q1)+sin(q1)*sin(q2)*cos(q3))*N.x +
            (sin(q1)*sin(q3)-sin(q2)*cos(q1)*cos(q3))*N.y +
            cos(q2)*cos(q3)*N.z), C)
    """

    # 断言：验证表达式 C.x 是否等于根据给定表达式 cos(q3)*A.x + sin(q2)*sin(q3)*A.y - sin(q3)*cos(q2)*A.z 在参考系 C 下化简的结果
    assert C.x == express((cos(q3)*A.x + sin(q2)*sin(q3)*A.y -
            sin(q3)*cos(q2)*A.z), C).simplify()
    
    # 断言：验证表达式 C.y 是否等于根据给定表达式 cos(q2)*A.y + sin(q2)*A.z 在参考系 C 下化简的结果
    assert C.y == express((cos(q2)*A.y + sin(q2)*A.z), C).simplify()
    
    # 断言：验证表达式 C.z 是否等于根据给定表达式 sin(q3)*A.x - sin(q2)*cos(q3)*A.y + cos(q2)*cos(q3)*A.z 在参考系 C 下化简的结果
    assert C.z == express((sin(q3)*A.x - sin(q2)*cos(q3)*A.y +
            cos(q2)*cos(q3)*A.z), C).simplify()
    
    # 断言：验证表达式 C.x 是否等于根据给定表达式 cos(q3)*B.x - sin(q3)*B.z 在参考系 C 下化简的结果
    assert C.x == express((cos(q3)*B.x - sin(q3)*B.z), C).simplify()
    
    # 断言：验证表达式 C.z 是否等于根据给定表达式 sin(q3)*B.x + cos(q3)*B.z 在参考系 C 下化简的结果
    assert C.z == express((sin(q3)*B.x + cos(q3)*B.z), C).simplify()
def test_time_derivative():
    # 定义一个惯性参考系 A
    A = ReferenceFrame('A')
    # 定义动力学符号 q 和其一阶导数 qd
    q = dynamicsymbols('q')
    qd = dynamicsymbols('q', 1)
    # 在参考系 A 中建立一个新的参考系 B，通过绕 A.z 轴旋转 q 角度
    B = A.orientnew('B', 'Axis', [q, A.z])
    # 定义一个双重点积 d，即 A.x | A.x
    d = A.x | A.x
    # 断言时间导数函数 time_derivative 对 d 的结果
    assert time_derivative(d, B) == (-qd) * (A.y | A.x) + \
           (-qd) * (A.x | A.y)
    
    # 定义另一个双重点积 d1，即 A.x | B.y
    d1 = A.x | B.y
    # 断言时间导数函数 time_derivative 对 d1 在参考系 A 中的结果
    assert time_derivative(d1, A) == - qd*(A.x|B.x)
    # 断言时间导数函数 time_derivative 对 d1 在参考系 B 中的结果
    assert time_derivative(d1, B) == - qd*(A.y|B.y)
    
    # 定义另一个双重点积 d2，即 A.x | B.x
    d2 = A.x | B.x
    # 断言时间导数函数 time_derivative 对 d2 在参考系 A 中的结果
    assert time_derivative(d2, A) == qd*(A.x|B.y)
    # 断言时间导数函数 time_derivative 对 d2 在参考系 B 中的结果
    assert time_derivative(d2, B) == - qd*(A.y|B.x)
    
    # 定义另一个双重点积 d3，即 A.x | B.z
    d3 = A.x | B.z
    # 断言时间导数函数 time_derivative 对 d3 在参考系 A 中的结果
    assert time_derivative(d3, A) == 0
    # 断言时间导数函数 time_derivative 对 d3 在参考系 B 中的结果
    assert time_derivative(d3, B) == - qd*(A.y|B.z)
    
    # 定义四个动力学符号 q1, q2, q3, q4 和它们的一阶和二阶导数
    q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
    q1d, q2d, q3d, q4d = dynamicsymbols('q1 q2 q3 q4', 1)
    q1dd, q2dd, q3dd, q4dd = dynamicsymbols('q1 q2 q3 q4', 2)
    # 在参考系 B 中建立一个新的参考系 C，通过绕 B.x 轴旋转 q4 角度
    C = B.orientnew('C', 'Axis', [q4, B.x])
    
    # 定义三个矢量 v1, v2, v3
    v1 = q1 * A.z
    v2 = q2*A.x + q3*B.y
    v3 = q1*A.x + q2*A.y + q3*A.z
    
    # 断言时间导数函数 time_derivative 对 v1 在参考系 B 中的结果
    assert time_derivative(v1, B) == q1d*A.z
    # 断言时间导数函数 time_derivative 对 v1 在参考系 C 中的结果
    assert time_derivative(v1, C) == - q1*sin(q)*q4d*A.x + \
           q1*cos(q)*q4d*A.y + q1d*A.z
    
    # 断言时间导数函数 time_derivative 对 v2 在参考系 A 中的结果
    assert time_derivative(v2, A) == q2d*A.x - q3*qd*B.x + q3d*B.y
    # 断言时间导数函数 time_derivative 对 v2 在参考系 C 中的结果
    assert time_derivative(v2, C) == q2d*A.x - q2*qd*A.y + \
           q2*sin(q)*q4d*A.z + q3d*B.y - q3*q4d*B.z
    
    # 断言时间导数函数 time_derivative 对 v3 在参考系 B 中的结果
    assert time_derivative(v3, B) == (q2*qd + q1d)*A.x + \
           (-q1*qd + q2d)*A.y + q3d*A.z
    
    # 断言时间导数函数 time_derivative 对 d 在参考系 C 中的结果
    assert time_derivative(d, C) == - qd*(A.y|A.x) + \
           sin(q)*q4d*(A.z|A.x) - qd*(A.x|A.y) + sin(q)*q4d*(A.x|A.z)
    
    # 断言 time_derivative 函数在参数 order 为 0.5 时抛出 ValueError 异常
    raises(ValueError, lambda: time_derivative(B.x, C, order=0.5))
    # 断言 time_derivative 函数在参数 order 为 -1 时抛出 ValueError 异常
    raises(ValueError, lambda: time_derivative(B.x, C, order=-1))


def test_get_motion_methods():
    # 初始化动力学符号 _t
    t = dynamicsymbols._t
    # 定义多个符号变量 s1, s2, s3, S1, S2, S3, S4, S5, S6, t1, t2, a, b, c
    s1, s2, s3 = symbols('s1 s2 s3')
    S1, S2, S3 = symbols('S1 S2 S3')
    S4, S5, S6 = symbols('S4 S5 S6')
    t1, t2 = symbols('t1 t2')
    a, b, c = dynamicsymbols('a b c')
    ad, bd, cd = dynamicsymbols('a b c', 1)
    a2d, b2d, c2d = dynamicsymbols('a b c', 2)
    
    # 定义矢量 v0, v01, v1, v2, v2d, v2dd
    v0 = S1*N.x + S2*N.y + S3*N.z
    v01 = S4*N.x + S5*N.y + S6*N.z
    v1 = s1*N.x + s2*N.y + s3*N.z
    v2 = a*N.x + b*N.y + c*N.z
    v2d = ad*N.x + bd*N.y + cd*N.z
    v2dd = a2d*N.x + b2d*N.y + c2d*N.z
    
    # 断言 get_motion_params 函数的各种参数组合结果
    assert get_motion_params(frame=N) == (0, 0, 0)
    assert get_motion_params(N, position=v1) == (0, 0, v1)
    assert get_motion_params(N, position=v2) == (v2dd, v2d, v2)
    assert get_motion_params(N, velocity=v1) == (0, v1, v1 * t)
    assert get_motion_params(N, velocity=v1, position=v0, timevalue1=t1) == \
           (0, v1, v0 + v1*(t - t1))
    answer = get_motion_params(N, velocity=v1, position=v2, timevalue1=t1)
    answer_expected = (0, v1, v1*t - v1*t1 + v2.subs(t, t1))
    # 断言检查是否 answer 等于 answer_expected
    assert answer == answer_expected

    # 调用函数 get_motion_params，传入参数 N、velocity=v2、position=v0、timevalue1=t1
    answer = get_motion_params(N, velocity=v2, position=v0, timevalue1=t1)
    
    # 构建积分向量 integral_vector，由三个分量组成：N.x 分量、N.y 分量、N.z 分量
    integral_vector = Integral(a, (t, t1, t))*N.x + Integral(b, (t, t1, t))*N.y \
            + Integral(c, (t, t1, t))*N.z
    
    # 设置预期的 answer_expected 值，包括 v2d、v2、v0 + integral_vector
    answer_expected = (v2d, v2, v0 + integral_vector)
    
    # 断言检查是否 answer 等于 answer_expected
    assert answer == answer_expected

    # 测试加速度参数
    # 断言检查 get_motion_params 函数返回值是否符合预期：(v1, v1 * t, v1 * t**2/2)
    assert get_motion_params(N, acceleration=v1) == \
           (v1, v1 * t, v1 * t**2/2)
    
    # 断言检查 get_motion_params 函数返回值是否符合预期，传入多个参数
    assert get_motion_params(N, acceleration=v1, velocity=v0,
                          position=v2, timevalue1=t1, timevalue2=t2) == \
           (v1, (v0 + v1*t - v1*t2),
            -v0*t1 + v1*t**2/2 + v1*t2*t1 - \
            v1*t1**2/2 + t*(v0 - v1*t2) + \
            v2.subs(t, t1))
    
    # 断言检查 get_motion_params 函数返回值是否符合预期，传入不同的参数
    assert get_motion_params(N, acceleration=v1, velocity=v0,
                             position=v01, timevalue1=t1, timevalue2=t2) == \
           (v1, v0 + v1*t - v1*t2,
            -v0*t1 + v01 + v1*t**2/2 + \
            v1*t2*t1 - v1*t1**2/2 + \
            t*(v0 - v1*t2))
    
    # 调用函数 get_motion_params，传入参数 acceleration=a*N.x, velocity=S1*N.x, position=S2*N.x, timevalue1=t1, timevalue2=t2
    answer = get_motion_params(N, acceleration=a*N.x, velocity=S1*N.x,
                          position=S2*N.x, timevalue1=t1, timevalue2=t2)
    
    # 构建积分 i1
    i1 = Integral(a, (t, t2, t))
    
    # 设置预期的 answer_expected 值，包括 a*N.x、(S1 + i1)*N.x、(S2 + Integral(S1 + i1, (t, t1, t)))*N.x
    answer_expected = (a*N.x, (S1 + i1)*N.x, \
        (S2 + Integral(S1 + i1, (t, t1, t)))*N.x)
    
    # 断言检查是否 answer 等于 answer_expected
    assert answer == answer_expected
def test_kin_eqs():
    # 定义四元数变量
    q0, q1, q2, q3 = dynamicsymbols('q0 q1 q2 q3')
    # 定义四元数的时间导数变量
    q0d, q1d, q2d, q3d = dynamicsymbols('q0 q1 q2 q3', 1)
    # 定义速度相关变量
    u1, u2, u3 = dynamicsymbols('u1 u2 u3')
    # 计算运动学方程，返回在特定参考系中的表达
    ke = kinematic_equations([u1,u2,u3], [q1,q2,q3], 'body', 313)
    # 断言运动学方程函数的返回值
    assert ke == kinematic_equations([u1,u2,u3], [q1,q2,q3], 'body', '313')
    # 计算四元数的动力学方程
    kds = kinematic_equations([u1, u2, u3], [q0, q1, q2, q3], 'quaternion')
    # 断言四元数的动力学方程函数的返回值
    assert kds == [-0.5 * q0 * u1 - 0.5 * q2 * u3 + 0.5 * q3 * u2 + q1d,
            -0.5 * q0 * u2 + 0.5 * q1 * u3 - 0.5 * q3 * u1 + q2d,
            -0.5 * q0 * u3 - 0.5 * q1 * u2 + 0.5 * q2 * u1 + q3d,
            0.5 * q1 * u1 + 0.5 * q2 * u2 + 0.5 * q3 * u3 + q0d]
    # 检查引发特定异常情况
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2], 'quaternion'))
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2, q3], 'quaternion', '123'))
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2, q3], 'foo'))
    raises(TypeError, lambda: kinematic_equations(u1, [q0, q1, q2, q3], 'quaternion'))
    raises(TypeError, lambda: kinematic_equations([u1], [q0, q1, q2, q3], 'quaternion'))
    raises(TypeError, lambda: kinematic_equations([u1, u2, u3], q0, 'quaternion'))
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2, q3], 'body'))
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2, q3], 'space'))
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2], 'body', '222'))
    # 断言函数计算结果
    assert kinematic_equations([0, 0, 0], [q0, q1, q2], 'space') == [S.Zero, S.Zero, S.Zero]


def test_partial_velocity():
    # 定义动力学符号变量
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1 q2 q3 u1 u2 u3')
    # 定义额外的速度相关变量
    u4, u5 = dynamicsymbols('u4, u5')
    # 定义半定向的参考系
    r = symbols('r')

    # 定义惯性参考系
    N = ReferenceFrame('N')
    # 以惯性参考系为基准创建新的定向
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    # 以Y为基准创建新的定向
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    # 以L为基准创建新的定向
    R = L.orientnew('R', 'Axis', [q3, L.y])
    # 设置R相对于N的角速度
    R.set_ang_vel(N, u1 * L.x + u2 * L.y + u3 * L.z)

    # 创建一个点C
    C = Point('C')
    # 设置点C在惯性参考系N中的速度
    C.set_vel(N, u4 * L.x + u5 * (Y.z ^ L.x))
    # 在点C的位置创建一个新的点Dmc，位置为r * L.z
    Dmc = C.locatenew('Dmc', r * L.z)
    # 根据速度列表，计算Dmc相对于N的速度，并根据R定向
    Dmc.v2pt_theory(C, N, R)

    # 定义速度列表和速度相关变量列表
    vel_list = [Dmc.vel(N), C.vel(N), R.ang_vel_in(N)]
    u_list = [u1, u2, u3, u4, u5]
    # 断言部分速度函数的返回值
    assert (partial_velocity(vel_list, u_list, N) ==
            [[- r*L.y, r*L.x, 0, L.x, cos(q2)*L.y - sin(q2)*L.z],
            [0, 0, 0, L.x, cos(q2)*L.y - sin(q2)*L.z],
            [L.x, L.y, L.z, 0, 0]])

    # 确保无论是否定义了帧间方向，都能计算部分速度
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    v = u4 * A.x + u5 * B.y
    assert partial_velocity((v, ), (u4, u5), A) == [[A.x, B.y]]

    # 检查引发特定异常情况
    raises(TypeError, lambda: partial_velocity(Dmc.vel(N), u_list, N))
    raises(TypeError, lambda: partial_velocity(vel_list, u1, N))


def test_dynamicsymbols():
    # 测试动力学符号的假设
    f1 = dynamicsymbols('f1')
    f2 = dynamicsymbols('f2', real=True)
    # 定义符号变量 f3，其为正数
    f3 = dynamicsymbols('f3', positive=True)
    
    # 定义符号变量 f4 和 f5，它们不满足交换律
    f4, f5 = dynamicsymbols('f4,f5', commutative=False)
    
    # 定义符号变量 f6，其为整数
    f6 = dynamicsymbols('f6', integer=True)
    
    # 断言 f1 的实部为 None
    assert f1.is_real is None
    
    # 断言 f2 的实部为真（即 f2 是实数）
    assert f2.is_real
    
    # 断言 f3 是正数
    assert f3.is_positive
    
    # 断言 f4*f5 不等于 f5*f4（验证 f4 和 f5 不满足交换律）
    assert f4*f5 != f5*f4
    
    # 断言 f6 是整数
    assert f6.is_integer
```