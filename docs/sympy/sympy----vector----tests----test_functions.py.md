# `D:\src\scipysrc\sympy\sympy\vector\tests\test_functions.py`

```
from sympy.vector.vector import Vector  # 导入 Vector 类
from sympy.vector.coordsysrect import CoordSys3D  # 导入 CoordSys3D 类
from sympy.vector.functions import express, matrix_to_vector, orthogonalize  # 导入 express, matrix_to_vector, orthogonalize 函数
from sympy.core.numbers import Rational  # 导入 Rational 类
from sympy.core.singleton import S  # 导入 S 单例
from sympy.core.symbol import symbols  # 导入 symbols 函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入 sqrt 函数
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入 cos 和 sin 函数
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix  # 导入 ImmutableDenseMatrix 类，并重命名为 Matrix
from sympy.testing.pytest import raises  # 导入 raises 函数

N = CoordSys3D('N')  # 创建一个名为 N 的三维坐标系对象
q1, q2, q3, q4, q5 = symbols('q1 q2 q3 q4 q5')  # 创建符号 q1, q2, q3, q4, q5

A = N.orient_new_axis('A', q1, N.k)  # 在 N 坐标系中绕 N.k 轴旋转 q1 角度，创建一个名为 A 的新坐标系对象
B = A.orient_new_axis('B', q2, A.i)  # 在 A 坐标系中绕 A.i 轴旋转 q2 角度，创建一个名为 B 的新坐标系对象
C = B.orient_new_axis('C', q3, B.j)  # 在 B 坐标系中绕 B.j 轴旋转 q3 角度，创建一个名为 C 的新坐标系对象

def test_express():
    assert express(Vector.zero, N) == Vector.zero  # 测试将零向量表达为 N 坐标系中的结果是否为零向量
    assert express(S.Zero, N) is S.Zero  # 测试将零标量表达为 N 坐标系中的结果是否为零标量
    assert express(A.i, C) == cos(q3)*C.i + sin(q3)*C.k  # 测试将 A.i 向量表达为 C 坐标系中的结果
    assert express(A.j, C) == sin(q2)*sin(q3)*C.i + cos(q2)*C.j - \
        sin(q2)*cos(q3)*C.k  # 测试将 A.j 向量表达为 C 坐标系中的结果
    assert express(A.k, C) == -sin(q3)*cos(q2)*C.i + sin(q2)*C.j + \
        cos(q2)*cos(q3)*C.k  # 测试将 A.k 向量表达为 C 坐标系中的结果
    assert express(A.i, N) == cos(q1)*N.i + sin(q1)*N.j  # 测试将 A.i 向量表达为 N 坐标系中的结果
    assert express(A.j, N) == -sin(q1)*N.i + cos(q1)*N.j  # 测试将 A.j 向量表达为 N 坐标系中的结果
    assert express(A.k, N) == N.k  # 测试将 A.k 向量表达为 N 坐标系中的结果
    assert express(A.i, A) == A.i  # 测试将 A.i 向量表达为 A 坐标系中的结果
    assert express(A.j, A) == A.j  # 测试将 A.j 向量表达为 A 坐标系中的结果
    assert express(A.k, A) == A.k  # 测试将 A.k 向量表达为 A 坐标系中的结果
    assert express(A.i, B) == B.i  # 测试将 A.i 向量表达为 B 坐标系中的结果
    assert express(A.j, B) == cos(q2)*B.j - sin(q2)*B.k  # 测试将 A.j 向量表达为 B 坐标系中的结果
    assert express(A.k, B) == sin(q2)*B.j + cos(q2)*B.k  # 测试将 A.k 向量表达为 B 坐标系中的结果
    assert express(A.i, C) == cos(q3)*C.i + sin(q3)*C.k  # 测试将 A.i 向量表达为 C 坐标系中的结果
    assert express(A.j, C) == sin(q2)*sin(q3)*C.i + cos(q2)*C.j - \
        sin(q2)*cos(q3)*C.k  # 测试将 A.j 向量表达为 C 坐标系中的结果
    assert express(A.k, C) == -sin(q3)*cos(q2)*C.i + sin(q2)*C.j + \
        cos(q2)*cos(q3)*C.k  # 测试将 A.k 向量表达为 C 坐标系中的结果
    # 检查确保单位向量正确转换
    assert express(N.i, N) == N.i  # 测试将 N.i 向量表达为 N 坐标系中的结果
    assert express(N.j, N) == N.j  # 测试将 N.j 向量表达为 N 坐标系中的结果
    assert express(N.k, N) == N.k  # 测试将 N.k 向量表达为 N 坐标系中的结果
    assert express(N.i, A) == (cos(q1)*A.i - sin(q1)*A.j)  # 测试将 N.i 向量表达为 A 坐标系中的结果
    assert express(N.j, A) == (sin(q1)*A.i + cos(q1)*A.j)  # 测试将 N.j 向量表达为 A 坐标系中的结果
    assert express(N.k, A) == A.k  # 测试将 N.k 向量表达为 A 坐标系中的结果
    assert express(N.i, B) == (cos(q1)*B.i - sin(q1)*cos(q2)*B.j +
            sin(q1)*sin(q2)*B.k)  # 测试将 N.i 向量表达为 B 坐标系中的结果
    assert express(N.j, B) == (sin(q1)*B.i + cos(q1)*cos(q2)*B.j -
            sin(q2)*cos(q1)*B.k)  # 测试将 N.j 向量表达为 B 坐标系中的结果
    assert express(N.k, B) == (sin(q2)*B.j + cos(q2)*B.k)  # 测试将 N.k 向量表达为 B 坐标系中的结果
    assert express(N.i, C) == (
        (cos(q1)*cos(q3) - sin(q1)*sin(q2)*sin(q3))*C.i -
        sin(q1)*cos(q2)*C.j +
        (sin(q3)*cos(q1) + sin(q1)*sin(q2)*cos(q3))*C.k)  # 测试将 N.i 向量表达为 C 坐标系中的结果
    assert express(N.j, C) == (
        (sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*C.i +
        cos(q1)*cos(q2)*C.j +
        (sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*C.k)  # 测试将 N.j 向量表达为 C 坐标系中的结果
    assert express(N.k, C) == (-sin(q3)*cos(q2)*C.i + sin(q2)*C.j +
            cos(q2)*cos(q3)*C.k)  # 测试将 N.k 向量表达为 C 坐标系中的结果

    assert express(A.i, N) == (cos(q1)*N.i + sin(q1)*N.j)  # 再次测试将 A.i 向量表达为 N 坐标系中的结果
    assert express(A.j, N) == (-sin(q1)*N.i + cos(q1)*N.j)  # 再次测试将 A.j 向量表达为 N 坐标系中的结果
    assert express(A.k, N) == N.k  # 再次测试将 A.k 向量表达为 N 坐标系中的结果
    assert express(A.i, A) == A.i
    # 断言语句，验证 express 函数对 A.i 和 B 的操作结果是否等于 B.i
    assert express(A.i, B) == B.i
    # 断言语句，验证 express 函数对 A.j 和 B 的操作结果是否等于 (cos(q2)*B.j - sin(q2)*B.k)
    assert express(A.j, B) == (cos(q2)*B.j - sin(q2)*B.k)
    # 断言语句，验证 express 函数对 A.k 和 B 的操作结果是否等于 (sin(q2)*B.j + cos(q2)*B.k)
    assert express(A.k, B) == (sin(q2)*B.j + cos(q2)*B.k)

    # 断言语句，验证 express 函数对 A.i 和 C 的操作结果是否等于 (cos(q3)*C.i + sin(q3)*C.k)
    assert express(A.i, C) == (cos(q3)*C.i + sin(q3)*C.k)
    # 断言语句，验证 express 函数对 A.j 和 C 的操作结果是否等于 (sin(q2)*sin(q3)*C.i + cos(q2)*C.j - sin(q2)*cos(q3)*C.k)
    assert express(A.j, C) == (sin(q2)*sin(q3)*C.i + cos(q2)*C.j - sin(q2)*cos(q3)*C.k)
    # 断言语句，验证 express 函数对 A.k 和 C 的操作结果是否等于 (-sin(q3)*cos(q2)*C.i + sin(q2)*C.j + cos(q2)*cos(q3)*C.k)
    assert express(A.k, C) == (-sin(q3)*cos(q2)*C.i + sin(q2)*C.j + cos(q2)*cos(q3)*C.k)

    # 断言语句，验证 express 函数对 B.i 和 N 的操作结果是否等于 (cos(q1)*N.i + sin(q1)*N.j)
    assert express(B.i, N) == (cos(q1)*N.i + sin(q1)*N.j)
    # 断言语句，验证 express 函数对 B.j 和 N 的操作结果是否等于 (-sin(q1)*cos(q2)*N.i + cos(q1)*cos(q2)*N.j + sin(q2)*N.k)
    assert express(B.j, N) == (-sin(q1)*cos(q2)*N.i + cos(q1)*cos(q2)*N.j + sin(q2)*N.k)
    # 断言语句，验证 express 函数对 B.k 和 N 的操作结果是否等于 (sin(q1)*sin(q2)*N.i - sin(q2)*cos(q1)*N.j + cos(q2)*N.k)
    assert express(B.k, N) == (sin(q1)*sin(q2)*N.i - sin(q2)*cos(q1)*N.j + cos(q2)*N.k)

    # 断言语句，验证 express 函数对 B.i 和 A 的操作结果是否等于 A.i
    assert express(B.i, A) == A.i
    # 断言语句，验证 express 函数对 B.j 和 A 的操作结果是否等于 (cos(q2)*A.j + sin(q2)*A.k)
    assert express(B.j, A) == (cos(q2)*A.j + sin(q2)*A.k)
    # 断言语句，验证 express 函数对 B.k 和 A 的操作结果是否等于 (-sin(q2)*A.j + cos(q2)*A.k)
    assert express(B.k, A) == (-sin(q2)*A.j + cos(q2)*A.k)

    # 断言语句，验证 express 函数对 B.i 和 B 的操作结果是否等于 B.i
    assert express(B.i, B) == B.i
    # 断言语句，验证 express 函数对 B.j 和 B 的操作结果是否等于 B.j
    assert express(B.j, B) == B.j
    # 断言语句，验证 express 函数对 B.k 和 B 的操作结果是否等于 B.k
    assert express(B.k, B) == B.k

    # 断言语句，验证 express 函数对 B.i 和 C 的操作结果是否等于 (cos(q3)*C.i + sin(q3)*C.k)
    assert express(B.i, C) == (cos(q3)*C.i + sin(q3)*C.k)
    # 断言语句，验证 express 函数对 B.j 和 C 的操作结果是否等于 C.j
    assert express(B.j, C) == C.j
    # 断言语句，验证 express 函数对 B.k 和 C 的操作结果是否等于 (-sin(q3)*C.i + cos(q3)*C.k)
    assert express(B.k, C) == (-sin(q3)*C.i + cos(q3)*C.k)

    # 断言语句，验证 express 函数对 C.i 和 N 的操作结果是否等于复杂表达式
    assert express(C.i, N) == (
        (cos(q1)*cos(q3) - sin(q1)*sin(q2)*sin(q3))*N.i +
        (sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*N.j -
        sin(q3)*cos(q2)*N.k)
    # 断言语句，验证 express 函数对 C.j 和 N 的操作结果是否等于复杂表达式
    assert express(C.j, N) == (
        -sin(q1)*cos(q2)*N.i + cos(q1)*cos(q2)*N.j + sin(q2)*N.k)
    # 断言语句，验证 express 函数对 C.k 和 N 的操作结果是否等于复杂表达式
    assert express(C.k, N) == (
        (sin(q3)*cos(q1) + sin(q1)*sin(q2)*cos(q3))*N.i +
        (sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*N.j +
        cos(q2)*cos(q3)*N.k)

    # 断言语句，验证 express 函数对 C.i 和 A 的操作结果是否等于复杂表达式
    assert express(C.i, A) == (cos(q3)*A.i + sin(q2)*sin(q3)*A.j - sin(q3)*cos(q2)*A.k)
    # 断言语句，验证 express 函数对 C.j 和 A 的操作结果是否等于 (cos(q2)*A.j + sin(q2)*A.k)
    assert express(C.j, A) == (cos(q2)*A.j + sin(q2)*A.k)
    # 断言语句，验证 express 函数对 C.k 和 A 的操作结果是否等于复杂表达式
    assert express(C.k, A) == (sin(q3)*A.i - sin(q2)*cos(q3)*A.j + cos(q2)*cos(q3)*A.k)

    # 断言语句，验证 express 函数对 C.i 和 B 的操作结果是否等于 (cos(q3)*B.i - sin(q3)*B.k)
    assert express(C.i, B) == (cos(q3)*B.i - sin(q3)*B.k)
    # 断言语句，验证 express 函数对 C.j 和 B 的操作结果是否等于 B.j
    assert express(C.j, B) == B.j
    # 断言语句，验证 express 函数对 C.k 和 B 的操作结果是否等于 (sin(q3)*B.i + cos(q3)*B.k)
    assert express(C.k, B) == (sin(q3)*B.i + cos(q3)*B.k)

    # 断言语句，验证 express 函数对 C.i 和 C 的操作结果是否等于 C.i
    assert express(C.i, C) == C.i
    # 断言语句，验证 express 函数对 C.j 和 C 的操作结果是否等于 C.j
    assert express(C.j, C) == C.j
    # 断言语句，验证 express 函数对 C.k 和 C 的操作结果是否等于 C.k
    assert express(C.k, C) == C.k == (C.k)

    # 断言语句，验证 express 函数对 N.i 和 N 的操作结果是否等于 N.i
    assert N.i == express((cos(q1)*A.i - sin(q1)*A.j), N).simplify()
    # 断言语句，验证 express 函数对 N.j 和 N 的操作结果是否等于 N.j
    assert N.j == express((sin(q1)*A.i + cos(q1)*A.j), N).simplify()

    # 断言语句，验证 express 函数对 N.i 和 N 的操作结果是否等于复杂表达式
    assert N.i == express((cos(q1)*B.i - sin(q1)*cos(q2)*B.j + sin(q
    # 断言：验证 A 坐标系的 j 分量是否等于给定表达式的简化结果
    assert A.j == express((sin(q2)*sin(q3)*C.i + cos(q2)*C.j -
            sin(q2)*cos(q3)*C.k), A).simplify()
    
    # 断言：验证 A 坐标系的 k 分量是否等于给定表达式的简化结果
    assert A.k == express((-sin(q3)*cos(q2)*C.i + sin(q2)*C.j +
            cos(q2)*cos(q3)*C.k), A).simplify()
    
    # 断言：验证 B 坐标系的 i 分量是否等于给定表达式的简化结果
    assert B.i == express((cos(q1)*N.i + sin(q1)*N.j), B).simplify()
    
    # 断言：验证 B 坐标系的 j 分量是否等于给定表达式的简化结果
    assert B.j == express((-sin(q1)*cos(q2)*N.i +
            cos(q1)*cos(q2)*N.j + sin(q2)*N.k), B).simplify()
    
    # 断言：验证 B 坐标系的 k 分量是否等于给定表达式的简化结果
    assert B.k == express((sin(q1)*sin(q2)*N.i -
            sin(q2)*cos(q1)*N.j + cos(q2)*N.k), B).simplify()
    
    # 断言：验证 B 坐标系的 j 分量是否等于给定表达式的简化结果
    assert B.j == express((cos(q2)*A.j + sin(q2)*A.k), B).simplify()
    
    # 断言：验证 B 坐标系的 k 分量是否等于给定表达式的简化结果
    assert B.k == express((-sin(q2)*A.j + cos(q2)*A.k), B).simplify()
    
    # 断言：验证 B 坐标系的 i 分量是否等于给定表达式的简化结果
    assert B.i == express((cos(q3)*C.i + sin(q3)*C.k), B).simplify()
    
    # 断言：验证 B 坐标系的 k 分量是否等于给定表达式的简化结果
    assert B.k == express((-sin(q3)*C.i + cos(q3)*C.k), B).simplify()
    
    # 断言：验证 C 坐标系的 i 分量是否等于给定表达式的简化结果
    assert C.i == express((cos(q3)*A.i + sin(q2)*sin(q3)*A.j -
            sin(q3)*cos(q2)*A.k), C).simplify()
    
    # 断言：验证 C 坐标系的 j 分量是否等于给定表达式的简化结果
    assert C.j == express((cos(q2)*A.j + sin(q2)*A.k), C).simplify()
    
    # 断言：验证 C 坐标系的 k 分量是否等于给定表达式的简化结果
    assert C.k == express((sin(q3)*A.i - sin(q2)*cos(q3)*A.j +
            cos(q2)*cos(q3)*A.k), C).simplify()
    
    # 断言：验证 C 坐标系的 i 分量是否等于给定表达式的简化结果
    assert C.i == express((cos(q3)*B.i - sin(q3)*B.k), C).simplify()
    
    # 断言：验证 C 坐标系的 k 分量是否等于给定表达式的简化结果
    assert C.k == express((sin(q3)*B.i + cos(q3)*B.k), C).simplify()
# 定义一个测试函数，用于测试 matrix_to_vector 函数的行为
def test_matrix_to_vector():
    # 创建一个 3x1 的矩阵 m，包含向量 [1], [2], [3]
    m = Matrix([[1], [2], [3]])
    # 断言 matrix_to_vector 函数对 m 和坐标系 C 的处理结果应该等于 C.i + 2*C.j + 3*C.k
    assert matrix_to_vector(m, C) == C.i + 2*C.j + 3*C.k

    # 重新赋值矩阵 m，全部为零
    m = Matrix([[0], [0], [0]])
    # 断言 matrix_to_vector 函数对 m 和坐标系 N 的处理结果应该等于零向量
    assert matrix_to_vector(m, N) == matrix_to_vector(m, C) == Vector.zero

    # 创建一个 3x1 的矩阵 m，包含符号 q1, q2, q3
    m = Matrix([[q1], [q2], [q3]])
    # 断言 matrix_to_vector 函数对 m 和坐标系 N 的处理结果应该等于 q1*N.i + q2*N.j + q3*N.k
    assert matrix_to_vector(m, N) == q1*N.i + q2*N.j + q3*N.k


# 定义一个测试函数，用于测试 orthogonalize 函数的行为
def test_orthogonalize():
    # 创建一个三维坐标系 C
    C = CoordSys3D('C')
    # 声明符号变量 a, b 为整数
    a, b = symbols('a b', integer=True)
    # 获取坐标系 C 的基向量 i, j, k
    i, j, k = C.base_vectors()

    # 定义几个向量 v1, v2, ..., v7
    v1 = i + 2*j
    v2 = 2*i + 3*j
    v3 = 3*i + 5*j
    v4 = 3*i + j
    v5 = 2*i + 2*j
    v6 = a*i + b*j
    v7 = 4*a*i + 4*b*j

    # 断言 orthogonalize 函数对 v1 和 v2 的正交化结果应为 [C.i + 2*C.j, C.i*Rational(2, 5) + -C.j/5]
    assert orthogonalize(v1, v2) == [C.i + 2*C.j, C.i*Rational(2, 5) + -C.j/5]

    # 断言 orthogonalize 函数对 v4 和 v5 的正交化结果应为 [(3*sqrt(10))*C.i/10 + (sqrt(10))*C.j/10, (-sqrt(10))*C.i/10 + (3*sqrt(10))*C.j/10]
    # 使用了来自维基百科的示例
    assert orthogonalize(v4, v5, orthonormal=True) == \
        [(3*sqrt(10))*C.i/10 + (sqrt(10))*C.j/10, (-sqrt(10))*C.i/10 + (3*sqrt(10))*C.j/10]

    # 断言 orthogonalize 函数对 v1, v2, v3 的正交化会引发 ValueError 异常
    raises(ValueError, lambda: orthogonalize(v1, v2, v3))

    # 断言 orthogonalize 函数对 v6, v7 的正交化会引发 ValueError 异常
    raises(ValueError, lambda: orthogonalize(v6, v7))
```