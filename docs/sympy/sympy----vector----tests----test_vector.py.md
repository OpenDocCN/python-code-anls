# `D:\src\scipysrc\sympy\sympy\vector\tests\test_vector.py`

```
from sympy.core import Rational, S, Add, Mul  # 导入有理数、符号、加法和乘法操作
from sympy.simplify import simplify, trigsimp  # 导入简化和三角函数简化函数
from sympy.core.function import (Derivative, Function, diff)  # 导入导数、函数和求导函数
from sympy.core.numbers import pi  # 导入π
from sympy.core.symbol import symbols  # 导入符号变量
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入余弦和正弦函数
from sympy.integrals.integrals import Integral  # 导入积分函数
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix  # 导入不可变密集矩阵
from sympy.vector.vector import Vector, BaseVector, VectorAdd, \
     VectorMul, VectorZero  # 导入向量相关类
from sympy.vector.coordsysrect import CoordSys3D  # 导入三维坐标系类
from sympy.vector.vector import Cross, Dot, cross  # 导入向量的叉乘和点乘操作
from sympy.testing.pytest import raises  # 导入测试框架中的异常处理函数
from sympy.vector.kind import VectorKind  # 导入向量类型
from sympy.core.kind import NumberKind  # 导入数值类型

C = CoordSys3D('C')  # 创建一个名为C的三维坐标系

i, j, k = C.base_vectors()  # 获取坐标系C的基向量
a, b, c = symbols('a b c')  # 创建符号变量a, b, c

def test_cross():
    """
    测试向量的叉乘操作
    """
    v1 = C.x * i + C.z * C.z * j  # 构造向量v1
    v2 = C.x * i + C.y * j + C.z * k  # 构造向量v2
    assert Cross(v1, v2) == Cross(C.x*C.i + C.z**2*C.j, C.x*C.i + C.y*C.j + C.z*C.k)
    assert Cross(v1, v2).doit() == C.z**3*C.i + (-C.x*C.z)*C.j + (C.x*C.y - C.x*C.z**2)*C.k
    assert cross(v1, v2) == C.z**3*C.i + (-C.x*C.z)*C.j + (C.x*C.y - C.x*C.z**2)*C.k
    assert Cross(v1, v2) == -Cross(v2, v1)
    assert Cross(v1, v2) + Cross(v2, v1) == Vector.zero

def test_dot():
    """
    测试向量的点乘操作
    """
    v1 = C.x * i + C.z * C.z * j  # 构造向量v1
    v2 = C.x * i + C.y * j + C.z * k  # 构造向量v2
    assert Dot(v1, v2) == Dot(C.x*C.i + C.z**2*C.j, C.x*C.i + C.y*C.j + C.z*C.k)
    assert Dot(v1, v2).doit() == C.x**2 + C.y*C.z**2
    assert Dot(v1, v2).doit() == C.x**2 + C.y*C.z**2
    assert Dot(v1, v2) == Dot(v2, v1)

def test_vector_sympy():
    """
    测试向量是否符合SymPy的哈希和相等性测试属性
    """
    v1 = 3*j
    assert v1 == j*3
    assert v1.components == {j: 3}
    v2 = 3*i + 4*j + 5*k
    v3 = 2*i + 4*j + i + 4*k + k
    assert v3 == v2
    assert v3.__hash__() == v2.__hash__()

def test_kind():
    """
    测试向量和数值类型的关系
    """
    assert C.i.kind is VectorKind(NumberKind)
    assert C.j.kind is VectorKind(NumberKind)
    assert C.k.kind is VectorKind(NumberKind)

    assert C.x.kind is NumberKind
    assert C.y.kind is NumberKind
    assert C.z.kind is NumberKind

    assert Mul._kind_dispatcher(NumberKind, VectorKind(NumberKind)) is VectorKind(NumberKind)
    assert Mul(2, C.i).kind is VectorKind(NumberKind)

    v1 = C.x * i + C.z * C.z * j
    v2 = C.x * i + C.y * j + C.z * k
    assert v1.kind is VectorKind(NumberKind)
    assert v2.kind is VectorKind(NumberKind)

    assert (v1 + v2).kind is VectorKind(NumberKind)
    assert Add(v1, v2).kind is VectorKind(NumberKind)
    assert Cross(v1, v2).doit().kind is VectorKind(NumberKind)
    assert VectorAdd(v1, v2).kind is VectorKind(NumberKind)
    assert VectorMul(2, v1).kind is VectorKind(NumberKind)
    assert VectorZero().kind is VectorKind(NumberKind)

    assert v1.projection(v2).kind is VectorKind(NumberKind)
    assert v2.projection(v1).kind is VectorKind(NumberKind)
def test_vectoradd():
    # 检查 Add(C.i, C.j) 的实例是否为 VectorAdd 类型
    assert isinstance(Add(C.i, C.j), VectorAdd)
    # 创建向量 v1 和 v2
    v1 = C.x * i + C.z * C.z * j
    v2 = C.x * i + C.y * j + C.z * k
    # 检查 Add(v1, v2) 的实例是否为 VectorAdd 类型
    assert isinstance(Add(v1, v2), VectorAdd)

    # https://github.com/sympy/sympy/issues/26121

    # 创建矩阵 E 和向量 a，并计算 E*a
    E = Matrix([C.i, C.j, C.k]).T
    a = Matrix([1, 2, 3])
    av = E*a

    # 检查 av 的第一个元素的种类是否为 VectorKind
    assert av[0].kind == VectorKind()
    # 检查 av 的第一个元素是否为 VectorAdd 类型


def test_vector():
    # 检查 i 是否为 BaseVector 的实例
    assert isinstance(i, BaseVector)
    # 检查 i 是否不等于 j
    assert i != j
    # 检查 j 是否不等于 k
    assert j != k
    # 检查 k 是否不等于 i
    assert k != i
    # 检查 i - i 是否等于 Vector.zero
    assert i - i == Vector.zero
    # 检查 i + Vector.zero 是否等于 i
    assert i + Vector.zero == i
    # 检查 i - Vector.zero 是否等于 i
    assert i - Vector.zero == i
    # 检查 Vector.zero 是否不等于 0
    assert Vector.zero != 0
    # 检查 -Vector.zero 是否等于 Vector.zero
    assert -Vector.zero == Vector.zero

    # 创建向量 v1
    v1 = a*i + b*j + c*k
    # 创建向量 v2、v3、v4、v5
    v2 = a**2*i + b**2*j + c**2*k
    v3 = v1 + v2
    v4 = 2 * v1
    v5 = a * i

    # 检查 v1 是否为 VectorAdd 类型
    assert isinstance(v1, VectorAdd)
    # 检查 v1 - v1 是否等于 Vector.zero
    assert v1 - v1 == Vector.zero
    # 检查 v1 + Vector.zero 是否等于 v1
    assert v1 + Vector.zero == v1
    # 检查 v1.dot(i) 是否等于 a
    assert v1.dot(i) == a
    # 检查 v1.dot(j) 是否等于 b
    assert v1.dot(j) == b
    # 检查 v1.dot(k) 是否等于 c
    assert v1.dot(k) == c
    # 检查 i.dot(v2) 是否等于 a**2
    assert i.dot(v2) == a**2
    # 检查 j.dot(v2) 是否等于 b**2
    assert j.dot(v2) == b**2
    # 检查 k.dot(v2) 是否等于 c**2
    assert k.dot(v2) == c**2
    # 检查 v3.dot(i) 是否等于 a**2 + a
    assert v3.dot(i) == a**2 + a
    # 检查 v3.dot(j) 是否等于 b**2 + b
    assert v3.dot(j) == b**2 + b
    # 检查 v3.dot(k) 是否等于 c**2 + c
    assert v3.dot(k) == c**2 + c

    # 检查 v1 + v2 是否等于 v2 + v1
    assert v1 + v2 == v2 + v1
    # 检查 v1 - v2 是否等于 -1 * (v2 - v1)
    assert v1 - v2 == -1 * (v2 - v1)
    # 检查 a * v1 是否等于 v1 * a
    assert a * v1 == v1 * a

    # 检查 v5 是否为 VectorMul 类型
    assert isinstance(v5, VectorMul)
    # 检查 v5 的基向量是否为 i
    assert v5.base_vector == i
    # 检查 v5 的测量数是否为 a
    assert v5.measure_number == a
    # 检查 v4 是否为 Vector 类型
    assert isinstance(v4, Vector)
    # 检查 v4 是否为 VectorAdd 类型
    assert isinstance(v4, VectorAdd)
    # 检查 v4 是否为 Vector 类型
    assert isinstance(v4, Vector)
    # 检查 Vector.zero 是否为 VectorZero 类型
    assert isinstance(Vector.zero, VectorZero)
    # 检查 Vector.zero 是否为 Vector 类型
    assert isinstance(Vector.zero, Vector)
    # 检查 v1 * 0 是否为 VectorZero 类型
    assert isinstance(v1 * 0, VectorZero)

    # 检查 v1 转换为矩阵时基于 C 的结果
    assert v1.to_matrix(C) == Matrix([[a], [b], [c]])

    # 检查 i 的分量是否为 {i: 1}
    assert i.components == {i: 1}
    # 检查 v5 的分量是否为 {i: a}
    assert v5.components == {i: a}
    # 检查 v1 的分量是否为 {i: a, j: b, k: c}

    assert v1.components == {i: a, j: b, k: c}

    # 检查 VectorAdd(v1, Vector.zero) 是否等于 v1
    assert VectorAdd(v1, Vector.zero) == v1
    # 检查 VectorMul(a, v1) 是否等于 v1 * a
    assert VectorMul(a, v1) == v1*a
    # 检查 VectorMul(1, i) 是否等于 i
    assert VectorMul(1, i) == i
    # 检查 VectorAdd(v1, Vector.zero) 是否等于 v1
    assert VectorAdd(v1, Vector.zero) == v1
    # 检查 VectorMul(0, Vector.zero) 是否等于 Vector.zero
    assert VectorMul(0, Vector.zero) == Vector.zero
    # 检查 v1.outer(1) 是否引发 TypeError 异常
    raises(TypeError, lambda: v1.outer(1))
    # 检查 v1.dot(1) 是否引发 TypeError 异常
    raises(TypeError, lambda: v1.dot(1))


def test_vector_magnitude_normalize():
    # 检查 Vector.zero 的大小是否为 0
    assert Vector.zero.magnitude() == 0
    # 检查 Vector.zero 的归一化是否等于 Vector.zero

    assert Vector.zero.normalize() == Vector.zero

    # 检查 i 的大小是否为 1
    assert i.magnitude() == 1
    # 检查 j 的大小是否为 1
    assert j.magnitude() == 1
    # 检查 k 的大小是否为 1
    assert k.magnitude() == 1
    # 检查 i 的归一化是否等于 i
    assert i.normalize() == i
    # 检查 j 的归一化是否等于 j
    assert j.normalize() == j
    # 检查 k 的归一化是否等于 k
    assert k.normalize() == k

    # 创建向量 v1
    v1 = a * i
    # 检查 v1 的归一化是否等于 (a/sqrt(a**2))*i
    assert v1.normalize() == (a/sqrt(a**2))*i
    # 检查 v1 的大小是否等于 sqrt(a**2)
    assert v1.magnitude() == sqrt(a**2)

    # 创建向量 v2
    v2 = a*i + b*j + c*k
    # 检查 v2 的大小是否等于 sqrt(a**2 + b**2 + c**2)
    assert v2.magnitude() == sqrt(a**2 + b**2 + c**2)
    # 检查 v2 的归一化是否等于 v2 / v2 的大小
    assert v2.normalize() == v2 / v2.magnitude()

    # 创建向量 v3
    v3 = i + j
    # 检查 v3 的归一化是否等于 (sqrt(2)/2)*C.i + (sqrt(2)/2)*C.j


def test_vector_s
    # 对 test2 进行化简操作
    test2 = simplify(test2)
    # 断言检查化简后的表达式是否与预期相等
    assert (test2 & i) == (A**2 * s**4 / (4 * pi * k * m**3))

    # 计算 test3 的表达式
    test3 = ((4 + 4 * a - 2 * (2 + 2 * a)) / (2 + 2 * a)) * i
    # 对 test3 进行化简操作
    test3 = simplify(test3)
    # 断言检查化简后的表达式是否为零
    assert (test3 & i) == 0

    # 计算 test4 的表达式
    test4 = ((-4 * a * b**2 - 2 * b**3 - 2 * a**2 * b) / (a + b)**2) * i
    # 对 test4 进行化简操作
    test4 = simplify(test4)
    # 断言检查化简后的表达式是否等于 -2 * b
    assert (test4 & i) == -2 * b

    # 计算 v 的表达式
    v = (sin(a)+cos(a))**2*i - j
    # 断言检查三角函数表达式的化简结果是否符合预期
    assert trigsimp(v) == (2*sin(a + pi/4)**2)*i + (-1)*j
    # 断言检查通过方法调用的方式是否得到相同的化简结果
    assert trigsimp(v) == v.trigsimp()

    # 断言检查零向量的化简结果是否为零向量
    assert simplify(Vector.zero) == Vector.zero
def test_vector_dot():
    # 检查 i 向量与零向量的点积，预期结果为 0
    assert i.dot(Vector.zero) == 0
    # 检查零向量与 i 向量的点积，预期结果为 0
    assert Vector.zero.dot(i) == 0
    # 检查 i 向量与零向量的按位与操作，预期结果为 0
    assert i & Vector.zero == 0

    # 检查 i 向量与自身的点积，预期结果为 1
    assert i.dot(i) == 1
    # 检查 i 向量与 j 向量的点积，预期结果为 0
    assert i.dot(j) == 0
    # 检查 i 向量与 k 向量的点积，预期结果为 0
    assert i.dot(k) == 0
    # 检查 i 向量与自身的按位与操作，预期结果为 1
    assert i & i == 1
    # 检查 i 向量与 j 向量的按位与操作，预期结果为 0
    assert i & j == 0
    # 检查 i 向量与 k 向量的按位与操作，预期结果为 0
    assert i & k == 0

    # 检查 j 向量与 i 向量的点积，预期结果为 0
    assert j.dot(i) == 0
    # 检查 j 向量与自身的点积，预期结果为 1
    assert j.dot(j) == 1
    # 检查 j 向量与 k 向量的点积，预期结果为 0
    assert j.dot(k) == 0
    # 检查 j 向量与 i 向量的按位与操作，预期结果为 0
    assert j & i == 0
    # 检查 j 向量与自身的按位与操作，预期结果为 1
    assert j & j == 1
    # 检查 j 向量与 k 向量的按位与操作，预期结果为 0
    assert j & k == 0

    # 检查 k 向量与 i 向量的点积，预期结果为 0
    assert k.dot(i) == 0
    # 检查 k 向量与 j 向量的点积，预期结果为 0
    assert k.dot(j) == 0
    # 检查 k 向量与自身的点积，预期结果为 1
    assert k.dot(k) == 1
    # 检查 k 向量与 i 向量的按位与操作，预期结果为 0
    assert k & i == 0
    # 检查 k 向量与 j 向量的按位与操作，预期结果为 0
    assert k & j == 0
    # 检查 k 向量与自身的按位与操作，预期结果为 1
    assert k & k == 1

    # 检查 k 向量与整数 1 的点积操作，预期会引发 TypeError 异常
    raises(TypeError, lambda: k.dot(1))


def test_vector_cross():
    # 检查 i 向量与零向量的叉乘，预期结果为零向量
    assert i.cross(Vector.zero) == Vector.zero
    # 检查零向量与 i 向量的叉乘，预期结果为零向量
    assert Vector.zero.cross(i) == Vector.zero

    # 检查 i 向量与自身的叉乘，预期结果为零向量
    assert i.cross(i) == Vector.zero
    # 检查 i 向量与 j 向量的叉乘，预期结果为 k 向量
    assert i.cross(j) == k
    # 检查 i 向量与 k 向量的叉乘，预期结果为 -j 向量
    assert i.cross(k) == -j
    # 检查 i 向量与自身的按位异或操作，预期结果为零向量
    assert i ^ i == Vector.zero
    # 检查 i 向量与 j 向量的按位异或操作，预期结果为 k 向量
    assert i ^ j == k
    # 检查 i 向量与 k 向量的按位异或操作，预期结果为 -j 向量
    assert i ^ k == -j

    # 检查 j 向量与 i 向量的叉乘，预期结果为 -k 向量
    assert j.cross(i) == -k
    # 检查 j 向量与自身的叉乘，预期结果为零向量
    assert j.cross(j) == Vector.zero
    # 检查 j 向量与 k 向量的叉乘，预期结果为 i 向量
    assert j.cross(k) == i
    # 检查 j 向量与 i 向量的按位异或操作，预期结果为 -k 向量
    assert j ^ i == -k
    # 检查 j 向量与自身的按位异或操作，预期结果为零向量
    assert j ^ j == Vector.zero
    # 检查 j 向量与 k 向量的按位异或操作，预期结果为 i 向量
    assert j ^ k == i

    # 检查 k 向量与 i 向量的叉乘，预期结果为 j 向量
    assert k.cross(i) == j
    # 检查 k 向量与 j 向量的叉乘，预期结果为 -i 向量
    assert k.cross(j) == -i
    # 检查 k 向量与自身的叉乘，预期结果为零向量
    assert k.cross(k) == Vector.zero
    # 检查 k 向量与 i 向量的按位异或操作，预期结果为 j 向量
    assert k ^ i == j
    # 检查 k 向量与 j 向量的按位异或操作，预期结果为 -i 向量
    assert k ^ j == -i
    # 检查 k 向量与自身的按位异或操作，预期结果为零向量
    assert k ^ k == Vector.zero

    # 检查 k 向量与整数 1 的叉乘操作，预期结果为 Cross(k, 1)
    assert k.cross(1) == Cross(k, 1)


def test_projection():
    v1 = i + j + k
    v2 = 3*i + 4*j
    v3 = 0*i + 0*j
    # 检查 v1 在自身上的投影，预期结果为 v1 向量本身
    assert v1.projection(v1) == i + j + k
    # 检查 v1 在 v2 上的投影，预期结果为 Rational(7, 3)*C.i + Rational(7, 3)*C.j + Rational(7, 3)*C.k
    assert v1.projection(v2) == Rational(7, 3)*C.i + Rational(7, 3)*C.j + Rational(7, 3)*C.k
    # 检查 v1 在自身上的标量投影，预期结果为 1
    assert v1.projection(v1, scalar=True) == S.One
    # 检查 v1 在 v2 上的标量投影，预期结果为 Rational(7, 3)
    assert v1.projection(v2, scalar=True) == Rational(7, 3)
    # 检查 v3 在 v1 上的投影，预期结果为零向量
    assert v3.projection(v1) == Vector.zero
    # 检查 v3 在 v1 上的标量投影，预期结果为 0
    assert v3.projection(v1, scalar=True) == S.Zero


def test_vector_diff_integrate():
    f = Function('f')
    v = f(a)*C.i + a**2*C.j - C.k
    # 检查向量 v 对变量 a 的导
```