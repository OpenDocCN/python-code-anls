# `D:\src\scipysrc\sympy\sympy\vector\tests\test_dyadic.py`

```
# 导入需要的符号和函数
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.vector import (CoordSys3D, Vector, Dyadic,
                          DyadicAdd, DyadicMul, DyadicZero,
                          BaseDyadic, express)

# 创建三维坐标系 A
A = CoordSys3D('A')

# 定义测试函数 test_dyadic
def test_dyadic():
    # 定义符号变量 a 和 b
    a, b = symbols('a, b')

    # 断言 Dyadic.zero 不等于数值 0
    assert Dyadic.zero != 0

    # 断言 Dyadic.zero 是 DyadicZero 类的实例
    assert isinstance(Dyadic.zero, DyadicZero)

    # 断言不同向量对应的 BaseDyadic 对象不相等
    assert BaseDyadic(A.i, A.j) != BaseDyadic(A.j, A.i)

    # 断言特定向量与零向量生成的 BaseDyadic 对象与 Dyadic.zero 相等
    assert (BaseDyadic(Vector.zero, A.i) ==
            BaseDyadic(A.i, Vector.zero) == Dyadic.zero)

    # 创建两个同向向量的 Dyadic 对象 d1 和 d2，以及一个不同向量的 Dyadic 对象 d3
    d1 = A.i | A.i
    d2 = A.j | A.j
    d3 = A.i | A.j

    # 断言 d1 是 BaseDyadic 类的实例
    assert isinstance(d1, BaseDyadic)

    # 创建 DyadicMul 对象 d_mul，并断言其是 DyadicMul 类的实例
    d_mul = a * d1
    assert isinstance(d_mul, DyadicMul)

    # 断言 d_mul 的 base_dyadic 属性等于 d1
    assert d_mul.base_dyadic == d1

    # 断言 d_mul 的 measure_number 属性等于 a
    assert d_mul.measure_number == a

    # 断言 a*d1 + b*d3 是 DyadicAdd 类的实例
    assert isinstance(a*d1 + b*d3, DyadicAdd)

    # 断言 d1 和 A.i 外积的结果与 A.i.outer(A.i) 相等
    assert d1 == A.i.outer(A.i)

    # 断言 d3 和 A.i、A.j 外积的结果与 A.i.outer(A.j) 相等
    assert d3 == A.i.outer(A.j)

    # 创建两个向量 v1 和 v2
    v1 = a * A.i - A.k
    v2 = A.i + b * A.j

    # 断言 v1 和 v2 的外积等于给定表达式
    assert v1 | v2 == v1.outer(v2) == a * (A.i | A.i) + (a * b) * (A.i | A.j) + \
           - (A.k | A.i) - b * (A.k | A.j)

    # 断言 d1 乘以零得到 Dyadic.zero
    assert d1 * 0 == Dyadic.zero

    # 断言 d1 不等于 Dyadic.zero
    assert d1 != Dyadic.zero

    # 断言 d1 乘以 2 等于 2 倍的 A.i 外积 A.i
    assert d1 * 2 == 2 * (A.i | A.i)

    # 断言 d1 除以 2. 等于 0.5 倍的 d1
    assert d1 / 2. == 0.5 * d1

    # 断言 d1 和 0 倍的 d1 的点积结果为 Vector.zero
    assert d1.dot(0 * d1) == Vector.zero

    # 断言 d1 和 d2 的点积结果为 Dyadic.zero
    assert d1 & d2 == Dyadic.zero

    # 断言 d1 和 A.i 的点积结果等于 A.i，以及等价的表达式
    assert d1.dot(A.i) == A.i == d1 & A.i

    # 断言 d1 和 0 向量的叉积结果为 Dyadic.zero
    assert d1.cross(Vector.zero) == Dyadic.zero

    # 断言 d1 和 A.i 的叉积结果为 Dyadic.zero
    assert d1.cross(A.i) == Dyadic.zero

    # 断言 d1 和 A.j 的叉积结果等于 d1 ^ A.j
    assert d1 ^ A.j == d1.cross(A.j)

    # 断言 d1 和 A.k 的叉积结果等于 - A.i | A.j
    assert d1.cross(A.k) == - A.i | A.j

    # 断言 d2 和 A.i 的叉积结果等于 - A.j | A.k，以及等价的表达式
    assert d2.cross(A.i) == - A.j | A.k == d2 ^ A.i

    # 断言 A.i 和 d1 的叉积结果等于 Dyadic.zero
    assert A.i ^ d1 == Dyadic.zero

    # 断言 A.j 和 d1 的叉积结果等于 - A.k | A.i，以及等价的表达式
    assert A.j.cross(d1) == - A.k | A.i == A.j ^ d1

    # 断言 Vector.zero 和 d1 的叉积结果为 Dyadic.zero
    assert Vector.zero.cross(d1) == Dyadic.zero

    # 断言 A.k 和 d1 的叉积结果等于 A.j | A.i
    assert A.k ^ d1 == A.j | A.i

    # 断言 A.i 和 d1 的点积结果等于 A.i 和 d1 的 & 运算结果，都等于 A.i
    assert A.i.dot(d1) == A.i & d1 == A.i

    # 断言 A.j 和 d1 的点积结果等于 Vector.zero
    assert A.j.dot(d1) == Vector.zero

    # 断言 Vector.zero 和 d1 的点积结果为 Vector.zero
    assert Vector.zero.dot(d1) == Vector.zero

    # 断言 A.j 和 d2 的 & 运算结果等于 A.j
    assert A.j & d2 == A.j

    # 断言 d1 和 d3 的点积结果等于 d1 和 d3 的 & 运算结果，都等于 A.i | A.j
    assert d1.dot(d3) == d1 & d3 == A.i | A.j == d3

    # 断言 d3 和 d1 的 & 运算结果等于 Dyadic.zero
    assert d3 & d1 == Dyadic.zero

    # 定义符号变量 q
    q = symbols('q')

    # 在 A 坐标系上根据旋转轴 A.k 创建新的坐标系 B
    B = A.orient_new_axis('B', q, A.k)

    # 断言在坐标系 B 中表达 d1 的结果与在 B 坐标系中表达 d1 和 B 相等
    assert express(d1, B) == express(d1, B, B)

    # 创建表达式 expr1，表示在坐标系 B 中表达 d1 的结果
    expr1 = ((cos(q)**2) * (B.i | B.i) + (-sin(q) * cos(q)) *
            (B.i | B.j) + (-sin(q) * cos(q)) * (B.j | B.i) + (sin(q)**2) *
            (B.j | B.j))

    # 断言表达式 express(d1, B) - expr1 简化后等于 Dyadic.zero
    assert (express(d1, B) - expr1).simplify() == Dyadic.zero

    # 创建表达式 expr2，表示在坐标系 B 和 A 中表达 d1 的结果
    expr2 = (cos(q)) * (B.i | A.i) + (-sin(q)) * (B.j | A.i)

    # 断言表达式 express(d1, B, A) - expr2 简化后等于 Dyadic.zero
    assert (express(d1, B, A) - expr2).simplify() == Dyadic.zero

    # 创建表达式 expr3，表示在坐标系 A 和 B 中表达 d1 的结果
    expr3 = (cos(q)) * (A.i | B.i) + (-sin(q)) * (A.i |
    # 定义符号变量 a, b, c, d, e, f
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    # 创建向量 v1 和 v2，使用符号变量 a, b, c, d, e, f 来定义
    v1 = a * A.i + b * A.j + c * A.k
    v2 = d * A.i + e * A.j + f * A.k
    # 计算向量 v1 和 v2 的外积
    d4 = v1.outer(v2)
    # 断言外积 d4 转换为基于 A 的矩阵与预期的矩阵相等
    assert d4.to_matrix(A) == Matrix([[a * d, a * e, a * f],
                                      [b * d, b * e, b * f],
                                      [c * d, c * e, c * f]])
    # 计算向量 v1 和自身的外积
    d5 = v1.outer(v1)
    # 基于向量 v1 和旋转轴 A.i 创建新的坐标系 C
    C = A.orient_new_axis('C', q, A.i)
    # 遍历 C 相对于 A 的旋转矩阵乘以 d5 转换为矩阵再乘以其转置的结果以及 d5 转换为 C 坐标系的矩阵
    for expected, actual in zip(C.rotation_matrix(A) * d5.to_matrix(A) * \
                               C.rotation_matrix(A).T, d5.to_matrix(C)):
        # 断言两者之间的简化差为 0
        assert (expected - actual).simplify() == 0
# 定义测试函数 test_dyadic_simplify
def test_dyadic_simplify():
    # 定义符号变量
    x, y, z, k, n, m, w, f, s, A = symbols('x, y, z, k, n, m, w, f, s, A')
    # 创建三维坐标系对象 N
    N = CoordSys3D('N')

    # 创建 dyadic 对象 dy，表示 N.i 和 N.i 的外积
    dy = N.i | N.i
    # 创建测试表达式 test1，包含符号变量 x 和 y 的分数和 dyadic 对象 dy
    test1 = (1 / x + 1 / y) * dy
    # 断言语句，验证 dy 和表达式 (x + y) / (x * y) 不相等
    assert (N.i & test1 & N.i) != (x + y) / (x * y)
    # 简化 test1 表达式
    test1 = test1.simplify()
    # 断言语句，验证简化后的 test1 和 simplify(test1) 相等
    assert test1.simplify() == simplify(test1)
    # 断言语句，验证 dy 和简化后的 test1 表达式为 (x + y) / (x * y)
    assert (N.i & test1 & N.i) == (x + y) / (x * y)

    # 创建包含符号变量和常数的表达式 test2，并赋值给 dyadic 对象 test2
    test2 = (A**2 * s**4 / (4 * pi * k * m**3)) * dy
    # 简化 test2 表达式
    test2 = test2.simplify()
    # 断言语句，验证 dy 和简化后的 test2 表达式相等
    assert (N.i & test2 & N.i) == (A**2 * s**4 / (4 * pi * k * m**3))

    # 创建包含符号变量的表达式 test3，并赋值给 dyadic 对象 test3
    test3 = ((4 + 4 * x - 2 * (2 + 2 * x)) / (2 + 2 * x)) * dy
    # 简化 test3 表达式
    test3 = test3.simplify()
    # 断言语句，验证 dy 和简化后的 test3 表达式为 0
    assert (N.i & test3 & N.i) == 0

    # 创建包含符号变量的表达式 test4，并赋值给 dyadic 对象 test4
    test4 = ((-4 * x * y**2 - 2 * y**3 - 2 * x**2 * y) / (x + y)**2) * dy
    # 简化 test4 表达式
    test4 = test4.simplify()
    # 断言语句，验证 dy 和简化后的 test4 表达式为 -2 * y
    assert (N.i & test4 & N.i) == -2 * y


# 定义测试函数 test_dyadic_srepr
def test_dyadic_srepr():
    # 导入 srepr 函数
    from sympy.printing.repr import srepr
    # 创建三维坐标系对象 N
    N = CoordSys3D('N')

    # 创建 dyadic 对象 dy，表示 N.i 和 N.j 的外积
    dy = N.i | N.j
    # 预期的 dyadic 对象字符串表示
    res = "BaseDyadic(CoordSys3D(Str('N'), Tuple(ImmutableDenseMatrix([["\
        "Integer(1), Integer(0), Integer(0)], [Integer(0), Integer(1), "\
        "Integer(0)], [Integer(0), Integer(0), Integer(1)]]), "\
        "VectorZero())).i, CoordSys3D(Str('N'), Tuple(ImmutableDenseMatrix("\
        "[[Integer(1), Integer(0), Integer(0)], [Integer(0), Integer(1), "\
        "Integer(0)], [Integer(0), Integer(0), Integer(1)]]), VectorZero())).j)"
    # 断言语句，验证 dy 的字符串表示与预期字符串 res 相等
    assert srepr(dy) == res
```