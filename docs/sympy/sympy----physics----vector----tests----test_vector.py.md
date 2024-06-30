# `D:\src\scipysrc\sympy\sympy\physics\vector\tests\test_vector.py`

```
from sympy.core.numbers import (Float, pi)  # 导入 Float 和 pi 符号
from sympy.core.symbol import symbols  # 导入 symbols 符号
from sympy.core.sorting import ordered  # 导入 ordered 函数
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入 cos 和 sin 函数
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix  # 导入 ImmutableDenseMatrix 并重命名为 Matrix
from sympy.physics.vector import ReferenceFrame, Vector, dynamicsymbols, dot  # 导入 ReferenceFrame, Vector, dynamicsymbols, dot
from sympy.physics.vector.vector import VectorTypeError  # 导入 VectorTypeError 异常类
from sympy.abc import x, y, z  # 导入符号 x, y, z
from sympy.testing.pytest import raises  # 导入 raises 函数

A = ReferenceFrame('A')  # 创建一个名为 A 的参考系


def test_free_dynamicsymbols():
    A, B, C, D = symbols('A, B, C, D', cls=ReferenceFrame)  # 创建符号 A, B, C, D 作为参考系
    a, b, c, d, e, f = dynamicsymbols('a, b, c, d, e, f')  # 创建动力学符号 a, b, c, d, e, f
    B.orient_axis(A, a, A.x)  # 在参考系 A 的 x 轴上用角度 a 定向参考系 B
    C.orient_axis(B, b, B.y)  # 在参考系 B 的 y 轴上用角度 b 定向参考系 C
    D.orient_axis(C, c, C.x)  # 在参考系 C 的 x 轴上用角度 c 定向参考系 D

    v = d*D.x + e*D.y + f*D.z  # 创建向量 v

    assert set(ordered(v.free_dynamicsymbols(A))) == {a, b, c, d, e, f}  # 断言向量 v 中 A 参考系的自由动力学符号
    assert set(ordered(v.free_dynamicsymbols(B))) == {b, c, d, e, f}  # 断言向量 v 中 B 参考系的自由动力学符号
    assert set(ordered(v.free_dynamicsymbols(C))) == {c, d, e, f}  # 断言向量 v 中 C 参考系的自由动力学符号
    assert set(ordered(v.free_dynamicsymbols(D))) == {d, e, f}  # 断言向量 v 中 D 参考系的自由动力学符号


def test_Vector():
    assert A.x != A.y  # 断言 A 参考系的 x 轴不等于 y 轴
    assert A.y != A.z  # 断言 A 参考系的 y 轴不等于 z 轴
    assert A.z != A.x  # 断言 A 参考系的 z 轴不等于 x 轴

    assert A.x + 0 == A.x  # 断言 A 参考系的 x 轴加 0 等于 x 轴本身

    v1 = x*A.x + y*A.y + z*A.z  # 创建向量 v1
    v2 = x**2*A.x + y**2*A.y + z**2*A.z  # 创建向量 v2
    v3 = v1 + v2  # 创建向量 v3，为 v1 和 v2 的和
    v4 = v1 - v2  # 创建向量 v4，为 v1 和 v2 的差

    assert isinstance(v1, Vector)  # 断言 v1 是 Vector 类的实例
    assert dot(v1, A.x) == x  # 断言 v1 与 A 参考系的 x 轴的点积结果是 x
    assert dot(v1, A.y) == y  # 断言 v1 与 A 参考系的 y 轴的点积结果是 y
    assert dot(v1, A.z) == z  # 断言 v1 与 A 参考系的 z 轴的点积结果是 z

    assert isinstance(v2, Vector)  # 断言 v2 是 Vector 类的实例
    assert dot(v2, A.x) == x**2  # 断言 v2 与 A 参考系的 x 轴的点积结果是 x 的平方
    assert dot(v2, A.y) == y**2  # 断言 v2 与 A 参考系的 y 轴的点积结果是 y 的平方
    assert dot(v2, A.z) == z**2  # 断言 v2 与 A 参考系的 z 轴的点积结果是 z 的平方

    assert isinstance(v3, Vector)  # 断言 v3 是 Vector 类的实例
    # 断言 v3 与 A 参考系的 x、y、z 轴的点积结果
    assert dot(v3, A.x) == x**2 + x
    assert dot(v3, A.y) == y**2 + y
    assert dot(v3, A.z) == z**2 + z

    assert isinstance(v4, Vector)  # 断言 v4 是 Vector 类的实例
    # 断言 v4 与 A 参考系的 x、y、z 轴的点积结果
    assert dot(v4, A.x) == x - x**2
    assert dot(v4, A.y) == y - y**2
    assert dot(v4, A.z) == z - z**2

    assert v1.to_matrix(A) == Matrix([[x], [y], [z]])  # 断言 v1 转换到 A 参考系的矩阵形式
    q = symbols('q')  # 创建符号 q
    B = A.orientnew('B', 'Axis', (q, A.x))  # 创建名为 B 的参考系，由 A 参考系绕其 x 轴旋转 q 角度
    # 断言 v1 转换到 B 参考系的矩阵形式
    assert v1.to_matrix(B) == Matrix([[x],
                                      [ y * cos(q) + z * sin(q)],
                                      [-y * sin(q) + z * cos(q)]])

    # 测试 separate 方法
    B = ReferenceFrame('B')  # 创建名为 B 的参考系
    v5 = x*A.x + y*A.y + z*B.z  # 创建向量 v5
    assert Vector(0).separate() == {}  # 断言零向量的 separate 方法结果为空字典
    assert v1.separate() == {A: v1}  # 断言 v1 的 separate 方法结果为包含 A 参考系的字典
    assert v5.separate() == {A: x*A.x + y*A.y, B: z*B.z}  # 断言 v5 的 separate 方法结果为包含 A 和 B 参考系的字典

    # 测试 free_symbols 属性
    v6 = x*A.x + y*A.y + z*A.z  # 创建向量 v6
    assert v6.free_symbols(A) == {x,y,z}  # 断言 v6 在 A 参考系下的自由符号集合为 {x, y, z}

    raises(TypeError, lambda: v3.applyfunc(v1))  # 断言在应用 v1 的 applyfunc 方法到 v3 时抛出 TypeError 异常


def test_Vector_diffs():
    q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')  # 创建动力学符号 q1, q2, q3, q4
    q1d, q2d, q3d, q4d = dynamicsymbols('q1 q2 q3 q4', 1)  # 创建动力学符号 q1d, q2d, q3d, q4d 的一阶导数
    q1dd, q2dd, q3dd, q4dd = dynamicsymbols('q1 q2 q3 q4', 2)  # 创建动力学符号 q1dd, q2dd, q3dd, q4dd 的二阶导数
    N = ReferenceFrame('N')  # 创建名为 N 的参考系
    A = N.orientnew('A', 'Axis', [q3, N.z])  # 创建名为 A 的参考
    # 计算向量 v5 在坐标系 N 中的分量
    v5 = q1*A.x + q2*A.y + q3*A.z

    # 验证向量 v1 在坐标系 N 中的时间导数
    assert v1.dt(N) == q2d * A.x + q2 * q3d * A.y + q3d * N.y
    # 验证向量 v1 在坐标系 A 中的时间导数
    assert v1.dt(A) == q2d * A.x + q3 * q3d * N.x + q3d * N.y
    # 验证向量 v1 在坐标系 B 中的时间导数
    assert v1.dt(B) == (q2d * A.x + q3 * q3d * N.x + q3d * N.y - q3 * cos(q3) * q2d * N.z)

    # 验证向量 v2 在坐标系 N 中的时间导数
    assert v2.dt(N) == (q2d * A.x + (q2 + q3) * q3d * A.y + q3d * B.x + q3d * N.y)
    # 验证向量 v2 在坐标系 A 中的时间导数
    assert v2.dt(A) == q2d * A.x + q3d * B.x + q3 * q3d * N.x + q3d * N.y
    # 验证向量 v2 在坐标系 B 中的时间导数
    assert v2.dt(B) == (q2d * A.x + q3d * B.x + q3 * q3d * N.x + q3d * N.y - q3 * cos(q3) * q2d * N.z)

    # 验证向量 v3 在坐标系 N 中的时间导数
    assert v3.dt(N) == (q2dd * A.x + q2d * q3d * A.y + (q3d**2 + q3 * q3dd) * N.x + q3dd * N.y +
                        (q3 * sin(q3) * q2d * q3d - cos(q3) * q2d * q3d - q3 * cos(q3) * q2dd) * N.z)
    # 验证向量 v3 在坐标系 A 中的时间导数
    assert v3.dt(A) == (q2dd * A.x + (2 * q3d**2 + q3 * q3dd) * N.x + (q3dd - q3 * q3d**2) * N.y +
                        (q3 * sin(q3) * q2d * q3d - cos(q3) * q2d * q3d - q3 * cos(q3) * q2dd) * N.z)
    # 验证向量 v3 在坐标系 B 中的时间导数，并化简为零
    assert (v3.dt(B) - (q2dd*A.x - q3*cos(q3)*q2d**2*A.y + (2*q3d**2 + q3*q3dd)*N.x +
        (q3dd - q3*q3d**2)*N.y + (q3*sin(q3)*q2d*q3d - cos(q3)*q2d*q3d - q3*cos(q3)*q2dd)*N.z)).express(B).simplify() == 0

    # 验证向量 v4 在坐标系 N 中的时间导数
    assert v4.dt(N) == (q2dd * A.x + q3d * (q2d + q3d) * A.y + q3dd * B.x +
                        (q3d**2 + q3 * q3dd) * N.x + q3dd * N.y + (q3 * sin(q3) * q2d * q3d - cos(q3) * q2d * q3d - q3 * cos(q3) * q2dd) * N.z)
    # 验证向量 v4 在坐标系 A 中的时间导数
    assert v4.dt(A) == (q2dd * A.x + q3dd * B.x + (2 * q3d**2 + q3 * q3dd) * N.x +
                        (q3dd - q3 * q3d**2) * N.y + (q3 * sin(q3) * q2d * q3d - cos(q3) * q2d * q3d - q3 * cos(q3) * q2dd) * N.z)
    # 验证向量 v4 在坐标系 B 中的时间导数，并化简为零
    assert (v4.dt(B) - (q2dd*A.x - q3*cos(q3)*q2d**2*A.y + q3dd*B.x +
                        (2*q3d**2 + q3*q3dd)*N.x + (q3dd - q3*q3d**2)*N.y +
                        (2*q3*sin(q3)*q2d*q3d - 2*cos(q3)*q2d*q3d - q3*cos(q3)*q2dd)*N.z)).express(B).simplify() == 0

    # 验证向量 v5 在坐标系 B 中的时间导数
    assert v5.dt(B) == q1d*A.x + (q3*q2d + q2d)*A.y + (-q2*q2d + q3d)*A.z
    # 验证向量 v5 在坐标系 A 中的时间导数
    assert v5.dt(A) == q1d*A.x + q2d*A.y + q3d*A.z
    # 验证向量 v5 在坐标系 N 中的时间导数
    assert v5.dt(N) == (-q2*q3d + q1d)*A.x + (q1*q3d + q2d)*A.y + q3d*A.z

    # 验证向量 v3 关于 q1d 在坐标系 N 的偏导数
    assert v3.diff(q1d, N) == 0
    # 验证向量 v3 关于 q2d 在坐标系 N 的偏导数
    assert v3.diff(q2d, N) == A.x - q3 * cos(q3) * N.z
    # 验证向量 v3 关于 q3d 在坐标系 N 的偏导数
    assert v3.diff(q3d, N) == q3 * N.x + N.y

    # 验证向量 v3 关于 q1d 在坐标系 A 的偏导数
    assert v3.diff(q1d, A) == 0
    # 验证向量 v3 关于 q2d 在坐标系 A 的偏导数
    assert v3.diff(q2d, A) == A.x - q3 * cos(q3) * N.z
    # 验证向量 v3 关于 q3d 在坐标系 A 的偏导数
    assert v3.diff(q3d, A) == q3 * N.x + N.y

    # 验证向量 v3 关于 q1d 在坐标系 B 的偏导数
    assert v3.diff(q1d, B) == 0
    # 验证向量 v3 关于 q2d 在坐标系 B 的偏导数
    assert v3.diff(q2d, B) == A.x - q3 * cos(q3) * N.z
    # 验证向量 v3 关于 q3d 在坐标系 B 的偏导数
    assert v3.diff(q3d, B) == q3 * N.x + N.y

    # 验证向量 v4 关于 q1d 在坐标系 N 的偏导数
    assert v4.diff(q1d, N) == 0
    # 验证向量 v4 关于 q2d 在坐标系 N 的偏导数
    assert v4.diff(q2d, N) == A.x - q3 * cos(q3) * N.z
    # 验证向量 v4 关于 q3d 在坐标系 N 的偏导数
    assert v4.diff(q3d, N) == B.x + q3 * N.x + N.y

    # 验证向量 v4 关于 q1d 在坐标系 A 的偏导数
    assert v4.diff(q1d, A) == 0
    # 验证向量 v4 关于 q2d 在坐标系 A 的偏导数
    assert v4.diff(q2d, A) == A.x - q3 * cos(q3) * N.z
    # 验证向量 v
    # 断言表达式，验证 v4 对 q2d 的偏导数是否等于 A.x - q3 * cos(q3) * N.z
    assert v4.diff(q2d, B) == A.x - q3 * cos(q3) * N.z
    # 断言表达式，验证 v4 对 q3d 的偏导数是否等于 B.x + q3 * N.x + N.y
    assert v4.diff(q3d, B) == B.x + q3 * N.x + N.y

    # diff() 函数应仅在导数框架的方向上表达向量分量，如果组件的框架取决于变量
    v6 = q2**2*N.y + q2**2*A.y + q2**2*B.y
    # 已经在 N 框架中表达
    n_measy = 2*q2
    # A_C_N 不依赖于 q2，因此不在 N 框架中表达
    a_measy = 2*q2
    # B_C_N 依赖于 q2，因此在 N 框架中表达
    b_measx = (q2**2*B.y).dot(N.x).diff(q2)
    b_measy = (q2**2*B.y).dot(N.y).diff(q2)
    b_measz = (q2**2*B.y).dot(N.z).diff(q2)
    # 计算 v6 对 q2 在 N 框架中的偏导数
    n_comp, a_comp = v6.diff(q2, N).args
    # 断言表达式，验证 v6 对 q2 在 N 框架中的偏导数是否只有 N 部分和 A 部分
    assert len(v6.diff(q2, N).args) == 2  # 只有 N 和 A 部分
    # 断言表达式，验证 n_comp 的第二个元素是否为 N
    assert n_comp[1] == N
    # 断言表达式，验证 a_comp 的第二个元素是否为 A
    assert a_comp[1] == A
    # 断言表达式，验证 n_comp 的第一个元素是否与 Matrix([b_measx, b_measy + n_measy, b_measz]) 相等
    assert n_comp[0] == Matrix([b_measx, b_measy + n_measy, b_measz])
    # 断言表达式，验证 a_comp 的第一个元素是否与 Matrix([0, a_measy, 0]) 相等
    assert a_comp[0] == Matrix([0, a_measy, 0])
# 定义一个测试函数，用于测试向量在参考框架中的各种操作
def test_vector_var_in_dcm():

    # 创建三个参考框架对象，分别命名为N、A、B
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')

    # 定义四个动态符号u1, u2, u3, u4
    u1, u2, u3, u4 = dynamicsymbols('u1 u2 u3 u4')

    # 定义向量v，包含四个分量的线性组合
    v = u1 * u2 * A.x + u3 * N.y + u4**2 * N.z

    # 断言：计算v对u1在参考框架N中的偏导数，不考虑导数计算的直接余弦矩阵 (var_in_dcm=False)
    assert v.diff(u1, N, var_in_dcm=False) == u2 * A.x
    # 断言：计算v对u1在参考框架A中的偏导数，不考虑导数计算的直接余弦矩阵 (var_in_dcm=False)
    assert v.diff(u1, A, var_in_dcm=False) == u2 * A.x
    # 断言：计算v对u3在参考框架N中的偏导数，不考虑导数计算的直接余弦矩阵 (var_in_dcm=False)
    assert v.diff(u3, N, var_in_dcm=False) == N.y
    # 断言：计算v对u3在参考框架A中的偏导数，不考虑导数计算的直接余弦矩阵 (var_in_dcm=False)
    assert v.diff(u3, A, var_in_dcm=False) == N.y
    # 断言：计算v对u3在参考框架B中的偏导数，不考虑导数计算的直接余弦矩阵 (var_in_dcm=False)
    assert v.diff(u3, B, var_in_dcm=False) == N.y
    # 断言：计算v对u4在参考框架N中的偏导数，不考虑导数计算的直接余弦矩阵 (var_in_dcm=False)
    assert v.diff(u4, N, var_in_dcm=False) == 2 * u4 * N.z

    # 断言：验证v对u1在参考框架N中的偏导数，此处会引发 ValueError
    raises(ValueError, lambda: v.diff(u1, N))


# 定义一个测试函数，用于测试向量的简化操作
def test_vector_simplify():
    # 定义符号变量
    x, y, z, k, n, m, w, f, s, A = symbols('x, y, z, k, n, m, w, f, s, A')
    # 创建一个参考框架对象N
    N = ReferenceFrame('N')

    # 定义测试向量test1，并对其进行简化操作
    test1 = (1 / x + 1 / y) * N.x
    # 断言：简化前后向量test1在N.x方向上的比较
    assert (test1 & N.x) != (x + y) / (x * y)
    test1 = test1.simplify()
    # 断言：简化后向量test1在N.x方向上的比较
    assert (test1 & N.x) == (x + y) / (x * y)

    # 定义测试向量test2，并对其进行简化操作
    test2 = (A**2 * s**4 / (4 * pi * k * m**3)) * N.x
    test2 = test2.simplify()
    # 断言：简化后向量test2在N.x方向上的比较
    assert (test2 & N.x) == (A**2 * s**4 / (4 * pi * k * m**3))

    # 定义测试向量test3，并对其进行简化操作
    test3 = ((4 + 4 * x - 2 * (2 + 2 * x)) / (2 + 2 * x)) * N.x
    test3 = test3.simplify()
    # 断言：简化后向量test3在N.x方向上的比较
    assert (test3 & N.x) == 0

    # 定义测试向量test4，并对其进行简化操作
    test4 = ((-4 * x * y**2 - 2 * y**3 - 2 * x**2 * y) / (x + y)**2) * N.x
    test4 = test4.simplify()
    # 断言：简化后向量test4在N.x方向上的比较
    assert (test4 & N.x) == -2 * y


# 定义一个测试函数，用于测试向量的数值化操作
def test_vector_evalf():
    # 定义符号变量a, b
    a, b = symbols('a b')
    # 创建一个向量v，包含pi乘以A.x的线性组合
    v = pi * A.x
    # 断言：计算向量v在小数精度2下的数值化结果
    assert v.evalf(2) == Float('3.1416', 2) * A.x
    # 创建一个向量v，包含pi乘以A.x加上5乘以a乘以A.y再减去b乘以A.z的线性组合
    v = pi * A.x + 5 * a * A.y - b * A.z
    # 断言：计算向量v在小数精度3下的数值化结果
    assert v.evalf(3) == Float('3.1416', 3) * A.x + Float('5', 3) * a * A.y - b * A.z
    # 断言：计算向量v在小数精度5下，且用a替换为1，b替换为5.8973的数值化结果
    assert v.evalf(5, subs={a: 1.234, b: 5.8973}) == Float('3.1415926536', 5) * A.x + Float('6.17', 5) * A.y - Float('5.8973', 5) * A.z


# 定义一个测试函数，用于测试向量之间的角度计算
def test_vector_angle():
    # 创建一个参考框架对象A
    A = ReferenceFrame('A')
    # 创建两个向量v1和v2
    v1 = A.x + A.y
    v2 = A.z
    # 断言：计算向量v1和v2之间的夹角，期望结果为pi/2
    assert v1.angle_between(v2) == pi/2
    # 创建一个参考框架对象B，并使B相对于A绕A.x轴旋转180度
    B = ReferenceFrame('B')
    B.orient_axis(A, A.x, pi)
    # 创建两个向量v3和v4
    v3 = A.x
    v4 = B.x
    # 断言：计算向量v3和v4之间的夹角，期望结果为0
    assert v3.angle_between(v4) == 0


# 定义一个测试函数，用于测试向量的替换操作
def test_vector_xreplace():
    # 定义符号变量x, y, z
    x, y, z = symbols('x y z')
    # 创建一个向量v，包含x的平方乘以A.x加上x乘以y乘以A.y再加上x乘以y乘以z乘以A.z的线性组合
    v = x**2 * A.x + x*y * A.y + x*y*z * A.z
    # 断言：用cos(x)替换向量v中的x后的结果
    assert v.xreplace({x: cos(x)}) == cos(x)**2 * A.x + y*cos(x) * A.y + y*z*cos(x) * A.z
    # 断言：用pi替换向量v中的x*y后的结果
    # 创建一个3x3的矩阵v2v1，其元素为对应的数值乘积
    v2v1 = Matrix([[d*a, d*b, d*c],
                   [e*a, e*b, e*c],
                   [f*a, f*b, f*c]])
    # 使用向量v1的外积生成一个矩阵，并断言其与v2v1相等
    assert v2.outer(v1).to_matrix(N) == v2v1
    # 使用向量v1和v2的点积生成一个矩阵，并断言其与v2v1相等
    assert (v2 | v1).to_matrix(N) == v2v1
# 定义一个测试函数，用于测试重载运算符的行为
def test_overloaded_operators():
    # 定义符号变量 a, b, c, d, e, f
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    # 创建一个参考坐标系 'N'
    N = ReferenceFrame('N')
    # 创建向量 v1，包含向量分量 a*N.x, b*N.y, c*N.z
    v1 = a*N.x + b*N.y + c*N.z
    # 创建向量 v2，包含向量分量 d*N.x, e*N.y, f*N.z
    v2 = d*N.x + e*N.y + f*N.z

    # 测试向量加法的交换律
    assert v1 + v2 == v2 + v1
    # 测试向量减法的交换律
    assert v1 - v2 == -v2 + v1
    # 测试向量点积（内积）的交换律
    assert v1 & v2 == v2 & v1
    # 测试向量叉积（外积），使用 ^ 表示，等同于 v1.cross(v2)
    assert v1 ^ v2 == v1.cross(v2)
    # 测试叉积的反交换性质，等同于 v2.cross(v1)
    assert v2 ^ v1 == v2.cross(v1)
```