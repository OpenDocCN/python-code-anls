# `D:\src\scipysrc\sympy\sympy\algebras\tests\test_quaternion.py`

```
# 从 sympy.testing.pytest 模块导入 slow 装饰器，用于标记耗时的测试
from sympy.testing.pytest import slow
# 从 sympy.core.function 模块导入 diff 函数，用于求解函数的微分
from sympy.core.function import diff
# 从 sympy.core.function 模块导入 expand 函数，用于展开表达式
from sympy.core.function import expand
# 从 sympy.core.numbers 模块导入 E、I、Rational、pi 等常数
from sympy.core.numbers import (E, I, Rational, pi)
# 从 sympy.core.singleton 模块导入 S，用于表示单例值
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块导入 Symbol 和 symbols 函数，用于创建符号变量
from sympy.core.symbol import (Symbol, symbols)
# 从 sympy.functions.elementary.complexes 模块导入 Abs、conjugate、im、re、sign 等函数
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re, sign)
# 从 sympy.functions.elementary.exponential 模块导入 log 函数
from sympy.functions.elementary.exponential import log
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt 函数
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy.functions.elementary.trigonometric 模块导入 acos、asin、cos、sin、atan2、atan 函数
from sympy.functions.elementary.trigonometric import (acos, asin, cos, sin, atan2, atan)
# 从 sympy.integrals.integrals 模块导入 integrate 函数，用于求解积分
from sympy.integrals.integrals import integrate
# 从 sympy.matrices.dense 模块导入 Matrix 类，用于处理密集矩阵
from sympy.matrices.dense import Matrix
# 从 sympy.simplify 模块导入 simplify 函数，用于简化表达式
from sympy.simplify import simplify
# 从 sympy.simplify.trigsimp 模块导入 trigsimp 函数，用于简化三角函数表达式
from sympy.simplify.trigsimp import trigsimp
# 从 sympy.algebras.quaternion 模块导入 Quaternion 类，用于处理四元数
from sympy.algebras.quaternion import Quaternion
# 从 sympy.testing.pytest 模块导入 raises 函数，用于测试是否引发异常
from sympy.testing.pytest import raises
# 导入 math 模块，用于数学计算
import math
# 从 itertools 模块导入 permutations、product 函数，用于生成排列和笛卡尔积

# 创建符号变量 w、x、y、z
w, x, y, z = symbols('w:z')
# 创建符号变量 phi
phi = symbols('phi')

# 定义测试函数 test_quaternion_construction
def test_quaternion_construction():
    # 创建四元数 q，参数为 w、x、y、z
    q = Quaternion(w, x, y, z)
    # 断言四元数 q 加上自身等于四元数 (2*w, 2*x, 2*y, 2*z)
    assert q + q == Quaternion(2*w, 2*x, 2*y, 2*z)

    # 创建四元数 q2，从轴角表示创建，轴向量为 (sqrt(3)/3, sqrt(3)/3, sqrt(3)/3)，角度为 2*pi/3
    q2 = Quaternion.from_axis_angle((sqrt(3)/3, sqrt(3)/3, sqrt(3)/3),
                                    pi*Rational(2, 3))
    # 断言四元数 q2 等于四元数 (1/2, 1/2, 1/2, 1/2)
    assert q2 == Quaternion(S.Half, S.Half,
                            S.Half, S.Half)

    # 创建旋转矩阵 M
    M = Matrix([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])
    # 通过旋转矩阵 M 创建四元数 q3，并进行三角函数简化
    q3 = trigsimp(Quaternion.from_rotation_matrix(M))
    # 断言简化后的四元数 q3 等于特定的四元数表达式
    assert q3 == Quaternion(
        sqrt(2)*sqrt(cos(phi) + 1)/2, 0, 0, sqrt(2 - 2*cos(phi))*sign(sin(phi))/2)

    # 创建非交换符号变量 nc
    nc = Symbol('nc', commutative=False)
    # 断言使用非交换符号变量创建四元数时引发 ValueError 异常
    raises(ValueError, lambda: Quaternion(w, x, nc, z))


# 定义测试函数 test_quaternion_construction_norm
def test_quaternion_construction_norm():
    # 创建四元数 q1，参数为符号变量 a、b、c、d
    q1 = Quaternion(*symbols('a:d'))

    # 创建四元数 q2，参数为 w、x、y、z
    q2 = Quaternion(w, x, y, z)
    # 断言四元数 q1 与 q2 的乘积的模的平方等于 q1 模的平方乘以 q2 模的平方
    assert expand((q1*q2).norm()**2 - (q1.norm()**2 * q2.norm()**2)) == 0

    # 创建单位模的四元数 q3，参数为 w、x、y、z
    q3 = Quaternion(w, x, y, z, norm=1)
    # 断言四元数 q1 与 q3 的乘积的模等于 q1 的模
    assert (q1 * q3).norm() == q1.norm()


# 定义测试函数 test_issue_25254
def test_issue_25254():
    # 创建四元数 p，参数为 (1, 0, 0, 0)
    p = Quaternion(1, 0, 0, 0)
    # 创建轴角表示的四元数 q，轴向量为 (1, 1, 1)，角度为 3*pi/4
    q = Quaternion.from_axis_angle((1, 1, 1), 3 * math.pi/4)
    # 计算四元数 q 的逆 qi
    qi = q.inverse()  # this operation cached the norm
    # 计算测试值 test，即 q * p * qi
    test = q * p * qi
    # 断言测试值 test 与 p 的模小于 1E-10
    assert ((test - p).norm() < 1E-10)


# 定义测试函数 test_to_and_from_Matrix
def test_to_and_from_Matrix():
    # 创建四元数 q，参数为 w、x、y、z
    q = Quaternion(w, x, y, z)
    # 从四元数 q 的矩阵表示创建四元数 q_full
    q_full = Quaternion.from_Matrix(q.to_Matrix())
    # 从四元数 q 的向量部分的矩阵表示创建四元数 q_vect
    q_vect = Quaternion.from_Matrix(q.to_Matrix(True))
    # 断言四元数 q 与 q_full 的差为零四元数
    assert (q - q_full).is_zero_quaternion()
    # 断言四元数 q 的向量部分与 q_vect 的差为零四元数
    assert (q.vector_part() - q_vect).is_zero_quaternion()


# 定义测试函数 test_product_matrices
def test_product_matrices():
    # 创建四元数 q1，参数为 w、x、y、z
    q1 = Quaternion(w, x, y, z)
    # 创建四元数 q2，参数为符号变量 a、b、c、d
    q2 = Quaternion(*(symbols("a:d")))
    # 断言四元数 q1 与 q2 的乘积的矩阵表示等于左乘和右乘矩阵乘积的矩阵表示
    assert (q1 * q2).to_Matrix() == q1.product_matrix_left * q2.to_Matrix()
    assert (q1 * q2).to_Matrix() == q2.product_matrix_right * q1.to_Matrix()

    # 计算 R1 和 R2
    R1 = (q1
    test_data = [ # axis, angle, expected_quaternion
        ((1, 0, 0), 0, (1, 0, 0, 0)),  # 测试数据1：绕x轴旋转0度，预期四元数为(1, 0, 0, 0)
        ((1, 0, 0), pi/2, (sqrt(2)/2, sqrt(2)/2, 0, 0)),  # 测试数据2：绕x轴旋转90度，预期四元数为(sqrt(2)/2, sqrt(2)/2, 0, 0)
        ((0, 1, 0), pi/2, (sqrt(2)/2, 0, sqrt(2)/2, 0)),  # 测试数据3：绕y轴旋转90度，预期四元数为(sqrt(2)/2, 0, sqrt(2)/2, 0)
        ((0, 0, 1), pi/2, (sqrt(2)/2, 0, 0, sqrt(2)/2)),  # 测试数据4：绕z轴旋转90度，预期四元数为(sqrt(2)/2, 0, 0, sqrt(2)/2)
        ((1, 0, 0), pi, (0, 1, 0, 0)),  # 测试数据5：绕x轴旋转180度，预期四元数为(0, 1, 0, 0)
        ((0, 1, 0), pi, (0, 0, 1, 0)),  # 测试数据6：绕y轴旋转180度，预期四元数为(0, 0, 1, 0)
        ((0, 0, 1), pi, (0, 0, 0, 1)),  # 测试数据7：绕z轴旋转180度，预期四元数为(0, 0, 0, 1)
        ((1, 1, 1), pi, (0, 1/sqrt(3),1/sqrt(3),1/sqrt(3))),  # 测试数据8：绕向量(1, 1, 1)旋转180度，预期四元数为(0, 1/sqrt(3),1/sqrt(3),1/sqrt(3))
        ((sqrt(3)/3, sqrt(3)/3, sqrt(3)/3), pi*2/3, (S.Half, S.Half, S.Half, S.Half))  # 测试数据9：绕单位向量(√3/3, √3/3, √3/3)旋转120度，预期四元数为(S.Half, S.Half, S.Half, S.Half)
    ]

    for axis, angle, expected in test_data:
        assert Quaternion.from_axis_angle(axis, angle) == Quaternion(*expected)
def test_quaternion_axis_angle_simplification():
    # 使用给定的轴和角度创建四元数对象
    result = Quaternion.from_axis_angle((1, 2, 3), asin(4))
    # 检查四元数实部是否正确计算
    assert result.a == cos(asin(4)/2)
    # 检查四元数第一个虚部是否正确计算
    assert result.b == sqrt(14)*sin(asin(4)/2)/14
    # 检查四元数第二个虚部是否正确计算
    assert result.c == sqrt(14)*sin(asin(4)/2)/7
    # 检查四元数第三个虚部是否正确计算
    assert result.d == 3*sqrt(14)*sin(asin(4)/2)/14

def test_quaternion_complex_real_addition():
    # 定义复数符号和实数符号
    a = symbols("a", complex=True)
    b = symbols("b", real=True)
    # 此符号不是复数：
    c = symbols("c", commutative=False)

    # 创建一个四元数对象
    q = Quaternion(w, x, y, z)
    # 检查复数和四元数的加法
    assert a + q == Quaternion(w + re(a), x + im(a), y, z)
    # 检查常数和四元数的加法
    assert 1 + q == Quaternion(1 + w, x, y, z)
    # 检查虚数和四元数的加法
    assert I + q == Quaternion(w, 1 + x, y, z)
    # 检查实数和四元数的加法
    assert b + q == Quaternion(w + b, x, y, z)
    # 检查是否会引发值错误异常
    raises(ValueError, lambda: c + q)
    raises(ValueError, lambda: q * c)

    # 检查四元数的负数运算
    assert -q == Quaternion(-w, -x, -y, -z)

    # 创建两个四元数对象
    q1 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
    q2 = Quaternion(1, 4, 7, 8)

    # 检查四元数和复数的加法
    assert q1 + (2 + 3*I) == Quaternion(5 + 7*I, 2 + 5*I, 0, 7 + 8*I)
    # 检查四元数和复数的加法
    assert q2 + (2 + 3*I) == Quaternion(3, 7, 7, 8)
    # 检查四元数和复数的乘法
    assert q1 * (2 + 3*I) == \
        Quaternion((2 + 3*I)*(3 + 4*I), (2 + 3*I)*(2 + 5*I), 0, (2 + 3*I)*(7 + 8*I))
    # 检查四元数和复数的乘法
    assert q2 * (2 + 3*I) == Quaternion(-10, 11, 38, -5)

    # 重新设置四元数对象
    q1 = Quaternion(1, 2, 3, 4)
    q0 = Quaternion(0, 0, 0, 0)
    # 检查四元数加零的结果
    assert q1 + q0 == q1
    # 检查四元数减零的结果
    assert q1 - q0 == q1
    # 检查四元数减自身的结果
    assert q1 - q1 == q0

def test_quaternion_subs():
    # 使用给定的轴和角度创建四元数对象
    q = Quaternion.from_axis_angle((0, 0, 1), phi)
    # 检查替换角度后的四元数对象是否正确
    assert q.subs(phi, 0) == Quaternion(1, 0, 0, 0)

def test_quaternion_evalf():
    # 检查四元数对象的浮点数值计算是否正确
    assert (Quaternion(sqrt(2), 0, 0, sqrt(3)).evalf() ==
            Quaternion(sqrt(2).evalf(), 0, 0, sqrt(3).evalf()))
    # 检查四元数对象的浮点数值计算是否正确
    assert (Quaternion(1/sqrt(2), 0, 0, 1/sqrt(2)).evalf() ==
            Quaternion((1/sqrt(2)).evalf(), 0, 0, (1/sqrt(2)).evalf()))

def test_quaternion_functions():
    # 创建四元数对象
    q = Quaternion(w, x, y, z)
    q1 = Quaternion(1, 2, 3, 4)
    q0 = Quaternion(0, 0, 0, 0)

    # 检查四元数的共轭运算是否正确
    assert conjugate(q) == Quaternion(w, -x, -y, -z)
    # 检查四元数的范数计算是否正确
    assert q.norm() == sqrt(w**2 + x**2 + y**2 + z**2)
    # 检查四元数的归一化计算是否正确
    assert q.normalize() == Quaternion(w, x, y, z) / sqrt(w**2 + x**2 + y**2 + z**2)
    # 检查四元数的逆运算是否正确
    assert q.inverse() == Quaternion(w, -x, -y, -z) / (w**2 + x**2 + y**2 + z**2)
    # 检查四元数的逆运算是否正确
    assert q.inverse() == q.pow(-1)
    # 检查是否会引发值错误异常
    raises(ValueError, lambda: q0.inverse())
    # 检查四元数的幂运算是否正确
    assert q.pow(2) == Quaternion(w**2 - x**2 - y**2 - z**2, 2*w*x, 2*w*y, 2*w*z)
    # 检查四元数的幂运算是否正确
    assert q**(2) == Quaternion(w**2 - x**2 - y**2 - z**2, 2*w*x, 2*w*y, 2*w*z)
    # 检查四元数的负幂运算是否正确
    assert q1.pow(-2) == Quaternion(
        Rational(-7, 225), Rational(-1, 225), Rational(-1, 150), Rational(-2, 225))
    # 检查四元数的负幂运算是否正确
    assert q1**(-2) == Quaternion(
        Rational(-7, 225), Rational(-1, 225), Rational(-1, 150), Rational(-2, 225))
    # 检查四元数的非法负幂运算是否引发类型错误异常
    assert q1.pow(-0.5) == NotImplemented
    raises(TypeError, lambda: q1**(-0.5))

    # 检查四元数的指数函数计算是否正确
    assert q1.exp() == \
    # 创建一个四元数对象，表示特定的旋转，使用平方根29的倍数和给定的E值
    Quaternion(E * cos(sqrt(29)),
               2 * sqrt(29) * E * sin(sqrt(29)) / 29,
               3 * sqrt(29) * E * sin(sqrt(29)) / 29,
               4 * sqrt(29) * E * sin(sqrt(29)) / 29)
    
    # 断言，检查q1的对数是否等于特定四元数对象的对数值
    assert q1.log() == \
    Quaternion(log(sqrt(30)),
               2 * sqrt(29) * acos(sqrt(30)/30) / 29,
               3 * sqrt(29) * acos(sqrt(30)/30) / 29,
               4 * sqrt(29) * acos(sqrt(30)/30) / 29)

    # 断言，检查q1的幂函数cosine和sine值是否等于特定四元数对象的值
    assert q1.pow_cos_sin(2) == \
    Quaternion(30 * cos(2 * acos(sqrt(30)/30)),
               60 * sqrt(29) * sin(2 * acos(sqrt(30)/30)) / 29,
               90 * sqrt(29) * sin(2 * acos(sqrt(30)/30)) / 29,
               120 * sqrt(29) * sin(2 * acos(sqrt(30)/30)) / 29)

    # 断言，检查四元数对象对变量x的微分是否等于特定的四元数对象
    assert diff(Quaternion(x, x, x, x), x) == Quaternion(1, 1, 1, 1)

    # 断言，检查四元数对象对变量x的积分是否等于特定的四元数对象
    assert integrate(Quaternion(x, x, x, x), x) == \
    Quaternion(x**2 / 2, x**2 / 2, x**2 / 2, x**2 / 2)

    # 断言，检查四元数对象对于给定的旋转和点的操作是否得到预期的结果
    assert Quaternion.rotate_point((1, 1, 1), q1) == (S.One / 5, 1, S(7) / 5)

    # 创建一个符号n作为Symbol对象
    n = Symbol('n')
    # 使用lambda表达式在q1上执行n次方操作，期望引发TypeError异常
    raises(TypeError, lambda: q1**n)

    # 创建一个符号n作为整数类型的Symbol对象
    n = Symbol('n', integer=True)
    # 使用lambda表达式在q1上执行n次方操作，期望引发TypeError异常
    raises(TypeError, lambda: q1**n)

    # 断言，检查给定四元数对象的标量部分是否等于预期值
    assert Quaternion(22, 23, 55, 8).scalar_part() == 22
    assert Quaternion(w, x, y, z).scalar_part() == w

    # 断言，检查给定四元数对象的向量部分是否等于预期的四元数对象
    assert Quaternion(22, 23, 55, 8).vector_part() == Quaternion(0, 23, 55, 8)
    assert Quaternion(w, x, y, z).vector_part() == Quaternion(0, x, y, z)

    # 断言，检查四元数对象的轴部分是否等于预期的四元数对象
    assert q1.axis() == Quaternion(0, 2*sqrt(29)/29, 3*sqrt(29)/29, 4*sqrt(29)/29)
    assert q1.axis().pow(2) == Quaternion(-1, 0, 0, 0)
    assert q0.axis().scalar_part() == 0

    # 断言，检查四元数对象的是否为纯四元数
    assert q0.is_pure() is True
    assert q1.is_pure() is False
    assert Quaternion(0, 0, 0, 3).is_pure() is True
    assert Quaternion(0, 2, 10, 3).is_pure() is True
    assert Quaternion(w, 2, 10, 3).is_pure() is None

    # 断言，检查四元数对象的角度是否等于预期的弧度值
    assert q1.angle() == 2*atan(sqrt(29))
    assert q.angle() == 2*atan2(sqrt(x**2 + y**2 + z**2), w)

    # 断言，检查两个四元数对象是否共面
    assert Quaternion.arc_coplanar(q1, Quaternion(2, 4, 6, 8)) is True
    assert Quaternion.arc_coplanar(q1, Quaternion(1, -2, -3, -4)) is True
    assert Quaternion.arc_coplanar(q1, Quaternion(1, 8, 12, 16)) is True
    assert Quaternion.arc_coplanar(q1, Quaternion(1, 2, 3, 4)) is True
    assert Quaternion.arc_coplanar(q1, Quaternion(w, 4, 6, 8)) is True
    assert Quaternion.arc_coplanar(q1, Quaternion(2, 7, 4, 1)) is False
    assert Quaternion.arc_coplanar(q1, Quaternion(w, x, y, z)) is None
    raises(ValueError, lambda: Quaternion.arc_coplanar(q1, q0))

    # 断言，检查三个四元数对象是否共面
    assert Quaternion.vector_coplanar(
        Quaternion(0, 8, 12, 16),
        Quaternion(0, 4, 6, 8),
        Quaternion(0, 2, 3, 4)) is True
    assert Quaternion.vector_coplanar(
        Quaternion(0, 0, 0, 0), Quaternion(0, 4, 6, 8), Quaternion(0, 2, 3, 4)) is True
    # 断言：验证三个四元数是否共面，应该返回False
    assert Quaternion.vector_coplanar(
        Quaternion(0, 8, 2, 6), Quaternion(0, 1, 6, 6), Quaternion(0, 0, 3, 4)) is False

    # 断言：验证三个四元数是否共面，应该返回None
    assert Quaternion.vector_coplanar(
        Quaternion(0, 1, 3, 4),
        Quaternion(0, 4, w, 6),  # 注意：此处w应为一个变量
        Quaternion(0, 6, 8, 1)) is None

    # 断言：验证一个四元数和两个其他四元数的共面关系，应该抛出ValueError异常
    raises(ValueError, lambda:
        Quaternion.vector_coplanar(q0, Quaternion(0, 4, 6, 8), q1))

    # 断言：验证两个四元数是否平行，应该返回True
    assert Quaternion(0, 1, 2, 3).parallel(Quaternion(0, 2, 4, 6)) is True

    # 断言：验证两个四元数是否平行，应该返回False
    assert Quaternion(0, 1, 2, 3).parallel(Quaternion(0, 2, 2, 6)) is False

    # 断言：验证一个四元数和一个具有未定义分量的四元数的平行关系，应该返回None
    assert Quaternion(0, 1, 2, 3).parallel(Quaternion(w, x, y, 6)) is None

    # 断言：验证一个四元数和另一个四元数的平行关系，应该抛出ValueError异常
    raises(ValueError, lambda: q0.parallel(q1))

    # 断言：验证两个四元数是否正交，应该返回True
    assert Quaternion(0, 1, 2, 3).orthogonal(Quaternion(0, -2, 1, 0)) is True

    # 断言：验证两个四元数是否正交，应该返回False
    assert Quaternion(0, 2, 4, 7).orthogonal(Quaternion(0, 2, 2, 6)) is False

    # 断言：验证一个四元数和一个具有未定义分量的四元数的正交关系，应该返回None
    assert Quaternion(0, 2, 4, 7).orthogonal(Quaternion(w, x, y, 6)) is None

    # 断言：验证一个四元数的索引向量计算是否正确
    assert q1.index_vector() == Quaternion(
        0, 2*sqrt(870)/29,
        3*sqrt(870)/29,
        4*sqrt(870)/29)

    # 断言：验证一个已知的四元数的索引向量计算是否正确
    assert Quaternion(0, 3, 9, 4).index_vector() == Quaternion(0, 3, 9, 4)

    # 断言：验证一个四元数的度量计算是否正确
    assert Quaternion(4, 3, 9, 4).mensor() == log(sqrt(122))
    assert Quaternion(3, 3, 0, 2).mensor() == log(sqrt(22))

    # 断言：验证一个四元数是否为零四元数，应该返回True
    assert q0.is_zero_quaternion() is True

    # 断言：验证一个四元数是否为零四元数，应该返回False
    assert q1.is_zero_quaternion() is False

    # 断言：验证一个四元数是否为零四元数，应该返回None
    assert Quaternion(w, 0, 0, 0).is_zero_quaternion() is None
# 定义一个测试函数，用于测试四元数转换的各种方法
def test_quaternion_conversions():
    # 创建一个四元数对象 q1，参数为 (1, 2, 3, 4)
    q1 = Quaternion(1, 2, 3, 4)

    # 断言：验证四元数 q1 转换为轴角表示的正确性
    assert q1.to_axis_angle() == ((2 * sqrt(29)/29,
                                   3 * sqrt(29)/29,
                                   4 * sqrt(29)/29),
                                   2 * acos(sqrt(30)/30))

    # 断言：验证四元数 q1 转换为旋转矩阵表示的正确性
    assert (q1.to_rotation_matrix() ==
            Matrix([[Rational(-2, 3), Rational(2, 15), Rational(11, 15)],
                    [Rational(2, 3), Rational(-1, 3), Rational(2, 3)],
                    [Rational(1, 3), Rational(14, 15), Rational(2, 15)]]))

    # 断言：验证四元数 q1 转换为在指定中心的旋转矩阵表示的正确性
    assert (q1.to_rotation_matrix((1, 1, 1)) ==
            Matrix([
                [Rational(-2, 3), Rational(2, 15), Rational(11, 15), Rational(4, 5)],
                [Rational(2, 3), Rational(-1, 3), Rational(2, 3), S.Zero],
                [Rational(1, 3), Rational(14, 15), Rational(2, 15), Rational(-2, 5)],
                [S.Zero, S.Zero, S.Zero, S.One]]))

    # 定义一个符号变量 theta
    theta = symbols("theta", real=True)
    # 创建一个四元数 q2，使用符号变量 theta 的值
    q2 = Quaternion(cos(theta/2), 0, 0, sin(theta/2))

    # 断言：验证四元数 q2 转换为旋转矩阵表示的正确性（简化后）
    assert trigsimp(q2.to_rotation_matrix()) == Matrix([
                                               [cos(theta), -sin(theta), 0],
                                               [sin(theta),  cos(theta), 0],
                                               [0,           0,          1]])

    # 断言：验证四元数 q2 转换为轴角表示的正确性
    assert q2.to_axis_angle() == ((0, 0, sin(theta/2)/Abs(sin(theta/2))),
                                   2*acos(cos(theta/2)))

    # 断言：验证四元数 q2 转换为在指定中心的旋转矩阵表示的正确性（简化后）
    assert trigsimp(q2.to_rotation_matrix((1, 1, 1))) == Matrix([
               [cos(theta), -sin(theta), 0, sin(theta) - cos(theta) + 1],
               [sin(theta),  cos(theta), 0, -sin(theta) - cos(theta) + 1],
               [0,           0,          1,  0],
               [0,           0,          0,  1]])


# 定义一个测试函数，用于测试旋转矩阵的齐次和非齐次表示的一致性
def test_rotation_matrix_homogeneous():
    # 创建一个四元数对象 q，参数为 (w, x, y, z)
    q = Quaternion(w, x, y, z)
    # 计算并断言：齐次表示的旋转矩阵乘以四元数 q 的模的平方与非齐次表示的旋转矩阵乘以四元数 q 的模的平方的简化结果相等
    R1 = q.to_rotation_matrix(homogeneous=True) * q.norm()**2
    R2 = simplify(q.to_rotation_matrix(homogeneous=False) * q.norm()**2)
    assert R1 == R2


# 定义一个测试函数，用于测试四元数旋转中的问题 #1593
def test_quaternion_rotation_iss1593():
    """
    There was a sign mistake in the definition,
    of the rotation matrix. This tests that particular sign mistake.
    See issue 1593 for reference.
    See wikipedia
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    for the correct definition
    """
    # 创建一个四元数 q，参数为 (cos(phi/2), sin(phi/2), 0, 0)，用于测试旋转矩阵中的符号问题
    q = Quaternion(cos(phi/2), sin(phi/2), 0, 0)
    # 断言：验证修正后的四元数 q 的旋转矩阵表示的正确性
    assert(trigsimp(q.to_rotation_matrix()) == Matrix([
                [1,        0,         0],
                [0, cos(phi), -sin(phi)],
                [0, sin(phi),  cos(phi)]]))


# 定义一个测试函数，用于测试四元数的乘法运算
def test_quaternion_multiplication():
    # 创建一个四元数 q1，参数为 (3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field=False)
    q1 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field=False)
    # 创建一个四元数 q2，参数为 (1, 2, 3, 5)
    q2 = Quaternion(1, 2, 3, 5)
    # 创建一个四元数 q3，参数为 (1, 1, 1, y)
    q3 = Quaternion(1, 1, 1, y)

    # 断言：验证四元数类的静态方法 _generic_mul 的正确性
    assert Quaternion._generic_mul(S(4), S.One) == 4
    # 断言：验证四元数 q1 与标量的乘法运算结果的正确性
    assert (Quaternion._generic_mul(S(4), q1) ==
            Quaternion(12 + 16*I, 8 + 20*I, 0, 28 + 32*I))
    # 断言：验证四元数 q2 与标量的乘法运算结果的正确性
    assert q2.mul(2) == Quaternion(2, 4, 6, 10)
    # 使用乘法操作符 * 计算两个四元数 q2 和 q3 的乘积，并断言结果是否等于指定的四元数
    assert q2.mul(q3) == Quaternion(-5*y - 4, 3*y - 2, 9 - 2*y, y + 4)

    # 使用乘法操作符 * 计算两个四元数 q2 和 q3 的乘积，并断言结果是否等于直接使用 * 运算符计算的结果
    assert q2.mul(q3) == q2*q3

    # 使用 symbols 函数定义复数 z，创建一个只有实部和虚部的四元数 z_quat
    z = symbols('z', complex=True)
    z_quat = Quaternion(re(z), im(z), 0, 0)

    # 使用 symbols 函数定义四个实数符号，创建一个具有这些符号值的四元数 q
    q = Quaternion(*symbols('q:4', real=True))

    # 断言复数 z 乘以四元数 q 的结果是否等于四元数 z_quat 乘以 q 的结果
    assert z * q == z_quat * q

    # 断言四元数 q 乘以复数 z 的结果是否等于 q 乘以四元数 z_quat 的结果
    assert q * z == q * z_quat
def test_issue_163
```