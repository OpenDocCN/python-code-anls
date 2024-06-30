# `D:\src\scipysrc\sympy\sympy\physics\optics\tests\test_utils.py`

```
from sympy.core.numbers import comp, Rational  # 导入复数和有理数相关的模块
from sympy.physics.optics.utils import (refraction_angle, fresnel_coefficients,
        deviation, brewster_angle, critical_angle, lens_makers_formula,
        mirror_formula, lens_formula, hyperfocal_distance,
        transverse_magnification)  # 导入光学相关的实用函数

from sympy.physics.optics.medium import Medium  # 导入光学介质相关的模块
from sympy.physics.units import e0  # 导入单位相关的模块

from sympy.core.numbers import oo  # 导入无穷大的符号
from sympy.core.symbol import symbols  # 导入符号变量相关的模块
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.matrices.dense import Matrix  # 导入矩阵相关的模块
from sympy.geometry.point import Point3D  # 导入三维点相关的模块
from sympy.geometry.line import Ray3D  # 导入三维射线相关的模块
from sympy.geometry.plane import Plane  # 导入平面相关的模块

from sympy.testing.pytest import raises  # 导入测试相关的模块


ae = lambda a, b, n: comp(a, b, 10**-n)  # 定义一个函数 ae，用于比较两个复数的大小精确到 10 的负 n 次方


def test_refraction_angle():
    n1, n2 = symbols('n1, n2')  # 定义符号变量 n1 和 n2
    m1 = Medium('m1')  # 创建光学介质对象 m1
    m2 = Medium('m2')  # 创建光学介质对象 m2
    r1 = Ray3D(Point3D(-1, -1, 1), Point3D(0, 0, 0))  # 创建一个三维射线对象 r1
    i = Matrix([1, 1, 1])  # 创建一个 3x1 的矩阵对象 i
    n = Matrix([0, 0, 1])  # 创建一个 3x1 的矩阵对象 n，表示法线向量
    normal_ray = Ray3D(Point3D(0, 0, 0), Point3D(0, 0, 1))  # 创建一个法线射线对象 normal_ray
    P = Plane(Point3D(0, 0, 0), normal_vector=[0, 0, 1])  # 创建一个平面对象 P，法线向量为 [0, 0, 1]

    # 测试光线折射角函数 refraction_angle 的不同输入情况
    assert refraction_angle(r1, 1, 1, n) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle([1, 1, 1], 1, 1, n) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle((1, 1, 1), 1, 1, n) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle(i, 1, 1, [0, 0, 1]) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle(i, 1, 1, (0, 0, 1)) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle(i, 1, 1, normal_ray) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle(i, 1, 1, plane=P) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle(r1, 1, 1, plane=P) == \
        Ray3D(Point3D(0, 0, 0), Point3D(1, 1, -1))
    assert refraction_angle(r1, m1, 1.33, plane=P) == \
        Ray3D(Point3D(0, 0, 0), Point3D(Rational(100, 133), Rational(100, 133), -789378201649271*sqrt(3)/1000000000000000))
    assert refraction_angle(r1, 1, m2, plane=P) == \
        Ray3D(Point3D(0, 0, 0), Point3D(1, 1, -1))
    # 断言：计算折射角度，检查是否与预期的射线相符
    assert refraction_angle(r1, n1, n2, plane=P) == \
        Ray3D(Point3D(0, 0, 0), Point3D(n1/n2, n1/n2, -sqrt(3)*sqrt(-2*n1**2/(3*n2**2) + 1)))
    
    # 断言：计算折射角度，期望结果为0，表示全反射(TIR)
    assert refraction_angle(r1, 1.33, 1, plane=P) == 0  # TIR
    
    # 断言：计算折射角度，考虑垂直光线的情况
    assert refraction_angle(r1, 1, 1, normal_ray) == \
        Ray3D(Point3D(0, 0, 0), direction_ratio=[1, 1, -1])
    
    # 断言：验证计算出的折射角度是否接近预期值，精度为小数点后5位
    assert ae(refraction_angle(0.5, 1, 2), 0.24207, 5)
    
    # 断言：验证计算出的折射角度是否接近预期值，精度为小数点后5位
    assert ae(refraction_angle(0.5, 2, 1), 1.28293, 5)
    
    # 引发异常：抛出值错误异常，期望参数引发异常
    raises(ValueError, lambda: refraction_angle(r1, m1, m2, normal_ray, P))
    
    # 引发异常：抛出类型错误异常，期望参数引发异常
    raises(TypeError, lambda: refraction_angle(m1, m1, m2)) # can add other values for arg[0]
    
    # 引发异常：抛出类型错误异常，期望参数引发异常
    raises(TypeError, lambda: refraction_angle(r1, m1, m2, None, i))
    
    # 引发异常：抛出类型错误异常，期望参数引发异常
    raises(TypeError, lambda: refraction_angle(r1, m1, m2, m2))
# 定义测试函数，用于测试菲涅尔系数计算函数的正确性
def test_fresnel_coefficients():
    # 断言检查：使用所有元素近似相等函数 ae() 对菲涅尔系数的计算结果进行检查
    assert all(ae(i, j, 5) for i, j in zip(
        fresnel_coefficients(0.5, 1, 1.33),   # 计算折射率为 1 到 1.33 的菲涅尔系数
        [0.11163, -0.17138, 0.83581, 0.82862]))  # 期望的菲涅尔系数结果

    # 断言检查：使用所有元素近似相等函数 ae() 对不同折射率情况下的菲涅尔系数进行检查
    assert all(ae(i, j, 5) for i, j in zip(
        fresnel_coefficients(0.5, 1.33, 1),   # 计算折射率为 1.33 到 1 的菲涅尔系数
        [-0.07726, 0.20482, 1.22724, 1.20482]))  # 期望的菲涅尔系数结果

    # 创建介质对象
    m1 = Medium('m1')
    m2 = Medium('m2', n=2)

    # 断言检查：使用所有元素近似相等函数 ae() 对介质 m1 到 m2 之间的菲涅尔系数进行检查
    assert all(ae(i, j, 5) for i, j in zip(
        fresnel_coefficients(0.3, m1, m2),   # 计算介质 m1 到 m2 之间的菲涅尔系数
        [0.31784, -0.34865, 0.65892, 0.65135]))  # 期望的菲涅尔系数结果

    # 预期的结果
    ans = [[-0.23563, -0.97184], [0.81648, -0.57738]]
    
    # 计算介质 m2 到 m1 之间的菲涅尔系数
    got = fresnel_coefficients(0.6, m2, m1)

    # 遍历结果并使用所有元素近似相等函数 ae() 进行检查
    for i, j in zip(got, ans):
        for a, b in zip(i.as_real_imag(), j):
            assert ae(a, b, 5)


# 定义测试函数，用于测试光线偏折计算的正确性
def test_deviation():
    # 定义符号变量 n1, n2
    n1, n2 = symbols('n1, n2')

    # 创建光线对象 r1，方向向量为 (-1, -1, -1)，起始点为 (-1, -1, 1)
    r1 = Ray3D(Point3D(-1, -1, 1), Point3D(0, 0, 0))

    # 定义法向量 n 为 (0, 0, 1)，光线方向向量 i 为 (-1, -1, -1)
    n = Matrix([0, 0, 1])
    i = Matrix([-1, -1, -1])

    # 创建法线光线对象 normal_ray，起始点为 (0, 0, 0)，方向向量为 (0, 0, 1)
    normal_ray = Ray3D(Point3D(0, 0, 0), Point3D(0, 0, 1))

    # 创建平面对象 P，法向量为 [0, 0, 1]，过点 (0, 0, 0)
    P = Plane(Point3D(0, 0, 0), normal_vector=[0, 0, 1])

    # 断言检查：使用 deviation() 函数计算光线 r1 在给定折射率和法线 n 下的偏折角是否为 0
    assert deviation(r1, 1, 1, normal=n) == 0

    # 断言检查：使用 deviation() 函数计算光线 r1 在给定折射率和平面 P 下的偏折角是否为 0
    assert deviation(r1, 1, 1, plane=P) == 0

    # 断言检查：使用 deviation() 函数计算光线 r1 在给定折射率和平面 P 下的偏折角是否接近给定值
    assert deviation(r1, 1, 1.1, plane=P).evalf(3) + 0.119 < 1e-3

    # 断言检查：使用 deviation() 函数计算光线 i 在给定折射率和法线 normal_ray 下的偏折角是否接近给定值
    assert deviation(i, 1, 1.1, normal=normal_ray).evalf(3) + 0.119 < 1e-3

    # 断言检查：使用 deviation() 函数计算光线 r1 在给定折射率和平面 P 下的偏折角是否为 None（全反射情况）
    assert deviation(r1, 1.33, 1, plane=P) is None

    # 断言检查：使用 deviation() 函数计算光线 r1 在给定折射率和法线 [0, 0, 1] 下的偏折角是否为 0
    assert deviation(r1, 1, 1, normal=[0, 0, 1]) == 0

    # 断言检查：使用 deviation() 函数计算光线 [-1, -1, -1] 在给定折射率和法线 [0, 0, 1] 下的偏折角是否为 0
    assert deviation([-1, -1, -1], 1, 1, normal=[0, 0, 1]) == 0

    # 断言检查：使用近似相等函数 ae() 检查 deviation() 函数计算的结果是否接近预期值
    assert ae(deviation(0.5, 1, 2), -0.25793, 5)

    # 断言检查：使用近似相等函数 ae() 检查 deviation() 函数计算的结果是否接近预期值
    assert ae(deviation(0.5, 2, 1), 0.78293, 5)


# 定义测试函数，用于测试布儒斯特角的计算
def test_brewster_angle():
    # 创建介质对象 m1, m2，折射率分别为 1 和 1.33
    m1 = Medium('m1', n=1)
    m2 = Medium('m2', n=1.33)

    # 断言检查：使用近似相等函数 ae() 检查 brewster_angle() 函数计算的布儒斯特角是否接近预期值
    assert ae(brewster_angle(m1, m2), 0.93, 2)

    # 创建介质对象 m1, m2，带有介电常数 e0，折射率分别为 1 和 1.33
    m1 = Medium('m1', permittivity=e0, n=1)
    m2 = Medium('m2', permittivity=e0, n=1.33)

    # 断言检查：使用近似相等函数 ae() 检查 brewster_angle() 函数计算的布儒斯特角是否接近预期值
    assert ae(brewster_angle(m1, m2), 0.93, 2)

    # 断言检查：使用近似相等函数 ae() 检查 brewster_angle() 函数计算的布儒斯特角是否接近预期值
    assert ae(brewster_angle(1, 1.33), 0.93, 2)


# 定义测试函数，用于测试临界角的计算
def test_critical_angle():
    # 创建介质对象 m1, m2，折射率分别为 1 和 1.33
    m1 = Medium('m1', n=1)
    # 使用无穷远焦距和物距 u 来验证镜公式计算，期望结果为 -u
    assert mirror_formula(focal_length=oo, u=u) == -u
    
    # 使用给定的焦距 f、物距 u 和像距 v 来调用镜公式计算函数，预期会引发 ValueError 异常
    raises(ValueError, lambda: mirror_formula(focal_length=f, u=u, v=v))
# 定义一个测试函数，用于测试透镜公式函数 lens_formula 的不同参数组合
def test_lens_formula():
    # 使用符号创建变量 u, v, f
    u, v, f = symbols('u, v, f')
    # 断言计算并验证透镜公式 lens_formula 的结果
    assert lens_formula(focal_length=f, u=u) == f*u/(f + u)
    assert lens_formula(focal_length=f, v=v) == f*v/(f - v)
    assert lens_formula(u=u, v=v) == u*v/(u - v)
    assert lens_formula(u=oo, v=v) == v
    assert lens_formula(u=oo, v=oo) is oo
    assert lens_formula(focal_length=oo, u=u) == u
    assert lens_formula(u=u, v=oo) == -u
    assert lens_formula(focal_length=oo, v=oo) is -oo
    assert lens_formula(focal_length=oo, v=v) == v
    assert lens_formula(focal_length=f, v=oo) == -f
    assert lens_formula(focal_length=oo, u=oo) is oo
    assert lens_formula(focal_length=oo, u=u) == u
    assert lens_formula(focal_length=f, u=oo) == f
    # 断言引发 ValueError 异常，检测 lens_formula 在非法参数组合下的行为
    raises(ValueError, lambda: lens_formula(focal_length=f, u=u, v=v))


# 定义一个测试函数，用于测试超焦距函数 hyperfocal_distance 的计算
def test_hyperfocal_distance():
    # 使用符号创建变量 f, N, c
    f, N, c = symbols('f, N, c')
    # 断言计算并验证超焦距函数 hyperfocal_distance 的结果
    assert hyperfocal_distance(f=f, N=N, c=c) == f**2/(N*c)
    # 断言使用 assert_almost_equal 函数验证计算结果是否在指定精度范围内
    assert ae(hyperfocal_distance(f=0.5, N=8, c=0.0033), 9.47, 2)


# 定义一个测试函数，用于测试横向放大倍数函数 transverse_magnification 的计算
def test_transverse_magnification():
    # 使用符号创建变量 si, so
    si, so = symbols('si, so')
    # 断言计算并验证横向放大倍数函数 transverse_magnification 的结果
    assert transverse_magnification(si, so) == -si/so
    assert transverse_magnification(30, 15) == -2


# 定义一个测试函数，用于测试透镜制造者公式在厚透镜情况下的计算
def test_lens_makers_formula_thick_lens():
    # 使用符号创建变量 n1, n2
    n1, n2 = symbols('n1, n2')
    # 创建介质对象 m1, m2，并指定其参数
    m1 = Medium('m1', permittivity=e0, n=1)
    m2 = Medium('m2', permittivity=e0, n=1.33)
    # 断言计算并验证透镜制造者公式 lens_makers_formula 在厚透镜情况下的结果
    assert ae(lens_makers_formula(m1, m2, 10, -10, d=1), -19.82, 2)
    assert lens_makers_formula(n1, n2, 1, -1, d=0.1) == n2/((2.0 - (0.1*n1 - 0.1*n2)/n1)*(n1 - n2))


# 定义一个测试函数，用于测试透镜制造者公式在平板透镜情况下的计算
def test_lens_makers_formula_plano_lens():
    # 使用符号创建变量 n1, n2
    n1, n2 = symbols('n1, n2')
    # 创建介质对象 m1, m2，并指定其参数
    m1 = Medium('m1', permittivity=e0, n=1)
    m2 = Medium('m2', permittivity=e0, n=1.33)
    # 断言计算并验证透镜制造者公式 lens_makers_formula 在平板透镜情况下的结果
    assert ae(lens_makers_formula(m1, m2, 10, oo), -40.30, 2)
    assert lens_makers_formula(n1, n2, 10, oo) == 10.0*n2/(n1 - n2)
```