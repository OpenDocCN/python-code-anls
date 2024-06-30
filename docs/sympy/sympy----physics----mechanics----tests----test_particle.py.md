# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_particle.py`

```
# 导入必要的符号和力学库函数
from sympy import symbols
from sympy.physics.mechanics import Point, Particle, ReferenceFrame, inertia
from sympy.physics.mechanics.body_base import BodyBase
from sympy.testing.pytest import raises, warns_deprecated_sympy


# 定义测试函数：测试粒子默认属性
def test_particle_default():
    # 创建粒子对象 'P'
    p = Particle('P')
    # 断言粒子名称为 'P'
    assert p.name == 'P'
    # 断言粒子质量符号为 'P_mass'
    assert p.mass == symbols('P_mass')
    # 断言质心名称为 'P_masscenter'
    assert p.masscenter.name == 'P_masscenter'
    # 断言势能为 0
    assert p.potential_energy == 0
    # 断言粒子的字符串表示为 'P'
    assert p.__str__() == 'P'
    # 断言粒子的正式字符串表示
    assert p.__repr__() == ("Particle('P', masscenter=P_masscenter, "
                            "mass=P_mass)")
    # 使用 lambda 表达式断言调用 p.frame 会引发 AttributeError 异常
    raises(AttributeError, lambda: p.frame)


# 定义测试函数：测试粒子初始化和属性设置
def test_particle():
    # 定义符号变量
    m, m2, v1, v2, v3, r, g, h = symbols('m m2 v1 v2 v3 r g h')
    # 创建点 P 和 P2
    P = Point('P')
    P2 = Point('P2')
    # 创建粒子对象 'pa'，指定质点 P 和质量 m
    p = Particle('pa', P, m)
    # 断言 p 是 BodyBase 的实例
    assert isinstance(p, BodyBase)
    # 断言粒子的质量为 m
    assert p.mass == m
    # 断言粒子的点为 P
    assert p.point == P
    # 测试质量设置器
    p.mass = m2
    assert p.mass == m2
    # 测试点设置器
    p.point = P2
    assert p.point == P2
    # 创建参考系 N 和点 O
    N = ReferenceFrame('N')
    O = Point('O')
    # 设置点 P2 的位置和速度
    P2.set_pos(O, r * N.y)
    P2.set_vel(N, v1 * N.x)
    # 使用 lambda 表达式断言创建粒子时传递相同参数会引发 TypeError 异常
    raises(TypeError, lambda: Particle(P, P, m))
    raises(TypeError, lambda: Particle('pa', m, m))
    # 断言线性动量函数的计算结果
    assert p.linear_momentum(N) == m2 * v1 * N.x
    # 断言角动量函数的计算结果
    assert p.angular_momentum(O, N) == -m2 * r * v1 * N.z
    # 更改 P2 的速度并重新测试线性动量和角动量
    P2.set_vel(N, v2 * N.y)
    assert p.linear_momentum(N) == m2 * v2 * N.y
    assert p.angular_momentum(O, N) == 0
    P2.set_vel(N, v3 * N.z)
    assert p.linear_momentum(N) == m2 * v3 * N.z
    assert p.angular_momentum(O, N) == m2 * r * v3 * N.x
    P2.set_vel(N, v1 * N.x + v2 * N.y + v3 * N.z)
    assert p.linear_momentum(N) == m2 * (v1 * N.x + v2 * N.y + v3 * N.z)
    assert p.angular_momentum(O, N) == m2 * r * (v3 * N.x - v1 * N.z)
    # 设置粒子的势能
    p.potential_energy = m * g * h
    assert p.potential_energy == m * g * h
    # TODO: 使得动能的计算结果不依赖于系统
    assert p.kinetic_energy(N) in [m2 * (v1 ** 2 + v2 ** 2 + v3 ** 2) / 2,
                                   m2 * v1 ** 2 / 2 + m2 * v2 ** 2 / 2 + m2 * v3 ** 2 / 2]


# 定义测试函数：测试平行轴定理
def test_parallel_axis():
    # 创建参考系 N 和符号变量
    N = ReferenceFrame('N')
    m, a, b = symbols('m, a, b')
    o = Point('o')
    # 创建位于 o 处偏移 a*N.x + b*N.y 的新点 p
    p = o.locatenew('p', a * N.x + b * N.y)
    # 创建质点 P 在 o 处，质量为 m
    P = Particle('P', o, m)
    # 使用平行轴定理计算惯性张量 Ip
    Ip = P.parallel_axis(p, N)
    # 计算预期的惯性张量 Ip_expected
    Ip_expected = inertia(N, m * b ** 2, m * a ** 2, m * (a ** 2 + b ** 2),
                          ixy=-m * a * b)
    # 断言计算结果符合预期
    assert Ip == Ip_expected


# 定义测试函数：测试设置势能的过时警告
def test_deprecated_set_potential_energy():
    # 定义符号变量
    m, g, h = symbols('m g h')
    # 创建点 P 和粒子 'pa'，指定质量 m
    P = Point('P')
    p = Particle('pa', P, m)
    # 使用 warns_deprecated_sympy 上下文确保设置势能会引发过时警告
    with warns_deprecated_sympy():
        p.set_potential_energy(m * g * h)
```