# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_loads.py`

```
from pytest import raises  # 导入 pytest 中的 raises 函数

from sympy import symbols  # 导入 sympy 库中的 symbols 函数
from sympy.physics.mechanics import (RigidBody, Particle, ReferenceFrame, Point,
                                     outer, dynamicsymbols, Force, Torque)
                                     # 导入 sympy 中的力学模块及其子模块和类
from sympy.physics.mechanics.loads import gravity, _parse_load
                                     # 导入力学模块中的 gravity 和 _parse_load 函数


def test_force_default():
    N = ReferenceFrame('N')  # 创建一个惯性参考系对象 N
    Po = Point('Po')  # 创建一个空间点对象 Po
    f1 = Force(Po, N.x)  # 创建一个作用在点 Po 上的力 f1，方向为 N.x
    assert f1.point == Po  # 断言 f1 的作用点是 Po
    assert f1.force == N.x  # 断言 f1 的力的方向是 N.x
    assert f1.__repr__() == 'Force(point=Po, force=N.x)'  # 断言 f1 的字符串表示形式
    # Test tuple behaviour
    assert isinstance(f1, tuple)  # 断言 f1 是一个元组
    assert f1[0] == Po  # 断言 f1 的第一个元素是 Po
    assert f1[1] == N.x  # 断言 f1 的第二个元素是 N.x
    assert f1 == (Po, N.x)  # 断言 f1 等于 (Po, N.x)
    assert f1 != (N.x, Po)  # 断言 f1 不等于 (N.x, Po)
    assert f1 != (Po, N.x + N.y)  # 断言 f1 不等于 (Po, N.x + N.y)
    assert f1 != (Point('Co'), N.x)  # 断言 f1 不等于 (Point('Co'), N.x)
    # Test body as input
    P = Particle('P', Po)  # 创建一个质点 P，位于点 Po 处
    f2 = Force(P, N.x)  # 创建一个作用在质点 P 上的力 f2，方向为 N.x
    assert f1 == f2  # 断言 f1 等于 f2


def test_torque_default():
    N = ReferenceFrame('N')  # 创建一个惯性参考系对象 N
    f1 = Torque(N, N.x)  # 创建一个关于参考系 N 的力矩 f1，大小为 N.x
    assert f1.frame == N  # 断言 f1 的参考系是 N
    assert f1.torque == N.x  # 断言 f1 的力矩大小是 N.x
    assert f1.__repr__() == 'Torque(frame=N, torque=N.x)'  # 断言 f1 的字符串表示形式
    # Test tuple behaviour
    assert isinstance(f1, tuple)  # 断言 f1 是一个元组
    assert f1[0] == N  # 断言 f1 的第一个元素是 N
    assert f1[1] == N.x  # 断言 f1 的第二个元素是 N.x
    assert f1 == (N, N.x)  # 断言 f1 等于 (N, N.x)
    assert f1 != (N.x, N)  # 断言 f1 不等于 (N.x, N)
    assert f1 != (N, N.x + N.y)  # 断言 f1 不等于 (N, N.x + N.y)
    assert f1 != (ReferenceFrame('A'), N.x)  # 断言 f1 不等于 (ReferenceFrame('A'), N.x)
    # Test body as input
    rb = RigidBody('P', frame=N)  # 创建一个刚体 rb，位于参考系 N 中
    f2 = Torque(rb, N.x)  # 创建一个关于刚体 rb 的力矩 f2，大小为 N.x
    assert f1 == f2  # 断言 f1 等于 f2


def test_gravity():
    N = ReferenceFrame('N')  # 创建一个惯性参考系对象 N
    m, M, g = symbols('m M g')  # 定义符号变量 m, M, g
    F1, F2 = dynamicsymbols('F1 F2')  # 定义动力学符号 F1, F2
    po = Point('po')  # 创建一个空间点对象 po
    pa = Particle('pa', po, m)  # 创建一个质点 pa，位于点 po 处，质量为 m
    A = ReferenceFrame('A')  # 创建一个新的参考系 A
    P = Point('P')  # 创建一个新的空间点对象 P
    I = outer(A.x, A.x)  # 计算 A.x 与自身的外积，得到一个张量
    B = RigidBody('B', P, A, M, (I, P))  # 创建一个刚体 B，位于点 P 处，参考系为 A，质量为 M，惯量为 I
    forceList = [(po, F1), (P, F2)]  # 定义一个力列表 forceList
    forceList.extend(gravity(g * N.y, pa, B))  # 将重力列表添加到 forceList 中
    l = [(po, F1), (P, F2), (po, g * m * N.y), (P, g * M * N.y)]  # 定义预期的力列表 l

    for i in range(len(l)):  # 遍历预期的力列表 l
        for j in range(len(l[i])):  # 遍历力列表中每个元素
            assert forceList[i][j] == l[i][j]  # 断言 forceList 中的每个元素与 l 中的对应元素相等


def test_parse_loads():
    N = ReferenceFrame('N')  # 创建一个惯性参考系对象 N
    po = Point('po')  # 创建一个空间点对象 po
    assert _parse_load(Force(po, N.z)) == (po, N.z)  # 断言解析力的结果与预期相符
    assert _parse_load(Torque(N, N.x)) == (N, N.x)  # 断言解析力矩的结果与预期相符
    f1 = _parse_load((po, N.x))  # 测试是否能识别一个力
    assert isinstance(f1, Force)  # 断言 f1 是 Force 类的实例
    assert f1 == Force(po, N.x)  # 断言 f1 等于 Force(po, N.x)
    t1 = _parse_load((N, N.y))  # 测试是否能识别一个力矩
    assert isinstance(t1, Torque)  # 断言 t1 是 Torque 类的实例
    assert t1 == Torque(N, N.y)  # 断言 t1 等于 Torque(N, N.y)
    # Bodies should be undetermined (even in case of a Particle)
    raises(ValueError, lambda: _parse_load((Particle('pa', po), N.x)))  # 断言在给定的情况下引发 ValueError 异常
    raises(ValueError, lambda: _parse_load((RigidBody('pa', po, N), N.x)))  # 断言在给定的情况下引发 ValueError 异常
    # Invalid tuple length
    raises(ValueError, lambda: _parse_load((po, N.x, po, N.x)))  # 断言在给定的情况下引发 ValueError 异常
    # Invalid type
    raises(TypeError, lambda: _parse_load([po, N.x]))  # 断言在给定的情况下引发 TypeError 异常
```