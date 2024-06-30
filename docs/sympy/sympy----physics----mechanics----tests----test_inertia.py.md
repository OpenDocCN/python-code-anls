# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_inertia.py`

```
from sympy import symbols  # 导入 sympy 模块中的 symbols 符号变量
from sympy.testing.pytest import raises  # 导入 sympy.testing.pytest 模块中的 raises 异常处理函数
from sympy.physics.mechanics import (inertia, inertia_of_point_mass,
                                     Inertia, ReferenceFrame, Point)  # 导入 sympy.physics.mechanics 模块中的多个类和函数


def test_inertia_dyadic():
    N = ReferenceFrame('N')  # 创建一个惯性参考系对象 N
    ixx, iyy, izz = symbols('ixx iyy izz')  # 创建符号变量 ixx, iyy, izz
    ixy, iyz, izx = symbols('ixy iyz izx')  # 创建符号变量 ixy, iyz, izx
    assert inertia(N, ixx, iyy, izz) == (ixx * (N.x | N.x) + iyy *
            (N.y | N.y) + izz * (N.z | N.z))  # 断言计算得到的惯性力矩 dyadic 与预期相等
    assert inertia(N, 0, 0, 0) == 0 * (N.x | N.x)  # 断言当惯性力矩参数全为 0 时，结果为零
    raises(TypeError, lambda: inertia(0, 0, 0, 0))  # 断言调用 inertia 函数时传入非预期类型的参数会抛出 TypeError 异常
    assert inertia(N, ixx, iyy, izz, ixy, iyz, izx) == (ixx * (N.x | N.x) +
            ixy * (N.x | N.y) + izx * (N.x | N.z) + ixy * (N.y | N.x) + iyy *
        (N.y | N.y) + iyz * (N.y | N.z) + izx * (N.z | N.x) + iyz * (N.z |
            N.y) + izz * (N.z | N.z))  # 断言计算得到的惯性力矩 dyadic 与预期相等


def test_inertia_of_point_mass():
    r, s, t, m = symbols('r s t m')  # 创建符号变量 r, s, t, m
    N = ReferenceFrame('N')  # 创建一个惯性参考系对象 N

    px = r * N.x  # 创建一个点的位置矢量 px
    I = inertia_of_point_mass(m, px, N)  # 计算点质量的惯性力矩
    assert I == m * r**2 * (N.y | N.y) + m * r**2 * (N.z | N.z)  # 断言计算得到的惯性力矩 dyadic 与预期相等

    py = s * N.y  # 创建一个点的位置矢量 py
    I = inertia_of_point_mass(m, py, N)  # 计算点质量的惯性力矩
    assert I == m * s**2 * (N.x | N.x) + m * s**2 * (N.z | N.z)  # 断言计算得到的惯性力矩 dyadic 与预期相等

    pz = t * N.z  # 创建一个点的位置矢量 pz
    I = inertia_of_point_mass(m, pz, N)  # 计算点质量的惯性力矩
    assert I == m * t**2 * (N.x | N.x) + m * t**2 * (N.y | N.y)  # 断言计算得到的惯性力矩 dyadic 与预期相等

    p = px + py + pz  # 创建一个复合点的位置矢量 p
    I = inertia_of_point_mass(m, p, N)  # 计算复合点的惯性力矩
    assert I == (m * (s**2 + t**2) * (N.x | N.x) -
                 m * r * s * (N.x | N.y) -
                 m * r * t * (N.x | N.z) -
                 m * r * s * (N.y | N.x) +
                 m * (r**2 + t**2) * (N.y | N.y) -
                 m * s * t * (N.y | N.z) -
                 m * r * t * (N.z | N.x) -
                 m * s * t * (N.z | N.y) +
                 m * (r**2 + s**2) * (N.z | N.z))  # 断言计算得到的惯性力矩 dyadic 与预期相等


def test_inertia_object():
    N = ReferenceFrame('N')  # 创建一个惯性参考系对象 N
    O = Point('O')  # 创建一个点对象 O
    ixx, iyy, izz = symbols('ixx iyy izz')  # 创建符号变量 ixx, iyy, izz
    I_dyadic = ixx * (N.x | N.x) + iyy * (N.y | N.y) + izz * (N.z | N.z)  # 创建惯性力矩 dyadic
    I = Inertia(inertia(N, ixx, iyy, izz), O)  # 创建惯性对象 I
    assert isinstance(I, tuple)  # 断言 I 是一个元组对象
    assert I.__repr__() == ('Inertia(dyadic=ixx*(N.x|N.x) + iyy*(N.y|N.y) + '
                            'izz*(N.z|N.z), point=O)')  # 断言 I 的字符串表示与预期相等
    assert I.dyadic == I_dyadic  # 断言 I 的 dyadic 属性与预期相等
    assert I.point == O  # 断言 I 的 point 属性与预期相等
    assert I[0] == I_dyadic  # 断言 I 的第一个元素与 dyadic 属性相等
    assert I[1] == O  # 断言 I 的第二个元素与 point 属性相等
    assert I == (I_dyadic, O)  # 断言 I 等于给定的元组，测试元组相等性
    raises(TypeError, lambda: I != (O, I_dyadic))  # 断言 I 与给定元组不相等会抛出 TypeError 异常，测试元组不相等情况
    assert I == Inertia(O, I_dyadic)  # 断言使用不同的参数顺序创建的 Inertia 对象等于原始对象
    assert I == Inertia.from_inertia_scalars(O, N, ixx, iyy, izz)  # 断言使用 from_inertia_scalars 方法创建的 Inertia 对象等于原始对象
    raises(TypeError, lambda: I + (1, 2))  # 断言对 I 添加元组会抛出 TypeError 异常
    raises(TypeError, lambda: (1, 2) + I)  # 断言在元组前面添加 I 会抛出 TypeError 异常
    raises(TypeError, lambda: I * 2)  # 断言将 I 乘以数值会抛出 TypeError 异常
    raises(TypeError, lambda: 2 * I)  # 断言将数值乘以 I 会抛出 TypeError 异常
```