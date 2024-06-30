# `D:\src\scipysrc\sympy\sympy\physics\mechanics\tests\test_functions.py`

```
from sympy import sin, cos, tan, pi, symbols, Matrix, S, Function
from sympy.physics.mechanics import (Particle, Point, ReferenceFrame,
                                     RigidBody)
from sympy.physics.mechanics import (angular_momentum, dynamicsymbols,
                                     kinetic_energy, linear_momentum,
                                     outer, potential_energy, msubs,
                                     find_dynamicsymbols, Lagrangian)

from sympy.physics.mechanics.functions import (
    center_of_mass, _validate_coordinates, _parse_linear_solver)
from sympy.testing.pytest import raises, warns_deprecated_sympy

# 定义符号变量 q1, q2, q3, q4, q5
q1, q2, q3, q4, q5 = symbols('q1 q2 q3 q4 q5')
# 创建惯性参考系 N
N = ReferenceFrame('N')
# 根据 N 参考系创建 A 参考系，并绕 z 轴旋转 q1
A = N.orientnew('A', 'Axis', [q1, N.z])
# 根据 A 参考系创建 B 参考系，并绕 x 轴旋转 q2
B = A.orientnew('B', 'Axis', [q2, A.x])
# 根据 B 参考系创建 C 参考系，并绕 y 轴旋转 q3
C = B.orientnew('C', 'Axis', [q3, B.y])

# 测试线性动量函数
def test_linear_momentum():
    # 创建惯性参考系 N
    N = ReferenceFrame('N')
    # 创建质点 Ac
    Ac = Point('Ac')
    # 设置 Ac 的速度为 25*N.y
    Ac.set_vel(N, 25 * N.y)
    # 创建惯性张量 I
    I = outer(N.x, N.x)
    # 创建刚体 A，质量为 20，惯性为 I，质心为 Ac
    A = RigidBody('A', Ac, N, 20, (I, Ac))
    # 创建质点 P
    P = Point('P')
    # 创建质点 Pa，质量为 1，质心为 P
    Pa = Particle('Pa', P, 1)
    # 设置质点 Pa 的速度为 10*N.x
    Pa.point.set_vel(N, 10 * N.x)
    # 检查 linear_momentum 函数对于给定参数的返回值
    raises(TypeError, lambda: linear_momentum(A, A, Pa))
    raises(TypeError, lambda: linear_momentum(N, N, Pa))
    assert linear_momentum(N, A, Pa) == 10 * N.x + 500 * N.y

# 测试角动量和线性动量函数
def test_angular_momentum_and_linear_momentum():
    # 定义符号变量 m, M, l, I 和动态符号 omega
    m, M, l, I = symbols('m, M, l, I')
    omega = dynamicsymbols('omega')
    # 创建惯性参考系 N 和旋转参考系 a
    N = ReferenceFrame('N')
    a = ReferenceFrame('a')
    # 创建固定点 O
    O = Point('O')
    # 在 O 点的 l*N.x 处创建质心 Ac
    Ac = O.locatenew('Ac', l * N.x)
    # 在 Ac 点的 l*N.x 处创建质点 P
    P = Ac.locatenew('P', l * N.x)
    # 设置 O 点的速度为 0*N.x
    O.set_vel(N, 0 * N.x)
    # 设置 a 参考系的角速度为 omega*N.z
    a.set_ang_vel(N, omega * N.z)
    # 根据相对运动理论计算 Ac 的速度
    Ac.v2pt_theory(O, N, a)
    # 根据相对运动理论计算 P 的速度
    P.v2pt_theory(O, N, a)
    # 创建质点 Pa，质量为 m，质心为 P
    Pa = Particle('Pa', P, m)
    # 创建刚体 A，质量为 M，惯性为 I*N.z*N.z，质心为 Ac
    A = RigidBody('A', Ac, a, M, (I * outer(N.z, N.z), Ac))
    # 预期的线性动量值
    expected = 2 * m * omega * l * N.y + M * l * omega * N.y
    assert linear_momentum(N, A, Pa) == expected
    raises(TypeError, lambda: angular_momentum(N, N, A, Pa))
    raises(TypeError, lambda: angular_momentum(O, O, A, Pa))
    raises(TypeError, lambda: angular_momentum(O, N, O, Pa))
    # 预期的角动量值
    expected = (I + M * l**2 + 4 * m * l**2) * omega * N.z
    assert angular_momentum(O, N, A, Pa) == expected

# 测试动能函数
def test_kinetic_energy():
    # 定义符号变量 m, M, l1 和动态符号 omega
    m, M, l1 = symbols('m M l1')
    omega = dynamicsymbols('omega')
    # 创建惯性参考系 N 和固定点 O
    N = ReferenceFrame('N')
    O = Point('O')
    # 设置 O 点的速度为 0*N.x
    O.set_vel(N, 0 * N.x)
    # 在 O 点的 l1*N.x 处创建质心 Ac
    Ac = O.locatenew('Ac', l1 * N.x)
    # 在 Ac 点的 l1*N.x 处创建质点 P
    P = Ac.locatenew('P', l1 * N.x)
    # 创建旋转参考系 a
    a = ReferenceFrame('a')
    # 设置 a 参考系的角速度为 omega*N.z
    a.set_ang_vel(N, omega * N.z)
    # 根据相对运动理论计算 Ac 的速度
    Ac.v2pt_theory(O, N, a)
    # 根据相对运动理论计算 P 的速度
    P.v2pt_theory(O, N, a)
    # 创建质点 Pa，质量为 m，质心为 P
    Pa = Particle('Pa', P, m)
    # 创建惯性张量 I
    I = outer(N.z, N.z)
    # 创建刚体 A，质量为 M，惯性为 I，质心为 Ac
    A = RigidBody('A', Ac, a, M, (I, Ac))
    # 检查 kinetic_energy 函数对于给定参数的返回值
    raises(TypeError, lambda: kinetic_energy(Pa, Pa, A))
    # 调用函数 raises，验证 kinetic_energy 函数在给定参数时是否会引发 TypeError 异常
    raises(TypeError, lambda: kinetic_energy(N, N, A))
    
    # 断言验证表达式是否成立：kinetic_energy(N, Pa, A) 减去 (M*l1**2*omega**2/2 + 2*l1**2*m*omega**2 + omega**2/2) 的结果等于 0
    assert 0 == (kinetic_energy(N, Pa, A) - (M*l1**2*omega**2/2
            + 2*l1**2*m*omega**2 + omega**2/2)).expand()
# 定义测试函数 test_potential_energy，用于测试势能计算的正确性
def test_potential_energy():
    # 定义符号变量
    m, M, l1, g, h, H = symbols('m M l1 g h H')
    # 定义动力学符号
    omega = dynamicsymbols('omega')
    # 定义一个惯性参考系 N
    N = ReferenceFrame('N')
    # 定义一个点 O，并设置其在 N 参考系中的速度为零
    O = Point('O')
    O.set_vel(N, 0 * N.x)
    # 在点 O 上定义点 Ac，位置向量为 l1 * N.x
    Ac = O.locatenew('Ac', l1 * N.x)
    # 在点 Ac 上定义点 P，位置向量也为 l1 * N.x
    P = Ac.locatenew('P', l1 * N.x)
    # 定义一个新的参考系 a，并设置其相对于 N 参考系的角速度为 omega * N.z
    a = ReferenceFrame('a')
    a.set_ang_vel(N, omega * N.z)
    # 使用 v2pt_theory 方法计算点 Ac 和 P 相对于点 O 的速度
    Ac.v2pt_theory(O, N, a)
    P.v2pt_theory(O, N, a)
    # 在点 P 上定义一个质点 Pa，质量为 m
    Pa = Particle('Pa', P, m)
    # 定义惯性张量 I
    I = outer(N.z, N.z)
    # 在点 Ac 处定义一个刚体 A，质量为 M，惯性张量为 (I, Ac)
    A = RigidBody('A', Ac, a, M, (I, Ac))
    # 设置质点 Pa 的势能为 m * g * h
    Pa.potential_energy = m * g * h
    # 设置刚体 A 的势能为 M * g * H
    A.potential_energy = M * g * H
    # 使用 assert 语句检查势能的计算是否正确
    assert potential_energy(A, Pa) == m * g * h + M * g * H


# 定义测试函数 test_Lagrangian，用于测试拉格朗日量的计算
def test_Lagrangian():
    # 定义符号变量
    M, m, g, h = symbols('M m g h')
    # 定义一个惯性参考系 N
    N = ReferenceFrame('N')
    # 定义一个点 O，并设置其在 N 参考系中的速度为零
    O = Point('O')
    O.set_vel(N, 0 * N.x)
    # 在点 O 上定义点 P，位置向量为 1 * N.x，并设置其速度为 10 * N.x
    P = O.locatenew('P', 1 * N.x)
    P.set_vel(N, 10 * N.x)
    # 在点 P 上定义一个质点 Pa，质量为 1
    Pa = Particle('Pa', P, 1)
    # 在点 O 上定义点 Ac，位置向量为 2 * N.y，并设置其速度为 5 * N.y
    Ac = O.locatenew('Ac', 2 * N.y)
    Ac.set_vel(N, 5 * N.y)
    # 在点 Ac 上定义一个新的参考系 a，并设置其相对于 N 参考系的角速度为 10 * N.z
    a = ReferenceFrame('a')
    a.set_ang_vel(N, 10 * N.z)
    # 定义惯性张量 I
    I = outer(N.z, N.z)
    # 在点 Ac 处定义一个刚体 A，质量为 20，惯性张量为 (I, Ac)
    A = RigidBody('A', Ac, a, 20, (I, Ac))
    # 设置质点 Pa 的势能为 m * g * h
    Pa.potential_energy = m * g * h
    # 设置刚体 A 的势能为 M * g * h
    A.potential_energy = M * g * h
    # 使用 raises 函数检查在计算拉格朗日量时是否会引发 TypeError
    raises(TypeError, lambda: Lagrangian(A, A, Pa))
    raises(TypeError, lambda: Lagrangian(N, N, Pa))


# 定义测试函数 test_msubs，用于测试 msubs 函数的替换功能
def test_msubs():
    # 定义符号变量和动力学符号
    a, b = symbols('a, b')
    x, y, z = dynamicsymbols('x, y, z')
    # 测试简单替换功能
    expr = Matrix([[a*x + b, x*y.diff() + y],
                   [x.diff().diff(), z + sin(z.diff())]])
    sol = Matrix([[a + b, y],
                  [x.diff().diff(), 1]])
    sd = {x: 1, z: 1, z.diff(): 0, y.diff(): 0}
    assert msubs(expr, sd) == sol
    # 测试智能替换功能
    expr = cos(x + y)*tan(x + y) + b*x.diff()
    sd = {x: 0, y: pi/2, x.diff(): 1}
    assert msubs(expr, sd, smart=True) == b + 1
    # 测试在给定参考系 N 中的向量替换
    N = ReferenceFrame('N')
    v = x*N.x + y*N.y
    d = x*(N.x|N.x) + y*(N.y|N.y)
    v_sol = 1*N.y
    d_sol = 1*(N.y|N.y)
    sd = {x: 0, y: 1}
    assert msubs(v, sd) == v_sol
    assert msubs(d, sd) == d_sol


# 定义测试函数 test_find_dynamicsymbols，用于测试 find_dynamicsymbols 函数的功能
def test_find_dynamicsymbols():
    # 定义符号变量和动力学符号
    a, b = symbols('a, b')
    x, y, z = dynamicsymbols('x, y, z')
    # 定义一个矩阵表达式 expr
    expr = Matrix([[a*x + b, x*y.diff() + y],
                   [x.diff().diff(), z + sin(z.diff())]])
    # 测试查找所有动力学符号
    sol = {x, y.diff(), y, x.diff().diff(), z, z.diff()}
    assert find_dynamicsymbols(expr) == sol
    # 测试排除指定符号列表后查找所有动力学符号
    exclude_list = [x, y, z]
    sol = {y.diff(), x.diff().diff(), z.diff()}
    assert find_dynamicsymbols(expr, exclude=exclude_list) == sol
    # 测试在给定参考系 A 中查找向量中的动力学符号
    d, e, f = dynamicsymbols('d, e, f')
    A = ReferenceFrame('A')
    v = d * A.x + e * A.y + f * A.z
    sol = {d, e, f}
    assert find_dynamicsymbols(v, reference_frame=A) == sol
    # 测试当只提供向量作为输入时是否引发 ValueError
    raises(ValueError, lambda: find_dynamicsymbols(v))
# 计算质心的测试函数
def test_center_of_mass():
    # 定义参考系对象a
    a = ReferenceFrame('a')
    # 定义实数符号m
    m = symbols('m', real=True)
    # 创建粒子p1，位于点p1_pt，质量为1
    p1 = Particle('p1', Point('p1_pt'), S.One)
    # 创建粒子p2，位于点p2_pt，质量为2
    p2 = Particle('p2', Point('p2_pt'), S(2))
    # 创建粒子p3，位于点p3_pt，质量为3
    p3 = Particle('p3', Point('p3_pt'), S(3))
    # 创建粒子p4，位于点p4_pt，质量为m
    p4 = Particle('p4', Point('p4_pt'), m)
    # 定义参考系对象b_f
    b_f = ReferenceFrame('b_f')
    # 定义质心点b_cm
    b_cm = Point('b_cm')
    # 定义符号mb
    mb = symbols('mb')
    # 创建刚体b，位于质心b_cm，参考系b_f，质量为mb，惯性张量为outer(b_f.x, b_f.x)
    b = RigidBody('b', b_cm, b_f, mb, (outer(b_f.x, b_f.x), b_cm))
    # 设置p2粒子相对于p1的位置为a.x
    p2.point.set_pos(p1.point, a.x)
    # 设置p3粒子相对于p1的位置为a.x + a.y
    p3.point.set_pos(p1.point, a.x + a.y)
    # 设置p4粒子相对于p1的位置为a.y
    p4.point.set_pos(p1.point, a.y)
    # 设置b的质心位置为p1的位置加上a.y + a.z
    b.masscenter.set_pos(p1.point, a.y + a.z)
    # 创建点对象point_o
    point_o = Point('o')
    # 设置点point_o的位置为p1的位置加上所有粒子和刚体b的质心的质心
    point_o.set_pos(p1.point, center_of_mass(p1.point, p1, p2, p3, p4, b))
    # 计算表达式expr
    expr = 5/(m + mb + 6)*a.x + (m + mb + 3)/(m + mb + 6)*a.y + mb/(m + mb + 6)*a.z
    # 断言point_o相对于p1的位置与expr的差为0
    assert point_o.pos_from(p1.point) - expr == 0


# 验证坐标的测试函数
def test_validate_coordinates():
    # 定义动力学符号
    q1, q2, q3, u1, u2, u3, ua1, ua2, ua3 = dynamicsymbols('q1:4 u1:4 ua1:4')
    # 定义符号s1, s2, s3
    s1, s2, s3 = symbols('s1:4')
    
    # 测试正常情况
    _validate_coordinates([q1, q2, q3], [u1, u2, u3],
                          u_auxiliary=[ua1, ua2, ua3])
    
    # 测试坐标和速度数目不相等的情况
    _validate_coordinates([q1, q2])
    _validate_coordinates([q1, q2], [u1])
    _validate_coordinates(speeds=[u1, u2])
    
    # 测试重复坐标的情况
    _validate_coordinates([q1, q2, q2], [u1, u2, u3], check_duplicates=False)
    raises(ValueError, lambda: _validate_coordinates([q1, q2, q2], [u1, u2, u3]))
    _validate_coordinates([q1, q2, q3], [u1, u2, u3], check_duplicates=False)
    raises(ValueError, lambda: _validate_coordinates([q1, q2, q3], [u1, u2, u3], check_duplicates=True))
    raises(ValueError, lambda: _validate_coordinates([q1, q2, q3], [q1, u2, u3], check_duplicates=True))
    
    # 测试辅助坐标不符合条件的情况
    _validate_coordinates([q1, q2, q3], [u1, u2, u3], check_duplicates=False,
                          u_auxiliary=[u1, ua2, ua2])
    raises(ValueError, lambda: _validate_coordinates([q1, q2, q3], [u1, u2, u3], u_auxiliary=[u1, ua2, ua3]))
    raises(ValueError, lambda: _validate_coordinates([q1, q2, q3], [u1, u2, u3], u_auxiliary=[q1, ua2, ua3]))
    raises(ValueError, lambda: _validate_coordinates([q1, q2, q3], [u1, u2, u3], u_auxiliary=[ua1, ua2, ua2]))
    
    # 测试非动力学符号的情况
    _validate_coordinates([q1 + q2, q3], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates([q1 + q2, q3]))
    _validate_coordinates([s1, q1, q2], [0, u1, u2], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates([s1, q1, q2], [0, u1, u2], is_dynamicsymbols=True))
    _validate_coordinates([s1 + s2 + s3, q1], [0, u1], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates([s1 + s2 + s3, q1], [0, u1], is_dynamicsymbols=True))
    _validate_coordinates(u_auxiliary=[s1, ua1], is_dynamicsymbols=False)
    raises(ValueError, lambda: _validate_coordinates(u_auxiliary=[s1, ua1]))
    # 获取时间动态符号 _t
    t = dynamicsymbols._t
    # 创建符号变量 a
    a = symbols('a')
    # 创建两个函数符号变量 f1 和 f2
    f1, f2 = symbols('f1:3', cls=Function)
    # 验证坐标是否有效，传入参数为 [f1(a), f2(a)]，不是动态符号
    _validate_coordinates([f1(a), f2(a)], is_dynamicsymbols=False)
    # 预期抛出 ValueError 异常，验证坐标无效，传入参数为 [f1(a), f2(a)]
    raises(ValueError, lambda: _validate_coordinates([f1(a), f2(a)]))
    # 预期抛出 ValueError 异常，验证速度坐标无效，传入参数为 [f1(a), f2(a)]
    raises(ValueError, lambda: _validate_coordinates(speeds=[f1(a), f2(a)]))
    # 设置时间动态符号 _t 的值为 a
    dynamicsymbols._t = a
    # 再次验证坐标是否有效，传入参数为 [f1(a), f2(a)]
    _validate_coordinates([f1(a), f2(a)])
    # 预期抛出 ValueError 异常，验证坐标无效，传入参数为 [f1(t), f2(t)]，此时 t 为时间动态符号
    raises(ValueError, lambda: _validate_coordinates([f1(t), f2(t)]))
    # 恢复时间动态符号 _t 的值为 t
    dynamicsymbols._t = t
# 测试解析线性求解器的函数
def test_parse_linear_solver():
    # 创建一个 3x3 的符号矩阵 A 和一个 3x2 的符号矩阵 b
    A, b = Matrix(3, 3, symbols('a:9')), Matrix(3, 2, symbols('b:6'))
    # 断言解析线性求解器函数 _parse_linear_solver 能够被调用
    assert _parse_linear_solver(Matrix.LUsolve) == Matrix.LUsolve  # Test callable
    # 断言通过字符串 'LU' 解析得到的线性求解器能够正确求解 A 和 b
    assert _parse_linear_solver('LU')(A, b) == Matrix.LUsolve(A, b)


# 测试已移除的、已移动的函数
def test_deprecated_moved_functions():
    # 从 sympy.physics.mechanics.functions 中导入惯性、点质量惯性、重力函数
    from sympy.physics.mechanics.functions import (
        inertia, inertia_of_point_mass, gravity)
    # 创建一个参考坐标系 N
    N = ReferenceFrame('N')
    # 在警告过期的 SymPy 函数时执行以下代码块
    with warns_deprecated_sympy():
        # 断言计算惯性函数返回正确的结果
        assert inertia(N, 0, 1, 0, 1) == (N.x | N.y) + (N.y | N.x) + (N.y | N.y)
    # 在警告过期的 SymPy 函数时执行以下代码块
    with warns_deprecated_sympy():
        # 断言计算点质量惯性函数返回正确的结果
        assert inertia_of_point_mass(1, N.x + N.y, N) == (
            (N.x | N.x) + (N.y | N.y) + 2 * (N.z | N.z) -
            (N.x | N.y) - (N.y | N.x))
    # 创建一个质点对象 P
    p = Particle('P')
    # 在警告过期的 SymPy 函数时执行以下代码块
    with warns_deprecated_sympy():
        # 断言计算重力函数返回正确的结果
        assert gravity(-2 * N.z, p) == [(p.masscenter, -2 * p.mass * N.z)]
```