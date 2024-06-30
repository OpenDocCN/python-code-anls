# `D:\src\scipysrc\sympy\sympy\physics\vector\tests\test_point.py`

```
from sympy.physics.vector import dynamicsymbols, Point, ReferenceFrame
from sympy.testing.pytest import raises, ignore_warnings
import warnings

# 定义测试函数，用于验证 v1pt_theory 方法
def test_point_v1pt_theorys():
    # 定义动力学符号变量
    q, q2 = dynamicsymbols('q q2')
    qd, q2d = dynamicsymbols('q q2', 1)
    qdd, q2dd = dynamicsymbols('q q2', 2)
    
    # 定义惯性参考系 N 和新的参考系 B
    N = ReferenceFrame('N')
    B = ReferenceFrame('B')
    
    # 设置参考系 B 的角速度
    B.set_ang_vel(N, qd * B.z)
    
    # 定义点 O 和点 P，P 相对于 O 在 B 参考系中沿着 B.x 方向的位移
    O = Point('O')
    P = O.locatenew('P', B.x)
    
    # 设置点 P 相对于参考系 B 的速度为 0
    P.set_vel(B, 0)
    
    # 设置点 O 相对于参考系 N 的速度为 0
    O.set_vel(N, 0)
    
    # 验证 v1pt_theory 方法计算的结果
    assert P.v1pt_theory(O, N, B) == qd * B.y
    
    # 修改点 O 相对于参考系 N 的速度为 N.x 方向
    O.set_vel(N, N.x)
    
    # 验证修改后 v1pt_theory 方法计算的结果
    assert P.v1pt_theory(O, N, B) == N.x + qd * B.y
    
    # 修改点 P 相对于参考系 B 的速度为 B.z 方向
    P.set_vel(B, B.z)
    
    # 验证修改后 v1pt_theory 方法计算的结果
    assert P.v1pt_theory(O, N, B) == B.z + N.x + qd * B.y

# 定义测试函数，用于验证 a1pt_theory 方法
def test_point_a1pt_theorys():
    # 定义动力学符号变量
    q, q2 = dynamicsymbols('q q2')
    qd, q2d = dynamicsymbols('q q2', 1)
    qdd, q2dd = dynamicsymbols('q q2', 2)
    
    # 定义惯性参考系 N 和新的参考系 B
    N = ReferenceFrame('N')
    B = ReferenceFrame('B')
    
    # 设置参考系 B 的角速度
    B.set_ang_vel(N, qd * B.z)
    
    # 定义点 O 和点 P，P 相对于 O 在 B 参考系中沿着 B.x 方向的位移
    O = Point('O')
    P = O.locatenew('P', B.x)
    
    # 设置点 P 相对于参考系 B 的速度为 0
    P.set_vel(B, 0)
    
    # 设置点 O 相对于参考系 N 的速度为 0
    O.set_vel(N, 0)
    
    # 验证 a1pt_theory 方法计算的结果
    assert P.a1pt_theory(O, N, B) == -(qd**2) * B.x + qdd * B.y
    
    # 修改点 P 相对于参考系 B 的速度为 q2d * B.z
    P.set_vel(B, q2d * B.z)
    
    # 验证修改后 a1pt_theory 方法计算的结果
    assert P.a1pt_theory(O, N, B) == -(qd**2) * B.x + qdd * B.y + q2dd * B.z
    
    # 修改点 O 相对于参考系 N 的速度为 q2d * N.x
    O.set_vel(N, q2d * N.x)
    
    # 验证修改后 a1pt_theory 方法计算的结果
    assert P.a1pt_theory(O, N, B) == ((q2dd - qd**2) * B.x + (q2d * qd + qdd) * B.y +
                               q2dd * B.z)

# 定义测试函数，用于验证 v2pt_theory 方法
def test_point_v2pt_theorys():
    # 定义动力学符号变量
    q = dynamicsymbols('q')
    qd = dynamicsymbols('q', 1)
    
    # 定义惯性参考系 N
    N = ReferenceFrame('N')
    
    # 用旋转矩阵定义新的参考系 B
    B = N.orientnew('B', 'Axis', [q, N.z])
    
    # 定义点 O 和点 P，P 相对于 O 在 B 参考系中的初始位置为 0
    O = Point('O')
    P = O.locatenew('P', 0)
    
    # 设置点 O 相对于参考系 N 的速度为 0
    O.set_vel(N, 0)
    
    # 验证 v2pt_theory 方法计算的结果
    assert P.v2pt_theory(O, N, B) == 0
    
    # 将点 P 相对于 O 在 B 参考系中的位置设置为 B.x
    P = O.locatenew('P', B.x)
    
    # 验证修改后 v2pt_theory 方法计算的结果
    assert P.v2pt_theory(O, N, B) == (qd * B.z ^ B.x)
    
    # 修改点 O 相对于参考系 N 的速度为 N.x 方向
    O.set_vel(N, N.x)
    
    # 验证修改后 v2pt_theory 方法计算的结果
    assert P.v2pt_theory(O, N, B) == N.x + qd * B.y

# 定义测试函数，用于验证 a2pt_theory 方法
def test_point_a2pt_theorys():
    # 定义动力学符号变量
    q = dynamicsymbols('q')
    qd = dynamicsymbols('q', 1)
    qdd = dynamicsymbols('q', 2)
    
    # 定义惯性参考系 N
    N = ReferenceFrame('N')
    
    # 用旋转矩阵定义新的参考系 B
    B = N.orientnew('B', 'Axis', [q, N.z])
    
    # 定义点 O 和点 P，P 相对于 O 在 B 参考系中的初始位置为 0
    O = Point('O')
    P = O.locatenew('P', 0)
    
    # 设置点 O 相对于参考系 N 的速度为 0
    O.set_vel(N, 0)
    
    # 验证 a2pt_theory 方法计算的结果
    assert P.a2pt_theory(O, N, B) == 0
    
    # 将点 P 相对于 O 在 B 参考系中的位置设置为 B.x
    P.set_pos(O, B.x)
    
    # 验证修改后 a2pt_theory 方法计算的结果
    assert P.a2pt_theory(O, N, B) == (-qd**2) * B.x + (qdd) * B.y

# 定义测试函数，验证点对象的其他功能
def test_point_funcs():
    # 定义动力学符号变量
    q, q2 = dynamicsymbols('q q2')
    qd, q2d = dynamicsymbols('q q2', 1)
    qdd, q2dd = dynamicsymbols('q q2', 2)
    
    # 定义惯性参考系 N 和新的参考系 B
    N = ReferenceFrame('N')
    B = ReferenceFrame('B')
    
    # 设置参考系 B 的角速度
    B.set_ang_vel(N, 5 * B.y)
    
    # 定义点 O 和点 P，P 相对于 O 在 B 参考系中的位置为 q * B.x + q2 * B.y
    O = Point('O')
    P = O.locatenew('P', q * B.x + q2 * B.y)
    
    # 验证 pos_from 方法计算的结果
    assert P.pos_from(O) == q * B.x + q2 * B.y
    
    # 设置点 P 相对于参考系 B 的速度为 qd * B.x + q2d * B.y
    P.set_vel(B, qd * B.x + q2d * B.y)
    
    # 验证 vel 方法计算的结果
    assert
    # 创建一个名为 O 的点对象
    O = Point('O')
    # 根据给定的向量表达式创建点 P，并将其相对于参考系 B 的速度设置为 qd * B.x + q2d * B.y
    P = O.locatenew('P', q * B.x + q2 * B.y)
    P.set_vel(B, qd * B.x + q2d * B.y)
    # 设置点 O 相对于参考系 N 的速度为零
    O.set_vel(N, 0)
    # 断言验证点 P 在理论上相对于点 O、参考系 N 和参考系 B 的速度应为 qd * B.x + q2d * B.y - 5 * q * B.z
    assert P.v1pt_theory(O, N, B) == qd * B.x + q2d * B.y - 5 * q * B.z
def test_point_pos():
    # 定义一个动态符号 q
    q = dynamicsymbols('q')
    # 创建一个参考坐标系 N
    N = ReferenceFrame('N')
    # 在 N 参考系下创建一个新的参考坐标系 B，其旋转轴为 N.z，角度为 q
    B = N.orientnew('B', 'Axis', [q, N.z])
    # 创建一个名为 O 的点，作为起点
    O = Point('O')
    # 在 O 点的基础上创建一个新点 P，其位置为 10*N.x + 5*B.x
    P = O.locatenew('P', 10 * N.x + 5 * B.x)
    # 断言语句，验证点 P 相对于点 O 的位置
    assert P.pos_from(O) == 10 * N.x + 5 * B.x
    # 在 P 点的基础上创建一个新点 Q，其位置为 10*N.y + 5*B.y
    Q = P.locatenew('Q', 10 * N.y + 5 * B.y)
    # 断言语句，验证点 Q 相对于点 P 的位置
    assert Q.pos_from(P) == 10 * N.y + 5 * B.y
    # 断言语句，验证点 Q 相对于点 O 的位置
    assert Q.pos_from(O) == 10 * N.x + 10 * N.y + 5 * B.x + 5 * B.y
    # 断言语句，验证点 O 相对于点 Q 的位置
    assert O.pos_from(Q) == -10 * N.x - 10 * N.y - 5 * B.x - 5 * B.y

def test_point_partial_velocity():
    # 创建参考坐标系 N 和 A
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    # 创建一个点 p
    p = Point('p')
    # 创建动态符号 u1 和 u2
    u1, u2 = dynamicsymbols('u1, u2')
    # 设置点 p 相对于参考坐标系 N 的速度
    p.set_vel(N, u1 * A.x + u2 * N.y)
    # 断言语句，验证点 p 对于动态符号 u1 的偏导速度
    assert p.partial_velocity(N, u1) == A.x
    # 断言语句，验证点 p 对于动态符号 u1 和 u2 的偏导速度
    assert p.partial_velocity(N, u1, u2) == (A.x, N.y)
    # 断言语句，使用 lambda 表达式检查给定条件是否引发 ValueError 异常
    raises(ValueError, lambda: p.partial_velocity(A, u1))

def test_point_vel(): #Basic functionality
    # 创建动态符号 q1 和 q2
    q1, q2 = dynamicsymbols('q1 q2')
    # 创建参考坐标系 N 和 B
    N = ReferenceFrame('N')
    B = ReferenceFrame('B')
    # 创建一个点 Q
    Q = Point('Q')
    # 创建一个点 O
    O = Point('O')
    # 设置点 Q 相对于点 O 的位置
    Q.set_pos(O, q1 * N.x)
    # 断言语句，验证点 O 相对于参考坐标系 N 的速度未定义
    raises(ValueError , lambda: Q.vel(N)) # Velocity of O in N is not defined
    # 设置点 O 相对于参考坐标系 N 的速度
    O.set_vel(N, q2 * N.y)
    # 断言语句，验证点 O 相对于参考坐标系 N 的速度
    assert O.vel(N) == q2 * N.y
    # 断言语句，验证点 O 相对于参考坐标系 B 的速度未定义
    raises(ValueError , lambda : O.vel(B)) #Velocity of O is not defined in B

def test_auto_point_vel():
    # 创建动态符号 t, q1 和 q2
    t = dynamicsymbols._t
    q1, q2 = dynamicsymbols('q1 q2')
    # 创建参考坐标系 N 和 B
    N = ReferenceFrame('N')
    B = ReferenceFrame('B')
    # 创建一个点 O
    O = Point('O')
    # 创建一个点 Q，并设置其相对于点 O 的位置
    Q = Point('Q')
    Q.set_pos(O, q1 * N.x)
    # 设置点 O 相对于参考坐标系 N 的速度
    O.set_vel(N, q2 * N.y)
    # 断言语句，验证点 Q 相对于参考坐标系 N 的速度
    assert Q.vel(N) == q1.diff(t) * N.x + q2 * N.y  # Velocity of Q using O
    # 创建点 P1，并设置其相对于点 O 的位置
    P1 = Point('P1')
    P1.set_pos(O, q1 * B.x)
    # 创建点 P2，并设置其相对于点 P1 的位置
    P2 = Point('P2')
    P2.set_pos(P1, q2 * B.z)
    # 使用 lambda 表达式检查给定条件是否引发 ValueError 异常
    raises(ValueError, lambda : P2.vel(B)) # O's velocity is defined in different frame, and no
    #point in between has its velocity defined
    # 使用 lambda 表达式检查给定条件是否引发 ValueError 异常
    raises(ValueError, lambda: P2.vel(N)) # Velocity of O not defined in N

def test_auto_point_vel_multiple_point_path():
    # 创建动态符号 t, q1 和 q2
    t = dynamicsymbols._t
    q1, q2 = dynamicsymbols('q1 q2')
    # 创建参考坐标系 B
    B = ReferenceFrame('B')
    # 创建点 P，并设置其相对于参考坐标系 B 的速度
    P = Point('P')
    P.set_vel(B, q1 * B.x)
    # 创建点 P1，并设置其相对于点 P 的位置，并设置其相对于参考坐标系 B 的速度
    P1 = Point('P1')
    P1.set_pos(P, q2 * B.y)
    P1.set_vel(B, q1 * B.z)
    # 创建点 P2，并设置其相对于点 P1 的位置
    P2 = Point('P2')
    P2.set_pos(P1, q1 * B.z)
    # 创建点 P3，并设置其相对于点 P2 的位置
    P3 = Point('P3')
    P3.set_pos(P2, 10 * q1 * B.y)
    # 断言语句，验证点 P3 相对于参考坐标系 B 的速度
    assert P3.vel(B) == 10 * q1.diff(t) * B.y + (q1 + q1.diff(t)) * B.z

def test_auto_vel_dont_overwrite():
    # 创建动态符号 t, q1, q2 和 u1
    t = dynamicsymbols._t
    q1, q2, u1 = dynamicsymbols('q1, q2, u1')
    # 创建参考坐标系 N
    N = ReferenceFrame('N')
    # 创建点 P，并设置其相对于参考坐标系 N 的速度
    P = Point('P1')
    P.set_vel(N, u1 * N.x)
    # 创建点 P1，并设置其相对于点 P 的位置
    P1 = Point('P1')
    P1.set_pos(P, q2 * N.y)
    # 断言语句，验证点 P1 相对于参考坐标系 N 的速度
    assert P1.vel(N) == q2.diff(t) * N.y + u1 * N.x
    # 断言语句，验证点 P 相对于参考坐标系 N 的速度
    assert P.vel(N) == u1 * N.x
    # 设置点 P1 相对于参考坐标系 N 的速度
    P1.set_vel(N, u1 * N.z)
    # 断言语句，验证点 P1 相对于参考坐标系 N 的速度
    assert P1.vel(N) == u1 * N.z

def test_auto_point_vel_if_tree_has_vel_but_inappropriate_pos_vector():
    # 创建动态符号 q1 和 q2
    q1, q2 = dynamicsymbols('q1 q2')
    # 创建参考坐标系 B 和 S
    B =
    # 使用 raises 函数来测试是否会抛出 ValueError 异常
    # 测试 P1 对象在基于 B 参考系的速度的计算时是否会抛出异常
    raises(ValueError, lambda : P1.vel(B)) # P1.pos_from(P) can't be expressed in B
    
    # 测试 P1 对象在基于 S 参考系的速度的计算时是否会抛出异常
    raises(ValueError, lambda : P1.vel(S)) # P.vel(S) not defined
def test_auto_point_vel_shortest_path():
    t = dynamicsymbols._t  # 定义时间符号
    q1, q2, u1, u2 = dynamicsymbols('q1 q2 u1 u2')  # 定义动力学符号变量
    B = ReferenceFrame('B')  # 创建参考框架 B
    P = Point('P')  # 创建点 P
    P.set_vel(B, u1 * B.x)  # 设置点 P 相对于参考框架 B 的速度
    P1 = Point('P1')  # 创建点 P1
    P1.set_pos(P, q2 * B.y)  # 设置点 P1 相对于点 P 的位置
    P1.set_vel(B, q1 * B.z)  # 设置点 P1 相对于参考框架 B 的速度
    P2 = Point('P2')  # 创建点 P2
    P2.set_pos(P1, q1 * B.z)  # 设置点 P2 相对于点 P1 的位置
    P3 = Point('P3')  # 创建点 P3
    P3.set_pos(P2, 10 * q1 * B.y)  # 设置点 P3 相对于点 P2 的位置
    P4 = Point('P4')  # 创建点 P4
    P4.set_pos(P3, q1 * B.x)  # 设置点 P4 相对于点 P3 的位置
    O = Point('O')  # 创建点 O
    O.set_vel(B, u2 * B.y)  # 设置点 O 相对于参考框架 B 的速度
    O1 = Point('O1')  # 创建点 O1
    O1.set_pos(O, q2 * B.z)  # 设置点 O1 相对于点 O 的位置
    P4.set_pos(O1, q1 * B.x + q2 * B.z)  # 设置点 P4 相对于点 O1 的位置
    with warnings.catch_warnings():  # 捕获警告
        warnings.simplefilter('error')  # 设置警告过滤器
        with ignore_warnings(UserWarning):  # 忽略特定类型的警告
            assert P4.vel(B) == q1.diff(t) * B.x + u2 * B.y + 2 * q2.diff(t) * B.z  # 断言速度计算公式

def test_auto_point_vel_connected_frames():
    t = dynamicsymbols._t  # 定义时间符号
    q, q1, q2, u = dynamicsymbols('q q1 q2 u')  # 定义动力学符号变量
    N = ReferenceFrame('N')  # 创建参考框架 N
    B = ReferenceFrame('B')  # 创建参考框架 B
    O = Point('O')  # 创建点 O
    O.set_vel(N, u * N.x)  # 设置点 O 相对于参考框架 N 的速度
    P = Point('P')  # 创建点 P
    P.set_pos(O, q1 * N.x + q2 * B.y)  # 设置点 P 相对于点 O 的位置
    raises(ValueError, lambda: P.vel(N))  # 断言计算点 P 相对于参考框架 N 的速度会引发 ValueError
    N.orient(B, 'Axis', (q, B.x))  # 使用旋转轴方法将参考框架 N 与 B 关联
    assert P.vel(N) == (u + q1.diff(t)) * N.x + q2.diff(t) * B.y - q2 * q.diff(t) * B.z  # 断言计算点 P 相对于参考框架 N 的速度公式

def test_auto_point_vel_multiple_paths_warning_arises():
    q, u = dynamicsymbols('q u')  # 定义动力学符号变量
    N = ReferenceFrame('N')  # 创建参考框架 N
    O = Point('O')  # 创建点 O
    P = Point('P')  # 创建点 P
    Q = Point('Q')  # 创建点 Q
    R = Point('R')  # 创建点 R
    P.set_vel(N, u * N.x)  # 设置点 P 相对于参考框架 N 的速度
    Q.set_vel(N, u * N.y)  # 设置点 Q 相对于参考框架 N 的速度
    R.set_vel(N, u * N.z)  # 设置点 R 相对于参考框架 N 的速度
    O.set_pos(P, q * N.z)  # 设置点 O 相对于点 P 的位置
    O.set_pos(Q, q * N.y)  # 设置点 O 相对于点 Q 的位置
    O.set_pos(R, q * N.x)  # 设置点 O 相对于点 R 的位置
    with warnings.catch_warnings():  # 捕获警告
        warnings.simplefilter("error")  # 设置警告过滤器
        raises(UserWarning, lambda: O.vel(N))  # 断言计算点 O 相对于参考框架 N 的速度会引发 UserWarning

def test_auto_vel_cyclic_warning_arises():
    P = Point('P')  # 创建点 P
    P1 = Point('P1')  # 创建点 P1
    P2 = Point('P2')  # 创建点 P2
    P3 = Point('P3')  # 创建点 P3
    N = ReferenceFrame('N')  # 创建参考框架 N
    P.set_vel(N, N.x)  # 设置点 P 相对于参考框架 N 的速度
    P1.set_pos(P, N.x)  # 设置点 P1 相对于点 P 的位置
    P2.set_pos(P1, N.y)  # 设置点 P2 相对于点 P1 的位置
    P3.set_pos(P2, N.z)  # 设置点 P3 相对于点 P2 的位置
    P1.set_pos(P3, N.x + N.y)  # 设置点 P1 相对于点 P3 的位置
    with warnings.catch_warnings():  # 捕获警告
        warnings.simplefilter("error")  # 设置警告过滤器
        raises(UserWarning, lambda: P2.vel(N))  # 断言计算点 P2 相对于参考框架 N 的速度会引发 UserWarning

def test_auto_vel_cyclic_warning_msg():
    P = Point('P')  # 创建点 P
    P1 = Point('P1')  # 创建点 P1
    P2 = Point('P2')  # 创建点 P2
    P3 = Point('P3')  # 创建点 P3
    N = ReferenceFrame('N')  # 创建参考框架 N
    P.set_vel(N, N.x)  # 设置点 P 相对于参考框架 N 的速度
    P1.set_pos(P, N.x)  # 设置点 P1 相对于点 P 的位置
    P2.set_pos(P1, N.y)  # 设置点 P2 相对于点 P1 的位置
    P3.set_pos(P2, N.z)  # 设置点 P3 相对于点 P2 的位置
    P1.set_pos(P3, N.x + N.y)  # 设置点 P1 相对于点 P3 的位置
    with warnings.catch_warnings(record=True) as w:  # 捕获警告，并记录警告信息
        warnings.simplefilter("always")  # 总是显示警告
        P2.vel(N)  # 计算点 P2 相对于参考框架 N 的速度
        msg = str(w[-1].message).replace("\n", " ")  # 获取最后一个警告的消息内容
        assert issubclass(w[-1].category, UserWarning)  # 断言最后一个警告是 UserWarning 类型
        assert 'Kinematic loops are defined among the positions of points. This is likely not desired and may cause errors in your calculations.' in msg  # 断言消息中包含特定的警告信息
def test_auto_vel_multiple_path_warning_msg():
    # 创建一个惯性参考系 N
    N = ReferenceFrame('N')
    # 创建点 O, P, Q
    O = Point('O')
    P = Point('P')
    Q = Point('Q')
    # 设置点 P 和 Q 在 N 参考系中的速度
    P.set_vel(N, N.x)
    Q.set_vel(N, N.y)
    # 设置点 O 相对于点 P 在 N 参考系中的位置
    O.set_pos(P, N.z)
    # 设置点 O 相对于点 Q 在 N 参考系中的位置
    O.set_pos(Q, N.y)
    
    # 捕获警告消息，因为点树中存在两条可能的路径，可能会引发警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # 计算点 O 相对于 N 参考系的速度
        O.vel(N)
        # 获取最后一条警告消息并将其转换为字符串格式
        msg = str(w[-1].message).replace("\n", " ")
        # 断言最后一条警告消息是用户警告类型
        assert issubclass(w[-1].category, UserWarning)
        # 断言消息中包含 'Velocity'
        assert 'Velocity' in msg
        # 断言消息中包含 'automatically calculated based on point'
        assert 'automatically calculated based on point' in msg
        # 断言消息中包含 'Velocities from these points are not necessarily the same. This may cause errors in your calculations.'
        assert 'Velocities from these points are not necessarily the same. This may cause errors in your calculations.' in msg

def test_auto_vel_derivative():
    # 定义动力学符号 q1, q2 和其导数 u1, u2
    q1, q2 = dynamicsymbols('q1:3')
    u1, u2 = dynamicsymbols('u1:3', 1)
    # 创建参考系 A, B, C
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')
    # B 相对于 A 参考系的旋转，绕着 A 参考系的 z 轴旋转 q1 角度
    B.orient_axis(A, A.z, q1)
    # 设置 B 相对于 A 参考系的角速度
    B.set_ang_vel(A, u1 * A.z)
    # C 相对于 B 参考系的旋转，绕着 B 参考系的 z 轴旋转 q2 角度
    C.orient_axis(B, B.z, q2)
    # 设置 C 相对于 B 参考系的角速度
    C.set_ang_vel(B, u2 * B.z)

    # 创建点 Am, Bm, Cm，并设置它们在各自参考系中的初始速度和位置
    Am = Point('Am')
    Am.set_vel(A, 0)
    Bm = Point('Bm')
    Bm.set_pos(Am, B.x)
    Bm.set_vel(B, 0)
    Bm.set_vel(C, 0)
    Cm = Point('Cm')
    Cm.set_pos(Bm, C.x)
    Cm.set_vel(C, 0)
    
    # 备份 Cm 的速度字典
    temp = Cm._vel_dict.copy()
    # 断言 Cm 相对于 A 参考系的速度满足给定表达式
    assert Cm.vel(A) == (u1 * B.y + (u1 + u2) * C.y)
    # 恢复 Cm 的速度字典
    Cm._vel_dict = temp
    # 使用 v2pt_theory 方法计算 Cm 相对于 A 参考系的速度，并断言其满足给定表达式
    Cm.v2pt_theory(Bm, B, C)
    assert Cm.vel(A) == (u1 * B.y + (u1 + u2) * C.y)

def test_auto_point_acc_zero_vel():
    # 创建惯性参考系 N 和点 O
    N = ReferenceFrame('N')
    O = Point('O')
    # 设置点 O 在 N 参考系中的速度为 0
    O.set_vel(N, 0)
    # 断言点 O 在 N 参考系中的加速度为 0 * N.x
    assert O.acc(N) == 0 * N.x

def test_auto_point_acc_compute_vel():
    # 定义动态符号 t 和 q1，创建惯性参考系 N 和 A
    t = dynamicsymbols._t
    q1 = dynamicsymbols('q1')
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    # A 相对于 N 参考系的旋转，绕着 N 参考系的 z 轴旋转 q1 角度
    A.orient_axis(N, N.z, q1)

    # 创建点 O 和 P，并设置它们在 N 参考系中的速度
    O = Point('O')
    O.set_vel(N, 0)
    P = Point('P')
    P.set_pos(O, A.x)
    # 断言点 P 在 N 参考系中的加速度满足给定表达式
    assert P.acc(N) == -q1.diff(t) ** 2 * A.x + q1.diff(t, 2) * A.y

def test_auto_acc_derivative():
    # 测试 Point.acc 方法是否能正确计算串联两个连杆末端的加速度，
    # 获得最少信息。
    # 定义动态符号 q1, q2 和其一阶、二阶导数 u1, u2, v1, v2
    q1, q2 = dynamicsymbols('q1:3')
    u1, u2 = dynamicsymbols('q1:3', 1)
    v1, v2 = dynamicsymbols('q1:3', 2)
    # 创建参考系 A, B, C
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')
    # B 相对于 A 参考系的旋转，绕着 A 参考系的 z 轴旋转 q1 角度
    B.orient_axis(A, A.z, q1)
    # C 相对于 B 参考系的旋转，绕着 B 参考系的 z 轴旋转 q2 角度
    C.orient_axis(B, B.z, q2)

    # 创建点 Am, Bm, Cm，并设置它们在各自参考系中的初始速度和位置
    Am = Point('Am')
    Am.set_vel(A, 0)
    Bm = Point('Bm')
    Bm.set_pos(Am, B.x)
    Bm.set_vel(B, 0)
    Bm.set_vel(C, 0)
    Cm = Point('Cm')
    Cm.set_pos(Bm, C.x)
    Cm.set_vel(C, 0)

    # 备份 Bm 和 Cm 的速度、加速度字典
    Bm_vel_dict, Cm_vel_dict = Bm._vel_dict.copy(), Cm._vel_dict.copy()
    Bm_acc_dict, Cm_acc_dict = Bm._acc_dict.copy(), Cm._acc_dict.copy()
    # 计算 Cm 相对于 A 参考系的加速度，并断言其满足给定表达式
    check = -u1 ** 2 * B.x + v1 * B.y - (u1 + u2) ** 2 * C.x + (v1 + v2) * C.y
    assert Cm.acc(A) == check
    # 恢复 Bm 和 Cm 的速度、加速度字典
    Bm._vel_dict, Cm._vel_dict = Bm_vel_dict, Cm_vel_dict
    Bm._acc_dict, Cm._acc_dict = Bm_acc_dict, Cm_acc_dict
    # 调用对象 Bm 的 v2pt_theory 方法，传入参数 Am, A, B
    Bm.v2pt_theory(Am, A, B)
    # 调用对象 Cm 的 v2pt_theory 方法，传入参数 Bm, A, C
    Cm.v2pt_theory(Bm, A, C)
    # 调用对象 Bm 的 a2pt_theory 方法，传入参数 Am, A, B
    Bm.a2pt_theory(Am, A, B)
    # 断言调用对象 Cm 的 a2pt_theory 方法，传入参数 Bm, A, C 的返回值等于变量 check 的值
    assert Cm.a2pt_theory(Bm, A, C) == check
```