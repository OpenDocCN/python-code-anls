# `D:\src\scipysrc\sympy\sympy\physics\vector\tests\test_frame.py`

```
# 导入数学符号 pi 和符号变量的功能模块
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
# 导入简化表达式和三角函数的模块
from sympy.simplify import trigsimp
from sympy.functions.elementary.trigonometric import (cos, sin)
# 导入矩阵相关的模块
from sympy.matrices.dense import (eye, zeros)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
# 导入简化表达式的模块
from sympy.simplify.simplify import simplify
# 导入物理向量相关的模块
from sympy.physics.vector import (ReferenceFrame, Vector, CoordinateSym,
                                  dynamicsymbols, time_derivative, express,
                                  dot)
# 导入向量框架检查函数和异常处理相关的模块
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector.vector import VectorTypeError
# 导入测试框架相关的模块
from sympy.testing.pytest import raises
# 导入警告相关的模块
import warnings
# 导入 pickle 模块，用于序列化和反序列化对象
import pickle



def test_dict_list():
    """测试 ReferenceFrame 类的 _dict_list 方法"""

    # 创建多个参考框架对象
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')
    D = ReferenceFrame('D')
    E = ReferenceFrame('E')
    F = ReferenceFrame('F')

    # 设置多个参考框架对象的坐标轴方向
    B.orient_axis(A, A.x, 1.0)
    C.orient_axis(B, B.x, 1.0)
    D.orient_axis(C, C.x, 1.0)

    # 断言参考框架 D 调用 _dict_list 方法返回的列表
    assert D._dict_list(A, 0) == [D, C, B, A]

    # 设置参考框架 E 的坐标轴方向
    E.orient_axis(D, D.x, 1.0)

    # 断言参考框架 C 调用 _dict_list 方法返回的列表
    assert C._dict_list(A, 0) == [C, B, A]
    # 断言参考框架 C 调用 _dict_list 方法返回的列表
    assert C._dict_list(E, 0) == [C, D, E]

    # 断言对于第二个参数只能使用 0、1、2 的限制条件
    raises(ValueError, lambda: C._dict_list(E, 5))
    # 断言找不到连接路径的情况
    raises(ValueError, lambda: F._dict_list(A, 0))


def test_coordinate_vars():
    """测试坐标变量功能"""

    # 创建参考框架对象 A
    A = ReferenceFrame('A')

    # 断言 CoordinateSym 类的实例化结果
    assert CoordinateSym('Ax', A, 0) == A[0]
    assert CoordinateSym('Ax', A, 1) == A[1]
    assert CoordinateSym('Ax', A, 2) == A[2]

    # 断言当索引超过范围时抛出 ValueError 异常
    raises(ValueError, lambda: CoordinateSym('Ax', A, 3))

    # 创建动力学符号变量 q 和 qd
    q = dynamicsymbols('q')
    qd = dynamicsymbols('q', 1)

    # 断言 A[0]、A[1]、A[2] 是 CoordinateSym 类的实例
    assert isinstance(A[0], CoordinateSym) and \
           isinstance(A[1], CoordinateSym) and \
           isinstance(A[2], CoordinateSym)

    # 断言 A 的变量映射
    assert A.variable_map(A) == {A[0]: A[0], A[1]: A[1], A[2]: A[2]}

    # 创建参考框架对象 B，使其相对于 A 以 q 和 A.z 为轴旋转
    B = A.orientnew('B', 'Axis', [q, A.z])

    # 断言 B 的变量映射
    assert B.variable_map(A) == {
        B[2]: A[2],
        B[1]: -A[0]*sin(q) + A[1]*cos(q),
        B[0]: A[0]*cos(q) + A[1]*sin(q)
    }

    # 断言 A 的变量映射
    assert A.variable_map(B) == {
        A[0]: B[0]*cos(q) - B[1]*sin(q),
        A[1]: B[0]*sin(q) + B[1]*cos(q),
        A[2]: B[2]
    }

    # 断言时间导数的计算结果
    assert time_derivative(B[0], A) == -A[0]*sin(q)*qd + A[1]*cos(q)*qd
    assert time_derivative(B[1], A) == -A[0]*cos(q)*qd - A[1]*sin(q)*qd
    assert time_derivative(B[2], A) == 0

    # 断言表达式的转换结果
    assert express(B[0], A, variables=True) == A[0]*cos(q) + A[1]*sin(q)
    assert express(B[1], A, variables=True) == -A[0]*sin(q) + A[1]*cos(q)
    assert express(B[2], A, variables=True) == A[2]

    # 断言时间导数的计算结果
    assert time_derivative(A[0]*A.x + A[1]*A.y + A[2]*A.z, B) == A[1]*qd*A.x - A[0]*qd*A.y
    assert time_derivative(B[0]*B.x + B[1]*B.y + B[2]*B.z, A) == - B[1]*qd*B.x + B[0]*qd*B.y

    # 断言表达式的转换结果
    assert express(B[0]*B[1]*B[2], A, variables=True) == \
           A[2]*(-A[0]*sin(q) + A[1]*cos(q))*(A[0]*cos(q) + A[1]*sin(q))
    # 断言：验证时间导数是否为零
    assert (time_derivative(B[0]*B[1]*B[2], A) -
            (A[2]*(-A[0]**2*cos(2*q) -
             2*A[0]*A[1]*sin(2*q) +
             A[1]**2*cos(2*q))*qd)).trigsimp() == 0
    
    # 断言：验证表达式是否正确转换为A坐标系
    assert express(B[0]*B.x + B[1]*B.y + B[2]*B.z, A) == \
           (B[0]*cos(q) - B[1]*sin(q))*A.x + (B[0]*sin(q) + \
           B[1]*cos(q))*A.y + B[2]*A.z
    
    # 断言：验证表达式是否在转换为A坐标系时简化
    assert express(B[0]*B.x + B[1]*B.y + B[2]*B.z, A,
                   variables=True).simplify() == A[0]*A.x + A[1]*A.y + A[2]*A.z
    
    # 断言：验证表达式是否正确转换为B坐标系
    assert express(A[0]*A.x + A[1]*A.y + A[2]*A.z, B) == \
           (A[0]*cos(q) + A[1]*sin(q))*B.x + \
           (-A[0]*sin(q) + A[1]*cos(q))*B.y + A[2]*B.z
    
    # 断言：验证表达式是否在转换为B坐标系时简化
    assert express(A[0]*A.x + A[1]*A.y + A[2]*A.z, B,
                   variables=True).simplify() == B[0]*B.x + B[1]*B.y + B[2]*B.z
    
    # 创建一个新的坐标系N，沿z轴旋转-q角度
    N = B.orientnew('N', 'Axis', [-q, B.z])
    # 断言：验证变量映射是否正确
    assert ({k: v.simplify() for k, v in N.variable_map(A).items()} ==
            {N[0]: A[0], N[2]: A[2], N[1]: A[1]})
    
    # 创建一个新的坐标系C，沿A.x + A.y + A.z轴旋转q角度
    C = A.orientnew('C', 'Axis', [q, A.x + A.y + A.z])
    mapping = A.variable_map(C)
    
    # 断言：验证A[0]的映射表达式是否简化
    assert trigsimp(mapping[A[0]]) == (2*C[0]*cos(q)/3 + C[0]/3 -
                                       2*C[1]*sin(q + pi/6)/3 +
                                       C[1]/3 - 2*C[2]*cos(q + pi/3)/3 +
                                       C[2]/3)
    
    # 断言：验证A[1]的映射表达式是否简化
    assert trigsimp(mapping[A[1]]) == -2*C[0]*cos(q + pi/3)/3 + \
           C[0]/3 + 2*C[1]*cos(q)/3 + C[1]/3 - 2*C[2]*sin(q + pi/6)/3 + C[2]/3
    
    # 断言：验证A[2]的映射表达式是否简化
    assert trigsimp(mapping[A[2]]) == -2*C[0]*sin(q + pi/6)/3 + C[0]/3 - \
           2*C[1]*cos(q + pi/3)/3 + C[1]/3 + 2*C[2]*cos(q)/3 + C[2]/3
# 定义一个测试函数，用于测试角速度相关的计算
def test_ang_vel():
    # 定义四个动力学符号变量 q1, q2, q3, q4
    q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
    # 定义四个动力学符号变量的一阶导数 q1d, q2d, q3d, q4d
    q1d, q2d, q3d, q4d = dynamicsymbols('q1 q2 q3 q4', 1)
    # 创建一个惯性参考框架 N
    N = ReferenceFrame('N')
    # 在参考框架 N 中创建一个新的参考框架 A，并沿着 N.z 轴旋转 q1 角度
    A = N.orientnew('A', 'Axis', [q1, N.z])
    # 在参考框架 A 中创建一个新的参考框架 B，并沿着 A.x 轴旋转 q2 角度
    B = A.orientnew('B', 'Axis', [q2, A.x])
    # 在参考框架 B 中创建一个新的参考框架 C，并沿着 B.y 轴旋转 q3 角度
    C = B.orientnew('C', 'Axis', [q3, B.y])
    # 在参考框架 N 中创建一个新的参考框架 D，并沿着 N.y 轴旋转 q4 角度
    D = N.orientnew('D', 'Axis', [q4, N.y])
    # 定义三个动力学符号变量 u1, u2, u3
    u1, u2, u3 = dynamicsymbols('u1 u2 u3')
    
    # 断言：计算并验证角速度
    assert A.ang_vel_in(N) == q1d*N.z
    assert B.ang_vel_in(N) == q1d*A.z + q2d*B.x
    assert C.ang_vel_in(N) == q1d*A.z + q2d*B.x + q3d*C.y

    # 在参考框架 N 中创建一个新的参考框架 A2，并沿着 N.y 轴旋转 q4 角度
    A2 = N.orientnew('A2', 'Axis', [q4, N.y])
    assert N.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == -q1d*N.z
    assert N.ang_vel_in(B) == -q1d*A.z - q2d*B.x
    assert N.ang_vel_in(C) == -q1d*A.z - q2d*B.x - q3d*B.y
    assert N.ang_vel_in(A2) == -q4d*N.y

    assert A.ang_vel_in(A) == 0
    assert A.ang_vel_in(B) == -q2d*B.x
    assert A.ang_vel_in(C) == -q2d*B.x - q3d*B.y
    assert A.ang_vel_in(A2) == q1d*N.z - q4d*N.y

    assert B.ang_vel_in(A) == q2d*A.x
    assert B.ang_vel_in(B) == 0
    assert B.ang_vel_in(C) == -q3d*B.y
    assert B.ang_vel_in(A2) == q1d*A.z + q2d*A.x - q4d*N.y

    assert C.ang_vel_in(A) == q2d*A.x + q3d*C.y
    assert C.ang_vel_in(B) == q3d*B.y
    assert C.ang_vel_in(C) == 0
    assert C.ang_vel_in(A2) == q1d*A.z + q2d*A.x + q3d*B.y - q4d*N.y

    assert A2.ang_vel_in(N) == q4d*A2.y
    assert A2.ang_vel_in(A) == q4d*A2.y - q1d*N.z
    assert A2.ang_vel_in(B) == q4d*N.y - q1d*A.z - q2d*A.x
    assert A2.ang_vel_in(C) == q4d*N.y - q1d*A.z - q2d*A.x - q3d*B.y
    assert A2.ang_vel_in(A2) == 0

    # 设置参考框架 C 的角速度为 u1*C.x + u2*C.y + u3*C.z
    C.set_ang_vel(N, u1*C.x + u2*C.y + u3*C.z)
    assert C.ang_vel_in(N) == u1*C.x + u2*C.y + u3*C.z
    assert N.ang_vel_in(C) == -u1*C.x - u2*C.y - u3*C.z + q4d*D.y

    # 定义另一个动力学符号变量 q0
    q0 = dynamicsymbols('q0')
    # 定义 q0 的一阶导数 q0d
    q0d = dynamicsymbols('q0', 1)
    # 在参考框架 N 中创建一个新的参考框架 E，使用四元数表示 (q0, q1, q2, q3)
    E = N.orientnew('E', 'Quaternion', (q0, q1, q2, q3))
    # 计算并验证 E 相对于 N 的角速度
    assert E.ang_vel_in(N) == (
        2 * (q1d * q0 + q2d * q3 - q3d * q2 - q0d * q1) * E.x +
        2 * (q2d * q0 + q3d * q1 - q1d * q3 - q0d * q2) * E.y +
        2 * (q3d * q0 + q1d * q2 - q2d * q1 - q0d * q3) * E.z)

    # 在参考框架 N 中创建一个新的参考框架 F，使用欧拉角表示 (q1, q2, q3)，顺序 313
    F = N.orientnew('F', 'Body', (q1, q2, q3), 313)
    # 计算并验证 F 相对于 N 的角速度
    assert F.ang_vel_in(N) == ((sin(q2)*sin(q3)*q1d + cos(q3)*q2d)*F.x +
        (sin(q2)*cos(q3)*q1d - sin(q3)*q2d)*F.y + (cos(q2)*q1d + q3d)*F.z)

    # 在参考框架 N 中创建一个新的参考框架 G，围绕 N.x + N.y 轴旋转 q1 角度
    G = N.orientnew('G', 'Axis', (q1, N.x + N.y))
    # 计算并验证 G 相对于 N 的角速度
    assert G.ang_vel_in(N) == q1d * (N.x + N.y).normalize()
    assert N.ang_vel_in(G) == -q1d * (N.x + N.y).normalize()


# 定义一个测试函数，用于测试方向余弦矩阵相关的计算
def test_dcm():
    # 定义四个动力学符号变量 q1, q2, q3, q4
    q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
    # 创建一个惯性参考框架 N
    N = ReferenceFrame('N')
    # 在参考框架 N 中创建一个新的参考框架 A，并沿着 N.z 轴旋转 q1 角度
    A = N.orientnew('A', 'Axis', [q1, N.z])
    # 在参考框架 A 中创建一个新的参考框架 B，并沿着 A.x 轴旋转 q2 角度
    B = A.orientnew('B', 'Axis', [q2, A.x])
    # 在参考框架 B 中创建一个新的参考框架 C，并沿着 B.y 轴旋转 q3 角度
    C = B.orientnew('C', 'Axis', [q3, B.y])
    # 创建一个新的方向对象D，使用'Axis'方式以q4和N.y作为参数进行定向
    D = N.orientnew('D', 'Axis', [q4, N.y])
    # 创建一个新的方向对象E，使用'Space'方式以q1、q2、q3作为参数进行定向，并指定坐标轴顺序为'123'
    E = N.orientnew('E', 'Space', [q1, q2, q3], '123')
    # 断言验证矩阵C的方向余弦矩阵是否与给定的Matrix对象相等
    assert N.dcm(C) == Matrix([
        [- sin(q1) * sin(q2) * sin(q3) + cos(q1) * cos(q3), - sin(q1) *
        cos(q2), sin(q1) * sin(q2) * cos(q3) + sin(q3) * cos(q1)], [sin(q1) *
        cos(q3) + sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2), sin(q1) *
            sin(q3) - sin(q2) * cos(q1) * cos(q3)], [- sin(q3) * cos(q2), sin(q2),
        cos(q2) * cos(q3)]])
    # 这里有些微妙。在断言中使用simplify函数是否可行？
    # 创建测试矩阵test_mat，验证D到C的方向余弦矩阵减去给定矩阵的差是否为零矩阵
    test_mat = D.dcm(C) - Matrix(
        [[cos(q1) * cos(q3) * cos(q4) - sin(q3) * (- sin(q4) * cos(q2) +
        sin(q1) * sin(q2) * cos(q4)), - sin(q2) * sin(q4) - sin(q1) *
            cos(q2) * cos(q4), sin(q3) * cos(q1) * cos(q4) + cos(q3) * (- sin(q4) *
        cos(q2) + sin(q1) * sin(q2) * cos(q4))], [sin(q1) * cos(q3) +
        sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2), sin(q1) * sin(q3) -
            sin(q2) * cos(q1) * cos(q3)], [sin(q4) * cos(q1) * cos(q3) -
        sin(q3) * (cos(q2) * cos(q4) + sin(q1) * sin(q2) * sin(q4)), sin(q2) *
                cos(q4) - sin(q1) * sin(q4) * cos(q2), sin(q3) * sin(q4) * cos(q1) +
                cos(q3) * (cos(q2) * cos(q4) + sin(q1) * sin(q2) * sin(q4))]])
    # 验证测试矩阵是否可以展开为零矩阵
    assert test_mat.expand() == zeros(3, 3)
    # 断言验证矩阵E的方向余弦矩阵是否与给定的Matrix对象相等
    assert E.dcm(N) == Matrix(
        [[cos(q2)*cos(q3), sin(q3)*cos(q2), -sin(q2)],
        [sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1), sin(q1)*sin(q2)*sin(q3) +
        cos(q1)*cos(q3), sin(q1)*cos(q2)], [sin(q1)*sin(q3) +
        sin(q2)*cos(q1)*cos(q3), - sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1),
         cos(q1)*cos(q2)]])
def test_w_diff_dcm1():
    # 定义两个参考系对象 A 和 B
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')

    # 定义动态符号变量 Cij
    c11, c12, c13 = dynamicsymbols('C11 C12 C13')
    c21, c22, c23 = dynamicsymbols('C21 C22 C23')
    c31, c32, c33 = dynamicsymbols('C31 C32 C33')

    # 定义动态符号变量的一阶导数 Cijd
    c11d, c12d, c13d = dynamicsymbols('C11 C12 C13', level=1)
    c21d, c22d, c23d = dynamicsymbols('C21 C22 C23', level=1)
    c31d, c32d, c33d = dynamicsymbols('C31 C32 C33', level=1)

    # 构建方向余弦矩阵 DCM
    DCM = Matrix([
        [c11, c12, c13],
        [c21, c22, c23],
        [c31, c32, c33]
    ])

    # 用 DCM 矩阵将参考系 B 定向到参考系 A
    B.orient(A, 'DCM', DCM)

    # 计算 B 参考系在 A 参考系中的单位向量 b1a, b2a, b3a
    b1a = (B.x).express(A)
    b2a = (B.y).express(A)
    b3a = (B.z).express(A)

    # 设置 B 参考系相对于 A 参考系的角速度
    # 根据 Kane 1985 第 2.1 节中的公式 (2.1.1)
    B.set_ang_vel(A, B.x*(dot((b3a).dt(A), B.y))
                   + B.y*(dot((b1a).dt(A), B.z))
                   + B.z*(dot((b2a).dt(A), B.x)))

    # 断言 B 参考系相对于 A 参考系的角速度等于表达式 expr
    # 这里是 Kane 1985 第 2.1 节中的公式 (2.1.21)
    expr = (  (c12*c13d + c22*c23d + c32*c33d)*B.x
            + (c13*c11d + c23*c21d + c33*c31d)*B.y
            + (c11*c12d + c21*c22d + c31*c32d)*B.z)
    assert B.ang_vel_in(A) - expr == 0

def test_w_diff_dcm2():
    # 定义动态符号变量 q1, q2, q3
    q1, q2, q3 = dynamicsymbols('q1:4')

    # 定义惯性参考系 N
    N = ReferenceFrame('N')

    # 根据动态变量 q1 创建参考系 A，并将其定向到 N 参考系上
    A = N.orientnew('A', 'axis', [q1, N.x])

    # 根据动态变量 q2 创建参考系 B，并将其定向到 A 参考系上
    B = A.orientnew('B', 'axis', [q2, A.y])

    # 根据动态变量 q3 创建参考系 C，并将其定向到 B 参考系上
    C = B.orientnew('C', 'axis', [q3, B.z])

    # 获取 C 参考系相对于 N 参考系的方向余弦矩阵的转置 DCM
    DCM = C.dcm(N).T

    # 根据 DCM 矩阵将参考系 D 定向到 N 参考系上
    D = N.orientnew('D', 'DCM', DCM)

    # 断言参考系 D 和 C 在 N 参考系中的方向余弦矩阵相同
    assert D.dcm(N) == C.dcm(N) == Matrix([
        [cos(q2)*cos(q3), sin(q1)*sin(q2)*cos(q3) + sin(q3)*cos(q1), sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3)],
        [-sin(q3)*cos(q2), -sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3), sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1)],
        [sin(q2), -sin(q1)*cos(q2), cos(q1)*cos(q2)]
    ])

    # 断言参考系 D 和 C 在 N 参考系中的角速度差为零
    assert (D.ang_vel_in(N) - C.ang_vel_in(N)).express(N).simplify() == 0

def test_orientnew_respects_parent_class():
    # 定义一个自定义的参考系类 MyReferenceFrame 继承自 ReferenceFrame
    class MyReferenceFrame(ReferenceFrame):
        pass

    # 创建 MyReferenceFrame 类的实例 B
    B = MyReferenceFrame('B')

    # 根据 B 参考系创建一个新的参考系 C，并确保 C 是 MyReferenceFrame 的实例
    C = B.orientnew('C', 'Axis', [0, B.x])
    assert isinstance(C, MyReferenceFrame)

def test_orientnew_respects_input_indices():
    # 定义惯性参考系 N
    N = ReferenceFrame('N')

    # 定义动态符号变量 q1
    q1 = dynamicsymbols('q1')

    # 根据动态变量 q1 创建参考系 A，并指定索引为默认值
    A = N.orientnew('a', 'Axis', [q1, N.z])

    # 修改默认的索引值
    minds = [x+'1' for x in N.indices]

    # 根据动态变量 q1 创建参考系 B，并指定修改后的索引值
    B = N.orientnew('b', 'Axis', [q1, N.z], indices=minds)

    # 断言参考系 A 和 N 的索引与 B 的索引相同
    assert N.indices == A.indices
    assert B.indices == minds

def test_orientnew_respects_input_latexs():
    # 定义惯性参考系 N
    N = ReferenceFrame('N')

    # 定义动态符号变量 q1
    q1 = dynamicsymbols('q1')

    # 根据动态变量 q1 创建参考系 A，并获取其默认的 LaTeX 向量符号
    A = N.orientnew('a', 'Axis', [q1, N.z])

    # 创建默认的和替代的 LaTeX 向量符号
    # 这里的注释缺失，需要补充
    # 定义默认的 LaTeX 向量表达式列表，格式化 A 对象的名称和索引
    def_latex_vecs = [(r"\mathbf{\hat{%s}_%s}" % (A.name.lower(),
                      A.indices[0])), (r"\mathbf{\hat{%s}_%s}" %
                      (A.name.lower(), A.indices[1])),
                      (r"\mathbf{\hat{%s}_%s}" % (A.name.lower(),
                      A.indices[2]))]

    # 设置新的名称和索引列表，将每个索引加上 '1'
    name = 'b'
    indices = [x+'1' for x in N.indices]
    # 根据新的名称和索引列表创建新的 LaTeX 向量表达式列表
    new_latex_vecs = [(r"\mathbf{\hat{%s}_{%s}}" % (name.lower(),
                      indices[0])), (r"\mathbf{\hat{%s}_{%s}}" %
                      (name.lower(), indices[1])),
                      (r"\mathbf{\hat{%s}_{%s}}" % (name.lower(),
                      indices[2]))]

    # 使用指定的轴和 LaTeX 向量表达式列表创建新的方向
    B = N.orientnew(name, 'Axis', [q1, N.z], latexs=new_latex_vecs)

    # 断言 A 对象的 LaTeX 向量表达式与默认表达式列表相同
    assert A.latex_vecs == def_latex_vecs
    # 断言 B 对象的 LaTeX 向量表达式与新创建的表达式列表相同
    assert B.latex_vecs == new_latex_vecs
    # 断言 B 对象的索引与新创建的索引列表不同
    assert B.indices != indices
def test_orientnew_respects_input_variables():
    N = ReferenceFrame('N')  # 创建一个新的参考系对象 'N'
    q1 = dynamicsymbols('q1')  # 创建一个动态符号 'q1'
    A = N.orientnew('a', 'Axis', [q1, N.z])  # 在参考系 'N' 中建立一个新的方向 'a'，沿着轴 [q1, N.z]

    # 构建非标准变量名
    name = 'b'  # 设置变量名 'b'
    new_variables = ['notb_'+x+'1' for x in N.indices]  # 根据参考系 'N' 的索引创建新变量名列表
    B = N.orientnew(name, 'Axis', [q1, N.z], variables=new_variables)  # 在参考系 'N' 中建立一个新的方向 'b'，沿着轴 [q1, N.z]，使用新变量名列表

    for j,var in enumerate(A.varlist):
        assert var.name == A.name + '_' + A.indices[j]  # 检查方向 'a' 的变量名是否符合预期命名规则

    for j,var in enumerate(B.varlist):
        assert var.name == new_variables[j]  # 检查方向 'b' 的变量名是否符合预期的新变量名规则

def test_issue_10348():
    u = dynamicsymbols('u:3')  # 创建一个动态符号列表 'u'，包含三个元素
    I = ReferenceFrame('I')  # 创建一个新的参考系对象 'I'
    I.orientnew('A', 'space', u, 'XYZ')  # 在参考系 'I' 中建立一个新的方向 'A'，使用动态符号列表 'u'，空间方向为 'XYZ'

def test_issue_11503():
    A = ReferenceFrame("A")  # 创建一个新的参考系对象 'A'
    A.orientnew("B", "Axis", [35, A.y])  # 在参考系 'A' 中建立一个新的方向 'B'，沿着轴 [35, A.y]
    C = ReferenceFrame("C")  # 创建一个新的参考系对象 'C'
    A.orient(C, "Axis", [70, C.z])  # 使参考系 'A' 相对于参考系 'C' 进行方向转换，沿着轴 [70, C.z]

def test_partial_velocity():
    N = ReferenceFrame('N')  # 创建一个新的参考系对象 'N'
    A = ReferenceFrame('A')  # 创建一个新的参考系对象 'A'

    u1, u2 = dynamicsymbols('u1, u2')  # 创建动态符号 'u1' 和 'u2'

    A.set_ang_vel(N, u1 * A.x + u2 * N.y)  # 设置参考系 'A' 相对于参考系 'N' 的角速度

    assert N.partial_velocity(A, u1) == -A.x  # 验证关于动态符号 'u1' 的偏导数
    assert N.partial_velocity(A, u1, u2) == (-A.x, -N.y)  # 验证关于动态符号 'u1' 和 'u2' 的偏导数

    assert A.partial_velocity(N, u1) == A.x  # 验证关于动态符号 'u1' 的偏导数
    assert A.partial_velocity(N, u1, u2) == (A.x, N.y)  # 验证关于动态符号 'u1' 和 'u2' 的偏导数

    assert N.partial_velocity(N, u1) == 0  # 验证参考系 'N' 关于其自身的偏导数
    assert A.partial_velocity(A, u1) == 0  # 验证参考系 'A' 关于其自身的偏导数

def test_issue_11498():
    A = ReferenceFrame('A')  # 创建一个新的参考系对象 'A'
    B = ReferenceFrame('B')  # 创建一个新的参考系对象 'B'

    # Identity transformation
    A.orient(B, 'DCM', eye(3))  # 使用单位矩阵进行参考系 'A' 相对于参考系 'B' 的方向转换
    assert A.dcm(B) == Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 验证 'A' 相对于 'B' 的方向余弦矩阵
    assert B.dcm(A) == Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 验证 'B' 相对于 'A' 的方向余弦矩阵

    # x -> y
    # y -> -z
    # z -> -x
    A.orient(B, 'DCM', Matrix([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))  # 使用指定矩阵进行 'A' 相对于 'B' 的方向转换
    assert B.dcm(A) == Matrix([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])  # 验证 'B' 相对于 'A' 的方向余弦矩阵
    assert A.dcm(B) == Matrix([[0, 0, -1], [1, 0, 0], [0, -1, 0]])  # 验证 'A' 相对于 'B' 的方向余弦矩阵
    assert B.dcm(A).T == A.dcm(B)  # 验证 'B' 相对于 'A' 的转置矩阵与 'A' 相对于 'B' 的方向余弦矩阵相等

def test_reference_frame():
    raises(TypeError, lambda: ReferenceFrame(0))  # 验证创建参考系时传入非法参数会引发类型错误异常
    raises(TypeError, lambda: ReferenceFrame('N', 0))  # 验证创建参考系时传入非法参数会引发类型错误异常
    raises(ValueError, lambda: ReferenceFrame('N', [0, 1]))  # 验证创建参考系时传入非法参数会引发值错误异常
    raises(TypeError, lambda: ReferenceFrame('N', [0, 1, 2]))  # 验证创建参考系时传入非法参数会引发类型错误异常
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], 0))  # 验证创建参考系时传入非法参数会引发类型错误异常
    raises(ValueError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], [0, 1]))  # 验证创建参考系时传入非法参数会引发值错误异常
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], [0, 1, 2]))  # 验证创建参考系时传入非法参数会引发类型错误异常
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], ['a', 'b', 'c'], 0))  # 验证创建参考系时传入非法参数会引发类型错误异常
    raises(ValueError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], ['a', 'b', 'c'], [0, 1]))  # 验证创建参考系时传入非法参数会引发值错误异常
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], ['a', 'b', 'c'], [0, 1, 2]))  # 验证创建参考系时传入非法参数会引发类型错误异常

    N = ReferenceFrame('N')  # 创建一个新的参考系对象 'N'
    assert N[0] == CoordinateSym('N_x', N, 0)  # 验证参考系 'N' 的第一个坐标符号对象
    assert N[1] == CoordinateSym('N_y', N, 1)  # 验证参考系 'N' 的第二个坐标符号对象
    assert N[2] == CoordinateSym('N_z', N, 2)
    # 断言N['c']等于N.z，验证索引操作的正确性
    assert N['c'] == N.z
    # 使用lambda函数和raises函数验证访问N['d']时引发ValueError异常
    raises(ValueError, lambda: N['d'])
    # 断言str(N)返回'N'，验证对象N的字符串表示
    assert str(N) == 'N'

    # 创建参考框架A和B
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    # 创建符号变量q0, q1, q2, q3
    q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
    # 使用lambda函数和raises函数验证A.orient(B, 'DCM', 0)引发TypeError异常
    raises(TypeError, lambda: A.orient(B, 'DCM', 0))
    # 使用lambda函数和raises函数验证B.orient(N, 'Space', [q1, q2, q3], '222')引发TypeError异常
    raises(TypeError, lambda: B.orient(N, 'Space', [q1, q2, q3], '222'))
    # 使用lambda函数和raises函数验证B.orient(N, 'Axis', [q1, N.x + 2 * N.y], '222')引发TypeError异常
    raises(TypeError, lambda: B.orient(N, 'Axis', [q1, N.x + 2 * N.y], '222'))
    # 使用lambda函数和raises函数验证B.orient(N, 'Axis', q1)引发TypeError异常
    raises(TypeError, lambda: B.orient(N, 'Axis', q1))
    # 使用lambda函数和raises函数验证B.orient(N, 'Axis', [q1])引发IndexError异常
    raises(IndexError, lambda: B.orient(N, 'Axis', [q1]))
    # 使用lambda函数和raises函数验证B.orient(N, 'Quaternion', [q0, q1, q2, q3], '222')引发TypeError异常
    raises(TypeError, lambda: B.orient(N, 'Quaternion', [q0, q1, q2, q3], '222'))
    # 使用lambda函数和raises函数验证B.orient(N, 'Quaternion', q0)引发TypeError异常
    raises(TypeError, lambda: B.orient(N, 'Quaternion', q0))
    # 使用lambda函数和raises函数验证B.orient(N, 'Quaternion', [q0, q1, q2])引发TypeError异常
    raises(TypeError, lambda: B.orient(N, 'Quaternion', [q0, q1, q2]))
    # 使用lambda函数和raises函数验证B.orient(N, 'Foo', [q0, q1, q2])引发NotImplementedError异常
    raises(NotImplementedError, lambda: B.orient(N, 'Foo', [q0, q1, q2]))
    # 使用lambda函数和raises函数验证B.orient(N, 'Body', [q1, q2], '232')引发TypeError异常
    raises(TypeError, lambda: B.orient(N, 'Body', [q1, q2], '232'))
    # 使用lambda函数和raises函数验证B.orient(N, 'Space', [q1, q2], '232')引发TypeError异常
    raises(TypeError, lambda: B.orient(N, 'Space', [q1, q2], '232'))

    # 将B参考框架相对于N设置角加速度为0
    N.set_ang_acc(B, 0)
    # 断言N相对于B的角加速度为Vector(0)
    assert N.ang_acc_in(B) == Vector(0)
    # 将N参考框架相对于B设置角速度为0
    N.set_ang_vel(B, 0)
    # 断言N相对于B的角速度为Vector(0)
    assert N.ang_vel_in(B) == Vector(0)
def test_check_frame():
    # 调用 _check_frame(0) 应当引发 VectorTypeError 异常
    raises(VectorTypeError, lambda: _check_frame(0))


def test_dcm_diff_16824():
    # NOTE : 这是 PR 14758 引入的 bug 的回归测试
    # 16824 号问题的标识，并且由 PR 16828 解决。
    
    # 这是 Kane & Levinson 的 1985 年书中第 264 页上问题 2.2 的解答。

    # 定义三个动力符号
    q1, q2, q3 = dynamicsymbols('q1:4')

    # 计算正弦和余弦值
    s1 = sin(q1)
    c1 = cos(q1)
    s2 = sin(q2)
    c2 = cos(q2)
    s3 = sin(q3)
    c3 = cos(q3)

    # 构建方向余弦矩阵 (Direction Cosine Matrix, DCM)
    dcm = Matrix([[c2*c3, s1*s2*c3 - s3*c1, c1*s2*c3 + s3*s1],
                  [c2*s3, s1*s2*s3 + c3*c1, c1*s2*s3 - c3*s1],
                  [-s2,   s1*c2,            c1*c2]])

    # 创建参考坐标系 A 和 B
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')

    # 将 B 参考坐标系用 DCM 方式相对于 A 参考坐标系定向
    B.orient(A, 'DCM', dcm)

    # 计算 A 到 B 的角速度
    AwB = B.ang_vel_in(A)

    # 计算 alpha2 和 beta2
    alpha2 = s3*c2*q1.diff() + c3*q2.diff()
    beta2 = s1*c2*q3.diff() + c1*q2.diff()

    # 断言角速度与导数的关系
    assert simplify(AwB.dot(A.y) - alpha2) == 0
    assert simplify(AwB.dot(B.y) - beta2) == 0

def test_orient_explicit():
    # 定义动力符号
    cxx, cyy, czz = dynamicsymbols('c_{xx}, c_{yy}, c_{zz}')
    cxy, cxz, cyx = dynamicsymbols('c_{xy}, c_{xz}, c_{yx}')
    cyz, czx, czy = dynamicsymbols('c_{yz}, c_{zx}, c_{zy}')
    dcxx, dcyy, dczz = dynamicsymbols('c_{xx}, c_{yy}, c_{zz}', 1)
    dcxy, dcxz, dcyx = dynamicsymbols('c_{xy}, c_{xz}, c_{yx}', 1)
    dcyz, dczx, dczy = dynamicsymbols('c_{yz}, c_{zx}, c_{zy}', 1)

    # 创建参考坐标系 A 和 B
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')

    # 构建方向余弦矩阵 B_C_A 和角速度 B_w_A
    B_C_A = Matrix([[cxx, cxy, cxz],
                    [cyx, cyy, cyz],
                    [czx, czy, czz]])
    B_w_A = ((cyx*dczx + cyy*dczy + cyz*dczz)*B.x +
            (czx*dcxx + czy*dcxy + czz*dcxz)*B.y +
            (cxx*dcyx + cxy*dcyy + cxz*dcyz)*B.z)

    # A 参考坐标系相对于 B 参考坐标系使用显式 DCM 方式定向
    A.orient_explicit(B, B_C_A)

    # 断言方向余弦矩阵和角速度的正确性
    assert B.dcm(A) == B_C_A
    assert A.ang_vel_in(B) == B_w_A
    assert B.ang_vel_in(A) == -B_w_A

def test_orient_dcm():
    # 定义动力符号
    cxx, cyy, czz = dynamicsymbols('c_{xx}, c_{yy}, c_{zz}')
    cxy, cxz, cyx = dynamicsymbols('c_{xy}, c_{xz}, c_{yx}')
    cyz, czx, czy = dynamicsymbols('c_{yz}, c_{zx}, c_{zy}')

    # 创建参考坐标系 A 和 B
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')

    # 构建方向余弦矩阵 B_C_A
    B_C_A = Matrix([[cxx, cxy, cxz],
                    [cyx, cyy, cyz],
                    [czx, czy, czz]])

    # B 参考坐标系相对于 A 参考坐标系使用 DCM 方式定向
    B.orient_dcm(A, B_C_A)

    # 断言方向余弦矩阵的正确性
    assert B.dcm(A) == B_C_A

def test_orient_axis():
    # 创建参考坐标系 A 和 B
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')

    # 使用不同的轴向方式定向 A 相对于 B
    A.orient_axis(B,-B.x, 1)
    A1 = A.dcm(B)
    A.orient_axis(B, B.x, -1)
    A2 = A.dcm(B)
    A.orient_axis(B, 1, -B.x)
    A3 = A.dcm(B)

    # 断言三种定向方式得到相同的方向余弦矩阵
    assert A1 == A2
    assert A2 == A3

    # 断言不支持的轴向方式会引发 TypeError 异常
    raises(TypeError, lambda: A.orient_axis(B, 1, 1))

def test_orient_body():
    # 创建参考坐标系 A 和 B
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')

    # 使用 'XYX' 顺序的旋转方式将 B 参考坐标系固定到 A 参考坐标系
    B.orient_body_fixed(A, (1,1,0), 'XYX')

    # 断言方向余弦矩阵的正确性
    assert B.dcm(A) == Matrix([[cos(1), sin(1)**2, -sin(1)*cos(1)], [0, cos(1), sin(1)], [sin(1), -sin(1)*cos(1), cos(1)**2]])

def test_orient_body_advanced():
    # 这个测试函数还没有实现，保留空白
    pass
    # 定义三个动力学符号 q1, q2, q3
    q1, q2, q3 = dynamicsymbols('q1:4')
    # 定义三个常数符号 c1, c2, c3
    c1, c2, c3 = symbols('c1:4')
    # 定义三个动力学符号的一阶导数 u1, u2, u3
    u1, u2, u3 = dynamicsymbols('q1:4', 1)

    # 测试所有符号为动力学符号的情况
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    # B 相对于 A 以 (q1, q2, q3) 为欧拉角，'zxy' 是旋转顺序
    B.orient_body_fixed(A, (q1, q2, q3), 'zxy')
    # 断言 A 相对于 B 的方向余弦矩阵
    assert A.dcm(B) == Matrix([
        [-sin(q1) * sin(q2) * sin(q3) + cos(q1) * cos(q3), -sin(q1) * cos(q2),
         sin(q1) * sin(q2) * cos(q3) + sin(q3) * cos(q1)],
        [sin(q1) * cos(q3) + sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2),
         sin(q1) * sin(q3) - sin(q2) * cos(q1) * cos(q3)],
        [-sin(q3) * cos(q2), sin(q2), cos(q2) * cos(q3)]])
    # 断言 B 相对于 A 的角速度矢量在 B 参考系下的矩阵表示
    assert B.ang_vel_in(A).to_matrix(B) == Matrix([
        [-sin(q3) * cos(q2) * u1 + cos(q3) * u2],
        [sin(q2) * u1 + u3],
        [sin(q3) * u2 + cos(q2) * cos(q3) * u1]])

    # 测试其中一个符号为常数的情况
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    # B 相对于 A 以 (q1, c2, q3) 为欧拉角，131 是固定轴旋转
    B.orient_body_fixed(A, (q1, c2, q3), 131)
    # 断言 A 相对于 B 的方向余弦矩阵
    assert A.dcm(B) == Matrix([
        [cos(c2), -sin(c2) * cos(q3), sin(c2) * sin(q3)],
        [sin(c2) * cos(q1), -sin(q1) * sin(q3) + cos(c2) * cos(q1) * cos(q3),
         -sin(q1) * cos(q3) - sin(q3) * cos(c2) * cos(q1)],
        [sin(c2) * sin(q1), sin(q1) * cos(c2) * cos(q3) + sin(q3) * cos(q1),
         -sin(q1) * sin(q3) * cos(c2) + cos(q1) * cos(q3)]])
    # 断言 B 相对于 A 的角速度矢量在 B 参考系下的矩阵表示
    assert B.ang_vel_in(A).to_matrix(B) == Matrix([
        [cos(c2) * u1 + u3],
        [-sin(c2) * cos(q3) * u1],
        [sin(c2) * sin(q3) * u1]])

    # 测试所有符号均为常数的情况
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    # B 相对于 A 以 (c1, c2, c3) 为欧拉角，123 是固定轴旋转
    B.orient_body_fixed(A, (c1, c2, c3), 123)
    # 断言 B 相对于 A 的角速度矢量为零矢量
    assert B.ang_vel_in(A) == Vector(0)
def test_orient_space_advanced():
    # space fixed is in the end like body fixed only in opposite order
    # 定义动力学符号
    q1, q2, q3 = dynamicsymbols('q1:4')
    # 定义常数符号
    c1, c2, c3 = symbols('c1:4')
    # 定义动力学符号及其一阶导数
    u1, u2, u3 = dynamicsymbols('q1:4', 1)

    # Test with everything as dynamicsymbols
    # 创建参考系 A 和 B
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    # B 相对于 A 使用 yxz 顺序的空间固定方向余弦矩阵
    B.orient_space_fixed(A, (q3, q2, q1), 'yxz')
    # 验证 A 到 B 的方向余弦矩阵
    assert A.dcm(B) == Matrix([
        [-sin(q1) * sin(q2) * sin(q3) + cos(q1) * cos(q3), -sin(q1) * cos(q2),
         sin(q1) * sin(q2) * cos(q3) + sin(q3) * cos(q1)],
        [sin(q1) * cos(q3) + sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2),
         sin(q1) * sin(q3) - sin(q2) * cos(q1) * cos(q3)],
        [-sin(q3) * cos(q2), sin(q2), cos(q2) * cos(q3)]])
    # 验证 B 相对于 A 的角速度矢量
    assert B.ang_vel_in(A).to_matrix(B) == Matrix([
        [-sin(q3) * cos(q2) * u1 + cos(q3) * u2],
        [sin(q2) * u1 + u3],
        [sin(q3) * u2 + cos(q2) * cos(q3) * u1]])

    # Test with constant symbol
    # 创建新的参考系 A 和 B
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    # B 相对于 A 使用指定参数的空间固定方向余弦矩阵
    B.orient_space_fixed(A, (q3, c2, q1), 131)
    # 验证 A 到 B 的方向余弦矩阵
    assert A.dcm(B) == Matrix([
        [cos(c2), -sin(c2) * cos(q3), sin(c2) * sin(q3)],
        [sin(c2) * cos(q1), -sin(q1) * sin(q3) + cos(c2) * cos(q1) * cos(q3),
         -sin(q1) * cos(q3) - sin(q3) * cos(c2) * cos(q1)],
        [sin(c2) * sin(q1), sin(q1) * cos(c2) * cos(q3) + sin(q3) * cos(q1),
         -sin(q1) * sin(q3) * cos(c2) + cos(q1) * cos(q3)]])
    # 验证 B 相对于 A 的角速度矢量
    assert B.ang_vel_in(A).to_matrix(B) == Matrix([
        [cos(c2) * u1 + u3],
        [-sin(c2) * cos(q3) * u1],
        [sin(c2) * sin(q3) * u1]])

    # Test all symbols not time dependent
    # 创建新的参考系 A 和 B
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    # B 相对于 A 使用指定参数的空间固定方向余弦矩阵
    B.orient_space_fixed(A, (c1, c2, c3), 123)
    # 验证 B 相对于 A 的角速度矢量
    assert B.ang_vel_in(A) == Vector(0)


def test_orient_body_simple_ang_vel():
    """This test ensures that the simplest form of that linear system solution
    is returned, thus the == for the expression comparison."""

    # 定义动力学符号
    psi, theta, phi = dynamicsymbols('psi, theta, varphi')
    t = dynamicsymbols._t
    # 创建参考系 A 和 B
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    # B 相对于 A 使用 ZXZ 顺序的本体固定方向余弦矩阵
    B.orient_body_fixed(A, (psi, theta, phi), 'ZXZ')
    # 获取 A 到 B 的角速度矢量
    A_w_B = B.ang_vel_in(A)
    # 验证角速度矢量的结构
    assert A_w_B.args[0][1] == B
    assert A_w_B.args[0][0][0] == (sin(theta)*sin(phi)*psi.diff(t) +
                                   cos(phi)*theta.diff(t))
    assert A_w_B.args[0][0][1] == (sin(theta)*cos(phi)*psi.diff(t) -
                                   sin(phi)*theta.diff(t))
    assert A_w_B.args[0][0][2] == cos(theta)*psi.diff(t) + phi.diff(t)


def test_orient_space():
    # 创建新的参考系 A 和 B
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    # B 相对于 A 使用 '123' 顺序的空间固定方向余弦矩阵
    B.orient_space_fixed(A, (0,0,0), '123')
    # 验证 A 到 B 的方向余弦矩阵
    assert B.dcm(A) == Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_orient_quaternion():
    # 创建新的参考系 A 和 B
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    # 使用给定四元数向量设置 B 相对于 A 的方向余弦矩阵
    B.orient_quaternion(A, (0,0,0,0))
    # 验证 A 到 B 的方向余弦矩阵
    assert B.dcm(A) == Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def test_looped_frame_warning():
    # 创建新的参考系 A 和 B
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    # 创建一个新的参考系对象C
    C = ReferenceFrame('C')
    
    # 定义符号a, b, c，并将参考系B相对于参考系A绕A.x轴旋转a弧度
    a, b, c = symbols('a b c')
    B.orient_axis(A, A.x, a)
    
    # 将参考系C相对于参考系B绕B.x轴旋转b弧度
    C.orient_axis(B, B.x, b)
    
    # 使用警告模块捕获警告消息
    with warnings.catch_warnings(record=True) as w:
        # 设置警告过滤器，以便捕获所有警告
        warnings.simplefilter("always")
        
        # 将参考系A相对于参考系C绕C.x轴旋转c弧度，此处可能导致循环定义的警告
        A.orient_axis(C, C.x, c)
        
        # 断言最后一个捕获的警告是用户警告类的子类
        assert issubclass(w[-1].category, UserWarning)
        
        # 断言警告消息包含特定的警告内容，提示可能导致计算错误的循环定义问题
        assert 'Loops are defined among the orientation of frames. ' + \
               'This is likely not desired and may cause errors in your calculations.' in str(w[-1].message)
# 定义一个测试函数，用于测试 ReferenceFrame 类的字典操作
def test_frame_dict():
    # 创建三个参考框架对象 A、B、C
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')

    # 定义符号变量 a, b, c
    a, b, c = symbols('a b c')

    # 将参考框架 B 相对于参考框架 A 的方向设定为绕 A.x 轴旋转角度 a
    B.orient_axis(A, A.x, a)
    # 断言 A 的方向余弦矩阵字典中记录了 B 相对于 A 的方向关系
    assert A._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(a), -sin(a)],[0, sin(a), cos(a)]])}
    # 断言 B 的方向余弦矩阵字典中记录了 A 相对于 B 的方向关系
    assert B._dcm_dict == {A: Matrix([[1, 0, 0],[0, cos(a), sin(a)],[0, -sin(a), cos(a)]])}
    # 断言 C 的方向余弦矩阵字典为空
    assert C._dcm_dict == {}

    # 将参考框架 B 相对于参考框架 C 的方向设定为绕 C.x 轴旋转角度 b
    B.orient_axis(C, C.x, b)
    # 断言 A 的方向余弦矩阵字典未被修改
    assert A._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(a), -sin(a)],[0, sin(a), cos(a)]])}
    # 断言 B 的方向余弦矩阵字典中记录了 A 和 C 相对于 B 的方向关系
    assert B._dcm_dict == {A: Matrix([[1, 0, 0],[0, cos(a), sin(a)],[0, -sin(a), cos(a)]]), \
        C: Matrix([[1, 0, 0],[0, cos(b), sin(b)],[0, -sin(b), cos(b)]])}
    # 断言 C 的方向余弦矩阵字典中记录了 B 相对于 C 的方向关系
    assert C._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(b), -sin(b)],[0, sin(b), cos(b)]])}

    # 将参考框架 A 相对于参考框架 B 的方向设定为绕 B.x 轴旋转角度 c
    A.orient_axis(B, B.x, c)
    # 断言 B 的方向余弦矩阵字典中记录了 C 相对于 B 的方向关系和 A 相对于 B 的方向关系
    assert B._dcm_dict == {C: Matrix([[1, 0, 0],[0, cos(b), sin(b)],[0, -sin(b), cos(b)]]),\
        A: Matrix([[1, 0, 0],[0, cos(c), -sin(c)],[0, sin(c), cos(c)]])}
    # 断言 A 的方向余弦矩阵字典中记录了 B 相对于 A 的方向关系
    assert A._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(c), sin(c)],[0, -sin(c), cos(c)]])}
    # 断言 C 的方向余弦矩阵字典中记录了 B 相对于 C 的方向关系
    assert C._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(b), -sin(b)],[0, sin(b), cos(b)]])}

# 定义一个测试函数，用于测试 ReferenceFrame 类的缓存字典操作
def test_dcm_cache_dict():
    # 创建四个参考框架对象 A、B、C、D
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')
    D = ReferenceFrame('D')

    # 定义符号变量 a, b, c
    a, b, c = symbols('a b c')

    # 将参考框架 B 相对于参考框架 A 的方向设定为绕 A.x 轴旋转角度 a
    B.orient_axis(A, A.x, a)
    # 将参考框架 C 相对于参考框架 B 的方向设定为绕 B.x 轴旋转角度 b
    C.orient_axis(B, B.x, b)
    # 将参考框架 D 相对于参考框架 C 的方向设定为绕 C.x 轴旋转角度 c
    D.orient_axis(C, C.x, c)

    # 断言 D 的方向余弦矩阵字典中记录了 C 相对于 D 的方向关系
    assert D._dcm_dict == {C: Matrix([[1, 0, 0],[0, cos(c), sin(c)],[0, -sin(c), cos(c)]])}
    # 断言 C 的方向余弦矩阵字典中记录了 B 相对于 C 的方向关系和 D 相对于 C 的方向关系
    assert C._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(b), sin(b)],[0, -sin(b), cos(b)]]), \
        D: Matrix([[1, 0, 0],[0, cos(c), -sin(c)],[0, sin(c), cos(c)]])}
    # 断言 B 的方向余弦矩阵字典中记录了 A 相对于 B 的方向关系和 C 相对于 B 的方向关系
    assert B._dcm_dict == {A: Matrix([[1, 0, 0],[0, cos(a), sin(a)],[0, -sin(a), cos(a)]]), \
        C: Matrix([[1, 0, 0],[0, cos(b), -sin(b)],[0, sin(b), cos(b)]])}
    # 断言 A 的方向余弦矩阵字典中记录了 B 相对于 A 的方向关系
    assert A._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(a), -sin(a)],[0, sin(a), cos(a)]])}

    # 断言 D 的方向余弦矩阵字典等于 D 的方向余弦矩阵缓存
    assert D._dcm_dict == D._dcm_cache

    # 对参考框架 A 调用 dcm 方法，检查计算得到的方向余弦矩阵关系被存储在 _dcm_cache 中而非 _dcm_dict
    D.dcm(A)
    assert list(A._dcm_cache.keys()) == [A, B, D]
    assert list(D._dcm_cache.keys()) == [C, A]
    assert list(A._dcm_dict.keys()) == [B]
    assert list(D._dcm_dict.keys()) == [C]
    assert A._dcm_dict != A._dcm_cache

    # 将参考框架 A 相对于参考框架 B 的方向设定为绕 B.x 轴旋转角度 b
    A.orient_axis(B, B.x, b)
    # 断言 A 的方向余弦矩阵字典中记录了 B 相对于 A 的方向关系
    assert A._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(b), sin(b)],[0, -sin(b), cos(b)]])}
    # 断言 A 的方向余弦矩阵字典与 A 的方向余弦矩阵缓存相等
    assert A._dcm_dict == A._dcm_cache
    # 断言 B 的方向余弦矩阵字典中记录了 C 相对于 B 的方向关系和 A
    # 创建一个参照系对象 'F'，包含指定的索引 ['1', '2', '3']
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    
    # 使用向量外积操作计算向量 N.x 和 N.y 的外积，并断言结果等于向量 N.xy
    assert N.xy == Vector.outer(N.x, N.y)
    
    # 使用向量外积操作计算向量 F.x 和 F.y 的外积，并断言结果等于向量 F.xy
    assert F.xy == Vector.outer(F.x, F.y)
# 定义测试函数，用于测试ReferenceFrame类的xz分量的计算
def test_xz_dyad():
    # 创建惯性参考系N
    N = ReferenceFrame('N')
    # 创建局部参考系F，并指定其索引为['1', '2', '3']
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    # 断言惯性参考系N的xz为N.x和N.z的外积（dyadic product）
    assert N.xz == Vector.outer(N.x, N.z)
    # 断言局部参考系F的xz为F.x和F.z的外积（dyadic product）

def test_yx_dyad():
    # 创建惯性参考系N
    N = ReferenceFrame('N')
    # 创建局部参考系F，并指定其索引为['1', '2', '3']
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    # 断言惯性参考系N的yx为N.y和N.x的外积（dyadic product）
    assert N.yx == Vector.outer(N.y, N.x)
    # 断言局部参考系F的yx为F.y和F.x的外积（dyadic product）

def test_yy_dyad():
    # 创建惯性参考系N
    N = ReferenceFrame('N')
    # 创建局部参考系F，并指定其索引为['1', '2', '3']
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    # 断言惯性参考系N的yy为N.y和N.y的外积（dyadic product）
    assert N.yy == Vector.outer(N.y, N.y)
    # 断言局部参考系F的yy为F.y和F.y的外积（dyadic product）

def test_yz_dyad():
    # 创建惯性参考系N
    N = ReferenceFrame('N')
    # 创建局部参考系F，并指定其索引为['1', '2', '3']
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    # 断言惯性参考系N的yz为N.y和N.z的外积（dyadic product）
    assert N.yz == Vector.outer(N.y, N.z)
    # 断言局部参考系F的yz为F.y和F.z的外积（dyadic product）

def test_zx_dyad():
    # 创建惯性参考系N
    N = ReferenceFrame('N')
    # 创建局部参考系F，并指定其索引为['1', '2', '3']
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    # 断言惯性参考系N的zx为N.z和N.x的外积（dyadic product）
    assert N.zx == Vector.outer(N.z, N.x)
    # 断言局部参考系F的zx为F.z和F.x的外积（dyadic product）

def test_zy_dyad():
    # 创建惯性参考系N
    N = ReferenceFrame('N')
    # 创建局部参考系F，并指定其索引为['1', '2', '3']
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    # 断言惯性参考系N的zy为N.z和N.y的外积（dyadic product）
    assert N.zy == Vector.outer(N.z, N.y)
    # 断言局部参考系F的zy为F.z和F.y的外积（dyadic product）

def test_zz_dyad():
    # 创建惯性参考系N
    N = ReferenceFrame('N')
    # 创建局部参考系F，并指定其索引为['1', '2', '3']
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    # 断言惯性参考系N的zz为N.z和N.z的外积（dyadic product）
    assert N.zz == Vector.outer(N.z, N.z)
    # 断言局部参考系F的zz为F.z和F.z的外积（dyadic product）

def test_unit_dyadic():
    # 创建惯性参考系N
    N = ReferenceFrame('N')
    # 创建局部参考系F，并指定其索引为['1', '2', '3']
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    # 断言惯性参考系N的u为N.xx + N.yy + N.zz
    assert N.u == N.xx + N.yy + N.zz
    # 断言局部参考系F的u为F.xx + F.yy + F.zz

def test_pickle_frame():
    # 创建惯性参考系N
    N = ReferenceFrame('N')
    # 创建参考系A
    A = ReferenceFrame('A')
    # 将参考系A相对于N绕N.x轴旋转1弧度
    A.orient_axis(N, N.x, 1)
    # 获取A相对于N的方向余弦矩阵
    A_C_N = A.dcm(N)
    # 序列化和反序列化惯性参考系N，获得新的参考系N1
    N1 = pickle.loads(pickle.dumps(N))
    # 获取N1中的第一个参考系A1，并获取其相对于N1的方向余弦矩阵
    A1 = tuple(N1._dcm_dict.keys())[0]
    # 断言A1相对于N1的方向余弦矩阵与A相对于N的方向余弦矩阵A_C_N相等
    assert A1.dcm(N1) == A_C_N
```