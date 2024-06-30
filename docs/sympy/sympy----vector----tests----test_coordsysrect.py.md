# `D:\src\scipysrc\sympy\sympy\vector\tests\test_coordsysrect.py`

```
# 导入需要的模块和函数，从 sympy.testing.pytest 模块导入 raises 函数
# 从 sympy.vector.coordsysrect 模块导入 CoordSys3D 类
# 从 sympy.vector.scalar 模块导入 BaseScalar 类
# 从 sympy.core.function 模块导入 expand 函数
# 从 sympy.core.numbers 模块导入 pi 常数
# 从 sympy.core.symbol 模块导入 symbols 符号函数
# 从 sympy.functions.elementary.hyperbolic 模块导入 cosh 和 sinh 双曲函数
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt 开方函数
# 从 sympy.functions.elementary.trigonometric 模块导入 acos、atan2、cos 和 sin 三角函数
# 从 sympy.matrices.dense 模块导入 zeros 零矩阵函数
# 从 sympy.matrices.immutable 模块导入 Matrix 类作为 ImmutableDenseMatrix 的别名
# 从 sympy.simplify.simplify 模块导入 simplify 函数
# 从 sympy.vector.functions 模块导入 express 函数
# 从 sympy.vector.point 模块导入 Point 类
# 从 sympy.vector.vector 模块导入 Vector 类
# 从 sympy.vector.orienters 模块导入 AxisOrienter、BodyOrienter、SpaceOrienter 和 QuaternionOrienter 类

# 定义符号变量 x, y, z, a, b, c, q, q1, q2, q3, q4
x, y, z = symbols('x y z')
a, b, c, q = symbols('a b c q')
q1, q2, q3, q4 = symbols('q1 q2 q3 q4')

# 定义测试函数 test_func_args，测试 CoordSys3D 类中的各种方法
def test_func_args():
    # 创建一个名为 A 的三维坐标系对象
    A = CoordSys3D('A')
    # 断言坐标系 A 的 x 分量的 func 方法应用于其参数应该等于 A.x 本身
    assert A.x.func(*A.x.args) == A.x
    # 定义一个表达式 expr
    expr = 3*A.x + 4*A.y
    # 断言表达式 expr 的 func 方法应用于其参数应该等于 expr 本身
    assert expr.func(*expr.args) == expr
    # 断言坐标系 A 的单位向量 i 的 func 方法应用于其参数应该等于 A.i 本身
    assert A.i.func(*A.i.args) == A.i
    # 定义一个向量 v
    v = A.x*A.i + A.y*A.j + A.z*A.k
    # 断言向量 v 的 func 方法应用于其参数应该等于 v 本身
    assert v.func(*v.args) == v
    # 断言坐标系 A 的原点 A.origin 的 func 方法应用于其参数应该等于 A.origin 本身
    assert A.origin.func(*A.origin.args) == A.origin

# 定义测试函数 test_coordsys3d_equivalence，测试 CoordSys3D 对象的等价性
def test_coordsys3d_equivalence():
    # 创建两个名为 A 的三维坐标系对象 A 和 A1
    A = CoordSys3D('A')
    A1 = CoordSys3D('A')
    # 断言 A1 应该等于 A
    assert A1 == A
    # 创建另一个名为 B 的三维坐标系对象
    B = CoordSys3D('B')
    # 断言 A 应该不等于 B
    assert A != B

# 定义测试函数 test_orienters，测试不同类型的定向器对象
def test_orienters():
    # 创建一个名为 A 的三维坐标系对象
    A = CoordSys3D('A')
    # 创建轴定向器对象 axis_orienter
    axis_orienter = AxisOrienter(a, A.k)
    # 创建身体定向器对象 body_orienter
    body_orienter = BodyOrienter(a, b, c, '123')
    # 创建空间定向器对象 space_orienter
    space_orienter = SpaceOrienter(a, b, c, '123')
    # 创建四元数定向器对象 q_orienter
    q_orienter = QuaternionOrienter(q1, q2, q3, q4)
    # 断言轴定向器对象的旋转矩阵应该等于特定的 3x3 矩阵
    assert axis_orienter.rotation_matrix(A) == Matrix([
        [ cos(a), sin(a), 0],
        [-sin(a), cos(a), 0],
        [      0,      0, 1]])
    # 断言身体定向器对象的旋转矩阵应该等于特定的 3x3 矩阵
    assert body_orienter.rotation_matrix() == Matrix([
        [ cos(b)*cos(c),  sin(a)*sin(b)*cos(c) + sin(c)*cos(a),
          sin(a)*sin(c) - sin(b)*cos(a)*cos(c)],
        [-sin(c)*cos(b), -sin(a)*sin(b)*sin(c) + cos(a)*cos(c),
         sin(a)*cos(c) + sin(b)*sin(c)*cos(a)],
        [        sin(b),                        -sin(a)*cos(b),
                 cos(a)*cos(b)]])
    # 断言空间定向器对象的旋转矩阵应该等于特定的 3x3 矩阵
    assert space_orienter.rotation_matrix() == Matrix([
        [cos(b)*cos(c), sin(c)*cos(b),       -sin(b)],
        [sin(a)*sin(b)*cos(c) - sin(c)*cos(a),
         sin(a)*sin(b)*sin(c) + cos(a)*cos(c), sin(a)*cos(b)],
        [sin(a)*sin(c) + sin(b)*cos(a)*cos(c), -sin(a)*cos(c) +
         sin(b)*sin(c)*cos(a), cos(a)*cos(b)]])
    # 断言四元数定向器对象的旋转矩阵应该等于特定的 3x3 矩阵
    assert q_orienter.rotation_matrix() == Matrix([
        [q1**2 + q2**2 - q3**2 - q4**2, 2*q1*q4 + 2*q2*q3,
         -2*q1*q3 + 2*q2*q4],
        [-2*q1*q4 + 2*q2*q3, q1**2 - q2**2 + q3**2 - q4**2,
         2*q1*q2 + 2*q3*q4],
        [2*q1*q3 + 2*q2*q4,
         -2*q1*q2 + 2*q3*q4, q1**2 - q2**2 - q3**2 + q4**2]])

# 定义测试函数 test_coordinate_vars，测试坐标变量的功能，特别是在坐标系重定向时
def test_coordinate_vars():
    """
    Tests the coordinate variables functionality with respect to
    reorientation of coordinate systems.
    """
    # 创建名为 A 的三维坐标系对象
    A = CoordSys3D('A')
    # 断言，验证 BaseScalar 类的实例化是否正确，并设置 A.x 的属性
    assert BaseScalar(0, A, 'A_x', r'\mathbf{{x}_{A}}') == A.x
    # 断言，验证 BaseScalar 类的实例化是否正确，并设置 A.y 的属性
    assert BaseScalar(1, A, 'A_y', r'\mathbf{{y}_{A}}') == A.y
    # 断言，验证 BaseScalar 类的实例化是否正确，并设置 A.z 的属性
    assert BaseScalar(2, A, 'A_z', r'\mathbf{{z}_{A}}') == A.z
    # 断言，验证 A.x 和 A.y 的哈希值是否相同
    assert BaseScalar(0, A, 'A_x', r'\mathbf{{x}_{A}}').__hash__() == A.x.__hash__()
    # 断言，验证 A.x、A.y、A.z 是否都是 BaseScalar 的实例
    assert isinstance(A.x, BaseScalar) and \
           isinstance(A.y, BaseScalar) and \
           isinstance(A.z, BaseScalar)
    # 断言，验证乘积的交换律 A.x*A.y == A.y*A.x
    assert A.x*A.y == A.y*A.x
    # 断言，验证 A 对象的 scalar_map 方法返回的映射是否正确
    assert A.scalar_map(A) == {A.x: A.x, A.y: A.y, A.z: A.z}
    # 断言，验证 A.x 的系统属性是否为 A
    assert A.x.system == A
    # 断言，验证 A.x 对 A.x 求导的结果是否为 1
    assert A.x.diff(A.x) == 1
    # 使用 A 对象的 orient_new_axis 方法创建名为 B 的新坐标系对象
    B = A.orient_new_axis('B', q, A.k)
    # 断言，验证 B 对象的 scalar_map 方法返回的映射是否正确
    assert B.scalar_map(A) == {B.z: A.z, B.y: -A.x*sin(q) + A.y*cos(q),
                                 B.x: A.x*cos(q) + A.y*sin(q)}
    # 断言，验证 A 对象的 scalar_map 方法返回的映射是否正确
    assert A.scalar_map(B) == {A.x: B.x*cos(q) - B.y*sin(q),
                                 A.y: B.x*sin(q) + B.y*cos(q), A.z: B.z}
    # 断言，验证 express 函数对 B.x 的 A 表示是否正确
    assert express(B.x, A, variables=True) == A.x*cos(q) + A.y*sin(q)
    # 断言，验证 express 函数对 B.y 的 A 表示是否正确
    assert express(B.y, A, variables=True) == -A.x*sin(q) + A.y*cos(q)
    # 断言，验证 express 函数对 B.z 的 A 表示是否正确
    assert express(B.z, A, variables=True) == A.z
    # 断言，验证 expand 和 express 函数对 B.x*B.y*B.z 的 A 表示是否正确展开
    assert expand(express(B.x*B.y*B.z, A, variables=True)) == \
           expand(A.z*(-A.x*sin(q) + A.y*cos(q))*(A.x*cos(q) + A.y*sin(q)))
    # 断言，验证 simplify 和 express 函数对 B.x*B.i + B.y*B.j + B.z*B.k 的 A 表示是否正确简化
    assert simplify(express(B.x*B.i + B.y*B.j + B.z*B.k, A, \
                            variables=True)) == \
           A.x*A.i + A.y*A.j + A.z*A.k
    # 断言，验证 express 函数对 A.x*A.i + A.y*A.j + A.z*A.k 的 B 表示是否正确
    assert express(A.x*A.i + A.y*A.j + A.z*A.k, B) == \
           (A.x*cos(q) + A.y*sin(q))*B.i + \
           (-A.x*sin(q) + A.y*cos(q))*B.j + A.z*B.k
    # 断言，验证 simplify 和 express 函数对 A.x*A.i + A.y*A.j + A.z*A.k 的 B 表示是否正确简化
    assert simplify(express(A.x*A.i + A.y*A.j + A.z*A.k, B, \
                            variables=True)) == \
           B.x*B.i + B.y*B.j + B.z*B.k
    # 使用 B 对象的 orient_new_axis 方法创建名为 N 的新坐标系对象
    N = B.orient_new_axis('N', -q, B.k)
    # 断言，验证 N 对象的 scalar_map 方法返回的映射是否正确
    assert N.scalar_map(A) == \
           {N.x: A.x, N.z: A.z, N.y: A.y}
    # 使用 A 对象的 orient_new_axis 方法创建名为 C 的新坐标系对象
    C = A.orient_new_axis('C', q, A.i + A.j + A.k)
    # 获取 A 对象到 C 对象的标量映射
    mapping = A.scalar_map(C)
    # 断言，验证 mapping 中 A.x 的表达式是否正确
    assert mapping[A.x].equals(C.x*(2*cos(q) + 1)/3 +
                            C.y*(-2*sin(q + pi/6) + 1)/3 +
                            C.z*(-2*cos(q + pi/3) + 1)/3)
    # 断言，验证 mapping 中 A.y 的表达式是否正确
    assert mapping[A.y].equals(C.x*(-2*cos(q + pi/3) + 1)/3 +
                            C.y*(2*cos(q) + 1)/3 +
                            C.z*(-2*sin(q + pi/6) + 1)/3)
    # 断言，验证 mapping 中 A.z 的表达式是否正确
    assert mapping[A.z].equals(C.x*(-2*sin(q + pi/6) + 1)/3 +
                            C.y*(-2*cos(q + pi/3) + 1)/3 +
                            C.z*(2*cos(q) + 1)/3)
    # 使用 A 对象的 locate_new 方法创建名为 D 的新坐标系对象
    D = A.locate_new('D', a*A.i + b*A.j + c*A.k)
    # 断言，验证 D 对象的 scalar_map 方法返回的映射是否正确
    assert D.scalar_map(A) == {D.z: A.z - c, D.x: A.x - a, D.y: A.y - b}
    # 使用 A 对象的 orient_new_axis 方法创建名为 E 的新坐标系对象
    E = A.orient_new_axis('E', a, A.k, a*A.i + b*A.j + c*A.k)
    # 断言，验证 A 对象的 scalar_map 方法返回的映射是否正确
    assert A.scalar_map(E) == {A.z: E.z + c,
                               A.x: E.x*cos(a) - E.y*sin(a) + a,
                               A.y: E.x*sin(a) + E.y*cos(a) + b}
    # 断言：计算 E 对象到 A 对象的标量映射结果是否等于以下字典
    assert E.scalar_map(A) == {
        E.x: (A.x - a) * cos(a) + (A.y - b) * sin(a),  # 计算 E.x 映射到 A 对象的表达式
        E.y: (-A.x + a) * sin(a) + (A.y - b) * cos(a),  # 计算 E.y 映射到 A 对象的表达式
        E.z: A.z - c  # 计算 E.z 映射到 A 对象的表达式
    }
    
    # 创建一个名为 F 的新点，其位置为 Vector.zero
    F = A.locate_new('F', Vector.zero)
    
    # 断言：计算 A 对象到 F 对象的标量映射结果是否等于以下字典
    assert A.scalar_map(F) == {
        A.z: F.z,  # 计算 A.z 映射到 F 对象的表达式
        A.x: F.x,  # 计算 A.x 映射到 F 对象的表达式
        A.y: F.y   # 计算 A.y 映射到 F 对象的表达式
    }
# 定义一个函数用于测试旋转矩阵的计算
def test_rotation_matrix():
    # 创建一个三维坐标系对象 N
    N = CoordSys3D('N')
    # 在 N 坐标系下，绕 k 轴使用 q1 角度旋转创建新坐标系 A
    A = N.orient_new_axis('A', q1, N.k)
    # 在 A 坐标系下，绕自身 i 轴使用 q2 角度旋转创建新坐标系 B
    B = A.orient_new_axis('B', q2, A.i)
    # 在 B 坐标系下，绕自身 j 轴使用 q3 角度旋转创建新坐标系 C
    C = B.orient_new_axis('C', q3, B.j)
    # 在 N 坐标系下，绕 j 轴使用 q4 角度旋转创建新坐标系 D
    D = N.orient_new_axis('D', q4, N.j)
    # 使用欧拉角 q1, q2, q3 创建一个新的空间定向坐标系 E
    E = N.orient_new_space('E', q1, q2, q3, '123')
    # 使用四元数 q1, q2, q3, q4 创建一个新的四元数定向坐标系 F
    F = N.orient_new_quaternion('F', q1, q2, q3, q4)
    # 使用三个参数 q1, q2, q3 和轴顺序 '123' 创建一个新的本体定向坐标系 G
    
    # 断言 C 坐标系的旋转矩阵与给定的矩阵相等
    assert N.rotation_matrix(C) == Matrix([
        [- sin(q1) * sin(q2) * sin(q3) + cos(q1) * cos(q3), - sin(q1) *
        cos(q2), sin(q1) * sin(q2) * cos(q3) + sin(q3) * cos(q1)], \
        [sin(q1) * cos(q3) + sin(q2) * sin(q3) * cos(q1), \
         cos(q1) * cos(q2), sin(q1) * sin(q3) - sin(q2) * cos(q1) * \
         cos(q3)], [- sin(q3) * cos(q2), sin(q2), cos(q2) * cos(q3)]])
    
    # 计算 D 坐标系相对于 C 坐标系的旋转矩阵并断言其结果为零矩阵
    test_mat = D.rotation_matrix(C) - Matrix(
        [[cos(q1) * cos(q3) * cos(q4) - sin(q3) * (- sin(q4) * cos(q2) +
        sin(q1) * sin(q2) * cos(q4)), - sin(q2) * sin(q4) - sin(q1) *
            cos(q2) * cos(q4), sin(q3) * cos(q1) * cos(q4) + cos(q3) * \
          (- sin(q4) * cos(q2) + sin(q1) * sin(q2) * cos(q4))], \
         [sin(q1) * cos(q3) + sin(q2) * sin(q3) * cos(q1), cos(q1) * \
          cos(q2), sin(q1) * sin(q3) - sin(q2) * cos(q1) * cos(q3)], \
         [sin(q4) * cos(q1) * cos(q3) - sin(q3) * (cos(q2) * cos(q4) + \
                                                   sin(q1) * sin(q2) * \
                                                   sin(q4)), sin(q2) *
                cos(q4) - sin(q1) * sin(q4) * cos(q2), sin(q3) * \
          sin(q4) * cos(q1) + cos(q3) * (cos(q2) * cos(q4) + \
                                         sin(q1) * sin(q2) * sin(q4))]])
    assert test_mat.expand() == zeros(3, 3)
    
    # 断言 E 坐标系相对于 N 坐标系的旋转矩阵与给定的矩阵相等
    assert E.rotation_matrix(N) == Matrix(
        [[cos(q2)*cos(q3), sin(q3)*cos(q2), -sin(q2)],
        [sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1), \
         sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3), sin(q1)*cos(q2)], \
         [sin(q1)*sin(q3) + sin(q2)*cos(q1)*cos(q3), - \
          sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1), cos(q1)*cos(q2)]])
    
    # 断言 F 坐标系相对于 N 坐标系的旋转矩阵与给定的矩阵相等
    assert F.rotation_matrix(N) == Matrix([[
        q1**2 + q2**2 - q3**2 - q4**2,
        2*q1*q4 + 2*q2*q3, -2*q1*q3 + 2*q2*q4],[ -2*q1*q4 + 2*q2*q3,
            q1**2 - q2**2 + q3**2 - q4**2, 2*q1*q2 + 2*q3*q4],
                                           [2*q1*q3 + 2*q2*q4,
                                            -2*q1*q2 + 2*q3*q4,
                                q1**2 - q2**2 - q3**2 + q4**2]])
    
    # 断言 G 坐标系相对于 N 坐标系的旋转矩阵与给定的矩阵相等
    assert G.rotation_matrix(N) == Matrix([[
        cos(q2)*cos(q3),  sin(q1)*sin(q2)*cos(q3) + sin(q3)*cos(q1),
        sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3)], [
            -sin(q3)*cos(q2), -sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3),
            sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1)],[
                sin(q2), -sin(q1)*cos(q2), cos(q1)*cos(q2)]])


def test_vector_with_orientation():
    """
    测试坐标系定向对基本向量操作的影响。
    """
    # 创建一个三维坐标系对象 N
    N = CoordSys3D('N')
    A = N.orient_new_axis('A', q1, N.k)
    # 定义一个新的方向 'A'，沿着轴 N.k，角度为 q1

    B = A.orient_new_axis('B', q2, A.i)
    # 定义一个相对于方向 'A' 的新方向 'B'，沿着轴 A.i，角度为 q2

    C = B.orient_new_axis('C', q3, B.j)
    # 定义一个相对于方向 'B' 的新方向 'C'，沿着轴 B.j，角度为 q3

    # Test to_matrix
    v1 = a*N.i + b*N.j + c*N.k
    # 定义一个向量 v1，以 N.i、N.j、N.k 为基向量，系数分别为 a、b、c

    assert v1.to_matrix(A) == Matrix([[ a*cos(q1) + b*sin(q1)],
                                      [-a*sin(q1) + b*cos(q1)],
                                      [                     c]])
    # 断言将向量 v1 转换为相对于方向 'A' 的矩阵，与预期的矩阵形式相等

    # Test dot
    assert N.i.dot(A.i) == cos(q1)
    # 断言 N.i 和 A.i 的点积为 cos(q1)
    assert N.i.dot(A.j) == -sin(q1)
    # 断言 N.i 和 A.j 的点积为 -sin(q1)
    assert N.i.dot(A.k) == 0
    # 断言 N.i 和 A.k 的点积为 0
    assert N.j.dot(A.i) == sin(q1)
    # 断言 N.j 和 A.i 的点积为 sin(q1)
    assert N.j.dot(A.j) == cos(q1)
    # 断言 N.j 和 A.j 的点积为 cos(q1)
    assert N.j.dot(A.k) == 0
    # 断言 N.j 和 A.k 的点积为 0
    assert N.k.dot(A.i) == 0
    # 断言 N.k 和 A.i 的点积为 0
    assert N.k.dot(A.j) == 0
    # 断言 N.k 和 A.j 的点积为 0
    assert N.k.dot(A.k) == 1
    # 断言 N.k 和 A.k 的点积为 1

    assert N.i.dot(A.i + A.j) == -sin(q1) + cos(q1) == \
           (A.i + A.j).dot(N.i)
    # 断言 N.i 和 (A.i + A.j) 的点积，与 (A.i + A.j) 和 N.i 的点积结果相等

    assert A.i.dot(C.i) == cos(q3)
    # 断言 A.i 和 C.i 的点积为 cos(q3)
    assert A.i.dot(C.j) == 0
    # 断言 A.i 和 C.j 的点积为 0
    assert A.i.dot(C.k) == sin(q3)
    # 断言 A.i 和 C.k 的点积为 sin(q3)
    assert A.j.dot(C.i) == sin(q2)*sin(q3)
    # 断言 A.j 和 C.i 的点积为 sin(q2)*sin(q3)
    assert A.j.dot(C.j) == cos(q2)
    # 断言 A.j 和 C.j 的点积为 cos(q2)
    assert A.j.dot(C.k) == -sin(q2)*cos(q3)
    # 断言 A.j 和 C.k 的点积为 -sin(q2)*cos(q3)
    assert A.k.dot(C.i) == -cos(q2)*sin(q3)
    # 断言 A.k 和 C.i 的点积为 -cos(q2)*sin(q3)
    assert A.k.dot(C.j) == sin(q2)
    # 断言 A.k 和 C.j 的点积为 sin(q2)
    assert A.k.dot(C.k) == cos(q2)*cos(q3)
    # 断言 A.k 和 C.k 的点积为 cos(q2)*cos(q3)

    # Test cross
    assert N.i.cross(A.i) == sin(q1)*A.k
    # 断言 N.i 和 A.i 的叉乘结果为 sin(q1)*A.k
    assert N.i.cross(A.j) == cos(q1)*A.k
    # 断言 N.i 和 A.j 的叉乘结果为 cos(q1)*A.k
    assert N.i.cross(A.k) == -sin(q1)*A.i - cos(q1)*A.j
    # 断言 N.i 和 A.k 的叉乘结果为 -sin(q1)*A.i - cos(q1)*A.j
    assert N.j.cross(A.i) == -cos(q1)*A.k
    # 断言 N.j 和 A.i 的叉乘结果为 -cos(q1)*A.k
    assert N.j.cross(A.j) == sin(q1)*A.k
    # 断言 N.j 和 A.j 的叉乘结果为 sin(q1)*A.k
    assert N.j.cross(A.k) == cos(q1)*A.i - sin(q1)*A.j
    # 断言 N.j 和 A.k 的叉乘结果为 cos(q1)*A.i - sin(q1)*A.j
    assert N.k.cross(A.i) == A.j
    # 断言 N.k 和 A.i 的叉乘结果为 A.j
    assert N.k.cross(A.j) == -A.i
    # 断言 N.k 和 A.j 的叉乘结果为 -A.i
    assert N.k.cross(A.k) == Vector.zero
    # 断言 N.k 和 A.k 的叉乘结果为 零向量

    assert N.i.cross(A.i) == sin(q1)*A.k
    # 再次断言 N.i 和 A.i 的叉乘结果为 sin(q1)*A.k
    assert N.i.cross(A.j) == cos(q1)*A.k
    # 再次断言 N.i 和 A.j 的叉乘结果为 cos(q1)*A.k
    assert N.i.cross(A.i + A.j) == sin(q1)*A.k + cos(q1)*A.k
    # 断言 N.i 和 (A.i + A.j) 的叉乘结果为 sin(q1)*A.k + cos(q1)*A.k
    assert (A.i + A.j).cross(N.i) == (-sin(q1) - cos(q1))*N.k
    # 断言 (A.i + A.j) 和 N.i 的叉乘结果为 (-sin(q1) - cos(q1))*N.k

    assert A.i.cross(C.i) == sin(q3)*C.j
    # 断言 A.i 和 C.i 的叉乘结果为 sin(q3)*C.j
    assert A.i.cross(C.j) == -sin(q3)*C.i + cos(q3)*C.k
    # 断言 A.i 和 C.j 的叉乘结果为 -sin(q3)*C.i + cos(q3)*C.k
    assert A.i.cross(C.k) == -cos(q3)*C.j
    # 断言 A.i 和 C.k 的叉乘结果为 -cos(q3)*C.j
    assert C.i.cross(A.i) == (-sin(q3)*cos(q2))*A.j + \
           (-sin(q2)*sin(q3))*A.k
    # 断言 C.i 和 A.i 的叉乘结果为 (-sin(q3)*cos(q2))*A.j + (-sin(q2)*sin(q3))*A.k
    assert C.j.cross(A.i) == (sin(q2))*A.j + (-cos(q2))*A.k
    # 断言 C.j 和 A.i 的叉乘结果为 (sin(q2))*A.j + (-cos(q2))*A.k
    assert express(C.k.cross(A.i), C).trigsimp() == cos(q3)*C.j
    # 断言 C.k 和 A.i 的叉乘结果，经过 C 表达简化后，为 cos(q3)*C.j
def test_orient_new_methods():
    # 创建一个名为N的三维坐标系对象
    N = CoordSys3D('N')
    # 使用AxisOrienter初始化一个名为orienter1的轴向定位器
    orienter1 = AxisOrienter(q4, N.j)
    # 使用SpaceOrienter初始化一个名为orienter2的空间定位器
    orienter2 = SpaceOrienter(q1, q2, q3, '123')
    # 使用QuaternionOrienter初始化一个名为orienter3的四元数定位器
    orienter3 = QuaternionOrienter(q1, q2, q3, q4)
    # 使用BodyOrienter初始化一个名为orienter4的身体定位器
    orienter4 = BodyOrienter(q1, q2, q3, '123')
    # 使用orienter1作为参数创建一个名为D的新定位对象
    D = N.orient_new('D', (orienter1, ))
    # 使用orienter2作为参数创建一个名为E的新定位对象
    E = N.orient_new('E', (orienter2, ))
    # 使用orienter3作为参数创建一个名为F的新定位对象
    F = N.orient_new('F', (orienter3, ))
    # 使用orienter4作为参数创建一个名为G的新定位对象
    G = N.orient_new('G', (orienter4, ))
    # 断言检查D的结果与N.orient_new_axis('D', q4, N.j)相等
    assert D == N.orient_new_axis('D', q4, N.j)
    # 断言检查E的结果与N.orient_new_space('E', q1, q2, q3, '123')相等
    assert E == N.orient_new_space('E', q1, q2, q3, '123')
    # 断言检查F的结果与N.orient_new_quaternion('F', q1, q2, q3, q4)相等
    assert F == N.orient_new_quaternion('F', q1, q2, q3, q4)
    # 断言检查G的结果与N.orient_new_body('G', q1, q2, q3, '123')相等
    assert G == N.orient_new_body('G', q1, q2, q3, '123')


def test_locatenew_point():
    """
    Tests Point class, and locate_new method in CoordSys3D.
    """
    # 创建一个名为A的三维坐标系对象
    A = CoordSys3D('A')
    # 断言检查A.origin是否是Point类的实例
    assert isinstance(A.origin, Point)
    # 定义一个向量v，表示为a*A.i + b*A.j + c*A.k
    v = a*A.i + b*A.j + c*A.k
    # 使用向量v作为偏移向量，创建一个名为C的新点
    C = A.locate_new('C', v)
    # 断言检查C.origin.position_wrt(A)，C.position_wrt(A)，C.origin.position_wrt(A.origin)是否均等于v
    assert C.origin.position_wrt(A) == \
           C.position_wrt(A) == \
           C.origin.position_wrt(A.origin) == v
    # 断言检查A.origin.position_wrt(C)，A.position_wrt(C)，A.origin.position_wrt(C.origin)是否均等于-v
    assert A.origin.position_wrt(C) == \
           A.position_wrt(C) == \
           A.origin.position_wrt(C.origin) == -v
    # 断言检查A.origin.express_coordinates(C)是否等于(-a, -b, -c)
    assert A.origin.express_coordinates(C) == (-a, -b, -c)
    # 使用向量-v作为偏移向量，创建一个名为p的新点
    p = A.origin.locate_new('p', -v)
    # 断言检查p.express_coordinates(A)是否等于(-a, -b, -c)
    assert p.express_coordinates(A) == (-a, -b, -c)
    # 断言检查p.position_wrt(C.origin)，p.position_wrt(C)是否均等于-2 * v
    assert p.position_wrt(C.origin) == p.position_wrt(C) == \
           -2 * v
    # 使用向量2*v作为偏移向量，创建一个名为p1的新点
    p1 = p.locate_new('p1', 2*v)
    # 断言检查p1.position_wrt(C.origin)是否等于Vector.zero
    assert p1.position_wrt(C.origin) == Vector.zero
    # 断言检查p1.express_coordinates(C)是否等于(0, 0, 0)
    assert p1.express_coordinates(C) == (0, 0, 0)
    # 使用向量A.i作为偏移向量，创建一个名为p2的新点
    p2 = p.locate_new('p2', A.i)
    # 断言检查p1.position_wrt(p2)是否等于2*v - A.i
    assert p1.position_wrt(p2) == 2*v - A.i
    # 断言检查p2.express_coordinates(C)是否等于(-2*a + 1, -2*b, -2*c)


def test_create_new():
    # 创建一个名为a的三维坐标系对象
    a = CoordSys3D('a')
    # 使用'spherical'转换方式创建一个名为c的新坐标系对象
    c = a.create_new('c', transformation='spherical')
    # 断言检查c._parent是否等于a
    assert c._parent == a
    # 断言检查c.transformation_to_parent()是否等于对应的球坐标系转换公式
    assert c.transformation_to_parent() == \
           (c.r*sin(c.theta)*cos(c.phi), c.r*sin(c.theta)*sin(c.phi), c.r*cos(c.theta))
    # 断言检查c.transformation_from_parent()是否等于对应的球坐标系逆转换公式
    assert c.transformation_from_parent() == \
           (sqrt(a.x**2 + a.y**2 + a.z**2), acos(a.z/sqrt(a.x**2 + a.y**2 + a.z**2)), atan2(a.y, a.x))


def test_evalf():
    # 创建一个名为A的三维坐标系对象
    A = CoordSys3D('A')
    # 定义一个向量v
    v = 3*A.i + 4*A.j + a*A.k
    # 断言检查v.n()是否等于v.evalf()
    assert v.n() == v.evalf()
    # 断言检查v.evalf(subs={a:1})是否等于v.subs(a, 1).evalf()
    assert v.evalf(subs={a:1}) == v.subs(a, 1).evalf()


def test_lame_coefficients():
    # 创建一个名为a的球坐标系对象
    a = CoordSys3D('a', 'spherical')
    # 断言检查a.lame_coefficients()是否等于(1, a.r, sin(a.theta)*a.r)
    assert a.lame_coefficients() == (1, a.r, sin(a.theta)*a.r)
    # 创建一个名为a的笛卡尔坐标系对象
    a = CoordSys3D('a')
    # 断言检查a.lame_coefficients()是否等于(1, 1, 1)
    assert a.lame_coefficients() == (1, 1, 1)
    # 创建一个名为a的笛卡尔坐标系对象
    a = CoordSys3D('a', 'cartesian')
    # 断言检查a.lame_coefficients()是否等于(1, 1, 1)
    assert a.lame_coefficients() == (1, 1, 1)
    # 创建一个名为a的柱坐标系对象
    a = CoordSys3D('a', 'cylindrical')
    # 断言检查a.lame_coefficients()是否等于(1, a.r, 1)
    assert a.lame_coefficients() == (1, a.r, 1)


def test_transformation_equations():

    x, y, z = symbols('x y z')
    # 创建一个名为a的球坐标系对象，变量名为["r", "theta", "phi"]
    a = CoordSys3D('a', transformation='spherical',
                   variable_names=["r", "theta", "phi"])
    # 获取球坐标系的基
    # 断言，验证坐标系转换到父坐标系的变换公式是否正确
    assert a.transformation_to_parent() == (
        r*sin(theta)*cos(phi),
        r*sin(theta)*sin(phi),
        r*cos(theta)
    )
    # 断言，验证拉姆系数是否正确
    assert a.lame_coefficients() == (1, r, r*sin(theta))
    # 断言，验证从父坐标系到坐标系的变换函数是否正确
    assert a.transformation_from_parent_function()(x, y, z) == (
        sqrt(x ** 2 + y ** 2 + z ** 2),
        acos((z) / sqrt(x**2 + y**2 + z**2)),
        atan2(y, x)
    )
    # 创建一个三维坐标系对象，采用柱坐标系变换，变量名为["r", "theta", "z"]
    a = CoordSys3D('a', transformation='cylindrical',
                   variable_names=["r", "theta", "z"])
    # 获取基向量标量值
    r, theta, z = a.base_scalars()
    # 断言，验证坐标系转换到父坐标系的变换公式是否正确
    assert a.transformation_to_parent() == (
        r*cos(theta),
        r*sin(theta),
        z
    )
    # 断言，验证拉姆系数是否正确
    assert a.lame_coefficients() == (1, a.r, 1)
    # 断言，验证从父坐标系到坐标系的变换函数是否正确
    assert a.transformation_from_parent_function()(x, y, z) == (sqrt(x**2 + y**2),
                            atan2(y, x), z)

    # 创建一个三维坐标系对象，采用笛卡尔坐标系变换
    a = CoordSys3D('a', 'cartesian')
    # 断言，验证坐标系转换到父坐标系的变换公式是否正确
    assert a.transformation_to_parent() == (a.x, a.y, a.z)
    # 断言，验证拉姆系数是否正确
    assert a.lame_coefficients() == (1, 1, 1)
    # 断言，验证从父坐标系到坐标系的变换函数是否正确
    assert a.transformation_from_parent_function()(x, y, z) == (x, y, z)

    # 变量和表达式

    # 使用元组定义的笛卡尔坐标系
    x, y, z = symbols('x y z')
    a = CoordSys3D('a', ((x, y, z), (x, y, z)))
    a._calculate_inv_trans_equations()
    # 断言，验证坐标系转换到父坐标系的变换公式是否正确
    assert a.transformation_to_parent() == (a.x1, a.x2, a.x3)
    # 断言，验证拉姆系数是否正确
    assert a.lame_coefficients() == (1, 1, 1)
    # 断言，验证从父坐标系到坐标系的变换函数是否正确
    assert a.transformation_from_parent_function()(x, y, z) == (x, y, z)
    r, theta, z = symbols("r theta z")

    # 使用元组定义的柱坐标系
    a = CoordSys3D('a', [(r, theta, z), (r*cos(theta), r*sin(theta), z)],
                   variable_names=["r", "theta", "z"])
    r, theta, z = a.base_scalalars()
    # 断言，验证坐标系转换到父坐标系的变换公式是否正确
    assert a.transformation_to_parent() == (
        r*cos(theta), r*sin(theta), z
    )
    # 断言，验证拉姆系数是否正确
    assert a.lame_coefficients() == (
        sqrt(sin(theta)**2 + cos(theta)**2),
        sqrt(r**2*sin(theta)**2 + r**2*cos(theta)**2),
        1
    )  # ==> this should simplify to (1, r, 1), tests are too slow with `simplify`.

    # 使用lambda定义的笛卡尔坐标系
    a = CoordSys3D('a', lambda x, y, z: (x, y, z))
    # 断言，验证坐标系转换到父坐标系的变换公式是否正确
    assert a.transformation_to_parent() == (a.x1, a.x2, a.x3)
    # 断言，验证拉姆系数是否正确
    assert a.lame_coefficients() == (1, 1, 1)
    a._calculate_inv_trans_equations()
    # 断言，验证从父坐标系到坐标系的变换函数是否正确
    assert a.transformation_from_parent_function()(x, y, z) == (x, y, z)

    # 使用lambda定义的球坐标系
    a = CoordSys3D('a', lambda r, theta, phi: (r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)),
                   variable_names=["r", "theta", "phi"])
    r, theta, phi = a.base_scalars()
    # 断言，验证坐标系转换到父坐标系的变换公式是否正确
    assert a.transformation_to_parent() == (
        r*sin(theta)*cos(phi), r*sin(phi)*sin(theta), r*cos(theta)
    )
    # 断言，验证拉姆系数是否正确
    assert a.lame_coefficients() == (
        sqrt(sin(phi)**2*sin(theta)**2 + sin(theta)**2*cos(phi)**2 + cos(theta)**2),
        sqrt(r**2*sin(phi)**2*cos(theta)**2 + r**2*sin(theta)**2 + r**2*cos(phi)**2*cos(theta)**2),
        sqrt(r**2*sin(phi)**2*sin(theta)**2 + r**2*sin(theta)**2*cos(phi)**2)
    )
    )  # ==> this should simplify to (1, r, sin(theta)*r), `simplify` is too slow.



    # 在这个语句中，将一个包含三个元素的元组与注释关联。这个注释可能试图说明一个预期的简化结果，但未详细说明`simplify`方法为何效率太低。
    # 注意：此行代码末尾的注释可能是由于代码注释和代码之间的符号存在冲突。



    # 使用lambda函数定义三维坐标系a。这里的lambda函数定义了从三个坐标(r, theta, z)到笛卡尔坐标的转换。
    a = CoordSys3D('a', lambda r, theta, z:
        (r*cos(theta), r*sin(theta), z),
        variable_names=["r", "theta", "z"]
    )



    # 将三维坐标系a的基本标量(r, theta, z)分配给变量r, theta, z。
    r, theta, z = a.base_scalars()
    # 验证三维坐标系a到父坐标系的转换是否为(r*cos(theta), r*sin(theta), z)。
    assert a.transformation_to_parent() == (r*cos(theta), r*sin(theta), z)
    # 验证三维坐标系a的拉梅系数是否为(sqrt(sin(theta)**2 + cos(theta)**2), sqrt(r**2*sin(theta)**2 + r**2*cos(theta)**2), 1)。
    assert a.lame_coefficients() == (
        sqrt(sin(theta)**2 + cos(theta)**2),
        sqrt(r**2*sin(theta)**2 + r**2*cos(theta)**2),
        1
    )  # ==> this should simplify to (1, a.x, 1)



    # 使用lambda函数作为三维坐标系a的转换参数会引发TypeError异常，因为这个参数不应该是一个字典。这里的注释可能暗示了正确的参数类型。
    raises(TypeError, lambda: CoordSys3D('a', transformation={
        x: x*sin(y)*cos(z), y:x*sin(y)*sin(z), z:  x*cos(y)}))
def test_check_orthogonality():
    # 定义符号变量 x, y, z
    x, y, z = symbols('x y z')
    # 定义符号变量 u, v
    u,v = symbols('u, v')
    
    # 创建 CoordSys3D 对象 a，使用给定的变换矩阵初始化
    a = CoordSys3D('a', transformation=((x, y, z), (x*sin(y)*cos(z), x*sin(y)*sin(z), x*cos(y))))
    # 断言 CoordSys3D 对象 a 的变换矩阵是否正交
    assert a._check_orthogonality(a._transformation) is True
    
    # 重新创建 CoordSys3D 对象 a，使用另一个变换矩阵初始化
    a = CoordSys3D('a', transformation=((x, y, z), (x * cos(y), x * sin(y), z)))
    # 断言 CoordSys3D 对象 a 的变换矩阵是否正交
    assert a._check_orthogonality(a._transformation) is True
    
    # 重新创建 CoordSys3D 对象 a，使用另一个变换矩阵初始化
    a = CoordSys3D('a', transformation=((u, v, z), (cosh(u) * cos(v), sinh(u) * sin(v), z)))
    # 断言 CoordSys3D 对象 a 的变换矩阵是否正交
    assert a._check_orthogonality(a._transformation) is True
    
    # 断言创建 CoordSys3D 对象时，如果变换矩阵不正交，则会引发 ValueError 异常
    raises(ValueError, lambda: CoordSys3D('a', transformation=((x, y, z), (x, x, z))))
    # 断言创建 CoordSys3D 对象时，如果变换矩阵不正交，则会引发 ValueError 异常
    raises(ValueError, lambda: CoordSys3D('a', transformation=(
        (x, y, z), (x*sin(y/2)*cos(z), x*sin(y)*sin(z), x*cos(y)))))

def test_rotation_trans_equations():
    # 创建 CoordSys3D 对象 a
    a = CoordSys3D('a')
    # 导入符号变量
    from sympy.core.symbol import symbols
    # 定义符号变量 q0
    q0 = symbols('q0')
    
    # 断言 CoordSys3D 对象 a 的旋转变换方程是否与基础标量匹配
    assert a._rotation_trans_equations(a._parent_rotation_matrix, a.base_scalars()) == (a.x, a.y, a.z)
    # 断言 CoordSys3D 对象 a 的逆旋转矩阵的变换方程是否与基础标量匹配
    assert a._rotation_trans_equations(a._inverse_rotation_matrix(), a.base_scalars()) == (a.x, a.y, a.z)
    
    # 通过绕轴 k 新定向创建 CoordSys3D 对象 b
    b = a.orient_new_axis('b', 0, -a.k)
    # 断言 CoordSys3D 对象 b 的旋转变换方程是否与基础标量匹配
    assert b._rotation_trans_equations(b._parent_rotation_matrix, b.base_scalars()) == (b.x, b.y, b.z)
    # 断言 CoordSys3D 对象 b 的逆旋转矩阵的变换方程是否与基础标量匹配
    assert b._rotation_trans_equations(b._inverse_rotation_matrix(), b.base_scalars()) == (b.x, b.y, b.z)
    
    # 通过绕轴 k 新定向创建 CoordSys3D 对象 c，使用符号变量 q0
    c = a.orient_new_axis('c', q0, -a.k)
    # 断言 CoordSys3D 对象 c 的旋转变换方程是否与基础标量匹配
    assert c._rotation_trans_equations(c._parent_rotation_matrix, c.base_scalars()) == \
           (-sin(q0) * c.y + cos(q0) * c.x, sin(q0) * c.x + cos(q0) * c.y, c.z)
    # 断言 CoordSys3D 对象 c 的逆旋转矩阵的变换方程是否与基础标量匹配
    assert c._rotation_trans_equations(c._inverse_rotation_matrix(), c.base_scalars()) == \
           (sin(q0) * c.y + cos(q0) * c.x, -sin(q0) * c.x + cos(q0) * c.y, c.z)
```