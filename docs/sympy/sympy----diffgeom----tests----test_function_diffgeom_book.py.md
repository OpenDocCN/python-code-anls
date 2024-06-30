# `D:\src\scipysrc\sympy\sympy\diffgeom\tests\test_function_diffgeom_book.py`

```
# 导入必要的库和模块，从 sympy.diffgeom.rn 中导入 R2, R2_p, R2_r, R3_r
# 这些模块提供了对实数二维和三维空间的不同表示的支持
from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r
# 导入不同几何操作相关的模块，包括曲线积分序列、微分、外积
from sympy.diffgeom import intcurve_series, Differential, WedgeProduct
# 导入 sympy 的核心符号、函数和导数相关模块
from sympy.core import symbols, Function, Derivative
# 导入 sympy 的简化函数相关模块，用于简化三角函数等表达式
from sympy.simplify import trigsimp, simplify
# 导入 sympy 的数学函数，如平方根、反正切、正弦、余弦等
from sympy.functions import sqrt, atan2, sin, cos
# 导入 sympy 的矩阵操作模块
from sympy.matrices import Matrix

# 大部分功能已在 test_functional_diffgeom_ch* 测试中涵盖，
# 这些测试基于 Sussman 和 Wisdom 的论文中的示例。
# 如果这些测试未覆盖某些内容，则会在其他测试函数中添加额外的测试。

# 来自 Sussman 和 Wisdom 的 "Functional Differential Geometry"，
# 作为 2011 年的版本。


在上面的代码中，注释详细解释了每个导入的模块以及整体测试功能的覆盖情况和原始文献的来源。
    # 使用给定的极坐标点 (r0, theta0) 创建平面点对象 p_p
    p_p = R2_p.point([r0, theta0])

    # 计算二维场的矢量场 f_field，并确保以下断言成立
    f_field = b1(R2.x, R2.y)*R2.dx + b2(R2.x, R2.y)*R2.dy
    assert f_field.rcall(R2.e_x).rcall(p_r) == b1(x0, y0)
    assert f_field.rcall(R2.e_y).rcall(p_r) == b2(x0, y0)

    # 计算二维标量场 s_field_r，并创建其微分对象 df，确保以下断言成立
    s_field_r = f(R2.x, R2.y)
    df = Differential(s_field_r)
    assert df(R2.e_x).rcall(p_r).doit() == Derivative(f(x0, y0), x0)
    assert df(R2.e_y).rcall(p_r).doit() == Derivative(f(x0, y0), y0)

    # 计算极坐标场 s_field_p，并创建其微分对象 df，确保以下断言成立
    s_field_p = f(R2.r, R2.theta)
    df = Differential(s_field_p)
    assert trigsimp(df(R2.e_x).rcall(p_p).doit()) == (
        cos(theta0)*Derivative(f(r0, theta0), r0) -
        sin(theta0)*Derivative(f(r0, theta0), theta0)/r0)
    assert trigsimp(df(R2.e_y).rcall(p_p).doit()) == (
        sin(theta0)*Derivative(f(r0, theta0), r0) +
        cos(theta0)*Derivative(f(r0, theta0), theta0)/r0)

    # 确保以下断言成立，验证微分几何计算的性质
    assert R2.dx(R2.e_x).rcall(p_r) == 1
    assert R2.dx(R2.e_x) == 1
    assert R2.dx(R2.e_y).rcall(p_r) == 0
    assert R2.dx(R2.e_y) == 0

    # 创建环绕路径 circ，验证以下断言成立
    circ = -R2.y*R2.e_x + R2.x*R2.e_y
    assert R2.dx(circ).rcall(p_r).doit() == -y0
    assert R2.dy(circ).rcall(p_r) == x0
    assert R2.dr(circ).rcall(p_r) == 0
    assert simplify(R2.dtheta(circ).rcall(p_r)) == 1

    # 验证以下断言成立，确保场和矢量的关系正确
    assert (circ - R2.e_theta).rcall(s_field_r).rcall(p_r) == 0
# 定义测试函数 test_functional_diffgeom_ch6
def test_functional_diffgeom_ch6():
    # 声明符号变量 u0, u1, u2, v0, v1, v2, w0, w1, w2，均为实数
    u0, u1, u2, v0, v1, v2, w0, w1, w2 = symbols('u0:3, v0:3, w0:3', real=True)

    # 创建二维空间中的向量 u 和 v
    u = u0*R2.e_x + u1*R2.e_y
    v = v0*R2.e_x + v1*R2.e_y

    # 创建 R2 空间中的外积对象 wp，并进行断言验证
    wp = WedgeProduct(R2.dx, R2.dy)
    assert wp(u, v) == u0*v1 - u1*v0

    # 创建三维空间中的向量 u, v, w
    u = u0*R3_r.e_x + u1*R3_r.e_y + u2*R3_r.e_z
    v = v0*R3_r.e_x + v1*R3_r.e_y + v2*R3_r.e_z
    w = w0*R3_r.e_x + w1*R3_r.e_y + w2*R3_r.e_z

    # 创建 R3_r 空间中的外积对象 wp，并进行断言验证
    wp = WedgeProduct(R3_r.dx, R3_r.dy, R3_r.dz)
    assert wp(u, v, w) == Matrix(3, 3, [u0, u1, u2, v0, v1, v2, w0, w1, w2]).det()

    # 声明符号函数 a, b, c
    a, b, c = symbols('a, b, c', cls=Function)

    # 创建依赖 R3_r 空间坐标的函数 a_f, b_f, c_f
    a_f = a(R3_r.x, R3_r.y, R3_r.z)
    b_f = b(R3_r.x, R3_r.y, R3_r.z)
    c_f = c(R3_r.x, R3_r.y, R3_r.z)

    # 构造 theta，表示 a_f*R3_r.dx + b_f*R3_r.dy + c_f*R3_r.dz
    theta = a_f*R3_r.dx + b_f*R3_r.dy + c_f*R3_r.dz

    # 创建 theta 的微分 dtheta，以及 a_f, b_f, c_f 的微分 da, db, dc
    dtheta = Differential(theta)
    da = Differential(a_f)
    db = Differential(b_f)
    dc = Differential(c_f)

    # 构造表达式 expr，并进行断言验证
    expr = dtheta - WedgeProduct(da, R3_r.dx) - WedgeProduct(db, R3_r.dy) - WedgeProduct(dc, R3_r.dz)
    assert expr.rcall(R3_r.e_x, R3_r.e_y) == 0
```