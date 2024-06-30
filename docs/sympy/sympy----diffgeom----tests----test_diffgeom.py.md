# `D:\src\scipysrc\sympy\sympy\diffgeom\tests\test_diffgeom.py`

```
# 导入必要的符号、变量和函数
from sympy.core import Lambda, Symbol, symbols
# 导入二维和三维流形空间及其坐标系
from sympy.diffgeom.rn import R2, R2_p, R2_r, R3_r, R3_c, R3_s, R2_origin
# 导入差分几何相关模块
from sympy.diffgeom import (Manifold, Patch, CoordSystem, Commutator, Differential, TensorProduct,
        WedgeProduct, BaseCovarDerivativeOp, CovarDerivativeOp, LieDerivative,
        covariant_order, contravariant_order, twoform_to_matrix, metric_to_Christoffel_1st,
        metric_to_Christoffel_2nd, metric_to_Riemann_components,
        metric_to_Ricci_components, intcurve_diffequ, intcurve_series)
# 导入简化和数学函数
from sympy.simplify import trigsimp, simplify
from sympy.functions import sqrt, atan2, sin
# 导入矩阵和测试工具
from sympy.matrices import Matrix
from sympy.testing.pytest import raises, nocache_fail
from sympy.testing.pytest import warns_deprecated_sympy

# 定义缩写 TP 代表 TensorProduct
TP = TensorProduct

# 定义坐标系变换的测试函数
def test_coordsys_transform():
    # 测试逆变换
    p, q, r, s = symbols('p q r s')
    rel = {('first', 'second'): [(p, q), (q, -p)]}
    R2_pq = CoordSystem('first', R2_origin, [p, q], rel)
    R2_rs = CoordSystem('second', R2_origin, [r, s], rel)
    r, s = R2_rs.symbols
    assert R2_rs.transform(R2_pq) == Matrix([[-s], [r]])

    # 无法进行逆变换的情况
    a, b = symbols('a b', positive=True)
    rel = {('first', 'second'): [(a,), (-a,)]}
    R2_a = CoordSystem('first', R2_origin, [a], rel)
    R2_b = CoordSystem('second', R2_origin, [b], rel)
    # 如果找不到满足 a = -b 的正数 a, b，则会抛出 NotImplementedError
    with raises(NotImplementedError):
        R2_b.transform(R2_a)

    # 逆变换模糊的情况
    c, d = symbols('c d')
    rel = {('first', 'second'): [(c,), (c**2,)]}
    R2_c = CoordSystem('first', R2_origin, [c], rel)
    R2_d = CoordSystem('second', R2_origin, [d], rel)
    # 如果坐标变换存在多个逆的情况，transform 方法应该抛出 ValueError
    with raises(ValueError):
        R2_d.transform(R2_c)

    # 测试间接变换
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    rel = {('C1', 'C2'): [(a, b), (2*a, 3*b)],
        ('C2', 'C3'): [(c, d), (3*c, 2*d)]}
    C1 = CoordSystem('C1', R2_origin, (a, b), rel)
    C2 = CoordSystem('C2', R2_origin, (c, d), rel)
    C3 = CoordSystem('C3', R2_origin, (e, f), rel)
    a, b = C1.symbols
    c, d = C2.symbols
    e, f = C3.symbols
    assert C2.transform(C1) == Matrix([c/2, d/3])
    assert C1.transform(C3) == Matrix([6*a, 6*b])
    assert C3.transform(C1) == Matrix([e/6, f/6])
    assert C3.transform(C2) == Matrix([e/3, f/2])

    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    rel = {('C1', 'C2'): [(a, b), (2*a, 3*b + 1)],
        ('C3', 'C2'): [(e, f), (-e - 2, 2*f)]}
    C1 = CoordSystem('C1', R2_origin, (a, b), rel)
    C2 = CoordSystem('C2', R2_origin, (c, d), rel)
    C3 = CoordSystem('C3', R2_origin, (e, f), rel)
    a, b = C1.symbols
    c, d = C2.symbols
    e, f = C3.symbols
    assert C2.transform(C1) == Matrix([c/2, (d - 1)/3])
    assert C1.transform(C3) == Matrix([-2*a - 2, (3*b + 1)/2])
    # 断言：检查 C3 对象相对于 C1 对象的坐标变换是否符合预期
    assert C3.transform(C1) == Matrix([-e/2 - 1, (2*f - 1)/3])
    # 断言：检查 C3 对象相对于 C2 对象的坐标变换是否符合预期
    assert C3.transform(C2) == Matrix([-e - 2, 2*f])

    # 使用旧的签名，Lambda 函数被使用
    # 定义符号变量 a, b, c, d, e, f
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    # 定义坐标系之间的变换关系
    rel = {('C1', 'C2'): Lambda((a, b), (2*a, 3*b + 1)),
           ('C3', 'C2'): Lambda((e, f), (-e - 2, 2*f))}
    # 创建坐标系 C1，使用符号 a, b 和定义的变换关系 rel
    C1 = CoordSystem('C1', R2_origin, (a, b), rel)
    # 创建坐标系 C2，使用符号 c, d 和定义的变换关系 rel
    C2 = CoordSystem('C2', R2_origin, (c, d), rel)
    # 创建坐标系 C3，使用符号 e, f 和定义的变换关系 rel
    C3 = CoordSystem('C3', R2_origin, (e, f), rel)
    # 重新定义变量 a, b, c, d, e, f 以匹配各个坐标系的符号
    a, b = C1.symbols
    c, d = C2.symbols
    e, f = C3.symbols
    # 断言：检查 C2 对象相对于 C1 对象的坐标变换是否符合预期
    assert C2.transform(C1) == Matrix([c/2, (d - 1)/3])
    # 断言：检查 C1 对象相对于 C3 对象的坐标变换是否符合预期
    assert C1.transform(C3) == Matrix([-2*a - 2, (3*b + 1)/2])
    # 断言：检查 C3 对象相对于 C1 对象的坐标变换是否符合预期
    assert C3.transform(C1) == Matrix([-e/2 - 1, (2*f - 1)/3])
    # 断言：检查 C3 对象相对于 C2 对象的坐标变换是否符合预期
    assert C3.transform(C2) == Matrix([-e - 2, 2*f])
def test_R2():
    # 定义实数符号变量
    x0, y0, r0, theta0 = symbols('x0, y0, r0, theta0', real=True)
    # 创建 R2_r 点对象，使用 x0, y0 作为坐标
    point_r = R2_r.point([x0, y0])
    # 创建 R2_p 点对象，使用 r0, theta0 作为极坐标
    point_p = R2_p.point([r0, theta0])

    # 断言 r**2 = x**2 + y**2 在点 point_r 处成立
    assert (R2.r**2 - R2.x**2 - R2.y**2).rcall(point_r) == 0
    # 断言 trigsimp((r**2 - x**2 - y**2).rcall(point_p)) 简化后为 0
    assert trigsimp((R2.r**2 - R2.x**2 - R2.y**2).rcall(point_p)) == 0
    # 断言 trigsimp(R2.e_r(x**2 + y**2).rcall(point_p).doit()) 简化后为 2*r0
    assert trigsimp(R2.e_r(R2.x**2 + R2.y**2).rcall(point_p).doit()) == 2*r0

    # 极坐标转换为直角坐标再转回极坐标应为恒等映射
    a, b = symbols('a b', positive=True)
    m = Matrix([[a], [b]])

    # 断言 m 等于 R2_p 到 R2_r 转换后再 R2_r 到 R2_p 转换后的简化结果
    assert m == R2_p.transform(R2_r, R2_r.transform(R2_p, m)).applyfunc(simplify)

    # 使用 deprecated 方法进行断言
    with warns_deprecated_sympy():
        assert m == R2_p.coord_tuple_transform_to(
            R2_r, R2_r.coord_tuple_transform_to(R2_p, m)).applyfunc(simplify)


def test_R3():
    # 定义正实数符号变量
    a, b, c = symbols('a b c', positive=True)
    m = Matrix([[a], [b], [c]])

    # 断言 m 等于 R3_c 到 R3_r 转换后再 R3_r 到 R3_c 转换后的简化结果
    assert m == R3_c.transform(R3_r, R3_r.transform(R3_c, m)).applyfunc(simplify)
    #TODO 断言 m 等于 R3_r 到 R3_c 转换后再 R3_c 到 R3_r 转换后的简化结果
    assert m == R3_s.transform(
        R3_r, R3_r.transform(R3_s, m)).applyfunc(simplify)
    #TODO 断言 m 等于 R3_r 到 R3_s 转换后再 R3_s 到 R3_r 转换后的简化结果
    assert m == R3_s.transform(
        R3_c, R3_c.transform(R3_s, m)).applyfunc(simplify)
    #TODO 断言 m 等于 R3_c 到 R3_s 转换后再 R3_s 到 R3_c 转换后的简化结果

    # 使用 deprecated 方法进行断言
    with warns_deprecated_sympy():
        assert m == R3_c.coord_tuple_transform_to(
            R3_r, R3_r.coord_tuple_transform_to(R3_c, m)).applyfunc(simplify)
        #TODO 断言 m 等于 R3_r.coord_tuple_transform_to(R3_c, R3_c.coord_tuple_transform_to(R3_r, m)).applyfunc(simplify)
        assert m == R3_s.coord_tuple_transform_to(
            R3_r, R3_r.coord_tuple_transform_to(R3_s, m)).applyfunc(simplify)
        #TODO 断言 m 等于 R3_r.coord_tuple_transform_to(R3_s, R3_s.coord_tuple_transform_to(R3_r, m)).applyfunc(simplify)
        assert m == R3_s.coord_tuple_transform_to(
            R3_c, R3_c.coord_tuple_transform_to(R3_s, m)).applyfunc(simplify)
        #TODO 断言 m 等于 R3_c.coord_tuple_transform_to(R3_s, R3_s.coord_tuple_transform_to(R3_c, m)).applyfunc(simplify)


def test_CoordinateSymbol():
    # 获取 R2_r 和 R2_p 的符号变量
    x, y = R2_r.symbols
    r, theta = R2_p.symbols
    # 断言 y.rewrite(R2_p) 等于 r*sin(theta)
    assert y.rewrite(R2_p) == r*sin(theta)


def test_point():
    # 定义符号变量 x 和 y
    x, y = symbols('x, y')
    # 创建 R2_r 中的点对象 p
    p = R2_r.point([x, y])
    # 断言 p 的自由符号集合为 {x, y}
    assert p.free_symbols == {x, y}
    # 断言 p 在 R2_r 和默认坐标下的坐标为 Matrix([x, y])
    assert p.coords(R2_r) == p.coords() == Matrix([x, y])
    # 断言 p 在 R2_p 坐标下的坐标为 Matrix([sqrt(x**2 + y**2), atan2(y, x)])


def test_commutator():
    # 断言 R2.e_x 和 R2.e_y 的对易子为 0
    assert Commutator(R2.e_x, R2.e_y) == 0
    # 断言 R2.x*R2.e_x 和 R2.x*R2.e_x 的对易子为 0
    assert Commutator(R2.x*R2.e_x, R2.x*R2.e_x) == 0
    # 断言 R2.x*R2.e_x 和 R2.x*R2.e_y 的对易子为 R2.x*R2.e_y
    assert Commutator(R2.x*R2.e_x, R2.x*R2.e_y) == R2.x*R2.e_y
    # 定义对易子 c
    c = Commutator(R2.e_x, R2.e_r)
    # 断言 c(R2.x) 等于 R2.y*(R2.x**2 + R2.y**2)**(-1)*sin(R2.theta)


def test_differential():
    # 定义变量 xdy 和 dxdy
    xdy = R2.x*R2.dy
    dxdy = Differential(xdy)
    # 断言 xdy 对象调用时返回自身 xdy
    assert xdy.rcall(None) == xdy
    # 断言 dxdy 函数以 R2.e_x 和 R2.e_y 作为参数返回 1
    assert dxdy(R2.e_x, R2.e_y) == 1
    # 断言 dxdy 函数以 R2.e_x 和 R2.x*R2.e_y 作为参数返回 R2.x
    assert dxdy(R2.e_x, R2.x*R2.e_y) == R2.x
    # 断言 Differential(dxdy) 返回 0
    assert Differential(dxdy) == 0
def test_products():
    # 检查张量积的计算结果是否正确
    assert TensorProduct(R2.dx, R2.dy)(R2.e_x, R2.e_y) == R2.dx(R2.e_x)*R2.dy(R2.e_y) == 1
    # 检查当第一个参数为 None 时的张量积结果
    assert TensorProduct(R2.dx, R2.dy)(None, R2.e_y) == R2.dx
    # 检查当第二个参数为 None 时的张量积结果
    assert TensorProduct(R2.dx, R2.dy)(R2.e_x, None) == R2.dy
    # 检查只传入一个参数时的张量积结果
    assert TensorProduct(R2.dx, R2.dy)(R2.e_x) == R2.dy
    # 检查单个变量与微分形式的张量积结果
    assert TensorProduct(R2.x, R2.dx) == R2.x*R2.dx
    # 检查基向量的张量积计算结果
    assert TensorProduct(R2.e_x, R2.e_y)(R2.x, R2.y) == R2.e_x(R2.x) * R2.e_y(R2.y) == 1
    assert TensorProduct(R2.e_x, R2.e_y)(None, R2.y) == R2.e_x
    assert TensorProduct(R2.e_x, R2.e_y)(R2.x, None) == R2.e_y
    assert TensorProduct(R2.e_x, R2.e_y)(R2.x) == R2.e_y
    assert TensorProduct(R2.x, R2.e_x) == R2.x * R2.e_x
    assert TensorProduct(R2.dx, R2.e_y)(R2.e_x, R2.y) == R2.dx(R2.e_x) * R2.e_y(R2.y) == 1
    assert TensorProduct(R2.dx, R2.e_y)(None, R2.y) == R2.dx
    assert TensorProduct(R2.dx, R2.e_y)(R2.e_x, None) == R2.e_y
    assert TensorProduct(R2.dx, R2.e_y)(R2.e_x) == R2.e_y
    assert TensorProduct(R2.x, R2.e_x) == R2.x * R2.e_x
    assert TensorProduct(R2.e_x, R2.dy)(R2.x, R2.e_y) == R2.e_x(R2.x) * R2.dy(R2.e_y) == 1
    assert TensorProduct(R2.e_x, R2.dy)(None, R2.e_y) == R2.e_x
    assert TensorProduct(R2.e_x, R2.dy)(R2.x, None) == R2.dy
    assert TensorProduct(R2.e_x, R2.dy)(R2.x) == R2.dy
    # 检查多个基向量的张量积结果
    assert TensorProduct(R2.e_y, R2.e_x)(R2.x**2 + R2.y**2, R2.x**2 + R2.y**2) == 4*R2.x*R2.y

    # 检查楔积的计算结果
    assert WedgeProduct(R2.dx, R2.dy)(R2.e_x, R2.e_y) == 1
    assert WedgeProduct(R2.e_x, R2.e_y)(R2.x, R2.y) == 1


def test_lie_derivative():
    # 检查李导数的计算结果
    assert LieDerivative(R2.e_x, R2.y) == R2.e_x(R2.y) == 0
    assert LieDerivative(R2.e_x, R2.x) == R2.e_x(R2.x) == 1
    assert LieDerivative(R2.e_x, R2.e_x) == Commutator(R2.e_x, R2.e_x) == 0
    assert LieDerivative(R2.e_x, R2.e_r) == Commutator(R2.e_x, R2.e_r)
    assert LieDerivative(R2.e_x + R2.e_y, R2.x) == 1
    assert LieDerivative(R2.e_x, TensorProduct(R2.dx, R2.dy))(R2.e_x, R2.e_y) == 0


@nocache_fail
def test_covar_deriv():
    # 转换度规为第二克里斯托弗符号并进行协变导数计算
    ch = metric_to_Christoffel_2nd(TP(R2.dx, R2.dx) + TP(R2.dy, R2.dy))
    cvd = BaseCovarDerivativeOp(R2_r, 0, ch)
    assert cvd(R2.x) == 1
    # 当缓存禁用时，以下断言会失败：
    assert cvd(R2.x*R2.e_x) == R2.e_x
    cvd = CovarDerivativeOp(R2.x*R2.e_x, ch)
    assert cvd(R2.x) == R2.x
    assert cvd(R2.x*R2.e_x) == R2.x*R2.e_x


def test_intcurve_diffequ():
    # 检查积分曲线微分方程的生成结果
    t = symbols('t')
    start_point = R2_r.point([1, 0])
    vector_field = -R2.y*R2.e_x + R2.x*R2.e_y
    equations, init_cond = intcurve_diffequ(vector_field, t, start_point)
    assert str(equations) == '[f_1(t) + Derivative(f_0(t), t), -f_0(t) + Derivative(f_1(t), t)]'
    assert str(init_cond) == '[f_0(0) - 1, f_1(0)]'
    equations, init_cond = intcurve_diffequ(vector_field, t, start_point, R2_p)
    assert str(equations) == '[Derivative(f_0(t), t), Derivative(f_1(t), t) - 1]'
    assert str(init_cond) == '[f_0(0) - 1, f_1(0)]'


def test_helpers_and_coordinate_dependent():
    # 未提供相关代码段，故此处不需要注释
    pass
    # 定义一阶微分形式：dr 和 dx
    one_form = R2.dr + R2.dx
    # 定义二阶微分形式：x*dr + r*dx 的微分
    two_form = Differential(R2.x*R2.dr + R2.r*R2.dx)
    # 定义三阶微分形式：y*(x*dr + r*dx) + x*d(r*dx) 的微分
    three_form = Differential(R2.y*two_form) + Differential(R2.x*Differential(R2.r*R2.dr))
    # 定义度量张量：dx与dx的张量积 + dy与dy的张量积
    metric = TensorProduct(R2.dx, R2.dx) + TensorProduct(R2.dy, R2.dy)
    # 定义含有模棱两可项的度量张量：dx与dx的张量积 + dr与dr的张量积
    metric_ambig = TensorProduct(R2.dx, R2.dx) + TensorProduct(R2.dr, R2.dr)
    # 定义格式不正确的形式：dr与dr的张量积 + dr
    misform_a = TensorProduct(R2.dr, R2.dr) + R2.dr
    # 定义格式不正确的形式：dr的四次方
    misform_b = R2.dr**4
    # 定义格式不正确的形式：dx与dy的张量积
    misform_c = R2.dx*R2.dy
    # 定义非对称的二阶微分形式：dx与dx的张量积 + dx与dy的张量积
    twoform_not_sym = TensorProduct(R2.dx, R2.dx) + TensorProduct(R2.dx, R2.dy)
    # 定义不是张量积的楔积形式：dx与dy的楔积
    twoform_not_TP = WedgeProduct(R2.dx, R2.dy)

    # 定义一阶向量：e_x 和 e_y 的和
    one_vector = R2.e_x + R2.e_y
    # 定义二阶向量：e_x 与 e_y 的张量积
    two_vector = TensorProduct(R2.e_x, R2.e_y)
    # 定义三阶向量：e_x 与 e_y 的张量积 与 e_x 的张量积
    three_vector = TensorProduct(R2.e_x, R2.e_y, R2.e_x)
    # 定义不是张量积的楔积形式：e_x 与 e_y 的楔积
    two_wp = WedgeProduct(R2.e_x, R2.e_y)

    # 断言验证各形式的协变阶数是否符合预期
    assert covariant_order(one_form) == 1
    assert covariant_order(two_form) == 2
    assert covariant_order(three_form) == 3
    assert covariant_order(two_form + metric) == 2
    assert covariant_order(two_form + metric_ambig) == 2
    assert covariant_order(two_form + twoform_not_sym) == 2
    assert covariant_order(two_form + twoform_not_TP) == 2

    # 断言验证各向量的逆变阶数是否符合预期
    assert contravariant_order(one_vector) == 1
    assert contravariant_order(two_vector) == 2
    assert contravariant_order(three_vector) == 3
    assert contravariant_order(two_vector + two_wp) == 2

    # 验证各不正确格式的形式是否会引发值错误异常
    raises(ValueError, lambda: covariant_order(misform_a))
    raises(ValueError, lambda: covariant_order(misform_b))
    raises(ValueError, lambda: covariant_order(misform_c))

    # 断言验证将给定的二阶微分形式转换为矩阵的操作是否符合预期
    assert twoform_to_matrix(metric) == Matrix([[1, 0], [0, 1]])
    assert twoform_to_matrix(twoform_not_sym) == Matrix([[1, 0], [1, 0]])
    assert twoform_to_matrix(twoform_not_TP) == Matrix([[0, -1], [1, 0]])

    # 验证将不正确格式的形式转换为矩阵是否会引发值错误异常
    raises(ValueError, lambda: twoform_to_matrix(one_form))
    raises(ValueError, lambda: twoform_to_matrix(three_form))
    raises(ValueError, lambda: twoform_to_matrix(metric_ambig))

    # 验证将给定的二阶微分形式转换为克里斯托费尔符号的第一类成分、第二类成分、黎曼张量分量、里奇张量分量的操作是否会引发值错误异常
    raises(ValueError, lambda: metric_to_Christoffel_1st(twoform_not_sym))
    raises(ValueError, lambda: metric_to_Christoffel_2nd(twoform_not_sym))
    raises(ValueError, lambda: metric_to_Riemann_components(twoform_not_sym))
    raises(ValueError, lambda: metric_to_Ricci_components(twoform_not_sym))
# 测试函数，检查正确参数的情况下是否引发异常

def test_correct_arguments():
    # 检查 R2.e_x(R2.e_x) 是否引发值错误异常
    raises(ValueError, lambda: R2.e_x(R2.e_x))
    # 检查 R2.e_x(R2.dx) 是否引发值错误异常
    raises(ValueError, lambda: R2.e_x(R2.dx))

    # 检查 Commutator(R2.e_x, R2.x) 是否引发值错误异常
    raises(ValueError, lambda: Commutator(R2.e_x, R2.x))
    # 检查 Commutator(R2.dx, R2.e_x) 是否引发值错误异常
    raises(ValueError, lambda: Commutator(R2.dx, R2.e_x))

    # 检查 Differential(Differential(R2.e_x)) 是否引发值错误异常
    raises(ValueError, lambda: Differential(Differential(R2.e_x)))

    # 检查 R2.dx(R2.x) 是否引发值错误异常
    raises(ValueError, lambda: R2.dx(R2.x))

    # 检查 LieDerivative(R2.dx, R2.dx) 是否引发值错误异常
    raises(ValueError, lambda: LieDerivative(R2.dx, R2.dx))
    # 检查 LieDerivative(R2.x, R2.dx) 是否引发值错误异常
    raises(ValueError, lambda: LieDerivative(R2.x, R2.dx))

    # 检查 CovarDerivativeOp(R2.dx, []) 是否引发值错误异常
    raises(ValueError, lambda: CovarDerivativeOp(R2.dx, []))
    # 检查 CovarDerivativeOp(R2.x, []) 是否引发值错误异常
    raises(ValueError, lambda: CovarDerivativeOp(R2.x, []))

    # 创建符号 'a'
    a = Symbol('a')
    # 检查 intcurve_series(R2.dx, a, R2_r.point([1, 2])) 是否引发值错误异常
    raises(ValueError, lambda: intcurve_series(R2.dx, a, R2_r.point([1, 2])))
    # 检查 intcurve_series(R2.x, a, R2_r.point([1, 2])) 是否引发值错误异常
    raises(ValueError, lambda: intcurve_series(R2.x, a, R2_r.point([1, 2])))

    # 检查 intcurve_diffequ(R2.dx, a, R2_r.point([1, 2])) 是否引发值错误异常
    raises(ValueError, lambda: intcurve_diffequ(R2.dx, a, R2_r.point([1, 2])))
    # 检查 intcurve_diffequ(R2.x, a, R2_r.point([1, 2])) 是否引发值错误异常
    raises(ValueError, lambda: intcurve_diffequ(R2.x, a, R2_r.point([1, 2])))

    # 检查 contravariant_order(R2.e_x + R2.dx) 是否引发值错误异常
    raises(ValueError, lambda: contravariant_order(R2.e_x + R2.dx))
    # 检查 covariant_order(R2.e_x + R2.dx) 是否引发值错误异常
    raises(ValueError, lambda: covariant_order(R2.e_x + R2.dx))

    # 检查 contravariant_order(R2.e_x*R2.e_y) 是否引发值错误异常
    raises(ValueError, lambda: contravariant_order(R2.e_x*R2.e_y))
    # 检查 covariant_order(R2.dx*R2.dy) 是否引发值错误异常
    raises(ValueError, lambda: covariant_order(R2.dx*R2.dy))


# 测试函数，检查简化操作是否正确

def test_simplify():
    # 获取 R2_r 的坐标函数和基一形式
    x, y = R2_r.coord_functions()
    dx, dy = R2_r.base_oneforms()
    ex, ey = R2_r.base_vectors()

    # 断言简化 x 是否等于 x
    assert simplify(x) == x
    # 断言简化 x*y 是否等于 x*y
    assert simplify(x*y) == x*y
    # 断言简化 dx*dy 是否等于 dx*dy
    assert simplify(dx*dy) == dx*dy
    # 断言简化 ex*ey 是否等于 ex*ey
    assert simplify(ex*ey) == ex*ey
    # 断言 (1-x)*dx / (1-x)**2 是否等于 dx / (1-x)

    assert ((1-x)*dx)/(1-x)**2 == dx/(1-x)


# 测试函数，检查问题编号 17917 是否解决

def test_issue_17917():
    # 定义向量场 X 和 Y
    X = R2.x*R2.e_x - R2.y*R2.e_y
    Y = (R2.x**2 + R2.y**2)*R2.e_x - R2.x*R2.y*R2.e_y

    # 断言 X 和 Y 的李导数展开结果是否等于预期结果
    assert LieDerivative(X, Y).expand() == (
        R2.x**2*R2.e_x - 3*R2.y**2*R2.e_x - R2.x*R2.y*R2.e_y)


# 测试函数，检查已弃用功能

def test_deprecations():
    # 创建流形 'M' 和补丁 'P'
    m = Manifold('M', 2)
    p = Patch('P', m)

    # 使用 warns_deprecated_sympy 上下文，检查 CoordSystem 的使用是否引发弃用警告
    with warns_deprecated_sympy():
        CoordSystem('Car2d', p, names=['x', 'y'])

    # 使用 warns_deprecated_sympy 上下文，检查 CoordSystem 的使用是否引发弃用警告
    with warns_deprecated_sympy():
        c = CoordSystem('Car2d', p, ['x', 'y'])

    # 使用 warns_deprecated_sympy 上下文，检查流形 m 的 patches 属性是否引发弃用警告
    with warns_deprecated_sympy():
        list(m.patches)

    # 使用 warns_deprecated_sympy 上下文，检查坐标系 c 的 transforms 属性是否引发弃用警告
    with warns_deprecated_sympy():
        list(c.transforms)
```