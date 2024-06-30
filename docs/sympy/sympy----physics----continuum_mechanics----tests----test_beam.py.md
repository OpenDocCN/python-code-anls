# `D:\src\scipysrc\sympy\sympy\physics\continuum_mechanics\tests\test_beam.py`

```
# 从sympy库中导入需要的函数和类
from sympy.core.function import expand
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.sets.sets import Interval
from sympy.simplify.simplify import simplify
from sympy.physics.continuum_mechanics.beam import Beam
from sympy.functions import SingularityFunction, Piecewise, meijerg, Abs, log
from sympy.testing.pytest import raises
from sympy.physics.units import meter, newton, kilo, giga, milli
from sympy.physics.continuum_mechanics.beam import Beam3D
from sympy.geometry import Circle, Polygon, Point2D, Triangle
from sympy.core.sympify import sympify

# 创建符号变量 x 和 y
x = Symbol('x')
y = Symbol('y')

# 创建符号变量 R1 和 R2
R1, R2 = symbols('R1, R2')

# 定义测试函数 test_Beam
def test_Beam():
    # 创建符号变量 E, E_1, I, I_1, A
    E = Symbol('E')
    E_1 = Symbol('E_1')
    I = Symbol('I')
    I_1 = Symbol('I_1')
    A = Symbol('A')

    # 创建 Beam 对象 b，指定长度、弹性模量和惯性矩
    b = Beam(1, E, I)
    assert b.length == 1
    assert b.elastic_modulus == E
    assert b.second_moment == I
    assert b.variable == x

    # 测试长度设置器
    b.length = 4
    assert b.length == 4

    # 测试弹性模量设置器
    b.elastic_modulus = E_1
    assert b.elastic_modulus == E_1

    # 测试惯性矩设置器
    b.second_moment = I_1
    assert b.second_moment is I_1

    # 测试变量设置器
    b.variable = y
    assert b.variable is y

    # 测试所有边界条件
    b.bc_deflection = [(0, 2)]
    b.bc_slope = [(0, 1)]
    b.bc_bending_moment = [(0, 5)]
    b.bc_shear_force = [(2, 1)]
    assert b.boundary_conditions == {'deflection': [(0, 2)], 'slope': [(0, 1)],
                                     'bending_moment': [(0, 5)], 'shear_force': [(2, 1)]}

    # 测试剪力边界条件方法
    b.bc_shear_force.extend([(1, 1), (2, 3)])
    sf_bcs = b.bc_shear_force
    assert sf_bcs == [(2, 1), (1, 1), (2, 3)]

    # 测试弯矩边界条件方法
    b.bc_bending_moment.extend([(1, 3), (5, 3)])
    bm_bcs = b.bc_bending_moment
    assert bm_bcs == [(0, 5), (1, 3), (5, 3)]

    # 测试斜率边界条件方法
    b.bc_slope.extend([(4, 3), (5, 0)])
    s_bcs = b.bc_slope
    assert s_bcs == [(0, 1), (4, 3), (5, 0)]

    # 测试挠度边界条件方法
    b.bc_deflection.extend([(4, 3), (5, 0)])
    d_bcs = b.bc_deflection
    assert d_bcs == [(0, 2), (4, 3), (5, 0)]

    # 测试更新后的边界条件
    bcs_new = b.boundary_conditions
    assert bcs_new == {
        'deflection': [(0, 2), (4, 3), (5, 0)],
        'slope': [(0, 1), (4, 3), (5, 0)],
        'bending_moment': [(0, 5), (1, 3), (5, 3)],
        'shear_force': [(2, 1), (1, 1), (2, 3)]}

    # 创建 Beam 对象 b1，指定长度、弹性模量和惯性矩
    b1 = Beam(30, E, I)
    
    # 在梁上施加载荷
    b1.apply_load(-8, 0, -1)
    b1.apply_load(R1, 10, -1)
    b1.apply_load(R2, 30, -1)
    b1.apply_load(120, 30, -2)
    
    # 设置挠度边界条件
    b1.bc_deflection = [(10, 0), (30, 0)]
    
    # 求解反力
    b1.solve_for_reaction_loads(R1, R2)

    # 测试找到的反力
    p = b1.reaction_loads
    q = {R1: 6, R2: 2}
    assert p == q

    # 测试负载分布函数。
    # 将 b1 的 load 方法赋给变量 p
    p = b1.load
    # 定义一个 SingularityFunction 对象 q，表示力的分布
    q = -8*SingularityFunction(x, 0, -1) + 6*SingularityFunction(x, 10, -1) \
    + 120*SingularityFunction(x, 30, -2) + 2*SingularityFunction(x, 30, -1)
    # 断言 p 和 q 相等
    assert p == q

    # 测试剪力分布函数
    p = b1.shear_force()
    q = 8*SingularityFunction(x, 0, 0) - 6*SingularityFunction(x, 10, 0) \
    - 120*SingularityFunction(x, 30, -1) - 2*SingularityFunction(x, 30, 0)
    assert p == q

    # 测试剪应力分布函数
    p = b1.shear_stress()
    q = (8*SingularityFunction(x, 0, 0) - 6*SingularityFunction(x, 10, 0) \
    - 120*SingularityFunction(x, 30, -1) \
    - 2*SingularityFunction(x, 30, 0))/A
    assert p == q

    # 测试弯矩分布函数
    p = b1.bending_moment()
    q = 8*SingularityFunction(x, 0, 1) - 6*SingularityFunction(x, 10, 1) \
    - 120*SingularityFunction(x, 30, 0) - 2*SingularityFunction(x, 30, 1)
    assert p == q

    # 测试斜率分布函数
    p = b1.slope()
    q = -4*SingularityFunction(x, 0, 2) + 3*SingularityFunction(x, 10, 2) \
    + 120*SingularityFunction(x, 30, 1) + SingularityFunction(x, 30, 2) \
    + Rational(4000, 3)
    assert p == q/(E*I)

    # 测试挠度分布函数
    p = b1.deflection()
    q = x*Rational(4000, 3) - 4*SingularityFunction(x, 0, 3)/3 \
    + SingularityFunction(x, 10, 3) + 60*SingularityFunction(x, 30, 2) \
    + SingularityFunction(x, 30, 3)/3 - 12000
    assert p == q/(E*I)

    # 使用符号进行测试
    l = Symbol('l')
    w0 = Symbol('w0')
    w2 = Symbol('w2')
    a1 = Symbol('a1')
    c = Symbol('c')
    c1 = Symbol('c1')
    d = Symbol('d')
    e = Symbol('e')
    f = Symbol('f')

    # 创建一个 Beam 对象 b2
    b2 = Beam(l, E, I)

    # 对 b2 应用集中力
    b2.apply_load(w0, a1, 1)
    b2.apply_load(w2, c1, -1)

    # 设置 b2 的挠度边界条件
    b2.bc_deflection = [(c, d)]
    # 设置 b2 的斜率边界条件
    b2.bc_slope = [(e, f)]

    # 测试载荷分布函数
    p = b2.load
    q = w0*SingularityFunction(x, a1, 1) + w2*SingularityFunction(x, c1, -1)
    assert p == q

    # 测试剪力分布函数
    p = b2.shear_force()
    q = -w0*SingularityFunction(x, a1, 2)/2 \
    - w2*SingularityFunction(x, c1, 0)
    assert p == q

    # 测试剪应力分布函数
    p = b2.shear_stress()
    q = (-w0*SingularityFunction(x, a1, 2)/2 \
    - w2*SingularityFunction(x, c1, 0))/A
    assert p == q

    # 测试弯矩分布函数
    p = b2.bending_moment()
    q = -w0*SingularityFunction(x, a1, 3)/6 - w2*SingularityFunction(x, c1, 1)
    assert p == q

    # 测试斜率分布函数
    p = b2.slope()
    q = (w0*SingularityFunction(x, a1, 4)/24 + w2*SingularityFunction(x, c1, 2)/2)/(E*I) + (E*I*f - w0*SingularityFunction(e, a1, 4)/24 - w2*SingularityFunction(e, c1, 2)/2)/(E*I)
    assert expand(p) == expand(q)

    # 测试挠度分布函数
    p = b2.deflection()
    q = x*(E*I*f - w0*SingularityFunction(e, a1, 4)/24 \
    - w2*SingularityFunction(e, c1, 2)/2)/(E*I)
    assert p == q
    # 将多项式表达式分成两部分，每部分表示一个力和反力对应的弯曲方程
    + (w0*SingularityFunction(x, a1, 5)/120 \
    + w2*SingularityFunction(x, c1, 3)/6)/(E*I) \
    + (E*I*(-c*f + d) + c*w0*SingularityFunction(e, a1, 4)/24 \
    + c*w2*SingularityFunction(e, c1, 2)/2 \
    - w0*SingularityFunction(c, a1, 5)/120 \
    - w2*SingularityFunction(c, c1, 3)/6)/(E*I)
    # 断言两个表达式的简化结果相等
    assert simplify(p - q) == 0

    # 创建一个梁对象，长度为9，弹性模量为E，惯性矩为I，加载方式为2
    b3 = Beam(9, E, I, 2)
    # 在梁上施加一个力：大小为-2，从位置2开始作用，作用到位置3结束，阶数为2
    b3.apply_load(value=-2, start=2, order=2, end=3)
    # 在梁的端点添加一个边界条件：从位置0到位置2的斜率为2
    b3.bc_slope.append((0, 2))
    # 创建符号变量C3和C4
    C3 = symbols('C3')
    C4 = symbols('C4')

    # 获取梁的加载力
    p = b3.load
    # 定义期望的加载力表达式q
    q = -2*SingularityFunction(x, 2, 2) + 2*SingularityFunction(x, 3, 0) \
    + 4*SingularityFunction(x, 3, 1) + 2*SingularityFunction(x, 3, 2)
    # 断言计算出的加载力和期望的加载力相等
    assert p == q

    # 获取梁的剪力图
    p = b3.shear_force()
    # 定义期望的剪力图表达式q
    q = 2*SingularityFunction(x, 2, 3)/3 - 2*SingularityFunction(x, 3, 1) \
    - 2*SingularityFunction(x, 3, 2) - 2*SingularityFunction(x, 3, 3)/3
    # 断言计算出的剪力图和期望的剪力图相等
    assert p == q

    # 获取梁的剪应力分布
    p = b3.shear_stress()
    # 定义期望的剪应力分布表达式q
    q = SingularityFunction(x, 2, 3)/3 - 1*SingularityFunction(x, 3, 1) \
    - 1*SingularityFunction(x, 3, 2) - 1*SingularityFunction(x, 3, 3)/3
    # 断言计算出的剪应力分布和期望的剪应力分布相等
    assert p == q

    # 获取梁的斜率图
    p = b3.slope()
    # 定义期望的斜率图表达式q
    q = 2 - (SingularityFunction(x, 2, 5)/30 - SingularityFunction(x, 3, 3)/3 \
    - SingularityFunction(x, 3, 4)/6 - SingularityFunction(x, 3, 5)/30)/(E*I)
    # 断言计算出的斜率图和期望的斜率图相等
    assert p == q

    # 获取梁的挠度图
    p = b3.deflection()
    # 定义期望的挠度图表达式q
    q = 2*x - (SingularityFunction(x, 2, 6)/180 \
    - SingularityFunction(x, 3, 4)/12 - SingularityFunction(x, 3, 5)/30 \
    - SingularityFunction(x, 3, 6)/180)/(E*I)
    # 断言计算出的挠度图和期望的挠度图相等，加上常数C4
    assert p == q + C4

    # 创建另一个梁对象，长度为4，弹性模量为E，惯性矩为I，加载方式为3
    b4 = Beam(4, E, I, 3)
    # 在梁上施加一个力：大小为-3，从位置0开始作用，作用到位置3结束，阶数为0
    b4.apply_load(-3, 0, 0, end=3)

    # 获取梁的加载力
    p = b4.load
    # 定义期望的加载力表达式q
    q = -3*SingularityFunction(x, 0, 0) + 3*SingularityFunction(x, 3, 0)
    # 断言计算出的加载力和期望的加载力相等
    assert p == q

    # 获取梁的剪力图
    p = b4.shear_force()
    # 定义期望的剪力图表达式q
    q = 3*SingularityFunction(x, 0, 1) \
    - 3*SingularityFunction(x, 3, 1)
    # 断言计算出的剪力图和期望的剪力图相等
    assert p == q

    # 获取梁的剪应力分布
    p = b4.shear_stress()
    # 定义期望的剪应力分布表达式q
    q = SingularityFunction(x, 0, 1) - SingularityFunction(x, 3, 1)
    # 断言计算出的剪应力分布和期望的剪应力分布相等
    assert p == q

    # 获取梁的斜率图
    p = b4.slope()
    # 定义期望的斜率图表达式q
    q = -3*SingularityFunction(x, 0, 3)/6 + 3*SingularityFunction(x, 3, 3)/6
    # 断言计算出的斜率图和期望的斜率图相等，除以(E*I)，并加上常数C3
    assert p == q/(E*I) + C3

    # 获取梁的挠度图
    p = b4.deflection()
    # 定义期望的挠度图表达式q
    q = -3*SingularityFunction(x, 0, 4)/24 + 3*SingularityFunction(x, 3, 4)/24
    # 断言计算出的挠度图和期望的挠度图相等，除以(E*I)，并加上常数C3乘以x再加上常数C4
    assert p == q/(E*I) + C3*x + C4

    # 施加在点荷载上使用end会抛出值错误
    raises(ValueError, lambda: b4.apply_load(-3, 0, -1, end=3))
    # 施加在点荷载上使用类型错误
    with raises(TypeError):
        b4.variable = 1
# 测试边界条件不足时的情况
def test_insufficient_bconditions():
    # 定义长度为正的符号变量 L
    L = symbols('L', positive=True)
    # 定义符号变量 E, I, P, a3, a4
    E, I, P, a3, a4 = symbols('E I P a3 a4')

    # 创建一个梁对象 b，基础特征为 'a'，传入长度 L、弹性模量 E、惯性矩 I
    b = Beam(L, E, I, base_char='a')
    # 在梁上施加集中力 R2，位置在 L 处，向下
    b.apply_load(R2, L, -1)
    # 在梁上施加集中力 R1，位置在 0 处，向下
    b.apply_load(R1, 0, -1)
    # 在梁上施加集中力 -P，位置在 L/2 处，向下
    b.apply_load(-P, L/2, -1)
    # 解算梁的支反力 R1, R2
    b.solve_for_reaction_loads(R1, R2)

    # 计算梁的斜率
    p = b.slope()
    # 预期的斜率 q
    q = P*SingularityFunction(x, 0, 2)/4 - P*SingularityFunction(x, L/2, 2)/2 + P*SingularityFunction(x, L, 2)/4
    # 斜率应满足公式 p == q/(E*I) + a3
    assert p == q/(E*I) + a3

    # 计算梁的挠度
    p = b.deflection()
    # 预期的挠度 q
    q = P*SingularityFunction(x, 0, 3)/12 - P*SingularityFunction(x, L/2, 3)/6 + P*SingularityFunction(x, L, 3)/12
    # 挠度应满足公式 p == q/(E*I) + a3*x + a4
    assert p == q/(E*I) + a3*x + a4

    # 设定梁的挠度边界条件为 [(0, 0)]
    b.bc_deflection = [(0, 0)]
    # 计算梁的挠度
    p = b.deflection()
    # 预期的挠度 q
    q = a3*x + P*SingularityFunction(x, 0, 3)/12 - P*SingularityFunction(x, L/2, 3)/6 + P*SingularityFunction(x, L, 3)/12
    # 挠度应满足公式 p == q/(E*I)
    assert p == q/(E*I)

    # 设定梁的挠度边界条件为 [(0, 0), (L, 0)]
    b.bc_deflection = [(0, 0), (L, 0)]
    # 计算梁的挠度
    p = b.deflection()
    # 预期的挠度 q
    q = -L**2*P*x/16 + P*SingularityFunction(x, 0, 3)/12 - P*SingularityFunction(x, L/2, 3)/6 + P*SingularityFunction(x, L, 3)/12
    # 挠度应满足公式 p == q/(E*I)
    assert p == q/(E*I)


# 测试静定梁的情况
def test_statically_indeterminate():
    # 定义符号变量 E, I
    E = Symbol('E')
    I = Symbol('I')
    # 定义符号变量 M1, M2
    M1, M2 = symbols('M1, M2')
    # 定义符号变量 F
    F = Symbol('F')
    # 定义正数的符号变量 l
    l = Symbol('l', positive=True)

    # 创建一个梁对象 b5，传入长度 l、弹性模量 E、惯性矩 I
    b5 = Beam(l, E, I)
    # 设定梁的挠度边界条件为 [(0, 0), (l, 0)]
    b5.bc_deflection = [(0, 0),(l, 0)]

    # 在梁上施加集中力 R1，位置在 0 处，向下
    b5.apply_load(R1, 0, -1)
    # 在梁上施加集中力 M1，位置在 0 处，逆时针
    b5.apply_load(M1, 0, -2)
    # 在梁上施加集中力 R2，位置在 l 处，向下
    b5.apply_load(R2, l, -1)
    # 在梁上施加集中力 M2，位置在 l 处，逆时针
    b5.apply_load(M2, l, -2)
    # 在梁上施加集中力 -F，位置在 l/2 处，向下
    b5.apply_load(-F, l/2, -1)

    # 解算梁的支反力 R1, R2, M1, M2
    b5.solve_for_reaction_loads(R1, R2, M1, M2)
    # 计算梁的反力
    p = b5.reaction_loads
    # 预期的反力 q
    q = {R1: F/2, R2: F/2, M1: -F*l/8, M2: F*l/8}
    # 反力应满足预期的结果
    assert p == q


# 测试梁的单位和力的情况
def test_beam_units():
    # 定义符号变量 E, I
    E = Symbol('E')
    I = Symbol('I')
    # 定义符号变量 R1, R2
    R1, R2 = symbols('R1, R2')

    # 定义单位 kN 和 gN
    kN = kilo*newton
    gN = giga*newton

    # 创建一个梁对象 b，传入长度 8 米、单位面积重 200 gN/m^2、惯性矩 400*10^6 mm^4
    b = Beam(8*meter, 200*gN/meter**2, 400*1000000*(milli*meter)**4)
    # 在梁上施加集中力 5 kN，位置在 2 米处，向下
    b.apply_load(5*kN, 2*meter, -1)
    # 在梁上施加集中力 R1，位置在 0 米处，向下
    b.apply_load(R1, 0*meter, -1)
    # 在梁上施加集中力 R2，位置在 8 米处，向下
    b.apply_load(R2, 8*meter, -1)
    # 在梁上施加均布力 10 kN/m，从 4 米到 8 米处，向下
    b.apply_load(10*kN/meter, 4*meter, 0, end=8*meter)
    # 设定梁的挠度边界条件为 [(0 米, 0 米), (8 米, 0 米)]
    b.bc_deflection = [(0*meter, 0*meter), (8*meter, 0*meter)]
    # 解算梁的支反力 R1, R2
    b.solve_for_reaction_loads(R1, R2)
    # 检查梁的反力结果是否符合预期
    assert b.reaction_loads == {R1: -13750*newton, R2: -31250*newton}

    # 创建一个梁对象 b，传入长度 3 米、弹性模量 E*newton/m^2、惯性矩 I*m^4
    b = Beam(3*meter, E*newton/meter**2, I*meter**4)
    # 在梁上施加集中力 8 kN，位置在 1 米处，向下
    b.apply_load(8*kN, 1*meter, -1)
    # 在梁上施加集中力 R1，位置在 0 米处
    # 计算反应力和弯矩的解析解
    b.solve_for_reaction_loads(R, M)
    # 断言梁的斜率函数与给定的解析表达式相等
    assert b.slope().expand() == ((10*x*SingularityFunction(x, 0, 0)
        - 10*(x - 4)*SingularityFunction(x, 4, 0))/E).expand()
    # 断言梁的挠度函数与给定的解析表达式相等
    assert b.deflection().expand() == ((5*x**2*SingularityFunction(x, 0, 0)
        - 10*Piecewise((0, Abs(x)/4 < 1), (x**2*meijerg(((-1, 1), ()), ((), (-2, 0)), x/4), True))
        + 40*SingularityFunction(x, 4, 1))/E).expand()

    # 创建一个长度为4的梁，具有变化的弹性模量 E - x 和惯性矩 I
    b = Beam(4, E - x, I)
    # 在距离梁左端 4 个单位处施加一个力大小为 20，方向向下
    b.apply_load(20, 4, -1)
    # 定义符号变量 R 和 M
    R, M = symbols('R, M')
    # 在梁左端施加一个反向向上的力 R
    b.apply_load(R, 0, -1)
    # 在梁左端施加一个反向的弯矩 M
    b.apply_load(M, 0, -2)
    # 设置梁的挠度边界条件为 [(0, 0)]
    b.bc_deflection = [(0, 0)]
    # 设置梁的斜率边界条件为 [(0, 0)]
    b.bc_slope = [(0, 0)]
    # 计算反应力和弯矩的解析解
    b.solve_for_reaction_loads(R, M)
    # 断言梁的斜率函数与给定的解析表达式相等
    assert b.slope().expand() == ((-80*(-log(-E) + log(-E + x))*SingularityFunction(x, 0, 0)
        + 80*(-log(-E + 4) + log(-E + x))*SingularityFunction(x, 4, 0) + 20*(-E*log(-E)
        + E*log(-E + x) + x)*SingularityFunction(x, 0, 0) - 20*(-E*log(-E + 4) + E*log(-E + x)
        + x - 4)*SingularityFunction(x, 4, 0))/I).expand()
# 定义测试函数 `test_composite_beam`
def test_composite_beam():
    # 定义符号 E 和 I
    E = Symbol('E')
    I = Symbol('I')
    
    # 创建长度为 2 的梁 b1 和 b2，使用给定的 E 和 1.5*I 或者 E 和 I
    b1 = Beam(2, E, 1.5*I)
    b2 = Beam(2, E, I)
    
    # 将两个梁 b1 和 b2 以固定支座方式连接成一个新梁 b
    b = b1.join(b2, "fixed")
    
    # 在梁 b 上施加载荷：-20N 在 x=0 处，80N 在 x=0 处，-2 处，20N 在 x=4 处
    b.apply_load(-20, 0, -1)
    b.apply_load(80, 0, -2)
    b.apply_load(20, 4, -1)
    
    # 设置边界条件：在 x=0 处的角度为 0，位移为 0
    b.bc_slope = [(0, 0)]
    b.bc_deflection = [(0, 0)]
    
    # 断言梁 b 的长度为 4
    assert b.length == 4
    
    # 断言梁 b 的二阶矩为 Piecewise((1.5*I, x <= 2), (I, x <= 4))
    assert b.second_moment == Piecewise((1.5*I, x <= 2), (I, x <= 4))
    
    # 断言梁 b 在 x=4 处的斜率为 120.0/(E*I)
    assert b.slope().subs(x, 4) == 120.0/(E*I)
    
    # 断言梁 b 在 x=2 处的斜率为 80.0/(E*I)
    assert b.slope().subs(x, 2) == 80.0/(E*I)
    
    # 断言梁 b 在 x=4 处的挠度的整数系数为 -302，即 1/(E*I) 的系数为 -302
    assert int(b.deflection().subs(x, 4).args[0]) == -302
    
    # 定义正数符号 l
    l = symbols('l', positive=True)
    
    # 定义反力和弯矩 R1, M1, R2, R3, P
    R1, M1, R2, R3, P = symbols('R1 M1 R2 R3 P')
    
    # 创建长度为 2*l 的梁 b1 和 b2，使用给定的 E 和 I
    b1 = Beam(2*l, E, I)
    b2 = Beam(2*l, E, I)
    
    # 将两个梁 b1 和 b2 以铰链支座方式连接成一个新梁 b
    b = b1.join(b2,"hinge")
    
    # 在梁 b 上施加载荷：M1 在 x=0 处，R1 在 x=0 处，-2 处，R2 在 x=l 处，-1 处，R3 在 x=4*l 处，-1 处，P 在 x=3*l 处，-1 处
    b.apply_load(M1, 0, -2)
    b.apply_load(R1, 0, -1)
    b.apply_load(R2, l, -1)
    b.apply_load(R3, 4*l, -1)
    b.apply_load(P, 3*l, -1)
    
    # 设置边界条件：在 x=0 处的角度为 0，位移为 0；在 x=l 处的位移为 0；在 x=4*l 处的位移为 0
    b.bc_slope = [(0, 0)]
    b.bc_deflection = [(0, 0), (l, 0), (4*l, 0)]
    
    # 解算反力：M1, R1, R2, R3
    b.solve_for_reaction_loads(M1, R1, R2, R3)
    
    # 断言梁 b 的反力为 {R3: -P/2, R2: P*Rational(-5, 4), M1: -P*l/4, R1: P*Rational(3, 4)}
    assert b.reaction_loads == {R3: -P/2, R2: P*Rational(-5, 4), M1: -P*l/4, R1: P*Rational(3, 4)}
    
    # 断言梁 b 在 x=3*l 处的斜率为 -7*P*l**2/(48*E*I)
    assert b.slope().subs(x, 3*l) == -7*P*l**2/(48*E*I)
    
    # 断言梁 b 在 x=2*l 处的挠度为 7*P*l**3/(24*E*I)
    assert b.deflection().subs(x, 2*l) == 7*P*l**3/(24*E*I)
    
    # 断言梁 b 在 x=3*l 处的挠度为 5*P*l**3/(16*E*I)
    assert b.deflection().subs(x, 3*l) == 5*P*l**3/(16*E*I)
    
    # 当具有相同二阶矩的梁连接时
    b1 = Beam(2, 500, 10)
    b2 = Beam(2, 500, 10)
    
    # 将两个梁 b1 和 b2 以固定支座方式连接成一个新梁 b
    b = b1.join(b2, "fixed")
    
    # 在梁 b 上施加载荷：M1 在 x=0 处，R1 在 x=0 处，-2 处，R2 在 x=1 处，-1 处，R3 在 x=4 处，-1 处，10 在 x=3 处，-1 处
    b.apply_load(M1, 0, -2)
    b.apply_load(R1, 0, -1)
    b.apply_load(R2, 1, -1)
    b.apply_load(R3, 4, -1)
    b.apply_load(10, 3, -1)
    
    # 设置边界条件：在 x=0 处的角度为 0，位移为 0；在 x=1 处的位移为 0；在 x=4 处的位移为 0
    b.bc_slope = [(0, 0)]
    b.bc_deflection = [(0, 0), (1, 0), (4, 0)]
    
    # 解算反力：M1, R1, R2, R3
    b.solve_for_reaction_loads(M1, R1, R2, R3)
    
    # 断言梁 b 的斜率为 -2*SingularityFunction(x, 0, 1)/5625 + SingularityFunction(x, 0, 2)/1875\
    #             - 133*SingularityFunction(x, 1, 2)/135000 + SingularityFunction(x, 3, 2)/1000\
    #             - 37*SingularityFunction(x, 4, 2)/67500
    assert b.slope() == -2*SingularityFunction(x, 0, 1)/5625 + SingularityFunction(x, 0, 2)/1875\
                - 133*SingularityFunction(x, 1, 2)/135000 + SingularityFunction(x, 3, 2)/1000\
                - 37*SingularityFunction(x, 4, 2)/67500
    
    # 断言梁 b 的挠度为 -SingularityFunction(x, 0, 2)/5625 + SingularityFunction(x, 0, 3)/5625\
    #                 - 133*SingularityFunction(x, 1, 3)/405000 + SingularityFunction(x, 3, 3)/3000\
    #                 - 37*SingularityFunction(x, 4, 3)/202500
    assert b.deflection() == -SingularityFunction(x, 0, 2)/5625 + SingularityFunction(x, 0, 3)/5625\
                    - 133*SingularityFunction(x, 1, 3)/405000 + SingularityFunction(x, 3, 3)/3000\
                    - 37*SingularityFunction(x, 4, 3)/202500


# 定义测试函数 `test_point_cflexure`
def
    # 应用支持反力在节点 10，并指定类型为 'pin'
    r10 = b.apply_support(10, type='pin')
    # 应用支持反力在节点 15，同时指定约束为 'fixed'，返回结果包括 r15 和 m15
    r15, m15 = b.apply_support(15, type='fixed')
    # 在节点 5 应用旋转铰支座
    b.apply_rotation_hinge(5)
    # 在节点 12 应用旋转铰支座
    b.apply_rotation_hinge(12)
    # 应用 -10 单位的加载在节点 5，沿 y 轴向下
    b.apply_load(-10, 5, -1)
    # 应用 -5 单位的加载在节点 10，沿 y 轴向下，同时应用于节点 15 沿 y 轴向下，加载区间为 0 到 15
    b.apply_load(-5, 10, 0, 15)
    # 使用指定的反力和弯矩反力解决结构
    b.solve_for_reaction_loads(r0, r10, r15, m15)
    # 确保函数抛出 NotImplementedError 异常，用于测试点弯曲方法
    with raises(NotImplementedError):
        b.point_cflexure()
# 定义一个测试函数，用于测试 Beam 类的 remove_load 方法
def test_remove_load():
    # 创建符号 E 和 I
    E = Symbol('E')
    I = Symbol('I')
    # 创建一个长度为 4 的梁对象 b，使用符号 E 和 I
    b = Beam(4, E, I)

    # 尝试移除梁上位置为 2 处的载荷，期望引发 ValueError 异常
    try:
        b.remove_load(2, 1, -1)
    # 如果没有载荷被应用到梁上，应当引发 ValueError 异常
    except ValueError:
        assert True
    else:
        assert False

    # 在梁上应用两个载荷
    b.apply_load(-3, 0, -2)
    b.apply_load(4, 2, -1)
    # 在梁的端点 2 处应用一个分布载荷，结束点为 3
    b.apply_load(-2, 2, 2, end=3)
    # 移除梁上端点 2 处到端点 3 处的分布载荷
    b.remove_load(-2, 2, 2, end=3)
    # 断言梁上的载荷总和
    assert b.load == -3*SingularityFunction(x, 0, -2) + 4*SingularityFunction(x, 2, -1)
    # 断言应用的载荷列表
    assert b.applied_loads == [(-3, 0, -2, None), (4, 2, -1, None)]

    # 尝试移除梁上位置为 (1, 2) 处的载荷，期望引发 ValueError 异常
    try:
        b.remove_load(1, 2, -1)
    # 如果从未在此位置应用过该大小的载荷，应当引发 ValueError 异常
    except ValueError:
        assert True
    else:
        assert False

    # 依次移除两个载荷
    b.remove_load(-3, 0, -2)
    b.remove_load(4, 2, -1)
    # 断言现在梁上的载荷总和为 0
    assert b.load == 0
    # 断言应用的载荷列表为空
    assert b.applied_loads == []


# 定义一个测试函数，用于测试 Beam 类的 apply_support 方法
def test_apply_support():
    # 创建符号 E 和 I
    E = Symbol('E')
    I = Symbol('I')

    # 创建一个长度为 4 的梁对象 b，使用符号 E 和 I
    b = Beam(4, E, I)
    # 在梁的端点 0 处应用固定支座
    b.apply_support(0, "cantilever")
    # 在梁的端点 4 处应用一个集中力
    b.apply_load(20, 4, -1)
    # 定义反力和弯矩符号
    M_0, R_0 = symbols('M_0, R_0')
    # 求解支反力和弯矩
    b.solve_for_reaction_loads(R_0, M_0)
    # 断言梁的斜率公式的简化形式
    assert simplify(b.slope()) == simplify((80*SingularityFunction(x, 0, 1) - 10*SingularityFunction(x, 0, 2)
                + 10*SingularityFunction(x, 4, 2))/(E*I))
    # 断言梁的挠度公式的简化形式
    assert simplify(b.deflection()) == simplify((40*SingularityFunction(x, 0, 2) - 10*SingularityFunction(x, 0, 3)/3
                + 10*SingularityFunction(x, 4, 3)/3)/(E*I))

    # 创建一个长度为 30 的梁对象 b，使用符号 E 和 I
    b = Beam(30, E, I)
    # 在梁的端点 10 处应用固定支座
    p0 = b.apply_support(10, "pin")
    # 在梁的端点 30 处应用滚动支座
    p1 = b.apply_support(30, "roller")
    # 在梁的端点 0 处应用一个集中力
    b.apply_load(-8, 0, -1)
    # 在梁的端点 30 处应用一个集中力
    b.apply_load(120, 30, -2)
    # 求解支反力
    b.solve_for_reaction_loads(p0, p1)
    # 断言梁的斜率公式
    assert b.slope() == (-4*SingularityFunction(x, 0, 2) + 3*SingularityFunction(x, 10, 2)
            + 120*SingularityFunction(x, 30, 1) + SingularityFunction(x, 30, 2) + Rational(4000, 3))/(E*I)
    # 断言梁的挠度公式
    assert b.deflection() == (x*Rational(4000, 3) - 4*SingularityFunction(x, 0, 3)/3 + SingularityFunction(x, 10, 3)
            + 60*SingularityFunction(x, 30, 2) + SingularityFunction(x, 30, 3)/3 - 12000)/(E*I)
    # 定义反力符号
    R_10 = Symbol('R_10')
    R_30 = Symbol('R_30')
    # 断言支反力字典中的值
    assert p0 == R_10
    assert b.reaction_loads == {R_10: 6, R_30: 2}
    assert b.reaction_loads[p0] == 6

    # 创建一个长度为 8 的梁对象 b，使用符号 E 和 I
    b = Beam(8, E, I)
    # 在梁的端点 0 处应用固定支座和弯矩
    p0, m0 = b.apply_support(0, "fixed")
    # 在梁的端点 8 处应用滚动支座
    p1 = b.apply_support(8, "roller")
    # 在梁的端点 0 到 8 的区间应用一个分布力
    b.apply_load(-5, 0, 0, 8)
    # 求解支反力和弯矩
    b.solve_for_reaction_loads(p0, m0, p1)
    # 定义反力和弯矩符号
    R_0 = Symbol('R_0')
    M_0 = Symbol('M_0')
    R_8 = Symbol('R_8')
    # 断言支反力字典中的值
    assert p0 == R_0
    assert m0 == M_0
    assert p1 == R_8
    assert b.reaction_loads == {R_0: 25, M_0: -40, R_8: 15}
    assert b.reaction_loads[m0] == -40

    # 创建符号 P 和 L，要求它们为正值
    P = Symbol('P', positive=True)
    L = Symbol('L', positive=True)
    # 创建一个长度为 L 的梁对象 b，使用符号 E 和 I
    b = Beam(L, E, I)
    # 在梁的端点 0 处应用固定支座
    b.apply_support(0, type='fixed')
    # 在梁的端点 L 处应用固定支座
    b.apply_support(L, type='fixed')
    # 在梁的端点 L/2 处应用一个集中力
    b.apply_load(-P, L/2, -1)
    # 定义反力和弯矩符号
    R_0, R_L, M_0, M_L = symbols('R_0, R_L, M_0, M_L')
    # 求解支反力和弯矩
    b.solve_for_reaction_loads(R_0, R_L, M_0, M_L)
    # 断言检查反应力字典是否符合预期值
    assert b.reaction_loads == {R_0: P/2, R_L: P/2, M_0: -L*P/8, M_L: L*P/8}
def test_apply_rotation_hinge():
    # 创建一个 Beam 对象，长为 15，高为 20，宽为 20
    b = Beam(15, 20, 20)
    # 在梁上施加固定支座在位置 0 处，并返回反力和弯矩
    r0, m0 = b.apply_support(0, type='fixed')
    # 在位置 10 处施加铰链支座，并返回反力
    r10 = b.apply_support(10, type='pin')
    # 在位置 15 处施加铰链支座，并返回反力
    r15 = b.apply_support(15, type='pin')
    # 在位置 7 处施加旋转铰链，并返回铰链反力
    p7 = b.apply_rotation_hinge(7)
    # 在位置 12 处施加旋转铰链，并返回铰链反力
    p12 = b.apply_rotation_hinge(12)
    # 在位置 7 处施加集中力 -10，方向向下
    b.apply_load(-10, 7, -1)
    # 在位置 10 和 15 处分别施加集中力 -2，方向向下
    b.apply_load(-2, 10, 0, 15)
    # 解算反力
    b.solve_for_reaction_loads(r0, m0, r10, r15)
    # 定义符号变量 R_0, M_0, R_10, R_15, P_7, P_12
    R_0, M_0, R_10, R_15, P_7, P_12 = symbols('R_0, M_0, R_10, R_15, P_7, P_12')
    # 预期反力结果字典
    expected_reactions = {R_0: 20/3, M_0: -140/3, R_10: 31/3, R_15: 3}
    # 预期旋转铰链结果字典
    expected_rotations = {P_7: 2281/2160, P_12: -5137/5184}
    # 反力符号列表
    reaction_symbols = [r0, m0, r10, r15]
    # 旋转铰链符号列表
    rotation_symbols = [p7, p12]
    # 公差
    tolerance = 1e-6
    # 断言：检查反力的数值是否与预期接近
    assert all(abs(b.reaction_loads[r] - expected_reactions[r]) < tolerance for r in reaction_symbols)
    # 断言：检查旋转铰链的数值是否与预期接近
    assert all(abs(b.rotation_jumps[r] - expected_rotations[r]) < tolerance for r in rotation_symbols)
    # 预期弯矩
    expected_bending_moment = (140 * SingularityFunction(x, 0, 0) / 3 - 20 * SingularityFunction(x, 0, 1) / 3
        - 11405 * SingularityFunction(x, 7, -1) / 27 + 10 * SingularityFunction(x, 7, 1)
        - 31 * SingularityFunction(x, 10, 1) / 3 + SingularityFunction(x, 10, 2)
        + 128425 * SingularityFunction(x, 12, -1) / 324 - 3 * SingularityFunction(x, 15, 1)
        - SingularityFunction(x, 15, 2))
    # 断言：检查计算的弯矩是否与预期相等
    assert b.bending_moment().expand() == expected_bending_moment.expand()
    # 预期斜率
    expected_slope = (-7*SingularityFunction(x, 0, 1)/60 + SingularityFunction(x, 0, 2)/120
        + 2281*SingularityFunction(x, 7, 0)/2160 - SingularityFunction(x, 7, 2)/80
        + 31*SingularityFunction(x, 10, 2)/2400 - SingularityFunction(x, 10, 3)/1200
        - 5137*SingularityFunction(x, 12, 0)/5184 + 3*SingularityFunction(x, 15, 2)/800
        + SingularityFunction(x, 15, 3)/1200)
    # 断言：检查计算的斜率是否与预期相等
    assert b.slope().expand() == expected_slope.expand()
    # 预期挠度
    expected_deflection = (-7 * SingularityFunction(x, 0, 2) / 120 + SingularityFunction(x, 0, 3) / 360
        + 2281 * SingularityFunction(x, 7, 1) / 2160 - SingularityFunction(x, 7, 3) / 240
        + 31 * SingularityFunction(x, 10, 3) / 7200 - SingularityFunction(x, 10, 4) / 4800
        - 5137 * SingularityFunction(x, 12, 1) / 5184 + SingularityFunction(x, 15, 3) / 800
        + SingularityFunction(x, 15, 4) / 4800)
    # 断言：检查计算的挠度是否与预期相等
    assert b.deflection().expand() == expected_deflection.expand()

    # 定义符号变量 E, I, F
    E = Symbol('E')
    I = Symbol('I')
    F = Symbol('F')
    # 创建另一个 Beam 对象，长为 10，弹性模量 E，惯性矩 I
    b = Beam(10, E, I)
    # 在位置 0 处施加固定支座，并返回反力和弯矩
    r0, m0 = b.apply_support(0, type="fixed")
    # 在位置 10 处施加铰链支座，并返回反力
    r10 = b.apply_support(10, type="pin")
    # 在位置 6 处施加旋转铰链
    b.apply_rotation_hinge(6)
    # 在位置 8 处施加集中力 F，方向向下
    b.apply_load(F, 8, -1)
    # 解算反力
    b.solve_for_reaction_loads(r0, m0, r10)
    # 断言：检查反力的计算结果是否符合预期
    assert b.reaction_loads == {R_0: -F/2, M_0: 3*F, R_10: -F/2}
    # 断言：检查弯矩的计算结果是否符合预期
    assert (b.bending_moment() == -3*F*SingularityFunction(x, 0, 0) + F*SingularityFunction(x, 0, 1)/2
            + 17*F*SingularityFunction(x, 6, -1) - F*SingularityFunction(x, 8, 1)
            + F*SingularityFunction(x, 10, 1)/2)
    # 计算预期的挠度，使用梁的单点函数和力的符号
    expected_deflection = -(-3*F*SingularityFunction(x, 0, 2)/2 + F*SingularityFunction(x, 0, 3)/12
            + 17*F*SingularityFunction(x, 6, 1) - F*SingularityFunction(x, 8, 3)/6
            + F*SingularityFunction(x, 10, 3)/12)/(E*I)
    # 断言梁的挠度展开后应与预期的挠度展开后相等
    assert b.deflection().expand() == expected_deflection.expand()

    # 定义符号变量 E, I, F, l1, l2, l3
    E = Symbol('E')
    I = Symbol('I')
    F = Symbol('F')
    l1 = Symbol('l1', positive=True)
    l2 = Symbol('l2', positive=True)
    l3 = Symbol('l3', positive=True)
    # 计算梁的总长度 L
    L = l1 + l2 + l3
    # 创建一个梁对象 b，使用给定的长度和材料参数
    b = Beam(L, E, I)
    # 应用固定支持于端点 0 处，类型为 "fixed"
    r0, m0 = b.apply_support(0, type="fixed")
    # 应用铰支持于端点 L 处，类型为 "pin"
    r1 = b.apply_support(L, type="pin")
    # 在距离 l1 处应用转动铰支持
    b.apply_rotation_hinge(l1)
    # 在距离 l1 + l2 处应用力 F，方向向下
    b.apply_load(F, l1+l2, -1)
    # 解算支反力
    b.solve_for_reaction_loads(r0, m0, r1)
    # 断言计算得到的支反力与预期值相等
    assert b.reaction_loads[r0] == -F*l3/(l2 + l3)
    assert b.reaction_loads[m0] == F*l1*l3/(l2 + l3)
    assert b.reaction_loads[r1] == -F*l2/(l2 + l3)
    # 计算预期的弯矩，使用梁的单点函数和力的符号
    expected_bending_moment = (-F*l1*l3*SingularityFunction(x, 0, 0)/(l2 + l3)
            + F*l2*SingularityFunction(x, l1 + l2 + l3, 1)/(l2 + l3)
            + F*l3*SingularityFunction(x, 0, 1)/(l2 + l3) - F*SingularityFunction(x, l1 + l2, 1)
            - (-2*F*l1**3*l3 - 3*F*l1**2*l2*l3 - 3*F*l1**2*l3**2 + F*l2**3*l3 + 3*F*l2**2*l3**2 + 2*F*l2*l3**3)
            *SingularityFunction(x, l1, -1)/(6*l2**2 + 12*l2*l3 + 6*l3**2))
    # 断言梁的弯矩简化后应与预期的弯矩简化后相等
    assert simplify(b.bending_moment().expand()) == simplify(expected_bending_moment.expand())


这些注释为给定的代码块中的每个语句提供了解释和说明，帮助理解每个步骤的目的和操作。
def test_apply_sliding_hinge():
    # 创建一个 Beam 对象，长度为 13，Elastic 模量为 20，惯性矩 I 为 20
    b = Beam(13, 20, 20)
    
    # 在梁的端点 0 应用固定支座，返回反力和弯矩
    r0, m0 = b.apply_support(0, type="fixed")
    
    # 在距离梁端 8 处应用滑动铰链
    w8 = b.apply_sliding_hinge(8)
    
    # 在梁的端点 13 应用铰链支座，返回反力
    r13 = b.apply_support(13, type="pin")
    
    # 在距离梁端 5 处施加 -10 单位载荷
    b.apply_load(-10, 5, -1)
    
    # 解算梁的反力和弯矩分布，给定初始反力和弯矩
    b.solve_for_reaction_loads(r0, m0, r13)
    
    # 声明符号变量 R_0, M_0, R_13, W_8
    R_0, M_0, R_13, W_8 = symbols('R_0, M_0, R_13, W_8')
    
    # 断言梁的反力为 {R_0: 10, M_0: -50, R_13: 0}
    assert b.reaction_loads == {R_0: 10, M_0: -50, R_13: 0}
    
    # 设置允许的误差范围
    tolerance = 1e-6
    
    # 断言距离梁端 8 处的挠度跳跃为 85/24，误差在允许的范围内
    assert abs(b.deflection_jumps[w8] - 85/24) < tolerance
    
    # 断言梁的弯矩为 50*SingularityFunction(x, 0, 0) - 10*SingularityFunction(x, 0, 1)
    # + 10*SingularityFunction(x, 5, 1) - 4250*SingularityFunction(x, 8, -2)/3
    assert (b.bending_moment() == 50*SingularityFunction(x, 0, 0) - 10*SingularityFunction(x, 0, 1)
            + 10*SingularityFunction(x, 5, 1) - 4250*SingularityFunction(x, 8, -2)/3)
    
    # 断言梁的挠度为 -SingularityFunction(x, 0, 2)/16 + SingularityFunction(x, 0, 3)/240
    # - SingularityFunction(x, 5, 3)/240 + 85*SingularityFunction(x, 8, 0)/24
    assert (b.deflection() == -SingularityFunction(x, 0, 2)/16 + SingularityFunction(x, 0, 3)/240
            - SingularityFunction(x, 5, 3)/240 + 85*SingularityFunction(x, 8, 0)/24)
    
    # 声明符号变量 E, I, I2
    E = Symbol('E')
    I = Symbol('I')
    I2 = Symbol('I2')
    
    # 创建两个长度分别为 5 和 8 的 Beam 对象，Elastic 模量为 E，惯性矩分别为 I 和 I2
    b1 = Beam(5, E, I)
    b2 = Beam(8, E, I2)
    
    # 将两个梁对象拼接成一个新的梁对象
    b = b1.join(b2)
    
    # 在梁的端点 0 应用固定支座，返回反力和弯矩
    r0, m0 = b.apply_support(0, type="fixed")
    
    # 在距离梁端 8 处应用滑动铰链
    b.apply_sliding_hinge(8)
    
    # 在梁的端点 13 应用铰链支座，返回反力
    r13 = b.apply_support(13, type="pin")
    
    # 在距离梁端 5 处施加 -10 单位载荷
    b.apply_load(-10, 5, -1)
    
    # 解算梁的反力和弯矩分布，给定初始反力和弯矩
    b.solve_for_reaction_loads(r0, m0, r13)
    
    # 声明符号变量 W_8
    W_8 = Symbol('W_8')
    
    # 断言梁的挠度跳跃为 {W_8: 4250/(3*E*I2)}
    assert b.deflection_jumps == {W_8: 4250/(3*E*I2)}
    
    # 声明符号变量 E, I, q, l1, l2, l3
    E = Symbol('E')
    I = Symbol('I')
    q = Symbol('q')
    l1 = Symbol('l1', positive=True)
    l2 = Symbol('l2', positive=True)
    l3 = Symbol('l3', positive=True)
    
    # 计算总长度 L
    L = l1 + l2 + l3
    
    # 创建一个长度为 L 的 Beam 对象，Elastic 模量为 E，惯性矩为 I
    b = Beam(L, E, I)
    
    # 在梁的端点 0 应用铰链支座，返回反力
    r0 = b.apply_support(0, type="pin")
    
    # 在梁的端点 l1 处应用铰链支座，返回反力
    r3 = b.apply_support(l1, type="pin")
    
    # 在距离梁端 l1+l2 处应用滑动铰链
    b.apply_sliding_hinge(l1 + l2)
    
    # 在梁的端点 L 处应用铰链支座，返回反力
    r10 = b.apply_support(L, type="pin")
    
    # 在长度为 l1 的梁上均匀施加单位载荷 q
    b.apply_load(q, 0, 0, l1)
    
    # 解算梁的反力分布，给定初始反力
    b.solve_for_reaction_loads(r0, r3, r10)
    
    # 断言梁的弯矩为 l1*q*SingularityFunction(x, 0, 1)/2 + l1*q*SingularityFunction(x, l1, 1)/2
    # - q*SingularityFunction(x, 0, 2)/2 + q*SingularityFunction(x, l1, 2)/2
    # + (-l1**3*l2*q/24 - l1**3*l3*q/24)*SingularityFunction(x, l1 + l2, -2)
    assert (b.bending_moment() == l1*q*SingularityFunction(x, 0, 1)/2 + l1*q*SingularityFunction(x, l1, 1)/2
            - q*SingularityFunction(x, 0, 2)/2 + q*SingularityFunction(x, l1, 2)/2
            + (-l1**3*l2*q/24 - l1**3*l3*q/24)*SingularityFunction(x, l1 + l2, -2))
    
    # 断言梁的挠度为 (l1**3*q*x/24 - l1*q*SingularityFunction(x, 0, 3)/12
    # - l1*q*SingularityFunction(x, l1, 3)/12 + q*SingularityFunction(x, 0, 4)/24
    # - q*SingularityFunction(x, l1, 4)/24
    # + (l1**3*l2*q/24 + l1**3*l3*q/24)*SingularityFunction(x, l1 + l2, 0))/(E*I)
    assert b.deflection() == (l1**3*q*x/24 - l1*q*SingularityFunction(x, 0, 3)/12
                             - l1*q*SingularityFunction(x, l1, 3)/12 + q*SingularityFunction(x, 0, 4)/24
                             - q*SingularityFunction
    # 断言语句，用于检查 max_shear[1] - (l*Abs(P)/2) 是否等于 0
    assert simplify(max_shear[1] - (l*Abs(P)/2)) == 0
# 定义一个测试函数，用于测试梁的最大弯矩计算函数
def test_max_bmoment():
    # 定义符号变量 E 和 I
    E = Symbol('E')
    I = Symbol('I')
    # 定义正数符号变量 l 和 P
    l, P = symbols('l, P', positive=True)

    # 创建一个梁对象 b，其长度为 l，弹性模量为 E，惯性矩为 I
    b = Beam(l, E, I)
    # 定义反力 R1 和 R2
    R1, R2 = symbols('R1, R2')
    # 在梁的起点施加一个集中力 R1
    b.apply_load(R1, 0, -1)
    # 在梁的端点施加一个集中力 R2
    b.apply_load(R2, l, -1)
    # 在梁的中点施加一个集中力 P
    b.apply_load(P, l/2, -1)
    # 解算梁的支反力
    b.solve_for_reaction_loads(R1, R2)
    # 获取梁的反力载荷
    b.reaction_loads
    # 断言最大弯矩的计算结果
    assert b.max_bmoment() == (l/2, P*l/4)

    # 创建一个新的梁对象 b，其长度为 l，弹性模量为 E，惯性矩为 I
    b = Beam(l, E, I)
    # 再次定义反力 R1 和 R2
    R1, R2 = symbols('R1, R2')
    # 在梁的起点施加一个集中力 R1
    b.apply_load(R1, 0, -1)
    # 在梁的端点施加一个集中力 R2
    b.apply_load(R2, l, -1)
    # 在梁的起点到端点施加一个均布力 P
    b.apply_load(P, 0, 0, end=l)
    # 解算梁的支反力
    b.solve_for_reaction_loads(R1, R2)
    # 断言最大弯矩的计算结果
    assert b.max_bmoment() == (l/2, P*l**2/8)


# 定义一个测试函数，用于测试梁的最大挠度计算函数
def test_max_deflection():
    # 定义符号变量 E, I, l, F，均为正数
    E, I, l, F = symbols('E, I, l, F', positive=True)
    # 创建一个梁对象 b，其长度为 l，弹性模量为 E，惯性矩为 I
    b = Beam(l, E, I)
    # 设定梁的挠曲边界条件为两端挠曲为零
    b.bc_deflection = [(0, 0),(l, 0)]
    # 设定梁的挠曲率边界条件为两端挠曲率为零
    b.bc_slope = [(0, 0),(l, 0)]
    # 在梁的起点施加一个集中力 F/2
    b.apply_load(F/2, 0, -1)
    # 在梁的起点施加一个反向的端点力 F*l/8
    b.apply_load(-F*l/8, 0, -2)
    # 在梁的端点施加一个集中力 F/2
    b.apply_load(F/2, l, -1)
    # 在梁的端点施加一个反向的端点力 F*l/8
    b.apply_load(F*l/8, l, -2)
    # 在梁的中点施加一个集中力 -F
    b.apply_load(-F, l/2, -1)
    # 断言最大挠度的计算结果
    assert b.max_deflection() == (l/2, F*l**3/(192*E*I))


# 定义一个测试函数，用于测试梁的 ILD 反力解算函数
def test_solve_for_ild_reactions():
    # 定义符号变量 E 和 I
    E = Symbol('E')
    I = Symbol('I')
    # 创建一个长度为 10 的梁对象 b，弹性模量为 E，惯性矩为 I
    b = Beam(10, E, I)
    # 在梁的起点施加一个铰支
    b.apply_support(0, type="pin")
    # 在梁的端点施加一个铰支
    b.apply_support(10, type="pin")
    # 定义反力 R_0 和 R_10
    R_0, R_10 = symbols('R_0, R_10')
    # 解算梁的 ILD 反力
    b.solve_for_ild_reactions(1, R_0, R_10)
    # 获取 ILD 变量 a
    a = b.ild_variable
    # 断言 ILD 反力的计算结果
    assert b.ild_reactions == {R_0: -SingularityFunction(a, 0, 0) + SingularityFunction(a, 0, 1)/10
                                    - SingularityFunction(a, 10, 1)/10,
                               R_10: -SingularityFunction(a, 0, 1)/10 + SingularityFunction(a, 10, 0)
                                     + SingularityFunction(a, 10, 1)/10}

    # 创建一个新的梁对象 b，其长度为 L，弹性模量为 E，惯性矩为 I
    E = Symbol('E')
    I = Symbol('I')
    F = Symbol('F')
    L = Symbol('L', positive=True)
    b = Beam(L, E, I)
    # 在梁的端点施加一个固支
    b.apply_support(L, type="fixed")
    # 在梁的起点施加一个集中力 F
    b.apply_load(F, 0, -1)
    # 定义反力 R_L 和 M_L
    R_L, M_L = symbols('R_L, M_L')
    # 解算梁的 ILD 反力
    b.solve_for_ild_reactions(F, R_L, M_L)
    # 获取 ILD 变量 a
    a = b.ild_variable
    # 断言 ILD 反力的计算结果
    assert b.ild_reactions == {R_L: -F*SingularityFunction(a, 0, 0) + F*SingularityFunction(a, L, 0) - F,
                               M_L: -F*L*SingularityFunction(a, 0, 0) - F*L + F*SingularityFunction(a, 0, 1)
                                    - F*SingularityFunction(a, L, 1)}

    # 创建一个长度为 20 的梁对象 b，弹性模量为 E，惯性矩为 I
    E = Symbol('E')
    I = Symbol('I')
    b = Beam(20, E, I)
    # 在梁的起点施加一个铰支，并获取其反力
    r0 = b.apply_support(0, type="pin")
    # 在梁的中点施加一个铰支，并获取其反力
    r5 = b.apply_support(5, type="pin")
    # 在梁的端点施加一个铰支，并获取其反力
    r10 = b.apply_support(10, type="pin")
    # 在梁的端点施加一个固支，并获取其反力
    r20, m20 = b.apply_support(20, type="fixed")
    # 解算梁的 ILD 反力
    b.solve_for_ild_reactions(1, r0, r5, r10, r20, m20)
    # 获取 ILD 变量 a
    a = b.ild_variable
    # 断言 ILD 反力的计算结果
    assert b.ild_reactions[r0].subs(a, 4) == -Rational(59, 475)
    assert b.ild_reactions[r5].subs(a, 4) == -Rational(2296, 2375)
    assert b.ild_reactions[r10].subs(a, 4) == Rational(243, 2375)
    assert b.ild_reactions[r20].subs(a, 12) == -Rational(83, 475)
    # 计算支持反力在位置 0 处的反力
    r0 = b.apply_support(0, type="pin")
    # 计算支持反力在位置 L1 + L2 处的反力
    rL = b.apply_support(L1 + L2, type="pin")
    # 解算梁的 ILD 反力
    b.solve_for_ild_reactions(F, r0, rL)
    # 解算梁的 ILD 剪力
    b.solve_for_ild_shear(L1, F, r0, rL)
    # 获取 ILD 变量 a 的值
    a = b.ild_variable
    # 预期的 ILD 剪力结果，由多个 SingularityFunction 表达式组成
    expected_shear = (-F*L1*SingularityFunction(a, 0, 0)/(L1 + L2) - F*L2*SingularityFunction(a, 0, 0)/(L1 + L2)
                      - F*SingularityFunction(-a, 0, 0) + F*SingularityFunction(a, L1 + L2, 0) + F
                      + F*SingularityFunction(a, 0, 1)/(L1 + L2) - F*SingularityFunction(a, L1 + L2, 1)/(L1 + L2)
                      - (-F*L1*SingularityFunction(a, 0, 0)/(L1 + L2) + F*L1*SingularityFunction(a, L1 + L2, 0)/(L1 + L2)
                         - F*L2*SingularityFunction(a, 0, 0)/(L1 + L2) + F*L2*SingularityFunction(a, L1 + L2, 0)/(L1 + L2)
                         + 2*F)*SingularityFunction(a, L1, 0))
    # 断言 ILD 剪力的展开式等于预期的 ILD 剪力
    assert b.ild_shear.expand() == expected_shear.expand()

    # 创建符号 E 和 I
    E = Symbol('E')
    I = Symbol('I')
    # 创建长度为 20、弹性模量为 E、惯性矩 I 的梁对象
    b = Beam(20, E, I)
    # 计算支持反力在位置 0 处的反力
    r0 = b.apply_support(0, type="pin")
    # 计算支持反力在位置 5 处的反力
    r5 = b.apply_support(5, type="pin")
    # 计算支持反力在位置 10 处的反力
    r10 = b.apply_support(10, type="pin")
    # 计算支持反力和弯矩在位置 20 处的反力和弯矩
    r20, m20 = b.apply_support(20, type="fixed")
    # 解算梁的 ILD 反力
    b.solve_for_ild_reactions(1, r0, r5, r10, r20, m20)
    # 解算梁的 ILD 剪力
    b.solve_for_ild_shear(6, 1, r0, r5, r10, r20, m20)
    # 获取 ILD 变量 a 的值
    a = b.ild_variable
    # 断言当 a = 12 时，ILD 剪力的代入结果等于有理数 96/475
    assert b.ild_shear.subs(a, 12) == Rational(96, 475)
    # 断言当 a = 4 时，ILD 剪力的代入结果等于负有理数 216/2375
    assert b.ild_shear.subs(a, 4) == -Rational(216, 2375)
# 定义一个用于测试梁的 ILD（影响线模型法）解算的函数
def test_solve_for_ild_moment():
    # 定义符号变量 E, I, F
    E = Symbol('E')
    I = Symbol('I')
    F = Symbol('F')
    # 定义正数符号变量 L1 和 L2
    L1 = Symbol('L1', positive=True)
    L2 = Symbol('L2', positive=True)
    
    # 创建梁对象 b，长度为 L1 + L2，使用给定的 E 和 I
    b = Beam(L1 + L2, E, I)
    
    # 在梁的起点处施加固定支座
    r0 = b.apply_support(0, type="pin")
    # 在梁的末端处施加固定支座
    rL = b.apply_support(L1 + L2, type="pin")
    
    # 获取 ILD（影响线模型法）的变量
    a = b.ild_variable
    
    # 解算 ILD 反力
    b.solve_for_ild_reactions(F, r0, rL)
    # 解算 ILD 弯矩
    b.solve_for_ild_moment(L1, F, r0, rL)
    
    # 断言计算得到的 ILD 弯矩在给定替换下的值
    assert b.ild_moment.subs(a, 3).subs(L1, 5).subs(L2, 5) == -3*F/2
    
    # 重新定义符号变量 E 和 I
    E = Symbol('E')
    I = Symbol('I')
    
    # 创建另一个梁对象 b，长度为 20，使用给定的 E 和 I
    b = Beam(20, E, I)
    
    # 在梁的不同位置施加固定支座
    r0 = b.apply_support(0, type="pin")
    r5 = b.apply_support(5, type="pin")
    r10 = b.apply_support(10, type="pin")
    r20, m20 = b.apply_support(20, type="fixed")
    
    # 解算 ILD 反力和弯矩
    b.solve_for_ild_reactions(1, r0, r5, r10, r20, m20)
    b.solve_for_ild_moment(5, 1, r0, r5, r10, r20, m20)
    
    # 断言计算得到的 ILD 弯矩在给定替换下的值
    assert b.ild_moment.subs(a, 12) == -Rational(96, 475)
    assert b.ild_moment.subs(a, 4) == Rational(36, 95)

# 定义一个测试梁在旋转铰链作用下的 ILD 解算的函数
def test_ild_with_rotation_hinge():
    # 定义符号变量 E, I, F
    E = Symbol('E')
    I = Symbol('I')
    F = Symbol('F')
    # 定义正数符号变量 L1, L2, L3
    L1 = Symbol('L1', positive=True)
    L2 = Symbol('L2', positive=True)
    L3 = Symbol('L3', positive=True)
    
    # 创建梁对象 b，长度为 L1 + L2 + L3，使用给定的 E 和 I
    b = Beam(L1 + L2 + L3, E, I)
    
    # 在梁的不同位置施加固定支座
    r0 = b.apply_support(0, type="pin")
    r1 = b.apply_support(L1 + L2, type="pin")
    r2 = b.apply_support(L1 + L2 + L3, type="pin")
    
    # 在指定位置添加旋转铰链
    b.apply_rotation_hinge(L1 + L2)
    
    # 解算 ILD 反力
    b.solve_for_ild_reactions(F, r0, r1, r2)
    # 获取 ILD（影响线模型法）的变量
    a = b.ild_variable
    
    # 断言计算得到的 ILD 反力在给定替换下的值
    assert b.ild_reactions[r0].subs(a, 4).subs(L1, 5).subs(L2, 5).subs(L3, 10) == -3*F/5
    assert b.ild_reactions[r0].subs(a, -10).subs(L1, 5).subs(L2, 5).subs(L3, 10) == 0
    assert b.ild_reactions[r0].subs(a, 25).subs(L1, 5).subs(L2, 5).subs(L3, 10) == 0
    assert b.ild_reactions[r1].subs(a, 4).subs(L1, 5).subs(L2, 5).subs(L3, 10) == -2*F/5
    assert b.ild_reactions[r2].subs(a, 18).subs(L1, 5).subs(L2, 5).subs(L3, 10) == -4*F/5
    
    # 解算 ILD 剪力
    b.solve_for_ild_shear(L1, F, r0, r1, r2)
    # 断言计算得到的 ILD 剪力在给定替换下的值
    assert b.ild_shear.subs(a, 7).subs(L1, 5).subs(L2, 5).subs(L3, 10) == -3*F/10
    assert b.ild_shear.subs(a, 70).subs(L1, 5).subs(L2, 5).subs(L3, 10) == 0
    
    # 解算 ILD 弯矩
    b.solve_for_ild_moment(L1, F, r0, r1, r2)
    # 断言计算得到的 ILD 弯矩在给定替换下的值
    assert b.ild_moment.subs(a, 1).subs(L1, 5).subs(L2, 5).subs(L3, 10) == -F/2
    assert b.ild_moment.subs(a, 8).subs(L1, 5).subs(L2, 5).subs(L3, 10) == -F

# 定义一个测试梁在滑动铰链作用下的 ILD 解算的函数
def test_ild_with_sliding_hinge():
    # 创建梁对象 b，长度为 13，使用给定的 E 和 I
    b = Beam(13, 200, 200)
    
    # 在梁的不同位置施加固定支座和滑动铰链
    r0 = b.apply_support(0, type="pin")
    r6 = b.apply_support(6, type="pin")
    r13, m13 = b.apply_support(13, type="fixed")
    w3 = b.apply_sliding_hinge(3)
    
    # 解算 ILD 反力和弯矩
    b.solve_for_ild_reactions(1, r0, r6, r13, m13)
    # 获取 ILD（影响线模型法）的变量
    a = b.ild_variable
    
    # 断言计算得到的 ILD 反力在给定替换下的值
    assert b.ild_reactions[r0].subs(a, 3) == -1
    assert b.ild_reactions[r6].subs(a, 3) == Rational(9, 14)
    assert b.ild_reactions[r13].subs(a, 9) == -Rational(207, 343)
    assert b.ild_reactions[m13].subs(a, 9) == -Rational(60, 49)
    assert b.ild_reactions[m13].subs(a, 15) == 0
    assert b.ild_reactions[m13].subs(a, -3) == 0
    
    # 断言计算得到的 ILD 挠度跳跃在给定替换下的值
    assert b.ild_deflection_jumps[w3].subs(a, 9) == -Rational(9, 35000)
    # 使用对象 b 调用 solve_for_ild_shear 方法，计算 ILD 剪力在特定条件下的值
    b.solve_for_ild_shear(7, 1, r0, r6, r13, m13)
    # 使用断言检查 ILD 剪力在 a=8 时的值是否等于 -200/343
    assert b.ild_shear.subs(a, 8) == -Rational(200, 343)
    # 使用对象 b 调用 solve_for_ild_moment 方法，计算 ILD 弯矩在特定条件下的值
    b.solve_for_ild_moment(8, 1, r0, r6, r13, m13)
    # 使用断言检查 ILD 弯矩在 a=3 时的值是否等于 -12/7
    assert b.ild_moment.subs(a, 3) == -Rational(12, 7)
# 定义一个测试函数，用于测试 Beam3D 类的功能
def test_Beam3D():
    # 定义符号变量 l, E, G, I, A
    l, E, G, I, A = symbols('l, E, G, I, A')
    # 定义符号变量 R1, R2, R3, R4
    R1, R2, R3, R4 = symbols('R1, R2, R3, R4')

    # 创建一个三维梁对象 b，使用给定的参数 l, E, G, I, A
    b = Beam3D(l, E, G, I, A)
    # 定义符号变量 m, q
    m, q = symbols('m, q')
    # 在梁对象 b 上施加一个沿 y 轴方向的加载 q
    b.apply_load(q, 0, 0, dir="y")
    # 在梁对象 b 上施加一个沿 z 轴方向的弯矩加载 m
    b.apply_moment_load(m, 0, 0, dir="z")
    # 设置梁对象 b 的边界条件为角度斜率为零
    b.bc_slope = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    # 设置梁对象 b 的边界条件为挠度为零
    b.bc_deflection = [(0, [0, 0, 0]), (l, [0, 0, 0])]
    # 解算梁对象 b 的角度斜率和挠度
    b.solve_slope_deflection()

    # 断言极点矩为 2*I
    assert b.polar_moment() == 2*I
    # 断言横向力为 [0, -q*x, 0]
    assert b.shear_force() == [0, -q*x, 0]
    # 断言剪切应力为 [0, -q*x/A, 0]
    assert b.shear_stress() == [0, -q*x/A, 0]
    # 断言轴向应力为 0
    assert b.axial_stress() == 0
    # 断言弯矩为 [0, 0, -m*x + q*x**2/2]
    assert b.bending_moment() == [0, 0, -m*x + q*x**2/2]
    # 计算预期的挠度，并断言与计算得到的挠度相等
    expected_deflection = (x*(A*G*q*x**3/4 + A*G*x**2*(-l*(A*G*l*(l*q - 2*m) +
        12*E*I*q)/(A*G*l**2 + 12*E*I)/2 - m) + 3*E*I*l*(A*G*l*(l*q - 2*m) +
        12*E*I*q)/(A*G*l**2 + 12*E*I) + x*(-A*G*l**2*q/2 +
        3*A*G*l**2*(A*G*l*(l*q - 2*m) + 12*E*I*q)/(A*G*l**2 + 12*E*I)/4 +
        A*G*l*m*Rational(3, 2) - 3*E*I*q))/(6*A*E*G*I))
    dx, dy, dz = b.deflection()
    assert dx == dz == 0
    assert simplify(dy - expected_deflection) == 0

    # 创建另一个三维梁对象 b2，使用给定的参数 30, E, G, I, A, x
    b2 = Beam3D(30, E, G, I, A, x)
    # 在梁对象 b2 上施加一个沿 y 轴方向的加载 50
    b2.apply_load(50, start=0, order=0, dir="y")
    # 设置梁对象 b2 的边界条件为挠度为零
    b2.bc_deflection = [(0, [0, 0, 0]), (30, [0, 0, 0])]
    # 在梁对象 b2 上施加一个在 x=0 处的力 R1
    b2.apply_load(R1, start=0, order=-1, dir="y")
    # 在梁对象 b2 上施加一个在 x=30 处的力 R2
    b2.apply_load(R2, start=30, order=-1, dir="y")
    # 解算梁对象 b2 的支反力 R1 和 R2
    b2.solve_for_reaction_loads(R1, R2)
    # 断言计算得到的支反力与预期值相等
    assert b2.reaction_loads == {R1: -750, R2: -750}

    # 解算梁对象 b2 的角度斜率和挠度
    b2.solve_slope_deflection()
    # 断言梁对象 b2 的斜率
    assert b2.slope() == [0, 0, 25*x**3/(3*E*I) - 375*x**2/(E*I) + 3750*x/(E*I)]
    # 计算预期的挠度，并断言与计算得到的挠度相等
    expected_deflection = 25*x**4/(12*E*I) - 125*x**3/(E*I) + 1875*x**2/(E*I) - \
        25*x**2/(A*G) + 750*x/(A*G)
    dx, dy, dz = b2.deflection()
    assert dx == dz == 0
    assert dy == expected_deflection

    # 创建另一个三维梁对象 b3，使用给定的参数 30, E, G, I, A, x
    b3 = Beam3D(30, E, G, I, A, x)
    # 在梁对象 b3 上施加一个沿 y 轴方向的加载 8
    b3.apply_load(8, start=0, order=0, dir="y")
    # 在梁对象 b3 上施加一个沿 z 轴方向的加载 9*x
    b3.apply_load(9*x, start=0, order=0, dir="z")
    # 在梁对象 b3 上施加一个在 x=0 处的力 R1
    b3.apply_load(R1, start=0, order=-1, dir="y")
    # 在梁对象 b3 上施加一个在 x=30 处的力 R2
    b3.apply_load(R2, start=30, order=-1, dir="y")
    # 在梁对象 b3 上施加一个在 x=0 处的力 R3
    b3.apply_load(R3, start=0, order=-1, dir="z")
    # 在梁对象 b3 上施加一个在 x=30 处的力 R4
    b3.apply_load(R4, start=30, order=-1, dir="z")
    # 解算梁对象 b3 的支反力 R1, R2, R3 和 R4
    b3.solve_for_reaction_loads(R1, R2, R3, R4)
    # 断言计算得到的支反力与预期值相等
    assert b3.reaction_loads == {R1: -120, R2: -120, R3: -1350, R4: -2700}


# 定义一个测试函数，用于测试 Beam3D 类的极点矩计算方法
def test_polar_moment_Beam3D():
    # 定义符号变量 l, E, G, A, I1, I2
    l, E, G, A, I1, I2 = symbols('l, E, G, A, I1, I2')
    # 创建一个三维梁对象 b，使用给定的参数 l, E, G, [I1, I2], A
    b = Beam3D(l, E, G, [I1, I2], A)
    # 断言计算得到的极点矩与预期值相等
    assert b.polar_moment() == I1 + I2


# 定义一个测试函数，用于测试 Beam 类的梁的加载情况
def test_parabolic_loads():
    # 定义符号变量 E, I, L
    E, I, L = symbols('E, I, L
    # 创建一个梁对象，长度为 2*L，杨氏模量为 E，惯性矩为 I
    beam = Beam(2*L, E, I)

    # 在梁的边界条件列表中添加位移边界条件 (0, 0)
    beam.bc_deflection.append((0, 0))
    # 在梁的边界条件列表中添加斜率边界条件 (0, 0)
    beam.bc_slope.append((0, 0))
    # 在梁上施加一个力载荷 R，作用位置为 0，加载模式为 -1
    beam.apply_load(R, 0, -1)
    # 在梁上施加一个弯矩载荷 M，作用位置为 0，加载模式为 -2
    beam.apply_load(M, 0, -2)

    # 在梁上施加一个从 x=0 到 x=L 的抛物线分布载荷
    beam.apply_load(1, 0, 2, end=L)

    # 求解反力载荷
    beam.solve_for_reaction_loads(R, M)

    # 断言：反力载荷 R 应该等于之前示例中的值 -L**3/3
    assert beam.reaction_loads[R] == -L**3/3

    # 检查常量载荷情况
    beam = Beam(2*L, E, I)
    beam.apply_load(P, 0, 0, end=L)
    # 使用指定的替换变量检查载荷，此处检查 x=5 时的载荷是否为 40
    loading = beam.load.xreplace({L: 10, E: 20, I: 30, P: 40})
    assert loading.xreplace({x: 5}) == 40
    # 检查 x=15 时的载荷是否为 0
    assert loading.xreplace({x: 15}) == 0

    # 检查斜坡载荷情况
    beam = Beam(2*L, E, I)
    beam.apply_load(P, 0, 1, end=L)
    # 断言：梁的载荷应为 (P*x - P*SingularityFunction(x, L, 1) - P*L*SingularityFunction(x, L, 0))
    assert beam.load == (P*SingularityFunction(x, 0, 1) -
                         P*SingularityFunction(x, L, 1) -
                         P*L*SingularityFunction(x, L, 0))

    # 检查高阶载荷情况：x**8 载荷从 x=0 到 x=L
    beam = Beam(2*L, E, I)
    beam.apply_load(P, 0, 8, end=L)
    # 使用指定的替换变量检查载荷，此处检查 x=5 时的载荷是否为 40*5**8
    loading = beam.load.xreplace({L: 10, E: 20, I: 30, P: 40})
    assert loading.xreplace({x: 5}) == 40*5**8
    # 检查 x=15 时的载荷是否为 0
    assert loading.xreplace({x: 15}) == 0
# 定义一个测试函数，用于测试梁的截面性质和加载情况
def test_cross_section():
    # 定义符号变量
    I = Symbol('I')  # 惯性矩
    l = Symbol('l')  # 长度
    E = Symbol('E')  # 弹性模量
    C3, C4 = symbols('C3, C4')  # 符号变量 C3 和 C4
    a, c, g, h, r, n = symbols('a, c, g, h, r, n')  # 符号变量 a, c, g, h, r, n

    # 测试设置二阶矩和截面设置器
    b0 = Beam(l, E, I)  # 创建一个梁对象
    assert b0.second_moment == I  # 断言：二阶矩是否为 I
    assert b0.cross_section == None  # 断言：截面是否为 None
    b0.cross_section = Circle((0, 0), 5)  # 设置圆形截面
    assert b0.second_moment == pi*Rational(625, 4)  # 断言：二阶矩是否为 pi*625/4
    assert b0.cross_section == Circle((0, 0), 5)  # 断言：截面是否为 Circle((0, 0), 5)
    b0.second_moment = 2*n - 6  # 设置二阶矩为 2*n - 6
    assert b0.second_moment == 2*n - 6  # 断言：二阶矩是否为 2*n - 6
    assert b0.cross_section == None  # 断言：截面是否为 None
    with raises(ValueError):
        b0.second_moment = Circle((0, 0), 5)  # 期望引发 ValueError 的操作：设置二阶矩为 Circle((0, 0), 5)

    # 带有圆形截面的梁
    b1 = Beam(50, E, Circle((0, 0), r))  # 创建一个梁对象，圆形截面
    assert b1.cross_section == Circle((0, 0), r)  # 断言：截面是否为 Circle((0, 0), r)
    assert b1.second_moment == pi*r*Abs(r)**3/4  # 断言：二阶矩是否为 pi*r*Abs(r)**3/4

    # 应用加载到梁上
    b1.apply_load(-10, 0, -1)
    b1.apply_load(R1, 5, -1)
    b1.apply_load(R2, 50, -1)
    b1.apply_load(90, 45, -2)
    b1.solve_for_reaction_loads(R1, R2)  # 求解支反力
    assert b1.load == (-10*SingularityFunction(x, 0, -1) + 82*SingularityFunction(x, 5, -1)/S(9)
                         + 90*SingularityFunction(x, 45, -2) + 8*SingularityFunction(x, 50, -1)/9)
    assert b1.bending_moment() == (10*SingularityFunction(x, 0, 1) - 82*SingularityFunction(x, 5, 1)/9
                                     - 90*SingularityFunction(x, 45, 0) - 8*SingularityFunction(x, 50, 1)/9)
    q = (-5*SingularityFunction(x, 0, 2) + 41*SingularityFunction(x, 5, 2)/S(9)
           + 90*SingularityFunction(x, 45, 1) + 4*SingularityFunction(x, 50, 2)/S(9))/(pi*E*r*Abs(r)**3)
    assert b1.slope() == C3 + 4*q  # 断言：斜率函数
    q = (-5*SingularityFunction(x, 0, 3)/3 + 41*SingularityFunction(x, 5, 3)/27 + 45*SingularityFunction(x, 45, 2)
           + 4*SingularityFunction(x, 50, 3)/27)/(pi*E*r*Abs(r)**3)
    assert b1.deflection() == C3*x + C4 + 4*q  # 断言：挠度函数

    # 带有矩形截面的梁
    b2 = Beam(20, E, Polygon((0, 0), (a, 0), (a, c), (0, c)))  # 创建一个梁对象，矩形截面
    assert b2.cross_section == Polygon((0, 0), (a, 0), (a, c), (0, c))  # 断言：截面是否为 Polygon((0, 0), (a, 0), (a, c), (0, c))
    assert b2.second_moment == a*c**3/12  # 断言：二阶矩是否为 a*c**3/12

    # 带有三角形截面的梁
    b3 = Beam(15, E, Triangle((0, 0), (g, 0), (g/2, h)))  # 创建一个梁对象，三角形截面
    assert b3.cross_section == Triangle(Point2D(0, 0), Point2D(g, 0), Point2D(g/2, h))  # 断言：截面是否为 Triangle(Point2D(0, 0), Point2D(g, 0), Point2D(g/2, h))
    assert b3.second_moment == g*h**3/36  # 断言：二阶矩是否为 g*h**3/36

    # 复合梁
    b = b2.join(b3, "fixed")  # 将 b2 和 b3 用固定端连接成复合梁
    b.apply_load(-30, 0, -1)  # 应用加载
    b.apply_load(65, 0, -2)
    b.apply_load(40, 0, -1)
    b.bc_slope = [(0, 0)]  # 设置边界条件：斜率为零
    b.bc_deflection = [(0, 0)]  # 设置边界条件：挠度为零

    assert b.second_moment == Piecewise((a*c**3/12, x <= 20), (g*h**3/36, x <= 35))  # 断言：二阶矩是否为分段函数
    assert b.cross_section == None  # 断言：截面是否为 None
    assert b.length == 35  # 断言：长度是否为 35
    assert b.slope().subs(x, 7) == 8400/(E*a*c**3)  # 断言：在 x=7 处的斜率
    assert b.slope().subs(x, 25) == 52200/(E*g*h**3) + 39600/(E*a*c**3)  # 断言：在 x=25 处的斜率
    assert b.deflection().subs(x, 30) == -537000/(E*g*h**3) - 712000/(E*a*c**3)  # 断言：在 x=30 处的挠度

# 定义一个测试函数，用于测试 Beam3D 的最大剪力
def test_max_shear_force_Beam3D():
    x = symbols('x')  # 定义符号变量 x
    b = Beam3D(20, 40, 21, 100, 25)  # 创建一个三维梁对象
    b.apply_load(15, start=0, order=0, dir="z")  # 在 z 方向施加加载
    # 在梁结构 b 上施加集中力 12*x，作用于起始位置 0，按 y 方向加载
    b.apply_load(12*x, start=0, order=0, dir="y")
    # 设置梁结构 b 的边界条件为：在位置 0 处的位移为 [0, 0, 0]，在位置 20 处的位移也为 [0, 0, 0]
    b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
    # 断言：梁结构 b 的最大剪力应该等于 [(0, 0), (20, 2400), (20, 300)]
    assert b.max_shear_force() == [(0, 0), (20, 2400), (20, 300)]
# 定义测试函数，用于测试 Beam3D 类的最大弯矩计算功能
def test_max_bending_moment_Beam3D():
    # 定义符号变量 x
    x = symbols('x')
    # 创建 Beam3D 对象，参数依次为长度、宽度、高度、弹性模量、截面惯性矩
    b = Beam3D(20, 40, 21, 100, 25)
    # 在梁上施加分布力 15 单位，作用在 z 方向
    b.apply_load(15, start=0, order=0, dir="z")
    # 在梁上施加变化的分布力 12*x，作用在 y 方向
    b.apply_load(12*x, start=0, order=0, dir="y")
    # 设定梁的边界条件：在 x=0 和 x=20 处的位移边界条件均为 [0, 0, 0]
    b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
    # 断言梁的最大弯矩计算结果
    assert b.max_bmoment() == [(0, 0), (20, 3000), (20, 16000)]

# 定义测试函数，用于测试 Beam3D 类的最大挠度计算功能
def test_max_deflection_Beam3D():
    # 定义符号变量 x
    x = symbols('x')
    # 创建 Beam3D 对象，参数依次为长度、宽度、高度、弹性模量、截面惯性矩
    b = Beam3D(20, 40, 21, 100, 25)
    # 在梁上施加分布力 15 单位，作用在 z 方向
    b.apply_load(15, start=0, order=0, dir="z")
    # 在梁上施加变化的分布力 12*x，作用在 y 方向
    b.apply_load(12*x, start=0, order=0, dir="y")
    # 设定梁的边界条件：在 x=0 和 x=20 处的位移边界条件均为 [0, 0, 0]
    b.bc_deflection = [(0, [0, 0, 0]), (20, [0, 0, 0])]
    # 求解梁的斜率和挠度
    b.solve_slope_deflection()
    # 定义常量 c、p、q
    c = sympify("495/14")
    p = sympify("-10 + 10*sqrt(10793)/43")
    q = sympify("(10 - 10*sqrt(10793)/43)**3/160 - 20/7 + (10 - 10*sqrt(10793)/43)**4/6400 + 20*sqrt(10793)/301 + 27*(10 - 10*sqrt(10793)/43)**2/560")
    # 断言梁的最大挠度计算结果
    assert b.max_deflection() == [(0, 0), (10, c), (p, q)]

# 定义测试函数，用于测试 Beam3D 类的扭转计算功能
def test_torsion_Beam3D():
    # 定义符号变量 x
    x = symbols('x')
    # 创建 Beam3D 对象，参数依次为长度、宽度、高度、弹性模量、截面惯性矩
    b = Beam3D(20, 40, 21, 100, 25)
    # 在梁上施加 15 单位力矩，作用在 x=5 处，方向为 x 轴
    b.apply_moment_load(15, 5, -2, dir='x')
    # 在梁上施加 25 单位力矩，作用在 x=10 处，方向为 x 轴
    b.apply_moment_load(25, 10, -2, dir='x')
    # 在梁上施加 -5 单位力矩，作用在 x=20 处，方向为 x 轴
    b.apply_moment_load(-5, 20, -2, dir='x')
    # 求解梁的扭转角
    b.solve_for_torsion()
    # 断言梁在特定点 x=3、x=9、x=12、x=17、x=20 处的扭转角
    assert b.angular_deflection().subs(x, 3) == sympify("1/40")
    assert b.angular_deflection().subs(x, 9) == sympify("17/280")
    assert b.angular_deflection().subs(x, 12) == sympify("53/840")
    assert b.angular_deflection().subs(x, 17) == sympify("2/35")
    assert b.angular_deflection().subs(x, 20) == sympify("3/56")
```