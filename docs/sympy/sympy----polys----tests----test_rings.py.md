# `D:\src\scipysrc\sympy\sympy\polys\tests\test_rings.py`

```
"""Test sparse polynomials. """

# 导入必要的模块和函数
from functools import reduce
from operator import add, mul

# 导入 SymPy 中与多项式相关的类和函数
from sympy.polys.rings import ring, xring, sring, PolyRing, PolyElement
from sympy.polys.fields import field, FracField
from sympy.polys.densebasic import ninf
from sympy.polys.domains import ZZ, QQ, RR, FF, EX
from sympy.polys.orderings import lex, grlex
from sympy.polys.polyerrors import GeneratorsError, \
    ExactQuotientFailed, MultivariatePolynomialError, CoercionFailed

# 导入 SymPy 中的测试工具
from sympy.testing.pytest import raises
from sympy.core import Symbol, symbols
from sympy.core.singleton import S
from sympy.core.numbers import pi
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt

# 定义测试函数 test_PolyRing___init__
def test_PolyRing___init__():
    # 定义符号变量 x, y, z, t
    x, y, z, t = map(Symbol, "xyzt")

    # 测试不同参数形式下生成多项式环对象的生成器数量
    assert len(PolyRing("x,y,z", ZZ, lex).gens) == 3
    assert len(PolyRing(x, ZZ, lex).gens) == 1
    assert len(PolyRing(("x", "y", "z"), ZZ, lex).gens) == 3
    assert len(PolyRing((x, y, z), ZZ, lex).gens) == 3
    assert len(PolyRing("", ZZ, lex).gens) == 0
    assert len(PolyRing([], ZZ, lex).gens) == 0

    # 测试异常情况下的生成器错误
    raises(GeneratorsError, lambda: PolyRing(0, ZZ, lex))

    # 测试多项式环对象的定义域是否正确
    assert PolyRing("x", ZZ[t], lex).domain == ZZ[t]
    assert PolyRing("x", 'ZZ[t]', lex).domain == ZZ[t]
    assert PolyRing("x", PolyRing("t", ZZ, lex), lex).domain == ZZ[t]

    # 测试嵌套生成器错误的情况
    raises(GeneratorsError, lambda: PolyRing("x", PolyRing("x", ZZ, lex), lex))

    # 测试多项式环对象的排序是否正确
    _lex = Symbol("lex")
    assert PolyRing("x", ZZ, lex).order == lex
    assert PolyRing("x", ZZ, _lex).order == lex
    assert PolyRing("x", ZZ, 'lex').order == lex

    # 测试多项式环对象的相等性和不相等性
    R1 = PolyRing("x,y", ZZ, lex)
    R2 = PolyRing("x,y", ZZ, lex)
    R3 = PolyRing("x,y,z", ZZ, lex)

    assert R1.x == R1.gens[0]
    assert R1.y == R1.gens[1]
    assert R1.x == R2.x
    assert R1.y == R2.y
    assert R1.x != R3.x
    assert R1.y != R3.y

# 定义测试函数 test_PolyRing___hash__
def test_PolyRing___hash__():
    # 创建有理数域 QQ 上的多项式环 R 和符号变量 x, y, z
    R, x, y, z = ring("x,y,z", QQ)
    
    # 测试多项式环对象 R 的哈希值是否可获取
    assert hash(R)

# 定义测试函数 test_PolyRing___eq__
def test_PolyRing___eq__():
    # 比较不同生成器和定义域下的多项式环对象是否相等和相同
    assert ring("x,y,z", QQ)[0] == ring("x,y,z", QQ)[0]
    assert ring("x,y,z", QQ)[0] is ring("x,y,z", QQ)[0]

    assert ring("x,y,z", QQ)[0] != ring("x,y,z", ZZ)[0]
    assert ring("x,y,z", QQ)[0] is not ring("x,y,z", ZZ)[0]

    assert ring("x,y,z", ZZ)[0] != ring("x,y,z", QQ)[0]
    assert ring("x,y,z", ZZ)[0] is not ring("x,y,z", QQ)[0]

    assert ring("x,y,z", QQ)[0] != ring("x,y", QQ)[0]
    assert ring("x,y,z", QQ)[0] is not ring("x,y", QQ)[0]

    assert ring("x,y", QQ)[0] != ring("x,y,z", QQ)[0]
    assert ring("x,y", QQ)[0] is not ring("x,y,z", QQ)[0]

# 定义测试函数 test_PolyRing_ring_new__
def test_PolyRing_ring_new():
    # 创建有理数域 QQ 上的多项式环 R 和符号变量 x, y, z
    R, x, y, z = ring("x,y,z", QQ)

    # 测试多项式环对象的 ring_new 方法
    assert R.ring_new(7) == R(7)
    assert R.ring_new(7*x*y*z) == 7*x*y*z

    # 定义测试多项式 f
    f = x**2 + 2*x*y + 3*x + 4*z**2 + 5*z + 6

    # 测试 ring_new 方法接受不同形式输入
    assert R.ring_new([[[1]], [[2], [3]], [[4, 5, 6]]]) == f
    assert R.ring_new({(2, 0, 0): 1, (1, 1, 0): 2, (1, 0, 0): 3, (0, 0, 2): 4, (0, 0, 1): 5, (0, 0, 0): 6}) == f
    `
    # 我们开始给代码加注释的行动，下面是第一行的注释：
    
    assert R.ring_new([((2, 0, 0), 1), ((1, 1, 0), 2), ((1, 0, 0), 3), ((0, 0, 2), 4), ((0, 0, 1), 5), ((0, 0, 0), 6)]) == f
    # 确保你理解了它，它会看起来像是面对的完整。 解 
     in
def test_PolyRing_drop():
    # 创建一个多项式环 R，包含变量 x, y, z，并使用整数环 ZZ
    R, x,y,z = ring("x,y,z", ZZ)

    # 断言去掉变量 x 后的环是 "y,z" 形式的多项式环，使用词典序排列 lex
    assert R.drop(x) == PolyRing("y,z", ZZ, lex)
    # 断言去掉变量 y 后的环是 "x,z" 形式的多项式环，使用词典序排列 lex
    assert R.drop(y) == PolyRing("x,z", ZZ, lex)
    # 断言去掉变量 z 后的环是 "x,y" 形式的多项式环，使用词典序排列 lex
    assert R.drop(z) == PolyRing("x,y", ZZ, lex)

    # 断言去掉索引为 0 的变量后的环是 "y,z" 形式的多项式环，使用词典序排列 lex
    assert R.drop(0) == PolyRing("y,z", ZZ, lex)
    # 断言连续去掉索引为 0 的变量两次后的环是 "z" 形式的多项式环，使用词典序排列 lex
    assert R.drop(0).drop(0) == PolyRing("z", ZZ, lex)
    # 断言连续去掉索引为 0 的变量三次后的结果是整数环 ZZ
    assert R.drop(0).drop(0).drop(0) == ZZ

    # 断言去掉索引为 1 的变量后的环是 "x,z" 形式的多项式环，使用词典序排列 lex
    assert R.drop(1) == PolyRing("x,z", ZZ, lex)

    # 断言去掉索引为 2 的变量后的环是 "x,y" 形式的多项式环，使用词典序排列 lex
    assert R.drop(2) == PolyRing("x,y", ZZ, lex)
    # 断言连续去掉索引为 2 和 1 的变量后的环是 "x" 形式的多项式环，使用词典序排列 lex
    assert R.drop(2).drop(1) == PolyRing("x", ZZ, lex)
    # 断言连续去掉索引为 2、1 和 0 的变量后的结果是整数环 ZZ
    assert R.drop(2).drop(1).drop(0) == ZZ

    # 断言试图去掉不存在的索引（索引为 3）会引发 ValueError 异常
    raises(ValueError, lambda: R.drop(3))
    # 断言试图去掉不存在的变量（x 和 y）会引发 ValueError 异常
    raises(ValueError, lambda: R.drop(x).drop(y))

def test_PolyRing___getitem__():
    # 创建一个多项式环 R，包含变量 x, y, z，并使用整数环 ZZ
    R, x,y,z = ring("x,y,z", ZZ)

    # 断言取出从索引 0 到最后的变量后的环与原环相同
    assert R[0:] == PolyRing("x,y,z", ZZ, lex)
    # 断言取出从索引 1 到最后的变量后的环是 "y,z" 形式的多项式环，使用词典序排列 lex
    assert R[1:] == PolyRing("y,z", ZZ, lex)
    # 断言取出从索引 2 到最后的变量后的环是 "z" 形式的多项式环，使用词典序排列 lex
    assert R[2:] == PolyRing("z", ZZ, lex)
    # 断言取出从索引 3 到最后的变量后的结果是整数环 ZZ
    assert R[3:] == ZZ

def test_PolyRing_is_():
    # 创建一个单变量环 R，变量是 x，使用有理数域 QQ 和词典序排列 lex
    R = PolyRing("x", QQ, lex)

    # 断言该环是单变量环
    assert R.is_univariate is True
    # 断言该环不是多变量环
    assert R.is_multivariate is False

    # 创建一个三变量环 R，变量是 x, y, z，使用有理数域 QQ 和词典序排列 lex
    R = PolyRing("x,y,z", QQ, lex)

    # 断言该环不是单变量环
    assert R.is_univariate is False
    # 断言该环是多变量环
    assert R.is_multivariate is True

    # 创建一个空环 R，不含变量，使用有理数域 QQ 和词典序排列 lex
    R = PolyRing("", QQ, lex)

    # 断言该环不是单变量环
    assert R.is_univariate is False
    # 断言该环不是多变量环
    assert R.is_multivariate is False

def test_PolyRing_add():
    # 创建一个单变量环 R，变量是 x，使用整数环 ZZ
    R, x = ring("x", ZZ)
    # 创建一个包含多项式的列表 F
    F = [ x**2 + 2*i + 3 for i in range(4) ]

    # 断言将列表 F 中的多项式相加后得到的结果与 reduce(add, F) 相同，并且等于 4*x**2 + 24
    assert R.add(F) == reduce(add, F) == 4*x**2 + 24

    # 创建一个空环 R
    R, = ring("", ZZ)

    # 断言将整数列表 [2, 5, 7] 相加后得到的结果是 14
    assert R.add([2, 5, 7]) == 14

def test_PolyRing_mul():
    # 创建一个单变量环 R，变量是 x，使用整数环 ZZ
    R, x = ring("x", ZZ)
    # 创建一个包含多项式的列表 F
    F = [ x**2 + 2*i + 3 for i in range(4) ]

    # 断言将列表 F 中的多项式相乘后得到的结果与 reduce(mul, F) 相同，并且等于 x**8 + 24*x**6 + 206*x**4 + 744*x**2 + 945
    assert R.mul(F) == reduce(mul, F) == x**8 + 24*x**6 + 206*x**4 + 744*x**2 + 945

    # 创建一个空环 R
    R, = ring("", ZZ)

    # 断言将整数列表 [2, 3, 5] 相乘后得到的结果是 30
    assert R.mul([2, 3, 5]) == 30

def test_PolyRing_symmetric_poly():
    # 创建一个四变量环 R，变量是 x, y, z, t，使用整数环 ZZ
    R, x, y, z, t = ring("x,y,z,t", ZZ)

    # 断言试图计算负数次对称多项式会引发 ValueError 异常
    raises(ValueError, lambda: R.symmetric_poly(-1))
    # 断言试图计算超过变量个数次数的对称多项式会引发 ValueError 异常
    raises(ValueError, lambda: R.symmetric_poly(5))

    # 断言计算次数为 0 的对称多项式得到单位元 R.one
    assert R.symmetric_poly(0) == R.one
    # 断言计算次数为 1 的对称多项式得到 x + y + z + t
    assert R.symmetric_poly(1) == x + y + z + t
    # 断言计算次数为 2 的对称多项式得到 x*y + x*z + x*t + y*z + y*t + z*t
    assert R.symmetric_poly(2) == x*y + x*z + x*t + y*z + y*t + z*t
    # 断言计算次数为 3 的对称多项式得到 x*y*z + x*y*t + x*z*t + y*z*t
    assert R.symmetric_poly(3) == x*y*z + x*y*t + x*z*t + y*z*t
    # 断言计算次数
    # 断言：验证域 R 的属性 domain 等于 QQ 的代数域，其元素为 sqrt(2) + sqrt(3)
    assert R.domain == QQ.algebraic_field(sqrt(2) + sqrt(3))
    
    # 断言：验证域 R 的生成元为空元组
    assert R.gens == ()
    
    # 断言：验证变量 a 等于域 R 中从 sympy 对象 r 转换而来的元素
    assert a == R.domain.from_sympy(r)
# 定义一个测试函数，测试 PolyElement 类的 __hash__ 方法
def test_PolyElement___hash__():
    # 创建一个多项式环 R，以及变量 x, y, z，使用有理数 QQ 作为系数环
    R, x, y, z = ring("x,y,z", QQ)
    # 断言计算 x*y*z 的哈希值
    assert hash(x*y*z)

# 定义一个测试函数，测试 PolyElement 类的 __eq__ 方法
def test_PolyElement___eq__():
    # 创建一个多项式环 R，以及变量 x, y，使用整数环 ZZ 和 lex 排序
    R, x, y = ring("x,y", ZZ, lex)

    # 一系列等式和不等式的断言，测试多项式相等和不相等的情况
    assert ((x*y + 5*x*y) == 6) == False
    assert ((x*y + 5*x*y) == 6*x*y) == True
    assert (6 == (x*y + 5*x*y)) == False
    assert (6*x*y == (x*y + 5*x*y)) == True

    assert ((x*y - x*y) == 0) == True
    assert (0 == (x*y - x*y)) == True

    assert ((x*y - x*y) == 1) == False
    assert (1 == (x*y - x*y)) == False

    assert ((x*y + 5*x*y) != 6) == True
    assert ((x*y + 5*x*y) != 6*x*y) == False
    assert (6 != (x*y + 5*x*y)) == True
    assert (6*x*y != (x*y + 5*x*y)) == False

    # 测试 R.one 的等价性
    assert R.one == QQ(1, 1) == R.one
    assert R.one == 1 == R.one

    # 创建一个新的多项式环 Rt，以及变量 t
    Rt, t = ring("t", ZZ)
    # 在 Rt 中创建新的多项式环 R，以及变量 x, y
    R, x, y = ring("x,y", Rt)

    # 断言 t**3*x/x 与 t**3 和 t**4 的比较结果
    assert (t**3*x/x == t**3) == True
    assert (t**3*x/x == t**4) == False

# 定义一个测试函数，测试 PolyElement 类的比较运算符 __lt__, __le__, __gt__, __ge__
def test_PolyElement__lt_le_gt_ge__():
    # 创建一个多项式环 R，以及变量 x, y，使用整数环 ZZ
    R, x, y = ring("x,y", ZZ)

    # 一系列比较运算的断言，测试多项式的大小关系
    assert R(1) < x < x**2 < x**3
    assert R(1) <= x <= x**2 <= x**3

    assert x**3 > x**2 > x > R(1)
    assert x**3 >= x**2 >= x >= R(1)

# 定义一个测试函数，测试 PolyElement 类的 __str__ 方法
def test_PolyElement__str__():
    # 导入符号 x, y
    x, y = symbols('x, y')

    # 循环遍历不同的环 dom
    for dom in [ZZ, QQ, ZZ[x], ZZ[x,y], ZZ[x][y]]:
        # 在每个 dom 中创建一个多项式环 R，以及变量 t
        R, t = ring('t', dom)
        # 断言多项式对象的字符串表示是否与预期相符
        assert str(2*t**2 + 1) == '2*t**2 + 1'

    # 再次循环遍历不同的环 dom
    for dom in [EX, EX[x]]:
        # 在每个 dom 中创建一个多项式环 R，以及变量 t
        R, t = ring('t', dom)
        # 断言多项式对象的字符串表示是否与预期相符
        assert str(2*t**2 + 1) == 'EX(2)*t**2 + EX(1)'

# 定义一个测试函数，测试 PolyElement 类的复制方法 copy()
def test_PolyElement_copy():
    # 创建一个多项式环 R，以及变量 x, y, z，使用整数环 ZZ
    R, x, y, z = ring("x,y,z", ZZ)

    # 创建多项式 f = x*y + 3*z
    f = x*y + 3*z
    # 复制多项式 f 得到 g
    g = f.copy()

    # 断言 f 和 g 相等
    assert f == g
    # 修改 g 的某些项
    g[(1, 1, 1)] = 7
    # 断言 f 和 g 不相等
    assert f != g

# 定义一个测试函数，测试 PolyElement 类的 as_expr() 方法
def test_PolyElement_as_expr():
    # 创建一个多项式环 R，以及变量 x, y, z，使用整数环 ZZ
    R, x, y, z = ring("x,y,z", ZZ)
    # 创建多项式 f = 3*x**2*y - x*y*z + 7*z**3 + 1
    f = 3*x**2*y - x*y*z + 7*z**3 + 1

    # 获取 R 中的符号 X, Y, Z
    X, Y, Z = R.symbols
    # 创建多项式 g
    g = 3*X**2*Y - X*Y*Z + 7*Z**3 + 1

    # 断言 f 和 g 不相等
    assert f != g
    # 断言 f 转换为表达式后与 g 相等
    assert f.as_expr() == g

    # 创建新的符号 U, V, W
    U, V, W = symbols("u,v,w")
    # 创建多项式 g
    g = 3*U**2*V - U*V*W + 7*W**3 + 1

    # 断言 f 和 g 不相等
    assert f != g
    # 断言使用符号 U, V, W 转换 f 后与 g 相等
    assert f.as_expr(U, V, W) == g

    # 使用 R.one 创建空环 R
    R, = ring("", ZZ)
    # 断言 R(3) 转换为表达式后为整数 3
    assert R(3).as_expr() == 3

# 定义一个测试函数，测试 PolyElement 类的 from_expr() 方法
def test_PolyElement_from_expr():
    # 导入符号 x, y, z
    x, y, z = symbols("x,y,z")
    # 创建一个多项式环 R，以及变量 X, Y, Z，使用整数环 ZZ
    R, X, Y, Z = ring((x, y, z), ZZ)

    # 一系列断言，测试从表达式构造多项式的正确性
    f = R.from_expr(1)
    assert f == 1 and isinstance(f, R.dtype)

    f = R.from_expr(x)
    assert f == X and isinstance(f, R.dtype)

    f = R.from_expr(x*y*z)
    assert f == X*Y*Z and isinstance(f, R.dtype)

    f = R.from_expr(x*y*z + x*y + x)
    assert f == X*Y*Z + X*Y + X and isinstance(f, R.dtype)

    f = R.from_expr(x**3*y*z + x**2*y**7 + 1)
    assert f == X**3*Y*Z + X**2*Y**7 + 1 and isinstance(f, R.dtype)

    # 创建一个特殊的环 r，以及 F[exp(2)]
    r, F = sring([exp(2)])
    # 从表达式 exp(2) 构造多项式 f
    f = r.from_expr(exp(2))
    assert f == F[0] and isinstance(f, r.dtype)

    # 测试无法构造的情况，期望抛出 ValueError
    raises(ValueError, lambda: R.from_expr(1/x))
    raises(ValueError, lambda: R.from_expr(2**x))
    # 调用 raises 函数验证是否会引发 ValueError 异常，使用 lambda 表达式检查 R.from_expr(7*x + sqrt(2)) 是否会抛出异常
    raises(ValueError, lambda: R.from_expr(7*x + sqrt(2)))

    # 使用 ring 函数创建一个空的环 R，并将其解构为 R 变量，ZZ 是整数环
    R, = ring("", ZZ)

    # 使用 R.from_expr(1) 创建一个环 R 中的元素 f，并断言 f 等于 1 且是 R.dtype 类型的实例
    assert f == 1 and isinstance(f, R.dtype)
# 定义一个测试函数，用于测试多项式元素的度量和系数计算
def test_PolyElement_degree():
    # 创建一个多项式环 R，并定义变量 x, y, z
    R, x, y, z = ring("x,y,z", ZZ)

    # 断言负无穷的表示应与 float 类型中的负无穷值相等
    assert ninf == float('-inf')

    # 测试多项式元素在整体上的度量
    assert R(0).degree() is ninf        # 多项式元素 0 的度应为负无穷
    assert R(1).degree() == 0           # 多项式元素 1 的度应为 0
    assert (x + 1).degree() == 1        # 多项式元素 x + 1 的度应为 1
    assert (2*y**3 + z).degree() == 0   # 多项式元素 2*y**3 + z 的度应为 0
    assert (x*y**3 + z).degree() == 1   # 多项式元素 x*y**3 + z 的度应为 1
    assert (x**5*y**3 + z).degree() == 5  # 多项式元素 x**5*y**3 + z 的度应为 5

    # 测试多项式元素关于特定变量 x 的度量
    assert R(0).degree(x) is ninf      # 多项式元素 0 关于 x 的度应为负无穷
    assert R(1).degree(x) == 0         # 多项式元素 1 关于 x 的度应为 0
    assert (x + 1).degree(x) == 1      # 多项式元素 x + 1 关于 x 的度应为 1
    assert (2*y**3 + z).degree(x) == 0 # 多项式元素 2*y**3 + z 关于 x 的度应为 0
    assert (x*y**3 + z).degree(x) == 1 # 多项式元素 x*y**3 + z 关于 x 的度应为 1
    assert (7*x**5*y**3 + z).degree(x) == 5  # 多项式元素 7*x**5*y**3 + z 关于 x 的度应为 5

    # 测试多项式元素关于特定变量 y 的度量
    assert R(0).degree(y) is ninf      # 多项式元素 0 关于 y 的度应为负无穷
    assert R(1).degree(y) == 0         # 多项式元素 1 关于 y 的度应为 0
    assert (x + 1).degree(y) == 0      # 多项式元素 x + 1 关于 y 的度应为 0
    assert (2*y**3 + z).degree(y) == 3 # 多项式元素 2*y**3 + z 关于 y 的度应为 3
    assert (x*y**3 + z).degree(y) == 3 # 多项式元素 x*y**3 + z 关于 y 的度应为 3
    assert (7*x**5*y**3 + z).degree(y) == 3  # 多项式元素 7*x**5*y**3 + z 关于 y 的度应为 3

    # 测试多项式元素关于特定变量 z 的度量
    assert R(0).degree(z) is ninf      # 多项式元素 0 关于 z 的度应为负无穷
    assert R(1).degree(z) == 0         # 多项式元素 1 关于 z 的度应为 0
    assert (x + 1).degree(z) == 0      # 多项式元素 x + 1 关于 z 的度应为 0
    assert (2*y**3 + z).degree(z) == 1 # 多项式元素 2*y**3 + z 关于 z 的度应为 1
    assert (x*y**3 + z).degree(z) == 1 # 多项式元素 x*y**3 + z 关于 z 的度应为 1
    assert (7*x**5*y**3 + z).degree(z) == 1  # 多项式元素 7*x**5*y**3 + z 关于 z 的度应为 1

    # 对于没有变量的情况，测试多项式元素的度量
    R, = ring("", ZZ)
    assert R(0).degree() is ninf      # 多项式元素 0 的度应为负无穷
    assert R(1).degree() == 0         # 多项式元素 1 的度应为 0

# 定义测试函数，用于测试多项式元素的尾度量
def test_PolyElement_tail_degree():
    # 创建一个多项式环 R，并定义变量 x, y, z
    R, x, y, z = ring("x,y,z", ZZ)

    # 测试多项式元素在整体上的尾度量
    assert R(0).tail_degree() is ninf     # 多项式元素 0 的尾度应为负无穷
    assert R(1).tail_degree() == 0        # 多项式元素 1 的尾度应为 0
    assert (x + 1).tail_degree() == 0     # 多项式元素 x + 1 的尾度应为 0
    assert (2*y**3 + x**3*z).tail_degree() == 0  # 多项式元素 2*y**3 + x**3*z 的尾度应为 0
    assert (x*y**3 + x**3*z).tail_degree() == 1  # 多项式元素 x*y**3 + x**3*z 的尾度应为 1
    assert (x**5*y**3 + x**3*z).tail_degree() == 3  # 多项式元素 x**5*y**3 + x**3*z 的尾度应为 3

    # 测试多项式元素关于特定变量 x 的尾度量
    assert R(0).tail_degree(x) is ninf      # 多项式元素 0 关于 x 的尾度应为负无穷
    assert R(1).tail_degree(x) == 0         # 多项式元素 1 关于 x 的尾度应为 0
    assert (x + 1).tail_degree(x) == 0      # 多项式元素 x + 1 关于 x 的尾度应为 0
    assert (2*y**3 + x**3*z).tail_degree(x) == 0  # 多项式元素 2*y**3 + x**3*z 关于 x 的尾度应为 0
    assert (x*y**3 + x**3*z).tail_degree(x) == 1  # 多项式元素 x*y**3 + x**3*z 关于 x 的尾度应为 1
    assert (7*x**5*y**3 + x**3*z).tail_degree(x) == 3  # 多项式元素 7*x**5*y**3 + x**3*z 关于 x 的尾度应为 3

    # 测试多项式元素关于特定变量 y 的尾度量
    assert R(0).tail_degree(y) is ninf      # 多项式元素 0 关于 y 的尾度应为负无穷
    assert R(1).tail_degree(y) == 0         # 多项式元素 1 关于 y 的尾度应为 0
    assert (x + 1).tail_degree(y) == 0      # 多项式元素 x + 1 关于 y 的尾度应为 0
    assert (2*y**3 + x**3*z).tail_degree(y) == 0  # 多项式元素 2*y**3 + x**3*z 关于 y 的尾度应为 0
    assert (x*y**3 + x**3*z).tail_degree(y) == 0  # 多项式元素 x*y**3 + x**3*z 关于 y 的尾度应为 0
    assert (7*x**5*y**3 + x**3*z).tail_degree(y) == 0  # 多项式元素 7*x**5*y
    # 断言：检查多项式 f 中 x^2 * y 的系数是否为 3
    assert f.coeff(x**2*y) == 3
    # 断言：检查多项式 f 中 x * y * z 的系数是否为 -1
    assert f.coeff(x*y*z) == -1
    # 断言：检查多项式 f 中 z^3 的系数是否为 7
    assert f.coeff(z**3) == 7

    # 断言：验证当尝试获取多项式 f 中不存在的项的系数时，会引发 ValueError 异常
    raises(ValueError, lambda: f.coeff(3*x**2*y))
    raises(ValueError, lambda: f.coeff(-x*y*z))
    raises(ValueError, lambda: f.coeff(7*z**3))

    # 创建一个空的整数环 R
    R, = ring("", ZZ)
    # 断言：检查整数环 R 中元素 3 的系数是否为 3
    assert R(3).coeff(1) == 3
# 定义测试函数，用于测试多项式元素的 LC（最低次项的系数）
def test_PolyElement_LC():
    # 创建一个多项式环 R，包括变量 x 和 y，使用有理数域 QQ 和词典序排序 lex
    R, x, y = ring("x,y", QQ, lex)
    # 断言：零多项式的 LC 应为有理数域 QQ 中的 0
    assert R(0).LC == QQ(0)
    # 断言：(1/2)*x 的 LC 应为有理数 QQ 中的 1/2
    assert (QQ(1,2)*x).LC == QQ(1, 2)
    # 断言：(1/4)*x*y + (1/2)*x 的 LC 应为有理数 QQ 中的 1/4
    assert (QQ(1,4)*x*y + QQ(1,2)*x).LC == QQ(1, 4)

# 定义测试函数，用于测试多项式元素的 LM（最高次项的指数）
def test_PolyElement_LM():
    # 创建一个多项式环 R，包括变量 x 和 y，使用有理数域 QQ 和词典序排序 lex
    R, x, y = ring("x,y", QQ, lex)
    # 断言：零多项式的 LM 应为元组 (0, 0)
    assert R(0).LM == (0, 0)
    # 断言：(1/2)*x 的 LM 应为元组 (1, 0)
    assert (QQ(1,2)*x).LM == (1, 0)
    # 断言：(1/4)*x*y + (1/2)*x 的 LM 应为元组 (1, 1)
    assert (QQ(1,4)*x*y + QQ(1,2)*x).LM == (1, 1)

# 定义测试函数，用于测试多项式元素的 LT（最高次项）
def test_PolyElement_LT():
    # 创建一个多项式环 R，包括变量 x 和 y，使用有理数域 QQ 和词典序排序 lex
    R, x, y = ring("x,y", QQ, lex)
    # 断言：零多项式的 LT 应为元组 ((0, 0), QQ(0))
    assert R(0).LT == ((0, 0), QQ(0))
    # 断言：(1/2)*x 的 LT 应为元组 ((1, 0), QQ(1, 2))
    assert (QQ(1,2)*x).LT == ((1, 0), QQ(1, 2))
    # 断言：(1/4)*x*y + (1/2)*x 的 LT 应为元组 ((1, 1), QQ(1, 4))
    assert (QQ(1,4)*x*y + QQ(1,2)*x).LT == ((1, 1), QQ(1, 4))

    # 创建一个空多项式环 R，只包含整数域 ZZ
    R, = ring("", ZZ)
    # 断言：零多项式的 LT 应为元组 ((), 0)
    assert R(0).LT == ((), 0)
    # 断言：常数多项式 1 的 LT 应为元组 ((), 1)
    assert R(1).LT == ((), 1)

# 定义测试函数，用于测试多项式元素的 leading_monom（最高次单项式）
def test_PolyElement_leading_monom():
    # 创建一个多项式环 R，包括变量 x 和 y，使用有理数域 QQ 和词典序排序 lex
    R, x, y = ring("x,y", QQ, lex)
    # 断言：零多项式的 leading_monom 应为 0
    assert R(0).leading_monom() == 0
    # 断言：(1/2)*x 的 leading_monom 应为 x
    assert (QQ(1,2)*x).leading_monom() == x
    # 断言：(1/4)*x*y + (1/2)*x 的 leading_monom 应为 x*y
    assert (QQ(1,4)*x*y + QQ(1,2)*x).leading_monom() == x*y

# 定义测试函数，用于测试多项式元素的 leading_term（最高次项）
def test_PolyElement_leading_term():
    # 创建一个多项式环 R，包括变量 x 和 y，使用有理数域 QQ 和词典序排序 lex
    R, x, y = ring("x,y", QQ, lex)
    # 断言：零多项式的 leading_term 应为 0
    assert R(0).leading_term() == 0
    # 断言：(1/2)*x 的 leading_term 应为 (1/2)*x
    assert (QQ(1,2)*x).leading_term() == QQ(1,2)*x
    # 断言：(1/4)*x*y + (1/2)*x 的 leading_term 应为 (1/4)*x*y
    assert (QQ(1,4)*x*y + QQ(1,2)*x).leading_term() == QQ(1,4)*x*y

# 定义测试函数，用于测试多项式元素的 terms（项的列表）
def test_PolyElement_terms():
    # 创建一个多项式环 R，包括变量 x, y, z，使用有理数域 QQ
    R, x,y,z = ring("x,y,z", QQ)
    # 计算多项式 x^2/3 + y^3/4 + z^4/5 的所有项
    terms = (x**2/3 + y**3/4 + z**4/5).terms()
    # 断言：计算得到的项列表应为 [((2,0,0), QQ(1,3)), ((0,3,0), QQ(1,4)), ((0,0,4), QQ(1,5))]
    assert terms == [((2,0,0), QQ(1,3)), ((0,3,0), QQ(1,4)), ((0,0,4), QQ(1,5))]

    # 创建一个多项式环 R，包括变量 x, y，使用整数域 ZZ 和词典序排序 lex
    R, x,y = ring("x,y", ZZ, lex)
    # 创建多项式 f = x*y^7 + 2*x^2*y^3
    f = x*y**7 + 2*x**2*y**3

    # 断言：f 的项列表应与 lex 排序后的结果相同
    assert f.terms() == f.terms(lex) == f.terms('lex') == [((2, 3), 2), ((1, 7), 1)]
    # 断言：f 的项列表应与 grlex 排序后的结果相同
    assert f.terms(grlex) == f.terms('grlex') == [((1, 7), 1), ((2, 3), 2)]

    # 创建一个多项式环 R，包括变量 x, y，使用整数域 ZZ 和 graded lex 排序
    R, x,y = ring("x,y", ZZ, grlex)
    # 多项式重新定义为 x*y^7 + 2*x^2*y^3
    f = x*y**7 + 2*x**2*y**3

    # 断言：f 的项列表应与 grlex 排序后的结果相同
    assert f.terms() == f.terms(grlex) == f.terms('grlex') == [((1, 7), 1), ((2, 3), 2)]
    # 断言：f 的项列表应与 lex 排序后的结果相同
    assert f.terms(lex) == f.terms('lex') == [((2, 3), 2), ((1, 7), 1)]

    # 创建一个空多项式环 R，只包含整数域 ZZ
    R, = ring("", ZZ)
    # 断言：常数多项式 3 的项列表应为 [(((), 3))]
    assert R(3).terms() == [((), 3)]

# 定义测试函数，用于测试多项式元素的 monoms（单项式的列表）
def test_PolyElement_monoms():
    # 创建一个多项式环 R，包括变量 x, y, z，
    # 创建整数环 Ruv，并定义变量 u, v
    Ruv, u,v = ring("u,v", ZZ)
    # 创建多项式环 Rxyz，包含变量 x, y, z
    Rxyz, x,y,z = ring("x,y,z", Ruv)

    # 断言检查 x + 3*y 的字典表示是否符合预期
    assert dict(x + 3*y) == {(1, 0, 0): 1, (0, 1, 0): 3}

    # 断言检查 u + x 的字典表示是否符合预期，同时检查 x + u 是否相同
    assert dict(u + x) == dict(x + u) == {(1, 0, 0): 1, (0, 0, 0): u}
    # 断言检查 u + x*y 的字典表示是否符合预期，同时检查 x*y + u 是否相同
    assert dict(u + x*y) == dict(x*y + u) == {(1, 1, 0): 1, (0, 0, 0): u}
    # 断言检查 u + x*y + z 的字典表示是否符合预期，同时检查 x*y + z + u 是否相同
    assert dict(u + x*y + z) == dict(x*y + z + u) == {(1, 1, 0): 1, (0, 0, 1): 1, (0, 0, 0): u}

    # 断言检查 u*x + x 的字典表示是否符合预期，同时检查 x + u*x 是否相同
    assert dict(u*x + x) == dict(x + u*x) == {(1, 0, 0): u + 1}
    # 断言检查 u*x + x*y 的字典表示是否符合预期，同时检查 x*y + u*x 是否相同
    assert dict(u*x + x*y) == dict(x*y + u*x) == {(1, 1, 0): 1, (1, 0, 0): u}
    # 断言检查 u*x + x*y + z 的字典表示是否符合预期，同时检查 x*y + z + u*x 是否相同
    assert dict(u*x + x*y + z) == dict(x*y + z + u*x) == {(1, 1, 0): 1, (0, 0, 1): 1, (1, 0, 0): u}

    # 断言检查 t + x 的操作是否引发 TypeError 异常
    raises(TypeError, lambda: t + x)
    # 断言检查 x + t 的操作是否引发 TypeError 异常
    raises(TypeError, lambda: x + t)
    # 断言检查 t + u 的操作是否引发 TypeError 异常
    raises(TypeError, lambda: t + u)
    # 断言检查 u + t 的操作是否引发 TypeError 异常
    raises(TypeError, lambda: u + t)

    # 创建有理函数域 Fuv，并定义变量 u, v
    Fuv, u,v = field("u,v", ZZ)
    # 创建多项式环 Rxyz，包含变量 x, y, z，但使用有理函数域 Fuv 作为系数环
    Rxyz, x,y,z = ring("x,y,z", Fuv)

    # 断言检查 u + x 的字典表示是否符合预期，同时检查 x + u 是否相同
    assert dict(u + x) == dict(x + u) == {(1, 0, 0): 1, (0, 0, 0): u}

    # 创建表达式环 EX，并定义变量 x, y, z
    Rxyz, x,y,z = ring("x,y,z", EX)

    # 断言检查 EX(pi) + x*y*z 的字典表示是否符合预期，同时检查 x*y*z + EX(pi) 是否相同
    assert dict(EX(pi) + x*y*z) == dict(x*y*z + EX(pi)) == {(1, 1, 1): EX(1), (0, 0, 0): EX(pi)}
# 定义测试函数 test_PolyElement___sub__()
def test_PolyElement___sub__():
    # 定义环 Rt 和变量 t，其基础为整数环 ZZ
    Rt, t = ring("t", ZZ)
    # 定义环 Ruv 和变量 u, v，其基础为整数环 ZZ
    Ruv, u, v = ring("u,v", ZZ)
    # 定义环 Rxyz 和变量 x, y, z，其基础为环 Ruv
    Rxyz, x, y, z = ring("x,y,z", Ruv)

    # 断言：计算多项式 x - 3*y 的字典表达式，结果应为 {(1, 0, 0): 1, (0, 1, 0): -3}
    assert dict(x - 3*y) == {(1, 0, 0): 1, (0, 1, 0): -3}

    # 断言：计算多项式 -u + x 的字典表达式，结果应为 {(1, 0, 0): 1, (0, 0, 0): -u}
    assert dict(-u + x) == dict(x - u) == {(1, 0, 0): 1, (0, 0, 0): -u}
    # 断言：计算多项式 -u + x*y 的字典表达式，结果应为 {(1, 1, 0): 1, (0, 0, 0): -u}
    assert dict(-u + x*y) == dict(x*y - u) == {(1, 1, 0): 1, (0, 0, 0): -u}
    # 断言：计算多项式 -u + x*y + z 的字典表达式，结果应为 {(1, 1, 0): 1, (0, 0, 1): 1, (0, 0, 0): -u}
    assert dict(-u + x*y + z) == dict(x*y + z - u) == {(1, 1, 0): 1, (0, 0, 1): 1, (0, 0, 0): -u}

    # 断言：计算多项式 -u*x + x 的字典表达式，结果应为 {(1, 0, 0): -u + 1}
    assert dict(-u*x + x) == dict(x - u*x) == {(1, 0, 0): -u + 1}
    # 断言：计算多项式 -u*x + x*y 的字典表达式，结果应为 {(1, 1, 0): 1, (1, 0, 0): -u}
    assert dict(-u*x + x*y) == dict(x*y - u*x) == {(1, 1, 0): 1, (1, 0, 0): -u}
    # 断言：计算多项式 -u*x + x*y + z 的字典表达式，结果应为 {(1, 1, 0): 1, (0, 0, 1): 1, (1, 0, 0): -u}
    assert dict(-u*x + x*y + z) == dict(x*y + z - u*x) == {(1, 1, 0): 1, (0, 0, 1): 1, (1, 0, 0): -u}

    # 断言：检测类型错误，尝试计算 t - x，应抛出 TypeError 异常
    raises(TypeError, lambda: t - x)
    # 断言：检测类型错误，尝试计算 x - t，应抛出 TypeError 异常
    raises(TypeError, lambda: x - t)
    # 断言：检测类型错误，尝试计算 t - u，应抛出 TypeError 异常
    raises(TypeError, lambda: t - u)
    # 断言：检测类型错误，尝试计算 u - t，应抛出 TypeError 异常
    raises(TypeError, lambda: u - t)

    # 定义域 Fuv 和变量 u, v，其基础为整数域 ZZ
    Fuv, u, v = field("u,v", ZZ)
    # 定义环 Rxyz 和变量 x, y, z，其基础为域 Fuv
    Rxyz, x, y, z = ring("x,y,z", Fuv)

    # 断言：计算多项式 -u + x 的字典表达式，结果应为 {(1, 0, 0): 1, (0, 0, 0): -u}
    assert dict(-u + x) == dict(x - u) == {(1, 0, 0): 1, (0, 0, 0): -u}

    # 定义环 Rxyz 和变量 x, y, z，其基础为扩展域 EX
    Rxyz, x, y, z = ring("x,y,z", EX)

    # 断言：计算多项式 -EX(pi) + x*y*z 的字典表达式，结果应为 {(1, 1, 1): EX(1), (0, 0, 0): -EX(pi)}
    assert dict(-EX(pi) + x*y*z) == dict(x*y*z - EX(pi)) == {(1, 1, 1): EX(1), (0, 0, 0): -EX(pi)}

# 定义测试函数 test_PolyElement___mul__()
def test_PolyElement___mul__():
    # 定义环 Rt 和变量 t，其基础为整数环 ZZ
    Rt, t = ring("t", ZZ)
    # 定义环 Ruv 和变量 u, v，其基础为整数环 ZZ
    Ruv, u, v = ring("u,v", ZZ)
    # 定义环 Rxyz 和变量 x, y, z，其基础为环 Ruv
    Rxyz, x, y, z = ring("x,y,z", Ruv)

    # 断言：计算多项式 u*x 的字典表达式，结果应为 {(1, 0, 0): u}
    assert dict(u*x) == dict(x*u) == {(1, 0, 0): u}

    # 断言：计算多项式 2*u*x + z 的字典表达式，结果应为 {(1, 0, 0): 2*u, (0, 0, 1): 1}
    assert dict(2*u*x + z) == dict(x*2*u + z) == {(1, 0, 0): 2*u, (0, 0, 1): 1}
    assert dict(u*2*x + z) == dict(2*x*u + z) == {(1, 0, 0): 2*u, (0, 0, 1): 1}
    assert dict(2*u*x + z) == dict(x*2*u + z) == {(1, 0, 0): 2*u, (0, 0, 1): 1}
    assert dict(u*x*2 + z) == dict(x*u*2 + z) == {(1, 0, 0): 2*u, (0, 0, 1): 1}

    # 断言：计算多项式 2*u*x*y + z 的字典表达式，结果应为 {(1, 1, 0): 2*u, (0, 0, 1): 1}
    assert dict(2*u*x*y + z) == dict(x*y*2*u + z) == {(1, 1, 0): 2*u, (0, 0, 1): 1}
    assert dict(u*2*x*y + z) == dict(2*x*y*u + z) == {(1, 1, 0): 2*u, (0, 0, 1): 1}
    assert dict(2*u*x*y + z) == dict(x*y*2*u + z) == {(1, 1, 0): 2*u, (0, 0, 1): 1}
    assert dict(u*x*y*2 + z) == dict(x*y*u*2 + z) == {(1, 1, 0): 2*u, (0, 0, 1): 1}

    # 断言：计算多项式 2*u*y*x + z 的字典表达式，结果应为 {(1, 1, 0): 2*u, (0, 0, 1): 1}
    assert dict(2*u*y*x
    # 断言：验证 x^2 - x 的商等于 x - 1
    assert (x**2 - x).quo(x) == x - 1
    
    # 断言：验证 (x^2 - 1) / x 等于 x - x^(-1)
    assert (x**2 - 1)/x == x - x**(-1)
    
    # 断言：验证 (x^2 - x) / x 等于 x - 1
    assert (x**2 - x)/x == x - 1
    
    # 断言：验证 (x^2 - 1) / (2*x) 等于 x/2 - x^(-1)/2
    assert (x**2 - 1)/(2*x) == x/2 - x**(-1)/2
    
    # 断言：验证 (x^2 - 1).quo(2*x) 等于 0
    assert (x**2 - 1).quo(2*x) == 0
    
    # 断言：验证 (x^2 - x) / (x - 1) 等于 (x^2 - x).quo(x - 1) 等于 x
    assert (x**2 - x)/(x - 1) == (x**2 - x).quo(x - 1) == x
    
    # 创建整数环 R，定义变量 x, y, z
    R, x, y, z = ring("x,y,z", ZZ)
    
    # 断言：验证 (x^2/3 + y^3/4 + z^4/5).terms() 的长度等于 0
    assert len((x**2/3 + y**3/4 + z**4/5).terms()) == 0
    
    # 创建有理数环 R，定义变量 x, y, z
    R, x, y, z = ring("x,y,z", QQ)
    
    # 断言：验证 (x^2/3 + y^3/4 + z^4/5).terms() 的长度等于 3
    assert len((x**2/3 + y**3/4 + z**4/5).terms()) == 3
    
    # 创建整数环 Rt，定义变量 t
    # 创建整数环 Ruv，定义变量 u, v
    # 创建整数环 Rxyz，定义变量 x, y, z，并在其基础上创建环 Ruv
    Rt, t = ring("t", ZZ)
    Ruv, u, v = ring("u,v", ZZ)
    Rxyz, x, y, z = ring("x,y,z", Ruv)
    
    # 断言：验证 dict((u**2*x + u)/u) 等于 {(1, 0, 0): u, (0, 0, 0): 1}
    assert dict((u**2*x + u)/u) == {(1, 0, 0): u, (0, 0, 0): 1}
    
    # 断言：验证 u / (u**2*x + u) 抛出 TypeError 异常
    raises(TypeError, lambda: u/(u**2*x + u))
    
    # 断言：验证 t / x 抛出 TypeError 异常
    raises(TypeError, lambda: t/x)
    
    # 断言：验证 x / t 抛出 TypeError 异常
    raises(TypeError, lambda: x/t)
    
    # 断言：验证 t / u 抛出 TypeError 异常
    raises(TypeError, lambda: t/u)
    
    # 断言：验证 u / t 抛出 TypeError 异常
    raises(TypeError, lambda: u/t)
    
    # 创建整数环 R，定义变量 x
    R, x = ring("x", ZZ)
    
    # 定义多项式 f 和 g
    f, g = x**2 + 2*x + 3, R(0)
    
    # 断言：验证 f.div(g) 等于 divmod(f, g) 等于 (q, r)
    # 其中 q 和 r 是 f 除以 g 的商和余数
    assert f.div(g) == divmod(f, g) == (q, r)
    
    # 断言：验证 f.rem(g) 等于 f % g 等于 r，即 f 除以 g 的余数
    assert f.rem(g) == f % g == r
    
    # 断言：验证 f.quo(g) 等于 f / g 等于 q，即 f 除以 g 的商
    assert f.quo(g) == f / g == q
    
    # 断言：验证 f.exquo(g) 抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    
    # 定义多项式 f 和 g
    f, g = 3*x**3 + x**2 + x + 5, 5*x**2 - 3*x + 1
    
    # 断言：验证 f.div(g) 等于 divmod(f, g) 等于 (q, r)
    # 其中 q 和 r 是 f 除以 g 的商和余数
    assert f.div(g) == divmod(f, g) == (q, r)
    
    # 断言：验证 f.rem(g) 等于 f % g 等于 r，即 f 除以 g 的余数
    assert f.rem(g) == f % g == r
    
    # 断言：验证 f.quo(g) 等于 f / g 等于 q，即 f 除以 g 的商
    assert f.quo(g) == f / g == q
    
    # 断言：验证 f.exquo(g) 抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    
    # 定义多项式 f 和 g
    f, g = 5*x**4 + 4*x**3 + 3*x**2 + 2*x + 1, x**2 + 2*x + 3
    
    # 断言：验证 f.div(g) 等于 divmod(f, g) 等于 (q, r)
    # 其中 q 和 r 是 f 除以 g 的商和余数
    assert f.div(g) == divmod(f, g) == (q, r)
    
    # 断言：验证 f.rem(g) 等于 f % g 等于 r，即 f 除以 g 的余数
    assert f.rem(g) == f % g == r
    
    # 断言：验证 f.quo(g) 等于 f / g 等于 q，即 f 除以 g 的商
    assert f.quo(g) == f / g == q
    
    # 断言：验证 f.exquo(g) 抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    
    # 定义多项式 f 和 g
    f, g = 5*x**5 + 4*x**4 + 3*x**3 + 2*x**2 + x, x**4 + 2*x**3 + 9
    
    # 断言：验证 f.div(g) 等于 divmod(f, g) 等于 (q, r)
    # 其中 q 和 r 是 f 除以 g 的商和余数
    assert f.div(g) == divmod(f, g) == (q, r)
    
    # 断言：验证 f.rem(g) 等于 f % g 等于 r，即 f 除以 g 的余数
    assert f.rem(g) == f % g == r
    
    # 断言：验证 f.quo(g) 等于 f / g 等于 q，即 f 除以 g 的商
    assert f.quo(g) == f / g == q
    
    # 断言：验证 f.exquo(g) 抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    
    # 创建有理数环 R，定义变量 x
    R, x = ring("x", QQ)
    
    # 定义多项式 f 和 g
    f, g = x**2 + 1, 2*x - 4
    
    # 断言：验证 f.div(g) 等于 divmod(f, g) 等于 (q, r)
    # 其中 q 是 f 除以 g 的商，r 是 f 除以 g 的余数
    assert f.div(g) == divmod(f, g) == (q, r)
    
    # 断言：验证 f.rem(g) 等于 f % g 等于 r，即 f 除以 g 的余数
    assert f.rem(g) == f % g == r
    
    # 断言：验证 f.quo(g) 等于 f / g 等于 q，即 f 除以 g 的商
    assert f.quo(g) == f / g == q
    
    # 断言：验证 f.exquo(g) 抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: f.exquo(g))
    
    # 定义多项式 f 和 g
    f,
    # 断言：使用 f 对 g 进行整除运算，应与 divmod(f, g) 和 (q, r) 相等
    assert f.div(g) == divmod(f, g) == (q, r)
    # 断言：使用 f 对 g 进行取余运算，应与 f % g 和 r 相等
    assert f.rem(g) == f % g == r
    # 断言：使用 f 对 g 进行浮点除法运算，应与 f / g 和 q 相等
    assert f.quo(g) == f / g == q
    # 断言：使用 f 对 g 进行精确商运算，应抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    # 创建多项式环 R，变量为 x 和 y，系数为整数 ZZ
    R, x, y = ring("x,y", ZZ)

    # 定义多项式 f 和 g
    f, g = x**2 - y**2, x - y
    # 计算期望的商和余数
    q, r = x + y, R(0)

    # 断言：使用 f 对 g 进行整除运算，应与 divmod(f, g) 和 (q, r) 相等
    assert f.div(g) == divmod(f, g) == (q, r)
    # 断言：使用 f 对 g 进行取余运算，应与 f % g 和 r 相等
    assert f.rem(g) == f % g == r
    # 断言：使用 f 对 g 进行浮点除法运算，应与 f / g 和 q 相等
    assert f.quo(g) == f / g == q
    # 断言：使用 f 对 g 进行精确商运算，应返回 q
    assert f.exquo(g) == q

    # 重新定义多项式 f 和 g
    f, g = x**2 + y**2, x - y
    # 计算期望的商和余数
    q, r = x + y, 2 * y**2

    # 断言：使用 f 对 g 进行整除运算，应与 divmod(f, g) 和 (q, r) 相等
    assert f.div(g) == divmod(f, g) == (q, r)
    # 断言：使用 f 对 g 进行取余运算，应与 f % g 和 r 相等
    assert f.rem(g) == f % g == r
    # 断言：使用 f 对 g 进行浮点除法运算，应与 f / g 和 q 相等
    assert f.quo(g) == f / g == q
    # 断言：使用 f 对 g 进行精确商运算，应抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    # 重新定义多项式 f 和 g
    f, g = x**2 + y**2, -x + y
    # 计算期望的商和余数
    q, r = -x - y, 2 * y**2

    # 断言：使用 f 对 g 进行整除运算，应与 divmod(f, g) 和 (q, r) 相等
    assert f.div(g) == divmod(f, g) == (q, r)
    # 断言：使用 f 对 g 进行取余运算，应与 f % g 和 r 相等
    assert f.rem(g) == f % g == r
    # 断言：使用 f 对 g 进行浮点除法运算，应与 f / g 和 q 相等
    assert f.quo(g) == f / g == q
    # 断言：使用 f 对 g 进行精确商运算，应抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    # 重新定义多项式 f 和 g
    f, g = x**2 + y**2, 2 * x - 2 * y
    # 计算期望的商和余数
    q, r = R(0), f

    # 断言：使用 f 对 g 进行整除运算，应与 divmod(f, g) 和 (q, r) 相等
    assert f.div(g) == divmod(f, g) == (q, r)
    # 断言：使用 f 对 g 进行取余运算，应与 f % g 和 r 相等
    assert f.rem(g) == f % g == r
    # 断言：使用 f 对 g 进行浮点除法运算，应与 f / g 和 q 相等
    assert f.quo(g) == f / g == q
    # 断言：使用 f 对 g 进行精确商运算，应抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    # 创建多项式环 R，变量为 x 和 y，系数为有理数 QQ
    R, x, y = ring("x,y", QQ)

    # 重新定义多项式 f 和 g
    f, g = x**2 - y**2, x - y
    # 计算期望的商和余数
    q, r = x + y, R(0)

    # 断言：使用 f 对 g 进行整除运算，应与 divmod(f, g) 和 (q, r) 相等
    assert f.div(g) == divmod(f, g) == (q, r)
    # 断言：使用 f 对 g 进行取余运算，应与 f % g 和 r 相等
    assert f.rem(g) == f % g == r
    # 断言：使用 f 对 g 进行浮点除法运算，应与 f / g 和 q 相等
    assert f.quo(g) == f / g == q
    # 断言：使用 f 对 g 进行精确商运算，应返回 q
    assert f.exquo(g) == q

    # 重新定义多项式 f 和 g
    f, g = x**2 + y**2, x - y
    # 计算期望的商和余数
    q, r = x + y, 2 * y**2

    # 断言：使用 f 对 g 进行整除运算，应与 divmod(f, g) 和 (q, r) 相等
    assert f.div(g) == divmod(f, g) == (q, r)
    # 断言：使用 f 对 g 进行取余运算，应与 f % g 和 r 相等
    assert f.rem(g) == f % g == r
    # 断言：使用 f 对 g 进行浮点除法运算，应与 f / g 和 q 相等
    assert f.quo(g) == f / g == q
    # 断言：使用 f 对 g 进行精确商运算，应抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    # 重新定义多项式 f 和 g
    f, g = x**2 + y**2, -x + y
    # 计算期望的商和余数
    q, r = -x - y, 2 * y**2

    # 断言：使用 f 对 g 进行整除运算，应与 divmod(f, g) 和 (q, r) 相等
    assert f.div(g) == divmod(f, g) == (q, r)
    # 断言：使用 f 对 g 进行取余运算，应与 f % g 和 r 相等
    assert f.rem(g) == f % g == r
    # 断言：使用 f 对 g 进行浮点除法运算，应与 f / g 和 q 相等
    assert f.quo(g) == f / g == q
    # 断言：使用 f 对 g 进行精确商运算，应抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    # 重新定义多项式 f 和 g
    f, g = x**2 + y**2, 2 * x - 2 * y
    # 计算期望的商和余数
    q
# 定义一个名为 test_PolyElement___pow__ 的测试函数
def test_PolyElement___pow__():
    # 创建一个多项式环 R 和变量 x
    R, x = ring("x", ZZ, grlex)
    # 创建一个多项式 f = 2*x + 3
    f = 2*x + 3

    # 断言 f 的零次幂等于 1
    assert f**0 == 1
    # 断言 f 的一次幂等于 f 自身
    assert f**1 == f
    # 断言计算 f 的负一次幂会引发 ValueError 异常
    raises(ValueError, lambda: f**(-1))

    # 断言计算 x 的负一次幂等于 x 的负一次幂
    assert x**(-1) == x**(-1)

    # 断言计算 f 的二次幂，并且验证三种方法得到的结果相同
    assert f**2 == f._pow_generic(2) == f._pow_multinomial(2) == 4*x**2 + 12*x + 9
    # 断言计算 f 的三次幂，并且验证三种方法得到的结果相同
    assert f**3 == f._pow_generic(3) == f._pow_multinomial(3) == 8*x**3 + 36*x**2 + 54*x + 27
    # 断言计算 f 的四次幂，并且验证三种方法得到的结果相同
    assert f**4 == f._pow_generic(4) == f._pow_multinomial(4) == 16*x**4 + 96*x**3 + 216*x**2 + 216*x + 81
    # 断言计算 f 的五次幂，并且验证三种方法得到的结果相同
    assert f**5 == f._pow_generic(5) == f._pow_multinomial(5) == 32*x**5 + 240*x**4 + 720*x**3 + 1080*x**2 + 810*x + 243

    # 重新定义多项式环 R 和变量 x, y, z
    R, x, y, z = ring("x,y,z", ZZ, grlex)
    # 创建多项式 f
    f = x**3*y - 2*x*y**2 - 3*z + 1
    # 创建多项式 g
    g = x**6*y**2 - 4*x**4*y**3 - 6*x**3*y*z + 2*x**3*y + 4*x**2*y**4 + 12*x*y**2*z - 4*x*y**2 + 9*z**2 - 6*z + 1

    # 断言计算 f 的二次幂，并且验证三种方法得到的结果相同
    assert f**2 == f._pow_generic(2) == f._pow_multinomial(2) == g

    # 重新定义多项式环 R 和变量 t
    R, t = ring("t", ZZ)
    # 创建多项式 f
    f = -11200*t**4 - 2604*t**2 + 49
    # 创建多项式 g
    g = 15735193600000000*t**16 + 14633730048000000*t**14 + 4828147466240000*t**12 \
      + 598976863027200*t**10 + 3130812416256*t**8 - 2620523775744*t**6 \
      + 92413760096*t**4 - 1225431984*t**2 + 5764801

    # 断言计算 f 的四次幂，并且验证三种方法得到的结果相同
    assert f**4 == f._pow_generic(4) == f._pow_multinomial(4) == g

# 定义一个名为 test_PolyElement_div 的测试函数
def test_PolyElement_div():
    # 创建一个多项式环 R 和变量 x
    R, x = ring("x", ZZ, grlex)

    # 创建多项式 f 和 g
    f = x**3 - 12*x**2 - 42
    g = x - 3

    # 计算 f 除以 g 的商 q 和余数 r
    q = x**2 - 9*x - 27
    r = -123

    # 断言 f 除以 g 的结果与预期相同
    assert f.div([g]) == ([q], r)

    # 重新定义多项式环 R 和变量 x
    R, x = ring("x", ZZ, grlex)
    # 创建多项式 f
    f = x**2 + 2*x + 2
    # 断言 f 除以 [1] 的结果与预期相同
    assert f.div([R(1)]) == ([f], 0)

    # 重新定义多项式环 R 和变量 x
    R, x = ring("x", QQ, grlex)
    # 创建多项式 f
    f = x**2 + 2*x + 2
    # 断言 f 除以 [2] 的结果与预期相同
    assert f.div([R(2)]) == ([QQ(1,2)*x**2 + x + 1], 0)

    # 重新定义多项式环 R 和变量 x, y
    R, x, y = ring("x,y", ZZ, grlex)
    # 创建多项式 f
    f = 4*x**2*y - 2*x*y + 4*x - 2*y + 8

    # 断言 f 除以 [2] 的结果与预期相同
    assert f.div([R(2)]) == ([2*x**2*y - x*y + 2*x - y + 4], 0)
    # 断言 f 除以 [2*y] 的结果与预期相同
    assert f.div([2*y]) == ([2*x**2 - x - 1], 4*x + 8)

    # 创建多项式 f 和 g
    f = x - 1
    g = y - 1

    # 断言 f 除以 g 的结果与预期相同
    assert f.div([g]) == ([0], f)

    # 创建多项式 f 和 G
    f = x*y**2 + 1
    G = [x*y + 1, y + 1]

    # 计算 f 除以 G 的商 Q 和余数 r
    Q = [y, -1]
    r = 2

    # 断言 f 除以 G 的结果与预期相同
    assert f.div(G) == (Q, r)

    # 创建多项式 f 和 G
    f = x**2*y + x*y**2 + y**2
    G = [x*y - 1, y**2 - 1]

    # 计算 f 除以 G 的商 Q 和余数 r
    Q = [x + y, 1]
    r = x + y + 1

    # 断言 f 除以 G 的结果与预期相同
    assert f.div(G) == (Q, r)

    # 创建多项式 f 和 G
    G = [y**2 - 1, x*y - 1]

    # 计算 f 除以 G 的商 Q 和余数 r
    Q = [x + 1, x]
    r = 2*x + 1

    # 断言 f 除以 G 的结果与预期相同
    assert f.div(G) == (Q, r)

    # 重新定义多项式环 R
    R, = ring("", ZZ)
    # 断言 R(3) 除以 R(2) 的结果与预期相同
    assert R(3).div(R(2)) == (0, 3)

    # 重新定义多项式环 R
    R, = ring("", QQ)
    # 断言 R(3) 除以 R(2) 的结果与预期相同
    assert R(3).div(R(2)) == (QQ(3, 2), 0)

# 定义一个名为 test_PolyElement_rem 的测试函数
def test_PolyElement_rem():
    # 创建一个多项式环 R 和变量 x
    R, x = ring("x", ZZ, grlex)

    # 创建多项式 f 和 g
    f = x**3 - 12*x**2 - 42
    g = x - 3
    r = -123

    # 断言 f 除以 g 的余数与预期相同
    assert f.rem([g]) == f.div([g])[1] == r

    # 重新定义多项式环
    # 断言：f.rem(G) 的结果应该等于 f.div(G)[1]，同时也等于 r
    assert f.rem(G) == f.div(G)[1] == r
# 定义一个测试函数，用于测试多项式元素的deflate方法
def test_PolyElement_deflate():
    # 创建一个整数多项式环R，并定义变量x
    R, x = ring("x", ZZ)

    # 断言语句，验证2*x**2对x**4 + 4*x**2 + 1的deflate结果
    assert (2*x**2).deflate(x**4 + 4*x**2 + 1) == ((2,), [2*x, x**2 + 4*x + 1])

    # 创建一个整数多项式环R，并定义变量x和y
    R, x, y = ring("x,y", ZZ)

    # 断言语句，验证R(0)对R(0)的deflate结果
    assert R(0).deflate(R(0)) == ((1, 1), [0, 0])
    # 断言语句，验证R(1)对R(0)的deflate结果
    assert R(1).deflate(R(0)) == ((1, 1), [1, 0])
    # 断言语句，验证R(1)对R(2)的deflate结果
    assert R(1).deflate(R(2)) == ((1, 1), [1, 2])
    # 断言语句，验证R(1)对2*y的deflate结果
    assert R(1).deflate(2*y) == ((1, 1), [1, 2*y])
    # 断言语句，验证2*y对2*y的deflate结果
    assert (2*y).deflate(2*y) == ((1, 1), [2*y, 2*y])
    # 断言语句，验证R(2)对2*y**2的deflate结果
    assert R(2).deflate(2*y**2) == ((1, 2), [2, 2*y])
    # 断言语句，验证2*y**2对2*y**2的deflate结果
    assert (2*y**2).deflate(2*y**2) == ((1, 2), [2*y, 2*y])

    # 定义多项式f和g
    f = x**4*y**2 + x**2*y + 1
    g = x**2*y**3 + x**2*y + 1

    # 断言语句，验证f对g的deflate结果
    assert f.deflate(g) == ((2, 1), [x**2*y**2 + x*y + 1, x*y**3 + x*y + 1])

# 定义一个测试函数，用于测试多项式元素的clear_denoms方法
def test_PolyElement_clear_denoms():
    # 创建有理数多项式环R，并定义变量x和y
    R, x, y = ring("x,y", QQ)

    # 断言语句，验证R(1)的clear_denoms结果
    assert R(1).clear_denoms() == (ZZ(1), 1)
    # 断言语句，验证R(7)的clear_denoms结果
    assert R(7).clear_denoms() == (ZZ(1), 7)

    # 断言语句，验证R(QQ(7,3))的clear_denoms结果
    assert R(QQ(7,3)).clear_denoms() == (3, 7)
    # 断言语句，验证R(QQ(7,3))的clear_denoms结果
    assert R(QQ(7,3)).clear_denoms() == (3, 7)

    # 断言语句，验证(3*x**2 + x)的clear_denoms结果
    assert (3*x**2 + x).clear_denoms() == (1, 3*x**2 + x)
    # 断言语句，验证(x**2 + QQ(1,2)*x)的clear_denoms结果
    assert (x**2 + QQ(1,2)*x).clear_denoms() == (2, 2*x**2 + x)

    # 创建带字典序的有理数多项式环rQQ，并定义变量x和t
    rQQ, x, t = ring("x,t", QQ, lex)
    # 创建带字典序的整数多项式环rZZ，并定义变量X和T
    rZZ, X, T = ring("x,t", ZZ, lex)
    F = [x - QQ(17824537287975195925064602467992950991718052713078834557692023531499318507213727406844943097,413954288007559433755329699713866804710749652268151059918115348815925474842910720000)*t**7
           - QQ(4882321164854282623427463828745855894130208215961904469205260756604820743234704900167747753,12936071500236232304854053116058337647210926633379720622441104650497671088840960000)*t**6
           - QQ(36398103304520066098365558157422127347455927422509913596393052633155821154626830576085097433,25872143000472464609708106232116675294421853266759441244882209300995342177681920000)*t**5
           - QQ(168108082231614049052707339295479262031324376786405372698857619250210703675982492356828810819,58212321751063045371843239022262519412449169850208742800984970927239519899784320000)*t**4
           - QQ(5694176899498574510667890423110567593477487855183144378347226247962949388653159751849449037,1617008937529529038106756639507292205901365829172465077805138081312208886105120000)*t**3
           - QQ(154482622347268833757819824809033388503591365487934245386958884099214649755244381307907779,60637835157357338929003373981523457721301218593967440417692678049207833228942000)*t**2
           - QQ(2452813096069528207645703151222478123259511586701148682951852876484544822947007791153163,2425513406294293557160134959260938308852048743758697616707707121968313329157680)*t
           - QQ(34305265428126440542854669008203683099323146152358231964773310260498715579162112959703,202126117191191129763344579938411525737670728646558134725642260164026110763140),
         t**8 + QQ(693749860237914515552,67859264524169150569)*t**7
              + QQ(27761407182086143225024,610733380717522355121)*t**6
              + QQ(7785127652157884044288,67859264524169150569)*t**5
              + QQ(36567075214771261409792,203577793572507451707)*t**4
              + QQ(36336335165196147384320,203577793572507451707)*t**3
              + QQ(7452455676042754048000,67859264524169150569)*t**2
              + QQ(2593331082514399232000,67859264524169150569)*t
              + QQ(390399197427343360000,67859264524169150569)]



# 定义多项式 F，其包含两个列表：第一个列表是关于变量 t 的多项式项的系数；第二个列表是 t 的幂次从 0 到 8 的项的系数
F = [
    x - QQ(17824537287975195925064602467992950991718052713078834557692023531499318507213727406844943097, 413954288007559433755329699713866804710749652268151059918115348815925474842910720000) * t**7
    - QQ(4882321164854282623427463828745855894130208215961904469205260756604820743234704900167747753, 12936071500236232304854053116058337647210926633379720622441104650497671088840960000) * t**6
    - QQ(36398103304520066098365558157422127347455927422509913596393052633155821154626830576085097433, 25872143000472464609708106232116675294421853266759441244882209300995342177681920000) * t**5
    - QQ(168108082231614049052707339295479262031324376786405372698857619250210703675982492356828810819, 58212321751063045371843239022262519412449169850208742800984970927239519899784320000) * t**4
    - QQ(5694176899498574510667890423110567593477487855183144378347226247962949388653159751849449037, 1617008937529529038106756639507292205901365829172465077805138081312208886105120000) * t**3
    - QQ(154482622347268833757819824809033388503591365487934245386958884099214649755244381307907779, 60637835157357338929003373981523457721301218593967440417692678049207833228942000) * t**2
    - QQ(2452813096069528207645703151222478123259511586701148682951852876484544822947007791153163, 2425513406294293557160134959260938308852048743758697616707707121968313329157680) * t
    - QQ(34305265428126440542854669008203683099323146152358231964773310260498715579162112959703, 202126117191191129763344579938411525737670728646558134725642260164026110763140)
    ,
    t**8 + QQ(693749860237914515552, 67859264524169150569) * t**7
    + QQ(27761407182086143225024, 610733380717522355121) * t**6
    + QQ(7785127652157884044288, 67859264524169150569) * t**5
    + QQ(36567075214771261409792, 203577793572507451707) * t**4
    + QQ(36336335165196147384320, 203577793572507451707) * t**3
    + QQ(7452455676042754048000, 67859264524169150569) * t**2
    + QQ(2593331082514399232000, 67859264524169150569) * t
    + QQ(390399197427343360000, 67859264524169150569)
]
    G = [3725588592068034903797967297424801242396746870413359539263038139343329273586196480000*X -
         160420835591776763325581422211936558925462474417709511019228211783493866564923546661604487873*T**7 -
         1406108495478033395547109582678806497509499966197028487131115097902188374051595011248311352864*T**6 -
         5241326875850889518164640374668786338033653548841427557880599579174438246266263602956254030352*T**5 -
         10758917262823299139373269714910672770004760114329943852726887632013485035262879510837043892416*T**4 -
         13119383576444715672578819534846747735372132018341964647712009275306635391456880068261130581248*T**3 -
         9491412317016197146080450036267011389660653495578680036574753839055748080962214787557853941760*T**2 -
         3767520915562795326943800040277726397326609797172964377014046018280260848046603967211258368000*T -
         632314652371226552085897259159210286886724229880266931574701654721512325555116066073245696000,
         610733380717522355121*T**8 +
         6243748742141230639968*T**7 +
         27761407182086143225024*T**6 +
         70066148869420956398592*T**5 +
         109701225644313784229376*T**4 +
         109009005495588442152960*T**3 +
         67072101084384786432000*T**2 +
         23339979742629593088000*T +
         3513592776846090240000]


# 断言语句，验证列表 F 中每个多项式经过清除有理系数后，其系数仍在整数环中，与 G 列表相等
assert [ f.clear_denoms()[1].set_ring(rZZ) for f in F ] == G
# 定义一个测试函数 test_PolyElement_cofactors，用于测试多项式元素的 cofactors 方法
def test_PolyElement_cofactors():
    # 创建多项式环 R，并定义变量 x, y
    R, x, y = ring("x,y", ZZ)

    # 初始化两个多项式 f, g 分别为 0
    f, g = R(0), R(0)
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (0, 0, 0)
    assert f.cofactors(g) == (0, 0, 0)

    # 初始化 f 为 2，g 为 0
    f, g = R(2), R(0)
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (2, 1, 0)
    assert f.cofactors(g) == (2, 1, 0)

    # 初始化 f 为 -2，g 为 0
    f, g = R(-2), R(0)
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (2, -1, 0)
    assert f.cofactors(g) == (2, -1, 0)

    # 初始化 f 为 0，g 为 -2*x
    f, g = R(0), R(-2)
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (2, 0, -1)
    assert f.cofactors(g) == (2, 0, -1)

    # 初始化 f 为 0，g 为 2*x + 4
    f, g = R(0), 2*x + 4
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (2*x + 4, 0, 1)
    assert f.cofactors(g) == (2*x + 4, 0, 1)

    # 初始化 f 为 2*x + 4，g 为 0
    f, g = 2*x + 4, R(0)
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (2*x + 4, 1, 0)
    assert f.cofactors(g) == (2*x + 4, 1, 0)

    # 初始化 f 和 g 为 2
    f, g = R(2), R(2)
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (2, 1, 1)
    assert f.cofactors(g) == (2, 1, 1)

    # 初始化 f 为 -2，g 为 2
    f, g = R(-2), R(2)
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (2, -1, 1)
    assert f.cofactors(g) == (2, -1, 1)

    # 初始化 f 和 g 为 2
    f, g = R(2), R(-2)
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (2, 1, -1)
    assert f.cofactors(g) == (2, 1, -1)

    # 初始化 f 和 g 为 -2
    f, g = R(-2), R(-2)
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (2, -1, -1)
    assert f.cofactors(g) == (2, -1, -1)

    # 初始化 f 为 x^2 + 2*x + 1，g 为 1
    f, g = x**2 + 2*x + 1, R(1)
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (1, x^2 + 2*x + 1, 1)
    assert f.cofactors(g) == (1, x**2 + 2*x + 1, 1)

    # 初始化 f 为 x^2 + 2*x + 1，g 为 2
    f, g = x**2 + 2*x + 1, R(2)
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (1, x^2 + 2*x + 1, 2)
    assert f.cofactors(g) == (1, x**2 + 2*x + 1, 2)

    # 初始化 f 为 2*x^2 + 4*x + 2，g 为 2
    f, g = 2*x**2 + 4*x + 2, R(2)
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (2, x^2 + 2*x + 1, 1)
    assert f.cofactors(g) == (2, x**2 + 2*x + 1, 1)

    # 初始化 f 为 2，g 为 2*x^2 + 4*x + 2
    f, g = R(2), 2*x**2 + 4*x + 2
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (2, 1, x^2 + 2*x + 1)
    assert f.cofactors(g) == (2, 1, x**2 + 2*x + 1)

    # 初始化 f 为 2*x^2 + 4*x + 2，g 为 x + 1
    f, g = 2*x**2 + 4*x + 2, x + 1
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (x + 1, 2*x + 2, 1)
    assert f.cofactors(g) == (x + 1, 2*x + 2, 1)

    # 初始化 f 为 x + 1，g 为 2*x^2 + 4*x + 2
    f, g = x + 1, 2*x**2 + 4*x + 2
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (x + 1, 1, 2*x + 2)
    assert f.cofactors(g) == (x + 1, 1, 2*x + 2)

    # 创建多项式环 R，并定义变量 x, y, z, t
    R, x, y, z, t = ring("x,y,z,t", ZZ)

    # 初始化 f 为 t^2 + 2*t + 1，g 为 2*t + 2
    f, g = t**2 + 2*t + 1, 2*t + 2
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (t + 1, t + 1, 2)
    assert f.cofactors(g) == (t + 1, t + 1, 2)

    # 初始化 f 为 z^2*t^2 + 2*z^2*t + z^2 + z*t + z，g 为 t^2 + 2*t + 1
    f, g = z**2*t**2 + 2*z**2*t + z**2 + z*t + z, t**2 + 2*t + 1
    # 定义期望的结果 h, cff, cfg
    h, cff, cfg = t + 1, z**2*t + z**2 + z, t + 1
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (h, cff, cfg)
    assert f.cofactors(g) == (h, cff, cfg)
    # 断言 g 和 f 的 cofactors 方法返回期望的结果 (h, cfg, cff)
    assert g.cofactors(f) == (h, cfg, cff)

    # 创建多项式环 R，并定义变量 x, y
    R, x, y = ring("x,y", QQ)

    # 初始化 f 为 1/2*x^2 + x + 1/2，g 为 1/2*x + 1/2
    f = QQ(1,2)*x**2 + x + QQ(1,2)
    g = QQ(1,2)*x + QQ(1,2)
    # 定义期望的结果 h
    h = x + 1
    # 断言 f 和 g 的 cofactors 方法返回期望的结果 (h, g, 1/2)
    assert f.cofactors(g) == (h, g, QQ(1,2))
    # 断言 g 和 f 的 cofactors 方法返回期望的结果 (h, 1/2, g)
    assert g.cofactors(f) == (h, QQ(1,2), g)

    # 创建多项式环
    # 断言，验证R(0)对象的L1范数是否为0
    assert R(0).l1_norm() == 0
    
    # 断言，验证R(1)对象的L1范数是否为1
    assert R(1).l1_norm() == 1
    
    # 断言，验证多项式x**3 + 4*x**2 + 2*x + 3的L1范数是否为10
    assert (x**3 + 4*x**2 + 2*x + 3).l1_norm() == 10
def test_PolyElement_diff():
    # 创建多项式环和变量环
    R, X = xring("x:11", QQ)
    
    # 定义多项式 f
    f = QQ(288,5)*X[0]**8*X[1]**6*X[4]**3*X[10]**2 + 8*X[0]**2*X[2]**3*X[4]**3 + 2*X[0]**2 - 2*X[1]**2
    
    # 断言对 X[0] 求偏导数
    assert f.diff(X[0]) == QQ(2304,5)*X[0]**7*X[1]**6*X[4]**3*X[10]**2 + 16*X[0]*X[2]**3*X[4]**3 + 4*X[0]
    
    # 断言对 X[4] 求偏导数
    assert f.diff(X[4]) == QQ(864,5)*X[0]**8*X[1]**6*X[4]**2*X[10]**2 + 24*X[0]**2*X[2]**3*X[4]**2
    
    # 断言对 X[10] 求偏导数
    assert f.diff(X[10]) == QQ(576,5)*X[0]**8*X[1]**6*X[4]**3*X[10]

def test_PolyElement___call__():
    # 创建多项式环和单变量 x
    R, x = ring("x", ZZ)
    
    # 定义多项式 f = 3*x + 1
    f = 3*x + 1
    
    # 断言 f(0) 的结果
    assert f(0) == 1
    
    # 断言 f(1) 的结果
    assert f(1) == 4
    
    # 引发 ValueError 的 lambda 断言
    raises(ValueError, lambda: f())
    
    # 引发 ValueError 的 lambda 断言（传递多个参数）
    raises(ValueError, lambda: f(0, 1))
    
    # 引发 CoercionFailed 的 lambda 断言
    raises(CoercionFailed, lambda: f(QQ(1,7)))
    
    # 创建多项式环和双变量 x, y
    R, x, y = ring("x,y", ZZ)
    
    # 定义多项式 f = 3*x + y**2 + 1
    f = 3*x + y**2 + 1
    
    # 断言 f(0, 0) 的结果
    assert f(0, 0) == 1
    
    # 断言 f(1, 7) 的结果
    assert f(1, 7) == 53
    
    # 在去除变量 x 后的环 Ry 上断言 f(0) 的结果
    Ry = R.drop(x)
    assert f(0) == Ry.y**2 + 1
    
    # 在去除变量 x 后的环 Ry 上断言 f(1) 的结果
    assert f(1) == Ry.y**2 + 4
    
    # 引发 ValueError 的 lambda 断言
    raises(ValueError, lambda: f())
    
    # 引发 ValueError 的 lambda 断言（传递多个参数）
    raises(ValueError, lambda: f(0, 1, 2))
    
    # 引发 CoercionFailed 的 lambda 断言
    raises(CoercionFailed, lambda: f(1, QQ(1,7)))
    
    # 引发 CoercionFailed 的 lambda 断言
    raises(CoercionFailed, lambda: f(QQ(1,7), 1))
    
    # 引发 CoercionFailed 的 lambda 断言
    raises(CoercionFailed, lambda: f(QQ(1,7), QQ(1,7)))

def test_PolyElement_evaluate():
    # 创建多项式环和单变量 x
    R, x = ring("x", ZZ)
    
    # 定义多项式 f = x**3 + 4*x**2 + 2*x + 3
    f = x**3 + 4*x**2 + 2*x + 3
    
    # 断言 f 在 x=0 处的值
    r = f.evaluate(x, 0)
    assert r == 3 and not isinstance(r, PolyElement)
    
    # 引发 CoercionFailed 的 lambda 断言
    raises(CoercionFailed, lambda: f.evaluate(x, QQ(1,7)))
    
    # 创建多项式环和三变量 x, y, z
    R, x, y, z = ring("x,y,z", ZZ)
    
    # 定义多项式 f = (x*y)**3 + 4*(x*y)**2 + 2*x*y + 3
    
    # 断言 f 在 x=0 处的值
    r = f.evaluate(x, 0)
    assert r == 3 and isinstance(r, R.drop(x).dtype)
    
    # 断言 f 在 [(x,0), (y,0)] 处的值
    r = f.evaluate([(x, 0), (y, 0)])
    assert r == 3 and isinstance(r, R.drop(x, y).dtype)
    
    # 断言 f 在 y=0 处的值
    r = f.evaluate(y, 0)
    assert r == 3 and isinstance(r, R.drop(y).dtype)
    
    # 断言 f 在 [(y,0), (x,0)] 处的值
    r = f.evaluate([(y, 0), (x, 0)])
    assert r == 3 and isinstance(r, R.drop(y, x).dtype)
    
    # 断言 f 在 [(x,0), (y,0), (z,0)] 处的值
    r = f.evaluate([(x, 0), (y, 0), (z, 0)])
    assert r == 3 and not isinstance(r, PolyElement)
    
    # 引发 CoercionFailed 的 lambda 断言
    raises(CoercionFailed, lambda: f.evaluate([(x, 1), (y, QQ(1,7))]))
    
    # 引发 CoercionFailed 的 lambda 断言
    raises(CoercionFailed, lambda: f.evaluate([(x, QQ(1,7)), (y, 1)]))
    
    # 引发 CoercionFailed 的 lambda 断言
    raises(CoercionFailed, lambda: f.evaluate([(x, QQ(1,7)), (y, QQ(1,7))]))

def test_PolyElement_subs():
    # 创建多项式环和单变量 x
    R, x = ring("x", ZZ)
    
    # 定义多项式 f = x**3 + 4*x**2 + 2*x + 3
    f = x**3 + 4*x**2 + 2*x + 3
    
    # 断言 f 在 x=0 处的值
    r = f.subs(x, 0)
    assert r == 3 and isinstance(r, R.dtype)
    
    # 引发 CoercionFailed 的 lambda 断言
    raises(CoercionFailed, lambda: f.subs(x, QQ(1,7)))
    
    # 创建多项式环和三变量 x, y, z
    R, x, y, z = ring("x,y,z", ZZ)
    
    # 定义多项式 f = x**3 + 4*x**2 + 2*x + 3
    
    # 断言 f 在 x=0 处的值
    r = f.subs(x, 0)
    assert r == 3 and isinstance(r, R.dtype)
    
    # 断言 f 在 [(x,0), (y,0)] 处的值
    r = f.subs([(x, 0), (y, 0)])
    assert r == 3 and isinstance(r, R.dtype)
    
    # 引发 CoercionFailed 的 lambda 断言
    raises(CoercionFailed, lambda: f.subs([(x, 1), (y, QQ(1,7))]))
    
    # 引发 CoercionFailed 的 lambda 断言
    raises(CoercionFailed, lambda: f.subs([(x, QQ(1,7)), (y, 1)]))
    
    # 引发 CoercionFailed 的 lambda 断言
    raises(CoercionFailed, lambda: f.subs([(x, QQ(1,7)), (y, QQ(1,7))]))

def test_PolyElement_symmetrize():
    # 创建多项式环和双变量 x, y
    R, x, y = ring("x,y", ZZ)
    
    # 定义对称的齐次多项式 f = x**2 + y**2
    
    # 断言齐次对称多项式的对称化结果
    f = x**2 + y**
    # 确保余数不为零
    assert rem != 0

    # 断言对称化后的多项式与原多项式的组合结果等于原多项式
    assert sym.compose(m) + rem == f

    # 不均匀的对称多项式
    f = x*y + 7
    # 对多项式进行对称化操作，返回对称化的结果(sym)、余数(rem)和变换(m)
    sym, rem, m = f.symmetrize()
    # 确保余数为零
    assert rem == 0
    # 断言对称化后的多项式与原多项式的组合结果等于原多项式
    assert sym.compose(m) + rem == f

    # 不均匀的非对称多项式
    f = y + 7
    # 对多项式进行对称化操作，返回对称化的结果(sym)、余数(rem)和变换(m)
    sym, rem, m = f.symmetrize()
    # 确保余数不为零
    assert rem != 0
    # 断言对称化后的多项式与原多项式的组合结果等于原多项式
    assert sym.compose(m) + rem == f

    # 常数多项式
    f = R.from_expr(3)
    # 对多项式进行对称化操作，返回对称化的结果(sym)、余数(rem)和变换(m)
    sym, rem, m = f.symmetrize()
    # 确保余数为零
    assert rem == 0
    # 断言对称化后的多项式与原多项式的组合结果等于原多项式
    assert sym.compose(m) + rem == f

    # 从字符串构造的常数多项式
    R, f = sring(3)
    # 对多项式进行对称化操作，返回对称化的结果(sym)、余数(rem)和变换(m)
    sym, rem, m = f.symmetrize()
    # 确保余数为零
    assert rem == 0
    # 断言对称化后的多项式与原多项式的组合结果等于原多项式
    assert sym.compose(m) + rem == f
# 定义测试函数 test_PolyElement_compose
def test_PolyElement_compose():
    # 创建一个多项式环 R 和变量 x
    R, x = ring("x", ZZ)
    # 定义多项式 f = x^3 + 4x^2 + 2x + 3
    f = x**3 + 4*x**2 + 2*x + 3

    # 对 f 在 x=0 处进行复合运算，预期结果为常数 3，且类型为 R.dtype
    r = f.compose(x, 0)
    assert r == 3 and isinstance(r, R.dtype)

    # 断言 f 在 x=x 处的复合运算结果等于 f 本身
    assert f.compose(x, x) == f
    # 断言 f 在 x=x^2 处的复合运算结果为 x^6 + 4x^4 + 2x^2 + 3
    assert f.compose(x, x**2) == x**6 + 4*x**4 + 2*x**2 + 3

    # 断言在不支持的环境下进行复合运算会引发 CoercionFailed 异常
    raises(CoercionFailed, lambda: f.compose(x, QQ(1,7)))

    # 重新定义多项式环 R 和多个变量 x, y, z
    R, x, y, z = ring("x,y,z", ZZ)
    # 重新定义多项式 f = x^3 + 4x^2 + 2x + 3
    f = x**3 + 4*x**2 + 2*x + 3

    # 对 f 在 x=0 处进行复合运算，预期结果为常数 3，且类型为 R.dtype
    r = f.compose(x, 0)
    assert r == 3 and isinstance(r, R.dtype)
    # 对 f 在 x=0, y=0 处进行复合运算，预期结果为常数 3，且类型为 R.dtype
    r = f.compose([(x, 0), (y, 0)])
    assert r == 3 and isinstance(r, R.dtype)

    # 对 (x^3 + 4x^2 + 2xyz + 3) 在 x=y*z^2 - 1 处进行复合运算
    r = (x**3 + 4*x**2 + 2*x*y*z + 3).compose(x, y*z**2 - 1)
    # 预期结果为 (y*z^2 - 1)^3 + 4*(y*z^2 - 1)^2 + 2*(y*z^2 - 1)*yz + 3
    q = (y*z**2 - 1)**3 + 4*(y*z**2 - 1)**2 + 2*(y*z**2 - 1)*y*z + 3
    assert r == q and isinstance(r, R.dtype)

# 定义测试函数 test_PolyElement_is_
def test_PolyElement_is_():
    # 创建多项式环 R 和多个变量 x, y, z，并指定为有理数域 QQ
    R, x, y, z = ring("x,y,z", QQ)

    # 断言 (x - x) 的属性检查：不是生成元、是常数项、是单项式、是项
    assert (x - x).is_generator == False
    assert (x - x).is_ground == True
    assert (x - x).is_monomial == True
    assert (x - x).is_term == True

    # 断言 (x - x + 1) 的属性检查：不是生成元、是常数项、是单项式、不是项
    assert (x - x + 1).is_generator == False
    assert (x - x + 1).is_ground == True
    assert (x - x + 1).is_monomial == True
    assert (x - x + 1).is_term == True

    # 断言 x 的属性检查：是生成元、不是常数项、是单项式、是项
    assert x.is_generator == True
    assert x.is_ground == False
    assert x.is_monomial == True
    assert x.is_term == True

    # 断言 (x*y) 的属性检查：不是生成元、不是常数项、是单项式、是项
    assert (x*y).is_generator == False
    assert (x*y).is_ground == False
    assert (x*y).is_monomial == True
    assert (x*y).is_term == True

    # 断言 (3*x) 的属性检查：不是生成元、不是常数项、不是单项式、是项
    assert (3*x).is_generator == False
    assert (3*x).is_ground == False
    assert (3*x).is_monomial == False
    assert (3*x).is_term == True

    # 断言 (3*x + 1) 的属性检查：不是生成元、不是常数项、不是单项式、不是项
    assert (3*x + 1).is_generator == False
    assert (3*x + 1).is_ground == False
    assert (3*x + 1).is_monomial == False
    assert (3*x + 1).is_term == False

    # 断言 R(0) 的属性检查：是零元
    assert R(0).is_zero is True
    assert R(1).is_zero is False

    # 断言 R(0) 的属性检查：不是单位元
    assert R(0).is_one is False
    assert R(1).is_one is True

    # 断言 (x - 1) 的属性检查：是首一多项式
    assert (x - 1).is_monic is True
    assert (2*x - 1).is_monic is False

    # 断言 (3*x + 2) 的属性检查：是原始多项式
    assert (3*x + 2).is_primitive is True
    assert (4*x + 2).is_primitive is False

    # 断言 (x + y + z + 1) 的属性检查：是线性多项式
    assert (x + y + z + 1).is_linear is True
    assert (x*y*z + 1).is_linear is False

    # 断言 (x*y + z + 1) 的属性检查：是二次多项式
    assert (x*y + z + 1).is_quadratic is True
    assert (x*y*z + 1).is_quadratic is False

    # 断言 (x - 1) 的属性检查：是无平方因子的多项式
    assert (x - 1).is_squarefree is True
    assert ((x - 1)**2).is_squarefree is False

    # 断言 (x^2 + x + 1) 的属性检查：是不可约多项式
    assert (x**2 + x + 1).is_irreducible is True
    assert (x**2 + 2*x + 1).is_irreducible is False

    # 创建一个有限域 FF(11) 上的单变量环
    _, t = ring("t", FF(11))

    # 断言 (7*t + 3) 的属性检查：是不可约多项式
    assert (7*t + 3).is_irreducible is True
    assert (7*t**2 + 3*t + 1).is_irreducible is False

    # 创建整数环上的单变量环
    _, u = ring("u", ZZ)
    # 定义多项式 f = u^16 + u^14 - u^10 - u^8 - u^6 + u^2

    f = u**16 + u**14 - u**10 - u**8 - u**6 + u**2

    # 断言 f 的属性检查：不是旋转多项式
    assert f.is_cyclotomic is False
    # 断言 (f + 1) 的属性检查：是旋转多项式
    assert (f + 1).is_cyclotomic is True

    # 在多变量环中调用 is_cyclotomic 将引发 MultivariatePolynomialError 异常
    raises(MultivariatePolynomialError, lambda: x.is_cyclotomic)

    # 创建一个空字符串环 R
    R, = ring("", ZZ)
    # 断言 R(4) 的属性检查：是无平方因子的多项式
    assert R(4).is_squarefree is True
    # 断言 R(6) 的属性检查：是不
    # 使用断言来验证 R(1) 对象的操作结果类型不是 R.dtype 类型
    assert isinstance(R(1).drop(0).drop(0).drop(0), R.dtype) is False
    
    # 使用 raises 函数来验证在连续调用 z.drop(0).drop(0).drop(0) 时是否会引发 ValueError 异常
    raises(ValueError, lambda: z.drop(0).drop(0).drop(0))
    
    # 使用 raises 函数来验证在调用 x.drop(0) 时是否会引发 ValueError 异常
    raises(ValueError, lambda: x.drop(0))
# 定义测试函数 test_PolyElement_coeff_wrt，用于测试 PolyElement 类的 coeff_wrt 方法
def test_PolyElement_coeff_wrt():
    # 创建整数环 R 和变量 x, y, z
    R, x, y, z = ring("x, y, z", ZZ)

    # 定义多项式 p
    p = 4*x**3 + 5*y**2 + 6*y**2*z + 7
    # 断言 p 关于 (1, 2) 生成元的系数为 6*z + 5
    assert p.coeff_wrt(1, 2) == 6*z + 5 # using generator index
    # 断言 p 关于 x 生成元的系数为 4
    assert p.coeff_wrt(x, 3) == 4 # using generator

    # 重新定义多项式 p
    p = 2*x**4 + 3*x*y**2*z + 10*y**2 + 10*x*z**2
    # 断言 p 关于 x 生成元的系数为 3*y**2*z + 10*z**2
    assert p.coeff_wrt(x, 1) == 3*y**2*z + 10*z**2
    # 断言 p 关于 y 生成元的系数为 3*x*z + 10
    assert p.coeff_wrt(y, 2) == 3*x*z + 10

    # 重新定义多项式 p
    p = 4*x**2 + 2*x*y + 5
    # 断言 p 关于 z 生成元的系数为 R(0)（即零元素）
    assert p.coeff_wrt(z, 1) == R(0)
    # 断言 p 关于 y 生成元的系数为 R(0)（即零元素）
    assert p.coeff_wrt(y, 2) == R(0)

# 定义测试函数 test_PolyElement_prem，用于测试 PolyElement 类的 prem 方法
def test_PolyElement_prem():
    # 创建整数环 R 和变量 x, y
    R, x, y = ring("x, y", ZZ)

    # 定义多项式 f 和 g
    f, g = x**2 + x*y, 2*x + 2
    # 断言 f 除以 g 的余式为 -4*y + 4
    assert f.prem(g) == -4*y + 4 # first generator is chosen by default

    # 重新定义多项式 f 和 g
    f, g = x**2 + 1, 2*x - 4
    # 断言 f 除以 g 的余式为 20，并且传入 x 生成元的余式也为 20
    assert f.prem(g) == f.prem(g, x) == 20
    # 断言 f 除以 g 的余式为 R(0)（即零元素）
    assert f.prem(g, 1) == R(0)

    # 重新定义多项式 f 和 g
    f, g = x*y + 2*x + 1, x + y
    # 断言 f 除以 g 的余式为 -y**2 - 2*y + 1
    assert f.prem(g) == -y**2 - 2*y + 1
    # 断言 f 除以 g 的余式为 -x**2 + 2*x + 1，并且传入 y 生成元的余式也为 -x**2 + 2*x + 1
    assert f.prem(g, 1) == f.prem(g, y) == -x**2 + 2*x + 1

    # 检查当 f = 0 时会抛出 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: f.prem(R(0)))

# 定义测试函数 test_PolyElement_pdiv，用于测试 PolyElement 类的 pdiv 方法
def test_PolyElement_pdiv():
    # 创建整数环 R 和变量 x, y
    R, x, y = ring("x,y", ZZ)

    # 定义多项式 f 和 g
    f, g = x**4 + 5*x**3 + 7*x**2, 2*x**2 + 3
    # 断言 f 除以 g 的商为 4*x**2 + 20*x + 22，余为 -60*x - 66
    assert f.pdiv(g) == f.pdiv(g, x) == (4*x**2 + 20*x + 22, -60*x - 66)

    # 重新定义多项式 f 和 g
    f, g = x**2 - y**2, x - y
    # 断言 f 除以 g 的商为 x + y，余为 0
    assert f.pdiv(g) == f.pdiv(g, 0) == (x + y, 0)

    # 重新定义多项式 f 和 g
    f, g = x*y + 2*x + 1, x + y
    # 断言 f 除以 g 的商为 y + 2，余为 -y**2 - 2*y + 1
    assert f.pdiv(g) == (y + 2, -y**2 - 2*y + 1)
    # 断言 f 除以 g 的商为 x + 1，余为 -x**2 + 2*x + 1，并且传入 y 生成元的结果也是一样的
    assert f.pdiv(g, y) == f.pdiv(g, 1) == (x + 1, -x**2 + 2*x + 1)

    # 检查当 f = 0 时会抛出 ZeroDivisionError 异常
    raises(ZeroDivisionError, lambda: f.prem(R(0)))

# 定义测试函数 test_PolyElement_pquo，用于测试 PolyElement 类的 pquo 方法
def test_PolyElement_pquo():
    # 创建整数环 R 和变量 x, y
    R, x, y = ring("x, y", ZZ)

    # 定义多项式 f 和 g
    f, g = x**4 - 4*x**2*y + 4*y**2, x**2 - 2*y
    # 断言 f 除以 g 的商为 x**2 - 2*y，并且传入 x 生成元的商也是一样的
    assert f.pquo(g) == f.pquo(g, x) == x**2 - 2*y
    # 断言 f 除以 g 的商为 4*x**2 - 8*y + 4，并且传入 y 生成元的结果也是一样的
    assert f.pquo(g, y) == 4*x**2 - 8*y + 4

    # 重新定义多项式 f 和 g
    f, g = x**4 - y**4, x**2 - y**2
    # 断言 f 除以 g 的商为 x**2 + y**2，并且传入 0 生成元的结果也是一样的
    assert f.pquo(g) == f.pquo(g, 0) == x**2 + y**2

# 定义测试函数 test_PolyElement_pexquo，用于测试 PolyElement 类的 pexquo 方法
def test_PolyElement_pexquo():
    # 创建整数环 R 和变量 x, y
    R, x, y = ring("x, y", ZZ)

    # 定义多项式 f 和 g
    f, g = x**2 - y**2, x - y
    # 断言 f 除以 g 的扩展商为 x + y，并且传入 x 生成元的结果也是一样的
    assert f.pexquo(g) == f.pexquo(g, x) == x + y
    # 断言 f 除以 g 的扩展商为 x + y + 1，并且传入 y 生成元的结果也是一样的
    assert f.pexquo(g, y) == f.pexquo(g, 1) == x + y + 1

    # 重新定义多项式 f 和 g
    f, g = x**2 + 3*x + 6, x + 2
    # 检查当无法整除时，会抛出 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: f.pexquo(g))

# 定义测试函数 test_PolyElement_gcdex，用于测试 PolyElement 类的 gcdex 和 half_gcdex 方法
def test_PolyElement_gcdex():
    # 创建有理数环 QQ 和变量 x
    _, x = ring("x", QQ)
    # 断言：使用 f 的 subresultants 方法计算 f 和 g 的结果，期望结果为 [0, 0]
    assert f.subresultants(g) == [0, 0]
    
    # 定义多项式 f 和 g，它们是相同的多项式 x^2 + x
    f, g = x**2 + x, x**2 + x
    # 断言：使用 f 的 subresultants 方法计算 f 和 g 的结果，期望结果为 [x^2 + x, x^2 + x]
    assert f.subresultants(g) == [x**2 + x, x**2 + x]
def test_PolyElement_resultant():
    _, x = ring("x", ZZ)  # 创建一个整数环的多项式环，并初始化变量 x
    f, g, h = x**2 - 2*x + 1, x**2 - 1, 0  # 初始化多项式 f, g, h

    assert f.resultant(g) == h  # 断言 f 和 g 的 resultant 等于 h

def test_PolyElement_discriminant():
    _, x = ring("x", ZZ)  # 创建一个整数环的多项式环，并初始化变量 x
    f, g = x**3 + 3*x**2 + 9*x - 13, -11664  # 初始化多项式 f 和 g

    assert f.discriminant() == g  # 断言 f 的 discriminant 等于 g

    F, a, b, c = ring("a,b,c", ZZ)  # 创建一个整数环的多项式环，并初始化变量 F, a, b, c
    _, x = ring("x", F)  # 在 F 上创建多项式环，并重新初始化变量 x

    f, g = a*x**2 + b*x + c, b**2 - 4*a*c  # 初始化多项式 f 和 g

    assert f.discriminant() == g  # 断言 f 的 discriminant 等于 g

def test_PolyElement_decompose():
    _, x = ring("x", ZZ)  # 创建一个整数环的多项式环，并初始化变量 x

    f = x**12 + 20*x**10 + 150*x**8 + 500*x**6 + 625*x**4 - 2*x**3 - 10*x + 9  # 初始化多项式 f
    g = x**4 - 2*x + 9  # 初始化多项式 g
    h = x**3 + 5*x  # 初始化多项式 h

    assert g.compose(x, h) == f  # 断言 g 在 x, h 下的 compose 等于 f
    assert f.decompose() == [g, h]  # 断言 f 的 decompose 结果为 [g, h]

def test_PolyElement_shift():
    _, x = ring("x", ZZ)  # 创建一个整数环的多项式环，并初始化变量 x
    assert (x**2 - 2*x + 1).shift(2) == x**2 + 2*x + 1  # 断言多项式的 shift(2) 结果正确
    assert (x**2 - 2*x + 1).shift_list([2]) == x**2 + 2*x + 1  # 断言多项式的 shift_list([2]) 结果正确

    R, x, y = ring("x, y", ZZ)  # 创建一个整数环的多项式环，并初始化变量 R, x, y
    assert (x*y).shift_list([1, 2]) == (x+1)*(y+2)  # 断言多项式 x*y 的 shift_list([1, 2]) 结果正确
    raises(MultivariatePolynomialError, lambda: (x*y).shift(1))  # 断言多项式 x*y 的 shift(1) 引发 MultivariatePolynomialError

def test_PolyElement_sturm():
    F, t = field("t", ZZ)  # 创建一个整数环的多项式域，并初始化变量 F, t
    _, x = ring("x", F)  # 在 F 上创建多项式环，并重新初始化变量 x

    f = 1024/(15625*t**8)*x**5 - 4096/(625*t**8)*x**4 + 32/(15625*t**4)*x**3 - 128/(625*t**4)*x**2 + F(1)/62500*x - F(1)/625
    # 初始化多项式 f

    assert f.sturm() == [
        x**3 - 100*x**2 + t**4/64*x - 25*t**4/16,
        3*x**2 - 200*x + t**4/64,
        (-t**4/96 + F(20000)/9)*x + 25*t**4/18,
        (-9*t**12 - 11520000*t**8 - 3686400000000*t**4)/(576*t**8 - 245760000*t**4 + 26214400000000),
    ]
    # 断言 f 的 Sturm 序列计算结果正确

def test_PolyElement_gff_list():
    _, x = ring("x", ZZ)  # 创建一个整数环的多项式环，并初始化变量 x

    f = x**5 + 2*x**4 - x**3 - 2*x**2  # 初始化多项式 f
    assert f.gff_list() == [(x, 1), (x + 2, 4)]  # 断言 f 的 gff_list 结果正确

    f = x*(x - 1)**3*(x - 2)**2*(x - 4)**2*(x - 5)  # 初始化另一个多项式 f
    assert f.gff_list() == [(x**2 - 5*x + 4, 1), (x**2 - 5*x + 4, 2), (x, 3)]
    # 断言 f 的 gff_list 结果正确

def test_PolyElement_norm():
    k = QQ  # 初始化有理数域
    K = QQ.algebraic_field(sqrt(2))  # 创建一个含有平方根 2 的代数数域
    sqrt2 = K.unit  # 获取代数数域的单位元素
    _, X, Y = ring("x,y", k)  # 创建一个有理数域上的多项式环，并初始化变量 X, Y
    _, x, y = ring("x,y", K)  # 在代数数域 K 上创建多项式环，并重新初始化变量 x, y

    assert (x*y + sqrt2).norm() == X**2*Y**2 - 2
    # 断言多项式 x*y + sqrt2 的 norm 结果正确

def test_PolyElement_sqf_norm():
    R, x = ring("x", QQ.algebraic_field(sqrt(3)))  # 创建一个含有平方根 3 的有理数域上的多项式环，并初始化变量 x
    X = R.to_ground().x  # 将 R 转换为其底部的多项式环，并获取变量 X

    assert (x**2 - 2).sqf_norm() == ([1], x**2 - 2*sqrt(3)*x + 1, X**4 - 10*X**2 + 1)
    # 断言多项式 x**2 - 2 的 sqf_norm 结果正确

    R, x = ring("x", QQ.algebraic_field(sqrt(2)))  # 创建一个含有平方根 2 的有理数域上的多项式环，并初始化变量 x
    X = R.to_ground().x  # 将 R 转换为其底部的多项式环，并获取变量 X

    assert (x**2 - 3).sqf_norm() == ([1], x**2 - 2*sqrt(2)*x - 1, X**4 - 10*X**2 + 1)
    # 断言多项式 x**2 - 3 的 sqf_norm 结果正确

def test_PolyElement_sqf_list():
    _, x = ring("x", ZZ)  # 创建一个整数环的多项式环，并初始化变量 x

    f = x**5 - x**3 - x**2 + 1  # 初始化多项式 f
    g = x**3 + 2*x**2 + 2*x + 1  # 初始化多项式 g
    h = x - 1  # 初始化多项式 h
    p = x**4 + x**3 - x - 1  # 初始化多项式 p

    assert f.sqf_part() == p  # 断言 f 的 sqf_part 结果正确
    assert f.sqf_list() == (1, [(g, 1), (h, 2)])  # 断言 f 的 sqf_list 结果正确

def test_issue_18894():
    items = [S(3)/16 + sqrt(3*sqrt(3) + 10)/8, S(1)/8 + 3*sqrt(3)/16, S(1)/8 + 3*sqrt(3)/16, -S(3)/16 + sqrt(3*sqrt(3) + 10)/8]
    R, a = sring(items, extension=True)  # 创建包含扩展项的环，并初始化变量 R, a
    # 计算多项式 f(x) = x^5 - x^3 - x^2 + 1 的值
    f = x**5 - x**3 - x**2 + 1
    
    # 定义多项式的三个因式 u(x) = x + 1, v(x) = x - 1, w(x) = x^2 + x + 1
    u = x + 1
    v = x - 1
    w = x**2 + x + 1
    
    # 使用 assert 断言来验证 f(x) 的因式分解结果是否为 [(u(x), 1), (v(x), 2), (w(x), 1)]
    assert f.factor_list() == (1, [(u, 1), (v, 2), (w, 1)])
# 定义一个测试函数 test_issue_21410，用于验证问题 21410 的解决情况
def test_issue_21410():
    # 创建一个有限域 F_2 上的多项式环 R，并初始化变量 x
    R, x = ring('x', FF(2))
    # 定义一个多项式 p = x^6 + x^5 + x^4 + x^3 + 1
    p = x**6 + x**5 + x**4 + x**3 + 1
    # 断言：调用 p 的 _pow_multinomial 方法，传入参数 4，预期结果是 x^24 + x^20 + x^16 + x^12 + 1
    assert p._pow_multinomial(4) == x**24 + x**20 + x**16 + x**12 + 1
```