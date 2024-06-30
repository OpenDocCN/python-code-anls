# `D:\src\scipysrc\sympy\sympy\simplify\tests\test_sqrtdenest.py`

```
# 导入所需的模块和函数
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.integrals.integrals import Integral
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.simplify.sqrtdenest import (
    _subsets as subsets, _sqrt_numeric_denest)

# 定义一系列的平方根表达式，每个都被赋予一个符号
r2, r3, r5, r6, r7, r10, r15, r29 = [sqrt(x) for x in (2, 3, 5, 6, 7, 10,
                                          15, 29)]

# 定义测试函数 test_sqrtdenest，用于验证 sqrtdenest 函数的正确性
def test_sqrtdenest():
    # 定义一个包含不同平方根表达式及其预期 sqrtdenest 结果的字典
    d = {sqrt(5 + 2 * r6): r2 + r3,
        sqrt(5. + 2 * r6): sqrt(5. + 2 * r6),
        sqrt(5. + 4*sqrt(5 + 2 * r6)): sqrt(5.0 + 4*r2 + 4*r3),
        sqrt(r2): sqrt(r2),
        sqrt(5 + r7): sqrt(5 + r7),
        sqrt(3 + sqrt(5 + 2*r7)):
         3*r2*(5 + 2*r7)**Rational(1, 4)/(2*sqrt(6 + 3*r7)) +
         r2*sqrt(6 + 3*r7)/(2*(5 + 2*r7)**Rational(1, 4)),
        sqrt(3 + 2*r3): 3**Rational(3, 4)*(r6/2 + 3*r2/2)/3}
    # 遍历字典中的每个表达式，并使用 sqrtdenest 函数进行验证
    for i in d:
        assert sqrtdenest(i) == d[i], i

# 定义测试函数 test_sqrtdenest2，用于验证 sqrtdenest 函数对更复杂表达式的处理是否正确
def test_sqrtdenest2():
    # 验证 sqrtdenest 对于特定复杂表达式的处理是否正确
    assert sqrtdenest(sqrt(16 - 2*r29 + 2*sqrt(55 - 10*r29))) == \
        r5 + sqrt(11 - 2*r29)
    e = sqrt(-r5 + sqrt(-2*r29 + 2*sqrt(-10*r29 + 55) + 16))
    assert sqrtdenest(e) == root(-2*r29 + 11, 4)
    r = sqrt(1 + r7)
    assert sqrtdenest(sqrt(1 + r)) == sqrt(1 + r)
    e = sqrt(((1 + sqrt(1 + 2*sqrt(3 + r2 + r5)))**2).expand())
    assert sqrtdenest(e) == 1 + sqrt(1 + 2*sqrt(r2 + r5 + 3))

    assert sqrtdenest(sqrt(5*r3 + 6*r2)) == \
        sqrt(2)*root(3, 4) + root(3, 4)**3

    assert sqrtdenest(sqrt(((1 + r5 + sqrt(1 + r3))**2).expand())) == \
        1 + r5 + sqrt(1 + r3)

    assert sqrtdenest(sqrt(((1 + r5 + r7 + sqrt(1 + r3))**2).expand())) == \
        1 + sqrt(1 + r3) + r5 + r7

    e = sqrt(((1 + cos(2) + cos(3) + sqrt(1 + r3))**2).expand())
    assert sqrtdenest(e) == cos(3) + cos(2) + 1 + sqrt(1 + r3)

    e = sqrt(-2*r10 + 2*r2*sqrt(-2*r10 + 11) + 14)
    assert sqrtdenest(e) == sqrt(-2*r10 - 2*r2 + 4*r5 + 14)

    # 检查 sqrtdenest 函数是否确保结果不比输入更复杂
    z = sqrt(-2*r29 + cos(2) + 2*sqrt(-10*r29 + 55) + 16)
    assert sqrtdenest(z) == z

    assert sqrtdenest(sqrt(r6 + sqrt(15))) == sqrt(r6 + sqrt(15))

    z = sqrt(15 - 2*sqrt(31) + 2*sqrt(55 - 10*r29))
    assert sqrtdenest(z) == z


# 定义测试函数 test_sqrtdenest_rec，用于验证 sqrtdenest 函数对递归表达式的处理是否正确
def test_sqrtdenest_rec():
    assert sqrtdenest(sqrt(-4*sqrt(14) - 2*r6 + 4*sqrt(21) + 33)) == \
        -r2 + r3 + 2*r7
    assert sqrtdenest(sqrt(-28*r7 - 14*r5 + 4*sqrt(35) + 82)) == \
        -7 + r5 + 2*r7
    assert sqrtdenest(sqrt(6*r2/11 + 2*sqrt(22)/11 + 6*sqrt(11)/11 + 2)) == \
        sqrt(11)*(r2 + 3 + sqrt(11))/11
    assert sqrtdenest(sqrt(468*r3 + 3024*r2 + 2912*r6 + 19735)) == \
        9*r3 + 26 + 56*r6
    z = sqrt(-490*r3 - 98*sqrt(115) - 98*sqrt(345) - 2107)
    assert sqrtdenest(z) == sqrt(-1)*(7*r5 + 7*r15 + 7*sqrt(23))
    z = sqrt(-4*sqrt(14) - 2*r6 + 4*sqrt(21) + 34)
    assert sqrtdenest(z) == z
    # 断言：对表达式进行平方根解嵌套，应等于 `-r10 + 1 + r2 + r5`
    assert sqrtdenest(sqrt(-8*r2 - 2*r5 + 18)) == -r10 + 1 + r2 + r5
    
    # 断言：对表达式进行平方根解嵌套，应等于 `sqrt(-1)*(-r10 + 1 + r2 + r5)`
    assert sqrtdenest(sqrt(8*r2 + 2*r5 - 18)) == sqrt(-1)*(-r10 + 1 + r2 + r5)
    
    # 断言：对表达式进行平方根解嵌套，应等于 `-r10/3 + r2 + r5 + 3`
    assert sqrtdenest(sqrt(8*r2/3 + 14*r5/3 + Rational(154, 9))) == -r10/3 + r2 + r5 + 3
    
    # 断言：对表达式进行平方根解嵌套，应等于 `sqrt(1 + r2 + r3 + r7)`
    assert sqrtdenest(sqrt(sqrt(2*r6 + 5) + sqrt(2*r7 + 8))) == sqrt(1 + r2 + r3 + r7)
    
    # 断言：对表达式进行平方根解嵌套，应等于 `1 + r3 + r5 + r15`
    assert sqrtdenest(sqrt(4*r15 + 8*r5 + 12*r3 + 24)) == 1 + r3 + r5 + r15
    
    # 定义变量 w
    w = 1 + r2 + r3 + r5 + r7
    # 断言：对表达式进行平方根解嵌套，应等于 w
    assert sqrtdenest(sqrt((w**2).expand())) == w
    
    # 定义变量 z
    z = sqrt((w**2).expand() + 1)
    # 断言：对表达式进行平方根解嵌套，应等于 z
    assert sqrtdenest(z) == z
    
    # 定义变量 z
    z = sqrt(2*r10 + 6*r2 + 4*r5 + 12 + 10*r15 + 30*r3)
    # 断言：对表达式进行平方根解嵌套，应等于 z
    assert sqrtdenest(z) == z
def test_issue_6241():
    # 定义变量 z，计算复杂的数学表达式
    z = sqrt( -320 + 32*sqrt(5) + 64*r15)
    # 断言 sqrtdenest 函数对 z 的处理结果与 z 相等
    assert sqrtdenest(z) == z


def test_sqrtdenest3():
    # 定义变量 z，计算复杂的数学表达式
    z = sqrt(13 - 2*r10 + 2*r2*sqrt(-2*r10 + 11))
    # 断言 sqrtdenest 函数对 z 的处理结果与预期结果相等
    assert sqrtdenest(z) == -1 + r2 + r10
    # 断言 sqrtdenest 函数对 z 的处理结果（最大迭代次数为1）与预期结果相等
    assert sqrtdenest(z, max_iter=1) == -1 + sqrt(2) + sqrt(10)
    # 定义变量 z，计算复杂的数学表达式
    z = sqrt(sqrt(r2 + 2) + 2)
    # 断言 sqrtdenest 函数对 z 的处理结果与 z 相等
    assert sqrtdenest(z) == z
    # 断言 sqrtdenest 函数对复杂表达式的处理结果与预期结果相等
    assert sqrtdenest(sqrt(-2*r10 + 4*r2*sqrt(-2*r10 + 11) + 20)) == \
        sqrt(-2*r10 - 4*r2 + 8*r5 + 20)
    # 断言 sqrtdenest 函数对复杂表达式的处理结果与预期结果相等
    assert sqrtdenest(sqrt((112 + 70*r2) + (46 + 34*r2)*r5)) == \
        r10 + 5 + 4*r2 + 3*r5
    # 定义变量 z，计算复杂的数学表达式
    z = sqrt(5 + sqrt(2*r6 + 5)*sqrt(-2*r29 + 2*sqrt(-10*r29 + 55) + 16))
    # 定义变量 r，计算复杂的数学表达式
    r = sqrt(-2*r29 + 11)
    # 断言 sqrtdenest 函数对 z 的处理结果与预期结果相等
    assert sqrtdenest(z) == sqrt(r2*r + r3*r + r10 + r15 + 5)

    # 定义变量 n，计算复杂的数学表达式
    n = sqrt(2*r6/7 + 2*r7/7 + 2*sqrt(42)/7 + 2)
    # 定义变量 d，计算复杂的数学表达式
    d = sqrt(16 - 2*r29 + 2*sqrt(55 - 10*r29))
    # 断言 sqrtdenest 函数对 n/d 的处理结果与预期结果相等
    assert sqrtdenest(n/d) == r7*(1 + r6 + r7)/(Mul(7, (sqrt(-2*r29 + 11) + r5),
                                                    evaluate=False))


def test_sqrtdenest4():
    # 查看 https://github.com/sympy/sympy/issues/3192 中的 Denest_en.pdf
    # 定义变量 z，计算复杂的数学表达式
    z = sqrt(8 - r2*sqrt(5 - r5) - sqrt(3)*(1 + r5))
    # 定义变量 z1，使用 sqrtdenest 处理 z 的结果
    z1 = sqrtdenest(z)
    # 定义变量 c，计算复杂的数学表达式
    c = sqrt(-r5 + 5)
    # 对 z1 进行展开并与复杂表达式的处理结果进行断言
    z1 = ((-r15*c - r3*c + c + r5*c - r6 - r2 + r10 + sqrt(30))/4).expand()
    assert sqrtdenest(z) == z1

    # 定义变量 z，计算复杂的数学表达式
    z = sqrt(2*r2*sqrt(r2 + 2) + 5*r2 + 4*sqrt(r2 + 2) + 8)
    # 断言 sqrtdenest 函数对 z 的处理结果与预期结果相等
    assert sqrtdenest(z) == r2 + sqrt(r2 + 2) + 2

    # 定义变量 w，计算复杂的数学表达式
    w = 2 + r2 + r3 + (1 + r3)*sqrt(2 + r2 + 5*r3)
    # 定义变量 z，计算复杂的数学表达式
    z = sqrt((w**2).expand())
    # 断言 sqrtdenest 函数对 z 的处理结果与 w 的展开结果相等
    assert sqrtdenest(z) == w.expand()


def test_sqrt_symbolic_denest():
    # 定义符号变量 x
    x = Symbol('x')
    # 定义变量 z，计算复杂的数学表达式
    z = sqrt(((1 + sqrt(sqrt(2 + x) + 3))**2).expand())
    # 断言 sqrtdenest 函数对 z 的处理结果与 z 相等
    assert sqrtdenest(z) == sqrt((1 + sqrt(sqrt(2 + x) + 3))**2)
    # 定义变量 z，计算复杂的数学表达式
    z = sqrt(((1 + sqrt(sqrt(2 + cos(1)) + 3))**2).expand())
    # 断言 sqrtdenest 函数对 z 的处理结果与预期结果相等
    assert sqrtdenest(z) == 1 + sqrt(sqrt(2 + cos(1)) + 3)
    # 定义变量 z，计算复杂的数学表达式
    z = ((1 + cos(2))**4 + 1).expand()
    # 断言 sqrtdenest 函数对 z 的处理结果与 z 相等
    assert sqrtdenest(z) == z
    # 定义变量 z，计算复杂的数学表达式
    z = sqrt(((1 + sqrt(sqrt(2 + cos(3*x)) + 3))**2 + 1).expand())
    # 断言 sqrtdenest 函数对 z 的处理结果与 z 相等
    assert sqrtdenest(z) == z
    # 定义变量 c，计算复杂的数学表达式
    c = cos(3)
    # 定义变量 c2，计算复杂的数学表达式
    c2 = c**2
    # 断言 sqrtdenest 函数对复杂表达式的处理结果与预期结果相等
    assert sqrtdenest(sqrt(2*sqrt(1 + r3)*c + c2 + 1 + r3*c2)) == \
        -1 - sqrt(1 + r3)*c
    # 定义变量 ra，计算复杂的数学表达式
    ra = sqrt(1 + r3)
    # 定义变量 z，计算复杂的数学表达式
    z = sqrt(20*ra*sqrt(3 + 3*r3) + 12*r3*ra*sqrt(3 + 3*r3) + 64*r3 + 112)
    # 断言 sqrtdenest 函数对 z 的处理结果与 z 相等
    assert sqrtdenest(z) == z


def test_issue_5857():
    # 导入符号变量 x, y
    from sympy.abc import x, y
    # 定义变量 z，计算复杂的数学表达式
    z = sqrt(1/(4*r3 + 7) + 1)
    # 定义变量 ans，计算复杂的数学表达式
    ans = (r2 + r6)/(r3 + 2)
    # 断言 sqrtdenest 函数对 z 的处理结果与预期结果相等
    assert sqrtdenest(z) == ans
    # 断言 sqrtdenest 函数对 1+z 的处理结果与预期结果相等
    assert sqrtdenest(1 + z) == 1 + ans
    # 断言 sqrtdenest 函数对 Integral(z + 1, (x,
    # 使用断言验证 subsets 函数对于参数 4 返回的结果是否符合预期
    assert subsets(4) == [
        # 列出所有可能的长度为 4 的子集，每个子集由四位二进制数组成
        [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0],
        [0, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1],
        [1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]]
# 测试一个特定的数学问题，验证 sqrtdenest 函数的功能
def test_issue_5653():
    # 断言 sqrtdenest 函数对于嵌套的平方根表达式进行处理后仍能保持一致
    assert sqrtdenest(sqrt(2 + sqrt(2 + sqrt(2)))) == sqrt(2 + sqrt(2 + sqrt(2)))

# 测试另一个数学问题，验证 sqrtdenest 函数的功能
def test_issue_12420():
    # 断言 sqrtdenest 函数对于复杂的平方根表达式能正确处理，预期结果是虚数单位 I
    assert sqrtdenest((3 - sqrt(2)*sqrt(4 + 3*I) + 3*I)/2) == I
    # 创建一个复杂的数学表达式 e
    e = 3 - sqrt(2)*sqrt(4 + I) + 3*I
    # 再次断言 sqrtdenest 函数对表达式 e 的处理结果与 e 自身相同
    assert sqrtdenest(e) == e

# 测试 sqrt_ratcomb 函数的特定情况
def test_sqrt_ratcomb():
    # 断言 sqrtdenest 函数对于一个特定的平方根组合表达式的处理结果为 0
    assert sqrtdenest(sqrt(1 + r3) + sqrt(3 + 3*r3) - sqrt(10 + 6*r3)) == 0

# 测试一个已知问题的处理，验证 sqrtdenest 函数的功能
def test_issue_18041():
    # 创建一个复杂的数学表达式 e
    e = -sqrt(-2 + 2*sqrt(3)*I)
    # 断言 sqrtdenest 函数对表达式 e 的处理结果与预期的结果相同
    assert sqrtdenest(e) == -1 - sqrt(3)*I

# 测试另一个已知问题的处理，验证 sqrtdenest 函数的功能
def test_issue_19914():
    # 创建整数对象 a, b, r 和 d2
    a = Integer(-8)
    b = Integer(-1)
    r = Integer(63)
    d2 = a*a - b*b*r

    # 断言 _sqrt_numeric_denest 函数的处理结果
    assert _sqrt_numeric_denest(a, b, r, d2) == sqrt(14)*I/2 + 3*sqrt(2)*I/2
    # 断言 sqrtdenest 函数对于一个复杂平方根表达式的处理结果与预期相同
    assert sqrtdenest(sqrt(-8-sqrt(63))) == sqrt(14)*I/2 + 3*sqrt(2)*I/2
```