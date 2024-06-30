# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\test_cg.py`

```
# 导入 SymPy 库中具体的模块，以便后续使用
from sympy.concrete.summations import Sum
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.cg import Wigner3j, Wigner6j, Wigner9j, CG, cg_simp
from sympy.functions.special.tensor_functions import KroneckerDelta

# 定义测试函数 test_cg_simp_add
def test_cg_simp_add():
    # 定义符号变量 j, m1, m1p, m2, m2p
    j, m1, m1p, m2, m2p = symbols('j m1 m1p m2 m2p')
    
    # Test Varshalovich 8.7.1 Eq 1
    # 使用 CG 类创建不同的 CG 系数对象 a, b, c, d, e
    a = CG(S.Half, S.Half, 0, 0, S.Half, S.Half)
    b = CG(S.Half, Rational(-1, 2), 0, 0, S.Half, Rational(-1, 2))
    c = CG(1, 1, 0, 0, 1, 1)
    d = CG(1, 0, 0, 0, 1, 0)
    e = CG(1, -1, 0, 0, 1, -1)
    
    # 断言简化后的表达式与预期值相等
    assert cg_simp(a + b) == 2
    assert cg_simp(c + d + e) == 3
    assert cg_simp(a + b + c + d + e) == 5
    assert cg_simp(a + b + c) == 2 + c
    assert cg_simp(2*a + b) == 2 + a
    assert cg_simp(2*c + d + e) == 3 + c
    assert cg_simp(5*a + 5*b) == 10
    assert cg_simp(5*c + 5*d + 5*e) == 15
    assert cg_simp(-a - b) == -2
    assert cg_simp(-c - d - e) == -3
    assert cg_simp(-6*a - 6*b) == -12
    assert cg_simp(-4*c - 4*d - 4*e) == -12
    
    # 重新赋值 a, b, c, d, e，用于测试 Varshalovich 8.7.1 Eq 2
    a = CG(S.Half, S.Half, S.Half, Rational(-1, 2), 0, 0)
    b = CG(S.Half, Rational(-1, 2), S.Half, S.Half, 0, 0)
    c = CG(1, 1, 1, -1, 0, 0)
    d = CG(1, 0, 1, 0, 0, 0)
    e = CG(1, -1, 1, 1, 0, 0)
    
    # 断言简化后的表达式与预期值相等
    assert cg_simp(a - b) == sqrt(2)
    assert cg_simp(c - d + e) == sqrt(3)
    assert cg_simp(a - b + c - d + e) == sqrt(2) + sqrt(3)
    assert cg_simp(a - b + c) == sqrt(2) + c
    assert cg_simp(2*a - b) == sqrt(2) + a
    assert cg_simp(2*c - d + e) == sqrt(3) + c
    assert cg_simp(5*a - 5*b) == 5*sqrt(2)
    assert cg_simp(5*c - 5*d + 5*e) == 5*sqrt(3)
    assert cg_simp(-a + b) == -sqrt(2)
    assert cg_simp(-c + d - e) == -sqrt(3)
    assert cg_simp(-6*a + 6*b) == -6*sqrt(2)
    assert cg_simp(-4*c + 4*d - 4*e) == -4*sqrt(3)
    
    # 继续测试使用符号变量 j 的情况，这里暂时省略最后的 CG 对象的赋值部分
    # 定义 CG 对象，并初始化变量 d 和 e
    d = CG(1, 0, 1, 0, j, 0)
    e = CG(1, -1, 1, 1, j, 0)
    # 断言验证两个 CG 对象的差为 sqrt(2)*KroneckerDelta(j, 0)
    assert cg_simp(a - b) == sqrt(2)*KroneckerDelta(j, 0)
    # 断言验证 CG 对象 c 减去 d 加上 e 的简化结果为 sqrt(3)*KroneckerDelta(j, 0)
    assert cg_simp(c - d + e) == sqrt(3)*KroneckerDelta(j, 0)
    # 断言验证 CG 对象 a 减去 b 加上 c 减去 d 加上 e 的简化结果为
    # sqrt(2)*KroneckerDelta(j, 0) + sqrt(3)*KroneckerDelta(j, 0)
    assert cg_simp(a - b + c - d + e) == sqrt(
        2)*KroneckerDelta(j, 0) + sqrt(3)*KroneckerDelta(j, 0)
    # 断言验证 CG 对象 a 减去 b 加上 c 的简化结果为 sqrt(2)*KroneckerDelta(j, 0) + c
    assert cg_simp(a - b + c) == sqrt(2)*KroneckerDelta(j, 0) + c
    # 断言验证 CG 对象 2a 减去 b 的简化结果为 sqrt(2)*KroneckerDelta(j, 0) + a
    assert cg_simp(2*a - b) == sqrt(2)*KroneckerDelta(j, 0) + a
    # 断言验证 CG 对象 2c 减去 d 加上 e 的简化结果为 sqrt(3)*KroneckerDelta(j, 0) + c
    assert cg_simp(2*c - d + e) == sqrt(3)*KroneckerDelta(j, 0) + c
    # 断言验证 CG 对象 5a 减去 5b 的简化结果为 5*sqrt(2)*KroneckerDelta(j, 0)
    assert cg_simp(5*a - 5*b) == 5*sqrt(2)*KroneckerDelta(j, 0)
    # 断言验证 CG 对象 5c 减去 5d 加上 5e 的简化结果为 5*sqrt(3)*KroneckerDelta(j, 0)
    assert cg_simp(5*c - 5*d + 5*e) == 5*sqrt(3)*KroneckerDelta(j, 0)
    # 断言验证 CG 对象 -a 加上 b 的简化结果为 -sqrt(2)*KroneckerDelta(j, 0)
    assert cg_simp(-a + b) == -sqrt(2)*KroneckerDelta(j, 0)
    # 断言验证 CG 对象 -c 加上 d 减去 e 的简化结果为 -sqrt(3)*KroneckerDelta(j, 0)
    assert cg_simp(-c + d - e) == -sqrt(3)*KroneckerDelta(j, 0)
    # 断言验证 CG 对象 -6a 加上 6b 的简化结果为 -6*sqrt(2)*KroneckerDelta(j, 0)
    assert cg_simp(-6*a + 6*b) == -6*sqrt(2)*KroneckerDelta(j, 0)
    # 断言验证 CG 对象 -4c 加上 4d 减去 4e 的简化结果为 -4*sqrt(3)*KroneckerDelta(j, 0)
    assert cg_simp(-4*c + 4*d - 4*e) == -4*sqrt(3)*KroneckerDelta(j, 0)
    
    # 测试 Varshalovich 8.7.2 方程 9
    # alpha=alphap, beta=betap 情况下的数值测试
    # 计算并赋值变量 a、b、c、d
    a = CG(S.Half, S.Half, S.Half, Rational(-1, 2), 1, 0)**2
    b = CG(S.Half, S.Half, S.Half, Rational(-1, 2), 0, 0)**2
    c = CG(1, 0, 1, 1, 1, 1)**2
    d = CG(1, 0, 1, 1, 2, 1)**2
    # 断言验证 cg_simp(a + b) 简化结果为 1
    assert cg_simp(a + b) == 1
    # 断言验证 cg_simp(c + d) 简化结果为 1
    assert cg_simp(c + d) == 1
    # 断言验证 cg_simp(a + b + c + d) 简化结果为 2
    assert cg_simp(a + b + c + d) == 2
    # 断言验证 cg_simp(4*a + 4*b) 简化结果为 4
    assert cg_simp(4*a + 4*b) == 4
    # 断言验证 cg_simp(4*c + 4*d) 简化结果为 4
    assert cg_simp(4*c + 4*d) == 4
    # 断言验证 cg_simp(5*a + 3*b) 简化结果为 3 + 2*a
    assert cg_simp(5*a + 3*b) == 3 + 2*a
    # 断言验证 cg_simp(5*c + 3*d) 简化结果为 3 + 2*c
    assert cg_simp(5*c + 3*d) == 3 + 2*c
    # 断言验证 cg_simp(-a - b) 简化结果为 -1
    assert cg_simp(-a - b) == -1
    # 断言验证 cg_simp(-c - d) 简化结果为 -1
    
    # 符号测试
    # 计算并赋值变量 a、b、c、d
    a = CG(S.Half, m1, S.Half, m2, 1, 1)**2
    b = CG(S.Half, m1, S.Half, m2, 1, 0)**2
    c = CG(S.Half, m1, S.Half, m2, 1, -1)**2
    d = CG(S.Half, m1, S.Half, m2, 0, 0)**2
    # 断言验证 cg_simp(a + b + c + d) 简化结果为 1
    assert cg_simp(a + b + c + d) == 1
    # 断言验证 cg_simp(4*a + 4*b + 4*c + 4*d) 简化结果为 4
    assert cg_simp(4*a + 4*b + 4*c + 4*d) == 4
    # 断言验证 cg_simp(3*a + 5*b + 3*c + 4*d) 简化结果为 3 + 2*b + d
    assert cg_simp(3*a + 5*b + 3*c + 4*d) == 3 + 2*b + d
    # 断言验证 cg_simp(-a - b - c - d) 简化结果为 -1
    
    # 计算并赋值变量 a 到 i
    a = CG(1, m1, 1, m2, 2, 2)**2
    b = CG(1, m1, 1, m2, 2, 1)**2
    c = CG(1, m1, 1, m2, 2, 0)**2
    d = CG(1, m1, 1, m2, 2, -1)**2
    e = CG(1, m1, 1, m2, 2, -2)**2
    f = CG(1, m1, 1, m2, 1, 1)**2
    g = CG(1, m1, 1, m2, 1, 0)**2
    h = CG(1, m1, 1, m2, 1, -1)**2
    i = CG(1, m1, 1, m2, 0, 0)**2
    # 断言验证 cg_simp(a + b + c + d + e + f + g + h + i) 简化结果为 1
    assert cg_simp(a + b + c + d + e + f + g + h + i) == 1
    # 计算第一个项：CG系数乘积
    b = CG(S.Half, m1, S.Half, m2, 1, 0)*CG(S.Half, m1p, S.Half, m2p, 1, 0)
    # 计算第二个项：CG系数乘积
    c = CG(S.Half, m1, S.Half, m2, 1, -1)*CG(S.Half, m1p, S.Half, m2p, 1, -1)
    # 计算第三个项：CG系数乘积
    d = CG(S.Half, m1, S.Half, m2, 0, 0)*CG(S.Half, m1p, S.Half, m2p, 0, 0)
    # 断言：简化后的和等于KroneckerDelta函数的乘积
    assert cg_simp(a + b + c + d) == KroneckerDelta(m1, m1p)*KroneckerDelta(m2, m2p)
    # 计算第一个项：CG系数乘积
    a = CG(1, m1, 1, m2, 2, 2)*CG(1, m1p, 1, m2p, 2, 2)
    # 计算第二个项：CG系数乘积
    b = CG(1, m1, 1, m2, 2, 1)*CG(1, m1p, 1, m2p, 2, 1)
    # 计算第三个项：CG系数乘积
    c = CG(1, m1, 1, m2, 2, 0)*CG(1, m1p, 1, m2p, 2, 0)
    # 计算第四个项：CG系数乘积
    d = CG(1, m1, 1, m2, 2, -1)*CG(1, m1p, 1, m2p, 2, -1)
    # 计算第五个项：CG系数乘积
    e = CG(1, m1, 1, m2, 2, -2)*CG(1, m1p, 1, m2p, 2, -2)
    # 计算第六个项：CG系数乘积
    f = CG(1, m1, 1, m2, 1, 1)*CG(1, m1p, 1, m2p, 1, 1)
    # 计算第七个项：CG系数乘积
    g = CG(1, m1, 1, m2, 1, 0)*CG(1, m1p, 1, m2p, 1, 0)
    # 计算第八个项：CG系数乘积
    h = CG(1, m1, 1, m2, 1, -1)*CG(1, m1p, 1, m2p, 1, -1)
    # 计算第九个项：CG系数乘积
    i = CG(1, m1, 1, m2, 0, 0)*CG(1, m1p, 1, m2p, 0, 0)
    # 断言：简化后的和等于KroneckerDelta函数的乘积
    assert cg_simp(
        a + b + c + d + e + f + g + h + i) == KroneckerDelta(m1, m1p)*KroneckerDelta(m2, m2p)
# 定义一个测试函数 `test_cg_simp_sum`
def test_cg_simp_sum():
    # 定义符号变量
    x, a, b, c, cp, alpha, beta, gamma, gammap = symbols('x a b c cp alpha beta gamma gammap')
    
    # Varshalovich 8.7.1 方程式 1 的断言测试
    assert cg_simp(x * Sum(CG(a, alpha, b, 0, a, alpha), (alpha, -a, a))) == x*(2*a + 1)*KroneckerDelta(b, 0)
    
    # Varshalovich 8.7.1 方程式 1 的另一个断言测试，同时包含 CG 对象的加法
    assert cg_simp(x * Sum(CG(a, alpha, b, 0, a, alpha), (alpha, -a, a)) + CG(1, 0, 1, 0, 1, 0)) == x*(2*a + 1)*KroneckerDelta(b, 0) + CG(1, 0, 1, 0, 1, 0)
    
    # Varshalovich 8.7.1 方程式 2 的断言测试
    assert cg_simp(2 * Sum(CG(1, alpha, 0, 0, 1, alpha), (alpha, -1, 1))) == 6
    
    # Varshalovich 8.7.1 方程式 2 的断言测试
    assert cg_simp(x*Sum((-1)**(a - alpha) * CG(a, alpha, a, -alpha, c, 0), (alpha, -a, a))) == x*sqrt(2*a + 1)*KroneckerDelta(c, 0)
    
    # Varshalovich 8.7.1 方程式 3 的断言测试
    assert cg_simp(3*Sum((-1)**(2 - alpha) * CG(2, alpha, 2, -alpha, 0, 0), (alpha, -2, 2))) == 3*sqrt(5)
    
    # Varshalovich 8.7.2 方程式 4 的断言测试
    assert cg_simp(Sum(CG(a, alpha, b, beta, c, gamma)*CG(a, alpha, b, beta, cp, gammap), (alpha, -a, a), (beta, -b, b))) == KroneckerDelta(c, cp)*KroneckerDelta(gamma, gammap)
    
    # Varshalovich 8.7.2 方程式 5 的断言测试
    assert cg_simp(Sum(CG(a, alpha, b, beta, c, gamma)*CG(a, alpha, b, beta, c, gammap), (alpha, -a, a), (beta, -b, b))) == KroneckerDelta(gamma, gammap)
    
    # Varshalovich 8.7.2 方程式 6 的断言测试
    assert cg_simp(Sum(CG(a, alpha, b, beta, c, gamma)*CG(a, alpha, b, beta, cp, gamma), (alpha, -a, a), (beta, -b, b))) == KroneckerDelta(c, cp)
    
    # Varshalovich 8.7.2 方程式 7 的断言测试
    assert cg_simp(Sum(CG(a, alpha, b, beta, c, gamma)**2, (alpha, -a, a), (beta, -b, b))) == 1
    
    # Varshalovich 8.7.2 方程式 8 的断言测试
    assert cg_simp(Sum(CG(2, alpha, 1, beta, 2, gamma)*CG(2, alpha, 1, beta, 2, gammap), (alpha, -2, 2), (beta, -1, 1))) == KroneckerDelta(gamma, gammap)


# 定义一个测试函数 `test_doit`
def test_doit():
    # 断言测试 Wigner 3j 符号的计算结果
    assert Wigner3j(S.Half, Rational(-1, 2), S.Half, S.Half, 0, 0).doit() == -sqrt(2)/2
    assert Wigner3j(1/2,1/2,1/2,1/2,1/2,1/2).doit() == 0
    assert Wigner3j(9/2,9/2,9/2,9/2,9/2,9/2).doit() ==  0
    
    # 断言测试 Wigner 6j 符号的计算结果
    assert Wigner6j(1, 2, 3, 2, 1, 2).doit() == sqrt(21)/105
    assert Wigner6j(3, 1, 2, 2, 2, 1).doit() == sqrt(21) / 105
    
    # 断言测试 Wigner 9j 符号的计算结果
    assert Wigner9j(2, 1, 1, Rational(3, 2), S.Half, 1, S.Half, S.Half, 0).doit() == sqrt(2)/12
    
    # 断言测试 CG 符号的计算结果
    assert CG(S.Half, S.Half, S.Half, Rational(-1, 2), 1, 0).doit() == sqrt(2)/2
    
    # 断言测试，J 减去 M 不是整数的情况
    assert Wigner3j(1, -1, S.Half, S.Half, 1, S.Half).doit() == 0
    assert CG(4, -1, S.Half, S.Half, 4, Rational(-1, 2)).doit() == 0
```