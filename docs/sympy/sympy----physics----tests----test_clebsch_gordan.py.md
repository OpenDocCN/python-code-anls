# `D:\src\scipysrc\sympy\sympy\physics\tests\test_clebsch_gordan.py`

```
# 从 sympy.core.numbers 模块导入 I（虚数单位）、pi（圆周率）、Rational（有理数类）
# 从 sympy.core.singleton 模块导入 S（符号 S）
# 从 sympy.core.symbol 模块导入 symbols（符号生成器）
# 从 sympy.functions.elementary.exponential 模块导入 exp（指数函数）
# 从 sympy.functions.elementary.miscellaneous 模块导入 sqrt（平方根函数）
# 从 sympy.functions.elementary.trigonometric 模块导入 cos（余弦函数）、sin（正弦函数）
# 从 sympy.functions.special.spherical_harmonics 模块导入 Ynm（球谐函数）
# 从 sympy.matrices.dense 模块导入 Matrix（矩阵类）
# 从 sympy.physics.wigner 模块导入一系列函数：clebsch_gordan（Clebsch-Gordan 系数）、wigner_9j（Wigner 9-j 符号）、wigner_6j（Wigner 6-j 符号）、gaunt（Gaunt 系数）、real_gaunt（实 Gaunt 系数）、racah（Racah 符号）、dot_rot_grad_Ynm（Ynm 的旋转梯度点积）、wigner_3j（Wigner 3-j 符号）、wigner_d_small（小Wigner d 函数）、wigner_d（Wigner d 函数）
# 从 sympy.testing.pytest 模块导入 raises（测试函数抛出异常）

# 用于测试的链接：https://en.wikipedia.org/wiki/Table_of_Clebsch%E2%80%93Gordan_coefficients
# 定义测试函数，用于测试 Clebsch-Gordan 系数文档中的例子
def test_clebsch_gordan_docs():
    # 断言 Clebsch-Gordan 系数函数的计算结果，检验参数为有理数和符号 S 的使用
    assert clebsch_gordan(Rational(3, 2), S.Half, 2, Rational(3, 2), S.Half, 2) == 1
    assert clebsch_gordan(Rational(3, 2), S.Half, 1, Rational(3, 2), Rational(-1, 2), 1) == sqrt(3)/2
    assert clebsch_gordan(Rational(3, 2), S.Half, 1, Rational(-1, 2), S.Half, 0) == -sqrt(2)/2

# 定义测试函数，用于测试 Clebsch-Gordan 系数的不同参数组合
def test_clebsch_gordan():
    # 定义和赋值各种有理数和符号 S 的变量
    h = S.One
    k = S.Half
    l = Rational(3, 2)
    i = Rational(-1, 2)
    n = Rational(7, 2)
    p = Rational(5, 2)
    # 一系列断言，验证不同参数组合下 Clebsch-Gordan 系数的计算结果是否正确
    assert clebsch_gordan(k, k, 1, k, k, 1) == 1
    assert clebsch_gordan(k, k, 1, k, k, 0) == 0
    assert clebsch_gordan(k, k, 1, i, i, -1) == 1
    assert clebsch_gordan(k, k, 1, k, i, 0) == sqrt(2)/2
    assert clebsch_gordan(k, k, 0, k, i, 0) == sqrt(2)/2
    assert clebsch_gordan(k, k, 1, i, k, 0) == sqrt(2)/2
    assert clebsch_gordan(k, k, 0, i, k, 0) == -sqrt(2)/2
    assert clebsch_gordan(h, k, l, 1, k, l) == 1
    assert clebsch_gordan(h, k, l, 1, i, k) == 1/sqrt(3)
    assert clebsch_gordan(h, k, k, 1, i, k) == sqrt(2)/sqrt(3)
    assert clebsch_gordan(h, k, k, 0, k, k) == -1/sqrt(3)
    assert clebsch_gordan(h, k, l, 0, k, k) == sqrt(2)/sqrt(3)
    assert clebsch_gordan(h, h, S(2), 1, 1, S(2)) == 1
    assert clebsch_gordan(h, h, S(2), 1, 0, 1) == 1/sqrt(2)
    assert clebsch_gordan(h, h, S(2), 0, 1, 1) == 1/sqrt(2)
    assert clebsch_gordan(h, h, 1, 1, 0, 1) == 1/sqrt(2)
    assert clebsch_gordan(h, h, 1, 0, 1, 1) == -1/sqrt(2)
    assert clebsch_gordan(l, l, S(3), l, l, S(3)) == 1
    assert clebsch_gordan(l, l, S(2), l, k, S(2)) == 1/sqrt(2)
    assert clebsch_gordan(l, l, S(3), l, k, S(2)) == 1/sqrt(2)
    assert clebsch_gordan(S(2), S(2), S(4), S(2), S(2), S(4)) == 1
    assert clebsch_gordan(S(2), S(2), S(3), S(2), 1, S(3)) == 1/sqrt(2)
    assert clebsch_gordan(S(2), S(2), S(3), 1, 1, S(2)) == 0
    assert clebsch_gordan(p, h, n, p, 1, n) == 1
    assert clebsch_gordan(p, h, p, p, 0, p) == sqrt(5)/sqrt(7)
    assert clebsch_gordan(p, h, l, k, 1, l) == 1/sqrt(15)

# 定义测试函数，用于测试 Wigner 相关函数
def test_wigner():
    # 定义内部函数，用于比较两个数的差的绝对值是否小于给定精度
    def tn(a, b):
        return (a - b).n(64) < S('1e-64')
    # 断言 Wigner 9-j 符号函数的计算结果，验证其是否接近给定的有理数
    assert tn(wigner_9j(1, 1, 1, 1, 1, 1, 1, 1, 0, prec=64), Rational(1, 18))
    # 断言：验证 Wigner 9j 符号函数的计算结果是否符合预期
    assert wigner_9j(3, 3, 2, 3, 3, 2, 3, 3, 2) == 3221*sqrt(
        70)/(246960*sqrt(105)) - 365/(3528*sqrt(70)*sqrt(105))
    
    # 断言：验证 Wigner 6j 符号函数的计算结果是否符合预期
    assert wigner_6j(5, 5, 5, 5, 5, 5) == Rational(1, 52)
    
    # 断言：使用自定义精度验证 Wigner 6j 符号函数的计算结果是否符合预期
    assert tn(wigner_6j(8, 8, 8, 8, 8, 8, prec=64), Rational(-12219, 965770))
    
    # 回归测试，用于检验 GitHub 问题 #8747
    half = S.Half
    # 断言：验证 Wigner 9j 符号函数的计算结果是否符合预期
    assert wigner_9j(0, 0, 0, 0, half, half, 0, half, half) == half
    
    # 断言：验证 Wigner 9j 符号函数的计算结果是否符合预期
    assert (wigner_9j(3, 5, 4,
                      7 * half, 5 * half, 4,
                      9 * half, 9 * half, 0)
            == -sqrt(Rational(361, 205821000)))
    
    # 断言：验证 Wigner 9j 符号函数的计算结果是否符合预期
    assert (wigner_9j(1, 4, 3,
                      5 * half, 4, 5 * half,
                      5 * half, 2, 7 * half)
            == -sqrt(Rational(3971, 373403520)))
    
    # 断言：验证 Wigner 9j 符号函数的计算结果是否符合预期
    assert (wigner_9j(4, 9 * half, 5 * half,
                      2, 4, 4,
                      5, 7 * half, 7 * half)
            == -sqrt(Rational(3481, 5042614500)))
def test_realgaunt():
    # 测试 real_gaunt 函数

    # 对于 l 从 0 到 2 的每个非零值，验证特定情况下的返回值
    for l in range(3):
        for m in range(-l, l+1):
            assert real_gaunt(0, l, l, 0, m, m) == 1/(2*sqrt(pi))

    # 验证具体的实例，确保返回值符合预期
    assert real_gaunt(1, 1, 2, 0, 0, 0) == sqrt(5)/(5*sqrt(pi))
    assert real_gaunt(1, 1, 2, 1, 1, 0) == -sqrt(5)/(10*sqrt(pi))
    assert real_gaunt(2, 2, 2, 0, 0, 0) == sqrt(5)/(7*sqrt(pi))
    assert real_gaunt(2, 2, 2, 0, 2, 2) == -sqrt(5)/(7*sqrt(pi))
    assert real_gaunt(2, 2, 2, -2, -2, 0) == -sqrt(5)/(7*sqrt(pi))
    assert real_gaunt(1, 1, 2, -1, 0, -1) == sqrt(15)/(10*sqrt(pi))
    assert real_gaunt(1, 1, 2, 0, 1, 1) == sqrt(15)/(10*sqrt(pi))
    assert real_gaunt(1, 1, 2, 1, 1, 2) == sqrt(15)/(10*sqrt(pi))
    assert real_gaunt(1, 1, 2, -1, 1, -2) == -sqrt(15)/(10*sqrt(pi))
    assert real_gaunt(1, 1, 2, -1, -1, 2) == -sqrt(15)/(10*sqrt(pi))
    assert real_gaunt(2, 2, 2, 0, 1, 1) == sqrt(5)/(14*sqrt(pi))
    assert real_gaunt(2, 2, 2, 1, 1, 2) == sqrt(15)/(14*sqrt(pi))
    assert real_gaunt(2, 2, 2, -1, -1, 2) == -sqrt(15)/(14*sqrt(pi))

    # 测试特定情况下返回 S.Zero
    assert real_gaunt(-2, -2, -2, -2, -2, 0) is S.Zero  # m test
    assert real_gaunt(-2, 1, 0, 1, 1, 1) is S.Zero  # l test
    assert real_gaunt(-2, -1, -2, -1, -1, 0) is S.Zero  # m and l test
    assert real_gaunt(-2, -2, -2, -2, -2, -2) is S.Zero  # m and k test
    assert real_gaunt(-2, -1, -2, -1, -1, -1) is S.Zero  # m, l and k test

    # 声明整数符号变量 x 和一个长度为 6 的整数列表 v
    x = symbols('x', integer=True)
    v = [0]*6
    # 循环遍历列表 v 的索引范围
    for i in range(len(v)):
        # 将 v[i] 赋值为 x，这里是因为非文字整数会导致错误
        v[i] = x
        # 断言函数 real_gaunt(*v) 会抛出 ValueError 异常
        raises(ValueError, lambda: real_gaunt(*v))
        # 将 v[i] 重新赋值为 0，准备下一次循环
        v[i] = 0
# 定义测试函数 test_racah
def test_racah():
    # 断言验证 racah 函数计算结果是否等于 Rational(-1,14)
    assert racah(3,3,3,3,3,3) == Rational(-1,14)
    # 断言验证 racah 函数计算结果是否等于 Rational(-3,70)
    assert racah(2,2,2,2,2,2) == Rational(-3,70)
    # 断言验证 racah 函数返回值的类型是否是 Float
    assert racah(7,8,7,1,7,7, prec=4).is_Float
    # 断言验证 racah 函数计算结果是否等于 -719*sqrt(598)/1158924
    assert racah(5.5,7.5,9.5,6.5,8,9) == -719*sqrt(598)/1158924
    # 断言验证 racah 函数计算结果与预期值的绝对误差是否小于 S('1e-4')
    assert abs(racah(5.5,7.5,9.5,6.5,8,9, prec=4) - (-0.01517)) < S('1e-4')

# 定义测试函数 test_dot_rota_grad_SH
def test_dot_rota_grad_SH():
    # 声明符号变量 theta, phi
    theta, phi = symbols("theta phi")
    # 断言验证 dot_rot_grad_Ynm 函数计算结果不等于 sqrt(30)*Ynm(2, 2, 1, 0)/(10*sqrt(pi))
    assert dot_rot_grad_Ynm(1, 1, 1, 1, 1, 0) !=  \
        sqrt(30)*Ynm(2, 2, 1, 0)/(10*sqrt(pi))
    # 断言验证 dot_rot_grad_Ynm 函数计算结果化简后等于 sqrt(30)*Ynm(2, 2, 1, 0)/(10*sqrt(pi))
    assert dot_rot_grad_Ynm(1, 1, 1, 1, 1, 0).doit() ==  \
        sqrt(30)*Ynm(2, 2, 1, 0)/(10*sqrt(pi))
    # 断言验证 dot_rot_grad_Ynm 函数计算结果不等于 0
    assert dot_rot_grad_Ynm(1, 5, 1, 1, 1, 2) !=  \
        0
    # 断言验证 dot_rot_grad_Ynm 函数计算结果化简后等于 0
    assert dot_rot_grad_Ynm(1, 5, 1, 1, 1, 2).doit() ==  \
        0
    # 断言验证 dot_rot_grad_Ynm 函数计算结果化简后等于 15*sqrt(3003)*Ynm(6, 6, theta, phi)/(143*sqrt(pi))
    assert dot_rot_grad_Ynm(3, 3, 3, 3, theta, phi).doit() ==  \
        15*sqrt(3003)*Ynm(6, 6, theta, phi)/(143*sqrt(pi))
    # 断言验证 dot_rot_grad_Ynm 函数计算结果化简后等于 sqrt(3)*Ynm(4, 4, theta, phi)/sqrt(pi)
    assert dot_rot_grad_Ynm(3, 3, 1, 1, theta, phi).doit() ==  \
        sqrt(3)*Ynm(4, 4, theta, phi)/sqrt(pi)
    # 断言验证 dot_rot_grad_Ynm 函数计算结果化简后等于 3*sqrt(55)*Ynm(5, 2, theta, phi)/(11*sqrt(pi))
    assert dot_rot_grad_Ynm(3, 2, 2, 0, theta, phi).doit() ==  \
        3*sqrt(55)*Ynm(5, 2, theta, phi)/(11*sqrt(pi))
    # 断言验证 dot_rot_grad_Ynm 函数计算结果展开后等于 
    # -sqrt(70)*Ynm(4, 4, theta, phi)/(11*sqrt(pi)) + 45*sqrt(182)*Ynm(6, 4, theta, phi)/(143*sqrt(pi))
    assert dot_rot_grad_Ynm(3, 2, 3, 2, theta, phi).doit().expand() ==  \
        -sqrt(70)*Ynm(4, 4, theta, phi)/(11*sqrt(pi)) + \
        45*sqrt(182)*Ynm(6, 4, theta, phi)/(143*sqrt(pi))

# 定义测试函数 test_wigner_d
def test_wigner_d():
    # 声明有理数表达式 S(1)/2
    half = S(1)/2
    # 断言验证 wigner_d_small 函数计算结果等于 Matrix([[1, 0], [0, 1]])
    assert wigner_d_small(half, 0) == Matrix([[1, 0], [0, 1]])
    # 断言验证 wigner_d_small 函数计算结果等于 Matrix([[1, 1], [-1, 1]])/sqrt(2)
    assert wigner_d_small(half, pi/2) == Matrix([[1, 1], [-1, 1]])/sqrt(2)
    # 断言验证 wigner_d_small 函数计算结果等于 Matrix([[0, 1], [-1, 0]])
    assert wigner_d_small(half, pi) == Matrix([[0, 1], [-1, 0]])

    # 声明实数符号变量 alpha, beta, gamma
    alpha, beta, gamma = symbols("alpha, beta, gamma", real=True)
    # 计算 Wigner-D 矩阵 D
    D = wigner_d(half, alpha, beta, gamma)
    # 断言验证 D 矩阵元素 D[0, 0] 的计算结果
    assert D[0, 0] == exp(I*alpha/2)*exp(I*gamma/2)*cos(beta/2)
    # 断言验证 D 矩阵元素 D[0, 1] 的计算结果
    assert D[0, 1] == exp(I*alpha/2)*exp(-I*gamma/2)*sin(beta/2)
    # 断言验证 D 矩阵元素 D[1, 0] 的计算结果
    assert D[1, 0] == -exp(-I*alpha/2)*exp(I*gamma/2)*sin(beta/2)
    # 断言验证 D 矩阵元素 D[1, 1] 的计算结果
    assert D[1, 1] == exp(-I*alpha/2)*exp(-I*gamma/2)*cos(beta/2)

    # 声明实数符号变量 theta, phi
    theta, phi = symbols("theta phi", real=True)
    # 构造向量 v
    v = Matrix([Ynm(1, mj, theta, phi) for mj in range(1, -2, -1)])
    # 计算矩阵乘法 w
    w = wigner_d(1, -pi/2, pi/2, -pi/2) @ v.subs({theta: pi/4, phi: pi})
    # 计算替换后的向量 w_
    w_ = v.subs({theta: pi/2, phi: pi/4})
    # 断言验证 w 和 w_ 的展开结果是否相等
    assert w.expand(func=True).as_real_imag() == w_.expand(func=True).as_real_imag()
```