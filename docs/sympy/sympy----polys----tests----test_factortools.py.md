# `D:\src\scipysrc\sympy\sympy\polys\tests\test_factortools.py`

```
# 多项式因式分解的工具，适用于特征为零的环境。

# 导入多项式相关的环和环形
from sympy.polys.rings import ring, xring
# 导入有限域、整数环、有理数环、整数环的虚部、有理数环的虚部、实数环、例外
from sympy.polys.domains import FF, ZZ, QQ, ZZ_I, QQ_I, RR, EX

# 导入多项式配置和多项式异常
from sympy.polys import polyconfig as config
from sympy.polys.polyerrors import DomainError
# 导入多项式类 ANP 和特殊多项式 f_polys、w_polys
from sympy.polys.polyclasses import ANP
from sympy.polys.specialpolys import f_polys, w_polys

# 导入复数单位和数学函数
from sympy.core.numbers import I
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
# 导入素数生成函数 nextprime
from sympy.ntheory.generate import nextprime
# 导入测试框架函数 raises 和 XFAIL
from sympy.testing.pytest import raises, XFAIL

# 从特殊多项式中获取 f_0 到 f_6 和 w_polys 中的 w_1 和 w_2
f_0, f_1, f_2, f_3, f_4, f_5, f_6 = f_polys()
w_1, w_2 = w_polys()

# 测试函数：使用多项式环 R 和变量 x 测试 dup_trial_division 函数
def test_dup_trial_division():
    R, x = ring("x", ZZ)
    # 断言：对 x^5 + 8x^4 + 25x^3 + 38x^2 + 28x + 8 进行 dup_trial_division，预期得到 [(x + 1, 2), (x + 2, 3)]
    assert R.dup_trial_division(x**5 + 8*x**4 + 25*x**3 + 38*x**2 + 28*x + 8, (x + 1, x + 2)) == [(x + 1, 2), (x + 2, 3)]

# 测试函数：使用多项式环 R 和变量 x, y 测试 dmp_trial_division 函数
def test_dmp_trial_division():
    R, x, y = ring("x,y", ZZ)
    # 断言：对 x^5 + 8x^4 + 25x^3 + 38x^2 + 28x + 8 进行 dmp_trial_division，预期得到 [(x + 1, 2), (x + 2, 3)]
    assert R.dmp_trial_division(x**5 + 8*x**4 + 25*x**3 + 38*x**2 + 28*x + 8, (x + 1, x + 2)) == [(x + 1, 2), (x + 2, 3)]

# 测试函数：使用多项式环 R 和变量 x 测试 dup_zz_mignotte_bound 函数
def test_dup_zz_mignotte_bound():
    R, x = ring("x", ZZ)
    # 断言：对 2x^2 + 3x + 4 进行 dup_zz_mignotte_bound，预期得到 6
    assert R.dup_zz_mignotte_bound(2*x**2 + 3*x + 4) == 6
    # 断言：对 x^3 + 14x^2 + 56x + 64 进行 dup_zz_mignotte_bound，预期得到 152
    assert R.dup_zz_mignotte_bound(x**3 + 14*x**2 + 56*x + 64) == 152

# 测试函数：使用多项式环 R 和变量 x, y 测试 dmp_zz_mignotte_bound 函数
def test_dmp_zz_mignotte_bound():
    R, x, y = ring("x,y", ZZ)
    # 断言：对 2x^2 + 3x + 4 进行 dmp_zz_mignotte_bound，预期得到 32
    assert R.dmp_zz_mignotte_bound(2*x**2 + 3*x + 4) == 32

# 测试函数：使用多项式环 R 和变量 x 测试 dup_zz_hensel_step 函数
def test_dup_zz_hensel_step():
    R, x = ring("x", ZZ)
    
    # 初始化多项式 f, g, h, s, t
    f = x**4 - 1
    g = x**3 + 2*x**2 - x - 2
    h = x - 2
    s = -2
    t = 2*x**2 - 2*x - 1

    # 使用 dup_zz_hensel_step 函数计算 G, H, S, T
    G, H, S, T = R.dup_zz_hensel_step(5, f, g, h, s, t)

    # 断言：G 应为 x^3 + 7x^2 - x - 7
    assert G == x**3 + 7*x**2 - x - 7
    # 断言：H 应为 x - 7
    assert H == x - 7
    # 断言：S 应为 8
    assert S == 8
    # 断言：T 应为 -8x^2 - 12x - 1
    assert T == -8*x**2 - 12*x - 1

# 测试函数：使用多项式环 R 和变量 x 测试 dup_zz_hensel_lift 函数
def test_dup_zz_hensel_lift():
    R, x = ring("x", ZZ)

    # 初始化多项式 f 和因子 F
    f = x**4 - 1
    F = [x - 1, x - 2, x + 2, x + 1]

    # 断言：对 5 使用 dup_zz_hensel_lift，预期得到 [x - 1, x - 182, x + 182, x + 1]
    assert R.dup_zz_hensel_lift(ZZ(5), f, F, 4) == [x - 1, x - 182, x + 182, x + 1]

# 测试函数：使用多项式环 R 和变量 x 测试 dup_zz_irreducible_p 函数
def test_dup_zz_irreducible_p():
    R, x = ring("x", ZZ)

    # 断言：对 3x^4 + 2x^3 + 6x^2 + 8x + 7 使用 dup_zz_irreducible_p，预期为 None
    assert R.dup_zz_irreducible_p(3*x**4 + 2*x**3 + 6*x**2 + 8*x + 7) is None
    # 断言：对 3x^4 + 2x^3 + 6x^2 + 8x + 4 使用 dup_zz_irreducible_p，预期为 None
    assert R.dup_zz_irreducible_p(3*x**4 + 2*x**3 + 6*x**2 + 8*x + 4) is None

    # 断言：对 3x^4 + 2x^3 + 6x^2 + 8x + 10 使用 dup_zz_irreducible_p，预期为 True
    assert R.dup_zz_irreducible_p(3*x**4 + 2*x**3 + 6*x**2 + 8*x + 10) is True
    # 断言：对 3x^4 + 2x^3 + 6x^2 + 8x + 14 使用 dup_zz_irreducible_p，预期为 True
    assert R.dup_zz_irreducible_p(3*x**4 + 2*x**3 + 6*x**2 + 8*x + 14) is True

# 测试函数：使用多项式环 R 和变量 x 测试 dup_cyclotomic_p 函数
def test_dup_cyclotomic_p():
    R, x = ring("x", ZZ)

    # 断言：对 x - 1 使用 dup_cyclotomic_p，预期为 True
    assert R.dup_cyclotomic_p(x - 1) is True
    # 断言：对 x + 1 使用 dup_cyclotomic_p，预期为 True
    assert R.dup_cyclotomic_p(x + 1) is True
    # 断言：对 x^2 + x +
    # 断言：检查在环 R 中，多项式 3*x + 1 是否不是分圆多项式
    assert R.dup_cyclotomic_p(3*x + 1) is False
    
    # 断言：检查在环 R 中，多项式 x**2 - 1 是否不是分圆多项式
    assert R.dup_cyclotomic_p(x**2 - 1) is False
    
    # 创建多项式 f = x**16 + x**14 - x**10 + x**8 - x**6 + x**2 + 1
    f = x**16 + x**14 - x**10 + x**8 - x**6 + x**2 + 1
    # 断言：检查在环 R 中，多项式 f 是否不是分圆多项式
    assert R.dup_cyclotomic_p(f) is False
    
    # 创建多项式 g = x**16 + x**14 - x**10 - x**8 - x**6 + x**2 + 1
    g = x**16 + x**14 - x**10 - x**8 - x**6 + x**2 + 1
    # 断言：检查在环 R 中，多项式 g 是否是分圆多项式
    assert R.dup_cyclotomic_p(g) is True
    
    # 创建有理数域 QQ 上的多项式环 R，其中变量为 x
    R, x = ring("x", QQ)
    # 断言：检查在环 R 中，多项式 x**2 + x + 1 是否是分圆多项式
    assert R.dup_cyclotomic_p(x**2 + x + 1) is True
    # 断言：检查在环 R 中，有理数 QQ(1,2)*x**2 + x + 1 是否不是分圆多项式
    assert R.dup_cyclotomic_p(QQ(1,2)*x**2 + x + 1) is False
    
    # 创建整数环 ZZ["y"] 上的多项式环 R，其中变量为 x
    R, x = ring("x", ZZ["y"])
    # 断言：检查在环 R 中，多项式 x**2 + x + 1 是否不是分圆多项式
    assert R.dup_cyclotomic_p(x**2 + x + 1) is False
# 定义一个测试函数，用于测试整数环上的多项式的相关方法
def test_dup_zz_cyclotomic_poly():
    # 创建整数环 R 和符号变量 x
    R, x = ring("x", ZZ)

    # 断言不同的 cyclotomic 多项式生成的结果
    assert R.dup_zz_cyclotomic_poly(1) == x - 1
    assert R.dup_zz_cyclotomic_poly(2) == x + 1
    assert R.dup_zz_cyclotomic_poly(3) == x**2 + x + 1
    assert R.dup_zz_cyclotomic_poly(4) == x**2 + 1
    assert R.dup_zz_cyclotomic_poly(5) == x**4 + x**3 + x**2 + x + 1
    assert R.dup_zz_cyclotomic_poly(6) == x**2 - x + 1
    assert R.dup_zz_cyclotomic_poly(7) == x**6 + x**5 + x**4 + x**3 + x**2 + x + 1
    assert R.dup_zz_cyclotomic_poly(8) == x**4 + 1
    assert R.dup_zz_cyclotomic_poly(9) == x**6 + x**3 + 1


# 定义一个测试函数，用于测试整数环上的 cyclotomic 因子分解方法
def test_dup_zz_cyclotomic_factor():
    # 创建整数环 R 和符号变量 x
    R, x = ring("x", ZZ)

    # 断言对于一些特定输入，cyclotomic 因子分解方法的返回结果
    assert R.dup_zz_cyclotomic_factor(0) is None
    assert R.dup_zz_cyclotomic_factor(1) is None

    assert R.dup_zz_cyclotomic_factor(2*x**10 - 1) is None
    assert R.dup_zz_cyclotomic_factor(x**10 - 3) is None
    assert R.dup_zz_cyclotomic_factor(x**10 + x**5 - 1) is None

    assert R.dup_zz_cyclotomic_factor(x + 1) == [x + 1]
    assert R.dup_zz_cyclotomic_factor(x - 1) == [x - 1]

    assert R.dup_zz_cyclotomic_factor(x**2 + 1) == [x**2 + 1]
    assert R.dup_zz_cyclotomic_factor(x**2 - 1) == [x - 1, x + 1]

    assert R.dup_zz_cyclotomic_factor(x**27 + 1) == \
        [x + 1, x**2 - x + 1, x**6 - x**3 + 1, x**18 - x**9 + 1]
    assert R.dup_zz_cyclotomic_factor(x**27 - 1) == \
        [x - 1, x**2 + x + 1, x**6 + x**3 + 1, x**18 + x**9 + 1]


# 定义一个测试函数，用于测试整数环上的多项式因子分解方法
def test_dup_zz_factor():
    # 创建整数环 R 和符号变量 x
    R, x = ring("x", ZZ)

    # 断言对于一些特定输入，多项式因子分解方法的返回结果
    assert R.dup_zz_factor(0) == (0, [])
    assert R.dup_zz_factor(7) == (7, [])
    assert R.dup_zz_factor(-7) == (-7, [])

    assert R.dup_zz_factor_sqf(0) == (0, [])
    assert R.dup_zz_factor_sqf(7) == (7, [])
    assert R.dup_zz_factor_sqf(-7) == (-7, [])

    assert R.dup_zz_factor(2*x + 4) == (2, [(x + 2, 1)])
    assert R.dup_zz_factor_sqf(2*x + 4) == (2, [x + 2])

    f = x**4 + x + 1

    for i in range(0, 20):
        assert R.dup_zz_factor(f) == (1, [(f, 1)])

    assert R.dup_zz_factor(x**2 + 2*x + 2) == \
        (1, [(x**2 + 2*x + 2, 1)])

    assert R.dup_zz_factor(18*x**2 + 12*x + 2) == \
        (2, [(3*x + 1, 2)])

    assert R.dup_zz_factor(-9*x**2 + 1) == \
        (-1, [(3*x - 1, 1),
              (3*x + 1, 1)])

    assert R.dup_zz_factor_sqf(-9*x**2 + 1) == \
        (-1, [3*x - 1,
              3*x + 1])

    # 当使用 flint 作为底层类型时，因子的顺序可能会不同
    c, factors = R.dup_zz_factor(x**3 - 6*x**2 + 11*x - 6)
    assert c == 1
    assert set(factors) == {(x - 3, 1), (x - 2, 1), (x - 1, 1)}

    assert R.dup_zz_factor_sqf(x**3 - 6*x**2 + 11*x - 6) == \
        (1, [x - 3,
             x - 2,
             x - 1])

    assert R.dup_zz_factor(3*x**3 + 10*x**2 + 13*x + 10) == \
        (1, [(x + 2, 1),
             (3*x**2 + 4*x + 5, 1)])

    assert R.dup_zz_factor_sqf(3*x**3 + 10*x**2 + 13*x + 10) == \
        (1, [x + 2,
             3*x**2 + 4*x + 5])
    c, factors = R.dup_zz_factor(-x**6 + x**2)
    # 使用 R.dup_zz_factor 函数对多项式 -x**6 + x**2 进行因式分解，返回系数 c 和因子列表 factors
    assert c == -1
    # 断言：验证 c 的值是否为 -1
    assert set(factors) == {(x, 2), (x - 1, 1), (x + 1, 1), (x**2 + 1, 1)}
    # 断言：验证 factors 的集合是否与给定的因子集合相同

    f = 1080*x**8 + 5184*x**7 + 2099*x**6 + 744*x**5 + 2736*x**4 - 648*x**3 + 129*x**2 - 324

    assert R.dup_zz_factor(f) == \
        (1, [(5*x**4 + 24*x**3 + 9*x**2 + 12, 1),
             (216*x**4 + 31*x**2 - 27, 1)])
    # 断言：验证对多项式 f 进行因式分解后的结果是否与给定的因子列表相同

    f = -29802322387695312500000000000000000000*x**25 \
      + 2980232238769531250000000000000000*x**20 \
      + 1743435859680175781250000000000*x**15 \
      + 114142894744873046875000000*x**10 \
      - 210106372833251953125*x**5 \
      + 95367431640625

    c, factors = R.dup_zz_factor(f)
    # 使用 R.dup_zz_factor 函数对多项式 f 进行因式分解，返回系数 c 和因子列表 factors
    assert c == -95367431640625
    # 断言：验证 c 的值是否为 -95367431640625
    assert set(factors) == {
        (5*x - 1, 1),
        (100*x**2 + 10*x - 1, 2),
        (625*x**4 + 125*x**3 + 25*x**2 + 5*x + 1, 1),
        (10000*x**4 - 3000*x**3 + 400*x**2 - 20*x + 1, 2),
        (10000*x**4 + 2000*x**3 + 400*x**2 + 30*x + 1, 2),
    }
    # 断言：验证 factors 的集合是否与给定的因子集合相同

    f = x**10 - 1

    config.setup('USE_CYCLOTOMIC_FACTOR', True)
    # 设置 config 的 USE_CYCLOTOMIC_FACTOR 参数为 True
    c0, F_0 = R.dup_zz_factor(f)
    # 使用 R.dup_zz_factor 函数对多项式 f 进行因式分解，返回系数 c0 和因子列表 F_0

    config.setup('USE_CYCLOTOMIC_FACTOR', False)
    # 设置 config 的 USE_CYCLOTOMIC_FACTOR 参数为 False
    c1, F_1 = R.dup_zz_factor(f)
    # 使用 R.dup_zz_factor 函数对多项式 f 进行因式分解，返回系数 c1 和因子列表 F_1

    assert c0 == c1 == 1
    # 断言：验证 c0 和 c1 的值是否都为 1
    assert set(F_0) == set(F_1) == {
        (x - 1, 1),
        (x + 1, 1),
        (x**4 - x**3 + x**2 - x + 1, 1),
        (x**4 + x**3 + x**2 + x + 1, 1),
    }
    # 断言：验证 F_0 和 F_1 的集合是否相同

    config.setup('USE_CYCLOTOMIC_FACTOR')
    # 重置 config 的 USE_CYCLOTOMIC_FACTOR 参数

    f = x**10 + 1

    config.setup('USE_CYCLOTOMIC_FACTOR', True)
    # 设置 config 的 USE_CYCLOTOMIC_FACTOR 参数为 True
    F_0 = R.dup_zz_factor(f)
    # 使用 R.dup_zz_factor 函数对多项式 f 进行因式分解，返回因子列表 F_0

    config.setup('USE_CYCLOTOMIC_FACTOR', False)
    # 设置 config 的 USE_CYCLOTOMIC_FACTOR 参数为 False
    F_1 = R.dup_zz_factor(f)
    # 使用 R.dup_zz_factor 函数对多项式 f 进行因式分解，返回因子列表 F_1

    assert F_0 == F_1 == \
        (1, [(x**2 + 1, 1),
             (x**8 - x**6 + x**4 - x**2 + 1, 1)])
    # 断言：验证 F_0 和 F_1 是否相同

    config.setup('USE_CYCLOTOMIC_FACTOR')
    # 重置 config 的 USE_CYCLOTOMIC_FACTOR 参数
def test_dmp_zz_wang():
    # 定义环 R，并初始化变量 x, y, z
    R, x, y, z = ring("x,y,z", ZZ)
    # 定义环 UV，并初始化变量 _x
    UV, _x = ring("x", ZZ)

    # 计算 R.dmp_zz_mignotte_bound(w_1) 的下一个素数
    p = ZZ(nextprime(R.dmp_zz_mignotte_bound(w_1)))
    # 断言 p 的值为 6291469
    assert p == 6291469

    # 初始化多项式和指数
    t_1, k_1, e_1 = y, 1, ZZ(-14)
    t_2, k_2, e_2 = z, 2, ZZ(3)
    t_3, k_3, e_3 = y + z, 2, ZZ(-11)
    t_4, k_4, e_4 = y - z, 1, ZZ(-17)

    # 将多项式 t_i, 系数 k_i, 指数 e_i 组成列表 T, K, E
    T = [t_1, t_2, t_3, t_4]
    K = [k_1, k_2, k_3, k_4]
    E = [e_1, e_2, e_3, e_4]

    # 将 T 中的每个 t.drop(x) 和 K 中的每个 k 组成元组，组成新的 T
    T = zip([t.drop(x) for t in T], K)

    # 初始化 A 列表
    A = [ZZ(-14), ZZ(3)]

    # 计算 S = R.dmp_eval_tail(w_1, A)
    S = R.dmp_eval_tail(w_1, A)
    # 将 S 分解成系数 cs 和 S 本身
    cs, s = UV.dup_primitive(S)

    # 断言 cs 的值为 1，s 的值与特定多项式相等
    assert cs == 1 and s == S == \
        1036728*_x**6 + 915552*_x**5 + 55748*_x**4 + 105621*_x**3 - 17304*_x**2 - 26841*_x - 644

    # 断言 R.dmp_zz_wang_non_divisors(E, cs, ZZ(4)) 的返回值为 [7, 3, 11, 17]
    assert R.dmp_zz_wang_non_divisors(E, cs, ZZ(4)) == [7, 3, 11, 17]
    # 断言 UV.dup_sqf_p(s) 为真，并且 UV.dup_degree(s) == R.dmp_degree(w_1)
    assert UV.dup_sqf_p(s) and UV.dup_degree(s) == R.dmp_degree(w_1)

    # 计算 UV.dup_zz_factor_sqf(s) 的返回值，将其赋值给 _ 和 H
    _, H = UV.dup_zz_factor_sqf(s)

    # 初始化多项式 h_i
    h_1 = 44*_x**2 + 42*_x + 1
    h_2 = 126*_x**2 - 9*_x + 28
    h_3 = 187*_x**2 - 23

    # 断言 H 的值与初始化的多项式列表相等
    assert H == [h_1, h_2, h_3]

    # 初始化 LC 列表
    LC = [lc.drop(x) for lc in [-4*y - 4*z, -y*z**2, y**2 - z**2]]

    # 断言 R.dmp_zz_wang_lead_coeffs(w_1, T, cs, E, H, A) 的返回值
    # 包含 (w_1, H, LC)
    assert R.dmp_zz_wang_lead_coeffs(w_1, T, cs, E, H, A) == (w_1, H, LC)

    # 计算 R.dmp_zz_wang_hensel_lifting(w_1, H, LC, A, p) 的返回值
    factors = R.dmp_zz_wang_hensel_lifting(w_1, H, LC, A, p)
    # 断言 R.dmp_expand(factors) == w_1
    assert R.dmp_expand(factors) == w_1


@XFAIL
def test_dmp_zz_wang_fail():
    # 同样的环初始化步骤
    R, x, y, z = ring("x,y,z", ZZ)
    UV, _x = ring("x", ZZ)

    # 计算 R.dmp_zz_mignotte_bound(w_1) 的下一个素数
    p = ZZ(nextprime(R.dmp_zz_mignotte_bound(w_1)))
    # 断言 p 的值为 6291469
    assert p == 6291469

    # 初始化多项式列表 H_i 和系数列表 c_i
    H_1 = [44*x**2 + 42*x + 1, 126*x**2 - 9*x + 28, 187*x**2 - 23]
    H_2 = [-4*x**2*y - 12*x**2 - 3*x*y + 1, -9*x**2*y - 9*x - 2*y, x**2*y**2 - 9*x**2 + y - 9]
    H_3 = [-4*x**2*y - 12*x**2 - 3*x*y + 1, -9*x**2*y - 9*x - 2*y, x**2*y**2 - 9*x**2 + y - 9]

    c_1 = -70686*x**5 - 5863*x**4 - 17826*x**3 + 2009*x**2 + 5031*x + 74
    c_2 = 9*x**5*y**4 + 12*x**5*y**3 - 45*x**5*y**2 - 108*x**5*y - 324*x**5 + 18*x**4*y**3 - 216*x**4*y**2 - 810*x**4*y + 2*x**3*y**4 + 9*x**3*y**3 - 252*x**3*y**2 - 288*x**3*y - 945*x**3 - 30*x**2*y**2 - 414*x**2*y + 2*x*y**3 - 54*x*y**2 - 3*x*y + 81*x + 12*y
    c_3 = -36*x**4*y**2 - 108*x**4*y - 27*x**3*y**2 - 36*x**3*y - 108*x**3 - 8*x**2*y**2 - 42*x**2*y - 6*x*y**2 + 9*x + 2*y

    # 断言 R.dmp_zz_diophantine(H_i, c_i, []) 的返回值
    # 分别为 [-3*x, -2, 1], [-x*y, -3*x, -6], [0, 0, -1]
    assert R.dmp_zz_diophantine(H_1, c_1, [], 5, p) == [-3*x, -2, 1]
    assert R.dmp_zz_diophantine(H_2, c_2, [ZZ(-14)], 5, p) == [-x*y, -3*x, -6]
    assert R.dmp_zz_diophantine(H_3, c_3, [ZZ(-14)], 5, p) == [0, 0, -1]


def test_issue_6355():
    # 这个测试检查 Wang 算法中的一个错误，该错误仅在特定随机数序列下出现。
    random_sequence = [-1, -1, 0, 0, 0, 0, -1, -1, 0, -1, 3, -1, 3, 3, 3, 3, -1, 3]

    # 同样的环初始化步骤
    R, x, y, z = ring("x,y,z", ZZ)
    # 初始化多项式 f
    f = 2*x**2 + y*z - y - z**2 + z

    # 断言 R.dmp_zz_wang(f, seed=random_sequence) 的返回值等于 [f]
    assert R.dmp_zz_wang(f, seed=random_sequence) == [f]


def test_dmp_zz_factor():
    # 初始化环 R 和变量 x
    R, x = ring("x", ZZ)
    # 断言 R.dmp_zz_factor(0) 的返回值为 (0, [])
    assert R.dmp_zz_factor(0) == (0, [])
    # 断言 R.dmp_zz_factor(7)
    ```python`
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于输入参数 7 的返回结果是否为 (7, [])
        assert R.dmp_zz_factor(7) == (7, [])
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于输入参数 -7 的返回结果是否为 (-7, [])
        assert R.dmp_zz_factor(-7) == (-7, [])
    
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于输入参数 x 的返回结果是否为 (1, [(x, 1)])
        assert R.dmp_zz_factor(x) == (1, [(x, 1)])
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于输入参数 4*x 的返回结果是否为 (4, [(x, 1)])
        assert R.dmp_zz_factor(4*x) == (4, [(x, 1)])
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于输入参数 4*x + 2 的返回结果是否为 (2, [(2*x + 1, 1)])
        assert R.dmp_zz_factor(4*x + 2) == (2, [(2*x + 1, 1)])
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于输入参数 x*y + 1 的返回结果是否为 (1, [(x*y + 1, 1)])
        assert R.dmp_zz_factor(x*y + 1) == (1, [(x*y + 1, 1)])
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于输入参数 y**2 + 1 的返回结果是否为 (1, [(y**2 + 1, 1)])
        assert R.dmp_zz_factor(y**2 + 1) == (1, [(y**2 + 1, 1)])
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于输入参数 y**2 - 1 的返回结果是否为 (1, [(y - 1, 1), (y + 1, 1)])
        assert R.dmp_zz_factor(y**2 - 1) == (1, [(y - 1, 1), (y + 1, 1)])
    
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于复杂多项式输入的返回结果是否正确
        assert R.dmp_zz_factor(x**2*y**2 + 6*x**2*y + 9*x**2 - 1) == (1, [(x*y + 3*x - 1, 1), (x*y + 3*x + 1, 1)])
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于复杂多项式输入的返回结果是否正确
        assert R.dmp_zz_factor(x**2*y**2 - 9) == (1, [(x*y - 3, 1), (x*y + 3, 1)])
    
        # 创建环 R 和变量 x, y, z，用于后续多项式的因式分解操作
        R, x, y, z = ring("x,y,z", ZZ)
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于复杂多项式输入的返回结果是否正确
        assert R.dmp_zz_factor(x**2*y**2*z**2 - 9) == \
            (1, [(x*y*z - 3, 1),
                 (x*y*z + 3, 1)])
    
        # 创建环 R 和变量 x, y, z, u，用于后续多项式的因式分解操作
        R, x, y, z, u = ring("x,y,z,u", ZZ)
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于复杂多项式输入的返回结果是否正确
        assert R.dmp_zz_factor(x**2*y**2*z**2*u**2 - 9) == \
            (1, [(x*y*z*u - 3, 1),
                 (x*y*z*u + 3, 1)])
    
        # 创建环 R 和变量 x, y, z，用于后续多项式的因式分解操作
        R, x, y, z = ring("x,y,z", ZZ)
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于复杂多项式 f_1 的返回结果是否正确
        assert R.dmp_zz_factor(f_1) == \
            (1, [(x + y*z + 20, 1),
                 (x*y + z + 10, 1),
                 (x*z + y + 30, 1)])
    
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于复杂多项式 f_2 的返回结果是否正确
        assert R.dmp_zz_factor(f_2) == \
            (1, [(x**2*y**2 + x**2*z**2 + y + 90, 1),
                 (x**3*y + x**3*z + z - 11, 1)])
    
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于复杂多项式 f_3 的返回结果是否正确
        assert R.dmp_zz_factor(f_3) == \
            (1, [(x**2*y**2 + x*z**4 + x + z, 1),
                 (x**3 + x*y*z + y**2 + y*z**3, 1)])
    
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于复杂多项式 f_4 的返回结果是否正确
        assert R.dmp_zz_factor(f_4) == \
            (-1, [(x*y**3 + z**2, 1),
                  (x**2*z + y**4*z**2 + 5, 1),
                  (x**3*y - z**2 - 3, 1),
                  (x**3*y**4 + z**2, 1)])
    
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于复杂多项式 f_5 的返回结果是否正确
        assert R.dmp_zz_factor(f_5) == \
            (-1, [(x + y - z, 3)])
    
        # 创建环 R 和变量 x, y, z, t，用于后续多项式的因式分解操作
        R, x, y, z, t = ring("x,y,z,t", ZZ)
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于复杂多项式 f_6 的返回结果是否正确
        assert R.dmp_zz_factor(f_6) == \
            (1, [(47*x*y + z**3*t**2 - t**2, 1),
                 (45*x**3 - 9*y**3 - y**2 + 3*z**3 + 2*z*t, 1)])
    
        # 创建环 R 和变量 x, y, z，用于后续多项式的因式分解操作
        R, x, y, z = ring("x,y,z", ZZ)
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于复杂多项式 w_1 的返回结果是否正确
        assert R.dmp_zz_factor(w_1) == \
            (1, [(x**2*y**2 - x**2*z**2 + y - z**2, 1),
                 (x**2*y*z**2 + 3*x*z + 2*y, 1),
                 (4*x**2*y + 4*x**2*z + x*y*z - 1, 1)])
    
        # 创建环 R 和变量 x, y，用于后续多项式的因式分解操作
        R, x, y = ring("x,y", ZZ)
        # 定义多项式 f
        f = -12*x**16*y + 240*x**12*y**3 - 768*x**10*y**4 + 1080*x**8*y**5 - 768*x**6*y**6 + 240*x**4*y**7 - 12*y**9
        # 断言，验证 R 对象的 dmp_zz_factor 方法对于复杂多项式 f 的返回结果是否正确
        assert R.dmp_zz_factor(f) == \
            (-12, [(y, 1),
                   (x**2 - y, 6),
                   (x**4 + 6*x**2*y + y**2, 1)])
def test_dup_qq_i_factor():
    # 创建带有有理数二次扩展因数的环 R 和变量 x
    R, x = ring("x", QQ_I)
    # 创建虚数单位 i
    i = QQ_I(0, 1)

    # 断言：计算 x^2 - 2 的二次扩展因数
    assert R.dup_qq_i_factor(x**2 - 2) == (QQ_I(1, 0), [(x**2 - 2, 1)])

    # 断言：计算 x^2 - 1 的二次扩展因数
    assert R.dup_qq_i_factor(x**2 - 1) == (QQ_I(1, 0), [(x - 1, 1), (x + 1, 1)])

    # 断言：计算 x^2 + 1 的二次扩展因数
    assert R.dup_qq_i_factor(x**2 + 1) == (QQ_I(1, 0), [(x - i, 1), (x + i, 1)])

    # 断言：计算 x^2/4 + 1 的二次扩展因数
    assert R.dup_qq_i_factor(x**2/4 + 1) == \
            (QQ_I(QQ(1, 4), 0), [(x - 2*i, 1), (x + 2*i, 1)])

    # 断言：计算 x^2 + 4 的二次扩展因数
    assert R.dup_qq_i_factor(x**2 + 4) == \
            (QQ_I(1, 0), [(x - 2*i, 1), (x + 2*i, 1)])

    # 断言：计算 x^2 + 2*x + 1 的二次扩展因数
    assert R.dup_qq_i_factor(x**2 + 2*x + 1) == \
            (QQ_I(1, 0), [(x + 1, 2)])

    # 断言：计算 x^2 + 2*i*x - 1 的二次扩展因数
    assert R.dup_qq_i_factor(x**2 + 2*i*x - 1) == \
            (QQ_I(1, 0), [(x + i, 2)])

    # 创建多项式 f
    f = 8192*x**2 + x*(22656 + 175232*i) - 921416 + 242313*i

    # 断言：计算 f 的二次扩展因数
    assert R.dup_qq_i_factor(f) == \
            (QQ_I(8192, 0), [(x + QQ_I(QQ(177, 128), QQ(1369, 128)), 2)])


def test_dmp_qq_i_factor():
    # 创建带有有理数二次扩展因数的环 R 和变量 x, y
    R, x, y = ring("x, y", QQ_I)
    # 创建虚数单位 i
    i = QQ_I(0, 1)

    # 断言：计算 x^2 + 2*y^2 的二次扩展因数
    assert R.dmp_qq_i_factor(x**2 + 2*y**2) == \
            (QQ_I(1, 0), [(x**2 + 2*y**2, 1)])

    # 断言：计算 x^2 + y^2 的二次扩展因数
    assert R.dmp_qq_i_factor(x**2 + y**2) == \
            (QQ_I(1, 0), [(x - i*y, 1), (x + i*y, 1)])

    # 断言：计算 x^2 + y^2/4 的二次扩展因数
    assert R.dmp_qq_i_factor(x**2 + y**2/4) == \
            (QQ_I(1, 0), [(x - i*y/2, 1), (x + i*y/2, 1)])

    # 断言：计算 4*x^2 + y^2 的二次扩展因数
    assert R.dmp_qq_i_factor(4*x**2 + y**2) == \
            (QQ_I(4, 0), [(x - i*y/2, 1), (x + i*y/2, 1)])


def test_dup_zz_i_factor():
    # 创建带有整数二次扩展因数的环 R 和变量 x
    R, x = ring("x", ZZ_I)
    # 创建虚数单位 i
    i = ZZ_I(0, 1)

    # 断言：计算 x^2 - 2 的二次扩展因数
    assert R.dup_zz_i_factor(x**2 - 2) == (ZZ_I(1, 0), [(x**2 - 2, 1)])

    # 断言：计算 x^2 - 1 的二次扩展因数
    assert R.dup_zz_i_factor(x**2 - 1) == (ZZ_I(1, 0), [(x - 1, 1), (x + 1, 1)])

    # 断言：计算 x^2 + 1 的二次扩展因数
    assert R.dup_zz_i_factor(x**2 + 1) == (ZZ_I(1, 0), [(x - i, 1), (x + i, 1)])

    # 断言：计算 x^2 + 4 的二次扩展因数
    assert R.dup_zz_i_factor(x**2 + 4) == \
            (ZZ_I(1, 0), [(x - 2*i, 1), (x + 2*i, 1)])

    # 断言：计算 x^2 + 2*x + 1 的二次扩展因数
    assert R.dup_zz_i_factor(x**2 + 2*x + 1) == \
            (ZZ_I(1, 0), [(x + 1, 2)])

    # 断言：计算 x^2 + 2*i*x - 1 的二次扩展因数
    assert R.dup_zz_i_factor(x**2 + 2*i*x - 1) == \
            (ZZ_I(1, 0), [(x + i, 2)])

    # 创建多项式 f
    f = 8192*x**2 + x*(22656 + 175232*i) - 921416 + 242313*i

    # 断言：计算 f 的二次扩展因数
    assert R.dup_zz_i_factor(f) == \
            (ZZ_I(0, 1), [((64 - 64*i)*x + (773 + 596*i), 2)])


def test_dmp_zz_i_factor():
    # 创建带有整数二次扩展因数的环 R 和变量 x, y
    R, x, y = ring("x, y", ZZ_I)
    # 创建虚数单位 i
    i = ZZ_I(0, 1)

    # 断言：计算 x^2 + 2*y^2 的二次扩展因数
    assert R.dmp_zz_i_factor(x**2 + 2*y**2) == \
            (ZZ_I(1, 0), [(x**2 + 2*y**2, 1)])

    # 断言：计算 x^2 + y^2 的二次扩展因数
    assert R.dmp_zz_i_factor(x**2 + y**2) == \
            (ZZ_I(1, 0), [(x - i*y, 1), (x + i*y, 1)])

    # 断言：计算 4*x^2 + y^2 的二次扩展因数
    assert R.dmp_zz_i_factor(4*x**2 + y**2) == \
            (ZZ_I(1, 0), [(2*x - i*y, 1), (2*x + i*y, 1)])


def test_dup_ext_factor():
    # 创建带有二次扩展因数的环 R 和变量 x
    R, x = ring("x", QQ.algebraic_field(I))

    # 定义 ANP 函数，生成带有有理数的扩展因数
    def anp(element):
        return ANP(element, [QQ(1), QQ(0), QQ(1)], QQ)

    # 断言：计算 0 的二次扩展因数
    assert R.dup_ext_factor(0) == (anp([]), [])

    # 创建多项式 f
    f = anp([QQ(1)])*x + anp([QQ(1)])

    # 断言：计算 f 的二次扩展因数
    assert R.dup_ext_factor(f) == (an
    g = anp([QQ(1)])*x**4 + anp([QQ(1, 7)])
    # 定义多项式 g = QQ(1)*x^4 + QQ(1/7)，使用 anp 函数构造

    assert R.dup_ext_factor(f) == (anp([QQ(7)]), [(g, 1)])
    # 断言 R.dup_ext_factor(f) 的返回值与预期值相等：(QQ(7), [(g, 1)])，即 ext_factor 函数应返回 (QQ(7), [(g, 1)]) 的结果

    f = anp([QQ(1)])*x**4 + anp([QQ(1)])
    # 定义多项式 f = QQ(1)*x^4 + QQ(1)，使用 anp 函数构造

    assert R.dup_ext_factor(f) == \
        (anp([QQ(1, 1)]), [(anp([QQ(1)])*x**2 + anp([QQ(-1), QQ(0)]), 1),
                           (anp([QQ(1)])*x**2 + anp([QQ( 1), QQ(0)]), 1)])
    # 断言 R.dup_ext_factor(f) 的返回值与预期值相等：
    # (QQ(1, 1), [(QQ(1)*x^2 + QQ(-1) + QQ(0), 1), (QQ(1)*x^2 + QQ(1) + QQ(0), 1)])

    f = anp([QQ(4, 1)])*x**2 + anp([QQ(9, 1)])
    # 定义多项式 f = QQ(4/1)*x^2 + QQ(9/1)，使用 anp 函数构造

    assert R.dup_ext_factor(f) == \
        (anp([QQ(4, 1)]), [(anp([QQ(1, 1)])*x + anp([-QQ(3, 2), QQ(0, 1)]), 1),
                           (anp([QQ(1, 1)])*x + anp([ QQ(3, 2), QQ(0, 1)]), 1)])
    # 断言 R.dup_ext_factor(f) 的返回值与预期值相等：
    # (QQ(4/1), [(QQ(1/1)*x + QQ(-3/2, 0/1), 1), (QQ(1/1)*x + QQ(3/2, 0/1), 1)])

    f = anp([QQ(4, 1)])*x**4 + anp([QQ(8, 1)])*x**3 + anp([QQ(77, 1)])*x**2 + anp([QQ(18, 1)])*x + anp([QQ(153, 1)])
    # 定义多项式 f = QQ(4/1)*x^4 + QQ(8/1)*x^3 + QQ(77/1)*x^2 + QQ(18/1)*x + QQ(153/1)，使用 anp 函数构造

    assert R.dup_ext_factor(f) == \
        (anp([QQ(4, 1)]), [(anp([QQ(1, 1)])*x + anp([-QQ(4, 1), QQ(1, 1)]), 1),
                           (anp([QQ(1, 1)])*x + anp([-QQ(3, 2), QQ(0, 1)]), 1),
                           (anp([QQ(1, 1)])*x + anp([ QQ(3, 2), QQ(0, 1)]), 1),
                           (anp([QQ(1, 1)])*x + anp([ QQ(4, 1), QQ(1, 1)]), 1)])
    # 断言 R.dup_ext_factor(f) 的返回值与预期值相等：
    # (QQ(4/1), [(QQ(1/1)*x + QQ(-4/1, 1/1), 1), (QQ(1/1)*x + QQ(-3/2, 0/1), 1),
    #           (QQ(1/1)*x + QQ(3/2, 0/1), 1), (QQ(1/1)*x + QQ(4/1, 1/1), 1)])

    R, x = ring("x", QQ.algebraic_field(sqrt(2)))
    # 创建有理函数环 R，变量 x，以 QQ.algebraic_field(sqrt(2)) 为基础

    def anp(element):
        return ANP(element, [QQ(1), QQ(0), QQ(-2)], QQ)
    # 定义函数 anp，用于创建代数数

    f = anp([QQ(1)])*x**4 + anp([QQ(1, 1)])
    # 定义多项式 f = QQ(1)*x^4 + QQ(1/1)，使用 anp 函数构造

    assert R.dup_ext_factor(f) == \
        (anp([QQ(1)]), [(anp([QQ(1)])*x**2 + anp([QQ(-1), QQ(0)])*x + anp([QQ(1)]), 1),
                        (anp([QQ(1)])*x**2 + anp([QQ( 1), QQ(0)])*x + anp([QQ(1)]), 1)])
    # 断言 R.dup_ext_factor(f) 的返回值与预期值相等：
    # (QQ(1), [(QQ(1)*x^2 + QQ(-1)*x + QQ(1), 1), (QQ(1)*x^2 + QQ(1)*x + QQ(1), 1)])

    f = anp([QQ(1, 1)])*x**2 + anp([QQ(2), QQ(0)])*x + anp([QQ(2, 1)])
    # 定义多项式 f = QQ(1/1)*x^2 + QQ(2, 0)*x + QQ(2/1)，使用 anp 函数构造

    assert R.dup_ext_factor(f) == \
        (anp([QQ(1, 1)]), [(anp([1])*x + anp([1, 0]), 2)])
    # 断言 R.dup_ext_factor(f) 的返回值与预期值相等：
    # (QQ(1/1), [(x + QQ(1, 0), 2)])

    assert R.dup_ext_factor(f**3) == \
        (anp([QQ(1, 1)]), [(anp([1])*x + anp([1, 0]), 6)])
    # 断言 R.dup_ext_factor(f**3) 的返回值与预期值相等：
    # (QQ(1/1), [(x + QQ(1, 0), 6)])

    f *= anp([QQ(2, 1)])
    # 将 f 乘以 QQ(2/1)，更新 f 的值

    assert R.dup_ext_factor(f) == \
        (anp([QQ(2, 1)]), [(anp([1])*x + anp([1, 0]), 2)])
    # 断言 R.dup_ext_factor(f) 的返回值与预期值相等：
    # (QQ(2/1), [(x + QQ(1, 0), 2)])

    assert R.dup_ext_factor(f**3) == \
        (anp([QQ(8, 1)]), [(anp([1])*x + anp([1, 0]), 6)])
    # 断言 R.dup_ext_factor(f**3) 的返回值与预期值相等：
    # (QQ(8/1), [(x + QQ(1, 0), 6)])
def test_dmp_ext_factor():
    # 使用 QQ 字段的平方根 2 创建代数扩展域 K
    K = QQ.algebraic_field(sqrt(2))
    # 创建一个多项式环 R，变量为 x, y，使用 K 作为系数环
    R, x, y = ring("x,y", K)
    # 获取 K 中的单位元素 sqrt(2)
    sqrt2 = K.unit

    # 定义一个辅助函数 anp，用来创建具有指定系数和阶的多项式
    def anp(x):
        return ANP(x, [QQ(1), QQ(0), QQ(-2)], QQ)

    # 断言语句，测试 R.dmp_ext_factor(0) 的返回值
    assert R.dmp_ext_factor(0) == (anp([]), [])

    # 构造一个多项式 f = (1)*x + (1)
    f = anp([QQ(1)]) * x + anp([QQ(1)])
    # 断言语句，测试 R.dmp_ext_factor(f) 的返回值
    assert R.dmp_ext_factor(f) == (anp([QQ(1)]), [(f, 1)])

    # 构造一个多项式 g = (2)*x + (2)
    g = anp([QQ(2)]) * x + anp([QQ(2)])
    # 断言语句，测试 R.dmp_ext_factor(g) 的返回值
    assert R.dmp_ext_factor(g) == (anp([QQ(2)]), [(f, 1)])

    # 构造一个多项式 f = (1)*x**2 + (-2)*y**2
    f = anp([QQ(1)]) * x**2 + anp([QQ(-2)]) * y**2
    # 断言语句，测试 R.dmp_ext_factor(f) 的返回值
    assert R.dmp_ext_factor(f) == \
        (anp([QQ(1)]), [(anp([QQ(1)]) * x + anp([QQ(-1), QQ(0)]) * y, 1),
                        (anp([QQ(1)]) * x + anp([QQ(1), QQ(0)]) * y, 1)])

    # 构造一个多项式 f = (2)*x**2 + (-4)*y**2
    f = anp([QQ(2)]) * x**2 + anp([QQ(-4)]) * y**2
    # 断言语句，测试 R.dmp_ext_factor(f) 的返回值
    assert R.dmp_ext_factor(f) == \
        (anp([QQ(2)]), [(anp([QQ(1)]) * x + anp([QQ(-1), QQ(0)]) * y, 1),
                        (anp([QQ(1)]) * x + anp([QQ(1), QQ(0)]) * y, 1)])

    # 定义三个多项式 f1, f2, f3
    f1 = y + 1
    f2 = y + sqrt2
    f3 = x**2 + x + 2 + 3 * sqrt2
    # 构造一个复合多项式 f = f1**2 * f2**2 * f3**2
    f = f1**2 * f2**2 * f3**2
    # 断言语句，测试 R.dmp_ext_factor(f) 的返回值
    assert R.dmp_ext_factor(f) == (K.one, [(f1, 2), (f2, 2), (f3, 2)])


def test_dup_factor_list():
    # 创建一个整数多项式环 R，变量为 x
    R, x = ring("x", ZZ)
    # 断言语句，测试 R.dup_factor_list(0) 的返回值
    assert R.dup_factor_list(0) == (0, [])
    # 断言语句，测试 R.dup_factor_list(7) 的返回值
    assert R.dup_factor_list(7) == (7, [])

    # 创建一个有理数多项式环 R，变量为 x
    R, x = ring("x", QQ)
    # 断言语句，测试 R.dup_factor_list(0) 的返回值
    assert R.dup_factor_list(0) == (0, [])
    # 断言语句，测试 R.dup_factor_list(QQ(1, 7)) 的返回值
    assert R.dup_factor_list(QQ(1, 7)) == (QQ(1, 7), [])

    # 创建一个整数多项式环 R，变量为 x，其中的系数是关于 t 的多项式环 ZZ['t']
    R, x = ring("x", ZZ['t'])
    # 断言语句，测试 R.dup_factor_list(0) 的返回值
    assert R.dup_factor_list(0) == (0, [])
    # 断言语句，测试 R.dup_factor_list(7) 的返回值
    assert R.dup_factor_list(7) == (7, [])

    # 创建一个有理数多项式环 R，变量为 x，其中的系数是关于 t 的多项式环 QQ['t']
    R, x = ring("x", QQ['t'])
    # 断言语句，测试 R.dup_factor_list(0) 的返回值
    assert R.dup_factor_list(0) == (0, [])
    # 断言语句，测试 R.dup_factor_list(QQ(1, 7)) 的返回值
    assert R.dup_factor_list(QQ(1, 7)) == (QQ(1, 7), [])

    # 创建一个整数多项式环 R，变量为 x
    R, x = ring("x", ZZ)
    # 断言语句，测试 R.dup_factor_list_include(0) 的返回值
    assert R.dup_factor_list_include(0) == [(0, 1)]
    # 断言语句，测试 R.dup_factor_list_include(7) 的返回值
    assert R.dup_factor_list_include(7) == [(7, 1)]

    # 断言语句，测试 R.dup_factor_list(x**2 + 2*x + 1) 的返回值
    assert R.dup_factor_list(x**2 + 2*x + 1) == (1, [(x + 1, 2)])
    # 断言语句，测试 R.dup_factor_list_include(x**2 + 2*x + 1) 的返回值
    assert R.dup_factor_list_include(x**2 + 2*x + 1) == [(x + 1, 2)]
    
    # issue 8037
    # 断言语句，测试 R.dup_factor_list(6*x**2 - 5*x - 6) 的返回值
    assert R.dup_factor_list(6*x**2 - 5*x - 6) == (1, [(2*x - 3, 1), (3*x + 2, 1)])

    # 创建一个有理数多项式环 R，变量为 x
    R, x = ring("x", QQ)
    # 断言语句，测试 R.dup_factor_list(QQ(1,2)*x**2 + x + QQ(1,2)) 的返回值
    assert R.dup_factor_list(QQ(1, 2)*x**2 + x + QQ(1, 2)) == (QQ(1, 2), [(x + 1, 2)])

    # 创建一个有限域 FF(2) 上的多项式环 R，变量为 x
    R, x = ring("x", FF(2))
    # 断言语句，测试 R.dup_factor_list(x**2 + 1) 的返回值
    assert R.dup_factor_list(x**2 + 1) == (1, [(x + 1, 2)])

    # 创建一个实数多项式环 R，变量为 x
    R, x = ring("x", RR)
    # 断言语句，测试 R.dup_factor_list(1.0*x**2 + 2.0*x + 1.0) 的返回值
    assert R.dup_factor_list(1.0*x**2 + 2.0*x + 1.0) == (1.0, [(1.0*x + 1.0, 2)])
    # 断言语句，测试 R.dup_factor_list(2.0*x**2 + 4.0*x + 2.0) 的返回值
    assert R.dup_factor_list(2.0*x**2 + 4.0*x + 2.0) == (2.0, [(1.0*x + 1.0, 2)])

    # 创建一个实数多项式 f
    f = 6.7225336055071*x**2 - 10.6463972754741*x - 0.33469524022264
    # 获取 R
    # 定义函数 anp，接受一个元素作为参数，返回其关于 QQ 的 ANP 对象
    def anp(element):
        return ANP(element, [QQ(1), QQ(0), QQ(1)], QQ)

    # 使用 anp 函数创建多项式 f = anp([QQ(1, 1)])*x**4 + anp([QQ(2, 1)])*x**2
    f = anp([QQ(1, 1)])*x**4 + anp([QQ(2, 1)])*x**2

    # 断言 f 的因式分解结果与预期相等
    assert R.dup_factor_list(f) == \
        (anp([QQ(1, 1)]), [(anp([QQ(1, 1)])*x, 2),
                           (anp([QQ(1, 1)])*x**2 + anp([])*x + anp([QQ(2, 1)]), 1)])

    # 创建环 R 和未知数 x，并在执行期间引发 DomainError 异常
    R, x = ring("x", EX)
    raises(DomainError, lambda: R.dup_factor_list(EX(sin(1))))
# 定义一个测试函数，用于测试多项式环的因子列表函数
def test_dmp_factor_list():
    # 创建整数环 R，并定义变量 x, y
    R, x, y = ring("x,y", ZZ)
    # 断言多项式的因子列表，对于常数多项式 0，预期结果是 (0, [])
    assert R.dmp_factor_list(0) == (ZZ(0), [])
    # 断言多项式的因子列表，对于常数多项式 7，预期结果是 (7, [])
    assert R.dmp_factor_list(7) == (7, [])

    # 创建有理数环 R，并定义变量 x, y
    R, x, y = ring("x,y", QQ)
    # 断言多项式的因子列表，对于常数多项式 0，预期结果是 (0, [])
    assert R.dmp_factor_list(0) == (QQ(0), [])
    # 断言多项式的因子列表，对于有理数 1/7，预期结果是 (1/7, [])
    assert R.dmp_factor_list(QQ(1, 7)) == (QQ(1, 7), [])

    # 创建整数环 Rt，并定义变量 t
    Rt, t = ring("t", ZZ)
    # 创建整数环 R，定义变量 x, y，使用 Rt 作为系数环
    R, x, y = ring("x,y", Rt)
    # 断言多项式的因子列表，对于常数多项式 0，预期结果是 (0, [])
    assert R.dmp_factor_list(0) == (0, [])
    # 断言多项式的因子列表，对于常数多项式 7，预期结果是 (7, [])
    assert R.dmp_factor_list(7) == (ZZ(7), [])

    # 创建有理数环 Rt，并定义变量 t
    Rt, t = ring("t", QQ)
    # 创建有理数环 R，定义变量 x, y，使用 Rt 作为系数环
    R, x, y = ring("x,y", Rt)
    # 断言多项式的因子列表，对于常数多项式 0，预期结果是 (0, [])
    assert R.dmp_factor_list(0) == (0, [])
    # 断言多项式的因子列表，对于有理数 1/7，预期结果是 (1/7, [])
    assert R.dmp_factor_list(QQ(1, 7)) == (QQ(1, 7), [])

    # 创建整数环 R，并定义变量 x
    R, x = ring("x", ZZ)
    # 断言多项式的因子列表，对于 x^2 + 2x + 1，预期结果是 (1, [(x + 1, 2)])
    assert R.dmp_factor_list(x**2 + 2*x + 1) == (1, [(x + 1, 2)])

    # 创建有理数环 R，并定义变量 x
    R, x = ring("x", QQ)
    # 断言多项式的因子列表，对于 (1/2)x^2 + x + (1/2)，预期结果是 ((1/2), [(x + 1, 2)])
    assert R.dmp_factor_list(QQ(1,2)*x**2 + x + QQ(1,2)) == (QQ(1,2), [(x + 1, 2)])

    # 创建整数环 R，并定义变量 x, y
    R, x, y = ring("x,y", ZZ)
    # 断言多项式的因子列表，对于 x^2 + 2x + 1，预期结果是 (1, [(x + 1, 2)])
    assert R.dmp_factor_list(x**2 + 2*x + 1) == (1, [(x + 1, 2)])

    # 创建有理数环 R，并定义变量 x, y
    R, x, y = ring("x,y", QQ)
    # 断言多项式的因子列表，对于 (1/2)x^2 + x + (1/2)，预期结果是 ((1/2), [(x + 1, 2)])
    assert R.dmp_factor_list(QQ(1,2)*x**2 + x + QQ(1,2)) == (QQ(1,2), [(x + 1, 2)])

    # 创建整数环 R，并定义变量 x, y
    R, x, y = ring("x,y", ZZ)
    # 定义多项式 f = 4x^2y + 4xy^2
    f = 4*x**2*y + 4*x*y**2
    # 断言多项式的因子列表，预期结果是 (4, [(y, 1), (x, 1), (x + y, 1)])
    assert R.dmp_factor_list(f) == \
        (4, [(y, 1),
             (x, 1),
             (x + y, 1)])

    # 断言多项式的因子列表（包括因子的系数），预期结果是 [(4*y, 1), (x, 1), (x + y, 1)]
    assert R.dmp_factor_list_include(f) == \
        [(4*y, 1),
         (x, 1),
         (x + y, 1)]

    # 创建有理数环 R，并定义变量 x, y
    R, x, y = ring("x,y", QQ)
    # 定义多项式 f = (1/2)x^2y + (1/2)xy^2
    f = QQ(1,2)*x**2*y + QQ(1,2)*x*y**2
    # 断言多项式的因子列表，预期结果是 ((1/2), [(y, 1), (x, 1), (x + y, 1)])
    assert R.dmp_factor_list(f) == \
        (QQ(1,2), [(y, 1),
                   (x, 1),
                   (x + y, 1)])

    # 创建实数环 R，并定义变量 x, y
    R, x, y = ring("x,y", RR)
    # 定义多项式 f = 2.0*x^2 - 8.0*y^2
    f = 2.0*x**2 - 8.0*y**2
    # 断言多项式的因子列表，预期结果是 (8.0, [(0.5*x - y, 1), (0.5*x + y, 1)])
    assert R.dmp_factor_list(f) == \
        (RR(8.0), [(0.5*x - y, 1),
                   (0.5*x + y, 1)])

    # 定义复杂的多项式 f
    f = 6.7225336055071*x**2*y**2 - 10.6463972754741*x*y - 0.33469524022264
    # 调用因子列表函数，获取结果的系数和因子列表
    coeff, factors = R.dmp_factor_list(f)
    # 断言结果的系数为 10.6463972754741
    assert coeff == RR(10.6463972754741)
    # 断言因子列表的长度为 1
    assert len(factors) == 1
    # 断言因子列表中第一个因子的最大范数为 1.0
    assert factors[0][0].max_norm() == RR(1.0)
    # 断言因子列表中第一个因子的次数为 1
    assert factors[0][1] == 1

    # 创建整数环 Rt，并定义变量 t
    Rt, t = ring("t", ZZ)
    # 创建整数环 R，定义变量 x, y，使用 Rt 作为系数环
    R, x, y = ring("x,y", Rt)
    # 定义多项式 f = 4tx^2 + 4t^2x
    f = 4*t*x**2 + 4*t**2*x
    # 断言多项式的因子列表，预期结果是 (4t, [(x, 1), (x + t, 1)])
    assert R.dmp_factor_list(f) == \
        (4*t, [(x, 1),
             (x + t, 1)])

    # 创建有理数环 Rt，并定义变量 t
    Rt, t = ring("t", QQ)
    # 创建有理数环 R，定义变量 x, y，使用 Rt 作为系数环
    R, x, y =
`
# 定义测试函数，检验多项式环中的不可约性质
def test_dmp_irreducible_p():
    # 创建多项式环 R，包含变量 x 和 y，使用整数环 ZZ
    R, x, y = ring("x,y", ZZ)
    # 断言多项式 x^2 + x + 1 在环 R 中是不可约的
    assert R.dmp_irreducible_p(x**2 + x + 1) is True
    # 断言多项式 x^2 + 2*x + 1 在环 R 中是可约的
    assert R.dmp_irreducible_p(x**2 + 2*x + 1) is False
```