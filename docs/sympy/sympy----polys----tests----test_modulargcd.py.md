# `D:\src\scipysrc\sympy\sympy\polys\tests\test_modulargcd.py`

```
# 导入 sympy.polys.rings 模块中的 ring 函数，用于创建多项式环
# 导入 sympy.polys.domains 模块中的 ZZ, QQ, AlgebraicField，分别表示整数环、有理数环和代数域
# 导入 sympy.polys.modulargcd 模块中的各个函数，用于多项式的模最大公约数计算
# 导入 sympy.functions.elementary.miscellaneous 模块中的 sqrt 函数
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, AlgebraicField
from sympy.polys.modulargcd import (
    modgcd_univariate,
    modgcd_bivariate,
    _chinese_remainder_reconstruction_multivariate,
    modgcd_multivariate,
    _to_ZZ_poly,
    _to_ANP_poly,
    func_field_modgcd,
    _func_field_modgcd_m)
from sympy.functions.elementary.miscellaneous import sqrt


# 定义测试函数 test_modgcd_univariate_integers，用于测试一元整数环的模最大公约数
def test_modgcd_univariate_integers():
    # 创建一元整数环 R，变量 x
    R, x = ring("x", ZZ)

    # 初始化多项式 f, g 为零
    f, g = R.zero, R.zero
    # 断言计算 f 和 g 的模最大公约数，预期结果是 (0, 0, 0)
    assert modgcd_univariate(f, g) == (0, 0, 0)

    # 设置 f 为零，g 为 x
    f, g = R.zero, x
    # 断言计算 f 和 g 的模最大公约数，预期结果是 (x, 0, 1)
    assert modgcd_univariate(f, g) == (x, 0, 1)
    # 断言计算 g 和 f 的模最大公约数，预期结果是 (x, 1, 0)
    assert modgcd_univariate(g, f) == (x, 1, 0)

    # 设置 f 为零，g 为 -x
    f, g = R.zero, -x
    # 断言计算 f 和 g 的模最大公约数，预期结果是 (x, 0, -1)
    assert modgcd_univariate(f, g) == (x, 0, -1)
    # 断言计算 g 和 f 的模最大公约数，预期结果是 (x, -1, 0)
    assert modgcd_univariate(g, f) == (x, -1, 0)

    # 设置 f 为 2*x，g 为 2
    f, g = 2*x, R(2)
    # 断言计算 f 和 g 的模最大公约数，预期结果是 (2, x, 1)
    assert modgcd_univariate(f, g) == (2, x, 1)

    # 设置 f 为 2*x + 2，g 为 6*x**2 - 6
    f, g = 2*x + 2, 6*x**2 - 6
    # 断言计算 f 和 g 的模最大公约数，预期结果是 (2*x + 2, 1, 3*x - 3)
    assert modgcd_univariate(f, g) == (2*x + 2, 1, 3*x - 3)

    # 设置 f 和 g 为具体的多项式
    f = x**4 + 8*x**3 + 21*x**2 + 22*x + 8
    g = x**3 + 6*x**2 + 11*x + 6
    h = x**2 + 3*x + 2
    cff = x**2 + 5*x + 4
    cfg = x + 3
    # 断言计算 f 和 g 的模最大公约数，预期结果是 (h, cff, cfg)
    assert modgcd_univariate(f, g) == (h, cff, cfg)

    # 设置 f 和 g 为具体的多项式
    f = x**4 - 4
    g = x**4 + 4*x**2 + 4
    h = x**2 + 2
    cff = x**2 - 2
    cfg = x**2 + 2
    # 断言计算 f 和 g 的模最大公约数，预期结果是 (h, cff, cfg)
    assert modgcd_univariate(f, g) == (h, cff, cfg)

    # 设置 f 和 g 为具体的多项式
    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    h = 1
    cff = f
    cfg = g
    # 断言计算 f 和 g 的模最大公约数，预期结果是 (h, cff, cfg)
    assert modgcd_univariate(f, g) == (h, cff, cfg)

    # 设置 f 和 g 为非常大的多项式
    f = - 352518131239247345597970242177235495263669787845475025293906825864749649589178600387510272*x**49 \
        + 46818041807522713962450042363465092040687472354933295397472942006618953623327997952*x**42 \
        + 378182690892293941192071663536490788434899030680411695933646320291525827756032*x**35 \
        + 112806468807371824947796775491032386836656074179286744191026149539708928*x**28 \
        - 12278371209708240950316872681744825481125965781519138077173235712*x**21 \
        + 289127344604779611146960547954288113529690984687482920704*x**14 \
        + 19007977035740498977629742919480623972236450681*x**7 \
        + 311973482284542371301330321821976049
    g =   365431878023781158602430064717380211405897160759702125019136*x**21 \
        + 197599133478719444145775798221171663643171734081650688*x**14 \
        - 9504116979659010018253915765478924103928886144*x**7 \
        - 311973482284542371301330321821976049
    # 断言计算 f 和 f 的一阶导数的模最大公约数，预期结果是 g
    assert modgcd_univariate(f, f.diff(x))[0] == g

    # 设置 f 和 g 为具体的多项式
    f = 1317378933230047068160*x + 2945748836994210856960
    g = 120352542776360960*x + 269116466014453760
    h = 120352542776360960*x + 269116466014453760
    cff = 10946
    cfg = 1
    # 断言计算 f 和 g 的模最大公约数，预期结果是 (h, cff, cfg)
    assert modgcd_univariate(f, g) == (h, cff, cfg)


# 定义测试函数 test_modgcd_bivariate_integers，用于测试二元整数环的模最大公约数
def test_modgcd_bivariate_integers():
    # 创建二元整数环 R，变量 x, y
    R, x, y = ring("x,y", ZZ)

    # 初始化多项式 f, g 为零
    f, g = R.zero, R.zero
    # 断言计算 f 和 g 的模最大公约数，预期结果是 (0, 0, 0)
    assert modgcd_bivariate(f, g) == (0, 0, 0)

    # 设置 f 为 2*x，g 为 2
    f, g = 2*x, R(2)
    # 断言计算 f 和 g 的模最大公约数，预期结果是 (2, x, 1)
    assert modgcd_bivariate(f, g) == (2, x, 1)

    # 设置 f 为 x + 2*y，g 为 x + y
    f, g = x + 2*y, x + y
    # 断言计算 f 和 g 的模最大公约数，预期结果是 (1, f
    # 计算 f 和 g 的多项式表达式，分别赋给 f 和 g
    f, g = x**2 + 2*x*y + y**2, x**3 + y**3
    # 断言 modgcd_bivariate(f, g) 的结果与给定的元组相等
    assert modgcd_bivariate(f, g) == (x + y, x + y, x**2 - x*y + y**2)

    # 计算 f 和 g 的多项式表达式，分别赋给 f 和 g
    f, g = x*y**2 + 2*x*y + x, x*y**3 + x
    # 断言 modgcd_bivariate(f, g) 的结果与给定的元组相等
    assert modgcd_bivariate(f, g) == (x*y + x, y + 1, y**2 - y + 1)

    # 计算 f 和 g 的多项式表达式，分别赋给 f 和 g
    f, g = x**2*y**2 + x**2*y + 1, x*y**2 + x*y + 1
    # 断言 modgcd_bivariate(f, g) 的结果与给定的元组相等
    assert modgcd_bivariate(f, g) == (1, f, g)

    # 计算 f 和 g 的多项式表达式，分别赋给 f 和 g
    f = 2*x*y**2 + 4*x*y + 2*x + y**2 + 2*y + 1
    g = 2*x*y**3 + 2*x + y**3 + 1
    # 断言 modgcd_bivariate(f, g) 的结果与给定的元组相等
    assert modgcd_bivariate(f, g) == (2*x*y + 2*x + y + 1, y + 1, y**2 - y + 1)

    # 计算 f 和 g 的多项式表达式，分别赋给 f 和 g
    f, g = 2*x**2 + 4*x + 2, x + 1
    # 断言 modgcd_bivariate(f, g) 的结果与给定的元组相等
    assert modgcd_bivariate(f, g) == (x + 1, 2*x + 2, 1)

    # 计算 f 和 g 的多项式表达式，分别赋给 f 和 g
    f, g = x + 1, 2*x**2 + 4*x + 2
    # 断言 modgcd_bivariate(f, g) 的结果与给定的元组相等
    assert modgcd_bivariate(f, g) == (x + 1, 1, 2*x + 2)

    # 计算 f 和 g 的多项式表达式，分别赋给 f 和 g
    f = 2*x**2 + 4*x*y - 2*x - 4*y
    g = x**2 + x - 2
    # 断言 modgcd_bivariate(f, g) 的结果与给定的元组相等
    assert modgcd_bivariate(f, g) == (x - 1, 2*x + 4*y, x + 2)

    # 计算 f 和 g 的多项式表达式，分别赋给 f 和 g
    f = 2*x**2 + 2*x*y - 3*x - 3*y
    g = 4*x*y - 2*x + 4*y**2 - 2*y
    # 断言 modgcd_bivariate(f, g) 的结果与给定的元组相等
    assert modgcd_bivariate(f, g) == (x + y, 2*x - 3, 4*y - 2)
def test_chinese_remainder():
    R, x, y = ring("x, y", ZZ)  # 创建多项式环 R，定义变量 x 和 y，使用整数环 ZZ

    p, q = 3, 5  # 设置两个整数 p 和 q

    hp = x**3*y - x**2 - 1  # 定义多项式 hp
    hq = -x**3*y - 2*x*y**2 + 2  # 定义多项式 hq

    hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)  # 调用多元中的中国剩余定理重构函数

    assert hpq.trunc_ground(p) == hp  # 断言 hpq 在模 p 下的截断结果等于 hp
    assert hpq.trunc_ground(q) == hq  # 断言 hpq 在模 q 下的截断结果等于 hq

    T, z = ring("z", R)  # 创建多项式环 T，定义变量 z，基于环 R

    p, q = 3, 7  # 设置两个整数 p 和 q

    hp = (x*y + 1)*z**2 + x  # 定义多项式 hp
    hq = (x**2 - 3*y)*z + 2  # 定义多项式 hq

    hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)  # 调用多元中的中国剩余定理重构函数

    assert hpq.trunc_ground(p) == hp  # 断言 hpq 在模 p 下的截断结果等于 hp
    assert hpq.trunc_ground(q) == hq  # 断言 hpq 在模 q 下的截断结果等于 hq


def test_modgcd_multivariate_integers():
    R, x, y = ring("x,y", ZZ)  # 创建多项式环 R，定义变量 x 和 y，使用整数环 ZZ

    f, g = R.zero, R.zero  # 初始化多项式 f 和 g 为零多项式
    assert modgcd_multivariate(f, g) == (0, 0, 0)  # 断言多项式 f 和 g 的多元整数最大公因数是 (0, 0, 0)

    f, g = 2*x**2 + 4*x + 2, x + 1  # 设置两个多项式 f 和 g
    assert modgcd_multivariate(f, g) == (x + 1, 2*x + 2, 1)  # 断言多项式 f 和 g 的多元整数最大公因数是 (x + 1, 2*x + 2, 1)

    f, g = x + 1, 2*x**2 + 4*x + 2  # 设置两个多项式 f 和 g
    assert modgcd_multivariate(f, g) == (x + 1, 1, 2*x + 2)  # 断言多项式 f 和 g 的多元整数最大公因数是 (x + 1, 1, 2*x + 2)

    f = 2*x**2 + 2*x*y - 3*x - 3*y  # 设置多项式 f
    g = 4*x*y - 2*x + 4*y**2 - 2*y  # 设置多项式 g
    assert modgcd_multivariate(f, g) == (x + y, 2*x - 3, 4*y - 2)  # 断言多项式 f 和 g 的多元整数最大公因数是 (x + y, 2*x - 3, 4*y - 2)

    f, g = x*y**2 + 2*x*y + x, x*y**3 + x  # 设置两个多项式 f 和 g
    assert modgcd_multivariate(f, g) == (x*y + x, y + 1, y**2 - y + 1)  # 断言多项式 f 和 g 的多元整数最大公因数是 (x*y + x, y + 1, y**2 - y + 1)

    f, g = x**2*y**2 + x**2*y + 1, x*y**2 + x*y + 1  # 设置两个多项式 f 和 g
    assert modgcd_multivariate(f, g) == (1, f, g)  # 断言多项式 f 和 g 的多元整数最大公因数是 (1, f, g)

    f = x**4 + 8*x**3 + 21*x**2 + 22*x + 8  # 设置多项式 f
    g = x**3 + 6*x**2 + 11*x + 6  # 设置多项式 g
    h = x**2 + 3*x + 2  # 设置多项式 h

    cff = x**2 + 5*x + 4  # 设置多项式 cff
    cfg = x + 3  # 设置多项式 cfg

    assert modgcd_multivariate(f, g) == (h, cff, cfg)  # 断言多项式 f 和 g 的多元整数最大公因数是 (h, cff, cfg)

    R, x, y, z, u = ring("x,y,z,u", ZZ)  # 创建多项式环 R，定义变量 x, y, z, u，使用整数环 ZZ

    f, g = x + y + z, -x - y - z - u  # 设置两个多项式 f 和 g
    assert modgcd_multivariate(f, g) == (1, f, g)  # 断言多项式 f 和 g 的多元整数最大公因数是 (1, f, g)

    f, g = u**2 + 2*u + 1, 2*u + 2  # 设置两个多项式 f 和 g
    assert modgcd_multivariate(f, g) == (u + 1, u + 1, 2)  # 断言多项式 f 和 g 的多元整数最大公因数是 (u + 1, u + 1, 2)

    f, g = z**2*u**2 + 2*z**2*u + z**2 + z*u + z, u**2 + 2*u + 1  # 设置两个多项式 f 和 g
    h, cff, cfg = u + 1, z**2*u + z**2 + z, u + 1  # 定义多项式 h, cff, cfg

    assert modgcd_multivariate(f, g) == (h, cff, cfg)  # 断言多项式 f 和 g 的多元整数最大公因数是 (h, cff, cfg)
    assert modgcd_multivariate(g, f) == (h, cfg, cff)  # 断言多项式 g 和 f 的多元整数最大公因数是 (h, cfg, cff)

    R, x, y, z = ring("x,y,z", ZZ)  # 创建多项式环 R，定义变量 x, y, z，使用整数环 ZZ

    f, g = x - y*z, x - y*z  # 设置两个多项式 f 和 g
    assert modgcd_multivariate(f, g) == (x - y*z, 1, 1)  # 断言多项式 f 和 g 的多元整数最大公因数是 (x - y*z, 1, 1)

    f, g, h = R.fateman_poly_F_1()  # 调用 R 的 fateman_poly_F_1 方法，返回多项式 f, g, h
    H, cff, cfg = modgcd_multivariate(f, g)  # 计算多元整数最大公因数

    assert H == h and H*cff == f and H*cfg == g  # 断言 H, cff, cfg 与预期的多项式相等

    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)  # 创建多项式环 R，定义变量 x, y, z, u, v，使用整数环 ZZ

    f, g, h = R.fateman_poly_F_1()  # 调用 R 的 fateman_poly_F_1 方法，返回多项式 f, g, h
    H, cff, cfg = modgcd_multivariate(f, g)  # 计算多元整数最大公因数

    assert H == h and H*cff == f and H*cfg == g  # 断言 H
    # 创建多项式环 R，包含变量 x, y, z, t，使用整数环 ZZ
    R, x, y, z, t = ring("x,y,z,t", ZZ)

    # 调用 fateman_poly_F_3() 函数获取三个多项式 f, g, h
    f, g, h = R.fateman_poly_F_3()

    # 调用 modgcd_multivariate 函数计算多项式 f 和 g 的多元模 GCD
    # 返回结果包括 GCD H 和其在 f, g 中的系数 cff, cfg
    H, cff, cfg = modgcd_multivariate(f, g)

    # 使用断言确保计算正确：H 应与 h 相等，同时满足 H*cff == f 和 H*cfg == g
    assert H == h and H*cff == f and H*cfg == g
# 定义函数用于测试转换为有理整数多项式
def test_to_ZZ_ANP_poly():
    # 在有理数域QQ上定义带平方根扩展的代数域A
    A = AlgebraicField(QQ, sqrt(2))
    # 在代数环R中定义单变量多项式环R和变量x
    R, x = ring("x", A)
    # 创建多项式f = x * (sqrt(2) + 1)
    f = x * (sqrt(2) + 1)

    # 在整数环ZZ上定义多变量环T，并声明变量x_和z_
    T, x_, z_ = ring("x_, z_", ZZ)
    # 创建多项式f_ = x_ * z_ + x_
    f_ = x_ * z_ + x_

    # 断言：将f转换为T环中的整数多项式应该得到f_
    assert _to_ZZ_poly(f, T) == f_
    # 断言：将f_转换为R环中的有理数多项式应该得到f
    assert _to_ANP_poly(f_, R) == f

    # 在代数环R中重新定义多变量环R和变量x, t, s
    R, x, t, s = ring("x, t, s", A)
    # 创建多项式f = x * t**2 + x * s + sqrt(2)
    f = x * t**2 + x * s + sqrt(2)

    # 在整数环D上定义多变量环D，并声明变量t_和s_
    D, t_, s_ = ring("t_, s_", ZZ)
    # 在整数环D上定义多变量环T，并声明变量x_和z_
    T, x_, z_ = ring("x_, z_", D)
    # 创建多项式f_ = (t_**2 + s_) * x_ + z_
    f_ = (t_**2 + s_) * x_ + z_

    # 断言：将f转换为T环中的整数多项式应该得到f_
    assert _to_ZZ_poly(f, T) == f_
    # 断言：将f_转换为R环中的有理数多项式应该得到f
    assert _to_ANP_poly(f_, R) == f


# 定义函数用于测试在代数扩展域上的模GCD算法
def test_modgcd_algebraic_field():
    # 在有理数域QQ上定义带平方根扩展的代数域A
    A = AlgebraicField(QQ, sqrt(2))
    # 在代数环R中定义单变量多项式环R和变量x
    R, x = ring("x", A)
    # 获取A的单位元
    one = A.one

    # 设定多项式f和g，分别为2*x和R(2)
    f, g = 2 * x, R(2)
    # 断言：在函数域上对f和g执行模GCD应得到(one, f, g)
    assert func_field_modgcd(f, g) == (one, f, g)

    # 设定多项式f和g，分别为2*x和R(sqrt(2))
    f, g = 2 * x, R(sqrt(2))
    # 断言：在函数域上对f和g执行模GCD应得到(one, f, g)
    assert func_field_modgcd(f, g) == (one, f, g)

    # 设定多项式f和g，分别为2*x + 2和6*x**2 - 6
    f, g = 2 * x + 2, 6 * x**2 - 6
    # 断言：在函数域上对f和g执行模GCD应得到(x + 1, R(2), 6*x - 6)
    assert func_field_modgcd(f, g) == (x + 1, R(2), 6 * x - 6)

    # 在代数环R中重新定义多变量环R和变量x和y
    R, x, y = ring("x, y", A)

    # 设定多项式f和g，分别为x + sqrt(2)*y和x + y
    f, g = x + sqrt(2) * y, x + y
    # 断言：在函数域上对f和g执行模GCD应得到(one, f, g)
    assert func_field_modgcd(f, g) == (one, f, g)

    # 设定多项式f和g，分别为x*y + sqrt(2)*y**2和R(sqrt(2))*y
    f, g = x * y + sqrt(2) * y**2, R(sqrt(2)) * y
    # 断言：在函数域上对f和g执行模GCD应得到(y, x + sqrt(2)*y, R(sqrt(2)))
    assert func_field_modgcd(f, g) == (y, x + sqrt(2) * y, R(sqrt(2)))

    # 设定多项式f和g，分别为x**2 + 2*sqrt(2)*x*y + 2*y**2和x + sqrt(2)*y
    f, g = x**2 + 2 * sqrt(2) * x * y + 2 * y**2, x + sqrt(2) * y
    # 断言：在函数域上对f和g执行模GCD应得到(g, g, one)
    assert func_field_modgcd(f, g) == (g, g, one)

    # 在有理数域QQ上重新定义带sqrt(2)和sqrt(3)扩展的代数域A
    A = AlgebraicField(QQ, sqrt(2), sqrt(3))
    # 在代数环R中定义多变量环R和变量x, y, z
    R, x, y, z = ring("x, y, z", A)

    # 设定多项式h为x**2*y**7 + sqrt(6)/21*z
    h = x**2 * y**7 + sqrt(6) / 21 * z
    # 设定多项式f和g，分别为h*(27*y**3 + 1)和h*(y + x)
    f, g = h * (27 * y**3 + 1), h * (y + x)
    # 断言：在函数域上对f和g执行模GCD应得到(h, 27*y**3+1, y+x)
    assert func_field_modgcd(f, g) == (h, 27 * y**3 + 1, y + x)

    # 设定多项式h为x**13*y**3 + 1/2*x**10 + 1/sqrt(2)
    h = x**13 * y**3 + 1 / 2 * x**10 + 1 / sqrt(2)
    # 设定多项式f和g，分别为h*(x + 1)和h*sqrt(2)/sqrt(3)
    f, g = h * (x + 1), h * sqrt(2) / sqrt(3)
    # 断言：在函数域上对f和g执行模GCD应得到(h, x + 1, R(sqrt(2)/sqrt(3)))
    assert func_field_modgcd(f, g) == (h, x + 1, R(sqrt(2) / sqrt(3)))

    # 在有理数域QQ上重新定义带sqrt(2)^(-1)*sqrt(3)扩展的代数域A
    A = AlgebraicField(QQ, sqrt(2)**(-1) * sqrt(3))
    # 在代数环R中定义单变量多项式环R和变量x
    R, x = ring("x", A)

    # 设定多项式f和g，分别为x + 1和x - 1
    f, g = x + 1, x - 1
    # 断言：在函数域上对f和g执行模GCD应得到(A.one, f, g)
    assert func_field_modgcd(f, g) == (A.one, f, g)


# 当func_field_modgcd支持函数域时，可以改变此测试
def test_modgcd_func_field():
    # 在整数环ZZ上定义单变量多项式环D和变量t
    D, t = ring("t", ZZ)
    # 在多变量环R中定义多变量环R和变量x和z
    R, x, z = ring
```