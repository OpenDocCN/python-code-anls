# `D:\src\scipysrc\sympy\sympy\polys\tests\test_euclidtools.py`

```
"""Tests for Euclidean algorithms, GCDs, LCMs and polynomial remainder sequences. """

# 导入所需的库和模块
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, RR

from sympy.polys.specialpolys import (
    f_polys,
    dmp_fateman_poly_F_1,
    dmp_fateman_poly_F_2,
    dmp_fateman_poly_F_3)

# 从特殊多项式模块导入多项式列表
f_0, f_1, f_2, f_3, f_4, f_5, f_6 = f_polys()

# 定义一个测试函数，测试双重扩展欧几里得算法
def test_dup_gcdex():
    # 创建有理数域上的多项式环R和变量x
    R, x = ring("x", QQ)

    # 定义两个多项式f和g
    f = x**4 - 2*x**3 - 6*x**2 + 12*x + 15
    g = x**3 + x**2 - 4*x - 4

    # 计算期望的商和最大公因式
    s = -QQ(1,5)*x + QQ(3,5)
    t = QQ(1,5)*x**2 - QQ(6,5)*x + 2
    h = x + 1

    # 断言双重扩展欧几里得算法的结果
    assert R.dup_half_gcdex(f, g) == (s, h)
    assert R.dup_gcdex(f, g) == (s, t, h)

    # 重新定义f和g，并再次测试
    f = x**4 + 4*x**3 - x + 1
    g = x**3 - x + 1

    # 计算期望的s、t和h
    s, t, h = R.dup_gcdex(f, g)
    S, T, H = R.dup_gcdex(g, f)

    # 断言加法和乘法的正确性
    assert R.dup_add(R.dup_mul(s, f),
                     R.dup_mul(t, g)) == h
    assert R.dup_add(R.dup_mul(S, g),
                     R.dup_mul(T, f)) == H

    # 重新定义f和g，并再次测试
    f = 2*x
    g = x**2 - 16

    # 计算期望的s、t和h
    s = QQ(1,32)*x
    t = -QQ(1,16)
    h = 1

    # 断言半扩展欧几里得算法和扩展欧几里得算法的结果
    assert R.dup_half_gcdex(f, g) == (s, h)
    assert R.dup_gcdex(f, g) == (s, t, h)


# 定义测试函数，测试多项式的逆
def test_dup_invert():
    # 创建有理数域上的多项式环R和变量x
    R, x = ring("x", QQ)
    assert R.dup_invert(2*x, x**2 - 16) == QQ(1,32)*x


# 定义测试函数，测试多项式的欧几里得算法
def test_dup_euclidean_prs():
    # 创建有理数域上的多项式环R和变量x
    R, x = ring("x", QQ)

    # 定义两个多项式f和g
    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    # 断言多项式的欧几里得算法结果
    assert R.dup_euclidean_prs(f, g) == [
        f,
        g,
        -QQ(5,9)*x**4 + QQ(1,9)*x**2 - QQ(1,3),
        -QQ(117,25)*x**2 - 9*x + QQ(441,25),
        QQ(233150,19773)*x - QQ(102500,6591),
        -QQ(1288744821,543589225)]


# 定义测试函数，测试多项式的原始部分剩余算法
def test_dup_primitive_prs():
    # 创建整数域上的多项式环R和变量x
    R, x = ring("x", ZZ)

    # 定义两个多项式f和g
    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    # 断言多项式的原始部分剩余算法结果
    assert R.dup_primitive_prs(f, g) == [
        f,
        g,
        -5*x**4 + x**2 - 3,
        13*x**2 + 25*x - 49,
        4663*x - 6150,
        1]


# 定义测试函数，测试多项式的子结果算法
def test_dup_subresultants():
    # 创建整数域上的多项式环R和变量x
    R, x = ring("x", ZZ)

    # 断言多项式的子结果算法结果
    assert R.dup_resultant(0, 0) == 0

    assert R.dup_resultant(1, 0) == 0
    assert R.dup_resultant(0, 1) == 0

    # 定义两个多项式f和g
    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    # 定义期望的结果a、b、c、d
    a = 15*x**4 - 3*x**2 + 9
    b = 65*x**2 + 125*x - 245
    c = 9326*x - 12300
    d = 260708

    # 断言多项式的子结果算法结果
    assert R.dup_subresultants(f, g) == [f, g, a, b, c, d]
    assert R.dup_resultant(f, g) == R.dup_LC(d)

    # 重新定义f和g，并再次测试
    f = x**2 - 2*x + 1
    g = x**2 - 1

    # 定义期望的结果a
    a = 2*x - 2

    # 断言多项式的子结果算法结果
    assert R.dup_subresultants(f, g) == [f, g, a]
    assert R.dup_resultant(f, g) == 0

    # 重新定义f和g，并再次测试
    f = x**2 + 1
    g = x**2 - 1

    # 定义期望的结果a
    a = -2

    # 断言多项式的子结果算法结果
    assert R.dup_subresultants(f, g) == [f, g, a]
    assert R.dup_resultant(f, g) == 4

    # 重新定义f和g，并再次测试
    f = x**2 - 1
    g = x**3 - x**2 + 2

    # 断言多项式的子结果算法结果
    assert R.dup_resultant(f, g) == 0

    # 重新定义f和g，并再次测试
    f = 3*x**3 - x
    g = 5*x**2 + 1

    # 断言多项式的子结果算法结果
    assert R.dup_resultant(f, g) == 64

    # 重新定义f和g，并再次测试
    f = x**2 - 2*x + 7
    g = x**3 - x + 5

    # 断言多项式的子结果算法结果
    assert R.dup_resultant(f, g) == 265

    # 重新定义f和g，并再次测试
    f = x**3 - 6*x**2 + 11*x - 6
    g = x**3 - 15*x**2 + 74*x - 120
    # 断言：验证 R.dup_resultant(f, g) 的返回值是否等于 -8640
    assert R.dup_resultant(f, g) == -8640
    
    # 定义多项式 f 和 g
    f = x**3 - 6*x**2 + 11*x - 6
    g = x**3 - 10*x**2 + 29*x - 20
    
    # 断言：验证 R.dup_resultant(f, g) 的返回值是否等于 0
    assert R.dup_resultant(f, g) == 0
    
    # 重新定义多项式 f 和 g
    f = x**3 - 1
    g = x**3 + 2*x**2 + 2*x - 1
    
    # 断言：验证 R.dup_resultant(f, g) 的返回值是否等于 16
    assert R.dup_resultant(f, g) == 16
    
    # 重新定义多项式 f 和 g
    f = x**8 - 2
    g = x - 1
    
    # 断言：验证 R.dup_resultant(f, g) 的返回值是否等于 -1
    assert R.dup_resultant(f, g) == -1
# 定义测试函数 test_dmp_subresultants，用于测试多项式环 R 的子结果式相关功能
def test_dmp_subresultants():
    # 创建多项式环 R，包含变量 x 和 y，系数环为整数环 ZZ
    R, x, y = ring("x,y", ZZ)

    # 断言零多项式的结果式为零
    assert R.dmp_resultant(0, 0) == 0
    # 断言零多项式的平常系数结果式第一个元素为零
    assert R.dmp_prs_resultant(0, 0)[0] == 0
    # 断言零多项式的整数系数结果式为零
    assert R.dmp_zz_collins_resultant(0, 0) == 0
    # 断言零多项式的有理数系数结果式为零
    assert R.dmp_qq_collins_resultant(0, 0) == 0

    # 断言 f = 3*x**2*y - y**3 - 4 的结果式为零
    assert R.dmp_resultant(1, 0) == 0
    assert R.dmp_resultant(1, 0) == 0
    assert R.dmp_resultant(1, 0) == 0

    # 断言 f = -x**3 + 5 和 g = 3*x**2*y + x**2 的子结果式为 [f, g, a]
    f = 3*x**2*y - y**3 - 4
    g = x**2 + x*y**3 - 9
    a = 3*x*y**4 + y**3 - 27*y + 4
    b = -3*y**10 - 12*y**7 + y**6 - 54*y**4 + 8*y**3 + 729*y**2 - 216*y + 16
    r = R.dmp_LC(b)
    assert R.dmp_subresultants(f, g) == [f, g, a]

    # 断言 f 和 g 的结果式为 r
    assert R.dmp_resultant(f, g) == r
    assert R.dmp_prs_resultant(f, g)[0] == r
    assert R.dmp_zz_collins_resultant(f, g) == r
    assert R.dmp_qq_collins_resultant(f, g) == r

    # 更改环 R，包含更多变量：z, u, v
    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)

    # 定义多项式 f 和 g
    f = 6*x**2 - 3*x*y - 2*x*z + y*z
    g = x**2 - x*u - x*v + u*v

    # 定义预期的结果式 r
    r = y**2*z**2 - 3*y**2*z*u - 3*y**2*z*v + 9*y**2*u*v - 2*y*z**2*u \
      - 2*y*z**2*v + 6*y*z*u**2 + 12*y*z*u*v + 6*y*z*v**2 - 18*y*u**2*v \
      - 18*y*u*v**2 + 4*z**2*u*v - 12*z*u**2*v - 12*z*u*v**2 + 36*u**2*v**2

    # 断言 f 和 g 的 ZZ-Collins 结果式为 r 的 x 部分
    assert R.dmp_zz_collins_resultant(f, g) == r.drop(x)

    # 更改环 R，系数环为有理数环 QQ
    R, x, y, z, u, v = ring("x,y,z,u,v", QQ)

    # 定义多项式 f 和 g，系数为有理数
    f = x**2 - QQ(1,2)*x*y - QQ(1,3)*x*z + QQ(1,6)*y*z
    g = x**2 - x*u - x*v + u*v

    # 定义预期的结果式 r
    r = QQ(1,36)*y**2*z**2 - QQ(1,12)*y**2*z*u - QQ(1,12)*y**2*z*v + QQ(1,4)*y**2*u*v \
      - QQ(1,18)*y*z**2*u - QQ(1,18)*y*z**2*v + QQ(1,6)*y*z*u**2 + QQ(1,3)*y*z*u*v \
      + QQ(1,6)*y*z*v**2 - QQ(1,2)*y*u**2*v - QQ(1,2)*y*u*v**2 + QQ(1,9)*z**2*u*v \
      - QQ(1,3)*z*u**2*v - QQ(1,3)*z*u*v**2 + u**2*v**2

    # 断言 f 和 g 的 QQ-Collins 结果式为 r 的 x 部分
    assert R.dmp_qq_collins_resultant(f, g) == r.drop(x)

    # 创建环 Rt 和 t，系数环为整数环 ZZ
    Rt, t = ring("t", ZZ)
    # 在 Rt 上创建环 Rx 和 x
    Rx, x = ring("x", Rt)

    # 定义多项式 f 和 g，系数为 t 和 x
    f = x**6 - 5*x**4 + 5*x**2 + 4
    g = -6*t*x**5 + x**4 + 20*t*x**3 - 3*x**2 - 10*t*x + 6

    # 断言 f 和 g 的副结果式为 2930944*t**6 + 2198208*t**4 + 549552*t**2 + 45796
    assert Rx.dup_resultant(f, g) == 2930944*t**6 + 2198208*t**4 + 549552*t**2 + 45796


# 定义测试函数 test_dup_discriminant，用于测试单变量多项式环 R 的判别式功能
def test_dup_discriminant():
    # 创建多项式环 R，包含变量 x，系数环为整数环 ZZ
    R, x = ring("x", ZZ)

    # 断言零多项式的判别式为零
    assert R.dup_discriminant(0) == 0
    # 断言 x 的判别式为 1
    assert R.dup_discriminant(x) == 1

    # 断言给定多项式的判别式的预期值
    assert R.dup_discriminant(x**3 + 3*x**2 + 9*x - 13) == -11664
    assert R.dup_discriminant(5*x**5 + x**3 + 2) == 31252160
    assert R.dup_discriminant(x**4 + 2*x**3 + 6*x**2 - 22*x + 13) == 0
    assert R.dup_discriminant(12*x**7 + 15*x**4 + 30*x**3 + x**2 + 1) == -220289699947514112


# 定义测试函数 test_dmp_discriminant，用于测试多变量多项式环 R 的判别式功能
def test_dmp_discriminant():
    # 创建多项式环 R，包含变量 x，系数环为整数环 ZZ
    R, x = ring("x", ZZ)

    # 断言零多项式的判别式为零
    assert R.dmp_discriminant(0) == 0
    # 定义一个多项式环 R，变量为 x, y，并使用整数环 ZZ
    R, x, y = ring("x,y", ZZ)

    # 断言：计算多项式环 R 中的二次型判别式，应为 0
    assert R.dmp_discriminant(0) == 0
    # 断言：计算多项式环 R 中 y 的二次型判别式，应为 0
    assert R.dmp_discriminant(y) == 0

    # 断言：计算多项式环 R 中 x^3 + 3*x^2 + 9*x - 13 的二次型判别式，应为 -11664
    assert R.dmp_discriminant(x**3 + 3*x**2 + 9*x - 13) == -11664
    # 断言：计算多项式环 R 中 5*x^5 + x^3 + 2 的二次型判别式，应为 31252160
    assert R.dmp_discriminant(5*x**5 + x**3 + 2) == 31252160
    # 断言：计算多项式环 R 中 x^4 + 2*x^3 + 6*x^2 - 22*x + 13 的二次型判别式，应为 0
    assert R.dmp_discriminant(x**4 + 2*x**3 + 6*x**2 - 22*x + 13) == 0
    # 断言：计算多项式环 R 中 12*x^7 + 15*x^4 + 30*x^3 + x^2 + 1 的二次型判别式，应为 -220289699947514112
    assert R.dmp_discriminant(12*x**7 + 15*x**4 + 30*x**3 + x**2 + 1) == -220289699947514112

    # 断言：计算多项式环 R 中 x^2*y + 2*y 的二次型判别式，应为 (-8*y^2).drop(x)
    assert R.dmp_discriminant(x**2*y + 2*y) == (-8*y**2).drop(x)
    # 断言：计算多项式环 R 中 x*y^2 + 2*x 的二次型判别式，应为 1
    assert R.dmp_discriminant(x*y**2 + 2*x) == 1

    # 重新定义多项式环 R，变量为 x, y, z，并使用整数环 ZZ
    R, x, y, z = ring("x,y,z", ZZ)
    # 断言：计算多项式环 R 中 x*y + z 的二次型判别式，应为 1
    assert R.dmp_discriminant(x*y + z) == 1

    # 重新定义多项式环 R，变量为 x, y, z, u，并使用整数环 ZZ
    R, x, y, z, u = ring("x,y,z,u", ZZ)
    # 断言：计算多项式环 R 中 x^2*y + x*z + u 的二次型判别式，应为 (-4*y*u + z^2).drop(x)
    assert R.dmp_discriminant(x**2*y + x*z + u) == (-4*y*u + z**2).drop(x)

    # 重新定义多项式环 R，变量为 x, y, z, u, v，并使用整数环 ZZ
    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)
    # 断言：计算多项式环 R 中 x^3*y + x^2*z + x*u + v 的二次型判别式，应为 (-27*y^2*v^2 + 18*y*z*u*v - 4*y*u^3 - 4*z^3*v + z^2*u^2).drop(x)
    assert R.dmp_discriminant(x**3*y + x**2*z + x*u + v) == \
        (-27*y**2*v**2 + 18*y*z*u*v - 4*y*u**3 - 4*z**3*v + z**2*u**2).drop(x)
# 定义一个测试函数，用于测试整数多项式环上不同最大公因数算法的结果
def test_dup_gcd():
    # 创建整数多项式环 R，变量为 x
    R, x = ring("x", ZZ)

    # 测试情况1：f 和 g 均为0
    f, g = 0, 0
    # 断言两种最大公因数算法的结果均为 (0, 0, 0)
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (0, 0, 0)

    # 测试情况2：f 为2，g 为0
    f, g = 2, 0
    # 断言两种最大公因数算法的结果均为 (2, 1, 0)
    assert R.dup_zz_heu_gcd(f, g) == R.dup_rr_prs_gcd(f, g) == (2, 1, 0)

    # 依此类推，对每种情况进行测试，并注释断言的含义和结果
    ...

    # 最后的测试环境切换到有理数域 QQ
    R, x = ring("x", QQ)
    # 定义多项式 f，这是一个长格式的多项式表达式
    f = - 352518131239247345597970242177235495263669787845475025293906825864749649589178600387510272*x**49 \
        + 46818041807522713962450042363465092040687472354933295397472942006618953623327997952*x**42 \
        + 378182690892293941192071663536490788434899030680411695933646320291525827756032*x**35 \
        + 112806468807371824947796775491032386836656074179286744191026149539708928*x**28 \
        - 12278371209708240950316872681744825481125965781519138077173235712*x**21 \
        + 289127344604779611146960547954288113529690984687482920704*x**14 \
        + 19007977035740498977629742919480623972236450681*x**7 \
        + 311973482284542371301330321821976049

    # 定义多项式 g，这也是一个长格式的多项式表达式
    g =   365431878023781158602430064717380211405897160759702125019136*x**21 \
        + 197599133478719444145775798221171663643171734081650688*x**14 \
        - 9504116979659010018253915765478924103928886144*x**7 \
        - 311973482284542371301330321821976049

    # 使用 R.dup_zz_heu_gcd 函数计算 f 和 f 的一阶导数的最大公约数，并断言结果应等于 g
    assert R.dup_zz_heu_gcd(f, R.dup_diff(f, 1))[0] == g

    # 使用 R.dup_rr_prs_gcd 函数计算 f 和 f 的一阶导数的 PRS（平行剩余序列）的最大公约数，并断言结果应等于 g
    assert R.dup_rr_prs_gcd(f, R.dup_diff(f, 1))[0] == g

    # 重新定义环 R 和变量 x，这次使用有理数域 QQ
    R, x = ring("x", QQ)

    # 定义有理数域 QQ 上的多项式 f
    f = QQ(1,2)*x**2 + x + QQ(1,2)

    # 定义有理数域 QQ 上的多项式 g
    g = QQ(1,2)*x + QQ(1,2)

    # 定义多项式 h = x + 1
    h = x + 1

    # 使用 R.dup_qq_heu_gcd 函数计算 f 和 g 的最大公约数，期望结果是 (h, g, QQ(1,2))
    assert R.dup_qq_heu_gcd(f, g) == (h, g, QQ(1,2))

    # 使用 R.dup_ff_prs_gcd 函数计算 f 和 g 的 PRS 的最大公约数，期望结果是 (h, g, QQ(1,2))
    assert R.dup_ff_prs_gcd(f, g) == (h, g, QQ(1,2))

    # 重新定义环 R 和变量 x，这次使用整数域 ZZ
    R, x = ring("x", ZZ)

    # 定义整数域 ZZ 上的多项式 f
    f = 1317378933230047068160*x + 2945748836994210856960

    # 定义整数域 ZZ 上的多项式 g
    g = 120352542776360960*x + 269116466014453760

    # 定义多项式 h = 120352542776360960*x + 269116466014453760
    h = 120352542776360960*x + 269116466014453760

    # 定义常数 cff = 10946 和 cfg = 1
    cff = 10946
    cfg = 1

    # 使用 R.dup_zz_heu_gcd 函数计算 f 和 g 的最大公约数，期望结果是 (h, cff, cfg)
    assert R.dup_zz_heu_gcd(f, g) == (h, cff, cfg)
# 定义一个测试函数 test_dmp_gcd
def test_dmp_gcd():
    # 创建多项式环 R，包含变量 x 和 y
    R, x, y = ring("x,y", ZZ)

    # 测试 f 和 g 均为 0 的情况
    f, g = 0, 0
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (0, 0, 0)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (0, 0, 0)

    # 测试 f = 2, g = 0 的情况
    f, g = 2, 0
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (2, 1, 0)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, 1, 0)

    # 测试 f = -2, g = 0 的情况
    f, g = -2, 0
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (2, -1, 0)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, -1, 0)

    # 测试 f = 0, g = -2 的情况
    f, g = 0, -2
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (2, 0, -1)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, 0, -1)

    # 测试 f = 0, g = 2*x + 4 的情况
    f, g = 0, 2*x + 4
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (2*x + 4, 0, 1)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2*x + 4, 0, 1)

    # 测试 f = 2*x + 4, g = 0 的情况
    f, g = 2*x + 4, 0
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (2*x + 4, 1, 0)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2*x + 4, 1, 0)

    # 测试 f = 2, g = 2 的情况
    f, g = 2, 2
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (2, 1, 1)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, 1, 1)

    # 测试 f = -2, g = 2 的情况
    f, g = -2, 2
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (2, -1, 1)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, -1, 1)

    # 测试 f = 2, g = -2 的情况
    f, g = 2, -2
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (2, 1, -1)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, 1, -1)

    # 测试 f = -2, g = -2 的情况
    f, g = -2, -2
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (2, -1, -1)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, -1, -1)

    # 测试 f = x**2 + 2*x + 1, g = 1 的情况
    f, g = x**2 + 2*x + 1, 1
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (1, x**2 + 2*x + 1, 1)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (1, x**2 + 2*x + 1, 1)

    # 测试 f = x**2 + 2*x + 1, g = 2 的情况
    f, g = x**2 + 2*x + 1, 2
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (1, x**2 + 2*x + 1, 2)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (1, x**2 + 2*x + 1, 2)

    # 测试 f = 2*x**2 + 4*x + 2, g = 2 的情况
    f, g = 2*x**2 + 4*x + 2, 2
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (2, x**2 + 2*x + 1, 1)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, x**2 + 2*x + 1, 1)

    # 测试 f = 2, g = 2*x**2 + 4*x + 2 的情况
    f, g = 2, 2*x**2 + 4*x + 2
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (2, 1, x**2 + 2*x + 1)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (2, 1, x**2 + 2*x + 1)

    # 测试 f = 2*x**2 + 4*x + 2, g = x + 1 的情况
    f, g = 2*x**2 + 4*x + 2, x + 1
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (x + 1, 2*x + 2, 1)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (x + 1, 2*x + 2, 1)

    # 测试 f = x + 1, g = 2*x**2 + 4*x + 2 的情况
    f, g = x + 1, 2*x**2 + 4*x + 2
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (x + 1, 1, 2*x + 2)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (x + 1, 1, 2*x + 2)

    # 创建多项式环 R，包含变量 x, y, z, u
    R, x, y, z, u = ring("x,y,z,u", ZZ)

    # 测试 f = u**2 + 2*u + 1, g = 2*u + 2 的情况
    f, g = u**2 + 2*u + 1, 2*u + 2
    # 断言调用 heu_gcd 和 prs_gcd 方法返回的结果均为 (u + 1, u + 1, 2)
    assert R.dmp_zz_heu_gcd(f, g) == R.dmp_rr_prs_gcd(f, g) == (u + 1
    # 断言：验证多项式 H 等于 h，以及 R.dmp_mul(H, cff) 等于 f 和 R.dmp_mul(H, cfg) 等于 g
    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    # 创建多项式环 R，定义变量 x, y, z, u, v, a, b, c, d，使用整数环 ZZ
    R, x, y, z, u, v, a, b, c, d = ring("x,y,z,u,v,a,b,c,d", ZZ)

    # 生成多项式 f, g, h，将整数列表转换为 R 类型，并计算它们的最大公因式 H, cff, cfg
    f, g, h = map(R.from_dense, dmp_fateman_poly_F_1(8, ZZ))
    H, cff, cfg = R.dmp_zz_heu_gcd(f, g)

    # 断言：验证多项式 H 等于 h，以及 R.dmp_mul(H, cff) 等于 f 和 R.dmp_mul(H, cfg) 等于 g
    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    # 创建多项式环 R，定义变量 x, y, z，使用整数环 ZZ
    R, x, y, z = ring("x,y,z", ZZ)

    # 生成多项式 f, g, h，将整数列表转换为 R 类型，并计算它们的最大公因式 H, cff, cfg
    f, g, h = map(R.from_dense, dmp_fateman_poly_F_2(2, ZZ))
    H, cff, cfg = R.dmp_zz_heu_gcd(f, g)

    # 断言：验证多项式 H 等于 h，以及 R.dmp_mul(H, cff) 等于 f 和 R.dmp_mul(H, cfg) 等于 g
    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    # 计算多项式 f 和 g 的最大公因式 H, cff, cfg
    H, cff, cfg = R.dmp_rr_prs_gcd(f, g)

    # 断言：验证多项式 H 等于 h，以及 R.dmp_mul(H, cff) 等于 f 和 R.dmp_mul(H, cfg) 等于 g
    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    # 生成多项式 f, g, h，将整数列表转换为 R 类型，并计算它们的最大公因式 H, cff, cfg
    f, g, h = map(R.from_dense, dmp_fateman_poly_F_3(2, ZZ))
    H, cff, cfg = R.dmp_zz_heu_gcd(f, g)

    # 断言：验证多项式 H 等于 h，以及 R.dmp_mul(H, cff) 等于 f 和 R.dmp_mul(H, cfg) 等于 g
    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    # 计算多项式 f 和 g 的最大公因式 H, cff, cfg
    H, cff, cfg = R.dmp_rr_prs_gcd(f, g)

    # 断言：验证多项式 H 等于 h，以及 R.dmp_mul(H, cff) 等于 f 和 R.dmp_mul(H, cfg) 等于 g
    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    # 创建多项式环 R，定义变量 x, y, z, u, v，使用整数环 ZZ
    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)

    # 生成多项式 f, g, h，将整数列表转换为 R 类型，并计算它们的最大公因式 H, cff, cfg
    f, g, h = map(R.from_dense, dmp_fateman_poly_F_3(4, ZZ))
    H, cff, cfg = R.dmp_inner_gcd(f, g)

    # 断言：验证多项式 H 等于 h，以及 R.dmp_mul(H, cff) 等于 f 和 R.dmp_mul(H, cfg) 等于 g
    assert H == h and R.dmp_mul(H, cff) == f \
                  and R.dmp_mul(H, cfg) == g

    # 创建多项式环 R，定义变量 x, y，使用有理数环 QQ
    R, x, y = ring("x,y", QQ)

    # 定义有理数多项式 f 和 g
    f = QQ(1,2)*x**2 + x + QQ(1,2)
    g = QQ(1,2)*x + QQ(1,2)

    # 定义预期的最大公因式 h，使用 R.dmp_qq_heu_gcd 函数验证结果
    h = x + 1
    assert R.dmp_qq_heu_gcd(f, g) == (h, g, QQ(1,2))
    # 使用 R.dmp_ff_prs_gcd 函数验证结果
    assert R.dmp_ff_prs_gcd(f, g) == (h, g, QQ(1,2))

    # 创建多项式环 R，定义变量 x, y，使用实数环 RR
    R, x, y = ring("x,y", RR)

    # 定义实数多项式 f 和 g
    f = 2.1*x*y**2 - 2.2*x*y + 2.1*x
    g = 1.0*x**3

    # 使用 R.dmp_ff_prs_gcd 函数计算 f 和 g 的最大公因式
    assert R.dmp_ff_prs_gcd(f, g) == \
        (1.0*x, 2.1*y**2 - 2.2*y + 2.1, 1.0*x**2)
# 定义一个测试函数，用于测试整数多项式环中的最小公倍数计算
def test_dup_lcm():
    # 创建整数多项式环 R，其中包含变量 x
    R, x = ring("x", ZZ)

    # 断言：计算两个整数的最小公倍数
    assert R.dup_lcm(2, 6) == 6

    # 断言：计算两个多项式的最小公倍数，这里 x**3 是一个整数多项式
    assert R.dup_lcm(2*x**3, 6*x) == 6*x**3
    assert R.dup_lcm(2*x**3, 3*x) == 6*x**3

    # 断言：计算两个多项式的最小公倍数，这里包含了不同的常数项
    assert R.dup_lcm(x**2 + x, x) == x**2 + x
    assert R.dup_lcm(x**2 + x, 2*x) == 2*x**2 + 2*x
    assert R.dup_lcm(x**2 + 2*x, x) == x**2 + 2*x
    assert R.dup_lcm(2*x**2 + x, x) == 2*x**2 + x
    assert R.dup_lcm(2*x**2 + x, 2*x) == 4*x**2 + 2*x


# 定义一个测试函数，用于测试多变量多项式环中的最小公倍数计算
def test_dmp_lcm():
    # 创建多变量多项式环 R，其中包含变量 x 和 y
    R, x, y = ring("x,y", ZZ)

    # 断言：计算两个整数的最小公倍数
    assert R.dmp_lcm(2, 6) == 6
    # 断言：计算两个多项式的最小公倍数，包含变量 x 和 y
    assert R.dmp_lcm(x, y) == x*y

    # 断言：计算两个多项式的最小公倍数，包含变量 x, y 和指数
    assert R.dmp_lcm(2*x**3, 6*x*y**2) == 6*x**3*y**2
    assert R.dmp_lcm(2*x**3, 3*x*y**2) == 6*x**3*y**2

    # 断言：计算两个多项式的最小公倍数，包含不同的变量和指数
    assert R.dmp_lcm(x**2*y, x*y**2) == x**2*y**2

    # 创建两个复杂的多项式 f, g 和预期的最小公倍数 h
    f = 2*x*y**5 - 3*x*y**4 - 2*x*y**3 + 3*x*y**2
    g = y**5 - 2*y**3 + y
    h = 2*x*y**7 - 3*x*y**6 - 4*x*y**5 + 6*x*y**4 + 2*x*y**3 - 3*x*y**2

    # 断言：计算复杂多项式 f 和 g 的最小公倍数应该等于 h
    assert R.dmp_lcm(f, g) == h

    # 创建两个更复杂的多项式 f, g 和预期的最小公倍数 h
    f = x**3 - 3*x**2*y - 9*x*y**2 - 5*y**3
    g = x**4 + 6*x**3*y + 12*x**2*y**2 + 10*x*y**3 + 3*y**4
    h = x**5 + x**4*y - 18*x**3*y**2 - 50*x**2*y**3 - 47*x*y**4 - 15*y**5

    # 断言：计算更复杂的多项式 f 和 g 的最小公倍数应该等于 h
    assert R.dmp_lcm(f, g) == h


# 定义一个测试函数，用于测试多变量多项式环中的内容计算
def test_dmp_content():
    # 创建多变量多项式环 R，其中包含变量 x 和 y
    R, x, y = ring("x,y", ZZ)

    # 断言：计算整数的内容（即绝对值）
    assert R.dmp_content(-2) == 2

    # 创建多项式 f 和预期的内容 f.drop(x)
    f, g, F = 3*y**2 + 2*y + 1, 1, 0

    # 循环计算多项式 F 的不同次数并加入到 F 中
    for i in range(0, 5):
        g *= f
        F += x**i*g

    # 断言：计算多项式 F 的内容，应该等于 f.drop(x)
    assert R.dmp_content(F) == f.drop(x)

    # 创建更多的多变量多项式环 R，包含变量 x, y 和 z
    R, x, y, z = ring("x,y,z", ZZ)

    # 断言：计算给定多项式的内容，应该等于 1
    assert R.dmp_content(f_4) == 1
    assert R.dmp_content(f_5) == 1

    # 创建更多的多变量多项式环 R，包含变量 x, y, z 和 t
    R, x, y, z, t = ring("x,y,z,t", ZZ)

    # 断言：计算给定多项式的内容，应该等于 1
    assert R.dmp_content(f_6) == 1


# 定义一个测试函数，用于测试整数多项式环中的因式分解
def test_dup_cancel():
    # 创建整数多项式环 R，其中包含变量 x
    R, x = ring("x", ZZ)

    # 定义两个整数多项式 f 和 g
    f = 2*x**2 - 2
    g = x**2 - 2*x + 1

    # 定义两个预期的结果 p 和 q
    p = 2*x + 2
    q = x - 1

    # 断言：计算 f 和 g 的因式分解结果应该等于 (p, q)
    assert R.dup_cancel(f, g) == (p, q)
    assert R.dup_cancel(f, g, include=False) == (1, 1, p, q)

    # 定义两个新的整数多项式 f 和 g
    f = -x - 2
    g = 3*x - 4

    # 定义两个预期的结果 F 和 G
    F = x + 2
    G = -3*x + 4

    # 断言：计算 f 和 g 的因式分解结果应该等于 (f, g)
    assert R.dup_cancel(f, g) == (f, g)
    assert R.dup_cancel(F, G) == (f, g)

    # 断言：计算 0 和 0 的因式分解结果应该等于 (0, 0)
    assert R.dup_cancel(0, 0) == (0, 0)
    assert R.dup_cancel(0, 0, include=False) == (1, 1, 0, 0)

    # 断言：计算 x 和 0 的因式分解结果应该等于 (1, 0)
    assert R.dup_cancel(x, 0) == (1, 0)
    assert R.dup_cancel(x, 0, include=False) == (1, 1, 1, 0)

    # 断言：计算 0 和 x 的因式分解结果应该等于 (0, 1)
    assert R.dup_cancel(0, x) == (0, 1)
    assert R.dup_cancel(0, x, include=False) == (1, 1, 0, 1)

    # 定义一个新的整数多项式 f 和 g
    f = 0
    g = x
    one = 1

    # 断言：计算 f 和 g 的因式分解结果，包括单位因子
    assert R.dup_cancel(f, g, include=True) == (f, one)


# 定义一个测试函数，用于测试多变量多项式环中的因式分解
def test_dmp_cancel():
    # 创建多变量多项式环 R，其中包含变量 x 和 y
    R, x, y = ring
    # 调用 R 对象的 dmp_cancel 方法，传入参数 f 和 g，验证其返回结果是否等于元组 (p, q)
    assert R.dmp_cancel(f, g) == (p, q)
    
    # 调用 R 对象的 dmp_cancel 方法，传入参数 f、g 和 include=False，验证其返回结果是否等于元组 (1, 1, p, q)
    assert R.dmp_cancel(f, g, include=False) == (1, 1, p, q)
    
    # 调用 R 对象的 dmp_cancel 方法，传入参数 0 和 0，验证其返回结果是否等于元组 (0, 0)
    assert R.dmp_cancel(0, 0) == (0, 0)
    
    # 调用 R 对象的 dmp_cancel 方法，传入参数 0、0 和 include=False，验证其返回结果是否等于元组 (1, 1, 0, 0)
    assert R.dmp_cancel(0, 0, include=False) == (1, 1, 0, 0)
    
    # 调用 R 对象的 dmp_cancel 方法，传入参数 y 和 0，验证其返回结果是否等于元组 (1, 0)
    assert R.dmp_cancel(y, 0) == (1, 0)
    
    # 调用 R 对象的 dmp_cancel 方法，传入参数 y、0 和 include=False，验证其返回结果是否等于元组 (1, 1, 1, 0)
    assert R.dmp_cancel(y, 0, include=False) == (1, 1, 1, 0)
    
    # 调用 R 对象的 dmp_cancel 方法，传入参数 0 和 y，验证其返回结果是否等于元组 (0, 1)
    assert R.dmp_cancel(0, y) == (0, 1)
    
    # 调用 R 对象的 dmp_cancel 方法，传入参数 0、y 和 include=False，验证其返回结果是否等于元组 (1, 1, 0, 1)
    assert R.dmp_cancel(0, y, include=False) == (1, 1, 0, 1)
```