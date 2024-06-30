# `D:\src\scipysrc\sympy\sympy\polys\tests\test_heuristicgcd.py`

```
# 从 sympy.polys.rings 模块导入 ring 函数
from sympy.polys.rings import ring
# 从 sympy.polys.domains 模块导入 ZZ
from sympy.polys.domains import ZZ
# 从 sympy.polys.heuristicgcd 模块导入 heugcd 函数
from sympy.polys.heuristicgcd import heugcd

# 定义一个测试函数 test_heugcd_univariate_integers，测试单变量整数情况下的 heugcd 函数
def test_heugcd_univariate_integers():
    # 创建整数多项式环 R 和变量 x
    R, x = ring("x", ZZ)

    # 第一组测试
    f = x**4 + 8*x**3 + 21*x**2 + 22*x + 8
    g = x**3 + 6*x**2 + 11*x + 6
    h = x**2 + 3*x + 2
    cff = x**2 + 5*x + 4
    cfg = x + 3
    # 断言 heugcd 函数的输出符合预期
    assert heugcd(f, g) == (h, cff, cfg)

    # 第二组测试
    f = x**4 - 4
    g = x**4 + 4*x**2 + 4
    h = x**2 + 2
    cff = x**2 - 2
    cfg = x**2 + 2
    # 断言 heugcd 函数的输出符合预期
    assert heugcd(f, g) == (h, cff, cfg)

    # 第三组测试
    f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    h = 1
    cff = f
    cfg = g
    # 断言 heugcd 函数的输出符合预期
    assert heugcd(f, g) == (h, cff, cfg)

    # 第四组测试
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
    # TODO: assert heugcd(f, f.diff(x))[0] == g

    # 第五组测试
    f = 1317378933230047068160*x + 2945748836994210856960
    g = 120352542776360960*x + 269116466014453760
    h = 120352542776360960*x + 269116466014453760
    cff = 10946
    cfg = 1
    # 断言 heugcd 函数的输出符合预期
    assert heugcd(f, g) == (h, cff, cfg)

# 定义一个测试函数 test_heugcd_multivariate_integers，测试多变量整数情况下的 heugcd 函数
def test_heugcd_multivariate_integers():
    # 创建多变量整数多项式环 R 和变量 x, y
    R, x, y = ring("x,y", ZZ)

    # 第一组测试
    f, g = 2*x**2 + 4*x + 2, x + 1
    # 断言 heugcd 函数的输出符合预期
    assert heugcd(f, g) == (x + 1, 2*x + 2, 1)

    # 第二组测试
    f, g = x + 1, 2*x**2 + 4*x + 2
    # 断言 heugcd 函数的输出符合预期
    assert heugcd(f, g) == (x + 1, 1, 2*x + 2)

    # 创建多变量整数多项式环 R 和变量 x, y, z, u
    R, x, y, z, u = ring("x,y,z,u", ZZ)

    # 第三组测试
    f, g = u**2 + 2*u + 1, 2*u + 2
    # 断言 heugcd 函数的输出符合预期
    assert heugcd(f, g) == (u + 1, u + 1, 2)

    # 第四组测试
    f, g = z**2*u**2 + 2*z**2*u + z**2 + z*u + z, u**2 + 2*u + 1
    h, cff, cfg = u + 1, z**2*u + z**2 + z, u + 1
    # 断言 heugcd 函数的输出符合预期
    assert heugcd(f, g) == (h, cff, cfg)
    # 断言交换输入顺序时 heugcd 函数的输出符合预期
    assert heugcd(g, f) == (h, cfg, cff)

    # 创建多变量整数多项式环 R 和变量 x, y, z
    R, x, y, z = ring("x,y,z", ZZ)

    # 获取 Fateman 多项式 F_1 的定义
    f, g, h = R.fateman_poly_F_1()
    # 使用 heugcd 函数计算并分解结果
    H, cff, cfg = heugcd(f, g)
    # 断言 heugcd 函数的输出符合预期
    assert H == h and H*cff == f and H*cfg == g

    # 创建多变量整数多项式环 R 和变量 x, y, z, u, v
    R, x, y, z, u, v = ring("x,y,z,u,v", ZZ)

    # 获取 Fateman 多项式 F_1 的定义
    f, g, h = R.fateman_poly_F_1()
    # 使用 heugcd 函数计算并分解结果
    H, cff, cfg = heugcd(f, g)
    # 断言 heugcd 函数的输出符合预期
    assert H == h and H*cff == f and H*cfg == g

    # 创建多变量整数多项式环 R 和变量 x, y, z, u, v, a, b
    R, x, y, z, u, v, a, b = ring("x,y,z,u,v,a,b", ZZ)

    # 获取 Fateman 多项式 F_1 的定义
    f, g, h = R.fateman_poly_F_1()
    # 使用 heugcd 函数计算并分解结果
    H, cff, cfg = heugcd(f, g)
    # 断言 heugcd 函数的输出符合预期
    assert H == h and H*cff == f and H*cfg == g
    # 使用给定的变量名创建整数环R，并定义变量x, y, z, u, v, a, b, c, d
    R, x, y, z, u, v, a, b, c, d = ring("x,y,z,u,v,a,b,c,d", ZZ)
    
    # 调用R对象的方法fateman_poly_F_1()，返回三个多项式f, g, h，并进行heugcd算法计算
    f, g, h = R.fateman_poly_F_1()
    H, cff, cfg = heugcd(f, g)
    
    # 断言确保heugcd计算正确：H等于h，并且H乘以cff等于f，H乘以cfg等于g
    assert H == h and H*cff == f and H*cfg == g
    
    # 重新定义整数环R，并定义变量x, y, z
    R, x, y, z = ring("x,y,z", ZZ)
    
    # 调用R对象的方法fateman_poly_F_2()，返回三个多项式f, g, h，并进行heugcd算法计算
    f, g, h = R.fateman_poly_F_2()
    H, cff, cfg = heugcd(f, g)
    
    # 断言确保heugcd计算正确：H等于h，并且H乘以cff等于f，H乘以cfg等于g
    assert H == h and H*cff == f and H*cfg == g
    
    # 重新定义整数环R，并定义变量x, y, z
    R, x, y, z = ring("x,y,z", ZZ)
    
    # 调用R对象的方法fateman_poly_F_3()，返回三个多项式f, g, h，并进行heugcd算法计算
    f, g, h = R.fateman_poly_F_3()
    H, cff, cfg = heugcd(f, g)
    
    # 断言确保heugcd计算正确：H等于h，并且H乘以cff等于f，H乘以cfg等于g
    assert H == h and H*cff == f and H*cfg == g
    
    # 重新定义整数环R，并定义变量x, y, z, t
    R, x, y, z, t = ring("x,y,z,t", ZZ)
    
    # 调用R对象的方法fateman_poly_F_3()，返回三个多项式f, g, h，并进行heugcd算法计算
    f, g, h = R.fateman_poly_F_3()
    H, cff, cfg = heugcd(f, g)
    
    # 断言确保heugcd计算正确：H等于h，并且H乘以cff等于f，H乘以cfg等于g
    assert H == h and H*cff == f and H*cfg == g
# 定义一个测试函数，用于验证问题编号 10996
def test_issue_10996():
    # 创建一个多项式环 R，并定义变量 x, y, z，类型为整数环 ZZ
    R, x, y, z = ring("x,y,z", ZZ)

    # 定义多项式 f
    f = 12*x**6*y**7*z**3 - 3*x**4*y**9*z**3 + 12*x**3*y**5*z**4
    # 定义多项式 g，中间包含换行符
    g = -48*x**7*y**8*z**3 + 12*x**5*y**10*z**3 - 48*x**5*y**7*z**2 + \
        36*x**4*y**7*z - 48*x**4*y**6*z**4 + 12*x**3*y**9*z**2 - 48*x**3*y**4 \
        - 9*x**2*y**9*z - 48*x**2*y**5*z**3 + 12*x*y**6 + 36*x*y**5*z**2 - 48*y**2*z

    # 使用 heugcd 函数计算 f 和 g 的最大公因式 H，以及对应的系数 cff 和 cfg
    H, cff, cfg = heugcd(f, g)

    # 断言 H 的值应该等于 12*x**3*y**4 - 3*x*y**6 + 12*y**2*z
    assert H == 12*x**3*y**4 - 3*x*y**6 + 12*y**2*z
    # 断言 H 乘以 cff 等于 f，H 乘以 cfg 等于 g
    assert H*cff == f and H*cfg == g


# 定义一个测试函数，用于验证问题编号 25793
def test_issue_25793():
    # 创建一个多项式环 R，只包含变量 x，类型为整数环 ZZ
    R, x = ring("x", ZZ)
    # 定义多项式 f，f = x - 4851，注释指出 f 失效的起始值大于 4850
    f = x - 4851  # failure starts for values more than 4850
    # 定义多项式 g，g = f*(2*x + 1)
    g = f*(2*x + 1)
    # 使用 R.dup_zz_heu_gcd 函数计算 f 和 g 的最大公因式 H，以及对应的系数 cff 和 cfg
    H, cff, cfg = R.dup_zz_heu_gcd(f, g)
    # 断言 H 的值应该等于 f
    assert H == f
    # 需要为 dmp 也进行测试，该测试在修改之前应该失败
```