# `D:\src\scipysrc\scipy\scipy\special\tests\test_ellip_harm.py`

```
# 导入所需的库和模块
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
                           assert_, suppress_warnings)
from scipy.special._testutils import assert_func_equal
from scipy.special import ellip_harm, ellip_harm_2, ellip_normal
from scipy.integrate import IntegrationWarning
from numpy import sqrt, pi

# 定义测试函数 test_ellip_potential
def test_ellip_potential():
    # 定义函数 change_coefficient，计算系数 x, y, z
    def change_coefficient(lambda1, mu, nu, h2, k2):
        x = sqrt(lambda1**2*mu**2*nu**2/(h2*k2))
        y = sqrt((lambda1**2 - h2)*(mu**2 - h2)*(h2 - nu**2)/(h2*(k2 - h2)))
        z = sqrt((lambda1**2 - k2)*(k2 - mu**2)*(k2 - nu**2)/(k2*(k2 - h2)))
        return x, y, z

    # 定义函数 solid_int_ellip，计算椭圆谐函数的乘积
    def solid_int_ellip(lambda1, mu, nu, n, p, h2, k2):
        return (ellip_harm(h2, k2, n, p, lambda1)*ellip_harm(h2, k2, n, p, mu)
               * ellip_harm(h2, k2, n, p, nu))

    # 定义函数 solid_int_ellip2，计算第一个椭圆谐函数乘积的另一种形式
    def solid_int_ellip2(lambda1, mu, nu, n, p, h2, k2):
        return (ellip_harm_2(h2, k2, n, p, lambda1)
                * ellip_harm(h2, k2, n, p, mu)*ellip_harm(h2, k2, n, p, nu))

    # 定义函数 summation，对椭球面积分进行求和
    def summation(lambda1, mu1, nu1, lambda2, mu2, nu2, h2, k2):
        tol = 1e-8
        sum1 = 0
        for n in range(20):
            xsum = 0
            for p in range(1, 2*n+2):
                xsum += (4*pi*(solid_int_ellip(lambda2, mu2, nu2, n, p, h2, k2)
                    * solid_int_ellip2(lambda1, mu1, nu1, n, p, h2, k2)) /
                    (ellip_normal(h2, k2, n, p)*(2*n + 1)))
            if abs(xsum) < 0.1*tol*abs(sum1):
                break
            sum1 += xsum
        return sum1, xsum

    # 定义函数 potential，计算椭球势能
    def potential(lambda1, mu1, nu1, lambda2, mu2, nu2, h2, k2):
        x1, y1, z1 = change_coefficient(lambda1, mu1, nu1, h2, k2)
        x2, y2, z2 = change_coefficient(lambda2, mu2, nu2, h2, k2)
        res = sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return 1/res

    # 定义测试点集
    pts = [
        (120, sqrt(19), 2, 41, sqrt(17), 2, 15, 25),
        (120, sqrt(16), 3.2, 21, sqrt(11), 2.9, 11, 20),
       ]

    # 使用 suppress_warnings 上下文管理器抑制特定集成警告
    with suppress_warnings() as sup:
        sup.filter(IntegrationWarning, "The occurrence of roundoff error")
        sup.filter(IntegrationWarning, "The maximum number of subdivisions")

        # 遍历测试点
        for p in pts:
            err_msg = repr(p)
            # 计算精确势能值和求和结果
            exact = potential(*p)
            result, last_term = summation(*p)
            # 断言精确度
            assert_allclose(exact, result, atol=0, rtol=1e-8, err_msg=err_msg)
            assert_(abs(result - exact) < 10*abs(last_term), err_msg)


# 定义测试函数 test_ellip_norm
def test_ellip_norm():

    # 定义几个函数 G01, G11, G12, G13, G22 分别计算不同的椭球面积分
    def G01(h2, k2):
        return 4*pi

    def G11(h2, k2):
        return 4*pi*h2*k2/3

    def G12(h2, k2):
        return 4*pi*h2*(k2 - h2)/3

    def G13(h2, k2):
        return 4*pi*k2*(k2 - h2)/3

    def G22(h2, k2):
        res = (2*(h2**4 + k2**4) - 4*h2*k2*(h2**2 + k2**2) + 6*h2**2*k2**2 +
        sqrt(h2**2 + k2**2 - h2*k2)*(-2*(h2**3 + k2**3) + 3*h2*k2*(h2 + k2)))
        return 16*pi/405*res
    # 定义函数 G21，计算椭圆积分表达式中的 G21 值
    def G21(h2, k2):
        res = (2*(h2**4 + k2**4) - 4*h2*k2*(h2**2 + k2**2) + 6*h2**2*k2**2
        + sqrt(h2**2 + k2**2 - h2*k2)*(2*(h2**3 + k2**3) - 3*h2*k2*(h2 + k2)))
        return 16*pi/405*res

    # 定义函数 G23，计算椭圆积分表达式中的 G23 值
    def G23(h2, k2):
        return 4*pi*h2**2*k2*(k2 - h2)/15

    # 定义函数 G24，计算椭圆积分表达式中的 G24 值
    def G24(h2, k2):
        return 4*pi*h2*k2**2*(k2 - h2)/15

    # 定义函数 G25，计算椭圆积分表达式中的 G25 值
    def G25(h2, k2):
        return 4*pi*h2*k2*(k2 - h2)**2/15

    # 定义函数 G32，计算椭圆积分表达式中的 G32 值
    def G32(h2, k2):
        res = (16*(h2**4 + k2**4) - 36*h2*k2*(h2**2 + k2**2) + 46*h2**2*k2**2
        + sqrt(4*(h2**2 + k2**2) - 7*h2*k2)*(-8*(h2**3 + k2**3) +
        11*h2*k2*(h2 + k2)))
        return 16*pi/13125*k2*h2*res

    # 定义函数 G31，计算椭圆积分表达式中的 G31 值
    def G31(h2, k2):
        res = (16*(h2**4 + k2**4) - 36*h2*k2*(h2**2 + k2**2) + 46*h2**2*k2**2
        + sqrt(4*(h2**2 + k2**2) - 7*h2*k2)*(8*(h2**3 + k2**3) -
        11*h2*k2*(h2 + k2)))
        return 16*pi/13125*h2*k2*res

    # 定义函数 G34，计算椭圆积分表达式中的 G34 值
    def G34(h2, k2):
        res = (6*h2**4 + 16*k2**4 - 12*h2**3*k2 - 28*h2*k2**3 + 34*h2**2*k2**2
        + sqrt(h2**2 + 4*k2**2 - h2*k2)*(-6*h2**3 - 8*k2**3 + 9*h2**2*k2 +
                                            13*h2*k2**2))
        return 16*pi/13125*h2*(k2 - h2)*res

    # 定义函数 G33，计算椭圆积分表达式中的 G33 值
    def G33(h2, k2):
        res = (6*h2**4 + 16*k2**4 - 12*h2**3*k2 - 28*h2*k2**3 + 34*h2**2*k2**2
        + sqrt(h2**2 + 4*k2**2 - h2*k2)*(6*h2**3 + 8*k2**3 - 9*h2**2*k2 -
        13*h2*k2**2))
        return 16*pi/13125*h2*(k2 - h2)*res

    # 定义函数 G36，计算椭圆积分表达式中的 G36 值
    def G36(h2, k2):
        res = (16*h2**4 + 6*k2**4 - 28*h2**3*k2 - 12*h2*k2**3 + 34*h2**2*k2**2
        + sqrt(4*h2**2 + k2**2 - h2*k2)*(-8*h2**3 - 6*k2**3 + 13*h2**2*k2 +
        9*h2*k2**2))
        return 16*pi/13125*k2*(k2 - h2)*res

    # 定义函数 G35，计算椭圆积分表达式中的 G35 值
    def G35(h2, k2):
        res = (16*h2**4 + 6*k2**4 - 28*h2**3*k2 - 12*h2*k2**3 + 34*h2**2*k2**2
        + sqrt(4*h2**2 + k2**2 - h2*k2)*(8*h2**3 + 6*k2**3 - 13*h2**2*k2 -
        9*h2*k2**2))
        return 16*pi/13125*k2*(k2 - h2)*res

    # 定义函数 G37，计算椭圆积分表达式中的 G37 值
    def G37(h2, k2):
        return 4*pi*h2**2*k2**2*(k2 - h2)**2/105

    # 定义已知的函数字典，将 (n, p) 映射到对应的椭圆积分函数
    known_funcs = {(0, 1): G01, (1, 1): G11, (1, 2): G12, (1, 3): G13,
                   (2, 1): G21, (2, 2): G22, (2, 3): G23, (2, 4): G24,
                   (2, 5): G25, (3, 1): G31, (3, 2): G32, (3, 3): G33,
                   (3, 4): G34, (3, 5): G35, (3, 6): G36, (3, 7): G37}

    # 定义函数 _ellip_norm，使用已知的函数字典计算椭圆积分值
    def _ellip_norm(n, p, h2, k2):
        func = known_funcs[n, p]
        return func(h2, k2)
    _ellip_norm = np.vectorize(_ellip_norm)

    # 定义函数 ellip_normal_known，对外接口，计算椭圆积分值
    def ellip_normal_known(h2, k2, n, p):
        return _ellip_norm(n, p, h2, k2)

    # 生成大和小的 h2 < k2 对
    np.random.seed(1234)
    h2 = np.random.pareto(0.5, size=1)
    k2 = h2 * (1 + np.random.pareto(0.5, size=h2.size))

    # 生成待测试的点集
    points = []
    for n in range(4):
        for p in range(1, 2*n+2):
            points.append((h2, k2, np.full(h2.size, n), np.full(h2.size, p)))
    points = np.array(points)

    # 忽略积分时可能出现的警告
    with suppress_warnings() as sup:
        sup.filter(IntegrationWarning, "The occurrence of roundoff error")
        # 断言函数 ellip_normal 和 ellip_normal_known 在给定点集上相等
        assert_func_equal(ellip_normal, ellip_normal_known, points, rtol=1e-12)
def test_ellip_harm_2():
    # 定义函数 I1，计算三个 ellip_harm_2 函数结果的加权平均值
    def I1(h2, k2, s):
        res = (ellip_harm_2(h2, k2, 1, 1, s)/(3 * ellip_harm(h2, k2, 1, 1, s))
               + ellip_harm_2(h2, k2, 1, 2, s)/(3 * ellip_harm(h2, k2, 1, 2, s))
               + ellip_harm_2(h2, k2, 1, 3, s)/(3 * ellip_harm(h2, k2, 1, 3, s)))
        return res

    # 使用 suppress_warnings 上下文管理器，过滤 IntegrationWarning 警告
    with suppress_warnings() as sup:
        sup.filter(IntegrationWarning, "The occurrence of roundoff error")
        # 断言 I1 函数的计算结果接近于指定值
        assert_almost_equal(I1(5, 8, 10), 1/(10*sqrt((100-5)*(100-8))))

        # 使用 arXiv:1204.0267 中的数据，断言 ellip_harm_2 函数的计算结果接近于指定值
        assert_almost_equal(ellip_harm_2(5, 8, 2, 1, 10), 0.00108056853382)
        assert_almost_equal(ellip_harm_2(5, 8, 2, 2, 10), 0.00105820513809)
        assert_almost_equal(ellip_harm_2(5, 8, 2, 3, 10), 0.00106058384743)
        assert_almost_equal(ellip_harm_2(5, 8, 2, 4, 10), 0.00106774492306)
        assert_almost_equal(ellip_harm_2(5, 8, 2, 5, 10), 0.00107976356454)


def test_ellip_harm():
    # 定义多个 E 函数，计算不同参数下的 ellip_harm 函数结果

    def E01(h2, k2, s):
        return 1

    def E11(h2, k2, s):
        return s

    def E12(h2, k2, s):
        return sqrt(abs(s*s - h2))

    def E13(h2, k2, s):
        return sqrt(abs(s*s - k2))

    def E21(h2, k2, s):
        return s*s - 1/3*((h2 + k2) + sqrt(abs((h2 + k2)*(h2 + k2)-3*h2*k2)))

    def E22(h2, k2, s):
        return s*s - 1/3*((h2 + k2) - sqrt(abs((h2 + k2)*(h2 + k2)-3*h2*k2)))

    def E23(h2, k2, s):
        return s * sqrt(abs(s*s - h2))

    def E24(h2, k2, s):
        return s * sqrt(abs(s*s - k2))

    def E25(h2, k2, s):
        return sqrt(abs((s*s - h2)*(s*s - k2)))

    def E31(h2, k2, s):
        return s*s*s - (s/5)*(2*(h2 + k2) + sqrt(4*(h2 + k2)*(h2 + k2) -
        15*h2*k2))

    def E32(h2, k2, s):
        return s*s*s - (s/5)*(2*(h2 + k2) - sqrt(4*(h2 + k2)*(h2 + k2) -
        15*h2*k2))

    def E33(h2, k2, s):
        return sqrt(abs(s*s - h2))*(s*s - 1/5*((h2 + 2*k2) + sqrt(abs((h2 +
        2*k2)*(h2 + 2*k2) - 5*h2*k2))))

    def E34(h2, k2, s):
        return sqrt(abs(s*s - h2))*(s*s - 1/5*((h2 + 2*k2) - sqrt(abs((h2 +
        2*k2)*(h2 + 2*k2) - 5*h2*k2))))

    def E35(h2, k2, s):
        return sqrt(abs(s*s - k2))*(s*s - 1/5*((2*h2 + k2) + sqrt(abs((2*h2
        + k2)*(2*h2 + k2) - 5*h2*k2))))

    def E36(h2, k2, s):
        return sqrt(abs(s*s - k2))*(s*s - 1/5*((2*h2 + k2) - sqrt(abs((2*h2
        + k2)*(2*h2 + k2) - 5*h2*k2))))

    def E37(h2, k2, s):
        return s * sqrt(abs((s*s - h2)*(s*s - k2)))

    # 断言 ellip_harm 函数的两次调用结果相等
    assert_equal(ellip_harm(5, 8, 1, 2, 2.5, 1, 1),
                 ellip_harm(5, 8, 1, 2, 2.5))

    # 创建一个空列表 point_ref
    known_funcs = {(0, 1): E01, (1, 1): E11, (1, 2): E12, (1, 3): E13,
                   (2, 1): E21, (2, 2): E22, (2, 3): E23, (2, 4): E24,
                   (2, 5): E25, (3, 1): E31, (3, 2): E32, (3, 3): E33,
                   (3, 4): E34, (3, 5): E35, (3, 6): E36, (3, 7): E37}

    # 创建一个空列表 point_ref
    point_ref = []
    # 定义一个函数，计算已知椭圆函数在指定点上的值，并返回结果列表
    def ellip_harm_known(h2, k2, n, p, s):
        # 循环遍历给定数组的大小
        for i in range(h2.size):
            # 根据索引从预定义函数字典中获取相应的函数
            func = known_funcs[(int(n[i]), int(p[i]))]
            # 调用获取到的函数，计算指定点上的函数值，并将结果添加到列表中
            point_ref.append(func(h2[i], k2[i], s[i]))
        # 返回计算得到的结果列表
        return point_ref

    # 设置随机种子，以便结果可重现
    np.random.seed(1234)
    # 生成服从指定参数的 Pareto 分布的随机数组
    h2 = np.random.pareto(0.5, size=30)
    # 根据 h2 生成 k2 数组，符合特定的关系
    k2 = h2*(1 + np.random.pareto(0.5, size=h2.size))
    # 生成服从指定参数的 Pareto 分布的随机数组
    s = np.random.pareto(0.5, size=h2.size)
    # 初始化空列表，用于存储生成的点
    points = []
    # 循环遍历 h2 数组的大小
    for i in range(h2.size):
        # 循环遍历数字 0 到 3
        for n in range(4):
            # 循环遍历数字 1 到 2*n+1
            for p in range(1, 2*n+2):
                # 将当前点的参数组成元组，并添加到 points 列表中
                points.append((h2[i], k2[i], n, p, s[i]))
    # 将 points 列表转换为 NumPy 数组
    points = np.array(points)
    # 调用自定义断言函数，检查 ellip_harm 函数和 ellip_harm_known 函数在给定点上的结果是否足够接近
    assert_func_equal(ellip_harm, ellip_harm_known, points, rtol=1e-12)
# 定义一个测试函数，用于验证椭圆函数调和函数在给定无效的参数时的回归情况
def test_ellip_harm_invalid_p():
    # 回归测试。预期结果应该是 NaN（Not a Number）。
    n = 4
    # 设置参数 p，使得 p > 2*n + 1
    p = 2*n + 2
    # 调用 ellip_harm 函数，传入参数 0.5, 2.0, n, p, 0.2，返回结果存储在 result 中
    result = ellip_harm(0.5, 2.0, n, p, 0.2)
    # 使用 NumPy 的断言函数 assert 来验证结果是否为 NaN
    assert np.isnan(result)
```