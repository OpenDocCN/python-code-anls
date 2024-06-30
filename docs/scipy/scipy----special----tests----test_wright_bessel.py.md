# `D:\src\scipysrc\scipy\scipy\special\tests\test_wright_bessel.py`

```
# 引入 itertools 库，用于生成迭代器的函数
from itertools import product

# 引入 pytest 库，用于编写和运行测试用例
import pytest

# 引入 numpy 库，并从中引入 assert_equal 和 assert_allclose 函数
import numpy as np
from numpy.testing import assert_equal, assert_allclose

# 引入 scipy.special 库，并从中分别引入 log_wright_bessel, loggamma, rgamma, wright_bessel 函数
import scipy.special as sc
from scipy.special import log_wright_bessel, loggamma, rgamma, wright_bessel


# 使用 pytest 的 parametrize 装饰器，对函数 test_wright_bessel_zero 进行参数化测试
@pytest.mark.parametrize('a', [0, 1e-6, 0.1, 0.5, 1, 10])
@pytest.mark.parametrize('b', [0, 1e-6, 0.1, 0.5, 1, 10])
def test_wright_bessel_zero(a, b):
    """Test at x = 0."""
    # 断言 wright_bessel(a, b, 0.) 的返回值与 rgamma(b) 相等
    assert_equal(wright_bessel(a, b, 0.), rgamma(b))
    # 断言 log_wright_bessel(a, b, 0.) 的返回值与 -loggamma(b) 相近
    assert_allclose(log_wright_bessel(a, b, 0.), -loggamma(b))


# 使用 pytest 的 parametrize 装饰器，对函数 test_wright_bessel_iv 进行参数化测试
@pytest.mark.parametrize('b', [0, 1e-6, 0.1, 0.5, 1, 10])
@pytest.mark.parametrize('x', [0, 1e-6, 0.1, 0.5, 1])
def test_wright_bessel_iv(b, x):
    """Test relation of wright_bessel and modified bessel function iv.

    iv(z) = (1/2*z)**v * Phi(1, v+1; 1/4*z**2).
    See https://dlmf.nist.gov/10.46.E2
    """
    # 如果 x 不等于 0，则进行以下断言
    if x != 0:
        # 计算 v 值
        v = b - 1
        # 计算 wright_bessel(1, v + 1, x**2 / 4.)
        wb = wright_bessel(1, v + 1, x**2 / 4.)
        # 断言 np.power(x / 2., v) * wb 与 sc.iv(v, x) 在指定的相对和绝对误差下相近
        assert_allclose(np.power(x / 2., v) * wb,
                        sc.iv(v, x),
                        rtol=1e-11, atol=1e-11)


# 使用 pytest 的 parametrize 装饰器，对函数 test_wright_functional 进行参数化测试
@pytest.mark.parametrize('a', [0, 1e-6, 0.1, 0.5, 1, 10])
@pytest.mark.parametrize('b', [1, 1 + 1e-3, 2, 5, 10])
@pytest.mark.parametrize('x', [0, 1e-6, 0.1, 0.5, 1, 5, 10, 100])
def test_wright_functional(a, b, x):
    """Test functional relation of wright_bessel.

    Phi(a, b-1, z) = a*z*Phi(a, b+a, z) + (b-1)*Phi(a, b, z)

    Note that d/dx Phi(a, b, x) = Phi(a, b-1, x)
    See Eq. (22) of
    B. Stankovic, On the Function of E. M. Wright,
    Publ. de l' Institut Mathematique, Beograd,
    Nouvelle S`er. 10 (1970), 113-124.
    """
    # 断言 wright_bessel(a, b - 1, x) 与右侧表达式的结果在指定的相对和绝对误差下相近
    assert_allclose(wright_bessel(a, b - 1, x),
                    a * x * wright_bessel(a, b + a, x)
                    + (b - 1) * wright_bessel(a, b, x),
                    rtol=1e-8, atol=1e-8)


# 定义一个包含测试数据的 numpy 数组，用于存储不达到 1e-11 精度的行
grid_a_b_x_value_acc = np.array([
    [0.1, 100.0, 709.7827128933841, 8.026353022981087e+34, 2e-8],
    [0.5, 10.0, 709.7827128933841, 2.680788404494657e+48, 9e-8],
    [0.5, 10.0, 1000.0, 2.005901980702872e+64, 1e-8],
    [0.5, 100.0, 1000.0, 3.4112367580445246e-117, 6e-8],
    [1.0, 20.0, 100000.0, 1.7717158630699857e+225, 3e-11],
    [1.0, 100.0, 100000.0, 1.0269334596230763e+22, np.nan],
    # 定义一个二维列表，包含多个子列表，每个子列表包含五个元素
    [
        [1.0000000000000222, 20.0, 100000.0, 1.7717158630001672e+225, 3e-11],
        [1.0000000000000222, 100.0, 100000.0, 1.0269334595866202e+22, np.nan],
        [1.5, 0.0, 500.0, 15648961196.432373, 3e-11],
        [1.5, 2.220446049250313e-14, 500.0, 15648961196.431465, 3e-11],
        [1.5, 1e-10, 500.0, 15648961192.344728, 3e-11],
        [1.5, 1e-05, 500.0, 15648552437.334162, 3e-11],
        [1.5, 0.1, 500.0, 12049870581.10317, 2e-11],
        [1.5, 20.0, 100000.0, 7.81930438331405e+43, 3e-9],
        [1.5, 100.0, 100000.0, 9.653370857459075e-130, np.nan],
    ]
@pytest.mark.parametrize(
    'a, b, x, phi, accuracy',
    grid_a_b_x_value_acc.tolist())


# 使用pytest.mark.parametrize装饰器，为函数test_wright_data_grid_less_accurate参数化测试用例
# 参数a, b, x, phi, accuracy从grid_a_b_x_value_acc数组中获取，并转换为列表形式
def test_wright_data_grid_less_accurate(a, b, x, phi, accuracy):
    """Test cases of test_data that do not reach relative accuracy of 1e-11

    Here we test for reduced accuracy or even nan.
    """
    # 如果accuracy为NaN，则断言wright_bessel(a, b, x)也返回NaN
    if np.isnan(accuracy):
        assert np.isnan(wright_bessel(a, b, x))
    else:
        # 否则，使用assert_allclose断言wright_bessel(a, b, x)的结果与phi在相对误差为accuracy的情况下的接近程度
        assert_allclose(wright_bessel(a, b, x), phi, rtol=accuracy)


@pytest.mark.parametrize(
    'a, b, x',
    list(
        product([0, 0.1, 0.5, 1.5, 5, 10], [1, 2], [1e-3, 1, 5, 10])
    )
)


# 使用pytest.mark.parametrize装饰器，为函数test_log_wright_bessel_same_as_wright_bessel参数化测试用例
# 参数a, b, x从指定的product列表中获取
def test_log_wright_bessel_same_as_wright_bessel(a, b, x):
    """Test that log_wright_bessel equals log of wright_bessel."""
    # 使用assert_allclose断言log_wright_bessel(a, b, x)的结果与np.log(wright_bessel(a, b, x))在相对误差为1e-8的情况下的接近程度
    assert_allclose(
        log_wright_bessel(a, b, x),
        np.log(wright_bessel(a, b, x)),
        rtol=1e-8,
    )


@pytest.mark.parametrize(
    'a, b, x, phi, accuracy',


# 使用pytest.mark.parametrize装饰器，为函数rgamma_cached参数化测试用例
# 参数a, b, x, phi, accuracy需要后续提供
def mp_log_wright_bessel(a, b, x, dps=100, maxterms=10_000, method="d"):
    """Compute log of Wright's generalized Bessel function as Series with mpmath."""
    # 使用mp.workdps设置精度为dps
    with mp.workdps(dps):
        # 将a, b, x转换为mp.mpf类型
        a, b, x = mp.mpf(a), mp.mpf(b), mp.mpf(x)
        # 使用mp.nsum计算Wright广义贝塞尔函数的对数级数表示
        res = mp.nsum(lambda k: x**k / mp.fac(k)
                      * rgamma_cached(a * k + b, dps=dps),
                      [0, mp.inf],
                      tol=dps, method=method, steps=[maxterms]
                      )
        # 返回结果的自然对数
        return mp.log(res)
    [
        # Tuple 1: (0, 0, 0, -np.inf, 1e-11)
        (0, 0, 0, -np.inf, 1e-11),
        # Tuple 2: (0, 0, 1, -np.inf, 1e-11)
        (0, 0, 1, -np.inf, 1e-11),
        # Tuple 3: (0, 1, 1.23, 1.23, 1e-11)
        (0, 1, 1.23, 1.23, 1e-11),
        # Tuple 4: (0, 1, 1e50, 1e50, 1e-11)
        (0, 1, 1e50, 1e50, 1e-11),
        # Tuple 5: (1e-5, 0, 700, 695.0421608273609, 1e-11)
        (1e-5, 0, 700, 695.0421608273609, 1e-11),
        # Tuple 6: (1e-5, 0, 1e3, 995.40052566540066, 1e-11)
        (1e-5, 0, 1e3, 995.40052566540066, 1e-11),
        # Tuple 7: (1e-5, 100, 1e3, 640.8197935670078, 1e-11)
        (1e-5, 100, 1e3, 640.8197935670078, 1e-11),
        # Tuple 8: (1e-3, 0, 1e4, 9987.2229532297262, 1e-11)
        (1e-3, 0, 1e4, 9987.2229532297262, 1e-11),
        # Tuple 9: (1e-3, 0, 1e5, 99641.920687169507, 1e-11)
        (1e-3, 0, 1e5, 99641.920687169507, 1e-11),
        # Tuple 10: (1e-3, 0, 1e6, 994118.55560054416, 1e-11)
        (1e-3, 0, 1e6, 994118.55560054416, 1e-11),  # maxterms=1_000_000
        # Tuple 11: (1e-3, 10, 1e5, 99595.47710802537, 1e-11)
        (1e-3, 10, 1e5, 99595.47710802537, 1e-11),
        # Tuple 12: (1e-3, 50, 1e5, 99401.240922855647, 1e-3)
        (1e-3, 50, 1e5, 99401.240922855647, 1e-3),
        # Tuple 13: (1e-3, 100, 1e5, 99143.465191656527, np.nan)
        (1e-3, 100, 1e5, 99143.465191656527, np.nan),
        # Tuple 14: (0.5, 0, 1e5, 4074.1112442197941, 1e-11)
        (0.5, 0, 1e5, 4074.1112442197941, 1e-11),
        # Tuple 15: (0.5, 0, 1e7, 87724.552120038896, 1e-11)
        (0.5, 0, 1e7, 87724.552120038896, 1e-11),
        # Tuple 16: (0.5, 100, 1e5, 3350.3928746306163, np.nan)
        (0.5, 100, 1e5, 3350.3928746306163, np.nan),
        # Tuple 17: (0.5, 100, 1e7, 86696.109975301719, 1e-11)
        (0.5, 100, 1e7, 86696.109975301719, 1e-11),
        # Tuple 18: (1, 0, 1e5, 634.06765787997266, 1e-11)
        (1, 0, 1e5, 634.06765787997266, 1e-11),
        # Tuple 19: (1, 0, 1e8, 20003.339639312035, 1e-11)
        (1, 0, 1e8, 20003.339639312035, 1e-11),
        # Tuple 20: (1.5, 0, 1e5, 197.01777556071194, 1e-11)
        (1.5, 0, 1e5, 197.01777556071194, 1e-11),
        # Tuple 21: (1.5, 0, 1e8, 3108.987414395706, 1e-11)
        (1.5, 0, 1e8, 3108.987414395706, 1e-11),
        # Tuple 22: (1.5, 100, 1e8, 2354.8915946283275, np.nan)
        (1.5, 100, 1e8, 2354.8915946283275, np.nan),
        # Tuple 23: (5, 0, 1e5, 9.8980480013203547, 1e-11)
        (5, 0, 1e5, 9.8980480013203547, 1e-11),
        # Tuple 24: (5, 0, 1e8, 33.642337258687465, 1e-11)
        (5, 0, 1e8, 33.642337258687465, 1e-11),
        # Tuple 25: (5, 0, 1e12, 157.53704288117429, 1e-11)
        (5, 0, 1e12, 157.53704288117429, 1e-11),
        # Tuple 26: (5, 100, 1e5, -359.13419630792148, 1e-11)
        (5, 100, 1e5, -359.13419630792148, 1e-11),
        # Tuple 27: (5, 100, 1e12, -337.07722086995229, 1e-4)
        (5, 100, 1e12, -337.07722086995229, 1e-4),
        # Tuple 28: (5, 100, 1e20, 2588.2471229986845, 2e-6)
        (5, 100, 1e20, 2588.2471229986845, 2e-6),
        # Tuple 29: (100, 0, 1e5, -347.62127990460517, 1e-11)
        (100, 0, 1e5, -347.62127990460517, 1e-11),
        # Tuple 30: (100, 0, 1e20, -313.08250350969449, 1e-11)
        (100, 0, 1e20, -313.08250350969449, 1e-11),
        # Tuple 31: (100, 100, 1e5, -359.1342053695754, 1e-11)
        (100, 100, 1e5, -359.1342053695754, 1e-11),
        # Tuple 32: (100, 100, 1e20, -359.1342053695754, 1e-11)
        (100, 100, 1e20, -359.1342053695754, 1e-11),
    ]
)
# 定义一个名为 test_log_wright_bessel 的函数，用于测试 log_wright_bessel 函数，特别是针对大的 x 值进行测试
def test_log_wright_bessel(a, b, x, phi, accuracy):
    """Test for log_wright_bessel, in particular for large x."""
    # 如果 accuracy 是 NaN，则断言 log_wright_bessel(a, b, x) 也是 NaN
    if np.isnan(accuracy):
        assert np.isnan(log_wright_bessel(a, b, x))
    # 否则，使用 assert_allclose 函数断言 log_wright_bessel(a, b, x) 等于 phi，允许的相对误差为 accuracy
    else:
        assert_allclose(log_wright_bessel(a, b, x), phi, rtol=accuracy)
```