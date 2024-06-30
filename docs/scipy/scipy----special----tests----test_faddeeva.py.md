# `D:\src\scipysrc\scipy\scipy\special\tests\test_faddeeva.py`

```
# 导入pytest库，用于编写和运行测试用例
import pytest

# 导入numpy库，并将其命名为np，用于数值计算
import numpy as np

# 从numpy.testing模块中导入assert_allclose函数，用于检查数值是否在指定的误差范围内相等
from numpy.testing import assert_allclose

# 导入scipy库中的special模块，并将其命名为sc，用于特殊数学函数的计算
import scipy.special as sc

# 从scipy.special._testutils模块中导入FuncData类，用于函数数据的测试
from scipy.special._testutils import FuncData

# 定义测试类TestVoigtProfile，用于测试Voigt分布的特性
class TestVoigtProfile:

    # 使用pytest.mark.parametrize装饰器定义参数化测试，测试输入包含NaN的情况
    @pytest.mark.parametrize('x, sigma, gamma', [
        (np.nan, 1, 1),
        (0, np.nan, 1),
        (0, 1, np.nan),
        (1, np.nan, 0),
        (np.nan, 1, 0),
        (1, 0, np.nan),
        (np.nan, 0, 1),
        (np.nan, 0, 0)
    ])
    # 定义测试函数test_nan，用于验证当输入包含NaN时，Voigt分布的返回值是否为NaN
    def test_nan(self, x, sigma, gamma):
        assert np.isnan(sc.voigt_profile(x, sigma, gamma))

    # 使用pytest.mark.parametrize装饰器定义参数化测试，测试输入为无穷大的情况
    @pytest.mark.parametrize('x, desired', [
        (-np.inf, 0),
        (np.inf, 0)
    ])
    # 定义测试函数test_inf，用于验证当输入为无穷大时，Voigt分布的返回值是否为0
    def test_inf(self, x, desired):
        assert sc.voigt_profile(x, 1, 1) == desired

    # 定义测试函数test_against_mathematica，用于验证Voigt分布的实现是否与Mathematica计算结果一致
    def test_against_mathematica(self):
        # 结果来自于Mathematica的计算结果：
        #
        # PDF[VoigtDistribution[gamma, sigma], x]
        #
        # 使用np.array定义包含数个点的数组，每个点包含x, sigma, gamma和期望值desired
        points = np.array([
            [-7.89, 45.06, 6.66, 0.0077921073660388806401],
            [-0.05, 7.98, 24.13, 0.012068223646769913478],
            [-13.98, 16.83, 42.37, 0.0062442236362132357833],
            [-12.66, 0.21, 6.32, 0.010052516161087379402],
            [11.34, 4.25, 21.96, 0.0113698923627278917805],
            [-11.56, 20.40, 30.53, 0.0076332760432097464987],
            [-9.17, 25.61, 8.32, 0.011646345779083005429],
            [16.59, 18.05, 2.50, 0.013637768837526809181],
            [9.11, 2.12, 39.33, 0.0076644040807277677585],
            [-43.33, 0.30, 45.68, 0.0036680463875330150996]
        ])
        # 创建FuncData对象，用于检查Voigt分布的实现是否与Mathematica一致
        FuncData(
            sc.voigt_profile,
            points,
            (0, 1, 2),
            3,
            atol=0,
            rtol=1e-15
        ).check()

    # 定义测试函数test_symmetry，用于验证Voigt分布在对称情况下的性质
    def test_symmetry(self):
        # 使用np.linspace生成等间隔的数字序列x
        x = np.linspace(0, 10, 20)
        # 使用assert_allclose检查Voigt分布在x和-x处的值是否在指定的误差范围内相等
        assert_allclose(
            sc.voigt_profile(x, 1, 1),
            sc.voigt_profile(-x, 1, 1),
            rtol=1e-15,
            atol=0
        )

    # 使用pytest.mark.parametrize装饰器定义参数化测试，测试特殊边界情况
    @pytest.mark.parametrize('x, sigma, gamma, desired', [
        (0, 0, 0, np.inf),
        (1, 0, 0, 0)
    ])
    # 定义测试函数test_corner_cases，用于验证Voigt分布在特殊边界情况下的返回值是否符合预期
    def test_corner_cases(self, x, sigma, gamma, desired):
        assert sc.voigt_profile(x, sigma, gamma) == desired

    # 使用pytest.mark.parametrize装饰器定义参数化测试，测试Voigt分布的连续性
    @pytest.mark.parametrize('sigma1, gamma1, sigma2, gamma2', [
        (0, 1, 1e-16, 1),
        (1, 0, 1, 1e-16),
        (0, 0, 1e-16, 1e-16)
    ])
    # 定义测试函数test_continuity，用于验证Voigt分布在不同参数下的连续性
    def test_continuity(self, sigma1, gamma1, sigma2, gamma2):
        # 使用np.linspace生成等间隔的数字序列x
        x = np.linspace(1, 10, 20)
        # 使用assert_allclose检查Voigt分布在不同参数下的值是否在指定的误差范围内相等
        assert_allclose(
            sc.voigt_profile(x, sigma1, gamma1),
            sc.voigt_profile(x, sigma2, gamma2),
            rtol=1e-16,
            atol=1e-16
        )
```