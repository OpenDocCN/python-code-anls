# `D:\src\scipysrc\scipy\scipy\constants\tests\test_constants.py`

```
import pytest
# 导入 pytest 模块，用于编写和运行测试用例

import scipy.constants as sc
# 导入 scipy.constants 模块，提供科学常数的支持

from scipy.conftest import array_api_compatible
# 从 scipy.conftest 模块导入 array_api_compatible，用于测试数组 API 的兼容性

from scipy._lib._array_api import xp_assert_equal, xp_assert_close
# 从 scipy._lib._array_api 模块导入 xp_assert_equal 和 xp_assert_close，用于比较数组是否相等和是否在接近范围内

from numpy.testing import assert_allclose
# 从 numpy.testing 模块导入 assert_allclose，用于测试两个数组是否在接近范围内相等


pytestmark = [array_api_compatible, pytest.mark.usefixtures("skip_xp_backends")]
# 定义 pytestmark 变量，包含 array_api_compatible 标记和 usefixtures("skip_xp_backends") 标记

skip_xp_backends = pytest.mark.skip_xp_backends
# 定义 skip_xp_backends 变量，用于跳过数组 API 后端的相关测试


class TestConvertTemperature:
# 定义一个测试类 TestConvertTemperature，用于测试温度转换相关功能
    # 定义测试方法，用于验证温度转换函数的正确性，使用了 xp 作为测试框架的扩展库
    def test_convert_temperature(self, xp):
        # 测试将华氏度 32 转换为摄氏度，期望结果是 0.0
        xp_assert_equal(sc.convert_temperature(xp.asarray(32.), 'f', 'Celsius'),
                        xp.asarray(0.0))
        # 测试将摄氏度 [0.0, 0.0] 转换为开尔文，期望结果是 [273.15, 273.15]
        xp_assert_equal(sc.convert_temperature(xp.asarray([0., 0.]),
                                               'celsius', 'Kelvin'),
                        xp.asarray([273.15, 273.15]))
        # 测试将开尔文 [0.0, 0.0] 转换为摄氏度，期望结果是 [-273.15, -273.15]
        xp_assert_equal(sc.convert_temperature(xp.asarray([0., 0.]), 'kelvin', 'c'),
                        xp.asarray([-273.15, -273.15]))
        # 测试将华氏度 [32.0, 32.0] 转换为开尔文，期望结果是 [273.15, 273.15]
        xp_assert_equal(sc.convert_temperature(xp.asarray([32., 32.]), 'f', 'k'),
                        xp.asarray([273.15, 273.15]))
        # 测试将开尔文 [273.15, 273.15] 转换为华氏度，期望结果是 [32.0, 32.0]
        xp_assert_equal(sc.convert_temperature(xp.asarray([273.15, 273.15]),
                                               'kelvin', 'F'),
                        xp.asarray([32., 32.]))
        # 测试将摄氏度 [0.0, 0.0] 转换为华氏度，期望结果是 [32.0, 32.0]
        xp_assert_equal(sc.convert_temperature(xp.asarray([0., 0.]), 'C', 'fahrenheit'),
                        xp.asarray([32., 32.]))
        # 测试将摄氏度 [0.0, 0.0] 转换为兰氏度，期望结果是 [491.67, 491.67]
        xp_assert_close(sc.convert_temperature(xp.asarray([0., 0.], dtype=xp.float64),
                                               'c', 'r'),
                        xp.asarray([491.67, 491.67], dtype=xp.float64),
                        rtol=0., atol=1e-13)
        # 测试将兰氏度 [491.67, 491.67] 转换为摄氏度，期望结果是 [0.0, 0.0]
        xp_assert_close(sc.convert_temperature(xp.asarray([491.67, 491.67],
                                                        dtype=xp.float64),
                                               'Rankine', 'C'),
                        xp.asarray([0., 0.], dtype=xp.float64), rtol=0., atol=1e-13)
        # 测试将兰氏度 [491.67, 491.67] 转换为华氏度，期望结果是 [32.0, 32.0]
        xp_assert_close(sc.convert_temperature(xp.asarray([491.67, 491.67],
                                                        dtype=xp.float64),
                                               'r', 'F'),
                        xp.asarray([32., 32.], dtype=xp.float64), rtol=0., atol=1e-13)
        # 测试将华氏度 [32.0, 32.0] 转换为兰氏度，期望结果是 [491.67, 491.67]
        xp_assert_close(sc.convert_temperature(xp.asarray([32., 32.], dtype=xp.float64),
                                               'fahrenheit', 'R'),
                        xp.asarray([491.67, 491.67], dtype=xp.float64),
                        rtol=0., atol=1e-13)
        # 测试将开尔文 [273.15, 273.15] 转换为兰氏度，期望结果是 [491.67, 491.67]
        xp_assert_close(sc.convert_temperature(xp.asarray([273.15, 273.15],
                                                        dtype=xp.float64),
                                               'K', 'R'),
                        xp.asarray([491.67, 491.67], dtype=xp.float64),
                        rtol=0., atol=1e-13)
        # 测试将兰氏度 [491.67, 0.0] 转换为开尔文，期望结果是 [273.15, 0.0]
        xp_assert_close(sc.convert_temperature(xp.asarray([491.67, 0.],
                                                          dtype=xp.float64),
                                               'rankine', 'kelvin'),
                        xp.asarray([273.15, 0.], dtype=xp.float64), rtol=0., atol=1e-13)

    # 使用装饰器指定只在 NumPy 后端运行，并说明 Python 列表输入使用 NumPy 后端的原因
    @skip_xp_backends(np_only=True, reasons=['Python list input uses NumPy backend'])
    # 定义测试方法，用于测试温度转换函数对数组输入的处理
    def test_convert_temperature_array_like(self):
        # 使用 assert_allclose 断言函数验证温度转换后的结果是否与期望值接近
        assert_allclose(sc.convert_temperature([491.67, 0.], 'rankine', 'kelvin'),
                        [273.15, 0.], rtol=0., atol=1e-13)


    # 使用装饰器 @skip_xp_backends 标记的测试方法，只在 NumPy 环境下执行
    @skip_xp_backends(np_only=True, reasons=['Python int input uses NumPy backend'])
    # 定义测试温度转换函数在特定输入情况下抛出错误的情况
    def test_convert_temperature_errors(self, xp):
        # 使用 pytest.raises 捕获 NotImplementedError 异常，验证当旧温度标度为 "cheddar" 时抛出异常
        with pytest.raises(NotImplementedError, match="old_scale="):
            sc.convert_temperature(1, old_scale="cheddar", new_scale="kelvin")
        # 使用 pytest.raises 捕获 NotImplementedError 异常，验证当新温度标度为 "brie" 时抛出异常
        with pytest.raises(NotImplementedError, match="new_scale="):
            sc.convert_temperature(1, old_scale="kelvin", new_scale="brie")
# 定义一个测试类 TestLambdaToNu，用于测试 lambda 到 nu 的转换功能
class TestLambdaToNu:
    
    # 测试 lambda 到 nu 的转换函数 test_lambda_to_nu
    def test_lambda_to_nu(self, xp):
        # 使用 xp_assert_equal 断言验证 lambda2nu 函数的输出是否等于预期的结果数组
        xp_assert_equal(sc.lambda2nu(xp.asarray([sc.speed_of_light, 1])),
                        xp.asarray([1, sc.speed_of_light]))


    # 装饰器：跳过不支持 NumPy 后端的测试，因为 Python 列表输入使用了 NumPy 后端
    @skip_xp_backends(np_only=True, reasons=['Python list input uses NumPy backend'])
    # 测试 lambda 到 nu 的数组类输入的转换函数 test_lambda_to_nu_array_like
    def test_lambda_to_nu_array_like(self, xp):
        # 使用 assert_allclose 验证 lambda2nu 函数对数组类输入的输出是否接近预期结果数组
        assert_allclose(sc.lambda2nu([sc.speed_of_light, 1]),
                        [1, sc.speed_of_light])


# 定义一个测试类 TestNuToLambda，用于测试 nu 到 lambda 的转换功能
class TestNuToLambda:
    
    # 测试 nu 到 lambda 的转换函数 test_nu_to_lambda
    def test_nu_to_lambda(self, xp):
        # 使用 xp_assert_equal 断言验证 nu2lambda 函数的输出是否等于预期的结果数组
        xp_assert_equal(sc.nu2lambda(xp.asarray([sc.speed_of_light, 1])),
                        xp.asarray([1, sc.speed_of_light]))

    # 装饰器：跳过不支持 NumPy 后端的测试，因为 Python 列表输入使用了 NumPy 后端
    # 测试 nu 到 lambda 的数组类输入的转换函数 test_nu_to_lambda_array_like
    @skip_xp_backends(np_only=True, reasons=['Python list input uses NumPy backend'])
    def test_nu_to_lambda_array_like(self, xp):
        # 使用 assert_allclose 验证 nu2lambda 函数对数组类输入的输出是否接近预期结果数组
        assert_allclose(sc.nu2lambda([sc.speed_of_light, 1]),
                        [1, sc.speed_of_light])
```