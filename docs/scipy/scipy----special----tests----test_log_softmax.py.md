# `D:\src\scipysrc\scipy\scipy\special\tests\test_log_softmax.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
from numpy.testing import assert_allclose  # 导入NumPy的测试工具assert_allclose用于检查数组是否接近

import pytest  # 导入pytest库，用于编写和运行测试用例

import scipy.special as sc  # 导入SciPy库中的special模块，用于数学特殊函数的计算


@pytest.mark.parametrize('x, expected', [
    (np.array([1000, 1]), np.array([0, -999])),  # 参数化测试用例：输入为数组[1000, 1]，期望输出为数组[0, -999]

    # 使用mpmath计算期望值（mpmath.mp.dps = 200），然后转换为浮点数。
    (np.arange(4), np.array([-3.4401896985611953,
                             -2.4401896985611953,
                             -1.4401896985611953,
                             -0.44018969856119533]))
])
def test_log_softmax(x, expected):
    assert_allclose(sc.log_softmax(x), expected, rtol=1e-13)  # 使用assert_allclose检查log_softmax函数的输出是否接近期望值


@pytest.fixture
def log_softmax_x():
    x = np.arange(4)  # 设置fixture log_softmax_x，返回一个长度为4的数组
    return x


@pytest.fixture
def log_softmax_expected():
    # 使用mpmath计算期望值（mpmath.mp.dps = 200），然后转换为浮点数。
    expected = np.array([-3.4401896985611953,
                         -2.4401896985611953,
                         -1.4401896985611953,
                         -0.44018969856119533])
    return expected  # 设置fixture log_softmax_expected，返回预期的结果数组


def test_log_softmax_translation(log_softmax_x, log_softmax_expected):
    # 平移性质测试：如果所有值增加相同的量，softmax结果不变。
    x = log_softmax_x + 100
    expected = log_softmax_expected
    assert_allclose(sc.log_softmax(x), expected, rtol=1e-13)  # 使用assert_allclose检查log_softmax函数的输出是否接近期望值


def test_log_softmax_noneaxis(log_softmax_x, log_softmax_expected):
    # 当axis=None时，softmax作用于整个数组，并保持形状不变。
    x = log_softmax_x.reshape(2, 2)
    expected = log_softmax_expected.reshape(2, 2)
    assert_allclose(sc.log_softmax(x), expected, rtol=1e-13)  # 使用assert_allclose检查log_softmax函数的输出是否接近期望值


@pytest.mark.parametrize('axis_2d, expected_2d', [
    (0, np.log(0.5) * np.ones((2, 2))),  # 参数化测试用例：axis=0时的预期结果为对数0.5的2x2数组
    (1, np.array([[0, -999], [0, -999]]))  # 参数化测试用例：axis=1时的预期结果为指定的2x2数组
])
def test_axes(axis_2d, expected_2d):
    assert_allclose(
        sc.log_softmax([[1000, 1], [1000, 1]], axis=axis_2d),  # 使用assert_allclose检查log_softmax函数的输出是否接近预期结果
        expected_2d,
        rtol=1e-13,
    )


@pytest.fixture
def log_softmax_2d_x():
    x = np.arange(8).reshape(2, 4)  # 设置fixture log_softmax_2d_x，返回一个2x4的数组
    return x


@pytest.fixture
def log_softmax_2d_expected():
    # 使用mpmath计算期望值（mpmath.mp.dps = 200），然后转换为浮点数。
    expected = np.array([[-3.4401896985611953,
                         -2.4401896985611953,
                         -1.4401896985611953,
                         -0.44018969856119533],
                        [-3.4401896985611953,
                         -2.4401896985611953,
                         -1.4401896985611953,
                         -0.44018969856119533]])
    return expected  # 设置fixture log_softmax_2d_expected，返回预期的结果数组


def test_log_softmax_2d_axis1(log_softmax_2d_x, log_softmax_2d_expected):
    x = log_softmax_2d_x
    expected = log_softmax_2d_expected
    assert_allclose(sc.log_softmax(x, axis=1), expected, rtol=1e-13)  # 使用assert_allclose检查log_softmax函数的输出是否接近期望值


def test_log_softmax_2d_axis0(log_softmax_2d_x, log_softmax_2d_expected):
    x = log_softmax_2d_x.T
    expected = log_softmax_2d_expected.T
    # 使用 assert_allclose 函数对输入张量 x 沿指定轴进行 log_softmax 操作，并与预期结果 expected 进行比较
    assert_allclose(sc.log_softmax(x, axis=0), expected, rtol=1e-13)
# 定义一个测试函数，用于测试三维输入的对数softmax计算
def test_log_softmax_3d(log_softmax_2d_x, log_softmax_2d_expected):
    # 将二维输入重新形状为三维，每个维度大小为2
    x_3d = log_softmax_2d_x.reshape(2, 2, 2)
    # 将预期的二维输出重新形状为三维，每个维度大小为2
    expected_3d = log_softmax_2d_expected.reshape(2, 2, 2)
    # 使用 log_softmax 函数计算在第1和第2维上的对数softmax，并断言其与预期的输出相近
    assert_allclose(sc.log_softmax(x_3d, axis=(1, 2)), expected_3d, rtol=1e-13)


# 定义一个测试函数，用于测试标量输入的对数softmax计算
def test_log_softmax_scalar():
    # 断言对数softmax函数应用于标量输入1.0时的输出应接近于0.0
    assert_allclose(sc.log_softmax(1.0), 0.0, rtol=1e-13)
```