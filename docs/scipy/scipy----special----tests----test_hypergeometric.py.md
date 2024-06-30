# `D:\src\scipysrc\scipy\scipy\special\tests\test_hypergeometric.py`

```
# 导入 pytest 库，用于编写和运行测试用例
import pytest
# 导入 numpy 库，并将其命名为 np，用于数值计算
import numpy as np
# 导入 numpy.testing 模块中的 assert_allclose 和 assert_equal 函数，用于测试数值计算结果的近似相等和精确相等
from numpy.testing import assert_allclose, assert_equal
# 导入 scipy.special 库，并将其命名为 sc，用于调用特殊函数
import scipy.special as sc


# 定义 TestHyperu 类，用于测试 scipy.special 中的 hyperu 函数
class TestHyperu:

    # 定义 test_negative_x 方法，测试当 x 为负数时的情况
    def test_negative_x(self):
        # 生成 a, b, x 的网格点组合，其中 a 和 b 取值为 [-1, -0.5, 0, 0.5, 1]，x 取值为从 -100 到 -1 的等间距数组
        a, b, x = np.meshgrid(
            [-1, -0.5, 0, 0.5, 1],
            [-1, -0.5, 0, 0.5, 1],
            np.linspace(-100, -1, 10),
        )
        # 断言 hyperu 函数对所有的输入返回 NaN 值
        assert np.all(np.isnan(sc.hyperu(a, b, x)))

    # 定义 test_special_cases 方法，测试特殊情况下的 hyperu 函数调用
    def test_special_cases(self):
        # 断言 hyperu 函数在参数为 (0, 1, 1) 时返回值为 1.0
        assert sc.hyperu(0, 1, 1) == 1.0

    # 定义 test_nan_inputs 方法，测试 hyperu 函数对 NaN 输入的处理
    @pytest.mark.parametrize('a', [0.5, 1, np.nan])
    @pytest.mark.parametrize('b', [1, 2, np.nan])
    @pytest.mark.parametrize('x', [0.25, 3, np.nan])
    def test_nan_inputs(self, a, b, x):
        # 断言 hyperu 函数对任何参数为 NaN 的情况返回 NaN
        assert np.isnan(sc.hyperu(a, b, x)) == np.any(np.isnan([a, b, x]))


# 定义 TestHyp1f1 类，用于测试 scipy.special 中的 hyp1f1 函数
class TestHyp1f1:

    # 定义 test_nan_inputs 方法，测试 hyp1f1 函数对 NaN 输入的处理
    @pytest.mark.parametrize('a, b, x', [
        (np.nan, 1, 1),
        (1, np.nan, 1),
        (1, 1, np.nan)
    ])
    def test_nan_inputs(self, a, b, x):
        # 断言 hyp1f1 函数对任何参数为 NaN 的情况返回 NaN
        assert np.isnan(sc.hyp1f1(a, b, x))

    # 定义 test_poles 方法，测试 hyp1f1 函数在极点处的返回值
    def test_poles(self):
        # 断言 hyp1f1 函数在参数为 (1, [0, -1, -2, -3, -4], 0.5) 时返回值为正无穷
        assert_equal(sc.hyp1f1(1, [0, -1, -2, -3, -4], 0.5), np.inf)

    # 定义 test_special_cases 方法，测试特殊情况下的 hyp1f1 函数调用
    @pytest.mark.parametrize('a, b, x, result', [
        (-1, 1, 0.5, 0.5),
        (1, 1, 0.5, 1.6487212707001281468),
        (2, 1, 0.5, 2.4730819060501922203),
        (1, 2, 0.5, 1.2974425414002562937),
        (-10, 1, 0.5, -0.38937441413785204475)
    ])
    def test_special_cases(self, a, b, x, result):
        # 断言 hyp1f1 函数在特定参数下的计算结果与预期结果 result 近似相等，允许的绝对误差为 0，相对误差为 1e-15
        assert_allclose(sc.hyp1f1(a, b, x), result, atol=0, rtol=1e-15)

    # 定义 test_geometric_convergence 方法，测试 hyp1f1 函数在几何收敛区域的计算结果
    @pytest.mark.parametrize('a, b, x, result', [
        (1, 1, 0.44, 1.5527072185113360455),
        (-1, 1, 0.44, 0.55999999999999999778),
        (100, 100, 0.89, 2.4351296512898745592),
        (-100, 100, 0.89, 0.40739062490768104667),
        (1.5, 100, 59.99, 3.8073513625965598107),
        (-1.5, 100, 59.99, 0.25099240047125826943)
    ])
    def test_geometric_convergence(self, a, b, x, result):
        # 断言 hyp1f1 函数在几何收敛区域的计算结果与预期结果 result 近似相等，允许的绝对误差为 0，相对误差为 1e-15
        assert_allclose(sc.hyp1f1(a, b, x), result, atol=0, rtol=1e-15)

    # 定义 test_a_negative_integer 方法，测试当 a 为负整数时的 hyp1f1 函数调用
    @pytest.mark.parametrize('a, b, x, result', [
        (-1, 1, 1.5, -0.5),
        (-10, 1, 1.5, 0.41801777430943080357),
        (-25, 1, 1.5, 0.25114491646037839809),
        (-50, 1, 1.5, -0.25683643975194756115),
        (-80, 1, 1.5, -0.24554329325751503601),
        (-150, 1, 1.5, -0.173364795515420454496),
    ])
    def test_a_negative_integer(self, a, b, x, result):
        # 断言 hyp1f1 函数在负整数 a 的特定参数下的计算结果与预期结果 result 近似相等，允许的绝对误差为 0，相对误差为 2e-14
        assert_allclose(sc.hyp1f1(a, b, x), result, atol=0, rtol=2e-14)
    @pytest.mark.parametrize('a, b, x, expected', [
        (0.01, 150, -4, 0.99973683897677527773),        # Test case gh-3492
        (1, 5, 0.01, 1.0020033381011970966),            # Test case gh-3593
        (50, 100, 0.01, 1.0050126452421463411),         # Test case gh-3593
        (1, 0.3, -1e3, -7.011932249442947651455e-04),   # Test case gh-14149
        (1, 0.3, -1e4, -7.001190321418937164734e-05),   # Test case gh-14149
        (9, 8.5, -350, -5.224090831922378361082e-20),   # Test case gh-17120
        (9, 8.5, -355, -4.595407159813368193322e-20),   # Test case gh-17120
        (75, -123.5, 15, 3.425753920814889017493e+06),
    ])
    def test_assorted_cases(self, a, b, x, expected):
        # Assert that the computed value using sc.hyp1f1(a, b, x) matches expected with tolerance settings
        assert_allclose(sc.hyp1f1(a, b, x), expected, atol=0, rtol=1e-14)

    def test_a_neg_int_and_b_equal_x(self):
        # This test verifies behavior when a Boost wrapper calls hypergeometric_pFq instead of hypergeometric_1F1.
        # Once Boost issue https://github.com/boostorg/math/issues/833 is fixed, this case may be moved.
        # The expected value was computed with mpmath.hyp1f1(a, b, x).
        a = -10.0
        b = 2.5
        x = 2.5
        expected = 0.0365323664364104338721
        computed = sc.hyp1f1(a, b, x)
        assert_allclose(computed, expected, atol=0, rtol=1e-13)

    @pytest.mark.parametrize('a, b, x, desired', [
        (-1, -2, 2, 2),     # Test case for gh-11099
        (-1, -4, 10, 3.5),  # Test case for gh-11099
        (-2, -2, 1, 2.5)    # Test case for gh-11099
    ])
    def test_gh_11099(self, a, b, x, desired):
        # Assert that sc.hyp1f1(a, b, x) computes the desired result as expected
        assert sc.hyp1f1(a, b, x) == desired

    @pytest.mark.parametrize('a', [-3, -2])
    def test_x_zero_a_and_b_neg_ints_and_a_ge_b(self, a):
        # Assert that sc.hyp1f1(a, -3, 0) returns 1 for given values of 'a'
        assert sc.hyp1f1(a, -3, 0) == 1

    # In the following tests with complex z, the reference values
    # were computed with mpmath.hyp1f1(a, b, z), and verified with
    # Wolfram Alpha Hypergeometric1F1(a, b, z), except for the
    # case a=0.1, b=1, z=7-24j, where Wolfram Alpha reported
    # "Standard computation time exceeded".  That reference value
    # was confirmed in an online Matlab session, with the commands
    #
    #  > format long
    #  > hypergeom(0.1, 1, 7-24i)
    #  ans =
    #   -3.712349651834209 + 4.554636556672912i
    #
    @pytest.mark.parametrize(
        'a, b, z, ref',
        [(-0.25, 0.5, 1+2j, 1.1814553180903435-1.2792130661292984j),    # Test case
         (0.25, 0.5, 1+2j, 0.24636797405707597+1.293434354945675j),      # Test case
         (25, 1.5, -2j, -516.1771262822523+407.04142751922024j),        # Test case
         (12, -1.5, -10+20j, -5098507.422706547-1341962.8043508842j),   # Test case
         pytest.param(
             10, 250, 10-15j, 1.1985998416598884-0.8613474402403436j,   # Test case with expected failure
             marks=pytest.mark.xfail,
         ),
         pytest.param(
             0.1, 1, 7-24j, -3.712349651834209+4.554636556672913j,       # Test case with expected failure
             marks=pytest.mark.xfail,
         )
         ],
    )
    # 调用 SciPy 的 hyp1f1 函数计算超几何函数 1F1(a, b, z)，并将结果存入变量 h
    h = sc.hyp1f1(a, b, z)
    # 使用 assert_allclose 断言函数的输出 h 应与参考值 ref 接近，相对误差容忍度设为 4e-15
    assert_allclose(h, ref, rtol=4e-15)

# 下面的测试函数涉及到 SciPy 中 hyp1f1 函数对于 b 是非正整数的特殊情况的测试。
# 在某些情况下，SciPy 的处理方式与 Boost (1.81+), mpmath 和 Mathematica (通过 Wolfram Alpha 在线) 不一致。
# 如果 SciPy 更新以与这些库一致，这些测试将需要更新。

@pytest.mark.parametrize('b', [0, -1, -5])
def test_legacy_case1(self, b):
    # 测试 hyp1f1(0, n, x) 对于 n <= 0 的情况。
    # 这是一个历史上的特例情况。
    # Boost (版本大于 1.80), Mathematica (通过 Wolfram Alpha 在线) 和 mpmath 在这种情况下返回 1，但 SciPy 的 hyp1f1 返回 inf。
    assert_equal(sc.hyp1f1(0, b, [-1.5, 0, 1.5]), [np.inf, np.inf, np.inf])

def test_legacy_case2(self):
    # 这是一个历史上的特例情况。
    # 在 boost (1.81+), mpmath 和 Mathematica 等软件中，该值为 1。
    assert sc.hyp1f1(-4, -3, 0) == np.inf
```