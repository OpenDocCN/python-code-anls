# `D:\src\scipysrc\scipy\scipy\special\tests\test_ndtri_exp.py`

```
import pytest  # 导入 pytest 测试框架
import numpy as np  # 导入 NumPy 库并简写为 np
from numpy.testing import assert_equal, assert_allclose  # 导入 NumPy 测试工具函数
from scipy.special import log_ndtr, ndtri_exp  # 导入 SciPy 库中的特殊函数
from scipy.special._testutils import assert_func_equal  # 导入 SciPy 的测试工具函数 assert_func_equal


def log_ndtr_ndtri_exp(y):
    return log_ndtr(ndtri_exp(y))  # 组合调用 log_ndtr 和 ndtri_exp 函数


@pytest.fixture(scope="class")  # 定义一个 pytest 的测试夹具，作用域为 class 级别
def uniform_random_points():
    random_state = np.random.RandomState(1234)  # 使用种子 1234 创建随机状态对象
    points = random_state.random_sample(1000)  # 生成 1000 个均匀分布的随机数
    return points  # 返回生成的随机数作为测试数据


class TestNdtriExp:
    """Tests that ndtri_exp is sufficiently close to an inverse of log_ndtr.

    We have separate tests for the five intervals (-inf, -10),
    [-10, -2), [-2, -0.14542), [-0.14542, -1e-6), and [-1e-6, 0).
    ndtri_exp(y) is computed in three different ways depending on if y
    is in (-inf, -2), [-2, log(1 - exp(-2))], or [log(1 - exp(-2), 0).
    Each of these intervals is given its own test with two additional tests
    for handling very small values and values very close to zero.
    """

    @pytest.mark.parametrize(
        "test_input", [-1e1, -1e2, -1e10, -1e20, -np.finfo(float).max]
    )
    def test_very_small_arg(self, test_input, uniform_random_points):
        scale = test_input  # 将 test_input 参数值赋给 scale
        points = scale * (0.5 * uniform_random_points + 0.5)  # 使用 scale 对随机点进行线性变换
        assert_func_equal(
            log_ndtr_ndtri_exp,  # 使用 assert_func_equal 函数进行测试
            lambda y: y, points,  # 测试 log_ndtr_ndtri_exp 函数是否与 lambda 表达式匹配
            rtol=1e-14,  # 相对误差的容忍度设为 1e-14
            nan_ok=True  # 允许处理 NaN 值
        )

    @pytest.mark.parametrize(
        "interval,expected_rtol",
        [
            ((-10, -2), 1e-14),
            ((-2, -0.14542), 1e-12),
            ((-0.14542, -1e-6), 1e-10),
            ((-1e-6, 0), 1e-6),
        ],
    )
    def test_in_interval(self, interval, expected_rtol, uniform_random_points):
        left, right = interval  # 将区间解包为 left 和 right
        points = (right - left) * uniform_random_points + left  # 生成指定区间内的随机点
        assert_func_equal(
            log_ndtr_ndtri_exp,  # 使用 assert_func_equal 函数进行测试
            lambda y: y, points,  # 测试 log_ndtr_ndtri_exp 函数是否与 lambda 表达式匹配
            rtol=expected_rtol,  # 相对误差的容忍度设为 expected_rtol
            nan_ok=True  # 允许处理 NaN 值
        )
    # 定义一个测试函数，用于测试在极端情况下的函数行为
    def test_extreme(self):
        # bigneg 是接近最大负双精度值的值，但不是最大的。以下是原因：
        # 回环计算：
        #    y = ndtri_exp(bigneg)
        #    bigneg2 = log_ndtr(y)
        # 当 bigneg 是一个非常大的负值时，在无限精度下，bigneg2 应该等于 bigneg。
        # 当 bigneg 足够大时，y 实际上等于 -sqrt(2)*sqrt(-bigneg)，而 log_ndtr(y) 实际上等于 -(y/sqrt(2))**2。
        # 如果我们使用 bigneg = np.finfo(float).min，那么根据构造，理论值是可以用 64 位浮点数表示的最大负有限值。
        # 这意味着计算过程中微小的变化可能导致返回值为 -inf。
        # （例如，将 1/sqrt(2) 的常数表示从 0.7071067811865475（由 1/np.sqrt(2) 返回的值）改为
        # 0.7071067811865476（这是 1/sqrt(2) 的最精确的 64 位浮点表示），导致从 np.finfo(float).min 开始的回环计算返回 -inf。
        # 因此，我们将 bigneg 的值移动几个 ULPs 向 0 方向，以避免这种敏感性。
        # 使用 reduce 方法四次应用 nextafter 函数。
        bigneg = np.nextafter.reduce([np.finfo(float).min, 0, 0, 0, 0])
        
        # tinyneg 是大约 -2.225e-308。
        tinyneg = -np.finfo(float).tiny
        
        # 创建一个包含 tinyneg 和 bigneg 的 NumPy 数组
        x = np.array([tinyneg, bigneg])
        
        # 调用 log_ndtr_ndtri_exp 函数，计算结果
        result = log_ndtr_ndtri_exp(x)
        
        # 使用 assert_allclose 断言函数的返回结果与期望值 x 在相对误差 1e-12 内的接近程度
        assert_allclose(result, x, rtol=1e-12)

    # 定义一个测试函数，用于测试渐近情况下的函数行为
    def test_asymptotes(self):
        # 使用 assert_equal 断言 ndtri_exp([-np.inf, 0.0]) 的返回值与预期值 [-np.inf, np.inf] 相等
        assert_equal(ndtri_exp([-np.inf, 0.0]), [-np.inf, np.inf])

    # 定义一个测试函数，用于测试超出定义域情况下的函数行为
    def test_outside_domain(self):
        # 使用 assert 断言调用 ndtri_exp(1.0) 返回 NaN
        assert np.isnan(ndtri_exp(1.0))
```