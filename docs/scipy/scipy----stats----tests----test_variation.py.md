# `D:\src\scipysrc\scipy\scipy\stats\tests\test_variation.py`

```
# 导入所需的库和模块
import math  # 导入数学库
import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架
from numpy.testing import suppress_warnings  # 导入用于抑制警告的函数

from scipy.stats import variation  # 导入variation函数
from scipy._lib._util import AxisError  # 导入AxisError异常类
from scipy.conftest import array_api_compatible  # 导入array_api_compatible标记
from scipy._lib._array_api import xp_assert_equal, xp_assert_close, is_numpy  # 导入相关函数和标记
from scipy.stats._axis_nan_policy import (too_small_nd_omit, too_small_nd_not_omit,
                                          SmallSampleWarning)  # 导入NaN处理相关模块和警告类

pytestmark = [array_api_compatible, pytest.mark.usefixtures("skip_xp_backends")]
skip_xp_backends = pytest.mark.skip_xp_backends  # 定义skip_xp_backends标记

class TestVariation:
    """
    Test class for scipy.stats.variation
    """

    def test_ddof(self, xp):
        x = xp.arange(9.0)
        xp_assert_close(variation(x, ddof=1), xp.asarray(math.sqrt(60/8)/4))  # 测试variation函数的ddof参数

    @pytest.mark.parametrize('sgn', [1, -1])
    def test_sign(self, sgn, xp):
        x = xp.asarray([1., 2., 3., 4., 5.])
        v = variation(sgn*x)  # 测试带有符号的数组对variation函数的影响
        expected = xp.asarray(sgn*math.sqrt(2)/3)
        xp_assert_close(v, expected, rtol=1e-10)

    def test_scalar(self, xp):
        # 测试标量输入时的variation函数的行为
        xp_assert_equal(variation(4.0), 0.0)

    @pytest.mark.parametrize('nan_policy, expected',
                             [('propagate', np.nan),
                              ('omit', np.sqrt(20/3)/4)])
    @skip_xp_backends(np_only=True,
                      reasons=['`nan_policy` only supports NumPy backend'])
    def test_variation_nan(self, nan_policy, expected, xp):
        x = xp.arange(10.)
        x[9] = xp.nan
        xp_assert_close(variation(x, nan_policy=nan_policy), expected)  # 测试NaN策略对variation函数的影响

    @skip_xp_backends(np_only=True,
                      reasons=['`nan_policy` only supports NumPy backend'])
    def test_nan_policy_raise(self, xp):
        x = xp.asarray([1.0, 2.0, xp.nan, 3.0])
        with pytest.raises(ValueError, match='input contains nan'):
            variation(x, nan_policy='raise')  # 测试'raise' NaN策略时的异常处理

    @skip_xp_backends(np_only=True,
                      reasons=['`nan_policy` only supports NumPy backend'])
    def test_bad_nan_policy(self, xp):
        with pytest.raises(ValueError, match='must be one of'):
            variation([1, 2, 3], nan_policy='foobar')  # 测试不支持的NaN策略时的异常处理

    @skip_xp_backends(np_only=True,
                      reasons=['`keepdims` only supports NumPy backend'])
    def test_keepdims(self, xp):
        x = xp.reshape(xp.arange(10), (2, 5))
        y = variation(x, axis=1, keepdims=True)  # 测试keepdims参数对variation函数的影响
        expected = np.array([[np.sqrt(2)/2],
                             [np.sqrt(2)/7]])
        xp_assert_close(y, expected)

    @skip_xp_backends(np_only=True,
                      reasons=['`keepdims` only supports NumPy backend'])
    @pytest.mark.parametrize('axis, expected',
                             [(0, np.empty((1, 0))),
                              (1, np.full((5, 1), fill_value=np.nan))])
    def test_keepdims_size0(self, axis, expected, xp):
        # 定义一个测试函数，测试在给定轴向、期望结果和执行环境下的变异函数（variation）的行为
        x = xp.zeros((5, 0))  # 创建一个 5x0 的全零数组
        if axis == 1:
            with pytest.warns(SmallSampleWarning, match=too_small_nd_not_omit):
                # 如果 axis 为 1，则期望触发 SmallSampleWarning 警告，并匹配 too_small_nd_not_omit
                y = variation(x, axis=axis, keepdims=True)
        else:
            # 否则，以 keepdims=True 的方式调用变异函数（variation）
            y = variation(x, axis=axis, keepdims=True)
        xp_assert_equal(y, expected)  # 断言变异函数的结果与期望结果相等

    @skip_xp_backends(np_only=True,
                      reasons=['`keepdims` only supports NumPy backend'])
    @pytest.mark.parametrize('incr, expected_fill', [(0, np.inf), (1, np.nan)])
    def test_keepdims_and_ddof_eq_len_plus_incr(self, incr, expected_fill, xp):
        # 测试在给定增量和期望填充值下，以 keepdims=True 和指定自由度 ddof 的方式调用变异函数（variation）
        x = xp.asarray([[1, 1, 2, 2], [1, 2, 3, 3]])  # 创建一个二维数组
        y = variation(x, axis=1, ddof=x.shape[1] + incr, keepdims=True)
        xp_assert_equal(y, xp.full((2, 1), fill_value=expected_fill))  # 断言变异函数的结果与期望结果相等

    @skip_xp_backends(np_only=True,
                      reasons=['`nan_policy` only supports NumPy backend'])
    def test_propagate_nan(self, xp):
        # 检查带有和不带有 NaN 值的输入结果的形状是否相同
        # 参见 GitHub 问题 gh-5817
        a = xp.reshape(xp.arange(8, dtype=float), (2, -1))  # 创建一个 2x4 的浮点型数组
        a[1, 0] = xp.nan  # 将特定位置设置为 NaN
        v = variation(a, axis=1, nan_policy="propagate")  # 使用 nan_policy="propagate" 调用变异函数
        xp_assert_close(v, [math.sqrt(5/4)/1.5, xp.nan], atol=1e-15)  # 断言变异函数的结果与期望结果相等，允许的绝对误差为 1e-15

    @skip_xp_backends(np_only=True, reasons=['Python list input uses NumPy backend'])
    def test_axis_none(self, xp):
        # 检查当 axis 参数为 None 时，变异函数（variation）对扁平化输入的计算结果
        y = variation([[0, 1], [2, 3]], axis=None)  # 对输入列表进行变异计算
        xp_assert_close(y, math.sqrt(5/4)/1.5)  # 断言变异函数的结果与期望结果相等

    def test_bad_axis(self, xp):
        # 检查当给定无效的 axis 参数时，是否引发 np.exceptions.AxisError 异常
        x = xp.asarray([[1, 2, 3], [4, 5, 6]])  # 创建一个二维数组
        with pytest.raises((AxisError, IndexError)):
            variation(x, axis=10)  # 尝试使用无效的 axis 参数调用变异函数，期望引发异常

    def test_mean_zero(self, xp):
        # 检查对于均值为零但不全为零的序列，变异函数（variation）返回无穷大（inf）的情况
        x = xp.asarray([10., -3., 1., -4., -4.])  # 创建一个一维数组
        y = variation(x)  # 调用变异函数计算结果
        xp_assert_equal(y, xp.asarray(xp.inf))  # 断言变异函数的结果与期望结果相等

        x2 = xp.stack([x, -10.*x])  # 创建一个包含 x 和 -10*x 的数组
        y2 = variation(x2, axis=1)  # 对第二轴进行变异计算
        xp_assert_equal(y2, xp.asarray([xp.inf, xp.inf]))  # 断言变异函数的结果与期望结果相等

    @pytest.mark.parametrize('x', [[0.]*5, [1, 2, np.inf, 9]])
    def test_return_nan(self, x, xp):
        x = xp.asarray(x)  # 将输入列表转换为数组
        # 测试变异函数（variation）返回 NaN 的一些情况
        y = variation(x)  # 调用变异函数计算结果
        xp_assert_equal(y, xp.asarray(xp.nan, dtype=x.dtype))  # 断言变异函数的结果与期望结果相等，且类型一致

    @pytest.mark.parametrize('axis, expected',
                             [(0, []), (1, [np.nan]*3), (None, np.nan)])
    def test_2d_size_zero_with_axis(self, axis, expected, xp):
        # 创建一个空的二维数组 x，行数为 3，列数为 0
        x = xp.empty((3, 0))
        # 用 suppress_warnings 上下文管理器，捕获特定警告
        with suppress_warnings() as sup:
            # 在 torch 中过滤特定类型的用户警告 "std*"
            sup.filter(UserWarning, "std*")
            # 如果 axis 不等于 0
            if axis != 0:
                # 如果当前使用的是 NumPy
                if is_numpy(xp):
                    # 使用 pytest.warns 检测 SmallSampleWarning 警告，匹配给定的信息
                    with pytest.warns(SmallSampleWarning, match="See documentation..."):
                        # 对 x 沿指定轴计算变异，得到结果 y
                        y = variation(x, axis=axis)
                else:
                    # 对 x 沿指定轴计算变异，得到结果 y
                    y = variation(x, axis=axis)
            else:
                # 对 x 沿指定轴计算变异，得到结果 y
                y = variation(x, axis=axis)
        # 使用 xp_assert_equal 断言 y 与期望值 expected 相等
        xp_assert_equal(y, xp.asarray(expected))

    def test_neg_inf(self, xp):
        # 边界情况，产生 -inf 的例子：ddof 等于非 NaN 值的数量，数值不是常数，且均值为负数
        x1 = xp.asarray([-3., -5.])
        # 使用 xp_assert_equal 断言对 x1 计算变异，期望结果为 -inf
        xp_assert_equal(variation(x1, ddof=2), xp.asarray(-xp.inf))

    @skip_xp_backends(np_only=True,
                      reasons=['`nan_policy` only supports NumPy backend'])
    def test_neg_inf_nan(self, xp):
        # 创建一个包含 NaN 值的二维数组 x2
        x2 = xp.asarray([[xp.nan, 1, -10, xp.nan],
                         [-20, -3, xp.nan, xp.nan]])
        # 使用 xp_assert_equal 断言对 x2 沿指定轴计算变异，期望结果为包含 -inf 的数组
        xp_assert_equal(variation(x2, axis=1, ddof=2, nan_policy='omit'),
                        [-xp.inf, -xp.inf])

    @skip_xp_backends(np_only=True,
                      reasons=['`nan_policy` only supports NumPy backend'])
    @pytest.mark.parametrize("nan_policy", ['propagate', 'omit'])
    def test_combined_edge_cases(self, nan_policy, xp):
        # 创建一个二维数组 x，包含特定数值和 NaN 值
        x = xp.array([[0, 10, xp.nan, 1],
                      [0, -5, xp.nan, 2],
                      [0, -5, xp.nan, 3]])
        # 根据 nan_policy 参数选择不同的处理方式
        if nan_policy == 'omit':
            # 在 pytest.warns 上下文中捕获 SmallSampleWarning 警告
            with pytest.warns(SmallSampleWarning, match=too_small_nd_omit):
                # 对 x 沿指定轴计算变异，得到结果 y
                y = variation(x, axis=0, nan_policy=nan_policy)
        else:
            # 对 x 沿指定轴计算变异，得到结果 y
            y = variation(x, axis=0, nan_policy=nan_policy)
        # 使用 xp_assert_close 断言 y 与期望结果匹配
        xp_assert_close(y, [xp.nan, xp.inf, xp.nan, math.sqrt(2/3)/2])

    @skip_xp_backends(np_only=True,
                      reasons=['`nan_policy` only supports NumPy backend'])
    @pytest.mark.parametrize(
        'ddof, expected',
        [(0, [np.sqrt(1/6), np.sqrt(5/8), np.inf, 0, np.nan, 0.0, np.nan]),
         (1, [0.5, np.sqrt(5/6), np.inf, 0, np.nan, 0, np.nan]),
         (2, [np.sqrt(0.5), np.sqrt(5/4), np.inf, np.nan, np.nan, 0, np.nan])]
    )
    def test_more_nan_policy_omit_tests(self, ddof, expected, xp):
        # 定义测试函数，测试带有不同 NaN 策略的 variation 函数
        # ddof: 自由度
        # expected: 期望的输出结果
        # xp: 对象，可能是 NumPy 或类似的数组操作库的抽象

        # 定义 NaN 的值
        nan = xp.nan
        # 创建一个二维数组，包含 NaN 值，用于测试
        x = xp.asarray([[1.0, 2.0, nan, 3.0],
                        [0.0, 4.0, 3.0, 1.0],
                        [nan, -.5, 0.5, nan],
                        [nan, 9.0, 9.0, nan],
                        [nan, nan, nan, nan],
                        [3.0, 3.0, 3.0, 3.0],
                        [0.0, 0.0, 0.0, 0.0]])
        
        # 使用 pytest.warns 检查是否会触发 SmallSampleWarning，匹配字符串 'too_small_nd_omit'
        with pytest.warns(SmallSampleWarning, match=too_small_nd_omit):
            # 调用 variation 函数计算变异性，指定 axis=1 表示按行计算，nan_policy='omit' 表示忽略 NaN 值
            v = variation(x, axis=1, ddof=ddof, nan_policy='omit')
        
        # 使用 xp_assert_close 函数检查计算结果 v 是否接近于期望值 expected
        xp_assert_close(v, expected)

    @skip_xp_backends(np_only=True,
                      reasons=['`nan_policy` only supports NumPy backend'])
    def test_variation_ddof(self, xp):
        # 跳过非 NumPy 后端的测试，因为 `nan_policy` 只支持 NumPy 后端
        # 测试 variation 函数的自由度 (ddof) 参数
        # 这是 gh-13341 的回归测试
        
        # 创建 NumPy 数组 a 和 nan_a，用于测试
        a = xp.asarray([1., 2., 3., 4., 5.])
        nan_a = xp.asarray([1, 2, 3, xp.nan, 4, 5, xp.nan])
        
        # 计算数组 a 的变异性，ddof=1 表示 delta 自由度为 1
        y = variation(a, ddof=1)
        
        # 计算包含 NaN 的数组 nan_a 的变异性，使用 'omit' 策略忽略 NaN 值，ddof=1
        nan_y = variation(nan_a, nan_policy="omit", ddof=1)
        
        # 使用 xp_assert_close 检查计算结果 y 和 nan_y 是否接近 math.sqrt(5/2)/3
        xp_assert_close(y, math.sqrt(5/2)/3)
        
        # 断言 y 和 nan_y 的值相等
        assert y == nan_y
```