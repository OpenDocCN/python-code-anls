# `D:\src\scipysrc\scipy\scipy\special\tests\test_bdtr.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
import scipy.special as sc  # 导入 SciPy 的 special 模块，用于特殊数学函数
import pytest  # 导入 pytest 库，用于单元测试
from numpy.testing import assert_allclose, assert_array_equal, suppress_warnings  # 导入 NumPy 测试模块中的断言函数和警告抑制功能


class TestBdtr:
    def test(self):
        val = sc.bdtr(0, 1, 0.5)  # 计算二项分布累积分布函数的值
        assert_allclose(val, 0.5)  # 断言计算结果与期望值的近似性

    def test_sum_is_one(self):
        val = sc.bdtr([0, 1, 2], 2, 0.5)  # 计算多个点的二项分布累积分布函数值
        assert_array_equal(val, [0.25, 0.75, 1.0])  # 断言计算结果与期望值数组的一致性

    def test_rounding(self):
        double_val = sc.bdtr([0.1, 1.1, 2.1], 2, 0.5)  # 计算非整数点的二项分布累积分布函数值
        int_val = sc.bdtr([0, 1, 2], 2, 0.5)  # 计算整数点的二项分布累积分布函数值
        assert_array_equal(double_val, int_val)  # 断言计算结果的一致性

    @pytest.mark.parametrize('k, n, p', [
        (np.inf, 2, 0.5),
        (1.0, np.inf, 0.5),
        (1.0, 2, np.inf)
    ])
    def test_inf(self, k, n, p):
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)  # 过滤掉 DeprecationWarning 类型的警告
            val = sc.bdtr(k, n, p)  # 计算带有无限值参数的二项分布累积分布函数值
        assert np.isnan(val)  # 断言计算结果为 NaN

    def test_domain(self):
        val = sc.bdtr(-1.1, 1, 0.5)  # 计算超出定义域的二项分布累积分布函数值
        assert np.isnan(val)  # 断言计算结果为 NaN


class TestBdtrc:
    def test_value(self):
        val = sc.bdtrc(0, 1, 0.5)  # 计算补二项分布累积分布函数的值
        assert_allclose(val, 0.5)  # 断言计算结果与期望值的近似性

    def test_sum_is_one(self):
        val = sc.bdtrc([0, 1, 2], 2, 0.5)  # 计算多个点的补二项分布累积分布函数值
        assert_array_equal(val, [0.75, 0.25, 0.0])  # 断言计算结果与期望值数组的一致性

    def test_rounding(self):
        double_val = sc.bdtrc([0.1, 1.1, 2.1], 2, 0.5)  # 计算非整数点的补二项分布累积分布函数值
        int_val = sc.bdtrc([0, 1, 2], 2, 0.5)  # 计算整数点的补二项分布累积分布函数值
        assert_array_equal(double_val, int_val)  # 断言计算结果的一致性

    @pytest.mark.parametrize('k, n, p', [
        (np.inf, 2, 0.5),
        (1.0, np.inf, 0.5),
        (1.0, 2, np.inf)
    ])
    def test_inf(self, k, n, p):
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)  # 过滤掉 DeprecationWarning 类型的警告
            val = sc.bdtrc(k, n, p)  # 计算带有无限值参数的补二项分布累积分布函数值
        assert np.isnan(val)  # 断言计算结果为 NaN

    def test_domain(self):
        val = sc.bdtrc(-1.1, 1, 0.5)  # 计算超出定义域的补二项分布累积分布函数值
        val2 = sc.bdtrc(2.1, 1, 0.5)  # 计算超出定义域的补二项分布累积分布函数值
        assert np.isnan(val2)  # 断言计算结果为 NaN
        assert_allclose(val, 1.0)  # 断言计算结果与期望值的近似性

    def test_bdtr_bdtrc_sum_to_one(self):
        bdtr_vals = sc.bdtr([0, 1, 2], 2, 0.5)  # 计算二项分布累积分布函数值
        bdtrc_vals = sc.bdtrc([0, 1, 2], 2, 0.5)  # 计算补二项分布累积分布函数值
        vals = bdtr_vals + bdtrc_vals  # 计算二项分布累积分布函数值与补二项分布累积分布函数值之和
        assert_allclose(vals, [1.0, 1.0, 1.0])  # 断言计算结果与期望值数组的一致性


class TestBdtri:
    def test_value(self):
        val = sc.bdtri(0, 1, 0.5)  # 计算二项分布逆累积分布函数的值
        assert_allclose(val, 0.5)  # 断言计算结果与期望值的近似性

    def test_sum_is_one(self):
        val = sc.bdtri([0, 1], 2, 0.5)  # 计算多个点的二项分布逆累积分布函数值
        actual = np.asarray([1 - 1/np.sqrt(2), 1/np.sqrt(2)])  # 计算期望值数组
        assert_allclose(val, actual)  # 断言计算结果与期望值数组的近似性

    def test_rounding(self):
        double_val = sc.bdtri([0.1, 1.1], 2, 0.5)  # 计算非整数点的二项分布逆累积分布函数值
        int_val = sc.bdtri([0, 1], 2, 0.5)  # 计算整数点的二项分布逆累积分布函数值
        assert_allclose(double_val, int_val)  # 断言计算结果的近似性

    @pytest.mark.parametrize('k, n, p', [
        (np.inf, 2, 0.5),
        (1.0, np.inf, 0.5),
        (1.0, 2, np.inf)
    ])
    def test_inf(self, k, n, p):
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning)  # 过滤掉 DeprecationWarning 类型的警告
            val = sc.bdtri(k, n, p)  # 计算带有无限值参数的二项分布逆累积分布函数值
        assert np.isnan(val)  # 断言计算结果为 NaN

    @pytest.mark.parametrize('k, n, p', [
        (-1.1, 1, 0.5),
        (2.1, 1, 0.5)
    ])
    # 定义一个测试方法，用于验证 bdtri 函数返回值的正确性
    def test_domain(self, k, n, p):
        # 调用 bdtri 函数计算概率分布的下侧累积分布函数值
        val = sc.bdtri(k, n, p)
        # 使用断言检查返回值是否为 NaN
        assert np.isnan(val)

    # 定义另一个测试方法，用于测试 bdtr 和 bdtri 函数之间的互逆性
    def test_bdtr_bdtri_roundtrip(self):
        # 调用 bdtr 函数计算概率分布的下侧累积分布函数值列表
        bdtr_vals = sc.bdtr([0, 1, 2], 2, 0.5)
        # 使用 bdtri 函数计算 bdtr 函数返回值的逆函数值列表
        roundtrip_vals = sc.bdtri([0, 1, 2], 2, bdtr_vals)
        # 使用断言检查逆函数计算的结果是否接近期望值列表 [0.5, 0.5, NaN]
        assert_allclose(roundtrip_vals, [0.5, 0.5, np.nan])
```