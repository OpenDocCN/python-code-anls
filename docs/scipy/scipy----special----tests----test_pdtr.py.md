# `D:\src\scipysrc\scipy\scipy\special\tests\test_pdtr.py`

```
# 导入所需的库
import numpy as np  # 导入 NumPy 库，并使用别名 np
import scipy.special as sc  # 导入 SciPy 库中的特殊函数模块，并使用别名 sc
from numpy.testing import assert_almost_equal, assert_array_equal  # 从 NumPy 测试模块中导入两个断言函数

# 定义测试类 TestPdtr
class TestPdtr:
    
    # 定义测试函数 test，测试 sc.pdtr 函数的基本用法
    def test(self):
        val = sc.pdtr(0, 1)  # 调用 sc.pdtr 函数计算累积分布函数值
        assert_almost_equal(val, np.exp(-1))  # 使用 assert_almost_equal 断言函数验证结果近似等于 np.exp(-1)

    # 定义测试函数 test_m_zero，测试 sc.pdtr 函数对于多个参数为 0 的情况
    def test_m_zero(self):
        val = sc.pdtr([0, 1, 2], 0)  # 调用 sc.pdtr 函数计算多个参数为 0 时的累积分布函数值
        assert_array_equal(val, [1, 1, 1])  # 使用 assert_array_equal 断言函数验证结果数组与预期相等

    # 定义测试函数 test_rounding，测试 sc.pdtr 函数在浮点数和整数参数情况下的一致性
    def test_rounding(self):
        double_val = sc.pdtr([0.1, 1.1, 2.1], 1.0)  # 调用 sc.pdtr 函数计算浮点数参数时的累积分布函数值
        int_val = sc.pdtr([0, 1, 2], 1.0)  # 调用 sc.pdtr 函数计算整数参数时的累积分布函数值
        assert_array_equal(double_val, int_val)  # 使用 assert_array_equal 断言函数验证两个结果数组相等

    # 定义测试函数 test_inf，测试 sc.pdtr 函数在参数为无穷大时的行为
    def test_inf(self):
        val = sc.pdtr(np.inf, 1.0)  # 调用 sc.pdtr 函数计算参数为正无穷时的累积分布函数值
        assert_almost_equal(val, 1.0)  # 使用 assert_almost_equal 断言函数验证结果近似等于 1.0

    # 定义测试函数 test_domain，测试 sc.pdtr 函数在参数不在定义域内的情况
    def test_domain(self):
        val = sc.pdtr(-1.1, 1.0)  # 调用 sc.pdtr 函数计算参数为负值时的累积分布函数值
        assert np.isnan(val)  # 使用 np.isnan 函数验证结果是否为 NaN

# 定义测试类 TestPdtrc
class TestPdtrc:
    
    # 定义测试函数 test_value，测试 sc.pdtrc 函数的基本用法
    def test_value(self):
        val = sc.pdtrc(0, 1)  # 调用 sc.pdtrc 函数计算补充累积分布函数值
        assert_almost_equal(val, 1 - np.exp(-1))  # 使用 assert_almost_equal 断言函数验证结果近似等于 1 - np.exp(-1)

    # 定义测试函数 test_m_zero，测试 sc.pdtrc 函数对于多个参数为 0 的情况
    def test_m_zero(self):
        val = sc.pdtrc([0, 1, 2], 0.0)  # 调用 sc.pdtrc 函数计算多个参数为 0 时的补充累积分布函数值
        assert_array_equal(val, [0, 0, 0])  # 使用 assert_array_equal 断言函数验证结果数组与预期相等

    # 定义测试函数 test_rounding，测试 sc.pdtrc 函数在浮点数和整数参数情况下的一致性
    def test_rounding(self):
        double_val = sc.pdtrc([0.1, 1.1, 2.1], 1.0)  # 调用 sc.pdtrc 函数计算浮点数参数时的补充累积分布函数值
        int_val = sc.pdtrc([0, 1, 2], 1.0)  # 调用 sc.pdtrc 函数计算整数参数时的补充累积分布函数值
        assert_array_equal(double_val, int_val)  # 使用 assert_array_equal 断言函数验证两个结果数组相等

    # 定义测试函数 test_inf，测试 sc.pdtrc 函数在参数为无穷大时的行为
    def test_inf(self):
        val = sc.pdtrc(np.inf, 1.0)  # 调用 sc.pdtrc 函数计算参数为正无穷时的补充累积分布函数值
        assert_almost_equal(val, 0.0)  # 使用 assert_almost_equal 断言函数验证结果近似等于 0.0

    # 定义测试函数 test_domain，测试 sc.pdtrc 函数在参数不在定义域内的情况
    def test_domain(self):
        val = sc.pdtrc(-1.1, 1.0)  # 调用 sc.pdtrc 函数计算参数为负值时的补充累积分布函数值
        assert np.isnan(val)  # 使用 np.isnan 函数验证结果是否为 NaN
```