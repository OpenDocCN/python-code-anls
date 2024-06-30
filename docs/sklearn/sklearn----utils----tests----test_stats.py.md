# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_stats.py`

```
# 导入必要的库和模块
import numpy as np  # 导入 NumPy 库并重命名为 np
from numpy.testing import assert_allclose  # 从 NumPy 的 testing 模块中导入 assert_allclose 函数
from pytest import approx  # 从 pytest 库中导入 approx 对象

# 导入需要测试的函数 _weighted_percentile
from sklearn.utils.stats import _weighted_percentile


# 定义测试函数 test_weighted_percentile，用于测试加权百分位数计算的准确性
def test_weighted_percentile():
    # 创建一个长度为 102 的空数组 y，数据类型为 np.float64
    y = np.empty(102, dtype=np.float64)
    y[:50] = 0  # 将前 50 个元素设为 0
    y[-51:] = 2  # 将最后 51 个元素设为 2
    y[-1] = 100000  # 将最后一个元素设为 100000
    y[50] = 1  # 将第 50 个元素设为 1
    # 创建一个长度为 102 的全为 1 的数组 sw，数据类型为 np.float64
    sw = np.ones(102, dtype=np.float64)
    sw[-1] = 0.0  # 将 sw 数组的最后一个元素设为 0.0
    # 调用 _weighted_percentile 函数计算加权百分位数，目标百分位数为 50
    score = _weighted_percentile(y, sw, 50)
    # 使用 pytest 的 approx 断言，检查计算得到的分数约等于 1
    assert approx(score) == 1


# 定义测试函数 test_weighted_percentile_equal，验证当所有元素均为 0 时加权百分位数的计算结果
def test_weighted_percentile_equal():
    # 创建一个长度为 102 的空数组 y，数据类型为 np.float64，所有元素填充为 0.0
    y = np.empty(102, dtype=np.float64)
    y.fill(0.0)
    # 创建一个长度为 102 的全为 1 的数组 sw，数据类型为 np.float64
    sw = np.ones(102, dtype=np.float64)
    sw[-1] = 0.0  # 将 sw 数组的最后一个元素设为 0.0
    # 调用 _weighted_percentile 函数计算加权百分位数，目标百分位数为 50
    score = _weighted_percentile(y, sw, 50)
    # 断言加权百分位数的计算结果等于 0
    assert score == 0


# 定义测试函数 test_weighted_percentile_zero_weight，验证当所有权重为 0 时加权百分位数的计算结果
def test_weighted_percentile_zero_weight():
    # 创建一个长度为 102 的空数组 y，数据类型为 np.float64，所有元素填充为 1.0
    y = np.empty(102, dtype=np.float64)
    y.fill(1.0)
    # 创建一个长度为 102 的全为 0 的数组 sw，数据类型为 np.float64
    sw = np.ones(102, dtype=np.float64)
    sw.fill(0.0)  # 将 sw 数组所有元素设为 0.0
    # 调用 _weighted_percentile 函数计算加权百分位数，目标百分位数为 50
    score = _weighted_percentile(y, sw, 50)
    # 使用 pytest 的 approx 断言，检查计算得到的分数约等于 1.0
    assert approx(score) == 1.0


# 定义测试函数 test_weighted_percentile_zero_weight_zero_percentile，验证加权百分位数在特定情况下的计算结果
def test_weighted_percentile_zero_weight_zero_percentile():
    # 创建一个包含整数的数组 y 和对应的权重数组 sw
    y = np.array([0, 1, 2, 3, 4, 5])
    sw = np.array([0, 0, 1, 1, 1, 0])

    # 测试百分位数为 0 时的加权百分位数计算结果
    score = _weighted_percentile(y, sw, 0)
    assert approx(score) == 2

    # 测试百分位数为 50 时的加权百分位数计算结果
    score = _weighted_percentile(y, sw, 50)
    assert approx(score) == 3

    # 测试百分位数为 100 时的加权百分位数计算结果
    score = _weighted_percentile(y, sw, 100)
    assert approx(score) == 4


# 定义测试函数 test_weighted_median_equal_weights，验证在权重相等时加权百分位数与中位数的关系
def test_weighted_median_equal_weights():
    # 创建一个随机数生成器 rng
    rng = np.random.RandomState(0)
    # 创建一个长度为 11 的随机整数数组 x
    x = rng.randint(10, size=11)
    weights = np.ones(x.shape)  # 创建与 x 形状相同的全为 1 的权重数组

    median = np.median(x)  # 计算 x 的中位数
    w_median = _weighted_percentile(x, weights)  # 调用 _weighted_percentile 计算加权百分位数
    assert median == approx(w_median)  # 断言加权百分位数约等于中位数


# 定义测试函数 test_weighted_median_integer_weights，验证在手动指定权重时加权百分位数与中位数的关系
def test_weighted_median_integer_weights():
    # 创建一个随机数生成器 rng
    rng = np.random.RandomState(0)
    x = rng.randint(20, size=10)  # 创建一个长度为 10 的随机整数数组 x
    weights = rng.choice(5, size=10)  # 创建一个长度为 10 的随机整数数组 weights，取值范围为 [0, 5)

    x_manual = np.repeat(x, weights)  # 根据权重数组 weights，重复元素生成新的数组 x_manual

    median = np.median(x_manual)  # 计算 x_manual 的中位数
    w_median = _weighted_percentile(x, weights)  # 调用 _weighted_percentile 计算加权百分位数
    assert median == approx(w_median)  # 断言加权百分位数约等于中位数


# 定义测试函数 test_weighted_percentile_2d，验证在处理二维数组和一维权重数组时的加权百分位数计算
def test_weighted_percentile_2d():
    # 创建一个随机数生成器 rng
    rng = np.random.RandomState(0)
    # 创建两个长度为 10 的随机整数数组 x1 和 x2
    x1 = rng.randint(10, size=10)
    x2 = rng.randint(20, size=10)
    # 将 x1 和 x2 合并成一个二维数组 x_2d
    x_2d = np.vstack((x1, x2)).T

    # 创建一个长度为 10 的随机整数数组 w1 作为权重数组
    w1 = rng.choice(5, size=10)

    # 测试处理二维数组 x_2d 和一维权重数组 w1 时的加权百分位数计算结果
    w_median = _weighted_percentile(x_2d, w1)
    p_axis_0 = [_weighted_percentile(x_2d[:, i], w1) for i in range(x_2d.shape[1])]
    assert_allclose(w_median, p_axis_0)

    # 创建一个二维权重数组 w_2d
    w2 = rng.choice(5, size=10)
    w_2d = np.vstack((w1, w2)).T

    # 测试处理二维数组 x_2d 和二维权重数组 w_2d 时的加权百分位数计算结果
    w_median = _weighted_percentile(x_2d, w_2d)
    p_axis_0 = [
        _weighted_percentile(x_2d[:, i], w_2d[:, i]) for i in range(x_2d.shape[
```