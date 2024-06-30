# `D:\src\scipysrc\scikit-learn\sklearn\tests\test_isotonic.py`

```
# 导入必要的库和模块
import copy  # 导入深拷贝函数
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
import warnings  # 导入警告处理模块

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例
from scipy.special import expit  # 从SciPy库中导入expit函数，用于逻辑斯蒂回归

import sklearn  # 导入scikit-learn库
from sklearn.datasets import make_regression  # 从scikit-learn中导入make_regression函数，用于生成回归数据集
from sklearn.isotonic import (  # 从scikit-learn中导入等渗回归相关模块和函数
    IsotonicRegression,
    _make_unique,
    check_increasing,
    isotonic_regression,
)
from sklearn.utils import shuffle  # 从scikit-learn中导入shuffle函数，用于打乱数据集
from sklearn.utils._testing import (  # 从scikit-learn中导入测试相关函数
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)
from sklearn.utils.validation import check_array  # 从scikit-learn中导入数据验证函数

# 测试函数：检验等渗回归是否具有排列不变性
def test_permutation_invariance():
    # 检查等渗回归是否具有排列不变性。
    # 这是一个回归测试，用于检查样本权重排序是否丢失。
    
    # 初始化等渗回归对象
    ir = IsotonicRegression()
    x = [1, 2, 3, 4, 5, 6, 7]  # 输入数据 x
    y = [1, 41, 51, 1, 2, 5, 24]  # 输入数据 y
    sample_weight = [1, 2, 3, 4, 5, 6, 7]  # 样本权重

    # 打乱输入数据
    x_s, y_s, sample_weight_s = shuffle(x, y, sample_weight, random_state=0)
    
    # 对数据进行等渗回归变换
    y_transformed = ir.fit_transform(x, y, sample_weight=sample_weight)
    y_transformed_s = ir.fit(x_s, y_s, sample_weight=sample_weight_s).transform(x)

    # 断言两种变换结果相等
    assert_array_equal(y_transformed, y_transformed_s)


# 测试函数：检验 check_increasing 函数对少量样本的正确性
def test_check_increasing_small_number_of_samples():
    x = [0, 1, 2]  # 输入数据 x
    y = [1, 1.1, 1.05]  # 输入数据 y

    # 捕获 UserWarning 警告，并且不抛出
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        is_increasing = check_increasing(x, y)

    # 断言结果为 True，即数据递增
    assert is_increasing


# 测试函数：检验 check_increasing 函数对递增数据的正确性
def test_check_increasing_up():
    x = [0, 1, 2, 3, 4, 5]  # 输入数据 x
    y = [0, 1.5, 2.77, 8.99, 8.99, 50]  # 输入数据 y

    # 捕获 UserWarning 警告，并且不抛出
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        is_increasing = check_increasing(x, y)

    # 断言结果为 True，即数据递增
    assert is_increasing


# 测试函数：检验 check_increasing 函数对极端递增数据的正确性
def test_check_increasing_up_extreme():
    x = [0, 1, 2, 3, 4, 5]  # 输入数据 x
    y = [0, 1, 2, 3, 4, 5]  # 输入数据 y

    # 捕获 UserWarning 警告，并且不抛出
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        is_increasing = check_increasing(x, y)

    # 断言结果为 True，即数据递增
    assert is_increasing


# 测试函数：检验 check_increasing 函数对递减数据的正确性
def test_check_increasing_down():
    x = [0, 1, 2, 3, 4, 5]  # 输入数据 x
    y = [0, -1.5, -2.77, -8.99, -8.99, -50]  # 输入数据 y

    # 捕获 UserWarning 警告，并且不抛出
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        is_increasing = check_increasing(x, y)

    # 断言结果为 False，即数据不递增
    assert not is_increasing


# 测试函数：检验 check_increasing 函数对极端递减数据的正确性
def test_check_increasing_down_extreme():
    x = [0, 1, 2, 3, 4, 5]  # 输入数据 x
    y = [0, -1, -2, -3, -4, -5]  # 输入数据 y

    # 捕获 UserWarning 警告，并且不抛出
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        is_increasing = check_increasing(x, y)

    # 断言结果为 False，即数据不递增
    assert not is_increasing


# 测试函数：检验 check_increasing 函数对递减数据的警告
def test_check_ci_warn():
    x = [0, 1, 2, 3, 4, 5]  # 输入数据 x
    y = [0, -1, 2, -3, 4, -5]  # 输入数据 y

    msg = "interval"  # 期望的警告信息
    # 捕获 UserWarning 警告，并且抛出符合预期的警告信息
    with pytest.warns(UserWarning, match=msg):
        is_increasing = check_increasing(x, y)

    # 断言结果为 False，即数据不递增
    assert not is_increasing


# 测试函数：检验 isotonic_regression 函数的基本功能
def test_isotonic_regression():
    y = np.array([3, 7, 5, 9, 8, 7, 10])  # 输入数据 y
    # 创建一个 NumPy 数组，包含给定的数据 [3, 6, 6, 8, 8, 8, 10]
    y_ = np.array([3, 6, 6, 8, 8, 8, 10])
    # 断言调用 isotonic_regression 函数后返回的结果与 y_ 数组相等
    assert_array_equal(y_, isotonic_regression(y))

    # 创建一个 NumPy 数组，包含给定的数据 [10, 0, 2]
    y = np.array([10, 0, 2])
    # 创建一个 NumPy 数组，包含给定的数据 [4, 4, 4]
    y_ = np.array([4, 4, 4])
    # 断言调用 isotonic_regression 函数后返回的结果与 y_ 数组相等
    assert_array_equal(y_, isotonic_regression(y))

    # 创建一个 NumPy 数组 x，包含从 0 到 len(y)-1 的整数
    x = np.arange(len(y))
    # 创建一个 IsotonicRegression 对象，设置 y_min=0.0, y_max=1.0
    ir = IsotonicRegression(y_min=0.0, y_max=1.0)
    # 对 IsotonicRegression 对象进行拟合，使用 x 和 y 数据
    ir.fit(x, y)
    # 断言调用 fit 方法后，transform(x) 和 fit_transform(x, y) 的结果相等
    assert_array_equal(ir.fit(x, y).transform(x), ir.fit_transform(x, y))
    # 断言调用 transform 方法后，结果与 predict(x) 的结果相等
    assert_array_equal(ir.transform(x), ir.predict(x))

    # 检查对象对于排列的不敏感性
    # 生成一个长度为 y 数组长度的随机排列数组 perm
    perm = np.random.permutation(len(y))
    # 创建一个新的 IsotonicRegression 对象，设置 y_min=0.0, y_max=1.0
    ir = IsotonicRegression(y_min=0.0, y_max=1.0)
    # 断言调用 fit_transform 方法后，使用 perm 排列和原始排列的结果相等
    assert_array_equal(ir.fit_transform(x[perm], y[perm]), ir.fit_transform(x, y)[perm])
    # 断言调用 transform 方法后，使用 perm 排列和原始排列的结果相等
    assert_array_equal(ir.transform(x[perm]), ir.transform(x)[perm])

    # 检查当所有 x 值相等时，不会崩溃
    # 创建一个 IsotonicRegression 对象，默认配置
    ir = IsotonicRegression()
    # 断言调用 fit_transform 方法后，使用全部为 1 的 x 数组的结果与 y 的均值相等
    assert_array_equal(ir.fit_transform(np.ones(len(x)), y), np.mean(y))
def test_isotonic_regression_ties_min():
    # 设置包含最小值处有并列情况的示例
    x = [1, 1, 2, 3, 4, 5]  # 输入变量 x
    y = [1, 2, 3, 4, 5, 6]  # 输出变量 y
    y_true = [1.5, 1.5, 3, 4, 5, 6]  # 预期输出 y_true

    # 检查 fit/transform 和 fit_transform 是否得到相同结果
    ir = IsotonicRegression()  # 创建 IsotonicRegression 实例
    ir.fit(x, y)  # 用 x, y 进行拟合
    assert_array_equal(ir.fit(x, y).transform(x), ir.fit_transform(x, y))  # 检查 transform 方法的结果
    assert_array_equal(y_true, ir.fit_transform(x, y))  # 检查 fit_transform 方法的结果


def test_isotonic_regression_ties_max():
    # 设置包含最大值处有并列情况的示例
    x = [1, 2, 3, 4, 5, 5]  # 输入变量 x
    y = [1, 2, 3, 4, 5, 6]  # 输出变量 y
    y_true = [1, 2, 3, 4, 5.5, 5.5]  # 预期输出 y_true

    # 检查 fit/transform 和 fit_transform 是否得到相同结果
    ir = IsotonicRegression()  # 创建 IsotonicRegression 实例
    ir.fit(x, y)  # 用 x, y 进行拟合
    assert_array_equal(ir.fit(x, y).transform(x), ir.fit_transform(x, y))  # 检查 transform 方法的结果
    assert_array_equal(y_true, ir.fit_transform(x, y))  # 检查 fit_transform 方法的结果


def test_isotonic_regression_ties_secondary_():
    """
    对 "secondary" 并列情况下的 Isotonic Regression 进行测试，
    使用 R 中 "isotone" 包中的 "pituitary" 数据集进行验证，
    参考文献：J. d. Leeuw, K. Hornik, P. Mair, Isotone Optimization in R:
    Pool-Adjacent-Violators Algorithm (PAVA) and Active Set Methods

    基于 pituitary 示例和上述文献中的 R 命令设置值：
    > library("isotone")
    > data("pituitary")
    > res1 <- gpava(pituitary$age, pituitary$size, ties="secondary")
    > res1$x

    `isotone` 版本：1.0-2, 2014-09-07
    R 版本：R version 3.1.1 (2014-07-10)
    """
    x = [8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14]  # 输入变量 x
    y = [21, 23.5, 23, 24, 21, 25, 21.5, 22, 19, 23.5, 25]  # 输出变量 y
    y_true = [
        22.22222,
        22.22222,
        22.22222,
        22.22222,
        22.22222,
        22.22222,
        22.22222,
        22.22222,
        22.22222,
        24.25,
        24.25,
    ]  # 预期输出 y_true

    # 检查 fit, transform 和 fit_transform 方法
    ir = IsotonicRegression()  # 创建 IsotonicRegression 实例
    ir.fit(x, y)  # 用 x, y 进行拟合
    assert_array_almost_equal(ir.transform(x), y_true, 4)  # 检查 transform 方法的结果
    assert_array_almost_equal(ir.fit_transform(x, y), y_true, 4)  # 检查 fit_transform 方法的结果


def test_isotonic_regression_with_ties_in_differently_sized_groups():
    """
    处理问题 9432 的非回归测试：
    https://github.com/scikit-learn/scikit-learn/issues/9432

    与 R 中的输出进行比较：
    > library("isotone")
    > x <- c(0, 1, 1, 2, 3, 4)
    > y <- c(0, 0, 1, 0, 0, 1)
    > res1 <- gpava(x, y, ties="secondary")
    > res1$x

    `isotone` 版本：1.1-0, 2015-07-24
    R 版本：R version 3.3.2 (2016-10-31)
    """
    x = np.array([0, 1, 1, 2, 3, 4])  # 输入变量 x
    y = np.array([0, 0, 1, 0, 0, 1])  # 输出变量 y
    y_true = np.array([0.0, 0.25, 0.25, 0.25, 0.25, 1.0])  # 预期输出 y_true

    ir = IsotonicRegression()  # 创建 IsotonicRegression 实例
    ir.fit(x, y)  # 用 x, y 进行拟合
    assert_array_almost_equal(ir.transform(x), y_true)  # 检查 transform 方法的结果
    assert_array_almost_equal(ir.fit_transform(x, y), y_true)  # 检查 fit_transform 方法的结果


def test_isotonic_regression_reversed():
    y = np.array([10, 9, 10, 7, 6, 6.1, 5])  # 输出变量 y
    y_result = np.array([10, 9.5, 9.5, 7, 6.05, 6.05, 5])  # 预期输出 y_result
    # 对 y 进行保序回归（isotonic regression），确保结果是非增的
    y_iso = isotonic_regression(y, increasing=False)
    # 断言保序回归的结果与预期结果 y_result 接近
    assert_allclose(y_iso, y_result)

    # 使用保序回归模型对 y 进行变换，确保结果是非增的
    y_ = IsotonicRegression(increasing=False).fit_transform(np.arange(len(y)), y)
    # 断言变换后的 y_ 的每个元素与预期结果 y_result 接近
    assert_allclose(y_, y_result)
    # 断言 y_ 的每对相邻元素之差都大于等于 0
    assert_array_equal(np.ones(y_[:-1].shape), ((y_[:-1] - y_[1:]) >= 0))
def test_isotonic_regression_auto_decreasing():
    # 设置 y 和 x 以进行递减
    y = np.array([10, 9, 10, 7, 6, 6.1, 5])
    x = np.arange(len(y))

    # 创建模型并进行拟合转换
    ir = IsotonicRegression(increasing="auto")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y_ = ir.fit_transform(x, y)
        # 用于解决 scipy <= 0.17.0 中的 Pearson 分割警告
        assert all(["invalid value encountered in " in str(warn.message) for warn in w])

    # 检查关系是否减少
    is_increasing = y_[0] < y_[-1]
    assert not is_increasing


def test_isotonic_regression_auto_increasing():
    # 设置 y 和 x 以进行递增
    y = np.array([5, 6.1, 6, 7, 10, 9, 10])
    x = np.arange(len(y))

    # 创建模型并进行拟合转换
    ir = IsotonicRegression(increasing="auto")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y_ = ir.fit_transform(x, y)
        # 用于解决 scipy <= 0.17.0 中的 Pearson 分割警告
        assert all(["invalid value encountered in " in str(warn.message) for warn in w])

    # 检查关系是否增加
    is_increasing = y_[0] < y_[-1]
    assert is_increasing


def test_assert_raises_exceptions():
    ir = IsotonicRegression()
    rng = np.random.RandomState(42)

    msg = "Found input variables with inconsistent numbers of samples"
    with pytest.raises(ValueError, match=msg):
        ir.fit([0, 1, 2], [5, 7, 3], [0.1, 0.6])

    with pytest.raises(ValueError, match=msg):
        ir.fit([0, 1, 2], [5, 7])

    msg = "X should be a 1d array"
    with pytest.raises(ValueError, match=msg):
        ir.fit(rng.randn(3, 10), [0, 1, 2])

    msg = "Isotonic regression input X should be a 1d array"
    with pytest.raises(ValueError, match=msg):
        ir.transform(rng.randn(3, 10))


def test_isotonic_sample_weight_parameter_default_value():
    # 检查 sample_weight 参数的默认值是否为 1
    ir = IsotonicRegression()
    # 随机测试数据
    rng = np.random.RandomState(42)
    n = 100
    x = np.arange(n)
    y = rng.randint(-50, 50, size=(n,)) + 50.0 * np.log(1 + np.arange(n))
    # 检查默认值是否正确使用
    weights = np.ones(n)
    y_set_value = ir.fit_transform(x, y, sample_weight=weights)
    y_default_value = ir.fit_transform(x, y)

    assert_array_equal(y_set_value, y_default_value)


def test_isotonic_min_max_boundaries():
    # 检查是否正确使用最小值和最大值
    ir = IsotonicRegression(y_min=2, y_max=4)
    n = 6
    x = np.arange(n)
    y = np.arange(n)
    y_test = [2, 2, 2, 3, 4, 4]
    y_result = np.round(ir.fit_transform(x, y))
    assert_array_equal(y_result, y_test)


def test_isotonic_sample_weight():
    ir = IsotonicRegression()
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [1, 41, 51, 1, 2, 5, 24]
    sample_weight = [1, 2, 3, 4, 5, 6, 7]
    expected_y = [1, 13.95, 13.95, 13.95, 13.95, 13.95, 24]
    # 检查样本权重是否正确影响结果
    ```
    # 使用给定的输入数据 x, y 和可选的样本权重 sample_weight，进行数据变换并拟合模型
    received_y = ir.fit_transform(x, y, sample_weight=sample_weight)
    
    # 断言接收到的转换后的数据 received_y 与期望的数据 expected_y 相等，用于验证模型转换的正确性
    assert_array_equal(expected_y, received_y)
# 测试 Isotonic 回归模型在越界处理时抛出异常的情况

def test_isotonic_regression_oob_raise():
    # 设置 y 和 x
    y = np.array([3, 7, 5, 9, 8, 7, 10])
    x = np.arange(len(y))

    # 创建模型并进行拟合
    ir = IsotonicRegression(increasing="auto", out_of_bounds="raise")
    ir.fit(x, y)

    # 检查是否抛出异常
    msg = "in x_new is below the interpolation range"
    with pytest.raises(ValueError, match=msg):
        ir.predict([min(x) - 10, max(x) + 10])


# 测试 Isotonic 回归模型在越界处理时进行截断的情况

def test_isotonic_regression_oob_clip():
    # 设置 y 和 x
    y = np.array([3, 7, 5, 9, 8, 7, 10])
    x = np.arange(len(y))

    # 创建模型并进行拟合
    ir = IsotonicRegression(increasing="auto", out_of_bounds="clip")
    ir.fit(x, y)

    # 从训练数据预测并测试 x，检查最小值和最大值是否匹配
    y1 = ir.predict([min(x) - 10, max(x) + 10])
    y2 = ir.predict(x)
    assert max(y1) == max(y2)
    assert min(y1) == min(y2)


# 测试 Isotonic 回归模型在越界处理时返回 NaN 的情况

def test_isotonic_regression_oob_nan():
    # 设置 y 和 x
    y = np.array([3, 7, 5, 9, 8, 7, 10])
    x = np.arange(len(y))

    # 创建模型并进行拟合
    ir = IsotonicRegression(increasing="auto", out_of_bounds="nan")
    ir.fit(x, y)

    # 从训练数据预测并测试 x，检查是否返回了两个 NaN
    y1 = ir.predict([min(x) - 10, max(x) + 10])
    assert sum(np.isnan(y1)) == 2


# 测试 Isotonic 回归模型在序列化和反序列化时的一致性

def test_isotonic_regression_pickle():
    y = np.array([3, 7, 5, 9, 8, 7, 10])
    x = np.arange(len(y))

    # 创建模型并进行拟合
    ir = IsotonicRegression(increasing="auto", out_of_bounds="clip")
    ir.fit(x, y)

    # 序列化模型对象并进行反序列化
    ir_ser = pickle.dumps(ir, pickle.HIGHEST_PROTOCOL)
    ir2 = pickle.loads(ir_ser)

    # 测试预测结果的一致性
    np.testing.assert_array_equal(ir.predict(x), ir2.predict(x))


# 测试 Isotonic 回归模型在存在重复最小值输入时的情况

def test_isotonic_duplicate_min_entry():
    x = [0, 0, 1]
    y = [0, 0, 1]

    # 创建模型并进行拟合
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    ir.fit(x, y)

    # 检查所有预测值是否有限
    all_predictions_finite = np.all(np.isfinite(ir.predict(x)))
    assert all_predictions_finite


# 测试 Isotonic 回归模型在指定最小和最大值范围时的情况

def test_isotonic_ymin_ymax():
    # 测试来自 @NelleV 的问题:
    # https://github.com/scikit-learn/scikit-learn/issues/6921
    x = np.array([
        1.263, 1.318, -0.572, 0.307, -0.707, -0.176, -1.599, 1.059,
        1.396, 1.906, 0.210, 0.028, -0.081, 0.444, 0.018, -0.377,
        -0.896, -0.377, -1.327, 0.180
    ])
    
    # 执行 Isotonic 回归，并限制 y 的最小和最大值范围
    y = isotonic_regression(x, y_min=0.0, y_max=0.1)
    
    # 断言所有预测值都大于等于 0，并且小于等于 0.1
    assert np.all(y >= 0)
    assert np.all(y <= 0.1)
    
    # 测试递减情况，因为逻辑不同
    y = isotonic_regression(x, y_min=0.0, y_max=0.1, increasing=False)
    
    # 断言所有预测值都大于等于 0，并且小于等于 0.1
    assert np.all(y >= 0)
    assert np.all(y <= 0.1)
    
    # 最后，测试只有一个边界值的情况
    y = isotonic_regression(x, y_min=0.0, increasing=False)
    
    # 断言所有预测值都大于等于 0
    assert np.all(y >= 0)


# 测试 Isotonic 回归模型在权重为零的情况下的行为

def test_isotonic_zero_weight_loop():
    # 测试来自 @ogrisel 的问题:
    
    # 设置随机数生成器（RNG）的种子为42，以确保结果可重现性
    rng = np.random.RandomState(42)
    
    # 创建一个等差数列作为回归的输入变量 x，并生成对应的输出变量 y
    n_samples = 50
    x = np.linspace(-3, 3, n_samples)
    y = x + rng.uniform(size=n_samples)
    
    # 创建一个具有保序性质的回归模型对象
    regression = IsotonicRegression()
    
    # 生成一些随机权重，并将其中一部分权重置为0
    w = rng.uniform(size=n_samples)
    w[5:8] = 0
    
    # 使用带权重的数据拟合回归模型
    regression.fit(x, y, sample_weight=w)
    
    # 这行代码可能会导致程序挂起（hang），即使是失败的情况下也会继续运行。
    regression.fit(x, y, sample_weight=w)
# 测试快速预测的功能是否不影响样本外的预测：
# https://github.com/scikit-learn/scikit-learn/pull/6206
def test_fast_predict():
    # 使用种子为123的随机数生成器创建随机数生成器对象
    rng = np.random.RandomState(123)
    # 设定样本数量为1000
    n_samples = 10**3
    # 生成在[-10, 10]范围内的随机X值
    X_train = 20.0 * rng.rand(n_samples) - 10
    # 根据X_train的expit函数结果生成随机y_train，转换为浮点64位整数
    y_train = (
        np.less(rng.rand(n_samples), expit(X_train)).astype("int64").astype("float64")
    )

    # 生成与样本数量相同的随机权重
    weights = rng.rand(n_samples)
    # 将其中一部分权重设为0，测试当部分权重为0时模型依然有效
    weights[rng.rand(n_samples) < 0.1] = 0

    # 创建IsotonicRegression类的慢速模型和快速模型对象
    slow_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    fast_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")

    # 使用所有输入数据建立插值函数，而不是删除冗余点
    # 下面两行代码摘自.fit()方法，未删除不必要的点
    X_train_fit, y_train_fit = slow_model._build_y(
        X_train, y_train, sample_weight=weights, trim_duplicates=False
    )
    slow_model._build_f(X_train_fit, y_train_fit)

    # 使用只包含必要数据进行拟合
    fast_model.fit(X_train, y_train, sample_weight=weights)

    # 生成测试数据集
    X_test = 20.0 * rng.rand(n_samples) - 10
    # 分别用慢速模型和快速模型预测结果
    y_pred_slow = slow_model.predict(X_test)
    y_pred_fast = fast_model.predict(X_test)

    # 断言慢速模型和快速模型的预测结果应该相等
    assert_array_equal(y_pred_slow, y_pred_fast)


def test_isotonic_copy_before_fit():
    # https://github.com/scikit-learn/scikit-learn/issues/6628
    # 创建IsotonicRegression对象
    ir = IsotonicRegression()
    # 复制IsotonicRegression对象
    copy.copy(ir)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_isotonic_dtype(dtype):
    # 设置y值和权重
    y = [2, 1, 4, 3, 5]
    weights = np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float64)
    # 创建IsotonicRegression对象
    reg = IsotonicRegression()

    # 针对不同的权重类型进行循环测试
    for sample_weight in (None, weights.astype(np.float32), weights):
        # 将y转换为指定dtype的numpy数组
        y_np = np.array(y, dtype=dtype)
        # 检查y_np数组的数据类型，并确保为二维数组
        expected_dtype = check_array(
            y_np, dtype=[np.float64, np.float32], ensure_2d=False
        ).dtype

        # 使用isotonic_regression函数对y_np进行拟合
        res = isotonic_regression(y_np, sample_weight=sample_weight)
        # 断言结果数据类型与预期数据类型相等
        assert res.dtype == expected_dtype

        # 创建X数组作为输入
        X = np.arange(len(y)).astype(dtype)
        # 对IsotonicRegression对象进行拟合，并预测结果
        reg.fit(X, y_np, sample_weight=sample_weight)
        res = reg.predict(X)
        # 断言预测结果数据类型与预期数据类型相等
        assert res.dtype == expected_dtype


@pytest.mark.parametrize("y_dtype", [np.int32, np.int64, np.float32, np.float64])
def test_isotonic_mismatched_dtype(y_dtype):
    # 回归测试，用于检查X和y的数据类型不同时是否会转换数据
    reg = IsotonicRegression()
    # 创建指定数据类型的y数组
    y = np.array([2, 1, 4, 3, 5], dtype=y_dtype)
    # 创建指定数据类型的X数组
    X = np.arange(len(y), dtype=np.float32)
    # 对IsotonicRegression对象进行拟合，并预测结果
    reg.fit(X, y)
    # 断言预测结果数据类型与X的数据类型相同
    assert reg.predict(X).dtype == X.dtype


def test_make_unique_dtype():
    # 创建包含重复值的x_list列表
    x_list = [2, 2, 2, 3, 5]
    # 针对不同的dtype进行循环测试
    for dtype in (np.float32, np.float64):
        # 创建指定dtype的numpy数组x
        x = np.array(x_list, dtype=dtype)
        # 将x复制给y
        y = x.copy()
        # 创建与x形状相同的全1数组w
        w = np.ones_like(x)
        # 调用_make_unique函数，处理x, y, w数组
        x, y, w = _make_unique(x, y, w)
        # 断言处理后的x数组不包含重复值
        assert_array_equal(x, [2, 3, 5])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
# 使用 pytest 的参数化功能来定义测试用例，参数 dtype 分别为 np.float64 和 np.float32
def test_make_unique_tolerance(dtype):
    # 检查相等性时考虑 np.finfo 的容差
    x = np.array([0, 1e-16, 1, 1 + 1e-14], dtype=dtype)
    y = x.copy()
    w = np.ones_like(x)
    x, y, w = _make_unique(x, y, w)
    if dtype == np.float64:
        x_out = np.array([0, 1, 1 + 1e-14])
    else:
        x_out = np.array([0, 1])
    # 断言数组 x 与预期结果 x_out 相等
    assert_array_equal(x, x_out)


def test_isotonic_make_unique_tolerance():
    # 检查重复 X 值时，目标的平均处理是否正确，考虑容差
    X = np.array([0, 1, 1 + 1e-16, 2], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)
    # 使用 IsotonicRegression 拟合数据
    ireg = IsotonicRegression().fit(X, y)
    # 对新数据进行预测
    y_pred = ireg.predict([0, 0.5, 1, 1.5, 2])

    # 断言预测结果与期望结果相等
    assert_array_equal(y_pred, np.array([0, 0.75, 1.5, 2.25, 3]))
    # 断言 X 的阈值与预期数组相等
    assert_array_equal(ireg.X_thresholds_, np.array([0.0, 1.0, 2.0]))
    # 断言 y 的阈值与预期数组相等
    assert_array_equal(ireg.y_thresholds_, np.array([0.0, 1.5, 3.0]))


def test_isotonic_non_regression_inf_slope():
    # 非回归测试，确保不返回无穷大值
    # 参考：https://github.com/scikit-learn/scikit-learn/issues/10903
    X = np.array([0.0, 4.1e-320, 4.4e-314, 1.0])
    y = np.array([0.42, 0.42, 0.44, 0.44])
    # 使用 IsotonicRegression 拟合数据
    ireg = IsotonicRegression().fit(X, y)
    # 对新数据进行预测
    y_pred = ireg.predict(np.array([0, 2.1e-319, 5.4e-316, 1e-10]))
    # 断言所有预测结果都是有限的
    assert np.all(np.isfinite(y_pred))


@pytest.mark.parametrize("increasing", [True, False])
# 使用 pytest 的参数化功能来定义测试用例，参数 increasing 分别为 True 和 False
def test_isotonic_thresholds(increasing):
    rng = np.random.RandomState(42)
    n_samples = 30
    X = rng.normal(size=n_samples)
    y = rng.normal(size=n_samples)
    # 使用 IsotonicRegression 拟合数据，指定是否递增
    ireg = IsotonicRegression(increasing=increasing).fit(X, y)
    X_thresholds, y_thresholds = ireg.X_thresholds_, ireg.y_thresholds_
    # 断言 X_thresholds 和 y_thresholds 的形状相同
    assert X_thresholds.shape == y_thresholds.shape

    # 输入的阈值严格是训练集的子集（除非数据已经严格单调，对于这些随机数据来说并非如此）
    assert X_thresholds.shape[0] < X.shape[0]
    assert np.isin(X_thresholds, X).all()

    # 输出的阈值在训练集的范围内
    assert y_thresholds.max() <= y.max()
    assert y_thresholds.min() >= y.min()

    # 断言 X_thresholds 是递增的
    assert all(np.diff(X_thresholds) > 0)
    if increasing:
        # 如果递增，则断言 y_thresholds 是递增的
        assert all(np.diff(y_thresholds) >= 0)
    else:
        # 如果非递增，则断言 y_thresholds 是非递增的
        assert all(np.diff(y_thresholds) <= 0)


def test_input_shape_validation():
    # 来自 issue #15012 的测试
    # 检查 IsotonicRegression 是否能处理只有一个特征的二维数组
    X = np.arange(10)
    X_2d = X.reshape(-1, 1)
    y = np.arange(10)

    # 分别拟合 X 和 X_2d
    iso_reg = IsotonicRegression().fit(X, y)
    iso_reg_2d = IsotonicRegression().fit(X_2d, y)

    # 断言各个属性相等
    assert iso_reg.X_max_ == iso_reg_2d.X_max_
    assert iso_reg.X_min_ == iso_reg_2d.X_min_
    assert iso_reg.y_max == iso_reg_2d.y_max
    assert iso_reg.y_min == iso_reg_2d.y_min
    assert_array_equal(iso_reg.X_thresholds_, iso_reg_2d.X_thresholds_)
    # 断言两个 IsolationForest 模型的 y_thresholds_ 属性相等
    assert_array_equal(iso_reg.y_thresholds_, iso_reg_2d.y_thresholds_)
    
    # 使用 iso_reg 模型对 X 进行预测，并将结果保存在 y_pred1 中
    y_pred1 = iso_reg.predict(X)
    # 使用 iso_reg_2d 模型对 X_2d 进行预测，并将结果保存在 y_pred2 中
    y_pred2 = iso_reg_2d.predict(X_2d)
    # 断言 y_pred1 和 y_pred2 数组的元素在允许误差范围内相等
    assert_allclose(y_pred1, y_pred2)
# 确保如果输入具有多于一个特征，则 IsotonicRegression 会引发错误
def test_isotonic_2darray_more_than_1_feature():
    # 创建一个包含 0 到 9 的一维数组作为特征 X
    X = np.arange(10)
    # 使用 np.c_ 将一维数组 X 转换为包含两个相同列的二维数组 X_2d
    X_2d = np.c_[X, X]
    # 创建另一个一维数组作为目标值 y
    y = np.arange(10)

    # 错误消息，指示应该是一个一维数组，或者包含一个特征的二维数组
    msg = "should be a 1d array or 2d array with 1 feature"
    # 使用 pytest 的 assertRaises 来检查是否引发 ValueError，且错误消息匹配 msg
    with pytest.raises(ValueError, match=msg):
        IsotonicRegression().fit(X_2d, y)

    # 创建 IsotonicRegression 对象，拟合一维数组 X 和目标值 y
    iso_reg = IsotonicRegression().fit(X, y)
    # 使用 pytest 的 assertRaises 来检查是否引发 ValueError，且错误消息匹配 msg
    with pytest.raises(ValueError, match=msg):
        iso_reg.predict(X_2d)

    # 使用 pytest 的 assertRaises 来检查是否引发 ValueError，且错误消息匹配 msg
    with pytest.raises(ValueError, match=msg):
        iso_reg.transform(X_2d)


# 检查 isotonic regression 的拟合函数不会覆盖 `sample_weight`
def test_isotonic_regression_sample_weight_not_overwritten():
    """Check that calling fitting function of isotonic regression will not
    overwrite `sample_weight`.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/20508
    """
    # 创建包含一个特征的回归数据集 X 和目标值 y
    X, y = make_regression(n_samples=10, n_features=1, random_state=41)
    # 创建一个与 y 大小相同的全为 1 的权重数组 sample_weight_original，并将第一个元素设为 10
    sample_weight_original = np.ones_like(y)
    sample_weight_original[0] = 10
    # 复制权重数组 sample_weight_original 到 sample_weight_fit
    sample_weight_fit = sample_weight_original.copy()

    # 使用 isotonic_regression 函数拟合数据集 y 和权重数组 sample_weight_fit
    isotonic_regression(y, sample_weight=sample_weight_fit)
    # 使用 assert_allclose 检查 sample_weight_fit 是否与 sample_weight_original 全部接近
    assert_allclose(sample_weight_fit, sample_weight_original)

    # 创建 IsotonicRegression 对象，拟合数据集 X, y 和权重数组 sample_weight_fit
    IsotonicRegression().fit(X, y, sample_weight=sample_weight_fit)
    # 使用 assert_allclose 检查 sample_weight_fit 是否与 sample_weight_original 全部接近
    assert_allclose(sample_weight_fit, sample_weight_original)


# 使用参数化测试检查 IsotonicRegression 的 `get_feature_names_out` 方法
@pytest.mark.parametrize("shape", ["1d", "2d"])
def test_get_feature_names_out(shape):
    """Check `get_feature_names_out` for `IsotonicRegression`."""
    # 创建一维数组作为特征 X
    X = np.arange(10)
    # 如果 shape 为 "2d"，则将一维数组 X 重塑为包含一个特征的二维数组
    if shape == "2d":
        X = X.reshape(-1, 1)
    # 创建一维数组作为目标值 y
    y = np.arange(10)

    # 创建 IsotonicRegression 对象，拟合特征 X 和目标值 y
    iso = IsotonicRegression().fit(X, y)
    # 调用 get_feature_names_out 方法，获取特征名称列表 names
    names = iso.get_feature_names_out()
    # 使用 assert 检查 names 是否是 np.ndarray 类型
    assert isinstance(names, np.ndarray)
    # 使用 assert 检查 names 的数据类型是否为 object
    assert names.dtype == object
    # 使用 assert_array_equal 检查 names 是否等于 ["isotonicregression0"]
    assert_array_equal(["isotonicregression0"], names)


# 检查 isotonic regression 的预测函数是否返回预期的输出类型
def test_isotonic_regression_output_predict():
    """Check that `predict` does return the expected output type.

    We need to check that `transform` will output a DataFrame and a NumPy array
    when we set `transform_output` to `pandas`.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/25499
    """
    # 导入 pytest 版本的 pandas，如未安装则跳过测试
    pd = pytest.importorskip("pandas")
    # 创建包含一个特征的回归数据集 X 和目标值 y
    X, y = make_regression(n_samples=10, n_features=1, random_state=42)
    # 创建 IsotonicRegression 对象
    regressor = IsotonicRegression()
    # 使用 sklearn.config_context 设置 transform_output 参数为 "pandas"，拟合数据集 X, y
    with sklearn.config_context(transform_output="pandas"):
        regressor.fit(X, y)
        # 调用 transform 方法，得到转换后的数据集 X_trans
        X_trans = regressor.transform(X)
        # 调用 predict 方法，得到预测值数组 y_pred
        y_pred = regressor.predict(X)

    # 使用 assert 检查 X_trans 是否是 pd.DataFrame 类型
    assert isinstance(X_trans, pd.DataFrame)
    # 使用 assert 检查 y_pred 是否是 np.ndarray 类型
    assert isinstance(y_pred, np.ndarray)
```