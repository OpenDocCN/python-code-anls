# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_random.py`

```
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal
from scipy.special import comb

from sklearn.utils._random import _our_rand_r_py
from sklearn.utils.random import _random_choice_csc, sample_without_replacement


###############################################################################
# test custom sampling without replacement algorithm
###############################################################################

# 测试自定义的无重复抽样算法

def test_invalid_sample_without_replacement_algorithm():
    # 测试当传入未知方法时是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        sample_without_replacement(5, 4, "unknown")


def test_sample_without_replacement_algorithms():
    # 测试不同的无重复抽样算法

    methods = ("auto", "tracking_selection", "reservoir_sampling", "pool")

    for m in methods:

        def sample_without_replacement_method(
            n_population, n_samples, random_state=None
        ):
            return sample_without_replacement(
                n_population, n_samples, method=m, random_state=random_state
            )

        # 检查样本数量小于总体数量时是否引发 ValueError 异常
        check_edge_case_of_sample_int(sample_without_replacement_method)
        # 检查抽样结果的长度和唯一性
        check_sample_int(sample_without_replacement_method)
        # 检查抽样分布的多样性
        check_sample_int_distribution(sample_without_replacement_method)


def check_edge_case_of_sample_int(sample_without_replacement):
    # 检查边界情况下的整数抽样

    # 当总体数量小于样本数量时应引发 ValueError 异常
    with pytest.raises(ValueError):
        sample_without_replacement(0, 1)
    with pytest.raises(ValueError):
        sample_without_replacement(1, 2)

    # 当总体数量等于样本数量时，应返回空数组
    assert sample_without_replacement(0, 0).shape == (0,)
    assert sample_without_replacement(1, 1).shape == (1,)

    # 当总体数量大于等于样本数量时，应返回符合预期的抽样结果
    assert sample_without_replacement(5, 0).shape == (0,)
    assert sample_without_replacement(5, 1).shape == (1,)

    # 当总体数量或样本数量小于零时应引发 ValueError 异常
    with pytest.raises(ValueError):
        sample_without_replacement(-1, 5)
    with pytest.raises(ValueError):
        sample_without_replacement(5, -1)


def check_sample_int(sample_without_replacement):
    # 检查整数抽样的结果

    # 受启发于 python-core 中的 test_random.py
    # 对于所有允许的 0 <= k <= N 范围内的 n_samples，验证抽样结果的长度和唯一性
    n_population = 100

    for n_samples in range(n_population + 1):
        s = sample_without_replacement(n_population, n_samples)
        assert len(s) == n_samples
        unique = np.unique(s)
        assert np.size(unique) == n_samples
        assert np.all(unique < n_population)

    # 测试边界情况：当总体数量和样本数量均为零时，应返回空数组
    assert np.size(sample_without_replacement(0, 0)) == 0


def check_sample_int_distribution(sample_without_replacement):
    # 检查整数抽样的分布

    # 受启发于 python-core 中的 test_random.py
    # 对于所有允许的 0 <= k <= N 范围内的 n_samples，验证抽样生成所有可能的排列
    n_population = 10
    # 设置试验次数，以防止在正常情况下减少误报
    n_trials = 10000
    
    # 遍历总体中的样本数范围
    for n_samples in range(n_population):
        # 计算组合数目并确保精确计算
        n_expected = comb(n_population, n_samples, exact=True)
    
        # 初始化输出字典
        output = {}
        # 进行多次试验
        for i in range(n_trials):
            # 使用无替换抽样生成的 frozenset 作为键，并将值设为 None
            output[frozenset(sample_without_replacement(n_population, n_samples))] = (
                None
            )
    
            # 如果输出字典长度等于期望的组合数，则提前结束循环
            if len(output) == n_expected:
                break
        else:
            # 如果循环正常结束（未提前中断），则抛出断言错误
            raise AssertionError(
                "number of combinations != number of expected (%s != %s)"
                % (len(output), n_expected)
            )
# 定义一个测试函数，用于验证 _random_choice_csc 函数的行为
def test_random_choice_csc(n_samples=10000, random_state=24):
    # 明确指定的类别列表，每个类别对应的可能性分布
    classes = [np.array([0, 1]), np.array([0, 1, 2])]
    class_probabilities = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]

    # 调用 _random_choice_csc 函数进行抽样
    got = _random_choice_csc(n_samples, classes, class_probabilities, random_state)
    # 断言返回结果是稀疏矩阵
    assert sp.issparse(got)

    # 对每个类别进行验证
    for k in range(len(classes)):
        # 计算每列的频率分布
        p = np.bincount(got.getcol(k).toarray().ravel()) / float(n_samples)
        # 断言计算得到的频率与预期的类别概率接近（精确到小数点后一位）
        assert_array_almost_equal(class_probabilities[k], p, decimal=1)

    # 隐式类别概率分布
    classes = [[0, 1], [1, 2]]  # 测试数组形式的支持
    class_probabilities = [np.array([0.5, 0.5]), np.array([0, 1 / 2, 1 / 2])]

    # 再次调用 _random_choice_csc 函数进行抽样
    got = _random_choice_csc(
        n_samples=n_samples, classes=classes, random_state=random_state
    )
    # 断言返回结果是稀疏矩阵
    assert sp.issparse(got)

    # 对每个类别进行验证
    for k in range(len(classes)):
        # 计算每列的频率分布
        p = np.bincount(got.getcol(k).toarray().ravel()) / float(n_samples)
        # 断言计算得到的频率与预期的类别概率接近（精确到小数点后一位）
        assert_array_almost_equal(class_probabilities[k], p, decimal=1)

    # 概率为 1.0 和 0.0 的边界情况
    classes = [np.array([0, 1]), np.array([0, 1, 2])]
    class_probabilities = [np.array([0.0, 1.0]), np.array([0.0, 1.0, 0.0])]

    # 再次调用 _random_choice_csc 函数进行抽样
    got = _random_choice_csc(n_samples, classes, class_probabilities, random_state)
    # 断言返回结果是稀疏矩阵
    assert sp.issparse(got)

    # 对每个类别进行验证
    for k in range(len(classes)):
        # 计算每列的频率分布，确保使用类别概率的长度作为 minlength
        p = (
            np.bincount(
                got.getcol(k).toarray().ravel(), minlength=len(class_probabilities[k])
            )
            / n_samples
        )
        # 断言计算得到的频率与预期的类别概率接近（精确到小数点后一位）
        assert_array_almost_equal(class_probabilities[k], p, decimal=1)

    # 单一类别的目标数据
    classes = [[1], [0]]  # 测试数组形式的支持
    class_probabilities = [np.array([0.0, 1.0]), np.array([1.0])]

    # 再次调用 _random_choice_csc 函数进行抽样
    got = _random_choice_csc(
        n_samples=n_samples, classes=classes, random_state=random_state
    )
    # 断言返回结果是稀疏矩阵
    assert sp.issparse(got)

    # 对每个类别进行验证
    for k in range(len(classes)):
        # 计算每列的频率分布
        p = np.bincount(got.getcol(k).toarray().ravel()) / n_samples
        # 断言计算得到的频率与预期的类别概率接近（精确到小数点后一位）
        assert_array_almost_equal(class_probabilities[k], p, decimal=1)


# 定义一个测试函数，用于验证 _random_choice_csc 在特定错误情况下的行为
def test_random_choice_csc_errors():
    # 类别数组和概率数组长度不匹配
    classes = [np.array([0, 1]), np.array([0, 1, 2, 3])]
    class_probabilities = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]
    # 使用 pytest 检查是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        _random_choice_csc(4, classes, class_probabilities, 1)

    # 类别数据类型不受支持
    classes = [np.array(["a", "1"]), np.array(["z", "1", "2"])]
    class_probabilities = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]
    # 使用 pytest 检查是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        _random_choice_csc(4, classes, class_probabilities, 1)

    # 类别数据类型不受支持
    classes = [np.array([4.2, 0.1]), np.array([0.1, 0.2, 9.4])]
    class_probabilities = [np.array([0.5, 0.5]), np.array([0.6, 0.1, 0.3])]
    # 使用 pytest 检查是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        _random_choice_csc(4, classes, class_probabilities, 1)

    # 给定的概率值不等于 1 的边界情况
    # 定义两个类别数组，每个类别有不同的可能性分布
    classes = [np.array([0, 1]), np.array([0, 1, 2])]
    class_probabilities = [np.array([0.5, 0.6]), np.array([0.6, 0.1, 0.3])]
    # 使用 pytest 模块的 raises 函数，期望捕获到 ValueError 异常
    with pytest.raises(ValueError):
        # 调用 _random_choice_csc 函数，预期传入的参数会触发异常
        _random_choice_csc(4, classes, class_probabilities, 1)
# 定义一个测试函数，用于验证 _our_rand_r_py 函数的正确性
def test_our_rand_r():
    # 断言：调用 _our_rand_r_py 函数，传入参数 1273642419，期望返回值为 131541053
    assert 131541053 == _our_rand_r_py(1273642419)
    # 断言：调用 _our_rand_r_py 函数，传入参数 0，期望返回值为 270369
    assert 270369 == _our_rand_r_py(0)
```