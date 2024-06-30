# `D:\src\scipysrc\scipy\scipy\stats\tests\test_crosstab.py`

```
import pytest  # 导入 pytest 模块，用于编写和运行测试用例
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_array_equal, assert_equal  # 导入 NumPy 提供的数组比较函数
from scipy.stats.contingency import crosstab  # 从 SciPy 库的统计模块中导入 crosstab 函数


@pytest.mark.parametrize('sparse', [False, True])
def test_crosstab_basic(sparse):
    a = [0, 0, 9, 9, 0, 0, 9]  # 定义列表 a，包含整数元素
    b = [2, 1, 3, 1, 2, 3, 3]  # 定义列表 b，包含整数元素
    expected_avals = [0, 9]  # 预期的 a 值列表
    expected_bvals = [1, 2, 3]  # 预期的 b 值列表
    expected_count = np.array([[1, 2, 1],  # 预期的计数数组
                               [1, 0, 2]])
    (avals, bvals), count = crosstab(a, b, sparse=sparse)  # 调用 crosstab 函数，计算结果保存在变量中
    assert_array_equal(avals, expected_avals)  # 断言 avals 数组与预期的 a 值列表相等
    assert_array_equal(bvals, expected_bvals)  # 断言 bvals 数组与预期的 b 值列表相等
    if sparse:
        assert_array_equal(count.toarray(), expected_count)  # 如果 sparse 为 True，则断言稀疏矩阵 count 的值与预期计数数组相等
    else:
        assert_array_equal(count, expected_count)  # 如果 sparse 为 False，则断言 count 数组与预期计数数组相等


def test_crosstab_basic_1d():
    # 验证单个输入序列的功能
    x = [1, 2, 3, 1, 2, 3, 3]  # 定义列表 x，包含整数元素
    expected_xvals = [1, 2, 3]  # 预期的 x 值列表
    expected_count = np.array([2, 2, 3])  # 预期的计数数组
    (xvals,), count = crosstab(x)  # 调用 crosstab 函数，计算结果保存在变量中
    assert_array_equal(xvals, expected_xvals)  # 断言 xvals 数组与预期的 x 值列表相等
    assert_array_equal(count, expected_count)  # 断言 count 数组与预期计数数组相等


def test_crosstab_basic_3d():
    # 验证三个输入序列的功能
    a = 'a'  # 定义字符 'a'
    b = 'b'  # 定义字符 'b'
    x = [0, 0, 9, 9, 0, 0, 9, 9]  # 定义列表 x，包含整数元素
    y = [a, a, a, a, b, b, b, a]  # 定义列表 y，包含字符 'a' 和 'b'
    z = [1, 2, 3, 1, 2, 3, 3, 1]  # 定义列表 z，包含整数元素
    expected_xvals = [0, 9]  # 预期的 x 值列表
    expected_yvals = [a, b]  # 预期的 y 值列表
    expected_zvals = [1, 2, 3]  # 预期的 z 值列表
    expected_count = np.array([[[1, 1, 0],  # 预期的计数数组
                                [0, 1, 1]],
                               [[2, 0, 1],
                                [0, 0, 1]]])
    (xvals, yvals, zvals), count = crosstab(x, y, z)  # 调用 crosstab 函数，计算结果保存在变量中
    assert_array_equal(xvals, expected_xvals)  # 断言 xvals 数组与预期的 x 值列表相等
    assert_array_equal(yvals, expected_yvals)  # 断言 yvals 数组与预期的 y 值列表相等
    assert_array_equal(zvals, expected_zvals)  # 断言 zvals 数组与预期的 z 值列表相等
    assert_array_equal(count, expected_count)  # 断言 count 数组与预期计数数组相等


@pytest.mark.parametrize('sparse', [False, True])
def test_crosstab_levels(sparse):
    a = [0, 0, 9, 9, 0, 0, 9]  # 定义列表 a，包含整数元素
    b = [1, 2, 3, 1, 2, 3, 3]  # 定义列表 b，包含整数元素
    expected_avals = [0, 9]  # 预期的 a 值列表
    expected_bvals = [0, 1, 2, 3]  # 预期的 b 值列表
    expected_count = np.array([[0, 1, 2, 1],  # 预期的计数数组
                               [0, 1, 0, 2]])
    (avals, bvals), count = crosstab(a, b, levels=[None, [0, 1, 2, 3]], sparse=sparse)  # 调用 crosstab 函数，计算结果保存在变量中
    assert_array_equal(avals, expected_avals)  # 断言 avals 数组与预期的 a 值列表相等
    assert_array_equal(bvals, expected_bvals)  # 断言 bvals 数组与预期的 b 值列表相等
    if sparse:
        assert_array_equal(count.toarray(), expected_count)  # 如果 sparse 为 True，则断言稀疏矩阵 count 的值与预期计数数组相等
    else:
        assert_array_equal(count, expected_count)  # 如果 sparse 为 False，则断言 count 数组与预期计数数组相等


@pytest.mark.parametrize('sparse', [False, True])
def test_crosstab_extra_levels(sparse):
    # 对于这对值 (-1, 3)，将被忽略，因为我们明确请求计算的 `a` 值为 [0, 9]。
    a = [0, 0, 9, 9, 0, 0, 9, -1]  # 定义列表 a，包含整数元素，包括一个 -1 值
    b = [1, 2, 3, 1, 2, 3, 3, 3]  # 定义列表 b，包含整数元素
    expected_avals = [0, 9]  # 预期的 a 值列表
    expected_bvals = [0, 1, 2, 3]  # 预期的 b 值列表
    expected_count = np.array([[0, 1, 2, 1],  # 预期的计数数组
                               [0, 1, 0, 2]])
    (avals, bvals), count = crosstab(a, b, levels=[[0, 9], [0, 1, 2, 3]], sparse=sparse)  # 调用 crosstab 函数，计算结果保存在变量中
    assert_array_equal(avals, expected_avals)  # 断言 avals 数组与预期的 a 值列表相等
    # 使用断言检查数组 bvals 是否与预期的 bvals 相等
    assert_array_equal(bvals, expected_bvals)
    
    # 如果 sparse 参数为真，则将稀疏矩阵 count 转换为数组并检查其是否与预期的 count 相等
    if sparse:
        assert_array_equal(count.toarray(), expected_count)
    else:
        # 如果 sparse 参数为假，则直接检查 count 是否与预期的 count 相等
        assert_array_equal(count, expected_count)
# 测试函数：验证 crosstab 函数在参数不满足要求时是否会引发异常 TypeError
def test_validation_at_least_one():
    # 使用 pytest 来检查是否引发了 TypeError 异常，异常信息匹配 'At least one'
    with pytest.raises(TypeError, match='At least one'):
        # 调用 crosstab 函数，没有传入任何参数
        crosstab()


# 测试函数：验证 crosstab 函数在参数长度不一致时是否会引发异常 ValueError
def test_validation_same_lengths():
    # 使用 pytest 来检查是否引发了 ValueError 异常，异常信息匹配 'must have the same length'
    with pytest.raises(ValueError, match='must have the same length'):
        # 调用 crosstab 函数，传入两个长度不同的列表作为参数
        crosstab([1, 2], [1, 2, 3, 4])


# 测试函数：验证 crosstab 函数在使用 sparse 参数且仅有两个输入序列时是否会引发异常 ValueError
def test_validation_sparse_only_two_args():
    # 使用 pytest 来检查是否引发了 ValueError 异常，异常信息匹配 'only two input sequences'
    with pytest.raises(ValueError, match='only two input sequences'):
        # 调用 crosstab 函数，传入三个输入序列和 sparse=True 参数
        crosstab([0, 1, 1], [8, 8, 9], [1, 3, 3], sparse=True)


# 测试函数：验证 crosstab 函数在 levels 参数的长度与输入序列不匹配时是否会引发异常 ValueError
def test_validation_len_levels_matches_args():
    # 使用 pytest 来检查是否引发了 ValueError 异常，异常信息匹配 'number of input sequences'
    with pytest.raises(ValueError, match='number of input sequences'):
        # 调用 crosstab 函数，传入两个输入序列和一个 levels 参数，levels 参数的长度与输入序列不匹配
        crosstab([0, 1, 1], [8, 8, 9], levels=([0, 1, 2, 3],))


# 测试函数：验证 crosstab 函数返回结果是否符合预期
def test_result():
    # 调用 crosstab 函数，传入两个简单的输入序列
    res = crosstab([0, 1], [1, 2])
    # 使用 assert_equal 来验证返回的结果 res 的 elements 属性和 count 属性是否与 res 本身相等
    assert_equal((res.elements, res.count), res)
```