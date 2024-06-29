# `D:\src\scipysrc\pandas\pandas\tests\groupby\methods\test_sample.py`

```
# 导入 pytest 模块
import pytest

# 从 pandas 模块中导入 DataFrame, Index, Series
from pandas import (
    DataFrame,
    Index,
    Series,
)
# 导入 pandas._testing 模块并重命名为 tm
import pandas._testing as tm

# 使用 pytest.mark.parametrize 装饰器，参数化测试用例
@pytest.mark.parametrize("n, frac", [(2, None), (None, 0.2)])
# 定义测试函数 test_groupby_sample_balanced_groups_shape，测试 groupby.sample 方法
def test_groupby_sample_balanced_groups_shape(n, frac):
    # 创建包含重复值的列表
    values = [1] * 10 + [2] * 10
    # 创建 DataFrame 对象
    df = DataFrame({"a": values, "b": values})

    # 对 DataFrame 进行分组并抽样，返回结果 DataFrame
    result = df.groupby("a").sample(n=n, frac=frac)
    # 创建预期的 DataFrame 对象
    values = [1] * 2 + [2] * 2
    expected = DataFrame({"a": values, "b": values}, index=result.index)
    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # 对 DataFrame 的某一列进行分组并抽样，返回结果 Series
    result = df.groupby("a")["b"].sample(n=n, frac=frac)
    # 创建预期的 Series 对象
    expected = Series(values, name="b", index=result.index)
    # 断言两个 Series 是否相等
    tm.assert_series_equal(result, expected)

# 定义测试函数 test_groupby_sample_unbalanced_groups_shape，测试 groupby.sample 方法
def test_groupby_sample_unbalanced_groups_shape():
    # 创建包含不同数量重复值的列表
    values = [1] * 10 + [2] * 20
    # 创建 DataFrame 对象
    df = DataFrame({"a": values, "b": values})

    # 对 DataFrame 进行分组并抽样，返回结果 DataFrame
    result = df.groupby("a").sample(n=5)
    # 创建预期的 DataFrame 对象
    values = [1] * 5 + [2] * 5
    expected = DataFrame({"a": values, "b": values}, index=result.index)
    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # 对 DataFrame 的某一列进行分组并抽样，返回结果 Series
    result = df.groupby("a")["b"].sample(n=5)
    # 创建预期的 Series 对象
    expected = Series(values, name="b", index=result.index)
    # 断言两个 Series 是否相等
    tm.assert_series_equal(result, expected)

# 定义测试函数 test_groupby_sample_index_value_spans_groups，测试 groupby.sample 方法
def test_groupby_sample_index_value_spans_groups():
    # 创建包含不同数量重复值的列表，并指定索引
    values = [1] * 3 + [2] * 3
    # 创建 DataFrame 对象
    df = DataFrame({"a": values, "b": values}, index=[1, 2, 2, 2, 2, 2])

    # 对 DataFrame 进行分组并抽样，返回结果 DataFrame
    result = df.groupby("a").sample(n=2)
    # 创建预期的 DataFrame 对象
    values = [1] * 2 + [2] * 2
    expected = DataFrame({"a": values, "b": values}, index=result.index)
    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # 对 DataFrame 的某一列进行分组并抽样，返回结果 Series
    result = df.groupby("a")["b"].sample(n=2)
    # 创建预期的 Series 对象
    expected = Series(values, name="b", index=result.index)
    # 断言两个 Series 是否相等
    tm.assert_series_equal(result, expected)

# 定义测试函数 test_groupby_sample_n_and_frac_raises，测试 groupby.sample 方法
def test_groupby_sample_n_and_frac_raises():
    # 创建 DataFrame 对象
    df = DataFrame({"a": [1, 2], "b": [1, 2]})
    # 定义错误消息
    msg = "Please enter a value for `frac` OR `n`, not both"

    # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        df.groupby("a").sample(n=1, frac=1.0)

    with pytest.raises(ValueError, match=msg):
        df.groupby("a")["b"].sample(n=1, frac=1.0)

# 定义测试函数 test_groupby_sample_frac_gt_one_without_replacement_raises，测试 groupby.sample 方法
def test_groupby_sample_frac_gt_one_without_replacement_raises():
    # 创建 DataFrame 对象
    df = DataFrame({"a": [1, 2], "b": [1, 2]})
    # 定义错误消息
    msg = "Replace has to be set to `True` when upsampling the population `frac` > 1."

    # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        df.groupby("a").sample(frac=1.5, replace=False)

    with pytest.raises(ValueError, match=msg):
        df.groupby("a")["b"].sample(frac=1.5, replace=False)

# 使用 pytest.mark.parametrize 装饰器，参数化测试用例
@pytest.mark.parametrize("n", [-1, 1.5])
# 定义测试函数 test_groupby_sample_invalid_n_raises，测试 groupby.sample 方法
def test_groupby_sample_invalid_n_raises(n):
    # 创建 DataFrame 对象
    df = DataFrame({"a": [1, 2], "b": [1, 2]})

    # 根据 n 的值定义不同的错误消息
    if n < 0:
        msg = "A negative number of rows requested. Please provide `n` >= 0."
    else:
        msg = "Only integers accepted as `n` values"

    # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        df.groupby("a").sample(n=n)

    with pytest.raises(ValueError, match=msg):
        df.groupby("a")["b"].sample(n=n)

# 定义测试函数 test_groupby_sample_oversample，测试 groupby.sample 方法
def test_groupby_sample_oversample():
    # 创建包含重复值的列表
    values = [1] * 10 + [2] * 10
    # 创建 DataFrame 对象
    df = DataFrame({"a": values, "b": values})
    # 对 DataFrame 进行按列 "a" 分组，并从每个分组中进行随机抽样，抽样比例为原样本的两倍，允许替换抽样
    result = df.groupby("a").sample(frac=2.0, replace=True)
    
    # 创建一个包含 20 个 1 和 20 个 2 的列表
    values = [1] * 20 + [2] * 20
    
    # 根据 values 列表创建一个 DataFrame，列名分别为 "a" 和 "b"，并使用 result 的索引
    expected = DataFrame({"a": values, "b": values}, index=result.index)
    
    # 使用 pandas.testing 模块中的 assert_frame_equal 函数比较 result 和 expected 的内容是否一致
    tm.assert_frame_equal(result, expected)
    
    # 对 DataFrame 进行按列 "a" 分组，并从每个分组中的列 "b" 进行随机抽样，抽样比例为原样本的两倍，允许替换抽样
    result = df.groupby("a")["b"].sample(frac=2.0, replace=True)
    
    # 根据 values 列表创建一个 Series，名称为 "b"，并使用 result 的索引
    expected = Series(values, name="b", index=result.index)
    
    # 使用 pandas.testing 模块中的 assert_series_equal 函数比较 result 和 expected 的内容是否一致
    tm.assert_series_equal(result, expected)
def test_groupby_sample_without_n_or_frac():
    # 创建一个包含重复元素的列表
    values = [1] * 10 + [2] * 10
    # 根据列表创建数据框，每列都使用 values 作为数据
    df = DataFrame({"a": values, "b": values})

    # 对数据框按列 'a' 进行分组，随机抽样，n 和 frac 均为 None
    result = df.groupby("a").sample(n=None, frac=None)
    # 创建预期的数据框，包含按 'a' 列分组后的索引
    expected = DataFrame({"a": [1, 2], "b": [1, 2]}, index=result.index)
    # 断言结果数据框与预期数据框相等
    tm.assert_frame_equal(result, expected)

    # 对数据框按列 'a' 进行分组，仅对 'b' 列进行随机抽样，n 和 frac 均为 None
    result = df.groupby("a")["b"].sample(n=None, frac=None)
    # 创建预期的序列，包含按 'a' 列分组后的索引
    expected = Series([1, 2], name="b", index=result.index)
    # 断言结果序列与预期序列相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "index, expected_index",
    [(["w", "x", "y", "z"], ["w", "w", "y", "y"]), ([3, 4, 5, 6], [3, 3, 5, 5])],
)
def test_groupby_sample_with_weights(index, expected_index):
    # GH 39927 - tests for integer index needed
    # 创建包含重复元素的列表和相应的数据框，使用整数或字符索引
    values = [1] * 2 + [2] * 2
    df = DataFrame({"a": values, "b": values}, index=Index(index))

    # 对数据框按列 'a' 进行分组，根据权重抽样，n=2，使用替换
    result = df.groupby("a").sample(n=2, replace=True, weights=[1, 0, 1, 0])
    # 创建预期的数据框，包含按 'a' 列分组后的索引
    expected = DataFrame({"a": values, "b": values}, index=Index(expected_index))
    # 断言结果数据框与预期数据框相等
    tm.assert_frame_equal(result, expected)

    # 对数据框按列 'a' 进行分组，仅对 'b' 列进行权重抽样，n=2，使用替换
    result = df.groupby("a")["b"].sample(n=2, replace=True, weights=[1, 0, 1, 0])
    # 创建预期的序列，包含按 'a' 列分组后的索引
    expected = Series(values, name="b", index=Index(expected_index))
    # 断言结果序列与预期序列相等
    tm.assert_series_equal(result, expected)


def test_groupby_sample_with_selections():
    # GH 39928
    # 创建一个包含重复元素的列表和相应的数据框
    values = [1] * 10 + [2] * 10
    df = DataFrame({"a": values, "b": values, "c": values})

    # 对数据框按列 'a' 进行分组，随机抽样，n 和 frac 均为 None
    result = df.groupby("a")[["b", "c"]].sample(n=None, frac=None)
    # 创建预期的数据框，包含按 'a' 列分组后的索引
    expected = DataFrame({"b": [1, 2], "c": [1, 2]}, index=result.index)
    # 断言结果数据框与预期数据框相等
    tm.assert_frame_equal(result, expected)


def test_groupby_sample_with_empty_inputs():
    # GH48459
    # 创建一个空的数据框
    df = DataFrame({"a": [], "b": []})
    # 对空的数据框按列 'a' 进行分组
    groupby_df = df.groupby("a")

    # 对分组后的数据框进行随机抽样
    result = groupby_df.sample()
    # 预期结果为原始空数据框
    expected = df
    # 断言结果数据框与预期数据框相等
    tm.assert_frame_equal(result, expected)
```