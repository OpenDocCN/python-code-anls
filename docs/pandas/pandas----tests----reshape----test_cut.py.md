# `D:\src\scipysrc\pandas\pandas\tests\reshape\test_cut.py`

```
# 导入 datetime 模块中的 datetime 类，用于处理日期和时间
from datetime import datetime

# 导入 numpy 库，用于科学计算中的数组操作
import numpy as np

# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 pandas 库，用于数据分析和处理
import pandas as pd

# 从 pandas 库中导入特定子模块和类
from pandas import (
    Categorical,           # 用于创建有序的分类数据
    DataFrame,             # 用于表示二维数据表格
    DatetimeIndex,         # 用于处理日期时间索引
    Index,                 # 用于表示索引对象
    Interval,              # 用于表示间隔对象
    IntervalIndex,         # 用于处理间隔索引
    Series,                # 用于表示一维数据序列
    TimedeltaIndex,        # 用于处理时间增量索引
    Timestamp,             # 用于表示时间戳
    cut,                   # 用于按照指定的间隔切分数据
    date_range,            # 用于生成日期范围
    interval_range,        # 用于生成间隔范围
    isna,                  # 用于检测缺失值
    qcut,                  # 用于按照分位数切分数据
    timedelta_range,       # 用于生成时间增量范围
    to_datetime,           # 用于将输入转换为 datetime 对象
)

# 导入 pandas 内部的测试工具模块
import pandas._testing as tm

# 从 pandas.api.types 模块中导入 CategoricalDtype 类
from pandas.api.types import CategoricalDtype

# 导入 pandas.core.reshape.tile 模块，用于数据重塑
import pandas.core.reshape.tile as tmod


# 定义测试函数 test_simple
def test_simple():
    # 创建包含 5 个 int64 类型的元素，值均为 1 的 numpy 数组
    data = np.ones(5, dtype="int64")
    # 使用 cut 函数将数据分成 4 个区间，返回区间的编号
    result = cut(data, 4, labels=False)

    # 创建期望结果，一个包含 5 个元素，值均为 1 的 numpy 数组
    expected = np.array([1, 1, 1, 1, 1])
    # 使用测试工具函数验证 result 是否等于 expected
    tm.assert_numpy_array_equal(result, expected, check_dtype=False)


# 使用 pytest 提供的参数化装饰器，多次运行同一个测试函数，每次使用不同的参数
@pytest.mark.parametrize("func", [list, np.array])
def test_bins(func):
    # 根据传入的 func 函数创建数据数组
    data = func([0.2, 1.4, 2.5, 6.2, 9.7, 2.1])
    # 使用 cut 函数将数据分成 3 个区间，同时返回区间编号和区间边界
    result, bins = cut(data, 3, retbins=True)

    # 根据 bins 创建 IntervalIndex 对象，表示区间范围
    intervals = IntervalIndex.from_breaks(bins.round(3))
    # 根据指定的索引值创建有序分类数据 expected
    expected = Categorical(intervals, ordered=True)

    # 使用测试工具函数验证 result 和 expected 是否相等
    tm.assert_categorical_equal(result, expected)
    # 使用测试工具函数验证 bins 和预期的 numpy 数组是否近似相等
    tm.assert_almost_equal(bins, np.array([0.1905, 3.36666667, 6.53333333, 9.7]))


# 定义测试函数 test_right
def test_right():
    # 创建包含浮点数元素的 numpy 数组 data
    data = np.array([0.2, 1.4, 2.5, 6.2, 9.7, 2.1, 2.575])
    # 使用 cut 函数将 data 分成 4 个区间，同时返回区间编号和区间边界
    result, bins = cut(data, 4, right=True, retbins=True)

    # 根据 bins 创建 IntervalIndex 对象，表示区间范围
    intervals = IntervalIndex.from_breaks(bins.round(3))
    # 根据指定的索引值创建有序分类数据 expected
    expected = Categorical(intervals, ordered=True)
    # 根据索引值创建 expected 对象，包含指定的区间索引
    expected = expected.take([0, 0, 0, 2, 3, 0, 0])

    # 使用测试工具函数验证 result 和 expected 是否相等
    tm.assert_categorical_equal(result, expected)
    # 使用测试工具函数验证 bins 和预期的 numpy 数组是否近似相等
    tm.assert_almost_equal(bins, np.array([0.1905, 2.575, 4.95, 7.325, 9.7]))


# 定义测试函数 test_no_right
def test_no_right():
    # 创建包含浮点数元素的 numpy 数组 data
    data = np.array([0.2, 1.4, 2.5, 6.2, 9.7, 2.1, 2.575])
    # 使用 cut 函数将 data 分成 4 个区间，同时返回区间编号和区间边界
    result, bins = cut(data, 4, right=False, retbins=True)

    # 根据 bins 创建 IntervalIndex 对象，表示区间范围，左闭右开
    intervals = IntervalIndex.from_breaks(bins.round(3), closed="left")
    # 根据指定的索引值创建有序分类数据 expected
    intervals = intervals.take([0, 0, 0, 2, 3, 0, 1])
    expected = Categorical(intervals, ordered=True)

    # 使用测试工具函数验证 result 和 expected 是否相等
    tm.assert_categorical_equal(result, expected)
    # 使用测试工具函数验证 bins 和预期的 numpy 数组是否近似相等
    tm.assert_almost_equal(bins, np.array([0.2, 2.575, 4.95, 7.325, 9.7095]))


# 定义测试函数 test_bins_from_interval_index
def test_bins_from_interval_index():
    # 使用 cut 函数将范围为 0 到 4 的整数切分成 3 个区间
    c = cut(range(5), 3)
    expected = c
    # 使用 cut 函数将范围为 0 到 4 的整数切分成 3 个区间，并与 expected 比较
    result = cut(range(5), bins=expected.categories)
    # 使用测试工具函数验证 result 和 expected 是否相等
    tm.assert_categorical_equal(result, expected)

    # 创建指定的分类类型 expected
    expected = Categorical.from_codes(
        np.append(c.codes, -1), categories=c.categories, ordered=True
    )
    # 使用 cut 函数将范围为 0 到 5 的整数切分成 expected 中定义的区间
    result = cut(range(6), bins=expected.categories)
    # 使用测试工具函数验证 result 和 expected 是否相等
    tm.assert_categorical_equal(result, expected)


# 定义测试函数 test_bins_from_interval_index_doc_example
def test_bins_from_interval_index_doc_example():
    # 创建年龄数据 ages
    ages = np.array([10, 15, 13, 12, 23, 25, 28, 59, 60])
    # 使用 cut 函数将 ages 数据按照给定的区间切分
    c = cut(ages, bins=[0, 18, 35, 70])

    # 预期的 IntervalIndex 对象 expected
    expected = IntervalIndex.from_tuples([(0, 18), (18, 35), (35, 70)])
    # 使用测试工具函数验证 c.categories 和 expected 是否相等
    tm.assert_index_equal(c.categories, expected)

    # 使用 cut 函数将 [25, 20, 50] 数据按照 c.categories 中的区间切分
    result = cut([25, 20, 50], bins=c.categories)
    # 使用测试工具函数验证 result.categories 和 expected 是否相等
    tm.assert_index_equal(result.categories, expected)
    # 使用测试工具函数验证 result.codes 和预期的 numpy 数组是否相等
    tm
    # 错误消息文本，用于匹配异常信息
    msg = "Overlapping IntervalIndex is not accepted"
    # 创建一个 IntervalIndex 对象，包含三个重叠的区间
    ii = IntervalIndex.from_tuples([(0, 10), (2, 12), (4, 14)])

    # 使用 pytest 的上下文管理器，检查是否会引发 ValueError 异常，并验证异常消息是否与 msg 匹配
    with pytest.raises(ValueError, match=msg):
        # 调用 cut 函数，传入参数 [5, 6] 和 bins=ii，期望引发异常
        cut([5, 6], bins=ii)
# 测试函数，用于检查切割函数 `cut()` 对于非单调递增的 bin 边界的处理是否正确
def test_bins_not_monotonic():
    # 错误消息，用于异常匹配
    msg = "bins must increase monotonically"
    # 测试数据，包含一个非单调递增的列表
    data = [0.2, 1.4, 2.5, 6.2, 9.7, 2.1]

    # 使用 pytest 的上下文管理器，期望捕获 ValueError 异常并匹配特定消息
    with pytest.raises(ValueError, match=msg):
        # 调用 cut() 函数，传入数据和非单调递增的 bin 边界 [0.1, 1.5, 1, 10]
        cut(data, [0.1, 1.5, 1, 10])


# 参数化测试函数，用于验证 cut() 函数对于不同情况下的处理是否正确
@pytest.mark.parametrize(
    "x, bins, expected",
    [
        (
            # 创建一个日期范围，从 "2017-12-31" 开始，3个时间点
            date_range("2017-12-31", periods=3),
            # 指定日期时间戳的最小值、指定日期 "2018-01-01"、日期时间戳的最大值为 bin 边界
            [Timestamp.min, Timestamp("2018-01-01"), Timestamp.max],
            # 从元组创建一个时间间隔索引，包含两个元组
            IntervalIndex.from_tuples(
                [
                    (Timestamp.min, Timestamp("2018-01-01")),
                    (Timestamp("2018-01-01"), Timestamp.max),
                ]
            ),
        ),
        (
            # 测试整数列表 [-1, 0, 1]
            [-1, 0, 1],
            # 使用 np.iinfo 获取整数数据类型的最小值、0、最大值作为 bin 边界
            np.array(
                [np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max], dtype="int64"
            ),
            # 创建整数类型的时间间隔索引，包含两个元组
            IntervalIndex.from_tuples(
                [(np.iinfo(np.int64).min, 0), (0, np.iinfo(np.int64).max)]
            ),
        ),
        (
            # 测试时间间隔数组
            [
                np.timedelta64(-1, "ns"),
                np.timedelta64(0, "ns"),
                np.timedelta64(1, "ns"),
            ],
            # 使用 np.iinfo 获取时间间隔数据类型的边界值作为 bin 边界
            np.array(
                [
                    np.timedelta64(-np.iinfo(np.int64).max, "ns"),
                    np.timedelta64(0, "ns"),
                    np.timedelta64(np.iinfo(np.int64).max, "ns"),
                ]
            ),
            # 创建时间间隔索引，包含两个元组
            IntervalIndex.from_tuples(
                [
                    (
                        np.timedelta64(-np.iinfo(np.int64).max, "ns"),
                        np.timedelta64(0, "ns"),
                    ),
                    (
                        np.timedelta64(0, "ns"),
                        np.timedelta64(np.iinfo(np.int64).max, "ns"),
                    ),
                ]
            ),
        ),
    ],
)
def test_bins_monotonic_not_overflowing(x, bins, expected):
    # GH 26045
    # 调用 cut() 函数，传入不同的 x, bins 参数组合，得到结果
    result = cut(x, bins)
    # 使用 assert_index_equal 检查结果的 categories 是否与预期的 expected 相等
    tm.assert_index_equal(result.categories, expected)


# 测试函数，用于检查切割函数 `cut()` 对于错误的标签数量的处理是否正确
def test_wrong_num_labels():
    # 错误消息，用于异常匹配
    msg = "Bin labels must be one fewer than the number of bin edges"
    # 测试数据，包含一个列表
    data = [0.2, 1.4, 2.5, 6.2, 9.7, 2.1]

    # 使用 pytest 的上下文管理器，期望捕获 ValueError 异常并匹配特定消息
    with pytest.raises(ValueError, match=msg):
        # 调用 cut() 函数，传入数据和 bin 边界 [0, 1, 10]，以及一个不符合要求的标签列表
        cut(data, [0, 1, 10], labels=["foo", "bar", "baz"])


# 参数化测试函数，用于验证 cut() 函数对于特殊情况下的处理是否正确
@pytest.mark.parametrize(
    "x,bins,msg",
    [
        # 测试空数组的情况
        ([], 2, "Cannot cut empty array"),
        # 测试 bins 参数为小数的情况
        ([1, 2, 3], 0.5, "`bins` should be a positive integer"),
    ],
)
def test_cut_corner(x, bins, msg):
    # 使用 pytest 的上下文管理器，期望捕获 ValueError 异常并匹配特定消息
    with pytest.raises(ValueError, match=msg):
        # 调用 cut() 函数，传入空数组或 bins 小于等于0.5 的情况
        cut(x, bins)


# 参数化测试函数，用于验证 cut() 和 qcut() 函数对于非一维输入的处理是否正确
@pytest.mark.parametrize("arg", [2, np.eye(2), DataFrame(np.eye(2))])
@pytest.mark.parametrize("cut_func", [cut, qcut])
def test_cut_not_1d_arg(arg, cut_func):
    # 错误消息，用于异常匹配
    msg = "Input array must be 1 dimensional"
    # 使用 pytest 的上下文管理器，期望捕获 ValueError 异常并匹配特定消息
    with pytest.raises(ValueError, match=msg):
        # 调用 cut_func 函数（cut 或 qcut），传入非一维数组 arg
        cut_func(arg, 2)


# 参数化测试函数，用于验证 cut() 函数对于包含无穷大的整数数组的处理是否正确
@pytest.mark.parametrize(
    "data",
    [
        [0, 1, 2, 3, 4, np.inf],
        [-np.inf, 0, 1, 2, 3, 4],
        [-np.inf, 0, 1, 2, 3, 4, np.inf],
    ],
)
def test_int_bins_with_inf(data):
    # GH 24314
    # 设置错误信息文本，用于匹配抛出的异常信息
    msg = "cannot specify integer `bins` when input data contains infinity"
    
    # 使用 pytest 的 `raises` 上下文管理器，断言在执行 cut(data, bins=3) 时会抛出 ValueError 异常，
    # 并且异常信息要与预设的 msg 文本匹配
    with pytest.raises(ValueError, match=msg):
        # 调用 cut 函数，传入参数 data 和 bins=3，期望此处抛出 ValueError 异常
        cut(data, bins=3)
# 测试超出范围的情况，参考问题 gh-1511

# 设置序列的名称为 "x"
name = "x"

# 创建一个 Series 对象，包含一组数据
ser = Series([0, -1, 0, 1, -3], name=name)

# 对序列进行分段，根据指定的断点 [0, 1]，返回分段后的索引
ind = cut(ser, [0, 1], labels=False)

# 创建预期结果的 Series 对象，包含期望的数据和名称
exp = Series([np.nan, np.nan, np.nan, 0, np.nan], name=name)

# 使用测试工具比较两个 Series 对象，确保它们相等
tm.assert_series_equal(ind, exp)


# 使用参数化测试框架，针对不同的参数组合进行测试

@pytest.mark.parametrize(
    "right,breaks,closed",
    [
        (True, [-1e-3, 0.25, 0.5, 0.75, 1], "right"),
        (False, [0, 0.25, 0.5, 0.75, 1 + 1e-3], "left"),
    ],
)
# 测试标签生成功能，使用不同的断点和闭合方式

# 创建一个包含重复元素的 NumPy 数组，用于测试
arr = np.tile(np.arange(0, 1.01, 0.1), 4)

# 调用 cut 函数对数组进行分段，根据参数设置返回分段结果和断点
result, bins = cut(arr, 4, retbins=True, right=right)

# 从给定的断点创建期望的 IntervalIndex 对象，指定闭合方式
ex_levels = IntervalIndex.from_breaks(breaks, closed=closed)

# 使用测试工具比较结果的分类对象的索引与期望的 IntervalIndex 对象
tm.assert_index_equal(result.categories, ex_levels)


# 测试确保将序列的名称传递给因子

# 设置序列的名称为 "foo"
name = "foo"

# 创建一个包含随机标准正态分布数据的 Series 对象
ser = Series(np.random.default_rng(2).standard_normal(100), name=name)

# 调用 cut 函数对序列进行分段，返回因子对象
factor = cut(ser, 4)

# 使用断言确保因子对象的名称与预期名称相同
assert factor.name == name


# 测试标签精度

# 创建一个 NumPy 数组，包含一组数据
arr = np.arange(0, 0.73, 0.01)

# 调用 cut 函数对数组进行分段，指定精度为 2
result = cut(arr, 4, precision=2)

# 从给定的断点创建期望的 IntervalIndex 对象
ex_levels = IntervalIndex.from_breaks([-0.00072, 0.18, 0.36, 0.54, 0.72])

# 使用测试工具比较结果的分类对象的索引与期望的 IntervalIndex 对象
tm.assert_index_equal(result.categories, ex_levels)


# 参数化测试，测试处理缺失值的情况

@pytest.mark.parametrize("labels", [None, False])
# 测试处理缺失值的情况，分段过程中出现缺失值

# 创建一个 NumPy 数组，包含一组数据
arr = np.arange(0, 0.75, 0.01)

# 将数组中每隔三个元素设为 NaN
arr[::3] = np.nan

# 调用 cut 函数对数组进行分段，指定标签选项
result = cut(arr, 4, labels=labels)

# 将结果转换为 NumPy 数组
result = np.asarray(result)

# 创建期望的结果数组，根据原数组的 NaN 值设定相应位置的值
expected = np.where(isna(arr), np.nan, result)

# 使用测试工具比较两个数组，确保它们几乎相等
tm.assert_almost_equal(result, expected)


# 测试处理无穷值的情况

# 创建一个包含整数数据的 NumPy 数组
data = np.arange(6)

# 创建一个包含整数数据的 Series 对象
data_ser = Series(data, dtype="int64")

# 设置分段的断点
bins = [-np.inf, 2, 4, np.inf]

# 调用 cut 函数对数组和 Series 对象进行分段，指定断点
result = cut(data, bins)
result_ser = cut(data_ser, bins)

# 从给定的断点创建期望的 IntervalIndex 对象
ex_uniques = IntervalIndex.from_breaks(bins)

# 使用测试工具比较结果的分类对象的索引与期望的 IntervalIndex 对象
tm.assert_index_equal(result.categories, ex_uniques)

# 使用断言检查特定索引位置的结果是否符合预期的区间
assert result[5] == Interval(4, np.inf)
assert result[0] == Interval(-np.inf, 2)
assert result_ser[5] == Interval(4, np.inf)
assert result_ser[0] == Interval(-np.inf, 2)


# 测试超出范围的情况

# 创建一个包含随机标准正态分布数据的 NumPy 数组
arr = np.random.default_rng(2).standard_normal(100)

# 调用 cut 函数对数组进行分段，指定断点
result = cut(arr, [-1, 0, 1])

# 创建预期的结果掩码数组，标记超出指定断点范围的位置
mask = isna(result)
ex_mask = (arr < -1) | (arr > 1)

# 使用测试工具比较两个数组，确保它们相等
tm.assert_numpy_array_equal(mask, ex_mask)


# 参数化测试，测试将标签传递给 cut 函数

@pytest.mark.parametrize(
    "get_labels,get_expected",
    [
        (
            lambda labels: labels,
            lambda labels: Categorical(
                ["Medium"] + 4 * ["Small"] + ["Medium", "Large"],
                categories=labels,
                ordered=True,
            ),
        ),
        (
            lambda labels: Categorical.from_codes([0, 1, 2], labels),
            lambda labels: Categorical.from_codes([1] + 4 * [0] + [1, 2], labels),
        ),
    ],
)
# 测试确保标签在分段过程中传递给 cut 函数

# 设置分段的断点和一组数据
bins = [0, 25, 50, 100]
arr = [50, 5, 10, 15, 20, 30, 70]

# 设置标签列表
labels = ["Small", "Medium", "Large"]

# 调用 cut 函数对数据进行分段，根据参数化的函数获取标签
result = cut(arr, bins, labels=get_labels(labels))

# 使用测试工具比较分类对象，确保其与预期的分类对象相等
tm.assert_categorical_equal(result, get_expected(labels))


# 测试兼容性，确保标签能够顺利传递给 cut 函数

# 参考问题 gh-16459
    # 创建一个包含整数的列表
    arr = [50, 5, 10, 15, 20, 30, 70]
    # 创建一个字符串列表，用于标记切分后的区间
    labels = ["Good", "Medium", "Bad"]
    
    # 使用 cut 函数对 arr 列表进行分段，分成 3 个区间，并使用 labels 参数指定标签
    result = cut(arr, 3, labels=labels)
    
    # 使用 cut 函数对 arr 列表进行分段，分成 3 个区间，并使用 Categorical 类构造带有指定标签的有序分类数据
    exp = cut(arr, 3, labels=Categorical(labels, categories=labels, ordered=True))
    
    # 使用 tm.assert_categorical_equal 函数断言 result 和 exp 的分类结果相等
    tm.assert_categorical_equal(result, exp)
@pytest.mark.parametrize("x", [np.arange(11.0), np.arange(11.0) / 1e10])
def test_round_frac_just_works(x):
    # 对函数 cut() 进行简单的功能测试
    cut(x, 2)


@pytest.mark.parametrize(
    "val,precision,expected",
    [
        (-117.9998, 3, -118),  # 测试负数的小数点精度取整
        (117.9998, 3, 118),   # 测试正数的小数点精度取整
        (117.9998, 2, 118),   # 测试不同的小数点精度取整
        (0.000123456, 2, 0.00012),  # 测试小数的小数点精度取整
    ],
)
def test_round_frac(val, precision, expected):
    # 测试 tmod 模块中 _round_frac() 函数的精度取整功能
    result = tmod._round_frac(val, precision=precision)
    assert result == expected


def test_cut_return_intervals():
    ser = Series([0, 1, 2, 3, 4, 5, 6, 7, 8])
    result = cut(ser, 3)

    exp_bins = np.linspace(0, 8, num=4).round(3)
    exp_bins[0] -= 0.008

    expected = Series(
        IntervalIndex.from_breaks(exp_bins, closed="right").take(
            [0, 0, 0, 1, 1, 1, 2, 2, 2]
        )
    ).astype(CategoricalDtype(ordered=True))
    tm.assert_series_equal(result, expected)


def test_series_ret_bins():
    # 测试 cut() 函数返回值为 bins 的情况
    ser = Series(np.arange(4))
    result, bins = cut(ser, 2, retbins=True)

    expected = Series(
        IntervalIndex.from_breaks([-0.003, 1.5, 3], closed="right").repeat(2)
    ).astype(CategoricalDtype(ordered=True))
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"duplicates": "drop"}, None),  # 测试 drop 重复 bin edges 的情况
        ({}, "Bin edges must be unique"),  # 测试没有指定 duplicates 时的重复 bin edges 错误
        ({"duplicates": "raise"}, "Bin edges must be unique"),  # 测试指定 raise 时的重复 bin edges 错误
        ({"duplicates": "foo"}, "invalid value for 'duplicates' parameter"),  # 测试不合法的 duplicates 参数值
    ],
)
def test_cut_duplicates_bin(kwargs, msg):
    # 测试 cut() 函数处理重复 bin edges 的情况
    bins = [0, 2, 4, 6, 10, 10]
    values = Series(np.array([1, 3, 5, 7, 9]), index=["a", "b", "c", "d", "e"])

    if msg is not None:
        with pytest.raises(ValueError, match=msg):
            cut(values, bins, **kwargs)
    else:
        result = cut(values, bins, **kwargs)
        expected = cut(values, pd.unique(np.asarray(bins)))
        tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("data", [9.0, -9.0, 0.0])
@pytest.mark.parametrize("length", [1, 2])
def test_single_bin(data, length):
    # 测试 cut() 函数对单个 bin 的处理
    ser = Series([data] * length)
    result = cut(ser, 1, labels=False)

    expected = Series([0] * length, dtype=np.intp)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "array_1_writeable,array_2_writeable", [(True, True), (True, False), (False, False)]
)
def test_cut_read_only(array_1_writeable, array_2_writeable):
    # 测试 cut() 函数对只读数组的处理
    array_1 = np.arange(0, 100, 10)
    array_1.flags.writeable = array_1_writeable

    array_2 = np.arange(0, 100, 10)
    array_2.flags.writeable = array_2_writeable

    hundred_elements = np.arange(100)
    tm.assert_categorical_equal(
        cut(hundred_elements, array_1), cut(hundred_elements, array_2)
    )


@pytest.mark.parametrize(
    "conv",
    [
        lambda v: Timestamp(v),  # 测试转换为 Timestamp 的函数 conv
        lambda v: to_datetime(v),  # 测试转换为 datetime 的函数 conv
        lambda v: np.datetime64(v),  # 测试转换为 np.datetime64 的函数 conv
        lambda v: Timestamp(v).to_pydatetime(),  # 测试转换为 Python datetime 的函数 conv
    ],
)
def test_datetime_bin(conv):
    # 定义测试函数 test_datetime_bin，接受一个转换器 conv 参数

    data = [np.datetime64("2012-12-13"), np.datetime64("2012-12-15")]
    # 创建一个包含两个 numpy 日期时间对象的列表

    bin_data = ["2012-12-12", "2012-12-14", "2012-12-16"]
    # 创建一个日期时间字符串列表 bin_data

    expected = Series(
        IntervalIndex(
            [
                Interval(Timestamp(bin_data[0]), Timestamp(bin_data[1])),
                Interval(Timestamp(bin_data[1]), Timestamp(bin_data[2])),
            ]
        )
    )
    # 创建一个预期的 Pandas Series 对象，其中包含 IntervalIndex，用于表示时间区间
    # 区间由 bin_data 中的日期时间字符串转换而来

    bins = [conv(v) for v in bin_data]
    # 调用 conv 函数，将 bin_data 中的日期时间字符串转换为指定类型的 bins

    result = Series(cut(data, bins=bins))
    # 使用 cut 函数根据 bins 切分 data，生成结果 Series 对象

    if type(bins[0]) is datetime:
        # 如果 bins 的元素类型为 datetime
        # 则设置预期的类型为微秒精度的时间间隔
        expected = expected.astype("interval[datetime64[us]]")

    expected = expected.astype(CategoricalDtype(ordered=True))
    # 将预期结果转换为有序的分类类型

    tm.assert_series_equal(result, expected)
    # 使用 pytest 的 tm 模块断言 result 与 expected 的 Series 对象相等


@pytest.mark.parametrize("box", [Series, Index, np.array, list])
def test_datetime_cut(unit, box):
    # 定义测试函数 test_datetime_cut，接受单位 unit 和集合类型 box 参数
    # 使用 pytest 的 parametrize 装饰器，传入不同的 box 参数进行测试

    # see gh-14714
    #
    # Testing time data when it comes in various collection types.

    data = to_datetime(["2013-01-01", "2013-01-02", "2013-01-03"]).astype(f"M8[{unit}]")
    # 将日期时间字符串列表转换为 Pandas 的 DatetimeIndex 对象，并设置为 unit 单位类型
    data = box(data)
    # 将 data 转换为指定的集合类型 box

    result, _ = cut(data, 3, retbins=True)
    # 使用 cut 函数将 data 切分为 3 个区间，并返回结果及 bins

    if unit == "s":
        # 如果 unit 是秒
        # 参见 https://github.com/pandas-dev/pandas/pull/56101#discussion_r1405325425
        # 为什么我们要四舍五入到 8 秒而不是 7 秒
        left = DatetimeIndex(
            ["2012-12-31 23:57:08", "2013-01-01 16:00:00", "2013-01-02 08:00:00"],
            dtype=f"M8[{unit}]",
        )
    else:
        left = DatetimeIndex(
            [
                "2012-12-31 23:57:07.200000",
                "2013-01-01 16:00:00",
                "2013-01-02 08:00:00",
            ],
            dtype=f"M8[{unit}]",
        )
    # 创建左边界 DatetimeIndex 对象，根据 unit 设置不同的精度

    right = DatetimeIndex(
        ["2013-01-01 16:00:00", "2013-01-02 08:00:00", "2013-01-03 00:00:00"],
        dtype=f"M8[{unit}]",
    )
    # 创建右边界 DatetimeIndex 对象，根据 unit 设置不同的精度

    exp_intervals = IntervalIndex.from_arrays(left, right)
    # 创建预期的区间索引对象，使用左右边界

    expected = Series(exp_intervals).astype(CategoricalDtype(ordered=True))
    # 创建预期的 Series 对象，包含有序的分类数据类型

    tm.assert_series_equal(Series(result), expected)
    # 使用 tm 模块断言 result 的 Series 对象与 expected 相等


@pytest.mark.parametrize("box", [list, np.array, Index, Series])
def test_datetime_tz_cut_mismatched_tzawareness(box):
    # 定义测试函数 test_datetime_tz_cut_mismatched_tzawareness，接受集合类型 box 参数
    # 用于测试不匹配的时区感知问题

    bins = box(
        [
            Timestamp("2013-01-01 04:57:07.200000"),
            Timestamp("2013-01-01 21:00:00"),
            Timestamp("2013-01-02 13:00:00"),
            Timestamp("2013-01-03 05:00:00"),
        ]
    )
    # 创建 bins，包含四个 Timestamp 对象，代表时刻

    ser = Series(date_range("20130101", periods=3, tz="US/Eastern"))
    # 创建一个包含三个具有美国东部时区的日期时间序列 Series 对象

    msg = "Cannot use timezone-naive bins with timezone-aware values"
    # 设置错误消息

    with pytest.raises(ValueError, match=msg):
        cut(ser, bins)
    # 使用 pytest 的 raises 断言捕获 ValueError 异常，并匹配错误消息


@pytest.mark.parametrize(
    "bins",
    [
        3,
        [
            Timestamp("2013-01-01 04:57:07.200000", tz="UTC").tz_convert("US/Eastern"),
            Timestamp("2013-01-01 21:00:00", tz="UTC").tz_convert("US/Eastern"),
            Timestamp("2013-01-02 13:00:00", tz="UTC").tz_convert("US/Eastern"),
            Timestamp("2013-01-03 05:00:00", tz="UTC").tz_convert("US/Eastern"),
        ],
    ],
)
# 使用 pytest 的 parametrize 装饰器，传入不同的 bins 参数进行测试
@pytest.mark.parametrize("box", [list, np.array, Index, Series])
def test_datetime_tz_cut(bins, box):
    # 为了解决问题 gh-19872 进行测试

    # 创建一个带有指定时区的时间序列
    tz = "US/Eastern"
    ser = Series(date_range("20130101", periods=3, tz=tz))

    # 如果 bins 不是整数，将其转换为指定类型（list, np.array, Index, Series）
    if not isinstance(bins, int):
        bins = box(bins)

    # 使用 cut 函数对时间序列进行分箱操作
    result = cut(ser, bins)

    # 创建一个时间区间索引对象 ii，包含三个时间区间
    ii = IntervalIndex(
        [
            Interval(
                Timestamp("2012-12-31 23:57:07.200000", tz=tz),
                Timestamp("2013-01-01 16:00:00", tz=tz),
            ),
            Interval(
                Timestamp("2013-01-01 16:00:00", tz=tz),
                Timestamp("2013-01-02 08:00:00", tz=tz),
            ),
            Interval(
                Timestamp("2013-01-02 08:00:00", tz=tz),
                Timestamp("2013-01-03 00:00:00", tz=tz),
            ),
        ]
    )

    # 如果 bins 是整数，调整 ii 的数据类型以匹配结果的时间单位
    if isinstance(bins, int):
        ii = ii.astype("interval[datetime64[ns, US/Eastern]]")

    # 创建预期的 Series 对象，数据类型为有序的分类数据类型
    expected = Series(ii).astype(CategoricalDtype(ordered=True))

    # 断言分箱操作的结果与预期结果相等
    tm.assert_series_equal(result, expected)


def test_datetime_nan_error():
    # 定义一个错误消息
    msg = "bins must be of datetime64 dtype"

    # 使用 pytest 的上下文管理器检查是否会引发 ValueError，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        cut(date_range("20130101", periods=3), bins=[0, 2, 4])


def test_datetime_nan_mask():
    # 对日期时间序列进行分箱操作，并保存结果
    result = cut(
        date_range("20130102", periods=5), bins=date_range("20130101", periods=2)
    )

    # 创建一个掩码，检查结果的类别是否包含 NaN 值
    mask = result.categories.isna()
    tm.assert_numpy_array_equal(mask, np.array([False]))

    # 创建一个掩码，检查结果中是否存在 NaN 值
    mask = result.isna()
    tm.assert_numpy_array_equal(mask, np.array([False, True, True, True, True]))


@pytest.mark.parametrize("tz", [None, "UTC", "US/Pacific"])
def test_datetime_cut_roundtrip(tz, unit):
    # 进行回转测试，解决问题 gh-19891

    # 创建一个带有时区和时间单位的时间序列
    ser = Series(date_range("20180101", periods=3, tz=tz, unit=unit))

    # 使用 cut 函数对时间序列进行分箱操作，并返回结果及分箱的时间点
    result, result_bins = cut(ser, 2, retbins=True)

    # 根据分箱后的结果再次进行分箱操作，创建预期的 Series 对象
    expected = cut(ser, result_bins)

    # 根据时间单位选择性地调整预期的分箱时间点
    if unit == "s":
        # 当时间单位为秒时，将时间序列的第一个条目四舍五入到最接近的8秒内，而不是7秒
        # 详细原因见链接 https://github.com/pandas-dev/pandas/pull/56101#discussion_r1405325425
        expected_bins = DatetimeIndex(
            ["2017-12-31 23:57:08", "2018-01-02 00:00:00", "2018-01-03 00:00:00"],
            dtype=f"M8[{unit}]",
        )
    else:
        expected_bins = DatetimeIndex(
            [
                "2017-12-31 23:57:07.200000",
                "2018-01-02 00:00:00",
                "2018-01-03 00:00:00",
            ],
            dtype=f"M8[{unit}]",
        )

    # 将预期的时间点转换为指定时区
    expected_bins = expected_bins.tz_localize(tz)

    # 断言分箱的结果及分箱的时间点与预期相等
    tm.assert_series_equal(result, expected)
    tm.assert_index_equal(result_bins, expected_bins)


def test_timedelta_cut_roundtrip():
    # 进行回转测试，解决问题 gh-19891

    # 创建一个时间增量序列
    ser = Series(timedelta_range("1day", periods=3))

    # 使用 cut 函数对时间增量序列进行分箱操作，并返回结果及分箱的时间点
    result, result_bins = cut(ser, 2, retbins=True)

    # 根据分箱后的结果再次进行分箱操作，创建预期的 Series 对象
    expected = cut(ser, result_bins)
    # 使用 pandas 测试框架中的方法来比较两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)
    
    # 创建一个 TimedeltaIndex 对象，包含指定的时间间隔字符串列表
        expected_bins = TimedeltaIndex(
            ["0 days 23:57:07.200000", "2 days 00:00:00", "3 days 00:00:00"]
        )
    
    # 使用 pandas 测试框架中的方法来比较两个 Index 对象是否相等
        tm.assert_index_equal(result_bins, expected_bins)
@pytest.mark.parametrize("bins", [6, 7])
@pytest.mark.parametrize(
    "box, compare",
    [
        (Series, tm.assert_series_equal),  # 使用 Series 类型和 tm.assert_series_equal 函数作为参数
        (np.array, tm.assert_categorical_equal),  # 使用 np.array 类型和 tm.assert_categorical_equal 函数作为参数
        (list, tm.assert_equal),  # 使用 list 类型和 tm.assert_equal 函数作为参数
    ],
)
def test_cut_bool_coercion_to_int(bins, box, compare):
    # issue 20303
    # 创建两个测试数据集，一个包含布尔值，一个包含整数
    data_expected = box([0, 1, 1, 0, 1] * 10)
    data_result = box([False, True, True, False, True] * 10)
    # 对每个数据集应用 cut 函数，生成预期和实际结果
    expected = cut(data_expected, bins, duplicates="drop")
    result = cut(data_result, bins, duplicates="drop")
    # 使用给定的比较函数比较结果和预期值
    compare(result, expected)


@pytest.mark.parametrize("labels", ["foo", 1, True])
def test_cut_incorrect_labels(labels):
    # GH 13318
    # 创建一个测试数据集和错误消息
    values = range(5)
    msg = "Bin labels must either be False, None or passed in as a list-like argument"
    # 使用 pytest 的断言来测试 cut 函数对于不合规标签的处理
    with pytest.raises(ValueError, match=msg):
        cut(values, 4, labels=labels)


@pytest.mark.parametrize("bins", [3, [0, 5, 15]])
@pytest.mark.parametrize("right", [True, False])
@pytest.mark.parametrize("include_lowest", [True, False])
def test_cut_nullable_integer(bins, right, include_lowest):
    # 创建一个随机整数数组，将其中一半值设为 NaN
    a = np.random.default_rng(2).integers(0, 10, size=50).astype(float)
    a[::2] = np.nan
    # 使用 cut 函数处理包含 NaN 的整数数组
    result = cut(
        pd.array(a, dtype="Int64"), bins, right=right, include_lowest=include_lowest
    )
    expected = cut(a, bins, right=right, include_lowest=include_lowest)
    # 使用 tm.assert_categorical_equal 断言比较结果和预期
    tm.assert_categorical_equal(result, expected)


@pytest.mark.parametrize(
    "data, bins, labels, expected_codes, expected_labels",
    [
        ([15, 17, 19], [14, 16, 18, 20], ["A", "B", "A"], [0, 1, 0], ["A", "B"]),
        ([1, 3, 5], [0, 2, 4, 6, 8], [2, 0, 1, 2], [2, 0, 1], [0, 1, 2]),
    ],
)
def test_cut_non_unique_labels(data, bins, labels, expected_codes, expected_labels):
    # GH 33141
    # 使用 cut 函数处理包含非唯一标签的数据
    result = cut(data, bins=bins, labels=labels, ordered=False)
    expected = Categorical.from_codes(
        expected_codes, categories=expected_labels, ordered=False
    )
    # 使用 tm.assert_categorical_equal 断言比较结果和预期
    tm.assert_categorical_equal(result, expected)


@pytest.mark.parametrize(
    "data, bins, labels, expected_codes, expected_labels",
    [
        ([15, 17, 19], [14, 16, 18, 20], ["C", "B", "A"], [0, 1, 2], ["C", "B", "A"]),
        ([1, 3, 5], [0, 2, 4, 6, 8], [3, 0, 1, 2], [0, 1, 2], [3, 0, 1, 2]),
    ],
)
def test_cut_unordered_labels(data, bins, labels, expected_codes, expected_labels):
    # GH 33141
    # 使用 cut 函数处理包含无序标签的数据
    result = cut(data, bins=bins, labels=labels, ordered=False)
    expected = Categorical.from_codes(
        expected_codes, categories=expected_labels, ordered=False
    )
    # 使用 tm.assert_categorical_equal 断言比较结果和预期
    tm.assert_categorical_equal(result, expected)


def test_cut_unordered_with_missing_labels_raises_error():
    # GH 33141
    # 测试在没有提供有序标签的情况下，使用 cut 函数是否会引发错误
    msg = "'labels' must be provided if 'ordered = False'"
    with pytest.raises(ValueError, match=msg):
        cut([0.5, 3], bins=[0, 1, 2], ordered=False)


def test_cut_unordered_with_series_labels():
    # https://github.com/pandas-dev/pandas/issues/36603
    # 创建一个包含整数的 Series 对象
    ser = Series([1, 2, 3, 4, 5])
    bins = Series([0, 2, 4, 6])
    # 创建一个包含字符串列表的 Pandas Series，用作切割数据的标签
    labels = Series(["a", "b", "c"])
    # 调用 cut 函数对 ser 进行分段切割，使用指定的 bins 和 labels 参数，并设置无序（ordered=False）
    result = cut(ser, bins=bins, labels=labels, ordered=False)
    # 创建一个预期的 Pandas Series，包含预期的分类结果
    expected = Series(["a", "a", "b", "b", "c"], dtype="category")
    # 使用测试模块 tm 的 assert_series_equal 函数比较 result 和 expected 两个 Series 是否相等
    tm.assert_series_equal(result, expected)
def test_cut_no_warnings():
    # 创建一个 DataFrame，包含一个随机整数列 "value"，范围在 [0, 100)，共 20 个数
    df = DataFrame({"value": np.random.default_rng(2).integers(0, 100, 20)})
    # 创建一个标签列表，每个标签表示一个区间，如 "0 - 9", "10 - 19", ..., "90 - 99"
    labels = [f"{i} - {i + 9}" for i in range(0, 100, 10)]
    # 使用上下文管理器禁止断言产生警告
    with tm.assert_produces_warning(False):
        # 对 "value" 列进行切分，根据给定的区间和标签，创建新的 "group" 列
        df["group"] = cut(df.value, range(0, 105, 10), right=False, labels=labels)


def test_cut_with_duplicated_index_lowest_included():
    # GH 42185
    # 创建一个预期的 Series，包含具有重复索引的区间对象
    expected = Series(
        [Interval(-0.001, 2, closed="right")] * 3
        + [Interval(2, 4, closed="right"), Interval(-0.001, 2, closed="right")],
        index=[0, 1, 2, 3, 0],
        dtype="category",
    ).cat.as_ordered()

    # 创建一个具有重复索引的 Series
    ser = Series([0, 1, 2, 3, 0], index=[0, 1, 2, 3, 0])
    # 对 Series 进行切分，生成预期的结果
    result = cut(ser, bins=[0, 2, 4], include_lowest=True)
    # 断言切分结果与预期结果相等
    tm.assert_series_equal(result, expected)


def test_cut_with_nonexact_categorical_indices():
    # GH 42424

    # 创建一个 Series 包含 0 到 99 的整数
    ser = Series(range(100))
    # 对 Series 进行切分，然后统计每个区间的计数，并选择前 5 个和后 5 个
    ser1 = cut(ser, 10).value_counts().head(5)
    ser2 = cut(ser, 10).value_counts().tail(5)
    # 创建一个 DataFrame，包含两个列，列名为 "1" 和 "2"，分别存储 ser1 和 ser2 的统计结果
    result = DataFrame({"1": ser1, "2": ser2})

    # 创建一个有序的分类索引，包含预定义的区间对象
    index = pd.CategoricalIndex(
        [
            Interval(-0.099, 9.9, closed="right"),
            Interval(9.9, 19.8, closed="right"),
            Interval(19.8, 29.7, closed="right"),
            Interval(29.7, 39.6, closed="right"),
            Interval(39.6, 49.5, closed="right"),
            Interval(49.5, 59.4, closed="right"),
            Interval(59.4, 69.3, closed="right"),
            Interval(69.3, 79.2, closed="right"),
            Interval(79.2, 89.1, closed="right"),
            Interval(89.1, 99, closed="right"),
        ],
        ordered=True,
    )

    # 创建预期的 DataFrame，包含与 result 相同的行列结构，但值为统计结果和 NaN 组成
    expected = DataFrame(
        {"1": [10] * 5 + [np.nan] * 5, "2": [np.nan] * 5 + [10] * 5}, index=index
    )

    # 断言 result 和 expected 的 DataFrame 结构和内容相等
    tm.assert_frame_equal(expected, result)


def test_cut_with_timestamp_tuple_labels():
    # GH 40661
    # 创建一个标签列表，包含 Timestamp 对象的元组
    labels = [(Timestamp(10),), (Timestamp(20),), (Timestamp(30),)]
    # 对数值列表进行切分，使用给定的区间和标签
    result = cut([2, 4, 6], bins=[1, 3, 5, 7], labels=labels)

    # 创建一个预期的有序分类对象，用于与切分结果进行比较
    expected = Categorical.from_codes([0, 1, 2], labels, ordered=True)
    # 断言切分结果和预期结果的有序分类对象相等
    tm.assert_categorical_equal(result, expected)


def test_cut_bins_datetime_intervalindex():
    # https://github.com/pandas-dev/pandas/issues/46218
    # 创建一个时间间隔范围，包含从指定日期到指定日期的每一天
    bins = interval_range(Timestamp("2022-02-25"), Timestamp("2022-02-27"), freq="1D")
    # 将 Series 转换为指定的日期时间类型，以触发 bug
    result = cut(Series([Timestamp("2022-02-26")]).astype("M8[ns]"), bins=bins)
    # 创建一个预期的有序分类对象，用于与切分结果进行比较
    expected = Categorical.from_codes([0], bins, ordered=True)
    # 断言切分结果的分类对象和预期结果的分类对象相等
    tm.assert_categorical_equal(result.array, expected)


def test_cut_with_nullable_int64():
    # GH 30787
    # 创建一个包含可空 Int64 类型的 Series
    series = Series([0, 1, 2, 3, 4, pd.NA, 6, 7], dtype="Int64")
    # 创建一个固定区间的索引对象
    bins = [0, 2, 4, 6, 8]
    intervals = IntervalIndex.from_breaks(bins)

    # 创建一个预期的 Series，包含从整数到对应区间的分类
    expected = Series(
        Categorical.from_codes([-1, 0, 0, 1, 1, -1, 2, 3], intervals, ordered=True)
    )

    # 对可空整数 Series 进行切分，生成预期的分类结果
    result = cut(series, bins=bins)

    # 断言切分结果和预期结果的 Series 结构和内容相等
    tm.assert_series_equal(result, expected)


def test_cut_datetime_array_no_attributeerror():
    # GH 55431
    # 创建一个时间序列对象 `ser`，包含两个日期时间字符串
    ser = Series(to_datetime(["2023-10-06 12:00:00+0000", "2023-10-07 12:00:00+0000"]))
    
    # 使用 `cut` 函数对时间序列的数组部分进行分割，分成两个区间
    result = cut(ser.array, bins=2)
    
    # 从分割结果中获取分类信息
    categories = result.categories
    
    # 根据指定的代码创建预期的分类对象 `expected`，以保证顺序和数据类型的一致性
    expected = Categorical.from_codes([0, 1], categories=categories, ordered=True)
    
    # 使用 `tm.assert_categorical_equal` 断言函数，验证 `result` 和 `expected` 相等性
    tm.assert_categorical_equal(
        result, expected, check_dtype=True, check_category_order=True
    )
```