# `D:\src\scipysrc\pandas\pandas\tests\reshape\test_qcut.py`

```
# 导入必要的模块
import os  # 导入操作系统接口模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架

import pandas as pd  # 导入Pandas库
from pandas import (  # 从Pandas中导入多个子模块和类
    Categorical,
    DatetimeIndex,
    Interval,
    IntervalIndex,
    NaT,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    cut,
    date_range,
    isna,
    qcut,
    timedelta_range,
)
import pandas._testing as tm  # 导入Pandas的测试工具模块
from pandas.api.types import CategoricalDtype  # 导入Pandas的分类数据类型

from pandas.tseries.offsets import Day  # 从Pandas的时间序列偏移模块导入Day偏移量


def test_qcut():
    arr = np.random.default_rng(2).standard_normal(1000)  # 生成1000个标准正态分布的随机数数组

    # 计算分位数区间，并返回区间的标签和边界
    labels, _ = qcut(arr, 4, retbins=True)
    ex_bins = np.quantile(arr, [0, 0.25, 0.5, 0.75, 1.0])  # 计算分位数

    result = labels.categories.left.values  # 获取分位数区间的左边界值
    assert np.allclose(result, ex_bins[:-1], atol=1e-2)  # 断言左边界值与预期值接近

    result = labels.categories.right.values  # 获取分位数区间的右边界值
    assert np.allclose(result, ex_bins[1:], atol=1e-2)  # 断言右边界值与预期值接近

    ex_levels = cut(arr, ex_bins, include_lowest=True)  # 使用指定的分位数区间进行分段
    tm.assert_categorical_equal(labels, ex_levels)  # 使用Pandas的测试工具断言分类对象相等


def test_qcut_bounds():
    arr = np.random.default_rng(2).standard_normal(1000)  # 生成1000个标准正态分布的随机数数组

    factor = qcut(arr, 10, labels=False)  # 使用10个等分位数进行分段，返回标签序号
    assert len(np.unique(factor)) == 10  # 断言分段后的唯一标签数为10


def test_qcut_specify_quantiles():
    arr = np.random.default_rng(2).standard_normal(100)  # 生成100个标准正态分布的随机数数组
    factor = qcut(arr, [0, 0.25, 0.5, 0.75, 1.0])  # 使用指定的分位数进行分段

    expected = qcut(arr, 4)  # 使用默认的4个等分位数进行分段
    tm.assert_categorical_equal(factor, expected)  # 使用Pandas的测试工具断言分类对象相等


def test_qcut_all_bins_same():
    with pytest.raises(ValueError, match="edges.*unique"):  # 断言抛出值错误并匹配特定消息
        qcut([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3)  # 对相同值列表进行分段，期望引发错误


def test_qcut_include_lowest():
    values = np.arange(10)  # 生成0到9的数组
    ii = qcut(values, 4)  # 使用4个等分位数对数组进行分段

    ex_levels = IntervalIndex(  # 创建预期的区间索引
        [
            Interval(-0.001, 2.25),
            Interval(2.25, 4.5),
            Interval(4.5, 6.75),
            Interval(6.75, 9),
        ]
    )
    tm.assert_index_equal(ii.categories, ex_levels)  # 使用Pandas的测试工具断言区间索引相等


def test_qcut_nas():
    arr = np.random.default_rng(2).standard_normal(100)  # 生成100个标准正态分布的随机数数组
    arr[:20] = np.nan  # 将前20个元素设置为NaN（缺失值）

    result = qcut(arr, 4)  # 使用4个等分位数对数组进行分段
    assert isna(result[:20]).all()  # 断言前20个分段结果为NaN（缺失值）


def test_qcut_index():
    result = qcut([0, 2], 2)  # 对给定数组使用2个等分位数进行分段

    intervals = [Interval(-0.001, 1), Interval(1, 2)]  # 创建预期的区间列表
    expected = Categorical(intervals, ordered=True)  # 创建预期的有序分类对象
    tm.assert_categorical_equal(result, expected)  # 使用Pandas的测试工具断言分类对象相等


def test_qcut_binning_issues(datapath):
    # 查看Github问题编号为1978和1979
    cut_file = datapath(os.path.join("reshape", "data", "cut_data.csv"))  # 获取数据文件路径
    arr = np.loadtxt(cut_file)  # 加载数据文件为NumPy数组
    result = qcut(arr, 20)  # 使用20个等分位数对数组进行分段

    starts = []  # 初始化起始点列表
    ends = []  # 初始化结束点列表

    for lev in np.unique(result):  # 遍历分段结果的唯一值
        s = lev.left  # 获取分段区间的左边界
        e = lev.right  # 获取分段区间的右边界
        assert s != e  # 断言左右边界不相等

        starts.append(float(s))  # 将左边界添加到起始点列表
        ends.append(float(e))  # 将右边界添加到结束点列表

    for (sp, sn), (ep, en) in zip(  # 遍历起始点和结束点的对应元组
        zip(starts[:-1], starts[1:]), zip(ends[:-1], ends[1:])
    ):
        assert sp < sn  # 断言前一个分段的结束点小于后一个分段的起始点
        assert ep < en  # 断言前一个分段的结束点小于后一个分段的结束点
        assert ep <= sn  # 断言前一个分段的结束点小于等于后一个分段的起始点


def test_qcut_return_intervals():
    ser = Series([0, 1, 2, 3, 4, 5, 6, 7, 8])  # 创建包含整数的Pandas系列
    res = qcut(ser, [0, 0.333, 0.666, 1])  # 使用指定的分位数进行分段
    # 创建一个包含 Interval 对象的 NumPy 数组，每个 Interval 表示一个数值区间
    exp_levels = np.array(
        [Interval(-0.001, 2.664), Interval(2.664, 5.328), Interval(5.328, 8)]
    )
    # 从 exp_levels 数组中选取特定索引的 Interval 对象，构成一个 Series 对象 exp
    exp = Series(exp_levels.take([0, 0, 0, 1, 1, 1, 2, 2, 2])).astype(
        CategoricalDtype(ordered=True)
    )
    # 使用 pandas 的测试工具 tm 来断言两个 Series 对象 res 和 exp 是否相等
    tm.assert_series_equal(res, exp)
@pytest.mark.parametrize("labels", ["foo", 1, True])
def test_qcut_incorrect_labels(labels):
    # GH 13318
    # 定义测试数据
    values = range(5)
    # 定义错误信息
    msg = "Bin labels must either be False, None or passed in as a list-like argument"
    # 使用 pytest 断言检查是否抛出 ValueError 异常，并匹配错误信息
    with pytest.raises(ValueError, match=msg):
        # 调用 qcut 函数进行测试
        qcut(values, 4, labels=labels)


@pytest.mark.parametrize("labels", [["a", "b", "c"], list(range(3))])
def test_qcut_wrong_length_labels(labels):
    # GH 13318
    # 定义测试数据
    values = range(10)
    # 定义错误信息
    msg = "Bin labels must be one fewer than the number of bin edges"
    # 使用 pytest 断言检查是否抛出 ValueError 异常，并匹配错误信息
    with pytest.raises(ValueError, match=msg):
        # 调用 qcut 函数进行测试
        qcut(values, 4, labels=labels)


@pytest.mark.parametrize(
    "labels, expected",
    [
        (["a", "b", "c"], ["a", "b", "c"]),
        (list(range(3)), [0, 1, 2]),
    ],
)
def test_qcut_list_like_labels(labels, expected):
    # GH 13318
    # 定义测试数据
    values = range(3)
    # 调用 qcut 函数进行测试
    result = qcut(values, 3, labels=labels)
    # 创建预期的 Categorical 对象
    expected = Categorical(expected, ordered=True)
    # 使用 assert_categorical_equal 断言两个对象是否相等
    tm.assert_categorical_equal(result, expected)


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"duplicates": "drop"}, None),
        ({}, "Bin edges must be unique"),
        ({"duplicates": "raise"}, "Bin edges must be unique"),
        ({"duplicates": "foo"}, "invalid value for 'duplicates' parameter"),
    ],
)
def test_qcut_duplicates_bin(kwargs, msg):
    # see gh-7751
    # 定义测试数据
    values = [0, 0, 0, 0, 1, 2, 3]

    if msg is not None:
        # 使用 pytest 断言检查是否抛出 ValueError 异常，并匹配错误信息
        with pytest.raises(ValueError, match=msg):
            # 调用 qcut 函数进行测试
            qcut(values, 3, **kwargs)
    else:
        # 调用 qcut 函数进行测试
        result = qcut(values, 3, **kwargs)
        # 创建预期的 IntervalIndex 对象
        expected = IntervalIndex([Interval(-0.001, 1), Interval(1, 3)])
        # 使用 assert_index_equal 断言两个对象是否相等
        tm.assert_index_equal(result.categories, expected)


@pytest.mark.parametrize(
    "data,start,end", [(9.0, 8.999, 9.0), (0.0, -0.001, 0.0), (-9.0, -9.001, -9.0)]
)
@pytest.mark.parametrize("length", [1, 2])
@pytest.mark.parametrize("labels", [None, False])
def test_single_quantile(data, start, end, length, labels):
    # see gh-15431
    # 创建 Series 对象
    ser = Series([data] * length)
    # 调用 qcut 函数进行测试
    result = qcut(ser, 1, labels=labels)

    if labels is None:
        # 创建预期的 IntervalIndex 对象
        intervals = IntervalIndex([Interval(start, end)] * length, closed="right")
        expected = Series(intervals).astype(CategoricalDtype(ordered=True))
    else:
        expected = Series([0] * length, dtype=np.intp)

    # 使用 assert_series_equal 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "ser",
    [
        DatetimeIndex(["20180101", NaT, "20180103"]),
        TimedeltaIndex(["0 days", NaT, "2 days"]),
    ],
    ids=lambda x: str(x.dtype),
)
def test_qcut_nat(ser, unit):
    # see gh-19768
    # 创建 Series 对象
    ser = Series(ser)
    # 调整 Series 的单位
    ser = ser.dt.as_unit(unit)
    # 创建左右边界的 Series 对象
    td = Timedelta(1, unit=unit).as_unit(unit)
    left = Series([ser[0] - td, np.nan, ser[2] - Day()], dtype=ser.dtype)
    right = Series([ser[2] - Day(), np.nan, ser[2]], dtype=ser.dtype)
    # 创建 IntervalIndex 对象
    intervals = IntervalIndex.from_arrays(left, right)
    expected = Series(Categorical(intervals, ordered=True))

    # 调用 qcut 函数进行测试
    result = qcut(ser, 2)
    # 使用 assert_series_equal 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize("bins", [3, np.linspace(0, 1, 4)])
def test_datetime_tz_qcut(bins):
    # 根据 GitHub issue 19872 进行测试
    tz = "US/Eastern"  # 设置时区为美国东部时间
    ser = Series(date_range("20130101", periods=3, tz=tz))  # 创建一个包含三个日期的时间序列，时区为tz

    result = qcut(ser, bins)  # 对时间序列进行分位数分箱处理
    expected = Series(
        IntervalIndex(
            [
                Interval(
                    Timestamp("2012-12-31 23:59:59.999999999", tz=tz),  # 第一个区间的起始时间戳
                    Timestamp("2013-01-01 16:00:00", tz=tz),  # 第一个区间的结束时间戳
                ),
                Interval(
                    Timestamp("2013-01-01 16:00:00", tz=tz),  # 第二个区间的起始时间戳
                    Timestamp("2013-01-02 08:00:00", tz=tz),  # 第二个区间的结束时间戳
                ),
                Interval(
                    Timestamp("2013-01-02 08:00:00", tz=tz),  # 第三个区间的起始时间戳
                    Timestamp("2013-01-03 00:00:00", tz=tz),  # 第三个区间的结束时间戳
                ),
            ]
        )
    ).astype(CategoricalDtype(ordered=True))  # 将结果转换为有序分类数据类型
    tm.assert_series_equal(result, expected)  # 断言结果序列与预期序列相等


@pytest.mark.parametrize(
    "arg,expected_bins",
    [
        [
            timedelta_range("1day", periods=3),  # 创建一个时间增量序列，每天增加一天，共三个增量
            TimedeltaIndex(["1 days", "2 days", "3 days"]),  # 预期的时间增量索引
        ],
        [
            date_range("20180101", periods=3),  # 创建一个日期序列，从20180101开始，共三天
            DatetimeIndex(["2018-01-01", "2018-01-02", "2018-01-03"]),  # 预期的日期时间索引
        ],
    ],
)
def test_date_like_qcut_bins(arg, expected_bins, unit):
    # 根据 GitHub issue 19891 进行测试
    arg = arg.as_unit(unit)  # 将输入的时间序列或日期序列转换为指定单位的单位
    expected_bins = expected_bins.as_unit(unit)  # 将预期的时间索引或日期时间索引转换为指定单位的单位
    ser = Series(arg)  # 创建一个包含转换后数据的序列
    result, result_bins = qcut(ser, 2, retbins=True)  # 对序列进行分位数分箱处理，并返回分箱后的结果及分箱的边界
    tm.assert_index_equal(result_bins, expected_bins)  # 断言分箱的边界与预期的边界索引相等


@pytest.mark.parametrize("bins", [6, 7])
@pytest.mark.parametrize(
    "box, compare",
    [
        (Series, tm.assert_series_equal),  # 使用 Series 比较工具来比较结果
        (np.array, tm.assert_categorical_equal),  # 使用 numpy 数组比较工具来比较结果
        (list, tm.assert_equal),  # 使用列表比较工具来比较结果
    ],
)
def test_qcut_bool_coercion_to_int(bins, box, compare):
    # 根据 issue 20303 进行测试
    data_expected = box([0, 1, 1, 0, 1] * 10)  # 创建一个预期的数据序列，其中包含重复的布尔值
    data_result = box([False, True, True, False, True] * 10)  # 创建一个待测试的数据序列，其中包含重复的布尔值
    expected = qcut(data_expected, bins, duplicates="drop")  # 对预期的数据序列进行分位数分箱处理，忽略重复值
    result = qcut(data_result, bins, duplicates="drop")  # 对待测试的数据序列进行分位数分箱处理，忽略重复值
    compare(result, expected)  # 使用指定的比较工具来比较结果


@pytest.mark.parametrize("q", [2, 5, 10])
def test_qcut_nullable_integer(q, any_numeric_ea_dtype):
    arr = pd.array(np.arange(100), dtype=any_numeric_ea_dtype)  # 创建一个包含100个元素的 Pandas 数组，数据类型为指定的数值类型
    arr[::2] = pd.NA  # 将数组的偶数索引位置设置为缺失值

    result = qcut(arr, q)  # 对数组进行分位数分箱处理
    expected = qcut(arr.astype(float), q)  # 将数组转换为浮点数后再进行分位数分箱处理，作为预期结果

    tm.assert_categorical_equal(result, expected)  # 断言分箱后的结果与预期结果相等
```