# `D:\src\scipysrc\pandas\pandas\tests\window\test_api.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试

from pandas.errors import (  # 从pandas.errors模块中导入以下异常类
    DataError,  # 数据错误异常类
    SpecificationError,  # 规范错误异常类
)

from pandas import (  # 从pandas库中导入以下对象和函数
    DataFrame,  # 数据帧对象，用于存储二维数据
    Index,  # 索引对象，用于管理轴标签
    MultiIndex,  # 多重索引对象，用于管理多级索引
    Period,  # 时期对象，用于时间范围的表示
    Series,  # 系列对象，用于存储一维数据
    Timestamp,  # 时间戳对象，用于时间点的表示
    concat,  # 连接函数，用于对象的合并
    date_range,  # 时间范围函数，用于生成日期范围
    timedelta_range,  # 时间差范围函数，用于生成时间差范围
)
import pandas._testing as tm  # 导入pandas的测试工具模块，命名为tm


def test_getitem(step):
    frame = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    r = frame.rolling(window=5, step=step)
    tm.assert_index_equal(r._selected_obj.columns, frame[::step].columns)

    r = frame.rolling(window=5, step=step)[1]
    assert r._selected_obj.name == frame[::step].columns[1]

    # technically this is allowed
    r = frame.rolling(window=5, step=step)[1, 3]
    tm.assert_index_equal(r._selected_obj.columns, frame[::step].columns[[1, 3]])

    r = frame.rolling(window=5, step=step)[[1, 3]]
    tm.assert_index_equal(r._selected_obj.columns, frame[::step].columns[[1, 3]])


def test_select_bad_cols():
    df = DataFrame([[1, 2]], columns=["A", "B"])
    g = df.rolling(window=5)
    with pytest.raises(KeyError, match="Columns not found: 'C'"):
        g[["C"]]
    with pytest.raises(KeyError, match="^[^A]+$"):
        # A should not be referenced as a bad column...
        # will have to rethink regex if you change message!
        g[["A", "C"]]


def test_attribute_access():
    df = DataFrame([[1, 2]], columns=["A", "B"])
    r = df.rolling(window=5)
    tm.assert_series_equal(r.A.sum(), r["A"].sum())
    msg = "'Rolling' object has no attribute 'F'"
    with pytest.raises(AttributeError, match=msg):
        r.F


def tests_skip_nuisance(step):
    df = DataFrame({"A": range(5), "B": range(5, 10), "C": "foo"})
    r = df.rolling(window=3, step=step)
    result = r[["A", "B"]].sum()
    expected = DataFrame(
        {"A": [np.nan, np.nan, 3, 6, 9], "B": [np.nan, np.nan, 18, 21, 24]},
        columns=list("AB"),
    )[::step]
    tm.assert_frame_equal(result, expected)


def test_sum_object_str_raises(step):
    df = DataFrame({"A": range(5), "B": range(5, 10), "C": "foo"})
    r = df.rolling(window=3, step=step)
    with pytest.raises(
        DataError, match="Cannot aggregate non-numeric type: object|string"
    ):
        # GH#42738, enforced in 2.0
        r.sum()


def test_agg(step):
    df = DataFrame({"A": range(5), "B": range(0, 10, 2)})

    r = df.rolling(window=3, step=step)
    a_mean = r["A"].mean()
    a_std = r["A"].std()
    a_sum = r["A"].sum()
    b_mean = r["B"].mean()
    b_std = r["B"].std()

    result = r.aggregate([np.mean, lambda x: np.std(x, ddof=1)])
    expected = concat([a_mean, a_std, b_mean, b_std], axis=1)
    expected.columns = MultiIndex.from_product([["A", "B"], ["mean", "<lambda>"]])
    tm.assert_frame_equal(result, expected)

    result = r.aggregate({"A": np.mean, "B": lambda x: np.std(x, ddof=1)})
    expected = concat([a_mean, b_std], axis=1)
    tm.assert_frame_equal(result, expected, check_like=True)

    result = r.aggregate({"A": ["mean", "std"]})
    # 使用 concat 函数按照 axis=1（列方向）连接 a_mean 和 a_std，生成期望的 DataFrame
    expected = concat([a_mean, a_std], axis=1)
    # 使用 MultiIndex.from_tuples 为期望的 DataFrame 列创建多级索引 [("A", "mean"), ("A", "std")]
    expected.columns = MultiIndex.from_tuples([("A", "mean"), ("A", "std")])
    # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)

    # 对 r["A"] 应用 aggregate 函数，计算 "mean" 和 "sum"
    result = r["A"].aggregate(["mean", "sum"])
    # 使用 concat 函数按照 axis=1（列方向）连接 a_mean 和 a_sum，生成期望的 DataFrame
    expected = concat([a_mean, a_sum], axis=1)
    # 设置期望 DataFrame 的列名为 ["mean", "sum"]
    expected.columns = ["mean", "sum"]
    # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)

    # 准备错误消息
    msg = "nested renamer is not supported"
    # 使用 pytest.raises 检查是否抛出 SpecificationError 异常，并且异常消息包含预期的错误消息
    with pytest.raises(SpecificationError, match=msg):
        # 使用 dict 进行重命名，预期抛出异常
        r.aggregate({"A": {"mean": "mean", "sum": "sum"}})

    # 使用 pytest.raises 检查是否抛出 SpecificationError 异常，并且异常消息包含预期的错误消息
    with pytest.raises(SpecificationError, match=msg):
        # 使用 dict 进行重命名，同时包含多个列，预期抛出异常
        r.aggregate(
            {"A": {"mean": "mean", "sum": "sum"}, "B": {"mean2": "mean", "sum2": "sum"}}
        )

    # 对 r 进行 aggregate 操作，计算 "mean" 和 "std"，以及 "B" 列的 "mean" 和 "std"
    result = r.aggregate({"A": ["mean", "std"], "B": ["mean", "std"]})
    # 使用 concat 函数按照 axis=1（列方向）连接 a_mean, a_std, b_mean, b_std，生成期望的 DataFrame
    expected = concat([a_mean, a_std, b_mean, b_std], axis=1)
    # 设置期望 DataFrame 的列名为 [("A", "mean"), ("A", "std"), ("B", "mean"), ("B", "std")]
    exp_cols = [("A", "mean"), ("A", "std"), ("B", "mean"), ("B", "std")]
    expected.columns = MultiIndex.from_tuples(exp_cols)
    # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等，允许列的顺序和名称不同
    tm.assert_frame_equal(result, expected, check_like=True)
python
def test_agg_apply(raw):
    # 创建一个测试函数，用于测试数据聚合和应用函数的功能

    # 创建一个DataFrame对象，包含两列数据，列"A"是0到4的整数序列，列"B"是0到8中的偶数序列
    df = DataFrame({"A": range(5), "B": range(0, 10, 2)})

    # 创建一个滚动窗口对象r，窗口大小为3
    r = df.rolling(window=3)

    # 对r对象中"A"列的数据进行滚动窗口求和操作，存储在a_sum中
    a_sum = r["A"].sum()

    # 使用agg方法对r对象进行聚合操作，对"A"列应用np.sum函数，对"B"列应用一个lambda函数计算标准差
    result = r.agg({"A": np.sum, "B": lambda x: np.std(x, ddof=1)})

    # 对r对象中"B"列应用lambda函数计算标准差，设置参数raw为函数输入的原始数据
    rcustom = r["B"].apply(lambda x: np.std(x, ddof=1), raw=raw)

    # 将a_sum和rcustom合并在一起，按列(axis=1)连接形成期望的DataFrame对象expected
    expected = concat([a_sum, rcustom], axis=1)

    # 使用tm.assert_frame_equal函数断言result和expected两个DataFrame对象相等
    tm.assert_frame_equal(result, expected, check_like=True)


def test_agg_consistency(step):
    # 创建一个测试函数，用于测试数据聚合的一致性

    # 创建一个DataFrame对象，包含两列数据，列"A"是0到4的整数序列，列"B"是0到8中的偶数序列
    df = DataFrame({"A": range(5), "B": range(0, 10, 2)})

    # 创建一个滚动窗口对象r，窗口大小为3，步长为step
    r = df.rolling(window=3, step=step)

    # 使用agg方法对r对象中所有列应用np.sum和np.mean函数，获取聚合结果的列索引
    result = r.agg([np.sum, np.mean]).columns

    # 创建一个预期的MultiIndex对象，包含"A"和"B"两列的"sum"和"mean"两个层级
    expected = MultiIndex.from_product([list("AB"), ["sum", "mean"]])

    # 使用tm.assert_index_equal函数断言result和expected两个MultiIndex对象相等
    tm.assert_index_equal(result, expected)

    # 使用agg方法对r对象中"A"列应用np.sum和np.mean函数，获取聚合结果的列索引
    result = r["A"].agg([np.sum, np.mean]).columns

    # 创建一个预期的Index对象，包含"sum"和"mean"两个层级
    expected = Index(["sum", "mean"])

    # 使用tm.assert_index_equal函数断言result和expected两个Index对象相等
    tm.assert_index_equal(result, expected)

    # 使用agg方法对r对象中"A"列应用np.sum和np.mean函数，获取聚合结果的列索引
    result = r.agg({"A": [np.sum, np.mean]}).columns

    # 创建一个预期的MultiIndex对象，包含"A"列的"sum"和"mean"两个层级
    expected = MultiIndex.from_tuples([("A", "sum"), ("A", "mean")])

    # 使用tm.assert_index_equal函数断言result和expected两个MultiIndex对象相等
    tm.assert_index_equal(result, expected)


def test_agg_nested_dicts():
    # 创建一个测试函数，用于测试嵌套字典在agg方法中的行为

    # 创建一个DataFrame对象，包含两列数据，列"A"是0到4的整数序列，列"B"是0到8中的偶数序列
    df = DataFrame({"A": range(5), "B": range(0, 10, 2)})

    # 创建一个滚动窗口对象r，窗口大小为3
    r = df.rolling(window=3)

    # 准备错误消息字符串，用于匹配抛出的SpecificationError异常
    msg = "nested renamer is not supported"

    # 使用pytest.raises捕获SpecificationError异常，测试agg方法中传入的嵌套字典是否会引发异常
    with pytest.raises(SpecificationError, match=msg):
        r.aggregate({"r1": {"A": ["mean", "sum"]}, "r2": {"B": ["mean", "sum"]}})

    # 创建一个期望的DataFrame对象expected，包含对r对象中"A"和"B"列的均值和标准差计算结果
    expected = concat([r["A"].mean(), r["A"].std(), r["B"].mean(), r["B"].std()], axis=1)

    # 设置expected对象的列索引为MultiIndex对象，指定列名为("ra", "mean"), ("ra", "std"), ("rb", "mean"), ("rb", "std")
    expected.columns = MultiIndex.from_tuples(
        [("ra", "mean"), ("ra", "std"), ("rb", "mean"), ("rb", "std")]
    )

    # 使用pytest.raises捕获SpecificationError异常，测试agg方法中传入的嵌套字典是否会引发异常
    with pytest.raises(SpecificationError, match=msg):
        r[["A", "B"]].agg({"A": {"ra": ["mean", "std"]}, "B": {"rb": ["mean", "std"]}})

    # 使用pytest.raises捕获SpecificationError异常，测试agg方法中传入的嵌套字典是否会引发异常
    with pytest.raises(SpecificationError, match=msg):
        r.agg({"A": {"ra": ["mean", "std"]}, "B": {"rb": ["mean", "std"]}})


def test_count_nonnumeric_types(step):
    # 创建一个测试函数，用于测试DataFrame中非数值类型的计数

    # 创建一个列名列表cols，包含各种数据类型的列名字符串
    cols = [
        "int",
        "float",
        "string",
        "datetime",
        "timedelta",
        "periods",
        "fl_inf",
        "fl_nan",
        "str_nan",
        "dt_nat",
        "periods_nat",
    ]

    # 创建一个包含各种数据类型的DataFrame对象df，列名由cols列表提供
    dt_nat_col = [Timestamp("20170101"), Timestamp("20170203"), Timestamp(None)]
    df = DataFrame(
        {
            "int": [1, 2, 3],
            "float": [4.0, 5.0, 6.0],
            "string": list("abc"),
            "datetime": date_range("20170101", periods=3),
            "timedelta": timedelta_range("1 s", periods=3, freq="s"),
            "periods": [
                Period("2012-01"),
                Period("2012-02"),
                Period("2012-03"),
            ],
            "fl_inf": [1.0, 2.0, np.inf],
            "fl_nan": [1.0, 2.0, np.nan],
            "str_nan": ["aa", "bb", np.nan],
            "dt_nat": dt_nat_col,
            "periods_nat": [
                Period("2012-01"),
                Period("2012-02"),
                Period(None),
            ],
        },
        columns=cols,
    )
    # 创建一个 DataFrame，包含各种数据类型的列，并根据给定的列名进行初始化
    expected = DataFrame(
        {
            "int": [1.0, 2.0, 2.0],
            "float": [1.0, 2.0, 2.0],
            "string": [1.0, 2.0, 2.0],
            "datetime": [1.0, 2.0, 2.0],
            "timedelta": [1.0, 2.0, 2.0],
            "periods": [1.0, 2.0, 2.0],
            "fl_inf": [1.0, 2.0, 2.0],
            "fl_nan": [1.0, 2.0, 1.0],
            "str_nan": [1.0, 2.0, 1.0],
            "dt_nat": [1.0, 2.0, 1.0],
            "periods_nat": [1.0, 2.0, 1.0],
        },
        columns=cols,  # 使用预定义的列名初始化 DataFrame 的列
    )[::step]  # 根据步长 step 对 DataFrame 进行切片操作，生成期望的 DataFrame
    
    # 对 DataFrame df 执行滚动窗口计数操作，窗口大小为 2，最小观测数为 0，步长为 step
    result = df.rolling(window=2, min_periods=0, step=step).count()
    # 使用测试工具函数 tm.assert_frame_equal 检查计算结果 result 是否与期望结果 expected 相等
    tm.assert_frame_equal(result, expected)
    
    # 对 DataFrame df 执行滚动窗口计数操作，窗口大小为 1，最小观测数为 0，步长为 step
    result = df.rolling(1, min_periods=0, step=step).count()
    # 生成期望的 DataFrame，根据 df 的非空值进行类型转换，并根据步长 step 进行切片操作
    expected = df.notna().astype(float)[::step]
    # 使用测试工具函数 tm.assert_frame_equal 检查计算结果 result 是否与期望结果 expected 相等
    tm.assert_frame_equal(result, expected)
# 定义测试函数，用于验证滚动窗口操作不修改对象属性
def test_preserve_metadata():
    # GH 10565：GitHub issue编号，可能是相关问题的追踪标识
    # 创建一个Series对象，其值为0到99，名称为"foo"
    s = Series(np.arange(100), name="foo")

    # 对Series对象s应用窗口大小为30的滚动求和操作
    s2 = s.rolling(30).sum()
    # 对Series对象s应用窗口大小为20的滚动求和操作
    s3 = s.rolling(20).sum()

    # 断言s2和s3的名称都为"foo"
    assert s2.name == "foo"
    assert s3.name == "foo"


# 使用pytest的parametrize装饰器，定义多个参数化测试用例
@pytest.mark.parametrize(
    "func,window_size,expected_vals",
    [
        (
            "rolling",
            2,
            [
                [np.nan, np.nan, np.nan, np.nan],
                [15.0, 20.0, 25.0, 20.0],
                [25.0, 30.0, 35.0, 30.0],
                [np.nan, np.nan, np.nan, np.nan],
                [20.0, 30.0, 35.0, 30.0],
                [35.0, 40.0, 60.0, 40.0],
                [60.0, 80.0, 85.0, 80],
            ],
        ),
        (
            "expanding",
            None,
            [
                [10.0, 10.0, 20.0, 20.0],
                [15.0, 20.0, 25.0, 20.0],
                [20.0, 30.0, 30.0, 20.0],
                [10.0, 10.0, 30.0, 30.0],
                [20.0, 30.0, 35.0, 30.0],
                [26.666667, 40.0, 50.0, 30.0],
                [40.0, 80.0, 60.0, 30.0],
            ],
        ),
    ],
)
# 定义测试函数，用于验证多种聚合函数在DataFrame上的应用
def test_multiple_agg_funcs(func, window_size, expected_vals):
    # GH 15072：GitHub issue编号，可能是相关问题的追踪标识
    # 创建一个DataFrame对象，包含股票数据及其低价和高价
    df = DataFrame(
        [
            ["A", 10, 20],
            ["A", 20, 30],
            ["A", 30, 40],
            ["B", 10, 30],
            ["B", 30, 40],
            ["B", 40, 80],
            ["B", 80, 90],
        ],
        columns=["stock", "low", "high"],
    )

    # 根据指定的聚合函数名称func获取对应的函数对象
    f = getattr(df.groupby("stock"), func)
    # 根据窗口大小或默认创建DataFrameGroupBy对象的滚动或扩展窗口
    if window_size:
        window = f(window_size)
    else:
        window = f()

    # 创建多级索引，用于预期的DataFrame对象
    index = MultiIndex.from_tuples(
        [("A", 0), ("A", 1), ("A", 2), ("B", 3), ("B", 4), ("B", 5), ("B", 6)],
        names=["stock", None],
    )
    # 创建多级索引，用于预期的DataFrame对象的列
    columns = MultiIndex.from_tuples(
        [("low", "mean"), ("low", "max"), ("high", "mean"), ("high", "min")]
    )
    # 根据预期的值列表创建预期的DataFrame对象
    expected = DataFrame(expected_vals, index=index, columns=columns)

    # 对窗口对象应用聚合函数，生成结果DataFrame对象
    result = window.agg({"low": ["mean", "max"], "high": ["mean", "min"]})

    # 使用测试工具tm.assert_frame_equal断言结果DataFrame与预期DataFrame相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数，用于验证在方法调用后不修改对象属性
def test_dont_modify_attributes_after_methods(
    arithmetic_win_operators, closed, center, min_periods, step
):
    # GH 39554：GitHub issue编号，可能是相关问题的追踪标识
    # 创建一个Series对象，其值为0到1，应用指定参数的滚动窗口对象
    roll_obj = Series(range(1)).rolling(
        1, center=center, closed=closed, min_periods=min_periods, step=step
    )
    # 创建预期的字典，包含滚动窗口对象的所有属性及其对应的值
    expected = {attr: getattr(roll_obj, attr) for attr in roll_obj._attributes}
    # 根据指定的算术窗口操作方法名，对滚动窗口对象应用方法
    getattr(roll_obj, arithmetic_win_operators)()
    # 创建结果的字典，包含滚动窗口对象的所有属性及其对应的值
    result = {attr: getattr(roll_obj, attr) for attr in roll_obj._attributes}
    # 断言结果字典与预期字典相等，确保方法调用未修改对象属性
    assert result == expected


# 定义测试函数，用于验证滚动窗口最小值操作在指定最小周期下的行为
def test_rolling_min_min_periods(step):
    # 创建一个Series对象，包含值为1到5
    a = Series([1, 2, 3, 4, 5])
    # 对Series对象应用窗口大小为100，最小周期为1的滚动最小值操作
    result = a.rolling(window=100, min_periods=1, step=step).min()
    # 创建预期的Series对象，其值为1，包含步长为step的索引
    expected = Series(np.ones(len(a)))[::step]
    # 使用测试工具tm.assert_series_equal断言结果Series与预期Series相等
    tm.assert_series_equal(result, expected)
    # 创建错误消息，用于验证滚动窗口最小值操作的最小周期限制
    msg = "min_periods 5 must be <= window 3"
    # 使用pytest的raises方法验证滚动窗口对象的最小周期限制抛出值错误异常
    with pytest.raises(ValueError, match=msg):
        Series([1, 2, 3]).rolling(window=3, min_periods=5, step=step).min()


# 定义测试函数，用于验证滚动窗口最大值操作在指定最小周期下的行为
def test_rolling_max_min_periods(step):
    # 创建一个 Pandas Series 对象，包含浮点数 [1.0, 2.0, 3.0, 4.0, 5.0]
    a = Series([1, 2, 3, 4, 5], dtype=np.float64)
    # 对 Series 对象进行滚动计算最大值，窗口大小为 100，最小周期为 1，步长由变量 `step` 决定
    result = a.rolling(window=100, min_periods=1, step=step).max()
    # 从 Series 对象中按照步长 `step` 取值，构建期望的结果
    expected = a[::step]
    # 使用 pytest 进行断言，验证是否抛出 ValueError 异常，并匹配指定的错误消息
    msg = "min_periods 5 must be <= window 3"
    with pytest.raises(ValueError, match=msg):
        # 对包含 [1, 2, 3] 的 Series 对象进行滚动计算最大值，窗口大小为 3，最小周期为 5，步长由变量 `step` 决定
        Series([1, 2, 3]).rolling(window=3, min_periods=5, step=step).max()
```