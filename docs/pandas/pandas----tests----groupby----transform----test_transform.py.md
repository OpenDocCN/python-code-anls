# `D:\src\scipysrc\pandas\pandas\tests\groupby\transform\test_transform.py`

```
"""test with the .transform"""

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas._libs import lib  # 导入 pandas 库的 C 扩展模块

from pandas.core.dtypes.common import ensure_platform_int  # 导入 pandas 库的数据类型相关工具函数

import pandas as pd  # 导入 pandas 库并重命名为 pd
from pandas import (  # 导入 pandas 库中的多个对象
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    concat,
    date_range,
)
import pandas._testing as tm  # 导入 pandas 库的测试工具
from pandas.tests.groupby import get_groupby_method_args  # 导入分组相关测试工具


def assert_fp_equal(a, b):
    assert (np.abs(a - b) < 1e-12).all()  # 断言两个浮点数数组在误差允许范围内相等


def test_transform():
    data = Series(np.arange(9) // 3, index=np.arange(9))  # 创建一个 Series 对象

    index = np.arange(9)  # 创建一个数组作为索引
    np.random.default_rng(2).shuffle(index)  # 使用随机数生成器对索引进行乱序
    data = data.reindex(index)  # 根据乱序后的索引重新排列 Series 对象

    grouped = data.groupby(lambda x: x // 3)  # 对 Series 对象进行分组

    transformed = grouped.transform(lambda x: x * x.sum())  # 对分组后的数据进行变换操作
    assert transformed[7] == 12  # 断言特定索引处的变换结果为 12

    # GH 8046
    # make sure that we preserve the input order
    # 确保我们保留了输入顺序

    df = DataFrame(  # 创建一个 DataFrame 对象
        np.arange(6, dtype="int64").reshape(3, 2), columns=["a", "b"], index=[0, 2, 1]
    )
    key = [0, 0, 1]  # 创建一个分组关键字列表
    expected = (
        df.sort_index()  # 对 DataFrame 按索引排序
        .groupby(key)  # 根据关键字分组
        .transform(lambda x: x - x.mean())  # 对每个分组应用均值去中心化操作
        .groupby(key)  # 对处理后的结果再次根据关键字分组
        .mean()  # 计算每个分组的均值
    )
    result = df.groupby(key).transform(lambda x: x - x.mean()).groupby(key).mean()  # 执行和预期相同的操作
    tm.assert_frame_equal(result, expected)  # 断言处理后的结果与预期相等

    def demean(arr):
        return arr - arr.mean(axis=0)

    people = DataFrame(  # 创建一个包含随机数据的 DataFrame 对象
        np.random.default_rng(2).standard_normal((5, 5)),
        columns=["a", "b", "c", "d", "e"],
        index=["Joe", "Steve", "Wes", "Jim", "Travis"],
    )
    key = ["one", "two", "one", "two", "one"]  # 创建一个分组关键字列表
    result = people.groupby(key).transform(demean).groupby(key).mean()  # 对每个分组进行去均值化处理，并计算均值
    expected = people.groupby(key, group_keys=False).apply(demean).groupby(key).mean()  # 执行和预期相同的操作
    tm.assert_frame_equal(result, expected)  # 断言处理后的结果与预期相等

    # GH 8430
    df = DataFrame(  # 创建一个包含随机数据的 DataFrame 对象
        np.random.default_rng(2).standard_normal((50, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=50, freq="B"),
    )
    g = df.groupby(pd.Grouper(freq="ME"))  # 按指定的频率进行时间分组
    g.transform(lambda x: x - 1)  # 对每个分组的数据执行减 1 操作

    # GH 9700
    df = DataFrame({"a": range(5, 10), "b": range(5)})  # 创建一个包含两列数据的 DataFrame 对象
    result = df.groupby("a").transform(max)  # 对每个分组的数据执行最大值转换操作
    expected = DataFrame({"b": range(5)})  # 创建一个预期结果的 DataFrame 对象
    tm.assert_frame_equal(result, expected)  # 断言处理后的结果与预期相等


def test_transform_fast():
    df = DataFrame(  # 创建一个包含两列数据的 DataFrame 对象
        {
            "id": np.arange(10) / 3,
            "val": np.random.default_rng(2).standard_normal(10),
        }
    )

    grp = df.groupby("id")["val"]  # 按指定列进行分组并获取指定列数据

    values = np.repeat(grp.mean().values, ensure_platform_int(grp.count().values))  # 计算每个分组的平均值并重复扩展
    expected = Series(values, index=df.index, name="val")  # 创建一个期望的 Series 对象

    result = grp.transform(np.mean)  # 对分组后的数据执行均值转换操作
    tm.assert_series_equal(result, expected)  # 断言处理后的结果与期望的结果相等

    result = grp.transform("mean")  # 使用字符串形式指定均值转换操作
    tm.assert_series_equal(result, expected)  # 断言处理后的结果与期望的结果相等


def test_transform_fast2():
    # GH 12737
    # Placeholder for future test case related to fast transform
    # 未来与快速转换相关的测试用例的占位符
    # 创建一个 DataFrame 对象，包含四列数据：grouping, f, d, i
    df = DataFrame(
        {
            "grouping": [0, 1, 1, 3],
            "f": [1.1, 2.1, 3.1, 4.5],
            "d": date_range("2014-1-1", "2014-1-4"),  # 创建一个日期范围的时间索引
            "i": [1, 2, 3, 4],
        },
        columns=["grouping", "f", "i", "d"],  # 指定列的顺序
    )
    
    # 对 DataFrame 按照 'grouping' 列进行分组，然后对每组数据执行 'first' 转换操作
    result = df.groupby("grouping").transform("first")

    # 创建一个时间戳索引对象，包含四个日期时间戳
    dates = Index(
        [
            Timestamp("2014-1-1"),
            Timestamp("2014-1-2"),
            Timestamp("2014-1-2"),
            Timestamp("2014-1-4"),
        ],
        dtype="M8[ns]",  # 设置时间戳的数据类型为 'datetime64[ns]'
    )
    
    # 创建一个预期的 DataFrame 对象，包含三列数据：'f', 'i', 'd'，并且 'd' 列使用上面创建的时间戳索引
    expected = DataFrame(
        {"f": [1.1, 2.1, 2.1, 4.5], "d": dates, "i": [1, 2, 2, 4]},
        columns=["f", "i", "d"],  # 指定列的顺序
    )
    
    # 使用测试工具函数 tm.assert_frame_equal() 检查 'result' 和 'expected' 是否相等
    tm.assert_frame_equal(result, expected)

    # 选择操作：从 'expected' DataFrame 中选择 'f' 和 'i' 列，并赋给 'result'
    result = df.groupby("grouping")[["f", "i"]].transform("first")
    
    # 更新 'expected'，只保留 'f' 和 'i' 两列
    expected = expected[["f", "i"]]
    
    # 再次使用 tm.assert_frame_equal() 检查 'result' 和更新后的 'expected' 是否相等
    tm.assert_frame_equal(result, expected)
def test_transform_fast3():
    # 定义测试函数，测试数据框的转换操作
    # 创建包含重复列的数据框
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["g", "a", "a"])
    # 使用"groupby"方法按"g"列分组，并对每组应用"first"函数
    result = df.groupby("g").transform("first")
    # 期望结果为删除"g"列后的数据框
    expected = df.drop("g", axis=1)
    # 断言转换后的结果与期望结果相等
    tm.assert_frame_equal(result, expected)


def test_transform_broadcast(tsframe, ts):
    # 定义测试函数，测试时间序列数据的分组转换操作
    # 使用"groupby"方法按月份分组
    grouped = ts.groupby(lambda x: x.month)
    # 对每组数据应用"np.mean"函数进行转换
    result = grouped.transform(np.mean)

    # 断言转换后的结果索引与原始时间序列索引相等
    tm.assert_index_equal(result.index, ts.index)
    # 遍历每个分组，断言转换后的结果与每组的均值相等
    for _, gp in grouped:
        assert_fp_equal(result.reindex(gp.index), gp.mean())

    # 使用"groupby"方法按月份分组时间序列数据框
    grouped = tsframe.groupby(lambda x: x.month)
    # 对每组数据框应用"np.mean"函数进行转换
    result = grouped.transform(np.mean)
    # 断言转换后的结果索引与原始时间序列数据框索引相等
    tm.assert_index_equal(result.index, tsframe.index)
    # 遍历每个分组，计算每列的均值，然后断言转换后的结果与每组的均值相等
    for _, gp in grouped:
        agged = gp.mean(axis=0)
        res = result.reindex(gp.index)
        for col in tsframe:
            assert_fp_equal(res[col], agged[col])


def test_transform_axis_ts(tsframe):
    # 定义测试函数，测试时间序列数据的分组转换操作
    # 确保在存在非单调索引时正确设置轴
    # GH12713

    # 从时间序列数据框中选取部分数据作为基础数据
    base = tsframe.iloc[0:5]
    r = len(base.index)
    c = len(base.columns)
    # 创建一个与基础数据形状相同的随机数数据框
    tso = DataFrame(
        np.random.default_rng(2).standard_normal((r, c)),
        index=base.index,
        columns=base.columns,
        dtype="float64",
    )
    # 单调索引情况
    ts = tso
    # 使用"groupby"方法按工作日分组，不生成组键
    grouped = ts.groupby(lambda x: x.weekday(), group_keys=False)
    # 对每组数据应用均值转换
    result = ts - grouped.transform("mean")
    # 期望结果为每组数据减去每组数据的均值
    expected = grouped.apply(lambda x: x - x.mean(axis=0))
    # 断言转换后的结果与期望结果相等
    tm.assert_frame_equal(result, expected)

    # 非单调索引情况
    ts = tso.iloc[[1, 0] + list(range(2, len(base)))]
    # 使用"groupby"方法按工作日分组，不生成组键
    grouped = ts.groupby(lambda x: x.weekday(), group_keys=False)
    # 对每组数据应用均值转换
    result = ts - grouped.transform("mean")
    # 期望结果为每组数据减去每组数据的均值
    expected = grouped.apply(lambda x: x - x.mean(axis=0))
    # 断言转换后的结果与期望结果相等
    tm.assert_frame_equal(result, expected)


def test_transform_dtype():
    # GH 9807
    # 检查转换后的数据类型输出是否保持不变
    # 创建包含整数的数据框
    df = DataFrame([[1, 3], [2, 3]])
    # 使用"groupby"方法按第二列分组，并对每组应用"mean"函数
    result = df.groupby(1).transform("mean")
    # 期望结果为包含浮点数的数据框
    expected = DataFrame([[1.5], [1.5]])
    # 断言转换后的结果与期望结果相等
    tm.assert_frame_equal(result, expected)


def test_transform_bug():
    # GH 5712
    # 在日期时间列上进行转换
    # 创建包含日期时间和整数的数据框
    df = DataFrame({"A": Timestamp("20130101"), "B": np.arange(5)})
    # 使用"groupby"方法按"A"列分组，并对"B"列应用"rank"函数，降序排列
    result = df.groupby("A")["B"].transform(lambda x: x.rank(ascending=False))
    # 期望结果为包含降序排列整数的序列
    expected = Series(np.arange(5, 0, step=-1), name="B", dtype="float64")
    # 断言转换后的结果与期望结果相等
    tm.assert_series_equal(result, expected)


def test_transform_numeric_to_boolean():
    # GH 16875
    # 转换布尔值时的不一致性
    # 创建包含浮点数和整数的数据框
    expected = Series([True, True], name="A")

    df = DataFrame({"A": [1.1, 2.2], "B": [1, 2]})
    # 使用"groupby"方法按"B"列分组，并对"A"列应用lambda函数，始终返回True
    result = df.groupby("B").A.transform(lambda x: True)
    # 断言转换后的结果与期望结果相等
    tm.assert_series_equal(result, expected)

    df = DataFrame({"A": [1, 2], "B": [1, 2]})
    # 使用"groupby"方法按"B"列分组，并对"A"列应用lambda函数，始终返回True
    result = df.groupby("B").A.transform(lambda x: True)
    # 断言转换后的结果与期望结果相等
    tm.assert_series_equal(result, expected)


def test_transform_datetime_to_timedelta():
    # GH 15429
    # 将日期时间转换为时间差
    df = DataFrame({"A": Timestamp("20130101"), "B": np.arange(5)})
    expected = Series(
        Timestamp("20130101") - Timestamp("20130101"), index=range(5), name="A"
    )

    # 在 transform 中进行日期计算，保持结果类型不变
    base_time = df["A"][0]  # 获取 DataFrame 第一行的时间戳作为基准时间
    result = (
        df.groupby("A")["A"].transform(lambda x: x.max() - x.min() + base_time)
        - base_time
    )
    tm.assert_series_equal(result, expected)

    # 在 transform 中进行日期计算，并导致返回 timedelta 类型的结果
    result = df.groupby("A")["A"].transform(lambda x: x.max() - x.min())
    tm.assert_series_equal(result, expected)
def test_transform_datetime_to_numeric():
    # GH 10972
    # 创建一个包含两列的数据框，其中一列是数字 1，另一列是从指定日期开始的两天日期范围
    df = DataFrame({"a": 1, "b": date_range("2015-01-01", periods=2, freq="D")})
    # 对数据框按 'a' 列分组，对 'b' 列应用转换函数，计算每个日期的星期几与平均星期几之间的差值
    result = df.groupby("a").b.transform(
        lambda x: x.dt.dayofweek - x.dt.dayofweek.mean()
    )

    # 预期的结果，是一个包含两个元素的序列，代表两天的星期几差值
    expected = Series([-0.5, 0.5], name="b")
    # 断言结果与预期相等
    tm.assert_series_equal(result, expected)

    # 将日期转换为整数表示的星期几
    df = DataFrame({"a": 1, "b": date_range("2015-01-01", periods=2, freq="D")})
    # 对数据框按 'a' 列分组，对 'b' 列应用转换函数，计算每个日期的星期几与最小星期几之间的差值
    result = df.groupby("a").b.transform(
        lambda x: x.dt.dayofweek - x.dt.dayofweek.min()
    )

    # 预期的结果，是一个包含两个元素的整数序列，代表两天的星期几
    expected = Series([0, 1], dtype=np.int32, name="b")
    # 断言结果与预期相等
    tm.assert_series_equal(result, expected)


def test_transform_casting():
    # 13046
    # 一组时间字符串
    times = [
        "13:43:27",
        "14:26:19",
        "14:29:01",
        "18:39:34",
        "18:40:18",
        "18:44:30",
        "18:46:00",
        "18:52:15",
        "18:59:59",
        "19:17:48",
        "19:21:38",
    ]
    # 创建一个数据框，包含 'A' 列和 'ID3' 列，以及从时间字符串生成的日期时间列
    df = DataFrame(
        {
            "A": [f"B-{i}" for i in range(11)],
            "ID3": np.take(
                ["a", "b", "c", "d", "e"], [0, 1, 2, 1, 3, 1, 1, 1, 4, 1, 1]
            ),
            "DATETIME": pd.to_datetime([f"2014-10-08 {time}" for time in times]),
        },
        index=pd.RangeIndex(11, name="idx"),
    )

    # 对数据框按 'ID3' 列分组，对 'DATETIME' 列应用转换函数，计算每个分组内连续两个日期时间之间的时间差
    result = df.groupby("ID3")["DATETIME"].transform(lambda x: x.diff())
    # 断言结果的数据类型是 numpy 中的日期时间类型
    assert lib.is_np_dtype(result.dtype, "m")

    # 对数据框选择 'ID3' 和 'DATETIME' 两列，按 'ID3' 列分组，对 'DATETIME' 列应用转换函数，计算每个分组内连续两个日期时间之间的时间差
    result = df[["ID3", "DATETIME"]].groupby("ID3").transform(lambda x: x.diff())
    # 断言结果的 'DATETIME' 列的数据类型是 numpy 中的日期时间类型
    assert lib.is_np_dtype(result.DATETIME.dtype, "m")


def test_transform_multiple(ts):
    # 对时间序列按年份和月份分组
    grouped = ts.groupby([lambda x: x.year, lambda x: x.month])
    # 对分组后的每个组内的数据进行两倍转换
    grouped.transform(lambda x: x * 2)
    # 对分组后的每个组内的数据进行均值转换
    grouped.transform(np.mean)


def test_dispatch_transform(tsframe):
    # 从时间序列框架中每隔五个取一个时间戳，重新索引时间序列框架
    df = tsframe[::5].reindex(tsframe.index)

    # 对重新索引后的数据框按月份分组
    grouped = df.groupby(lambda x: x.month)

    # 对分组后的数据框进行向前填充
    filled = grouped.ffill()
    # 定义一个函数，对数据框进行向前填充
    fillit = lambda x: x.ffill()
    # 预期的结果，对每个分组应用向前填充函数后的数据框
    expected = df.groupby(lambda x: x.month).transform(fillit)
    # 断言填充后的结果与预期相等
    tm.assert_frame_equal(filled, expected)


def test_transform_transformation_func(transformation_func):
    # GH 30918
    # 创建一个包含 'A' 和 'B' 两列的数据框， 'A' 列有三个不同的值， 'B' 列包含整数和缺失值
    df = DataFrame(
        {
            "A": ["foo", "foo", "foo", "foo", "bar", "bar", "baz"],
            "B": [1, 2, np.nan, 3, 3, np.nan, 4],
        },
        index=date_range("2020-01-01", "2020-01-07"),
    )
    if transformation_func == "cumcount":
        # 如果指定的转换函数是 'cumcount'，则对 'B' 列进行累计计数转换
        test_op = lambda x: x.transform("cumcount")
        # 定义一个模拟操作函数，对 'B' 列进行累计计数转换
        mock_op = lambda x: Series(range(len(x)), x.index)
    elif transformation_func == "fillna":
        # 如果指定的转换函数是 'fillna'，则对 'B' 列进行缺失值填充，填充值为 0
        test_op = lambda x: x.transform("fillna", value=0)
        # 定义一个模拟操作函数，对 'B' 列进行缺失值填充，填充值为 0
        mock_op = lambda x: x.fillna(value=0)
    elif transformation_func == "ngroup":
        # 如果指定的转换函数是 'ngroup'，则对数据框进行分组编号转换
        test_op = lambda x: x.transform("ngroup")
        counter = -1

        def mock_op(x):
            nonlocal counter
            counter += 1
            return Series(counter, index=x.index)

    else:
        # 否则，对指定列应用给定的转换函数
        test_op = lambda x: x.transform(transformation_func)
        # 定义一个模拟操作函数，对指定列应用给定的转换函数
        mock_op = lambda x: getattr(x, transformation_func)()
    result = test_op(df.groupby("A"))

    # 对数据框按列"A"分组后，应用测试函数test_op，得到结果
    # test_op 函数的返回值赋给 result

    # 按照与迭代df.groupby(...)相同的顺序传递分组
    # 但重新排序以匹配df的索引，因为这是一个转换操作
    groups = [df[["B"]].iloc[4:6], df[["B"]].iloc[6:], df[["B"]].iloc[:4]]
    # 创建一个包含三个分组的列表，每个分组是df的子集DataFrame，只包含"B"列的不同切片

    # 期望的结果是对所有分组进行模拟操作(mock_op)，然后连接它们并按索引排序
    expected = concat([mock_op(g) for g in groups]).sort_index()
    # 将每个分组g传递给mock_op函数，并将所有结果连接起来，然后按索引排序

    # sort_index 不保留频率信息
    expected = expected.set_axis(df.index)
    # 设置期望结果的轴标签为df的索引，以保持一致性

    if transformation_func in ("cumcount", "ngroup"):
        # 如果变换函数是"cumcount"或"ngroup"
        tm.assert_series_equal(result, expected)
        # 使用测试框架中的assert_series_equal函数比较result和expected，确保它们相等
    else:
        tm.assert_frame_equal(result, expected)
        # 否则，使用测试框架中的assert_frame_equal函数比较result和expected，确保它们相等
def test_transform_select_columns(df):
    # 定义一个匿名函数，用于计算均值
    f = lambda x: x.mean()
    # 对 DataFrame 按列 "A" 分组，并对 "C" 和 "D" 列应用均值函数
    result = df.groupby("A")[["C", "D"]].transform(f)

    # 选择 DataFrame 中的 "C" 和 "D" 列
    selection = df[["C", "D"]]
    # 根据 "A" 列分组，并对选择的 "C" 和 "D" 列应用均值函数
    expected = selection.groupby(df["A"]).transform(f)

    # 使用测试工具验证 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


def test_transform_nuisance_raises(df):
    # 将 DataFrame 的列名修改为 ["A", "B", "B", "D"]
    df.columns = ["A", "B", "B", "D"]

    # 对 df 按列 "A" 进行分组
    grouped = df.groupby("A")

    # 获取分组后的 "B" 列
    gbc = grouped["B"]
    # 使用 pytest 检查是否会引发 TypeError，错误消息中包含 "Could not convert"
    with pytest.raises(TypeError, match="Could not convert"):
        gbc.transform(lambda x: np.mean(x))

    # 再次使用 pytest 检查是否会引发 TypeError，错误消息中包含 "Could not convert"
    with pytest.raises(TypeError, match="Could not convert"):
        df.groupby("A").transform(lambda x: np.mean(x))


def test_transform_function_aliases(df):
    # 对 DataFrame 按列 "A" 进行分组，并应用 "mean" 函数，只针对数值列
    result = df.groupby("A").transform("mean", numeric_only=True)
    # 对 DataFrame 按列 "A" 进行分组，并应用 np.mean 函数
    expected = df.groupby("A")[["C", "D"]].transform(np.mean)
    # 使用测试工具验证 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)

    # 对 DataFrame 按列 "A" 进行分组，并应用 "mean" 函数到 "C" 列
    result = df.groupby("A")["C"].transform("mean")
    # 对 DataFrame 按列 "A" 进行分组，并应用 np.mean 函数到 "C" 列
    expected = df.groupby("A")["C"].transform(np.mean)
    # 使用测试工具验证 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


def test_series_fast_transform_date():
    # 创建包含日期和分组信息的 DataFrame
    df = DataFrame(
        {"grouping": [np.nan, 1, 1, 3], "d": date_range("2014-1-1", "2014-1-4")}
    )
    # 对 DataFrame 按 "grouping" 列进行分组，并应用 "first" 函数到 "d" 列
    result = df.groupby("grouping")["d"].transform("first")
    # 预期的日期序列
    dates = [
        pd.NaT,
        Timestamp("2014-1-2"),
        Timestamp("2014-1-2"),
        Timestamp("2014-1-4"),
    ]
    expected = Series(dates, name="d", dtype="M8[ns]")
    # 使用测试工具验证 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", [lambda x: np.nansum(x), sum])
def test_transform_length(func):
    # 创建包含数值和 NaN 的 DataFrame
    df = DataFrame({"col1": [1, 1, 2, 2], "col2": [1, 2, 3, np.nan]})
    if func is sum:
        expected = Series([3.0, 3.0, np.nan, np.nan])
    else:
        expected = Series([3.0] * 4)

    # 对 DataFrame 按 "col1" 列进行分组，并应用指定的函数到 "col2" 列
    results = [
        df.groupby("col1").transform(func)["col2"],
        df.groupby("col1")["col2"].transform(func),
    ]
    for result in results:
        # 使用测试工具验证 result 和 expected 是否相等，不检查名称
        tm.assert_series_equal(result, expected, check_names=False)


def test_transform_coercion():
    # 14457
    # 创建包含字符串和数值的 DataFrame
    df = DataFrame({"A": ["a", "a", "b", "b"], "B": [0, 1, 3, 4]})
    # 对 DataFrame 按 "A" 列进行分组
    g = df.groupby("A")

    # 分别应用 np.mean 函数到每个分组的 DataFrame，并比较结果
    expected = g.transform(np.mean)
    result = g.transform(lambda x: np.mean(x, axis=0))
    # 使用测试工具验证 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


def test_groupby_transform_with_int():
    # GH 3740, 确保在 item-by-item transform 时能正确进行类型转换

    # 创建包含整数和浮点数的 DataFrame
    df = DataFrame(
        {
            "A": [1, 1, 1, 2, 2, 2],
            "B": Series(1, dtype="float64"),
            "C": Series([1, 2, 3, 1, 2, 3], dtype="float64"),
            "D": "foo",
        }
    )
    # 忽略所有 NumPy 的错误
    with np.errstate(all="ignore"):
        # 对 DataFrame 按 "A" 列进行分组，并应用标准化函数到 "B" 和 "C" 列
        result = df.groupby("A")[["B", "C"]].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    # 创建预期的 DataFrame，包含列 "B" 和 "C"，"B" 列填充 NaN，"C" 列填充特定的浮点数序列
    expected = DataFrame(
        {"B": np.nan, "C": Series([-1, 0, 1, -1, 0, 1], dtype="float64")}
    )
    # 使用测试工具比较 result 和 expected 的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # int 类型的情况
    df = DataFrame(
        {
            "A": [1, 1, 1, 2, 2, 2],
            "B": 1,
            "C": [1, 2, 3, 1, 2, 3],
            "D": "foo",
        }
    )
    # 忽略所有 NumPy 的错误
    with np.errstate(all="ignore"):
        # 使用 pytest 检查是否抛出 TypeError 异常，异常信息匹配 "Could not convert"
        with pytest.raises(TypeError, match="Could not convert"):
            # 对 df 按 "A" 列分组，对 "B" 和 "C" 列应用转换函数(lambda 函数)，计算每组数据的标准化值
            df.groupby("A").transform(lambda x: (x - x.mean()) / x.std())
        # 对 df 按 "A" 列分组，对 "B" 和 "C" 列应用转换函数(lambda 函数)，计算每组数据的标准化值
        result = df.groupby("A")[["B", "C"]].transform(
            lambda x: (x - x.mean()) / x.std()
        )
    # 创建预期的 DataFrame，包含列 "B" 和 "C"，"B" 列填充 NaN，"C" 列填充特定的浮点数序列
    expected = DataFrame({"B": np.nan, "C": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]})
    # 使用测试工具比较 result 和 expected 的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # 需要将 int 类型转换为 float 类型的情况
    s = Series([2, 3, 4, 10, 5, -1])
    df = DataFrame({"A": [1, 1, 1, 2, 2, 2], "B": 1, "C": s, "D": "foo"})
    # 忽略所有 NumPy 的错误
    with np.errstate(all="ignore"):
        # 使用 pytest 检查是否抛出 TypeError 异常，异常信息匹配 "Could not convert"
        with pytest.raises(TypeError, match="Could not convert"):
            # 对 df 按 "A" 列分组，对 "B" 和 "C" 列应用转换函数(lambda 函数)，计算每组数据的标准化值
            df.groupby("A").transform(lambda x: (x - x.mean()) / x.std())
        # 对 df 按 "A" 列分组，对 "B" 和 "C" 列应用转换函数(lambda 函数)，计算每组数据的标准化值
        result = df.groupby("A")[["B", "C"]].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    # 分别处理 s 的前三个元素和后三个元素，计算它们的标准化值并合并成一个 DataFrame
    s1 = s.iloc[0:3]
    s1 = (s1 - s1.mean()) / s1.std()
    s2 = s.iloc[3:6]
    s2 = (s2 - s2.mean()) / s2.std()
    # 创建预期的 DataFrame，包含列 "B" 和 "C"，"B" 列填充 NaN，"C" 列填充 s1 和 s2 的标准化结果
    expected = DataFrame({"B": np.nan, "C": concat([s1, s2])})
    # 使用测试工具比较 result 和 expected 的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # 不对 int 类型的数据进行降级转换
    result = df.groupby("A")[["B", "C"]].transform(lambda x: x * 2 / 2)
    # 创建预期的 DataFrame，包含列 "B" 和 "C"，"B" 列数据乘以 2 并除以 2，"C" 列保持不变
    expected = DataFrame({"B": 1.0, "C": [2.0, 3.0, 4.0, 10.0, 5.0, -1.0]})
    # 使用测试工具比较 result 和 expected 的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
# 测试处理带有 NaN 组的分组转换函数
def test_groupby_transform_with_nan_group():
    # GH 9941: GitHub issue number for reference
    # 创建包含两列 'a' 和 'b' 的 DataFrame，其中 'b' 列包含 NaN 值
    df = DataFrame({"a": range(10), "b": [1, 1, 2, 3, np.nan, 4, 4, 5, 5, 5]})
    # 对 DataFrame 按 'b' 列进行分组，并对每个分组的 'a' 列应用 max 函数
    result = df.groupby(df.b)["a"].transform(max)
    # 预期的结果是一个 Series，包含预期的最大值计算结果
    expected = Series([1.0, 1.0, 2.0, 3.0, np.nan, 6.0, 6.0, 9.0, 9.0, 9.0], name="a")
    # 使用测试工具函数验证结果与预期是否相等
    tm.assert_series_equal(result, expected)


# 测试处理混合类型数据的转换函数
def test_transform_mixed_type():
    # 创建一个具有多级索引的 DataFrame
    index = MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1], [1, 2, 3, 1, 2, 3]])
    df = DataFrame(
        {
            "d": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],  # 列 'd' 包含浮点数
            "c": np.tile(["a", "b", "c"], 2),   # 列 'c' 包含重复的字符串
            "v": np.arange(1.0, 7.0),           # 列 'v' 包含连续的浮点数
        },
        index=index,
    )

    # 定义一个函数 f，用于在每个分组上执行特定的操作
    def f(group):
        group["g"] = group["d"] * 2  # 在分组中添加一个新列 'g'，其值为 'd' 列的两倍
        return group[:1]             # 返回每个分组的第一行

    # 根据 'c' 列对 DataFrame 进行分组
    grouped = df.groupby("c")
    # 准备一个警告消息，用于测试时的验证
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用测试工具函数验证 apply 方法是否会产生特定的 DeprecationWarning 警告
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对分组对象应用函数 f
        result = grouped.apply(f)

    # 验证返回结果中 'd' 列的数据类型是否为 np.float64
    assert result["d"].dtype == np.float64

    # 验证对每个分组应用函数 f 后，结果是否与预期的结果一致
    for key, group in grouped:
        res = f(group)
        tm.assert_frame_equal(res, result.loc[key])


# 参数化测试函数，用于测试 Cython 转换函数的 Series 操作
@pytest.mark.parametrize(
    "op, args, targop",
    [
        ("cumprod", (), lambda x: x.cumprod()),         # 累积乘积操作
        ("cumsum", (), lambda x: x.cumsum()),           # 累积求和操作
        ("shift", (-1,), lambda x: x.shift(-1)),        # 向下移动一位操作
        ("shift", (1,), lambda x: x.shift()),           # 向上移动一位操作
    ],
)
def test_cython_transform_series(op, args, targop):
    # 创建一个包含随机标签和数据的 Series
    s = Series(np.random.default_rng(2).standard_normal(1000))
    # 复制 Series，并将其中一部分数据设置为 NaN
    s_missing = s.copy()
    s_missing.iloc[2:10] = np.nan
    # 创建随机生成的标签数组
    labels = np.random.default_rng(2).integers(0, 50, size=1000).astype(float)

    # 针对每个数据集（包含 NaN 和不含 NaN）执行以下操作
    for data in [s, s_missing]:
        # 计算预期的转换结果，使用 targop 函数
        expected = data.groupby(labels).transform(targop)

        # 使用测试工具函数验证两种方式（函数调用和属性调用）得到的结果是否与预期一致
        tm.assert_series_equal(expected, data.groupby(labels).transform(op, *args))
        tm.assert_series_equal(expected, getattr(data.groupby(labels), op)(*args))


@pytest.mark.parametrize("op", ["cumprod", "cumsum"])
@pytest.mark.parametrize(
    "input, exp",
    [
        # 当所有值都是 NaN 时的情况
        ({"key": ["b"] * 10, "value": np.nan}, Series([np.nan] * 10, name="value")),
        # 当只有一个值是 NaN 的情况
        (
            {"key": ["b"] * 10 + ["a"] * 2, "value": [3] * 3 + [np.nan] + [3] * 8},
            {
                # 对 ("cumprod", False) 的情况进行注释
                ("cumprod", False): [3.0, 9.0, 27.0] + [np.nan] * 7 + [3.0, 9.0],
                # 对 ("cumprod", True) 的情况进行注释
                ("cumprod", True): [
                    3.0,
                    9.0,
                    27.0,
                    np.nan,
                    81.0,
                    243.0,
                    729.0,
                    2187.0,
                    6561.0,
                    19683.0,
                    3.0,
                    9.0,
                ],
                # 对 ("cumsum", False) 的情况进行注释
                ("cumsum", False): [3.0, 6.0, 9.0] + [np.nan] * 7 + [3.0, 6.0],
                # 对 ("cumsum", True) 的情况进行注释
                ("cumsum", True): [
                    3.0,
                    6.0,
                    9.0,
                    np.nan,
                    12.0,
                    15.0,
                    18.0,
                    21.0,
                    24.0,
                    27.0,
                    3.0,
                    6.0,
                ],
            },
        ),
    ],
)
def test_groupby_cum_skipna(op, skipna, input, exp):
    # 根据输入创建 DataFrame 对象
    df = DataFrame(input)
    # 对指定分组键 "key" 下的 "value" 列执行 op 操作，并生成结果序列
    result = df.groupby("key")["value"].transform(op, skipna=skipna)
    # 根据期望值类型决定如何获取期望结果
    if isinstance(exp, dict):
        expected = exp[(op, skipna)]
    else:
        expected = exp
    # 将期望值转换为 Series 对象，列名为 "value"
    expected = Series(expected, name="value")
    # 使用测试工具库比较期望结果与实际结果
    tm.assert_series_equal(expected, result)


@pytest.fixture
def frame():
    # 创建包含随机浮点数数据的 Series 对象
    floating = Series(np.random.default_rng(2).standard_normal(10))
    # 复制浮点数数据，并在指定索引范围内插入缺失值
    floating_missing = floating.copy()
    floating_missing.iloc[2:7] = np.nan
    # 创建重复字符串列表
    strings = list("abcde") * 2
    # 复制字符串列表，并在指定索引位置插入缺失值
    strings_missing = strings[:]
    strings_missing[5] = np.nan

    # 创建 DataFrame 对象，包含多种数据类型的列
    df = DataFrame(
        {
            "float": floating,
            "float_missing": floating_missing,
            "int": [1, 1, 1, 1, 2] * 2,
            "datetime": date_range("1990-1-1", periods=10),
            "timedelta": pd.timedelta_range(1, freq="s", periods=10),
            "string": strings,
            "string_missing": strings_missing,
            "cat": Categorical(strings),
        },
    )
    return df


@pytest.fixture
def frame_mi(frame):
    # 创建 MultiIndex 索引并应用于 DataFrame
    frame.index = MultiIndex.from_product([range(5), range(2)])
    return frame


@pytest.mark.slow
@pytest.mark.parametrize(
    "op, args, targop",
    [
        ("cumprod", (), lambda x: x.cumprod()),  # 累积乘积操作
        ("cumsum", (), lambda x: x.cumsum()),    # 累积求和操作
        ("shift", (-1,), lambda x: x.shift(-1)),  # 向前移动一位操作
        ("shift", (1,), lambda x: x.shift()),     # 向后移动一位操作
    ],
)
@pytest.mark.parametrize("df_fix", ["frame", "frame_mi"])
@pytest.mark.parametrize(
    "gb_target",
    [
        {"by": np.random.default_rng(2).integers(0, 50, size=10).astype(float)},  # 按指定数组进行分组
        {"level": 0},  # 按多级索引的第一级进行分组
        {"by": "string"},  # 按字符串列进行分组
        pytest.param({"by": "string_missing"}, marks=pytest.mark.xfail),  # 预期分组失败
        {"by": ["int", "string"]},  # 按多列进行分组
    ],
)
def test_cython_transform_frame(request, op, args, targop, df_fix, gb_target):
    # 获取指定名称的测试数据集
    df = request.getfixturevalue(df_fix)
    # 根据指定的分组目标进行分组操作
    gb = df.groupby(group_keys=False, **gb_target)

    if op != "shift" and "int" not in gb_target:
        # 数值操作的快速路径会提升数据类型，因此需要分开应用并连接
        i = gb[["int"]].apply(targop)
        f = gb[["float", "float_missing"]].apply(targop)
        expected = concat([f, i], axis=1)
    else:
        if op != "shift" or not isinstance(gb_target.get("by"), (str, list)):
            warn = None
        else:
            warn = DeprecationWarning
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        # 验证操作在分组列上的使用是否会产生警告信息
        with tm.assert_produces_warning(warn, match=msg):
            expected = gb.apply(targop)

    # 对期望结果按列索引进行排序
    expected = expected.sort_index(axis=1)
    if op == "shift":
        # 对于 "shift" 操作，填充缺失值
        expected["string_missing"] = expected["string_missing"].fillna(np.nan)
        expected["string"] = expected["string"].fillna(np.nan)

    # 对实际结果进行转换操作，并按列索引进行排序
    result = gb[expected.columns].transform(op, *args).sort_index(axis=1)
    # 使用测试工具库比较转换后的结果与期望结果
    tm.assert_frame_equal(result, expected)
    # 对实际结果执行指定操作，并按列索引进行排序
    result = getattr(gb[expected.columns], op)(*args).sort_index(axis=1)
    # 使用测试工具tm.assert_frame_equal()比较result和expected两个数据框是否相等
        tm.assert_frame_equal(result, expected)
# 测试用例：cython_transform_frame_column
@pytest.mark.slow
# 参数化测试：op为cumprod时，使用lambda函数计算累积乘积
@pytest.mark.parametrize(
    "op, args, targop",
    [
        ("cumprod", (), lambda x: x.cumprod()),
        # 参数化测试：op为cumsum时，使用lambda函数计算累积和
        ("cumsum", (), lambda x: x.cumsum()),
        # 参数化测试：op为shift且参数为-1时，使用lambda函数实现向上移位
        ("shift", (-1,), lambda x: x.shift(-1)),
        # 参数化测试：op为shift且参数为1时，使用lambda函数实现向下移位
        ("shift", (1,), lambda x: x.shift()),
    ],
)
# 参数化测试：df_fix为'frame'或'frame_mi'，分别对应不同的数据框架
@pytest.mark.parametrize("df_fix", ["frame", "frame_mi"])
# 参数化测试：gb_target为不同的分组目标，包括随机生成的浮点数数组，级别0，字符串，整数和字符串数组
@pytest.mark.parametrize(
    "gb_target",
    [
        {"by": np.random.default_rng(2).integers(0, 50, size=10).astype(float)},
        {"level": 0},
        {"by": "string"},
        # TODO: create xfail condition given other params
        # {"by": 'string_missing'},
        {"by": ["int", "string"]},
    ],
)
# 参数化测试：column为不同的列类型，包括浮点数，缺失浮点数，整数，日期时间，时间间隔，字符串和缺失字符串
@pytest.mark.parametrize(
    "column",
    [
        "float",
        "float_missing",
        "int",
        "datetime",
        "timedelta",
        "string",
        "string_missing",
    ],
)
# 测试函数：cython_transform_frame_column，测试Cython优化的数据框列转换操作
def test_cython_transform_frame_column(
    request, op, args, targop, df_fix, gb_target, column
):
    # 获取测试框架，根据df_fix参数选择使用'frame'或'frame_mi'
    df = request.getfixturevalue(df_fix)
    # 按指定分组目标进行分组，group_keys=False表示不返回分组键
    gb = df.groupby(group_keys=False, **gb_target)
    # 设置当前列为变量c
    c = column
    # 如果列c不是"float"、"int"、"float_missing"，且操作不是"shift"，
    # 或者如果列是"timedelta"且操作是"cumsum"，则抛出TypeError异常
    if (
        c not in ["float", "int", "float_missing"]
        and op != "shift"
        and not (c == "timedelta" and op == "cumsum")
    ):
        # 错误消息
        msg = "|".join(
            [
                "does not support .* operations",
                "does not support operation",
                ".* is not supported for object dtype",
                "is not implemented for this dtype",
            ]
        )
        # 断言抛出指定消息的TypeError异常
        with pytest.raises(TypeError, match=msg):
            gb[c].transform(op)
        with pytest.raises(TypeError, match=msg):
            getattr(gb[c], op)()
    else:
        # 预期结果：使用apply函数应用targop函数到列c
        expected = gb[c].apply(targop)
        expected.name = c
        # 如果列c是"string_missing"或"string"，则将预期结果中的缺失值填充为NaN
        if c in ["string_missing", "string"]:
            expected = expected.fillna(np.nan)

        # 进行列转换操作op，并与预期结果expected进行比较
        res = gb[c].transform(op, *args)
        tm.assert_series_equal(expected, res)
        # 调用列c上的操作op，并与预期结果expected进行比较
        res2 = getattr(gb[c], op)(*args)
        tm.assert_series_equal(expected, res2)


# 测试函数：test_transform_numeric_ret，测试数值列转换的返回结果
@pytest.mark.parametrize(
    "cols,expected",
    [
        # 参数化测试：cols为'a'时，预期返回Series([1, 1, 1], name="a")
        ("a", Series([1, 1, 1], name="a")),
        (
            # 参数化测试：cols为['a', 'c']时，预期返回DataFrame({"a": [1, 1, 1], "c": [1, 1, 1]})
            ["a", "c"],
            DataFrame({"a": [1, 1, 1], "c": [1, 1, 1]}),
        ),
    ],
)
# 参数化测试：agg_func为'count'、'rank'或'size'，分别测试不同的聚合函数
@pytest.mark.parametrize("agg_func", ["count", "rank", "size"])
# 测试函数：test_transform_numeric_ret，测试数值列转换的返回结果
def test_transform_numeric_ret(cols, expected, agg_func):
    # 创建数据框架df，包括列'a'为日期范围，列'b'为0到2的整数，列'c'为7到9的整数
    df = DataFrame(
        {"a": date_range("2018-01-01", periods=3), "b": range(3), "c": range(7, 10)}
    )
    # 对数据框架df按列'b'分组，对列cols应用聚合函数agg_func，并将结果保存为result
    result = df.groupby("b")[cols].transform(agg_func)

    # 如果聚合函数为'rank'，则将预期结果expected转换为浮点数类型
    if agg_func == "rank":
        expected = expected.astype("float")
    # 如果聚合函数为'size'且cols为['a', 'c']，则将预期结果expected中的'a'列重命名为None
    elif agg_func == "size" and cols == ["a", "c"]:
        expected = expected["a"].rename(None)
    # 断言结果result与预期结果expected相等
    tm.assert_equal(result, expected)


# 测试函数：test_transform_ffill，测试前向填充操作
def test_transform_ffill():
    # 创建数据列表data，包括键'key'和值'values'
    data = [["a", 0.0], ["a", float("nan")], ["b", 1.0], ["b", float("nan")]]
    # 创建数据框架df，列名分别为'key'和'values'
    df = DataFrame(data, columns=["key", "values"])
    # 对数据框架df按'key'列分组，并进行前向填充操作，将结果保存为result
    result = df.groupby("key").transform("ffill")
    # 创建一个预期的 DataFrame 对象，包含一列名为 "values" 的数据，数据分别为 [0.0, 0.0, 1.0, 1.0]
    expected = DataFrame({"values": [0.0, 0.0, 1.0, 1.0]})
    # 使用测试框架中的方法比较 result 和 expected，确认它们是否相等
    tm.assert_frame_equal(result, expected)
    
    # 对 DataFrame df 进行分组操作，按照 "key" 列分组，然后对 "values" 列应用前向填充（ffill）操作
    result = df.groupby("key")["values"].transform("ffill")
    
    # 创建一个预期的 Series 对象，包含一列名为 "values" 的数据，数据为 [0.0, 0.0, 1.0, 1.0]
    expected = Series([0.0, 0.0, 1.0, 1.0], name="values")
    # 使用测试框架中的方法比较 result 和 expected，确认它们是否相等
    tm.assert_series_equal(result, expected)
# 使用 pytest 的 parametrize 装饰器为 test_group_fill_methods 函数定义多组参数化测试用例
@pytest.mark.parametrize("mix_groupings", [True, False])
@pytest.mark.parametrize("as_series", [True, False])
@pytest.mark.parametrize("val1,val2", [("foo", "bar"), (1, 2), (1.0, 2.0)])
@pytest.mark.parametrize(
    "fill_method,limit,exp_vals",
    [
        (
            "ffill",
            None,
            [np.nan, np.nan, "val1", "val1", "val1", "val2", "val2", "val2"],
        ),
        ("ffill", 1, [np.nan, np.nan, "val1", "val1", np.nan, "val2", "val2", np.nan]),
        (
            "bfill",
            None,
            ["val1", "val1", "val1", "val2", "val2", "val2", np.nan, np.nan],
        ),
        ("bfill", 1, [np.nan, "val1", "val1", np.nan, "val2", "val2", np.nan, np.nan]),
    ],
)
# 定义测试函数 test_group_fill_methods，接收多个参数化的输入
def test_group_fill_methods(
    mix_groupings, as_series, val1, val2, fill_method, limit, exp_vals
):
    # 创建一个包含 NaN 的值列表 vals，用于测试
    vals = [np.nan, np.nan, val1, np.nan, np.nan, val2, np.nan, np.nan]
    # 复制 exp_vals 列表，以便后续修改
    _exp_vals = list(exp_vals)
    
    # 根据 exp_vals 中的占位符值将其替换为具体的 val1 或 val2 值
    for index, exp_val in enumerate(_exp_vals):
        if exp_val == "val1":
            _exp_vals[index] = val1
        elif exp_val == "val2":
            _exp_vals[index] = val2

    # 根据 mix_groupings 决定生成的 keys 列表是交替还是重复的 ['a', 'b'] 组合
    if mix_groupings:  # ['a', 'b', 'a, 'b', ...]
        keys = ["a", "b"] * len(vals)

        # 定义函数 interweave，将列表中的元素交替插入，用于修改 _exp_vals 和 vals
        def interweave(list_obj):
            temp = []
            for x in list_obj:
                temp.extend([x, x])

            return temp

        # 对 _exp_vals 和 vals 应用 interweave 函数，使其交替插入
        _exp_vals = interweave(_exp_vals)
        vals = interweave(vals)
    else:  # ['a', 'a', 'a', ... 'b', 'b', 'b']
        keys = ["a"] * len(vals) + ["b"] * len(vals)
        _exp_vals = _exp_vals * 2
        vals = vals * 2

    # 创建 DataFrame 对象 df，包含 key 和 val 列
    df = DataFrame({"key": keys, "val": vals})
    # 根据 as_series 决定是对 Series 还是 DataFrame 进行填充操作
    if as_series:
        # 调用 df.groupby("key")["val"] 的 fill_method 方法进行填充操作
        result = getattr(df.groupby("key")["val"], fill_method)(limit=limit)
        # 创建预期结果 Series exp，以 _exp_vals 为数据，列名为 "val"
        exp = Series(_exp_vals, name="val")
        # 使用 pytest 的 assert_series_equal 函数比较 result 和 exp
        tm.assert_series_equal(result, exp)
    else:
        # 调用 df.groupby("key") 的 fill_method 方法进行填充操作
        result = getattr(df.groupby("key"), fill_method)(limit=limit)
        # 创建预期结果 DataFrame exp，包含一列 "val"，数据为 _exp_vals
        exp = DataFrame({"val": _exp_vals})
        # 使用 pytest 的 assert_frame_equal 函数比较 result 和 exp
        tm.assert_frame_equal(result, exp)


# 使用 pytest 的 parametrize 装饰器为 test_pad_stable_sorting 函数定义多组参数化测试用例
@pytest.mark.parametrize("fill_method", ["ffill", "bfill"])
# 定义测试函数 test_pad_stable_sorting，接收填充方法 fill_method 作为参数
def test_pad_stable_sorting(fill_method):
    # 创建两个列表 x 和 y，分别包含多个 0 和 NaN 或 1 的元素
    x = [0] * 20
    y = [np.nan] * 10 + [1] * 10

    # 根据 fill_method 是否为 "bfill" 反转列表 y 的顺序
    if fill_method == "bfill":
        y = y[::-1]

    # 创建 DataFrame 对象 df，包含两列 "x" 和 "y"
    df = DataFrame({"x": x, "y": y})
    # 从 df 中删除 "x" 列，作为预期结果 expected
    expected = df.drop("x", axis=1)

    # 调用 df.groupby("x") 的 fill_method 方法进行填充操作，生成结果 result
    result = getattr(df.groupby("x"), fill_method)()

    # 使用 pytest 的 assert_frame_equal 函数比较 result 和 expected
    tm.assert_frame_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器为 test_pct_change 函数定义多组参数化测试用例
@pytest.mark.parametrize(
    "freq",
    [
        None,
        pytest.param(
            "D",
            marks=pytest.mark.xfail(
                reason="GH#23918 before method uses freq in vectorized approach"
            ),
        ),
    ],
)
@pytest.mark.parametrize("periods", [1, -1])
# 定义测试函数 test_pct_change，接收 frame_or_series、freq 和 periods 作为参数
def test_pct_change(frame_or_series, freq, periods):
    # 创建一个包含数值和 NaN 的 vals 列表
    vals = [3, np.nan, np.nan, np.nan, 1, 2, 4, 10, np.nan, 4]
    # 创建一个包含 "a" 和 "b" 的 keys 列表
    keys = ["a", "b"]
    # 将键重复以匹配值的长度，创建包含键值对的 DataFrame
    key_v = np.repeat(keys, len(vals))
    df = DataFrame({"key": key_v, "vals": vals * 2})

    # 将 df_g 设置为 df 的引用
    df_g = df
    # 按键分组 df_g
    grp = df_g.groupby(df.key)

    # 计算预期值，即每个组内 vals 列的百分比变化
    expected = grp["vals"].obj / grp["vals"].shift(periods) - 1

    # 按键分组 df
    gb = df.groupby("key")

    # 根据 frame_or_series 的类型，选择要操作的数据
    if frame_or_series is Series:
        gb = gb["vals"]  # 如果是 Series，则只操作 vals 列
    else:
        expected = expected.to_frame("vals")  # 否则，将预期结果转换为 DataFrame 格式

    # 计算组内的百分比变化
    result = gb.pct_change(periods=periods, freq=freq)
    # 使用断言验证结果与预期是否一致
    tm.assert_equal(result, expected)
@pytest.mark.parametrize(
    "func, expected_status",
    [
        ("ffill", ["shrt", "shrt", "lng", np.nan, "shrt", "ntrl", "ntrl"]),
        ("bfill", ["shrt", "lng", "lng", "shrt", "shrt", "ntrl", np.nan]),
    ],
)
def test_ffill_bfill_non_unique_multilevel(func, expected_status):
    # 设置测试参数和预期结果，测试前向填充和后向填充函数在多级索引中的行为
    # GH 19437
    date = pd.to_datetime(
        [
            "2018-01-01",
            "2018-01-01",
            "2018-01-01",
            "2018-01-01",
            "2018-01-02",
            "2018-01-01",
            "2018-01-02",
        ]
    )
    # 设置符号列和状态列
    symbol = ["MSFT", "MSFT", "MSFT", "AAPL", "AAPL", "TSLA", "TSLA"]
    status = ["shrt", np.nan, "lng", np.nan, "shrt", "ntrl", np.nan]

    # 创建包含日期、符号和状态的数据帧
    df = DataFrame({"date": date, "symbol": symbol, "status": status})
    # 将日期和符号设置为索引
    df = df.set_index(["date", "symbol"])
    # 使用 getattr 动态调用 groupby 对象的函数（ffill 或 bfill）
    result = getattr(df.groupby("symbol")["status"], func)()

    # 创建预期结果的多级索引
    index = MultiIndex.from_tuples(
        tuples=list(zip(*[date, symbol])), names=["date", "symbol"]
    )
    # 创建预期的 Series 结果
    expected = Series(expected_status, index=index, name="status")

    # 使用 assert_series_equal 检查实际结果和预期结果是否一致
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", [np.any, np.all])
def test_any_all_np_func(func):
    # 测试 np.any 和 np.all 函数在数据帧中的行为
    # GH 20653
    df = DataFrame(
        [["foo", True], [np.nan, True], ["foo", True]], columns=["key", "val"]
    )

    # 创建预期的 Series 结果
    exp = Series([True, np.nan, True], name="val")

    # 使用 transform 对 groupby 对象的 val 列应用 func 函数
    res = df.groupby("key")["val"].transform(func)
    # 使用 assert_series_equal 检查实际结果和预期结果是否一致
    tm.assert_series_equal(res, exp)


def test_groupby_transform_rename():
    # 测试自定义函数 demean_rename 在数据帧的 groupby 变换中的行为
    # https://github.com/pandas-dev/pandas/issues/23461
    def demean_rename(x):
        # 计算 x - x 的均值
        result = x - x.mean()

        if isinstance(x, Series):
            return result

        # 重命名结果的列名
        result = result.rename(columns={c: f"{c}_demeaned" for c in result.columns})

        return result

    # 创建包含 group 和 value 列的数据帧
    df = DataFrame({"group": list("ababa"), "value": [1, 1, 1, 2, 2]})
    # 创建预期的 DataFrame 结果
    expected = DataFrame({"value": [-1.0 / 3, -0.5, -1.0 / 3, 0.5, 2.0 / 3]})

    # 使用 transform 对 groupby 对象应用 demean_rename 函数
    result = df.groupby("group").transform(demean_rename)
    # 使用 assert_frame_equal 检查实际结果和预期结果是否一致
    tm.assert_frame_equal(result, expected)
    # 对 groupby 对象的 value 列应用 demean_rename 函数
    result_single = df.groupby("group").value.transform(demean_rename)
    # 使用 assert_series_equal 检查实际结果和预期结果是否一致
    tm.assert_series_equal(result_single, expected["value"])


@pytest.mark.parametrize("func", [min, max, np.min, np.max, "first", "last"])
def test_groupby_transform_timezone_column(func):
    # 测试时区列变换中函数 func 的行为
    # GH 24198
    # 创建当前时间的带时区的时间戳
    ts = pd.to_datetime("now", utc=True).tz_convert("Asia/Singapore")
    # 创建包含 end_time 和 id 列的数据帧
    result = DataFrame({"end_time": [ts], "id": [1]})
    # 使用 transform 对 groupby 对象的 end_time 列应用 func 函数
    result["max_end_time"] = result.groupby("id").end_time.transform(func)
    # 创建预期的 DataFrame 结果
    expected = DataFrame([[ts, 1, ts]], columns=["end_time", "id", "max_end_time"])
    # 使用 assert_frame_equal 检查实际结果和预期结果是否一致
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "func, values",
    [
        ("idxmin", ["1/1/2011"] * 2 + ["1/3/2011"] * 7 + ["1/10/2011"]),
        ("idxmax", ["1/2/2011"] * 2 + ["1/9/2011"] * 7 + ["1/10/2011"]),
    ],
)
def test_groupby_transform_with_datetimes(func, values):
    # 测试带有日期时间的 groupby 变换中函数 func 的行为
    # GH 15306
    # 创建日期范围
    dates = date_range("1/1/2011", periods=10, freq="D")
    # 创建一个 DataFrame 对象 `stocks`，其中包含价格列 `price`，价格从 0 到 9，索引为给定的日期序列 `dates`
    stocks = DataFrame({"price": np.arange(10.0)}, index=dates)
    
    # 向 `stocks` DataFrame 添加一列 `week_id`，用于存储每个日期对应的 ISO 日历周数
    stocks["week_id"] = dates.isocalendar().week
    
    # 对 `stocks` DataFrame 按照 `week_id` 列进行分组，然后对每组中的 `price` 列应用自定义函数 `func` 进行转换
    result = stocks.groupby(stocks["week_id"])["price"].transform(func)
    
    # 创建一个预期的 Series 对象 `expected`，其中的数据是将给定的日期数据转换为纳秒单位的时间戳，并使用 `price` 作为列名
    expected = Series(
        data=pd.to_datetime(values).as_unit("ns"), index=dates, name="price"
    )
    
    # 使用测试工具库 `tm` 中的函数 `assert_series_equal` 检查 `result` 和 `expected` 两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
def test_groupby_transform_dtype():
    # GH 22243
    # 创建一个包含两列的 DataFrame，其中 "a" 列为整数 1，"val" 列包含浮点数 1.35
    df = DataFrame({"a": [1], "val": [1.35]})

    # 对 "val" 列进行 transform，使用 lambda 函数将每个元素前面加上 "+"
    result = df["val"].transform(lambda x: x.map(lambda y: f"+{y}"))
    # 期望的结果是一个 Series，包含["+1.35"]，数据类型为对象型
    expected1 = Series(["+1.35"], name="val", dtype="object")
    tm.assert_series_equal(result, expected1)

    # 对 "val" 列根据 "a" 列进行分组，并进行 transform 操作，将每个元素前面加上 "+"
    result = df.groupby("a")["val"].transform(lambda x: x.map(lambda y: f"+{y}"))
    tm.assert_series_equal(result, expected1)

    # 对 "val" 列根据 "a" 列进行分组，并进行 transform 操作，将每个元素加上 "+(" 和 ")"
    result = df.groupby("a")["val"].transform(lambda x: x.map(lambda y: f"+({y})"))
    # 期望的结果是一个 Series，包含["+(1.35)"]，数据类型为对象型
    expected2 = Series(["+(1.35)"], name="val", dtype="object")
    tm.assert_series_equal(result, expected2)

    # 将 "val" 列的数据类型转换为对象型
    df["val"] = df["val"].astype(object)
    # 对 "val" 列根据 "a" 列进行分组，并进行 transform 操作，将每个元素前面加上 "+"
    result = df.groupby("a")["val"].transform(lambda x: x.map(lambda y: f"+{y}"))
    tm.assert_series_equal(result, expected1)


def test_transform_absent_categories(all_numeric_accumulations):
    # GH 16771
    # 使用 Cython 进行转换，涉及到的组数比行数多
    x_vals = [1]
    x_cats = range(2)
    y = [1]
    # 创建一个 DataFrame，包含 "x" 列为分类数据，"y" 列为整数数据
    df = DataFrame({"x": Categorical(x_vals, x_cats), "y": y})
    # 对 "y" 列根据 "x" 列进行分组，使用给定的聚合函数进行 transform 操作
    result = getattr(df.y.groupby(df.x, observed=False), all_numeric_accumulations)()
    expected = df.y
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", ["ffill", "bfill", "shift"])
@pytest.mark.parametrize("key, val", [("level", 0), ("by", Series([0]))])
def test_ffill_not_in_axis(func, key, val):
    # GH 21521
    # 创建一个包含 NaN 的 DataFrame
    df = DataFrame([[np.nan]])
    # 对 DataFrame 根据指定的 key 和 val 进行分组，并进行前向填充、后向填充或数据移动的操作
    result = getattr(df.groupby(**{key: val}), func)()
    expected = df

    tm.assert_frame_equal(result, expected)


def test_transform_invalid_name_raises():
    # GH#27486
    # 创建一个包含 "a" 列的 DataFrame
    df = DataFrame({"a": [0, 1, 1, 2]})
    # 根据 ["a", "b", "b", "c"] 列表对 DataFrame 进行分组
    g = df.groupby(["a", "b", "b", "c"])
    # 使用 pytest 检测，应该会抛出 ValueError 异常，异常信息中包含 "not a valid function name"
    with pytest.raises(ValueError, match="not a valid function name"):
        g.transform("some_arbitrary_name")

    # 确保 DataFrameGroupBy 对象具有 "aggregate" 方法
    assert hasattr(g, "aggregate")
    # 使用 pytest 检测，应该会抛出 ValueError 异常，异常信息中包含 "not a valid function name"
    with pytest.raises(ValueError, match="not a valid function name"):
        g.transform("aggregate")

    # 对 SeriesGroupBy 进行测试
    g = df["a"].groupby(["a", "b", "b", "c"])
    # 使用 pytest 检测，应该会抛出 ValueError 异常，异常信息中包含 "not a valid function name"
    with pytest.raises(ValueError, match="not a valid function name"):
        g.transform("some_arbitrary_name")


def test_transform_agg_by_name(request, reduction_func, frame_or_series):
    func = reduction_func

    # 创建一个包含 "a" 和 "b" 两列的 DataFrame，索引为 ["A", "B", "C", "D", "E", "F"]
    obj = DataFrame(
        {"a": [0, 0, 0, 1, 1, 1], "b": range(6)},
        index=["A", "B", "C", "D", "E", "F"],
    )
    if frame_or_series is Series:
        obj = obj["a"]

    # 根据 np.repeat([0, 1], 3) 对 obj 进行分组
    g = obj.groupby(np.repeat([0, 1], 3))

    if func == "corrwith" and isinstance(obj, Series):
        # 如果 func 是 "corrwith" 并且 obj 是 Series，则不执行下面的操作
        assert not hasattr(g, func)
        return

    # 获取聚合方法的参数
    args = get_groupby_method_args(reduction_func, obj)
    if func == "corrwith":
        warn = FutureWarning
        msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        msg = ""
    # 使用 pytest 的上下文管理器 assert_produces_warning 检查是否产生了特定的警告信息
    with tm.assert_produces_warning(warn, match=msg):
        # 调用 g 对象的 transform 方法进行数据转换，使用 func 函数和 args 参数
        result = g.transform(func, *args)

    # 断言结果 result 的索引与原始数据对象 obj 的索引相等
    # 这行代码是验证转换操作的定义
    tm.assert_index_equal(result.index, obj.index)

    # 如果 func 不是 "ngroup" 或 "size"，并且 obj 是二维的
    if func not in ("ngroup", "size") and obj.ndim == 2:
        # 对于 size/ngroup 返回的是 Series，不像其他的转换操作返回 DataFrame
        # 断言结果 result 的列索引与原始数据对象 obj 的列索引相等
        tm.assert_index_equal(result.columns, obj.columns)

    # 验证值是否在每个组中进行了广播传播
    # 断言最后一列的倒数三行数据值是相同的，确保值在每个组内广播传播
    assert len(set(DataFrame(result).iloc[-3:, -1])) == 1
# GH 27496
# 创建一个包含日期时间和时区信息的数据框
df = DataFrame(
    {
        "time": [
            Timestamp("2010-07-15 03:14:45"),
            Timestamp("2010-11-19 18:47:06"),
        ],
        "timezone": ["Etc/GMT+4", "US/Eastern"],
    }
)
# 使用 groupby 和 transform 函数，根据时区对时间列进行本地化
result = df.groupby(["timezone"])["time"].transform(
    lambda x: x.dt.tz_localize(x.name)
)
# 预期的结果，包含已本地化时区的时间戳序列
expected = Series(
    [
        Timestamp("2010-07-15 03:14:45", tz="Etc/GMT+4"),
        Timestamp("2010-11-19 18:47:06", tz="US/Eastern"),
    ],
    name="time",
)
# 断言结果与预期相等
tm.assert_series_equal(result, expected)


# GH#29631
# 创建一个包含数值和分组信息的数据框
df = DataFrame({"A": [1, 1, 2, 2], "B": [1, -1, 1, 2]})
# 根据列"A"进行分组
gb = df.groupby("A")

def func(grp):
    # 定义一个函数，用于处理每个分组的数据帧
    if grp.ndim == 2:
        # 如果数据帧的维度为2，则引发未实现错误
        raise NotImplementedError("Don't cross the streams")
    # 返回每个分组乘以2后的结果
    return grp * 2

# 获取分组对象的一些属性和迭代器
obj = gb._obj_with_exclusions
gen = gb._grouper.get_iterator(obj)
# 定义两种转换路径，快速和慢速
fast_path, slow_path = gb._define_paths(func)
# 获取第一个分组
_, group = next(gen)

# 使用 pytest 检查快速路径是否引发了预期的错误
with pytest.raises(NotImplementedError, match="Don't cross the streams"):
    fast_path(group)

# 对数据框进行转换操作，并检查结果与预期是否相等
result = gb.transform(func)
expected = DataFrame([2, -2, 2, 4], columns=["B"])
tm.assert_frame_equal(result, expected)


# GH 7883
# 创建一个包含多级索引的数据框
df = DataFrame(
    {
        "A": ["foo", "bar", "foo", "bar", "foo", "flux", "foo", "flux"],
        "B": ["one", "one", "two", "three", "two", "six", "five", "three"],
        "C": range(8),
        "D": range(8),
        "E": range(8),
    }
)
# 将"A"和"B"列设置为索引，并按索引排序
df = df.set_index(["A", "B"])
df = df.sort_index()
# 对每个"A"组进行转换，使每个组的值都等于其最后一个元素
result = df.groupby(level="A").transform(lambda x: x.iloc[-1])
# 预期的结果，包含按索引最后元素值转换后的数据框
expected = DataFrame(
    {
        "C": [3, 3, 7, 7, 4, 4, 4, 4],
        "D": [3, 3, 7, 7, 4, 4, 4, 4],
        "E": [3, 3, 7, 7, 4, 4, 4, 4],
    },
    index=MultiIndex.from_tuples(
        [
            ("bar", "one"),
            ("bar", "three"),
            ("flux", "six"),
            ("flux", "three"),
            ("foo", "five"),
            ("foo", "one"),
            ("foo", "two"),
            ("foo", "two"),
        ],
        names=["A", "B"],
    ),
)
# 断言结果与预期相等
tm.assert_frame_equal(result, expected)


# GH 32494
# 检查 groupby-transform 是否在同时使用分类和非分类键分组时
# 不会尝试扩展输出以包含非观察到的分类，而是匹配输入形状
# 此测试用例暂未提供代码内容
    df_with_categorical = DataFrame(
        {
            "A": Categorical(["a", "b", "a"], categories=["a", "b", "c"]),
            "B": [1, 2, 3],
            "C": ["a", "b", "a"],
        }
    )
    # 创建一个包含分类数据的 DataFrame，列'A'使用了预定义的分类，列'B'和'C'是普通列
    df_without_categorical = DataFrame(
        {"A": ["a", "b", "a"], "B": [1, 2, 3], "C": ["a", "b", "a"]}
    )

    # DataFrame 情况下
    # 对 df_with_categorical 按照列'A'和'C'进行分组，并应用 'sum' 聚合函数
    result = df_with_categorical.groupby(["A", "C"], observed=observed).transform("sum")
    # 对 df_without_categorical 按照列'A'和'C'进行分组，并应用 'sum' 聚合函数
    expected = df_without_categorical.groupby(["A", "C"]).transform("sum")
    # 使用测试框架检查结果是否相等
    tm.assert_frame_equal(result, expected)
    # 期望的显式结果 DataFrame，只包含 'B' 列的聚合结果
    expected_explicit = DataFrame({"B": [4, 2, 4]})
    # 使用测试框架检查结果是否与期望的显式结果相等
    tm.assert_frame_equal(result, expected_explicit)

    # Series 情况下
    # 对 df_with_categorical 按照列'A'和'C'进行分组，返回分组后的 'B' 列
    gb = df_with_categorical.groupby(["A", "C"], observed=observed)
    gbp = gb["B"]
    # 对分组后的 'B' 列应用 'sum' 聚合函数
    result = gbp.transform("sum")
    # 对 df_without_categorical 按照列'A'和'C'进行分组，返回分组后的 'B' 列，并应用 'sum' 聚合函数
    expected = df_without_categorical.groupby(["A", "C"])["B"].transform("sum")
    # 使用测试框架检查结果是否相等
    tm.assert_series_equal(result, expected)
    # 期望的显式结果 Series，包含与结果相同的 'B' 列聚合结果
    expected_explicit = Series([4, 2, 4], name="B")
    # 使用测试框架检查结果是否与期望的显式结果相等
    tm.assert_series_equal(result, expected_explicit)
def test_string_rank_grouping():
    # GH 19354
    # 创建一个包含两列的数据框，列'A'中有重复的值
    df = DataFrame({"A": [1, 1, 2], "B": [1, 2, 3]})
    # 对分组后的每个组应用 rank 函数，返回排名结果
    result = df.groupby("A").transform("rank")
    # 预期的数据框，包含列'B'中的排名结果
    expected = DataFrame({"B": [1.0, 2.0, 1.0]})
    # 使用测试工具检查 result 是否等于 expected
    tm.assert_frame_equal(result, expected)


def test_transform_cumcount():
    # GH 27472
    # 创建一个包含两列的数据框，列'a'中有多个重复值
    df = DataFrame({"a": [0, 0, 0, 1, 1, 1], "b": range(6)})
    # 根据数组 np.repeat([0, 1], 3) 对数据框进行分组
    grp = df.groupby(np.repeat([0, 1], 3))

    # 对每个组应用 cumcount 函数，返回计数结果
    result = grp.cumcount()
    # 预期的序列，包含计数结果
    expected = Series([0, 1, 2, 0, 1, 2])
    # 使用测试工具检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)

    # 对每个组应用 transform 函数，并返回计数结果
    result = grp.transform("cumcount")
    # 使用测试工具检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("keys", [["A1"], ["A1", "A2"]])
def test_null_group_lambda_self(sort, dropna, keys):
    # GH 17093
    size = 50
    # 创建一个包含随机生成的布尔值数组，表示每个元素是否为空
    nulls1 = np.random.default_rng(2).choice([False, True], size)
    nulls2 = np.random.default_rng(2).choice([False, True], size)
    # 判断每个组是否包含空值
    nulls_grouper = nulls1 if len(keys) == 1 else nulls1 | nulls2

    # 创建一个包含随机生成的浮点数数组，部分元素设置为 NaN
    a1 = np.random.default_rng(2).integers(0, 5, size=size).astype(float)
    a1[nulls1] = np.nan
    a2 = np.random.default_rng(2).integers(0, 5, size=size).astype(float)
    a2[nulls2] = np.nan
    values = np.random.default_rng(2).integers(0, 5, size=a1.shape)
    # 创建一个数据框，包含列'A1', 'A2', 'B'，其中部分元素可能为 NaN
    df = DataFrame({"A1": a1, "A2": a2, "B": values})

    # 如果 dropna 为 True 且组中存在空值，则将期望值中的相应元素设置为 NaN
    expected_values = values
    if dropna and nulls_grouper.any():
        expected_values = expected_values.astype(float)
        expected_values[nulls_grouper] = np.nan
    expected = DataFrame(expected_values, columns=["B"])

    # 根据 keys 对数据框进行分组
    gb = df.groupby(keys, dropna=dropna, sort=sort)
    # 对每个组中的'B'列应用 lambda 函数，返回结果
    result = gb[["B"]].transform(lambda x: x)
    # 使用测试工具检查 result 是否等于 expected
    tm.assert_frame_equal(result, expected)


def test_null_group_str_reducer(request, dropna, reduction_func):
    # GH 17093
    if reduction_func == "corrwith":
        msg = "incorrectly raises"
        request.applymarker(pytest.mark.xfail(reason=msg))

    index = [1, 2, 3, 4]  # 测试 transform 保留非标准索引
    # 创建一个包含两列的数据框，其中一列可能包含 NaN
    df = DataFrame({"A": [1, 1, np.nan, np.nan], "B": [1, 2, 2, 3]}, index=index)
    # 根据'A'列对数据框进行分组
    gb = df.groupby("A", dropna=dropna)

    # 获取分组操作的参数
    args = get_groupby_method_args(reduction_func, df)

    # 手动处理不符合通用模式的 reducer
    # 如果 reduction_func 是特定的聚合函数，则设置 dropna=False 的期望值，然后根据需要替换
    if reduction_func == "first":
        expected = DataFrame({"B": [1, 1, 2, 2]}, index=index)
    elif reduction_func == "last":
        expected = DataFrame({"B": [2, 2, 3, 3]}, index=index)
    elif reduction_func == "nth":
        expected = DataFrame({"B": [1, 1, 2, 2]}, index=index)
    elif reduction_func == "size":
        expected = Series([2, 2, 2, 2], index=index)
    elif reduction_func == "corrwith":
        expected = DataFrame({"B": [1.0, 1.0, 1.0, 1.0]}, index=index)
    # 如果dropna为False，按"A"列分组DataFrame，并创建一个分组对象expected_gb
    expected_gb = df.groupby("A", dropna=False)
    # 初始化一个空列表buffer，用于存储每个分组经过reduction_func函数处理后的结果
    buffer = []
    # 遍历expected_gb分组对象，idx为分组的索引，group为分组的数据
    for idx, group in expected_gb:
        # 使用getattr函数获取group["B"]列的reduction_func函数的结果，添加到buffer中
        res = getattr(group["B"], reduction_func)()
        buffer.append(Series(res, index=group.index))
    # 将buffer中的结果连接起来并转换为DataFrame，列名为"B"，赋值给变量expected
    expected = concat(buffer).to_frame("B")
    
    # 如果dropna为True，根据reduction_func函数的值确定dtype的类型
    if dropna:
        dtype = object if reduction_func in ("any", "all") else float
        # 将expected转换为指定的dtype类型
        expected = expected.astype(dtype)
        # 如果expected的维度为2，则将第2行和第3行，第1列的元素设为NaN
        if expected.ndim == 2:
            expected.iloc[[2, 3], 0] = np.nan
        # 否则，将第2行和第3行的元素设为NaN
        else:
            expected.iloc[[2, 3]] = np.nan

    # 对分组对象gb应用reduction_func函数，传入参数args，并将结果赋值给result
    result = gb.transform(reduction_func, *args)
    # 使用tm.assert_equal函数断言result与expected相等
    tm.assert_equal(result, expected)
# 定义函数用于测试在分组数据上应用转换函数时的行为
def test_null_group_str_transformer(dropna, transformation_func):
    # 创建一个包含两列的DataFrame，包括空值，指定索引
    df = DataFrame({"A": [1, 1, np.nan], "B": [1, 2, 2]}, index=[1, 2, 3])
    # 获取应用于转换函数的参数列表
    args = get_groupby_method_args(transformation_func, df)
    # 根据'A'列进行分组，指定是否丢弃NaN
    gb = df.groupby("A", dropna=dropna)

    buffer = []
    # 迭代每个分组的键和对应的索引和组
    for k, (idx, group) in enumerate(gb):
        if transformation_func == "cumcount":
            # DataFrame没有cumcount方法，创建一个新的DataFrame来存放结果
            res = DataFrame({"B": range(len(group))}, index=group.index)
        elif transformation_func == "ngroup":
            # 如果是ngroup函数，创建一个新的DataFrame，用当前组的索引填充
            res = DataFrame(len(group) * [k], index=group.index, columns=["B"])
        else:
            # 对于其他转换函数，从组中获取'B'列，应用指定的转换函数，并存储结果
            res = getattr(group[["B"]], transformation_func)(*args)
        buffer.append(res)
    # 如果指定丢弃NaN，则添加一个包含NaN值的DataFrame到缓冲区中
    if dropna:
        dtype = object if transformation_func in ("any", "all") else None
        buffer.append(DataFrame([[np.nan]], index=[3], dtype=dtype, columns=["B"]))
    # 将缓冲区中的所有DataFrame连接成一个预期的DataFrame
    expected = concat(buffer)

    if transformation_func in ("cumcount", "ngroup"):
        # 对于ngroup和cumcount函数，预期结果只包含'B'列的Series，重命名为None
        expected = expected["B"].rename(None)

    # 对分组对象应用指定的转换函数，获取实际结果
    result = gb.transform(transformation_func, *args)
    # 使用测试工具包检查实际结果与预期结果是否相等
    tm.assert_equal(result, expected)


# 定义函数用于测试在分组数据上应用缩减函数时的行为
def test_null_group_str_reducer_series(request, dropna, reduction_func):
    # GH 17093
    # 创建一个带有非标准索引的Series
    index = [1, 2, 3, 4]  # test transform preserves non-standard index
    ser = Series([1, 2, 2, 3], index=index)
    # 根据给定的键分组Series，指定是否丢弃NaN
    gb = ser.groupby([1, 1, np.nan, np.nan], dropna=dropna)

    if reduction_func == "corrwith":
        # 对于corrwith函数，SeriesGroupBy对象不支持此操作，直接返回
        assert not hasattr(gb, reduction_func)
        return

    # 获取应用于缩减函数的参数列表
    args = get_groupby_method_args(reduction_func, ser)

    # 手动处理不符合通用模式的缩减函数
    # 针对不同的缩减函数设置预期值，初始设定dropna=False，必要时替换
    if reduction_func == "first":
        expected = Series([1, 1, 2, 2], index=index)
    elif reduction_func == "last":
        expected = Series([2, 2, 3, 3], index=index)
    elif reduction_func == "nth":
        expected = Series([1, 1, 2, 2], index=index)
    elif reduction_func == "size":
        expected = Series([2, 2, 2, 2], index=index)
    elif reduction_func == "corrwith":
        expected = Series([1, 1, 2, 2], index=index)
    else:
        # 对于其他缩减函数，手动分组Series并应用指定的缩减函数，存储结果
        expected_gb = ser.groupby([1, 1, np.nan, np.nan], dropna=False)
        buffer = []
        for idx, group in expected_gb:
            res = getattr(group, reduction_func)()
            buffer.append(Series(res, index=group.index))
        expected = concat(buffer)
    # 如果指定丢弃NaN，则设置预期结果的数据类型，并将特定位置的值替换为NaN
    if dropna:
        dtype = object if reduction_func in ("any", "all") else float
        expected = expected.astype(dtype)
        expected.iloc[[2, 3]] = np.nan

    # 对分组对象应用指定的缩减函数，获取实际结果
    result = gb.transform(reduction_func, *args)
    # 使用测试工具包检查实际结果与预期结果是否相等
    tm.assert_series_equal(result, expected)


# 定义函数用于测试在分组Series上应用转换函数时的行为
def test_null_group_str_transformer_series(dropna, transformation_func):
    # GH 17093
    # 创建一个带有索引的Series
    ser = Series([1, 2, 2], index=[1, 2, 3])
    # 获取应用于转换函数的参数列表
    args = get_groupby_method_args(transformation_func, ser)
    # 根据指定条件对序列进行分组，生成一个序列分组对象
    gb = ser.groupby([1, 1, np.nan], dropna=dropna)

    # 初始化一个空列表，用于存储转换后的结果
    buffer = []

    # 遍历分组对象，k 是组索引，(idx, group) 是组的索引和组本身
    for k, (idx, group) in enumerate(gb):
        if transformation_func == "cumcount":
            # 如果转换函数是 "cumcount"，则创建一个序列，其中值是组内的累计计数，索引是组的索引
            res = Series(range(len(group)), index=group.index)
        elif transformation_func == "ngroup":
            # 如果转换函数是 "ngroup"，则创建一个序列，其值是当前组的编号 k，索引是组的索引
            res = Series(k, index=group.index)
        else:
            # 对于其他转换函数，通过 getattr 动态调用，并传递参数 *args 执行相应的操作
            res = getattr(group, transformation_func)(*args)
        # 将生成的结果序列添加到 buffer 中
        buffer.append(res)

    # 如果 dropna 参数为 True，则处理缺失值的情况
    if dropna:
        # 如果转换函数是 "any" 或 "all"，则 dtype 设为 object，否则为 None
        dtype = object if transformation_func in ("any", "all") else None
        # 向 buffer 中添加一个包含 NaN 值的序列，索引为 [3]，dtype 根据前面的设置确定
        buffer.append(Series([np.nan], index=[3], dtype=dtype))

    # 将 buffer 中的所有序列连接起来，得到预期的结果序列 expected
    expected = concat(buffer)

    # 如果转换函数是 "fillna"，则设置警告类型为 FutureWarning
    warn = FutureWarning if transformation_func == "fillna" else None
    # 设置警告信息
    msg = "SeriesGroupBy.fillna is deprecated"
    # 使用 assert_produces_warning 上下文管理器来确保产生警告，并与指定的消息进行匹配
    with tm.assert_produces_warning(warn, match=msg):
        # 调用 gb 对象的 transform 方法进行数据转换，传递参数 *args
        result = gb.transform(transformation_func, *args)

    # 使用 assert_equal 函数检查 result 是否等于预期的 expected 结果
    tm.assert_equal(result, expected)
@pytest.mark.parametrize(
    "func, expected_values",
    [
        # 参数化测试函数，第一个参数为 Series.sort_values 函数，期望结果为 [5, 4, 3, 2, 1]
        (Series.sort_values, [5, 4, 3, 2, 1]),
        # 参数化测试匿名函数，期望结果为 [5.0, np.nan, 3, 2, np.nan]
        (lambda x: x.head(1), [5.0, np.nan, 3, 2, np.nan]),
    ],
)
@pytest.mark.parametrize("keys", [["a1"], ["a1", "a2"]])
@pytest.mark.parametrize("keys_in_index", [True, False])
def test_transform_aligns(func, frame_or_series, expected_values, keys, keys_in_index):
    # GH#45648 - transform should align with the input's index
    # 创建 DataFrame 包含列 "a1" 和 "b"
    df = DataFrame({"a1": [1, 1, 3, 2, 2], "b": [5, 4, 3, 2, 1]})
    # 如果 keys 包含 "a2"，则创建 "a2" 列并与 "a1" 列相同
    if "a2" in keys:
        df["a2"] = df["a1"]
    # 如果 keys_in_index 为 True，则将 df 设置为以 keys 为索引，追加到现有索引中
    if keys_in_index:
        df = df.set_index(keys, append=True)

    # 根据 keys 进行分组
    gb = df.groupby(keys)
    # 如果 frame_or_series 是 Series，则取 gb 的 "b" 列
    if frame_or_series is Series:
        gb = gb["b"]

    # 对 gb 应用 func 函数
    result = gb.transform(func)
    # 创建期望的 DataFrame，包含 "b" 列和 expected_values，索引与 df 相同
    expected = DataFrame({"b": expected_values}, index=df.index)
    # 如果 frame_or_series 是 Series，则只保留 "b" 列作为期望结果
    if frame_or_series is Series:
        expected = expected["b"]
    # 使用测试框架的断言方法验证 result 是否等于 expected
    tm.assert_equal(result, expected)


@pytest.mark.parametrize("keys", ["A", ["A", "B"]])
def test_as_index_no_change(keys, df, groupby_func):
    # GH#49834 - as_index should have no impact on DataFrameGroupBy.transform
    if keys == "A":
        # 如果 keys 是 "A"，则删除 df 的 "B" 列
        df = df.drop(columns="B")
    # 获取 groupby_func 在 df 上的参数列表
    args = get_groupby_method_args(groupby_func, df)
    # 分别创建 as_index 为 True 和 False 的 DataFrameGroupBy 对象
    gb_as_index_true = df.groupby(keys, as_index=True)
    gb_as_index_false = df.groupby(keys, as_index=False)
    # 如果 groupby_func 是 "corrwith"，则设置警告和消息
    if groupby_func == "corrwith":
        warn = FutureWarning
        msg = "DataFrameGroupBy.corrwith is deprecated"
    else:
        warn = None
        msg = ""
    # 使用测试框架的断言方法验证 gb_as_index_true.transform 的结果与 gb_as_index_false.transform 的结果是否相等，并产生相应的警告
    with tm.assert_produces_warning(warn, match=msg):
        result = gb_as_index_true.transform(groupby_func, *args)
    with tm.assert_produces_warning(warn, match=msg):
        expected = gb_as_index_false.transform(groupby_func, *args)
    tm.assert_equal(result, expected)


@pytest.mark.parametrize("how", ["idxmax", "idxmin"])
@pytest.mark.parametrize("numeric_only", [True, False])
def test_idxmin_idxmax_transform_args(how, skipna, numeric_only):
    # GH#55268 - ensure *args are passed through when calling transform
    # 创建包含列 "a", "b", "c" 的 DataFrame
    df = DataFrame({"a": [1, 1, 1, 2], "b": [3.0, 4.0, np.nan, 6.0], "c": list("abcd")})
    # 根据 "a" 列进行分组
    gb = df.groupby("a")
    # 如果 skipna 为 True，则使用 skipna 和 numeric_only 作为参数调用 gb.transform(how)
    if skipna:
        result = gb.transform(how, skipna, numeric_only)
        expected = gb.transform(how, skipna=skipna, numeric_only=numeric_only)
        # 使用测试框架的断言方法验证 result 是否等于 expected
        tm.assert_frame_equal(result, expected)
    else:
        # 否则，设置异常消息为 "DataFrameGroupBy.how with skipna=False encountered an NA value"
        msg = f"DataFrameGroupBy.{how} with skipna=False encountered an NA value"
        # 使用 pytest 的断言方法验证 gb.transform(how) 调用时是否引发 ValueError 异常，并检查异常消息是否匹配 msg
        with pytest.raises(ValueError, match=msg):
            gb.transform(how, skipna, numeric_only)


def test_transform_sum_one_column_no_matching_labels():
    # 创建包含列 "X" 的 DataFrame
    df = DataFrame({"X": [1.0]})
    # 创建包含 "Y" 的 Series
    series = Series(["Y"])
    # 对 df 按照 series 的值进行分组，并应用 "sum" 函数
    result = df.groupby(series, as_index=False).transform("sum")
    # 创建期望的 DataFrame，包含 "X" 列和 [1.0]，索引与 df 相同
    expected = DataFrame({"X": [1.0]})
    # 使用测试框架的断言方法验证 result 是否等于 expected
    tm.assert_frame_equal(result, expected)


def test_transform_sum_no_matching_labels():
    # 创建包含列 "X" 的 DataFrame
    df = DataFrame({"X": [1.0, -93204, 4935]})
    # 创建包含 "A", "B", "C" 的 Series
    series = Series(["A", "B", "C"])
    # 使用给定的数据框按照指定的 series 列进行分组，并对每组进行求和转换，保持索引
    result = df.groupby(series, as_index=False).transform("sum")
    
    # 创建一个预期的数据框，包含特定的列 "X" 和对应的值
    expected = DataFrame({"X": [1.0, -93204, 4935]})
    
    # 使用测试工具（如 pandas.testing 中的 assert_frame_equal 函数）比较两个数据框的内容是否相等
    tm.assert_frame_equal(result, expected)
# 测试用例：对具有匹配标签的单列进行transform("sum")操作
def test_transform_sum_one_column_with_matching_labels():
    # 创建一个包含列"X"的DataFrame，列数据为[1.0, -93204, 4935]
    df = DataFrame({"X": [1.0, -93204, 4935]})
    # 创建一个包含标签"A", "B", "A"的Series
    series = Series(["A", "B", "A"])

    # 对DataFrame按照Series进行分组，并对每组进行"sum"操作
    result = df.groupby(series, as_index=False).transform("sum")
    # 创建预期结果DataFrame，其中列"X"的值分别为[4936.0, -93204, 4936.0]
    expected = DataFrame({"X": [4936.0, -93204, 4936.0]})
    # 使用测试框架检查result和expected是否相等
    tm.assert_frame_equal(result, expected)


# 测试用例：对具有缺失标签的单列进行transform("sum")操作
def test_transform_sum_one_column_with_missing_labels():
    # 创建一个包含列"X"的DataFrame，列数据为[1.0, -93204, 4935]
    df = DataFrame({"X": [1.0, -93204, 4935]})
    # 创建一个包含标签"A", "C"的Series
    series = Series(["A", "C"])

    # 对DataFrame按照Series进行分组，并对每组进行"sum"操作
    result = df.groupby(series, as_index=False).transform("sum")
    # 创建预期结果DataFrame，其中列"X"的值分别为[1.0, -93204, np.nan]
    expected = DataFrame({"X": [1.0, -93204, np.nan]})
    # 使用测试框架检查result和expected是否相等
    tm.assert_frame_equal(result, expected)


# 测试用例：对同时具有匹配标签和缺失标签的单列进行transform("sum")操作
def test_transform_sum_one_column_with_matching_labels_and_missing_labels():
    # 创建一个包含列"X"的DataFrame，列数据为[1.0, -93204, 4935]
    df = DataFrame({"X": [1.0, -93204, 4935]})
    # 创建一个包含标签"A", "A"的Series
    series = Series(["A", "A"])

    # 对DataFrame按照Series进行分组，并对每组进行"sum"操作
    result = df.groupby(series, as_index=False).transform("sum")
    # 创建预期结果DataFrame，其中列"X"的值分别为[-93203.0, -93203.0, np.nan]
    expected = DataFrame({"X": [-93203.0, -93203.0, np.nan]})
    # 使用测试框架检查result和expected是否相等
    tm.assert_frame_equal(result, expected)


# 测试用例：对具有未观察到的类别但无类型强制转换的情况进行transform("min")操作
@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_min_one_unobserved_category_no_type_coercion(dtype):
    # 创建一个包含列"A"和"B"的DataFrame，其中"A"为分类数据，"B"为整数数据
    df = DataFrame({"A": Categorical([1, 1, 2], categories=[1, 2, 3]), "B": [3, 4, 5]})
    # 将列"B"的数据类型转换为指定的dtype
    df["B"] = df["B"].astype(dtype)
    # 对DataFrame按照列"A"进行分组，不观察未出现的类别
    gb = df.groupby("A", observed=False)
    # 对每组进行"min"操作
    result = gb.transform("min")

    # 创建预期结果DataFrame，其中列"B"的值为[3, 3, 5]，数据类型为dtype
    expected = DataFrame({"B": [3, 3, 5]}, dtype=dtype)
    # 使用测试框架检查result和expected是否相等
    tm.assert_frame_equal(expected, result)


# 测试用例：对所有数据为空的情况进行transform("min")操作，不进行类型强制转换
def test_min_all_empty_data_no_type_coercion():
    # 创建一个空DataFrame，包含列"X"和"Y"，其中"X"为分类数据，但不包含任何实际数据，"Y"为空列表
    df = DataFrame(
        {
            "X": Categorical(
                [],
                categories=[1, "randomcat", 100],
            ),
            "Y": [],
        }
    )
    # 将列"Y"的数据类型转换为"int32"
    df["Y"] = df["Y"].astype("int32")

    # 对DataFrame按照列"X"进行分组，不观察未出现的类别
    gb = df.groupby("X", observed=False)
    # 对每组进行"min"操作
    result = gb.transform("min")

    # 创建预期结果DataFrame，其中列"Y"为空列表，数据类型为"int32"
    expected = DataFrame({"Y": []}, dtype="int32")
    # 使用测试框架检查result和expected是否相等
    tm.assert_frame_equal(expected, result)


# 测试用例：对单维度数据进行transform("min")操作，不进行类型强制转换
def test_min_one_dim_no_type_coercion():
    # 创建一个包含列"Y"的DataFrame，列数据为[9435, -5465765, 5055, 0, 954960]
    df = DataFrame({"Y": [9435, -5465765, 5055, 0, 954960]})
    # 将列"Y"的数据类型转换为"int32"
    df["Y"] = df["Y"].astype("int32")
    # 创建一个包含标签[1, 2, 2, 5, 1]的分类数据
    categories = Categorical([1, 2, 2, 5, 1], categories=[1, 2, 3, 4, 5])

    # 对DataFrame按照分类数据categories进行分组，不观察未出现的类别
    gb = df.groupby(categories, observed=False)
    # 对每组进行"min"操作
    result = gb.transform("min")

    # 创建预期结果DataFrame，其中列"Y"的值为[9435, -5465765, -5465765, 0, 9435]，数据类型为"int32"
    expected = DataFrame({"Y": [9435, -5465765, -5465765, 0, 9435]}, dtype="int32")
    # 使用测试框架检查result和expected是否相等
    tm.assert_frame_equal(expected, result)


# 测试用例：对包含NaN的累积和标签进行操作
def test_nan_in_cumsum_group_label():
    # 创建一个包含列"A"和"B"的DataFrame，其中"A"列包含整数和NaN，"B"列包含整数
    df = DataFrame({"A": [1, None], "B": [2, 3]}, dtype="Int16")
    # 对DataFrame按照"A"列进行分组，获取"B"列，并对每组进行累积和操作
    gb = df.groupby("A")["B"]
    result = gb.cumsum()

    # 创建预期结果Series，其中包含累积和的值，包括NaN
    expected = Series([2, None], dtype="Int16", name="B")
    # 使用测试框架检查result和expected是否相等
    tm.assert_series_equal(expected, result)
```