# `D:\src\scipysrc\pandas\pandas\tests\groupby\methods\test_nth.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行单元测试

import pandas as pd  # 导入 Pandas 库
from pandas import (  # 从 Pandas 中导入以下模块和类
    DataFrame,  # 用于创建和操作二维数据结构
    Index,  # 用于创建索引对象
    MultiIndex,  # 用于创建多级索引对象
    Series,  # 用于创建和操作一维数据结构
    Timestamp,  # 用于处理时间戳
    isna,  # 用于检测缺失值
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块，用于测试辅助工具


def test_first_last_nth(df):
    # tests for first / last / nth

    grouped = df.groupby("A")  # 根据列"A"对 DataFrame 进行分组
    first = grouped.first()  # 获取每组的第一个元素
    expected = df.loc[[1, 0], ["B", "C", "D"]]  # 选择特定行和列的子集作为预期结果
    expected.index = Index(["bar", "foo"], name="A")  # 设置预期结果的索引
    expected = expected.sort_index()  # 按索引排序预期结果
    tm.assert_frame_equal(first, expected)  # 使用测试辅助工具比较结果是否相等

    nth = grouped.nth(0)  # 获取每组的第一个或指定位置的元素
    expected = df.loc[[0, 1]]  # 选择特定行的子集作为预期结果
    tm.assert_frame_equal(nth, expected)  # 使用测试辅助工具比较结果是否相等

    last = grouped.last()  # 获取每组的最后一个元素
    expected = df.loc[[5, 7], ["B", "C", "D"]]  # 选择特定行和列的子集作为预期结果
    expected.index = Index(["bar", "foo"], name="A")  # 设置预期结果的索引
    tm.assert_frame_equal(last, expected)  # 使用测试辅助工具比较结果是否相等

    nth = grouped.nth(-1)  # 获取每组的倒数第一个或指定位置的元素
    expected = df.iloc[[5, 7]]  # 按位置选择特定行的子集作为预期结果
    tm.assert_frame_equal(nth, expected)  # 使用测试辅助工具比较结果是否相等

    nth = grouped.nth(1)  # 获取每组的第二个或指定位置的元素
    expected = df.iloc[[2, 3]]  # 按位置选择特定行的子集作为预期结果
    tm.assert_frame_equal(nth, expected)  # 使用测试辅助工具比较结果是否相等

    # it works!
    grouped["B"].first()  # 获取列"B"每组的第一个元素
    grouped["B"].last()  # 获取列"B"每组的最后一个元素
    grouped["B"].nth(0)  # 获取列"B"每组的第一个或指定位置的元素

    df = df.copy()  # 复制 DataFrame，以便修改副本
    df.loc[df["A"] == "foo", "B"] = np.nan  # 将符合条件的行列"B"设置为 NaN
    grouped = df.groupby("A")  # 根据列"A"重新分组
    assert isna(grouped["B"].first()["foo"])  # 检查是否符合预期的缺失值
    assert isna(grouped["B"].last()["foo"])  # 检查是否符合预期的缺失值
    assert isna(grouped["B"].nth(0).iloc[0])  # 检查是否符合预期的缺失值

    # v0.14.0 whatsnew
    df = DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=["A", "B"])  # 创建新的 DataFrame
    g = df.groupby("A")  # 根据列"A"对 DataFrame 进行分组
    result = g.first()  # 获取每组的第一个元素
    expected = df.iloc[[1, 2]].set_index("A")  # 按位置选择特定行，并设置索引作为预期结果
    tm.assert_frame_equal(result, expected)  # 使用测试辅助工具比较结果是否相等

    expected = df.iloc[[1, 2]]  # 按位置选择特定行作为预期结果
    result = g.nth(0, dropna="any")  # 获取每组的第一个或指定位置的元素，丢弃包含任何 NaN 的行
    tm.assert_frame_equal(result, expected)  # 使用测试辅助工具比较结果是否相等


@pytest.mark.parametrize("method", ["first", "last"])
def test_first_last_with_na_object(method, nulls_fixture):
    # https://github.com/pandas-dev/pandas/issues/32123
    groups = DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, nulls_fixture]}).groupby("a")  # 根据列"a"对 DataFrame 进行分组
    result = getattr(groups, method)()  # 调用指定方法（"first"或"last"）获取每组的第一个或最后一个元素

    if method == "first":
        values = [1, 3]  # 预期的第一个元素值列表
    else:
        values = [2, 3]  # 预期的最后一个元素值列表

    values = np.array(values, dtype=result["b"].dtype)  # 将值列表转换为 NumPy 数组，并使用结果的数据类型
    idx = Index([1, 2], name="a")  # 创建索引对象作为预期结果的索引
    expected = DataFrame({"b": values}, index=idx)  # 创建 DataFrame 作为预期结果

    tm.assert_frame_equal(result, expected)  # 使用测试辅助工具比较结果是否相等


@pytest.mark.parametrize("index", [0, -1])
def test_nth_with_na_object(index, nulls_fixture):
    # https://github.com/pandas-dev/pandas/issues/32123
    df = DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, nulls_fixture]})  # 创建新的 DataFrame
    groups = df.groupby("a")  # 根据列"a"对 DataFrame 进行分组
    result = groups.nth(index)  # 获取每组的第一个或最后一个元素或指定位置的元素

    expected = df.iloc[[0, 2]] if index == 0 else df.iloc[[1, 3]]  # 按位置选择特定行作为预期结果
    tm.assert_frame_equal(result, expected)  # 使用测试辅助工具比较结果是否相等


@pytest.mark.parametrize("method", ["first", "last"])
def test_first_last_with_None(method):
    # https://github.com/pandas-dev/pandas/issues/32800
    # None should be preserved as object dtype
    df = DataFrame.from_dict({"id": ["a"], "value": [None]})  # 从字典创建新的 DataFrame
    groups = df.groupby("id", as_index=False)  # 根据列"id"对 DataFrame 进行分组，且不作为索引
    result = getattr(groups, method)()  # 调用指定方法（"first"或"last"）获取每组的第一个或最后一个元素

    tm.assert_frame_equal(result, df)  # 使用测试辅助工具比较结果是否相等
@pytest.mark.parametrize("method", ["first", "last"])
# 使用 pytest.mark.parametrize 装饰器定义测试参数化，method 可以是 "first" 或 "last"
@pytest.mark.parametrize(
    "df, expected",
    [
        (
            DataFrame({"id": "a", "value": [None, "foo", np.nan]}),
            DataFrame({"value": ["foo"]}, index=Index(["a"], name="id")),
        ),
        (
            DataFrame({"id": "a", "value": [np.nan]}, dtype=object),
            DataFrame({"value": [None]}, index=Index(["a"], name="id")),
        ),
    ],
)
# 继续使用 pytest.mark.parametrize 装饰器，定义 DataFrame 和期望结果的参数化输入
def test_first_last_with_None_expanded(method, df, expected):
    # GH 32800, 38286
    # 根据 GitHub 问题编号解释测试目的
    result = getattr(df.groupby("id"), method)()
    # 获取 df.groupby("id") 的属性 method 执行结果
    tm.assert_frame_equal(result, expected)
    # 使用测试工具 tm.assert_frame_equal 比较结果与期望值的DataFrame是否相等


def test_first_last_nth_dtypes():
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.array(np.random.default_rng(2).standard_normal(8), dtype="float32"),
        }
    )
    df["E"] = True
    df["F"] = 1

    # tests for first / last / nth
    # 针对 first / last / nth 进行测试
    grouped = df.groupby("A")
    first = grouped.first()
    # 获取分组后的第一个数据
    expected = df.loc[[1, 0], ["B", "C", "D", "E", "F"]]
    expected.index = Index(["bar", "foo"], name="A")
    expected = expected.sort_index()
    # 期望的结果DataFrame，排序后与实际结果比较
    tm.assert_frame_equal(first, expected)

    last = grouped.last()
    # 获取分组后的最后一个数据
    expected = df.loc[[5, 7], ["B", "C", "D", "E", "F"]]
    expected.index = Index(["bar", "foo"], name="A")
    expected = expected.sort_index()
    # 期望的结果DataFrame，排序后与实际结果比较
    tm.assert_frame_equal(last, expected)

    nth = grouped.nth(1)
    # 获取分组后的第二个数据
    expected = df.iloc[[2, 3]]
    # 期望的结果DataFrame，与实际结果比较
    tm.assert_frame_equal(nth, expected)


def test_first_last_nth_dtypes2():
    # GH 2763, first/last shifting dtypes
    # 根据 GitHub 问题编号解释测试目的，检测 first/last 操作对数据类型的影响
    idx = list(range(10))
    idx.append(9)
    ser = Series(data=range(11), index=idx, name="IntCol")
    assert ser.dtype == "int64"
    f = ser.groupby(level=0).first()
    # 获取按 level 分组后的第一个数据
    assert f.dtype == "int64"


def test_first_last_nth_nan_dtype():
    # GH 33591
    df = DataFrame({"data": ["A"], "nans": Series([None], dtype=object)})
    grouped = df.groupby("data")

    expected = df.set_index("data").nans
    # 获取分组后的第一个数据，比较 Series 是否相等
    tm.assert_series_equal(grouped.nans.first(), expected)
    # 获取分组后的最后一个数据，比较 Series 是否相等
    tm.assert_series_equal(grouped.nans.last(), expected)

    expected = df.nans
    # 获取分组后的倒数第一个数据，比较 Series 是否相等
    tm.assert_series_equal(grouped.nans.nth(-1), expected)
    # 获取分组后的第一个数据，比较 Series 是否相等
    tm.assert_series_equal(grouped.nans.nth(0), expected)


def test_first_strings_timestamps():
    # GH 11244
    test = DataFrame(
        {
            Timestamp("2012-01-01 00:00:00"): ["a", "b"],
            Timestamp("2012-01-02 00:00:00"): ["c", "d"],
            "name": ["e", "e"],
            "aaaa": ["f", "g"],
        }
    )
    result = test.groupby("name").first()
    # 获取分组后的第一个数据
    expected = DataFrame(
        [["a", "c", "f"]],
        columns=Index([Timestamp("2012-01-01"), Timestamp("2012-01-02"), "aaaa"]),
        index=Index(["e"], name="name"),
    )
    # 期望的结果DataFrame，与实际结果比较
    tm.assert_frame_equal(result, expected)


def test_nth():
    # 该测试函数未提供代码和注释，可能用于其他测试目的
    # 创建一个包含NaN值的DataFrame
    df = DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=["A", "B"])
    # 根据列"A"对DataFrame进行分组
    gb = df.groupby("A")

    # 断言分组后的DataFrame的第一个元素与预期DataFrame的第1和第3行相等
    tm.assert_frame_equal(gb.nth(0), df.iloc[[0, 2]])
    # 断言分组后的DataFrame的第二个元素与预期DataFrame的第2行相等
    tm.assert_frame_equal(gb.nth(1), df.iloc[[1]])
    # 断言分组后的DataFrame的第三个元素与预期DataFrame的空集相等
    tm.assert_frame_equal(gb.nth(2), df.loc[[]])
    # 断言分组后的DataFrame的倒数第一个元素与预期DataFrame的第2和第3行相等
    tm.assert_frame_equal(gb.nth(-1), df.iloc[[1, 2]])
    # 断言分组后的DataFrame的倒数第二个元素与预期DataFrame的第1行相等
    tm.assert_frame_equal(gb.nth(-2), df.iloc[[0]])
    # 断言分组后的DataFrame的倒数第三个元素与预期DataFrame的空集相等
    tm.assert_frame_equal(gb.nth(-3), df.loc[[]])
    
    # 断言分组后的DataFrame的"B"列的第一个元素与预期DataFrame的"B"列的第1和第3行相等
    tm.assert_series_equal(gb.B.nth(0), df.B.iloc[[0, 2]])
    # 断言分组后的DataFrame的"B"列的第二个元素与预期DataFrame的"B"列的第2行相等
    tm.assert_series_equal(gb.B.nth(1), df.B.iloc[[1]])
    # 断言分组后的DataFrame的"B"列的第一个元素与预期DataFrame的"B"列的第1和第3行相等
    tm.assert_frame_equal(gb[["B"]].nth(0), df[["B"]].iloc[[0, 2]])

    # 断言分组后的DataFrame的第一个元素与预期DataFrame的第2和第3行相等，忽略NaN值
    tm.assert_frame_equal(gb.nth(0, dropna="any"), df.iloc[[1, 2]])
    # 断言分组后的DataFrame的倒数第一个元素与预期DataFrame的第2和第3行相等，忽略NaN值
    tm.assert_frame_equal(gb.nth(-1, dropna="any"), df.iloc[[1, 2]])

    # 断言分组后的DataFrame的第七个元素与预期DataFrame的空集相等，忽略NaN值
    tm.assert_frame_equal(gb.nth(7, dropna="any"), df.iloc[:0])
    # 断言分组后的DataFrame的第二个元素与预期DataFrame的空集相等，忽略NaN值
    tm.assert_frame_equal(gb.nth(2, dropna="any"), df.iloc[:0])
def test_nth2():
    """
    # 测试函数 test_nth2

    # out of bounds, regression from 0.13.1
    # GH 6621

    # 创建一个 DataFrame 对象，包含 'color' 和 'food' 列作为索引，'one' 和 'two' 作为数据列
    df = DataFrame(
        {
            "color": {0: "green", 1: "green", 2: "red", 3: "red", 4: "red"},
            "food": {0: "ham", 1: "eggs", 2: "eggs", 3: "ham", 4: "pork"},
            "two": {
                0: 1.5456590000000001,
                1: -0.070345000000000005,
                2: -2.4004539999999999,
                3: 0.46206000000000003,
                4: 0.52350799999999997,
            },
            "one": {
                0: 0.56573799999999996,
                1: -0.9742360000000001,
                2: 1.033801,
                3: -0.78543499999999999,
                4: 0.70422799999999997,
            },
        }
    ).set_index(["color", "food"])

    # 按照第一级索引进行分组，选择每组的第二个元素
    result = df.groupby(level=0, as_index=False).nth(2)
    # 选择 DataFrame 的最后一行作为预期结果
    expected = df.iloc[[-1]]
    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # 按照第一级索引进行分组，选择每组的第三个元素
    result = df.groupby(level=0, as_index=False).nth(3)
    # 选择空 DataFrame 作为预期结果
    expected = df.loc[[]]
    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


def test_nth3():
    """
    # 测试函数 test_nth3

    # GH 7559
    # from the vbench

    # 创建一个包含随机整数的 DataFrame 对象
    df = DataFrame(np.random.default_rng(2).integers(1, 10, (100, 2)), dtype="int64")
    # 选择 DataFrame 的第二列作为 Series 对象
    ser = df[1]
    # 选择 DataFrame 的第一列作为分组依据
    gb = df[0]
    # 计算每个分组的第一个元素
    expected = ser.groupby(gb).first()
    # 使用 apply 函数计算每个分组的第一个元素
    expected2 = ser.groupby(gb).apply(lambda x: x.iloc[0])
    # 断言两个 Series 是否相等，忽略名称检查
    tm.assert_series_equal(expected2, expected, check_names=False)
    # 断言 expected 的名称为 1
    assert expected.name == 1
    # 断言 expected2 的名称为 1
    assert expected2.name == 1

    # 验证第一个分组的第一个元素
    v = ser[gb == 1].iloc[0]
    # 断言 expected 的第一个元素与 v 相等
    assert expected.iloc[0] == v
    # 断言 expected2 的第一个元素与 v 相等
    assert expected2.iloc[0] == v

    # 使用 pytest 检查异常是否抛出
    with pytest.raises(ValueError, match="For a DataFrame"):
        ser.groupby(gb, sort=False).nth(0, dropna=True)


def test_nth4():
    """
    # 测试函数 test_nth4

    # doc example

    # 创建一个包含 NaN 值的 DataFrame 对象
    df = DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=["A", "B"])
    # 按照 'A' 列进行分组
    gb = df.groupby("A")
    # 获取每个分组的第一个非 NaN 值
    result = gb.B.nth(0, dropna="all")
    # 选择 DataFrame 的第 'B' 列中的第二行和第三行作为预期结果
    expected = df.B.iloc[[1, 2]]
    # 断言两个 Series 是否相等
    tm.assert_series_equal(result, expected)


def test_nth5():
    """
    # 测试函数 test_nth5

    # test multiple nth values

    # 创建一个包含 NaN 值的 DataFrame 对象
    df = DataFrame([[1, np.nan], [1, 3], [1, 4], [5, 6], [5, 7]], columns=["A", "B"])
    # 按照 'A' 列进行分组
    gb = df.groupby("A")

    # 断言多个 nth 值的结果是否正确
    tm.assert_frame_equal(gb.nth(0), df.iloc[[0, 3]])
    tm.assert_frame_equal(gb.nth([0]), df.iloc[[0, 3]])
    tm.assert_frame_equal(gb.nth([0, 1]), df.iloc[[0, 1, 3, 4]])
    tm.assert_frame_equal(gb.nth([0, -1]), df.iloc[[0, 2, 3, 4]])
    tm.assert_frame_equal(gb.nth([0, 1, 2]), df.iloc[[0, 1, 2, 3, 4]])
    tm.assert_frame_equal(gb.nth([0, 1, -1]), df.iloc[[0, 1, 2, 3, 4]])
    tm.assert_frame_equal(gb.nth([2]), df.iloc[[2]])
    tm.assert_frame_equal(gb.nth([3, 4]), df.loc[[]])


def test_nth_bdays(unit):
    """
    # 测试函数 test_nth_bdays

    # 创建一个日期范围，以工作日为频率，作为索引
    business_dates = pd.date_range(
        start="4/1/2014", end="6/30/2014", freq="B", unit=unit
    )
    # 创建一个以日期范围为索引的 DataFrame 对象，所有值为 1
    df = DataFrame(1, index=business_dates, columns=["a", "b"])
    # 按照年和月对 DataFrame 进行分组
    key = [df.index.year, df.index.month]
    # 获取每个分组的第一个、第四个和倒数两个工作日
    result = df.groupby(key, as_index=False).nth([0, 3, -2, -1])
    # 创建一个预期的日期列表，将其转换为 pandas 的 datetime 格式，并且按照指定的时间单位重新采样
    expected_dates = pd.to_datetime(
        [
            "2014/4/1",
            "2014/4/4",
            "2014/4/29",
            "2014/4/30",
            "2014/5/1",
            "2014/5/6",
            "2014/5/29",
            "2014/5/30",
            "2014/6/2",
            "2014/6/5",
            "2014/6/27",
            "2014/6/30",
        ]
    ).as_unit(unit)
    
    # 根据预期的日期索引和列名创建一个 DataFrame 对象，所有值初始化为 1
    expected = DataFrame(1, columns=["a", "b"], index=expected_dates)
    
    # 使用 pandas 中的 assert_frame_equal 函数比较结果 DataFrame 和预期 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试多个分组条件下的 nth 方法
def test_nth_multi_grouper(three_group):
    # PR 9090, related to issue 8979
    # test nth on multiple groupers
    # 根据列"A"和"B"进行分组
    grouped = three_group.groupby(["A", "B"])
    # 获取每组的第一个元素
    result = grouped.nth(0)
    # 期望的结果，选择原数据中的第 0、3、4、7 行
    expected = three_group.iloc[[0, 3, 4, 7]]
    # 使用测试工具比较结果和期望是否相等
    tm.assert_frame_equal(result, expected)


# 使用参数化装饰器，定义一个测试函数，用于测试不同数据集和预期结果的情况
@pytest.mark.parametrize(
    "data, expected_first, expected_last",
    [
        (
            {
                "id": ["A"],
                "time": Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                "foo": [1],
            },
            {
                "id": ["A"],
                "time": Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                "foo": [1],
            },
            {
                "id": ["A"],
                "time": Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                "foo": [1],
            },
        ),
        (
            {
                "id": ["A", "B", "A"],
                "time": [
                    Timestamp("2012-01-01 13:00:00", tz="America/New_York"),
                    Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                    Timestamp("2012-03-01 12:00:00", tz="Europe/London"),
                ],
                "foo": [1, 2, 3],
            },
            {
                "id": ["A", "B"],
                "time": [
                    Timestamp("2012-01-01 13:00:00", tz="America/New_York"),
                    Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                ],
                "foo": [1, 2],
            },
            {
                "id": ["A", "B"],
                "time": [
                    Timestamp("2012-03-01 12:00:00", tz="Europe/London"),
                    Timestamp("2012-02-01 14:00:00", tz="US/Central"),
                ],
                "foo": [3, 2],
            },
        ),
    ],
)
def test_first_last_tz(data, expected_first, expected_last):
    # GH15884
    # Test that the timezone is retained when calling first
    # or last on groupby with as_index=False
    # 创建 DataFrame 对象
    df = DataFrame(data)

    # 对'id'列进行分组，保持索引不作为结果的一部分，并获取每组的第一个元素
    result = df.groupby("id", as_index=False).first()
    expected = DataFrame(expected_first)
    cols = ["id", "time", "foo"]
    # 使用测试工具比较结果和期望是否相等
    tm.assert_frame_equal(result[cols], expected[cols])

    # 对'id'列进行分组，保持索引不作为结果的一部分，并获取每组的第一个'time'列元素
    result = df.groupby("id", as_index=False)["time"].first()
    # 使用测试工具比较结果和期望是否相等，只选择'id'和'time'列
    tm.assert_frame_equal(result, expected[["id", "time"]])

    # 对'id'列进行分组，保持索引不作为结果的一部分，并获取每组的最后一个元素
    result = df.groupby("id", as_index=False).last()
    expected = DataFrame(expected_last)
    cols = ["id", "time", "foo"]
    # 使用测试工具比较结果和期望是否相等
    tm.assert_frame_equal(result[cols], expected[cols])

    # 对'id'列进行分组，保持索引不作为结果的一部分，并获取每组的最后一个'time'列元素
    result = df.groupby("id", as_index=False)["time"].last()
    # 使用测试工具比较结果和期望是否相等，只选择'id'和'time'列
    tm.assert_frame_equal(result, expected[["id", "time"]])


# 使用参数化装饰器，定义一个测试函数，测试多列情况下的第一个和最后一个方法
@pytest.mark.parametrize(
    "method, ts, alpha",
    [
        ["first", Timestamp("2013-01-01", tz="US/Eastern"), "a"],
        ["last", Timestamp("2013-01-02", tz="US/Eastern"), "b"],
    ],
)
def test_first_last_tz_multi_column(method, ts, alpha, unit):
    # GH 21603
    # Test that category_string is of category dtype
    category_string = Series(list("abc")).astype("category")
    # 使用 pandas 库生成一个包含日期时间范围的 DatetimeIndex，以美东时区为准，时间单位由变量 `unit` 指定
    dti = pd.date_range("20130101", periods=3, tz="US/Eastern", unit=unit)
    
    # 创建一个 DataFrame 对象，包含以下列：'group' 组，'category_string' 分类字符串，'datetimetz' 带时区的日期时间索引
    df = DataFrame(
        {
            "group": [1, 1, 2],
            "category_string": category_string,  # 从变量 category_string 中获取数据填充
            "datetimetz": dti,  # 使用上面生成的日期时间索引填充
        }
    )
    
    # 对 DataFrame 按照 'group' 列进行分组，然后调用由变量 method 指定的方法
    result = getattr(df.groupby("group"), method)()
    
    # 创建一个预期的 DataFrame，包含以下列：'category_string' 分类变量，'datetimetz' 带时区的日期时间
    expected = DataFrame(
        {
            "category_string": pd.Categorical(
                [alpha, "c"], dtype=category_string.dtype
            ),  # 使用变量 alpha 和固定值 "c" 构建一个分类变量列
            "datetimetz": [ts, Timestamp("2013-01-03", tz="US/Eastern")],  # 使用变量 ts 和特定日期时间创建一个带时区的 Timestamp 列
        },
        index=Index([1, 2], name="group"),  # 设置索引为 [1, 2]，名称为 "group"
    )
    
    # 将 'datetimetz' 列中的日期时间值转换为由变量 unit 指定的时间单位
    expected["datetimetz"] = expected["datetimetz"].dt.as_unit(unit)
    
    # 使用测试模块 tm 中的 assert_frame_equal 函数，比较 result 和 expected 两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
# 使用 pytest 的 parametrize 装饰器来多次运行以下测试，每次传入不同的参数组合
@pytest.mark.parametrize(
    "values",
    [
        pd.array([True, False], dtype="boolean"),  # 创建包含布尔值的 Pandas 数组
        pd.array([1, 2], dtype="Int64"),  # 创建包含整数的 Pandas 数组，使用 Int64 类型
        pd.to_datetime(["2020-01-01", "2020-02-01"]),  # 将日期字符串转换为 Pandas 的日期时间对象
        pd.to_timedelta([1, 2], unit="D"),  # 创建时间增量，单位为天
    ],
)
# 对于给定的每个函数（'first', 'last', 'min', 'max'），运行以下测试
@pytest.mark.parametrize("function", ["first", "last", "min", "max"])
def test_first_last_extension_array_keeps_dtype(values, function):
    # 创建包含 'a' 和指定值的 DataFrame
    df = DataFrame({"a": [1, 2], "b": values})
    # 按照 'a' 列进行分组
    grouped = df.groupby("a")
    # 创建具有指定名称的索引
    idx = Index([1, 2], name="a")
    # 创建预期的 Series，包含给定值并使用指定的索引
    expected_series = Series(values, name="b", index=idx)
    # 创建预期的 DataFrame，包含给定值并使用指定的索引
    expected_frame = DataFrame({"b": values}, index=idx)

    # 调用 grouped['b'] 的特定函数（'first', 'last', 'min', 'max'），并比较结果与预期的 Series
    result_series = getattr(grouped["b"], function)()
    tm.assert_series_equal(result_series, expected_series)

    # 对 grouped 执行聚合操作，并比较结果与预期的 DataFrame
    result_frame = grouped.agg({"b": function})
    tm.assert_frame_equal(result_frame, expected_frame)


def test_nth_multi_index_as_expected():
    # 创建具有多重索引的 DataFrame
    three_group = DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
        }
    )
    # 按照 ['A', 'B'] 列进行分组
    grouped = three_group.groupby(["A", "B"])
    # 调用 nth 函数并比较结果与预期的 DataFrame
    result = grouped.nth(0)
    expected = three_group.iloc[[0, 3, 4, 7]]
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "op, n, expected_rows",
    [
        ("head", -1, [0]),  # 选择第一组
        ("head", 0, []),  # 选择空集
        ("head", 1, [0, 2]),  # 选择前两组
        ("head", 7, [0, 1, 2]),  # 选择所有组
        ("tail", -1, [1]),  # 选择最后一组
        ("tail", 0, []),  # 选择空集
        ("tail", 1, [1, 2]),  # 选择后两组
        ("tail", 7, [0, 1, 2]),  # 选择所有组
    ],
)
@pytest.mark.parametrize("columns", [None, [], ["A"], ["B"], ["A", "B"]])
def test_groupby_head_tail(op, n, expected_rows, columns, as_index):
    # 创建包含指定值的 DataFrame
    df = DataFrame([[1, 2], [1, 4], [5, 6]], columns=["A", "B"])
    # 按照 'A' 列进行分组，使用 as_index 参数
    g = df.groupby("A", as_index=as_index)
    # 创建预期的 DataFrame，包含指定行和列
    expected = df.iloc[expected_rows]
    if columns is not None:
        g = g[columns]
        expected = expected[columns]
    # 调用特定操作（'head' 或 'tail'）并比较结果与预期的 DataFrame
    result = getattr(g, op)(n)
    tm.assert_frame_equal(result, expected)


def test_group_selection_cache():
    # 测试组选择缓存
    # 暂无代码实现
    pass
    # GH 12839 nth, head, and tail should return same result consistently
    # 创建一个 DataFrame 包含三行数据，两列命名为 "A" 和 "B"
    df = DataFrame([[1, 2], [1, 4], [5, 6]], columns=["A", "B"])
    # 期望结果为 DataFrame 的第一行和第三行
    expected = df.iloc[[0, 2]]

    # 按列 "A" 对 DataFrame 进行分组
    g = df.groupby("A")
    # 对每个分组取前两行的结果
    result1 = g.head(n=2)
    # 对每个分组取第一行的结果
    result2 = g.nth(0)
    # 检查 result1 是否与整个 DataFrame 相等
    tm.assert_frame_equal(result1, df)
    # 检查 result2 是否与期望的结果相等
    tm.assert_frame_equal(result2, expected)

    # 重新按列 "A" 对 DataFrame 进行分组
    g = df.groupby("A")
    # 对每个分组取后两行的结果
    result1 = g.tail(n=2)
    # 对每个分组取第一行的结果
    result2 = g.nth(0)
    # 检查 result1 是否与整个 DataFrame 相等
    tm.assert_frame_equal(result1, df)
    # 检查 result2 是否与期望的结果相等
    tm.assert_frame_equal(result2, expected)

    # 重新按列 "A" 对 DataFrame 进行分组
    g = df.groupby("A")
    # 对每个分组取第一行的结果
    result1 = g.nth(0)
    # 对每个分组取前两行的结果
    result2 = g.head(n=2)
    # 检查 result1 是否与期望的结果相等
    tm.assert_frame_equal(result1, expected)
    # 检查 result2 是否与整个 DataFrame 相等
    tm.assert_frame_equal(result2, df)

    # 重新按列 "A" 对 DataFrame 进行分组
    g = df.groupby("A")
    # 对每个分组取第一行的结果
    result1 = g.nth(0)
    # 对每个分组取后两行的结果
    result2 = g.tail(n=2)
    # 检查 result1 是否与期望的结果相等
    tm.assert_frame_equal(result1, expected)
    # 检查 result2 是否与整个 DataFrame 相等
    tm.assert_frame_equal(result2, df)
def test_nth_empty():
    # GH 16064
    # 创建一个空的 DataFrame，具有指定的索引和列名
    df = DataFrame(index=[0], columns=["a", "b", "c"])
    # 对 DataFrame 按列 'a' 进行分组，并取第10个元素
    result = df.groupby("a").nth(10)
    # 创建一个预期结果 DataFrame，选择前0行数据
    expected = df.iloc[:0]
    # 使用测试工具比较两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # 对 DataFrame 按列 'a' 和 'b' 进行分组，并取第10个元素
    result = df.groupby(["a", "b"]).nth(10)
    # 创建一个预期结果 DataFrame，选择前0行数据
    expected = df.iloc[:0]
    # 使用测试工具比较两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


def test_nth_column_order():
    # GH 20760
    # 检查 nth 方法是否保持列顺序不变
    # 创建一个 DataFrame，指定数据和列名
    df = DataFrame(
        [[1, "b", 100], [1, "a", 50], [1, "a", np.nan], [2, "c", 200], [2, "d", 150]],
        columns=["A", "C", "B"],
    )
    # 对 DataFrame 按列 'A' 进行分组，并取第0个元素
    result = df.groupby("A").nth(0)
    # 创建一个预期结果 DataFrame，选择指定行的数据
    expected = df.iloc[[0, 3]]
    # 使用测试工具比较两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # 对 DataFrame 按列 'A' 进行分组，并取倒数第1个元素，如果有 NaN 就丢弃
    result = df.groupby("A").nth(-1, dropna="any")
    # 创建一个预期结果 DataFrame，选择指定行的数据
    expected = df.iloc[[1, 4]]
    # 使用测试工具比较两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dropna", [None, "any", "all"])
def test_nth_nan_in_grouper(dropna):
    # GH 26011
    # 创建一个 DataFrame，包含 NaN 值和其他列
    df = DataFrame(
        {
            "a": [np.nan, "a", np.nan, "b", np.nan],
            "b": [0, 2, 4, 6, 8],
            "c": [1, 3, 5, 7, 9],
        }
    )
    # 对 DataFrame 按列 'a' 进行分组，并取第0个元素
    result = df.groupby("a").nth(0, dropna=dropna)
    # 创建一个预期结果 DataFrame，选择指定行的数据
    expected = df.iloc[[1, 3]]
    # 使用测试工具比较两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dropna", [None, "any", "all"])
def test_nth_nan_in_grouper_series(dropna):
    # GH 26454
    # 创建一个 DataFrame，包含 NaN 值和其他列
    df = DataFrame(
        {
            "a": [np.nan, "a", np.nan, "b", np.nan],
            "b": [0, 2, 4, 6, 8],
        }
    )
    # 对 DataFrame 的 'b' 列按列 'a' 进行分组，并取第0个元素
    result = df.groupby("a")["b"].nth(0, dropna=dropna)
    # 创建一个预期结果 Series，选择指定行的数据
    expected = df["b"].iloc[[1, 3]]
    # 使用测试工具比较两个 Series 是否相等
    tm.assert_series_equal(result, expected)


def test_first_categorical_and_datetime_data_nat():
    # GH 20520
    # 创建一个 DataFrame，包含分类数据和日期时间数据
    df = DataFrame(
        {
            "group": ["first", "first", "second", "third", "third"],
            "time": 5 * [np.datetime64("NaT")],
            "categories": Series(["a", "b", "c", "a", "b"], dtype="category"),
        }
    )
    # 对 DataFrame 按列 'group' 进行分组，并取每组的第一个元素
    result = df.groupby("group").first()
    # 创建一个预期结果 DataFrame，选择指定列的数据，并设置类别类型
    expected = DataFrame(
        {
            "time": 3 * [np.datetime64("NaT")],
            "categories": Series(["a", "c", "a"]).astype(
                pd.CategoricalDtype(["a", "b", "c"])
            ),
        }
    )
    # 设置预期结果的索引名
    expected.index = Index(["first", "second", "third"], name="group")
    # 使用测试工具比较两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


def test_first_multi_key_groupby_categorical():
    # GH 22512
    # 创建一个 DataFrame，包含多个键的分组和分类数据
    df = DataFrame(
        {
            "A": [1, 1, 1, 2, 2],
            "B": [100, 100, 200, 100, 100],
            "C": ["apple", "orange", "mango", "mango", "orange"],
            "D": ["jupiter", "mercury", "mars", "venus", "venus"],
        }
    )
    # 将 'D' 列的数据类型转换为分类数据
    df = df.astype({"D": "category"})
    # 对 DataFrame 按列 'A' 和 'B' 进行分组，并取每组的第一个元素
    result = df.groupby(by=["A", "B"]).first()
    # 创建一个预期结果 DataFrame，选择指定列的数据，并设置类别类型
    expected = DataFrame(
        {
            "C": ["apple", "mango", "mango"],
            "D": Series(["jupiter", "mars", "venus"]).astype(
                pd.CategoricalDtype(["jupiter", "mars", "mercury", "venus"])
            ),
        }
    )
    # 创建一个 MultiIndex 对象，使用元组列表作为输入，其中元组包含两个元素分别表示各级索引的值
    expected.index = MultiIndex.from_tuples(
        [(1, 100), (1, 200), (2, 100)], names=["A", "B"]
    )
    # 使用测试工具 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("method", ["first", "last", "nth"])
def test_groupby_last_first_nth_with_none(method, nulls_fixture):
    # GH29645
    # 定义预期的结果为包含单个元素 "y" 的 Series
    expected = Series(["y"])
    # 创建一个 Series 对象，并按照索引级别进行分组
    data = Series(
        [nulls_fixture, nulls_fixture, nulls_fixture, "y", nulls_fixture],
        index=[0, 0, 0, 0, 0],
    ).groupby(level=0)

    # 根据不同的方法名调用对应的方法
    if method == "nth":
        result = getattr(data, method)(3)
    else:
        result = getattr(data, method)()

    # 断言结果与预期相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "arg, expected_rows",
    [
        [slice(None, 3, 2), [0, 1, 4, 5]],
        [slice(None, -2), [0, 2, 5]],
        [[slice(None, 2), slice(-2, None)], [0, 1, 2, 3, 4, 6, 7]],
        [[0, 1, slice(-2, None)], [0, 1, 2, 3, 4, 6, 7]],
    ],
)
def test_slice(slice_test_df, slice_test_grouped, arg, expected_rows):
    # Test slices     GH #42947

    # 使用 .nth[] 进行切片操作
    result = slice_test_grouped.nth[arg]
    # 使用 .nth() 方法进行切片操作，等价于 .nth[]
    equivalent = slice_test_grouped.nth(arg)
    # 期望结果是特定行索引对应的 DataFrame 切片
    expected = slice_test_df.iloc[expected_rows]

    # 断言两个结果是否相等
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(equivalent, expected)


def test_nth_indexed(slice_test_df, slice_test_grouped):
    # Test index notation     GH #44688

    # 使用索引表示法进行 .nth[] 操作
    result = slice_test_grouped.nth[0, 1, -2:]
    # 使用 .nth() 方法，传入列表和切片作为参数
    equivalent = slice_test_grouped.nth([0, 1, slice(-2, None)])
    # 期望结果是特定行索引对应的 DataFrame 切片
    expected = slice_test_df.iloc[[0, 1, 2, 3, 4, 6, 7]]

    # 断言两个结果是否相等
    tm.assert_frame_equal(result, expected)
    tm.assert_frame_equal(equivalent, expected)


def test_invalid_argument(slice_test_grouped):
    # Test for error on invalid argument

    # 测试传入无效参数时是否抛出 TypeError 异常
    with pytest.raises(TypeError, match="Invalid index"):
        slice_test_grouped.nth(3.14)


def test_negative_step(slice_test_grouped):
    # Test for error on negative slice step

    # 测试传入负数步长时是否抛出 ValueError 异常
    with pytest.raises(ValueError, match="Invalid step"):
        slice_test_grouped.nth(slice(None, None, -1))


def test_np_ints(slice_test_df, slice_test_grouped):
    # Test np ints work

    # 测试传入 NumPy 整数数组是否正常工作
    result = slice_test_grouped.nth(np.array([0, 1]))
    # 期望结果是特定行索引对应的 DataFrame 切片
    expected = slice_test_df.iloc[[0, 1, 2, 3, 4]]
    tm.assert_frame_equal(result, expected)


def test_groupby_nth_interval():
    # GH#24205
    # 创建一个复合索引的结果 DataFrame
    idx_result = MultiIndex(
        [
            pd.CategoricalIndex([pd.Interval(0, 1), pd.Interval(1, 2)]),
            pd.CategoricalIndex([pd.Interval(0, 10), pd.Interval(10, 20)]),
        ],
        [[0, 0, 0, 1, 1], [0, 1, 1, 0, -1]],
    )
    # 创建一个与 idx_result 对应的 DataFrame
    df_result = DataFrame({"col": range(len(idx_result))}, index=idx_result)
    # 对结果 DataFrame 按照多级索引进行分组，并选择每组的第一个元素
    result = df_result.groupby(level=[0, 1], observed=False).nth(0)
    # 期望的结果值
    val_expected = [0, 1, 3]
    # 期望的复合索引
    idx_expected = MultiIndex(
        [
            pd.CategoricalIndex([pd.Interval(0, 1), pd.Interval(1, 2)]),
            pd.CategoricalIndex([pd.Interval(0, 10), pd.Interval(10, 20)]),
        ],
        [[0, 0, 1], [0, 1, 0]],
    )
    # 创建与期望结果对应的 DataFrame
    expected = DataFrame(val_expected, index=idx_expected, columns=["col"])
    # 断言两个结果是否相等
    tm.assert_frame_equal(result, expected)
    "ignore:invalid value encountered in remainder:RuntimeWarning"
# 定义测试函数 `test_head_tail_dropna_true`，用于测试 DataFrame 对象的分组操作
def test_head_tail_dropna_true():
    # 创建一个 DataFrame 对象 `df`，包含四行数据，两列命名为 "X" 和 "Y"
    df = DataFrame(
        [["a", "z"], ["b", np.nan], ["c", np.nan], ["c", np.nan]], columns=["X", "Y"]
    )
    # 期望的结果 DataFrame 包含一行数据，与 `df` 结构相同
    expected = DataFrame([["a", "z"]], columns=["X", "Y"])

    # 对 `df` 根据列 "X" 和 "Y" 进行分组，取每组的第一行，生成结果 DataFrame `result`
    result = df.groupby(["X", "Y"]).head(n=1)
    # 使用测试工具 `tm` 来断言 `result` 与 `expected` 结果一致
    tm.assert_frame_equal(result, expected)

    # 对 `df` 根据列 "X" 和 "Y" 进行分组，取每组的最后一行，生成结果 DataFrame `result`
    result = df.groupby(["X", "Y"]).tail(n=1)
    # 使用测试工具 `tm` 来断言 `result` 与 `expected` 结果一致
    tm.assert_frame_equal(result, expected)

    # 对 `df` 根据列 "X" 和 "Y" 进行分组，取每组的第一个元素，生成结果 DataFrame `result`
    result = df.groupby(["X", "Y"]).nth(n=0)
    # 使用测试工具 `tm` 来断言 `result` 与 `expected` 结果一致
    tm.assert_frame_equal(result, expected)


# 定义测试函数 `test_head_tail_dropna_false`，用于测试 DataFrame 对象的分组操作
def test_head_tail_dropna_false():
    # 创建一个 DataFrame 对象 `df`，包含三行数据，两列命名为 "X" 和 "Y"
    df = DataFrame([["a", "z"], ["b", np.nan], ["c", np.nan]], columns=["X", "Y"])
    # 期望的结果 DataFrame `expected` 与 `df` 结构相同
    expected = DataFrame([["a", "z"], ["b", np.nan], ["c", np.nan]], columns=["X", "Y"])

    # 对 `df` 根据列 "X" 和 "Y" 进行分组，取每组的第一行，生成结果 DataFrame `result`
    result = df.groupby(["X", "Y"], dropna=False).head(n=1)
    # 使用测试工具 `tm` 来断言 `result` 与 `expected` 结果一致
    tm.assert_frame_equal(result, expected)

    # 对 `df` 根据列 "X" 和 "Y" 进行分组，取每组的最后一行，生成结果 DataFrame `result`
    result = df.groupby(["X", "Y"], dropna=False).tail(n=1)
    # 使用测试工具 `tm` 来断言 `result` 与 `expected` 结果一致
    tm.assert_frame_equal(result, expected)

    # 对 `df` 根据列 "X" 和 "Y" 进行分组，取每组的第一个元素，生成结果 DataFrame `result`
    result = df.groupby(["X", "Y"], dropna=False).nth(n=0)
    # 使用测试工具 `tm` 来断言 `result` 与 `expected` 结果一致
    tm.assert_frame_equal(result, expected)


# 使用参数化测试，测试 `nth` 方法在不同选择和缺失值处理方式下的行为
@pytest.mark.parametrize("selection", ("b", ["b"], ["b", "c"]))
@pytest.mark.parametrize("dropna", ["any", "all", None])
def test_nth_after_selection(selection, dropna):
    # 创建一个 DataFrame `df`，包含三列 "a", "b", "c" 和三行数据
    df = DataFrame(
        {
            "a": [1, 1, 2],
            "b": [np.nan, 3, 4],
            "c": [5, 6, 7],
        }
    )
    # 根据列 "a" 分组并选择指定的列 `selection`，生成结果 `gb`
    gb = df.groupby("a")[selection]
    # 使用 `nth` 方法获取每组的第一个元素，指定缺失值处理方式 `dropna`
    result = gb.nth(0, dropna=dropna)
    # 根据 `dropna` 参数不同，计算期望的结果 `expected`
    if dropna == "any" or (dropna == "all" and selection != ["b", "c"]):
        locs = [1, 2]
    else:
        locs = [0, 2]
    expected = df.loc[locs, selection]
    # 使用测试工具 `tm` 来断言 `result` 与 `expected` 结果一致
    tm.assert_equal(result, expected)


# 使用参数化测试，测试 `nth` 方法在不同数据类型下的精度问题
@pytest.mark.parametrize(
    "data",
    [
        (
            Timestamp("2011-01-15 12:50:28.502376"),
            Timestamp("2011-01-20 12:50:28.593448"),
        ),
        (24650000000000001, 24650000000000002),
    ],
)
def test_groupby_nth_int_like_precision(data):
    # 创建一个 DataFrame `df`，包含两列 "a" 和 "b"，数据根据参数 `data` 提供
    df = DataFrame({"a": [1, 1], "b": data})

    # 根据列 "a" 分组，生成分组对象 `grouped`
    grouped = df.groupby("a")
    # 使用 `nth` 方法获取每组的第一个元素，生成结果 `result`
    result = grouped.nth(0)
    # 期望的结果 DataFrame `expected` 包含第一组每列的第一个元素
    expected = DataFrame({"a": 1, "b": [data[0]]})

    # 使用测试工具 `tm` 来断言 `result` 与 `expected` 结果一致
    tm.assert_frame_equal(result, expected)
```