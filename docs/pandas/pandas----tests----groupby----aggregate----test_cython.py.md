# `D:\src\scipysrc\pandas\pandas\tests\groupby\aggregate\test_cython.py`

```
"""
test cython .agg behavior
"""

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas.core.dtypes.common import (  # 从 pandas 库中导入数据类型检查函数
    is_float_dtype,
    is_integer_dtype,
)

import pandas as pd  # 导入 pandas 库，并使用 pd 别名
from pandas import (  # 从 pandas 中导入多个类和函数
    DataFrame,
    Index,
    NaT,
    Series,
    Timedelta,
    Timestamp,
    bdate_range,
)
import pandas._testing as tm  # 导入 pandas 内部测试工具


@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器定义测试参数
    "op_name",
    [
        "count",  # 统计计数
        "sum",  # 求和
        "std",  # 求标准差
        "var",  # 求方差
        "sem",  # 求标准误差
        "mean",  # 求平均值
        pytest.param(
            "median",
            # 忽略空切片和全部为 NaN 的均值警告
            marks=[pytest.mark.filterwarnings("ignore::RuntimeWarning")],
        ),
        "prod",  # 求积
        "min",  # 求最小值
        "max",  # 求最大值
    ],
)
def test_cythonized_aggers(op_name):  # 定义测试函数，参数为操作名称
    data = {  # 定义测试数据字典
        "A": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1.0, np.nan, np.nan],  # 列 A 的数据
        "B": ["A", "B"] * 6,  # 列 B 的数据
        "C": np.random.default_rng(2).standard_normal(12),  # 列 C 的随机标准正态分布数据
    }
    df = DataFrame(data)  # 创建 DataFrame 对象
    df.loc[2:10:2, "C"] = np.nan  # 将部分行的 C 列设为 NaN

    op = lambda x: getattr(x, op_name)()  # 定义操作函数，通过 op_name 调用 DataFrame 方法

    # 单列操作
    grouped = df.drop(["B"], axis=1).groupby("A")  # 对去除 B 列后的 DataFrame 按 A 列分组
    exp = {cat: op(group["C"]) for cat, group in grouped}  # 生成预期结果字典
    exp = DataFrame({"C": exp})  # 转换为 DataFrame
    exp.index.name = "A"  # 设置索引名称
    result = op(grouped)  # 执行操作
    tm.assert_frame_equal(result, exp)  # 断言结果与预期一致

    # 多列操作
    grouped = df.groupby(["A", "B"])  # 对 DataFrame 按 A 和 B 列分组
    expd = {}  # 初始化预期结果字典
    for (cat1, cat2), group in grouped:  # 遍历分组结果
        expd.setdefault(cat1, {})[cat2] = op(group["C"])  # 设置预期结果
    exp = DataFrame(expd).T.stack()  # 转换为 DataFrame，并堆叠行列
    exp.index.names = ["A", "B"]  # 设置索引名称
    exp.name = "C"  # 设置列名称为 C

    result = op(grouped)["C"]  # 执行操作并获取 C 列结果
    if op_name in ["sum", "prod"]:  # 如果操作是求和或求积
        tm.assert_series_equal(result, exp)  # 断言 Series 结果与预期一致


def test_cython_agg_boolean():  # 定义测试布尔值聚合函数
    frame = DataFrame(  # 创建 DataFrame 对象
        {
            "a": np.random.default_rng(2).integers(0, 5, 50),  # 列 a 的随机整数数据
            "b": np.random.default_rng(2).integers(0, 2, 50).astype("bool"),  # 列 b 的随机布尔值数据
        }
    )
    result = frame.groupby("a")["b"].mean()  # 按 a 列分组并计算 b 列的均值
    # GH#53425
    expected = frame.groupby("a")["b"].agg(np.mean)  # 使用 np.mean 计算预期值

    tm.assert_series_equal(result, expected)  # 断言 Series 结果与预期一致


def test_cython_agg_nothing_to_agg():  # 定义测试没有可聚合内容的函数
    frame = DataFrame(  # 创建 DataFrame 对象
        {"a": np.random.default_rng(2).integers(0, 5, 50), "b": ["foo", "bar"] * 25}  # 包含整数和字符串列
    )

    msg = "Cannot use numeric_only=True with SeriesGroupBy.mean and non-numeric dtypes"
    with pytest.raises(TypeError, match=msg):  # 断言抛出特定异常和消息
        frame.groupby("a")["b"].mean(numeric_only=True)  # 尝试使用 numeric_only=True 聚合非数值列

    frame = DataFrame(  # 重新定义 DataFrame 对象
        {"a": np.random.default_rng(2).integers(0, 5, 50), "b": ["foo", "bar"] * 25}  # 包含整数和字符串列
    )

    result = frame[["b"]].groupby(frame["a"]).mean(numeric_only=True)  # 按 a 列分组并计算 b 列的均值（仅数值列）
    expected = DataFrame(  # 创建预期结果的空 DataFrame
        [], index=frame["a"].sort_values().drop_duplicates(), columns=[]
    )
    tm.assert_frame_equal(result, expected)  # 断言 DataFrame 结果与预期一致


def test_cython_agg_nothing_to_agg_with_dates():  # 定义测试带日期的没有可聚合内容的函数
    frame = DataFrame(  # 创建 DataFrame 对象
        {
            "a": np.random.default_rng(2).integers(0, 5, 50),  # 列 a 的随机整数数据
            "b": ["foo", "bar"] * 25,  # 列 b 的字符串数据
            "dates": pd.date_range("now", periods=50, freq="min"),  # 生成日期范围
        }
    )
    # 错误消息文本，指示不能在 SeriesGroupBy.mean 中使用 numeric_only=True，因为数据类型不是数字
    msg = "Cannot use numeric_only=True with SeriesGroupBy.mean and non-numeric dtypes"
    # 使用 pytest 的上下文管理器检查是否引发了 TypeError，并匹配错误消息以验证异常被正确引发
    with pytest.raises(TypeError, match=msg):
        # 对数据框 frame 按列 'b' 进行分组，并计算分组后列 'dates' 的平均值，要求列为数值类型
        frame.groupby("b").dates.mean(numeric_only=True)
# 定义测试函数，用于验证特定聚合操作在DataFrame上的行为
def test_cython_agg_return_dict():
    # 创建一个DataFrame对象，包含四列数据：A列包含字符串，B列包含字符串，C列和D列包含随机生成的标准正态分布数据
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.random.default_rng(2).standard_normal(8),
        }
    )

    # 对DataFrame按A列进行分组，然后对B列应用lambda函数，计算每个分组中各元素出现次数，并转换为字典
    ts = df.groupby("A")["B"].agg(lambda x: x.value_counts().to_dict())
    
    # 创建预期结果Series对象，包含两个字典作为元素，对应于A列中的"bar"和"foo"
    expected = Series(
        [{"two": 1, "one": 1, "three": 1}, {"two": 2, "one": 2, "three": 1}],
        index=Index(["bar", "foo"], name="A"),
        name="B",
    )
    
    # 使用测试工具方法验证ts与预期结果expected是否相等
    tm.assert_series_equal(ts, expected)


# 定义测试函数，验证在特定条件下聚合操作的行为是否失败
def test_cython_fail_agg():
    # 创建一个时间序列对象，包含50个工作日的日期范围
    dr = bdate_range("1/1/2000", periods=50)
    # 创建一个Series对象，包含50个元素，重复"A"至"E"五次，并指定日期范围为索引
    ts = Series(["A", "B", "C", "D", "E"] * 10, index=dr)

    # 对Series对象按月份进行分组
    grouped = ts.groupby(lambda x: x.month)
    # 对分组后的数据求和
    summed = grouped.sum()
    # 使用np.sum函数对分组后的数据进行聚合
    expected = grouped.agg(np.sum)
    
    # 使用测试工具方法验证summed与expected是否相等
    tm.assert_series_equal(summed, expected)


# 定义测试函数，验证在不同操作下的通用聚合行为
@pytest.mark.parametrize(
    "op, targop",
    [
        ("mean", np.mean),
        ("median", np.median),
        ("var", np.var),
        ("sum", np.sum),
        ("prod", np.prod),
        ("min", np.min),
        ("max", np.max),
        ("first", lambda x: x.iloc[0]),
        ("last", lambda x: x.iloc[-1]),
    ],
)
def test__cython_agg_general(op, targop):
    # 创建一个包含1000个随机标准正态分布数据的DataFrame对象
    df = DataFrame(np.random.default_rng(2).standard_normal(1000))
    # 创建一个包含1000个随机整数的标签数组，作为分组依据
    labels = np.random.default_rng(2).integers(0, 50, size=1000).astype(float)
    kwargs = {"ddof": 1} if op == "var" else {}
    if op not in ["first", "last"]:
        kwargs["axis"] = 0

    # 调用自定义的_cython_agg_general方法，对DataFrame对象按标签数组进行分组并应用指定的聚合操作
    result = df.groupby(labels)._cython_agg_general(op, alt=None, numeric_only=True)
    # 使用numpy库中的聚合函数对DataFrame对象按标签数组进行分组并应用指定的聚合操作
    expected = df.groupby(labels).agg(targop, **kwargs)
    
    # 使用测试工具方法验证result与expected是否相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数，验证在空分组情况下聚合操作的行为
@pytest.mark.parametrize(
    "op, targop",
    [
        ("mean", np.mean),
        ("median", lambda x: np.median(x) if len(x) > 0 else np.nan),
        ("var", lambda x: np.var(x, ddof=1)),
        ("min", np.min),
        ("max", np.max),
    ],
)
def test_cython_agg_empty_buckets(op, targop, observed):
    # 创建一个包含三个整数的DataFrame对象
    df = DataFrame([11, 12, 13])
    # 创建一个包含0到50的范围内，步长为5的整数数组作为分组依据
    grps = range(0, 55, 5)

    # 调用自定义的_cython_agg_general方法，对DataFrame对象按指定区间划分进行分组并应用指定的聚合操作
    g = df.groupby(pd.cut(df[0], grps), observed=observed)
    result = g._cython_agg_general(op, alt=None, numeric_only=True)

    # 对DataFrame对象按指定区间划分进行分组，然后使用lambda函数应用指定的聚合操作
    g = df.groupby(pd.cut(df[0], grps), observed=observed)
    expected = g.agg(lambda x: targop(x))
    
    # 使用测试工具方法验证result与expected是否相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数，验证在空分组情况下应用NaN操作的聚合行为
def test_cython_agg_empty_buckets_nanops(observed):
    # 创建一个包含三个整数的DataFrame对象，指定列名为"a"
    df = DataFrame([11, 12, 13], columns=["a"])
    # 创建一个0到20的区间范围，步长为5的区间对象
    grps = np.arange(0, 25, 5, dtype=int)
    # 调用自定义的_cython_agg_general方法，对DataFrame对象按区间范围划分进行分组并应用指定的聚合操作
    result = df.groupby(pd.cut(df["a"], grps), observed=observed)._cython_agg_general(
        "sum", alt=None, numeric_only=True
    )
    # 创建一个包含等宽区间范围的时间索引对象
    intervals = pd.interval_range(0, 20, freq=5)
    # 创建预期的 DataFrame，包括列"a"和对应的数值列表，使用指定的区间作为索引
    expected = DataFrame(
        {"a": [0, 0, 36, 0]},
        index=pd.CategoricalIndex(intervals, name="a", ordered=True),
    )
    
    # 如果观测到的值为真，则过滤预期 DataFrame，保留"a"列不等于0的行
    if observed:
        expected = expected[expected.a != 0]

    # 使用测试工具比较结果 DataFrame 和预期 DataFrame，确保它们相等
    tm.assert_frame_equal(result, expected)

    # 计算产品（prod）
    result = df.groupby(pd.cut(df["a"], grps), observed=observed)._cython_agg_general(
        "prod", alt=None, numeric_only=True
    )
    
    # 创建预期的 DataFrame，包括列"a"和对应的数值列表，使用指定的区间作为索引
    expected = DataFrame(
        {"a": [1, 1, 1716, 1]},
        index=pd.CategoricalIndex(intervals, name="a", ordered=True),
    )
    
    # 如果观测到的值为真，则过滤预期 DataFrame，保留"a"列不等于1的行
    if observed:
        expected = expected[expected.a != 1]

    # 使用测试工具比较结果 DataFrame 和预期 DataFrame，确保它们相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("op", ["first", "last", "max", "min"])
@pytest.mark.parametrize(
    "data", [Timestamp("2016-10-14 21:00:44.557"), Timedelta("17088 days 21:00:44.557")]
)
def test_cython_with_timestamp_and_nat(op, data):
    # 创建一个包含两列的 DataFrame，其中一列使用给定的时间戳或 NaT（Not a Time）值
    df = DataFrame({"a": [0, 1], "b": [data, NaT]})
    # 创建一个名为 'a' 的索引
    index = Index([0, 1], name="a")

    # 将预期的 DataFrame 定义为与 df 结构相同，但数据与 'b' 列的数据相同
    expected = DataFrame({"b": [data, NaT]}, index=index)

    # 对 df 按 'a' 列进行分组，并使用给定的聚合操作（op）
    result = df.groupby("a").aggregate(op)
    # 断言结果 DataFrame 与预期 DataFrame 相等
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize(
    "agg",
    [
        "min",
        "max",
        "count",
        "sum",
        "prod",
        "var",
        "mean",
        "median",
        "ohlc",
        "cumprod",
        "cumsum",
        "shift",
        "any",
        "all",
        "quantile",
        "first",
        "last",
        "rank",
        "cummin",
        "cummax",
    ],
)
def test_read_only_buffer_source_agg(agg):
    # 创建一个包含 'sepal_length' 和 'species' 列的 DataFrame
    df = DataFrame(
        {
            "sepal_length": [5.1, 4.9, 4.7, 4.6, 5.0],
            "species": ["setosa", "setosa", "setosa", "setosa", "setosa"],
        }
    )
    # 设置 'sepal_length' 列的数据块为不可写入状态
    df._mgr.blocks[0].values.flags.writeable = False

    # 对 df 按 'species' 列进行分组，并使用给定的聚合操作（agg）
    result = df.groupby(["species"]).agg({"sepal_length": agg})
    # 创建一个预期的 DataFrame，与结果 DataFrame 结构相同
    expected = df.copy().groupby(["species"]).agg({"sepal_length": agg})

    # 断言结果与预期 DataFrame 相等
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "op_name",
    [
        "count",
        "sum",
        "std",
        "var",
        "sem",
        "mean",
        "median",
        "prod",
        "min",
        "max",
    ],
)
def test_cython_agg_nullable_int(op_name):
    # 创建一个包含 'A' 和 'B' 列的 DataFrame，'B' 列包含可空整数类型数据
    df = DataFrame(
        {
            "A": ["A", "B"] * 5,
            "B": pd.array([1, 2, 3, 4, 5, 6, 7, 8, 9, pd.NA], dtype="Int64"),
        }
    )
    # 对 'B' 列执行给定的聚合操作（op_name）
    result = getattr(df.groupby("A")["B"], op_name)()
    # 将 'B' 列数据转换为浮点数类型，创建一个新的 DataFrame df2
    df2 = df.assign(B=df["B"].astype("float64"))
    # 对 df2 的 'B' 列执行给定的聚合操作（op_name）
    expected = getattr(df2.groupby("A")["B"], op_name)()
    # 如果操作名为 'mean' 或 'median'，则不转换整数
    if op_name in ("mean", "median"):
        convert_integer = False
    else:
        convert_integer = True
    # 根据 convert_integer 参数将预期结果的数据类型转换为合适的类型
    expected = expected.convert_dtypes(convert_integer=convert_integer)
    # 断言结果 Series 与预期结果相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_count_masked_returns_masked_dtype(dtype):
    # 创建一个包含 'A', 'B', 'C' 列的 DataFrame，'B', 'C' 列包含指定数据类型的数据
    df = DataFrame(
        {
            "A": [1, 1],
            "B": pd.array([1, pd.NA], dtype=dtype),
            "C": pd.array([1, 1], dtype=dtype),
        }
    )
    # 对 'A' 列进行分组，并计算分组后每列的计数
    result = df.groupby("A").count()
    # 创建一个预期的 DataFrame，包含对分组后每列的计数结果
    expected = DataFrame(
        [[1, 2]], index=Index([1], name="A"), columns=["B", "C"], dtype="Int64"
    )
    # 断言结果 DataFrame 与预期 DataFrame 相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("with_na", [True, False])
@pytest.mark.parametrize(
    "op_name, action",
    [
        # 统计项：总数（整数类型）
        # ("count", "always_int"),
    
        # 汇总项：总和（大整数类型）
        ("sum", "large_int"),
    
        # 统计项：标准差（始终为浮点数）
        # ("std", "always_float"),
    
        # 统计项：方差（始终为浮点数）
        ("var", "always_float"),
    
        # 统计项：标准误差（始终为浮点数）
        # ("sem", "always_float"),
    
        # 统计项：均值（始终为浮点数）
        ("mean", "always_float"),
    
        # 统计项：中位数（始终为浮点数）
        ("median", "always_float"),
    
        # 汇总项：乘积（大整数类型）
        ("prod", "large_int"),
    
        # 保留项：最小值
        ("min", "preserve"),
    
        # 保留项：最大值
        ("max", "preserve"),
    
        # 保留项：第一个值
        ("first", "preserve"),
    
        # 保留项：最后一个值
        ("last", "preserve"),
    ],
@pytest.mark.parametrize(
    "data",
    [
        pd.array([1, 2, 3, 4], dtype="Int64"),  # 创建一个包含整数数组的参数化测试数据，数据类型为Int64
        pd.array([1, 2, 3, 4], dtype="Int8"),   # 创建一个包含整数数组的参数化测试数据，数据类型为Int8
        pd.array([0.1, 0.2, 0.3, 0.4], dtype="Float32"),  # 创建一个包含浮点数数组的参数化测试数据，数据类型为Float32
        pd.array([0.1, 0.2, 0.3, 0.4], dtype="Float64"),  # 创建一个包含浮点数数组的参数化测试数据，数据类型为Float64
        pd.array([True, True, False, False], dtype="boolean"),  # 创建一个包含布尔值数组的参数化测试数据，数据类型为boolean
    ],
)
def test_cython_agg_EA_known_dtypes(data, op_name, action, with_na):
    if with_na:
        data[3] = pd.NA  # 如果 with_na 为真，则将数组中的第四个元素设置为缺失值

    df = DataFrame({"key": ["a", "a", "b", "b"], "col": data})  # 创建一个包含数据列的DataFrame，列名为'key'和'col'
    grouped = df.groupby("key")  # 根据'key'列进行分组，创建分组对象

    if action == "always_int":
        # 如果 action 为'always_int'，预期结果的数据类型为Int64
        expected_dtype = pd.Int64Dtype()
    elif action == "large_int":
        # 如果 action 为'large_int'，根据数据类型决定预期的结果数据类型
        if is_float_dtype(data.dtype):
            expected_dtype = data.dtype  # 如果数据类型是浮点数，则预期结果的数据类型保持不变
        elif is_integer_dtype(data.dtype):
            # 如果数据类型是整数，则预期结果的数据类型为对应的非可空类型
            expected_dtype = data.dtype
        else:
            expected_dtype = pd.Int64Dtype()  # 其它类型，默认为Int64
    elif action == "always_float":
        # 如果 action 为'always_float'，根据数据类型决定预期的结果数据类型
        if is_float_dtype(data.dtype):
            expected_dtype = data.dtype  # 如果数据类型是浮点数，则预期结果的数据类型保持不变
        else:
            expected_dtype = pd.Float64Dtype()  # 其它类型，默认为Float64
    elif action == "preserve":
        expected_dtype = data.dtype  # 如果 action 为'preserve'，预期结果的数据类型与数据的原始类型相同

    result = getattr(grouped, op_name)()  # 调用分组对象的指定操作，返回结果
    assert result["col"].dtype == expected_dtype  # 断言操作后结果的'col'列数据类型符合预期

    result = grouped.aggregate(op_name)  # 对分组对象进行聚合操作，返回结果
    assert result["col"].dtype == expected_dtype  # 断言聚合操作后结果的'col'列数据类型符合预期

    result = getattr(grouped["col"], op_name)()  # 对分组对象的'col'列进行指定操作，返回结果
    assert result.dtype == expected_dtype  # 断言操作后结果的数据类型符合预期

    result = grouped["col"].aggregate(op_name)  # 对分组对象的'col'列进行聚合操作，返回结果
    assert result.dtype == expected_dtype  # 断言聚合操作后结果的数据类型符合预期
```