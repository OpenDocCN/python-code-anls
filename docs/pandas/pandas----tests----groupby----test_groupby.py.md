# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_groupby.py`

```
# 导入必要的模块和库

from datetime import datetime  # 导入 datetime 模块中的 datetime 类
import decimal  # 导入 decimal 模块，用于高精度计算
from decimal import Decimal  # 从 decimal 模块中导入 Decimal 类，用于操作高精度数字
import re  # 导入 re 模块，用于正则表达式操作

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试框架

from pandas.errors import SpecificationError  # 从 pandas.errors 模块中导入 SpecificationError 类
import pandas.util._test_decorators as td  # 导入 pandas.util._test_decorators 模块

from pandas.core.dtypes.common import is_string_dtype  # 从 pandas.core.dtypes.common 模块中导入 is_string_dtype 函数

import pandas as pd  # 导入 Pandas 库，并用 pd 作为别名
from pandas import (  # 从 Pandas 库中导入多个类和函数
    Categorical,
    DataFrame,
    Grouper,
    Index,
    Interval,
    MultiIndex,
    RangeIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    to_datetime,
)
import pandas._testing as tm  # 导入 pandas._testing 模块，用于测试
from pandas.core.arrays import BooleanArray  # 从 pandas.core.arrays 模块中导入 BooleanArray 类
import pandas.core.common as com  # 导入 pandas.core.common 模块，常用函数和类的集合

# 设置 pytest 标记，忽略特定的运行时警告
pytestmark = pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")


def test_repr():
    # 测试 Grouper 类的字符串表示
    result = repr(Grouper(key="A", level="B"))
    expected = "Grouper(key='A', level='B', sort=False, dropna=True)"
    assert result == expected


def test_groupby_nonobject_dtype(multiindex_dataframe_random_data):
    # 测试在多级索引 DataFrame 上使用非对象类型的分组
    key = multiindex_dataframe_random_data.index.codes[0]
    grouped = multiindex_dataframe_random_data.groupby(key)
    result = grouped.sum()

    expected = multiindex_dataframe_random_data.groupby(key.astype("O")).sum()
    assert result.index.dtype == np.int8
    assert expected.index.dtype == np.int64
    tm.assert_frame_equal(result, expected, check_index_type=False)


def test_groupby_nonobject_dtype_mixed():
    # 测试混合帧非转换的情况
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.array(np.random.default_rng(2).standard_normal(8), dtype="float32"),
        }
    )
    df["value"] = range(len(df))

    def max_value(group):
        return group.loc[group["value"].idxmax()]

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        applied = df.groupby("A").apply(max_value)
    result = applied.dtypes
    expected = df.dtypes
    tm.assert_series_equal(result, expected)


def test_pass_args_kwargs(ts, tsframe):
    def f(x, q=None, axis=0):
        return np.percentile(x, q, axis=axis)

    g = lambda x: np.percentile(x, 80, axis=0)

    # 测试在 Series 上的分组操作
    ts_grouped = ts.groupby(lambda x: x.month)
    agg_result = ts_grouped.agg(np.percentile, 80, axis=0)
    apply_result = ts_grouped.apply(np.percentile, 80, axis=0)
    trans_result = ts_grouped.transform(np.percentile, 80, axis=0)

    agg_expected = ts_grouped.quantile(0.8)
    trans_expected = ts_grouped.transform(g)

    tm.assert_series_equal(apply_result, agg_expected)
    tm.assert_series_equal(agg_result, agg_expected)
    tm.assert_series_equal(trans_result, trans_expected)

    # 测试传递参数和关键字参数的情况
    agg_result = ts_grouped.agg(f, q=80)
    apply_result = ts_grouped.apply(f, q=80)
    trans_result = ts_grouped.transform(f, q=80)
    tm.assert_series_equal(agg_result, agg_expected)
    # 检查两个序列是否相等，用于比较应用函数和预期结果的序列
    tm.assert_series_equal(apply_result, agg_expected)
    tm.assert_series_equal(trans_result, trans_expected)

    # 对于每种as_index的值，分别对时间序列数据进行分组并进行聚合操作
    for as_index in [True, False]:
        # 根据月份进行分组，as_index参数指定是否保留分组列为索引
        df_grouped = tsframe.groupby(lambda x: x.month, as_index=as_index)
        # 对分组后的DataFrame执行聚合操作，计算80%分位数
        agg_result = df_grouped.agg(np.percentile, 80, axis=0)
        # 对分组后的DataFrame应用函数，计算0.8分位数
        apply_result = df_grouped.apply(DataFrame.quantile, 0.8)
        # 计算预期的分位数
        expected = df_grouped.quantile(0.8)
        # 检查应用函数计算的结果与预期结果是否相等
        tm.assert_frame_equal(apply_result, expected, check_names=False)
        # 检查聚合操作计算的结果与预期结果是否相等
        tm.assert_frame_equal(agg_result, expected)

        # 对分组后的DataFrame应用函数，计算多个分位数（0.4和0.8）
        apply_result = df_grouped.apply(DataFrame.quantile, [0.4, 0.8])
        # 计算预期的多个分位数
        expected_seq = df_grouped.quantile([0.4, 0.8])
        # 如果不保留as_index的分组列为索引
        if not as_index:
            # apply将操作视为变换；.quantile知道它是一个减少操作
            # 重新设置索引为0到3的范围
            apply_result.index = range(4)
            # 插入新列"level_0"，其值为[1, 1, 2, 2]
            apply_result.insert(loc=0, column="level_0", value=[1, 1, 2, 2])
            # 插入新列"level_1"，其值为[0.4, 0.8, 0.4, 0.8]
            apply_result.insert(loc=1, column="level_1", value=[0.4, 0.8, 0.4, 0.8])
        # 检查应用函数计算的结果与预期多个分位数序列是否相等
        tm.assert_frame_equal(apply_result, expected_seq, check_names=False)

        # 对分组后的DataFrame执行聚合操作，使用自定义函数f，并传入参数q=80
        agg_result = df_grouped.agg(f, q=80)
        # 对分组后的DataFrame应用函数，计算指定分位数（0.8）
        apply_result = df_grouped.apply(DataFrame.quantile, q=0.8)
        # 检查聚合操作计算的结果与预期结果是否相等
        tm.assert_frame_equal(agg_result, expected)
        # 检查应用函数计算的结果与预期结果是否相等
        tm.assert_frame_equal(apply_result, expected, check_names=False)
def test_len():
    # 创建一个 DataFrame，包含随机生成的标准正态分布数据，10行4列
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    # 对 DataFrame 按年、月、日进行分组
    grouped = df.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day])
    # 断言分组后的数量等于 DataFrame 的长度
    assert len(grouped) == len(df)

    # 对 DataFrame 按年、月进行分组
    grouped = df.groupby([lambda x: x.year, lambda x: x.month])
    # 计算预期的分组数量
    expected = len({(x.year, x.month) for x in df.index})
    # 断言分组后的数量等于预期的分组数量
    assert len(grouped) == expected


def test_len_nan_group():
    # issue 11016
    # 创建包含 NaN 值的 DataFrame
    df = DataFrame({"a": [np.nan] * 3, "b": [1, 2, 3]})
    # 断言按 'a' 列分组后的数量为 0
    assert len(df.groupby("a")) == 0
    # 断言按 'b' 列分组后的数量为 3
    assert len(df.groupby("b")) == 3
    # 断言按 ['a', 'b'] 列分组后的数量为 0
    assert len(df.groupby(["a", "b"])) == 0


def test_groupby_timedelta_median():
    # issue 57926
    # 创建包含 Timedelta 类型数据的 DataFrame
    expected = Series(data=Timedelta("1D"), index=["foo"])
    df = DataFrame({"label": ["foo", "foo"], "timedelta": [pd.NaT, Timedelta("1D")]})
    # 对 'label' 列进行分组，并计算 'timedelta' 列的中位数
    gb = df.groupby("label")["timedelta"]
    actual = gb.median()
    # 断言计算得到的中位数与预期值相等
    tm.assert_series_equal(actual, expected, check_names=False)


@pytest.mark.parametrize("keys", [["a"], ["a", "b"]])
def test_len_categorical(dropna, observed, keys):
    # GH#57595
    # 创建包含 Categorical 类型数据的 DataFrame
    df = DataFrame(
        {
            "a": Categorical([1, 1, 2, np.nan], categories=[1, 2, 3]),
            "b": Categorical([1, 1, 2, np.nan], categories=[1, 2, 3]),
            "c": 1,
        }
    )
    # 根据给定的 keys 和参数进行分组
    gb = df.groupby(keys, observed=observed, dropna=dropna)
    result = len(gb)
    # 根据不同的条件计算预期的分组数量
    if observed and dropna:
        expected = 2
    elif observed and not dropna:
        expected = 3
    elif len(keys) == 1:
        expected = 3 if dropna else 4
    else:
        expected = 9 if dropna else 16
    # 断言计算得到的分组数量与预期值相等
    assert result == expected, f"{result} vs {expected}"


def test_basic_regression():
    # regression
    # 创建包含一系列数值的 Series
    result = Series([1.0 * x for x in list(range(1, 10)) * 10])

    data = np.random.default_rng(2).random(1100) * 10.0
    groupings = Series(data)

    # 对结果 Series 按照 groupings 进行分组，并计算均值
    grouped = result.groupby(groupings)
    grouped.mean()


def test_indices_concatenation_order():
    # GH 2808

    def f1(x):
        # 根据条件过滤 DataFrame，并将结果设置为多级索引
        y = x[(x.b % 2) == 1] ** 2
        if y.empty:
            multiindex = MultiIndex(levels=[[]] * 2, codes=[[]] * 2, names=["b", "c"])
            res = DataFrame(columns=["a"], index=multiindex)
            return res
        else:
            y = y.set_index(["b", "c"])
            return y

    def f2(x):
        # 根据条件过滤 DataFrame，并将结果设置为多级索引
        y = x[(x.b % 2) == 1] ** 2
        if y.empty:
            return DataFrame()
        else:
            y = y.set_index(["b", "c"])
            return y

    def f3(x):
        # 根据条件过滤 DataFrame，如果结果为空则创建一个特定结构的 DataFrame
        y = x[(x.b % 2) == 1] ** 2
        if y.empty:
            multiindex = MultiIndex(
                levels=[[]] * 2, codes=[[]] * 2, names=["foo", "bar"]
            )
            res = DataFrame(columns=["a", "b"], index=multiindex)
            return res
        else:
            return y

    # 创建包含特定数据的 DataFrame
    df = DataFrame({"a": [1, 2, 2, 2], "b": range(4), "c": range(5, 9)})

    df2 = DataFrame({"a": [3, 2, 2, 2], "b": range(4), "c": range(5, 9)})
    # 正确的结果
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用 assert_produces_warning 上下文管理器确保在执行 f1 函数时会产生 DeprecationWarning 警告，并匹配指定的消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按列 "a" 进行分组，然后应用函数 f1
        result1 = df.groupby("a").apply(f1)
    # 同样的操作，对第二个 DataFrame df2 进行相同的分组和函数应用
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result2 = df2.groupby("a").apply(f1)
    # 断言两个结果是否相等
    tm.assert_frame_equal(result1, result2)
    
    # 应该失败（不同级别的索引）
    msg = "Cannot concat indices that do not have the same number of levels"
    # 使用 pytest.raises 上下文管理器确保在执行 f2 函数时会抛出 AssertionError，并匹配指定的消息
    with pytest.raises(AssertionError, match=msg):
        # 对 DataFrame 按列 "a" 进行分组，然后应用函数 f2，预期会失败
        df.groupby("a").apply(f2)
    # 同样的操作，对第二个 DataFrame df2 进行相同的分组和函数应用，也预期会失败
    with pytest.raises(AssertionError, match=msg):
        df2.groupby("a").apply(f2)
    
    # 应该失败（形状不正确）
    # 使用相同的错误消息进行异常断言
    with pytest.raises(AssertionError, match=msg):
        # 对 DataFrame 按列 "a" 进行分组，然后应用函数 f3，预期会失败
        df.groupby("a").apply(f3)
    # 同样的操作，对第二个 DataFrame df2 进行相同的分组和函数应用，也预期会失败
    with pytest.raises(AssertionError, match=msg):
        df2.groupby("a").apply(f3)
def test_attr_wrapper(ts):
    # 根据每个时间戳的星期几进行分组
    grouped = ts.groupby(lambda x: x.weekday())

    # 计算每个分组的标准差
    result = grouped.std()
    # 使用numpy计算每个分组的标准差（自由度为1）
    expected = grouped.agg(lambda x: np.std(x, ddof=1))
    # 断言两个Series对象是否相等
    tm.assert_series_equal(result, expected)

    # 这很酷
    # 对分组进行描述统计
    result = grouped.describe()
    # 为每个分组计算描述统计信息，形成DataFrame
    expected = {name: gp.describe() for name, gp in grouped}
    expected = DataFrame(expected).T
    # 断言两个DataFrame对象是否相等
    tm.assert_frame_equal(result, expected)

    # 获取属性
    result = grouped.dtype
    # 计算每个分组中元素的数据类型
    expected = grouped.agg(lambda x: x.dtype)
    # 断言两个Series对象是否相等
    tm.assert_series_equal(result, expected)

    # 确保引发错误
    msg = "'SeriesGroupBy' object has no attribute 'foo'"
    with pytest.raises(AttributeError, match=msg):
        # 尝试获取grouped对象的属性'foo'，断言引发AttributeError异常
        getattr(grouped, "foo")


def test_frame_groupby(tsframe):
    # 根据每个时间戳的星期几进行分组
    grouped = tsframe.groupby(lambda x: x.weekday())

    # 聚合操作
    aggregated = grouped.aggregate("mean")
    # 断言聚合后的结果长度为5
    assert len(aggregated) == 5
    # 断言聚合后结果的列数为4
    assert len(aggregated.columns) == 4

    # 按字符串进行聚合
    tscopy = tsframe.copy()
    tscopy["weekday"] = [x.weekday() for x in tscopy.index]
    stragged = tscopy.groupby("weekday").aggregate("mean")
    # 断言两个DataFrame对象是否相等（忽略列名检查）
    tm.assert_frame_equal(stragged, aggregated, check_names=False)

    # 变换操作
    grouped = tsframe.head(30).groupby(lambda x: x.weekday())
    transformed = grouped.transform(lambda x: x - x.mean())
    # 断言变换后结果的长度为30
    assert len(transformed) == 30
    # 断言变换后结果的列数为4
    assert len(transformed.columns) == 4

    # 变换传播
    transformed = grouped.transform(lambda x: x.mean())
    # 遍历每个分组，断言变换后结果与分组均值相等
    for name, group in grouped:
        mean = group.mean()
        for idx in group.index:
            tm.assert_series_equal(transformed.xs(idx), mean, check_names=False)

    # 迭代分组
    for weekday, group in grouped:
        # 断言每个分组的第一个时间戳的星期几与分组的键相等
        assert group.index[0].weekday() == weekday

    # 获取groups和group_indices属性
    groups = grouped.groups
    indices = grouped.indices

    # 遍历分组和索引，断言取出的索引与group的索引相等
    for k, v in groups.items():
        samething = tsframe.index.take(indices[k])
        assert (samething == v).all()


def test_frame_set_name_single(df):
    # 根据'A'列进行分组
    grouped = df.groupby("A")

    # 计算均值，确保结果的索引名为'A'
    result = grouped.mean(numeric_only=True)
    assert result.index.name == "A"

    # 当as_index=False时，分组后的均值结果索引名不为'A'
    result = df.groupby("A", as_index=False).mean(numeric_only=True)
    assert result.index.name != "A"

    # 对'C'和'D'列进行聚合计算均值
    result = grouped[["C", "D"]].agg("mean")
    assert result.index.name == "A"

    # 对'C'列计算均值和'D'列计算标准差
    result = grouped.agg({"C": "mean", "D": "std"})
    assert result.index.name == "A"

    # 单独对'C'列计算均值，确保结果的索引名为'A'
    result = grouped["C"].mean()
    assert result.index.name == "A"
    result = grouped["C"].agg("mean")
    assert result.index.name == "A"
    result = grouped["C"].agg(["mean", "std"])
    assert result.index.name == "A"

    # 断言引发SpecificationError异常，因为不支持嵌套的重命名
    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        grouped["C"].agg({"foo": "mean", "bar": "std"})


def test_multi_func(df):
    col1 = df["A"]
    col2 = df["B"]

    # 根据'A'和'B'列进行分组
    grouped = df.groupby([col1.get, col2.get])
    agged = grouped.mean(numeric_only=True)
    expected = df.groupby(["A", "B"]).mean()

    # TODO groupby get drops names
    # 使用 pandas.testing 模块中的 assert_frame_equal 函数来比较两个 DataFrame 的内容是否相同
    tm.assert_frame_equal(
        agged.loc[:, ["C", "D"]], expected.loc[:, ["C", "D"]], check_names=False
    )

    # 创建一个 DataFrame 对象 df，包含随机生成的数据和分类键
    df = DataFrame(
        {
            "v1": np.random.default_rng(2).standard_normal(6),
            "v2": np.random.default_rng(2).standard_normal(6),
            "k1": np.array(["b", "b", "b", "a", "a", "a"]),
            "k2": np.array(["1", "1", "1", "2", "2", "2"]),
        },
        index=["one", "two", "three", "four", "five", "six"],
    )
    
    # 创建一个名为 "grouped" 的分组对象，根据 "k1" 和 "k2" 列进行分组
    grouped = df.groupby(["k1", "k2"])
    
    # 对分组后的 DataFrame 执行汇总操作，计算每个分组的和
    grouped.agg("sum")
# 定义一个测试函数，用于测试在多个键和多个函数下的分组聚合操作
def test_multi_key_multiple_functions(df):
    # 按列"A"和"B"进行分组，选择列"C"，生成分组对象
    grouped = df.groupby(["A", "B"])["C"]

    # 对分组对象应用聚合函数"mean"和"std"，生成包含这两个统计量的 DataFrame
    agged = grouped.agg(["mean", "std"])
    
    # 期望的结果是分别计算均值和标准差，构造相应的 DataFrame
    expected = DataFrame({"mean": grouped.agg("mean"), "std": grouped.agg("std")})
    
    # 使用测试框架验证实际结果和期望结果是否相等
    tm.assert_frame_equal(agged, expected)


# 定义另一个测试函数，用于测试在多个键和多个函数列表下的分组聚合操作
def test_frame_multi_key_function_list():
    # 创建一个 DataFrame 包含多个键"A"、"B"和随机数列"D", "E", "F"
    data = DataFrame(
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
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )
    
    # 按列"A"和"B"进行分组，生成分组对象
    grouped = data.groupby(["A", "B"])
    
    # 定义需要应用的聚合函数列表
    funcs = ["mean", "std"]
    
    # 对分组对象应用函数列表中的函数，生成包含统计量的 DataFrame
    agged = grouped.agg(funcs)
    
    # 构造期望的结果，分别对"D", "E", "F"列应用函数列表中的函数
    expected = pd.concat(
        [grouped["D"].agg(funcs), grouped["E"].agg(funcs), grouped["F"].agg(funcs)],
        keys=["D", "E", "F"],
        axis=1,
    )
    
    # 使用测试框架验证实际结果和期望结果是否相等
    assert isinstance(agged.index, MultiIndex)
    assert isinstance(expected.index, MultiIndex)
    tm.assert_frame_equal(agged, expected)


# 定义另一个测试函数，用于测试在多个键和多个函数列表下的分组聚合操作中部分失败的情况
def test_frame_multi_key_function_list_partial_failure():
    # 创建一个 DataFrame 包含多个键"A", "B", "C"和随机数列"D", "E", "F"
    data = DataFrame(
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
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )
    
    # 按列"A"和"B"进行分组，生成分组对象
    grouped = data.groupby(["A", "B"])
    
    # 定义需要应用的聚合函数列表
    funcs = ["mean", "std"]
    
    # 设置期望的错误消息
    msg = re.escape("agg function failed [how->mean,dtype->")
    
    # 使用 pytest 框架验证分组对象应用函数列表中的函数时是否会抛出预期的类型错误，并匹配期望的错误消息
    with pytest.raises(TypeError, match=msg):
        grouped.agg(funcs)
def test_groupby_multiple_columns(df, op):
    # 将输入的数据框赋值给变量data
    data = df
    # 按照"A"和"B"两列进行分组，返回一个分组对象
    grouped = data.groupby(["A", "B"])

    # 对分组对象应用操作op，得到结果result1
    result1 = op(grouped)

    # 初始化空列表keys和values，用于存储分组键和操作结果
    keys = []
    values = []
    # 遍历按"A"列分组后的分组对象
    for n1, gp1 in data.groupby("A"):
        # 再次按"B"列分组
        for n2, gp2 in gp1.groupby("B"):
            # 将分组键(n1, n2)添加到keys列表中
            keys.append((n1, n2))
            # 对分组对象gp2选取"C"和"D"两列，应用操作op，并将结果添加到values列表中
            values.append(op(gp2.loc[:, ["C", "D"]]))

    # 使用keys创建多级索引MultiIndex对象，指定索引的名称为"A"和"B"
    mi = MultiIndex.from_tuples(keys, names=["A", "B"])
    # 将values中的操作结果沿轴1（列方向）拼接成DataFrame，并进行转置
    expected = pd.concat(values, axis=1).T
    # 设置拼接后DataFrame的索引为mi
    expected.index = mi

    # 对每个列("C"和"D")进行检查
    for col in ["C", "D"]:
        # 对分组对象grouped中的每列应用操作op，得到结果result_col
        result_col = op(grouped[col])
        # 从result1中获取列col的数据，并赋值给pivoted
        pivoted = result1[col]
        # 从expected中获取列col的数据，并赋值给exp
        exp = expected[col]
        # 使用tm.assert_series_equal函数比较result_col和exp的内容
        tm.assert_series_equal(result_col, exp)
        # 使用tm.assert_series_equal函数比较pivoted和exp的内容
        tm.assert_series_equal(pivoted, exp)

    # 对单列"C"进行分组并计算平均值，结果赋值给result
    result = data["C"].groupby([data["A"], data["B"]]).mean()
    # 使用data按"A"和"B"分组后计算"C"列的平均值，结果赋值给expected
    expected = data.groupby(["A", "B"]).mean()["C"]
    # 使用tm.assert_series_equal函数比较result和expected的内容
    tm.assert_series_equal(result, expected)


def test_as_index_select_column():
    # GH 5764
    # 创建一个包含三行两列数据的DataFrame
    df = DataFrame([[1, 2], [1, 4], [5, 6]], columns=["A", "B"])
    # 对"A"列进行分组，不作为索引，然后获取组"A"为1的"B"列数据，结果赋值给result
    result = df.groupby("A", as_index=False)["B"].get_group(1)
    # 创建一个Series对象，包含[2, 4]，名称为"B"
    expected = Series([2, 4], name="B")
    # 使用tm.assert_series_equal函数比较result和expected的内容
    tm.assert_series_equal(result, expected)

    # 对"A"列进行分组，不作为索引，同时使用group_keys=True保留组的键，
    # 然后对每组的"B"列数据应用累积求和操作，结果赋值给result
    result = df.groupby("A", as_index=False, group_keys=True)["B"].apply(
        lambda x: x.cumsum()
    )
    # 创建一个Series对象，包含[2, 6, 6]，名称为"B"，索引为[0, 1, 2]
    expected = Series([2, 6, 6], name="B", index=range(3))
    # 使用tm.assert_series_equal函数比较result和expected的内容
    tm.assert_series_equal(result, expected)


def test_groupby_as_index_select_column_sum_empty_df():
    # GH 35246
    # 创建一个空DataFrame，列名为["A", "B", "C"]，索引名为"alpha"
    df = DataFrame(columns=Index(["A", "B", "C"], name="alpha"))
    # 对"A"列进行分组，不作为索引，然后对"B"列进行求和，结果赋值给left
    left = df.groupby(by="A", as_index=False)["B"].sum(numeric_only=False)

    # 创建一个空的DataFrame，列与df的前两列相同，行数为0
    expected = DataFrame(columns=df.columns[:2], index=range(0))
    # 设置expected的列名为空列表，以匹配GH#50744的要求
    expected.columns.names = [None]
    # 使用tm.assert_frame_equal函数比较left和expected的内容
    tm.assert_frame_equal(left, expected)


def test_ops_not_as_index(reduction_func):
    # GH 10355, 21090
    # 使用as_index=False时不应修改分组列

    # 如果reduction_func为"corrwith"、"nth"或"ngroup"，则跳过此测试
    if reduction_func in ("corrwith", "nth", "ngroup"):
        pytest.skip(f"GH 5755: Test not applicable for {reduction_func}")

    # 创建一个包含100行两列数据的DataFrame，列名为["a", "b"]
    df = DataFrame(
        np.random.default_rng(2).integers(0, 5, size=(100, 2)), columns=["a", "b"]
    )
    # 使用getattr函数调用df.groupby("a")后的reduction_func函数，并将结果赋值给expected
    expected = getattr(df.groupby("a"), reduction_func)()
    # 如果reduction_func为"size"，则将expected的名称改为"size"
    if reduction_func == "size":
        expected = expected.rename("size")
    # 重置expected的索引
    expected = expected.reset_index()

    # 如果reduction_func不是"size"，则将expected的"a"列的dtype设置为df中"a"列的dtype
    if reduction_func != "size":
        expected["a"] = expected["a"].astype(df["a"].dtype)

    # 对df按"a"列进行分组，不作为索引，结果赋值给g
    g = df.groupby("a", as_index=False)

    # 使用getattr函数调用g后的reduction_func函数，并将结果赋值给result
    result = getattr(g, reduction_func)()
    # 使用tm.assert_frame_equal函数比较result和expected的内容
    tm.assert_frame_equal(result, expected)

    # 对g应用agg函数，使用reduction_func作为参数，结果赋值给result
    result = g.agg(reduction_func)
    # 使用tm.assert_frame_equal函数比较result和expected的内容
    tm.assert_frame_equal(result, expected)

    # 对g["b"]进行reduction_func操作，并将结果赋值给result
    result = getattr(g["b"], reduction_func)()
    # 使用tm.assert_frame_equal函数比较result和expected的内容
    tm.assert_frame_equal(result, expected)

    # 对g["b"]应用agg函数，使用reduction_func作为参数，结果赋值给result
    result = g["b"].agg(reduction_func)
    # 使用tm.assert_frame_equal函数比较result和expected的内容
    tm.assert_frame_equal(result, expected)


def test_as_index_series_return_frame(df):
    # 对df按"A"列进行分组，不作为索引，结果赋值给grouped
    grouped = df.groupby("A", as_index=False)
    # 根据列"A"和"B"进行分组，并且不将"A"和"B"作为索引
    grouped2 = df.groupby(["A", "B"], as_index=False)

    # 对分组后的数据框grouped，针对列"C"计算求和
    result = grouped["C"].agg("sum")
    # 从整体求和的结果grouped中选取"A"和"C"列，形成期望的数据框expected
    expected = grouped.agg("sum").loc[:, ["A", "C"]]
    # 断言result是一个DataFrame对象
    assert isinstance(result, DataFrame)
    # 使用测试模块tm来确保result和expected数据框相等
    tm.assert_frame_equal(result, expected)

    # 对另一个分组数据框grouped2，同样对列"C"计算求和
    result2 = grouped2["C"].agg("sum")
    # 从整体求和的结果grouped2中选取"A"、"B"和"C"列，形成期望的数据框expected2
    expected2 = grouped2.agg("sum").loc[:, ["A", "B", "C"]]
    # 断言result2是一个DataFrame对象
    assert isinstance(result2, DataFrame)
    # 使用测试模块tm来确保result2和expected2数据框相等
    tm.assert_frame_equal(result2, expected2)

    # 对grouped再次对列"C"进行求和，使用简化的语法形式
    result = grouped["C"].sum()
    # 从整体求和的结果grouped中选取"A"和"C"列，形成期望的数据框expected
    expected = grouped.sum().loc[:, ["A", "C"]]
    # 断言result是一个DataFrame对象
    assert isinstance(result, DataFrame)
    # 使用测试模块tm来确保result和expected数据框相等
    tm.assert_frame_equal(result, expected)

    # 对grouped2再次对列"C"进行求和，使用简化的语法形式
    result2 = grouped2["C"].sum()
    # 从整体求和的结果grouped2中选取"A"、"B"和"C"列，形成期望的数据框expected2
    expected2 = grouped2.sum().loc[:, ["A", "B", "C"]]
    # 断言result2是一个DataFrame对象
    assert isinstance(result2, DataFrame)
    # 使用测试模块tm来确保result2和expected2数据框相等
    tm.assert_frame_equal(result2, expected2)
def test_as_index_series_column_slice_raises(df):
    # GH15072: 测试用例标识号
    # 根据"A"列对DataFrame进行分组，返回GroupBy对象，不将"A"列作为索引
    grouped = df.groupby("A", as_index=False)
    # 设置预期的错误消息，用于断言检查异常
    msg = r"Column\(s\) C already selected"

    # 使用pytest的raises装饰器检查是否抛出IndexError异常，并匹配预期的错误消息
    with pytest.raises(IndexError, match=msg):
        # 尝试从GroupBy对象中选择列"C"，并获取子元素"D"
        grouped["C"].__getitem__("D")


def test_groupby_as_index_cython(df):
    data = df

    # single-key: 单关键字分组
    grouped = data.groupby("A", as_index=False)
    # 计算分组后每组的均值，仅包括数值型列
    result = grouped.mean(numeric_only=True)
    # 创建预期的DataFrame，包括分组键"A"和各数值型列的均值
    expected = data.groupby(["A"]).mean(numeric_only=True)
    # 将分组键"A"插入到DataFrame的第一列
    expected.insert(0, "A", expected.index)
    # 重新设置预期DataFrame的索引为RangeIndex
    expected.index = RangeIndex(len(expected))
    # 断言两个DataFrame是否相等
    tm.assert_frame_equal(result, expected)

    # multi-key: 多关键字分组
    grouped = data.groupby(["A", "B"], as_index=False)
    # 计算分组后每组的均值
    result = grouped.mean()
    # 创建预期的DataFrame，包括分组键"A"和"B"，以及各数值型列的均值
    expected = data.groupby(["A", "B"]).mean()

    # 从预期DataFrame的索引中提取各分组键的数组
    arrays = list(zip(*expected.index.values))
    # 将分组键"A"插入到DataFrame的第一列
    expected.insert(0, "A", arrays[0])
    # 将分组键"B"插入到DataFrame的第二列
    expected.insert(1, "B", arrays[1])
    # 重新设置预期DataFrame的索引为RangeIndex
    expected.index = RangeIndex(len(expected))
    # 断言两个DataFrame是否相等
    tm.assert_frame_equal(result, expected)


def test_groupby_as_index_series_scalar(df):
    grouped = df.groupby(["A", "B"], as_index=False)

    # GH #421: 测试用例标识号

    # 对分组后的"C"列应用长度函数，计算结果
    result = grouped["C"].agg(len)
    # 计算分组后各列的长度，选择"A"、"B"、"C"列
    expected = grouped.agg(len).loc[:, ["A", "B", "C"]]
    # 断言两个DataFrame是否相等
    tm.assert_frame_equal(result, expected)


def test_groupby_multiple_key():
    # 创建一个DataFrame，包含随机数值
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    # 根据年、月、日进行分组
    grouped = df.groupby([lambda x: x.year, lambda x: x.month, lambda x: x.day])
    # 对分组后的数据进行求和聚合
    agged = grouped.sum()
    # 检查DataFrame的数值是否近似相等
    tm.assert_almost_equal(df.values, agged.values)


def test_groupby_multi_corner(df):
    # 测试在存在全部为NA值的列时，分组聚合的情况
    df = df.copy()
    # 将新列"bad"设置为全部为NaN
    df["bad"] = np.nan
    # 对"A"、"B"列进行分组，并计算均值
    agged = df.groupby(["A", "B"]).mean()

    # 创建预期的DataFrame，包括均值计算结果和全部为NaN的列"bad"
    expected = df.groupby(["A", "B"]).mean()
    expected["bad"] = np.nan

    # 断言两个DataFrame是否相等
    tm.assert_frame_equal(agged, expected)


def test_raises_on_nuisance(df):
    grouped = df.groupby("A")
    # 设置预期的错误消息，用于断言检查异常
    msg = re.escape("agg function failed [how->mean,dtype->")
    # 使用pytest的raises装饰器检查是否抛出TypeError异常，并匹配预期的错误消息
    with pytest.raises(TypeError, match=msg):
        # 对分组后的数据应用均值聚合函数
        grouped.agg("mean")
    with pytest.raises(TypeError, match=msg):
        # 计算分组后的均值
        grouped.mean()

    # 仅保留"A"、"C"、"D"列，创建新的DataFrame
    df = df.loc[:, ["A", "C", "D"]]
    # 将新列"E"设置为当前日期和时间
    df["E"] = datetime.now()
    # 对"A"列进行分组
    grouped = df.groupby("A")
    # 设置预期的错误消息，用于断言检查异常
    msg = "datetime64 type does not support operation 'sum'"
    # 使用pytest的raises装饰器检查是否抛出TypeError异常，并匹配预期的错误消息
    with pytest.raises(TypeError, match=msg):
        # 对分组后的数据应用求和聚合函数
        grouped.agg("sum")
    with pytest.raises(TypeError, match=msg):
        # 计算分组后的求和
        grouped.sum()


@pytest.mark.parametrize(
    "agg_function",
    ["max", "min"],
)
def test_keep_nuisance_agg(df, agg_function):
    # GH 38815: 测试用例标识号
    grouped = df.groupby("A")
    # 根据参数agg_function选择相应的聚合函数并应用
    result = getattr(grouped, agg_function)()
    # 创建预期的DataFrame，包含聚合后的结果
    expected = result.copy()
    # 根据分组键"A"，选择相应的列"B"并应用对应的聚合函数
    expected.loc["bar", "B"] = getattr(df.loc[df["A"] == "bar", "B"], agg_function)()
    expected.loc["foo", "B"] = getattr(df.loc[df["A"] == "foo", "B"], agg_function)()
    # 断言两个DataFrame是否相等
    tm.assert_frame_equal(result, expected)
    # 定义一个包含统计函数名称的列表，用于后续数据分析和计算操作
    ["sum", "mean", "prod", "std", "var", "sem", "median"],
# 定义测试函数，使用 pytest 框架进行测试参数化
@pytest.mark.parametrize("numeric_only", [True, False])
def test_omit_nuisance_agg(df, agg_function, numeric_only):
    # 创建 DataFrame 按列 "A" 进行分组
    grouped = df.groupby("A")

    # 不删除干扰列的聚合函数列表
    no_drop_nuisance = ("var", "std", "sem", "mean", "prod", "median")
    # 如果聚合函数在不删除干扰列的列表中且 numeric_only 参数为 False
    if agg_function in no_drop_nuisance and not numeric_only:
        # 根据聚合函数类型设置异常类型和匹配的错误消息
        if agg_function in ("std", "sem"):
            klass = ValueError
            msg = "could not convert string to float: 'one'"
        else:
            klass = TypeError
            msg = re.escape(f"agg function failed [how->{agg_function},dtype->")
        # 断言调用特定聚合函数时会抛出异常
        with pytest.raises(klass, match=msg):
            getattr(grouped, agg_function)(numeric_only=numeric_only)
    else:
        # 否则，调用指定的聚合函数并返回结果
        result = getattr(grouped, agg_function)(numeric_only=numeric_only)
        # 如果 numeric_only 为 False 并且聚合函数为 "sum"
        if not numeric_only and agg_function == "sum":
            # 预期返回的列名列表
            columns = ["A", "B", "C", "D"]
        else:
            columns = ["A", "C", "D"]
        # 计算预期的聚合结果
        expected = getattr(df.loc[:, columns].groupby("A"), agg_function)(
            numeric_only=numeric_only
        )
        # 断言结果 DataFrame 等于预期的 DataFrame
        tm.assert_frame_equal(result, expected)


# 测试函数：测试在 Python 中对单列进行聚合时是否引发 ValueError
def test_raise_on_nuisance_python_single(df):
    # 创建 DataFrame 按列 "A" 进行分组
    grouped = df.groupby("A")
    # 断言调用 skew() 方法时会抛出 ValueError 异常并匹配特定错误消息
    with pytest.raises(ValueError, match="could not convert"):
        grouped.skew()


# 测试函数：测试在 Python 中对多列进行聚合时是否引发 TypeError
def test_raise_on_nuisance_python_multiple(three_group):
    # 创建 DataFrame 按列 ["A", "B"] 进行分组
    grouped = three_group.groupby(["A", "B"])
    # 预期的错误消息
    msg = re.escape("agg function failed [how->mean,dtype->")
    # 断言调用 agg("mean") 方法时会抛出 TypeError 异常并匹配预期的错误消息
    with pytest.raises(TypeError, match=msg):
        grouped.agg("mean")
    # 断言调用 mean() 方法时会抛出 TypeError 异常并匹配预期的错误消息
    with pytest.raises(TypeError, match=msg):
        grouped.mean()


# 测试函数：测试处理空分组情况下的聚合操作
def test_empty_groups_corner(multiindex_dataframe_random_data):
    # 创建包含随机数据的 DataFrame
    df = DataFrame(
        {
            "k1": np.array(["b", "b", "b", "a", "a", "a"]),
            "k2": np.array(["1", "1", "1", "2", "2", "2"]),
            "k3": ["foo", "bar"] * 3,
            "v1": np.random.default_rng(2).standard_normal(6),
            "v2": np.random.default_rng(2).standard_normal(6),
        }
    )

    # 根据列 "k1" 和 "k2" 进行分组
    grouped = df.groupby(["k1", "k2"])
    # 执行 v1 和 v2 列的均值聚合操作，并与预期结果进行比较
    result = grouped[["v1", "v2"]].agg("mean")
    expected = grouped.mean(numeric_only=True)
    tm.assert_frame_equal(result, expected)

    # 对多级索引的 DataFrame 进行分组并应用均值计算
    grouped = multiindex_dataframe_random_data[3:5].groupby(level=0)
    agged = grouped.apply(lambda x: x.mean())
    agged_A = grouped["A"].apply("mean")
    # 断言均值聚合后的 Series 等于预期的 Series
    tm.assert_series_equal(agged["A"], agged_A)
    # 断言结果的索引名称为 "first"
    assert agged.index.name == "first"


# 测试函数：测试传入无效函数时是否引发 TypeError
def test_nonsense_func():
    # 创建包含单个元素的 DataFrame
    df = DataFrame([0])
    # 预期的错误消息
    msg = r"unsupported operand type\(s\) for \+: 'int' and 'str'"
    # 断言调用 groupby(lambda x: x + "foo") 方法时会抛出 TypeError 异常并匹配预期的错误消息
    with pytest.raises(TypeError, match=msg):
        df.groupby(lambda x: x + "foo")


# 测试函数：测试对多级索引 DataFrame 的聚合输出进行包装
def test_wrap_aggregated_output_multindex(multiindex_dataframe_random_data):
    # 转置多级索引 DataFrame
    df = multiindex_dataframe_random_data.T
    # 添加新的列 "baz", "two" 到 DataFrame
    df["baz", "two"] = "peekaboo"
    # 创建一个包含两个 NumPy 数组的列表，每个数组包含三个元素 [0, 0, 1]
    keys = [np.array([0, 0, 1]), np.array([0, 0, 1])]
    # 创建一个正则表达式，用于转义指定的字符串 "agg function failed [how->mean,dtype->"
    msg = re.escape("agg function failed [how->mean,dtype->")
    # 使用 pytest 来确保以下代码块中的 TypeError 异常被抛出，并且异常消息匹配预期的正则表达式 msg
    with pytest.raises(TypeError, match=msg):
        # 对 DataFrame df 进行按照 keys 列表中的键进行分组，并尝试应用 "mean" 聚合函数
        df.groupby(keys).agg("mean")
    # 从 DataFrame df 中删除 "baz" 和 "two" 列，然后对剩余列按照 keys 列表中的键进行分组，并应用 "mean" 聚合函数
    agged = df.drop(columns=("baz", "two")).groupby(keys).agg("mean")
    # 断言 agged 的列类型是 MultiIndex
    assert isinstance(agged.columns, MultiIndex)

    # 定义一个函数 aggfun，用于聚合操作，接收一个序列 ser 作为参数
    def aggfun(ser):
        # 如果序列 ser 的名称为 ("foo", "one")，则抛出 TypeError 异常，并附带特定的错误消息
        if ser.name == ("foo", "one"):
            raise TypeError("Test error message")
        # 否则返回序列 ser 的总和
        return ser.sum()

    # 使用 pytest 来确保以下代码块中的 TypeError 异常被抛出，并且异常消息为 "Test error message"
    with pytest.raises(TypeError, match="Test error message"):
        # 对 DataFrame df 进行按照 keys 列表中的键进行分组，并尝试应用自定义的 aggfun 函数进行聚合
        df.groupby(keys).aggregate(aggfun)
def test_groupby_level_apply(multiindex_dataframe_random_data):
    # 对多级索引的DataFrame按第一级索引分组，并计算每组的数量
    result = multiindex_dataframe_random_data.groupby(level=0).count()
    # 断言分组后的索引名称为 "first"
    assert result.index.name == "first"
    # 对多级索引的DataFrame按第二级索引分组，并计算每组的数量
    result = multiindex_dataframe_random_data.groupby(level=1).count()
    # 断言分组后的索引名称为 "second"
    assert result.index.name == "second"

    # 对多级索引的DataFrame的"A"列按第一级索引分组，并计算每组的数量
    result = multiindex_dataframe_random_data["A"].groupby(level=0).count()
    # 断言分组后的索引名称为 "first"
    assert result.index.name == "first"


def test_groupby_level_mapper(multiindex_dataframe_random_data):
    # 将多级索引的DataFrame重置索引，使其变为普通的索引DataFrame
    deleveled = multiindex_dataframe_random_data.reset_index()

    # 定义用于分组的映射字典
    mapper0 = {"foo": 0, "bar": 0, "baz": 1, "qux": 1}
    mapper1 = {"one": 0, "two": 0, "three": 1}

    # 根据第一级索引的映射进行分组，并计算每组的总和
    result0 = multiindex_dataframe_random_data.groupby(mapper0, level=0).sum()
    # 根据第二级索引的映射进行分组，并计算每组的总和
    result1 = multiindex_dataframe_random_data.groupby(mapper1, level=1).sum()

    # 根据重置索引后的DataFrame的"first"列的映射创建数组
    mapped_level0 = np.array(
        [mapper0.get(x) for x in deleveled["first"]], dtype=np.int64
    )
    # 根据重置索引后的DataFrame的"second"列的映射创建数组
    mapped_level1 = np.array(
        [mapper1.get(x) for x in deleveled["second"]], dtype=np.int64
    )
    # 根据映射后的第一级索引数组进行分组，并计算每组的总和，期望的结果
    expected0 = multiindex_dataframe_random_data.groupby(mapped_level0).sum()
    # 根据映射后的第二级索引数组进行分组，并计算每组的总和，期望的结果
    expected1 = multiindex_dataframe_random_data.groupby(mapped_level1).sum()
    # 设置期望结果的索引名称
    expected0.index.name, expected1.index.name = "first", "second"

    # 使用测试工具函数比较结果是否相等
    tm.assert_frame_equal(result0, expected0)
    tm.assert_frame_equal(result1, expected1)


def test_groupby_level_nonmulti():
    # 创建一个Series对象，其索引为 [1, 2, 3, 1, 4, 5, 2, 6]，名称为 "foo"
    s = Series([1, 2, 3, 10, 4, 5, 20, 6], Index([1, 2, 3, 1, 4, 5, 2, 6], name="foo"))
    # 创建一个期望的Series对象，其索引为 range(1, 7)，名称为 "foo"
    expected = Series([11, 22, 3, 4, 5, 6], Index(range(1, 7), name="foo"))

    # 对Series按索引级别 0 进行分组，并计算每组的总和
    result = s.groupby(level=0).sum()
    # 使用测试工具函数比较结果是否相等
    tm.assert_series_equal(result, expected)
    # 同上，对Series按索引级别 [0] 进行分组，并计算每组的总和
    result = s.groupby(level=[0]).sum()
    # 使用测试工具函数比较结果是否相等
    tm.assert_series_equal(result, expected)
    # 同上，对Series按索引级别 -1 进行分组，并计算每组的总和
    result = s.groupby(level=-1).sum()
    # 使用测试工具函数比较结果是否相等
    tm.assert_series_equal(result, expected)
    # 同上，对Series按索引级别 [-1] 进行分组，并计算每组的总和
    result = s.groupby(level=[-1]).sum()
    # 使用测试工具函数比较结果是否相等
    tm.assert_series_equal(result, expected)

    # 预期引发 ValueError 异常，因为 level=1 对于非多级索引的Series无效
    msg = "level > 0 or level < -1 only valid with MultiIndex"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=1)
    # 预期引发 ValueError 异常，因为 level=-2 对于非多级索引的Series无效
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=-2)
    # 预期引发 ValueError 异常，因为 level=[] 对于非多级索引的Series无效
    msg = "No group keys passed!"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[])
    # 预期引发 ValueError 异常，因为 level=[0, 0] 对于非多级索引的Series无效
    msg = "multiple levels only valid with MultiIndex"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[0, 0])
    # 预期引发 ValueError 异常，因为 level=[0, 1] 对于非多级索引的Series无效
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[0, 1])
    # 预期引发 ValueError 异常，因为 level=[1] 对于非多级索引的Series无效
    msg = "level > 0 or level < -1 only valid with MultiIndex"
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[1])


def test_groupby_complex():
    # 创建一个复数Series对象，其索引为 [0, 0, 1, 1]
    a = Series(data=np.arange(4) * (1 + 2j), index=[0, 0, 1, 1])
    # 创建一个期望的复数Series对象
    expected = Series((1 + 2j, 5 + 10j))

    # 对复数Series按索引级别 0 进行分组，并计算每组的总和
    result = a.groupby(level=0).sum()
    # 使用测试工具函数比较结果是否相等
    tm.assert_series_equal(result, expected)


def test_groupby_complex_mean():
    # 创建一个DataFrame对象，包含三个字典形式的数据行
    df = DataFrame(
        [
            {"a": 2, "b": 1 + 2j},
            {"a": 1, "b": 1 + 1j},
            {"a": 1, "b": 1 + 2j},
        ]
    )
    # 使用 pandas 的 groupby 方法按列 "b" 对 DataFrame df 进行分组，并计算每组的平均值
    result = df.groupby("b").mean()

    # 创建一个预期的 DataFrame 对象 expected，包含指定的数据和索引
    expected = DataFrame(
        [[1.0], [1.5]],
        index=Index([(1 + 1j), (1 + 2j)], name="b"),  # 指定索引为复数 (1+1j) 和 (1+2j)，索引名为 "b"
        columns=Index(["a"]),  # 指定列名为 "a"
    )

    # 使用 pandas.testing 模块中的 assert_frame_equal 函数比较 result 和 expected 两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于验证复数类型的分组操作
def test_groupby_complex_numbers(using_infer_string):
    # GH 17927
    # 创建一个包含复数和整数的数据框 DataFrame
    df = DataFrame(
        [
            {"a": 1, "b": 1 + 1j},
            {"a": 1, "b": 1 + 2j},
            {"a": 4, "b": 1},
        ]
    )
    # 根据 using_infer_string 参数确定 dtype 类型
    dtype = "string[pyarrow_numpy]" if using_infer_string else object
    # 创建期望的数据框，指定复数的索引和数据类型
    expected = DataFrame(
        np.array([1, 1, 1], dtype=np.int64),
        index=Index([(1 + 1j), (1 + 2j), (1 + 0j)], name="b"),
        columns=Index(["a"], dtype=dtype),
    )
    # 对数据框进行分组统计，不排序
    result = df.groupby("b", sort=False).count()
    # 使用测试工具检查结果是否符合期望
    tm.assert_frame_equal(result, expected)

    # 按复数大小的幅度排序索引
    expected.index = Index([(1 + 0j), (1 + 1j), (1 + 2j)], name="b")
    # 对数据框进行分组统计，按索引排序
    result = df.groupby("b", sort=True).count()
    # 使用测试工具检查结果是否符合期望
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于验证不同索引方式的系列分组操作
def test_groupby_series_indexed_differently():
    # 创建第一个系列，指定自定义索引
    s1 = Series(
        [5.0, -9.0, 4.0, 100.0, -5.0, 55.0, 6.7],
        index=Index(["a", "b", "c", "d", "e", "f", "g"]),
    )
    # 创建第二个系列，指定自定义索引
    s2 = Series(
        [1.0, 1.0, 4.0, 5.0, 5.0, 7.0], index=Index(["a", "b", "d", "f", "g", "h"])
    )

    # 按照 s2 系列分组 s1 系列
    grouped = s1.groupby(s2)
    # 对分组后的结果进行求平均值
    agged = grouped.mean()
    # 重新索引 s2 并按照 s1 索引求平均值，作为期望的结果
    exp = s1.groupby(s2.reindex(s1.index)).mean()
    # 使用测试工具检查结果是否符合期望
    tm.assert_series_equal(agged, exp)


# 定义一个测试函数，用于验证具有层次化列的数据框分组操作
def test_groupby_with_hier_columns():
    # 创建层次化索引元组列表
    tuples = list(
        zip(
            *[
                ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
                ["one", "two", "one", "two", "one", "two", "one", "two"],
            ]
        )
    )
    # 创建层次化索引对象
    index = MultiIndex.from_tuples(tuples)
    # 创建层次化列索引对象
    columns = MultiIndex.from_tuples(
        [("A", "cat"), ("B", "dog"), ("B", "cat"), ("A", "dog")]
    )
    # 创建随机数据框，指定索引和列索引
    df = DataFrame(
        np.random.default_rng(2).standard_normal((8, 4)), index=index, columns=columns
    )

    # 按照第一层级分组，并计算平均值
    result = df.groupby(level=0).mean()
    # 使用测试工具检查结果的列索引是否与原始的列索引相同
    tm.assert_index_equal(result.columns, columns)

    # 使用 agg 方法按第一层级分组并计算平均值
    result = df.groupby(level=0).agg("mean")
    # 使用测试工具检查结果的列索引是否与原始的列索引相同
    tm.assert_index_equal(result.columns, columns)

    # 使用 apply 方法按第一层级分组并应用 lambda 函数计算平均值
    result = df.groupby(level=0).apply(lambda x: x.mean())
    # 使用测试工具检查结果的列索引是否与原始的列索引相同
    tm.assert_index_equal(result.columns, columns)

    # 添加一个无关紧要的列
    # 按第一层级分组并计算平均值，只考虑数值列
    sorted_columns, _ = columns.sortlevel(0)
    result = df.groupby(level=0).mean(numeric_only=True)
    # 使用测试工具检查结果的列索引是否与去除最后一列后的原始列索引相同
    tm.assert_index_equal(result.columns, df.columns[:-1])


# 定义一个测试函数，用于验证按 ndarray 进行分组的数据框操作
def test_grouping_ndarray(df):
    # 按照 df["A"] 的值进行分组
    grouped = df.groupby(df["A"].values)
    # 对分组后的结果进行求和
    result = grouped.sum()
    # 按照 df["A"] 重命名后的索引进行分组并求和，作为期望的结果
    expected = df.groupby(df["A"].rename(None)).sum()
    # 使用测试工具检查结果是否符合期望
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于验证具有多标签索引的数据框分组操作
def test_groupby_wrong_multi_labels():
    # 创建自定义索引名为 "index" 的索引对象
    index = Index([0, 1, 2, 3, 4], name="index")
    # 创建数据框，包含多个列和自定义索引
    data = DataFrame(
        {
            "foo": ["foo1", "foo1", "foo2", "foo1", "foo3"],
            "bar": ["bar1", "bar2", "bar2", "bar1", "bar1"],
            "baz": ["baz1", "baz1", "baz1", "baz2", "baz2"],
            "spam": ["spam2", "spam3", "spam2", "spam1", "spam1"],
            "data": [20, 30, 40, 50, 60],
        },
        index=index,
    )

    # 按照多列进行分组
    grouped = data.groupby(["foo", "bar", "baz", "spam"])
    # 对分组数据进行聚合计算，计算每个分组的均值并存储在结果中
    result = grouped.agg("mean")
    
    # 使用默认的聚合函数计算分组的均值，并将结果存储在期望值中
    expected = grouped.mean()
    
    # 使用测试框架中的函数比较两个数据帧的内容是否相等
    tm.assert_frame_equal(result, expected)
def test_groupby_series_with_name(df):
    # 对DataFrame按照"A"列进行分组，并计算每组的平均值，返回一个Series，索引为"A"
    result = df.groupby(df["A"]).mean(numeric_only=True)
    # 对DataFrame按照"A"列进行分组，并计算每组的平均值，返回一个DataFrame，"A"列不作为索引
    result2 = df.groupby(df["A"], as_index=False).mean(numeric_only=True)
    # 断言索引的名称为"A"
    assert result.index.name == "A"
    # 断言结果中包含"A"列
    assert "A" in result2

    # 对DataFrame按照"A"和"B"列进行分组，并计算每组的平均值，返回一个MultiIndex DataFrame
    result = df.groupby([df["A"], df["B"]]).mean()
    # 对DataFrame按照"A"和"B"列进行分组，并计算每组的平均值，返回一个DataFrame，"A"和"B"列不作为索引
    result2 = df.groupby([df["A"], df["B"]], as_index=False).mean()
    # 断言索引的名称为("A", "B")
    assert result.index.names == ("A", "B")
    # 断言结果中包含"A"列和"B"列
    assert "A" in result2
    assert "B" in result2


def test_seriesgroupby_name_attr(df):
    # GH 6265
    # 对DataFrame按照"A"列进行分组，然后选择"C"列，返回一个GroupBy对象
    result = df.groupby("A")["C"]
    # 断言GroupBy对象中的计数结果的名称为"C"
    assert result.count().name == "C"
    # 断言GroupBy对象中的均值结果的名称为"C"
    assert result.mean().name == "C"

    # 定义一个lambda函数用于对GroupBy对象进行聚合操作
    testFunc = lambda x: np.sum(x) * 2
    # 断言对GroupBy对象应用聚合函数后的结果的名称为"C"
    assert result.agg(testFunc).name == "C"


def test_consistency_name():
    # GH 12363

    # 创建一个DataFrame对象，包含列"A", "B", "C", "D"，其中"C"列使用随机数生成数据
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": np.random.default_rng(2).standard_normal(8) + 1.0,
            "D": np.arange(8),
        }
    )

    # 期望的结果：对"A"列进行分组，计算"B"列的计数
    expected = df.groupby(["A"]).B.count()
    # 实际结果：对"A"列进行分组，计算"B"列的计数
    result = df.B.groupby(df.A).count()
    # 断言实际结果与期望结果相等
    tm.assert_series_equal(result, expected)


def test_groupby_name_propagation(df):
    # GH 6124
    # 定义一个函数summarize，对输入的DataFrame进行汇总操作，返回一个Series对象
    def summarize(df, name=None):
        return Series({"count": 1, "mean": 2, "omissions": 3}, name=name)

    # 定义一个函数summarize_random_name，根据DataFrame中第一行的"A"列的值，为每个Series提供不同的名称
    def summarize_random_name(df):
        # 在这种情况下，由于名称不一致，groupby不应尝试传播Series的名称。
        return Series({"count": 1, "mean": 2, "omissions": 3}, name=df.iloc[0]["A"])

    # 进行警告断言：DataFrameGroupBy.apply操作在分组列上操作
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对DataFrame按照"A"列进行分组，然后应用summarize函数，返回一个DataFrame
        metrics = df.groupby("A").apply(summarize)
    # 断言metrics的列名为None
    assert metrics.columns.name is None
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对DataFrame按照"A"列进行分组，然后应用summarize函数，指定列名为"metrics"，返回一个DataFrame
        metrics = df.groupby("A").apply(summarize, "metrics")
    # 断言metrics的列名为"metrics"
    assert metrics.columns.name == "metrics"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对DataFrame按照"A"列进行分组，然后应用summarize_random_name函数，返回一个DataFrame
        metrics = df.groupby("A").apply(summarize_random_name)
    # 断言metrics的列名为None
    assert metrics.columns.name is None


def test_groupby_nonstring_columns():
    # 创建一个DataFrame对象，包含10行10列的数据，每列都是0到9的整数
    df = DataFrame([np.arange(10) for x in range(10)])
    # 对DataFrame按照第一列(索引为0)进行分组，计算每组的均值，返回一个DataFrame
    grouped = df.groupby(0)
    # 计算分组后的均值，返回一个DataFrame
    result = grouped.mean()
    # 期望的结果：对DataFrame按照第一列进行分组，计算每组的均值，返回一个DataFrame
    expected = df.groupby(df[0]).mean()
    # 断言实际结果与期望结果相等
    tm.assert_frame_equal(result, expected)


def test_groupby_mixed_type_columns():
    # GH 13432, py3中无法比较的类型
    # 创建一个DataFrame对象，包含1行2列的数据，列名分别为"A", "B", 0
    df = DataFrame([[0, 1, 2]], columns=["A", "B", 0])
    # 期望的结果：对"A"列进行分组，计算每组的第一行，返回一个DataFrame，列名为"B"和0，索引名为"A"
    expected = DataFrame([[1, 2]], columns=["B", 0], index=Index([0], name="A"))

    # 对DataFrame按照"A"列进行分组，计算每组的第一行，返回一个DataFrame
    result = df.groupby("A").first()
    # 断言实际结果与期望结果相等
    tm.assert_frame_equal(result, expected)

    # 对DataFrame按照"A"列进行分组，计算每组的总和，返回一个DataFrame
    result = df.groupby("A").sum()
    # 断言实际结果与期望结果相等
    tm.assert_frame_equal(result, expected)


def test_cython_grouper_series_bug_noncontig():
    # 创建一个形状为(100, 100)的NaN数组
    arr = np.empty((100, 100))
    arr.fill(np.nan)
    # 创建一个Series对象，包含数组的第一列
    obj = Series(arr[:, 0])
    # 使用0到9的整数重复10次，作为索引
    inds = np.tile(range(10), 10)
    # 根据给定的索引inds对对象obj进行分组，然后计算每组的中位数，并存储在result中
    result = obj.groupby(inds).agg(Series.median)
    # 断言：检查result中的所有值是否为缺失值（NaN），如果是，则断言失败
    assert result.isna().all()
def test_series_grouper_noncontig_index():
    # 创建一个包含100个重复元素的索引，每个元素为字符串"a"乘以10
    index = Index(["a" * 10] * 100)

    # 使用随机数生成器创建一个Series对象，长度为50，数据为标准正态分布随机数，
    # 索引为index的每隔一个元素（步长为2）取一个
    values = Series(np.random.default_rng(2).standard_normal(50), index=index[::2])

    # 使用随机数生成器创建一个长度为50的整数数组，元素范围在0到4之间
    labels = np.random.default_rng(2).integers(0, 5, 50)

    # 将Series对象按照labels进行分组
    grouped = values.groupby(labels)

    # 定义一个lambda函数f，用于计算分组后每个组的索引元素的唯一性数量
    f = lambda x: len(set(map(id, x.index)))

    # 对分组后的对象应用函数f，即计算每个分组的索引元素的唯一性数量
    grouped.agg(f)


def test_convert_objects_leave_decimal_alone():
    # 创建一个长度为5的Series对象，数据为0到4的整数
    s = Series(range(5))
    
    # 创建一个包含5个元素的字符串数组
    labels = np.array(["a", "b", "c", "d", "e"], dtype="O")

    # 定义一个转换函数convert_fast，计算Series对象均值并转换为Decimal类型
    def convert_fast(x):
        return Decimal(str(x.mean()))

    # 定义一个强制纯净转换函数convert_force_pure，确保输入的数据不共享基础数据
    def convert_force_pure(x):
        # 断言数据的基础部分长度大于0
        assert len(x.values.base) > 0
        return Decimal(str(x.mean()))

    # 将Series对象按照labels进行分组
    grouped = s.groupby(labels)

    # 对分组后的对象应用convert_fast函数，并进行断言检查结果类型和数据类型
    result = grouped.agg(convert_fast)
    assert result.dtype == np.object_
    assert isinstance(result.iloc[0], Decimal)

    # 对分组后的对象应用convert_force_pure函数，并进行断言检查结果类型和数据类型
    result = grouped.agg(convert_force_pure)
    assert result.dtype == np.object_
    assert isinstance(result.iloc[0], Decimal)


def test_groupby_dtype_inference_empty():
    # 创建一个空的DataFrame对象，包含两列：一列空数据列"x"，一列空的int64范围数组列"range"
    df = DataFrame({"x": [], "range": np.arange(0, dtype="int64")})
    
    # 断言DataFrame对象的"x"列数据类型为np.float64
    assert df["x"].dtype == np.float64

    # 对DataFrame对象按照"x"列进行分组，并取每组的第一个值
    result = df.groupby("x").first()
    
    # 创建预期的DataFrame对象，包含一个空的Series对象，索引为浮点数0，数据类型为int64
    exp_index = Index([], name="x", dtype=np.float64)
    expected = DataFrame({"range": Series([], index=exp_index, dtype="int64")})
    
    # 使用断言比较result和expected是否相等，按块比较
    tm.assert_frame_equal(result, expected, by_blocks=True)


def test_groupby_unit64_float_conversion():
    # 创建一个DataFrame对象，包含三列数据："first"、"second"、"value"
    df = DataFrame({"first": [1], "second": [1], "value": [16148277970000000000]})
    
    # 对DataFrame对象按照["first", "second"]两列进行分组，并取"value"列的最大值
    result = df.groupby(["first", "second"])["value"].max()
    
    # 创建预期的Series对象，包含一个元素列表，数据类型为value列的MultiIndex对象
    expected = Series(
        [16148277970000000000],
        MultiIndex.from_product([[1], [1]], names=["first", "second"]),
        name="value",
    )
    
    # 使用断言比较result和expected是否相等
    tm.assert_series_equal(result, expected)


def test_groupby_list_infer_array_like(df):
    # 对DataFrame对象按照df["A"]列的列表形式进行分组，并计算每组的均值，只保留数值列
    result = df.groupby(list(df["A"])).mean(numeric_only=True)
    
    # 预期结果：按照df["A"]列进行分组，计算每组的均值，只保留数值列
    expected = df.groupby(df["A"]).mean(numeric_only=True)
    
    # 使用断言比较result和expected是否相等，忽略名称检查
    tm.assert_frame_equal(result, expected, check_names=False)

    # 使用pytest断言，期望捕获KeyError并匹配字符串'foo'
    with pytest.raises(KeyError, match=r"^'foo'$"):
        df.groupby(list(df["A"][:-1]))

    # 创建一个DataFrame对象，包含三列数据："foo"、"bar"、"val"
    # "val"列为标准正态分布随机数
    df = DataFrame(
        {
            "foo": [0, 1],
            "bar": [3, 4],
            "val": np.random.default_rng(2).standard_normal(2),
        }
    )

    # 对DataFrame对象按照["foo", "bar"]两列进行分组，并计算每组的均值
    result = df.groupby(["foo", "bar"]).mean()
    
    # 预期结果：按照[df["foo"], df["bar"]]形式进行分组，计算每组的均值，只保留"val"列
    expected = df.groupby([df["foo"], df["bar"]]).mean()[["val"]]


def test_groupby_keys_same_size_as_index():
    # 设置频率为秒级的时间索引，从"2015-09-29T11:34:44-0700"开始，长度为2
    index = date_range(
        start=Timestamp("2015-09-29T11:34:44-0700"), periods=2, freq="s"
    )
    
    # 创建一个DataFrame对象，包含两行两列数据：["metric", "values"]
    df = DataFrame([["A", 10], ["B", 15]], columns=["metric", "values"], index=index)
    
    # 对DataFrame对象按照[Grouper(level=0, freq="s"), "metric"]进行分组，并计算每组的均值
    result = df.groupby([Grouper(level=0, freq="s"), "metric"]).mean()
    
    # 创建预期的DataFrame对象，设置索引为[df.index, "metric"]，数据类型转换为float
    expected = df.set_index([df.index, "metric"]).astype(float)
    
    # 使用断言比较result和expected是否相等
    tm.assert_frame_equal(result, expected)


def test_groupby_one_row():
    # GH 11741
    # 定义一个正则表达式字符串，用于匹配预期的异常消息
    msg = r"^'Z'$"
    # 创建一个包含随机数据的 DataFrame 对象 df1，共 1 行 4 列，列名为 A, B, C, D
    df1 = DataFrame(
        np.random.default_rng(2).standard_normal((1, 4)), columns=list("ABCD")
    )
    # 使用 pytest 框架的 raises 函数检查是否引发 KeyError 异常，并匹配预期的异常消息
    with pytest.raises(KeyError, match=msg):
        # 尝试对 df1 根据 "Z" 列进行分组操作
        df1.groupby("Z")
    
    # 创建另一个包含随机数据的 DataFrame 对象 df2，共 2 行 4 列，列名为 A, B, C, D
    df2 = DataFrame(
        np.random.default_rng(2).standard_normal((2, 4)), columns=list("ABCD")
    )
    # 使用 pytest 框架的 raises 函数检查是否引发 KeyError 异常，并匹配预期的异常消息
    with pytest.raises(KeyError, match=msg):
        # 尝试对 df2 根据 "Z" 列进行分组操作
        df2.groupby("Z")
# 定义一个名为 test_groupby_nat_exclude 的测试函数
def test_groupby_nat_exclude():
    # 创建一个 DataFrame 对象 df，包含三列：values、dt、str，其中包括 NaN 和 Timestamp 类型的数据
    df = DataFrame(
        {
            "values": np.random.default_rng(2).standard_normal(8),
            "dt": [
                np.nan,
                Timestamp("2013-01-01"),
                np.nan,
                Timestamp("2013-02-01"),
                np.nan,
                Timestamp("2013-02-01"),
                np.nan,
                Timestamp("2013-01-01"),
            ],
            "str": [np.nan, "a", np.nan, "a", np.nan, "a", np.nan, "b"],
        }
    )
    # 对 df 按照 "dt" 列进行分组
    grouped = df.groupby("dt")

    # 预期的分组结果列表
    expected = [Index([1, 7]), Index([3, 5])]
    # 获取分组的键，并排序
    keys = sorted(grouped.groups.keys())
    # 断言分组的键的数量为 2
    assert len(keys) == 2
    # 遍历每个键和对应的预期结果
    for k, e in zip(keys, expected):
        # 对比每个分组的索引，确保匹配预期结果 e
        # grouped.groups 的键是 np.datetime64 类型，带有系统时区信息，这里只比较值而不考虑时区
        tm.assert_index_equal(grouped.groups[k], e)

    # 确认分组对象不被过滤
    tm.assert_frame_equal(grouped._grouper.groupings[0].obj, df)
    # 断言分组的数量为 2
    assert grouped.ngroups == 2

    # 预期的索引映射字典
    expected = {
        Timestamp("2013-01-01 00:00:00"): np.array([1, 7], dtype=np.intp),
        Timestamp("2013-02-01 00:00:00"): np.array([3, 5], dtype=np.intp),
    }

    # 遍历分组的索引
    for k in grouped.indices:
        # 断言每个分组的索引数组与预期的数组相等
        tm.assert_numpy_array_equal(grouped.indices[k], expected[k])

    # 断言获取分组 Timestamp("2013-01-01") 的数据框与 df 中相应行匹配
    tm.assert_frame_equal(grouped.get_group(Timestamp("2013-01-01")), df.iloc[[1, 7]])
    # 断言获取分组 Timestamp("2013-02-01") 的数据框与 df 中相应行匹配
    tm.assert_frame_equal(grouped.get_group(Timestamp("2013-02-01")), df.iloc[[3, 5]])

    # 使用 pytest 断言应该抛出 KeyError，匹配正则表达式 "^NaT$"
    with pytest.raises(KeyError, match=r"^NaT$"):
        grouped.get_group(pd.NaT)

    # 创建一个包含 NaN 和 pd.NaT 的 DataFrame 对象 nan_df
    nan_df = DataFrame(
        {"nan": [np.nan, np.nan, np.nan], "nat": [pd.NaT, pd.NaT, pd.NaT]}
    )
    # 断言列 "nan" 的数据类型为 "float64"
    assert nan_df["nan"].dtype == "float64"
    # 断言列 "nat" 的数据类型为 "datetime64[s]"
    assert nan_df["nat"].dtype == "datetime64[s]"

    # 遍历键 ["nan", "nat"]
    for key in ["nan", "nat"]:
        # 对 nan_df 按照键进行分组
        grouped = nan_df.groupby(key)
        # 断言分组的 groups 属性为空字典
        assert grouped.groups == {}
        # 断言分组的数量为 0
        assert grouped.ngroups == 0
        # 断言分组的 indices 属性为空字典
        assert grouped.indices == {}
        # 使用 pytest 断言应该抛出 KeyError，匹配正则表达式 "^nan$"
        with pytest.raises(KeyError, match=r"^nan$"):
            grouped.get_group(np.nan)
        # 使用 pytest 断言应该抛出 KeyError，匹配正则表达式 "^NaT$"
        with pytest.raises(KeyError, match=r"^NaT$"):
            grouped.get_group(pd.NaT)
    # 创建一个 DataFrame 对象，包含五列数据：A, B, C, D 和 E
    df = DataFrame(
        {
            "A": A,  # 使用变量 A 的值作为列 "A" 的数据
            "B": B,  # 使用变量 B 的值作为列 "B" 的数据
            "C": A,  # 使用变量 A 的值再次作为列 "C" 的数据
            "D": B,  # 使用变量 B 的值再次作为列 "D" 的数据
            "E": np.random.default_rng(2).standard_normal(25000),  # 生成一个 25000 个随机数的标准正态分布数据列 "E"
        }
    )
    
    # 根据列 "A", "B", "C", "D" 对 DataFrame 进行分组，并对分组后的组内数据进行求和操作
    left = df.groupby(["A", "B", "C", "D"]).sum()
    
    # 根据列 "D", "C", "B", "A" 对 DataFrame 进行分组，并对分组后的组内数据进行求和操作
    right = df.groupby(["D", "C", "B", "A"]).sum()
    
    # 检查左右两个分组的长度是否相等，如果不相等会触发 AssertionError
    assert len(left) == len(right)
def test_groupby_sort_multi():
    df = DataFrame(
        {
            "a": ["foo", "bar", "baz"],
            "b": [3, 2, 1],
            "c": [0, 1, 2],
            "d": np.random.default_rng(2).standard_normal(3),
        }
    )

    # 将 DataFrame 中的列"a", "b", "c"转换为元组列表
    tups = [tuple(row) for row in df[["a", "b", "c"]].values]
    tups = com.asarray_tuplesafe(tups)
    # 按照列"a", "b", "c"对 DataFrame 进行分组求和，排序结果
    result = df.groupby(["a", "b", "c"], sort=True).sum()
    # 断言分组后的索引值与转换后的元组列表相等
    tm.assert_numpy_array_equal(result.index.values, tups[[1, 2, 0]])

    # 将 DataFrame 中的列"c", "a", "b"转换为元组列表
    tups = [tuple(row) for row in df[["c", "a", "b"]].values]
    tups = com.asarray_tuplesafe(tups)
    # 按照列"c", "a", "b"对 DataFrame 进行分组求和，排序结果
    result = df.groupby(["c", "a", "b"], sort=True).sum()
    # 断言分组后的索引值与转换后的元组列表相等
    tm.assert_numpy_array_equal(result.index.values, tups)

    # 将 DataFrame 中的列"b", "c", "a"转换为元组列表
    tups = [tuple(x) for x in df[["b", "c", "a"]].values]
    tups = com.asarray_tuplesafe(tups)
    # 按照列"b", "c", "a"对 DataFrame 进行分组求和，排序结果
    result = df.groupby(["b", "c", "a"], sort=True).sum()
    # 断言分组后的索引值与转换后的元组列表相等
    tm.assert_numpy_array_equal(result.index.values, tups[[2, 1, 0]])

    df = DataFrame(
        {
            "a": [0, 1, 2, 0, 1, 2],
            "b": [0, 0, 0, 1, 1, 1],
            "d": np.random.default_rng(2).standard_normal(6),
        }
    )
    # 按照列"a", "b"对 DataFrame 进行分组，选择"d"列并求和
    grouped = df.groupby(["a", "b"])["d"]
    result = grouped.sum()

    def _check_groupby(df, result, keys, field, f=lambda x: x.sum()):
        # 将 DataFrame 指定列转换为元组列表
        tups = [tuple(row) for row in df[keys].values]
        tups = com.asarray_tuplesafe(tups)
        # 使用转换后的元组列表对 DataFrame 进行分组并应用函数 f
        expected = f(df.groupby(tups)[field])
        # 断言结果字典中每个键对应的值与期望值相等
        for k, v in expected.items():
            assert result[k] == v

    # 验证 _check_groupby 函数的功能
    _check_groupby(df, result, ["a", "b"], "d")


def test_dont_clobber_name_column():
    df = DataFrame(
        {"key": ["a", "a", "a", "b", "b", "b"], "name": ["foo", "bar", "baz"] * 2}
    )

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 断言在调用 apply 函数时会产生 DeprecationWarning 警告消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby("key", group_keys=False).apply(lambda x: x)
    # 断言结果 DataFrame 与原始 DataFrame 相等
    tm.assert_frame_equal(result, df)


def test_skip_group_keys():
    tsf = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )

    # 按照月份对 DataFrame 进行分组，不包含组键信息
    grouped = tsf.groupby(lambda x: x.month, group_keys=False)
    # 对每个分组按照"A"列排序，并选择前三行
    result = grouped.apply(lambda x: x.sort_values(by="A")[:3])

    # 生成每个分组按照"A"列排序后选择前三行的 DataFrame 列表
    pieces = [group.sort_values(by="A")[:3] for key, group in grouped]

    # 将所有 pieces 列表中的 DataFrame 拼接成一个 DataFrame
    expected = pd.concat(pieces)
    # 断言结果 DataFrame 与期望的 DataFrame 相等
    tm.assert_frame_equal(result, expected)

    # 对"A"列按照月份进行分组，不包含组键信息
    grouped = tsf["A"].groupby(lambda x: x.month, group_keys=False)
    # 对每个分组按照"A"列排序，并选择前三行
    result = grouped.apply(lambda x: x.sort_values()[:3])

    # 生成每个分组按照"A"列排序后选择前三行的 Series 列表
    pieces = [group.sort_values()[:3] for key, group in grouped]

    # 将所有 pieces 列表中的 Series 拼接成一个 Series
    expected = pd.concat(pieces)
    # 断言结果 Series 与期望的 Series 相等
    tm.assert_series_equal(result, expected)


def test_no_nonsense_name(float_frame):
    # GH #995
    s = float_frame["C"].copy()
    s.name = None

    # 按照 float_frame["A"] 列进行分组并聚合求和
    result = s.groupby(float_frame["A"]).agg("sum")
    # 断言结果 Series 的名称为 None
    assert result.name is None


def test_multifunc_sum_bug():
    # GH #1065
    x = DataFrame(np.arange(9).reshape(3, 3))
    x["test"] = 0
    # 将列表 [1.3, 1.5, 1.6] 赋给字典 x 中键名为 "fl" 的项
    x["fl"] = [1.3, 1.5, 1.6]
    
    # 根据 DataFrame x 中的 "test" 列进行分组
    grouped = x.groupby("test")
    
    # 对分组后的数据进行聚合操作：
    # - 对 "fl" 列求和（sum）
    # - 对列名为 2 的列求大小（size，即统计非空元素的个数）
    result = grouped.agg({"fl": "sum", 2: "size"})
    
    # 断言聚合结果中 "fl" 列的数据类型为 np.float64
    assert result["fl"].dtype == np.float64
def test_handle_dict_return_value(df):
    # 定义函数 f，接收一个分组，返回包含最大值和最小值的字典
    def f(group):
        return {"max": group.max(), "min": group.min()}

    # 定义函数 g，接收一个分组，返回包含最大值和最小值的 Series 对象
    def g(group):
        return Series({"max": group.max(), "min": group.min()})

    # 对 DataFrame 进行分组，对每个分组的 "C" 列应用函数 f
    result = df.groupby("A")["C"].apply(f)
    # 对 DataFrame 进行分组，对每个分组的 "C" 列应用函数 g
    expected = df.groupby("A")["C"].apply(g)

    # 断言 result 是一个 Series 对象
    assert isinstance(result, Series)
    # 使用测试工具函数比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("grouper", ["A", ["A", "B"]])
def test_set_group_name(df, grouper, using_infer_string):
    # 定义函数 f，接收一个分组，断言其名称不为空，并返回该分组
    def f(group):
        assert group.name is not None
        return group

    # 定义函数 freduce，接收一个分组，断言其名称不为空，如果符合特定条件，则引发异常，否则返回分组求和的结果
    def freduce(group):
        assert group.name is not None
        if using_infer_string and grouper == "A" and is_string_dtype(group.dtype):
            with pytest.raises(TypeError, match="does not support"):
                group.sum()
        else:
            return group.sum()

    # 定义函数 freducex，接收一个参数 x，调用函数 freduce 处理该参数
    def freducex(x):
        return freduce(x)

    # 对 DataFrame 进行 grouper 列的分组，不使用键作为索引
    grouped = df.groupby(grouper, group_keys=False)

    # 确保以下所有操作正常运行
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用测试工具函数，确保产生警告信息，并匹配指定的消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        grouped.apply(f)
    # 对分组应用函数 freduce
    grouped.aggregate(freduce)
    # 对分组应用指定列的函数 freduce
    grouped.aggregate({"C": freduce, "D": freduce})
    # 对分组应用函数 f 进行变换
    grouped.transform(f)

    # 对分组后的 "C" 列应用函数 f
    grouped["C"].apply(f)
    # 对分组后的 "C" 列应用函数 freduce
    grouped["C"].aggregate(freduce)
    # 对分组后的 "C" 列应用函数列表 [freduce, freducex]
    grouped["C"].aggregate([freduce, freducex])
    # 对分组后的 "C" 列应用函数 f 进行变换
    grouped["C"].transform(f)


def test_group_name_available_in_inference_pass():
    # 测试用例 gh-15062
    # 创建 DataFrame，包含列 "a" 和 "b"
    df = DataFrame({"a": [0, 0, 1, 1, 2, 2], "b": np.arange(6)})

    # 定义空列表 names，用于存储分组的名称
    names = []

    # 定义函数 f，接收一个分组，将其名称添加到 names 列表中，然后返回分组的副本
    def f(group):
        names.append(group.name)
        return group.copy()

    # 确保以下操作正常运行
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    # 使用测试工具函数，确保产生警告信息，并匹配指定的消息
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 对 DataFrame 按列 "a" 进行分组，不排序，不使用键作为索引，对每个分组应用函数 f
        df.groupby("a", sort=False, group_keys=False).apply(f)

    # 期望的分组名称列表
    expected_names = [0, 1, 2]
    # 断言 names 列表与期望的分组名称列表相等
    assert names == expected_names


def test_no_dummy_key_names(df):
    # 测试用例 gh-1291
    # 对 DataFrame 按列 "A" 的值进行分组，并求和
    result = df.groupby(df["A"].values).sum()
    # 断言结果的索引名称为 None
    assert result.index.name is None

    # 对 DataFrame 按列 "A" 和 "B" 的值进行分组，并求和
    result = df.groupby([df["A"].values, df["B"].values]).sum()
    # 断言结果的索引名称元组为 (None, None)
    assert result.index.names == (None, None)


def test_groupby_sort_multiindex_series():
    # 测试用例，解决 series 多级索引的 groupby 排序参数未传递的问题
    # GH 9444
    # 创建 MultiIndex
    index = MultiIndex(
        levels=[[1, 2], [1, 2]],
        codes=[[0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0]],
        names=["a", "b"],
    )
    # 创建 Series
    mseries = Series([0, 1, 2, 3, 4, 5], index=index)
    # 创建期望的结果 Series
    index = MultiIndex(
        levels=[[1, 2], [1, 2]], codes=[[0, 0, 1], [1, 0, 0]], names=["a", "b"]
    )
    mseries_result = Series([0, 2, 4], index=index)

    # 对 Series 按指定的多级索引进行分组，不排序，获取每组的第一个元素
    result = mseries.groupby(level=["a", "b"], sort=False).first()
    # 使用测试工具函数，比较结果和期望的结果 Series 是否相等
    tm.assert_series_equal(result, mseries_result)
    # 对 Series 按指定的多级索引进行分组，排序，获取每组的第一个元素
    result = mseries.groupby(level=["a", "b"], sort=True).first()
    # 使用测试工具函数，比较结果和期望的结果 Series 是否相等
    tm.assert_series_equal(result, mseries_result.sort_index())


def test_groupby_reindex_inside_function():
    periods = 1000
    # 创建时间范围索引，从"2012/1/1"开始，以5分钟为频率，生成指定周期数的时间索引
    ind = date_range(start="2012/1/1", freq="5min", periods=periods)
    
    # 创建 DataFrame 对象，包含"high"和"low"两列，数据分别为0到periods-1，使用上述时间索引作为行索引
    df = DataFrame({"high": np.arange(periods), "low": np.arange(periods)}, index=ind)

    # 定义一个函数 agg_before，用于对数据子集进行聚合操作
    def agg_before(func, fix=False):
        """
        运行聚合函数在数据子集上。
        """
        # 定义内部函数 _func，接受一个数据参数
        def _func(data):
            # 选择出索引小时部分小于11的数据行，然后删除空值
            d = data.loc[data.index.map(lambda x: x.hour < 11)].dropna()
            # 如果 fix 参数为真，则尝试访问数据的第一个元素
            if fix:
                data[data.index[0]]
            # 如果选择的数据行为空，返回 None
            if len(d) == 0:
                return None
            # 否则，对选择的数据行应用给定的函数 func，并返回结果
            return func(d)

        return _func

    # 按照日期创建分组对象 grouped，以年月日为分组依据
    grouped = df.groupby(lambda x: datetime(x.year, x.month, x.day))
    
    # 使用 agg_before 函数对分组后的数据进行聚合，应用 np.max 函数
    closure_bad = grouped.agg({"high": agg_before(np.max)})
    
    # 使用带有 fix 参数为真的 agg_before 函数对分组后的数据进行聚合，应用 np.max 函数
    closure_good = grouped.agg({"high": agg_before(np.max, True)})
    
    # 使用测试工具函数检查 closure_bad 和 closure_good 是否相等
    tm.assert_frame_equal(closure_bad, closure_good)
def test_groupby_multiindex_missing_pair():
    # GH9049
    # 创建一个包含 group1、group2 和 value 列的 DataFrame
    df = DataFrame(
        {
            "group1": ["a", "a", "a", "b"],
            "group2": ["c", "c", "d", "c"],
            "value": [1, 1, 1, 5],
        }
    )
    # 将 group1 和 group2 列设为索引
    df = df.set_index(["group1", "group2"])
    # 根据多级索引 group1 和 group2 进行分组
    df_grouped = df.groupby(level=["group1", "group2"], sort=True)

    # 对分组后的 DataFrame 进行求和聚合
    res = df_grouped.agg("sum")
    # 创建预期的 DataFrame，包含指定的多级索引和聚合后的值
    idx = MultiIndex.from_tuples(
        [("a", "c"), ("a", "d"), ("b", "c")], names=["group1", "group2"]
    )
    exp = DataFrame([[2], [1], [5]], index=idx, columns=["value"])

    # 使用测试框架检查 res 是否等于 exp
    tm.assert_frame_equal(res, exp)


def test_groupby_multiindex_not_lexsorted(performance_warning):
    # GH 11640

    # 定义按字典序排序的多级索引版本
    lexsorted_mi = MultiIndex.from_tuples(
        [("a", ""), ("b1", "c1"), ("b2", "c2")], names=["b", "c"]
    )
    lexsorted_df = DataFrame([[1, 3, 4]], columns=lexsorted_mi)
    assert lexsorted_df.columns._is_lexsorted()

    # 定义未按字典序排序的多级索引版本
    not_lexsorted_df = DataFrame(
        columns=["a", "b", "c", "d"], data=[[1, "b1", "c1", 3], [1, "b2", "c2", 4]]
    )
    # 对数据进行透视表处理，使用 a 作为索引，(b, c) 作为列，d 作为值
    not_lexsorted_df = not_lexsorted_df.pivot_table(
        index="a", columns=["b", "c"], values="d"
    )
    # 重置索引，确保列未按字典序排序
    not_lexsorted_df = not_lexsorted_df.reset_index()
    assert not not_lexsorted_df.columns._is_lexsorted()

    # 创建预期的 DataFrame，对 lexsorted_df 按 a 列进行分组求均值
    expected = lexsorted_df.groupby("a").mean()
    # 使用性能警告检查框架，检查 result 是否与预期相等
    with tm.assert_produces_warning(performance_warning):
        result = not_lexsorted_df.groupby("a").mean()
    tm.assert_frame_equal(expected, result)

    # 对于不同的层级和排序方式，测试一个转换函数是否能正常工作
    # GH 14776
    # 创建一个包含 x、y 和 z 列的 DataFrame，将 (x, y) 设置为索引
    df = DataFrame(
        {"x": ["a", "a", "b", "a"], "y": [1, 1, 2, 2], "z": [1, 2, 3, 4]}
    ).set_index(["x", "y"])
    # 检查索引是否未按字典序排序
    assert not df.index._is_lexsorted()

    # 针对不同的层级和排序方式，应用去重函数，并使用测试框架检查结果是否与预期一致
    for level in [0, 1, [0, 1]]:
        for sort in [False, True]:
            result = df.groupby(level=level, sort=sort, group_keys=False).apply(
                DataFrame.drop_duplicates
            )
            expected = df
            tm.assert_frame_equal(expected, result)

            # 对排序后的 DataFrame 进行相同的操作，再次使用测试框架检查结果是否与预期一致
            result = (
                df.sort_index()
                .groupby(level=level, sort=sort, group_keys=False)
                .apply(DataFrame.drop_duplicates)
            )
            expected = df.sort_index()
            tm.assert_frame_equal(expected, result)


def test_index_label_overlaps_location():
    # 检查在 GH5375 问题修复后，标签和位置索引没有混淆
    # 创建一个包含列表 "ABCDE" 的 DataFrame，使用 [2, 0, 2, 1, 1] 作为索引
    df = DataFrame(list("ABCDE"), index=[2, 0, 2, 1, 1])
    # 根据列表 "ababb" 进行分组
    g = df.groupby(list("ababb"))
    # 使用过滤函数筛选出长度大于 2 的分组
    actual = g.filter(lambda x: len(x) > 2)
    # 创建预期的 DataFrame，包含根据位置索引选择的行
    expected = df.iloc[[1, 3, 4]]
    # 使用测试框架检查 actual 是否等于 expected
    tm.assert_frame_equal(actual, expected)

    # 创建一个 Series，使用 df 的第一列作为数据
    ser = df[0]
    # 根据列表 "ababb" 对 Series 进行分组
    g = ser.groupby(list("ababb"))
    # 使用过滤函数筛选出长度大于 2 的分组
    actual = g.filter(lambda x: len(x) > 2)
    # 创建预期的 Series，包含根据位置索引选择的元素
    expected = ser.take([1, 3, 4])
    # 使用测试框架检查 actual 是否等于 expected
    tm.assert_series_equal(actual, expected)

    # 将 df 的索引转换为浮点数类型，确保不会因为排序问题引发混淆
    df.index = df.index.astype(float)
    # 根据列表 "ababb" 对 DataFrame 进行分组
    g = df.groupby(list("ababb"))
    # 使用 Pandas 中的 `filter` 方法筛选长度大于 2 的元素，并将结果赋给 `actual`
    actual = g.filter(lambda x: len(x) > 2)
    
    # 从 DataFrame `df` 中选择索引为 1、3、4 的行，并将结果赋给 `expected`
    expected = df.iloc[[1, 3, 4]]
    
    # 使用 Pandas 测试模块 (`tm`) 比较 `actual` 和 `expected` 是否相等
    tm.assert_frame_equal(actual, expected)
    
    # 从 DataFrame `df` 的第 0 列中选择数据，并将结果赋给 `ser`
    ser = df[0]
    
    # 根据列表 "ababb" 对 `ser` 进行分组，并将结果赋给 `g`
    g = ser.groupby(list("ababb"))
    
    # 使用 Pandas 中的 `filter` 方法筛选长度大于 2 的组，并将结果赋给 `actual`
    actual = g.filter(lambda x: len(x) > 2)
    
    # 从 `ser` 中选择索引为 1、3、4 的元素，并将结果赋给 `expected`
    expected = ser.take([1, 3, 4])
    
    # 使用 Pandas 测试模块 (`tm`) 比较 `actual` 和 `expected` 是否相等
    tm.assert_series_equal(actual, expected)
def test_transform_doesnt_clobber_ints():
    # GH 7972
    # 设置整数 n 为 6
    n = 6
    # 创建一个包含 n 个元素的整数数组 x
    x = np.arange(n)
    # 创建 DataFrame df，包含三列：a 为 x 除以 2 的整数部分，b 为 2.0 乘以 x，c 为 3.0 乘以 x
    df = DataFrame({"a": x // 2, "b": 2.0 * x, "c": 3.0 * x})
    # 创建 DataFrame df2，与 df 相似，但 a 列转换为浮点数
    df2 = DataFrame({"a": x // 2 * 1.0, "b": 2.0 * x, "c": 3.0 * x})

    # 对 df 按列 a 进行分组
    gb = df.groupby("a")
    # 对分组结果应用 transform 函数，计算每个分组的均值
    result = gb.transform("mean")

    # 对 df2 按列 a 进行分组
    gb2 = df2.groupby("a")
    # 对分组结果应用 transform 函数，计算每个分组的均值
    expected = gb2.transform("mean")
    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sort_column",
    ["ints", "floats", "strings", ["ints", "floats"], ["ints", "strings"]],
)
@pytest.mark.parametrize(
    "group_column", ["int_groups", "string_groups", ["int_groups", "string_groups"]]
)
def test_groupby_preserves_sort(sort_column, group_column):
    # 确保 groupby 操作能够保留原始对象的排序顺序
    # 涉及问题 #8588 和 #9651

    # 创建 DataFrame df 包含多列数据
    df = DataFrame(
        {
            "int_groups": [3, 1, 0, 1, 0, 3, 3, 3],
            "string_groups": ["z", "a", "z", "a", "a", "g", "g", "g"],
            "ints": [8, 7, 4, 5, 2, 9, 1, 1],
            "floats": [2.3, 5.3, 6.2, -2.4, 2.2, 1.1, 1.1, 5],
            "strings": ["z", "d", "a", "e", "word", "word2", "42", "47"],
        }
    )

    # 根据 sort_column 对 DataFrame 进行排序
    df = df.sort_values(by=sort_column)
    # 对 DataFrame df 根据 group_column 进行分组
    g = df.groupby(group_column)

    # 定义函数 test_sort，用于检查 DataFrame 是否按 sort_column 排序
    def test_sort(x):
        tm.assert_frame_equal(x, x.sort_values(by=sort_column))

    # 运行 g.apply(test_sort) 函数并检查是否产生 DeprecationWarning
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        g.apply(test_sort)


def test_pivot_table_values_key_error():
    # 该测试复现问题 #14938 中的错误
    # 创建 DataFrame df 包含 eventDate 和 thename 两列数据
    df = DataFrame(
        {
            "eventDate": date_range(datetime.today(), periods=20, freq="ME").tolist(),
            "thename": range(20),
        }
    )

    # 从 eventDate 列创建 year 列和 month 列
    df["year"] = df.set_index("eventDate").index.year
    df["month"] = df.set_index("eventDate").index.month

    # 断言在重置索引后的 DataFrame 执行 pivot_table 操作时，会引发 KeyError 异常
    with pytest.raises(KeyError, match="'badname'"):
        df.reset_index().pivot_table(
            index="year", columns="month", values="badname", aggfunc="count"
        )


@pytest.mark.parametrize("columns", ["C", ["C"]])
@pytest.mark.parametrize("keys", [["A"], ["A", "B"]])
@pytest.mark.parametrize(
    "values",
    [
        [True],
        [0],
        [0.0],
        ["a"],
        Categorical([0]),
        [to_datetime(0)],
        date_range(0, 1, 1, tz="US/Eastern"),
        pd.period_range("2016-01-01", periods=3, freq="D"),
        pd.array([0], dtype="Int64"),
        pd.array([0], dtype="Float64"),
        pd.array([False], dtype="boolean"),
    ],
    ids=[
        "bool",
        "int",
        "float",
        "str",
        "cat",
        "dt64",
        "dt64tz",
        "period",
        "Int64",
        "Float64",
        "boolean",
    ],
)
@pytest.mark.parametrize("method", ["attr", "agg", "apply"])
@pytest.mark.parametrize(
    "op", ["idxmax", "idxmin", "min", "max", "sum", "prod", "skew"]
)
# 定义一个测试函数，用于处理空组合的情况，例如 GH8093 和 GH26411
def test_empty_groupby(columns, keys, values, method, op, dropna, using_infer_string):
    # 默认没有覆盖的数据类型
    override_dtype = None

    # 如果 values 是 BooleanArray 类型且操作为 "sum" 或 "prod"
    if isinstance(values, BooleanArray) and op in ["sum", "prod"]:
        # 期望返回 Int64 类型的结果
        override_dtype = "Int64"

    # 如果 values 的第一个元素是布尔类型且操作为 "prod" 或 "sum"
    if isinstance(values[0], bool) and op in ("prod", "sum"):
        # 布尔值的求和或乘积结果为整数类型
        override_dtype = "int64"

    # 创建一个 DataFrame，包含指定列和相同的 values 值，列名为 "A", "B", "C"
    df = DataFrame({"A": values, "B": values, "C": values}, columns=list("ABC"))

    # 如果 values 具有 'dtype' 属性
    if hasattr(values, "dtype"):
        # 检查 DataFrame 的数据类型是否与 values 的数据类型一致
        assert (df.dtypes == values.dtype).all()

    # 取 DataFrame 的前 0 行
    df = df.iloc[:0]

    # 对 DataFrame 进行分组操作，返回一个 GroupBy 对象 gb
    gb = df.groupby(keys, group_keys=False, dropna=dropna, observed=False)[columns]

    # 定义一个内部函数，根据 method 参数选择对 gb 进行操作
    def get_result(**kwargs):
        if method == "attr":
            return getattr(gb, op)(**kwargs)
        else:
            return getattr(gb, method)(op, **kwargs)

    # 定义一个内部函数，返回对于分类数据的无效预期结果
    def get_categorical_invalid_expected():
        # 如果未设置 'observed=True'，分类数据特殊处理，会有 NaN 条目
        # 对于 groupby 未观察到的组，期望结果是 'df.set_index(keys)[columns]'
        lev = Categorical([0], dtype=values.dtype)
        if len(keys) != 1:
            idx = MultiIndex.from_product([lev, lev], names=keys)
        else:
            idx = Index(lev, name=keys[0])  # 所有列都被删除，但仍然会有一行
        # 根据 using_infer_string 标志选择不同的列对象
        if using_infer_string:
            columns = Index([], dtype="string[pyarrow_numpy]")
        else:
            columns = []
        expected = DataFrame([], columns=columns, index=idx)
        return expected

    # 检查第一个数据类型是否是 PeriodDtype
    is_per = isinstance(df.dtypes.iloc[0], pd.PeriodDtype)
    # 检查第一个数据类型是否是 datetime64
    is_dt64 = df.dtypes.iloc[0].kind == "M"
    # 检查 values 是否是 Categorical 类型
    is_cat = isinstance(values, Categorical)

    # 如果 values 是非有序的分类数据，且操作为 "min", "max", "idxmin", "idxmax"
    if (
        isinstance(values, Categorical)
        and not values.ordered
        and op in ["min", "max", "idxmin", "idxmax"]
    ):
        if op in ["min", "max"]:
            # 抛出 TypeError 异常，无法对非有序的分类数据执行 min/max 操作
            msg = f"Cannot perform {op} with non-ordered Categorical"
            klass = TypeError
        else:
            # 抛出 ValueError 异常，由于未观察到的分类，无法获取空组的 {op} 操作
            msg = f"Can't get {op} of an empty group due to unobserved categories"
            klass = ValueError
        with pytest.raises(klass, match=msg):
            get_result()

        # 如果操作为 "min", "max", "idxmin", "idxmax" 且 columns 是列表类型
        if op in ["min", "max", "idxmin", "idxmax"] and isinstance(columns, list):
            # 执行数值类型的 get_result 操作
            result = get_result(numeric_only=True)
            # 获取对于分类数据的无效预期结果
            expected = get_categorical_invalid_expected()
            # 使用 TestManager 进行结果比较
            tm.assert_equal(result, expected)
        return
    if op in ["prod", "sum", "skew"]:
        # 操作需要更多条件而非仅顺序
        if is_dt64 or is_cat or is_per:
            # GH#41291：处理特定类型不支持的异常情况
            # datetime64 类型 -> prod 和 sum 操作无效
            if is_dt64:
                msg = "datetime64 类型不支持"
            elif is_per:
                msg = "Period 类型不支持"
            else:
                msg = "category 类型不支持"
            if op == "skew":
                msg = "|".join([msg, "不支持 'skew' 操作"])
            # 断言抛出 TypeError 异常，并匹配特定消息
            with pytest.raises(TypeError, match=msg):
                get_result()

            if not isinstance(columns, list):
                # 例如 SeriesGroupBy，返回
                return
            elif op == "skew":
                # TODO: 测试 numeric_only=True 的情况
                return
            else:
                # 例如 op 在 ["prod", "sum"] 中，处理 DataFrameGroupBy
                # 操作需要更多条件而非仅顺序
                # GH#41291
                # 获取结果，使用 numeric_only=True 参数
                result = get_result(numeric_only=True)

                # 当 numeric_only=True 时，这些列被丢弃，返回空 DataFrame
                expected = df.set_index(keys)[[]]
                if is_cat:
                    expected = get_categorical_invalid_expected()
                # 断言结果与期望值相等
                tm.assert_equal(result, expected)
                return

    # 获取结果
    result = get_result()
    # 根据 keys 设置索引并选择特定列
    expected = df.set_index(keys)[columns]
    if op in ["idxmax", "idxmin"]:
        # 将结果转换为与 df 索引相同的数据类型
        expected = expected.astype(df.index.dtype)
    if override_dtype is not None:
        # 如果有指定的数据类型转换，进行转换
        expected = expected.astype(override_dtype)
    if len(keys) == 1:
        # 如果 keys 长度为 1，则设置索引名称
        expected.index.name = keys[0]
    # 断言结果与期望值相等
    tm.assert_equal(result, expected)
# 测试空的 groupby 应用到非唯一列
def test_empty_groupby_apply_nonunique_columns():
    # 创建一个空的 DataFrame
    df = DataFrame(np.random.default_rng(2).standard_normal((0, 4)))
    # 将第三列转换为 int64 类型
    df[3] = df[3].astype(np.int64)
    # 更改列名为 [0, 1, 2, 0]
    df.columns = [0, 1, 2, 0]
    # 根据第一列进行分组，不包含分组键
    gb = df.groupby(df[1], group_keys=False)
    # 设置警告消息，当 DataFrameGroupBy.apply 操作于分组列时产生 DeprecationWarning
    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        # 应用 lambda 函数到分组对象
        res = gb.apply(lambda x: x)
    # 断言结果中的数据类型与原始 DataFrame 中的数据类型相同
    assert (res.dtypes == df.dtypes).all()


# 测试元组作为分组键
def test_tuple_as_grouping():
    # 创建一个 DataFrame，包含具有元组标签的列
    df = DataFrame(
        {
            ("a", "b"): [1, 1, 1, 1],
            "a": [2, 2, 2, 2],
            "b": [2, 2, 2, 2],
            "c": [1, 1, 1, 1],
        }
    )
    
    # 使用 pytest 断言检测是否会引发 KeyError，匹配 "('a', 'b')"
    with pytest.raises(KeyError, match=r"('a', 'b')"):
        df[["a", "b", "c"]].groupby(("a", "b"))

    # 对 DataFrame 进行分组并计算 'c' 列的总和
    result = df.groupby(("a", "b"))["c"].sum()
    # 预期结果是包含索引 ("a", "b") 的 Series，值为 [4]
    expected = Series([4], name="c", index=Index([1], name=("a", "b")))
    # 使用 pytest 的 tm.assert_series_equal 函数断言结果与预期值相等
    tm.assert_series_equal(result, expected)


# 测试正确的元组键错误处理
def test_tuple_correct_keyerror():
    # 创建一个具有多级索引的 DataFrame
    df = DataFrame(1, index=range(3), columns=MultiIndex.from_product([[1, 2], [3, 4]]))
    # 使用 pytest 断言检测是否会引发 KeyError，匹配 "^(7, 8)$"
    with pytest.raises(KeyError, match=r"^\(7, 8\)$"):
        df.groupby((7, 8)).mean()


# 测试 groupby 操作中的 ohlc 非首次聚合
def test_groupby_agg_ohlc_non_first():
    # 创建一个具有日期索引和列名的 DataFrame
    df = DataFrame(
        [[1], [1]],
        columns=Index(["foo"], name="mycols"),
        index=date_range("2018-01-01", periods=2, freq="D", name="dti"),
    )

    # 预期的 DataFrame 结果，包含多级列
    expected = DataFrame(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        columns=MultiIndex.from_tuples(
            (
                ("foo", "sum", "foo"),
                ("foo", "ohlc", "open"),
                ("foo", "ohlc", "high"),
                ("foo", "ohlc", "low"),
                ("foo", "ohlc", "close"),
            ),
            names=["mycols", None, None],
        ),
        index=date_range("2018-01-01", periods=2, freq="D", name="dti"),
    )

    # 使用 Grouper 对象按日对 DataFrame 进行分组并进行 sum 和 ohlc 聚合
    result = df.groupby(Grouper(freq="D")).agg(["sum", "ohlc"])
    
    # 使用 pytest 的 tm.assert_frame_equal 函数断言结果与预期值相等
    tm.assert_frame_equal(result, expected)


# 测试 multiindex 的 series 分组键长度与 group axis 相等
def test_groupby_multiindex_series_keys_len_equal_group_axis():
    # 创建一个多级索引的数组和索引名称
    index_array = [["x", "x"], ["a", "b"], ["k", "k"]]
    index_names = ["first", "second", "third"]
    # 使用给定的索引数组和索引名称创建一个多重索引对象
    ri = MultiIndex.from_arrays(index_array, names=index_names)
    
    # 使用数据和之前创建的多重索引对象创建一个序列对象
    s = Series(data=[1, 2], index=ri)
    
    # 对序列按照指定的索引列进行分组，并对分组后的组进行求和操作
    result = s.groupby(["first", "third"]).sum()

    # 定义新的索引数组和索引名称
    index_array = [["x"], ["k"]]
    index_names = ["first", "third"]
    
    # 使用新的索引数组和索引名称创建一个多重索引对象
    ei = MultiIndex.from_arrays(index_array, names=index_names)
    
    # 创建一个期望的序列对象，包含预期的数据和索引
    expected = Series([3], index=ei)
    
    # 使用测试模块中的方法，比较结果序列和期望序列是否相等
    tm.assert_series_equal(result, expected)
def test_groupby_groups_in_BaseGrouper():
    # GH 26326
    # 测试DataFrame在使用pandas.Grouper分组后的正确分组情况
    mi = MultiIndex.from_product([["A", "B"], ["C", "D"]], names=["alpha", "beta"])
    # 创建一个多级索引的DataFrame
    df = DataFrame({"foo": [1, 2, 1, 2], "bar": [1, 2, 3, 4]}, index=mi)
    # 使用多级索引创建DataFrame
    result = df.groupby([Grouper(level="alpha"), "beta"])
    # 根据Grouper(level="alpha")和"beta"进行分组
    expected = df.groupby(["alpha", "beta"])
    # 根据"alpha"和"beta"进行分组
    assert result.groups == expected.groups
    # 断言分组的结果是否与预期相同

    result = df.groupby(["beta", Grouper(level="alpha")])
    # 根据"beta"和Grouper(level="alpha")进行分组
    expected = df.groupby(["beta", "alpha"])
    # 根据"beta"和"alpha"进行分组
    assert result.groups == expected.groups
    # 断言分组的结果是否与预期相同


def test_groups_sort_dropna(sort, dropna):
    # GH#56966, GH#56851
    # 测试数据框根据排序和是否删除NaN值进行分组
    df = DataFrame([[2.0, 1.0], [np.nan, 4.0], [0.0, 3.0]])
    # 创建一个包含NaN值的DataFrame
    keys = [(2.0, 1.0), (np.nan, 4.0), (0.0, 3.0)]
    # 创建包含键的列表
    values = [
        Index([0], dtype="int64"),
        Index([1], dtype="int64"),
        Index([2], dtype="int64"),
    ]
    # 创建包含索引的列表

    if sort:
        taker = [2, 0] if dropna else [2, 0, 1]
    else:
        taker = [0, 2
    # 使用给定的时区创建一个时区感知对象
    tz = tz_naive_fixture
    
    # 创建一个包含"id"和"time"列的数据字典，其中"time"列包含时间戳数据
    data = {
        "id": ["A", "B", "A", "B", "A", "B"],
        "time": [
            Timestamp("2019-01-01 12:00:00"),
            Timestamp("2019-01-01 12:30:00"),
            None,
            None,
            Timestamp("2019-01-01 14:00:00"),
            Timestamp("2019-01-01 14:30:00"),
        ],
    }
    
    # 使用数据字典创建一个 DataFrame，并将"time"列的时间戳数据转换为指定时区的本地时间
    df = DataFrame(data).assign(time=lambda x: x.time.dt.tz_localize(tz))

    # 根据"id"列分组 DataFrame
    grouped = df.groupby("id")
    
    # 调用指定操作符（op）对分组后的 DataFrame 进行操作，获取操作结果
    result = getattr(grouped, op)()
    
    # 创建预期的 DataFrame，其中"time"列的时间戳数据也转换为指定时区的本地时间
    expected = DataFrame(expected).assign(time=lambda x: x.time.dt.tz_localize(tz))
    
    # 使用断言方法比较操作结果和预期结果的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
def test_groupby_only_none_group():
    # see GH21624
    # 在 GH21624 中有提到此问题
    # 当 g 列为 None 时，此代码曾因为 "ValueError: Length of passed values is 1, index implies 0" 而崩溃
    df = DataFrame({"g": [None], "x": 1})
    # 对于 g 列进行分组，并计算 x 列的和
    actual = df.groupby("g")["x"].transform("sum")
    # 期望的结果是包含一个 NaN 值的 Series，名称为 "x"
    expected = Series([np.nan], name="x")

    tm.assert_series_equal(actual, expected)


def test_groupby_duplicate_index():
    # GH#29189
    # 在 GH#29189 中提到此处的 groupby 调用曾经会引发错误
    ser = Series([2, 5, 6, 8], index=[2.0, 4.0, 4.0, 5.0])
    # 按索引的第一层级进行分组
    gb = ser.groupby(level=0)

    # 计算分组后每组的均值
    result = gb.mean()
    # 期望的结果是包含正确均值的 Series
    expected = Series([2, 5.5, 8], index=[2.0, 4.0, 5.0])
    tm.assert_series_equal(result, expected)


def test_group_on_empty_multiindex(transformation_func, request):
    # GH 47787
    # 当只有一行数据时，这些转换应保持模式不变
    df = DataFrame(
        data=[[1, Timestamp("today"), 3, 4]],
        columns=["col_1", "col_2", "col_3", "col_4"],
    )
    # 将 col_3 和 col_4 列转换为整数类型
    df["col_3"] = df["col_3"].astype(int)
    df["col_4"] = df["col_4"].astype(int)
    # 将 col_1 和 col_2 列设置为多重索引
    df = df.set_index(["col_1", "col_2"])
    if transformation_func == "fillna":
        args = ("ffill",)
    else:
        args = ()
    # 如果使用 fillna 函数，则设置警告类型为 FutureWarning
    warn = FutureWarning if transformation_func == "fillna" else None
    warn_msg = "DataFrameGroupBy.fillna is deprecated"
    # 断言应产生警告，并匹配给定的警告消息
    with tm.assert_produces_warning(warn, match=warn_msg):
        # 对空的 DataFrame 进行分组转换操作
        result = df.iloc[:0].groupby(["col_1"]).transform(transformation_func, *args)
    with tm.assert_produces_warning(warn, match=warn_msg):
        # 预期的结果应与完整 DataFrame 组合后的结果相同
        expected = df.groupby(["col_1"]).transform(transformation_func, *args).iloc[:0]
    if transformation_func in ("diff", "shift"):
        expected = expected.astype(int)
    tm.assert_equal(result, expected)

    warn_msg = "SeriesGroupBy.fillna is deprecated"
    with tm.assert_produces_warning(warn, match=warn_msg):
        # 对空的 Series 进行分组转换操作
        result = (
            df["col_3"]
            .iloc[:0]
            .groupby(["col_1"])
            .transform(transformation_func, *args)
        )
    with tm.assert_produces_warning(warn, match=warn_msg):
        # 预期的结果应与完整 Series 组合后的结果相同
        expected = (
            df["col_3"]
            .groupby(["col_1"])
            .transform(transformation_func, *args)
            .iloc[:0]
        )
    if transformation_func in ("diff", "shift"):
        expected = expected.astype(int)
    tm.assert_equal(result, expected)


def test_groupby_crash_on_nunique():
    # Fix following 30253
    # 修复自 30253 后的问题
    dti = date_range("2016-01-01", periods=2, name="foo")
    df = DataFrame({("A", "B"): [1, 2], ("A", "C"): [1, 3], ("D", "B"): [0, 0]})
    df.columns.names = ("bar", "baz")
    df.index = dti

    # 将 DataFrame 转置，然后按索引的第一层级进行分组
    df = df.T
    gb = df.groupby(level=0)
    # 计算每组的唯一值数量
    result = gb.nunique()

    # 期望的结果是一个包含正确唯一值数量的 DataFrame
    expected = DataFrame({"A": [1, 2], "D": [1, 1]}, index=dti)
    expected.columns.name = "bar"
    expected = expected.T

    tm.assert_frame_equal(result, expected)

    # 同样的操作，但针对空列
    gb2 = df[[]].groupby(level=0)
    exp = expected[[]]

    # 计算空列分组后的唯一值数量
    res = gb2.nunique()
    # 使用 pandas.testing 模块中的 assert_frame_equal 函数比较 res 和 exp 两个 DataFrame 是否相等
    tm.assert_frame_equal(res, exp)
def test_groupby_list_level():
    # GH 9790
    # 创建一个 3x3 的浮点数 DataFrame
    expected = DataFrame(np.arange(0, 9).reshape(3, 3), dtype=float)
    # 对 DataFrame 按照第一层级进行分组并计算平均值
    result = expected.groupby(level=[0]).mean()
    # 使用断言检查结果与期望值是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "max_seq_items, expected",
    [
        (5, "{0: [0], 1: [1], 2: [2], 3: [3], 4: [4]}"),
        (4, "{0: [0], 1: [1], 2: [2], 3: [3], ...}"),
        (1, "{0: [0], ...}"),
    ],
)
def test_groups_repr_truncates(max_seq_items, expected):
    # GH 1135
    # 创建一个包含随机数据的 DataFrame，列名为 'a' 并与索引保持一致
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 1)))
    df["a"] = df.index

    # 设置显示选项，限制最大序列条目数
    with pd.option_context("display.max_seq_items", max_seq_items):
        # 获取按 'a' 列分组的结果的字符串表示形式
        result = df.groupby("a").groups.__repr__()
        # 使用断言检查结果与期望值是否相等
        assert result == expected

        # 使用数组作为分组依据，获取分组结果的字符串表示形式
        result = df.groupby(np.array(df.a)).groups.__repr__()
        # 使用断言检查结果与期望值是否相等
        assert result == expected


def test_group_on_two_row_multiindex_returns_one_tuple_key():
    # GH 18451
    # 创建一个包含字典数据的 DataFrame，将 'a' 和 'b' 列设置为索引
    df = DataFrame([{"a": 1, "b": 2, "c": 99}, {"a": 1, "b": 2, "c": 88}])
    df = df.set_index(["a", "b"])

    # 对 DataFrame 按 ['a', 'b'] 列进行分组
    grp = df.groupby(["a", "b"])
    # 获取分组后的索引信息
    result = grp.indices
    # 预期的结果字典，键为元组 (1, 2)，值为索引数组
    expected = {(1, 2): np.array([0, 1], dtype=np.int64)}

    # 使用断言检查结果字典的长度是否为 1
    assert len(result) == 1
    key = (1, 2)
    # 使用断言检查结果中对应键的值与预期是否完全一致
    assert (result[key] == expected[key]).all()


@pytest.mark.parametrize(
    "klass, attr, value",
    [
        (DataFrame, "level", "a"),
        (DataFrame, "as_index", False),
        (DataFrame, "sort", False),
        (DataFrame, "group_keys", False),
        (DataFrame, "observed", True),
        (DataFrame, "dropna", False),
        (Series, "level", "a"),
        (Series, "as_index", False),
        (Series, "sort", False),
        (Series, "group_keys", False),
        (Series, "observed", True),
        (Series, "dropna", False),
    ],
)
def test_subsetting_columns_keeps_attrs(klass, attr, value):
    # GH 9959 - When subsetting columns, don't drop attributes
    # 创建一个包含 {'a': [1], 'b': [2], 'c': [3]} 的 DataFrame
    df = DataFrame({"a": [1], "b": [2], "c": [3]})
    if attr != "axis":
        # 如果属性不是 'axis'，则按 'a' 列设置索引
        df = df.set_index("a")

    # 使用 {attr: value} 参数创建按 'a' 列分组的结果
    expected = df.groupby("a", **{attr: value})
    # 获取期望的结果，选择 'b' 列（DataFrame）或 'b' 列（Series）
    result = expected[["b"]] if klass is DataFrame else expected["b"]
    # 使用断言检查结果对象的属性是否与期望一致
    assert getattr(result, attr) == getattr(expected, attr)


@pytest.mark.parametrize("func", ["sum", "any", "shift"])
def test_groupby_column_index_name_lost(func):
    # GH: 29764 groupby loses index sometimes
    # 创建一个包含 ['a'] 索引名为 'idx' 的 Index 对象
    expected = Index(["a"], name="idx")
    # 创建一个包含 [[1]] 数据的 DataFrame，列名为 expected
    df = DataFrame([[1]], columns=expected)
    # 对 DataFrame 按 [1] 列进行分组
    df_grouped = df.groupby([1])
    # 获取分组后应用 func 函数的结果的列索引
    result = getattr(df_grouped, func)().columns
    # 使用断言检查结果的列索引与期望是否完全一致
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "infer_string",
    [
        False,
        pytest.param(True, marks=td.skip_if_no("pyarrow")),
    ],
)
def test_groupby_duplicate_columns(infer_string):
    # GH: 31735
    if infer_string:
        # 如果 infer_string 为 True，则尝试导入 'pyarrow'，否则跳过测试
        pytest.importorskip("pyarrow")
    # 创建一个包含 {'A': [...], 'B': [...], 'B': [...]} 的 DataFrame，数据类型为对象
    df = DataFrame(
        {"A": ["f", "e", "g", "h"], "B": ["a", "b", "c", "d"], "C": [1, 2, 3, 4]}
    ).astype(object)
    # 重命名列名为 ["A", "B", "B"]
    df.columns = ["A", "B", "B"]
    # 使用 pandas 的上下文管理器设置选项 "future.infer_string" 为指定的 infer_string 值
    with pd.option_context("future.infer_string", infer_string):
        # 对 DataFrame df 进行分组，按列索引 [0, 0, 0, 0] 进行分组并计算每组的最小值
        result = df.groupby([0, 0, 0, 0]).min()
    
    # 创建预期的 DataFrame，包含一行数据 [["e", "a", 1]]，索引为 np.array([0])，列名为 ["A", "B", "B"]，数据类型为 object
    expected = DataFrame(
        [["e", "a", 1]], index=np.array([0]), columns=["A", "B", "B"], dtype=object
    )
    
    # 使用 pandas.testing 模块中的 assert_frame_equal 函数比较 result 和 expected 两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
def test_groupby_series_with_tuple_name():
    # GH 37755
    # 创建一个 Series 对象，使用元组作为名称和索引
    ser = Series([1, 2, 3, 4], index=[1, 1, 2, 2], name=("a", "a"))
    # 设置索引的名称为元组 ("b", "b")
    ser.index.name = ("b", "b")
    # 对 Series 对象按照第一级索引分组，并取每组的最后一个元素
    result = ser.groupby(level=0).last()
    # 创建一个预期的 Series 对象，使用相同的元组名称作为名称和索引
    expected = Series([2, 4], index=[1, 2], name=("a", "a"))
    # 设置预期结果的索引名称为 ("b", "b")
    expected.index.name = ("b", "b")
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "func, values", [("sum", [97.0, 98.0]), ("mean", [24.25, 24.5])]
)
def test_groupby_numerical_stability_sum_mean(func, values):
    # GH#38778
    # 创建一个包含数值和分组信息的 DataFrame 对象
    data = [1e16, 1e16, 97, 98, -5e15, -5e15, -5e15, -5e15]
    df = DataFrame({"group": [1, 2] * 4, "a": data, "b": data})
    # 对 DataFrame 对象按照 "group" 列分组，并应用给定的函数（sum 或 mean）
    result = getattr(df.groupby("group"), func)()
    # 创建预期的 DataFrame 对象，包含按照 "group" 列计算得到的 values
    expected = DataFrame({"a": values, "b": values}, index=Index([1, 2], name="group"))
    # 断言两个 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)


def test_groupby_numerical_stability_cumsum():
    # GH#38934
    # 创建一个包含数值和分组信息的 DataFrame 对象
    data = [1e16, 1e16, 97, 98, -5e15, -5e15, -5e15, -5e15]
    df = DataFrame({"group": [1, 2] * 4, "a": data, "b": data})
    # 对 DataFrame 对象按照 "group" 列分组，并计算每个分组的累积和
    result = df.groupby("group").cumsum()
    # 创建预期的 DataFrame 对象，包含按照 "group" 列计算得到的累积和
    exp_data = (
        [1e16] * 2 + [1e16 + 96, 1e16 + 98] + [5e15 + 97, 5e15 + 98] + [97.0, 98.0]
    )
    expected = DataFrame({"a": exp_data, "b": exp_data})
    # 断言两个 DataFrame 对象是否相等，要求精确匹配
    tm.assert_frame_equal(result, expected, check_exact=True)


def test_groupby_cumsum_skipna_false():
    # GH#46216 don't propagate np.nan above the diagonal
    # 创建一个包含随机数和 NaN 值的 DataFrame 对象
    arr = np.random.default_rng(2).standard_normal((5, 5))
    df = DataFrame(arr)
    for i in range(5):
        df.iloc[i, i] = np.nan

    df["A"] = 1
    gb = df.groupby("A")

    # 对分组后的 DataFrame 对象计算累积和，不忽略 NaN 值
    res = gb.cumsum(skipna=False)

    # 创建预期的 DataFrame 对象，不忽略 NaN 值的累积和
    expected = df[[0, 1, 2, 3, 4]].cumsum(skipna=False)
    # 断言两个 DataFrame 对象是否相等
    tm.assert_frame_equal(res, expected)


def test_groupby_cumsum_timedelta64():
    # GH#46216 don't ignore is_datetimelike in libgroupby.group_cumsum
    # 创建一个包含日期时间数据和 NaN 值的 Series 对象
    dti = date_range("2016-01-01", periods=5)
    ser = Series(dti) - dti[0]
    ser[2] = pd.NaT

    df = DataFrame({"A": 1, "B": ser})
    gb = df.groupby("A")

    # 对分组后的 DataFrame 对象计算累积和，包括日期时间数据
    res = gb.cumsum(numeric_only=False, skipna=True)
    # 创建预期的 DataFrame 对象，按分组累积和计算日期时间数据
    exp = DataFrame({"B": [ser[0], ser[1], pd.NaT, ser[4], ser[4] * 2]})
    # 断言两个 DataFrame 对象是否相等
    tm.assert_frame_equal(res, exp)

    # 对分组后的 DataFrame 对象计算累积和，不忽略 NaN 值
    res = gb.cumsum(numeric_only=False, skipna=False)
    # 创建预期的 DataFrame 对象，按分组累积和计算日期时间数据（NaN 值不变）
    exp = DataFrame({"B": [ser[0], ser[1], pd.NaT, pd.NaT, pd.NaT]})
    # 断言两个 DataFrame 对象是否相等
    tm.assert_frame_equal(res, exp)


def test_groupby_mean_duplicate_index(rand_series_with_duplicate_datetimeindex):
    # 创建一个包含重复日期索引的 Series 对象
    dups = rand_series_with_duplicate_datetimeindex
    # 对 Series 对象按照第一级索引分组，并计算每组的均值
    result = dups.groupby(level=0).mean()
    # 创建预期的 Series 对象，按日期索引计算每组的均值
    expected = dups.groupby(dups.index).mean()
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


def test_groupby_all_nan_groups_drop():
    # GH 15036
    # 创建一个包含 NaN 索引的 Series 对象
    s = Series([1, 2, 3], [np.nan, np.nan, np.nan])
    # 对 Series 对象按照索引分组，并计算每组的和
    result = s.groupby(s.index).sum()
    # 创建预期的 Series 对象，期望结果是空的 Series 对象
    expected = Series([], index=Index([], dtype=np.float64), dtype=np.int64)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("numeric_only", [True, False])
def test_groupby_empty_multi_column(as_index, numeric_only):
    # GH 15106 & GH 41998
    # 参数化测试函数，用于测试空的多列分组情况
    # 创建一个空的 DataFrame，列名为 "A", "B", "C"
    df = DataFrame(data=[], columns=["A", "B", "C"])
    # 根据列 "A", "B" 对 DataFrame 进行分组，根据参数 as_index 决定是否保留分组的列作为索引
    gb = df.groupby(["A", "B"], as_index=as_index)
    # 对分组后的结果进行求和操作，numeric_only 参数指定是否仅对数值列进行求和
    result = gb.sum(numeric_only=numeric_only)
    # 如果 as_index 为 True，则创建一个多级索引对象 MultiIndex，否则创建一个 RangeIndex
    if as_index:
        index = MultiIndex([[], []], [[], []], names=["A", "B"])
        # 如果 numeric_only 为 False，则设置列名为 ["C"]，否则为空列表
        columns = ["C"] if not numeric_only else []
    else:
        index = RangeIndex(0)
        # 如果 numeric_only 为 False，则设置列名为 ["A", "B", "C"]，否则设置为 ["A", "B"]
        columns = ["A", "B", "C"] if not numeric_only else ["A", "B"]
    # 创建一个期望的 DataFrame，数据为空，列名为 columns，索引为 index
    expected = DataFrame([], columns=columns, index=index)
    # 使用测试工具 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
# 测试函数：验证对非数值类型数据框进行分组聚合
def test_groupby_aggregation_non_numeric_dtype():
    # 创建包含性别和值的数据框
    df = DataFrame(
        [["M", [1]], ["M", [1]], ["W", [10]], ["W", [20]]], columns=["MW", "v"]
    )

    # 预期的聚合结果数据框，包含每个性别对应的值列表
    expected = DataFrame(
        {
            "v": [[1, 1], [10, 20]],
        },
        index=Index(["M", "W"], dtype="object", name="MW"),
    )

    # 按照性别列进行分组
    gb = df.groupby(by=["MW"])
    # 对分组后的结果进行求和聚合
    result = gb.sum()
    # 使用测试工具比较实际结果和预期结果的数据框是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数：验证对包含多个非数值类型列的数据框进行分组聚合
def test_groupby_aggregation_multi_non_numeric_dtype():
    # 创建数据框，包含整数列、时间增量列和时间增量乘以10的列
    df = DataFrame(
        {
            "x": [1, 0, 1, 1, 0],
            "y": [Timedelta(i, "days") for i in range(1, 6)],
            "z": [Timedelta(i * 10, "days") for i in range(1, 6)],
        }
    )

    # 预期的聚合结果数据框，包含每个整数值对应的时间增量列表
    expected = DataFrame(
        {
            "y": [Timedelta(i, "days") for i in range(7, 9)],
            "z": [Timedelta(i * 10, "days") for i in range(7, 9)],
        },
        index=Index([0, 1], dtype="int64", name="x"),
    )

    # 按照整数列进行分组
    gb = df.groupby(by=["x"])
    # 对分组后的结果进行求和聚合
    result = gb.sum()
    # 使用测试工具比较实际结果和预期结果的数据框是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数：验证对包含数值和非数值类型列的数据框进行分组聚合
def test_groupby_aggregation_numeric_with_non_numeric_dtype():
    # 创建数据框，包含整数列、时间增量列和数值列
    df = DataFrame(
        {
            "x": [1, 0, 1, 1, 0],
            "y": [Timedelta(i, "days") for i in range(1, 6)],
            "z": list(range(1, 6)),
        }
    )

    # 预期的聚合结果数据框，包含每个整数值对应的时间增量和数值的和
    expected = DataFrame(
        {"y": [Timedelta(7, "days"), Timedelta(8, "days")], "z": [7, 8]},
        index=Index([0, 1], dtype="int64", name="x"),
    )

    # 按照整数列进行分组
    gb = df.groupby(by=["x"])
    # 对分组后的结果进行求和聚合
    result = gb.sum()
    # 使用测试工具比较实际结果和预期结果的数据框是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数：验证对过滤后的数据框进行分组标准差计算
def test_groupby_filtered_df_std():
    # 创建包含字典列表的数据框
    dicts = [
        {"filter_col": False, "groupby_col": True, "bool_col": True, "float_col": 10.5},
        {"filter_col": True, "groupby_col": True, "bool_col": True, "float_col": 20.5},
        {"filter_col": True, "groupby_col": True, "bool_col": True, "float_col": 30.5},
    ]
    df = DataFrame(dicts)

    # 过滤出 filter_col 列为 True 的子数据框
    df_filter = df[df["filter_col"] == True]  # noqa: E712
    # 按照 groupby_col 列进行分组
    dfgb = df_filter.groupby("groupby_col")
    # 对分组后的结果计算标准差
    result = dfgb.std()
    # 预期的标准差结果数据框
    expected = DataFrame(
        [[0.0, 0.0, 7.071068]],
        columns=["filter_col", "bool_col", "float_col"],
        index=Index([True], name="groupby_col"),
    )
    # 断言实际结果和预期结果的数据框是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数：验证对包含日期分类键的多列数据框进行分组并返回索引
def test_datetime_categorical_multikey_groupby_indices():
    # 创建数据框，包含字符列 a、日期分类列 b 和分类编码列 c
    df = DataFrame(
        {
            "a": Series(list("abc")),
            "b": Series(
                to_datetime(["2018-01-01", "2018-02-01", "2018-03-01"]),
                dtype="category",
            ),
            "c": Categorical.from_codes([-1, 0, 1], categories=[0, 1]),
        }
    )
    # 按照列 a 和列 b 进行分组，并返回分组后的索引
    result = df.groupby(["a", "b"], observed=False).indices
    # 预期的索引结果字典，包含每个分组键对应的索引数组
    expected = {
        ("a", Timestamp("2018-01-01 00:00:00")): np.array([0]),
        ("b", Timestamp("2018-02-01 00:00:00")): np.array([1]),
        ("c", Timestamp("2018-03-01 00:00:00")): np.array([2]),
    }
    # 使用断言比较实际结果和预期结果的索引是否相等
    assert result == expected
    # GH34037
    # 创建名为 name_l 的列表，包含 5 个 "Alice" 和 5 个 "Bob" 字符串
    name_l = ["Alice"] * 5 + ["Bob"] * 5
    # 创建名为 val_l 的列表，包含 NaN, NaN, 1, 2, 3, NaN, 1, 2, 3, 4 的值
    val_l = [np.nan, np.nan, 1, 2, 3] + [np.nan, 1, 2, 3, 4]
    # 创建 DataFrame test_df，由 name_l 和 val_l 列表转置后组成，列名为 "name" 和 "val"
    test_df = DataFrame([name_l, val_l]).T
    # 设置 test_df 的列名为 "name" 和 "val"
    test_df.columns = ["name", "val"]

    # 定义字符串 result_error_msg，用于匹配测试异常信息的正则表达式
    result_error_msg = (
        r"^[a-zA-Z._]*\(\) got an unexpected keyword argument 'min_period'"
    )
    # 使用 pytest 的 pytest.raises 上下文管理器捕获 TypeError 异常，验证是否匹配 result_error_msg 的错误信息
    with pytest.raises(TypeError, match=result_error_msg):
        # 对 test_df 按 "name" 分组后，对 "val" 列进行滚动窗口计算，期望在此过程中抛出 TypeError 异常
        test_df.groupby("name")["val"].rolling(window=2, min_period=1).sum()
# 使用 pytest 的 parametrize 装饰器来定义多个测试参数化输入
@pytest.mark.parametrize(
    "dtype",  # 参数名为 dtype
    [
        object,  # 第一个参数是 Python 内置的 object 类型
        pytest.param("string[pyarrow_numpy]", marks=td.skip_if_no("pyarrow")),  # 第二个参数是一个字符串，同时添加了一个 pytest 的标记条件
    ],
)
def test_by_column_values_with_same_starting_value(dtype):
    # 测试函数：检验按列值分组后具有相同起始值的情况
    # 创建一个 DataFrame 对象 df，包含三列："Name"、"Credit" 和 "Mood"
    df = DataFrame(
        {
            "Name": ["Thomas", "Thomas", "Thomas John"],
            "Credit": [1200, 1300, 900],
            "Mood": Series(["sad", "happy", "happy"], dtype=dtype),  # 根据参数 dtype 指定 Mood 列的数据类型
        }
    )
    # 定义一个聚合操作字典
    aggregate_details = {"Mood": Series.mode, "Credit": "sum"}
    # 对 df 按 "Name" 列进行分组，并应用聚合操作
    result = df.groupby(["Name"]).agg(aggregate_details)
    # 预期的结果 DataFrame，设置 "Name" 列为索引
    expected_result = DataFrame(
        {
            "Mood": [["happy", "sad"], "happy"],
            "Credit": [2500, 900],
            "Name": ["Thomas", "Thomas John"],
        }
    ).set_index("Name")
    # 使用 pytest 的 assert_frame_equal 函数比较 result 和 expected_result 是否相等
    tm.assert_frame_equal(result, expected_result)


def test_groupby_none_in_first_mi_level():
    # 测试函数：检验在多级索引的第一个级别中是否包含 None 值的情况
    # 创建一个 Series 对象 ser，使用 MultiIndex.from_arrays 方法创建多级索引
    arr = [[None, 1, 0, 1], [2, 3, 2, 3]]
    ser = Series(1, index=MultiIndex.from_arrays(arr, names=["a", "b"]))
    # 对 ser 按 level=[0, 1] 进行分组，并应用求和操作
    result = ser.groupby(level=[0, 1]).sum()
    # 预期的结果 Series，设置了新的 MultiIndex
    expected = Series(
        [1, 2], MultiIndex.from_tuples([(0.0, 2), (1.0, 3)], names=["a", "b"])
    )
    # 使用 pytest 的 assert_series_equal 函数比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


def test_groupby_none_column_name():
    # 测试函数：检验是否能够处理列名为 None 的情况
    # 创建一个 DataFrame 对象 df，其中列名包括 None 和 "b"、"c"
    df = DataFrame({None: [1, 1, 2, 2], "b": [1, 1, 2, 3], "c": [4, 5, 6, 7]})
    # 对 df 按 by=[None] 进行分组，并应用求和操作
    result = df.groupby(by=[None]).sum()
    # 预期的结果 DataFrame，设置了新的 Index，其名称为 None
    expected = DataFrame({"b": [2, 5], "c": [9, 13]}, index=Index([1, 2], name=None))
    # 使用 pytest 的 assert_frame_equal 函数比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("selection", [None, "a", ["a"]])
def test_single_element_list_grouping(selection):
    # 测试函数：检验在不同选择条件下，单元素列表分组的情况
    # 创建一个 DataFrame 对象 df，包含三列："a"、"b" 和 "c"
    df = DataFrame({"a": [1, 2], "b": [np.nan, 5], "c": [np.nan, 2]}, index=["x", "y"])
    # 根据选择条件 selection 决定对 df 进行不同的分组操作
    grouped = df.groupby(["a"]) if selection is None else df.groupby(["a"])[selection]
    # 将分组结果中的键（key）提取出来，形成列表 result
    result = [key for key, _ in grouped]
    # 预期的结果列表 expected
    expected = [(1,), (2,)]
    # 使用标准的断言方式，检验 result 和 expected 是否相等
    assert result == expected


def test_groupby_string_dtype():
    # 测试函数：检验处理字符串类型列的情况
    # 创建一个 DataFrame 对象 df，包含两列："str_col" 和 "num_col"
    df = DataFrame({"str_col": ["a", "b", "c", "a"], "num_col": [1, 2, 3, 2]})
    # 将 df 中的 "str_col" 列转换为 string 类型
    df["str_col"] = df["str_col"].astype("string")
    # 创建预期的结果 DataFrame expected
    expected = DataFrame(
        {
            "str_col": [
                "a",
                "b",
                "c",
            ],
            "num_col": [1.5, 2.0, 3.0],  # 按 "str_col" 列分组后，"num_col" 列的均值
        }
    )
    expected["str_col"] = expected["str_col"].astype("string")
    # 对 df 按 "str_col" 列进行分组，并计算均值
    grouped = df.groupby("str_col", as_index=False)
    result = grouped.mean()
    # 使用 pytest 的 assert_frame_equal 函数比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "level_arg, multiindex", [([0], False), ((0,), False), ([0], True), ((0,), True)]
)
def test_single_element_listlike_level_grouping(level_arg, multiindex):
    # 测试函数：检验单元素列表形式的级别（level）分组情况
    # 创建一个 DataFrame 对象 df，包含三列："a"、"b" 和 "c"
    df = DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}, index=["x", "y"])
    # 如果 multiindex 为 True，则将 df 转换为多级索引形式
    if multiindex:
        df = df.set_index(["a", "b"])
    # 对 df 按 level=level_arg 进行分组
    result = [key for key, _ in df.groupby(level=level_arg)]
    # 预期的结果列表 expected
    expected = [(1,), (2,)] if multiindex else [("x",), ("y",)]
    # 使用标准的断言方式，检验 result 和 expected 是否相等
    assert result == expected
@pytest.mark.parametrize("func", ["sum", "cumsum", "cumprod", "prod"])
def test_groupby_avoid_casting_to_float(func):
    # 使用 pytest 的 parametrize 装饰器，对函数 test_groupby_avoid_casting_to_float 进行参数化测试，func 可取 "sum", "cumsum", "cumprod", "prod"
    # GH#37493，指示这段代码与 GitHub 上 issue 编号为 37493 相关联

    # 设置一个大整数值作为变量 val
    val = 922337203685477580
    # 创建一个 DataFrame 对象 df，包含两列：'a' 列值为 1，'b' 列为包含 val 的列表
    df = DataFrame({"a": 1, "b": [val]})
    # 通过 getattr 调用 df.groupby("a") 的 func 方法，并将结果与 val 做减法
    result = getattr(df.groupby("a"), func)() - val
    # 创建一个期望的 DataFrame 对象 expected，包含一列 'b'，其值为 [0]，索引为 [1]
    expected = DataFrame({"b": [0]}, index=Index([1], name="a"))
    # 如果 func 是 "cumsum" 或 "cumprod"，则对期望的 DataFrame 进行重置索引
    if func in ["cumsum", "cumprod"]:
        expected = expected.reset_index(drop=True)
    # 使用 pytest 的 tm.assert_frame_equal 方法断言 result 与 expected 的内容是否相等


@pytest.mark.parametrize("func, val", [("sum", 3), ("prod", 2)])
def test_groupby_sum_support_mask(any_numeric_ea_dtype, func, val):
    # 使用 pytest 的 parametrize 装饰器，对函数 test_groupby_sum_support_mask 进行参数化测试，func 可取 "sum", "prod"；val 分别为 3 和 2
    # GH#37493，指示这段代码与 GitHub 上 issue 编号为 37493 相关联

    # 创建一个 DataFrame 对象 df，包含两列：'a' 列值为 1，'b' 列为包含 [1, 2, pd.NA] 的列表，数据类型为 any_numeric_ea_dtype
    df = DataFrame({"a": 1, "b": [1, 2, pd.NA]}, dtype=any_numeric_ea_dtype)
    # 通过 getattr 调用 df.groupby("a") 的 func 方法
    result = getattr(df.groupby("a"), func)()
    # 创建一个期望的 DataFrame 对象 expected，包含一列 'b'，其值为 [val]，索引为 [1]，数据类型为 any_numeric_ea_dtype
    expected = DataFrame(
        {"b": [val]},
        index=Index([1], name="a", dtype=any_numeric_ea_dtype),
        dtype=any_numeric_ea_dtype,
    )
    # 使用 pytest 的 tm.assert_frame_equal 方法断言 result 与 expected 的内容是否相等


@pytest.mark.parametrize("val, dtype", [(111, "int"), (222, "uint")])
def test_groupby_overflow(val, dtype):
    # 使用 pytest 的 parametrize 装饰器，对函数 test_groupby_overflow 进行参数化测试，val 分别为 111 和 222；dtype 分别为 "int" 和 "uint"
    # GH#37493，指示这段代码与 GitHub 上 issue 编号为 37493 相关联

    # 创建一个 DataFrame 对象 df，包含两列：'a' 列值为 1，'b' 列为包含 [val, val] 的列表，数据类型为 f"{dtype}8"
    df = DataFrame({"a": 1, "b": [val, val]}, dtype=f"{dtype}8")
    # 使用 groupby("a") 后调用 sum() 方法得到 result
    result = df.groupby("a").sum()
    # 创建一个期望的 DataFrame 对象 expected，包含一列 'b'，其值为 [val * 2]，索引为 [1]，数据类型为 f"{dtype}64"
    expected = DataFrame(
        {"b": [val * 2]},
        index=Index([1], name="a", dtype=f"{dtype}8"),
        dtype=f"{dtype}64",
    )
    # 使用 pytest 的 tm.assert_frame_equal 方法断言 result 与 expected 的内容是否相等

    # 再次使用 groupby("a") 后调用 cumsum() 方法得到 result
    result = df.groupby("a").cumsum()
    # 创建一个期望的 DataFrame 对象 expected，包含一列 'b'，其值为 [val, val * 2]，数据类型为 f"{dtype}64"
    expected = DataFrame({"b": [val, val * 2]}, dtype=f"{dtype}64")
    # 使用 pytest 的 tm.assert_frame_equal 方法断言 result 与 expected 的内容是否相等

    # 最后使用 groupby("a") 后调用 prod() 方法得到 result
    result = df.groupby("a").prod()
    # 创建一个期望的 DataFrame 对象 expected，包含一列 'b'，其值为 [val * val]，索引为 [1]，数据类型为 f"{dtype}64"
    expected = DataFrame(
        {"b": [val * val]},
        index=Index([1], name="a", dtype=f"{dtype}8"),
        dtype=f"{dtype}64",
    )
    # 使用 pytest 的 tm.assert_frame_equal 方法断言 result 与 expected 的内容是否相等


@pytest.mark.parametrize("skipna, val", [(True, 3), (False, pd.NA)])
def test_groupby_cumsum_mask(any_numeric_ea_dtype, skipna, val):
    # 使用 pytest 的 parametrize 装饰器，对函数 test_groupby_cumsum_mask 进行参数化测试，skipna 可取 True 或 False；val 可为 3 或 pd.NA
    # GH#37493，指示这段代码与 GitHub 上 issue 编号为 37493 相关联

    # 创建一个 DataFrame 对象 df，包含两列：'a' 列值为 1，'b' 列为包含 [1, pd.NA, 2] 的列表，数据类型为 any_numeric_ea_dtype
    df = DataFrame({"a": 1, "b": [1, pd.NA, 2]}, dtype=any_numeric_ea_dtype)
    # 使用 groupby("a") 后调用 cumsum(skipna=skipna) 方法得到 result
    result = df.groupby("a").cumsum(skipna=skipna)
    # 创建一个期望的 DataFrame 对象 expected，包含一列 'b'，其值为 [1, pd.NA, val]，数据类型为 any_numeric_ea_dtype
    expected = DataFrame(
        {"b": [1, pd.NA, val]},
        dtype=any_numeric_ea_dtype,
    )
    # 使用 pytest 的 tm.assert_frame_equal 方法断言 result 与 expected 的内容是否相等


@pytest.mark.parametrize(
    "val_in, index, val_out",
    [
        (
            [1.0, 2.0, 3.0, 4.0, 5.0],
            ["foo", "foo", "bar", "baz", "blah"],
            [3.0, 4.0, 5.0, 3.0],
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ["foo", "foo", "bar", "baz", "blah", "blah"],
            [3.0, 4.0, 11.0, 3.0],
        ),
    ],
)
def test_groupby_index_name_in_index_content(val_in, index, val_out):
    # 使用 pytest 的 parametrize 装饰器，对函数 test_groupby_index_name_in_index_content 进行参数化测试
    # val_in, index, val_out 分别对应两组测试数据

    # 创建一个 Series 对象 series，包含 val_in 作为数据，名称为 "values"，index 为 Index 对象，名称为 "blah"
    series = Series(data=val_in, name="values", index=Index(index, name="blah"))
    # 使用 groupby("blah") 后调用 sum() 方法得到 result
    result = series.groupby("blah").sum()
    # 创建一个期望的 Series 对象 expected，包含 val_out 作为数据，名称为 "values"，index 为 Index 对象，名称为 "blah"
    expected = Series(
        data=val_out,
        name="values",
        index=Index(["bar", "baz", "blah", "foo"], name="blah"),
    )
    # 使用 pytest 的 tm.assert_series_equal 方法断言 result 与 expected 的内容是否相等

    # 将 series 转换为 DataFrame 对象
    # 使用测试工具库中的函数 tm.assert_frame_equal 比较 result 和 expected 两个数据框是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("n", [1, 10, 32, 100, 1000])
def test_sum_of_booleans(n):
    # 标记此测试为 GH 50347
    # 创建一个 DataFrame，包含一个整数列和一个布尔列，布尔列中的值都是 True，重复 n 次
    df = DataFrame({"groupby_col": 1, "bool": [True] * n)
    # 将布尔列中的值转换为布尔 Series，等效于 df["bool"] == True
    df["bool"] = df["bool"].eq(True)
    # 对 DataFrame 进行 groupby 操作，并对布尔列进行求和
    result = df.groupby("groupby_col").sum()
    # 创建预期结果的 DataFrame，期望布尔列的和为 n，索引为 [1]
    expected = DataFrame({"bool": [n]}, index=Index([1], name="groupby_col"))
    # 使用测试框架的方法比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in remainder:RuntimeWarning"
)
@pytest.mark.parametrize("method", ["head", "tail", "nth", "first", "last"])
def test_groupby_method_drop_na(method):
    # 标记此测试为 GH 21755
    # 创建一个 DataFrame 包含两列，其中一列包含字符串和 NaN，另一列是整数范围
    df = DataFrame({"A": ["a", np.nan, "b", np.nan, "c"], "B": range(5)})

    # 根据不同的 method 参数选择不同的 groupby 方法调用
    if method == "nth":
        result = getattr(df.groupby("A"), method)(n=0)
    else:
        result = getattr(df.groupby("A"), method)()

    # 根据 method 参数设置预期结果的 DataFrame
    if method in ["first", "last"]:
        expected = DataFrame({"B": [0, 2, 4]}).set_index(
            Series(["a", "b", "c"], name="A")
        )
    else:
        expected = DataFrame({"A": ["a", "b", "c"], "B": [0, 2, 4]}, index=[0, 2, 4])
    # 使用测试框架的方法比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


def test_groupby_reduce_period():
    # 标记此测试为 GH#51040
    # 创建一个包含 100 天的 PeriodIndex 对象
    pi = pd.period_range("2016-01-01", periods=100, freq="D")
    # 创建一个重复列表以及将 PeriodIndex 转换为 Series 对象
    grps = list(range(10)) * 10
    ser = pi.to_series()
    # 对 Series 对象进行 groupby 操作，根据 grps 列表的值进行分组
    gb = ser.groupby(grps)

    # 使用 pytest 检查特定操作抛出的异常情况
    with pytest.raises(TypeError, match="Period type does not support sum operations"):
        gb.sum()
    with pytest.raises(
        TypeError, match="Period type does not support cumsum operations"
    ):
        gb.cumsum()
    with pytest.raises(TypeError, match="Period type does not support prod operations"):
        gb.prod()
    with pytest.raises(
        TypeError, match="Period type does not support cumprod operations"
    ):
        gb.cumprod()

    # 对 groupby 结果执行 max 操作，比较结果是否符合预期
    res = gb.max()
    expected = ser[-10:]
    expected.index = Index(range(10), dtype=int)
    tm.assert_series_equal(res, expected)

    # 对 groupby 结果执行 min 操作，比较结果是否符合预期
    res = gb.min()
    expected = ser[:10]
    expected.index = Index(range(10), dtype=int)
    tm.assert_series_equal(res, expected)


def test_obj_with_exclusions_duplicate_columns():
    # 标记此测试为 GH#50806
    # 创建一个包含单个列表的 DataFrame，并设置列名为重复的整数
    df = DataFrame([[0, 1, 2, 3]])
    df.columns = [0, 1, 2, 0]
    # 根据 DataFrame 的一列进行 groupby 操作
    gb = df.groupby(df[1])
    # 获取 groupby 对象的 _obj_with_exclusions 属性
    result = gb._obj_with_exclusions
    # 创建预期结果的 DataFrame，选择特定的列
    expected = df.take([0, 2, 3], axis=1)
    # 使用测试框架的方法比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("numeric_only", [True, False])
def test_groupby_numeric_only_std_no_result(numeric_only):
    # 标记此测试为 GH 51080
    # 创建包含非数值字典的 DataFrame
    dicts_non_numeric = [{"a": "foo", "b": "bar"}, {"a": "car", "b": "dar"}]
    df = DataFrame(dicts_non_numeric)
    # 根据 'a' 列进行 groupby 操作，不排序并且不作为索引
    dfgb = df.groupby("a", as_index=False, sort=False)

    # 根据 numeric_only 参数选择是否计算标准差
    if numeric_only:
        result = dfgb.std(numeric_only=True)
        expected_df = DataFrame(["foo", "car"], columns=["a"])
        # 使用测试框架的方法比较 result 和 expected_df 是否相等
        tm.assert_frame_equal(result, expected_df)
    else:
        # 预期此操作抛出 ValueError 异常，错误信息包含特定字符串
        with pytest.raises(
            ValueError, match="could not convert string to float: 'bar'"
        ):
            dfgb.std(numeric_only=numeric_only)
# 定义一个测试函数，用于测试在使用分类区间列时的分组操作
def test_grouping_with_categorical_interval_columns():
    # 创建一个包含两列的数据帧，一列为数值，一列为类别字符串
    df = DataFrame({"x": [0.1, 0.2, 0.3, -0.4, 0.5], "w": ["a", "b", "a", "c", "a"]})
    # 对数值列进行分位数分段，得到分段结果 qq
    qq = pd.qcut(df["x"], q=np.linspace(0, 1, 5))
    # 根据分段结果 qq 和类别列 "w" 进行分组，计算分组后每组 "x" 列的均值，observed=False 表示不要求观察所有可能的类别
    result = df.groupby([qq, "w"], observed=False)["x"].agg("mean")
    
    # 创建第一层索引为分类类型，表示数值区间
    categorical_index_level_1 = Categorical(
        [
            Interval(-0.401, 0.1, closed="right"),
            Interval(0.1, 0.2, closed="right"),
            Interval(0.2, 0.3, closed="right"),
            Interval(0.3, 0.5, closed="right"),
        ],
        ordered=True,
    )
    # 创建第二层索引为简单字符串列表
    index_level_2 = ["a", "b", "c"]
    # 使用两层索引的笛卡尔积创建多级索引对象
    mi = MultiIndex.from_product(
        [categorical_index_level_1, index_level_2], names=["x", "w"]
    )
    # 创建预期结果的 Series，数据为 NaN 的位置为 None
    expected = Series(
        np.array(
            [
                0.1,
                np.nan,
                -0.4,
                np.nan,
                0.2,
                np.nan,
                0.3,
                np.nan,
                np.nan,
                0.5,
                np.nan,
                np.nan,
            ]
        ),
        index=mi,
        name="x",
    )
    # 断言分组计算的结果与预期结果相等
    tm.assert_series_equal(result, expected)


# 使用参数化测试来测试在存在 NaN 值时进行分组求和应该返回 NaN 的情况
@pytest.mark.parametrize("bug_var", [1, "a"])
def test_groupby_sum_on_nan_should_return_nan(bug_var):
    # 创建一个包含 NaN 值的数据帧
    df = DataFrame({"A": [bug_var, bug_var, bug_var, np.nan]})
    # 对数据帧进行分组，分组函数是一个简单的标识函数 lambda x: x
    dfgb = df.groupby(lambda x: x)
    # 对分组后的结果进行求和，要求至少有一个非 NaN 值才返回非 NaN
    result = dfgb.sum(min_count=1)

    # 创建预期的结果数据帧，将 NaN 替换为 None
    expected_df = DataFrame([bug_var, bug_var, bug_var, None], columns=["A"])
    # 断言分组求和的结果与预期结果相等
    tm.assert_frame_equal(result, expected_df)


# 使用参数化测试来测试在不同方法下的分组操作
@pytest.mark.parametrize(
    "method",
    [
        "count",
        "corr",
        "cummax",
        "cummin",
        "cumprod",
        "describe",
        "rank",
        "quantile",
        "diff",
        "shift",
        "all",
        "any",
        "idxmin",
        "idxmax",
        "ffill",
        "bfill",
        "pct_change",
    ],
)
def test_groupby_selection_with_methods(df, method):
    # 创建一个具有时间索引的日期范围
    rng = date_range("2014", periods=len(df))
    # 将数据帧的索引设置为创建的日期范围
    df.index = rng

    # 对数据帧按照 "A" 列进行分组，选择 "C" 列，并根据指定的方法进行操作
    g = df.groupby(["A"])[["C"]]
    # 创建一个预期的分组对象，也是选择 "C" 列，并按照 "A" 列进行分组
    g_exp = df[["C"]].groupby(df["A"])
    # TODO check groupby with > 1 col ?

    # 使用 getattr 动态调用分组对象的方法进行计算
    res = getattr(g, method)()
    exp = getattr(g_exp, method)()

    # 断言两个分组计算结果是否相等
    # 应该始终返回数据帧！
    tm.assert_frame_equal(res, exp)


# 测试在不同方法下的分组操作，与上一个测试函数类似，但是包含了更多的方法调用
def test_groupby_selection_other_methods(df):
    # 创建一个具有时间索引的日期范围
    rng = date_range("2014", periods=len(df))
    # 设置数据帧的列名为 "foo"，并将索引设置为创建的日期范围
    df.columns.name = "foo"
    df.index = rng

    # 对数据帧按照 "A" 列进行分组，选择 "C" 列
    g = df.groupby(["A"])[["C"]]
    # 创建一个预期的分组对象，同样选择 "C" 列，并按照 "A" 列进行分组
    g_exp = df[["C"]].groupby(df["A"])

    # 对不仅仅是简单属性的方法进行断言，如 apply(lambda x: x.sum())
    tm.assert_frame_equal(g.apply(lambda x: x.sum()), g_exp.apply(lambda x: x.sum()))

    # 断言按日期重采样后的均值计算结果是否相等
    tm.assert_frame_equal(g.resample("D").mean(), g_exp.resample("D").mean())
    # 断言按日期重采样后的 ohlc 计算结果是否相等
    tm.assert_frame_equal(g.resample("D").ohlc(), g_exp.resample("D").ohlc())

    # 断言筛选出长度为 3 的分组结果是否相等
    tm.assert_frame_equal(
        g.filter(lambda x: len(x) == 3), g_exp.filter(lambda x: len(x) == 3)
    )
    # 创建时间戳对象列表，将字符串格式的时间转换为指定时间单位的时间戳
    idx2 = to_datetime(
        [
            "2016-08-31 22:08:12.000",
            "2016-08-31 22:09:12.200",
            "2016-08-31 22:20:12.400",
        ]
    ).as_unit(unit)

    # 创建包含测试数据的数据帧，包括量化数据和时间戳数据
    test_data = DataFrame(
        {"quant": [1.0, 1.0, 3.0], "quant2": [1.0, 1.0, 3.0], "time2": idx2}
    )

    # 生成时间范围，从指定的时间开始，按分钟频率生成13个时间点
    time2 = date_range("2016-08-31 22:08:00", periods=13, freq="1min", unit=unit)
    # 创建预期输出的数据帧，包含时间戳和相应的量化数据
    expected_output = DataFrame(
        {
            "time2": time2,
            "quant": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "quant2": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
    )

    # 根据时间2列进行分组，按1分钟频率对测试数据进行分组
    gb = test_data.groupby(Grouper(key="time2", freq="1min"))
    # 对分组后的数据进行计数，并重置索引
    result = gb.count().reset_index()

    # 使用测试工具断言数据帧的相等性，比较计算结果和预期输出
    tm.assert_frame_equal(result, expected_output)
def test_groupby_series_with_datetimeindex_month_name():
    # GH 48509
    # 创建一个时间序列，索引从 "2022-01-01" 开始，包含3个时间点，值为 [0, 1, 0]
    s = Series([0, 1, 0], index=date_range("2022-01-01", periods=3), name="jan")
    # 对时间序列进行分组，统计每组的数量
    result = s.groupby(s).count()
    # 创建预期的结果时间序列，值为 [2, 1]
    expected = Series([2, 1], name="jan")
    # 设置预期结果时间序列的索引名为 "jan"
    expected.index.name = "jan"
    # 使用测试工具比较实际结果和预期结果
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("test_series", [True, False])
@pytest.mark.parametrize(
    "kwarg, value, name, warn",
    [
        ("by", "a", 1, None),
        ("by", ["a"], (1,), None),
        ("level", 0, 1, None),
        ("level", [0], (1,), None),
    ],
)
def test_get_group_len_1_list_likes(test_series, kwarg, value, name, warn):
    # GH#25971
    # 创建一个DataFrame对象，包含列名为'b'的数据[3, 4, 5]，并使用索引名称为 'a' 的索引
    obj = DataFrame({"b": [3, 4, 5]}, index=Index([1, 1, 2], name="a"))
    # 如果test_series为True，只保留DataFrame中的'b'列
    if test_series:
        obj = obj["b"]
    # 根据给定的关键字参数进行分组操作，返回分组对象
    gb = obj.groupby(**{kwarg: value})
    # 从分组对象中获取特定分组的结果
    result = gb.get_group(name)
    # 根据test_series的值，创建预期的结果对象
    if test_series:
        expected = Series([3, 4], index=Index([1, 1], name="a"), name="b")
    else:
        expected = DataFrame({"b": [3, 4]}, index=Index([1, 1], name="a"))
    # 使用测试工具比较实际结果和预期结果
    tm.assert_equal(result, expected)


def test_groupby_ngroup_with_nan():
    # GH#50100
    # 创建一个DataFrame对象，包含列名为'a'的分类数据，其中包含一个NaN值，和列名为'b'的数据[1]
    df = DataFrame({"a": Categorical([np.nan]), "b": [1]})
    # 根据给定的列进行分组，并使用ngroup()方法返回组索引
    result = df.groupby(["a", "b"], dropna=False, observed=False).ngroup()
    # 创建预期的结果序列，值为 [0]
    expected = Series([0])
    # 使用测试工具比较实际结果和预期结果
    tm.assert_series_equal(result, expected)


def test_groupby_ffill_with_duplicated_index():
    # GH#43412
    # 创建一个DataFrame对象，包含列名为'a'的数据[1, 2, 3, 4, NaN, NaN]，并使用指定的索引[0, 1, 2, 0, 1, 2]
    df = DataFrame({"a": [1, 2, 3, 4, np.nan, np.nan]}, index=[0, 1, 2, 0, 1, 2])
    # 根据索引的级别进行分组，并对每组使用前向填充方法
    result = df.groupby(level=0).ffill()
    # 创建预期的结果DataFrame对象，包含列名为'a'的数据[1, 2, 3, 4, 2, 3]，并使用相同的索引
    expected = DataFrame({"a": [1, 2, 3, 4, 2, 3]}, index=[0, 1, 2, 0, 1, 2])
    # 使用测试工具比较实际结果和预期结果，允许检查数据类型
    tm.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize("test_series", [True, False])
def test_decimal_na_sort(test_series):
    # GH#54847
    # 在safe_sort中，捕获TypeError和decimal.InvalidOperation异常
    # 如果下面的断言引发异常，则只需捕获TypeError
    assert not isinstance(decimal.InvalidOperation, TypeError)
    # 创建一个DataFrame对象，包含列名为'key'和'value'的数据，其中'key'列包含Decimal和NaN值
    df = DataFrame(
        {
            "key": [Decimal(1), Decimal(1), None, None],
            "value": [Decimal(2), Decimal(3), Decimal(4), Decimal(5)],
        }
    )
    # 根据'key'列进行分组，并根据test_series的值选择是否保留'value'列
    gb = df.groupby("key", dropna=False)
    if test_series:
        gb = gb["value"]
    # 获取分组操作的结果索引
    result = gb._grouper.result_index
    # 创建预期的结果索引对象，包含Decimal和NaN值，索引名称为'key'
    expected = Index([Decimal(1), None], name="key")
    # 使用测试工具比较实际结果和预期结果
    tm.assert_index_equal(result, expected)


def test_groupby_dropna_with_nunique_unique():
    # GH#42016
    # 创建一个DataFrame对象，包含数据列表df，列名为'a', 'b', 'c', 'partner'
    df = [[1, 1, 1, "A"], [1, None, 1, "A"], [1, None, 2, "A"], [1, None, 3, "A"]]
    df_dropna = DataFrame(df, columns=["a", "b", "c", "partner"])
    # 根据'a', 'b', 'c'列进行分组，并对'partner'列应用'nunique'和'unique'聚合函数
    result = df_dropna.groupby(["a", "b", "c"], dropna=False).agg(
        {"partner": ["nunique", "unique"]}
    )

    # 创建预期的MultiIndex对象，包含多级索引，分别是(1, 1.0, 1), (1, NaN, 1), (1, NaN, 2), (1, NaN, 3)
    index = MultiIndex.from_tuples(
        [(1, 1.0, 1), (1, np.nan, 1), (1, np.nan, 2), (1, np.nan, 3)],
        names=["a", "b", "c"],
    )
    # 创建预期的MultiIndex对象的列名，包含('partner', 'nunique'), ('partner', 'unique')
    columns = MultiIndex.from_tuples([("partner", "nunique"), ("partner", "unique")])
    # 创建预期的 DataFrame 对象，包含指定的数据行和列，使用给定的索引和列名
    expected = DataFrame(
        [(1, ["A"]), (1, ["A"]), (1, ["A"]), (1, ["A"])], index=index, columns=columns
    )

    # 使用测试框架中的方法来比较两个 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试带有重复列的命名聚合函数 groupby_agg_namedagg_with_duplicate_columns
def test_groupby_agg_namedagg_with_duplicate_columns():
    # 创建一个包含列 'col1', 'col2', 'col3', 'col4', 'col5' 的 DataFrame
    df = DataFrame(
        {
            "col1": [2, 1, 1, 0, 2, 0],
            "col2": [4, 5, 36, 7, 4, 5],
            "col3": [3.1, 8.0, 12, 10, 4, 1.1],
            "col4": [17, 3, 16, 15, 5, 6],
            "col5": [-1, 3, -1, 3, -2, -1],
        }
    )

    # 对 DataFrame 进行 groupby 操作，按照列 'col1', 'col1', 'col2' 分组，并进行聚合操作
    result = df.groupby(by=["col1", "col1", "col2"], as_index=False).agg(
        # 新列 'new_col'，使用 'col1' 列的最小值作为聚合结果
        new_col=pd.NamedAgg(column="col1", aggfunc="min"),
        # 新列 'new_col1'，使用 'col1' 列的最大值作为聚合结果
        new_col1=pd.NamedAgg(column="col1", aggfunc="max"),
        # 新列 'new_col2'，使用 'col2' 列的计数作为聚合结果
        new_col2=pd.NamedAgg(column="col2", aggfunc="count"),
    )

    # 期望的结果 DataFrame，包含 'col1', 'col2', 'new_col', 'new_col1', 'new_col2' 列
    expected = DataFrame(
        {
            "col1": [0, 0, 1, 1, 2],
            "col2": [5, 7, 5, 36, 4],
            "new_col": [0, 0, 1, 1, 2],
            "new_col1": [0, 0, 1, 1, 2],
            "new_col2": [1, 1, 1, 1, 2],
        }
    )

    # 断言结果 DataFrame 和期望的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试多级索引代码 groupby_multi_index_codes
def test_groupby_multi_index_codes():
    # 创建一个包含列 'A', 'B', 'C' 的 DataFrame
    df = DataFrame(
        {"A": [1, 2, 3, 4], "B": [1, float("nan"), 2, float("nan")], "C": [2, 4, 6, 8]}
    )

    # 对 DataFrame 进行 groupby 操作，按照列 'A', 'B' 分组，并对分组后的数据进行求和操作
    df_grouped = df.groupby(["A", "B"], dropna=False).sum()

    # 获取分组后的索引
    index = df_grouped.index
    # 断言索引是否与由索引 DataFrame 生成的 MultiIndex 相等
    tm.assert_index_equal(index, MultiIndex.from_frame(index.to_frame()))
```