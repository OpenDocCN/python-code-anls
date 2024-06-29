# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_map.py`

```
# 导入必要的库和模块
from collections import (
    Counter,
    defaultdict,
)
from decimal import Decimal
import math

import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    bdate_range,
    date_range,
    isna,
    timedelta_range,
)
import pandas._testing as tm


# 定义测试函数 test_series_map_box_timedelta
def test_series_map_box_timedelta():
    # 创建一个时间增量序列，每隔1天1秒，周期为5，频率为每小时
    ser = Series(timedelta_range("1 day 1 s", periods=5, freq="h"))

    # 定义一个函数 f，用于返回时间增量的总秒数
    def f(x):
        return x.total_seconds()

    # 对序列进行映射操作，应用函数 f
    ser.map(f)


# 定义测试函数 test_map_callable
def test_map_callable(datetime_series):
    # 忽略所有 numpy 的错误状态
    with np.errstate(all="ignore"):
        # 断言映射后的序列与 np.sqrt 函数的结果相等
        tm.assert_series_equal(datetime_series.map(np.sqrt), np.sqrt(datetime_series))

    # 对序列应用 math.exp 函数，断言映射后的序列与 np.exp 函数的结果相等
    tm.assert_series_equal(datetime_series.map(math.exp), np.exp(datetime_series))

    # 创建一个空的对象序列 s
    s = Series(dtype=object, name="foo", index=Index([], name="bar"))
    # 对空序列应用 lambda 函数，预期结果与原序列相等
    rs = s.map(lambda x: x)
    tm.assert_series_equal(s, rs)

    # 检查所有的元数据 (GH 9322)
    assert s is not rs
    assert s.index is rs.index
    assert s.dtype == rs.dtype
    assert s.name == rs.name

    # 创建一个具有索引但无数据的序列 s
    s = Series(index=[1, 2, 3], dtype=np.float64)
    # 对序列应用 lambda 函数，预期结果与原序列相等
    rs = s.map(lambda x: x)
    tm.assert_series_equal(s, rs)


# 定义测试函数 test_map_same_length_inference_bug
def test_map_same_length_inference_bug():
    # 创建一个整数序列 s
    s = Series([1, 2])

    # 定义一个函数 f，用于返回元组 (x, x + 1)
    def f(x):
        return (x, x + 1)

    # 创建一个新的整数序列 s
    s = Series([1, 2, 3])
    # 对序列应用函数 f，预期结果为 [(1, 2), (2, 3), (3, 4)]
    result = s.map(f)
    expected = Series([(1, 2), (2, 3), (3, 4)])
    tm.assert_series_equal(result, expected)

    # 创建一个字符串序列 s
    s = Series(["foo,bar"])
    # 对序列应用 lambda 函数，预期结果为 [("foo", "bar")]
    result = s.map(lambda x: x.split(","))
    expected = Series([("foo", "bar")])
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_series_map_box_timestamps
def test_series_map_box_timestamps():
    # 创建一个日期时间序列，从 "2000-01-01" 开始，周期为3天
    ser = Series(date_range("1/1/2000", periods=3))

    # 定义一个函数 func，返回日期时间的小时、日和月份
    def func(x):
        return (x.hour, x.day, x.month)

    # 对序列应用函数 func，预期结果为 [(0, 1, 1), (0, 2, 1), (0, 3, 1)]
    result = ser.map(func)
    expected = Series([(0, 1, 1), (0, 2, 1), (0, 3, 1)])
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_map_series_stringdtype
def test_map_series_stringdtype(any_string_dtype, using_infer_string):
    # 在 StringDType 上进行映射测试，GH#40823
    ser1 = Series(
        data=["cat", "dog", "rabbit"],
        index=["id1", "id2", "id3"],
        dtype=any_string_dtype,
    )
    ser2 = Series(["id3", "id2", "id1", "id7000"], dtype=any_string_dtype)
    # 对 ser2 应用 ser1 的映射
    result = ser2.map(ser1)

    # 如果 ser2 的 dtype 是 object，则将 item 设置为 np.nan
    item = pd.NA
    if ser2.dtype == object:
        item = np.nan

    # 预期的结果
    expected = Series(data=["rabbit", "dog", "cat", item], dtype=any_string_dtype)
    if using_infer_string and any_string_dtype == "object":
        expected = expected.astype("string[pyarrow_numpy]")

    tm.assert_series_equal(result, expected)


# 参数化测试函数，用于测试带有 NaN 值的分类数据映射
@pytest.mark.parametrize(
    "data, expected_dtype",
    [(["1-1", "1-1", np.nan], "category"), (["1-1", "1-2", np.nan], object)],
)
def test_map_categorical_with_nan_values(data, expected_dtype, using_infer_string):
    # GH 20714 bug fixed in: GH 24275
    def func(val):
        return val.split("-")[0]

    # 创建一个分类类型的序列 s
    s = Series(data, dtype="category")
    # 对 Series 对象 s 应用函数 func，并在遇到缺失值时忽略处理
    result = s.map(func, na_action="ignore")
    
    # 如果使用 infer_string 且期望的数据类型是 object 类型，则将期望的数据类型更改为 "string[pyarrow_numpy]"
    if using_infer_string and expected_dtype == object:
        expected_dtype = "string[pyarrow_numpy]"
    
    # 创建一个包含字符串 "1", "1", 和 np.nan 的 Series 对象，其数据类型为期望的数据类型
    expected = Series(["1", "1", np.nan], dtype=expected_dtype)
    
    # 使用测试工具（如 pandas 的 assert_series_equal 函数）比较 result 和 expected 两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
# 测试空整数序列映射
def test_map_empty_integer_series():
    # 标识 Issue GH52384
    s = Series([], dtype=int)  # 创建一个空的整数类型的 Series 对象
    result = s.map(lambda x: x)  # 对该 Series 进行映射操作，lambda 函数无实际效果
    tm.assert_series_equal(result, s)  # 断言映射结果与原始 Series 相等


# 测试带有日期时间索引的空整数序列映射
def test_map_empty_integer_series_with_datetime_index():
    # 标识 Issue GH 21245
    s = Series([], index=date_range(start="2018-01-01", periods=0), dtype=int)
    # 创建一个带有空日期时间索引的空整数类型的 Series 对象
    result = s.map(lambda x: x)  # 对该 Series 进行映射操作，lambda 函数无实际效果
    tm.assert_series_equal(result, s)  # 断言映射结果与原始 Series 相等


# 测试简单字符串调用与 astype 相同的映射
def test_map_simple_str_callables_same_as_astype(
    string_series, func, using_infer_string
):
    # 测试确保我们首先逐行评估，然后才进行向量化评估
    result = string_series.map(func)  # 对字符串类型的 Series 应用给定的函数
    expected = string_series.astype(
        str if not using_infer_string else "string[pyarrow_numpy]"
    )  # 根据是否使用推断类型，将 Series 转换为相应类型
    tm.assert_series_equal(result, expected)  # 断言映射结果与预期结果相等


# 测试列表输入引发异常
def test_list_raises(string_series):
    with pytest.raises(TypeError, match="'list' object is not callable"):
        string_series.map([lambda x: x])  # 尝试将列表作为映射函数，预期引发 TypeError 异常


# 测试映射函数的基本功能
def test_map():
    data = {
        "A": [0.0, 1.0, 2.0, 3.0, 4.0],
        "B": [0.0, 1.0, 0.0, 1.0, 0.0],
        "C": ["foo1", "foo2", "foo3", "foo4", "foo5"],
        "D": bdate_range("1/1/2009", periods=5),
    }

    source = Series(data["B"], index=data["C"])  # 创建一个 Series 对象作为映射的源
    target = Series(data["C"][:4], index=data["D"][:4])  # 创建一个 Series 对象作为映射的目标

    merged = target.map(source)  # 使用目标 Series 对源 Series 进行映射

    for k, v in merged.items():  # 遍历映射后的结果
        assert v == source[target[k]]  # 断言映射结果与源 Series 对目标索引的映射结果相等

    # 输入也可以是一个字典
    merged = target.map(source.to_dict())  # 使用目标 Series 对源 Series 转换为字典后进行映射

    for k, v in merged.items():  # 再次遍历映射后的结果
        assert v == source[target[k]]  # 断言映射结果与源 Series 对目标索引的映射结果相等


# 测试日期时间映射
def test_map_datetime(datetime_series):
    # 对日期时间类型的 Series 进行映射操作
    result = datetime_series.map(lambda x: x * 2)  # 将 Series 中的每个元素乘以2
    tm.assert_series_equal(result, datetime_series * 2)  # 断言映射结果与预期结果相等


# 测试类别类型的映射
def test_map_category():
    # 标识 Issue GH 10324
    a = Series([1, 2, 3, 4])  # 创建一个整数类型的 Series
    b = Series(["even", "odd", "even", "odd"], dtype="category")  # 创建一个类别类型的 Series
    c = Series(["even", "odd", "even", "odd"])  # 创建一个普通字符串类型的 Series

    exp = Series(["odd", "even", "odd", np.nan], dtype="category")  # 期望的类别类型映射结果
    tm.assert_series_equal(a.map(b), exp)  # 断言类别类型的映射结果与预期结果相等
    exp = Series(["odd", "even", "odd", np.nan])  # 期望的普通字符串类型映射结果
    tm.assert_series_equal(a.map(c), exp)  # 断言普通字符串类型的映射结果与预期结果相等


# 测试类别类型和数值索引的映射
def test_map_category_numeric():
    a = Series(["a", "b", "c", "d"])  # 创建一个字符串类型的 Series
    b = Series([1, 2, 3, 4], index=pd.CategoricalIndex(["b", "c", "d", "e"]))  # 创建一个带有类别索引的数值类型的 Series
    c = Series([1, 2, 3, 4], index=Index(["b", "c", "d", "e"]))  # 创建一个普通索引的数值类型的 Series

    exp = Series([np.nan, 1, 2, 3])  # 期望的映射结果
    tm.assert_series_equal(a.map(b), exp)  # 断言映射结果与预期结果相等
    tm.assert_series_equal(a.map(c), exp)  # 断言映射结果与预期结果相等


# 测试类别类型和字符串索引的映射
def test_map_category_string():
    a = Series(["a", "b", "c", "d"])  # 创建一个字符串类型的 Series
    b = Series(
        ["B", "C", "D", "E"],  # 创建一个带有类别的字符串类型的 Series
        dtype="category",
        index=pd.CategoricalIndex(["b", "c", "d", "e"]),
    )
    c = Series(["B", "C", "D", "E"], index=Index(["b", "c", "d", "e"]))  # 创建一个普通字符串索引的 Series

    exp = Series(
        pd.Categorical([np.nan, "B", "C", "D"], categories=["B", "C", "D", "E"])
    )  # 期望的类别类型映射结果
    tm.assert_series_equal(a.map(b), exp)  # 断言映射结果与预期结果相等
    exp = Series([np.nan, "B", "C", "D"])  # 期望的普通字符串类型映射结果
    tm.assert_series_equal(a.map(c), exp)  # 断言映射结果与预期结果相等
def test_map_empty(request, index):
    # 如果索引是 MultiIndex 类型，则应用 xfail 标记，表示预期此测试会失败
    if isinstance(index, MultiIndex):
        request.applymarker(
            pytest.mark.xfail(
                reason="Initializing a Series from a MultiIndex is not supported"
            )
        )

    # 创建一个 Series 对象，使用给定的索引
    s = Series(index)
    # 使用空字典来映射 Series，返回结果
    result = s.map({})

    # 创建一个预期的 Series 对象，所有值为 NaN，索引与原始 Series 相同
    expected = Series(np.nan, index=s.index)
    # 使用 pytest 的工具函数比较两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


def test_map_compat():
    # 相关问题报告编号 GH 8024
    # 创建一个布尔型 Series，指定索引
    s = Series([True, True, False], index=[1, 2, 3])
    # 使用字典映射 True 和 False，返回结果
    result = s.map({True: "foo", False: "bar"})
    # 创建一个预期的 Series 对象，根据映射关系生成相应的值
    expected = Series(["foo", "foo", "bar"], index=[1, 2, 3])
    # 比较结果与预期的 Series 对象是否相等
    tm.assert_series_equal(result, expected)


def test_map_int():
    # 创建两个 Series 对象，一个包含浮点数，另一个包含整数
    left = Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4})
    right = Series({1: 11, 2: 22, 3: 33})

    # 断言左边的 Series 对象的数据类型是 np.float64
    assert left.dtype == np.float64
    # 断言右边的 Series 对象的数据类型是 np.integer 的子类
    assert issubclass(right.dtype.type, np.integer)

    # 使用右边的 Series 对象对左边的 Series 进行映射
    merged = left.map(right)
    # 断言合并后的 Series 对象的数据类型是 np.float64
    assert merged.dtype == np.float64
    # 断言合并后的 Series 对象中 "d" 索引位置的值是 NaN
    assert isna(merged["d"])
    # 断言合并后的 Series 对象中 "c" 索引位置的值不是 NaN
    assert not isna(merged["c"])


def test_map_type_inference():
    # 创建一个整数类型的 Series 对象
    s = Series(range(3))
    # 使用 lambda 函数进行映射，返回结果
    s2 = s.map(lambda x: np.where(x == 0, 0, 1))
    # 断言 s2 的数据类型是 np.integer 的子类
    assert issubclass(s2.dtype.type, np.integer)


def test_map_decimal(string_series):
    # 使用 lambda 函数将 string_series 中的每个元素转换为 Decimal 类型
    result = string_series.map(lambda x: Decimal(str(x)))
    # 断言结果的数据类型是 np.object_
    assert result.dtype == np.object_
    # 断言结果的第一个元素是 Decimal 类型
    assert isinstance(result.iloc[0], Decimal)


def test_map_na_exclusion():
    # 创建一个包含 NaN 的 Series 对象
    s = Series([1.5, np.nan, 3, np.nan, 5])

    # 使用 lambda 函数将每个元素乘以 2，忽略 NaN 值
    result = s.map(lambda x: x * 2, na_action="ignore")
    # 创建一个预期的 Series 对象，每个元素乘以 2
    exp = s * 2
    # 使用 pytest 的工具函数比较两个 Series 对象是否相等
    tm.assert_series_equal(result, exp)


def test_map_dict_with_tuple_keys():
    """
    由于 v0.14.0 版本中的新 MultiIndex-ing 行为，
    传递给 map 的具有元组键的字典被转换为多重索引，
    阻止了正确映射元组值的情况。
    """
    # 相关问题报告编号 GH 18496
    # 创建一个 DataFrame 对象，包含包含元组的 Series 对象
    df = DataFrame({"a": [(1,), (2,), (3, 4), (5, 6)]})
    # 创建一个字典，将元组映射到对应的标签
    label_mappings = {(1,): "A", (2,): "B", (3, 4): "A", (5, 6): "B"}

    # 使用 map 函数将 "a" 列中的元组映射为标签，并赋值给新列 "labels"
    df["labels"] = df["a"].map(label_mappings)
    # 创建一个预期的 Series 对象，包含预期的标签值
    df["expected_labels"] = Series(["A", "B", "A", "B"], index=df.index)
    # 使用 pytest 的工具函数比较 "labels" 列和 "expected_labels" 列是否相等，不检查列名
    tm.assert_series_equal(df["labels"], df["expected_labels"], check_names=False)


def test_map_counter():
    # 创建一个字符串类型的 Series 对象，指定索引
    s = Series(["a", "b", "c"], index=[1, 2, 3])
    # 创建一个 Counter 对象，用于计数
    counter = Counter()
    counter["b"] = 5
    counter["c"] += 1
    # 使用 Counter 对象将 Series 中的值映射为计数值，返回结果
    result = s.map(counter)
    # 创建一个预期的 Series 对象，包含映射后的计数值
    expected = Series([0, 5, 1], index=[1, 2, 3])
    # 使用 pytest 的工具函数比较两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


def test_map_defaultdict():
    # 创建一个整数类型的 Series 对象，指定索引
    s = Series([1, 2, 3], index=["a", "b", "c"])
    # 创建一个 defaultdict 对象，指定默认值生成函数
    default_dict = defaultdict(lambda: "blank")
    default_dict[1] = "stuff"
    # 使用 defaultdict 对象将 Series 中的值映射为默认值，返回结果
    result = s.map(default_dict)
    # 创建一个预期的 Series 对象，包含映射后的值
    expected = Series(["stuff", "blank", "blank"], index=["a", "b", "c"])
    # 使用 pytest 的工具函数比较两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


def test_map_dict_na_key():
    # https://github.com/pandas-dev/pandas/issues/17648
    # 检查 np.nan 键是否被适当映射
    # 创建一个包含 np.nan 的 Series 对象
    s = Series([1, 2, np.nan])
    # 创建一个字典，映射整数键到相应的标签
    expected = Series(["a", "b", "c"])
    # 使用字典将 Series 中的值映射为标签，返回结果
    result = s.map({1: "a", 2: "b", np.nan: "c"})
    # 使用测试工具库中的函数来比较两个序列是否相等
    tm.assert_series_equal(result, expected)
# GH 48813
# 创建一个包含数字和NaN的Series对象
s = Series([1, 2, np.nan])
# 使用lambda函数作为默认值生成一个defaultdict对象，用于处理缺失键
default_map = defaultdict(lambda: "missing", {1: "a", 2: "b", np.nan: "c"})
# 使用Series的map方法，根据default_map映射对应的值到结果Series中，处理缺失值的策略由na_action参数决定
result = s.map(default_map, na_action=na_action)
# 创建预期的Series对象，根据na_action参数确定NaN值的处理方式
expected = Series({0: "a", 1: "b", 2: "c" if na_action is None else np.nan})
# 使用pandas的测试工具验证结果Series与预期Series是否相等
tm.assert_series_equal(result, expected)


# GH 48813
# 创建一个包含数字和NaN的Series对象
s = Series([1, 2, np.nan])
# 使用lambda函数作为默认值生成一个defaultdict对象，用于处理缺失键
default_map = defaultdict(lambda: "missing", {1: "a", 2: "b", 3: "c"})
# 使用Series的map方法，根据default_map映射对应的值到结果Series中，处理缺失值的策略由na_action参数决定
result = s.map(default_map, na_action=na_action)
# 创建预期的Series对象，根据na_action参数确定NaN值的处理方式
expected = Series({0: "a", 1: "b", 2: "missing" if na_action is None else np.nan})
# 使用pandas的测试工具验证结果Series与预期Series是否相等
tm.assert_series_equal(result, expected)


# GH 48813
# 创建一个包含数字和NaN的Series对象
s = Series([1, 2, np.nan])
# 使用lambda函数作为默认值生成一个defaultdict对象，用于处理缺失键
default_map = defaultdict(lambda: "missing", {1: "a", 2: "b", np.nan: "c"})
# 复制default_map以便后续验证是否被修改
expected_default_map = default_map.copy()
# 使用Series的map方法，根据default_map映射对应的值到结果Series中，但不返回结果
s.map(default_map, na_action=na_action)
# 验证原始default_map对象是否未被修改
assert default_map == expected_default_map


# GH#47527
# 创建一个包含数字和NaN的Series对象
ser = Series([1, np.nan, 2])
# 根据传入的参数类型（字典或Series），创建映射关系
mapping = arg_func({1: 10, np.nan: 42})
# 使用Series的map方法，根据mapping映射对应的值到结果Series中，忽略NaN值
result = ser.map(mapping, na_action="ignore")
# 创建预期的Series对象，期望的映射结果
expected = Series([10, np.nan, np.nan])
# 使用pandas的测试工具验证结果Series与预期Series是否相等
tm.assert_series_equal(result, expected)


# GH#47527
# 创建一个包含数字和NaN的Series对象
ser = Series([1, np.nan, 2])
# 使用lambda函数作为默认值生成一个defaultdict对象，用于处理缺失键
mapping = defaultdict(int, {1: 10, np.nan: 42})
# 使用Series的map方法，根据mapping映射对应的值到结果Series中
result = ser.map(mapping)
# 创建预期的Series对象，期望的映射结果
expected = Series([10, 42, 0])
# 使用pandas的测试工具验证结果Series与预期Series是否相等
tm.assert_series_equal(result, expected)


# GH#47527
# 创建一个包含分类数据的Series对象
values = pd.Categorical([1, np.nan, 2], categories=[10, 1, 2])
# 根据传入的映射关系和na_action参数，使用Series的map方法对分类数据进行映射
result = ser.map({1: 10, np.nan: 42}, na_action=na_action)
# 创建预期的Series对象，期望的映射结果
tm.assert_series_equal(result, expected)


# Test Series.map with a dictionary subclass that defines __missing__,
# i.e. sets a default value (GH #15999).
# 创建一个自定义的字典子类，该子类定义了__missing__方法，用于处理缺失键
class DictWithMissing(dict):
    def __missing__(self, key):
        return "missing"

# 创建一个包含数字的Series对象
s = Series([1, 2, 3])
# 使用自定义的字典子类创建一个映射关系
dictionary = DictWithMissing({3: "three"})
# 使用Series的map方法，根据dictionary映射对应的值到结果Series中
result = s.map(dictionary)
# 创建预期的Series对象，期望的映射结果
expected = Series(["missing", "missing", "three"])
# 使用pandas的测试工具验证结果Series与预期Series是否相等
tm.assert_series_equal(result, expected)


# 创建一个自定义的字典子类，该子类未定义__missing__方法
class DictWithoutMissing(dict):
    pass

# 创建一个包含数字的Series对象
s = Series([1, 2, 3])
# 使用未定义__missing__方法的自定义字典子类创建一个映射关系
dictionary = DictWithoutMissing({3: "three"})
# 使用Series的map方法，根据dictionary映射对应的值到结果Series中
result = s.map(dictionary)
# 创建预期的Series对象，期望的映射结果
expected = Series([np.nan, np.nan, "three"])
# 使用pandas的测试工具验证结果Series与预期Series是否相等
tm.assert_series_equal(result, expected)


# https://github.com/pandas-dev/pandas/issues/29733
# Check collections.abc.Mapping support as mapper for Series.map
# 创建一个包含数字的Series对象
s = Series([1, 2, 3])
    # 创建一个非字典映射子类的实例，用于模拟不是字典的对象
    not_a_dictionary = non_dict_mapping_subclass({3: "three"})
    
    # 调用 Series 对象的 map 方法，使用上面创建的非字典映射对象进行映射操作
    result = s.map(not_a_dictionary)
    
    # 创建一个预期的 Series 对象，包含了预期的映射结果
    expected = Series([np.nan, np.nan, "three"])
    
    # 使用测试模块 tm 中的 assert_series_equal 方法，比较 result 和 expected 两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
# 测试使用非字典映射子类作为 Series.map 的映射器
def test_map_abc_mapping_with_missing(non_dict_mapping_subclass):
    # https://github.com/pandas-dev/pandas/issues/29733
    # 检查 collections.abc.Mapping 是否支持作为 Series.map 的映射器
    class NonDictMappingWithMissing(non_dict_mapping_subclass):
        # 自定义 __missing__ 方法，处理缺失键返回 "missing"
        def __missing__(self, key):
            return "missing"

    # 创建一个 Series 对象
    s = Series([1, 2, 3])
    # 创建一个非字典映射子类的实例
    not_a_dictionary = NonDictMappingWithMissing({3: "three"})
    # 使用非字典映射子类对 Series 进行映射操作
    result = s.map(not_a_dictionary)
    # __missing__ 是字典的概念，不是 Mapping 的概念，
    # 因此它不应该改变结果！
    expected = Series([np.nan, np.nan, "three"])
    # 断言 Series 相等
    tm.assert_series_equal(result, expected)


# 测试对 datetime64 数据类型进行映射
def test_map_box_dt64(unit):
    # 创建 Timestamp 对象列表
    vals = [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-02")]
    # 转换成指定单位的 Series
    ser = Series(vals).dt.as_unit(unit)
    # 断言 Series 的数据类型
    assert ser.dtype == f"datetime64[{unit}]"
    # 对 Series 中的每个元素执行映射操作
    res = ser.map(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}")
    # 期望的结果 Series
    exp = Series(["Timestamp_1_None", "Timestamp_2_None"])
    # 断言 Series 相等
    tm.assert_series_equal(res, exp)


# 测试对带有时区信息的 datetime64 数据类型进行映射
def test_map_box_dt64tz(unit):
    # 创建带时区信息的 Timestamp 对象列表
    vals = [
        pd.Timestamp("2011-01-01", tz="US/Eastern"),
        pd.Timestamp("2011-01-02", tz="US/Eastern"),
    ]
    # 转换成指定单位和时区的 Series
    ser = Series(vals).dt.as_unit(unit)
    # 断言 Series 的数据类型
    assert ser.dtype == f"datetime64[{unit}, US/Eastern]"
    # 对 Series 中的每个元素执行映射操作
    res = ser.map(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}")
    # 期望的结果 Series
    exp = Series(["Timestamp_1_US/Eastern", "Timestamp_2_US/Eastern"])
    # 断言 Series 相等
    tm.assert_series_equal(res, exp)


# 测试对 timedelta64 数据类型进行映射
def test_map_box_td64(unit):
    # 创建 Timedelta 对象列表
    vals = [pd.Timedelta("1 days"), pd.Timedelta("2 days")]
    # 转换成指定单位的 Series
    ser = Series(vals).dt.as_unit(unit)
    # 断言 Series 的数据类型
    assert ser.dtype == f"timedelta64[{unit}]"
    # 对 Series 中的每个元素执行映射操作
    res = ser.map(lambda x: f"{type(x).__name__}_{x.days}")
    # 期望的结果 Series
    exp = Series(["Timedelta_1", "Timedelta_2"])
    # 断言 Series 相等
    tm.assert_series_equal(res, exp)


# 测试对 Period 数据类型进行映射
def test_map_box_period():
    # 创建 Period 对象列表
    vals = [pd.Period("2011-01-01", freq="M"), pd.Period("2011-01-02", freq="M")]
    # 创建 Period 对象的 Series
    ser = Series(vals)
    # 断言 Series 的数据类型
    assert ser.dtype == "Period[M]"
    # 对 Series 中的每个元素执行映射操作
    res = ser.map(lambda x: f"{type(x).__name__}_{x.freqstr}")
    # 期望的结果 Series
    exp = Series(["Period_M", "Period_M"])
    # 断言 Series 相等
    tm.assert_series_equal(res, exp)


# 测试对分类数据进行映射
def test_map_categorical(na_action, using_infer_string):
    # 创建有序的分类数据
    values = pd.Categorical(list("ABBABCD"), categories=list("DCBA"), ordered=True)
    # 创建一个带有分类数据的 Series
    s = Series(values, name="XX", index=list("abcdefg"))

    # 对 Series 中的每个元素执行映射操作，转换为小写字母
    result = s.map(lambda x: x.lower(), na_action=na_action)
    # 期望的结果 Series
    exp_values = pd.Categorical(list("abbabcd"), categories=list("dcba"), ordered=True)
    exp = Series(exp_values, name="XX", index=list("abcdefg"))
    # 断言 Series 相等
    tm.assert_series_equal(result, exp)
    # 断言分类数据相等
    tm.assert_categorical_equal(result.values, exp_values)

    # 对 Series 中的每个元素执行映射操作，转换为固定值 "A"
    result = s.map(lambda x: "A", na_action=na_action)
    # 期望的结果 Series
    exp = Series(["A"] * 7, name="XX", index=list("abcdefg"))
    # 断言 Series 相等
    tm.assert_series_equal(result, exp)
    # 断言数据类型为 object 或者 string（根据 using_infer_string 参数）
    assert result.dtype == object if not using_infer_string else "string"
    (
        # 第一个元素是一个列表，包含一个 None 对象和一个名为 "XX" 的 Series 对象
        [None, Series(["A", "B", "nan"], name="XX")],
        # 第二个元素是一个列表，包含两个对象
        [
            # 第一个对象是字符串 "ignore"
            "ignore",
            # 第二个对象是一个 Series 对象，包含字符串 "A", "B", 和 np.nan
            # Series 的名称为 "XX"，数据类型为 pd.CategoricalDtype，有序，类别为 ['D', 'C', 'B', 'A']
            Series(
                ["A", "B", np.nan],
                name="XX",
                dtype=pd.CategoricalDtype(list("DCBA"), True),
            ),
        ],
    ),
# 定义测试函数，用于测试在不同的缺失数据处理方式下，pd.Series 对象的 map 方法的行为
def test_map_categorical_na_action(na_action, expected):
    # 创建一个有序的分类数据类型，包含字符集合"DCBA"
    dtype = pd.CategoricalDtype(list("DCBA"), ordered=True)
    # 创建包含值"A", "B"以及一个缺失值(np.nan)的分类数据
    values = pd.Categorical(list("AB") + [np.nan], dtype=dtype)
    # 创建 Series 对象，命名为"XX"
    s = Series(values, name="XX")
    # 使用 map 方法对 Series 进行映射，传入参数 na_action 用于处理缺失值
    result = s.map(str, na_action=na_action)
    # 断言 Series 对象 result 与期望结果 expected 相等
    tm.assert_series_equal(result, expected)


# 定义测试函数，用于测试在带有时区信息的日期时间索引的 Series 对象上使用 map 方法
def test_map_datetimetz():
    # 创建一个包含从"2011-01-01"到"2011-01-02"每小时的日期时间索引，且时区为"Asia/Tokyo"
    values = date_range("2011-01-01", "2011-01-02", freq="h").tz_localize("Asia/Tokyo")
    # 创建 Series 对象，命名为"XX"
    s = Series(values, name="XX")

    # 对 Series 使用 map 方法，对每个元素执行 lambda 函数，增加一天的偏移量
    result = s.map(lambda x: x + pd.offsets.Day())
    # 创建期望的 Series 对象，包含从"2011-01-02"到"2011-01-03"每小时的日期时间索引，时区为"Asia/Tokyo"
    exp_values = date_range("2011-01-02", "2011-01-03", freq="h").tz_localize("Asia/Tokyo")
    exp = Series(exp_values, name="XX")
    # 断言 Series 对象 result 与期望结果 exp 相等
    tm.assert_series_equal(result, exp)

    # 对 Series 使用 map 方法，对每个元素执行 lambda 函数，获取其小时数
    result = s.map(lambda x: x.hour)
    # 创建期望的 Series 对象，包含0到23的整数，以及一个0，数据类型为 np.int64
    exp = Series(list(range(24)) + [0], name="XX", dtype=np.int64)
    # 断言 Series 对象 result 与期望结果 exp 相等
    tm.assert_series_equal(result, exp)

    # 定义一个非向量化的函数 f，接受一个 pd.Timestamp 对象，并返回其时区信息的字符串表示
    def f(x):
        if not isinstance(x, pd.Timestamp):
            raise ValueError
        return str(x.tz)

    # 对 Series 使用 map 方法，应用函数 f
    result = s.map(f)
    # 创建期望的 Series 对象，包含"Asia/Tokyo"字符串，重复25次
    exp = Series(["Asia/Tokyo"] * 25, name="XX")
    # 断言 Series 对象 result 与期望结果 exp 相等
    tm.assert_series_equal(result, exp)


# 使用 pytest 的参数化功能，定义多组参数化测试用例
@pytest.mark.parametrize(
    "vals,mapping,exp",
    [
        # 测试用例1：将列表["abc"]映射到字典{np.nan: "not NaN"}，期望结果包含3个 np.nan 和一个 "not NaN"
        (list("abc"), {np.nan: "not NaN"}, [np.nan] * 3 + ["not NaN"]),
        # 测试用例2：将列表["abc"]映射到字典{"a": "a letter"}，期望结果包含一个 "a letter" 和 3 个 np.nan
        (list("abc"), {"a": "a letter"}, ["a letter"] + [np.nan] * 3),
        # 测试用例3：将列表[0, 1, 2]映射到字典{0: 42}，期望结果包含一个 42 和 3 个 np.nan
        (list(range(3)), {0: 42}, [42] + [np.nan] * 3),
    ],
)
# 定义测试函数，测试在不同映射和缺失数据处理方式下，pd.Series 对象的 map 方法的行为
def test_map_missing_mixed(vals, mapping, exp, using_infer_string):
    # 创建 Series 对象，包含列表 vals 和一个 np.nan
    s = Series(vals + [np.nan])
    # 使用 map 方法，将 Series 中的值按照 mapping 进行映射
    result = s.map(mapping)
    # 创建期望的 Series 对象 exp
    exp = Series(exp)
    # 如果使用了推断字符串且 mapping 包含 {np.nan: "not NaN"}，则将最后一个元素设置为 np.nan
    if using_infer_string and mapping == {np.nan: "not NaN"}:
        exp.iloc[-1] = np.nan
    # 断言 Series 对象 result 与期望结果 exp 相等
    tm.assert_series_equal(result, exp)


# 定义测试函数，测试在具有日期时间索引的 Series 对象上调用 map 方法，映射为标量的行为
def test_map_scalar_on_date_time_index_aware_series():
    # 创建一个包含从"2020-01-01"到"2020-01-10"每天的日期时间索引，时区为"UTC"，数据类型为 np.float64
    series = Series(
        np.arange(10, dtype=np.float64),
        index=date_range("2020-01-01", periods=10, tz="UTC"),
        name="ts",
    )
    # 对 Series 的索引应用 map 方法，映射为常数1
    result = Series(series.index).map(lambda x: 1)
    # 创建期望的 Series 对象，每个元素为1，数据类型为 int64
    tm.assert_series_equal(result, Series(np.ones(len(series)), dtype="int64"))


# 定义测试函数，测试将浮点数映射为字符串时的精度问题
def test_map_float_to_string_precision():
    # 创建一个包含值1/3的 Series 对象，数据类型为 float64
    ser = Series(1 / 3)
    # 使用 map 方法，将每个元素转换为字符串，并转换为字典
    result = ser.map(lambda val: str(val)).to_dict()
    # 创建期望的字典，包含一个键值对{0: "0.3333333333333333"}
    expected = {0: "0.3333333333333333"}
    # 断言结果字典 result 与期望字典 expected 相等
    assert result == expected


# 定义测试函数，测试将字符串列表映射为 Timedelta 对象的行为
def test_map_to_timedelta():
    # 创建有效字符串列表，包含"00:00:01"和"00:00:02"
    list_of_valid_strings = ["00:00:01", "00:00:02"]
    # 将列表中的字符串转换为 Timedelta 对象，并创建 Series 对象 a
    a = pd.to_timedelta(list_of_valid_strings)
    # 创建 Series 对象，包含列表 list_of_valid_strings，将每个元素映射为 Timedelta 对象，并创建 Series 对象 b
    b = Series(list_of_valid_strings).map(pd.to_timedelta)
    # 断言 Series 对象 a 和 b 相等
    tm.assert_series_equal(Series(a), b)

    # 创建字符串列表，包含"00:00:01"、np.nan、pd.NaT和pd.NaT
    list_of_strings = ["00:00:01", np.nan, pd.NaT, pd.NaT]
    # 将列表中的字符串转换为 Timedelta 对象，并创建 Series 对象 a
    a = pd.to_timedelta(list_of_strings)
    # 创建 Series 对象 ser，包含列表 list_of_strings
    ser = Series(list_of_strings)
    # 使用 map 方法，将 ser 中的每个元素映射为 Timedelta 对象，并创建 Series 对象 b
    b = ser.map(pd.to_timedelta)
    # 断言 Series 对象 a 和 b 相等
    tm.assert_series_equal(Series(a), b)


# 定义测试函数，测试将 Series 对象中的元素映射为其类型的行为
```