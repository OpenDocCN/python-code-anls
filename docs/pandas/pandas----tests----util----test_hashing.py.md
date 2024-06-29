# `D:\src\scipysrc\pandas\pandas\tests\util\test_hashing.py`

```
# 导入所需的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 库

import pandas as pd  # 导入 Pandas 库
from pandas import (  # 从 Pandas 中导入多个子模块和函数
    DataFrame,
    Index,
    MultiIndex,
    Series,
    period_range,
    timedelta_range,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块
from pandas.core.util.hashing import hash_tuples  # 导入 Pandas 内部哈希函数
from pandas.util import (  # 导入 Pandas 实用工具函数
    hash_array,
    hash_pandas_object,
)


@pytest.fixture(  # 定义 Pytest 的测试夹具
    params=[  # 夹具参数化，包含不同类型的 Series 对象
        Series([1, 2, 3] * 3, dtype="int32"),
        Series([None, 2.5, 3.5] * 3, dtype="float32"),
        Series(["a", "b", "c"] * 3, dtype="category"),
        Series(["d", "e", "f"] * 3),
        Series([True, False, True] * 3),
        Series(pd.date_range("20130101", periods=9)),
        Series(pd.date_range("20130101", periods=9, tz="US/Eastern")),
        Series(timedelta_range("2000", periods=9)),
    ]
)
def series(request):  # 定义返回 Series 对象的测试夹具
    return request.param


@pytest.fixture(params=[True, False])  # 定义返回布尔值的测试夹具
def index(request):
    return request.param


def test_consistency():
    # 检查哈希函数是否在输入不变的情况下保持一致性
    # 这是哈希函数的基准测试
    result = hash_pandas_object(Index(["foo", "bar", "baz"]))
    expected = Series(
        np.array(
            [3600424527151052760, 1374399572096150070, 477881037637427054],
            dtype="uint64",
        ),
        index=["foo", "bar", "baz"],
    )
    tm.assert_series_equal(result, expected)


def test_hash_array(series):
    arr = series.values  # 获取 Series 对象的值作为数组
    tm.assert_numpy_array_equal(hash_array(arr), hash_array(arr))


@pytest.mark.parametrize("dtype", ["U", object])
def test_hash_array_mixed(dtype):
    result1 = hash_array(np.array(["3", "4", "All"]))  # 对数组进行哈希处理
    result2 = hash_array(np.array([3, 4, "All"], dtype=dtype))  # 对混合类型数组进行哈希处理

    tm.assert_numpy_array_equal(result1, result2)


@pytest.mark.parametrize("val", [5, "foo", pd.Timestamp("20130101")])
def test_hash_array_errors(val):
    msg = "must pass a ndarray-like"
    with pytest.raises(TypeError, match=msg):  # 检查传入参数不是数组的错误处理
        hash_array(val)


def test_hash_array_index_exception():
    # 检查是否抛出 TypeError 而不是 AttributeError
    obj = pd.DatetimeIndex(["2018-10-28 01:20:00"], tz="Europe/Berlin")

    msg = "Use hash_pandas_object instead"
    with pytest.raises(TypeError, match=msg):  # 检查索引对象的哈希处理异常
        hash_array(obj)


def test_hash_tuples():
    tuples = [(1, "one"), (1, "two"), (2, "one")]
    result = hash_tuples(tuples)  # 对元组列表进行哈希处理

    expected = hash_pandas_object(MultiIndex.from_tuples(tuples)).values  # 使用 Pandas 处理元组列表的哈希预期结果
    tm.assert_numpy_array_equal(result, expected)

    # 只需要支持 MultiIndex 和元组列表
    msg = "|".join(["object is not iterable", "zip argument #1 must support iteration"])
    with pytest.raises(TypeError, match=msg):  # 检查非迭代对象的错误处理
        hash_tuples(tuples[0])


@pytest.mark.parametrize("val", [5, "foo", pd.Timestamp("20130101")])
def test_hash_tuples_err(val):
    msg = "must be convertible to a list-of-tuples"
    with pytest.raises(TypeError, match=msg):  # 检查不可转换为元组列表的错误处理
        hash_tuples(val)


def test_multiindex_unique():
    # TODO: Add implementation for test_multiindex_unique
    # 创建一个多级索引对象 `mi`，其中包含四个元组 (118, 472), (236, 118), (51, 204), (102, 51)
    mi = MultiIndex.from_tuples([(118, 472), (236, 118), (51, 204), (102, 51)])
    # 使用断言验证多级索引 `mi` 是否唯一
    assert mi.is_unique is True
    
    # 调用 `hash_pandas_object` 函数对多级索引 `mi` 进行哈希处理，返回处理后的结果
    result = hash_pandas_object(mi)
    # 使用断言验证处理后的结果 `result` 是否唯一
    assert result.is_unique is True
def test_multiindex_objects():
    # 创建一个多级索引对象 `mi`
    mi = MultiIndex(
        levels=[["b", "d", "a"], [1, 2, 3]],
        codes=[[0, 1, 0, 2], [2, 0, 0, 1]],
        names=["col1", "col2"],
    )
    # 调用 `_sort_levels_monotonic()` 方法对多级索引进行单调排序
    recons = mi._sort_levels_monotonic()

    # 断言多级索引 `mi` 和排序后的 `recons` 相等
    assert mi.equals(recons)
    # 断言将 `mi` 和 `recons` 转换为索引对象后相等
    assert Index(mi.values).equals(Index(recons.values))


@pytest.mark.parametrize(
    "obj",
    [
        Series([1, 2, 3]),  # 创建包含整数的 Series 对象
        Series([1.0, 1.5, 3.2]),  # 创建包含浮点数的 Series 对象
        Series([1.0, 1.5, np.nan]),  # 创建包含 NaN 的 Series 对象
        Series([1.0, 1.5, 3.2], index=[1.5, 1.1, 3.3]),  # 创建带索引的 Series 对象
        Series(["a", "b", "c"]),  # 创建包含字符串的 Series 对象
        Series(["a", np.nan, "c"]),  # 创建包含 NaN 的 Series 对象
        Series(["a", None, "c"]),  # 创建包含 None 的 Series 对象
        Series([True, False, True]),  # 创建包含布尔值的 Series 对象
        Series(dtype=object),  # 创建指定数据类型为 object 的 Series 对象
        DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]}),  # 创建具有两列的 DataFrame 对象
        DataFrame(),  # 创建空的 DataFrame 对象
        DataFrame(np.full((10, 4), np.nan)),  # 创建填充 NaN 值的 DataFrame 对象
        DataFrame(  # 创建具有四列的 DataFrame 对象，其中包含不同的数据类型和日期索引
            {
                "A": [0.0, 1.0, 2.0, 3.0, 4.0],
                "B": [0.0, 1.0, 0.0, 1.0, 0.0],
                "C": Index(["foo1", "foo2", "foo3", "foo4", "foo5"], dtype=object),
                "D": pd.date_range("20130101", periods=5),
            }
        ),
        DataFrame(range(5), index=pd.date_range("2020-01-01", periods=5)),  # 创建具有日期索引的 DataFrame 对象
        Series(range(5), index=pd.date_range("2020-01-01", periods=5)),  # 创建具有日期索引的 Series 对象
        Series(period_range("2020-01-01", periods=10, freq="D")),  # 创建包含日期范围的 Series 对象
        Series(pd.date_range("20130101", periods=3, tz="US/Eastern")),  # 创建包含时区信息的 Series 对象
    ],
)
def test_hash_pandas_object(obj, index):
    # 使用 hash_pandas_object 函数计算对象 `obj` 的哈希值，考虑索引 `index`
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    # 断言两次哈希值计算的结果相等
    tm.assert_series_equal(a, b)


@pytest.mark.parametrize(
    "obj",
    [
        Series([1, 2, 3]),  # 创建包含整数的 Series 对象
        Series([1.0, 1.5, 3.2]),  # 创建包含浮点数的 Series 对象
        Series([1.0, 1.5, np.nan]),  # 创建包含 NaN 的 Series 对象
        Series([1.0, 1.5, 3.2], index=[1.5, 1.1, 3.3]),  # 创建带索引的 Series 对象
        Series(["a", "b", "c"]),  # 创建包含字符串的 Series 对象
        Series(["a", np.nan, "c"]),  # 创建包含 NaN 的 Series 对象
        Series(["a", None, "c"]),  # 创建包含 None 的 Series 对象
        Series([True, False, True]),  # 创建包含布尔值的 Series 对象
        DataFrame({"x": ["a", "b", "c"], "y": [1, 2, 3]}),  # 创建具有两列的 DataFrame 对象
        DataFrame(np.full((10, 4), np.nan)),  # 创建填充 NaN 值的 DataFrame 对象
        DataFrame(  # 创建具有四列的 DataFrame 对象，其中包含不同的数据类型和日期索引
            {
                "A": [0.0, 1.0, 2.0, 3.0, 4.0],
                "B": [0.0, 1.0, 0.0, 1.0, 0.0],
                "C": Index(["foo1", "foo2", "foo3", "foo4", "foo5"], dtype=object),
                "D": pd.date_range("20130101", periods=5),
            }
        ),
        DataFrame(range(5), index=pd.date_range("2020-01-01", periods=5)),  # 创建具有日期索引的 DataFrame 对象
        Series(range(5), index=pd.date_range("2020-01-01", periods=5)),  # 创建具有日期索引的 Series 对象
        Series(period_range("2020-01-01", periods=10, freq="D")),  # 创建包含日期范围的 Series 对象
        Series(pd.date_range("20130101", periods=3, tz="US/Eastern")),  # 创建包含时区信息的 Series 对象
    ],
)
def test_hash_pandas_object_diff_index_non_empty(obj):
    # 使用 hash_pandas_object 函数分别计算带索引和不带索引的对象 `obj` 的哈希值
    a = hash_pandas_object(obj, index=True)
    b = hash_pandas_object(obj, index=False)
    # 断言两个哈希值不完全相等
    assert not (a == b).all()
    [
        # 创建一个 pandas Index 对象，包含整数 1, 2, 3
        Index([1, 2, 3]),
        # 创建一个 pandas Index 对象，包含布尔值 True, False, True
        Index([True, False, True]),
        # 创建一个时间增量序列，起始于 "1 day"，包含两个时间点
        timedelta_range("1 day", periods=2),
        # 创建一个日期范围序列，从 "2020-01-01" 开始，频率为每天 ("D")，包含两个日期
        period_range("2020-01-01", freq="D", periods=2),
        # 创建一个 MultiIndex 对象，其中包含由 Cartesian 乘积生成的元组
        MultiIndex.from_product(
            [range(5), ["foo", "bar", "baz"], pd.date_range("20130101", periods=2)]
        ),
        # 创建一个 MultiIndex 对象，其第一级索引为指定的分类值，第二级索引为整数范围
        MultiIndex.from_product([pd.CategoricalIndex(list("aabc")), range(3)]),
    ],
def test_hash_pandas_index(obj, index):
    # 调用 hash_pandas_object 函数计算 Pandas 对象的哈希值，使用指定的索引
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    # 断言两次计算的哈希值相等，验证哈希函数的一致性
    tm.assert_series_equal(a, b)


def test_hash_pandas_series(series, index):
    # 调用 hash_pandas_object 函数计算 Pandas Series 的哈希值，使用指定的索引
    a = hash_pandas_object(series, index=index)
    b = hash_pandas_object(series, index=index)
    # 断言两次计算的哈希值相等，验证哈希函数的一致性
    tm.assert_series_equal(a, b)


def test_hash_pandas_series_diff_index(series):
    # 调用 hash_pandas_object 函数计算 Pandas Series 的哈希值
    a = hash_pandas_object(series, index=True)
    b = hash_pandas_object(series, index=False)
    # 断言两次计算的哈希值不全相等，验证哈希函数与索引的关系
    assert not (a == b).all()


@pytest.mark.parametrize("klass", [Index, Series])
@pytest.mark.parametrize("dtype", ["float64", "object"])
def test_hash_pandas_empty_object(klass, dtype, index):
    # 创建空的 Pandas 对象，根据数据类型和索引
    # 这些对象无论是否有索引，在定义上都是相同的。
    obj = klass([], dtype=dtype)
    # 调用 hash_pandas_object 函数计算 Pandas 对象的哈希值，使用指定的索引
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    # 断言两次计算的哈希值相等，验证哈希函数的一致性
    tm.assert_series_equal(a, b)


@pytest.mark.parametrize(
    "s1",
    [
        ["a", "b", "c", "d"],
        [1000, 2000, 3000, 4000],
        pd.date_range(0, periods=4),
    ],
)
@pytest.mark.parametrize("categorize", [True, False])
def test_categorical_consistency(s1, categorize):
    # 查看 gh-15143
    #
    # 检查分类数据的哈希值是否与其值一致，
    # 而不是其代码。这对任何数据类型的分类数据都应该有效。
    s1 = Series(s1)
    # 将 Series 转换为分类数据类型，并设置类别为原始 Series 的值
    s2 = s1.astype("category").cat.set_categories(s1)
    # 设置类别为原始 Series 值的反向列表
    s3 = s2.cat.set_categories(list(reversed(s1)))

    # 这些对象的哈希值应该完全相同。
    h1 = hash_pandas_object(s1, categorize=categorize)
    h2 = hash_pandas_object(s2, categorize=categorize)
    h3 = hash_pandas_object(s3, categorize=categorize)

    # 断言这三个哈希值相等，验证哈希函数的一致性
    tm.assert_series_equal(h1, h2)
    tm.assert_series_equal(h1, h3)


def test_categorical_with_nan_consistency(unit):
    # 生成一个日期时间索引
    dti = pd.date_range("2012-01-01", periods=5, name="B", unit=unit)
    # 创建一个分类数据，类别使用日期时间索引
    cat = pd.Categorical.from_codes([-1, 0, 1, 2, 3, 4], categories=dti)
    # 使用 hash_array 计算分类数据的哈希值，不进行类别化处理
    expected = hash_array(cat, categorize=False)

    # 创建一个时间戳并转换为指定的单位
    ts = pd.Timestamp("2012-01-01").as_unit(unit)
    # 创建另一个分类数据，类别为单个时间戳
    cat2 = pd.Categorical.from_codes([-1, 0], categories=[ts])
    # 使用 hash_array 计算分类数据的哈希值，不进行类别化处理
    result = hash_array(cat2, categorize=False)

    # 断言结果的第一个元素和第二个元素在期望的哈希值中
    assert result[0] in expected
    assert result[1] in expected


def test_pandas_errors():
    # 测试 Pandas 对象的错误情况
    msg = "Unexpected type for hashing"
    # 断言在处理 Pandas 时间戳时会抛出类型错误异常
    with pytest.raises(TypeError, match=msg):
        hash_pandas_object(pd.Timestamp("20130101"))


def test_hash_keys():
    # 使用不同的哈希键，同样的数据应该有不同的哈希值。
    #
    # 这仅对对象数据类型有影响。
    obj = Series(list("abc"))

    # 使用指定的哈希键计算 Pandas 对象的哈希值
    a = hash_pandas_object(obj, hash_key="9876543210123456")
    b = hash_pandas_object(obj, hash_key="9876543210123465")

    # 断言两次计算的哈希值不全相等，验证哈希键的影响
    assert (a != b).all()


def test_df_hash_keys():
    # 测试 DataFrame 版本的 test_hash_keys。
    # https://github.com/pandas-dev/pandas/issues/41404
    obj = DataFrame({"x": np.arange(3), "y": list("abc")})

    # 使用指定的哈希键计算 Pandas 对象的哈希值
    a = hash_pandas_object(obj, hash_key="9876543210123456")
    # 使用 hash_pandas_object 函数计算 DataFrame 或 Series 的哈希值，使用指定的哈希密钥
    b = hash_pandas_object(obj, hash_key="9876543210123465")
    
    # 断言：确保对象 a 和 b 中的所有元素都不相等，否则抛出异常
    assert (a != b).all()
def test_df_encoding():
    # 检查 DataFrame 是否能识别可选的编码方式。
    # https://github.com/pandas-dev/pandas/issues/41404
    # https://github.com/pandas-dev/pandas/pull/42049
    obj = DataFrame({"x": np.arange(3), "y": list("a+c")})

    # 使用 UTF-8 编码对 DataFrame 进行哈希处理
    a = hash_pandas_object(obj, encoding="utf8")
    # 使用 UTF-7 编码对 DataFrame 进行哈希处理
    b = hash_pandas_object(obj, encoding="utf7")

    # 注意：在 UTF-7 中，"+" 被编码为 "+-"
    assert a[0] == b[0]
    assert a[1] != b[1]
    assert a[2] == b[2]


def test_invalid_key():
    # 这仅适用于对象类型数据。
    msg = "key should be a 16-byte string encoded"

    # 使用错误的哈希键进行 DataFrame 的哈希处理，预期会引发 ValueError 错误
    with pytest.raises(ValueError, match=msg):
        hash_pandas_object(Series(list("abc")), hash_key="foo")


def test_already_encoded(index):
    # 如果已经编码过，则正常运行。
    obj = Series(list("abc")).str.encode("utf8")
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    # 断言两个哈希后的 Series 相等
    tm.assert_series_equal(a, b)


def test_alternate_encoding(index):
    obj = Series(list("abc"))
    a = hash_pandas_object(obj, index=index)
    b = hash_pandas_object(obj, index=index)
    # 断言两个哈希后的 Series 相等
    tm.assert_series_equal(a, b)


@pytest.mark.parametrize("l_exp", range(8))
@pytest.mark.parametrize("l_add", [0, 1])
def test_same_len_hash_collisions(l_exp, l_add):
    length = 2 ** (l_exp + 8) + l_add
    idx = np.array([str(i) for i in range(length)], dtype=object)

    # 对数组进行 UTF-8 哈希处理，并断言第一个和第二个元素不相等
    result = hash_array(idx, "utf8")
    assert not result[0] == result[1]


def test_hash_collisions():
    # 哈希冲突很糟糕。
    #
    # https://github.com/pandas-dev/pandas/issues/14711#issuecomment-264885726
    hashes = [
        "Ingrid-9Z9fKIZmkO7i7Cn51Li34pJm44fgX6DYGBNj3VPlOH50m7HnBlPxfIwFMrcNJNMP6PSgLmwWnInciMWrCSAlLEvt7JkJl4IxiMrVbXSa8ZQoVaq5xoQPjltuJEfwdNlO6jo8qRRHvD8sBEBMQASrRa6TsdaPTPCBo3nwIBpE7YzzmyH0vMBhjQZLx1aCT7faSEx7PgFxQhHdKFWROcysamgy9iVj8DO2Fmwg1NNl93rIAqC3mdqfrCxrzfvIY8aJdzin2cHVzy3QUJxZgHvtUtOLxoqnUHsYbNTeq0xcLXpTZEZCxD4PGubIuCNf32c33M7HFsnjWSEjE2yVdWKhmSVodyF8hFYVmhYnMCztQnJrt3O8ZvVRXd5IKwlLexiSp4h888w7SzAIcKgc3g5XQJf6MlSMftDXm9lIsE1mJNiJEv6uY6pgvC3fUPhatlR5JPpVAHNSbSEE73MBzJrhCAbOLXQumyOXigZuPoME7QgJcBalliQol7YZ9",
        "Tim-b9MddTxOWW2AT1Py6vtVbZwGAmYCjbp89p8mxsiFoVX4FyDOF3wFiAkyQTUgwg9sVqVYOZo09Dh1AzhFHbgij52ylF0SEwgzjzHH8TGY8Lypart4p4onnDoDvVMBa0kdthVGKl6K0BDVGzyOXPXKpmnMF1H6rJzqHJ0HywfwS4XYpVwlAkoeNsiicHkJUFdUAhG229INzvIAiJuAHeJDUoyO4DCBqtoZ5TDend6TK7Y914yHlfH3g1WZu5LksKv68VQHJriWFYusW5e6ZZ6dKaMjTwEGuRgdT66iU5nqWTHRH8WSzpXoCFwGcTOwyuqPSe0fTe21DVtJn1FKj9F9nEnR9xOvJUO7E0piCIF4Ad9yAIDY4DBimpsTfKXCu1vdHpKYerzbndfuFe5AhfMduLYZJi5iAw8qKSwR5h86ttXV0Mc0QmXz8dsRvDgxjXSmupPxBggdlqUlC828hXiTPD7am0yETBV0F3bEtvPiNJfremszcV8NcqAoARMe",
    ]

    # 这两者应该是不同的。
    result1 = hash_array(np.asarray(hashes[0:1], dtype=object), "utf8")
    expected1 = np.array([14963968704024874985], dtype=np.uint64)
    tm.assert_numpy_array_equal(result1, expected1)

    result2 = hash_array(np.asarray(hashes[1:2], dtype=object), "utf8")
    # 创建一个 NumPy 数组，其中包含一个无符号 64 位整数，作为预期输出结果
    expected2 = np.array([16428432627716348016], dtype=np.uint64)
    # 使用测试工具（tm.assert_numpy_array_equal）比较 result2 和 expected2 数组是否相等
    tm.assert_numpy_array_equal(result2, expected2)
    
    # 将列表 hashes 转换为 NumPy 数组，并指定数据类型为 object，然后使用 "utf8" 编码计算其哈希值
    result = hash_array(np.asarray(hashes, dtype=object), "utf8")
    # 使用测试工具（tm.assert_numpy_array_equal）比较 result 和 np.concatenate([expected1, expected2], axis=0) 的数组是否相等
    tm.assert_numpy_array_equal(result, np.concatenate([expected1, expected2], axis=0))
@pytest.mark.parametrize(
    "data, result_data",
    [
        [[tuple("1"), tuple("2")], [10345501319357378243, 8331063931016360761]],
        [[(1,), (2,)], [9408946347443669104, 3278256261030523334]],
    ],
)
def test_hash_with_tuple(data, result_data):
    # GH#28969 array containing a tuple raises on call to arr.astype(str)
    #  apparently a numpy bug github.com/numpy/numpy/issues/9441
    
    # 创建包含数据的DataFrame对象，其中列名为"data"
    df = DataFrame({"data": data})
    # 对DataFrame对象进行哈希处理
    result = hash_pandas_object(df)
    # 期望的哈希结果
    expected = Series(result_data, dtype=np.uint64)
    # 断言两个Series对象是否相等
    tm.assert_series_equal(result, expected)


def test_hashable_tuple_args():
    # require that the elements of such tuples are themselves hashable
    # 创建包含数据的DataFrame对象，其中列名为"data"
    df3 = DataFrame(
        {
            "data": [
                (
                    1,
                    [],
                ),
                (
                    2,
                    {},
                ),
            ]
        }
    )
    # 使用pytest断言捕获预期的TypeError异常，错误信息包含'unhashable type: 'list''
    with pytest.raises(TypeError, match="unhashable type: 'list'"):
        # 对DataFrame对象进行哈希处理
        hash_pandas_object(df3)


def test_hash_object_none_key():
    # https://github.com/pandas-dev/pandas/issues/30887
    # 使用hash_key参数设置为None对Series对象进行哈希处理
    result = pd.util.hash_pandas_object(Series(["a", "b"]), hash_key=None)
    # 期望的哈希结果
    expected = Series([4578374827886788867, 17338122309987883691], dtype="uint64")
    # 断言两个Series对象是否相等
    tm.assert_series_equal(result, expected)
```