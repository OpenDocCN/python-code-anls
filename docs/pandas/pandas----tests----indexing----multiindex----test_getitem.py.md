# `D:\src\scipysrc\pandas\pandas\tests\indexing\multiindex\test_getitem.py`

```
import numpy as np  # 导入 numpy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas import (  # 从 pandas 库中导入以下模块：
    DataFrame,  # 数据帧对象，用于处理二维表格数据
    Index,  # 索引对象，用于管理轴标签
    MultiIndex,  # 多级索引对象，用于管理多维标签
    Series,  # 系列对象，用于表示一维数据结构
)
import pandas._testing as tm  # 导入 pandas 内部测试工具
from pandas.core.indexing import IndexingError  # 导入 pandas 索引错误模块

# ----------------------------------------------------------------------------
# test indexing of Series with multi-level Index
# ----------------------------------------------------------------------------

# 使用 pytest 的 parametrize 装饰器指定多个参数化的测试用例
@pytest.mark.parametrize(
    "access_method",
    [lambda s, x: s[:, x], lambda s, x: s.loc[:, x], lambda s, x: s.xs(x, level=1)],
)
@pytest.mark.parametrize(
    "level1_value, expected",
    [(0, Series([1], index=[0])), (1, Series([2, 3], index=[1, 2]))],
)
def test_series_getitem_multiindex(access_method, level1_value, expected):
    # GH 6018
    # series regression getitem with a multi-index
    # 多级索引的系列获取功能测试

    # 创建一个多级索引对象 mi
    mi = MultiIndex.from_tuples([(0, 0), (1, 1), (2, 1)], names=["A", "B"])
    # 创建一个系列对象 ser，设置其索引为 mi
    ser = Series([1, 2, 3], index=mi)
    # 设置期望结果的索引名称为 "A"
    expected.index.name = "A"

    # 调用给定的访问方法 access_method，获取结果
    result = access_method(ser, level1_value)
    # 使用测试工具 tm 检查结果是否与期望一致
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("level0_value", ["D", "A"])
def test_series_getitem_duplicates_multiindex(level0_value):
    # GH 5725 the 'A' happens to be a valid Timestamp so the doesn't raise
    # the appropriate error, only in PY3 of course!
    # 如果 'A' 是有效的时间戳，则不会引发相应的错误，仅适用于 PY3

    # 创建一个多级索引对象 index
    index = MultiIndex(
        levels=[[level0_value, "B", "C"], [0, 26, 27, 37, 57, 67, 75, 82]],
        codes=[[0, 0, 0, 1, 2, 2, 2, 2, 2, 2], [1, 3, 4, 6, 0, 2, 2, 3, 5, 7]],
        names=["tag", "day"],
    )
    # 使用 numpy 生成随机数据，创建一个数据帧对象 df
    arr = np.random.default_rng(2).standard_normal((len(index), 1))
    df = DataFrame(arr, index=index, columns=["val"])

    # 确认对缺失值的索引引发 KeyError 错误
    if level0_value != "A":
        with pytest.raises(KeyError, match=r"^'A'$"):
            df.val["A"]

    # 使用 pytest 检查对不存在的值 'X' 的索引是否引发 KeyError 错误
    with pytest.raises(KeyError, match=r"^'X'$"):
        df.val["X"]

    # 获取指定 level0_value 的结果
    result = df.val[level0_value]
    # 创建期望结果的系列对象 expected
    expected = Series(
        arr.ravel()[0:3], name="val", index=Index([26, 37, 57], name="day")
    )
    # 使用测试工具 tm 检查结果是否与期望一致
    tm.assert_series_equal(result, expected)


def test_series_getitem(multiindex_year_month_day_dataframe_random_data, indexer_sl):
    # 获取数据帧对象 multiindex_year_month_day_dataframe_random_data 中标签为 'A' 的系列对象 s
    s = multiindex_year_month_day_dataframe_random_data["A"]
    # 根据索引重新索引系列对象 s，得到期望的结果 expected
    expected = s.reindex(s.index[42:65])
    expected.index = expected.index.droplevel(0).droplevel(0)

    # 使用索引器 indexer_sl 获取结果 result
    result = indexer_sl(s)[2000, 3]
    # 使用测试工具 tm 检查结果是否与期望一致
    tm.assert_series_equal(result, expected)


def test_series_getitem_returns_scalar(
    multiindex_year_month_day_dataframe_random_data, indexer_sl
):
    # 获取数据帧对象 multiindex_year_month_day_dataframe_random_data 中标签为 'A' 的系列对象 s
    s = multiindex_year_month_day_dataframe_random_data["A"]
    # 获取 s.iloc[49] 作为期望的结果 expected
    expected = s.iloc[49]

    # 使用索引器 indexer_sl 获取结果 result
    result = indexer_sl(s)[2000, 3, 10]
    # 断言结果是否与期望一致
    assert result == expected


@pytest.mark.parametrize(
    "indexer,expected_error,expected_error_msg",
    [
        # 返回一个 lambda 函数，该函数接受参数 s，使用 s.__getitem__((2000, 3, 4)) 进行索引操作，预期会引发 KeyError 异常，匹配的异常信息应为 "\(2000, 3, 4\)"
        (lambda s: s.__getitem__((2000, 3, 4)), KeyError, r"^\(2000, 3, 4\)$"),
        
        # 返回一个 lambda 函数，该函数接受参数 s，使用 s[(2000, 3, 4)] 进行索引操作，预期会引发 KeyError 异常，匹配的异常信息应为 "\(2000, 3, 4\)"
        (lambda s: s[(2000, 3, 4)], KeyError, r"^\(2000, 3, 4\)$"),
        
        # 返回一个 lambda 函数，该函数接受参数 s，使用 s.loc[(2000, 3, 4)] 进行索引操作，预期会引发 KeyError 异常，匹配的异常信息应为 "\(2000, 3, 4\)"
        (lambda s: s.loc[(2000, 3, 4)], KeyError, r"^\(2000, 3, 4\)$"),
        
        # 返回一个 lambda 函数，该函数接受参数 s，使用 s.loc[(2000, 3, 4, 5)] 进行索引操作，预期会引发 IndexingError 异常，异常信息应为 "Too many indexers"
        (lambda s: s.loc[(2000, 3, 4, 5)], IndexingError, "Too many indexers"),
        
        # 返回一个 lambda 函数，该函数接受参数 s，使用 s.__getitem__(len(s)) 进行索引操作，预期会引发 KeyError 异常，异常信息匹配空字符串 ""
        (lambda s: s.__getitem__(len(s)), KeyError, ""),
        
        # 返回一个 lambda 函数，该函数接受参数 s，使用 s[len(s)] 进行索引操作，预期会引发 KeyError 异常，异常信息匹配空字符串 ""
        (lambda s: s[len(s)], KeyError, ""),
        
        # 返回一个 lambda 函数，该函数接受参数 s，使用 s.iloc[len(s)] 进行索引操作，预期会引发 IndexError 异常，异常信息为 "single positional indexer is out-of-bounds"
        (
            lambda s: s.iloc[len(s)],
            IndexError,
            "single positional indexer is out-of-bounds",
        ),
    ],
def test_series_getitem_indexing_errors(
    multiindex_year_month_day_dataframe_random_data,
    indexer,
    expected_error,
    expected_error_msg,
):
    # 从多级索引的DataFrame中获取列"A"作为Series
    s = multiindex_year_month_day_dataframe_random_data["A"]
    # 使用pytest检查是否会引发指定类型的错误，并验证错误消息是否匹配
    with pytest.raises(expected_error, match=expected_error_msg):
        indexer(s)


def test_series_getitem_corner_generator(
    multiindex_year_month_day_dataframe_random_data,
):
    # 从多级索引的DataFrame中获取列"A"作为Series
    s = multiindex_year_month_day_dataframe_random_data["A"]
    # 使用生成器表达式进行条件筛选，生成结果Series
    result = s[(x > 0 for x in s)]
    # 期望的结果是根据条件筛选后的Series
    expected = s[s > 0]
    # 使用pandas测试工具验证两个Series是否相等
    tm.assert_series_equal(result, expected)


# ----------------------------------------------------------------------------
# 测试DataFrame的多级索引情况下的索引操作
# ----------------------------------------------------------------------------


def test_getitem_simple(multiindex_dataframe_random_data):
    # 转置多级索引DataFrame
    df = multiindex_dataframe_random_data.T
    # 期望的结果是DataFrame的第一列值
    expected = df.values[:, 0]
    # 获取特定索引("foo", "one")的列值作为一维数组
    result = df["foo", "one"].values
    # 使用pandas测试工具验证两个数组几乎相等
    tm.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "indexer,expected_error_msg",
    [
        # 使用lambda函数获取指定列索引("foo", "four")，期望引发KeyError并验证错误消息
        (lambda df: df[("foo", "four")], r"^\('foo', 'four'\)$"),
        # 使用lambda函数获取不存在的列"foobar"，期望引发KeyError并验证错误消息
        (lambda df: df["foobar"], r"^'foobar'$"),
    ],
)
def test_frame_getitem_simple_key_error(
    multiindex_dataframe_random_data, indexer, expected_error_msg
):
    # 转置多级索引DataFrame
    df = multiindex_dataframe_random_data.T
    # 使用pytest检查是否会引发KeyError，并验证错误消息是否匹配
    with pytest.raises(KeyError, match=expected_error_msg):
        indexer(df)


def test_tuple_string_column_names():
    # 创建一个多级索引
    mi = MultiIndex.from_tuples([("a", "aa"), ("a", "ab"), ("b", "ba"), ("b", "bb")])
    # 创建DataFrame，列名为多级索引，行数据为范围值
    df = DataFrame([range(4), range(1, 5), range(2, 6)], columns=mi)
    # 在DataFrame中添加单级索引列
    df["single_index"] = 0

    # 复制DataFrame，并将列名转换为单级索引
    df_flat = df.copy()
    df_flat.columns = df_flat.columns.to_flat_index()
    # 在转换后的DataFrame中添加新的单级索引列
    df_flat["new_single_index"] = 0

    # 根据列名列表获取DataFrame的子集
    result = df_flat[[("a", "aa"), "new_single_index"]]
    # 期望的结果DataFrame，包含特定列名的数据
    expected = DataFrame(
        [[0, 0], [1, 0], [2, 0]], columns=Index([("a", "aa"), "new_single_index"])
    )
    # 使用pandas测试工具验证两个DataFrame是否相等
    tm.assert_frame_equal(result, expected)


def test_frame_getitem_multicolumn_empty_level():
    # 创建一个普通DataFrame，列名为多级索引
    df = DataFrame({"a": ["1", "2", "3"], "b": ["2", "3", "4"]})
    df.columns = [
        ["level1 item1", "level1 item2"],  # 设置多级索引的第一级
        ["", "level2 item2"],  # 设置多级索引的第二级
        ["level3 item1", "level3 item2"],  # 设置多级索引的第三级
    ]

    # 获取特定的多级索引列"level1 item1"
    result = df["level1 item1"]
    # 期望的结果DataFrame，包含特定的列名"level3 item1"，但不包含多级索引
    expected = DataFrame(
        [["1"], ["2"], ["3"]], index=df.index, columns=["level3 item1"]
    )
    # 使用pandas测试工具验证两个DataFrame是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "indexer,expected_slice",
    [
        # 使用lambda函数获取特定列索引"foo"，期望获取DataFrame的切片
        (lambda df: df["foo"], slice(3)),
        # 使用lambda函数获取特定列索引"bar"，期望获取DataFrame的切片
        (lambda df: df["bar"], slice(3, 5)),
        # 使用lambda函数使用.loc方法获取特定列索引"bar"，期望获取DataFrame的切片
        (lambda df: df.loc[:, "bar"], slice(3, 5)),
    ],
)
def test_frame_getitem_toplevel(
    multiindex_dataframe_random_data, indexer, expected_slice
):
    # 转置多级索引DataFrame
    df = multiindex_dataframe_random_data.T
    # 期望的结果是DataFrame根据预期切片获取的列
    expected = df.reindex(columns=df.columns[expected_slice])
    expected.columns = expected.columns.droplevel(0)
    # 使用indexer函数获取DataFrame的结果
    result = indexer(df)
    # 使用pandas测试工具验证两个DataFrame是否相等
    tm.assert_frame_equal(result, expected)
def test_frame_mixed_depth_get():
    # 创建一个包含多个数组的列表，每个数组代表DataFrame的多级索引的一部分
    arrays = [
        ["a", "top", "top", "routine1", "routine1", "routine2"],
        ["", "OD", "OD", "result1", "result2", "result1"],
        ["", "wx", "wy", "", "", ""],
    ]

    # 将多个数组打包并排序，创建多级索引对象
    tuples = sorted(zip(*arrays))
    index = MultiIndex.from_tuples(tuples)

    # 使用随机数生成器创建指定形状的DataFrame，使用多级索引作为列名
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 6)), columns=index)

    # 测试DataFrame的列索引取值
    result = df["a"]
    expected = df["a", "", ""].rename("a")
    tm.assert_series_equal(result, expected)

    # 再次测试DataFrame的列索引取值
    result = df["routine1", "result1"]
    expected = df["routine1", "result1", ""]
    expected = expected.rename(("routine1", "result1"))
    tm.assert_series_equal(result, expected)


def test_frame_getitem_nan_multiindex(nulls_fixture):
    # GH#29751
    # 在包含NaN值的多级索引上使用loc操作
    n = nulls_fixture  # 为了提高代码可读性
    cols = ["a", "b", "c"]

    # 创建一个带有NaN值的DataFrame，并设置多级索引
    df = DataFrame(
        [[11, n, 13], [21, n, 23], [31, n, 33], [41, n, 43]],
        columns=cols,
    ).set_index(["a", "b"])
    df["c"] = df["c"].astype("int64")

    # 使用loc对DataFrame进行切片操作，测试结果是否符合预期
    idx = (21, n)
    result = df.loc[:idx]
    expected = DataFrame([[11, n, 13], [21, n, 23]], columns=cols).set_index(["a", "b"])
    expected["c"] = expected["c"].astype("int64")
    tm.assert_frame_equal(result, expected)

    result = df.loc[idx:]
    expected = DataFrame(
        [[21, n, 23], [31, n, 33], [41, n, 43]], columns=cols
    ).set_index(["a", "b"])
    expected["c"] = expected["c"].astype("int64")
    tm.assert_frame_equal(result, expected)

    idx1, idx2 = (21, n), (31, n)
    result = df.loc[idx1:idx2]
    expected = DataFrame([[21, n, 23], [31, n, 33]], columns=cols).set_index(["a", "b"])
    expected["c"] = expected["c"].astype("int64")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "indexer,expected",
    [
        (
            # 第一个测试案例，列名为 ["b"], 值为 ["bar", np.nan]
            (["b"], ["bar", np.nan]),
            # 创建一个 DataFrame 对象，包含两行两列的整数数据
            (
                DataFrame(
                    [[2, 3], [5, 6]],
                    # 列名使用 MultiIndex 从元组创建，包括 ("b", "bar") 和 ("b", np.nan)
                    columns=MultiIndex.from_tuples([("b", "bar"), ("b", np.nan)]),
                    dtype="int64",
                )
            ),
        ),
        (
            # 第二个测试案例，列名为 ["a", "b"]
            (["a", "b"]),
            # 创建一个 DataFrame 对象，包含两行三列的整数数据
            (
                DataFrame(
                    [[1, 2, 3], [4, 5, 6]],
                    # 列名使用 MultiIndex 从元组创建，包括 ("a", "foo"), ("b", "bar") 和 ("b", np.nan)
                    columns=MultiIndex.from_tuples(
                        [("a", "foo"), ("b", "bar"), ("b", np.nan)]
                    ),
                    dtype="int64",
                )
            ),
        ),
        (
            # 第三个测试案例，列名为 ["b"]
            (["b"]),
            # 创建一个 DataFrame 对象，包含两行两列的整数数据
            (
                DataFrame(
                    [[2, 3], [5, 6]],
                    # 列名使用 MultiIndex 从元组创建，包括 ("b", "bar") 和 ("b", np.nan)
                    columns=MultiIndex.from_tuples([("b", "bar"), ("b", np.nan)]),
                    dtype="int64",
                )
            ),
        ),
        (
            # 第四个测试案例，列名为 ["b"], 值为 ["bar"]
            (["b"], ["bar"]),
            # 创建一个 DataFrame 对象，包含两行一列的整数数据
            (
                DataFrame(
                    [[2], [5]],
                    # 列名使用 MultiIndex 从元组创建，只包括 ("b", "bar")
                    columns=MultiIndex.from_tuples([("b", "bar")]),
                    dtype="int64",
                )
            ),
        ),
        (
            # 第五个测试案例，列名为 ["b"], 值为 [np.nan]
            (["b"], [np.nan]),
            # 创建一个 Series 对象，包含两个整数值，名为 ("b", np.nan)
            (
                DataFrame(
                    [[3], [6]],
                    # 列名使用 MultiIndex 手动创建，包括 levels 和 codes
                    columns=MultiIndex(
                        codes=[[1], [-1]],
                        levels=[["a", "b"], ["bar", "foo"]]
                    ),
                    dtype="int64",
                )
            ),
        ),
        (
            # 第六个测试案例，元组 ("b", np.nan)
            (("b", np.nan), 
            # 创建一个 Series 对象，包含两个整数值，名为 ("b", np.nan)
            Series([3, 6], dtype="int64", name=("b", np.nan))),
        ),
    ],
# -----------------------------------------------------------------------------
# test indexing of DataFrame with multi-level Index with duplicates
# -----------------------------------------------------------------------------

# 定义一个用于测试的函数，测试 DataFrame 在具有重复索引的多级索引情况下的索引操作
@pytest.fixture
def dataframe_with_duplicate_index():
    """Fixture for DataFrame used in tests for gh-4145 and gh-4146"""
    # 定义测试数据
    data = [["a", "d", "e", "c", "f", "b"], [1, 4, 5, 3, 6, 2], [1, 4, 5, 3, 6, 2]]
    index = ["h1", "h3", "h5"]
    # 定义多级索引
    columns = MultiIndex(
        levels=[["A", "B"], ["A1", "A2", "B1", "B2"]],
        codes=[[0, 0, 0, 1, 1, 1], [0, 3, 3, 0, 1, 2]],
        names=["main", "sub"],
    )
    return DataFrame(data, index=index, columns=columns)

# 定义测试函数，测试 DataFrame 中的多级索引操作，用于 GH 4145 的问题
@pytest.mark.parametrize(
    "indexer", [lambda df: df[("A", "A1")], lambda df: df.loc[:, ("A", "A1")]]
)
def test_frame_mi_access(dataframe_with_duplicate_index, indexer):
    # 获取数据框架
    df = dataframe_with_duplicate_index
    # 创建索引
    index = Index(["h1", "h3", "h5"])
    columns = MultiIndex.from_tuples([("A", "A1")], names=["main", "sub"])
    # 预期结果为特定的 DataFrame
    expected = DataFrame([["a", 1, 1]], index=columns, columns=index).T

    # 执行索引操作
    result = indexer(df)
    # 断言结果与预期一致
    tm.assert_frame_equal(result, expected)

# 定义测试函数，测试 DataFrame 中的多级索引操作，用于 GH 4146 的问题，预期结果为 Series 类型
def test_frame_mi_access_returns_series(dataframe_with_duplicate_index):
    # 获取数据框架
    df = dataframe_with_duplicate_index
    # 预期结果为 Series 类型
    expected = Series(["a", 1, 1], index=["h1", "h3", "h5"], name="A1")
    # 执行索引操作
    result = df["A"]["A1"]
    # 断言结果与预期一致
    tm.assert_series_equal(result, expected)

# 定义测试函数，测试 DataFrame 中的多级索引操作，预期结果为 DataFrame 类型
def test_frame_mi_access_returns_frame(dataframe_with_duplicate_index):
    # 获取数据框架
    df = dataframe_with_duplicate_index
    # 预期结果为特定的 DataFrame
    expected = DataFrame(
        [["d", 4, 4], ["e", 5, 5]],
        index=Index(["B2", "B2"], name="sub"),
        columns=["h1", "h3", "h5"],
    ).T
    # 执行索引操作
    result = df["A"]["B2"]
    # 断言结果与预期一致
    tm.assert_frame_equal(result, expected)

# 定义测试函数，测试 DataFrame 中的空切片操作
def test_frame_mi_empty_slice():
    # 创建全零的数据框架，用于测试
    df = DataFrame(0, index=range(2), columns=MultiIndex.from_product([[1], [2]]))
    # 执行空切片操作
    result = df[[]]
    # 创建预期结果为全零的数据框架
    expected = DataFrame(
        index=[0, 1], columns=MultiIndex(levels=[[1], [2]], codes=[[], []])
    )
    # 断言结果与预期一致
    tm.assert_frame_equal(result, expected)

# 定义测试函数，测试 loc 方法在空的多级索引上的操作
def test_loc_empty_multiindex():
    # 创建多级索引数组
    arrays = [["a", "a", "b", "a"], ["a", "a", "b", "b"]]
    # 使用数组创建多级索引对象
    index = MultiIndex.from_arrays(arrays, names=("idx1", "idx2"))
    # 创建一个 DataFrame，包含一列数据 [1, 2, 3, 4]，指定索引和列名
    df = DataFrame([1, 2, 3, 4], index=index, columns=["value"])

    # 当多重索引为空时，df.loc[df.loc[:, "value"] == 0, :] 等同于使用 False 掩码的 df.loc
    empty_multiindex = df.loc[df.loc[:, "value"] == 0, :].index
    # 使用空的多重索引从 DataFrame 中选择数据
    result = df.loc[empty_multiindex, :]
    # 期望结果是使用全为 False 的掩码从 DataFrame 中选择数据
    expected = df.loc[[False] * len(df.index), :]
    # 使用测试框架检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)

    # 在空的多重索引上使用 loc 替换值
    df.loc[df.loc[df.loc[:, "value"] == 0].index, "value"] = 5
    # 更新后的 DataFrame 作为结果
    result = df
    # 期望结果是初始的 DataFrame，不受更改影响
    expected = DataFrame([1, 2, 3, 4], index=index, columns=["value"])
    # 使用测试框架检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
```