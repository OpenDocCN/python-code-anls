# `D:\src\scipysrc\pandas\pandas\tests\copy_view\test_methods.py`

```
# 导入所需的库
import numpy as np
import pytest

# 导入 pandas 库及其特定组件
import pandas as pd
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Period,
    Series,
    Timestamp,
    date_range,
    period_range,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


# 定义测试函数：测试 DataFrame 的深拷贝
def test_copy():
    # 创建 DataFrame 对象 df
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 深拷贝 df，并赋值给 df_copy
    df_copy = df.copy()

    # 断言：深拷贝默认情况下会对索引进行浅拷贝
    assert df_copy.index is not df.index
    assert df_copy.columns is not df.columns
    assert df_copy.index.is_(df.index)
    assert df_copy.columns.is_(df.columns)

    # 断言：深拷贝不共享内存
    assert not np.shares_memory(get_array(df_copy, "a"), get_array(df, "a"))
    assert not df_copy._mgr.blocks[0].refs.has_reference()
    assert not df_copy._mgr.blocks[1].refs.has_reference()

    # 断言：对拷贝进行修改不会影响原始数据
    df_copy.iloc[0, 0] = 0
    assert df.iloc[0, 0] == 1


# 定义测试函数：测试 DataFrame 的浅拷贝
def test_copy_shallow():
    # 创建 DataFrame 对象 df
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 浅拷贝 df，并赋值给 df_copy
    df_copy = df.copy(deep=False)

    # 断言：浅拷贝同样会对索引进行浅拷贝
    assert df_copy.index is not df.index
    assert df_copy.columns is not df.columns
    assert df_copy.index.is_(df.index)
    assert df_copy.columns.is_(df.columns)

    # 断言：浅拷贝仍然共享内存
    assert np.shares_memory(get_array(df_copy, "a"), get_array(df, "a"))
    assert df_copy._mgr.blocks[0].refs.has_reference()
    assert df_copy._mgr.blocks[1].refs.has_reference()

    # 断言：对浅拷贝进行修改不会影响原始数据
    df_copy.iloc[0, 0] = 0
    assert df.iloc[0, 0] == 1
    # 修改触发了写时复制（copy-on-write），不再共享内存
    assert not np.shares_memory(get_array(df_copy, "a"), get_array(df, "a"))
    # 但对于其他列/块仍然共享内存
    assert np.shares_memory(get_array(df_copy, "c"), get_array(df, "c"))


# 使用 pytest 的装饰器标记，忽略特定的 DeprecationWarning
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
# 参数化测试：copy 参数为 True, None, False
@pytest.mark.parametrize("copy", [True, None, False])
@pytest.mark.parametrize(
    "method",
    # 使用 lambda 函数对数据框进行列重命名，将列名转换为小写，可选择是否复制数据框
    lambda df, copy: df.rename(columns=str.lower, copy=copy),
    
    # 使用 lambda 函数重新索引数据框的列，只包括列 "a" 和 "c"，可选择是否复制数据框
    lambda df, copy: df.reindex(columns=["a", "c"], copy=copy),
    
    # 使用 lambda 函数根据另一个数据框的结构重新索引当前数据框，可选择是否复制数据框
    lambda df, copy: df.reindex_like(df, copy=copy),
    
    # 使用 lambda 函数对齐两个数据框的索引，并返回第一个数据框，可选择是否复制数据框
    lambda df, copy: df.align(df, copy=copy)[0],
    
    # 使用 lambda 函数设置数据框的索引，指定新的索引为 ["a", "b", "c"]，可选择是否复制数据框
    lambda df, copy: df.set_axis(["a", "b", "c"], axis="index", copy=copy),
    
    # 使用 lambda 函数重命名数据框的行索引，将行索引命名为 "test"，可选择是否复制数据框
    lambda df, copy: df.rename_axis(index="test", copy=copy),
    
    # 使用 lambda 函数重命名数据框的列索引，将列索引命名为 "test"，可选择是否复制数据框
    lambda df, copy: df.rename_axis(columns="test", copy=copy),
    
    # 使用 lambda 函数将数据框的列 "b" 转换为 int64 类型，可选择是否复制数据框
    lambda df, copy: df.astype({"b": "int64"}, copy=copy),
    
    # 使用 lambda 函数截取数据框的行，保留索引从 0 到 5 的行，可选择是否复制数据框
    lambda df, copy: df.truncate(0, 5, copy=copy),
    
    # 使用 lambda 函数推断数据框中各列的数据类型，可选择是否复制数据框
    lambda df, copy: df.infer_objects(copy=copy),
    
    # 使用 lambda 函数将数据框转换为时间戳索引，可选择是否复制数据框
    lambda df, copy: df.to_timestamp(copy=copy),
    
    # 使用 lambda 函数将数据框转换为指定频率的周期索引，频率为每天 ("D")，可选择是否复制数据框
    lambda df, copy: df.to_period(freq="D", copy=copy),
    
    # 使用 lambda 函数将数据框本地化为 "US/Central" 时区，可选择是否复制数据框
    lambda df, copy: df.tz_localize("US/Central", copy=copy),
    
    # 使用 lambda 函数将数据框转换到 "US/Central" 时区，可选择是否复制数据框
    lambda df, copy: df.tz_convert("US/Central", copy=copy),
    
    # 使用 lambda 函数设置数据框的标志，不允许重复标签，可选择是否复制数据框
    lambda df, copy: df.set_flags(allows_duplicate_labels=False, copy=copy),
# 定义测试函数，用于测试各种序列操作方法和关键字参数
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
# 参数化测试，测试的关键字参数为 `copy`，分别测试 True、None 和 False 三种情况
@pytest.mark.parametrize("copy", [True, None, False])
# 参数化测试，测试的方法包括多个 lambda 表达式，每个表达式接受 `ser` 和 `copy` 作为参数
@pytest.mark.parametrize(
    "method",
    [
        lambda ser, copy: ser.rename(index={0: 100}, copy=copy),  # 重命名索引
        lambda ser, copy: ser.rename(None, copy=copy),  # 重命名不指定索引
        lambda ser, copy: ser.reindex(index=ser.index, copy=copy),  # 重新索引
        lambda ser, copy: ser.reindex_like(ser, copy=copy),  # 按照给定对象重新索引
        lambda ser, copy: ser.align(ser, copy=copy)[0],  # 对齐操作
        lambda ser, copy: ser.set_axis(["a", "b", "c"], axis="index", copy=copy),  # 设置轴标签
        lambda ser, copy: ser.rename_axis(index="test", copy=copy),  # 重命名轴名称
        lambda ser, copy: ser.astype("int64", copy=copy),  # 类型转换
        lambda ser, copy: ser.swaplevel(0, 1, copy=copy),  # 交换索引级别
        lambda ser, copy: ser.truncate(0, 5, copy=copy),  # 截取指定范围数据
        lambda ser, copy: ser.infer_objects(copy=copy),  # 推断对象类型
        lambda ser, copy: ser.to_timestamp(copy=copy),  # 转换为时间戳
        lambda ser, copy: ser.to_period(freq="D", copy=copy),  # 转换为周期
        lambda ser, copy: ser.tz_localize("US/Central", copy=copy),  # 本地化时区
        lambda ser, copy: ser.tz_convert("US/Central", copy=copy),  # 转换时区
        lambda ser, copy: ser.set_flags(allows_duplicate_labels=False, copy=copy),  # 设置标志
    ],
    # 每个方法对应的测试用例名称
    ids=[
        "rename (dict)",
        "rename",
        "reindex",
        "reindex_like",
        "align",
        "set_axis",
        "rename_axis0",
        "astype",
        "swaplevel",
        "truncate",
        "infer_objects",
        "to_timestamp",
        "to_period",
        "tz_localize",
        "tz_convert",
        "set_flags",
    ],
)
# 定义测试方法，接受 `request`、`method` 和 `copy` 作为参数
def test_methods_series_copy_keyword(request, method, copy):
    index = None
    # 根据测试用例名称设置索引
    if "to_timestamp" in request.node.callspec.id:
        index = period_range("2012-01-01", freq="D", periods=3)
    elif "to_period" in request.node.callspec.id:
        index = date_range("2012-01-01", freq="D", periods=3)
    elif "tz_localize" in request.node.callspec.id:
        index = date_range("2012-01-01", freq="D", periods=3)
    elif "tz_convert" in request.node.callspec.id:
        index = date_range("2012-01-01", freq="D", periods=3, tz="Europe/Brussels")
    elif "swaplevel" in request.node.callspec.id:
        index = MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]])

    # 创建一个测试用的 Series 对象，包含三个元素的列表，使用给定索引
    ser = Series([1, 2, 3], index=index)
    # 使用给定的方法处理序列 ser，并返回处理后的新序列 ser2
    ser2 = method(ser, copy=copy)
    # 断言：确保序列 ser2 和序列 ser 共享内存，即它们指向相同的数据数组
    assert np.shares_memory(get_array(ser2), get_array(ser))
# -----------------------------------------------------------------------------
# DataFrame methods returning new DataFrame using shallow copy

# 定义测试函数 test_reset_index
def test_reset_index():
    # Case: resetting the index (i.e. adding a new column) + mutating the
    # resulting dataframe
    # 创建 DataFrame 对象 df，指定列 a, b, c 和索引 [10, 11, 12]
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]}, index=[10, 11, 12]
    )
    # 复制 df，保存原始 DataFrame
    df_orig = df.copy()
    # 对 df 执行 reset_index 操作，返回新的 DataFrame df2
    df2 = df.reset_index()
    # 验证新 DataFrame df2 的数据完整性
    df2._mgr._verify_integrity()

    # 仍然共享内存 (df2 是浅复制)
    # 验证 df2 中的列 'b' 和原始 df 中的列 'b' 是否共享内存
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    # 验证 df2 中的列 'c' 和原始 df 中的列 'c' 是否共享内存
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    # 修改 df2 触发对该列/块的写时复制
    df2.iloc[0, 2] = 0
    # 验证 df2 中的列 'b' 和原始 df 中的列 'b' 不再共享内存
    assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    # 验证 df2 中的列 'c' 和原始 df 中的列 'c' 仍然共享内存
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    # 验证 df 和 df_orig 是否相等
    tm.assert_frame_equal(df, df_orig)


# 使用 pytest.mark.parametrize 装饰器，对 test_reset_index_series_drop 函数进行参数化测试
@pytest.mark.parametrize("index", [pd.RangeIndex(0, 2), Index([1, 2])])
# 定义测试函数 test_reset_index_series_drop
def test_reset_index_series_drop(index):
    # 创建 Series 对象 ser，指定索引 index
    ser = Series([1, 2], index=index)
    # 复制 ser，保存原始 Series
    ser_orig = ser.copy()
    # 对 ser 执行 reset_index 操作，drop=True，返回新的 Series ser2
    ser2 = ser.reset_index(drop=True)
    # 验证 ser 和 ser2 是否共享内存
    assert np.shares_memory(get_array(ser), get_array(ser2))
    # 验证 ser 的内部管理器是否没有对第一个元素的引用
    assert not ser._mgr._has_no_reference(0)

    # 修改 ser2 的第一个元素
    ser2.iloc[0] = 100
    # 验证 ser 和 ser_orig 是否相等
    tm.assert_series_equal(ser, ser_orig)


# 定义测试函数 test_groupby_column_index_in_references
def test_groupby_column_index_in_references():
    # 创建 DataFrame 对象 df，指定列 A, B, C
    df = DataFrame(
        {"A": ["a", "b", "c", "d"], "B": [1, 2, 3, 4], "C": ["a", "a", "b", "b"]}
    )
    # 将列 'A' 设置为索引列
    df = df.set_index("A")
    # 使用列 'C' 进行分组，并计算分组后的和，保存结果为 result
    key = df["C"]
    result = df.groupby(key, observed=True).sum()
    # 使用列 'C' 进行分组，并计算分组后的和，保存期望结果为 expected
    expected = df.groupby("C", observed=True).sum()
    # 验证 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数 test_rename_columns
def test_rename_columns():
    # Case: renaming columns returns a new dataframe
    # + afterwards modifying the result
    # 创建 DataFrame 对象 df，指定列 a, b, c
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 复制 df，保存原始 DataFrame
    df_orig = df.copy()
    # 对 df 执行 rename 操作，使用 str.upper 函数重命名列，返回新的 DataFrame df2
    df2 = df.rename(columns=str.upper)

    # 验证 df2 中的列 'A' 和原始 df 中的列 'a' 是否共享内存
    assert np.shares_memory(get_array(df2, "A"), get_array(df, "a"))
    # 修改 df2，触发对该列/块的写时复制
    df2.iloc[0, 0] = 0
    # 验证 df2 中的列 'A' 和原始 df 中的列 'a' 不再共享内存
    assert not np.shares_memory(get_array(df2, "A"), get_array(df, "a"))
    # 验证 df2 中的列 'C' 和原始 df 中的列 'c' 是否共享内存
    assert np.shares_memory(get_array(df2, "C"), get_array(df, "c"))
    # 创建期望结果 DataFrame 对象 expected
    expected = DataFrame({"A": [0, 2, 3], "B": [4, 5, 6], "C": [0.1, 0.2, 0.3]})
    # 验证 df2 和 expected 是否相等
    tm.assert_frame_equal(df2, expected)
    # 验证 df 和 df_orig 是否相等
    tm.assert_frame_equal(df, df_orig)


# 定义测试函数 test_rename_columns_modify_parent
def test_rename_columns_modify_parent():
    # Case: renaming columns returns a new dataframe
    # + afterwards modifying the original (parent) dataframe
    # 创建 DataFrame 对象 df，指定列 a, b, c
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 对 df 执行 rename 操作，使用 str.upper 函数重命名列，返回新的 DataFrame df2
    df2 = df.rename(columns=str.upper)
    # 复制 df2，保存原始 DataFrame
    df2_orig = df2.copy()

    # 验证 df2 中的列 'A' 和原始 df 中的列 'a' 是否共享内存
    assert np.shares_memory(get_array(df2, "A"), get_array(df, "a"))
    # 修改 df 中的数据
    df.iloc[0, 0] = 0
    # 验证 df2 中的列 'A' 和原始 df 中的列 'a' 不再共享内存
    assert not np.shares_memory(get_array(df2, "A"), get_array(df, "a"))
    # 验证 df2 中的列 'C' 和原始 df 中的列 'c' 是否共享内存
    assert np.shares_memory(get_array(df2, "C"), get_array(df, "c"))
    # 创建期望结果 DataFrame 对象 expected
    expected = DataFrame({"a": [0, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 验证 df 和 expected 是否相等
    tm.assert_frame_equal(df, expected)
    # 使用测试工具tm.assert_frame_equal比较两个数据框df2和df2_orig是否相等
    tm.assert_frame_equal(df2, df2_orig)
def test_pipe():
    # 创建包含列"a"和"b"的DataFrame
    df = DataFrame({"a": [1, 2, 3], "b": 1.5})
    # 备份原始DataFrame
    df_orig = df.copy()

    # 定义一个函数testfunc，接受一个DataFrame并返回它本身
    def testfunc(df):
        return df

    # 使用DataFrame的pipe方法，将testfunc应用于df，得到df2
    df2 = df.pipe(testfunc)

    # 断言列"a"在df2和df之间共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # 修改df2的值会触发对该列的写时复制
    df2.iloc[0, 0] = 0
    # 断言df和df_orig相等
    tm.assert_frame_equal(df, df_orig)
    # 断言列"a"在df2和df之间不共享内存
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言列"b"在df2和df之间共享内存
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))


def test_pipe_modify_df():
    # 创建包含列"a"和"b"的DataFrame
    df = DataFrame({"a": [1, 2, 3], "b": 1.5})
    # 备份原始DataFrame
    df_orig = df.copy()

    # 定义一个函数testfunc，接受一个DataFrame并修改其第一个元素为100，然后返回
    def testfunc(df):
        df.iloc[0, 0] = 100
        return df

    # 使用DataFrame的pipe方法，将testfunc应用于df，得到df2
    df2 = df.pipe(testfunc)

    # 断言列"b"在df2和df之间共享内存
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    # 断言df和df_orig相等
    tm.assert_frame_equal(df, df_orig)
    # 断言列"a"在df2和df之间不共享内存
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言列"b"在df2和df之间共享内存
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))


def test_reindex_columns():
    # 情况：重新索引列返回一个新的DataFrame
    # + 之后修改结果
    # 创建包含列"a", "b", "c"的DataFrame
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 备份原始DataFrame
    df_orig = df.copy()
    # 使用DataFrame的reindex方法，重新索引列为["a", "c"]，得到df2
    df2 = df.reindex(columns=["a", "c"])

    # 断言列"a"在df2和df之间共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 修改df2的值会触发对该列的写时复制
    df2.iloc[0, 0] = 0
    # 断言列"a"在df2和df之间不共享内存
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言列"c"在df2和df之间共享内存
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    # 断言df和df_orig相等
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "index",
    [
        lambda idx: idx,
        lambda idx: idx.view(),
        lambda idx: idx.copy(),
        lambda idx: list(idx),
    ],
    ids=["identical", "view", "copy", "values"],
)
def test_reindex_rows(index):
    # 情况：使用与当前索引匹配的索引重新索引行时可以使用浅复制
    # 创建包含列"a", "b", "c"的DataFrame
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 备份原始DataFrame
    df_orig = df.copy()
    # 使用DataFrame的reindex方法，使用给定的索引函数index重新索引行，得到df2
    df2 = df.reindex(index=index(df.index))

    # 断言列"a"在df2和df之间共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 修改df2的值会触发对该列的写时复制
    df2.iloc[0, 0] = 0
    # 断言列"a"在df2和df之间不共享内存
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言列"c"在df2和df之间共享内存
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    # 断言df和df_orig相等
    tm.assert_frame_equal(df, df_orig)


def test_drop_on_column():
    # 创建包含列"a", "b", "c"的DataFrame
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 备份原始DataFrame
    df_orig = df.copy()
    # 使用DataFrame的drop方法，删除列"a"，得到df2
    df2 = df.drop(columns="a")
    # 验证DataFrame的内部完整性
    df2._mgr._verify_integrity()

    # 断言列"b"在df2和df之间共享内存
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    # 断言列"c"在df2和df之间共享内存
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    # 修改df2的值会触发对该列的写时复制
    df2.iloc[0, 0] = 0
    # 断言列"b"在df2和df之间不共享内存
    assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    # 断言列"c"在df2和df之间共享内存
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    # 使用 pandas 的 tm 模块中的 assert_frame_equal 函数来比较两个数据框 df 和 df_orig 是否相等
    tm.assert_frame_equal(df, df_orig)
def test_select_dtypes():
    # 定义测试函数，测试数据框使用 `select_dtypes()` 方法选择列后返回一个新的数据框
    # + 之后修改结果
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()  # 备份原始数据框
    df2 = df.select_dtypes("int64")  # 选择所有整数类型的列，返回新的数据框
    df2._mgr._verify_integrity()  # 验证内部数据结构的完整性

    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言：检查是否共享内存，验证选择的列是否与原始数据框的相同

    # 修改 df2 会触发对该列/块的写时复制
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言：修改 df2 后，检查是否不再共享内存，验证写时复制的效果
    tm.assert_frame_equal(df, df_orig)  # 断言：验证修改原始数据框是否不受影响


@pytest.mark.parametrize(
    "filter_kwargs", [{"items": ["a"]}, {"like": "a"}, {"regex": "a"}]
)
def test_filter(filter_kwargs):
    # 定义测试函数，测试数据框使用 `filter()` 方法选择列后返回一个新的数据框
    # + 之后修改结果
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    df_orig = df.copy()  # 备份原始数据框
    df2 = df.filter(**filter_kwargs)  # 根据给定的过滤参数选择列，返回新的数据框
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # 修改 df2 会触发对该列/块的写时复制
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言：修改 df2 后，检查是否不再共享内存，验证写时复制的效果
    tm.assert_frame_equal(df, df_orig)  # 断言：验证修改原始数据框是否不受影响


def test_shift_no_op():
    # 定义测试函数，测试数据框使用 `shift(periods=0)` 方法返回自身
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        index=date_range("2020-01-01", "2020-01-03"),
        columns=["a", "b"],
    )
    df_orig = df.copy()  # 备份原始数据框
    df2 = df.shift(periods=0)  # 对数据框进行零位移操作，返回自身
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    df.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))
    # 断言：修改原始数据框后，验证是否不再共享内存
    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    tm.assert_frame_equal(df2, df_orig)  # 断言：验证修改原始数据框是否不受影响


def test_shift_index():
    # 定义测试函数，测试数据框使用 `shift(periods=1, axis=0)` 方法进行索引位移
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        index=date_range("2020-01-01", "2020-01-03"),
        columns=["a", "b"],
    )
    df2 = df.shift(periods=1, axis=0)  # 对索引进行正向位移操作

    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言：检查是否不共享内存，验证索引位移后的效果


def test_shift_rows_freq():
    # 定义测试函数，测试数据框使用 `shift(periods=1, freq="1D")` 方法进行日期频率位移
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6]],
        index=date_range("2020-01-01", "2020-01-03"),
        columns=["a", "b"],
    )
    df_orig = df.copy()  # 备份原始数据框
    df_orig.index = date_range("2020-01-02", "2020-01-04")  # 修改备份的索引
    df2 = df.shift(periods=1, freq="1D")  # 对日期频率进行位移操作

    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言：检查是否共享内存，验证日期频率位移后的效果
    df.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))
    # 断言：修改原始数据框后，检查是否不再共享内存
    tm.assert_frame_equal(df2, df_orig)  # 断言：验证修改原始数据框是否不受影响


def test_shift_columns():
    # 定义测试函数，测试数据框使用 `shift(periods=1, axis=1)` 方法进行列位移
    df = DataFrame(
        [[1, 2], [3, 4], [5, 6]], columns=date_range("2020-01-01", "2020-01-02")
    )
    df2 = df.shift(periods=1, axis=1)  # 对列进行正向位移操作

    assert np.shares_memory(get_array(df2, "2020-01-02"), get_array(df, "2020-01-01"))
    # 断言：检查是否共享内存，验证列位移后的效果
    df.iloc[0, 0] = 0
    assert not np.shares_memory(
        get_array(df2, "2020-01-02"), get_array(df, "2020-01-01")
    )
    # 断言：修改原始数据框后，检查是否不再共享内存
    # 创建一个预期的 DataFrame 对象，包含三行两列的数据
    expected = DataFrame(
        [[np.nan, 1], [np.nan, 3], [np.nan, 5]],
        # 指定列的日期范围为从 "2020-01-01" 到 "2020-01-02"
        columns=date_range("2020-01-01", "2020-01-02"),
    )
    # 使用测试工具比较两个 DataFrame 对象 df2 和 expected 是否相等
    tm.assert_frame_equal(df2, expected)
def test_pop():
    # 创建一个包含三列的 DataFrame
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 复制原始 DataFrame
    df_orig = df.copy()
    # 通过切片创建原始 DataFrame 的视图
    view_original = df[:]
    # 使用 pop 方法删除并返回列 'a'，结果存储在 result 中
    result = df.pop("a")

    # 断言结果的内存与视图的列 'a' 共享
    assert np.shares_memory(result.values, get_array(view_original, "a"))
    # 断言修改后的 DataFrame 的列 'b' 与原始视图的列 'b' 共享内存
    assert np.shares_memory(get_array(df, "b"), get_array(view_original, "b"))

    # 修改 result 的第一个元素，断言不再共享内存
    result.iloc[0] = 0
    assert not np.shares_memory(result.values, get_array(view_original, "a"))
    # 修改 df 的第一个元素，断言不再共享内存
    df.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df, "b"), get_array(view_original, "b"))
    # 断言修改后的 DataFrame 等于原始 DataFrame
    tm.assert_frame_equal(view_original, df_orig)


@pytest.mark.parametrize(
    "func",
    [
        lambda x, y: x.align(y),  # align 函数的基本用法
        lambda x, y: x.align(y.a, axis=0),  # 沿轴 0 对齐 DataFrame 和 DataFrame 的列 'a'
        lambda x, y: x.align(y.a.iloc[slice(0, 1)], axis=1),  # 沿轴 1 对齐 DataFrame 和 DataFrame 的列 'a' 的切片
    ],
)
def test_align_frame(func):
    # 创建一个包含两列的 DataFrame
    df = DataFrame({"a": [1, 2, 3], "b": "a"})
    # 复制原始 DataFrame
    df_orig = df.copy()
    # 复制并改变了顺序的 DataFrame
    df_changed = df[["b", "a"]].copy()
    # 调用 align 函数，返回对齐后的结果 df2 和 _
    df2, _ = func(df, df_changed)

    # 断言 df2 的列 'a' 与 df 的列 'a' 共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 修改 df2 的第一个元素，断言不再共享内存
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言修改后的 DataFrame 等于原始 DataFrame
    tm.assert_frame_equal(df, df_orig)


def test_align_series():
    # 创建一个 Series
    ser = Series([1, 2])
    # 复制原始 Series
    ser_orig = ser.copy()
    # 复制的另一个 Series
    ser_other = ser.copy()
    # 调用 align 方法，返回对齐后的结果 ser2 和 ser_other_result
    ser2, ser_other_result = ser.align(ser_other)

    # 断言 ser2 与原始 Series 共享内存
    assert np.shares_memory(ser2.values, ser.values)
    # 断言 ser_other_result 与 ser_other 共享内存
    assert np.shares_memory(ser_other_result.values, ser_other.values)
    # 修改 ser2 的第一个元素，断言不再共享内存
    ser2.iloc[0] = 0
    ser_other_result.iloc[0] = 0
    assert not np.shares_memory(ser2.values, ser.values)
    assert not np.shares_memory(ser_other_result.values, ser_other.values)
    # 断言修改后的 Series 等于原始 Series
    tm.assert_series_equal(ser, ser_orig)
    tm.assert_series_equal(ser_other, ser_orig)


def test_align_copy_false():
    # 创建一个包含两列的 DataFrame
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # 复制原始 DataFrame
    df_orig = df.copy()
    # 调用 align 方法，返回对齐后的结果 df2 和 df3
    df2, df3 = df.align(df)

    # 断言 df2 的列 'b' 与 df 的列 'b' 共享内存
    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    # 断言 df2 的列 'a' 与 df 的列 'a' 共享内存
    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    # 修改 df2 的第一个元素，断言原始 DataFrame 未改变
    df2.loc[0, "a"] = 0
    tm.assert_frame_equal(df, df_orig)  # Original is unchanged

    # 修改 df3 的第一个元素，断言原始 DataFrame 未改变
    df3.loc[0, "a"] = 0
    tm.assert_frame_equal(df, df_orig)  # Original is unchanged


def test_align_with_series_copy_false():
    # 创建一个包含两列的 DataFrame
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # 创建一个 Series
    ser = Series([1, 2, 3], name="x")
    # 复制原始 Series
    ser_orig = ser.copy()
    # 复制原始 DataFrame
    df_orig = df.copy()
    # 调用 align 方法，返回对齐后的结果 df2 和 ser2
    df2, ser2 = df.align(ser, axis=0)

    # 断言 df2 的列 'b' 与 df 的列 'b' 共享内存
    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    # 断言 df2 的列 'a' 与 df 的列 'a' 共享内存
    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))
    # 断言 ser2 的值与原始 Series 共享内存
    assert np.shares_memory(get_array(ser, "x"), get_array(ser2, "x"))

    # 修改 df2 的第一个元素，断言原始 DataFrame 未改变
    df2.loc[0, "a"] = 0
    tm.assert_frame_equal(df, df_orig)  # Original is unchanged

    # 修改 ser2 的第一个元素，断言原始 Series 未改变
    ser2.loc[0] = 0
    tm.assert_series_equal(ser, ser_orig)  # Original is unchanged


def test_to_frame():
    # 情况：使用 to_frame 将 Series 转换为 DataFrame
    ser = Series([1, 2, 3])
    ser_orig = ser.copy()

    # 使用 to_frame 方法将 Series 转换为 DataFrame
    df = ser[:].to_frame()

    # 目前这总是返回一个视图
    # 断言检查序列 `ser.values` 是否与 `df` 的第一列共享内存
    assert np.shares_memory(ser.values, get_array(df, 0))

    # 修改 `df` 的第一行第一列的值为 0
    df.iloc[0, 0] = 0

    # 当修改 `df` 时会触发对该列的写时复制操作
    assert not np.shares_memory(ser.values, get_array(df, 0))
    # 使用测试工具确保 `ser` 没有被修改
    tm.assert_series_equal(ser, ser_orig)

    # 将原始序列 `ser` 转换为DataFrame `df`的副本
    df = ser[:].to_frame()
    # 修改 `ser` 的第一个元素为 0
    ser.iloc[0] = 0

    # 使用测试工具确保 `df` 与 `ser_orig` 的DataFrame表示形式相等
    tm.assert_frame_equal(df, ser_orig.to_frame())
@pytest.mark.parametrize(
    "method, idx",
    [
        # 定义第一个参数为浅复制两次的函数，第二个参数为索引 0
        (lambda df: df.copy(deep=False).copy(deep=False), 0),
        # 定义第一个参数为重置索引两次的函数，第二个参数为索引 2
        (lambda df: df.reset_index().reset_index(), 2),
        # 定义第一个参数为列名转大写再转小写的函数，第二个参数为索引 0
        (lambda df: df.rename(columns=str.upper).rename(columns=str.lower), 0),
        # 定义第一个参数为浅复制后选取数字类型的函数，第二个参数为索引 0
        (lambda df: df.copy(deep=False).select_dtypes(include="number"), 0),
    ],
    # 指定每组参数的标识
    ids=["shallow-copy", "reset_index", "rename", "select_dtypes"],
)
def test_chained_methods(request, method, idx):
    # 创建一个 DataFrame 对象，包含三列数据
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 复制原始 DataFrame 对象
    df_orig = df.copy()

    # 修改 df2，但不修改 df
    df2 = method(df)
    # 修改 df2 中的特定位置的元素为 0
    df2.iloc[0, idx] = 0
    # 断言 df 和 df_orig 相等
    tm.assert_frame_equal(df, df_orig)

    # 再次修改 df，但不修改 df2
    df2 = method(df)
    # 修改 df 中的特定位置的元素为 0
    df.iloc[0, 0] = 0
    # 断言 df2 的部分列和 df_orig 相等
    tm.assert_frame_equal(df2.iloc[:, idx:], df_orig)


@pytest.mark.parametrize("obj", [Series([1, 2], name="a"), DataFrame({"a": [1, 2]})])
def test_to_timestamp(obj):
    # 设置对象的索引为时间戳对象
    obj.index = Index([Period("2012-1-1", freq="D"), Period("2012-1-2", freq="D")])

    # 复制原始对象
    obj_orig = obj.copy()
    # 对象转换为时间戳格式
    obj2 = obj.to_timestamp()

    # 断言两个数组共享内存
    assert np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))

    # 修改 obj2 中的元素触发拷贝写操作
    obj2.iloc[0] = 0
    # 断言两对象相等
    assert not np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))
    tm.assert_equal(obj, obj_orig)


@pytest.mark.parametrize("obj", [Series([1, 2], name="a"), DataFrame({"a": [1, 2]})])
def test_to_period(obj):
    # 设置对象的索引为时间戳对象
    obj.index = Index([Timestamp("2019-12-31"), Timestamp("2020-12-31")])

    # 复制原始对象
    obj_orig = obj.copy()
    # 对象转换为周期格式，频率为年
    obj2 = obj.to_period(freq="Y")

    # 断言两个数组共享内存
    assert np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))

    # 修改 obj2 中的元素触发拷贝写操作
    obj2.iloc[0] = 0
    # 断言两对象相等
    assert not np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))
    tm.assert_equal(obj, obj_orig)


def test_set_index():
    # 创建一个 DataFrame 对象，包含三列数据
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 复制原始 DataFrame 对象
    df_orig = df.copy()
    # 对象设置索引为列 'a'
    df2 = df.set_index("a")

    # 断言两个数组共享内存
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    # 修改 df2 中的元素触发拷贝写操作
    df2.iloc[0, 1] = 0
    # 断言两对象不共享内存
    assert not np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    # 断言两对象相等
    tm.assert_frame_equal(df, df_orig)


def test_set_index_mutating_parent_does_not_mutate_index():
    # 创建一个 DataFrame 对象，包含两列数据
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    # 设置索引为列 'a'
    result = df.set_index("a")
    # 复制预期结果
    expected = result.copy()

    # 修改 df 中的元素
    df.iloc[0, 0] = 100
    # 断言结果与预期相等
    tm.assert_frame_equal(result, expected)


def test_add_prefix():
    # 创建一个 DataFrame 对象，包含三列数据
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 复制原始 DataFrame 对象
    df_orig = df.copy()
    # 添加前缀 'CoW_' 到列名
    df2 = df.add_prefix("CoW_")

    # 断言两个数组共享内存
    assert np.shares_memory(get_array(df2, "CoW_a"), get_array(df, "a"))
    # 修改 df2 中的元素
    df2.iloc[0, 0] = 0

    # 断言两对象不共享内存
    assert not np.shares_memory(get_array(df2, "CoW_a"), get_array(df, "a"))

    # 断言两个数组共享内存
    assert np.shares_memory(get_array(df2, "CoW_c"), get_array(df, "c"))
    # 创建预期的 DataFrame，包含三列 "CoW_a", "CoW_b", "CoW_c"，每列对应的值分别为列表中的值
    expected = DataFrame(
        {"CoW_a": [0, 2, 3], "CoW_b": [4, 5, 6], "CoW_c": [0.1, 0.2, 0.3]}
    )
    # 使用 pytest 的 assert_frame_equal 函数比较 df2 和预期的 DataFrame 是否相等
    tm.assert_frame_equal(df2, expected)
    # 使用 pytest 的 assert_frame_equal 函数比较 df 和 df_orig 是否相等
    tm.assert_frame_equal(df, df_orig)
def test_add_suffix():
    # GH 49473
    # 创建一个包含三列的 DataFrame 对象
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 复制原始 DataFrame 对象
    df_orig = df.copy()
    # 给每列的列名添加后缀 "_CoW"，生成新的 DataFrame 对象 df2
    df2 = df.add_suffix("_CoW")
    # 断言新生成的列与原始列共享内存（即共享数据）
    assert np.shares_memory(get_array(df2, "a_CoW"), get_array(df, "a"))
    # 修改 df2 的一个元素，确保其与原始列不再共享内存
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a_CoW"), get_array(df, "a"))
    # 断言其他列仍与原始列共享内存
    assert np.shares_memory(get_array(df2, "c_CoW"), get_array(df, "c"))
    # 创建期望的 DataFrame 对象 expected
    expected = DataFrame(
        {"a_CoW": [0, 2, 3], "b_CoW": [4, 5, 6], "c_CoW": [0.1, 0.2, 0.3]}
    )
    # 断言 df2 与期望的 DataFrame 对象 expected 相等
    tm.assert_frame_equal(df2, expected)
    # 断言原始 DataFrame 对象 df 与复制前的 df_orig 相等
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("axis, val", [(0, 5.5), (1, np.nan)])
def test_dropna(axis, val):
    # 创建一个包含三列的 DataFrame 对象，其中包含 NaN 值
    df = DataFrame({"a": [1, 2, 3], "b": [4, val, 6], "c": "d"})
    # 复制原始 DataFrame 对象
    df_orig = df.copy()
    # 根据指定的轴向删除 NaN 值，生成新的 DataFrame 对象 df2
    df2 = df.dropna(axis=axis)

    # 断言删除 NaN 后，df2 的列与原始列共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # 修改 df2 的一个元素，确保其与原始列不再共享内存
    df2.iloc[0, 0] = 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言原始 DataFrame 对象 df 与复制前的 df_orig 相等
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("val", [5, 5.5])
def test_dropna_series(val):
    # 创建一个包含三个元素的 Series 对象，其中包含 NaN 值
    ser = Series([1, val, 4])
    # 复制原始 Series 对象
    ser_orig = ser.copy()
    # 删除 NaN 值，生成新的 Series 对象 ser2
    ser2 = ser.dropna()
    # 断言删除 NaN 后，ser2 的值与原始 Series 对象共享内存
    assert np.shares_memory(ser2.values, ser.values)

    # 修改 ser2 的一个元素，确保其与原始 Series 对象不再共享内存
    ser2.iloc[0] = 0
    assert not np.shares_memory(ser2.values, ser.values)
    # 断言原始 Series 对象 ser 与复制前的 ser_orig 相等
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize(
    "method",
    [
        lambda df: df.head(),
        lambda df: df.head(2),
        lambda df: df.tail(),
        lambda df: df.tail(3),
    ],
)
def test_head_tail(method):
    # 创建一个包含两列的 DataFrame 对象
    df = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    # 复制原始 DataFrame 对象
    df_orig = df.copy()
    # 调用不同的方法获取 DataFrame 对象 df 的头部或尾部数据，生成新的 DataFrame 对象 df2
    df2 = method(df)
    # 验证 df2 的数据块完整性
    df2._mgr._verify_integrity()

    # 在这里明确地采用 CoW 来进行复制操作（避免对非常便宜操作跟踪引用）
    # 断言 df2 的某一列与原始列不共享内存
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    # 修改 df2 的一个元素，触发 CoW 操作
    df2.iloc[0, 0] = 0
    # 断言 df2 的某一列与原始列不共享内存
    assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言原始 DataFrame 对象 df 与复制前的 df_orig 相等
    tm.assert_frame_equal(df, df_orig)


def test_infer_objects():
    # 创建一个包含四列的 DataFrame 对象
    df = DataFrame({"a": [1, 2], "b": "c", "c": 1, "d": "x"})
    # 复制原始 DataFrame 对象
    df_orig = df.copy()
    # 推断 DataFrame 对象中的对象类型，生成新的 DataFrame 对象 df2
    df2 = df.infer_objects()

    # 断言推断后的 DataFrame 对象 df2 的列与原始列共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    # 修改 df2 的一些元素，确保其与原始列不再共享内存
    df2.iloc[0, 0] = 0
    df2.iloc[0, 1] = "d"
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    # 断言原始 DataFrame 对象 df 与复制前的 df_orig 相等
    tm.assert_frame_equal(df, df_orig)


def test_infer_objects_no_reference():
    pass
    # 创建一个 DataFrame 对象 df，包含列 'a', 'b', 'c', 'd', 'e'，并赋初值
    df = DataFrame(
        {
            "a": [1, 2],  # 列 'a' 包含整数列表 [1, 2]
            "b": "c",     # 列 'b' 包含字符串 'c'，自动广播为整个列
            "c": 1,       # 列 'c' 包含整数值 1，自动广播为整个列
            "d": Series(  # 列 'd' 包含日期时间对象的 Series
                [Timestamp("2019-12-31"), Timestamp("2020-12-31")],
                dtype="object"
            ),
            "e": "b",     # 列 'e' 包含字符串 'b'，自动广播为整个列
        }
    )
    # 将 DataFrame 中的数据类型推断为最适合的类型
    df = df.infer_objects()
    
    # 从 DataFrame 中提取列 'a', 'b', 'd' 的数据为数组
    arr_a = get_array(df, "a")
    arr_b = get_array(df, "b")
    arr_d = get_array(df, "d")
    
    # 修改 DataFrame 中的特定元素值
    df.iloc[0, 0] = 0                        # 修改第一行第一列的元素为整数 0
    df.iloc[0, 1] = "d"                      # 修改第一行第二列的元素为字符串 'd'
    df.iloc[0, 3] = Timestamp("2018-12-31")  # 修改第一行第四列的元素为日期时间对象 '2018-12-31'
    
    # 断言验证数组 arr_a, arr_b, arr_d 与 DataFrame 中对应列的内存共享情况
    assert np.shares_memory(arr_a, get_array(df, "a"))  # 断言数组 arr_a 与列 'a' 的数据共享内存
    # TODO(CoW): Block splitting causes references here
    assert not np.shares_memory(arr_b, get_array(df, "b"))  # 断言数组 arr_b 与列 'b' 的数据不共享内存
    assert np.shares_memory(arr_d, get_array(df, "d"))     # 断言数组 arr_d 与列 'd' 的数据共享内存
def test_infer_objects_reference():
    df = DataFrame(
        {
            "a": [1, 2],  # 创建包含列 'a' 的 DataFrame，列数据为 [1, 2]
            "b": "c",     # 第二列 'b' 是字符串 "c"
            "c": 1,       # 第三列 'c' 是整数 1
            "d": Series(  # 第四列 'd' 是包含 Timestamp 对象的 Series
                [Timestamp("2019-12-31"), Timestamp("2020-12-31")], dtype="object"
            ),
        }
    )
    view = df[:]  # noqa: F841 创建一个视图 'view'，未使用但可能会被用到
    df = df.infer_objects()  # 推断 DataFrame 中的对象类型并返回修改后的 DataFrame

    arr_a = get_array(df, "a")  # 获取 DataFrame 列 'a' 的数组
    arr_b = get_array(df, "b")  # 获取 DataFrame 列 'b' 的数组
    arr_d = get_array(df, "d")  # 获取 DataFrame 列 'd' 的数组

    df.iloc[0, 0] = 0  # 修改 DataFrame 第一行第一列的值为 0
    df.iloc[0, 1] = "d"  # 修改 DataFrame 第一行第二列的值为 "d"
    df.iloc[0, 3] = Timestamp("2018-12-31")  # 修改 DataFrame 第一行第四列的值为 Timestamp 对象 "2018-12-31"
    assert not np.shares_memory(arr_a, get_array(df, "a"))  # 断言数组 arr_a 与修改后的 'a' 列数组不共享内存
    assert not np.shares_memory(arr_b, get_array(df, "b"))  # 断言数组 arr_b 与修改后的 'b' 列数组不共享内存
    assert np.shares_memory(arr_d, get_array(df, "d"))  # 断言数组 arr_d 与修改后的 'd' 列数组共享内存


@pytest.mark.parametrize(
    "kwargs",
    [
        {"before": "a", "after": "b", "axis": 1},  # 参数化测试用例，设定 'before' 为 'a'，'after' 为 'b'，'axis' 为 1
        {"before": 0, "after": 1, "axis": 0},     # 参数化测试用例，设定 'before' 为 0，'after' 为 1，'axis' 为 0
    ],
)
def test_truncate(kwargs):
    df = DataFrame({"a": [1, 2, 3], "b": 1, "c": 2})  # 创建包含列 'a', 'b', 'c' 的 DataFrame
    df_orig = df.copy()  # 复制原始 DataFrame
    df2 = df.truncate(**kwargs)  # 根据给定参数截取 DataFrame
    df2._mgr._verify_integrity()  # 验证截取后的 DataFrame 的完整性

    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))  # 断言截取后的 'a' 列数组与原始的 'a' 列数组共享内存

    df2.iloc[0, 0] = 0  # 修改截取后的 DataFrame 的第一行第一列的值为 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))  # 断言截取后的 'a' 列数组与原始的 'a' 列数组不共享内存
    tm.assert_frame_equal(df, df_orig)  # 使用测试框架断言两个 DataFrame 相等


@pytest.mark.parametrize("method", ["assign", "drop_duplicates"])
def test_assign_drop_duplicates(method):
    df = DataFrame({"a": [1, 2, 3]})  # 创建包含列 'a' 的 DataFrame
    df_orig = df.copy()  # 复制原始 DataFrame
    df2 = getattr(df, method)()  # 调用指定的方法（assign 或 drop_duplicates）
    df2._mgr._verify_integrity()  # 验证执行方法后的 DataFrame 的完整性

    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))  # 断言执行方法后的 'a' 列数组与原始的 'a' 列数组共享内存

    df2.iloc[0, 0] = 0  # 修改执行方法后的 DataFrame 的第一行第一列的值为 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))  # 断言执行方法后的 'a' 列数组与原始的 'a' 列数组不共享内存
    tm.assert_frame_equal(df, df_orig)  # 使用测试框架断言两个 DataFrame 相等


@pytest.mark.parametrize("obj", [Series([1, 2]), DataFrame({"a": [1, 2]})])
def test_take(obj):
    # 检查在原始顺序中取得所有行时是否没有复制
    obj_orig = obj.copy()  # 复制原始对象
    obj2 = obj.take([0, 1])  # 获取原始对象的子集
    assert np.shares_memory(obj2.values, obj.values)  # 断言 obj2 的值数组与原始对象的值数组共享内存

    obj2.iloc[0] = 0  # 修改 obj2 的第一行的值为 0
    assert not np.shares_memory(obj2.values, obj.values)  # 断言 obj2 的值数组与原始对象的值数组不共享内存
    tm.assert_equal(obj, obj_orig)  # 使用测试框架断言两个对象相等


@pytest.mark.parametrize("obj", [Series([1, 2]), DataFrame({"a": [1, 2]})])
def test_between_time(obj):
    obj.index = date_range("2018-04-09", periods=2, freq="1D20min")  # 设置对象的索引
    obj_orig = obj.copy()  # 复制原始对象
    obj2 = obj.between_time("0:00", "1:00")  # 获取对象在指定时间范围内的数据
    assert np.shares_memory(obj2.values, obj.values)  # 断言 obj2 的值数组与原始对象的值数组共享内存

    obj2.iloc[0] = 0  # 修改 obj2 的第一行的值为 0
    assert not np.shares_memory(obj2.values, obj.values)  # 断言 obj2 的值数组与原始对象的值数组不共享内存
    tm.assert_equal(obj, obj_orig)  # 使用测试框架断言两个对象相等


def test_reindex_like():
    df = DataFrame({"a": [1, 2], "b": "a"})  # 创建包含列 'a', 'b' 的 DataFrame
    other = DataFrame({"b": "a", "a": [1, 2]})  # 创建另一个与 df 结构相同但列顺序不同的 DataFrame

    df_orig = df.copy()  # 复制原始 DataFrame
    df2 = df.reindex_like(other)  # 根据 other 的结构重新索引 df
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))  # 断言重新索引后的 'a' 列数组与原始的 'a' 列数组共享内存

    df2.iloc[0, 1] = 0  # 修改重新索引后的 DataFrame 的第一行第二列的值为 0
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))  # 断言重新索引后的 'a' 列数组与原始的 'a' 列数组不共享内存
    tm.assert_frame_equal(df, df_orig)  # 使用测试框架断言两个 DataFrame 相等


def test_sort_index():
    # GH 49473
    ser = Series([1, 2, 3])  # 创建包含数据 [1, 2, 3] 的 Series
    ser_orig = ser.copy()  # 复制原始 Series
    ser2 = ser.sort_index()  # 对 Series 进行索引排序
    # 使用 NumPy 函数检查两个 Pandas Series 是否共享内存
    assert np.shares_memory(ser.values, ser2.values)
    
    # 改变 ser 的值会触发列或块的写时复制（copy-on-write）
    ser2.iloc[0] = 0
    
    # 再次检查两个 Pandas Series 是否共享内存，期望它们不再共享
    assert not np.shares_memory(ser2.values, ser.values)
    
    # 使用 Pandas 的测试函数确保修改后的 ser2 和原始的 ser 相等
    tm.assert_series_equal(ser, ser_orig)
@pytest.mark.parametrize(
    "obj, kwargs",
    [(Series([1, 2, 3], name="a"), {}), (DataFrame({"a": [1, 2, 3]}), {"by": "a"})],
)
# 参数化测试函数，测试不同的对象和参数组合
def test_sort_values(obj, kwargs):
    # 复制原始对象
    obj_orig = obj.copy()
    # 对对象进行排序，使用传入的参数
    obj2 = obj.sort_values(**kwargs)
    # 断言两个数组是否共享内存
    assert np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))

    # 修改 obj2 的第一个元素，触发对列/块的写时复制
    obj2.iloc[0] = 0
    # 断言两个数组是否不共享内存
    assert not np.shares_memory(get_array(obj2, "a"), get_array(obj, "a"))
    # 使用测试工具函数检查 obj 和 obj_orig 是否相等
    tm.assert_equal(obj, obj_orig)


@pytest.mark.parametrize(
    "obj, kwargs",
    [(Series([1, 2, 3], name="a"), {}), (DataFrame({"a": [1, 2, 3]}), {"by": "a"})],
)
# 参数化测试函数，测试不同的对象和参数组合
def test_sort_values_inplace(obj, kwargs):
    # 复制原始对象
    obj_orig = obj.copy()
    # 创建对象的视图
    view = obj[:]
    # 就地排序对象，使用传入的参数
    obj.sort_values(inplace=True, **kwargs)

    # 断言两个数组是否共享内存
    assert np.shares_memory(get_array(obj, "a"), get_array(view, "a"))

    # 修改 obj 的第一个元素，触发对列/块的写时复制
    obj.iloc[0] = 0
    # 断言两个数组是否不共享内存
    assert not np.shares_memory(get_array(obj, "a"), get_array(view, "a"))
    # 使用测试工具函数检查 view 和 obj_orig 是否相等
    tm.assert_equal(view, obj_orig)


@pytest.mark.parametrize("decimals", [-1, 0, 1])
# 参数化测试函数，测试不同的小数位数
def test_round(decimals):
    # 创建 DataFrame 对象
    df = DataFrame({"a": [1, 2], "b": "c"})
    # 复制原始对象
    df_orig = df.copy()
    # 对 DataFrame 进行舍入操作，使用传入的小数位数
    df2 = df.round(decimals=decimals)

    # 断言两个数组是否共享内存
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    # 如果小数位数大于等于 0
    if decimals >= 0:
        # 确保如果操作无效则进行惰性复制
        assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    else:
        # 断言两个数组是否不共享内存
        assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # 修改 df2 的第一个元素的值为 "d" 和 4
    df2.iloc[0, 1] = "d"
    df2.iloc[0, 0] = 4
    # 断言两个数组是否不共享内存
    assert not np.shares_memory(get_array(df2, "b"), get_array(df, "b"))
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 使用测试工具函数检查 df 和 df_orig 是否相等
    tm.assert_frame_equal(df, df_orig)


# 测试重新排序级别的函数
def test_reorder_levels():
    # 创建 MultiIndex 对象
    index = MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1), (2, 2)], names=["one", "two"]
    )
    # 创建 DataFrame 对象
    df = DataFrame({"a": [1, 2, 3, 4]}, index=index)
    # 复制原始对象
    df_orig = df.copy()
    # 对 DataFrame 对象重新排序级别，使用指定的顺序
    df2 = df.reorder_levels(order=["two", "one"])
    # 断言两个数组是否共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # 修改 df2 的第一个元素的值为 0
    df2.iloc[0, 0] = 0
    # 断言两个数组是否不共享内存
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 使用测试工具函数检查 df 和 df_orig 是否相等
    tm.assert_frame_equal(df, df_orig)


# 测试 Series 对象重新排序级别的函数
def test_series_reorder_levels():
    # 创建 MultiIndex 对象
    index = MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1), (2, 2)], names=["one", "two"]
    )
    # 创建 Series 对象
    ser = Series([1, 2, 3, 4], index=index)
    # 复制原始对象
    ser_orig = ser.copy()
    # 对 Series 对象重新排序级别，使用指定的顺序
    ser2 = ser.reorder_levels(order=["two", "one"])
    # 断言两个数组是否共享内存
    assert np.shares_memory(ser2.values, ser.values)

    # 修改 ser2 的第一个元素的值为 0
    ser2.iloc[0] = 0
    # 断言两个数组是否不共享内存
    assert not np.shares_memory(ser2.values, ser.values)
    # 使用测试工具函数检查 ser 和 ser_orig 是否相等
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize("obj", [Series([1, 2, 3]), DataFrame({"a": [1, 2, 3]})])
# 参数化测试函数，测试不同的对象
def test_swaplevel(obj):
    # 创建 MultiIndex 对象
    index = MultiIndex.from_tuples([(1, 1), (1, 2), (2, 1)], names=["one", "two"])
    # 设置对象的索引
    obj.index = index
    # 复制原始对象
    obj_orig = obj.copy()
    # 对对象进行级别交换
    obj2 = obj.swaplevel()
    # 断言：验证 obj2 的值数组与 obj 的值数组共享内存
    assert np.shares_memory(obj2.values, obj.values)
    
    # 修改 obj2 的第一个元素为 0
    obj2.iloc[0] = 0
    
    # 断言：验证 obj2 的值数组与 obj 的值数组不再共享内存
    assert not np.shares_memory(obj2.values, obj.values)
    
    # 使用测试工具（tm.assert_equal）验证 obj 与 obj_orig 是否相等
    tm.assert_equal(obj, obj_orig)
def test_frame_set_axis():
    # GH 49473
    # 创建一个 DataFrame 包含三列数据
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [0.1, 0.2, 0.3]})
    # 复制原始 DataFrame
    df_orig = df.copy()
    # 使用 set_axis 方法重新设置索引为 ["a", "b", "c"]，返回新的 DataFrame
    df2 = df.set_axis(["a", "b", "c"], axis="index")

    # 断言：新旧两个 DataFrame 的 "a" 列共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # 修改 df2 的一个元素会触发对该列/块的写时复制
    df2.iloc[0, 0] = 0
    # 断言：新旧两个 DataFrame 的 "a" 列不再共享内存
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言：确保 df 和 df_orig 相等
    tm.assert_frame_equal(df, df_orig)


def test_series_set_axis():
    # GH 49473
    # 创建一个 Series 包含三个元素
    ser = Series([1, 2, 3])
    # 复制原始 Series
    ser_orig = ser.copy()
    # 使用 set_axis 方法重新设置索引为 ["a", "b", "c"]，返回新的 Series
    ser2 = ser.set_axis(["a", "b", "c"], axis="index")
    # 断言：新旧两个 Series 共享内存
    assert np.shares_memory(ser, ser2)

    # 修改 ser2 的一个元素会触发对该列/块的写时复制
    ser2.iloc[0] = 0
    # 断言：新旧两个 Series 不再共享内存
    assert not np.shares_memory(ser2, ser)
    # 断言：确保 ser 和 ser_orig 相等
    tm.assert_series_equal(ser, ser_orig)


def test_set_flags():
    # 创建一个 Series 包含三个元素
    ser = Series([1, 2, 3])
    # 复制原始 Series
    ser_orig = ser.copy()
    # 使用 set_flags 方法设置不允许重复标签，返回新的 Series
    ser2 = ser.set_flags(allows_duplicate_labels=False)

    # 断言：新旧两个 Series 共享内存
    assert np.shares_memory(ser, ser2)

    # 修改 ser2 的一个元素会触发对该列/块的写时复制
    ser2.iloc[0] = 0
    # 断言：新旧两个 Series 不再共享内存
    assert not np.shares_memory(ser2, ser)
    # 断言：确保 ser 和 ser_orig 相等
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize("kwargs", [{"mapper": "test"}, {"index": "test"}])
def test_rename_axis(kwargs):
    # GH 49473
    # 创建一个 DataFrame 包含一列数据，设置索引名称为 "a"
    df = DataFrame({"a": [1, 2, 3, 4]}, index=Index([1, 2, 3, 4], name="a"))
    # 复制原始 DataFrame
    df_orig = df.copy()
    # 使用 rename_axis 方法根据参数重新命名索引轴，返回新的 DataFrame
    df2 = df.rename_axis(**kwargs)
    # 断言：新旧两个 DataFrame 的 "a" 列共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # 修改 df2 的一个元素会触发对该列/块的写时复制
    df2.iloc[0, 0] = 0
    # 断言：新旧两个 DataFrame 的 "a" 列不再共享内存
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言：确保 df 和 df_orig 相等
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "func, tz", [("tz_convert", "Europe/Berlin"), ("tz_localize", None)]
)
def test_tz_convert_localize(func, tz):
    # GH 49473
    # 创建一个 Series 包含两个元素，设置时间索引，带有时区信息
    ser = Series(
        [1, 2], index=date_range(start="2014-08-01 09:00", freq="h", periods=2, tz=tz)
    )
    # 复制原始 Series
    ser_orig = ser.copy()
    # 调用 tz_convert 或 tz_localize 方法转换/本地化时区，返回新的 Series
    ser2 = getattr(ser, func)("US/Central")
    # 断言：新旧两个 Series 的值共享内存
    assert np.shares_memory(ser.values, ser2.values)

    # 修改 ser2 的一个元素会触发对该列/块的写时复制
    ser2.iloc[0] = 0
    # 断言：新旧两个 Series 的值不再共享内存
    assert not np.shares_memory(ser2.values, ser.values)
    # 断言：确保 ser 和 ser_orig 相等
    tm.assert_series_equal(ser, ser_orig)


def test_droplevel():
    # GH 49473
    # 创建一个 MultiIndex 索引
    index = MultiIndex.from_tuples([(1, 1), (1, 2), (2, 1)], names=["one", "two"])
    # 创建一个 DataFrame 包含三列数据，使用上述 MultiIndex 作为索引
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, index=index)
    # 复制原始 DataFrame
    df_orig = df.copy()
    # 使用 droplevel 方法去除第一级索引，返回新的 DataFrame
    df2 = df.droplevel(0)

    # 断言：新旧两个 DataFrame 的 "c" 列共享内存
    assert np.shares_memory(get_array(df2, "c"), get_array(df, "c"))
    # 断言：新旧两个 DataFrame 的 "a" 列共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # 修改 df2 的一个元素会触发对该列/块的写时复制
    df2.iloc[0, 0] = 0

    # 断言：新旧两个 DataFrame 的 "a" 列不再共享内存
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言：新旧两个 DataFrame 的 "b" 列共享内存
    assert np.shares_memory(get_array(df2, "b"), get_array(df, "b"))

    # 断言：确保 df 和 df_orig 相等
    tm.assert_frame_equal(df, df_orig)


def test_squeeze():
    # 创建一个 DataFrame 包含一列数据
    df = DataFrame({"a": [1, 2, 3]})
    # 复制原始 DataFrame
    df_orig = df.copy()
    # 调用 squeeze 方法，将 DataFrame 中只有一列的情况下转换为 Series
    series = df.squeeze()
    # 确保 series.values 和通过 get_array 函数获取的数组共享内存，即使是因为 squeeze 而触发的
    assert np.shares_memory(series.values, get_array(df, "a"))
    
    # 对压缩了的 series 进行修改会触发该列/块的复制写入操作
    series.iloc[0] = 0
    
    # 确保 series.values 和通过 get_array 函数获取的数组不再共享内存
    assert not np.shares_memory(series.values, get_array(df, "a"))
    
    # 使用断言验证 df 和 df_orig 在内容上是否相等
    tm.assert_frame_equal(df, df_orig)
# 定义测试函数 `test_items`
def test_items():
    # 创建包含三列的数据框 df
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    # 复制 df 到 df_orig
    df_orig = df.copy()

    # 两次测试，第二次会触发项目缓存，确保在那时也能正常工作
    for i in range(2):
        # 遍历 df 的列名和序列
        for name, ser in df.items():
            # 断言 get_array(ser, name) 和 get_array(df, name) 共享内存
            assert np.shares_memory(get_array(ser, name), get_array(df, name))

            # 修改 ser 的第一个元素会触发写时复制（copy-on-write）
            ser.iloc[0] = 0

            # 断言 get_array(ser, name) 和 get_array(df, name) 不再共享内存
            assert not np.shares_memory(get_array(ser, name), get_array(df, name))
            # 断言 df 和 df_orig 相等
            tm.assert_frame_equal(df, df_orig)


# 使用参数化装饰器，测试函数 test_putmask
@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_putmask(dtype):
    # 创建包含三列的数据框 df，指定数据类型为 dtype
    df = DataFrame({"a": [1, 2], "b": 1, "c": 2}, dtype=dtype)
    # 创建 df 的视图 view
    view = df[:]
    # 复制 df 到 df_orig
    df_orig = df.copy()
    # 将 df 中满足条件 df == df 的元素设为 5
    df[df == df] = 5

    # 断言 get_array(view, "a") 和 get_array(df, "a") 不共享内存
    assert not np.shares_memory(get_array(view, "a"), get_array(df, "a"))
    # 断言 view 和 df_orig 相等
    tm.assert_frame_equal(view, df_orig)


# 使用参数化装饰器，测试函数 test_putmask_no_reference
@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_putmask_no_reference(dtype):
    # 创建包含三列的数据框 df，指定数据类型为 dtype
    df = DataFrame({"a": [1, 2], "b": 1, "c": 2}, dtype=dtype)
    # 获取列 "a" 的数组 arr_a
    arr_a = get_array(df, "a")
    # 将 df 中满足条件 df == df 的元素设为 5
    df[df == df] = 5
    # 断言 arr_a 和 get_array(df, "a") 共享内存
    assert np.shares_memory(arr_a, get_array(df, "a"))


# 使用参数化装饰器，测试函数 test_putmask_aligns_rhs_no_reference
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_putmask_aligns_rhs_no_reference(dtype):
    # 创建包含两列的数据框 df，指定数据类型为 dtype
    df = DataFrame({"a": [1.5, 2], "b": 1.5}, dtype=dtype)
    # 获取列 "a" 的数组 arr_a
    arr_a = get_array(df, "a")
    # 将 df 中满足条件 df == df 的元素设为包含两行的数据框
    df[df == df] = DataFrame({"a": [5.5, 5]})
    # 断言 arr_a 和 get_array(df, "a") 共享内存
    assert np.shares_memory(arr_a, get_array(df, "a"))


# 使用参数化装饰器，测试函数 test_putmask_dont_copy_some_blocks
@pytest.mark.parametrize(
    "val, exp, warn", [(5.5, True, FutureWarning), (5, False, None)]
)
def test_putmask_dont_copy_some_blocks(val, exp, warn):
    # 创建包含三列的数据框 df
    df = DataFrame({"a": [1, 2], "b": 1, "c": 1.5})
    # 创建 df 的视图 view
    view = df[:]
    # 复制 df 到 df_orig
    df_orig = df.copy()
    # 创建 indexer 数据框，用于选择元素
    indexer = DataFrame(
        [[True, False, False], [True, False, False]], columns=list("abc")
    )
    # 使用警告断言，测试赋值操作是否会触发警告
    with tm.assert_produces_warning(warn, match="incompatible dtype"):
        df[indexer] = val

    # 断言 get_array(view, "a") 和 get_array(df, "a") 不共享内存
    assert not np.shares_memory(get_array(view, "a"), get_array(df, "a"))
    # 断言 get_array(view, "b") 和 get_array(df, "b") 共享内存与否符合 exp
    assert np.shares_memory(get_array(view, "b"), get_array(df, "b")) is exp
    # 断言 get_array(view, "c") 和 get_array(df, "c") 共享内存
    assert np.shares_memory(get_array(view, "c"), get_array(df, "c"))
    # 断言 df 的内部数据管理器中是否没有引用 1 的块
    assert df._mgr._has_no_reference(1) is not exp
    # 断言 df 的内部数据管理器中是否没有引用 2 的块
    assert not df._mgr._has_no_reference(2)
    # 断言 view 和 df_orig 相等
    tm.assert_frame_equal(view, df_orig)


# 使用参数化装饰器，测试函数 test_where_mask_noop
@pytest.mark.parametrize("dtype", ["int64", "Int64"])
@pytest.mark.parametrize(
    "func",
    [
        lambda ser: ser.where(ser > 0, 10),
        lambda ser: ser.mask(ser <= 0, 10),
    ],
)
def test_where_mask_noop(dtype, func):
    # 创建包含三个元素的系列 ser，指定数据类型为 dtype
    ser = Series([1, 2, 3], dtype=dtype)
    # 复制 ser 到 ser_orig
    ser_orig = ser.copy()

    # 执行 func 函数得到结果 result
    result = func(ser)
    # 断言 get_array(ser) 和 get_array(result) 共享内存
    assert np.shares_memory(get_array(ser), get_array(result))

    # 修改 result 的第一个元素为 10
    result.iloc[0] = 10
    # 断言 get_array(ser) 和 get_array(result) 不再共享内存
    assert not np.shares_memory(get_array(ser), get_array(result))
    # 断言 ser 和 ser_orig 相等
    tm.assert_series_equal(ser, ser_orig)
    [
        # 创建一个 lambda 函数，接受一个参数 ser，将 ser 中小于 0 的值替换为 10，其他值保持不变
        lambda ser: ser.where(ser < 0, 10),
    
        # 创建一个 lambda 函数，接受一个参数 ser，将 ser 中大于等于 0 的值替换为 10，其他值保持不变
        lambda ser: ser.mask(ser >= 0, 10),
    ],
def test_where_mask(dtype, func):
    # 创建一个包含整数的序列
    ser = Series([1, 2, 3], dtype=dtype)
    # 复制原始序列
    ser_orig = ser.copy()

    # 调用给定的函数处理序列
    result = func(ser)

    # 检查结果与原始序列不共享内存
    assert not np.shares_memory(get_array(ser), get_array(result))
    # 检查处理后的序列与原始序列相等
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize("dtype, val", [("int64", 10.5), ("Int64", 10)])
@pytest.mark.parametrize(
    "func",
    [
        lambda df, val: df.where(df < 0, val),
        lambda df, val: df.mask(df >= 0, val),
    ],
)
def test_where_mask_noop_on_single_column(dtype, val, func):
    # 创建包含两列的数据框
    df = DataFrame({"a": [1, 2, 3], "b": [-4, -5, -6]}, dtype=dtype)
    # 复制原始数据框
    df_orig = df.copy()

    # 调用函数处理数据框
    result = func(df, val)
    # 检查结果中的'b'列与原始数据框中的'b'列共享内存
    assert np.shares_memory(get_array(df, "b"), get_array(result, "b"))
    # 检查结果中的'a'列与原始数据框中的'a'列不共享内存
    assert not np.shares_memory(get_array(df, "a"), get_array(result, "a"))

    # 修改结果中的元素
    result.iloc[0, 1] = 10
    # 确保修改后的'b'列与原始数据框中的'b'列不共享内存
    assert not np.shares_memory(get_array(df, "b"), get_array(result, "b"))
    # 检查数据框与原始数据框相等
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("func", ["mask", "where"])
def test_chained_where_mask(func):
    # 创建包含两列的数据框
    df = DataFrame({"a": [1, 4, 2], "b": 1})
    # 复制原始数据框
    df_orig = df.copy()
    
    # 检查对单列执行链式操作时是否引发异常
    with tm.raises_chained_assignment_error():
        getattr(df["a"], func)(df["a"] > 2, 5, inplace=True)
    # 检查数据框是否与原始数据框相等
    tm.assert_frame_equal(df, df_orig)

    # 检查对多列执行链式操作时是否引发异常
    with tm.raises_chained_assignment_error():
        getattr(df[["a"]], func)(df["a"] > 2, 5, inplace=True)
    # 检查数据框是否与原始数据框相等
    tm.assert_frame_equal(df, df_orig)


def test_asfreq_noop():
    # 创建包含缺失值的数据框
    df = DataFrame(
        {"a": [0.0, None, 2.0, 3.0]},
        index=date_range("1/1/2000", periods=4, freq="min"),
    )
    # 复制原始数据框
    df_orig = df.copy()
    # 对数据框进行频率转换
    df2 = df.asfreq(freq="min")
    # 检查转换后的'a'列与原始数据框中的'a'列共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))

    # 修改转换后的数据框中的元素，触发复制
    df2.iloc[0, 0] = 0
    # 确保修改后的'a'列与原始数据框中的'a'列不共享内存
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 检查数据框与原始数据框相等
    tm.assert_frame_equal(df, df_orig)


def test_iterrows():
    # 创建包含两列的数据框
    df = DataFrame({"a": 0, "b": 1}, index=[1, 2, 3])
    # 复制原始数据框
    df_orig = df.copy()

    # 使用iterrows迭代数据框
    for _, sub in df.iterrows():
        # 修改迭代得到的子数据框中的元素
        sub.iloc[0] = 100
    # 检查数据框与原始数据框相等
    tm.assert_frame_equal(df, df_orig)


def test_interpolate_creates_copy():
    # 创建包含缺失值的数据框
    df = DataFrame({"a": [1.5, np.nan, 3]})
    # 使用切片创建视图
    view = df[:]
    # 复制原始数据框
    expected = df.copy()

    # 前向填充缺失值，原地修改
    df.ffill(inplace=True)
    # 修改数据框中的元素
    df.iloc[0, 0] = 100.5
    # 检查视图与原始数据框相等
    tm.assert_frame_equal(view, expected)


def test_isetitem():
    # 创建包含三列的数据框
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    # 复制原始数据框
    df_orig = df.copy()
    # 浅复制数据框，触发写时复制
    df2 = df.copy(deep=False)
    # 使用isetitem原地修改数据框中的元素
    df2.isetitem(1, np.array([-1, -2, -3]))  # This is inplace
    # 检查修改后的'c'列与原始数据框中的'c'列共享内存
    assert np.shares_memory(get_array(df, "c"), get_array(df2, "c"))
    # 检查'a'列与原始数据框中的'a'列共享内存
    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    # 修改df2中的元素
    df2.loc[0, "a"] = 0
    # 检查数据框与原始数据框相等
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
def test_isetitem_series(dtype):
    # 略
    # 创建一个 DataFrame，包含两列 'a' 和 'b'，'a' 列使用列表，'b' 列使用 NumPy 数组初始化
    df = DataFrame({"a": [1, 2, 3], "b": np.array([4, 5, 6], dtype=dtype)})
    
    # 创建一个 Series，包含元素 [7, 8, 9]
    ser = Series([7, 8, 9])
    
    # 复制 Series，得到一个原始副本
    ser_orig = ser.copy()
    
    # 将 Series 对象 ser 插入到 DataFrame 的第 0 行
    df.isetitem(0, ser)
    
    # 断言：验证 'a' 列的 NumPy 数组与 Series 共享内存
    assert np.shares_memory(get_array(df, "a"), get_array(ser))
    
    # 断言：验证 DataFrame 的内部管理器在第 0 行没有无引用的引用
    assert not df._mgr._has_no_reference(0)
    
    # 对 DataFrame 进行变异操作，修改 'a' 列的第 0 行为 0，验证 Series ser_orig 未受影响
    df.loc[0, "a"] = 0
    tm.assert_series_equal(ser, ser_orig)
    
    # 重新创建一个 DataFrame，与之前相同，重新创建一个 Series
    df = DataFrame({"a": [1, 2, 3], "b": np.array([4, 5, 6], dtype=dtype)})
    ser = Series([7, 8, 9])
    
    # 将 Series 对象 ser 插入到 DataFrame 的第 0 行
    df.isetitem(0, ser)
    
    # 对 Series 进行变异操作，修改其第 0 个元素为 0
    ser.loc[0] = 0
    
    # 预期结果 DataFrame expected，'a' 列的第一个元素变为 0，'b' 列不变
    expected = DataFrame({"a": [7, 8, 9], "b": np.array([4, 5, 6], dtype=dtype)})
    
    # 断言：验证 DataFrame df 与预期结果 expected 相等
    tm.assert_frame_equal(df, expected)
def test_isetitem_frame():
    # 创建一个 DataFrame 对象 df，包含列 'a'、'b'、'c'，并且列 'b'、'c' 的值为标量 1 和 2
    df = DataFrame({"a": [1, 2, 3], "b": 1, "c": 2})
    # 创建一个 DataFrame 对象 rhs，包含列 'a'，且列 'b' 的值为标量 2
    rhs = DataFrame({"a": [4, 5, 6], "b": 2})
    # 将 rhs 的前两行数据更新到 df 中的行 0 和 1
    df.isetitem([0, 1], rhs)
    # 断言 df 的列 'a' 与 rhs 的列 'a' 共享内存
    assert np.shares_memory(get_array(df, "a"), get_array(rhs, "a"))
    # 断言 df 的列 'b' 与 rhs 的列 'b' 共享内存
    assert np.shares_memory(get_array(df, "b"), get_array(rhs, "b"))
    # 断言 df 的管理器中第 0 行并非没有引用
    assert not df._mgr._has_no_reference(0)
    # 创建一个 df 的副本 expected
    expected = df.copy()
    # 修改 rhs 中第一行的数据
    rhs.iloc[0, 0] = 100
    rhs.iloc[0, 1] = 100
    # 使用测试工具函数检验 df 与 expected 是否相等
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("key", ["a", ["a"]])
def test_get(key):
    # 创建一个 DataFrame 对象 df，包含列 'a' 和 'b'
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # 创建 df 的一个副本 df_orig
    df_orig = df.copy()

    # 使用 get 方法获取 key 对应的结果 result
    result = df.get(key)

    # 断言 result 的列 'a' 与 df 的列 'a' 共享内存
    assert np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    # 修改 result 的第一行数据
    result.iloc[0] = 0
    # 断言 result 的列 'a' 与 df 的列 'a' 不共享内存
    assert not np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    # 使用测试工具函数检验 df 与 df_orig 是否相等
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("axis, key", [(0, 0), (1, "a")])
@pytest.mark.parametrize(
    "dtype", ["int64", "float64"], ids=["single-block", "mixed-block"]
)
def test_xs(axis, key, dtype):
    # 根据 dtype 创建一个 DataFrame 对象 df
    single_block = dtype == "int64"
    df = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": np.array([7, 8, 9], dtype=dtype)}
    )
    # 创建 df 的一个副本 df_orig
    df_orig = df.copy()

    # 使用 xs 方法根据 axis 和 key 获取结果 result
    result = df.xs(key, axis=axis)

    # 如果 axis 为 1 或者 single_block 为 True，则断言 df 的列 'a' 与 result 共享内存
    if axis == 1 or single_block:
        assert np.shares_memory(get_array(df, "a"), get_array(result))
    else:
        # 否则，断言 result 的管理器中第 0 行没有引用
        assert result._mgr._has_no_reference(0)

    # 修改 result 的第一行数据
    result.iloc[0] = 0
    # 使用测试工具函数检验 df 与 df_orig 是否相等
    tm.assert_frame_equal(df, df_orig)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("key, level", [("l1", 0), (2, 1)])
def test_xs_multiindex(key, level, axis):
    # 创建一个二维数组 arr 和多级索引 index
    arr = np.arange(18).reshape(6, 3)
    index = MultiIndex.from_product([["l1", "l2"], [1, 2, 3]], names=["lev1", "lev2"])
    df = DataFrame(arr, index=index, columns=list("abc"))
    # 如果 axis 为 1，则转置 df 并创建其副本
    if axis == 1:
        df = df.transpose().copy()
    # 创建 df 的一个副本 df_orig
    df_orig = df.copy()

    # 使用 xs 方法根据 key 和 level 获取结果 result
    result = df.xs(key, level=level, axis=axis)

    # 如果 level 为 0，则断言 df 的第一列与 result 的第一列共享内存
    if level == 0:
        assert np.shares_memory(
            get_array(df, df.columns[0]), get_array(result, result.columns[0])
        )
    # 修改 result 的第一行第一列数据
    result.iloc[0, 0] = 0

    # 使用测试工具函数检验 df 与 df_orig 是否相等
    tm.assert_frame_equal(df, df_orig)


def test_update_frame():
    # 创建两个 DataFrame 对象 df1 和 df2
    df1 = DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    df2 = DataFrame({"b": [100.0]}, index=[1])
    # 创建 df1 的一个副本 df1_orig
    df1_orig = df1.copy()
    # 创建 df1 的视图 view
    view = df1[:]
    # 使用 update 方法将 df2 的数据更新到 df1 中
    df1.update(df2)

    # 创建期望的 DataFrame 对象 expected
    expected = DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 100.0, 6.0]})
    # 使用测试工具函数检验 df1 与 expected 是否相等
    tm.assert_frame_equal(df1, expected)
    # 断言 df1 的列 'a' 与 view 的列 'a' 共享内存
    assert np.shares_memory(get_array(df1, "a"), get_array(view, "a"))
    # 断言 df1 的列 'b' 与 view 的列 'b' 不共享内存
    assert not np.shares_memory(get_array(df1, "b"), get_array(view, "b"))


def test_update_series():
    # 创建两个 Series 对象 ser1 和 ser2
    ser1 = Series([1.0, 2.0, 3.0])
    ser2 = Series([100.0], index=[1])
    # 创建 ser1 的一个副本 ser1_orig
    ser1_orig = ser1.copy()
    # 创建 ser1 的视图 view
    view = ser1[:]

    # 使用 update 方法将 ser2 的数据更新到 ser1 中
    ser1.update(ser2)

    # 创建期望的 Series 对象 expected
    expected = Series([1.0, 100.0, 3.0])
    # 使用测试工具函数检验 ser1 与 expected 是否相等
    tm.assert_series_equal(ser1, expected)
    # 断言 ser1 的数据与 view 的数据不相等
    # 使用测试工具比较两个序列是否相等
    tm.assert_series_equal(view, ser1_orig)
def test_update_chained_assignment():
    # 创建包含单列"a"的DataFrame对象
    df = DataFrame({"a": [1, 2, 3]})
    # 创建包含单个值100.0，索引为1的Series对象
    ser2 = Series([100.0], index=[1])
    # 复制df，创建原始数据的备份
    df_orig = df.copy()
    # 验证在链式赋值错误的情况下引发异常
    with tm.raises_chained_assignment_error():
        df["a"].update(ser2)
    # 验证更新后df与原始df相等
    tm.assert_frame_equal(df, df_orig)

    # 验证在链式赋值错误的情况下引发异常
    with tm.raises_chained_assignment_error():
        df[["a"]].update(ser2.to_frame())
    # 验证更新后df与原始df相等
    tm.assert_frame_equal(df, df_orig)


def test_inplace_arithmetic_series():
    # 创建包含整数值的Series对象
    ser = Series([1, 2, 3])
    # 复制ser，创建原始数据的备份
    ser_orig = ser.copy()
    # 获取ser的数组数据
    data = get_array(ser)
    # 将ser中的所有值乘以2（原地操作）
    ser *= 2
    # 验证原数组数据与更新后的数组数据不共享内存
    assert not np.shares_memory(get_array(ser), data)
    # 验证原数组数据与原始ser的数据相等
    tm.assert_numpy_array_equal(data, get_array(ser_orig))


def test_inplace_arithmetic_series_with_reference():
    # 创建包含整数值的Series对象
    ser = Series([1, 2, 3])
    # 复制ser，创建原始数据的备份
    ser_orig = ser.copy()
    # 创建ser的视图
    view = ser[:]
    # 将ser中的所有值乘以2（原地操作）
    ser *= 2
    # 验证ser与其视图不共享内存
    assert not np.shares_memory(get_array(ser), get_array(view))
    # 验证原始ser与视图相等
    tm.assert_series_equal(ser_orig, view)


def test_transpose():
    # 创建包含两列"a"和"b"的DataFrame对象
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    # 复制df，创建原始数据的备份
    df_orig = df.copy()
    # 对df进行转置操作，返回转置后的DataFrame对象
    result = df.transpose()
    # 验证转置后的第一列与原始df的第一列共享内存
    assert np.shares_memory(get_array(df, "a"), get_array(result, 0))

    # 修改转置后的DataFrame的第一个元素
    result.iloc[0, 0] = 100
    # 验证修改后的转置结果与原始df相等
    tm.assert_frame_equal(df, df_orig)


def test_transpose_different_dtypes():
    # 创建包含两列"a"和"b"的DataFrame对象，其中"b"列为浮点数
    df = DataFrame({"a": [1, 2, 3], "b": 1.5})
    # 复制df，创建原始数据的备份
    df_orig = df.copy()
    # 对df进行转置操作，返回转置后的DataFrame对象
    result = df.T

    # 验证转置后的第一列不与原始df的第一列共享内存
    assert not np.shares_memory(get_array(df, "a"), get_array(result, 0))
    # 修改转置后的DataFrame的第一个元素
    result.iloc[0, 0] = 100
    # 验证修改后的转置结果与原始df相等
    tm.assert_frame_equal(df, df_orig)


def test_transpose_ea_single_column():
    # 创建包含单列"a"的DataFrame对象，列数据类型为Int64
    df = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    # 对df进行转置操作，返回转置后的DataFrame对象
    result = df.T

    # 验证转置后的第一列不与原始df的第一列共享内存
    assert not np.shares_memory(get_array(df, "a"), get_array(result, 0))


def test_transform_frame():
    # 创建包含两列"a"和"b"的DataFrame对象
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    # 复制df，创建原始数据的备份
    df_orig = df.copy()

    # 定义一个用于transform的函数，修改第一行的值为100
    def func(ser):
        ser.iloc[0] = 100
        return ser

    # 对df应用transform函数
    df.transform(func)
    # 验证transform后的df与原始df相等
    tm.assert_frame_equal(df, df_orig)


def test_transform_series():
    # 创建包含整数值的Series对象
    ser = Series([1, 2, 3])
    # 复制ser，创建原始数据的备份
    ser_orig = ser.copy()

    # 定义一个用于transform的函数，修改第一行的值为100
    def func(ser):
        ser.iloc[0] = 100
        return ser

    # 对ser应用transform函数
    ser.transform(func)
    # 验证transform后的ser与原始ser相等
    tm.assert_series_equal(ser, ser_orig)


def test_count_read_only_array():
    # 创建包含两列"a"和"b"的DataFrame对象
    df = DataFrame({"a": [1, 2], "b": 3})
    # 对df进行计数操作，返回包含计数结果的Series对象
    result = df.count()
    # 修改计数结果的第一个元素为100
    result.iloc[0] = 100
    # 创建期望的Series对象
    expected = Series([100, 2], index=["a", "b"])
    # 验证修改后的计数结果与期望的结果相等
    tm.assert_series_equal(result, expected)


def test_insert_series():
    # 创建包含单列"a"的DataFrame对象
    df = DataFrame({"a": [1, 2, 3]})
    # 创建包含整数值的Series对象
    ser = Series([1, 2, 3])
    # 复制ser，创建原始数据的备份
    ser_orig = ser.copy()
    # 在df的第1列位置插入ser作为名为"b"的新列
    df.insert(loc=1, value=ser, column="b")
    # 验证插入的新列与原始ser共享内存
    assert np.shares_memory(get_array(ser), get_array(df, "b"))
    # 验证修改新列的第一个元素后，ser与原始ser相等
    assert not df._mgr._has_no_reference(1)
    df.iloc[0, 1] = 100
    tm.assert_series_equal(ser, ser_orig)


def test_eval():
    # 创建包含两列"a"和"b"的DataFrame对象
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    # 复制df，创建原始数据的备份
    df_orig = df.copy()
    # 使用 DataFrame 的 eval 方法计算新列 "c" 的值，结果存储在 result 中
    result = df.eval("c = a+b")
    # 断言：确保 df 的列 "a" 和 result 的列 "a" 共享相同的内存
    assert np.shares_memory(get_array(df, "a"), get_array(result, "a"))

    # 修改 result 的第一行第一列的值为 100
    result.iloc[0, 0] = 100
    # 使用测试工具 tm 断言：确保修改 result 后的 DataFrame 与原始 df 相等
    tm.assert_frame_equal(df, df_orig)
# 定义一个测试函数，用于测试 DataFrame 的 eval 方法在 inplace 模式下的行为
def test_eval_inplace():
    # 创建一个 DataFrame 包含两列：'a' 和 'b'，'a' 列有三个元素 [1, 2, 3]，'b' 列有一个元素 1
    df = DataFrame({"a": [1, 2, 3], "b": 1})
    # 备份原始 DataFrame
    df_orig = df.copy()
    # 创建 df 的一个视图
    df_view = df[:]

    # 使用 eval 方法计算表达式 "c = a+b"，并将结果保存在新列 'c' 中，同时修改原 DataFrame
    df.eval("c = a+b", inplace=True)
    # 断言 df 的 'a' 列和 df_view 的 'a' 列共享内存，即它们是同一个对象的不同视图
    assert np.shares_memory(get_array(df, "a"), get_array(df_view, "a"))

    # 修改 df 的第一行第一列元素为 100
    df.iloc[0, 0] = 100
    # 断言修改后的 df 与 df_orig 相等
    tm.assert_frame_equal(df_view, df_orig)


# 定义一个测试函数，用于测试 DataFrame 的 apply 方法在修改行时的行为
def test_apply_modify_row():
    # Case 1: 应用一个函数在每一行上作为 Series 对象，函数会改变行对象（需要触发 CoW 如果行是视图）
    # 创建一个 DataFrame 包含两列：'A' 和 'B'，'A' 列有两个元素 [1, 2]，'B' 列有两个元素 [3, 4]
    df = DataFrame({"A": [1, 2], "B": [3, 4]})
    # 备份原始 DataFrame
    df_orig = df.copy()

    # 定义一个函数 transform，它接受一个行 Series 对象，并修改 'B' 列的值为 100
    def transform(row):
        row["B"] = 100
        return row

    # 应用 transform 函数在每一行上，axis=1 表示按行操作
    df.apply(transform, axis=1)

    # 断言修改后的 df 与 df_orig 相等，说明行 Series 是一个副本而不是视图
    tm.assert_frame_equal(df, df_orig)

    # Case 2: row Series 是一个副本
    # 创建一个 DataFrame 包含两列：'A' 和 'B'，'A' 列有两个元素 [1, 2]，'B' 列有两个元素 ["b", "c"]
    df = DataFrame({"A": [1, 2], "B": ["b", "c"]})
    # 备份原始 DataFrame
    df_orig = df.copy()

    # 使用 tm.assert_produces_warning(None) 禁止产生警告
    with tm.assert_produces_warning(None):
        # 应用 transform 函数在每一行上，axis=1 表示按行操作
        df.apply(transform, axis=1)

    # 断言 df 没有改变，与 df_orig 相等
    tm.assert_frame_equal(df, df_orig)
```