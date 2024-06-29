# `D:\src\scipysrc\pandas\pandas\tests\util\test_assert_frame_equal.py`

```
# 导入 pytest 库，用于单元测试
import pytest

# 导入 pandas 库，并从中导入 DataFrame 类
import pandas as pd
from pandas import DataFrame

# 导入 pandas 内部的测试工具模块
import pandas._testing as tm

# 定义一个 pytest fixture，用于参数化测试
@pytest.fixture(params=[True, False])
def by_blocks_fixture(request):
    return request.param

# 定义一个函数，用于比较两个 DataFrame 是否相等，这个函数是私有的
def _assert_frame_equal_both(a, b, **kwargs):
    """
    Check that two DataFrame equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : DataFrame
        The first DataFrame to compare.
    b : DataFrame
        The second DataFrame to compare.
    kwargs : dict
        The arguments passed to `tm.assert_frame_equal`.
    """
    tm.assert_frame_equal(a, b, **kwargs)
    tm.assert_frame_equal(b, a, **kwargs)

# 参数化测试函数，测试当行顺序不匹配时的 DataFrame 相等性
@pytest.mark.parametrize("check_like", [True, False])
def test_frame_equal_row_order_mismatch(check_like, frame_or_series):
    # 创建两个 DataFrame 对象 df1 和 df2，行索引不同
    df1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    df2 = DataFrame({"A": [3, 2, 1], "B": [6, 5, 4]}, index=["c", "b", "a"])

    if not check_like:  # 如果不忽略行列顺序
        msg = f"{frame_or_series.__name__}.index are different"
        # 断言会抛出 AssertionError，匹配错误消息 msg
        with pytest.raises(AssertionError, match=msg):
            tm.assert_frame_equal(
                df1, df2, check_like=check_like, obj=frame_or_series.__name__
            )
    else:
        _assert_frame_equal_both(
            df1, df2, check_like=check_like, obj=frame_or_series.__name__
        )

# 参数化测试函数，测试当形状不匹配时的 DataFrame 相等性
@pytest.mark.parametrize(
    "df1,df2",
    [
        ({"A": [1, 2, 3]}, {"A": [1, 2, 3, 4]}),
        ({"A": [1, 2, 3], "B": [4, 5, 6]}, {"A": [1, 2, 3]}),
    ],
)
def test_frame_equal_shape_mismatch(df1, df2, frame_or_series):
    # 创建两个 DataFrame 对象 df1 和 df2，形状不同
    df1 = DataFrame(df1)
    df2 = DataFrame(df2)
    msg = f"{frame_or_series.__name__} are different"

    # 断言会抛出 AssertionError，匹配错误消息 msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, obj=frame_or_series.__name__)

# 参数化测试函数，测试当索引类型不匹配时的 DataFrame 相等性
@pytest.mark.parametrize(
    "df1,df2,msg",
    [
        # 索引类型不同的 DataFrame 对象
        (
            DataFrame.from_records({"a": [1, 2], "c": ["l1", "l2"]}, index=["a"]),
            DataFrame.from_records({"a": [1.0, 2.0], "c": ["l1", "l2"]}, index=["a"]),
            "DataFrame\\.index are different",
        ),
        # 多级索引类型不同的 DataFrame 对象
        (
            DataFrame.from_records(
                {"a": [1, 2], "b": [2.1, 1.5], "c": ["l1", "l2"]}, index=["a", "b"]
            ),
            DataFrame.from_records(
                {"a": [1.0, 2.0], "b": [2.1, 1.5], "c": ["l1", "l2"]}, index=["a", "b"]
            ),
            "MultiIndex level \\[0\\] are different",
        ),
    ],
)
def test_frame_equal_index_dtype_mismatch(df1, df2, msg, check_index_type):
    kwargs = {"check_index_type": check_index_type}

    if check_index_type:
        # 断言会抛出 AssertionError，匹配错误消息 msg
        with pytest.raises(AssertionError, match=msg):
            tm.assert_frame_equal(df1, df2, **kwargs)
    else:
        tm.assert_frame_equal(df1, df2, **kwargs)

# 测试空 DataFrame 的数据类型
def test_empty_dtypes(check_dtype):
    columns = ["col1", "col2"]
    df1 = DataFrame(columns=columns)
    df2 = DataFrame(columns=columns)
    # 创建一个字典 kwargs，其中包含一个键为 "check_dtype"，对应的值为 check_dtype 变量的值
    kwargs = {"check_dtype": check_dtype}
    
    # 将 df1 数据框中的 "col1" 列转换为 int64 类型
    df1["col1"] = df1["col1"].astype("int64")

    # 如果 check_dtype 为真（True）
    if check_dtype:
        # 定义错误消息，用于匹配 DataFrame 属性不同的断言错误
        msg = r"Attributes of DataFrame\..* are different"
        # 使用 pytest 的 assert_raises 上下文管理器来捕获 AssertionError，并匹配特定的错误消息
        with pytest.raises(AssertionError, match=msg):
            # 断言 df1 和 df2 相等，使用 tm.assert_frame_equal 函数，并传入 kwargs 字典作为关键字参数
            tm.assert_frame_equal(df1, df2, **kwargs)
    else:
        # 如果 check_dtype 不为真，则直接比较 df1 和 df2 是否相等，使用 tm.assert_frame_equal 函数，并传入 kwargs 字典作为关键字参数
        tm.assert_frame_equal(df1, df2, **kwargs)
# 使用 pytest.mark.parametrize 装饰器标记测试函数，参数化 check_like 为 True 和 False
@pytest.mark.parametrize("check_like", [True, False])
def test_frame_equal_index_mismatch(check_like, frame_or_series, using_infer_string):
    # 根据 using_infer_string 的值确定 dtype 的类型
    if using_infer_string:
        dtype = "string"
    else:
        dtype = "object"
    
    # 构造 AssertionError 的匹配消息
    msg = f"""{frame_or_series.__name__}\\.index are different

{frame_or_series.__name__}\\.index values are different \\(33\\.33333 %\\)
\\[left\\]:  Index\\(\\['a', 'b', 'c'\\], dtype='{dtype}'\\)
\\[right\\]: Index\\(\\['a', 'b', 'd'\\], dtype='{dtype}'\\)
At positional index 2, first diff: c != d"""

    # 创建两个 DataFrame 对象 df1 和 df2，每个都有列 A 和 B，但索引不同
    df1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    df2 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "d"])

    # 使用 pytest.raises 捕获 AssertionError 异常，并验证匹配的消息 msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(
            df1, df2, check_like=check_like, obj=frame_or_series.__name__
        )


# 使用 pytest.mark.parametrize 装饰器标记测试函数，参数化 check_like 为 True 和 False
@pytest.mark.parametrize("check_like", [True, False])
def test_frame_equal_columns_mismatch(check_like, frame_or_series, using_infer_string):
    # 根据 using_infer_string 的值确定 dtype 的类型
    if using_infer_string:
        dtype = "string"
    else:
        dtype = "object"
    
    # 构造 AssertionError 的匹配消息
    msg = f"""{frame_or_series.__name__}\\.columns are different

{frame_or_series.__name__}\\.columns values are different \\(50\\.0 %\\)
\\[left\\]:  Index\\(\\['A', 'B'\\], dtype='{dtype}'\\)
\\[right\\]: Index\\(\\['A', 'b'\\], dtype='{dtype}'\\)"""

    # 创建两个 DataFrame 对象 df1 和 df2，df2 的列名 'B' 改为 'b'
    df1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    df2 = DataFrame({"A": [1, 2, 3], "b": [4, 5, 6]}, index=["a", "b", "c"])

    # 使用 pytest.raises 捕获 AssertionError 异常，并验证匹配的消息 msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(
            df1, df2, check_like=check_like, obj=frame_or_series.__name__
        )


# 定义测试函数 test_frame_equal_block_mismatch，参数为 by_blocks_fixture 和 frame_or_series
def test_frame_equal_block_mismatch(by_blocks_fixture, frame_or_series):
    # 获取 frame_or_series 的名称
    obj = frame_or_series.__name__
    
    # 构造 AssertionError 的匹配消息
    msg = f"""{obj}\\.iloc\\[:, 1\\] \\(column name="B"\\) are different

{obj}\\.iloc\\[:, 1\\] \\(column name="B"\\) values are different \\(33\\.33333 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\[4, 5, 6\\]
\\[right\\]: \\[4, 5, 7\\]"""

    # 创建两个 DataFrame 对象 df1 和 df2，其中 df2 的列 B 的值有所不同
    df1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 7]})

    # 使用 pytest.raises 捕获 AssertionError 异常，并验证匹配的消息 msg
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2, by_blocks=by_blocks_fixture, obj=obj)


# 定义参数化测试函数，参数包括 df1, df2, msg
@pytest.mark.parametrize(
    "df1,df2,msg",
    [
        (
            {"A": ["á", "à", "ä"], "E": ["é", "è", "ë"]},
            {"A": ["á", "à", "ä"], "E": ["é", "è", "e̊"]},
            """{obj}\\.iloc\\[:, 1\\] \\(column name="E"\\) are different

{obj}\\.iloc\\[:, 1\\] \\(column name="E"\\) values are different \\(33\\.33333 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\[é, è, ë\\]
\\[right\\]: \\[é, è, e̊\\]""",
        ),
        (
            {"A": ["á", "à", "ä"], "E": ["é", "è", "ë"]},
            {"A": ["a", "a", "a"], "E": ["e", "e", "e"]},
            """{obj}\\.iloc\\[:, 0\\] \\(column name="A"\\) are different

{obj}\\.iloc\\[:, 0\\] \\(column name="A"\\) values are different \\(100\\.0 %\\)
# Import necessary libraries and modules for the test cases
import pytest
import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_frame_equal as tm

# Test function to compare DataFrames for equality with specific Unicode objects
@pytest.mark.parametrize(
    "df1, df2, msg, by_blocks_fixture, frame_or_series",
    [
        (
            {"index": [0, 1, 2]},
            {"left": ['á', 'à', 'ä'], "right": ['a', 'a', 'a']},
            # Message template for the assertion error if DataFrames are not equal
            """\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\[á, à, ä\\]
\\[right\\]: \\[a, a, a\\]""",
        ),
    ],
)
def test_frame_equal_unicode(df1, df2, msg, by_blocks_fixture, frame_or_series):
    # see gh-20503
    #
    # Test ensures that `tm.assert_frame_equals` raises the right exception
    # when comparing DataFrames containing differing unicode objects.
    
    # Convert dictionaries to DataFrames
    df1 = DataFrame(df1)
    df2 = DataFrame(df2)
    
    # Format the error message with the name of the DataFrame or Series
    msg = msg.format(obj=frame_or_series.__name__)
    
    # Check that an AssertionError with the formatted message is raised
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(
            df1, df2, by_blocks=by_blocks_fixture, obj=frame_or_series.__name__
        )


# Test function to check behavior when extension dtype mismatches are ignored
def test_assert_frame_equal_extension_dtype_mismatch():
    # https://github.com/pandas-dev/pandas/issues/32747
    
    # Create DataFrames with different extension dtypes
    left = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    right = left.astype(int)

    # Expected error message when dtypes are checked
    msg = (
        "Attributes of DataFrame\\.iloc\\[:, 0\\] "
        '\\(column name="a"\\) are different\n\n'
        'Attribute "dtype" are different\n'
        "\\[left\\]:  Int64\n"
        "\\[right\\]: int[32|64]"
    )

    # Assert that DataFrames are equal ignoring dtype
    tm.assert_frame_equal(left, right, check_dtype=False)

    # Assert that an AssertionError with the specific message is raised
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(left, right, check_dtype=True)


# Test function to check behavior when interval dtype mismatches are ignored
def test_assert_frame_equal_interval_dtype_mismatch():
    # https://github.com/pandas-dev/pandas/issues/32747
    
    # Create DataFrames with interval dtype and object dtype
    left = DataFrame({"a": [pd.Interval(0, 1)]}, dtype="interval")
    right = left.astype(object)

    # Expected error message when dtypes are checked
    msg = (
        "Attributes of DataFrame\\.iloc\\[:, 0\\] "
        '\\(column name="a"\\) are different\n\n'
        'Attribute "dtype" are different\n'
        "\\[left\\]:  interval\\[int64, right\\]\n"
        "\\[right\\]: object"
    )

    # Assert that DataFrames are equal ignoring dtype
    tm.assert_frame_equal(left, right, check_dtype=False)

    # Assert that an AssertionError with the specific message is raised
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(left, right, check_dtype=True)


# Test function to check behavior when extension dtype mismatches are ignored
def test_assert_frame_equal_ignore_extension_dtype_mismatch():
    # https://github.com/pandas-dev/pandas/issues/35715
    
    # Create DataFrames with different extension dtypes
    left = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    right = DataFrame({"a": [1, 2, 3]}, dtype="Int32")
    
    # Assert that DataFrames are equal ignoring dtype
    tm.assert_frame_equal(left, right, check_dtype=False)


# Test function to check behavior when extension dtype mismatches are ignored across classes
def test_assert_frame_equal_ignore_extension_dtype_mismatch_cross_class():
    # https://github.com/pandas-dev/pandas/issues/35715
    
    # Create DataFrames with different extension dtypes
    left = DataFrame({"a": [1, 2, 3]}, dtype="Int64")
    right = DataFrame({"a": [1, 2, 3]}, dtype="int64")
    
    # Assert that DataFrames are equal ignoring dtype
    tm.assert_frame_equal(left, right, check_dtype=False)


# Test function to check behavior when datetime-like dtype mismatches are ignored
@pytest.mark.parametrize(
    "dtype", ["timedelta64[ns]", "datetime64[ns, UTC]", "Period[D]"]
)
def test_assert_frame_equal_datetime_like_dtype_mismatch(dtype):
    # Create DataFrames with different datetime-like dtypes
    df1 = DataFrame({"a": []}, dtype=dtype)
    df2 = DataFrame({"a": []})
    
    # Assert that DataFrames are equal ignoring dtype
    tm.assert_frame_equal(df1, df2, check_dtype=False)


# Test function to ensure DataFrame constructor behavior with duplicate labels
def test_allows_duplicate_labels():
    # Create empty DataFrames with and without duplicate labels flag set
    left = DataFrame()
    right = DataFrame().set_flags(allows_duplicate_labels=False)
    
    # Assert that each DataFrame is equal to itself
    tm.assert_frame_equal(left, left)
    tm.assert_frame_equal(right, right)
    # 比较两个数据帧 `left` 和 `right` 是否相等，不检查标志位
    tm.assert_frame_equal(left, right, check_flags=False)
    
    # 再次比较两个数据帧 `right` 和 `left` 是否相等，不检查标志位
    tm.assert_frame_equal(right, left, check_flags=False)
    
    # 使用 pytest 断言捕获 Assertion 错误，并匹配错误信息中包含 "<Flags" 的情况
    with pytest.raises(AssertionError, match="<Flags"):
        # 断言两个数据帧 `left` 和 `right` 相等，抛出 AssertionError 如果不相等
        tm.assert_frame_equal(left, right)
    
    # 使用 pytest 断言捕获 Assertion 错误，并匹配错误信息中包含 "<Flags" 的情况
    with pytest.raises(AssertionError, match="<Flags"):
        # 再次断言两个数据帧 `left` 和 `right` 相等，抛出 AssertionError 如果不相等
        tm.assert_frame_equal(left, right)
# 定义一个测试函数，用于验证 DataFrame 是否相等，即使列的数据类型和索引不同
def test_assert_frame_equal_columns_mixed_dtype():
    # GH#39168
    # 创建一个 DataFrame 对象，包含一行数据，列名分别是 "foo"、"bar" 和整数 42，行索引是混合类型
    df = DataFrame([[0, 1, 2]], columns=["foo", "bar", 42], index=[1, "test", 2])
    # 调用测试工具函数 assert_frame_equal 验证 df 和自身相等，检查是否类似
    tm.assert_frame_equal(df, df, check_like=True)


# 定义一个测试函数，用于验证 DataFrame 或 Series 对象是否相等，且数据类型精确匹配
def test_frame_equal_extension_dtype(frame_or_series, any_numeric_ea_dtype):
    # GH#39410
    # 创建一个 DataFrame 或 Series 对象，数据为 [1, 2]，数据类型由参数 any_numeric_ea_dtype 决定
    obj = frame_or_series([1, 2], dtype=any_numeric_ea_dtype)
    # 调用测试工具函数 assert_equal 验证 obj 和自身相等，精确检查
    tm.assert_equal(obj, obj, check_exact=True)


# 定义一个参数化测试函数，验证不同数据类型的 DataFrame 或 Series 对象是否相等
@pytest.mark.parametrize("indexer", [(0, 1), (1, 0)])
def test_frame_equal_mixed_dtypes(frame_or_series, any_numeric_ea_dtype, indexer):
    # GH#39739
    # 定义两种数据类型组合
    dtypes = (any_numeric_ea_dtype, "int64")
    # 创建两个 DataFrame 或 Series 对象，数据为 [1, 2]，数据类型分别由 dtypes 决定
    obj1 = frame_or_series([1, 2], dtype=dtypes[indexer[0]])
    obj2 = frame_or_series([1, 2], dtype=dtypes[indexer[1]])
    # 调用测试工具函数 assert_equal 验证 obj1 和 obj2 相等，精确检查数据值，不检查数据类型
    tm.assert_equal(obj1, obj2, check_exact=True, check_dtype=False)


# 定义一个测试函数，验证两个 DataFrame 对象是否类似，但索引不同
def test_assert_frame_equal_check_like_different_indexes():
    # GH#39739
    # 创建两个空 DataFrame 对象，索引类型分别为 "object" 和 RangeIndex
    df1 = DataFrame(index=pd.Index([], dtype="object"))
    df2 = DataFrame(index=pd.RangeIndex(start=0, stop=0, step=1))
    # 使用 pytest 断言检查调用 assert_frame_equal 函数是否会抛出 AssertionError，并匹配特定错误信息
    with pytest.raises(AssertionError, match="DataFrame.index are different"):
        tm.assert_frame_equal(df1, df2, check_like=True)


# 定义一个测试函数，验证两个 DataFrame 对象在 allows_duplicate_labels 属性上的不同
def test_assert_frame_equal_checking_allow_dups_flag():
    # GH#45554
    # 创建两个 DataFrame 对象，每个包含两行数据
    left = DataFrame([[1, 2], [3, 4]])
    left.flags.allows_duplicate_labels = False

    right = DataFrame([[1, 2], [3, 4]])
    right.flags.allows_duplicate_labels = True
    # 调用测试工具函数 assert_frame_equal 验证 left 和 right 相等，不检查 flags 属性
    tm.assert_frame_equal(left, right, check_flags=False)

    # 使用 pytest 断言检查调用 assert_frame_equal 函数是否会抛出 AssertionError，并匹配特定错误信息
    with pytest.raises(AssertionError, match="allows_duplicate_labels"):
        tm.assert_frame_equal(left, right, check_flags=True)


# 定义一个测试函数，验证两个 DataFrame 对象在类似性检查时，包含分类类型的多级索引
def test_assert_frame_equal_check_like_categorical_midx():
    # GH#48975
    # 创建两个 DataFrame 对象，每个包含三行数据，索引为分类类型的多级索引
    left = DataFrame(
        [[1], [2], [3]],
        index=pd.MultiIndex.from_arrays(
            [
                pd.Categorical(["a", "b", "c"]),
                pd.Categorical(["a", "b", "c"]),
            ]
        ),
    )
    right = DataFrame(
        [[3], [2], [1]],
        index=pd.MultiIndex.from_arrays(
            [
                pd.Categorical(["c", "b", "a"]),
                pd.Categorical(["c", "b", "a"]),
            ]
        ),
    )
    # 调用测试工具函数 assert_frame_equal 验证 left 和 right 相等，检查是否类似
    tm.assert_frame_equal(left, right, check_like=True)


# 定义一个测试函数，验证两个 DataFrame 对象中具有 NA 值的列的比较
def test_assert_frame_equal_ea_column_definition_in_exception_mask():
    # GH#50323
    # 创建两个 DataFrame 对象，每个包含一列数据，其中包含 NA 值
    df1 = DataFrame({"a": pd.Series([pd.NA, 1], dtype="Int64")})
    df2 = DataFrame({"a": pd.Series([1, 1], dtype="Int64")})

    # 设置期望的错误消息模式
    msg = r'DataFrame.iloc\[:, 0\] \(column name="a"\) NA mask values are different'
    # 使用 pytest 断言检查调用 assert_frame_equal 函数是否会抛出 AssertionError，并匹配特定错误信息
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)


# 定义一个测试函数，验证两个 DataFrame 对象中具有不同值的列的比较
def test_assert_frame_equal_ea_column_definition_in_exception():
    # GH#50323
    # 创建两个 DataFrame 对象，每个包含一列数据，其中包含不同的值
    df1 = DataFrame({"a": pd.Series([pd.NA, 1], dtype="Int64")})
    df2 = DataFrame({"a": pd.Series([pd.NA, 2], dtype="Int64")})

    # 设置期望的错误消息模式
    msg = r'DataFrame.iloc\[:, 0\] \(column name="a"\) values are different'
    # 使用 pytest 断言检查调用 assert_frame_equal 函数是否会抛出 AssertionError，并匹配特定错误信息
    with pytest.raises(AssertionError, match=msg):
        tm.assert_frame_equal(df1, df2)
    # 使用 pytest 的上下文管理器来检查是否引发了 AssertionError，并且匹配给定的错误消息
    with pytest.raises(AssertionError, match=msg):
        # 使用 pytest 的 assert_frame_equal 函数来比较两个数据帧 df1 和 df2
        # check_exact=True 表示进行精确的比较，包括顺序和具体数值
        tm.assert_frame_equal(df1, df2, check_exact=True)
# 测试函数，用于验证 assert_frame_equal 函数在处理带有时间戳列的数据帧时是否能正确引发断言错误
def test_assert_frame_equal_ts_column():
    # 创建第一个数据帧 df1，包含名为 "a" 的列，列值为时间戳
    df1 = DataFrame({"a": [pd.Timestamp("2019-12-31"), pd.Timestamp("2020-12-31")]})
    # 创建第二个数据帧 df2，也包含名为 "a" 的列，列值为时间戳，但值略有不同
    df2 = DataFrame({"a": [pd.Timestamp("2020-12-31"), pd.Timestamp("2020-12-31")]})

    # 设置断言错误的消息字符串，用于匹配异常信息
    msg = r'DataFrame.iloc\[:, 0\] \(column name="a"\) values are different'
    # 使用 pytest.raises 捕获 AssertionErro 异常，并验证异常消息是否与 msg 匹配
    with pytest.raises(AssertionError, match=msg):
        # 调用 assert_frame_equal 函数，比较 df1 和 df2 是否相等，预期引发 AssertionError
        tm.assert_frame_equal(df1, df2)


# 测试函数，用于验证 assert_frame_equal 函数在处理包含集合的数据帧时能否正确地进行比较
def test_assert_frame_equal_set():
    # 创建第一个数据帧 df1，包含名为 "set_column" 的列，列值为集合
    df1 = DataFrame({"set_column": [{1, 2, 3}, {4, 5, 6}]})
    # 创建第二个数据帧 df2，包含名为 "set_column" 的列，列值为集合，与 df1 相同
    df2 = DataFrame({"set_column": [{1, 2, 3}, {4, 5, 6}]})

    # 调用 assert_frame_equal 函数，比较 df1 和 df2 是否相等，预期无异常
    tm.assert_frame_equal(df1, df2)


# 测试函数，用于验证 assert_frame_equal 函数在处理包含集合的数据帧时能否正确检测到值不匹配的情况
def test_assert_frame_equal_set_mismatch():
    # 创建第一个数据帧 df1，包含名为 "set_column" 的列，列值为集合
    df1 = DataFrame({"set_column": [{1, 2, 3}, {4, 5, 6}]})
    # 创建第二个数据帧 df2，包含名为 "set_column" 的列，列值为集合，但与 df1 在第二行的值不同
    df2 = DataFrame({"set_column": [{1, 2, 3}, {4, 5, 7}]})

    # 设置断言错误的消息字符串，用于匹配异常信息
    msg = r'DataFrame.iloc\[:, 0\] \(column name="set_column"\) values are different'
    # 使用 pytest.raises 捕获 AssertionErro 异常，并验证异常消息是否与 msg 匹配
    with pytest.raises(AssertionError, match=msg):
        # 调用 assert_frame_equal 函数，比较 df1 和 df2 是否相等，预期引发 AssertionError
        tm.assert_frame_equal(df1, df2)
```