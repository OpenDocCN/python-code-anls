# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_explode.py`

```
# 导入正则表达式模块
import re

# 导入NumPy库并重命名为np
import numpy as np
# 导入pytest测试框架
import pytest

# 导入Pandas库并重命名为pd
import pandas as pd
# 导入Pandas测试工具模块
import pandas._testing as tm


# 定义测试函数test_error，用于测试Pandas DataFrame的explode方法中的错误情况
def test_error():
    # 创建一个包含不同数据类型的DataFrame
    df = pd.DataFrame(
        {"A": pd.Series([[0, 1, 2], np.nan, [], (3, 4)], index=list("abcd")), "B": 1}
    )
    
    # 使用pytest.raises捕获并期望抛出的ValueError异常，匹配指定错误信息
    with pytest.raises(
        ValueError, match="column must be a scalar, tuple, or list thereof"
    ):
        # 调用DataFrame的explode方法，传入一个非法的参数，期望抛出异常
        df.explode([list("AA")])

    # 使用pytest.raises捕获并期望抛出的ValueError异常，匹配指定错误信息
    with pytest.raises(ValueError, match="column must be unique"):
        # 再次调用explode方法，传入使得列不唯一的参数，期望抛出异常
        df.explode(list("AA"))

    # 修改DataFrame的列名为'A'
    df.columns = list("AA")
    # 使用pytest.raises捕获并期望抛出的ValueError异常，匹配指定错误信息
    with pytest.raises(
        ValueError,
        match=re.escape("DataFrame columns must be unique. Duplicate columns: ['A']"),
    ):
        # 第三次调用explode方法，传入使得列不唯一的参数，期望抛出异常
        df.explode("A")


# 使用pytest.mark.parametrize装饰器，定义多组参数化测试用例
@pytest.mark.parametrize(
    "input_subset, error_message",
    [
        (
            list("AC"),
            "columns must have matching element counts",
        ),
        (
            [],
            "column must be nonempty",
        ),
    ],
)
# 定义测试函数test_error_multi_columns，用于测试在多列情况下Pandas DataFrame的explode方法中的错误情况
def test_error_multi_columns(input_subset, error_message):
    # 创建包含不同数据类型和列表的DataFrame
    df = pd.DataFrame(
        {
            "A": [[0, 1, 2], np.nan, [], (3, 4)],
            "B": 1,
            "C": [["a", "b", "c"], "foo", [], ["d", "e", "f"]],
        },
        index=list("abcd"),
    )
    # 使用pytest.raises捕获并期望抛出的ValueError异常，匹配参数化传入的错误信息
    with pytest.raises(ValueError, match=error_message):
        # 调用DataFrame的explode方法，传入参数化的列名列表，期望抛出异常
        df.explode(input_subset)


# 使用pytest.mark.parametrize装饰器，定义多组参数化测试用例
@pytest.mark.parametrize(
    "scalar",
    ["a", 0, 1.5, pd.Timedelta("1 days"), pd.Timestamp("2019-12-31")],
)
# 定义测试函数test_basic，用于测试Pandas DataFrame的explode方法基本功能
def test_basic(scalar):
    # 创建一个包含不同数据类型的DataFrame
    df = pd.DataFrame(
        {scalar: pd.Series([[0, 1, 2], np.nan, [], (3, 4)], index=list("abcd")), "B": 1}
    )
    # 调用DataFrame的explode方法，传入参数化的标量，生成结果DataFrame
    result = df.explode(scalar)
    # 创建预期结果的DataFrame
    expected = pd.DataFrame(
        {
            scalar: pd.Series(
                [0, 1, 2, np.nan, np.nan, 3, 4], index=list("aaabcdd"), dtype=object
            ),
            "B": 1,
        }
    )
    # 使用Pandas测试工具模块tm断言结果DataFrame与预期DataFrame相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数test_multi_index_rows，测试Pandas DataFrame在多级索引行情况下的explode方法
def test_multi_index_rows():
    # 创建一个包含多级索引的DataFrame，其中包含不同数据类型的列
    df = pd.DataFrame(
        {"A": np.array([[0, 1, 2], np.nan, [], (3, 4)], dtype=object), "B": 1},
        index=pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)]),
    )

    # 调用DataFrame的explode方法，对指定列进行展开操作
    result = df.explode("A")
    # 创建预期结果的DataFrame
    expected = pd.DataFrame(
        {
            "A": pd.Series(
                [0, 1, 2, np.nan, np.nan, 3, 4],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("a", 1),
                        ("a", 1),
                        ("a", 1),
                        ("a", 2),
                        ("b", 1),
                        ("b", 2),
                        ("b", 2),
                    ]
                ),
                dtype=object,
            ),
            "B": 1,
        }
    )
    # 使用Pandas测试工具模块tm断言结果DataFrame与预期DataFrame相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数test_multi_index_columns，测试Pandas DataFrame在多级索引列情况下的explode方法
def test_multi_index_columns():
    # 创建一个包含多级索引列的DataFrame，其中包含不同数据类型的列
    df = pd.DataFrame(
        {("A", 1): np.array([[0, 1, 2], np.nan, [], (3, 4)], dtype=object), ("A", 2): 1}
    )

    # 调用DataFrame的explode方法，对指定列进行展开操作
    result = df.explode(("A", 1))
    # 创建预期的 DataFrame 对象，用于与结果进行比较
    expected = pd.DataFrame(
        {
            ("A", 1): pd.Series(
                [0, 1, 2, np.nan, np.nan, 3, 4],  # 创建一个包含指定数据和索引的 Series 对象
                index=pd.Index([0, 0, 0, 1, 2, 3, 3]),  # 指定 Series 对象的索引为一个 Pandas Index 对象
                dtype=object,  # 设置 Series 对象的数据类型为 object
            ),
            ("A", 2): 1,  # 创建一个标量值 1，作为 DataFrame 的一列
        }
    )
    # 使用测试工具 tm.assert_frame_equal 来比较结果 DataFrame 和预期的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
def test_usecase():
    # 创建一个包含指定数据的 DataFrame，使用列名 A, B, C
    # 解析列 B 中的元素并展开，索引使用列 C
    df = pd.DataFrame(
        [[11, range(5), 10], [22, range(3), 20]], columns=list("ABC")
    ).set_index("C")
    # 对 DataFrame 进行 explode 操作，展开列 B
    result = df.explode("B")

    # 创建预期的 DataFrame，包含展开后的数据
    expected = pd.DataFrame(
        {
            "A": [11, 11, 11, 11, 11, 22, 22, 22],
            "B": np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=object),
            "C": [10, 10, 10, 10, 10, 20, 20, 20],
        },
        columns=list("ABC"),
    ).set_index("C")

    # 使用测试工具比较结果 DataFrame 和预期 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)

    # GH-8517
    # 创建另一个 DataFrame，包含日期、姓名和文本列
    df = pd.DataFrame(
        [["2014-01-01", "Alice", "A B"], ["2014-01-02", "Bob", "C D"]],
        columns=["dt", "name", "text"],
    )
    # 将文本列按空格拆分为列表，并对 DataFrame 进行 explode 操作
    result = df.assign(text=df.text.str.split(" ")).explode("text")
    # 创建预期的 DataFrame，包含展开后的数据
    expected = pd.DataFrame(
        [
            ["2014-01-01", "Alice", "A"],
            ["2014-01-01", "Alice", "B"],
            ["2014-01-02", "Bob", "C"],
            ["2014-01-02", "Bob", "D"],
        ],
        columns=["dt", "name", "text"],
        index=[0, 0, 1, 1],
    )
    # 使用测试工具比较结果 DataFrame 和预期 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "input_dict, input_index, expected_dict, expected_index",
    [
        (
            {"col1": [[1, 2], [3, 4]], "col2": ["foo", "bar"]},
            [0, 0],
            {"col1": [1, 2, 3, 4], "col2": ["foo", "foo", "bar", "bar"]},
            [0, 0, 0, 0],
        ),
        (
            {"col1": [[1, 2], [3, 4]], "col2": ["foo", "bar"]},
            pd.Index([0, 0], name="my_index"),
            {"col1": [1, 2, 3, 4], "col2": ["foo", "foo", "bar", "bar"]},
            pd.Index([0, 0, 0, 0], name="my_index"),
        ),
        (
            {"col1": [[1, 2], [3, 4]], "col2": ["foo", "bar"]},
            pd.MultiIndex.from_arrays(
                [[0, 0], [1, 1]], names=["my_first_index", "my_second_index"]
            ),
            {"col1": [1, 2, 3, 4], "col2": ["foo", "foo", "bar", "bar"]},
            pd.MultiIndex.from_arrays(
                [[0, 0, 0, 0], [1, 1, 1, 1]],
                names=["my_first_index", "my_second_index"],
            ),
        ),
        (
            {"col1": [[1, 2], [3, 4]], "col2": ["foo", "bar"]},
            pd.MultiIndex.from_arrays([[0, 0], [1, 1]], names=["my_index", None]),
            {"col1": [1, 2, 3, 4], "col2": ["foo", "foo", "bar", "bar"]},
            pd.MultiIndex.from_arrays(
                [[0, 0, 0, 0], [1, 1, 1, 1]], names=["my_index", None]
            ),
        ),
    ],
)
def test_duplicate_index(input_dict, input_index, expected_dict, expected_index):
    # GH 28005
    # 根据输入的字典和索引创建 DataFrame 对象，数据类型为 object
    df = pd.DataFrame(input_dict, index=input_index, dtype=object)
    # 对 DataFrame 进行 explode 操作，展开列 col1
    result = df.explode("col1")
    # 创建预期的 DataFrame，包含展开后的数据
    expected = pd.DataFrame(expected_dict, index=expected_index, dtype=object)
    # 使用测试工具比较结果 DataFrame 和预期 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


def test_ignore_index():
    # GH 34932
    # 创建包含 id 和 values 列的 DataFrame
    df = pd.DataFrame({"id": range(0, 20, 10), "values": [list("ab"), list("cd")]})
    # 对 values 列进行 explode 操作，忽略索引重置
    result = df.explode("values", ignore_index=True)
    # 创建预期的 Pandas DataFrame 对象，包含两列："id" 和 "values"。
    # "id" 列包含值 [0, 0, 10, 10]，"values" 列包含字符列表 ['a', 'b', 'c', 'd']。
    expected = pd.DataFrame(
        {"id": [0, 0, 10, 10], "values": list("abcd")}, index=[0, 1, 2, 3]
    )
    
    # 使用测试框架中的函数 tm.assert_frame_equal() 比较 result 和 expected 两个 DataFrame 是否相等。
    tm.assert_frame_equal(result, expected)
# 测试函数，用于验证 Pandas DataFrame 的 explode 方法在处理集合类型列时的行为
def test_explode_sets():
    # 创建包含集合类型列的 DataFrame，每个集合包含字符串 "x" 和 "y"
    df = pd.DataFrame({"a": [{"x", "y"}], "b": [1]}, index=[1])
    # 对列 'a' 使用 explode 方法，然后按列 'a' 的值排序结果
    result = df.explode(column="a").sort_values(by="a")
    # 创建预期的 DataFrame，展开集合类型列 'a' 为单独的行，并根据列 'a' 排序
    expected = pd.DataFrame({"a": ["x", "y"], "b": [1, 1]}, index=[1, 1])
    # 使用测试工具比较结果 DataFrame 和预期的 DataFrame
    tm.assert_frame_equal(result, expected)


# 使用参数化测试装饰器进行多组测试数据的测试
@pytest.mark.parametrize(
    "input_subset, expected_dict, expected_index",
    [
        (
            # 第一组测试数据
            list("AC"),
            {
                "A": pd.Series(
                    [0, 1, 2, np.nan, np.nan, 3, 4, np.nan],
                    index=list("aaabcdde"),
                    dtype=object,
                ),
                "B": 1,
                "C": ["a", "b", "c", "foo", np.nan, "d", "e", np.nan],
            },
            list("aaabcdde"),
        ),
        (
            # 第二组测试数据
            list("A"),
            {
                "A": pd.Series(
                    [0, 1, 2, np.nan, np.nan, 3, 4, np.nan],
                    index=list("aaabcdde"),
                    dtype=object,
                ),
                "B": 1,
                "C": [
                    ["a", "b", "c"],
                    ["a", "b", "c"],
                    ["a", "b", "c"],
                    "foo",
                    [],
                    ["d", "e"],
                    ["d", "e"],
                    np.nan,
                ],
            },
            list("aaabcdde"),
        ),
    ],
)
# 测试函数，验证 Pandas DataFrame 的 explode 方法在处理多列时的行为
def test_multi_columns(input_subset, expected_dict, expected_index):
    # 创建包含列表类型列 'A'、标量 'B' 和列表类型列 'C' 的 DataFrame
    df = pd.DataFrame(
        {
            "A": [[0, 1, 2], np.nan, [], (3, 4), np.nan],
            "B": 1,
            "C": [["a", "b", "c"], "foo", [], ["d", "e"], np.nan],
        },
        index=list("abcde"),
    )
    # 对输入的列名列表使用 explode 方法，展开对应列为单独的行
    result = df.explode(input_subset)
    # 创建预期的 DataFrame，根据参数化测试的预期字典和索引
    expected = pd.DataFrame(expected_dict, expected_index)
    # 使用测试工具比较结果 DataFrame 和预期的 DataFrame
    tm.assert_frame_equal(result, expected)


# 测试函数，验证 Pandas DataFrame 的 explode 方法在处理 NaN 和空列表时的行为
def test_multi_columns_nan_empty():
    # 创建包含列表类型列 'A'、 'B' 和列表类型列 'C' 的 DataFrame
    df = pd.DataFrame(
        {
            "A": [[0, 1], [5], [], [2, 3]],
            "B": [9, 8, 7, 6],
            "C": [[1, 2], np.nan, [], [3, 4]],
        }
    )
    # 对列 'A' 和 'C' 使用 explode 方法，展开对应列为单独的行
    result = df.explode(["A", "C"])
    # 创建预期的 DataFrame，包含根据 NaN 和空列表扩展后的列 'A' 和 'C'
    expected = pd.DataFrame(
        {
            "A": np.array([0, 1, 5, np.nan, 2, 3], dtype=object),
            "B": [9, 9, 8, 7, 6, 6],
            "C": np.array([1, 2, np.nan, np.nan, 3, 4], dtype=object),
        },
        index=[0, 0, 1, 2, 3, 3],
    )
    # 使用测试工具比较结果 DataFrame 和预期的 DataFrame
    tm.assert_frame_equal(result, expected)
```