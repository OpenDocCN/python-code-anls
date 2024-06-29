# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_mangle_dupes.py`

```
"""
Tests that duplicate columns are handled appropriately when parsed by the
CSV engine. In general, the expected result is that they are either thoroughly
de-duplicated (if mangling requested) or ignored otherwise.
"""

# 从io库中导入StringIO，用于创建内存中的文件对象
from io import StringIO

# 导入pytest库
import pytest

# 从pandas库中导入DataFrame类
from pandas import DataFrame

# 导入pandas的测试模块
import pandas._testing as tm

# 对于xfail_pyarrow标记，用于标记测试在使用pyarrow时预期会失败
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")

# 标记整个模块的警告过滤器，忽略特定的警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@xfail_pyarrow  # 如果使用pyarrow引擎会导致ValueError: Found non-unique column index
def test_basic(all_parsers):
    # 选择所有解析器中的一个来执行测试
    parser = all_parsers

    # 定义测试数据
    data = "a,a,b,b,b\n1,2,3,4,5"
    # 使用所选解析器读取CSV数据，并指定分隔符为逗号
    result = parser.read_csv(StringIO(data), sep=",")

    # 预期的DataFrame结果，用于比较
    expected = DataFrame([[1, 2, 3, 4, 5]], columns=["a", "a.1", "b", "b.1", "b.2"])
    # 使用测试模块中的函数来断言实际结果与预期结果是否相同
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # 如果使用pyarrow引擎会导致ValueError: Found non-unique column index
def test_basic_names(all_parsers):
    # See gh-7160
    # 选择所有解析器中的一个来执行测试
    parser = all_parsers

    # 定义测试数据
    data = "a,b,a\n0,1,2\n3,4,5"
    # 预期的DataFrame结果，用于比较
    expected = DataFrame([[0, 1, 2], [3, 4, 5]], columns=["a", "b", "a.1"])

    # 使用所选解析器读取CSV数据
    result = parser.read_csv(StringIO(data))
    # 使用测试模块中的函数来断言实际结果与预期结果是否相同
    tm.assert_frame_equal(result, expected)


def test_basic_names_raise(all_parsers):
    # See gh-7160
    # 选择所有解析器中的一个来执行测试
    parser = all_parsers

    # 定义测试数据
    data = "0,1,2\n3,4,5"
    # 使用pytest的断言来确保解析器在遇到重复列名时会引发ValueError异常
    with pytest.raises(ValueError, match="Duplicate names"):
        parser.read_csv(StringIO(data), names=["a", "b", "a"])


@xfail_pyarrow  # 如果使用pyarrow引擎会导致ValueError: Found non-unique column index
@pytest.mark.parametrize(
    "data,expected",
    [
        ("a,a,a.1\n1,2,3", DataFrame([[1, 2, 3]], columns=["a", "a.2", "a.1"])),
        (
            "a,a,a.1,a.1.1,a.1.1.1,a.1.1.1.1\n1,2,3,4,5,6",
            DataFrame(
                [[1, 2, 3, 4, 5, 6]],
                columns=["a", "a.2", "a.1", "a.1.1", "a.1.1.1", "a.1.1.1.1"],
            ),
        ),
        (
            "a,a,a.3,a.1,a.2,a,a\n1,2,3,4,5,6,7",
            DataFrame(
                [[1, 2, 3, 4, 5, 6, 7]],
                columns=["a", "a.4", "a.3", "a.1", "a.2", "a.5", "a.6"],
            ),
        ),
    ],
)
def test_thorough_mangle_columns(all_parsers, data, expected):
    # see gh-17060
    # 选择所有解析器中的一个来执行测试
    parser = all_parsers

    # 使用所选解析器读取CSV数据
    result = parser.read_csv(StringIO(data))
    # 使用测试模块中的函数来断言实际结果与预期结果是否相同
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,names,expected",
    [
        (
            "a,b,b\n1,2,3",  # 第一个测试数据：CSV格式字符串，包含两行数据，逗号分隔
            ["a.1", "a.1", "a.1.1"],  # 预期的列名列表，用于创建DataFrame对象
            DataFrame(  # 创建一个DataFrame对象
                [["a", "b", "b"], ["1", "2", "3"]],  # DataFrame的数据，二维列表形式
                columns=["a.1", "a.1.1", "a.1.1.1"]  # DataFrame的列名
            ),
        ),
        (
            "a,b,c,d,e,f\n1,2,3,4,5,6",  # 第二个测试数据：CSV格式字符串，包含两行数据，逗号分隔
            ["a", "a", "a.1", "a.1.1", "a.1.1.1", "a.1.1.1.1"],  # 预期的列名列表，用于创建DataFrame对象
            DataFrame(  # 创建一个DataFrame对象
                [["a", "b", "c", "d", "e", "f"], ["1", "2", "3", "4", "5", "6"]],  # DataFrame的数据，二维列表形式
                columns=["a", "a.1", "a.1.1", "a.1.1.1", "a.1.1.1.1", "a.1.1.1.1.1"]  # DataFrame的列名
            ),
        ),
        (
            "a,b,c,d,e,f,g\n1,2,3,4,5,6,7",  # 第三个测试数据：CSV格式字符串，包含两行数据，逗号分隔
            ["a", "a", "a.3", "a.1", "a.2", "a", "a"],  # 预期的列名列表，用于创建DataFrame对象
            DataFrame(  # 创建一个DataFrame对象
                [
                    ["a", "b", "c", "d", "e", "f", "g"],  # DataFrame的数据，二维列表形式
                    ["1", "2", "3", "4", "5", "6", "7"],
                ],
                columns=["a", "a.1", "a.3", "a.1.1", "a.2", "a.2.1", "a.3.1"]  # DataFrame的列名
            ),
        ),
    ],
# XFAIL 标记测试用例，预期这个测试将失败，因为 DataFrame 的列不同
@xfail_pyarrow  # AssertionError: DataFrame.columns are different
# 测试函数：测试混淆未命名占位符的情况
def test_mangled_unnamed_placeholders(all_parsers):
    # 变量原始键值
    orig_key = "0"
    # 使用给定的解析器对象
    parser = all_parsers

    # 原始值列表
    orig_value = [1, 2, 3]
    # 创建 DataFrame 对象，键为 orig_key，值为 orig_value
    df = DataFrame({orig_key: orig_value})

    # 这个测试递归更新 `df`
    for i in range(3):
        # 期望的 DataFrame 对象
        expected = DataFrame()

        for j in range(i + 1):
            # 构建列名，例如 "Unnamed: 0" 或 "Unnamed: 0.1"
            col_name = "Unnamed: 0" + f".{1 * j}" * min(j, 1)
            # 在期望的 DataFrame 中插入列，值为 [0, 1, 2]
            expected.insert(loc=0, column=col_name, value=[0, 1, 2])

        # 将 orig_key 列添加到期望的 DataFrame 中
        expected[orig_key] = orig_value

        # 读取 DataFrame，并将结果赋给 df
        df = parser.read_csv(StringIO(df.to_csv()))

        # 断言 df 和 expected 是否相等
        tm.assert_frame_equal(df, expected)


# XFAIL 标记测试用例，预期这个测试将失败，因为找到了非唯一的列索引
@xfail_pyarrow  # ValueError: Found non-unique column index
# 测试函数：测试在已存在重复列名的情况下混淆列名
def test_mangle_dupe_cols_already_exists(all_parsers):
    # GH#14704
    # 使用给定的解析器对象
    parser = all_parsers

    # 数据字符串，包含重复的列名和后缀
    data = "a,a,a.1,a,a.3,a.1,a.1.1\n1,2,3,4,5,6,7"
    # 解析数据，并将结果赋给 result
    result = parser.read_csv(StringIO(data))

    # 期望的 DataFrame 对象，指定列名和后缀
    expected = DataFrame(
        [[1, 2, 3, 4, 5, 6, 7]],
        columns=["a", "a.2", "a.1", "a.4", "a.3", "a.1.2", "a.1.1"],
    )

    # 断言 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# XFAIL 标记测试用例，预期这个测试将失败，因为找到了非唯一的列索引
@xfail_pyarrow  # ValueError: Found non-unique column index
# 测试函数：测试在已存在未命名列的情况下混淆列名
def test_mangle_dupe_cols_already_exists_unnamed_col(all_parsers):
    # GH#14704
    # 使用给定的解析器对象
    parser = all_parsers

    # 数据字符串，包含未命名的列和后缀
    data = ",Unnamed: 0,,Unnamed: 2\n1,2,3,4"
    # 解析数据，并将结果赋给 result
    result = parser.read_csv(StringIO(data))

    # 期望的 DataFrame 对象，指定列名和后缀
    expected = DataFrame(
        [[1, 2, 3, 4]],
        columns=["Unnamed: 0.1", "Unnamed: 0", "Unnamed: 2.1", "Unnamed: 2"],
    )

    # 断言 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 参数化测试：使用不同的 usecol 和 engine 参数来测试混淆列名的情况
@pytest.mark.parametrize("usecol, engine", [([0, 1, 1], "python"), ([0, 1, 1], "c")])
# 测试函数：测试在指定列名和引擎下混淆列名的情况
def test_mangle_cols_names(all_parsers, usecol, engine):
    # GH 11823
    # 使用给定的解析器对象
    parser = all_parsers
    # 数据字符串
    data = "1,2,3"
    # 指定列名，包含重复的列名 "A"
    names = ["A", "A", "B"]
    # 断言在读取时是否抛出 ValueError 异常，并包含 "Duplicate names" 字符串
    with pytest.raises(ValueError, match="Duplicate names"):
        parser.read_csv(StringIO(data), names=names, usecols=usecol, engine=engine)
```