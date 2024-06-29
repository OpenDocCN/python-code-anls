# `D:\src\scipysrc\pandas\pandas\tests\io\parser\usecols\test_usecols_basic.py`

```
"""
Tests the usecols functionality during parsing
for all of the parsers defined in parsers.py
"""

# 导入需要的模块和库
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
    DataFrame,
    Index,
    array,
)
import pandas._testing as tm

# 忽略特定警告的标记
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 定义几个错误消息常量
_msg_validate_usecols_arg = (
    "'usecols' must either be list-like "
    "of all strings, all unicode, all "
    "integers or a callable."
)
_msg_validate_usecols_names = (
    "Usecols do not match columns, columns expected but not found: {0}"
)
_msg_pyarrow_requires_names = (
    "The pyarrow engine does not allow 'usecols' to be integer column "
    "positions. Pass a list of string column names instead."
)

# 标记用于在 pytest 中标记需要在 pyarrow 引擎下失败的测试
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")

# 再次忽略特定警告的标记
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame is deprecated:DeprecationWarning"
)

# 测试函数，测试混合数据类型 usecols 时是否能正确抛出 ValueError 异常
def test_raise_on_mixed_dtype_usecols(all_parsers):
    # 准备测试数据和使用的 usecols 参数
    data = """a,b,c
        1000,2000,3000
        4000,5000,6000
        """
    usecols = [0, "b", 2]
    parser = all_parsers

    # 断言应该抛出指定异常并匹配指定的错误消息
    with pytest.raises(ValueError, match=_msg_validate_usecols_arg):
        parser.read_csv(StringIO(data), usecols=usecols)


# 参数化测试函数，测试不同的 usecols 参数
@pytest.mark.parametrize("usecols", [(1, 2), ("b", "c")])
def test_usecols(all_parsers, usecols, request):
    # 准备测试数据和使用的 usecols 参数
    data = """\
a,b,c
1,2,3
4,5,6
7,8,9
10,11,12"""
    parser = all_parsers

    # 如果使用 pyarrow 引擎并且 usecols 的第一个元素是整数，应该抛出 ValueError 异常
    if parser.engine == "pyarrow" and isinstance(usecols[0], int):
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), usecols=usecols)
        return

    # 否则，正常读取数据并进行断言比较
    result = parser.read_csv(StringIO(data), usecols=usecols)
    expected = DataFrame([[2, 3], [5, 6], [8, 9], [11, 12]], columns=["b", "c"])
    tm.assert_frame_equal(result, expected)


# 测试函数，使用指定的列名进行读取数据
def test_usecols_with_names(all_parsers):
    # 准备测试数据和列名
    data = """\
a,b,c
1,2,3
4,5,6
7,8,9
10,11,12"""
    parser = all_parsers
    names = ["foo", "bar"]

    # 如果使用 pyarrow 引擎，应该抛出 ValueError 异常
    if parser.engine == "pyarrow":
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), names=names, usecols=[1, 2], header=0)
        return

    # 否则，正常读取数据并进行断言比较
    result = parser.read_csv(StringIO(data), names=names, usecols=[1, 2], header=0)
    expected = DataFrame([[2, 3], [5, 6], [8, 9], [11, 12]], columns=names)
    tm.assert_frame_equal(result, expected)


# 参数化测试函数，测试相对于列名的 usecols 参数
@pytest.mark.parametrize(
    "names,usecols", [(["b", "c"], [1, 2]), (["a", "b", "c"], ["b", "c"])]
)
def test_usecols_relative_to_names(all_parsers, names, usecols):
    # 准备测试数据和使用的 usecols 参数
    data = """\
1,2,3
4,5,6
7,8,9
10,11,12"""
    parser = all_parsers
    # 如果解析器使用的是 pyarrow 引擎，并且 usecols 的第一个元素不是整数
    if parser.engine == "pyarrow" and not isinstance(usecols[0], int):
        # 抛出 ArrowKeyError：在 include_columns 中列 'fb' 不存在
        # 跳过当前测试，原因是该问题在 Arrow GitHub 上有相关的 issue 记录
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    # 使用 parser 对象读取给定的 CSV 数据，通过 StringIO 将数据包装成文件对象
    result = parser.read_csv(StringIO(data), names=names, header=None, usecols=usecols)

    # 预期的 DataFrame 结果，包含特定数据和列名
    expected = DataFrame([[2, 3], [5, 6], [8, 9], [11, 12]], columns=["b", "c"])
    # 使用测试框架中的 assert 函数比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，测试在指定列名和无标题的情况下，使用指定列索引读取 CSV 数据
def test_usecols_relative_to_names2(all_parsers):
    # 准备测试数据，包括四行三列的 CSV 数据
    data = """\
1,2,3
4,5,6
7,8,9
10,11,12"""
    # 获取测试用的解析器对象
    parser = all_parsers

    # 使用解析器读取 CSV 数据，指定列名为 ["a", "b"]，无标题行，只使用列索引为 [0, 1] 的数据列
    result = parser.read_csv(
        StringIO(data), names=["a", "b"], header=None, usecols=[0, 1]
    )

    # 期望的结果是一个 DataFrame，包含列名为 ["a", "b"] 的四行数据
    expected = DataFrame([[1, 2], [4, 5], [7, 8], [10, 11]], columns=["a", "b"])
    # 断言读取的结果与期望的结果相同
    tm.assert_frame_equal(result, expected)


# 注解：标记为预期失败的测试函数，原因是与 pyarrow 的兼容性问题相关
@xfail_pyarrow
def test_usecols_name_length_conflict(all_parsers):
    # 准备测试数据，包括四行三列的 CSV 数据
    data = """\
1,2,3
4,5,6
7,8,9
10,11,12"""
    # 获取测试用的解析器对象
    parser = all_parsers
    # 准备用于异常断言的消息
    msg = "Number of passed names did not match number of header fields in the file"
    # 使用 pytest 断言，预期读取 CSV 数据时会抛出 ValueError 异常，并匹配指定消息
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), names=["a", "b"], header=None, usecols=[1])


# 定义一个测试函数，测试在只使用单个列名字符串的情况下，是否能正确抛出异常
def test_usecols_single_string(all_parsers):
    # 准备测试数据，包括三行三列的 CSV 数据
    data = """foo, bar, baz
1000, 2000, 3000
4000, 5000, 6000"""
    # 获取测试用的解析器对象
    parser = all_parsers

    # 使用 pytest 断言，预期读取 CSV 数据时会抛出 ValueError 异常，并匹配指定消息
    with pytest.raises(ValueError, match=_msg_validate_usecols_arg):
        parser.read_csv(StringIO(data), usecols="foo")


# 注解：标记为跳过测试的测试函数，原因是与 pyarrow 的兼容性问题相关
@skip_pyarrow
@pytest.mark.parametrize(
    "data", ["a,b,c,d\n1,2,3,4\n5,6,7,8", "a,b,c,d\n1,2,3,4,\n5,6,7,8,"]
)
def test_usecols_index_col_false(all_parsers, data):
    # 准备测试数据，包括两种格式的 CSV 数据
    # 数据格式 1: "a,b,c,d\n1,2,3,4\n5,6,7,8"
    # 数据格式 2: "a,b,c,d\n1,2,3,4,\n5,6,7,8,"
    parser = all_parsers
    # 指定需要使用的列名为 ["a", "c", "d"]
    usecols = ["a", "c", "d"]
    # 准备期望的 DataFrame 结果，包括两行数据
    expected = DataFrame({"a": [1, 5], "c": [3, 7], "d": [4, 8]})

    # 使用解析器读取 CSV 数据，预期不创建索引列
    result = parser.read_csv(StringIO(data), usecols=usecols, index_col=False)
    # 断言读取的结果与期望的结果相同
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试在指定索引列与使用列存在冲突的情况下，读取 CSV 数据
@pytest.mark.parametrize("index_col", ["b", 0])
@pytest.mark.parametrize("usecols", [["b", "c"], [1, 2]])
def test_usecols_index_col_conflict(all_parsers, usecols, index_col, request):
    # 准备测试数据，包括两行两列的 CSV 数据
    data = "a,b,c,d\nA,a,1,one\nB,b,2,two"
    # 获取测试用的解析器对象
    parser = all_parsers

    # 如果解析器引擎是 pyarrow 并且 usecols 的第一个元素是整数，则预期会抛出 ValueError 异常，并匹配指定消息
    if parser.engine == "pyarrow" and isinstance(usecols[0], int):
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), usecols=usecols, index_col=index_col)
        return

    # 准备期望的 DataFrame 结果，包括两行数据，列名为 ["c"]
    expected = DataFrame({"c": [1, 2]}, index=Index(["a", "b"], name="b"))

    # 使用解析器读取 CSV 数据，指定使用的列名和索引列
    result = parser.read_csv(StringIO(data), usecols=usecols, index_col=index_col)
    # 断言读取的结果与期望的结果相同
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试在指定索引列与使用列存在冲突的情况下，读取 CSV 数据
def test_usecols_index_col_conflict2(all_parsers):
    # 准备测试数据，包括两行两列的 CSV 数据
    data = "a,b,c,d\nA,a,1,one\nB,b,2,two"
    # 获取测试用的解析器对象
    parser = all_parsers

    # 准备期望的 DataFrame 结果，包括两行数据，索引列为 ["b", "c"]
    expected = DataFrame({"b": ["a", "b"], "c": [1, 2], "d": ("one", "two")})
    expected = expected.set_index(["b", "c"])

    # 使用解析器读取 CSV 数据，指定使用的列名和索引列
    result = parser.read_csv(
        StringIO(data), usecols=["b", "c", "d"], index_col=["b", "c"]
    )
    # 断言读取的结果与期望的结果相同
    tm.assert_frame_equal(result, expected)


# 注解：标记为跳过测试的测试函数，原因是与 pyarrow 的兼容性问题相关
@skip_pyarrow
def test_usecols_implicit_index_col(all_parsers):
    # 准备测试数据，等待实现
    parser = all_parsers
    # 定义一个包含 CSV 数据的字符串
    data = "a,b,c\n4,apple,bat,5.7\n8,orange,cow,10"
    
    # 使用 pandas 的 read_csv 函数解析 CSV 数据，并指定只使用列 "a" 和 "b"
    result = parser.read_csv(StringIO(data), usecols=["a", "b"])
    
    # 创建期望的 DataFrame，包含列 "a" 和 "b"，并设定索引为 [4, 8]
    expected = DataFrame({"a": ["apple", "orange"], "b": ["bat", "cow"]}, index=[4, 8])
    
    # 使用 pandas 的 assert_frame_equal 函数比较 result 和 expected 两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
# 测试函数：使用指定列作为数据框的列，其中索引列位于中间位置
def test_usecols_index_col_middle(all_parsers):
    # 从参数中获取解析器对象
    parser = all_parsers
    # 定义包含多行数据的字符串
    data = """a,b,c,d
1,2,3,4
"""
    # 使用指定的列（"b", "c", "d"）作为列，"c" 作为索引列读取 CSV 数据
    result = parser.read_csv(StringIO(data), usecols=["b", "c", "d"], index_col="c")
    # 期望的数据框
    expected = DataFrame({"b": [2], "d": [4]}, index=Index([3], name="c"))
    # 断言结果与期望相等
    tm.assert_frame_equal(result, expected)


# 测试函数：使用指定列作为数据框的列，其中索引列位于末尾位置
def test_usecols_index_col_end(all_parsers):
    # 从参数中获取解析器对象
    parser = all_parsers
    # 定义包含多行数据的字符串
    data = """a,b,c,d
1,2,3,4
"""
    # 使用指定的列（"b", "c", "d"）作为列，"d" 作为索引列读取 CSV 数据
    result = parser.read_csv(StringIO(data), usecols=["b", "c", "d"], index_col="d")
    # 期望的数据框
    expected = DataFrame({"b": [2], "c": [3]}, index=Index([4], name="d"))
    # 断言结果与期望相等
    tm.assert_frame_equal(result, expected)


# 测试函数：使用正则表达式作为分隔符读取 CSV 数据
def test_usecols_regex_sep(all_parsers):
    # 从参数中获取解析器对象
    parser = all_parsers
    # 定义包含多行数据的字符串
    data = "a  b  c\n4  apple  bat  5.7\n8  orange  cow  10"

    # 如果解析器引擎是 "pyarrow"，则抛出异常
    if parser.engine == "pyarrow":
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=r"\s+", usecols=("a", "b"))
        return

    # 使用正则表达式 "\s+" 作为分隔符，读取指定的列 ("a", "b")
    result = parser.read_csv(StringIO(data), sep=r"\s+", usecols=("a", "b"))
    # 期望的数据框
    expected = DataFrame({"a": ["apple", "orange"], "b": ["bat", "cow"]}, index=[4, 8])
    # 断言结果与期望相等
    tm.assert_frame_equal(result, expected)


# 测试函数：使用空白符作为分隔符读取 CSV 数据，指定列包含空白
@skip_pyarrow  # 列 'a' 在 include_columns 中在 CSV 文件中不存在
def test_usecols_with_whitespace(all_parsers):
    # 从参数中获取解析器对象
    parser = all_parsers
    # 定义包含多行数据的字符串
    data = "a  b  c\n4  apple  bat  5.7\n8  orange  cow  10"

    # 使用正则表达式 "\s+" 作为分隔符，读取指定的列 ("a", "b")
    result = parser.read_csv(StringIO(data), sep=r"\s+", usecols=("a", "b"))
    # 期望的数据框
    expected = DataFrame({"a": ["apple", "orange"], "b": ["bat", "cow"]}, index=[4, 8])
    # 断言结果与期望相等
    tm.assert_frame_equal(result, expected)


# 参数化测试函数：使用整数或字符串列头来选择列
@pytest.mark.parametrize(
    "usecols,expected",
    [
        # 按索引选择列
        ([0, 1], DataFrame(data=[[1000, 2000], [4000, 5000]], columns=["2", "0"])),
        # 按名称选择列
        (
            ["0", "1"],
            DataFrame(data=[[2000, 3000], [5000, 6000]], columns=["0", "1"]),
        ),
    ],
)
def test_usecols_with_integer_like_header(all_parsers, usecols, expected, request):
    # 从参数中获取解析器对象
    parser = all_parsers
    # 定义包含多行数据的字符串
    data = """2,0,1
1000,2000,3000
4000,5000,6000"""

    # 如果解析器引擎是 "pyarrow" 并且 usecols[0] 是整数，则抛出异常
    if parser.engine == "pyarrow" and isinstance(usecols[0], int):
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), usecols=usecols)
        return

    # 使用指定的列读取 CSV 数据
    result = parser.read_csv(StringIO(data), usecols=usecols)
    # 断言结果与期望相等
    tm.assert_frame_equal(result, expected)


# 标记为预期失败的测试函数：使用空集合作为 usecols 读取 CSV 数据
@xfail_pyarrow  # 形状不匹配
def test_empty_usecols(all_parsers):
    # 定义包含多行数据的字符串
    data = "a,b,c\n1,2,3\n4,5,6"
    # 期望的空数据框
    expected = DataFrame(columns=Index([]))
    # 从参数中获取解析器对象
    parser = all_parsers

    # 使用空集合作为 usecols 读取 CSV 数据
    result = parser.read_csv(StringIO(data), usecols=set())
    # 断言结果与期望相等
    tm.assert_frame_equal(result, expected)


# 测试函数：使用 NumPy 数组作为 usecols 读取 CSV 数据
def test_np_array_usecols(all_parsers):
    # 从参数中获取解析器对象
    parser = all_parsers
    # 定义包含多行数据的字符串
    data = "a,b,c\n1,2,3"
    # 使用 NumPy 数组作为列名列表
    usecols = np.array(["a", "b"])

    # 期望的数据框
    expected = DataFrame([[1, 2]], columns=usecols)
    # 使用解析器从给定的数据字符串读取 CSV 数据，并仅使用指定列（usecols参数）。
    result = parser.read_csv(StringIO(data), usecols=usecols)
    # 使用测试工具（tm.assert_frame_equal）比较result和expected两个DataFrame是否相等。
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(  # 使用 pytest 的参数化功能来多次运行这个测试函数，以不同的参数组合
    "usecols,expected",  # 参数化的参数：usecols 和 expected
    [  # 参数化的测试数据列表开始
        (
            lambda x: x.upper() in ["AAA", "BBB", "DDD"],  # 第一组 usecols 参数是一个函数，用于选择特定的列名
            DataFrame(  # 第一组 expected 是一个 DataFrame，包含指定列名和对应的数据
                {
                    "AaA": {0: 0.056674972999999997, 1: 2.6132309819999997, 2: 3.5689350380000002},
                    "bBb": {0: 8, 1: 2, 2: 7},
                    "ddd": {0: "a", 1: "b", 2: "a"},
                }
            ),
        ),
        (
            lambda x: False,  # 第二组 usecols 参数是一个总是返回 False 的函数，不选择任何列
            DataFrame(columns=Index([])),  # 第二组 expected 是一个空的 DataFrame
        ),
    ],  # 参数化的测试数据列表结束
)
def test_callable_usecols(all_parsers, usecols, expected):
    # see gh-14154
    data = """AaA,bBb,CCC,ddd  # 测试数据包含列名
0.056674973,8,True,a  # 数据行
2.613230982,2,False,b  # 数据行
3.568935038,7,False,a"""  # 数据行
    parser = all_parsers  # 获取解析器对象

    if parser.engine == "pyarrow":  # 如果解析器引擎是 pyarrow
        msg = "The pyarrow engine does not allow 'usecols' to be a callable"  # 错误消息
        with pytest.raises(ValueError, match=msg):  # 检查是否抛出 ValueError 异常，并匹配错误消息
            parser.read_csv(StringIO(data), usecols=usecols)  # 尝试使用 usecols 参数读取 CSV 数据
        return  # 如果引发异常，测试函数返回

    result = parser.read_csv(StringIO(data), usecols=usecols)  # 使用指定的 usecols 参数读取 CSV 数据
    tm.assert_frame_equal(result, expected)  # 断言读取的结果与期望的结果相等


# ArrowKeyError: Column 'fa' in include_columns does not exist in CSV file
@skip_pyarrow  # 标记为跳过使用 pyarrow 的测试
@pytest.mark.parametrize("usecols", [["a", "c"], lambda x: x in ["a", "c"]])  # 参数化 usecols 参数
def test_incomplete_first_row(all_parsers, usecols):
    # see gh-6710
    data = "1,2\n1,2,3"  # 测试数据：包含不完整的行
    parser = all_parsers  # 获取解析器对象
    names = ["a", "b", "c"]  # 列名
    expected = DataFrame({"a": [1, 1], "c": [np.nan, 3]})  # 期望的 DataFrame 结果

    result = parser.read_csv(StringIO(data), names=names, usecols=usecols)  # 使用指定的 usecols 参数读取 CSV 数据
    tm.assert_frame_equal(result, expected)  # 断言读取的结果与期望的结果相等


@skip_pyarrow  # 标记为跳过使用 pyarrow 的测试
@pytest.mark.parametrize(  # 参数化多个参数
    "data,usecols,kwargs,expected",  # 参数化的参数：data, usecols, kwargs, expected
    [  # 参数化的测试数据列表开始
        (
            "19,29,39\n" * 2 + "10,20,30,40",  # 第一组测试数据：包含不同行数的 CSV 数据
            [0, 1, 2],  # 第一组 usecols 参数：选择指定列的索引
            {"header": None},  # 第一组 kwargs 参数：指定额外的参数，这里是 header=None
            [[19, 29, 39], [19, 29, 39], [10, 20, 30]],  # 第一组 expected 参数：期望的结果，是一个二维列表
        ),
        (
            ("A,B,C\n1,2,3\n3,4,5\n1,2,4,5,1,6\n1,2,3,,,1,\n1,2,3\n5,6,7"),  # 第二组测试数据：包含不同列数的 CSV 数据
            ["A", "B", "C"],  # 第二组 usecols 参数：选择指定列名
            {},  # 第二组 kwargs 参数：空字典
            {  # 第二组 expected 参数：期望的结果，是一个字典，键是列名，值是列的数据列表
                "A": [1, 3, 1, 1, 1, 5],
                "B": [2, 4, 2, 2, 2, 6],
                "C": [3, 5, 4, 3, 3, 7],
            },
        ),
    ],  # 参数化的测试数据列表结束
)
def test_uneven_length_cols(all_parsers, data, usecols, kwargs, expected):
    # see gh-8985
    parser = all_parsers  # 获取解析器对象
    result = parser.read_csv(StringIO(data), usecols=usecols, **kwargs)  # 使用指定的 usecols 和 kwargs 参数读取 CSV 数据
    expected = DataFrame(expected)  # 根据期望的结果创建 DataFrame
    tm.assert_frame_equal(result, expected)  # 断言读取的结果与期望的结果相等


@pytest.mark.parametrize(  # 参数化 usecols 参数
    "usecols,kwargs,expected,msg",  # 参数化的参数：usecols, kwargs, expected, msg
    [
        (
            ["a", "b", "c", "d"],  # 列名列表，指定要读取的列
            {},  # 参数字典，暂无特定用途
            DataFrame({"a": [1, 5], "b": [2, 6], "c": [3, 7], "d": [4, 8]}),  # 数据框对象，包含指定列的数据
            None,  # 错误消息，如果有列名不匹配时显示的格式化字符串
        ),
        (
            ["a", "b", "c", "f"],  # 列名列表，指定要读取的列
            {},  # 参数字典，暂无特定用途
            None,  # 数据框对象，没有数据框对象返回，因为列名 'f' 不存在
            _msg_validate_usecols_names.format(r"\['f'\]")  # 错误消息，格式化字符串，指示列名 'f' 不存在
        ),
        (
            ["a", "b", "f"],  # 列名列表，指定要读取的列
            {},  # 参数字典，暂无特定用途
            None,  # 数据框对象，没有数据框对象返回，因为列名 'f' 不存在
            _msg_validate_usecols_names.format(r"\['f'\]")  # 错误消息，格式化字符串，指示列名 'f' 不存在
        ),
        (
            ["a", "b", "f", "g"],  # 列名列表，指定要读取的列
            {},  # 参数字典，暂无特定用途
            None,  # 数据框对象，没有数据框对象返回，因为列名 'f' 和 'g' 都不存在
            _msg_validate_usecols_names.format(r"\[('f', 'g'|'g', 'f')\]")  # 错误消息，格式化字符串，指示列名 'f' 和 'g' 都不存在
        ),
        # see gh-14671
        (
            None,  # 列名列表为 None，不指定特定列
            {"header": 0, "names": ["A", "B", "C", "D"]},  # 参数字典，指定头部和列名
            DataFrame({"A": [1, 5], "B": [2, 6], "C": [3, 7], "D": [4, 8]}),  # 数据框对象，包含指定列名和数据
            None,  # 错误消息，如果有列名不匹配时显示的格式化字符串
        ),
        (
            ["A", "B", "C", "f"],  # 列名列表，指定要读取的列
            {"header": 0, "names": ["A", "B", "C", "D"]},  # 参数字典，指定头部和列名
            None,  # 数据框对象，没有数据框对象返回，因为列名 'f' 不存在
            _msg_validate_usecols_names.format(r"\['f'\]")  # 错误消息，格式化字符串，指示列名 'f' 不存在
        ),
        (
            ["A", "B", "f"],  # 列名列表，指定要读取的列
            {"names": ["A", "B", "C", "D"]},  # 参数字典，指定列名
            None,  # 数据框对象，没有数据框对象返回，因为列名 'f' 不存在
            _msg_validate_usecols_names.format(r"\['f'\]")  # 错误消息，格式化字符串，指示列名 'f' 不存在
        ),
    ],
def test_raises_on_usecols_names_mismatch(
    all_parsers, usecols, kwargs, expected, msg, request
):
    # 创建一个包含示例数据的 CSV 字符串
    data = "a,b,c,d\n1,2,3,4\n5,6,7,8"
    
    # 更新参数中的 usecols
    kwargs.update(usecols=usecols)
    
    # 获取所使用的解析器对象
    parser = all_parsers

    # 如果使用的是 pyarrow 引擎，并且 usecols 和 expected 都不是 None
    if parser.engine == "pyarrow" and not (
        usecols is not None and expected is not None
    ):
        # 跳过测试，因为箭头引擎在这种情况下会引发异常
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    # 如果 expected 是 None
    if expected is None:
        # 断言在读取 CSV 时会引发 ValueError，并匹配特定的错误信息
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    else:
        # 否则，执行读取 CSV 并进行结果比较
        result = parser.read_csv(StringIO(data), **kwargs)
        tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("usecols", [["A", "C"], [0, 2]])
def test_usecols_subset_names_mismatch_orig_columns(all_parsers, usecols, request):
    # 创建一个包含示例数据的 CSV 字符串
    data = "a,b,c,d\n1,2,3,4\n5,6,7,8"
    
    # 列名列表
    names = ["A", "B", "C", "D"]
    
    # 获取所使用的解析器对象
    parser = all_parsers

    # 如果使用的是 pyarrow 引擎
    if parser.engine == "pyarrow":
        # 如果 usecols 的第一个元素是整数
        if isinstance(usecols[0], int):
            # 断言在读取 CSV 时会引发 ValueError，并匹配特定的错误信息
            with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
                parser.read_csv(StringIO(data), header=0, names=names, usecols=usecols)
            return
        # 否则，跳过测试，因为箭头引擎在这种情况下会引发异常
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    # 否则，执行读取 CSV 并进行结果比较
    result = parser.read_csv(StringIO(data), header=0, names=names, usecols=usecols)
    expected = DataFrame({"A": [1, 5], "C": [3, 7]})
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("names", [None, ["a", "b"]])
def test_usecols_indices_out_of_bounds(all_parsers, names):
    # 解析器对象
    parser = all_parsers
    
    # 包含示例数据的 CSV 字符串
    data = """
a,b
1,2
    """
    
    # 错误类型
    err = ParserError
    msg = "Defining usecols with out-of-bounds"
    
    # 如果使用的是 pyarrow 引擎
    if parser.engine == "pyarrow":
        # 更新错误类型和消息
        err = ValueError
        msg = _msg_pyarrow_requires_names

    # 断言在读取 CSV 时会引发特定错误类型，并匹配特定的错误信息
    with pytest.raises(err, match=msg):
        parser.read_csv(StringIO(data), usecols=[0, 2], names=names, header=0)


def test_usecols_additional_columns(all_parsers):
    # 解析器对象
    parser = all_parsers
    
    # 只包含 "a,b\nx,y,z" 的 CSV 字符串
    usecols = lambda header: header.strip() in ["a", "b", "c"]

    # 如果使用的是 pyarrow 引擎
    if parser.engine == "pyarrow":
        # 错误消息
        msg = "The pyarrow engine does not allow 'usecols' to be a callable"
        
        # 断言在读取 CSV 时会引发 ValueError，并匹配特定的错误信息
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO("a,b\nx,y,z"), index_col=False, usecols=usecols)
        return
    
    # 否则，执行读取 CSV 并进行结果比较
    result = parser.read_csv(StringIO("a,b\nx,y,z"), index_col=False, usecols=usecols)
    expected = DataFrame({"a": ["x"], "b": "y"})
    tm.assert_frame_equal(result, expected)


def test_usecols_additional_columns_integer_columns(all_parsers):
    # 解析器对象
    parser = all_parsers
    
    # 只包含 "a,b\nx,y,z" 的 CSV 字符串
    usecols = lambda header: header.strip() in ["0", "1"]
    # 如果解析器的引擎是 "pyarrow"，则设置错误消息
    if parser.engine == "pyarrow":
        msg = "The pyarrow engine does not allow 'usecols' to be a callable"
        # 使用 pytest 检查是否引发值错误，并匹配指定的错误消息
        with pytest.raises(ValueError, match=msg):
            # 尝试使用指定的 usecols 参数读取 CSV 数据，预期会引发值错误
            parser.read_csv(StringIO("0,1\nx,y,z"), index_col=False, usecols=usecols)
        # 如果发生错误，直接返回，函数结束
        return
    
    # 使用指定的 usecols 参数读取 CSV 数据，生成结果
    result = parser.read_csv(StringIO("0,1\nx,y,z"), index_col=False, usecols=usecols)
    
    # 生成预期的 DataFrame 对象，用于与结果进行比较
    expected = DataFrame({"0": ["x"], "1": "y"})
    
    # 使用 pytest 比较实际结果和预期结果的 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)
# 定义函数 test_usecols_dtype，用于测试给定解析器的功能
def test_usecols_dtype(all_parsers):
    # 从参数 all_parsers 中获取解析器对象
    parser = all_parsers
    # 定义包含 CSV 格式数据的字符串
    data = """
col1,col2,col3
a,1,x
b,2,y
"""
    # 使用解析器读取内存中的 CSV 数据，并指定使用的列和数据类型
    result = parser.read_csv(
        StringIO(data),
        usecols=["col1", "col2"],  # 仅使用 col1 和 col2 列
        dtype={"col1": "string", "col2": "uint8", "col3": "string"},  # 指定各列的数据类型
    )
    # 定义预期的 DataFrame 结果，包含指定的数据和数据类型
    expected = DataFrame(
        {"col1": array(["a", "b"]), "col2": np.array([1, 2], dtype="uint8")}
    )
    # 使用测试工具比较结果 DataFrame 和预期 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
```