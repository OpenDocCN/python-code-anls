# `D:\src\scipysrc\pandas\pandas\tests\io\parser\common\test_chunksize.py`

```
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# 导入所需的模块和库
from io import StringIO  # 导入 StringIO 类用于在内存中创建文件对象

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架

from pandas._libs import parsers as libparsers  # 导入 pandas 内部的 parsers 模块
from pandas.errors import DtypeWarning  # 导入 pandas 的 DtypeWarning 异常类

from pandas import (
    DataFrame,  # 导入 pandas 的 DataFrame 类
    concat,  # 导入 pandas 的 concat 函数
)
import pandas._testing as tm  # 导入 pandas 内部的测试模块

# 设置 pytest 的标记，忽略特定警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 使用 pytest 的参数化装饰器，测试函数接受 index_col 参数
@pytest.mark.parametrize("index_col", [0, "index"])
def test_read_chunksize_with_index(all_parsers, index_col):
    parser = all_parsers  # 获取测试数据集的解析器
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""

    # 预期的 DataFrame 结构
    expected = DataFrame(
        [
            ["foo", 2, 3, 4, 5],
            ["bar", 7, 8, 9, 10],
            ["baz", 12, 13, 14, 15],
            ["qux", 12, 13, 14, 15],
            ["foo2", 12, 13, 14, 15],
            ["bar2", 12, 13, 14, 15],
        ],
        columns=["index", "A", "B", "C", "D"],
    )
    expected = expected.set_index("index")  # 将 "index" 列设置为 DataFrame 的索引

    # 如果解析器的引擎是 "pyarrow"，则验证是否抛出特定异常
    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with parser.read_csv(StringIO(data), index_col=0, chunksize=2) as reader:
                list(reader)
        return  # 结束当前测试函数

    # 使用解析器读取 CSV 数据，设置 index_col 和 chunksize 参数
    with parser.read_csv(StringIO(data), index_col=0, chunksize=2) as reader:
        chunks = list(reader)  # 读取数据并存储为 chunk 列表
    tm.assert_frame_equal(chunks[0], expected[:2])  # 验证第一个 chunk 的数据
    tm.assert_frame_equal(chunks[1], expected[2:4])  # 验证第二个 chunk 的数据
    tm.assert_frame_equal(chunks[2], expected[4:])  # 验证第三个 chunk 的数据


# 使用 pytest 的参数化装饰器，测试函数接受 chunksize 参数为非法值的情况
@pytest.mark.parametrize("chunksize", [1.3, "foo", 0])
def test_read_chunksize_bad(all_parsers, chunksize):
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers  # 获取测试数据集的解析器

    msg = r"'chunksize' must be an integer >=1"
    # 如果解析器的引擎是 "pyarrow"，则使用特定的异常消息
    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"

    # 验证是否抛出特定异常消息
    with pytest.raises(ValueError, match=msg):
        with parser.read_csv(StringIO(data), chunksize=chunksize) as _:
            pass  # 不执行任何操作，只是为了验证是否抛出异常


# 使用 pytest 的参数化装饰器，测试函数接受 chunksize 和 nrows 参数的情况
@pytest.mark.parametrize("chunksize", [2, 8])
def test_read_chunksize_and_nrows(all_parsers, chunksize):
    # see gh-15755
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers  # 获取测试数据集的解析器
    kwargs = {"index_col": 0, "nrows": 5}

    # 如果解析器的引擎是 "pyarrow"，则验证是否抛出特定异常
    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
        return  # 结束当前测试函数

    expected = parser.read_csv(StringIO(data), **kwargs)  # 读取并获取预期的 DataFrame
    with parser.read_csv(StringIO(data), chunksize=chunksize, **kwargs) as reader:
        tm.assert_frame_equal(concat(reader), expected)  # 验证合并所有 chunk 后的结果与预期是否一致
# 测试函数，用于验证在改变 chunksize 和 nrows 大小时的行为
def test_read_chunksize_and_nrows_changing_size(all_parsers):
    # 测试数据，包括索引和多列数据
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    # 获取测试用的解析器
    parser = all_parsers
    # 设置读取选项，包括索引列和读取行数
    kwargs = {"index_col": 0, "nrows": 5}

    # 如果使用的是 pyarrow 引擎，则抛出错误信息
    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
        return

    # 从数据中读取预期的数据帧
    expected = parser.read_csv(StringIO(data), **kwargs)
    
    # 使用不同的 chunksize 从数据中读取内容，并使用测试工具验证结果
    with parser.read_csv(StringIO(data), chunksize=8, **kwargs) as reader:
        tm.assert_frame_equal(reader.get_chunk(size=2), expected.iloc[:2])
        tm.assert_frame_equal(reader.get_chunk(size=4), expected.iloc[2:5])
        
        # 预期抛出 StopIteration 错误
        with pytest.raises(StopIteration, match=""):
            reader.get_chunk(size=3)


# 测试函数，验证在使用 get_chunk 方法时传递 chunksize 的行为
def test_get_chunk_passed_chunksize(all_parsers):
    # 获取测试用的解析器
    parser = all_parsers
    # 测试数据，包括多行和多列数据
    data = """A,B,C
1,2,3
4,5,6
7,8,9
1,2,3"""

    # 如果使用的是 pyarrow 引擎，则抛出错误信息
    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with parser.read_csv(StringIO(data), chunksize=2) as reader:
                reader.get_chunk()
        return

    # 使用指定的 chunksize 从数据中读取内容，并验证结果
    with parser.read_csv(StringIO(data), chunksize=2) as reader:
        result = reader.get_chunk()

    # 期望的结果数据帧
    expected = DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "B", "C"])
    tm.assert_frame_equal(result, expected)


# 测试函数，验证在不同参数设置下的 chunksize 行为的兼容性
@pytest.mark.parametrize("kwargs", [{}, {"index_col": 0}])
def test_read_chunksize_compat(all_parsers, kwargs):
    # 详见 gh-12185
    # 测试数据，包括索引和多列数据
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    # 获取测试用的解析器
    parser = all_parsers
    # 从数据中读取结果数据帧
    result = parser.read_csv(StringIO(data), **kwargs)

    # 如果使用的是 pyarrow 引擎，则抛出错误信息
    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with parser.read_csv(StringIO(data), chunksize=2, **kwargs) as reader:
                concat(reader)
        return

    # 使用指定的 chunksize 从数据中读取内容，并验证结果
    with parser.read_csv(StringIO(data), chunksize=2, **kwargs) as reader:
        via_reader = concat(reader)
    tm.assert_frame_equal(via_reader, result)


# 测试函数，验证在使用不同 chunksize 和列名时的行为
def test_read_chunksize_jagged_names(all_parsers):
    # 详见 gh-23509
    # 获取测试用的解析器
    parser = all_parsers
    # 测试数据，包括大量行和列名
    data = "\n".join(["0"] * 7 + [",".join(["0"] * 10)])

    # 期望的结果数据帧
    expected = DataFrame([[0] + [np.nan] * 9] * 7 + [[0] * 10])

    # 如果使用的是 pyarrow 引擎，则抛出错误信息
    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with parser.read_csv(
                StringIO(data), names=range(10), chunksize=4
            ) as reader:
                concat(reader)
        return

    # 使用指定的 chunksize 和列名从数据中读取内容，并验证结果
    with parser.read_csv(StringIO(data), names=range(10), chunksize=4) as reader:
        result = concat(reader)
    # 使用测试框架中的 assert_frame_equal 函数比较 result 和 expected 两个数据框架是否相等
    tm.assert_frame_equal(result, expected)
def test_chunk_begins_with_newline_whitespace(all_parsers):
    # 见 gh-10022，此测试函数用于检查处理器是否正确处理以换行符和空白开头的数据行
    parser = all_parsers
    data = "\n hello\nworld\n"

    # 使用给定的解析器解析 CSV 数据，并期望结果为一个特定的 DataFrame
    result = parser.read_csv(StringIO(data), header=None)
    expected = DataFrame([" hello", "world"])
    tm.assert_frame_equal(result, expected)


@pytest.mark.slow
def test_chunks_have_consistent_numerical_type(all_parsers, monkeypatch):
    # 主要是 C 解析器的问题，此测试函数用于检查处理器在处理具有一致数值类型的数据块时的行为
    heuristic = 2**3
    parser = all_parsers
    integers = [str(i) for i in range(heuristic - 1)]
    data = "a\n" + "\n".join(integers + ["1.0", "2.0"] + integers)

    # 设置默认缓冲启发式为指定值，确保强制类型转换可以正常工作且无警告
    with monkeypatch.context() as m:
        m.setattr(libparsers, "DEFAULT_BUFFER_HEURISTIC", heuristic)
        result = parser.read_csv(StringIO(data))

    # 断言处理结果中第一列的数据类型为 np.float64
    assert type(result.a[0]) is np.float64
    # 断言处理结果中第一列的整体数据类型为 float


def test_warn_if_chunks_have_mismatched_type(all_parsers):
    warning_type = None
    parser = all_parsers
    size = 10000

    # 见 gh-3866：如果数据块中的数据类型不一致且无法通过数值类型强制转换，则发出警告
    if parser.engine == "c" and parser.low_memory:
        warning_type = DtypeWarning
        # 使用较大的数据量以触发警告路径
        size = 499999

    integers = [str(i) for i in range(size)]
    data = "a\n" + "\n".join(integers + ["a", "b"] + integers)

    buf = StringIO(data)

    if parser.engine == "pyarrow":
        # 如果使用 pyarrow 引擎，不支持 '(nrows|chunksize)' 选项
        df = parser.read_csv(
            buf,
        )
    else:
        # 使用特定的警告类型和消息来读取 CSV 数据，并检查是否正确处理警告
        df = parser.read_csv_check_warnings(
            warning_type,
            r"Columns \(0: a\) have mixed types. "
            "Specify dtype option on import or set low_memory=False.",
            buf,
        )

    # 断言处理结果中第一列的数据类型为 object


@pytest.mark.parametrize("iterator", [True, False])
def test_empty_with_nrows_chunksize(all_parsers, iterator):
    # 见 gh-9535，此测试函数用于检查处理器在使用 nrows 或 chunksize 选项时的行为
    parser = all_parsers
    expected = DataFrame(columns=["foo", "bar"])

    nrows = 10
    data = StringIO("foo,bar\n")

    if parser.engine == "pyarrow":
        # 如果使用 pyarrow 引擎，不支持 '(nrows|chunksize)' 选项，并期望抛出 ValueError 异常
        msg = (
            "The '(nrows|chunksize)' option is not supported with the 'pyarrow' engine"
        )
        with pytest.raises(ValueError, match=msg):
            if iterator:
                with parser.read_csv(data, chunksize=nrows) as reader:
                    next(iter(reader))
            else:
                parser.read_csv(data, nrows=nrows)
        return

    if iterator:
        # 使用迭代器方式读取 CSV 数据，并检查结果是否符合预期
        with parser.read_csv(data, chunksize=nrows) as reader:
            result = next(iter(reader))
    else:
        # 直接读取指定行数的 CSV 数据，并检查结果是否符合预期
        result = parser.read_csv(data, nrows=nrows)

    # 断言处理结果是否与预期的 DataFrame 相等


def test_read_csv_memory_growth_chunksize(all_parsers):
    # 见 gh-24805，此测试函数用于确保处理器在迭代处理所有数据块时不会崩溃
    parser = all_parsers
    #
    # 让我们确保在迭代处理所有数据块时不会崩溃
    #
    #
    # 使用 tm.ensure_clean() 上下文管理器确保操作后的路径被清理
    with tm.ensure_clean() as path:
        # 以写入模式打开文件，使用 UTF-8 编码
        with open(path, "w", encoding="utf-8") as f:
            # 循环写入数字和换行符到文件中，总共写入1000行
            for i in range(1000):
                f.write(str(i) + "\n")
    
        # 如果解析器的引擎是 "pyarrow"
        if parser.engine == "pyarrow":
            # 设置错误消息
            msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
            # 使用 pytest 的 raises 断言来捕获 ValueError 异常，匹配特定的错误消息
            with pytest.raises(ValueError, match=msg):
                # 使用 parser.read_csv() 方法读取 CSV 文件，并指定 chunksize 为 20
                with parser.read_csv(path, chunksize=20) as result:
                    # 迭代处理结果，不做任何操作
                    for _ in result:
                        pass
            # 如果使用了 'pyarrow' 引擎，函数直接返回
            return
    
        # 否则，使用 parser.read_csv() 方法读取 CSV 文件，指定 chunksize 为 20
        with parser.read_csv(path, chunksize=20) as result:
            # 迭代处理结果，不做任何操作
            for _ in result:
                pass
# 测试函数，用于验证使用不同解析器解析 CSV 文件时，设置 'chunksize' 和 'usecols' 选项的行为
def test_chunksize_with_usecols_second_block_shorter(all_parsers):
    # 获取解析器对象
    parser = all_parsers
    # 定义包含不同行数的 CSV 数据
    data = """1,2,3,4
5,6,7,8
9,10,11
"""

    # 如果解析器使用 'pyarrow' 引擎，抛出 ValueError 异常，并匹配指定的错误消息
    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                names=["a", "b"],
                chunksize=2,
                usecols=[0, 1],
                header=None,
            )
        return

    # 使用给定的 'chunksize' 和 'usecols' 选项读取 CSV 数据，并返回结果的迭代器
    result_chunks = parser.read_csv(
        StringIO(data),
        names=["a", "b"],
        chunksize=2,
        usecols=[0, 1],
        header=None,
    )

    # 期望得到的 DataFrame 片段列表
    expected_frames = [
        DataFrame({"a": [1, 5], "b": [2, 6]}),
        DataFrame({"a": [9], "b": [10]}, index=[2]),
    ]

    # 遍历结果迭代器，验证每个片段是否与期望的 DataFrame 片段相等
    for i, result in enumerate(result_chunks):
        tm.assert_frame_equal(result, expected_frames[i])


# 测试函数，用于验证在解析 CSV 文件时，设置 'chunksize' 选项的行为
def test_chunksize_second_block_shorter(all_parsers):
    # 获取解析器对象
    parser = all_parsers
    # 定义包含不同行数的 CSV 数据
    data = """a,b,c,d
1,2,3,4
5,6,7,8
9,10,11
"""

    # 如果解析器使用 'pyarrow' 引擎，抛出 ValueError 异常，并匹配指定的错误消息
    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), chunksize=2)
        return

    # 使用给定的 'chunksize' 选项读取 CSV 数据，并返回结果的迭代器
    result_chunks = parser.read_csv(StringIO(data), chunksize=2)

    # 期望得到的 DataFrame 片段列表
    expected_frames = [
        DataFrame({"a": [1, 5], "b": [2, 6], "c": [3, 7], "d": [4, 8]}),
        DataFrame({"a": [9], "b": [10], "c": [11], "d": [np.nan]}, index=[2]),
    ]

    # 遍历结果迭代器，验证每个片段是否与期望的 DataFrame 片段相等
    for i, result in enumerate(result_chunks):
        tm.assert_frame_equal(result, expected_frames[i])
```