# `D:\src\scipysrc\pandas\pandas\tests\io\parser\common\test_iterator.py`

```
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# 导入所需的模块和库
from io import StringIO  # 导入StringIO类，用于在内存中操作字符串IO
import pytest  # 导入pytest测试框架

from pandas import (  # 导入pandas相关函数和类
    DataFrame,  # 导入DataFrame类
    concat,  # 导入concat函数，用于连接数据框
)
import pandas._testing as tm  # 导入pandas测试工具模块

# 设置pytest标记，忽略特定警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


def test_iterator(all_parsers):
    # see gh-6607
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    parser = all_parsers  # 获取测试参数all_parsers中的解析器对象
    kwargs = {"index_col": 0}  # 定义关键字参数字典

    expected = parser.read_csv(StringIO(data), **kwargs)  # 用解析器读取CSV数据并存储到expected中

    if parser.engine == "pyarrow":
        # 如果解析引擎是'pyarrow'，抛出不支持iterator选项的异常
        msg = "The 'iterator' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), iterator=True, **kwargs)
        return

    # 使用解析器以迭代器方式读取CSV数据
    with parser.read_csv(StringIO(data), iterator=True, **kwargs) as reader:
        first_chunk = reader.read(3)  # 读取前3行数据作为第一个块
        tm.assert_frame_equal(first_chunk, expected[:3])  # 断言第一个块的数据与预期的前3行数据相等

        last_chunk = reader.read(5)  # 继续读取5行数据作为最后一个块
    tm.assert_frame_equal(last_chunk, expected[3:])  # 断言最后一个块的数据与预期的剩余数据相等


def test_iterator2(all_parsers):
    parser = all_parsers  # 获取测试参数all_parsers中的解析器对象
    data = """A,B,C
foo,1,2,3
bar,4,5,6
baz,7,8,9
"""

    if parser.engine == "pyarrow":
        # 如果解析引擎是'pyarrow'，抛出不支持iterator选项的异常
        msg = "The 'iterator' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), iterator=True)
        return

    # 使用解析器以迭代器方式读取CSV数据
    with parser.read_csv(StringIO(data), iterator=True) as reader:
        result = list(reader)  # 将读取结果转换为列表

    # 创建预期的DataFrame对象
    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        index=["foo", "bar", "baz"],
        columns=["A", "B", "C"],
    )
    tm.assert_frame_equal(result[0], expected)  # 断言读取结果与预期数据框相等


def test_iterator_stop_on_chunksize(all_parsers):
    # gh-3967: stopping iteration when chunksize is specified
    parser = all_parsers  # 获取测试参数all_parsers中的解析器对象
    data = """A,B,C
foo,1,2,3
bar,4,5,6
baz,7,8,9
"""
    if parser.engine == "pyarrow":
        # 如果解析引擎是'pyarrow'，抛出不支持chunksize选项的异常
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), chunksize=1)
        return

    # 使用解析器以块大小为1的方式读取CSV数据
    with parser.read_csv(StringIO(data), chunksize=1) as reader:
        result = list(reader)  # 将读取结果转换为列表

    assert len(result) == 3  # 断言结果列表长度为3
    # 创建预期的DataFrame对象
    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        index=["foo", "bar", "baz"],
        columns=["A", "B", "C"],
    )
    tm.assert_frame_equal(concat(result), expected)  # 断言连接结果与预期数据框相等


def test_nrows_iterator_without_chunksize(all_parsers):
    # GH 59079
    parser = all_parsers  # 获取测试参数all_parsers中的解析器对象
    data = """A,B,C
foo,1,2,3
bar,4,5,6
baz,7,8,9
"""
    if parser.engine == "pyarrow":
        # 如果解析引擎是'pyarrow'，抛出不支持iterator选项的异常
        msg = "The 'iterator' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), iterator=True, nrows=2)
        return
    # 使用 parser 对象读取 CSV 数据，数据来源为内存中的字符串 data，以迭代器模式读取前两行数据
    with parser.read_csv(StringIO(data), iterator=True, nrows=2) as reader:
        # 获取读取的数据块（前两行数据）
        result = reader.get_chunk()
    
    # 创建预期的 DataFrame 对象，包含数据 [[1, 2, 3], [4, 5, 6]]，行索引为 ["foo", "bar"]，列索引为 ["A", "B", "C"]
    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6]],
        index=["foo", "bar"],
        columns=["A", "B", "C"],
    )
    
    # 使用 pandas.testing 模块的 assert_frame_equal 方法比较 result 和 expected 两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
# 使用 pytest 的 parametrize 装饰器，为 test_iterator_skipfooter_errors 函数参数化多组参数
@pytest.mark.parametrize(
    "kwargs", [{"iterator": True, "chunksize": 1}, {"iterator": True}, {"chunksize": 1}]
)
# 定义测试函数 test_iterator_skipfooter_errors，用于测试 CSV 解析器在特定条件下的行为
def test_iterator_skipfooter_errors(all_parsers, kwargs):
    # 定义错误消息变量，用于断言异常时的匹配
    msg = "'skipfooter' not supported for iteration"
    # 获取当前测试使用的解析器
    parser = all_parsers
    # 准备测试数据字符串
    data = "a\n1\n2"

    # 如果解析器引擎是 "pyarrow"，更新错误消息
    if parser.engine == "pyarrow":
        msg = (
            "The '(chunksize|iterator)' option is not supported with the "
            "'pyarrow' engine"
        )

    # 使用 pytest 的 raises 断言来验证 ValueError 异常，匹配错误消息
    with pytest.raises(ValueError, match=msg):
        # 使用解析器读取 CSV 数据，同时传入预定义的 kwargs 参数
        with parser.read_csv(StringIO(data), skipfooter=1, **kwargs) as _:
            pass


# 定义测试函数 test_iteration_open_handle，用于测试在打开文件句柄时的迭代行为
def test_iteration_open_handle(all_parsers):
    # 获取当前测试使用的解析器
    parser = all_parsers
    # 定义解析 CSV 文件时的参数
    kwargs = {"header": None}

    # 使用 tm.ensure_clean 上下文管理器，确保测试时使用的文件路径是干净的
    with tm.ensure_clean() as path:
        # 使用 open 打开文件句柄，写入测试数据字符串
        with open(path, "w", encoding="utf-8") as f:
            f.write("AAA\nBBB\nCCC\nDDD\nEEE\nFFF\nGGG")

        # 再次使用 open 打开文件句柄，迭代文件的每一行
        with open(path, encoding="utf-8") as f:
            for line in f:
                # 如果当前行包含 "CCC"，则终止迭代
                if "CCC" in line:
                    break

            # 使用解析器读取剩余的文件内容，并传入预定义的 kwargs 参数
            result = parser.read_csv(f, **kwargs)
            # 定义预期的 DataFrame 结果
            expected = DataFrame({0: ["DDD", "EEE", "FFF", "GGG"]})
            # 使用 tm.assert_frame_equal 来断言结果是否符合预期
            tm.assert_frame_equal(result, expected)
```