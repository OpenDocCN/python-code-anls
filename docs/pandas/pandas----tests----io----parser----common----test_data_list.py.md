# `D:\src\scipysrc\pandas\pandas\tests\io\parser\common\test_data_list.py`

```
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# 导入 csv 模块，用于处理 CSV 文件
import csv
# 从 io 模块导入 StringIO 类，用于将字符串数据模拟成文件对象
from io import StringIO

# 导入 pytest 模块，用于测试
import pytest

# 从 pandas 库中导入 DataFrame 类
from pandas import DataFrame
# 导入 pandas 内部测试工具模块
import pandas._testing as tm

# 从 pandas.io.parsers 模块中导入 TextParser 类
from pandas.io.parsers import TextParser

# 忽略特定警告，例如将 BlockManager 传递给 DataFrame 的警告
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 定义一个 pytest 装饰器，用于标记某些测试在 pyarrow 引擎下可能会失败
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")


@xfail_pyarrow
# 测试函数：测试从列表数据读取的情况
def test_read_data_list(all_parsers):
    # 获取所有解析器对象
    parser = all_parsers
    # 设置关键字参数
    kwargs = {"index_col": 0}
    # 模拟 CSV 数据
    data = "A,B,C\nfoo,1,2,3\nbar,4,5,6"

    # 构建数据列表
    data_list = [["A", "B", "C"], ["foo", "1", "2", "3"], ["bar", "4", "5", "6"]]
    # 使用 parser 对象读取 CSV 数据
    expected = parser.read_csv(StringIO(data), **kwargs)

    # 使用 TextParser 对象处理数据列表，设置每次处理块大小为 2
    with TextParser(data_list, chunksize=2, **kwargs) as parser:
        result = parser.read()

    # 断言处理结果与预期是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数：测试从列表数据读取的情况
def test_reader_list(all_parsers):
    # 模拟 CSV 数据
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    # 获取所有解析器对象
    parser = all_parsers
    # 设置关键字参数
    kwargs = {"index_col": 0}

    # 将 CSV 数据转换为行列表
    lines = list(csv.reader(StringIO(data)))
    # 使用 TextParser 对象处理行列表，设置每次处理块大小为 2
    with TextParser(lines, chunksize=2, **kwargs) as reader:
        chunks = list(reader)

    # 使用 parser 对象读取 CSV 数据
    expected = parser.read_csv(StringIO(data), **kwargs)

    # 断言处理结果的每个块与预期是否相等
    tm.assert_frame_equal(chunks[0], expected[:2])
    tm.assert_frame_equal(chunks[1], expected[2:4])
    tm.assert_frame_equal(chunks[2], expected[4:])


# 测试函数：测试从列表数据读取并跳过指定行的情况
def test_reader_list_skiprows(all_parsers):
    # 模拟 CSV 数据
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    # 获取所有解析器对象
    parser = all_parsers
    # 设置关键字参数
    kwargs = {"index_col": 0}

    # 将 CSV 数据转换为行列表
    lines = list(csv.reader(StringIO(data)))
    # 使用 TextParser 对象处理行列表，设置每次处理块大小为 2，并跳过第一行
    with TextParser(lines, chunksize=2, skiprows=[1], **kwargs) as reader:
        chunks = list(reader)

    # 使用 parser 对象读取 CSV 数据
    expected = parser.read_csv(StringIO(data), **kwargs)

    # 断言处理结果的第一个块与预期是否相等
    tm.assert_frame_equal(chunks[0], expected[1:3])


# 测试函数：测试从简单列表数据读取并解析的情况
def test_read_csv_parse_simple_list(all_parsers):
    # 获取所有解析器对象
    parser = all_parsers
    # 模拟 CSV 数据
    data = """foo
bar baz
qux foo
foo
bar"""

    # 使用 parser 对象读取 CSV 数据，不设置列名
    result = parser.read_csv(StringIO(data), header=None)
    # 期望的 DataFrame 结果
    expected = DataFrame(["foo", "bar baz", "qux foo", "foo", "bar"])
    # 断言处理结果与预期是否相等
    tm.assert_frame_equal(result, expected)
```