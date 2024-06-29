# `D:\src\scipysrc\pandas\pandas\tests\io\parser\common\test_file_buffer_url.py`

```
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# 导入所需模块
from io import (
    BytesIO,
    StringIO,
)
import os  # 导入操作系统接口模块
import platform  # 导入平台信息模块
from urllib.error import URLError  # 导入处理 URL 错误的模块
import uuid  # 导入生成唯一标识符的模块

import numpy as np  # 导入数值计算模块 NumPy
import pytest  # 导入单元测试框架 pytest

from pandas.compat import WASM  # 导入与平台兼容相关的 WASM 模块
from pandas.errors import (  # 导入 pandas 错误处理模块
    EmptyDataError,
    ParserError,
)
import pandas.util._test_decorators as td  # 导入 pandas 测试装饰器

from pandas import (  # 导入 pandas 主要模块
    DataFrame,
    Index,
)
import pandas._testing as tm  # 导入 pandas 测试模块

# 设置 pytest 的标记，忽略特定警告
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 标记用例，标记为网络相关测试
@pytest.mark.network
@pytest.mark.single_cpu
def test_url(all_parsers, csv_dir_path, httpserver):
    parser = all_parsers
    kwargs = {"sep": "\t"}

    local_path = os.path.join(csv_dir_path, "salaries.csv")  # 构建本地文件路径
    with open(local_path, encoding="utf-8") as f:
        httpserver.serve_content(content=f.read())  # 使用 HTTP 服务器提供文件内容

    url_result = parser.read_csv(httpserver.url, **kwargs)  # 从 URL 读取 CSV 数据
    local_result = parser.read_csv(local_path, **kwargs)  # 从本地文件读取 CSV 数据
    tm.assert_frame_equal(url_result, local_result)  # 断言 URL 和本地文件读取的数据一致


# 标记用例，标记为慢速测试
@pytest.mark.slow
def test_local_file(all_parsers, csv_dir_path):
    parser = all_parsers
    kwargs = {"sep": "\t"}

    local_path = os.path.join(csv_dir_path, "salaries.csv")  # 构建本地文件路径
    local_result = parser.read_csv(local_path, **kwargs)  # 从本地文件读取 CSV 数据
    url = "file://localhost/" + local_path  # 构建本地文件的 URL 地址

    try:
        url_result = parser.read_csv(url, **kwargs)  # 从本地文件 URL 读取 CSV 数据
        tm.assert_frame_equal(url_result, local_result)  # 断言 URL 和本地文件读取的数据一致
    except URLError:
        # 在某些系统上失败
        pytest.skip("Failing on: " + " ".join(platform.uname()))


@xfail_pyarrow  # 标记为预期失败的测试用例，原因是断言 DataFrame 的索引不同
def test_path_path_lib(all_parsers):
    parser = all_parsers
    df = DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),  # 创建一个 DataFrame 对象
        columns=Index(list("ABCD"), dtype=object),  # 指定列索引
        index=Index([f"i-{i}" for i in range(30)], dtype=object),  # 指定行索引
    )
    result = tm.round_trip_pathlib(df.to_csv, lambda p: parser.read_csv(p, index_col=0))  # 执行路径处理并读取 CSV 数据
    tm.assert_frame_equal(df, result)  # 断言原始 DataFrame 与读取后的 DataFrame 一致


# 标记用例，条件为 WASM 不为真时执行，原因是 WASM 平台上的文件系统访问受限
@pytest.mark.skipif(WASM, reason="limited file system access on WASM")
def test_nonexistent_path(all_parsers):
    # gh-2428: pls no segfault
    # gh-14086: raise more helpful FileNotFoundError
    # GH#29233 "File foo" instead of "File b'foo'"
    parser = all_parsers
    path = f"{uuid.uuid4()}.csv"  # 生成随机的 CSV 文件名

    msg = r"\[Errno 2\]"
    with pytest.raises(FileNotFoundError, match=msg) as e:
        parser.read_csv(path)  # 尝试读取不存在的 CSV 文件
    assert path == e.value.filename  # 断言异常信息中的文件名与预期一致


# 标记用例，条件为 WASM 不为真时执行，原因是 WASM 平台上的文件系统访问受限
@pytest.mark.skipif(WASM, reason="limited file system access on WASM")
@td.skip_if_windows  # 根据平台条件跳过 Windows 平台的测试
def test_no_permission(all_parsers):
    # GH 23784
    parser = all_parsers

    msg = r"\[Errno 13\]"
    # 使用 tm.ensure_clean() 确保在代码块执行完毕后清理 path
    with tm.ensure_clean() as path:
        # 改变 path 文件的权限为 0，使其不可读
        os.chmod(path, 0)  # make file unreadable

        # 验证当前进程无法打开该文件（非以 sudo 运行）
        try:
            # 尝试以 utf-8 编码方式打开 path 文件
            with open(path, encoding="utf-8"):
                pass
            # 如果能够打开文件，表示当前以 sudo 权限运行，跳过测试
            pytest.skip("Running as sudo.")
        except PermissionError:
            pass

        # 使用 pytest 检查是否会抛出 PermissionError 异常，并匹配特定的错误信息 msg
        with pytest.raises(PermissionError, match=msg) as e:
            # 尝试解析读取 path 文件内容作为 CSV
            parser.read_csv(path)
        # 断言 path 和捕获到的异常对象 e 的 filename 属性相等，以确认异常的文件名匹配
        assert path == e.value.filename
@pytest.mark.parametrize(
    "data,kwargs,expected,msg",
    [
        # 定义测试参数和期望结果，以及相关消息
        # gh-10728: WHITESPACE_LINE
        (
            "a,b,c\n4,5,6\n ",
            {},  # 空字典作为参数传入
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),  # 期望的DataFrame对象
            None,  # 没有特定的消息
        ),
        # gh-10548: EAT_LINE_COMMENT
        (
            "a,b,c\n4,5,6\n#comment",
            {"comment": "#"},  # 传入了带有注释字符的参数字典
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # EAT_CRNL_NOP
        (
            "a,b,c\n4,5,6\n\r",
            {},  # 空字典作为参数传入
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # EAT_COMMENT
        (
            "a,b,c\n4,5,6#comment",
            {"comment": "#"},  # 传入了带有注释字符的参数字典
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # SKIP_LINE
        (
            "a,b,c\n4,5,6\nskipme",
            {"skiprows": [2]},  # 指定要跳过的行数列表作为参数
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # EAT_LINE_COMMENT
        (
            "a,b,c\n4,5,6\n#comment",
            {"comment": "#", "skip_blank_lines": False},  # 参数字典中包含跳过空白行的选项
            DataFrame([[4, 5, 6]], columns=["a", "b", "c"]),
            None,
        ),
        # IN_FIELD
        (
            "a,b,c\n4,5,6\n ",
            {"skip_blank_lines": False},  # 参数字典中包含跳过空白行的选项
            DataFrame([["4", 5, 6], [" ", None, None]], columns=["a", "b", "c"]),
            None,
        ),
        # EAT_CRNL
        (
            "a,b,c\n4,5,6\n\r",
            {"skip_blank_lines": False},  # 参数字典中包含跳过空白行的选项
            DataFrame([[4, 5, 6], [None, None, None]], columns=["a", "b", "c"]),
            None,
        ),
        # ESCAPED_CHAR
        (
            "a,b,c\n4,5,6\n\\",
            {"escapechar": "\\"},  # 参数字典中包含转义字符的选项
            None,
            "(EOF following escape character)|(unexpected end of data)",  # 匹配异常消息
        ),
        # ESCAPE_IN_QUOTED_FIELD
        (
            'a,b,c\n4,5,6\n"\\',
            {"escapechar": "\\"},  # 参数字典中包含转义字符的选项
            None,
            "(EOF inside string starting at row 2)|(unexpected end of data)",  # 匹配异常消息
        ),
        # IN_QUOTED_FIELD
        (
            'a,b,c\n4,5,6\n"',
            {"escapechar": "\\"},  # 参数字典中包含转义字符的选项
            None,
            "(EOF inside string starting at row 2)|(unexpected end of data)",  # 匹配异常消息
        ),
    ],
    ids=[
        "whitespace-line",
        "eat-line-comment",
        "eat-crnl-nop",
        "eat-comment",
        "skip-line",
        "eat-line-comment",
        "in-field",
        "eat-crnl",
        "escaped-char",
        "escape-in-quoted-field",
        "in-quoted-field",
    ],
)
def test_eof_states(all_parsers, data, kwargs, expected, msg, request):
    # 获取测试使用的解析器对象
    parser = all_parsers

    # 如果解析器引擎为 "pyarrow" 并且参数中包含 "comment" 选项，则抛出 ValueError 异常
    if parser.engine == "pyarrow" and "comment" in kwargs:
        msg = "The 'comment' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)  # 调用解析器的 read_csv 方法，并传入参数
        return  # 结束当前测试函数
    # 检查解析器引擎是否为 "pyarrow"，且数据中不包含回车符 "\r"
    if parser.engine == "pyarrow" and "\r" not in data:
        # 如果条件满足，说明会导致以下异常，因此跳过当前测试
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    # 如果期望结果为 None
    if expected is None:
        # 使用 pytest 检查是否会引发 ParserError 异常，并验证错误消息匹配 msg
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data), **kwargs)
    else:
        # 否则，解析 CSV 数据并返回结果
        result = parser.read_csv(StringIO(data), **kwargs)
        # 使用 tm.assert_frame_equal 检查返回结果与期望结果的数据框是否相等
        tm.assert_frame_equal(result, expected)
# see gh-13398
# 定义测试函数，用于测试临时文件中的数据解析功能
def test_temporary_file(all_parsers, temp_file):
    # 将传入的解析器对象赋值给变量 parser
    parser = all_parsers
    # 定义测试数据
    data = "0 0"

    # 使用临时文件对象进行文件操作，以写入模式打开文件，并指定 UTF-8 编码
    with temp_file.open(mode="w+", encoding="utf-8") as new_file:
        # 向文件中写入测试数据
        new_file.write(data)
        # 刷新文件缓冲区
        new_file.flush()
        # 将文件指针定位到文件开头
        new_file.seek(0)

        # 如果使用的是 'pyarrow' 引擎，则抛出 ValueError 异常，匹配错误信息
        if parser.engine == "pyarrow":
            msg = "the 'pyarrow' engine does not support regex separators"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(new_file, sep=r"\s+", header=None)
            return

        # 使用指定分隔符和无标题行的选项读取 CSV 数据，并将结果赋值给 result
        result = parser.read_csv(new_file, sep=r"\s+", header=None)

        # 期望的数据框
        expected = DataFrame([[0, 0]])
        # 断言读取的数据框与期望的数据框相等
        tm.assert_frame_equal(result, expected)


# see gh-5500
# 定义测试函数，测试包含特殊结束符的数据解析
def test_internal_eof_byte(all_parsers):
    # 将传入的解析器对象赋值给变量 parser
    parser = all_parsers
    # 定义测试数据，包含特殊结束符
    data = "a,b\n1\x1a,2"

    # 期望的数据框
    expected = DataFrame([["1\x1a", 2]], columns=["a", "b"])
    # 使用 StringIO 对象创建内存中的文件并进行 CSV 数据解析
    result = parser.read_csv(StringIO(data))
    # 断言读取的数据框与期望的数据框相等
    tm.assert_frame_equal(result, expected)


# see gh-16559
# 定义测试函数，测试从文件中读取包含特殊结束符的数据
def test_internal_eof_byte_to_file(all_parsers):
    # 将传入的解析器对象赋值给变量 parser
    parser = all_parsers
    # 定义二进制格式的测试数据，包含特殊结束符
    data = b'c1,c2\r\n"test \x1a    test", test\r\n'
    # 期望的数据框
    expected = DataFrame([["test \x1a    test", " test"]], columns=["c1", "c2"])
    # 生成随机的临时文件路径
    path = f"__{uuid.uuid4()}__.csv"

    # 使用 ensure_clean 函数确保文件不存在的情况下执行文件操作
    with tm.ensure_clean(path) as path:
        # 以二进制写入模式打开文件
        with open(path, "wb") as f:
            f.write(data)

        # 使用解析器对象读取 CSV 文件
        result = parser.read_csv(path)
        # 断言读取的数据框与期望的数据框相等
        tm.assert_frame_equal(result, expected)


# gh-14418
#
# Don't close user provided file handles.
# 定义测试函数，测试使用 StringIO 对象进行文件操作时不关闭文件句柄
def test_file_handle_string_io(all_parsers):
    # 将传入的解析器对象赋值给变量 parser
    parser = all_parsers
    # 定义测试数据
    data = "a,b\n1,2"

    # 使用 StringIO 对象创建内存中的文件，并使用解析器对象读取 CSV 数据
    fh = StringIO(data)
    parser.read_csv(fh)
    # 断言文件句柄未关闭
    assert not fh.closed


# gh-14418
#
# Don't close user provided file handles.
# 定义测试函数，测试使用 open 函数打开文件时不关闭文件句柄
def test_file_handles_with_open(all_parsers, csv1):
    # 将传入的解析器对象赋值给变量 parser
    parser = all_parsers

    # 遍历不同的文件打开模式
    for mode in ["r", "rb"]:
        # 使用 open 函数打开指定路径的文件
        with open(csv1, mode, encoding="utf-8" if mode == "r" else None) as f:
            # 使用解析器对象读取 CSV 数据
            parser.read_csv(f)
            # 断言文件句柄未关闭
            assert not f.closed


# see gh-15337
# 定义测试函数，测试传入无效的文件缓冲类时是否抛出 ValueError 异常
def test_invalid_file_buffer_class(all_parsers):
    # 定义一个无效的缓冲类
    class InvalidBuffer:
        pass

    # 将传入的解析器对象赋值给变量 parser
    parser = all_parsers
    # 定义异常消息字符串
    msg = "Invalid file path or buffer object type"

    # 使用 pytest 框架验证是否抛出 ValueError 异常，并匹配异常消息
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(InvalidBuffer())


# see gh-15337
# 定义测试函数，测试传入无效的模拟对象时是否抛出 ValueError 异常
def test_invalid_file_buffer_mock(all_parsers):
    # 将传入的解析器对象赋值给变量 parser
    parser = all_parsers
    # 定义异常消息字符串
    msg = "Invalid file path or buffer object type"

    # 定义一个简单的模拟类
    class Foo:
        pass

    # 使用 pytest 框架验证是否抛出 ValueError 异常，并匹配异常消息
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(Foo())


# gh-16135: we want to ensure that "tell" and "seek"
# aren't actually being used when we call `read_csv`
#
# Thus, while the object may look "invalid" (these
# methods are attributes of the `StringIO` class),
# it is still a valid file-object for our purposes.
# 定义测试函数，测试读取 CSV 数据时是否真正使用了 "tell" 和 "seek" 方法
def test_valid_file_buffer_seems_invalid(all_parsers):
    # 无需进行文件操作，因为测试的目的是验证方法的调用
    pass
    # 定义一个自定义的类 NoSeekTellBuffer，继承自 StringIO
    class NoSeekTellBuffer(StringIO):
        # 重写 tell 方法，抛出属性错误异常，禁止调用 tell 方法
        def tell(self):
            raise AttributeError("No tell method")

        # 重写 seek 方法，抛出属性错误异常，禁止调用 seek 方法
        def seek(self, pos, whence=0):
            raise AttributeError("No seek method")

    # 定义字符串数据变量 data，包含换行符和数字
    data = "a\n1"
    # 定义变量 parser，赋值为 all_parsers，用于解析 CSV 数据
    parser = all_parsers
    # 创建预期的 DataFrame 对象 expected，包含列名 "a" 和值 [1]
    expected = DataFrame({"a": [1]})

    # 使用 parser 对象的 read_csv 方法读取 NoSeekTellBuffer 对象包装的 data 数据
    result = parser.read_csv(NoSeekTellBuffer(data))
    # 使用 tm 模块的 assert_frame_equal 方法比较 result 和 expected 的内容是否相等
    tm.assert_frame_equal(result, expected)
# 使用 pytest 的 parametrize 装饰器，对 io_class 和 encoding 进行参数化测试
@pytest.mark.parametrize("io_class", [StringIO, BytesIO])
@pytest.mark.parametrize("encoding", [None, "utf-8"])
def test_read_csv_file_handle(all_parsers, io_class, encoding):
    """
    Test whether read_csv does not close user-provided file handles.

    GH 36980
    """
    # 获取所有解析器
    parser = all_parsers
    # 期望的 DataFrame 结果
    expected = DataFrame({"a": [1], "b": [2]})
    
    # 创建包含测试内容的文件句柄
    content = "a,b\n1,2"
    handle = io_class(content.encode("utf-8") if io_class == BytesIO else content)

    # 调用 read_csv 方法读取内容，并断言结果与期望值相等
    tm.assert_frame_equal(parser.read_csv(handle, encoding=encoding), expected)
    # 断言文件句柄没有被关闭
    assert not handle.closed


def test_memory_map_compression(all_parsers, compression):
    """
    Support memory map for compressed files.

    GH 37621
    """
    # 获取所有解析器
    parser = all_parsers
    # 期望的 DataFrame 结果
    expected = DataFrame({"a": [1], "b": [2]})

    # 使用 ensure_clean 上下文管理器创建临时文件路径
    with tm.ensure_clean() as path:
        # 将期望的 DataFrame 写入 CSV 文件，支持压缩格式
        expected.to_csv(path, index=False, compression=compression)

        # 如果解析器使用 pyarrow 引擎，断言会引发 ValueError 异常
        if parser.engine == "pyarrow":
            msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(path, memory_map=True, compression=compression)
            return

        # 使用 memory_map=True 和指定的压缩格式读取 CSV 文件内容
        result = parser.read_csv(path, memory_map=True, compression=compression)

    # 断言读取结果与期望的 DataFrame 相等
    tm.assert_frame_equal(
        result,
        expected,
    )


def test_context_manager(all_parsers, datapath):
    # make sure that opened files are closed
    """
    确保打开的文件会被关闭。

    """
    # 获取所有解析器
    parser = all_parsers

    # 获取示例数据路径
    path = datapath("io", "data", "csv", "iris.csv")

    # 如果解析器使用 pyarrow 引擎，断言会引发 ValueError 异常
    if parser.engine == "pyarrow":
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(path, chunksize=1)
        return

    # 使用 chunksize=1 读取 CSV 文件内容
    reader = parser.read_csv(path, chunksize=1)
    # 断言读取器的文件句柄没有被关闭
    assert not reader.handles.handle.closed
    try:
        # 使用上下文管理器读取下一个数据块，并抛出 AssertionError 异常
        with reader:
            next(reader)
            raise AssertionError
    except AssertionError:
        # 断言读取器的文件句柄已经关闭
        assert reader.handles.handle.closed


def test_context_manageri_user_provided(all_parsers, datapath):
    # make sure that user-provided handles are not closed
    """
    确保用户提供的文件句柄不会被关闭。

    """
    # 获取所有解析器
    parser = all_parsers

    # 使用 open 打开示例数据路径
    with open(datapath("io", "data", "csv", "iris.csv"), encoding="utf-8") as path:
        # 如果解析器使用 pyarrow 引擎，断言会引发 ValueError 异常
        if parser.engine == "pyarrow":
            msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(path, chunksize=1)
            return

        # 使用 chunksize=1 读取 CSV 文件内容
        reader = parser.read_csv(path, chunksize=1)
        # 断言读取器的文件句柄没有被关闭
        assert not reader.handles.handle.closed
        try:
            # 使用上下文管理器读取下一个数据块，并抛出 AssertionError 异常
            with reader:
                next(reader)
                raise AssertionError
        except AssertionError:
            # 断言读取器的文件句柄没有被关闭
            assert not reader.handles.handle.closed


@skip_pyarrow  # ParserError: Empty CSV file
def test_file_descriptor_leak(all_parsers):
    # GH 31488
    """
    文件描述符泄漏测试。

    GH 31488
    """
    # 获取所有解析器
    parser = all_parsers
    # 使用 tm.ensure_clean() 上下文管理器，确保在操作完成后清理临时文件或状态
    with tm.ensure_clean() as path:
        # 使用 pytest.raises() 捕获 EmptyDataError 异常，验证异常消息为 "No columns to parse from file"
        with pytest.raises(EmptyDataError, match="No columns to parse from file"):
            # 使用 parser.read_csv() 方法读取指定路径的 CSV 文件
            parser.read_csv(path)
# 定义一个函数用于测试内存映射功能的各种解析器
def test_memory_map(all_parsers, csv_dir_path):
    # 构建测试文件的路径
    mmap_file = os.path.join(csv_dir_path, "test_mmap.csv")
    # 从参数中获取指定的解析器
    parser = all_parsers

    # 预期的数据框，包含三列数据
    expected = DataFrame(
        {"a": [1, 2, 3], "b": ["one", "two", "three"], "c": ["I", "II", "III"]}
    )

    # 如果解析器使用的引擎是 'pyarrow'，则抛出 ValueError 异常
    if parser.engine == "pyarrow":
        msg = "The 'memory_map' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(mmap_file, memory_map=True)
        # 函数提前返回，不再执行下面的代码
        return

    # 使用内存映射模式读取 CSV 文件
    result = parser.read_csv(mmap_file, memory_map=True)
    # 断言读取的数据框与预期的数据框相等
    tm.assert_frame_equal(result, expected)
```