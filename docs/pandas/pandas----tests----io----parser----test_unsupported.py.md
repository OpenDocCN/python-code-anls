# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_unsupported.py`

```
"""
Tests that features that are currently unsupported in
either the Python or C parser are actually enforced
and are clearly communicated to the user.

Ultimately, the goal is to remove test cases from this
test suite as new feature support is added to the parsers.
"""

# 引入所需的模块和库
from io import StringIO
import os
from pathlib import Path

import pytest  # 导入 pytest 测试框架

from pandas.errors import ParserError  # 导入 pandas 中的 ParserError 错误类

import pandas._testing as tm  # 导入 pandas 测试工具

from pandas.io.parsers import read_csv  # 导入 pandas 的 read_csv 函数
import pandas.io.parsers.readers as parsers  # 导入 pandas.io.parsers.readers 模块

# 忽略特定警告
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 定义一个 pytest fixture，用于测试不支持的引擎
@pytest.fixture(params=["python", "python-fwf"], ids=lambda val: val)
def python_engine(request):
    return request.param


class TestUnsupportedFeatures:
    # 测试 mangle_dupe_cols=False 的情况
    def test_mangle_dupe_cols_false(self):
        # see gh-12935
        data = "a b c\n1 2 3"

        # 对于 "c" 和 "python" 引擎，预期会抛出 TypeError 异常，并匹配 "unexpected keyword"
        for engine in ("c", "python"):
            with pytest.raises(TypeError, match="unexpected keyword"):
                read_csv(StringIO(data), engine=engine, mangle_dupe_cols=True)

    # 测试 C 引擎不支持的情况
    def test_c_engine(self):
        # see gh-6607
        data = "a b c\n1 2 3"
        msg = "does not support"

        # 使用 C 引擎时，指定不支持的选项（预期会抛出 ValueError 异常，并匹配指定的消息）
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine="c", sep=r"\s")
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine="c", sep="\t", quotechar=chr(128))
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine="c", skipfooter=1)

        # 指定了 C 引擎不支持的选项，但没有指定 Python 引擎不支持的选项，预期会产生 ParserWarning 警告
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), sep=r"\s")
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), sep="\t", quotechar=chr(128))
        with tm.assert_produces_warning(parsers.ParserWarning):
            read_csv(StringIO(data), skipfooter=1)

        # 对于包含错误标记化数据的文本，预期会抛出 ParserError 异常，并匹配指定的消息
        text = """                      A       B       C       D        E
one two three   four
a   b   10.0032 5    -0.5109 -2.3358 -0.4645  0.05076  0.3640
a   q   20      4     0.4473  1.4152  0.2834  1.00661  0.1744
x   q   30      3    -0.6662 -0.5243 -0.3580  0.89145  2.5838"""
        msg = "Error tokenizing data"

        with pytest.raises(ParserError, match=msg):
            read_csv(StringIO(text), sep="\\s+")
        with pytest.raises(ParserError, match=msg):
            read_csv(StringIO(text), engine="c", sep="\\s+")

        # 对于只支持长度为 1 的千位分隔符的情况，预期会抛出 ValueError 异常，并匹配指定的消息
        msg = "Only length-1 thousands markers supported"
        data = """A|B|C
1|2,334|5
10|13|10."""
"""
    msg = "ValueError not raised by invalid 'thousands' parameter"
    # 测试 read_csv 函数对于无效 'thousands' 参数能否正确抛出 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        read_csv(StringIO(data), thousands=",,")

    msg = "ValueError not raised by empty 'thousands' parameter"
    # 测试 read_csv 函数对于空 'thousands' 参数能否正确抛出 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        read_csv(StringIO(data), thousands="")

    msg = "Only length-1 line terminators supported"
    data = "a,b,c~~1,2,3~~4,5,6"
    # 测试 read_csv 函数对于不支持的行结束符能否正确抛出 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        read_csv(StringIO(data), lineterminator="~~")

def test_python_engine(self, python_engine):
    from pandas.io.parsers.readers import _python_unsupported as py_unsupported

    data = """1,2,3,,
    1,2,3,4,
    1,2,3,4,5
    1,2,,,
    1,2,3,4,"""

    for default in py_unsupported:
        msg = (
            f"The {default!r} option is not "
            f"supported with the {python_engine!r} engine"
        )
        # 测试 read_csv 函数对于不支持的选项在指定引擎下能否正确抛出 ValueError 异常
        kwargs = {default: object()}
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine=python_engine, **kwargs)

def test_python_engine_file_no_iter(self, python_engine):
    # see gh-16530
    class NoNextBuffer:
        def __init__(self, csv_data) -> None:
            self.data = csv_data

        def __next__(self):
            return self.data.__next__()

        def read(self):
            return self.data

        def readline(self):
            return self.data

    data = "a\n1"
    msg = "'NoNextBuffer' object is not iterable|argument 1 must be an iterator"
    # 测试 read_csv 函数对于不支持迭代器对象能否正确抛出 TypeError 异常
    with pytest.raises(TypeError, match=msg):
        read_csv(NoNextBuffer(data), engine=python_engine)

def test_pyarrow_engine(self):
    from pandas.io.parsers.readers import _pyarrow_unsupported as pa_unsupported

    data = """1,2,3,,
    1,2,3,4,
    1,2,3,4,5
    1,2,,,
    1,2,3,4,"""

    for default in pa_unsupported:
        msg = f"The {default!r} option is not supported with the 'pyarrow' engine"
        kwargs = {default: object()}
        default_needs_bool = {"warn_bad_lines", "error_bad_lines"}
        if default == "dialect":
            kwargs[default] = "excel"  # test a random dialect
        elif default in default_needs_bool:
            kwargs[default] = True
        elif default == "on_bad_lines":
            kwargs[default] = "warn"
        # 测试 read_csv 函数对于不支持的选项在 'pyarrow' 引擎下能否正确抛出 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            read_csv(StringIO(data), engine="pyarrow", **kwargs)
    # 定义一个测试方法，用于检查在给定所有解析器的情况下，是否正确处理错误行函数的设置
    def test_on_bad_lines_callable_python_or_pyarrow(self, all_parsers):
        # GH 5686
        # GH 54643
        # 创建一个包含"a,b\n1,2"内容的内存中的文本流
        sio = StringIO("a,b\n1,2")
        # 定义一个匿名函数，该函数在这里用于占位，实际应用中需要根据需求来定义具体的行为
        bad_lines_func = lambda x: x
        # 获取当前测试中使用的解析器
        parser = all_parsers
        # 检查解析器的引擎类型是否不是"python"或者"pyarrow"
        if all_parsers.engine not in ["python", "pyarrow"]:
            # 如果不是，构造错误消息
            msg = (
                "on_bad_line can only be a callable "
                "function if engine='python' or 'pyarrow'"
            )
            # 使用 pytest 来验证是否抛出 ValueError 异常，并检查异常消息是否匹配预期
            with pytest.raises(ValueError, match=msg):
                parser.read_csv(sio, on_bad_lines=bad_lines_func)
        else:
            # 如果引擎类型是"python"或者"pyarrow"，则调用解析器读取 CSV 数据，传入错误行处理函数
            parser.read_csv(sio, on_bad_lines=bad_lines_func)
# 定义测试函数，用于验证在无效 usecols 参数下文件句柄的关闭
def test_close_file_handle_on_invalid_usecols(all_parsers):
    # 获取所有解析器对象
    parser = all_parsers

    # 设定错误类型为 ValueError
    error = ValueError

    # 如果解析器的引擎是 "pyarrow"，则跳过当前测试，因为会引发 pyarrow.lib.ArrowKeyError
    if parser.engine == "pyarrow":
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")

    # 使用 tm.ensure_clean 创建一个临时文件 fname，确保在测试结束时删除
    with tm.ensure_clean("test.csv") as fname:
        # 向临时文件写入文本内容 "col1,col2\na,b\n1,2"，编码为 UTF-8
        Path(fname).write_text("col1,col2\na,b\n1,2", encoding="utf-8")
        
        # 禁用 pytest 的警告检查
        with tm.assert_produces_warning(False):
            # 预期在读取 CSV 文件时会引发 error 类型的异常，并且异常信息中包含 "col3"
            with pytest.raises(error, match="col3"):
                parser.read_csv(fname, usecols=["col1", "col2", "col3"])
        
        # 如果在 Windows 系统下，文件句柄还指向该文件时，删除操作可能会失败，因此先删除文件句柄
        os.unlink(fname)


# 定义测试函数，用于验证在无效文件输入时引发 ValueError 异常
def test_invalid_file_inputs(request, all_parsers):
    # 获取所有解析器对象
    parser = all_parsers
    
    # 如果解析器的引擎是 "python"，标记当前测试为预期失败，原因是 "python" 引擎支持列表输入
    if parser.engine == "python":
        request.applymarker(
            pytest.mark.xfail(reason=f"{parser.engine} engine supports lists.")
        )

    # 预期在读取空列表作为文件输入时会引发 ValueError 异常，并且异常信息中包含 "Invalid"
    with pytest.raises(ValueError, match="Invalid"):
        parser.read_csv([])


# 定义测试函数，用于验证在指定无效 dtype_backend 参数时引发 ValueError 异常
def test_invalid_dtype_backend(all_parsers):
    # 获取所有解析器对象
    parser = all_parsers
    
    # 错误信息提示
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    
    # 预期在读取 CSV 文件时使用了无效的 dtype_backend 参数时会引发 ValueError 异常，并且异常信息符合指定的 msg
    with pytest.raises(ValueError, match=msg):
        parser.read_csv("test", dtype_backend="numpy")
```