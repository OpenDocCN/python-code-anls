# `D:\src\scipysrc\pandas\pandas\tests\io\test_common.py`

```
"""
Tests for the pandas.io.common functionalities
"""

# 导入所需模块和库
import codecs  # 提供编解码器和文件对象接口的实用功能
import errno  # 定义常见的错误码
from functools import partial  # 创建可调用对象的高级工具
from io import (  # 提供对流和缓冲区接口的核心工具
    BytesIO,  # 用于操作字节数据的流
    StringIO,  # 用于操作字符串数据的流
    UnsupportedOperation,  # 当对流执行不支持的操作时引发的异常
)
import mmap  # 提供在文件上执行内存映射的支持
import os  # 提供与操作系统交互的功能
from pathlib import Path  # 提供处理路径的对象
import pickle  # 用于序列化和反序列化Python对象
import tempfile  # 提供创建临时文件和目录的功能

import numpy as np  # 数值计算库
import pytest  # Python的单元测试框架

from pandas.compat import (  # pandas的兼容性模块
    WASM,  # WebAssembly的兼容性检查
    is_platform_windows,  # 检查操作系统是否为Windows
)

import pandas as pd  # 提供数据分析功能的库
import pandas._testing as tm  # pandas测试工具

import pandas.io.common as icom  # pandas的I/O公共功能

# 忽略特定的警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


class CustomFSPath:
    """For testing fspath on unknown objects"""

    def __init__(self, path) -> None:
        self.path = path

    def __fspath__(self):
        return self.path


HERE = os.path.abspath(os.path.dirname(__file__))  # 获取当前文件所在目录的绝对路径


# https://github.com/cython/cython/issues/1720
class TestCommonIOCapabilities:
    data1 = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""

    def test_expand_user(self):
        filename = "~/sometest"
        expanded_name = icom._expand_user(filename)  # 执行用户目录扩展

        assert expanded_name != filename  # 断言扩展后的文件名与原始文件名不同
        assert os.path.isabs(expanded_name)  # 断言扩展后的文件名是绝对路径
        assert os.path.expanduser(filename) == expanded_name  # 断言使用os.path扩展用户路径后与icm._expand_user的结果相同

    def test_expand_user_normal_path(self):
        filename = "/somefolder/sometest"
        expanded_name = icom._expand_user(filename)  # 执行用户目录扩展

        assert expanded_name == filename  # 断言未更改路径
        assert os.path.expanduser(filename) == expanded_name  # 断言使用os.path扩展用户路径后与icm._expand_user的结果相同

    def test_stringify_path_pathlib(self):
        rel_path = icom.stringify_path(Path("."))  # 将Path对象转换为字符串路径
        assert rel_path == "."  # 断言结果为当前目录路径
        redundant_path = icom.stringify_path(Path("foo//bar"))  # 将Path对象转换为字符串路径，处理多余的分隔符
        assert redundant_path == os.path.join("foo", "bar")  # 断言结果与预期路径拼接相同

    def test_stringify_path_fspath(self):
        p = CustomFSPath("foo/bar.csv")
        result = icom.stringify_path(p)  # 将自定义对象转换为字符串路径
        assert result == "foo/bar.csv"  # 断言结果与预期路径相同

    def test_stringify_file_and_path_like(self):
        # GH 38125: do not stringify file objects that are also path-like
        fsspec = pytest.importorskip("fsspec")  # 导入并检查fsspec库是否可用
        with tm.ensure_clean() as path:  # 确保在临时路径中进行操作
            with fsspec.open(f"file://{path}", mode="wb") as fsspec_obj:
                assert fsspec_obj == icom.stringify_path(fsspec_obj)  # 断言不对同时是文件对象和路径对象的对象进行字符串化处理

    @pytest.mark.parametrize("path_type", [str, CustomFSPath, Path])
    def test_infer_compression_from_path(self, compression_format, path_type):
        extension, expected = compression_format
        path = path_type("foo/bar.csv" + extension)
        compression = icom.infer_compression(path, compression="infer")  # 推断路径的压缩格式
        assert compression == expected  # 断言推断的压缩格式与预期相同

    @pytest.mark.parametrize("path_type", [str, CustomFSPath, Path])
    # 定义测试函数，测试带有路径参数的get_handle方法
    def test_get_handle_with_path(self, path_type):
        # 在临时目录中创建临时文件夹，并生成文件名
        with tempfile.TemporaryDirectory(dir=Path.home()) as tmp:
            filename = path_type("~/" + Path(tmp).name + "/sometest")
            # 调用icom模块的get_handle方法，以写入模式打开文件
            with icom.get_handle(filename, "w") as handles:
                # 断言文件句柄的路径是绝对路径
                assert Path(handles.handle.name).is_absolute()
                # 断言展开用户路径后与句柄的文件名相匹配
                assert os.path.expanduser(filename) == handles.handle.name

    # 定义测试函数，测试带有缓冲区参数的get_handle方法
    def test_get_handle_with_buffer(self):
        # 使用StringIO创建输入缓冲区
        with StringIO() as input_buffer:
            # 调用icom模块的get_handle方法，以读取模式打开缓冲区
            with icom.get_handle(input_buffer, "r") as handles:
                # 断言句柄与输入缓冲区对象相等
                assert handles.handle == input_buffer
            # 断言输入缓冲区没有被关闭
            assert not input_buffer.closed
        # 断言输入缓冲区已经关闭
        assert input_buffer.closed

    # 测试BytesIOWrapper(get_handle)方法返回正确数量的字节
    def test_bytesiowrapper_returns_correct_bytes(self):
        # 测试包含拉丁字母、ucs-2和ucs-4字符的数据
        data = """a,b,c
# 使用 icom 模块的 get_handle 方法打开数据流，并以二进制模式读取
with icom.get_handle(StringIO(data), "rb", is_text=False) as handles:
    # 初始化一个空的结果字节串
    result = b""
    # 指定每次读取的块大小为 5 字节
    chunksize = 5
    # 循环读取数据流中的内容
    while True:
        # 从处理句柄中读取指定大小的数据块
        chunk = handles.handle.read(chunksize)
        # 确保每个读取的块大小不超过指定的 chunksize
        assert len(chunk) <= chunksize
        # 如果实际读取的块大小小于 chunksize，则可能已经到达文件末尾
        if len(chunk) < chunksize:
            # 在读取到文件末尾时，确保再次读取返回空内容
            assert len(handles.handle.read()) == 0
            # 将最后一个块添加到结果中
            result += chunk
            break
        # 将读取的块添加到结果中
        result += chunk
    # 断言最终的结果与原始数据编码成 utf-8 后相同
    assert result == data.encode("utf-8")

# 测试 pyarrow 是否能够处理通过 get_handle 打开的文件
def test_get_handle_pyarrow_compat(self):
    # 使用 pytest 的 importorskip 方法导入 pyarrow.csv 模块，如果导入失败则跳过测试
    pa_csv = pytest.importorskip("pyarrow.csv")
    
    # 定义包含不同字符集的测试数据
    data = """a,b,c
1,2,3
©,®,®
Look,a snake,🐍"""
    
    # 定义预期的 Pandas DataFrame 结果
    expected = pd.DataFrame(
        {"a": ["1", "©", "Look"], "b": ["2", "®", "a snake"], "c": ["3", "®", "🐍"]}
    )
    
    # 使用 StringIO 创建一个数据流
    s = StringIO(data)
    # 使用 icom 模块的 get_handle 方法打开数据流，并以二进制模式读取
    with icom.get_handle(s, "rb", is_text=False) as handles:
        # 使用 pyarrow.csv 模块读取处理句柄中的内容，并转换为 Pandas DataFrame
        df = pa_csv.read_csv(handles.handle).to_pandas()
        # 断言读取的 DataFrame 与预期的 DataFrame 相等
        tm.assert_frame_equal(df, expected)
        # 断言数据流没有被关闭
        assert not s.closed

# 测试迭代器功能
def test_iterator(self):
    # 使用 pandas 的 read_csv 方法读取数据流 self.data1，并指定 chunksize 为 1
    with pd.read_csv(StringIO(self.data1), chunksize=1) as reader:
        # 将所有 chunk 合并成一个 DataFrame，忽略索引
        result = pd.concat(reader, ignore_index=True)
    # 使用 pandas 读取数据流 self.data1 作为预期结果
    expected = pd.read_csv(StringIO(self.data1))
    # 断言合并后的结果 DataFrame 与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)
    
    # GH12153
    # 使用 pandas 的 read_csv 方法读取数据流 self.data1，并指定 chunksize 为 1
    with pd.read_csv(StringIO(self.data1), chunksize=1) as it:
        # 读取第一个 chunk，并与预期的第一行 DataFrame 相比较
        first = next(it)
        tm.assert_frame_equal(first, expected.iloc[[0]])
        # 合并剩余的 chunk，并与预期的第二行至末尾 DataFrame 相比较
        tm.assert_frame_equal(pd.concat(it), expected.iloc[1:])

# 参数化测试，验证不同的读取方法在文件不存在时的异常处理
@pytest.mark.skipif(WASM, reason="limited file system access on WASM")
@pytest.mark.parametrize(
    "reader, module, error_class, fn_ext",
    [
        (pd.read_csv, "os", FileNotFoundError, "csv"),
        (pd.read_fwf, "os", FileNotFoundError, "txt"),
        (pd.read_excel, "xlrd", FileNotFoundError, "xlsx"),
        (pd.read_feather, "pyarrow", OSError, "feather"),
        (pd.read_hdf, "tables", FileNotFoundError, "h5"),
        (pd.read_stata, "os", FileNotFoundError, "dta"),
        (pd.read_sas, "os", FileNotFoundError, "sas7bdat"),
        (pd.read_json, "os", FileNotFoundError, "json"),
        (pd.read_pickle, "os", FileNotFoundError, "pickle"),
    ],
)
    # 定义测试方法test_read_non_existent，用于测试读取不存在文件时的情况，包括读取器、模块、错误类和文件扩展名参数
    def test_read_non_existent(self, reader, module, error_class, fn_ext):
        # 使用pytest的importorskip装饰器，如果模块不可用则跳过测试
        pytest.importorskip(module)

        # 构建文件路径，指向不存在的文件，使用当前目录HERE下的"data"子目录，文件名包含给定的文件扩展名fn_ext
        path = os.path.join(HERE, "data", "does_not_exist." + fn_ext)

        # 定义多个期望的错误消息，用于匹配异常信息中的多种可能情况
        msg1 = rf"File (b')?.+does_not_exist\.{fn_ext}'? does not exist"
        msg2 = rf"\[Errno 2\] No such file or directory: '.+does_not_exist\.{fn_ext}'"
        msg3 = "Expected object or value"
        msg4 = "path_or_buf needs to be a string file path or file-like"
        msg5 = (
            rf"\[Errno 2\] File .+does_not_exist\.{fn_ext} does not exist: "
            rf"'.+does_not_exist\.{fn_ext}'"
        )
        msg6 = rf"\[Errno 2\] 没有那个文件或目录: '.+does_not_exist\.{fn_ext}'"
        msg7 = (
            rf"\[Errno 2\] File o directory non esistente: '.+does_not_exist\.{fn_ext}'"
        )
        msg8 = rf"Failed to open local file.+does_not_exist\.{fn_ext}"

        # 使用pytest的raises断言来验证reader(path)操作抛出error_class类型的异常，并匹配预期的错误消息
        with pytest.raises(
            error_class,
            match=rf"({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})",
        ):
            reader(path)

    # 使用pytest的parametrize装饰器定义多组参数化测试
    @pytest.mark.parametrize(
        "method, module, error_class, fn_ext",
        [
            (pd.DataFrame.to_csv, "os", OSError, "csv"),
            (pd.DataFrame.to_html, "os", OSError, "html"),
            (pd.DataFrame.to_excel, "xlrd", OSError, "xlsx"),
            (pd.DataFrame.to_feather, "pyarrow", OSError, "feather"),
            (pd.DataFrame.to_parquet, "pyarrow", OSError, "parquet"),
            (pd.DataFrame.to_stata, "os", OSError, "dta"),
            (pd.DataFrame.to_json, "os", OSError, "json"),
            (pd.DataFrame.to_pickle, "os", OSError, "pickle"),
        ],
    )
    # NOTE: Missing parent directory for pd.DataFrame.to_hdf is handled by PyTables
    # 定义测试方法test_write_missing_parent_directory，用于测试写入时缺失父目录的情况，包括方法、模块、错误类和文件扩展名参数
    def test_write_missing_parent_directory(self, method, module, error_class, fn_ext):
        # 使用pytest的importorskip装饰器，如果模块不可用则跳过测试
        pytest.importorskip(module)

        # 创建一个虚拟的DataFrame对象dummy_frame，用于测试写入操作
        dummy_frame = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4], "c": [3, 4, 5]})

        # 构建文件路径，指向不存在的文件夹"missing_folder"下的文件，使用当前目录HERE，文件名包含给定的文件扩展名fn_ext
        path = os.path.join(HERE, "data", "missing_folder", "does_not_exist." + fn_ext)

        # 使用pytest的raises断言来验证method(dummy_frame, path)操作抛出error_class类型的异常，并匹配预期的错误消息
        with pytest.raises(
            error_class,
            match=r"Cannot save file into a non-existent directory: .*missing_folder",
        ):
            method(dummy_frame, path)

    # 使用pytest的mark.skipif装饰器，如果WASM为True，则跳过此测试，原因是WASM环境下有限的文件系统访问权限
    @pytest.mark.skipif(WASM, reason="limited file system access on WASM")
    # 使用 pytest 的 parametrize 装饰器，为测试方法参数化，以便多次运行测试用例
    @pytest.mark.parametrize(
        "reader, module, error_class, fn_ext",
        [
            # 参数化测试数据，包括读取函数、所需模块、预期错误类型、文件扩展名
            (pd.read_csv, "os", FileNotFoundError, "csv"),
            (pd.read_table, "os", FileNotFoundError, "csv"),
            (pd.read_fwf, "os", FileNotFoundError, "txt"),
            (pd.read_excel, "xlrd", FileNotFoundError, "xlsx"),
            (pd.read_feather, "pyarrow", OSError, "feather"),
            (pd.read_hdf, "tables", FileNotFoundError, "h5"),
            (pd.read_stata, "os", FileNotFoundError, "dta"),
            (pd.read_sas, "os", FileNotFoundError, "sas7bdat"),
            (pd.read_json, "os", FileNotFoundError, "json"),
            (pd.read_pickle, "os", FileNotFoundError, "pickle"),
        ],
    )
    # 定义测试方法，用于验证文件读取函数在用户主目录扩展后是否能正确处理异常情况
    def test_read_expands_user_home_dir(
        self, reader, module, error_class, fn_ext, monkeypatch
    ):
        # 如果所需模块不可用，则跳过测试
        pytest.importorskip(module)

        # 构造文件路径，包含用户主目录的扩展，并设置 monkeypatch 以模拟用户主目录的路径
        path = os.path.join("~", "does_not_exist." + fn_ext)
        monkeypatch.setattr(icom, "_expand_user", lambda x: os.path.join("foo", x))

        # 定义匹配错误消息的正则表达式模式，用于断言特定异常被正确抛出
        msg1 = rf"File (b')?.+does_not_exist\.{fn_ext}'? does not exist"
        msg2 = rf"\[Errno 2\] No such file or directory: '.+does_not_exist\.{fn_ext}'"
        msg3 = "Unexpected character found when decoding 'false'"
        msg4 = "path_or_buf needs to be a string file path or file-like"
        msg5 = (
            rf"\[Errno 2\] File .+does_not_exist\.{fn_ext} does not exist: "
            rf"'.+does_not_exist\.{fn_ext}'"
        )
        msg6 = rf"\[Errno 2\] 没有那个文件或目录: '.+does_not_exist\.{fn_ext}'"
        msg7 = (
            rf"\[Errno 2\] File o directory non esistente: '.+does_not_exist\.{fn_ext}'"
        )
        msg8 = rf"Failed to open local file.+does_not_exist\.{fn_ext}"

        # 使用 pytest.raises 断言特定的异常被抛出，并匹配其中任何一个预定义的错误消息模式
        with pytest.raises(
            error_class,
            match=rf"({msg1}|{msg2}|{msg3}|{msg4}|{msg5}|{msg6}|{msg7}|{msg8})",
        ):
            # 调用被测试的文件读取函数，传入构造的文件路径，验证异常情况
            reader(path)
    # 使用 pytest 的参数化装饰器来定义测试用例，每个元组包含读取函数、依赖模块和文件路径
    @pytest.mark.parametrize(
        "reader, module, path",
        [
            # 测试用例：读取 CSV 文件
            (pd.read_csv, "os", ("io", "data", "csv", "iris.csv")),
            # 测试用例：读取文本文件（通用）
            (pd.read_table, "os", ("io", "data", "csv", "iris.csv")),
            # 测试用例：读取固定宽度格式的文本文件
            (
                pd.read_fwf,
                "os",
                ("io", "data", "fixed_width", "fixed_width_format.txt"),
            ),
            # 测试用例：读取 Excel 文件
            (pd.read_excel, "xlrd", ("io", "data", "excel", "test1.xlsx")),
            # 测试用例：读取 Feather 文件
            (
                pd.read_feather,
                "pyarrow",
                ("io", "data", "feather", "feather-0_3_1.feather"),
            ),
            # 测试用例：读取 HDF5 文件
            (
                pd.read_hdf,
                "tables",
                ("io", "data", "legacy_hdf", "pytables_native2.h5"),
            ),
            # 测试用例：读取 Stata 文件
            (pd.read_stata, "os", ("io", "data", "stata", "stata10_115.dta")),
            # 测试用例：读取 SAS 文件
            (pd.read_sas, "os", ("io", "sas", "data", "test1.sas7bdat")),
            # 测试用例：读取 JSON 文件
            (pd.read_json, "os", ("io", "json", "data", "tsframe_v012.json")),
            # 测试用例：读取 Pickle 文件
            (
                pd.read_pickle,
                "os",
                ("io", "data", "pickle", "categorical.0.25.0.pickle"),
            ),
        ],
    )
    # 定义测试方法：测试读取不同文件路径下的数据，并进行比较
    def test_read_fspath_all(self, reader, module, path, datapath):
        # 使用 pytest 的 importorskip 函数导入必要的模块或跳过测试
        pytest.importorskip(module)
        # 调用 datapath 函数获取文件的完整路径
        path = datapath(*path)

        # 创建 CustomFSPath 对象
        mypath = CustomFSPath(path)
        # 使用指定的读取函数读取数据
        result = reader(mypath)
        # 读取原始文件数据
        expected = reader(path)

        # 根据文件扩展名判断文件类型，选择不同的比较方法
        if path.endswith(".pickle"):
            # 如果是 Pickle 文件，使用 assert_categorical_equal 方法比较结果
            # 这里假设是比较分类数据
            tm.assert_categorical_equal(result, expected)
        else:
            # 否则使用 assert_frame_equal 方法比较结果，假设是比较数据框架
            tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，用于测试以指定的写入器名称和参数写入数据到文件系统路径的功能
    def test_write_fspath_all(self, writer_name, writer_kwargs, module):
        # 如果写入器名称在 ["to_latex"] 中，需要使用 Styler 实现，否则跳过测试
        if writer_name in ["to_latex"]:  # uses Styler implementation
            pytest.importorskip("jinja2")
        
        # 确保字符串和文件系统路径参数的有效性并进行清理
        p1 = tm.ensure_clean("string")
        p2 = tm.ensure_clean("fspath")
        
        # 创建一个简单的 DataFrame
        df = pd.DataFrame({"A": [1, 2]})

        # 使用两个上下文管理器分别打开字符串和文件系统路径
        with p1 as string, p2 as fspath:
            # 根据指定的模块导入必要的依赖
            pytest.importorskip(module)
            
            # 使用自定义的文件系统路径对象创建路径
            mypath = CustomFSPath(fspath)
            
            # 获取 DataFrame 的指定写入器方法
            writer = getattr(df, writer_name)

            # 将 DataFrame 内容分别写入字符串和文件系统路径
            writer(string, **writer_kwargs)
            writer(mypath, **writer_kwargs)
            
            # 打开字符串和文件系统路径对应的文件进行读取比较
            with open(string, "rb") as f_str, open(fspath, "rb") as f_path:
                if writer_name == "to_excel":
                    # 如果是写入 Excel 格式，读取结果进行 DataFrame 比较
                    # Excel 文件包含时间创建数据，可能导致持续集成失败，因此特殊处理
                    result = pd.read_excel(f_str, **writer_kwargs)
                    expected = pd.read_excel(f_path, **writer_kwargs)
                    tm.assert_frame_equal(result, expected)
                else:
                    # 否则，直接读取字符串和文件系统路径中的数据并进行比较
                    result = f_str.read()
                    expected = f_path.read()
                    assert result == expected

    # 定义另一个测试函数，测试将 DataFrame 写入 HDF5 格式文件并比较结果
    def test_write_fspath_hdf5(self):
        # 同 test_write_fspath_all，但 HDF5 文件不一定是字节完全相同的，需要特殊处理
        # 因此，读取后比较数据的相等性
        pytest.importorskip("tables")

        # 创建一个简单的 DataFrame
        df = pd.DataFrame({"A": [1, 2]})
        
        # 确保字符串和文件系统路径参数的有效性并进行清理
        p1 = tm.ensure_clean("string")
        p2 = tm.ensure_clean("fspath")

        # 使用两个上下文管理器分别打开字符串和文件系统路径
        with p1 as string, p2 as fspath:
            # 使用自定义的文件系统路径对象创建路径
            mypath = CustomFSPath(fspath)
            
            # 将 DataFrame 写入 HDF5 格式文件，使用相同的键名 "bar"
            df.to_hdf(mypath, key="bar")
            df.to_hdf(string, key="bar")

            # 读取并比较 HDF5 文件中的数据
            result = pd.read_hdf(fspath, key="bar")
            expected = pd.read_hdf(string, key="bar")

        # 使用测试框架提供的方法比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
@pytest.fixture
def mmap_file(datapath):
    return datapath("io", "data", "csv", "test_mmap.csv")


class TestMMapWrapper:
    @pytest.mark.skipif(WASM, reason="limited file system access on WASM")
    def test_constructor_bad_file(self, mmap_file):
        # 创建一个不是文件的 StringIO 对象
        non_file = StringIO("I am not a file")
        # 定义一个匿名函数，模拟 fileno 方法返回值为 -1
        non_file.fileno = lambda: -1

        # 根据平台不同设置不同的错误消息和异常类
        if is_platform_windows():
            msg = "The parameter is incorrect"
            err = OSError
        else:
            msg = "[Errno 22]"
            err = mmap.error

        # 使用 pytest 来验证调用 _maybe_memory_map 方法时是否会抛出特定的异常
        with pytest.raises(err, match=msg):
            icom._maybe_memory_map(non_file, True)

        # 打开真实的文件，确保其正常打开
        with open(mmap_file, encoding="utf-8") as target:
            pass

        # 使用 pytest 来验证当文件关闭后调用 _maybe_memory_map 方法是否会抛出特定的异常
        msg = "I/O operation on closed file"
        with pytest.raises(ValueError, match=msg):
            icom._maybe_memory_map(target, True)

    @pytest.mark.skipif(WASM, reason="limited file system access on WASM")
    def test_next(self, mmap_file):
        # 打开文件以读取内容
        with open(mmap_file, encoding="utf-8") as target:
            # 读取文件所有行
            lines = target.readlines()

            # 使用 icom.get_handle 方法处理文件句柄，确保文件通过内存映射方式打开
            with icom.get_handle(
                target, "r", is_text=True, memory_map=True
            ) as wrappers:
                wrapper = wrappers.handle
                # 断言处理器的缓冲区为 mmap.mmap 类型
                assert isinstance(wrapper.buffer.buffer, mmap.mmap)

                # 逐行比较处理器返回的下一行内容与实际文件中的内容
                for line in lines:
                    next_line = next(wrapper)
                    assert next_line.strip() == line.strip()

                # 使用 pytest 验证文件读取到末尾时调用 next 方法是否会抛出 StopIteration 异常
                with pytest.raises(StopIteration, match=r"^$"):
                    next(wrapper)

    def test_unknown_engine(self):
        # 确保在测试期间路径干净，创建一个测试用的 DataFrame，并将其保存为 CSV 文件
        with tm.ensure_clean() as path:
            df = pd.DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=pd.Index(list("ABCD"), dtype=object),
                index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),
            )
            df.to_csv(path)
            # 使用 pytest 验证调用 pd.read_csv 时使用未知的引擎参数是否会抛出 ValueError 异常
            with pytest.raises(ValueError, match="Unknown engine"):
                pd.read_csv(path, engine="pyt")

    def test_binary_mode(self):
        """
        'encoding' shouldn't be passed to 'open' in binary mode.

        GH 35058
        """
        # 确保在测试期间路径干净，创建一个测试用的 DataFrame，并将其保存为二进制模式的 CSV 文件
        with tm.ensure_clean() as path:
            df = pd.DataFrame(
                1.1 * np.arange(120).reshape((30, 4)),
                columns=pd.Index(list("ABCD"), dtype=object),
                index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),
            )
            df.to_csv(path, mode="w+b")
            # 使用 tm.assert_frame_equal 方法验证 DataFrame 在读取后与原始 DataFrame 是否相等
            tm.assert_frame_equal(df, pd.read_csv(path, index_col=0))

    @pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
    @pytest.mark.parametrize("compression_", ["bz2", "xz"])
    # 定义一个测试方法，用于检查在指定编码和压缩方式下是否缺少 UTF BOM（字节顺序标记）警告
    def test_warning_missing_utf_bom(self, encoding, compression_):
        """
        bz2 and xz do not write the byte order mark (BOM) for utf-16/32.

        https://stackoverflow.com/questions/55171439

        GH 35681
        """
        # 创建一个包含数值数据的 Pandas DataFrame，30行4列
        df = pd.DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=pd.Index(list("ABCD"), dtype=object),
            index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),
        )
        # 使用临时文件上下文确保操作后文件系统的干净状态
        with tm.ensure_clean() as path:
            # 使用上下文确保在写入时产生 Unicode 警告，并匹配指定的警告信息
            with tm.assert_produces_warning(UnicodeWarning, match="byte order mark"):
                df.to_csv(path, compression=compression_, encoding=encoding)

            # 读取操作应该失败（否则不需要警告）
            # 定义一个正则表达式模式来匹配可能的 Unicode 错误信息
            msg = (
                r"UTF-\d+ stream does not start with BOM|"
                r"'utf-\d+' codec can't decode byte"
            )
            # 使用 pytest 断言应抛出 UnicodeError，并匹配定义的错误消息
            with pytest.raises(UnicodeError, match=msg):
                pd.read_csv(path, compression=compression_, encoding=encoding)
# 定义一个测试函数，用于检查是否为 fsspec URL
def test_is_fsspec_url():
    # 断言以下 URL 是 fsspec URL
    assert icom.is_fsspec_url("gcs://pandas/somethingelse.com")
    assert icom.is_fsspec_url("gs://pandas/somethingelse.com")
    # 下面这个是唯一不需要 fsspec 处理的远程 URL
    assert not icom.is_fsspec_url("http://pandas/somethingelse.com")
    assert not icom.is_fsspec_url("random:pandas/somethingelse.com")
    assert not icom.is_fsspec_url("/local/path")
    assert not icom.is_fsspec_url("relative/local/path")
    # 字符串中的 fsspec URL 不应该被识别
    assert not icom.is_fsspec_url("this is not fsspec://url")
    assert not icom.is_fsspec_url("{'url': 'gs://pandas/somethingelse.com'}")
    # 接受符合 RFC 3986 标准的所有 URL
    assert icom.is_fsspec_url("RFC-3986+compliant.spec://something")

# 参数化测试，测试不同的编码和格式
@pytest.mark.parametrize("encoding", [None, "utf-8"])
@pytest.mark.parametrize("format", ["csv", "json"])
def test_codecs_encoding(encoding, format):
    # GH39247
    # 创建一个期望的 DataFrame
    expected = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD"), dtype=object),
        index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    # 确保在临时路径上进行操作
    with tm.ensure_clean() as path:
        # 使用编码方式打开文件，写入期望的格式数据
        with codecs.open(path, mode="w", encoding=encoding) as handle:
            getattr(expected, f"to_{format}")(handle)
        # 使用编码方式打开文件，读取数据到 DataFrame
        with codecs.open(path, mode="r", encoding=encoding) as handle:
            if format == "csv":
                df = pd.read_csv(handle, index_col=0)
            else:
                df = pd.read_json(handle)
    # 断言期望的 DataFrame 和读取出的 DataFrame 相等
    tm.assert_frame_equal(expected, df)

# 测试获取编码器和解码器
def test_codecs_get_writer_reader():
    # GH39247
    # 创建一个期望的 DataFrame
    expected = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD"), dtype=object),
        index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    # 确保在临时路径上进行操作
    with tm.ensure_clean() as path:
        # 使用 UTF-8 编码打开文件，写入 DataFrame 的 CSV 格式数据
        with open(path, "wb") as handle:
            with codecs.getwriter("utf-8")(handle) as encoded:
                expected.to_csv(encoded)
        # 使用 UTF-8 解码打开文件，读取数据到 DataFrame
        with open(path, "rb") as handle:
            with codecs.getreader("utf-8")(handle) as encoded:
                df = pd.read_csv(encoded, index_col=0)
    # 断言期望的 DataFrame 和读取出的 DataFrame 相等
    tm.assert_frame_equal(expected, df)

# 参数化测试，测试不同的 io 类型、模式和错误消息
@pytest.mark.parametrize(
    "io_class,mode,msg",
    [
        (BytesIO, "t", "a bytes-like object is required, not 'str'"),
        (StringIO, "b", "string argument expected, got 'bytes'"),
    ],
)
def test_explicit_encoding(io_class, mode, msg):
    # GH39247; 此测试确保如果用户提供 mode="*t" 或 "*b"，则使用它
    # 在这个测试案例中，故意请求错误的模式会导致错误
    expected = pd.DataFrame(
        1.1 * np.arange(120).reshape((30, 4)),
        columns=pd.Index(list("ABCD"), dtype=object),
        index=pd.Index([f"i-{i}" for i in range(30)], dtype=object),
    )
    # 使用指定的输入输出类实例化一个上下文管理器对象，并将其赋值给变量buffer
    with io_class() as buffer:
        # 使用pytest断言捕获预期的TypeError异常，并且异常消息必须匹配msg参数
        with pytest.raises(TypeError, match=msg):
            # 调用expected对象的to_csv方法，将其内容以指定模式"w{mode}"写入到buffer中
            expected.to_csv(buffer, mode=f"w{mode}")
@pytest.mark.parametrize("encoding_errors", ["strict", "replace"])
@pytest.mark.parametrize("format", ["csv", "json"])
def test_encoding_errors(encoding_errors, format):
    # 标记测试用例，使用参数化测试，测试编码错误处理
    msg = "'utf-8' codec can't decode byte"
    bad_encoding = b"\xe4"

    if format == "csv":
        # 如果格式为csv，创建包含错误编码的内容
        content = b"," + bad_encoding + b"\n" + bad_encoding * 2 + b"," + bad_encoding
        reader = partial(pd.read_csv, index_col=0)
    else:
        # 如果格式为json，创建包含错误编码的内容
        content = (
            b'{"'
            + bad_encoding * 2
            + b'": {"'
            + bad_encoding
            + b'":"'
            + bad_encoding
            + b'"}}'
        )
        reader = partial(pd.read_json, orient="index")
    
    # 在临时路径上创建文件，并写入内容
    with tm.ensure_clean() as path:
        file = Path(path)
        file.write_bytes(content)

        if encoding_errors != "replace":
            # 如果不是使用替换策略来处理编码错误，预期会抛出UnicodeDecodeError异常
            with pytest.raises(UnicodeDecodeError, match=msg):
                reader(path, encoding_errors=encoding_errors)
        else:
            # 使用替换策略来处理编码错误，读取文件并验证结果
            df = reader(path, encoding_errors=encoding_errors)
            decoded = bad_encoding.decode(errors=encoding_errors)
            expected = pd.DataFrame({decoded: [decoded]}, index=[decoded * 2])
            tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize("encoding_errors", [0, None])
def test_encoding_errors_badtype(encoding_errors):
    # 标记测试用例，使用参数化测试，测试错误的编码类型处理
    content = StringIO("A,B\n1,2\n3,4\n")
    reader = partial(pd.read_csv, encoding_errors=encoding_errors)
    expected_error = "encoding_errors must be a string, got "
    expected_error += f"{type(encoding_errors).__name__}"
    # 预期会抛出值错误，匹配错误消息
    with pytest.raises(ValueError, match=expected_error):
        reader(content)


def test_bad_encdoing_errors():
    # 标记测试用例，测试错误的编码处理
    with tm.ensure_clean() as path:
        # 使用pytest预期会抛出查找错误处理器名字时的查找错误异常
        with pytest.raises(LookupError, match="unknown error handler name"):
            icom.get_handle(path, "w", errors="bad")


@pytest.mark.skipif(WASM, reason="limited file system access on WASM")
def test_errno_attribute():
    # 标记测试用例，使用参数化测试，测试错误号属性
    with pytest.raises(FileNotFoundError, match="\\[Errno 2\\]") as err:
        # 预期会抛出文件未找到错误，匹配特定错误消息
        pd.read_csv("doesnt_exist")
        assert err.errno == errno.ENOENT


def test_fail_mmap():
    # 标记测试用例，测试内存映射失败情况
    with pytest.raises(UnsupportedOperation, match="fileno"):
        with BytesIO() as buffer:
            icom.get_handle(buffer, "rb", memory_map=True)


def test_close_on_error():
    # 标记测试用例，测试错误时的关闭处理
    class TestError:
        def close(self):
            raise OSError("test")

    with pytest.raises(OSError, match="test"):
        with BytesIO() as buffer:
            with icom.get_handle(buffer, "rb") as handles:
                handles.created_handles.append(TestError())


@pytest.mark.parametrize(
    "reader",
    [
        pd.read_csv,
        pd.read_fwf,
        pd.read_excel,
        pd.read_feather,
        pd.read_hdf,
        pd.read_stata,
        pd.read_sas,
        pd.read_json,
        pd.read_pickle,
    ],
)
def test_pickle_reader(reader):
    # 标记参数化测试用例，测试不同的数据格式读取器
    # 未完全添加注释，需要根据具体情况补充
    # 使用 BytesIO 创建一个内存缓冲区对象，可以在其中存储数据
    with BytesIO() as buffer:
        # 使用 pickle 库将 reader 对象序列化并存储到 buffer 中
        pickle.dump(reader, buffer)
```