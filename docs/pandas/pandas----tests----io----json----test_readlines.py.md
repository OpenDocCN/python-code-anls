# `D:\src\scipysrc\pandas\pandas\tests\io\json\test_readlines.py`

```
from collections.abc import Iterator  # 导入 Iterator 抽象基类，用于声明迭代器类型
from io import StringIO  # 导入 StringIO 类，用于在内存中操作文本数据
from pathlib import Path  # 导入 Path 类，用于操作文件路径

import numpy as np  # 导入 numpy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试

import pandas as pd  # 导入 pandas 库，用于数据分析和处理
from pandas import (  # 从 pandas 中导入 DataFrame 和 read_json 函数
    DataFrame,
    read_json,
)
import pandas._testing as tm  # 导入 pandas 测试模块，用于测试框架的辅助函数

from pandas.io.json._json import JsonReader  # 从 pandas 内部导入 JsonReader 类

pytestmark = pytest.mark.filterwarnings(  # 设置 pytest 的标记，忽略特定警告信息
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)


@pytest.fixture
def lines_json_df():
    df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    return df.to_json(lines=True, orient="records")  # 将 DataFrame 转换为 JSON 字符串，每行一条记录


@pytest.fixture(params=["ujson", "pyarrow"])
def engine(request):
    if request.param == "pyarrow":
        pytest.importorskip("pyarrow.json")  # 如果使用 pyarrow 引擎，则检查是否可以导入 pyarrow.json
    return request.param  # 返回参数化的引擎名称


def test_read_jsonl():
    # GH9180: 测试读取 JSON Lines 格式数据
    result = read_json(StringIO('{"a": 1, "b": 2}\n{"b":2, "a" :1}\n'), lines=True)
    expected = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)


def test_read_jsonl_engine_pyarrow(datapath, engine):
    # 测试使用指定引擎（ujson 或 pyarrow）读取 JSON Lines 格式的文件
    result = read_json(
        datapath("io", "json", "data", "line_delimited.json"),  # JSON Lines 文件路径
        lines=True,
        engine=engine,  # 指定的引擎
    )
    expected = DataFrame({"a": [1, 3, 5], "b": [2, 4, 6]})
    tm.assert_frame_equal(result, expected)


def test_read_datetime(request, engine):
    # GH33787: 测试日期时间数据的读取
    if engine == "pyarrow":
        # GH 48893: 如果使用 pyarrow 引擎，则添加标记以忽略某些情况下的失败
        reason = "Pyarrow only supports a file path as an input and line delimited json"
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    df = DataFrame(
        [([1, 2], ["2020-03-05", "2020-04-08T09:58:49+00:00"], "hector")],
        columns=["accounts", "date", "name"],
    )
    json_line = df.to_json(lines=True, orient="records")

    if engine == "pyarrow":
        result = read_json(StringIO(json_line), engine=engine)
    else:
        result = read_json(StringIO(json_line), engine=engine)
    expected = DataFrame(
        [[1, "2020-03-05", "hector"], [2, "2020-04-08T09:58:49+00:00", "hector"]],
        columns=["accounts", "date", "name"],
    )
    tm.assert_frame_equal(result, expected)


def test_read_jsonl_unicode_chars():
    # GH15132: 测试读取包含非 ASCII Unicode 字符的 JSON Lines 数据
    # \u201d == RIGHT DOUBLE QUOTATION MARK

    # 模拟文件句柄
    json = '{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n'
    json = StringIO(json)
    result = read_json(json, lines=True)
    expected = DataFrame([["foo\u201d", "bar"], ["foo", "bar"]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)

    # 模拟字符串
    json = '{"a": "foo”", "b": "bar"}\n{"a": "foo", "b": "bar"}\n'
    result = read_json(StringIO(json), lines=True)
    expected = DataFrame([["foo\u201d", "bar"], ["foo", "bar"]], columns=["a", "b"])
    tm.assert_frame_equal(result, expected)


def test_to_jsonl():
    # GH9180: 测试将 DataFrame 转换为 JSON Lines 格式
    df = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    result = df.to_json(orient="records", lines=True)
    expected = '{"a":1,"b":2}\n{"a":1,"b":2}\n'
    assert result == expected
    # 创建一个 DataFrame 对象，包含两行数据和两列，数据包含特殊字符 '}' 和 '"'，需要进行转义
    df = DataFrame([["foo}", "bar"], ['foo"', "bar"]], columns=["a", "b"])
    # 将 DataFrame 对象转换为 JSON 格式，每行一条记录，每行末尾带换行符
    result = df.to_json(orient="records", lines=True)
    # 预期的 JSON 字符串，包含转义后的特殊字符
    expected = '{"a":"foo}","b":"bar"}\n{"a":"foo\\"","b":"bar"}\n'
    # 断言转换后的 JSON 字符串与预期的 JSON 字符串相等
    assert result == expected
    # 调用 read_json 函数，传入 StringIO 对象和 lines=True 参数，与原 DataFrame 进行比较
    tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)

    # GH15096: columns 和数据中的转义字符处理
    # 创建另一个 DataFrame 对象，包含两行数据和两列，列名和数据中均包含转义字符 '\\'
    df = DataFrame([["foo\\", "bar"], ['foo"', "bar"]], columns=["a\\", "b"])
    # 将 DataFrame 对象转换为 JSON 格式，每行一条记录，每行末尾带换行符
    result = df.to_json(orient="records", lines=True)
    # 预期的 JSON 字符串，包含转义后的特殊字符 '\\'
    expected = '{"a\\\\":"foo\\\\","b":"bar"}\n{"a\\\\":"foo\\"","b":"bar"}\n'
    # 断言转换后的 JSON 字符串与预期的 JSON 字符串相等
    assert result == expected
    # 调用 read_json 函数，传入 StringIO 对象和 lines=True 参数，与原 DataFrame 进行比较
    tm.assert_frame_equal(read_json(StringIO(result), lines=True), df)
# 测试函数，验证DataFrame转换为JSON格式并计算换行符的数量
def test_to_jsonl_count_new_lines():
    # GH36888：GitHub issue编号
    # 创建一个DataFrame对象，包含两行数据和两列"a"和"b"
    df = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    # 将DataFrame转换为JSON格式，以记录为单位，并计算换行符的数量
    actual_new_lines_count = df.to_json(orient="records", lines=True).count("\n")
    # 期望的换行符数量
    expected_new_lines_count = 2
    # 断言实际换行符数量与期望值相等
    assert actual_new_lines_count == expected_new_lines_count


# 参数化测试函数，测试读取JSON数据时的分块处理
@pytest.mark.parametrize("chunksize", [1, 1.0])
def test_readjson_chunks(request, lines_json_df, chunksize, engine):
    # 基本测试：验证read_json(chunks=True)与read_json(chunks=False)结果相同
    # GH17048：GitHub issue编号，涉及lines=True时的内存使用
    if engine == "pyarrow":
        # GH 48893：GitHub issue编号
        # 如果使用Pyarrow引擎，则标记为xfail，并提供原因
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    # 以未分块方式读取JSON数据
    unchunked = read_json(StringIO(lines_json_df), lines=True)
    # 使用分块方式读取JSON数据，并将结果拼接成一个DataFrame对象
    with read_json(
        StringIO(lines_json_df), lines=True, chunksize=chunksize, engine=engine
    ) as reader:
        chunked = pd.concat(reader)

    # 断言分块读取的结果与未分块读取的结果相等
    tm.assert_frame_equal(chunked, unchunked)


# 测试函数，验证当lines=False时，chunksize参数会引发异常
def test_readjson_chunksize_requires_lines(lines_json_df, engine):
    # 期望的错误消息
    msg = "chunksize can only be passed if lines=True"
    # 使用pytest.raises断言抛出指定异常类型，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        with read_json(
            StringIO(lines_json_df), lines=False, chunksize=2, engine=engine
        ) as _:
            pass


# 测试函数，验证使用chunksize参数读取Series对象时的分块处理
def test_readjson_chunks_series(request, engine):
    if engine == "pyarrow":
        # GH 48893：GitHub issue编号
        # 如果使用Pyarrow引擎，则标记为xfail，并提供原因
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason))

    # 创建一个Series对象
    s = pd.Series({"A": 1, "B": 2})

    # 将Series对象转换为JSON格式，以记录为单位，并创建一个StringIO对象
    strio = StringIO(s.to_json(lines=True, orient="records"))
    # 以未分块方式读取JSON数据并转换为Series对象
    unchunked = read_json(strio, lines=True, typ="Series", engine=engine)

    # 将Series对象转换为JSON格式，以记录为单位，并再次创建StringIO对象
    strio = StringIO(s.to_json(lines=True, orient="records"))
    # 使用分块方式读取JSON数据，并将结果拼接成一个Series对象
    with read_json(
        strio, lines=True, typ="Series", chunksize=1, engine=engine
    ) as reader:
        chunked = pd.concat(reader)

    # 断言分块读取的结果与未分块读取的结果相等
    tm.assert_series_equal(chunked, unchunked)


# 测试函数，验证使用chunksize参数读取JSON数据时的每个分块内容
def test_readjson_each_chunk(request, lines_json_df, engine):
    if engine == "pyarrow":
        # GH 48893：GitHub issue编号
        # 如果使用Pyarrow引擎，则标记为xfail，并提供原因
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    # 其他测试验证read_json(chunksize=True)的最终结果是否正确
    # 该测试验证中间分块的内容
    with read_json(
        StringIO(lines_json_df), lines=True, chunksize=2, engine=engine
    ) as reader:
        # 将所有分块内容存储在列表中
        chunks = list(reader)
    # 断言第一个分块的形状为(2, 2)
    assert chunks[0].shape == (2, 2)
    # 断言第二个分块的形状为(1, 2)
    assert chunks[1].shape == (1, 2)


# 测试函数，验证从文件中使用分块方式读取JSON数据
def test_readjson_chunks_from_file(request, engine):
    ```python`
    # 如果使用的引擎是 "pyarrow"
    if engine == "pyarrow":
        # 设置失败标记的原因
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        # 应用标记以表示预期失败的测试用例
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))
    
    # 确保在测试期间使用的文件 "test.json" 是干净的
    with tm.ensure_clean("test.json") as path:
        # 创建一个 DataFrame 对象
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # 将 DataFrame 对象保存为 JSON 文件，每行一个记录
        df.to_json(path, lines=True, orient="records")
        
        # 使用指定引擎和参数读取 JSON 文件并以 chunksize 为 1 合并为一个 DataFrame
        with read_json(path, lines=True, chunksize=1, engine=engine) as reader:
            chunked = pd.concat(reader)
        
        # 以指定引擎读取整个 JSON 文件并创建一个 DataFrame
        unchunked = read_json(path, lines=True, engine=engine)
        
        # 断言两个 DataFrame 在内容上是否相等
        tm.assert_frame_equal(unchunked, chunked)
# 使用 pytest.mark.parametrize 装饰器为 test_readjson_chunks_closes 函数创建多个测试参数化实例
@pytest.mark.parametrize("chunksize", [None, 1])
def test_readjson_chunks_closes(chunksize):
    # 确保在使用 "test.json" 路径时进行清理
    with tm.ensure_clean("test.json") as path:
        # 创建一个包含两列的 DataFrame
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # 将 DataFrame 以 JSON 格式写入指定路径，每行一个记录，记录格式为字典
        df.to_json(path, lines=True, orient="records")
        # 创建 JsonReader 对象，用于从 JSON 文件读取数据
        reader = JsonReader(
            path,
            orient=None,
            typ="frame",
            dtype=True,
            convert_axes=True,
            convert_dates=True,
            keep_default_dates=True,
            precise_float=False,
            date_unit=None,
            encoding=None,
            lines=True,
            chunksize=chunksize,
            compression=None,
            nrows=None,
        )
        # 使用 reader 对象读取数据
        with reader:
            reader.read()
        # 断言检查 reader 对象的流是否已关闭
        assert (
            reader.handles.handle.closed
        ), f"didn't close stream with chunksize = {chunksize}"


# 使用 pytest.mark.parametrize 装饰器为 test_readjson_invalid_chunksize 函数创建多个测试参数化实例
@pytest.mark.parametrize("chunksize", [0, -1, 2.2, "foo"])
def test_readjson_invalid_chunksize(lines_json_df, chunksize, engine):
    # 期望抛出的错误消息
    msg = r"'chunksize' must be an integer >=1"

    # 使用 pytest.raises 断言检查是否抛出 ValueError 异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        # 调用 read_json 函数，使用 StringIO 对象作为输入流，读取 JSON 数据
        with read_json(
            StringIO(lines_json_df), lines=True, chunksize=chunksize, engine=engine
        ) as _:
            pass


# 使用 pytest.mark.parametrize 装饰器为 test_readjson_chunks_multiple_empty_lines 函数创建多个测试参数化实例
@pytest.mark.parametrize("chunksize", [None, 1, 2])
def test_readjson_chunks_multiple_empty_lines(chunksize):
    # 包含多个空行的 JSON 字符串
    j = """

    {"A":1,"B":4}



    {"A":2,"B":5}







    {"A":3,"B":6}
    """
    # 创建原始 DataFrame
    orig = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    # 使用 read_json 函数读取 JSON 数据，每行一个记录，返回一个迭代器或单个 DataFrame
    test = read_json(StringIO(j), lines=True, chunksize=chunksize)
    # 如果指定了 chunksize，则进行拼接操作
    if chunksize is not None:
        with test:
            test = pd.concat(test)
    # 使用 tm.assert_frame_equal 断言检查原始 DataFrame 和读取的 DataFrame 是否相等
    tm.assert_frame_equal(orig, test, obj=f"chunksize: {chunksize}")


# test_readjson_unicode 函数测试读取包含 Unicode 字符的 JSON 数据
def test_readjson_unicode(request, monkeypatch, engine):
    # 如果使用 pyarrow 引擎，则标记测试为预期失败
    if engine == "pyarrow":
        # GH 48893
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    # 确保在使用 "test.json" 路径时进行清理
    with tm.ensure_clean("test.json") as path:
        # 用 monkeypatch 修改 locale.getpreferredencoding 函数返回值
        monkeypatch.setattr("locale.getpreferredencoding", lambda do_setlocale: "cp949")
        # 使用 UTF-8 编码打开文件，并写入包含特殊 Unicode 字符的 JSON 数据
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"£©µÀÆÖÞßéöÿ":["АБВГДабвгд가"]}')

        # 调用 read_json 函数读取 JSON 数据，返回 DataFrame
        result = read_json(path, engine=engine)
        # 创建预期的 DataFrame
        expected = DataFrame({"£©µÀÆÖÞßéöÿ": ["АБВГДабвгд가"]})
        # 使用 tm.assert_frame_equal 断言检查读取结果与预期结果是否相等
        tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器为 test_readjson_nrows 函数创建多个测试参数化实例
@pytest.mark.parametrize("nrows", [1, 2])
def test_readjson_nrows(nrows, engine):
    # GH 33916
    # 测试使用 nrows 参数从行格式 JSON 中读取 Series
    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""
    # 调用 read_json 函数读取 JSON 数据，每行一个记录，返回 DataFrame
    result = read_json(StringIO(jsonl), lines=True, nrows=nrows)
    # 创建预期的 DataFrame，根据 nrows 参数进行截取
    expected = DataFrame({"a": [1, 3, 5, 7], "b": [2, 4, 6, 8]}).iloc[:nrows]
    # 使用 tm.assert_frame_equal 断言检查读取结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)
# 测试读取行格式 JSON 到 Series，使用 nrows 和 chunksize 参数
def test_readjson_nrows_chunks(request, nrows, chunksize, engine):
    if engine == "pyarrow":
        # 如果使用 pyarrow 引擎，标记为预期失败，并说明原因
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    # 定义一个包含多行 JSON 格式的字符串
    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""

    if engine != "pyarrow":
        # 使用 read_json 函数读取 JSON 字符串，设置 lines=True，以及 nrows 和 chunksize 参数
        with read_json(
            StringIO(jsonl), lines=True, nrows=nrows, chunksize=chunksize, engine=engine
        ) as reader:
            # 将读取的结果通过 concat 方法合并为一个 DataFrame
            chunked = pd.concat(reader)
    else:
        # 使用 read_json 函数读取 JSON 字符串（仅当 engine 是 pyarrow 时）
        with read_json(
            jsonl, lines=True, nrows=nrows, chunksize=chunksize, engine=engine
        ) as reader:
            # 将读取的结果通过 concat 方法合并为一个 DataFrame
            chunked = pd.concat(reader)

    # 构造预期结果的 DataFrame
    expected = DataFrame({"a": [1, 3, 5, 7], "b": [2, 4, 6, 8]}).iloc[:nrows]
    # 使用 assert_frame_equal 方法比较 chunked 和 expected，确保它们相等
    tm.assert_frame_equal(chunked, expected)


# 测试当未设置 lines=True 时，nrows 参数是否会引发 ValueError
def test_readjson_nrows_requires_lines(engine):
    # 定义一个包含多行 JSON 格式的字符串
    jsonl = """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}"""
    # 设置期望的错误信息
    msg = "nrows can only be passed if lines=True"
    # 使用 pytest.raises 检查是否会引发 ValueError，且错误信息匹配 msg
    with pytest.raises(ValueError, match=msg):
        read_json(jsonl, lines=False, nrows=2, engine=engine)


# 测试从文件 URL 读取行格式 JSON，使用 chunksize 参数
def test_readjson_lines_chunks_fileurl(request, datapath, engine):
    if engine == "pyarrow":
        # 如果使用 pyarrow 引擎，标记为预期失败，并说明原因
        reason = (
            "Pyarrow only supports a file path as an input and line delimited json"
            "and doesn't support chunksize parameter."
        )
        request.applymarker(pytest.mark.xfail(reason=reason, raises=ValueError))

    # 期望得到的 DataFrame 列表
    df_list_expected = [
        DataFrame([[1, 2]], columns=["a", "b"], index=[0]),
        DataFrame([[3, 4]], columns=["a", "b"], index=[1]),
        DataFrame([[5, 6]], columns=["a", "b"], index=[2]),
    ]
    # 获取测试数据文件的路径
    os_path = datapath("io", "json", "data", "line_delimited.json")
    # 将路径转换为文件 URL
    file_url = Path(os_path).as_uri()
    # 使用 read_json 函数读取文件 URL 中的 JSON 数据，设置 lines=True 和 chunksize 参数
    with read_json(file_url, lines=True, chunksize=1, engine=engine) as url_reader:
        # 遍历读取的结果
        for index, chuck in enumerate(url_reader):
            # 使用 assert_frame_equal 方法比较 chuck 和 df_list_expected[index]，确保它们相等
            tm.assert_frame_equal(chuck, df_list_expected[index])


# 测试 chunksize 参数是否逐步增加
def test_chunksize_is_incremental():
    # 生成包含大量重复内容的 JSON 数据
    jsonl = (
        """{"a": 1, "b": 2}
        {"a": 3, "b": 4}
        {"a": 5, "b": 6}
        {"a": 7, "b": 8}\n"""
        * 1000
    )
    # 定义一个名为 MyReader 的类，用于封装读取操作
    class MyReader:
        # 初始化方法，接收 contents 参数并设置初始读取计数为 0
        def __init__(self, contents) -> None:
            self.read_count = 0
            # 使用 StringIO 将 contents 转换为可读取的字符串流
            self.stringio = StringIO(contents)

        # 读取方法，增加读取计数并返回通过 stringio 读取的内容
        def read(self, *args):
            self.read_count += 1
            return self.stringio.read(*args)

        # 返回一个迭代器，使 MyReader 对象可迭代
        def __iter__(self) -> Iterator:
            self.read_count += 1
            return iter(self.stringio)

    # 创建一个 MyReader 的实例，用 jsonl 内容初始化
    reader = MyReader(jsonl)
    # 断言，确保通过 read_json 函数使用该 reader 对象读取的行数大于 1
    assert len(list(read_json(reader, lines=True, chunksize=100))) > 1
    # 断言，确保 reader 对象的读取计数大于 10
    assert reader.read_count > 10
@pytest.mark.parametrize("orient_", ["split", "index", "table"])
# 使用 pytest 的 parametrize 装饰器，指定 orient_ 参数的多个测试用例
def test_to_json_append_orient(orient_):
    # GH 35849
    # 测试当 orient 不为 'records' 时是否抛出 ValueError 异常
    df = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    # 定义错误消息的正则表达式模式
    msg = (
        r"mode='a' \(append\) is only supported when "
        "lines is True and orient is 'records'"
    )
    with pytest.raises(ValueError, match=msg):
        # 调用 DataFrame 的 to_json 方法，测试是否会抛出异常
        df.to_json(mode="a", orient=orient_)


def test_to_json_append_lines():
    # GH 35849
    # 测试当 lines 不为 True 时是否抛出 ValueError 异常
    df = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    # 定义错误消息的正则表达式模式
    msg = (
        r"mode='a' \(append\) is only supported when "
        "lines is True and orient is 'records'"
    )
    with pytest.raises(ValueError, match=msg):
        # 调用 DataFrame 的 to_json 方法，测试是否会抛出异常
        df.to_json(mode="a", lines=False, orient="records")


@pytest.mark.parametrize("mode_", ["r", "x"])
# 使用 pytest 的 parametrize 装饰器，指定 mode_ 参数的多个测试用例
def test_to_json_append_mode(mode_):
    # GH 35849
    # 测试当 mode 不为 'w' 或 'a' 时是否抛出 ValueError 异常
    df = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    # 定义错误消息，使用 f-string 表达式动态生成
    msg = (
        f"mode={mode_} is not a valid option."
        "Only 'w' and 'a' are currently supported."
    )
    with pytest.raises(ValueError, match=msg):
        # 调用 DataFrame 的 to_json 方法，测试是否会抛出异常
        df.to_json(mode=mode_, lines=False, orient="records")


def test_to_json_append_output_consistent_columns():
    # GH 35849
    # 测试保存的 JSON 文件读取结果是否符合预期，测试列相同但行新增
    df1 = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df2 = DataFrame({"col1": [3, 4], "col2": ["c", "d"]})

    expected = DataFrame({"col1": [1, 2, 3, 4], "col2": ["a", "b", "c", "d"]})
    with tm.ensure_clean("test.json") as path:
        # 将两个 DataFrame 以 'records' 方式追加写入同一个文件
        df1.to_json(path, lines=True, orient="records")
        df2.to_json(path, mode="a", lines=True, orient="records")

        # 读取文件，并比较结果是否符合预期
        result = read_json(path, lines=True)
        tm.assert_frame_equal(result, expected)


def test_to_json_append_output_inconsistent_columns():
    # GH 35849
    # 测试保存的 JSON 文件读取结果是否符合预期，测试列不完全相同的情况下新增数据
    df1 = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df3 = DataFrame({"col2": ["e", "f"], "col3": ["!", "#"]})

    expected = DataFrame(
        {
            "col1": [1, 2, None, None],
            "col2": ["a", "b", "e", "f"],
            "col3": [np.nan, np.nan, "!", "#"],
        }
    )
    with tm.ensure_clean("test.json") as path:
        # 将两个 DataFrame 以 'records' 方式追加写入同一个文件
        df1.to_json(path, mode="a", lines=True, orient="records")
        df3.to_json(path, mode="a", lines=True, orient="records")

        # 读取文件，并比较结果是否符合预期
        result = read_json(path, lines=True)
        tm.assert_frame_equal(result, expected)


def test_to_json_append_output_different_columns():
    # GH 35849
    # 测试保存的 JSON 文件读取结果是否符合预期，测试列完全不同的情况下新增数据
    # 创建四个 DataFrame 对象，每个对象包含不同的列和数据
    df1 = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df2 = DataFrame({"col1": [3, 4], "col2": ["c", "d"]})
    df3 = DataFrame({"col2": ["e", "f"], "col3": ["!", "#"]})
    df4 = DataFrame({"col4": [True, False]})

    # 创建期望的 DataFrame，包含所有可能的列和值，其中部分为 NaN，并将 col4 转换为 float 类型
    expected = DataFrame(
        {
            "col1": [1, 2, 3, 4, None, None, None, None],
            "col2": ["a", "b", "c", "d", "e", "f", np.nan, np.nan],
            "col3": [np.nan, np.nan, np.nan, np.nan, "!", "#", np.nan, np.nan],
            "col4": [None, None, None, None, None, None, True, False],
        }
    ).astype({"col4": "float"})

    # 使用 tm.ensure_clean 上下文管理器确保文件 "test.json" 存在并在退出时删除
    with tm.ensure_clean("test.json") as path:
        # 将四个 DataFrame 以追加模式写入同一个文件 "test.json"，每个 DataFrame 作为一行记录
        df1.to_json(path, mode="a", lines=True, orient="records")
        df2.to_json(path, mode="a", lines=True, orient="records")
        df3.to_json(path, mode="a", lines=True, orient="records")
        df4.to_json(path, mode="a", lines=True, orient="records")

        # 读取文件 "test.json" 中的数据为 DataFrame 对象 result
        result = read_json(path, lines=True)
        
        # 使用 tm.assert_frame_equal 方法比较 result 和预期的 expected DataFrame
        tm.assert_frame_equal(result, expected)
def test_to_json_append_output_different_columns_reordered():
    # GH 35849
    # Testing that resulting output reads in as expected.
    # Testing specific result column order.
    
    # 创建四个不同的数据框，每个数据框有特定的列和数据
    df1 = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    df2 = DataFrame({"col1": [3, 4], "col2": ["c", "d"]})
    df3 = DataFrame({"col2": ["e", "f"], "col3": ["!", "#"]})
    df4 = DataFrame({"col4": [True, False]})
    
    # 创建预期的数据框，按照特定的列顺序组织数据，并将 "col4" 的类型转换为浮点型
    expected = DataFrame(
        {
            "col4": [True, False, None, None, None, None, None, None],
            "col2": [np.nan, np.nan, "e", "f", "c", "d", "a", "b"],
            "col3": [np.nan, np.nan, "!", "#", np.nan, np.nan, np.nan, np.nan],
            "col1": [None, None, None, None, 3, 4, 1, 2],
        }
    ).astype({"col4": "float"})
    
    # 在临时文件中保存数据框，以便后续读取和比较结果
    with tm.ensure_clean("test.json") as path:
        # 将数据框以追加模式写入同一文件
        df4.to_json(path, mode="a", lines=True, orient="records")
        df3.to_json(path, mode="a", lines=True, orient="records")
        df2.to_json(path, mode="a", lines=True, orient="records")
        df1.to_json(path, mode="a", lines=True, orient="records")
        
        # 读取保存的 JSON 文件
        result = read_json(path, lines=True)
        # 断言读取的结果与预期结果相等
        tm.assert_frame_equal(result, expected)
```