# `D:\src\scipysrc\pandas\pandas\tests\io\parser\conftest.py`

```
from __future__ import annotations
# 引入使用未来版本的 annotations

import os
# 导入操作系统相关模块

import pytest
# 导入 pytest 测试框架

from pandas.compat._optional import VERSIONS
# 从 pandas 兼容模块中导入 VERSIONS

from pandas import (
    read_csv,
    read_table,
)
# 从 pandas 中导入 read_csv 和 read_table 函数

import pandas._testing as tm
# 导入 pandas 测试模块作为 tm 别名

class BaseParser:
    engine: str | None = None
    # 引擎类型，默认为 None
    low_memory = True
    # 低内存模式，默认为 True
    float_precision_choices: list[str | None] = []
    # 浮点精度选项列表，默认为空列表

    def update_kwargs(self, kwargs):
        # 更新关键字参数方法
        kwargs = kwargs.copy()
        # 复制参数字典
        kwargs.update({"engine": self.engine, "low_memory": self.low_memory})
        # 更新引擎类型和低内存模式到参数字典中

        return kwargs
        # 返回更新后的参数字典

    def read_csv(self, *args, **kwargs):
        # CSV 文件读取方法
        kwargs = self.update_kwargs(kwargs)
        # 更新关键字参数
        return read_csv(*args, **kwargs)
        # 调用 pandas 的 read_csv 函数并返回结果

    def read_csv_check_warnings(
        self,
        warn_type: type[Warning],
        warn_msg: str,
        *args,
        raise_on_extra_warnings=True,
        check_stacklevel: bool = True,
        **kwargs,
    ):
        # 读取 CSV 文件并检查警告方法
        # 在这里检查堆栈级别而不是在测试中进行，因为这里调用了 read_csv 并且警告应该指向这里。
        kwargs = self.update_kwargs(kwargs)
        # 更新关键字参数
        with tm.assert_produces_warning(
            warn_type,
            match=warn_msg,
            raise_on_extra_warnings=raise_on_extra_warnings,
            check_stacklevel=check_stacklevel,
        ):
            return read_csv(*args, **kwargs)
            # 使用 read_csv 函数读取 CSV 文件，并根据参数设置检查警告

    def read_table(self, *args, **kwargs):
        # 表格文件读取方法
        kwargs = self.update_kwargs(kwargs)
        # 更新关键字参数
        return read_table(*args, **kwargs)
        # 调用 pandas 的 read_table 函数并返回结果

    def read_table_check_warnings(
        self,
        warn_type: type[Warning],
        warn_msg: str,
        *args,
        raise_on_extra_warnings=True,
        **kwargs,
    ):
        # 读取表格文件并检查警告方法
        # 在这里检查堆栈级别而不是在测试中进行，因为这里调用了 read_table 并且警告应该指向这里。
        kwargs = self.update_kwargs(kwargs)
        # 更新关键字参数
        with tm.assert_produces_warning(
            warn_type, match=warn_msg, raise_on_extra_warnings=raise_on_extra_warnings
        ):
            return read_table(*args, **kwargs)
            # 使用 read_table 函数读取表格文件，并根据参数设置检查警告


class CParser(BaseParser):
    engine = "c"
    # 引擎类型为 "c"
    float_precision_choices = [None, "high", "round_trip"]
    # 浮点精度选项列表包括 None、"high"、"round_trip"


class CParserHighMemory(CParser):
    low_memory = False
    # 高内存模式，低内存模式设置为 False


class CParserLowMemory(CParser):
    low_memory = True
    # 低内存模式，低内存模式设置为 True


class PythonParser(BaseParser):
    engine = "python"
    # 引擎类型为 "python"
    float_precision_choices = [None]
    # 浮点精度选项列表包括 None


class PyArrowParser(BaseParser):
    engine = "pyarrow"
    # 引擎类型为 "pyarrow"
    float_precision_choices = [None]
    # 浮点精度选项列表包括 None


@pytest.fixture
def csv_dir_path(datapath):
    """
    The directory path to the data files needed for parser tests.
    """
    return datapath("io", "parser", "data")
    # 返回用于解析器测试所需数据文件的目录路径


@pytest.fixture
def csv1(datapath):
    """
    The path to the data file "test1.csv" needed for parser tests.
    """
    return os.path.join(datapath("io", "data", "csv"), "test1.csv")
    # 返回用于解析器测试所需的数据文件 "test1.csv" 的路径


_cParserHighMemory = CParserHighMemory
# _cParserHighMemory 别名指向 CParserHighMemory 类
_cParserLowMemory = CParserLowMemory
# _cParserLowMemory 别名指向 CParserLowMemory 类
_pythonParser = PythonParser
# _pythonParser 别名指向 PythonParser 类
_pyarrowParser = PyArrowParser
# _pyarrowParser 别名指向 PyArrowParser 类

_py_parsers_only = [_pythonParser]
# _py_parsers_only 列表包含 _pythonParser 变量
_c_parsers_only = [_cParserHighMemory, _cParserLowMemory]
_pyarrow_parsers_only = [pytest.param(_pyarrowParser, marks=pytest.mark.single_cpu)]

_all_parsers = [*_c_parsers_only, *_py_parsers_only, *_pyarrow_parsers_only]

_py_parser_ids = ["python"]
_c_parser_ids = ["c_high", "c_low"]
_pyarrow_parsers_ids = ["pyarrow"]

_all_parser_ids = [*_c_parser_ids, *_py_parser_ids, *_pyarrow_parsers_ids]


@pytest.fixture(params=_all_parsers, ids=_all_parser_ids)
def all_parsers(request):
    """
    Fixture to provide instances of all CSV parsers.

    Selects a parser instance based on the request parameter and initializes it.
    If the parser uses the 'pyarrow' engine, it ensures pyarrow is imported and sets CPU count to 1.

    Returns:
        An instance of the selected parser.
    """
    parser = request.param()
    if parser.engine == "pyarrow":
        pytest.importorskip("pyarrow", VERSIONS["pyarrow"])
        # Try finding a way to disable threads all together
        # for more stable CI runs
        import pyarrow

        pyarrow.set_cpu_count(1)
    return parser


@pytest.fixture(params=_c_parsers_only, ids=_c_parser_ids)
def c_parser_only(request):
    """
    Fixture to provide instances of CSV parsers using the C engine.

    Returns:
        An instance of a CSV parser using the C engine.
    """
    return request.param()


@pytest.fixture(params=_py_parsers_only, ids=_py_parser_ids)
def python_parser_only(request):
    """
    Fixture to provide instances of CSV parsers using the Python engine.

    Returns:
        An instance of a CSV parser using the Python engine.
    """
    return request.param()


@pytest.fixture(params=_pyarrow_parsers_only, ids=_pyarrow_parsers_ids)
def pyarrow_parser_only(request):
    """
    Fixture to provide instances of CSV parsers using the Pyarrow engine.

    Returns:
        An instance of a CSV parser using the Pyarrow engine.
    """
    return request.param()


def _get_all_parser_float_precision_combinations():
    """
    Generates all possible combinations of CSV parsers and float precision settings.

    Returns:
        Dictionary with 'params' containing combinations and 'ids' for corresponding identifiers.
    """
    params = []
    ids = []
    for parser, parser_id in zip(_all_parsers, _all_parser_ids):
        if hasattr(parser, "values"):
            # Wrapped in pytest.param, get the actual parser back
            parser = parser.values[0]
        for precision in parser.float_precision_choices:
            # Re-wrap in pytest.param for pyarrow
            mark = pytest.mark.single_cpu if parser.engine == "pyarrow" else ()
            param = pytest.param((parser(), precision), marks=mark)
            params.append(param)
            ids.append(f"{parser_id}-{precision}")

    return {"params": params, "ids": ids}


@pytest.fixture(
    params=_get_all_parser_float_precision_combinations()["params"],
    ids=_get_all_parser_float_precision_combinations()["ids"],
)
def all_parsers_all_precisions(request):
    """
    Fixture to provide instances of all CSV parsers with all allowable float precision combinations.

    Returns:
        A tuple containing an instance of the parser and a float precision setting.
    """
    return request.param


_utf_values = [8, 16, 32]

_encoding_seps = ["", "-", "_"]
_encoding_prefixes = ["utf", "UTF"]

_encoding_fmts = [
    f"{prefix}{sep}{{0}}" for sep in _encoding_seps for prefix in _encoding_prefixes
]


@pytest.fixture(params=_utf_values)
def utf_value(request):
    """
    Fixture to provide all possible integer values for UTF encoding.

    Returns:
        An integer representing a UTF encoding value.
    """
    return request.param


@pytest.fixture(params=_encoding_fmts)
def encoding_fmt(request):
    """
    Fixture to provide all possible string formats for UTF encoding.

    Returns:
        A string representing a format for UTF encoding.
    """
    return request.param
    # 为所有可能的 UTF 编码字符串格式提供的测试数据准备
    """
    这段代码定义了一个用于测试的夹具（Fixture），用于提供所有可能的 UTF 编码字符串格式作为参数。
    在测试过程中，通过 request.param 可以获取到每一个参数值，用于测试不同的编码格式。
    """
    return request.param
# 定义 pytest 的参数化 fixture，用于测试各种数值格式的解析
@pytest.fixture(
    params=[
        ("-1,0", -1.0),            # 字符串和期望的浮点数结果
        ("-1,2e0", -1.2),          # 字符串和期望的浮点数结果
        ("-1e0", -1.0),            # 字符串和期望的浮点数结果
        ("+1e0", 1.0),             # 字符串和期望的浮点数结果
        ("+1e+0", 1.0),            # 字符串和期望的浮点数结果
        ("+1e-1", 0.1),            # 字符串和期望的浮点数结果
        ("+,1e1", 1.0),            # 字符串和期望的浮点数结果
        ("+1,e0", 1.0),            # 字符串和期望的浮点数结果
        ("-,1e1", -1.0),           # 字符串和期望的浮点数结果
        ("-1,e0", -1.0),           # 字符串和期望的浮点数结果
        ("0,1", 0.1),              # 字符串和期望的浮点数结果
        ("1,", 1.0),               # 字符串和期望的浮点数结果
        (",1", 0.1),               # 字符串和期望的浮点数结果
        ("-,1", -0.1),             # 字符串和期望的浮点数结果
        ("1_,", 1.0),              # 字符串和期望的浮点数结果
        ("1_234,56", 1234.56),     # 字符串和期望的浮点数结果
        ("1_234,56e0", 1234.56),   # 字符串和期望的浮点数结果
        # 以下是不应被解析为浮点数的负面案例
        ("_", "_"),                # 不应解析为浮点数的字符串
        ("-_", "-_"),              # 不应解析为浮点数的字符串
        ("-_1", "-_1"),            # 不应解析为浮点数的字符串
        ("-_1e0", "-_1e0"),        # 不应解析为浮点数的字符串
        ("_1", "_1"),              # 不应解析为浮点数的字符串
        ("_1,", "_1,"),            # 不应解析为浮点数的字符串
        ("_1,_", "_1,_"),          # 不应解析为浮点数的字符串
        ("_1e0", "_1e0"),          # 不应解析为浮点数的字符串
        ("1,2e_1", "1,2e_1"),      # 不应解析为浮点数的字符串
        ("1,2e1_0", "1,2e1_0"),    # 不应解析为浮点数的字符串
        ("1,_2", "1,_2"),          # 不应解析为浮点数的字符串
        (",1__2", ",1__2"),        # 不应解析为浮点数的字符串
        (",1e", ",1e"),            # 不应解析为浮点数的字符串
        ("-,1e", "-,1e"),          # 不应解析为浮点数的字符串
        ("1_000,000_000", "1_000,000_000"),  # 不应解析为浮点数的字符串
        ("1,e1_2", "1,e1_2"),      # 不应解析为浮点数的字符串
        ("e11,2", "e11,2"),        # 不应解析为浮点数的字符串
        ("1e11,2", "1e11,2"),      # 不应解析为浮点数的字符串
        ("1,2,2", "1,2,2"),        # 不应解析为浮点数的字符串
        ("1,2_1", "1,2_1"),        # 不应解析为浮点数的字符串
        ("1,2e-10e1", "1,2e-10e1"),# 不应解析为浮点数的字符串
        ("--1,2", "--1,2"),        # 不应解析为浮点数的字符串
        ("1a_2,1", "1a_2,1"),      # 不应解析为浮点数的字符串
        ("1,2E-1", 0.12),          # 字符串和期望的浮点数结果
        ("1,2E1", 12.0),           # 字符串和期望的浮点数结果
    ]
)
def numeric_decimal(request):
    """
    Fixture for all numeric formats which should get recognized. The first entry
    represents the value to read while the second represents the expected result.
    """
    return request.param


# 定义 pytest fixture，如果使用 pyarrow 引擎，标记为 xfail
@pytest.fixture
def pyarrow_xfail(request):
    """
    Fixture that xfails a test if the engine is pyarrow.

    Use if failure is due to unsupported keywords or inconsistent results.
    """
    if "all_parsers" in request.fixturenames:
        parser = request.getfixturevalue("all_parsers")
    elif "all_parsers_all_precisions" in request.fixturenames:
        # 返回值是元组 (引擎, 精度)
        parser = request.getfixturevalue("all_parsers_all_precisions")[0]
    else:
        return
    if parser.engine == "pyarrow":
        mark = pytest.mark.xfail(reason="pyarrow doesn't support this.")
        request.applymarker(mark)


# 定义 pytest fixture，如果使用 pyarrow 引擎，跳过测试
@pytest.fixture
def pyarrow_skip(request):
    """
    Fixture that skips a test if the engine is pyarrow.

    Use if failure is due to parsing failure from pyarrow.csv.read_csv
    """
    if "all_parsers" in request.fixturenames:
        parser = request.getfixturevalue("all_parsers")
    elif "all_parsers_all_precisions" in request.fixturenames:
        # 返回值是元组 (引擎, 精度)
        parser = request.getfixturevalue("all_parsers_all_precisions")[0]
    else:
        return
    if parser.engine == "pyarrow":
        pytest.skip(reason="https://github.com/apache/arrow/issues/38676")
```