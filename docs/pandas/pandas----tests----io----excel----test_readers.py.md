# `D:\src\scipysrc\pandas\pandas\tests\io\excel\test_readers.py`

```
from __future__ import annotations
# 引入将来版本的注解特性，支持从当前版本向后兼容的特性

from datetime import (
    datetime,
    time,
)
# 从datetime模块中导入datetime和time类

from functools import partial
# 导入functools模块中的partial函数，用于部分应用一个函数

from io import BytesIO
# 从io模块导入BytesIO类，用于在内存中创建二进制数据流

import os
# 导入os模块，提供了对操作系统的接口

from pathlib import Path
# 从pathlib模块导入Path类，用于处理文件路径

import platform
# 导入platform模块，用于访问平台相关属性和功能

import re
# 导入re模块，提供了对正则表达式的支持

from urllib.error import URLError
# 从urllib.error模块导入URLError异常类，用于处理URL相关的错误

import uuid
# 导入uuid模块，用于生成UUID

from zipfile import BadZipFile
# 从zipfile模块导入BadZipFile异常类，用于处理损坏的ZIP文件

import numpy as np
# 导入numpy库，并将其命名为np，用于支持大型、多维数组和矩阵运算

import pytest
# 导入pytest库，用于编写和运行测试用例

from pandas._config import using_pyarrow_string_dtype
# 从pandas._config模块导入using_pyarrow_string_dtype函数

import pandas.util._test_decorators as td
# 导入pandas.util._test_decorators模块，并将其命名为td，用于测试装饰器

import pandas as pd
# 导入pandas库，并将其命名为pd，用于数据分析和处理

from pandas import (
    DataFrame,
    Index,
    MultiIndex,
    Series,
    read_csv,
)
# 从pandas库导入DataFrame、Index、MultiIndex、Series和read_csv函数

import pandas._testing as tm
# 导入pandas._testing模块，并将其命名为tm，用于测试相关的辅助功能

from pandas.core.arrays import (
    ArrowStringArray,
    StringArray,
)
# 从pandas.core.arrays模块导入ArrowStringArray和StringArray类，用于处理特殊类型的数据

read_ext_params = [".xls", ".xlsx", ".xlsm", ".xlsb", ".ods"]
# 定义支持的Excel文件扩展名列表

engine_params = [
    # 添加要测试的引擎
    # 安装defusedxml后，会触发对xlrd和openpyxl的弃用警告，因此在此处处理
    pytest.param(
        "xlrd",
        marks=[
            td.skip_if_no("xlrd"),
        ],
    ),
    pytest.param(
        "openpyxl",
        marks=[
            td.skip_if_no("openpyxl"),
        ],
    ),
    pytest.param(
        None,
        marks=[
            td.skip_if_no("xlrd"),
        ],
    ),
    pytest.param("pyxlsb", marks=td.skip_if_no("pyxlsb")),
    pytest.param("odf", marks=td.skip_if_no("odf")),
    pytest.param("calamine", marks=td.skip_if_no("python_calamine")),
]
# 定义用于测试的Excel读取引擎参数列表，包含各种标记用于测试装饰

def _is_valid_engine_ext_pair(engine, read_ext: str) -> bool:
    """
    过滤掉无效的（engine, ext）组合，而不是跳过，因为这会产生500多个pytest.skips。
    """
    engine = engine.values[0]
    if engine == "openpyxl" and read_ext == ".xls":
        return False
    if engine == "odf" and read_ext != ".ods":
        return False
    if read_ext == ".ods" and engine not in {"odf", "calamine"}:
        return False
    if engine == "pyxlsb" and read_ext != ".xlsb":
        return False
    if read_ext == ".xlsb" and engine not in {"pyxlsb", "calamine"}:
        return False
    if engine == "xlrd" and read_ext != ".xls":
        return False
    return True
# 定义函数_is_valid_engine_ext_pair，用于验证引擎和文件扩展名是否有效

def _transfer_marks(engine, read_ext):
    """
    engine是一个带有一些标记的pytest.param对象，read_ext只是一个字符串。
    我们需要生成一个新的pytest.param对象，继承标记。
    """
    values = engine.values + (read_ext,)
    new_param = pytest.param(values, marks=engine.marks)
    return new_param
# 定义函数_transfer_marks，用于转移标记以创建新的pytest.param对象

@pytest.fixture(
    params=[
        _transfer_marks(eng, ext)
        for eng in engine_params
        for ext in read_ext_params
        if _is_valid_engine_ext_pair(eng, ext)
    ],
    ids=str,
)
def engine_and_read_ext(request):
    """
    用于Excel读取引擎和读取扩展名的夹具，仅包含有效的配对。
    """
    return request.param
# 定义engine_and_read_ext夹具，用于生成有效的Excel读取引擎和读取扩展名参数对

@pytest.fixture
def engine(engine_and_read_ext):
    engine, read_ext = engine_and_read_ext
    return engine
# 定义engine夹具，返回从engine_and_read_ext夹具中提取的引擎参数

@pytest.fixture
def read_ext(engine_and_read_ext):
    engine, read_ext = engine_and_read_ext
    return read_ext
# 定义read_ext夹具，返回从engine_and_read_ext夹具中提取的读取扩展名参数
# 创建一个临时文件路径，文件名包含唯一的 UUID 和指定的文件扩展名
def tmp_excel(read_ext, tmp_path):
    tmp = tmp_path / f"{uuid.uuid4()}{read_ext}"
    # 创建空文件
    tmp.touch()
    # 返回临时文件的路径字符串表示形式
    return str(tmp)


@pytest.fixture
def df_ref(datapath):
    """
    Obtain the reference data from read_csv with the Python engine.
    """
    # 获取数据文件路径
    filepath = datapath("io", "data", "csv", "test1.csv")
    # 使用 Python 引擎读取 CSV 文件，并返回数据帧
    df_ref = read_csv(filepath, index_col=0, parse_dates=True, engine="python")
    return df_ref


def get_exp_unit(read_ext: str, engine: str | None) -> str:
    # 默认时间单位为 "us"
    unit = "us"
    # 根据文件扩展名和引擎类型决定时间单位
    if (read_ext == ".ods") ^ (engine == "calamine"):
        unit = "s"
    return unit


def adjust_expected(expected: DataFrame, read_ext: str, engine: str | None) -> None:
    # 清除索引名
    expected.index.name = None
    # 获取期望的时间单位
    unit = get_exp_unit(read_ext, engine)
    # 错误: "Index" 没有 "as_unit" 属性
    # 根据指定的时间单位调整索引
    expected.index = expected.index.as_unit(unit)  # type: ignore[attr-defined]


def xfail_datetimes_with_pyxlsb(engine, request):
    # 如果引擎为 "pyxlsb"，标记测试用例为预期失败，原因是不支持包含日期时间的表格
    if engine == "pyxlsb":
        request.applymarker(
            pytest.mark.xfail(
                reason="Sheets containing datetimes not supported by pyxlsb"
            )
        )


class TestReaders:
    @pytest.mark.parametrize("col", [[True, None, False], [True], [True, False]])
    def test_read_excel_type_check(self, col, datapath):
        # GH 58159
        # 创建包含布尔列的数据帧，指定列类型为布尔型
        df = DataFrame({"bool_column": col}, dtype="boolean")
        # 获取测试文件路径
        f_path = datapath("io", "data", "excel", "test_boolean_types.xlsx")

        # 将数据帧写入 Excel 文件
        df.to_excel(f_path, index=False)
        # 使用 openpyxl 引擎读取 Excel 文件，并指定布尔列的数据类型为布尔型
        df2 = pd.read_excel(f_path, dtype={"bool_column": "boolean"}, engine="openpyxl")
        # 断言读取的数据帧与写入的数据帧相等
        tm.assert_frame_equal(df, df2)

    def test_pass_none_type(self, datapath):
        # GH 58159
        # 获取测试文件路径
        f_path = datapath("io", "data", "excel", "test_none_type.xlsx")

        # 使用 pd.ExcelFile 上下文管理器读取 Excel 文件
        with pd.ExcelFile(f_path) as excel:
            # 使用 openpyxl 引擎读取特定 sheet 的数据，并指定处理缺失值及数据类型
            parsed = pd.read_excel(
                excel,
                sheet_name="Sheet1",
                keep_default_na=True,
                na_values=["nan", "None", "abcd"],
                dtype="boolean",
                engine="openpyxl",
            )
        # 创建期望的数据帧，包含布尔类型数据
        expected = DataFrame(
            {"Test": [True, None, False, None, False, None, True]},
            dtype="boolean",
        )

        # 断言读取的数据帧与期望的数据帧相等
        tm.assert_frame_equal(parsed, expected)

    @pytest.fixture(autouse=True)
    def cd_and_set_engine(self, engine, datapath, monkeypatch):
        """
        Change directory and set engine for read_excel calls.
        """
        # 创建一个偏函数，用于在 monkeypatch 中设置 pd.read_excel 的引擎参数
        func = partial(pd.read_excel, engine=engine)
        # 切换到指定路径，用于读取 Excel 文件
        monkeypatch.chdir(datapath("io", "data", "excel"))
        # 替换 pd.read_excel 函数，使其默认使用指定的引擎
        monkeypatch.setattr(pd, "read_excel", func)
    # 测试读取引擎的使用情况，使用指定的读取扩展名、引擎和 monkeypatch 对象
    def test_engine_used(self, read_ext, engine, monkeypatch):
        # GH 38884
        # 定义一个模拟的解析器函数，返回指定的引擎对象
        def parser(self, *args, **kwargs):
            return self.engine

        # 使用 monkeypatch 修改 pd.ExcelFile 类的 parse 方法为上面定义的 parser 函数
        monkeypatch.setattr(pd.ExcelFile, "parse", parser)

        # 预期的默认引擎映射表
        expected_defaults = {
            "xlsx": "openpyxl",
            "xlsm": "openpyxl",
            "xlsb": "pyxlsb",
            "xls": "xlrd",
            "ods": "odf",
        }

        # 打开测试文件（以二进制模式），然后使用 pd.read_excel 读取文件内容
        with open("test1" + read_ext, "rb") as f:
            result = pd.read_excel(f)

        # 确定预期的引擎
        if engine is not None:
            expected = engine
        else:
            expected = expected_defaults[read_ext[1:]]
        
        # 断言读取结果与预期引擎相符
        assert result == expected

    # 测试引擎的关键字参数设置情况，使用指定的读取扩展名和引擎
    def test_engine_kwargs(self, read_ext, engine):
        # GH#52214
        # 预期的默认引擎关键字参数映射表
        expected_defaults = {
            "xlsx": {"foo": "abcd"},
            "xlsm": {"foo": 123},
            "xlsb": {"foo": "True"},
            "xls": {"foo": True},
            "ods": {"foo": "abcd"},
        }

        # 根据不同的引擎设置异常消息
        if engine in {"xlrd", "pyxlsb"}:
            msg = re.escape(r"open_workbook() got an unexpected keyword argument 'foo'")
        elif engine == "odf":
            msg = re.escape(r"load() got an unexpected keyword argument 'foo'")
        else:
            msg = re.escape(r"load_workbook() got an unexpected keyword argument 'foo'")

        # 如果引擎不为空，使用 pytest.raises 检查是否抛出预期的 TypeError 异常
        if engine is not None:
            with pytest.raises(TypeError, match=msg):
                pd.read_excel(
                    "test1" + read_ext,
                    sheet_name="Sheet1",
                    index_col=0,
                    engine_kwargs=expected_defaults[read_ext[1:]],
                )

    # 测试 usecols 参数为整数的情况
    def test_usecols_int(self, read_ext):
        # usecols 作为整数时的异常消息
        msg = "Passing an integer for `usecols`"
        # 使用 pytest.raises 检查是否抛出预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(
                "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols=3
            )

        # usecols 作为整数时的异常消息
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(
                "test1" + read_ext,
                sheet_name="Sheet2",
                skiprows=[1],
                index_col=0,
                usecols=3,
            )

    # 测试 usecols 参数为列表的情况
    def test_usecols_list(self, request, engine, read_ext, df_ref):
        # 使用 xfail_datetimes_with_pyxlsb 函数处理日期时间相关的问题
        xfail_datetimes_with_pyxlsb(engine, request)

        # 根据 df_ref 调整预期的 DataFrame 结果，使用指定的读取扩展名和引擎
        expected = df_ref[["B", "C"]]
        adjust_expected(expected, read_ext, engine)

        # 使用 pd.read_excel 分别读取两个工作表的数据，指定使用的列为 0、2 和 3
        df1 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols=[0, 2, 3]
        )
        df2 = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols=[0, 2, 3],
        )

        # 比较读取的 DataFrame 与预期的 DataFrame 是否一致
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)
    # 测试使用列名字符串作为 usecols 参数来读取 Excel 文件
    def test_usecols_str(self, request, engine, read_ext, df_ref):
        # 如果使用 pyxlsb 引擎，标记为 xfail
        xfail_datetimes_with_pyxlsb(engine, request)

        # 从参考数据中选择指定列，准备作为预期结果
        expected = df_ref[["A", "B", "C"]]
        adjust_expected(expected, read_ext, engine)

        # 读取 Excel 文件中指定列的数据到 DataFrame
        df2 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A:D"
        )
        df3 = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols="A:D",
        )

        # 比较读取的数据与预期结果是否一致
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)

        # 更新预期结果，选择不同的列
        expected = df_ref[["B", "C"]]
        adjust_expected(expected, read_ext, engine)

        # 读取 Excel 文件中指定列的数据到 DataFrame
        df2 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A,C,D"
        )
        df3 = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols="A,C,D",
        )
        # 比较读取的数据与预期结果是否一致
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)

        # 读取 Excel 文件中指定列的数据到 DataFrame
        df2 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A,C:D"
        )
        df3 = pd.read_excel(
            "test1" + read_ext,
            sheet_name="Sheet2",
            skiprows=[1],
            index_col=0,
            usecols="A,C:D",
        )
        # 比较读取的数据与预期结果是否一致
        tm.assert_frame_equal(df2, expected)
        tm.assert_frame_equal(df3, expected)

    # 测试使用不同位置的整数列索引顺序作为 usecols 参数来读取 Excel 文件
    @pytest.mark.parametrize(
        "usecols", [[0, 1, 3], [0, 3, 1], [1, 0, 3], [1, 3, 0], [3, 0, 1], [3, 1, 0]]
    )
    def test_usecols_diff_positional_int_columns_order(
        self, request, engine, read_ext, usecols, df_ref
    ):
        # 如果使用 pyxlsb 引擎，标记为 xfail
        xfail_datetimes_with_pyxlsb(engine, request)

        # 从参考数据中选择指定列，准备作为预期结果
        expected = df_ref[["A", "C"]]
        adjust_expected(expected, read_ext, engine)

        # 读取 Excel 文件中指定列的数据到 DataFrame
        result = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols=usecols
        )
        # 比较读取的数据与预期结果是否一致
        tm.assert_frame_equal(result, expected)

    # 测试使用不同位置的字符串列名顺序作为 usecols 参数来读取 Excel 文件
    @pytest.mark.parametrize("usecols", [["B", "D"], ["D", "B"]])
    def test_usecols_diff_positional_str_columns_order(self, read_ext, usecols, df_ref):
        # 从参考数据中选择指定列，准备作为预期结果
        expected = df_ref[["B", "D"]]
        expected.index = range(len(expected))

        # 读取 Excel 文件中指定列的数据到 DataFrame
        result = pd.read_excel("test1" + read_ext, sheet_name="Sheet1", usecols=usecols)
        # 比较读取的数据与预期结果是否一致
        tm.assert_frame_equal(result, expected)

    # 测试读取 Excel 文件时不进行切片操作
    def test_read_excel_without_slicing(self, request, engine, read_ext, df_ref):
        # 如果使用 pyxlsb 引擎，标记为 xfail
        xfail_datetimes_with_pyxlsb(engine, request)

        # 从参考数据中选择所有列，准备作为预期结果
        expected = df_ref
        adjust_expected(expected, read_ext, engine)

        # 读取 Excel 文件中所有数据到 DataFrame
        result = pd.read_excel("test1" + read_ext, sheet_name="Sheet1", index_col=0)
        # 比较读取的数据与预期结果是否一致
        tm.assert_frame_equal(result, expected)
    # 定义测试函数，用于测试从 Excel 文件读取特定列的功能，当列范围为字符串时
    def test_usecols_excel_range_str(self, request, engine, read_ext, df_ref):
        # 调用辅助函数，处理引擎和数据框的期望结果
        xfail_datetimes_with_pyxlsb(engine, request)

        # 从参考数据框中选择特定列"C"和"D"作为期望结果
        expected = df_ref[["C", "D"]]
        # 根据读取的文件类型和引擎调整期望结果
        adjust_expected(expected, read_ext, engine)

        # 读取 Excel 文件中的数据，仅使用"A", "D", "E"列，以DataFrame形式返回
        result = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, usecols="A,D:E"
        )
        # 断言读取结果与期望结果相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，用于测试在使用不存在的列时是否会引发错误
    def test_usecols_excel_range_str_invalid(self, read_ext):
        # 设置错误消息
        msg = "Invalid column name: E1"

        # 断言读取 Excel 文件时，使用不存在的列"D", "E1"会引发 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, sheet_name="Sheet1", usecols="D:E1")

    # 定义测试函数，用于测试指定索引列为非整数标签时是否会引发错误
    def test_index_col_label_error(self, read_ext):
        # 设置错误消息
        msg = "list indices must be integers.*, not str"

        # 断言读取 Excel 文件时，指定索引列为列表["A"]会引发 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            pd.read_excel(
                "test1" + read_ext,
                sheet_name="Sheet1",
                index_col=["A"],
                usecols=["A", "C"],
            )

    # 定义测试函数，用于测试指定单一索引列为字符串时的行为
    def test_index_col_str(self, read_ext):
        # 从 Excel 文件中读取数据，指定索引列为"A"，返回结果与预期的空DataFrame相等
        result = pd.read_excel("test1" + read_ext, sheet_name="Sheet3", index_col="A")
        expected = DataFrame(
            columns=["B", "C", "D", "E", "F"], index=Index([], name="A")
        )
        # 断言读取结果与期望结果相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，用于测试指定多列索引为空时的行为
    def test_index_col_empty(self, read_ext):
        # 从 Excel 文件中读取数据，指定多列索引为["A", "B", "C"]，返回结果与预期的空MultiIndex DataFrame相等
        result = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet3", index_col=["A", "B", "C"]
        )
        expected = DataFrame(
            columns=["D", "E", "F"],
            index=MultiIndex(levels=[[]] * 3, codes=[[]] * 3, names=["A", "B", "C"]),
        )
        # 断言读取结果与期望结果相等
        tm.assert_frame_equal(result, expected)

    # 使用参数化测试，测试在具有未命名列的情况下指定索引列的行为
    @pytest.mark.parametrize("index_col", [None, 2])
    def test_index_col_with_unnamed(self, read_ext, index_col):
        # 从 Excel 文件中读取数据，根据参数化的索引列参数返回结果DataFrame
        result = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet4", index_col=index_col
        )
        expected = DataFrame(
            [["i1", "a", "x"], ["i2", "b", "y"]], columns=["Unnamed: 0", "col1", "col2"]
        )
        # 如果 index_col 不为 None，则将结果 DataFrame 设置索引为指定列
        if index_col:
            expected = expected.set_index(expected.columns[index_col])

        # 断言读取结果与期望结果相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，用于测试在使用不存在的列时是否会引发错误
    def test_usecols_pass_non_existent_column(self, read_ext):
        # 设置错误消息
        msg = (
            "Usecols do not match columns, "
            "columns expected but not found: "
            r"\['E'\]"
        )

        # 断言读取 Excel 文件时，使用不存在的列"E"会引发 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, usecols=["E"])

    # 定义测试函数，用于测试在错误类型的 usecols 参数时是否会引发错误
    def test_usecols_wrong_type(self, read_ext):
        # 设置错误消息
        msg = (
            "'usecols' must either be list-like of "
            "all strings, all unicode, all integers or a callable."
        )

        # 断言读取 Excel 文件时，使用错误类型的 usecols 参数会引发 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, usecols=["E1", 0])
    # 测试从 Excel 文件中读取数据并停止迭代
    def test_excel_stop_iterator(self, read_ext):
        # 使用 pandas 读取 Excel 文件的内容到 DataFrame
        parsed = pd.read_excel("test2" + read_ext, sheet_name="Sheet1")
        # 预期的 DataFrame，包含特定数据和列名
        expected = DataFrame([["aaaa", "bbbbb"]], columns=["Test", "Test1"])
        # 断言读取的 DataFrame 是否与预期的 DataFrame 相等
        tm.assert_frame_equal(parsed, expected)

    # 测试 Excel 文件中的单元格错误处理（NA值）
    def test_excel_cell_error_na(self, request, engine, read_ext):
        # 根据引擎和请求处理日期时间的特殊情况
        xfail_datetimes_with_pyxlsb(engine, request)

        # 如果使用 Calamine 引擎且读取的是 .ods 文件，标记为预期失败
        if engine == "calamine" and read_ext == ".ods":
            request.applymarker(
                pytest.mark.xfail(reason="Calamine can't extract error from ods files")
            )

        # 使用 pandas 读取 Excel 文件的内容到 DataFrame
        parsed = pd.read_excel("test3" + read_ext, sheet_name="Sheet1")
        # 预期的 DataFrame，包含单个 NaN 值的列
        expected = DataFrame([[np.nan]], columns=["Test"])
        # 断言读取的 DataFrame 是否与预期的 DataFrame 相等
        tm.assert_frame_equal(parsed, expected)

    # 测试从 Excel 表格中读取数据
    def test_excel_table(self, request, engine, read_ext, df_ref):
        # 根据引擎和请求处理日期时间的特殊情况
        xfail_datetimes_with_pyxlsb(engine, request)

        # 设置预期的 DataFrame 为参考 DataFrame
        expected = df_ref
        # 根据文件扩展名和引擎调整预期的 DataFrame
        adjust_expected(expected, read_ext, engine)

        # 从 Sheet1 中读取 Excel 文件的内容到 DataFrame，使用第一列作为索引
        df1 = pd.read_excel("test1" + read_ext, sheet_name="Sheet1", index_col=0)
        # 从 Sheet2 中读取 Excel 文件的内容到 DataFrame，跳过第二行，并使用第一列作为索引
        df2 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet2", skiprows=[1], index_col=0
        )
        # TODO 添加索引到文件
        # 断言读取的 DataFrame 是否与预期的 DataFrame 相等
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

        # 从 Sheet1 中读取 Excel 文件的内容到 DataFrame，使用第一列作为索引，并跳过最后一行
        df3 = pd.read_excel(
            "test1" + read_ext, sheet_name="Sheet1", index_col=0, skipfooter=1
        )
        # 断言读取的 DataFrame 是否与 df1 的除去最后一行的部分相等
        tm.assert_frame_equal(df3, df1.iloc[:-1])
    # 测试读取包含特殊数据类型的 Excel 文件
    def test_reader_special_dtypes(self, request, engine, read_ext):
        # 跳过对于使用 pyxlsb 引擎的 datetime 测试
        xfail_datetimes_with_pyxlsb(engine, request)

        # 获取期望的 DataFrame，包含不同的数据类型列
        unit = get_exp_unit(read_ext, engine)
        expected = DataFrame.from_dict(
            {
                "IntCol": [1, 2, -3, 4, 0],
                "FloatCol": [1.25, 2.25, 1.83, 1.92, 0.0000000005],
                "BoolCol": [True, False, True, True, False],
                "StrCol": [1, 2, 3, 4, 5],
                "Str2Col": ["a", 3, "c", "d", "e"],
                "DateCol": Index(
                    [
                        datetime(2013, 10, 30),
                        datetime(2013, 10, 31),
                        datetime(1905, 1, 1),
                        datetime(2013, 12, 14),
                        datetime(2015, 3, 14),
                    ],
                    dtype=f"M8[{unit}]",
                ),
            },
        )
        basename = "test_types"

        # 读取 Excel 文件，并检查是否与期望的 DataFrame 相等
        actual = pd.read_excel(basename + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, expected)

        # 如果不强制转换数值，那么整数会被读取为浮点数
        float_expected = expected.copy()
        float_expected.loc[float_expected.index[1], "Str2Col"] = 3.0
        actual = pd.read_excel(basename + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, float_expected)

        # 检查设置索引的情况（假设 xls 和 xlsx 在此处相同）
        for icol, name in enumerate(expected.columns):
            actual = pd.read_excel(
                basename + read_ext, sheet_name="Sheet1", index_col=icol
            )
            exp = expected.set_index(name)
            tm.assert_frame_equal(actual, exp)

        # 将 "StrCol" 列转换为字符串并检查
        expected["StrCol"] = expected["StrCol"].apply(str)
        actual = pd.read_excel(
            basename + read_ext, sheet_name="Sheet1", converters={"StrCol": str}
        )
        tm.assert_frame_equal(actual, expected)

    # GH8212 - 支持转换器和缺失值处理
    def test_reader_converters(self, read_ext):
        basename = "test_converters"

        # 期望的 DataFrame，包含不同的转换器应用到列
        expected = DataFrame.from_dict(
            {
                "IntCol": [1, 2, -3, -1000, 0],
                "FloatCol": [12.5, np.nan, 18.3, 19.2, 0.000000005],
                "BoolCol": ["Found", "Found", "Found", "Not found", "Found"],
                "StrCol": ["1", np.nan, "3", "4", "5"],
            }
        )

        # 定义转换器字典，针对每列应用不同的转换函数
        converters = {
            "IntCol": lambda x: int(x) if x != "" else -1000,
            "FloatCol": lambda x: 10 * x if x else np.nan,
            2: lambda x: "Found" if x != "" else "Not found",
            3: lambda x: str(x) if x else "",
        }

        # 读取 Excel 文件，并检查是否与期望的 DataFrame 相等
        actual = pd.read_excel(
            basename + read_ext, sheet_name="Sheet1", converters=converters
        )
        tm.assert_frame_equal(actual, expected)
    # 定义一个测试方法，用于验证读取不同数据类型的Excel文件的行为
    def test_reader_dtype(self, read_ext):
        # GH 8212
        # 设置基本文件名
        basename = "testdtype"
        # 读取 Excel 文件内容，存入 actual 变量
        actual = pd.read_excel(basename + read_ext)

        # 预期的 DataFrame 结构
        expected = DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [2.5, 3.5, 4.5, 5.5],
                "c": [1, 2, 3, 4],
                "d": [1.0, 2.0, np.nan, 4.0],
            }
        )

        # 断言 actual 与 expected 是否相等
        tm.assert_frame_equal(actual, expected)

        # 以指定的数据类型再次读取 Excel 文件内容，存入 actual 变量
        actual = pd.read_excel(
            basename + read_ext, dtype={"a": "float64", "b": "float32", "c": str}
        )

        # 调整预期的 DataFrame 结构中的数据类型
        expected["a"] = expected["a"].astype("float64")
        expected["b"] = expected["b"].astype("float32")
        expected["c"] = Series(["001", "002", "003", "004"], dtype=object)

        # 断言 actual 与 调整后的 expected 是否相等
        tm.assert_frame_equal(actual, expected)

        # 预期的错误消息
        msg = "Unable to convert column d to type int64"
        # 使用 pytest 检查是否会引发 ValueError 异常并包含特定消息
        with pytest.raises(ValueError, match=msg):
            pd.read_excel(basename + read_ext, dtype={"d": "int64"})

    @pytest.mark.parametrize(
        "dtype,expected",
        [
            # 第一个参数化测试用例：不指定 dtype
            (
                None,
                {
                    "a": [1, 2, 3, 4],
                    "b": [2.5, 3.5, 4.5, 5.5],
                    "c": [1, 2, 3, 4],
                    "d": [1.0, 2.0, np.nan, 4.0],
                },
            ),
            # 第二个参数化测试用例：指定不同的 dtype
            (
                {"a": "float64", "b": "float32", "c": str, "d": str},
                {
                    "a": Series([1, 2, 3, 4], dtype="float64"),
                    "b": Series([2.5, 3.5, 4.5, 5.5], dtype="float32"),
                    "c": Series(["001", "002", "003", "004"], dtype=object),
                    "d": Series(["1", "2", np.nan, "4"], dtype=object),
                },
            ),
        ],
    )
    # 参数化测试方法，用于验证不同 dtype 下读取 Excel 文件的行为
    def test_reader_dtype_str(self, read_ext, dtype, expected):
        # see gh-20377
        # 设置基本文件名
        basename = "testdtype"

        # 读取 Excel 文件内容，根据参数 dtype 指定数据类型，存入 actual 变量
        actual = pd.read_excel(basename + read_ext, dtype=dtype)
        # 根据参数 expected 构建预期的 DataFrame 结构
        expected = DataFrame(expected)
        # 断言 actual 与 expected 是否相等
        tm.assert_frame_equal(actual, expected)
    # 测试函数，用于测试指定数据类型后端的数据读取和比较
    def test_dtype_backend(self, read_ext, dtype_backend, engine, tmp_excel):
        # 如果文件扩展名是".xlsb"或".xls"，则跳过测试并显示消息
        if read_ext in (".xlsb", ".xls"):
            pytest.skip(f"No engine for filetype: '{read_ext}'")

        # 创建包含不同数据类型的DataFrame
        df = DataFrame(
            {
                "a": Series([1, 3], dtype="Int64"),
                "b": Series([2.5, 4.5], dtype="Float64"),
                "c": Series([True, False], dtype="boolean"),
                "d": Series(["a", "b"], dtype="string"),
                "e": Series([pd.NA, 6], dtype="Int64"),
                "f": Series([pd.NA, 7.5], dtype="Float64"),
                "g": Series([pd.NA, True], dtype="boolean"),
                "h": Series([pd.NA, "a"], dtype="string"),
                "i": Series([pd.Timestamp("2019-12-31")] * 2),
                "j": Series([pd.NA, pd.NA], dtype="Int64"),
            }
        )

        # 将DataFrame写入Excel文件
        df.to_excel(tmp_excel, sheet_name="test", index=False)

        # 从Excel文件读取数据作为结果
        result = pd.read_excel(
            tmp_excel, sheet_name="test", dtype_backend=dtype_backend
        )

        # 根据数据类型后端选择预期的DataFrame
        if dtype_backend == "pyarrow":
            # 导入所需的库和类
            import pyarrow as pa
            from pandas.arrays import ArrowExtensionArray

            # 使用ArrowExtensionArray创建预期的DataFrame
            expected = DataFrame(
                {
                    col: ArrowExtensionArray(pa.array(df[col], from_pandas=True))
                    for col in df.columns
                }
            )

            # 调整时间戳的精度为微秒，因为pyarrow默认使用微秒而不是纳秒
            expected["i"] = ArrowExtensionArray(
                expected["i"].array._pa_array.cast(pa.timestamp(unit="us"))
            )

            # 对于pyarrow，支持空类型，因此j列不需要默认为Int64
            expected["j"] = ArrowExtensionArray(pa.array([None, None]))
        else:
            # 否则，预期结果与原始DataFrame相同
            expected = df
            # 获取时间单位，例如"ns"或"us"
            unit = get_exp_unit(read_ext, engine)
            # 将i列的时间戳转换为指定单位
            expected["i"] = expected["i"].astype(f"M8[{unit}]")

        # 使用测试工具比较结果DataFrame和预期DataFrame
        tm.assert_frame_equal(result, expected)

    # 测试函数，用于测试同时指定数据类型后端和数据类型的数据读取和比较
    def test_dtype_backend_and_dtype(self, read_ext, tmp_excel):
        # 如果文件扩展名是".xlsb"或".xls"，则跳过测试并显示消息
        if read_ext in (".xlsb", ".xls"):
            pytest.skip(f"No engine for filetype: '{read_ext}'")

        # 创建包含NaN的DataFrame
        df = DataFrame({"a": [np.nan, 1.0], "b": [2.5, np.nan]})

        # 将DataFrame写入Excel文件
        df.to_excel(tmp_excel, sheet_name="test", index=False)

        # 从Excel文件读取数据作为结果，同时指定dtype_backend和dtype参数
        result = pd.read_excel(
            tmp_excel,
            sheet_name="test",
            dtype_backend="numpy_nullable",
            dtype="float64",
        )

        # 使用测试工具比较结果DataFrame和预期DataFrame
        tm.assert_frame_equal(result, df)

    # 标记测试用例为预期失败，说明使用pyarrow字符串数据类型时的情况
    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="infer_string takes precedence"
    )
    # 用于测试不同的 dtype 后端是否支持字符串存储格式的读取操作
    def test_dtype_backend_string(self, read_ext, string_storage, tmp_excel):
        # GH#36712
        # 如果读取的文件格式为 .xlsb 或 .xls，则跳过测试，因为没有对应的引擎支持
        if read_ext in (".xlsb", ".xls"):
            pytest.skip(f"No engine for filetype: '{read_ext}'")

        # 导入 pyarrow 库，如果导入失败，则跳过测试
        pa = pytest.importorskip("pyarrow")

        # 设置上下文环境，指定字符串存储模式（Python 或 Arrow）
        with pd.option_context("mode.string_storage", string_storage):
            # 创建一个包含两列的 DataFrame，其中包含对象类型的数组
            df = DataFrame(
                {
                    "a": np.array(["a", "b"], dtype=np.object_),
                    "b": np.array(["x", pd.NA], dtype=np.object_),
                }
            )
            # 将 DataFrame 写入 Excel 文件
            df.to_excel(tmp_excel, sheet_name="test", index=False)
            # 从 Excel 文件读取数据，使用 numpy_nullable 作为 dtype 后端
            result = pd.read_excel(
                tmp_excel, sheet_name="test", dtype_backend="numpy_nullable"
            )

            # 根据字符串存储模式不同，设置预期的 DataFrame
            if string_storage == "python":
                expected = DataFrame(
                    {
                        "a": StringArray(np.array(["a", "b"], dtype=np.object_)),
                        "b": StringArray(np.array(["x", pd.NA], dtype=np.object_)),
                    }
                )
            else:
                expected = DataFrame(
                    {
                        "a": ArrowStringArray(pa.array(["a", "b"])),
                        "b": ArrowStringArray(pa.array(["x", None])),
                    }
                )
            # 断言读取的结果与预期结果相等
            tm.assert_frame_equal(result, expected)

    # 测试在存在重复列名时，dtype 是否能正确处理的问题
    @pytest.mark.parametrize("dtypes, exp_value", [({}, 1), ({"a.1": "int64"}, 1)])
    def test_dtype_mangle_dup_cols(self, read_ext, dtypes, exp_value):
        # GH#35211
        # 定义基本文件名
        basename = "df_mangle_dup_col_dtypes"
        # 创建 dtype 字典，包含对象类型及任何其他指定的 dtypes
        dtype_dict = {"a": object, **dtypes}
        # 备份 dtype 字典
        dtype_dict_copy = dtype_dict.copy()
        # GH#42462
        # 从 Excel 文件中读取数据，使用指定的 dtype
        result = pd.read_excel(basename + read_ext, dtype=dtype_dict)
        # 设置预期的 DataFrame 结果
        expected = DataFrame(
            {
                "a": Series([1], dtype=object),
                "a.1": Series([exp_value], dtype=object if not dtypes else None),
            }
        )
        # 检查 dtype 字典是否发生了改变
        assert dtype_dict == dtype_dict_copy, "dtype dict changed"
        # 断言读取的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 测试读取带有空格的文件名的 Excel 文件
    def test_reader_spaces(self, read_ext):
        # see gh-32207
        # 定义基本文件名
        basename = "test_spaces"

        # 从 Excel 文件中读取数据
        actual = pd.read_excel(basename + read_ext)
        # 设置预期的 DataFrame 结果，包含一个包含特定字符串的列
        expected = DataFrame(
            {
                "testcol": [
                    "this is great",
                    "4    spaces",
                    "1 trailing ",
                    " 1 leading",
                    "2  spaces  multiple  times",
                ]
            }
        )
        # 断言读取的结果与预期结果相等
        tm.assert_frame_equal(actual, expected)

    # 参数化测试，验证不同情况下读取 Excel 文件的结果是否符合预期
    @pytest.mark.parametrize(
        "basename,expected",
        [
            ("gh-35802", DataFrame({"COLUMN": ["Test (1)"]})),
            ("gh-36122", DataFrame(columns=["got 2nd sa"])),
        ],
    )
    # 测试函数，用于读取以 .ods 格式存储的 Excel 文件中的嵌套 XML 数据
    def test_read_excel_ods_nested_xml(self, engine, read_ext, basename, expected):
        # 检查引擎是否为 "odf"，如果不是则跳过测试，并输出跳过原因
        if engine != "odf":
            pytest.skip(f"Skipped for engine: {engine}")

        # 使用 pandas 的 read_excel 函数读取指定文件，并将结果保存在 actual 变量中
        actual = pd.read_excel(basename + read_ext)
        # 使用 testdata.utils 中的 assert_frame_equal 函数比较 actual 和 expected 的数据框
        tm.assert_frame_equal(actual, expected)

    # 测试函数，用于读取 Excel 文件的所有工作表，并确保返回一个字典
    def test_reading_all_sheets(self, read_ext):
        # 测试读取所有工作表，将 sheet_name 参数设置为 None
        # 确保返回的是一个字典
        # 参考 PR #9450
        basename = "test_multisheet"
        dfs = pd.read_excel(basename + read_ext, sheet_name=None)
        # 确保返回的字典中包含预期的所有键值
        expected_keys = ["Charlie", "Alpha", "Beta"]
        tm.assert_contains_all(expected_keys, dfs.keys())
        # 检查工作表顺序是否被保留
        assert expected_keys == list(dfs.keys())

    # 测试函数，用于读取 Excel 文件中的特定多个工作表
    def test_reading_multiple_specific_sheets(self, read_ext):
        # 测试通过指定一个混合的整数和字符串列表来读取特定工作表
        # 确保去除重复的工作表引用（位置/名称）
        # 确保返回的是一个字典
        # 参考 PR #9450
        basename = "test_multisheet"
        # 显式请求包含重复值的工作表列表，确保只返回唯一的工作表
        expected_keys = [2, "Charlie", "Charlie"]
        dfs = pd.read_excel(basename + read_ext, sheet_name=expected_keys)
        expected_keys = list(set(expected_keys))
        tm.assert_contains_all(expected_keys, dfs.keys())
        assert len(expected_keys) == len(dfs.keys())

    # 测试函数，用于读取 Excel 文件中的所有工作表（包括空白工作表）
    def test_reading_all_sheets_with_blank(self, read_ext):
        # 测试读取所有工作表，将 sheet_name 参数设置为 None
        # 处理存在空白工作表的情况
        # Issue #11711
        basename = "blank_with_header"
        dfs = pd.read_excel(basename + read_ext, sheet_name=None)
        expected_keys = ["Sheet1", "Sheet2", "Sheet3"]
        tm.assert_contains_all(expected_keys, dfs.keys())

    # GH6403
    # 测试函数，用于读取一个带有空白数据的 Excel 文件中的指定工作表
    def test_read_excel_blank(self, read_ext):
        actual = pd.read_excel("blank" + read_ext, sheet_name="Sheet1")
        # 使用 testdata.utils 中的 assert_frame_equal 函数比较 actual 和一个空的 DataFrame
        tm.assert_frame_equal(actual, DataFrame())

    # 测试函数，用于读取一个带有表头的空白 Excel 文件中的指定工作表
    def test_read_excel_blank_with_header(self, read_ext):
        # 创建一个期望的 DataFrame，包含指定的列名
        expected = DataFrame(columns=["col_1", "col_2"])
        actual = pd.read_excel("blank_with_header" + read_ext, sheet_name="Sheet1")
        tm.assert_frame_equal(actual, expected)

    # 测试函数，用于验证异常消息是否包含指定的工作表名称
    def test_exception_message_includes_sheet_name(self, read_ext):
        # GH 48706
        # 检查是否引发 ValueError 异常，并验证异常消息是否包含指定的工作表名称 "Sheet1"
        with pytest.raises(ValueError, match=r" \(sheet: Sheet1\)$"):
            pd.read_excel("blank_with_header" + read_ext, header=[1], sheet_name=None)
        # 检查是否引发 ZeroDivisionError 异常，并验证异常消息是否包含指定的工作表名称 "Sheet1"
        with pytest.raises(ZeroDivisionError, match=r" \(sheet: Sheet1\)$"):
            pd.read_excel("test1" + read_ext, usecols=lambda x: 1 / 0, sheet_name=None)

    @pytest.mark.filterwarnings("ignore:Cell A4 is marked:UserWarning:openpyxl")
    def test_date_conversion_overflow(self, request, engine, read_ext):
        # 调用 xfail_datetimes_with_pyxlsb 函数处理与日期时间相关的 Excel 表格读取问题
        xfail_datetimes_with_pyxlsb(engine, request)

        # 创建预期的 DataFrame，包含日期列和字符串列
        expected = DataFrame(
            [
                [pd.Timestamp("2016-03-12"), "Marc Johnson"],
                [pd.Timestamp("2016-03-16"), "Jack Black"],
                [1e20, "Timothy Brown"],
            ],
            columns=["DateColWithBigInt", "StringCol"],
        )

        # 如果使用的引擎是 openpyxl，则对当前测试标记为失败，可能不被 openpyxl 支持
        if engine == "openpyxl":
            request.applymarker(
                pytest.mark.xfail(reason="Maybe not supported by openpyxl")
            )

        # 如果引擎未指定且文件扩展名为 .xlsx 或 .xlsm，则对当前测试标记为失败，默认使用 openpyxl，可能不被支持
        if engine is None and read_ext in (".xlsx", ".xlsm"):
            request.applymarker(
                pytest.mark.xfail(reason="Defaults to openpyxl, maybe not supported")
            )

        # 读取文件 "testdateoverflow" + read_ext 中的内容，并与预期的 DataFrame 进行比较
        result = pd.read_excel("testdateoverflow" + read_ext)
        tm.assert_frame_equal(result, expected)

    def test_sheet_name(self, request, read_ext, engine, df_ref):
        # 调用 xfail_datetimes_with_pyxlsb 函数处理与日期时间相关的 Excel 表格读取问题
        xfail_datetimes_with_pyxlsb(engine, request)

        # 定义文件名和工作表名称
        filename = "test1"
        sheet_name = "Sheet1"

        # 获取预期的 DataFrame，并根据读取的文件扩展名和引擎类型进行调整
        expected = df_ref
        adjust_expected(expected, read_ext, engine)

        # 读取文件 filename + read_ext 中的工作表 sheet_name，指定第一列为索引列，并与预期的 DataFrame 进行比较
        df1 = pd.read_excel(
            filename + read_ext, sheet_name=sheet_name, index_col=0
        )  # doc
        # 读取文件 filename + read_ext 中的工作表 sheet_name，指定第一列为索引列，并与预期的 DataFrame 进行比较
        df2 = pd.read_excel(filename + read_ext, index_col=0, sheet_name=sheet_name)

        # 比较两个读取结果的 DataFrame 是否相等
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

    def test_excel_read_buffer(self, read_ext):
        # 定义文件路径 pth
        pth = "test1" + read_ext
        # 从文件路径 pth 直接读取预期的 DataFrame，指定工作表名为 "Sheet1"，第一列为索引列
        expected = pd.read_excel(pth, sheet_name="Sheet1", index_col=0)
        # 使用文件对象方式读取文件 pth，并与预期的 DataFrame 进行比较
        with open(pth, "rb") as f:
            actual = pd.read_excel(f, sheet_name="Sheet1", index_col=0)
            tm.assert_frame_equal(expected, actual)

    def test_bad_engine_raises(self):
        # 定义一个错误的引擎名
        bad_engine = "foo"
        # 通过 pytest.raises 检查是否会引发 ValueError 异常，异常信息包含 "Unknown engine: foo"
        with pytest.raises(ValueError, match="Unknown engine: foo"):
            pd.read_excel("", engine=bad_engine)

    @pytest.mark.parametrize(
        "sheet_name",
        [3, [0, 3], [3, 0], "Sheet4", ["Sheet1", "Sheet4"], ["Sheet4", "Sheet1"]],
    )
    def test_bad_sheetname_raises(self, read_ext, sheet_name):
        # GH 39250
        # 定义异常信息字符串，用于匹配错误信息
        msg = "Worksheet index 3 is invalid|Worksheet named 'Sheet4' not found"
        # 通过 pytest.raises 检查是否会引发 ValueError 异常，异常信息匹配 msg 中定义的内容
        with pytest.raises(ValueError, match=msg):
            pd.read_excel("blank" + read_ext, sheet_name=sheet_name)

    def test_missing_file_raises(self, read_ext):
        # 定义一个不存在的文件名
        bad_file = f"foo{read_ext}"
        # 定义多语言环境下的文件未找到的匹配字符串
        match = "|".join(
            [
                "(No such file or directory",
                "没有那个文件或目录",
                "File o directory non esistente)",
            ]
        )
        # 通过 pytest.raises 检查是否会引发 FileNotFoundError 异常，异常信息匹配 match 中定义的内容
        with pytest.raises(FileNotFoundError, match=match):
            pd.read_excel(bad_file)
    def test_corrupt_bytes_raises(self, engine):
        # 定义一个损坏的字节流作为测试数据
        bad_stream = b"foo"
        
        # 根据不同的引擎设置错误类型和错误消息
        if engine is None:
            error = ValueError
            msg = (
                "Excel file format cannot be determined, you must "
                "specify an engine manually."
            )
        elif engine == "xlrd":
            from xlrd import XLRDError

            error = XLRDError
            msg = (
                "Unsupported format, or corrupt file: Expected BOF "
                "record; found b'foo'"
            )
        elif engine == "calamine":
            from python_calamine import CalamineError

            error = CalamineError
            msg = "Cannot detect file format"
        else:
            error = BadZipFile
            msg = "File is not a zip file"
        
        # 使用 pytest 的断言来验证是否引发了预期的错误类型和错误消息
        with pytest.raises(error, match=msg):
            pd.read_excel(BytesIO(bad_stream))

    @pytest.mark.network
    @pytest.mark.single_cpu
    def test_read_from_http_url(self, httpserver, read_ext):
        # 在 httpserver 上服务一个测试文件
        with open("test1" + read_ext, "rb") as f:
            httpserver.serve_content(content=f.read())
        
        # 从 HTTP URL 读取 Excel 文件，并与本地文件内容比较
        url_table = pd.read_excel(httpserver.url)
        local_table = pd.read_excel("test1" + read_ext)
        tm.assert_frame_equal(url_table, local_table)

    @td.skip_if_not_us_locale
    @pytest.mark.single_cpu
    def test_read_from_s3_url(self, read_ext, s3_public_bucket, s3so):
        # 在 S3 存储桶上上传一个测试文件
        # Bucket created in tests/io/conftest.py
        with open("test1" + read_ext, "rb") as f:
            s3_public_bucket.put_object(Key="test1" + read_ext, Body=f)

        # 构建 S3 URL 并从中读取 Excel 文件，与本地文件内容比较
        url = f"s3://{s3_public_bucket.name}/test1" + read_ext
        url_table = pd.read_excel(url, storage_options=s3so)
        local_table = pd.read_excel("test1" + read_ext)
        tm.assert_frame_equal(url_table, local_table)

    @pytest.mark.single_cpu
    def test_read_from_s3_object(self, read_ext, s3_public_bucket, s3so):
        # GH 38788
        # 在 S3 存储桶上上传一个测试文件
        # Bucket created in tests/io/conftest.py
        with open("test1" + read_ext, "rb") as f:
            s3_public_bucket.put_object(Key="test1" + read_ext, Body=f)

        # 使用 s3fs 打开 S3 对象，并读取其中的 Excel 文件内容
        import s3fs
        s3 = s3fs.S3FileSystem(**s3so)
        with s3.open(f"s3://{s3_public_bucket.name}/test1" + read_ext) as f:
            url_table = pd.read_excel(f)

        local_table = pd.read_excel("test1" + read_ext)
        tm.assert_frame_equal(url_table, local_table)

    @pytest.mark.slow
    def test_read_from_file_url(self, read_ext, datapath):
        # FILE
        # 从本地文件系统读取 Excel 文件内容，并与通过文件 URL 读取的内容比较
        localtable = os.path.join(datapath("io", "data", "excel"), "test1" + read_ext)
        local_table = pd.read_excel(localtable)

        try:
            url_table = pd.read_excel("file://localhost/" + localtable)
        except URLError:
            # 在某些系统上可能失败
            platform_info = " ".join(platform.uname()).strip()
            pytest.skip(f"failing on {platform_info}")

        tm.assert_frame_equal(url_table, local_table)
    # 测试从 pathlib.Path 读取文件内容的函数
    def test_read_from_pathlib_path(self, read_ext):
        # GH12655
        # 创建文件路径字符串
        str_path = "test1" + read_ext
        # 从 Excel 文件中读取数据到 DataFrame，设定第一列为索引
        expected = pd.read_excel(str_path, sheet_name="Sheet1", index_col=0)

        # 创建 Path 对象
        path_obj = Path("test1" + read_ext)
        # 从 Excel 文件中读取数据到 DataFrame，设定第一列为索引
        actual = pd.read_excel(path_obj, sheet_name="Sheet1", index_col=0)

        # 使用测试工具比较预期结果和实际结果的 DataFrame 是否相等
        tm.assert_frame_equal(expected, actual)

    # 测试从本地文件路径关闭文件后读取内容的函数
    def test_close_from_py_localpath(self, read_ext):
        # GH31467
        # 创建文件路径字符串
        str_path = os.path.join("test1" + read_ext)
        # 打开文件以二进制读取模式
        with open(str_path, "rb") as f:
            # 从 Excel 文件中读取数据到 DataFrame，设定第一列为索引
            x = pd.read_excel(f, sheet_name="Sheet1", index_col=0)
            # 删除变量 x，确保文件已关闭
            del x
            # 应该不会抛出异常，因为传递的文件已经关闭
            f.read()

    # 测试读取秒数数据的函数
    def test_reader_seconds(self, request, engine, read_ext):
        # 使用 pytest 标记测试数据处理函数以兼容 pyxlsb 引擎
        xfail_datetimes_with_pyxlsb(engine, request)

        # GH 55045
        # 如果使用 calamine 引擎并且文件扩展名为 .ods，则标记为预期失败
        if engine == "calamine" and read_ext == ".ods":
            request.applymarker(
                pytest.mark.xfail(
                    reason="ODS file contains bad datetime (seconds as text)"
                )
            )

        # 创建预期的 DataFrame，包含时间数据
        expected = DataFrame.from_dict(
            {
                "Time": [
                    time(1, 2, 3),
                    time(2, 45, 56, 100000),
                    time(4, 29, 49, 200000),
                    time(6, 13, 42, 300000),
                    time(7, 57, 35, 400000),
                    time(9, 41, 28, 500000),
                    time(11, 25, 21, 600000),
                    time(13, 9, 14, 700000),
                    time(14, 53, 7, 800000),
                    time(16, 37, 0, 900000),
                    time(18, 20, 54),
                ]
            }
        )

        # 从 Excel 文件中读取数据到 DataFrame，没有设定索引列
        actual = pd.read_excel("times_1900" + read_ext, sheet_name="Sheet1")
        # 使用测试工具比较预期结果和实际结果的 DataFrame 是否相等
        tm.assert_frame_equal(actual, expected)

        # 从 Excel 文件中读取数据到 DataFrame，没有设定索引列
        actual = pd.read_excel("times_1904" + read_ext, sheet_name="Sheet1")
        # 使用测试工具比较预期结果和实际结果的 DataFrame 是否相等
        tm.assert_frame_equal(actual, expected)
    # 测试读取带有多级索引的 Excel 文件的方法
    def test_read_excel_multiindex(self, request, engine, read_ext):
        # 标记为已知问题gh-4679，跳过在指定引擎上使用 pyxlsb 格式的日期时间测试
        xfail_datetimes_with_pyxlsb(engine, request)

        # 获取期望的时间单位
        unit = get_exp_unit(read_ext, engine)

        # 创建一个多级索引对象，包含两个级别和各自的标签
        mi = MultiIndex.from_product([["foo", "bar"], ["a", "b"]])
        mi_file = "testmultiindex" + read_ext

        # 构建预期的 DataFrame，包含不同数据类型的列，并将第三列转换为特定时间单位
        expected = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=mi,
        )
        expected[mi[2]] = expected[mi[2]].astype(f"M8[{unit}]")

        # 从 Excel 文件中读取名为 "mi_column" 的工作表，设置指定的行头和多级列头
        actual = pd.read_excel(
            mi_file, sheet_name="mi_column", header=[0, 1], index_col=0
        )
        tm.assert_frame_equal(actual, expected)

        # 从 Excel 文件中读取名为 "mi_index" 的工作表，设置多级索引列
        expected.index = mi
        expected.columns = ["a", "b", "c", "d"]
        actual = pd.read_excel(mi_file, sheet_name="mi_index", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)

        # 从 Excel 文件中读取名为 "both" 的工作表，设置多级索引行和列头
        expected.columns = mi
        actual = pd.read_excel(
            mi_file, sheet_name="both", index_col=[0, 1], header=[0, 1]
        )
        tm.assert_frame_equal(actual, expected)

        # 从 Excel 文件中读取名为 "mi_index_name" 的工作表，设置多级索引行并指定列头
        expected.columns = ["a", "b", "c", "d"]
        expected.index = mi.set_names(["ilvl1", "ilvl2"])
        actual = pd.read_excel(mi_file, sheet_name="mi_index_name", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)

        # 从 Excel 文件中读取名为 "mi_column_name" 的工作表，设置指定的行头和多级列头
        expected.index = list(range(4))
        expected.columns = mi.set_names(["c1", "c2"])
        actual = pd.read_excel(
            mi_file, sheet_name="mi_column_name", header=[0, 1], index_col=0
        )
        tm.assert_frame_equal(actual, expected)

        # 标记为已知问题gh-11317
        # 从 Excel 文件中读取名为 "name_with_int" 的工作表，设置指定的行头和多级列头
        expected.columns = mi.set_levels([1, 2], level=1).set_names(["c1", "c2"])
        actual = pd.read_excel(
            mi_file, sheet_name="name_with_int", index_col=0, header=[0, 1]
        )
        tm.assert_frame_equal(actual, expected)

        # 从 Excel 文件中读取名为 "both_name" 的工作表，设置多级索引行和列头
        expected.columns = mi.set_names(["c1", "c2"])
        expected.index = mi.set_names(["ilvl1", "ilvl2"])
        actual = pd.read_excel(
            mi_file, sheet_name="both_name", index_col=[0, 1], header=[0, 1]
        )
        tm.assert_frame_equal(actual, expected)

        # 从 Excel 文件中读取名为 "both_name_skiprows" 的工作表，设置多级索引行和列头，并跳过前两行
        actual = pd.read_excel(
            mi_file,
            sheet_name="both_name_skiprows",
            index_col=[0, 1],
            header=[0, 1],
            skiprows=2,
        )
        tm.assert_frame_equal(actual, expected)
    @pytest.mark.parametrize(
        "sheet_name,idx_lvl2",
        [  # 参数化测试数据：定义了两组参数，分别对应不同的表名和索引级别2
            ("both_name_blank_after_mi_name", [np.nan, "b", "a", "b"]),  # 第一组参数：表名和索引级别2的值数组
            ("both_name_multiple_blanks", [np.nan] * 4),  # 第二组参数：表名和索引级别2的空值数组
        ],
    )
    def test_read_excel_multiindex_blank_after_name(
        self, request, engine, read_ext, sheet_name, idx_lvl2
    ):
        # GH34673
        # 执行一个与 GH34673 相关的动作（这里是一个函数调用）
        xfail_datetimes_with_pyxlsb(engine, request)

        # 设置多级索引文件名
        mi_file = "testmultiindex" + read_ext
        # 创建一个二级多级索引对象
        mi = MultiIndex.from_product([["foo", "bar"], ["a", "b"]], names=["c1", "c2"])

        # 获取期望的数据单元
        unit = get_exp_unit(read_ext, engine)
        # 构造期望的数据帧
        expected = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=mi,
            index=MultiIndex.from_arrays(
                (["foo", "foo", "bar", "bar"], idx_lvl2),
                names=["ilvl1", "ilvl2"],
            ),
        )
        # 将时间戳列转换为指定单位的时间类型
        expected[mi[2]] = expected[mi[2]].astype(f"M8[{unit}]")
        
        # 读取 Excel 文件并获得结果
        result = pd.read_excel(
            mi_file,
            sheet_name=sheet_name,  # 指定表名
            index_col=[0, 1],  # 指定索引列
            header=[0, 1],  # 指定标题行
        )
        # 断言读取结果与期望结果相等
        tm.assert_frame_equal(result, expected)

    def test_read_excel_multiindex_header_only(self, read_ext):
        # see gh-11733.
        # 查看 gh-11733 的相关情况。
        #
        # 如果没有标题行，请不要尝试解析标题名。
        
        # 设置多级索引文件名
        mi_file = "testmultiindex" + read_ext
        # 从 Excel 中读取数据，指定表名和标题行索引
        result = pd.read_excel(mi_file, sheet_name="index_col_none", header=[0, 1])

        # 期望的列名为两级索引的产品
        exp_columns = MultiIndex.from_product([("A", "B"), ("key", "val")])
        # 构造期望的数据帧
        expected = DataFrame([[1, 2, 3, 4]] * 2, columns=exp_columns)
        # 断言读取结果与期望结果相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，测试旧的索引格式的 Excel 文件读取功能
    def test_excel_old_index_format(self, read_ext):
        # 创建测试用的文件名，包含特定的扩展名
        filename = "test_index_name_pre17" + read_ext

        # 创建一个二维数组作为测试数据，包含空值和字符串
        data = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                ["R0C0", "R0C1", "R0C2", "R0C3", "R0C4"],
                ["R1C0", "R1C1", "R1C2", "R1C3", "R1C4"],
                ["R2C0", "R2C1", "R2C2", "R2C3", "R2C4"],
                ["R3C0", "R3C1", "R3C2", "R3C3", "R3C4"],
                ["R4C0", "R4C1", "R4C2", "R4C3", "R4C4"],
            ],
            dtype=object,
        )
        
        # 定义列名
        columns = ["C_l0_g0", "C_l0_g1", "C_l0_g2", "C_l0_g3", "C_l0_g4"]
        
        # 创建一个多级索引对象
        mi = MultiIndex(
            levels=[
                ["R0", "R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"],
                ["R1", "R_l1_g0", "R_l1_g1", "R_l1_g2", "R_l1_g3", "R_l1_g4"],
            ],
            codes=[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]],
            names=[None, None],
        )
        
        # 创建一个单级索引对象
        si = Index(
            ["R0", "R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"], name=None
        )

        # 创建期望的 DataFrame 对象，使用指定的数据和索引
        expected = DataFrame(data, index=si, columns=columns)

        # 读取 Excel 文件中指定单元格作为单级索引列，并进行断言比较
        actual = pd.read_excel(filename, sheet_name="single_names", index_col=0)
        tm.assert_frame_equal(actual, expected)

        # 将期望的索引更改为多级索引
        expected.index = mi

        # 读取 Excel 文件中指定单元格作为多级索引列，并进行断言比较
        actual = pd.read_excel(filename, sheet_name="multi_names", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)

        # 创建不带索引名称的版本的测试数据
        data = np.array(
            [
                ["R0C0", "R0C1", "R0C2", "R0C3", "R0C4"],
                ["R1C0", "R1C1", "R1C2", "R1C3", "R1C4"],
                ["R2C0", "R2C1", "R2C2", "R2C3", "R2C4"],
                ["R3C0", "R3C1", "R3C2", "R3C3", "R3C4"],
                ["R4C0", "R4C1", "R4C2", "R4C3", "R4C4"],
            ]
        )
        
        # 创建不带索引名称的版本的列名
        columns = ["C_l0_g0", "C_l0_g1", "C_l0_g2", "C_l0_g3", "C_l0_g4"]
        
        # 创建一个多级索引对象
        mi = MultiIndex(
            levels=[
                ["R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"],
                ["R_l1_g0", "R_l1_g1", "R_l1_g2", "R_l1_g3", "R_l1_g4"],
            ],
            codes=[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],
            names=[None, None],
        )
        
        # 创建一个单级索引对象
        si = Index(["R_l0_g0", "R_l0_g1", "R_l0_g2", "R_l0_g3", "R_l0_g4"], name=None)

        # 创建期望的 DataFrame 对象，使用指定的数据和索引
        expected = DataFrame(data, index=si, columns=columns)

        # 读取 Excel 文件中指定单元格作为单级索引列，并进行断言比较
        actual = pd.read_excel(filename, sheet_name="single_no_names", index_col=0)
        tm.assert_frame_equal(actual, expected)

        # 将期望的索引更改为多级索引
        expected.index = mi

        # 读取 Excel 文件中指定单元格作为多级索引列，并进行断言比较
        actual = pd.read_excel(filename, sheet_name="multi_no_names", index_col=[0, 1])
        tm.assert_frame_equal(actual, expected)
    # 定义测试函数，用于测试 pd.read_excel 方法在 header 参数为布尔值时的行为
    def test_read_excel_bool_header_arg(self, read_ext):
        # GH 6114: 检验传递布尔值给 header 参数的无效性
        msg = "Passing a bool to header is invalid"
        # 遍历布尔值 True 和 False
        for arg in [True, False]:
            # 使用 pytest 断言检查是否会抛出 TypeError 异常，并验证错误消息
            with pytest.raises(TypeError, match=msg):
                pd.read_excel("test1" + read_ext, header=arg)

    # 定义测试函数，测试 pd.read_excel 方法在 skiprows 参数不同设置下的行为
    def test_read_excel_skiprows(self, request, engine, read_ext):
        # GH 4903: 调用 xfail_datetimes_with_pyxlsb 函数处理引擎和请求
        xfail_datetimes_with_pyxlsb(engine, request)

        # 获取期望时间单位
        unit = get_exp_unit(read_ext, engine)

        # 读取 Excel 文件，跳过指定行，读取特定表单
        actual = pd.read_excel(
            "testskiprows" + read_ext, sheet_name="skiprows_list", skiprows=[0, 2]
        )
        # 期望的 DataFrame 结果
        expected = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=["a", "b", "c", "d"],
        )
        # 转换特定列的数据类型为期望的时间单位
        expected["c"] = expected["c"].astype(f"M8[{unit}]")
        # 使用测试工具 tm 断言实际输出与期望输出相等
        tm.assert_frame_equal(actual, expected)

        # 使用 NumPy 数组作为 skiprows 参数进行读取
        actual = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=np.array([0, 2]),
        )
        tm.assert_frame_equal(actual, expected)

        # 使用 lambda 函数作为 skiprows 参数进行读取
        actual = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=lambda x: x in [0, 2],
        )
        tm.assert_frame_equal(actual, expected)

        # 读取跳过指定行数后的 Excel 数据，同时指定列名
        actual = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=3,
            names=["a", "b", "c", "d"],
        )
        # 期望的 DataFrame 结果
        expected = DataFrame(
            [
                # [1, 2.5, pd.Timestamp("2015-01-01"), True],
                [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=["a", "b", "c", "d"],
        )
        # 转换特定列的数据类型为期望的时间单位
        expected["c"] = expected["c"].astype(f"M8[{unit}]")
        # 使用测试工具 tm 断言实际输出与期望输出相等
        tm.assert_frame_equal(actual, expected)

    # 定义测试函数，测试 pd.read_excel 方法在 skiprows 参数为 lambda 表达式时的行为
    def test_read_excel_skiprows_callable_not_in(self, request, engine, read_ext):
        # GH 4903: 调用 xfail_datetimes_with_pyxlsb 函数处理引擎和请求
        xfail_datetimes_with_pyxlsb(engine, request)
        # 获取期望时间单位
        unit = get_exp_unit(read_ext, engine)

        # 使用 lambda 表达式作为 skiprows 参数，读取不在指定行数列表中的行
        actual = pd.read_excel(
            "testskiprows" + read_ext,
            sheet_name="skiprows_list",
            skiprows=lambda x: x not in [1, 3, 5],
        )
        # 期望的 DataFrame 结果
        expected = DataFrame(
            [
                [1, 2.5, pd.Timestamp("2015-01-01"), True],
                # [2, 3.5, pd.Timestamp("2015-01-02"), False],
                [3, 4.5, pd.Timestamp("2015-01-03"), False],
                # [4, 5.5, pd.Timestamp("2015-01-04"), True],
            ],
            columns=["a", "b", "c", "d"],
        )
        # 转换特定列的数据类型为期望的时间单位
        expected["c"] = expected["c"].astype(f"M8[{unit}]")
        # 使用测试工具 tm 断言实际输出与期望输出相等
        tm.assert_frame_equal(actual, expected)
    # 测试读取 Excel 文件时限制读取行数功能是否正常，限制为 5 行
    def test_read_excel_nrows(self, read_ext):
        # GH 16645
        # 定义要读取的行数
        num_rows_to_pull = 5
        # 实际读取 Excel 文件的前 num_rows_to_pull 行数据
        actual = pd.read_excel("test1" + read_ext, nrows=num_rows_to_pull)
        # 读取完整的 Excel 文件数据作为期望结果
        expected = pd.read_excel("test1" + read_ext)
        # 从期望结果中提取前 num_rows_to_pull 行数据
        expected = expected[:num_rows_to_pull]
        # 使用测试工具比较实际结果和期望结果是否相等
        tm.assert_frame_equal(actual, expected)

    # 测试当指定读取的行数大于文件中的行数时的情况
    def test_read_excel_nrows_greater_than_nrows_in_file(self, read_ext):
        # GH 16645
        # 读取完整的 Excel 文件数据作为期望结果
        expected = pd.read_excel("test1" + read_ext)
        # 获取文件中记录的行数
        num_records_in_file = len(expected)
        # 设定要读取的行数为文件中行数加上 10
        num_rows_to_pull = num_records_in_file + 10
        # 实际读取 Excel 文件的前 num_rows_to_pull 行数据
        actual = pd.read_excel("test1" + read_ext, nrows=num_rows_to_pull)
        # 使用测试工具比较实际结果和期望结果是否相等
        tm.assert_frame_equal(actual, expected)

    # 测试当指定的 nrows 参数不是整数时是否会引发 ValueError 异常
    def test_read_excel_nrows_non_integer_parameter(self, read_ext):
        # GH 16645
        # 设置预期的错误消息
        msg = "'nrows' must be an integer >=0"
        # 使用 pytest 断言捕获 ValueError 异常并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            pd.read_excel("test1" + read_ext, nrows="5")

    # 参数化测试，验证不同参数设置下读取 Excel 文件的行为是否一致
    @pytest.mark.parametrize(
        "filename,sheet_name,header,index_col,skiprows",
        [
            ("testmultiindex", "mi_column", [0, 1], 0, None),
            ("testmultiindex", "mi_index", None, [0, 1], None),
            ("testmultiindex", "both", [0, 1], [0, 1], None),
            ("testmultiindex", "mi_column_name", [0, 1], 0, None),
            ("testskiprows", "skiprows_list", None, None, [0, 2]),
            ("testskiprows", "skiprows_list", None, None, lambda x: x in (0, 2)),
        ],
    )
    # 测试读取 Excel 文件时不同参数设置的行为是否一致
    def test_read_excel_nrows_params(
        self, read_ext, filename, sheet_name, header, index_col, skiprows
    ):
        """
        For various parameters, we should get the same result whether we
        limit the rows during load (nrows=3) or after (df.iloc[:3]).
        """
        # GH 46894
        # 读取指定文件的前 3 行数据作为期望结果
        expected = pd.read_excel(
            filename + read_ext,
            sheet_name=sheet_name,
            header=header,
            index_col=index_col,
            skiprows=skiprows,
        ).iloc[:3]
        # 实际读取指定文件的前 3 行数据
        actual = pd.read_excel(
            filename + read_ext,
            sheet_name=sheet_name,
            header=header,
            index_col=index_col,
            skiprows=skiprows,
            nrows=3,
        )
        # 使用测试工具比较实际结果和期望结果是否相等
        tm.assert_frame_equal(actual, expected)

    # 测试使用不推荐的位置参数调用 read_excel 是否会引发 TypeError 异常
    def test_deprecated_kwargs(self, read_ext):
        # 使用 pytest 断言捕获 TypeError 异常，验证是否包含指定错误消息
        with pytest.raises(TypeError, match="but 3 positional arguments"):
            pd.read_excel("test1" + read_ext, "Sheet1", 0)

    # 测试在没有标题行的情况下使用列表索引列是否正常工作
    def test_no_header_with_list_index_col(self, read_ext):
        # GH 31783
        # 构造测试文件名
        file_name = "testmultiindex" + read_ext
        # 构造测试数据和索引
        data = [("B", "B"), ("key", "val"), (3, 4), (3, 4)]
        idx = MultiIndex.from_tuples(
            [("A", "A"), ("key", "val"), (1, 2), (1, 2)], names=(0, 1)
        )
        # 构造期望的 DataFrame
        expected = DataFrame(data, index=idx, columns=(2, 3))
        # 实际读取 Excel 文件数据，指定无标题行和列表索引列
        result = pd.read_excel(
            file_name, sheet_name="index_col_none", index_col=[0, 1], header=None
        )
        # 使用测试工具比较实际结果和期望结果是否相等
        tm.assert_frame_equal(expected, result)
    def test_one_col_noskip_blank_line(self, read_ext):
        # GH 39808
        # 构建文件名，添加文件扩展名
        file_name = "one_col_blank_line" + read_ext
        # 定义数据列表，包括 NaN 值
        data = [0.5, np.nan, 1, 2]
        # 构建预期的 DataFrame 对象
        expected = DataFrame(data, columns=["numbers"])
        # 使用 Pandas 读取 Excel 文件
        result = pd.read_excel(file_name)
        # 断言读取结果与预期的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    def test_multiheader_two_blank_lines(self, read_ext):
        # GH 40442
        # 构建文件名，添加文件扩展名
        file_name = "testmultiindex" + read_ext
        # 构建多级索引列
        columns = MultiIndex.from_tuples([("a", "A"), ("b", "B")])
        # 定义数据列表，包括 NaN 值
        data = [[np.nan, np.nan], [np.nan, np.nan], [1, 3], [2, 4]]
        # 构建预期的 DataFrame 对象
        expected = DataFrame(data, columns=columns)
        # 使用 Pandas 读取 Excel 文件，并指定多级索引的头部
        result = pd.read_excel(
            file_name, sheet_name="mi_column_empty_rows", header=[0, 1]
        )
        # 断言读取结果与预期的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    def test_trailing_blanks(self, read_ext):
        """
        Sheets can contain blank cells with no data. Some of our readers
        were including those cells, creating many empty rows and columns
        """
        # 构建文件名，添加文件扩展名
        file_name = "trailing_blanks" + read_ext
        # 使用 Pandas 读取 Excel 文件
        result = pd.read_excel(file_name)
        # 断言读取结果的形状为 (3, 3)
        assert result.shape == (3, 3)

    def test_ignore_chartsheets_by_str(self, request, engine, read_ext):
        # GH 41448
        # 如果文件扩展名为 .ods，则跳过测试
        if read_ext == ".ods":
            pytest.skip("chartsheets do not exist in the ODF format")
        # 如果引擎为 "pyxlsb"，标记为预期失败，因为无法区分图表工作表和普通工作表
        if engine == "pyxlsb":
            request.applymarker(
                pytest.mark.xfail(
                    reason="pyxlsb can't distinguish chartsheets from worksheets"
                )
            )
        # 使用 Pandas 读取 Excel 文件，预期会引发特定的 ValueError
        with pytest.raises(ValueError, match="Worksheet named 'Chart1' not found"):
            pd.read_excel("chartsheet" + read_ext, sheet_name="Chart1")

    def test_ignore_chartsheets_by_int(self, request, engine, read_ext):
        # GH 41448
        # 如果文件扩展名为 .ods，则跳过测试
        if read_ext == ".ods":
            pytest.skip("chartsheets do not exist in the ODF format")
        # 如果引擎为 "pyxlsb"，标记为预期失败，因为无法区分图表工作表和普通工作表
        if engine == "pyxlsb":
            request.applymarker(
                pytest.mark.xfail(
                    reason="pyxlsb can't distinguish chartsheets from worksheets"
                )
            )
        # 使用 Pandas 读取 Excel 文件，预期会引发特定的 ValueError
        with pytest.raises(
            ValueError, match="Worksheet index 1 is invalid, 1 worksheets found"
        ):
            pd.read_excel("chartsheet" + read_ext, sheet_name=1)

    def test_euro_decimal_format(self, read_ext):
        # copied from read_csv
        # 使用 Pandas 读取带有欧元符号的小数格式的 Excel 文件
        result = pd.read_excel("test_decimal" + read_ext, decimal=",", skiprows=1)
        # 定义预期的 DataFrame 对象
        expected = DataFrame(
            [
                [1, 1521.1541, 187101.9543, "ABC", "poi", 4.738797819],
                [2, 121.12, 14897.76, "DEF", "uyt", 0.377320872],
                [3, 878.158, 108013.434, "GHI", "rez", 2.735694704],
            ],
            columns=["Id", "Number1", "Number2", "Text1", "Text2", "Number3"],
        )
        # 断言读取结果与预期的 DataFrame 相等
        tm.assert_frame_equal(result, expected)
class TestExcelFileRead:
    # 定义测试类 TestExcelFileRead

    def test_raises_bytes_input(self, engine, read_ext):
        # 在测试中验证处理字节输入的异常情况
        # 使用 pytest 断言检查是否引发 TypeError 异常，并匹配预期的错误信息消息
        msg = "Expected file path name or file-like object"
        with pytest.raises(TypeError, match=msg):
            with open("test1" + read_ext, "rb") as f:
                pd.read_excel(f.read(), engine=engine)

    @pytest.fixture(autouse=True)
    def cd_and_set_engine(self, engine, datapath, monkeypatch):
        """
        Change directory and set engine for ExcelFile objects.
        """
        # 自动使用的 pytest fixture，用于改变目录并设置 ExcelFile 对象的引擎
        func = partial(pd.ExcelFile, engine=engine)
        monkeypatch.chdir(datapath("io", "data", "excel"))
        monkeypatch.setattr(pd, "ExcelFile", func)

    def test_engine_used(self, read_ext, engine):
        # 测试确保正确的引擎被使用
        # 定义不同文件类型（扩展名）对应的预期默认引擎
        expected_defaults = {
            "xlsx": "openpyxl",
            "xlsm": "openpyxl",
            "xlsb": "pyxlsb",
            "xls": "xlrd",
            "ods": "odf",
        }

        # 使用 pd.ExcelFile 打开指定文件，并检查其使用的引擎
        with pd.ExcelFile("test1" + read_ext) as excel:
            result = excel.engine

        # 根据测试参数 engine 或者使用默认的文件扩展名确定预期引擎
        if engine is not None:
            expected = engine
        else:
            expected = expected_defaults[read_ext[1:]]
        assert result == expected

    def test_excel_passes_na(self, read_ext):
        # 测试确保 read_excel 在处理 NA 值时的正确性
        # 使用 pd.ExcelFile 打开指定文件，并读取 Sheet1 的数据，自定义 NA 值处理方式
        with pd.ExcelFile("test4" + read_ext) as excel:
            parsed = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=False, na_values=["apple"]
            )
        # 预期的 DataFrame 结果
        expected = DataFrame(
            [["NA"], [1], ["NA"], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)

        # 重新读取文件，采用默认的 NA 值处理方式
        with pd.ExcelFile("test4" + read_ext) as excel:
            parsed = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=True, na_values=["apple"]
            )
        # 预期的 DataFrame 结果
        expected = DataFrame(
            [[np.nan], [1], [np.nan], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)

        # 针对另一个文件进行类似的测试，处理不同的 NA 值情况
        with pd.ExcelFile("test5" + read_ext) as excel:
            parsed = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=False, na_values=["apple"]
            )
        # 预期的 DataFrame 结果
        expected = DataFrame(
            [["1.#QNAN"], [1], ["nan"], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)

        with pd.ExcelFile("test5" + read_ext) as excel:
            parsed = pd.read_excel(
                excel, sheet_name="Sheet1", keep_default_na=True, na_values=["apple"]
            )
        # 预期的 DataFrame 结果
        expected = DataFrame(
            [[np.nan], [1], [np.nan], [np.nan], ["rabbit"]], columns=["Test"]
        )
        tm.assert_frame_equal(parsed, expected)

    @pytest.mark.parametrize("na_filter", [None, True, False])
    # 使用 pytest.mark.parametrize 标记，为测试方法参数化设置不同的 na_filter 值
    def test_excel_passes_na_filter(self, read_ext, na_filter):
        # 定义一个空字典
        kwargs = {}

        # 如果na_filter不为None，则将其添加到kwargs字典中
        if na_filter is not None:
            kwargs["na_filter"] = na_filter

        # 打开Excel文件，使用ExcelFile对象读取数据
        with pd.ExcelFile("test5" + read_ext) as excel:
            # 从Excel文件中读取数据到parsed变量中
            parsed = pd.read_excel(
                excel,
                sheet_name="Sheet1",
                keep_default_na=True,
                na_values=["apple"],
                **kwargs,
            )

        # 根据na_filter的值设置期望的数据
        if na_filter is False:
            expected = [["1.#QNAN"], [1], ["nan"], ["apple"], ["rabbit"]]
        else:
            expected = [[np.nan], [1], [np.nan], [np.nan], ["rabbit"]]

        # 创建DataFrame对象expected
        expected = DataFrame(expected, columns=["Test"])
        # 检查parsed和expected是否相等
        tm.assert_frame_equal(parsed, expected)

    def test_excel_table_sheet_by_index(self, request, engine, read_ext, df_ref):
        # 跳过使用pyxlsb引擎的日期时间测试
        xfail_datetimes_with_pyxlsb(engine, request)

        # 设置期望的数据
        expected = df_ref
        adjust_expected(expected, read_ext, engine)

        # 打开Excel文件，使用ExcelFile对象读取数据到df1和df2中
        with pd.ExcelFile("test1" + read_ext) as excel:
            df1 = pd.read_excel(excel, sheet_name=0, index_col=0)
            df2 = pd.read_excel(excel, sheet_name=1, skiprows=[1], index_col=0)
        # 检查df1和df2是否与期望数据相等
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

        # 使用parse方法读取数据到df1和df2中
        with pd.ExcelFile("test1" + read_ext) as excel:
            df1 = excel.parse(0, index_col=0)
            df2 = excel.parse(1, skiprows=[1], index_col=0)
        # 检查df1和df2是否与期望数据相等
        tm.assert_frame_equal(df1, expected)
        tm.assert_frame_equal(df2, expected)

        # 读取数据到df3中，跳过最后一行
        with pd.ExcelFile("test1" + read_ext) as excel:
            df3 = pd.read_excel(excel, sheet_name=0, index_col=0, skipfooter=1)
        # 检查df3是否与df1去掉最后一行的数据相等
        tm.assert_frame_equal(df3, df1.iloc[:-1])

        # 使用parse方法读取数据到df3中，跳过最后一行
        with pd.ExcelFile("test1" + read_ext) as excel:
            df3 = excel.parse(0, index_col=0, skipfooter=1)

        # 检查df3是否与df1去掉最后一行的数据相等
        tm.assert_frame_equal(df3, df1.iloc[:-1])

    def test_sheet_name(self, request, engine, read_ext, df_ref):
        # 跳过使用pyxlsb引擎的日期时间测试
        xfail_datetimes_with_pyxlsb(engine, request)

        # 设置期望的数据
        expected = df_ref
        adjust_expected(expected, read_ext, engine)

        # 定义文件名和工作表名
        filename = "test1"
        sheet_name = "Sheet1"

        # 打开Excel文件，使用ExcelFile对象读取数据到df1_parse中
        with pd.ExcelFile(filename + read_ext) as excel:
            df1_parse = excel.parse(sheet_name=sheet_name, index_col=0)  # doc

        # 打开Excel文件，使用ExcelFile对象读取数据到df2_parse中
        with pd.ExcelFile(filename + read_ext) as excel:
            df2_parse = excel.parse(index_col=0, sheet_name=sheet_name)

        # 检查df1_parse和df2_parse是否与期望数据相等
        tm.assert_frame_equal(df1_parse, expected)
        tm.assert_frame_equal(df2_parse, expected)

    @pytest.mark.parametrize(
        "sheet_name",
        [3, [0, 3], [3, 0], "Sheet4", ["Sheet1", "Sheet4"], ["Sheet4", "Sheet1"],
    )
    def test_bad_sheetname_raises(self, read_ext, sheet_name):
        # GH 39250
        # 定义错误消息
        msg = "Worksheet index 3 is invalid|Worksheet named 'Sheet4' not found"
        # 检查是否抛出值错误，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            with pd.ExcelFile("blank" + read_ext) as excel:
                excel.parse(sheet_name=sheet_name)
    # 定义测试函数，测试通过缓冲读取的方式读取 Excel 文件内容
    def test_excel_read_buffer(self, engine, read_ext):
        # 构造文件路径
        pth = "test1" + read_ext
        # 从 Excel 文件中读取数据作为预期结果
        expected = pd.read_excel(pth, sheet_name="Sheet1", index_col=0, engine=engine)

        # 使用二进制模式打开文件
        with open(pth, "rb") as f:
            # 使用 pandas 的 ExcelFile 对象打开文件
            with pd.ExcelFile(f) as xls:
                # 从 ExcelFile 对象读取数据作为实际结果
                actual = pd.read_excel(xls, sheet_name="Sheet1", index_col=0)

        # 使用测试框架比较预期结果和实际结果是否相等
        tm.assert_frame_equal(expected, actual)

    # 定义测试函数，验证读取 Excel 文件后文件是否关闭
    def test_reader_closes_file(self, engine, read_ext):
        # 使用二进制模式打开文件
        with open("test1" + read_ext, "rb") as f:
            # 使用 pandas 的 ExcelFile 对象打开文件
            with pd.ExcelFile(f) as xlsx:
                # 尝试解析 Excel 文件
                pd.read_excel(xlsx, sheet_name="Sheet1", index_col=0, engine=engine)

        # 断言文件已关闭
        assert f.closed

    # 定义测试函数，验证传递 ExcelFile 时引擎参数的冲突情况
    def test_conflicting_excel_engines(self, read_ext):
        # GH 26566
        msg = "Engine should not be specified when passing an ExcelFile"

        # 使用 ExcelFile 对象打开 Excel 文件
        with pd.ExcelFile("test1" + read_ext) as xl:
            # 断言在指定引擎参数时引发 ValueError 异常
            with pytest.raises(ValueError, match=msg):
                pd.read_excel(xl, engine="foo")

    # 定义测试函数，测试通过二进制数据读取 Excel 文件内容
    def test_excel_read_binary(self, engine, read_ext):
        # GH 15914
        # 从 Excel 文件中读取数据作为预期结果
        expected = pd.read_excel("test1" + read_ext, engine=engine)

        # 使用二进制模式打开文件并读取数据
        with open("test1" + read_ext, "rb") as f:
            data = f.read()

        # 使用 BytesIO 对象封装二进制数据，然后读取数据作为实际结果
        actual = pd.read_excel(BytesIO(data), engine=engine)

        # 使用测试框架比较预期结果和实际结果是否相等
        tm.assert_frame_equal(expected, actual)

    # 定义测试函数，测试通过文件对象读取 Excel 文件内容
    def test_excel_read_binary_via_read_excel(self, read_ext, engine):
        # GH 38424
        # 使用二进制模式打开文件
        with open("test1" + read_ext, "rb") as f:
            # 从文件对象直接读取 Excel 数据
            result = pd.read_excel(f, engine=engine)

        # 从文件路径读取 Excel 数据作为预期结果
        expected = pd.read_excel("test1" + read_ext, engine=engine)

        # 使用测试框架比较预期结果和实际结果是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，验证在超出索引列范围时读取 Excel 文件是否引发 ValueError 异常
    def test_read_excel_header_index_out_of_range(self, engine):
        # GH#43143
        # 使用二进制模式打开文件
        with open("df_header_oob.xlsx", "rb") as f:
            # 断言在指定超出最大范围的索引列头时引发 ValueError 异常
            with pytest.raises(ValueError, match="exceeds maximum"):
                pd.read_excel(f, header=[0, 1])

    # 使用参数化测试，测试带有索引列头的 Excel 文件读取情况
    @pytest.mark.parametrize("filename", ["df_empty.xlsx", "df_equals.xlsx"])
    def test_header_with_index_col(self, filename):
        # GH 33476
        # 创建预期的 DataFrame 结构
        idx = Index(["Z"], name="I2")
        cols = MultiIndex.from_tuples([("A", "B"), ("A", "B.1")], names=["I11", "I12"])
        expected = DataFrame([[1, 3]], index=idx, columns=cols, dtype="int64")

        # 从 Excel 文件中读取数据作为实际结果
        result = pd.read_excel(filename, sheet_name="Sheet1", index_col=0, header=[0, 1])

        # 使用测试框架比较预期结果和实际结果是否相等
        tm.assert_frame_equal(expected, result)
    # 测试读取包含多级索引的日期时间数据
    def test_read_datetime_multiindex(self, request, engine, read_ext):
        # 跳过对于 pyxlsb 引擎的日期时间数据处理
        xfail_datetimes_with_pyxlsb(engine, request)

        # 构建测试文件名
        f = "test_datetime_mi" + read_ext
        # 使用 pd.ExcelFile 打开 Excel 文件
        with pd.ExcelFile(f) as excel:
            # 使用指定引擎读取 Excel 文件，设置多级索引和列标题
            actual = pd.read_excel(excel, header=[0, 1], index_col=0, engine=engine)

        # 获取时间单位
        unit = get_exp_unit(read_ext, engine)

        # 创建测试用的 DatetimeIndex
        dti = pd.DatetimeIndex(["2020-02-29", "2020-03-01"], dtype=f"M8[{unit}]")
        # 创建预期的多级索引
        expected_column_index = MultiIndex.from_arrays(
            [dti[:1], dti[1:]],
            names=[
                dti[0].to_pydatetime(),
                dti[1].to_pydatetime(),
            ],
        )
        # 创建预期的空 DataFrame
        expected = DataFrame([], index=[], columns=expected_column_index)

        # 使用测试工具比较预期结果和实际结果
        tm.assert_frame_equal(expected, actual)

    # 测试引擎的无效选项
    def test_engine_invalid_option(self, read_ext):
        # read_ext 包含 '.'，因此使用奇怪的格式化
        with pytest.raises(ValueError, match="Value must be one of *"):
            # 设置 pd.option_context 进行引擎选项的上下文环境
            with pd.option_context(f"io.excel{read_ext}.reader", "abc"):
                pass

    # 测试忽略图表工作表
    def test_ignore_chartsheets(self, request, engine, read_ext):
        # 跳过对于 .ods 格式的图表工作表处理
        if read_ext == ".ods":
            pytest.skip("chartsheets do not exist in the ODF format")
        # 对于 pyxlsb 引擎，标记测试失败
        if engine == "pyxlsb":
            request.applymarker(
                pytest.mark.xfail(
                    reason="pyxlsb can't distinguish chartsheets from worksheets"
                )
            )
        # 使用 pd.ExcelFile 打开 Excel 文件，验证工作表名称为 ["Sheet1"]
        with pd.ExcelFile("chartsheet" + read_ext) as excel:
            assert excel.sheet_names == ["Sheet1"]

    # 测试处理损坏文件时的行为
    def test_corrupt_files_closed(self, engine, tmp_excel):
        # GH41778
        # 定义损坏文件可能引发的异常
        errors = (BadZipFile,)
        # 如果引擎为 None，则跳过该测试
        if engine is None:
            pytest.skip(f"Invalid test for engine={engine}")
        # 对于 xlrd 引擎，增加可能引发的异常类型
        elif engine == "xlrd":
            import xlrd
            errors = (BadZipFile, xlrd.biffh.XLRDError)
        # 对于 calamine 引擎，增加可能引发的异常类型
        elif engine == "calamine":
            from python_calamine import CalamineError
            errors = (CalamineError,)

        # 创建一个损坏的临时 Excel 文件
        Path(tmp_excel).write_text("corrupt", encoding="utf-8")
        # 禁用警告，以确保损坏文件处理时不会产生警告
        with tm.assert_produces_warning(False):
            try:
                # 尝试使用指定引擎打开临时 Excel 文件
                pd.ExcelFile(tmp_excel, engine=engine)
            except errors:
                pass
```