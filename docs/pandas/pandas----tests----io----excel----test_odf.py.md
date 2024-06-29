# `D:\src\scipysrc\pandas\pandas\tests\io\excel\test_odf.py`

```
# 导入 functools 库，用于创建偏函数
import functools

# 导入 numpy 库，并简写为 np
import numpy as np

# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 pandas 库，并简写为 pd
import pandas as pd

# 导入 pandas 内部测试模块，用于比较数据框是否相等
import pandas._testing as tm

# 确保系统中有 odf 模块，如果没有则跳过测试
pytest.importorskip("odf")


# 自动运行的测试夹具，用于修改默认的 Excel 读取引擎为 "odf"
@pytest.fixture(autouse=True)
def cd_and_set_engine(monkeypatch, datapath):
    # 创建偏函数，将 pd.read_excel 函数的 engine 参数设为 "odf"
    func = functools.partial(pd.read_excel, engine="odf")
    # 使用 monkeypatch 修改 pd.read_excel 的实现为上述偏函数
    monkeypatch.setattr(pd, "read_excel", func)
    # 修改当前工作目录为指定路径下的 Excel 数据文件夹
    monkeypatch.chdir(datapath("io", "data", "excel"))


# 测试读取无效类型时是否能够抛出 ValueError 异常
def test_read_invalid_types_raises():
    # 打开 invalid_value_type.ods 文件，检查是否能捕获 "Unrecognized type awesome_new_type" 异常
    with pytest.raises(ValueError, match="Unrecognized type awesome_new_type"):
        pd.read_excel("invalid_value_type.ods")


# 测试从 WriterTable（写入表格）文件中读取表格数据
def test_read_writer_table():
    # 创建预期的数据框，用于与读取结果进行比较
    index = pd.Index(["Row 1", "Row 2", "Row 3"], name="Header")
    expected = pd.DataFrame(
        [[1, np.nan, 7], [2, np.nan, 8], [3, np.nan, 9]],
        index=index,
        columns=["Column 1", "Unnamed: 2", "Column 3"],
    )

    # 从 writertable.odt 文件中读取名为 "Table1" 的工作表数据
    result = pd.read_excel("writertable.odt", sheet_name="Table1", index_col=0)

    # 使用内部测试工具比较读取结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)


# 测试从带有 XML 元素间换行的文件中读取表格数据
def test_read_newlines_between_xml_elements_table():
    # 创建预期的数据框，用于与读取结果进行比较
    expected = pd.DataFrame(
        [[1.0, 4.0, 7], [np.nan, np.nan, 8], [3.0, 6.0, 9]],
        columns=["Column 1", "Column 2", "Column 3"],
    )

    # 从 test_newlines.ods 文件中读取表格数据
    result = pd.read_excel("test_newlines.ods")

    # 使用内部测试工具比较读取结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)


# 测试从带有非空单元格的文件中读取表格数据
def test_read_unempty_cells():
    # 创建预期的数据框，用于与读取结果进行比较
    expected = pd.DataFrame(
        [1, np.nan, 3, np.nan, 5],
        columns=["Column 1"],
    )

    # 从 test_unempty_cells.ods 文件中读取表格数据
    result = pd.read_excel("test_unempty_cells.ods")

    # 使用内部测试工具比较读取结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)


# 测试从带有单元格注释的文件中读取表格数据
def test_read_cell_annotation():
    # 创建预期的数据框，用于与读取结果进行比较
    expected = pd.DataFrame(
        ["test", np.nan, "test 3"],
        columns=["Column 1"],
    )

    # 从 test_cell_annotation.ods 文件中读取表格数据
    result = pd.read_excel("test_cell_annotation.ods")

    # 使用内部测试工具比较读取结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)
```