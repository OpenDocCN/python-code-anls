# `D:\src\scipysrc\pandas\pandas\tests\io\excel\test_xlsxwriter.py`

```
# 导入需要的库和模块
import contextlib  # 上下文管理模块，用于管理上下文中的资源
import uuid  # 用于生成唯一标识符的模块

import pytest  # 测试框架pytest

from pandas import DataFrame  # 导入pandas库中的DataFrame类

from pandas.io.excel import ExcelWriter  # 导入pandas库中的ExcelWriter类

# 使用 pytest.importorskip 来导入 xlsxwriter 库，如果导入失败则跳过测试
xlsxwriter = pytest.importorskip("xlsxwriter")

@pytest.fixture
def ext():
    return ".xlsx"

@pytest.fixture
def tmp_excel(ext, tmp_path):
    # 创建一个临时的 Excel 文件路径，并返回其路径字符串
    tmp = tmp_path / f"{uuid.uuid4()}{ext}"
    tmp.touch()  # 创建空文件
    return str(tmp)

def test_column_format(tmp_excel):
    # 测试确保列格式应用到单元格上，针对问题 #9167
    # 仅适用于 xlsxwriter 引擎

    # 创建一个 DataFrame 对象
    frame = DataFrame({"A": [123456, 123456], "B": [123456, 123456]})

    # 使用 ExcelWriter 打开临时 Excel 文件
    with ExcelWriter(tmp_excel) as writer:
        frame.to_excel(writer)  # 将 DataFrame 写入 Excel

        # 向 B 列添加数字格式，并确保应用到单元格
        num_format = "#,##0"
        write_workbook = writer.book  # 获取写入的工作簿对象
        write_worksheet = write_workbook.worksheets()[0]  # 获取第一个工作表对象
        col_format = write_workbook.add_format({"num_format": num_format})  # 创建列格式对象
        write_worksheet.set_column("B:B", None, col_format)  # 设置 B 列的列宽和格式

    # 使用 openpyxl 加载临时 Excel 文件
    with contextlib.closing(openpyxl.load_workbook(tmp_excel)) as read_workbook:
        try:
            read_worksheet = read_workbook["Sheet1"]  # 尝试获取名为 "Sheet1" 的工作表对象
        except TypeError:
            # 兼容处理
            read_worksheet = read_workbook.get_sheet_by_name(name="Sheet1")

    # 从单元格中获取数字格式
    try:
        cell = read_worksheet["B2"]  # 尝试获取 B2 单元格对象
    except TypeError:
        # 兼容处理
        cell = read_worksheet.cell("B2")

    try:
        read_num_format = cell.number_format  # 获取单元格的数字格式
    except AttributeError:
        read_num_format = cell.style.number_format._format_code  # 兼容处理

    # 断言读取到的数字格式与预期的格式相同
    assert read_num_format == num_format

def test_write_append_mode_raises(tmp_excel):
    # 测试当使用 xlsxwriter 引擎时，以 "a" 模式打开 ExcelWriter 是否会引发 ValueError 异常
    msg = "Append mode is not supported with xlsxwriter!"

    with pytest.raises(ValueError, match=msg):
        ExcelWriter(tmp_excel, engine="xlsxwriter", mode="a")

@pytest.mark.parametrize("nan_inf_to_errors", [True, False])
def test_engine_kwargs(tmp_excel, nan_inf_to_errors):
    # 测试 engine_kwargs 参数是否正确传递给引擎
    # GH 42286

    engine_kwargs = {"options": {"nan_inf_to_errors": nan_inf_to_errors}}
    # 使用 ExcelWriter 打开临时 Excel 文件，并传入 engine_kwargs 参数
    with ExcelWriter(tmp_excel, engine="xlsxwriter", engine_kwargs=engine_kwargs) as writer:
        # 断言 writer.book.nan_inf_to_errors 是否等于 nan_inf_to_errors
        assert writer.book.nan_inf_to_errors == nan_inf_to_errors

def test_book_and_sheets_consistent(tmp_excel):
    # 测试当用户修改工作簿时，确保 sheets 属性与工作簿保持一致
    # GH#45687

    # 使用 ExcelWriter 打开临时 Excel 文件，指定引擎为 xlsxwriter
    with ExcelWriter(tmp_excel, engine="xlsxwriter") as writer:
        # 断言 writer.sheets 应为空字典
        assert writer.sheets == {}
        # 向工作簿中添加名为 "test_name" 的工作表，并断言 writer.sheets 中包含这个工作表
        sheet = writer.book.add_worksheet("test_name")
        assert writer.sheets == {"test_name": sheet}
```