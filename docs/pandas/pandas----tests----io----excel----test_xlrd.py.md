# `D:\src\scipysrc\pandas\pandas\tests\io\excel\test_xlrd.py`

```
# 导入所需的模块和库
import io  # 导入 io 模块，用于处理字节流

import numpy as np  # 导入 numpy 库，用于数据处理
import pytest  # 导入 pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 pandas 库，用于数据分析和处理
import pandas._testing as tm  # 导入 pandas 测试模块，用于测试框架

from pandas.io.excel import ExcelFile  # 从 pandas.io.excel 模块导入 ExcelFile 类，用于处理 Excel 文件
from pandas.io.excel._base import inspect_excel_format  # 导入 inspect_excel_format 函数，用于检查 Excel 文件格式

xlrd = pytest.importorskip("xlrd")  # 使用 pytest.importorskip 导入 xlrd 库，并确保它能正常导入

@pytest.fixture
def read_ext_xlrd():
    """
    使用 xlrd 读取 Excel 文件的有效扩展名。
    
    类似于 read_ext，但不包括 .ods、.xlsb，以及对于 xlrd>2 不包括 .xlsx、.xlsm
    """
    return ".xls"


def test_read_xlrd_book(read_ext_xlrd, datapath):
    # 设置参数和路径
    engine = "xlrd"
    sheet_name = "Sheet1"
    pth = datapath("io", "data", "excel", "test1.xls")
    
    # 使用 xlrd 打开 Excel 文件
    with xlrd.open_workbook(pth) as book:
        with ExcelFile(book, engine=engine) as xl:
            # 从 Excel 文件中读取数据到 DataFrame
            result = pd.read_excel(xl, sheet_name=sheet_name, index_col=0)

        # 用 xlrd 从 Excel 文件中读取预期的 DataFrame 数据
        expected = pd.read_excel(
            book, sheet_name=sheet_name, engine=engine, index_col=0
        )
        
    # 断言结果与预期相等
    tm.assert_frame_equal(result, expected)


def test_read_xlsx_fails(datapath):
    # GH 29375
    from xlrd.biffh import XLRDError
    
    # 设置路径
    path = datapath("io", "data", "excel", "test1.xlsx")
    
    # 使用 xlrd 读取 xlsx 文件时应抛出 XLRDError 异常
    with pytest.raises(XLRDError, match="Excel xlsx file; not supported"):
        pd.read_excel(path, engine="xlrd")


def test_nan_in_xls(datapath):
    # GH 54564
    # 设置路径
    path = datapath("io", "data", "excel", "test6.xls")

    # 创建预期的 DataFrame
    expected = pd.DataFrame({0: np.r_[0, 2].astype("int64"), 1: np.r_[1, np.nan]})

    # 从 Excel 文件中读取数据到 DataFrame
    result = pd.read_excel(path, header=None)

    # 断言结果与预期相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "file_header",
    [
        b"\x09\x00\x04\x00\x07\x00\x10\x00",
        b"\x09\x02\x06\x00\x00\x00\x10\x00",
        b"\x09\x04\x06\x00\x00\x00\x10\x00",
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
    ],
)
def test_read_old_xls_files(file_header):
    # GH 41226
    # 创建一个字节流对象
    f = io.BytesIO(file_header)
    
    # 检查该字节流的 Excel 文件格式是否为 "xls"
    assert inspect_excel_format(f) == "xls"
```