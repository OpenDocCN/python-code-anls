# `D:\src\scipysrc\pandas\pandas\io\excel\__init__.py`

```
# 导入所需的模块和函数
from pandas.io.excel._base import (
    ExcelFile,        # 导入 ExcelFile 类，用于处理 Excel 文件的读取
    ExcelWriter,      # 导入 ExcelWriter 类，用于处理 Excel 文件的写入
    read_excel,       # 导入 read_excel 函数，用于从 Excel 文件读取数据
)
from pandas.io.excel._odswriter import ODSWriter as _ODSWriter  # 导入 ODSWriter 类，用于写入 OpenDocument Spreadsheet (ODS) 格式文件
from pandas.io.excel._openpyxl import OpenpyxlWriter as _OpenpyxlWriter  # 导入 OpenpyxlWriter 类，用于写入 OpenPyXL 格式文件
from pandas.io.excel._util import register_writer  # 导入 register_writer 函数，用于注册 Excel 文件写入器
from pandas.io.excel._xlsxwriter import XlsxWriter as _XlsxWriter  # 导入 XlsxWriter 类，用于写入 XLSX 格式文件

__all__ = ["read_excel", "ExcelWriter", "ExcelFile"]  # 将 read_excel、ExcelWriter、ExcelFile 加入到模块的公开接口列表中

# 注册 OpenpyxlWriter 类为 Excel 文件写入器
register_writer(_OpenpyxlWriter)

# 注册 XlsxWriter 类为 Excel 文件写入器
register_writer(_XlsxWriter)

# 注册 ODSWriter 类为 Excel 文件写入器
register_writer(_ODSWriter)
```