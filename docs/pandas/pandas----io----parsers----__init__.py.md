# `D:\src\scipysrc\pandas\pandas\io\parsers\__init__.py`

```
# 从 pandas 库中导入读取文本文件的相关模块和函数
from pandas.io.parsers.readers import (
    TextFileReader,  # 导入 TextFileReader 类，用于逐块读取文本文件
    TextParser,      # 导入 TextParser 类，用于处理文本文件解析的基类
    read_csv,        # 导入 read_csv 函数，用于读取逗号分隔值（CSV）文件
    read_fwf,        # 导入 read_fwf 函数，用于读取固定宽度格式（FWF）的文本文件
    read_table,      # 导入 read_table 函数，用于读取通用分隔值格式的文本文件
)

# 定义 __all__ 列表，指定在导入时仅导出以下几个符号
__all__ = ["TextFileReader", "TextParser", "read_csv", "read_fwf", "read_table"]
```