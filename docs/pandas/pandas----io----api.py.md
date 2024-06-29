# `D:\src\scipysrc\pandas\pandas\io\api.py`

```
# 导入所需的数据输入输出库函数

from pandas.io.clipboards import read_clipboard
from pandas.io.excel import (
    ExcelFile,               # 导入 ExcelFile 类，用于处理 Excel 文件
    ExcelWriter,             # 导入 ExcelWriter 类，用于写入 Excel 文件
    read_excel,              # 导入 read_excel 函数，用于读取 Excel 文件
)
from pandas.io.feather_format import read_feather  # 导入 read_feather 函数，用于读取 Feather 格式文件
from pandas.io.html import read_html                # 导入 read_html 函数，用于读取 HTML 文件或页面
from pandas.io.json import read_json                # 导入 read_json 函数，用于读取 JSON 文件
from pandas.io.orc import read_orc                  # 导入 read_orc 函数，用于读取 ORC 文件
from pandas.io.parquet import read_parquet          # 导入 read_parquet 函数，用于读取 Parquet 文件
from pandas.io.parsers import (
    read_csv,               # 导入 read_csv 函数，用于读取 CSV 文件
    read_fwf,               # 导入 read_fwf 函数，用于读取固定宽度格式的文件
    read_table,             # 导入 read_table 函数，用于读取通用分隔符文件
)
from pandas.io.pickle import (
    read_pickle,            # 导入 read_pickle 函数，用于读取 pickle 序列化的对象
    to_pickle,              # 导入 to_pickle 函数，用于将对象序列化为 pickle 格式并保存
)
from pandas.io.pytables import (
    HDFStore,               # 导入 HDFStore 类，用于处理 HDF5 格式的数据
    read_hdf,               # 导入 read_hdf 函数，用于读取 HDF5 文件
)
from pandas.io.sas import read_sas                  # 导入 read_sas 函数，用于读取 SAS 文件
from pandas.io.spss import read_spss                # 导入 read_spss 函数，用于读取 SPSS 文件
from pandas.io.sql import (
    read_sql,               # 导入 read_sql 函数，用于执行 SQL 查询并返回结果集
    read_sql_query,         # 导入 read_sql_query 函数，用于执行 SQL 查询并返回结果集
    read_sql_table,         # 导入 read_sql_table 函数，用于读取 SQL 数据库中的表
)
from pandas.io.stata import read_stata              # 导入 read_stata 函数，用于读取 Stata 文件
from pandas.io.xml import read_xml                  # 导入 read_xml 函数，用于读取 XML 文件

__all__ = [
    "ExcelFile",            # 将 ExcelFile 类添加到 __all__ 列表中，使其在模块中可访问
    "ExcelWriter",          # 将 ExcelWriter 类添加到 __all__ 列表中，使其在模块中可访问
    "HDFStore",             # 将 HDFStore 类添加到 __all__ 列表中，使其在模块中可访问
    "read_clipboard",       # 将 read_clipboard 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_csv",             # 将 read_csv 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_excel",           # 将 read_excel 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_feather",         # 将 read_feather 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_fwf",             # 将 read_fwf 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_hdf",             # 将 read_hdf 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_html",            # 将 read_html 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_json",            # 将 read_json 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_orc",             # 将 read_orc 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_parquet",         # 将 read_parquet 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_pickle",          # 将 read_pickle 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_sas",             # 将 read_sas 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_spss",            # 将 read_spss 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_sql",             # 将 read_sql 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_sql_query",       # 将 read_sql_query 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_sql_table",       # 将 read_sql_table 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_stata",           # 将 read_stata 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_table",           # 将 read_table 函数添加到 __all__ 列表中，使其在模块中可访问
    "read_xml",             # 将 read_xml 函数添加到 __all__ 列表中，使其在模块中可访问
    "to_pickle",            # 将 to_pickle 函数添加到 __all__ 列表中，使其在模块中可访问
]
```