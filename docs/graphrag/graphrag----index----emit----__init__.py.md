# `.\graphrag\graphrag\index\emit\__init__.py`

```py
# 版权声明和许可证声明，指明版权归属和许可证类型
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入 CSVTableEmitter 类，用于将数据管道的结果以 CSV 格式写入存储
from .csv_table_emitter import CSVTableEmitter
# 导入 create_table_emitter 和 create_table_emitters 函数，用于动态创建数据表发射器
from .factories import create_table_emitter, create_table_emitters
# 导入 JsonTableEmitter 类，用于将数据管道的结果以 JSON 格式写入存储
from .json_table_emitter import JsonTableEmitter
# 导入 ParquetTableEmitter 类，用于将数据管道的结果以 Parquet 格式写入存储
from .parquet_table_emitter import ParquetTableEmitter
# 导入 TableEmitter 类，作为所有数据表发射器的基类
from .table_emitter import TableEmitter
# 导入 TableEmitterType 类型，用于指明数据表发射器的类型
from .types import TableEmitterType

# 将这些导入的符号列入 __all__ 列表，以便在使用 from module import * 时被导入
__all__ = [
    "CSVTableEmitter",          # CSV 格式数据表发射器
    "JsonTableEmitter",         # JSON 格式数据表发射器
    "ParquetTableEmitter",      # Parquet 格式数据表发射器
    "TableEmitter",             # 数据表发射器基类
    "TableEmitterType",         # 数据表发射器类型
    "create_table_emitter",     # 创建单个数据表发射器的工厂函数
    "create_table_emitters",    # 创建多个数据表发射器的工厂函数
]
```