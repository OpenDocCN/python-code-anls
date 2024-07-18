# `.\graphrag\graphrag\index\emit\parquet_table_emitter.py`

```py
# 版权声明和许可证信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""ParquetTableEmitter module."""

# 导入所需的模块和库
import logging           # 导入日志记录模块
import traceback         # 导入异常跟踪模块

import pandas as pd      # 导入 pandas 库
from pyarrow.lib import ArrowInvalid, ArrowTypeError   # 导入 pyarrow 库中的异常类

# 导入项目内部模块
from graphrag.index.storage import PipelineStorage      # 导入 PipelineStorage 类
from graphrag.index.typing import ErrorHandlerFn       # 导入 ErrorHandlerFn 类型

from .table_emitter import TableEmitter    # 导入本地的 TableEmitter 类

log = logging.getLogger(__name__)    # 获取当前模块的日志记录器


class ParquetTableEmitter(TableEmitter):
    """ParquetTableEmitter class."""
    
    _storage: PipelineStorage    # 类型注解，用于指定 _storage 属性的类型为 PipelineStorage
    _on_error: ErrorHandlerFn    # 类型注解，用于指定 _on_error 属性的类型为 ErrorHandlerFn

    def __init__(
        self,
        storage: PipelineStorage,
        on_error: ErrorHandlerFn,
    ):
        """Create a new Parquet Table Emitter."""
        self._storage = storage    # 初始化 _storage 属性
        self._on_error = on_error  # 初始化 _on_error 属性

    async def emit(self, name: str, data: pd.DataFrame) -> None:
        """Emit a dataframe to storage."""
        filename = f"{name}.parquet"    # 构建输出文件名
        log.info("emitting parquet table %s", filename)    # 记录日志信息，指示正在写入 parquet 表格

        try:
            await self._storage.set(filename, data.to_parquet())    # 将 DataFrame 写入 Parquet 文件到指定的 storage
        except ArrowTypeError as e:
            log.exception("Error while emitting parquet table")    # 记录异常信息到日志
            self._on_error(
                e,
                traceback.format_exc(),
                None,
            )    # 在发生 ArrowTypeError 异常时，调用错误处理函数 _on_error
        except ArrowInvalid as e:
            log.exception("Error while emitting parquet table")    # 记录异常信息到日志
            self._on_error(
                e,
                traceback.format_exc(),
                None,
            )    # 在发生 ArrowInvalid 异常时，调用错误处理函数 _on_error
```