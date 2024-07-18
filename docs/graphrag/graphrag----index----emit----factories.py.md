# `.\graphrag\graphrag\index\emit\factories.py`

```py
# 从 graphrag.index.storage 模块导入 PipelineStorage 类
# 从 graphrag.index.typing 模块导入 ErrorHandlerFn 类型
# 从当前目录下的 csv_table_emitter 模块导入 CSVTableEmitter 类
# 从当前目录下的 json_table_emitter 模块导入 JsonTableEmitter 类
# 从当前目录下的 parquet_table_emitter 模块导入 ParquetTableEmitter 类
# 从当前目录下的 table_emitter 模块导入 TableEmitter 类
# 从当前目录下的 types 模块导入 TableEmitterType 类型

def create_table_emitter(
    emitter_type: TableEmitterType, storage: PipelineStorage, on_error: ErrorHandlerFn
) -> TableEmitter:
    """根据指定的类型创建表发射器对象。"""
    match emitter_type:
        case TableEmitterType.Json:
            return JsonTableEmitter(storage)
        case TableEmitterType.Parquet:
            return ParquetTableEmitter(storage, on_error)
        case TableEmitterType.CSV:
            return CSVTableEmitter(storage)
        case _:
            msg = f"Unsupported table emitter type: {emitter_type}"
            raise ValueError(msg)

def create_table_emitters(
    emitter_types: list[TableEmitterType],
    storage: PipelineStorage,
    on_error: ErrorHandlerFn,
) -> list[TableEmitter]:
    """根据指定的类型列表创建表发射器对象列表。"""
    return [
        create_table_emitter(emitter_type, storage, on_error)
        for emitter_type in emitter_types
    ]
```