# `.\graphrag\graphrag\index\verbs\snapshot.py`

```py
# 从 datashaper 模块导入 TableContainer, VerbInput 和 verb 函数
# 从 graphrag.index.storage 导入 PipelineStorage 类
from datashaper import TableContainer, VerbInput, verb
from graphrag.index.storage import PipelineStorage

# 使用 verb 装饰器定义名为 "snapshot" 的异步函数
@verb(name="snapshot")
async def snapshot(
    input: VerbInput,        # 输入参数，类型为 VerbInput，用于获取输入数据
    name: str,               # 必需的字符串参数，指定快照的名称
    formats: list[str],      # 列表类型的字符串，包含指定的输出格式（如 "parquet" 或 "json"）
    storage: PipelineStorage, # PipelineStorage 类型的参数，用于存储快照数据的对象
    **_kwargs: dict,         # 其余的关键字参数，不做具体说明
) -> TableContainer:         # 返回类型为 TableContainer，包含快照数据的容器
    """Take a entire snapshot of the tabular data."""
    
    data = input.get_input()  # 获取输入数据
    
    # 遍历指定的输出格式列表
    for fmt in formats:
        if fmt == "parquet":
            # 如果格式是 "parquet"，将数据以 Parquet 格式存储到指定的存储对象中
            await storage.set(name + ".parquet", data.to_parquet())
        elif fmt == "json":
            # 如果格式是 "json"，将数据以 JSON 格式存储到指定的存储对象中
            await storage.set(
                name + ".json", data.to_json(orient="records", lines=True)
            )

    # 返回一个包含原始数据的 TableContainer 对象
    return TableContainer(table=data)
```