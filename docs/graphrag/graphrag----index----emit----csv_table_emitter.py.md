# `.\graphrag\graphrag\index\emit\csv_table_emitter.py`

```py
# 导入日志模块
import logging

# 导入 pandas 库，并使用别名 pd
import pandas as pd

# 从 graphrag.index.storage 模块中导入 PipelineStorage 类
from graphrag.index.storage import PipelineStorage

# 从当前包中导入 TableEmitter 类
from .table_emitter import TableEmitter

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


# 定义 CSVTableEmitter 类，继承自 TableEmitter 类
class CSVTableEmitter(TableEmitter):
    """CSVTableEmitter class."""

    # 类变量 _storage，类型为 PipelineStorage
    _storage: PipelineStorage

    # 初始化方法，接收 PipelineStorage 类型的参数 storage
    def __init__(self, storage: PipelineStorage):
        """Create a new CSV Table Emitter."""
        # 将传入的 storage 参数赋值给实例变量 _storage
        self._storage = storage

    # 异步方法 emit，用于将 DataFrame 写入存储中
    async def emit(self, name: str, data: pd.DataFrame) -> None:
        """Emit a dataframe to storage."""
        # 根据传入的 name 构造 CSV 文件名
        filename = f"{name}.csv"
        
        # 记录日志，指示正在写入 CSV 表格
        log.info("emitting CSV table %s", filename)
        
        # 调用 _storage 对象的 set 方法，将 DataFrame 转换为 CSV 格式并存储
        await self._storage.set(
            filename,
            data.to_csv(),
        )
```