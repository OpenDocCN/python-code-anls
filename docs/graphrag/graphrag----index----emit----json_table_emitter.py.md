# `.\graphrag\graphrag\index\emit\json_table_emitter.py`

```py
# 从 Microsoft Corporation. 版权声明处导入模块
# 使用 MIT 许可证进行许可

"""JsonTableEmitter module."""
# 导入日志记录模块
import logging

# 导入 pandas 库，并使用 pd 别名
import pandas as pd

# 从 graphrag.index.storage 模块中导入 PipelineStorage 类
from graphrag.index.storage import PipelineStorage

# 从当前目录中的 table_emitter 模块中导入 TableEmitter 类
from .table_emitter import TableEmitter

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


class JsonTableEmitter(TableEmitter):
    """JsonTableEmitter class."""
    
    # 类型提示：_storage 属性为 PipelineStorage 类型
    _storage: PipelineStorage

    def __init__(self, storage: PipelineStorage):
        """Create a new Json Table Emitter."""
        # 初始化方法，接收 PipelineStorage 类型的 storage 参数
        self._storage = storage

    async def emit(self, name: str, data: pd.DataFrame) -> None:
        """Emit a dataframe to storage."""
        # 构造文件名，以 .json 结尾
        filename = f"{name}.json"

        # 记录日志，指示正在发出 JSON 表格文件
        log.info("emitting JSON table %s", filename)
        
        # 将数据帧转换为 JSON 字符串并存储到指定的存储介质中
        await self._storage.set(
            filename,
            data.to_json(orient="records", lines=True, force_ascii=False),
        )
```