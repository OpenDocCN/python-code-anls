# `.\graphrag\graphrag\index\emit\table_emitter.py`

```py
# 2024年版权声明和许可声明
# 版权归 Microsoft Corporation 所有，根据 MIT 许可证授权

"""TableEmitter protocol for emitting tables to a destination."""
# TableEmitter 协议用于将表格数据发送到目标位置

from typing import Protocol
# 导入 Protocol 类型，用于定义协议

import pandas as pd
# 导入 pandas 库，用于处理数据框架

class TableEmitter(Protocol):
    """TableEmitter protocol for emitting tables to a destination."""
    # TableEmitter 协议用于将表格数据发送到目标位置

    async def emit(self, name: str, data: pd.DataFrame) -> None:
        """Emit a dataframe to storage."""
        # 发送数据框架到存储位置
```