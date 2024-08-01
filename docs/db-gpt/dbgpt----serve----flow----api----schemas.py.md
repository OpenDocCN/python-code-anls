# `.\DB-GPT-src\dbgpt\serve\flow\api\schemas.py`

```py
# 从 dbgpt._private.pydantic 模块导入 ConfigDict 类
from dbgpt._private.pydantic import ConfigDict

# 导入 dbgpt.core.awel.flow.flow_factory 模块中的 FlowPanel 类
from dbgpt.core.awel.flow.flow_factory import FlowPanel

# 从上一级目录导入 SERVE_APP_NAME_HUMP 常量
from ..config import SERVE_APP_NAME_HUMP

# 将 FlowPanel 类赋值给 ServeRequest，用作服务请求的别名
ServeRequest = FlowPanel

# 定义 ServerResponse 类，继承自 FlowPanel 类，用于表示流响应模型
class ServerResponse(FlowPanel):
    """Flow response model"""

    # TODO define your own fields here

    # 创建 ConfigDict 实例，设置标题为带有 SERVE_APP_NAME_HUMP 常量的字符串
    model_config = ConfigDict(title=f"ServerResponse for {SERVE_APP_NAME_HUMP}")
```