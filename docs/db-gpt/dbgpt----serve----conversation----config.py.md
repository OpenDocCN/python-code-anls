# `.\DB-GPT-src\dbgpt\serve\conversation\config.py`

```py
from dataclasses import dataclass, field   # 导入必要的模块和类
from typing import Optional   # 导入 Optional 类型

from dbgpt.serve.core import BaseServeConfig   # 导入自定义模块 BaseServeConfig

# 应用名称常量
APP_NAME = "conversation"
# 服务应用名称常量
SERVE_APP_NAME = "dbgpt_serve_conversation"
# 驼峰式服务应用名称常量
SERVE_APP_NAME_HUMP = "dbgpt_serve_Conversation"
# Serve 配置键前缀常量
SERVE_CONFIG_KEY_PREFIX = "dbgpt.serve.conversation."
# Serve 服务组件名称，由服务应用名称和后缀 '_service' 组成
SERVE_SERVICE_COMPONENT_NAME = f"{SERVE_APP_NAME}_service"
# 数据库表名称
SERVER_APP_TABLE_NAME = "dbgpt_serve_conversation"

@dataclass
class ServeConfig(BaseServeConfig):
    """Parameters for the serve command"""

    # TODO: add your own parameters here
    # API 密钥参数，可选字符串类型，默认为 None，用于接口的认证
    api_keys: Optional[str] = field(
        default=None, metadata={"help": "API keys for the endpoint, if None, allow all"}
    )

    # 默认模型名称参数，可选字符串类型，默认为 None，用于指定默认的模型
    default_model: Optional[str] = field(
        default=None,
        metadata={"help": "Default model name"},
    )
```