# `.\DB-GPT-src\dbgpt\serve\rag\config.py`

```py
from dataclasses import dataclass, field
from typing import Optional

from dbgpt.serve.core import BaseServeConfig

# 应用名称
APP_NAME = "rag"
# Serve 应用名称
SERVE_APP_NAME = "dbgpt_rag"
# Serve 应用名称的驼峰形式
SERVE_APP_NAME_HUMP = "dbgpt_rag"
# Serve 配置键前缀
SERVE_CONFIG_KEY_PREFIX = "dbgpt_rag"
# Serve 服务组件名称，使用 Serve 应用名称拼接
SERVE_SERVICE_COMPONENT_NAME = f"{SERVE_APP_NAME}_service"

@dataclass
class ServeConfig(BaseServeConfig):
    """Parameters for the serve command"""

    # API 密钥，用于端点，如果为 None，则允许所有请求
    api_keys: Optional[str] = field(
        default=None, metadata={"help": "API keys for the endpoint, if None, allow all"}
    )

    # 默认用户名称，用于提示
    default_user: Optional[str] = field(
        default=None,
        metadata={"help": "Default user name for prompt"},
    )
    # 默认系统代码，用于提示
    default_sys_code: Optional[str] = field(
        default=None,
        metadata={"help": "Default system code for prompt"},
    )
```