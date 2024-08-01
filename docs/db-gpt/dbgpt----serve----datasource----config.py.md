# `.\DB-GPT-src\dbgpt\serve\datasource\config.py`

```py
# 从 dataclasses 模块中导入 dataclass 和 field 装饰器
# 从 typing 模块中导入 Optional 类型
# 从 dbgpt.serve.core 模块中导入 BaseServeConfig 类
from dataclasses import dataclass, field
from typing import Optional

from dbgpt.serve.core import BaseServeConfig

# 定义常量 APP_NAME 为 "datasource"
APP_NAME = "datasource"
# 定义常量 SERVE_APP_NAME 为 "dbgpt_datasource"
SERVE_APP_NAME = "dbgpt_datasource"
# 定义常量 SERVE_APP_NAME_HUMP 为 "dbgpt_datasource"
SERVE_APP_NAME_HUMP = "dbgpt_datasource"
# 定义常量 SERVE_CONFIG_KEY_PREFIX 为 "dbgpt_datasource"
SERVE_CONFIG_KEY_PREFIX = "dbgpt_datasource"
# 定义常量 SERVE_SERVICE_COMPONENT_NAME 为 "dbgpt_datasource_service"
SERVE_SERVICE_COMPONENT_NAME = f"{SERVE_APP_NAME}_service"

# 定义 ServeConfig 类，继承自 BaseServeConfig 类
@dataclass
class ServeConfig(BaseServeConfig):
    """Parameters for the serve command"""

    # 定义 api_keys 属性，类型为 Optional[str]，默认值为 None，带有 metadata 帮助信息
    api_keys: Optional[str] = field(
        default=None, metadata={"help": "API keys for the endpoint, if None, allow all"}
    )

    # 定义 default_user 属性，类型为 Optional[str]，默认值为 None，带有 metadata 帮助信息
    default_user: Optional[str] = field(
        default=None,
        metadata={"help": "Default user name for prompt"},
    )
    # 定义 default_sys_code 属性，类型为 Optional[str]，默认值为 None，带有 metadata 帮助信息
    default_sys_code: Optional[str] = field(
        default=None,
        metadata={"help": "Default system code for prompt"},
    )
```