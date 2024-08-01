# `.\DB-GPT-src\dbgpt\serve\prompt\config.py`

```py
# 导入所需模块和类
from dataclasses import dataclass, field
from typing import Optional

# 导入自定义模块中的类
from dbgpt.serve.core import BaseServeConfig

# 定义应用名称常量
APP_NAME = "prompt"
# 定义服务应用名称常量
SERVE_APP_NAME = "dbgpt_serve_prompt"
# 定义帕斯卡命名风格的服务应用名称常量
SERVE_APP_NAME_HUMP = "dbgpt_serve_Prompt"
# 定义服务配置键前缀常量
SERVE_CONFIG_KEY_PREFIX = "dbgpt.serve.prompt."
# 定义服务器应用表名称常量
SERVER_APP_TABLE_NAME = "dbgpt_serve_prompt"

# 定义ServeConfig类，继承自BaseServeConfig类，用于存储服务配置参数
@dataclass
class ServeConfig(BaseServeConfig):
    """Parameters for the serve command"""

    # API密钥参数，可选字符串，默认为None，带有帮助元数据说明
    api_keys: Optional[str] = field(
        default=None, metadata={"help": "API keys for the endpoint, if None, allow all"}
    )

    # 默认用户名称参数，可选字符串，默认为None，带有帮助元数据说明
    default_user: Optional[str] = field(
        default=None,
        metadata={"help": "Default user name for prompt"},
    )
    # 默认系统代码参数，可选字符串，默认为None，带有帮助元数据说明
    default_sys_code: Optional[str] = field(
        default=None,
        metadata={"help": "Default system code for prompt"},
    )
```