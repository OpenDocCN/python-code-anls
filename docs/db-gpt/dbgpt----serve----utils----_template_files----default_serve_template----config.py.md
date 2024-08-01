# `.\DB-GPT-src\dbgpt\serve\utils\_template_files\default_serve_template\config.py`

```py
# 从 dataclasses 模块导入 dataclass 和 field 函数
# 从 typing 模块导入 Optional 类型提示

# 从 dbgpt.serve.core 模块导入 BaseServeConfig 类
from dbgpt.serve.core import BaseServeConfig

# 定义常量 APP_NAME，使用模板引擎替换后的小写应用名称
APP_NAME = "{__template_app_name__all_lower__}"

# 定义常量 SERVE_APP_NAME，拼接出服务应用的名称，小写形式
SERVE_APP_NAME = "dbgpt_serve_{__template_app_name__all_lower__}"

# 定义常量 SERVE_APP_NAME_HUMP，拼接出服务应用的名称，驼峰命名形式
SERVE_APP_NAME_HUMP = "dbgpt_serve_{__template_app_name__hump__}"

# 定义常量 SERVE_CONFIG_KEY_PREFIX，服务配置键的前缀
SERVE_CONFIG_KEY_PREFIX = "dbgpt.serve.{__template_app_name__all_lower__}."

# 定义常量 SERVE_SERVICE_COMPONENT_NAME，服务组件的名称，包括服务应用名称
SERVE_SERVICE_COMPONENT_NAME = f"{SERVE_APP_NAME}_service"

# 定义常量 SERVER_APP_TABLE_NAME，数据库表格名称，包括服务应用名称的小写形式
# 这个常量表示与服务应用相关联的数据库表格名称
SERVER_APP_TABLE_NAME = "dbgpt_serve_{__template_app_name__all_lower__}"

@dataclass
class ServeConfig(BaseServeConfig):
    """Parameters for the serve command"""

    # TODO: add your own parameters here
    # 定义字段 api_keys，可选字符串类型，默认为 None
    # metadata 中包含一个帮助文档，说明该字段用于存储端点的 API 密钥，如果为 None，则允许所有请求
    api_keys: Optional[str] = field(
        default=None, metadata={"help": "API keys for the endpoint, if None, allow all"}
    )
```