# `.\DB-GPT-src\dbgpt\serve\utils\_template_files\default_serve_template\api\schemas.py`

```py
# 导入必要的模块和类
from typing import Any, Dict

from dbgpt._private.pydantic import BaseModel, ConfigDict, Field, model_to_dict
# 导入应用的配置信息中的服务应用名
from ..config import SERVE_APP_NAME_HUMP


class ServeRequest(BaseModel):
    """{__template_app_name__hump__} request model"""

    # TODO 定义自己的字段在这里

    # 定义模型配置，设置标题为服务应用名的 ServeRequest
    model_config = ConfigDict(title=f"ServeRequest for {SERVE_APP_NAME_HUMP}")

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        # 调用通用函数 model_to_dict 将模型转换为字典并返回
        return model_to_dict(self, **kwargs)


class ServerResponse(BaseModel):
    """{__template_app_name__hump__} response model"""

    # TODO 定义自己的字段在这里

    # 定义模型配置，设置标题为服务应用名的 ServerResponse
    model_config = ConfigDict(title=f"ServerResponse for {SERVE_APP_NAME_HUMP}")

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        # 调用通用函数 model_to_dict 将模型转换为字典并返回
        return model_to_dict(self, **kwargs)
```