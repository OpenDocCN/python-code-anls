# `.\DB-GPT-src\dbgpt\serve\conversation\api\schemas.py`

```py
# Define your Pydantic schemas here
# 导入必要的模块和类
from typing import Any, Dict, Optional

from dbgpt._private.pydantic import BaseModel, ConfigDict, Field, model_to_dict

from ..config import SERVE_APP_NAME_HUMP

# 定义一个名为 ServeRequest 的 Pydantic 模型，表示会话请求
class ServeRequest(BaseModel):
    """Conversation request model"""

    # 配置模型的标题
    model_config = ConfigDict(title=f"ServeRequest for {SERVE_APP_NAME_HUMP}")

    # 用于查询的字段
    chat_mode: str = Field(
        default=None,
        description="The chat mode.",
        examples=[
            "chat_normal",
        ],
    )
    # 可选的会话 ID
    conv_uid: Optional[str] = Field(
        default=None,
        description="The conversation uid.",
        examples=[
            "5e7100bc-9017-11ee-9876-8fe019728d79",
        ],
    )
    # 可选的用户名
    user_name: Optional[str] = Field(
        default=None,
        description="The user name.",
        examples=[
            "zhangsan",
        ],
    )
    # 可选的系统代码
    sys_code: Optional[str] = Field(
        default=None,
        description="The system code.",
        examples=[
            "dbgpt",
        ],
    )

    # 将模型转换为字典的方法
    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return model_to_dict(self, **kwargs)


# 定义一个名为 ServerResponse 的 Pydantic 模型，表示会话响应
class ServerResponse(BaseModel):
    """Conversation response model"""

    # 配置模型的标题和受保护的命名空间
    model_config = ConfigDict(
        title=f"ServerResponse for {SERVE_APP_NAME_HUMP}", protected_namespaces=()
    )

    # 必填的会话 ID
    conv_uid: str = Field(
        ...,
        description="The conversation uid.",
        examples=[
            "5e7100bc-9017-11ee-9876-8fe019728d79",
        ],
    )
    # 必填的用户输入，作为会话摘要返回
    user_input: str = Field(
        ...,
        description="The user input, we return it as the summary the conversation.",
        examples=[
            "Hello world",
        ],
    )
    # 必填的聊天模式
    chat_mode: str = Field(
        ...,
        description="The chat mode.",
        examples=[
            "chat_normal",
        ],
    )
    # 可选的选择参数
    select_param: Optional[str] = Field(
        default=None,
        description="The select param.",
        examples=[
            "my_knowledge_space_name",
        ],
    )
    # 可选的模型名称
    model_name: Optional[str] = Field(
        default=None,
        description="The model name.",
        examples=[
            "vicuna-13b-v1.5",
        ],
    )
    # 可选的用户名
    user_name: Optional[str] = Field(
        default=None,
        description="The user name.",
        examples=[
            "zhangsan",
        ],
    )
    # 可选的系统代码
    sys_code: Optional[str] = Field(
        default=None,
        description="The system code.",
        examples=[
            "dbgpt",
        ],
    )

    # 将模型转换为字典的方法
    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return model_to_dict(self, **kwargs)


# 定义一个名为 MessageVo 的 Pydantic 模型，表示消息对象
class MessageVo(BaseModel):
    # 配置模型的受保护的命名空间
    model_config = ConfigDict(protected_namespaces=())
    # 必填的角色字段，表示当前消息的发送者角色
    role: str = Field(
        ...,
        description="The role that sends out the current message.",
        examples=["human", "ai", "view"],
    )
    context: str = Field(
        ...,
        description="The current message content.",
        examples=[
            "Hello",
            "Hi, how are you?",
        ],
    )

    # 表示当前消息的内容，是一个字符串类型的字段
    order: int = Field(
        ...,
        description="The current message order.",
        examples=[
            1,
            2,
        ],
    )

    # 表示当前消息的顺序，是一个整数类型的字段
    time_stamp: Optional[Any] = Field(
        default=None,
        description="The current message time stamp.",
        examples=[
            "2023-01-07 09:00:00",
        ],
    )

    # 表示当前消息的时间戳，可以为空值，是一个可选的字段，可以是任意类型的数据
    model_name: Optional[str] = Field(
        default=None,
        description="The model name.",
        examples=[
            "vicuna-13b-v1.5",
        ],
    )

    # 表示模型的名称，可以为空，是一个字符串类型的字段
    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        # 将模型对象转换为字典的方法
        return model_to_dict(self, **kwargs)
```