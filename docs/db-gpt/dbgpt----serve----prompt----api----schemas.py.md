# `.\DB-GPT-src\dbgpt\serve\prompt\api\schemas.py`

```py
# 定义 Pydantic 模型的基类 Base 模型，用于数据验证和文档生成
from typing import Optional

# 导入 Pydantic 的基类 BaseModel 和其他必要的类
from dbgpt._private.pydantic import BaseModel, ConfigDict, Field

# 从上层模块导入 SERVE_APP_NAME_HUMP 用于配置名称
from ..config import SERVE_APP_NAME_HUMP

# 定义 ServeRequest 类，继承自 Pydantic 的 BaseModel
class ServeRequest(BaseModel):
    """Prompt request model."""

    # 配置模型的配置字典，包括标题，用于生成文档
    model_config = ConfigDict(title=f"ServeRequest for {SERVE_APP_NAME_HUMP}")

    # 聊天场景，可选字符串类型，用 Field 进行详细配置
    chat_scene: Optional[str] = Field(
        None,
        description="The chat scene, e.g. chat_with_db_execute, chat_excel, chat_with_db_qa.",
        examples=["chat_with_db_execute", "chat_excel", "chat_with_db_qa"],
    )

    # 子聊天场景，可选字符串类型，用 Field 进行详细配置
    sub_chat_scene: Optional[str] = Field(
        None,
        description="The sub chat scene.",
        examples=["sub_scene_1", "sub_scene_2", "sub_scene_3"],
    )

    # 提示类型，可选字符串类型，用 Field 进行详细配置
    prompt_type: Optional[str] = Field(
        None,
        description="The prompt type, either common or private.",
        examples=["common", "private"],
    )

    # 提示名称，可选字符串类型，用 Field 进行详细配置
    prompt_name: Optional[str] = Field(
        None,
        description="The prompt name.",
        examples=["code_assistant", "joker", "data_analysis_expert"],
    )

    # 提示内容，可选字符串类型，用 Field 进行详细配置
    content: Optional[str] = Field(
        None,
        description="The prompt content.",
        examples=[
            "Write a qsort function in python",
            "Tell me a joke about AI",
            "You are a data analysis expert.",
        ],
    )

    # 提示描述，可选字符串类型，用 Field 进行详细配置
    prompt_desc: Optional[str] = Field(
        None,
        description="The prompt description.",
        examples=[
            "This is a prompt for code assistant.",
            "This is a prompt for joker.",
            "This is a prompt for data analysis expert.",
        ],
    )

    # 用户名，可选字符串类型，用 Field 进行详细配置
    user_name: Optional[str] = Field(
        None,
        description="The user name.",
        examples=["zhangsan", "lisi", "wangwu"],
    )

    # 系统代码，可选字符串类型，用 Field 进行详细配置
    sys_code: Optional[str] = Field(
        None,
        description="The system code.",
        examples=["dbgpt", "auth_manager", "data_platform"],
    )


# 定义 ServerResponse 类，继承自 ServeRequest，表示服务器返回的响应模型
class ServerResponse(ServeRequest):
    """Prompt response model"""

    # 配置模型的配置字典，包括标题，用于生成文档
    model_config = ConfigDict(title=f"ServerResponse for {SERVE_APP_NAME_HUMP}")

    # 提示 ID，可选整数类型，用 Field 进行详细配置
    id: Optional[int] = Field(
        None,
        description="The prompt id.",
        examples=[1, 2, 3],
    )

    # 提示创建时间，可选字符串类型，用 Field 进行详细配置
    gmt_created: Optional[str] = Field(
        None,
        description="The prompt created time.",
        examples=["2021-08-01 12:00:00", "2021-08-01 12:00:01", "2021-08-01 12:00:02"],
    )

    # 提示修改时间，可选字符串类型，用 Field 进行详细配置
    gmt_modified: Optional[str] = Field(
        None,
        description="The prompt modified time.",
        examples=["2021-08-01 12:00:00", "2021-08-01 12:00:01", "2021-08-01 12:00:02"],
    )
```