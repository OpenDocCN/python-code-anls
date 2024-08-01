# `.\DB-GPT-src\dbgpt\app\openapi\api_view_model.py`

```py
from typing import Any, Dict, Generic, Optional, TypeVar

from dbgpt._private.pydantic import BaseModel, ConfigDict, Field, model_to_dict

T = TypeVar("T")


class Result(BaseModel, Generic[T]):
    success: bool
    err_code: Optional[str] = None  # 错误代码，可选
    err_msg: Optional[str] = None   # 错误消息，可选
    data: Optional[T] = None        # 数据对象，可选

    @classmethod
    def succ(cls, data: T):
        """
        创建一个成功的 Result 对象

        Parameters:
        - data: 结果数据

        Returns:
        - 成功的 Result 对象
        """
        return Result(success=True, err_code=None, err_msg=None, data=data)

    @classmethod
    def failed(cls, code: str = "E000X", msg=None):
        """
        创建一个失败的 Result 对象

        Parameters:
        - code: 错误代码，默认为 "E000X"
        - msg: 错误消息，可选

        Returns:
        - 失败的 Result 对象
        """
        return Result(success=False, err_code=code, err_msg=msg, data=None)

    def to_dict(self) -> Dict[str, Any]:
        """
        将对象转换为字典表示

        Returns:
        - 包含对象属性的字典
        """
        return model_to_dict(self)


class ChatSceneVo(BaseModel):
    chat_scene: str = Field(..., description="chat_scene")  # 聊天场景
    scene_name: str = Field(..., description="chat_scene name show for user")  # 用户显示的聊天场景名称
    scene_describe: str = Field("", description="chat_scene describe ")  # 聊天场景描述
    param_title: str = Field("", description="chat_scene required parameter title")  # 聊天场景所需参数标题
    show_disable: bool = Field(False, description="chat_scene show disable")  # 是否禁用聊天场景的显示


class ConversationVo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # 模型配置

    """
    dialogue_uid
    """
    conv_uid: str = ""  # 对话唯一标识

    """ 
    user input 
    """
    user_input: str = ""  # 用户输入内容

    """
    user
    """
    user_name: Optional[str] = Field(None, description="user name")  # 用户名

    """ 
    the scene of chat 
    """
    chat_mode: str = ""  # 聊天场景模式

    """
    chat scene select param 
    """
    select_param: Optional[str] = Field(None, description="chat scene select param")  # 聊天场景选择参数

    """
    llm model name
    """
    model_name: Optional[str] = Field(None, description="llm model name")  # LLM 模型名称

    """Used to control whether the content is returned incrementally or in full each time. 
    If this parameter is not provided, the default is full return.
    """
    incremental: bool = False  # 控制返回内容是每次增量返回还是完整返回，默认完整返回

    sys_code: Optional[str] = Field(None, description="System code")  # 系统代码


class MessageVo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # 模型配置

    """
    role that sends out the current message
    """
    role: str  # 发送当前消息的角色

    """
    current message 
    """
    context: str  # 当前消息内容

    """ message postion order """
    order: int  # 消息的顺序

    """
    time the current message was sent 
    """
    time_stamp: Optional[Any] = Field(
        None, description="time the current message was sent"
    )  # 当前消息发送时间戳

    """
    model_name
    """
    model_name: str  # 模型名称
```