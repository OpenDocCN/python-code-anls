# `.\DB-GPT-src\dbgpt\core\schema\api.py`

```py
"""API schema module."""

# 导入所需的模块和类
import time
import uuid
from enum import IntEnum
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union

# 导入 Pydantic 相关类和函数
from dbgpt._private.pydantic import BaseModel, Field, model_to_dict

# 定义类型变量 T，用于泛型支持
T = TypeVar("T")


class Result(BaseModel, Generic[T]):
    """Common result entity for API response."""

    # 是否操作成功的标志，True 表示成功，False 表示失败
    success: bool = Field(
        ..., description="Whether it is successful, True: success, False: failure"
    )
    # 错误代码，可选
    err_code: str | None = Field(None, description="Error code")
    # 错误消息，可选
    err_msg: str | None = Field(None, description="Error message")
    # 返回数据，类型为 T，可选
    data: T | None = Field(None, description="Return data")

    @staticmethod
    def succ(data: T) -> "Result[T]":
        """Build a successful result entity.

        Args:
            data (T): Return data

        Returns:
            Result[T]: Result entity
        """
        return Result(success=True, err_code=None, err_msg=None, data=data)

    @staticmethod
    def failed(msg: str, err_code: Optional[str] = "E000X") -> "Result[Any]":
        """Build a failed result entity.

        Args:
            msg (str): Error message
            err_code (Optional[str], optional): Error code. Defaults to "E000X".
        """
        return Result(success=False, err_code=err_code, err_msg=msg, data=None)

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """Convert to dict."""
        return model_to_dict(self, **kwargs)


class APIChatCompletionRequest(BaseModel):
    """Chat completion request entity."""

    # 模型名称
    model: str = Field(..., description="Model name")
    # 消息内容，可以是字符串或字典列表
    messages: Union[str, List[Dict[str, str]]] = Field(..., description="Messages")
    # 温度参数，可选
    temperature: Optional[float] = Field(0.7, description="Temperature")
    # Top-p 参数，可选
    top_p: Optional[float] = Field(1.0, description="Top p")
    # Top-k 参数，可选
    top_k: Optional[int] = Field(-1, description="Top k")
    # 完成次数，可选
    n: Optional[int] = Field(1, description="Number of completions")
    # 最大 token 数，可选
    max_tokens: Optional[int] = Field(None, description="Max tokens")
    # 停止标识，可选
    stop: Optional[Union[str, List[str]]] = Field(None, description="Stop")
    # 是否流式处理，可选
    stream: Optional[bool] = Field(False, description="Stream")
    # 用户标识，可选
    user: Optional[str] = Field(None, description="User")
    # 重复惩罚参数，可选
    repetition_penalty: Optional[float] = Field(1.0, description="Repetition penalty")
    # 频率惩罚参数，可选
    frequency_penalty: Optional[float] = Field(0.0, description="Frequency penalty")
    # 存在性惩罚参数，可选
    presence_penalty: Optional[float] = Field(0.0, description="Presence penalty")


class DeltaMessage(BaseModel):
    """Delta message entity for chat completion response."""

    # 角色，可选
    role: Optional[str] = None
    # 内容，可选
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    """Chat completion response choice entity."""

    # 选择索引
    index: int = Field(..., description="Choice index")
    # Delta 消息
    delta: DeltaMessage = Field(..., description="Delta message")
    # 完成原因，可选，可能是 stop 或 length
    finish_reason: Optional[Literal["stop", "length"]] = Field(
        None, description="Finish reason"
    )


class ChatCompletionStreamResponse(BaseModel):
    """Chat completion response stream entity."""
    
    # 表示聊天完成响应流的实体
    
    id: str = Field(
        default_factory=lambda: f"chatcmpl-{str(uuid.uuid1())}", description="Stream ID"
    )
    # 字段：id，用于唯一标识该聊天完成响应流的ID，
    # 默认工厂函数生成一个基于UUIDv1的唯一ID字符串，
    # 描述为“Stream ID”
    
    created: int = Field(
        default_factory=lambda: int(time.time()), description="Created time"
    )
    # 字段：created，表示该流创建的时间戳（整数类型），
    # 默认工厂函数生成当前时间的时间戳，
    # 描述为“Created time”
    
    model: str = Field(..., description="Model name")
    # 字段：model，表示模型的名称（字符串类型），
    # 由外部提供，必须存在，
    # 描述为“Model name”
    
    choices: List[ChatCompletionResponseStreamChoice] = Field(
        ..., description="Chat completion response choices"
    )
    # 字段：choices，表示聊天完成响应的选择列表（ChatCompletionResponseStreamChoice类型的列表），
    # 由外部提供，必须存在，
    # 描述为“Chat completion response choices”
class ChatMessage(BaseModel):
    """Chat message entity."""

    role: str = Field(..., description="Role of the message")
    content: str = Field(..., description="Content of the message")


class UsageInfo(BaseModel):
    """Usage info entity."""

    prompt_tokens: int = Field(0, description="Prompt tokens")
    total_tokens: int = Field(0, description="Total tokens")
    completion_tokens: Optional[int] = Field(0, description="Completion tokens")


class ChatCompletionResponseChoice(BaseModel):
    """Chat completion response choice entity."""

    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Chat message")
    finish_reason: Optional[Literal["stop", "length"]] = Field(
        None, description="Finish reason"
    )


class ChatCompletionResponse(BaseModel):
    """Chat completion response entity."""

    id: str = Field(
        default_factory=lambda: f"chatcmpl-{str(uuid.uuid1())}", description="Stream ID"
    )
    object: str = "chat.completion"
    created: int = Field(
        default_factory=lambda: int(time.time()), description="Created time"
    )
    model: str = Field(..., description="Model name")
    choices: List[ChatCompletionResponseChoice] = Field(
        ..., description="Chat completion response choices"
    )
    usage: UsageInfo = Field(..., description="Usage info")


class ErrorResponse(BaseModel):
    """Error response entity."""

    object: str = Field("error", description="Object type")
    message: str = Field(..., description="Error message")
    code: int = Field(..., description="Error code")


class EmbeddingsRequest(BaseModel):
    """Embeddings request entity."""

    model: Optional[str] = Field(None, description="Model name")
    engine: Optional[str] = Field(None, description="Engine name")
    input: Union[str, List[Any]] = Field(..., description="Input data")
    user: Optional[str] = Field(None, description="User name")
    encoding_format: Optional[str] = Field(None, description="Encoding format")


class EmbeddingsResponse(BaseModel):
    """Embeddings response entity."""

    object: str = Field("list", description="Object type")
    data: List[Dict[str, Any]] = Field(..., description="Data list")
    model: str = Field(..., description="Model name")
    usage: UsageInfo = Field(..., description="Usage info")


class RelevanceRequest(BaseModel):
    """Relevance request entity."""

    model: str = Field(..., description="Rerank model name")
    query: str = Field(..., description="Query text")
    documents: List[str] = Field(..., description="Document texts")


class RelevanceResponse(BaseModel):
    """Relevance response entity."""

    object: str = Field("list", description="Object type")
    model: str = Field(..., description="Rerank model name")
    data: List[float] = Field(..., description="Data list, relevance scores")
    usage: UsageInfo = Field(..., description="Usage info")


class ModelPermission(BaseModel):
    """Model permission entity."""
    """Model permission entity."""
    
    # 唯一标识符，使用 UUID v1 自动生成，作为权限的唯一 ID
    id: str = Field(
        default_factory=lambda: f"modelperm-{str(uuid.uuid1())}",
        description="Permission ID",
    )
    # 表示这是一个模型权限对象
    object: str = Field("model_permission", description="Object type")
    # 记录对象创建的时间戳（单位：秒）
    created: int = Field(
        default_factory=lambda: int(time.time()), description="Created time"
    )
    # 是否允许创建引擎
    allow_create_engine: bool = Field(False, description="Allow create engine")
    # 是否允许采样
    allow_sampling: bool = Field(True, description="Allow sampling")
    # 是否允许记录 logprobs
    allow_logprobs: bool = Field(True, description="Allow logprobs")
    # 是否允许搜索索引
    allow_search_indices: bool = Field(True, description="Allow search indices")
    # 是否允许查看权限
    allow_view: bool = Field(True, description="Allow view")
    # 是否允许进行精细调整
    allow_fine_tuning: bool = Field(False, description="Allow fine tuning")
    # 适用的组织，默认为所有组织（通配符表示）
    organization: str = Field("*", description="Organization")
    # 可选的组名称，可以为 None
    group: Optional[str] = Field(None, description="Group")
    # 是否是阻塞权限，如果为 True，则可能会阻止某些操作
    is_blocking: bool = Field(False, description="Is blocking")
# 定义 ModelCard 类，继承自 BaseModel
class ModelCard(BaseModel):
    """Model card entity."""

    # 定义 id 字段，表示模型的 ID
    id: str = Field(..., description="Model ID")
    # 定义 object 字段，默认值为 "model"，表示对象类型为模型
    object: str = Field("model", description="Object type")
    # 定义 created 字段，默认值为当前时间戳，表示创建时间
    created: int = Field(
        default_factory=lambda: int(time.time()), description="Created time"
    )
    # 定义 owned_by 字段，默认值为 "DB-GPT"，表示所有者
    owned_by: str = Field("DB-GPT", description="Owned by")
    # 定义 root 字段，可选值为字符串，表示根节点
    root: Optional[str] = Field(None, description="Root")
    # 定义 parent 字段，可选值为字符串，表示父节点
    parent: Optional[str] = Field(None, description="Parent")
    # 定义 permission 字段，列表类型，表示权限
    permission: List[ModelPermission] = Field(
        default_factory=list, description="Permission"
    )


# 定义 ModelList 类，继承自 BaseModel
class ModelList(BaseModel):
    """Model list entity."""

    # 定义 object 字段，默认值为 "list"，表示对象类型为列表
    object: str = Field("list", description="Object type")
    # 定义 data 字段，列表类型，表示模型列表数据
    data: List[ModelCard] = Field(default_factory=list, description="Model list data")


# 定义 ErrorCode 枚举类，表示错误代码枚举
class ErrorCode(IntEnum):
    """Error code enumeration.

    https://platform.openai.com/docs/guides/error-codes/api-errors.

    Adapted from fastchat.constants.
    """

    # 定义不同的错误代码及其对应的值
    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006
```