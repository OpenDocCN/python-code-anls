# `.\AutoGPT\autogpts\autogpt\autogpt\core\resource\model_providers\schema.py`

```py
# 导入必要的模块和类
import abc
import enum
import math
from typing import (
    Callable,
    ClassVar,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
)

# 导入 pydantic 模块中的 BaseModel, Field, SecretStr, validator
from pydantic import BaseModel, Field, SecretStr, validator

# 导入自定义模块中的 SystemConfiguration, UserConfigurable
from autogpt.core.configuration import SystemConfiguration, UserConfigurable
# 导入自定义模块中的 Embedding, ProviderBudget, ProviderCredentials, ProviderSettings, ProviderUsage, ResourceType
from autogpt.core.resource.schema import (
    Embedding,
    ProviderBudget,
    ProviderCredentials,
    ProviderSettings,
    ProviderUsage,
    ResourceType,
)
# 导入自定义模块中的 JSONSchema
from autogpt.core.utils.json_schema import JSONSchema

# 定义 ModelProviderService 枚举类，描述模型提供的服务类型
class ModelProviderService(str, enum.Enum):
    EMBEDDING = "embedding"
    CHAT = "chat_completion"
    TEXT = "text_completion"

# 定义 ModelProviderName 枚举类，描述模型提供者的名称
class ModelProviderName(str, enum.Enum):
    OPENAI = "openai"

# 定义 ChatMessage 类，描述聊天消息
class ChatMessage(BaseModel):
    # 定义 Role 枚举类，描述消息的角色
    class Role(str, enum.Enum):
        USER = "user"
        SYSTEM = "system"
        ASSISTANT = "assistant"
        FUNCTION = "function"
        """May be used for the return value of function calls"""

    # 消息的角色
    role: Role
    # 消息内容
    content: str

    # 静态方法，创建助手消息
    @staticmethod
    def assistant(content: str) -> "ChatMessage":
        return ChatMessage(role=ChatMessage.Role.ASSISTANT, content=content)

    # 静态方法，创建用户消息
    @staticmethod
    def user(content: str) -> "ChatMessage":
        return ChatMessage(role=ChatMessage.Role.USER, content=content)

    # 静态方法，创建系统消息
    @staticmethod
    def system(content: str) -> "ChatMessage":
        return ChatMessage(role=ChatMessage.Role.SYSTEM, content=content)

# 定义 ChatMessageDict 类型字典，描述聊天消息的结构
class ChatMessageDict(TypedDict):
    role: str
    content: str

# 定义 AssistantFunctionCall 类，描述助手函数调用
class AssistantFunctionCall(BaseModel):
    name: str
    arguments: str

# 定义 AssistantFunctionCallDict 类型字典，描述助手函数调用的结构
class AssistantFunctionCallDict(TypedDict):
    name: str
    arguments: str

# 定义 AssistantToolCall 类，描述助手工具调用
class AssistantToolCall(BaseModel):
    # id: str
    type: Literal["function"]
    function: AssistantFunctionCall

# 定义 AssistantToolCallDict 类型字典，描述助手工具调用的结构
class AssistantToolCallDict(TypedDict):
    # id: str
    type: Literal["function"]
    function: AssistantFunctionCallDict
# 定义一个助手聊天消息类，继承自ChatMessage类
class AssistantChatMessage(ChatMessage):
    # 角色为助手
    role: Literal["assistant"] = "assistant"
    # 消息内容可选
    content: Optional[str]
    # 工具调用列表可选
    tool_calls: Optional[list[AssistantToolCall]]

# 定义一个助手聊天消息字典类，继承自TypedDict
class AssistantChatMessageDict(TypedDict, total=False):
    # 角色为字符串
    role: str
    # 内容为字符串
    content: str
    # 工具调用列表为AssistantToolCallDict类型的列表
    tool_calls: list[AssistantToolCallDict]

# 定义一个用于LLM可调用函数的通用表示对象
class CompletionModelFunction(BaseModel):
    """General representation object for LLM-callable functions."""

    # 函数名
    name: str
    # 描述
    description: str
    # 参数为字符串到JSONSchema对象的字典
    parameters: dict[str, "JSONSchema"]

    @property
    def schema(self) -> dict[str, str | dict | list]:
        """Returns an OpenAI-consumable function specification"""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    # 遍历参数字典，将参数名和参数对象转换为字典
                    name: param.to_dict() for name, param in self.parameters.items()
                },
                "required": [
                    # 获取必需参数的名称列表
                    name for name, param in self.parameters.items() if param.required
                ],
            },
        }

    @staticmethod
    def parse(schema: dict) -> "CompletionModelFunction":
        # 解析函数对象
        return CompletionModelFunction(
            name=schema["name"],
            description=schema["description"],
            parameters=JSONSchema.parse_properties(schema["parameters"]),
        )

    def fmt_line(self) -> str:
        # 格式化输出函数信息
        params = ", ".join(
            f"{name}{'?' if not p.required else ''}: " f"{p.typescript_type}"
            for name, p in self.parameters.items()
        )
        return f"{self.name}: {self.description}. Params: ({params})"

# 模型信息结构体
class ModelInfo(BaseModel):
    """Struct for model information.

    Would be lovely to eventually get this directly from APIs, but needs to be
    scraped from websites for now.
    """

    # 模型名称
    name: str
    # 服务提供者
    service: ModelProviderService
    # 提供者名称
    provider_name: ModelProviderName
    # 提示令牌成本，默认为0.0
    prompt_token_cost: float = 0.0
    # 完成令牌成本，默认为0.0
    completion_token_cost: float = 0.0
class ModelResponse(BaseModel):
    """定义了一个模型响应的标准结构。"""

    prompt_tokens_used: int  # 用于记录使用的提示令牌数量
    completion_tokens_used: int  # 用于记录使用的完成令牌数量
    model_info: ModelInfo  # 包含模型信息的对象


class ModelProviderConfiguration(SystemConfiguration):
    """模型提供者的配置信息。"""

    retries_per_request: int = UserConfigurable()  # 每个请求的重试次数
    extra_request_headers: dict[str, str] = Field(default_factory=dict)  # 额外的请求头信息


class ModelProviderCredentials(ProviderCredentials):
    """模型提供者的凭证信息。"""

    api_key: SecretStr | None = UserConfigurable(default=None)  # API密钥
    api_type: SecretStr | None = UserConfigurable(default=None)  # API类型
    api_base: SecretStr | None = UserConfigurable(default=None)  # API基础地址
    api_version: SecretStr | None = UserConfigurable(default=None)  # API版本
    deployment_id: SecretStr | None = UserConfigurable(default=None)  # 部署ID

    class Config:
        extra = "ignore"  # 配置类，忽略额外的字段


class ModelProviderUsage(ProviderUsage):
    """特定模型提供者的使用情况。"""

    completion_tokens: int = 0  # 完成令牌数量
    prompt_tokens: int = 0  # 提示令牌数量
    total_tokens: int = 0  # 总令牌数量

    def update_usage(
        self,
        model_response: ModelResponse,
    ) -> None:
        """更新使用情况。"""
        self.completion_tokens += model_response.completion_tokens_used  # 更新完成令牌数量
        self.prompt_tokens += model_response.prompt_tokens_used  # 更新提示令牌数量
        self.total_tokens += (
            model_response.completion_tokens_used + model_response.prompt_tokens_used
        )  # 更新总令牌数量


class ModelProviderBudget(ProviderBudget):
    """模型提供者的预算信息。"""

    total_budget: float = UserConfigurable()  # 总预算
    total_cost: float  # 总花费
    remaining_budget: float  # 剩余预算
    usage: ModelProviderUsage  # 使用情况

    def update_usage_and_cost(
        self,
        model_response: ModelResponse,
    ) -> float:
        """更新提供商的使用情况和成本。

        Returns:
            float: 给定模型响应的（计算得到的）成本。
        """
        # 获取模型信息
        model_info = model_response.model_info
        # 更新使用情况
        self.usage.update_usage(model_response)
        # 计算产生的成本
        incurred_cost = (
            model_response.completion_tokens_used * model_info.completion_token_cost
            + model_response.prompt_tokens_used * model_info.prompt_token_cost
        )
        # 更新总成本
        self.total_cost += incurred_cost
        # 更新剩余预算
        self.remaining_budget -= incurred_cost
        # 返回产生的成本
        return incurred_cost
# 定义模型提供者设置类，继承自ProviderSettings类
class ModelProviderSettings(ProviderSettings):
    # 资源类型为模型
    resource_type: ResourceType = ResourceType.MODEL
    # 模型提供者配置
    configuration: ModelProviderConfiguration
    # 模型提供者凭证
    credentials: ModelProviderCredentials
    # 模型提供者预算
    budget: ModelProviderBudget

# 定义抽象基类ModelProvider，用于抽象特定模型提供者的细节
class ModelProvider(abc.ABC):
    """A ModelProvider abstracts the details of a particular provider of models."""

    # 默认设置为ModelProviderSettings类的类变量
    default_settings: ClassVar[ModelProviderSettings]

    # 模型提供者预算
    _budget: Optional[ModelProviderBudget]
    # 模型提供者配置
    _configuration: ModelProviderConfiguration

    # 抽象方法，计算文本中的标记数
    @abc.abstractmethod
    def count_tokens(self, text: str, model_name: str) -> int:
        ...

    # 抽象方法，获取特定模型的分词器
    @abc.abstractmethod
    def get_tokenizer(self, model_name: str) -> "ModelTokenizer":
        ...

    # 抽象方法，获取特定模型的标记限制
    @abc.abstractmethod
    def get_token_limit(self, model_name: str) -> int:
        ...

    # 获取已发生成本
    def get_incurred_cost(self) -> float:
        # 如果有预算，则返回总成本
        if self._budget:
            return self._budget.total_cost
        # 否则返回0
        return 0

    # 获取剩余预算
    def get_remaining_budget(self) -> float:
        # 如果有预算，则返回剩余预算
        if self._budget:
            return self._budget.remaining_budget
        # 否则返回正无穷
        return math.inf

# 定义模型分词器协议
class ModelTokenizer(Protocol):
    """A ModelTokenizer provides tokenization specific to a model."""

    # 抽象方法，对文本进行编码
    @abc.abstractmethod
    def encode(self, text: str) -> list:
        ...

    # 抽象方法，对标记进行解码
    @abc.abstractmethod
    def decode(self, tokens: list) -> str:
        ...


####################
# 嵌入模型 #
####################

# 定义嵌入模型信息结构
class EmbeddingModelInfo(ModelInfo):
    """Struct for embedding model information."""

    # LLM服务为嵌入模型
    llm_service = ModelProviderService.EMBEDDING
    # 最大标记数
    max_tokens: int
    # 嵌入维度
    embedding_dimensions: int

# 定义嵌入模型响应结构
class EmbeddingModelResponse(ModelResponse):
    """Standard response struct for a response from an embedding model."""

    # 嵌入向量，默认为空列表
    embedding: Embedding = Field(default_factory=list)

    # 验证器，验证不应使用完成标记
    @classmethod
    @validator("completion_tokens_used")
    def _verify_no_completion_tokens_used(cls, v):
        if v > 0:
            raise ValueError("Embeddings should not have completion tokens used.")
        return v
# 定义一个继承自 ModelProvider 的 EmbeddingModelProvider 类
class EmbeddingModelProvider(ModelProvider):
    # 声明一个异步抽象方法，用于创建嵌入模型
    @abc.abstractmethod
    async def create_embedding(
        self,
        text: str,
        model_name: str,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs,
    ) -> EmbeddingModelResponse:
        ...


###############
# Chat Models #
###############


# 定义一个继承自 ModelInfo 的 ChatModelInfo 类，用于存储语言模型信息
class ChatModelInfo(ModelInfo):
    """Struct for language model information."""
    
    # 指定语言模型服务类型为 CHAT
    llm_service = ModelProviderService.CHAT
    # 最大 token 数
    max_tokens: int
    # 是否具有函数调用 API，默认为 False
    has_function_call_api: bool = False


# 定义一个泛型类 ChatModelResponse，用于存储语言模型的响应
_T = TypeVar("_T")
class ChatModelResponse(ModelResponse, Generic[_T]):
    """Standard response struct for a response from a language model."""
    
    # 语言模型的响应
    response: AssistantChatMessage
    # 解析后的结果，默认为 None
    parsed_result: _T = None


# 定义一个继承自 ModelProvider 的 ChatModelProvider 类
class ChatModelProvider(ModelProvider):
    # 声明一个抽象方法，用于计算消息的 token 数
    @abc.abstractmethod
    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: str,
    ) -> int:
        ...

    # 声明一个异步抽象方法，用于创建聊天完成
    @abc.abstractmethod
    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: str,
        completion_parser: Callable[[AssistantChatMessage], _T] = lambda _: None,
        functions: Optional[list[CompletionModelFunction]] = None,
        **kwargs,
    ) -> ChatModelResponse[_T]:
        ...
```