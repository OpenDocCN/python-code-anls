# `.\DB-GPT-src\dbgpt\core\__init__.py`

```py
# 导入 dbgpt 核心模块中的缓存相关接口和类
from dbgpt.core.interface.cache import (
    CacheClient,     # 缓存客户端接口
    CacheConfig,     # 缓存配置类
    CacheKey,        # 缓存键类
    CachePolicy,     # 缓存策略类
    CacheValue,      # 缓存值类
)

# 导入 dbgpt 核心模块中的嵌入式和重新排名嵌入式接口
from dbgpt.core.interface.embeddings import Embeddings, RerankEmbeddings  # noqa: F401

# 导入 dbgpt 核心模块中的知识相关接口
from dbgpt.core.interface.knowledge import Chunk, Document  # noqa: F401

# 导入 dbgpt 核心模块中的语言模型接口
from dbgpt.core.interface.llm import (
    DefaultMessageConverter,   # 默认消息转换器类
    LLMClient,                 # 语言模型客户端类
    MessageConverter,          # 消息转换器接口
    ModelExtraMedata,          # 模型额外元数据类
    ModelInferenceMetrics,     # 模型推理指标类
    ModelMetadata,             # 模型元数据类
    ModelOutput,               # 模型输出类
    ModelRequest,              # 模型请求类
    ModelRequestContext,       # 模型请求上下文类
)

# 导入 dbgpt 核心模块中的消息相关接口
from dbgpt.core.interface.message import (
    AIMessage,                  # AI 消息类
    BaseMessage,                # 基础消息类
    ConversationIdentifier,     # 会话标识类
    HumanMessage,               # 人类消息类
    MessageIdentifier,          # 消息标识类
    MessageStorageItem,         # 消息存储项类
    ModelMessage,               # 模型消息类
    ModelMessageRoleType,       # 模型消息角色类型枚举
    OnceConversation,           # 单次会话类
    StorageConversation,        # 存储会话类
    SystemMessage,              # 系统消息类
)

# 导入 dbgpt 核心模块中的输出解析器相关接口
from dbgpt.core.interface.output_parser import (
    BaseOutputParser,           # 基础输出解析器类
    SQLOutputParser,            # SQL 输出解析器类
)

# 导入 dbgpt 核心模块中的提示模板相关接口
from dbgpt.core.interface.prompt import (
    BasePromptTemplate,         # 基础提示模板类
    ChatPromptTemplate,         # 聊天提示模板类
    HumanPromptTemplate,        # 人类提示模板类
    MessagesPlaceholder,        # 消息占位符类
    PromptManager,              # 提示管理器类
    PromptTemplate,             # 提示模板类
    StoragePromptTemplate,      # 存储提示模板类
    SystemPromptTemplate,       # 系统提示模板类
)

# 导入 dbgpt 核心模块中的序列化相关接口
from dbgpt.core.interface.serialization import Serializable, Serializer  # noqa: F401

# 导入 dbgpt 核心模块中的存储相关接口
from dbgpt.core.interface.storage import (
    DefaultStorageItemAdapter,  # 默认存储项适配器类
    InMemoryStorage,            # 内存存储类
    QuerySpec,                  # 查询规范类
    ResourceIdentifier,         # 资源标识符类
    StorageError,               # 存储错误类
    StorageInterface,           # 存储接口类
    StorageItem,                # 存储项类
    StorageItemAdapter,         # 存储项适配器类
)

# 定义模块的公开接口列表
__ALL__ = [
    "ModelInferenceMetrics",
    "ModelRequest",
    "ModelRequestContext",
    "ModelOutput",
    "ModelMetadata",
    "ModelMessage",
    "LLMClient",
    "ModelMessageRoleType",
    "ModelExtraMedata",
    "MessageConverter",
    "DefaultMessageConverter",
    "OnceConversation",
    "StorageConversation",
    "BaseMessage",
    "SystemMessage",
    "AIMessage",
    "HumanMessage",
    "MessageStorageItem",
    "ConversationIdentifier",
    "MessageIdentifier",
    "PromptTemplate",
    "PromptManager",
    "StoragePromptTemplate",
    "BasePromptTemplate",
    "ChatPromptTemplate",
    "MessagesPlaceholder",
    "SystemPromptTemplate",
    "HumanPromptTemplate",
    "BaseOutputParser",
    "SQLOutputParser",
    "Serializable",
    "Serializer",
    "CacheKey",
    "CacheValue",
    "CacheClient",
    "CachePolicy",
    "CacheConfig",
    "ResourceIdentifier",
    "StorageItem",
    "StorageItemAdapter",
    "StorageInterface",
    "InMemoryStorage",
    "DefaultStorageItemAdapter",
    "QuerySpec",
    "StorageError",
    "Embeddings",
    "RerankEmbeddings",
    "Chunk",
    "Document",
]
```