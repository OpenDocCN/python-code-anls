# `.\AutoGPT\autogpts\autogpt\autogpt\core\resource\model_providers\__init__.py`

```py
# 导入 openai 模块中的相关内容
from .openai import (
    OPEN_AI_CHAT_MODELS,  # 导入 OPEN_AI_CHAT_MODELS 变量
    OPEN_AI_EMBEDDING_MODELS,  # 导入 OPEN_AI_EMBEDDING_MODELS 变量
    OPEN_AI_MODELS,  # 导入 OPEN_AI_MODELS 变量
    OpenAIModelName,  # 导入 OpenAIModelName 类
    OpenAIProvider,  # 导入 OpenAIProvider 类
    OpenAISettings,  # 导入 OpenAISettings 类
)
# 导入 schema 模块中的相关内容
from .schema import (
    AssistantChatMessage,  # 导入 AssistantChatMessage 类
    AssistantChatMessageDict,  # 导入 AssistantChatMessageDict 类
    AssistantFunctionCall,  # 导入 AssistantFunctionCall 类
    AssistantFunctionCallDict,  # 导入 AssistantFunctionCallDict 类
    ChatMessage,  # 导入 ChatMessage 类
    ChatModelInfo,  # 导入 ChatModelInfo 类
    ChatModelProvider,  # 导入 ChatModelProvider 类
    ChatModelResponse,  # 导入 ChatModelResponse 类
    CompletionModelFunction,  # 导入 CompletionModelFunction 类
    Embedding,  # 导入 Embedding 类
    EmbeddingModelInfo,  # 导入 EmbeddingModelInfo 类
    EmbeddingModelProvider,  # 导入 EmbeddingModelProvider 类
    EmbeddingModelResponse,  # 导入 EmbeddingModelResponse 类
    ModelInfo,  # 导入 ModelInfo 类
    ModelProvider,  # 导入 ModelProvider 类
    ModelProviderBudget,  # 导入 ModelProviderBudget 类
    ModelProviderCredentials,  # 导入 ModelProviderCredentials 类
    ModelProviderName,  # 导入 ModelProviderName 类
    ModelProviderService,  # 导入 ModelProviderService 类
    ModelProviderSettings,  # 导入 ModelProviderSettings 类
    ModelProviderUsage,  # 导入 ModelProviderUsage 类
    ModelResponse,  # 导入 ModelResponse 类
    ModelTokenizer,  # 导入 ModelTokenizer 类
)

# 定义 __all__ 列表，包含需要导出的模块内容
__all__ = [
    "AssistantChatMessage",  # 导出 AssistantChatMessage 类
    "AssistantChatMessageDict",  # 导出 AssistantChatMessageDict 类
    "AssistantFunctionCall",  # 导出 AssistantFunctionCall 类
    "AssistantFunctionCallDict",  # 导出 AssistantFunctionCallDict 类
    "ChatMessage",  # 导出 ChatMessage 类
    "ChatModelInfo",  # 导出 ChatModelInfo 类
    "ChatModelProvider",  # 导出 ChatModelProvider 类
    "ChatModelResponse",  # 导出 ChatModelResponse 类
    "CompletionModelFunction",  # 导出 CompletionModelFunction 类
    "Embedding",  # 导出 Embedding 类
    "EmbeddingModelInfo",  # 导出 EmbeddingModelInfo 类
    "EmbeddingModelProvider",  # 导出 EmbeddingModelProvider 类
    "EmbeddingModelResponse",  # 导出 EmbeddingModelResponse 类
    "ModelInfo",  # 导出 ModelInfo 类
    "ModelProvider",  # 导出 ModelProvider 类
    "ModelProviderBudget",  # 导出 ModelProviderBudget 类
    "ModelProviderCredentials",  # 导出 ModelProviderCredentials 类
    "ModelProviderName",  # 导出 ModelProviderName 类
    "ModelProviderService",  # 导出 ModelProviderService 类
    "ModelProviderSettings",  # 导出 ModelProviderSettings 类
    "ModelProviderUsage",  # 导出 ModelProviderUsage 类
    "ModelResponse",  # 导出 ModelResponse 类
    "ModelTokenizer",  # 导出 ModelTokenizer 类
    "OPEN_AI_MODELS",  # 导出 OPEN_AI_MODELS 变量
    "OPEN_AI_CHAT_MODELS",  # 导出 OPEN_AI_CHAT_MODELS 变量
    "OPEN_AI_EMBEDDING_MODELS",  # 导出 OPEN_AI_EMBEDDING_MODELS 变量
    "OpenAIModelName",  # 导出 OpenAIModelName 类
    "OpenAIProvider",  # 导出 OpenAIProvider 类
    "OpenAISettings",  # 导出 OpenAISettings 类
]
```