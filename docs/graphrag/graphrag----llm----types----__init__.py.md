# `.\graphrag\graphrag\llm\types\__init__.py`

```py
# 版权声明，指明代码版权归属和使用许可
# 从当前目录导入以下模块和类
from .llm import LLM                   # 导入LLM类
from .llm_cache import LLMCache        # 导入LLMCache类
from .llm_callbacks import (           # 导入多个回调函数
    ErrorHandlerFn,                    # 导入错误处理函数
    IsResponseValidFn,                 # 导入响应有效性检查函数
    LLMInvocationFn,                   # 导入LLM调用函数
    OnCacheActionFn,                   # 导入缓存操作回调函数
)
from .llm_config import LLMConfig      # 导入LLM配置类
from .llm_invocation_result import LLMInvocationResult  # 导入LLM调用结果类
from .llm_io import (                  # 导入输入输出相关类
    LLMInput,                          # 导入LLM输入类
    LLMOutput,                         # 导入LLM输出类
)
from .llm_types import (               # 导入多个LLM相关类型
    CompletionInput,                   # 导入完成输入类
    CompletionLLM,                     # 导入完成LLM类
    CompletionOutput,                  # 导入完成输出类
    EmbeddingInput,                    # 导入嵌入输入类
    EmbeddingLLM,                      # 导入嵌入LLM类
    EmbeddingOutput,                   # 导入嵌入输出类
)

# __all__ 列表，指明可以从该模块导入的公共接口
__all__ = [
    "LLM",                             # LLM类
    "CompletionInput",                 # 完成输入类
    "CompletionLLM",                   # 完成LLM类
    "CompletionOutput",                # 完成输出类
    "EmbeddingInput",                  # 嵌入输入类
    "EmbeddingLLM",                    # 嵌入LLM类
    "EmbeddingOutput",                 # 嵌入输出类
    "ErrorHandlerFn",                  # 错误处理函数
    "IsResponseValidFn",               # 响应有效性检查函数
    "LLMCache",                        # LLM缓存类
    "LLMConfig",                       # LLM配置类
    "LLMInput",                        # LLM输入类
    "LLMInvocationFn",                 # LLM调用函数
    "LLMInvocationResult",             # LLM调用结果类
    "LLMOutput",                       # LLM输出类
    "OnCacheActionFn",                 # 缓存操作回调函数
]
```