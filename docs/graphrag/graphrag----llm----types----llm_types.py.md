# `.\graphrag\graphrag\llm\types\llm_types.py`

```py
# 引入 TypeAlias 类型别名，用于简化复杂的类型定义
from typing import TypeAlias

# 从当前目录的 llm 模块中导入 LLM 类
from .llm import LLM

# 定义 EmbeddingInput 类型别名，表示为一个字符串列表
EmbeddingInput: TypeAlias = list[str]

# 定义 EmbeddingOutput 类型别名，表示为一个嵌套列表，内部元素为浮点数
EmbeddingOutput: TypeAlias = list[list[float]]

# 定义 CompletionInput 类型别名，表示为一个字符串
CompletionInput: TypeAlias = str

# 定义 CompletionOutput 类型别名，表示为一个字符串
CompletionOutput: TypeAlias = str

# 定义 EmbeddingLLM 类型别名，表示为一个 LLM 泛型类，接受 EmbeddingInput 和 EmbeddingOutput 作为参数
EmbeddingLLM: TypeAlias = LLM[EmbeddingInput, EmbeddingOutput]

# 定义 CompletionLLM 类型别名，表示为一个 LLM 泛型类，接受 CompletionInput 和 CompletionOutput 作为参数
CompletionLLM: TypeAlias = LLM[CompletionInput, CompletionOutput]
```