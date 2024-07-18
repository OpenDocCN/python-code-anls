# `.\graphrag\graphrag\config\models\text_embedding_config.py`

```py
# 版权声明及许可声明，指明代码版权及使用许可
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""默认配置的参数化设置。"""

# 导入 Pydantic 中的 Field 类
from pydantic import Field

# 导入默认配置文件和文本嵌入目标枚举
import graphrag.config.defaults as defs
from graphrag.config.enums import TextEmbeddingTarget

# 导入 LLMConfig 类
from .llm_config import LLMConfig


class TextEmbeddingConfig(LLMConfig):
    """文本嵌入配置部分，继承自 LLMConfig。"""

    # 批处理大小，用于控制每个批次的样本数
    batch_size: int = Field(
        description="要使用的批处理大小。",
        default=defs.EMBEDDING_BATCH_SIZE
    )
    
    # 批处理的最大令牌数，限制每个批次中的最大令牌数量
    batch_max_tokens: int = Field(
        description="要使用的批处理的最大令牌数。",
        default=defs.EMBEDDING_BATCH_MAX_TOKENS,
    )
    
    # 要使用的嵌入目标，可以是 'all' 或 'required'
    target: TextEmbeddingTarget = Field(
        description="要使用的目标。可以是 'all' 或 'required'。",
        default=defs.EMBEDDING_TARGET,
    )
    
    # 要跳过的特定嵌入列表，初始化为空列表
    skip: list[str] = Field(
        description="要跳过的特定嵌入。",
        default=[]
    )
    
    # 向量存储配置，可以为空字典或 None
    vector_store: dict | None = Field(
        description="向量存储配置。",
        default=None
    )
    
    # 覆盖策略配置，可以为空字典或 None
    strategy: dict | None = Field(
        description="要使用的覆盖策略。",
        default=None
    )

    def resolved_strategy(self) -> dict:
        """获取解析后的文本嵌入策略。"""
        # 导入文本嵌入策略类型枚举
        from graphrag.index.verbs.text.embed import TextEmbedStrategyType
        
        # 如果策略存在，则返回策略；否则返回默认策略
        return self.strategy or {
            "type": TextEmbedStrategyType.openai,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            "batch_size": self.batch_size,
            "batch_max_tokens": self.batch_max_tokens,
        }
```