# `.\graphrag\graphrag\config\models\entity_extraction_config.py`

```py
# 版权声明和许可信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入路径处理工具
from pathlib import Path

# 导入字段定义工具
from pydantic import Field

# 导入默认配置模块
import graphrag.config.defaults as defs

# 导入语言模型配置类
from .llm_config import LLMConfig


# 实体抽取配置类，继承自语言模型配置类
class EntityExtractionConfig(LLMConfig):
    """Configuration section for entity extraction."""
    
    # 实体抽取提示语句，可以为None
    prompt: str | None = Field(
        description="The entity extraction prompt to use.", default=None
    )
    
    # 实体抽取的实体类型列表，默认从默认配置中获取
    entity_types: list[str] = Field(
        description="The entity extraction entity types to use.",
        default=defs.ENTITY_EXTRACTION_ENTITY_TYPES,
    )
    
    # 最大实体抽取数，默认从默认配置中获取
    max_gleanings: int = Field(
        description="The maximum number of entity gleanings to use.",
        default=defs.ENTITY_EXTRACTION_MAX_GLEANINGS,
    )
    
    # 覆盖默认的实体抽取策略，可以为None
    strategy: dict | None = Field(
        description="Override the default entity extraction strategy", default=None
    )

    # 获取解析后的实体抽取策略
    def resolved_strategy(self, root_dir: str, encoding_model: str) -> dict:
        """Get the resolved entity extraction strategy."""
        
        # 导入实体抽取策略类型枚举
        from graphrag.index.verbs.entities.extraction import ExtractEntityStrategyType
        
        # 如果策略为空，则使用默认策略
        return self.strategy or {
            "type": ExtractEntityStrategyType.graph_intelligence,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            "extraction_prompt": (Path(root_dir) / self.prompt).read_text()
            if self.prompt
            else None,
            "max_gleanings": self.max_gleanings,
            # 在 create_base_text_units 中进行了预分块处理
            "encoding_name": encoding_model,
            "prechunked": True,
        }
```