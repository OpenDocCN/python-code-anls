# `.\graphrag\graphrag\config\models\claim_extraction_config.py`

```py
# 版权声明和许可信息
# 2024 年版权归 Microsoft Corporation 所有，根据 MIT 许可证授权

"""默认配置的参数设置。"""

# 从标准库中导入 Path 类
from pathlib import Path

# 从 pydantic 库中导入 Field 类
from pydantic import Field

# 导入 graphrag.config.defaults 模块作为 defs 别名
import graphrag.config.defaults as defs

# 从当前包中导入 LLMConfig 类
from .llm_config import LLMConfig


class ClaimExtractionConfig(LLMConfig):
    """用于声明提取的配置部分。继承自 LLMConfig 类。"""

    # 是否启用声明提取功能，布尔类型
    enabled: bool = Field(
        description="Whether claim extraction is enabled.",
    )

    # 要使用的声明提取提示语句，字符串或 None
    prompt: str | None = Field(
        description="The claim extraction prompt to use.", default=None
    )

    # 要使用的声明描述，字符串，默认值来自 defs 模块中的 CLAIM_DESCRIPTION
    description: str = Field(
        description="The claim description to use.",
        default=defs.CLAIM_DESCRIPTION,
    )

    # 要使用的最大实体收集数量，整数，默认值来自 defs 模块中的 CLAIM_MAX_GLEANINGS
    max_gleanings: int = Field(
        description="The maximum number of entity gleanings to use.",
        default=defs.CLAIM_MAX_GLEANINGS,
    )

    # 要使用的覆盖策略，字典或 None
    strategy: dict | None = Field(
        description="The override strategy to use.", default=None
    )

    def resolved_strategy(self, root_dir: str) -> dict:
        """获取已解析的声明提取策略。"""

        # 从 graphrag.index.verbs.covariates.extract_covariates 模块导入 ExtractClaimsStrategyType 类型
        from graphrag.index.verbs.covariates.extract_covariates import (
            ExtractClaimsStrategyType,
        )

        # 如果策略已提供，则使用它；否则使用默认值
        return self.strategy or {
            "type": ExtractClaimsStrategyType.graph_intelligence,
            "llm": self.llm.model_dump(),  # 调用 llm 对象的模型转储方法
            **self.parallelization.model_dump(),  # 并行化对象的模型转储结果合并
            "extraction_prompt": (Path(root_dir) / self.prompt).read_text()
            if self.prompt
            else None,  # 如果存在提示语句，读取相应的文本内容
            "claim_description": self.description,  # 使用声明描述
            "max_gleanings": self.max_gleanings,  # 使用最大实体收集数量
        }
```