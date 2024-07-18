# `.\graphrag\graphrag\config\models\summarize_descriptions_config.py`

```py
# 版权声明和许可信息
# 版权所有 (c) 2024 Microsoft Corporation.
# 根据 MIT 许可证授权

"""默认配置的参数设置。"""

from pathlib import Path  # 导入处理路径的模块 Path

from pydantic import Field  # 导入 Pydantic 库中的 Field 类

import graphrag.config.defaults as defs  # 导入默认配置文件中的 defs 模块

from .llm_config import LLMConfig  # 从当前目录导入 LLMConfig 类

class SummarizeDescriptionsConfig(LLMConfig):
    """描述摘要部分的配置类。继承自 LLMConfig 类。"""

    prompt: str | None = Field(
        description="用于描述摘要的提示语句。", default=None
    )
    max_length: int = Field(
        description="描述摘要的最大长度。",
        default=defs.SUMMARIZE_DESCRIPTIONS_MAX_LENGTH,
    )
    strategy: dict | None = Field(
        description="用于描述摘要的覆盖策略。", default=None
    )

    def resolved_strategy(self, root_dir: str) -> dict:
        """获取已解析的描述摘要策略。"""
        from graphrag.index.verbs.entities.summarize import SummarizeStrategyType

        return self.strategy or {
            "type": SummarizeStrategyType.graph_intelligence,
            "llm": self.llm.model_dump(),
            **self.parallelization.model_dump(),
            "summarize_prompt": (Path(root_dir) / self.prompt).read_text()
            if self.prompt
            else None,
            "max_summary_length": self.max_length,
        }
```