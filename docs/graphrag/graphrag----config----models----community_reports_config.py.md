# `.\graphrag\graphrag\config\models\community_reports_config.py`

```py
# 版权声明
# 2024年由Microsoft Corporation持有版权，根据MIT许可证授权

"""默认配置的参数化设置。"""

# 导入模块
from pathlib import Path

from pydantic import Field

import graphrag.config.defaults as defs

from .llm_config import LLMConfig


class CommunityReportsConfig(LLMConfig):
    """社区报告的配置部分。"""

    # 社区报告提取提示
    prompt: str | None = Field(
        description="要使用的社区报告提取提示。", default=None
    )
    # 社区报告的最大长度（以标记计算）
    max_length: int = Field(
        description="社区报告的最大长度（以标记计算）。",
        default=defs.COMMUNITY_REPORT_MAX_LENGTH,
    )
    # 生成报告时要使用的最大输入长度（以标记计算）
    max_input_length: int = Field(
        description="要在生成报告时使用的最大输入长度（以标记计算）。",
        default=defs.COMMUNITY_REPORT_MAX_INPUT_LENGTH,
    )
    # 要使用的覆盖策略
    strategy: dict | None = Field(
        description="要使用的覆盖策略。", default=None
    )

    def resolved_strategy(self, root_dir) -> dict:
        """获取已解析的社区报告提取策略。"""
        # 导入模块
        from graphrag.index.verbs.graph.report import CreateCommunityReportsStrategyType

        return self.strategy or {
            # 默认情况下使用图智能
            "type": CreateCommunityReportsStrategyType.graph_intelligence,
            # LLM模型
            "llm": self.llm.model_dump(),
            # 并行化模型
            **self.parallelization.model_dump(),
            # 提取提示
            "extraction_prompt": (Path(root_dir) / self.prompt).read_text()
            if self.prompt
            else None,
            # 最大报告长度
            "max_report_length": self.max_length,
            # 最大输入长度
            "max_input_length": self.max_input_length,
        }
```