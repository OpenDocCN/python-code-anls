# `.\graphrag\graphrag\config\input_models\claim_extraction_config_input.py`

```py
# 导入必要的模块和类
"""Parameterization settings for the default configuration."""
# 导入类型提示扩展，用于指定字段的可选性
from typing_extensions import NotRequired

# 从当前包中导入LLMConfigInput类
from .llm_config_input import LLMConfigInput


# 定义一个新的配置类ClaimExtractionConfigInput，继承自LLMConfigInput
class ClaimExtractionConfigInput(LLMConfigInput):
    """Configuration section for claim extraction."""

    # 下面是该配置类的字段声明，每个字段都标注了其可选性和可能的类型
    enabled: NotRequired[bool | None]  # 是否启用，可选的布尔值或None
    prompt: NotRequired[str | None]    # 提示信息，可选的字符串或None
    description: NotRequired[str | None]  # 描述信息，可选的字符串或None
    max_gleanings: NotRequired[int | str | None]  # 最大获取数，可选的整数、字符串或None
    strategy: NotRequired[dict | None]  # 策略参数，可选的字典或None
```