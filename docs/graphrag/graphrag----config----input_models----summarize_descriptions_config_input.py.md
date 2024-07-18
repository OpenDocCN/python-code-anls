# `.\graphrag\graphrag\config\input_models\summarize_descriptions_config_input.py`

```py
# 从 typing_extensions 导入 NotRequired 类型
from typing_extensions import NotRequired

# 从当前包中导入 LLMConfigInput 类
from .llm_config_input import LLMConfigInput

# SummarizeDescriptionsConfigInput 类继承自 LLMConfigInput 类，用于描述摘要生成的配置部分
class SummarizeDescriptionsConfigInput(LLMConfigInput):
    """Configuration section for description summarization."""
    
    # prompt 属性：用于描述生成摘要时的提示文本，可选类型为字符串或 None
    prompt: NotRequired[str | None]
    
    # max_length 属性：生成摘要的最大长度限制，可选类型为整数、字符串或 None
    max_length: NotRequired[int | str | None]
    
    # strategy 属性：摘要生成策略的配置，可选类型为字典或 None
    strategy: NotRequired[dict | None]
```