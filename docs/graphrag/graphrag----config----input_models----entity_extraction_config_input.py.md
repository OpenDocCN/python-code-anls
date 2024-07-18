# `.\graphrag\graphrag\config\input_models\entity_extraction_config_input.py`

```py
# 版权声明和许可证信息，表明代码版权归 Microsoft Corporation 所有，采用 MIT 许可证
# 导入必要的模块和类
from typing_extensions import NotRequired

# 从当前目录下的 llm_config_input 模块中导入 LLMConfigInput 类
from .llm_config_input import LLMConfigInput

# 定义一个名为 EntityExtractionConfigInput 的类，继承自 LLMConfigInput 类
class EntityExtractionConfigInput(LLMConfigInput):
    """Configuration section for entity extraction."""
    
    # 定义类的属性和类型注解
    prompt: NotRequired[str | None]  # 用于指定提示信息的可选字符串或空值
    entity_types: NotRequired[list[str] | str | None]  # 用于指定实体类型的可选列表或字符串或空值
    max_gleanings: NotRequired[int | str | None]  # 用于指定最大提取量的可选整数或字符串或空值
    strategy: NotRequired[dict | None]  # 用于指定策略的可选字典或空值
```