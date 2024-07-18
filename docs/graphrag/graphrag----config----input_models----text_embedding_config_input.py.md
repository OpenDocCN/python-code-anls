# `.\graphrag\graphrag\config\input_models\text_embedding_config_input.py`

```py
# 版权声明，指出代码版权归 Microsoft Corporation 所有，采用 MIT 许可证授权
# 引入 typing_extensions 中的 NotRequired 类型
# 引入 graphrag.config.enums 中的 TextEmbeddingTarget 枚举类型
from typing_extensions import NotRequired
from graphrag.config.enums import (
    TextEmbeddingTarget,
)

# 从当前目录下的 llm_config_input 模块中导入 LLMConfigInput 类
from .llm_config_input import LLMConfigInput

# TextEmbeddingConfigInput 类继承自 LLMConfigInput 类，用于配置文本嵌入相关的参数
class TextEmbeddingConfigInput(LLMConfigInput):
    """Configuration section for text embeddings."""

    # 批处理大小，可选类型为 int、str 或 None
    batch_size: NotRequired[int | str | None]
    # 每个批次的最大标记数，可选类型为 int、str 或 None
    batch_max_tokens: NotRequired[int | str | None]
    # 目标文本嵌入类型，可选类型为 TextEmbeddingTarget 枚举或 str 或 None
    target: NotRequired[TextEmbeddingTarget | str | None]
    # 跳过的文本列表，可选类型为列表[str]、str 或 None
    skip: NotRequired[list[str] | str | None]
    # 向量存储，可选类型为 dict 或 None
    vector_store: NotRequired[dict | None]
    # 策略配置，可选类型为 dict 或 None
    strategy: NotRequired[dict | None]
```