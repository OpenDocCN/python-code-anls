# `.\graphrag\graphrag\config\input_models\local_search_config_input.py`

```py
# 导入必要的模块，从 typing_extensions 中导入 NotRequired 和 TypedDict
from typing_extensions import NotRequired, TypedDict

# 定义一个 TypedDict 类型 LocalSearchConfigInput，用于描述默认配置的参数
class LocalSearchConfigInput(TypedDict):
    """The default configuration section for Cache."""

    # text_unit_prop 参数的类型为 NotRequired[float | str | None]
    text_unit_prop: NotRequired[float | str | None]
    # community_prop 参数的类型为 NotRequired[float | str | None]
    community_prop: NotRequired[float | str | None]
    # conversation_history_max_turns 参数的类型为 NotRequired[int | str | None]
    conversation_history_max_turns: NotRequired[int | str | None]
    # top_k_entities 参数的类型为 NotRequired[int | str | None]
    top_k_entities: NotRequired[int | str | None]
    # top_k_relationships 参数的类型为 NotRequired[int | str | None]
    top_k_relationships: NotRequired[int | str | None]
    # max_tokens 参数的类型为 NotRequired[int | str | None]
    max_tokens: NotRequired[int | str | None]
    # llm_max_tokens 参数的类型为 NotRequired[int | str | None]
    llm_max_tokens: NotRequired[int | str | None]
```