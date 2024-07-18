# `.\graphrag\graphrag\config\models\local_search_config.py`

```py
# 引入pydantic模块中的BaseModel和Field类
from pydantic import BaseModel, Field

# 引入graphrag.config.defaults模块，命名为defs
import graphrag.config.defaults as defs

# 定义LocalSearchConfig类，继承自BaseModel，表示默认配置节的Cache部分
class LocalSearchConfig(BaseModel):
    """The default configuration section for Cache."""

    # text_unit_prop字段，浮点数类型，默认值为defs.LOCAL_SEARCH_TEXT_UNIT_PROP，
    # 代表文本单元比例
    text_unit_prop: float = Field(
        description="The text unit proportion.",
        default=defs.LOCAL_SEARCH_TEXT_UNIT_PROP,
    )
    
    # community_prop字段，浮点数类型，默认值为defs.LOCAL_SEARCH_COMMUNITY_PROP，
    # 代表社区比例
    community_prop: float = Field(
        description="The community proportion.",
        default=defs.LOCAL_SEARCH_COMMUNITY_PROP,
    )
    
    # conversation_history_max_turns字段，整数类型，默认值为defs.LOCAL_SEARCH_CONVERSATION_HISTORY_MAX_TURNS，
    # 代表会话历史最大轮数
    conversation_history_max_turns: int = Field(
        description="The conversation history maximum turns.",
        default=defs.LOCAL_SEARCH_CONVERSATION_HISTORY_MAX_TURNS,
    )
    
    # top_k_entities字段，整数类型，默认值为defs.LOCAL_SEARCH_TOP_K_MAPPED_ENTITIES，
    # 代表前k个映射实体
    top_k_entities: int = Field(
        description="The top k mapped entities.",
        default=defs.LOCAL_SEARCH_TOP_K_MAPPED_ENTITIES,
    )
    
    # top_k_relationships字段，整数类型，默认值为defs.LOCAL_SEARCH_TOP_K_RELATIONSHIPS，
    # 代表前k个映射关系
    top_k_relationships: int = Field(
        description="The top k mapped relations.",
        default=defs.LOCAL_SEARCH_TOP_K_RELATIONSHIPS,
    )
    
    # temperature字段，浮点数或None类型，默认值为defs.LOCAL_SEARCH_LLM_TEMPERATURE，
    # 代表用于token生成的温度
    temperature: float | None = Field(
        description="The temperature to use for token generation.",
        default=defs.LOCAL_SEARCH_LLM_TEMPERATURE,
    )
    
    # top_p字段，浮点数或None类型，默认值为defs.LOCAL_SEARCH_LLM_TOP_P，
    # 代表用于token生成的top-p值
    top_p: float | None = Field(
        description="The top-p value to use for token generation.",
        default=defs.LOCAL_SEARCH_LLM_TOP_P,
    )
    
    # n字段，整数或None类型，默认值为defs.LOCAL_SEARCH_LLM_N，
    # 代表要生成的完成数
    n: int | None = Field(
        description="The number of completions to generate.",
        default=defs.LOCAL_SEARCH_LLM_N,
    )
    
    # max_tokens字段，整数类型，默认值为defs.LOCAL_SEARCH_MAX_TOKENS，
    # 代表最大token数目
    max_tokens: int = Field(
        description="The maximum tokens.", default=defs.LOCAL_SEARCH_MAX_TOKENS
    )
    
    # llm_max_tokens字段，整数类型，默认值为defs.LOCAL_SEARCH_LLM_MAX_TOKENS，
    # 代表LLM的最大token数目
    llm_max_tokens: int = Field(
        description="The LLM maximum tokens.", default=defs.LOCAL_SEARCH_LLM_MAX_TOKENS
    )
```