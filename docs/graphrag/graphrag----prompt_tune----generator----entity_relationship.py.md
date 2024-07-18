# `.\graphrag\graphrag\prompt_tune\generator\entity_relationship.py`

```py
# 版权声明及许可声明，说明代码的版权和许可协议
"""实体关系示例生成模块。"""

# 引入 asyncio 和 json 模块
import asyncio
import json

# 从 graphrag.llm.types.llm_types 模块中引入 CompletionLLM 类型
from graphrag.llm.types.llm_types import CompletionLLM
# 从 graphrag.prompt_tune.prompt 模块中引入以下常量和函数
from graphrag.prompt_tune.prompt import (
    ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT,
    ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
    UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
)

# 定义常量 MAX_EXAMPLES，表示最大示例数为 5
MAX_EXAMPLES = 5

# 异步函数定义，用于生成实体关系示例
async def generate_entity_relationship_examples(
    llm: CompletionLLM,
    persona: str,
    entity_types: str | list[str] | None,
    docs: str | list[str],
    language: str,
    json_mode: bool = False,
) -> list[str]:
    """
    生成用于生成实体配置的实体/关系示例列表。

    根据 json_mode 参数返回实体/关系示例的 JSON 格式或元组分隔符格式。
    """
    # 如果 docs 是字符串，则转换为列表
    docs_list = [docs] if isinstance(docs, str) else docs
    # 历史记录，包含系统角色和用户 persona
    history = [{"role": "system", "content": persona}]

    # 如果存在 entity_types 参数
    if entity_types:
        # 如果 entity_types 是字符串，则直接使用；否则将列表转换为逗号分隔的字符串
        entity_types_str = (
            entity_types if isinstance(entity_types, str) else ", ".join(entity_types)
        )

        # 生成消息列表，根据 json_mode 参数选择不同的提示文本模板进行格式化
        messages = [
            (
                ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT
                if json_mode
                else ENTITY_RELATIONSHIPS_GENERATION_PROMPT
            ).format(entity_types=entity_types_str, input_text=doc, language=language)
            for doc in docs_list
        ]
    else:
        # 生成消息列表，使用未分类实体关系生成的提示文本模板进行格式化
        messages = [
            UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT.format(
                input_text=doc, language=language
            )
            for doc in docs_list
        ]

    # 最多保留 MAX_EXAMPLES 条消息
    messages = messages[:MAX_EXAMPLES]

    # 创建异步任务列表，每个任务调用 llm 函数进行实例生成
    tasks = [llm(message, history=history, json=json_mode) for message in messages]

    # 并发执行所有任务，获取所有任务的返回结果
    responses = await asyncio.gather(*tasks)

    # 根据 json_mode 参数选择返回格式，将响应转换为 JSON 字符串或字符串
    return [
        json.dumps(response.json or "") if json_mode else str(response.output)
        for response in responses
    ]
```