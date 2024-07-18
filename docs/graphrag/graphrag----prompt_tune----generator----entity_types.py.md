# `.\graphrag\graphrag\prompt_tune\generator\entity_types.py`

```py
# 引入模块和类别以进行微调
from graphrag.llm.types.llm_types import CompletionLLM
from graphrag.prompt_tune.generator.defaults import DEFAULT_TASK
from graphrag.prompt_tune.prompt.entity_types import (
    ENTITY_TYPE_GENERATION_JSON_PROMPT,
    ENTITY_TYPE_GENERATION_PROMPT,
)

# 定义异步函数，生成实体类型类别
async def generate_entity_types(
    llm: CompletionLLM,            # 接收一个 CompletionLLM 类的实例，用于生成完成内容
    domain: str,                   # 领域名称，用于任务格式化
    persona: str,                  # 个人信息，作为对话历史的一部分
    docs: str | list[str],         # 输入的文档，可以是字符串或字符串列表
    task: str = DEFAULT_TASK,      # 任务描述，默认使用默认任务
    json_mode: bool = False,       # 是否以 JSON 模式生成
) -> str | list[str]:             # 函数返回字符串或字符串列表
    """
    Generate entity type categories from a given set of (small) documents.

    Example Output:
    "entity_types": ['military unit', 'organization', 'person', 'location', 'event', 'date', 'equipment']
    """

    # 格式化任务描述，将领域信息插入任务模板中
    formatted_task = task.format(domain=domain)

    # 如果输入文档是列表，则转换成用换行符连接的单个字符串
    docs_str = "\n".join(docs) if isinstance(docs, list) else docs

    # 根据 JSON 模式选择相应的提示模板
    entity_types_prompt = (
        ENTITY_TYPE_GENERATION_JSON_PROMPT    # 如果是 JSON 模式，使用 JSON 提示模板
        if json_mode
        else ENTITY_TYPE_GENERATION_PROMPT    # 否则使用普通提示模板
    ).format(task=formatted_task, input_text=docs_str)

    # 创建对话历史记录，包含系统角色和个人信息内容
    history = [{"role": "system", "content": persona}]

    # 使用 LLM 实例处理提示，获取生成的响应
    response = await llm(entity_types_prompt, history=history, json=json_mode)

    # 如果是 JSON 模式，则从响应中提取实体类型列表
    if json_mode:
        return (response.json or {}).get("entity_types", [])

    # 否则，返回响应的输出内容的字符串表示形式
    return str(response.output)
```