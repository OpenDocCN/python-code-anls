# `.\graphrag\graphrag\prompt_tune\generator\community_reporter_role.py`

```py
# 从 graphrag.llm.types.llm_types 模块导入 CompletionLLM 类型
# 从 graphrag.prompt_tune.prompt 模块导入 GENERATE_COMMUNITY_REPORTER_ROLE_PROMPT 常量
from graphrag.llm.types.llm_types import CompletionLLM
from graphrag.prompt_tune.prompt import (
    GENERATE_COMMUNITY_REPORTER_ROLE_PROMPT,
)

# 异步函数：生成一个社区报告角色用于社区总结
async def generate_community_reporter_role(
    llm: CompletionLLM, domain: str, persona: str, docs: str | list[str]
) -> str:
    """Generate an LLM persona to use for GraphRAG prompts.

    Parameters
    ----------
    - llm (CompletionLLM): 要用于生成的 LLM 对象
    - domain (str): 要为其生成角色的领域
    - persona (str): 要生成角色的人物形象
    - docs (str | list[str]): 生成角色的输入文本或文档列表

    Returns
    -------
    - str: 生成的领域提示的响应文本.
    """
    # 如果 docs 是列表，则将其转换为空格分隔的字符串
    docs_str = " ".join(docs) if isinstance(docs, list) else docs
    # 使用给定的 domain、persona 和 docs_str 格式化生成社区报告角色的提示
    domain_prompt = GENERATE_COMMUNITY_REPORTER_ROLE_PROMPT.format(
        domain=domain, persona=persona, input_text=docs_str
    )

    # 调用 LLM 对象的异步方法，传递生成的领域提示，并获取响应
    response = await llm(domain_prompt)

    # 将响应对象的输出转换为字符串并返回
    return str(response.output)
```