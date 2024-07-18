# `.\graphrag\graphrag\prompt_tune\generator\community_report_rating.py`

```py
"""Generate a rating description for community report rating."""

# 导入必要的模块和函数
from graphrag.llm.types.llm_types import CompletionLLM
from graphrag.prompt_tune.prompt import (
    GENERATE_REPORT_RATING_PROMPT,
)

# 定义异步函数，生成社区报告评分的描述
async def generate_community_report_rating(
    llm: CompletionLLM, domain: str, persona: str, docs: str | list[str]
) -> str:
    """Generate an LLM persona to use for GraphRAG prompts.

    Parameters
    ----------
    - llm (CompletionLLM): The LLM to use for generation
    - domain (str): The domain to generate a rating for
    - persona (str): The persona to generate a rating for for
    - docs (str | list[str]): Documents used to contextualize the rating

    Returns
    -------
    - str: The generated rating description prompt response.
    """

    # 如果文档是列表，则将其转换为单个字符串
    docs_str = " ".join(docs) if isinstance(docs, list) else docs

    # 根据模板生成领域、角色和文档的评分描述提示
    domain_prompt = GENERATE_REPORT_RATING_PROMPT.format(
        domain=domain, persona=persona, input_text=docs_str
    )

    # 使用给定的LLM生成响应
    response = await llm(domain_prompt)

    # 返回生成的评分描述响应，去除首尾的空白字符
    return str(response.output).strip()
```