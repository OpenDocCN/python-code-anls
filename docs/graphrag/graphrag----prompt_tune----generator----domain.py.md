# `.\graphrag\graphrag\prompt_tune\generator\domain.py`

```py
# 版权所有 (c) 2024 微软公司。
# 根据 MIT 许可证授权

"""GraphRAG 提示的领域生成。"""

# 从 graphrag.llm.types.llm_types 模块导入 CompletionLLM 类型
from graphrag.llm.types.llm_types import CompletionLLM
# 从 graphrag.prompt_tune.prompt.domain 模块导入 GENERATE_DOMAIN_PROMPT 常量
from graphrag.prompt_tune.prompt.domain import GENERATE_DOMAIN_PROMPT


# 异步函数：生成用于 GraphRAG 提示的领域信息
async def generate_domain(llm: CompletionLLM, docs: str | list[str]) -> str:
    """生成用于 GraphRAG 提示的语言模型人物设定。

    Parameters
    ----------
    - llm (CompletionLLM): 用于生成的语言模型
    - docs (str | list[str]): 用于生成人物设定的领域文档或文本列表

    Returns
    -------
    - str: 生成的领域提示响应。
    """
    # 如果 docs 是列表，则将其连接成单个字符串
    docs_str = " ".join(docs) if isinstance(docs, list) else docs
    # 根据格式化字符串 GENERATE_DOMAIN_PROMPT，生成领域提示语句
    domain_prompt = GENERATE_DOMAIN_PROMPT.format(input_text=docs_str)

    # 使用语言模型 llm 处理领域提示语句，获取响应
    response = await llm(domain_prompt)

    # 返回响应的字符串表示形式
    return str(response.output)
```