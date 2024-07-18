# `.\graphrag\graphrag\prompt_tune\generator\language.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Language detection for GraphRAG prompts."""

# 导入必要的模块和类
from graphrag.llm.types.llm_types import CompletionLLM
from graphrag.prompt_tune.prompt import DETECT_LANGUAGE_PROMPT

# 异步函数：用于检测输入文本的语言，以便为GraphRAG提示选择适当的语言
async def detect_language(llm: CompletionLLM, docs: str | list[str]) -> str:
    """Detect input language to use for GraphRAG prompts.

    Parameters
    ----------
    - llm (CompletionLLM): The LLM to use for generation
        用于生成的语言模型（LLM）
    - docs (str | list[str]): The docs to detect language from
        用于检测语言的文档或文本列表

    Returns
    -------
    - str: The detected language.
        检测到的语言字符串
    """
    # 如果输入参数是列表，则将其合并为单个字符串
    docs_str = " ".join(docs) if isinstance(docs, list) else docs
    # 构建用于语言检测的提示文本
    language_prompt = DETECT_LANGUAGE_PROMPT.format(input_text=docs_str)

    # 使用语言模型生成器LLM处理语言检测提示，并获取响应
    response = await llm(language_prompt)

    # 返回响应的输出结果，转换为字符串
    return str(response.output)
```