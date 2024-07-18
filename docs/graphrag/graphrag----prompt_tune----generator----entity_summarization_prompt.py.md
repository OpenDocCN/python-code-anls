# `.\graphrag\graphrag\prompt_tune\generator\entity_summarization_prompt.py`

```py
# 版权声明和许可信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""实体摘要生成模块。"""

# 导入路径操作模块
from pathlib import Path

# 导入实体摘要的模板常量
from graphrag.prompt_tune.template import ENTITY_SUMMARIZATION_PROMPT

# 实体摘要文件名常量
ENTITY_SUMMARIZATION_FILENAME = "summarize_descriptions.txt"


def create_entity_summarization_prompt(
    persona: str,
    language: str,
    output_path: Path | None = None,
) -> str:
    """
    创建用于实体摘要的提示语。如果提供了 output_path，则将提示语写入文件。

    Parameters
    ----------
    - persona (str): 用于实体摘要提示语的人物角色
    - language (str): 用于实体摘要提示语的语言
    - output_path (Path | None): 要写入提示语的路径。默认为 None。如果为 None，则不将提示语写入文件。默认为 None。
    """
    # 使用给定的 persona 和 language 格式化实体摘要提示语
    prompt = ENTITY_SUMMARIZATION_PROMPT.format(persona=persona, language=language)

    # 如果提供了 output_path，则准备写入文件
    if output_path:
        # 确保输出路径存在，如果不存在则创建
        output_path.mkdir(parents=True, exist_ok=True)

        # 在输出路径下创建实体摘要文件
        output_path = output_path / ENTITY_SUMMARIZATION_FILENAME
        
        # 将提示语写入到文件中
        with output_path.open("w") as file:
            file.write(prompt)

    # 返回生成的实体摘要提示语
    return prompt
```