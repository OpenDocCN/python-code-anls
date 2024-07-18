# `.\graphrag\graphrag\prompt_tune\generator\community_report_summarization.py`

```py
# 从 pathlib 模块中导入 Path 类
from pathlib import Path

# 从 graphrag.prompt_tune.template 模块导入 COMMUNITY_REPORT_SUMMARIZATION_PROMPT 变量
from graphrag.prompt_tune.template import COMMUNITY_REPORT_SUMMARIZATION_PROMPT

# 定义一个常量，表示生成的社区报告摘要的文件名
COMMUNITY_SUMMARIZATION_FILENAME = "community_report.txt"

# 创建一个函数，用于生成社区报告摘要的提示语。如果提供了 output_path 参数，将提示语写入文件。
def create_community_summarization_prompt(
    persona: str,
    role: str,
    report_rating_description: str,
    language: str,
    output_path: Path | None = None,
) -> str:
    """Create a prompt for community summarization. If output_path is provided, write the prompt to a file.

    Parameters
    ----------
    - persona (str): The persona to use for the community summarization prompt
    - role (str): The role to use for the community summarization prompt
    - report_rating_description (str): The rating description to use for the community summarization prompt
    - language (str): The language to use for the community summarization prompt
    - output_path (Path | None): The path to write the prompt to. Default is None. If None, the prompt is not written to a file. Default is None.

    Returns
    -------
    - str: The community summarization prompt
    """
    # 使用提供的参数填充 COMMUNITY_REPORT_SUMMARIZATION_PROMPT 模板，生成最终的提示语
    prompt = COMMUNITY_REPORT_SUMMARIZATION_PROMPT.format(
        persona=persona,
        role=role,
        report_rating_description=report_rating_description,
        language=language,
    )

    # 如果提供了 output_path 参数
    if output_path:
        # 确保输出目录存在，如果不存在则创建
        output_path.mkdir(parents=True, exist_ok=True)

        # 构建输出文件的完整路径，包括文件名 COMMUNITY_SUMMARIZATION_FILENAME
        output_path = output_path / COMMUNITY_SUMMARIZATION_FILENAME
        
        # 将生成的提示语写入到文件中
        with output_path.open("w") as file:
            file.write(prompt)

    # 返回生成的提示语
    return prompt
```