# `.\graphrag\graphrag\prompt_tune\generator\entity_extraction_prompt.py`

```py
# 从 pathlib 模块导入 Path 类
from pathlib import Path

# 导入默认配置
import graphrag.config.defaults as defs
# 从 utils.tokens 模块导入 num_tokens_from_string 函数
from graphrag.index.utils.tokens import num_tokens_from_string
# 从 prompt_tune.template 模块导入所需常量和模板
from graphrag.prompt_tune.template import (
    EXAMPLE_EXTRACTION_TEMPLATE,
    GRAPH_EXTRACTION_JSON_PROMPT,
    GRAPH_EXTRACTION_PROMPT,
    UNTYPED_EXAMPLE_EXTRACTION_TEMPLATE,
    UNTYPED_GRAPH_EXTRACTION_PROMPT,
)

# 定义常量，用于指定实体提取结果的保存文件名
ENTITY_EXTRACTION_FILENAME = "entity_extraction.txt"

# 定义函数，生成用于实体提取的提示文本
def create_entity_extraction_prompt(
    entity_types: str | list[str] | None,
    docs: list[str],
    examples: list[str],
    language: str,
    max_token_count: int,
    encoding_model: str = defs.ENCODING_MODEL,
    json_mode: bool = False,
    output_path: Path | None = None,
) -> str:
    """
    Create a prompt for entity extraction.

    Parameters
    ----------
    - entity_types (str | list[str]): 要提取的实体类型
    - docs (list[str]): 要从中提取实体的文档列表
    - examples (list[str]): 用于实体提取的示例列表
    - language (str): 输入和输出的语言
    - encoding_model (str): 用于计算令牌数的模型名称
    - max_token_count (int): 用于提示的最大令牌数
    - json_mode (bool): 是否使用 JSON 模式生成提示。默认为 False
    - output_path (Path | None): 写入提示的文件路径。默认为 None。如果为 None，则不写入文件

    Returns
    -------
    - str: 实体提取提示文本
    """
    
    # 根据是否提供实体类型选择合适的提示模板
    prompt = (
        GRAPH_EXTRACTION_JSON_PROMPT if json_mode else GRAPH_EXTRACTION_PROMPT
    ) if entity_types else UNTYPED_GRAPH_EXTRACTION_PROMPT

    # 如果实体类型是列表，则将其转换为逗号分隔的字符串
    if isinstance(entity_types, list):
        entity_types = ", ".join(entity_types)

    # 计算剩余的可用令牌数
    tokens_left = (
        max_token_count
        - num_tokens_from_string(prompt, model=encoding_model)
        - num_tokens_from_string(entity_types, model=encoding_model)
        if entity_types
        else 0
    )

    # 初始化空字符串，用于存储示例提示文本
    examples_prompt = ""

    # 迭代处理示例，直到令牌耗尽或处理完所有示例
    for i, output in enumerate(examples):
        input = docs[i]
        
        # 根据是否提供实体类型选择合适的示例模板
        example_formatted = (
            EXAMPLE_EXTRACTION_TEMPLATE.format(
                n=i + 1, input_text=input, entity_types=entity_types, output=output
            )
            if entity_types
            else UNTYPED_EXAMPLE_EXTRACTION_TEMPLATE.format(
                n=i + 1, input_text=input, output=output
            )
        )

        # 计算示例文本的令牌数
        example_tokens = num_tokens_from_string(example_formatted, model=encoding_model)

        # 如果不是第一个示例且示例文本令牌数超出剩余可用令牌数，则中断迭代
        if i > 0 and example_tokens > tokens_left:
            break

        # 将格式化后的示例文本添加到总的提示文本中
        examples_prompt += example_formatted
        # 更新剩余可用令牌数
        tokens_left -= example_tokens
    # 格式化提示信息，根据实体类型、示例和语言进行格式化，如果没有实体类型则不包含实体类型信息
    prompt = (
        prompt.format(
            entity_types=entity_types, examples=examples_prompt, language=language
        )
        if entity_types
        else prompt.format(examples=examples_prompt, language=language)
    )

    # 如果输出路径存在，则创建父目录（如果不存在），确保输出路径存在
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

        # 设置输出文件路径为输出路径下的 ENTITY_EXTRACTION_FILENAME 文件
        output_path = output_path / ENTITY_EXTRACTION_FILENAME
        
        # 将格式化后的提示信息写入到输出文件中
        with output_path.open("w") as file:
            file.write(prompt)

    # 返回格式化后的提示信息
    return prompt
```