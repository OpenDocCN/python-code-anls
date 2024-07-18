# `.\graphrag\graphrag\llm\openai\_json.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""JSON cleaning and formatting utilities."""

# 定义一个函数，用于清理和格式化 JSON 字符串
def clean_up_json(json_str: str) -> str:
    """Clean up json string."""
    # 去除字符串中的特殊字符和换行符
    json_str = (
        json_str.replace("\\n", "")
        .replace("\n", "")
        .replace("\r", "")
        .replace('"[{', "[{")
        .replace('}]"', "}]")
        .replace("\\", "")
        .strip()
    )

    # 移除 JSON Markdown 标记
    if json_str.startswith("```json"):
        json_str = json_str[len("```py") :]
    if json_str.endswith("```"):
        json_str = json_str[: len(json_str) - len("```py")]

    # 返回处理后的 JSON 字符串
    return json_str
```