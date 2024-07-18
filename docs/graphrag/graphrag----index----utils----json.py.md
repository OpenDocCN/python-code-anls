# `.\graphrag\graphrag\index\utils\json.py`

```py
# 清理和格式化 JSON 的辅助函数

def clean_up_json(json_str: str):
    """Clean up json string."""
    # 去除字符串中的换行符和回车符，以及额外的转义符号
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
    if json_str.startswith("json"):
        json_str = json_str[len("json") :]
    if json_str.endswith("```"):
        json_str = json_str[: len(json_str) - len("```py")]

    # 返回清理后的 JSON 字符串
    return json_str
```