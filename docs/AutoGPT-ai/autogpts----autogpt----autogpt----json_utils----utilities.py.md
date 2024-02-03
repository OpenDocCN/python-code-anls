# `.\AutoGPT\autogpts\autogpt\autogpt\json_utils\utilities.py`

```py
"""Utilities for the json_fixes package."""
# 导入所需的模块
import json
import logging
import re
from typing import Any

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


# 从响应内容中提取字典
def extract_dict_from_response(response_content: str) -> dict[str, Any]:
    # 有时响应内容包含在带有 ``` 的代码块中
    pattern = r"```(?:json|JSON)*([\s\S]*?)```"
    # 在响应内容中查找匹配的模式
    match = re.search(pattern, response_content)

    if match:
        # 如果找到匹配的模式，则使用匹配的内容作为响应内容
        response_content = match.group(1).strip()
    else:
        # 如果未找到带有 ``` 的代码块，则尝试查找包含 JSON 的字符串
        json_pattern = r"{[\s\S]*}"
        match = re.search(json_pattern, response_content)

        if match:
            # 如果找到 JSON 字符串，则使用匹配的内容作为响应内容
            response_content = match.group()

    # 将响应内容解析为 JSON 格式
    result = json.loads(response_content)
    # 如果解析结果不是字典类型，则抛出数值错误
    if not isinstance(result, dict):
        raise ValueError(
            f"Response '''{response_content}''' evaluated to "
            f"non-dict value {repr(result)}"
        )
    # 返回解析后的字典结果
    return result
```