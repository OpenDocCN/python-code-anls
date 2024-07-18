# `.\graphrag\graphrag\index\utils\string.py`

```py
# 著作权声明和许可证声明，说明该模块的版权和许可证信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入必要的模块
"""String utilities."""
import html  # 导入处理 HTML 的模块
import re    # 导入正则表达式模块
from typing import Any  # 导入类型提示模块


# 定义函数 clean_str，用于清理输入字符串，去除 HTML 转义字符、控制字符和其他不需要的字符
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # 如果输入不是字符串类型，则直接返回输入
    if not isinstance(input, str):
        return input

    # 对输入字符串进行去除 HTML 转义和去除首尾空白字符处理
    result = html.unescape(input.strip())
    
    # 使用正则表达式去除控制字符，参考来源：https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
```