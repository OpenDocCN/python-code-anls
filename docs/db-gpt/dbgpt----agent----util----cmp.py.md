# `.\DB-GPT-src\dbgpt\agent\util\cmp.py`

```py
"""Compare string utility functions."""
# 导入标准库中的 string 模块，用于处理字符串中的标点符号
import string


def cmp_string_equal(
    a: str,
    b: str,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    ignore_whitespace: bool = False,
) -> bool:
    """Compare two strings are equal or not.

    Args:
        a(str): The first string.
        b(str): The second string.
        ignore_case(bool): Ignore case or not.
        ignore_punctuation(bool): Ignore punctuation or not.
        ignore_whitespace(bool): Ignore whitespace or not.
    
    Returns:
        bool: True if strings are equal after applying specified options, False otherwise.
    """
    # 如果指定忽略大小写，则将两个字符串转换为小写
    if ignore_case:
        a = a.lower()
        b = b.lower()
    # 如果指定忽略标点符号，则从字符串中移除所有标点符号
    if ignore_punctuation:
        a = a.translate(str.maketrans("", "", string.punctuation))
        b = b.translate(str.maketrans("", "", string.punctuation))
    # 如果指定忽略空白符，则从字符串中移除所有空白字符
    if ignore_whitespace:
        a = "".join(a.split())
        b = "".join(b.split())
    # 返回经过处理后的两个字符串是否相等的比较结果
    return a == b
```