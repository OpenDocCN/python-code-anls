# `MetaGPT\metagpt\utils\json_to_markdown.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/11 11:50
@Author  : femto Zheng
@File    : json_to_markdown.py
"""

# 定义函数，将 JSON 对象转换为 Markdown 格式
def json_to_markdown(data, depth=2):
    """
    Convert a JSON object to Markdown with headings for keys and lists for arrays, supporting nested objects.

    Args:
        data: JSON object (dictionary) or value.
        depth (int): Current depth level for Markdown headings.

    Returns:
        str: Markdown representation of the JSON data.
    """
    markdown = ""

    # 如果数据是字典类型
    if isinstance(data, dict):
        # 遍历字典的键值对
        for key, value in data.items():
            # 如果值是列表
            if isinstance(value, list):
                # 处理 JSON 数组
                markdown += "#" * depth + f" {key}\n\n"
                items = [str(item) for item in value]
                markdown += "- " + "\n- ".join(items) + "\n\n"
            # 如果值是字典
            elif isinstance(value, dict):
                # 处理嵌套的 JSON 对象
                markdown += "#" * depth + f" {key}\n\n"
                markdown += json_to_markdown(value, depth + 1)
            else:
                # 处理其他值
                markdown += "#" * depth + f" {key}\n\n{value}\n\n"
    else:
        # 处理非字典类型的 JSON 数据
        markdown = str(data)

    return markdown

```