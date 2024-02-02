# `MetaGPT\tests\metagpt\utils\test_json_to_markdown.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/11 11:53
@Author  : femto Zheng
@File    : test_json_to_markdown.py
"""

# 从metagpt.utils.json_to_markdown模块中导入json_to_markdown函数
from metagpt.utils.json_to_markdown import json_to_markdown

# 定义测试函数test_json_to_markdown
def test_json_to_markdown():
    # 定义一个嵌套的JSON数据
    json_data = {
        "title": "Sample JSON to Markdown Conversion",
        "description": "Convert JSON to Markdown with headings and lists.",
        "tags": ["json", "markdown", "conversion"],
        "content": {
            "section1": {"subsection1": "This is a subsection.", "subsection2": "Another subsection."},
            "section2": "This is the second section content.",
        },
    }

    # 使用json_to_markdown函数将JSON转换为Markdown
    markdown_output = json_to_markdown(json_data)

    # 期望的Markdown输出
    expected = """## title

Sample JSON to Markdown Conversion

## description

Convert JSON to Markdown with headings and lists.

## tags

- json
- markdown
- conversion

## content

### section1

#### subsection1

This is a subsection.

#### subsection2

Another subsection.

### section2

This is the second section content.

"""
    # 断言生成的Markdown与期望的Markdown相等
    assert expected == markdown_output

```