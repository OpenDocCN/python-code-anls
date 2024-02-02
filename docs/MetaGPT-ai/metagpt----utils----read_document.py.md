# `MetaGPT\metagpt\utils\read_document.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/29 15:45
@Author  : alexanderwu
@File    : read_document.py
"""

# 导入 docx 模块
import docx

# 定义函数，读取 docx 文件内容
def read_docx(file_path: str) -> list:
    """Open a docx file"""
    # 使用 docx 模块打开指定路径的 docx 文件
    doc = docx.Document(file_path)

    # 创建一个空列表，用于存储段落内容
    paragraphs_list = []

    # 遍历文档中的段落，并将它们的内容添加到列表中
    for paragraph in doc.paragraphs:
        paragraphs_list.append(paragraph.text)

    # 返回段落内容列表
    return paragraphs_list

```