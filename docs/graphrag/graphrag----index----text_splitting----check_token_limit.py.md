# `.\graphrag\graphrag\index\text_splitting\check_token_limit.py`

```py
# 版权声明和许可证信息，声明代码版权及其使用许可证
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入TokenTextSplitter类从text_splitting模块中
from .text_splitting import TokenTextSplitter

# 定义函数用于检查文本的标记数量是否超过限制
def check_token_limit(text, max_token):
    """Check token limit."""
    # 创建TokenTextSplitter实例，用于将文本按照指定的标记数量切分
    text_splitter = TokenTextSplitter(chunk_size=max_token, chunk_overlap=0)
    # 使用text_splitter对象将文本进行分割，返回分割后的文档列表
    docs = text_splitter.split_text(text)
    # 如果分割后的文档数量大于1，返回0；否则返回1
    if len(docs) > 1:
        return 0
    else:
        return 1
```