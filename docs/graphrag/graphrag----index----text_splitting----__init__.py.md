# `.\graphrag\graphrag\index\text_splitting\__init__.py`

```py
# 版权声明及许可信息，标明代码版权归 Microsoft Corporation 所有，并采用 MIT 许可证
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""索引引擎文本拆分包的根目录。"""

# 导入模块和函数，包括从本地模块中导入的:
# - 检查标记限制的函数 check_token_limit
# - 文本编码解码相关的函数和类 DecodeFn, EncodeFn, EncodedText
# - 文本长度相关的函数 LengthFn
# - 不做任何操作的文本拆分器 NoopTextSplitter
# - 列表文本拆分器 TextListSplitter 和其类型 TextListSplitterType
# - 文本拆分器 TextSplitter 和 TokenTextSplitter
# - 分词器 Tokenizer
# - 在标记上拆分文本的函数 split_text_on_tokens
from .check_token_limit import check_token_limit
from .text_splitting import (
    DecodeFn,
    EncodedText,
    EncodeFn,
    LengthFn,
    NoopTextSplitter,
    TextListSplitter,
    TextListSplitterType,
    TextSplitter,
    Tokenizer,
    TokenTextSplitter,
    split_text_on_tokens,
)

# 指定了模块中可以导出的公共接口列表，方便其他代码通过 from 包名 import * 使用
__all__ = [
    "DecodeFn",
    "EncodeFn",
    "EncodedText",
    "LengthFn",
    "NoopTextSplitter",
    "TextListSplitter",
    "TextListSplitterType",
    "TextSplitter",
    "TokenTextSplitter",
    "Tokenizer",
    "check_token_limit",
    "split_text_on_tokens",
]
```