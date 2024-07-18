# `.\graphrag\graphrag\index\verbs\text\translate\strategies\mock.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入所需模块和类型声明
from typing import Any

# 从datashaper模块导入VerbCallbacks类型
from datashaper import VerbCallbacks

# 从graphrag.index.cache模块导入PipelineCache类
from graphrag.index.cache import PipelineCache

# 从当前目录的typing模块导入TextTranslationResult类型
from .typing import TextTranslationResult


# 异步函数定义，用于运行文本翻译链
async def run(
    input: str | list[str],    # 输入参数可以是单个字符串或字符串列表
    _args: dict[str, Any],     # 其他参数字典，不使用
    _reporter: VerbCallbacks,  # 用于进度报告的回调对象，不使用
    _cache: PipelineCache,     # 缓存对象，不使用
) -> TextTranslationResult:
    """Run the Claim extraction chain."""
    # 如果输入是字符串，转换为字符串列表
    input = [input] if isinstance(input, str) else input
    # 返回文本翻译结果对象，其中包含翻译后的文本列表
    return TextTranslationResult(translations=[_translate_text(text) for text in input])


# 内部函数，用于对单个文本进行翻译
def _translate_text(text: str) -> str:
    """Translate a single piece of text."""
    # 返回翻译后的文本字符串
    return f"{text} translated"
```