# `.\graphrag\graphrag\index\verbs\text\chunk\strategies\sentence.py`

```py
# 导入必要的模块和类
from collections.abc import Iterable  # 从 collections.abc 模块导入 Iterable 类型
from typing import Any  # 从 typing 模块导入 Any 类型

import nltk  # 导入 nltk 库，用于自然语言处理
from datashaper import ProgressTicker  # 从 datashaper 模块导入 ProgressTicker 类

from .typing import TextChunk  # 从当前包的 typing 模块导入 TextChunk 类型


def run(
    input: list[str], _args: dict[str, Any], tick: ProgressTicker
) -> Iterable[TextChunk]:
    """Chunks text into multiple parts. A pipeline verb."""
    # 遍历输入列表中的文档索引和文本
    for doc_idx, text in enumerate(input):
        # 使用 nltk 库对文本进行句子分割，返回句子列表
        sentences = nltk.sent_tokenize(text)
        # 遍历每个句子，并使用 TextChunk 类创建 TextChunk 对象
        for sentence in sentences:
            yield TextChunk(
                text_chunk=sentence,  # 句子文本作为 text_chunk 属性
                source_doc_indices=[doc_idx],  # 当前文档索引作为 source_doc_indices 属性
            )
        # 调用 ProgressTicker 对象 tick 方法，传入参数 1，用于进度更新
        tick(1)


这段代码定义了一个名为 `run` 的函数，它接受一个输入列表和两个额外参数 `_args` 和 `tick`，然后通过使用 `nltk` 库将每个文本分割成句子，并使用 `TextChunk` 类创建句子对象，最后调用 `ProgressTicker` 对象的 `tick` 方法进行进度更新。
```