# `MetaGPT\tests\metagpt\learn\test_text_to_embedding.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/18
@Author  : mashenquan
@File    : test_text_to_embedding.py
@Desc    : Unit tests.
"""

# 导入 pytest 模块
import pytest

# 从 metagpt.config 模块中导入 CONFIG 变量
from metagpt.config import CONFIG
# 从 metagpt.learn.text_to_embedding 模块中导入 text_to_embedding 函数
from metagpt.learn.text_to_embedding import text_to_embedding

# 使用 pytest 的 asyncio 标记来标识异步测试
@pytest.mark.asyncio
async def test_text_to_embedding():
    # 先决条件：检查 CONFIG.OPENAI_API_KEY 是否存在
    assert CONFIG.OPENAI_API_KEY

    # 调用 text_to_embedding 函数，传入文本 "Panda emoji"，并等待返回结果
    v = await text_to_embedding(text="Panda emoji")
    # 断言返回的数据长度大于 0
    assert len(v.data) > 0

# 如果当前文件被直接执行，则运行 pytest 测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```