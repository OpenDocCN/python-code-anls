# `MetaGPT\tests\metagpt\tools\test_openai_text_to_embedding.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/26
@Author  : mashenquan
@File    : test_openai_text_to_embedding.py
"""

# 导入 pytest 模块
import pytest

# 从 metagpt.config 模块中导入 CONFIG 对象
from metagpt.config import CONFIG
# 从 metagpt.tools.openai_text_to_embedding 模块中导入 oas3_openai_text_to_embedding 函数
from metagpt.tools.openai_text_to_embedding import oas3_openai_text_to_embedding

# 使用 pytest.mark.asyncio 装饰器标记异步测试函数
@pytest.mark.asyncio
async def test_embedding():
    # 先决条件
    # 检查 CONFIG.OPENAI_API_KEY 是否存在且不等于 "YOUR_API_KEY"
    assert CONFIG.OPENAI_API_KEY and CONFIG.OPENAI_API_KEY != "YOUR_API_KEY"
    # 检查 CONFIG.OPENAI_API_TYPE 是否不存在
    assert not CONFIG.OPENAI_API_TYPE
    # 检查 CONFIG.OPENAI_API_MODEL 是否存在
    assert CONFIG.OPENAI_API_MODEL

    # 调用 oas3_openai_text_to_embedding 函数，传入文本 "Panda emoji"，并等待结果
    result = await oas3_openai_text_to_embedding("Panda emoji")
    # 断言结果存在
    assert result
    # 断言结果中有 model 属性
    assert result.model
    # 断言结果中的 data 列表长度大于 0
    assert len(result.data) > 0
    # 断言结果中的第一个 data 对象的 embedding 属性长度大于 0
    assert len(result.data[0].embedding) > 0

# 如果当前文件被直接执行，则运行 pytest 测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```