# `MetaGPT\tests\metagpt\tools\test_iflytek_tts.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/26
@Author  : mashenquan
@File    : test_iflytek_tts.py
"""
# 导入 pytest 模块
import pytest

# 从 metagpt.config 模块中导入 CONFIG 对象
from metagpt.config import CONFIG
# 从 metagpt.tools.iflytek_tts 模块中导入 oas3_iflytek_tts 函数

# 使用 pytest.mark.asyncio 装饰器标记异步测试函数
@pytest.mark.asyncio
async def test_tts():
    # Prerequisites
    # 断言 CONFIG.IFLYTEK_APP_ID 存在
    assert CONFIG.IFLYTEK_APP_ID
    # 断言 CONFIG.IFLYTEK_API_KEY 存在
    assert CONFIG.IFLYTEK_API_KEY
    # 断言 CONFIG.IFLYTEK_API_SECRET 存在
    assert CONFIG.IFLYTEK_API_SECRET

    # 调用 oas3_iflytek_tts 函数，传入参数并获取结果
    result = await oas3_iflytek_tts(
        text="你好，hello",
        app_id=CONFIG.IFLYTEK_APP_ID,
        api_key=CONFIG.IFLYTEK_API_KEY,
        api_secret=CONFIG.IFLYTEK_API_SECRET,
    )
    # 断言结果存在
    assert result

# 如果当前文件被直接执行，则运行 pytest 测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```