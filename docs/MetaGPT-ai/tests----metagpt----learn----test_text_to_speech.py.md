# `MetaGPT\tests\metagpt\learn\test_text_to_speech.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/18
@Author  : mashenquan
@File    : test_text_to_speech.py
@Desc    : Unit tests.
"""

# 导入 pytest 模块
import pytest

# 从 metagpt.config 模块中导入 CONFIG 对象
from metagpt.config import CONFIG
# 从 metagpt.learn.text_to_speech 模块中导入 text_to_speech 函数
from metagpt.learn.text_to_speech import text_to_speech

# 使用 pytest.mark.asyncio 装饰器标记异步测试函数
@pytest.mark.asyncio
async def test_text_to_speech():
    # 前提条件
    assert CONFIG.IFLYTEK_APP_ID
    assert CONFIG.IFLYTEK_API_KEY
    assert CONFIG.IFLYTEK_API_SECRET
    assert CONFIG.AZURE_TTS_SUBSCRIPTION_KEY and CONFIG.AZURE_TTS_SUBSCRIPTION_KEY != "YOUR_API_KEY"
    assert CONFIG.AZURE_TTS_REGION

    # 测试 Azure TTS
    data = await text_to_speech("panda emoji")
    assert "base64" in data or "http" in data

    # 测试讯飞 TTS
    ## 模拟会话环境
    old_options = CONFIG.options.copy()
    new_options = old_options.copy()
    new_options["AZURE_TTS_SUBSCRIPTION_KEY"] = ""
    CONFIG.set_context(new_options)
    try:
        data = await text_to_speech("panda emoji")
        assert "base64" in data or "http" in data
    finally:
        CONFIG.set_context(old_options)

# 如果当前脚本为主程序，则执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```