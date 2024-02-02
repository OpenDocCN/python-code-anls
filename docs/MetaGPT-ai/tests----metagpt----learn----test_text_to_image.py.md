# `MetaGPT\tests\metagpt\learn\test_text_to_image.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/18
@Author  : mashenquan
@File    : test_text_to_image.py
@Desc    : Unit tests.
"""

# 导入 pytest 模块
import pytest

# 从 metagpt.config 模块中导入 CONFIG 对象
from metagpt.config import CONFIG
# 从 metagpt.learn.text_to_image 模块中导入 text_to_image 函数
from metagpt.learn.text_to_image import text_to_image

# 标记异步测试
@pytest.mark.asyncio
async def test_metagpt_llm():
    # 先决条件
    assert CONFIG.METAGPT_TEXT_TO_IMAGE_MODEL_URL
    assert CONFIG.OPENAI_API_KEY

    # 调用 text_to_image 函数，传入文本和图片尺寸类型参数
    data = await text_to_image("Panda emoji", size_type="512x512")
    # 断言返回的数据中包含 base64 或 http
    assert "base64" in data or "http" in data

    # 模拟会话环境
    old_options = CONFIG.options.copy()
    new_options = old_options.copy()
    new_options["METAGPT_TEXT_TO_IMAGE_MODEL_URL"] = None
    CONFIG.set_context(new_options)
    try:
        # 再次调用 text_to_image 函数，传入文本和图片尺寸类型参数
        data = await text_to_image("Panda emoji", size_type="512x512")
        # 断言返回的数据中包含 base64 或 http
        assert "base64" in data or "http" in data
    finally:
        CONFIG.set_context(old_options)

# 如果当前文件被直接执行，则执行 pytest 测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```