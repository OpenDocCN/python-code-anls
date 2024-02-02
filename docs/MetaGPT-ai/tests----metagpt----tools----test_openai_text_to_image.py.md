# `MetaGPT\tests\metagpt\tools\test_openai_text_to_image.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/26
@Author  : mashenquan
@File    : test_openai_text_to_image.py
"""

# 导入 pytest 模块
import pytest

# 从 metagpt.config 模块中导入 CONFIG 对象
from metagpt.config import CONFIG
# 从 metagpt.tools.openai_text_to_image 模块中导入 OpenAIText2Image 类和 oas3_openai_text_to_image 函数
from metagpt.tools.openai_text_to_image import (
    OpenAIText2Image,
    oas3_openai_text_to_image,
)

# 标记为异步测试
@pytest.mark.asyncio
async def test_draw():
    # 先决条件
    # 检查 CONFIG.OPENAI_API_KEY 是否存在且不等于 "YOUR_API_KEY"
    assert CONFIG.OPENAI_API_KEY and CONFIG.OPENAI_API_KEY != "YOUR_API_KEY"
    # 检查 CONFIG.OPENAI_API_TYPE 是否不存在
    assert not CONFIG.OPENAI_API_TYPE
    # 检查 CONFIG.OPENAI_API_MODEL 是否存在
    assert CONFIG.OPENAI_API_MODEL

    # 调用 oas3_openai_text_to_image 函数，传入文本 "Panda emoji"，获取二进制数据
    binary_data = await oas3_openai_text_to_image("Panda emoji")
    # 断言二进制数据存在
    assert binary_data

# 标记为异步测试
@pytest.mark.asyncio
async def test_get_image():
    # 调用 OpenAIText2Image 类的 get_image_data 方法，传入图片 URL，获取图片数据
    data = await OpenAIText2Image.get_image_data(
        url="https://www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png"
    )
    # 断言图片数据存在
    assert data

# 如果当前模块是主模块
if __name__ == "__main__":
    # 运行 pytest 测试，并输出结果
    pytest.main([__file__, "-s"])

```