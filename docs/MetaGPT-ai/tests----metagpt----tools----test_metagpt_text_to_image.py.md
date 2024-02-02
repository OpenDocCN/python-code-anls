# `MetaGPT\tests\metagpt\tools\test_metagpt_text_to_image.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/26
@Author  : mashenquan
@File    : test_metagpt_text_to_image.py
"""
# 导入所需的模块
import base64
from unittest.mock import AsyncMock
import pytest
# 导入配置文件中的配置
from metagpt.config import CONFIG
# 导入需要测试的函数
from metagpt.tools.metagpt_text_to_image import oas3_metagpt_text_to_image

# 使用 pytest 的异步测试标记
@pytest.mark.asyncio
async def test_draw(mocker):
    # mock
    # 创建一个模拟的 post 请求
    mock_post = mocker.patch("aiohttp.ClientSession.post")
    # 创建一个模拟的响应
    mock_response = AsyncMock()
    # 设置模拟响应的状态码为 200
    mock_response.status = 200
    # 设置模拟响应的 JSON 数据
    mock_response.json.return_value = {"images": [base64.b64encode(b"success")], "parameters": {"size": 1110}}
    # 设置模拟 post 请求的返回值
    mock_post.return_value.__aenter__.return_value = mock_response

    # Prerequisites
    # 断言配置中是否存在 METAGPT_TEXT_TO_IMAGE_MODEL_URL
    assert CONFIG.METAGPT_TEXT_TO_IMAGE_MODEL_URL

    # 调用 oas3_metagpt_text_to_image 函数，传入参数 "Panda emoji"，并获取返回的二进制数据
    binary_data = await oas3_metagpt_text_to_image("Panda emoji")
    # 断言返回的二进制数据是否存在
    assert binary_data

# 如果当前文件被直接运行，则执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```