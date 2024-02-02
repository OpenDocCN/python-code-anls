# `MetaGPT\metagpt\tools\metagpt_text_to_image.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/18
@Author  : mashenquan
@File    : metagpt_text_to_image.py
@Desc    : MetaGPT Text-to-Image OAS3 api, which provides text-to-image functionality.
"""
# 导入所需的模块
import base64
from typing import Dict, List
import aiohttp
import requests
from pydantic import BaseModel
# 导入自定义的配置和日志模块
from metagpt.config import CONFIG
from metagpt.logs import logger

# 定义 MetaGPTText2Image 类
class MetaGPTText2Image:
    def __init__(self, model_url):
        """
        :param model_url: Model reset api url
        """
        # 初始化函数，接收模型重置 API 的 URL
        self.model_url = model_url if model_url else CONFIG.METAGPT_TEXT_TO_IMAGE_MODEL

    async def text_2_image(self, text, size_type="512x512"):
        """Text to image

        :param text: The text used for image conversion.
        :param size_type: One of ['512x512', '512x768']
        :return: The image data is returned in Base64 encoding.
        """
        # 将文本转换为图片
        headers = {"Content-Type": "application/json"}
        dims = size_type.split("x")
        data = {
            # 定义图片生成的参数
        }

        class ImageResult(BaseModel):
            images: List
            parameters: Dict

        try:
            # 使用 aiohttp 发送异步请求
            async with aiohttp.ClientSession() as session:
                async with session.post(self.model_url, headers=headers, json=data) as response:
                    result = ImageResult(**await response.json())
            if len(result.images) == 0:
                return 0
            data = base64.b64decode(result.images[0])
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred:{e}")
        return 0


# 导出函数
async def oas3_metagpt_text_to_image(text, size_type: str = "512x512", model_url=""):
    """Text to image

    :param text: The text used for image conversion.
    :param model_url: Model reset api
    :param size_type: One of ['512x512', '512x768']
    :return: The image data is returned in Base64 encoding.
    """
    if not text:
        return ""
    if not model_url:
        model_url = CONFIG.METAGPT_TEXT_TO_IMAGE_MODEL_URL
    return await MetaGPTText2Image(model_url).text_2_image(text, size_type=size_type)

```