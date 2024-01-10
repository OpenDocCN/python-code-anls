# `MetaGPT\metagpt\learn\text_to_image.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/18
@Author  : mashenquan
@File    : text_to_image.py
@Desc    : Text-to-Image skill, which provides text-to-image functionality.
"""
# 导入base64模块
import base64
# 导入相关配置和常量
from metagpt.config import CONFIG
from metagpt.const import BASE64_FORMAT
# 导入文本转图片的工具模块
from metagpt.tools.metagpt_text_to_image import oas3_metagpt_text_to_image
from metagpt.tools.openai_text_to_image import oas3_openai_text_to_image
# 导入S3工具模块
from metagpt.utils.s3 import S3

# 定义异步函数，将文本转换为图片
async def text_to_image(text, size_type: str = "512x512", openai_api_key="", model_url="", **kwargs):
    """Text to image

    :param text: The text used for image conversion.
    :param openai_api_key: OpenAI API key, For more details, checkout: `https://platform.openai.com/account/api-keys`
    :param size_type: If using OPENAI, the available size options are ['256x256', '512x512', '1024x1024'], while for MetaGPT, the options are ['512x512', '512x768'].
    :param model_url: MetaGPT model url
    :return: The image data is returned in Base64 encoding.
    """
    # 图片声明
    image_declaration = "data:image/png;base64,"
    # 如果存在MetaGPT模型URL或者model_url参数，则调用oas3_metagpt_text_to_image函数
    if CONFIG.METAGPT_TEXT_TO_IMAGE_MODEL_URL or model_url:
        binary_data = await oas3_metagpt_text_to_image(text, size_type, model_url)
    # 如果存在OpenAI API密钥或者openai_api_key参数，则调用oas3_openai_text_to_image函数
    elif CONFIG.OPENAI_API_KEY or openai_api_key:
        binary_data = await oas3_openai_text_to_image(text, size_type)
    else:
        raise ValueError("Missing necessary parameters.")
    # 将二进制数据转换为Base64编码
    base64_data = base64.b64encode(binary_data).decode("utf-8")

    # 实例化S3对象
    s3 = S3()
    # 如果S3对象有效，则缓存数据并返回URL
    url = await s3.cache(data=base64_data, file_ext=".png", format=BASE64_FORMAT) if s3.is_valid else ""
    # 如果存在URL，则返回图片链接
    if url:
        return f"![{text}]({url})"
    # 否则返回Base64编码的图片数据
    return image_declaration + base64_data if base64_data else ""

```