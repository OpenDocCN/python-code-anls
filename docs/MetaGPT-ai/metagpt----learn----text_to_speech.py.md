# `MetaGPT\metagpt\learn\text_to_speech.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/17
@Author  : mashenquan
@File    : text_to_speech.py
@Desc    : Text-to-Speech skill, which provides text-to-speech functionality
"""

# 从metagpt.config模块导入CONFIG对象
from metagpt.config import CONFIG
# 从metagpt.const模块导入BASE64_FORMAT常量
from metagpt.const import BASE64_FORMAT
# 从metagpt.tools.azure_tts模块导入oas3_azsure_tts函数
from metagpt.tools.azure_tts import oas3_azsure_tts
# 从metagpt.tools.iflytek_tts模块导入oas3_iflytek_tts函数
from metagpt.tools.iflytek_tts import oas3_iflytek_tts
# 从metagpt.utils.s3模块导入S3类
from metagpt.utils.s3 import S3

# 定义异步函数text_to_speech，接收多个参数
async def text_to_speech(
    text,
    lang="zh-CN",
    voice="zh-CN-XiaomoNeural",
    style="affectionate",
    role="Girl",
    subscription_key="",
    region="",
    iflytek_app_id="",
    iflytek_api_key="",
    iflytek_api_secret="",
    **kwargs,
):
    """Text to speech
    For more details, check out:`https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts`

    :param lang: The value can contain a language code such as en (English), or a locale such as en-US (English - United States). For more details, checkout: `https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts`
    :param voice: For more details, checkout: `https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts`, `https://speech.microsoft.com/portal/voicegallery`
    :param style: Speaking style to express different emotions like cheerfulness, empathy, and calm. For more details, checkout: `https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts`
    :param role: With roles, the same voice can act as a different age and gender. For more details, checkout: `https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts`
    :param text: The text used for voice conversion.
    :param subscription_key: key is used to access your Azure AI service API, see: `https://portal.azure.com/` > `Resource Management` > `Keys and Endpoint`
    :param region: This is the location (or region) of your resource. You may need to use this field when making calls to this API.
    :param iflytek_app_id: Application ID is used to access your iFlyTek service API, see: `https://console.xfyun.cn/services/tts`
    :param iflytek_api_key: WebAPI argument, see: `https://console.xfyun.cn/services/tts`
    :param iflytek_api_secret: WebAPI argument, see: `https://console.xfyun.cn/services/tts`
    :return: Returns the Base64-encoded .wav/.mp3 file data if successful, otherwise an empty string.

    """

    # 检查是否有Azure TTS的订阅密钥和区域，或者传入的参数中是否包含订阅密钥和区域
    if (CONFIG.AZURE_TTS_SUBSCRIPTION_KEY and CONFIG.AZURE_TTS_REGION) or (subscription_key and region):
        # 设置音频声明为wav格式的Base64编码
        audio_declaration = "data:audio/wav;base64,"
        # 调用oas3_azsure_tts函数获取Base64编码的音频数据
        base64_data = await oas3_azsure_tts(text, lang, voice, style, role, subscription_key, region)
        # 创建S3对象
        s3 = S3()
        # 如果S3对象有效，则缓存数据并返回URL
        url = await s3.cache(data=base64_data, file_ext=".wav", format=BASE64_FORMAT) if s3.is_valid else ""
        # 如果URL存在，则返回带URL的文本
        if url:
            return f"[{text}]({url})"
        # 如果Base64编码的音频数据存在，则返回Base64编码的音频数据
        return audio_declaration + base64_data if base64_data else base64_data
    # 检查是否有讯飞TTS的应用ID、API密钥和API密钥，或者传入的参数中是否包含这些信息
    if (CONFIG.IFLYTEK_APP_ID and CONFIG.IFLYTEK_API_KEY and CONFIG.IFLYTEK_API_SECRET) or (
        iflytek_app_id and iflytek_api_key and iflytek_api_secret
    ):
        # 设置音频声明为mp3格式的Base64编码
        audio_declaration = "data:audio/mp3;base64,"
        # 调用oas3_iflytek_tts函数获取Base64编码的音频数据
        base64_data = await oas3_iflytek_tts(
            text=text, app_id=iflytek_app_id, api_key=iflytek_api_key, api_secret=iflytek_api_secret
        )
        # 创建S3对象
        s3 = S3()
        # 如果S3对象有效，则缓存数据并返回URL
        url = await s3.cache(data=base64_data, file_ext=".mp3", format=BASE64_FORMAT) if s3.is_valid else ""
        # 如果URL存在，则返回带URL的文本
        if url:
            return f"[{text}]({url})"
        # 如果Base64编码的音频数据存在，则返回Base64编码的音频数据
        return audio_declaration + base64_data if base64_data else base64_data

    # 如果以上条件都不满足，则抛出值错误
    raise ValueError(
        "AZURE_TTS_SUBSCRIPTION_KEY, AZURE_TTS_REGION, IFLYTEK_APP_ID, IFLYTEK_API_KEY, IFLYTEK_API_SECRET error"
    )

```