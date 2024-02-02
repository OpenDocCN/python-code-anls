# `MetaGPT\metagpt\tools\azure_tts.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/6/9 22:22
@Author  : Leo Xiao
@File    : azure_tts.py
@Modified by: mashenquan, 2023/8/17. Azure TTS OAS3 api, which provides text-to-speech functionality
"""
import base64  # 导入base64模块，用于Base64编解码
from pathlib import Path  # 导入Path类，用于处理文件路径
from uuid import uuid4  # 导入uuid4函数，用于生成唯一标识符

import aiofiles  # 导入aiofiles模块，用于异步文件操作
from azure.cognitiveservices.speech import AudioConfig, SpeechConfig, SpeechSynthesizer  # 导入Azure语音服务相关模块

from metagpt.config import CONFIG  # 导入配置模块
from metagpt.logs import logger  # 导入日志模块


class AzureTTS:
    """Azure Text-to-Speech"""

    def __init__(self, subscription_key, region):
        """
        :param subscription_key: key is used to access your Azure AI service API, see: `https://portal.azure.com/` > `Resource Management` > `Keys and Endpoint`
        :param region: This is the location (or region) of your resource. You may need to use this field when making calls to this API.
        """
        self.subscription_key = subscription_key if subscription_key else CONFIG.AZURE_TTS_SUBSCRIPTION_KEY  # 设置subscription_key属性
        self.region = region if region else CONFIG.AZURE_TTS_REGION  # 设置region属性

    # 参数参考：https://learn.microsoft.com/zh-cn/azure/cognitive-services/speech-service/language-support?tabs=tts#voice-styles-and-roles
    async def synthesize_speech(self, lang, voice, text, output_file):
        speech_config = SpeechConfig(subscription=self.subscription_key, region=self.region)  # 创建语音配置对象
        speech_config.speech_synthesis_voice_name = voice  # 设置语音合成的声音名称
        audio_config = AudioConfig(filename=output_file)  # 创建音频配置对象
        synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)  # 创建语音合成器对象

        # More detail: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice
        ssml_string = (
            "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
            f"xml:lang='{lang}' xmlns:mstts='http://www.w3.org/2001/mstts'>"
            f"<voice name='{voice}'>{text}</voice></speak>"
        )  # 构建SSML字符串

        return synthesizer.speak_ssml_async(ssml_string).get()  # 返回语音合成结果

    @staticmethod
    def role_style_text(role, style, text):
        return f'<mstts:express-as role="{role}" style="{style}">{text}</mstts:express-as>'  # 返回带有角色和风格的文本

    @staticmethod
    def role_text(role, text):
        return f'<mstts:express-as role="{role}">{text}</mstts:express-as>'  # 返回带有角色的文本

    @staticmethod
    def style_text(style, text):
        return f'<mstts:express-as style="{style}">{text}</mstts:express-as>'  # 返回带有风格的文本


# Export
async def oas3_azsure_tts(text, lang="", voice="", style="", role="", subscription_key="", region=""):
    """Text to speech
    For more details, check out:`https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts`

    :param lang: The value can contain a language code such as en (English), or a locale such as en-US (English - United States). For more details, checkout: `https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts`
    :param voice: For more details, checkout: `https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts`, `https://speech.microsoft.com/portal/voicegallery`
    :param style: Speaking style to express different emotions like cheerfulness, empathy, and calm. For more details, checkout: `https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts`
    :param role: With roles, the same voice can act as a different age and gender. For more details, checkout: `https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts`
    :param text: The text used for voice conversion.
    :param subscription_key: key is used to access your Azure AI service API, see: `https://portal.azure.com/` > `Resource Management` > `Keys and Endpoint`
    :param region: This is the location (or region) of your resource. You may need to use this field when making calls to this API.
    :return: Returns the Base64-encoded .wav file data if successful, otherwise an empty string.

    """
    if not text:
        return ""  # 如果文本为空，则返回空字符串

    if not lang:
        lang = "zh-CN"  # 如果语言为空，则默认为中文
    if not voice:
        voice = "zh-CN-XiaomoNeural"  # 如果声音为空，则默认为小墨神经
    if not role:
        role = "Girl"  # 如果角色为空，则默认为女孩
    if not style:
        style = "affectionate"  # 如果风格为空，则默认为亲切
    if not subscription_key:
        subscription_key = CONFIG.AZURE_TTS_SUBSCRIPTION_KEY  # 如果订阅密钥为空，则使用配置中的默认值
    if not region:
        region = CONFIG.AZURE_TTS_REGION  # 如果区域为空，则使用配置中的默认值

    xml_value = AzureTTS.role_style_text(role=role, style=style, text=text)  # 构建带有角色和风格的文本
    tts = AzureTTS(subscription_key=subscription_key, region=region)  # 创建AzureTTS对象
    filename = Path(__file__).resolve().parent / (str(uuid4()).replace("-", "") + ".wav")  # 生成唯一的.wav文件名
    try:
        await tts.synthesize_speech(lang=lang, voice=voice, text=xml_value, output_file=str(filename))  # 合成语音
        async with aiofiles.open(filename, mode="rb") as reader:
            data = await reader.read()  # 读取文件数据
            base64_string = base64.b64encode(data).decode("utf-8")  # 将文件数据进行Base64编码
    except Exception as e:
        logger.error(f"text:{text}, error:{e}")  # 记录错误日志
        return ""  # 返回空字符串
    finally:
        filename.unlink(missing_ok=True)  # 删除临时文件

    return base64_string  # 返回Base64编码的.wav文件数据

```