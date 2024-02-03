# `.\AutoGPT\autogpts\autogpt\autogpt\speech\eleven_labs.py`

```py
"""ElevenLabs speech module"""
# 导入必要的模块
from __future__ import annotations

import logging
import os

import requests
from playsound import playsound

from autogpt.core.configuration import SystemConfiguration, UserConfigurable

# 导入 VoiceBase 类
from .base import VoiceBase

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义占位符
PLACEHOLDERS = {"your-voice-id"}


# 定义 ElevenLabsConfig 类，继承自 SystemConfiguration
class ElevenLabsConfig(SystemConfiguration):
    # 定义 api_key 属性，可从环境变量 ELEVENLABS_API_KEY 中获取
    api_key: str = UserConfigurable(from_env="ELEVENLABS_API_KEY")
    # 定义 voice_id 属性，可从环境变量 ELEVENLABS_VOICE_ID 中获取
    voice_id: str = UserConfigurable(from_env="ELEVENLABS_VOICE_ID")


# 定义 ElevenLabsSpeech 类，继承自 VoiceBase
class ElevenLabsSpeech(VoiceBase):
    """ElevenLabs speech class"""

    # 定义 _setup 方法，用于设置声音、API 密钥等
    def _setup(self, config: ElevenLabsConfig) -> None:
        """Set up the voices, API key, etc.

        Returns:
            None: None
        """

        # 默认声音列表
        default_voices = ["ErXwobaYiN019PkySvjV", "EXAVITQu4vr4xnSDxMaL"]
        # 声音选项字典
        voice_options = {
            "Rachel": "21m00Tcm4TlvDq8ikWAM",
            "Domi": "AZnzlk1XvdvUeBnXmlld",
            "Bella": "EXAVITQu4vr4xnSDxMaL",
            "Antoni": "ErXwobaYiN019PkySvjV",
            "Elli": "MF3mGyEYCl7XYWbV9V6O",
            "Josh": "TxGEqnHWrfWFTfGW9XjX",
            "Arnold": "VR6AewLTigWG4xSOukaG",
            "Adam": "pNInz6obpgDQGcFmaJgB",
            "Sam": "yoZ06aMxZJJ28mfd3POQ",
        }
        # 设置请求头信息
        self._headers = {
            "Content-Type": "application/json",
            "xi-api-key": config.api_key,
        }
        # 复制默认声音列表
        self._voices = default_voices.copy()
        # 如果配置中的声音 ID 在声音选项中，则将其替换为对应的声音 ID
        if config.voice_id in voice_options:
            config.voice_id = voice_options[config.voice_id]
        # 使用自定义声音
        self._use_custom_voice(config.voice_id, 0)
    def _use_custom_voice(self, voice, voice_index) -> None:
        """Use a custom voice if provided and not a placeholder

        Args:
            voice (str): The voice ID
            voice_index (int): The voice index

        Returns:
            None: None
        """
        # 如果 voice 不为空且不是占位符，则将其设置为指定索引处的自定义声音
        if voice and voice not in PLACEHOLDERS:
            self._voices[voice_index] = voice

    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """Speak text using elevenlabs.io's API

        Args:
            text (str): The text to speak
            voice_index (int, optional): The voice to use. Defaults to 0.

        Returns:
            bool: True if the request was successful, False otherwise
        """
        # 构建文本转语音的 API 请求 URL
        tts_url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{self._voices[voice_index]}"
        )
        # 发送 POST 请求，传递文本内容
        response = requests.post(tts_url, headers=self._headers, json={"text": text})

        # 如果请求成功，将响应内容写入文件 speech.mpeg，播放该文件，然后删除该文件
        if response.status_code == 200:
            with open("speech.mpeg", "wb") as f:
                f.write(response.content)
            playsound("speech.mpeg", True)
            os.remove("speech.mpeg")
            return True
        else:
            # 如果请求失败，记录警告信息和响应内容，返回 False
            logger.warning("Request failed with status code:", response.status_code)
            logger.info("Response content:", response.content)
            return False
```