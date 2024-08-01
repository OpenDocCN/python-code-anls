# `.\DB-GPT-src\dbgpt\util\speech\eleven_labs.py`

```py
"""ElevenLabs speech module"""
# 导入必要的库和模块
import logging
import os

import requests

# 导入配置类和基础语音类
from dbgpt._private.config import Config
from dbgpt.util.speech.base import VoiceBase

# 定义占位符集合
PLACEHOLDERS = {"your-voice-id"}

# 获取日志记录器
logger = logging.getLogger(__name__)


class ElevenLabsSpeech(VoiceBase):
    """ElevenLabs speech class"""

    def _setup(self) -> None:
        """Set up the voices, API key, etc.

        Returns:
            None: None
        """

        # 实例化配置对象
        cfg = Config()

        # 默认语音选项
        default_voices = ["ErXwobaYiN019PkySvjV", "EXAVITQu4vr4xnSDxMaL"]

        # 声音选项映射
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

        # 设置 HTTP 请求头部
        self._headers = {
            "Content-Type": "application/json",
            "xi-api_v1-key": cfg.elevenlabs_api_key,
        }

        # 复制默认声音列表
        self._voices = default_voices.copy()

        # 如果配置中的语音 ID 在声音选项中，则替换为对应的 ID
        if cfg.elevenlabs_voice_1_id in voice_options:
            cfg.elevenlabs_voice_1_id = voice_options[cfg.elevenlabs_voice_1_id]
        if cfg.elevenlabs_voice_2_id in voice_options:
            cfg.elevenlabs_voice_2_id = voice_options[cfg.elevenlabs_voice_2_id]

        # 使用自定义声音设置
        self._use_custom_voice(cfg.elevenlabs_voice_1_id, 0)
        self._use_custom_voice(cfg.elevenlabs_voice_2_id, 1)

    def _use_custom_voice(self, voice, voice_index) -> None:
        """Use a custom voice if provided and not a placeholder

        Args:
            voice (str): The voice ID
            voice_index (int): The voice index

        Returns:
            None: None
        """
        # 检查是否为占位符值，如果不是则使用给定的声音 ID
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
        # 导入播放音频的模块
        from playsound import playsound

        # 构造 TTS 请求 URL
        tts_url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{self._voices[voice_index]}"
        )

        # 发送 TTS POST 请求
        response = requests.post(tts_url, headers=self._headers, json={"text": text})

        # 检查响应状态码
        if response.status_code == 200:
            # 将响应内容写入音频文件
            with open("speech.mpeg", "wb") as f:
                f.write(response.content)
            
            # 播放生成的音频文件
            playsound("speech.mpeg", True)
            
            # 删除临时生成的音频文件
            os.remove("speech.mpeg")
            
            return True
        else:
            # 记录警告日志并输出响应内容
            logger.warn("Request failed with status code:", response.status_code)
            logger.info("Response content:", response.content)
            
            return False
```