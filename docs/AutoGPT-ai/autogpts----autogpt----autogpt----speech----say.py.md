# `.\AutoGPT\autogpts\autogpt\autogpt\speech\say.py`

```py
""" Text to speech module """
# 导入必要的模块
from __future__ import annotations

import os
import threading
from threading import Semaphore
from typing import Literal, Optional

from autogpt.core.configuration.schema import SystemConfiguration, UserConfigurable

# 导入其他模块
from .base import VoiceBase
from .eleven_labs import ElevenLabsConfig, ElevenLabsSpeech
from .gtts import GTTSVoice
from .macos_tts import MacOSTTS
from .stream_elements_speech import StreamElementsConfig, StreamElementsSpeech

# 用于控制队列的信号量，防止主线程被阻塞
_QUEUE_SEMAPHORE = Semaphore(
    1
)  # The amount of sounds to queue before blocking the main thread

# 文本到语音配置类，继承自系统配置类
class TTSConfig(SystemConfiguration):
    speak_mode: bool = False
    elevenlabs: Optional[ElevenLabsConfig] = None
    streamelements: Optional[StreamElementsConfig] = None
    provider: Literal[
        "elevenlabs", "gtts", "macos", "streamelements"
    ] = UserConfigurable(
        default="gtts",
        from_env=lambda: os.getenv("TEXT_TO_SPEECH_PROVIDER")
        or (
            "macos"
            if os.getenv("USE_MAC_OS_TTS")
            else "elevenlabs"
            if os.getenv("ELEVENLABS_API_KEY")
            else "streamelements"
            if os.getenv("USE_BRIAN_TTS")
            else "gtts"
        ),
    )  # type: ignore

# 文本到语音提供者类
class TextToSpeechProvider:
    def __init__(self, config: TTSConfig):
        self._config = config
        # 获取默认的语音引擎和当前的语音引擎
        self._default_voice_engine, self._voice_engine = self._get_voice_engine(config)

    # 播放文本的方法
    def say(self, text, voice_index: int = 0) -> None:
        # 内部方法，用于实际播放文本
        def _speak() -> None:
            # 使用当前语音引擎播放文本
            success = self._voice_engine.say(text, voice_index)
            # 如果播放失败，则使用默认语音引擎播放
            if not success:
                self._default_voice_engine.say(text, voice_index)
            # 释放队列信号量
            _QUEUE_SEMAPHORE.release()

        # 如果处于说话模式
        if self._config.speak_mode:
            # 获取队列信号量
            _QUEUE_SEMAPHORE.acquire(True)
            # 创建线程并启动播放文本
            thread = threading.Thread(target=_speak)
            thread.start()
    # 返回对象的字符串表示形式，包括类名和语音引擎名称
    def __repr__(self):
        return "{class_name}(provider={voice_engine_name})".format(
            class_name=self.__class__.__name__,
            voice_engine_name=self._voice_engine.__class__.__name__,
        )

    # 静态方法，根据给定配置获取要使用的语音引擎
    def _get_voice_engine(config: TTSConfig) -> tuple[VoiceBase, VoiceBase]:
        """Get the voice engine to use for the given configuration"""
        # 获取 TTS 提供商
        tts_provider = config.provider
        # 根据不同的提供商选择不同的语音引擎
        if tts_provider == "elevenlabs":
            voice_engine = ElevenLabsSpeech(config.elevenlabs)
        elif tts_provider == "macos":
            voice_engine = MacOSTTS()
        elif tts_provider == "streamelements":
            voice_engine = StreamElementsSpeech(config.streamelements)
        else:
            voice_engine = GTTSVoice()

        # 返回默认语音引擎和选择的语音引擎
        return GTTSVoice(), voice_engine
```