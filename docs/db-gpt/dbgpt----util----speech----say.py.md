# `.\DB-GPT-src\dbgpt\util\speech\say.py`

```py
""" Text to speech module """
# 导入必要的模块和库
import threading
from threading import Semaphore

# 导入配置模块和语音引擎模块
from dbgpt._private.config import Config
from dbgpt.util.speech.base import VoiceBase
from dbgpt.util.speech.brian import BrianSpeech
from dbgpt.util.speech.eleven_labs import ElevenLabsSpeech
from dbgpt.util.speech.gtts import GTTSVoice
from dbgpt.util.speech.macos_tts import MacOSTTS

# 控制队列的信号量，用于阻塞主线程前排队的声音数量
_QUEUE_SEMAPHORE = Semaphore(
    1
)  # The amount of sounds to queue before blocking the main thread


def say_text(text: str, voice_index: int = 0) -> None:
    """Speak the given text using the given voice index"""
    # 获取配置信息
    cfg = Config()
    # 获取默认语音引擎和指定的语音引擎
    default_voice_engine, voice_engine = _get_voice_engine(cfg)

    def speak() -> None:
        # 使用指定的语音引擎朗读文本
        success = voice_engine.say(text, voice_index)
        # 如果朗读失败，则使用默认语音引擎朗读文本
        if not success:
            default_voice_engine.say(text)

        # 释放信号量，允许其他线程继续排队
        _QUEUE_SEMAPHORE.release()

    # 获取信号量，阻塞直到有信号量可用
    _QUEUE_SEMAPHORE.acquire(True)
    # 创建线程并启动朗读任务
    thread = threading.Thread(target=speak)
    thread.start()


def _get_voice_engine(config: Config) -> tuple[VoiceBase, VoiceBase]:
    """Get the voice engine to use for the given configuration"""
    # 根据配置选择合适的语音引擎
    default_voice_engine = GTTSVoice()
    if config.elevenlabs_api_key:
        voice_engine = ElevenLabsSpeech()
    elif config.use_mac_os_tts:
        voice_engine = MacOSTTS()
    elif config.use_brian_tts == "True":
        voice_engine = BrianSpeech()
    else:
        voice_engine = GTTSVoice()

    # 返回默认语音引擎和选择的语音引擎
    return default_voice_engine, voice_engine
```