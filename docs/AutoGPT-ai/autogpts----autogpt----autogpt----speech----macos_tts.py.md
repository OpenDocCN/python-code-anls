# `.\AutoGPT\autogpts\autogpt\autogpt\speech\macos_tts.py`

```py
# 导入必要的模块
from __future__ import annotations
import os
from autogpt.speech.base import VoiceBase

# 定义 MacOS TTS Voice 类，继承自 VoiceBase 类
class MacOSTTS(VoiceBase):
    """MacOS TTS Voice."""

    # 初始化方法，用于设置
    def _setup(self) -> None:
        pass

    # 文本转语音方法，根据 voice_index 参数选择不同的语音播放方式
    def _speech(self, text: str, voice_index: int = 0) -> bool:
        """Play the given text."""
        # 如果 voice_index 为 0，使用默认语音播放给定文本
        if voice_index == 0:
            os.system(f'say "{text}"')
        # 如果 voice_index 为 1，使用 "Ava (Premium)" 语音播放给定文本
        elif voice_index == 1:
            os.system(f'say -v "Ava (Premium)" "{text}"')
        # 其他情况下，使用 "Samantha" 语音播放给定文本
        else:
            os.system(f'say -v Samantha "{text}"')
        # 返回 True 表示成功播放文本
        return True
```