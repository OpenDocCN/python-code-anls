# `.\AutoGPT\autogpts\autogpt\autogpt\speech\__init__.py`

```py
# 这个模块包含了语音识别和语音合成的函数
"""This module contains the speech recognition and speech synthesis functions."""

# 导入自动语音生成模块中的文本到语音提供者和语音合成配置
from autogpt.speech.say import TextToSpeechProvider, TTSConfig

# 暴露给外部的模块成员，包括文本到语音提供者和语音合成配置
__all__ = ["TextToSpeechProvider", "TTSConfig"]
```