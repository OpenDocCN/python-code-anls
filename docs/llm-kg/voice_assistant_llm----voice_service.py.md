# `.\voice_assistant_llm\voice_service.py`

```
# 导入必要的模块：os 用于操作系统相关功能，time 用于时间相关操作，pygame 用于音频播放，gtts 是 Google Text-to-Speech 的库
import os
import time
import pygame
from gtts import gTTS

# 定义一个函数，用于播放文本转语音的内容
def play_text_to_speech(text, language='en', slow=False):
    # 使用 gTTS 库生成语音，text 参数是要转换的文本内容，lang 是语言选择，默认为英语，slow 控制语速，默认为 False
    tts = gTTS(text=text, lang=language, slow=slow)
    
    # 将生成的语音保存为临时的音频文件
    temp_audio_file = "temp_audio.mp3"
    tts.save(temp_audio_file)
    
    # 初始化 pygame 的音频模块
    pygame.mixer.init()
    
    # 加载临时音频文件到 pygame 的音乐对象
    pygame.mixer.music.load(temp_audio_file)
    
    # 播放音乐
    pygame.mixer.music.play()

    # 等待音乐播放结束
    while pygame.mixer.music.get_busy():
        # 设置音乐播放时钟的帧率，每秒钟的帧数为 10
        pygame.time.Clock().tick(10)

    # 停止音乐播放
    pygame.mixer.music.stop()
    
    # 退出 pygame 的音频模块
    pygame.mixer.quit()

    # 等待 3 秒钟，以确保音频文件可以被完全删除
    time.sleep(3)
    
    # 删除临时生成的音频文件
    os.remove(temp_audio_file)
```