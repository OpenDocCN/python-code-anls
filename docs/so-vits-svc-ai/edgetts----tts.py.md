# `so-vits-svc\edgetts\tts.py`

```
# 导入必要的模块
import asyncio
import random
import sys

import edge_tts
from edge_tts import VoicesManager
from langdetect import DetectorFactory, detect

# 设置语言检测器的种子
DetectorFactory.seed = 0

# 从命令行参数获取文本内容、语言、语速、音量和性别（可选）
TEXT = sys.argv[1]
LANG = detect(TEXT) if sys.argv[2] == "Auto" else sys.argv[2]
RATE = sys.argv[3]
VOLUME = sys.argv[4]
GENDER = sys.argv[5] if len(sys.argv) == 6 else None
OUTPUT_FILE = "tts.wav"

# 打印 TTS 运行信息
print("Running TTS...")
print(f"Text: {TEXT}, Language: {LANG}, Gender: {GENDER}, Rate: {RATE}, Volume: {VOLUME}")

# 定义异步函数
async def _main() -> None:
    # 创建语音管理器对象
    voices = await VoicesManager.create()
    if GENDER is not None:
        # 如果指定了性别，根据语言和性别查找对应的语音
        if LANG == "zh-cn" or LANG == "zh-tw":
            LOCALE = LANG[:-2] + LANG[-2:].upper()
            voice = voices.find(Gender=GENDER, Locale=LOCALE)
        else:
            voice = voices.find(Gender=GENDER, Language=LANG)
        # 从匹配的语音中随机选择一个
        VOICE = random.choice(voice)["Name"]
        print(f"Using random {LANG} voice: {VOICE}")
    else:
        # 如果未指定性别，则使用语言作为语音
        VOICE = LANG
        
    # 创建 TTS 通信对象，并保存为音频文件
    communicate = edge_tts.Communicate(text = TEXT, voice = VOICE, rate = RATE, volume = VOLUME)
    await communicate.save(OUTPUT_FILE)

# 如果作为脚本直接执行
if __name__ == "__main__":
    # 如果运行在 Windows 平台
    if sys.platform.startswith("win"):
        # 设置 Windows 平台的事件循环策略，并运行异步函数
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(_main())
    else:
        # 获取事件循环对象，并运行异步函数
        loop = asyncio.get_event_loop_policy().get_event_loop()
        try:
            loop.run_until_complete(_main())
        finally:
            loop.close()
```