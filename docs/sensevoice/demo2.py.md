# `.\SenseVoiceSmall-src\demo2.py`

```
#!/usr/bin/env python3
# 指定使用的解释器为 Python 3

# -*- encoding: utf-8 -*-
# 设置文件编码为 UTF-8

# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
# 版权声明，指明代码的版权所有者和使用许可

from model import SenseVoiceSmall
# 从 model 模块导入 SenseVoiceSmall 类

from funasr.utils.postprocess_utils import rich_transcription_postprocess
# 从 funasr.utils.postprocess_utils 模块导入 rich_transcription_postprocess 函数

model_dir = "iic/SenseVoiceSmall"
# 设置模型目录的路径

m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
# 从预训练模型加载 SenseVoiceSmall，指定使用 GPU 设备

m.eval()
# 将模型设置为评估模式，以禁用 dropout 等训练时特性

res = m.inference(
    data_in=f"{kwargs['model_path']}/example/en.mp3",
    # 指定输入音频文件的路径
    language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
    # 自动识别语言，也可以手动指定语言
    use_itn=False,
    # 禁用插入式文本规范化
    ban_emo_unk=False,
    # 允许情感未知的情况
    **kwargs,
    # 解包 kwargs 字典，传递额外参数
)

text = rich_transcription_postprocess(res[0][0]["text"])
# 对模型输出的文本进行后处理，提取和转换为最终文本格式

print(text)
# 输出处理后的文本
```