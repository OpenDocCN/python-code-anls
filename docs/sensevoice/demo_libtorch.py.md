# `.\SenseVoiceSmall-src\demo_libtorch.py`

```
#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# 指定脚本使用 Python 3 解释器，并定义文件编码为 UTF-8
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# 从 pathlib 模块导入 Path，用于路径操作
from pathlib import Path
# 从 funasr_torch 导入 SenseVoiceSmall 类，用于加载模型
from funasr_torch import SenseVoiceSmall
# 从 postprocess_utils 导入 rich_transcription_postprocess 函数，用于后处理转录结果
from funasr_torch.utils.postprocess_utils import rich_transcription_postprocess

# 定义模型目录路径
model_dir = "iic/SenseVoiceSmall"

# 创建 SenseVoiceSmall 模型实例，指定模型目录、批处理大小和设备
model = SenseVoiceSmall(model_dir, batch_size=10, device="cuda:0")

# 创建一个包含音频文件路径的列表，使用用户主目录和模型目录构造路径
wav_or_scp = ["{}/.cache/modelscope/hub/{}/example/en.mp3".format(Path.home(), model_dir)]

# 调用模型进行推理，传入音频路径，自动识别语言，并启用文本标准化
res = model(wav_or_scp, language="auto", use_itn=True)
# 打印经过后处理的转录结果列表
print([rich_transcription_postprocess(i) for i in res])
```