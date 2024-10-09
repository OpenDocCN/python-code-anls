# `.\SenseVoiceSmall-src\demo_onnx.py`

```
#!/usr/bin/env python3
# 指定脚本的解释器为 Python 3
# -*- encoding: utf-8 -*-
# 指定文件的编码为 UTF-8
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
# 指明版权信息及其链接
#  MIT License  (https://opensource.org/licenses/MIT)
# 指明使用的许可证及其链接

from pathlib import Path
# 从 pathlib 模块导入 Path，用于处理文件路径
from funasr_onnx import SenseVoiceSmall
# 从 funasr_onnx 模块导入 SenseVoiceSmall 类，表示语音模型
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
# 从 funasr_onnx.utils.postprocess_utils 导入 rich_transcription_postprocess 函数，用于后处理转录结果

model_dir = "iic/SenseVoiceSmall"
# 定义模型目录的路径

model = SenseVoiceSmall(model_dir, batch_size=10, quantize=True)
# 实例化 SenseVoiceSmall 模型，指定模型目录、批处理大小和量化选项

# inference
# 注释说明开始进行推理

wav_or_scp = ["{}/.cache/modelscope/hub/{}/example/en.mp3".format(Path.home(), model_dir)]
# 构建包含音频文件路径的列表，路径由用户主目录和模型目录组成

res = model(wav_or_scp, language="auto", use_itn=True)
# 调用模型进行推理，传入音频路径，自动检测语言并启用文本规范化

print([rich_transcription_postprocess(i) for i in res])
# 输出每个结果经过 rich_transcription_postprocess 后的转录文本
```