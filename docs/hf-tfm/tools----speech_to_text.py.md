# `.\transformers\tools\speech_to_text.py`

```
#!/usr/bin/env python
# 声明 Python 脚本的解释器为在环境变量中搜索的第一个 Python 解释器
# 设定编码格式为 utf-8

# 版权声明
# 版权所有2023年 HuggingFace Inc. 团队保留所有权利
# 根据 Apache 许可证 2.0 版本（“许可证”）许可
# 您只能在符合许可证的情况下使用此文件
# 您可以在以下网址获得许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面协议同意，否则不得分发软件
# 依"原样"的方式分发，没有任何形式的担保或条件，无论是明示的还是暗示的。
# 请查看许可证以了解有关特定语言和限制的详细信息

from ..models.whisper import WhisperForConditionalGeneration, WhisperProcessor
# 从上层目录中的models.whisper模块导入WhisperForConditionalGeneration和WhisperProcessor类
from .base import PipelineTool
# 从当前目录中的base模块导入PipelineTool类


class SpeechToTextTool(PipelineTool):
    # 定义SpeechToTextTool类，继承自PipelineTool类
    default_checkpoint = "openai/whisper-base"
    # 默认的检查点为"openai/whisper-base"
    description = (
        "This is a tool that transcribes an audio into text. It takes an input named `audio` and returns the "
        "transcribed text."
    )
    # 工具的描述信息，将音频转录为文本，以 `audio` 为输入并返回转录的文本
    name = "transcriber"
    # 工具的名称为"transcriber"
    pre_processor_class = WhisperProcessor
    # 前处理器类为WhisperProcessor类
    model_class = WhisperForConditionalGeneration
    # 模型类为WhisperForConditionalGeneration类

    inputs = ["audio"]
    # 输入要求包含音频
    outputs = ["text"]
    # 输出结果为文本

    def encode(self, audio):
        # 定义encode方法，接收参数audio
        return self.pre_processor(audio, return_tensors="pt").input_features
        # 使用pre_processor的方法处理音频，并返回input_features

    def forward(self, inputs):
        # 定义forward方法，接收参数inputs
        return self.model.generate(inputs=inputs)
        # 使用模型生成结果并返回

    def decode(self, outputs):
        # 定义decode方法，接收参数outputs
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        # 使用pre_processor的batch_decode方法解码输出，跳过特殊标记后返回结果的第一个元素
```