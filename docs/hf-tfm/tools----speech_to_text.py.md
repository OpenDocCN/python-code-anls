# `.\tools\speech_to_text.py`

```py
#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入必要的模块和类
from ..models.whisper import WhisperForConditionalGeneration, WhisperProcessor
from .base import PipelineTool

# 定义一个继承自PipelineTool的子类SpeechToTextTool
class SpeechToTextTool(PipelineTool):
    # 默认的模型检查点路径
    default_checkpoint = "openai/whisper-base"
    # 工具的描述信息
    description = (
        "This is a tool that transcribes an audio into text. It takes an input named `audio` and returns the "
        "transcribed text."
    )
    # 工具的名称
    name = "transcriber"
    # 预处理器类，用于处理输入数据
    pre_processor_class = WhisperProcessor
    # 模型类，用于生成输出
    model_class = WhisperForConditionalGeneration

    # 定义输入的名称列表
    inputs = ["audio"]
    # 定义输出的名称列表
    outputs = ["text"]

    # 编码方法，将输入的音频转换成模型可以处理的张量形式
    def encode(self, audio):
        return self.pre_processor(audio, return_tensors="pt").input_features

    # 前向传播方法，使用模型生成输出
    def forward(self, inputs):
        return self.model.generate(inputs=inputs)

    # 解码方法，将模型输出的张量转换成文本形式
    def decode(self, outputs):
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0]
```