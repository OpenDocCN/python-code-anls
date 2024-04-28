# `.\transformers\tools\text_to_speech.py`

```
#!/usr/bin/env python
# coding=utf-8

# 版权声明
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# 根据 Apache 许可证 2.0 版本授权
# Licensed under the Apache License, Version 2.0 (the "License");
# 除非符合许可证的规定，否则不得使用此文件
# you may not use this file except in compliance with the License.
# 可以在以下网址获取许可证的副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

# 导入自定义模块
from ..models.speecht5 import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
from ..utils import is_datasets_available
from .base import PipelineTool

# 检查是否安装了 datasets 库
if is_datasets_available():
    from datasets import load_dataset

# 定义 TextToSpeechTool 类，继承自 PipelineTool 类
class TextToSpeechTool(PipelineTool):
    default_checkpoint = "microsoft/speecht5_tts"
    description = (
        "This is a tool that reads an English text out loud. It takes an input named `text` which should contain the "
        "text to read (in English) and returns a waveform object containing the sound."
    )
    name = "text_reader"
    pre_processor_class = SpeechT5Processor
    model_class = SpeechT5ForTextToSpeech
    post_processor_class = SpeechT5HifiGan

    inputs = ["text"]
    outputs = ["audio"]

    # 设置方法，用于初始化后处理器
    def setup(self):
        if self.post_processor is None:
            self.post_processor = "microsoft/speecht5_hifigan"
        super().setup()

    # 编码方法，将文本编码为模型输入
    def encode(self, text, speaker_embeddings=None):
        inputs = self.pre_processor(text=text, return_tensors="pt", truncation=True)

        if speaker_embeddings is None:
            if not is_datasets_available():
                raise ImportError("Datasets needs to be installed if not passing speaker embeddings.")

            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7305]["xvector"]).unsqueeze(0)

        return {"input_ids": inputs["input_ids"], "speaker_embeddings": speaker_embeddings}

    # 前向传播方法，生成语音输出
    def forward(self, inputs):
        with torch.no_grad():
            return self.model.generate_speech(**inputs)

    # 解码方法，将模型输出解码为音频
    def decode(self, outputs):
        with torch.no_grad():
            return self.post_processor(outputs).cpu().detach()
```