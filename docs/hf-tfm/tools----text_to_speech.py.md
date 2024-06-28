# `.\tools\text_to_speech.py`

```
#!/usr/bin/env python
# coding=utf-8

# 导入 PyTorch 库
import torch

# 从上层目录中导入相应模块和类
from ..models.speecht5 import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
from ..utils import is_datasets_available
from .base import PipelineTool

# 如果 datasets 可用，则从 datasets 库中导入 load_dataset 函数
if is_datasets_available():
    from datasets import load_dataset


# 定义 TextToSpeechTool 类，继承自 PipelineTool 基类
class TextToSpeechTool(PipelineTool):
    # 默认模型检查点
    default_checkpoint = "microsoft/speecht5_tts"
    # 工具描述
    description = (
        "This is a tool that reads an English text out loud. It takes an input named `text` which should contain the "
        "text to read (in English) and returns a waveform object containing the sound."
    )
    # 工具名称
    name = "text_reader"
    # 预处理器类
    pre_processor_class = SpeechT5Processor
    # 模型类
    model_class = SpeechT5ForTextToSpeech
    # 后处理器类
    post_processor_class = SpeechT5HifiGan

    # 输入要求
    inputs = ["text"]
    # 输出结果
    outputs = ["audio"]

    # 设置方法，初始化后处理器
    def setup(self):
        if self.post_processor is None:
            self.post_processor = "microsoft/speecht5_hifigan"
        super().setup()

    # 编码方法，将文本编码为模型输入格式，支持截断处理
    def encode(self, text, speaker_embeddings=None):
        # 使用预处理器将文本编码为输入张量
        inputs = self.pre_processor(text=text, return_tensors="pt", truncation=True)

        # 如果未提供说话者嵌入向量，则加载默认数据集中的说话者嵌入向量
        if speaker_embeddings is None:
            if not is_datasets_available():
                # 如果 datasets 库不可用，则抛出 ImportError
                raise ImportError("Datasets needs to be installed if not passing speaker embeddings.")

            # 加载指定数据集的验证集中的说话者嵌入向量
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            # 获取特定说话者的嵌入向量并添加批处理维度
            speaker_embeddings = torch.tensor(embeddings_dataset[7305]["xvector"]).unsqueeze(0)

        # 返回输入格式化后的字典，包括输入文本和说话者嵌入向量
        return {"input_ids": inputs["input_ids"], "speaker_embeddings": speaker_embeddings}

    # 前向传播方法，使用模型生成语音数据
    def forward(self, inputs):
        # 使用无梯度计算环境执行语音生成
        with torch.no_grad():
            return self.model.generate_speech(**inputs)

    # 解码方法，使用后处理器处理生成的语音输出
    def decode(self, outputs):
        # 使用无梯度计算环境执行后处理器，将输出从 GPU 移动到 CPU，并且断开梯度追踪
        with torch.no_grad():
            return self.post_processor(outputs).cpu().detach()
```