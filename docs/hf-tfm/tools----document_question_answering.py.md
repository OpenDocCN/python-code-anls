# `.\transformers\tools\document_question_answering.py`

```py
#!/usr/bin/env python
# coding=utf-8

# 版权声明
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# 根据 Apache 许可证版本 2.0 进行许可
# 除非遵循许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则不得分发软件
# 软件按原样分发，没有担保或条件，无论是明示还是暗示
# 有关特定语言的权限和限制，请参见许可证
import re

from ..models.auto import AutoProcessor
from ..models.vision_encoder_decoder import VisionEncoderDecoderModel
from ..utils import is_vision_available
from .base import PipelineTool


# 检查 PIL 库是否可用
if is_vision_available():
    from PIL import Image


# 文档问答工具类
class DocumentQuestionAnsweringTool(PipelineTool):
    # 默认检查点
    default_checkpoint = "naver-clova-ix/donut-base-finetuned-docvqa"
    # 描述
    description = (
        "This is a tool that answers a question about an document (pdf). It takes an input named `document` which "
        "should be the document containing the information, as well as a `question` that is the question about the "
        "document. It returns a text that contains the answer to the question."
    )
    # 工具名称
    name = "document_qa"
    # 预处理器类
    pre_processor_class = AutoProcessor
    # 模型类
    model_class = VisionEncoderDecoderModel

    # 输入和输出
    inputs = ["image", "text"]
    outputs = ["text"]

    def __init__(self, *args, **kwargs):
        # 如果 PIL 库不可用，则抛出异常
        if not is_vision_available():
            raise ValueError("Pillow must be installed to use the DocumentQuestionAnsweringTool.")

        super().__init__(*args, **kwargs)

    # 编码方法
    def encode(self, document: "Image", question: str):
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
        prompt = task_prompt.replace("{user_input}", question)
        decoder_input_ids = self.pre_processor.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        pixel_values = self.pre_processor(document, return_tensors="pt").pixel_values

        return {"decoder_input_ids": decoder_input_ids, "pixel_values": pixel_values}

    # 前向传播方法
    def forward(self, inputs):
        return self.model.generate(
            inputs["pixel_values"].to(self.device),
            decoder_input_ids=inputs["decoder_input_ids"].to(self.device),
            max_length=self.model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=self.pre_processor.tokenizer.pad_token_id,
            eos_token_id=self.pre_processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.pre_processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        ).sequences
    # 解码输出序列，得到文本序列
    def decode(self, outputs):
        # 使用预处理器批量解码输出，取第一个序列
        sequence = self.pre_processor.batch_decode(outputs)[0]
        # 移除结尾的 EOS 标记
        sequence = sequence.replace(self.pre_processor.tokenizer.eos_token, "")
        # 移除填充的 PAD 标记
        sequence = sequence.replace(self.pre_processor.tokenizer.pad_token, "")
        # 使用正则表达式移除第一个出现的任务开始标记
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        # 将文本序列转换为 JSON 格式
        sequence = self.pre_processor.token2json(sequence)

        # 返回 JSON 格式的答案
        return sequence["answer"]
```