# `.\tools\document_question_answering.py`

```
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
import re  # 导入正则表达式模块

from ..models.auto import AutoProcessor  # 导入自动处理器模块
from ..models.vision_encoder_decoder import VisionEncoderDecoderModel  # 导入视觉编码解码模型
from ..utils import is_vision_available  # 导入视觉功能可用性检查函数
from .base import PipelineTool  # 导入流水线工具基类


if is_vision_available():  # 如果视觉功能可用
    from PIL import Image  # 导入图像处理库PIL中的Image模块


class DocumentQuestionAnsweringTool(PipelineTool):
    default_checkpoint = "naver-clova-ix/donut-base-finetuned-docvqa"  # 默认检查点路径
    description = (
        "This is a tool that answers a question about an document (pdf). It takes an input named `document` which "
        "should be the document containing the information, as well as a `question` that is the question about the "
        "document. It returns a text that contains the answer to the question."
    )  # 工具描述
    name = "document_qa"  # 工具名称
    pre_processor_class = AutoProcessor  # 预处理器类
    model_class = VisionEncoderDecoderModel  # 模型类

    inputs = ["image", "text"]  # 输入类型：图像、文本
    outputs = ["text"]  # 输出类型：文本

    def __init__(self, *args, **kwargs):
        if not is_vision_available():  # 如果视觉功能不可用
            raise ValueError("Pillow must be installed to use the DocumentQuestionAnsweringTool.")  # 抛出数值错误

        super().__init__(*args, **kwargs)  # 调用父类初始化方法

    def encode(self, document: "Image", question: str):
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"  # 任务提示字符串模板
        prompt = task_prompt.replace("{user_input}", question)  # 根据问题替换用户输入
        decoder_input_ids = self.pre_processor.tokenizer(  # 使用预处理器的分词器对提示进行编码
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        pixel_values = self.pre_processor(document, return_tensors="pt").pixel_values  # 使用预处理器处理文档图像，获取像素值

        return {"decoder_input_ids": decoder_input_ids, "pixel_values": pixel_values}  # 返回编码的输入数据字典

    def forward(self, inputs):
        return self.model.generate(  # 使用模型生成答案
            inputs["pixel_values"].to(self.device),  # 图像像素值移到指定设备
            decoder_input_ids=inputs["decoder_input_ids"].to(self.device),  # 解码器输入IDs移到指定设备
            max_length=self.model.decoder.config.max_position_embeddings,  # 最大生成长度
            early_stopping=True,  # 开启早停机制
            pad_token_id=self.pre_processor.tokenizer.pad_token_id,  # 填充标记ID
            eos_token_id=self.pre_processor.tokenizer.eos_token_id,  # 结束标记ID
            use_cache=True,  # 使用缓存
            num_beams=1,  # 波束搜索数量
            bad_words_ids=[[self.pre_processor.tokenizer.unk_token_id]],  # 不良词汇ID
            return_dict_in_generate=True,  # 生成时返回字典格式
        ).sequences  # 返回生成的序列
    # 定义一个方法 `decode`，接收 `self` 和 `outputs` 作为参数
    def decode(self, outputs):
        # 使用预处理器的方法对输出进行批量解码，并取第一个序列
        sequence = self.pre_processor.batch_decode(outputs)[0]
        # 替换序列中的结束标记为""
        sequence = sequence.replace(self.pre_processor.tokenizer.eos_token, "")
        # 替换序列中的填充标记为""
        sequence = sequence.replace(self.pre_processor.tokenizer.pad_token, "")
        # 使用正则表达式移除序列中第一个出现的任务开始标记
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        # 使用预处理器的方法将序列转换为 JSON 格式
        sequence = self.pre_processor.token2json(sequence)

        # 返回 JSON 格式结果中的答案部分
        return sequence["answer"]
```