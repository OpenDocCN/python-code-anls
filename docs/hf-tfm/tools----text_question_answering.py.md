# `.\transformers\tools\text_question_answering.py`

```py
#!/usr/bin/env python
# coding=utf-8

# 版权声明
# 版权所有 © 2023 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言的权限和限制。
from ..models.auto import AutoModelForSeq2SeqLM, AutoTokenizer
from .base import PipelineTool

# 问题回答的提示信息模板
QA_PROMPT = """Here is a text containing a lot of information: '''{text}'''.

Can you answer this question about the text: '{question}'"""

# 文本问题回答工具类
class TextQuestionAnsweringTool(PipelineTool):
    # 默认的检查点
    default_checkpoint = "google/flan-t5-base"
    # 描述信息
    description = (
        "This is a tool that answers questions related to a text. It takes two arguments named `text`, which is the "
        "text where to find the answer, and `question`, which is the question, and returns the answer to the question."
    )
    # 工具名称
    name = "text_qa"
    # 预处理器类
    pre_processor_class = AutoTokenizer
    # 模型类
    model_class = AutoModelForSeq2SeqLM

    # 输入参数
    inputs = ["text", "text"]
    # 输出参数
    outputs = ["text"]

    # 编码方法
    def encode(self, text: str, question: str):
        # 根据模板生成问题回答的提示信息
        prompt = QA_PROMPT.format(text=text, question=question)
        return self.pre_processor(prompt, return_tensors="pt")

    # 前向传播方法
    def forward(self, inputs):
        # 生成输出序列
        output_ids = self.model.generate(**inputs)

        # 获取输入序列的批量大小和输出序列的批量大小
        in_b, _ = inputs["input_ids"].shape
        out_b = output_ids.shape[0]

        # 重塑输出序列并返回第一个结果
        return output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])[0][0]

    # 解码方法
    def decode(self, outputs):
        return self.pre_processor.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
```