# `.\tools\text_question_answering.py`

```
#!/usr/bin/env python
# coding=utf-8

# 导入必要的模块和类
from ..models.auto import AutoModelForSeq2SeqLM, AutoTokenizer
from .base import PipelineTool

# 定义一个包含占位符文本和问题的模板字符串
QA_PROMPT = """Here is a text containing a lot of information: '''{text}'''.

Can you answer this question about the text: '{question}'"""

# 定义一个工具类，继承自PipelineTool基类，用于文本问答任务
class TextQuestionAnsweringTool(PipelineTool):
    # 默认使用的模型的检查点名称
    default_checkpoint = "google/flan-t5-base"
    # 工具的描述信息
    description = (
        "This is a tool that answers questions related to a text. It takes two arguments named `text`, which is the "
        "text where to find the answer, and `question`, which is the question, and returns the answer to the question."
    )
    # 工具的名称
    name = "text_qa"
    # 预处理器使用的类别
    pre_processor_class = AutoTokenizer
    # 模型使用的类别
    model_class = AutoModelForSeq2SeqLM

    # 输入参数列表，包括文本和问题
    inputs = ["text", "text"]
    # 输出参数列表，只有文本答案
    outputs = ["text"]

    # 编码函数，将文本和问题格式化为模型输入
    def encode(self, text: str, question: str):
        # 根据模板生成特定格式的问题提示文本
        prompt = QA_PROMPT.format(text=text, question=question)
        # 使用预处理器处理文本，并返回PyTorch张量格式的输入
        return self.pre_processor(prompt, return_tensors="pt")

    # 前向推理函数，执行模型生成文本答案
    def forward(self, inputs):
        # 使用模型生成输出标识符
        output_ids = self.model.generate(**inputs)

        # 计算输入和输出张量的形状信息
        in_b, _ = inputs["input_ids"].shape
        out_b = output_ids.shape[0]

        # 重新整形输出张量，保证符合预期的格式
        return output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])[0][0]

    # 解码函数，将模型输出的标识符转换为文本答案
    def decode(self, outputs):
        # 使用预处理器解码，去除特殊标记并清理空白字符
        return self.pre_processor.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
```