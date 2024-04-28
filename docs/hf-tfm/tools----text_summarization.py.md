# `.\transformers\tools\text_summarization.py`

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
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言的权限和限制。
from ..models.auto import AutoModelForSeq2SeqLM, AutoTokenizer
from .base import PipelineTool

# 文本摘要工具类
class TextSummarizationTool(PipelineTool):
    """
    示例：

    ```py
    from transformers.tools import TextSummarizationTool

    summarizer = TextSummarizationTool()
    summarizer(long_text)
    ```py
    """

    # 默认的检查点
    default_checkpoint = "philschmid/bart-large-cnn-samsum"
    # 描述信息
    description = (
        "This is a tool that summarizes an English text. It takes an input `text` containing the text to summarize, "
        "and returns a summary of the text."
    )
    # 工具名称
    name = "summarizer"
    # 预处理器类
    pre_processor_class = AutoTokenizer
    # 模型类
    model_class = AutoModelForSeq2SeqLM

    # 输入和输出
    inputs = ["text"]
    outputs = ["text"]

    # 编码方法
    def encode(self, text):
        return self.pre_processor(text, return_tensors="pt", truncation=True)

    # 前向传播方法
    def forward(self, inputs):
        return self.model.generate(**inputs)[0]

    # 解码方法
    def decode(self, outputs):
        return self.pre_processor.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
```