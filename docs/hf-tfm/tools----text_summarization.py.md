# `.\tools\text_summarization.py`

```py
# 指定脚本使用的 Python 解释器，并声明编码格式为 UTF-8

# 导入必要的库和模块
from ..models.auto import AutoModelForSeq2SeqLM, AutoTokenizer
from .base import PipelineTool

# 定义文本摘要工具类，继承自 PipelineTool 基类
class TextSummarizationTool(PipelineTool):
    """
    Example:

    ```
    from transformers.tools import TextSummarizationTool

    summarizer = TextSummarizationTool()
    summarizer(long_text)
    ```
    """

    # 默认使用的模型检查点
    default_checkpoint = "philschmid/bart-large-cnn-samsum"
    # 工具的描述信息
    description = (
        "This is a tool that summarizes an English text. It takes an input `text` containing the text to summarize, "
        "and returns a summary of the text."
    )
    # 工具的名称
    name = "summarizer"
    # 预处理器类，用于处理输入文本
    pre_processor_class = AutoTokenizer
    # 模型类，用于生成摘要
    model_class = AutoModelForSeq2SeqLM

    # 输入数据的名称列表
    inputs = ["text"]
    # 输出数据的名称列表
    outputs = ["text"]

    # 对输入文本进行编码的方法，使用预处理器返回 PyTorch 张量，并进行截断处理
    def encode(self, text):
        return self.pre_processor(text, return_tensors="pt", truncation=True)

    # 执行前向传播的方法，使用模型生成摘要
    def forward(self, inputs):
        return self.model.generate(**inputs)[0]

    # 对生成的输出进行解码的方法，跳过特殊符号并清理分词空格
    def decode(self, outputs):
        return self.pre_processor.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
```