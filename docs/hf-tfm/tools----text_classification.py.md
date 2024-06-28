# `.\tools\text_classification.py`

```py
#!/usr/bin/env python
# coding=utf-8

# 导入torch模块
import torch

# 从上级目录中导入AutoModelForSequenceClassification和AutoTokenizer类
from ..models.auto import AutoModelForSequenceClassification, AutoTokenizer

# 从base模块中导入PipelineTool类
from .base import PipelineTool


class TextClassificationTool(PipelineTool):
    """
    文本分类工具类，继承自PipelineTool基类。

    Example:

    ```
    from transformers.tools import TextClassificationTool

    classifier = TextClassificationTool()
    classifier("This is a super nice API!", labels=["positive", "negative"])
    ```
    """

    # 默认的预训练模型
    default_checkpoint = "facebook/bart-large-mnli"
    # 工具描述信息
    description = (
        "This is a tool that classifies an English text using provided labels. It takes two inputs: `text`, which "
        "should be the text to classify, and `labels`, which should be the list of labels to use for classification. "
        "It returns the most likely label in the list of provided `labels` for the input text."
    )
    # 工具名称
    name = "text_classifier"
    # 预处理器类，使用AutoTokenizer
    pre_processor_class = AutoTokenizer
    # 模型类，使用AutoModelForSequenceClassification

    model_class = AutoModelForSequenceClassification

    # 输入参数列表
    inputs = ["text", ["text"]]
    # 输出参数列表
    outputs = ["text"]

    def setup(self):
        # 调用父类的setup方法
        super().setup()
        # 获取模型配置
        config = self.model.config
        # 初始化entailment_id为-1
        self.entailment_id = -1
        # 遍历id2label字典，找到以"entail"开头的标签对应的索引
        for idx, label in config.id2label.items():
            if label.lower().startswith("entail"):
                self.entailment_id = int(idx)
        # 如果未找到对应的entailment标签，抛出数值错误异常
        if self.entailment_id == -1:
            raise ValueError("Could not determine the entailment ID from the model config, please pass it at init.")

    def encode(self, text, labels):
        # 编码函数，将输入的文本和标签进行编码处理
        self._labels = labels
        return self.pre_processor(
            [text] * len(labels),
            [f"This example is {label}" for label in labels],
            return_tensors="pt",
            padding="max_length",
        )

    def decode(self, outputs):
        # 解码函数，根据模型输出的logits确定最可能的标签
        logits = outputs.logits
        label_id = torch.argmax(logits[:, 2]).item()
        return self._labels[label_id]
```