# `.\transformers\tools\text_classification.py`

```
#!/usr/bin/env python
# coding=utf-8

# 包含版权信息
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

# 导入 torch 模块
import torch

# 导入 AutoModelForSequenceClassification 和 AutoTokenizer 类
from ..models.auto import AutoModelForSequenceClassification, AutoTokenizer
# 从 base 模块导入 PipelineTool 类
from .base import PipelineTool


# 定义 TextClassificationTool 类，继承于 PipelineTool 类
class TextClassificationTool(PipelineTool):
    """
    Example:

    ```py
    from transformers.tools import TextClassificationTool

    classifier = TextClassificationTool()
    classifier("This is a super nice API!", labels=["positive", "negative"])
    ```
    """
    # 定义类属性
    default_checkpoint = "facebook/bart-large-mnli"
    description = (
        "This is a tool that classifies an English text using provided labels. It takes two inputs: `text`, which "
        "should be the text to classify, and `labels`, which should be the list of labels to use for classification. "
        "It returns the most likely label in the list of provided `labels` for the input text."
    )
    name = "text_classifier"
    pre_processor_class = AutoTokenizer
    model_class = AutoModelForSequenceClassification

    # 定义输入和输出
    inputs = ["text", ["text"]]
    outputs = ["text"]

    # 初始化方法
    def setup(self):
        super().setup()
        # 获取模型配置
        config = self.model.config
        self.entailment_id = -1
        # 遍历标签与索引的字典，寻找以 "entail" 开头的标签
        for idx, label in config.id2label.items():
            if label.lower().startswith("entail"):
                self.entailment_id = int(idx)
        # 如果未找到符合条件的标签，则抛出 ValueError
        if self.entailment_id == -1:
            raise ValueError("Could not determine the entailment ID from the model config, please pass it at init.")

    # 编码方法
    def encode(self, text, labels):
        self._labels = labels
        # 使用预处理器对文本和标签进行编码
        return self.pre_processor(
            [text] * len(labels),
            [f"This example is {label}" for label in labels],
            return_tensors="pt",
            padding="max_length",
        )

    # 解码方法
    def decode(self, outputs):
        # 获取输出的预测概率
        logits = outputs.logits
        # 获取预测标签的索引
        label_id = torch.argmax(logits[:, 2]).item()
        # 返回对应的标签
        return self._labels[label_id]
```