# `.\data\processors\xnli.py`

```
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" XNLI utils (dataset loading and evaluation)"""

import os

from ...utils import logging
from .utils import DataProcessor, InputExample

# 获取日志记录器实例
logger = logging.get_logger(__name__)

class XnliProcessor(DataProcessor):
    """
    Processor for the XNLI dataset. Adapted from
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207
    """

    def __init__(self, language, train_language=None):
        # 初始化 XNLIProcessor 类的实例
        self.language = language
        self.train_language = train_language

    def get_train_examples(self, data_dir):
        """See base class."""
        # 如果没有指定 train_language，则使用 language
        lg = self.language if self.train_language is None else self.train_language
        # 读取并解析训练数据的每一行，从指定路径读取文件
        lines = self._read_tsv(os.path.join(data_dir, f"XNLI-MT-1.0/multinli/multinli.train.{lg}.tsv"))
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            # 创建全局唯一标识符，形如 "train-{i}"
            guid = f"train-{i}"
            # 第一列是文本 A
            text_a = line[0]
            # 第二列是文本 B
            text_b = line[1]
            # 第三列是标签，如果是 "contradictory" 则映射为 "contradiction"
            label = "contradiction" if line[2] == "contradictory" else line[2]
            # 确保 text_a 是字符串类型
            if not isinstance(text_a, str):
                raise ValueError(f"Training input {text_a} is not a string")
            # 确保 text_b 是字符串类型
            if not isinstance(text_b, str):
                raise ValueError(f"Training input {text_b} is not a string")
            # 确保 label 是字符串类型
            if not isinstance(label, str):
                raise ValueError(f"Training label {label} is not a string")
            # 创建 InputExample 对象并添加到 examples 列表中
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        # 返回构建的训练示例列表
        return examples
    # 从指定路径读取测试数据集的 TSV 文件并返回每行内容组成的列表
    lines = self._read_tsv(os.path.join(data_dir, "XNLI-1.0/xnli.test.tsv"))
    
    # 初始化一个空列表，用于存储处理后的样本数据
    examples = []
    
    # 遍历每行数据，i 是行索引，line 是行内容的列表
    for i, line in enumerate(lines):
        # 跳过第一行标题行
        if i == 0:
            continue
        
        # 获取语言标签，该数据集中位于行的第一个位置
        language = line[0]
        
        # 如果语言标签不等于当前实例的语言，则跳过此条数据
        if language != self.language:
            continue
        
        # 构建一个唯一的全局标识符，格式为 "test-索引"
        guid = f"test-{i}"
        
        # 获取第一个文本句子，位于行的第七个位置
        text_a = line[6]
        
        # 获取第二个文本句子，位于行的第八个位置
        text_b = line[7]
        
        # 获取标签信息，位于行的第二个位置
        label = line[1]
        
        # 如果 text_a 不是字符串类型，则引发数值错误异常
        if not isinstance(text_a, str):
            raise ValueError(f"Training input {text_a} is not a string")
        
        # 如果 text_b 不是字符串类型，则引发数值错误异常
        if not isinstance(text_b, str):
            raise ValueError(f"Training input {text_b} is not a string")
        
        # 如果 label 不是字符串类型，则引发数值错误异常
        if not isinstance(label, str):
            raise ValueError(f"Training label {label} is not a string")
        
        # 创建一个输入样本对象，并将其添加到 examples 列表中
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    
    # 返回处理后的样本列表
    return examples

# 返回一个包含可能的标签的列表，这些标签用于表示不同的语言推理结果
def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]
# 定义一个字典，将任务名称映射到处理器类上
xnli_processors = {
    "xnli": XnliProcessor,
}

# 定义一个字典，将任务名称映射到输出模式上，这里输出模式为分类
xnli_output_modes = {
    "xnli": "classification",
}

# 定义一个字典，将任务名称映射到标签数量上，这里 "xnli" 任务有 3 个标签
xnli_tasks_num_labels = {
    "xnli": 3,
}
```