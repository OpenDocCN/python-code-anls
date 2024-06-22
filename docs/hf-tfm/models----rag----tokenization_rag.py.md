# `.\transformers\models\rag\tokenization_rag.py`

```
# 该脚本定义了 RagTokenizer 类，用于 RAG 模型的输入/输出数据的编码和解码。
# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for RAG."""
import os
import warnings
from typing import List, Optional

# 导入所需的模块和类
from ...tokenization_utils_base import BatchEncoding
from ...utils import logging
from .configuration_rag import RagConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# RagTokenizer 类定义
class RagTokenizer:
    # 初始化方法，接收 question_encoder 和 generator 两个 Tokenizer 对象
    def __init__(self, question_encoder, generator):
        self.question_encoder = question_encoder
        self.generator = generator
        self.current_tokenizer = self.question_encoder

    # 将模型和两个 Tokenizer 保存到指定路径
    def save_pretrained(self, save_directory):
        # 检查保存目录是否为文件
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        # 创建保存目录
        os.makedirs(save_directory, exist_ok=True)
        # 定义 question_encoder 和 generator 的保存路径
        question_encoder_path = os.path.join(save_directory, "question_encoder_tokenizer")
        generator_path = os.path.join(save_directory, "generator_tokenizer")
        # 分别保存两个 Tokenizer
        self.question_encoder.save_pretrained(question_encoder_path)
        self.generator.save_pretrained(generator_path)

    # 从预训练的模型加载 RagTokenizer
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 动态导入 AutoTokenizer 类
        from ..auto.tokenization_auto import AutoTokenizer

        # 从传入的参数中获取配置对象
        config = kwargs.pop("config", None)

        # 如果未传入配置对象，则从预训练模型加载配置
        if config is None:
            config = RagConfig.from_pretrained(pretrained_model_name_or_path)

        # 分别从预训练模型加载 question_encoder 和 generator 两个 Tokenizer
        question_encoder = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, config=config.question_encoder, subfolder="question_encoder_tokenizer"
        )
        generator = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, config=config.generator, subfolder="generator_tokenizer"
        )
        # 返回 RagTokenizer 实例
        return cls(question_encoder=question_encoder, generator=generator)

    # 调用当前的 Tokenizer
    def __call__(self, *args, **kwargs):
        return self.current_tokenizer(*args, **kwargs)

    # 调用 generator Tokenizer 的 batch_decode 方法
    def batch_decode(self, *args, **kwargs):
        return self.generator.batch_decode(*args, **kwargs)

    # 调用 generator Tokenizer 的 decode 方法
    def decode(self, *args, **kwargs):
        return self.generator.decode(*args, **kwargs)

    # 切换到输入模式（使用 question_encoder Tokenizer）
    def _switch_to_input_mode(self):
        self.current_tokenizer = self.question_encoder

    # 切换到输出模式（使用 generator Tokenizer）
    def _switch_to_target_mode(self):
        self.current_tokenizer = self.generator
    # 发出警告，提示 `prepare_seq2seq_batch` 方法将在 🤗 Transformers 版本 5 中被移除，建议使用正常的 `__call__` 方法准备输入，并在 `with_target_tokenizer` 上下文管理器下使用分词器准备目标。查看特定分词器的文档以获取更多详情
    warnings.warn(
        "`prepare_seq2seq_batch` is deprecated and will be removed in version 5 of 🤗 Transformers. Use the "
        "regular `__call__` method to prepare your inputs and the tokenizer under the `with_target_tokenizer` "
        "context manager to prepare your targets. See the documentation of your specific tokenizer for more "
        "details",
        FutureWarning,
    )
    # 如果未指定最大长度，则将其设置为当前分词器的模型最大长度
    if max_length is None:
        max_length = self.current_tokenizer.model_max_length
    # 使用当前对象作为分词器，准备模型输入
    model_inputs = self(
        src_texts,  # 源文本列表
        add_special_tokens=True,  # 添加特殊标记
        return_tensors=return_tensors,  # 返回张量类型
        max_length=max_length,  # 最大长度
        padding=padding,  # 填充策略
        truncation=truncation,  # 截断策略
        **kwargs,
    )
    # 如果目标文本列表未提供，则返回模型输入
    if tgt_texts is None:
        return model_inputs
    # 处理目标文本
    # 如果未指定最大目标长度，则将其设置为当前分词器的模型最大长度
    if max_target_length is None:
        max_target_length = self.current_tokenizer.model_max_length
    # 使用当前对象作为分词器，准备标签
    labels = self(
        text_target=tgt_texts,  # 目标文本列表
        add_special_tokens=True,  # 添加特殊标记
        return_tensors=return_tensors,  # 返回张量类型
        padding=padding,  # 填充策略
        max_length=max_target_length,  # 最大长度
        truncation=truncation,  # 截断策略
        **kwargs,
    )
    # 将标签的输入 ID 存储在模型输入中的 "labels" 键下
    model_inputs["labels"] = labels["input_ids"]
    # 返回模型输入
    return model_inputs
```