# `.\models\rag\tokenization_rag.py`

```
# coding=utf-8
# 声明文件编码格式为 UTF-8

# 版权声明和许可证信息

# 导入必要的模块和类
import os
import warnings
from typing import List, Optional

# 导入日志记录工具
from ...tokenization_utils_base import BatchEncoding
from ...utils import logging
from .configuration_rag import RagConfig

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


class RagTokenizer:
    def __init__(self, question_encoder, generator):
        # 初始化 RAG Tokenizer 类，接受问题编码器和生成器作为参数
        self.question_encoder = question_encoder
        self.generator = generator
        self.current_tokenizer = self.question_encoder

    def save_pretrained(self, save_directory):
        # 将当前 tokenizer 实例保存到指定目录下
        if os.path.isfile(save_directory):
            # 如果保存路径是一个文件，抛出错误
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        # 创建目录，如果目录已存在则不报错
        os.makedirs(save_directory, exist_ok=True)
        # 分别保存问题编码器和生成器的 tokenizer 到指定目录下的不同子目录
        question_encoder_path = os.path.join(save_directory, "question_encoder_tokenizer")
        generator_path = os.path.join(save_directory, "generator_tokenizer")
        self.question_encoder.save_pretrained(question_encoder_path)
        self.generator.save_pretrained(generator_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 从预训练模型或路径加载 RAG Tokenizer 实例
        # 动态导入 AutoTokenizer 类
        from ..auto.tokenization_auto import AutoTokenizer

        # 获取配置信息，如果未提供则从预训练模型加载
        config = kwargs.pop("config", None)
        if config is None:
            config = RagConfig.from_pretrained(pretrained_model_name_or_path)

        # 根据配置加载问题编码器和生成器的 tokenizer
        question_encoder = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, config=config.question_encoder, subfolder="question_encoder_tokenizer"
        )
        generator = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, config=config.generator, subfolder="generator_tokenizer"
        )
        return cls(question_encoder=question_encoder, generator=generator)

    def __call__(self, *args, **kwargs):
        # 实现 __call__ 方法，允许实例像函数一样被调用
        return self.current_tokenizer(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        # 调用生成器的批量解码方法
        return self.generator.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        # 调用生成器的解码方法
        return self.generator.decode(*args, **kwargs)

    def _switch_to_input_mode(self):
        # 切换当前 tokenizer 到问题编码器模式
        self.current_tokenizer = self.question_encoder

    def _switch_to_target_mode(self):
        # 切换当前 tokenizer 到生成器模式
        self.current_tokenizer = self.generator
    # 警告：`prepare_seq2seq_batch`已被弃用，并将在🤗 Transformers版本5中移除。请使用常规的`__call__`方法准备输入，并在`with_target_tokenizer`上下文管理器下使用分词器准备目标。查看特定分词器的文档获取更多详情
    warnings.warn(
        "`prepare_seq2seq_batch` is deprecated and will be removed in version 5 of 🤗 Transformers. Use the "
        "regular `__call__` method to prepare your inputs and the tokenizer under the `with_target_tokenizer` "
        "context manager to prepare your targets. See the documentation of your specific tokenizer for more "
        "details",
        FutureWarning,
    )
    
    # 如果未提供最大长度参数，则使用当前分词器的模型最大长度
    if max_length is None:
        max_length = self.current_tokenizer.model_max_length
    
    # 使用模型的__call__方法准备输入，包括源文本、添加特殊标记、返回的张量类型、最大长度、填充方式和截断标志
    model_inputs = self(
        src_texts,
        add_special_tokens=True,
        return_tensors=return_tensors,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        **kwargs,
    )
    
    # 如果未提供目标文本，则直接返回模型输入
    if tgt_texts is None:
        return model_inputs
    
    # 处理目标文本
    # 如果未提供最大目标长度参数，则使用当前分词器的模型最大长度
    if max_target_length is None:
        max_target_length = self.current_tokenizer.model_max_length
    
    # 使用模型的__call__方法准备目标标签，包括目标文本、添加特殊标记、返回的张量类型、填充方式、最大长度和截断标志
    labels = self(
        text_target=tgt_texts,
        add_special_tokens=True,
        return_tensors=return_tensors,
        padding=padding,
        max_length=max_target_length,
        truncation=truncation,
        **kwargs,
    )
    
    # 将准备好的目标标签的输入ID存储在模型输入字典中的"labels"键下
    model_inputs["labels"] = labels["input_ids"]
    
    # 返回最终的模型输入字典，包括源文本、可能的目标文本标签
    return model_inputs
```