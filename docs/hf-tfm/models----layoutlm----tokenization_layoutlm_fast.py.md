# `.\models\layoutlm\tokenization_layoutlm_fast.py`

```py
# coding=utf-8
# 设置脚本编码为 UTF-8

# Copyright 2018 The Microsoft Research Asia LayoutLM Team Authors.
# 版权声明，指出此代码的版权信息

# Licensed under the Apache License, Version 2.0 (the "License");
# 使用 Apache License, Version 2.0 授权许可

# you may not use this file except in compliance with the License.
# 除非遵循本许可证，否则不能使用本文件

# You may obtain a copy of the License at
# 可以在以下网址获取许可证副本

#     http://www.apache.org/licenses/LICENSE-2.0
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# 根据适用法律规定或书面同意的情况下，软件

# distributed under the License is distributed on an "AS IS" BASIS,
# 分发时遵循"原样"分发

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有明示或暗示的任何保证或条件

# See the License for the specific language governing permissions and
# 详见许可证，获取特定语言的权限以及

# limitations under the License.
# 许可下的限制

""" Tokenization class for model LayoutLM."""

# LayoutLM 模型的分词器类

import json
# 导入 json 模块，用于处理 JSON 格式数据
from typing import List, Optional, Tuple
# 导入 typing 模块，用于类型提示

from tokenizers import normalizers
# 从 tokenizers 库中导入 normalizers 模块

from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 从 ...tokenization_utils_fast 中导入 PreTrainedTokenizerFast 类
from ...utils import logging
# 从 ...utils 中导入 logging 模块
from .tokenization_layoutlm import LayoutLMTokenizer
# 从当前目录下的 tokenization_layoutlm 模块中导入 LayoutLMTokenizer 类


logger = logging.get_logger(__name__)
# 获取当前脚本的日志记录器对象

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}
# 定义词汇表文件名和分词器文件名的映射关系

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/layoutlm-base-uncased": (
            "https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main/vocab.txt"
        ),
        "microsoft/layoutlm-large-uncased": (
            "https://huggingface.co/microsoft/layoutlm-large-uncased/resolve/main/vocab.txt"
        ),
    },
    "tokenizer_file": {
        "microsoft/layoutlm-base-uncased": (
            "https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main/tokenizer.json"
        ),
        "microsoft/layoutlm-large-uncased": (
            "https://huggingface.co/microsoft/layoutlm-large-uncased/resolve/main/tokenizer.json"
        ),
    },
}
# 定义预训练模型和对应的词汇表文件及分词器文件的下载地址映射关系

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/layoutlm-base-uncased": 512,
    "microsoft/layoutlm-large-uncased": 512,
}
# 定义不同预训练模型的位置嵌入大小

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/layoutlm-base-uncased": {"do_lower_case": True},
    "microsoft/layoutlm-large-uncased": {"do_lower_case": True},
}
# 定义不同预训练模型的初始化配置，如是否小写化等设置


# Copied from transformers.models.bert.tokenization_bert_fast.BertTokenizerFast with Bert->LayoutLM,BERT->LayoutLM
# 从 transformers.models.bert.tokenization_bert_fast.BertTokenizerFast 复制并修改为 LayoutLMTokenizerFast

class LayoutLMTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" LayoutLM tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """
    # 构造一个基于 WordPiece 并由 HuggingFace 的 tokenizers 库支持的 "快速" LayoutLM 分词器

    def __init__(
        self,
        vocab_file: str,
        tokenizer_file: str,
        **kwargs
    ):
        # 初始化方法，接受词汇表文件和分词器文件的路径参数及其他关键字参数

        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            **kwargs
        )
        # 调用父类的初始化方法，传递词汇表文件和分词器文件路径及其他参数
    # 定义一些常量，这些常量用于初始化 LayoutLMTokenizer 类的实例
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = LayoutLMTokenizer
    
    # LayoutLMTokenizer 类的构造函数，初始化实例时会调用
    def __init__(
        self,
        vocab_file=None,  # 词汇表文件的路径
        tokenizer_file=None,  # 分词器文件的路径（可选）
        do_lower_case=True,  # 是否将输入文本转换为小写（默认为 True）
        unk_token="[UNK]",  # 未知词汇的特殊标记
        sep_token="[SEP]",  # 分隔符标记，用于多个序列的组合
        pad_token="[PAD]",  # 填充标记，用于序列的长度不同时进行填充
        cls_token="[CLS]",  # 分类器标记，用于序列分类任务中的第一个标记
        mask_token="[MASK]",  # 掩码标记，用于掩码语言建模任务中的掩码预测
        tokenize_chinese_chars=True,  # 是否分词中文字符（默认为 True）
        strip_accents=None,  # 是否去除所有的重音符号（默认根据 lowercase 的值确定）
        **kwargs,  # 其他可选参数
    ):
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

# 调用父类的构造方法，初始化一个新的对象，传入必要的参数和可选参数


        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())

# 从后端的标记器（tokenizer）获取其正规化器（normalizer）的状态，并将其解析为 JSON 格式


        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):

# 检查解析出的正规化器状态是否与当前对象的设定不一致，如果不一致则执行下面的操作


            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

# 根据解析出的正规化器类型，更新正规化器的设置以与当前对象的设定一致


        self.do_lower_case = do_lower_case

# 更新对象的小写设置为传入的参数值


    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A LayoutLM sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

# 构建模型输入，根据输入的序列或序列对来生成用于序列分类任务的特殊标记，包括连接和添加特殊标记


        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

# 如果有第二个序列，将其添加到输出列表中，并添加分隔符标记后返回输出列表


    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None

# 创建用于 LayoutLM 序列的 token type IDs，根据输入的序列或序列对生成相应的类型 ID 列表```
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

# 调用父类的构造方法，初始化一个新的对象，传入必要的参数和可选参数


        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())

# 从后端的标记器（tokenizer）获取其正规化器（normalizer）的状态，并将其解析为 JSON 格式


        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):

# 检查解析出的正规化器状态是否与当前对象的设定不一致，如果不一致则执行下面的操作


            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

# 根据解析出的正规化器类型，更新正规化器的设置以与当前对象的设定一致


        self.do_lower_case = do_lower_case

# 更新对象的小写设置为传入的参数值


    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A LayoutLM sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

# 构建模型输入，根据输入的序列或序列对来生成用于序列分类任务的特殊标记，包括连接和添加特殊标记


        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

# 如果有第二个序列，将其添加到输出列表中，并添加分隔符标记后返回输出列表


    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None

# 创建用于 LayoutLM 序列的 token type IDs，根据输入的序列或序列对生成相应的类型 ID 列表
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A LayoutLM sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs according to the given sequence(s).
        """
        # Define separators for the beginning and end of the first sequence
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        # If only one sequence is provided, return a mask with zeros for the first sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # If two sequences are provided, concatenate their token IDs and return a mask with zeros for the first sequence
        # and ones for the second sequence
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary of the tokenizer model to the specified directory.

        Args:
            save_directory (str):
                Directory path where the vocabulary will be saved.
            filename_prefix (str, *optional*):
                Optional prefix for the saved files.

        Returns:
            Tuple[str]: Tuple containing the filenames saved.
        """
        # Save the vocabulary files using the tokenizer model's save method
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # Return the filenames as a tuple
        return tuple(files)
```