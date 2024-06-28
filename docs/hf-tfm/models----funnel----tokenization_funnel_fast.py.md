# `.\models\funnel\tokenization_funnel_fast.py`

```
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
""" Tokenization class for Funnel Transformer."""

import json
from typing import List, Optional, Tuple

from tokenizers import normalizers  # 导入tokenizers模块中的normalizers

from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入预训练分词器类
from ...utils import logging  # 导入日志工具
from .tokenization_funnel import FunnelTokenizer  # 导入FunnelTokenizer类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 定义词汇文件和分词器文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 定义可用的模型名称列表
_model_names = [
    "small",
    "small-base",
    "medium",
    "medium-base",
    "intermediate",
    "intermediate-base",
    "large",
    "large-base",
    "xlarge",
    "xlarge-base",
]

# 定义预训练模型对应的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "funnel-transformer/small": "https://huggingface.co/funnel-transformer/small/resolve/main/vocab.txt",
        "funnel-transformer/small-base": "https://huggingface.co/funnel-transformer/small-base/resolve/main/vocab.txt",
        "funnel-transformer/medium": "https://huggingface.co/funnel-transformer/medium/resolve/main/vocab.txt",
        "funnel-transformer/medium-base": (
            "https://huggingface.co/funnel-transformer/medium-base/resolve/main/vocab.txt"
        ),
        "funnel-transformer/intermediate": (
            "https://huggingface.co/funnel-transformer/intermediate/resolve/main/vocab.txt"
        ),
        "funnel-transformer/intermediate-base": (
            "https://huggingface.co/funnel-transformer/intermediate-base/resolve/main/vocab.txt"
        ),
        "funnel-transformer/large": "https://huggingface.co/funnel-transformer/large/resolve/main/vocab.txt",
        "funnel-transformer/large-base": "https://huggingface.co/funnel-transformer/large-base/resolve/main/vocab.txt",
        "funnel-transformer/xlarge": "https://huggingface.co/funnel-transformer/xlarge/resolve/main/vocab.txt",
        "funnel-transformer/xlarge-base": (
            "https://huggingface.co/funnel-transformer/xlarge-base/resolve/main/vocab.txt"
        ),
    },
    # tokenizer_file 字典，包含了多个键值对，每个键是模型名称，对应的值是其对应的 tokenizer.json 文件的下载链接
    "tokenizer_file": {
        # 模型 "funnel-transformer/small" 的 tokenizer.json 下载链接
        "funnel-transformer/small": "https://huggingface.co/funnel-transformer/small/resolve/main/tokenizer.json",
        # 模型 "funnel-transformer/small-base" 的 tokenizer.json 下载链接
        "funnel-transformer/small-base": (
            "https://huggingface.co/funnel-transformer/small-base/resolve/main/tokenizer.json"
        ),
        # 模型 "funnel-transformer/medium" 的 tokenizer.json 下载链接
        "funnel-transformer/medium": "https://huggingface.co/funnel-transformer/medium/resolve/main/tokenizer.json",
        # 模型 "funnel-transformer/medium-base" 的 tokenizer.json 下载链接
        "funnel-transformer/medium-base": (
            "https://huggingface.co/funnel-transformer/medium-base/resolve/main/tokenizer.json"
        ),
        # 模型 "funnel-transformer/intermediate" 的 tokenizer.json 下载链接
        "funnel-transformer/intermediate": (
            "https://huggingface.co/funnel-transformer/intermediate/resolve/main/tokenizer.json"
        ),
        # 模型 "funnel-transformer/intermediate-base" 的 tokenizer.json 下载链接
        "funnel-transformer/intermediate-base": (
            "https://huggingface.co/funnel-transformer/intermediate-base/resolve/main/tokenizer.json"
        ),
        # 模型 "funnel-transformer/large" 的 tokenizer.json 下载链接
        "funnel-transformer/large": "https://huggingface.co/funnel-transformer/large/resolve/main/tokenizer.json",
        # 模型 "funnel-transformer/large-base" 的 tokenizer.json 下载链接
        "funnel-transformer/large-base": (
            "https://huggingface.co/funnel-transformer/large-base/resolve/main/tokenizer.json"
        ),
        # 模型 "funnel-transformer/xlarge" 的 tokenizer.json 下载链接
        "funnel-transformer/xlarge": "https://huggingface.co/funnel-transformer/xlarge/resolve/main/tokenizer.json",
        # 模型 "funnel-transformer/xlarge-base" 的 tokenizer.json 下载链接
        "funnel-transformer/xlarge-base": (
            "https://huggingface.co/funnel-transformer/xlarge-base/resolve/main/tokenizer.json"
        ),
    },
# 定义一个字典，包含预训练位置嵌入的大小，其中键为形如"funnel-transformer/{name}"的字符串，值为固定的512
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {f"funnel-transformer/{name}": 512 for name in _model_names}

# 定义一个字典，包含预训练初始化配置信息，其中键为形如"funnel-transformer/{name}"的字符串，每个值是一个包含"do_lower_case"键的字典，其值为True
PRETRAINED_INIT_CONFIGURATION = {f"funnel-transformer/{name}": {"do_lower_case": True} for name in _model_names}


class FunnelTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" Funnel Transformer tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    
    """
    # 获取给定的词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇文件映射表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型的初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 慢速分词器类，用于创建实例
    slow_tokenizer_class = FunnelTokenizer
    # 预训练模型输入的最大长度
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 分类器令牌类型的ID，设置为2
    cls_token_type_id: int = 2
    # 使用给定的参数初始化对象，继承父类的初始化方法
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        bos_token="<s>",
        eos_token="</s>",
        clean_text=True,
        tokenize_chinese_chars=True,
        strip_accents=None,
        wordpieces_prefix="##",
        **kwargs,
    ):
        # 调用父类的初始化方法，传入参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            clean_text=clean_text,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            wordpieces_prefix=wordpieces_prefix,
            **kwargs,
        )

        # 获取当前的标准化器状态，并将其转换为 JSON 格式
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查标准化器状态是否与当前实例化时的设置不一致，如果不一致则重新设置标准化器
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            # 获取当前标准化器的类，并更新状态
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            # 使用更新后的状态重新设置标准化器
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        # 将初始化时的小写设置保存到对象的属性中
        self.do_lower_case = do_lower_case

    # 从token_ids_0和token_ids_1（可选）构建带有特殊标记的模型输入，用于序列分类任务
    # 该方法来自于transformers.models.funnel.tokenization_funnel_fast.FunnelTokenizerFast.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        通过连接和添加特殊标记，从序列或序列对构建用于序列分类任务的模型输入。Funnel 序列的格式如下：

        - 单个序列： `[CLS] X [SEP]`
        - 序列对： `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 包含适当特殊标记的输入 ID 列表。
        """
        # 初始化输出列表，添加 [CLS] 标记和 token_ids_0
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # 如果有 token_ids_1，则添加 [SEP] 和 token_ids_1
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output
    # 根据两个序列的 token IDs 创建用于序列对分类任务的 token 类型 ID 列表。Funnel Transformer 序列对 mask 的格式如下：
    # ```
    # 2 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
    # | 第一个序列       | 第二个序列       |
    # ```
    # 如果 `token_ids_1` 是 `None`，则方法只返回 mask 的第一个部分（全为 0）。

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A Funnel
        Transformer sequence pair mask has the following format:

        ```
        2 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]  # 分隔符的 token ID 列表
        cls = [self.cls_token_id]  # 类别开始的 token ID 列表
        if token_ids_1 is None:
            # 如果没有第二个序列，返回只包含第一个序列 token 类型 ID 的列表
            return len(cls) * [self.cls_token_type_id] + len(token_ids_0 + sep) * [0]
        # 否则，返回同时包含第一个和第二个序列 token 类型 ID 的列表
        return len(cls) * [self.cls_token_type_id] + len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # 从 transformers.models.bert.tokenization_bert_fast.BertTokenizerFast.save_vocabulary 复制而来
    # 保存词汇表到指定的目录，返回保存的文件名组成的元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```