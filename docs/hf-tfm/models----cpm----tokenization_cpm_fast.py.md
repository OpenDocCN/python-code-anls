# `.\models\cpm\tokenization_cpm_fast.py`

```py
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes."""
import os
from shutil import copyfile
from typing import List, Optional, Tuple

from ...tokenization_utils_fast import AddedToken, PreTrainedTokenizerFast
from ...utils import logging

# 获取全局的日志记录器
logger = logging.get_logger(__name__)

# 定义预设的词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型与其对应的词汇文件和分词器文件的映射关系
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "TsinghuaAI/CPM-Generate": "https://huggingface.co/TsinghuaAI/CPM-Generate/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "TsinghuaAI/CPM-Generate": "https://huggingface.co/TsinghuaAI/CPM-Generate/resolve/main/tokenizer.json",
    },
}

# 定义 CpmTokenizerFast 类，继承自 PreTrainedTokenizerFast
class CpmTokenizerFast(PreTrainedTokenizerFast):
    """Runs pre-tokenization with Jieba segmentation tool. It is used in CPM models."""

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        additional_special_tokens=["<eop>", "<eod>"],
        **kwargs,
    ):
        # 继承父类的初始化方法，设定各种标记和参数
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查是否可以保存慢速分词器，基于词汇文件的存在性
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    # 从 transformers.models.xlnet.tokenization_xlnet_fast.XLNetTokenizerFast 复制而来
    # 用于构建带有特殊标记的输入序列
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 定义一个方法用于生成用于序列分类任务的模型输入，通过连接和添加特殊标记来构建。XLNet 序列的格式如下：
    #
    # - 单个序列：`X <sep> <cls>`
    # - 序列对：`A <sep> B <sep> <cls>`
    #
    # Args:
    #     token_ids_0 (`List[int]`):
    #         要添加特殊标记的 ID 列表。
    #     token_ids_1 (`List[int]`, *optional*):
    #         第二个序列的可选 ID 列表，用于序列对。
    #
    # Returns:
    #     `List[int]`: 包含适当特殊标记的输入 ID 列表。
    def create_inputs_for_sequence_classification(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]  # 分隔符的 ID 列表
        cls = [self.cls_token_id]  # 类别标记的 ID 列表

        if token_ids_1 is None:
            return token_ids_0 + sep + cls  # 单个序列的情况
        return token_ids_0 + sep + token_ids_1 + sep + cls  # 序列对的情况

    # 从两个序列创建用于序列对分类任务的 token 类型 ID 列表。XLNet 的序列对 mask 格式如下：
    #
    # ```
    # 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
    # | 第一个序列        | 第二个序列     |
    # ```
    #
    # 如果 `token_ids_1` 是 `None`，则该方法仅返回 mask 的第一部分（全为 0）。
    #
    # Args:
    #     token_ids_0 (`List[int]`):
    #         第一个序列的 ID 列表。
    #     token_ids_1 (`List[int]`, *optional*):
    #         第二个序列的可选 ID 列表，用于序列对。
    #
    # Returns:
    #     `List[int]`: 根据给定序列(s)生成的 token 类型 ID 列表。
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]  # 分隔符的 ID 列表
        cls_segment_id = [2]  # 类别片段 ID 列表

        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0] + cls_segment_id  # 只有第一个序列的情况
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1] + cls_segment_id  # 序列对的情况
    # 将词汇表保存到指定目录下的文件中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果无法保存慢速分词器的词汇表，则引发值错误异常
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存目录不存在，则记录错误日志并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 构建输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与输出路径不一致，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回保存的词汇表文件路径
        return (out_vocab_file,)

    # 对批量文本或文本对进行编码处理
    def _batch_encode_plus(self, batch_text_or_text_pairs, *args, **kwargs):
        # 使用结巴分词器处理每个文本，去除空格和特殊字符后进行拼接
        batch_text_or_text_pairs = [
            " ".join([x.translate(self.translator) for x in self.jieba.cut(text, cut_all=False)])
            for text in batch_text_or_text_pairs
        ]
        # 调用父类方法对处理后的文本进行编码处理
        return super()._batch_encode_plus(batch_text_or_text_pairs, *args, **kwargs)

    # 解码处理方法
    def _decode(self, *args, **kwargs):
        # 调用父类方法进行解码处理
        text = super()._decode(*args, **kwargs)
        # 替换文本中的特殊空格和分隔符
        text = text.replace(" ", "").replace("\u2582", " ").replace("\u2583", "\n")
        # 返回处理后的文本
        return text
```