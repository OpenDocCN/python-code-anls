# `.\models\mobilebert\tokenization_mobilebert_fast.py`

```
# coding=utf-8
# 设置文件编码为UTF-8

# Copyright 2020 The HuggingFace Team. All rights reserved.
# 版权声明

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证要求，否则不得使用本文件
# 可以在以下网址获取许可证副本：
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，不提供任何明示或暗示的担保或条件
# 有关详细信息，请参阅许可证

"""Tokenization classes for MobileBERT."""
# MobileBERT 的分词类

import json
from typing import List, Optional, Tuple

from tokenizers import normalizers

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_mobilebert import MobileBertTokenizer

# 导入必要的模块和类

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 定义词汇文件和分词器文件的名称映射字典

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"mobilebert-uncased": "https://huggingface.co/google/mobilebert-uncased/resolve/main/vocab.txt"},
    "tokenizer_file": {
        "mobilebert-uncased": "https://huggingface.co/google/mobilebert-uncased/resolve/main/tokenizer.json"
    },
}

# 预训练模型的词汇文件和分词器文件映射字典

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"mobilebert-uncased": 512}

# 预训练模型的位置嵌入大小字典，此处是 MobileBERT-uncased 的大小为 512

PRETRAINED_INIT_CONFIGURATION = {}

# 预训练模型的初始化配置为空字典


# Copied from transformers.models.bert.tokenization_bert_fast.BertTokenizerFast with BERT->MobileBERT,Bert->MobileBert
# 从 transformers.models.bert.tokenization_bert_fast.BertTokenizerFast 复制而来，将 BERT 替换为 MobileBERT，Bert 替换为 MobileBert
class MobileBertTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" MobileBERT tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """

# 构造一个“快速” MobileBERT 分词器，基于 HuggingFace 的 tokenizers 库，基于 WordPiece
# 此分词器继承自 PreTrainedTokenizerFast，包含大多数主要方法，用户可以参考该超类获取更多方法信息
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        clean_text (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the text before tokenization by removing any control characters and replacing all
            whitespaces by the classic one.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this
            issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original MobileBERT).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
    """
    # These constants define the file names expected for different vocabularies
    vocab_files_names = VOCAB_FILES_NAMES
    # This maps the expected pretrained vocabulary files for different models
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # This specifies the initial configuration for pretrained models
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # This maps maximum input sizes for pretrained models that use positional embeddings
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # This defines the class of the tokenizer which will be used, MobileBertTokenizer in this case

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        # 调用父类的构造函数，初始化模型的词汇文件、分词器文件等参数
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

        # 从后端分词器获取当前的正常化状态
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查正常化器状态是否与初始化时的参数相匹配，若不匹配则更新
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            # 获取正常化器的类名，并根据当前设置更新状态
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            # 更新后端分词器的正常化器
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        # 设置当前实例的小写参数
        self.do_lower_case = do_lower_case

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A MobileBERT sequence has the following format:

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
        # 构建模型输入，根据输入的token_ids_0和token_ids_1连接和添加特殊标记
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # 如果有第二个序列token_ids_1，则将其加入到输出中
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        # 返回包含特殊标记的输入列表
        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def create_mobilebert_sequence_classification_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A MobileBERT sequence
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
        # Define separator and classifier tokens
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        # If token_ids_1 is None, return mask for single sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # Return mask for sequence pair
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the tokenizer model's vocabulary to a specified directory.

        Args:
            save_directory (str):
                Directory path where the vocabulary files will be saved.
            filename_prefix (Optional[str]):
                Optional prefix for the saved files.

        Returns:
            Tuple[str]: Tuple containing the filenames where the vocabulary is saved.
        """
        # Save the tokenizer model's vocabulary to the specified directory
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```