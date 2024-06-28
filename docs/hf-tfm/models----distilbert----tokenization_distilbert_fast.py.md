# `.\models\distilbert\tokenization_distilbert_fast.py`

```
# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Tokenization classes for DistilBERT."""

import json
from typing import List, Optional, Tuple

from tokenizers import normalizers

# 导入必要的日志记录模块
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
# 导入DistilBERT的标记器
from .tokenization_distilbert import DistilBertTokenizer

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 定义词汇和标记器文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型的词汇和标记器文件的URL映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "distilbert-base-uncased": "https://huggingface.co/distilbert-base-uncased/resolve/main/vocab.txt",
        "distilbert-base-uncased-distilled-squad": (
            "https://huggingface.co/distilbert-base-uncased-distilled-squad/resolve/main/vocab.txt"
        ),
        "distilbert-base-cased": "https://huggingface.co/distilbert-base-cased/resolve/main/vocab.txt",
        "distilbert-base-cased-distilled-squad": (
            "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/vocab.txt"
        ),
        "distilbert-base-german-cased": "https://huggingface.co/distilbert-base-german-cased/resolve/main/vocab.txt",
        "distilbert-base-multilingual-cased": (
            "https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/vocab.txt"
        ),
    },
    "tokenizer_file": {
        "distilbert-base-uncased": "https://huggingface.co/distilbert-base-uncased/resolve/main/tokenizer.json",
        "distilbert-base-uncased-distilled-squad": (
            "https://huggingface.co/distilbert-base-uncased-distilled-squad/resolve/main/tokenizer.json"
        ),
        "distilbert-base-cased": "https://huggingface.co/distilbert-base-cased/resolve/main/tokenizer.json",
        "distilbert-base-cased-distilled-squad": (
            "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/tokenizer.json"
        ),
        "distilbert-base-german-cased": (
            "https://huggingface.co/distilbert-base-german-cased/resolve/main/tokenizer.json"
        ),
        "distilbert-base-multilingual-cased": (
            "https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/tokenizer.json"
        ),
    },
}

# 定义预训练模型的位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "distilbert-base-uncased": 512,
    "distilbert-base-uncased-distilled-squad": 512,
    "distilbert-base-cased": 512,
    "distilbert-base-cased-distilled-squad": 512,
}
    # 定义模型名称为键，对应的最大输入长度为值的字典条目
    "distilbert-base-german-cased": 512,
    "distilbert-base-multilingual-cased": 512,
}

# 预训练模型的初始化配置，包含了不同模型的配置信息
PRETRAINED_INIT_CONFIGURATION = {
    "distilbert-base-uncased": {"do_lower_case": True},  # 使用小写字符
    "distilbert-base-uncased-distilled-squad": {"do_lower_case": True},  # 使用小写字符
    "distilbert-base-cased": {"do_lower_case": False},  # 区分大小写
    "distilbert-base-cased-distilled-squad": {"do_lower_case": False},  # 区分大小写
    "distilbert-base-german-cased": {"do_lower_case": False},  # 区分大小写，适用于德语
    "distilbert-base-multilingual-cased": {"do_lower_case": False},  # 区分大小写，适用于多语言
}


class DistilBertTokenizerFast(PreTrainedTokenizerFast):
    r"""
    构建一个“快速”的 DistilBERT 分词器（基于 HuggingFace 的 *tokenizers* 库）。基于 WordPiece。

    此分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。
    """
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
            value for `lowercase` (as in the original BERT).
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            The prefix for subwords.
    """
    # 定义一些常量和映射
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 指定模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 指定慢速分词器的类，这里使用的是 DistilBertTokenizer
    slow_tokenizer_class = DistilBertTokenizer

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
        # 调用父类的初始化方法，设置词汇文件、分词器文件、大小写敏感、未知标记、分隔标记、填充标记、类别标记、掩码标记、处理中文字符等参数
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

        # 获取当前后端分词器的规范化器状态并转换为字典
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查规范化器状态中的属性是否与当前初始化参数一致，若不一致则更新
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            # 获取当前规范化器的类，并更新相关属性
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        # 设置对象的大小写敏感属性
        self.do_lower_case = do_lower_case

    # Copied from transformers.models.bert.tokenization_bert_fast.BertTokenizerFast.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

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
        # 构建带有特殊标记的输入序列，根据是否提供第二个序列决定是否添加第二个分隔符和第二个序列的 token IDs
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    # Copied from transformers.models.bert.tokenization_bert_fast.BertTokenizerFast.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        """
        Create token type IDs tensor from given sequences.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs corresponding to the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional list of IDs corresponding to the second sequence for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs.
        """
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
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
        # Define special tokens for separation and classification
        sep = [self.sep_token_id]  # List containing the separator token ID
        cls = [self.cls_token_id]  # List containing the classification token ID
        
        # If only one sequence is provided (token_ids_1 is None), return a mask with 0s
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]  # Return a list of zeros representing token type IDs
        
        # If two sequences are provided, concatenate their lengths and return a mask with 0s for the first sequence and 1s for the second sequence
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # Copied from transformers.models.bert.tokenization_bert_fast.BertTokenizerFast.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary files to the specified directory.

        Args:
            save_directory (str):
                Directory path where the vocabulary files will be saved.
            filename_prefix (Optional[str]):
                Optional prefix for the vocabulary filenames.

        Returns:
            Tuple[str]: Tuple containing the paths of the saved vocabulary files.
        """
        # Call the internal tokenizer's model save method to save vocabulary files
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # Return the paths of the saved files as a tuple
        return tuple(files)
```