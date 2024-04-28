# `.\transformers\models\rembert\tokenization_rembert_fast.py`

```py
# coding=utf-8
# 这个文件是用 UTF-8 编码的 Python 文件

# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
# 这个文件的版权归属于 Google AI、Google Brain 和 HuggingFace Inc. 团队

# Licensed under the Apache License, Version 2.0 (the "License");
# 这个文件使用 Apache License, Version 2.0 进行许可

# you may not use this file except in compliance with the License.
# 你在使用这个文件时需要遵守 License 的条款

# You may obtain a copy of the License at
# 你可以在这个地址获取 License 的副本：
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，软件

# distributed under the License is distributed on an "AS IS" BASIS,
# 根据 License 以"原样"分发

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不提供任何明示或暗示的担保或条件

# See the License for the specific language governing permissions and
# 请参阅 License 中关于权限和

# limitations under the License.
# 限制的具体说明。

# Tokenization classes for RemBERT model.
# RemBERT 模型的分词类

import os
from shutil import copyfile
from typing import List, Optional, Tuple

from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging

if is_sentencepiece_available():
    # 如果可以使用 SentencePiece，导入 RemBertTokenizer
    from .tokenization_rembert import RemBertTokenizer
else:
    # 否则设置 RemBertTokenizer 为 None
    RemBertTokenizer = None

logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    # 预训练的 vocab 文件和对应的 URL
    "vocab_file": {
        "google/rembert": "https://huggingface.co/google/rembert/resolve/main/sentencepiece.model",
    },
    "tokenizer_file": {
        "google/rembert": "https://huggingface.co/google/rembert/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    # 预训练模型的 Positional Embeddings 大小
    "google/rembert": 256,
}

SPIECE_UNDERLINE = "▁"


class RemBertTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" RemBert tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models). This
    tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods
    """
    Args:
        vocab_file (`str`):
            SentencePiece file (generally has a *.spm* extension) that contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        remove_space (`bool`, *optional*, defaults to `True`):
            Whether or not to strip the text when tokenizing (removing excess spaces before and after the string).
        keep_accents (`bool`, *optional*, defaults to `False`):
            Whether or not to keep accents when tokenizing.
        bos_token (`str`, *optional*, defaults to `"[CLS]"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"[SEP]"`):
            The end of sequence token. .. note:: When building a sequence using special tokens, this is not the token
            that is used for the end of sequence. The token used is the `sep_token`.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    # 文件名与对应的词汇表文件名的映射
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型词汇表文件的映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型输入的最大长度
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 慢速标记器类
    slow_tokenizer_class = RemBertTokenizer
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        remove_space=True,
        keep_accents=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="<unk>",
        sep_token="[SEP]",
        pad_token="<pad>",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs,
    ):
        # 如果 mask_token 是字符串类型，将其包装成 AddedToken 对象，去掉左侧空格，保留右侧空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        
        # 调用父类的初始化方法，传入各种参数
        super().__init__(
            vocab_file,
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
            **kwargs,
        )
        
        # 设置对象的属性
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查是否存在 vocab_file 文件，返回布尔值
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或一对序列中构建模型输入，用于序列分类任务，通过连接和添加特殊符号。一个 RemBERT 序列的格式如下：

        - 单个序列：`[CLS] X [SEP]`
        - 一对序列：`[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊符号的 ID 列表
            token_ids_1 (`List[int]`, *可选*, 默认为 `None`):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 包含适当特殊符号的 [输入 ID](../glossary#input-ids) 列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*, defaults to `None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Set to True if the token list is already formatted with special tokens for the model

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        # Check if the token list already has special tokens added
        if already_has_special_tokens:
            # Raise an error if a second sequence is provided when special tokens already exist
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            # Return a list with 1 for special tokens and 0 for sequence tokens
            return [1 if x in [self.sep_token_id, self.cls_token_id] else 0 for x in token_ids_0]

        # If a second sequence is provided
        if token_ids_1 is not None:
            # Create a list with special tokens and sequence tokens for both sequences
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # If only one sequence is provided
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. A RemBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```py

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*, defaults to `None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        
        # Define special tokens for separation and classification
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        # If only one sequence is provided
        if token_ids_1 is None:
            # Return a list with 0s for the first portion of the mask
            return len(cls + token_ids_0 + sep) * [0]
        # If two sequences are provided
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    # 保存词汇表到指定目录下
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录不存在，则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        # 设置输出词汇表文件路径，如果有前缀则添加前缀，否则直接使用词汇表文件名
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果输出词汇表文件路径与当前词汇表文件路径不同，则复制词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回输出词汇表文件路径的元组
        return (out_vocab_file,)
```