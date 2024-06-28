# `.\models\mbart\tokenization_mbart.py`

```py
# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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

import os  # 导入操作系统功能模块
from shutil import copyfile  # 导入文件复制函数copyfile
from typing import Any, Dict, List, Optional, Tuple  # 导入类型注解相关模块

import sentencepiece as spm  # 导入sentencepiece模块，用于分词

from ...tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer  # 导入分词工具相关模块
from ...utils import logging  # 导入日志记录模块

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义特殊标记字符
SPIECE_UNDERLINE = "▁"

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/mbart-large-en-ro": (
            "https://huggingface.co/facebook/mbart-large-en-ro/resolve/main/sentencepiece.bpe.model"
        ),
        "facebook/mbart-large-cc25": (
            "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/sentencepiece.bpe.model"
        ),
    }
}

# 预训练模型的位置嵌入尺寸映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/mbart-large-en-ro": 1024,
    "facebook/mbart-large-cc25": 1024,
}

# Fairseq语言代码列表
FAIRSEQ_LANGUAGE_CODES = [
    "ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN",
    "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO",
    "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"
]  # fmt: skip

class MBartTokenizer(PreTrainedTokenizer):
    """
    Construct an MBART tokenizer.

    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    The tokenization method is `<tokens> <eos> <language code>` for source language documents, and `<language code>
    <tokens> <eos>` for target language documents.

    Examples:

    ```
    >>> from transformers import MBartTokenizer

    >>> tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX", tgt_lang="ro_RO")
    >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
    >>> expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"
    >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_romanian, return_tensors="pt")
    ```
    """

    vocab_files_names = VOCAB_FILES_NAMES  # 设置词汇文件的名称
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 设置最大模型输入尺寸
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 设置预训练模型的词汇文件映射
    model_input_names = ["input_ids", "attention_mask"]  # 定义模型输入名称列表

    prefix_tokens: List[int] = []  # 前缀标记列表初始化为空
    suffix_tokens: List[int] = []  # 后缀标记列表初始化为空
    # 初始化方法，用于创建一个新的对象实例
    def __init__(
        self,
        vocab_file,                      # 词汇表文件的路径，用于加载词汇表
        bos_token="<s>",                 # 开始序列的特殊符号，默认为"<s>"
        eos_token="</s>",                # 结束序列的特殊符号，默认为"</s>"
        sep_token="</s>",                # 分隔符的特殊符号，默认为"</s>"
        cls_token="<s>",                 # 类别序列的特殊符号，默认为"<s>"
        unk_token="<unk>",               # 未知符号的特殊符号，默认为"<unk>"
        pad_token="<pad>",               # 填充符号的特殊符号，默认为"<pad>"
        mask_token="<mask>",             # 掩码符号的特殊符号，默认为"<mask>"
        tokenizer_file=None,             # 分词器模型文件的路径，可选
        src_lang=None,                   # 源语言的语言代码，可选
        tgt_lang=None,                   # 目标语言的语言代码，可选
        sp_model_kwargs: Optional[Dict[str, Any]] = None,  # SentencePiece 模型的额外参数，可选
        additional_special_tokens=None,  # 额外的特殊符号列表，可选
        **kwargs,                        # 其他未明确指定的参数，以字典形式接收
        # Mask token behaves like a normal word, including the space before it
        mask_token = (
            AddedToken(mask_token, lstrip=True, normalized=False) if isinstance(mask_token, str) else mask_token
        )

        # Initialize SentencePiece model keyword arguments, defaulting to an empty dictionary if not provided
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # Create a SentencePieceProcessor object and load the vocabulary file
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        # Ensure alignment between fairseq and SentencePiece vocabularies for specific tokens
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # Offset for fairseq vocabulary to align with SentencePiece model's tokens
        self.fairseq_offset = 1

        # Determine the size of the SentencePiece model's vocabulary
        self.sp_model_size = len(self.sp_model)

        # Map language codes to IDs based on fairseq language codes
        self.lang_code_to_id = {
            code: self.sp_model_size + i + self.fairseq_offset for i, code in enumerate(FAIRSEQ_LANGUAGE_CODES)
        }

        # Reverse mapping from ID to language code
        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}

        # Define the ID for the <mask> token in the fairseq context
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset

        # Update fairseq token mappings with language code mappings
        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)

        # Reverse mapping from fairseq token IDs to tokens
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        # Extend additional special tokens with language codes if provided
        _additional_special_tokens = list(self.lang_code_to_id.keys())
        if additional_special_tokens is not None:
            # Only add those special tokens if they are not already present
            _additional_special_tokens.extend(
                [t for t in additional_special_tokens if t not in _additional_special_tokens]
            )

        # Initialize the superclass with tokenization parameters and additional settings
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            tokenizer_file=None,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=_additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

        # Set the current source language and its corresponding ID
        self._src_lang = src_lang if src_lang is not None else "en_XX"
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]

        # Set source language special tokens based on the chosen source language
        self.set_src_lang_special_tokens(self._src_lang)
    # 返回对象的状态字典，包括所有实例变量及其值
    def __getstate__(self):
        state = self.__dict__.copy()
        # 将 sp_model 设置为 None，用于序列化状态
        state["sp_model"] = None
        # 将 sp_model_proto 设置为序列化后的 sp_model 模型原型
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    # 设置对象的状态，使用传入的状态字典 d
    def __setstate__(self, d):
        self.__dict__ = d

        # 兼容旧版本代码
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 根据 sp_model_kwargs 创建 sp_model 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 从序列化的 sp_model_proto 中加载 sp_model 的状态
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    # 返回特定属性 vocab_size 的计算值
    @property
    def vocab_size(self):
        return len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset + 1  # 加 1 是为了掩码标记

    # 返回特定属性 src_lang 的值
    @property
    def src_lang(self) -> str:
        return self._src_lang

    # 设置特定属性 src_lang 的值，并更新 src_lang 的特殊标记
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    # 根据 token_ids_0 和 token_ids_1 判断特殊标记的掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        根据是否已经有特殊标记，获取未添加特殊标记的 token 列表的序列标识符。该方法在使用分词器的 `prepare_for_model` 方法添加特殊标记时调用。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                token 列表是否已经格式化为模型的特殊标记。

        Returns:
            `List[int]`: 包含整数的列表，范围在 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 创建前缀和后缀的全 1 列表作为特殊标记的掩码
        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    # 构建带有特殊标记的输入序列
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
    def build_inputs_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记。MBART 序列具有以下格式，其中 `X` 表示序列：

        - `input_ids` (用于编码器)：`X [eos, src_lang_code]`
        - `decoder_input_ids` (用于解码器)：`X [eos, tgt_lang_code]`

        BOS 标记不被使用。序列对不是预期的使用情况，但会在没有分隔符的情况下处理。

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 带有适当特殊标记的输入 ID 列表。
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # 对于 API 一致性，留下对序列对的逻辑处理
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传入的两个序列创建用于序列对分类任务的掩码。MBART 不使用 token type ids，因此返回一个零列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 零列表。
        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """用于翻译管道，准备用于 generate 函数的输入"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def get_vocab(self):
        """
        返回词汇表，将词汇映射到其对应的 ID。

        Returns:
            dict: 包含所有词汇及其 ID 的字典。
        """
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """
        使用 subword 编码器对文本进行分词。

        Args:
            text (str): 要分词的文本。

        Returns:
            List[str]: 分词后的字符串列表。
        """
        return self.sp_model.encode(text, out_type=str)
    def _convert_token_to_id(self, token):
        """Converts a token (str) into an id using the vocabulary.

        Args:
            token (str): The token to convert.

        Returns:
            int: The corresponding id from the fairseq_tokens_to_ids dictionary
                 if present, otherwise uses the SentencePiece model to fetch
                 the id. Returns unk_token_id if the SentencePiece model returns 0.
        """
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) into a token (str) using the vocabulary.

        Args:
            index (int): The index to convert into a token.

        Returns:
            str: The corresponding token from the fairseq_ids_to_tokens dictionary
                 if present, otherwise uses the SentencePiece model to fetch
                 the token.
        """
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) into a single string.

        Args:
            tokens (List[str]): List of tokens to concatenate.

        Returns:
            str: The concatenated string formed from tokens, with SPIECE_UNDERLINE
                 replaced by a space and leading/trailing whitespace removed.
        """
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Saves the vocabulary to a directory.

        Args:
            save_directory (str): The directory path where the vocabulary will be saved.
            filename_prefix (Optional[str]): Optional prefix for the vocabulary file name.

        Returns:
            Tuple[str]: A tuple containing the path of the saved vocabulary file.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "ro_RO",
        **kwargs,
    ) -> BatchEncoding:
        """Prepares a batch for sequence-to-sequence model.

        Args:
            src_texts (List[str]): List of source texts.
            src_lang (str, optional): Source language code. Defaults to "en_XX".
            tgt_texts (Optional[List[str]], optional): List of target texts. Defaults to None.
            tgt_lang (str, optional): Target language code. Defaults to "ro_RO".
            **kwargs: Additional keyword arguments passed to the superclass method.

        Returns:
            BatchEncoding: The prepared batch containing encoded inputs for the model.
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        """Switches the model to input mode by setting source language special tokens."""
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        """Switches the model to target mode by setting target language special tokens."""
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Resets special tokens to match the source language settings.

        Args:
            src_lang (str): Source language code.

        Returns:
            None
        """
        self.cur_lang_code = self.lang_code_to_id[src_lang]
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """设置目标语言的特殊标记。无前缀，后缀为[eos, tgt_lang_code]。"""
        # 将当前语言代码设置为给定语言对应的 ID
        self.cur_lang_code = self.lang_code_to_id[lang]
        # 清空前缀标记列表
        self.prefix_tokens = []
        # 设置后缀标记列表为 [eos, 当前语言代码]
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
```