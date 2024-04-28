# `.\transformers\models\nllb\tokenization_nllb.py`

```
# coding=utf-8
# 声明文件编码格式为 UTF-8

# 版权声明
# Copyright 2022 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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

# 导入标准库中的 os 模块
import os
# 从 shutil 模块中导入 copyfile 函数
from shutil import copyfile
# 从 typing 模块中导入 Any、Dict、List、Optional、Tuple 类型
from typing import Any, Dict, List, Optional, Tuple

# 导入 sentencepiece 库，用于分词
import sentencepiece as spm

# 从 ...tokenization_utils 模块中导入 AddedToken、BatchEncoding、PreTrainedTokenizer 类
from ...tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer
# 从 ...utils 模块中导入 logging 函数
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义表示空格的特殊字符
SPIECE_UNDERLINE = "▁"

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/nllb-200-distilled-600M": (
            "https://huggingface.co/facebook/nllb-200-distilled-600M/blob/main/sentencepiece.bpe.model"
        ),
    }
}

# 预训练模型的位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/nllb-200-distilled-600M": 1024,
}
# 定义一个包含多种语言代码的列表
FAIRSEQ_LANGUAGE_CODES = ['ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn', 'ajp_Arab', 'aka_Latn', ... 'yor_Latn', 'yue_Hant', 'zho_Hans', 'zho_Hant', 'zul_Latn']  # fmt: skip

# NllbTokenizer 类，继承自 PreTrainedTokenizer
class NllbTokenizer(PreTrainedTokenizer):
    """
    构建一个 NLLB 分词器

    从 `RobertaTokenizer` 和 `XLNetTokenizer` 改编而来，基于 SentencePiece

    对源语言文档，分词方法是 `<tokens> <eos> <language code>`，对目标语言文档，分词方法是 `<language code> <tokens> <eos>`

    示例：

    ```python
    >>> from transformers import NllbTokenizer

    >>> tokenizer = NllbTokenizer.from_pretrained(
    # 定义一个函数，用于初始化一个 tokenizer 对象
    def __init__(
        # 词汇表文件的路径
        vocab_file: str,
        # 序列起始标记，通常用于序列分类任务
        bos_token: str = "<s>",
        # 序列结束标记
        eos_token: str = "</s>",
        # 序列分隔标记，用于多个序列拼接或者序列分类任务
        sep_token: str = "</s>",
        # 分类器标记，在进行序列分类时使用，通常为序列的第一个标记
        cls_token: str = "<s>",
        # 未知标记，当词汇表中不存在某个词时，使用该标记代替
        unk_token: str = "<unk>",
        # 填充标记，在进行序列长度不一致的批处理时使用
        pad_token: str = "<pad>",
        # 掩码标记，用于掩盖模型训练中的一些词汇
        mask_token: str = "<mask>",
        # 自定义的 tokenizer 文件路径，替代默认的词汇表文件
        tokenizer_file: str = None,
        # 源语言代码，用于翻译任务
        src_lang: str = None,
        # 目标语言代码，用于翻译任务
        tgt_lang: str = None,
        # sp_model_kwargs 的其他关键字参数
        sp_model_kwargs: Dict[str, str] = None
    ):
        # 定义了一些常用的词汇表文件名称
        vocab_files_names = VOCAB_FILES_NAMES
    # 将最大模型输入大小设为预训练位置嵌入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 将预训练词汇文件映射设为预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 将模型输入名称设为["input_ids", "attention_mask"]
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化前缀标记和后缀标记为空列表
    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    # 初始化函数，接收多个参数，包括词汇文件、起始标记、结束标记、分隔标记等
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        tokenizer_file=None,
        src_lang=None,
        tgt_lang=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        additional_special_tokens=None,
        legacy_behaviour=False,
        **kwargs,
    # 获取对象的状态
    def __getstate__(self):
        state = self.__dict__.copy()
        # 将sp_model设置为None
        state["sp_model"] = None
        # 将sp_model_proto设置为sp_model的序列化模型原型
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    # 设置对象的状态
    def __setstate__(self, d):
        # 将对象的状态设置为d
        self.__dict__ = d

        # 如果对象没有sp_model_kwargs属性，则将其设置为空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用sp_model_kwargs初始化sp_model
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 从序列化的原型中加载sp_model
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    # 获取词汇大小
    @property
    def vocab_size(self):
        return len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset + 1  # 加1是为了遮罩标记

    # 获取源语言
    @property
    def src_lang(self) -> str:
        return self._src_lang

    # 设置源语言
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    # 获取特殊标记的掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An NLLB sequence has the following format, where `X` represents the sequence:

        - `input_ids` (for encoder) `X [eos, src_lang_code]`
        - `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def create_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. nllb does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.

        """

        sep = [self.sep_token_id]  # Create a list containing the separation token id
        cls = [self.cls_token_id]  # Create a list containing the classification token id

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]  # Return a list of zeros based on token_ids_0
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]  # Return a list of zeros based on token_ids_0 and token_ids_1

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs  # Return processed inputs for translation

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}  # Create a vocabulary dictionary
        vocab.update(self.added_tokens_encoder)  # Update with added token encoder
        return vocab  # Return the vocabulary dictionary

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)  # Tokenize the input text using sp_model and return as a list

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()  # Concatenate tokens, replace special character, and remove leading/trailing whitespace
        return out_string  # Return the concatenated string
```  
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与输出路径不同且当前词汇表文件存在，就复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，就将模型序列化后写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "eng_Latn",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "fra_Latn",
        **kwargs,
    ) -> BatchEncoding:
        # 设置源语言和目标语言，并调用父类方法准备批次数据
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        # 切换为输入模式，设置特殊的源语言标记
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        # 切换为目标模式，设置特殊的目标语言标记
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting.
        - In legacy mode: No prefix and suffix=[eos, src_lang_code].
        - In default mode: Prefix=[src_lang_code], suffix = [eos]
        """
        # 根据源语言设置重置特殊标记
        self.cur_lang_code = self.lang_code_to_id[src_lang]
        if self.legacy_behaviour:
            self.prefix_tokens = []
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.prefix_tokens = [self.cur_lang_code]
            self.suffix_tokens = [self.eos_token_id]

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target lang setting.
        - In legacy mode: No prefix and suffix=[eos, tgt_lang_code].
        - In default mode: Prefix=[tgt_lang_code], suffix = [eos]
        """
        # 根据目标语言设置重置特殊标记
        self.cur_lang_code = self.lang_code_to_id[lang]
        if self.legacy_behaviour:
            self.prefix_tokens = []
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.prefix_tokens = [self.cur_lang_code]
            self.suffix_tokens = [self.eos_token_id]
```