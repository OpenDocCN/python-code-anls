# `.\models\wav2vec2\tokenization_wav2vec2.py`

```py
# coding=utf-8
# Copyright 2021 The Facebook Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization class for Wav2Vec2."""

import json
import os
import sys
import warnings
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

# 导入父目录中的工具函数和类
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken, BatchEncoding
from ...utils import (
    ModelOutput,
    PaddingStrategy,
    TensorType,
    add_end_docstrings,
    is_flax_available,
    is_tf_available,
    is_torch_available,
    logging,
    to_py_obj,
)

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 如果类型检查开启，根据可用的深度学习框架导入相应库
if TYPE_CHECKING:
    if is_torch_available():
        import torch
    if is_tf_available():
        import tensorflow as tf
    if is_flax_available():
        import jax.numpy as jnp  # noqa: F401

# 指定预训练模型的文件名
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}

# 指定预训练模型和其对应的文件下载链接
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/wav2vec2-base-960h": "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json",
    },
    "tokenizer_config_file": {
        "facebook/wav2vec2-base-960h": (
            "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/tokenizer_config.json"
        ),
    },
}

# 指定特定预训练模型的位置编码最大长度，此处设置为系统支持的最大整数
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/wav2vec2-base-960h": sys.maxsize}
# WAV2VEC2_KWARGS_DOCSTRING 是一个包含详细参数说明的多行字符串常量，用于描述 Wav2Vec2 模型的初始化参数及其作用。
WAV2VEC2_KWARGS_DOCSTRING = r"""
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            verbose (`bool`, *optional*, defaults to `True`):
                Whether or not to print more information and warnings.
"""

# ListOfDict 是一个类型别名，表示一个列表，其中每个元素是一个字典，字典的键为字符串，值为整数或字符串。
ListOfDict = List[Dict[str, Union[int, str]]]

# Wav2Vec2CTCTokenizerOutput 是一个数据类，继承自 ModelOutput，表示 Wav2Vec2 模型的输出类型，包含解码后的文本及其偏移量信息。
@dataclass
class Wav2Vec2CTCTokenizerOutput(ModelOutput):
    """
    Output type of [` Wav2Vec2CTCTokenizer`], with transcription.

    Args:
        text (list of `str` or `str`):
            Decoded logits in text from. Usually the speech transcription.
        char_offsets (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]]`):
            Offsets of the decoded characters. In combination with sampling rate and model downsampling rate char
            offsets can be used to compute time stamps for each charater. Total logit score of the beam associated with
            produced text.
        word_offsets (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]]`):
            Offsets of the decoded words. In combination with sampling rate and model downsampling rate word offsets
            can be used to compute time stamps for each word.
    """

    text: Union[List[str], str]
    # 定义变量 char_offsets 和 word_offsets，它们的类型可以是 List[ListOfDict] 或 ListOfDict 或 None
    char_offsets: Union[List[ListOfDict], ListOfDict] = None
    word_offsets: Union[List[ListOfDict], ListOfDict] = None
# 定义一个名为 Wav2Vec2CTCTokenizer 的类，继承自 PreTrainedTokenizer
class Wav2Vec2CTCTokenizer(PreTrainedTokenizer):

    """
    Constructs a Wav2Vec2CTC tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        word_delimiter_token (`str`, *optional*, defaults to `"|"`):
            The token used for defining the end of a word.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to accept lowercase input and lowercase the output when decoding.
        target_lang (`str`, *optional*):
            A target language the tokenizer should set by default. `target_lang` has to be defined for multi-lingual,
            nested vocabulary such as [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all).

        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """

    # 定义类变量，指定模型词汇文件的名称列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型词汇文件的映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 模型最大输入尺寸的列表
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化方法，用于实例化类对象
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        replace_word_delimiter_char=" ",
        do_lower_case=False,
        target_lang=None,
        **kwargs,
        ):
        self._word_delimiter_token = word_delimiter_token
        # 设置词分隔符标记

        self.do_lower_case = do_lower_case
        # 是否将所有字符转换为小写

        self.replace_word_delimiter_char = replace_word_delimiter_char
        # 是否替换词分隔符字符

        self.target_lang = target_lang
        # 目标语言

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.vocab = json.load(vocab_handle)
        # 使用 UTF-8 编码打开词汇文件，并加载为 JSON 格式的词汇表

        # if target lang is defined vocab must be a nested dict
        # with each target lang being one vocabulary
        if target_lang is not None:
            self.encoder = self.vocab[target_lang]
        else:
            self.encoder = self.vocab
        # 如果定义了目标语言，则编码器是词汇表中对应目标语言的嵌套字典，否则为整个词汇表

        self.decoder = {v: k for k, v in self.encoder.items()}
        # 解码器将编码器的键值对颠倒，用于从编码到原始值的映射

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            word_delimiter_token=word_delimiter_token,
            replace_word_delimiter_char=replace_word_delimiter_char,
            target_lang=target_lang,
            **kwargs,
        )
        # 调用父类的初始化方法，设置特殊的令牌和参数

        # make sure that tokens made of several
        # characters are not split at tokenization
        for token in self.encoder.keys():
            if len(token) > 1:
                self.add_tokens(AddedToken(token, rstrip=True, lstrip=True, normalized=False))
        # 确保由多个字符组成的令牌在标记化过程中不会被拆分

    def set_target_lang(self, target_lang: str):
        """
        Set the target language of a nested multi-lingual dictionary
        """
        if self.vocab == self.encoder:
            raise ValueError(f"{self.vocab} is not a multi-lingual, nested tokenizer. Cannot set target language.")
        # 如果词汇表等于编码器，抛出值错误异常，因为它不是多语言嵌套分词器，无法设置目标语言

        if target_lang not in self.vocab:
            raise ValueError(f"{target_lang} does not exist. Choose one of {', '.join(self.vocab.keys())}.")
        # 如果目标语言不在词汇表中，抛出值错误异常，提示选择有效的目标语言

        self.target_lang = target_lang
        self.init_kwargs["target_lang"] = target_lang
        self.encoder = self.vocab[target_lang]
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置新的目标语言，并更新编码器和解码器

        # make sure that tokens made of several
        # characters are not split at tokenization
        for token in self.encoder.keys():
            if len(token) > 1:
                self.add_tokens(AddedToken(token, rstrip=True, lstrip=True, normalized=False))
        # 确保由多个字符组成的令牌在标记化过程中不会被拆分

    @property
    def word_delimiter_token(self) -> str:
        """
        `str`: Word delimiter token. Log an error if used while not having been set.
        """
        if self._word_delimiter_token is None and self.verbose:
            logger.error("Using word_delimiter_token, but it is not set yet.")
            return None
        return str(self._word_delimiter_token)
        # 返回词分隔符标记，如果未设置且 verbose 标志为真，则记录错误并返回 None

    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the word_delimiter_token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self._word_delimiter_token is None:
            return None
        return self.convert_tokens_to_ids(self.word_delimiter_token)
        # 返回词分隔符在词汇表中的 ID，如果未设置则返回 None
    # 定义属性 `word_delimiter_token` 的 setter 方法，用于设置单词分隔符的值
    @word_delimiter_token.setter
    def word_delimiter_token(self, value):
        self._word_delimiter_token = value

    # 定义属性 `word_delimiter_token_id` 的 setter 方法，用于将指定值转换为其对应的 ID
    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value):
        self._word_delimiter_token = self.convert_tokens_to_ids(value)

    # 返回词汇表的大小，即解码器中条目的数量
    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    # 获取词汇表，返回由编码器和添加的特殊标记编码器组成的字典
    def get_vocab(self) -> Dict:
        vocab = dict(self.encoder)
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 重写方法 `_add_tokens`，用于添加新的标记到词汇表中，不会去除任何空白
    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        # 创建要添加的标记列表
        to_add = []
        for token in new_tokens:
            if isinstance(token, str):
                # 如果标记是字符串，创建一个 `AddedToken` 对象并加入列表
                to_add.append(AddedToken(token, rstrip=False, lstrip=False, normalized=False))
            else:
                # 否则直接加入列表
                to_add.append(token)

        # 调用父类方法 `_add_tokens`，将新标记添加到词汇表中并返回添加的数量
        return super()._add_tokens(to_add, special_tokens)

    # 方法 `_tokenize`，将输入文本转换为标记序列，使用指定的单词分隔符
    def _tokenize(self, text, **kwargs):
        """
        Converts a string into a sequence of tokens (string), using the tokenizer.
        """
        if self.do_lower_case:
            # 如果设置为小写模式，将文本转换为大写
            text = text.upper()

        # 将文本中的空格替换为单词分隔符，并返回结果列表
        return list(text.replace(" ", self.word_delimiter_token))

    # 方法 `_convert_token_to_id`，根据词汇表将标记转换为对应的 ID
    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 方法 `_convert_id_to_token`，根据词汇表将 ID 转换为对应的标记
    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        # 获取对应索引的标记，如果找不到，则使用未知标记
        result = self.decoder.get(index, self.unk_token)
        return result

    # 方法 `convert_tokens_to_string`，将标记列表转换为字符串表示形式
    def convert_tokens_to_string(
        self,
        tokens: List[str],
        group_tokens: bool = True,
        spaces_between_special_tokens: bool = False,
        output_char_offsets: bool = False,
        output_word_offsets: bool = False,
    ):
        # 该方法的功能描述应继续添加在此处
        pass  # 在此处添加方法的功能描述
    `
        )n
        Dict[str, Union[str,ict[str, Union[str, float]]:
            """
            Converts a connectionist-temporal-classification (CTC) output tokens[str, Union[str, float]]:
            """
            Converts a connectionist-temporal-classification (CTC) output tokens into a single Union[str, float]]:
            """
            Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
    on[str, float]]:
            """
            Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回n[str, float]]:
            """
            Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏 float]]:
            """
            Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表at]]:
            """
            Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字t]]:
            """
            Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典       """
            Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
       """
            Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) ==    Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsetserts a connectionist-temporal-classification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets":nnectionist-temporal-classification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [],ectionist-temporal-classification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "emporal-classification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "wordoral-classification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsetsclassification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets":sification (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []tion (CTC) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
    ) output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果output tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要tput tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相put tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的ut tokens into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的ns into a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens a single string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，ngle string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照le string.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风ng.
            """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格     """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解   """
            # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码        # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
       # 如果 tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            tokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            ifokens 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
               ns 为空，返回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby回一个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars,个包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((包含空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token空字符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
               符串、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将、空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置空字符偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
    偏移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_re移列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len列表和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens)和空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
    空单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 单词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个词偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 C偏移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空移列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
    列表的字典
            if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars       if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter    if len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: charif len(tokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.padtokens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替okens) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 tokenns) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_charss) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
               ) == 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_del== 0:
                return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if       return {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == selfreturn {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiterurn {"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for{"text": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processedtext": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
           ": "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏 "", "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移 "char_offsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏ffsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_charsets": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
               ": [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self [], "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets "word_offsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetffsets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars,sets": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_tokens": []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                #: []}
            # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char       # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed     # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度  # 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len# 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char 如果需要分组 tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets tokens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets)okens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) !=kens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise Valueens，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {ns，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} ands，将相同的 tokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokenstokens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processedkens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to bens 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the sames 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but 合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                       合并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len并为不重复的 tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offset tokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {lenokens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} andkens，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
    ns，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {s，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed，按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                   按照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                #照 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的 CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字CTC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "TC 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char"C 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应 风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的风格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
               格解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i,解码
            if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in    if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars if group_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsetsoup_tokens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["charens:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = chars:
                # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                   # 使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要使用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏用 groupby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算upby 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏y 函数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word数将 tokens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = Noneens 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
               s 分组，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
    ，并计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word计算每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self每个 token 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(charken 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delen 的重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果重复次数
                chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符            chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char      chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                ifars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    charrepetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = Noneepetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将ions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的= zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成ip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token ken, len(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分n(list(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
           st(group_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_charp_iter))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " ifr))) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special)) for token, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
    n, group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join group_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
           roup_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将oup_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为up_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，_iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
    iter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lowerter in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
    er in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string =r in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
     in groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个n groupby(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的by(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符y(tokens)))
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移
            else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的   else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典 else:
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
                # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return           # 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"# 否则，直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text直接将 chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char chars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets,ars 设置为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
       为 tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
       tokens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _computens，并将 char_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
           har_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetar_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List_repetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[intepetitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int],petitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], charsitions 设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars:设置为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List为每个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], c个 token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，token 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符ken 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的n 重复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积复次数为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
    为 1
                chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np          chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char         chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions     chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
     chars = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始 = tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼 tokens
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 
                char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去   char_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个r_repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组repetitions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
           itions = len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([= len(tokens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0okens) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices) * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            * [1]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始]
    
            # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和        # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索   # 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移 过滤掉 self.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典elf.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
           lf.pad_token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
    _token，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char":en，这个 token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": token 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": sen 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s 用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars用作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices,作 CTC 的空白 token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
           token
            processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
            ]
    
            #        processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
            ]
    
            # 过滤掉 C       processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
            ]
    
            # 过滤掉 CTC token   processed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
            ]
    
            # 过滤掉 CTC token
            offsets =ocessed_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
            ]
    
            # 过滤掉 CTC token
            offsets = list(filter(lambdased_chars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
            ]
    
            # 过滤掉 CTC token
            offsets = list(filter(lambda offsets: offsetsars = list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
            ]
    
            # 过滤掉 CTC token
            offsets = list(filter(lambda offsets: offsets["char"]list(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
            ]
    
            # 过滤掉 CTC token
            offsets = list(filter(lambda offsets: offsets["char"] != ctc_token, offsets))
    st(filter(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
            ]
    
            # 过滤掉 CTC token
            offsets = list(filter(lambda offsets: offsets["char"] != ctc_token, offsets))
            #er(lambda char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
            ]
    
            # 过滤掉 CTC token
            offsets = list(filter(lambda offsets: offsets["char"] != ctc_token, offsets))
            # 返回过滤后的偏移列表
           char: char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
            ]
    
            # 过滤掉 CTC token
            offsets = list(filter(lambda offsets: offsets["char"] != ctc_token, offsets))
            # 返回过滤后的偏移列表
            return offsets
    
    char != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
            ]
    
            # 过滤掉 CTC token
            offsets = list(filter(lambda offsets: offsets["char"] != ctc_token, offsets))
            # 返回过滤后的偏移列表
            return offsets
    
        @staticmethodar != self.pad_token, chars))
    
            # 替换分隔符 token
            processed_chars = [
                self.replace_word_delimiter_char if char == self.word_delimiter_token else char for char in processed_chars
            ]
    
            # 初始化 char_offsets 和 word_offsets 为 None
            char_offsets = word_offsets = None
            # 如果需要输出字符偏移或单词偏移，计算字符偏移
            if output_char_offsets or output_word_offsets:
                char_offsets = self._compute_offsets(char_repetitions, chars, self.pad_token)
    
                # 检查 char_offsets 和 processed_chars 长度是否一致
                if len(char_offsets) != len(processed_chars):
                    raise ValueError(
                        f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                        " have to be of the same length, but are: "
                        f"`len(offsets)`: {len(char_offsets)} and `len(processed_tokens)`:"
                        f" {len(processed_chars)}"
                    )
    
                # 将 char_offsets 中的每个字典的 "char" 键更新为对应的处理后的 token
                for i, char in enumerate(processed_chars):
                    char_offsets[i]["char"] = char
    
                # 如果需要输出单词偏移，计算单词偏移
                word_offsets = None
                if output_word_offsets:
                    word_offsets = self._get_word_offsets(char_offsets, self.replace_word_delimiter_char)
    
                # 如果不需要输出字符偏移，将 char_offsets 设置为 None
                if not output_char_offsets:
                    char_offsets = None
    
            # 将处理后的字符列表连接成一个字符串，特殊 token 之间用空格分隔
            join_char = " " if spaces_between_special_tokens else ""
            string = join_char.join(processed_chars).strip()
    
            # 如果需要将字符串转换为小写，进行转换
            if self.do_lower_case:
                string = string.lower()
    
            # 返回一个包含生成的字符串、字符偏移和单词偏移的字典
            return {"text": string, "char_offsets": char_offsets, "word_offsets": word_offsets}
    
        @staticmethod
        def _compute_offsets(
            char_repetitions: List[int], chars: List[str], ctc_token: int
        ) -> List[Dict[str, Union[str, int]]]:
            # 计算字符的结束索引，使用字符重复次数的累积和
            end_indices = np.asarray(char_repetitions).cumsum()
            # 计算字符的开始索引，拼接一个 0 和去掉最后一个结束索引的数组
            start_indices = np.concatenate(([0], end_indices[:-1]))
    
            # 根据字符、开始索引和结束索引创建偏移字典列表
            offsets = [
                {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
            ]
    
            # 过滤掉 CTC token
            offsets = list(filter(lambda offsets: offsets["char"] != ctc_token, offsets))
            # 返回过滤后的偏移列表
            return offsets
    
        @staticmethod
    def _get_word_offsets(
        offsets: Dict[str, Union[str, float]], word_delimiter_char: str = " "
    ) -> Dict[str, Union[str, float]]:
        # 初始化一个空列表，用于存储单词的偏移量信息
        word_offsets = []

        # 初始化上一个字符的状态为"SPACE"
        last_state = "SPACE"
        # 初始化单词字符串为空
        word = ""
        # 初始化单词的起始偏移量和结束偏移量为0
        start_offset = 0
        end_offset = 0

        # 遍历偏移量字典的索引和值
        for i, offset in enumerate(offsets):
            # 获取当前字符
            char = offset["char"]
            # 根据当前字符是否为单词分隔符，确定当前状态是"SPACE"还是"WORD"
            state = "SPACE" if char == word_delimiter_char else "WORD"

            # 如果当前状态和上一个状态相同，则继续处理当前单词
            if state == last_state:
                # 更新结束偏移量
                end_offset = offset["end_offset"]
                # 将当前字符添加到单词字符串中
                word += char
            else:
                # 如果状态不同，表示单词边界发生变化
                if state == "SPACE":
                    # 完成一个单词的识别，将其信息加入到单词偏移量列表中
                    word_offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})
                else:
                    # 开始识别一个新单词，更新起始偏移量和结束偏移量，并重新设置单词字符串
                    start_offset = offset["start_offset"]
                    end_offset = offset["end_offset"]
                    word = char

            # 更新上一个状态为当前状态
            last_state = state
        
        # 最后处理最后一个单词，如果上一个状态是"WORD"，则将其加入到单词偏移量列表中
        if last_state == "WORD":
            word_offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})

        # 返回单词偏移量列表
        return word_offsets

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # 如果已经将文本分割为单词，则在文本前添加一个空格
        if is_split_into_words:
            text = " " + text
        # 返回处理后的文本和附加的关键字参数
        return (text, kwargs)

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        group_tokens: bool = True,
        spaces_between_special_tokens: bool = False,
        output_word_offsets: Optional[bool] = False,
        output_char_offsets: Optional[bool] = False,
        ```
    ) -> str:
        """
        special _decode function is needed for Wav2Vec2Tokenizer because added tokens should be treated exactly the
        same as tokens of the base vocabulary and therefore the function `convert_tokens_to_string` has to be called on
        the whole token list and not individually on added tokens
        """
        # 将 token_ids 转换为 tokens，并过滤掉特殊 token（如果需要）
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        # 遍历过滤后的 tokens
        for token in filtered_tokens:
            # 如果需要跳过特殊 token 并且 token 在所有特殊 token 的集合中，则跳过此 token
            if skip_special_tokens and token in self.all_special_ids:
                continue
            # 将 token 添加到结果列表中
            result.append(token)

        # 将过滤后的 tokens 转换为字符串输出
        string_output = self.convert_tokens_to_string(
            result,
            group_tokens=group_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            output_word_offsets=output_word_offsets,
            output_char_offsets=output_char_offsets,
        )

        # 获取字符串形式的文本结果
        text = string_output["text"]

        # 根据需要清理 token 化的空格
        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)

        # 如果需要输出单词偏移或字符偏移，则返回 `Wav2Vec2CTCTokenizerOutput` 对象
        if output_word_offsets or output_char_offsets:
            return Wav2Vec2CTCTokenizerOutput(
                text=text,
                char_offsets=string_output["char_offsets"],
                word_offsets=string_output["word_offsets"],
            )
        else:
            # 否则，直接返回文本结果
            return text

    # 从 `tokenization_utils_base.py` 覆写，因为 tokenizer 可以输出 `ModelOutput`，这不应该是批量输出的列表，并且在这里需要对 `output_char_offsets` 进行文档化
    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        output_char_offsets: bool = False,
        output_word_offsets: bool = False,
        **kwargs,
    ) -> List[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces.
            output_char_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output character offsets. Character offsets can be used in combination with the
                sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.

                <Tip>

                Please take a look at the Example of [`~Wav2Vec2CTCTokenizer.decode`] to better understand how to make
                use of `output_char_offsets`. [`~Wav2Vec2CTCTokenizer.batch_decode`] works the same way with batched
                output.

                </Tip>

            output_word_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate
                and model downsampling rate to compute the time-stamps of transcribed words.

                <Tip>

                Please take a look at the Example of [`~Wav2Vec2CTCTokenizer.decode`] to better understand how to make
                use of `output_word_offsets`. [`~Wav2Vec2CTCTokenizer.batch_decode`] works the same way with batched
                output.

                </Tip>

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `List[str]` or [`~models.wav2vec2.tokenization_wav2vec2.Wav2Vec2CTCTokenizerOutput`]: The list of decoded
            sentences. Will be a [`~models.wav2vec2.tokenization_wav2vec2.Wav2Vec2CTCTokenizerOutput`] when
            `output_char_offsets == True` or `output_word_offsets == True`.
        """
        # Decode each sequence in the batch using the `decode` method of the tokenizer
        batch_decoded = [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                output_char_offsets=output_char_offsets,
                output_word_offsets=output_word_offsets,
                **kwargs,
            )
            for seq in sequences
        ]
        
        # If either `output_char_offsets` or `output_word_offsets` is True
        if output_char_offsets or output_word_offsets:
            # Transform list of dictionaries to a dictionary of lists
            return Wav2Vec2CTCTokenizerOutput({k: [d[k] for d in batch_decoded] for k in batch_decoded[0]})
        
        # Otherwise, return the list of decoded sentences
        return batch_decoded
    # 重写自 `tokenization_utils_base.py`，因为这里需要文档关于 `output_char_offsets` 和 `output_word_offsets`
    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        output_char_offsets: bool = False,
        output_word_offsets: bool = False,
        **kwargs,
    ):
        # 检查保存目录是否存在，若不存在则报错
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 拼接词汇文件路径，根据给定的前缀和标准的文件名
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 将词汇表写入到 JSON 格式的文件中
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 返回保存的词汇文件路径的元组
        return (vocab_file,)
class Wav2Vec2Tokenizer(PreTrainedTokenizer):
    """
    Constructs a Wav2Vec2 tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        word_delimiter_token (`str`, *optional*, defaults to `"|"`):
            The token used for defining the end of a word.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the output when decoding.
        do_normalize (`bool`, *optional*, defaults to `False`):
            Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
            improve the performance for some models, *e.g.*,
            [wav2vec2-lv60](https://huggingface.co/models?search=lv60).
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not [`~Wav2Vec2Tokenizer.__call__`] should return `attention_mask`.

            <Tip>

            Wav2Vec2 models that have set `config.feat_extract_norm == "group"`, such as
            [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), have **not** been trained using
            `attention_mask`. For such models, `input_values` should simply be padded with 0 and no `attention_mask`
            should be passed.

            For Wav2Vec2 models that have set `config.feat_extract_norm == "layer"`, such as
            [wav2vec2-lv60](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self), `attention_mask` should be
            passed for batched inference.

            </Tip>

        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """

    # 定义类变量，包含了词汇文件的名称列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义预训练模型的词汇文件和配置文件的映射
    pretrained_vocab_files_map = {
        "vocab_file": {
            "facebook/wav2vec2-base-960h": "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json"
        },
        "tokenizer_config_file": {
            "facebook/wav2vec2-base-960h": (
                "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/tokenizer.json"
            ),
        },
    }
    # 定义模型的输入名称列表
    model_input_names = ["input_values", "attention_mask"]
    # 初始化函数，用于创建一个新的 Wav2Vec2Tokenizer 对象
    def __init__(
        self,
        vocab_file,                          # 词汇文件路径，用于加载词汇表
        bos_token="<s>",                     # 开始标记，默认为 "<s>"
        eos_token="</s>",                    # 结束标记，默认为 "</s>"
        unk_token="<unk>",                   # 未知标记，默认为 "<unk>"
        pad_token="<pad>",                   # 填充标记，默认为 "<pad>"
        word_delimiter_token="|",            # 单词分隔标记，默认为 "|"
        do_lower_case=False,                 # 是否将文本转换为小写，默认为 False
        do_normalize=False,                  # 是否对文本进行正规化，默认为 False
        return_attention_mask=False,         # 是否返回注意力掩码，默认为 False
        **kwargs,                            # 其他关键字参数
    ):
        # 发出警告，表明 Wav2Vec2Tokenizer 类已被弃用，并将在 Transformers 的第五版中移除
        warnings.warn(
            "The class `Wav2Vec2Tokenizer` is deprecated and will be removed in version 5 of Transformers. Please use"
            " `Wav2Vec2Processor` or `Wav2Vec2CTCTokenizer` instead.",
            FutureWarning,
        )

        # 设置单词分隔标记
        self._word_delimiter_token = word_delimiter_token

        # 设置是否转换为小写
        self.do_lower_case = do_lower_case
        # 设置是否返回注意力掩码
        self.return_attention_mask = return_attention_mask
        # 设置是否正规化
        self.do_normalize = do_normalize

        # 从 UTF-8 编码的词汇文件中加载词汇表到 encoder 字典中
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)

        # 通过 encoder 字典创建 decoder 字典，用于反向查找
        self.decoder = {v: k for k, v in self.encoder.items()}

        # 调用父类的初始化函数，传入各种标记和参数
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            do_normalize=do_normalize,
            return_attention_mask=return_attention_mask,
            word_delimiter_token=word_delimiter_token,
            **kwargs,
        )

    @property
    def word_delimiter_token(self) -> str:
        """
        `str`: 单词分隔标记。如果在未设置的情况下使用，记录错误日志。
        """
        # 如果单词分隔标记为 None 且 verbose 为 True，则记录错误日志并返回 None
        if self._word_delimiter_token is None and self.verbose:
            logger.error("Using word_delimiter_token, but it is not set yet.")
            return None
        # 否则，返回单词分隔标记的字符串形式
        return str(self._word_delimiter_token)

    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: 单词分隔标记在词汇表中的 id。如果未设置，则返回 None。
        """
        # 如果单词分隔标记为 None，则返回 None
        if self._word_delimiter_token is None:
            return None
        # 否则，返回单词分隔标记在词汇表中的 id
        return self.convert_tokens_to_ids(self.word_delimiter_token)

    @word_delimiter_token.setter
    def word_delimiter_token(self, value):
        # 设置单词分隔标记的值
        self._word_delimiter_token = value

    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value):
        # 根据给定的值设置单词分隔标记在词汇表中的 id
        self._word_delimiter_token = self.convert_tokens_to_ids(value)

    @add_end_docstrings(WAV2VEC2_KWARGS_DOCSTRING)
    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
        **kwargs,
    ):
        # 调用实例对象，接受原始语音输入以及一系列处理参数，并返回处理后的结果
    ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences.

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy array or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
        """
        # 检查输入是否为批处理的 numpy 数组，并且数组维度大于1
        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        # 如果是批处理的 numpy 数组，并且维度大于2，抛出异常
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        # 检查是否为批处理数据，可以是 numpy 数组或者列表/元组中包含 numpy 数组、元组、列表
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        # 确保输入格式为列表
        if is_batched and not isinstance(raw_speech[0], np.ndarray):
            raw_speech = [np.asarray(speech) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech)

        # 如果不是批处理形式，则将输入封装成列表
        if not is_batched:
            raw_speech = [raw_speech]

        # 如果需要进行归一化处理
        if self.do_normalize:
            # 对每个序列进行零均值和单位方差归一化
            raw_speech = [(x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5) for x in raw_speech]

        # 将输入编码为适合填充的格式
        encoded_inputs = BatchEncoding({"input_values": raw_speech})

        # 使用指定参数对输入进行填充
        padded_inputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return padded_inputs

    @property
    def vocab_size(self) -> int:
        # 返回解码器中的词汇表大小
        return len(self.decoder)

    def get_vocab(self) -> Dict:
        # 返回编码器和添加的特殊标记编码器的字典
        return dict(self.encoder, **self.added_tokens_encoder)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        # 根据词汇表将单词转换为对应的索引
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        # 根据词汇表将索引转换为对应的单词
        result = self.decoder.get(index, self.unk_token)
        return result
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a list of connectionist-temporal-classification (CTC) output tokens into a single string.

        Args:
            tokens (List[str]): List of tokens to be converted into a string.

        Returns:
            str: Converted string from tokens.
        """
        # Group tokens into non-repeating tokens in CTC style decoding
        grouped_tokens = [token_group[0] for token_group in groupby(tokens)]

        # Filter out self.pad_token which serves as the CTC-blank token
        filtered_tokens = list(filter(lambda token: token != self.pad_token, grouped_tokens))

        # Replace delimiter token with spaces and join tokens into a single string
        string = "".join([" " if token == self.word_delimiter_token else token for token in filtered_tokens]).strip()

        # Convert to lowercase if do_lower_case is True
        if self.do_lower_case:
            string = string.lower()

        return string

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        """
        Special _decode function for Wav2Vec2Tokenizer to handle added tokens exactly like base vocabulary tokens.

        Args:
            token_ids (List[int]): List of token IDs to be decoded into a string.
            skip_special_tokens (bool): Whether to skip special tokens.
            clean_up_tokenization_spaces (bool): Whether to clean up tokenization spaces.

        Returns:
            str: Decoded string from token IDs.
        """
        # Convert token IDs to tokens, filtering out special tokens if skip_special_tokens is True
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            result.append(token)

        # Convert filtered tokens into a single string
        text = self.convert_tokens_to_string(result)

        # Determine whether to clean up tokenization spaces
        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves the vocabulary as a JSON file in the specified directory.

        Args:
            save_directory (str): Directory path where vocabulary JSON should be saved.
            filename_prefix (Optional[str]): Optional prefix for the vocabulary JSON file name.

        Returns:
            Tuple[str]: Tuple containing the path to the saved vocabulary file.
        """
        # Ensure save_directory exists; otherwise log an error and return
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # Construct the full path for the vocabulary file
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # Write the vocabulary dictionary to the JSON file
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        return (vocab_file,)
```