# `.\transformers\models\wav2vec2_phoneme\tokenization_wav2vec2_phoneme.py`

```
# 设置文件编码格式为 utf-8
# 版权声明
# 根据 Apache 许可版本 2.0 进行许可
# 除非符合许可条件，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何明示或暗示的担保或条件，包括但不限于
# 特定目的的适用性和适销性担保
# 有关特定语言管理权限和限制的详细信息，请参见许可证
"""Tokenization class for Wav2Vec2Phoneme."""

# 导入必要的模块
import json
import os
import sys
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken
from ...utils import (
    ModelOutput,
    is_flax_available,
    is_tf_available,
    is_torch_available,
    logging,
    requires_backends,
    to_py_obj,
)

# 获取程序日志记录器
logger = logging.get_logger(__name__)

# 如果是类型检查，则导入对应的模块
if TYPE_CHECKING:
    if is_torch_available():
        import torch
    if is_tf_available():
        import tensorflow as tf
    if is_flax_available():
        import jax.numpy as jnp  # noqa: F401

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}

# 定义预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/wav2vec2-lv-60-espeak-cv-ft": (
            "https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/resolve/main/vocab.json"
        ),
    },
    "tokenizer_config_file": {
        "facebook/wav2vec2-lv-60-espeak-cv-ft": (
            "https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/resolve/main/tokenizer_config.json"
        ),
    },
}

# Wav2Vec2Phoneme 没有最大输入长度
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/wav2vec2-lv-60-espeak-cv-ft": sys.maxsize}

# 定义类型别名 ListOfDict
ListOfDict = List[Dict[str, Union[int, str]]]

# 定义 Wav2Vec2PhonemeCTCTokenizerOutput 输出类
@dataclass
class Wav2Vec2PhonemeCTCTokenizerOutput(ModelOutput):
    """
    Output type of [` Wav2Vec2PhonemeCTCTokenizer`], with transcription.
    Args:
        text (list of `str` or `str`):
            Decoded logits in text from. Usually the speech transcription.
        char_offsets (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]`):
            Offsets of the decoded characters. In combination with sampling rate and model downsampling rate char
            offsets can be used to compute time stamps for each charater. Total logit score of the beam associated with
            produced text.
    """
    text: Union[List[str], str]
    char_offsets: Union[List[ListOfDict], ListOfDict] = None

# 定义 Wav2Vec2PhonemeCTCTokenizer 类
class Wav2Vec2PhonemeCTCTokenizer(PreTrainedTokenizer):
    """
    Constructs a Wav2Vec2PhonemeCTC tokenizer.
    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.



    # Args: section for initializing the Tokenizer class with specified arguments and default values
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
        do_phonemize (`bool`, *optional*, defaults to `True`):
            Whether the tokenizer should phonetize the input or not. Only if a sequence of phonemes is passed to the
            tokenizer, `do_phonemize` should be set to `False`.
        phonemizer_lang (`str`, *optional*, defaults to `"en-us"`):
            The language of the phoneme set to which the tokenizer should phonetize the input text to.
        phonemizer_backend (`str`, *optional*, defaults to `"espeak"`):
            The backend phonetization library that shall be used by the phonemizer library. Defaults to `espeak-ng`.
            See the [phonemizer package](https://github.com/bootphon/phonemizer#readme) for more information.

        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    """

    # Set up default mappings and constants for the Tokenizer class
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    # Initialize Tokenizer class with specified parameters and default values
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        phone_delimiter_token=" ",
        word_delimiter_token=None,
        do_phonemize=True,
        phonemizer_lang="en-us",
        phonemizer_backend="espeak",
        **kwargs,
    # 初始化函数，设置一些实例属性
    def __init__(
        self,
        unk_token: str,
        bos_token: str,
        eos_token: str,
        pad_token: str,
        word_delimiter_token: str,
        phone_delimiter_token: str,
        vocab_file: str,
        do_phonemize: bool,
        phonemizer_lang: str,
        phonemizer_backend: str,
        **kwargs,
    ):
        # 设置单词分隔符，用于分割单词
        self._word_delimiter_token = word_delimiter_token
        # 设置电话分隔符，用于分隔电话号码
        self._phone_delimiter_token = phone_delimiter_token
        # 设置是否进行音素转换的标志
        self.do_phonemize = do_phonemize
        # 设置音素转换所用的语言
        self.phonemizer_lang = phonemizer_lang
        # 设置音素转换的后端
        self.phonemizer_backend = phonemizer_backend

        # 如果需要进行音素转换，则初始化音素转换后端
        if do_phonemize:
            self.init_backend(self.phonemizer_lang)

        # 从词汇文件中加载编码器
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 根据编码器创建解码器
        self.decoder = {v: k for k, v in self.encoder.items()}

        # 调用父类的初始化方法，传入相关参数
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            word_delimiter_token=word_delimiter_token,
            phone_delimiter_token=phone_delimiter_token,
            do_phonemize=do_phonemize,
            phonemizer_lang=phonemizer_lang,
            phonemizer_backend=phonemizer_backend,
            **kwargs,
        )

    # 返回词汇表的大小
    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    # 返回词汇表
    def get_vocab(self) -> Dict:
        vocab = dict(self.encoder.copy())
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 添加新的词汇到词汇表
    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        # 重写方法以避免删除词汇
        to_add = []
        for token in new_tokens:
            if isinstance(token, str):
                to_add.append(AddedToken(token, rstrip=False, lstrip=False, normalized=True, special=special_tokens))
            else:
                to_add.append(token)

        return super()._add_tokens(to_add, special_tokens)

    # 初始化音素转换后端
    def init_backend(self, phonemizer_lang: str):
        """
        Initializes the backend.

        Args:
            phonemizer_lang (`str`): The language to be used.
        """
        # 检查是否需要相关后端
        requires_backends(self, "phonemizer")
        # 导入音素转换的后端
        from phonemizer.backend import BACKENDS

        # 选择并初始化指定的后端
        self.backend = BACKENDS[self.phonemizer_backend](phonemizer_lang, language_switch="remove-flags")

    # 为标记化做准备
    def prepare_for_tokenization(
        self,
        text: str,
        is_split_into_words: bool = False,
        phonemizer_lang: Optional[str] = None,
        do_phonemize: Optional[bool] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.

        This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
        `kwargs` at the end of the encoding process to be sure all the arguments have been used.

        Args:
            text (`str`):
                The text to prepare.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            phonemizer_lang (`str`, *optional*):
                The language of the phoneme set to which the tokenizer should phonetize the input text to.
            do_phonemize (`bool`, *optional*):
                Whether the tokenizer should phonetize the input text or not. Only if a sequence of phonemes is passed
                to the tokenizer, `do_phonemize` should be set to `False`.

        Returns:
            `Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.
        """
        # 如果输入已经分词好，则在文本前加上空格
        if is_split_into_words:
            text = " " + text

        # 设置tokenizer是否进行phonetization（发音转换）
        if do_phonemize is not None:
            self.do_phonemize = do_phonemize

        # 设置正确的发音转换语言
        if phonemizer_lang is not None:
            self.phonemizer_lang = phonemizer_lang
            self.init_backend(phonemizer_lang)

        # 返回预处理好的文本和未使用的kwargs
        return (text, {})

    def _tokenize(self, text, **kwargs):
        """
        Converts a string into a sequence of tokens (string), using the tokenizer.
        """

        # 确保去除文本两端的空格，以防止<unk>出现
        text = text.strip()

        # 如果需要进行发音转换
        if self.do_phonemize:
            text = text.lower()

            # 创建发音转换后的音素列表
            text = self.phonemize(text, self.phonemizer_lang)

        # 确保音素之间有空格
        tokens = text.split(" ")

        # 过滤掉空白的音素
        tokens = list(filter(lambda p: p.strip() != "", tokens))
        return tokens
    def phonemize(self, text: str, phonemizer_lang: Optional[str] = None) -> str:
        # 导入分隔符类
        from phonemizer.separator import Separator

        # 如果设置了词分隔符，则在词之间添加空格
        word_delimiter = self.word_delimiter_token + " " if self.word_delimiter_token is not None else ""
        # 如果指定了语言且与当前语言不同，则重新初始化后端
        if phonemizer_lang is not None and phonemizer_lang != self.phonemizer_lang:
            self.init_backend(phonemizer_lang)
        else:
            phonemizer_lang = self.phonemizer_lang

        # 创建分隔符对象定义音节分隔符和词分隔符
        separator = Separator(phone=self.phone_delimiter_token, word=word_delimiter, syllable="")
        # 对文本进行音素转换
        phonemes = self.backend.phonemize(
            [text],
            separator=separator,
        )
        # 去掉结果中的空格
        phonemes = phonemes[0].strip()

        return phonemes

    @property
    def word_delimiter_token(self) -> str:
        """
        `str`: Word delimiter token. Log an error if used while not having been set.
        """
        # 如果词分隔符未设置，则记录错误
        if self._word_delimiter_token is None:
            if self.verbose:
                logger.error("Using word_delimiter_token, but it is not set yet.")
            return None
        return str(self._word_delimiter_token)

    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the word_delimiter_token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        # 如果词分隔符未设置，则返回 None
        if self._word_delimiter_token is None:
            return None
        return self.convert_tokens_to_ids(self.word_delimiter_token)

    @word_delimiter_token.setter
    def word_delimiter_token(self, value):
        # 设置词分隔符
        self._word_delimiter_token = value

    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value):
        # 将词分隔符转换为对应的 ID
        self._word_delimiter_token = self.convert_tokens_to_ids(value)

    @property
    def phone_delimiter_token(self) -> str:
        """
        `str`: Word delimiter token. Log an error if used while not having been set.
        """
        # 如果音素分隔符未设置，则记录错误
        if self._phone_delimiter_token is None:
            if self.verbose:
                logger.error("Using phone_delimiter_token, but it is not set yet.")
            return None
        return str(self._phone_delimiter_token)

    @property
    def phone_delimiter_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the phone_delimiter_token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        # 如果音素分隔符未设置，则返回 None
        if self._phone_delimiter_token is None:
            return None
        return self.convert_tokens_to_ids(self.phone_delimiter_token)

    @phone_delimiter_token.setter
    def phone_delimiter_token(self, value):
        # 设置音素分隔符
        self._phone_delimiter_token = value

    @phone_delimiter_token_id.setter
    def phone_delimiter_token_id(self, value):
        # 将音素分隔符转换为对应的 ID
        self._phone_delimiter_token = self.convert_tokens_to_ids(value)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        # 根据词汇表将 token(str) 转换为 ID(integer)
        return self.encoder.get(token, self.encoder.get(self.unk_token))
    # 使用词汇表将索引（整数）转换为标记（字符串）
    def _convert_id_to_token(self, index: int) -> str:
        result = self.decoder.get(index, self.unk_token)  # 通过索引在词汇表中查找对应标记，如果不存在则返回 unk_token
        return result  # 返回查找到的标记

    # 将连接主时序分类（CTC）输出的标记转换为单个字符串
    def convert_tokens_to_string(
        self,
        tokens: List[str],  # 输入标记列表
        group_tokens: bool = True,  # 是否将相同标记组合成CTC解码中不重复的标记
        spaces_between_special_tokens: bool = False,  # 特殊标记之间是否加空格
        filter_word_delimiter_token: bool = True,  # 是否过滤单词分隔标记
        output_char_offsets: bool = False,  # 是否输出字符偏移信息
    ) -> str:
        
        if group_tokens:  # 如果需要组合标记
            # 使用groupby函数将相同标记进行分组，并统计每组中标记的个数
            chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
        else:
            chars = tokens  # 否则直接使用输入的标记列表
            char_repetitions = len(tokens) * [1]  # 所有标记的个数为1

        processed_chars = list(filter(lambda char: char != self.pad_token, chars))  # 过滤掉标记列表中的 pad_token

        if filter_word_delimiter_token and self.word_delimiter_token is not None:  # 如果需要过滤单词分隔标记且单词分隔标记不为空
            processed_chars = list(filter(lambda token: token != self.word_delimiter_token, processed_chars))  # 过滤掉标记列表中的 word_delimiter_token

        char_offsets = None  # 初始化字符偏移为空
        if output_char_offsets:  # 如果需要输出字符偏移信息
            # 根据条件计算字符偏移
            word_delimiter_token_for_offsets = (
                self.word_delimiter_token if filter_word_delimiter_token is True else None
            )
            char_offsets = self._compute_offsets(
                char_repetitions, chars, self.pad_token, word_delimiter_token=word_delimiter_token_for_offsets
            )

            if len(char_offsets) != len(processed_chars):  # 如果字符偏移与处理后的标记列表长度不一致
                raise ValueError(
                    f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                    " have to be of the same length, but are: `len(offsets)`: "
                    f"{len(char_offsets)} and `len(processed_tokens)`: {len(processed_chars)}"
                )

            # 设置字符偏移的标记为正确的处理标记
            for i, char in enumerate(processed_chars):
                char_offsets[i]["char"] = char

        string = " ".join(processed_chars).strip()  # 将处理后的标记列表用空格连接成字符串

        return {"text": string, "char_offsets": char_offsets}  # 返回字符串和字符偏移信息的字典

    @staticmethod
    def _compute_offsets(
        char_repetitions: List[int],  # 标记重复次数列表
        chars: List[str],  # 标记列表
        ctc_token: int,  # CTC标记
        word_delimiter_token: Optional[int] = None  # 可选的单词分隔标记
``` 
    ) -> List[Dict[str, Union[str, int]]]:
        # 计算字符重复次数的累积和
        end_indices = np.asarray(char_repetitions).cumsum()
        # 计算起始索引
        start_indices = np.concatenate(([0], end_indices[:-1]))

        # 为每个字符创建包含起始和结束偏移量的字典
        offsets = [
            {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
        ]

        # 过滤掉 CTC token
        offsets = list(filter(lambda offsets: offsets["char"] != ctc_token, offsets))

        # 如果需要的话，过滤掉单词分隔符 token
        if word_delimiter_token is not None:
            offsets = list(filter(lambda offsets: offsets["char"] != word_delimiter_token, offsets))

        return offsets

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        group_tokens: bool = True,
        filter_word_delimiter_token: bool = True,
        spaces_between_special_tokens: bool = False,
        output_char_offsets: bool = False,
    ) -> str:
        """
        特殊的 _decode 函数用于 Wav2Vec2PhonemeTokenizer，因为添加的 token 应该和基础词汇表的 token 一样对待，
        所以必须在整个 token 列表上调用 `convert_tokens_to_string` 函数，而不是分别处理添加的 token
        """
        # 将 token_ids 转换为 tokens
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            result.append(token)

        # 将 tokens 转换为字符串输出
        string_output = self.convert_tokens_to_string(
            result,
            group_tokens=group_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            filter_word_delimiter_token=filter_word_delimiter_token,
            output_char_offsets=output_char_offsets,
        )

        text = string_output["text"]

        # 如果需要清除 tokenization 空格
        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)

        # 如果需要输出字符偏移
        if output_char_offsets:
            return Wav2Vec2PhonemeCTCTokenizerOutput(text=text, char_offsets=string_output["char_offsets"])
        else:
            return text

    # 从 `tokenization_utils_base.py` 覆盖���因为我们需要这里的 `output_char_offsets` 文档
    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        output_char_offsets: bool = False,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces.
            output_char_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output character offsets. Character offsets can be used in combination with the
                sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.

                <Tip>

                Please take a look at the Example of [`~models.wav2vec2.tokenization_wav2vec2.decode`] to better
                understand how to make use of `output_word_offsets`.
                [`~model.wav2vec2_phoneme.tokenization_wav2vec2_phoneme.batch_decode`] works the same way with
                phonemes.

                </Tip>

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str` or [`~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`]: The decoded
            sentence. Will be a [`~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`]
            when `output_char_offsets == True`.
        """
        # Convert inputs to python lists
        token_ids = to_py_obj(token_ids)

        # Call the private _decode method with specified arguments
        return self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            output_char_offsets=output_char_offsets,
            **kwargs,
        )

    # overwritten from `tokenization_utils_base.py` because tokenizer can output
    # `ModelOutput` which should not be a list for batched output and because
    # we need docs for `output_char_offsets` here
    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        output_char_offsets: bool = False,
        **kwargs,
    def batch_decode(sequences: Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor],
                     skip_special_tokens: bool = False,
                     clean_up_tokenization_spaces: bool = True,
                     output_char_offsets: bool = False,
                     **kwargs):
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

                Please take a look at the Example of [`~models.wav2vec2.tokenization_wav2vec2.decode`] to better
                understand how to make use of `output_word_offsets`.
                [`~model.wav2vec2_phoneme.tokenization_wav2vec2_phoneme.batch_decode`] works analogous with phonemes
                and batched output.

                </Tip>

            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `List[str]` or [`~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`]: The
            decoded sentence. Will be a
            [`~models.wav2vec2.tokenization_wav2vec2_phoneme.Wav2Vec2PhonemeCTCTokenizerOutput`] when
            `output_char_offsets == True`.
        """
        batch_decoded = [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                output_char_offsets=output_char_offsets,
                **kwargs,
            )
            for seq in sequences
        ]
        if output_char_offsets:
            # transform list of dicts to dict of lists
            return Wav2Vec2PhonemeCTCTokenizerOutput({k: [d[k] for d in batch_decoded] for k in batch_decoded[0]})

        return batch_decoded

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            # write encoder dictionary to a JSON file
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        return (vocab_file,)
```