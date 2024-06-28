# `.\models\wav2vec2_phoneme\tokenization_wav2vec2_phoneme.py`

```
# 设置文件编码为UTF-8

# 版权声明，版权归Facebook Inc.和HuggingFace Inc.团队所有

# 引入必要的库和模块
import json  # 导入处理JSON格式的模块
import os  # 导入操作系统功能的模块
import sys  # 导入系统相关的模块
from dataclasses import dataclass  # 导入用于定义数据类的装饰器
from itertools import groupby  # 导入用于迭代操作的工具函数
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union  # 导入类型提示相关的工具函数

import numpy as np  # 导入处理数值数组的库

# 导入HuggingFace库中的相关工具和类
from ...tokenization_utils import PreTrainedTokenizer  # 导入预训练分词器的基类
from ...tokenization_utils_base import AddedToken  # 导入添加的特殊标记类
from ...utils import (  # 导入HuggingFace库中的一些实用工具函数和类
    ModelOutput,  # 导入模型输出的基类
    is_flax_available,  # 判断是否可以使用Flax库
    is_tf_available,  # 判断是否可以使用TensorFlow库
    is_torch_available,  # 判断是否可以使用PyTorch库
    logging,  # 日志记录工具
    requires_backends,  # 判断所需的后端库是否可用
    to_py_obj,  # 将对象转换为Python对象
)

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 如果类型检查开启，则根据当前可用的深度学习框架导入相应的库
if TYPE_CHECKING:
    if is_torch_available():
        import torch  # 导入PyTorch库
    if is_tf_available():
        import tensorflow as tf  # 导入TensorFlow库
    if is_flax_available():
        import jax.numpy as jnp  # 导入Flax库中的NumPy模块（忽略Flax库导入的警告）

# 定义词汇文件和分词器配置文件的名称映射
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",  # 词汇表文件名
    "tokenizer_config_file": "tokenizer_config.json",  # 分词器配置文件名
}

# 预训练模型的词汇文件映射，包括对应的下载链接
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

# 预训练模型的位置编码嵌入大小映射，这里给出了一个特定模型的最大输入长度
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/wav2vec2-lv-60-espeak-cv-ft": sys.maxsize}

# 定义一种数据类型，表示列表中包含字典的结构
ListOfDict = List[Dict[str, Union[int, str]]]

@dataclass
class Wav2Vec2PhonemeCTCTokenizerOutput(ModelOutput):
    """
    [`Wav2Vec2PhonemeCTCTokenizer`]的输出类型，带有音素。

    Args:
        text (list of `str` or `str`):
            解码的文本，通常是语音转录。
        char_offsets (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]]`):
            解码字符的偏移量。结合采样率和模型下采样率，可以用来计算每个字符的时间戳。
    """
    text: Union[List[str], str]  # 文本内容，可以是字符串或字符串列表
    char_offsets: Union[List[ListOfDict], ListOfDict] = None  # 字符的偏移量，可以是列表的列表或列表


class Wav2Vec2PhonemeCTCTokenizer(PreTrainedTokenizer):
    """
    构造一个Wav2Vec2PhonemeCTC分词器。
    """
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
        do_phonemize (`bool`, *optional*, defaults to `True`):
            Whether the tokenizer should phonetize the input or not. Only if a sequence of phonemes is passed to the
            tokenizer, `do_phonemize` should be set to `False`.
        phonemizer_lang (`str`, *optional*, defaults to `"en-us"`):
            The language of the phoneme set to which the tokenizer should phonetize the input text to.
        phonemizer_backend (`str`, *optional*. defaults to `"espeak"`):
            The backend phonetization library that shall be used by the phonemizer library. Defaults to `espeak-ng`.
            See the [phonemizer package](https://github.com/bootphon/phonemizer#readme). for more information.

        **kwargs
            Additional keyword arguments passed along to [`PreTrainedTokenizer`]
    ):
        self._word_delimiter_token = word_delimiter_token  # 设置单词分隔符令牌
        self._phone_delimiter_token = phone_delimiter_token  # 设置电话分隔符令牌
        self.do_phonemize = do_phonemize  # 是否执行音素化操作的标志
        self.phonemizer_lang = phonemizer_lang  # 音素化使用的语言
        self.phonemizer_backend = phonemizer_backend  # 音素化使用的后端

        if do_phonemize:
            self.init_backend(self.phonemizer_lang)  # 若需执行音素化，则初始化音素化后端

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)  # 从文件中加载编码器映射
        self.decoder = {v: k for k, v in self.encoder.items()}  # 创建解码器映射，反转编码器的键值对

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
        )  # 调用父类的初始化方法，传入相关参数

    @property
    def vocab_size(self) -> int:
        return len(self.decoder)  # 返回解码器的大小作为词汇表大小

    def get_vocab(self) -> Dict:
        vocab = dict(self.encoder.copy())  # 复制编码器的内容作为词汇表
        vocab.update(self.added_tokens_encoder)  # 添加额外的编码器映射
        return vocab  # 返回完整的词汇表

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        # 覆盖方法以避免去除空格！
        to_add = []
        for token in new_tokens:
            if isinstance(token, str):
                to_add.append(AddedToken(token, rstrip=False, lstrip=False, normalized=True, special=special_tokens))
            else:
                to_add.append(token)

        return super()._add_tokens(to_add, special_tokens)  # 调用父类的添加令牌方法，添加新令牌

    def init_backend(self, phonemizer_lang: str):
        """
        Initializes the backend.

        Args:
            phonemizer_lang (`str`): The language to be used.
        """
        requires_backends(self, "phonemizer")  # 检查必要的后端
        from phonemizer.backend import BACKENDS

        self.backend = BACKENDS[self.phonemizer_backend](phonemizer_lang, language_switch="remove-flags")  # 初始化音素化后端

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
        # If `is_split_into_words` is True, prepend a space to `text`
        if is_split_into_words:
            text = " " + text

        # Set the instance variable `self.do_phonemize` based on the provided `do_phonemize`
        if do_phonemize is not None:
            self.do_phonemize = do_phonemize

        # Set the instance variable `self.phonemizer_lang` and initialize backend if `phonemizer_lang` is provided
        if phonemizer_lang is not None:
            self.phonemizer_lang = phonemizer_lang
            self.init_backend(phonemizer_lang)

        # Return the modified `text` and an empty dictionary (unused kwargs)
        return (text, {})

    def _tokenize(self, text, **kwargs):
        """
        Converts a string into a sequence of tokens (string), using the tokenizer.
        """

        # Remove leading and trailing whitespace from `text`
        text = text.strip()

        # Phonemize the `text` if `self.do_phonemize` is True
        if self.do_phonemize:
            # Convert `text` to lowercase
            text = text.lower()

            # Generate a list of phonemes for the `text` in `self.phonemizer_lang`
            text = self.phonemize(text, self.phonemizer_lang)

        # Split `text` into tokens using whitespace as delimiter
        tokens = text.split(" ")

        # Remove empty tokens from the list
        tokens = list(filter(lambda p: p.strip() != "", tokens))

        # Return the list of tokens
        return tokens
    def phonemize(self, text: str, phonemizer_lang: Optional[str] = None) -> str:
        # 导入分离器模块
        from phonemizer.separator import Separator
        
        # 如果设置了单词分隔符标记，并且不为 None，则加上空格
        word_delimiter = self.word_delimiter_token + " " if self.word_delimiter_token is not None else ""
        
        # 如果指定了语言且与当前使用的语言不同，则重新初始化后端
        if phonemizer_lang is not None and phonemizer_lang != self.phonemizer_lang:
            self.init_backend(phonemizer_lang)
        else:
            # 否则使用当前的语言设置
            phonemizer_lang = self.phonemizer_lang
        
        # 创建分隔符对象，用于指定音素之间的分隔符
        separator = Separator(phone=self.phone_delimiter_token, word=word_delimiter, syllable="")
        
        # 对输入的文本进行音素化处理，返回一个包含音素的列表，取第一个元素并去除两端空白
        phonemes = self.backend.phonemize(
            [text],
            separator=separator,
        )
        phonemes = phonemes[0].strip()

        return phonemes

    @property
    def word_delimiter_token(self) -> str:
        """
        `str`: 单词分隔符标记。如果在尚未设置时使用，则记录错误日志。
        """
        if self._word_delimiter_token is None:
            if self.verbose:
                logger.error("Using word_delimiter_token, but it is not set yet.")
            return None
        return str(self._word_delimiter_token)

    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: 单词分隔符标记在词汇表中的ID。如果尚未设置，则返回 `None`。
        """
        if self._word_delimiter_token is None:
            return None
        return self.convert_tokens_to_ids(self.word_delimiter_token)

    @word_delimiter_token.setter
    def word_delimiter_token(self, value):
        # 设置单词分隔符标记的值
        self._word_delimiter_token = value

    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value):
        # 根据给定的值将其转换为词汇表中的ID，并设置为单词分隔符标记
        self._word_delimiter_token = self.convert_tokens_to_ids(value)

    @property
    def phone_delimiter_token(self) -> str:
        """
        `str`: 音素分隔符标记。如果在尚未设置时使用，则记录错误日志。
        """
        if self._phone_delimiter_token is None:
            if self.verbose:
                logger.error("Using phone_delimiter_token, but it is not set yet.")
            return None
        return str(self._phone_delimiter_token)

    @property
    def phone_delimiter_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: 音素分隔符标记在词汇表中的ID。如果尚未设置，则返回 `None`。
        """
        if self._phone_delimiter_token is None:
            return None
        return self.convert_tokens_to_ids(self.phone_delimiter_token)

    @phone_delimiter_token.setter
    def phone_delimiter_token(self, value):
        # 设置音素分隔符标记的值
        self._phone_delimiter_token = value

    @phone_delimiter_token_id.setter
    def phone_delimiter_token_id(self, value):
        # 根据给定的值将其转换为词汇表中的ID，并设置为音素分隔符标记
        self._phone_delimiter_token = self.convert_tokens_to_ids(value)

    def _convert_token_to_id(self, token: str) -> int:
        """将给定的 token（字符串）转换为索引（整数），使用词汇表进行映射。"""
        return self.encoder.get(token, self.encoder.get(self.unk_token))
    # 将索引转换为标记字符串，使用词汇表进行解码
    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) into a token (str) using the vocabulary."""
        # 从解码器中获取索引对应的标记，如果不存在则使用未知标记（unk_token）
        result = self.decoder.get(index, self.unk_token)
        return result

    # 将连接主义时间分类（CTC）输出的标记列表转换为单个字符串
    def convert_tokens_to_string(
        self,
        tokens: List[str],
        group_tokens: bool = True,
        spaces_between_special_tokens: bool = False,
        filter_word_delimiter_token: bool = True,
        output_char_offsets: bool = False,
    ) -> str:
        """
        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
        """
        # 将相同的标记组合成非重复的标记，用于CTC风格的解码
        if group_tokens:
            # 使用itertools.groupby按标记分组，并记录每组的长度
            chars, char_repetitions = zip(*((token, len(list(group_iter))) for token, group_iter in groupby(tokens)))
        else:
            chars = tokens
            char_repetitions = len(tokens) * [1]

        # 过滤掉self.pad_token，这是用作CTC空白标记的特殊标记
        processed_chars = list(filter(lambda char: char != self.pad_token, chars))

        # 如果设置了过滤单词分隔符标记并且存在self.word_delimiter_token，则也过滤该标记
        if filter_word_delimiter_token and self.word_delimiter_token is not None:
            processed_chars = list(filter(lambda token: token != self.word_delimiter_token, processed_chars))

        # 如果需要输出字符偏移量，则计算偏移量
        char_offsets = None
        if output_char_offsets:
            # 计算字符偏移量，需要考虑CTC标记和单词分隔符标记
            word_delimiter_token_for_offsets = (
                self.word_delimiter_token if filter_word_delimiter_token is True else None
            )
            char_offsets = self._compute_offsets(
                char_repetitions, chars, self.pad_token, word_delimiter_token=word_delimiter_token_for_offsets
            )

            # 检查偏移量和处理后的标记列表长度是否一致
            if len(char_offsets) != len(processed_chars):
                raise ValueError(
                    f"`char_offsets`: {char_offsets} and `processed_tokens`: {processed_chars}"
                    " have to be of the same length, but are: `len(offsets)`: "
                    f"{len(char_offsets)} and `len(processed_tokens)`: {len(processed_chars)}"
                )

            # 将偏移量中的标记字段设置为正确的处理后的标记
            for i, char in enumerate(processed_chars):
                char_offsets[i]["char"] = char

        # 将处理后的标记列表连接成字符串，并去除首尾空格
        string = " ".join(processed_chars).strip()

        # 返回包含文本字符串和字符偏移量的字典
        return {"text": string, "char_offsets": char_offsets}

    # 计算字符偏移量的静态方法
    @staticmethod
    def _compute_offsets(
        char_repetitions: List[int], chars: List[str], ctc_token: int, word_delimiter_token: Optional[int] = None
    ):
    ) -> List[Dict[str, Union[str, int]]]:
        # 将字符重复次数数组转换为累积和数组，用于计算起始和结束索引
        end_indices = np.asarray(char_repetitions).cumsum()
        start_indices = np.concatenate(([0], end_indices[:-1]))

        # 根据字符、起始索引、结束索引创建偏移量字典列表
        offsets = [
            {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
        ]

        # 过滤掉 CTC 标记的偏移量
        offsets = list(filter(lambda offsets: offsets["char"] != ctc_token, offsets))

        # 如果需要，过滤掉单词分隔符标记的偏移量
        if word_delimiter_token is not None:
            offsets = list(filter(lambda offsets: offsets["char"] != word_delimiter_token, offsets))

        # 返回偏移量列表
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
        特殊的 _decode 函数用于 Wav2Vec2PhonemeTokenizer，因为添加的特殊标记应该与基础词汇表中的标记完全相同，
        因此必须在整个标记列表上调用 `convert_tokens_to_string` 函数，而不是单独处理添加的标记
        """
        # 将 token_ids 转换为 tokens 列表，跳过特殊标记（如果设置）
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        for token in filtered_tokens:
            # 如果设置了跳过特殊标记且 token 是特殊标记之一，则跳过该 token
            if skip_special_tokens and token in self.all_special_ids:
                continue
            result.append(token)

        # 将过滤后的 tokens 列表转换为字符串输出
        string_output = self.convert_tokens_to_string(
            result,
            group_tokens=group_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            filter_word_delimiter_token=filter_word_delimiter_token,
            output_char_offsets=output_char_offsets,
        )

        text = string_output["text"]

        # 如果需要清除标记化空格，则调用清除空格的函数
        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)

        # 如果需要输出字符偏移量，则返回带偏移量的特定类型的对象
        if output_char_offsets:
            return Wav2Vec2PhonemeCTCTokenizerOutput(text=text, char_offsets=string_output["char_offsets"])
        else:
            return text

    # 重写自 `tokenization_utils_base.py`，因为这里需要文档说明 `output_char_offsets`
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

        # Call the internal _decode method with specified parameters
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
    ):
        """
        Batch decodes sequences of token ids into strings or `ModelOutput` objects.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List or batch of tokenized input sequences.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces.
            output_char_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output character offsets.

        Returns:
            `List[str]` or `List[~transformers.file_utils.ModelOutput]`: List of decoded sentences or model outputs.
        """
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
        # Perform batch decoding using self.decode for each sequence in sequences
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
        # Check if output_char_offsets is True
        if output_char_offsets:
            # Transform list of dictionaries to a dictionary of lists
            return Wav2Vec2PhonemeCTCTokenizerOutput({k: [d[k] for d in batch_decoded] for k in batch_decoded[0]})

        # Return the batch_decoded list
        return batch_decoded

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Check if save_directory exists; log an error and return if not
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # Construct the full path for the vocabulary file
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # Write the vocabulary (self.encoder) to the vocab_file in JSON format
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # Return the tuple containing the vocab_file path
        return (vocab_file,)
```