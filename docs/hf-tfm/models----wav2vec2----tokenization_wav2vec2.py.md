# `.\transformers\models\wav2vec2\tokenization_wav2vec2.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 版本 2.0 许可协议，授予权利
# 在遵守协议的情况下可以使用此文件
# 您可以在以下网址获取许可协议的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则本软件
# 根据"原样"分发，不附带任何担保或条件，
# 无论是明示的还是暗示的。请参阅特定语言的许可证
# 限制与权限的明确说明。

"""Wav2Vec2 的分词类。"""

# 导入所需的包和模块
import json
import os
import sys
import warnings
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

# 导入 Hugging Face 库中的相关模块和函数
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

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 如果正在进行类型检查，则根据不同的框架导入相关的模块
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

# 定义预训练词汇文件的映射
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

# 定义预训练位置嵌入的尺寸
# Wav2Vec2 没有最大输入长度限制
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/wav2vec2-base-960h": sys.maxsize}
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

ListOfDict = List[Dict[str, Union[int, str]]]

# 定义一个数据类，表示Wav2Vec2CTCTokenizer的输出
@dataclass
class Wav2Vec2CTCTokenizerOutput(ModelOutput):
    """
    Output type of [` Wav2Vec2CTCTokenizer`], with transcription.

    Args:
        text (list of `str` or `str`):
            Decoded logits in text from. Usually the speech transcription.
        char_offsets (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]`):
            Offsets of the decoded characters. In combination with sampling rate and model downsampling rate char
            offsets can be used to compute time stamps for each charater. Total logit score of the beam associated with
            produced text.
        word_offsets (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]`):
            Offsets of the decoded words. In combination with sampling rate and model downsampling rate word offsets
            can be used to compute time stamps for each word.
    """

    text: Union[List[str], str]
    # 定义一个变量 char_offsets，类型为 Union[List[ListOfDict], ListOfDict]，初始值为 None
    char_offsets: Union[List[ListOfDict], ListOfDict] = None
    # 定义一个变量 word_offsets，类型为 Union[List[ListOfDict], ListOfDict]，初始值为 None
    word_offsets: Union[List[ListOfDict], ListOfDict] = None
class Wav2Vec2CTCTokenizer(PreTrainedTokenizer):
    """
    构造一个 Wav2Vec2CTC 分词器。

    这个分词器继承自 [`PreTrainedTokenizer`]，包含了一些主要方法。用户应该参考超类获得更多关于这些方法的信息。

    参数:
        vocab_file (`str`):
            包含词汇表的文件。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            句子开头的标记。
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            句子结尾的标记。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知标记。词汇表中没有的标记无法转换为 ID，将被设置为此标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            用于填充的标记，例如在批处理不同长度的序列时使用。
        word_delimiter_token (`str`, *optional*, defaults to `"|"`):
            用于定义单词结尾的标记。
        do_lower_case (`bool`, *optional*, defaults to `False`):
            是否接受小写输入并在解码时将输出转换为小写。
        target_lang (`str`, *optional*):
            分词器应默认设置的目标语言。对于多语言、嵌套词汇表，`target_lang` 必须为 [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) 这样的词汇表定义。

        **kwargs
            传递给 [`PreTrainedTokenizer`] 的额外关键字参数
    """

    vocab_files_names = VOCAB_FILES_NAMES  # 词汇表文件名
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练词汇表文件映射
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 最大模型输入尺寸
    model_input_names = ["input_ids", "attention_mask"]  # 模型输入名

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
        self._word_delimiter_token = word_delimiter_token  # 将传入的 word_delimiter_token 值赋给对象的 _word_delimiter_token 属性

        self.do_lower_case = do_lower_case  # 将传入的 do_lower_case 值赋给对象的 do_lower_case 属性
        self.replace_word_delimiter_char = replace_word_delimiter_char  # 将传入的 replace_word_delimiter_char 值赋给对象的 replace_word_delimiter_char 属性
        self.target_lang = target_lang  # 将传入的 target_lang 值赋给对象的 target_lang 属性

        with open(vocab_file, encoding="utf-8") as vocab_handle:  # 打开指定的 vocab_file 文件，并使用 utf-8 编码
            self.vocab = json.load(vocab_handle)  # 读取 vocab_handle 文件的内容并赋值给对象的 vocab 属性
        
        # 如果指定了 target lang，则 vocab 必须是一个嵌套字典，每个 target lang 对应一个词汇表
        if target_lang is not None:
            self.encoder = self.vocab[target_lang]  # 将 vocab 中指定 target_lang 的值赋给对象的 encoder 属性
        else:
            self.encoder = self.vocab  # 将 vocab 的值赋给对象的 encoder 属性

        self.decoder = {v: k for k, v in self.encoder.items()}  # 通过遍历 encoder 中的项生成 key 和 value 互换的字典

        super().__init__(  # 调用父类的构造方法
            unk_token=unk_token,  # 传入 unk_token 参数
            bos_token=bos_token,  # 传入 bos_token 参数
            eos_token=eos_token,  # 传入 eos_token 参数
            pad_token=pad_token,  # 传入 pad_token 参数
            do_lower_case=do_lower_case,  # 传入 do_lower_case 参数
            word_delimiter_token=word_delimiter_token,  # 传入 word_delimiter_token 参数
            replace_word_delimiter_char=replace_word_delimiter_char,  # 传入 replace_word_delimiter_char 参数
            target_lang=target_lang,  # 传入 target_lang 参数
            **kwargs,  # 传入其他参数
        )

        # 确保由多个字符组成的标记在分词时不会被分割
        for token in self.encoder.keys():  # 遍历 encoder 中的键
            if len(token) > 1:  # 如果键的长度大于 1
                self.add_tokens(AddedToken(token, rstrip=True, lstrip=True, normalized=False))  # 添加指定的 token

    def set_target_lang(self, target_lang: str):  # 声明一个名为 set_target_lang 的方法
        """
        Set the target language of a nested multi-lingual dictionary
        """
        if self.vocab == self.encoder:  # 如果 vocab 等于 encoder
            raise ValueError(f"{self.vocab} is not a multi-lingual, nested tokenizer. Cannot set target language.")  # 抛出 ValueError 异常

        if target_lang not in self.vocab:  # 如果指定的 target_lang 不在 vocab 中
            raise ValueError(f"{target_lang} does not exist. Choose one of {', '.join(self.vocab.keys())}.")  # 抛出 ValueError 异常

        self.target_lang = target_lang  # 将指定的 target_lang 值赋给对象的 target_lang 属性
        self.init_kwargs["target_lang"] = target_lang  # 将指定的 target_lang 值添加到 init_kwargs 字典中
        self.encoder = self.vocab[target_lang]  # 将 vocab 中指定 target_lang 的值赋给对象的 encoder 属性
        self.decoder = {v: k for k, v in self.encoder.items()}  # 通过遍历 encoder 中的项生成 key 和 value 互换的字典

        # 确保由多个字符组成的标记在分词时不会被分割
        for token in self.encoder.keys():  # 遍历 encoder 中的键
            if len(token) > 1:  # 如果键的长度大于 1
                self.add_tokens(AddedToken(token, rstrip=True, lstrip=True, normalized=False))  # 添加指定的 token

    @property
    def word_delimiter_token(self) -> str:  # 声明一个名为 word_delimiter_token 的属性方法，返回 str 类型的值
        """
        `str`: Word delimiter token. Log an error if used while not having been set.
        """
        if self._word_delimiter_token is None and self.verbose:  # 如果 _word_delimiter_token 是 None 且 verbose 是 True
            logger.error("Using word_delimiter_token, but it is not set yet.")  # 记录错误日志
            return None  # 返回 None
        return str(self._word_delimiter_token)  # 返回 _word_delimiter_token 的字符串表示形式

    @property
    def word_delimiter_token_id(self) -> Optional[int]:  # 声明一个名为 word_delimiter_token_id 的属性方法，返回 Optional[int] 类型的值
        """
        `Optional[int]`: Id of the word_delimiter_token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self._word_delimiter_token is None:  # 如果 _word_delimiter_token 是 None
            return None  # 返回 None
        return self.convert_tokens_to_ids(self.word_delimiter_token)  # 调用 convert_tokens_to_ids 方法将 word_delimiter_token 转换为 ID，并返回
    # 设置 word_delimiter_token 属性的 setter 方法
    @word_delimiter_token.setter
    def word_delimiter_token(self, value):
        # 将输入的 value 值赋给 _word_delimiter_token 属性
        self._word_delimiter_token = value
    
    # 设置 word_delimiter_token_id 属性的 setter 方法    
    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value):
        # 将输入的 value 值转换为 token ID 后赋给 _word_delimiter_token 属性
        self._word_delimiter_token = self.convert_tokens_to_ids(value)
    
    # 获取 vocab_size 属性的 getter 方法
    @property
    def vocab_size(self) -> int:
        # 返回词典中的词汇数量
        return len(self.decoder)
    
    # 获取完整词典的方法
    def get_vocab(self) -> Dict:
        # 创建一个新的词典对象
        vocab = dict(self.encoder)
        # 将 added_tokens_encoder 添加到新词典中
        vocab.update(self.added_tokens_encoder)
        # 返回完整词典
        return vocab
    
    # 添加新的 tokens 的方法，重写以确保不进行删除操作
    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        # 创建一个新的 token 列表
        to_add = []
        # 遍历输入的新 token
        for token in new_tokens:
            # 如果 token 是字符串
            if isinstance(token, str):
                # 创建一个 AddedToken 对象，并设置不删除任何字符
                to_add.append(AddedToken(token, rstrip=False, lstrip=False, normalized=False))
            # 如果 token 已经是 AddedToken 对象
            else:
                # 直接添加到列表
                to_add.append(token)
        # 调用父类的 _add_tokens 方法添加 token，并返回添加的数量
        return super()._add_tokens(to_add, special_tokens)
    
    # 文本分词方法
    def _tokenize(self, text, **kwargs):
        """
        Converts a string into a sequence of tokens (string), using the tokenizer.
        """
        # 如果需要转换为小写
        if self.do_lower_case:
            # 转换为大写
            text = text.upper()
        # 将文本中的空格替换为 word_delimiter_token，并转换为列表
        return list(text.replace(" ", self.word_delimiter_token))
    
    # 将 token 转换为 ID 的方法    
    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        # 尝试在词典中查找 token 对应的 ID，找不到则返回 unk_token 的 ID
        return self.encoder.get(token, self.encoder.get(self.unk_token))
    
    # 将 ID 转换为 token 的方法
    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        # 尝试在词典中查找 ID 对应的 token，找不到则返回 unk_token
        result = self.decoder.get(index, self.unk_token)
        return result
    
    # 将 token 列表转换为字符串的方法
    def convert_tokens_to_string(
        self,
        tokens: List[str],
        group_tokens: bool = True,
        spaces_between_special_tokens: bool = False,
        output_char_offsets: bool = False,
        output_word_offsets: bool = False,
    ):
    def _compute_offsets(
        char_repetitions: List[int], chars: List[str], ctc_token: int
    ) -> List[Dict[str, Union[str, int]]]:
        # 计算字符偏移量
        end_indices = np.asarray(char_repetitions).cumsum()  # 计算每个字符重复次数
        start_indices = np.concatenate(([0], end_indices[:-1]))  # 计算每个字符的起始位置
    
        # 生成偏移量字典列表
        offsets = [
            {"char": t, "start_offset": s, "end_offset": e} for t, s, e in zip(chars, start_indices, end_indices)
        ]
    
        # 过滤掉CTC令牌
        offsets = list(filter(lambda offsets: offsets["char"] != ctc_token, offsets))
        return offsets
    # 获取单词偏移量的方法，将偏移量字典转换为单词偏移量列表
    def _get_word_offsets(
        offsets: Dict[str, Union[str, float]], word_delimiter_char: str = " "
    ) -> Dict[str, Union[str, float]]:
        # 存储单词偏移量的列表
        word_offsets = []

        # 上一个状态，初始为“SPACE”
        last_state = "SPACE"
        # 当前单词
        word = ""
        # 单词起始偏移量
        start_offset = 0
        # 单词结束偏移量
        end_offset = 0
        # 遍历偏移量字典
        for i, offset in enumerate(offsets):
            # 当前字符
            char = offset["char"]
            # 当前状态：如果字符为分隔符，则状态为“SPACE”，否则为“WORD”
            state = "SPACE" if char == word_delimiter_char else "WORD"

            # 如果当前状态与上一个状态相同
            if state == last_state:
                # 如果仍在同一状态，则将字符追加到当前单词中，并更新结束偏移量
                end_offset = offset["end_offset"]
                word += char
            else:
                # 状态转换
                if state == "SPACE":
                    # 完成一个单词的识别，将单词及其偏移量添加到列表中
                    word_offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})
                else:
                    # 开始识别一个新的单词，更新起始偏移量和结束偏移量，并重置当前单词
                    start_offset = offset["start_offset"]
                    end_offset = offset["end_offset"]
                    word = char

            # 更新上一个状态为当前状态
            last_state = state
        # 处理最后一个单词，如果上一个状态为“WORD”，则将其添加到单词偏移量列表中
        if last_state == "WORD":
            word_offsets.append({"word": word, "start_offset": start_offset, "end_offset": end_offset})

        # 返回单词偏移量列表
        return word_offsets

    # 为了进行标记化准备文本的方法，可选地在文本前添加空格以确保文本的正确标记化
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # 如果文本已经拆分成单词，则在文本前添加空格
        if is_split_into_words:
            text = " " + text
        # 返回处理后的文本及其他参数
        return (text, kwargs)

    # 解码方法，将标记化的标识符转换为原始文本
    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        group_tokens: bool = True,
        spaces_between_special_tokens: bool = False,
        output_word_offsets: Optional[bool] = False,
        output_char_offsets: Optional[bool] = False,
    # 定义函数的返回类型为字符串
    ) -> str:
        """
        # 为了处理Wav2Vec2Tokenizer中的特殊_decode函数，添加的标记应该与基本词汇的标记处理方式完全一致，
        # 因此必须在整个标记列表上调用`convert_tokens_to_string`函数，而不是单独处理添加的标记
        """
        # 将标记 IDs 转换为标记（tokens），跳过特殊标记
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        # 遍历过滤后的标记列表
        for token in filtered_tokens:
            # 如果跳过特殊标记且标记在所有特殊标记中
            if skip_special_tokens and token in self.all_special_ids:
                continue
            # 将标记添加到结果列表中
            result.append(token)

        # 将标记列表转换为字符串输出
        string_output = self.convert_tokens_to_string(
            result,
            group_tokens=group_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            output_word_offsets=output_word_offsets,
            output_char_offsets=output_char_offsets,
        )

        # 获取文本内容
        text = string_output["text"]

        # 如果需要清理标记化空格，则进行标记化处理
        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            text = self.clean_up_tokenization(text)

        # 如果需要输出单词偏移或字符偏移，则返回Wav2Vec2CTCTokenizerOutput对象
        if output_word_offsets or output_char_offsets:
            return Wav2Vec2CTCTokenizerOutput(
                text=text,
                char_offsets=string_output["char_offsets"],
                word_offsets=string_output["word_offsets"],
            )
        else:
            # 否则返回文本内容
            return text

    # 从`tokenization_utils_base.py`中覆盖，因为分词器可以输出`ModelOutput`，
    # 对于批量输出不应该是一个列表，同时我们需要在这里提供`output_char_offsets`的文档
    def batch_decode(
        sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        output_char_offsets: bool = False,
        output_word_offsets: bool = False,
        **kwargs,
    def batch_decode(sequences: List[Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]]) -> List[str]:
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
        # Decode each sequence in the given list based on the specified arguments
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
        
        # Check if output_char_offsets or output_word_offsets is True
        if output_char_offsets or output_word_offsets:
            # Transform list of dicts to dict of lists and return
            return Wav2Vec2CTCTokenizerOutput({k: [d[k] for d in batch_decoded] for k in batch_decoded[0]})

        # Return the list of decoded sequences
        return batch_decoded
    # 从 `tokenization_utils_base.py` 覆盖的方法，因为我们需要在这里为 `output_char_offsets` 和 `output_word_offsets` 提供文档
    # 解码方法，将标记 ID 转换为文本
    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        output_char_offsets: bool = False,
        output_word_offsets: bool = False,
        **kwargs,
    
    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 拼接词汇表文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 写入词汇表到文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 返回词汇表文件路径
        return (vocab_file,)
class Wav2Vec2Tokenizer(PreTrainedTokenizer):
    """
    构建一个 Wav2Vec2 分词器。

    该分词器继承自 [`PreTrainedTokenizer`]，其中包含一些主要方法。用户应参考超类以获取有关这些方法的更多信息。

    Args:
        vocab_file (`str`):
            包含词汇表的文件。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            句子开头标记。
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            句子结尾标记。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知标记。词汇表中不存在的标记不能转换为 ID，并被设置为此标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            用于填充的标记，例如在批处理不同长度的序列时。
        word_delimiter_token (`str`, *optional*, defaults to `"|"`):
            用于定义单词结束的标记。
        do_lower_case (`bool`, *optional*, defaults to `False`):
            解码输出时是否将其转换为小写。
        do_normalize (`bool`, *optional*, defaults to `False`):
            是否对输入进行零均值单位方差归一化。归一化可以显著提高某些模型的性能，例如 [wav2vec2-lv60](https://huggingface.co/models?search=lv60)。
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            [`~Wav2Vec2Tokenizer.__call__`] 是否应返回 `attention_mask`。

            <Tip>

            设置了 `config.feat_extract_norm == "group"` 的 Wav2Vec2 模型，例如 [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h)，**未**使用 `attention_mask` 进行训练。对于这样的模型，`input_values` 应该简单地用 0 填充，不应传递 `attention_mask`。

            对于设置了 `config.feat_extract_norm == "layer"` 的 Wav2Vec2 模型，例如 [wav2vec2-lv60](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self)，应该传递 `attention_mask` 进行批量推理。

            </Tip>

        **kwargs
            传递给 [`PreTrainedTokenizer`] 的其他关键字参数
    """

    vocab_files_names = VOCAB_FILES_NAMES
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
    model_input_names = ["input_values", "attention_mask"]
    # 初始化方法，接受参数，设置默认参数值
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=False,
        do_normalize=False,
        return_attention_mask=False,
        **kwargs,
    ):
        # 发出警告，提示即将被移除
        warnings.warn(
            "The class `Wav2Vec2Tokenizer` is deprecated and will be removed in version 5 of Transformers. Please use"
            " `Wav2Vec2Processor` or `Wav2Vec2CTCTokenizer` instead.",
            FutureWarning,
        )

        # 设置词分隔符
        self._word_delimiter_token = word_delimiter_token

        # 设置参数值
        self.do_lower_case = do_lower_case
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize

        # 从词汇文件中加载编码器
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)

        # 反转编码器的键值对，用于解码
        self.decoder = {v: k for k, v in self.encoder.items()}

        # 调用父类的初始化方法，设置参数
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

    # 词分隔符属性的访问方法
    @property
    def word_delimiter_token(self) -> str:
        """
        `str`: Padding token. Log an error if used while not having been set.
        """
        # 如果词分隔符为空并且 verbose 标志为真，则记录错误并返回空
        if self._word_delimiter_token is None and self.verbose:
            logger.error("Using word_delimiter_token, but it is not set yet.")
            return None
        return str(self._word_delimiter_token)

    # 词分隔符 ID 属性的访问方法
    @property
    def word_delimiter_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the word_delimiter_token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        # 如果词分隔符为空，则返回空，否则返回其在词汇表中的 ID
        if self._word_delimiter_token is None:
            return None
        return self.convert_tokens_to_ids(self.word_delimiter_token)

    # 设置词分隔符属性的方法
    @word_delimiter_token.setter
    def word_delimiter_token(self, value):
        self._word_delimiter_token = value

    # 设置词分隔符 ID 属性的方法
    @word_delimiter_token_id.setter
    def word_delimiter_token_id(self, value):
        self._word_delimiter_token = self.convert_tokens_to_ids(value)

    # 调用装饰器函数添加文档字符串
    @add_end_docstrings(WAV2VEC2_KWARGS_DOCSTRING)
    # 调用实例对象时执行的方法
    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
        **kwargs,
    def prepare_inputs(
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]]
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

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        # make sure input is in list format
        if is_batched and not isinstance(raw_speech[0], np.ndarray):
            raw_speech = [np.asarray(speech) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        # zero-mean and unit-variance normalization
        if self.do_normalize:
            raw_speech = [(x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5) for x in raw_speech]

        # convert into correct format for padding
        encoded_inputs = BatchEncoding({"input_values": raw_speech})

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
        return len(self.decoder)

    def get_vocab(self) -> Dict:
        return dict(self.encoder, **self.added_tokens_encoder)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        result = self.decoder.get(index, self.unk_token)
        return result
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
        """
        # 将连接时序分类（CTC）输出的标记转换为单个字符串
        # 将相同标记组合成CTC风格解码中的非重复标记
        grouped_tokens = [token_group[0] for token_group in groupby(tokens)]

        # 过滤掉self.pad_token，它用作CTC空白标记
        filtered_tokens = list(filter(lambda token: token != self.pad_token, grouped_tokens))

        # 替换分隔符标记
        string = "".join([" " if token == self.word_delimiter_token else token for token in filtered_tokens]).strip()

        if self.do_lower_case:
            # 如果需要转换为小写，则进行小写处理
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
        special _decode function is needed for Wav2Vec2Tokenizer because added tokens should be treated exactly the
        same as tokens of the base vocabulary and therefore the function `convert_tokens_to_string` has to be called on
        the whole token list and not individually on added tokens
        """
        # 为Wav2Vec2Tokenizer需要特殊的_decode函数，因为添加的标记应该与基础词汇表的标记完全相同，
        # 因此必须在整个标记列表上调用`convert_tokens_to_string`函数，而不是分别在添加的标记上调用
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            result.append(token)

        # 将标记列表转换为字符串
        text = self.convert_tokens_to_string(result)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            # 如果需要清理标记化空格，则进行清理
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            # 如果保存目录不是目录，则记录错误并返回
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建词汇表文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            # 将词典编码写入词汇表文件
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        return (vocab_file,)
```