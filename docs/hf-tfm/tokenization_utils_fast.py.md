# `.\transformers\tokenization_utils_fast.py`

```
# 定义了一组常用的编码字符集为 utf-8
# 版权声明，指明代码版权所有者和许可协议
"""
Tokenization classes for fast tokenizers (provided by HuggingFace's tokenizers library). For slow (python) tokenizers
see tokenization_utils.py
"""
# 引入必要的库和模块
import copy  # 引入用于深拷贝对象的模块
import json  # 引入用于处理 JSON 数据的模块
import os  # 引入用于处理操作系统相关功能的模块
from collections import defaultdict  # 引入用于创建默认字典的模块
from typing import Any, Dict, List, Optional, Tuple, Union  # 引入类型提示相关的模块

import tokenizers.pre_tokenizers as pre_tokenizers_fast  # 引入 HuggingFace tokenizers 库中的预分词器模块
from tokenizers import Encoding as EncodingFast  # 引入 HuggingFace tokenizers 库中的编码类
from tokenizers import Tokenizer as TokenizerFast  # 引入 HuggingFace tokenizers 库中的分词器类
from tokenizers.decoders import Decoder as DecoderFast  # 引入 HuggingFace tokenizers 库中的解码器类
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer  # 引入 HuggingFace tokenizers 库中的训练器类

from .convert_slow_tokenizer import convert_slow_tokenizer  # 从当前目录下的 convert_slow_tokenizer 模块中引入慢速分词器转换函数
from .tokenization_utils import PreTrainedTokenizer  # 从当前目录下的 tokenization_utils 模块中引入预训练分词器类
from .tokenization_utils_base import (  # 从当前目录下的 tokenization_utils_base 模块中引入一系列基础分词器相关类和函数
    INIT_TOKENIZER_DOCSTRING,
    AddedToken,
    BatchEncoding,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from .utils import PaddingStrategy, add_end_docstrings, logging  # 从当前目录下的 utils 模块中引入填充策略、添加文档末尾的字符串函数和日志记录功能

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 快速分词器（由 HuggingFace tokenizers 库提供）可以保存在单个文件中
TOKENIZER_FILE = "tokenizer.json"  # 快速分词器模型文件名
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"  # 特殊标记映射文件名
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"  # 快速分词器配置文件名

# 慢速分词器有一个额外的添加标记文件
ADDED_TOKENS_FILE = "added_tokens.json"  # 添加标记文件名

INIT_TOKENIZER_DOCSTRING += """
        tokenizer_object ([`tokenizers.Tokenizer`]):
            A [`tokenizers.Tokenizer`] object from 🤗 tokenizers to instantiate from. See [Using tokenizers from 🤗
            tokenizers](../fast_tokenizers) for more information.
        tokenizer_file ([`str`]):
            A path to a local JSON file representing a previously serialized [`tokenizers.Tokenizer`] object from 🤗
            tokenizers.
"""

MODEL_TO_TRAINER_MAPPING = {
    "BPE": BpeTrainer,
    "Unigram": UnigramTrainer,
    "WordLevel": WordLevelTrainer,
    "WordPiece": WordPieceTrainer,
}

VOCAB_FILES_NAMES = {"tokenizer_file": TOKENIZER_FILE}


@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
# 快速预训练分词器类，继承自基础预训练分词器基类
class PreTrainedTokenizerFast(PreTrainedTokenizerBase):
    """
    Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Inherits from [`~tokenization_utils_base.PreTrainedTokenizerBase`].
    """
    Handles all the shared methods for tokenization and special tokens, as well as methods for
    downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

    This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """

    # 定义类属性，存储词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 慢速分词器类，默认为 None
    slow_tokenizer_class: PreTrainedTokenizer = None

    @property
    def is_fast(self) -> bool:
        # 返回 True，表示使用快速分词器
        return True

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """
        `bool`: Whether or not the slow tokenizer can be saved. Usually for sentencepiece based slow tokenizer, this
        can only be `True` if the original `"sentencepiece.model"` was not deleted.
        """
        # 返回 True，表示慢速分词器可以保存
        return True

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        # 返回基础词汇表的大小（不包括添加的特殊标记）
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def get_vocab(self) -> Dict[str, int]:
        # 获取词汇表，包括添加的特殊标记
        return self._tokenizer.get_vocab(with_added_tokens=True)

    @property
    def vocab(self) -> Dict[str, int]:
        # 返回词汇表
        return self.get_vocab()

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        """
        Returns the sorted mapping from string to index. The added tokens encoder is cached for performance
        optimisation in `self._added_tokens_encoder` for the slow tokenizers.
        """
        # 返回从字符串到索引的排序映射，用于缓存性能优化
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        """
        Returns the added tokens in the vocabulary as a dictionary of index to AddedToken.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        # 返回词汇表中添加的特殊标记，以索引到 AddedToken 的字典形式
        return self._tokenizer.get_added_tokens_decoder()

    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        # 返回词汇表中添加的特殊标记，以标记到索引的字典形式
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        # 返回包括添加的特殊标记在内的完整词汇表大小
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def backend_tokenizer(self) -> TokenizerFast:
        """
        `tokenizers.implementations.BaseTokenizer`: The Rust tokenizer used as a backend.
        """
        # 返回用作后端的 Rust 分词器
        return self._tokenizer

    @property
    def decoder(self) -> DecoderFast:
        """
        `tokenizers.decoders.Decoder`: The Rust decoder for this tokenizer.
        """
        # 返回此分词器的 Rust 解码器
        return self._tokenizer.decoder
    def _convert_encoding(
        self,
        encoding: EncodingFast,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], List[EncodingFast]]:
        """
        Convert the encoding representation (from low-level HuggingFace tokenizer output) to a python Dict and a list
        of encodings, take care of building a batch from overflowing tokens.

        Overflowing tokens are converted to additional examples (like batches) so the output values of the dict are
        lists (overflows) of lists (tokens).

        Output shape: (overflows, sequence length)
        """
        # 指定是否返回 token_type_ids，默认为 None
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        # 指定是否返回 attention_mask，默认为 None
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        # 如果需要返回 overflowing_tokens 并且 encoding 中有 overflowing tokens，则将其加入 encodings 列表
        if return_overflowing_tokens and encoding.overflowing is not None:
            encodings = [encoding] + encoding.overflowing
        else:
            encodings = [encoding]

        # 创建一个 defaultdict 用于存储编码结果
        encoding_dict = defaultdict(list)
        # 遍历 encodings 列表
        for e in encodings:
            encoding_dict["input_ids"].append(e.ids)

            # 如果需要返回 token_type_ids，则添加到 encoding_dict 中
            if return_token_type_ids:
                encoding_dict["token_type_ids"].append(e.type_ids)
            # 如果需要返回 attention_mask，则添加到 encoding_dict 中
            if return_attention_mask:
                encoding_dict["attention_mask"].append(e.attention_mask)
            # 如果需要返回 special_tokens_mask，则添加到 encoding_dict 中
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
            # 如果需要返回 offsets_mapping，则添加到 encoding_dict 中
            if return_offsets_mapping:
                encoding_dict["offset_mapping"].append(e.offsets)
            # 如果需要返回 length，则添加到 encoding_dict 中
            if return_length:
                encoding_dict["length"].append(len(e.ids))

        # 返回编码结果字典和 encodings 列表
        return encoding_dict, encodings

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        # 如果 tokens 为 None，则返回 None
        if tokens is None:
            return None

        # 如果 tokens 是字符串，则调用 _convert_token_to_id_with_added_voc 方法转换为 token id
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        # 如果 tokens 是列表，则遍历列表中的每个 token，并调用 _convert_token_to_id_with_added_voc 方法转换为 token id
        return [self._convert_token_to_id_with_added_voc(token) for token in tokens]

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        # 使用 tokenizer 将 token 转换为对应的 id
        index = self._tokenizer.token_to_id(token)
        # 如果返回的 index 为 None，则返回 unk_token_id
        if index is None:
            return self.unk_token_id
        # 否则返回对应的 index
        return index
    # 将索引转换为对应的标记字符串，并返回结果
    def _convert_id_to_token(self, index: int) -> Optional[str]:
        return self._tokenizer.id_to_token(int(index))

    # 添加新标记到标记器中
    def _add_tokens(self, new_tokens: List[Union[str, AddedToken]], special_tokens=False) -> int:
        # 如果需要添加特殊标记，则调用标记器的添加特殊标记方法
        if special_tokens:
            return self._tokenizer.add_special_tokens(new_tokens)
        # 否则调用标记器的添加标记方法
        return self._tokenizer.add_tokens(new_tokens)

    # 计算使用特殊标记编码序列时添加的特殊标记数目
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        <Tip>

        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
        this inside your training loop.

        </Tip>

        Args:
            pair (`bool`, *optional*, defaults to `False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            `int`: Number of special tokens added to sequences.
        """
        # 返回使用特殊标记编码序列时添加的特殊标记数目
        return self._tokenizer.num_special_tokens_to_add(pair)

    # 将标记索引或索引列表转换为对应的标记或标记列表
    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        # 如果输入是单个索引，则将其转换为对应的标记并返回
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids)
        # 如果输入是索引列表，则遍历列表，将每个索引转换为对应的标记
        tokens = []
        for index in ids:
            index = int(index)
            # 如果跳过特殊标记并且当前索引是特殊标记之一，则跳过当前索引
            if skip_special_tokens and index in self.all_special_ids:
                continue
            # 将当前索引转换为对应的标记，并添加到标记列表中
            tokens.append(self._tokenizer.id_to_token(index))
        # 返回标记列表
        return tokens

    # 将文本标记化为标记列表
    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        # 调用encode_plus方法对文本进行标记化，并返回标记列表
        return self.encode_plus(text=text, text_pair=pair, add_special_tokens=add_special_tokens, **kwargs).tokens()

    # 设置截断和填充策略
    def set_truncation_and_padding(
        self,
        padding_strategy: PaddingStrategy,
        truncation_strategy: TruncationStrategy,
        max_length: int,
        stride: int,
        pad_to_multiple_of: Optional[int],
    ):
        """
        Define the truncation and the padding strategies for fast tokenizers (provided by HuggingFace tokenizers
        library) and restore the tokenizer settings afterwards.

        The provided tokenizer has no padding / truncation strategy before the managed section. If your tokenizer set a
        padding / truncation strategy before, then it will be reset to no padding / truncation when exiting the managed
        section.

        Args:
            padding_strategy ([`~utils.PaddingStrategy`]):
                The kind of padding that will be applied to the input
            truncation_strategy ([`~tokenization_utils_base.TruncationStrategy`]):
                The kind of truncation that will be applied to the input
            max_length (`int`):
                The maximum size of a sequence.
            stride (`int`):
                The stride to use when handling overflow.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
        """
        # Store the current truncation and padding settings
        _truncation = self._tokenizer.truncation
        _padding = self._tokenizer.padding
        # Set truncation and padding on the backend tokenizer
        if truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE:
            # If the truncation strategy is set to 'do not truncate', and there was a previous truncation strategy,
            # reset it to no truncation
            if _truncation is not None:
                self._tokenizer.no_truncation()
        else:
            # Define the target truncation settings
            target = {
                "max_length": max_length,
                "stride": stride,
                "strategy": truncation_strategy.value,
                "direction": self.truncation_side,
            }

            # Check if the current truncation settings match the target settings
            if _truncation is None:
                current = None
            else:
                current = {k: _truncation.get(k, None) for k in target}

            # If current truncation settings don't match target settings, enable truncation with target settings
            if current != target:
                self._tokenizer.enable_truncation(**target)

        if padding_strategy == PaddingStrategy.DO_NOT_PAD:
            # If the padding strategy is set to 'do not pad', and there was a previous padding strategy,
            # reset it to no padding
            if _padding is not None:
                self._tokenizer.no_padding()
        else:
            # Define the target padding settings
            length = max_length if padding_strategy == PaddingStrategy.MAX_LENGTH else None
            target = {
                "length": length,
                "direction": self.padding_side,
                "pad_id": self.pad_token_id,
                "pad_token": self.pad_token,
                "pad_type_id": self.pad_token_type_id,
                "pad_to_multiple_of": pad_to_multiple_of,
            }
            # If current padding settings don't match target settings, enable padding with target settings
            if _padding != target:
                self._tokenizer.enable_padding(**target)
    # 定义一个方法用于批量编码文本或文本对
    def _batch_encode_plus(
        self,
        # 输入参数为文本列表、文本对列表、预分词输入列表或预分词输入对列表
        batch_text_or_text_pairs: Union[
            List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]
        ],
        # 是否添加特殊标记，默认为True
        add_special_tokens: bool = True,
        # 填充策略，默认为不填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 截断策略，默认为不截断
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        # 最大长度限制，默认为None
        max_length: Optional[int] = None,
        # 步长，默认为0
        stride: int = 0,
        # 是否已分词，默认为False
        is_split_into_words: bool = False,
        # 填充到指定长度的倍数，默认为None
        pad_to_multiple_of: Optional[int] = None,
        # 返回张量，默认为None
        return_tensors: Optional[str] = None,
        # 返回token类型ID，默认为None
        return_token_type_ids: Optional[bool] = None,
        # 返回注意力掩码，默认为None
        return_attention_mask: Optional[bool] = None,
        # 返回溢出的token，默认为False
        return_overflowing_tokens: bool = False,
        # 返回特殊标记掩码，默认为False
        return_special_tokens_mask: bool = False,
        # 返回偏移映射，默认为False
        return_offsets_mapping: bool = False,
        # 返回长度，默认为False
        return_length: bool = False,
        # 是否详细输出，默认为True
        verbose: bool = True,
    # 定义一个方法用于编码文本或文本对
    def _encode_plus(
        self,
        # 输入文本，可以是文本或预分词输入
        text: Union[TextInput, PreTokenizedInput],
        # 第二个文本，可选，可以是文本或预分词输入
        text_pair: Optional[Union[TextInput, PreTokenizedInput]] = None,
        # 是否添加特殊标记，默认为True
        add_special_tokens: bool = True,
        # 填充策略，默认为不填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 截断策略，默认为不截断
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        # 最大长度限制，默认为None
        max_length: Optional[int] = None,
        # 步长，默认为0
        stride: int = 0,
        # 是否已分词，默认为False
        is_split_into_words: bool = False,
        # 填充到指定长度的倍数，默认为None
        pad_to_multiple_of: Optional[int] = None,
        # 返回张量，默认为None
        return_tensors: Optional[bool] = None,
        # 返回token类型ID，默认为None
        return_token_type_ids: Optional[bool] = None,
        # 返回注意力掩码，默认为None
        return_attention_mask: Optional[bool] = None,
        # 返回溢出的token，默认为False
        return_overflowing_tokens: bool = False,
        # 返回特殊标记掩码，默认为False
        return_special_tokens_mask: bool = False,
        # 返回偏移映射，默认为False
        return_offsets_mapping: bool = False,
        # 返回长度，默认为False
        return_length: bool = False,
        # 是否详细输出，默认为True
        verbose: bool = True,
        # 其他关键字参数
        **kwargs,
    # 定义一个方法，用于将输入文本转换为批量编码
    def __call__(
        self,
        text: Union[str, List[str], List[int]],
        text_pair: Optional[Union[str, List[str], List[int]]] = None,
        is_split_into_words: bool = False,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: Optional[bool] = None,
        return_special_tokens_mask: Optional[bool] = None,
        return_offsets_mapping: Optional[bool] = None,
        return_length: Optional[bool] = None,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        # 将输入文本转换为批量输入
        batched_input = [(text, text_pair)] if text_pair else [text]
        # 调用内部方法进行批量编码
        batched_output = self._batch_encode_plus(
            batched_input,
            is_split_into_words=is_split_into_words,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

        # 如果返回的张量为None，则移除批量维度
        # 如果溢出的标记作为输出的批量返回，则在这种情况下保留它们
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        # 检查是否需要警告关于序列过长
        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

        # 返回��量编码结果
        return batched_output

    # 将标记转换为字符串
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.backend_tokenizer.decoder.decode(tokens)

    # 解码方法
    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        # 检查是否使用源标记器
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        # 如果token_ids是整数，则转换为列表
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        # 解码标记
        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        # 清理标记化空格
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

    # 保存预训练模型
    def _save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        file_names: Tuple[str],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        """
        Save a tokenizer using the slow-tokenizer/legacy format: vocabulary + added tokens as well as in a unique JSON
        file containing {config + vocab + added-tokens}.
        """
        # 将保存目录转换为字符串类型
        save_directory = str(save_directory)

        # 如果慢速分词器类为None且legacy_format为True，则引发值错误
        if self.slow_tokenizer_class is None and legacy_format is True:
            raise ValueError(
                "Your tokenizer does not have a legacy version defined and therefore cannot register this version. You"
                " might consider leaving the legacy_format at `None` or setting it to `False`."
            )

        # 判断是否保存慢速分词器
        save_slow = (
            (legacy_format is None or legacy_format is True)
            and self.slow_tokenizer_class is not None
            and self.can_save_slow_tokenizer
        )
        # 判断是否保存快速分词器
        save_fast = legacy_format is None or legacy_format is False

        # 如果需要保存慢速分词器
        if save_slow:
            # 构建添加的标记文件路径
            added_tokens_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE
            )
            # 确保向前兼容
            added_vocab = {tok: index for tok, index in self.added_tokens_encoder.items() if index >= self.vocab_size}
            # 如果存在添加的词汇
            if added_vocab:
                with open(added_tokens_file, "w", encoding="utf-8") as f:
                    out_str = json.dumps(added_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                    f.write(out_str)

            # 保存词汇文件
            vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)
            file_names = file_names + vocab_files + (added_tokens_file,)

        # 如果需要保存快速分词器
        if save_fast:
            # 构建分词器文件路径
            tokenizer_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_FILE
            )
            # 保存分词器
            self.backend_tokenizer.save(tokenizer_file)
            file_names = file_names + (tokenizer_file,)

        # 返回文件名列表
        return file_names

    def train_new_from_iterator(
        self,
        text_iterator,
        vocab_size,
        length=None,
        new_special_tokens=None,
        special_tokens_map=None,
        **kwargs,
```