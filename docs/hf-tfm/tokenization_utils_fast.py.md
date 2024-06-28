# `.\tokenization_utils_fast.py`

```py
# 设置脚本的字符编码为 UTF-8
# 版权声明：2020年由 HuggingFace Inc. 团队提供
#
# 根据 Apache 许可证版本 2.0（“许可证”）授权使用此文件；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，不提供任何明示或
# 暗示的担保或条件。
# 有关详细信息，请参阅许可证。
"""
 Tokenization classes for fast tokenizers (provided by HuggingFace's tokenizers library). For slow (python) tokenizers
 see tokenization_utils.py
"""

# 导入必要的库和模块
import copy
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 fast tokenizers 相关模块和类
import tokenizers.pre_tokenizers as pre_tokenizers_fast
from tokenizers import Encoding as EncodingFast
from tokenizers import Tokenizer as TokenizerFast
from tokenizers.decoders import Decoder as DecoderFast
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer

# 导入其他模块和类
from .convert_slow_tokenizer import convert_slow_tokenizer
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
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
from .utils import PaddingStrategy, add_end_docstrings, logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义文件名常量
# fast tokenizers 可以保存在单个文件中
TOKENIZER_FILE = "tokenizer.json"
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

# slow tokenizers 需要额外的添加 tokens 文件
ADDED_TOKENS_FILE = "added_tokens.json"

# 更新 INIT_TOKENIZER_DOCSTRING 文档字符串，增加关于 tokenizer_object 和 tokenizer_file 的说明
INIT_TOKENIZER_DOCSTRING += """
        tokenizer_object ([`tokenizers.Tokenizer`]):
            A [`tokenizers.Tokenizer`] object from 🤗 tokenizers to instantiate from. See [Using tokenizers from 🤗
            tokenizers](../fast_tokenizers) for more information.
        tokenizer_file ([`str`]):
            A path to a local JSON file representing a previously serialized [`tokenizers.Tokenizer`] object from 🤗
            tokenizers.
"""

# 映射模型类型到对应的 Trainer 类
MODEL_TO_TRAINER_MAPPING = {
    "BPE": BpeTrainer,
    "Unigram": UnigramTrainer,
    "WordLevel": WordLevelTrainer,
    "WordPiece": WordPieceTrainer,
}

# 定义 VOCAB_FILES_NAMES 字典，指定了 tokenizer_file 的文件名
VOCAB_FILES_NAMES = {"tokenizer_file": TOKENIZER_FILE}


# 使用装饰器将 INIT_TOKENIZER_DOCSTRING 添加到类 PreTrainedTokenizerFast 上，并继承自 PreTrainedTokenizerBase
@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class PreTrainedTokenizerFast(PreTrainedTokenizerBase):
    """
    Base class for all fast tokenizers (wrapping HuggingFace tokenizers library).

    Inherits from [`~tokenization_utils_base.PreTrainedTokenizerBase`].
    """
    """
    Handles all the shared methods for tokenization and special tokens, as well as methods for
    downloading/caching/loading pretrained tokenizers, as well as adding tokens to the vocabulary.

    This class also contains the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """

    # 定义一个类属性，用于存储词汇表文件的名称
    vocab_files_names = VOCAB_FILES_NAMES
    # 慢速分词器类的类型提示，默认为 None
    slow_tokenizer_class: PreTrainedTokenizer = None

    @property
    def is_fast(self) -> bool:
        # 返回一个布尔值，指示是否使用了快速分词器
        return True

    @property
    def can_save_slow_tokenizer(self) -> bool:
        """
        `bool`: Whether or not the slow tokenizer can be saved. Usually for sentencepiece based slow tokenizer, this
        can only be `True` if the original `"sentencepiece.model"` was not deleted.
        """
        # 返回一个布尔值，指示是否可以保存慢速分词器
        return True

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        # 返回基本词汇表的大小（不包括添加的特殊标记）
        return self._tokenizer.get_vocab_size(with_added_tokens=False)

    def get_vocab(self) -> Dict[str, int]:
        # 返回包括添加的特殊标记在内的词汇表
        return self._tokenizer.get_vocab(with_added_tokens=True)

    @property
    def vocab(self) -> Dict[str, int]:
        # 返回包括添加的特殊标记在内的词汇表
        return self.get_vocab()

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        """
        Returns the sorted mapping from string to index. The added tokens encoder is cached for performance
        optimisation in `self._added_tokens_encoder` for the slow tokenizers.
        """
        # 返回从字符串到索引的排序映射，用于添加的特殊标记编码器
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        """
        Returns the added tokens in the vocabulary as a dictionary of index to AddedToken.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        # 返回词汇表中添加的特殊标记，作为索引到 AddedToken 对象的字典
        return self._tokenizer.get_added_tokens_decoder()

    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        # 返回词汇表中添加的特殊标记，作为 token 到索引的字典
        return {k.content: v for v, k in sorted(self.added_tokens_decoder.items(), key=lambda item: item[0])}

    def __len__(self) -> int:
        """
        Size of the full vocabulary with the added tokens.
        """
        # 返回包括添加的特殊标记在内的词汇表的大小
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def backend_tokenizer(self) -> TokenizerFast:
        """
        `tokenizers.implementations.BaseTokenizer`: The Rust tokenizer used as a backend.
        """
        # 返回作为后端使用的 Rust 分词器对象
        return self._tokenizer

    @property
    def decoder(self) -> DecoderFast:
        """
        `tokenizers.decoders.Decoder`: The Rust decoder for this tokenizer.
        """
        # 返回用于此分词器的 Rust 解码器对象
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
        # Determine if `return_token_type_ids` should be inferred based on model input names
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        # Determine if `return_attention_mask` should be inferred based on model input names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        # Initialize `encodings` with current encoding or handle overflowing tokens
        if return_overflowing_tokens and encoding.overflowing is not None:
            encodings = [encoding] + encoding.overflowing
        else:
            encodings = [encoding]

        # Initialize a defaultdict to collect various encoding attributes as lists
        encoding_dict = defaultdict(list)
        # Iterate over each encoding in `encodings`
        for e in encodings:
            # Append token ids to the `input_ids` list in `encoding_dict`
            encoding_dict["input_ids"].append(e.ids)

            # Append token type ids if `return_token_type_ids` is enabled
            if return_token_type_ids:
                encoding_dict["token_type_ids"].append(e.type_ids)
            # Append attention mask if `return_attention_mask` is enabled
            if return_attention_mask:
                encoding_dict["attention_mask"].append(e.attention_mask)
            # Append special tokens mask if `return_special_tokens_mask` is enabled
            if return_special_tokens_mask:
                encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
            # Append offset mappings if `return_offsets_mapping` is enabled
            if return_offsets_mapping:
                encoding_dict["offset_mapping"].append(e.offsets)
            # Append length of token ids if `return_length` is enabled
            if return_length:
                encoding_dict["length"].append(len(e.ids))

        # Return the collected encoding attributes as `encoding_dict` and the list of `encodings`
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
        # If `tokens` is None, return None
        if tokens is None:
            return None

        # If `tokens` is a string, convert it to token id using `_convert_token_to_id_with_added_voc`
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        # If `tokens` is a list of strings, convert each token to token ids using `_convert_token_to_id_with_added_voc`
        return [self._convert_token_to_id_with_added_voc(token) for token in tokens]

    def _convert_token_to_id_with_added_voc(self, token: str) -> int:
        # Get the token id from `_tokenizer`, or return `unk_token_id` if token is unknown
        index = self._tokenizer.token_to_id(token)
        if index is None:
            return self.unk_token_id
        return index
    # 根据给定的索引将其转换为对应的标记（字符串）
    def _convert_id_to_token(self, index: int) -> Optional[str]:
        return self._tokenizer.id_to_token(int(index))

    # 向分词器添加新的标记（单词或特殊标记）
    def _add_tokens(self, new_tokens: List[Union[str, AddedToken]], special_tokens=False) -> int:
        if special_tokens:
            # 添加特殊标记到分词器
            return self._tokenizer.add_special_tokens(new_tokens)
        else:
            # 添加普通标记到分词器
            return self._tokenizer.add_tokens(new_tokens)

    # 返回编码序列时添加的特殊标记数量
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        返回在编码序列时添加的特殊标记数量。

        <Tip>

        这会对虚拟输入进行编码并检查添加的标记数量，因此效率较低。不要将此函数放在训练循环中。

        </Tip>

        Args:
            pair (`bool`, *optional*, 默认为 `False`):
                是否在序列对（sequence pair）情况下计算添加的特殊标记数量，或单独序列的情况。

        Returns:
            `int`: 添加到序列中的特殊标记数量。
        """
        return self._tokenizer.num_special_tokens_to_add(pair)

    # 将给定的标记索引或索引列表转换为对应的标记或标记列表
    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        使用词汇表和已添加的标记，将单个索引或索引序列转换为标记或标记序列。

        Args:
            ids (`int` 或 `List[int]`):
                要转换为标记或标记序列的标记 ID（或标记 IDs）。
            skip_special_tokens (`bool`, *optional*, 默认为 `False`):
                是否在解码时跳过特殊标记。

        Returns:
            `str` 或 `List[str]`: 解码后的标记（或标记列表）。
        """
        if isinstance(ids, int):
            return self._tokenizer.id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._tokenizer.id_to_token(index))
        return tokens

    # 对文本进行分词处理，返回标记列表
    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        return self.encode_plus(text=text, text_pair=pair, add_special_tokens=add_special_tokens, **kwargs).tokens()

    # 设置截断和填充策略，以及相关的参数
    def set_truncation_and_padding(
        self,
        padding_strategy: PaddingStrategy,
        truncation_strategy: TruncationStrategy,
        max_length: int,
        stride: int,
        pad_to_multiple_of: Optional[int],
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
        # Preserve the current truncation and padding settings of the tokenizer
        _truncation = self._tokenizer.truncation
        _padding = self._tokenizer.padding

        # Set truncation strategy on the backend tokenizer
        if truncation_strategy == TruncationStrategy.DO_NOT_TRUNCATE:
            # If DO_NOT_TRUNCATE is specified, ensure no truncation is applied
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

            # Compare current truncation settings with the target settings
            if _truncation is None:
                current = None
            else:
                current = {k: _truncation.get(k, None) for k in target}

            # Enable truncation if current settings differ from the target settings
            if current != target:
                self._tokenizer.enable_truncation(**target)

        # Set padding strategy on the backend tokenizer
        if padding_strategy == PaddingStrategy.DO_NOT_PAD:
            # If DO_NOT_PAD is specified, ensure no padding is applied
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

            # Compare current padding settings with the target settings
            if _padding != target:
                self._tokenizer.enable_padding(**target)
    # 定义一个方法用于批量编码文本或文本对
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput], List[TextInputPair], List[PreTokenizedInput], List[PreTokenizedInputPair]
        ],
        add_special_tokens: bool = True,  # 是否添加特殊的标记符号，默认为True
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 最大长度限制，默认为无限制
        stride: int = 0,  # 步长，默认为0
        is_split_into_words: bool = False,  # 输入是否已分成单词，默认为False
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的倍数，默认为不填充到倍数
        return_tensors: Optional[str] = None,  # 返回的张量类型，默认为None
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型ID，默认为None
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码，默认为None
        return_overflowing_tokens: bool = False,  # 是否返回溢出的token，默认为False
        return_special_tokens_mask: bool = False,  # 是否返回特殊token的掩码，默认为False
        return_offsets_mapping: bool = False,  # 是否返回偏移映射，默认为False
        return_length: bool = False,  # 是否返回长度，默认为False
        verbose: bool = True,  # 是否详细输出信息，默认为True
    ):
    
    # 定义一个方法用于编码单个文本或文本对
    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 输入的文本或预分词的文本
        text_pair: Optional[Union[TextInput, PreTokenizedInput]] = None,  # 可选的文本对
        add_special_tokens: bool = True,  # 是否添加特殊的标记符号，默认为True
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 最大长度限制，默认为无限制
        stride: int = 0,  # 步长，默认为0
        is_split_into_words: bool = False,  # 输入是否已分成单词，默认为False
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的倍数，默认为不填充到倍数
        return_tensors: Optional[bool] = None,  # 返回的张量类型，默认为None
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型ID，默认为None
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码，默认为None
        return_overflowing_tokens: bool = False,  # 是否返回溢出的token，默认为False
        return_special_tokens_mask: bool = False,  # 是否返回特殊token的掩码，默认为False
        return_offsets_mapping: bool = False,  # 是否返回偏移映射，默认为False
        return_length: bool = False,  # 是否返回长度，默认为False
        verbose: bool = True,  # 是否详细输出信息，默认为True
        **kwargs,  # 其他关键字参数，用于扩展功能
    ):
    ) -> BatchEncoding:
        # 将输入文本和可能存在的文本对作为一个批次输入，根据需要包装成元组
        batched_input = [(text, text_pair)] if text_pair else [text]
        # 调用内部方法进行批量编码处理，生成批次输出
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

        # 如果没有返回张量并且没有返回溢出的token，则移除前导的批次轴
        # 如果溢出的token作为一批输出返回，则在此情况下保留它们
        if return_tensors is None and not return_overflowing_tokens:
            # 重新处理批次输出，确保每个值正确处理为单个元素或列表的形式
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        # 检查并警告序列长度是否超过设定的最大长度
        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

        # 返回处理后的批次输出
        return batched_output

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # 使用后端的tokenizer decoder将token列表转换为字符串
        return self.backend_tokenizer.decoder.decode(tokens)

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        # 检查是否需要使用源tokenizer进行解码
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        # 将token_ids转换为列表形式（如果输入为单个整数）
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        # 使用内部的tokenizer解码token_ids，根据需要跳过特殊token
        text = self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        # 检查是否需要清理token化空间
        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        # 如果需要清理token化空间，则执行清理操作并返回清理后的文本
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            # 否则直接返回解码后的文本
            return text

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
        # 将保存目录转换为字符串
        save_directory = str(save_directory)

        # 如果没有定义慢速分词器的类且需要遗留格式，则引发值错误
        if self.slow_tokenizer_class is None and legacy_format is True:
            raise ValueError(
                "Your tokenizer does not have a legacy version defined and therefore cannot register this version. You"
                " might consider leaving the legacy_format at `None` or setting it to `False`."
            )

        # 决定是否保存慢速分词器
        save_slow = (
            (legacy_format is None or legacy_format is True)
            and self.slow_tokenizer_class is not None
            and self.can_save_slow_tokenizer
        )
        # 决定是否保存快速分词器
        save_fast = legacy_format is None or legacy_format is False

        # 如果需要保存慢速分词器
        if save_slow:
            # 构造添加的标记文件路径
            added_tokens_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE
            )
            # 确保对未来兼容
            added_vocab = {tok: index for tok, index in self.added_tokens_encoder.items() if index >= self.vocab_size}
            # 如果有添加的词汇，写入JSON文件
            if added_vocab:
                with open(added_tokens_file, "w", encoding="utf-8") as f:
                    out_str = json.dumps(added_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                    f.write(out_str)

            # 保存词汇表文件并获取文件名列表
            vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)
            file_names = file_names + vocab_files + (added_tokens_file,)

        # 如果需要保存快速分词器
        if save_fast:
            # 构造分词器文件路径
            tokenizer_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_FILE
            )
            # 调用后端分词器的保存方法
            self.backend_tokenizer.save(tokenizer_file)
            file_names = file_names + (tokenizer_file,)

        # 返回所有保存的文件名列表
        return file_names
```