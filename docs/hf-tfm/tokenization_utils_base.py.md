# `.\tokenization_utils_base.py`

```
# 设置文件编码为 UTF-8
# 版权声明为 HuggingFace Inc. 团队，版权年份为 2020 年
# 使用 Apache 许可证 2.0 版本，详细信息请访问指定网址获取
# 本代码库提供的代码受版权法保护，除非符合许可证规定，否则不得使用
"""
包含慢速和快速标记化类共有的基础类：
- PreTrainedTokenizerBase：包含所有用户界面的编码方法
- Special token mixing：包含特殊标记逻辑
- BatchEncoding：用于快速标记化器的输出字典包装，带有特殊方法
"""

import copy  # 导入复制函数
import json  # 导入 JSON 序列化和反序列化函数
import os  # 导入操作系统相关函数
import re  # 导入正则表达式模块
import warnings  # 导入警告处理模块
from collections import UserDict  # 导入用户定义字典类
from collections.abc import Mapping, Sized  # 导入映射和可计数集合抽象基类
from contextlib import contextmanager  # 导入上下文管理器
from dataclasses import dataclass  # 导入 dataclass 装饰器
from functools import lru_cache  # 导入 lru_cache 装饰器
from typing import (  # 导入类型提示
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np  # 导入 NumPy 库
from packaging import version  # 导入版本管理模块

from . import __version__  # 导入当前模块的版本信息
from .dynamic_module_utils import custom_object_save  # 导入自定义对象保存函数
from .utils import (  # 导入工具函数
    ExplicitEnum,
    PaddingStrategy,
    PushToHubMixin,
    TensorType,
    add_end_docstrings,
    add_model_info_to_auto_map,
    cached_file,
    copy_func,
    download_url,
    extract_commit_hash,
    is_flax_available,
    is_jax_tensor,
    is_mlx_available,
    is_numpy_array,
    is_offline_mode,
    is_remote_url,
    is_tf_available,
    is_tf_tensor,
    is_tokenizers_available,
    is_torch_available,
    is_torch_device,
    is_torch_tensor,
    logging,
    requires_backends,
    to_py_obj,
)

if TYPE_CHECKING:  # 检查是否在类型检查模式下运行
    if is_torch_available():  # 如果 Torch 可用
        import torch  # 导入 Torch 库
    if is_tf_available():  # 如果 TensorFlow 可用
        import tensorflow as tf  # 导入 TensorFlow 库
    if is_flax_available():  # 如果 Flax 可用
        import jax.numpy as jnp  # 导入 JAX 的 NumPy 接口，用于类型检查
        from .pipelines.conversational import Conversation  # 导入会话式对话模块

if is_tokenizers_available():  # 如果 Tokenizers 可用
    from tokenizers import AddedToken  # 导入 Tokenizers 的 AddedToken 类
    from tokenizers import Encoding as EncodingFast  # 导入 Tokenizers 的 Encoding 类作为 EncodingFast
else:  # 如果 Tokenizers 不可用
    @dataclass(frozen=False, eq=True)  # 定义一个 dataclass 装饰器
    # 定义一个名为 AddedToken 的类，表示要添加到 Tokenizer 的一个标记
    # AddedToken 可以具有特殊选项，定义其行为方式
    class AddedToken:
        """
        AddedToken represents a token to be added to a Tokenizer An AddedToken can have special options defining the
        way it should behave.

        The `normalized` will default to `not special` if it is not specified, similarly to the definition in
        `tokenizers`.
        """

        # 初始化方法，用于设置 AddedToken 的属性
        def __init__(
            self, content: str, single_word=False, lstrip=False, rstrip=False, special=False, normalized=None
        ):
            self.content = content  # 标记的内容
            self.single_word = single_word  # 是否是单词
            self.lstrip = lstrip  # 是否去除左侧空白
            self.rstrip = rstrip  # 是否去除右侧空白
            self.special = special  # 是否是特殊标记
            self.normalized = normalized if normalized is not None else not special  # 标记是否已标准化，默认与 special 相反

        # 返回对象的状态，用于序列化
        def __getstate__(self):
            return self.__dict__

        # 返回标记的内容
        def __str__(self):
            return self.content

    # 定义一个名为 EncodingFast 的数据类
    @dataclass
    class EncodingFast:
        """This is dummy class because without the `tokenizers` library we don't have these objects anyway"""
        
        pass  # 仅作为示例，因为没有 `tokenizers` 库，这些对象实际上并不存在
# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# 设置一个非常大的整数作为输入模型的最大长度，用于模型具有无限输入大小的情况
VERY_LARGE_INTEGER = int(1e30)
# 设置一个大的整数，稍微小于VERY_LARGE_INTEGER，用于需要大量但不是非常大的情况
LARGE_INTEGER = int(1e20)

# 定义类型别名和命名元组
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]

# 旧版慢速分词器保存在三个单独的文件中
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

# 快速分词器（由HuggingFace tokenizer库提供）可以保存在单个文件中
FULL_TOKENIZER_FILE = "tokenizer.json"
_re_tokenizer_file = re.compile(r"tokenizer\.(.*)\.json")

class TruncationStrategy(ExplicitEnum):
    """
    `PreTrainedTokenizerBase.__call__` 方法中 `truncation` 参数的可能取值。
    在IDE中进行选项补全时非常有用。
    """
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"

class CharSpan(NamedTuple):
    """
    原始字符串中的字符范围。

    Args:
        start (`int`): 原始字符串中第一个字符的索引。
        end (`int`): 原始字符串中最后一个字符后面的字符的索引。
    """
    start: int
    end: int

class TokenSpan(NamedTuple):
    """
    编码字符串（token列表）中的token范围。

    Args:
        start (`int`): 范围中第一个token的索引。
        end (`int`): 范围中最后一个token后面的token的索引。
    """
    start: int
    end: int

class BatchEncoding(UserDict):
    """
    [`~tokenization_utils_base.PreTrainedTokenizerBase.__call__`],
    [`~tokenization_utils_base.PreTrainedTokenizerBase.encode_plus`] 和
    [`~tokenization_utils_base.PreTrainedTokenizerBase.batch_encode_plus`] 方法的输出（tokens, attention_masks等）。

    这个类继承自Python字典类，可以像字典一样使用。此外，这个类还提供了从单词/字符空间到token空间的映射工具方法。
    """
    """
    Args:
        data (`dict`, *optional*):
            Dictionary of lists/arrays/tensors returned by the `__call__`/`encode_plus`/`batch_encode_plus` methods
            ('input_ids', 'attention_mask', etc.).
        encoding (`tokenizers.Encoding` or `Sequence[tokenizers.Encoding]`, *optional*):
            If the tokenizer is a fast tokenizer which outputs additional information like mapping from word/character
            space to token space the `tokenizers.Encoding` instance or list of instance (for batches) hold this
            information.
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
        prepend_batch_axis (`bool`, *optional*, defaults to `False`):
            Whether or not to add a batch axis when converting to tensors (see `tensor_type` above).
        n_sequences (`Optional[int]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """

    # 初始化方法，用于将输入数据转换为张量并存储相关编码信息
    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        encoding: Optional[Union[EncodingFast, Sequence[EncodingFast]]] = None,
        tensor_type: Union[None, str, TensorType] = None,
        prepend_batch_axis: bool = False,
        n_sequences: Optional[int] = None,
    ):
        # 调用父类初始化方法，传入数据字典
        super().__init__(data)

        # 如果 encoding 是 EncodingFast 类型，则转换为列表形式
        if isinstance(encoding, EncodingFast):
            encoding = [encoding]

        # 存储编码信息到实例变量 _encodings 中
        self._encodings = encoding

        # 如果 n_sequences 为 None，并且 encoding 不为 None 且非空列表，则从第一个编码对象获取 n_sequences
        if n_sequences is None and encoding is not None and len(encoding):
            n_sequences = encoding[0].n_sequences

        # 存储 n_sequences 到实例变量 _n_sequences 中
        self._n_sequences = n_sequences

        # 调用 convert_to_tensors 方法，将输入数据转换为张量（PyTorch/TensorFlow/Numpy），根据 tensor_type 和 prepend_batch_axis 参数进行处理
        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)

    @property
    def n_sequences(self) -> Optional[int]:
        """
        `Optional[int]`: The number of sequences used to generate each sample from the batch encoded in this
        [`BatchEncoding`]. Currently can be one of `None` (unknown), `1` (a single sentence) or `2` (a pair of
        sentences)
        """
        # 返回存储在 _n_sequences 实例变量中的序列数信息
        return self._n_sequences

    @property
    def is_fast(self) -> bool:
        """
        `bool`: Indicate whether this [`BatchEncoding`] was generated from the result of a [`PreTrainedTokenizerFast`]
        or not.
        """
        # 返回一个布尔值，指示 _encodings 实例变量是否为 None（即是否由 PreTrainedTokenizerFast 生成了该 BatchEncoding 实例）
        return self._encodings is not None
    def __getitem__(self, item: Union[int, str]) -> Union[Any, EncodingFast]:
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_ids', 'attention_mask',
        etc.).

        If the key is an integer, get the `tokenizers.Encoding` for batch item with index `key`.

        If the key is a slice, returns the value of the dict associated to `key` ('input_ids', 'attention_mask', etc.)
        with the constraint of slice.
        """
        # 如果 `item` 是字符串，则返回与该键关联的字典值（如 'input_ids'、'attention_mask' 等）
        if isinstance(item, str):
            return self.data[item]
        # 如果 `item` 是整数，并且 `_encodings` 不为 None，则返回索引为 `item` 的批次的 `tokenizers.Encoding`
        elif self._encodings is not None:
            return self._encodings[item]
        # 如果 `item` 是切片对象，则返回满足切片条件的字典值（如 'input_ids', 'attention_mask' 等）
        elif isinstance(item, slice):
            return {key: self.data[key][item] for key in self.data.keys()}
        # 如果 `item` 类型不符合上述三种情况，则引发 KeyError
        else:
            raise KeyError(
                "Invalid key. Only three types of key are available: "
                "(1) string, (2) integers for backend Encoding, and (3) slices for data subsetting."
            )

    def __getattr__(self, item: str):
        try:
            # 尝试从 `self.data` 中获取属性 `item` 的值
            return self.data[item]
        except KeyError:
            # 如果 `item` 不存在于 `self.data` 中，则引发 AttributeError
            raise AttributeError

    def __getstate__(self):
        # 返回对象的序列化状态，包括 `self.data` 和 `_encodings`
        return {"data": self.data, "encodings": self._encodings}

    def __setstate__(self, state):
        # 如果 `state` 中包含 `data`，则将其赋值给 `self.data`
        if "data" in state:
            self.data = state["data"]

        # 如果 `state` 中包含 `_encodings`，则将其赋值给 `self._encodings`
        if "encodings" in state:
            self._encodings = state["encodings"]

    def keys(self):
        # 返回 `self.data` 的键列表
        return self.data.keys()

    def values(self):
        # 返回 `self.data` 的值列表
        return self.data.values()

    def items(self):
        # 返回 `self.data` 的键值对列表
        return self.data.items()

    # After this point:
    # Extended properties and methods only available for fast (Rust-based) tokenizers
    # provided by HuggingFace tokenizers library.

    @property
    def encodings(self) -> Optional[List[EncodingFast]]:
        """
        `Optional[List[tokenizers.Encoding]]`: The list all encodings from the tokenization process. Returns `None` if
        the input was tokenized through Python (i.e., not a fast) tokenizer.
        """
        # 返回 `_encodings`，即快速（基于 Rust 的）分词器生成的编码列表；如果是通过 Python 进行分词（非快速分词器），则返回 None
        return self._encodings

    def tokens(self, batch_index: int = 0) -> List[str]:
        """
        Return the list of tokens (sub-parts of the input strings after word/subword splitting and before conversion to
        integer indices) at a given batch index (only works for the output of a fast tokenizer).

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[str]`: The list of tokens at that index.
        """
        # 如果 `_encodings` 为 None，则抛出 ValueError，说明不支持 `tokens()` 方法
        if not self._encodings:
            raise ValueError(
                "tokens() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        # 返回指定批次索引处的 token 列表
        return self._encodings[batch_index].tokens
    # 返回一个列表，将每个 token 映射到其原始句子的 id：
    # - 对于在序列周围或序列之间添加的特殊 token，映射为 `None`。
    # - 对于第一个序列中的单词对应的 token，映射为 `0`。
    # - 当对一对序列进行联合编码时，对于第二个序列中单词对应的 token，映射为 `1`。
    def sequence_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        """
        Return a list mapping the tokens to the id of their original sentences:

            - `None` for special tokens added around or between sequences,
            - `0` for tokens corresponding to words in the first sequence,
            - `1` for tokens corresponding to words in the second sequence when a pair of sequences was jointly
              encoded.

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[Optional[int]]`: A list indicating the sequence id corresponding to each token. Special tokens added
            by the tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding
            sequence.
        """
        # 如果没有 `_encodings` 属性，抛出 ValueError 异常，提示无法使用该方法
        if not self._encodings:
            raise ValueError(
                "sequence_ids() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        # 返回指定批次索引处的 `_encodings` 对象的 sequence_ids 属性
        return self._encodings[batch_index].sequence_ids

    # 返回一个列表，将每个 token 映射到其初始句子中的实际单词（仅适用于快速分词器）
    def words(self, batch_index: int = 0) -> List[Optional[int]]:
        """
        Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[Optional[int]]`: A list indicating the word corresponding to each token. Special tokens added by the
            tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding word
            (several tokens will be mapped to the same word index if they are parts of that word).
        """
        # 如果没有 `_encodings` 属性，抛出 ValueError 异常，提示无法使用该方法
        if not self._encodings:
            raise ValueError(
                "words() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        # 发出警告，提示 `words()` 方法已被废弃，建议使用 `word_ids()` 方法
        warnings.warn(
            "`BatchEncoding.words()` property is deprecated and should be replaced with the identical, "
            "but more self-explanatory `BatchEncoding.word_ids()` property.",
            FutureWarning,
        )
        # 返回调用 `word_ids()` 方法的结果，传入指定的批次索引
        return self.word_ids(batch_index)
    # 返回一个列表，将每个token映射到其在初始句子中的实际单词，适用于快速分词器。
    def word_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        """
        Return a list mapping the tokens to their actual word in the initial sentence for a fast tokenizer.

        Args:
            batch_index (`int`, *optional*, defaults to 0): The index to access in the batch.

        Returns:
            `List[Optional[int]]`: A list indicating the word corresponding to each token. Special tokens added by the
            tokenizer are mapped to `None` and other tokens are mapped to the index of their corresponding word
            (several tokens will be mapped to the same word index if they are parts of that word).
        """
        # 如果_encodings为空，则抛出值错误异常
        if not self._encodings:
            raise ValueError(
                "word_ids() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        # 返回指定批次索引的word_ids列表
        return self._encodings[batch_index].word_ids

    # 获取给定token所表示的序列的索引
    def token_to_sequence(self, batch_or_token_index: int, token_index: Optional[int] = None) -> int:
        """
        Get the index of the sequence represented by the given token. In the general use case, this method returns `0`
        for a single sequence or the first sequence of a pair, and `1` for the second sequence of a pair

        Can be called as:

        - `self.token_to_sequence(token_index)` if batch size is 1
        - `self.token_to_sequence(batch_index, token_index)` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
        words are defined by the user). In this case it allows to easily associate encoded tokens with provided
        tokenized words.

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token in the
                sequence.

        Returns:
            `int`: Index of the word in the input sequence.
        """

        # 如果_encodings为空，则抛出值错误异常
        if not self._encodings:
            raise ValueError("token_to_sequence() is not available when using Python based tokenizers")
        # 如果token_index不为None，则batch_index为batch_or_token_index
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        # 如果batch_index小于0，则将其转换为有效的索引
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        # 如果token_index小于0，则将其转换为有效的索引
        if token_index < 0:
            token_index = self._seq_len + token_index
        # 返回指定编码中指定token的序列索引
        return self._encodings[batch_index].token_to_sequence(token_index)
    def token_to_word(self, batch_or_token_index: int, token_index: Optional[int] = None) -> int:
        """
        Get the index of the word corresponding (i.e. comprising) to an encoded token in a sequence of the batch.

        Can be called as:

        - `self.token_to_word(token_index)` if batch size is 1
        - `self.token_to_word(batch_index, token_index)` if batch size is greater than 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e.,
        words are defined by the user). In this case it allows to easily associate encoded tokens with provided
        tokenized words.

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token in the
                sequence.

        Returns:
            `int`: Index of the word in the input sequence.
        """

        # 如果没有编码信息，则抛出错误，Python 基础的分词器不支持 token_to_word 方法
        if not self._encodings:
            raise ValueError("token_to_word() is not available when using Python based tokenizers")

        # 确定 batch_index 和 token_index 的值
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0  # 默认的 batch_index 如果只有一个序列
            token_index = batch_or_token_index

        # 处理负数的 batch_index 和 token_index
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if token_index < 0:
            token_index = self._seq_len + token_index

        # 调用具体编码对象的 token_to_word 方法，返回单词在输入序列中的索引
        return self._encodings[batch_index].token_to_word(token_index)

    def word_to_tokens(
        self, batch_or_word_index: int, word_index: Optional[int] = None, sequence_index: int = 0
    ) -> Optional[TokenSpan]:
        """
        Get the encoded token span corresponding to a word in a sequence of the batch.

        Token spans are returned as a [`~tokenization_utils_base.TokenSpan`] with:

        - **start** -- Index of the first token.
        - **end** -- Index of the token following the last token.

        Can be called as:

        - `self.word_to_tokens(word_index, sequence_index: int = 0)` if batch size is 1
        - `self.word_to_tokens(batch_index, word_index, sequence_index: int = 0)` if batch size is greater or equal to
          1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_word_index (`int`):
                Index of the sequence in the batch. If the batch only comprises one sequence, this can be the index of
                the word in the sequence.
            word_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided word index belongs to.

        Returns:
            ([`~tokenization_utils_base.TokenSpan`], *optional*): Span of tokens in the encoded sequence. Returns
            `None` if no tokens correspond to the word. This can happen especially when the token is a special token
            that has been used to format the tokenization. For example when we add a class token at the very beginning
            of the tokenization.
        """

        # Check if encodings are available; raise an error if not
        if not self._encodings:
            raise ValueError("word_to_tokens() is not available when using Python based tokenizers")

        # Determine whether batch_index or word_index was provided
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index

        # Adjust negative batch_index to account for batch size
        if batch_index < 0:
            batch_index = self._batch_size + batch_index

        # Adjust negative word_index to account for sequence length
        if word_index < 0:
            word_index = self._seq_len + word_index

        # Retrieve the token span for the specified word and sequence index
        span = self._encodings[batch_index].word_to_tokens(word_index, sequence_index)

        # Return the TokenSpan object constructed from span, or None if span is None
        return TokenSpan(*span) if span is not None else None
    def token_to_chars(self, batch_or_token_index: int, token_index: Optional[int] = None) -> CharSpan:
        """
        Get the character span corresponding to an encoded token in a sequence of the batch.

        Character spans are returned as a [`~tokenization_utils_base.CharSpan`] with:

        - **start** -- Index of the first character in the original string associated to the token.
        - **end** -- Index of the character following the last character in the original string associated to the
          token.

        Can be called as:

        - `self.token_to_chars(token_index)` if batch size is 1
        - `self.token_to_chars(batch_index, token_index)` if batch size is greater or equal to 1

        Args:
            batch_or_token_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the token in the sequence.
            token_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the token or tokens in
                the sequence.

        Returns:
            [`~tokenization_utils_base.CharSpan`]: Span of characters in the original string, or None, if the token
            (e.g. <s>, </s>) doesn't correspond to any chars in the origin string.
        """

        # 如果没有编码信息，则抛出错误，Python 版本的分词器不支持 token_to_chars()
        if not self._encodings:
            raise ValueError("token_to_chars() is not available when using Python based tokenizers")

        # 如果 token_index 不为 None，则说明参数中包含 batch_index 和 token_index
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            # 如果 token_index 为 None，则参数中只有 batch_or_token_index，此时 batch_index 设为 0
            batch_index = 0
            token_index = batch_or_token_index
        
        # 获取字符跨度的起始和结束索引
        span_indices = self._encodings[batch_index].token_to_chars(token_index)

        # 如果 span_indices 不为 None，则返回 CharSpan 对象，否则返回 None
        return CharSpan(*span_indices) if span_indices is not None else None
    ) -> int:
        """
        Get the index of the token in the encoded output comprising a character in the original string for a sequence
        of the batch.

        Can be called as:

        - `self.char_to_token(char_index)` if batch size is 1
        - `self.char_to_token(batch_index, char_index)` if batch size is greater or equal to 1

        This method is particularly suited when the input sequences are provided as pre-tokenized sequences (i.e. words
        are defined by the user). In this case it allows to easily associate encoded tokens with provided tokenized
        words.

        Args:
            batch_or_char_index (`int`):
                Index of the sequence in the batch. If the batch only comprise one sequence, this can be the index of
                the word in the sequence
            char_index (`int`, *optional*):
                If a batch index is provided in *batch_or_token_index*, this can be the index of the word in the
                sequence.
            sequence_index (`int`, *optional*, defaults to 0):
                If pair of sequences are encoded in the batch this can be used to specify which sequence in the pair (0
                or 1) the provided character index belongs to.


        Returns:
            `int`: Index of the token.
        """

        # 如果没有编码信息，则抛出异常，因为在使用基于 Python 的分词器时无法使用 char_to_token()
        if not self._encodings:
            raise ValueError("char_to_token() is not available when using Python based tokenizers")

        # 根据参数情况确定 batch_index 和 char_index 的值
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index

        # 调用内部编码对象的 char_to_token 方法，返回字符对应的 token 索引
        return self._encodings[batch_index].char_to_token(char_index, sequence_index)
    # 定义一个方法，用于获取给定批次中的序列中指定单词在原始字符串中的字符跨度

    def word_to_chars(batch_or_word_index: int, word_index: Optional[int] = None, sequence_index: int = 0) -> CharSpan:
        """
        获取给定批次中的序列中指定单词在原始字符串中的字符跨度。

        字符跨度以 CharSpan 命名元组的形式返回，具有以下字段：
        - start: 原始字符串中第一个字符的索引
        - end: 原始字符串中最后一个字符之后的索引

        可以按以下方式调用：
        - `self.word_to_chars(word_index)` 如果批次大小为 1
        - `self.word_to_chars(batch_index, word_index)` 如果批次大小大于等于 1

        参数:
            batch_or_word_index (`int`):
                批次中序列的索引。如果批次只包含一个序列，则可以是序列中单词的索引。
            word_index (`int`, *optional*):
                如果在 `batch_or_word_index` 中提供了批次索引，则可以是序列中单词的索引。
            sequence_index (`int`, *optional*, 默认为 0):
                如果批次编码了一对序列，则可以用来指定所提供单词索引属于哪一个序列 (0 或 1)。

        返回:
            `CharSpan` 或 `List[CharSpan]`: 字符串中相关字符或字符组的跨度。CharSpan 是一个命名元组，具有以下字段：
            - start: 原始字符串中与令牌关联的第一个字符的索引
            - end: 原始字符串中与令牌关联的最后一个字符之后的索引
        """

        # 如果未提供编码，则抛出 ValueError
        if not self._encodings:
            raise ValueError("word_to_chars() 在使用基于 Python 的分词器时不可用")
        
        # 根据参数 word_index 的存在与否，确定 batch_index 的值
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        
        # 调用 _encodings 中相应批次和序列索引的 word_to_chars 方法，并返回其结果
        return CharSpan(*(self._encodings[batch_index].word_to_chars(word_index, sequence_index)))
    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        """
        将所有值发送到指定设备，通过调用 `v.to(device)` (仅适用于 PyTorch)。

        Args:
            device (`str` or `torch.device`): 要放置张量的设备。

        Returns:
            [`BatchEncoding`]: 修改后的相同实例。
        """
        # 检查是否使用了必需的后端
        requires_backends(self, ["torch"])

        # 这个检查捕捉到像 APEX 在模块的所有输入上盲目调用 "to" 的情况
        # 否则它将继续向下传递，并将包含标记索引的 LongTensor 转换为 HalfTensor
        if isinstance(device, str) or is_torch_device(device) or isinstance(device, int):
            # 将数据字典中的每个值都转移到指定设备上
            self.data = {k: v.to(device=device) for k, v in self.data.items()}
        else:
            # 如果尝试将 BatchEncoding 转换为不支持的类型，则发出警告
            logger.warning(f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported.")
        # 返回修改后的实例
        return self
class SpecialTokensMixin:
    """
    A mixin derived by [`PreTrainedTokenizer`] and [`PreTrainedTokenizerFast`] to handle specific behaviors related to
    special tokens. In particular, this class hold the attributes which can be used to directly access these special
    tokens in a model-independent manner and allow to set and update the special tokens.

    Args:
        bos_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the beginning of a sentence.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the end of a sentence.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing an out-of-vocabulary token.
        sep_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token separating two different sentences in the same input (used by BERT for instance).
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        cls_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing the class of the input (used by BERT for instance).
        mask_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token representing a masked token (used by masked-language modeling pretraining objectives, like
            BERT).
        additional_special_tokens (tuple or list of `str` or `tokenizers.AddedToken`, *optional*):
            A tuple or a list of additional tokens, which will be marked as `special`, meaning that they will be
            skipped when decoding if `skip_special_tokens` is set to `True`.
    """

    # Define a class attribute listing the names of all special token attributes
    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]
    # 初始化函数，用于创建一个新的对象实例
    def __init__(self, verbose=False, **kwargs):
        # 下面的属性用于存储特殊 token 的值，初始设为 None
        self._bos_token = None  # 开始 token
        self._eos_token = None  # 结束 token
        self._unk_token = None  # 未知 token
        self._sep_token = None  # 分隔 token
        self._pad_token = None  # 填充 token
        self._cls_token = None  # 类别 token
        self._mask_token = None  # 掩码 token
        self._pad_token_type_id = 0  # 填充 token 的类型 ID，默认为 0
        self._additional_special_tokens = []  # 额外的特殊 token 列表
        self.verbose = verbose  # 是否为详细模式

        # 直接设置隐藏的值以允许初始化特殊 token，这些 token 可能尚未在词汇表中，用于序列化/反序列化
        # TODO 在某些时候清理此处代码（可能通过切换到快速分词器实现）

        # 遍历关键字参数 kwargs
        for key, value in kwargs.items():
            # 如果值为 None，则跳过
            if value is None:
                continue
            # 检查是否为特殊 token 属性之一
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                # 如果是额外特殊 token 的情况
                if key == "additional_special_tokens":
                    assert isinstance(value, (list, tuple)), f"Value {value} is not a list or tuple"
                    assert all(
                        isinstance(t, (str, AddedToken)) for t in value
                    ), "One of the tokens is not a string or an AddedToken"
                    setattr(self, key, value)  # 设置额外特殊 token 列表
                # 如果值为字符串或 AddedToken 对象，则直接设置
                elif isinstance(value, (str, AddedToken)):
                    setattr(self, key, value)  # 设置特殊 token 的值
                else:
                    raise TypeError(f"Special token {key} has to be either str or AddedToken but got: {type(value)}")
    ) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary and will be isolated before the tokenization
        algorithm is applied. Added tokens and tokens from the vocabulary of the tokenization algorithm are therefore
        not treated in the same way.

        Note, when adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix
        of the model so that its embedding matrix matches the tokenizer.

        In order to do that, please use the [`~PreTrainedModel.resize_token_embeddings`] method.

        Args:
            new_tokens (`str`, `tokenizers.AddedToken` or a list of *str* or `tokenizers.AddedToken`):
                Tokens are only added if they are not already in the vocabulary. `tokenizers.AddedToken` wraps a string
                token to let you personalize its behavior: whether this token should only match against a single word,
                whether this token should strip all potential whitespaces on the left side, whether this token should
                strip all potential whitespaces on the right side, etc.
            special_tokens (`bool`, *optional*, defaults to `False`):
                Can be used to specify if the token is a special token. This mostly changes the normalization behavior
                (special tokens like CLS or [MASK] are usually not lower-cased for instance).

                See details for `tokenizers.AddedToken` in HuggingFace tokenizers library.

        Returns:
            `int`: Number of tokens added to the vocabulary.

        Examples:

        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
        model = BertModel.from_pretrained("google-bert/bert-base-uncased")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        if not new_tokens:
            return 0  # 如果没有新的token要添加，则直接返回0

        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]  # 确保new_tokens是列表或元组形式

        return self._add_tokens(new_tokens, special_tokens=special_tokens)  # 调用内部方法_add_tokens来实际添加token

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        raise NotImplementedError  # 这是一个占位符方法，需要在子类中实现

    @property



        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary and will be isolated before the tokenization
        algorithm is applied. Added tokens and tokens from the vocabulary of the tokenization algorithm are therefore
        not treated in the same way.

        Note, when adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix
        of the model so that its embedding matrix matches the tokenizer.

        In order to do that, please use the [`~PreTrainedModel.resize_token_embeddings`] method.

        Args:
            new_tokens (`str`, `tokenizers.AddedToken` or a list of *str* or `tokenizers.AddedToken`):
                Tokens are only added if they are not already in the vocabulary. `tokenizers.AddedToken` wraps a string
                token to let you personalize its behavior: whether this token should only match against a single word,
                whether this token should strip all potential whitespaces on the left side, whether this token should
                strip all potential whitespaces on the right side, etc.
            special_tokens (`bool`, *optional*, defaults to `False`):
                Can be used to specify if the token is a special token. This mostly changes the normalization behavior
                (special tokens like CLS or [MASK] are usually not lower-cased for instance).

                See details for `tokenizers.AddedToken` in HuggingFace tokenizers library.

        Returns:
            `int`: Number of tokens added to the vocabulary.

        Examples:

        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
        model = BertModel.from_pretrained("google-bert/bert-base-uncased")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        if not new_tokens:
            return 0  # If there are no new tokens to add, return 0

        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]  # Ensure new_tokens is in list or tuple form

        return self._add_tokens(new_tokens, special_tokens=special_tokens)  # Call internal method _add_tokens to actually add tokens

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        raise NotImplementedError  # This is a placeholder method, needs to be implemented in subclass

    @property
    def bos_token(self) -> str:
        """
        `str`: Beginning of sentence token. Log an error if used while not having been set.
        """
        # 如果未设置开始句子的标记，则记录错误并返回 None
        if self._bos_token is None:
            if self.verbose:
                logger.error("Using bos_token, but it is not set yet.")
            return None
        # 返回开始句子的标记
        return str(self._bos_token)

    @property
    def eos_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        # 如果未设置结束句子的标记，则记录错误并返回 None
        if self._eos_token is None:
            if self.verbose:
                logger.error("Using eos_token, but it is not set yet.")
            return None
        # 返回结束句子的标记
        return str(self._eos_token)

    @property
    def unk_token(self) -> str:
        """
        `str`: Unknown token. Log an error if used while not having been set.
        """
        # 如果未设置未知标记，则记录错误并返回 None
        if self._unk_token is None:
            if self.verbose:
                logger.error("Using unk_token, but it is not set yet.")
            return None
        # 返回未知标记
        return str(self._unk_token)

    @property
    def sep_token(self) -> str:
        """
        `str`: Separation token, to separate context and query in an input sequence. Log an error if used while not
        having been set.
        """
        # 如果未设置分隔标记，则记录错误并返回 None
        if self._sep_token is None:
            if self.verbose:
                logger.error("Using sep_token, but it is not set yet.")
            return None
        # 返回分隔标记
        return str(self._sep_token)

    @property
    def pad_token(self) -> str:
        """
        `str`: Padding token. Log an error if used while not having been set.
        """
        # 如果未设置填充标记，则记录错误并返回 None
        if self._pad_token is None:
            if self.verbose:
                logger.error("Using pad_token, but it is not set yet.")
            return None
        # 返回填充标记
        return str(self._pad_token)

    @property
    def cls_token(self) -> str:
        """
        `str`: Classification token, to extract a summary of an input sequence leveraging self-attention along the full
        depth of the model. Log an error if used while not having been set.
        """
        # 如果未设置分类标记，则记录错误并返回 None
        if self._cls_token is None:
            if self.verbose:
                logger.error("Using cls_token, but it is not set yet.")
            return None
        # 返回分类标记
        return str(self._cls_token)

    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.
        """
        # 如果未设置掩码标记，则记录错误并返回 None
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        # 返回掩码标记
        return str(self._mask_token)
    def additional_special_tokens(self) -> List[str]:
        """
        `List[str]`: Returns a list of additional special tokens. Raises an error if accessed before being set.
        """
        # 如果 _additional_special_tokens 为 None，则打印错误信息并返回 None
        if self._additional_special_tokens is None:
            if self.verbose:
                logger.error("Using additional_special_tokens, but it is not set yet.")
            return None
        # 将 _additional_special_tokens 转换成字符串列表并返回
        return [str(tok) for tok in self._additional_special_tokens]

    @bos_token.setter
    def bos_token(self, value):
        # 如果 value 不是字符串或 AddedToken 对象，并且不为 None，则抛出值错误异常
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the BOS token")
        # 设置 _bos_token 属性为给定值
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        # 如果 value 不是字符串或 AddedToken 对象，并且不为 None，则抛出值错误异常
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the EOS token")
        # 设置 _eos_token 属性为给定值
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        # 如果 value 不是字符串或 AddedToken 对象，并且不为 None，则抛出值错误异常
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the UNK token")
        # 设置 _unk_token 属性为给定值
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        # 如果 value 不是字符串或 AddedToken 对象，并且不为 None，则抛出值错误异常
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the SEP token")
        # 设置 _sep_token 属性为给定值
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        # 如果 value 不是字符串或 AddedToken 对象，并且不为 None，则抛出值错误异常
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the PAD token")
        # 设置 _pad_token 属性为给定值
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        # 如果 value 不是字符串或 AddedToken 对象，并且不为 None，则抛出值错误异常
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the CLS token")
        # 设置 _cls_token 属性为给定值
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        # 如果 value 不是字符串或 AddedToken 对象，并且不为 None，则抛出值错误异常
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the MASK token")
        # 设置 _mask_token 属性为给定值
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        # 设置 _additional_special_tokens 属性为给定值，如果 value 是 None，则设置为 None
        self._additional_special_tokens = value if value is not None else None

    @property
    def bos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Returns the ID of the beginning of sentence token in the vocabulary. Returns `None` if the
        token has not been set.
        """
        # 如果 _bos_token 为 None，则返回 None
        if self._bos_token is None:
            return None
        # 调用 convert_tokens_to_ids 方法将 _bos_token 转换成对应的 ID 并返回
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Returns the ID of the end of sentence token in the vocabulary. Returns `None` if the token has
        not been set.
        """
        # 如果 _eos_token 为 None，则返回 None
        if self._eos_token is None:
            return None
        # 调用 convert_tokens_to_ids 方法将 _eos_token 转换成对应的 ID 并返回
        return self.convert_tokens_to_ids(self.eos_token)
    @property
    def unk_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the unknown token in the vocabulary. Returns `None` if the token has not been set.
        """
        # 如果未设置未知标记，则返回 None
        if self._unk_token is None:
            return None
        # 否则，将未知标记转换为其对应的 id 并返回
        return self.convert_tokens_to_ids(self.unk_token)
    
    @property
    def sep_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the separation token in the vocabulary, to separate context and query in an input
        sequence. Returns `None` if the token has not been set.
        """
        # 如果未设置分隔标记，则返回 None
        if self._sep_token is None:
            return None
        # 否则，将分隔标记转换为其对应的 id 并返回
        return self.convert_tokens_to_ids(self.sep_token)
    
    @property
    def pad_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        """
        # 如果未设置填充标记，则返回 None
        if self._pad_token is None:
            return None
        # 否则，将填充标记转换为其对应的 id 并返回
        return self.convert_tokens_to_ids(self.pad_token)
    
    @property
    def pad_token_type_id(self) -> int:
        """
        `int`: Id of the padding token type in the vocabulary.
        """
        # 直接返回填充标记类型的 id
        return self._pad_token_type_id
    
    @property
    def cls_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the classification token in the vocabulary, to extract a summary of an input sequence
        leveraging self-attention along the full depth of the model.
    
        Returns `None` if the token has not been set.
        """
        # 如果未设置分类标记，则返回 None
        if self._cls_token is None:
            return None
        # 否则，将分类标记转换为其对应的 id 并返回
        return self.convert_tokens_to_ids(self.cls_token)
    
    @property
    def mask_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the mask token in the vocabulary, used when training a model with masked-language
        modeling. Returns `None` if the token has not been set.
        """
        # 如果未设置掩码标记，则返回 None
        if self._mask_token is None:
            return None
        # 否则，将掩码标记转换为其对应的 id 并返回
        return self.convert_tokens_to_ids(self.mask_token)
    
    @property
    def additional_special_tokens_ids(self) -> List[int]:
        """
        `List[int]`: Ids of all the additional special tokens in the vocabulary. Log an error if used while not having
        been set.
        """
        # 返回所有额外特殊标记的 id 列表
        return self.convert_tokens_to_ids(self.additional_special_tokens)
    
    @bos_token_id.setter
    def bos_token_id(self, value):
        # 如果设置了开始标记的 id，则将其转换为对应的 token 并存储
        self._bos_token = self.convert_ids_to_tokens(value) if value is not None else None
    
    @eos_token_id.setter
    def eos_token_id(self, value):
        # 如果设置了结束标记的 id，则将其转换为对应的 token 并存储
        self._eos_token = self.convert_ids_to_tokens(value) if value is not None else None
    
    @unk_token_id.setter
    def unk_token_id(self, value):
        # 如果设置了未知标记的 id，则将其转换为对应的 token 并存储
        self._unk_token = self.convert_ids_to_tokens(value) if value is not None else None
    
    @sep_token_id.setter
    def sep_token_id(self, value):
        # 如果设置了分隔标记的 id，则将其转换为对应的 token 并存储
        self._sep_token = self.convert_ids_to_tokens(value) if value is not None else None
    
    @pad_token_id.setter
    def pad_token_id(self, value):
        # 设置填充标记的 ID，并转换成对应的标记字符串，如果值为 None，则将 _pad_token 设为 None
        self._pad_token = self.convert_ids_to_tokens(value) if value is not None else None

    @cls_token_id.setter
    def cls_token_id(self, value):
        # 设置类别标记的 ID，并转换成对应的标记字符串，如果值为 None，则将 _cls_token 设为 None
        self._cls_token = self.convert_ids_to_tokens(value) if value is not None else None

    @mask_token_id.setter
    def mask_token_id(self, value):
        # 设置掩码标记的 ID，并转换成对应的标记字符串，如果值为 None，则将 _mask_token 设为 None
        self._mask_token = self.convert_ids_to_tokens(value) if value is not None else None

    @additional_special_tokens_ids.setter
    def additional_special_tokens_ids(self, values):
        # 设置额外特殊标记的 ID 列表，并逐个转换成对应的标记字符串
        self._additional_special_tokens = [self.convert_ids_to_tokens(value) for value in values]

    @property
    def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]:
        """
        `Dict[str, Union[str, List[str]]]`: 将特殊标记类属性（如 `cls_token`、`unk_token` 等）映射到它们的值（如 `'<unk>'`、`'<cls>'` 等）的字典。

        将 `tokenizers.AddedToken` 类型的潜在标记转换为字符串。
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    @property
    def special_tokens_map_extended(self) -> Dict[str, Union[str, AddedToken, List[Union[str, AddedToken]]]]:
        """
        `Dict[str, Union[str, tokenizers.AddedToken, List[Union[str, tokenizers.AddedToken]]]]`: 将特殊标记类属性（如 `cls_token`、`unk_token` 等）映射到它们的值（如 `'<unk>'`、`'<cls>'` 等）的字典。

        不将 `tokenizers.AddedToken` 类型的标记转换为字符串，以便更精细地控制特殊标记的分词。
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr
    def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]:
        """
        `List[Union[str, tokenizers.AddedToken]]`: 返回所有特殊标记（如 `<unk>`、`<cls>` 等）的列表，
        其顺序与每个标记的索引无关。如果需要正确的索引，请查看 `self.added_tokens_encoder`。
        无法按顺序创建了，因为键是 `AddedToken` 而不是 `String`。

        不要将 `tokenizers.AddedToken` 类型的标记转换为字符串，以便更精细地控制特殊标记的分词过程。
        """
        all_tokens = []
        seen = set()
        for value in self.special_tokens_map_extended.values():
            if isinstance(value, (list, tuple)):
                tokens_to_add = [token for token in value if str(token) not in seen]
            else:
                tokens_to_add = [value] if str(value) not in seen else []
            seen.update(map(str, tokens_to_add))
            all_tokens.extend(tokens_to_add)
        return all_tokens

    @property
    def all_special_tokens(self) -> List[str]:
        """
        `List[str]`: 返回唯一特殊标记（`'<unk>'`、`'<cls>'` 等）的列表。

        将 `tokenizers.AddedToken` 类型的标记转换为字符串。
        """
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: 返回特殊标记（`'<unk>'`、`'<cls>'` 等）映射到类属性的 id 列表。
        """
        all_toks = self.all_special_tokens
        all_ids = self.convert_tokens_to_ids(all_toks)
        return all_ids
"""

"""


INIT_TOKENIZER_DOCSTRING = r"""
    Class attributes (overridden by derived classes)

        - **vocab_files_names** (`Dict[str, str]`) -- A dictionary with, as keys, the `__init__` keyword name of each
          vocabulary file required by the model, and as associated values, the filename for saving the associated file
          (string).
        - **pretrained_vocab_files_map** (`Dict[str, Dict[str, str]]`) -- A dictionary of dictionaries, with the
          high-level keys being the `__init__` keyword name of each vocabulary file required by the model, the
          low-level being the `short-cut-names` of the pretrained models with, as associated values, the `url` to the
          associated pretrained vocabulary file.
        - **max_model_input_sizes** (`Dict[str, Optional[int]]`) -- A dictionary with, as keys, the `short-cut-names`
          of the pretrained models, and as associated values, the maximum length of the sequence inputs of this model,
          or `None` if the model has no maximum input size.
        - **pretrained_init_configuration** (`Dict[str, Dict[str, Any]]`) -- A dictionary with, as keys, the
          `short-cut-names` of the pretrained models, and as associated values, a dictionary of specific arguments to
          pass to the `__init__` method of the tokenizer class for this pretrained model when loading the tokenizer
          with the [`~tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained`] method.
        - **model_input_names** (`List[str]`) -- A list of inputs expected in the forward pass of the model.
        - **padding_side** (`str`) -- The default value for the side on which the model should have padding applied.
          Should be `'right'` or `'left'`.
        - **truncation_side** (`str`) -- The default value for the side on which the model should have truncation
          applied. Should be `'right'` or `'left'`.

"""

# 使用一个原始的字符串（raw string）定义类的文档字符串，用于描述各种类属性和其功能
@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class PreTrainedTokenizerBase(SpecialTokensMixin, PushToHubMixin):
    """
    Base class for [`PreTrainedTokenizer`] and [`PreTrainedTokenizerFast`].

    Handles shared (mostly boiler plate) methods for those two classes.
    """

    # 字典，存储每个词汇文件的初始化关键字名和文件名
    vocab_files_names: Dict[str, str] = {}

    # 字典的字典，存储预训练模型的初始化关键字名和对应的预训练模型名称及其 URL
    pretrained_vocab_files_map: Dict[str, Dict[str, str]] = {}

    # 字典，存储每个预训练模型的初始化关键字名和模型输入的最大长度
    max_model_input_sizes: Dict[str, Optional[int]] = {}

    # 字典的字典，存储每个预训练模型的初始化关键字名和加载 tokenizer 时传递给其 `__init__` 方法的特定参数
    pretrained_init_configuration: Dict[str, Dict[str, Any]] = {}

    # 列表，存储模型前向传播时期望的输入名称
    model_input_names: List[str] = ["input_ids", "token_type_ids", "attention_mask"]

    # 字符串，表示默认的填充方向，应为 `'right'` 或 `'left'`
    padding_side: str = "right"

    # 字符串，表示默认的截断方向，应为 `'right'` 或 `'left'`
    truncation_side: str = "right"

    # 类变量，指向慢速 tokenizer 类（如果有的话）
    slow_tokenizer_class = None
    # 定义类初始化方法，接受可选参数进行对象初始化
    def __init__(self, **kwargs):
        # 初始化输入参数以及用于保存和重新加载参数（见 from_pretrained 和 save_pretrained 方法）
        self.init_inputs = ()
        # 深复制 kwargs 参数，用于后续重载操作
        self.init_kwargs = copy.deepcopy(kwargs)
        # 提取 name_or_path 参数，默认为空字符串
        self.name_or_path = kwargs.pop("name_or_path", "")
        # 获取并设置 model_max_length 参数，默认为非常大的值
        model_max_length = kwargs.pop("model_max_length", kwargs.pop("max_len", None))
        self.model_max_length = model_max_length if model_max_length is not None else VERY_LARGE_INTEGER
    
        # 设置默认的数据填充和剪裁方向，右填充和左填充可以由子类覆盖。根据提供的 kwargs 参数进行调整
        self.padding_side = kwargs.pop("padding_side", self.padding_side)  # 默认是右侧填充
        if self.padding_side not in ["right", "left"]:
            raise ValueError(
                f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
            )
        self.truncation_side = kwargs.pop("truncation_side", self.truncation_side)  # 默认是右侧剪裁
        if self.truncation_side not in ["right", "left"]:
            raise ValueError(
                f"Truncation side should be selected between 'right' and 'left', current value: {self.truncation_side}"
            )
        # 获取并存储模型输入名称参数，默认为当前类定义的模型输入名称列表
        self.model_input_names = kwargs.pop("model_input_names", self.model_input_names)
    
        # 默认为进行快速和慢速分词器的分词空间清理
        self.clean_up_tokenization_spaces = kwargs.pop("clean_up_tokenization_spaces", True)
    
        # 默认为不分离特殊符号参数，默认为 False
        self.split_special_tokens = kwargs.pop("split_special_tokens", False)
    
        # 初始化已注意到的弃用警告字典（避免重复警告）
        self.deprecation_warnings = {}
        self._in_target_context_manager = False
    
        # 存储一个 Jinja 模板对象，用于格式化对话历史为可分词的字符串
        self.chat_template = kwargs.pop("chat_template", None)
        if isinstance(self.chat_template, (list, tuple)):
            # 当 chat_template 是一个列表或元组时，将其转换成单个字典结构，便于后续操作
            self.chat_template = {template["name"]: template["template"] for template in self.chat_template}
    
        # 调用父类的 __init__ 方法，使用余下的参数
        super().__init__(**kwargs)
    
    # 计算和返回单句的最大长度
    @staticmethod
    def max_len_single_sentence() -> int:
        """
        `int`: 单句可以输入到模型的最大长度。
        """
        return self.model_max_length - self.num_special_tokens_to_add(pair=False)
    
    # 计算并返回一对句子的最大结合长度
    @staticmethod
    def max_len_sentences_pair() -> int:
        """
        `int`: 一对句子可以输入到模型的最大结合长度。
        """
        return self.model_max_length - self.num_special_tokens_to_add(pair=True)
    # 设置属性 max_len_single_sentence 的 setter 方法，用于设置单个句子的最大长度
    def max_len_single_sentence(self, value) -> int:
        # 检查是否为向后兼容性，允许设置 'max_len_single_sentence'
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=False) and self.verbose:
            # 如果设置的值符合向后兼容性要求且 verbose 为真，则发出警告
            if not self.deprecation_warnings.get("max_len_single_sentence", False):
                logger.warning(
                    "Setting 'max_len_single_sentence' is now deprecated. This value is automatically set up."
                )
            # 标记 'max_len_single_sentence' 已发出过警告
            self.deprecation_warnings["max_len_single_sentence"] = True
        else:
            # 如果设置的值不符合向后兼容性要求，则抛出 ValueError 异常
            raise ValueError(
                "Setting 'max_len_single_sentence' is now deprecated. This value is automatically set up."
            )

    # 设置属性 max_len_sentences_pair 的 setter 方法，用于设置句对的最大长度
    def max_len_sentences_pair(self, value) -> int:
        # 检查是否为向后兼容性，允许设置 'max_len_sentences_pair'
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=True) and self.verbose:
            # 如果设置的值符合向后兼容性要求且 verbose 为真，则发出警告
            if not self.deprecation_warnings.get("max_len_sentences_pair", False):
                logger.warning(
                    "Setting 'max_len_sentences_pair' is now deprecated. This value is automatically set up."
                )
            # 标记 'max_len_sentences_pair' 已发出过警告
            self.deprecation_warnings["max_len_sentences_pair"] = True
        else:
            # 如果设置的值不符合向后兼容性要求，则抛出 ValueError 异常
            raise ValueError("Setting 'max_len_sentences_pair' is now deprecated. This value is automatically set up.")

    # 设置 _processor_class 属性的私有方法，用于设置处理器类别
    def _set_processor_class(self, processor_class: str):
        """Sets processor class as an attribute."""
        self._processor_class = processor_class

    # 定义属性 added_tokens_decoder 的 getter 方法，返回一个字典，表示添加的特殊标记的解码器
    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        # 抛出 NotImplementedError，因为该方法需要在子类中实现具体逻辑
        raise NotImplementedError()

    # 定义对象的字符串表示形式，用于返回对象的详细描述信息
    def __repr__(self) -> str:
        # 将 added_tokens_decoder 属性的内容转换成字符串形式
        added_tokens_decoder_rep = "\n\t".join([f"{k}: {v.__repr__()}," for k, v in self.added_tokens_decoder.items()])
        # 返回对象的字符串表示形式，包括对象的各种属性信息和 added_tokens_decoder 的内容
        return (
            f"{self.__class__.__name__}(name_or_path='{self.name_or_path}',"
            f" vocab_size={self.vocab_size}, model_max_length={self.model_max_length}, is_fast={self.is_fast},"
            f" padding_side='{self.padding_side}', truncation_side='{self.truncation_side}',"
            f" special_tokens={self.special_tokens_map}, clean_up_tokenization_spaces={self.clean_up_tokenization_spaces}), "
            " added_tokens_decoder={\n\t" + added_tokens_decoder_rep + "\n}"
        )

    # 定义对象的长度方法，返回对象的长度信息
    def __len__(self) -> int:
        # 抛出 NotImplementedError，因为该方法需要在子类中实现具体逻辑
        raise NotImplementedError()

    # 获取词汇表的方法，返回一个字典，表示 token 到 index 的映射
    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.

        `tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the
        vocab.

        Returns:
            `Dict[str, int]`: The vocabulary.
        """
        # 抛出 NotImplementedError，因为该方法需要在子类中实现具体逻辑
        raise NotImplementedError()
    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], "Conversation"],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_dict: bool = False,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Applies a chat template to format a conversation.

        Args:
            conversation (Union[List[Dict[str, str]], "Conversation"]): The conversation data to format.
            chat_template (Optional[str]): Optional template string to format messages.
            add_generation_prompt (bool): Whether to add a generation prompt at the end.
            tokenize (bool): Whether to tokenize the formatted output.
            padding (bool): Whether to apply padding to the tokens.
            truncation (bool): Whether to truncate tokens if exceeding max_length.
            max_length (Optional[int]): Maximum length of the formatted output.
            return_tensors (Optional[Union[str, TensorType]]): Return type for tokenized outputs.
            return_dict (bool): Whether to return the output as a dictionary.
            tokenizer_kwargs (Optional[Dict[str, Any]]): Optional keyword arguments for the tokenizer.
            **kwargs: Additional keyword arguments.

        Returns:
            Formatted conversation based on the provided chat template.
        """
        try:
            import jinja2
            from jinja2.exceptions import TemplateError
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            raise ImportError("apply_chat_template requires jinja2 to be installed.")

        if version.parse(jinja2.__version__) < version.parse("3.0.0"):
            raise ImportError(
                "apply_chat_template requires jinja2>=3.0.0 to be installed. Your version is " f"{jinja2.__version__}."
            )

        def raise_exception(message):
            """
            Helper function to raise a TemplateError with a specified message.

            Args:
                message (str): Error message to raise.

            Raises:
                TemplateError: Exception with the provided message.
            """
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        return jinja_env.from_string(chat_template)

    @lru_cache
    def _compile_jinja_template(self, chat_template):
        """
        Compiles a Jinja template using a sandboxed environment.

        Args:
            chat_template (str): The Jinja template string to compile.

        Returns:
            Jinja Template: Compiled Jinja template object.
        """
        try:
            import jinja2
            from jinja2.exceptions import TemplateError
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            raise ImportError("_compile_jinja_template requires jinja2 to be installed.")

        if version.parse(jinja2.__version__) < version.parse("3.0.0"):
            raise ImportError(
                "_compile_jinja_template requires jinja2>=3.0.0 to be installed. Your version is " f"{jinja2.__version__}."
            )

        def raise_exception(message):
            """
            Helper function to raise a TemplateError with a specified message.

            Args:
                message (str): Error message to raise.

            Raises:
                TemplateError: Exception with the provided message.
            """
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        return jinja_env.from_string(chat_template)

    @property
    def default_chat_template(self):
        """
        Property representing the default chat template in ChatML format.

        Returns:
            str: Default chat template.
        """
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using a default chat template "
            "that implements the ChatML format (without BOS/EOS tokens!). If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        return (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        trust_remote_code=False,
        **kwargs,
    ):
        """
        Creates an instance of the class from a pretrained model or path.

        Args:
            pretrained_model_name_or_path (Union[str, os.PathLike]): Name or path of the pretrained model.
            *init_inputs: Additional positional arguments for initialization.
            cache_dir (Optional[Union[str, os.PathLike]]): Optional directory to cache downloaded files.
            force_download (bool): Whether to force download the model files.
            local_files_only (bool): Whether to use only local files without downloading.
            token (Optional[Union[str, bool]]): Token to authenticate access to the pretrained model.
            revision (str): Revision of the pretrained model to use.
            trust_remote_code (bool): Whether to trust remote code for model initialization.
            **kwargs: Additional keyword arguments.

        Returns:
            Instance of the class initialized with the pretrained model.
        """
    @staticmethod
    def _eventually_correct_t5_max_length(pretrained_model_name_or_path, max_model_length, init_max_model_length):
        # This method should be deleted in Transformers v5
        # Its only purpose is to potentially throw a warning
        # that incorrectly defined max lengths of T5's tokenizer are used
        # which we will correct in Transformers v5.
        # 返回最大模型长度，这个方法在 Transformers v5 中应该被删除，只是用来可能发出警告，
        # 告知使用了错误定义的 T5 分词器的最大长度，我们将在 Transformers v5 中进行更正。
        return max_model_length

    @classmethod
    def convert_added_tokens(cls, obj: Union[AddedToken, Any], save=False, add_type_field=True):
        # 如果 obj 是字典且包含 "__type" 键且其值为 "AddedToken"
        if isinstance(obj, dict) and "__type" in obj and obj["__type"] == "AddedToken":
            obj.pop("__type")  # 移除 "__type" 键
            return AddedToken(**obj)  # 返回一个 AddedToken 对象
        # 如果 obj 是 AddedToken 对象且需要保存
        if isinstance(obj, AddedToken) and save:
            obj = obj.__getstate__()  # 获取对象状态
            if add_type_field:
                obj["__type"] = "AddedToken"  # 添加 "__type" 字段
            else:
                # 不保存 "special" 字段，适用于之前的分词器
                obj.pop("special")
            return obj
        elif isinstance(obj, (list, tuple)):
            # 如果 obj 是列表或元组，则递归地转换列表中的每个元素
            return [cls.convert_added_tokens(o, save=save, add_type_field=add_type_field) for o in obj]
        elif isinstance(obj, dict):
            # 如果 obj 是字典，则递归地转换字典中的每个值
            return {k: cls.convert_added_tokens(v, save=save, add_type_field=add_type_field) for k, v in obj.items()}
        return obj  # 返回原始对象

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        # 保存当前模型到指定目录
        pass  # 空函数体，用于定义方法结构，实际保存逻辑需根据具体情况添加

    def _save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        file_names: Tuple[str],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ):
        # 在指定目录下保存模型文件
        pass  # 空函数体，用于定义方法结构，实际保存逻辑需根据具体情况添加
    ) -> Tuple[str]:
        """
        Save a tokenizer using the slow-tokenizer/legacy format: vocabulary + added tokens.

        Fast tokenizers can also be saved in a unique JSON file containing {config + vocab + added-tokens} using the
        specific [`~tokenization_utils_fast.PreTrainedTokenizerFast._save_pretrained`]
        """
        # 如果不是传统格式，则抛出数值错误异常
        if legacy_format is False:
            raise ValueError(
                "Only fast tokenizers (instances of PreTrainedTokenizerFast) can be saved in non legacy format."
            )

        # 将保存目录转换为字符串类型
        save_directory = str(save_directory)

        # 构建添加的 tokens 文件路径，包括可选的前缀和固定的文件名后缀 ADDED_TOKENS_FILE
        added_tokens_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE
        )
        # 获取添加的 tokens 的词汇表，仅包括索引大于等于词汇表大小的 token
        added_vocab = {tok: index for tok, index in self.added_tokens_encoder.items() if index >= self.vocab_size}
        # 如果存在添加的 tokens，则写入到文件中
        if added_vocab:
            with open(added_tokens_file, "w", encoding="utf-8") as f:
                out_str = json.dumps(added_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                f.write(out_str)
                logger.info(f"added tokens file saved in {added_tokens_file}")

        # 调用实例方法保存词汇表文件，并返回文件路径的元组
        vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)

        # 返回文件名列表和词汇表文件路径的元组，包括添加的 tokens 文件路径
        return file_names + vocab_files + (added_tokens_file,)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary + added tokens).

        This method won't save the configuration and special token mappings of the tokenizer. Use
        [`~PreTrainedTokenizerFast._save_pretrained`] to save the whole state of the tokenizer.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        # 抛出未实现错误，提示子类应该实现该方法
        raise NotImplementedError
    @add_end_docstrings(
        ENCODE_KWARGS_DOCSTRING,
        """
            **kwargs: Passed along to the `.tokenize()` method.
        """,
        """
        Returns:
            `List[int]`, `torch.Tensor`, `tf.Tensor` or `np.ndarray`: The tokenized ids of the text.
        """,
    )

# 使用装饰器 `add_end_docstrings`，添加额外的文档字符串至 `encode` 方法
# 包含 `ENCODE_KWARGS_DOCSTRING` 变量定义的文档字符串
# 声明了 `**kwargs` 参数会传递给 `.tokenize()` 方法
# 描述了返回值为一个包含整数、PyTorch 张量、TensorFlow 张量或 NumPy 数组的列表，表示文本的标记化 ID


    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> List[int]:

# 定义 `encode` 方法，用于将字符串转换为 ID 序列，利用 tokenizer 和词汇表进行编码
# 参数 `text`：`Union[TextInput, PreTokenizedInput, EncodedInput]`，第一个要编码的序列，可以是字符串、字符串列表（使用 `tokenize` 方法标记化的字符串）或整数列表（使用 `convert_tokens_to_ids` 方法标记化的字符串 ID）
# 参数 `text_pair`：`Optional[Union[TextInput, PreTokenizedInput, EncodedInput]]`，可选的第二个要编码的序列，同样可以是字符串、字符串列表或整数列表
# 参数 `add_special_tokens`：`bool`，默认为 `True`，是否添加与模型相关的特殊标记
# 参数 `padding`：`Union[bool, str, PaddingStrategy]`，默认为 `False`，是否对输入进行填充
# 参数 `truncation`：`Union[bool, str, TruncationStrategy]`，是否进行截断
# 参数 `max_length`：`Optional[int]`，可选的最大长度限制
# 参数 `stride`：`int`，默认为 `0`，步长设置
# 参数 `return_tensors`：`Optional[Union[str, TensorType]]`，是否返回张量类型
# `**kwargs`：额外的关键字参数，将传递给底层模型特定的 `encode` 方法


        encoded_inputs = self.encode_plus(
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            **kwargs,
        )

# 调用 `encode_plus` 方法，将 `text` 和 `text_pair`（如果存在）编码为输入 IDs
# 传递其他参数如 `add_special_tokens`、`padding`、`truncation`、`max_length`、`stride`、`return_tensors` 和 `**kwargs` 给 `encode_plus` 方法
# 将返回的结果存储在 `encoded_inputs` 变量中


        return encoded_inputs["input_ids"]

# 返回 `encoded_inputs` 字典中键为 `"input_ids"` 的值，即文本的标记化 ID 列表
    # 定义一个方法用于计算需要添加的特殊标记数量，抛出未实现错误，需要在子类中实现
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        raise NotImplementedError

    # 定义一个方法用于获取填充和截断策略，返回策略参数
    def _get_padding_truncation_strategies(
        self, padding=False, truncation=None, max_length=None, pad_to_multiple_of=None, verbose=True, **kwargs
    ):
    
    # 装饰器：添加文档字符串到 __call__ 方法，使用默认和额外的编码参数文档字符串
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义 __call__ 方法，用于对输入进行编码处理
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
    
    # 定义 _call_one 方法，用于对单个输入进行编码处理
    def _call_one(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
    
    # 装饰器：添加文档字符串到 _call_one 方法，使用默认和额外的编码参数文档字符串
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义一个方法 `encode_plus`，用于处理文本及其可选的配对文本的编码和处理
    def encode_plus(
        self,
        # 输入文本，可以是单一文本、预处理后的输入或已编码的输入
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        # 可选的配对文本，可以是单一文本、预处理后的输入或已编码的输入
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        # 是否添加特殊标记，如 `[CLS]` 和 `[SEP]`
        add_special_tokens: bool = True,
        # 是否进行填充
        padding: Union[bool, str, PaddingStrategy] = False,
        # 是否进行截断
        truncation: Union[bool, str, TruncationStrategy] = None,
        # 最大长度限制
        max_length: Optional[int] = None,
        # 滑动窗口步长
        stride: int = 0,
        # 输入是否已经分割成单词
        is_split_into_words: bool = False,
        # 填充到指定的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，如 'pt' 表示 PyTorch 张量
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回 token 类型 ID
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码
        return_attention_mask: Optional[bool] = None,
        # 是否返回超出最大长度的 token
        return_overflowing_tokens: bool = False,
        # 是否返回特殊 token 掩码
        return_special_tokens_mask: bool = False,
        # 是否返回 token 的偏移映射
        return_offsets_mapping: bool = False,
        # 是否返回编码后的长度
        return_length: bool = False,
        # 是否启用详细模式，控制是否输出详细信息
        verbose: bool = True,
        # 其他参数
        **kwargs,
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences.

        <Tip warning={true}>

        This method is deprecated, `__call__` should be used instead.

        </Tip>

        Args:
            text (`str`, `List[str]` or `List[int]` (the latter only for not-fast tokenizers)):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略，以及最大长度和其他参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法 `_encode_plus` 进行编码
        return self._encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
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
    # 抽象方法，用于派生类实现，用于编码给定文本或文本对的方法
    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        # 抛出未实现错误，提醒子类需要实现这个方法
        raise NotImplementedError

    # 使用指定的文档字符串装饰器添加文档注释
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 批量编码给定文本或文本对的方法
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.

        <Tip warning={true}>

        This method is deprecated, `__call__` should be used instead.

        </Tip>

        Args:
            batch_text_or_text_pairs (`List[str]`, `List[Tuple[str, str]]`, `List[List[str]]`, `List[Tuple[List[str], List[str]]]`, and for not-fast tokenizers, also `List[List[int]]`, `List[Tuple[List[int], List[int]]]`):
                Batch of sequences or pair of sequences to be encoded. This can be a list of
                string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
                details in `encode_plus`).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略以及最大长度，以及其他参数设置
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用实际的编码方法，返回编码后的结果
        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
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
    # 重写父类中的方法，用于批量编码输入文本或文本对
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        # 抛出未实现的错误，强制子类实现该方法
        raise NotImplementedError

    # 在批量编码过程中对输入进行填充
    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ):
        # 未实现的填充方法，应在子类中实现具体逻辑
        raise NotImplementedError

    # 根据输入的token_ids构建特殊的token类型标识
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed. [What are token type
        IDs?](../glossary#token-type-ids)

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The token type ids.
        """
        # 如果只有一个序列，所有token type为0
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        # 如果有两个序列，第一个序列的token type为0，第二个序列的token type为1
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    # 构建包含特殊token的输入序列
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        # 未实现的方法，应在子类中定义如何构建带有特殊token的输入序列
        raise NotImplementedError
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        This implementation does not add special tokens and this method should be overridden in a subclass.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The model input with special tokens.
        """
        # 如果没有第二个序列，直接返回第一个序列
        if token_ids_1 is None:
            return token_ids_0
        # 否则将两个序列连接起来
        return token_ids_0 + token_ids_1

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    ):
        """
        Placeholder method that should be overridden in a subclass to prepare model inputs with special tokens.

        This method includes various options for processing inputs such as padding, truncation, and returning
        tensors in specific formats.

        Args:
            ids (`List[int]`): List of input token IDs.
            pair_ids (`Optional[List[int]]`, *optional*): List of token IDs for the second sequence in pair inputs.
            add_special_tokens (`bool`, *optional*): Whether to add special tokens to the input sequences.
            padding (`Union[bool, str, PaddingStrategy]`, *optional*): Padding strategy or boolean for padding sequences.
            truncation (`Union[bool, str, TruncationStrategy]`, *optional*): Truncation strategy or boolean for truncating sequences.
            max_length (`Optional[int]`, *optional*): Maximum length of the sequences after processing.
            stride (`int`, *optional*): Stride to use when overflowing tokens.
            pad_to_multiple_of (`Optional[int]`, *optional*): Pad to a multiple of this value.
            return_tensors (`Optional[Union[str, TensorType]]`, *optional*): Return type of tensors (e.g., 'tf', 'pt').
            return_token_type_ids (`Optional[bool]`, *optional*): Whether to return token type IDs.
            return_attention_mask (`Optional[bool]`, *optional*): Whether to return attention mask.
            return_overflowing_tokens (`bool`, *optional*): Whether to return overflowing tokens.
            return_special_tokens_mask (`bool`, *optional*): Whether to return special tokens mask.
            return_offsets_mapping (`bool`, *optional*): Whether to return offsets mapping.
            return_length (`bool`, *optional*): Whether to return sequence lengths.
            verbose (`bool`, *optional*): Whether to print verbose logs.
            prepend_batch_axis (`bool`, *optional*): Whether to prepend batch axis to the returned tensors.
            **kwargs: Additional keyword arguments for specific implementations.

        Returns:
            `Dict[str, Union[torch.Tensor, tf.Tensor, np.ndarray]]`: Dictionary with model inputs prepared according to the specified arguments.
        """
        raise NotImplementedError

    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    ):
        """
        Truncate sequences of token IDs to a specified maximum length.

        Args:
            ids (`List[int]`): List of input token IDs.
            pair_ids (`Optional[List[int]]`, *optional*): List of token IDs for the second sequence in pair inputs.
            num_tokens_to_remove (`int`, *optional*): Number of tokens to remove from the sequences.
            truncation_strategy (`Union[str, TruncationStrategy]`, *optional*): Strategy for truncation ('longest_first' or 'only_first').
            stride (`int`, *optional*): Stride to use when overflowing tokens.
        """
        raise NotImplementedError

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ):
        """
        Internal method to pad encoded inputs to a specified length.

        Args:
            encoded_inputs (`Union[Dict[str, EncodedInput], BatchEncoding]`): Dictionary or BatchEncoding object containing encoded inputs.
            max_length (`Optional[int]`, *optional*): Maximum length to pad sequences to.
            padding_strategy (`PaddingStrategy`, *optional*): Strategy for padding sequences.
            pad_to_multiple_of (`Optional[int]`, *optional*): Pad to a multiple of this value.
            return_attention_mask (`Optional[bool]`, *optional*): Whether to return attention mask.

        Returns:
            `Dict[str, torch.Tensor]`: Dictionary containing padded inputs.
        """
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens into a single string representation.

        Args:
            tokens (`List[str]`): List of tokens to join.

        Returns:
            `str`: Joined string of tokens.
        """
        raise NotImplementedError

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ):
        """
        Batch decode sequences of token IDs into a list of strings.

        Args:
            sequences (`Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"]`): List or tensor of token IDs.
            skip_special_tokens (`bool`, *optional*): Whether to skip special tokens during decoding.
            clean_up_tokenization_spaces (`bool`, *optional*): Whether to clean up tokenization spaces in the decoded text.
            **kwargs: Additional keyword arguments for specific implementations.
        """
        raise NotImplementedError
    ) -> List[str]:
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `List[str]`: The list of decoded sentences.
        """
        # Return a list comprehension that decodes each sequence in `sequences`
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs,
            )
            for seq in sequences  # Iterate over each sequence in the input `sequences`
        ]

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids into a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        # Convert `token_ids` to Python list representation
        token_ids = to_py_obj(token_ids)

        # Call the internal decode method `_decode` with specified arguments
        return self._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,

    ) -> str:
        """
        Internal method to convert token ids into a string, with options to remove special tokens and clean up
        tokenization spaces.

        Args:
            token_ids (`Union[int, List[int]]`):
                List of tokenized input ids.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        # Return the result of decoding `token_ids` into a string using tokenizer and options
        return self.convert_tokens_to_string(
            self.convert_ids_to_tokens(token_ids),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
    ) -> str:
        raise NotImplementedError


# 抛出未实现错误，指示这个方法尚未被具体实现
def get_special_tokens_mask(
    self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
) -> List[int]:
    """
    Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
    special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

    Args:
        token_ids_0 (`List[int]`):
            List of ids of the first sequence.
        token_ids_1 (`List[int]`, *optional*):
            List of ids of the second sequence.
        already_has_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not the token list is already formatted with special tokens for the model.

    Returns:
        A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
    """
    assert already_has_special_tokens and token_ids_1 is None, (
        "You cannot use ``already_has_special_tokens=False`` with this tokenizer. "
        "Please use a slow (full python) tokenizer to activate this argument. "
        "Or set `return_special_tokens_mask=True` when calling the encoding method "
        "to get the special tokens mask in any tokenizer. "
    )

    # 从类属性中缓存所有特殊 token 的 id
    all_special_ids = self.all_special_ids

    # 创建一个特殊 tokens 掩码列表，标记哪些 token 是特殊 token
    special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]

    return special_tokens_mask

@staticmethod
def clean_up_tokenization(out_string: str) -> str:
    """
    Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

    Args:
        out_string (`str`): The text to clean up.

    Returns:
        `str`: The cleaned-up string.
    """
    # 清理简单的英文标记化残留，例如标点前的空格和缩写形式
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string
    def _eventual_warn_about_too_long_sequence(self, ids: List[int], max_length: Optional[int], verbose: bool):
        """
        根据输入和内部状态，可能会触发关于序列过长的警告，超过了模型指定的最大长度。

        Args:
            ids (`List[str]`): 标记化产生的 id 列表
            max_length (`int`, *optional*): 所需的最大长度（如果设置则不会触发警告）
            verbose (`bool`): 是否打印更多信息和警告。

        """
        # 如果 max_length 未设置且 ids 的长度超过 self.model_max_length 并且 verbose 为 True
        if max_length is None and len(ids) > self.model_max_length and verbose:
            # 如果尚未记录过这个警告，则记录该警告并打印日志
            if not self.deprecation_warnings.get("sequence-length-is-longer-than-the-specified-maximum", False):
                logger.warning(
                    "Token indices sequence length is longer than the specified maximum sequence length "
                    f"for this model ({len(ids)} > {self.model_max_length}). Running this sequence through the model "
                    "will result in indexing errors"
                )
            self.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True

    def _switch_to_input_mode(self):
        """
        将分词器切换到输入模式的私有方法（当分词器具有不同的输入/输出模式时）
        """
        pass

    def _switch_to_target_mode(self):
        """
        将分词器切换到目标模式的私有方法（当分词器具有不同的输入/输出模式时）
        """
        pass

    @contextmanager
    def as_target_tokenizer(self):
        """
        临时将分词器设置为用于编码目标的模式。对于需要为标签处理稍微不同的序列到序列模型的分词器非常有用。
        """
        warnings.warn(
            "`as_target_tokenizer` is deprecated and will be removed in v5 of Transformers. You can tokenize your "
            "labels by using the argument `text_target` of the regular `__call__` method (either in the same call as "
            "your input texts if you use the same keyword arguments, or in a separate call."
        )
        self._switch_to_target_mode()  # 切换到目标模式
        self._in_target_context_manager = True  # 设置目标模式上下文管理器为 True
        yield  # 返回控制权给调用方
        self._in_target_context_manager = False  # 取消目标模式上下文管理器
        self._switch_to_input_mode()  # 切换回输入模式

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoTokenizer"):
        """
        Register this class with a given auto class. This should only be used for custom tokenizers as the ones in the
        library are already mapped with `AutoTokenizer`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoTokenizer"`):
                The auto class to register this new tokenizer with.
        """
        # 检查 auto_class 是否不是字符串类型，如果不是，则取其类名作为字符串
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        # 导入 transformers 模块中的 auto 子模块
        import transformers.models.auto as auto_module

        # 检查 auto_module 中是否存在给定名称的 auto_class 类或模块
        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        # 将注册的 auto_class 赋值给类属性 _auto_class
        cls._auto_class = auto_class

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: str = "longest",
        return_tensors: str = None,
        truncation: bool = True,
        **kwargs,
"""
`prepare_seq2seq_batch` is deprecated and will be removed in version 5 of HuggingFace Transformers. Use the regular
`__call__` method to prepare your inputs and targets.

Here is a short example:

model_inputs = tokenizer(src_texts, text_target=tgt_texts, ...)

If you either need to use different keyword arguments for the source and target texts, you should do two calls like
this:

model_inputs = tokenizer(src_texts, ...)
labels = tokenizer(text_target=tgt_texts, ...)
model_inputs["labels"] = labels["input_ids"]

See the documentation of your specific tokenizer for more details on the specific arguments to the tokenizer of choice.
For a more complete example, see the implementation of `prepare_seq2seq_batch`.
"""
        # 发出未来警告，提示`prepare_seq2seq_batch`即将在 Transformers 版本 5 中移除
        warnings.warn(formatted_warning, FutureWarning)
        # mBART-specific kwargs that should be ignored by other models.
        # 移除仅适用于 mBART 的特定 kwargs，其他模型忽略这些参数
        kwargs.pop("src_lang", None)
        kwargs.pop("tgt_lang", None)
        # 如果未设置 max_length，则使用 self.model_max_length 的值
        if max_length is None:
            max_length = self.model_max_length
        # 调用当前对象（可能是模型）的 `__call__` 方法，准备模型的输入
        model_inputs = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        # 如果目标文本（tgt_texts）为 None，则返回模型输入
        if tgt_texts is None:
            return model_inputs
        # 处理目标文本（tgt_texts）
        # 如果未设置 max_target_length，则使用 max_length 的值
        if max_target_length is None:
            max_target_length = max_length
        # 使用当前对象的目标专用分词器上下文处理目标文本（tgt_texts）
        with self.as_target_tokenizer():
            labels = self(
                tgt_texts,
                add_special_tokens=True,
                return_tensors=return_tensors,
                padding=padding,
                max_length=max_target_length,
                truncation=truncation,
                **kwargs,
            )
        # 将标签（labels）的输入 ID 添加到模型输入字典中的 "labels" 键
        model_inputs["labels"] = labels["input_ids"]
        # 返回模型输入字典
        return model_inputs


def get_fast_tokenizer_file(tokenization_files: List[str]) -> str:
    """
    Get the tokenization file to use for this version of transformers.

    Args:
        tokenization_files (`List[str]`): The list of available configuration files.

    Returns:
        `str`: The tokenization file to use.
    """
    # 初始化一个空字典，用于存储文件名与版本号的映射关系
    tokenizer_files_map = {}
    # 遍历提供的 tokenization_files 列表
    for file_name in tokenization_files:
        # 使用正则表达式搜索文件名中的版本号信息
        search = _re_tokenizer_file.search(file_name)
        # 如果找到版本号信息
        if search is not None:
            # 提取版本号并作为字典的键，文件名作为值存储
            v = search.groups()[0]
            tokenizer_files_map[v] = file_name
    # 对版本号进行排序
    available_versions = sorted(tokenizer_files_map.keys())

    # 默认使用 FULL_TOKENIZER_FILE，然后尝试查找一些更新的版本
    tokenizer_file = FULL_TOKENIZER_FILE
    # 解析当前 transformers 版本号
    transformers_version = version.parse(__version__)
    # 遍历可用版本号列表
    for v in available_versions:
        # 如果当前版本号小于或等于 transformers 版本号
        if version.parse(v) <= transformers_version:
            # 更新 tokenizer_file 为对应版本号的文件名
            tokenizer_file = tokenizer_files_map[v]
        else:
            # 因为版本号已经排序，无需继续查找更高版本
            # 在此处退出循环
            break

    # 返回确定的 tokenizer 文件名
    return tokenizer_file
# 将 push_to_hub 方法的文档字符串更新，需要先复制该方法，以避免改变原始的文档字符串。
PreTrainedTokenizerBase.push_to_hub = copy_func(PreTrainedTokenizerBase.push_to_hub)

# 检查复制后的 push_to_hub 方法是否有文档字符串，如果有，则格式化其文档字符串，
# 将其中的占位符替换为指定的对象、对象类别和对象文件描述信息。
if PreTrainedTokenizerBase.push_to_hub.__doc__ is not None:
    PreTrainedTokenizerBase.push_to_hub.__doc__ = PreTrainedTokenizerBase.push_to_hub.__doc__.format(
        object="tokenizer", object_class="AutoTokenizer", object_files="tokenizer files"
    )
```