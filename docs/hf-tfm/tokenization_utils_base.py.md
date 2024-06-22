# `.\transformers\tokenization_utils_base.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有 2020 年 HuggingFace Inc. 团队
#
# 根据 Apache 许可证版本 2.0 进行许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 将按"原样"分发，不提供任何形式的担保或
# 条件，无论是明示的还是暗示的。
# 有关详细信息，请参见许可证。
"""
用于慢速和快速标记化类共有的基类：PreTrainedTokenizerBase（承载所有用户可见的编码方法）、
特殊标记混合（承载特殊标记逻辑）和 BatchEncoding（用于包装输出字典的特殊方法，适用于快速标记化器）
"""

# 导入模块
import copy  # 用于深拷贝对象
import json  # 用于 JSON 数据的处理
import os  # 用于操作文件和目录路径
import re  # 用于正则表达式的操作
import warnings  # 用于警告信息的处理
from collections import UserDict  # 用于创建自定义字典类型
from collections.abc import Mapping, Sized  # 用于抽象基类的支持
from contextlib import contextmanager  # 用于上下文管理器的创建
from dataclasses import dataclass  # 用于创建数据类
from functools import lru_cache  # 用于缓存函数的结果
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union  # 用于类型提示

import numpy as np  # 导入 NumPy 库
from packaging import version  # 用于处理版本号

# 导入自定义模块和工具函数
from . import __version__  # 导入当前模块的版本信息
from .dynamic_module_utils import custom_object_save  # 导入自定义对象保存函数
from .utils import (  # 导入各种工具函数
    ExplicitEnum,  # 枚举类的实现
    PaddingStrategy,  # 填充策略的实现
    PushToHubMixin,  # 推送到 Hub 的混合类
    TensorType,  # 张量类型的定义
    add_end_docstrings,  # 添加文档字符串的装饰器
    add_model_info_to_auto_map,  # 向自动映射添加模型信息
    cached_file,  # 缓存文件的函数
    copy_func,  # 复制函数的工具函数
    download_url,  # 下载 URL 的函数
    extract_commit_hash,  # 提取提交哈希的函数
    is_flax_available,  # 判断 Flax 是否可用
    is_jax_tensor,  # 判断是否为 Jax 张量
    is_numpy_array,  # 判断是否为 NumPy 数组
    is_offline_mode,  # 判断是否为离线模式
    is_remote_url,  # 判断是否为远程 URL
    is_tf_available,  # 判断是否为 TensorFlow 可用
    is_tf_tensor,  # 判断是否为 TensorFlow 张量
    is_tokenizers_available,  # 判断是否为 Tokenizers 可用
    is_torch_available,  # 判断是否为 PyTorch 可用
    is_torch_device,  # 判断是否为 PyTorch 设备
    is_torch_tensor,  # 判断是否为 PyTorch 张量
    logging,  # 日志记录工具
    requires_backends,  # 要求后端的装饰器
    to_py_obj,  # 转换为 Python 对象的函数
)

# 如果类型检查可用，则导入相应的模块
if TYPE_CHECKING:
    if is_torch_available():
        import torch  # 导入 PyTorch 模块
    if is_tf_available():
        import tensorflow as tf  # 导入 TensorFlow 模块
    if is_flax_available():
        import jax.numpy as jnp  # 导入 Jax 模块

    from .pipelines.conversational import Conversation  # 导入对话管道

# 如果 Tokenizers 可用，则导入相应的模块和类
if is_tokenizers_available():
    from tokenizers import AddedToken  # 导入 AddedToken 类
    from tokenizers import Encoding as EncodingFast  # 导入 Encoding 类
else:

    @dataclass(frozen=False, eq=True)  # 数据类的装饰器
    # 定义一个类 AddedToken，表示要添加到 Tokenizer 中的一个 token
    # AddedToken 可以具有特殊选项，定义它的行为方式
    # 如果未指定，`normalized` 将默认为 `not special`，类似于 `tokenizers` 中的定义
    class AddedToken:
        
        def __init__(
            self, content: str, single_word=False, lstrip=False, rstrip=False, special=False, normalized=None
        ):
            # 初始化 AddedToken 对象的属性
            self.content = content
            self.single_word = single_word
            self.lstrip = lstrip
            self.rstrip = rstrip
            self.special = special
            # 如果 normalized 未指定，则默认为 not special
            self.normalized = normalized if normalized is not None else not special

        # 返回对象的状态
        def __getstate__(self):
            return self.__dict__

        # 返回对象的字符串表示
        def __str__(self):
            return self.content

    # 定义一个数据类 EncodingFast，这是一个虚拟类，因为没有 `tokenizers` 库，我们无法使用这些对象
    @dataclass
    class EncodingFast:
        pass
# 获取名为__name__的logger对象
logger = logging.get_logger(__name__)

# 定义一个非常大的整数，用于设置具有无限大小输入的模型的最大输入长度
VERY_LARGE_INTEGER = int(1e30)
# 定义一个大但略小于VERY_LARGE_INTEGER的整数，用于需要较大值但稍小于VERY_LARGE_INTEGER的情况
LARGE_INTEGER = int(1e20)

# 定义类型别名和命名元组
TextInput = str
PreTokenizedInput = List[str]
EncodedInput = List[int]
TextInputPair = Tuple[str, str]
PreTokenizedInputPair = Tuple[List[str], List[str]]
EncodedInputPair = Tuple[List[int], List[int]]

# 以前慢速的分词器保存在三个单独的文件中
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

# 快速分词器（由HuggingFace分词器库提供）可以保存在单个文件中
FULL_TOKENIZER_FILE = "tokenizer.json"
_re_tokenizer_file = re.compile(r"tokenizer\.(.*)\.json")

# 定义截断策略的枚举类
class TruncationStrategy(ExplicitEnum):
    """
    [`PreTrainedTokenizerBase.__call__`]中`truncation`参数的可能值。在IDE中进行制表完成时很有用。
    """
    ONLY_FIRST = "only_first"
    ONLY_SECOND = "only_second"
    LONGEST_FIRST = "longest_first"
    DO_NOT_TRUNCATE = "do_not_truncate"

# 定义原始字符串中的字符跨度的命名元组
class CharSpan(NamedTuple):
    """
    原始字符串中的字符跨度。

    Args:
        start (`int`): 原始字符串中第一个字符的索引。
        end (`int`): 原始字符串中最后一个字符之后的字符的索引。
    """
    start: int
    end: int

# 定义编码字符串（标记列表）中的标记跨度的命名元组
class TokenSpan(NamedTuple):
    """
    编码字符串（标记列表）中的标记跨度。

    Args:
        start (`int`): 跨度中第一个标记的索引。
        end (`int`): 跨度中最后一个标记之后的标记的索引。
    """
    start: int
    end: int

# 继承自python字典的BatchEncoding类，用于保存[`~tokenization_utils_base.PreTrainedTokenizerBase.__call__`]、
# [`~tokenization_utils_base.PreTrainedTokenizerBase.encode_plus`]和
# [`~tokenization_utils_base.PreTrainedTokenizerBase.batch_encode_plus`]方法的输出（标记、注意力掩码等）
class BatchEncoding(UserDict):
    """
    包含[`~tokenization_utils_base.PreTrainedTokenizerBase.__call__`]、
    [`~tokenization_utils_base.PreTrainedTokenizerBase.encode_plus`]和
    [`~tokenization_utils_base.PreTrainedTokenizerBase.batch_encode_plus`]方法的输出（标记、注意力掩码等）。

    该类派生自python字典，可以像字典一样使用。此外，该类还公开了从单词/字符空间到标记空间的映射实用程序方法。
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

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        encoding: Optional[Union[EncodingFast, Sequence[EncodingFast]]] = None,
        tensor_type: Union[None, str, TensorType] = None,
        prepend_batch_axis: bool = False,
        n_sequences: Optional[int] = None,
    ):
        # 调用父类的初始化方法
        super().__init__(data)

        # 如果 encoding 是 EncodingFast 类型，则转换为列表
        if isinstance(encoding, EncodingFast):
            encoding = [encoding]

        # 将 encoding 赋值给实例变量 _encodings
        self._encodings = encoding

        # 如果 n_sequences 为 None 并且 encoding 不为 None 且 encoding 长度不为 0
        if n_sequences is None and encoding is not None and len(encoding):
            # 获取第一个 encoding 的 n_sequences 赋值给实例变量 _n_sequences
            n_sequences = encoding[0].n_sequences

        # 将 n_sequences 赋值给实例变量 _n_sequences
        self._n_sequences = n_sequences

        # 调用 convert_to_tensors 方法将数据转换为张量
        self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)

    @property
    def n_sequences(self) -> Optional[int]:
        """
        `Optional[int]`: The number of sequences used to generate each sample from the batch encoded in this
        [`BatchEncoding`]. Currently can be one of `None` (unknown), `1` (a single sentence) or `2` (a pair of
        sentences)
        """
        # 返回实例变量 _n_sequences
        return self._n_sequences

    @property
    def is_fast(self) -> bool:
        """
        `bool`: Indicate whether this [`BatchEncoding`] was generated from the result of a [`PreTrainedTokenizerFast`]
        or not.
        """
        # 返回 _encodings 是否为 None 的布尔值
        return self._encodings is not None
    # 定义 __getitem__ 方法，用于获取对象的元素
    def __getitem__(self, item: Union[int, str]) -> Union[Any, EncodingFast]:
        """
        如果键是字符串，则返回与 `key` 关联的字典值（'input_ids'、'attention_mask'等）。

        如果键是整数，则获取索引为 `key` 的批量项目的 `tokenizers.Encoding`。

        如果键是切片，则返回与 `key` 关联的字典值（'input_ids'、'attention_mask'等），带有切片约束。
        """
        # 如果键是字符串，则返回对应的值
        if isinstance(item, str):
            return self.data[item]
        # 如果存在编码信息，则返回对应索引的编码信息
        elif self._encodings is not None:
            return self._encodings[item]
        # 如果键是切片，则返回带有切片约束的字典值
        elif isinstance(item, slice):
            return {key: self.data[key][item] for key in self.data.keys()}
        else:
            # 抛出 KeyError 异常，说明键无效
            raise KeyError(
                "Invalid key. Only three types of key are available: "
                "(1) string, (2) integers for backend Encoding, and (3) slices for data subsetting."
            )

    # 定义 __getattr__ 方法，用于获取对象的属性
    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            # 如果属性不存在，则抛出 AttributeError 异常
            raise AttributeError

    # 定义 __getstate__ 方法，用于获取对象的状态
    def __getstate__(self):
        return {"data": self.data, "encodings": self._encodings}

    # 定义 __setstate__ 方法，用于设置对象的状态
    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

        if "encodings" in state:
            self._encodings = state["encodings"]

    # 定义 keys 方法，返回对象数据的键集合
    def keys(self):
        return self.data.keys()

    # 定义 values 方法，返回对象数据的值集合
    def values(self):
        return self.data.values()

    # 定义 items 方法，返回对象数据的键值对集合
    def items(self):
        return self.data.items()

    # 从这里开始：
    # 仅适用于 HuggingFace tokenizers 库提供的快速（基于 Rust）分词器的扩展属性和方法

    # 定义 encodings 属性，返回编码信息列表
    @property
    def encodings(self) -> Optional[List[EncodingFast]]:
        """
        `Optional[List[tokenizers.Encoding]]`: The list all encodings from the tokenization process. Returns `None` if
        the input was tokenized through Python (i.e., not a fast) tokenizer.
        """
        return self._encodings

    # 定义 tokens 方法，返回给定批次索引处的令牌列表
    def tokens(self, batch_index: int = 0) -> List[str]:
        """
        返回给定批次索引处的令牌列表（在单词/子词拆分和转换为整数索引之前的输入字符串的子部分）（仅适用于快速分词器的输出）。

        Args:
            batch_index (`int`, *optional*, defaults to 0): 要访问的批次索引。

        Returns:
            `List[str]`: 该索引处的令牌列表。
        """
        # 如果没有编码信息，则抛出 ValueError 异常
        if not self._encodings:
            raise ValueError(
                "tokens() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        return self._encodings[batch_index].tokens
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
        # 如果没有_encodings，则抛出异常
        if not self._encodings:
            raise ValueError(
                "sequence_ids() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        # 返回指定索引处的序列 id
        return self._encodings[batch_index].sequence_ids

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
        # 如果没有_encodings，则抛出异常
        if not self._encodings:
            raise ValueError(
                "words() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        # 发出警告，建议使用word_ids()代替words()
        warnings.warn(
            "`BatchEncoding.words()` property is deprecated and should be replaced with the identical, "
            "but more self-explanatory `BatchEncoding.word_ids()` property.",
            FutureWarning,
        )
        # 返回指定索引处的单词 id
        return self.word_ids(batch_index)
    # 返回一个列表，将标记映射到初始句子中的实际单词，用于快速分词器
    def word_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        # 如果没有编码，则抛出值错误
        if not self._encodings:
            raise ValueError(
                "word_ids() is not available when using non-fast tokenizers (e.g. instance of a `XxxTokenizerFast`"
                " class)."
            )
        # 返回对应于每个标记的单词的列表
        return self._encodings[batch_index].word_ids
    
    # 获取给定标记表示的序列的索引
    def token_to_sequence(self, batch_or_token_index: int, token_index: Optional[int] = None) -> int:
        # 如果没有编码，则抛出值错误
        if not self._encodings:
            raise ValueError("token_to_sequence() is not available when using Python based tokenizers")
        # 如果提供了标记索引，则将批次索引设置为批次或标记索引
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        # 如果批次索引小于0，则将其设置为批次大小加上批次索引
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        # 如果标记索引小于0，则将其设置为序列长度加上标记索引
        if token_index < 0:
            token_index = self._seq_len + token_index
        # 返回输入序列中单词的索引
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

        if not self._encodings:
            raise ValueError("token_to_word() is not available when using Python based tokenizers")
        if token_index is not None:  # 如果 token_index 不为 None，则说明传入了 batch_index 和 token_index
            batch_index = batch_or_token_index  # 将 batch_or_token_index 视为 batch_index
        else:  # 如果 token_index 为 None，则说明只传入了 token_index，而 batch_index 默认为 0
            batch_index = 0
            token_index = batch_or_token_index  # 将 batch_or_token_index 视为 token_index
        if batch_index < 0:  # 处理负数索引，转换为正数索引
            batch_index = self._batch_size + batch_index
        if token_index < 0:  # 处理负数索引，转换为正数索引
            token_index = self._seq_len + token_index
        return self._encodings[batch_index].token_to_word(token_index)  # 调用 Encodings 对象的 token_to_word 方法获取单词索引

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

        # 检查是否存在编码
        if not self._encodings:
            raise ValueError("word_to_tokens() is not available when using Python based tokenizers")
        # 如果提供了 word_index，则将 batch_or_word_index 视为 batch_index
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            # 否则，将 batch_index 设为 0，将 batch_or_word_index 视为 word_index
            batch_index = 0
            word_index = batch_or_word_index
        # 如果 batch_index 或 word_index 为负数，则将其转换为对应正数索引
        if batch_index < 0:
            batch_index = self._batch_size + batch_index
        if word_index < 0:
            word_index = self._seq_len + word_index
        # 获取对应的编码对象中的单词到标记的跨度
        span = self._encodings[batch_index].word_to_tokens(word_index, sequence_index)
        # 返回 TokenSpan 对象，如果跨度为 None 则返回 None
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

        # 如果未初始化编码信息，则抛出错误
        if not self._encodings:
            raise ValueError("token_to_chars() is not available when using Python based tokenizers")
        # 如果指定了 token_index，则将 batch_index 设为 batch_or_token_index，否则设为 0
        if token_index is not None:
            batch_index = batch_or_token_index
        else:
            batch_index = 0
            token_index = batch_or_token_index
        # 获取 token 对应的字符跨度索引
        span_indices = self._encodings[batch_index].token_to_chars(token_index)

        # 返回字符跨度对象或 None（如果 token 不对应任何字符）
        return CharSpan(*span_indices) if span_indices is not None else None

    def char_to_token(
        self, batch_or_char_index: int, char_index: Optional[int] = None, sequence_index: int = 0
    ) -> int:
        """
        获取编码输出中原始字符串中字符所对应的标记的索引，针对批次的序列。

        可以通过以下方式调用：

        - 如果批次大小为1，则为 `self.char_to_token(char_index)`
        - 如果批次大小大于等于1，则为 `self.char_to_token(batch_index, char_index)`

        当输入序列被预先标记化（即用户定义了单词）时，此方法特别适用。在这种情况下，它允许轻松地将编码的标记与提供的标记化单词关联起来。

        Args:
            batch_or_char_index (`int`):
                批次中序列的索引。如果批次只包含一个序列，则可以是序列中单词的索引。
            char_index (`int`, *可选*):
                如果在 *batch_or_token_index* 中提供了批次索引，则可以是序列中单词的索引。
            sequence_index (`int`, *可选*，默认为0):
                如果批次中编码了序列对，则可以用于指定所提供的字符索引属于序列对中的哪个序列（0 或 1）。

        Returns:
            `int`: 标记的索引。
        """

        # 如果没有编码，则抛出 ValueError
        if not self._encodings:
            raise ValueError("char_to_token() is not available when using Python based tokenizers")
        # 如果 char_index 不为 None，则 batch_index 为 batch_or_char_index，否则 batch_index 为 0，char_index 为 batch_or_char_index
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        # 返回编码中 batch_index 对应序列的 char_index 对应字符的标记索引
        return self._encodings[batch_index].char_to_token(char_index, sequence_index)

    def word_to_chars(
        self, batch_or_word_index: int, word_index: Optional[int] = None, sequence_index: int = 0
    def word_to_chars(batch_or_word_index: int, word_index: Optional[int] = None, sequence_index: int = 0) -> CharSpan:
        """
        获取批次中给定单词在原始字符串中对应的字符跨度。

        字符跨度以 CharSpan 命名元组返回，具有以下结构：

        - start: 原始字符串中第一个字符的索引
        - end: 原始字符串中最后一个字符之后的索引

        可以调用方式有：

        - `self.word_to_chars(word_index)` 如果批次大小为 1
        - `self.word_to_chars(batch_index, word_index)` 如果批次大小大于或等于 1

        参数:
            batch_or_word_index (`int`):
                批次中序列的索引。如果批次只包含一个序列，则这可以是序列中单词的索引
            word_index (`int`, *可选*):
                如果在 *batch_or_token_index* 中提供了批次索引，则这可以是序列中单词的索引。
            sequence_index (`int`, *可选*, 默认为 0):
                如果批次中编码了一对序列，则可以用来指定提供的单词索引属于该对序列中的哪个序列（0 或 1）。

        返回:
            `CharSpan` 或 `List[CharSpan]`: 字符串中相关字符或字符的跨度。CharSpan 是具有以下结构的命名元组：

                - start: 原始字符串中与标记相关的第一个字符的索引
                - end: 原始字符串中与标记相关的最后一个字符之后的索引
        """

        if not self._encodings:
            raise ValueError("word_to_chars() 在使用基于 Python 的标记器时不可用")
        if word_index is not None:
            batch_index = batch_or_word_index
        else:
            batch_index = 0
            word_index = batch_or_word_index
        return CharSpan(*(self._encodings[batch_index].word_to_chars(word_index, sequence_index)))
    def char_to_word(self, batch_or_char_index: int, char_index: Optional[int] = None, sequence_index: int = 0) -> int:
        """
        获取批次中字符的原始字符串中对应的单词。

        可以调用方式：

        - 如果批次大小为1，则 `self.char_to_word(char_index)`
        - 如果批次大小大于1，则 `self.char_to_word(batch_index, char_index)`

        当输入序列以预先标记化的序列（即由用户定义的单词）提供时，此方法特别适用。在这种情况下，它允许轻松地将编码的标记与提供的标记化单词关联起来。

        参数:
            batch_or_char_index (`int`):
                批次中序列的索引。如果批次只包含一个序列，则可以是原始字符串中的字符索引。
            char_index (`int`, *可选*):
                如果在 *batch_or_token_index* 中提供了批次索引，则这可以是原始字符串中的字符索引。
            sequence_index (`int`, *可选*，默认为0):
                如果批次中编码了一对序列，可以用来指定提供的字符索引属于该对序列中的哪一个（0或1）。

        返回:
            `int` 或 `List[int]`: 关联的编码标记的索引或索引。
        """

        if not self._encodings:
            raise ValueError("char_to_word() is not available when using Python based tokenizers")
        if char_index is not None:
            batch_index = batch_or_char_index
        else:
            batch_index = 0
            char_index = batch_or_char_index
        return self._encodings[batch_index].char_to_word(char_index, sequence_index)

    def convert_to_tensors(
        self, tensor_type: Optional[Union[str, TensorType]] = None, prepend_batch_axis: bool = False
    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        """
        通过调用 `v.to(device)` 将所有值发送到设备（仅适用于 PyTorch）。

        参数:
            device (`str` 或 `torch.device`): 要将张量放置在的设备。

        返回:
            [`BatchEncoding`]: 修改后的相同实例。
        """
        requires_backends(self, ["torch"])

        # 这个检查捕获到类似 APEX 盲目地在模块的所有输入上调用 "to" 的情况
        # 否则，它将传递下去并将包含令牌索引的 LongTensor 转换为 HalfTensor
        if isinstance(device, str) or is_torch_device(device) or isinstance(device, int):
            self.data = {k: v.to(device=device) for k, v in self.data.items()}
        else:
            logger.warning(f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported.")
        return self
class SpecialTokensMixin:
    """
    特殊标记的混合类，被 [`PreTrainedTokenizer`] 和 [`PreTrainedTokenizerFast`] 继承，用于处理与特殊标记相关的特定行为。
    特别是，该类持有属性，可以用于以与模型无关的方式直接访问这些特殊标记，并允许设置和更新特殊标记。

    Args:
        bos_token (`str` or `tokenizers.AddedToken`, *optional*):
            代表句子开头的特殊标记。
        eos_token (`str` or `tokenizers.AddedToken`, *optional*):
            代表句子结尾的特殊标记。
        unk_token (`str` or `tokenizers.AddedToken`, *optional*):
            代表未知词的特殊标记。
        sep_token (`str` or `tokenizers.AddedToken`, *optional*):
            用于将同一输入中的两个不同句子分隔开的特殊标记（例如，BERT 中使用）。
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            用于使标记数组大小相同以进行批处理的特殊标记。然后将被注意机制或损失计算忽略。
        cls_token (`str` or `tokenizers.AddedToken`, *optional*):
            代表输入类别的特殊标记（例如，BERT 中使用）。
        mask_token (`str` or `tokenizers.AddedToken`, *optional*):
            代表被屏蔽的标记的特殊标记（例如，BERT 中用于掩码语言模型预训练目标）。
        additional_special_tokens (tuple or list of `str` or `tokenizers.AddedToken`, *optional*):
            附加特殊标记的元组或列表，它们将被标记为 `special`，这意味着如果 `skip_special_tokens` 设置为 `True`，它们将在解码时被跳过。
    """

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
    # 初始化函数，设置特殊标记的默认值和其他属性
    def __init__(self, verbose=False, **kwargs):
        # 初始化各种特殊标记为 None
        self._bos_token = None
        self._eos_token = None
        self._unk_token = None
        self._sep_token = None
        self._pad_token = None
        self._cls_token = None
        self._mask_token = None
        self._pad_token_type_id = 0
        self._additional_special_tokens = []
        self.verbose = verbose

        # 直接设置隐藏值以允许使用尚未在词汇表中的特殊标记进行初始化。对于序列化/反序列化是必要的
        # TODO 在某个时候清理这一点（可能通过切换到快速分词器来实现）

        # 遍历传入的关键字参数
        for key, value in kwargs.items():
            # 如果值为 None，则跳过
            if value is None:
                continue
            # 如果关键字在特殊标记属性中
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                # 如果关键字是 additional_special_tokens
                if key == "additional_special_tokens":
                    # 断言值是列表或元组
                    assert isinstance(value, (list, tuple)), f"Value {value} is not a list or tuple"
                    # 断言所有元素是字符串或 AddedToken 类型
                    assert all(
                        isinstance(t, (str, AddedToken)) for t in value
                    ), "One of the tokens is not a string or an AddedToken"
                    # 设置属性值
                    setattr(self, key, value)
                # 如果值是字符串或 AddedToken 类型
                elif isinstance(value, (str, AddedToken)):
                    # 设置属性值
                    setattr(self, key, value)
                else:
                    # 抛出类型错误
                    raise TypeError(f"Special token {key} has to be either str or AddedToken but got: {type(value)}")

    # 清理特殊标记，现在已弃用，保留向后兼容性，将在 transformers v5 中删除
    def sanitize_special_tokens(self) -> int:
        """
        The `sanitize_special_tokens` is now deprecated kept for backward compatibility and will be removed in
        transformers v5.
        """
        # 记录警告信息
        logger.warning_once("The `sanitize_special_tokens` will be removed in transformers v5.")
        # 添加所有扩展的特殊标记
        return self.add_tokens(self.all_special_tokens_extended, special_tokens=True)

    # 添加特殊标记
    def add_special_tokens(
        self, special_tokens_dict: Dict[str, Union[str, AddedToken]], replace_additional_special_tokens=True
    # 添加标记
    def add_tokens(
        self, new_tokens: Union[str, AddedToken, List[Union[str, AddedToken]]], special_tokens: bool = False
    ) -> int:
        """
        给分词器类添加一系列新的标记。如果新的标记不在词汇表中，则从当前词汇表的长度开始添加，并在应用标记化算法之前将它们隔离开来。因此，添加的标记和标记化算法的词汇表中的标记不会以相同的方式处理。

        注意，当向词汇表添加新标记时，您应确保调整模型的标记嵌入矩阵大小，使其与分词器匹配。

        为此，请使用 [`~PreTrainedModel.resize_token_embeddings`] 方法。

        参数:
            new_tokens (`str`, `tokenizers.AddedToken` 或 *str* 或 `tokenizers.AddedToken` 的列表):
                仅当标记尚未在词汇表中时才添加标记。`tokenizers.AddedToken` 封装了一个字符串标记，让您可以个性化其行为：这个标记是否只匹配单词，这个标记是否应该在左侧剥离所有潜在的空格，这个标记是否应该在右侧剥离所有潜在的空格等。
            special_tokens (`bool`, *可选*, 默认为 `False`):
                可以用来指定标记是否是特殊标记。这主要改变了标准化行为（例如，特殊标记如 CLS 或 [MASK] 通常不会被小写处理）。

                在 HuggingFace 分词器库中查看 `tokenizers.AddedToken` 的详细信息。

        返回:
            `int`: 添加到词汇表中的标记数。

        示例:

        ```python
        # 让我们看看如何扩展 Bert 模型和分词器的词汇表
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("我们已经添加了", num_added_toks, "个标记")
        # 注意：resize_token_embeddings 期望接收新词汇表的完整大小，即分词器的长度。
        model.resize_token_embeddings(len(tokenizer))
        ```py"""
        if not new_tokens:
            return 0

        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]

        return self._add_tokens(new_tokens, special_tokens=special_tokens)

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        raise NotImplementedError

    @property
    def bos_token(self) -> str:
        """
        `str`: Beginning of sentence token. Log an error if used while not having been set.
        """
        # 如果未设置起始符号，则记录错误并返回 None
        if self._bos_token is None:
            if self.verbose:
                logger.error("Using bos_token, but it is not set yet.")
            return None
        return str(self._bos_token)

    @property
    def eos_token(self) -> str:
        """
        `str`: End of sentence token. Log an error if used while not having been set.
        """
        # 如果未设置结束符号，则记录错误并返回 None
        if self._eos_token is None:
            if self.verbose:
                logger.error("Using eos_token, but it is not set yet.")
            return None
        return str(self._eos_token)

    @property
    def unk_token(self) -> str:
        """
        `str`: Unknown token. Log an error if used while not having been set.
        """
        # 如果未设置未知符号，则记录错误并返回 None
        if self._unk_token is None:
            if self.verbose:
                logger.error("Using unk_token, but it is not set yet.")
            return None
        return str(self._unk_token)

    @property
    def sep_token(self) -> str:
        """
        `str`: Separation token, to separate context and query in an input sequence. Log an error if used while not
        having been set.
        """
        # 如果未设置分隔符，则记录错误并返回 None
        if self._sep_token is None:
            if self.verbose:
                logger.error("Using sep_token, but it is not set yet.")
            return None
        return str(self._sep_token)

    @property
    def pad_token(self) -> str:
        """
        `str`: Padding token. Log an error if used while not having been set.
        """
        # 如果未设置填充符号，则记录错误并返回 None
        if self._pad_token is None:
            if self.verbose:
                logger.error("Using pad_token, but it is not set yet.")
            return None
        return str(self._pad_token)

    @property
    def cls_token(self) -> str:
        """
        `str`: Classification token, to extract a summary of an input sequence leveraging self-attention along the full
        depth of the model. Log an error if used while not having been set.
        """
        # 如果未设置分类符号，则记录错误并返回 None
        if self._cls_token is None:
            if self.verbose:
                logger.error("Using cls_token, but it is not set yet.")
            return None
        return str(self._cls_token)

    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.
        """
        # 如果未设置掩码符号，则记录错误并返回 None
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @property
    def additional_special_tokens(self) -> List[str]:
        """
        `List[str]`: 返回所有可能使用的额外特殊标记。如果尚未设置，则记录错误。
        """
        # 如果未设置额外特殊标记，记录错误并返回 None
        if self._additional_special_tokens is None:
            if self.verbose:
                logger.error("Using additional_special_tokens, but it is not set yet.")
            return None
        # 返回额外特殊标记的字符串形式列表
        return [str(tok) for tok in self._additional_special_tokens]

    @bos_token.setter
    def bos_token(self, value):
        # 如果值不是字符串或 AddedToken 类型，并且不为 None，则引发 ValueError
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the BOS token")
        # 设置 BOS（Beginning of Sentence）标记
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        # 如果值不是字符串或 AddedToken 类型，并且不为 None，则引发 ValueError
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the EOS token")
        # 设置 EOS（End of Sentence）标记
        self._eos_token = value

    @unk_token.setter
    def unk_token(self, value):
        # 如果值不是字符串或 AddedToken 类型，并且不为 None，则引发 ValueError
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the UNK token")
        # 设置 UNK（Unknown）标记
        self._unk_token = value

    @sep_token.setter
    def sep_token(self, value):
        # 如果值不是字符串或 AddedToken 类型，并且不为 None，则引发 ValueError
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the SEP token")
        # 设置 SEP（Separator）标记
        self._sep_token = value

    @pad_token.setter
    def pad_token(self, value):
        # 如果值不是字符串或 AddedToken 类型，并且不为 None，则引发 ValueError
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the PAD token")
        # 设置 PAD（Padding）标记
        self._pad_token = value

    @cls_token.setter
    def cls_token(self, value):
        # 如果值不是字符串或 AddedToken 类型，并且不为 None，则引发 ValueError
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the CLS token")
        # 设置 CLS（Classification）标记
        self._cls_token = value

    @mask_token.setter
    def mask_token(self, value):
        # 如果值不是字符串或 AddedToken 类型，并且不为 None，则引发 ValueError
        if not isinstance(value, (str, AddedToken)) and value is not None:
            raise ValueError("Cannot set a non-string value as the MASK token")
        # 设置 MASK（Masking）标记
        self._mask_token = value

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        # 设置额外特殊标记，如果值为 None，则设置为 None
        self._additional_special_tokens = value if value is not None else None

    @property
    def bos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: 词汇表中句子开头标记的 ID。如果标记未设置，则返回 `None`。
        """
        # 如果 BOS（Beginning of Sentence）标记未设置，则返回 None
        if self._bos_token is None:
            return None
        # 将 BOS 标记转换为对应的 ID
        return self.convert_tokens_to_ids(self.bos_token)

    @property
    def eos_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: 词汇表中句子结尾标记的 ID。如果标记未设置，则返回 `None`。
        """
        # 如果 EOS（End of Sentence）标记未设置，则返回 None
        if self._eos_token is None:
            return None
        # 将 EOS 标记转换为对应的 ID
        return self.convert_tokens_to_ids(self.eos_token)
    @property
    def unk_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the unknown token in the vocabulary. Returns `None` if the token has not been set.
        """
        # 如果未设置未知标记，则返回 None
        if self._unk_token is None:
            return None
        # 将未知标记转换为对应的 id
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
        # 将分隔标记转换为对应的 id
        return self.convert_tokens_to_ids(self.sep_token)

    @property
    def pad_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        """
        # 如果未设置填充标记，则返回 None
        if self._pad_token is None:
            return None
        # 将填充标记转换为对应的 id
        return self.convert_tokens_to_ids(self.pad_token)

    @property
    def pad_token_type_id(self) -> int:
        """
        `int`: Id of the padding token type in the vocabulary.
        """
        # 返回填充标记类型的 id
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
        # 将分类标记转换为对应的 id
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
        # 将掩码标记转换为对应的 id
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
        # 设置开始标记的 id
        self._bos_token = self.convert_ids_to_tokens(value) if value is not None else None

    @eos_token_id.setter
    def eos_token_id(self, value):
        # 设置结束标记的 id
        self._eos_token = self.convert_ids_to_tokens(value) if value is not None else None

    @unk_token_id.setter
    def unk_token_id(self, value):
        # 设置未知标记的 id
        self._unk_token = self.convert_ids_to_tokens(value) if value is not None else None

    @sep_token_id.setter
    def sep_token_id(self, value):
        # 设置分隔标记的 id
        self._sep_token = self.convert_ids_to_tokens(value) if value is not None else None

    @pad_token_id.setter
    # 设置填充标记的ID，并将其转换为标记
    def pad_token_id(self, value):
        self._pad_token = self.convert_ids_to_tokens(value) if value is not None else None

    # 设置类别标记的ID，并将其转换为标记
    @cls_token_id.setter
    def cls_token_id(self, value):
        self._cls_token = self.convert_ids_to_tokens(value) if value is not None else None

    # 设置掩码标记的ID，并将其转换为标记
    @mask_token_id.setter
    def mask_token_id(self, value):
        self._mask_token = self.convert_ids_to_tokens(value) if value is not None else None

    # 设置额外特殊标记的ID列表，并将其转换为标记
    @additional_special_tokens_ids.setter
    def additional_special_tokens_ids(self, values):
        self._additional_special_tokens = [self.convert_ids_to_tokens(value) for value in values]

    # 返回特殊标记的映射，将特殊标记的类属性映射到其值
    @property
    def special_tokens_map(self) -> Dict[str, Union[str, List[str]]]:
        """
        `Dict[str, Union[str, List[str]]]`: A dictionary mapping special token class attributes (`cls_token`,
        `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).

        Convert potential tokens of `tokenizers.AddedToken` type to string.
        """
        set_attr = {}
        # 遍历所有特殊标记属性，将其值添加到映射中
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    # 返回扩展的特殊标记映射，将特殊标记的类属性映射到其值，不将`tokenizers.AddedToken`类型的标记转换为字符串
    @property
    def special_tokens_map_extended(self) -> Dict[str, Union[str, AddedToken, List[Union[str, AddedToken]]]]:
        """
        `Dict[str, Union[str, tokenizers.AddedToken, List[Union[str, tokenizers.AddedToken]]]]`: A dictionary mapping
        special token class attributes (`cls_token`, `unk_token`, etc.) to their values (`'<unk>'`, `'<cls>'`, etc.).

        Don't convert tokens of `tokenizers.AddedToken` type to string so they can be used to control more finely how
        special tokens are tokenized.
        """
        set_attr = {}
        # 遍历所有特殊标记属性，将其值添加到映射中
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    # 返回属性
    @property
    def all_special_tokens_extended(self) -> List[Union[str, AddedToken]]:
        """
        `List[Union[str, tokenizers.AddedToken]]`: 返回所有特殊标记（例如`'<unk>'`，`'<cls>'`等），它们的顺序与每个标记的索引无关。
        如果想知道正确的索引，可以查看`self.added_tokens_encoder`。由于键是`AddedTokens`而不是`Strings`，我们无法再创建顺序了。

        不要将`tokenizers.AddedToken`类型的标记转换为字符串，这样可以更精细地控制特殊标记的标记化方式。
        """
        all_tokens = []  # 初始化一个空列表来存储所有特殊标记
        seen = set()  # 使用集合来跟踪已经遍历过的特殊标记
        for value in self.special_tokens_map_extended.values():  # 遍历扩展特殊标记映射的值
            if isinstance(value, (list, tuple)):  # 如果值是列表或元组
                tokens_to_add = [token for token in value if str(token) not in seen]  # 获取未见过的特殊标记
            else:
                tokens_to_add = [value] if str(value) not in seen else []  # 如果值不是列表或元组，检查是否为未见过的特殊标记
            seen.update(map(str, tokens_to_add))  # 更新已经遍历过的特殊标记集合
            all_tokens.extend(tokens_to_add)  # 将未见过的特殊标记添加到结果列表中
        return all_tokens  # 返回所有特殊标记列表

    @property
    def all_special_tokens(self) -> List[str]:
        """
        `List[str]`: 返回唯一特殊标记的列表（例如`'<unk>'`，`'<cls>'`等）。

        将`tokenizers.AddedToken`类型的标记转换为字符串。
        """
        all_toks = [str(s) for s in self.all_special_tokens_extended]  # 将所有特殊标记扩展列表中的标记转换为字符串
        return all_toks  # 返回所有特殊标记的字符串列表

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: 列出映射到类属性的特殊标记的标记ID（例如`'<unk>'`，`'<cls>'`等）。
        """
        all_toks = self.all_special_tokens  # 获取所有特殊标记的字符串列表
        all_ids = self.convert_tokens_to_ids(all_toks)  # 将所有特殊标记转换为对应的标记ID
        return all_ids  # 返回所有特殊标记的标记ID列表
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


@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class PreTrainedTokenizerBase(SpecialTokensMixin, PushToHubMixin):
    """
    Base class for [`PreTrainedTokenizer`] and [`PreTrainedTokenizerFast`].

    Handles shared (mostly boiler plate) methods for those two classes.
    """

    vocab_files_names: Dict[str, str] = {}
    pretrained_vocab_files_map: Dict[str, Dict[str, str]] = {}
    pretrained_init_configuration: Dict[str, Dict[str, Any]] = {}
    max_model_input_sizes: Dict[str, Optional[int]] = {}
    _auto_class: Optional[str] = None

    # first name has to correspond to main model input name
    # to make sure `tokenizer.pad(...)` works correctly
    model_input_names: List[str] = ["input_ids", "token_type_ids", "attention_mask"]
    padding_side: str = "right"
    truncation_side: str = "right"
    slow_tokenizer_class = None
    # 初始化方法，接受关键字参数
    def __init__(self, **kwargs):
        # 用于保存和重新加载的输入和关键字参数（参见“from_pretrained”和“save_pretrained”）
        self.init_inputs = ()
        self.init_kwargs = copy.deepcopy(kwargs)  # 深度拷贝关键字参数
        self.name_or_path = kwargs.pop("name_or_path", "")  # 弹出'name_or_path'键的值，默认为空字符串
        self._processor_class = kwargs.pop("processor_class", None)  # 弹出'processor_class'键的值，默认为None

        # 对于向后兼容性，如果提供了max_len，则从max_len中设置model_max_length
        model_max_length = kwargs.pop("model_max_length", kwargs.pop("max_len", None))
        self.model_max_length = model_max_length if model_max_length is not None else VERY_LARGE_INTEGER

        # 填充和截断默认是右边的，在子类中被覆盖。如果在kwargs中指定了，它会被修改。
        self.padding_side = kwargs.pop("padding_side", self.padding_side)
        if self.padding_side not in ["right", "left"]:
            raise ValueError(
                f"Padding side should be selected between 'right' and 'left', current value: {self.padding_side}"
            )

        self.truncation_side = kwargs.pop("truncation_side", self.truncation_side)
        if self.truncation_side not in ["right", "left"]:
            raise ValueError(
                f"Padding side should be selected between 'right' and 'left', current value: {self.truncation_side}"
            )

        self.model_input_names = kwargs.pop("model_input_names", self.model_input_names)  # 弹出'model_input_names'键的值，默认为self.model_input_names

        # 默认情况下，对于快速和慢速标记器，清理标记化空格
        self.clean_up_tokenization_spaces = kwargs.pop("clean_up_tokenization_spaces", True)

        # 默认情况下，不为快速和慢速标记器拆分特殊标记
        self.split_special_tokens = kwargs.pop("split_special_tokens", False)

        self.deprecation_warnings = {}  # 用于存储我们已经注意到过时警告的时间（避免过度记录）
        self._in_target_context_manager = False

        # 存储一个Jinja模板，格式化聊天历史为可标记化的字符串
        self.chat_template = kwargs.pop("chat_template", None)

        super().__init__(**kwargs)  # 调用父类的初始化方法，传递剩余的关键字参数

    @property
    def max_len_single_sentence(self) -> int:
        """
        `int`: 可以输入模型的句子的最大长度。
        """
        return self.model_max_length - self.num_special_tokens_to_add(pair=False)  # 返回单个句子的最大长度

    @property
    def max_len_sentences_pair(self) -> int:
        """
        `int`: 可以输入模型的一对句子的最大组合长度。
        """
        return self.model_max_length - self.num_special_tokens_to_add(pair=True)  # 返回句子对的最大长度

    @max_len_single_sentence.setter
    # 计算单个句子的最大长度，用于设置 'max_len_single_sentence'
    def max_len_single_sentence(self, value) -> int:
        # 对于向后兼容性，允许尝试设置 'max_len_single_sentence'
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=False) and self.verbose:
            # 如果设置的值等于模型最大长度减去非成对特殊标记的数量，并且 verbose 为 True
            if not self.deprecation_warnings.get("max_len_single_sentence", False):
                # 如果之前没有警告过 'max_len_single_sentence'
                logger.warning(
                    "Setting 'max_len_single_sentence' is now deprecated. This value is automatically set up."
                )
            self.deprecation_warnings["max_len_single_sentence"] = True
        else:
            # 否则，抛出 ValueError，指明 'max_len_single_sentence' 设置已经被弃用
            raise ValueError(
                "Setting 'max_len_single_sentence' is now deprecated. This value is automatically set up."
            )

    # 设置 'max_len_sentences_pair' 的属性
    @max_len_sentences_pair.setter
    def max_len_sentences_pair(self, value) -> int:
        # 对于向后兼容性，允许尝试设置 'max_len_sentences_pair'
        if value == self.model_max_length - self.num_special_tokens_to_add(pair=True) and self.verbose:
            # 如果设置的值等于模型最大长度减去成对特殊标记的数量，并且 verbose 为 True
            if not self.deprecation_warnings.get("max_len_sentences_pair", False):
                # 如果之前没有警告过 'max_len_sentences_pair'
                logger.warning(
                    "Setting 'max_len_sentences_pair' is now deprecated. This value is automatically set up."
                )
            self.deprecation_warnings["max_len_sentences_pair"] = True
        else:
            # 否则，抛出 ValueError，指明 'max_len_sentences_pair' 设置已经被弃用
            raise ValueError("Setting 'max_len_sentences_pair' is now deprecated. This value is automatically set up.")

    # 设置处理器类别作为属性
    def _set_processor_class(self, processor_class: str):
        """Sets processor class as an attribute."""
        self._processor_class = processor_class

    # 返回添加的特殊标记的解码器
    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        raise NotImplementedError()

    # 返回类的字符串表示形式
    def __repr__(self) -> str:
        # 生成类的字符串表示形式，包括名称或路径、词汇量、模型最大长度、是否为快速模式、填充位置、截断位置、特殊标记等信息
        added_tokens_decoder_rep = "\n\t".join([f"{k}: {v.__repr__()}," for k, v in self.added_tokens_decoder.items()])
        return (
            f"{self.__class__.__name__}(name_or_path='{self.name_or_path}',"
            f" vocab_size={self.vocab_size}, model_max_length={self.model_max_length}, is_fast={self.is_fast},"
            f" padding_side='{self.padding_side}', truncation_side='{self.truncation_side}',"
            f" special_tokens={self.special_tokens_map}, clean_up_tokenization_spaces={self.clean_up_tokenization_spaces}), "
            " added_tokens_decoder={\n\t" + added_tokens_decoder_rep + "\n}"
        )

    # 返回对象的长度
    def __len__(self) -> int:
        raise NotImplementedError()

    # 返回词汇表，作为标记到索引的字典
    def get_vocab(self) -> Dict[str, int]:
        """
        Returns the vocabulary as a dictionary of token to index.

        `tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the
        vocab.

        Returns:
            `Dict[str, int]`: The vocabulary.
        """
        raise NotImplementedError()
    # 应用聊天模板到对话数据中，格式化输出
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
        **tokenizer_kwargs,
    ):
    # 使用LRU缓存，优化已编译的Jinja模板
    @lru_cache
    def _compile_jinja_template(self, chat_template):
        # 尝试导入必要的Jinja2库
        try:
            import jinja2
            from jinja2.exceptions import TemplateError
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            # 抛出导入错误，如果Jinja2库未安装
            raise ImportError("apply_chat_template requires jinja2 to be installed.")
        # 检查Jinja2版本是否符合要求
        if version.parse(jinja2.__version__) <= version.parse("3.0.0"):
            raise ImportError(
                "apply_chat_template requires jinja2>=3.0.0 to be installed. Your version is " f"{jinja2.__version__}."
            )
        # 定义用于引发异常的函数
        def raise_exception(message):
            raise TemplateError(message)
        # 创建Jinja2环境
        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        # 从字符串编译Jinja模板
        return jinja_env.from_string(chat_template)

    @property
    # 默认聊天模板，用于格式化输入成标准的ChatML格式
    def default_chat_template(self):
        """
        This template formats inputs in the standard ChatML format. See
        https://github.com/openai/openai-python/blob/main/chatml.md
        """
        # 发出警告，说明默认聊天模板正在使用，并提供如何更改的链接
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using a default chat template "
            "that implements the ChatML format (without BOS/EOS tokens!). If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回默认聊天模板
        return (
            "{% for message in messages %}"
            "{{'' + message['role'] + '\n' + message['content'] + '' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ 'assistant\n' }}"
            "{% endif %}"
        )

    @classmethod
    # 从预训练模型中加载模型和配置
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *init_inputs,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
    @classmethod
    # 内部方法：从预训练模型加载模型和配置
    def _from_pretrained(
        cls,
        resolved_vocab_files,
        pretrained_model_name_or_path,
        init_configuration,
        *init_inputs,
        token=None,
        cache_dir=None,
        local_files_only=False,
        _commit_hash=None,
        _is_local=False,
        **kwargs,
    ):
    # 静态方法
    @staticmethod
    # 定义一个方法，用于在 Transformers v5 中删除
    def _eventually_correct_t5_max_length(pretrained_model_name_or_path, max_model_length, init_max_model_length):
        # 该方法的唯一目的是可能引发警告
        # 指出使用了错误定义的 T5 分词器的最大长度
        # 我们将在 Transformers v5 中进行更正
        return max_model_length

    @classmethod
    def convert_added_tokens(cls, obj: Union[AddedToken, Any], save=False, add_type_field=True):
        # 如果对象是字典且包含 "__type" 键且值为 "AddedToken"
        if isinstance(obj, dict) and "__type" in obj and obj["__type"] == "AddedToken":
            obj.pop("__type")
            return AddedToken(**obj)
        # 如果对象是 AddedToken 类型且需要保存
        if isinstance(obj, AddedToken) and save:
            obj = obj.__getstate__()
            # 如果需要添加类型字段
            if add_type_field:
                obj["__type"] = "AddedToken"
            else:
                # 不保存 "special" 字段给之前的分词器
                obj.pop("special")
            return obj
        # 如果对象是列表或元组
        elif isinstance(obj, (list, tuple)):
            return [cls.convert_added_tokens(o, save=save, add_type_field=add_type_field) for o in obj]
        # 如果对象是字典
        elif isinstance(obj, dict):
            return {k: cls.convert_added_tokens(v, save=save, add_type_field=add_type_field) for k, v in obj.items()}
        # 其他情况直接返回对象
        return obj

    # 保存预训练模型
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    # 保存预训练模型的内部方法
    def _save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        file_names: Tuple[str],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        """
        Save a tokenizer using the slow-tokenizer/legacy format: vocabulary + added tokens.

        Fast tokenizers can also be saved in a unique JSON file containing {config + vocab + added-tokens} using the
        specific [`~tokenization_utils_fast.PreTrainedTokenizerFast._save_pretrained`]
        """
        # 如果 legacy_format 不是 True，抛出数值错误
        if legacy_format is False:
            raise ValueError(
                "Only fast tokenizers (instances of PreTrainedTokenizerFast) can be saved in non legacy format."
            )

        # 将保存目录转换为字符串类型
        save_directory = str(save_directory)

        # 组合添加的标记文件名
        added_tokens_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + ADDED_TOKENS_FILE
        )
        # 获取添加的词汇表，索引大于等于 vocab_size 的标记
        added_vocab = {tok: index for tok, index in self.added_tokens_encoder.items() if index >= self.vocab_size}
        # 如果存在添加的词汇，将其保存到文件中
        if added_vocab:
            with open(added_tokens_file, "w", encoding="utf-8") as f:
                out_str = json.dumps(added_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                f.write(out_str)
                logger.info(f"added tokens file saved in {added_tokens_file}")

        # 保存词汇表文件，并获取文件名列表
        vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)

        # 返回保存的文件名列表以及添加的标记文件名
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
        # 抛出未实现错误，表示该方法需要在子类中被重写实现
        raise NotImplementedError
    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        """
        Converts a string into a sequence of tokens, replacing unknown tokens with the `unk_token`.

        Args:
            text (`str`):
                The sequence to be encoded.
            pair (`str`, *optional*):
                A second sequence to be encoded with the first.
            add_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to add the special tokens associated with the corresponding model.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific encode method. See details in
                [`~PreTrainedTokenizerBase.__call__`]

        Returns:
            `List[str]`: The list of tokens.
        """
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
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Same as doing `self.convert_tokens_to_ids(self.tokenize(text))`.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """
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

        return encoded_inputs["input_ids"]
    # 返回需要添加的特殊标记数量，如果是在处理文本对，则考虑是否是一对文本
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        # 抛出未实现错误，该方法应该由子类实现
        raise NotImplementedError

    # 获取填充和截断策略的名称和参数
    def _get_padding_truncation_strategies(
        self, padding=False, truncation=None, max_length=None, pad_to_multiple_of=None, verbose=True, **kwargs
    # 为 __call__ 方法添加文档字符串，包括编码的参数文档字符串和额外的参数文档字符串
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        # 输入文本，可以是单个文本、预分词输入、文本列表或预分词输入列表
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        # 可选的第二个文本，适用于处理文本对
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        # 目标文本，用于生成目标编码序列
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        # 可选的第二个目标文本，适用于处理文本对
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        # 是否添加特殊标记
        add_special_tokens: bool = True,
        # 填充策略，可以是布尔值、字符串或填充策略对象
        padding: Union[bool, str, PaddingStrategy] = False,
        # 截断策略，可以是布尔值、字符串或截断策略对象
        truncation: Union[bool, str, TruncationStrategy] = None,
        # 最大长度限制
        max_length: Optional[int] = None,
        # 步长，用于分割长文本
        stride: int = 0,
        # 输入是否已经被分割成单词
        is_split_into_words: bool = False,
        # 填充到的长度的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回标记类型 ID
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的标记
        return_overflowing_tokens: bool = False,
        # 是否返回特殊标记掩码
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射
        return_offsets_mapping: bool = False,
        # 是否返回序列长度
        return_length: bool = False,
        # 是否打印详细信息
        verbose: bool = True,
        **kwargs,
    # 调用单个输入的编码方法，并添加文档字符串
    def _call_one(
        self,
        # 输入文本，可以是单个文本、预分词输入、文本列表或预分词输入列表
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        # 可选的第二个文本，适用于处理文本对
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        # 是否添加特殊标记
        add_special_tokens: bool = True,
        # 填充策略，可以是布尔值、字符串或填充策略对象
        padding: Union[bool, str, PaddingStrategy] = False,
        # 截断策略，可以是布尔值、字符串或截断策略对象
        truncation: Union[bool, str, TruncationStrategy] = None,
        # 最大长度限制
        max_length: Optional[int] = None,
        # 步长，用于分割长文本
        stride: int = 0,
        # 输入是否已经被分割成单词
        is_split_into_words: bool = False,
        # 填充到的长度的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回标记类型 ID
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的标记
        return_overflowing_tokens: bool = False,
        # 是否返回特殊标记掩码
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射
        return_offsets_mapping: bool = False,
        # 是否返回序列长度
        return_length: bool = False,
        # 是否打印详细信息
        verbose: bool = True,
        **kwargs,
    # 为_call_one方法添加文档字符串，包括编码的参数文档字符串和额外的参数文档字符串
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 文本编码方法，用于将文本转换成模型可接受的输入格式
    def encode_plus(
        self,
        # 输入的主要文本，可以是TextInput、PreTokenizedInput或EncodedInput类型之一
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        # 可选的文本对，用于处理文本对任务
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        # 是否在编码时添加特殊标记
        add_special_tokens: bool = True,
        # 是否进行填充
        padding: Union[bool, str, PaddingStrategy] = False,
        # 是否进行截断
        truncation: Union[bool, str, TruncationStrategy] = None,
        # 编码后的最大长度
        max_length: Optional[int] = None,
        # 滑动窗口步长
        stride: int = 0,
        # 输入是否已经被分词为单词
        is_split_into_words: bool = False,
        # 填充长度至某个数的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，如TensorFlow张量或PyTorch张量
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回token类型ID
        return_token_type_ids: Optional[bool] = None,
        # 是否返回attention mask
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的token
        return_overflowing_tokens: bool = False,
        # 是否返回特殊token的mask
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射
        return_offsets_mapping: bool = False,
        # 是否返回编码后的长度
        return_length: bool = False,
        # 是否打印详细信息
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

        # 获取填充和截断策略以及其他参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用 _encode_plus 方法对文本进行编码
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
    # 定义一个方法用于编码文本或文本对，并生成批量编码结果
    def _encode_plus(
        self,
        # 输入的文本或文本对，可以是文本输入、预分词输入或已编码输入的任意一种形式
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        # 可选参数：第二个文本或文本对，同样支持多种输入形式
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        # 是否添加特殊令牌，默认为 True
        add_special_tokens: bool = True,
        # 填充策略，默认为不填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 截断策略，默认为不截断
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        # 最大长度限制，默认为无限制
        max_length: Optional[int] = None,
        # 滑动窗口步长，默认为 0
        stride: int = 0,
        # 是否已分词，默认为 False
        is_split_into_words: bool = False,
        # 填充到的长度的倍数，默认为 None（不填充到倍数）
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，可选参数，默认为 None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回令牌类型 ID，默认为 None
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码，默认为 None
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的令牌，默认为 False
        return_overflowing_tokens: bool = False,
        # 是否返回特殊令牌掩码，默认为 False
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射，默认为 False
        return_offsets_mapping: bool = False,
        # 是否返回长度，默认为 False
        return_length: bool = False,
        # 是否显示详细信息，默认为 True
        verbose: bool = True,
        # 其他关键字参数
        **kwargs,
    ) -> BatchEncoding:
        # 抛出未实现错误，由子类实现具体方法
        raise NotImplementedError

    # 对 batch_encode_plus 方法进行装饰，添加文档字符串
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 批量编码文本或文本对，并生成批量编码结果
    def batch_encode_plus(
        self,
        # 批量的文本或文本对，支持多种输入形式
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        # 是否添加特殊令牌，默认为 True
        add_special_tokens: bool = True,
        # 填充策略，默认为 False
        padding: Union[bool, str, PaddingStrategy] = False,
        # 截断策略，默认为 None
        truncation: Union[bool, str, TruncationStrategy] = None,
        # 最大长度限制，默认为无限制
        max_length: Optional[int] = None,
        # 滑动窗口步长，默认为 0
        stride: int = 0,
        # 是否已分词，默认为 False
        is_split_into_words: bool = False,
        # 填充到的长度的倍数，默认为 None（不填充到倍数）
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，可选参数，默认为 None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回令牌类型 ID，默认为 None
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码，默认为 None
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的令牌，默认为 False
        return_overflowing_tokens: bool = False,
        # 是否返回特殊令牌掩码，默认为 False
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射，默认为 False
        return_offsets_mapping: bool = False,
        # 是否返回长度，默认为 False
        return_length: bool = False,
        # 是否显示详细信息，默认为 True
        verbose: bool = True,
        # 其他关键字参数
        **kwargs,
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.

        <Tip warning={true}>

        This method is deprecated, `__call__` should be used instead.

        </Tip>

        Args:
            batch_text_or_text_pairs (`List[str]`, `List[Tuple[str, str]]`, `List[List[str]]`, `List[Tuple[List[str], List[str]]`, and for not-fast tokenizers, also `List[List[int]]`, `List[Tuple[List[int], List[int]]]`):
                Batch of sequences or pair of sequences to be encoded. This can be a list of
                string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
                details in `encode_plus`).
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

        # 调用_batch_encode_plus方法进行批量编码
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
    # 批量编码输入文本或文本对，并返回编码结果
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
        # 是否添加特殊标记，默认为True
        add_special_tokens: bool = True,
        # 填充策略，默认为不填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 截断策略，默认为不截断
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        # 最大长度，默认为None
        max_length: Optional[int] = None,
        # 步长，默认为0
        stride: int = 0,
        # 输入是否已分词，默认为False
        is_split_into_words: bool = False,
        # 填充到倍数，默认为None
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回token类型ID，默认为None
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码，默认为None
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出token，默认为False
        return_overflowing_tokens: bool = False,
        # 是否返回特殊token掩码，默认为False
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射，默认为False
        return_offsets_mapping: bool = False,
        # 是否返回长度，默认为False
        return_length: bool = False,
        # 是否详细显示，默认为True
        verbose: bool = True,
        # 其他参数
        **kwargs,
    ) -> BatchEncoding:
        # 抛出未实现错误
        raise NotImplementedError

    # 对编码后的输入进行填充
    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        # 填充标志，默认为True
        padding: Union[bool, str, PaddingStrategy] = True,
        # 最大长度，默认为None
        max_length: Optional[int] = None,
        # 填充到倍数，默认为None
        pad_to_multiple_of: Optional[int] = None,
        # 是否返回注意力掩码，默认为None
        return_attention_mask: Optional[bool] = None,
        # 返回的张量类型，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否详细显示，默认为True
        verbose: bool = True,
    ):
        
    # 从序列中创建token类型ID
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
        # 如果只有一个序列，则返回全为0的token类型ID
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        # 如果有两个序列，则返回第一个序列全为0，第二个序列全为1的token类型ID
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    # 构建带有特殊token的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 定义一个方法，用于构建模型输入，将两个序列连接起来并添加特殊标记
    # 这个实现不会添加特殊标记，应该在子类中重写这个方法
    def __call__(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    # 准备模型输入
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
    
    # 截断序列
    def truncate_sequences(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    
    # 填充
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    
    # 将 tokens 转换为字符串
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
        often want to remove sub-word tokenization artifacts at the same time.

        Args:
            tokens (`List[str]`): The token to join in a string.

        Returns:
            `str`: The joined tokens.
        """
        raise NotImplementedError

    # 批量解码
    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
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
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `List[str]`: The list of decoded sentences.
        """
        # 返回由调用 decode 方法得到的字符串列表，将一组标记的列表转换为字符串
        return [
            self.decode(
                seq,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs,
            )
            for seq in sequences
        ]

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
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
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        # 转换输入为 Python 列表
        token_ids = to_py_obj(token_ids)
        # 调用内部的 _decode 方法完成解码
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
        # 抛出未实现错误，因为这是一个抽象方法，子类应该实现它
        raise NotImplementedError

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列 id。当使用 tokenizer 的 `prepare_for_model` 或 `encode_plus` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                第一个序列的 id 列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个序列的 id 列表。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                标记列表是否已经使用模型的特殊标记进行格式化。

        Returns:
            包含整数 0 或 1 的列表：1 表示特殊标记，0 表示序列标记。
        """
        assert already_has_special_tokens and token_ids_1 is None, (
            "You cannot use ``already_has_special_tokens=False`` with this tokenizer. "
            "Please use a slow (full python) tokenizer to activate this argument. "
            "Or set `return_special_tokens_mask=True` when calling the encoding method "
            "to get the special tokens mask in any tokenizer. "
        )

        # 缓存 all_special_ids 属性
        all_special_ids = self.all_special_ids  

        # 生成特殊标记的掩码列表，如果标记在 all_special_ids 中，则标记为 1，否则为 0
        special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]

        return special_tokens_mask

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """
        清理简单的英文标记化残留，如标点前的空格和缩写形式。

        Args:
            out_string (`str`): 要清理的文本。

        Returns:
            `str`: 清理后的字符串。
        """
        # 清理标点符号前的空格和缩写形式
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
```  
    def _eventual_warn_about_too_long_sequence(self, ids: List[int], max_length: Optional[int], verbose: bool):
        """
        根据输入和内部状态，可能会触发关于序列过长的警告，超过模型所支持的最大长度

        Args:
            ids (`List[str]`): Tokenization 生成的 ids
            max_length (`int`, *optional*): 所需的最大长度（如果设置了则不会触发警告）
            verbose (`bool`): 是否打印更多信息和警告
        """
        如果 max_length 为 None 并且 ids 的长度大于 self.model_max_length 并且 verbose 为 True
        if max_length is None and len(ids) > self.model_max_length and verbose:
            如果尚未发出关于序列长度超过指定最大长度的警告
            if not self.deprecation_warnings.get("sequence-length-is-longer-than-the-specified-maximum", False):
                打印警告信息
                logger.warning(
                    "Token indices sequence length is longer than the specified maximum sequence length "
                    f"for this model ({len(ids)} > {self.model_max_length}). Running this sequence through the model "
                    "will result in indexing errors"
                )
            将关于序列长度超过指定最大长度的警告标记为已发出
            self.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True

    def _switch_to_input_mode(self):
        """
        将分词器切换到输入模式（当分词器具有不同的输入/输出模式时）
        """
        pass

    def _switch_to_target_mode(self):
        """
        将分词器切换到目标模式（当分词器具有不同的输入/输出模式时）
        """
        pass

    @contextmanager
    def as_target_tokenizer(self):
        """
        临时设置分词器用于编码目标。适用于与需要对标签进行稍微不同处理的序列到序列模型相关的分词器。
        """
        发出警告信息
        warnings.warn(
            "`as_target_tokenizer` is deprecated and will be removed in v5 of Transformers. You can tokenize your "
            "labels by using the argument `text_target` of the regular `__call__` method (either in the same call as "
            "your input texts if you use the same keyword arguments, or in a separate call."
        )
        将分词器切换到目标模式
        self._switch_to_target_mode()
        设置标志以指示当前处于目标上下文管理器中
        self._in_target_context_manager = True
        返回
        yield
        设置标志以指示当前不再处于目标上下文管理器中
        self._in_target_context_manager = False
        将分词器切换回输入模式
        self._switch_to_input_mode()

    @classmethod
    # 注册自定义的类与指定的自动类。仅应用于自定义分词器，因为库中的分词器已经与 `AutoTokenizer` 进行了映射。
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
        # 如果 auto_class 不是字符串，将其转换为 auto_class 的名称
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        # 导入 transformers.models.auto 模块
        import transformers.models.auto as auto_module

        # 检查 auto_class 是否存在于 auto_module 中，如果不存在则引发 ValueError
        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        # 将 auto_class 分配给类的 _auto_class 属性
        cls._auto_class = auto_class

    # 准备用于序列到序列模型的批次数据
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
# 函数已弃用，将在 HuggingFace Transformers 的第 5 版中移除。使用常规的 __call__ 方法来准备输入和目标。
# 下面是一个简短的示例：
# model_inputs = tokenizer(src_texts, text_target=tgt_texts, ...)
# 如果需要为源文本和目标文本使用不同的关键字参数，应该像这样进行两次调用：
# model_inputs = tokenizer(src_texts, ...)
# labels = tokenizer(text_target=tgt_texts, ...)
# model_inputs["labels"] = labels["input_ids"]
# 查看您选择的分词器的特定参数的文档以获取更多详细信息。
# 有关更完整的示例，请查看 `prepare_seq2seq_batch` 的实现。

warnings.warn(formatted_warning, FutureWarning)
# mBART 特定的 kwargs，其他模型应该忽略。
kwargs.pop("src_lang", None)
kwargs.pop("tgt_lang", None)
if max_length is None:
    max_length = self.model_max_length
model_inputs = self(
    src_texts,
    add_special_tokens=True,
    return_tensors=return_tensors,
    max_length=max_length,
    padding=padding,
    truncation=truncation,
    **kwargs,
)
if tgt_texts is None:
    return model_inputs
# 处理 tgt_texts
if max_target_length is None:
    max_target_length = max_length
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
model_inputs["labels"] = labels["input_ids"]
return model_inputs


def get_fast_tokenizer_file(tokenization_files: List[str]) -> str:
    """
    获取用于此版本 transformers 的分词文件。

    Args:
        tokenization_files (`List[str]`): 可用配置文件的列表。

    Returns:
        `str`: 要使用的分词文件。
    """
    tokenizer_files_map = {}
    for file_name in tokenization_files:
        search = _re_tokenizer_file.search(file_name)
        if search is not None:
            v = search.groups()[0]
            tokenizer_files_map[v] = file_name
    available_versions = sorted(tokenizer_files_map.keys())

    # 默认使用 FULL_TOKENIZER_FILE，然后尝试查看一些更新版本。
    tokenizer_file = FULL_TOKENIZER_FILE
    transformers_version = version.parse(__version__)
    for v in available_versions:
        if version.parse(v) <= transformers_version:
            tokenizer_file = tokenizer_files_map[v]
        else:
            # 由于版本已排序，所以没有继续查找的必要。
            break

    return tokenizer_file
# 复制 PreTrainedTokenizerBase 类的 push_to_hub 方法，以便更新文档字符串，避免更改原始文档字符串。
PreTrainedTokenizerBase.push_to_hub = copy_func(PreTrainedTokenizerBase.push_to_hub)
# 如果 PreTrainedTokenizerBase.push_to_hub 方法存在文档字符串，则对其进行格式化，用于对象、对象类和对象文件的替换。
if PreTrainedTokenizerBase.push_to_hub.__doc__ is not None:
    PreTrainedTokenizerBase.push_to_hub.__doc__ = PreTrainedTokenizerBase.push_to_hub.__doc__.format(
        object="tokenizer", object_class="AutoTokenizer", object_files="tokenizer files"
    )
```