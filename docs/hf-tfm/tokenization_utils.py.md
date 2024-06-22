# `.\transformers\tokenization_utils.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

"""
用于 Python 分词器的标记化类。对于快速分词器（由 HuggingFace 的 tokenizers 库提供），请参见 tokenization_utils_fast.py
"""
import bisect
import itertools
import re
import unicodedata
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union, overload

from .tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    INIT_TOKENIZER_DOCSTRING,
    AddedToken,
    BatchEncoding,
    EncodedInput,
    EncodedInputPair,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from .utils import PaddingStrategy, TensorType, add_end_docstrings, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 慢速分词器保存在一个词汇表加上三个单独的文件中
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

class Trie:
    """
    Python 中的 Trie。根据单词列表创建 Trie。Trie 用于在一次遍历中拆分`added_tokens`
    参考链接 https://en.wikipedia.org/wiki/Trie
    """

    def __init__(self):
        self.data = {}  # Trie 数据结构
        self._tokens = set()  # 存储单词集合

    def add(self, word: str):
        """
        遍历单词中的每个字符（utf-8 字符），递归将其添加到内部`data` Trie 表示中。
        使用特殊键`""`表示终止。

        此函数是幂等的，两次添加相同的单词将不会改变 Trie

        示例：

        ```python
        >>> trie = Trie()
        >>> trie.add("Hello 友達")
        >>> trie.data
        {"H": {"e": {"l": {"l": {"o": {" ": {"友": {"達": {"": 1}}}}}}}}}

        >>> trie.add("Hello")
        >>> trie.data
        {"H": {"e": {"l": {"l": {"o": {"": 1, " ": {"友": {"達": {"": 1}}}}}}}}}
        ```py
        """
        if not word:
            # 防止空字符串
            return

        self._tokens.add(word)  # 将单词添加到集合中
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}  # 递归添加字符
            ref = ref[char]
        ref[""] = 1  # 终止标志
    # 定义一个方法用于根据给定的偏移量对文本进行切割
    def cut_text(self, text, offsets):
        # 现在我们已经有了所有的偏移量，只需要进行实际的分割
        # 我们最终需要添加字符串的第一部分和最后一部分
        offsets.append(len(text))  # 将文本的长度作为最后一个偏移量添加到列表中
        tokens = []  # 初始化一个空列表用于存储切割后的文本片段
        start = 0  # 初始化起始位置为0
        for end in offsets:  # 遍历所有的偏移量
            if start > end:  # 如果起始位置大于结束位置，说明存在错误
                logger.error(
                    "There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it"
                    " anyway."
                )
                continue  # 继续下一次循环
            elif start == end:  # 如果起始位置等于结束位置，可能是在索引0处有匹配
                # 我们也防止在连续匹配时出现零宽切割
                continue  # 继续下一次循环
            tokens.append(text[start:end])  # 将切割后的文本片段添加到列表中
            start = end  # 更新起始位置为当前结束位置

        return tokens  # 返回切割后的文本片段列表
def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # 检查字符是否为空格字符
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    # \t、\n 和 \r 本质上是控制字符，但我们将它们视为空格，因为它们通常被认为是空格。
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    # 获取字符的 Unicode 分类
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `char` is a control character."""
    # 检查字符是否为控制字符
    # These are technically control characters but we count them as whitespace
    # characters.
    # 这些本质上是控制字符，但我们将它们视为空格字符。
    if char == "\t" or char == "\n" or char == "\r":
        return False
    # 获取字符的 Unicode 分类
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    # 检查字符是否为标点字符
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    # 我们将所有非字母/数字的 ASCII 字符视为标点符号。
    # "^", "$", 和 "`" 等字符不属于 Unicode 标点符号类，但出于一致性的考虑，我们仍将它们视为标点符号。
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    # 获取字符的 Unicode 分类
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_end_of_word(text):
    """Checks whether the last character in text is one of a punctuation, control or whitespace character."""
    # 检查文本中最后一个字符是否是标点符号、控制字符或空格字符
    last_char = text[-1]
    return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


def _is_start_of_word(text):
    """Checks whether the first character in text is one of a punctuation, control or whitespace character."""
    # 检查文本中第一个字符是否是标点符号、控制字符或空格字符
    first_char = text[0]
    return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))


def _insert_one_token_to_ordered_list(token_list: List[str], new_token: str):
    """
    Inserts one token to an ordered list if it does not already exist. Note: token_list must be sorted.
    """
    # 如果一个令牌不存在于有序列表中，则将一个令牌插入有序列表中。注意：token_list 必须是已排序的。
    insertion_idx = bisect.bisect_left(token_list, new_token)
    # 检查 new_token 是否已经存在于有序的 token_list 中
    if insertion_idx < len(token_list) and token_list[insertion_idx] == new_token:
        # 如果 new_token 已经在 token_list 中，则不添加
        return
    else:
        # 否则，插入 new_token 到指定位置
        token_list.insert(insertion_idx, new_token)


@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class PreTrainedTokenizer(PreTrainedTokenizerBase):
    """
    Base class for all slow tokenizers.

    Inherits from [`~tokenization_utils_base.PreTrainedTokenizerBase`].

    Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading
    pretrained tokenizers as well as adding tokens to the vocabulary.

    This class also contain the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    """
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """

    # 初始化 Tokenizer 类
    def __init__(self, **kwargs):
        # 1. 初始化父类

        # 初始化 tokens_trie 属性为 Trie 对象
        self.tokens_trie = Trie()

        # 2. 如果子类没有初始化 `_added_tokens_decoder`，则初始化为字典
        if not hasattr(self, "_added_tokens_decoder"):
            self._added_tokens_decoder: Dict[int, AddedToken] = {}

        # 3. 如果传入了 `added_tokens_decoder`，表示从已保存的 tokenizer 中加载，覆盖原有的值
        self._added_tokens_decoder.update(kwargs.pop("added_tokens_decoder", {}))
        self._added_tokens_encoder: Dict[str, int] = {k.content: v for v, k in self._added_tokens_decoder.items()}

        # 4. 初始化父类
        super().__init__(**kwargs)

        # 4. 如果一些特殊 token 不在词汇表中，添加到词汇表末尾
        # 添加顺序与 self.SPECIAL_TOKENS_ATTRIBUTES 中的顺序相同
        self._add_tokens(
            [token for token in self.all_special_tokens_extended if token not in self._added_tokens_encoder],
            special_tokens=True,
        )

        # 设置 _decode_use_source_tokenizer 为 False
        self._decode_use_source_tokenizer = False

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        raise NotImplementedError

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        """
        Returns the sorted mapping from string to index. The added tokens encoder is cached for performance
        optimisation in `self._added_tokens_encoder` for the slow tokenizers.
        """
        return {k.content: v for v, k in sorted(self._added_tokens_decoder.items(), key=lambda item: item[0])}

    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        """
        Returns the added tokens in the vocabulary as a dictionary of index to AddedToken.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        return dict(sorted(self._added_tokens_decoder.items(), key=lambda item: item[0]))

    @added_tokens_decoder.setter
    def added_tokens_decoder(self, value: Dict[int, Union[AddedToken, str]]) -> Dict[int, AddedToken]:
        # 如果值为字符串，抛出错误，用户应该定义行为
        for index, token in value.items():
            if not isinstance(token, (str, AddedToken)) or not isinstance(index, int):
                raise ValueError(
                    f"The provided `added_tokens_decoder` has an element of type {index.__class__, token.__class__}, should be a dict of {int, Union[AddedToken, str]}"
                )

            # 将值添加到 _added_tokens_decoder 和 _added_tokens_encoder 中
            self._added_tokens_decoder[index] = AddedToken(token) if isinstance(token, str) else token
            self._added_tokens_encoder[str(token)] = index
    def get_added_vocab(self) -> Dict[str, int]:
        """
        返回一个字典，其中包含词汇表中添加的标记，键为标记，值为索引。结果可能与快速调用的结果不同，因为我们总是添加标记，即使它们已经在词汇表中。这是我们应该改变的一点。

        Returns:
            `Dict[str, int]`: 添加的标记。
        """
        return self._added_tokens_encoder

    def __len__(self):
        """
        返回包含添加标记的完整词汇表的大小。计数 `keys` 而不是 `values`，因为如果词汇表中有空隙，我们将在错误的索引处添加标记。
        """
        return len(set(self.get_vocab().keys()))

    def _update_trie(self, unique_no_split_tokens: Optional[str] = []):
        """
        更新 Trie 数据结构，用于存储标记的前缀。
        """
        for token in self._added_tokens_decoder.values():
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token.content)
        for token in unique_no_split_tokens:
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        返回在使用特殊标记编码序列时添加的标记数量。

        <Tip>

        这会对虚拟输入进行编码并检查添加的标记数量，因此效率不高。不要将其放在训练循环中。

        </Tip>

        Args:
            pair (`bool`, *optional*, 默认为 `False`):
                是否在序列对或单个序列的情况下计算添加的标记数量。

        Returns:
            `int`: 添加到序列的特殊标记数量。
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def _tokenize(self, text, **kwargs):
        """
        使用标记器将字符串转换为标记序列（字符串）。对于基于词的词汇表，按单词拆分；对于基于子词的词汇表（BPE/SentencePieces/WordPieces），按子词拆分。

        不要考虑添加的标记。
        """
        raise NotImplementedError
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        # 如果 tokens 为 None，则直接返回 None
        if tokens is None:
            return None

        # 如果 tokens 是字符串，则调用 _convert_token_to_id_with_added_voc 函数转换为对应的 token id
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        # 初始化一个空列表 ids，用于存储转换后的 token id
        ids = []
        # 遍历 tokens 中的每个 token
        for token in tokens:
            # 将每个 token 转换为对应的 token id，并添加到 ids 列表中
            ids.append(self._convert_token_to_id_with_added_voc(token))
        # 返回转换后的 token id 列表
        return ids

    # 将 token 转换为对应的 token id，并添加到词汇表中
    def _convert_token_to_id_with_added_voc(self, token):
        # 如果 token 为 None，则直接返回 None
        if token is None:
            return None

        # 如果 token 在已添加 token 编码器中，则返回其对应的 token id
        if token in self._added_tokens_encoder:
            return self._added_tokens_encoder[token]
        # 否则调用 _convert_token_to_id 函数将 token 转换为对应的 token id
        return self._convert_token_to_id(token)

    # 将 token 转换为对应的 token id
    def _convert_token_to_id(self, token):
        # 抛出 NotImplementedError 异常，该函数应由子类实现
        raise NotImplementedError

    # 编码文本（及可能的文本对），生成模型输入所需的张量表示
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
    # 定义函数 prepare_for_model，接受输入文本和可选的文本对，返回编码后的批处理输入
    ) -> BatchEncoding:
        # 定义内部函数 get_input_ids 用于将输入文本转换为对应的 token IDs
        def get_input_ids(text):
            # 如果输入是字符串，将其标记化并转换为 token IDs
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            # 如果输入是字符串列表或元组且至少包含一个字符串
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                # 如果指定了 is_split_into_words 参数，对每个字符串进行分词并转换为 token IDs
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                # 否则直接将每个字符串转换为 token IDs
                else:
                    return self.convert_tokens_to_ids(text)
            # 如果输入是整数列表或元组且至少包含一个整数，直接返回该列表
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            # 其他情况，根据 is_split_into_words 参数抛出异常
            else:
                if is_split_into_words:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string or a list/tuple of strings when"
                        " `is_split_into_words=True`."
                    )
                else:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of"
                        " integers."
                    )

        # 如果设置了 return_offsets_mapping 参数，抛出 NotImplementedError
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # 获取第一个文本的 token IDs
        first_ids = get_input_ids(text)
        # 如果提供了第二个文本，获取第二个文本的 token IDs
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        # 调用 prepare_for_model 方法，将 token IDs 编码为模型输入
        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )
    # 定义一个私有方法，用于批量编码文本或文本对
    def _batch_encode_plus(
        # 输入参数为包含不同类型输入的列表
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
        # 最大长度限制，默认为None
        max_length: Optional[int] = None,
        # 步长，默认为0
        stride: int = 0,
        # 是否已分词，默认为False
        is_split_into_words: bool = False,
        # 填充到指定长度的倍数，默认为None
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回token类型ID，默认为None
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码，默认为None
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的token，默认为False
        return_overflowing_tokens: bool = False,
        # 是否返回特殊标记掩码，默认为False
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射，默认为False
        return_offsets_mapping: bool = False,
        # 是否返回长度，默认为False
        return_length: bool = False,
        # 是否显示详细信息，默认为True
        verbose: bool = True,
        # 其他关键字参数
        **kwargs,
    ) -> BatchEncoding:
        # 定义内部函数，用于将文本转换为输入 IDs
        def get_input_ids(text):
            # 如果输入是字符串，则进行分词并将其转换为 IDs
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            # 如果输入是字符串列表或元组，则根据情况进行处理
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                # 如果 is_split_into_words 为 True，则分别处理每个文本并合并结果
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                # 否则直接将所有文本转换为 IDs
                else:
                    return self.convert_tokens_to_ids(text)
            # 如果输入是整数列表或元组，则直接返回
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            # 如果输入不符合预期类型，则引发 ValueError
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        # 如果要求返回偏移映射，则引发 NotImplementedError
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        # 初始化输入 ID 列表
        input_ids = []
        # 遍历批量文本或文本对
        for ids_or_pair_ids in batch_text_or_text_pairs:
            # 如果输入不是列表或元组，则假定只有一个文本
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            # 如果 is_split_into_words 为 True，并且输入的第一个元素不是列表或元组，则将其视为单个文本
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            # 否则，假定输入是文本对
            else:
                ids, pair_ids = ids_or_pair_ids

            # 获取第一个文本的输入 IDs
            first_ids = get_input_ids(ids)
            # 如果存在第二个文本，则获取第二个文本的输入 IDs
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            # 将第一个文本和第二个文本的输入 IDs 组成一个元组，并添加到输入 ID 列表中
            input_ids.append((first_ids, second_ids))

        # 使用 _batch_prepare_for_model 方法对输入进行预处理，返回处理后的结果
        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        # 将处理后的结果封装成 BatchEncoding 对象并返回
        return BatchEncoding(batch_outputs)

    # 添加 ENCODE_KWARGS_DOCSTRING 和 ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING 的文档字符串
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],  # 接收一个列表，列表中元素可以是预分词输入对或者包含整数列表和空值的元组
        add_special_tokens: bool = True,  # 是否添加特殊标记，默认为 True
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 最大长度，默认为 None
        stride: int = 0,  # 步长，默认为 0
        pad_to_multiple_of: Optional[int] = None,  # 填充到的长度，默认为 None
        return_tensors: Optional[str] = None,  # 返回的张量类型，默认为 None
        return_token_type_ids: Optional[bool] = None,  # 是否返回 token 类型 IDs，默认为 None
        return_attention_mask: Optional[bool] = None,  # 是否返回 attention mask，默认为 None
        return_overflowing_tokens: bool = False,  # 是否返回溢出的 token，默认为 False
        return_special_tokens_mask: bool = False,  # 是否返回特殊 token 的掩码，默认为 False
        return_length: bool = False,  # 是否返回长度，默认为 False
        verbose: bool = True,  # 是否详细输出信息，默认为 True
    ) -> BatchEncoding:  # 返回 BatchEncoding 对象
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs  # tokenized 输入 ID 或输入 ID 对的列表
        """

        batch_outputs = {}  # 初始化批处理输出字典
        for first_ids, second_ids in batch_ids_pairs:  # 对于每个输入 ID 对
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # 在此之后批处理填充
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # 在此之后批处理填充
                return_attention_mask=False,  # 在此之后批处理填充
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # 最终将整个批次转换为张量
                prepend_batch_axis=False,
                verbose=verbose,
            )

            for key, value in outputs.items():  # 对于输出中的每个键值对
                if key not in batch_outputs:  # 如果键不在批处理输出中
                    batch_outputs[key] = []  # 初始化键的值为空列表
                batch_outputs[key].append(value)  # 将值添加到对应键的列表中

        batch_outputs = self.pad(  # 进行填充
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)  # 将批处理输出转换为 BatchEncoding 对象

        return batch_outputs  # 返回批处理输出对象

    def prepare_for_tokenization(
        self, text: str, is_split_into_words: bool = False, **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        执行任何必要的转换，以便在标记化之前使用。

        这个方法应该从kwargs中弹出参数，并将剩余的`kwargs`作为返回。我们在编码过程结束时测试`kwargs`，以确保所有参数都已被使用。

        Args:
            text (`str`):
                要准备的文本。
            is_split_into_words (`bool`, *可选*, 默认为 `False`):
                输入是否已经预标记化（例如，已经拆分成单词）。如果设置为`True`，则标记器假设输入已经被拆分成单词（例如，通过在空格上拆分），它将对其进行标记化。这对于NER或标记分类很有用。
            kwargs (`Dict[str, Any]`, *可选*):
                用于标记化的关键字参数。

        Returns:
            `Tuple[str, Dict[str, Any]]`: 准备好的文本和未使用的kwargs。
        """
        return (text, kwargs)

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列ID。当使用标记器的`prepare_for_model`或`encode_plus`方法添加特殊标记时，将调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                第一个序列的ID列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的ID列表。
            already_has_special_tokens (`bool`, *可选*, 默认为 `False`):
                标记列表是否已经使用特殊标记格式化为模型。

        Returns:
            一个整数列表，范围为[0, 1]：特殊标记为1，序列标记为0。
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

    @overload
    def convert_ids_to_tokens(self, ids: int, skip_special_tokens: bool = False) -> str:
        ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens: bool = False) -> List[str]:
        ...

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
        """
        将ID转换为标记。

        Args:
            ids (Union[int, List[int]]):
                要转换的ID或ID列表。
            skip_special_tokens (`bool`, *可选*, 默认为 `False`):
                是否跳过特殊标记。

        Returns:
            `str` 或 `List[str]`: 标记化的结果。
        """
    def convert_ids_to_tokens(self, ids) -> Union[str, List[str]]:
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
        # Check if the input is a single integer
        if isinstance(ids, int):
            # Check if the id is in the added tokens decoder
            if ids in self._added_tokens_decoder:
                return self._added_tokens_decoder[ids].content
            else:
                return self._convert_id_to_token(ids)
        
        # If input is a list of integers
        tokens = []
        for index in ids:
            index = int(index)
            # Skip special tokens if specified
            if skip_special_tokens and index in self.all_special_ids:
                continue
            # Check if the index is in the added tokens decoder
            if index in self._added_tokens_decoder:
                tokens.append(self._added_tokens_decoder[index].content)
            else:
                tokens.append(self._convert_id_to_token(index))
        
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        # Placeholder method to convert an index to a token, to be implemented in subclasses
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # Convert a list of tokens to a string by joining them with spaces
        return " ".join(tokens)

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        # 从关键字参数中弹出'use_source_tokenizer'，默认为False
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        # 将token_ids转换为tokens
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        # 计算需要添加的特殊token
        legacy_added_tokens = set(self._added_tokens_encoder.keys()) - set(self.all_special_tokens) | {
            token for token in self.additional_special_tokens if self.convert_tokens_to_ids(token) >= self.vocab_size
        }
        # 避免混合字节级和Unicode级别的字节级BPT
        # 我们需要单独构建特殊token和字节级token的字符串
        # 参考 https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        # 在版本5中，特殊token应在convert_tokens_to_string中处理，而_convert_tokens_to_string中处理
        for token in filtered_tokens:
            # 跳过特殊token
            if skip_special_tokens and token in self.all_special_ids:
                continue
            # 处理旧版本的特殊token
            if token in legacy_added_tokens:
                if current_sub_text:
                    string = self.convert_tokens_to_string(current_sub_text)
                    if len(string) > 0:
                        sub_texts.append(string)
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        # 如果spaces_between_special_tokens为True，则在特殊token之间添加空格
        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        # 清理tokenization的空格
        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        # 如果需要清理tokenization的空格，则进行清理
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text
```