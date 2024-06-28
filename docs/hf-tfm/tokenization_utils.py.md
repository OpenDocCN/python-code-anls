# `.\tokenization_utils.py`

```py
# coding=utf-8
# 版权所有 2020 年 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件基于“原样”提供，不提供任何形式的明示或暗示担保或条件。
# 有关详细信息，请参阅许可证。
"""
用于 Python Tokenizers 的标记化类。对于快速标记化器（由 HuggingFace 的 tokenizers 库提供），请参见 tokenization_utils_fast.py
"""
import bisect  # 导入 bisect 模块，用于高效地插入和搜索元素
import itertools  # 导入 itertools 模块，用于创建迭代器的函数
import re  # 导入 re 模块，用于支持正则表达式操作
import unicodedata  # 导入 unicodedata 模块，提供对 Unicode 字符数据库的访问功能
from collections import OrderedDict  # 导入 OrderedDict 类，实现有序字典
from typing import Any, Dict, List, Optional, Tuple, Union, overload  # 导入类型提示

from .tokenization_utils_base import (  # 从 tokenization_utils_base 模块导入以下符号
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
from .utils import PaddingStrategy, TensorType, add_end_docstrings, logging  # 从 utils 模块导入符号

logger = logging.get_logger(__name__)  # 获取当前模块的 logger 对象

# Slow tokenizers are saved in a vocabulary plus three separated files
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"  # 定义特殊标记映射文件名
ADDED_TOKENS_FILE = "added_tokens.json"  # 定义添加的标记文件名
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"  # 定义标记器配置文件名


class Trie:
    """
    Trie（字典树）的实现。基于给定的单词列表创建 Trie 结构，用于在一个步骤中分割 `added_tokens`。
    参考资料 https://en.wikipedia.org/wiki/Trie
    """

    def __init__(self):
        self.data = {}  # 初始化 Trie 数据结构
        self._tokens = set()  # 初始化存储添加的标记的集合

    def add(self, word: str):
        """
        将给定单词添加到 Trie 中。
        通过每个字符（UTF-8 字符）递归地添加到内部 `data` Trie 表示中。
        使用特殊键 `""` 表示终止状态。
        
        此函数是幂等的，添加两次相同的单词不会改变 Trie 结构。

        示例:

        ```
        >>> trie = Trie()
        >>> trie.add("Hello 友達")
        >>> trie.data
        {"H": {"e": {"l": {"l": {"o": {" ": {"友": {"達": {"": 1}}}}}}}}}

        >>> trie.add("Hello")
        >>> trie.data
        {"H": {"e": {"l": {"l": {"o": {"": 1, " ": {"友": {"達": {"": 1}}}}}}}}}
        ```
        """
        if not word:
            # 避免空字符串
            return
        
        self._tokens.add(word)  # 将单词添加到集合中
        ref = self.data  # 设置初始引用为 Trie 的根节点
        for char in word:
            ref[char] = ref.get(char, {})  # 如果字符不存在于 Trie 中，则创建一个新的空字典
            ref = ref[char]  # 移动到下一个字符的字典
        ref[""] = 1  # 在最后字符处标记为结束状态
    # 定义一个方法 cut_text，用于根据给定的偏移量列表将文本 text 切分成多个部分并返回
    def cut_text(self, text, offsets):
        # 将文本的总长度作为最后一个偏移量，确保所有部分都被切分
        offsets.append(len(text))
        # 初始化一个空列表，用于存储切分后的文本部分（即 tokens）
        tokens = []
        # 初始化起始位置 start 为 0
        start = 0
        # 遍历偏移量列表
        for end in offsets:
            # 如果起始位置大于结束位置，表示有错误，记录错误信息到日志
            if start > end:
                logger.error(
                    "There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it"
                    " anyway."
                )
                # 继续处理下一个偏移量
                continue
            # 如果起始位置等于结束位置，可能是在索引 0 处匹配到了，或者是连续匹配导致的零宽度切分，跳过处理
            elif start == end:
                continue
            # 将从 start 到 end 的文本部分加入到 tokens 列表中
            tokens.append(text[start:end])
            # 更新起始位置为当前结束位置，为下一部分切分做准备
            start = end

        # 返回切分后的文本部分列表 tokens
        return tokens
# 检查字符是否为空白字符
def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    # 获取字符的Unicode分类
    cat = unicodedata.category(char)
    # 如果Unicode分类为"Zs"（空格分隔符），则判断为True
    if cat == "Zs":
        return True
    return False


# 检查字符是否为控制字符
def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    # 获取字符的Unicode分类
    cat = unicodedata.category(char)
    # 如果Unicode分类以"C"开头（控制字符），则判断为True
    if cat.startswith("C"):
        return True
    return False


# 检查字符是否为标点符号
def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    # 判断字符是否为ASCII中的标点符号范围内的字符
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    # 获取字符的Unicode分类
    cat = unicodedata.category(char)
    # 如果Unicode分类以"P"开头（标点符号），则判断为True
    if cat.startswith("P"):
        return True
    return False


# 检查文本的最后一个字符是否是标点符号、控制字符或空白字符
def _is_end_of_word(text):
    """Checks whether the last character in text is one of a punctuation, control or whitespace character."""
    # 获取文本的最后一个字符
    last_char = text[-1]
    # 返回最后一个字符是否为标点符号、控制字符或空白字符的布尔值
    return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


# 检查文本的第一个字符是否是标点符号、控制字符或空白字符
def _is_start_of_word(text):
    """Checks whether the first character in text is one of a punctuation, control or whitespace character."""
    # 获取文本的第一个字符
    first_char = text[0]
    # 返回第一个字符是否为标点符号、控制字符或空白字符的布尔值
    return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))


# 将一个新的token插入到有序列表中，如果该token已经存在，则不插入
def _insert_one_token_to_ordered_list(token_list: List[str], new_token: str):
    """
    Inserts one token to an ordered list if it does not already exist. Note: token_list must be sorted.
    """
    # 使用二分查找确定插入位置
    insertion_idx = bisect.bisect_left(token_list, new_token)
    # 检查新的token是否已经存在于有序的token_list中
    if insertion_idx < len(token_list) and token_list[insertion_idx] == new_token:
        # 如果存在，则直接返回，不做插入操作
        return
    else:
        # 如果不存在，则插入新的token到token_list中的对应位置
        token_list.insert(insertion_idx, new_token)
    """
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).
    """

    # 1. 初始化类的构造函数
    def __init__(self, **kwargs):
        # 2. 初始化 `self.tokens_trie` 为一个 Trie 数据结构对象
        self.tokens_trie = Trie()

        # 3. 如果子类没有初始化 `_added_tokens_decoder`，则初始化一个空的字典
        if not hasattr(self, "_added_tokens_decoder"):
            self._added_tokens_decoder: Dict[int, AddedToken] = {}

        # 4. 如果传入了 `added_tokens_decoder` 参数，表示从保存的分词器中加载，将其更新到 `_added_tokens_decoder`
        self._added_tokens_decoder.update(kwargs.pop("added_tokens_decoder", {}))
        # 使用 `_added_tokens_decoder` 构建 `_added_tokens_encoder`，将内容从 `AddedToken` 转换为字符串到整数的映射
        self._added_tokens_encoder: Dict[str, int] = {k.content: v for v, k in self._added_tokens_decoder.items()}

        # 5. 调用父类的构造函数
        super().__init__(**kwargs)

        # 6. 如果某些特殊标记不在词汇表中，将它们添加到词汇表末尾
        #    添加顺序与 `self.SPECIAL_TOKENS_ATTRIBUTES` 相同，依赖于 `tokenizers` 对象
        self._add_tokens(
            [token for token in self.all_special_tokens_extended if token not in self._added_tokens_encoder],
            special_tokens=True,
        )

        # 7. 设定 `_decode_use_source_tokenizer` 标志为 False
        self._decode_use_source_tokenizer = False

    @property
    def is_fast(self) -> bool:
        # 返回 False，表明不是一个快速的分词器
        return False

    @property
    def vocab_size(self) -> int:
        """
        `int`: 基础词汇表的大小（不包括添加的特殊标记）。
        """
        # 抛出未实现错误，要求子类实现这个属性
        raise NotImplementedError

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        """
        返回从字符串到索引的排序映射。为了性能优化，缓存了慢速分词器中的 `_added_tokens_encoder`。
        """
        # 将 `_added_tokens_decoder` 按索引排序并转换为字符串到整数的映射返回
        return {k.content: v for v, k in sorted(self._added_tokens_decoder.items(), key=lambda item: item[0])}

    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        """
        返回词汇表中的添加标记，作为索引到 AddedToken 的字典。

        Returns:
            `Dict[str, int]`: 添加的标记。
        """
        # 按索引排序并返回 `_added_tokens_decoder` 的内容
        return dict(sorted(self._added_tokens_decoder.items(), key=lambda item: item[0]))

    @added_tokens_decoder.setter
    def added_tokens_decoder(self, value: Dict[int, Union[AddedToken, str]]) -> Dict[int, AddedToken]:
        # 如果值是字符串类型，抛出错误，用户应该定义正确的行为
        for index, token in value.items():
            if not isinstance(token, (str, AddedToken)) or not isinstance(index, int):
                raise ValueError(
                    f"The provided `added_tokens_decoder` has an element of type {index.__class__, token.__class__}, should be a dict of {int, Union[AddedToken, str]}"
                )

            # 如果 token 是字符串类型，将其转换为 AddedToken 对象后存储
            self._added_tokens_decoder[index] = AddedToken(token) if isinstance(token, str) else token
            # 更新 `_added_tokens_encoder`，将 token 转换为字符串后存储其索引
            self._added_tokens_encoder[str(token)] = index
    def get_added_vocab(self) -> Dict[str, int]:
        """
        返回已添加到词汇表中的词汇作为一个字典，键为词汇，值为索引。结果可能与快速调用不同，因为我们当前总是添加这些词汇，即使它们已经存在于词汇表中。这是我们应该改变的事情。

        Returns:
            `Dict[str, int]`: 已添加的词汇表。
        """
        return self._added_tokens_encoder

    def __len__(self):
        """
        返回包含已添加词汇的完整词汇表的大小。计数的是 `keys` 而不是 `values`，因为如果词汇表中有空洞，我们会在错误的索引处添加分词器。
        """
        return len(set(self.get_vocab().keys()))

    def _update_trie(self, unique_no_split_tokens: Optional[str] = []):
        """
        更新 Trie 树，将新增的无需分割的词汇加入到词汇表中。

        Args:
            unique_no_split_tokens (`Optional[str]`, *optional*, defaults to `[]`):
                需要添加到 Trie 树中的唯一词汇列表。
        """
        for token in self._added_tokens_decoder.values():
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token.content)
        for token in unique_no_split_tokens:
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        返回在编码序列时添加的特殊标记数量。

        <Tip>

        这会对一个虚拟输入进行编码，并检查添加的特殊标记数量，因此不是效率高的操作。不要将其放在训练循环内。

        </Tip>

        Args:
            pair (`bool`, *optional*, defaults to `False`):
                是否计算序列对或单序列中添加的特殊标记数量。

        Returns:
            `int`: 添加到序列中的特殊标记数量。
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def _tokenize(self, text, **kwargs):
        """
        使用分词器将字符串转换为一系列标记（字符串）。基于词汇表的单词分割或基于子词的分割（BPE/SentencePieces/WordPieces）。

        不处理已添加的标记。

        Args:
            text (str): 要分词的文本。
            **kwargs: 其他参数传递给分词器的选项。

        Raises:
            NotImplementedError: 如果子类没有实现这个方法。
        """
        raise NotImplementedError
    # 将 tokens 转换为其对应的 ID，根据 tokens 的类型返回单个 ID 或 ID 列表
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        # 如果 tokens 为 None，直接返回 None
        if tokens is None:
            return None
        
        # 如果 tokens 是一个字符串，调用 _convert_token_to_id_with_added_voc 方法进行转换
        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        # 如果 tokens 是一个列表，则遍历每个 token 并调用 _convert_token_to_id_with_added_voc 方法进行转换
        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    # 根据 token 查找其对应的 ID，首先在自定义的添加 tokens 编码器中查找，然后再调用 _convert_token_to_id 方法
    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None
        
        # 如果 token 在自定义的添加 tokens 编码器中，则返回对应的 ID
        if token in self._added_tokens_encoder:
            return self._added_tokens_encoder[token]
        
        # 否则调用 _convert_token_to_id 方法进行转换
        return self._convert_token_to_id(token)

    # 用于将 token 转换为 ID，需要在子类中实现具体的转换逻辑
    def _convert_token_to_id(self, token):
        raise NotImplementedError

    # 根据给定的文本输入进行编码处理，支持添加特殊 tokens，填充策略，截断策略等多种参数设置
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
    ):
        # 具体实现细节需要在该方法的实现中完成，根据传入的参数进行文本编码和处理
        pass
    ) -> BatchEncoding:
        # 定义内部函数，根据输入文本返回对应的输入 ID 列表
        def get_input_ids(text):
            # 如果输入是字符串，将其标记化为 tokens，并转换成对应的 ID 列表
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            # 如果输入是字符串列表或元组且首个元素是字符串，根据 is_split_into_words 参数处理
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    # 将列表中的每个字符串按单词切分后标记化，再转换成对应的 ID 列表
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                else:
                    # 直接将字符串列表或元组标记化为 tokens，并转换成对应的 ID 列表
                    return self.convert_tokens_to_ids(text)
            # 如果输入是整数列表或元组，直接返回该列表
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                # 若参数 is_split_into_words 为 True 时，要求输入必须是字符串或字符串列表/元组
                if is_split_into_words:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string or a list/tuple of strings when"
                        " `is_split_into_words=True`."
                    )
                else:
                    # 否则，要求输入必须是字符串、字符串列表/元组或整数列表/元组
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of"
                        " integers."
                    )

        # 若设置了 return_offsets_mapping 参数，则抛出未实现的错误
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # 获取第一个文本的输入 ID 列表
        first_ids = get_input_ids(text)
        # 如果有第二个文本，则获取其输入 ID 列表；否则为 None
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        # 调用 prepare_for_model 方法，准备输入模型所需的数据格式
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
    # 定义一个方法 _batch_encode_plus，用于批量编码文本或文本对
    def _batch_encode_plus(
        self,
        # 输入参数 batch_text_or_text_pairs 可以是多种类型的列表，包括单文本、文本对、预分词输入、编码输入等
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        # 是否添加特殊标记，默认为 True
        add_special_tokens: bool = True,
        # 填充策略，默认为不填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 截断策略，默认为不截断
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        # 最大长度限制，可选
        max_length: Optional[int] = None,
        # 步长，默认为 0
        stride: int = 0,
        # 是否已经分成单词，默认为 False
        is_split_into_words: bool = False,
        # 填充到指定的倍数，默认为 None
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，默认为 None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回 token 类型 ID，默认为 None
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力遮罩，默认为 None
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的 token，默认为 False
        return_overflowing_tokens: bool = False,
        # 是否返回特殊 token 掩码，默认为 False
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
        # 定义内部函数 get_input_ids，用于将文本或文本对转换为输入 IDs
        def get_input_ids(text):
            # 如果输入是字符串，则进行分词和转换为 IDs
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            # 如果输入是字符串列表或元组，并且第一个元素是字符串，则根据 is_split_into_words 参数处理
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    # 将每个字符串分词后合并成一个 tokens 列表，并转换为 IDs
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                else:
                    # 直接将字符串列表或元组转换为 IDs
                    return self.convert_tokens_to_ids(text)
            # 如果输入是整数列表或元组，则直接返回
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                # 如果输入不合法，则抛出 ValueError 异常
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        # 如果要求返回偏移映射，但使用 Python 分词器不支持该功能，则抛出 NotImplementedError 异常
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        # 初始化 input_ids 列表
        input_ids = []
        # 遍历批量文本或文本对
        for ids_or_pair_ids in batch_text_or_text_pairs:
            # 如果元素不是列表或元组，则假设为单文本，没有配对文本
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            # 如果开启了 is_split_into_words 并且第一个元素不是列表或元组，则也假设为单文本
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                # 否则，假设为文本对，分别赋值给 ids 和 pair_ids
                ids, pair_ids = ids_or_pair_ids

            # 获取第一个文本的输入 IDs
            first_ids = get_input_ids(ids)
            # 如果存在第二个文本，则获取其输入 IDs；否则 pair_ids 为 None
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            # 将文本对的输入 IDs 添加到 input_ids 列表中
            input_ids.append((first_ids, second_ids))

        # 调用内部方法 _batch_prepare_for_model 处理输入 IDs，返回 batch_outputs
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

        # 返回 BatchEncoding 对象，其中包含处理后的批量输出
        return BatchEncoding(batch_outputs)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        # Initialize an empty dictionary to store batch outputs
        batch_outputs = {}
        
        # Iterate through each pair of input IDs in batch_ids_pairs
        for first_ids, second_ids in batch_ids_pairs:
            # Call prepare_for_model to process the input ids pairs
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            # Aggregate outputs into batch_outputs dictionary
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # Pad batch outputs based on padding_strategy
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # Convert batch_outputs to BatchEncoding object with specified tensor_type
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        # Return the final batch outputs as a BatchEncoding object
        return batch_outputs
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
            kwargs (`Dict[str, Any]`, *optional*):
                Keyword arguments to use for the tokenization.

        Returns:
            `Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.
        """
        return (text, kwargs)

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
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
        if already_has_special_tokens:
            # 如果已经有特殊标记，而且提供了第二个序列的 IDs，则引发错误
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            # 调用超类方法，返回特殊标记掩码
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        
        # 如果没有特殊标记，返回全零列表，表示所有标记都是序列标记而非特殊标记
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

    @overload
    def convert_ids_to_tokens(self, ids: int, skip_special_tokens: bool = False) -> str:
        ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens: bool = False) -> List[str]:
        ...

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts token ids into strings.

        Args:
            ids (`int` or `List[int]`):
                Token ids to convert.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether to skip special tokens during conversion.

        Returns:
            `str` or `List[str]`: Converted token(s) into string(s).
        """
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
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
        # Check if the input ids is a single integer
        if isinstance(ids, int):
            # Check if the integer id corresponds to an added token
            if ids in self._added_tokens_decoder:
                # Return the content of the added token
                return self._added_tokens_decoder[ids].content
            else:
                # Otherwise, convert the integer id to a token using the vocabulary
                return self._convert_id_to_token(ids)
        
        # If ids is a list of integers, process each index
        tokens = []
        for index in ids:
            index = int(index)  # Ensure index is treated as an integer
            # Skip special tokens if specified and the index is in special token ids
            if skip_special_tokens and index in self.all_special_ids:
                continue
            # Check if the index corresponds to an added token
            if index in self._added_tokens_decoder:
                # Append the content of the added token to tokens
                tokens.append(self._added_tokens_decoder[index].content)
            else:
                # Otherwise, convert the index to a token using the vocabulary
                tokens.append(self._convert_id_to_token(index))
        
        # Return the list of tokens
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        # Placeholder method to convert an integer index to a token
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # Joins a list of tokens into a single string separated by spaces
        return " ".join(tokens)

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ):
        # Method for decoding a list of token ids into a string
        # The parameters provide options for handling special tokens and tokenization spaces
        pass
        ) -> str:
        # 从kwargs中弹出"use_source_tokenizer"参数并设置为self._decode_use_source_tokenizer属性
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        # 将token_ids转换为tokens，并根据skip_special_tokens过滤特殊token
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        
        # 计算不属于self.all_special_tokens的self._added_tokens_encoder.keys()集合与token中的特殊token
        legacy_added_tokens = set(self._added_tokens_encoder.keys()) - set(self.all_special_tokens) | {
            token for token in self.additional_special_tokens if self.convert_tokens_to_ids(token) >= self.vocab_size
        }
        
        # 为了避免在字节级别BPT中混合字节级和unicode，需要分别构建添加的token和字节级token的字符串
        # 参考：https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        
        # 在版本5中，特殊token应该在convert_tokens_to_string和_convert_tokens_to_string中处理
        # TODO @ArthurZ in version 5, special tokens should be handled in convert_tokens_to_string, while _convert_tokens_to_string
        for token in filtered_tokens:
            # 如果skip_special_tokens为True且token是特殊token，则跳过
            if skip_special_tokens and token in self.all_special_ids:
                continue
            # 如果token是legacy_added_tokens中的token
            if token in legacy_added_tokens:
                if current_sub_text:
                    # 将当前的sub_text转换为字符串并添加到sub_texts中
                    string = self.convert_tokens_to_string(current_sub_text)
                    if len(string) > 0:
                        sub_texts.append(string)
                    current_sub_text = []
                # 将token直接添加到sub_texts中
                sub_texts.append(token)
            else:
                # 将token添加到current_sub_text中
                current_sub_text.append(token)
        
        # 如果current_sub_text非空，则将其转换为字符串并添加到sub_texts中
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        # 如果spaces_between_special_tokens为True，则用空格连接sub_texts中的字符串
        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            # 否则将sub_texts中的字符串连接起来
            text = "".join(sub_texts)

        # 如果clean_up_tokenization_spaces不为None，则使用其值；否则使用self.clean_up_tokenization_spaces的值
        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        
        # 如果clean_up_tokenization_spaces为True，则使用clean_up_tokenization方法清理text并返回
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            # 否则直接返回text
            return text
```