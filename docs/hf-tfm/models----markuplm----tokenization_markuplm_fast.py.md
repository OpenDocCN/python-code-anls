# `.\models\markuplm\tokenization_markuplm_fast.py`

```
# 设置文件编码格式为 UTF-8
# 版权声明：2022 年 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本使用本文件；
# 除非符合许可证的规定，否则不得使用本文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于"原样"分发的，无论明示或默示的保证或条件。
# 有关详细信息，请参阅许可证。
"""
MarkupLM 的快速标记类。它重写了慢速标记器类的两个方法，即 _batch_encode_plus 和 _encode_plus，
在这些方法中使用了 Rust 标记器。
"""

import json
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

from tokenizers import pre_tokenizers, processors

# 导入文件工具和常量定义
from ...file_utils import PaddingStrategy, TensorType, add_end_docstrings
# 导入基础标记工具类
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    AddedToken,
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
# 导入快速标记工具类
from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 导入日志工具
from ...utils import logging
# 导入特定的标记化类
from .tokenization_markuplm import MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING, MarkupLMTokenizer

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/markuplm-base": "https://huggingface.co/microsoft/markuplm-base/resolve/main/vocab.json",
        "microsoft/markuplm-large": "https://huggingface.co/microsoft/markuplm-large/resolve/main/vocab.json",
    },
    "merges_file": {
        "microsoft/markuplm-base": "https://huggingface.co/microsoft/markuplm-base/resolve/main/merges.txt",
        "microsoft/markuplm-large": "https://huggingface.co/microsoft/markuplm-large/resolve/main/merges.txt",
    },
}

# 定义预训练位置嵌入的尺寸映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/markuplm-base": 512,
    "microsoft/markuplm-large": 512,
}

@lru_cache()
def bytes_to_unicode():
    """
    返回 utf-8 字节列表及其映射到 unicode 字符串的映射表。我们特别避免映射到空格或控制字符，以免在 bpe 编码时出错。
    可逆的 bpe 编码适用于 unicode 字符串。这意味着如果您希望避免 UNKs，您需要在词汇中包含大量的 unicode 字符。
    当您处理类似 10B 令牌数据集时，您可能需要约 5K 个 unicode 字符以获得良好的覆盖率。
    这相当于正常 32K bpe 词汇表的显著比例。为了避免这种情况，我们希望在 utf-8 字节和 unicode 字符串之间建立查找表。
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    # 循环迭代范围在0到255之间的整数
    for b in range(2**8):
        # 如果当前整数不在列表bs中
        if b not in bs:
            # 将当前整数b添加到bs列表中
            bs.append(b)
            # 向cs列表中添加新的元素，该元素为2的8次方加上当前迭代次数n
            cs.append(2**8 + n)
            # 增加迭代计数n的值
            n += 1
    
    # 将cs列表中的每个整数转换为对应的Unicode字符，并形成一个新的列表cs
    cs = [chr(n) for n in cs]
    
    # 使用zip函数将bs列表和cs列表中的元素一一配对，然后生成一个字典
    return dict(zip(bs, cs))
# 定义一个函数，用于获取单词中的符号对集合。这里假设单词是由符号元组表示的，每个符号可以是长度可变的字符串。
def get_pairs(word):
    """
    Return set of symbol pairs in a word. Word is represented as tuple of symbols (symbols being variable-length
    strings).
    """
    # 初始化一个空集合，用于存储符号对
    pairs = set()
    # 获取单词的第一个符号作为前一个符号
    prev_char = word[0]
    # 遍历单词中除第一个符号之外的所有符号
    for char in word[1:]:
        # 将前一个符号和当前符号作为一个符号对加入到集合中
        pairs.add((prev_char, char))
        # 更新前一个符号为当前符号，以便下一次迭代使用
        prev_char = char
    # 返回所有的符号对集合
    return pairs


class MarkupLMTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a MarkupLM tokenizer. Based on byte-level Byte-Pair-Encoding (BPE).

    [`MarkupLMTokenizerFast`] can be used to turn HTML strings into to token-level `input_ids`, `attention_mask`,
    `token_type_ids`, `xpath_tags_seq` and `xpath_tags_seq`. This tokenizer inherits from [`PreTrainedTokenizer`] which
    contains most of the main methods.

    Users should refer to this superclass for more information regarding those methods.
    """
    # 导入所需的库或模块
    Args:
        vocab_file (`str`):
            # 词汇表文件的路径。
        merges_file (`str`):
            # 合并文件的路径。
        errors (`str`, *optional*, defaults to `"replace"`):
            # 解码字节为 UTF-8 时遇到错误的处理方式。详见 Python 文档中的 bytes.decode 描述。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            # 在预训练期间用作序列开头的特殊标记。也可用作序列分类器的标记。
            # <Tip>提示：在使用特殊标记构建序列时，并非使用此标记作为序列的开头标记。实际上使用的是 `cls_token`。</Tip>
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            # 序列结尾的特殊标记。
            # <Tip>提示：在使用特殊标记构建序列时，并非使用此标记作为序列的结尾标记。实际上使用的是 `sep_token`。</Tip>
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            # 分隔符标记，在构建来自多个序列的序列时使用，例如序列分类或问题回答中的文本和问题。同时也用作使用特殊标记构建序列的最后一个标记。
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            # 分类器标记，在进行序列分类（整个序列而不是每个标记的分类）时使用。在使用特殊标记构建序列时，它是序列的第一个标记。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            # 未知标记。如果词汇表中不存在的标记，将无法将其转换为 ID，而会被设置为此标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            # 用于填充的标记，例如在对不同长度的序列进行批处理时使用。
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            # 用于掩码值的标记。在使用掩码语言建模训练模型时使用，模型将尝试预测此标记。
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            # 是否在输入之前添加一个初始空格。这样可以像对待其他单词一样对待前导单词。（RoBERTa 分词器通过前导空格来检测单词的开头）。
    
    # 以下变量可能为预训练模型配置的文件和大小映射提供了默认值
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = MarkupLMTokenizer
    def __init__(
        self,
        vocab_file,
        merges_file,
        tags_dict,
        tokenizer_file=None,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        max_depth=50,
        max_width=1000,
        pad_width=1001,
        pad_token_label=-100,
        only_label_first_subword=True,
        trim_offsets=False,
        **kwargs,
    ):
        """
        Initialize the class with required and optional parameters for tokenization and tagging.

        Args:
            vocab_file (str): Path to vocabulary file.
            merges_file (str): Path to merges file for tokenization.
            tags_dict (dict): Dictionary mapping tag names to IDs.
            tokenizer_file (str, optional): Path to tokenizer file. Defaults to None.
            errors (str, optional): Error handling method during tokenization. Defaults to "replace".
            bos_token (str, optional): Beginning of sequence token. Defaults to "<s>".
            eos_token (str, optional): End of sequence token. Defaults to "</s>".
            sep_token (str, optional): Separator token. Defaults to "</s>".
            cls_token (str, optional): Classification token. Defaults to "<s>".
            unk_token (str, optional): Unknown token. Defaults to "<unk>".
            pad_token (str, optional): Padding token. Defaults to "<pad>".
            mask_token (str, optional): Mask token. Defaults to "<mask>".
            add_prefix_space (bool, optional): Whether to add prefix space during tokenization. Defaults to False.
            max_depth (int, optional): Maximum depth for XPath processing. Defaults to 50.
            max_width (int, optional): Maximum width for XPath processing. Defaults to 1000.
            pad_width (int, optional): Padding width for XPath processing. Defaults to 1001.
            pad_token_label (int, optional): Padding token label for subword tagging. Defaults to -100.
            only_label_first_subword (bool, optional): Whether to label only the first subword. Defaults to True.
            trim_offsets (bool, optional): Whether to trim offsets. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        pass

    def get_xpath_seq(self, xpath):
        """
        Given the xpath expression of one particular node (like "/html/body/div/li[1]/div/span[2]"), return a list of
        tag IDs and corresponding subscripts, taking into account max depth.
        """
        xpath_tags_list = []
        xpath_subs_list = []

        xpath_units = xpath.split("/")
        for unit in xpath_units:
            if not unit.strip():
                continue
            name_subs = unit.strip().split("[")
            tag_name = name_subs[0]
            sub = 0 if len(name_subs) == 1 else int(name_subs[1][:-1])
            xpath_tags_list.append(self.tags_dict.get(tag_name, self.unk_tag_id))
            xpath_subs_list.append(min(self.max_width, sub))

        xpath_tags_list = xpath_tags_list[: self.max_depth]
        xpath_subs_list = xpath_subs_list[: self.max_depth]
        xpath_tags_list += [self.pad_tag_id] * (self.max_depth - len(xpath_tags_list))
        xpath_subs_list += [self.pad_width] * (self.max_depth - len(xpath_subs_list))

        return xpath_tags_list, xpath_subs_list

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        xpaths: Union[List[List[int]], List[List[List[int]]]] = None,
        node_labels: Optional[Union[List[int], List[List[int]]]] = None,
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
        **kwargs,
    ):
        """
        Encode the input text(s) along with associated parameters into token IDs, token type IDs, and attention masks.

        Args:
            text (Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]): Input text or texts.
            text_pair (Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]], optional): Second input text or texts. Defaults to None.
            xpaths (Union[List[List[int]], List[List[List[int]]]], optional): List of XPath sequences. Defaults to None.
            node_labels (Optional[Union[List[int], List[List[int]]]], optional): Node labels corresponding to XPaths. Defaults to None.
            add_special_tokens (bool, optional): Whether to add special tokens. Defaults to True.
            padding (Union[bool, str, PaddingStrategy], optional): Padding strategy or maximum length for padding. Defaults to False.
            truncation (Union[bool, str, TruncationStrategy], optional): Truncation strategy or maximum length for truncation. Defaults to None.
            max_length (Optional[int], optional): Maximum length of the returned sequences. Defaults to None.
            stride (int, optional): Stride for overflowing tokens. Defaults to 0.
            pad_to_multiple_of (Optional[int], optional): Pad to a multiple of specified value. Defaults to None.
            return_tensors (Optional[Union[str, TensorType]], optional): Type of tensors to return. Defaults to None.
            return_token_type_ids (Optional[bool], optional): Whether to return token type IDs. Defaults to None.
            return_attention_mask (Optional[bool], optional): Whether to return attention mask. Defaults to None.
            return_overflowing_tokens (bool, optional): Whether to return overflowing tokens. Defaults to False.
            return_special_tokens_mask (bool, optional): Whether to return special tokens mask. Defaults to False.
            return_offsets_mapping (bool, optional): Whether to return offsets mapping. Defaults to False.
            return_length (bool, optional): Whether to return length of the encoded sequence. Defaults to False.
            verbose (bool, optional): Whether to output verbose information. Defaults to True.
            **kwargs: Additional keyword arguments.
        """
        pass
    # 定义一个方法用于批量编码文本或文本对，并返回批编码结果
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,  # 是否为文本对
        xpaths: Optional[List[List[List[int]]]] = None,  # XPath信息，用于处理HTML/XML类型的输入
        node_labels: Optional[Union[List[int], List[List[int]]]] = None,  # 节点标签信息
        add_special_tokens: bool = True,  # 是否添加特殊标记（如[CLS]和[SEP]）
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略，可以是布尔值、字符串或填充策略对象
        truncation: Union[bool, str, TruncationStrategy] = None,  # 截断策略，可以是布尔值、字符串或截断策略对象
        max_length: Optional[int] = None,  # 最大长度限制
        stride: int = 0,  # 滑动窗口的步长
        pad_to_multiple_of: Optional[int] = None,  # 将序列填充到某个整数的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回token_type_ids
        return_attention_mask: Optional[bool] = None,  # 是否返回attention_mask
        return_overflowing_tokens: bool = False,  # 是否返回溢出的tokens
        return_special_tokens_mask: bool = False,  # 是否返回特殊token的掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回序列长度
        verbose: bool = True,  # 是否打印详细信息
        **kwargs,  # 其它关键字参数
    ) -> BatchEncoding:
        # 处理'padding'、'truncation'、'max_length'等参数，保证向后兼容性
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法进行批量编码处理，并返回结果
        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            is_pair=is_pair,
            xpaths=xpaths,
            node_labels=node_labels,
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

    # 定义一个方法用于将单个文本（或文本对）进行标记化处理，并返回标记化结果
    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        batched_input = [(text, pair)] if pair else [text]
        # 调用_tokenizer的encode_batch方法，将文本（或文本对）进行批量编码
        encodings = self._tokenizer.encode_batch(
            batched_input, add_special_tokens=add_special_tokens, is_pretokenized=False, **kwargs
        )

        # 返回编码结果中的第一个序列的token列表
        return encodings[0].tokens

    # 应用函数装饰器，添加文档字符串到下面的函数中
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        xpaths: Optional[List[List[int]]] = None,
        node_labels: Optional[List[int]] = None,
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
        **kwargs,
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences. .. warning:: This method is deprecated,
        `__call__` should be used instead.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The first sequence to be encoded. This can be a string, a list of strings or a list of list of strings.
            text_pair (`List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a list of strings (words of a single example) or a
                list of list of strings (words of a batch of examples).
        """

        # 获取填充和截断策略，同时处理过时的参数以保证向后兼容性
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法 `_encode_plus` 进行实际的编码和准备工作
        return self._encode_plus(
            text=text,
            xpaths=xpaths,
            text_pair=text_pair,
            node_labels=node_labels,
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
    # 定义一个方法用于批量编码文本或文本对，支持多种输入类型
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,  # 是否是文本对
        xpaths: Optional[List[List[List[int]]]] = None,  # XPath 路径列表，用于定位文本在原始数据中的位置
        node_labels: Optional[List[List[int]]] = None,  # 节点标签列表，用于标识文本对应的节点信息
        add_special_tokens: bool = True,  # 是否添加特殊标记（如[CLS]和[SEP]）
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 最大长度限制
        stride: int = 0,  # 滑动窗口步长
        pad_to_multiple_of: Optional[int] = None,  # 填充长度的倍数
        return_tensors: Optional[str] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回token type ids
        return_attention_mask: Optional[bool] = None,  # 是否返回attention mask
        return_overflowing_tokens: bool = False,  # 是否返回溢出的 tokens
        return_special_tokens_mask: bool = False,  # 是否返回特殊 tokens 的 mask
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回长度
        verbose: bool = True,  # 是否显示详细信息
    ):
        # 定义一个方法用于编码单个文本或文本对
        def _encode_plus(
            self,
            text: Union[TextInput, PreTokenizedInput],  # 输入的文本或预分词的输入
            text_pair: Optional[PreTokenizedInput] = None,  # 可选的文本对
            xpaths: Optional[List[List[int]]] = None,  # XPath 路径列表，用于定位文本在原始数据中的位置
            node_labels: Optional[List[int]] = None,  # 节点标签列表，用于标识文本对应的节点信息
            add_special_tokens: bool = True,  # 是否添加特殊标记（如[CLS]和[SEP]）
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认不填充
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认不截断
            max_length: Optional[int] = None,  # 最大长度限制
            stride: int = 0,  # 滑动窗口步长
            pad_to_multiple_of: Optional[int] = None,  # 填充长度的倍数
            return_tensors: Optional[bool] = None,  # 返回的张量类型
            return_token_type_ids: Optional[bool] = None,  # 是否返回token type ids
            return_attention_mask: Optional[bool] = None,  # 是否返回attention mask
            return_overflowing_tokens: bool = False,  # 是否返回溢出的 tokens
            return_special_tokens_mask: bool = False,  # 是否返回特殊 tokens 的 mask
            return_offsets_mapping: bool = False,  # 是否返回偏移映射
            return_length: bool = False,  # 是否返回长度
            verbose: bool = True,  # 是否显示详细信息
            **kwargs,  # 其他关键字参数
        ):
    ) -> BatchEncoding:
        # 将输入组成批处理输入
        # 两种选项：
        # 1) 只有文本，如果文本是字符串列表，则 text 必须是列表
        # 2) 文本 + 文本对，此时 text 是字符串，text_pair 是字符串列表
        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_xpaths = [xpaths]
        batched_node_labels = [node_labels] if node_labels is not None else None
        # 调用 _batch_encode_plus 方法进行批处理编码
        batched_output = self._batch_encode_plus(
            batched_input,
            is_pair=bool(text_pair is not None),
            xpaths=batched_xpaths,
            node_labels=batched_node_labels,
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

        # 如果 return_tensors 是 None 并且不返回 overflowing tokens，则移除首个批处理维度
        # 在这种情况下，overflowing tokens 作为输出的一个批次被返回，因此保留它们
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,
            )

        # 检查并警告关于过长序列的情况
        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

        # 返回批处理输出
        return batched_output

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ):
        # 该方法用于对编码输入进行填充
        # 返回填充后的编码输入
        pass

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens. A RoBERTa sequence has the following format:
        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens added.
        """
        if token_ids_1 is None:
            # Return a single sequence with special tokens `<s>` (CLS), sequence tokens, and `</s>` (SEP)
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        
        # For a pair of sequences, concatenate special tokens `<s>` (CLS), sequence 1 tokens, `</s>` (SEP),
        # sequence 2 tokens, and another `</s>` (SEP) at the end
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of zeros representing token type ids (not used in RoBERTa).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            # Return zeros for token type ids for a single sequence with special tokens
            return len(cls + token_ids_0 + sep) * [0]
        
        # Return zeros for token type ids for a pair of sequences with special tokens
        return len(cls + token_ids_0 + sep + token_ids_1 + sep) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the tokenizer's vocabulary to a directory.

        Args:
            save_directory (str):
                Directory where the vocabulary files will be saved.
            filename_prefix (str, *optional*):
                Optional prefix for the saved files.

        Returns:
            Tuple[str]: Tuple containing the saved file paths.
        """
        # Save the model's vocabulary files to the specified directory with an optional filename prefix
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```