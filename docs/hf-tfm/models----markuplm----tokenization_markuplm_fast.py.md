# `.\transformers\models\markuplm\tokenization_markuplm_fast.py`

```
# 设置文件编码
# 版权声明
# 根据Apache许可证2.0版发布，仅允许在遵守许可证的情况下使用该文件
# 可在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 如果适用法律或书面同意要求，软件将按"现状"分发，没有任何形式的担保或条件，无论是明示的还是默示的
# 请查看特定语言的许可证，以了解权限和限制
"""
MARKUPLM的快速标记类。它覆盖了慢速标记器类的2个方法，即_batch_encode_plus和_encode_plus，在其中使用了Rust标记器。
"""

import json
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

from tokenizers import pre_tokenizers, processors

# 导入文件工具和其他标记化类
from ...file_utils import PaddingStrategy, TensorType, add_end_docstrings
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
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_markuplm import MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING, MarkupLMTokenizer

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义标记文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 预训练标记文件的映射字典
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

# 预训练位置嵌入大小的映射字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/markuplm-base": 512,
    "microsoft/markuplm-large": 512,
}

# 用LRU缓存装饰器缓存函数的返回值
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on. The reversible bpe codes work on unicode strings. This means you need a large #
    of unicode characters in your vocab if you want to avoid UNKs. When you're at something like a 10B token dataset
    you end up needing around 5K for decent coverage. This is a significant percentage of your normal, say, 32K bpe
    vocab. To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    # 遍历 0 到 2^8-1 的整数
    for b in range(2**8):
        # 如果 b 不在 bs 列表中
        if b not in bs:
            # 将 b 添加到 bs 列表
            bs.append(b)
            # 将 2^8 + n 添加到 cs 列表
            cs.append(2**8 + n)
            # n 自增 1
            n += 1
    # 将 cs 列表中的整数转换为对应的字符，组成新的列表
    cs = [chr(n) for n in cs]
    # 使用 bs 和 cs 两个列表创建一个字典并返回
    return dict(zip(bs, cs))
# 定义一个函数，返回单词中的符号对集合。单词表示为符号元组（符号为可变长度字符串）。
def get_pairs(word):
    # 初始化一个空集合用于存储符号对
    pairs = set()
    # 获取单词的第一个符号作为前一个字符
    prev_char = word[0]
    # 遍历单词中除了第一个符号之外的所有符号
    for char in word[1:]:
        # 将前一个字符和当前字符组成一个符号对，并添加到符号对集合中
        pairs.add((prev_char, char))
        # 更新前一个字符为当前字符，用于下一轮迭代
        prev_char = char
    # 返回符号对集合
    return pairs

# 声明一个类MarkupLMTokenizerFast，继承自PreTrainedTokenizerFast
class MarkupLMTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个MarkupLM分词器。基于字节级别的字节对编码（BPE）。

    [`MarkupLMTokenizerFast`]可用于将HTML字符串转换为标记级输入`input_ids`、`attention_mask`、`token_type_ids`、`xpath_tags_seq`和`xpath_tags_seq`。
    该分词器继承自[`PreTrainedTokenizer`]，其中包含大部分主要方法。

    用户应参考此超类以获取有关这些方法的更多信息。
    """
    # 参数说明：词汇表文件路径
    vocab_file (`str`):
        Path to the vocabulary file.
    # 参数说明：合并文件路径
    merges_file (`str`):
        Path to the merges file.
    # 参数说明：解码字节到 UTF-8 时遇到错误时的处理方式，默认为替换
    errors (`str`, *optional*, defaults to `"replace"`):
        Paradigm to follow when decoding bytes to UTF-8. See
        [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
    # 参数说明：序列开始标记，用于预训练时的起始标记，并可用于序列分类标记
    bos_token (`str`, *optional*, defaults to `"<s>"`):
        The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
    # <Tip>提示：当构建序列时使用特殊标记时，不是用于序列开头的标记. 用于序列开头的是 `cls_token`.
    # 参数说明：序列结束标记
    eos_token (`str`, *optional*, defaults to `"</s>"`):
        The end of sequence token.
    # <Tip>提示：当构建序列使用特殊标记时，不是用于序列结束的标记. 用于序列结束的是 `sep_token`.
    # 参数说明：分隔标记，用于构建多个序列为一个序列，例如序列分类或问题回答时的分隔标记
    sep_token (`str`, *optional*, defaults to `"</s>"`):
        The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
        sequence classification or for a text and a question for question answering. It is also used as the last
        token of a sequence built with special tokens.
    # 参数说明：分类标记，用于序列分类任务（整个序列分类而不是每个标记分类）
    cls_token (`str`, *optional*, defaults to `"<s>"`):
        The classifier token which is used when doing sequence classification (classification of the whole sequence
        instead of per-token classification). It is the first token of the sequence when built with special tokens.
    # 参数说明：未知标记，表示不在词汇表中的标记
    unk_token (`str`, *optional*, defaults to `"<unk>"`):
        The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
        token instead.
    # 参数说明：填充标记，用于填充不同长度的序列
    pad_token (`str`, *optional*, defaults to `"<pad>"`):
        The token used for padding, for example when batching sequences of different lengths.
    # 参数说明：掩码标记，用于掩码语言模型训练时模型尝试预测的标记
    mask_token (`str`, *optional*, defaults to `"<mask>"`):
        The token used for masking values. This is the token used when training this model with masked language
        modeling. This is the token which the model will try to predict.
    # 参数说明：是否在输入前添加初始空格，允许将开头的单词视为任何其他单词
    add_prefix_space (`bool`, *optional*, defaults to `False`):
        Whether or not to add an initial space to the input. This allows to treat the leading word just as any
        other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
    """

    # 词汇表文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练位置嵌入的最大输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 慢速分词器类
    slow_tokenizer_class = MarkupLMTokenizer
    # 初始化方法，用于设置各种参数
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
    )
    # 给定xpath表达式，返回标签ID和相应的下标列表，考虑到最大深度
    def get_xpath_seq(self, xpath):
        """
        Given the xpath expression of one particular node (like "/html/body/div/li[1]/div/span[2]"), return a list of
        tag IDs and corresponding subscripts, taking into account max depth.
        """
        xpath_tags_list = []
        xpath_subs_list = []

        # 根据"/"分割xpath表达式
        xpath_units = xpath.split("/")
        for unit in xpath_units:
            if not unit.strip():
                continue
            # 将标签名和下标分割开
            name_subs = unit.strip().split("[")
            tag_name = name_subs[0]
            sub = 0 if len(name_subs) == 1 else int(name_subs[1][:-1])
            # 获取标签ID和下标，如果标签名不存在，则使用未知标签ID
            xpath_tags_list.append(self.tags_dict.get(tag_name, self.unk_tag_id))
            xpath_subs_list.append(min(self.max_width, sub))

        # 限制标签ID和下标列表的长度，超过最大深度则截断，不足则使用填充标签ID和填充宽度进行填充
        xpath_tags_list = xpath_tags_list[: self.max_depth]
        xpath_subs_list = xpath_subs_list[: self.max_depth]
        xpath_tags_list += [self.pad_tag_id] * (self.max_depth - len(xpath_tags_list))
        xpath_subs_list += [self.pad_width] * (self.max_depth - len(xpath_subs_list))

        return xpath_tags_list, xpath_subs_list

    # 编码文本或文本对，返回相应的输入编码
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
    )
    # 编码参数文档
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,
        xpaths: Optional[List[List[List[int]]]] = None,
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
    ) -> BatchEncoding:
        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略以及最大长度参数，并更新kwargs
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

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

    def tokenize(self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False, **kwargs) -> List[str]:
        # 将单个文本或文本对组成批量输入
        batched_input = [(text, pair)] if pair else [text]
        # 使用tokenizer的encode_batch方法对批量输入进行编码
        encodings = self._tokenizer.encode_batch(
            batched_input, add_special_tokens=add_special_tokens, is_pretokenized=False, **kwargs
        )
        # 返回编码结果中第一个元素的tokens列表
        return encodings[0].tokens

    # 添加结束文档字符串参数
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
```  
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 定义参数 text 的类型为 Union[TextInput, PreTokenizedInput]
        text_pair: Optional[PreTokenizedInput] = None,  # 定义参数 text_pair 的类型为 Optional[PreTokenizedInput]，默认为 None
        xpaths: Optional[List[List[int]]] = None,  # 定义参数 xpaths 的类型为 Optional[List[List[int]]]，默认为 None
        node_labels: Optional[List[int]] = None,  # 定义参数 node_labels 的类型为 Optional[List[int]]，默认为 None
        add_special_tokens: bool = True,  # 定义参数 add_special_tokens 的类型为 bool，默认为 True
        padding: Union[bool, str, PaddingStrategy] = False,  # 定义参数 padding 的类型为 Union[bool, str, PaddingStrategy]，默认为 False
        truncation: Union[bool, str, TruncationStrategy] = None,  # 定义参数 truncation 的类型为 Union[bool, str, TruncationStrategy]，默认为 None
        max_length: Optional[int] = None,  # 定义参数 max_length 的类型为 Optional[int]，默认为 None
        stride: int = 0,  # 定义参数 stride 的类型为 int，默认为 0
        pad_to_multiple_of: Optional[int] = None,  # 定义参数 pad_to_multiple_of 的类型为 Optional[int]，默认为 None
        return_tensors: Optional[Union[str, TensorType]] = None,  # 定义参数 return_tensors 的类型为 Optional[Union[str, TensorType]]，默认为 None
        return_token_type_ids: Optional[bool] = None,  # 定义参数 return_token_type_ids 的类型为 Optional[bool]，默认为 None
        return_attention_mask: Optional[bool] = None,  # 定义参数 return_attention_mask 的类型为 Optional[bool]，默认为 None
        return_overflowing_tokens: bool = False,  # 定义参数 return_overflowing_tokens 的类型为 bool，默认为 False
        return_special_tokens_mask: bool = False,  # 定义参数 return_special_tokens_mask 的类型为 bool，默认为 False
        return_offsets_mapping: bool = False,  # 定义参数 return_offsets_mapping 的类型为 bool，默认为 False
        return_length: bool = False,  # 定义参数 return_length 的类型为 bool，默认为 False
        verbose: bool = True,  # 定义参数 verbose 的类型为 bool，默认为 True
        **kwargs,  # 接收额外的关键字参数
    ) -> BatchEncoding:  # 指定返回类型为 BatchEncoding
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

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获得填充和截断策略，同时处理旧版本的参数名，并且更新 kwargs
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用 _encode_plus 方法，并将所有参数传递过去
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
            **kwargs,  # 传递额外的关键字参数
        )
    class Tokenizer:
        # 对一批文本或文本对进行编码
        def _batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[
                List[TextInput],  # 批处理的文本或文本对
                List[TextInputPair],  # 批处理的文本对
                List[PreTokenizedInput],  # 预分词输入
            ],
            is_pair: bool = None,  # 标记是否为文本对
            xpaths: Optional[List[List[List[int]]] = None,  # XPath列表的列表，用于指定与文本相关的标签路径
            node_labels: Optional[List[List[int]] = None,  # 节点标签列表的列表
            add_special_tokens: bool = True,  # 是否添加特殊标记
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
            max_length: Optional[int] = None,  # 最大长度限制
            stride: int = 0,  # 步长
            pad_to_multiple_of: Optional[int] = None,  # 填充至多少的倍数
            return_tensors: Optional[str] = None,  # 返回的张量类型
            return_token_type_ids: Optional[bool] = None,  # 是否返回标记类型ID
            return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
            return_overflowing_tokens: bool = False,  # 是否返回溢出的标记
            return_special_tokens_mask: bool = False,  # 是否返回特殊标记掩码
            return_offsets_mapping: bool = False,  # 是否返回偏移映射
            return_length: bool = False,  # 是否返回长度
            verbose: bool = True,  # 是否启用详细输出
        # 对单个文本或文本对进行编码
        def _encode_plus(
            self,
            text: Union[TextInput, PreTokenizedInput],  # 文本输入
            text_pair: Optional[PreTokenizedInput] = None,  # 可选的文本对输入
            xpaths: Optional[List[List[int]]] = None,  # XPath列表，用于指定与文本相关的标签路径
            node_labels: Optional[List[int]] = None,  # 节点标签列表
            add_special_tokens: bool = True,  # 是否添加特殊标记
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略
            truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
            max_length: Optional[int] = None,  # 最大长度限制
            stride: int = 0,  # 步长
            pad_to_multiple_of: Optional[int] = None,  # 填充至多少的倍数
            return_tensors: Optional[bool] = None,  # 返回的张量类型
            return_token_type_ids: Optional[bool] = None,  # 是否返回标记类型ID
            return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
            return_overflowing_tokens: bool = False,  # 是否返回溢出的标记
            return_special_tokens_mask: bool = False,  # 是否返回特殊标记掩码
            return_offsets_mapping: bool = False,  # 是否返回偏移映射
            return_length: bool = False,  # 是否返回长度
            verbose: bool = True,  # 是否启用详细输出
            **kwargs,  # 其他关键字参数
    )-> BatchEncoding:
        # 将输入组成批量输入
        # 两种选项:
        # 1) 只有文本，如果文本必须是一个字符串列表
        # 2) 文本 + 文本对，此时文本 = 字符串，文本对 = 字符串列表
        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_xpaths = [xpaths]
        batched_node_labels = [node_labels] if node_labels is not None else None
        # 调用_batch_encode_plus方法进行批量编码
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

        # 如果返回张量为None，则可以移除前导批量轴
        # 溢出的标记作为一批输出返回，所以在这种情况下保留它们
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

        # 返回批量输出
        return batched_output

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
```  
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A RoBERTa sequence has the following format:
        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            # 如果只有一个 token_ids，则返回特殊 token 前后加上特殊 token 的 ID 的列表
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 创建分别包含特殊 token ID 的列表
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 返回将两个 token_ids 合并，并添加特殊 token ID 后的列表
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
            `List[int]`: List of zeros.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            # 如果只有一个 token_ids，则返回零列表的长度
            return len(cls + token_ids_0 + sep) * [0]
        # 如果有两个 token_ids，返回零列表的长度
        return len(cls + token_ids_0 + sep + token_ids_1 + sep) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 保存 tokenizer 的模型到指定目录，根据指定的前缀命名
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回保存的文件路径元组
        return tuple(files)
```