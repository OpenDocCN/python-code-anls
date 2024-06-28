# `.\models\markuplm\tokenization_markuplm.py`

```
# coding=utf-8
# 版权 Microsoft Research 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache License, Version 2.0 授权，除非符合许可，否则不得使用此文件。
# 您可以在以下网址获取许可的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发的软件，
# 没有任何形式的担保或条件，包括但不限于对适销性和特定用途的隐含担保。
# 有关详细信息，请参阅许可证。
"""MarkupLM 的标记化类。"""

import json  # 导入 JSON 库
import os    # 导入操作系统相关功能
from functools import lru_cache  # 导入 lru_cache 装饰器
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示

import regex as re  # 导入正则表达式库

from ...file_utils import PaddingStrategy, TensorType, add_end_docstrings  # 从文件工具中导入相关功能
from ...tokenization_utils import AddedToken, PreTrainedTokenizer  # 导入标记化工具
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)  # 导入标记化基础工具
from ...utils import logging  # 导入日志工具


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

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

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/markuplm-base": 512,
    "microsoft/markuplm-large": 512,
}


@lru_cache()
def bytes_to_unicode():
    """
    返回 utf-8 字节列表及其对应的 Unicode 字符映射。
    避免映射到空白字符和控制字符，以免在 bpe 编码中出错。
    可逆的 bpe 编码适用于 Unicode 字符串，因此如果要避免 UNKs，
    则需要在词汇表中包含大量的 Unicode 字符。
    例如，对于大约 100 亿个令牌的数据集，您需要大约 5000 个 Unicode 字符以确保良好的覆盖率。
    这相当于您正常使用的 32K bpe 词汇表的显著百分比。
    为了避免这种情况，我们需要 utf-8 字节和 Unicode 字符串之间的查找表。
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    # 返回单词中所有相邻字符对组成的集合。这里单词被表示为由符号元组构成（符号是长度可变的字符串）。
    def get_symbol_pairs(word):
        # 初始化一个空集合来存放符号对
        pairs = set()
        # 从单词的第一个符号开始迭代到最后一个符号
        prev_char = word[0]  # 获取单词的第一个符号作为前一个符号
        for char in word[1:]:  # 从第二个符号开始迭代到最后一个符号
            # 将前一个符号和当前符号组成一个符号对，并添加到集合中
            pairs.add((prev_char, char))
            # 更新前一个符号为当前符号，以便下一次迭代使用
            prev_char = char
        # 返回包含所有符号对的集合
        return pairs
class MarkupLMTokenizer(PreTrainedTokenizer):
    r"""
    Construct a MarkupLM tokenizer. Based on byte-level Byte-Pair-Encoding (BPE). [`MarkupLMTokenizer`] can be used to
    turn HTML strings into to token-level `input_ids`, `attention_mask`, `token_type_ids`, `xpath_tags_seq` and
    `xpath_tags_seq`. This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.
    """
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
    """
    # 载入预定义的词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型词汇文件的映射表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练位置嵌入大小的映射表
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 初始化方法，用于实例化对象
    def __init__(
        self,
        vocab_file,
        merges_file,
        tags_dict,
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
        **kwargs,
    ):
        # 初始化方法参数，设置对象的属性
        ...

    def get_xpath_seq(self, xpath):
        """
        根据给定的 xpath 表达式（如 "/html/body/div/li[1]/div/span[2]"），返回标签 ID 和对应的下标列表，考虑最大深度限制。
        """
        # 初始化空列表，用于存储 xpath 表达式中的标签 ID 和下标
        xpath_tags_list = []
        xpath_subs_list = []

        # 按"/"分割 xpath 表达式
        xpath_units = xpath.split("/")
        for unit in xpath_units:
            # 如果单元为空，则跳过
            if not unit.strip():
                continue
            # 分割标签名和下标（如果有）
            name_subs = unit.strip().split("[")
            tag_name = name_subs[0]
            # 下标默认为0，如果存在则取其整数形式
            sub = 0 if len(name_subs) == 1 else int(name_subs[1][:-1])
            # 获取标签名对应的标签 ID，如果不存在则使用默认的未知标签 ID
            xpath_tags_list.append(self.tags_dict.get(tag_name, self.unk_tag_id))
            # 下标取最大宽度和实际值的较小者
            xpath_subs_list.append(min(self.max_width, sub))

        # 限制列表长度不超过最大深度
        xpath_tags_list = xpath_tags_list[: self.max_depth]
        xpath_subs_list = xpath_subs_list[: self.max_depth]
        # 如果列表长度不足最大深度，使用填充标签 ID 和填充宽度进行填充
        xpath_tags_list += [self.pad_tag_id] * (self.max_depth - len(xpath_tags_list))
        xpath_subs_list += [self.pad_width] * (self.max_depth - len(xpath_subs_list))

        # 返回标签 ID 列表和下标列表
        return xpath_tags_list, xpath_subs_list

    @property
    def vocab_size(self):
        # 返回编码器的大小，即词汇表的大小
        return len(self.encoder)

    def get_vocab(self):
        # 获取词汇表，包括原始编码器和额外添加的编码器
        vocab = self.encoder.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab
    def bpe(self, token):
        # 如果 token 已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 将 token 转换为字符元组
        word = tuple(token)
        # 获取 token 的所有字符对
        pairs = get_pairs(word)

        # 如果没有字符对，则直接返回 token
        if not pairs:
            return token

        # 进入循环，直到 token 无法再进行 BPE 分割
        while True:
            # 找出当前字符对中频率最低的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果该字符对不在 BPE 词汇表中，则跳出循环
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 遍历 token 中的字符
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    # 如果找不到 first，则将剩余部分添加到 new_word 中并结束循环
                    new_word.extend(word[i:])
                    break
                else:
                    # 将 first 之前的部分添加到 new_word 中
                    new_word.extend(word[i:j])
                    i = j

                # 如果当前字符是 first 并且下一个字符是 second，则将它们合并为一个新的字符添加到 new_word 中
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    # 否则将当前字符直接添加到 new_word 中
                    new_word.append(word[i])
                    i += 1
            # 更新 word 为新的字符元组
            new_word = tuple(new_word)
            word = new_word
            # 如果 word 只剩一个字符，则跳出循环
            if len(word) == 1:
                break
            else:
                # 否则继续生成新的字符对进行下一轮合并
                pairs = get_pairs(word)
        # 将最终合并后的字符元组转换为字符串形式
        word = " ".join(word)
        # 将 token 及其对应的合并结果存入缓存中
        self.cache[token] = word
        # 返回最终合并后的字符串
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        # 初始化空列表用于存储 BPE 分割后的 token
        bpe_tokens = []
        # 使用正则表达式按照 self.pat 提供的模式将 text 分割成 token
        for token in re.findall(self.pat, text):
            # 将每个 token 编码成字节序列，然后映射到 Unicode 字符串，避免 BPE 中的控制符号（在我们的情况下是空格）
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # 将所有字节映射为 Unicode 字符串，避免 BPE 中的控制符（在我们的情况下是空格）
            # 对每个经过 BPE 处理后的 token 进行拆分并添加到 bpe_tokens 列表中
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        # 返回最终的 BPE 分割后的 token 列表
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 根据 token 查找其在词汇表中对应的 id，如果找不到则返回 unk_token 对应的 id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 根据 index 查找其在词汇表中对应的 token
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 打印警告信息，提示目前不支持生成任务，解码是实验性的且可能会变化
        logger.warning(
            "MarkupLM now does not support generative tasks, decoding is experimental and subject to change."
        )
        # 将 tokens 列表中的 token 合并为一个字符串
        text = "".join(tokens)
        # 将合并后的字符串转换为字节序列，然后根据 byte_decoder 将其解码为 utf-8 编码的字符串，处理过程中可能会出现错误
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        # 返回最终解码后的文本字符串
        return text
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建词汇表文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构建合并文件路径
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 保存词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 保存合并文件
        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            # 遍历并按照 token_index 排序保存 BPE 合并信息
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # 检查是否需要在文本前添加空格，并处理传入的其他参数
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
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
            # 返回包含特殊 token 的单个序列输入
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 返回包含特殊 token 的序列对输入
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def build_xpath_tags_with_special_tokens(
        self, xpath_tags_0: List[int], xpath_tags_1: Optional[List[int]] = None
    ):
        # 待实现，用于构建带有特殊 token 的 XPath 标签序列
    ) -> List[int]:
        # 定义用于填充的特殊标记序列
        pad = [self.pad_xpath_tags_seq]
        # 如果 xpath_tags_1 的长度为 0，则返回前后添加填充标记后的 xpath_tags_0 序列
        if len(xpath_tags_1) == 0:
            return pad + xpath_tags_0 + pad
        # 否则返回前后添加填充标记后的 xpath_tags_0 序列，再加上 xpath_tags_1 序列和填充标记
        return pad + xpath_tags_0 + pad + xpath_tags_1 + pad

    def build_xpath_subs_with_special_tokens(
        self, xpath_subs_0: List[int], xpath_subs_1: Optional[List[int]] = None
    ) -> List[int]:
        # 定义用于填充的特殊标记序列
        pad = [self.pad_xpath_subs_seq]
        # 如果 xpath_subs_1 为 None 或者其长度为 0，则返回前后添加填充标记后的 xpath_subs_0 序列
        if len(xpath_subs_1) == 0:
            return pad + xpath_subs_0 + pad
        # 否则返回前后添加填充标记后的 xpath_subs_0 序列，再加上 xpath_subs_1 序列和填充标记
        return pad + xpath_subs_0 + pad + xpath_subs_1 + pad

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Args:
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        # 如果已经有特殊标记，则调用父类的方法返回特殊标记掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果 token_ids_1 为 None，则返回一个列表，以1开头，后接 token_ids_0 的长度个0，最后接1
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        # 否则返回一个列表，以1开头，后接 token_ids_0 的长度个0，再接两个1，再接 token_ids_1 的长度个0，最后接1
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

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
        # 定义用于分隔的特殊标记和类别标记
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # 如果 token_ids_1 为 None，则返回一个长度为 cls + token_ids_0 + sep 的列表，全为0
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则返回一个长度为 cls + token_ids_0 + sep + token_ids_1 + sep 的列表，全为0
        return len(cls + token_ids_0 + sep + token_ids_1 + sep) * [0]

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义一个方法，使对象可以被调用，接收多种文本输入格式以及其他参数
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
        # 调用 `batch_encode_plus` 方法，并传递所有参数
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
        ):
    ) -> BatchEncoding:
        # 获取填充和截断策略，支持旧版参数 'truncation_strategy' 和 'pad_to_max_length' 的兼容性
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

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,
        xpaths: Optional[List[List[List[int]]]] = None,
        node_labels: Optional[List[List[int]]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
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
        if return_offsets_mapping:
            # 如果要返回偏移映射，则抛出 NotImplementedError
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        # 调用 _batch_prepare_for_model 方法准备批量输入数据
        batch_outputs = self._batch_prepare_for_model(
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
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        # 返回 BatchEncoding 对象，封装批处理输出结果
        return BatchEncoding(batch_outputs)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def _batch_prepare_for_model(
        self,
        batch_text_or_text_pairs,
        is_pair: bool = None,
        xpaths: Optional[List[List[int]]] = None,
        node_labels: Optional[List[List[int]]] = None,
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
        manages a moving window (with user defined stride) for overflowing tokens.

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        # Initialize an empty dictionary to store batch outputs
        batch_outputs = {}

        # Iterate over each example in the batch, paired with xpaths
        for idx, example in enumerate(zip(batch_text_or_text_pairs, xpaths)):
            # Unpack the example into text or text pairs and xpaths
            batch_text_or_text_pair, xpaths_example = example

            # Call a method to prepare the inputs for the model
            outputs = self.prepare_for_model(
                batch_text_or_text_pair[0] if is_pair else batch_text_or_text_pair,  # First sequence or single sequence
                batch_text_or_text_pair[1] if is_pair else None,  # Second sequence (if pair) or None
                xpaths_example,  # XPath example for special handling
                node_labels=node_labels[idx] if node_labels is not None else None,  # Node labels if provided
                add_special_tokens=add_special_tokens,  # Whether to add special tokens
                padding=PaddingStrategy.DO_NOT_PAD.value,  # Padding strategy (no padding here)
                truncation=truncation_strategy.value,  # Truncation strategy for sequences
                max_length=max_length,  # Maximum length of sequences
                stride=stride,  # Stride for overflowing tokens
                pad_to_multiple_of=None,  # No padding to multiple of any specific number
                return_attention_mask=False,  # Do not return attention masks
                return_token_type_ids=return_token_type_ids,  # Whether to return token type IDs
                return_overflowing_tokens=return_overflowing_tokens,  # Whether to return overflowing tokens
                return_special_tokens_mask=return_special_tokens_mask,  # Whether to return special tokens mask
                return_length=return_length,  # Whether to return the length of sequences
                return_tensors=None,  # Do not convert batch to tensors immediately
                prepend_batch_axis=False,  # Do not prepend batch axis
                verbose=verbose,  # Verbosity level
            )

            # Aggregate outputs into batch_outputs dictionary
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # Pad the batch outputs according to specified padding strategy and maximum length
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,  # Padding strategy enumeration value
            max_length=max_length,  # Maximum length of sequences for padding
            pad_to_multiple_of=pad_to_multiple_of,  # Pad to multiple of specified value
            return_attention_mask=return_attention_mask,  # Whether to return attention masks
        )

        # Convert batch_outputs into a BatchEncoding object with specified tensor type
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        # Return the final prepared batch outputs
        return batch_outputs

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING)
    # 定义一个方法，用于将文本编码成模型可以接受的输入格式
    def encode(
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
    ) -> List[int]:
        # 调用 encode_plus 方法进行文本编码，并获取编码后的输入字典
        encoded_inputs = self.encode_plus(
            text=text,
            text_pair=text_pair,
            xpaths=xpaths,
            node_labels=node_labels,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
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

        # 返回编码结果中的输入 token IDs 列表
        return encoded_inputs["input_ids"]

    # 使用 add_end_docstrings 装饰器为 encode_plus 方法添加文档字符串
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
                Optional second sequence to be encoded. This can be a list of strings (nodes of a single example) or a
                list of list of strings (nodes of a batch of examples).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略以及相关参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用_encode_plus方法进行编码
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
    ) -> BatchEncoding:
        # 如果设置了返回偏移映射，则抛出未实现的错误
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # 调用实例方法 prepare_for_model()，准备输入以供模型使用
        return self.prepare_for_model(
            text=text,  # 主要文本输入
            text_pair=text_pair,  # 可选的第二文本输入（用于双输入模型）
            xpaths=xpaths,  # XPath 标签序列
            node_labels=node_labels,  # 节点标签序列
            add_special_tokens=add_special_tokens,  # 是否添加特殊标记（如 [CLS], [SEP]）
            padding=padding_strategy.value,  # 填充策略
            truncation=truncation_strategy.value,  # 截断策略
            max_length=max_length,  # 最大长度限制
            stride=stride,  # 滑动窗口步长
            pad_to_multiple_of=pad_to_multiple_of,  # 填充到某个倍数
            return_tensors=return_tensors,  # 返回的张量类型
            prepend_batch_axis=True,  # 是否在批处理维度前添加批处理轴
            return_attention_mask=return_attention_mask,  # 是否返回注意力掩码
            return_token_type_ids=return_token_type_ids,  # 是否返回token类型IDs
            return_overflowing_tokens=return_overflowing_tokens,  # 是否返回溢出的token
            return_special_tokens_mask=return_special_tokens_mask,  # 是否返回特殊token的掩码
            return_length=return_length,  # 是否返回序列长度
            verbose=verbose,  # 是否输出详细信息
        )

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, MARKUPLM_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 主要文本输入或预分词输入
        text_pair: Optional[PreTokenizedInput] = None,  # 可选的第二文本输入（用于双输入模型）
        xpaths: Optional[List[List[int]]] = None,  # XPath 标签序列列表
        node_labels: Optional[List[int]] = None,  # 节点标签列表
        add_special_tokens: bool = True,  # 是否添加特殊标记（如 [CLS], [SEP]）
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略
        truncation: Union[bool, str, TruncationStrategy] = None,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度限制
        stride: int = 0,  # 滑动窗口步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到某个倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型IDs
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的token
        return_special_tokens_mask: bool = False,  # 是否返回特殊token的掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回序列长度
        verbose: bool = True,  # 是否输出详细信息
        prepend_batch_axis: bool = False,  # 是否在批处理维度前添加批处理轴
        **kwargs,  # 其他参数
    def truncate_sequences(
        self,
        ids: List[int],  # 序列的ID列表
        xpath_tags_seq: List[List[int]],  # XPath 标签序列的列表
        xpath_subs_seq: List[List[int]],  # XPath 子序列的列表
        pair_ids: Optional[List[int]] = None,  # 可选的第二序列的ID列表
        pair_xpath_tags_seq: Optional[List[List[int]]] = None,  # 可选的第二XPath标签序列的列表
        pair_xpath_subs_seq: Optional[List[List[int]]] = None,  # 可选的第二XPath子序列的列表
        labels: Optional[List[int]] = None,  # 标签列表（如分类任务的标签）
        num_tokens_to_remove: int = 0,  # 需要移除的token数量
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",  # 截断策略
        stride: int = 0,  # 滑动窗口步长
    # 定义一个私有方法 `_pad`，用于对输入进行填充操作
    def _pad(
        # 输入参数 `encoded_inputs` 可以是字典（单个样本）或者批编码（多个样本）
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        # 最大长度参数，指定填充后的最大长度
        max_length: Optional[int] = None,
        # 填充策略，默认为不填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 如果指定，将填充长度调整为该数的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 是否返回注意力掩码，默认根据填充策略自动确定
        return_attention_mask: Optional[bool] = None,
```