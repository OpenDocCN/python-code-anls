# `.\models\layoutlmv3\tokenization_layoutlmv3.py`

```py
# coding=utf-8
# 设置文件编码为UTF-8，确保支持各种语言字符集
# Copyright The HuggingFace Inc. team. All rights reserved.
# 版权声明，保留所有权利
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据Apache 2.0许可证授权，允许使用此代码
# you may not use this file except in compliance with the License.
# 除非遵守许可证，否则禁止使用此文件
# You may obtain a copy of the License at
# 可在上述链接获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，否则不得在软件中使用
# distributed under the License is distributed on an "AS IS" BASIS,
# 软件按原样提供，不附带任何担保
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不提供任何明示或默示的担保或条件
# See the License for the specific language governing permissions and
# 详细了解许可证的具体条款和条件，请参阅许可证
# limitations under the License.
# 许可证下的限制
"""Tokenization class for LayoutLMv3. Same as LayoutLMv2, but RoBERTa-like BPE tokenization instead of WordPiece."""
# 为LayoutLMv3设计的分词类，与LayoutLMv2相同，但使用类似RoBERTa的BPE分词而不是WordPiece

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import regex as re

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
# 导入所需的模块和类

from ...utils import PaddingStrategy, TensorType, add_end_docstrings, logging
# 导入工具类和函数

logger = logging.get_logger(__name__)
# 获取用于当前文件名的日志记录器

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}
# 定义词汇文件的名称映射

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/layoutlmv3-base": "https://huggingface.co/microsoft/layoutlmv3-base/raw/main/vocab.json",
        "microsoft/layoutlmv3-large": "https://huggingface.co/microsoft/layoutlmv3-large/raw/main/vocab.json",
    },
    "merges_file": {
        "microsoft/layoutlmv3-base": "https://huggingface.co/microsoft/layoutlmv3-base/raw/main/merges.txt",
        "microsoft/layoutlmv3-large": "https://huggingface.co/microsoft/layoutlmv3-large/raw/main/merges.txt",
    },
}
# 预训练模型使用的词汇文件映射及其对应的URL

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/layoutlmv3-base": 512,
    "microsoft/layoutlmv3-large": 512,
}
# 预训练模型的位置嵌入尺寸映射

"""


"""


@lru_cache()
# 使用LRU缓存装饰器，缓存函数的调用结果，提高性能
# Copied from transformers.models.roberta.tokenization_roberta.bytes_to_unicode
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    # 返回utf-8字节的列表和到Unicode字符串的映射
    # 避免映射到BPE代码无法处理的空白字符和控制字符
    # 可逆BPE代码适用于Unicode字符串。这意味着如果要避免UNK（未知标记），词汇表中需要大量的Unicode字符
    # 在处理约100亿个标记的数据集compatibility with BPE tokenization
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    # Adding utf-8 bytes not present in bs, creating mapping for BPE tokenization
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    # Converting indices to corresponding unicode characters
    cs = [chr(n) for n in cs]
    # 使用内置的 zip 函数将两个列表 bs 和 cs 中的元素一一配对，生成一个元组的列表
    # 使用 dict() 函数将这个元组的列表转换为字典，并将其作为函数的返回值
    return dict(zip(bs, cs))
# Copied from transformers.models.roberta.tokenization_roberta.get_pairs
# 定义函数 get_pairs，用于获取单词中的符号对集合
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    单词被表示为符号的元组（符号是可变长度的字符串）。
    """
    # 初始化空集合用于存放符号对
    pairs = set()
    # 获取前一个字符作为初始符号
    prev_char = word[0]
    # 遍历单词中的每个字符（从第二个字符开始）
    for char in word[1:]:
        # 将前一个字符和当前字符组成的符号对添加到集合中
        pairs.add((prev_char, char))
        # 更新前一个字符为当前字符，为下一个符号对做准备
        prev_char = char
    # 返回所有符号对的集合
    return pairs


class LayoutLMv3Tokenizer(PreTrainedTokenizer):
    r"""
    Construct a LayoutLMv3 tokenizer. Based on [`RoBERTatokenizer`] (Byte Pair Encoding or BPE).
    [`LayoutLMv3Tokenizer`] can be used to turn words, word-level bounding boxes and optional word labels to
    token-level `input_ids`, `attention_mask`, `token_type_ids`, `bbox`, and optional `labels` (for token
    classification).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    [`LayoutLMv3Tokenizer`] runs end-to-end tokenization: punctuation splitting and wordpiece. It also turns the
    word-level bounding boxes into token-level bounding boxes.

    """

    # 定义类属性，存储词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义类属性，存储预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义类属性，存储模型最大输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义类属性，存储模型输入名称列表
    model_input_names = ["input_ids", "attention_mask", "bbox"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=True,
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[0, 0, 0, 0],
        pad_token_box=[0, 0, 0, 0],
        pad_token_label=-100,
        only_label_first_subword=True,
        **kwargs,
    ):
        # 调用父类的初始化方法，传入必要参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        # 初始化 LayoutLMv3Tokenizer 的特有属性
        # 存储词汇文件路径
        self.vocab_file = vocab_file
        # 存储合并文件路径
        self.merges_file = merges_file
        # 错误处理方式
        self.errors = errors
        # 起始符号
        self.cls_token = cls_token
        # 结束符号
        self.sep_token = sep_token
        # 未知符号
        self.unk_token = unk_token
        # 填充符号
        self.pad_token = pad_token
        # 掩码符号
        self.mask_token = mask_token
        # 起始符号对应的边界框
        self.cls_token_box = cls_token_box
        # 结束符号对应的边界框
        self.sep_token_box = sep_token_box
        # 填充符号对应的边界框
        self.pad_token_box = pad_token_box
        # 填充符号对应的标签
        self.pad_token_label = pad_token_label
        # 是否仅标签化第一个子词
        self.only_label_first_subword = only_label_first_subword
        # 其他参数
        self.special_tokens_map_extended = {}
        self.unique_no_split_tokens = set()
        self._extra_ids = 0

        # 调用初始化方法，加载词汇表
        self._additional_special_tokens = []
        self.add_special_tokens(
            {"bos_token": bos_token, "eos_token": eos_token, "unk_token": unk_token, "pad_token": pad_token}
        )
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 从指定的 vocab_file 中加载编码器，以 JSON 格式读取文件内容并存储在 self.encoder 中
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        
        # 创建解码器，使用 self.encoder 字典的键值对调，存储在 self.decoder 中
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # 设置处理解码时的错误处理策略
        self.errors = errors  # how to handle errors in decoding
        
        # 使用 bytes_to_unicode 函数创建编码器的字节到 Unicode 字符的映射，存储在 self.byte_encoder 中
        self.byte_encoder = bytes_to_unicode()
        
        # 创建解码器的反向映射，使用 self.byte_encoder 字典的键值对调，存储在 self.byte_decoder 中
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # 从指定的 merges_file 中读取 BPE 合并操作，解析为元组列表并使用其顺序创建 self.bpe_ranks 字典
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        
        # 初始化缓存字典
        self.cache = {}
        
        # 设置是否在标记前加空格的标志
        self.add_prefix_space = add_prefix_space

        # 使用正则表达式创建 self.pat 以处理特定文本模式，包括缩写和单词
        # 添加 re.IGNORECASE 以便可以对大小写不敏感的情况进行 BPE 合并
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 设置额外的属性
        self.cls_token_box = cls_token_box
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword

        # 调用父类的初始化方法，传递所需参数和额外的关键字参数 **kwargs
        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            cls_token_box=cls_token_box,
            sep_token_box=sep_token_box,
            pad_token_box=pad_token_box,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            **kwargs,
        )

    @property
    # 从 transformers.models.roberta.tokenization_roberta.RobertaTokenizer.vocab_size 处复制而来
    def vocab_size(self):
        # 返回 self.encoder 字典的长度，即词汇表的大小
        return len(self.encoder)

    # 从 transformers.models.roberta.tokenization_roberta.RobertaTokenizer.get_vocab 处复制而来
    # 复制自 transformers.models.roberta.tokenization_roberta.RobertaTokenizer.get_vocab 方法
    def get_vocab(self):
        # 从 self.encoder 字典创建 vocab 字典的副本
        vocab = dict(self.encoder).copy()
        # 将 self.added_tokens_encoder 字典合并到 vocab 字典中
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 复制自 transformers.models.roberta.tokenization_roberta.RobertaTokenizer.bpe 方法
    def bpe(self, token):
        # 如果 token 已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        # 获得所有可能的字符对
        pairs = get_pairs(word)

        # 如果没有字符对，则直接返回 token
        if not pairs:
            return token

        # 反复处理字符对，直到无法再合并
        while True:
            # 找到频率最低的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 遍历当前词中的字符
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    # 如果找不到 first，则将剩余部分添加到 new_word 中
                    new_word.extend(word[i:])
                    break
                else:
                    # 将 first 之前的部分添加到 new_word 中
                    new_word.extend(word[i:j])
                    i = j

                # 检查当前位置是否匹配 bigram，如果匹配则合并为一个新的字符
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    # 否则将当前字符添加到 new_word 中
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            # 如果合并后只剩一个字符，则结束循环
            if len(word) == 1:
                break
            else:
                # 否则继续处理新的字符对
                pairs = get_pairs(word)
        # 将处理后的字符列表连接成一个字符串
        word = " ".join(word)
        # 将结果存入缓存并返回
        self.cache[token] = word
        return word

    # 复制自 transformers.models.roberta.tokenization_roberta.RobertaTokenizer._tokenize 方法
    def _tokenize(self, text):
        """对字符串进行分词处理。"""
        bpe_tokens = []
        # 使用正则表达式找到所有匹配的 token
        for token in re.findall(self.pat, text):
            # 将 token 中的每个字节编码成 Unicode 字符串，避免 BPE 中的控制标记（例如空格）
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # 将所有字节映射为 unicode 字符串，避免 BPE 的控制标记（在我们的情况下是空格）
            # 使用 BPE 算法处理 token，并将结果拆分为多个子 token
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # 复制自 transformers.models.roberta.tokenization_roberta.RobertaTokenizer._convert_token_to_id 方法
    def _convert_token_to_id(self, token):
        """使用词汇表将 token（字符串）转换为对应的 id。"""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 复制自 transformers.models.roberta.tokenization_roberta.RobertaTokenizer._convert_id_to_token 方法
    def _convert_id_to_token(self, index):
        """使用词汇表将索引（整数）转换为对应的 token（字符串）。"""
        return self.decoder.get(index)

    # 复制自 transformers.models.roberta.tokenization_roberta.RobertaTokenizer.convert_tokens_to_string 方法
    def convert_tokens_to_string(self, tokens):
        """将一系列 token（字符串）转换为单个字符串。"""
        text = "".join(tokens)
        # 使用字节数组将每个字符解码为 UTF-8 编码的字符串，避免错误，使用指定的错误处理方法
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    # Copied from transformers.models.roberta.tokenization_roberta.RobertaTokenizer.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否为一个目录，如果不是则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 确定词汇文件的路径，结合指定的前缀和文件名
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 确定合并文件的路径，结合指定的前缀和文件名
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )
    
        # 打开词汇文件并将编码器内容以 JSON 格式写入文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    
        index = 0
        # 打开合并文件并写入版本信息
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            # 遍历并写入 BPE 标记及其索引，确保索引连续性，同时记录警告信息
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1
    
        # 返回词汇文件和合并文件的路径
        return vocab_file, merge_file
    
    # Copied from transformers.models.roberta.tokenization_roberta.RobertaTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过连接和添加特殊标记，为序列分类任务构建模型输入。RoBERTa 的序列格式如下：
    
        - 单个序列: `<s> X </s>`
        - 序列对: `<s> A </s></s> B </s>`
    
        Args:
            token_ids_0 (`List[int]`):
                将添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 ID 列表，用于序列对任务。
    
        Returns:
            `List[int]`: 包含适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
        """
        if token_ids_1 is None:
            # 对于单个序列，添加起始和结束标记
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        
        # 对于序列对，添加起始和结束标记，并根据 RoBERTa 的格式添加额外的结束标记
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep
    
    # Copied from transformers.models.roberta.tokenization_roberta.RobertaTokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        # If the token list already has special tokens, delegate to the base class method to get the mask
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If there is no token_ids_1 (no sequence pair), return a mask with special tokens around token_ids_0
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        
        # For sequence pairs, return a mask with special tokens around both token_ids_0 and token_ids_1
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    # Copied from transformers.models.roberta.tokenization_roberta.RobertaTokenizer.create_token_type_ids_from_sequences
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
        # Define special tokens for start of sequence and separator
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If there is no token_ids_1, return a list of zeros corresponding to the length of cls + token_ids_0 + sep
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # For sequence pairs, return a list of zeros corresponding to the length of cls + token_ids_0 + sep + sep + token_ids_1 + sep
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # If the text starts with a token that should not be split, no space is added before the text in any case.
        # It's necessary to match the fast tokenization
        if (
            (is_split_into_words or add_prefix_space)
            and (len(text) > 0 and not text[0].isspace())
            and sum([text.startswith(no_split_token) for no_split_token in self.added_tokens_encoder]) == 0
        ):
            text = " " + text
        return (text, kwargs)

    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # Copied from transformers.models.layoutlmv2.tokenization_layoutlmv2.LayoutLMv2Tokenizer.__call__
    # 使用 __call__ 方法作为对象的调用接口，接受多种形式的文本输入和相关参数
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
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
        # 使用 add_end_docstrings 函数添加文档字符串，包括 LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING 和 LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING 的内容
        @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
        # 方法的主体是从 layoutlmv2.tokenization_layoutlmv2.LayoutLMv2Tokenizer.batch_encode_plus 复制过来的
        def batch_encode_plus(
            self,
            batch_text_or_text_pairs: Union[
                List[TextInput],
                List[TextInputPair],
                List[PreTokenizedInput],
            ],
            is_pair: bool = None,
            boxes: Optional[List[List[List[int]]]] = None,
            word_labels: Optional[Union[List[int], List[List[int]]]] = None,
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
        # 为了向后兼容而设置的参数，用于确定填充和截断策略，以及最大长度和其他关键字参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法 _batch_encode_plus 进行批量编码，传递各种参数
        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            is_pair=is_pair,
            boxes=boxes,
            word_labels=word_labels,
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
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        # 调用私有方法 _batch_prepare_for_model，准备批量输入数据用于模型处理
        batch_outputs = self._batch_prepare_for_model(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            is_pair=is_pair,
            boxes=boxes,
            word_labels=word_labels,
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

        # 返回 BatchEncoding 类的实例，将批处理输出封装成 BatchEncoding 对象
        return BatchEncoding(batch_outputs)

    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2.LayoutLMv2Tokenizer._batch_prepare_for_model 复制而来
    def _batch_prepare_for_model(
        self,
        batch_text_or_text_pairs,
        is_pair: bool = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[List[int]]] = None,
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

        batch_outputs = {}  # 初始化空字典，用于存储批处理输出结果

        # 遍历批处理中的每个示例，同时迭代处理文本对或文本与框的组合
        for idx, example in enumerate(zip(batch_text_or_text_pairs, boxes)):
            batch_text_or_text_pair, boxes_example = example
            # 根据是否为文本对，选择合适的输入文本或文本对，以及相关的框信息
            outputs = self.prepare_for_model(
                batch_text_or_text_pair[0] if is_pair else batch_text_or_text_pair,
                batch_text_or_text_pair[1] if is_pair else None,
                boxes_example,
                word_labels=word_labels[idx] if word_labels is not None else None,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # 不进行填充，批处理中后续会进行填充
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # 不进行填充，批处理中后续会进行填充
                return_attention_mask=False,  # 不返回注意力掩码，批处理中后续会返回
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # 最终将整个批次转换为张量
                prepend_batch_axis=False,
                verbose=verbose,
            )

            # 将每个输出添加到对应的键中，确保每个键对应一个列表，存储所有示例的输出
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # 对批处理的输出进行填充，使用指定的填充策略和最大长度
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # 将填充后的输出封装成 BatchEncoding 对象，使用指定的张量类型
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        # 返回封装后的批处理输出对象
        return batch_outputs

    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING)
    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2.LayoutLMv2Tokenizer.encode 复制而来
    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[int]] = None,
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
        # 调用 encode_plus 方法对文本进行编码，并返回编码后的输入特征
        encoded_inputs = self.encode_plus(
            text=text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
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

        # 返回编码后的输入特征中的 input_ids（输入 token 的 IDs）
        return encoded_inputs["input_ids"]

    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2.LayoutLMv2Tokenizer.encode_plus 复制而来
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[int]] = None,
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

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略，以及其他相关的参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用_encode_plus方法，进行编码和处理文本
        return self._encode_plus(
            text=text,
            boxes=boxes,
            text_pair=text_pair,
            word_labels=word_labels,
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

    # Copied from transformers.models.layoutlmv2.tokenization_layoutlmv2.LayoutLMv2Tokenizer._encode_plus
    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[int]] = None,
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
        # 如果设置了返回偏移映射，抛出未实现错误，因为 Python tokenizer 不支持此功能
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # 调用 prepare_for_model 方法，准备输入以供模型使用
        return self.prepare_for_model(
            text=text,  # 主要文本输入
            text_pair=text_pair,  # 可选的第二个文本输入（用于双输入模型）
            boxes=boxes,  # 文本框坐标信息（用于图像文本输入）
            word_labels=word_labels,  # 单词级别标签（用于标注任务）
            add_special_tokens=add_special_tokens,  # 是否添加特殊标记（如 [CLS], [SEP]）
            padding=padding_strategy.value,  # 填充策略（布尔值、字符串或填充策略对象）
            truncation=truncation_strategy.value,  # 截断策略（布尔值、字符串或截断策略对象）
            max_length=max_length,  # 最大长度限制
            stride=stride,  # 滑动窗口步长
            pad_to_multiple_of=pad_to_multiple_of,  # 填充到某个倍数
            return_tensors=return_tensors,  # 返回的张量类型
            prepend_batch_axis=True,  # 是否在结果中添加批次维度
            return_attention_mask=return_attention_mask,  # 是否返回注意力掩码
            return_token_type_ids=return_token_type_ids,  # 是否返回 token 类型 IDs
            return_overflowing_tokens=return_overflowing_tokens,  # 是否返回溢出的 tokens
            return_special_tokens_mask=return_special_tokens_mask,  # 是否返回特殊 tokens 掩码
            return_length=return_length,  # 是否返回输入长度信息
            verbose=verbose,  # 是否显示详细信息
        )

    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 主要文本输入或预标记输入
        text_pair: Optional[PreTokenizedInput] = None,  # 可选的第二个文本输入
        boxes: Optional[List[List[int]]] = None,  # 文本框坐标信息
        word_labels: Optional[List[int]] = None,  # 单词级别标签
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略
        truncation: Union[bool, str, TruncationStrategy] = None,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度限制
        stride: int = 0,  # 滑动窗口步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到某个倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回 token 类型 IDs
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的 tokens
        return_special_tokens_mask: bool = False,  # 是否返回特殊 tokens 掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回输入长度信息
        verbose: bool = True,  # 是否显示详细信息
        prepend_batch_axis: bool = False,  # 是否在结果中添加批次维度
        **kwargs,  # 其他关键字参数
    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2.LayoutLMv2Tokenizer.truncate_sequences 复制而来
    def truncate_sequences(
        self,
        ids: List[int],  # 输入的 token IDs 列表
        token_boxes: List[List[int]],  # 对应的 token 边框坐标列表
        pair_ids: Optional[List[int]] = None,  # 可选的第二个文本输入的 token IDs 列表
        pair_token_boxes: Optional[List[List[int]]] = None,  # 可选的第二个文本输入的 token 边框坐标列表
        labels: Optional[List[int]] = None,  # 标签列表（用于标注任务）
        num_tokens_to_remove: int = 0,  # 要删除的 tokens 数量
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",  # 截断策略
        stride: int = 0,  # 滑动窗口步长
    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2.LayoutLMv2Tokenizer._pad 复制而来
    # 定义一个私有方法 `_pad`，用于填充输入数据，确保它们达到指定的最大长度
    # 方法参数说明：
    # - encoded_inputs: 可以是字典形式的编码输入或者批量编码对象，用于输入数据的编码
    # - max_length: 可选参数，指定填充后的最大长度
    # - padding_strategy: 填充策略，默认为不填充（PaddingStrategy.DO_NOT_PAD）
    # - pad_to_multiple_of: 可选参数，指定填充后长度的倍数
    # - return_attention_mask: 可选参数，是否返回注意力掩码，默认为 None
```