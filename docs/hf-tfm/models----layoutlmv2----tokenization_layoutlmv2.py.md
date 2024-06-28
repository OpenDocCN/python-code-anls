# `.\models\layoutlmv2\tokenization_layoutlmv2.py`

```
# coding=utf-8
# Copyright Microsoft Research and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization class for LayoutLMv2."""

import collections  # 导入 collections 模块
import os  # 导入 os 模块
import sys  # 导入 sys 模块
import unicodedata  # 导入 unicodedata 模块
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示相关的类和函数

from ...tokenization_utils import AddedToken, PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace  # 导入 tokenization_utils 中的类和函数
from ...tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)  # 导入 tokenization_utils_base 中的类和函数
from ...utils import PaddingStrategy, TensorType, add_end_docstrings, logging  # 导入 utils 中的类和函数

logger = logging.get_logger(__name__)  # 获取日志记录器

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}  # 定义词汇文件名字典

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/layoutlmv2-base-uncased": (
            "https://huggingface.co/microsoft/layoutlmv2-base-uncased/resolve/main/vocab.txt"
        ),
        "microsoft/layoutlmv2-large-uncased": (
            "https://huggingface.co/microsoft/layoutlmv2-large-uncased/resolve/main/vocab.txt"
        ),
    }
}  # 预训练词汇文件映射

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/layoutlmv2-base-uncased": 512,
    "microsoft/layoutlmv2-large-uncased": 512,
}  # 预训练位置嵌入尺寸

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/layoutlmv2-base-uncased": {"do_lower_case": True},
    "microsoft/layoutlmv2-large-uncased": {"do_lower_case": True},
}  # 预训练初始化配置

"""

"""


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()  # 创建一个有序字典用于存储词汇
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()  # 读取词汇文件中的所有行
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")  # 去除每行末尾的换行符
        vocab[token] = index  # 将词汇和索引存入字典
    return vocab  # 返回构建的词汇字典


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()  # 去除文本两端空白字符
    if not text:
        return []  # 如果文本为空，则返回空列表
    tokens = text.split()  # 使用空格分割文本，得到词汇列表
    return tokens  # 返回分割后的词汇列表


table = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P"))  # 创建一个字典，包含所有标点符号的 Unicode 编码

def subfinder(mylist, pattern):
    matches = []  # 初始化匹配列表
    indices = []  # 初始化索引列表
    for idx, i in enumerate(range(len(mylist))):
        if mylist[i] == pattern[0] and mylist[i : i + len(pattern)] == pattern:
            matches.append(pattern)  # 如果找到匹配的模式，添加到匹配列表
            indices.append(idx)  # 记录模式首次出现的索引
    if matches:
        return matches[0], indices[0]  # 如果有匹配项，返回第一个匹配的模式和其索引
    else:
        return None, 0  # 如果没有匹配项，返回 None 和 0


class LayoutLMv2Tokenizer(PreTrainedTokenizer):
    r"""
    """
    构建一个 LayoutLMv2 的分词器。基于 WordPiece。[`LayoutLMv2Tokenizer`] 可以用于将单词、单词级别边界框和可选的单词标签转换为
    标记级别的 `input_ids`、`attention_mask`、`token_type_ids`、`bbox`，以及可选的 `labels`（用于标记分类）。

    该分词器继承自 [`PreTrainedTokenizer`]，其中包含大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。

    [`LayoutLMv2Tokenizer`] 运行端到端的分词：标点符号分割和 WordPiece。它还将单词级别的边界框转换为标记级别的边界框。
    """

    # 定义预训练模型所需的词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义预训练模型所需的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义预训练模型输入的最大长度列表
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义预训练模型的初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_label=-100,
        only_label_first_subword=True,
        tokenize_chinese_chars=True,
        strip_accents=None,
        model_max_length: int = 512,
        additional_special_tokens: Optional[List[str]] = None,
        **kwargs,
    ):
    ):
        # 如果 sep_token 是字符串，则创建一个特殊的 AddedToken 对象
        sep_token = AddedToken(sep_token, special=True) if isinstance(sep_token, str) else sep_token
        # 如果 unk_token 是字符串，则创建一个特殊的 AddedToken 对象
        unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
        # 如果 pad_token 是字符串，则创建一个特殊的 AddedToken 对象
        pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token
        # 如果 cls_token 是字符串，则创建一个特殊的 AddedToken 对象
        cls_token = AddedToken(cls_token, special=True) if isinstance(cls_token, str) else cls_token
        # 如果 mask_token 是字符串，则创建一个特殊的 AddedToken 对象
        mask_token = AddedToken(mask_token, special=True) if isinstance(mask_token, str) else mask_token

        # 如果指定的词汇文件不存在，抛出 ValueError 异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        
        # 加载词汇表文件并将其存储在 self.vocab 中
        self.vocab = load_vocab(vocab_file)
        # 创建一个从 id 到 token 的有序字典 self.ids_to_tokens
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 根据 do_basic_tokenize 的设置决定是否使用基础的分词器
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            # 如果需要基础的分词，创建 BasicTokenizer 对象
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        
        # 使用给定的词汇表和 unk_token 创建 WordpieceTokenizer 对象
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 设置额外的属性
        self.cls_token_box = cls_token_box
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword
        
        # 调用父类的构造函数，初始化参数
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            cls_token_box=cls_token_box,
            sep_token_box=sep_token_box,
            pad_token_box=pad_token_box,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            model_max_length=model_max_length,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    def do_lower_case(self):
        # 返回基础分词器的小写设置
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        # 返回词汇表大小
        return len(self.vocab)

    def get_vocab(self):
        # 返回包含词汇表和添加的特殊 token 编码的字典
        return dict(self.vocab, **self.added_tokens_encoder)
    # 将文本进行分词处理，返回分词后的结果列表
    def _tokenize(self, text):
        split_tokens = []
        # 如果需要进行基本的分词处理
        if self.do_basic_tokenize:
            # 使用基本分词器对文本进行分词，忽略不需要分词的特殊标记
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # 如果分词后的 token 在不需要分割的集合中
                if token in self.basic_tokenizer.never_split:
                    # 直接加入到分词结果中
                    split_tokens.append(token)
                else:
                    # 使用 WordPiece 分词器对 token 进行进一步分词处理
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 否则，直接使用 WordPiece 分词器对文本进行分词处理
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        # 返回分词后的结果列表
        return split_tokens

    # 根据词汇表将 token 转换为对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 根据词汇表将 id 转换为对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    # 将 token 列表转换为单个字符串，同时去除特殊标记 "##"
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # 构建包含特殊标记的输入序列，用于序列分类任务
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 如果只有一个输入序列
        if token_ids_1 is None:
            # 返回带有 [CLS] 和 [SEP] 特殊标记的输入序列
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 分别定义 [CLS] 和 [SEP] 的特殊标记
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 返回带有 [CLS], [SEP] 和两个序列之间的 [SEP] 特殊标记的输入序列
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # 获取包含特殊标记的 token id 序列的掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
        """Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using `build_inputs_with_special_tokens` method.
        
        Args:
            token_ids_0 (`List[int]`):
                List of token ids (must be pure token ids without special tokens).
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of token ids for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether the token list is already formated with special tokens or not.

        Returns:
            `List[int]`: A list of integers in the range [0, 1], with 1 specifying special tokens and 0 specifying
            regular tokens.
        """
        # 如果输入的 token_ids 已经包含了特殊标记
        if already_has_special_tokens:
            # 返回与 token_ids 0 和 token_ids 1 长度相同的全零列表
            return [0] * len(token_ids_0)
        # 定义一个用于存储特殊标记掩码的列表
        special_tokens_mask = [1]  # [CLS] token
        # 如果有第二个序列 token_ids_1
        if token_ids_1 is not None:
            # 添加一个 [SEP] token 的掩码
            special_tokens_mask += [1] * len(token_ids_1)  # [SEP] tokens
        # 返回特殊标记的掩码与 token_ids_0 长度相同的列表
        return special_tokens_mask + [0] * (len(token_ids_0) - len(special_tokens_mask))
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

        if already_has_special_tokens:
            # 如果已经存在特殊标记，则调用父类方法获取特殊标记的掩码
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            # 如果存在第二个序列，返回包含特殊标记的掩码：[CLS] + token_ids_0 + [SEP] + token_ids_1 + [SEP]
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # 如果只有一个序列，返回包含特殊标记的掩码：[CLS] + token_ids_0 + [SEP]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format: :: 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 | first sequence | second
        sequence | If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # 获取分隔符和类别标记的 ID
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            # 如果只有一个序列，返回一个长度为 cls + token_ids_0 + sep 长度的全零列表
            return len(cls + token_ids_0 + sep) * [0]
        # 如果有两个序列，返回两个序列加上分隔符的长度分别对应的掩码列表：[CLS] + token_ids_0 + [SEP] + token_ids_1 + [SEP]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    # 将词汇表保存到指定目录下的文件中，返回保存的文件路径元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引
        index = 0
        # 检查保存目录是否存在，如果存在则构建词汇表文件路径
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 否则直接使用指定的保存路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开词汇表文件，准备写入内容
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表中的每个词汇及其索引，并按索引顺序写入文件
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 如果当前词汇的索引不是期望的连续索引，记录警告信息
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 写入词汇到文件，并增加索引计数
                writer.write(token + "\n")
                index += 1
        # 返回保存的词汇表文件路径元组
        return (vocab_file,)

    # 调用函数的装饰器，添加文档字符串到__call__方法
    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
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
    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义一个方法用于批量编码文本或文本对，并返回批处理编码结果
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,  # 是否为文本对
        boxes: Optional[List[List[List[int]]]] = None,  # 文本框的坐标信息（可选）
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,  # 单词标签（可选）
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略
        truncation: Union[bool, str, TruncationStrategy] = None,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度限制（可选）
        stride: int = 0,  # 步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定倍数的长度（可选）
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型（可选）
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型ID（可选）
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码（可选）
        return_overflowing_tokens: bool = False,  # 是否返回溢出的token
        return_special_tokens_mask: bool = False,  # 是否返回特殊token掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回长度
        verbose: bool = True,  # 是否详细输出信息
        **kwargs,  # 其他关键字参数
    ) -> BatchEncoding:
        # 获取填充和截断策略，并处理旧版本参数兼容性
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法执行批量编码
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
    # 定义一个方法 `_batch_encode_plus`，用于批量编码文本或文本对，并生成批编码结果的对象
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,  # 是否为文本对
        boxes: Optional[List[List[List[int]]]] = None,  # 盒子坐标，用于文本识别任务
        word_labels: Optional[List[List[int]]] = None,  # 单词标签列表
        add_special_tokens: bool = True,  # 是否添加特殊标记（例如 [CLS], [SEP]）
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 最大长度限制
        stride: int = 0,  # 步进值，默认为0
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回 token 类型 IDs
        return_attention_mask: Optional[bool] = None,  # 是否返回 attention mask
        return_overflowing_tokens: bool = False,  # 是否返回溢出的 tokens
        return_special_tokens_mask: bool = False,  # 是否返回特殊 tokens 的 mask
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回长度信息
        verbose: bool = True,  # 是否打印详细信息
        **kwargs,  # 其他未命名参数
    ) -> BatchEncoding:  # 方法返回类型为 BatchEncoding 对象
        # 如果请求返回偏移映射，则抛出 NotImplementedError
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        # 调用内部方法 `_batch_prepare_for_model` 准备批量数据以供模型处理
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

        # 将批处理输出转换为 BatchEncoding 对象并返回
        return BatchEncoding(batch_outputs)

    # 将函数 `_batch_encode_plus` 与文档字符串拼接并添加到类中作为方法装饰器
    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 批量准备输入数据以供模型处理，处理文本或文本对的批次
    def _batch_prepare_for_model(
        self,
        batch_text_or_text_pairs,  # 输入的文本或文本对的批次
        is_pair: bool = None,  # 标志是否为文本对
        boxes: Optional[List[List[int]]] = None,  # 文本框的位置信息（可选）
        word_labels: Optional[List[List[int]]] = None,  # 单词级别的标签（可选）
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度限制（可选）
        stride: int = 0,  # 步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的倍数（可选）
        return_tensors: Optional[str] = None,  # 返回的张量类型（可选）
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型id（可选）
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码（可选）
        return_overflowing_tokens: bool = False,  # 是否返回溢出的token
        return_special_tokens_mask: bool = False,  # 是否返回特殊token的掩码
        return_length: bool = False,  # 是否返回批次长度
        verbose: bool = True,  # 是否打印详细信息
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

        # Iterate over each example in the batch, consisting of text or text pairs and corresponding boxes
        for idx, example in enumerate(zip(batch_text_or_text_pairs, boxes)):
            batch_text_or_text_pair, boxes_example = example
            
            # Determine if the current example is a single text or a pair of texts
            if is_pair:
                input_ids_or_pair = batch_text_or_text_pair[0]  # First sequence of input ids
            else:
                input_ids_or_pair = batch_text_or_text_pair  # Single sequence of input ids
            
            # Prepare inputs for the model using the specified parameters
            outputs = self.prepare_for_model(
                input_ids_or_pair,
                batch_text_or_text_pair[1] if is_pair else None,  # Second sequence of input ids if it exists
                boxes_example,
                word_labels=word_labels[idx] if word_labels is not None else None,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # Do not pad here; it's done in batch
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # Pad in batch afterward
                return_attention_mask=False,  # Do not return attention masks here; it's done in batch
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # Convert to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            # Aggregate outputs into batch_outputs dictionary
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # Perform padding across the batch
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # Convert batch_outputs to BatchEncoding format
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        # Return the final prepared batch_outputs
        return batch_outputs

    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING)
    # 定义一个方法 `encode`，用于将输入文本和相关信息编码成模型可以处理的输入格式，并返回编码后的输入 ID 列表
    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 主要输入文本，可以是普通文本或预分词后的输入
        text_pair: Optional[PreTokenizedInput] = None,  # 可选的第二个输入文本，用于处理句对任务
        boxes: Optional[List[List[int]]] = None,  # 文本框的坐标信息列表，用于处理文本与空间信息结合的任务
        word_labels: Optional[List[int]] = None,  # 单词级别的标签列表，用于处理序列标注任务
        add_special_tokens: bool = True,  # 是否添加特殊令牌（如[CLS], [SEP]）
        padding: Union[bool, str, PaddingStrategy] = False,  # 是否进行填充处理
        truncation: Union[bool, str, TruncationStrategy] = None,  # 是否进行截断处理
        max_length: Optional[int] = None,  # 最大序列长度限制
        stride: int = 0,  # 滑动窗口的步长
        pad_to_multiple_of: Optional[int] = None,  # 填充长度的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型（如`pt`表示PyTorch张量）
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型 IDs
        return_attention_mask: Optional[bool] = None,  # 是否返回attention mask
        return_overflowing_tokens: bool = False,  # 是否返回溢出的 token
        return_special_tokens_mask: bool = False,  # 是否返回特殊令牌的 mask
        return_offsets_mapping: bool = False,  # 是否返回字符偏移映射
        return_length: bool = False,  # 是否返回编码后的长度
        verbose: bool = True,  # 是否启用详细输出模式
        **kwargs,  # 其他未指定的参数
    ) -> List[int]:  # 返回一个整数列表，表示编码后的输入 ID
        # 使用 `encode_plus` 方法对输入进行编码，并获取编码后的结果字典
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

        # 返回编码后结果中的 `input_ids` 键对应的值，即编码后的输入 ID 列表
        return encoded_inputs["input_ids"]

    # 使用 `add_end_docstrings` 装饰器添加文档字符串，详细说明 `encode_plus` 方法的参数和功能
    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 主要输入文本，可以是普通文本或预分词后的输入
        text_pair: Optional[PreTokenizedInput] = None,  # 可选的第二个输入文本，用于处理句对任务
        boxes: Optional[List[List[int]]] = None,  # 文本框的坐标信息列表，用于处理文本与空间信息结合的任务
        word_labels: Optional[List[int]] = None,  # 单词级别的标签列表，用于处理序列标注任务
        add_special_tokens: bool = True,  # 是否添加特殊令牌（如[CLS], [SEP]）
        padding: Union[bool, str, PaddingStrategy] = False,  # 是否进行填充处理
        truncation: Union[bool, str, TruncationStrategy] = None,  # 是否进行截断处理
        max_length: Optional[int] = None,  # 最大序列长度限制
        stride: int = 0,  # 滑动窗口的步长
        pad_to_multiple_of: Optional[int] = None,  # 填充长度的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型（如`pt`表示PyTorch张量）
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型 IDs
        return_attention_mask: Optional[bool] = None,  # 是否返回attention mask
        return_overflowing_tokens: bool = False,  # 是否返回溢出的 token
        return_special_tokens_mask: bool = False,  # 是否返回特殊令牌的 mask
        return_offsets_mapping: bool = False,  # 是否返回字符偏移映射
        return_length: bool = False,  # 是否返回编码后的长度
        verbose: bool = True,  # 是否启用详细输出模式
        **kwargs,  # 其他未指定的参数
    ):
        pass  # 方法体略，实际实现中将会进行文本编码并返回编码后的结果字典
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
        # 获取填充和截断策略以及其他相关参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用 _encode_plus 方法，对文本进行编码和处理
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
    ) -> BatchEncoding:
        if return_offsets_mapping:
            # 如果请求返回偏移映射，则抛出未实现错误
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # 调用内部方法，准备输入以供模型处理
        return self.prepare_for_model(
            text=text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
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

    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(
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
        prepend_batch_axis: bool = False,
        **kwargs,
    ):
        # 准备输入以供模型处理，根据参数配置进行处理
        # 详细文档参考 LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING 和 LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
        pass

    def truncate_sequences(
        self,
        ids: List[int],
        token_boxes: List[List[int]],
        pair_ids: Optional[List[int]] = None,
        pair_token_boxes: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    # 定义一个私有方法 `_pad`，用于填充输入序列以达到指定的最大长度
    # encoded_inputs: 可以是单个编码输入的字典或批编码对象
    # max_length: 可选参数，指定填充后的最大长度
    # padding_strategy: 填充策略，默认为不填充
    # pad_to_multiple_of: 可选参数，填充后的长度将是该参数的倍数
    # return_attention_mask: 可选参数，控制是否返回注意力掩码
# 从transformers.models.bert.tokenization_bert.BasicTokenizer复制的代码
class BasicTokenizer(object):
    """
    构建一个BasicTokenizer对象，用于执行基本的分词（如标点符号分割、转换为小写等）。

    Args:
        do_lower_case (`bool`, *可选*, 默认为 `True`):
            在分词时是否将输入转换为小写。
        never_split (`Iterable`, *可选*):
            在分词时永远不会被分割的token集合。仅在`do_basic_tokenize=True`时生效。
        tokenize_chinese_chars (`bool`, *可选*, 默认为 `True`):
            是否分词中文字符。

            对于日语，这可能需要禁用（参见这个
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *可选*):
            是否去除所有的重音符号。如果没有指定此选项，则会由`lowercase`的值来确定（与原始BERT一样）。
        do_split_on_punc (`bool`, *可选*, 默认为 `True`):
            在某些情况下，我们希望跳过基本的标点符号分割，以便后续的分词可以捕捉到单词的完整上下文，例如缩写词。

    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case  # 是否进行小写转换
        self.never_split = set(never_split)  # 永远不分割的token集合，转换成集合类型
        self.tokenize_chinese_chars = tokenize_chinese_chars  # 是否分割中文字符
        self.strip_accents = strip_accents  # 是否去除重音符号
        self.do_split_on_punc = do_split_on_punc  # 是否基于标点符号分割
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        # 如果提供了never_split参数，则将其与self.never_split合并成一个新的集合，用于记录不需要分割的token集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，例如去除多余的空格等
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        # 如果设置了tokenize_chinese_chars标志位，则调用_tokenize_chinese_chars方法处理中文字符
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 使用NFC规范化Unicode文本，确保不同的Unicode编码的同一字符被视为相同
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空格分割文本，生成原始token列表
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        # 遍历每个token，根据条件处理token并分割
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    # 如果开启了小写化，则将token转换为小写
                    token = token.lower()
                    if self.strip_accents is not False:
                        # 如果strip_accents不为False，则移除token中的重音符号
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    # 否则只移除token中的重音符号
                    token = self._run_strip_accents(token)
            # 将处理后的token列表添加到split_tokens中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 使用空格分割处理后的token列表，生成最终的output_tokens
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 使用NFD规范化Unicode文本，将重音符号与字符分开表示
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历每个字符，如果字符的Unicode category是Mn（Nonspacing_Mark），则跳过该字符，否则添加到output中
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        # 将字符列表连接成字符串，返回处理后的文本
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要在标点符号处分割或者指定的文本在never_split中，直接返回原始文本列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        # 遍历字符列表
        while i < len(chars):
            char = chars[i]
            # 如果当前字符是标点符号，将其作为新的列表项添加到输出列表中，并标记下一个字符为新单词的起始
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果当前字符不是标点符号
                if start_new_word:
                    output.append([])  # 添加一个新的空列表项
                start_new_word = False  # 取消新单词的起始标记
                output[-1].append(char)  # 将当前字符添加到当前单词的最后一个列表项中
            i += 1

        # 将列表中的列表项合并为字符串并返回
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        # 遍历文本中的每个字符
        for char in text:
            cp = ord(char)
            # 如果字符是中文字符，添加空格字符作为分隔符
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)  # 否则直接添加当前字符
        # 将字符列表转换为字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查传入的码点是否属于CJK字符的Unicode块范围
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True  # 是CJK字符返回True
        return False  # 否则返回False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        # 遍历文本中的每个字符
        for char in text:
            cp = ord(char)
            # 如果字符是无效字符或控制字符，直接跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果字符是空白字符，替换为单个空格字符，否则直接添加当前字符
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将字符列表转换为字符串并返回
        return "".join(output)
# Copied from transformers.models.bert.tokenization_bert.WordpieceTokenizer
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化 WordpieceTokenizer 类的实例，设置词汇表、未知 token 和单词最大字符数
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` will return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """
        # 初始化输出 token 列表
        output_tokens = []
        # 将输入文本按空白字符分割成 token，并逐个处理
        for token in whitespace_tokenize(text):
            # 将当前 token 转换为字符列表
            chars = list(token)
            # 若当前 token 的字符数超过设定的最大字符数，则添加未知 token 并跳过
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            # 初始化标志变量和起始位置
            is_bad = False
            start = 0
            sub_tokens = []
            # 迭代处理字符列表直到处理完所有字符
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                # 从当前起始位置到结束位置，逐步减少子字符串长度，直到找到在词汇表中存在的最长子字符串
                while start < end:
                    substr = "".join(chars[start:end])
                    # 如果起始位置不是第一个字符，则在找到的子字符串前加上 "##"
                    if start > 0:
                        substr = "##" + substr
                    # 如果找到了在词汇表中的子字符串，则保存当前子字符串并退出内循环
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                # 如果未找到合适的子字符串，则标记为无效，并结束外循环
                if cur_substr is None:
                    is_bad = True
                    break
                # 将找到的子字符串添加到 sub_tokens 列表中
                sub_tokens.append(cur_substr)
                # 更新起始位置为当前子字符串的结束位置
                start = end

            # 根据标志变量决定将未知 token 或有效子 token 添加到输出列表
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        # 返回最终的 wordpiece token 列表
        return output_tokens
```