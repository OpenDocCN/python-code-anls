# `.\models\tapas\tokenization_tapas.py`

```
# coding=utf-8
# Copyright 2020 Google Research and The HuggingFace Inc. team.
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
""" Tokenization class for TAPAS model."""

import collections  # 导入 collections 模块
import datetime  # 导入 datetime 模块
import enum  # 导入 enum 枚举类型
import itertools  # 导入 itertools 模块
import math  # 导入 math 数学运算模块
import os  # 导入 os 操作系统接口模块
import re  # 导入 re 正则表达式模块
import unicodedata  # 导入 unicodedata Unicode 数据库
from dataclasses import dataclass  # 导入 dataclass 装饰器，用于定义不可变数据类
from typing import Callable, Dict, Generator, List, Optional, Text, Tuple, Union  # 导入类型提示模块

import numpy as np  # 导入 NumPy 数学计算库

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace  # 导入 tokenization_utils 模块中的相关函数
from ...tokenization_utils_base import (  # 导入 tokenization_utils_base 模块中的函数和类
    ENCODE_KWARGS_DOCSTRING,
    VERY_LARGE_INTEGER,
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
)
from ...utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available, logging  # 导入 utils 模块中的相关功能

if is_pandas_available():
    import pandas as pd  # 如果 pandas 可用，则导入 pandas 模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}  # 定义词汇表文件名

PRETRAINED_VOCAB_FILES_MAP = {  # 预训练模型词汇表文件映射为空字典
    # Map is intentionally left empty
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {name: 512 for name in PRETRAINED_VOCAB_FILES_MAP.keys()}  # 预训练位置嵌入大小映射，初始化为512
PRETRAINED_INIT_CONFIGURATION = {name: {"do_lower_case": True} for name in PRETRAINED_VOCAB_FILES_MAP.keys()}  # 预训练模型初始化配置，所有模型均为小写处理


class TapasTruncationStrategy(ExplicitEnum):
    """
    Possible values for the `truncation` argument in [`~TapasTokenizer.__call__`]. Useful for tab-completion in an IDE.
    """
    DROP_ROWS_TO_FIT = "drop_rows_to_fit"  # 截断策略：删除行以适应
    DO_NOT_TRUNCATE = "do_not_truncate"  # 截断策略：不截断


TableValue = collections.namedtuple("TokenValue", ["token", "column_id", "row_id"])  # 命名元组，用于表示表格中的一个单元格值


@dataclass(frozen=True)
class TokenCoordinates:
    column_index: int  # 列索引
    row_index: int  # 行索引
    token_index: int  # 令牌索引


@dataclass
class TokenizedTable:
    rows: List[List[List[Text]]]  # 表格的令牌化行列表
    selected_tokens: List[TokenCoordinates]  # 所选令牌的坐标列表


@dataclass(frozen=True)
class SerializedExample:
    tokens: List[Text]  # 序列化示例的令牌列表
    column_ids: List[int]  # 列标识符列表
    row_ids: List[int]  # 行标识符列表
    segment_ids: List[int]  # 段标识符列表


def _is_inner_wordpiece(token: Text):
    """判断是否为内部词片段"""
    return token.startswith("##")


def load_vocab(vocab_file):
    """加载词汇表文件到字典中"""
    vocab = collections.OrderedDict()  # 使用有序字典存储词汇表
    with open(vocab_file, "r", encoding="utf-8") as reader:  # 打开词汇表文件
        tokens = reader.readlines()  # 读取文件中的所有行
    for index, token in enumerate(tokens):  # 遍历行索引和行内容
        token = token.rstrip("\n")  # 去除行末换行符
        vocab[token] = index  # 将词汇和索引存入字典
    return vocab  # 返回加载后的词汇表字典


def whitespace_tokenize(text):
    """对文本进行基本的空格清理和分割"""
    text = text.strip()  # 去除文本两端空白字符
    # 如果文本为空，则返回空列表
    if not text:
        return []
    # 使用空格分割文本，生成令牌列表
    tokens = text.split()
    # 返回生成的令牌列表
    return tokens
"""
class TapasTokenizer(PreTrainedTokenizer):
    r"""
    Construct a TAPAS tokenizer. Based on WordPiece. Flattens a table and one or more related sentences to be used by
    TAPAS models.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods. [`TapasTokenizer`] creates several token type ids to
    encode tabular structure. To be more precise, it adds 7 token type ids, in the following order: `segment_ids`,
    `column_ids`, `row_ids`, `prev_labels`, `column_ranks`, `inv_column_ranks` and `numeric_relations`:

    - segment_ids: indicate whether a token belongs to the question (0) or the table (1). 0 for special tokens and
      padding.
    - column_ids: indicate to which column of the table a token belongs (starting from 1). Is 0 for all question
      tokens, special tokens and padding.
    - row_ids: indicate to which row of the table a token belongs (starting from 1). Is 0 for all question tokens,
      special tokens and padding. Tokens of column headers are also 0.
    - prev_labels: indicate whether a token was (part of) an answer to the previous question (1) or not (0). Useful in
      a conversational setup (such as SQA).
    - column_ranks: indicate the rank of a table token relative to a column, if applicable. For example, if you have a
      column "number of movies" with values 87, 53 and 69, then the column ranks of these tokens are 3, 1 and 2
      respectively. 0 for all question tokens, special tokens and padding.
    - inv_column_ranks: indicate the inverse rank of a table token relative to a column, if applicable. For example, if
      you have a column "number of movies" with values 87, 53 and 69, then the inverse column ranks of these tokens are
      1, 3 and 2 respectively. 0 for all question tokens, special tokens and padding.
    - numeric_relations: indicate numeric relations between the question and the tokens of the table. 0 for all
      question tokens, special tokens and padding.

    [`TapasTokenizer`] runs end-to-end tokenization on a table and associated sentences: punctuation splitting and
    wordpiece.

    """

    vocab_files_names = VOCAB_FILES_NAMES  # 词汇文件的名称列表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练词汇文件映射
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 预训练位置嵌入的最大模型输入大小
    # 初始化函数，用于设置和配置Tokenizer对象的各种参数和选项
    def __init__(
        # 词汇文件路径，用于加载Tokenizer的词汇表
        self,
        vocab_file,
        # 是否将输入文本转换为小写，默认为True
        do_lower_case=True,
        # 是否进行基本的分词，默认为True
        do_basic_tokenize=True,
        # 指定不进行分割的特殊标记列表，如果为None则没有特殊标记
        never_split=None,
        # 未知标记的字符串表示，默认为"[UNK]"
        unk_token="[UNK]",
        # 分隔标记的字符串表示，默认为"[SEP]"
        sep_token="[SEP]",
        # 填充标记的字符串表示，默认为"[PAD]"
        pad_token="[PAD]",
        # 类别标记的字符串表示，默认为"[CLS]"
        cls_token="[CLS]",
        # 掩码标记的字符串表示，默认为"[MASK]"
        mask_token="[MASK]",
        # 空标记的字符串表示，默认为"[EMPTY]"
        empty_token="[EMPTY]",
        # 是否对中文字符进行分词，默认为True
        tokenize_chinese_chars=True,
        # 是否去除字符串中的重音符号，默认为None（不去除）
        strip_accents=None,
        # 单元格修剪长度，指定列名称的最大长度，默认为-1（不限制）
        cell_trim_length: int = -1,
        # 最大列ID，默认为None（不限制）
        max_column_id: int = None,
        # 最大行ID，默认为None（不限制）
        max_row_id: int = None,
        # 是否去除列名的空格，默认为False
        strip_column_names: bool = False,
        # 是否更新答案坐标，默认为False
        update_answer_coordinates: bool = False,
        # 最小问题长度，默认为None（不限制）
        min_question_length=None,
        # 最大问题长度，默认为None（不限制）
        max_question_length=None,
        # 模型的最大长度，默认为512
        model_max_length: int = 512,
        # 额外的特殊标记列表，可以为None
        additional_special_tokens: Optional[List[str]] = None,
        # 其他可选参数，以字典形式接收
        **kwargs,
    ):
        ):
            # 检查是否安装了 Pandas 库，若未安装则抛出 ImportError 异常
            if not is_pandas_available():
                raise ImportError("Pandas is required for the TAPAS tokenizer.")

            # 处理额外的特殊标记，确保空标记在其中
            if additional_special_tokens is not None:
                if empty_token not in additional_special_tokens:
                    additional_special_tokens.append(empty_token)
            else:
                additional_special_tokens = [empty_token]

            # 检查词汇文件是否存在，若不存在则抛出 ValueError 异常
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                    " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                )

            # 加载词汇表并创建词汇到 ID 的映射，保持有序字典
            self.vocab = load_vocab(vocab_file)
            self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

            # 设置是否进行基本的分词处理
            self.do_basic_tokenize = do_basic_tokenize
            if do_basic_tokenize:
                self.basic_tokenizer = BasicTokenizer(
                    do_lower_case=do_lower_case,
                    never_split=never_split,
                    tokenize_chinese_chars=tokenize_chinese_chars,
                    strip_accents=strip_accents,
                )

            # 使用词汇表初始化 WordpieceTokenizer 对象
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

            # 设置额外的属性
            self.cell_trim_length = cell_trim_length
            # 设置列的最大 ID，如果未提供则使用 model_max_length 或设为一个非常大的整数
            self.max_column_id = (
                max_column_id
                if max_column_id is not None
                else model_max_length
                if model_max_length is not None
                else VERY_LARGE_INTEGER
            )
            # 设置行的最大 ID，如果未提供则使用 model_max_length 或设为一个非常大的整数
            self.max_row_id = (
                max_row_id
                if max_row_id is not None
                else model_max_length
                if model_max_length is not None
                else VERY_LARGE_INTEGER
            )
            # 是否去除列名中的空白字符
            self.strip_column_names = strip_column_names
            # 是否更新答案的坐标
            self.update_answer_coordinates = update_answer_coordinates
            # 最小问题长度限制
            self.min_question_length = min_question_length
            # 最大问题长度限制
            self.max_question_length = max_question_length

            # 调用父类的构造方法，初始化基本参数和额外的特殊标记等
            super().__init__(
                do_lower_case=do_lower_case,
                do_basic_tokenize=do_basic_tokenize,
                never_split=never_split,
                unk_token=unk_token,
                sep_token=sep_token,
                pad_token=pad_token,
                cls_token=cls_token,
                mask_token=mask_token,
                empty_token=empty_token,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
                cell_trim_length=cell_trim_length,
                max_column_id=max_column_id,
                max_row_id=max_row_id,
                strip_column_names=strip_column_names,
                update_answer_coordinates=update_answer_coordinates,
                min_question_length=min_question_length,
                max_question_length=max_question_length,
                model_max_length=model_max_length,
                additional_special_tokens=additional_special_tokens,
                **kwargs,
            )

        @property
    # 返回当前实例中的基本分词器的小写设置
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    # 返回当前词汇表的大小
    @property
    def vocab_size(self):
        return len(self.vocab)

    # 返回词汇表和添加的特殊token编码器组成的字典
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # 将文本进行标记化处理，返回标记列表
    def _tokenize(self, text):
        # 检查格式化后的文本是否为空文本，如果是，则返回一个特殊token的列表
        if format_text(text) == EMPTY_TEXT:
            return [self.additional_special_tokens[0]]
        split_tokens = []
        # 如果设置了基本分词，则使用基本分词器处理文本
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # 如果token属于不分割的特殊token集合，则直接加入split_tokens
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 否则使用wordpiece_tokenizer进一步分割token，加入split_tokens
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 否则直接使用wordpiece_tokenizer处理文本
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    # 根据token返回其在词汇表中的id，如果找不到则返回UNK（未知token）的id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 根据id返回词汇表中对应的token，如果找不到则返回UNK（未知token）
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    # 将token序列转换为单个字符串，去除"##"并去除两端空格
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # 将词汇表保存到指定目录中的文件中，并返回保存的文件路径
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        # 如果保存目录已存在，则在其下创建词汇表文件
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 否则直接在指定的保存目录或者文件名前缀下创建词汇表文件
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 使用utf-8编码打开文件，并逐行写入词汇表中的token
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
    def create_attention_mask_from_sequences(self, query_ids: List[int], table_values: List[TableValue]) -> List[int]:
        """
        根据查询的token ID和表格值创建注意力掩码。

        Args:
            query_ids (`List[int]`): 与查询相关的token ID列表。
            table_values (`List[TableValue]`): 表格值的列表，其中包含命名元组，包括token值、列ID和行ID。

        Returns:
            `List[int]`: 包含注意力掩码值的整数列表。
        """
        # 创建一个全为1的列表，长度为查询token数加1再加上表格值数加1
        return [1] * (1 + len(query_ids) + 1 + len(table_values))

    def create_segment_token_type_ids_from_sequences(
        self, query_ids: List[int], table_values: List[TableValue]
    ) -> List[int]:
        """
        根据查询的token ID和表格值创建段落token类型ID。

        Args:
            query_ids (`List[int]`): 与查询相关的token ID列表。
            table_values (`List[TableValue]`): 表格值的列表，其中包含命名元组，包括token值、列ID和行ID。

        Returns:
            `List[int]`: 包含段落token类型ID值的整数列表。
        """
        # 如果有表格值，则提取出所有表格值的第一个元素（token值），否则为空列表
        table_ids = list(zip(*table_values))[0] if table_values else []
        # 返回一个以0填充的列表，长度为查询token数加1再加上1，再加上以1填充的列表，长度为表格值中token值的数量
        return [0] * (1 + len(query_ids) + 1) + [1] * len(table_ids)

    def create_column_token_type_ids_from_sequences(
        self, query_ids: List[int], table_values: List[TableValue]
    ) -> List[int]:
        """
        根据查询的token ID和表格值创建列token类型ID。

        Args:
            query_ids (`List[int]`): 与查询相关的token ID列表。
            table_values (`List[TableValue]`): 表格值的列表，其中包含命名元组，包括token值、列ID和行ID。

        Returns:
            `List[int]`: 包含列token类型ID值的整数列表。
        """
        # 如果有表格值，则提取出所有表格值的第二个元素（列ID），否则为空列表
        table_column_ids = list(zip(*table_values))[1] if table_values else []
        # 返回一个以0填充的列表，长度为查询token数加1再加上1，再加上表格值中列ID数量的列表
        return [0] * (1 + len(query_ids) + 1) + list(table_column_ids)

    def create_row_token_type_ids_from_sequences(
        self, query_ids: List[int], table_values: List[TableValue]
    ) -> List[int]:
        """
        根据查询的token ID和表格值创建行token类型ID。
        
        Args:
            query_ids (`List[int]`): 与查询相关的token ID列表。
            table_values (`List[TableValue]`): 表格值的列表，其中包含命名元组，包括token值、列ID和行ID。

        Returns:
            `List[int]`: 包含行token类型ID值的整数列表。
        """
        # 如果有表格值，则提取出所有表格值的第三个元素（行ID），否则为空列表
        table_row_ids = list(zip(*table_values))[2] if table_values else []
        # 返回一个以0填充的列表，长度为查询token数加1再加上1，再加上表格值中行ID数量的列表
        return [0] * (1 + len(query_ids) + 1) + list(table_row_ids)
    ) -> List[int]:
        """
        Creates the row token type IDs according to the query token IDs and a list of table values.

        Args:
            query_ids (`List[int]`): list of token IDs corresponding to the ID.
            table_values (`List[TableValue]`): lift of table values, which are named tuples containing the
                token value, the column ID and the row ID of said token.

        Returns:
            `List[int]`: List of ints containing the row token type IDs values.
        """
        # Extract row IDs from table_values if it's not empty, otherwise initialize as an empty list
        table_row_ids = list(zip(*table_values))[2] if table_values else []
        # Generate row token type IDs list by concatenating [0], query_ids, [0] (for padding), and table_row_ids
        return [0] * (1 + len(query_ids) + 1) + list(table_row_ids)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a question and flattened table for question answering or sequence classification tasks
        by concatenating and adding special tokens.

        Args:
            token_ids_0 (`List[int]`): The ids of the question.
            token_ids_1 (`List[int]`, *optional*): The ids of the flattened table.

        Returns:
            `List[int]`: The model input with special tokens.
        """
        # Check if token_ids_1 is provided; raise error if not provided with TAPAS
        if token_ids_1 is None:
            raise ValueError("With TAPAS, you must provide both question IDs and table IDs.")
        # Concatenate cls_token_id, token_ids_0, sep_token_id, and token_ids_1 to build model input
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of question IDs.
            token_ids_1 (`List[int]`, *optional*):
                List of flattened table IDs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        # If already_has_special_tokens is True, delegate to the parent class method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        
        # If token_ids_1 is not None, return a mask indicating special tokens (1) and sequence tokens (0)
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))
        # If token_ids_1 is None, return a mask indicating special tokens (1) and sequence tokens (0)
        return [1] + ([0] * len(token_ids_0)) + [1]

    @add_end_docstrings(TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义一个方法，使其能像函数一样被调用
    def __call__(
        self,
        # 表格数据，使用 pandas 的 DataFrame 类型
        table: "pd.DataFrame",
        # 查询输入，可以是文本输入、预分词输入、编码输入，或它们的列表
        queries: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                EncodedInput,
                List[TextInput],
                List[PreTokenizedInput],
                List[EncodedInput],
            ]
        ] = None,
        # 答案的坐标，可以是单个或多个坐标的列表
        answer_coordinates: Optional[Union[List[Tuple], List[List[Tuple]]]] = None,
        # 答案的文本形式，可以是单个或多个文本输入的列表
        answer_text: Optional[Union[List[TextInput], List[List[TextInput]]]] = None,
        # 是否添加特殊标记，默认为 True
        add_special_tokens: bool = True,
        # 填充策略，可以是布尔值、字符串或填充策略对象，默认为 False
        padding: Union[bool, str, PaddingStrategy] = False,
        # 截断策略，可以是布尔值、字符串或截断策略对象，默认为 False
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        # 最大长度限制，默认为 None
        max_length: Optional[int] = None,
        # 填充到的最接近的倍数，默认为 None
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，默认为 None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回 token 类型 ID，默认为 None
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码，默认为 None
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的 token，默认为 False
        return_overflowing_tokens: bool = False,
        # 是否返回特殊 token 掩码，默认为 False
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射，默认为 False
        return_offsets_mapping: bool = False,
        # 是否返回长度，默认为 False
        return_length: bool = False,
        # 是否启用详细输出，默认为 True
        verbose: bool = True,
        **kwargs,
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 批量编码加方法的定义，具有与 __call__ 相似的参数
    def batch_encode_plus(
        self,
        # 表格数据，使用 pandas 的 DataFrame 类型
        table: "pd.DataFrame",
        # 查询输入的列表，可以是文本输入、预分词输入或编码输入的列表
        queries: Optional[
            Union[
                List[TextInput],
                List[PreTokenizedInput],
                List[EncodedInput],
            ]
        ] = None,
        # 答案的坐标列表的列表形式
        answer_coordinates: Optional[List[List[Tuple]]] = None,
        # 答案的文本列表的列表形式
        answer_text: Optional[List[List[TextInput]]] = None,
        # 是否添加特殊标记，默认为 True
        add_special_tokens: bool = True,
        # 填充策略，可以是布尔值、字符串或填充策略对象，默认为 False
        padding: Union[bool, str, PaddingStrategy] = False,
        # 截断策略，可以是布尔值、字符串或截断策略对象，默认为 False
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        # 最大长度限制，默认为 None
        max_length: Optional[int] = None,
        # 填充到的最接近的倍数，默认为 None
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，默认为 None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回 token 类型 ID，默认为 None
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码，默认为 None
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的 token，默认为 False
        return_overflowing_tokens: bool = False,
        # 是否返回特殊 token 掩码，默认为 False
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射，默认为 False
        return_offsets_mapping: bool = False,
        # 是否返回长度，默认为 False
        return_length: bool = False,
        # 是否启用详细输出，默认为 True
        verbose: bool = True,
        **kwargs,
    # 获取问题 tokens 的方法定义，输入参数是一个查询
    def _get_question_tokens(self, query):
        """Tokenizes the query, taking into account the max and min question length."""
        
        # 使用内部方法 tokenize 对查询进行分词处理，返回分词后的结果
        query_tokens = self.tokenize(query)
        # 如果设定了最大问题长度且查询分词后的长度超过最大问题长度，则记录警告并返回空字符串和空列表
        if self.max_question_length is not None and len(query_tokens) > self.max_question_length:
            logger.warning("Skipping query as its tokens are longer than the max question length")
            return "", []
        # 如果设定了最小问题长度且查询分词后的长度少于最小问题长度，则记录警告并返回空字符串和空列表
        if self.min_question_length is not None and len(query_tokens) < self.min_question_length:
            logger.warning("Skipping query as its tokens are shorter than the min question length")
            return "", []

        # 返回原始查询和其分词后的结果列表
        return query, query_tokens
    # 定义一个方法 `_batch_encode_plus`，用于批量编码输入数据并返回编码后的批处理结果
    def _batch_encode_plus(
        self,
        table,  # 表格数据，待编码的输入表格
        queries: Union[  # 查询数据，可以是文本输入、预分词输入或编码输入的列表
            List[TextInput],
            List[PreTokenizedInput],
            List[EncodedInput],
        ],
        answer_coordinates: Optional[List[List[Tuple]]] = None,  # 答案坐标，可选的二维列表，每个元素是一组坐标元组
        answer_text: Optional[List[List[TextInput]]] = None,  # 答案文本，可选的二维列表，每个元素是一组文本输入
        add_special_tokens: bool = True,  # 是否添加特殊标记，如 [CLS], [SEP]
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略，指定填充的方式
        truncation: Union[bool, str, TapasTruncationStrategy] = False,  # 截断策略，指定截断的方式
        max_length: Optional[int] = None,  # 最大长度，限制编码后的最大长度
        pad_to_multiple_of: Optional[int] = None,  # 填充到的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = True,  # 是否返回token类型ID
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的token
        return_special_tokens_mask: bool = False,  # 是否返回特殊token的掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回长度信息
        verbose: bool = True,  # 是否启用详细输出模式
        **kwargs,  # 其他参数，灵活处理额外的关键字参数
    ) -> BatchEncoding:  # 返回类型为 BatchEncoding 对象
        # 对输入的表格数据进行标记化处理，得到表格数据的token表示
        table_tokens = self._tokenize_table(table)

        # 初始化查询数据的token表示列表
        queries_tokens = []
        # 遍历查询数据列表，对每个查询进行处理
        for idx, query in enumerate(queries):
            # 调用内部方法 `_get_question_tokens` 处理查询，获取查询文本和token表示
            query, query_tokens = self._get_question_tokens(query)
            # 更新查询数据列表中的查询文本
            queries[idx] = query
            # 将查询的token表示添加到查询token列表中
            queries_tokens.append(query_tokens)

        # 调用内部方法 `_batch_prepare_for_model` 准备模型输入数据，进行编码和准备
        batch_outputs = self._batch_prepare_for_model(
            table,  # 表格数据
            queries,  # 查询数据
            tokenized_table=table_tokens,  # 表格数据的token表示
            queries_tokens=queries_tokens,  # 查询数据的token表示
            answer_coordinates=answer_coordinates,  # 答案坐标
            padding=padding,  # 填充策略
            truncation=truncation,  # 截断策略
            answer_text=answer_text,  # 答案文本
            add_special_tokens=add_special_tokens,  # 是否添加特殊token
            max_length=max_length,  # 最大长度
            pad_to_multiple_of=pad_to_multiple_of,  # 填充到的倍数
            return_tensors=return_tensors,  # 返回的张量类型
            prepend_batch_axis=True,  # 是否在返回结果中添加批处理维度
            return_attention_mask=return_attention_mask,  # 是否返回注意力掩码
            return_token_type_ids=return_token_type_ids,  # 是否返回token类型ID
            return_overflowing_tokens=return_overflowing_tokens,  # 是否返回溢出的token
            return_special_tokens_mask=return_special_tokens_mask,  # 是否返回特殊token的掩码
            return_length=return_length,  # 是否返回长度信息
            verbose=verbose,  # 是否启用详细输出模式
        )

        # 返回 BatchEncoding 对象，封装了批处理后的编码结果
        return BatchEncoding(batch_outputs)
    # 定义一个方法 `_batch_prepare_for_model`，用于准备数据以供模型处理
    def _batch_prepare_for_model(
        self,
        raw_table: "pd.DataFrame",  # 原始数据表格，类型为 Pandas DataFrame
        raw_queries: Union[  # 原始查询数据的列表，可以是不同类型的输入数据
            List[TextInput],  # 文本输入列表
            List[PreTokenizedInput],  # 预标记化输入列表
            List[EncodedInput],  # 编码输入列表
        ],
        tokenized_table: Optional[TokenizedTable] = None,  # 可选的表格数据经过标记化的形式
        queries_tokens: Optional[List[List[str]]] = None,  # 可选的查询标记化后的词列表
        answer_coordinates: Optional[List[List[Tuple]]] = None,  # 可选的答案坐标列表
        answer_text: Optional[List[List[TextInput]]] = None,  # 可选的答案文本列表
        add_special_tokens: bool = True,  # 是否添加特殊标记，默认为 True
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略，默认为 False
        truncation: Union[bool, str, TapasTruncationStrategy] = False,  # 截断策略，默认为 False
        max_length: Optional[int] = None,  # 最大长度限制，可选
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型，可选
        return_token_type_ids: Optional[bool] = True,  # 是否返回token类型id，默认为 True
        return_attention_mask: Optional[bool] = True,  # 是否返回注意力掩码，默认为 True
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记掩码，默认为 False
        return_offsets_mapping: bool = False,  # 是否返回偏移映射，默认为 False
        return_length: bool = False,  # 是否返回长度信息，默认为 False
        verbose: bool = True,  # 是否打印详细信息，默认为 True
        prepend_batch_axis: bool = False,  # 是否在结果中添加批处理维度，默认为 False
        **kwargs,  # 其它参数，灵活传递
    ):
    ) -> BatchEncoding:
        batch_outputs = {}  # 初始化一个空字典，用于存储批处理的输出结果

        # 遍历输入的四个列表的元素，每次迭代生成一个示例
        for index, example in enumerate(zip(raw_queries, queries_tokens, answer_coordinates, answer_text)):
            raw_query, query_tokens, answer_coords, answer_txt = example  # 解包示例元组到各个变量
            # 调用 self.prepare_for_model 方法准备模型输入，并获取输出结果
            outputs = self.prepare_for_model(
                raw_table,
                raw_query,
                tokenized_table=tokenized_table,
                query_tokens=query_tokens,
                answer_coordinates=answer_coords,
                answer_text=answer_txt,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # 设置不进行单独填充，而是批处理后再进行
                truncation=truncation,
                max_length=max_length,
                pad_to_multiple_of=None,  # 批处理后再进行填充
                return_attention_mask=False,  # 批处理后再进行填充
                return_token_type_ids=return_token_type_ids,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # 在最后将整个批次转换为张量
                prepend_batch_axis=False,
                verbose=verbose,
                prev_answer_coordinates=answer_coordinates[index - 1] if index != 0 else None,  # 前一个答案的坐标
                prev_answer_text=answer_text[index - 1] if index != 0 else None,  # 前一个答案的文本
            )

            # 将每个输出项添加到批处理输出字典中
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # 对批处理输出进行填充处理
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # 将填充后的批处理输出转换为 BatchEncoding 类型
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs  # 返回填充后的批处理输出对象
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 使用装饰器添加文档字符串，文档字符串内容包括 ENCODE_KWARGS_DOCSTRING 和 TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    def encode_plus(
        self,
        table: "pd.DataFrame",
        # 表格数据，必须是一个 Pandas 的 DataFrame，所有单元格的值必须是文本格式。可以使用 *.astype(str)* 转换数据框为字符串格式。
        query: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                EncodedInput,
            ]
        ] = None,
        # 查询问题，可以是字符串或者字符串列表的形式，用于编码和查询相关的表格信息。
        answer_coordinates: Optional[List[Tuple]] = None,
        # 答案坐标，用于指定表格中答案的坐标位置。
        answer_text: Optional[List[TextInput]] = None,
        # 答案文本，用于指定表格中答案的文本内容。
        add_special_tokens: bool = True,
        # 是否添加特殊标记，通常用于控制是否在编码过程中添加特殊标记，如 [CLS], [SEP] 等。
        padding: Union[bool, str, PaddingStrategy] = False,
        # 填充策略，用于控制输入序列的填充方式，可以是布尔值、字符串或者填充策略对象。
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        # 截断策略，用于控制输入序列的截断方式，可以是布尔值、字符串或者截断策略对象。
        max_length: Optional[int] = None,
        # 最大长度，用于控制编码后的序列的最大长度。
        pad_to_multiple_of: Optional[int] = None,
        # 填充到的倍数，用于控制序列填充后的长度为指定倍数。
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 返回张量类型，用于指定返回的编码结果的张量类型，如 'pt' 表示返回 PyTorch 张量。
        return_token_type_ids: Optional[bool] = None,
        # 是否返回 token 类型 ID，用于指定是否返回编码后序列的 token 类型 ID。
        return_attention_mask: Optional[bool] = None,
        # 是否返回注意力掩码，用于指定是否返回编码后序列的注意力掩码。
        return_special_tokens_mask: bool = False,
        # 是否返回特殊标记掩码，用于指定是否返回编码后序列的特殊标记掩码。
        return_offsets_mapping: bool = False,
        # 是否返回偏移映射，用于指定是否返回编码后序列的字符偏移映射。
        return_length: bool = False,
        # 是否返回长度，用于指定是否返回编码后序列的长度。
        verbose: bool = True,
        # 是否详细输出，用于控制是否输出详细的编码过程信息。
        **kwargs,
        # 其他参数，用于接收可能存在的其他关键字参数。
    ) -> BatchEncoding:
        """
        Prepare a table and a string for the model.

        Args:
            table (`pd.DataFrame`):
                Table containing tabular data. Note that all cell values must be text. Use *.astype(str)* on a Pandas
                dataframe to convert it to string.
            query (`str` or `List[str]`):
                Question related to a table to be encoded.
            answer_coordinates (`List[Tuple]` or `List[List[Tuple]]`, *optional*):
                Answer coordinates of each table-question pair in the batch. The answer_coordinates must be a single
                list of one or more tuples. Each tuple must be a (row_index, column_index) pair. The first data row
                (not the column header row) has index 0. The first column has index 0.
            answer_text (`List[str]` or `List[List[str]]`, *optional*):
                Answer text of each table-question pair in the batch. The answer_text must be a single list of one or
                more strings. Each string must be the answer text of a corresponding answer coordinate.
        """
        # 检查特殊情况，如果设置了return_token_type_ids但未设置add_special_tokens为True，则引发值错误
        if return_token_type_ids is not None and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        # 检查参数的一致性，如果提供了answer_coordinates但未提供answer_text，或者反之，则引发值错误
        if (answer_coordinates and not answer_text) or (not answer_coordinates and answer_text):
            raise ValueError("In case you provide answers, both answer_coordinates and answer_text should be provided")

        # 检查是否包含不支持的参数，如果kwargs中包含'is_split_into_words'，则引发未实现错误
        if "is_split_into_words" in kwargs:
            raise NotImplementedError("Currently TapasTokenizer only supports questions as strings.")

        # 检查是否请求返回偏移映射，由于Python tokenizers不支持该功能，因此引发未实现错误
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        # 调用内部方法_encode_plus，用给定参数编码表和查询，并返回编码结果
        return self._encode_plus(
            table=table,
            query=query,
            answer_coordinates=answer_coordinates,
            answer_text=answer_text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )
    # 定义一个方法 `_encode_plus`，用于对输入进行编码加工，并返回适用于模型输入的格式化数据
    def _encode_plus(
        self,
        table: "pd.DataFrame",  # 输入的表格数据，类型为 Pandas DataFrame
        query: Union[  # 查询文本，可以是文本输入的几种形式之一
            TextInput,  # 文本输入
            PreTokenizedInput,  # 预标记化的输入
            EncodedInput,  # 编码后的输入
        ],
        answer_coordinates: Optional[List[Tuple]] = None,  # 答案的坐标信息（可选）
        answer_text: Optional[List[TextInput]] = None,  # 答案的文本信息（可选）
        add_special_tokens: bool = True,  # 是否添加特殊标记（默认为 True）
        padding: Union[bool, str, PaddingStrategy] = False,  # 是否进行填充，填充策略可以是布尔值、字符串或填充策略对象
        truncation: Union[bool, str, TapasTruncationStrategy] = False,  # 是否截断输入，截断策略可以是布尔值、字符串或截断策略对象
        max_length: Optional[int] = None,  # 最大长度限制（可选）
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的倍数长度（可选）
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型（可选）
        return_token_type_ids: Optional[bool] = True,  # 是否返回 token 类型 ID（默认为 True）
        return_attention_mask: Optional[bool] = True,  # 是否返回注意力掩码（默认为 True）
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记的掩码（默认为 False）
        return_offsets_mapping: bool = False,  # 是否返回偏移映射（默认为 False）
        return_length: bool = False,  # 是否返回长度信息（默认为 False）
        verbose: bool = True,  # 是否显示详细信息（默认为 True）
        **kwargs,  # 其他关键字参数
    ):
        if query is None:  # 如果查询文本为 None
            query = ""  # 将查询文本设为空字符串
            logger.warning(  # 记录警告日志，提醒用户
                "TAPAS is a question answering model but you have not passed a query. Please be aware that the "
                "model will probably not behave correctly."
            )

        # 对表格进行标记化处理，生成表格 token
        table_tokens = self._tokenize_table(table)
        # 获取查询文本的 token 化结果和原始文本
        query, query_tokens = self._get_question_tokens(query)

        # 调用 self.prepare_for_model 方法，准备模型输入数据
        return self.prepare_for_model(
            table,  # 输入表格数据
            query,  # 查询文本
            tokenized_table=table_tokens,  # 表格的标记化结果
            query_tokens=query_tokens,  # 查询文本的标记化结果
            answer_coordinates=answer_coordinates,  # 答案的坐标信息
            answer_text=answer_text,  # 答案的文本信息
            add_special_tokens=add_special_tokens,  # 是否添加特殊标记
            truncation=truncation,  # 截断策略
            padding=padding,  # 填充策略
            max_length=max_length,  # 最大长度限制
            pad_to_multiple_of=pad_to_multiple_of,  # 填充到倍数长度
            return_tensors=return_tensors,  # 返回的张量类型
            prepend_batch_axis=True,  # 是否在结果中添加批次维度
            return_attention_mask=return_attention_mask,  # 是否返回注意力掩码
            return_token_type_ids=return_token_type_ids,  # 是否返回 token 类型 ID
            return_special_tokens_mask=return_special_tokens_mask,  # 是否返回特殊标记的掩码
            return_length=return_length,  # 是否返回长度信息
            verbose=verbose,  # 是否显示详细信息
        )

    # 将 ENCODE_KWARGS_DOCSTRING 和 TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING 添加为文档字符串
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义一个方法，准备数据以供模型使用
    def prepare_for_model(
        self,
        raw_table: "pd.DataFrame",  # 接收原始数据表格，类型为 pandas DataFrame
        raw_query: Union[  # 接收原始查询，可以是多种类型的输入数据
            TextInput,  # 文本输入
            PreTokenizedInput,  # 预分词的输入
            EncodedInput,  # 编码后的输入
        ],
        tokenized_table: Optional[TokenizedTable] = None,  # 可选参数，已经分词的表格数据
        query_tokens: Optional[TokenizedTable] = None,  # 可选参数，查询的分词结果
        answer_coordinates: Optional[List[Tuple]] = None,  # 可选参数，答案的坐标列表
        answer_text: Optional[List[TextInput]] = None,  # 可选参数，答案的文本列表
        add_special_tokens: bool = True,  # 是否添加特殊标记，默认为 True
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略，默认为 False
        truncation: Union[bool, str, TapasTruncationStrategy] = False,  # 截断策略，默认为 False
        max_length: Optional[int] = None,  # 可选参数，最大长度限制
        pad_to_multiple_of: Optional[int] = None,  # 可选参数，填充到指定的长度倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 可选参数，返回的张量类型
        return_token_type_ids: Optional[bool] = True,  # 是否返回 token 类型 id，默认为 True
        return_attention_mask: Optional[bool] = True,  # 是否返回注意力掩码，默认为 True
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记掩码，默认为 False
        return_offsets_mapping: bool = False,  # 是否返回偏移映射，默认为 False
        return_length: bool = False,  # 是否返回长度信息，默认为 False
        verbose: bool = True,  # 是否显示详细信息，默认为 True
        prepend_batch_axis: bool = False,  # 是否在结果中添加批次轴，默认为 False
        **kwargs,  # 其他关键字参数
    ):
        # 方法的具体实现在这里，根据参数准备数据以供模型使用

    # 定义一个方法，用于获取截断后的表格行
    def _get_truncated_table_rows(
        self,
        query_tokens: List[str],  # 查询的分词结果列表
        tokenized_table: TokenizedTable,  # 已分词的表格数据
        num_rows: int,  # 需要获取的行数
        num_columns: int,  # 表格的列数
        max_length: int,  # 最大长度限制
        truncation_strategy: Union[str, TapasTruncationStrategy],  # 截断策略，可以是字符串或 Tapas 截断策略对象
    ) -> Tuple[int, int]:
        """
        Truncates a sequence pair in-place following the strategy.

        Args:
            query_tokens (`List[str]`):
                List of strings corresponding to the tokenized query.
            tokenized_table (`TokenizedTable`):
                Tokenized table object representing the table.
            num_rows (`int`):
                Total number of rows in the table.
            num_columns (`int`):
                Total number of columns in the table.
            max_length (`int`):
                Maximum length constraint for the sequence pair.
            truncation_strategy (`str` or [`TapasTruncationStrategy`]):
                Truncation strategy to use. Only supports `"drop_rows_to_fit"` strategy.

        Returns:
            `Tuple[int, int]`: Tuple containing the number of rows after truncation and the number of tokens available
            for each table element.
        """
        # Ensure `truncation_strategy` is an instance of `TapasTruncationStrategy`
        if not isinstance(truncation_strategy, TapasTruncationStrategy):
            truncation_strategy = TapasTruncationStrategy(truncation_strategy)

        # Set `max_length` to default `self.model_max_length` if not provided
        if max_length is None:
            max_length = self.model_max_length

        # Implement truncation strategy: 'drop_rows_to_fit'
        if truncation_strategy == TapasTruncationStrategy.DROP_ROWS_TO_FIT:
            while True:
                # Calculate maximum number of tokens that can fit the table
                num_tokens = self._get_max_num_tokens(
                    query_tokens, tokenized_table, num_rows=num_rows, num_columns=num_columns, max_length=max_length
                )

                # If tokens fit the table, exit loop
                if num_tokens is not None:
                    # We could fit the table.
                    break

                # Attempt to drop a row to fit the table within the length constraint
                num_rows -= 1

                # Exit loop if no rows can be dropped further
                if num_rows < 1:
                    break
        elif truncation_strategy != TapasTruncationStrategy.DO_NOT_TRUNCATE:
            # Raise error if an unknown truncation strategy is provided
            raise ValueError(f"Unknown truncation strategy {truncation_strategy}.")

        # Return the number of rows after truncation and the number of tokens available,
        # ensuring at least 1 token is available if `num_tokens` is None
        return num_rows, num_tokens or 1

    def _tokenize_table(
        self,
        table=None,
    ):
        """
        Tokenizes column headers and cell texts of a table.

        Args:
            table (`pd.Dataframe`):
                Table to tokenize. Returns: `TokenizedTable`: TokenizedTable object.
        """
        tokenized_rows = []
        tokenized_row = []
        # tokenize column headers
        for column in table:
            # Check if column names should be stripped before tokenization
            if self.strip_column_names:
                # Tokenize an empty string for stripped column names
                tokenized_row.append(self.tokenize(""))
            else:
                # Tokenize the column name
                tokenized_row.append(self.tokenize(column))
        # Add tokenized column headers to the list of tokenized rows
        tokenized_rows.append(tokenized_row)

        # tokenize cell values
        for idx, row in table.iterrows():
            tokenized_row = []
            for cell in row:
                # Tokenize each cell value
                tokenized_row.append(self.tokenize(cell))
            # Add tokenized row to the list of tokenized rows
            tokenized_rows.append(tokenized_row)

        token_coordinates = []
        # Create token coordinates for each token in the tokenized table
        for row_index, row in enumerate(tokenized_rows):
            for column_index, cell in enumerate(row):
                for token_index, _ in enumerate(cell):
                    # Append token coordinates to the list
                    token_coordinates.append(
                        TokenCoordinates(
                            row_index=row_index,
                            column_index=column_index,
                            token_index=token_index,
                        )
                    )

        # Return a TokenizedTable object containing tokenized rows and token coordinates
        return TokenizedTable(
            rows=tokenized_rows,
            selected_tokens=token_coordinates,
        )

    def _question_encoding_cost(self, question_tokens):
        # Calculate the encoding cost for a question, including two extra tokens for SEP and CLS
        return len(question_tokens) + 2

    def _get_token_budget(self, question_tokens, max_length=None):
        """
        Computes the number of tokens left for the table after tokenizing a question, taking into account the max
        sequence length of the model.

        Args:
            question_tokens (`List[String]`):
                List of tokens representing the question. Returns: `int`: the number of tokens left for the table,
                given the model max length.
        """
        # Determine the remaining token budget for the table after encoding the question
        return (max_length if max_length is not None else self.model_max_length) - self._question_encoding_cost(
            question_tokens
        )
    def _get_table_values(self, table, num_columns, num_rows, num_tokens) -> Generator[TableValue, None, None]:
        """Iterates over partial table and returns token, column and row indexes."""
        # 遍历选定的表格中的令牌
        for tc in table.selected_tokens:
            # 第一行是表头行，跳过
            if tc.row_index >= num_rows + 1:
                continue
            # 如果列索引超过指定的列数，跳过
            if tc.column_index >= num_columns:
                continue
            # 获取表格中指定位置的单元格内容
            cell = table.rows[tc.row_index][tc.column_index]
            # 获取单元格中指定的令牌
            token = cell[tc.token_index]
            word_begin_index = tc.token_index
            # 不添加部分单词。查找起始词片段并检查是否符合令牌预算。
            while word_begin_index >= 0 and _is_inner_wordpiece(cell[word_begin_index]):
                word_begin_index -= 1
            # 如果起始词片段超过指定的令牌数量，跳过
            if word_begin_index >= num_tokens:
                continue
            # 返回表格中的值，包括令牌、列索引加一、行索引
            yield TableValue(token, tc.column_index + 1, tc.row_index)

    def _get_table_boundaries(self, table):
        """Return maximal number of rows, columns and tokens."""
        # 初始化最大的行数、列数和令牌数
        max_num_tokens = 0
        max_num_columns = 0
        max_num_rows = 0
        # 遍历选定的表格中的令牌
        for tc in table.selected_tokens:
            # 更新最大的列数、行数和令牌数
            max_num_columns = max(max_num_columns, tc.column_index + 1)
            max_num_rows = max(max_num_rows, tc.row_index + 1)
            max_num_tokens = max(max_num_tokens, tc.token_index + 1)
            # 确保最大的列数和行数不超过预设的最大值
            max_num_columns = min(self.max_column_id, max_num_columns)
            max_num_rows = min(self.max_row_id, max_num_rows)
        # 返回最大的行数、列数和令牌数
        return max_num_rows, max_num_columns, max_num_tokens

    def _get_table_cost(self, table, num_columns, num_rows, num_tokens):
        # 计算使用指定令牌数量时的表格代价
        return sum(1 for _ in self._get_table_values(table, num_columns, num_rows, num_tokens))

    def _get_max_num_tokens(self, question_tokens, tokenized_table, num_columns, num_rows, max_length):
        """Computes max number of tokens that can be squeezed into the budget."""
        # 获取问题令牌的预算
        token_budget = self._get_token_budget(question_tokens, max_length)
        # 获取表格的行数、列数和最大的令牌数
        _, _, max_num_tokens = self._get_table_boundaries(tokenized_table)
        # 如果单元格修剪长度大于等于零且最大令牌数超过单元格修剪长度，则将最大令牌数设为单元格修剪长度
        if self.cell_trim_length >= 0 and max_num_tokens > self.cell_trim_length:
            max_num_tokens = self.cell_trim_length
        num_tokens = 0
        # 遍历最大令牌数加一的范围
        for num_tokens in range(max_num_tokens + 1):
            # 计算使用指定令牌数量时的表格代价
            cost = self._get_table_cost(tokenized_table, num_columns, num_rows, num_tokens + 1)
            # 如果代价超过了令牌预算，停止遍历
            if cost > token_budget:
                break
        # 如果使用的令牌数小于最大令牌数
        if num_tokens < max_num_tokens:
            # 如果单元格修剪长度大于等于零，则不允许动态修剪
            if self.cell_trim_length >= 0:
                return None
            # 如果使用的令牌数为零，则返回空
            if num_tokens == 0:
                return None
        # 返回可使用的最大令牌数
        return num_tokens

    def _get_num_columns(self, table):
        # 获取表格的列数
        num_columns = table.shape[1]
        # 如果列数超过预设的最大列数，则抛出数值错误异常
        if num_columns >= self.max_column_id:
            raise ValueError("Too many columns")
        # 返回表格的列数
        return num_columns
    def _get_num_rows(self, table, drop_rows_to_fit):
        # 获取表格的行数
        num_rows = table.shape[0]
        # 如果行数超过最大允许的行数
        if num_rows >= self.max_row_id:
            # 如果允许删除超出部分的行
            if drop_rows_to_fit:
                # 将行数调整为最大允许行数减一
                num_rows = self.max_row_id - 1
            else:
                # 否则抛出异常，提示行数过多
                raise ValueError("Too many rows")
        # 返回最终确定的行数
        return num_rows

    def _serialize_text(self, question_tokens):
        """将文本序列化为索引数组。"""
        tokens = []
        segment_ids = []
        column_ids = []
        row_ids = []

        # 在序列化文本开头添加 [CLS] 标记
        tokens.append(self.cls_token)
        segment_ids.append(0)
        column_ids.append(0)
        row_ids.append(0)

        # 遍历问题的每个词汇
        for token in question_tokens:
            tokens.append(token)
            segment_ids.append(0)
            column_ids.append(0)
            row_ids.append(0)

        # 返回序列化后的 tokens, segment_ids, column_ids, row_ids
        return tokens, segment_ids, column_ids, row_ids

    def _serialize(
        self,
        question_tokens,
        table,
        num_columns,
        num_rows,
        num_tokens,
    ):
        """序列化表格和文本。"""
        tokens, segment_ids, column_ids, row_ids = self._serialize_text(question_tokens)

        # 在问题和表格 tokens 之间添加 [SEP] 标记
        tokens.append(self.sep_token)
        segment_ids.append(0)
        column_ids.append(0)
        row_ids.append(0)

        # 获取表格中的每个单元格的值，并添加到序列化结果中
        for token, column_id, row_id in self._get_table_values(table, num_columns, num_rows, num_tokens):
            tokens.append(token)
            segment_ids.append(1)  # 表示这是来自表格的内容
            column_ids.append(column_id)
            row_ids.append(row_id)

        # 返回序列化后的 SerializedExample 对象
        return SerializedExample(
            tokens=tokens,
            segment_ids=segment_ids,
            column_ids=column_ids,
            row_ids=row_ids,
        )

    def _get_column_values(self, table, col_index):
        """获取表格中指定列的数值。"""
        table_numeric_values = {}
        # 遍历表格的每一行
        for row_index, row in table.iterrows():
            cell = row[col_index]
            # 如果单元格的值是数值类型，则加入到结果字典中
            if cell.numeric_value is not None:
                table_numeric_values[row_index] = cell.numeric_value
        # 返回包含数值的字典
        return table_numeric_values

    def _get_cell_token_indexes(self, column_ids, row_ids, column_id, row_id):
        """获取特定列和行索引对应的 token 索引。"""
        # 遍历所有 token 的索引
        for index in range(len(column_ids)):
            # 如果找到与指定列和行索引对应的 token 索引
            if column_ids[index] - 1 == column_id and row_ids[index] - 1 == row_id:
                # 返回该 token 索引
                yield index
    def _get_numeric_column_ranks(self, column_ids, row_ids, table):
        """Returns column ranks for all numeric columns."""

        # 初始化列的排名和反向排名的列表，长度为列的数量
        ranks = [0] * len(column_ids)
        inv_ranks = [0] * len(column_ids)

        # 如果表格对象不为空
        if table is not None:
            # 遍历表格的所有列
            for col_index in range(len(table.columns)):
                # 获取当前列的所有数值
                table_numeric_values = self._get_column_values(table, col_index)

                # 如果当前列没有数值则跳过
                if not table_numeric_values:
                    continue

                try:
                    # 获取用于排序数值的函数
                    key_fn = get_numeric_sort_key_fn(table_numeric_values.values())
                except ValueError:
                    # 如果获取排序函数时发生错误则跳过当前列
                    continue

                # 将当前列的数值转换为排序后的字典形式
                table_numeric_values = {row_index: key_fn(value) for row_index, value in table_numeric_values.items()}

                # 创建一个反向映射字典，将数值映射到行索引的列表
                table_numeric_values_inv = collections.defaultdict(list)
                for row_index, value in table_numeric_values.items():
                    table_numeric_values_inv[value].append(row_index)

                # 对唯一的数值进行排序
                unique_values = sorted(table_numeric_values_inv.keys())

                # 根据数值的排名为每个单元格设置排名和反向排名
                for rank, value in enumerate(unique_values):
                    for row_index in table_numeric_values_inv[value]:
                        for index in self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index):
                            ranks[index] = rank + 1
                            inv_ranks[index] = len(unique_values) - rank

        # 返回列的排名和反向排名列表
        return ranks, inv_ranks

    def _get_numeric_sort_key_fn(self, table_numeric_values, value):
        """
        Returns the sort key function for comparing value to table values. The function returned will be a suitable
        input for the key param of the sort(). See number_annotation_utils._get_numeric_sort_key_fn for details

        Args:
            table_numeric_values: Numeric values of a column
            value: Numeric value in the question

        Returns:
            A function key function to compare column and question values.
        """
        # 如果表格数值为空，则返回 None
        if not table_numeric_values:
            return None
        # 将所有列的数值放入一个列表，并加入当前问题的数值
        all_values = list(table_numeric_values.values())
        all_values.append(value)
        try:
            # 获取所有数值的排序函数
            return get_numeric_sort_key_fn(all_values)
        except ValueError:
            # 如果获取排序函数时发生错误，则返回 None
            return None
    # 返回数值关系的嵌入

    # 创建一个字典，将表格单元格映射到其与问题中任何值的所有关系的集合
    cell_indices_to_relations = collections.defaultdict(set)
    
    # 如果问题和表格都不为空，则处理数值值跨度并添加到问题中
    if question is not None and table is not None:
        for numeric_value_span in question.numeric_spans:
            for value in numeric_value_span.values:
                for column_index in range(len(table.columns)):
                    # 获取该列的所有数值
                    table_numeric_values = self._get_column_values(table, column_index)
                    # 获取排序键函数
                    sort_key_fn = self._get_numeric_sort_key_fn(table_numeric_values, value)
                    if sort_key_fn is None:
                        continue
                    # 遍历每个单元格的数值，并确定数值关系
                    for row_index, cell_value in table_numeric_values.items():
                        relation = get_numeric_relation(value, cell_value, sort_key_fn)
                        if relation is not None:
                            cell_indices_to_relations[column_index, row_index].add(relation)

    # 为每个单元格的所有词片段添加一个特殊特征
    for (column_index, row_index), relations in cell_indices_to_relations.items():
        relation_set_index = 0
        for relation in relations:
            # 确保关系值大于等于Relation.EQ的值
            assert relation.value >= Relation.EQ.value
            relation_set_index += 2 ** (relation.value - Relation.EQ.value)
        # 获取单元格词片段的索引并设置数值关系
        for cell_token_index in self._get_cell_token_indexes(column_ids, row_ids, column_index, row_index):
            numeric_relations[cell_token_index] = relation_set_index

    # 返回计算得到的数值关系列表
    return numeric_relations
    # 返回用于计算答案损失的数值列表
    def _get_numeric_values(self, table, column_ids, row_ids):
        numeric_values = [float("nan")] * len(column_ids)  # 初始化一个长度为列数的数值列表，初始值为 NaN

        if table is not None:
            num_rows = table.shape[0]  # 获取表格的行数
            num_columns = table.shape[1]  # 获取表格的列数

            # 遍历表格的每一列和每一行
            for col_index in range(num_columns):
                for row_index in range(num_rows):
                    numeric_value = table.iloc[row_index, col_index].numeric_value  # 获取指定单元格的数值
                    if numeric_value is not None:
                        if numeric_value.float_value is None:
                            continue
                        float_value = numeric_value.float_value  # 获取数值的浮点值
                        if float_value == float("inf"):  # 如果浮点值为无穷大，则跳过
                            continue
                        # 获取当前单元格对应的 token 索引，并将数值赋给对应索引的数值列表
                        for index in self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index):
                            numeric_values[index] = float_value

        return numeric_values

    # 返回一个用于降低长单词价值的每个 token 的缩放比例列表
    def _get_numeric_values_scale(self, table, column_ids, row_ids):
        numeric_values_scale = [1.0] * len(column_ids)  # 初始化一个长度为列数的缩放比例列表，初始值为 1.0

        if table is None:
            return numeric_values_scale  # 如果表格为空，则直接返回初始的缩放比例列表

        num_rows = table.shape[0]  # 获取表格的行数
        num_columns = table.shape[1]  # 获取表格的列数

        # 遍历表格的每一列和每一行
        for col_index in range(num_columns):
            for row_index in range(num_rows):
                indices = list(self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index))  # 获取单元格对应的 token 索引列表
                num_indices = len(indices)  # 获取 token 索引列表的长度
                if num_indices > 1:
                    # 如果单元格对应的 token 索引数量大于 1，则将缩放比例设置为索引的数量
                    for index in indices:
                        numeric_values_scale[index] = float(num_indices)

        return numeric_values_scale

    # 将输入列表填充到模型最大长度
    def _pad_to_seq_length(self, inputs):
        while len(inputs) > self.model_max_length:  # 当输入列表长度超过模型最大长度时
            inputs.pop()  # 移除末尾的元素
        while len(inputs) < self.model_max_length:  # 当输入列表长度小于模型最大长度时
            inputs.append(0)  # 在末尾添加值为 0 的元素

    # 根据答案坐标获取所有答案的 token 索引列表和缺失答案数量
    def _get_all_answer_ids_from_coordinates(
        self,
        column_ids,
        row_ids,
        answers_list,
    ):
        """Maps lists of answer coordinates to token indexes."""
        answer_ids = [0] * len(column_ids)  # 初始化一个长度为列数的答案 ID 列表，初始值为 0
        found_answers = set()  # 用于存储已找到的答案坐标的集合
        all_answers = set()  # 用于存储所有答案坐标的集合

        for answers in answers_list:  # 遍历答案坐标列表
            column_index, row_index = answers  # 获取列索引和行索引
            all_answers.add((column_index, row_index))  # 将答案坐标添加到所有答案集合中
            for index in self._get_cell_token_indexes(column_ids, row_ids, column_index, row_index):
                # 获取答案坐标对应的 token 索引，并将答案标记为已找到
                found_answers.add((column_index, row_index))
                answer_ids[index] = 1  # 将答案对应的 token 索引位置设置为 1，表示找到了答案

        missing_count = len(all_answers) - len(found_answers)  # 计算未找到的答案数量
        return answer_ids, missing_count  # 返回答案 ID 列表和未找到的答案数量
    def _get_all_answer_ids(self, column_ids, row_ids, answer_coordinates):
        """
        Maps answer coordinates of a question to token indexes.

        In the SQA format (TSV), the coordinates are given as (row, column) tuples. Here, we first swap them to
        (column, row) format before calling _get_all_answer_ids_from_coordinates.
        """

        def _to_coordinates(answer_coordinates_question):
            # 转换答案坐标格式为 (column, row) 形式
            return [(coords[1], coords[0]) for coords in answer_coordinates_question]

        # 调用 _get_all_answer_ids_from_coordinates 方法，传入调整后的答案坐标
        return self._get_all_answer_ids_from_coordinates(
            column_ids, row_ids, answers_list=(_to_coordinates(answer_coordinates))
        )

    def _find_tokens(self, text, segment):
        """Return start index of segment in text or None."""
        # 记录调试信息，输出文本和查找的段落
        logging.info(f"text: {text} {segment}")
        # 在文本中查找段落的起始索引
        for index in range(1 + len(text) - len(segment)):
            for seg_index, seg_token in enumerate(segment):
                # 如果当前位置的字符与段落不匹配，则终止此次匹配
                if text[index + seg_index].piece != seg_token.piece:
                    break
            else:
                # 如果完全匹配，则返回段落在文本中的起始索引
                return index
        # 如果未找到匹配的段落，则返回 None
        return None

    def _find_answer_coordinates_from_answer_text(
        self,
        tokenized_table,
        answer_text,
    ):
        """Returns all occurrences of answer_text in the table."""
        # 记录调试信息，输出答案文本
        logging.info(f"answer text: {answer_text}")
        # 遍历表格的每一行和每一列，寻找答案文本的位置
        for row_index, row in enumerate(tokenized_table.rows):
            if row_index == 0:
                # 跳过表头行，不在表头中搜索答案
                continue
            for col_index, cell in enumerate(row):
                # 在单元格中查找答案文本的 token 索引
                token_index = self._find_tokens(cell, answer_text)
                if token_index is not None:
                    # 如果找到匹配的答案文本，则生成对应的 token 坐标
                    yield TokenCoordinates(
                        row_index=row_index,
                        column_index=col_index,
                        token_index=token_index,
                    )

    def _find_answer_ids_from_answer_texts(
        self,
        column_ids,
        row_ids,
        tokenized_table,
        answer_texts,
    ):
        """
        Returns answer IDs corresponding to given answer texts in a tokenized table.

        This function iterates through provided answer texts, finds their token positions in the tokenized table,
        and yields corresponding answer IDs based on column and row IDs.
        """
        # 循环遍历每个答案文本，查找其在 token 化表格中的位置，并返回对应的答案 ID
        for answer_text in answer_texts:
            for token_coord in self._find_answer_coordinates_from_answer_text(tokenized_table, answer_text):
                yield (column_ids[token_coord.column_index], row_ids[token_coord.row_index])
    ):
        """
        Maps question with answer texts to the first matching token indexes.
        """
        answer_ids = [0] * len(column_ids)
        for answer_text in answer_texts:
            for coordinates in self._find_answer_coordinates_from_answer_text(
                tokenized_table,
                answer_text,
            ):
                # Maps answer coordinates to indexes; this can fail if tokens/rows have
                # been pruned.
                indexes = list(
                    self._get_cell_token_indexes(
                        column_ids,
                        row_ids,
                        column_id=coordinates.column_index,
                        row_id=coordinates.row_index - 1,
                    )
                )
                indexes.sort()
                coordinate_answer_ids = []
                if indexes:
                    begin_index = coordinates.token_index + indexes[0]
                    end_index = begin_index + len(answer_text)
                    for index in indexes:
                        if index >= begin_index and index < end_index:
                            coordinate_answer_ids.append(index)
                if len(coordinate_answer_ids) == len(answer_text):
                    for index in coordinate_answer_ids:
                        answer_ids[index] = 1
                    break
        return answer_ids

    def _get_answer_ids(self, column_ids, row_ids, answer_coordinates):
        """
        Maps answer coordinates of a question to token indexes.
        """
        answer_ids, missing_count = self._get_all_answer_ids(column_ids, row_ids, answer_coordinates)

        if missing_count:
            raise ValueError("Couldn't find all answers")
        return answer_ids

    def get_answer_ids(self, column_ids, row_ids, tokenized_table, answer_texts_question, answer_coordinates_question):
        """
        Retrieves answer IDs based on whether to update answer coordinates or not.
        """
        if self.update_answer_coordinates:
            return self._find_answer_ids_from_answer_texts(
                column_ids,
                row_ids,
                tokenized_table,
                answer_texts=[self.tokenize(at) for at in answer_texts_question],
            )
        return self._get_answer_ids(column_ids, row_ids, answer_coordinates_question)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ):
        """
        Handles padding of encoded inputs according to specified strategies.
        """
        # Everything related to converting logits to predictions

    def _get_cell_token_probs(self, probabilities, segment_ids, row_ids, column_ids):
        """
        Yields token probabilities for cell tokens based on conditions.
        """
        for i, p in enumerate(probabilities):
            segment_id = segment_ids[i]
            col = column_ids[i] - 1
            row = row_ids[i] - 1
            if col >= 0 and row >= 0 and segment_id == 1:
                yield i, p
    # 计算每个单元格的平均概率，根据标记的概率值进行聚合计算
    def _get_mean_cell_probs(self, probabilities, segment_ids, row_ids, column_ids):
        """Computes average probability per cell, aggregating over tokens."""
        # 使用默认字典存储坐标对应的概率列表
        coords_to_probs = collections.defaultdict(list)
        # 遍历获取每个单元格中的标记概率
        for i, prob in self._get_cell_token_probs(probabilities, segment_ids, row_ids, column_ids):
            # 获取单元格所在列和行，将其从1-based转换为0-based
            col = column_ids[i] - 1
            row = row_ids[i] - 1
            # 将概率添加到坐标对应的概率列表中
            coords_to_probs[(col, row)].append(prob)
        # 计算每个坐标对应的单元格概率的平均值，并返回结果字典
        return {coords: np.array(cell_probs).mean() for coords, cell_probs in coords_to_probs.items()}
    
    # 转换逻辑值到预测结果的所有相关内容结束
# Copied from transformers.models.bert.tokenization_bert.BasicTokenizer
class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            In some instances we want to skip the basic punctuation splitting so that later tokenization can capture
            the full context of the words, such as contractions.
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果 `never_split` 为 None，则初始化为空列表
        if never_split is None:
            never_split = []
        # 设置是否将输入文本转换为小写
        self.do_lower_case = do_lower_case
        # 将 `never_split` 转换为集合，用于存储不需要分割的特殊标记
        self.never_split = set(never_split)
        # 设置是否对中文字符进行单独的分词处理
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 设置是否去除所有的重音符号
        self.strip_accents = strip_accents
        # 设置是否进行基本的标点符号分割
        self.do_split_on_punc = do_split_on_punc
    # 对文本进行基本的分词处理。如果需要子词分词，请参考WordPieceTokenizer。
    # 
    # Args:
    #     never_split (`List[str]`, *optional*)
    #         为了向后兼容保留。现在直接在基类级别实现（参见`PreTrainedTokenizer.tokenize`）不分割的标记列表。
    def tokenize(self, text, never_split=None):
        # 如果传入了never_split参数，则将其与self.never_split合并为一个新的集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，去除可能存在的特殊字符
        text = self._clean_text(text)

        # 以下代码块是为了处理多语言和中文模型而添加的，自2018年11月1日起。
        # 现在英语模型也适用，尽管由于英语模型未经过任何中文数据的训练，
        # 并且通常不包含任何中文数据（因为维基百科在英语版本中确实包含一些中文词汇）。
        if self.tokenize_chinese_chars:
            # 如果启用了中文字符分词，则调用内部方法_tokenize_chinese_chars处理文本
            text = self._tokenize_chinese_chars(text)
        
        # 使用Unicode NFC规范化文本，确保统一表示同一字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用whitespace_tokenize对文本进行空白字符分割，获取原始token列表
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        
        # 遍历原始token列表，处理每个token
        for token in orig_tokens:
            # 如果token不在never_split中，则可能需要进一步处理
            if token not in never_split:
                if self.do_lower_case:
                    # 如果需要小写化处理，则将token转换为小写
                    token = token.lower()
                    # 如果需要去除重音符号，则调用_run_strip_accents方法处理token
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    # 如果仅需要去除重音符号，则调用_run_strip_accents方法处理token
                    token = self._run_strip_accents(token)
            # 将处理后的token列表拼接到split_tokens中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将split_tokens中的token用空白字符连接成字符串，再进行空白字符分割，获取最终输出的token列表
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    # 从文本中去除重音符号
    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            # 如果字符的Unicode类别为Mn（Mark, Nonspacing），则跳过该字符
            if cat == "Mn":
                continue
            # 将不含重音符号的字符加入到output列表中
            output.append(char)
        # 将output列表中的字符连接成字符串并返回
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要在标点符号处分割，或者指定的文本在never_split列表中，则返回原始文本列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                # 如果是标点符号，创建一个新列表存储当前标点符号
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    # 如果是新词的开始，创建一个空列表
                    output.append([])
                start_new_word = False
                # 将当前字符添加到当前词的列表中
                output[-1].append(char)
            i += 1

        # 将列表中的子列表转换为字符串并返回
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                # 如果是中文字符，则在其前后添加空格
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        # 将列表中的字符连接成字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查给定的码点是否是CJK字符的码点范围内
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
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                # 如果是空白字符，则替换为单个空格
                output.append(" ")
            else:
                output.append(char)
        # 将列表中的字符连接成字符串并返回
        return "".join(output)
# Copied from transformers.models.bert.tokenization_bert.WordpieceTokenizer
# WordpieceTokenizer 类，用于运行 WordPiece 分词算法。

class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""
    # 初始化 WordpieceTokenizer 类
    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化词汇表、未知标记和每个单词最大输入字符数
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    # 对文本进行 WordPiece 分词处理
    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """
        # 初始化输出的 token 列表
        output_tokens = []
        # 使用 whitespace_tokenize 函数对文本进行分词
        for token in whitespace_tokenize(text):
            chars = list(token)
            # 如果单词长度超过设定的最大字符数，则使用未知标记代替
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            # 贪婪算法，尝试寻找最长匹配的词片段
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    # 对非首字符的片段加上 '##' 前缀，表示连接词的一部分
                    if start > 0:
                        substr = "##" + substr
                    # 如果片段在词汇表中，则认为是一个有效的词片段
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                # 如果没有找到匹配的词片段，则将该 token 标记为未知标记
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            # 如果标记为无效，则使用未知标记代替
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        # 返回最终的 wordpiece tokens 列表
        return output_tokens


# Below: utilities for TAPAS tokenizer (independent from PyTorch/Tensorflow).
# This includes functions to parse numeric values (dates and numbers) from both the table and questions in order
# to create the column_ranks, inv_column_ranks, numeric_values, numeric values_scale and numeric_relations in
# prepare_for_model of TapasTokenizer.
# These are meant to be used in an academic setup, for production use cases Gold mine or Aqua should be used.


# taken from constants.py of the original implementation
# URL: https://github.com/google-research/tapas/blob/master/tapas/utils/constants.py
# 定义了不同类型的关系，用于在表格处理中连接不同的元素

class Relation(enum.Enum):
    HEADER_TO_CELL = 1  # 连接表头到单元格
    CELL_TO_HEADER = 2  # 连接单元格到表头
    QUERY_TO_HEADER = 3  # 连接查询到表头
    QUERY_TO_CELL = 4  # 连接查询到单元格
    ROW_TO_CELL = 5  # 连接行到单元格
    CELL_TO_ROW = 6  # 连接单元格到行
    EQ = 7  # 标注值等于单元格值
    # 定义常量 LT，表示注释值小于单元格值
    LT = 8  # Annotation value is less than cell value
    
    # 定义常量 GT，表示注释值大于单元格值
    GT = 9  # Annotation value is greater than cell value
# 使用 dataclass 装饰器定义日期类，支持可选的年、月、日属性
@dataclass
class Date:
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None

# 使用 dataclass 装饰器定义数值类，支持可选的浮点值和日期属性
@dataclass
class NumericValue:
    float_value: Optional[float] = None
    date: Optional[Date] = None

# 使用 dataclass 装饰器定义数值区间类，包含开始和结束索引以及数值列表属性
@dataclass
class NumericValueSpan:
    begin_index: int = None
    end_index: int = None
    values: List[NumericValue] = None

# 使用 dataclass 装饰器定义单元格类，包含文本和可选的数值属性
@dataclass
class Cell:
    text: Text
    numeric_value: Optional[NumericValue] = None

# 使用 dataclass 装饰器定义问题类，包含原始文本、归一化后的文本和可选的数值区间列表属性
@dataclass
class Question:
    original_text: Text  # 原始问题字符串
    text: Text  # 归一化后的问题字符串
    numeric_spans: Optional[List[NumericValueSpan]] = None

# 下面是从 number_utils.py 中导入的所有函数以及从 text_utils.py 中导入的两个函数（即 get_all_spans 和 normalize_for_match）
# 原始实现的 URL 可查阅：
# - https://github.com/google-research/tapas/blob/master/tapas/utils/number_utils.py
# - https://github.com/google-research/tapas/blob/master/tapas/utils/text_utils.py

# 用于解析日期表达式的常量
# 命名元组 _DateMask 指定了哪些字段（年、月、日）将被填充
_DateMask = collections.namedtuple("_DateMask", ["year", "month", "day"])

# 常量 _YEAR 表示只填充年份
_YEAR = _DateMask(True, False, False)

# 常量 _YEAR_MONTH 表示填充年份和月份
_YEAR_MONTH = _DateMask(True, True, False)

# 常量 _YEAR_MONTH_DAY 表示填充年份、月份和日期
_YEAR_MONTH_DAY = _DateMask(True, True, True)

# 常量 _MONTH 表示只填充月份
_MONTH = _DateMask(False, True, False)

# 常量 _MONTH_DAY 表示填充月份和日期
_MONTH_DAY = _DateMask(False, True, True)

# _DATE_PATTERNS 是一个元组，每个元素包含一个日期格式和一个对应的 _DateMask，用于 datetime.strptime 的参数
_DATE_PATTERNS = (
    ("%B", _MONTH),
    ("%Y", _YEAR),
    ("%Ys", _YEAR),
    ("%b %Y", _YEAR_MONTH),
    ("%B %Y", _YEAR_MONTH),
    ("%B %d", _MONTH_DAY),
    ("%b %d", _MONTH_DAY),
    ("%d %b", _MONTH_DAY),
    ("%d %B", _MONTH_DAY),
    ("%B %d, %Y", _YEAR_MONTH_DAY),
    ("%d %B %Y", _YEAR_MONTH_DAY),
    ("%m-%d-%Y", _YEAR_MONTH_DAY),
    ("%Y-%m-%d", _YEAR_MONTH_DAY),
    ("%Y-%m", _YEAR_MONTH),
    ("%B %Y", _YEAR_MONTH),
    ("%d %b %Y", _YEAR_MONTH_DAY),
    ("%Y-%m-%d", _YEAR_MONTH_DAY),
    ("%b %d, %Y", _YEAR_MONTH_DAY),
    ("%d.%m.%Y", _YEAR_MONTH_DAY),
    ("%A, %b %d", _MONTH_DAY),
    ("%A, %B %d", _MONTH_DAY),
)

# _FIELD_TO_REGEX 是一个元组，每个元素包含一个日期格式和一个对应的正则表达式，用于将日期格式转换为正则表达式
_FIELD_TO_REGEX = (
    ("%A", r"\w+"),    # 本地化全名的星期几
    ("%B", r"\w+"),    # 本地化全名的月份
    ("%Y", r"\d{4}"),  # 带世纪的年份作为十进制数
    ("%b", r"\w{3}"),  # 本地化缩写的月份
    ("%d", r"\d{1,2}"),  # 月份中的天数，作为零填充的十进制数
    ("%m", r"\d{1,2}"),  # 月份作为零填充的十进制数
)

def _process_date_pattern(dp):
    """为每个日期模式计算一个正则表达式作为预过滤器。"""
    pattern, mask = dp
    regex = pattern
    regex = regex.replace(".", re.escape("."))  # 转义点号
    regex = regex.replace("-", re.escape("-"))  # 转义破折号
    regex = regex.replace(" ", r"\s+")  # 替换空格为匹配任意空白字符的正则表达式
    # 遍历 `_FIELD_TO_REGEX` 列表中的每个元素，元素包含字段名和对应的正则表达式
    for field, field_regex in _FIELD_TO_REGEX:
        # 替换当前正则表达式 `regex` 中的字段名 `field` 为对应的字段正则表达式 `field_regex`
        regex = regex.replace(field, field_regex)
    
    # 断言检查，确保替换后的 `regex` 中不包含 `%` 符号，否则输出当前的 `regex`
    assert "%" not in regex, regex
    
    # 返回编译后的模式 `pattern`、掩码 `mask` 和以 `regex` 开头和结尾的编译后的正则表达式对象
    return pattern, mask, re.compile("^" + regex + "$")
def _process_date_patterns():
    # 调用 _process_date_pattern 函数处理 _DATE_PATTERNS 中的每个模式，并返回处理后的元组
    return tuple(_process_date_pattern(dp) for dp in _DATE_PATTERNS)


_PROCESSED_DATE_PATTERNS = _process_date_patterns()

_MAX_DATE_NGRAM_SIZE = 5

# Following DynSp:
# https://github.com/Microsoft/DynSP/blob/master/util.py#L414.
_NUMBER_WORDS = [
    "zero",     # 数字 0 对应的英文单词
    "one",      # 数字 1 对应的英文单词
    "two",      # 数字 2 对应的英文单词
    "three",    # 数字 3 对应的英文单词
    "four",     # 数字 4 对应的英文单词
    "five",     # 数字 5 对应的英文单词
    "six",      # 数字 6 对应的英文单词
    "seven",    # 数字 7 对应的英文单词
    "eight",    # 数字 8 对应的英文单词
    "nine",     # 数字 9 对应的英文单词
    "ten",      # 数字 10 对应的英文单词
    "eleven",   # 数字 11 对应的英文单词
    "twelve",   # 数字 12 对应的英文单词
]

_ORDINAL_WORDS = [
    "zeroth",    # 序数 0 对应的英文单词
    "first",     # 序数 1 对应的英文单词
    "second",    # 序数 2 对应的英文单词
    "third",     # 序数 3 对应的英文单词
    "fourth",    # 序数 4 对应的英文单词
    "fith",      # 序数 5 对应的英文单词 (可能应为 fifth)
    "sixth",     # 序数 6 对应的英文单词
    "seventh",   # 序数 7 对应的英文单词
    "eighth",    # 序数 8 对应的英文单词
    "ninth",     # 序数 9 对应的英文单词
    "tenth",     # 序数 10 对应的英文单词
    "eleventh",  # 序数 11 对应的英文单词
    "twelfth",   # 序数 12 对应的英文单词
]

_ORDINAL_SUFFIXES = ["st", "nd", "rd", "th"]  # 各种序数的后缀列表

_NUMBER_PATTERN = re.compile(r"((^|\s)[+-])?((\.\d+)|(\d+(,\d\d\d)*(\.\d*)?))")
# 匹配简单的数值表达式的正则表达式模式，包括正负号、逗号分隔的千位数和小数点数值

# Following DynSp:
# https://github.com/Microsoft/DynSP/blob/master/util.py#L293.
_MIN_YEAR = 1700    # 可接受的最小年份
_MAX_YEAR = 2016    # 可接受的最大年份

_INF = float("INF")  # 无穷大的浮点数表示


def _get_numeric_value_from_date(date, mask):
    """Converts date (datetime Python object) to a NumericValue object with a Date object value."""
    if date.year < _MIN_YEAR or date.year > _MAX_YEAR:
        raise ValueError(f"Invalid year: {date.year}")

    new_date = Date()
    if mask.year:
        new_date.year = date.year
    if mask.month:
        new_date.month = date.month
    if mask.day:
        new_date.day = date.day
    return NumericValue(date=new_date)


def _get_span_length_key(span):
    """Sorts span by decreasing length first and increasing first index second."""
    return span[1] - span[0], -span[0]


def _get_numeric_value_from_float(value):
    """Converts float (Python) to a NumericValue object with a float value."""
    return NumericValue(float_value=value)


# Doesn't parse ordinal expressions such as '18th of february 1655'.
def _parse_date(text):
    """Attempts to format a text as a standard date string (yyyy-mm-dd)."""
    text = re.sub(r"Sept\b", "Sep", text)  # 替换文本中的 "Sept" 为 "Sep"
    for in_pattern, mask, regex in _PROCESSED_DATE_PATTERNS:
        if not regex.match(text):
            continue
        try:
            date = datetime.datetime.strptime(text, in_pattern).date()  # 尝试解析文本为日期对象
        except ValueError:
            continue
        try:
            return _get_numeric_value_from_date(date, mask)  # 转换日期为 NumericValue 对象并返回
        except ValueError:
            continue
    return None


def _parse_number(text):
    """Parses simple cardinal and ordinals numbers."""
    for suffix in _ORDINAL_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]  # 去除文本末尾的序数后缀
            break
    text = text.replace(",", "")  # 去除文本中的逗号
    try:
        value = float(text)  # 尝试将文本转换为浮点数
    except ValueError:
        return None
    if math.isnan(value):
        return None
    if value == _INF:
        return None
    return value


def get_all_spans(text, max_ngram_length):
    """
    Split a text into all possible ngrams up to 'max_ngram_length'. Split points are white space and punctuation.

    Args:
      text: Text to split.
      max_ngram_length: maximal ngram length.
    """
    # 初始化一个空列表，用于存储起始索引
    start_indexes = []
    # 遍历文本中的每个字符及其索引
    for index, char in enumerate(text):
        # 如果当前字符不是字母或数字，则跳过当前循环，继续下一个字符
        if not char.isalnum():
            continue
        # 如果当前字符是字母或数字，并且满足以下条件之一：
        # 1. 是文本的第一个字符
        # 2. 前一个字符不是字母或数字
        # 则将当前索引添加到起始索引列表中
        if index == 0 or not text[index - 1].isalnum():
            start_indexes.append(index)
        # 如果当前字符是字母或数字，并且满足以下条件之一：
        # 1. 是文本的最后一个字符
        # 2. 后一个字符不是字母或数字
        # 针对起始索引列表中的最后几个元素生成 n-gram 的起始索引和结束索引
        if index + 1 == len(text) or not text[index + 1].isalnum():
            for start_index in start_indexes[-max_ngram_length:]:
                # 返回生成器，生成 n-gram 的起始索引和结束索引（不包含结束索引本身）
                yield start_index, index + 1
# 将文本转换为小写，并去除多余的空格
def normalize_for_match(text):
    return " ".join(text.lower().split())


# 将文本转换为小写并去除标点符号
def format_text(text):
    """Lowercases and strips punctuation."""
    text = text.lower().strip()
    # 如果文本是 "n/a"、"?" 或 "nan"，则置为空文本
    if text == "n/a" or text == "?" or text == "nan":
        text = EMPTY_TEXT

    # 使用正则表达式替换非字母数字字符为空格，并将下划线替换为空格
    text = re.sub(r"[^\w\d]+", " ", text).replace("_", " ")
    # 去除多余的空格
    text = " ".join(text.split())
    text = text.strip()
    # 如果处理后的文本非空，则返回处理后的文本；否则返回空文本
    if text:
        return text
    return EMPTY_TEXT


# 解析文本，提取最长的数字值和日期跨度
def parse_text(text):
    """
    Extracts longest number and date spans.

    Args:
      text: text to annotate

    Returns:
      List of longest numeric value spans.
    """
    span_dict = collections.defaultdict(list)

    # 提取所有数字模式的匹配项，并解析成数字
    for match in _NUMBER_PATTERN.finditer(text):
        span_text = text[match.start() : match.end()]
        number = _parse_number(span_text)
        if number is not None:
            # 将解析出的数字值添加到对应位置的列表中
            span_dict[match.span()].append(_get_numeric_value_from_float(number))

    # 提取所有单词长度为1的文本片段，并处理其中的数字和序数词
    for begin_index, end_index in get_all_spans(text, max_ngram_length=1):
        if (begin_index, end_index) in span_dict:
            continue
        span_text = text[begin_index:end_index]

        number = _parse_number(span_text)
        if number is not None:
            span_dict[begin_index, end_index].append(_get_numeric_value_from_float(number))
        
        # 检查是否为数字词或序数词，并将其添加到对应位置的列表中
        for number, word in enumerate(_NUMBER_WORDS):
            if span_text == word:
                span_dict[begin_index, end_index].append(_get_numeric_value_from_float(float(number)))
                break
        for number, word in enumerate(_ORDINAL_WORDS):
            if span_text == word:
                span_dict[begin_index, end_index].append(_get_numeric_value_from_float(float(number)))
                break

    # 提取所有长度不超过_MAX_DATE_NGRAM_SIZE的文本片段，并解析日期
    for begin_index, end_index in get_all_spans(text, max_ngram_length=_MAX_DATE_NGRAM_SIZE):
        span_text = text[begin_index:end_index]
        date = _parse_date(span_text)
        if date is not None:
            span_dict[begin_index, end_index].append(date)

    # 根据片段长度对结果进行排序，从长到短
    spans = sorted(span_dict.items(), key=lambda span_value: _get_span_length_key(span_value[0]), reverse=True)
    selected_spans = []

    # 选择不重叠的最长片段
    for span, value in spans:
        for selected_span, _ in selected_spans:
            if selected_span[0] <= span[0] and span[1] <= selected_span[1]:
                break
        else:
            selected_spans.append((span, value))

    # 根据起始索引排序选定的片段
    selected_spans.sort(key=lambda span_value: span_value[0][0])

    numeric_value_spans = []
    # 创建NumericValueSpan对象并添加到列表中
    for span, values in selected_spans:
        numeric_value_spans.append(NumericValueSpan(begin_index=span[0], end_index=span[1], values=values))
    return numeric_value_spans
# - https://github.com/google-research/tapas/blob/master/tapas/utils/text_utils.py

# 定义基本的数值类型，可以是 float 或包含可选的三个 float 的元组
_PrimitiveNumericValue = Union[float, Tuple[Optional[float], Optional[float], Optional[float]]]

# 定义排序键的函数类型，接受一个 NumericValue 参数，返回一个元组和省略号的 float
_SortKeyFn = Callable[[NumericValue], Tuple[float, Ellipsis]]

# 日期元组的大小
_DATE_TUPLE_SIZE = 3

# 表示空文本的常量
EMPTY_TEXT = "EMPTY"

# 表示数值类型的字符串常量
NUMBER_TYPE = "number"
# 表示日期类型的字符串常量
DATE_TYPE = "date"


def _get_value_type(numeric_value):
    # 根据 NumericValue 的内容返回其类型字符串
    if numeric_value.float_value is not None:
        return NUMBER_TYPE
    elif numeric_value.date is not None:
        return DATE_TYPE
    # 如果无法识别类型，则抛出异常
    raise ValueError(f"Unknown type: {numeric_value}")


def _get_value_as_primitive_value(numeric_value):
    """Maps a NumericValue proto to a float or tuple of float."""
    # 根据 NumericValue 返回其对应的 float 或者包含三个 float 的元组
    if numeric_value.float_value is not None:
        return numeric_value.float_value
    if numeric_value.date is not None:
        date = numeric_value.date
        value_tuple = [None, None, None]
        # 将日期的各个字段转换为 float，构成一个简单的基本数值
        if date.year is not None:
            value_tuple[0] = float(date.year)
        if date.month is not None:
            value_tuple[1] = float(date.month)
        if date.day is not None:
            value_tuple[2] = float(date.day)
        return tuple(value_tuple)
    # 如果无法识别类型，则抛出异常
    raise ValueError(f"Unknown type: {numeric_value}")


def _get_all_types(numeric_values):
    # 返回所有 NumericValue 中的类型集合
    return {_get_value_type(value) for value in numeric_values}


def get_numeric_sort_key_fn(numeric_values):
    """
    Creates a function that can be used as a sort key or to compare the values. Maps to primitive types and finds the
    biggest common subset. Consider the values "05/05/2010" and "August 2007". With the corresponding primitive values
    (2010.,5.,5.) and (2007.,8., None). These values can be compared by year and date so we map to the sequence (2010.,
    5.), (2007., 8.). If we added a third value "2006" with primitive value (2006., None, None), we could only compare
    by the year so we would map to (2010.,), (2007.,) and (2006.,).

    Args:
     numeric_values: Values to compare

    Returns:
     A function that can be used as a sort key function (mapping numeric values to a comparable tuple)

    Raises:
      ValueError if values don't have a common type or are not comparable.
    """
    value_types = _get_all_types(numeric_values)
    # 如果数值的类型不唯一，则抛出异常
    if len(value_types) != 1:
        raise ValueError(f"No common value type in {numeric_values}")

    value_type = next(iter(value_types))
    if value_type == NUMBER_TYPE:
        # 数字类型的原始值是简单的 float，此处无需处理
        return _get_value_as_primitive_value

    # 此时类型只能是日期，意味着原始类型是一个三元组的 float
    valid_indexes = set(range(_DATE_TUPLE_SIZE))
    # 遍历传入的 numeric_values 列表中的每个数值
    for numeric_value in numeric_values:
        # 调用函数 _get_value_as_primitive_value，获取 numeric_value 的原始值
        value = _get_value_as_primitive_value(numeric_value)
        # 断言 value 是一个元组
        assert isinstance(value, tuple)
        # 遍历元组 value 中的每个元素及其索引
        for tuple_index, inner_value in enumerate(value):
            # 如果 inner_value 是 None，则从 valid_indexes 中移除该索引
            if inner_value is None:
                valid_indexes.discard(tuple_index)

    # 如果 valid_indexes 集合为空集，表示没有共同的有效索引
    if not valid_indexes:
        # 抛出 ValueError 异常，指示 numeric_values 中没有共同的有效值
        raise ValueError(f"No common value in {numeric_values}")

    # 定义一个排序关键字函数 _sort_key_fn，接受 numeric_value 作为参数
    def _sort_key_fn(numeric_value):
        # 获取 numeric_value 的原始值
        value = _get_value_as_primitive_value(numeric_value)
        # 返回一个元组，包含 valid_indexes 中索引位置的值
        return tuple(value[index] for index in valid_indexes)

    # 返回排序关键字函数 _sort_key_fn
    return _sort_key_fn
# 对给定的行索引到数值列表的映射进行数值合并
def _consolidate_numeric_values(row_index_to_values, min_consolidation_fraction, debug_info):
    """
    Finds the most common numeric values in a column and returns them

    Args:
        row_index_to_values:
            每个行索引对应的数值列表。
        min_consolidation_fraction:
            需要进行合并的最小比例。
        debug_info:
            仅用于调试的额外信息。

    Returns:
        每个行索引对应的最常见数值的第一个匹配值。没有匹配值的行将被丢弃。如果无法合并值，则返回空列表。
    """
    # 统计不同类型出现的次数
    type_counts = collections.Counter()
    for numeric_values in row_index_to_values.values():
        type_counts.update(_get_all_types(numeric_values))
    
    if not type_counts:
        return {}

    # 找到出现次数最多的类型
    max_count = max(type_counts.values())
    if max_count < len(row_index_to_values) * min_consolidation_fraction:
        # logging.log_every_n(logging.INFO, f'Can\'t consolidate types: {debug_info} {row_index_to_values} {max_count}', 100)
        return {}

    valid_types = set()
    for value_type, count in type_counts.items():
        if count == max_count:
            valid_types.add(value_type)
    
    # 如果有多个最常见的类型，确保 DATE_TYPE 在其中
    if len(valid_types) > 1:
        assert DATE_TYPE in valid_types
        max_type = DATE_TYPE
    else:
        max_type = next(iter(valid_types))

    # 创建新的行索引到数值的映射
    new_row_index_to_value = {}
    for index, values in row_index_to_values.items():
        # 提取第一个匹配的值
        for value in values:
            if _get_value_type(value) == max_type:
                new_row_index_to_value[index] = value
                break

    return new_row_index_to_value


def _get_numeric_values(text):
    """解析文本并返回其中的数值。"""
    numeric_spans = parse_text(text)
    return itertools.chain(*(span.values for span in numeric_spans))


def _get_column_values(table, col_index):
    """
    解析表格中指定列的文本，并返回一个字典，将行索引映射到数值列表。
    这是原始实现中 number_annotation_utils.py 中的 _get_column_values 函数。

    Args:
      table: Pandas dataframe
      col_index: 整数，指示要获取数值的列的索引
    """
    index_to_values = {}
    for row_index, row in table.iterrows():
        text = normalize_for_match(row[col_index].text)
        index_to_values[row_index] = list(_get_numeric_values(text))
    return index_to_values


def get_numeric_relation(value, other_value, sort_key_fn):
    """比较两个值并返回它们的关系或 None。"""
    value = sort_key_fn(value)
    other_value = sort_key_fn(other_value)
    if value == other_value:
        return Relation.EQ
    if value < other_value:
        return Relation.LT
    if value > other_value:
        return Relation.GT
    return None


def add_numeric_values_to_question(question):
    """向问题中添加数值范围。"""
    # 将原始问题文本保存在变量 original_text 中
    original_text = question
    # 对问题文本进行规范化处理，使其适合匹配操作
    question = normalize_for_match(question)
    # 解析处理后的问题文本，提取其中的数值范围信息
    numeric_spans = parse_text(question)
    # 返回一个 Question 对象，包含原始文本、规范化后文本和数值范围信息
    return Question(original_text=original_text, text=question, numeric_spans=numeric_spans)
def filter_invalid_unicode(text):
    """
    检查并过滤无效的 Unicode 编码。
    
    Args:
        text: 要检查的文本。

    Returns:
        若 'text' 是无效的 Unicode，则返回空字符串和 True；否则返回原文本和 False。
    """
    return ("", True) if isinstance(text, bytes) else (text, False)


def filter_invalid_unicode_from_table(table):
    """
    从表格中移除无效的 Unicode 编码。检查表格单元格文本是否包含无效的 Unicode 编码，
    如果是，则将单元格文本重置为空字符串，并为每个无效的单元格记录警告日志。

    Args:
        table: 要清理的表格。
    """
    # to do: add table id support
    if not hasattr(table, "table_id"):
        table.table_id = 0

    for row_index, row in table.iterrows():
        for col_index, cell in enumerate(row):
            cell, is_invalid = filter_invalid_unicode(cell)
            if is_invalid:
                logging.warning(
                    f"Scrub an invalid table body @ table_id: {table.table_id}, row_index: {row_index}, "
                    f"col_index: {col_index}",
                )
    for col_index, column in enumerate(table.columns):
        column, is_invalid = filter_invalid_unicode(column)
        if is_invalid:
            logging.warning(f"Scrub an invalid table header @ table_id: {table.table_id}, col_index: {col_index}")


def add_numeric_table_values(table, min_consolidation_fraction=0.7, debug_info=None):
    """
    逐列解析表格中的文本，并添加合并后的数值。合并是指查找具有共同类型（日期或数字）的值。

    Args:
        table: 要注释的表格。
        min_consolidation_fraction: 列中需要具有合并值的单元格的分数。
        debug_info: 用于记录日志的附加信息。
    
    Returns:
        添加了数值属性的表格副本。
    """
    table = table.copy()
    # 首先，过滤掉表格中的无效 Unicode
    filter_invalid_unicode_from_table(table)

    # 其次，将单元格值替换为 Cell 对象
    for row_index, row in table.iterrows():
        for col_index, cell in enumerate(row):
            table.iloc[row_index, col_index] = Cell(text=cell)

    # 第三，为这些 Cell 对象添加 numeric_value 属性
    for col_index, column in enumerate(table.columns):
        column_values = _consolidate_numeric_values(
            _get_column_values(table, col_index),
            min_consolidation_fraction=min_consolidation_fraction,
            debug_info=(debug_info, column),
        )

        for row_index, numeric_value in column_values.items():
            table.iloc[row_index, col_index].numeric_value = numeric_value

    return table
```