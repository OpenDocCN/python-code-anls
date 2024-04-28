# `.\transformers\models\tapas\tokenization_tapas.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 2.0 版本许可证使用此文件
# 在不违反许可证的前提下，您不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件是基于“原样”分发
# 没有任何形式的保证或条件，无论是明示的还是暗示的
# 有关特定语言的权限和限制，请参阅许可证


# 导入需要的包
import collections
import datetime
import enum
import itertools
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Generator, List, Optional, Text, Tuple, Union
# 导入 numpy 包
import numpy as np

# 从 tokenization_utils 模块中导入预训练标记化器和一些辅助函数
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    VERY_LARGE_INTEGER,
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
)
# 导入工具包
from ...utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available, logging

# 如果可用，则导入 pandas 包
if is_pandas_available():
    import pandas as pd

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    }
}

# 定义预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {name: 512 for name in PRETRAINED_VOCAB_FILES_MAP.keys()}
# 定义预训练初始化配置
PRETRAINED_INIT_CONFIGURATION = {name: {"do_lower_case": True} for name in PRETRAINED_VOCAB_FILES_MAP.keys()}


# 定义截断策略枚举类
class TapasTruncationStrategy(ExplicitEnum):
    """
    `truncation` 参数的可能值。可用于在 IDE 中的 tab 补全。
    """

    DROP_ROWS_TO_FIT = "drop_rows_to_fit"
    DO_NOT_TRUNCATE = "do_not_truncate"


# 定义表格值的命名元组
TableValue = collections.namedtuple("TokenValue", ["token", "column_id", "row_id"])


# 定义标记坐标数据类
@dataclass(frozen=True)
class TokenCoordinates:
    column_index: int
    row_index: int
    token_index: int


# 定义标记化表格数据类
@dataclass
class TokenizedTable:
    rows: List[List[List[Text]]]
    selected_tokens: List[TokenCoordinates]


# 定义序列化示例类
@dataclass(frozen=True)
class SerializedExample:
    tokens: List[Text]
    column_ids: List[int]
    row_ids: List[int]
    segment_ids: List[int]


# 内部单词片段判断函数
def _is_inner_wordpiece(token: Text):
    return token.startswith("##")


# 加载词汇表
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


# 使用空格进行标记化
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    # 如果文本为空，则返回空列表
    if not text:
        return []
    # 使用空格分割文本，得到一个包含各个词语的列表
    tokens = text.split()
    # 返回词语列表
    return tokens
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
    # 定义类 TapasTokenizer，继承自 PreTrainedTokenizer
    vocab_files_names = VOCAB_FILES_NAMES
    # 设定词汇文件名变量为 VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 设定预训练词汇文件映射为 PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 设定最大模型输入大小为 PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 初始化方法，用于创建一个新的Tokenizer对象
    def __init__(
        self,
        # 词汇表文件名
        vocab_file,
        # 是否将输入文本转换为小写
        do_lower_case=True,
        # 是否进行基本的分词操作
        do_basic_tokenize=True,
        # 永远不要拆分的标记列表
        never_split=None,
        # 未知标记
        unk_token="[UNK]",
        # 分隔标记
        sep_token="[SEP]",
        # 填充标记
        pad_token="[PAD]",
        # 类别标记
        cls_token="[CLS]",
        # 掩盖标记
        mask_token="[MASK]",
        # 空标记
        empty_token="[EMPTY]",
        # 是否对中文字符进行分词
        tokenize_chinese_chars=True,
        # 是否去除文本中的重音符号
        strip_accents=None,
        # 单元格修剪长度
        cell_trim_length: int = -1,
        # 最大列标识
        max_column_id: int = None,
        # 最大行标识
        max_row_id: int = None,
        # 是否去除列名
        strip_column_names: bool = False,
        # 更新答案坐标
        update_answer_coordinates: bool = False,
        # 最小问题长度
        min_question_length=None,
        # 最大问题长度
        max_question_length=None,
        # 模型最大长度
        model_max_length: int = 512,
        # 额外的特殊标记列表
        additional_special_tokens: Optional[List[str]] = None,
        # 其他参数
        **kwargs,
        # 检查是否缺少 pandas 库，如果是则抛出导入错误
        if not is_pandas_available():
            raise ImportError("Pandas is required for the TAPAS tokenizer.")

        # 如果存在额外的特殊标记，则检查空标记是否存在其中，如果不存在则添加
        if additional_special_tokens is not None:
            if empty_token not in additional_special_tokens:
                additional_special_tokens.append(empty_token)
        # 如果不存在额外的特殊标记，则创建包含空标记的列表
        else:
            additional_special_tokens = [empty_token]

        # 如果指定的词汇文件不存在，则抛出数值错误
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表文件，并创建词汇和 id 的有序字典
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 设置是否进行基本分词
        self.do_basic_tokenize = do_basic_tokenize
        # 如果进行基本分词，则创建 BasicTokenizer 对象
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 创建 WordpieceTokenizer 对象
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 设置额外属性
        self.cell_trim_length = cell_trim_length
        # 配置最大列 id
        self.max_column_id = (
            max_column_id
            if max_column_id is not None
            else model_max_length
            if model_max_length is not None
            else VERY_LARGE_INTEGER
        )
        # 配置最大行 id
        self.max_row_id = (
            max_row_id
            if max_row_id is not None
            else model_max_length
            if model_max_length is not None
            else VERY_LARGE_INTEGER
        )
        self.strip_column_names = strip_column_names
        self.update_answer_coordinates = update_answer_coordinates
        self.min_question_length = min_question_length
        self.max_question_length = max_question_length

        super().__init__(
            # 设置基本分词需要的参数
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
    def do_lower_case(self):
        # 返回基本分词器的小写设置
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        # 返回词汇表的大小
        return len(self.vocab)

    def get_vocab(self):
        # 返回词汇表及其对应的编码的字典，包括添加的特殊标记编码
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        # 如果文本格式化后为空，则返回一个包含一个特殊标记的列表
        if format_text(text) == EMPTY_TEXT:
            return [self.additional_special_tokens[0]]
        split_tokens = []
        # 如果进行基本分词
        if self.do_basic_tokenize:
            # 对文本进行基本分词，并避免将所有特殊标记分割
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # 如果 token 在不分割的集合中
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 使用词块分词器对 token 进行分词
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 直接使用词块分词器对文本进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用词汇表将 token 转换为对应的 id，如果 token 不存在则返回未知标记的 id
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用词汇表将 index 转换为对应的 token，如果 index 不存在则返回未知标记
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将一系列 token 转换为单个字符串
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        # 如果保存目录存在
        if os.path.isdir(save_directory):
            # 构造词汇表文件路径
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 如果保存目录不存在，则直接使用给定的路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开词汇表文件，写入词汇表内容
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表中的 token，按照索引排序写入文件
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # 如果索引不连续，记录警告信息
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 将 token 写入文件
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
    def create_attention_mask_from_sequences(self, query_ids: List[int], table_values: List[TableValue]) -> List[int]:
        """
        根据查询标记ID和表值列表创建注意力掩码。

        Args:
            query_ids (`List[int]`): 对应于查询的标记ID列表。
            table_values (`List[TableValue]`): 表值列表，其中包含具有标记值、列ID和行ID的命名元组。

        Returns:
            `List[int]`: 包含注意力掩码值的整数列表。
        """
        # 创建一个全为1的列表，长度为 1 + 查询标记ID列表长度 + 1 + 表值列表长度
        return [1] * (1 + len(query_ids) + 1 + len(table_values))

    def create_segment_token_type_ids_from_sequences(
        self, query_ids: List[int], table_values: List[TableValue]
    ) -> List[int]:
        """
        根据查询标记ID和表值列表创建段标记类型ID。

        Args:
            query_ids (`List[int]`): 对应于查询的标记ID列表。
            table_values (`List[TableValue]`): 表值列表，其中包含具有标记值、列ID和行ID的命名元组。

        Returns:
            `List[int]`: 包含段标记类型ID值的整数列表。
        """
        # 如果表值列表不为空，则提取表值中的表ID
        table_ids = list(zip(*table_values))[0] if table_values else []
        # 创建一个长度为 1 + 查询标记ID列表长度 + 1 的全为0的列表，再加上长度为表值列表长度的全为1的列表
        return [0] * (1 + len(query_ids) + 1) + [1] * len(table_ids)

    def create_column_token_type_ids_from_sequences(
        self, query_ids: List[int], table_values: List[TableValue]
    ) -> List[int]:
        """
        根据查询标记ID和表值列表创建列标记类型ID。

        Args:
            query_ids (`List[int]`): 对应于查询的标记ID列表。
            table_values (`List[TableValue]`): 表值列表，其中包含具有标记值、列ID和行ID的命名元组。

        Returns:
            `List[int]`: 包含列标记类型ID值的整数列表。
        """
        # 如果表值列表不为空，则提取表值中的列ID
        table_column_ids = list(zip(*table_values))[1] if table_values else []
        # 创建一个长度为 1 + 查询标记ID列表长度 + 1 的全为0的列表，再加上表值中的列ID列表
        return [0] * (1 + len(query_ids) + 1) + list(table_column_ids)

    def create_row_token_type_ids_from_sequences(
        self, query_ids: List[int], table_values: List[TableValue]
    ) -> List[int]:
        """
        根据查询标记ID和表值列表创建行标记类型ID。

        Args:
            query_ids (`List[int]`): 对应于查询的标记ID列表。
            table_values (`List[TableValue]`): 表值列表，其中包含具有标记值、列ID和行ID的命名元组。

        Returns:
            `List[int]`: 包含行标记类型ID值的整数列表。
        """
    ) -> List[int]:
        """
        创建行标记类型 ID，根据查询标记 ID 和表值列表。

        Args:
            query_ids (`List[int]`): 对应于 ID 的标记 ID 列表。
            table_values (`List[TableValue]`): 表值列表，其中包含命名元组，包含标记值、列 ID 和该标记的行 ID。

        Returns:
            `List[int]`: 包含行标记类型 ID 值的整数列表。
        """
        table_row_ids = list(zip(*table_values))[2] if table_values else []  # 表值不为空时，提取行 ID 列表
        return [0] * (1 + len(query_ids) + 1) + list(table_row_ids)  # 返回填充 0 的列表，长度是查询标记 ID 长度+2，并添加表值行 ID 列表

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从问题和扁平化表格构建模型输入，用于问题回答或序列分类任务，通过连接和添加特殊标记。

        Args:
            token_ids_0 (`List[int]`): 问题的 ID。
            token_ids_1 (`List[int]`, *optional*): 扁平化表格的 ID。

        Returns:
            `List[int]`: 带有特殊标记的模型输入。
        """
        if token_ids_1 is None:
            raise ValueError("With TAPAS, you must provide both question IDs and table IDs.")  # 如果扁平化表格 ID 为空，则抛出值错误异常

        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1  # 返回连接并添加特殊标记后的模型输入列表

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列 ID。在使用分词器 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                问题 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                扁平化表格 ID 列表。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                标记列表是否已经用于模型的特殊标记格式化。

        Returns:
            `List[int]`: 一个整数列表，范围在 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:  # 如果标记列表已经包含特殊标记
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )  # 调用父类方法获取特殊标记掩码

        if token_ids_1 is not None:  # 如果扁平化表格 ID 不为空
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))  # 返回特殊标记掩码列表
        return [1] + ([0] * len(token_ids_0)) + [1]  # 返回特殊标记掩码列表

    @add_end_docstrings(TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        table: "pd.DataFrame",
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
        answer_coordinates: Optional[Union[List[Tuple], List[List[Tuple]]]] = None,
        answer_text: Optional[Union[List[TextInput], List[List[TextInput]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        max_length: Optional[int] = None,
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
    """Tokenizes and encodes the input data in batch with additional optional arguments.

    Args:
        table (pd.DataFrame): The input data as a pandas DataFrame.
        queries (Optional[Union[List[TextInput], List[PreTokenizedInput], List[EncodedInput]]]): The input queries as a list of text, pre-tokenized, or encoded inputs. Defaults to None.
        answer_coordinates (Optional[Union[List[Tuple], List[List[Tuple]]]]): The coordinates of the answers in the table. Defaults to None.
        answer_text (Optional[Union[List[TextInput], List[List[TextInput]]]]): The text of the answers in the table. Defaults to None.
        add_special_tokens (bool): Whether to add special tokens. Defaults to True.
        padding (str, bool, PaddingStrategy): Padding strategy. Defaults to False.
        truncation (str, bool, TapasTruncationStrategy): Truncation strategy. Defaults to False.
        max_length (Optional[int]): The maximum sequence length. Defaults to None.
        pad_to_multiple_of (Optional[int]): The padding length. Defaults to None.
        return_tensors (Optional[Union[str, TensorType]]): The type of output tensors. Defaults to None.
        return_token_type_ids (Optional[bool]): Whether to return the token type IDs. Defaults to None.
        return_attention_mask (Optional[bool]): Whether to return the attention mask. Defaults to None.
        return_overflowing_tokens (bool): Whether to return overflowing tokens. Defaults to False.
        return_special_tokens_mask (bool): Whether to return the special tokens mask. Defaults to False.
        return_offsets_mapping (bool): Whether to return the offsets mapping. Defaults to False.
        return_length (bool): Whether to return the sequence length. Defaults to False.
        verbose (bool): Whether to print information about the encoding process. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        BatchEncoding: The encoded inputs and additional information.
"""
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def batch_encode_plus(
        self,
        table: "pd.DataFrame",
        queries: Optional[
            Union[
                List[TextInput],
                List[PreTokenizedInput],
                List[EncodedInput],
            ]
        ] = None,
        answer_coordinates: Optional[List[List[Tuple]]] = None,
        answer_text: Optional[List[List[TextInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        max_length: Optional[int] = None,
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
    """Encodes the input data in batch with additional optional arguments.

    Args:
        table (pd.DataFrame): The input data as a pandas DataFrame.
        queries (Optional[Union[List[TextInput], List[PreTokenizedInput], List[EncodedInput]]]): The input queries as a list of text, pre-tokenized, or encoded inputs. Defaults to None.
        answer_coordinates (Optional[List[List[Tuple]]]): The coordinates of the answers in the table. Defaults to None.
        answer_text (Optional[List[List[TextInput]]]): The text of the answers in the table. Defaults to None.
        add_special_tokens (bool): Whether to add special tokens. Defaults to True.
        padding (str, bool, PaddingStrategy): Padding strategy. Defaults to False.
        truncation (str, bool, TapasTruncationStrategy): Truncation strategy. Defaults to False.
        max_length (Optional[int]): The maximum sequence length. Defaults to None.
        pad_to_multiple_of (Optional[int]): The padding length. Defaults to None.
        return_tensors (Optional[Union[str, TensorType]]): The type of output tensors. Defaults to None.
        return_token_type_ids (Optional[bool]): Whether to return the token type IDs. Defaults to None.
        return_attention_mask (Optional[bool]): Whether to return the attention mask. Defaults to None.
        return_overflowing_tokens (bool): Whether to return overflowing tokens. Defaults to False.
        return_special_tokens_mask (bool): Whether to return the special tokens mask. Defaults to False.
        return_offsets_mapping (bool): Whether to return the offsets mapping. Defaults to False.
        return_length (bool): Whether to return the sequence length. Defaults to False.
        verbose (bool): Whether to print information about the encoding process. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        BatchEncoding: The encoded inputs and additional information.
"""
    def _get_question_tokens(self, query):
        """Tokenizes the query, taking into account the max and min question length.

        Args:
            query (str): The query text.

        Returns:
            Tuple[str, List[str]]: The query and the tokenized query.
        """

        query_tokens = self.tokenize(query)
        if self.max_question_length is not None and len(query_tokens) > self.max_question_length:
            logger.warning("Skipping query as its tokens are longer than the max question length")
            return "", []
        if self.min_question_length is not None and len(query_tokens) < self.min_question_length:
            logger.warning("Skipping query as its tokens are shorter than the min question length")
            return "", []

        return query, query_tokens
    # 定义一个方法，用于批量编码输入数据并返回批量编码结果
    def _batch_encode_plus(
        self,
        table,  # 表示输入的表格数据
        queries: Union[  # 表示输入的查询数据，可以是文本输入、预标记化输入或编码输入的列表
            List[TextInput],
            List[PreTokenizedInput],
            List[EncodedInput],
        ],
        answer_coordinates: Optional[List[List[Tuple]]] = None,  # 可选参数，表示答案的坐标
        answer_text: Optional[List[List[TextInput]]] = None,  # 可选参数，表示答案的文本
        add_special_tokens: bool = True,  # 表示是否添加特殊标记
        padding: Union[bool, str, PaddingStrategy] = False,  # 表示是否进行填充，可以是布尔值、字符串或填充策略
        truncation: Union[bool, str, TapasTruncationStrategy] = False,  # 表示是否进行截断，可以是布尔值、字符串或截断策略
        max_length: Optional[int] = None,  # 可选参数，表示最大长度
        pad_to_multiple_of: Optional[int] = None,  # 可选参数，表示填充到的长度
        return_tensors: Optional[Union[str, TensorType]] = None,  # 可选参数，表示返回的张量类型
        return_token_type_ids: Optional[bool] = True,  # 可选参数，表示是否返回token类型ID
        return_attention_mask: Optional[bool] = None,  # 可选参数，表示是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 表示是否返回溢出的标记
        return_special_tokens_mask: bool = False,  # 表示是否返回特殊标记掩码
        return_offsets_mapping: bool = False,  # 表示是否返回偏移映射
        return_length: bool = False,  # 表示是否返回长度
        verbose: bool = True,  # 表示是否显示详细信息
        **kwargs,  # 表示其他未命名的参数
    ) -> BatchEncoding:  # 表示返回值为BatchEncoding类型
        # 对表格数据进行标记化
        table_tokens = self._tokenize_table(table)

        queries_tokens = []
        # 遍历查询数据，获取问题的标记化形式并更新原始查询数据
        for idx, query in enumerate(queries):
            query, query_tokens = self._get_question_tokens(query)
            queries[idx] = query
            queries_tokens.append(query_tokens)

        # 调用_batch_prepare_for_model方法对输入数据进行准备，并返回批处理输出
        batch_outputs = self._batch_prepare_for_model(
            table,
            queries,
            tokenized_table=table_tokens,
            queries_tokens=queries_tokens,
            answer_coordinates=answer_coordinates,
            padding=padding,
            truncation=truncation,
            answer_text=answer_text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
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

        # 返回批量编码结果
        return BatchEncoding(batch_outputs)
    # 准备输入数据以供模型处理的内部函数
    def _batch_prepare_for_model(
        # 原始表格的 DataFrame 格式数据
        raw_table: "pd.DataFrame",
        # 原始查询的格式可以是文本输入、预标记输入或编码输入的列表
        raw_queries: Union[
            List[TextInput],
            List[PreTokenizedInput],
            List[EncodedInput],
        ],
        # 可选的已标记化表格数据
        tokenized_table: Optional[TokenizedTable] = None,
        # 查询的标记化结果
        queries_tokens: Optional[List[List[str]]] = None,
        # 答案的坐标，以列表形式存储的列表
        answer_coordinates: Optional[List[List[Tuple]]] = None,
        # 答案的文本，以列表形式存储的列表
        answer_text: Optional[List[List[TextInput]]] = None,
        # 是否添加特殊 token
        add_special_tokens: bool = True,
        # 填充策略，可以是布尔值、字符串或填充策略对象
        padding: Union[bool, str, PaddingStrategy] = False,
        # 截断策略，可以是布尔值、字符串或截断策略对象
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        # 最大长度限制
        max_length: Optional[int] = None,
        # 将填充长度调整为指定值的倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，可以是字符串或张量类型对象
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回令牌类型 ID
        return_token_type_ids: Optional[bool] = True,
        # 是否返回注意力遮罩
        return_attention_mask: Optional[bool] = True,
        # 是否返回特殊 token 的遮罩
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射
        return_offsets_mapping: bool = False,
        # 是否返回长度信息
        return_length: bool = False,
        # 是否输出详细信息
        verbose: bool = True,
        # 是否在批处理维度前添加新维度
        prepend_batch_axis: bool = False,
        # 其它关键字参数
        **kwargs,
    # 定义函数 encode，接收表格table和查询query等参数，并返回编码后的结果
    def encode(
        self,
        # 表格数据，类型为pd.DataFrame
        table: "pd.DataFrame",
        # 查询数据，类型为文本输入、预分词输入或编码输入类型,默认为None
        query: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                EncodedInput,
            ]
        ] = None,
        # 是否添加特殊标记，默认为True
        add_special_tokens: bool = True,
        # 填充策略，默认为False
        padding: Union[bool, str, PaddingStrategy] = False,
        # 截断策略，默认为False
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        # 最大长度限制，默认为None
        max_length: Optional[int] = None,
        # 返回类型，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 其他关键字参数
        **kwargs,


    # 初始化空字典batch_outputs
    batch_outputs = {}

    # 遍历数据集中的每个例子
    for index, example in enumerate(zip(raw_queries, queries_tokens, answer_coordinates, answer_text)):
        # 解压缩每个例子的数据
        raw_query, query_tokens, answer_coords, answer_txt = example
        # 对原始表格和原始查询准备模型输入
        outputs = self.prepare_for_model(
            raw_table,
            raw_query,
            tokenized_table=tokenized_table,
            query_tokens=query_tokens,
            answer_coordinates=answer_coords,
            answer_text=answer_txt,
            add_special_tokens=add_special_tokens,
            padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterwards
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=None,  # we pad in batch afterwards
            return_attention_mask=False,  # we pad in batch afterwards
            return_token_type_ids=return_token_type_ids,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=None,  # We convert the whole batch to tensors at the end
            prepend_batch_axis=False,
            verbose=verbose,
            prev_answer_coordinates=answer_coordinates[index - 1] if index != 0 else None,
            prev_answer_text=answer_text[index - 1] if index != 0 else None,
        )

        # 遍历模型输出的键值对
        for key, value in outputs.items():
            # 如果键不存在于batch_outputs中，则将其初始化为空列表
            if key not in batch_outputs:
                batch_outputs[key] = []
            # 将值添加到对应键的列表中
            batch_outputs[key].append(value)

    # 对输出进行填充处理
    batch_outputs = self.pad(
        batch_outputs,
        padding=padding,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        return_attention_mask=return_attention_mask,
    )

    # 将填充后的输出转换为BatchEncoding对象
    batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

    # 返回编码结果
    return batch_outputs
    ) -> List[int]:
        """
        Prepare a table and a string for the model. This method does not return token type IDs, attention masks, etc.
        which are necessary for the model to work correctly. Use that method if you want to build your processing on
        your own, otherwise refer to `__call__`.

        Args:
            table (`pd.DataFrame`):
                Table containing tabular data. Note that all cell values must be text. Use *.astype(str)* on a Pandas
                dataframe to convert it to string.
            query (`str` or `List[str]`):
                Question related to a table to be encoded.
        """
        # 调用encode_plus方法将表格和查询编码为模型输入
        encoded_inputs = self.encode_plus(
            table,
            query=query,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

        # 返回编码后的输入中的input_ids部分
        return encoded_inputs["input_ids"]

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(
        self,
        table: "pd.DataFrame",
        query: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                EncodedInput,
            ]
        ] = None,
        answer_coordinates: Optional[List[Tuple]] = None,
        answer_text: Optional[List[TextInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
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
        # 检查设置 return_token_type_ids 为 True 但同时 add_special_tokens 为 False 的情况，抛出错误
        if return_token_type_ids is not None and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        # 检查是否提供了答案，答案坐标和答案文本应同时提供
        if (answer_coordinates and not answer_text) or (not answer_coordinates and answer_text):
            raise ValueError("In case you provide answers, both answer_coordinates and answer_text should be provided")

        # 检查是否传入了额外的参数，此处会抛出错误
        if "is_split_into_words" in kwargs:
            raise NotImplementedError("Currently TapasTokenizer only supports questions as strings.")

        # 当使用 Python tokenizers 时，不支持返回 offset_mapping
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        # 调用 _encode_plus 方法处理输入参数，并返回 BatchEncoding 实例
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
    # 定义内部方法用于编码输入及表格
    def _encode_plus(
        self,
        table: "pd.DataFrame",
        query: Union[
            TextInput,
            PreTokenizedInput,
            EncodedInput,
        ],
        answer_coordinates: Optional[List[Tuple]] = None,  # 可选参数，回答坐标的列表
        answer_text: Optional[List[TextInput]] = None,  # 可选参数，回答文本的列表
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充参数
        truncation: Union[bool, str, TapasTruncationStrategy] = False,  # 截断参数
        max_length: Optional[int] = None,  # 最大长度限制
        pad_to_multiple_of: Optional[int] = None,  # 填充到多个
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量类型
        return_token_type_ids: Optional[bool] = True,  # 是否返回标记类型ID
        return_attention_mask: Optional[bool] = True,  # 是否返回注意力掩码
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回长度
        verbose: bool = True,  # 是否详细输出
        **kwargs,  # 其他关键字参数
    ):
        if query is None:  # 如果查询参数为空
            query = ""  # 将查询参数设置为空字符串
            logger.warning(  # 输出警告信息
                "TAPAS is a question answering model but you have not passed a query. Please be aware that the "
                "model will probably not behave correctly."
            )

        table_tokens = self._tokenize_table(table)  # 使用内部方法对表格进行标记化处理
        query, query_tokens = self._get_question_tokens(query)  # 获取查询的标记化处理结果

        return self.prepare_for_model(  # 调用内部方法用于为模型准备输入
            table,
            query,
            tokenized_table=table_tokens,
            query_tokens=query_tokens,
            answer_coordinates=answer_coordinates,
            answer_text=answer_text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)  # 添加结束文档字符串
    # 准备数据以供模型使用
    def prepare_for_model(
        self,
        raw_table: "pd.DataFrame",  # 原始表格数据，类型为 pandas DataFrame
        raw_query: Union[  # 原始查询数据，可以是文本输入、预分词输入或编码输入
            TextInput,  # 文本输入对象
            PreTokenizedInput,  # 预分词输入对象
            EncodedInput,  # 编码输入对象
        ],
        tokenized_table: Optional[TokenizedTable] = None,  # 可选参数，经过分词处理的表格数据
        query_tokens: Optional[TokenizedTable] = None,  # 可选参数，查询经过分词处理的结果
        answer_coordinates: Optional[List[Tuple]] = None,  # 可选参数，答案在表格中的坐标
        answer_text: Optional[List[TextInput]] = None,  # 可选参数，答案的文本形式
        add_special_tokens: bool = True,  # 是否添加特殊标记，默认为 True
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略，默认为 False
        truncation: Union[bool, str, TapasTruncationStrategy] = False,  # 截断策略，默认为 False
        max_length: Optional[int] = None,  # 最大长度限制，默认为 None
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的长度，默认为 None
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型，默认为 None
        return_token_type_ids: Optional[bool] = True,  # 是否返回 token 类型 ID，默认为 True
        return_attention_mask: Optional[bool] = True,  # 是否返回注意力掩码，默认为 True
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记掩码，默认为 False
        return_offsets_mapping: bool = False,  # 是否返回偏移映射，默认为 False
        return_length: bool = False,  # 是否返回长度，默认为 False
        verbose: bool = True,  # 是否显示详细信息，默认为 True
        prepend_batch_axis: bool = False,  # 是否在返回张量中添加批次维度，默认为 False
        **kwargs,  # 其他关键字参数
    # 获取截断后的表格行数据
    def _get_truncated_table_rows(
        self,
        query_tokens: List[str],  # 查询的分词结果
        tokenized_table: TokenizedTable,  # 经过分词处理的表格数据
        num_rows: int,  # 表格的行数
        num_columns: int,  # 表格的列数
        max_length: int,  # 最大长度限制
        truncation_strategy: Union[str, TapasTruncationStrategy],  # 截断策略
    # 定义一个函数，用于在原地按照指定策略截断一个序列对
    def _truncate_sequence_pair_in_place(
        # 参数query_tokens: 包含tokenized查询的字符串列表
        query_tokens: List[str],
        # 参数tokenized_table: 经过分词处理的表格
        tokenized_table: TokenizedTable,
        # 参数num_rows: 表格的总行数
        num_rows: int,
        # 参数num_columns: 表格的总列数
        num_columns: int,
        # 参数max_length: 总的最大长度
        max_length: int,
        # 参数truncation_strategy: 截断策略，只应在截断时调用此方法，只有 "drop_rows_to_fit" 策略可用
        truncation_strategy: str or [TapasTruncationStrategy]
    ) -> Tuple[int, int]:
        """
        Truncates a sequence pair in-place following the strategy.
        在原地按照指定策略截断一个序列对

        Args:
            query_tokens (`List[str]`):
                List of strings corresponding to the tokenized query.
                与tokenized查询对应的字符串列表
            tokenized_table (`TokenizedTable`):
                Tokenized table
                分词表格
            num_rows (`int`):
                Total number of table rows
                表格的总行数
            num_columns (`int`):
                Total number of table columns
                表格的总列数
            max_length (`int`):
                Total maximum length.
                总的最大长度
            truncation_strategy (`str` or [`TapasTruncationStrategy`]):
                Truncation strategy to use. Seeing as this method should only be called when truncating, the only
                available strategy is the `"drop_rows_to_fit"` strategy.
                要使用的截断策略。由于只有在截断时才应调用此方法，因此唯一可用的策略是 "drop_rows_to_fit" 策略。

        Returns:
            `Tuple(int, int)`: tuple containing the number of rows after truncation, and the number of tokens available
            for each table element.
            包含截断后行数和每个表格元素可用的标记数的元组。
        """
        if not isinstance(truncation_strategy, TapasTruncationStrategy):
            truncation_strategy = TapasTruncationStrategy(truncation_strategy)

        if max_length is None:
            max_length = self.model_max_length

        if truncation_strategy == TapasTruncationStrategy.DROP_ROWS_TO_FIT:
            while True:
                num_tokens = self._get_max_num_tokens(
                    query_tokens, tokenized_table, num_rows=num_rows, num_columns=num_columns, max_length=max_length
                )

                if num_tokens is not None:
                    # We could fit the table.
                    # 我们可以适应表格。
                    break

                # Try to drop a row to fit the table.
                # 尝试删除一行以适应表格。
                num_rows -= 1

                if num_rows < 1:
                    break
        elif truncation_strategy != TapasTruncationStrategy.DO_NOT_TRUNCATE:
            raise ValueError(f"Unknown truncation strategy {truncation_strategy}.")

        return num_rows, num_tokens or 1
        # 返回截断后的行数和标记数（如果存在），否则返回1

    def _tokenize_table(
        self,
        table=None,
    ):
        """
        Tokenizes column headers and cell texts of a table.

        Args:
            table (`pd.Dataframe`):
                Table. Returns: `TokenizedTable`: TokenizedTable object.
        """
        tokenized_rows = []
        tokenized_row = []
        # tokenize column headers
        for column in table:
            if self.strip_column_names:
                tokenized_row.append(self.tokenize(""))
            else:
                tokenized_row.append(self.tokenize(column))
        tokenized_rows.append(tokenized_row)

        # tokenize cell values
        for idx, row in table.iterrows():
            tokenized_row = []
            for cell in row:
                tokenized_row.append(self.tokenize(cell))
            tokenized_rows.append(tokenized_row)

        token_coordinates = []
        for row_index, row in enumerate(tokenized_rows):
            for column_index, cell in enumerate(row):
                for token_index, _ in enumerate(cell):
                    token_coordinates.append(
                        TokenCoordinates(
                            row_index=row_index,
                            column_index=column_index,
                            token_index=token_index,
                        )
                    )

        return TokenizedTable(
            rows=tokenized_rows,
            selected_tokens=token_coordinates,
        )

    def _question_encoding_cost(self, question_tokens):
        # Two extra spots of SEP and CLS.
        return len(question_tokens) + 2

    def _get_token_budget(self, question_tokens, max_length=None):
        """
        Computes the number of tokens left for the table after tokenizing a question, taking into account the max
        sequence length of the model.

        Args:
            question_tokens (`List[String]`):
                List of question tokens. Returns: `int`: the number of tokens left for the table, given the model max
                length.
        """
        return (max_length if max_length is not None else self.model_max_length) - self._question_encoding_cost(
            question_tokens
        )
    def _get_table_values(self, table, num_columns, num_rows, num_tokens) -> Generator[TableValue, None, None]:
        """Iterates over partial table and returns token, column and row indexes."""
        # 遍历部分表格，并返回标记、列和行索引
        for tc in table.selected_tokens:
            # 第一行是标题行
            if tc.row_index >= num_rows + 1:
                continue
            if tc.column_index >= num_columns:
                continue
            cell = table.rows[tc.row_index][tc.column_index]
            token = cell[tc.token_index]
            word_begin_index = tc.token_index
            # 不添加部分单词。找到起始单词片段并检查其是否符合标记预算
            while word_begin_index >= 0 and _is_inner_wordpiece(cell[word_begin_index]):
                word_begin_index -= 1
            if word_begin_index >= num_tokens:
                continue
            yield TableValue(token, tc.column_index + 1, tc.row_index)

    def _get_table_boundaries(self, table):
        """Return maximal number of rows, columns and tokens."""
        # 返回最大行数、列数和标记数
        max_num_tokens = 0
        max_num_columns = 0
        max_num_rows = 0
        for tc in table.selected_tokens:
            max_num_columns = max(max_num_columns, tc.column_index + 1)
            max_num_rows = max(max_num_rows, tc.row_index + 1)
            max_num_tokens = max(max_num_tokens, tc.token_index + 1)
            max_num_columns = min(self.max_column_id, max_num_columns)
            max_num_rows = min(self.max_row_id, max_num_rows)
        return max_num_rows, max_num_columns, max_num_tokens

    def _get_table_cost(self, table, num_columns, num_rows, num_tokens):
        return sum(1 for _ in self._get_table_values(table, num_columns, num_rows, num_tokens))

    def _get_max_num_tokens(self, question_tokens, tokenized_table, num_columns, num_rows, max_length):
        """Computes max number of tokens that can be squeezed into the budget."""
        # 计算可以在预算内挤入的最大标记数
        token_budget = self._get_token_budget(question_tokens, max_length)
        _, _, max_num_tokens = self._get_table_boundaries(tokenized_table)
        if self.cell_trim_length >= 0 and max_num_tokens > self.cell_trim_length:
            max_num_tokens = self.cell_trim_length
        num_tokens = 0
        for num_tokens in range(max_num_tokens + 1):
            cost = self._get_table_cost(tokenized_table, num_columns, num_rows, num_tokens + 1)
            if cost > token_budget:
                break
        if num_tokens < max_num_tokens:
            if self.cell_trim_length >= 0:
                # 如果设置了cell_trim_length，则不允许动态修剪
                return None
            if num_tokens == 0:
                return None
        return num_tokens

    def _get_num_columns(self, table):
        num_columns = table.shape[1]
        if num_columns >= self.max_column_id:
            raise ValueError("Too many columns")
        return num_columns
    # 获取表格的行数，根据参数决定是否删除多余的行
    def _get_num_rows(self, table, drop_rows_to_fit):
        # 获取表格的行数
        num_rows = table.shape[0]
        # 如果行数超过最大行数限制
        if num_rows >= self.max_row_id:
            # 如果需要删除多余的行
            if drop_rows_to_fit:
                # 将行数设置为最大行数减一
                num_rows = self.max_row_id - 1
            else:
                # 否则抛出异常
                raise ValueError("Too many rows")
        # 返回行数
        return num_rows

    # 将文本序列化为索引数组
    def _serialize_text(self, question_tokens):
        """Serializes texts in index arrays."""
        tokens = []
        segment_ids = []
        column_ids = []
        row_ids = []

        # 在开头添加[CLS] token
        tokens.append(self.cls_token)
        segment_ids.append(0)
        column_ids.append(0)
        row_ids.append(0)

        # 遍历问题中的每个token
        for token in question_tokens:
            tokens.append(token)
            segment_ids.append(0)
            column_ids.append(0)
            row_ids.append(0)

        # 返回文本序列化后的tokens, segment_ids, column_ids, row_ids
        return tokens, segment_ids, column_ids, row_ids

    # 序列化表格和文本
    def _serialize(
        self,
        question_tokens,
        table,
        num_columns,
        num_rows,
        num_tokens,
    ):
        """Serializes table and text."""
        # 调用_serialize_text方法，序列化文本
        tokens, segment_ids, column_ids, row_ids = self._serialize_text(question_tokens)

        # 在问题和表格的tokens之间添加[SEP] token
        tokens.append(self.sep_token)
        segment_ids.append(0)
        column_ids.append(0)
        row_ids.append(0)

        # 遍历表格的值，获取每个token的segment_id, column_id, row_id
        for token, column_id, row_id in self._get_table_values(table, num_columns, num_rows, num_tokens):
            tokens.append(token)
            segment_ids.append(1)
            column_ids.append(column_id)
            row_ids.append(row_id)

        # 返回序列化后的表格和文本
        return SerializedExample(
            tokens=tokens,
            segment_ids=segment_ids,
            column_ids=column_ids,
            row_ids=row_ids,
        )

    # 获取表格中某一列的数值
    def _get_column_values(self, table, col_index):
        table_numeric_values = {}
        for row_index, row in table.iterrows():
            cell = row[col_index]
            if cell.numeric_value is not None:
                table_numeric_values[row_index] = cell.numeric_value
        return table_numeric_values

    # 获取单元格token的索引
    def _get_cell_token_indexes(self, column_ids, row_ids, column_id, row_id):
        for index in range(len(column_ids)):
            if column_ids[index] - 1 == column_id and row_ids[index] - 1 == row_id:
                yield index
    def _get_numeric_column_ranks(self, column_ids, row_ids, table):
        """Returns column ranks for all numeric columns."""

        # 创建一个列表，用于存储每个列的排名
        ranks = [0] * len(column_ids)
        # 创建一个列表，用于存储每个列的逆序排名
        inv_ranks = [0] * len(column_ids)

        # 原始代码来自原始实现的 tf_example_utils.py
        # 如果表不为空
        if table is not None:
            # 遍历表的每一列
            for col_index in range(len(table.columns)):
                # 获取表中指定列的所有数值
                table_numeric_values = self._get_column_values(table, col_index)

                # 如果这一列不包含数值，则跳过
                if not table_numeric_values:
                    continue

                # 尝试获取数值排序关键函数
                try:
                    key_fn = get_numeric_sort_key_fn(table_numeric_values.values())
                except ValueError:
                    # 如果无法获取，表示这一列不是数值类型，跳过
                    continue

                # 使用关键函数对列中的数值进行排序
                table_numeric_values = {row_index: key_fn(value) for row_index, value in table_numeric_values.items()}

                # 创建一个逆向键值对，用于将数值映射回原始行索引
                table_numeric_values_inv = collections.defaultdict(list)
                for row_index, value in table_numeric_values.items():
                    table_numeric_values_inv[value].append(row_index)

                # 对唯一数值进行排序
                unique_values = sorted(table_numeric_values_inv.keys())

                # 遍历唯一数值，并为每个数值确定排名
                for rank, value in enumerate(unique_values):
                    for row_index in table_numeric_values_inv[value]:
                        for index in self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index):
                            ranks[index] = rank + 1
                            inv_ranks[index] = len(unique_values) - rank

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
        # 如果表中没有数值，则返回 None
        if not table_numeric_values:
            return None
        # 将表中所有的数值和问题中的数值合并成一个列表
        all_values = list(table_numeric_values.values())
        all_values.append(value)
        # 尝试获取数值排序关键函数
        try:
            return get_numeric_sort_key_fn(all_values)
        except ValueError:
            return None
    # 获取数值关系嵌入

    def _get_numeric_relations(self, question, column_ids, row_ids, table):
        """
        返回数值关系嵌入

        Args:
            question: 问题对象。
            column_ids: 将单词片段位置映射到列id。
            row_ids: 将单词片段位置映射到行id。
            table: 包含数值单元格值的表格。
        """

        # 初始化数值关系列表
        numeric_relations = [0] * len(column_ids)

        # 首先，我们将任何数值值范围添加到问题中：
        # 创建一个字典，将表格单元格映射到它们与问题中任何值的所有关系的集合
        cell_indices_to_relations = collections.defaultdict(set)
        # 如果问题和表格都不为空
        if question is not None and table is not None:
            # 遍历问题中的数值范围
            for numeric_value_span in question.numeric_spans:
                # 遍历每个值
                for value in numeric_value_span.values:
                    # 遍历表格中的列
                    for column_index in range(len(table.columns)):
                        # 获取表格中的数值值
                        table_numeric_values = self._get_column_values(table, column_index)
                        # 获取数值排序关键函数
                        sort_key_fn = self._get_numeric_sort_key_fn(table_numeric_values, value)
                        # 如果排序关键函数为空，则跳过
                        if sort_key_fn is None:
                            continue
                        # 遍历表格中的行和单元格值
                        for row_index, cell_value in table_numeric_values.items():
                            # 获取数值关系
                            relation = get_numeric_relation(value, cell_value, sort_key_fn)
                            # 如果关系不为空
                            if relation is not None:
                                # 将关系添加到字典中
                                cell_indices_to_relations[column_index, row_index].add(relation)

        # 对于每个单元格，为其所有单词片段添加一个特殊特征
        for (column_index, row_index), relations in cell_indices_to_relations.items():
            relation_set_index = 0
            for relation in relations:
                assert relation.value >= Relation.EQ.value
                relation_set_index += 2 ** (relation.value - Relation.EQ.value)
            # 获取单元格的所有单词片段索引
            for cell_token_index in self._get_cell_token_indexes(column_ids, row_ids, column_index, row_index):
                # 在数值关系列表中设置关系索引
                numeric_relations[cell_token_index] = relation_set_index

        # 返回数值关系列表
        return numeric_relations
    def _get_numeric_values(self, table, column_ids, row_ids):
        """Returns numeric values for computation of answer loss."""

        # 初始化具有 NaN 值的数值数组
        numeric_values = [float("nan")] * len(column_ids)

        # 如果表不为空
        if table is not None:
            # 获取表的行数和列数
            num_rows = table.shape[0]
            num_columns = table.shape[1]

            # 遍历每一列
            for col_index in range(num_columns):
                # 遍历每一行
                for row_index in range(num_rows):
                    # 获取表格中的数值对象
                    numeric_value = table.iloc[row_index, col_index].numeric_value
                    # 如果数值对象不为空
                    if numeric_value is not None:
                        # 如果数值对象的浮点值不为空
                        if numeric_value.float_value is not None:
                            float_value = numeric_value.float_value
                            # 如果浮点值不是正无穷大
                            if float_value != float("inf"):
                                # 获取对应单元格的索引，并存储浮点值
                                for index in self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index):
                                    numeric_values[index] = float_value

        return numeric_values

    def _get_numeric_values_scale(self, table, column_ids, row_ids):
        """Returns a scale to each token to down weigh the value of long words."""

        # 初始化具有 1.0 值的数值比例数组
        numeric_values_scale = [1.0] * len(column_ids)

        # 如果表为空，则直接返回数值比例数组
        if table is None:
            return numeric_values_scale

        # 获取表的行数和列数
        num_rows = table.shape[0]
        num_columns = table.shape[1]

        # 遍历每一列
        for col_index in range(num_columns):
            # 遍历每一行
            for row_index in range(num_rows):
                # 获取单元格对应的索引列表
                indices = list(self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index))
                num_indices = len(indices)
                # 如果索引数量大于1
                if num_indices > 1:
                    # 更新每个索引对应的数值比例
                    for index in indices:
                        numeric_values_scale[index] = float(num_indices)

        return numeric_values_scale

    def _pad_to_seq_length(self, inputs):
        # 如果输入的长度大于最大长度，则删除尾部元素
        while len(inputs) > self.model_max_length:
            inputs.pop()
        # ���果输入的长度小于最大长度，则添加零填充
        while len(inputs) < self.model_max_length:
            inputs.append(0)

    def _get_all_answer_ids_from_coordinates(
        self,
        column_ids,
        row_ids,
        answers_list,
    ):
        """Maps lists of answer coordinates to token indexes."""

        # 初始化具有零值的答案 ID 数组
        answer_ids = [0] * len(column_ids)
        found_answers = set()
        all_answers = set()
        # 遍历每个答案坐标列表
        for answers in answers_list:
            column_index, row_index = answers
            all_answers.add((column_index, row_index))
            # 获取每个坐标对应的单元格索引
            for index in self._get_cell_token_indexes(column_ids, row_ids, column_index, row_index):
                found_answers.add((column_index, row_index))
                answer_ids[index] = 1

        # 计算未找到的答案数量
        missing_count = len(all_answers) - len(found_answers)
        return answer_ids, missing_count
    # 获取问题的所有答案坐标对应的标记索引
    def _get_all_answer_ids(self, column_ids, row_ids, answer_coordinates):
        """
        Maps answer coordinates of a question to token indexes.

        In the SQA format (TSV), the coordinates are given as (row, column) tuples. Here, we first swap them to
        (column, row) format before calling _get_all_answer_ids_from_coordinates.
        """

        # 将答案坐标从（行，列）格式转换为（列，行）格式
        def _to_coordinates(answer_coordinates_question):
            return [(coords[1], coords[0]) for coords in answer_coordinates_question]

        # 调用 _get_all_answer_ids_from_coordinates，以获取所有答案的标记索引
        return self._get_all_answer_ids_from_coordinates(
            column_ids, row_ids, answers_list=(_to_coordinates(answer_coordinates))
        )

    # 在文本中查找段落的起始索引
    def _find_tokens(self, text, segment):
        """Return start index of segment in text or None."""
        logging.info(f"text: {text} {segment}")
        for index in range(1 + len(text) - len(segment)):
            for seg_index, seg_token in enumerate(segment):
                if text[index + seg_index].piece != seg_token.piece:
                    break
            else:
                return index
        return None

    # 从答案文本中查找所有出现在表格中的答案坐标
    def _find_answer_coordinates_from_answer_text(
        self,
        tokenized_table,
        answer_text,
    ):
        """Returns all occurrences of answer_text in the table."""
        logging.info(f"answer text: {answer_text}")
        for row_index, row in enumerate(tokenized_table.rows):
            if row_index == 0:
                # 不在表头中搜索答案
                continue
            for col_index, cell in enumerate(row):
                # 如果找到答案文本，返回标记的坐标
                token_index = self._find_tokens(cell, answer_text)
                if token_index is not None:
                    yield TokenCoordinates(
                        row_index=row_index,
                        column_index=col_index,
                        token_index=token_index,
                    )

    # 从答案文本中查找答案的标记索引
    def _find_answer_ids_from_answer_texts(
        self,
        column_ids,
        row_ids,
        tokenized_table,
        answer_texts,
    ):
        """Maps question with answer texts to the first matching token indexes."""
        # 初始化答案索引列表，长度为列标识的数量
        answer_ids = [0] * len(column_ids)
        # 遍历每个答案文本
        for answer_text in answer_texts:
            # 遍历找到的答案文本对应的坐标
            for coordinates in self._find_answer_coordinates_from_answer_text(
                tokenized_table,
                answer_text,
            ):
                # 将答案坐标映射到索引，如果单元格或行已被修剪，则可能失败
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
                # 如果有索引
                if indexes:
                    begin_index = coordinates.token_index + indexes[0]
                    end_index = begin_index + len(answer_text)
                    # 遍历索引，检查是否与答案文本匹配
                    for index in indexes:
                        if index >= begin_index and index < end_index:
                            coordinate_answer_ids.append(index)
                # 如果匹配的索引数量等于答案文本长度，则标记这些索引对应的答案位置为1
                if len(coordinate_answer_ids) == len(answer_text):
                    for index in coordinate_answer_ids:
                        answer_ids[index] = 1
                    break
        return answer_ids

    def _get_answer_ids(self, column_ids, row_ids, answer_coordinates):
        """Maps answer coordinates of a question to token indexes."""
        # 获取所有答案坐标对应的答案索引，以及未找到的数量
        answer_ids, missing_count = self._get_all_answer_ids(column_ids, row_ids, answer_coordinates)

        # 如果有未找到的答案，引发错误
        if missing_count:
            raise ValueError("Couldn't find all answers")
        return answer_ids

    def get_answer_ids(self, column_ids, row_ids, tokenized_table, answer_texts_question, answer_coordinates_question):
        # 如果更新答案坐标为真，则根据答案文本查找答案索引
        if self.update_answer_coordinates:
            return self._find_answer_ids_from_answer_texts(
                column_ids,
                row_ids,
                tokenized_table,
                answer_texts=[self.tokenize(at) for at in answer_texts_question],
            )
        # 否则，直接获取答案索引
        return self._get_answer_ids(column_ids, row_ids, answer_coordinates_question)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ):
        # 填充输入编码的方法，根据给定策略和最大长度进行填充
        # 返回填充后的输入编码
        ...

    def _get_cell_token_probs(self, probabilities, segment_ids, row_ids, column_ids):
        # 遍历概率列表及其对应的段ID、行ID和列ID
        for i, p in enumerate(probabilities):
            segment_id = segment_ids[i]
            col = column_ids[i] - 1
            row = row_ids[i] - 1
            # 如果列和行索引均为正，并且段ID为1（表示单元格内容）
            if col >= 0 and row >= 0 and segment_id == 1:
                # 返回概率索引及其对应的概率值
                yield i, p
    # 计算每个单元格平均概率，按照标记的行列进行聚合
    def _get_mean_cell_probs(self, probabilities, segment_ids, row_ids, column_ids):
        # 建立一个字典，以坐标为键，以概率列表为值
        coords_to_probs = collections.defaultdict(list)
        # 遍历单元格的概率和坐标
        for i, prob in self._get_cell_token_probs(probabilities, segment_ids, row_ids, column_ids):
            # 获取列坐标和行坐标，并将其减1，以适应从0开始的索引
            col = column_ids[i] - 1
            row = row_ids[i] - 1
            # 将概率添加到对应坐标的概率列表中
            coords_to_probs[(col, row)].append(prob)
        # 计算每个单元格概率列表的平均值，并返回一个字典，键为坐标，值为平均概率
        return {coords: np.array(cell_probs).mean() for coords, cell_probs in coords_to_probs.items()}
    
    # 与将逻辑回归转换为预测相关的部分结束
# Copied from transformers.models.bert.tokenization_bert.BasicTokenizer
# BasicTokenizer 类，用于执行基本的分词操作（标点符号拆分、转换为小写等）。
class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
            是否在分词时将输入转换为小写，默认为 True。
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
            在分词过程中永远不会被拆分的令牌集合。仅在 `do_basic_tokenize=True` 时生效。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.
            是否对中文字符进行分词，默认为 True。

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
            这在日语中可能需要停用（参见此[问题](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
            是否删除所有的重音符号。如果未指定此选项，则会根据 `lowercase` 的值来确定（与原始的 BERT 一样）。
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            In some instances we want to skip the basic punctuation splitting so that later tokenization can capture
            the full context of the words, such as contractions.
            在某些情况下，我们希望跳过基本的标点符号拆分，以便后续的分词可以捕获单词的完整上下文，比如缩写等。
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
        # 是否转换为小写
        self.do_lower_case = do_lower_case
        # 在分词时永远不会拆分的令牌集合
        self.never_split = set(never_split)
        # 是否对中文字符进行分词
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 是否删除重音符号
        self.strip_accents = strip_accents
        # 是否在标点符号处进行拆分
        self.do_split_on_punc = do_split_on_punc
    # 对文本进行基本的分词处理。
    # 如果需要对子词进行分词，请使用 WordPieceTokenizer。
    def tokenize(self, text, never_split=None):
        """
        基本的文本分词处理。对于子词的分词，请查看 WordPieceTokenizer。

        Args:
            never_split (`List[str]`, *optional*)
                为了向后兼容而保留。现在直接在基类级别实现（请参见 `PreTrainedTokenizer.tokenize`）不需要分割的标记列表。
        """
        # 使用 union() 将两个集合进行合并，并返回一个新的集合。
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 处理文本内容，清除可能存在的特殊字符等
        text = self._clean_text(text)

        # 2018年11月1日为多语言和中文模型添加了这部分处理。
        # 这现在也应用于英文模型，但是因为英文模型没有在任何中文数据上训练过，
        # 一般不会有中文数据在其中（英文维基百科中存在一些中文词汇，所以词汇中会包含一些中文字符）。
        if self.tokenize_chinese_chars:
            # 对中文字符进行分词处理
            text = self._tokenize_chinese_chars(text)
        # 防止相同字符使用不同的 Unicode 编码被视为不同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 对文本进行按空格分词处理
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        # 遍历分词后的单词
        for token in orig_tokens:
            # 如果单词不在不需要分割的列表中
            if token not in never_split:
                # 如果需要进行小写处理
                if self.do_lower_case:
                    token = token.lower()
                    # 如果需要去除重音符号
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果需要去除重音符号
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 将处理后的单词按标点符号进行分割
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将分词后的单词以空格连接并再次按空格分词得到最终结果
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    # 从文本中去除重音符号
    def _run_strip_accents(self, text):
        """从文本中去除重音符号。"""
        # 将文本中的字符进行 Unicode 规范化处理
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的字符
        for char in text:
            cat = unicodedata.category(char)
            # 如果字符属于 "Mn"（Mark, Nonspacing），表示重音符号
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)
    # 在文本上执行基于标点符号的分词
    def _run_split_on_punc(self, text, never_split=None):
        # 如果不需要在标点符号处分词，或者该文本在不分词的列表中
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            # 返回包含整个文本的列表
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号
            if _is_punctuation(char):
                # 添加一个新分词
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    # 添加一个新分词
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        # 连接分词列表，返回分词后的文本
        return ["".join(x) for x in output]

    # 对中文字符进行分词处理
    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            # 如果是中文字符
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    # 检查字符是否是中文字符
    def _is_chinese_char(self, cp):
        # 检查是否是CJK字符（中日韩字符）
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  # 是CJK字符
            return True
        # 不是CJK字符
        return False

    # 对文本执行无效字符移除和空格清理
    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            # 如果是无效字符或控制字符
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                # 跳过
                continue
            # 如果是空格字符
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 返回处理后的文本
        return "".join(output)
# 从transformers.models.bert.tokenization_bert.WordpieceTokenizer复制而来，定义了WordpieceTokenizer类
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化WordpieceTokenizer类的属性
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        对文本进行WordPiece分词。使用贪婪最长匹配算法根据给定的词汇表进行分词。

        例如，`input = "unaffable"` 将返回 `["un", "##aff", "##able"]`。

        Args:
            text: 单个标记或以空格分隔的标记。这应该已经通过 *BasicTokenizer* 处理过。

        Returns:
            一个wordpiece标记列表。
        """

        output_tokens = []
        # 遍历分词后的每个标记
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            # 使用贪婪算法匹配最长词
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


# 以下：用于TAPAS tokenizer的实用程序（独立于PyTorch/Tensorflow）。
# 这包括解析数字值（日期和数字）的函数，可从表和问题中创建column_ranks、inv_column_ranks、numeric_values、numeric values_scale和numeric_relations，
# 以便在TapasTokenizer的prepare_for_model中使用。
# 这些旨在用于学术设置，在生产用例中应使用Gold mine或Aqua。


# 从原始实现的constants.py中获取，定义了Relation枚举类
# URL: https://github.com/google-research/tapas/blob/master/tapas/utils/constants.py
class Relation(enum.Enum):
    HEADER_TO_CELL = 1  # 连接标题到单元格。
    CELL_TO_HEADER = 2  # 连接单元格到标题。
    QUERY_TO_HEADER = 3  # 连接查询到标题。
    QUERY_TO_CELL = 4  # 连接查询到单元格。
    ROW_TO_CELL = 5  # 连接行到单元格。
    CELL_TO_ROW = 6  # 连接单元格到行。
    EQ = 7  # 注解值与单元格值相同
    # LT用于表示注释值小于单元格值
    LT = 8  # Annotation value is less than cell value
    # GT用于表示注释值大于单元格值
    GT = 9  # Annotation value is greater than cell value
from dataclasses import dataclass
from typing import Optional, List
import collections
import re

# 定义日期类，包含年月日三个可选的整型属性
@dataclass
class Date:
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None

# 定义数值类，包含可选的浮点型数值属性和日期属性
@dataclass
class NumericValue:
    float_value: Optional[float] = None
    date: Optional[Date] = None

# 定义数值段类，包含起始索引、结束索引和数值列表属性
@dataclass
class NumericValueSpan:
    begin_index: int = None
    end_index: int = None
    values: List[NumericValue] = None

# 定义单元格类，包含文本和可选的数值属性
@dataclass
class Cell:
    text: Text
    numeric_value: Optional[NumericValue] = None

# 定义问题类，包含原始问题文本、规范化后的问题文本和可选的数值段列表属性
@dataclass
class Question:
    original_text: Text  # The original raw question string.
    text: Text  # The question string after normalization.
    numeric_spans: Optional[List[NumericValueSpan]] = None

# 定义正则表达式和日期掩码的常量
_DateMask = collections.namedtuple("_DateMask", ["year", "month", "day"])
_YEAR = _DateMask(True, False, False)
_YEAR_MONTH = _DateMask(True, True, False)
_YEAR_MONTH_DAY = _DateMask(True, True, True)
_MONTH = _DateMask(False, True, False)
_MONTH_DAY = _DateMask(False, True, True)

# 定义日期格式和日期掩码的映射关系
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

_FIELD_TO_REGEX = (
    ("%A", r"\w+"),  # Weekday as locale’s full name.
    ("%B", r"\w+"),  # Month as locale’s full name.
    ("%Y", r"\d{4}"),  # Year with century as a decimal number.
    ("%b", r"\w{3}"),  # Month as locale’s abbreviated name.
    ("%d", r"\d{1,2}"),  # Day of the month as a zero-padded decimal number.
    ("%m", r"\d{1,2}"),  # Month as a zero-padded decimal number.
)

# 定义日期模式处理函数，用于计算日期模式的预过滤表达式
def _process_date_pattern(dp):
    pattern, mask = dp
    regex = pattern
    regex = regex.replace(".", re.escape("."))
    regex = regex.replace("-", re.escape("-"))
    regex = regex.replace(" ", r"\s+")
    # 遍历字段和字段正则表达式的元组列表
    for field, field_regex in _FIELD_TO_REGEX:
        # 用字段正则表达式替换正则表达式中的字段
        regex = regex.replace(field, field_regex)
    # 确保没有漏掉任何字段
    assert "%" not in regex, regex
    # 返回模式、掩码和编译后的正则表达式对象
    return pattern, mask, re.compile("^" + regex + "$")
# 处理日期模式并返回为元组
def _process_date_patterns():
    return tuple(_process_date_pattern(dp) for dp in _DATE_PATTERNS)

# 处理过的日期模式
_PROCESSED_DATE_PATTERNS = _process_date_patterns()

# 最大日期ngram大小为5
_MAX_DATE_NGRAM_SIZE = 5

# 定义数字单词列表
_NUMBER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
]

# 定义序数词列表
_ORDINAL_WORDS = [
    "zeroth",
    "first",
    "second",
    "third",
    "fourth",
    "fith",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
]

# 定义序数词后缀列表
_ORDINAL_SUFFIXES = ["st", "nd", "rd", "th"]

# 定义正则表达式匹配数字模式的模式
_NUMBER_PATTERN = re.compile(r"((^|\s)[+-])?((\.\d+)|(\d+(,\d\d\d)*(\.\d*)?))")

# 最小年份和最大年份
_MIN_YEAR = 1700
_MAX_YEAR = 2016

# 无穷大值
_INF = float("INF")


def _get_numeric_value_from_date(date, mask):
    """将日期（datetime Python对象）转换为具有Date对象值的NumericValue对象。"""
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
    """按长度递减和第一个索引递增对span进行排序。"""
    return span[1] - span[0], -span[0]


def _get_numeric_value_from_float(value):
    """将浮点数（Python）转换为具有浮点数值的NumericValue对象。"""
    return NumericValue(float_value=value)


# 不解析序数表达式，例如 '18th of february 1655'。
def _parse_date(text):
    """尝试将文本格式化为标准日期字符串（yyyy-mm-dd）。"""
    text = re.sub(r"Sept\b", "Sep", text)
    for in_pattern, mask, regex in _PROCESSED_DATE_PATTERNS:
        if not regex.match(text):
            continue
        try:
            date = datetime.datetime.strptime(text, in_pattern).date()
        except ValueError:
            continue
        try:
            return _get_numeric_value_from_date(date, mask)
        except ValueError:
            continue
    return None


def _parse_number(text):
    """解析简单的基数和序数数字。"""
    for suffix in _ORDINAL_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    text = text.replace(",", "")
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isnan(value):
        return None
    if value == _INF:
        return None
    return value


def get_all_spans(text, max_ngram_length):
    """
    将文本分割为最多 'max_ngram_length' 的所有可能的ngram。分割点为空格和标点符号。

    Args:
      text: 要拆分的文本。
      max_ngram_length: 最大ngram长度。
    # 生成器函数的注释，说明其返回的内容是Spans，即开始-结束索引的元组
    Yields:
      Spans, tuples of begin-end index.
    """
    # 初始化开始索引列表
    start_indexes = []
    # 遍历文本的字符及其索引
    for index, char in enumerate(text):
        # 如果字符不是字母或数字，跳过当前循环
        if not char.isalnum():
            continue
        # 如果当前字符是字母或数字且前一个字符不是字母或数字，说明当前字符是一个新的单词的开始
        if index == 0 or not text[index - 1].isalnum():
            # 将当前索引加入开始索引列表
            start_indexes.append(index)
        # 如果当前字符是字母或数字且下一个字符不是字母或数字，说明当前字符是一个单词的结束
        if index + 1 == len(text) or not text[index + 1].isalnum():
            # 遍历开始索引列表中的索引，生成对应的开始-结束索引的元组
            for start_index in start_indexes[-max_ngram_length:]:
                yield start_index, index + 1
# 将文本规范化以用于匹配
def normalize_for_match(text):
    # 将文本转小写并用空格连接
    return " ".join(text.lower().split())


# 格式化文本
def format_text(text):
    """将文本转小写并去除标点符号"""
    # 将文本转小写并去除两端空格
    text = text.lower().strip()
    # 如果文本为 "n/a"、"?"或"nan"，则设置为空字符串
    if text == "n/a" or text == "?" or text == "nan":
        text = EMPTY_TEXT
    # 使用正则表达式替换所有非字母数字的字符为空格，并将下划线替换为空格
    text = re.sub(r"[^\w\d]+", " ", text).replace("_", " ")
    # 将文本按空格分割并重新拼接
    text = " ".join(text.split())
    # 去除文本两端空格
    text = text.strip()
    # 如果文本不为空，返回文本，否则返回空字符串
    if text:
        return text
    return EMPTY_TEXT


# 解析文本
def parse_text(text):
    """
    提取文本中最长的数字和日期范围

    参数:
      text: 要进行注释的文本

    返回:
      包含最长数值范围的列表
    """
    # 创建一个默认字典，存储每个范围及其对应的数值
    span_dict = collections.defaultdict(list)
    # 遍历文本中匹配数字模式的部分
    for match in _NUMBER_PATTERN.finditer(text):
        # 获取匹配到的文本
        span_text = text[match.start() : match.end()]
        # 解析数字
        number = _parse_number(span_text)
        # 如果成功解析了数字，将其添加到对应范围的列表中
        if number is not None:
            span_dict[match.span()].append(_get_numeric_value_from_float(number))

    # 遍历文本中的所有范围，最大长度为1
    for begin_index, end_index in get_all_spans(text, max_ngram_length=1):
        # 跳过已经处理过的范围
        if (begin_index, end_index) in span_dict:
            continue
        # 获取范围内的文本
        span_text = text[begin_index:end_index]
        # 解析数字
        number = _parse_number(span_text)
        # 如果成功解析了数字，将其添加到对应范围的列表中
        if number is not None:
            span_dict[begin_index, end_index].append(_get_numeric_value_from_float(number))
        # 检查是否为数字词汇
        for number, word in enumerate(_NUMBER_WORDS):
            if span_text == word:
                span_dict[begin_index, end_index].append(_get_numeric_value_from_float(float(number)))
                break
        # 检查是否为序数词汇
        for number, word in enumerate(_ORDINAL_WORDS):
            if span_text == word:
                span_dict[begin_index, end_index].append(_get_numeric_value_from_float(float(number)))
                break

    # 遍历文本中的所有范围，最大长度为_MAX_DATE_NGRAM_SIZE
    for begin_index, end_index in get_all_spans(text, max_ngram_length=_MAX_DATE_NGRAM_SIZE):
        # 获取范围内的文本
        span_text = text[begin_index:end_index]
        # 解析日期
        date = _parse_date(span_text)
        # 如果成功解析了日期，将其添加到对应范围的列表中
        if date is not None:
            span_dict[begin_index, end_index].append(date)

    # 按范围长度降序排序spans
    spans = sorted(span_dict.items(), key=lambda span_value: _get_span_length_key(span_value[0]), reverse=True)
    # 选择最长的不重叠的spans
    selected_spans = []
    for span, value in spans:
        for selected_span, _ in selected_spans:
            if selected_span[0] <= span[0] and span[1] <= selected_span[1]:
                break
        else:
            selected_spans.append((span, value))

    # 按起始位置排序selected_spans
    selected_spans.sort(key=lambda span_value: span_value[0][0])

    # 构建NumericValueSpan对象列表并返回
    numeric_value_spans = []
    for span, values in selected_spans:
        numeric_value_spans.append(NumericValueSpan(begin_index=span[0], end_index=span[1], values=values))
    return numeric_value_spans
# Import necessary modules and types
_PrimitiveNumericValue = Union[float, Tuple[Optional[float], Optional[float], Optional[float]]]
_SortKeyFn = Callable[[NumericValue], Tuple[float, Ellipsis]]

_DATE_TUPLE_SIZE = 3

# Define constants for empty values
EMPTY_TEXT = "EMPTY"

# Define constants for types
NUMBER_TYPE = "number"
DATE_TYPE = "date"

# Function to determine value type (number or date) based on input NumericValue
def _get_value_type(numeric_value):
    if numeric_value.float_value is not None:
        return NUMBER_TYPE
    elif numeric_value.date is not None:
        return DATE_TYPE
    raise ValueError(f"Unknown type: {numeric_value}")

# Function to map NumericValue to primitive value (float or tuple of float)
def _get_value_as_primitive_value(numeric_value):
    if numeric_value.float_value is not None:
        return numeric_value.float_value
    if numeric_value.date is not None:
        date = numeric_value.date
        value_tuple = [None, None, None]
        if date.year is not None:
            value_tuple[0] = float(date.year)
        if date.month is not None:
            value_tuple[1] = float(date.month)
        if date.day is not None:
            value_tuple[2] = float(date.day)
        return tuple(value_tuple)
    raise ValueError(f"Unknown type: {numeric_value}")

# Function to get all unique value types present in a list of NumericValue
def _get_all_types(numeric_values):
    return {_get_value_type(value) for value in numeric_values}

# Function to create a sort key function for comparing numeric values
def get_numeric_sort_key_fn(numeric_values):
    """
    Creates a function that can be used as a sort key or to compare the values. Maps to primitive types and finds the
    biggest common subset.
    Args:
     numeric_values: Values to compare
    Returns:
     A function that can be used as a sort key function (mapping numeric values to a comparable tuple)
    Raises:
      ValueError if values don't have a common type or are not comparable.
    """
    value_types = _get_all_types(numeric_values)
    if len(value_types) != 1:
        raise ValueError(f"No common value type in {numeric_values}")

    value_type = next(iter(value_types))
    if value_type == NUMBER_TYPE:
        return _get_value_as_primitive_value

    valid_indexes = set(range(_DATE_TUPLE_SIZE))
    # 遍历 numeric_values 列表中的每个元素
    for numeric_value in numeric_values:
        # 将 numeric_value 转换为原始值，并进行断言检查其类型为元组
        value = _get_value_as_primitive_value(numeric_value)
        assert isinstance(value, tuple)
        # 遍历元组中的每个元素
        for tuple_index, inner_value in enumerate(value):
            # 如果 inner_value 为 None，则从 valid_indexes 集合中移除对应的索引
            if inner_value is None:
                valid_indexes.discard(tuple_index)

    # 如果 valid_indexes 为空集，则抛出 ValueError 异常
    if not valid_indexes:
        raise ValueError(f"No common value in {numeric_values}")

    # 定义一个排序对比函数 _sort_key_fn，用于根据 valid_indexes 中的索引顺序对 numeric_value 进行排序，返回元组
    def _sort_key_fn(numeric_value):
        # 将 numeric_value 转换为原始值
        value = _get_value_as_primitive_value(numeric_value)
        # 根据 valid_indexes 中的索引顺序构建一个元组作为排序键
        return tuple(value[index] for index in valid_indexes)

    # 返回排序对比函数 _sort_key_fn
    return _sort_key_fn
# 定义函数，用于合并列中的最常见的数字值并返回
def _consolidate_numeric_values(row_index_to_values, min_consolidation_fraction, debug_info):
    """
    找到列中最常见的数字值并返回

    Args:
        row_index_to_values:
            每一行索引对应所有单元格中的值
        min_consolidation_fraction:
            需要合并值的单元格的比例
        debug_info:
            仅用于日志记录的额外信息

    Returns:
        对于每一行索引，匹配最常见值的第一个值。没有匹配值的行被丢弃。如果值无法合并，则返回空列表。
    """
    # 统计不同类型值的出现频率
    type_counts = collections.Counter()
    for numeric_values in row_index_to_values.values():
        type_counts.update(_get_all_types(numeric_values))
    if not type_counts:
        return {}
    max_count = max(type_counts.values())
    if max_count < len(row_index_to_values) * min_consolidation_fraction:
        # logging.log_every_n(logging.INFO, f'Can\'t consolidate types: {debug_info} {row_index_to_values} {max_count}', 100)
        return {}

    valid_types = set()
    for value_type, count in type_counts.items():
        if count == max_count:
            valid_types.add(value_type)
    if len(valid_types) > 1:
        assert DATE_TYPE in valid_types
        max_type = DATE_TYPE
    else:
        max_type = next(iter(valid_types))

    new_row_index_to_value = {}
    for index, values in row_index_to_values.items():
        # 提取第一个匹配的值
        for value in values:
            if _get_value_type(value) == max_type:
                new_row_index_to_value[index] = value
                break

    return new_row_index_to_value

# 解析文本并返回其中的数字值
def _get_numeric_values(text):
    """解析文本并返回数字值。"""
    numeric_spans = parse_text(text)
    return itertools.chain(*(span.values for span in numeric_spans))

# 解析列中的文本并返回将行索引映射到值的字典
def _get_column_values(table, col_index):
    """
    解析列中的文本并返回将行索引映射到值的字典。这是原始实现中 number_annotation_utils.py 的 _get_column_values 函数

    Args:
      table: Pandas dataframe
      col_index: 整数，指示要获取数值值的列的索引
    """
    index_to_values = {}
    for row_index, row in table.iterrows():
        text = normalize_for_match(row[col_index].text)
        index_to_values[row_index] = list(_get_numeric_values(text))
    return index_to_values

# 比较两个值并返回它们的关系或 None
def get_numeric_relation(value, other_value, sort_key_fn):
    """比较两个值并返回它们之间的关系或 None。"""
    value = sort_key_fn(value)
    other_value = sort_key_fn(other_value)
    if value == other_value:
        return Relation.EQ
    if value < other_value:
        return Relation.LT
    if value > other_value:
        return Relation.GT
    return None

# 向问题中添加数字值范围
def add_numeric_values_to_question(question):
    """向问题中添加数字值范围。"""
    # 将原始问题文本存储在original_text变量中
    original_text = question
    # 对问题文本进行规范化处理，以便进行匹配
    question = normalize_for_match(question)
    # 解析问题文本，提取其中的数值范围
    numeric_spans = parse_text(question)
    # 返回包含原始文本、规范化后文本和数值范围的Question对象
    return Question(original_text=original_text, text=question, numeric_spans=numeric_spans)
def filter_invalid_unicode(text):
    """Return an empty string and True if 'text' is in invalid unicode."""
    # 如果输入的文本是字节流，返回空字符串和 True
    return ("", True) if isinstance(text, bytes) else (text, False)


def filter_invalid_unicode_from_table(table):
    """
    Removes invalid unicode from table. Checks whether a table cell text contains an invalid unicode encoding. If yes,
    reset the table cell text to an empty str and log a warning for each invalid cell

    Args:
        table: table to clean.
    """
    # to do: add table id support
    # 如果表格对象没有 table_id 属性，则设置 table_id 为 0
    if not hasattr(table, "table_id"):
        table.table_id = 0

    # 遍历表格的每一行
    for row_index, row in table.iterrows():
        # 遍历行的每一个单元格
        for col_index, cell in enumerate(row):
            # 调用 filter_invalid_unicode 函数，检查单元格是否包含无效的 Unicode 编码
            cell, is_invalid = filter_invalid_unicode(cell)
            # 如果单元格包含无效编码，记录警告日志，并将单元格文本重置为空字符串
            if is_invalid:
                logging.warning(
                    f"Scrub an invalid table body @ table_id: {table.table_id}, row_index: {row_index}, "
                    f"col_index: {col_index}",
                )
    
    # 遍历表格的每一列
    for col_index, column in enumerate(table.columns):
        # 调用 filter_invalid_unicode 函数，检查列标题是否包含无效的 Unicode 编码
        column, is_invalid = filter_invalid_unicode(column)
        # 如果列标题包含无效编码，记录警告日志
        if is_invalid:
            logging.warning(f"Scrub an invalid table header @ table_id: {table.table_id}, col_index: {col_index}")


def add_numeric_table_values(table, min_consolidation_fraction=0.7, debug_info=None):
    """
    Parses text in table column-wise and adds the consolidated values. Consolidation refers to finding values with a
    common types (date or number)

    Args:
        table:
            Table to annotate.
        min_consolidation_fraction:
            Fraction of cells in a column that need to have consolidated value.
        debug_info:
            Additional information used for logging.
    """
    # 复制表格，确保不对原表格进行修改
    table = table.copy()
    # 首先，过滤表格中的无效 Unicode
    filter_invalid_unicode_from_table(table)

    # 其次，将单元格的文本替换为 Cell 对象
    for row_index, row in table.iterrows():
        for col_index, cell in enumerate(row):
            # 用 Cell 对象替换每个单元格的文本
            table.iloc[row_index, col_index] = Cell(text=cell)

    # 第三，为这些 Cell 对象添加 numeric_value 属性
    for col_index, column in enumerate(table.columns):
        # 获取列值，并对其中的数值进行合并
        column_values = _consolidate_numeric_values(
            _get_column_values(table, col_index),
            min_consolidation_fraction=min_consolidation_fraction,
            debug_info=(debug_info, column),
        )

        # 将合并后的值添加为 numeric_value 属性
        for row_index, numeric_value in column_values.items():
            table.iloc[row_index, col_index].numeric_value = numeric_value

    return table
```