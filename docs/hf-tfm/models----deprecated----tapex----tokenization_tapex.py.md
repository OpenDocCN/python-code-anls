# `.\models\deprecated\tapex\tokenization_tapex.py`

```py
# 定义编码为utf-8
# 版权信息，版权由Microsoft Research 和 The HuggingFace Inc.团队所有
# 根据Apache许可证2.0版版，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 不提供任何保证或条件，无论是明示还是暗示的
# 查看特定语言控制权限和许可根据许可证下什么样
"""TAPEX的标记化类"""

import json
import os
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import regex as re

# 导入必要的库和模块
from ....file_utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available
from ....tokenization_utils import AddedToken, PreTrainedTokenizer
from ....tokenization_utils_base import ENCODE_KWARGS_DOCSTRING, BatchEncoding, TextInput, TruncationStrategy
from ....utils import logging

# 如果pandas库可用则导入
if is_pandas_available():
    import pandas as pd

# 获取logger记录器
logger = logging.get_logger(__name__)

# 定义vocab文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/tapex-base": "https://huggingface.co/microsoft/tapex-base/resolve/main/vocab.json",
    },
    "merges_file": {
        "microsoft/tapex-base": "https://huggingface.co/microsoft/tapex-base/resolve/main/merges.txt",
    },
}

# 预训练位置编码大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/tapex-base": 512,
}

# 预训练初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/tapex-base": {"do_lower_case": True},
}

# TaExp截断策略
class TapexTruncationStrategy(ExplicitEnum):
    """
    Possible values for the `truncation` argument in [`~TapasTokenizer.__call__`]. Useful for tab-completion in an IDE.
    """

    DROP_ROWS_TO_FIT = "drop_rows_to_fit"


@lru_cache()
# 返回utf-8字节的列表和映射到unicode字符串的函数
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings...
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


# 获取单词的所有可能成对组合
def get_pairs(word):
    # 返回一个单词中的符号对集合。单词被表示为符号元组（符号为可变长度字符串）。
    pairs = set()  # 初始化一个空的符号对集合
    prev_char = word[0]  # 获取单词中的第一个符号
    for char in word[1:]:  # 遍历除了第一个符号以外的所有符号
        pairs.add((prev_char, char))  # 把前一个符号和当前符号组成的符号对加入集合中
        prev_char = char  # 更新前一个符号为当前符号
    return pairs  # 返回符号对集合
class IndexedRowTableLinearize:
    """
    FORMAT: col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...
    """

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        # 确保表内容字典中包含"header"和"rows"键，否则抛出异常
        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE
        # 处理表头
        table_str = self.process_header(table_content["header"]) + " "
        # 处理行
        for i, row_example in enumerate(table_content["rows"]):
            # 注意：行索引应从1开始而不是0
            table_str += self.process_row(row_example, row_index=i + 1) + " "
        # 返回处理后的字符串，去除两侧的空格
        return table_str.strip()

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        # 将表头列表转换为字符串，并添加特殊符号
        return "col : " + " | ".join(headers)

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        row_str = ""
        row_cell_values = []
        for cell_value in row:
            # 如果单元格的值是整数，则转换为字符串
            if isinstance(cell_value, int):
                row_cell_values.append(str(cell_value))
            else:
                row_cell_values.append(cell_value)
        # 将行中的单元格值连接为字符串
        row_str += " | ".join(row_cell_values)
        # 返回带有特殊符号的行字符串，包括行索引
        return "row " + str(row_index) + " : " + row_str


class TapexTokenizer(PreTrainedTokenizer):
    r"""
    Construct a TAPEX tokenizer. Based on byte-level Byte-Pair-Encoding (BPE).

    This tokenizer can be used to flatten one or more table(s) and concatenate them with one or more related sentences
    to be used by TAPEX models. The format that the TAPEX tokenizer creates is the following:

    sentence col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...

    The tokenizer supports a single table + single query, a single table and multiple queries (in which case the table
    will be duplicated for every query), a single query and multiple tables (in which case the query will be duplicated
    for every table), and multiple tables and queries. In other words, you can provide a batch of tables + questions to
    the tokenizer for instance to prepare them for the model.

    Tokenization itself is based on the BPE algorithm. It is identical to the one used by BART, RoBERTa and GPT-2.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # TokenizerConfig 类的构造函数，用于配置 Tokenizer 对象的参数
        Args:
            vocab_file (`str`):
                词汇表文件的路径。
            merges_file (`str`):
                合并文件的路径。
            do_lower_case (`bool`, *optional*, defaults to `True`):
                在进行标记化时是否将输入转换为小写。
            errors (`str`, *optional*, defaults to `"replace"`):
                将字节解码为 UTF-8 时要遵循的范例。更多信息请参见 [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode)。
            bos_token (`str`, *optional*, defaults to `"<s>"`):
                在预训练期间用作序列开头的标记。可用作序列分类器标记。
    
                <Tip>
    
                构建序列时使用的不是此标记作为序列开头的标记。使用的标记是 `cls_token`。
    
                </Tip>
    
            eos_token (`str`, *optional*, defaults to `"</s>"`):
                序列结束的标记。
    
                <Tip>
    
                构建序列时使用的不是此标记作为序列结束的标记。使用的标记是 `sep_token`。
    
                </Tip>
    
            sep_token (`str`, *optional*, defaults to `"</s>"`):
                分隔符标记，在从多个序列构建序列时使用，例如用于序列分类的两个序列或用于文本和问题的问题回答。还用作使用特殊标记构建的序列的最后一个标记。
            cls_token (`str`, *optional*, defaults to `"<s>"`):
                用于进行序列分类（整个序列而不是每个标记的分类）时使用的分类器标记。使用特殊标记构建时序列的第一个标记。
            unk_token (`str`, *optional*, defaults to `"<unk>"`):
                未知标记。不在词汇表中的标记无法转换为 ID，而是设置为此标记。
            pad_token (`str`, *optional*, defaults to `"<pad>"`):
                用于填充的标记，例如在批处理不同长度的序列时。
            mask_token (`str`, *optional*, defaults to `"<mask>"`):
                用于掩码值的标记。训练掩码语言建模时使用此标记。模型将尝试预测此标记。
            add_prefix_space (`bool`, *optional*, defaults to `False`):
                是否在输入前添加初始空格。这允许将第一个单词视为任何其他单词一样处理。（BART 标记器通过前导空格检测单词的开始）。
            max_cell_length (`int`, *optional*, defaults to 15):
                在线性化表格时每个单元格的最大字符数。如果超过此数字，则进行截断。
    # 词汇文件的名称列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇文件映射字典
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型的最大输入尺寸字典
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 预训练模型的初始化配置字典
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化方法
    def __init__(
        self,
        vocab_file,
        merges_file,
        do_lower_case=True,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        max_cell_length=15,
        **kwargs,
        # 如果 bos_token 是字符串类型，则将其转换为 AddedToken 对象，并保留左侧空白字符
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        # 如果 eos_token 是字符串类型，则将其转换为 AddedToken 对象，并保留左侧空白字符
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        # 如果 sep_token 是字符串类型，则将其转换为 AddedToken 对象，并保留左侧空白字符
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        # 如果 cls_token 是字符串类型，则将其转换为 AddedToken 对象，并保留左侧空白字符
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        # 如果 unk_token 是字符串类型，则将其转换为 AddedToken 对象，并保留左侧空白字符
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 如果 pad_token 是字符串类型，则将其转换为 AddedToken 对象，并保留左侧空白字符
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # 如果 mask_token 是字符串类型，则将其转换为 AddedToken 对象，并去除左侧空白字符
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 打开词汇文件，并用 JSON 加载内容到 encoder 字典中
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建 decoder 字典，其键值为 encoder 字典的反转
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置解码错误处理方式
        self.errors = errors
        # 创建字节编码器对象
        self.byte_encoder = bytes_to_unicode()
        # 创建字节解码器对象，其键值为 byte_encoder 字典的反转
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 打开合并文件，并读取内容到 bpe_merges 列表中
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        # 将 bpe_merges 列表中的每个元素按空格分割，并转换为元组，形成 bpe_ranks 字典
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 初始化缓存字典
        self.cache = {}
        # 设置是否在特殊标记前添加前缀空格的属性
        self.add_prefix_space = add_prefix_space
        # 设置是否将所有标记转换为小写的属性
        self.do_lower_case = do_lower_case

        # 创建正则表达式模式，用于对文本进行拆分
        # 正则表达式模式用于匹配缩写、数字、标点符号、空格等
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 调用父类的构造方法，传入参数，并初始化对象的其他属性
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            do_lower_case=do_lower_case,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            max_cell_length=max_cell_length,
            **kwargs,
        )

        # 设置最大单元长度属性
        self.max_cell_length = max_cell_length
        # 创建 IndexedRowTableLinearize 类的对象，并赋值给 table_linearize 属性
        self.table_linearize = IndexedRowTableLinearize()

    # 构建包含特殊标记的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def build_model_inputs(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A TAPEX sequence has the following format:
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
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

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
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ):
        """
        Args:
        Create token type IDs from a pair of sequence for sequence classification tasks. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of integers identifying the token type IDs for the input sequences.
        """
    ) -> List[int]:
        # 创建一个用于序列对分类任务的mask，TAPEX不使用token type ids，因此返回一个值为0的列表
        Args:
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. TAPEX does not:
        make use of token type ids, therefore a list of zeros is returned.
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of zeros.
        """
        # 定义分隔符和类别标识符
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # 若没有传入第二个序列的token ids，则返回只包含第一个序列token ids的mask
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则返回包含两个序列token ids的mask
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # 处理文本用于标记化，根据参数进行空格处理
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        # BPE算法的实现
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        # 对字符串进行标记化处理
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens
    # 将标记（str）转换为id，使用词汇表进行转换
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将id（整数）转换为标记（str），使用词汇表进行转换
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    # 将一系列标记（字符串）转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    # 保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            # 将词汇表以 JSON 格式写入文件
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            # 写入版本信息
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # 如果BPE合并索引不连续，记录警告信息
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    # 添加文档字符串
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPEX_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义一个特殊方法__call__，用于对数据进行处理并返回处理后的结果
    def __call__(
        # 输入参数table，表示输入的数据表格，可以是单个DataFrame或DataFrame列表
        self,
        table: Union["pd.DataFrame", List["pd.DataFrame"]] = None,
        # 输入参数query，表示输入的查询文本，可以是单个文本或文本列表
        query: Optional[Union[TextInput, List[TextInput]]] = None,
        # 输入参数answer，表示输入的答案文本，可以是单个答案或答案列表
        answer: Union[str, List[str]] = None,
        # 是否添加特殊标记到输入数据中，默认为True
        add_special_tokens: bool = True,
        # 是否对输入进行填充处理，默认为False，可以是布尔值、字符串或PaddingStrategy枚举
        padding: Union[bool, str, PaddingStrategy] = False,
        # 是否对输入进行截断处理，默认为None，可以是布尔值、字符串或TruncationStrategy枚举
        truncation: Union[bool, str, TruncationStrategy] = None,
        # 最大长度限制，默认为None
        max_length: Optional[int] = None,
        # 步长参数，默认为0
        stride: int = 0,
        # 填充到的倍数，默认为None
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回token类型ID，默认为None
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码，默认为None
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的token，默认为False
        return_overflowing_tokens: bool = False,
        # 是否返回特殊标记的掩码，默认为False
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射，默认为False
        return_offsets_mapping: bool = False,
        # 是否返回长度信息，默认为False
        return_length: bool = False,
        # 是否显示详细信息，默认为True
        verbose: bool = True,
        # 其他参数
        **kwargs,
        ) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several table-sequence pair(s).

        Args:
            table (`pd.DataFrame`, `List[pd.DataFrame]`):
                Table(s) containing tabular data.
            query (`str` or `List[str]`, *optional*):
                Sentence or batch of sentences related to one or more table(s) to be encoded. Note that the number of
                sentences must match the number of tables.
            answer (`str` or `List[str]`, *optional*):
                Optionally, the corresponding answer to the questions as supervision.
        """

        # If table is not None, call source_call_func with specified arguments
        if table is not None:
            return self.source_call_func(
                table=table,
                query=query,
                answer=answer,
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
        # If answer is not None, call target_call_func with specified arguments
        elif answer is not None:
            return self.target_call_func(
                answer=answer,
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
        else:
            # Raise an error if neither table nor answer is provided
            raise ValueError("You need to provide either a `table` or an `answer`.")
    # 定义一个方法，用于调用源表的函数
    def source_call_func(
        # 表示将要使用的表格数据，可以是单个数据帧或数据帧列表
        table: Union["pd.DataFrame", List["pd.DataFrame"]],
        # 表示查询文本，可以是单个文本输入或文本输入列表，默认为 None
        query: Optional[Union[TextInput, List[TextInput]]] = None,
        # 表示答案文本，可以是单个字符串或字符串列表，默认为 None
        answer: Union[str, List[str]] = None,
        # 是否添加特殊标记，默认为 True
        add_special_tokens: bool = True,
        # 是否填充，默认为 False，可以是布尔值、字符串或填充策略
        padding: Union[bool, str, PaddingStrategy] = False,
        # 是否截断，默认为 None，可以是布尔值、字符串或截断策略
        truncation: Union[bool, str, TruncationStrategy] = None,
        # 最大长度，默认为 None
        max_length: Optional[int] = None,
        # 步幅，默认为 0
        stride: int = 0,
        # 填充到的大小，默认为 None
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，默认为 None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 返回的 token 类型 ID，默认为 None
        return_token_type_ids: Optional[bool] = None,
        # 返回的注意力掩码，默认为 None
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的 token，默认为 False
        return_overflowing_tokens: bool = False,
        # 是否返回特殊标记掩码，默认为 False
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射，默认为 False
        return_offsets_mapping: bool = False,
        # 是否返回长度，默认为 False
        return_length: bool = False,
        # 是否冗长输出，默认为 True
        verbose: bool = True,
        # 其它可选参数
        **kwargs,
        # 定义函数参数和返回类型注释
    ) -> BatchEncoding:
        # 检查输入类型以获得更清晰的错误信息
        valid_table = False # 初始化 table 类型有效标志
        valid_query = False # 初始化 query 类型有效标志

        # 检查 table 是否具有有效类型
        if isinstance(table, pd.DataFrame): # 如果 table 是 pd.DataFrame 类型
            valid_table = True # 标记 table 类型有效
        elif isinstance(table, (list, tuple)) and isinstance(table[0], pd.DataFrame): # 如果 table 是列表或元组，并且第一个元素是 pd.DataFrame 类型
            valid_table = True # 标记 table 类型有效

        # 检查 query 是否具有有效类型
        if query is None or isinstance(query, str): # 如果 query 为 None 或者是字符串类型
            valid_query = True # 标记 query 类型有效
        elif isinstance(query, (list, tuple)): # 如果 query 是列表或元组
            if len(query) == 0 or isinstance(query[0], str): # 如果 query 长度为 0 或第一个元素为字符串类型
                valid_query = True # 标记 query 类型有效

        if not valid_table: # 如果 table 类型不合法
            raise ValueError(
                "table input must of type `pd.DataFrame` (single example), `List[pd.DataFrame]` (batch of examples). "
            ) # 抛出数值错误类型异常
        if not valid_query: # 如果 query 类型不合法
            raise ValueError("query input must of type `str` (single example), `List[str]` (batch of examples). ") # 抛出数值错误类型异常
        is_batched = isinstance(table, (list, tuple)) or isinstance(query, (list, tuple)) # 判断是否批量输入

        if is_batched: # 如果是批量输入
            return self.batch_encode_plus(
                table=table,
                query=query,
                answer=answer,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
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
        else: # 如果不是批量输入
            return self.encode_plus(
                table=table,
                query=query,
                answer=answer,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
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

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPEX_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 批量编码输入数据并添加特殊标记
    # table: 数据表格，可以是单个数据帧或数据帧列表
    # query: 可选的文本输入列表
    # answer: 答案列表
    # add_special_tokens: 是否添加特殊标记
    # padding: 填充策略，可以是布尔值、字符串或填充策略对象
    # truncation: 截断策略，可以是布尔值、字符串或截断策略对象
    # max_length: 最大长度
    # pad_to_multiple_of: 填充到指定长度的倍数
    # return_tensors: 返回张量类型
    # return_token_type_ids: 是否返回token类型ID
    # return_attention_mask: 是否返回attention掩码
    # return_overflowing_tokens: 是否返回溢出的token
    # return_special_tokens_mask: 是否返回特殊标记掩码
    # return_offsets_mapping: 是否返回偏移映射
    # return_length: 是否返回长度
    # verbose: 是否打印详细信息
    def batch_encode_plus(
        self,
        table: Union["pd.DataFrame", List["pd.DataFrame"]],
        query: Optional[List[TextInput]] = None,
        answer: List[str] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str] = None,
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
    ) -> BatchEncoding:
        """
        <Tip warning={true}>

        This method is deprecated, `__call__` should be used instead.

        </Tip>
        """
        # 针对'truncation_strategy'、'pad_to_max_length'的后向兼容性处理
        # 获取填充和截断策略
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 执行批量编码操作
        return self._batch_encode_plus(
            table=table,
            query=query,
            answer=answer,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
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
    # 定义一个批量编码函数，接受表格、查询文本和答案等参数，返回批处理后的编码结果
    def _batch_encode_plus(
        self,
        table: Union["pd.DataFrame", List["pd.DataFrame"]],
        query: Optional[List[TextInput]] = None,
        answer: Optional[List[str]] = None,
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
        # 如果要返回偏移映射，但当前使用 Python tokenizers 不支持
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )
        
        # 处理单个表格，多个查询的情况
        if isinstance(table, pd.DataFrame) and isinstance(query, (list, tuple)):
            table = [table] * len(query)
        
        # 处理多个表格，单个查询的情况
        if isinstance(table, (list, tuple)) and isinstance(query, str):
            query = [query] * len(table)
        
        # 进行模型准备前的批处理
        batch_outputs = self._batch_prepare_for_model(
            table=table,
            query=query,
            answer=answer,
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
        
        # 返回经编码后的批处理结果
        return BatchEncoding(batch_outputs)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPEX_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 为模型准备批量输入数据
    def _batch_prepare_for_model(
        # 输入数据表格，可以是单个 DataFrame 或 DataFrame 列表
        self,
        table: Union["pd.DataFrame", List["pd.DataFrame"]],
        # 查询文本，可以是单个文本或文本列表，可选参数
        query: Optional[Union[TextInput, List[TextInput]]] = None,
        # 答案，可以是单个答案字符串或答案字符串列表，可选参数
        answer: Optional[Union[str, List[str]]] = None,
        # 是否添加特殊标记，默认为 True
        add_special_tokens: bool = True,
        # 填充策略，默认为不填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 截断策略，默认为不截断
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        # 最大长度限制，默认为无限制
        max_length: Optional[int] = None,
        # 步幅，默认为 0
        stride: int = 0,
        # 填充到的长度的倍数，可选参数
        pad_to_multiple_of: Optional[int] = None,
        # 返回张量类型，可选参数
        return_tensors: Optional[str] = None,
        # 是否返回 token 类型 ID
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的 token
        return_overflowing_tokens: bool = False,
        # 是否返回特殊 token 掩码
        return_special_tokens_mask: bool = False,
        # 是否返回长度信息
        return_length: bool = False,
        # 是否显示详细信息，默认为 True
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        This method adds special tokens, truncates sequences if overflowing while taking into account the special
        tokens and manages a moving window (with user defined stride) for overflowing tokens.
        """
        # 初始化批处理输出字典
        batch_outputs = {}
        # 如果没有提供答案，用 None 填充答案列表
        if answer is None:
            answer = [None] * len(table)
        # 遍历表、查询和答案列表的元素
        for _table, _query, _answer in zip(table, query, answer):
            # 准备表格、查询和答案的文本，处理截断策略和最大长度
            text = self.prepare_table_query(
                _table, _query, _answer, truncation_strategy=truncation_strategy, max_length=max_length
            )

            # 如果设置了小写，将文本转换为小写
            if self.do_lower_case:
                text = text.lower()

            # 对文本进行分词
            tokens = self.tokenize(text)
            # 准备模型输入，将分词转换为 ID，并处理特殊标记、截断、填充等
            outputs = self.prepare_for_model(
                ids=self.convert_tokens_to_ids(tokens),
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # 我们之后在批处理中填充
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # 我们之后在批处理中填充
                return_attention_mask=False,  # 我们之后在批处理中填充
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # 我们最后将整个批次转换为张量
                prepend_batch_axis=False,
                verbose=verbose,
            )

            # 将输出添加到批处理输出字典中
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # 在批处理中填充输出
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # 创建批处理编码对象
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        # 返回批处理编码对象
        return batch_outputs

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING)
    def encode(
        self,
        table: "pd.DataFrame",
        query: Optional[TextInput] = None,
        answer: Optional[str] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, TapexTruncationStrategy] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
```  
    # 定义一个方法，用于准备模型的输入数据，不包括 token type IDs、attention masks 等在内的处理。如果想要自己构建处理逻辑，则使用该方法；否则，请参考 `__call__` 方法。
    def prepare_input(self, table: "pd.DataFrame", query: Optional[TextInput] = None, answer: Optional[str] = None, add_special_tokens: bool = True, padding: Union[bool, str, PaddingStrategy] = False, truncation: Union[bool, str] = None, max_length: Optional[int] = None, return_tensors: Optional[Union[str, TensorType]] = None, **kwargs) -> List[int]:
        # 使用 encode_plus 方法处理输入内容，获得编码后的输入
        encoded_inputs = self.encode_plus(
            table,
            query=query,
            answer=answer,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs,
        )
        # 返回编码后的输入的 input_ids
        return encoded_inputs["input_ids"]

    # 添加额外文档，包括 ENCODE_KWARGS_DOCSTRING 和 TAPEX_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPEX_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义 encode_plus 方法，用于编码输入内容
    def encode_plus(
        self,
        table: "pd.DataFrame",
        query: Optional[TextInput] = None,
        answer: Optional[str] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str] = None,
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
        # 对 'truncation_strategy', 'pad_to_max_length' 进行后向兼容处理
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )
        # 返回经过 _encode_plus 方法处理后的结果
        return self._encode_plus(
            table=table,
            query=query,
            answer=answer,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
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
    def _encode_plus(
        self,
        table: "pd.DataFrame",  # 表格数据，DataFrame 类型
        query: Optional[TextInput] = None,  # 查询文本，可选，默认为空
        answer: Optional[str] = None,  # 答案文本，可选，默认为空
        add_special_tokens: bool = True,  # 是否添加特殊标记，默认为 True
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 最大长度限制，可选，默认为空
        stride: int = 0,  # 步长，默认为 0
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定长度的倍数，可选，默认为空
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量类型，可选，默认为空
        return_token_type_ids: Optional[bool] = None,  # 是否返回 token 类型 ID，可选，默认为空
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码，可选，默认为空
        return_overflowing_tokens: bool = False,  # 是否返回溢出的 token，默认为 False
        return_special_tokens_mask: bool = False,  # 是否返回特殊 token 掩码，默认为 False
        return_offsets_mapping: bool = False,  # 是否返回 offset 映射，默认为 False
        return_length: bool = False,  # 是否返回长度信息，默认为 False
        verbose: bool = True,  # 是否显示详细信息，默认为 True
        **kwargs,  # 其它参数，字典形式
    ) -> BatchEncoding:  # 返回 BatchEncoding 对象
        # 如果需要返回 offset 映射，则抛出异常
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # 准备表格查询文本
        text = self.prepare_table_query(
            table, query, answer, truncation_strategy=truncation_strategy, max_length=max_length
        )

        # 如果需要，进行小写处理
        if self.do_lower_case:
            text = text.lower()

        # 对文本进行标记化处理
        tokens = self.tokenize(text)

        # 准备模型输入
        return self.prepare_for_model(
            ids=self.convert_tokens_to_ids(tokens),
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
    def target_call_func(
        self,
        answer: Union[str, List[str]],  # 接受字符串或字符串列表作为参数
        add_special_tokens: bool = True,  # 是否添加特殊标记，默认为True
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充参数，默认为False
        truncation: Union[bool, str, TruncationStrategy] = None,  # 截断参数，默认为None
        max_length: Optional[int] = None,  # 最大长度，默认为None
        stride: int = 0,  # 步幅，默认为0
        pad_to_multiple_of: Optional[int] = None,  # 填充到某个倍数，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量类型，默认为None
        return_token_type_ids: Optional[bool] = None,  # 返回token类型ID，默认为None
        return_attention_mask: Optional[bool] = None,  # 返回注意力掩码，默认为None
        return_overflowing_tokens: bool = False,  # 是否返回溢出的token，默认为False
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记的掩码，默认为False
        return_offsets_mapping: bool = False,  # 是否返回偏移映射，默认为False
        return_length: bool = False,  # 是否返回长度，默认为False
        verbose: bool = True,  # 是否详细输出，默认为True
        **kwargs,  # 其他关键字参数
    ) -> BatchEncoding:  # 返回BatchEncoding对象
        """
        The method tokenizes and prepares the answer label for the model.

        Args:
            answer (`str` or `List[str]`):  # 参数，可以是单个字符串或字符串列表，用于训练模型的查询的对应答案监督。
                Corresponding answer supervision to the queries for training the model.
        """
        is_batched = isinstance(answer, (list, tuple))  # 检查是否为批量输入

        if is_batched:  # 如果是批量输入
            return self.target_batch_encode_plus(
                answer=answer,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
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
        else:  # 如果是单个输入
            return self.target_encode_plus(
                answer=answer,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
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
```py  
    def target_batch_encode_plus(
        self,
        answer: List[str],
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str] = None,
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
    ) -> BatchEncoding:
        """
        Prepare answer strings for the model.

        Args:
            answer `List[str]`:
                Corresponding answer supervision to the queries for training the model.
        """
        # 获取填充和截断策略，同时处理过时的参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        return self._target_batch_encode_plus(
            answer=answer,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
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

    def _target_batch_encode_plus(
        self,
        answer: List[str],
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
    # 定义方法target_encode，接收一个字符串参数answer，是否添加特殊标记add_special_tokens，默认为True
    # 是否填充padding，默认为False或者一个PaddingStrategy枚举值，表明不填充
    # 截断策略，默认为None或者一个TruncationStrategy枚举值，或者一个TapexTruncationStrategy对象
    # 最大长度，默认为None
    # 返回张量类型，默认为None
    # 其他关键字参数，将会被传递到target_encode_plus方法中
    def target_encode(
        self,
        answer: str,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy, TapexTruncationStrategy] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> List[int]:
        """
        Prepare the answer string for the model. This method does not return token type IDs, attention masks, etc.
        which are necessary for the model to work correctly. Use this method if you want to build your processing on
        your own, otherwise refer to `__call__`.

        Args:
            answer `str`:
                Corresponding answer supervision to the queries for training the model
        """
        # 调用target_encode_plus方法对答案进行编码，得到编码输出
        encoded_outputs = self.target_encode_plus(
            answer=answer,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

        # 返回编码输出的input_ids字段，作为List[int]类型的结果
        return encoded_outputs["input_ids"]
    def target_encode_plus(
        self,
        answer: str,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str] = None,
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
        Prepare a answer string for the model.

        Args:
            answer `str`:
                Corresponding answer supervision to the queries for training the model.
        """
        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略，以及一些其他参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )
        
        # 返回编码的目标字符串
        return self._target_encode_plus(
            answer=answer,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
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

    def _target_encode_plus(
        self,
        answer: str,
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
    # 定义一个函数，用于将回答文本编码成模型输入的批量编码
    ) -> BatchEncoding:
        # 如果需要返回偏移映射，则抛出NotImplementedError
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # 将回答文本赋值给text变量
        text = answer

        # 如果需要，将文本转换为小写
        if self.do_lower_case:
            text = text.lower()

        # 对文本进行标记化处理
        tokens = self.tokenize(text)

        # 准备模型输入
        return self.prepare_for_model(
            # 将标记转换为ID
            ids=self.convert_tokens_to_ids(tokens),
            add_special_tokens=add_special_tokens,
            # 填充策略
            padding=padding_strategy.value,
            # 截断策略
            truncation=truncation_strategy.value,
            # 最大长度
            max_length=max_length,
            # 步长
            stride=stride,
            # 填充到的倍数
            pad_to_multiple_of=pad_to_multiple_of,
            # 返回张量
            return_tensors=return_tensors,
            # 在张量中添加批次轴
            prepend_batch_axis=True,
            # 返回注意力掩码
            return_attention_mask=return_attention_mask,
            # 返回标记类型ID
            return_token_type_ids=return_token_type_ids,
            # 返回溢出标记
            return_overflowing_tokens=return_overflowing_tokens,
            # 返回特殊标记掩码
            return_special_tokens_mask=return_special_tokens_mask,
            # 返回长度
            return_length=return_length,
            # 详细信息
            verbose=verbose,
        )

    # 准备表格查询
    def prepare_table_query(
        self,
        table,
        query,
        answer=None,
        # 截断策略
        truncation_strategy=Union[str, TruncationStrategy, TapexTruncationStrategy],
        # 最大长度
        max_length=None,
```  
    ):
        """
        This method can be used to linearize a table and add a corresponding query.

        Optionally, it also handles truncation of the table (cells).

        An answer can be provided for more precise truncation.
        """
        # 如果表格不为空
        if not table.empty:
            # 步骤1：创建表格字典
            table_content = {"header": list(table.columns), "rows": [list(row.values) for i, row in table.iterrows()]}

            # 步骤2: 修改表格内部
            # 始终根据self.max_cell_length截断表格单元
            # 如果设置了truncation_strategy，则可以选择截断行
            self.truncate_table_cells(table_content, query, answer)
            if truncation_strategy == TapexTruncationStrategy.DROP_ROWS_TO_FIT:
                self.truncate_table_rows(table_content, query, answer, max_length=max_length)

            # 步骤3: 线性化表格
            linear_table = self.table_linearize.process_table(table_content)
        else:
            linear_table = ""

        # 如果linear_table为空
        if linear_table == "":
            logger.warning(
                "You provide an empty table, or all cells contain much tokens (e.g., >= 1024 tokens). "
                + f"Please carefully check the corresponding table with the query : {query}."
            )
        # 如果query为空
        if query == "":
            logger.warning("You provide nothing to query with respect to the table.")
        # 步骤4: 连接query和linear_table
        separator = " " if query and linear_table else ""
        joint_input = (query + separator + linear_table) if query else linear_table

        return joint_input

    # 截断表格单元
    def truncate_table_cells(self, table_content: Dict, question: str, answer: List):
        # TODO (Qian): is it possible to revert the original cell if it is in the final answer?
        cell_mapping = {}
        # 遍历表格的每一行
        for row in table_content["rows"]:
            # 遍历每个单元
            for i, cell in enumerate(row):
                # 截断单元
                truncate_cell = self.truncate_cell(cell)
                if truncate_cell is not None:
                    # 将原始单元和截断后的单元映射起来
                    cell_mapping[cell] = truncate_cell
                    row[i] = truncate_cell

        # 修改答案列表
        if answer is not None:
            for i, case in enumerate(answer):
                if case in cell_mapping.keys():
                    # 如果答案在映射中，替换成截断后的单元
                    answer[i] = cell_mapping[case]

    # 截断单元
    def truncate_cell(self, cell_value):
        # 不对以下情况进行处理
        if isinstance(cell_value, int) or isinstance(cell_value, float):
            return cell_value
        if cell_value.strip() != "":
            # 将单元值分词
            try_tokens = self.tokenize(cell_value)
            # 如果单元值长度大于等于self.max_cell_length
            if len(try_tokens) >= self.max_cell_length:
                # 保留前self.max_cell_length个词
                retain_tokens = try_tokens[: self.max_cell_length]
                retain_cell_value = self.convert_tokens_to_string(retain_tokens)
                return retain_cell_value
            else:
                return None
        else:
            return cell_value
    def truncate_table_rows(
        self, table_content: Dict, question: str, answer: Optional[Union[str, List[str]]] = None, max_length=None
    ):
        """
        Args:
        table_content:
            {"header": xxx, "rows": xxx, "id" (Optionally): xxx}

        question:
            natural language sentence

        answer:
            if for training, is the supervision; otherwise will be empty
        """
        # 估计删除比例和剩余的词汇长度
        delete_ratio, remain_token_len = self.estimate_delete_ratio(table_content, question, max_length)
        # 随机删除无关的行
        self.delete_unrelated_rows(table_content, question, answer, delete_ratio)
        # 确保结果小于最大长度
        maximum_keep_rows = 0
        for ind, row_example in enumerate(table_content["rows"]):
            value_string = self.table_linearize.process_row(row_example, ind + 1)
            value_token_len = len(self.tokenize(value_string))
            # 超过大小限制，采取行动
            if value_token_len > remain_token_len:
                break
            remain_token_len -= value_token_len
            maximum_keep_rows += 1
        # 删除多余的行
        del table_content["rows"][maximum_keep_rows:]

    def estimate_delete_ratio(self, table_content: Dict, question: str, max_length=None):
        if "header" not in table_content or "rows" not in table_content:
            raise ValueError("The table content should contain both 'header' and 'rows' keys.")
        # 计算标题的标记数，特殊标记只会被预添加到问题中
        question_tokens = self.tokenize(question, add_special_tokens=True)
        # 计算标题的标记数
        header_string = self.table_linearize.process_header(table_content["header"])
        header_tokens = self.tokenize(header_string, add_special_tokens=False)
        # 将所有单元格的值拆分为标记，并查看可以容纳多少个标记
        used_token_len = len(question_tokens) + len(header_tokens)
        # 剩余的标记空间用于行
        remain_token_len = max_length - used_token_len

        value_string = ""
        for _, row_example in enumerate(table_content["rows"]):
            # 使用一个通用索引粗略估算整体标记长度
            value_string += self.table_linearize.process_row(row_example, 100) + " "
        value_token_len = len(self.tokenize(value_string))

        if value_token_len < remain_token_len:
            # 不会删除任何行
            return 0.0, remain_token_len
        else:
            # 计算一个粗略的删除比例
            return 1.0 - remain_token_len / value_token_len, remain_token_len
    def delete_unrelated_rows(self, table_content: Dict, question: str, answer: List, delete_ratio: float):
        """
        The argument answer is used only during training.
        """
        # 初始化用于存储需要删除的非相关行的索引列表
        truncated_unrelated_indices = []
        # 初始化用于存储相关行的索引列表
        related_indices = []
        # 如果答案为空或者不存在，则初始化一个空的答案集合
        if answer is None or len(answer) == 0:
            answer_set = set()
        else:
            # 将答案列表中的每个答案转换为小写并加入答案集合中
            answer_set = {ans_ex.lower() for ans_ex in answer}
        # 将问题关键词加入答案集合中
        if question is not None:
            answer_set.update(question.split())
        # 初始化问题关键词集合
        question_set = set(question.strip("?!.,").split(" "))
        # 获取表格内容中的行数
        row_max_len = len(table_content["rows"])
        # 遍历表格中的每一行
        for _row_idx, row in enumerate(table_content["rows"]):
            # 将当前行中的每个单元格内容转换为小写，并构建成集合
            lower_row = {str(cell).lower() for cell in row}
            # 如果当前行既不包含答案集合中的内容，也不包含问题关键词集合中的内容
            if len(lower_row & answer_set) == 0 and len(lower_row & question_set) == 0:
                # 将当前行的索引加入需要删除的非相关行索引列表中
                truncated_unrelated_indices.append(_row_idx)
            else:
                # 将当前行的前后两行索引加入相关行索引列表中，以保留更多信息
                related_indices.extend([_row_idx - 2, _row_idx - 1, _row_idx, _row_idx + 1, _row_idx + 2])

        # 从需要删除的非相关行索引列表中移除邻近的相关行索引
        truncated_unrelated_indices = [
            _row_idx for _row_idx in truncated_unrelated_indices if _row_idx not in related_indices
        ]
        # 根据删除比例选择要删除的行数
        drop_items = min(len(truncated_unrelated_indices), int(len(table_content["rows"]) * delete_ratio))
        # 随机选择要删除的行索引
        drop_row_indices = random.choices(truncated_unrelated_indices, k=drop_items)

        # 逆序遍历表格的行索引
        for _row_idx in reversed(range(row_max_len)):
            # 如果当前行索引在要删除的行索引列表中，则删除该行
            if _row_idx in drop_row_indices:
                del table_content["rows"][_row_idx]

        # 当删除比例过大时，记录警告日志
        if "id" in table_content and len(drop_row_indices) > 0:
            logger.warning("Delete {:.2f} rows in table {}".format(len(drop_row_indices), table_content["id"]))
```