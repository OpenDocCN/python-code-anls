# `.\models\deprecated\tapex\tokenization_tapex.py`

```
# 设置文件编码为 UTF-8
# 版权声明和许可条款
# 该文件受 Apache License, Version 2.0 许可，除非符合许可条款，否则不得使用该文件
# 获取完整的许可条款，请访问 http://www.apache.org/licenses/LICENSE-2.0
# 本软件基于"原样"的基础分发，不提供任何明示或暗示的担保或条件
# 更多细节请参阅许可条款

"""TAPEX 的标记类。"""

import json  # 导入 json 模块
import os  # 导入 os 模块
import random  # 导入 random 模块
from functools import lru_cache  # 从 functools 模块导入 lru_cache 装饰器
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示模块

import regex as re  # 导入 regex 模块作为 re 别名

from ....file_utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available  # 导入文件工具和相关函数
from ....tokenization_utils import AddedToken, PreTrainedTokenizer  # 导入 Tokenizer 相关类和函数
from ....tokenization_utils_base import ENCODE_KWARGS_DOCSTRING, BatchEncoding, TextInput, TruncationStrategy  # 导入 Tokenizer 基础类和相关功能
from ....utils import logging  # 导入日志模块

# 如果可用，导入 pandas 模块
if is_pandas_available():
    import pandas as pd

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/tapex-base": "https://huggingface.co/microsoft/tapex-base/resolve/main/vocab.json",
    },
    "merges_file": {
        "microsoft/tapex-base": "https://huggingface.co/microsoft/tapex-base/resolve/main/merges.txt",
    },
}

# 定义预训练模型的位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/tapex-base": 512,
}

# 定义预训练模型的初始化配置映射
PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/tapex-base": {"do_lower_case": True},
}


class TapexTruncationStrategy(ExplicitEnum):
    """
    [`~TapasTokenizer.__call__`] 的 `truncation` 参数的可能取值。在 IDE 中进行代码补全时非常有用。
    """
    DROP_ROWS_TO_FIT = "drop_rows_to_fit"


@lru_cache()
def bytes_to_unicode():
    """
    返回 utf-8 字节列表及其对应的 unicode 字符映射。我们特别避免映射到空格/控制字符，以免引起 BPE 编码错误。
    可逆的 BPE 编码工作在 unicode 字符串上。这意味着如果要避免 UNK 标记，词汇表中需要大量的 unicode 字符。
    当处理类似于 10B 令牌的数据集时，您大约需要 5K 个字符才能实现良好的覆盖率。这在常规的 32K BPE 词汇表中占据了相当大的比例。
    为了避免这种情况，我们希望在 utf-8 字节和 unicode 字符串之间建立查找表。
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
    返回单词中的符号对集合。单词被表示为符号元组（符号是可变长度的字符串）。
    """
    # 初始化一个空集合，用于存储符号对
    pairs = set()
    # 获取单词的第一个符号作为前一个符号
    prev_char = word[0]
    # 遍历单词中除第一个符号外的所有符号
    for char in word[1:]:
        # 将前一个符号和当前符号作为一个符号对加入到集合中
        pairs.add((prev_char, char))
        # 更新前一个符号为当前符号，为下一次循环做准备
        prev_char = char
    # 返回存储了单词中所有符号对的集合
    return pairs
class IndexedRowTableLinearize:
    """
    FORMAT: col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...
    """

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        # 检查输入的表格内容中是否包含 "header" 和 "rows" 键，如果不包含则触发断言异常
        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE
        # 处理表头，将表头转换成特定格式的字符串
        table_str = self.process_header(table_content["header"]) + " "
        # 处理每一行数据
        for i, row_example in enumerate(table_content["rows"]):
            # 注意：行索引从1开始而不是从0开始
            table_str += self.process_row(row_example, row_index=i + 1) + " "
        # 去除首尾空格并返回处理后的字符串
        return table_str.strip()

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        # 返回格式化后的表头字符串，格式为 "col : col1 | col2 | col 3"
        return "col : " + " | ".join(headers)

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        # 初始化空字符串来存储行的字符串表示
        row_str = ""
        # 初始化列表来存储每个单元格的值
        row_cell_values = []
        # 遍历行中的每个单元格的值
        for cell_value in row:
            # 如果单元格的值是整数，将其转换为字符串后添加到列表中
            if isinstance(cell_value, int):
                row_cell_values.append(str(cell_value))
            else:
                # 否则直接将单元格的值添加到列表中
                row_cell_values.append(cell_value)
        # 将每个单元格的值用 " | " 连接起来，并添加到行字符串表示中
        row_str += " | ".join(row_cell_values)
        # 返回格式化后的行字符串，格式为 "row 1 : val1 | val2 | val3"
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
    # 定义一个函数，用于初始化和配置词汇和特殊标记的设置
    def __init__(
        vocab_file: str,
        merges_file: str,
        do_lower_case: bool = True,
        errors: str = "replace",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        sep_token: str = "</s>",
        cls_token: str = "<s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        add_prefix_space: bool = False,
        max_cell_length: int = 15
    ):
        """
        Args:
            vocab_file (`str`):
                词汇文件的路径。
            merges_file (`str`):
                合并文件的路径。
            do_lower_case (`bool`, *optional*, defaults to `True`):
                在分词时是否将输入转换为小写。
            errors (`str`, *optional*, defaults to `"replace"`):
                解码字节为 UTF-8 时使用的策略。参见 [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) 获取更多信息。
            bos_token (`str`, *optional*, defaults to `"<s>"`):
                在预训练期间用作序列开头的特殊标记。可用作序列分类器的标记。
    
                <Tip>
    
                当使用特殊标记构建序列时，此处的标记并非序列开头的标记。序列开头的标记是 `cls_token`。
    
                </Tip>
    
            eos_token (`str`, *optional*, defaults to `"</s>"`):
                用作序列结尾的特殊标记。
    
                <Tip>
    
                当使用特殊标记构建序列时，此处的标记并非序列结尾的标记。序列结尾的标记是 `sep_token`。
    
                </Tip>
    
            sep_token (`str`, *optional*, defaults to `"</s>"`):
                分隔标记，在构建多序列组合序列时使用，例如用于序列分类或问答任务的文本和问题的组合序列。也用作使用特殊标记构建的序列的最后一个标记。
            cls_token (`str`, *optional*, defaults to `"<s>"`):
                分类器标记，在进行序列分类（整体序列而非逐标记分类）时使用。在使用特殊标记构建序列时是序列的第一个标记。
            unk_token (`str`, *optional*, defaults to `"<unk>"`):
                未知标记。词汇表中不存在的标记将被设置为此标记。
            pad_token (`str`, *optional*, defaults to `"<pad>"`):
                用于填充的标记，例如在批处理不同长度序列时使用。
            mask_token (`str`, *optional*, defaults to `"<mask>"`):
                用于掩码值的标记。在使用掩码语言建模训练模型时使用。模型将尝试预测此标记。
            add_prefix_space (`bool`, *optional*, defaults to `False`):
                是否在输入的开头添加空格。这允许将开头的单词视为任何其他单词。 （BART 分词器通过前导空格检测单词的开始）。
            max_cell_length (`int`, *optional*, defaults to 15):
                线性化表格时每个单元格的最大字符数。如果超过此数字，则进行截断。
        """
    # 从全局常量中获取词汇表文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 从全局常量中获取预训练词汇文件映射表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 从全局常量中获取预训练位置嵌入的最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 从全局常量中获取预训练初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 定义模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化方法，用于创建一个新的实例
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
        ):
        # 如果传入的特殊标记是字符串类型，则将其转换为 AddedToken 对象，保留其前后空格
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # 将 mask_token 转换为 AddedToken 对象，处理左侧空格但不处理右侧空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 使用 UTF-8 编码打开词汇文件，并将其加载为编码器字典
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建解码器字典，键值对颠倒
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # 用于处理解码中的错误
        # 创建字节到 Unicode 字符的编码器
        self.byte_encoder = bytes_to_unicode()
        # 创建字节到 Unicode 字符的解码器，键值对颠倒
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 使用 UTF-8 编码打开合并文件，读取 BPE 合并规则并创建 BPE 排序字典
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space
        self.do_lower_case = do_lower_case

        # 编译正则表达式模式，用于识别和分割单词、数字、标点符号及空格
        # 添加 re.IGNORECASE 以便能够处理首字母大写的缩略词
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 调用父类的初始化方法，传入相关参数
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

        # 设置最大单元长度
        self.max_cell_length = max_cell_length
        # 初始化表格线性化对象
        self.table_linearize = IndexedRowTableLinearize()

    # 构建包含特殊标记的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
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
        # If only one sequence is provided, add `<s>` (CLS) token, sequence tokens, and `</s>` (SEP) token
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        
        # For pairs of sequences, concatenate tokens with appropriate special tokens
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
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
        # If the input tokens already have special tokens, delegate to the superclass method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If only one sequence is provided, mark special tokens at the beginning and end
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        
        # For pairs of sequences, mark special tokens at the beginning and end of each sequence
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs tensor from a list of token ids. This is used for sequence classification tasks where each
        sequence pair gets a different token type ID (0 or 1).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs representing the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs representing the second sequence in a pair.

        Returns:
            `List[int]`: A list of token type IDs where each ID corresponds to a token in the input sequences.
        """
    ) -> List[int]:
        """
        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
        Returns:
            `List[int]`: List of zeros.
        """
        # 定义分隔符和类别标识符的列表
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # 如果没有第二个序列的 token IDs，返回由零组成的列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # 否则返回包含两个序列的 token IDs 的列表，并在适当位置插入分隔符
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        """
        Args:
            text (str): The input text to be tokenized.
            is_split_into_words (bool): Whether the input text is already split into words.
            **kwargs: Additional keyword arguments.
                add_prefix_space (bool): Whether to add a prefix space to the text if necessary.
        Returns:
            tuple: A tuple containing the modified text and remaining keyword arguments.
        """
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        
        # 如果文本已经分成单词或需要添加前缀空格，并且第一个字符不是空白，则在文本前添加空格
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        
        # 返回修改后的文本及其余的关键字参数
        return (text, kwargs)

    @property
    def vocab_size(self):
        # 返回编码器中的词汇表大小
        return len(self.encoder)

    def get_vocab(self):
        # 返回编码器和添加的特殊 token 编码器组成的字典
        return dict(self.encoder, **self.added_tokens_encoder)

    def bpe(self, token):
        """
        Args:
            token (str): The token to apply BPE encoding.
        Returns:
            str: The token after BPE encoding.
        """
        # 如果 token 已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(token)
        pairs = get_pairs(word)

        # 如果没有找到需要合并的 pair，则返回原始 token
        if not pairs:
            return token
        
        while True:
            # 找到在 BPE 词汇表中最小的 bigram
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            
            # 如果找到的 bigram 不在 BPE 词汇表中，则停止合并
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
            
            # 如果合并后的 token 长度为 1，则停止合并
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        
        # 将 tuple 转换为字符串，并将结果缓存起来
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """
        Tokenize a string using Byte-Pair Encoding (BPE).
        
        Args:
            text (str): The input text to tokenize.
        
        Returns:
            List[str]: List of tokens after tokenization.
        """
        bpe_tokens = []
        
        # 使用正则表达式找到文本中的所有符合规则的 token
        for token in re.findall(self.pat, text):
            # 将 token 转换为字节编码，并映射成 unicode 字符串，避免 BPE 中的控制 token（例如空格）
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )
            
            # 对 token 应用 BPE 并分割结果，加入到最终的 token 列表中
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        
        # 返回经 BPE 处理后的 token 列表
        return bpe_tokens
    # 使用词汇表将给定的 token（字符串）转换为对应的 id
    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 使用词汇表将给定的 id（整数）转换为对应的 token（字符串）
    def _convert_id_to_token(self, index):
        return self.decoder.get(index)

    # 将一系列的 token（字符串列表）转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        text = "".join(tokens)
        # 使用 byte_decoder 将 byte 数组转换为 utf-8 编码的字符串
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    # 将词汇表保存到指定目录下的文件中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            # 如果保存目录不存在，则记录错误信息并返回
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构造词汇表文件路径和合并规则文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将 encoder 写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将 BPE merges 写入合并规则文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # 若 BPE 合并索引不连续，则记录警告信息
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    # 附加文档字符串装饰器，用于添加编码方法的额外参数文档说明
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPEX_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义一个特殊方法 __call__，使对象可以像函数一样被调用
    def __call__(
        # 参数 table 可以是单个 pandas DataFrame 或 DataFrame 列表
        self,
        table: Union["pd.DataFrame", List["pd.DataFrame"]] = None,
        # 参数 query 可以是单个文本输入或文本输入列表，可选
        query: Optional[Union[TextInput, List[TextInput]]] = None,
        # 参数 answer 可以是单个答案字符串或答案字符串列表
        answer: Union[str, List[str]] = None,
        # 是否添加特殊标记，默认为 True
        add_special_tokens: bool = True,
        # 填充选项，可以是布尔值、字符串或 PaddingStrategy 枚举
        padding: Union[bool, str, PaddingStrategy] = False,
        # 截断选项，可以是布尔值、字符串或 TruncationStrategy 枚举
        truncation: Union[bool, str, TruncationStrategy] = None,
        # 最大长度限制，可选
        max_length: Optional[int] = None,
        # 滑动窗口的步长，默认为 0
        stride: int = 0,
        # 填充到指定的倍数，默认为 None 不进行填充
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，可选
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回 token_type_ids
        return_token_type_ids: Optional[bool] = None,
        # 是否返回 attention_mask
        return_attention_mask: Optional[bool] = None,
        # 是否返回超出长度的 token
        return_overflowing_tokens: bool = False,
        # 是否返回特殊 token 的掩码
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射
        return_offsets_mapping: bool = False,
        # 是否返回长度信息
        return_length: bool = False,
        # 是否启用详细模式，默认为 True
        verbose: bool = True,
        # 其它可选参数，使用 kwargs 接收
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

        # 如果传入了 table 参数，则调用源调用函数处理
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
        # 如果没有传入 table 参数但传入了 answer 参数，则调用目标调用函数处理
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
        # 如果既没有传入 table 参数也没有传入 answer 参数，则抛出 ValueError 异常
        else:
            raise ValueError("You need to provide either a `table` or an `answer`.")
    # 定义一个方法source_call_func，用于处理文本数据和生成模型输入
    def source_call_func(
        self,
        # 参数table接受一个单个或多个Pandas DataFrame对象作为输入数据表格
        table: Union["pd.DataFrame", List["pd.DataFrame"]],
        # 可选参数query，接受一个单个或多个文本输入或文本输入列表作为查询条件
        query: Optional[Union[TextInput, List[TextInput]]] = None,
        # 参数answer，接受一个单个字符串或字符串列表作为答案
        answer: Union[str, List[str]] = None,
        # 是否添加特殊标记到模型输入中，默认为True
        add_special_tokens: bool = True,
        # 是否进行填充操作，默认为False，或者可以选择填充方式
        padding: Union[bool, str, PaddingStrategy] = False,
        # 是否进行截断操作，默认为None，或者可以选择截断方式
        truncation: Union[bool, str, TruncationStrategy] = None,
        # 最大输入长度限制，默认为None，表示不限制
        max_length: Optional[int] = None,
        # 滑动窗口的步长，默认为0
        stride: int = 0,
        # 是否填充到指定的倍数，默认为None，表示不需要填充到倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，可以指定为字符串或TensorType对象，默认为None
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 是否返回token类型ID，默认为None，表示不返回
        return_token_type_ids: Optional[bool] = None,
        # 是否返回注意力掩码，默认为None，表示不返回
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的token，默认为False，表示不返回
        return_overflowing_tokens: bool = False,
        # 是否返回特殊token掩码，默认为False，表示不返回
        return_special_tokens_mask: bool = False,
        # 是否返回偏移映射，默认为False，表示不返回
        return_offsets_mapping: bool = False,
        # 是否返回序列长度，默认为False，表示不返回
        return_length: bool = False,
        # 是否显示详细信息，默认为True
        verbose: bool = True,
        # 其他可选参数，作为kwargs收集
        **kwargs,
    ) -> BatchEncoding:
        # Input type checking for clearer error

        # Initialize flags for valid input types
        valid_table = False
        valid_query = False

        # Check if the 'table' argument is a pandas DataFrame or a list/tuple of DataFrames
        if isinstance(table, pd.DataFrame):
            valid_table = True
        elif isinstance(table, (list, tuple)) and isinstance(table[0], pd.DataFrame):
            valid_table = True

        # Check if the 'query' argument is None or a string, or a list/tuple of strings
        if query is None or isinstance(query, str):
            valid_query = True
        elif isinstance(query, (list, tuple)):
            if len(query) == 0 or isinstance(query[0], str):
                valid_query = True

        # Raise ValueError if 'table' or 'query' does not match expected types
        if not valid_table:
            raise ValueError(
                "table input must of type `pd.DataFrame` (single example), `List[pd.DataFrame]` (batch of examples). "
            )
        if not valid_query:
            raise ValueError("query input must of type `str` (single example), `List[str]` (batch of examples). ")

        # Determine if batch processing is required based on the types of 'table' or 'query'
        is_batched = isinstance(table, (list, tuple)) or isinstance(query, (list, tuple))

        # If batch processing is required, call 'batch_encode_plus' method
        if is_batched:
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
        else:
            # If not batched, call 'encode_plus' method
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
        # 获取填充和截断策略，以及最大长度，处理参数兼容性问题
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法 `_batch_encode_plus` 进行批量编码
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
    # 定义一个方法 `_batch_encode_plus`，用于对输入的表格数据进行批量编码处理，并返回编码后的结果对象 BatchEncoding
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
        # 如果要求返回偏移映射信息，抛出未实现错误，因为 Python tokenizers 不支持这个功能
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        if isinstance(table, pd.DataFrame) and isinstance(query, (list, tuple)):
            # 单个表格，多个查询的情况下，为每个查询复制表格数据
            table = [table] * len(query)
        if isinstance(table, (list, tuple)) and isinstance(query, str):
            # 多个表格，单个查询的情况下，为每个表格复制相同的查询
            query = [query] * len(table)

        # 调用内部方法 `_batch_prepare_for_model` 准备模型输入数据
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

        # 将处理好的批量输出包装成 BatchEncoding 对象并返回
        return BatchEncoding(batch_outputs)

    # 添加额外的文档字符串，以补充关于编码参数的说明信息
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPEX_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义一个方法 _batch_prepare_for_model，用于为模型批量准备输入数据
    self,
    # 第一个参数 self 是类方法的隐式参数，指向当前实例对象
    table: Union["pd.DataFrame", List["pd.DataFrame"]],
    # 参数 table 可以是单个 pandas DataFrame 或 DataFrame 列表，存储输入数据
    query: Optional[Union[TextInput, List[TextInput]]] = None,
    # 参数 query 是可选的，可以是单个文本输入或文本输入列表，用于查询
    answer: Optional[Union[str, List[str]]] = None,
    # 参数 answer 是可选的，可以是单个答案字符串或答案字符串列表，用于目标答案
    add_special_tokens: bool = True,
    # 参数 add_special_tokens 是一个布尔值，指示是否添加特殊标记
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    # 参数 padding_strategy 指定填充策略，默认为不填充
    truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    # 参数 truncation_strategy 指定截断策略，默认为不截断
    max_length: Optional[int] = None,
    # 参数 max_length 是可选的，指定最大长度限制
    stride: int = 0,
    # 参数 stride 是一个整数，指定滑动窗口的步幅
    pad_to_multiple_of: Optional[int] = None,
    # 参数 pad_to_multiple_of 是可选的，指定填充到的倍数
    return_tensors: Optional[str] = None,
    # 参数 return_tensors 是可选的，指定返回的张量类型
    return_token_type_ids: Optional[bool] = None,
    # 参数 return_token_type_ids 是可选的，指示是否返回 token 类型 ID
    return_attention_mask: Optional[bool] = None,
    # 参数 return_attention_mask 是可选的，指示是否返回注意力掩码
    return_overflowing_tokens: bool = False,
    # 参数 return_overflowing_tokens 是一个布尔值，指示是否返回溢出的 token
    return_special_tokens_mask: bool = False,
    # 参数 return_special_tokens_mask 是一个布尔值，指示是否返回特殊 token 掩码
    return_length: bool = False,
    # 参数 return_length 是一个布尔值，指示是否返回长度信息
    verbose: bool = True,
    # 参数 verbose 是一个布尔值，指示是否输出详细信息
    ) -> BatchEncoding:
        """
        This method adds special tokens, truncates sequences if overflowing while taking into account the special
        tokens and manages a moving window (with user defined stride) for overflowing tokens.
        """
        batch_outputs = {}  # 初始化一个空字典用于存储批处理输出结果
        if answer is None:  # 如果未提供答案，则将其初始化为与表格数量相同的 None 列表
            answer = [None] * len(table)
        for _table, _query, _answer in zip(table, query, answer):
            text = self.prepare_table_query(
                _table, _query, _answer, truncation_strategy=truncation_strategy, max_length=max_length
            )

            if self.do_lower_case:  # 如果指定需要小写化文本，则执行小写化操作
                text = text.lower()

            tokens = self.tokenize(text)  # 对文本进行分词处理，生成 token 列表
            outputs = self.prepare_for_model(
                ids=self.convert_tokens_to_ids(tokens),  # 将 token 转换为对应的 token ID
                add_special_tokens=add_special_tokens,  # 是否添加特殊 token
                padding=PaddingStrategy.DO_NOT_PAD.value,  # 设置不进行填充，批处理中之后进行填充
                truncation=truncation_strategy.value,  # 设置截断策略
                max_length=max_length,  # 设置最大长度
                stride=stride,  # 设置滑动窗口的步长
                pad_to_multiple_of=None,  # 在批处理中进行填充
                return_attention_mask=False,  # 在批处理中进行填充
                return_token_type_ids=return_token_type_ids,  # 返回 token 类型 ID
                return_overflowing_tokens=return_overflowing_tokens,  # 返回溢出的 token
                return_special_tokens_mask=return_special_tokens_mask,  # 返回特殊 token 掩码
                return_length=return_length,  # 返回长度
                return_tensors=None,  # 最终将整个批次转换为张量
                prepend_batch_axis=False,  # 不在输出张量中添加批次轴
                verbose=verbose,  # 是否输出详细信息
            )

            for key, value in outputs.items():  # 将每个输出的值添加到批处理输出字典中
                if key not in batch_outputs:  # 如果键不在批处理输出字典中，则将其初始化为空列表
                    batch_outputs[key] = []
                batch_outputs[key].append(value)  # 将值添加到对应键的列表中

        batch_outputs = self.pad(  # 对批处理输出进行填充处理
            batch_outputs,
            padding=padding_strategy.value,  # 设置填充策略
            max_length=max_length,  # 设置最大长度
            pad_to_multiple_of=pad_to_multiple_of,  # 设置填充到的倍数
            return_attention_mask=return_attention_mask,  # 返回注意力掩码
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)  # 将填充后的批处理输出转换为 BatchEncoding 类型

        return batch_outputs  # 返回批处理输出对象

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
    ) -> List[int]:
        """
        Prepare a table, a string and possible answer for the model. This method does not return token type IDs,
        attention masks, etc. which are necessary for the model to work correctly. Use this method if you want to build
        your processing on your own, otherwise refer to `__call__`.
        """
        # 调用 `encode_plus` 方法，对输入的表格、查询、答案进行编码处理，返回编码后的结果
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

        # 返回编码后的输入序列的 `input_ids`，即标识化后的输入表格数据
        return encoded_inputs["input_ids"]

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPEX_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
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
        # 获取填充和截断策略，并处理与之相关的参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法 `_encode_plus`，进行实际的编码处理，并返回 `BatchEncoding` 对象
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
    # 定义私有方法 `_encode_plus`，用于将表格数据、查询和答案编码为模型输入的批处理编码
    def _encode_plus(
        self,
        table: "pd.DataFrame",
        query: Optional[TextInput] = None,
        answer: Optional[str] = None,
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
        # 如果请求返回偏移映射，则抛出 NotImplementedError
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # 准备表格、查询和答案，根据截断和最大长度策略生成文本
        text = self.prepare_table_query(
            table, query, answer, truncation_strategy=truncation_strategy, max_length=max_length
        )

        # 如果需要，将文本转换为小写
        if self.do_lower_case:
            text = text.lower()

        # 对文本进行分词处理
        tokens = self.tokenize(text)

        # 准备模型输入，包括将词汇转换为 IDs、添加特殊令牌、填充和截断等操作
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
        answer: Union[str, List[str]],
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
        The method tokenizes and prepares the answer label for the model.

        Args:
            answer (`str` or `List[str]`):
                Corresponding answer supervision to the queries for training the model.
        """
        # 检查 `answer` 是否为批量输入（列表或元组）
        is_batched = isinstance(answer, (list, tuple))

        # 如果 `answer` 是批量输入，则调用批量编码方法 `target_batch_encode_plus`
        if is_batched:
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
        # 如果 `answer` 不是批量输入，则调用单条编码方法 `target_encode_plus`
        else:
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
        # 获取填充和截断策略以及相关参数，确保向后兼容性
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法进行编码处理，返回批量编码结果
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
    ):
        """
        Internal method to perform batch encoding of answers.

        Args:
            answer `List[str]`:
                List of answer strings to encode.
            add_special_tokens `bool`:
                Whether to add special tokens.
            padding_strategy `PaddingStrategy`:
                Strategy for padding sequences.
            truncation_strategy `TruncationStrategy`:
                Strategy for truncating sequences.
            max_length `Optional[int]`:
                Maximum length of the sequences.
            stride `int`:
                Stride for tokenization.
            pad_to_multiple_of `Optional[int]`:
                Pad to a multiple of this value.
            return_tensors `Optional[Union[str, TensorType]]`:
                Optionally return tensors.
            return_token_type_ids `Optional[bool]`:
                Whether to return token type IDs.
            return_attention_mask `Optional[bool]`:
                Whether to return attention masks.
            return_overflowing_tokens `bool`:
                Whether to return overflowing tokens.
            return_special_tokens_mask `bool`:
                Whether to return special tokens mask.
            return_offsets_mapping `bool`:
                Whether to return offsets mapping.
            return_length `bool`:
                Whether to return sequence lengths.
            verbose `bool`:
                Whether to print verbose information.
            **kwargs:
                Additional keyword arguments.
        
        Returns:
            `BatchEncoding`: Batch encoding containing encoded answers.
        """
        # 实现批量编码处理的具体逻辑
        # 这里会包括对答案进行标记化、填充和截断等操作
        # 返回最终的批量编码结果
        pass
    ) -> BatchEncoding:
        # 初始化一个空的批次输出字典
        batch_outputs = {}
        # 遍历每个答案文本
        for text in answer:
            # 如果设定为小写处理，则将文本转换为小写
            if self.do_lower_case:
                text = text.lower()

            # 对文本进行分词处理
            tokens = self.tokenize(text)
            # 准备模型输入，包括将分词转换为 ID，设定特殊标记的添加策略，
            # 设定截断策略、最大长度、步长等参数
            outputs = self.prepare_for_model(
                ids=self.convert_tokens_to_ids(tokens),
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # 在后续批次中进行填充
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # 在后续批次中进行填充
                return_attention_mask=False,  # 在后续批次中进行填充
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # 在最后将整个批次转换为张量
                prepend_batch_axis=False,
                verbose=verbose,
            )

            # 将输出结果添加到批次输出字典中
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # 对批次输出进行填充处理
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # 将填充后的输出转换为 BatchEncoding 类型
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        # 返回 BatchEncoding 对象
        return BatchEncoding(batch_outputs)

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
        # 对答案字符串进行编码，返回不包含其他信息（如 token type IDs、attention masks 等）的输出
        encoded_outputs = self.target_encode_plus(
            answer=answer,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

        # 返回编码后的输入 ID 列表
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
        # 获取填充和截断策略，并处理兼容性问题
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法 `_target_encode_plus`，进行编码处理
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
    ):
        # 在内部方法中执行实际的编码处理，具体处理细节略去
        pass
    ) -> BatchEncoding:
        # 如果需要返回偏移映射，则抛出未实现的错误，Python tokenizers 不支持此功能
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # 将答案文本赋给变量 text
        text = answer

        # 如果需要进行小写处理
        if self.do_lower_case:
            # 将文本转换为小写
            text = text.lower()

        # 对文本进行分词处理，得到 tokens
        tokens = self.tokenize(text)

        # 准备模型输入
        return self.prepare_for_model(
            ids=self.convert_tokens_to_ids(tokens),  # 将 tokens 转换为对应的 token IDs
            add_special_tokens=add_special_tokens,  # 是否添加特殊 tokens
            padding=padding_strategy.value,  # 填充策略
            truncation=truncation_strategy.value,  # 截断策略
            max_length=max_length,  # 最大长度限制
            stride=stride,  # 步长
            pad_to_multiple_of=pad_to_multiple_of,  # 填充至某个倍数长度
            return_tensors=return_tensors,  # 返回的张量类型
            prepend_batch_axis=True,  # 是否添加批处理维度
            return_attention_mask=return_attention_mask,  # 是否返回注意力掩码
            return_token_type_ids=return_token_type_ids,  # 是否返回 token 类型 IDs
            return_overflowing_tokens=return_overflowing_tokens,  # 是否返回溢出 tokens
            return_special_tokens_mask=return_special_tokens_mask,  # 是否返回特殊 tokens 掩码
            return_length=return_length,  # 是否返回长度
            verbose=verbose,  # 是否详细输出
        )
    ):
        """
        This method can be used to linearize a table and add a corresponding query.

        Optionally, it also handles truncation of the table (cells).

        An answer can be provided for more precise truncation.
        """
        if not table.empty:
            # step 1: create table dictionary
            # 将表格内容转换为包含表头和行数据的字典
            table_content = {"header": list(table.columns), "rows": [list(row.values) for i, row in table.iterrows()]}

            # step 2: modify table internally
            # always truncate table cells based on self.max_cell_length
            # optionally truncate rows if truncation_strategy is set to it
            # 根据 self.max_cell_length 截断表格单元格，根据截断策略处理行
            self.truncate_table_cells(table_content, query, answer)
            if truncation_strategy == TapexTruncationStrategy.DROP_ROWS_TO_FIT:
                self.truncate_table_rows(table_content, query, answer, max_length=max_length)

            # step 3: linearize table
            # 线性化表格数据
            linear_table = self.table_linearize.process_table(table_content)
        else:
            linear_table = ""

        if linear_table == "":
            logger.warning(
                "You provide an empty table, or all cells contain much tokens (e.g., >= 1024 tokens). "
                + f"Please carefully check the corresponding table with the query : {query}."
            )
        if query == "":
            logger.warning("You provide nothing to query with respect to the table.")
        # step 4: concatenate query with linear_table
        # 拼接查询和线性化后的表格数据
        separator = " " if query and linear_table else ""
        joint_input = (query + separator + linear_table) if query else linear_table

        return joint_input

    def truncate_table_cells(self, table_content: Dict, question: str, answer: List):
        # TODO (Qian): is it possible to revert the original cell if it is in the final answer?
        # 截断表格单元格，并记录截断前后的映射关系
        cell_mapping = {}
        for row in table_content["rows"]:
            for i, cell in enumerate(row):
                truncate_cell = self.truncate_cell(cell)
                if truncate_cell is not None:
                    cell_mapping[cell] = truncate_cell
                    row[i] = truncate_cell

        # modify the answer list
        # 修改答案列表，如果答案中有映射到截断后的单元格，则更新为截断后的值
        if answer is not None:
            for i, case in enumerate(answer):
                if case in cell_mapping.keys():
                    answer[i] = cell_mapping[case]

    def truncate_cell(self, cell_value):
        # do not process on these cases
        # 如果单元格的值是整数或浮点数，则直接返回
        if isinstance(cell_value, int) or isinstance(cell_value, float):
            return cell_value
        # 如果单元格值不为空白，则尝试分词并根据 self.max_cell_length 截断
        if cell_value.strip() != "":
            try_tokens = self.tokenize(cell_value)
            if len(try_tokens) >= self.max_cell_length:
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
        # 估计删除比例和剩余可容纳的标记长度
        delete_ratio, remain_token_len = self.estimate_delete_ratio(table_content, question, max_length)
        
        # 随机删除不相关的行
        self.delete_unrelated_rows(table_content, question, answer, delete_ratio)
        
        # 保证结果长度小于 max_length
        maximum_keep_rows = 0
        for ind, row_example in enumerate(table_content["rows"]):
            # 处理表格行数据并计算其标记长度
            value_string = self.table_linearize.process_row(row_example, ind + 1)
            value_token_len = len(self.tokenize(value_string))
            
            # 如果超过最大长度限制，则停止处理
            if value_token_len > remain_token_len:
                break
            
            # 更新剩余标记长度并增加保留的行数
            remain_token_len -= value_token_len
            maximum_keep_rows += 1
        
        # 删除超出最大长度限制的行
        del table_content["rows"][maximum_keep_rows:]

    def estimate_delete_ratio(self, table_content: Dict, question: str, max_length=None):
        # 检查表格内容是否包含必需的 'header' 和 'rows' 键
        if "header" not in table_content or "rows" not in table_content:
            raise ValueError("The table content should contain both 'header' and 'rows' keys.")
        
        # 计算问题的标记数（包括特殊标记）
        question_tokens = self.tokenize(question, add_special_tokens=True)
        
        # 处理表头并计算其标记数（不包括特殊标记）
        header_string = self.table_linearize.process_header(table_content["header"])
        header_tokens = self.tokenize(header_string, add_special_tokens=False)
        
        # 计算问题和表头的总标记数
        used_token_len = len(question_tokens) + len(header_tokens)
        
        # 计算剩余的标记空间用于行数据
        remain_token_len = max_length - used_token_len
        
        # 计算表格所有行的标记数以粗略估计总长度
        value_string = ""
        for _, row_example in enumerate(table_content["rows"]):
            value_string += self.table_linearize.process_row(row_example, 100) + " "
        value_token_len = len(self.tokenize(value_string))
        
        # 如果总标记数小于剩余的标记空间，则不需要删除行
        if value_token_len < remain_token_len:
            return 0.0, remain_token_len
        else:
            # 计算大致的删除比例
            return 1.0 - remain_token_len / value_token_len, remain_token_len
    # 定义一个方法，用于删除与给定问题和答案不相关的表格行
    def delete_unrelated_rows(self, table_content: Dict, question: str, answer: List, delete_ratio: float):
        """
        The argument answer is used only during training.
        参数 answer 仅在训练过程中使用。
        """
        # 用于存储被截断的不相关行的索引列表
        truncated_unrelated_indices = []
        # 用于存储相关行的索引列表
        related_indices = []

        # 如果 answer 为 None 或者空列表，则创建一个空的答案集合
        if answer is None or len(answer) == 0:
            answer_set = set()
        else:
            # 将答案列表中的每个答案转换为小写并添加到答案集合中
            answer_set = {ans_ex.lower() for ans_ex in answer}

        # 如果存在问题，将问题分割成单词并添加到答案集合中
        if question is not None:
            answer_set.update(question.split())

        # 将问题去除标点符号后分割成单词并存储为问题集合
        question_set = set(question.strip("?!.,").split(" "))

        # 计算表格内容中行的最大长度
        row_max_len = len(table_content["rows"])

        # 遍历表格内容中的每一行
        for _row_idx, row in enumerate(table_content["rows"]):
            # 将当前行中每个单元格的值转换为小写，并存储为集合
            lower_row = {str(cell).lower() for cell in row}

            # 如果当前行既不包含答案集合中的任何单词，也不包含问题集合中的任何单词，则将其索引添加到截断不相关行的索引列表中
            if len(lower_row & answer_set) == 0 and len(lower_row & question_set) == 0:
                truncated_unrelated_indices.append(_row_idx)
            else:
                # 如果当前行包含答案集合或问题集合中的单词，则将当前行索引及其前后两行的索引添加到相关行的索引列表中
                related_indices.extend([_row_idx - 2, _row_idx - 1, _row_idx, _row_idx + 1, _row_idx + 2])

        # 从截断的不相关行索引列表中移除相关行的索引
        truncated_unrelated_indices = [
            _row_idx for _row_idx in truncated_unrelated_indices if _row_idx not in related_indices
        ]

        # 计算要删除的行数，最小为截断不相关行索引列表的长度和总行数乘以删除比例的整数部分
        drop_items = min(len(truncated_unrelated_indices), int(len(table_content["rows"]) * delete_ratio))

        # 从截断不相关行索引列表中随机选择要删除的行数，并存储为删除行的索引列表
        drop_row_indices = random.choices(truncated_unrelated_indices, k=drop_items)

        # 倒序遍历表格内容中的行索引
        for _row_idx in reversed(range(row_max_len)):
            # 如果当前行索引在删除行索引列表中，则从表格内容中删除该行
            if _row_idx in drop_row_indices:
                del table_content["rows"][_row_idx]

        # 如果表格内容中包含 ID 并且删除的行数大于 0，则记录警告日志
        if "id" in table_content and len(drop_row_indices) > 0:
            logger.warning("Delete {:.2f} rows in table {}".format(len(drop_row_indices), table_content["id"]))
```