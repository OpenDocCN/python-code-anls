# `.\models\realm\tokenization_realm.py`

```
# coding=utf-8
# Copyright 2022 The REALM authors and The HuggingFace Inc. team.
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
"""Tokenization classes for REALM."""

# Import necessary libraries
import collections  # 导入 collections 模块
import os  # 导入 os 模块
import unicodedata  # 导入 unicodedata 模块
from typing import List, Optional, Tuple  # 导入类型提示相关的模块

# Import from tokenization_utils
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
# Import from tokenization_utils_base
from ...tokenization_utils_base import BatchEncoding
# Import logging from utils
from ...utils import PaddingStrategy, logging

# Get logger instance for current module
logger = logging.get_logger(__name__)

# Define constant for vocabulary file names
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# Define mapping of pretrained model names to their respective vocabulary file URLs
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/realm-cc-news-pretrained-embedder": (
            "https://huggingface.co/google/realm-cc-news-pretrained-embedder/resolve/main/vocab.txt"
        ),
        "google/realm-cc-news-pretrained-encoder": (
            "https://huggingface.co/google/realm-cc-news-pretrained-encoder/resolve/main/vocab.txt"
        ),
        "google/realm-cc-news-pretrained-scorer": (
            "https://huggingface.co/google/realm-cc-news-pretrained-scorer/resolve/main/vocab.txt"
        ),
        "google/realm-cc-news-pretrained-openqa": (
            "https://huggingface.co/google/realm-cc-news-pretrained-openqa/aresolve/main/vocab.txt"
        ),
        "google/realm-orqa-nq-openqa": "https://huggingface.co/google/realm-orqa-nq-openqa/resolve/main/vocab.txt",
        "google/realm-orqa-nq-reader": "https://huggingface.co/google/realm-orqa-nq-reader/resolve/main/vocab.txt",
        "google/realm-orqa-wq-openqa": "https://huggingface.co/google/realm-orqa-wq-openqa/resolve/main/vocab.txt",
        "google/realm-orqa-wq-reader": "https://huggingface.co/google/realm-orqa-wq-reader/resolve/main/vocab.txt",
    }
}

# Define sizes of positional embeddings for different pretrained models
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/realm-cc-news-pretrained-embedder": 512,
    "google/realm-cc-news-pretrained-encoder": 512,
    "google/realm-cc-news-pretrained-scorer": 512,
    "google/realm-cc-news-pretrained-openqa": 512,
    "google/realm-orqa-nq-openqa": 512,
    "google/realm-orqa-nq-reader": 512,
    "google/realm-orqa-wq-openqa": 512,
    "google/realm-orqa-wq-reader": 512,
}

# Define initial configurations for different pretrained models
PRETRAINED_INIT_CONFIGURATION = {
    "google/realm-cc-news-pretrained-embedder": {"do_lower_case": True},
    "google/realm-cc-news-pretrained-encoder": {"do_lower_case": True},
    "google/realm-cc-news-pretrained-scorer": {"do_lower_case": True},
    "google/realm-cc-news-pretrained-openqa": {"do_lower_case": True},
}
    # 定义一个字典，包含多个键值对，每个键是字符串，对应的值是一个字典，具有一个布尔型键"do_lower_case"，其值为True
    "google/realm-orqa-nq-openqa": {"do_lower_case": True},
    "google/realm-orqa-nq-reader": {"do_lower_case": True},
    "google/realm-orqa-wq-openqa": {"do_lower_case": True},
    "google/realm-orqa-wq-reader": {"do_lower_case": True},
}

# 定义一个函数 load_vocab，用于加载一个词汇文件到一个有序字典中
def load_vocab(vocab_file):
    vocab = collections.OrderedDict()  # 创建一个有序字典对象 vocab
    with open(vocab_file, "r", encoding="utf-8") as reader:  # 打开词汇文件以读取模式，并指定编码为 utf-8
        tokens = reader.readlines()  # 读取文件的所有行并存储在 tokens 列表中
    for index, token in enumerate(tokens):  # 遍历 tokens 列表的索引和元素
        token = token.rstrip("\n")  # 去掉 token 末尾的换行符
        vocab[token] = index  # 将 token 和其索引添加到 vocab 字典中
    return vocab  # 返回加载完成的词汇字典

# 定义一个函数 whitespace_tokenize，用于对文本进行基本的空白符清理和分割
def whitespace_tokenize(text):
    text = text.strip()  # 去除文本两端的空白符
    if not text:  # 如果文本为空
        return []  # 返回空列表
    tokens = text.split()  # 使用空白符对文本进行分割，并存储结果在 tokens 列表中
    return tokens  # 返回分割后的 tokens 列表

# 定义一个类 RealmTokenizer，继承自 PreTrainedTokenizer 类
class RealmTokenizer(PreTrainedTokenizer):
    r"""
    Construct a REALM tokenizer.

    [`RealmTokenizer`] is identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation splitting and
    wordpiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # 定义类的初始化方法，用于初始化一个新的Tokenizer对象
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
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
    ):
        if not os.path.isfile(vocab_file):
            # 如果给定的词汇文件不存在，则抛出数值错误异常
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = RealmTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载给定路径下的词汇表文件并存储到实例变量 self.vocab 中
        self.vocab = load_vocab(vocab_file)
        # 使用 collections.OrderedDict 创建从词汇 ID 到词汇的有序映射
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 根据参数决定是否执行基础分词
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            # 如果需要进行基础分词，则创建 BasicTokenizer 对象
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 使用给定的词汇表和未知标记创建 WordpieceTokenizer 对象
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))
        # 调用父类构造函数，初始化实例
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

    @property
    def do_lower_case(self):
        # 返回基础分词器的小写标志
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        # 返回词汇表的大小（词汇数量）
        return len(self.vocab)

    def get_vocab(self):
        # 返回包含词汇表及其附加标记编码器的字典
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        # 对输入文本进行分词处理，返回分词后的 token 列表
        split_tokens = []
        if self.do_basic_tokenize:
            # 如果需要进行基础分词
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # 如果 token 在 never_split 集合中，则直接添加到分词结果列表中
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 否则，使用 WordpieceTokenizer 对 token 进行进一步分词，并添加到分词结果列表中
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 如果不需要进行基础分词，则直接使用 WordpieceTokenizer 对整个文本进行分词处理
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 根据词汇表将 token 转换为对应的 ID，如果未找到，则使用未知标记的 ID
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 根据词汇表将索引转换为对应的 token，如果索引未找到，则使用未知标记
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将 token 列表转换为单个字符串，去除连字符（"##"），并去除两端空白
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens. A REALM sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens.
        """
        # If only one sequence is provided, add `[CLS]`, the sequence tokens, and `[SEP]`
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        
        # For sequence pairs, construct `[CLS]`, tokens of first sequence, `[SEP]`, tokens of second sequence, and final `[SEP]`
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

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

        # If the token list already has special tokens, delegate to the base class method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Construct a special tokens mask for sequences without existing special tokens
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs tensor from given sequence or pair of sequences. A REALM token type IDs sequence has the
        following format:

        - single sequence: `[0] * (len(token_ids_0) + 2)`
        - pair of sequences: `[0] * (len(token_ids_0) + 2) + [1] * (len(token_ids_1) + 1)`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs with the appropriate length and values.
        """
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A REALM sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # Define special tokens for separation and classification
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        # If only one sequence is provided, return a mask with zeros for its length
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # For sequence pairs, concatenate tokens with special tokens and create the mask
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Initialize index for vocabulary items
        index = 0
        
        # Determine the vocabulary file path based on whether save_directory is a directory or a file path
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        
        # Write the vocabulary items to the specified file
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # Iterate over sorted vocabulary items and write them to the file
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # Check for non-consecutive indices and log a warning if found
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # Write the token to the file followed by a newline
                writer.write(token + "\n")
                index += 1
        
        # Return the path to the saved vocabulary file
        return (vocab_file,)
# 定义一个名为 BasicTokenizer 的类，用于执行基本的分词操作（如标点符号分割、转换为小写等）。
class BasicTokenizer(object):

    """
    构造一个 BasicTokenizer 实例，用于运行基本的分词操作（如标点符号分割、转换为小写等）。

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在分词时将输入转换为小写。
        never_split (`Iterable`, *optional*):
            在分词过程中永远不会被拆分的 token 集合。仅在 `do_basic_tokenize=True` 时有效。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否分词中文字符。
            
            对于日语，这应该被禁用（参见此 [issue](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否去除所有的重音符号。如果未指定此选项，则会根据 `lowercase` 的值（与原始 BERT 相同）来确定。
    """

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        # 如果 never_split 参数为 None，则将其设置为空列表
        if never_split is None:
            never_split = []
        # 设置是否将输入转换为小写
        self.do_lower_case = do_lower_case
        # 将 never_split 转换为集合，这些 token 在分词时不会被拆分
        self.never_split = set(never_split)
        # 是否分词中文字符
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 是否去除重音符号
        self.strip_accents = strip_accents
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 如果提供了 never_split 参数，则将其与 self.never_split 取并集，否则使用 self.never_split
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，处理特殊字符等
        text = self._clean_text(text)

        # 以下内容于2018年11月1日添加，用于多语言和中文模型。
        # 现在也应用于英语模型，但这并不重要，因为英语模型没有在任何中文数据上训练，
        # 通常不包含任何中文数据（尽管词汇表中有些中文词汇，因为英文维基百科中有一些中文词汇）。
        if self.tokenize_chinese_chars:
            # 对中文字符进行特殊处理
            text = self._tokenize_chinese_chars(text)
        # 将文本按空白符分割为原始token
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    # 如果设置为小写，则将token转换为小写
                    token = token.lower()
                    if self.strip_accents is not False:
                        # 如果需要去除重音符号，则执行去除重音符号操作
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    # 如果需要去除重音符号，则执行去除重音符号操作
                    token = self._run_strip_accents(token)
            # 将token根据标点符号进行分割，并扩展到split_tokens中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将分割后的token重新按空白符合并，并返回
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 将文本中的重音符号规范化为NFD形式
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            # 如果字符类别为Mn（非间距连字符），则跳过该字符
            if cat == "Mn":
                continue
            output.append(char)
        # 将处理后的字符列表连接成字符串并返回
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果指定了never_split，并且text在never_split中，则不分割，直接返回
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号，则将其作为一个新的列表项添加到output中
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，根据start_new_word标志判断是否创建新的列表项
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        # 将分割后的字符列表重新连接成字符串并返回
        return ["".join(x) for x in output]
    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)  # 获取字符的 Unicode 码点
            if self._is_chinese_char(cp):  # 如果字符是中日韩字符，则在其前后添加空格
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)  # 如果不是中日韩字符，则直接添加字符
        return "".join(output)  # 将处理后的字符列表连接成字符串并返回

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)  # CJK 统一汉字
            or (cp >= 0x3400 and cp <= 0x4DBF)  # CJK 扩展A
            or (cp >= 0x20000 and cp <= 0x2A6DF)  # CJK 扩展B
            or (cp >= 0x2A700 and cp <= 0x2B73F)  # CJK 扩展C
            or (cp >= 0x2B740 and cp <= 0x2B81F)  # CJK 扩展D
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  # CJK 扩展E
            or (cp >= 0xF900 and cp <= 0xFAFF)  # CJK 兼容汉字
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  # CJK 兼容表意文字
        ):  # 判断 Unicode 码点是否在中日韩字符范围内
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)  # 获取字符的 Unicode 码点
            if cp == 0 or cp == 0xFFFD or _is_control(char):  # 如果字符是无效字符或控制字符，则跳过
                continue
            if _is_whitespace(char):  # 如果字符是空白字符，则替换为单个空格
                output.append(" ")
            else:
                output.append(char)  # 否则保留字符
        return "".join(output)  # 将处理后的字符列表连接成字符串并返回
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化 WordpieceTokenizer 类，设置词汇表、未知标记和每个单词最大字符数
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

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
        # 对文本进行分词处理，以空白字符为分隔符
        for token in whitespace_tokenize(text):
            chars = list(token)
            # 如果单词长度超过设定的最大字符数，将其视为未知标记
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                # 采用贪婪算法寻找最长匹配的子串
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    # 检查子串是否在词汇表中
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            # 如果无法成功分词，则添加未知标记；否则添加分词结果到输出列表
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
```