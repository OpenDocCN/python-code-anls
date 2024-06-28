# `.\models\layoutlm\tokenization_layoutlm.py`

```py
# coding=utf-8
# Copyright 2018 The Microsoft Research Asia LayoutLM Team Authors.
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
""" Tokenization class for model LayoutLM."""

import collections  # 导入 collections 模块，用于处理数据集合
import os  # 导入 os 模块，用于处理操作系统相关功能
import unicodedata  # 导入 unicodedata 模块，用于 Unicode 字符数据的处理
from typing import List, Optional, Tuple  # 导入类型提示相关的功能

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace  # 导入其他模块中的相关功能
from ...utils import logging  # 导入日志记录工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}  # 定义词汇表文件名字典

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/layoutlm-base-uncased": (
            "https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main/vocab.txt"
        ),
        "microsoft/layoutlm-large-uncased": (
            "https://huggingface.co/microsoft/layoutlm-large-uncased/resolve/main/vocab.txt"
        ),
    }
}  # 预训练词汇表文件映射字典，指定 LayoutLM 模型的预训练词汇表文件及其来源 URL

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/layoutlm-base-uncased": 512,
    "microsoft/layoutlm-large-uncased": 512,
}  # 预训练位置嵌入大小字典，指定 LayoutLM 不同预训练模型的位置嵌入大小

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/layoutlm-base-uncased": {"do_lower_case": True},
    "microsoft/layoutlm-large-uncased": {"do_lower_case": True},
}  # 预训练初始化配置字典，指定 LayoutLM 不同预训练模型的初始化配置

# Copied from transformers.models.bert.tokenization_bert.load_vocab
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()  # 创建一个有序字典对象 vocab
    with open(vocab_file, "r", encoding="utf-8") as reader:  # 打开词汇表文件进行读取
        tokens = reader.readlines()  # 读取文件的所有行
    for index, token in enumerate(tokens):  # 遍历行号和行内容
        token = token.rstrip("\n")  # 去除行末的换行符
        vocab[token] = index  # 将单词和对应索引存入字典
    return vocab  # 返回加载后的词汇表字典

# Copied from transformers.models.bert.tokenization_bert.whitespace_tokenize
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()  # 去除文本两端空白字符
    if not text:  # 如果文本为空
        return []  # 返回空列表
    tokens = text.split()  # 使用空格分割文本，得到单词列表
    return tokens  # 返回分割后的单词列表

# Copied from transformers.models.bert.tokenization_bert.BertTokenizer with Bert->LayoutLM,BERT->LayoutLM
class LayoutLMTokenizer(PreTrainedTokenizer):
    r"""
    Construct a LayoutLM tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original LayoutLM).
    ```

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    ```
    # 初始化方法，用于创建一个新的 Tokenizer 对象
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
    ```
    ):
        # 如果提供的词汇文件路径不是一个文件，则抛出数值错误异常，提示找不到词汇文件
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = LayoutLMTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表并赋值给实例变量 self.vocab
        self.vocab = load_vocab(vocab_file)
        # 创建一个有序字典，将词汇表中的 id 和 token 对调，赋值给实例变量 self.ids_to_tokens
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 初始化是否进行基础分词的标志
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要进行基础分词
        if do_basic_tokenize:
            # 创建 BasicTokenizer 实例并赋值给 self.basic_tokenizer
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        # 创建 WordpieceTokenizer 实例并赋值给 self.wordpiece_tokenizer，使用未知标记 unk_token
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的初始化方法，并传递相应参数
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
        # 返回基础分词器的小写标志位
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        # 返回词汇表的大小
        return len(self.vocab)

    def get_vocab(self):
        # 返回包含额外 tokens 编码器的词汇表字典
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, split_special_tokens=False):
        # 分词后的 token 列表
        split_tokens = []
        # 如果需要进行基础分词
        if self.do_basic_tokenize:
            # 使用基础分词器对文本进行分词
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果 token 是不能分割的特殊 token
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 使用 WordpieceTokenizer 对 token 进行分词，并添加到 split_tokens 中
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 直接使用 WordpieceTokenizer 对文本进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将 token 转换为对应的 id，如果 token 不在词汇表中，则使用 unk_token
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将索引 index 转换为对应的 token，如果索引不在 ids_to_tokens 中，则使用 unk_token
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将 tokens 序列转换为单个字符串，去除 " ##" 并去除首尾空格
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs from a sequence or a pair of sequences for sequence classification tasks. This method assigns
        different token type IDs to distinguish between the first sequence, the second sequence (if provided), and padding.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs representing the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs representing the second sequence.

        Returns:
            `List[int]`: List of token type IDs.
        """
        if token_ids_1 is None:
            # For a single sequence, token type IDs are 0 for all tokens
            return [0] * len(token_ids_0)
        # For a pair of sequences, assign token type 0 to the first sequence and token type 1 to the second sequence
        token_type_ids = [0] * len(token_ids_0) + [1] * len(token_ids_1)
        return token_type_ids
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A LayoutLM sequence
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
            `List[int]`: List of token type IDs according to the given sequence(s).
        """
        # Define separator and classification token IDs
        sep = [self.sep_token_id]  # Separation token ID
        cls = [self.cls_token_id]  # Classification token ID

        # If only one sequence is provided (token_ids_1 is None), return mask for the first sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]

        # If two sequences are provided, return combined mask for both sequences
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary of the model to a specified directory or file.

        Args:
            save_directory (str):
                Directory path where the vocabulary will be saved.
            filename_prefix (str, *optional*):
                Optional prefix to prepend to the vocabulary file name.

        Returns:
            Tuple[str]: A tuple containing the path to the saved vocabulary file.
        """
        index = 0

        # Determine the full path for saving the vocabulary file
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory

        # Write the vocabulary to the specified file
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # Check for non-consecutive indices and issue a warning if found
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1

        # Return the path to the saved vocabulary file
        return (vocab_file,)
# Copied from transformers.models.bert.tokenization_bert.BasicTokenizer
# 从transformers.models.bert.tokenization_bert.BasicTokenizer复制而来

class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).
    创建一个BasicTokenizer对象，执行基本的分词（标点符号拆分，小写转换等）。

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
            是否在分词时将输入转换为小写。
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
            在分词过程中不会被拆分的token集合，仅在`do_basic_tokenize=True`时生效。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.
            是否分词中文字符。对于日语，应该禁用此选项（参见此issue）。
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
            是否删除所有的重音符号。如果未指定此选项，则将由`lowercase`的值决定（与原始BERT相同）。
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            In some instances we want to skip the basic punctuation splitting so that later tokenization can capture
            the full context of the words, such as contractions.
            在某些情况下，我们希望跳过基本的标点符号拆分，以便后续的分词可以捕获词语的完整上下文，如缩写。

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
        # 初始化BasicTokenizer对象
        self.do_lower_case = do_lower_case
        # 是否进行小写处理
        self.never_split = set(never_split)
        # 设置不会被拆分的token集合
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 是否分词中文字符
        self.strip_accents = strip_accents
        # 是否删除重音符号
        self.do_split_on_punc = do_split_on_punc
        # 是否基于标点符号进行拆分
    # 对输入的文本进行基本的分词处理。用于子词分词，请参见 WordPieceTokenizer。

    # 如果传入了 never_split 参数，则将其与类属性 never_split 的集合取并集，以获取最终的不分割的标记集合。
    never_split = self.never_split.union(set(never_split)) if never_split else self.never_split

    # 清理文本，去除可能存在的特殊符号和空白字符。
    text = self._clean_text(text)

    # 若设置了 tokenize_chinese_chars 标志为 True，则对包含中文字符的文本进行特殊处理。
    if self.tokenize_chinese_chars:
        text = self._tokenize_chinese_chars(text)

    # 对文本进行 Unicode 规范化，确保文本中的字符使用 NFC 规范。
    unicode_normalized_text = unicodedata.normalize("NFC", text)

    # 使用空白字符进行基本的分词，得到原始的 token 列表。
    orig_tokens = whitespace_tokenize(unicode_normalized_text)

    # 初始化空列表，用于存储最终的分词结果。
    split_tokens = []

    # 遍历原始 token 列表，对每个 token 进行处理。
    for token in orig_tokens:
        # 如果 token 不在不分割的标记集合中，则进行进一步处理。
        if token not in never_split:
            # 如果设置了 do_lower_case 标志为 True，则将 token 转换为小写。
            if self.do_lower_case:
                token = token.lower()
                # 如果 strip_accents 不为 False，则移除 token 中的重音符号。
                if self.strip_accents is not False:
                    token = self._run_strip_accents(token)
            # 如果 strip_accents 标志为 True，则移除 token 中的重音符号。
            elif self.strip_accents:
                token = self._run_strip_accents(token)

        # 将处理后的 token 列表拼接到 split_tokens 中。
        split_tokens.extend(self._run_split_on_punc(token, never_split))

    # 将拼接后的分词结果使用空白字符再次进行分割，得到最终的输出 token 列表。
    output_tokens = whitespace_tokenize(" ".join(split_tokens))

    # 返回最终的输出 token 列表作为函数的返回值。
    return output_tokens
    def _run_split_on_punc(self, text, never_split=None):
        """按标点符号分割文本。

        Args:
            text (str): 要分割的文本字符串。
            never_split (set): 不希望分割的文本集合。

        Returns:
            list: 分割后的文本列表。

        Notes:
            如果不需要按标点符号分割或者指定的文本在never_split中，直接返回原文本。
        """
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])  # 将标点符号作为单独的列表项添加到输出列表中
                start_new_word = True  # 标记需要开始一个新单词
            else:
                if start_new_word:
                    output.append([])  # 如果需要开始一个新单词，添加一个空列表
                start_new_word = False  # 取消开始新单词的标记
                output[-1].append(char)  # 将当前字符添加到最后一个单词的列表中
            i += 1

        return ["".join(x) for x in output]  # 将列表中的字符列表连接成字符串后返回一个列表

    def _tokenize_chinese_chars(self, text):
        """在每个CJK字符周围添加空格。

        Args:
            text (str): 要处理的文本字符串。

        Returns:
            str: 处理后的文本字符串。
        """
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")  # 在CJK字符前后添加空格
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)  # 将列表中的字符连接成一个字符串后返回

    def _is_chinese_char(self, cp):
        """检查CP是否为CJK字符的码点。

        Args:
            cp (int): Unicode码点值。

        Returns:
            bool: 如果是CJK字符则返回True，否则返回False。
        """
        # 这里的CJK字符指的是CJK统一表意文字的Unicode块：
        # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # 注意，CJK Unicode块并不包含所有的日语和韩语字符，
        # 现代韩语的谚文字母和片假名、片假名分别属于不同的Unicode块，
        # 这些字符用于书写空格分隔的词语，因此不会被特殊对待而被处理。
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
        """清除文本中的无效字符和空白字符。

        Args:
            text (str): 要清理的文本字符串。

        Returns:
            str: 清理后的文本字符串。
        """
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")  # 将空白字符替换为单个空格
            else:
                output.append(char)
        return "".join(output)  # 将列表中的字符连接成一个字符串后返回
# Copied from transformers.models.bert.tokenization_bert.WordpieceTokenizer
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化WordpieceTokenizer对象，设置词汇表、未知token和单词最大字符数限制
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
        # 初始化输出token列表
        output_tokens = []
        # 使用空白字符分词器对文本进行分词，返回的是一个token列表
        for token in whitespace_tokenize(text):
            chars = list(token)
            # 如果token的字符数超过最大字符数限制，则将其替换为未知token
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    # 构建当前子字符串
                    substr = "".join(chars[start:end])
                    # 如果不是第一个子字符串，则在前面加上"##"
                    if start > 0:
                        substr = "##" + substr
                    # 如果当前子字符串在词汇表中，则选择当前子字符串作为最长匹配词
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                # 如果没有找到匹配的词，则标记为无效
                if cur_substr is None:
                    is_bad = True
                    break
                # 将匹配的词加入到sub_tokens列表中
                sub_tokens.append(cur_substr)
                start = end

            # 如果token被标记为无效，则使用未知token代替
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        # 返回最终的token列表
        return output_tokens
```