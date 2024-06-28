# `.\models\lxmert\tokenization_lxmert.py`

```py
# coding=utf-8
# Copyright 2020 The Google AI Team, Stanford University and The HuggingFace Inc. team.
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

import collections  # 引入 collections 模块，用于 OrderedDict 的创建
import os  # 引入 os 模块，用于操作系统相关功能
import unicodedata  # 引入 unicodedata 模块，用于 Unicode 数据库中的字符属性查询
from typing import List, Optional, Tuple  # 引入类型提示相关的工具

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace  # 引入 LxmertTokenizer 所需的模块
from ...utils import logging  # 引入 logging 模块，用于日志记录

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}  # 定义词汇表文件名的映射

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "unc-nlp/lxmert-base-uncased": "https://huggingface.co/unc-nlp/lxmert-base-uncased/resolve/main/vocab.txt",
    }
}  # 预训练模型词汇表文件的映射

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "unc-nlp/lxmert-base-uncased": 512,
}  # 预训练模型的位置编码嵌入大小映射

PRETRAINED_INIT_CONFIGURATION = {
    "unc-nlp/lxmert-base-uncased": {"do_lower_case": True},
}  # 预训练模型的初始化配置映射


# Copied from transformers.models.bert.tokenization_bert.load_vocab
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()  # 创建一个有序字典对象 vocab
    with open(vocab_file, "r", encoding="utf-8") as reader:  # 打开词汇表文件
        tokens = reader.readlines()  # 逐行读取文件内容
    for index, token in enumerate(tokens):  # 遍历行索引和行内容
        token = token.rstrip("\n")  # 去除行尾的换行符
        vocab[token] = index  # 将 token 添加到 vocab 字典，并使用索引作为值
    return vocab  # 返回构建好的词汇表字典


# Copied from transformers.models.bert.tokenization_bert.whitespace_tokenize
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()  # 去除文本两端的空白字符
    if not text:  # 如果文本为空
        return []  # 返回空列表
    tokens = text.split()  # 使用空白字符分割文本，得到 token 列表
    return tokens  # 返回分割后的 token 列表


# Copied from transformers.models.bert.tokenization_bert.BertTokenizer with bert-base-cased->unc-nlp/lxmert-base-uncased, BERT->Lxmert, BertTokenizer->LxmertTokenizer
class LxmertTokenizer(PreTrainedTokenizer):
    r"""
    Construct a Lxmert tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    pass  # LxmertTokenizer 类暂时没有实现额外的方法或属性，因此只需保留文档字符串即可
    # 定义一个类，用于处理词汇表和标记化参数的配置
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇文件名映射表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 预训练位置嵌入的最大输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # 初始化方法，用于设置词汇文件、标记化的参数及其它配置
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
        # 检查词汇文件是否存在，如果不存在则抛出 ValueError 异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = LxmertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表到 self.vocab
        self.vocab = load_vocab(vocab_file)
        # 根据词汇表创建一个从 id 到 token 的有序字典 self.ids_to_tokens
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 根据参数设置是否进行基本的 tokenization
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            # 如果需要进行基本 tokenization，则初始化 BasicTokenizer
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        # 使用给定的词汇表和未知标记 unk_token 初始化 WordpieceTokenizer
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的初始化方法，设置各种参数和特殊标记
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
        # 返回当前的 do_lower_case 参数值，由 basic_tokenizer 决定
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        # 返回词汇表的大小，即词汇表中条目的数量
        return len(self.vocab)

    def get_vocab(self):
        # 返回一个包含词汇表和 added_tokens_encoder 的合并字典
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, split_special_tokens=False):
        # 初始化分割后的 token 列表
        split_tokens = []
        if self.do_basic_tokenize:
            # 如果需要进行基本 tokenization，则使用 BasicTokenizer 分词
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果 token 在 never_split 集合中，则直接添加到 split_tokens 中
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 否则，使用 WordpieceTokenizer 进行进一步分词，并添加到 split_tokens 中
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 如果不需要基本 tokenization，则直接使用 WordpieceTokenizer 进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        # 返回分割后的 token 列表
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将 token 转换为其对应的 id，如果不存在则返回 unk_token 对应的 id
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将 id 转换为对应的 token，如果不存在则返回 unk_token
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将 token 列表转换为单个字符串，去除特殊标记 " ##"
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A Lxmert sequence has the following format:

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
        # If only one sequence is provided, concatenate it with [CLS] and [SEP] tokens
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        
        # For a pair of sequences, concatenate them with [CLS], [SEP] (between sequences), and final [SEP]
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

        # If the input already contains special tokens, delegate to superclass method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Compute the mask indicating positions of special tokens in the concatenated sequence(s)
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs from token lists representing sequences.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs representing the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: A list of token type IDs where each ID corresponds to the segment ID of a token.
        """
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A Lxmert sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # Define the separator and classification tokens as lists containing the corresponding token IDs
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        # If token_ids_1 is None, return a list of zeros corresponding to the length of cls + token_ids_0 + sep
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # Otherwise, concatenate lists to form a sequence pair mask:
        #   - First sequence: cls + token_ids_0 + sep, all assigned token type ID 0
        #   - Second sequence: token_ids_1 + sep, all assigned token type ID 1
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        
        # Determine the vocabulary file path based on whether save_directory is a directory or a direct file path
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        
        # Write the vocabulary to the determined file path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # Check for non-consecutive vocabulary indices and log a warning if found
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # Write each token followed by a newline character
                writer.write(token + "\n")
                index += 1
        
        # Return the path to the saved vocabulary file as a tuple
        return (vocab_file,)
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
        """
        初始化函数，设置基本分词器的参数。

        Args:
            do_lower_case (bool, optional): 是否在分词时将输入转换为小写，默认为 True。
            never_split (Iterable, optional): 在分词过程中永远不会被分割的标记集合，默认为 None。
            tokenize_chinese_chars (bool, optional): 是否分词中文字符，默认为 True。
                对于日语可能需要禁用此选项（参见相关问题）。
            strip_accents (bool, optional): 是否去除所有重音符号。如果未指定，则由 lowercase 的值决定。
            do_split_on_punc (bool, optional): 是否进行基本的标点符号分割，默认为 True。
                在某些情况下，我们希望跳过基本的标点符号分割，以便后续的分词可以捕捉词语的完整上下文，例如缩略词。
        """
        # 如果 never_split 为 None，则设为一个空列表
        if never_split is None:
            never_split = []
        # 设置是否在分词时转换为小写
        self.do_lower_case = do_lower_case
        # 将 never_split 转换为集合，表示在分词时永远不会被分割的标记
        self.never_split = set(never_split)
        # 设置是否分词中文字符
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 设置是否去除所有重音符号
        self.strip_accents = strip_accents
        # 设置是否进行基本的标点符号分割
        self.do_split_on_punc = do_split_on_punc
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 将输入的never_split列表与实例属性self.never_split的集合进行并集操作，如果never_split为None则使用空集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本内容，去除不必要的字符或格式
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        # 如果开启了tokenize_chinese_chars选项，则对文本中的中文字符进行特定处理
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 使用Unicode规范化函数将文本标准化为NFC形式，处理Unicode中可能存在的不同编码的字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空白符分词函数对标准化后的文本进行分词，得到原始token列表
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            # 如果token不在never_split集合中，则进一步处理
            if token not in never_split:
                # 如果设置了小写处理，则将token转换为小写
                if self.do_lower_case:
                    token = token.lower()
                    # 如果需要去除重音符号，则调用私有方法_run_strip_accents处理token
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果需要去除重音符号，则调用私有方法_run_strip_accents处理token
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 使用私有方法_run_split_on_punc对token进行进一步的标点符号分割处理，加入split_tokens列表
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 使用空白符分词函数对处理后的token列表进行再次分词，得到最终的输出token列表
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 使用Unicode规范化函数将文本标准化为NFD形式，处理Unicode中可能存在的不同编码的字符
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            # 获取字符的Unicode类别
            cat = unicodedata.category(char)
            # 如果字符的类别为Mn（Mark, Nonspacing），表示为重音符号，跳过处理
            if cat == "Mn":
                continue
            # 将不含重音符号的字符添加到output列表中
            output.append(char)
        # 将处理后的字符列表拼接成字符串并返回
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要根据标点符号分割文本，或者文本在never_split列表中，则直接返回文本列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果当前字符是标点符号
            if _is_punctuation(char):
                # 在输出列表中添加新的子列表，用于存放标点符号
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，检查是否需要开始新的单词
                if start_new_word:
                    output.append([])
                start_new_word = False
                # 将当前字符添加到当前单词的子列表中
                output[-1].append(char)
            i += 1

        # 将子列表中的字符连接成字符串，并返回结果列表
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果是中日韩字符，添加空格到输出列表中
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        # 将输出列表中的字符连接成字符串，并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查给定的代码点是否属于中日韩字符的Unicode块
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
            # 如果字符是无效字符或者控制字符，直接跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果是空白字符，用单个空格替换
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将输出列表中的字符连接成字符串，并返回
        return "".join(output)
# Copied from transformers.models.bert.tokenization_bert.WordpieceTokenizer
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化 WordpieceTokenizer 类的实例
        self.vocab = vocab  # 词汇表，用于词片段的匹配
        self.unk_token = unk_token  # 未知标记，用于表示未能识别的词片段
        self.max_input_chars_per_word = max_input_chars_per_word  # 单词的最大输入字符数

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
        # 初始化空的输出词片段列表
        output_tokens = []
        # 使用 whitespace_tokenize 函数对文本进行分词
        for token in whitespace_tokenize(text):
            chars = list(token)  # 将当前分词转换为字符列表
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)  # 如果分词长度超过最大字符数，将其标记为未知标记
                continue

            is_bad = False  # 标志变量，表示当前分词是否无法分解成词片段
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                # 使用贪婪最长匹配算法寻找当前字符片段的词片段
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr  # 如果不是第一个片段，则在片段前加上 '##' 表示连接
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True  # 如果未找到匹配的词片段，则将该分词标记为无法识别
                    break
                sub_tokens.append(cur_substr)  # 将找到的词片段添加到词片段列表中
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)  # 如果无法分解成词片段，则使用未知标记替代
            else:
                output_tokens.extend(sub_tokens)  # 将词片段列表扩展到输出列表中
        return output_tokens  # 返回最终的词片段列表作为结果
```