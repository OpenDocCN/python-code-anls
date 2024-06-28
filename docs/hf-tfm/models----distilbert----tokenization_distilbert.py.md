# `.\models\distilbert\tokenization_distilbert.py`

```py
# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Tokenization classes for DistilBERT."""

import collections
import os
import unicodedata
from typing import List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练模型对应的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "distilbert-base-uncased": "https://huggingface.co/distilbert-base-uncased/resolve/main/vocab.txt",
        "distilbert-base-uncased-distilled-squad": (
            "https://huggingface.co/distilbert-base-uncased-distilled-squad/resolve/main/vocab.txt"
        ),
        "distilbert-base-cased": "https://huggingface.co/distilbert-base-cased/resolve/main/vocab.txt",
        "distilbert-base-cased-distilled-squad": (
            "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/vocab.txt"
        ),
        "distilbert-base-german-cased": "https://huggingface.co/distilbert-base-german-cased/resolve/main/vocab.txt",
        "distilbert-base-multilingual-cased": (
            "https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/vocab.txt"
        ),
    }
}

# 预训练模型对应的位置编码大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "distilbert-base-uncased": 512,
    "distilbert-base-uncased-distilled-squad": 512,
    "distilbert-base-cased": 512,
    "distilbert-base-cased-distilled-squad": 512,
    "distilbert-base-german-cased": 512,
    "distilbert-base-multilingual-cased": 512,
}

# 预训练模型初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "distilbert-base-uncased": {"do_lower_case": True},
    "distilbert-base-uncased-distilled-squad": {"do_lower_case": True},
    "distilbert-base-cased": {"do_lower_case": False},
    "distilbert-base-cased-distilled-squad": {"do_lower_case": False},
    "distilbert-base-german-cased": {"do_lower_case": False},
    "distilbert-base-multilingual-cased": {"do_lower_case": False},
}

# 从 transformers.models.bert.tokenization_bert.load_vocab 复制而来的函数
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    # 使用 OrderedDict 来存储词汇表
    vocab = collections.OrderedDict()
    # 以 UTF-8 编码读取词汇文件
    with open(vocab_file, "r", encoding="utf-8") as reader:
        # 逐行读取词汇文件内容
        tokens = reader.readlines()
    # 将每个词汇添加到 vocab 字典中，并用其在文件中的顺序作为值
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab
# Copied from transformers.models.bert.tokenization_bert.whitespace_tokenize
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # 去除文本两端的空白字符
    text = text.strip()
    # 如果处理后文本为空，则返回空列表
    if not text:
        return []
    # 使用空格分割文本，得到 token 列表
    tokens = text.split()
    # 返回分割后的 token 列表
    return tokens


class DistilBertTokenizer(PreTrainedTokenizer):
    r"""
    Construct a DistilBERT tokenizer. Based on WordPiece.

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
            value for `lowercase` (as in the original BERT).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    # 加载预训练模型的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 加载预训练模型的初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 加载预训练模型的最大输入大小配置
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型的输入名称列表
    model_input_names = ["input_ids", "attention_mask"]

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
        # 检查给定的词汇文件是否存在，否则抛出值错误
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = DistilBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表
        self.vocab = load_vocab(vocab_file)
        # 构建从标识符到词汇的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 设置是否进行基本的分词操作
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要进行基本分词，则初始化基本分词器
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 初始化基于词汇表的 WordPiece 分词器
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的初始化方法，传递相同的参数
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
    # 返回基本分词器是否进行小写处理的属性
    # 来自 transformers.models.bert.tokenization_bert.BertTokenizer.do_lower_case
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    # 返回词汇表的大小
    # 来自 transformers.models.bert.tokenization_bert.BertTokenizer.vocab_size
    def vocab_size(self):
        return len(self.vocab)

    # 返回词汇表及其附加编码器的字典
    # 来自 transformers.models.bert.tokenization_bert.BertTokenizer.get_vocab
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # 将文本标记化为子词的方法，这个方法会被具体的子类实现
    # 来自 transformers.models.bert.tokenization_bert.BertTokenizer._tokenize
    def _tokenize(self, text, split_special_tokens=False):
        # 初始化空列表，用于存储分词后的 tokens
        split_tokens = []
        # 如果需要进行基本的分词处理
        if self.do_basic_tokenize:
            # 使用 basic_tokenizer 对文本进行分词
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果 token 在 never_split 集合中
                if token in self.basic_tokenizer.never_split:
                    # 直接添加到 split_tokens 中
                    split_tokens.append(token)
                else:
                    # 否则，使用 wordpiece_tokenizer 进一步分词，并将结果合并到 split_tokens 中
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 如果不需要基本的分词处理，直接使用 wordpiece_tokenizer 进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        # 返回分词后的 tokens 列表
        return split_tokens

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer._convert_token_to_id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用 vocab 字典将 token 转换为对应的 id，如果 token 不在 vocab 中，则使用 unk_token 对应的 id
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用 ids_to_tokens 字典将 index 转换为对应的 token，如果 index 不在 ids_to_tokens 中，则返回 unk_token
        return self.ids_to_tokens.get(index, self.unk_token)

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将 tokens 列表中的 token 连接成一个字符串，并移除特殊标记 ' ##'，最后去除首尾的空格
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.build_inputs_with_special_tokens
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
        # 如果没有提供 token_ids_1，则构建单个序列的输入列表，包括特殊 token `[CLS]` 和 `[SEP]`
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 否则，构建序列对的输入列表，包括两个序列的特殊 token `[CLS]`、`[SEP]` 以及分隔符 `[SEP]`
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
        """
        Retrieve sequence ids from a sequence of tokens that should not be masked.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens.

        Returns:
            `List[int]`: List of indices indicating which tokens are special tokens.
        """
        # 如果已经有特殊 token，则直接返回全为 1 的掩码列表，长度为 token_ids_0 的长度
        if already_has_special_tokens:
            return [1] * len(token_ids_0)
        # 否则，构建一个掩码列表，长度为 token_ids_0 的长度加上特殊 token `[CLS]` 和 `[SEP]` 的长度
        # 并设置特殊 token 对应位置为 1，其余位置为 0
        cls_sep = [self.cls_token_id, self.sep_token_id]
        return list(map(lambda x: 1 if x in cls_sep else 0, token_ids_0))
    # 从不包含特殊token的token列表中提取序列id。当使用tokenizer的`prepare_for_model`方法添加特殊token时调用此方法。
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False) -> List[int]:
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
            # 如果token列表已包含特殊token，则调用父类的方法获取特殊token的掩码
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            # 如果有第二个token列表，则返回一个包含特殊token的掩码列表
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # 如果只有一个token列表，则返回一个包含特殊token的掩码列表
        return [1] + ([0] * len(token_ids_0)) + [1]

    # 从给定的序列创建token类型ID的方法，用于序列对分类任务
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
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
        sep = [self.sep_token_id]  # 分隔token的ID列表
        cls = [self.cls_token_id]  # 类别开始token的ID列表
        if token_ids_1 is None:
            # 如果没有第二个token列表，只返回第一个序列部分的token类型ID列表（全为0）
            return len(cls + token_ids_0 + sep) * [0]
        # 如果有第二个token列表，返回两个序列的token类型ID列表，第一个序列为0，第二个序列为1
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    # 定义保存词汇表的方法，接受保存目录和可选的文件名前缀作为参数，并返回保存的文件名元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引
        index = 0
        # 检查保存目录是否存在，构建词汇表文件路径
        if os.path.isdir(save_directory):
            # 如果保存目录是一个目录，则在该目录下创建词汇表文件路径
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 如果保存目录是一个文件路径，则直接使用该路径作为词汇表文件路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开词汇表文件，准备写入
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表中的每个词汇及其索引，按索引排序
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 如果当前索引不等于预期的索引值，记录警告日志
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 写入词汇到文件，并添加换行符
                writer.write(token + "\n")
                # 更新索引值
                index += 1
        # 返回保存的词汇表文件路径的元组
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
        # 如果 `never_split` 参数未提供，将其设为空列表
        if never_split is None:
            never_split = []
        # 设定是否将输入内容全部转换为小写
        self.do_lower_case = do_lower_case
        # 将 `never_split` 转换为集合，这些标记在分词时不会被分开
        self.never_split = set(never_split)
        # 设定是否对中文字符进行分词处理
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 设定是否去除所有的重音符号
        self.strip_accents = strip_accents
        # 设定是否在标点符号处进行基础分词
        self.do_split_on_punc = do_split_on_punc
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        # 如果给定了 `never_split` 参数，则将其转换为集合并与 `self.never_split` 取并集，否则直接使用 `self.never_split`
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        
        # 清理文本，去除不必要的字符
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        # 如果开启了 tokenize_chinese_chars 参数，则对文本中的中文字符进行特殊处理
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        
        # 将文本进行 Unicode 规范化为 NFC 格式，确保字符的一致性
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        
        # 将规范化后的文本按空白字符进行分词
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        
        # 对每个 token 进行处理
        for token in orig_tokens:
            # 如果 token 不在 never_split 中，则继续处理
            if token not in never_split:
                # 如果开启了小写化处理，则将 token 转换为小写
                if self.do_lower_case:
                    token = token.lower()
                    # 如果开启了去除重音处理，则去除 token 的重音
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果开启了去除重音处理，则去除 token 的重音
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            
            # 将处理后的 token 再进行标点符号分割处理，并加入到 split_tokens 中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将处理后的分词按空白字符再次分割，并返回最终的输出 tokens
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 将文本进行 Unicode 规范化为 NFD 格式，分解字符为基字符和附加记号
        text = unicodedata.normalize("NFD", text)
        output = []
        
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 分类
            cat = unicodedata.category(char)
            # 如果字符的分类是 Mark, Nonspacing，则跳过该字符，不加入输出
            if cat == "Mn":
                continue
            # 否则将字符加入输出列表
            output.append(char)
        
        # 将列表中的字符连接成字符串并返回
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要在标点处分割或者文本在never_split列表中，直接返回文本列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号，则将其作为单独的列表项加入output，并标记可以开始一个新单词
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，则将字符添加到当前列表项中
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        # 将列表中的子列表合并为字符串，并返回分割后的文本列表
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果是中文字符，则在其前后添加空格，并加入到输出列表中
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                # 如果不是中文字符，则直接添加到输出列表中
                output.append(char)
        # 将输出列表转换为字符串，并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查Unicode码点是否位于CJK统一表意文字区块中，返回是否是中文字符的布尔值
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
            # 如果字符为无效字符或控制字符，跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果是空白字符，则替换为单个空格；否则保留原字符
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将输出列表转换为字符串，并返回清理后的文本
        return "".join(output)
# Copied from transformers.models.bert.tokenization_bert.WordpieceTokenizer
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化 WordpieceTokenizer 类的实例
        self.vocab = vocab  # 词汇表，用于词片段(token)的匹配
        self.unk_token = unk_token  # 未知标记，用于替换无法识别的词片段
        self.max_input_chars_per_word = max_input_chars_per_word  # 单个词的最大字符数

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
        output_tokens = []  # 存储最终的词片段(token)列表
        for token in whitespace_tokenize(text):  # 对输入文本进行空白字符分割，并遍历每个分割后的单词
            chars = list(token)  # 将单词转换为字符列表
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)  # 如果单词字符数超过最大限制，则使用未知标记替代
                continue

            is_bad = False  # 标记是否出现无法识别的子词
            start = 0  # 初始化子词起始位置
            sub_tokens = []  # 存储当前单词分割后的词片段
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])  # 获取当前起始位置到结束位置的子字符串
                    if start > 0:
                        substr = "##" + substr  # 对非初始子词添加 ## 前缀
                    if substr in self.vocab:  # 如果找到匹配的词片段在词汇表中
                        cur_substr = substr  # 记录当前词片段
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True  # 如果找不到匹配的词片段，则标记为无法识别
                    break
                sub_tokens.append(cur_substr)  # 将找到的词片段添加到子词列表中
                start = end  # 更新起始位置为当前结束位置

            if is_bad:
                output_tokens.append(self.unk_token)  # 如果整个单词无法识别，则使用未知标记替代
            else:
                output_tokens.extend(sub_tokens)  # 将识别出的词片段添加到最终结果中
        return output_tokens  # 返回最终的词片段列表
```