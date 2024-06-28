# `.\models\prophetnet\tokenization_prophetnet.py`

```py
# coding=utf-8
# Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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

import collections  # 导入 collections 库，用于高效的数据结构
import os  # 导入 os 库，提供与操作系统交互的功能
import unicodedata  # 导入 unicodedata 库，用于 Unicode 字符数据的处理
from typing import Iterable, List, Optional, Tuple  # 导入类型提示相关的模块

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace  # 导入 tokenization_utils 模块中的相关函数
from ...utils import logging  # 导入 logging 模块中的 logging 函数


logger = logging.get_logger(__name__)  # 获取当前模块的 logger 对象

VOCAB_FILES_NAMES = {"vocab_file": "prophetnet.tokenizer"}  # 定义词汇文件名映射的字典

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/prophetnet-large-uncased": (
            "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/prophetnet.tokenizer"
        ),
    }
}  # 定义预训练模型对应的词汇文件映射

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/prophetnet-large-uncased": {"do_lower_case": True},  # 预训练模型的初始化配置
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/prophetnet-large-uncased": 512,  # 预训练模型的位置嵌入尺寸
}


# Copied from transformers.models.bert.tokenization_bert.whitespace_tokenize
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()  # 去除文本两端的空白字符
    if not text:
        return []  # 如果文本为空，则返回空列表
    tokens = text.split()  # 使用空格分割文本，得到词汇列表
    return tokens  # 返回分割后的词汇列表


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
    # 构造一个 BasicTokenizer 类，执行基本的分词（如分割标点符号、小写化等）

    def __init__(
        self,
        do_lower_case=True,  # 是否在分词时进行小写处理，默认为 True
        never_split=None,  # 在分词过程中永不分割的标记集合，仅在 do_basic_tokenize=True 时有效
        tokenize_chinese_chars=True,  # 是否分割中文字符，默认为 True
        strip_accents=None,  # 是否去除所有重音符号，默认根据 lowercase 的值决定（与原始 BERT 相同）
        do_split_on_punc=True,  # 是否在某些情况下跳过基本标点分割，以便后续的分词能够捕获词汇的完整上下文，如缩略词
    ):
        self.do_lower_case = do_lower_case
        self.never_split = never_split if never_split is not None else set()
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc
    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果没有指定 never_split 参数，则初始化为一个空列表
        if never_split is None:
            never_split = []
        # 设置对象的属性值
        self.do_lower_case = do_lower_case
        # 将 never_split 转换为集合类型并赋值给对象的属性
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 如果传入了 never_split 参数，则将当前对象的 never_split 属性与参数 never_split 的集合并
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清洗文本，去除不必要的字符
        text = self._clean_text(text)

        # 对于中文字符的处理，如果开启了 tokenize_chinese_chars，则进行中文字符的特殊处理
        # 这个特性最早于2018年11月1日添加，用于多语言和中文模型，现在也应用于英文模型
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        
        # 对文本进行 Unicode 规范化处理，确保相同字符的不同 Unicode 编码在处理中被视为相同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空白字符进行分词，得到原始的 token 列表
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        # 遍历原始的 token 列表，根据条件进行分割和处理
        for token in orig_tokens:
            if token not in never_split:
                # 如果开启了小写处理，则将 token 转换为小写
                if self.do_lower_case:
                    token = token.lower()
                    # 如果 strip_accents 不为 False，则移除 token 中的重音符号
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果开启了 strip_accents，则移除 token 中的重音符号
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 将处理过的 token 经过分割处理后加入 split_tokens 列表
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将分割处理后的 token 列表再次使用空白字符进行分词，得到最终的输出 token 列表
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 对文本进行 Unicode 规范化处理，将字符分解为基字符和附加记号
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的每个字符，根据字符的分类决定是否保留
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue  # 如果是 Mark, Nonspacing 类别的字符，则跳过
            output.append(char)
        # 将处理过的字符列表连接成字符串并返回
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """根据标点符号分割文本。"""
        # 如果不需要在标点符号处分割或者文本在 never_split 中，则直接返回原文本列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号，则单独作为一个列表项加入输出，并标记可以开始一个新词
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，根据是否开始新词来添加到当前最后一个列表项中
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """在每个中日韩（CJK）字符周围添加空格。"""
        output = []
        for char in text:
            cp = ord(char)
            # 如果是中日韩字符，则在其前后添加空格
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """检查给定的码点是否是中日韩字符的码点。"""
        # 这里的中日韩字符指的是CJK统一表意文字区块中的字符：
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # 需要注意，CJK统一表意文字区块并不包括所有的日文和韩文字符，
        # 现代韩文的字符属于不同的区块，日文的平假名和片假名也是如此。
        # 这些字符用于书写空格分隔的单词，因此不会特别处理，而是像其他语言一样处理。
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
        """对文本执行无效字符删除和空白字符清理。"""
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符是无效字符或控制字符，则跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果是空白字符，则用一个空格替换
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
# 从transformers.models.bert.tokenization_bert.WordpieceTokenizer复制而来，用于执行WordPiece分词的类
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化WordpieceTokenizer对象，设置词汇表、未知token和每个单词的最大输入字符数
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
        # 将文本分词为其WordPiece tokens。使用贪婪的最长匹配算法，并使用给定的词汇表进行分词
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                # 如果token长度超过最大字符数限制，则将其替换为未知token
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
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


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    # 从文件中加载词汇表到一个有序字典中
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\
")
        vocab[token] = index
    return vocab


class ProphetNetTokenizer(PreTrainedTokenizer):
    r"""
    Construct a ProphetNetTokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
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
        x_sep_token (`str`, *optional*, defaults to `"[X_SEP]"`):
            Special second separator token, which can be generated by [`ProphetNetForConditionalGeneration`]. It is
            used to separate bullet-point like sentences in summarization, *e.g.*.
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
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

    # Define constants related to vocabulary files, pretrained models, and configurations
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # Define model input names required for `tokenizer.pad(...)` to function correctly
    # For `ProphetNet`, `token_type_ids` is not a required argument.
    model_input_names: List[str] = ["input_ids", "attention_mask"]
    # 初始化方法，接受多个参数来配置分词器实例
    def __init__(
        self,
        vocab_file: str,
        do_lower_case: Optional[bool] = True,
        do_basic_tokenize: Optional[bool] = True,
        never_split: Optional[Iterable] = None,
        unk_token: Optional[str] = "[UNK]",
        sep_token: Optional[str] = "[SEP]",
        x_sep_token: Optional[str] = "[X_SEP]",
        pad_token: Optional[str] = "[PAD]",
        mask_token: Optional[str] = "[MASK]",
        tokenize_chinese_chars: Optional[bool] = True,
        strip_accents: Optional[bool] = None,
        **kwargs,
    ):
        # 检查给定的词汇文件是否存在，如果不存在则抛出异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表文件内容到实例变量中
        self.vocab = load_vocab(vocab_file)
        # 创建一个从id到token的有序字典，以便根据id查找对应的token
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 根据参数决定是否进行基本分词
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要进行基本分词，则初始化BasicTokenizer实例
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 使用给定的词汇表和未知token初始化WordpieceTokenizer实例
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的初始化方法，传递相同的参数和额外的关键字参数
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            x_sep_token=x_sep_token,
            pad_token=pad_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

    # 返回词汇表大小的属性方法
    @property
    def vocab_size(self):
        return len(self.vocab)

    # 返回包含词汇表和添加token编码器的字典
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # 对给定文本进行分词，返回分词后的token列表
    def _tokenize(self, text):
        split_tokens = []
        # 如果需要进行基本分词
        if self.do_basic_tokenize:
            # 使用BasicTokenizer分词器对文本进行分词
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # 如果token在never_split集合中，则直接添加到分词结果列表中
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 否则，使用WordpieceTokenizer对token进行进一步分词，并将结果扩展到split_tokens列表中
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 如果不需要基本分词，则直接使用WordpieceTokenizer对文本进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    # 根据token查找其在词汇表中对应的id，如果不存在则返回unk_token对应的id
    def _convert_token_to_id(self, token: str):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 根据id查找其在词汇表中对应的token，如果不存在则返回unk_token
    def _convert_id_to_token(self, index: int):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)
    def convert_tokens_to_string(self, tokens: str):
        """
        Converts a sequence of tokens (string) into a single string.
        Args:
            tokens (`str`): A sequence of tokens.

        Returns:
            `str`: The concatenated string without '##' symbols.
        """
        # Join tokens into a single string, remove '##' and strip leading/trailing spaces
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: Optional[bool] = False,
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`): List of IDs.
            token_ids_1 (`List[int]`, *optional*): Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*): Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            # Return a list of zeros of the same length as token_ids_0, with a single 1 appended
            return ([0] * len(token_ids_0)) + [1]
        else:
            # Return a list of zeros of the combined length of token_ids_0 and token_ids_1, each followed by a 1
            return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A ProphetNet
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`): List of IDs.
            token_ids_1 (`List[int]`, *optional*): Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            # Return a list of zeros with a length equal to the sum of token_ids_0 and one separator token
            return len(token_ids_0 + sep) * [0]
        else:
            # Return a list of zeros with a length equal to the combined sum of token_ids_0, token_ids_1, and two separator tokens
            return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    # 将词汇表保存到指定目录下的文件中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引
        index = 0
        # 检查保存目录是否存在
        if os.path.isdir(save_directory):
            # 构建词汇表文件路径，包括可选的文件名前缀和默认的词汇表文件名
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 如果保存目录不存在，则直接将其作为文件路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开文件，写入词汇表内容
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表中的每个词汇和对应的索引
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 检查当前索引是否连续
                if index != token_index:
                    # 如果不连续，记录警告信息
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    # 更新索引
                    index = token_index
                # 将词汇写入文件，每个词汇后面加上换行符
                writer.write(token + "\n")
                # 更新索引
                index += 1
        # 返回保存的文件路径，以元组形式返回
        return (vocab_file,)

    # 构建包含特殊标记的模型输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过连接和添加特殊标记，从序列或序列对构建用于序列分类任务的模型输入。BERT 序列的格式如下：

        - 单个序列：`[CLS] X [SEP]`
        - 序列对：`[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                将要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列对的 ID 列表（可选）。

        Returns:
            `List[int]`: 包含适当特殊标记的输入 ID 列表。
        """
        # 如果没有第二个序列对，则直接返回第一个序列加上分隔标记的结果
        if token_ids_1 is None:
            return token_ids_0 + [self.sep_token_id]
        # 构造分隔标记列表
        sep = [self.sep_token_id]
        # 返回连接后的两个序列及其之间的分隔标记列表
        return token_ids_0 + sep + token_ids_1 + sep
```