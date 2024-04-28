# `.\transformers\models\lxmert\tokenization_lxmert.py`

```py
# 该文件实现了 LXMERT 模型的 Tokenizer 类
# 引入必要的库和模块
import collections
import os
import unicodedata
from typing import List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 vocab.txt 文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 定义预训练模型 vocab 文件的下载 URL
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "unc-nlp/lxmert-base-uncased": "https://huggingface.co/unc-nlp/lxmert-base-uncased/resolve/main/vocab.txt",
    }
}

# 定义预训练模型 Positional Embeddings 的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "unc-nlp/lxmert-base-uncased": 512,
}

# 定义预训练模型的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "unc-nlp/lxmert-base-uncased": {"do_lower_case": True},
}

# 定义加载词汇表的函数
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

# 定义简单的空格分词函数
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

# 定义 LxmertTokenizer 类，继承自 PreTrainedTokenizer
class LxmertTokenizer(PreTrainedTokenizer):
    r"""
    Construct a Lxmert tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            词汇表文件路径。
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在进行标记化时将输入转换为小写。
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            是否在 WordPiece 标记化之前执行基本标记化。
        never_split (`Iterable`, *optional*):
            在标记化过程中永远不会拆分的标记集合。仅在 `do_basic_tokenize=True` 时生效。
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            未知标记。不在词汇表中的标记无法转换为 ID，并将其设置为此标记。
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            分隔符标记，用于从多个序列构建序列，例如，用于序列分类或用于文本和问题的问答。还用作带有特殊标记构建的序列的最后一个标记。
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            用于填充的标记，例如，在批处理不同长度的序列时使用。
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            分类器标记，用于进行序列分类（对整个序列进行分类，而不是对每个标记进行分类）。它是使用特殊标记构建的序列的第一个标记。
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            用于掩码值的标记。这是在进行掩码语言建模时使用的标记。这是模型将尝试预测的标记。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否标记化中文字符。

            这对于日语可能应该被禁用（参见此 [问题](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否去除所有重音符号。如果未指定此选项，则将根据 `lowercase` 的值确定（与原始 Lxmert 一样）。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

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
    # 定义 LxmertTokenizer 类，继承自 PreTrainedTokenizer
    ):
        # 如果给定的 vocabulary 文件不存在，则抛出异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = LxmertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载 vocabulary，并存储在 self.vocab 中
        self.vocab = load_vocab(vocab_file)
        # 根据 vocabulary 创建从 ids 到 tokens 的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 设置是否进行基本标记化
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要进行基本标记化，则创建 BasicTokenizer 对象
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        # 创建 WordpieceTokenizer 对象，使用给定的 unk_token
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类 PreTrainedTokenizer 的初始化函数，传入所需参数
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

    # 返回是否执行小写转换的属性
    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    # 返回 vocabulary 大小的属性
    @property
    def vocab_size(self):
        return len(self.vocab)

    # 获取所有 tokens 映射的字典
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # 对文本进行标记化，返回标记化后的 tokens 列表
    def _tokenize(self, text, split_special_tokens=False):
        split_tokens = []
        # 如果需要基本标记化
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果 token 在 never_split 集合中
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    # 根据 token 返回对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 根据 id 返回对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    # 将 token 序列转换为字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # 构建带有特殊标记的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从序列或序列对构建模型输入，用于序列分类任务，通过连接和添加特殊标记。一个 Lxmert 序列具有以下格式：

        - 单个序列：`[CLS] X [SEP]`
        - 序列对：`[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                序列对的第二个 ID 列表（可选）。

        Returns:
            `List[int]`: 具有适当特殊标记的[input IDs](../glossary#input-ids)列表。
        """
        # 如果没有第二个序列对应的 token_ids_1，则返回只有一个序列的特殊标记
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 创建包含特殊标记的列表：CLS 标记、第一个序列的 token_ids_0、SEP 标记、第二个序列的 token_ids_1、SEP 标记
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的 token 列表中检索序列 ID。当使用 tokenizer `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                序列对的第二个 ID 列表（可选）。
            already_has_special_tokens (`bool`, *可选*, 默认为 `False`):
                token 列表是否已经使用特殊标记格式化为模型。

        Returns:
            `List[int]`: 一个整数列表，范围在[0, 1]之间：特殊标记为 1，序列标记为 0。
        """

        # 如果已经包含特殊标记，则调用父类的方法
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果有第二个序列对应的 token_ids_1，则返回特殊标记在 token_ids_0 和 token_ids_1 中的位置
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # 否则只返回第一个序列的特殊标记位置
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A Lxmert sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```py

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # 设置分隔符的 ID 列表
        sep = [self.sep_token_id]
        # 设置 CLS 标记的 ID 列表
        cls = [self.cls_token_id]
        # 如果第二个序列的 token_ids_1 为空，仅返回第一部分的 mask (0s)
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 返回包含两个序列的 mask，第一个序列部分的 token type ID 为 0，第二个序列部分的 token type ID 为 1
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引
        index = 0
        # 如果保存目录已存在
        if os.path.isdir(save_directory):
            # 构建词汇文件路径
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 构建词汇文件路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开词汇文件进行写操作
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表中的 token 和对应的索引
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 检查索引是否连续，如果不连续则发出警告
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 将 token 写入文件
                writer.write(token + "\n")
                index += 1
        # 返回保存的词汇文件路径
        return (vocab_file,)
# 从transformers.models.bert.tokenization_bert.BasicTokenizer复制而来的类
class BasicTokenizer(object):
    """
    构造一个 BasicTokenizer，用于运行基本的分词（标点符号分割，小写处理等）。

    Args:
        do_lower_case (`bool`, *optional*, 默认为 `True`):
            当分词时是否将输入转换为小写。
        never_split (`Iterable`, *optional*):
            在分词过程中永远不会被分割的标记集合。仅当`do_basic_tokenize=True`时才会生效。
        tokenize_chinese_chars (`bool`, *optional*, 默认为 `True`):
            是否对中文字符进行分词。

            对于日语，这可能需要停用（参见此[issue](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否删除所有重音符号。如果未指定此选项，则将由`lowercase`的值来确定（与原始BERT相同）。
        do_split_on_punc (`bool`, *optional*, 默认为 `True`):
            在某些情况下，我们希望跳过基本的标点符号分割，以便后续的分词可以捕捉单词的完整上下文，比如缩写。

    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果没有提供never_split参数，将其设为一个空列表
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)  # 将传入的never_split转化为集合
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc
    # 对文本进行基本的分词处理，如果需要进行子词分词，则使用WordPieceTokenizer
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 如果存在never_split参数，则将其与self.never_split取并集
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，去除不需要的字符
        text = self._clean_text(text)

        # 2018年11月1日添加，用于多语言和中文模型。现在也应用于英语模型，但是这不重要因为英语模型没有训练过中文数据
        # 通常不会包含中文数据（英文维基百科中有一些中文词汇）。
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 防止将具有不同Unicode码点的同一字符视为不同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                # 如果需要小写处理，则将token转换为小写
                if self.do_lower_case:
                    token = token.lower()
                    # 如果需要去除重音符号，则执行去除操作
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果需要去除重音符号，则执行去除操作
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 将处理过的token合并到split_tokens列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        # 返回处理后的token列表
        return output_tokens

    # 去除文本中的重音符号
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            # 如果字符属于"Mark, Nonspacing"类型，则跳过
            if cat == "Mn":
                continue
            # 将字符添加到output列表中
            output.append(char)
        # 将列表中的字符连接成字符串并返回
        return "".join(output)
    # 对文本进行标点符号分割
    def _run_split_on_punc(self, text, never_split=None):
        # 如果不需要进行标点符号分割，或者指定的文本不需要分割，则直接返回文本
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换成字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号，则将其单独放入列表中
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果是非标点符号，则根据标志位进行处理
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        # 将列表中的字符重新组合成字符串并返回
        return ["".join(x) for x in output]
    
    def _tokenize_chinese_chars(self, text):
        """给中文字符两侧添加空格"""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
    
    def _is_chinese_char(self, cp):
        """检查CP是否是CJK字符的代码点"""
        # 这里将 "chinese character" 定义为在CJK Unicode块中的任何字符
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
        """对文本进行无效字符去除和空格清理"""
        output = []
        for char in text:
            cp = ord(char)
            # 如果是无效字符则直接跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果是空格字符，则替换成空格
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
# 从transformers.models.bert.tokenization_bert中复制了WordpieceTokenizer类
class WordpieceTokenizer(object):
    """运行WordPiece标记化。"""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化WordpieceTokenizer对象
        self.vocab = vocab
        # 未知标记
        self.unk_token = unk_token
        # 每个单词最大输入字符数
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        将文本标记化为其单词片段。这使用贪婪的最长匹配算法来使用给定的词汇表进行标记化。

        例如，`input = "unaffable"`将作为输出返回`["un", "##aff", "##able"]`。

        Args:
            text: 单个标记或以空格分隔的标记。这应该已经通过*BasicTokenizer*。

        Returns:
            单词片段标记的列表。
        """

        # 初始化输出标记列表
        output_tokens = []
        # 对文本中的每个标记进行循环
        for token in whitespace_tokenize(text):
            # 将标记分成字符列表
            chars = list(token)
            # 如果字符数超过最大输入字符数，则将未知标记添加到输出标记列表，并继续下一个标记的处理
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            # 初始化变量以追踪处理的标记是否出现问题
            is_bad = False
            # 初始化标记的起始索引
            start = 0
            # 初始化子标记列表
            sub_tokens = []
            # 当起始索引小于字符列表的长度时，循环处理标记的每个子标记
            while start < len(chars):
                # 初始化结束索引
                end = len(chars)
                # 初始化当前子字符串
                cur_substr = None
                # 当起始索引小于结束索引时，循环尝试从标记的字符中提取子字符串
                while start < end:
                    # 提取从起始索引到结束索引的子字符串
                    substr = "".join(chars[start:end])
                    # 如果起始索引大于0，则在子字符串前加上"##"
                    if start > 0:
                        substr = "##" + substr
                    # 如果子字符串在词汇表中，则将其设置为当前子字符串并退出循环
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    # 否则，将结束索引减小以尝试更短的子字符串
                    end -= 1
                # 如果未找到可用的子字符串，则将标记标记为出现问题
                if cur_substr is None:
                    is_bad = True
                    break
                # 将当前子字符串添加到子标记列表中
                sub_tokens.append(cur_substr)
                # 更新起始索引以处理下一个子标记
                start = end

            # 如果标记出现问题，则将未知标记添加到输出标记列表，否则将子标记列表添加到输出标记列表中
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        # 返回标记化后的输出标记列表
        return output_tokens
```