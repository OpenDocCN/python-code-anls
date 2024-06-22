# `.\models\deprecated\retribert\tokenization_retribert.py`

```py
# coding=utf-8
# 版权 2018 年的 HuggingFace 公司团队
#
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在下面的网址获得许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发，
# 没有任何担保或条件，无论是明示的还是暗示的。
# 有关详细信息，请参阅许可证。

"""RetriBERT 的标记化类。"""

import collections
import os
import unicodedata
from typing import List, Optional, Tuple

# 导入必要的模块
from ....tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ....utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 词汇表文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "yjernite/retribert-base-uncased": (
            "https://huggingface.co/yjernite/retribert-base-uncased/resolve/main/vocab.txt"
        ),
    }
}

# 预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "yjernite/retribert-base-uncased": 512,
}

# 预训练初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "yjernite/retribert-base-uncased": {"do_lower_case": True},
}

# 从 transformers.models.bert.tokenization_bert.load_vocab 复制而来
def load_vocab(vocab_file):
    """将词汇表文件加载到字典中。"""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

# 从 transformers.models.bert.tokenization_bert.whitespace_tokenize 复制而来
def whitespace_tokenize(text):
    """对文本进行基本的空白字符清理和拆分。"""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

class RetriBertTokenizer(PreTrainedTokenizer):
    r"""
    构建一个 RetriBERT 分词器。

    [`RetriBertTokenizer`] 与 [`BertTokenizer`] 相同，并运行端到端的分词：标点符号拆分和 wordpiece。

    此分词器继承自 [`PreTrainedTokenizer`]，其中包含大多数主要方法。用户应参考
    这个超类以获取有关这些方法的更多信息。
    # 参数说明：
    # vocab_file (`str`): 包含词汇表的文件。
    # do_lower_case (`bool`, *optional*, defaults to `True`): 在进行标记化时是否将输入转换为小写。
    # do_basic_tokenize (`bool`, *optional*, defaults to `True`): 在 WordPiece 之前是否进行基本标记化。
    # never_split (`Iterable`, *optional*): 在标记化时永远不会被拆分的令牌集合。仅在 `do_basic_tokenize=True` 时有效。
    # unk_token (`str`, *optional*, defaults to `"[UNK]"`): 未知的标记。不在词汇表中的标记无法转换为 ID，并被设置为这个标记。
    # sep_token (`str`, *optional*, defaults to `"[SEP]"`): 分隔符标记，在构建来自多个序列的序列时使用，例如用于序列分类的两个序列，或用于问题回答的文本和问题。在使用特殊标记构建的序列的最后一个标记也会使用它。
    # pad_token (`str`, *optional*, defaults to `"[PAD]"`): 用于填充的标记，例如在批处理不同长度的序列时使用。
    # cls_token (`str`, *optional*, defaults to `"[CLS]"`): 分类器标记，在进行序列分类时使用（而不是对每个标记进行分类）。在使用特殊标记构建的序列的第一个标记。
    # mask_token (`str`, *optional*, defaults to `"[MASK]"`): 用于屏蔽值的标记。在使用掩码语言建模训练此模型时使用。这是模型将尝试预测的标记。
    # tokenize_chinese_chars (`bool`, *optional*, defaults to `True`): 是否标记化中文字符。这可能应该在日语中停用（参见此问题）。
    # strip_accents (`bool`, *optional*): 是否去除所有重音符号。如果未指定此选项，则将由 `lowercase` 的值确定（与原始 BERT 一样）。

    # 定义变量
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    model_input_names = ["input_ids", "attention_mask"]

    # 从transformers.models.bert.tokenization_bert.BertTokenizer.__init__中复制而来
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,  # 是否将输入文本都转换成小写，默认为 True
        do_basic_tokenize=True,  # 是否进行基本的分词，默认为 True
        never_split=None,  # 永远不需要分割的特殊词列表，默认为 None
        unk_token="[UNK]",  # 未知词的token，默认为 "[UNK]"
        sep_token="[SEP]",  # 分隔符token，默认为 "[SEP]"
        pad_token="[PAD]",  # 填充token，默认为 "[PAD]"
        cls_token="[CLS]",  # 分类token，默认为 "[CLS]"
        mask_token="[MASK]",  # 掩码token，默认为 "[MASK]"
        tokenize_chinese_chars=True,  # 是否对中文进行分词，默认为 True
        strip_accents=None,  # 是否去掉重音符号，默认为 None
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):  # 如果给定的词汇表文件不存在
            raise ValueError(  # 抛出数值错误
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)  # 载入词汇表
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])  # 将词汇表转换为 token 到 id 的字典
        self.do_basic_tokenize = do_basic_tokenize  # 设置是否进行基本分词
        if do_basic_tokenize:  # 如果需要进行基本分词
            self.basic_tokenizer = BasicTokenizer(  # 初始化基本分词器
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))  # 初始化基于词汇表的分词器

        super().__init__(  # 调用父类的初始化方法
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
    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.do_lower_case
    def do_lower_case(self):  # 获取是否将输入文本都转换成小写的属性
        return self.basic_tokenizer.do_lower_case

    @property
    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.vocab_size
    def vocab_size(self):  # 获取词汇表的大小
        return len(self.vocab)

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.get_vocab
    def get_vocab(self):  # 获取词汇表
        return dict(self.vocab, **self.added_tokens_encoder)

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer._tokenize
    def _tokenize(self, text, split_special_tokens=False):  # 对文本进行分词
        split_tokens = []  # 初始化分词结果列表
        if self.do_basic_tokenize:  # 如果需要进行基本分词
            for token in self.basic_tokenizer.tokenize(  # 遍历基本分词器对文本进行分词
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # If the token is part of the never_split set
                if token in self.basic_tokenizer.never_split:  # 如果分割后的 token 在不需要分割的特殊词列表中
                    split_tokens.append(token)  # 直接添加到结果列表
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)  # 否则继续使用基于词汇表的分词器进行分词
        else:  # 如果不需要进行基本分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)  # 直接使用基于词汇表的分词器进行分词
        return split_tokens  # 返回分词结果
    # 根据词汇表将一个token转换为对应的id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))
    
    # 根据词汇表将一个id转换为对应的token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)
    
    # 将一系列token转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string
    
    # 构建包含特殊标记的模型输入
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
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep
        
    # 获取特殊标记的掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Get the token mask indicating whether each token is a special token (e.g. [CLS], [SEP]) or not.
    
        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`):
                List of IDs for the second sequence.
            already_has_special_tokens (`bool`, *optional*):
                Whether the token list is already formatted with special tokens or not.
    
        Returns:
            `List[int]`: List of 0s and 1s, with 1s indicating the special tokens.
        """
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

        # If token list already has special tokens, call super method to get special tokens mask
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If token list has two sequences, form the special tokens mask accordingly
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # If token list has only one sequence, form the special tokens mask accordingly
        return [1] + ([0] * len(token_ids_0)) + [1]

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
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
        sep = [self.sep_token_id]  # Get the separator token ID
        cls = [self.cls_token_id]  # Get the classification token ID
        # If there is only one sequence, return token type IDs for the first sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # If there are two sequences, return token type IDs for both sequences
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.save_vocabulary
    # 保存词汇表到指定目录下，返回保存文件路径的元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引值
        index = 0
        # 如果保存目录已存在
        if os.path.isdir(save_directory):
            # 构建词汇表文件路径
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 构建词汇表文件路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开词汇表文件，以写入模式，并指定编码格式为utf-8
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表中的token及其索引，按索引升序排列
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 如果索引不连续，输出警告信息
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 将token写入词汇表文件并换行
                writer.write(token + "\n")
                index += 1
        # 返回保存的词汇表文件路径的元组
        return (vocab_file,)
# 定义一个 BasicTokenizer 类，用于运行基本的分词（如标点符号分割、转换为小写等）

class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).
    
    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在分词时将输入转换为小写
        never_split (`Iterable`, *optional*):
            在分词时不会被分割的 token 集合。仅在 `do_basic_tokenize=True` 时生效
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否对中文字符进行分词。对于日语，这个选项可能需要关闭
        strip_accents (`bool`, *optional*):
            是否去除所有的重音符号。如果没有指定该选项，则根据 `lowercase` 的值来确定（与原始的 BERT 一样）
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            在某些情况下，我们希望跳过基本的标点符号分割，以便后续的分词能够捕捉单词的完整上下文，比如缩略词
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果 never_split 没有指定，则设为一个空列表
        if never_split is None:
            never_split = []
        
        # 设置类的属性
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc
    # 定义一个名为 tokenize 的方法
    def tokenize(self, text, never_split=None):
        """
        执行基本的文本分词。对于亚词分词，请参见 WordPieceTokenizer。
    
        参数:
            never_split (`List[str]`, *可选*)
                保持向后兼容性。现在已经在基类级别直接实现（请参见 `PreTrainedTokenizer.tokenize`）。不应该分割的令牌列表。
        """
        # 将 never_split 参数与实例属性 never_split 合并成一个新的集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本
        text = self._clean_text(text)
    
        # 这个功能是在 2018 年 11 月 1 日为多语言和中文模型添加的。现在也应用于英语模型,但这并不重要,因为英语模型没有被训练在任何中文数据上,
        # 并且通常也没有任何中文数据（维基百科中的英语页面中有一些中文词汇,所以词汇表中也有一些中文字符）。
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 防止将具有不同 Unicode 代码点的相同字符视为不同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空白符分割文本为原始令牌
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        # 遍历原始令牌
        for token in orig_tokens:
            # 如果令牌不在 never_split 列表中
            if token not in never_split:
                # 如果需要转换为小写
                if self.do_lower_case:
                    token = token.lower()
                    # 如果需要去除重音符号
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果需要去除重音符号
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 将分割后的令牌添加到 split_tokens 列表
            split_tokens.extend(self._run_split_on_punc(token, never_split))
        # 使用空白符重新连接分割后的令牌,并进行分词
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens
    
    # 定义一个名为 _run_strip_accents 的私有方法
    def _run_strip_accents(self, text):
        """删除文本中的重音符号。"""
        # 将文本标准化为 NFC 形式
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 类别
            cat = unicodedata.category(char)
            # 如果类别为 Mn (结合记号),则跳过该字符
            if cat == "Mn":
                continue
            # 否则将该字符添加到输出列表
            output.append(char)
        # 将输出列表拼接成字符串并返回
        return "".join(output)
    # 在文本上分割标点符号
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要分割标点或者文本在不分割列表中，则直接返回文本
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        # 遍历文本中的字符
        while i < len(chars):
            char = chars[i]
            # 如果字符是标点符号，则单独放到一个列表里，表示分割
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果是非标点符号，判断是否需要新的单词
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    # 给中文字符添加空白
    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        # 遍历文本中的字符
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    # 检查是否为中文字符
    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 这里将中文字符定义为在CJK Unicode块中的字符范围内
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):
            return True

        return False

    # 清除文本中的无效字符和空白
    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        # 遍历文本中的字符
        for char in text:
            cp = ord(char)
            # 如果字符是无效字符或控制字符，则跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果是空白字符，替换为空格���否则保留字符
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
# 从transformers.models.bert.tokenization_bert.WordpieceTokenizer中复制的类
class WordpieceTokenizer(object):
    """执行WordPiece标记化。"""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化WordPiece标记化器，使用给定的词汇表、未知标记和每个单词的最大输入字符数
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        将一段文本标记化为其词片段。这使用贪婪最长匹配算法使用给定的词汇表进行标记化。

        例如，`input = "unaffable"`将返回输出`["un", "##aff", "##able"]`。

        Args:
            text: 一个单个标记或以空格分隔的标记。这应该已经通过*BasicTokenizer*。

        Returns:
            一个词片段标记列表。
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
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
```