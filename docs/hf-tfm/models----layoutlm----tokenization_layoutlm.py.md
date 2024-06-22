# `.\models\layoutlm\tokenization_layoutlm.py`

```py
# 设置编码为 utf-8
# 版权说明

# 导入必要的库和模块
import collections
import os
import unicodedata
from typing import List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/layoutlm-base-uncased": (
            "https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main/vocab.txt"
        ),
        "microsoft/layoutlm-large-uncased": (
            "https://huggingface.co/microsoft/layoutlm-large-uncased/resolve/main/vocab.txt"
        ),
    }
}

# 预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/layoutlm-base-uncased": 512,
    "microsoft/layoutlm-large-uncased": 512,
}

# 预训练模型的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/layoutlm-base-uncased": {"do_lower_case": True},
    "microsoft/layoutlm-large-uncased": {"do_lower_case": True},
}

# 从词汇文件加载词汇表的方法
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

# 基本的按空格分词方法
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

# LayoutLM 的分词器类
class LayoutLMTokenizer(PreTrainedTokenizer):
    r"""
    Construct a LayoutLM tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            词汇表文件路径。
        do_lower_case (`bool`, *optional*, defaults to `True`):
            在进行标记化时是否将输入转换为小写。
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            是否在进行 WordPiece 标记化之前进行基本标记化。
        never_split (`Iterable`, *optional*):
            永远不会在标记化过程中分割的标记集合。仅在 `do_basic_tokenize=True` 时有效。
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            未知标记。词汇表中不存在的标记将被设置为此标记。
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            分隔符标记，用于从多个序列构建序列，例如用于序列分类或用于文本和问题的问答。它也用作由特殊标记构建的序列的最后一个标记。
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            用于填充的标记，例如在对不同长度的序列进行批处理时。
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            用于序列分类的分类器标记（整个序列的分类，而不是每个标记的分类）。它是使用特殊标记构建的序列的第一个标记。
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            用于屏蔽值的标记。这是训练掩码语言模型时使用的标记。这是模型将尝试预测的标记。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否标记化中文字符。

            对于日语，这可能应该被停用（参见这个 [issue](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否删除所有重音符号。如果未指定此选项，则将根据 `lowercase` 的值确定（与原始 LayoutLM 中一样）。
    """

    # 词汇表文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型词汇表文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 预训练模型最大输入尺寸
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
        # 检查是否提供了词汇表文件
        if not os.path.isfile(vocab_file):
            # 若不存在，则抛出数值错误异常
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = LayoutLMTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表
        self.vocab = load_vocab(vocab_file)
        # 构建从 IDs 到 Tokens 的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 设置是否执行基本标记化
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要基本标记化，则初始化基本标记化器
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        # 初始化 WordpieceTokenizer
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的初始化函数，传递参数和关键字参数
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
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, split_special_tokens=False):
        # 分词列表
        split_tokens = []
        
        # 若需要基本标记化
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果标记属于不分割集合
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 使用 WordpieceTokenizer 进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用词汇表将标记（字符串）转换为 ID
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用词汇表将索引（整数）转换为标记（字符串）
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将标记序列转换为单个字符串
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
            """
            Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
            adding special tokens. A LayoutLM sequence has the following format:
    
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
            cls = [self.cls_token_id]  # Assign the value of `cls_token_id` to `cls`
            sep = [self.sep_token_id]  # Assign the value of `sep_token_id` to `sep`
            return cls + token_ids_0 + sep + token_ids_1 + sep  # Concatenate special tokens with token IDs and return the resulting list
    
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
    
            if already_has_special_tokens:
                return super().get_special_tokens_mask(
                    token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
                )
    
            if token_ids_1 is not None:
                return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
            return [1] + ([0] * len(token_ids_0)) + [1]  # Return a special tokens mask for a single sequence
    
        def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 定义一个方法，返回用于序列对分类任务的掩码
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A LayoutLM sequence
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
        # 定义分隔符和CLS标记
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # 如果token_ids_1为空，返回只包含第一个序列部分的掩码（全为0）
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 返回包含第一个和第二个序列部分的掩码
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # 保存词汇表到文件
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引值
        index = 0
        # 如果存储目录已存在
        if os.path.isdir(save_directory):
            # 创建词汇表文件路径
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 创建词汇表文件路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开词汇表文件，以utf-8编码方式写入
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表，并按照索引值排序后写入文件
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # 如果词汇表索引值不连续，输出警告信息
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        # 返回保存的词汇表文件路径
        return (vocab_file,)
# 从 transformers.models.bert.tokenization_bert.BasicTokenizer 复制而来的 BasicTokenizer 类
class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).
    
    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在标记时将输入转换为小写。
        never_split (`Iterable`, *optional*):
            在标记化过程中永远不会被分割的标记集合。仅在 `do_basic_tokenize=True` 时生效。
        tokenize_chinese_chars (`bool`, *optional* defaults to `True`):
            是否标记化中文字符。

            对于日文，应该禁用该选项（参见此处链接的问题）。
        strip_accents (`bool`, *optional*):
            是否移除所有重音符号。如果没有指定该选项，则将由 `lowercase` 的值决定（与原始 BERT 类似）。
        do_split_on_punc (`bool`, *optional* defaults to `True`):
            在某些情况下，我们希望跳过基本的标点符号分割，以便后续的标记化可以捕获单词的完整上下文，例如缩略词。
    """
    
    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果没有指定 `never_split`，则将其设置为一个空列表
        if never_split is None:
            never_split = []
        # 初始化 BasicTokenizer 类的属性
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc
``` 
    # 对文本进行基本的分词处理。对于子词分词，参见WordPieceTokenizer。
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 将never_split和self.never_split的并集用于不进行分词的词汇
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本内容
        text = self._clean_text(text)

        # 以下部分代码是为了支持多语言和中文模型，这个功能从2018年11月1日开始添加
        # 现在也应用于英文模型，但由于英文模型没有在任何中文数据上训练，通常也不包含任何中文数据（英文维基百科
        # 包含一些中文单词，所以词汇表中有些中文字符）
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 标准化文本中的unicode字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 将原始文本按空白字符分割为token
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            # 如果token不在never_split中
            if token not in never_split:
                # 如果do_lower_case为True，将token转换为小写
                if self.do_lower_case:
                    token = token.lower()
                    # 如果strip_accents不为False，运行去掉重音的操作
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果strip_accents为True，运行去掉重音的操作
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 在分隔标点的token上运行分隔操作
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将分割后的token按空格合并为字符串，再次按空白字符分割得到output_tokens
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    # 去掉文本中的重音
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            # 如果字符的类别为"Mn"，表示为重音，直接跳过
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要在标点符号处分割，或者指定了不需要分割的文本，则返回原始文本列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        # 初始化索引和是否开始新单词的标志
        i = 0
        start_new_word = True
        # 初始化输出列表
        output = []
        # 遍历字符列表
        while i < len(chars):
            char = chars[i]
            # 如果当前字符是标点符号
            if _is_punctuation(char):
                # 将标点符号作为单独的列表项添加到输出列表
                output.append([char])
                start_new_word = True
            else:
                # 如果当前字符不是标点符号
                if start_new_word:
                    # 如果是新单词的开始，添加一个空列表到输出列表
                    output.append([])
                start_new_word = False
                # 将当前字符添加到输出列表的最后一个列表项中
                output[-1].append(char)
            i += 1
        # 将输出列表中的列表项合并成字符串，并返回结果列表
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        # 初始化输出列表
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 编码
            cp = ord(char)
            # 如果字符是中文字符，则在其前后添加空格
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                # 如果不是中文字符，则直接添加到输出列表中
                output.append(char)
        # 将输出列表中的字符合并成字符串，并返回结果
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查字符的 Unicode 编码是否在中文字符范围内
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
        # 初始化输出列表
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 编码
            cp = ord(char)
            # 如果字符是空字符或替换字符或控制字符，则跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果字符是空白字符，则将其替换为普通空格
            if _is_whitespace(char):
                output.append(" ")
            else:
                # 如果不是空白字符，则直接添加到输出列表中
                output.append(char)
        # 将输出列表中的字符合并成字符串，并返回结果
        return "".join(output)
# 从 transformers.models.bert.tokenization_bert.WordpieceTokenizer 复制而来的 WordpieceTokenizer 类
class WordpieceTokenizer(object):
    """运行 WordPiece 分词。"""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化函数，接受词汇表、未知token和每个词的最大字符数作为参数
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        # 对文本进行分词，使用贪婪的最长匹配算法来使用给定的词汇表进行分词

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.

        Args:
            text: 单个token或以空格分隔的tokens。应该已经通过 *BasicTokenizer* 处理过。

        Returns:
            一个 wordpiece tokens 的列表。

        output_tokens = []
        #对文本中的每个token进行处理
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                # 如果token的字符数超过指定的阈值，则添加未知token
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
                # 如果无法找到匹配的词，则添加未知token
                output_tokens.append(self.unk_token)
            else:
                # 可以找到匹配的词，则添加到输出列表中
                output_tokens.extend(sub_tokens)
        return output_tokens
```