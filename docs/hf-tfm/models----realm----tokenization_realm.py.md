# `.\transformers\models\realm\tokenization_realm.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 在 Apache 许可证 2.0 版本下授权使用该代码
# 只有在符合许可证的情况下才能使用该文件
# 可以获取许可证的副本链接
# https://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则依法提供的软件均为"AS IS"基础，不提供任何形式的明示或默示保证
# 请查看许可证中规定的具体语言以获取权限和限制
"""为 REALM 提供 Tokenization 类"""

# 导入必要的库

# 导入 collections 库
import collections
# 导入 os 库
import os
# 导入 unicodedata 库
import unicodedata
# 导入 List, Optional, Tuple 类型
from typing import List, Optional, Tuple

# 导入相关父类的方法和工具函数
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...tokenization_utils_base import BatchEncoding
from ...utils import PaddingStrategy, logging

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 定义词汇表文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练词汇表文件路径映射
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

# 预训练位置嵌入大小配置
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

# 预训练初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "google/realm-cc-news-pretrained-embedder": {"do_lower_case": True},
    "google/realm-cc-news-pretrained-encoder": {"do_lower_case": True},
    "google/realm-cc-news-pretrained-scorer": {"do_lower_case": True},
    "google/realm-cc-news-pretrained-openqa": {"do_lower_case": True},
    # 设置模型配置参数，指定是否将输入文本转换为小写
    "google/realm-orqa-nq-openqa": {"do_lower_case": True},
    "google/realm-orqa-nq-reader": {"do_lower_case": True},
    "google/realm-orqa-wq-openqa": {"do_lower_case": True},
    "google/realm-orqa-wq-reader": {"do_lower_case": True},
# 结束前面的代码块

# 从给定的词汇文件加载词汇表到一个有序字典中
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    # 创建一个有序的空字典
    vocab = collections.OrderedDict()
    # 打开词汇文件，并以 utf-8 编码读取内容
    with open(vocab_file, "r", encoding="utf-8") as reader:
        # 逐行读取词汇文件内容
        tokens = reader.readlines()
    # 遍历读取的词汇列表，将每个词汇与其 index 存入字典
    for index, token in enumerate(tokens):
        # 去除词汇末尾的换行符
        token = token.rstrip("\n")
        vocab[token] = index
    # 返回构建好的词汇字典
    return vocab


# 对文本进行基本的空白字符清理和分割
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # 去除文本首尾的空白字符
    text = text.strip()
    # 如果清理后文本为空，则返回空列表
    if not text:
        return []
    # 以空格为分隔符将文本分割成单词
    tokens = text.split()
    return tokens


# 定义 RealmTokenizer 类，继承自 PreTrainedTokenizer 类
class RealmTokenizer(PreTrainedTokenizer):
    r"""
    Construct a REALM tokenizer.

    [`RealmTokenizer`] is identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation splitting and
    wordpiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    # 此类定义了一个记号器(Tokenizer)的基类，用于将输入文本转换为模型可接受的输入格式
    # 它定义了一些常见的特殊标记符号，如未知标记(UNK)、分隔标记(SEP)、填充标记(PAD)等
    # 同时提供了一些可配置的选项，如是否转换为小写、是否对中文字符进行分词等
    Args:
        vocab_file (`str`):
            包含词汇表的文件路径
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否将输入文本转换为小写
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            是否在WordPiece分词之前进行基础分词
        never_split (`Iterable`, *optional*):
            一个不应该被分割的词汇集合，只有在do_basic_tokenize=True时有效
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            未知标记，当输入的词不在词汇表中时使用此标记
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            分隔标记，用于构建由多个序列组成的序列，例如用于序列分类或问答任务
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            填充标记，用于对长度不同的序列进行填充
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            分类标记，用于序列分类任务中
        mask_token (`str`, *optional`, defaults to `"[MASK]"`):
            掩码标记，用于掩蔽输入以进行掩码语言建模训练
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否对中文字符进行分词
        strip_accents (`bool`, *optional`):
            是否去除所有重音符号
    # 定义 RealmTokenizer 类
        ):
            # 检查指定的词汇表文件是否存在
            if not os.path.isfile(vocab_file):
                # 如果不存在则引发值错误，并提供相应的提示信息
                raise ValueError(
                    f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                    " model use `tokenizer = RealmTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                )
            # 从指定的词汇表文件中加载词汇表
            self.vocab = load_vocab(vocab_file)
            # 创建一个有序字典，将词汇表中的词语映射到其对应的 ID 
            self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
            # 设置是否进行基本的tokenization
            self.do_basic_tokenize = do_basic_tokenize
            # 如果需要进行基本的tokenization
            if do_basic_tokenize:
                # 创建 BasicTokenizer 实例，用于执行基本的tokenization
                self.basic_tokenizer = BasicTokenizer(
                    do_lower_case=do_lower_case,
                    never_split=never_split,
                    tokenize_chinese_chars=tokenize_chinese_chars,
                    strip_accents=strip_accents,
                )
            # 创建 WordpieceTokenizer 实例，用于执行WordPiece tokenization
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))
            # 调用父类的初始化方法
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
    
        # 获取是否进行小写处理的属性
        @property
        def do_lower_case(self):
            return self.basic_tokenizer.do_lower_case
    
        # 获取词汇表大小的属性
        @property
        def vocab_size(self):
            return len(self.vocab)
    
        # 获取完整的词汇表（包括添加的token）
        def get_vocab(self):
            return dict(self.vocab, **self.added_tokens_encoder)
    
        # 对输入文本进行tokenization
        def _tokenize(self, text):
            split_tokens = []
            # 如果需要进行基本的tokenization
            if self.do_basic_tokenize:
                # 使用 BasicTokenizer 对文本进行基本的tokenization
                for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                    # 如果token在 never_split 集合中，直接添加到 split_tokens 中
                    if token in self.basic_tokenizer.never_split:
                        split_tokens.append(token)
                    # 否则使用 WordpieceTokenizer 进行进一步的tokenization
                    else:
                        split_tokens += self.wordpiece_tokenizer.tokenize(token)
            # 如果不需要进行基本的tokenization，直接使用 WordpieceTokenizer 进行tokenization
            else:
                split_tokens = self.wordpiece_tokenizer.tokenize(text)
            return split_tokens
    
        # 将token转换为 ID
        def _convert_token_to_id(self, token):
            """Converts a token (str) in an id using the vocab."""
            return self.vocab.get(token, self.vocab.get(self.unk_token))
    
        # 将 ID 转换为 token
        def _convert_id_to_token(self, index):
            """Converts an index (integer) in a token (str) using the vocab."""
            return self.ids_to_tokens.get(index, self.unk_token)
    
        # 将一个token序列转换为字符串
        def convert_tokens_to_string(self, tokens):
            """Converts a sequence of tokens (string) in a single string."""
            out_string = " ".join(tokens).replace(" ##", "").strip()
            return out_string
    
        # 构建包含特殊token的输入序列
        def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def build_model_inputs(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
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
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # Check if only one sequence is provided
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # Create lists for special tokens and return the combined input
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False) -> List[int]:
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

        # If token list already has special tokens, call the superclass method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Calculate the mask for special tokens based on input sequences
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
    def create_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A REALM sequence
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
            `List[int`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # Define the separator token ID
        sep = [self.sep_token_id]
        # Define the classification token ID
        cls = [self.cls_token_id]
        # If token_ids_1 is None, return the mask with only the first sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # Return the mask with both sequences
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Initialize index
        index = 0
        # Check if save_directory is a directory
        if os.path.isdir(save_directory):
            # Set vocab_file path within the directory
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # Set vocab_file path without a directory
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # Open vocab_file in write mode
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # Iterate over vocab items, sorted by token index
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # Check for non-consecutive token indices
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # Write token to file
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).
    
    构造一个 BasicTokenizer，用于执行基本的标记化（分割标点符号、转换为小写等）。

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
            是否在标记化时将输入转换为小写。

        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`.
            在标记化时永远不会被拆分的标记集合。仅在 `do_basic_tokenize=True` 时有效。

        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.
            是否对中文字符进行标记化。 
            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
            对于日语，这可能需要停用（参见此问题）。

        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
            是否去除所有重音符号。如果未指定此选项，则将由 `lowercase` 的值确定（与原始 BERT 中的情况相同）。
    """

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        # 如果未指定 `never_split`，则将其设置为一个空列表
        if never_split is None:
            never_split = []
        # 将输入参数分配给相应的属性
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
    # 对输入文本进行基础分词操作
    def tokenize(self, text, never_split=None):
        # 将 never_split 列表与实例的 never_split 属性进行合并，得到最终的 never_split 列表
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 对输入文本进行清洁处理
        text = self._clean_text(text)
    
        # 如果配置了对中文字符进行特殊处理，则进行相应的分词
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 使用空白符进行分词，得到原始 token 列表
        orig_tokens = whitespace_tokenize(text)
        # 初始化分割后的 token 列表
        split_tokens = []
        # 遍历原始 token 列表
        for token in orig_tokens:
            # 如果 token 不在 never_split 列表中
            if token not in never_split:
                # 如果需要转为小写
                if self.do_lower_case:
                    token = token.lower()
                    # 如果需要去除重音符号
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果需要去除重音符号
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 将分割后的 token 添加到 split_tokens 列表
            split_tokens.extend(self._run_split_on_punc(token, never_split))
        # 使用空白符对 split_tokens 列表重新进行分词，得到最终输出
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens
    
    # 去除字符串中的重音符号
    def _run_strip_accents(self, text):
        # 使用 Unicode 正规化将字符串转换为标准形式
        text = unicodedata.normalize("NFD", text)
        # 初始化结果列表
        output = []
        # 遍历字符串中的每个字符
        for char in text:
            # 获取字符的类别
            cat = unicodedata.category(char)
            # 如果字符类别是 Mn（组合记号），则跳过
            if cat == "Mn":
                continue
            # 否则将字符添加到结果列表
            output.append(char)
        # 将结果列表拼接成字符串并返回
        return "".join(output)
    
    # 根据标点符号分割字符串
    def _run_split_on_punc(self, text, never_split=None):
        # 如果 never_split 列表中包含该 text，则直接返回 text 作为列表
        if never_split is not None and text in never_split:
            return [text]
        # 将字符串转换为字符列表
        chars = list(text)
        # 初始化下标和是否开始新词的标志
        i = 0
        start_new_word = True
        # 初始化结果列表
        output = []
        # 遍历字符列表
        while i < len(chars):
            char = chars[i]
            # 如果当前字符是标点符号
            if _is_punctuation(char):
                # 将该字符作为单独的列表元素添加到结果列表
                output.append([char])
                # 设置开始新词标志为 True
                start_new_word = True
            else:
                # 如果需要开始新词
                if start_new_word:
                    # 添加一个新的空列表作为结果列表的元素
                    output.append([])
                # 设置开始新词标志为 False
                start_new_word = False
                # 将当前字符添加到结果列表的最后一个元素中
                output[-1].append(char)
            # 下标自增 1
            i += 1
        # 将结果列表中的每个子列表拼接成字符串，组成最终结果
        return ["".join(x) for x in output]
    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        # 初始化一个空列表，用于存储处理后的文本
        output = []
        # 遍历输入的文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 编码
            cp = ord(char)
            # 如果字符是中文字符，则在其前后添加空格，并添加到处理后的文本列表中
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                # 如果不是中文字符，则直接添加到处理后的文本列表中
                output.append(char)
        # 将处理后的文本列表连接成字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查给定的 Unicode 编码是否在中文字符的 Unicode 块范围内
        # 中文字符的 Unicode 块范围参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)  # 基本汉字
            or (cp >= 0x3400 and cp <= 0x4DBF)  # 汉字扩展 A
            or (cp >= 0x20000 and cp <= 0x2A6DF)  # 汉字扩展 B
            or (cp >= 0x2A700 and cp <= 0x2B73F)  # 汉字扩展 C
            or (cp >= 0x2B740 and cp <= 0x2B81F)  # 汉字扩展 D
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  # 汉字扩展 E
            or (cp >= 0xF900 and cp <= 0xFAFF)  # CJK 兼容字
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  # 兼容扩展
        ):  # 如果在以上范围内，则返回 True 表示是中文字符
            return True

        return False  # 如果不在以上范围内，则返回 False 表示不是中文字符

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        # 初始化一个空列表，用于存储处理后的文本
        output = []
        # 遍历输入的文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 编码
            cp = ord(char)
            # 如果字符是无效字符（空字符、替换字符或控制字符），则跳过处理
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果字符是空白字符，则替换为单个空格
            if _is_whitespace(char):
                output.append(" ")
            else:
                # 否则直接添加到处理后的文本列表中
                output.append(char)
        # 将处理后的文本列表连接成字符串并返回
        return "".join(output)
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""  # WordPiece 分词处理器类的定义，用于运行 WordPiece 分词。

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化 WordpieceTokenizer 类，设定词汇表、未知标记和每个单词的最大输入字符数。
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
        # 将文本分词为其 WordPiece 组成部分。采用贪婪的最长匹配优先算法，使用给定的词汇表进行分词。
        
        # 初始化输出的 tokens 列表
        output_tokens = []
        # 对文本进行空格分词，得到 token 列表
        for token in whitespace_tokenize(text):
            # 将 token 转换为字符列表
            chars = list(token)
            # 若字符数超过单词的最大输入字符数，则将未知标记添加到输出 tokens 中，并跳过当前循环
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            # 初始化标志变量
            is_bad = False
            # 初始化起始位置
            start = 0
            # 初始化子 token 列表
            sub_tokens = []
            # 当起始位置小于字符列表长度时，进行循环
            while start < len(chars):
                # 设置结束位置为字符列表的长度
                end = len(chars)
                # 初始化当前子字符串为空
                cur_substr = None
                # 当起始位置小于结束位置时，进行循环
                while start < end:
                    # 构建子字符串
                    substr = "".join(chars[start:end])
                    # 若起始位置大于 0，则在子字符串前添加"##"前缀
                    if start > 0:
                        substr = "##" + substr
                    # 若子字符串在词汇表中，则将其设为当前子字符串并跳出循环
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    # 否则，减小结束位置
                    end -= 1
                # 若当前子字符串为 None，则将标志变量设为 True，并跳出循环
                if cur_substr is None:
                    is_bad = True
                    break
                # 将当前子字符串添加到子 token 列表中
                sub_tokens.append(cur_substr)
                # 更新起始位置为结束位置
                start = end

            # 若标志变量为 True，则将未知标记添加到输出 tokens 中，否则将子 token 列表添加到输出 tokens 中
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        # 返回输出 tokens 列表
        return output_tokens
```  
```