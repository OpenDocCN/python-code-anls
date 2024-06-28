# `.\models\squeezebert\tokenization_squeezebert.py`

```
# 设置文件编码为 UTF-8
# 版权声明：2020 年由 SqueezeBert 作者和 HuggingFace Inc. 团队共同持有
#
# 根据 Apache 许可证版本 2.0 进行许可
# 除非符合许可证的条款，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于 "按原样" 分发的
# 没有任何明示或暗示的担保或条件
# 请参阅许可证了解特定语言的权限及限制

"""SqueezeBERT 的标记化类。"""

# 引入必要的库和模块
import collections
import os
import unicodedata
from typing import List, Optional, Tuple

# 从 tokenization_utils 模块中导入必要的函数和类
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
# 从 utils 模块中导入日志记录功能
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "squeezebert/squeezebert-uncased": (
            "https://huggingface.co/squeezebert/squeezebert-uncased/resolve/main/vocab.txt"
        ),
        "squeezebert/squeezebert-mnli": "https://huggingface.co/squeezebert/squeezebert-mnli/resolve/main/vocab.txt",
        "squeezebert/squeezebert-mnli-headless": (
            "https://huggingface.co/squeezebert/squeezebert-mnli-headless/resolve/main/vocab.txt"
        ),
    }
}

# 预训练模型的位置编码尺寸映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "squeezebert/squeezebert-uncased": 512,
    "squeezebert/squeezebert-mnli": 512,
    "squeezebert/squeezebert-mnli-headless": 512,
}

# 预训练模型的初始化配置映射
PRETRAINED_INIT_CONFIGURATION = {
    "squeezebert/squeezebert-uncased": {"do_lower_case": True},
    "squeezebert/squeezebert-mnli": {"do_lower_case": True},
    "squeezebert/squeezebert-mnli-headless": {"do_lower_case": True},
}

# 从 transformers.models.bert.tokenization_bert.load_vocab 复制的函数
def load_vocab(vocab_file):
    """加载词汇文件到一个字典中。"""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

# 从 transformers.models.bert.tokenization_bert.whitespace_tokenize 复制的函数
def whitespace_tokenize(text):
    """在文本上执行基本的空白字符清理和分割。"""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

# 从 transformers.models.bert.tokenization_bert.BertTokenizer 复制的类，并将 Bert 改为 SqueezeBert
class SqueezeBertTokenizer(PreTrainedTokenizer):
    r"""
    构建一个 SqueezeBERT 分词器。基于 WordPiece。

    这个分词器继承自 [`PreTrainedTokenizer`]，其中包含大多数主要方法。用户应参考其文档
    # 定义一个类，用于处理词汇表和标记化的相关功能，继承自PreTrainedTokenizerBase类，
    # 可以查阅更多关于这些方法的信息。
    
    Args:
        vocab_file (`str`):
            包含词汇表的文件路径。
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在标记化时将输入转换为小写。
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            是否在WordPiece标记化之前进行基本的分词处理。
        never_split (`Iterable`, *optional*):
            在标记化时不会被分割的标记集合。仅在 `do_basic_tokenize=True` 时有效。
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            未知标记。词汇表中不存在的标记将被设置为这个标记。
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            分隔符标记，用于将多个序列组合成一个序列，例如用于序列分类或问答任务中。
            也用作带有特殊标记的序列的最后一个标记。
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            用于填充的标记，例如在对不同长度的序列进行批处理时使用。
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            分类器标记，用于序列分类任务中。构建带有特殊标记时是序列的第一个标记。
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            用于掩码值的标记。在使用掩码语言建模训练时，模型将尝试预测这个标记。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否对中文字符进行标记化。对于日语，可能需要禁用此选项（参见此处的相关问题）。
        strip_accents (`bool`, *optional*):
            是否去除所有的重音符号。如果未指定此选项，则根据 `lowercase` 的值（如原始的SqueezeBERT中）确定。
    """
    
    vocab_files_names = VOCAB_FILES_NAMES  # 词汇表文件名列表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练词汇表文件映射
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION  # 预训练模型初始化配置
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 预训练模型最大输入尺寸
    # 初始化函数，用于初始化一个新的 Tokenizer 对象
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
        # 检查给定的词汇文件是否存在，如果不存在则抛出 ValueError 异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = SqueezeBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表文件并赋值给实例变量 vocab
        self.vocab = load_vocab(vocab_file)
        # 根据词汇表创建一个从 id 到 token 的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 是否执行基本的分词操作
        self.do_basic_tokenize = do_basic_tokenize
        # 如果执行基本分词，则创建 BasicTokenizer 对象并赋值给实例变量 basic_tokenizer
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        # 根据词汇表创建 WordpieceTokenizer 对象，并赋值给实例变量 wordpiece_tokenizer
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的初始化方法，传递相关参数和额外参数
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

    # 返回实例变量 basic_tokenizer 的 do_lower_case 属性值
    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    # 返回词汇表的大小（即词汇表中不同 token 的数量）
    @property
    def vocab_size(self):
        return len(self.vocab)

    # 返回词汇表和 added_tokens_encoder 的合并字典
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # 将输入的文本进行分词处理，返回分词后的列表
    def _tokenize(self, text, split_special_tokens=False):
        split_tokens = []
        # 如果执行基本分词操作
        if self.do_basic_tokenize:
            # 使用 basic_tokenizer 对象对文本进行分词
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果分词结果在 never_split 集合中，则直接添加到 split_tokens 中
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                # 否则使用 wordpiece_tokenizer 进行进一步的分词，并将结果添加到 split_tokens 中
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 如果不执行基本分词，则直接使用 wordpiece_tokenizer 对文本进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    # 根据 token 返回其对应的 id，如果 token 不在词汇表中，则返回 unk_token 对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 根据 id 返回其对应的 token，如果 id 不在 ids_to_tokens 中，则返回 unk_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)
    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) into a single string.

        Args:
            tokens (`List[str]`): List of tokens to be joined into a string.

        Returns:
            `str`: The concatenated string of tokens with "##" markers removed.
        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Builds model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens. A SqueezeBERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`): List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*): Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens added.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence IDs from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`): List of IDs.
            token_ids_1 (`List[int]`, *optional*): Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers indicating the presence of special tokens (1) or sequence tokens (0).
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates token type IDs from a sequence or a pair of sequences for sequence classification tasks. Token type IDs
        distinguish between the first and the second sequences in a pair.

        Args:
            token_ids_0 (`List[int]`): List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*): Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs with appropriate distinctions for sequence pairs.
        """
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A SqueezeBERT sequence
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
        # Define special tokens for separation and classification
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        # If only one sequence is provided, return a mask with all zeros
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # Otherwise, concatenate both sequences with separators and return a mask
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Initialize index for vocabulary token numbering
        index = 0
        
        # Determine the path and filename of the vocabulary file
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        
        # Write the vocabulary to the specified file
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # Iterate over sorted vocabulary items and write each token to the file
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # Check if the indices are consecutive and log a warning if not
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # Write the token followed by a newline
                writer.write(token + "\n")
                index += 1
        
        # Return the path to the saved vocabulary file
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
        # 如果 `never_split` 参数未提供，则设为一个空列表
        if never_split is None:
            never_split = []
        # 设置是否将输入转换为小写
        self.do_lower_case = do_lower_case
        # 将 `never_split` 转换为集合，用于快速查找不应分割的特定标记
        self.never_split = set(never_split)
        # 设置是否对中文字符进行分词
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 设置是否去除所有重音符号，如果未指定，则由 `lowercase` 的值决定（与原始 BERT 相同）
        self.strip_accents = strip_accents
        # 设置是否在基本标点符号上进行分割
        self.do_split_on_punc = do_split_on_punc
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 使用 union() 方法将 self.never_split 和传入的 never_split 合并成一个新的集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本内容，如去除多余空格等
        text = self._clean_text(text)

        # 以下代码块是为了支持多语言和中文模型，对文本进行分词处理
        # 这个功能于2018年11月1日添加，适用于多语言和中文模型
        if self.tokenize_chinese_chars:
            # 对包含中文字符的文本进行特殊处理
            text = self._tokenize_chinese_chars(text)
        
        # 使用 NFC 标准规范化 Unicode 文本，以避免相同字符的不同 Unicode 编码被视为不同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空白字符进行分词，得到原始 token 列表
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        # 初始化分割后的 token 列表
        split_tokens = []
        
        # 遍历原始 token 列表
        for token in orig_tokens:
            # 如果 token 不在 never_split 中，则继续处理
            if token not in never_split:
                # 如果设置为小写处理，则将 token 转换为小写
                if self.do_lower_case:
                    token = token.lower()
                    # 如果 strip_accents 不为 False，则移除 token 中的重音符号
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果 strip_accents 为 True，则移除 token 中的重音符号
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 将处理后的 token 添加到分割后的 token 列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将分割后的 token 列表使用空白字符连接，并进行最终的空白字符分词
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        # 返回最终处理后的 token 列表
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 使用 NFD 标准规范化 Unicode 文本，以便处理重音符号
        text = unicodedata.normalize("NFD", text)
        # 初始化输出列表
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 分类
            cat = unicodedata.category(char)
            # 如果字符的分类为 Mn（Mark, Nonspacing），则跳过该字符
            if cat == "Mn":
                continue
            # 将非重音符号的字符添加到输出列表中
            output.append(char)
        # 将输出列表中的字符连接成字符串并返回
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """在文本上执行标点符号的分割。"""
        # 如果不需要在标点处分割，或者文本在never_split中，直接返回文本列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号，则将其作为新的单词输出列表的一个单独项
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，根据情况添加到当前单词中或开始新单词
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        # 将列表中的字符列表连接成字符串列表并返回
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """在CJK字符周围添加空格。"""
        output = []
        for char in text:
            cp = ord(char)
            # 如果是CJK字符，将其周围添加空格
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        # 将列表转换为字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """检查CP是否是CJK字符的码位。"""
        # 这里定义的"中文字符"包括CJK统一表意字符范围内的所有字符
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
        """对文本执行无效字符移除和空白字符清理。"""
        output = []
        for char in text:
            cp = ord(char)
            # 如果是无效字符或控制字符，跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果是空白字符，替换为单个空格，否则保留字符
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将列表转换为字符串并返回
        return "".join(output)
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化 WordpieceTokenizer 类
        self.vocab = vocab  # 词汇表，用于词片段的匹配
        self.unk_token = unk_token  # 未知 token，在词汇表中找不到匹配时使用
        self.max_input_chars_per_word = max_input_chars_per_word  # 单词最大长度限制，默认为 100

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
        
        output_tokens = []  # 存储最终的 wordpiece tokens 结果列表
        for token in whitespace_tokenize(text):  # 遍历通过空格分隔的文本中的每个 token
            chars = list(token)  # 将 token 拆分为字符列表
            if len(chars) > self.max_input_chars_per_word:  # 如果 token 的长度超过设定的最大输入字符数
                output_tokens.append(self.unk_token)  # 将未知 token 添加到输出结果中
                continue

            is_bad = False  # 标记当前 token 是否无法分割为 wordpiece tokens
            start = 0  # 初始化起始索引
            sub_tokens = []  # 存储当前 token 分割后的 wordpiece tokens
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])  # 获取从 start 到 end 的子字符串
                    if start > 0:
                        substr = "##" + substr  # 如果 start > 0，表示为一个片段的延续
                    if substr in self.vocab:  # 如果当前子字符串在词汇表中
                        cur_substr = substr  # 记录当前有效的子字符串
                        break
                    end -= 1  # 否则尝试减小 end，缩小子字符串范围
                if cur_substr is None:
                    is_bad = True  # 如果无法找到有效的子字符串，则标记为无法处理
                    break
                sub_tokens.append(cur_substr)  # 将有效的子字符串添加到 sub_tokens 列表中
                start = end  # 更新 start 为 end，继续处理下一个子字符串

            if is_bad:
                output_tokens.append(self.unk_token)  # 如果 token 无法处理，则添加未知 token
            else:
                output_tokens.extend(sub_tokens)  # 否则将处理后的 wordpiece tokens 添加到结果列表中
        return output_tokens  # 返回最终的 wordpiece tokens 列表
```