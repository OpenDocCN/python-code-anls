# `.\models\mobilebert\tokenization_mobilebert.py`

```py
# coding=utf-8
# 上面是指定源代码文件的编码格式为UTF-8

# 版权声明和许可信息，这段代码受 Apache License, Version 2.0 许可，详细信息可以在给定的 URL 查看
# http://www.apache.org/licenses/LICENSE-2.0

"""Tokenization classes for MobileBERT."""
# 以上是对本文件模块的简要描述和标识，说明其包含 MobileBERT 的分词类

import collections  # 导入 collections 模块，用于高性能容器数据类型的支持
import os  # 导入 os 模块，用于与操作系统进行交互
import unicodedata  # 导入 unicodedata 模块，用于对 Unicode 字符数据库的访问和操作
from typing import List, Optional, Tuple  # 导入类型提示的相关内容

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
# 导入必要的 tokenizer 相关模块和函数
from ...utils import logging  # 导入 logging 模块用于日志记录

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}  # 定义词汇表文件名

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"mobilebert-uncased": "https://huggingface.co/google/mobilebert-uncased/resolve/main/vocab.txt"}
}
# 定义预训练模型的词汇表文件映射，提供了 mobilebert-uncased 的词汇表文件下载链接

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"mobilebert-uncased": 512}
# 定义预训练模型的位置嵌入尺寸，这里是 mobilebert-uncased 的位置嵌入尺寸为 512

PRETRAINED_INIT_CONFIGURATION = {}
# 定义预训练模型的初始化配置，此处为空字典

# 从 transformers.models.bert.tokenization_bert.load_vocab 复制而来
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    # 加载词汇表文件到一个有序字典中
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()  # 逐行读取词汇表文件内容
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")  # 去除每行末尾的换行符
        vocab[token] = index  # 将词汇表中的词条和对应的索引存入字典
    return vocab  # 返回加载后的词汇表字典

# 从 transformers.models.bert.tokenization_bert.whitespace_tokenize 复制而来
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()  # 去除文本首尾空格
    if not text:
        return []  # 如果文本为空，则返回空列表
    tokens = text.split()  # 使用空格分割文本，得到token列表
    return tokens  # 返回分割后的token列表

# 从 transformers.models.bert.tokenization_bert.BertTokenizer 复制而来，修改为 MobileBertTokenizer
class MobileBertTokenizer(PreTrainedTokenizer):
    r"""
    Construct a MobileBERT tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # 创建 MobileBERT 分词器，基于 WordPiece 算法实现
    # 定义 Transformer 的 Tokenizer 类
    class PreTrainedTokenizer:
        # 类属性：指定了用于加载词汇表文件的名称
        vocab_files_names = VOCAB_FILES_NAMES
        # 类属性：指定了预训练模型的词汇表文件映射
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        # 类属性：指定了预训练模型初始化的配置
        pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
        # 类属性：指定了预训练模型的最大输入大小
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

        # 初始化方法，接收多个参数
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
                " model use `tokenizer = MobileBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表并将其存储在 self.vocab 中
        self.vocab = load_vocab(vocab_file)
        # 根据词汇表生成一个从 ids 到 tokens 的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 根据参数 do_basic_tokenize 决定是否执行基本的分词处理
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            # 如果需要基本分词，则创建 BasicTokenizer 对象
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        # 创建 WordpieceTokenizer 对象，使用给定的词汇表和未知 token
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的初始化方法，传递相关参数
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
        # 返回 basic_tokenizer 的 do_lower_case 属性
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        # 返回词汇表的大小
        return len(self.vocab)

    def get_vocab(self):
        # 返回包含词汇表和 added_tokens_encoder 的字典
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, split_special_tokens=False):
        # 初始化分词结果列表
        split_tokens = []
        if self.do_basic_tokenize:
            # 如果需要基本分词，则使用 basic_tokenizer 对文本进行分词处理
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果 token 在 never_split 集合中，则直接加入结果列表
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 否则，将 token 使用 wordpiece_tokenizer 进行进一步分词处理并加入结果列表
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 如果不需要基本分词，则直接使用 wordpiece_tokenizer 对文本进行分词处理
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        # 返回最终的分词结果列表
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 根据 token 查找对应的 id，如果不存在则返回 unk_token 对应的 id
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 根据 index 查找对应的 token，如果不存在则返回 unk_token
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将 tokens 列表转换成一个字符串，同时去除 " ##" 并且去除两端的空格
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        # 构建包含特殊 token 的输入序列
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens. A MobileBERT sequence has the following format:

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
        # Check if only one sequence is provided
        if token_ids_1 is None:
            # Return input IDs with [CLS], sequence tokens, and [SEP]
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        
        # For sequence pair, create token IDs with [CLS], first sequence, [SEP], second sequence, and final [SEP]
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

        # If the input token lists already have special tokens, delegate to the base class method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Calculate special tokens mask for sequences without existing special tokens
        if token_ids_1 is not None:
            # For sequence pair, return a mask with 1s for special tokens and 0s for sequence tokens
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        else:
            # For single sequence, return a mask with 1s for special tokens and 0s for sequence tokens
            return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs tensor from token id pairs for sequence pairs. Token type IDs are binary tensors with 0s and 1s.
        0 indicates the first sequence, and 1 indicates the second sequence.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional list of IDs for the second sequence in a pair.

        Returns:
            `List[int]`: A list of token type IDs representing the sequences.
        """
    def create_mobilebert_attention_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A MobileBERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of token IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of token IDs for sequence pairs.

        Returns:
            `List[int]`: List representing token type IDs according to the given sequence(s).
        """
        # Define the separator token ID
        sep = [self.sep_token_id]
        # Define the classification token ID
        cls = [self.cls_token_id]
        
        # If token_ids_1 is None, return a mask with only the first sequence (0s)
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # Concatenate token IDs for both sequences with separators and compute the mask
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Initialize index for checking consecutive vocabulary indices
        index = 0
        
        # Determine the vocabulary file path based on whether save_directory is a directory or a filename
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
                # Check for non-consecutive vocabulary indices and issue a warning if found
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # Write token followed by a newline
                writer.write(token + "\n")
                index += 1
        
        # Return the path to the saved vocabulary file
        return (vocab_file,)
# Copied from transformers.models.bert.tokenization_bert.BasicTokenizer
# 定义 BasicTokenizer 类，用于执行基本的分词操作（如标点符号分割、小写处理等）。
class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在分词时将输入转换为小写。
        never_split (`Iterable`, *optional*):
            在分词过程中永不分割的 token 集合。仅在 `do_basic_tokenize=True` 时有效。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否分词中包含中文字符。这对于日文来说可能需要禁用（参见这个问题）。
        strip_accents (`bool`, *optional*):
            是否去除所有的重音符号。如果未指定，则根据 `lowercase` 的值决定（与原始的 BERT 一致）。
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            在某些情况下，我们希望跳过基本的标点符号分割，以便稍后的分词可以捕获单词的完整上下文，例如缩略词。

    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果 `never_split` 参数为 None，则初始化为空列表。
        if never_split is None:
            never_split = []
        
        # 将类实例化时传入的参数赋值给对应的类成员变量。
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)  # 将 `never_split` 转换为集合，方便快速查找。
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
        # 使用联合操作将 `never_split` 参数与对象属性 `self.never_split` 合并成一个新的集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本中的特殊字符和格式
        text = self._clean_text(text)

        # 以下代码段是为了处理多语言和中文模型而添加的，从2018年11月1日开始生效。
        # 现在也应用于英文模型，尽管这些模型没有在任何中文数据上训练，
        # 通常不包含任何中文数据（英文维基百科中有些中文单词，因此词汇表中有些中文字符）。
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        
        # 标准化文本中的 Unicode 编码，确保相同字符使用同一种 Unicode 规范
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 将标准化后的文本按空白分词，得到原始的 token 列表
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            # 如果 token 不在 `never_split` 中，则根据 tokenizer 的设置进行处理
            if token not in never_split:
                if self.do_lower_case:
                    # 如果设置为小写，则将 token 转换为小写
                    token = token.lower()
                    # 如果设置了去除重音符号，则执行去除重音操作
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    # 如果仅设置了去除重音符号，则执行去除重音操作
                    token = self._run_strip_accents(token)
            # 将处理后的 token 按标点分割，加入到 `split_tokens` 列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将处理后的 `split_tokens` 再次按空白分词，得到最终的 token 列表
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 标准化文本中的 Unicode 编码，将组合字符分解为基字符和重音符号
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            # 检查字符的 Unicode 类别，如果是重音符号，则跳过
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            # 将不是重音符号的字符添加到输出列表中
            output.append(char)
        # 将输出列表中的字符连接成字符串，返回处理后的文本
        return "".join(output)
    # 在给定的文本上根据标点符号进行分割
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要根据标点符号分割或者文本在不分割列表中，则返回原文本列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号，则作为新的单词处理
            if _is_punctuation(char):
                output.append([char])  # 将标点符号作为单独的列表项加入输出
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])  # 开始新单词时创建一个空列表
                start_new_word = False
                output[-1].append(char)  # 将当前字符添加到最后一个单词列表的末尾
            i += 1

        return ["".join(x) for x in output]  # 将分割后的单词列表重新组合成字符串并返回

    # 在中文字符周围添加空格
    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符是中文字符，则在其前后添加空格
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)  # 将列表中的字符连接成字符串并返回

    # 检查字符的 Unicode 码点是否是中文字符
    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 这里将“中文字符”定义为CJK Unicode块中的字符：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # 需要注意的是，CJK Unicode块并不包含所有日文和韩文字符，现代韩文和日文的字母属于不同的Unicode块。
        # 这些字母用于书写空格分隔的单词，因此不特别处理，与其他语言一样处理。
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)  # 基本CJK字符（4E00-9FFF）
            or (cp >= 0x3400 and cp <= 0x4DBF)  # CJK扩展A（3400-4DBF）
            or (cp >= 0x20000 and cp <= 0x2A6DF)  # CJK扩展B（20000-2A6DF）
            or (cp >= 0x2A700 and cp <= 0x2B73F)  # CJK扩展C（2A700-2B73F）
            or (cp >= 0x2B740 and cp <= 0x2B81F)  # CJK扩展D（2B740-2B81F）
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  # CJK扩展E（2B820-2CEAF）
            or (cp >= 0xF900 and cp <= 0xFAFF)  # 兼容CJK字符（F900-FAFF）
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  # 兼容扩展（2F800-2FA1F）
        ):
            return True

        return False

    # 在文本中执行无效字符移除和空白清理操作
    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符是无效字符或控制字符，则跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果字符是空白字符，则替换为单个空格
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)  # 将清理后的字符列表连接成字符串并返回
# Copied from transformers.models.bert.tokenization_bert.WordpieceTokenizer
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化WordpieceTokenizer类的实例，设置词汇表、未知标记和每个单词最大输入字符数
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
        # 初始化一个空列表，用于存储WordPiece分词的结果
        output_tokens = []
        # 对输入的文本进行空格分词，得到单词列表
        for token in whitespace_tokenize(text):
            # 将单词转换为字符列表
            chars = list(token)
            # 如果单词长度超过设定的最大字符数，则将未知标记添加到输出列表，并继续下一个单词
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            # 初始化标志变量和起始索引
            is_bad = False
            start = 0
            sub_tokens = []
            # 使用最长匹配优先的贪婪算法进行分词
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                # 从当前位置向前截取子串，并加上前缀"##"，检查是否在词汇表中
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                # 如果没有找到匹配的子串，则标记为无效
                if cur_substr is None:
                    is_bad = True
                    break
                # 将找到的子串加入子词列表，并更新起始索引
                sub_tokens.append(cur_substr)
                start = end

            # 根据是否标记为无效，将对应的结果添加到输出列表
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        # 返回最终的WordPiece分词结果列表
        return output_tokens
```