# `.\models\convbert\tokenization_convbert.py`

```
# 指定文件编码为 UTF-8
# 版权声明，基于 Apache License, Version 2.0
# 详细信息可查阅 http://www.apache.org/licenses/LICENSE-2.0
#
# 该脚本定义了 ConvBERT 的 tokenization 类

import collections  # 导入 collections 模块
import os  # 导入操作系统模块
import unicodedata  # 导入 unicodedata 模块
from typing import List, Optional, Tuple  # 导入类型提示模块中的 List, Optional, Tuple

# 导入 tokenization_utils 模块中的相关函数和类
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
# 导入 logging 模块中的 logger 对象
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 定义预训练模型词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "YituTech/conv-bert-base": "https://huggingface.co/YituTech/conv-bert-base/resolve/main/vocab.txt",
        "YituTech/conv-bert-medium-small": (
            "https://huggingface.co/YituTech/conv-bert-medium-small/resolve/main/vocab.txt"
        ),
        "YituTech/conv-bert-small": "https://huggingface.co/YituTech/conv-bert-small/resolve/main/vocab.txt",
    }
}

# 定义预训练模型的位置编码嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "YituTech/conv-bert-base": 512,
    "YituTech/conv-bert-medium-small": 512,
    "YituTech/conv-bert-small": 512,
}

# 定义预训练模型的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "YituTech/conv-bert-base": {"do_lower_case": True},
    "YituTech/conv-bert-medium-small": {"do_lower_case": True},
    "YituTech/conv-bert-small": {"do_lower_case": True},
}

# 从 transformers.models.bert.tokenization_bert.load_vocab 复制而来的函数
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()  # 创建一个有序字典对象
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()  # 逐行读取词汇文件内容
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")  # 去除每行末尾的换行符
        vocab[token] = index  # 将 token 加入到 vocab 字典中，索引为 index
    return vocab  # 返回构建好的词汇表字典

# 从 transformers.models.bert.tokenization_bert.whitespace_tokenize 复制而来的函数
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()  # 去除文本首尾空白字符
    if not text:
        return []  # 若文本为空，则返回空列表
    tokens = text.split()  # 使用空白字符分割文本，生成 token 列表
    return tokens  # 返回分割好的 token 列表

# 从 transformers.models.bert.tokenization_bert.BertTokenizer 复制而来的类定义
class ConvBertTokenizer(PreTrainedTokenizer):
    r"""
    Construct a ConvBERT tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # 定义一个类，用于处理基于特定词汇表的词汇和特殊标记的初始化配置
    
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    
    # 类的初始化方法，接收多个参数用于配置词汇表和标记化过程
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
        # vocab_file: 包含词汇的文件路径
        # do_lower_case: 是否将输入文本转换为小写进行标记化，默认为True
        # do_basic_tokenize: 是否在WordPiece之前进行基本的标记化，默认为True
        # never_split: 永远不会在标记化过程中拆分的标记集合，在do_basic_tokenize=True时生效
        # unk_token: 未知标记，用于词汇表中不存在的标记
        # sep_token: 分隔标记，用于构建多个序列的序列
        # pad_token: 填充标记，用于批处理不同长度的序列
        # cls_token: 分类器标记，在序列分类时作为序列的第一个标记
        # mask_token: 掩码标记，用于掩码语言建模中的训练
        # tokenize_chinese_chars: 是否标记化中文字符，默认为True；在处理日语时应禁用（参见链接）
        # strip_accents: 是否去除所有重音符号，如果未指定，则由lowercase的值决定
    ):
        # 检查是否存在指定的词汇文件，如果不存在则抛出异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表
        self.vocab = load_vocab(vocab_file)
        # 创建从词汇 ID 到词汇符号的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 根据参数设置是否执行基本的分词
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要基本分词，则创建 BasicTokenizer 对象
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        # 创建 WordpieceTokenizer 对象
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的初始化方法，传递相应参数
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
        # 返回 BasicTokenizer 是否执行小写处理的属性值
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        # 返回词汇表大小
        return len(self.vocab)

    def get_vocab(self):
        # 返回词汇表及其附加的特殊符号编码器
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, split_special_tokens=False):
        # 分词函数，根据设定选择使用基本分词器或 Wordpiece 分词器
        split_tokens = []
        if self.do_basic_tokenize:
            # 使用基本分词器进行分词，根据参数决定是否保留特殊符号的分割
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果 token 是不应分割的特殊符号，则直接添加到结果中
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 否则使用 WordpieceTokenizer 进一步分词
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 如果不使用基本分词器，则直接使用 WordpieceTokenizer 进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将词汇符号转换为其对应的 ID
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将 ID 转换为其对应的词汇符号
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将 token 序列转换为单个字符串，并去除"##"子词标记
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        # 构建包含特殊符号的输入序列
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Generate token type IDs from two sequences. Token type IDs differentiate between the different sequences
        in the input (e.g., segment A and segment B in a sequence pair).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs representing the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs representing the second sequence in a sequence pair.

        Returns:
            `List[int]`: List of token type IDs.
        """

        if token_ids_1 is None:
            # If there is only one sequence, return token type IDs as all zeros
            return [0] * len(token_ids_0)
        
        # For a sequence pair, generate token type IDs where the first sequence is 0s and the second sequence is 1s
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)
    def create_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A ConvBERT sequence
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
        # Define the separator and classification tokens
        sep = [self.sep_token_id]  # List containing the separation token ID
        cls = [self.cls_token_id]  # List containing the classification token ID

        if token_ids_1 is None:
            # If token_ids_1 is None, return a mask with 0s for the first sequence only
            return len(cls + token_ids_0 + sep) * [0]
        else:
            # If token_ids_1 is provided, return a mask with 0s for the first sequence and 1s for the second sequence
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary of the model to a file.

        Args:
            save_directory (str):
                Directory path where the vocabulary file will be saved.
            filename_prefix (str, *optional*):
                Prefix to be added to the vocabulary file name.

        Returns:
            Tuple[str]: Tuple containing the path to the saved vocabulary file.
        """
        index = 0  # Initialize index for iterating over vocabulary items

        # Determine the full path and filename of the vocabulary file
        if os.path.isdir(save_directory):
            # If save_directory is a directory, construct the full path including the prefix and standard file name
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # If save_directory is a file path, directly use it as the vocabulary file name
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory

        # Write the vocabulary to the determined file path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # Iterate through sorted vocabulary items and write each token to the file
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # Check if the indices are consecutive and log a warning if not
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index  # Update index to current token's index
                writer.write(token + "\n")  # Write the token followed by a newline
                index += 1  # Increment index for the next token

        return (vocab_file,)  # Return tuple containing the path to the saved vocabulary file
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
        do_lower_case=True,                # 初始化方法，设置是否小写化输入，默认为True
        never_split=None,                  # 初始化方法，设置永远不分割的token集合，默认为None
        tokenize_chinese_chars=True,       # 初始化方法，设置是否分割中文字符，默认为True
        strip_accents=None,                # 初始化方法，设置是否去除所有重音符号，默认根据小写化设置决定
        do_split_on_punc=True,             # 初始化方法，设置是否基本标点符号分割，默认为True
    ):
        if never_split is None:
            never_split = []               # 如果never_split为None，则设置为空列表
        self.do_lower_case = do_lower_case  # 将参数赋值给对象属性，控制是否小写化
        self.never_split = set(never_split)  # 将参数转换为集合并赋值给对象属性，设置永远不分割的token集合
        self.tokenize_chinese_chars = tokenize_chinese_chars  # 将参数赋值给对象属性，控制是否分割中文字符
        self.strip_accents = strip_accents  # 将参数赋值给对象属性，控制是否去除重音符号
        self.do_split_on_punc = do_split_on_punc  # 将参数赋值给对象属性，控制是否基本标点符号分割
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        # 如果传入了 never_split 参数，则将其与对象的 never_split 属性合并成一个新的集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本中的不规范字符或格式
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        # 如果启用了 tokenize_chinese_chars，对文本中的中文字符进行特殊处理
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        
        # prevents treating the same character with different unicode codepoints as different characters
        # 使用 NFC 规范化 Unicode 文本，确保不同的 Unicode 编码的同一字符被视为相同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 将文本按空白字符分割成原始 token 列表
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    # 如果设置了 do_lower_case，将 token 转换为小写
                    token = token.lower()
                    if self.strip_accents is not False:
                        # 如果 strip_accents 不为 False，则去除 token 中的重音符号
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    # 如果 strip_accents 为 True，去除 token 中的重音符号
                    token = self._run_strip_accents(token)
            # 将 token 按标点符号分割，并加入到 split_tokens 列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将分割后的 token 列表合并成字符串，并按空白字符分割，返回最终的 token 列表
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 将文本标准化为 NFD 形式，分解为基字符和重音符号
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue  # 如果字符类别为 Mn（重音符号），则跳过
            output.append(char)
        # 将处理后的字符列表连接成字符串并返回
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
            # 如果字符是标点符号，则将其作为单独的列表项添加到output中，并标记开始一个新单词
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，且标记为开始新单词，则创建一个新的空列表项
                if start_new_word:
                    output.append([])
                # 将当前字符添加到最后一个列表项中，并标记不再开始新单词
                output[-1].append(char)
                start_new_word = False
            i += 1

        # 将列表中的列表项合并为字符串，并返回结果列表
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符是CJK字符，则在字符前后加入空格，并添加到输出列表中
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                # 如果不是CJK字符，则直接添加到输出列表中
                output.append(char)
        # 将列表中的字符连接为一个字符串，并返回结果
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 判断cp是否在CJK字符的Unicode范围内
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)
        ):
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符是无效字符或控制字符，则跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果是空白字符，则替换为单个空格；否则直接添加到输出列表中
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将列表中的字符连接为一个字符串，并返回清理后的文本
        return "".join(output)
# Copied from transformers.models.bert.tokenization_bert.WordpieceTokenizer
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化 WordpieceTokenizer 类的实例
        self.vocab = vocab  # 词汇表，用于词片段化
        self.unk_token = unk_token  # 未知标记，用于替换无法识别的词片段
        self.max_input_chars_per_word = max_input_chars_per_word  # 单个词的最大字符数限制

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
        # 初始化输出词片段列表
        output_tokens = []
        # 对文本进行空白符分割，得到基本的 token
        for token in whitespace_tokenize(text):
            chars = list(token)
            # 如果 token 的字符数超过设定的最大字符数限制，则使用未知标记替代
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            # 使用贪婪的最长匹配算法进行词片段化
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                # 从最长到最短尝试生成子串，并在匹配到词汇表中的词片段时停止
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                # 如果未找到匹配的词片段，则标记为无法识别
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            # 如果存在无法识别的情况，则使用未知标记替代
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        # 返回最终的词片段化结果列表
        return output_tokens
```