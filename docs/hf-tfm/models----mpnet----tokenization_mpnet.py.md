# `.\models\mpnet\tokenization_mpnet.py`

```py
# 设置编码格式为 UTF-8
# 版权声明，包括 HuggingFace Inc. 团队和 Microsoft Corporation 的版权声明
# 版权声明，包括 NVIDIA CORPORATION 的版权声明
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获得许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 如果适用法律要求或书面同意，此软件是基于"原样"提供的，不附带任何明示或暗示的担保或条件
# 请参阅许可证了解特定语言的权限和限制
"""Tokenization classes for MPNet."""

# 导入必要的模块和函数
import collections
import os
import unicodedata
from typing import List, Optional, Tuple

# 从 tokenization_utils 模块中导入特定的函数和类
from ...tokenization_utils import AddedToken, PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
# 从 utils 模块中导入 logging 函数
from ...utils import logging

# 获取 logger 对象用于日志记录
logger = logging.get_logger(__name__)

# 定义词汇文件的名称映射字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 定义预训练模型的词汇文件映射字典
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/mpnet-base": "https://huggingface.co/microsoft/mpnet-base/resolve/main/vocab.txt",
    }
}

# 定义预训练模型的位置嵌入尺寸字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/mpnet-base": 512,
}

# 定义预训练模型的初始化配置字典
PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/mpnet-base": {"do_lower_case": True},
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    # 使用 collections.OrderedDict 创建一个有序字典来存储词汇表
    vocab = collections.OrderedDict()
    # 打开词汇文件并按行读取所有 token
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    # 将每个 token 和其对应的索引存储到 vocab 字典中
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")  # 去除每个 token 末尾的换行符
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # 去除文本两端的空白字符
    text = text.strip()
    if not text:
        return []  # 如果文本为空，则返回空列表
    # 使用空白字符分割文本，并返回分割后的 token 列表
    tokens = text.split()
    return tokens


class MPNetTokenizer(PreTrainedTokenizer):
    """

    This tokenizer inherits from [`BertTokenizer`] which contains most of the methods. Users should refer to the
    superclass for more information regarding methods.

    """

    # 设置类的词汇文件名字典属性
    vocab_files_names = VOCAB_FILES_NAMES
    # 设置类的预训练模型词汇文件映射属性
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 设置类的预训练模型初始化配置属性
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 设置类的最大模型输入尺寸属性
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 设置类的模型输入名称属性
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="[UNK]",
        pad_token="<pad>",
        mask_token="<mask>",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
        ):
            # 如果 bos_token 是字符串，则将其封装为特殊的 AddedToken 对象；否则保持不变
            bos_token = AddedToken(bos_token, special=True) if isinstance(bos_token, str) else bos_token
            # 如果 eos_token 是字符串，则将其封装为特殊的 AddedToken 对象；否则保持不变
            eos_token = AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token
            # 如果 sep_token 是字符串，则将其封装为特殊的 AddedToken 对象；否则保持不变
            sep_token = AddedToken(sep_token, special=True) if isinstance(sep_token, str) else sep_token
            # 如果 cls_token 是字符串，则将其封装为特殊的 AddedToken 对象；否则保持不变
            cls_token = AddedToken(cls_token, special=True) if isinstance(cls_token, str) else cls_token
            # 如果 unk_token 是字符串，则将其封装为特殊的 AddedToken 对象；否则保持不变
            unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
            # 如果 pad_token 是字符串，则将其封装为特殊的 AddedToken 对象；否则保持不变
            pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token

            # 将 mask_token 封装为特殊的 AddedToken 对象，且指定 lstrip=True 以保留其前面的空格
            mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token

            # 如果指定的 vocab_file 不是文件路径，则抛出 ValueError 异常
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                    " model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                )
            # 加载指定路径下的词汇表文件，并赋值给 self.vocab
            self.vocab = load_vocab(vocab_file)
            # 创建一个有序字典，将 self.vocab 中的键值对颠倒，以便通过 id 访问 token
            self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
            # 设置是否执行基本分词的标志
            self.do_basic_tokenize = do_basic_tokenize
            # 如果需要执行基本分词，则初始化 BasicTokenizer 对象并赋值给 self.basic_tokenizer
            if do_basic_tokenize:
                self.basic_tokenizer = BasicTokenizer(
                    do_lower_case=do_lower_case,
                    never_split=never_split,
                    tokenize_chinese_chars=tokenize_chinese_chars,
                    strip_accents=strip_accents,
                )
            # 使用给定的词汇表和未知 token 初始化 WordpieceTokenizer 对象，并赋值给 self.wordpiece_tokenizer
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

            # 调用父类的初始化方法，设置各种参数，包括 token 的特殊处理和其他参数
            super().__init__(
                do_lower_case=do_lower_case,
                do_basic_tokenize=do_basic_tokenize,
                never_split=never_split,
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                sep_token=sep_token,
                cls_token=cls_token,
                pad_token=pad_token,
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
            # 返回词汇表的大小，即 self.vocab 的长度
            return len(self.vocab)

        def get_vocab(self):
            # 创建一个新的字典，复制 added_tokens_encoder 中的内容并更新为 self.vocab 的内容
            # 返回更新后的字典
            vocab = self.added_tokens_encoder.copy()
            vocab.update(self.vocab)
            return vocab
    def _tokenize(self, text):
        """
        Tokenizes a single text string into a list of tokens using both basic and wordpiece tokenization.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            List[str]: List of tokens after tokenization.
        """
        split_tokens = []
        if self.do_basic_tokenize:
            # Iterate through tokens returned by basic_tokenizer
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # Check if token is in the set of never_split tokens
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)  # Append as is
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)  # Wordpiece tokenize
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)  # Use only wordpiece tokenizer
        return split_tokens

    def _convert_token_to_id(self, token):
        """
        Converts a token (str) to its corresponding ID using the vocabulary.

        Args:
            token (str): The token to be converted.

        Returns:
            int: The ID corresponding to the token. Defaults to the ID of unknown token if not found.
        """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """
        Converts an index (integer) to its corresponding token (str) using the vocabulary.

        Args:
            index (int): The index to be converted.

        Returns:
            str: The token corresponding to the index. Defaults to the unknown token if not found.
        """
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (list of str) into a single string, removing prefix '##'.

        Args:
            tokens (List[str]): List of tokens to be joined into a string.

        Returns:
            str: The concatenated string of tokens.
        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Builds model input from sequences, adding special tokens for sequence classification tasks in MPNet format.

        Args:
            token_ids_0 (`List[int]`):
                List of input IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional list of input IDs for the second sequence (for pair tasks).

        Returns:
            `List[int]`: List of input IDs with added special tokens appropriate for MPNet.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves a mask indicating the special tokens from input token IDs.

        Args:
            token_ids_0 (`List[int]`):
                List of input IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional list of input IDs for the second sequence (for pair tasks).
            already_has_special_tokens (`bool`):
                Flag indicating whether the input already includes special tokens.

        Returns:
            `List[int]`: Mask indicating special tokens (1 for special tokens, 0 otherwise).
        """
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Set to True if the token list is already formatted with special tokens for the model

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        # 如果已经有特殊标记，则调用父类的方法来获取特殊标记的掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果没有特殊标记，计算token_ids_0和token_ids_1的特殊标记掩码
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. MPNet does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # 创建用于序列对分类任务的掩码，对于MPNet来说，不使用token类型ID，因此返回一个零值列表
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        # 如果保存目录已存在
        if os.path.isdir(save_directory):
            # 拼接词汇表文件的路径
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开词汇表文件并写入词汇
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表中的token，按照token索引排序逐个写入文件
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 如果索引不连续，发出警告
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 写入token
                writer.write(token + "\n")
                index += 1
        # 返回保存的词汇表文件路径
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
        # 如果 `never_split` 参数未提供，则初始化为空列表
        if never_split is None:
            never_split = []
        # 设置是否将输入文本转换为小写
        self.do_lower_case = do_lower_case
        # 将 `never_split` 转换为集合，用于存储不会被分割的标记
        self.never_split = set(never_split)
        # 设置是否分词中文字符
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 设置是否去除所有的重音符号
        self.strip_accents = strip_accents
        # 设置是否在标点符号处分割单词
        self.do_split_on_punc = do_split_on_punc
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 使用 union() 方法将 self.never_split 和传入的 never_split 合并成一个新的集合，确保不分割的 token 集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清洗文本，去除不必要的字符和空格等
        text = self._clean_text(text)

        # 这段代码是在2018年11月1日为多语言和中文模型添加的。现在也适用于英语模型，尽管英语模型未经过任何中文数据训练，
        # 通常不包含中文数据（尽管英语维基百科中可能含有一些中文单词，所以词汇表中会包含一些中文字符）
        if self.tokenize_chinese_chars:
            # 对中文字符进行分词处理
            text = self._tokenize_chinese_chars(text)
        # 将文本中的Unicode字符规范化为NFC形式，以便更好地进行分词和处理
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空白字符分割文本，得到原始的token列表
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            # 如果token不在never_split集合中，进行进一步处理
            if token not in never_split:
                if self.do_lower_case:
                    # 如果需要小写处理，将token转换为小写
                    token = token.lower()
                    # 如果strip_accents不是False，移除token中的重音符号
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    # 如果需要strip_accents，移除token中的重音符号
                    token = self._run_strip_accents(token)
            # 根据标点符号进行token的分割，并将分割后的结果加入split_tokens列表
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 使用空白字符重新连接split_tokens中的token，得到最终的输出token列表
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 将文本中的Unicode字符规范化为NFD形式
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            # 获取字符的Unicode类别
            cat = unicodedata.category(char)
            # 如果字符类别为"Mn"（Mark, Nonspacing），表示该字符为重音符号，跳过该字符
            if cat == "Mn":
                continue
            # 否则将字符添加到输出列表中
            output.append(char)
        # 将列表中的字符连接成字符串，并返回
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要根据标点符号分割文本，或者文本在不分割列表中，则直接返回原文本作为列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        # 遍历字符列表
        while i < len(chars):
            char = chars[i]
            # 如果当前字符是标点符号，则作为新的单词分隔符，添加到输出列表中
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，且是新单词的开始，则创建一个新的空列表
                if start_new_word:
                    output.append([])
                start_new_word = False
                # 将当前字符添加到最后一个子列表中
                output[-1].append(char)
            i += 1

        # 将列表中的子列表连接成字符串并返回
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        # 遍历文本中的每个字符
        for char in text:
            cp = ord(char)
            # 如果字符是中文字符，则在其前后添加空格
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        # 将列表中的字符连接成字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查给定的 Unicode 代码点是否是中日韩字符（CJK 字符）
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
        # 遍历文本中的每个字符
        for char in text:
            cp = ord(char)
            # 如果字符是无效字符或者控制字符，则跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果字符是空白字符，则用单个空格替换
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将列表中的字符连接成字符串并返回
        return "".join(output)
# Copied from transformers.models.bert.tokenization_bert.WordpieceTokenizer
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化 WordpieceTokenizer 类的实例
        self.vocab = vocab  # 词汇表，用于词片段的匹配
        self.unk_token = unk_token  # 未知 token 的表示，如果无法匹配词片段
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
        output_tokens = []  # 存储最终的词片段 token 结果
        for token in whitespace_tokenize(text):  # 对文本进行分词，使用空白字符分隔
            chars = list(token)  # 将单词拆分为字符列表
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)  # 如果单词字符数超过最大限制，则添加未知 token
                continue

            is_bad = False  # 是否无法匹配词片段的标志
            start = 0  # 当前处理字符的起始位置
            sub_tokens = []  # 存储当前单词的词片段 token
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])  # 构建当前子串
                    if start > 0:
                        substr = "##" + substr  # 如果不是单词的开头，则加上 "##" 前缀
                    if substr in self.vocab:  # 如果词片段在词汇表中存在
                        cur_substr = substr  # 记录当前词片段
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True  # 如果无法找到匹配的词片段，则标记为无效
                    break
                sub_tokens.append(cur_substr)  # 将匹配的词片段加入到子 token 列表中
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)  # 如果整个单词无法匹配有效的词片段，则添加未知 token
            else:
                output_tokens.extend(sub_tokens)  # 将有效的词片段加入最终的输出结果中
        return output_tokens  # 返回最终的词片段 token 列表
```