# `.\models\electra\tokenization_electra.py`

```py
# 设置编码方式为utf-8
# 版权声明
# 根据Apache许可证2.0授权
# 仅在遵守许可证的情况下使用该文件
# 您可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，基于许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以确定特定语言的许可权限和限制

# 导入所需的库
import collections
import os
import unicodedata
from typing import List, Optional, Tuple
# 从tokenization_utils中导入PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace函数
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging

# 获取logger对象
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/electra-small-generator": (
            "https://huggingface.co/google/electra-small-generator/resolve/main/vocab.txt"
        ),
        "google/electra-base-generator": "https://huggingface.co/google/electra-base-generator/resolve/main/vocab.txt",
        "google/electra-large-generator": (
            "https://huggingface.co/google/electra-large-generator/resolve/main/vocab.txt"
        ),
        "google/electra-small-discriminator": (
            "https://huggingface.co/google/electra-small-discriminator/resolve/main/vocab.txt"
        ),
        "google/electra-base-discriminator": (
            "https://huggingface.co/google/electra-base-discriminator/resolve/main/vocab.txt"
        ),
        "google/electra-large-discriminator": (
            "https://huggingface.co/google/electra-large-discriminator/resolve/main/vocab.txt"
        ),
    }
}

# 预训练模型的位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/electra-small-generator": 512,
    "google/electra-base-generator": 512,
    "google/electra-large-generator": 512,
    "google/electra-small-discriminator": 512,
    "google/electra-base-discriminator": 512,
    "google/electra-large-discriminator": 512,
}

# 预训练模型的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "google/electra-small-generator": {"do_lower_case": True},
    "google/electra-base-generator": {"do_lower_case": True},
    "google/electra-large-generator": {"do_lower_case": True},
    "google/electra-small-discriminator": {"do_lower_case": True},
    "google/electra-base-discriminator": {"do_lower_case": True},
    "google/electra-large-discriminator": {"do_lower_case": True},
}

# 从transformers.models.bert.tokenization_bert.load_vocab复制函数
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    # 使用collections.OrderedDict创建一个有序字典
    vocab = collections.OrderedDict()
    # 以utf-8编码方式打开词汇文件，读取内容到tokens列表中
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    # 遍历tokens列表，同时获取索引和元素
    for index, token in enumerate(tokens):
        # 去除token末尾的换行符
        token = token.rstrip("\n")
        # 将token作为键，index作为值，添加到vocab字典中
        vocab[token] = index
    # 返回构建好的vocab字典
    return vocab
# 从transformers.models.bert.tokenization_bert.whitespace_tokenize复制而来，用于对文本进行基本的空格清理和分词
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # 去除文本两端的空白字符
    text = text.strip()
    # 如果文本为空，则返回空列表
    if not text:
        return []
    # 使用空格对文本进行分词
    tokens = text.split()
    # 返回分词结果列表
    return tokens


# 从transformers.models.bert.tokenization_bert.BertTokenizer复制而来，将Bert改为Electra，BERT改为Electra
class ElectraTokenizer(PreTrainedTokenizer):
    r"""
    Construct a Electra tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    # 定义一个类，用于处理和构建词汇表
    class BertTokenizer(PreTrainedTokenizer):
    
        # 初始化函数，用于初始化词汇表相关设置
        def __init__(
            self,
            # 包含词汇表的文件路径
            vocab_file,
            # 是否将输入内容转换为小写
            do_lower_case=True,
            # 是否进行基本的分词处理
            do_basic_tokenize=True,
            # 永远不会被分割的标记集合
            never_split=None,
            # 未知标记，默认为"[UNK]"
            unk_token="[UNK]",
            # 分隔标记，默认为"[SEP]"
            sep_token="[SEP]",
            # 填充标记，默认为"[PAD]"
            pad_token="[PAD]",
            # 分类标记，默认为"[CLS]"
            cls_token="[CLS]",
            # 掩盖标记，默认为"[MASK]"
            mask_token="[MASK]",
            # 是否对中文进行分词处理，默认为True
            tokenize_chinese_chars=True,
            # 是否去除所有重音符号
            strip_accents=None,
            **kwargs,
        
        # 词汇表文件的默认名称列表
        vocab_files_names = VOCAB_FILES_NAMES
        # 预训练模型的词汇表文件映射
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        # 预训练模型的初始化配置
        pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
        # 预训练模型的最大输入大小
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义初始化函数，接收单词表文件路径和一些参数
    ):
        # 如果单词表文件不存在，抛出数值错误异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = ElectraTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载单词表
        self.vocab = load_vocab(vocab_file)
        # 创建 token 到 id 的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 设置是否进行基本分词
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要基本分词，则创建基本分词器
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        # 创建 WordPiece 分词器
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的初始化函数
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

    # 获得是否进行小写处理的属性
    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    # 返回单词表的大小
    @property
    def vocab_size(self):
        return len(self.vocab)

    # 获取单词表
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # 分词
    def _tokenize(self, text, split_special_tokens=False):
        split_tokens = []
        # 如果需要基本分词
        if self.do_basic_tokenize:
            # 对文本进行基本分词
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果 token 在不分割集合中
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    # 将 token 转换为 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 将 id 转换为 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    # 将 token 序列转换为字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # 构建包含特殊 token 的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 定义一个方法，用于处理序列或序列对以生成用于序列分类任务的模型输入，在序列中连接和添加特殊标记。一个 Electra 序列具有以下格式：
    
    - 单个序列：`[CLS] X [SEP]`
    - 序列对：`[CLS] A [SEP] B [SEP]`
    
    # Args:
    #     token_ids_0 (`List[int]`):
    #         将添加特殊标记的 ID 列表。
    #     token_ids_1 (`List[int]`, *optional*):
    #         用于序列对的第二个可选 ID 列表。
    
    # Returns:
    #     `List[int]`: 具有适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep
    
    # 检索从未添加特殊标记的令牌列表中获取的序列 ID。在使用 tokenizer 的 `prepare_for_model` 方法添加特殊标记时调用此方法。
    
    # Args:
    #     token_ids_0 (`List[int]`):
    #         ID 列表。
    #     token_ids_1 (`List[int]`, *optional*):
    #         用于序列对的第二个可选 ID 列表。
    #     already_has_special_tokens (`bool`, *optional*, 默认值为 `False`):
    #         表示标记列表是否已格式化为模型的特殊标记。
    
    # Returns:
    #     `List[int]`: 一个整数列表，范围为 [0, 1]：特殊标记为 1，序列标记为 0。
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
    
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
    
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]
    
    # 从序列中创建令牌类型 ID，该序列没有添加特殊标记。
    
    # Args:
    #     token_ids_0 (`List[int]`):
    #         ID 列表。
    #     token_ids_1 (`List[int]`, *optional*):
    #         用于序列对的第二个可选 ID 列表。
    # 创建一个用于序列对分类任务的mask，格式为0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1，分隔符前为第一个序列，分隔符后为第二个序列
    def create_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        sep = [self.sep_token_id]  # 分隔符标识的token id
        cls = [self.cls_token_id]  # 文本开始标识的token id
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]  # 如果没有第二个序列，返回只含有第一个序列的mask
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]  # 否则返回包含两个序列的mask

    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0  # 索引
        if os.path.isdir(save_directory):  # 如果save_directory是目录
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )  # 生成词汇文件路径
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory  # 否则直接使用save_directory作为词汇文件路径
        with open(vocab_file, "w", encoding="utf-8") as writer:  # 打开词汇文件进行写入
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):  # 遍历词汇表的token和索引
                if index != token_index:  # 如果索引不连续
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )  # 输出警告信息
                    index = token_index  # 更新索引
                writer.write(token + "\n")  # 写入token到文件中
                index += 1  # 更新索引
        return (vocab_file,)  # 返回词汇文件路径的元组
# Copied from transformers.models.bert.tokenization_bert.BasicTokenizer
# 定义 BasicTokenizer 类，用于执行基本的分词（标点符号分割，小写化等）

class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
            是否在进行分词时将输入内容转换为小写，默认为 True
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
            在分词时永远不会被分割的 token 集合，仅在执行基本分词时有效
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.
            是否对中文字符进行分词，默认为 True
            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
            这对于日语可能需要停用（参考这个 issue）
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
            是否去除所有的重音符号。如果没有指定此选项，则由 `lowercase` 的值来确定（就像原始的 BERT 一样）
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            In some instances we want to skip the basic punctuation splitting so that later tokenization can capture
            the full context of the words, such as contractions.
            在某些情况下，我们希望跳过基本的标点符号分割，以便后续的分词可以捕捉单词的完整上下文，比如缩略词。
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc
    # 对给定的文本进行基本的分词。对于子词分词，请参见 WordPieceTokenizer。

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 如果传入了 never_split 参数，则将其与 self.never_split 合并为一个集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，去除无用字符
        text = self._clean_text(text)

        # 这段代码是在 2018 年 11 月添加的，用于多语言和中文模型。
        # 现在也适用于英文模型，但由于英文模型没有训练任何中文数据，通常不会有中文数据（英文维基百科中有一些中文单词）。
        if self.tokenize_chinese_chars:
            # 对中文字符进行分词处理
            text = self._tokenize_chinese_chars(text)
        # 将具有不同 Unicode 编码点的同一字符视为相同的字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空格进行分词
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        # 对每个原始 token 进行处理
        for token in orig_tokens:
            # 如果 token 不在 never_split 中
            if token not in never_split:
                # 如果需要小写化 token
                if self.do_lower_case:
                    # 将 token 转换为小写
                    token = token.lower()
                    # 如果需要去除重音符号
                    if self.strip_accents is not False:
                        # 去除 token 中的重音符号
                        token = self._run_strip_accents(token)
                # 如果需要去除重音符号
                elif self.strip_accents:
                    # 去除 token 中的重音符号
                    token = self._run_strip_accents(token)
            # 对 token 进行标点符号分割
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 使用空格重新连接分割后的 token，形成最终的输出 token
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    # 去除文本中的重音符号
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 将文本中的字符标准化为 NFD 格式
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 分类
            cat = unicodedata.category(char)
            # 如果字符为重音符号，则跳过
            if cat == "Mn":
                continue
            # 将非重音符号的字符加入输出列表中
            output.append(char)
        # 将输出列表中的字符连接成字符串
        return "".join(output)
```  
    # 在文本上执行标点符号拆分
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不执行标点符号拆分，或者文本在永不拆分的列表中，则返回原始文本列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        # 初始化索引和新词标记
        i = 0
        start_new_word = True
        # 初始化输出列表
        output = []
        # 遍历字符列表
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号
            if _is_punctuation(char):
                # 将标点符号作为一个新词添加到输出列表中
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号
                if start_new_word:
                    # 如果是新词的开头，创建一个新的词
                    output.append([])
                start_new_word = False
                # 将字符添加到当前词的列表中
                output[-1].append(char)
            i += 1

        # 将列表中的子列表合并为字符串
        return ["".join(x) for x in output]

    # 对中文字符进行标记化处理，添加空格
    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        # 初始化输出列表
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 码点
            cp = ord(char)
            # 如果是中文字符，添加空格，并在字符前后加上空格
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                # 如果不是中文字符，直接添加到输出列表中
                output.append(char)
        # 将输出列表中的字符合并为字符串
        return "".join(output)

    # 检查字符的码点是否是中文字符
    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 定义中文字符的 Unicode 块范围
        # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
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

    # 清理文本中的无效字符和空白字符
    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        # 初始化输出列表
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 码点
            cp = ord(char)
            # 如果字符是无效字符或控制字符，则跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果是空白字符，替换为单个空格
            if _is_whitespace(char):
                output.append(" ")
            else:
                # 如果不是空白字符，则直接添加到输出列表中
                output.append(char)
        # 将输出列表中的字符合并为字符串
        return "".join(output)
# 从transformers.models.bert.tokenization_bert.WordpieceTokenizer中复制的代码
class WordpieceTokenizer(object):
    """运行WordPiece标记化。"""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化WordpieceTokenizer对象，设置词汇表、未知标记和每个词的最大输入字符数
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        将文本标记化成其单词片段。这使用贪婪的最长匹配优先算法来使用给定的词汇表进行标记化。

        例如，`输入 = "unaffable"` 将返回为输出 `["un", "##aff", "##able"]`。

        Args:
            text: 一个单个标记或以空格分隔的标记。这应该已经通过*BasicTokenizer* 处理过。

        Returns:
            单词片段标记的列表。
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