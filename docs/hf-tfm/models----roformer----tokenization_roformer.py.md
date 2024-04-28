# `.\transformers\models\roformer\tokenization_roformer.py`

```
# 导入必要的库和模块
import collections
import os
import unicodedata
from typing import List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging

# 获取日志器
logger = logging.get_logger(__name__)

# 定义词汇表文件名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 定义预训练词汇表文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "junnyu/roformer_chinese_small": "https://huggingface.co/junnyu/roformer_chinese_small/resolve/main/vocab.txt",
        "junnyu/roformer_chinese_base": "https://huggingface.co/junnyu/roformer_chinese_base/resolve/main/vocab.txt",
        "junnyu/roformer_chinese_char_small": (
            "https://huggingface.co/junnyu/roformer_chinese_char_small/resolve/main/vocab.txt"
        ),
        "junnyu/roformer_chinese_char_base": (
            "https://huggingface.co/junnyu/roformer_chinese_char_base/resolve/main/vocab.txt"
        ),
        "junnyu/roformer_small_discriminator": (
            "https://huggingface.co/junnyu/roformer_small_discriminator/resolve/main/vocab.txt"
        ),
        "junnyu/roformer_small_generator": (
            "https://huggingface.co/junnyu/roformer_small_generator/resolve/main/vocab.txt"
        ),
    }
}

# 定义预训练位置嵌入的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "junnyu/roformer_chinese_small": 1536,
    "junnyu/roformer_chinese_base": 1536,
    "junnyu/roformer_chinese_char_small": 512,
    "junnyu/roformer_chinese_char_base": 512,
    "junnyu/roformer_small_discriminator": 128,
    "junnyu/roformer_small_generator": 128,
}

# 定义预训练初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "junnyu/roformer_chinese_small": {"do_lower_case": True},
    "junnyu/roformer_chinese_base": {"do_lower_case": True},
    "junnyu/roformer_chinese_char_small": {"do_lower_case": True},
    "junnyu/roformer_chinese_char_base": {"do_lower_case": True},
    "junnyu/roformer_small_discriminator": {"do_lower_case": True},
    "junnyu/roformer_small_generator": {"do_lower_case": True},
}

# 定义一个函数，用于从给定的词汇表文件中加载词汇表
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()


上述代码定义了一些与 RoFormer 相关的常量和函数。主要包括:

1. 定义了词汇表文件名称常量 `VOCAB_FILES_NAMES`。
2. 定义了预训练的词汇表文件的映射 `PRETRAINED_VOCAB_FILES_MAP`，用于获取不同预训练模型的词汇表文件地址。
3. 定义了预训练位置嵌入的大小 `PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES`。
4. 定义了预训练初始化配置 `PRETRAINED_INIT_CONFIGURATION`。
5. 实现了一个 `load_vocab` 函数，用于从给定的词汇表文件中加载词汇表。

这些常量和函数为 RoFormer 模型的使用和初始化提供了支持。
    # 遍历 tokens 列表，并返回每个元素的索引和值
    for index, token in enumerate(tokens):
        # 去除每个 token 值末尾的换行符
        token = token.rstrip("\n")
        # 将处理后的 token 作为 key，index 作为 value，添加到 vocab 字典中
        vocab[token] = index
    # 返回结果字典
    return vocab
# 从 transformers.models.bert.tokenization_bert.whitespace_tokenize 复制过来的函数
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # 去除文本两端的空白字符
    text = text.strip()
    # 如果文本为空，则返回空列表
    if not text:
        return []
    # 通过空白字符分割文本，得到 token 列表
    tokens = text.split()
    # 返回 token 列表
    return tokens


# 从 transformers.models.bert.tokenization_bert.BasicTokenizer 复制过来的类
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
        # 如果 never_split 未指定，则将其设为空列表
        if never_split is None:
            never_split = []
        # 设置是否将输入文本转换为小写
        self.do_lower_case = do_lower_case
        # 将 never_split 转换为集合，确保唯一性
        self.never_split = set(never_split)
        # 设置是否对中文字符进行分词
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 设置是否去除所有重音符号
        self.strip_accents = strip_accents
        # 设置是否在基本标点符号处分割文本
        self.do_split_on_punc = do_split_on_punc
    # 对文本进行基本的分词处理。对于子词分词，请参阅WordPieceTokenizer。

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 如果提供了never_split参数，则将其与实例中的never_split合并
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，去除不必要的字符
        text = self._clean_text(text)

        # 在2018年11月1日添加了这段代码，用于多语言和中文模型。
        # 现在也应用于英语模型，但这没关系，因为英语模型没有在任何中文数据上进行训练，
        # 通常不包含任何中文数据（因为英语维基百科中确实有一些中文词汇）。
        if self.tokenize_chinese_chars:
            # 如果需要，对文本中的中文字符进行分词处理
            text = self._tokenize_chinese_chars(text)
        # 将文本统一标准化为Unicode NFC格式
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 将文本按空格分割成原始token列表
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        # 初始化空列表用于存储拆分后的token
        split_tokens = []
        # 遍历原始token列表
        for token in orig_tokens:
            # 如果token不在不拆分列表中
            if token not in never_split:
                # 如果需要转换为小写
                if self.do_lower_case:
                    # 转换为小写
                    token = token.lower()
                    # 如果需要去除重音符号
                    if self.strip_accents is not False:
                        # 运行去除重音符号操作
                        token = self._run_strip_accents(token)
                # 如果需要去除重音符号
                elif self.strip_accents:
                    # 运行去除重音符号操作
                    token = self._run_strip_accents(token)
            # 将拆分后的token列表扩展至split_tokens列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将拆分后的token列表按空格连接为字符串，并按空格重新分割为输出tokens列表
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        # 返回输出tokens列表
        return output_tokens

    # 从文本中去除重音符号
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 将文本标准化为Unicode NFD格式
        text = unicodedata.normalize("NFD", text)
        # 初始化空列表output，用于存储去除重音符号后的文本
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的Unicode分类
            cat = unicodedata.category(char)
            # 如果字符的Unicode分类为"Mark, Nonspacing"
            if cat == "Mn":
                # 跳过该字符
                continue
            # 将字符添加到output列表中
            output.append(char)
        # 将output列表中的字符连接为字符串，并返回
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要按标点符号分割文本，或者文本在不分割列表中，则直接返回文本列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        # 初始化索引和新词开始标志
        i = 0
        start_new_word = True
        # 初始化输出列表
        output = []
        # 遍历字符列表
        while i < len(chars):
            char = chars[i]
            # 如果字符是标点符号，则将其作为新词添加到输出列表，并设置新词开始标志为 True
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果是非标点符号字符
                if start_new_word:
                    # 如果是新词的开头，则添加一个空列表作为新词
                    output.append([])
                start_new_word = False
                # 将字符添加到当前词的列表中
                output[-1].append(char)
            i += 1

        # 将列表中的字符列表合并为字符串列表并返回
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        # 初始化输出列表
        output = []
        # 遍历文本中的每个字符
        for char in text:
            cp = ord(char)
            # 如果字符是中文字符，则在其前后添加空格并添加到输出列表中
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                # 如果不是中文字符，则直接添加到输出列表中
                output.append(char)
        # 将输出列表中的字符合并为字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查字符的 Unicode 编码是否属于中文字符的范围
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
            cp = ord(char)
            # 如果字符为无效字符或控制字符，则跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果字符为空白字符，则将其替换为单个空格
            if _is_whitespace(char):
                output.append(" ")
            else:
                # 如果不是空白字符，则直接添加到输出列表中
                output.append(char)
        # 将输出列表中的字符合并为字符串并返回
        return "".join(output)
# 从transformers.models.bert.tokenization_bert.WordpieceTokenizer中复制代码到此处
# 定义WordpieceTokenizer类
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    # 初始化WordpieceTokenizer类的参数
    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    # 对文本进行分词
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

        output_tokens = []
        for token in whitespace_tokenize(text):  # 对文本进行空格分割
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


# 定义RoFormerTokenizer类，继承自PreTrainedTokenizer
class RoFormerTokenizer(PreTrainedTokenizer):
    r"""
    Construct a RoFormer tokenizer. Based on [Rust Jieba](https://pypi.org/project/rjieba/).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    # 定义一个函数，接受一些参数
    Args:
        # 词汇表文件名
        vocab_file (`str`):
            File containing the vocabulary.
        # 是否在进行标记化时将输入转换为小写
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        # 是否在WordPiece之前进行基本标记化
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece.
        # 永远不要分割的标记集合，在do_basic_tokenize=True时生效
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        # 未知标记，不在词汇表中的标记将被设置为此标记
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        # 分隔符标记，用于从多个序列构建序列时使用
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        # 用于填充的标记
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        # 分类器标记，用于进行序列分类
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        # 用于掩码值的标记，用于预训练的掩码语言建模
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        # 是否在标记化中分词汉字
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

    # 示例
    Example:

    ```python
    >>> from transformers import RoFormerTokenizer

    >>> tokenizer = RoFormerTokenizer.from_pretrained("junnyu/roformer_chinese_base")
    >>> tokenizer.tokenize("���天天气非常好。")
    ['今', '天', '天', '气', '非常', '好', '。']
    ```

    # 以下变量的含义需要从其他文件中获取，具体细节不在这里说明
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 初始化方法，接受参数：
    # - vocab_file: 词汇表文件路径
    # - do_lower_case: 是否将单词转换为小写，默认为True
    # - do_basic_tokenize: 是否进行基本分词，默认为True
    # - never_split: 不进行分词的单词列表，默认为None
    # - unk_token: 未知单词的标记，默认为"[UNK]"
    # - sep_token: 分隔符的标记，默认为"[SEP]"
    # - pad_token: 填充标记，默认为"[PAD]"
    # - cls_token: 类别标记，默认为"[CLS]"
    # - mask_token: 掩码标记，默认为"[MASK]"
    # - tokenize_chinese_chars: 是否进行中文字符分词，默认为True
    # - strip_accents: 是否去除重音符号，默认为None
    # - **kwargs: 其他关键字参数
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
        # 如果词汇表文件不存在，则抛出数值错误
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表
        self.vocab = load_vocab(vocab_file)
        # 将词汇表转换成按顺序排列的单词到编号的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 是否进行基本分词
        self.do_basic_tokenize = do_basic_tokenize
        # 如果进行基本分词，则初始化基本分词器
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 初始化词片分词器
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))
        # 尝试导入 rjieba 库，如果导入失败则抛出导入错误
        try:
            import rjieba
        except ImportError:
            raise ImportError(
                "You need to install rjieba to use RoFormerTokenizer. "
                "See https://pypi.org/project/rjieba/ for installation."
            )
        # 初始化结巴分词器
        self.jieba = rjieba

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

    # 返回是否进行小写处理
    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    # 返回词汇表大小
    @property
    def vocab_size(self):
        return len(self.vocab)

    # 获取对象的状态
    def __getstate__(self):
        state = self.__dict__.copy()
        state["jieba"] = None
        return state

    # 设置对象的状态
    def __setstate__(self, d):
        self.__dict__ = d
        import rjieba

        self.jieba = rjieba

    # 获取词汇表
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)
    def _tokenize(self, text, use_jieba=True):
        # 初始化空列表以存储分词结果
        split_tokens = []
        # 若使用结巴分词
        if use_jieba:
            # 遍历结巴分词的结果
            for wholword in self.jieba.cut(text, False):
                # 如果分词结果在词汇表中
                if wholword in self.vocab:
                    # 将分词结果添加到列表中
                    split_tokens.append(wholword)
                else:
                    # 否则，使用BERT分词器进行分词
                    char_list = self._tokenize(wholword, use_jieba=False)
                    # 将分词器结果扩展到列表中
                    split_tokens.extend(char_list)
        else:
            # 如果执行基本分词
            if self.do_basic_tokenize:
                # 使用基本分词器分词
                for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                    # 如果分词结果在不分割集合中
                    if token in self.basic_tokenizer.never_split:
                        # 将分词结果添加到列表中
                        split_tokens.append(token)
                    else:
                        # 否则，使用词块分词器进行分词
                        split_tokens += self.wordpiece_tokenizer.tokenize(token)
            else:
                # 否则，使用词块分词器进行分词
                split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将标记（字符串）转换为 ID，使用词汇表
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将索引（整数）转换为标记（字符串），使用词汇表
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将一系列标记（字符串）转换为单个字符串
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A RoFormer sequence has the following format:

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
        # 如果没有第二个token序列
        if token_ids_1 is None:
            # 返回带有特殊token的输入ID列表
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 返回带有特殊token的输入ID列表（包含两个序列）
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
        ):
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

        # 如果已经添加了特殊标记，则调用父类方法以获取特殊标记的掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果有第二个序列的 token_ids_1，则创建一个特殊标记掩码
        if token_ids_1 is not None:
            # 返回一个特殊标记掩码，其中包括第一个序列的特殊标记、第一个序列的序列标记、第二个序列的特殊标记和第二个序列的序列标记
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # 如果没有第二个序列的 token_ids_1，则只创建第一个序列的特殊标记掩码
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A RoFormer
        sequence pair mask has the following format:

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
        # 获取分隔符和类别标记
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # 如果没有第二个序列的 token_ids_1，则只返回第一个序列的 token 类别标记掩码
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 如果有第二个序列的 token_ids_1，则返回包含两个序列的 token 类别标记掩码
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    # 保存词汇表到文件
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引为 0
        index = 0
        # 检查保存目录是否存在
        if os.path.isdir(save_directory):
            # 构建词汇文件路径，如果提供了文件名前缀，则使用前缀，否则为空，连接后加上默认的词汇表文件名
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 如果保存目录不存在，则直接使用保存目录作为文件路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开文件准备写入
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表，按照索引排序
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 如果当前索引与预期索引不一致，发出警告
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 将词汇写入文件，并换行
                writer.write(token + "\n")
                # 索引递增
                index += 1
        # 返回保存的文件路径
        return (vocab_file,)
```