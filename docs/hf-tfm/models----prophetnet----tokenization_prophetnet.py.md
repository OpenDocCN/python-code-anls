# `.\transformers\models\prophetnet\tokenization_prophetnet.py`

```py
# 导入必要的模块和类型注解
import collections
import os
import unicodedata
from typing import Iterable, List, Optional, Tuple

# 从 tokenization_utils 模块导入 PreTrainedTokenizer 类以及一些辅助函数
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 vocab_file 作为 tokenizer 文件名
VOCAB_FILES_NAMES = {"vocab_file": "prophetnet.tokenizer"}

# 定义预训练的 tokenizer 文件所在的 URL
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/prophetnet-large-uncased": (
            "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/prophetnet.tokenizer"
        ),
    }
}

# 定义预训练的 tokenizer 初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/prophetnet-large-uncased": {"do_lower_case": True},
}

# 定义预训练的 tokenizer 位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/prophetnet-large-uncased": 512,
}

# 复制自 transformers.models.bert.tokenization_bert.whitespace_tokenize
# 定义一个基于空白符进行分词的函数
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # 去除文本前后的空白字符
    text = text.strip()
    # 如果文本为空，返回空列表
    if not text:
        return []
    # 使用空白符分割文本，返回分词结果
    tokens = text.split()
    return tokens

# 复制自 transformers.models.bert.tokenization_bert.BasicTokenizer
# 定义一个基本的 tokenizer 类
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
    # 初始化方法，用于设置Tokenizer的各种参数
    def __init__(
        self,
        do_lower_case=True,  # 是否将文本转换为小写，默认为True
        never_split=None,    # 指定不需要拆分的token列表，默认为None
        tokenize_chinese_chars=True,  # 是否拆分中文字符，默认为True
        strip_accents=None,   # 是否去除文本中的重音，默认为None
        do_split_on_punc=True,  # 是否根据标点符号进行拆分，默认为True
    ):
        # 如果never_split为None，则将其设为空列表
        if never_split is None:
            never_split = []
        # 将参数赋值给对象的属性
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc

    # 对文本进行基本的分词操作，用于构建token
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 如果传入的never_split为None，则将其设为初始化时的never_split
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，去除不需要的字符
        text = self._clean_text(text)

        # 如果设定了拆分中文字符的选项，则拆分中文字符
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 将文本进行unicode标准化，避免将不同unicode编码的同一字符看作不同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 将文本按空格进行分词
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        # 遍历分词后的原始token列表
        for token in orig_tokens:
            # 如果token不在never_split列表中
            if token not in never_split:
                # 如果需要将文本转换为小写，则将token转换为小写
                if self.do_lower_case:
                    token = token.lower()
                    # 如果strip_accents不为False，则去除token中的重音
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果strip_accents为True，则去除token中的重音
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 将token进行标点符号拆分，并将结果扩展到split_tokens列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将拆分后的token列表按空格连接成字符串，并重新进行空格分词，得到最终的token列表
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        # 返回最终的token列表
        return output_tokens

    # 去除文本中的重音
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 将文本进行unicode标准化，将字符分解为基本字符和组合字符
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的字符
        for char in text:
            # 获取字符的Unicode类别
            cat = unicodedata.category(char)
            # 如果字符的Unicode类别为Mn（Mark, Nonspacing），则跳过该字符
            if cat == "Mn":
                continue
            # 将非重音字符加入输出列表中
            output.append(char)
        # 将输出列表中的字符连接成字符串并返回
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要在标点符号处分割或者给定的文本在never_split列表中，则直接返回文本列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        # 初始化循环变量
        i = 0
        # 标志是否开始新单词
        start_new_word = True
        # 输出结果列表
        output = []
        # 循环遍历文本字符列表
        while i < len(chars):
            # 当前字符
            char = chars[i]
            # 如果是标点符号，则将其作为单独的词添加到输出列表中
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果开始新单词，则在输出列表中添加一个空列表
                if start_new_word:
                    output.append([])
                start_new_word = False
                # 将当前字符添加到最后一个单词列表中
                output[-1].append(char)
            i += 1

        # 将列表中的字符列表连接成字符串列表并返回
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        # 输出结果列表
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 码点
            cp = ord(char)
            # 如果是中文字符，则在其前后添加空格
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
        # 判断码点是否在中文字符范围内
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
        # 输出结果列表
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 码点
            cp = ord(char)
            # 如果字符是无效字符或者控制字符，则跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果字符是空白字符，则替换为空格，否则保留字符
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将列表中的字符连接成字符串并返回
        return "".join(output)
# 从transformers.models.bert.tokenization_bert.WordpieceTokenizer复制而来
# WordpieceTokenizer类，运行WordPiece分词

class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab  # 词汇表
        self.unk_token = unk_token  # 未知token
        self.max_input_chars_per_word = max_input_chars_per_word  # 单词最大字符数

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

        output_tokens = []  # 输出的词片段列表
        for token in whitespace_tokenize(text):  # 对文本进行空格分词
            chars = list(token)  # 将token转换为字符列表
            if len(chars) > self.max_input_chars_per_word:  # 如果字符数超过单词最大字符数
                output_tokens.append(self.unk_token)  # 将未知token添加到输出词片段列表中
                continue

            is_bad = False  # 是否为不合适的标记
            start = 0  # 起始位置
            sub_tokens = []  # 子词片段列表
            while start < len(chars):  # 当起始位置小于字符列表长度时
                end = len(chars)  # 结束位置为字符列表长度
                cur_substr = None  # 当前子串为空
                while start < end:  # 当起始位置小于结束位置时
                    substr = "".join(chars[start:end])  # 将字符列表[start:end]连接成字符串
                    if start > 0:  # 如果起始位置大于0
                        substr = "##" + substr  # 在子串前添加"##"
                    if substr in self.vocab:  # 如果子串在词汇表中
                        cur_substr = substr  # 当前子串为substr
                        break
                    end -= 1  # 结束位置减一
                if cur_substr is None:  # 如果当前子串为空
                    is_bad = True  # 设置为不合适的标记
                    break
                sub_tokens.append(cur_substr)  # 将当前子串添加到子词片段列表中
                start = end  # 将结束位置赋给起始位置

            if is_bad:  # 如果为不合适的标记
                output_tokens.append(self.unk_token)  # 将未知token添加到输出词片段列表中
            else:
                output_tokens.extend(sub_tokens)  # 否则，将子词片段列表扩展到输出词片段列表中
        return output_tokens  # 返回输出词片段列表


# 加载词汇表文件到字典中
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()  # 创建有序字典
    with open(vocab_file, "r", encoding="utf-8") as reader:  # 打开词汇表文件，使用utf-8编码
        tokens = reader.readlines()  # 读取每一行作为token
    for index, token in enumerate(tokens):  # 遍历tokens
        token = token.rstrip("\n")  # 去掉每个token结尾的换行符
        vocab[token] = index  # 将token添加到词汇表中
    return vocab  # 返回词汇表


# ProphetNetTokenizer类，基于WordPiece构建
# 这个分词器继承自PreTrainedTokenizer，其中包含大部分主要方法。用户应参考这个超类以获取有关这些方法的更多信息。
    Args:
        vocab_file (`str`):
            File containing the vocabulary. 词汇表文件路径
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing. 是否在标记化时将输入转换为小写
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            Whether or not to do basic tokenization before WordPiece. 是否在使用 WordPiece 之前进行基本标记化
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when `do_basic_tokenize=True`
            不会在标记化过程中拆分的标记集合
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. 未知标记
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens. 分隔符标记
        x_sep_token (`str`, *optional*, defaults to `"[X_SEP]"`):
            Special second separator token, which can be generated by [`ProphetNetForConditionalGeneration`]. It is
            used to separate bullet-point like sentences in summarization, *e.g.*. 特殊分隔符标记
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths. 用于填充的标记
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict. 用于掩盖值的标记
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)). 是否标记化中文字符
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT). 是否去除所有重音符号

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # first name has to correspond to main model input name
    # to make sure `tokenizer.pad(...)` works correctly
    # `ProphetNet` doesn't have `token_type_ids` as argument.
    model_input_names: List[str] = ["input_ids", "attention_mask"]
    # 初始化函数，接受多个参数，其中vocab_file为词汇表文件路径，其他参数有默认值
    def __init__(
        self,
        vocab_file: str,
        do_lower_case: Optional[bool] = True,
        do_basic_tokenize: Optional[bool] = True,
        never_split: Optional[Iterable] = None,
        unk_token: Optional[str] = "[UNK]",
        sep_token: Optional[str] = "[SEP]",
        x_sep_token: Optional[str] = "[X_SEP]",
        pad_token: Optional[str] = "[PAD]",
        mask_token: Optional[str] = "[MASK]",
        tokenize_chinese_chars: Optional[bool] = True,
        strip_accents: Optional[bool] = None,
        **kwargs,
    ):
        # 如果指定的词汇表文件不存在，则抛出数值错误异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表文件为词汇表
        self.vocab = load_vocab(vocab_file)
        # 将词汇表转换为有序字典，键为id，值为token
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要基本分词则创建 BasicTokenizer 对象
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 创建 WordpieceTokenizer 对象
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的初始化函数
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            x_sep_token=x_sep_token,
            pad_token=pad_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

    @property
    # 获取词汇表大小的属性
    def vocab_size(self):
        return len(self.vocab)

    # 获取词汇表的方法
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # 分词方法，将文本转换为单词片段列表
    def _tokenize(self, text):
        split_tokens = []
        # 如果需要基本分词，则调用基本分词器进行分词
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # 如果单词是无需分割的特殊单词，则直接添加到结果里
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 否则直接使用 WordpieceTokenizer 进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    # 将 token 转换为 id 的方法，若找不到则使用 unk_token
    def _convert_token_to_id(self, token: str):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 将 id 转换为 token 的方法，若找不到则使用 unk_token
    def _convert_id_to_token(self, index: int):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)
    def convert_tokens_to_string(self, tokens: str):
        """
        Converts a sequence of tokens (string) in a single string.

        Args:
            tokens (str): A sequence of tokens.

        Returns:
            str: A single string formed by joining the tokens.
        """
        # Join tokens into a single string, remove any '##' indicating subwords, and strip leading/trailing whitespace
        out_string = " ".join(tokens).replace(" ##", "").strip()
        # Return the resulting string
        return out_string

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: Optional[bool] = False,
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (List[int]): List of IDs.
            token_ids_1 (List[int], optional): Optional second list of IDs for sequence pairs.
            already_has_special_tokens (bool, optional): Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            List[int]: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        # If the token list already has special tokens, call the parent class method to get special tokens mask
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If token_ids_1 is None, return a mask with zeros for token_ids_0 length followed by 1
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        # Otherwise, return a mask with zeros for token_ids_0 length, 1, zeros for token_ids_1 length, and 1
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A ProphetNet
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```py

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (List[int]): List of IDs.
            token_ids_1 (List[int], optional): Optional second list of IDs for sequence pairs.

        Returns:
            List[int]: List of token type IDs according to the given sequence(s).
        """
        # Create a separator token list
        sep = [self.sep_token_id]
        # If token_ids_1 is None, return a mask with zeros for token_ids_0 length plus 1
        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0]
        # Otherwise, return a mask with zeros for token_ids_0 length, 1, zeros for token_ids_1 length, and 1
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引
        index = 0
        # 判断保存目录是否为目录类型
        if os.path.isdir(save_directory):
            # 构建词汇表文件路径，包括保存目录和文件名前缀
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 构建词汇表文件路径，包括文件名前缀和保存目录
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开词汇表文件进行写入操作
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表，按索引排序
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 检查是否有不连续的索引
                if index != token_index:
                    # 记录警告日志
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 写入词汇表文件
                writer.write(token + "\n")
                index += 1
        # 返回保存的词汇表文件路径
        return (vocab_file,)

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
            `List[int`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 判断是否为单个句子
        if token_ids_1 is None:
            return token_ids_0 + [self.sep_token_id]
        # 构建包含特殊token的输入序列
        sep = [self.sep_token_id]
        return token_ids_0 + sep + token_ids_1 + sep
```