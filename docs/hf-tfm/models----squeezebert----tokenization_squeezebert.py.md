# `.\transformers\models\squeezebert\tokenization_squeezebert.py`

```
# 设置 UTF-8 编码
# 版权声明和许可证信息
# 导入所需模块和函数
# 导入 logging 模块
import collections  # 导入 collections 模块
import os  # 导入 os 模块
import unicodedata  # 导入 unicodedata 模块
from typing import List, Optional, Tuple  # 导入 typing 模块

# 导入 tokenization_utils 模块中的 PreTrainedTokenizer 类和辅助函数
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
# 导入 logging 模块中的 get_logger 函数
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 定义预训练模型词汇文件的映射字典
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

# 定义预训练模型位置编码大小的字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "squeezebert/squeezebert-uncased": 512,
    "squeezebert/squeezebert-mnli": 512,
    "squeezebert/squeezebert-mnli-headless": 512,
}

# 定义预训练模型初始化配置的字典
PRETRAINED_INIT_CONFIGURATION = {
    "squeezebert/squeezebert-uncased": {"do_lower_case": True},
    "squeezebert/squeezebert-mnli": {"do_lower_case": True},
    "squeezebert/squeezebert-mnli-headless": {"do_lower_case": True},
}

# 从文件中加载词汇表到字典中
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()  # 创建一个有序字典对象
    with open(vocab_file, "r", encoding="utf-8") as reader:  # 打开词汇文件
        tokens = reader.readlines()  # 读取文件中的每一行
    for index, token in enumerate(tokens):  # 遍历每一个索引和词汇
        token = token.rstrip("\n")  # 去除行尾的换行符
        vocab[token] = index  # 将词汇和其索引添加到词汇表字典中
    return vocab  # 返回加载的词汇表字典

# 将文本基于空白字符进行分词
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()  # 去除文本首尾的空白字符
    if not text:  # 如果文本为空
        return []  # 返回空列表
    tokens = text.split()  # 使用空白字符分割文本
    return tokens  # 返回分词结果列表

# SqueezeBERT 分词器类
class SqueezeBertTokenizer(PreTrainedTokenizer):
    r"""
    Construct a SqueezeBERT tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    # 定义一个提供额外信息的基类，可以查看这里的方法更多信息。
    # 定义了以下参数：
    # vocab_file (`str`): 包含词汇表的文件。
    # do_lower_case (`bool`, *可选*, 默认为 `True`): 在进行标记化时是否将输入转换为小写。
    # do_basic_tokenize (`bool`, *可选*, 默认为 `True`): 是否在 WordPiece 之前进行基本标记化。
    # never_split (`Iterable`, *可选*): 在标记化时永远不会拆分的标记集合。仅在 `do_basic_tokenize=True` 时生效。
    # unk_token (`str`, *可选*, 默认为 `"[UNK]"`): 未知标记。无法转换为 ID 的标记将被设为该标记。
    # sep_token (`str`, *可选*, 默认为 `"[SEP]"`): 分隔符标记，用于从多个序列构建序列时使用，例如，序列分类或文本和问题的问题回答。还用作使用特殊标记构建的序列的最后一个标记。
    # pad_token (`str`, *可选*, 默认为 `"[PAD]"`): 用于填充的标记，例如当批量处理长度不同的序列时。
    # cls_token (`str`, *可选*, 默认为 `"[CLS]"`): 分类器标记，用于进行序列分类（整个序列的分类而不是每个标记的分类）。在使用特殊标记构建时，它是序列的第一个标记。
    # mask_token (`str`, *可选*, 默认为 `"[MASK]"`): 用于蒙版值的标记。这是训练时使用掩码语言建模的模型所使用的标记。这是模型将尝试预测的标记。
    # tokenize_chinese_chars (`bool`, *可选*, 默认为 `True`): 是否标记化中文字符。
    # 这可能应该对日文取消激活(参见此处问题)。

    # 是否删除所有重音符号。如果没有指定此选项，则将由 `lowercase` 的值确定(如原始 SqueezeBERT 中所示)。
    # vocab_files_names = VOCAB_FILES_NAMES
    # pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 初始化函数，接收参数并设置默认值
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
        # 检查输入的词汇文件是否存在，如果不存在则抛出数值错误
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = SqueezeBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 从词汇文件中加载词汇表
        self.vocab = load_vocab(vocab_file)
        # 创建从ID到词汇的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 设置基础分词选项
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要基础分词，则创建基础分词器对象
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 创建WordPiece分词器对象
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))
        
        # 调用父类初始化方法
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

    # 返回是否小写化的属性
    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    # 返回词汇表大小的属性
    @property
    def vocab_size(self):
        return len(self.vocab)

    # 返回词汇表的方法
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # 分词方法，将文本转换为分词后的列表
    def _tokenize(self, text, split_special_tokens=False):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果分词在never_split中，则直接加入，否则使用WordPieceTokenizer分词
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    # 将token转换为ID
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 将ID转换为token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)
    # 将一系列 tokens（字符串）转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        # 使用空格将 tokens 组合成一个字符串，并去除 "##"，然后去除首尾空格
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # 通过连接和添加特殊 token 来构建用于序列分类任务的模型输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A SqueezeBERT sequence has the following format:

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

        # 如果不存在第二个 token_ids_1，则返回包含特殊 token 的 token_ids_0
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 创建 CLS 和 SEP 的 token_ids
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 返回包含特殊 token 的 token_ids
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # 获取没有添加特殊 token 的 token 列表中的序列 id
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

        # 如果已经添加了特殊 token，则返回其特殊 token mask
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果不存在第二个 token_ids_1，则返回由特殊 token 构成的 mask
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    # 从序列构建 token type ids
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def create_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
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
        # Define the separator and class tokens
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # If token_ids_1 is None, only return the first portion of the mask (0s)
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # Return the mask for sequence pair
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        # Check if the save_directory is a directory
        if os.path.isdir(save_directory):
            # Define the vocabulary file path
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # Open the vocabulary file for writing
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # Iterate through the vocabulary items and write them to the file
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        # Return the vocabulary file path as a tuple
        return (vocab_file,)
# 定义 BasicTokenizer 类，用于运行基本的标记化（标点符号拆分、转换为小写等）
class BasicTokenizer(object):
    """
    创建一个 BasicTokenizer 类，它将运行基本的标记化（标点符号拆分、转换为小写等）。

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在标记化时将输入转换为小写。
        never_split (`Iterable`, *optional*):
            在标记化过程中永远不会被拆分的标记的集合。仅在 `do_basic_tokenize=True` 时生效
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否对中文字符进行标记化。

            这可能适用于日语（请参阅此 [issue](https://github.com/huggingface/transformers/issues/328)）时，应该停用此选项。
        strip_accents (`bool`, *optional*):
            是否去除所有重音符号。如果没有指定此选项，则将根据 `lowercase` 的值来确定。（与原始 BERT 相同）
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            在某些情况下，我们希望跳过基本的标点符号拆分，以便后续的标记化可以捕获单词的完整上下文，比如缩写。
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
    # 对给定文本进行基本的分词处理。对于子词分词，请参见WordPieceTokenizer。

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 如果存在never_split列表，则将其与self.never_split集合取并集，否则只使用self.never_split集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，去除不必要的空格等
        text = self._clean_text(text)

        # 以下部分是为了处理中文字符，在多语言和中文模型中添加了此功能，但由于英文模型没有在任何中文数据上训练，
        # 并且通常不包含任何中文数据（因为英文维基百科确实包含一些中文词汇），所以不影响英文模型。
        if self.tokenize_chinese_chars:
            # 对中文字符进行分词处理
            text = self._tokenize_chinese_chars(text)
        # 对文本进行unicode规范化，以便防止将具有不同unicode代码点的同一字符视为不同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空格分词函数对unicode规范化的文本进行分词
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        # 用于存储分词结果的列表
        split_tokens = []
        # 遍历原始分词结果
        for token in orig_tokens:
            # 如果分词结果不在never_split列表中，则进行处理
            if token not in never_split:
                # 如果设定了小写处理，则将分词结果转换为小写
                if self.do_lower_case:
                    token = token.lower()
                    # 如果需要去除重音符号，则进行处理
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果需要去除重音符号，则进行处理
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 将处理后的分词结果添加到split_tokens列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将处理后的分词结果通过空格连接，并使用空格分词函数再次对其进行分词
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        # 返回最终分词结果
        return output_tokens

    # 从文本中去除重音符号
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 对文本进行unicode规范化，以便去除重音符号
        text = unicodedata.normalize("NFD", text)
        # 存储处理后的文本字符列表
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的unicode分类
            cat = unicodedata.category(char)
            # 如果字符是重音符号，则跳过
            if cat == "Mn":
                continue
            # 将非重音符号的字符添加到output列表中
            output.append(char)
        # 将列表中的字符连接成字符串并返回
        return "".join(output)
    # 在文本上执行标点符号拆分
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不执行标点符号拆分，或者给定的文本不应该被拆分，则返回原始文本
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
            # 如果是标点符号，则创建新的子列表
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，将字符添加到当前子列表中
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        # 将子列表中的字符连接成字符串，并返回列表
        return ["".join(x) for x in output]

    # 在中文字符周围添加空格
    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        # 遍历文本中的每个字符
        for char in text:
            cp = ord(char)
            # 如果是中文字符，则在其前后添加空格
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        # 将列表中的字符连接成字符串
        return "".join(output)

    # 检查字符是否为中文字符
    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 这里将“中文字符”定义为CJK Unicode块中的任何内容
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

    # 对文本执行无效字符移除和空白清理
    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        # 遍历文本中的每个字符
        for char in text:
            cp = ord(char)
            # 如果字符为0或0xFFFD，或者是控制字符，则跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果字符是空白字符，则将其替换为单个空格
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将列表中的字符连接成字符串
        return "".join(output)
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化 WordpieceTokenizer 类
        self.vocab = vocab  # 词汇表
        self.unk_token = unk_token  # 未知标记
        self.max_input_chars_per_word = max_input_chars_per_word  # 每个单词的最大输入字符数

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

        output_tokens = []  # 存储最终的 wordpiece tokens
        for token in whitespace_tokenize(text):  # 对输入的文本进行空格分割
            chars = list(token)  # 将单词拆分为字符列表
            if len(chars) > self.max_input_chars_per_word:  # 如果单词长度超过最大输入字符数
                output_tokens.append(self.unk_token)  # 添加未知标记
                continue

            is_bad = False  # 标记是否出现问题
            start = 0  # 记录当前位置的起始索引
            sub_tokens = []  # 存储当前单词的 wordpiece tokens
            while start < len(chars):  # 当起始索引小于单词长度时执行
                end = len(chars)  # 设置结束索引为单词长度
                cur_substr = None  # 当前子串初始化为空
                while start < end:  # 当起始索引小于结束索引时执行
                    substr = "".join(chars[start:end])  # 从当前起始索引到结束索引构建子串
                    if start > 0:  # 如果起始索引大于0
                        substr = "##" + substr  # 在子串前添加 '##' 表示连接关系
                    if substr in self.vocab:  # 如果子串在词汇表中
                        cur_substr = substr  # 更新当前子串
                        break
                    end -= 1  # 结束索引减1
                if cur_substr is None:  # 如果当前子串为空
                    is_bad = True  # 标记出现问题
                    break
                sub_tokens.append(cur_substr)  # 将当前子串添加到 wordpiece tokens 中
                start = end  # 更新起始索引为结束索引

            if is_bad:  # 如果出现问题
                output_tokens.append(self.unk_token)  # 添加未知标记
            else:
                output_tokens.extend(sub_tokens)  # 添加当前单词的 wordpiece tokens
        return output_tokens  # 返回最终的 wordpiece tokens
```