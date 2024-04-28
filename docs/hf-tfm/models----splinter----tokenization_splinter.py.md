# `.\transformers\models\splinter\tokenization_splinter.py`

```
# 定义了代码文件的编码方式为 utf-8
# 版权声明，版权归 Tel AViv University, AllenAI 和 The HuggingFace Inc. 团队所有
# 版权声明，保留所有权利
#
# 根据 Apache 许可证版本 2.0 许可，除非符合许可，否则不得使用此文件
# 您可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
#
# 除非适用法律规定或书面同意，否则根据许可分发的软件均基于"按原样"基础分发
# 没有明示或暗示的担保或任何形式的条件，详情请见许可
# 标记化类为 Splinter

import collections
import os
import unicodedata
from typing import List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "tau/splinter-base": "https://huggingface.co/tau/splinter-base/resolve/main/vocab.txt",
        "tau/splinter-base-qass": "https://huggingface.co/tau/splinter-base-qass/resolve/main/vocab.txt",
        "tau/splinter-large": "https://huggingface.co/tau/splinter-large/resolve/main/vocab.txt",
        "tau/splinter-large-qass": "https://huggingface.co/tau/splinter-large-qass/resolve/main/vocab.txt",
    }
}

# 预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "tau/splinter-base": 512,
    "tau/splinter-base-qass": 512,
    "tau/splinter-large": 512,
    "tau/splinter-large-qass": 512,
}

# 预训练初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "tau/splinter-base": {"do_lower_case": False},
    "tau/splinter-base-qass": {"do_lower_case": False},
    "tau/splinter-large": {"do_lower_case": False},
    "tau/splinter-large-qass": {"do_lower_case": False},
}

# 加载词汇表到字典
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

# 在文本上运行基本的空格清理和拆分
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

# Splinter 分词器类，基于 WordPiece
class SplinterTokenizer(PreTrainedTokenizer):
    r"""
    Construct a Splinter tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            词汇表文件路径。
        do_lower_case (`bool`, *optional*, defaults to `True`):
            在进行标记化时是否将输入转换为小写。
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            在使用 WordPiece 进行标记化之前是否进行基本标记化。
        never_split (`Iterable`, *optional*):
            在标记化时永远不会被拆分的标记集合。仅在 `do_basic_tokenize=True` 时有效。
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            未知标记。如果一个标记不在词汇表中，则无法转换为 ID，并将其设置为此标记。
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            分隔符标记，用于从多个序列构建序列，例如用于序列分类或用于文本和问题的问题回答。也用作带有特殊标记的序列的最后一个标记。
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            用于填充的标记，例如在批处理不同长度的序列时。
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            分类器标记，在进行序列分类（整个序列而不是每个标记的分类）时使用。它是使用特殊标记构建的序列的第一个标记。
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            用于屏蔽值的标记。这是在使用掩码语言模型进行此模型训练时使用的标记。这是模型将尝试预测的标记。
        question_token (`str`, *optional*, defaults to `"[QUESTION]"`):
            用于构建问题表示的标记。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否对中文字符进行标记化。

            对于日语，这可能应该停用（参见此问题）。
        strip_accents (`bool`, *optional*):
            是否去除所有重音符号。如果未指定此选项，则将由 `lowercase` 的值确定（与原始 BERT 相同）。
    """
    # 词汇表文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇表文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型的初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 预训练模型的最大输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 初始化方法，接受一系列参数，设置tokenizer的属性
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
        question_token="[QUESTION]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        # 检查是否存在给定的词汇文件，如果不存在则抛出异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表
        self.vocab = load_vocab(vocab_file)
        # 将词汇表转换为有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要基本分词，则实例化BasicTokenizer对象
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 实例化WordpieceTokenizer对象
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))
        self.question_token = question_token
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

    # 获取问题标记的id
    @property
    def question_token_id(self):
        """
        `Optional[int]`: Id of the question token in the vocabulary, used to condition the answer on a question
        representation.
        """
        return self.convert_tokens_to_ids(self.question_token)

    # 获取是否需要小写处理
    @property
    def do_lower_case(self): 
        return self.basic_tokenizer.do_lower_case

    # 获取词汇表的大小
    @property
    def vocab_size(self): 
        return len(self.vocab)

    # 获取词汇表
    def get_vocab(self): 
        return dict(self.vocab, **self.added_tokens_encoder)

    # 分词方法
    def _tokenize(self, text):
        split_tokens = []
        # 如果需要基本分词
        if self.do_basic_tokenize:
            # 对基本分词器进行分词
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # 如果标记属于never_split集合
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        # 否则对文本进行Wordpiece分词
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens
    def _convert_token_to_id(self, token):
        """Converts a token (str) into an id using the vocabulary.
        
        Args:
            token (str): The token to convert into an id.
        
        Returns:
            int: The id corresponding to the token, or the id of the unknown token if the token is not found in the vocabulary.
        """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) into a token (str) using the vocabulary.
        
        Args:
            index (int): The index to convert into a token.
        
        Returns:
            str: The token corresponding to the index, or the unknown token if the index is not found in the mapping.
        """
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) into a single string.
        
        Args:
            tokens (List[str]): A list of tokens to concatenate.
        
        Returns:
            str: The concatenated string of tokens with '##' removed and leading/trailing whitespace stripped.
        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a pair of sequences for question answering tasks by concatenating and adding special
        tokens. A Splinter sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences for question answering: `[CLS] question_tokens [QUESTION] . [SEP] context_tokens [SEP]`

        Args:
            token_ids_0 (List[int]):
                The token IDs representing the first sequence (question or context depending on padding side).
            token_ids_1 (List[int], optional):
                The token IDs representing the second sequence (context or question depending on padding side).
        
        Returns:
            List[int]: List of input IDs with the appropriate special tokens.
        """
        if token_ids_1 is None:
            # Return single-sequence inputs
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        question_suffix = [self.question_token_id] + [self.convert_tokens_to_ids(".")]
        if self.padding_side == "right":
            # Input is question-then-context
            return cls + token_ids_0 + question_suffix + sep + token_ids_1 + sep
        else:
            # Input is context-then-question
            return cls + token_ids_0 + sep + token_ids_1 + question_suffix + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves the special tokens mask from the provided sequences of token IDs.
        
        Args:
            token_ids_0 (List[int]): The token IDs of the first sequence.
            token_ids_1 (List[int], optional): The token IDs of the second sequence.
            already_has_special_tokens (bool): Whether the sequences already contain special tokens.
        
        Returns:
            List[int]: A mask indicating the special tokens in the input sequences.
        """
    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
                                already_has_special_tokens: bool = False) -> List[int]:
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
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed. [What are token type
        IDs?](../glossary#token-type-ids)

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The token type ids.
        """
        sep = [self.sep_token_id]  # Define separator token
        cls = [self.cls_token_id]  # Define class token
        question_suffix = [self.question_token_id] + [self.convert_tokens_to_ids(".")]  # Define question suffix
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]  # Return a list of zeros with the length of cls, token_ids_0, and sep combined

        if self.padding_side == "right":
            # Input is question-then-context
            return len(cls + token_ids_0 + question_suffix + sep) * [0] + len(token_ids_1 + sep) * [1]  # Return a list of zeros followed by a list of ones based on the input sequence
        else:
            # Input is context-then-question
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + question_suffix + sep) * [1]  # Return a list of zeros followed by a list of ones based on the input sequence
    # 定义保存词汇表的方法，接受一个保存目录和可选的文件名前缀参数，返回元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引
        index = 0
        # 如果保存目录已经存在
        if os.path.isdir(save_directory):
            # 组合词汇表文件的路径，如果有前缀则加上，否则直接使用文件名
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 如果保存目录不存在，直接使用给定的路径作为文件名
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 以写入模式打开词汇表文件，指定编码为 utf-8
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表，按照索引排序
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 如果当前索引和预期的索引不一致
                if index != token_index:
                    # 发出警告，索引不连续
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    # 更新索引
                    index = token_index
                # 将词汇写入文件
                writer.write(token + "\n")
                # 更新索引
                index += 1
        # 返回保存的词汇表文件路径
        return (vocab_file,)
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
    """
    # 定义 BasicTokenizer 类，用于运行基本的分词（标点符号分割、小写转换等）。

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None):
        # 构造函数，初始化 BasicTokenizer 实例
        if never_split is None:
            # 如果 never_split 为 None，则将其设为空列表
            never_split = []
        # 将 do_lower_case 参数赋给实例变量 do_lower_case
        self.do_lower_case = do_lower_case
        # 将 never_split 转换成集合并赋给实例变量 never_split
        self.never_split = set(never_split)
        # 将 tokenize_chinese_chars 参数赋给实例变量 tokenize_chinese_chars
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 将 strip_accents 参数赋给实例变量 strip_accents
        self.strip_accents = strip_accents
    # 对文本进行基本的分词处理，仅在“空白字符”上进行分割，用于子词级别的分词，参见 WordPieceTokenizer。
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.

        Args:
            **never_split**: (*optional*) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 如果提供了 never_split 参数，则将其与 self.never_split 集合取并集
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，例如去除无用的字符
        text = self._clean_text(text)

        # 这段代码是在2018年11月1日添加的，用于多语言和中文模型。
        # 现在也应用于英语模型，但这并不重要，因为英语模型没有在任何中文数据上训练，
        # 通常也没有任何中文数据（英文维基百科中有一些中文单词，因此词汇表中有中文字符）。
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 对原始文本进行空格分割
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        # 遍历原始文本中的每个 token
        for token in orig_tokens:
            # 如果 token 不在 never_split 中，则考虑进行处理
            if token not in never_split:
                # 如果 do_lower_case 为真，则将 token 转换为小写
                if self.do_lower_case:
                    token = token.lower()
                    # 如果 strip_accents 不是 False，则移除 token 中的重音符号
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果 strip_accents 为真，则移除 token 中的重音符号
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 将处理后的 token 拆分，并加入到 split_tokens 中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将拆分后的 token 再次以空格分割，并返回
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    # 移除文本中的重音符号
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的每个字符
        for char in text:
            cat = unicodedata.category(char)
            # 如果字符的分类是 Mn（Mark, Nonspacing），则跳过该字符
            if cat == "Mn":
                continue
            output.append(char)
        # 将处理后的字符列表转换为字符串并返回
        return "".join(output)

    # 在文本中的标点符号上进行分割
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果 never_split 不为空且 text 在 never_split 中，则返回 text 本身
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果 char 是标点符号，则单独作为一个 token
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果 start_new_word 为真，则创建一个新的 token 列表
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        # 将拆分后的 token 列表转换为字符串列表并返回
        return ["".join(x) for x in output]
    # 将中文字符周围添加空格
    def _tokenize_chinese_chars(self, text):
        # 初始化输出列表
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 编码
            cp = ord(char)
            # 如果是中文字符，添加空格，并将字符添加到输出列表中
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        # 将输出列表中的字符连接成字符串并返回
        return "".join(output)
    
    # 检查是否为中文字符的 Unicode 编码
    def _is_chinese_char(self, cp):
        # 检查 Unicode 编码是否在 CJK Unicode 块中，返回布尔值
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
        # 如果不在 CJK Unicode 块中，返回 False
        return False
    
    # 清理文本中的无效字符和空白字符
    def _clean_text(self, text):
        # 初始化输出列表
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 编码
            cp = ord(char)
            # 如果字符为 0 或 0xFFFD，或者是控制字符，跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果是空白字符，用空格替换
            if _is_whitespace(char):
                output.append(" ")
            # 否则将字符添加到输出列表中
            else:
                output.append(char)
        # 将输出列表中的字符连接成字符串并返回
        return "".join(output)
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""  # WordPiece 分词器类的定义，用于运行 WordPiece 分词算法。

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        """
        初始化 WordPiece 分词器对象。

        Args:
          vocab: 词汇表，包含了所有可能的词和子词。
          unk_token: 未知词标记，用于表示词汇表中不存在的词。
          max_input_chars_per_word: 单个词的最大字符数，默认为 100。
        """
        self.vocab = vocab  # 保存词汇表
        self.unk_token = unk_token  # 保存未知词标记
        self.max_input_chars_per_word = max_input_chars_per_word  # 保存单个词的最大字符数

    def tokenize(self, text):
        """
        将文本分词为 WordPiece 形式的子词列表。该方法使用贪婪的最长匹配算法根据给定的词汇表进行分词。

        例如，输入为 "unaffable"，输出为 `["un", "##aff", "##able"]`。

        Args:
          text: 单个词或以空格分隔的多个词。应该已经通过 BasicTokenizer 进行了处理。

        Returns:
          WordPiece 形式的子词列表。
        """

        output_tokens = []  # 初始化输出的子词列表
        for token in whitespace_tokenize(text):  # 遍历通过空白字符分隔的文本词汇
            chars = list(token)  # 将词汇转换为字符列表
            if len(chars) > self.max_input_chars_per_word:  # 如果词汇的字符数超过了设定的最大字符数
                output_tokens.append(self.unk_token)  # 将未知词标记添加到输出的子词列表中
                continue

            is_bad = False  # 初始化标志位，用于表示是否为不良词汇
            start = 0  # 初始化起始索引
            sub_tokens = []  # 初始化子词列表
            while start < len(chars):  # 当起始索引小于词汇长度时执行循环
                end = len(chars)  # 初始化结束索引为词汇长度
                cur_substr = None  # 初始化当前子字符串
                while start < end:  # 当起始索引小于结束索引时执行循环
                    substr = "".join(chars[start:end])  # 获取从起始到结束索引的子字符串
                    if start > 0:  # 如果起始索引大于 0
                        substr = "##" + substr  # 在子字符串前添加 "##" 表示它是一个子词的一部分
                    if substr in self.vocab:  # 如果子字符串在词汇表中
                        cur_substr = substr  # 将当前子字符串更新为符合条件的子字符串
                        break
                    end -= 1  # 结束索引减 1，继续尝试更短的子字符串
                if cur_substr is None:  # 如果当前子字符串为 None
                    is_bad = True  # 将不良词汇标志位设置为 True
                    break
                sub_tokens.append(cur_substr)  # 将符合条件的子字符串添加到子词列表中
                start = end  # 更新起始索引为结束索引，准备处理下一个子字符串

            if is_bad:  # 如果是不良词汇
                output_tokens.append(self.unk_token)  # 将未知词标记添加到输出的子词列表中
            else:
                output_tokens.extend(sub_tokens)  # 将符合条件的子词列表扩展到输出的子词列表中
        return output_tokens  # 返回 WordPiece 形式的子词列表
```