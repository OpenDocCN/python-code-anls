# `.\models\bert_japanese\tokenization_bert_japanese.py`

```py
# coding=utf-8
# 版权所有 2018 年 Google AI 语言团队和 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何形式的担保或条件，包括但不限于有关适销性或特定用途的保证。
# 请查阅许可证以了解具体的法律规定和限制。
"""Tokenization classes."""


import collections
import copy
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# 从 tokenization_utils 模块导入必要的函数和类
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
# 从 utils 模块导入 logging 函数
from ...utils import is_sentencepiece_available, is_sudachi_projection_available, logging


# 如果 sentencepiece 可用，则导入 sentencepiece 库
if is_sentencepiece_available():
    import sentencepiece as spm
else:
    spm = None

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "spm_file": "spiece.model"}

# 定义 subword 分隔符
SPIECE_UNDERLINE = "▁"

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "cl-tohoku/bert-base-japanese": "https://huggingface.co/cl-tohoku/bert-base-japanese/resolve/main/vocab.txt",
        "cl-tohoku/bert-base-japanese-whole-word-masking": (
            "https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/vocab.txt"
        ),
        "cl-tohoku/bert-base-japanese-char": (
            "https://huggingface.co/cl-tohoku/bert-base-japanese-char/resolve/main/vocab.txt"
        ),
        "cl-tohoku/bert-base-japanese-char-whole-word-masking": (
            "https://huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking/resolve/main/vocab.txt"
        ),
    }
}

# 定义预训练模型的位置嵌入尺寸映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "cl-tohoku/bert-base-japanese": 512,
    "cl-tohoku/bert-base-japanese-whole-word-masking": 512,
    "cl-tohoku/bert-base-japanese-char": 512,
    "cl-tohoku/bert-base-japanese-char-whole-word-masking": 512,
}

# 定义预训练模型的初始化配置映射
PRETRAINED_INIT_CONFIGURATION = {
    "cl-tohoku/bert-base-japanese": {
        "do_lower_case": False,
        "word_tokenizer_type": "mecab",
        "subword_tokenizer_type": "wordpiece",
    },
    "cl-tohoku/bert-base-japanese-whole-word-masking": {
        "do_lower_case": False,
        "word_tokenizer_type": "mecab",
        "subword_tokenizer_type": "wordpiece",
    },
    "cl-tohoku/bert-base-japanese-char": {
        "do_lower_case": False,
        "word_tokenizer_type": "mecab",
        "subword_tokenizer_type": "character",
    },
    "cl-tohoku/bert-base-japanese-char-whole-word-masking": {
        "do_lower_case": False,
        "word_tokenizer_type": "mecab",
        "subword_tokenizer_type": "character",
    },
}


# 从 transformers.models.bert.tokenization_bert.load_vocab 复制而来
`
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()  # 创建一个有序字典用于存储词汇
    with open(vocab_file, "r", encoding="utf-8") as reader:  # 打开词汇文件，指定编码为 UTF-8
        tokens = reader.readlines()  # 读取文件中的所有行
    for index, token in enumerate(tokens):  # 遍历所有读取的行，索引从 0 开始
        token = token.rstrip("\n")  # 移除行尾的换行符
        vocab[token] = index  # 将词汇添加到字典中，键为词汇，值为其索引
    return vocab  # 返回加载的词汇字典

# Copied from transformers.models.bert.tokenization_bert.whitespace_tokenize
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()  # 移除文本开头和结尾的空白字符
    if not text:  # 如果文本为空，则返回空列表
        return []
    tokens = text.split()  # 将文本按空格分割成词语列表
    return tokens  # 返回分割后的词语列表

class BertJapaneseTokenizer(PreTrainedTokenizer):
    r"""
    Construct a BERT tokenizer for Japanese text.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer
    to: this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to a one-wordpiece-per-line vocabulary file.
        spm_file (`str`, *optional*):
            Path to [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm or .model
            extension) that contains the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether to lower case the input. Only has an effect when do_basic_tokenize=True.
        do_word_tokenize (`bool`, *optional*, defaults to `True`):
            Whether to do word tokenization.
        do_subword_tokenize (`bool`, *optional*, defaults to `True`):
            Whether to do subword tokenization.
        word_tokenizer_type (`str`, *optional*, defaults to `"basic"`):
            Type of word tokenizer. Choose from ["basic", "mecab", "sudachi", "jumanpp"].
        subword_tokenizer_type (`str`, *optional*, defaults to `"wordpiece"`):
            Type of subword tokenizer. Choose from ["wordpiece", "character", "sentencepiece",].
        mecab_kwargs (`dict`, *optional*):
            Dictionary passed to the `MecabTokenizer` constructor.
        sudachi_kwargs (`dict`, *optional*):
            Dictionary passed to the `SudachiTokenizer` constructor.
        jumanpp_kwargs (`dict`, *optional*):
            Dictionary passed to the `JumanppTokenizer` constructor.
    """

    vocab_files_names = VOCAB_FILES_NAMES  # 设置词汇文件名
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 设置预训练词汇文件的映射关系
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION  # 设置预训练模型的初始化配置
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 设置模型输入的最大大小
    # 初始化函数，用于初始化对象
    def __init__(
        self,
        vocab_file,
        spm_file=None,
        do_lower_case=False,
        do_word_tokenize=True,
        do_subword_tokenize=True,
        word_tokenizer_type="basic",
        subword_tokenizer_type="wordpiece",
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        mecab_kwargs=None,
        sudachi_kwargs=None,
        jumanpp_kwargs=None,
        **kwargs,
    ):
    
    @property
    # 返回属性 do_lower_case 的值
    def do_lower_case(self):
        return self.lower_case

    # 将对象序列化为字典形式，用于 pickle 操作
    def __getstate__(self):
        state = dict(self.__dict__)
        # 如果使用的是 mecab、sudachi 或 jumanpp 分词器，则删除 word_tokenizer 属性，因为它们不支持序列化
        if self.word_tokenizer_type in ["mecab", "sudachi", "jumanpp"]:
            del state["word_tokenizer"]
        return state

    # 从字典状态中恢复对象的状态
    def __setstate__(self, state):
        self.__dict__ = state
        # 根据 word_tokenizer_type 属性重新初始化 word_tokenizer
        if self.word_tokenizer_type == "mecab":
            self.word_tokenizer = MecabTokenizer(
                do_lower_case=self.do_lower_case, never_split=self.never_split, **(self.mecab_kwargs or {})
            )
        elif self.word_tokenizer_type == "sudachi":
            self.word_tokenizer = SudachiTokenizer(
                do_lower_case=self.do_lower_case, never_split=self.never_split, **(self.sudachi_kwargs or {})
            )
        elif self.word_tokenizer_type == "jumanpp":
            self.word_tokenizer = JumanppTokenizer(
                do_lower_case=self.do_lower_case, never_split=self.never_split, **(self.jumanpp_kwargs or {})
            )

    # 对文本进行分词处理
    def _tokenize(self, text):
        if self.do_word_tokenize:
            # 使用 word_tokenizer 对文本进行分词，如果需要，会忽略特殊标记的切分
            tokens = self.word_tokenizer.tokenize(text, never_split=self.all_special_tokens)
        else:
            tokens = [text]

        if self.do_subword_tokenize:
            # 对词级标记进行子词处理
            split_tokens = [sub_token for token in tokens for sub_token in self.subword_tokenizer.tokenize(token)]
        else:
            split_tokens = tokens

        return split_tokens

    @property
    # 返回词汇表的大小
    def vocab_size(self):
        if self.subword_tokenizer_type == "sentencepiece":
            return len(self.subword_tokenizer.sp_model)
        return len(self.vocab)

    # 获取词汇表
    def get_vocab(self):
        if self.subword_tokenizer_type == "sentencepiece":
            # 如果使用 sentencepiece 分词器，返回从索引到词汇的映射
            vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
            vocab.update(self.added_tokens_encoder)
            return vocab
        # 否则，返回词汇表和添加的特殊标记的编码映射
        return dict(self.vocab, **self.added_tokens_encoder)

    # 将 token 转换为对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if self.subword_tokenizer_type == "sentencepiece":
            # 使用 sentencepiece 分词器将 token 转换为 id
            return self.subword_tokenizer.sp_model.PieceToId(token)
        # 否则，使用 vocab 将 token 转换为 id，如果未找到则使用 unk_token
        return self.vocab.get(token, self.vocab.get(self.unk_token))
    # 将索引转换为对应的词汇（字符串），使用当前的词汇表进行转换
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if self.subword_tokenizer_type == "sentencepiece":
            # 如果使用 sentencepiece 分词器，则通过索引获取对应的词片段
            return self.subword_tokenizer.sp_model.IdToPiece(index)
        # 否则，使用预先定义的词汇表将索引转换为对应的标记（token），如果索引不存在，则使用未知标记（unk_token）
        return self.ids_to_tokens.get(index, self.unk_token)

    # 将一系列标记（tokens）转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        if self.subword_tokenizer_type == "sentencepiece":
            # 如果使用 sentencepiece 分词器，则使用其解码功能将标记序列解码为单个字符串
            return self.subword_tokenizer.sp_model.decode(tokens)
        # 否则，将标记序列连接成一个字符串，去除连字符（" ##"），并去除两端的空格
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # 从输入的标记 ID 列表构建带有特殊标记的模型输入
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
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            # 如果只有一个输入序列，将其前后加上特殊标记 [CLS] 和 [SEP]
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 如果有两个输入序列，加上 [CLS]，连接第一个序列和第二个序列，最后加上两个 [SEP]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # 获取特殊标记的掩码，标识哪些位置是特殊标记
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: List of integers indicating special tokens (1 for special token, 0 for regular token).
        """
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

        if already_has_special_tokens:
            # If tokens already have special tokens, delegate to parent class method
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            # For sequence pair: add special tokens around both sequences
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # For single sequence: add special tokens around the sequence
        return [1] + ([0] * len(token_ids_0)) + [1]

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
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
        sep = [self.sep_token_id]  # Separator token ID
        cls = [self.cls_token_id]  # Classification token ID
        if token_ids_1 is None:
            # If no second sequence, return token type IDs for the first sequence only
            return len(cls + token_ids_0 + sep) * [0]
        # Return token type IDs for both sequences concatenated with special tokens
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    # 保存词汇表到指定目录下的文件
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if os.path.isdir(save_directory):
            # 如果使用的子词分割器类型是 sentencepiece
            if self.subword_tokenizer_type == "sentencepiece":
                # 构建保存 sentencepiece 词汇文件的完整路径
                vocab_file = os.path.join(
                    save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["spm_file"]
                )
            else:
                # 构建保存普通词汇文件的完整路径
                vocab_file = os.path.join(
                    save_directory,
                    (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"],
                )
        else:
            # 如果保存目录不存在，则直接将文件名作为保存路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory

        # 根据子词分割器类型写入词汇表内容到文件
        if self.subword_tokenizer_type == "sentencepiece":
            # 使用二进制方式写入 sentencepiece 模型内容到文件
            with open(vocab_file, "wb") as writer:
                content_spiece_model = self.subword_tokenizer.sp_model.serialized_model_proto()
                writer.write(content_spiece_model)
        else:
            # 使用 UTF-8 编码以文本方式写入普通词汇表内容到文件
            with open(vocab_file, "w", encoding="utf-8") as writer:
                index = 0
                # 按词汇表索引排序，将词汇和索引写入文件
                for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                    # 检查词汇表索引是否连续，记录最后的索引
                    if index != token_index:
                        logger.warning(
                            f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                            " Please check that the vocabulary is not corrupted!"
                        )
                        index = token_index
                    # 写入词汇及换行符
                    writer.write(token + "\n")
                    index += 1
        # 返回保存的文件路径
        return (vocab_file,)
# 定义了一个名为 MecabTokenizer 的类，用于基本的 MeCab 形态分析器的分词处理
class MecabTokenizer:
    """Runs basic tokenization with MeCab morphological parser."""

    # 初始化方法，设置了几个参数来配置分词器的行为
    def __init__(
        self,
        do_lower_case=False,  # 控制是否将所有字符转换为小写
        never_split=None,  # 永远不要分割的词汇列表，如果没有指定，默认为空
        normalize_text=True,  # 控制是否对文本进行规范化处理
        mecab_dic: Optional[str] = "ipadic",  # MeCab 使用的字典，默认为 "ipadic"
        mecab_option: Optional[str] = None,  # MeCab 的其他选项，可选，默认为 None
    ):

    # 方法用于对文本进行分词处理
    def tokenize(self, text, never_split=None, **kwargs):
        """Tokenizes a piece of text."""
        # 如果需要对文本进行规范化处理，则使用 unicodedata 进行 NFKC 规范化
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        # 确定不会被分割的词汇列表，考虑实例化时设置的 never_split 和传入参数中的 never_split
        never_split = self.never_split + (never_split if never_split is not None else [])
        # 初始化空的 tokens 列表用于存储分词结果
        tokens = []

        # 使用 MeCab 对文本进行分词处理
        for word in self.mecab(text):
            # 获取当前词的表层形式（token）
            token = word.surface

            # 如果设置了 do_lower_case 为 True，并且当前 token 不在 never_split 中，则将其转换为小写
            if self.do_lower_case and token not in never_split:
                token = token.lower()

            # 将处理后的 token 添加到 tokens 列表中
            tokens.append(token)

        # 返回最终的分词结果 tokens
        return tokens


# 定义了一个名为 SudachiTokenizer 的类，用于基本的 Sudachi 形态分析器的分词处理
class SudachiTokenizer:
    """Runs basic tokenization with Sudachi morphological parser."""

    # 初始化方法，设置了几个参数来配置分词器的行为
    def __init__(
        self,
        do_lower_case=False,  # 控制是否将所有字符转换为小写
        never_split=None,  # 永远不要分割的词汇列表，如果没有指定，默认为空
        normalize_text=True,  # 控制是否对文本进行规范化处理
        trim_whitespace=False,  # 控制是否修剪文本中的空白字符
        sudachi_split_mode="A",  # Sudachi 的分割模式，默认为 "A"
        sudachi_config_path=None,  # Sudachi 的配置文件路径，可选，默认为 None
        sudachi_resource_dir=None,  # Sudachi 的资源目录路径，可选，默认为 None
        sudachi_dict_type="core",  # Sudachi 使用的词典类型，默认为 "core"
        sudachi_projection=None,  # Sudachi 的 projection 参数，可选，默认为 None
    ):
    ):
        """
        Constructs a SudachiTokenizer.

        Args:
            **do_lower_case**: (*optional*) boolean (default True)
                Whether to lowercase the input.
            **never_split**: (*optional*) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of tokens not to split.
            **normalize_text**: (*optional*) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
            **trim_whitespace**: (*optional*) boolean (default False)
                Whether to trim all whitespace, tab, newline from tokens.
            **sudachi_split_mode**: (*optional*) string
                Split mode of sudachi, choose from `["A", "B", "C"]`.
            **sudachi_config_path**: (*optional*) string
                Path to Sudachi configuration file.
            **sudachi_resource_dir**: (*optional*) string
                Directory containing Sudachi resources.
            **sudachi_dict_type**: (*optional*) string
                Dictionary type of Sudachi, choose from `["small", "core", "full"]`.
            **sudachi_projection**: (*optional*) string
                Word projection mode of Sudachi, choose from `["surface", "normalized", "reading", "dictionary", "dictionary_and_surface", "normalized_and_surface", "normalized_nouns"]`.
        """

        self.do_lower_case = do_lower_case  # 设置是否将输入转换为小写
        self.never_split = never_split if never_split is not None else []  # 设置不需要分割的标记列表
        self.normalize_text = normalize_text  # 设置是否在分词前对文本进行Unicode标准化
        self.trim_whitespace = trim_whitespace  # 设置是否去除所有标记中的空白、制表符和换行符

        try:
            from sudachipy import dictionary, tokenizer  # 导入Sudachi相关库
        except ImportError:
            raise ImportError(
                "You need to install sudachipy to use SudachiTokenizer. "
                "See https://github.com/WorksApplications/SudachiPy for installation."
            )

        if sudachi_split_mode == "A":
            self.split_mode = tokenizer.Tokenizer.SplitMode.A  # 设置Sudachi的分割模式为A
        elif sudachi_split_mode == "B":
            self.split_mode = tokenizer.Tokenizer.SplitMode.B  # 设置Sudachi的分割模式为B
        elif sudachi_split_mode == "C":
            self.split_mode = tokenizer.Tokenizer.SplitMode.C  # 设置Sudachi的分割模式为C
        else:
            raise ValueError("Invalid sudachi_split_mode is specified.")  # 报错，如果指定的Sudachi分割模式无效

        self.projection = sudachi_projection  # 设置Sudachi的词汇投影模式

        # 创建Sudachi字典对象
        sudachi_dictionary = dictionary.Dictionary(
            config_path=sudachi_config_path, resource_dir=sudachi_resource_dir, dict=sudachi_dict_type
        )
        
        # 检查Sudachi的投影模式是否可用，并设置相应的Sudachi对象
        if is_sudachi_projection_available():
            self.sudachi = sudachi_dictionary.create(self.split_mode, projection=self.projection)
        elif self.projection is not None:
            raise ImportError("You need to install sudachipy>=0.6.8 to specify `projection` field in sudachi_kwargs.")
        else:
            self.sudachi = sudachi_dictionary.create(self.split_mode)
    # Tokenizes a piece of text based on the specified tokenizer settings.
    def tokenize(self, text, never_split=None, **kwargs):
        """Tokenizes a piece of text."""
        # Normalize text if enabled to ensure consistent representation
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        # Combine the default never_split tokens with any user-provided ones
        never_split = self.never_split + (never_split if never_split is not None else [])
        # Initialize an empty list to store tokens
        tokens = []

        # Iterate over tokens returned by the sudachi tokenizer
        for word in self.sudachi.tokenize(text):
            # Retrieve the surface form (actual text) of the token
            token = word.surface()

            # Convert token to lowercase if specified and not in the never_split list
            if self.do_lower_case and token not in never_split:
                token = token.lower()

            # Trim whitespace from tokens if specified
            if self.trim_whitespace:
                # Skip tokens that are completely whitespace
                if token.strip() == "":
                    continue
                else:
                    # Remove leading and trailing whitespace
                    token = token.strip()

            # Add processed token to the list of tokens
            tokens.append(token)

        # Return the list of tokens
        return tokens
class JumanppTokenizer:
    """Runs basic tokenization with jumanpp morphological parser."""

    def __init__(
        self,
        do_lower_case=False,
        never_split=None,
        normalize_text=True,
        trim_whitespace=False,
    ):
        """
        Constructs a JumanppTokenizer.

        Args:
            **do_lower_case**: (*optional*) boolean (default True)
                Whether to lowercase the input.
            **never_split**: (*optional*) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of tokens not to split.
            **normalize_text**: (*optional*) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
            **trim_whitespace**: (*optional*) boolean (default False)
                Whether to trim all whitespace, tab, newline from tokens.
        """

        self.do_lower_case = do_lower_case  # 是否将输入转换为小写，默认为 False
        self.never_split = never_split if never_split is not None else []  # 不希望被分割的特定 token 列表，默认为空列表
        self.normalize_text = normalize_text  # 是否对文本进行 Unicode 规范化，默认为 True
        self.trim_whitespace = trim_whitespace  # 是否去除所有空白符（空格、制表符、换行符），默认为 False

        try:
            import rhoknp
        except ImportError:
            raise ImportError(
                "You need to install rhoknp to use JumanppTokenizer. "
                "See https://github.com/ku-nlp/rhoknp for installation."
            )

        self.juman = rhoknp.Jumanpp()  # 初始化 Juman++ 分词器对象

    def tokenize(self, text, never_split=None, **kwargs):
        """Tokenizes a piece of text."""
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)  # 如果需要，对文本进行 Unicode 规范化

        text = text.strip()  # 去除文本两端的空白符

        never_split = self.never_split + (never_split if never_split is not None else [])  # 合并当前实例的和传入的不希望分割的 token 列表
        tokens = []

        for mrph in self.juman.apply_to_sentence(text).morphemes:
            token = mrph.text  # 获取分词结果中的每个词素文本

            if self.do_lower_case and token not in never_split:
                token = token.lower()  # 如果需要，并且该 token 不在不分割列表中，则将其转换为小写

            if self.trim_whitespace:
                if token.strip() == "":
                    continue  # 如果需要，并且 token 是空字符串，则跳过
                else:
                    token = token.strip()  # 去除 token 前后的空白符

            tokens.append(token)  # 将处理后的 token 添加到 tokens 列表中

        return tokens


class CharacterTokenizer:
    """Runs Character tokenization."""

    def __init__(self, vocab, unk_token, normalize_text=True):
        """
        Constructs a CharacterTokenizer.

        Args:
            **vocab**:
                Vocabulary object.
            **unk_token**: str
                A special symbol for out-of-vocabulary token.
            **normalize_text**: (`optional`) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
        """
        self.vocab = vocab  # 初始化词汇表对象
        self.unk_token = unk_token  # 初始化未知 token 的特殊符号
        self.normalize_text = normalize_text  # 是否对文本进行 Unicode 规范化，默认为 True
    def tokenize(self, text):
        """
        将文本分词为字符列表。

        例如，`input = "apple"` 将返回 `["a", "p", "p", "l", "e"]`。

        Args:
            text: 单个标记或以空格分隔的标记。
                  应该已经通过 *BasicTokenizer* 处理过。

        Returns:
            包含字符的列表。
        """
        # 如果需要规范化文本，使用 Unicode 规范化函数将其转换为兼容 NFC 表示
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        # 初始化空的输出 tokens 列表
        output_tokens = []
        # 遍历输入文本中的每个字符
        for char in text:
            # 如果字符不在词汇表中，则添加未知标记到输出 tokens 列表
            if char not in self.vocab:
                output_tokens.append(self.unk_token)
                continue

            # 否则，将字符添加到输出 tokens 列表中
            output_tokens.append(char)

        # 返回最终的字符列表
        return output_tokens
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
        do_lower_case=True,                    # 初始化方法，设置是否小写化输入，默认为True
        never_split=None,                      # 设置不进行分割的特定token集合，默认为None
        tokenize_chinese_chars=True,           # 设置是否对中文字符进行分词，默认为True
        strip_accents=None,                    # 设置是否去除所有重音符号，默认根据小写化选项决定
        do_split_on_punc=True,                 # 设置是否基于标点符号进行基本分词，默认为True
    ):
        if never_split is None:
            never_split = []                   # 如果never_split参数为None，则设为一个空列表
        self.do_lower_case = do_lower_case     # 将输入小写化选项保存到实例变量中
        self.never_split = set(never_split)    # 将never_split参数转换为集合类型并保存到实例变量中
        self.tokenize_chinese_chars = tokenize_chinese_chars  # 保存是否分词中文字符的选项到实例变量中
        self.strip_accents = strip_accents     # 将去除重音符号的选项保存到实例变量中
        self.do_split_on_punc = do_split_on_punc  # 将基于标点符号进行分词的选项保存到实例变量中
    # 对文本进行基本的分词处理。如需子词分词，请参考 WordPieceTokenizer。
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 如果指定了 never_split 列表，则将其与类属性 never_split 的集合进行合并，以确保不拆分这些特定的 token
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本中的特殊字符和空白
        text = self._clean_text(text)

        # 以下代码段于2018年11月1日添加，用于多语言和中文模型。
        # 现在也适用于英语模型，但由于英语模型没有在任何中文数据上进行训练，
        # 并且通常不包含任何中文数据（英语维基百科中确实包含一些中文词汇），
        # 因此这并不重要。
        if self.tokenize_chinese_chars:
            # 如果需要对中文字符进行特殊处理，则调用 _tokenize_chinese_chars 方法
            text = self._tokenize_chinese_chars(text)
        # 对文本进行 Unicode 规范化，确保统一字符的表示形式
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 将规范化后的文本按空白字符分词，得到原始 token 列表
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        # 初始化分词结果列表
        split_tokens = []
        # 遍历每个原始 token
        for token in orig_tokens:
            # 如果 token 不在 never_split 集合中，则考虑是否进行小写处理和重音符号处理
            if token not in never_split:
                if self.do_lower_case:
                    # 如果需要小写处理，则将 token 转换为小写
                    token = token.lower()
                    if self.strip_accents is not False:
                        # 如果需要去除重音符号，则调用 _run_strip_accents 方法
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    # 否则，如果仅需要去除重音符号，则调用 _run_strip_accents 方法
                    token = self._run_strip_accents(token)
            # 将处理后的 token 经过标点符号分割后加入分词结果列表
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将处理后的分词结果再次按空白字符分割，得到最终的输出 tokens
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    # 从文本中去除重音符号
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 对文本进行 Unicode 规范化，转换为标准形式
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 分类
            cat = unicodedata.category(char)
            # 如果字符的分类为 Mn（Nonspacing_Mark），表示是重音符号，跳过处理
            if cat == "Mn":
                continue
            # 否则将字符添加到输出列表中
            output.append(char)
        # 将输出列表中的字符拼接成字符串并返回
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要根据标点符号分割文本，或者文本在不分割列表中，则直接返回包含整个文本的列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号，则创建一个新的列表作为输出的一部分，并将标志设置为开始新单词
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，且应该继续当前单词，则将字符添加到当前输出的最后一个列表中
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        # 将分割后的列表中的各个子列表连接成字符串，并返回列表
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符是中日韩字符，则在字符前后添加空格，并加入到输出列表中
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                # 如果不是中日韩字符，则直接将字符加入到输出列表中
                output.append(char)
        # 将输出列表连接成一个字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查给定的 Unicode 码点是否属于中日韩字符的范围
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
        for char in text:
            cp = ord(char)
            # 如果字符是无效字符或控制字符，则跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果是空白字符，则替换为单个空格；否则保留字符
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将清理后的字符列表连接成一个字符串并返回
        return "".join(output)
# Copied from transformers.models.bert.tokenization_bert.WordpieceTokenizer
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化 WordpieceTokenizer 类的实例，设置词汇表、未知标记和每个单词的最大字符数
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
        # 初始化输出的 tokens 列表
        output_tokens = []
        # 遍历通过 whitespace_tokenize 函数分词后的文本
        for token in whitespace_tokenize(text):
            # 将每个 token 拆分为字符列表
            chars = list(token)
            # 如果 token 的字符数超过设定的最大输入字符数，则用未知标记替换
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            # 使用贪婪算法将 token 分割为子 token
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    # 检查子字符串是否在词汇表中
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            # 如果存在无法识别的子 token，则用未知标记替换
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class SentencepieceTokenizer(object):
    """
    Runs sentencepiece tokenization. Based on transformers.models.albert.tokenization_albert.AlbertTokenizer.
    """

    def __init__(
        self,
        vocab,
        unk_token,
        do_lower_case=False,
        remove_space=True,
        keep_accents=True,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # 初始化 SentencepieceTokenizer 类的实例，设置词汇表、未知标记以及其他可选参数
        self.vocab = vocab
        self.unk_token = unk_token
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents

        # 如果没有传入 SentencePiece 参数，则设为默认空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # 创建 SentencePieceProcessor 对象并加载词汇表
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab)
    # 对文本进行预处理，根据设置去除空格或保留原始格式，并替换特定的双引号格式
    def preprocess_text(self, inputs):
        if self.remove_space:
            # 如果需要去除空格，则去除首尾空格并将多余空格替换为单个空格
            outputs = " ".join(inputs.strip().split())
        else:
            # 否则保留原始输入文本
            outputs = inputs
        # 替换特定的双引号格式为标准双引号
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            # 如果不保留重音符号，则规范化 Unicode 字符串，去除组合字符
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            # 如果需要将文本转换为小写，则进行小写转换
            outputs = outputs.lower()

        return outputs

    # 使用 SentencePiece 对文本进行分词处理
    def tokenize(self, text):
        """
        Tokenizes text by sentencepiece. Based on [SentencePiece](https://github.com/google/sentencepiece).
        Tokenization needs the given vocabulary.

        Args:
            text: A string needs to be tokenized.

        Returns:
            A list of sentencepiece tokens.
        """
        # 对输入文本先进行预处理
        text = self.preprocess_text(text)
        # 使用 SentencePiece 模型对文本进行编码，并以字符串形式输出
        pieces = self.sp_model.encode(text, out_type=str)
        new_pieces = []
        for piece in pieces:
            # 处理特定形式的词片段，如以数字结尾且最后一个字符是逗号的情况
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                # 对词片段去除最后的逗号并处理成新的词片段列表
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                # 调整处理后的词片段的格式
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                # 将处理后的词片段添加到新的词片段列表中
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                # 将普通词片段直接添加到新的词片段列表中
                new_pieces.append(piece)

        return new_pieces
```