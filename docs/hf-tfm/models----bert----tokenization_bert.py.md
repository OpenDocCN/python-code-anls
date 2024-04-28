# `.\transformers\models\bert\tokenization_bert.py`

```
# 设置文件编码为 UTF-8
# 版权声明，指明版权持有人和许可方
# 版权许可协议，使用 Apache License 2.0
# 你可以在遵循许可协议的情况下使用此文件
# 获取许可协议的副本地址
#     http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律要求或书面同意，本软件按“原样”分发，没有任何明示或暗示的保证或条件
# 根据适用法律要求或书面同意，软件分发时不包含任何形式的担保或条件，明示或暗示
# 有关语言代码的标准化和规范化的辅助功能
"""Bert 的分词类。"""

# 导入所需模块
import collections
import os
import unicodedata
from typing import List, Optional, Tuple

# 导入分词工具类和辅助函数
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 词汇表文件名字典，用于指定所需的词汇表文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练词汇表文件映射，用于指定预训练模型所需的词汇表文件路径
PRETRAINED_VOCAB_FILES_MAP = {
    # 定义一个字典，包含了各种 BERT 模型的词汇表文件链接
    "vocab_file": {
        # BERT 模型 "bert-base-uncased" 的词汇表文件链接
        "bert-base-uncased": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
        # BERT 模型 "bert-large-uncased" 的词汇表文件链接
        "bert-large-uncased": "https://huggingface.co/bert-large-uncased/resolve/main/vocab.txt",
        # BERT 模型 "bert-base-cased" 的词汇表文件链接
        "bert-base-cased": "https://huggingface.co/bert-base-cased/resolve/main/vocab.txt",
        # BERT 模型 "bert-large-cased" 的词汇表文件链接
        "bert-large-cased": "https://huggingface.co/bert-large-cased/resolve/main/vocab.txt",
        # BERT 模型 "bert-base-multilingual-uncased" 的词汇表文件链接
        "bert-base-multilingual-uncased": (
            "https://huggingface.co/bert-base-multilingual-uncased/resolve/main/vocab.txt"
        ),
        # BERT 模型 "bert-base-multilingual-cased" 的词汇表文件链接
        "bert-base-multilingual-cased": "https://huggingface.co/bert-base-multilingual-cased/resolve/main/vocab.txt",
        # BERT 模型 "bert-base-chinese" 的词汇表文件链接
        "bert-base-chinese": "https://huggingface.co/bert-base-chinese/resolve/main/vocab.txt",
        # BERT 模型 "bert-base-german-cased" 的词汇表文件链接
        "bert-base-german-cased": "https://huggingface.co/bert-base-german-cased/resolve/main/vocab.txt",
        # BERT 模型 "bert-large-uncased-whole-word-masking" 的词汇表文件链接
        "bert-large-uncased-whole-word-masking": (
            "https://huggingface.co/bert-large-uncased-whole-word-masking/resolve/main/vocab.txt"
        ),
        # BERT 模型 "bert-large-cased-whole-word-masking" 的词汇表文件链接
        "bert-large-cased-whole-word-masking": (
            "https://huggingface.co/bert-large-cased-whole-word-masking/resolve/main/vocab.txt"
        ),
        # BERT 模型 "bert-large-uncased-whole-word-masking-finetuned-squad" 的词汇表文件链接
        "bert-large-uncased-whole-word-masking-finetuned-squad": (
            "https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
        ),
        # BERT 模型 "bert-large-cased-whole-word-masking-finetuned-squad" 的词汇表文件链接
        "bert-large-cased-whole-word-masking-finetuned-squad": (
            "https://huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
        ),
        # BERT 模型 "bert-base-cased-finetuned-mrpc" 的词汇表文件链接
        "bert-base-cased-finetuned-mrpc": (
            "https://huggingface.co/bert-base-cased-finetuned-mrpc/resolve/main/vocab.txt"
        ),
        # BERT 模型 "bert-base-german-dbmdz-cased" 的词汇表文件链接
        "bert-base-german-dbmdz-cased": "https://huggingface.co/bert-base-german-dbmdz-cased/resolve/main/vocab.txt",
        # BERT 模型 "bert-base-german-dbmdz-uncased" 的词汇表文件链接
        "bert-base-german-dbmdz-uncased": (
            "https://huggingface.co/bert-base-german-dbmdz-uncased/resolve/main/vocab.txt"
        ),
        # BERT 模型 "TurkuNLP/bert-base-finnish-cased-v1" 的词汇表文件链接
        "TurkuNLP/bert-base-finnish-cased-v1": (
            "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/vocab.txt"
        ),
        # BERT 模型 "TurkuNLP/bert-base-finnish-uncased-v1" 的词汇表文件链接
        "TurkuNLP/bert-base-finnish-uncased-v1": (
            "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/vocab.txt"
        ),
        # BERT 模型 "wietsedv/bert-base-dutch-cased" 的词汇表文件链接
        "wietsedv/bert-base-dutch-cased": (
            "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/vocab.txt"
        ),
    }
}

# 预训练模型的位置嵌入大小字典，将模型名称映射到其对应的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "bert-base-uncased": 512,
    "bert-large-uncased": 512,
    "bert-base-cased": 512,
    "bert-large-cased": 512,
    "bert-base-multilingual-uncased": 512,
    "bert-base-multilingual-cased": 512,
    "bert-base-chinese": 512,
    "bert-base-german-cased": 512,
    "bert-large-uncased-whole-word-masking": 512,
    "bert-large-cased-whole-word-masking": 512,
    "bert-large-uncased-whole-word-masking-finetuned-squad": 512,
    "bert-large-cased-whole-word-masking-finetuned-squad": 512,
    "bert-base-cased-finetuned-mrpc": 512,
    "bert-base-german-dbmdz-cased": 512,
    "bert-base-german-dbmdz-uncased": 512,
    "TurkuNLP/bert-base-finnish-cased-v1": 512,
    "TurkuNLP/bert-base-finnish-uncased-v1": 512,
    "wietsedv/bert-base-dutch-cased": 512,
}

# 预训练模型的初始化配置字典，将模型名称映射到其对应的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "bert-base-uncased": {"do_lower_case": True},
    "bert-large-uncased": {"do_lower_case": True},
    "bert-base-cased": {"do_lower_case": False},
    "bert-large-cased": {"do_lower_case": False},
    "bert-base-multilingual-uncased": {"do_lower_case": True},
    "bert-base-multilingual-cased": {"do_lower_case": False},
    "bert-base-chinese": {"do_lower_case": False},
    "bert-base-german-cased": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking": {"do_lower_case": False},
    "bert-large-uncased-whole-word-masking-finetuned-squad": {"do_lower_case": True},
    "bert-large-cased-whole-word-masking-finetuned-squad": {"do_lower_case": False},
    "bert-base-cased-finetuned-mrpc": {"do_lower_case": False},
    "bert-base-german-dbmdz-cased": {"do_lower_case": False},
    "bert-base-german-dbmdz-uncased": {"do_lower_case": True},
    "TurkuNLP/bert-base-finnish-cased-v1": {"do_lower_case": False},
    "TurkuNLP/bert-base-finnish-uncased-v1": {"do_lower_case": True},
    "wietsedv/bert-base-dutch-cased": {"do_lower_case": False},
}

# 加载词汇表文件到字典中的函数
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    # 创建一个有序字典对象，用于保存词汇表
    vocab = collections.OrderedDict()
    # 打开词汇表文件，按行读取其中的词汇并添加到字典中
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    # 遍历读取的词汇列表，逐个添加到词汇表字典中
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")  # 去除行末的换行符
        vocab[token] = index
    # 返回构建好的词汇表字典
    return vocab

# 对文本进行基本的空格清理和拆分的函数
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # 去除文本两侧的空格
    text = text.strip()
    # 若文本为空，则返回空列表
    if not text:
        return []
    # 使用空格分割文本，得到分词列表
    tokens = text.split()
    # 返回分词列表
    return tokens

# BertTokenizer 类，继承自 PreTrainedTokenizer 类
class BertTokenizer(PreTrainedTokenizer):
    r"""
    Construct a BERT tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            词汇表文件路径。
        do_lower_case (`bool`, *optional*, defaults to `True`):
            在进行分词时是否将输入转换为小写。
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            在使用 WordPiece 分词之前是否进行基本分词。
        never_split (`Iterable`, *optional*):
            在分词过程中永远不会被拆分的标记集合。仅在 `do_basic_tokenize=True` 时生效。
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            未知标记。词汇表中不存在的标记会被设置为此标记。
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            分隔符标记，在构建包含多个序列的序列时使用，例如用于序列分类或用于文本和问题之间的问答。也用作使用特殊标记构建的序列的最后一个标记。
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            用于填充的标记，例如在对不同长度的序列进行批处理时。
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            分类器标记，用于进行序列分类（整个序列的分类而不是每个标记的分类）。使用特殊标记构建序列时，它是序列的第一个标记。
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            用于掩码值的标记。这是用于使用掩码语言建模训练此模型的标记。模型将尝试预测此标记。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否对中文字符进行分词。

            对于日语，这可能需要停用（参见此 [问题](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否去除所有重音符号。如果未指定此选项，则将由 `lowercase` 的值确定（与原始 BERT 一样）。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

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
        # 检查词汇表文件是否存在，如果不存在则引发 ValueError 异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表
        self.vocab = load_vocab(vocab_file)
        # 创建一个从 ids 到 tokens 的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 设置是否进行基本分词的标志
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要进行基本分词，则创建 BasicTokenizer 对象
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        # 创建 WordpieceTokenizer 对象
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的构造函数
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
        # 返回是否执行小写转换的标志
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        # 返回词汇表的大小
        return len(self.vocab)

    def get_vocab(self):
        # 返回词汇表和添加的特殊 token 编码器的组合
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, split_special_tokens=False):
        # 分词函数，将文本分割成 token 序列
        split_tokens = []
        # 如果需要进行基本分词
        if self.do_basic_tokenize:
            # 对文本进行基本分词，never_split 参数指定不需要分割的 token
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果 token 在不需要分割的 token 集合中
                if token in self.basic_tokenizer.never_split:
                    # 将该 token 添加到分词结果中
                    split_tokens.append(token)
                else:
                    # 使用 WordpieceTokenizer 对 token 进行分词，并将分词结果添加到 split_tokens 中
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 使用 WordpieceTokenizer 对文本进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将 token 转换为对应的 id
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将 id 转换为对应的 token
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将 token 序列转换为单个字符串
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

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
        # 如果只有一个句子，则在开始加入 [CLS] 标记，中间添加句子的词对应的 ID，最后加入 [SEP] 标记
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 否则，构建带有两个句子的序列
        cls = [self.cls_token_id]  # [CLS] 标记
        sep = [self.sep_token_id]  # [SEP] 标记
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

        # 如果输入的 token_ids 已经包含特殊标记，则直接调用基类的方法
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 否则，根据是否有两个句子来添加特殊标记的掩码
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

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
        # 定义分隔符的 token ID 列表，用于标志两个序列的分隔位置
        sep = [self.sep_token_id]
        # 定义类别标志的 token ID 列表，用于标志序列的开始
        cls = [self.cls_token_id]
        # 如果 token_ids_1 为空，则只返回第一部分的 mask（全部为 0）
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则，返回序列对应的 token type ID 列表，其中第一部分为 0，第二部分为 1
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引为 0
        index = 0
        # 检查保存目录是否存在
        if os.path.isdir(save_directory):
            # 若存在，则设置词汇表文件路径
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 若不存在，则直接设置词汇表文件路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开词汇表文件进行写入操作，设置编码为 UTF-8
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表字典，按照 token_index 排序写入词汇表文件
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 若当前索引与 token_index 不相等，发出警告
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 将 token 写入词汇表文件，并添加换行符
                writer.write(token + "\n")
                # 索引加一
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
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            In some instances we want to skip the basic punctuation splitting so that later tokenization can capture
            the full context of the words, such as contractions.
    """

    # 初始化 BasicTokenizer 类
    def __init__(
        self,
        do_lower_case=True,  # 是否将输入转换为小写
        never_split=None,  # 在标记化期间永远不会被拆分的标记集合
        tokenize_chinese_chars=True,  # 是否对中文字符进行标记化
        strip_accents=None,  # 是否去除所有重音符号
        do_split_on_punc=True,  # 是否在基本标点符号上拆分
    ):
        # 如果 `never_split` 未指定，将其设置为空列表
        if never_split is None:
            never_split = []
        # 初始化实例变量
        self.do_lower_case = do_lower_case  # 是否转换为小写
        self.never_split = set(never_split)  # 永远不会被拆分的标记集合
        self.tokenize_chinese_chars = tokenize_chinese_chars  # 是否标记化中文字符
        self.strip_accents = strip_accents  # 是否去除重音符号
        self.do_split_on_punc = do_split_on_punc  # 是否在基本标点符号上拆分
    # 对文本进行基本的分词。用于子词分词，请参见WordPieceTokenizer。

    # 如果never_split不为空，则将其与self.never_split进行合并，以获取不分割的标记列表。
    never_split = self.never_split.union(set(never_split)) if never_split else self.never_split

    # 清理文本，去除无效字符
    text = self._clean_text(text)

    # 如果设置为对中文字符进行分词，则调用_tokenize_chinese_chars()方法处理中文字符
    if self.tokenize_chinese_chars:
        text = self._tokenize_chinese_chars(text)

    # 将文本进行Unicode范式化，将文本统一表示为规范化的形式
    unicode_normalized_text = unicodedata.normalize("NFC", text)

    # 对规范化后的文本进行空格分词，得到原始标记列表
    orig_tokens = whitespace_tokenize(unicode_normalized_text)

    # 初始化分割后的标记列表
    split_tokens = []

    # 遍历原始标记列表
    for token in orig_tokens:
        # 如果标记不在不分割的标记列表中
        if token not in never_split:
            # 如果设置为小写化标记，则将标记转换为小写
            if self.do_lower_case:
                token = token.lower()
                # 如果设置为去除重音符号，则去除标记中的重音符号
                if self.strip_accents is not False:
                    token = self._run_strip_accents(token)
            # 如果设置为去除重音符号，则去除标记中的重音符号
            elif self.strip_accents:
                token = self._run_strip_accents(token)
        # 在标点符号处进行分割，扩展标记列表
        split_tokens.extend(self._run_split_on_punc(token, never_split))

    # 将分割后的标记列表重新进行空格分词，得到输出的标记列表
    output_tokens = whitespace_tokenize(" ".join(split_tokens))

    # 返回输出的标记列表
    return output_tokens

# 从文本中去除重音符号
def _run_strip_accents(self, text):
    # 对文本进行NFD范式化，将文本中的重音符号分离出来
    text = unicodedata.normalize("NFD", text)
    # 初始化输出列表
    output = []
    # 遍历文本中的字符
    for char in text:
        # 获取字符的Unicode类别
        cat = unicodedata.category(char)
        # 如果字符为重音符号，则跳过
        if cat == "Mn":
            continue
        # 将非重音符号的字符添加到输出列表中
        output.append(char)
    # 将输出列表中的字符组合成字符串返回
    return "".join(output)
    # 在给定文本上执行基于标点符号的分割
    def _run_split_on_punc(self, text, never_split=None):
        # 如果不执行基于标点符号的分割，或者文本在不分割列表中，则直接返回文本
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果字符是标点符号，则将其作为单独的列表项添加到输出列表中
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果当前字符不是标点符号，则将其添加到当前单词的列表中
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    # 在中文字符周围添加空格
    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符是中文字符，则在其周围添加空格
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    # 检查字符是否是CJK字符的代码点
    def _is_chinese_char(self, cp):
        # 这里定义的“中文字符”是CJK Unicode块中的任何字符
        # 详见：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
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

    # 在文本上执行无效字符删除和空格清理
    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符为0或0xFFFD或是控制字符，则跳过该字符
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果字符是空格，则将其替换为单个空格字符
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化 WordpieceTokenizer 类，设置词汇表、未知标记和每个单词的最大输入字符数
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

        output_tokens = []  # 存储输出的 wordpiece token
        for token in whitespace_tokenize(text):  # 遍历以空格分隔的文本中的每个 token
            chars = list(token)  # 将 token 拆分为字符列表
            if len(chars) > self.max_input_chars_per_word:  # 如果 token 的字符数超过了最大输入字符数
                output_tokens.append(self.unk_token)  # 将未知标记添加到输出 token 列表中
                continue

            is_bad = False  # 标记 token 是否为无效
            start = 0  # 初始化子串的起始索引
            sub_tokens = []  # 存储子 token
            while start < len(chars):  # 当起始索引小于字符列表的长度时
                end = len(chars)  # 结束索引为字符列表的长度
                cur_substr = None  # 初始化当前子串
                while start < end:  # 当起始索引小于结束索引时
                    substr = "".join(chars[start:end])  # 将字符列表中的字符连接为子串
                    if start > 0:  # 如果起始索引大于 0
                        substr = "##" + substr  # 添加 ## 前缀表示该子串为一个词中的一部分
                    if substr in self.vocab:  # 如果子串在词汇表中
                        cur_substr = substr  # 更新当前子串
                        break
                    end -= 1  # 否则，缩小结束索引
                if cur_substr is None:  # 如果当前子串为空
                    is_bad = True  # 标记 token 为无效
                    break
                sub_tokens.append(cur_substr)  # 将当前子串添加到子 token 列表中
                start = end  # 更新起始索引为结束索引

            if is_bad:  # 如果 token 为无效
                output_tokens.append(self.unk_token)  # 将未知标记添加到输出 token 列表中
            else:
                output_tokens.extend(sub_tokens)  # 否则，将子 token 列表添加到输出 token 列表中
        return output_tokens  # 返回输出的 wordpiece tokens
```  
```