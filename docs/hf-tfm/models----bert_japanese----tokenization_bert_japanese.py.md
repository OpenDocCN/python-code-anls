# `.\transformers\models\bert_japanese\tokenization_bert_japanese.py`

```
# 导入必要的库和模块
import collections  # 导入collections模块，用于处理集合类型数据
import copy  # 导入copy模块，用于复制对象
import os  # 导入os模块，用于操作系统相关功能
import unicodedata  # 导入unicodedata模块，用于Unicode字符数据的处理
from typing import Any, Dict, List, Optional, Tuple  # 导入typing模块，用于类型提示

# 导入tokenization_utils中的PreTrainedTokenizer类和一些辅助函数
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace  
# 导入utils模块中的is_sentencepiece_available函数和logging模块
from ...utils import is_sentencepiece_available, logging  

# 如果sentencepiece可用，则导入sentencepiece模块，否则置为None
if is_sentencepiece_available():  
    import sentencepiece as spm  
else:  
    spm = None  

# 获取logger对象，用于记录日志
logger = logging.get_logger(__name__)  

# 定义词汇文件的名称常量字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "spm_file": "spiece.model"}  

# 定义分隔符常量，用于表示subword分词的起始
SPIECE_UNDERLINE = "▁"  

# 预训练模型词汇文件的映射字典
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

# 预训练模型的位置编码大小字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {  
    "cl-tohoku/bert-base-japanese": 512,
    "cl-tohoku/bert-base-japanese-whole-word-masking": 512,
    "cl-tohoku/bert-base-japanese-char": 512,
    "cl-tohoku/bert-base-japanese-char-whole-word-masking": 512,
}

# 预训练模型的初始化配置字典
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

# 定义load_vocab函数，用于加载词汇表文件
# 该函数实际上是从transformers.models.bert.tokenization_bert.load_vocab复制过来的
def load_vocab(vocab_file):  
``` 
    # 将一个词汇文件加载到字典中
    vocab = collections.OrderedDict()
    # 打开词汇文件进行读取，使用 UTF-8 编码
    with open(vocab_file, "r", encoding="utf-8") as reader:
        # 逐行读取文件内容，并将每行内容存储到列表中
        tokens = reader.readlines()
    # 遍历 tokens 列表，获取每行索引及对应的内容
    for index, token in enumerate(tokens):
        # 去除每行末尾的换行符
        token = token.rstrip("\n")
        # 将每行内容作为键，索引作为值，添加到 vocab 字典中
        vocab[token] = index
    # 返回加载完成的词汇字典
    return vocab
# 从transformers.models.bert.tokenization_bert.whitespace_tokenize复制了whitespace_tokenize函数
def whitespace_tokenize(text):
    """对文本进行基本的空白符清理和分词"""
    # 去除文本两端的空白符
    text = text.strip()
    # 如果文本为空，则返回空列表
    if not text:
        return []
    # 使用空白符进行文本分词
    tokens = text.split()
    return tokens


# 定义一个BertJapaneseTokenizer类，继承自PreTrainedTokenizer
class BertJapaneseTokenizer(PreTrainedTokenizer):
    r"""
    为日文文本构建一个BERT分词器。

    此分词器继承自[`PreTrainedTokenizer`]，其中包含大多数主要方法。用户应该参考：
    此超类以获取有关这些方法的更多信息。

    Args:
        vocab_file (`str`):
            一个每行一个单词的词汇文件的路径。
        spm_file (`str`, *optional*):
            包含词汇的 [SentencePiece](https://github.com/google/sentencepiece) 文件的路径
            （通常具有 .spm 或 .model 扩展名）。
        do_lower_case (`bool`, *optional*, 默认为 `True`):
            是否对输入进行小写处理。仅在 do_basic_tokenize=True 时有效。
        do_word_tokenize (`bool`, *optional*, 默认为 `True`):
            是否进行单词分词。
        do_subword_tokenize (`bool`, *optional*, 默认为 `True`):
            是否进行子词分词。
        word_tokenizer_type (`str`, *optional*, 默认为 `"basic"`):
            单词分词器的类型。可从 ["basic", "mecab", "sudachi", "jumanpp"] 中选择。
        subword_tokenizer_type (`str`, *optional*, 默认为 `"wordpiece"`):
            子词分词器的类型。可从 ["wordpiece", "character", "sentencepiece",] 中选择。
        mecab_kwargs (`dict`, *optional*):
            传递给 `MecabTokenizer` 构造函数的字典。
        sudachi_kwargs (`dict`, *optional*):
            传递给 `SudachiTokenizer` 构造函数的字典。
        jumanpp_kwargs (`dict`, *optional*):
            传递给 `JumanppTokenizer` 构造函数的字典。
    """

    # 定义类变量
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # 初始化方法
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
        # 调用父类的初始化方法
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        # 将参数赋值给实例变量
        self.vocab_file = vocab_file
        self.spm_file = spm_file
        self.do_lower_case = do_lower_case
        self.do_word_tokenize = do_word_tokenize
        self.do_subword_tokenize = do_subword_tokenize
        self.word_tokenizer_type = word_tokenizer_type
        self.subword_tokenizer_type = subword_tokenizer_type
        self.never_split = never_split
        self.mecab_kwargs = mecab_kwargs
        self.sudachi_kwargs = sudachi_kwargs
        self.jumanpp_kwargs = jumanpp_kwargs

    # 类属性的getter方法
    @property
    def do_lower_case(self):
        return self.lower_case

    # 重新定义对象的序列化方法
    def __getstate__(self):
        # 复制对象的字典表示
        state = dict(self.__dict__)
        # 如果单词分词器的类型为 "mecab"、"sudachi" 或 "jumanpp"，则从状态中删除单词分词器
        if self.word_tokenizer_type in ["mecab", "sudachi", "jumanpp"]:
            del state["word_tokenizer"]
        return state
    # 定义一个特殊方法，用于反序列化对象的状态
    def __setstate__(self, state):
        # 将对象的字典状态更新为给定状态
        self.__dict__ = state
        # 如果使用的词级别分词器是 "mecab"，则使用 MecabTokenizer 进行分词
        if self.word_tokenizer_type == "mecab":
            self.word_tokenizer = MecabTokenizer(
                # 设定是否将所有字符转换为小写
                do_lower_case=self.do_lower_case, 
                # 指定不分割的特殊标记
                never_split=self.never_split, 
                # 将 self.mecab_kwargs 的关键字参数解包传递给 MecabTokenizer 构造函数
                **(self.mecab_kwargs or {})
            )
        # 如果使用的词级别分词器是 "sudachi"，则使用 SudachiTokenizer 进行分词
        elif self.word_tokenizer_type == "sudachi":
            self.word_tokenizer = SudachiTokenizer(
                # 设定是否将所有字符转换为小写
                do_lower_case=self.do_lower_case, 
                # 指定不分割的特殊标记
                never_split=self.never_split, 
                # 将 self.sudachi_kwargs 的关键字参数解包传递给 SudachiTokenizer 构造函数
                **(self.sudachi_kwargs or {})
            )
        # 如果使用的词级别分词器是 "jumanpp"，则使用 JumanppTokenizer 进行分词
        elif self.word_tokenizer_type == "jumanpp":
            self.word_tokenizer = JumanppTokenizer(
                # 设定是否将所有字符转换为小写
                do_lower_case=self.do_lower_case, 
                # 指定不分割的特殊标记
                never_split=self.never_split, 
                # 将 self.jumanpp_kwargs 的关键字参数解包传递给 JumanppTokenizer 构造函数
                **(self.jumanpp_kwargs or {})
            )

    # 定义一个私有方法，用于对文本进行分词处理
    def _tokenize(self, text):
        # 如果设置了执行词级别分词，则使用词级别分词器进行分词，否则直接使用文本作为 token
        if self.do_word_tokenize:
            tokens = self.word_tokenizer.tokenize(text, never_split=self.all_special_tokens)
        else:
            tokens = [text]

        # 如果设置了执行子词级别分词，则对每个词进行子词级别分词处理
        if self.do_subword_tokenize:
            split_tokens = [sub_token for token in tokens for sub_token in self.subword_tokenizer.tokenize(token)]
        else:
            split_tokens = tokens

        # 返回分词结果
        return split_tokens

    # 定义一个属性，用于获取词汇表的大小
    @property
    def vocab_size(self):
        # 如果使用的子词级别分词器是 "sentencepiece"，则返回子词级别分词器的词汇表大小
        if self.subword_tokenizer_type == "sentencepiece":
            return len(self.subword_tokenizer.sp_model)
        # 否则返回词汇表的大小
        return len(self.vocab)

    # 定义一个方法，用于获取词汇表
    def get_vocab(self):
        # 如果使用的子词级别分词器是 "sentencepiece"，则返回包含特殊标记的词汇表
        if self.subword_tokenizer_type == "sentencepiece":
            vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
            # 添加用户自定义的特殊标记到词汇表中
            vocab.update(self.added_tokens_encoder)
            return vocab
        # 否则返回包含用户自定义特殊标记的词汇表
        return dict(self.vocab, **self.added_tokens_encoder)

    # 定义一个私有方法，用于将 token 转换为对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 如果使用的子词级别分词器是 "sentencepiece"，则使用 SentencePiece 模型将 token 转换为 id
        if self.subword_tokenizer_type == "sentencepiece":
            return self.subword_tokenizer.sp_model.PieceToId(token)
        # 否则根据词汇表将 token 转换为 id，并在找不到对应 token 时返回未知标记的 id
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 定义一个私有方法，用于将 id 转换为对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 如果使用的子词级别分词器是 "sentencepiece"，则使用 SentencePiece 模型将 id 转换为 token
        if self.subword_tokenizer_type == "sentencepiece":
            return self.subword_tokenizer.sp_model.IdToPiece(index)
        # 否则根据词汇表将 id 转换为 token，并在找不到对应 id 时返回未知标记
        return self.ids_to_tokens.get(index, self.unk_token)

    # 定义一个方法，用于将 token 序列转换为字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 如果使用的子词级别分词器是 "sentencepiece"，则使用 SentencePiece 模型将 tokens 转换为字符串
        if self.subword_tokenizer_type == "sentencepiece":
            return self.subword_tokenizer.sp_model.decode(tokens)
        # 否则将 tokens 拼接为字符串，同时去除子词级别分词器添加的 "##" 标记，并去除首尾的空格
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # 从 transformers.models.bert.tokenization_bert.BertTokenizer 复制的方法，用于构建包含特殊标记的输入序列
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        为序列分类任务构建模型输入，通过连接和添加特殊标记。BERT 序列的格式如下：

        - 单个序列: `[CLS] X [SEP]`
        - 序列对: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 带有适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # 从 transformers.models.bert.tokenization_bert.BertTokenizer.get_special_tokens_mask 复制而来
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列 ID。在使用 tokenizer 的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 的列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                标记列表是否已经用于模型的特殊标记格式化。

        Returns:
            `List[int]`: 一个整数列表，范围在 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    # 从 transformers.models.bert.tokenization_bert.BertTokenizer.create_token_type_ids_from_sequences 复制而来
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
        # 定义 [SEP] 标记的 ID 列表，用于分隔不同的序列
        sep = [self.sep_token_id]
        # 定义 [CLS] 标记的 ID 列表，用于序列的开始
        cls = [self.cls_token_id]
        # 如果第二个序列为 None，则只返回第一个序列部分的 mask（全为 0）
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 返回一个由 0 和 1 组成的 mask 列表，用于区分两个序列
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录已存在
        if os.path.isdir(save_directory):
            # 如果使用 sentencepiece 分词器
            if self.subword_tokenizer_type == "sentencepiece":
                # 构造保存文件路径，添加前缀和后缀
                vocab_file = os.path.join(
                    save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["spm_file"]
                )
            else:
                # 构造保存文件路径，添加前缀和后缀
                vocab_file = os.path.join(
                    save_directory,
                    (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"],
                )
        else:
            # 如果保存目录不存在，则保存路径为指定的目录
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory

        # 如果使用 sentencepiece 分词器
        if self.subword_tokenizer_type == "sentencepiece":
            # 以二进制写入模式打开文件
            with open(vocab_file, "wb") as writer:
                # 获取 sentencepiece 模型的序列化内容并写入文件
                content_spiece_model = self.subword_tokenizer.sp_model.serialized_model_proto()
                writer.write(content_spiece_model)
        else:
            # 使用 utf-8 编码以文本写入模式打开文件
            with open(vocab_file, "w", encoding="utf-8") as writer:
                index = 0
                # 遍历词汇表中的单词及其索引，并按照索引排序
                for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                    # 如果索引不连续，发出警告
                    if index != token_index:
                        logger.warning(
                            f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                            " Please check that the vocabulary is not corrupted!"
                        )
                        index = token_index
                    # 将单词写入文件
                    writer.write(token + "\n")
                    index += 1
        # 返回保存的词汇表文件路径
        return (vocab_file,)
class MecabTokenizer:
    """Runs basic tokenization with MeCab morphological parser."""

    def __init__(
        self,
        do_lower_case=False,  # 是否将所有字符转换为小写，默认为 False
        never_split=None,  # 不需要拆分的单词列表，默认为 None
        normalize_text=True,  # 是否对文本进行标准化处理，默认为 True
        mecab_dic: Optional[str] = "ipadic",  # MeCab 使用的词典，默认为 "ipadic"
        mecab_option: Optional[str] = None,  # MeCab 的其他选项，默认为 None
    def tokenize(self, text, never_split=None, **kwargs):
        """Tokenizes a piece of text."""
        if self.normalize_text:  # 如果需要对文本进行标准化处理
            text = unicodedata.normalize("NFKC", text)  # 对文本进行 NFKC 标准化处理

        never_split = self.never_split + (never_split if never_split is not None else [])  # 合并 never_split 参数
        tokens = []  # 初始化 tokens 列表，用于存储分词结果

        for word in self.mecab(text):  # 对文本进行分词
            token = word.surface  # 获取分词结果中的表面形式

            if self.do_lower_case and token not in never_split:  # 如果需要将字符转换为小写且不在 never_split 列表中
                token = token.lower()  # 将 token 转换为小写形式

            tokens.append(token)  # 将 token 添加到 tokens 列表中

        return tokens  # 返回分词结果


class SudachiTokenizer:
    """Runs basic tokenization with Sudachi morphological parser."""

    def __init__(
        self,
        do_lower_case=False,  # 是否将所有字符转换为小写，默认为 False
        never_split=None,  # 不需要拆分的单词列表，默认为 None
        normalize_text=True,  # 是否对文本进行标准化处理，默认为 True
        trim_whitespace=False,  # 是否去除分词结果中的空白符，默认为 False
        sudachi_split_mode="A",  # Sudachi 分词模式，默认为 "A"
        sudachi_config_path=None,  # Sudachi 配置文件路径，默认为 None
        sudachi_resource_dir=None,  # Sudachi 资源文件目录，默认为 None
        sudachi_dict_type="core",  # Sudachi 使用的词典类型，默认为 "core"
    ):
        """
        Constructs a SudachiTokenizer.

        Args:
            **do_lower_case**: (*optional*) boolean (default True)
                是否将输入转换为小写。
            **never_split**: (*optional*) list of str
                保留以备向后兼容。现在直接在基类级别实现（参见[`PreTrainedTokenizer.tokenize`]）不分割的标记列表。
            **normalize_text**: (*optional*) boolean (default True)
                是否在标记化之前对文本应用Unicode标准化。
            **trim_whitespace**: (*optional*) boolean (default False)
                是否修剪所有空白、制表符、换行符。
            **sudachi_split_mode**: (*optional*) string
                Sudachi的分割模式，可选择"A"、"B"、"C"。
            **sudachi_config_path**: (*optional*) string
            **sudachi_resource_dir**: (*optional*) string
            **sudachi_dict_type**: (*optional*) string
                Sudachi的字典类型，可选择"small"、"core"、"full"。
        """

        self.do_lower_case = do_lower_case
        self.never_split = never_split if never_split is not None else []
        self.normalize_text = normalize_text
        self.trim_whitespace = trim_whitespace

        try:
            from sudachipy import dictionary, tokenizer
        except ImportError:
            raise ImportError(
                "You need to install sudachipy to use SudachiTokenizer. "
                "See https://github.com/WorksApplications/SudachiPy for installation."
            )

        if sudachi_split_mode == "A":
            self.split_mode = tokenizer.Tokenizer.SplitMode.A
        elif sudachi_split_mode == "B":
            self.split_mode = tokenizer.Tokenizer.SplitMode.B
        elif sudachi_split_mode == "C":
            self.split_mode = tokenizer.Tokenizer.SplitMode.C
        else:
            raise ValueError("Invalid sudachi_split_mode is specified.")

        self.sudachi = dictionary.Dictionary(
            config_path=sudachi_config_path, resource_dir=sudachi_resource_dir, dict=sudachi_dict_type
        ).create(self.split_mode)

    def tokenize(self, text, never_split=None, **kwargs):
        """Tokenizes a piece of text."""
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        never_split = self.never_split + (never_split if never_split is not None else [])
        tokens = []

        for word in self.sudachi.tokenize(text):
            token = word.surface()

            if self.do_lower_case and token not in never_split:
                token = token.lower()

            if self.trim_whitespace:
                if token.strip() == "":
                    continue
                else:
                    token = token.strip()

            tokens.append(token)

        return tokens
class JumanppTokenizer:
    """Runs basic tokenization with jumanpp morphological parser."""

    def __init__(
        self,
        do_lower_case=False,  # 是否将输入转换为小写，默认为 False
        never_split=None,  # 不进行拆分的标记列表，默认为空
        normalize_text=True,  # 是否在分词之前对文本进行 Unicode 规范化，默认为 True
        trim_whitespace=False,  # 是否修剪所有空白、制表符和换行符，默认为 False
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

        self.do_lower_case = do_lower_case  # 是否将输入转换为小写
        self.never_split = never_split if never_split is not None else []  # 不进行拆分的标记列表
        self.normalize_text = normalize_text  # 是否在分词之前对文本进行 Unicode 规范化
        self.trim_whitespace = trim_whitespace  # 是否修剪所有空白、制表符和换行符

        try:
            import rhoknp  # 导入 rhoknp 库
        except ImportError:
            raise ImportError(
                "You need to install rhoknp to use JumanppTokenizer. "
                "See https://github.com/ku-nlp/rhoknp for installation."
            )

        self.juman = rhoknp.Jumanpp()  # 创建 Jumanpp 对象

    def tokenize(self, text, never_split=None, **kwargs):
        """Tokenizes a piece of text."""
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)  # 对文本进行 Unicode 规范化

        text = text.strip()  # 去除文本两侧的空白字符

        never_split = self.never_split + (never_split if never_split is not None else [])  # 合并用户提供的和默认的不拆分标记列表
        tokens = []

        for mrph in self.juman.apply_to_sentence(text).morphemes:  # 对文本进行 Juman++ 分词
            token = mrph.text  # 获取词素文本

            if self.do_lower_case and token not in never_split:  # 如果需要小写化且当前词素不在不拆分列表中
                token = token.lower()  # 将词素转换为小写

            if self.trim_whitespace:  # 如果需要修剪空白字符
                if token.strip() == "":  # 如果词素去除两侧空白后为空字符串
                    continue  # 跳过该词素
                else:
                    token = token.strip()  # 去除词素两侧的空白字符

            tokens.append(token)  # 将处理后的词素添加到 tokens 列表中

        return tokens  # 返回分词结果列表


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
        self.vocab = vocab  # 词汇表对象
        self.unk_token = unk_token  # 未登录词标记
        self.normalize_text = normalize_text  # 是否在分词之前对文本进行 Unicode 规范化
    # 将文本标记为字符。

    # 例如，`input = "apple"` 将返回 `["a", "p", "p", "l", "e"]`。

    # Args:
    #     text: 一个单个标记或由空格分隔的标记。
    #           这应该已经通过 *BasicTokenizer* 处理过。

    # Returns:
    #     一个字符列表。
    
    def tokenize(self, text):
        # 如果需要对文本进行规范化，则对文本进行 Unicode 规范化
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        # 初始化输出标记列表
        output_tokens = []
        # 遍历文本中的每个字符
        for char in text:
            # 如果字符不在词汇表中，则将其替换为未知标记，并继续下一个字符
            if char not in self.vocab:
                output_tokens.append(self.unk_token)
                continue

            # 将字符添加到输出标记列表中
            output_tokens.append(char)

        # 返回输出标记列表
        return output_tokens
# 从transformers.models.bert.tokenization_bert.BasicTokenizer复制而来，定义了一个BasicTokenizer类
class BasicTokenizer(object):
    """
    构造一个BasicTokenizer，执行基本的标记化（标点符号分割、小写化等）。

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在标记化时将输入转换为小写。
        never_split (`Iterable`, *optional*):
            在标记化过程中永远不会被分割的标记集合。仅在`do_basic_tokenize=True`时有效。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否对中文字符进行标记化。

            这可能会对日文产生负面影响（参见这个[issue](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否去除所有的重音符号。如果未指定此选项，则将由`lowercase`的值决定（与原始BERT相同）。
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            在某些情况下，我们希望跳过基本的标点符号分割，以便后续的标记化可以捕获单词的完整上下文，例如缩写。
    """

    # 初始化BasicTokenizer对象
    def __init__(
        self,
        do_lower_case=True,  # 是否小写化输入，默认为True
        never_split=None,  # 永远不被分割的标记集合，默认为None
        tokenize_chinese_chars=True,  # 是否对中文字符进行标记化，默认为True
        strip_accents=None,  # 是否去除重音符号，默认为None
        do_split_on_punc=True,  # 是否在标点符号上分割，默认为True
    ):
        if never_split is None:
            never_split = []
        # 将传入的永远不分割的标记集合转换为集合类型
        self.never_split = set(never_split)
        self.do_lower_case = do_lower_case  # 是否小写化输入
        self.tokenize_chinese_chars = tokenize_chinese_chars  # 是否对中文字符进行标记化
        self.strip_accents = strip_accents  # 是否去除重音符号
        self.do_split_on_punc = do_split_on_punc  # 是否在标点符号上分割
    # 对文本进行基本的分词处理。如需子词分词，请参见WordPieceTokenizer。

    # 如果never_split不为None，则将其与self.never_split进行并集操作，得到不需要分割的token列表
    never_split = self.never_split.union(set(never_split)) if never_split else self.never_split

    # 清理文本，去除其中的特殊字符
    text = self._clean_text(text)

    # 若开启了对中文字符的分词，则对文本中的中文字符进行分词处理
    if self.tokenize_chinese_chars:
        text = self._tokenize_chinese_chars(text)

    # 将文本进行Unicode标准化，防止不同Unicode编码表示的相同字符被当作不同字符处理
    unicode_normalized_text = unicodedata.normalize("NFC", text)

    # 使用空白字符进行分词
    orig_tokens = whitespace_tokenize(unicode_normalized_text)

    # 初始化空列表，用于存储分词后的token
    split_tokens = []

    # 遍历原始token列表
    for token in orig_tokens:
        # 如果token不在不需要分割的token列表中
        if token not in never_split:
            # 如果需要将token转换为小写，并且不需要保留重音符号，则对token进行小写转换
            if self.do_lower_case:
                token = token.lower()
                # 如果需要去除重音符号，则对token进行去除重音符号的处理
                if self.strip_accents is not False:
                    token = self._run_strip_accents(token)
            # 如果需要去除重音符号，则对token进行去除重音符号的处理
            elif self.strip_accents:
                token = self._run_strip_accents(token)
        # 将处理后的token进行分割处理，并添加到split_tokens列表中
        split_tokens.extend(self._run_split_on_punc(token, never_split))

    # 使用空白字符进行分词，重新组合分割后的token列表
    output_tokens = whitespace_tokenize(" ".join(split_tokens))

    # 返回处理后的token列表
    return output_tokens


def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    # 对文本进行Unicode标准化，将字符拆分为基字符和组合字符
    text = unicodedata.normalize("NFD", text)

    # 初始化空列表，用于存储去除重音符号后的字符
    output = []

    # 遍历文本中的每个字符
    for char in text:
        # 获取字符的Unicode类别
        cat = unicodedata.category(char)

        # 如果字符的Unicode类别为Mn（Mark, Nonspacing），即重音符号，则跳过该字符
        if cat == "Mn":
            continue
        # 否则将字符添加到输出列表中
        output.append(char)

    # 将输出列表中的字符重新组合成字符串
    return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要根据标点符号分割文本，或者指定了不进行分割的文本，则直接返回原文本
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        # 遍历文本中的每个字符
        while i < len(chars):
            char = chars[i]
            # 如果字符是标点符号，则将其作为一个独立的词加入到输出列表中，并标记开始一个新词
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，且标记为开始一个新词，则在输出列表中添加一个新的空列表
                if start_new_word:
                    output.append([])
                start_new_word = False
                # 将当前字符加入到最后一个词的列表中
                output[-1].append(char)
            i += 1

        # 将列表中的词重新组合成字符串并返回
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        # 遍历文本中的每个字符
        for char in text:
            cp = ord(char)
            # 如果是中日韩字符，则在其前后添加空格
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        # 将列表中的字符重新组合成字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查码点是否为中日韩字符的 Unicode 范围
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
            # 如果字符为无效字符或控制字符，则跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果是空白字符，则替换为单个空格，否则保留字符
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将列表中的字符重新组合成字符串并返回
        return "".join(output)
# 从 transformers.models.bert.tokenization_bert.WordpieceTokenizer 复制代码，定义了一个 WordpieceTokenizer 类
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化 WordpieceTokenizer 实例，设置词汇表、未知标记和每个单词最大输入字符数
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

        # 将文本分词成 WordPiece tokens
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                # 如果单词长度超过最大输入字符数，将该单词替换为未知标记
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    # 从最长的子串开始匹配，直到找到词汇表中的词或最短的子串
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    # 如果找不到词汇表中的词，则标记为错误
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                # 如果存在错误，将单词替换为未知标记
                output_tokens.append(self.unk_token)
            else:
                # 将子词添加到输出 tokens 中
                output_tokens.extend(sub_tokens)
        return output_tokens


# 定义 SentencepieceTokenizer 类，运行 sentencepiece tokenization，基于 transformers.models.albert.tokenization_albert.AlbertTokenizer
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
        # 初始化 SentencepieceTokenizer 实例，设置词汇表、未知标记、是否小写、是否移除空格和是否保留重音
        self.vocab = vocab
        self.unk_token = unk_token
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents

        # 如果未提供 sp_model_kwargs，则使用空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # 使用 sentencepiece 的参数初始化 sentencepiece 模型
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载 sentencepiece 词汇表
        self.sp_model.Load(self.vocab)
    # 对输入文本进行预处理，根据参数决定是否移除空格、替换引号等操作
    def preprocess_text(self, inputs):
        # 如果需要移除空格，则去除首尾空格并用单个空格重新连接
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        # 替换连续的两个反引号为双引号
        outputs = outputs.replace("``", '"').replace("''", '"')

        # 如果不保留重音符号，则进行 Unicode 标准化，去除组合字符
        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        # 如果需要转换为小写，则将文本转换为小写
        if self.do_lower_case:
            outputs = outputs.lower()

        # 返回预处理后的文本
        return outputs

    # 使用 SentencePiece 对文本进行分词
    def tokenize(self, text):
        """
        Tokenizes text by sentencepiece. Based on [SentencePiece](https://github.com/google/sentencepiece).
        Tokenization needs the given vocabulary.

        Args:
            text: A string needs to be tokenized.

        Returns:
            A list of sentencepiece tokens.
        """
        # 预处理输入文本
        text = self.preprocess_text(text)
        # 使用 SentencePiece 模型对文本进行编码，输出字符串形式的分词结果
        pieces = self.sp_model.encode(text, out_type=str)
        new_pieces = []
        # 遍历分词结果
        for piece in pieces:
            # 对长于1个字符且以逗号结尾且倒数第二个字符是数字的分词进行处理
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                # 对分词结果去除逗号后的部分进行再次分词
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SPIECE_UNDERLINE, ""))
                # 如果原始分词不以特殊符号开头，但拆分后的第一个分词以特殊符号开头，则修正
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    # 如果拆分后的第一个分词只有一个字符，则去除
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                # 将逗号加入到分词结果中
                cur_pieces.append(piece[-1])
                # 将拆分后的分词加入到新的分词列表中
                new_pieces.extend(cur_pieces)
            else:
                # 如果不满足上述条件，则将分词直接加入到新的分词列表中
                new_pieces.append(piece)

        # 返回处理后的分词列表
        return new_pieces
```