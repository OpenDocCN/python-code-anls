# `.\models\electra\tokenization_electra.py`

```
# 以 UTF-8 编码声明文件编码方式
# 版权声明及许可信息
# 该代码基于 Apache License, Version 2.0 开源许可证发布，详情请访问指定网址获取完整许可信息
# 导入所需的标准库模块和函数
# collections 模块提供了额外的数据类型供 Python 内置数据类型的扩展
# os 模块提供了与操作系统交互的功能
# unicodedata 模块包含用于 Unicode 数据库的访问功能
# 从 typing 模块导入 List, Optional, Tuple，用于类型提示
# 从 tokenization_utils 模块中导入 PreTrainedTokenizer 类和一些辅助函数
# 从 utils 模块导入 logging 函数
from typing import List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义一个字典，指定每个文件的名称及其对应的默认文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 定义一个嵌套字典，指定预训练模型和其对应的词汇文件 URL
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

# 定义一个字典，指定每个预训练模型和其对应的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/electra-small-generator": 512,
    "google/electra-base-generator": 512,
    "google/electra-large-generator": 512,
    "google/electra-small-discriminator": 512,
    "google/electra-base-discriminator": 512,
    "google/electra-large-discriminator": 512,
}

# 定义一个字典，指定每个预训练模型的初始配置
PRETRAINED_INIT_CONFIGURATION = {
    "google/electra-small-generator": {"do_lower_case": True},
    "google/electra-base-generator": {"do_lower_case": True},
    "google/electra-large-generator": {"do_lower_case": True},
    "google/electra-small-discriminator": {"do_lower_case": True},
    "google/electra-base-discriminator": {"do_lower_case": True},
    "google/electra-large-discriminator": {"do_lower_case": True},
}

# 从 transformers.models.bert.tokenization_bert.load_vocab 函数复制过来的加载词汇表的函数定义
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    # 创建一个有序字典用于存储词汇表
    vocab = collections.OrderedDict()
    # 使用 UTF-8 编码打开词汇文件
    with open(vocab_file, "r", encoding="utf-8") as reader:
        # 逐行读取词汇文件内容
        tokens = reader.readlines()
    # 对 tokens 列表进行遍历，同时获取索引和每个元素 token
    for index, token in enumerate(tokens):
        # 去除 token 字符串末尾的换行符 "\n"
        token = token.rstrip("\n")
        # 将 token 添加到 vocab 字典中，键为 token，值为 index
        vocab[token] = index
    
    # 返回填充完毕的 vocab 字典作为结果
    return vocab
# 从transformers.models.bert.tokenization_bert.whitespace_tokenize复制而来，定义了一个函数用于基本的空白符号分割和清理文本。
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # 清除文本两侧的空白符号
    text = text.strip()
    # 如果清理后的文本为空，则返回空列表
    if not text:
        return []
    # 使用空白符号分割文本，得到token列表
    tokens = text.split()
    # 返回分割后的token列表
    return tokens


# 从transformers.models.bert.tokenization_bert.BertTokenizer复制而来，修改为支持Electra，构建Electra分词器。
class ElectraTokenizer(PreTrainedTokenizer):
    r"""
    Construct a Electra tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # 定义一个类，用于处理预训练模型的词汇表和相关配置信息
    
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    
    # 初始化方法，用于创建一个新的Tokenizer实例
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
        """
        Args:
            vocab_file (`str`):
                包含词汇表的文件。
            do_lower_case (`bool`, *optional*, defaults to `True`):
                是否在进行分词时将输入转换为小写。
            do_basic_tokenize (`bool`, *optional*, defaults to `True`):
                是否在WordPiece分词前进行基本分词。
            never_split (`Iterable`, *optional*):
                在分词过程中不应拆分的标记集合。仅在 `do_basic_tokenize=True` 时有效。
            unk_token (`str`, *optional*, defaults to `"[UNK]"`):
                未知标记。当输入中的标记不在词汇表中时，将其替换为此标记。
            sep_token (`str`, *optional*, defaults to `"[SEP]"`):
                分隔符标记，在构建多个序列的序列时使用，例如序列分类或问答问题时使用。也用作构建带有特殊标记的序列的最后一个标记。
            pad_token (`str`, *optional*, defaults to `"[PAD]"`):
                用于填充的标记，例如在批处理不同长度的序列时使用。
            cls_token (`str`, *optional*, defaults to `"[CLS]"`):
                分类器标记，在进行序列分类时使用（整个序列的分类而不是每个标记的分类）。它是构建带有特殊标记的序列的第一个标记。
            mask_token (`str`, *optional*, defaults to `"[MASK]"`):
                用于屏蔽值的标记。这是在进行遮蔽语言建模训练时使用的标记。模型将尝试预测此标记。
            tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
                是否对中文字符进行分词。
    
                对于日语可能需要禁用此选项（参见此问题: https://github.com/huggingface/transformers/issues/328）。
            strip_accents (`bool`, *optional*):
                是否删除所有重音符号。如果未指定此选项，则将根据 `lowercase` 的值来确定（与原始Electra一样）。
        """
    ):
        # 如果给定的词汇文件不存在，抛出数值错误异常，提示找不到指定路径的词汇文件
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = ElectraTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表文件到 self.vocab 中
        self.vocab = load_vocab(vocab_file)
        # 使用 collections.OrderedDict 创建 ids 到 tokens 的有序映射
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 设置是否进行基本标记化处理的标志
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要进行基本标记化处理
        if do_basic_tokenize:
            # 初始化 BasicTokenizer 对象，设置参数
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        # 初始化 WordpieceTokenizer 对象，传入词汇表和未知标记
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的初始化方法，设置参数
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
        # 返回基本标记化器的小写标志
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        # 返回词汇表的大小（词汇数）
        return len(self.vocab)

    def get_vocab(self):
        # 返回词汇表和已添加标记编码器的结合
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, split_special_tokens=False):
        # 分词结果列表
        split_tokens = []
        # 如果需要进行基本标记化处理
        if self.do_basic_tokenize:
            # 使用 BasicTokenizer 对象进行标记化处理
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果标记在不分割集合中
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 使用 WordpieceTokenizer 对标记进一步分词
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 使用 WordpieceTokenizer 对整个文本进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 根据词汇表将标记转换为其对应的 id
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 根据 id 将其转换为对应的标记
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将标记序列合并为单个字符串，去除 "##" 符号
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A Electra sequence has the following format:

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
        # If token_ids_1 is not provided, return the single-sequence format
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        
        # For pair of sequences, concatenate tokens with special tokens separating them
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
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
        # If the tokens already have special tokens, delegate to the superclass method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Calculate the mask for tokens with special tokens added
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs tensor from token id tensors. `0` for the first sentence tokens, `1` for the second sentence
        tokens.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs according to the sequences provided.
        """
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A Electra sequence
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
        # Define the separator and classification tokens
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        # If only one sequence is provided (token_ids_1 is None), return a mask with all zeros
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # If both sequences are provided, return a mask with zeros for the first sequence and ones for the second
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        
        # Determine the vocabulary file path based on the provided save_directory
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        
        # Write the vocabulary to the specified file
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # Check if vocabulary indices are consecutive and warn if not
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        
        # Return the path to the saved vocabulary file
        return (vocab_file,)
# Copied from transformers.models.bert.tokenization_bert.BasicTokenizer
# 基本分词器类，用于执行基本的分词操作（如分割标点符号、转换为小写等）。
class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
            是否在分词时将输入转换为小写。

        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
            在分词过程中永远不会被分开的标记集合。仅在 `do_basic_tokenize=True` 时有效。

        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.
            是否对中文字符进行分词。建议对日文关闭此选项（参见这个问题链接）。

        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
            是否去除所有重音符号。如果未指定此选项，则由 `lowercase` 的值来确定（与原始的BERT一致）。

        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            In some instances we want to skip the basic punctuation splitting so that later tokenization can capture
            the full context of the words, such as contractions.
            在某些情况下，我们希望跳过基本的标点符号分割，以便后续的分词可以捕获单词的完整上下文，比如缩写词。
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果 `never_split` 为 `None`，则初始化为一个空列表
        if never_split is None:
            never_split = []
        # 设置是否将输入转换为小写
        self.do_lower_case = do_lower_case
        # 将 `never_split` 转换为集合，用于存储永不分割的标记集合
        self.never_split = set(never_split)
        # 设置是否对中文字符进行分词
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 设置是否去除所有重音符号，如果未指定则根据 `lowercase` 的值确定
        self.strip_accents = strip_accents
        # 设置是否执行基本的标点符号分割
        self.do_split_on_punc = do_split_on_punc
    def tokenize`
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 使用 union() 方法将 self.never_split 和给定的 never_split 合并成一个新的集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，去除不必要的字符或格式
        text = self._clean_text(text)

        # 2018 年 11 月 1 日添加此功能，用于多语言和中文模型。现在也适用于英语模型，但这并不重要，
        # 因为英语模型未经过任何中文数据的训练，通常在其中也没有中文数据（英语维基百科中有些中文词汇，
        # 因此词汇表中有一些中文字符）。
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 使用 NFC 标准规范化文本中的 Unicode 字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用 whitespace_tokenize() 函数将文本分割成原始的单词列表
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    # 如果 do_lower_case 为 True，则将 token 转换为小写
                    token = token.lower()
                    # 如果 strip_accents 不为 False，则运行 _run_strip_accents() 方法去除重音符号
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    # 如果 strip_accents 为 True，则运行 _run_strip_accents() 方法去除重音符号
                    token = self._run_strip_accents(token)
            # 将 token 拆分并扩展到 split_tokens 列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 使用 whitespace_tokenize() 函数将 split_tokens 列表重新组合成字符串，并再次进行分词
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        # 返回最终的 token 列表
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 使用 NFD 标准规范化文本中的 Unicode 字符
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            # 获取字符的 Unicode 类别
            cat = unicodedata.category(char)
            # 如果 Unicode 类别为 Mn（Nonspacing_Mark），则跳过该字符
            if cat == "Mn":
                continue
            # 否则将字符添加到输出列表中
            output.append(char)
        # 将输出列表中的字符组合成字符串并返回
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要根据标点符号分割文本，或者文本在never_split列表中，则直接返回包含整个文本的列表
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
            # 如果当前字符是标点符号，则将其作为单独的列表项添加到输出列表中，并设置开始一个新单词的标志为True
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，检查是否应该开始一个新单词
                if start_new_word:
                    output.append([])
                start_new_word = False
                # 将当前字符添加到最后一个列表项中
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
            # 如果字符是CJK字符，将其前后加上空格并添加到输出列表中
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                # 如果不是CJK字符，直接添加到输出列表中
                output.append(char)
        # 将列表转换为字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 判断字符编码点是否在CJK Unicode块内
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
        # 遍历文本中的每个字符
        for char in text:
            cp = ord(char)
            # 如果字符是空白字符或控制字符，将其替换为空格
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将列表转换为字符串并返回
        return "".join(output)
# Copied from transformers.models.bert.tokenization_bert.WordpieceTokenizer
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化 WordpieceTokenizer 类，设置词汇表、未知 token 和最大输入字符数
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
        # 初始化输出 token 列表
        output_tokens = []
        # 对文本进行分词，使用空白字符进行分隔
        for token in whitespace_tokenize(text):
            chars = list(token)
            # 如果 token 长度超过最大输入字符数，则添加未知 token
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            # 使用贪婪最长匹配算法进行 tokenization
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    # 对于非首字符的 substr，添加 '##' 前缀
                    if start > 0:
                        substr = "##" + substr
                    # 如果 substr 在词汇表中，则作为当前 token
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                # 如果未找到合适的 token，则标记为 bad token
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            # 如果存在 bad token，则添加未知 token，否则添加所有子 token
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
```