# `.\models\deprecated\retribert\tokenization_retribert.py`

```py
# 指定编码格式为 UTF-8
# 版权声明，引用的代码库遵循 Apache License, Version 2.0
# 导入 collections 模块，用于构建数据结构
# 导入 os 模块，提供了一些与操作系统交互的功能
# 导入 unicodedata 模块，用于对 Unicode 字符进行数据库查询
# 导入 typing 模块中的 List、Optional、Tuple 类型
# 从 tokenization_utils 模块中导入 PreTrainedTokenizer 类，用于构建 tokenizer
# 从 tokenization_utils 模块中导入 _is_control、_is_punctuation、_is_whitespace 函数
# 从 utils 模块中导入 logging 函数

# 获取 logger 对象，用于记录日志信息
logger = logging.get_logger(__name__)

# 定义一个字典，指定 VOCAB 文件的名称为 "vocab.txt"
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 定义一个字典，映射预训练模型到对应的 VOCAB 文件路径
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "yjernite/retribert-base-uncased": (
            "https://huggingface.co/yjernite/retribert-base-uncased/resolve/main/vocab.txt"
        ),
    }
}

# 定义一个字典，映射预训练模型到对应的位置编码大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "yjernite/retribert-base-uncased": 512,
}

# 定义一个字典，指定预训练模型的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "yjernite/retribert-base-uncased": {"do_lower_case": True},
}


# 从 transformers.models.bert.tokenization_bert.load_vocab 复制过来的函数
# 加载给定的词汇表文件到一个有序字典中
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


# 从 transformers.models.bert.tokenization_bert.whitespace_tokenize 复制过来的函数
# 对输入的文本进行基本的空白字符清理和分割
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


# 定义 RetriBertTokenizer 类，继承自 PreTrainedTokenizer
class RetriBertTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a RetriBERT tokenizer.

    [`RetriBertTokenizer`] is identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation splitting
    and wordpiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer
    to: this superclass for more information regarding those methods.
    """
    # 词汇文件的名称列表，用于指定预训练模型使用的词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型使用的预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型的最大输入大小，指定了每个模型的最大输入序列长度
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 预训练模型的初始化配置，包含了模型初始化时的各种参数设置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 模型输入的名称列表，定义了模型输入时所使用的名称
    model_input_names = ["input_ids", "attention_mask"]

    # 以下代码段为从transformers.models.bert.tokenization_bert.BertTokenizer.__init__中复制而来
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
        # 检查词汇文件是否存在，如果不存在则抛出数值错误异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表
        self.vocab = load_vocab(vocab_file)
        # 构建一个从 id 到 token 的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 设置是否进行基本的分词操作
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要进行基本分词，则初始化 BasicTokenizer 对象
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        # 初始化 WordpieceTokenizer 对象
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

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

    @property
    # 从 transformers.models.bert.tokenization_bert.BertTokenizer.do_lower_case 复制而来
    def do_lower_case(self):
        # 返回基本分词器的小写标志
        return self.basic_tokenizer.do_lower_case

    @property
    # 从 transformers.models.bert.tokenization_bert.BertTokenizer.vocab_size 复制而来
    def vocab_size(self):
        # 返回词汇表的大小（词汇量）
        return len(self.vocab)

    # 从 transformers.models.bert.tokenization_bert.BertTokenizer.get_vocab 复制而来
    def get_vocab(self):
        # 返回词汇表和添加的特殊 token 编码器的组合字典
        return dict(self.vocab, **self.added_tokens_encoder)

    # 从 transformers.models.bert.tokenization_bert.BertTokenizer._tokenize 复制而来
    def _tokenize(self, text, split_special_tokens=False):
        # 初始化分词后的 tokens 列表
        split_tokens = []
        # 如果需要进行基本分词
        if self.do_basic_tokenize:
            # 使用基本分词器对文本进行分词
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果 token 在不分割集合中
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 使用 WordpieceTokenizer 对 token 进行进一步分词
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 直接使用 WordpieceTokenizer 对文本进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        # 返回分词后的 tokens 列表
        return split_tokens
    # 从词汇表中将 token 转换为对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 从词汇表中将 id 转换为对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    # 将一系列的 token 转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # 构建包含特殊 token 的模型输入
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
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # 获取包含特殊 token 的 mask（掩码）
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. It adds the special tokens according
        to the BERT-like model requirements.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs corresponding to the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for BERT.

        Returns:
            `List[int]`: List of `1` and `0`, with `1` indicating a special token, `0` indicating a regular token.
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

        # Check if the token list already has special tokens added
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If token_ids_1 is provided, construct a mask for sequence pairs
        if token_ids_1 is not None:
            # Return a list indicating positions of special tokens and sequence tokens
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # Otherwise, construct a mask for a single sequence
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
        # Define special tokens
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        # If token_ids_1 is None, return a mask for a single sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # Otherwise, return a mask for sequence pairs
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.save_vocabulary
    # 保存词汇表到指定目录中的文件，返回保存的文件路径元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引
        index = 0
        # 检查保存目录是否存在，若存在则拼接文件路径，否则直接使用指定路径
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 使用 utf-8 编码打开文件准备写入
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表 self.vocab，按照词汇索引排序
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 检查索引是否连续，若不连续则发出警告
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 将词汇写入文件，每个词汇占一行
                writer.write(token + "\n")
                index += 1
        # 返回保存的文件路径元组
        return (vocab_file,)
# 从transformers.models.bert.tokenization_bert.BasicTokenizer复制而来的类定义
class BasicTokenizer(object):
    """
    构造一个BasicTokenizer对象，用于运行基本的分词操作（如分割标点符号、转换为小写等）。

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在分词时将输入文本转换为小写。
        never_split (`Iterable`, *optional*):
            在分词时永远不会被分割的标记集合。仅在`do_basic_tokenize=True`时生效。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否对中文字符进行分词。

            对于日语，这应该被禁用（参见这个
            [问题](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否去除所有的重音符号。如果未指定此选项，则由`lowercase`的值决定（与原始BERT相同）。
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            在某些情况下，我们希望跳过基本的标点分割，以便后续的分词可以捕捉到完整的词语上下文，比如缩写词。

    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果`never_split`未提供，则设置为一个空列表
        if never_split is None:
            never_split = []
        # 初始化对象属性
        self.do_lower_case = do_lower_case  # 是否转换为小写
        self.never_split = set(never_split)  # 永不分割的标记集合，转换为集合类型
        self.tokenize_chinese_chars = tokenize_chinese_chars  # 是否分词中文字符
        self.strip_accents = strip_accents  # 是否去除重音符号
        self.do_split_on_punc = do_split_on_punc  # 是否在标点符号处分割
    # 对文本进行基本的分词操作。用于子词分词，请参见WordPieceTokenizer。
    # 
    # Args:
    #     never_split (`List[str]`, *optional*): 用于向后兼容的参数。现在直接在基类级别实现
    #         （参见`PreTrainedTokenizer.tokenize`）。不要分割的标记列表。
    def tokenize(self, text, never_split=None):
        # 如果给定了never_split参数，则将其与对象属性self.never_split合并，形成新的集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，去除不必要的字符
        text = self._clean_text(text)

        # 2018年11月1日添加，用于多语言和中文模型。现在也适用于英语模型，但这并不重要，
        # 因为英语模型未在任何中文数据上训练，并且通常不包含任何中文数据
        # （英文维基百科中包含一些中文单词，因此词汇表中会包含中文字符）。
        if self.tokenize_chinese_chars:
            # 对中文字符进行分词处理
            text = self._tokenize_chinese_chars(text)
        # 将Unicode文本进行规范化处理，防止不同的Unicode编码造成字符被视为不同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空白字符分词
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        # 遍历每个token
        for token in orig_tokens:
            # 如果token不在never_split中，则考虑大小写问题和重音符号处理
            if token not in never_split:
                if self.do_lower_case:
                    # 如果设置为小写处理，则将token转换为小写
                    token = token.lower()
                    # 如果需要去除重音符号，则执行去除操作
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    # 如果需要去除重音符号，则执行去除操作
                    token = self._run_strip_accents(token)
            # 将分割后的token添加到列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 使用空白字符再次分词，将所有分割后的token重新组合成字符串列表
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        # 返回最终的token列表
        return output_tokens

    # 从文本中去除重音符号
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 将文本标准化为NFD格式，以便处理重音符号
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的Unicode分类
            cat = unicodedata.category(char)
            # 如果字符属于"Mark, Nonspacing"（重音符号），则跳过该字符
            if cat == "Mn":
                continue
            # 否则将字符添加到输出列表中
            output.append(char)
        # 将输出列表中的字符重新组合成字符串，并返回
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要在标点处分割或者指定了不分割的文本，则直接返回包含原始文本的列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号，则将其作为单独的列表项加入输出列表，并标记为开始新单词
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，根据是否开始新单词决定是追加到当前列表项还是新建列表项
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        # 将列表中的列表项连接为字符串，并返回分割后的文本列表
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符是CJK字符，则在其前后添加空格，并加入输出列表；否则直接加入输出列表
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        # 将列表转换为字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 判断字符编码是否在CJK统一表意文字范围内
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
            # 如果字符编码为0或0xFFFD（无效字符），或者是控制字符，则跳过当前字符
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果是空白字符，则替换为单个空格；否则保留字符
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将列表转换为字符串并返回
        return "".join(output)
# Copied from transformers.models.bert.tokenization_bert.WordpieceTokenizer
# WordpieceTokenizer 类，用于运行 WordPiece 分词算法

class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""
    # 初始化方法，设置词汇表、未知标记和单词最大字符数
    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab  # 词汇表，包含所有的词和子词
        self.unk_token = unk_token  # 未知标记，用于未能识别的词或子词
        self.max_input_chars_per_word = max_input_chars_per_word  # 单词的最大字符数限制，默认为100

    # 对文本进行 WordPiece 分词
    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` will return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """
        # 初始化输出 token 列表
        output_tokens = []
        # 对文本进行空白字符分割，获取每一个 token
        for token in whitespace_tokenize(text):
            chars = list(token)
            # 如果 token 的字符数超过最大字符数限制，则将未知标记加入输出 tokens
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            # 使用贪婪的最长匹配算法进行分词
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                # 从当前位置向前递减，生成子字符串并加上前缀 "##"，检查其是否在词汇表中
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                # 如果找不到合适的子字符串，则将 is_bad 标记设为 True
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            # 如果 is_bad 为 True，则将未知标记加入输出 tokens；否则将子 tokens 加入输出 tokens
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        
        # 返回最终的 wordpiece tokens 列表
        return output_tokens
```