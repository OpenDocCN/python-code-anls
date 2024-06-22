# `.\models\convbert\tokenization_convbert.py`

```py
# 设置文件编码格式为 UTF-8
# 版权声明
# 根据 Apache 许可 2.0 版本，禁止未经许可使用此文件
# 可以在以下链接获取许可副本
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "YituTech/conv-bert-base": "https://huggingface.co/YituTech/conv-bert-base/resolve/main/vocab.txt",
        "YituTech/conv-bert-medium-small": (
            "https://huggingface.co/YituTech/conv-bert-medium-small/resolve/main/vocab.txt"
        ),
        "YituTech/conv-bert-small": "https://huggingface.co/YituTech/conv-bert-small/resolve/main/vocab.txt",
    }
}

# 预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "YituTech/conv-bert-base": 512,
    "YituTech/conv-bert-medium-small": 512,
    "YituTech/conv-bert-small": 512,
}

# 预训练初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "YituTech/conv-bert-base": {"do_lower_case": True},
    "YituTech/conv-bert-medium-small": {"do_lower_case": True},
    "YituTech/conv-bert-small": {"do_lower_case": True},
}

# 从文件读取词汇表
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

# 基本的空白字符分词函数
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

# ConvBERT 分词器类，继承自 PreTrainedTokenizer
class ConvBertTokenizer(PreTrainedTokenizer):
    r"""
    Construct a ConvBERT tokenizer. Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    # 参数说明：
    # 1. vocab_file：包含词汇表的文件
    # 2. do_lower_case：是否在标记化时将输入转换为小写，默认为True
    # 3. do_basic_tokenize：在WordPiece之前是否进行基本标记化，默认为True
    # 4. never_split：在标记化过程中永远不会分割的标记集合，仅在do_basic_tokenize=True时有效
    # 5. unk_token：未知标记。不在词汇表中的标记无法转换为ID，并将设置为此标记
    # 6. sep_token：分隔符标记，用于从多个序列构建序列时使用，例如用于序列分类或用于文本和问题用于问答。还用作具有特殊标记的序列的最后一个标记。
    # 7. pad_token：用于填充的标记，例如在批处理不同长度的序列时使用
    # 8. cls_token：分类器标记，在进行序列分类时使用（对整个序列进行分类，而不是对每个标记进行分类）。它是使用特殊标记构建的序列的第一个标记。
    # 9. mask_token：用于掩盖值的标记。这是在使用掩盖语言建模训练模型时使用的标记。这是模型将尝试预测的标记。
    # 10. tokenize_chinese_chars：是否对中文字符进行标记化，默认为True

    vocab_files_names = VOCAB_FILES_NAMES  # 词汇表文件的命名列表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练的词汇文件映射
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION  # 预训练的初始化配置
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 预训练的最大模型输入大小

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
    # 定义一个类，用于处理Bert模型的tokenizer
    ):
        # 如果给定的vocab_file不是一个文件路径，则抛出数值错误
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 从vocab_file中加载词汇表
        self.vocab = load_vocab(vocab_file)
        # 创建一个有序字典，将词汇表中的ids和tokens对应起来
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 根据参数do_basic_tokenize是否为True，决定是否需要进行基本的分词处理
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要进行基本分词处理，则初始化BasicTokenizer对象
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )

        # 初始化WordpieceTokenizer对象，传入词汇表和未知标记
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的初始化方法，传入各种参数
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

    # 定义do_lower_case为属性，返回basic_tokenizer对象的do_lower_case属性
    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    # 定义vocab_size为属性，返回词汇表的长度
    @property
    def vocab_size(self):
        return len(self.vocab)

    # 获取词汇表，包括已添加的特殊标记
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # 对输入的文本进行分词处理
    def _tokenize(self, text, split_special_tokens=False):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果token在never_split中，则直接添加到split_tokens
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                # 否则将token传给wordpiece_tokenizer进行分词处理，再添加到split_tokens中
                else:
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        # 如果不需要进行基本分词处理，则直接使用wordpiece_tokenizer进行分词处理
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    # 将token转换为id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 将id转换为token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    # 将token序列转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # 构建包含特殊标记的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A ConvBERT sequence has the following format:

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

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 定义一个方法用于创建序列对分类任务中的掩码
    def create_sequence_pair_mask(self, token_ids_0:List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A ConvBERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```py

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # 定义分隔符以及类别标记的列表
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # 如果第二个序列为空，只返回掩码的第一部分（全是0）
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 返回根据两个序列生成的掩码
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # 保存词汇表文件
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引
        index = 0
        # 判断保存目录是否存在
        if os.path.isdir(save_directory):
            # 拼接文件路径
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开文件，遍历词汇表并写入文件
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    # 添加警告日志
                    index = token_index
                # 写入词汇表
                writer.write(token + "\n")
                index += 1
        # 返回保存的文件路径元组
        return (vocab_file,)
# 定义 BasicTokenizer 类，用于执行基本的分词操作（拆分标点符号、转换为小写等）
class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在分词时将输入转换为小写。
        never_split (`Iterable`, *optional*):
            在分词时永远不会拆分的 token 集合。仅在 `do_basic_tokenize=True` 时有效。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否分词中文字符。
            
            对于日语，这可能应该被禁用（参见这个 [issue](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否去除所有的重音符号。如果未指定此选项，则会根据 `lowercase` 的值确定（与原始 BERT 一致）。
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            在某些情况下，我们希望跳过基本的标点符号拆分，以便后续的分词可以捕获单词的完整上下文，例如缩写词。

    """

    # 初始化 BasicTokenizer 类
    def __init__(
        self,
        do_lower_case=True,  # 是否将输入转换为小写，默认为 True
        never_split=None,  # 永远不会拆分的 token 集合，默认为 None
        tokenize_chinese_chars=True,  # 是否分词中文字符，默认为 True
        strip_accents=None,  # 是否去除所有的重音符号，默认为 None
        do_split_on_punc=True,  # 是否在分词时拆分标点符号，默认为 True
    ):
        # 如果 never_split 未指定，则设置为一个空列表
        if never_split is None:
            never_split = []
        # 将输入参数赋值给对象的属性
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)  # 永远不会拆分的 token 集合
        self.tokenize_chinese_chars = tokenize_chinese_chars  # 是否分词中文字符
        self.strip_accents = strip_accents  # 是否去除所有的重音符号
        self.do_split_on_punc = do_split_on_punc  # 是否在分词时拆分标点符号
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 将给定的不可分割token列表与默认的不可分割token列表合并，返回新的set
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，去除不必要的字符
        text = self._clean_text(text)

        # 这段代码是在2018年11月1日添加的，用于多语言和中文模型。现在也应用于英语模型，但这没关系，因为英语模型没有在任何中文数据上训练，并且通常没有任何中文数据（词汇表中有中文字符，因为英语维基百科中有一些中文词汇）。
        if self.tokenize_chinese_chars:
            # 将中文字符拆分为单个字符
            text = self._tokenize_chinese_chars(text)
        # 将文本标准化为统一的unicode编码形式
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 将标准化后的文本按空格拆分为词汇
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    # 将token转换为小写
                    token = token.lower()
                    if self.strip_accents is not False:
                        # 去除token中的重音符号
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    # 去除token中的重音符号
                    token = self._run_strip_accents(token)
            # 将拆分后的token添加到列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将拆分后的token按空格拼接为文本，并按空格拆分为词汇
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 将文本标准化为NFD形式
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            # 获取unicode字符的类别
            cat = unicodedata.category(char)
            # 如果字符类别为Mn则跳过，不将其添加到输出中
            if cat == "Mn":
                continue
            # 将字符添加到输出中
            output.append(char)
        return "".join(output)
    # 在给定的文本上拆分标点符号
    def _run_split_on_punc(self, text, never_split=None):
        # 如果不需要拆分标点，或者该文本在不需要拆分的列表中，则返回原文本
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
            # 如果是标点符号
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        # 将拆分后的字符列表转换为字符串列表并返回
        return ["".join(x) for x in output]

    # 在中文字符周围添加空格
    def _tokenize_chinese_chars(self, text):
        output = []
        # 遍历文本中的字符
        for char in text:
            cp = ord(char)
            # 如果是中文字符，则在其前后添加空格
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        # 返回处理后的文本
        return "".join(output)

    # 判断给定的字符是否是中文字符
    def _is_chinese_char(self, cp):
        # 检查代码点是否属于CJK字符范围
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
        # 如果不是中文字符，则返回False
        return False

    # 在文本上执行无效字符的移除和空格的清理
    def _clean_text(self, text):
        output = []
        # 遍历文本中的字符
        for char in text:
            cp = ord(char)
            # 如果是无效字符或控制字符，则跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果是空白字符，则替换为单个空格
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 返回清理后的文本
        return "".join(output)
# 从transformers.models.bert.tokenization_bert.WordpieceTokenizer复制而来的类
class WordpieceTokenizer(object):
    """运行WordPiece标记化。"""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化WordpieceTokenizer对象
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        将文本分词为其词片段。这使用贪婪的最长匹配优先算法，使用给定的词汇表执行标记化。

        例如， `input = "unaffable"` 将返回输出 `["un", "##aff", "##able"]`。

        参数:
            text: 一个单个标记或空格分隔的标记。这应该已经通过*BasicTokenizer*。

        返回:
            一个词片段标记列表。
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            # 将标记按空白分隔
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                # 如果标记长度超过最大输入字符数，将unk_token加入输出列表
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