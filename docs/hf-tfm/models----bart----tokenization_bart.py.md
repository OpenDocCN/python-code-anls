# `.\models\bart\tokenization_bart.py`

```py
# 定义一个名为 `bytes_to_unicode` 的函数，并且使用 `@lru_cache()` 装饰器进行缓存，使其结果可以被缓存以提高性能
@lru_cache()
def bytes_to_unicode():
    """
    返回 utf-8 字节列表及其与 Unicode 字符串的映射。特别地，避免将空格/控制字符映射到 BPE 代码中会出错的情况。

    可逆的 BPE（Byte Pair Encoding）代码适用于 Unicode 字符串。这意味着你的词汇表中需要有大量的 Unicode 字符。
    """
    # 返回具体的映射关系和描述性文本，这些信息在 BPE 算法中是必需的
    return [
        '\u2581' + chr(i) for i in range(0, 128)
    ] + [chr(i) for i in range(128, 256)]
    # 定义一个函数，返回一个字典，用于 utf-8 字节和 Unicode 字符串之间的映射
    def make_utf8_to_unicode_lookup():
        # 创建一个包含可打印 ASCII 字符、特殊符号和扩展 Latin-1 范围的字节列表
        bs = (
            list(range(ord("!"), ord("~") + 1)) + 
            list(range(ord("¡"), ord("¬") + 1)) + 
            list(range(ord("®"), ord("ÿ") + 1))
        )
        # 复制 bs 列表到 cs 列表
        cs = bs[:]
        # 初始化计数器 n
        n = 0
        # 遍历所有可能的 8 位字节值
        for b in range(2**8):
            # 如果 b 不在 bs 中，则将 b 添加到 bs 和对应的扩展编码添加到 cs 中
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        # 将 cs 中的整数转换为对应的 Unicode 字符串
        cs = [chr(n) for n in cs]
        # 返回 bs 和 cs 对应的字典
        return dict(zip(bs, cs))
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    # 初始化一个空集合用于存放符号对
    pairs = set()
    # 获取单词的第一个字符作为前一个字符
    prev_char = word[0]
    # 遍历单词中的每个字符，形成符号对并添加到集合中
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    # 返回包含符号对的集合
    return pairs


class BartTokenizer(PreTrainedTokenizer):
    """
    Constructs a BART tokenizer, which is smilar to the ROBERTa tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```
    >>> from transformers import BartTokenizer

    >>> tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # 构造函数，用于初始化一个 BART 分词器对象
    def __init__(self, vocab_file, merges_file, errors='replace', special_tokens_dict=None, max_len=None, **kwargs):
        # 调用父类构造函数初始化 BART 分词器
        super().__init__(vocab_file, merges_file, errors=errors, special_tokens_dict=special_tokens_dict, **kwargs)
        # 设置最大长度属性
        self.max_len = max_len

    # 实现从预训练模型加载 BART 分词器的类方法
    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        # 调用父类的类方法加载预训练模型
        return super().from_pretrained(*inputs, **kwargs)

    # 重写父类方法，根据文本生成输入 ID 列表
    def __call__(self, text, **kwargs):
        # 调用父类方法生成输入 ID 列表
        return super().__call__(text, **kwargs)
    # 定义一个函数的参数列表，用于初始化一个类的实例或调用函数时传递参数。
    Args:
        vocab_file (`str`):
            词汇表文件的路径。
        merges_file (`str`):
            合并文件的路径。
        errors (`str`, *optional*, defaults to `"replace"`):
            当解码字节为 UTF-8 时的错误处理方式。详见 [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode)。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            预训练过程中用作序列开头的特殊 token。可以用作序列分类器的 token。
            <Tip>
            在构建序列时使用特殊 token 时，实际用于序列开头的 token 是 `cls_token`。
            </Tip>
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            序列结尾的特殊 token。
            <Tip>
            在构建序列时使用特殊 token 时，实际用于序列结尾的 token 是 `sep_token`。
            </Tip>
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            分隔符 token，在构建多个序列的合并序列时使用，例如序列分类或问答任务中的问题和文本序列。也作为使用特殊 token 构建序列时的最后一个 token。
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            分类器 token，在序列分类任务中使用（整个序列的分类而不是每个 token 的分类）。在使用特殊 token 构建序列时，它是序列的第一个 token。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知 token，如果词汇表中不存在某个 token，则将其替换为该 token。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            用于填充的 token，在批处理不同长度的序列时使用。
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            用于掩码值的 token，在进行掩码语言建模训练时使用，模型将尝试预测该 token。
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            是否在输入的开头添加一个空格，这允许将第一个词视为其他词一样处理。（BART tokenizer 通过前导空格检测单词的开头）。
    """
    
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        **kwargs,
    ):
        # 如果初始的特殊标记是字符串类型，则使用AddedToken进行处理，保持左右空格的原样
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # 处理mask_token，使其像普通单词一样，包括前面的空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 使用utf-8编码打开vocab_file文件，加载其中的内容到self.encoder字典中
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)

        # 创建self.decoder字典，将self.encoder的键值对反转，用于从索引到单词的解码
        self.decoder = {v: k for k, v in self.encoder.items()}

        # 设定解码中遇到错误时的处理方式
        self.errors = errors  # how to handle errors in decoding

        # 使用bytes_to_unicode函数生成字节编码到Unicode的映射表
        self.byte_encoder = bytes_to_unicode()

        # 创建self.byte_decoder字典，将self.byte_encoder的键值对反转，用于从Unicode到字节编码的解码
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # 使用utf-8编码打开merges_file文件，读取内容并按行分割，去掉首尾空行后将其转换为元组列表bpe_merges
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]

        # 创建self.bpe_ranks字典，将bpe_merges列表转换为字典，键为元组，值为其在列表中的索引
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        # 初始化缓存字典为空字典
        self.cache = {}

        # 设定是否在前缀空格之前添加特殊标记的选项
        self.add_prefix_space = add_prefix_space

        # 编译正则表达式模式pat，用于匹配字符串中的各种形式的标点、字母和数字
        # 应该添加re.IGNORECASE标志，以便处理大写形式的缩写
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 调用父类的构造方法，传递初始化参数
        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回self.encoder字典的长度，即词汇表的大小
        return len(self.encoder)

    def get_vocab(self):
        # 返回包含self.encoder和self.added_tokens_encoder所有键值对的字典
        return dict(self.encoder, **self.added_tokens_encoder)
    def _tokenize(self, text):
        """Tokenize a string."""
        # 初始化空列表，用于存储BPE处理后的token
        bpe_tokens = []
        # 使用正则表达式找到所有匹配self.pat的token，并进行处理
        for token in re.findall(self.pat, text):
            # 将token按utf-8编码，并映射到unicode字符串，避免BPE的控制token（在我们的情况下是空格）
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            # 将BPE处理后的token按空格分割，并加入到bpe_tokens列表中
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        # 返回处理后的token列表
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用self.encoder获取token对应的id，若token不存在，则使用self.unk_token的id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用self.decoder获取index对应的token
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将tokens列表连接成一个字符串
        text = "".join(tokens)
        # 将字符串按字节解码成utf-8格式，并处理可能的错误
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        # 返回解码后的文本字符串
        return text
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建词汇表文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构建合并文件路径
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 写入词汇表到文件中
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 写入合并数据到文件中
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            # 遍历 BPE rank 数据并写入文件
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # 记录警告，指出 BPE 合并索引不连续的情况
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回保存的词汇表文件路径和合并文件路径
        return vocab_file, merge_file

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BART sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 如果没有第二个序列，则返回添加特殊 token 后的单个序列
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        
        # 构建两个序列合并后的输入序列
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
        """
        Retrieve sequence ids where special tokens are added.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional list of IDs of the second sequence.
            already_has_special_tokens (`bool`, *optional*):
                Whether the sequences already contain special tokens.

        Returns:
            `List[int]`: A list of binary indicators where 1 indicates a special token and 0 indicates a regular token.
        """
        # 计算特殊 token 的掩码
        special_tokens_mask = [1] * len(token_ids_0)
        if token_ids_1 is not None:
            special_tokens_mask += [1] * len(token_ids_1)
        return special_tokens_mask
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
        # If the token list already has special tokens, delegate to the superclass method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If there's only one token list provided, return a mask with special tokens added at both ends
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        # If two token lists are provided, return a mask with special tokens at both ends of each sequence
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. BART does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # Initialize special tokens for separator and class, but BART does not use token type ids
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If there's only one sequence, return a list of zeros with the length of cls + token_ids_0 + sep
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # If two sequences are provided, return a list of zeros with the length of cls + token_ids_0 + sep + sep + token_ids_1 + sep
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        """
        Prepares text for tokenization by optionally adding a prefix space based on conditions.

        Args:
            text (str): The input text to be tokenized.
            is_split_into_words (bool, optional): Whether the text is already split into words.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the modified text and remaining keyword arguments.
        """
        # Check if prefix space should be added based on conditions
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)
```