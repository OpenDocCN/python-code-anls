# `.\transformers\models\roberta\tokenization_roberta.py`

```
# 定义函数 bytes_to_unicode，使用 lru_cache 装饰器缓存结果
@lru_cache()
# 函数返回 utf-8 字节的列表和将字节映射到 Unicode 字符串的映射
def bytes_to_unicode():
    """
    # 返回 utf-8 字节列表和字节映射到 Unicode 字符串的映射
    # 避免将空白字符/控制字符映射到 Unicode，因为 BPE（字节对编码）代码无法处理它们
    """
    # 生成一个用于避免未知标记（UNK）的字节到Unicode字符串的查找表
    """
        if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
        decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
        tables between utf-8 bytes and unicode strings.
        """
        # 定义一个包含可见ASCII字符、一些特殊字符和扩展ASCII字符的字节列表
        bs = (
            list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        )
        # 复制bs列表给cs
        cs = bs[:]
        # 初始化n为0
        n = 0
        # 遍历0到255的所有字节
        for b in range(2**8):
            # 如果当前字节不在bs中
            if b not in bs:
                # 将当前字节添加到bs列表中
                bs.append(b)
                # 将对应的Unicode码添加到cs列表中
                cs.append(2**8 + n)
                # 更新n
                n += 1
        # 将cs列表中的Unicode码转换为对应的Unicode字符
        cs = [chr(n) for n in cs]
        # 返回字节到Unicode字符串的映射字典
        return dict(zip(bs, cs))
# 返回单词中的符号对集合
# 单词表示为符号的元组（符号为可变长度的字符串）
def get_pairs(word):
    pairs = set()  # 初始化一个空集合
    prev_char = word[0]  # 记录前一个字符
    for char in word[1:]:  # 遍历单词中的字符
        pairs.add((prev_char, char))  # 将前一个字符和当前字符作为元组加入集合
        prev_char = char  # 更新前一个字符
    return pairs  # 返回集合


# 构造一个 RoBERTa 分词器，从 GPT-2 分词器派生，使用字节级的字节对编码
# 这个分词器已经进行了训练，将空格视为标记的一部分（有点像 sentencepiece）
# 所以一个单词将根据它是否在句子的开头（没有空格）来进行不同的编码
class RobertaTokenizer(PreTrainedTokenizer):
    """
    Constructs a RoBERTa tokenizer, derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import RobertaTokenizer

    >>> tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
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
    Args:
        vocab_file (`str`):
            Path to the vocabulary file. 词汇文件的路径
        merges_file (`str`):
            Path to the merges file. 合并文件的路径
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See 错误处理方式，默认为"replace"，在将字节解码为UTF-8时使用。
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
            开头序列的特殊标记，在预训练中使用。可以作为序列分类器的标记。

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
            结尾序列的特殊标记。

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
            分隔符标记，用于从多个序列构建序列，例如用于序列分类的两个序列，或用于文本和问题之间的解答。也用作使用特殊标记构建的序列的最后一个标记。
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
            在进行序列分类（整个序列而非每个标记的分类）时使用的分类器标记。在使用特殊标记构建序列时，它是序列的第一个标记。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
            未知标记。如果某个标记不在词汇表中，无法转换为ID，将设为此标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
            用于填充的标记，例如在批处理不同长度的序列时使用。
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
            用于掩码值的标记。在用掩码语言模型训练此模型时使用的标记。这是模型将尝试预测的标记。
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
            是否在输入中添加初始空格。这允许将领先的词视为其他词一样。(RoBERTa 分词器通过前面的空格来检测单词的开始)。

    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    # 初始化函数，用于创建一个新的Tokenizer对象
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
        # 如果bos_token是字符串类型，则将其封装为AddedToken对象，否则保持不变
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        # 如果pad_token是字符串类型，则将其封装为AddedToken对象，否则保持不变
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        # 如果eos_token是字符串类型，则将其封装为AddedToken对象，否则保持不变
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        # 如果unk_token是字符串类型，则将其封装为AddedToken对象，否则保持不变
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 如果sep_token是字符串类型，则将其封装为AddedToken对象，否则保持不变
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        # 如果cls_token是字符串类型，则将其封装为AddedToken对象，否则保持不变
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token

        # 将mask_token视为普通单词，即在其之前包含空格
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        # 以下特殊token不在vocab.json中，让我们按正确的顺序将它们添加进去

        # 用utf-8编码打开vocab_file文件，并加载其中的内容作为encoder
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 将encoder中的键值对调，得到decoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置错误处理方式
        self.errors = errors  # how to handle errors in decoding
        # 创建byte编码到unicode编码的映射
        self.byte_encoder = bytes_to_unicode()
        # 将byte_encoder中的键值对调，得到byte_decoder
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 用utf-8编码打开merges_file文件，并加载其中的内容
        with open(merges_file, encoding="utf-8") as merges_handle:
            # 将内容按换行符分割，并且去掉首尾空行
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        # 将每一行内容按空格分割，组成元组，并将所有元组组成的列表与其对应的序号组成字典
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 初始化缓存
        self.cache = {}
        # 是否在前缀之前添加空格
        self.add_prefix_space = add_prefix_space

        # 应该添加re.IGNORECASE，以便可以为缩写的大写版本执行BPE合并
        # 创建用于匹配的正则表达式模式
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 调用父类的初始化函数，将参数传递给父类
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

    # 返回vocab的大小
    @property
    def vocab_size(self):
        return len(self.encoder)

    # 返回包含所有token的vocab字典
    def get_vocab(self):
        vocab = dict(self.encoder).copy()
        vocab.update(self.added_tokens_encoder)
        return vocab
    # 实现 BPE（字节对编码）算法，将输入的 token 进行编码
    def bpe(self, token):
        # 如果 token 已经在缓存中存在，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 将 token 转换为元组
        word = tuple(token)
        # 获取 token 中的字节对
        pairs = get_pairs(word)

        # 如果没有字节对，则返回原始 token
        if not pairs:
            return token

        # 循环处理所有的字节对，直到无法再进行合并
        while True:
            # 找到当前权重最小的字节对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果这个字节对不在 BPE 数据中，跳出循环
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                # 如果当前位置是字节对的一部分，则合并成一个新的字节
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            # 如果新的 word 的长度为 1，则跳出循环
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        # 将结果从元组转换为字符串
        word = " ".join(word)
        # 将结果存入缓存
        self.cache[token] = word
        return word

    # 将输入的文本进行分词处理
    def _tokenize(self, text):
        # 初始化空的 BPE token 列表
        bpe_tokens = []
        # 使用正则表达式找到文本中的所有 token
        for token in re.findall(self.pat, text):
            # 将每个 token 转换为字节编码的字符串，并合并 BPE 分词结果
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # 将 token 转换为对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将 id 转换为对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    # 将一系列 token 转换为字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 合并所有的 token，并将其转换为 utf-8 编码
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建词汇文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构建合并文件路径
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 写入编码器内容到词汇文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 写入 BPE token 和 token 索引到合并文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A RoBERTa sequence has the following format:

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
        # 如果只有一个 token_ids，则在首尾加上特殊 token，返回结果
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 如果有两个 token_ids，则构建带有特殊 token 的输入
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

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
        # 如果已经有特殊标记，则调用父类的方法获取特殊标记掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果只有第一个序列的 token_ids_1 为 None
        if token_ids_1 is None:
            # 返回特殊标记掩码，序列开始、序列 token、序列结束的标记分别为 1, 0, 1
            return [1] + ([0] * len(token_ids_0)) + [1]
        # 否则，返回特殊标记掩码，序列开始、序列 token、序列结束、另一个序列的开始、序列 token、序列结束的标记分别为 1, 0, 1, 1, 0, 1
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # 分隔符和类别标记的 ID
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # 如果只有一个序列
        if token_ids_1 is None:
            # 返回长度为序列长度加上分隔符和类别标记的长度的全零列表
            return len(cls + token_ids_0 + sep) * [0]
        # 否则，返回长度为两个序列长度加上分隔符和类别标记的长度的全零列表
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # 从参数中获取是否需要添加前缀空格的信息
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # 如果文本已经被分割成单词或需要添加前缀空格，且文本长度大于0且第一个字符不是空格
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            # 在文本前添加空格
            text = " " + text
        # 返回文本和参数
        return (text, kwargs)
```