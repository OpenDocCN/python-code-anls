# `.\models\qwen2\tokenization_qwen2.py`

```
# 定义常量，指定文件名字典，包括词汇表文件和合并文件
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",     # 词汇表文件名
    "merges_file": "merges.txt",    # 合并文件名
}

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"qwen/qwen-tokenizer": "https://huggingface.co/qwen/qwen-tokenizer/resolve/main/vocab.json"},   # 词汇表文件映射
    "merges_file": {"qwen/qwen-tokenizer": "https://huggingface.co/qwen/qwen-tokenizer/resolve/main/merges.txt"},  # 合并文件映射
}

# 定义预训练模型最大输入尺寸
MAX_MODEL_INPUT_SIZES = {"qwen/qwen-tokenizer": 32768}   # 模型最大输入尺寸映射

# 定义用于预分词的正则表达式模式
PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # 匹配缩写词和单词、数字、非字母数字字符、空白行、空格

@lru_cache()
# 从 transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode 复制而来
def bytes_to_unicode():
    """
    返回 utf-8 字节列表及其与 Unicode 字符的映射表。避免映射到空白字符或控制字符，以避免 BPE 代码错误。

    可逆的 BPE 代码在 Unicode 字符串上工作。这意味着如果要避免 UNK（未知标记），需要在词汇表中包含大量的 Unicode 字符。
    例如，对于约 100 亿个标记的数据集，您大约需要包含 5000 个 Unicode 字符才能获得良好的覆盖率。
    """
    # 定义 Unicode 字节和 Unicode 字符映射表的起始范围
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    # 返回 Unicode 字节到字符的映射字典
    return dict(zip(bs, cs))


# 从 transformers.models.gpt2.tokenization_gpt2.get_pairs 复制而来
def get_pairs(word):
    """
    返回单词中的符号对集合。

    单词表示为符号元组（符号是长度可变的字符串）。
    """
    # 初始化符号对集合和前一个字符
    pairs = set()
    prev_char = word[0]
    # 遍历单词中的字符，生成符号对集合
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    # 返回符号对集合
    return pairs


class Qwen2Tokenizer(PreTrainedTokenizer):
    """
    Qwen2 的 tokenizer 类，继承自 PreTrainedTokenizer 类。
    """
    # 定义一个 Qwen2 tokenizer，基于字节级的 Byte-Pair-Encoding。
    
    # 和 GPT2Tokenizer 类似，此分词器经过训练以将空格视为标记的一部分，因此一个单词在句子开头（没有空格）和其他位置可能会被编码成不同的标记：
    
    vocab_files_names = VOCAB_FILES_NAMES
    # 从 transformers 库导入的文件名列表，包含词汇文件名
    
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型词汇文件的映射，用于指定各种预训练模型的词汇文件
    
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES
    # 不同模型的最大输入长度限制，以 token 数量计算
    
    model_input_names = ["input_ids", "attention_mask"]
    # 模型输入所需的标记名称列表，包括输入 IDs 和注意力掩码
    # 初始化方法，用于创建一个新的tokenizer对象
    def __init__(
        self,
        vocab_file,  # 词汇文件路径，用于指定词汇表
        merges_file,  # 合并文件路径，用于指定BPE合并规则
        errors="replace",  # 解码错误处理方式，默认替换错误字符
        unk_token="<|endoftext|>",  # 未知标记，默认为特定的结束标记
        bos_token=None,  # 开始标记，如果指定则创建特殊的添加标记对象
        eos_token="<|endoftext|>",  # 结束标记，默认为特定的结束标记
        pad_token="<|endoftext|>",  # 填充标记，默认为特定的结束标记
        clean_up_tokenization_spaces=False,  # 是否清除标记化空格
        split_special_tokens=False,  # 是否拆分特殊标记
        **kwargs,  # 其他关键字参数
    ):
        # 如果bos_token是字符串，则创建一个特殊的添加标记对象，不进行左右剥离
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        # 如果eos_token是字符串，则创建一个特殊的添加标记对象，不进行左右剥离
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        # 如果unk_token是字符串，则创建一个特殊的添加标记对象，不进行左右剥离
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        # 如果pad_token是字符串，则创建一个特殊的添加标记对象，不进行左右剥离
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )

        # 从vocab_file中加载词汇表到self.encoder
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建self.decoder，将self.encoder的键值对颠倒
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # 设置解码时的错误处理方式
        self.byte_encoder = bytes_to_unicode()  # 使用字节到Unicode的编码器
        # 创建self.byte_decoder，将self.byte_encoder的键值对颠倒
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_merges = []
        # 从merges_file中读取BPE合并规则，创建bpe_merges列表
        with open(merges_file, encoding="utf-8") as merges_handle:
            for line in merges_handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                bpe_merges.append(tuple(line.split()))
        # 使用BPE合并规则创建self.bpe_ranks字典
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        
        # 注意：缓存可以无限增长，对于长时间运行的进程（特别是没有空格分隔单词的语言文本，如中文），缓存可能会变得非常大。
        # 这不是内存泄漏，但看起来像是。GPT2Tokenizer也有同样的问题，因此我们保持一致。
        self.cache = {}  # 初始化缓存，用于存储tokenization的结果
        
        # 编译预处理的正则表达式模式，用于分隔文本
        self.pat = re.compile(PRETOKENIZE_REGEX)

        # 如果kwargs中包含"add_prefix_space"并且其值为True，则发出警告
        if kwargs.get("add_prefix_space", False):
            logger.warning_once(
                f"{self.__class__.__name__} does not support `add_prefix_space`, setting it to True has no effect."
            )

        # 调用父类的初始化方法，设置错误处理方式、开始标记、结束标记、填充标记、未知标记等
        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

    @property
    # 返回词汇表大小
    def vocab_size(self) -> int:
        return len(self.encoder)
    # 从 GPT2Tokenizer 类中复制而来，返回词汇表的字典，包括编码器和添加的特殊标记编码器
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # 从 GPT2Tokenizer 类中复制而来，执行 BPE（字节对编码）算法，将 token 分解为 BPE tokens
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            # 找出当前最小的 bigram，根据 bpe_ranks 中的排序
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
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

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        # 将 tuple 转换为字符串，并缓存结果
        word = " ".join(word)
        self.cache[token] = word
        return word

    # 从 GPT2Tokenizer 类中复制而来，对文本进行分词处理
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # 将 token 转换为 UTF-8 编码的字节，并用 byte_encoder 映射到 unicode 字符串，避免 BPE 的控制标记（在我们的情况下是空格）
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )
            # 使用 BPE 算法处理 token，将结果拆分为多个 BPE token，并添加到 bpe_tokens 中
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # 从 GPT2Tokenizer 类中复制而来，将 token 转换为其对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 从 GPT2Tokenizer 类中复制而来，将 id 转换为其对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    # 从 GPT2Tokenizer 类中复制而来，将 tokens 序列转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        # 使用 byte_decoder 将每个字符的字节解码为 UTF-8 字符串，并处理可能的错误
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    # `spaces_between_special_tokens`默认为True，用于慢速标记器中的_decode，无法在其他地方配置，
    # 但对于Qwen2Tokenizer，它应该默认为False
    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = False,
        spaces_between_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        # 调用父类方法来解码token_ids为字符串
        return super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

    # 从transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.save_vocabulary复制而来
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存路径不是目录，则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建词汇表文件名和合并文件名
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将编码器内容以JSON格式写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将BPE标记和它们的索引写入合并文件
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

    # 准备文本进行标记化前的预处理，包括Unicode规范化和传递额外的参数
    def prepare_for_tokenization(self, text, **kwargs):
        text = unicodedata.normalize("NFC", text)
        return (text, kwargs)
```