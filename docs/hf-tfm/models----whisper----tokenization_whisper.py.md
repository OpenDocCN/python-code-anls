# `.\models\whisper\tokenization_whisper.py`

```py
# 定义一个函数，用于将 UTF-8 字节映射为 Unicode 字符。避免映射到空白字符或控制字符，以确保 BPE 处理正常运行。
def bytes_to_unicode():
    """
    返回 utf-8 字节列表及其对应的 Unicode 字符映射。特别避免映射到空白字符或控制字符，以免在 BPE 处理时出错。

    可逆的 BPE 编码适用于 Unicode 字符串。这意味着如果要避免 UNK 标记，需要在词汇表中包含大量的 Unicode 字符。
    例如，处理约 100 亿个标记的数据集时，可能需要大约 5000 个字符才能覆盖得好。这占了通常 32K BPE 词汇表的显著比例。
    为了避免这种情况，我们需要 UTF-8 字节与 Unicode 字符串之间的查找表。
    """
    # 定义基本的 UTF-8 字节范围
    bs = (
        list(range(ord("!"), ord("~") + 1)) +  # printable ASCII 字符
        list(range(ord("¡"), ord("¬") + 1)) +  # Latin-1 扩展字符
        list(range(ord("®"), ord("ÿ") + 1))  # Latin-1 补充字符
    )
    cs = bs[:]  # 复制基本字节范围
    n = 0
    # 遍历所有 2^8 个字节值，确保包含所有可能的字节值
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]  # 将编码的字节映射为 Unicode 字符
    return dict(zip(bs, cs))  # 返回字节到 Unicode 字符的映射字典


# 获取与当前模块相关联的日志记录器
logger = logging.get_logger(__name__)


# 定义一个函数，用于获取给定单词中的符号对集合
def get_pairs(word):
    """
    返回单词中的符号对集合。

    单词被表示为符号的元组（符号是长度可变的字符串）。
    """
    pairs = set()
    prev_char = word[0]  # 获取单词的第一个符号
    # 对单词中除第一个字符外的每个字符进行迭代
    for char in word[1:]:
        # 将前一个字符和当前字符作为一个元组加入到集合中
        pairs.add((prev_char, char))
        # 更新前一个字符为当前字符，以便下一次迭代使用
        prev_char = char
    # 返回存储了字符对的集合
    return pairs
# 支持的语言列表，每个键值对表示语言代码和语言名称的映射关系
LANGUAGES = {
    "en": "english",        # 英语
    "zh": "chinese",        # 中文
    "de": "german",         # 德语
    "es": "spanish",        # 西班牙语
    "ru": "russian",        # 俄语
    "ko": "korean",         # 韩语
    "fr": "french",         # 法语
    "ja": "japanese",       # 日语
    "pt": "portuguese",     # 葡萄牙语
    "tr": "turkish",        # 土耳其语
    "pl": "polish",         # 波兰语
    "ca": "catalan",        # 加泰罗尼亚语
    "nl": "dutch",          # 荷兰语
    "ar": "arabic",         # 阿拉伯语
    "sv": "swedish",        # 瑞典语
    "it": "italian",        # 意大利语
    "id": "indonesian",     # 印尼语
    "hi": "hindi",          # 印地语
    "fi": "finnish",        # 芬兰语
    "vi": "vietnamese",     # 越南语
    "he": "hebrew",         # 希伯来语
    "uk": "ukrainian",      # 乌克兰语
    "el": "greek",          # 希腊语
    "ms": "malay",          # 马来语
    "cs": "czech",          # 捷克语
    "ro": "romanian",       # 罗马尼亚语
    "da": "danish",         # 丹麦语
    "hu": "hungarian",      # 匈牙利语
    "ta": "tamil",          # 泰米尔语
    "no": "norwegian",      # 挪威语
    "th": "thai",           # 泰语
    "ur": "urdu",           # 乌尔都语
    "hr": "croatian",       # 克罗地亚语
    "bg": "bulgarian",      # 保加利亚语
    "lt": "lithuanian",     # 立陶宛语
    "la": "latin",          # 拉丁语
    "mi": "maori",          # 毛利语
    "ml": "malayalam",      # 马拉雅拉姆语
    "cy": "welsh",          # 威尔士语
    "sk": "slovak",         # 斯洛伐克语
    "te": "telugu",         # 泰卢固语
    "fa": "persian",        # 波斯语
    "lv": "latvian",        # 拉脱维亚语
    "bn": "bengali",        # 孟加拉语
    "sr": "serbian",        # 塞尔维亚语
    "az": "azerbaijani",    # 阿塞拜疆语
    "sl": "slovenian",      # 斯洛文尼亚语
    "kn": "kannada",        # 卡纳达语
    "et": "estonian",       # 爱沙尼亚语
    "mk": "macedonian",     # 马其顿语
    "br": "breton",         # 布列塔尼语
    "eu": "basque",         # 巴斯克语
    "is": "icelandic",      # 冰岛语
    "hy": "armenian",       # 亚美尼亚语
    "ne": "nepali",         # 尼泊尔语
    "mn": "mongolian",      # 蒙古语
    "bs": "bosnian",        # 波斯尼亚语
    "kk": "kazakh",         # 哈萨克语
    "sq": "albanian",       # 阿尔巴尼亚语
    "sw": "swahili",        # 斯瓦希里语
    "gl": "galician",       # 加利西亚语
    "mr": "marathi",        # 马拉地语
    "pa": "punjabi",        # 旁遮普语
    "si": "sinhala",        # 僧伽罗语
    "km": "khmer",          # 高棉语
    "sn": "shona",          # 绍纳语
    "yo": "yoruba",         # 约鲁巴语
    "so": "somali",         # 索马里语
    "af": "afrikaans",      # 南非荷兰语
    "oc": "occitan",        # 奥克语
    "ka": "georgian",       # 格鲁吉亚语
    "be": "belarusian",     # 白俄罗斯语
    "tg": "tajik",          # 塔吉克语
    "sd": "sindhi",         # 信德语
    "gu": "gujarati",       # 古吉拉特语
    "am": "amharic",        # 阿姆哈拉语
    "yi": "yiddish",        # 意第绪语
    "lo": "lao",            # 老挝语
    "uz": "uzbek",          # 乌兹别克语
    "fo": "faroese",        # 法罗语
    "ht": "haitian creole", # 海地克里奥尔语
    "ps": "pashto",         # 普什图语
    "tk": "turkmen",        # 土库曼语
    "nn": "nynorsk",        # 新挪威语
    "mt": "maltese",        # 马耳他语
    "sa": "sanskrit",       # 梵语
    "lb": "luxembourgish",  # 卢森堡语
    "my": "myanmar",        # 缅甸语
    "bo": "tibetan",        # 藏语
    "tl": "tagalog",        # 菲律宾语
    "mg": "malagasy",       # 马达加斯加语
    "as": "assamese",       # 阿萨姆语
    "tt": "tatar",          # 鞑靼语
    "haw": "hawaiian",      # 夏威夷语
    "ln": "lingala",        # 林加拉语
    "ha": "hausa",          # 豪萨语
    "ba": "bashkir",        # 巴什基尔语
    "jw": "javanese",       # 爪哇语
    "su": "sundanese",      # 巽他语
    "yue": "cantonese",     # 粤语
}

# 根据语言名称查找对应的语言代码，包含几个语言别名
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",         # 缅甸语
    "valencian": "ca",       # 瓦伦西亚语
    "flemish": "nl",         # 佛兰芒语
    "haitian": "ht",         # 海地克里奥尔语
    "letzeburgesch": "lb",   # 卢森堡语
    "pushto": "ps",          # 普什图语
    "panjabi": "pa",         # 旁遮普语
    "moldavian":
    # 这个类的目的是为了初始化一个文本处理模型的配置，用于处理文本生成和处理任务。

    # 下面这些变量定义了与模型配置相关的常量和映射

    vocab_files_names = VOCAB_FILES_NAMES  # 从外部引入的词汇表文件名常量
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练词汇文件的映射表
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES  # 最大模型输入尺寸的常量
    model_input_names = ["input_ids", "attention_mask"]  # 模型输入的名称列表

    def __init__(
        self,
        vocab_file,
        merges_file,
        normalizer_file=None,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_prefix_space=False,
        language=None,
        task=None,
        predict_timestamps=False,
        **kwargs,
    ):
        # 初始化函数，用于创建一个新的文本处理模型配置实例

        self.vocab_file = vocab_file  # 词汇文件路径
        self.merges_file = merges_file  # 合并文件路径
        self.normalizer_file = normalizer_file  # 规范化器文件路径（可选）
        self.errors = errors  # 解码字节为UTF-8时的错误处理方式
        self.unk_token = unk_token  # 未知标记（默认为"<|endoftext|>"）
        self.bos_token = bos_token  # 序列起始标记（默认为"<|endoftext|>"）
        self.eos_token = eos_token  # 序列结束标记（默认为"<|endoftext|>"）
        self.pad_token = pad_token  # 填充标记（可选）
        self.add_prefix_space = add_prefix_space  # 是否在输入前添加空格（默认为False）
        self.language = language  # 文本语言（可选）
        self.task = task  # 任务标识符（可选）
        self.predict_timestamps = predict_timestamps  # 是否预测时间戳（默认为False）
        self.kwargs = kwargs  # 其它未命名参数
    ):
        # 如果给定的 bos_token 是字符串，则创建一个特殊的 AddedToken 对象
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(bos_token, str)
            else bos_token
        )
        # 如果给定的 eos_token 是字符串，则创建一个特殊的 AddedToken 对象
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(eos_token, str)
            else eos_token
        )
        # 如果给定的 unk_token 是字符串，则创建一个特殊的 AddedToken 对象
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(unk_token, str)
            else unk_token
        )
        # 如果给定的 pad_token 是字符串，则创建一个特殊的 AddedToken 对象
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(pad_token, str)
            else pad_token
        )

        # 使用 UTF-8 编码打开词汇文件，并加载其中的编码器
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 根据编码器创建解码器，反转键值对
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置解码时的错误处理方式
        self.errors = errors  # how to handle errors in decoding
        # 创建字节到 Unicode 的编码映射
        self.byte_encoder = bytes_to_unicode()
        # 创建 Unicode 到字节的解码映射
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 使用 UTF-8 编码打开 BPE 合并文件，读取并处理成列表
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        # 将 BPE 合并操作转换为元组并创建一个排名字典
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 初始化缓存字典
        self.cache = {}
        # 设置是否在词前添加空格的选项
        self.add_prefix_space = add_prefix_space

        # 如果提供了正规化器文件，则使用 UTF-8 编码打开它并加载英语拼写正规化器
        if normalizer_file is not None:
            with open(normalizer_file, encoding="utf-8") as vocab_handle:
                self.english_spelling_normalizer = json.load(vocab_handle)
        else:
            # 否则，将英语拼写正规化器设置为 None
            self.english_spelling_normalizer = None

        # 正则表达式模式，用于匹配文本中的特定模式，包括缩略词和时间戳
        # 添加 re.IGNORECASE 选项以支持大小写不敏感的 BPE 合并操作
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.timestamp_pat = re.compile(r"<\|(\d+\.\d+)\|>")

        # 初始化父类 GPT2Tokenizer，并传递参数
        self.language = language
        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        # 设置语言模型的任务类型和是否预测时间戳的选项
        self.task = task
        self.predict_timestamps = predict_timestamps

    @property
    # 返回词汇表的大小
    def vocab_size(self) -> int:
        return len(self.encoder)

    # 返回当前词汇表及额外添加的 token
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 从 transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.bpe 复制，修改为使用 Whisper
    def bpe(self, token):
        # 如果缓存中已经存在该 token 的处理结果，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 将 token 转换成元组形式
        word = tuple(token)
        # 获取 token 的所有字符对
        pairs = get_pairs(word)

        # 如果 token 没有字符对，则直接返回原始 token
        if not pairs:
            return token

        # 开始 BPE 算法的处理过程，直到不能再合并字符对为止
        while True:
            # 找到频率最低的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果找到的字符对不在预训练的 BPE 词汇表中，则停止合并
            if bigram not in self.bpe_ranks:
                break
            # 将词汇表中的第一个字符和第二个字符分开
            first, second = bigram
            new_word = []
            i = 0
            # 遍历 token 的所有字符
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    # 如果找不到第一个字符，则直接添加剩余的字符
                    new_word.extend(word[i:])
                    break
                else:
                    # 将第一个字符前面的字符添加到新的单词中
                    new_word.extend(word[i:j])
                    i = j

                # 如果当前字符和下一个字符组成一个字符对，则合并
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    # 否则将当前字符添加到新的单词中
                    new_word.append(word[i])
                    i += 1
            # 更新处理后的 token 为元组形式
            new_word = tuple(new_word)
            word = new_word
            # 如果只剩下一个字符，则停止处理
            if len(word) == 1:
                break
            else:
                # 否则继续获取新的字符对
                pairs = get_pairs(word)
        # 将处理后的 token 转换为字符串形式
        word = " ".join(word)
        # 将处理结果加入缓存中
        self.cache[token] = word
        # 返回处理后的结果
        return word

    def set_prefix_tokens(self, language: str = None, task: str = None, predict_timestamps: bool = None):
        """
        Override the prefix tokens appended to the start of the label sequence. This method can be used standalone to
        update the prefix tokens as required when fine-tuning. Example:

        ```
        >>> # instantiate the tokenizer and set the prefix token to Spanish
        >>> tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="spanish")
        >>> # now switch the prefix token from Spanish to French
        >>> tokenizer.set_prefix_tokens(language="french")
        ```

        Args:
            language (`str`, *optional*, defaults to `None`):
                The language of the transcription text.
            task (`str`, *optional*, defaults to `None`):
                Task identifier to append at the start of sequence (if any).
            predict_timestamps (`bool`, *optional*, defaults to `None`):
                Whether to omit the `<|notimestamps|>` token at the start of the sequence.
        """
        # 更新语言设置，如果未提供则保持原样
        self.language = language if language is not None else self.language
        # 更新任务标识，如果未提供则保持原样
        self.task = task if task is not None else self.task
        # 更新是否预测时间戳的设置，如果未提供则保持原样
        self.predict_timestamps = predict_timestamps if predict_timestamps is not None else self.predict_timestamps

    @property
    # 返回一个包含特殊前缀 token 的列表，用于初始化模型输入
    def prefix_tokens(self) -> List[int]:
        # 将特殊开始转录标记转换为其对应的 token ID
        bos_token_id = self.convert_tokens_to_ids("<|startoftranscript|>")
        # 将特殊翻译标记转换为其对应的 token ID
        translate_token_id = self.convert_tokens_to_ids("<|translate|>")
        # 将特殊转录标记转换为其对应的 token ID
        transcribe_token_id = self.convert_tokens_to_ids("<|transcribe|>")
        # 将特殊无时间戳标记转换为其对应的 token ID
        notimestamps_token_id = self.convert_tokens_to_ids("<|notimestamps|>")
        # 取得所有语言代码的元组
        langs = tuple(LANGUAGES.keys())

        # 如果指定了语言
        if self.language is not None:
            # 将语言名称转换为小写
            self.language = self.language.lower()
            # 如果语言在语言到语言代码的映射中
            if self.language in TO_LANGUAGE_CODE:
                # 获取语言代码
                language_id = TO_LANGUAGE_CODE[self.language]
            # 如果语言在语言代码列表中
            elif self.language in TO_LANGUAGE_CODE.values():
                # 直接使用语言代码
                language_id = self.language
            else:
                # 判断语言是否是两位字母代码
                is_language_code = len(self.language) == 2
                # 抛出不支持的语言异常，提示支持的语言列表
                raise ValueError(
                    f"Unsupported language: {self.language}. Language should be one of:"
                    f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                )

        # 如果指定了任务
        if self.task is not None:
            # 如果任务不在任务ID列表中，则抛出异常
            if self.task not in TASK_IDS:
                raise ValueError(f"Unsupported task: {self.task}. Task should be in: {TASK_IDS}")

        # 构建前缀序列，起始于开始 token
        bos_sequence = [bos_token_id]
        # 如果指定了语言，则将语言相关 token ID 添加到序列中
        if self.language is not None:
            bos_sequence.append(bos_token_id + 1 + langs.index(language_id))
        # 如果指定了任务，则根据任务类型添加对应的 token ID
        if self.task is not None:
            bos_sequence.append(transcribe_token_id if self.task == "transcribe" else translate_token_id)
        # 如果不需要预测时间戳，则添加不含时间戳的 token ID
        if not self.predict_timestamps:
            bos_sequence.append(notimestamps_token_id)
        # 返回构建好的前缀序列
        return bos_sequence

    # 从一个序列构建模型输入，通过添加结束 token
    # 拷贝自 transformers.models.speech_to_text.tokenization_speech_to_text.Speech2TextTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        # 如果没有第二个序列，则直接添加第一个序列和结束 token
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + [self.eos_token_id]
        # 否则，按照 API 一致性保留对序列对的处理逻辑，添加前缀 token、两个序列以及结束 token
        return self.prefix_tokens + token_ids_0 + token_ids_1 + [self.eos_token_id]

    # 获取包含特殊 token 的掩码
    # 拷贝自 transformers.models.speech_to_text.tokenization_speech_to_text.Speech2TextTokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
    # 返回特殊标记掩码列表，用于指示是否为特殊标记
    def get_special_tokens_mask(
        self, 
        token_ids_0: List[int], 
        token_ids_1: Optional[List[int]] = None, 
        already_has_special_tokens: bool = False
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

        # 前缀部分的特殊标记列表全为1
        prefix_ones = [1] * len(self.prefix_tokens)
        # 后缀部分的特殊标记列表包含一个1
        suffix_ones = [1]
        if token_ids_1 is None:
            # 如果没有第二个序列，返回前缀标记后接0序列标记，再接一个后缀标记
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        # 如果有第二个序列，返回前缀标记后接两个0序列标记，再接一个后缀标记
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    # 从GPT2Tokenizer._tokenize复制并改名为Whisper的私有方法_tokenize
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        # 使用正则表达式查找文本中的所有匹配项
        for token in re.findall(self.pat, text):
            # 将每个token编码为字节，并使用字节编码器映射到Unicode字符串，避免BPE的控制标记
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            # 将BPE处理后的token分割并扩展到bpe_tokens列表中
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # 从GPT2Tokenizer._convert_token_to_id复制并改名为Whisper的私有方法_convert_token_to_id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 根据词汇表将token转换为对应的ID，若未找到使用unk_token对应的ID
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 从GPT2Tokenizer._convert_id_to_token复制并改名为Whisper的私有方法_convert_id_to_token
    def _convert_id_to_token(self, index):
        """
        Converts an index (integer) in a token (str) using the vocab. Whisper's base tokenizer always decodes OOV
        tokens as "", thus we do not use the `unk_token` here.
        """
        # 根据词汇表将ID转换为对应的token，若未找到使用空字符串表示未知token
        return self.decoder.get(index, "")

    # 私有方法_normalize已废弃，在v5版本中将被移除，建议使用normalize方法
    def _normalize(self, text):
        warnings.warn(
            "The private method `_normalize` is deprecated and will be removed in v5 of Transformers."
            "You can normalize an input string using the Whisper English normalizer using the `normalize` method."
        )
        # 直接返回文本的normalize结果
        return self.normalize(text)
    # 发出警告，提示私有方法 `_basic_normalize` 已被弃用，将在 Transformers 的 v5 版本中移除
    # 建议使用 `basic_normalize` 方法来规范化输入字符串
    def _basic_normalize(self, text, remove_diacritics=False):
        warnings.warn(
            "The private method `_basic_normalize` is deprecated and will be removed in v5 of Transformers."
            "You can normalize an input string using the Whisper basic normalizer using the `basic_normalize` method."
        )
        # 调用 `basic_normalize` 方法来规范化文本
        return self.basic_normalize(text, remove_diacritics=remove_diacritics)

    def normalize(self, text):
        """
        使用 `EnglishTextNormalizer` 类来规范化给定的字符串，该类对英语文本进行常见转换。
        """
        # 创建 `EnglishTextNormalizer` 实例
        normalizer = EnglishTextNormalizer(self.english_spelling_normalizer)
        # 调用实例的规范化方法来处理文本
        return normalizer(text)

    @staticmethod
    def basic_normalize(text, remove_diacritics=False):
        """
        使用 `BasicTextNormalizer` 类来规范化给定的字符串，该类对多语言文本进行常见转换。
        """
        # 创建 `BasicTextNormalizer` 实例
        normalizer = BasicTextNormalizer(remove_diacritics=remove_diacritics)
        # 调用实例的规范化方法来处理文本
        return normalizer(text)

    def _decode_with_timestamps(self, token_ids, skip_special_tokens=False, time_precision=0.02) -> str:
        """
        解码带有时间戳的 token 序列，时间戳的 token ID 大于特殊 token 的 ID 范围，会被 `decode()` 忽略。
        该方法将带有时间戳的 token 序列解码，例如 "<|1.08|>"。
        """
        # 时间戳开始的 token ID
        timestamp_begin = self.all_special_ids[-1] + 1
        # 输出列表，用于存储解码后的字符串或 token 列表
        outputs = [[]]

        # 当前最大时间戳
        cur_max_timestamp = 0.0
        # 前一个段落的长度
        prev_segments_len = 0.0

        for token in token_ids:
            if token >= timestamp_begin:
                # 计算时间戳
                timestamp = float((token - timestamp_begin) * time_precision)

                if timestamp < cur_max_timestamp:
                    # 下一个段落已开始
                    prev_segments_len += cur_max_timestamp

                cur_max_timestamp = timestamp

                # 添加带有时间戳的标记到输出列表
                outputs.append(f"<|{(timestamp + prev_segments_len):.2f}|>")
                outputs.append([])
            else:
                # 将 token 添加到当前段落的输出列表中
                outputs[-1].append(token)

        # 解码输出列表中的每个段落，并将结果连接成一个字符串
        outputs = [
            s if isinstance(s, str) else self.decode(s, skip_special_tokens=skip_special_tokens) for s in outputs
        ]
        return "".join(outputs)
    def _compute_offsets(self, token_ids, time_precision=0.02):
        """
        Compute offsets for a given tokenized input

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            time_precision (`float`, `optional`, defaults to 0.02):
                The time ratio to convert from token to time.
        """
        offsets = []
        
        # 确保 token_ids 是放置在 CPU 上的 torch 张量
        if "torch" in str(type(token_ids)) and (hasattr(token_ids, "cpu") and callable(token_ids.cpu)):
            token_ids = token_ids.cpu()
        
        # 将 token_ids 转换为 numpy 数组
        token_ids = np.array(token_ids)
        
        # 检查是否只能处理单个输入
        if token_ids.shape[0] > 1 and len(token_ids.shape) > 1:
            raise ValueError("Can only process a single input at a time")
        
        # 确定时间戳开始的位置
        timestamp_begin = self.all_special_ids[-1] + 1
        
        # 标记出时间戳所在的位置
        timestamp_tokens = token_ids >= timestamp_begin
        
        # 找出连续的时间戳位置
        consecutive = np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1
        
        # 如果没有连续的时间戳或者只有一个时间戳，则返回空列表
        if consecutive.shape[0] == 0 and timestamp_tokens.sum() <= 1:
            return []
        elif np.where(timestamp_tokens)[0][-1] + 1 not in consecutive:
            consecutive = np.append(consecutive, np.where(timestamp_tokens)[0][-1] + 1)
        
        # 初始化最后一个时间戳位置
        last_slice = np.where(timestamp_tokens)[0][0]
        
        # 遍历连续的时间戳位置
        for current_slice in consecutive:
            sliced_tokens = token_ids[last_slice:current_slice]
            
            # 如果切片长度大于1，则处理时间戳位置并进行预处理
            if len(sliced_tokens) > 1:
                start_timestamp_position = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_position = sliced_tokens[-1].item() - timestamp_begin
                
                # 从文本输出中去除时间戳标记的 token
                sliced_tokens = self._preprocess_token_ids(sliced_tokens)
                
                # 解码处理后的 token
                text = self._decode(sliced_tokens)
                
                # 过滤文本中的时间戳标记
                text = self._filter_timestamp_ids(text)
                
                # 将处理后的信息添加到偏移列表中
                offsets.append(
                    {
                        "text": text,
                        "timestamp": (
                            start_timestamp_position * time_precision,
                            end_timestamp_position * time_precision,
                        ),
                    }
                )
    def _preprocess_token_ids(self, token_ids, skip_special_tokens: bool = False):
        """
        Pre-process the token ids for decoding by removing the prompt tokens ids and timestamp token ids.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Typically, obtained using the `__call__` method of the tokenizer.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens from the token ids. If `True`, the prompt token ids will be
                removed.
        """
        # 如果 skip_special_tokens 为 True，则获取特殊标记的 token id
        if skip_special_tokens:
            prompt_token_id = self.convert_tokens_to_ids("<|startofprev|>")
            decoder_start_token_id = self.convert_tokens_to_ids("<|startoftranscript|>")
            # 调用 _strip_prompt 方法去除 token_ids 中的提示和时间戳 token id
            token_ids = self._strip_prompt(token_ids, prompt_token_id, decoder_start_token_id)

        return token_ids

    def _filter_timestamp_ids(self, token_ids):
        """
        Filter out timestamp ids from token_ids using regex pattern.

        Args:
            token_ids (`str`): Token ids to filter.

        Returns:
            `str`: Token ids with timestamps removed.
        """
        # 使用正则表达式模式 self.timestamp_pat 去除 token_ids 中的时间戳
        return re.sub(self.timestamp_pat, "", token_ids)

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        output_offsets: bool = False,
        time_precision: float = 0.02,
        decode_with_timestamps: bool = False,
        normalize: bool = False,
        basic_normalize: bool = False,
        remove_diacritics: bool = False,
        **kwargs,
    ):
        """
        Convert token ids into human-readable text.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens from the token ids during decoding.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `None`):
                Whether to clean up extra spaces around tokenized output.
            output_offsets (`bool`, *optional*, defaults to `False`):
                Whether to return the offsets of tokens in the original input.
            time_precision (`float`, *optional*, defaults to `0.02`):
                Precision of time-related information in seconds.
            decode_with_timestamps (`bool`, *optional*, defaults to `False`):
                Whether to decode timestamps along with token ids.
            normalize (`bool`, *optional*, defaults to `False`):
                Whether to normalize the decoded text.
            basic_normalize (`bool`, *optional*, defaults to `False`):
                Whether to apply basic normalization to the decoded text.
            remove_diacritics (`bool`, *optional*, defaults to `False`):
                Whether to remove diacritics from the decoded text.
            **kwargs: Additional keyword arguments.

        Returns:
            `str` or (`str`, `List[Tuple[int, int]]`) or (`str`, `List[Tuple[int, int]]`, `List[str]`): Depending on
            the combination of arguments, returns decoded text, offsets of tokens, and possibly normalized forms.
        """
        # 实现将 token_ids 转换为人类可读文本的方法，根据参数控制输出格式和内容

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        normalize: bool = False,
        basic_normalize: bool = False,
        remove_diacritics: bool = False,
        **kwargs,
    ):
        """
        Decode token ids into human-readable text.

        Args:
            token_ids (`Union[int, List[int]]`):
                Tokenized input ids.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to skip decoding special tokens.
            normalize (`bool`, *optional*, defaults to `False`):
                Whether to normalize the decoded text.
            basic_normalize (`bool`, *optional*, defaults to `False`):
                Whether to apply basic normalization to the decoded text.
            remove_diacritics (`bool`, *optional*, defaults to `False`):
                Whether to remove diacritics from the decoded text.
            **kwargs: Additional keyword arguments.

        Returns:
            `str`: Decoded text based on token_ids and specified options.
        """
        # 实现将 token_ids 解码为人类可读文本的方法，根据参数控制输出格式和内容
    ) -> str:
        # 从kwargs中取出"use_source_tokenizer"参数，并设置为实例变量
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)
        # 使用convert_ids_to_tokens方法将token_ids转换为tokens列表，跳过特殊token
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        # 避免在字节级BPT中混合使用字节级和Unicode，需要分别构建字符串，用于添加的token和字节级token
        # 参考：https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            # 如果skip_special_tokens为True且token是特殊token，则跳过
            if skip_special_tokens and token in self.all_special_ids:
                continue
            # 如果token在added_tokens_encoder中，则将当前子文本转换为字符串并清空，然后添加token
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        # 如果还有未添加到sub_texts的current_sub_text，则将其添加到sub_texts中
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        # 将所有子文本拼接成最终的文本
        text = "".join(sub_texts)

        # 根据参数normalize或basic_normalize对文本进行相应处理并返回
        if normalize:
            clean_text = self.normalize(text)
            return clean_text
        elif basic_normalize:
            clean_text = self.basic_normalize(text, remove_diacritics=remove_diacritics)
            return clean_text
        else:
            return text

    # 从transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.convert_tokens_to_string复制，将tokens列表转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将tokens列表连接成字符串
        text = "".join(tokens)
        # 使用byte_decoder将字节编码的文本解码为utf-8格式的字符串
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    # 将词汇表保存到指定目录下的文件中，并返回保存的文件路径
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建词汇表文件的路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构建合并文件的路径
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )
        # 构建规范化文件的路径
        normalizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["normalizer_file"]
        )

        # 将编码器(encoder)的内容以 JSON 格式写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将BPE合并信息写入合并文件
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

        # 如果存在英语拼写规范化器，将其内容以 JSON 格式写入规范化文件
        if self.english_spelling_normalizer is not None:
            with open(normalizer_file, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(self.english_spelling_normalizer, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                )

        # 返回保存的文件路径：词汇表文件、合并文件、规范化文件
        return vocab_file, merge_file, normalizer_file

    # 从GPT2Tokenizer.prepare_for_tokenization复制，准备文本进行分词处理
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # 如果文本已经被分成词或需要在文本前添加空格，则在文本前加一个空格
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)

    @property
    # 从GPT2Tokenizer.default_chat_template复制，默认聊天模板
    # 返回一个简单的聊天模板，忽略角色信息，仅将消息与EOS标记连接起来。
    def default_chat_template(self):
        # 警告日志：如果未定义聊天模板，则使用默认模板。
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回格式化后的聊天模板字符串，包含消息内容和EOS标记。
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"

    # 获取解码器提示的标识符列表，根据任务和语言设置前缀标记，并生成解码所需的强制标记。
    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        self.set_prefix_tokens(task=task, language=language, predict_timestamps=not no_timestamps)
        # 前缀标记的形式为: <|startoftranscript|> <|lang_id|> <|task|> <|notimestamps|>
        # 不希望在位置1处强制BOS标记，因为这是生成时的起始标记，
        # 因此我们将前缀标记切片为: <|lang_id|> <|task|> <|notimestamps|>
        forced_tokens = self.prefix_tokens[1:]
        # 返回带有强制标记的标识符和其在列表中的位置的元组列表
        forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_tokens)]
        return forced_decoder_ids

    # 调用静态方法 `_decode_asr`，将ASR模型输出解码为文字结果。
    def _decode_asr(self, model_outputs, *, return_timestamps, return_language, time_precision):
        return _decode_asr(
            self,
            model_outputs,
            return_timestamps=return_timestamps,
            return_language=return_language,
            time_precision=time_precision,
        )

    # 将提示文本转换为可以传递给生成器的标识符列表，避免特殊标记。
    def get_prompt_ids(self, text: str, return_tensors="np"):
        """Converts prompt text to IDs that can be passed to [`~WhisperForConditionalGeneration.generate`]."""
        batch_encoding = self("<|startofprev|>", " " + text.strip(), add_special_tokens=False)

        # 检查特殊标记
        prompt_text_ids = batch_encoding["input_ids"][1:]
        special_token_id = next((x for x in prompt_text_ids if x >= self.all_special_ids[0]), None)
        if special_token_id is not None:
            token = self.convert_ids_to_tokens(special_token_id)
            # 如果在提示文本中遇到不允许的特殊标记，则引发错误
            raise ValueError(f"Encountered text in the prompt corresponding to disallowed special token: {token}.")

        # 将批量编码转换为指定类型的张量
        batch_encoding.convert_to_tensors(tensor_type=return_tensors)
        return batch_encoding["input_ids"]

    # 静态方法：从标识符列表中去除前缀和解码器起始标记。
    @staticmethod
    def _strip_prompt(token_ids: List[int], prompt_token_id: int, decoder_start_token_id: int):
        has_prompt = isinstance(token_ids, list) and token_ids and token_ids[0] == prompt_token_id
        if has_prompt:
            if decoder_start_token_id in token_ids:
                # 返回从解码器起始标记开始的标识符列表
                return token_ids[token_ids.index(decoder_start_token_id):]
            else:
                # 如果解码器起始标记不在列表中，返回空列表
                return []

        # 如果没有前缀，则直接返回原始标识符列表
        return token_ids
def _decode_asr(tokenizer, model_outputs, *, return_timestamps, return_language, time_precision):
    """
    Internal method meant to only be used by asr pipeline. Handles all the little quirks specific to whisper to handle
    the various options not allowed in other seq2seq models
    """

    # =========== Overview ============
    # - iterate over all outputs
    # - all tokens within output
    # - Each token can be
    #   - language token
    #   - special token
    #   - timestamp token
    #   - text token
    # - We accumulate the text tokens.
    # - We split on end timestamps
    # - Lots of complexity comes from stride and timestamps

    last_language = None

    def new_chunk():
        return {"language": last_language, "timestamp": [None, None], "text": ""}

    # Welcome to the state machine !
    chunks = []  # 初始化空列表，用于存储文本块的信息
    chunk = new_chunk()  # 创建一个新的文本块对象
    time_offset = 0.0  # 时间偏移量初始化为0.0
    timestamp_begin = tokenizer.convert_tokens_to_ids("<|notimestamps|>") + 1  # 获取时间戳开始的特殊标记ID
    previous_tokens = []  # 初始化空列表，用于存储先前处理的token
    previous_token_timestamps = []  # 初始化空列表，用于存储先前处理的token的时间戳
    skip = False  # 标志位，用于控制是否跳过处理
    right_stride_start = None  # 初始化右侧步幅开始标记为None

    all_special_ids = set(tokenizer.all_special_ids)  # 获取所有特殊token的ID集合
    # - iterate over all outputs
    if previous_tokens:
        if return_timestamps:
            logger.warning(
                "Whisper did not predict an ending timestamp, which can happen if audio is cut off in the middle of a word. "
                "Also make sure WhisperTimeStampLogitsProcessor was used during generation."
            )
        # Happens when we don't use timestamps
        resolved_tokens, resolved_token_timestamps = _find_longest_common_sequence(
            previous_tokens, previous_token_timestamps
        )
        resolved_text = tokenizer.decode(resolved_tokens)  # 解码得到文本
        chunk["text"] = resolved_text  # 将解码得到的文本存入当前文本块对象
        if return_timestamps == "word":
            chunk["words"] = _collate_word_timestamps(
                tokenizer, resolved_tokens, resolved_token_timestamps, last_language
            )  # 整理单词级别的时间戳信息
        chunks.append(chunk)  # 将当前文本块对象添加到文本块列表中

    # Preparing and cleaning up the pipeline output
    full_text = "".join(chunk["text"] for chunk in chunks)  # 将所有文本块中的文本合并为完整文本
    if return_timestamps or return_language:
        for chunk in chunks:
            if not return_timestamps:
                chunk.pop("timestamp")  # 如果不需要时间戳信息，则移除当前文本块对象中的时间戳
            else:
                chunk["timestamp"] = tuple(chunk["timestamp"])  # 将当前文本块对象中的时间戳转换为元组形式
            if not return_language:
                chunk.pop("language")  # 如果不需要语言信息，则移除当前文本块对象中的语言信息

        if return_timestamps == "word":
            new_chunks = []
            for chunk in chunks:
                new_chunks.extend(chunk["words"])  # 扩展单词级别的时间戳信息到新的文本块列表中
            optional = {"chunks": new_chunks}  # 构建输出的可选信息字典
        else:
            optional = {"chunks": chunks}  # 构建输出的可选信息字典
    else:
        optional = {}  # 如果不需要时间戳和语言信息，则置为空字典
    return full_text, optional  # 返回完整文本和可选信息字典


def _find_longest_common_sequence(sequences, token_timestamp_sequences=None):
    # It would be much harder to do O(n) because of fault tolerance.
    # We actually have a really good property which is that the total sequence
    # MUST be those subsequences in order.
    pass  # 占位符函数，用于查找最长公共子序列
    # 如果提供了 token_timestamp_sequences 参数，将按照相同的方式分割这些序列。

    # 从 sequences 列表中获取第一个序列
    left_sequence = sequences[0]
    # 计算左侧序列的长度
    left_length = len(left_sequence)
    # 初始化总序列为空列表
    total_sequence = []

    # 如果 token_timestamp_sequences 参数被提供
    if token_timestamp_sequences:
        # 从 token_timestamp_sequences 列表中获取第一个序列
        left_token_timestamp_sequence = token_timestamp_sequences[0]
        # 初始化总的 token_timestamp_sequence 为空列表
        total_token_timestamp_sequence = []
    # 遍历 sequences 列表中除第一个元素外的所有序列，同时获取它们的索引值 seq_idx 和序列内容 right_sequence
    for seq_idx, right_sequence in enumerate(sequences[1:]):
        # 初始化 max_ 变量为 0.0，用于存储最大匹配值
        max_ = 0.0
        # 初始化 max_indices 为元组 (left_length, left_length, 0, 0)，记录最大匹配时的索引范围
        max_indices = (left_length, left_length, 0, 0)

        # 这里我们正在滑动匹配
        # [a, b, c, d]
        #          [c, d, f]
        # =        [c] == [d]
        #
        # [a, b, c, d]
        #       [c, d, f]
        # =     [c, d] == [c, d]
        #
        # （省略中间部分）

        # 获取 right_sequence 的长度
        right_length = len(right_sequence)

        # 遍历左侧序列 left_sequence 和右侧序列 right_sequence 的所有可能匹配位置
        for i in range(1, left_length + right_length):
            # epsilon 用于偏向长的完美匹配
            eps = i / 10000.0

            # 针对左侧序列和右侧序列进行切片，确保不越界
            left_start = max(0, left_length - i)
            left_stop = min(left_length, left_length + right_length - i)
            left = np.array(left_sequence[left_start:left_stop])

            right_start = max(0, i - left_length)
            right_stop = min(right_length, i)
            right = np.array(right_sequence[right_start:right_stop])

            # 只能匹配相同长度的子序列
            if len(left) != len(right):
                # 如果长度不同，抛出运行时错误
                raise RuntimeError(
                    "There is a bug within whisper `decode_asr` function, please report it. Dropping to prevent bad inference."
                )

            # 计算左右序列的匹配度
            matches = np.sum(left == right)
            matching = matches / i + eps

            # 如果匹配数大于 1 并且匹配度大于 max_，更新 max_ 和 max_indices
            if matches > 1 and matching > max_:
                max_ = matching
                max_indices = (left_start, left_stop, right_start, right_stop)

        # 将 max_indices 解构为 left_start, left_stop, right_start, right_stop
        (left_start, left_stop, right_start, right_stop) = max_indices

        # 这是一个小冲突优化，因为这些序列在音频中有重叠
        # 对于重叠的左侧，我们会更加信任左侧序列
        # 对于重叠的右侧，我们会更加信任右侧序列
        left_mid = (left_stop + left_start) // 2
        right_mid = (right_stop + right_start) // 2

        # 将 left_sequence 的一部分添加到 total_sequence 中，并更新 left_sequence 和 left_length
        total_sequence.extend(left_sequence[:left_mid])
        left_sequence = right_sequence[right_mid:]
        left_length = len(left_sequence)

        # 如果 token_timestamp_sequences 存在，则将其对应部分也加入 total_token_timestamp_sequence 中
        if token_timestamp_sequences:
            total_token_timestamp_sequence.extend(left_token_timestamp_sequence[:left_mid])
            left_token_timestamp_sequence = token_timestamp_sequences[seq_idx + 1][right_mid:]

    # 将剩余的 left_sequence 加入 total_sequence 中
    total_sequence.extend(left_sequence)

    # 如果 token_timestamp_sequences 不存在，则返回 total_sequence
    if token_timestamp_sequences is None:
        return total_sequence
    # 如果 token_timestamp_sequences 列表长度大于 0，则执行以下操作
    if len(token_timestamp_sequences) > 0:
        # 将 left_token_timestamp_sequence 扩展到 total_token_timestamp_sequence 中
        total_token_timestamp_sequence.extend(left_token_timestamp_sequence)
        # 返回总序列和合并后的总 token 时间戳序列
        return total_sequence, total_token_timestamp_sequence
    else:
        # 如果 token_timestamp_sequences 列表为空，则返回总序列和空列表作为 token 时间戳序列
        return total_sequence, []
# 将给定的 tokens 列表按照单词进行分组，并返回单词列表、以及每个单词对应的 token_id 序列。
def _collate_word_timestamps(tokenizer, tokens, token_timestamps, language):
    # 调用内部函数 _combine_tokens_into_words，将 tokens 组合成单词
    words, _, token_indices = _combine_tokens_into_words(tokenizer, tokens, language)
    # 构建 timings 列表，每个元素包含单词和其起始和结束的时间戳元组
    timings = [
        {
            "text": word,
            "timestamp": (token_timestamps[indices[0]][0], token_timestamps[indices[-1]][1]),
        }
        for word, indices in zip(words, token_indices)
    ]
    return timings


# 将 tokens 按照空格或标点符号分割成单词，并进行必要的标点符号处理
def _combine_tokens_into_words(
    tokenizer,
    tokens: List[int],
    language: str = None,
    prepend_punctuations: str = "\"'“¡¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
):
    """
    Groups tokens by word. Returns a tuple containing a list of strings with the words, and a list of `token_id`
    sequences with the tokens making up each word.
    """
    # 如果未指定 language，则使用 tokenizer 的默认语言
    if language is None:
        language = tokenizer.language
    # 如果 language 仍未指定，设置为英语
    if language is None:
        language = "english"

    # 对于中文、日文、泰文、老挝文、缅甸文和广东话，不使用空格分割
    if language in {"chinese", "japanese", "thai", "lao", "myanmar", "cantonese"}:
        # 调用 _split_tokens_on_unicode，根据 Unicode 分割 tokens
        words, word_tokens, token_indices = _split_tokens_on_unicode(tokenizer, tokens)
    else:
        # 否则，调用 _split_tokens_on_spaces，按空格分割 tokens
        words, word_tokens, token_indices = _split_tokens_on_spaces(tokenizer, tokens)

    # 合并前置和后置标点符号到单词列表中
    _merge_punctuations(words, word_tokens, token_indices, prepend_punctuations, append_punctuations)
    return words, word_tokens, token_indices


# 将 tokens 按照 Unicode 码点进行分割成单词
def _split_tokens_on_unicode(tokenizer, tokens: List[int]):
    """Combine tokens into words by splitting at any position where the tokens are decoded as valid unicode points."""
    # 使用 tokenizer 解码 tokens，以获取完整的解码结果
    decoded_full = tokenizer.decode(tokens, decode_with_timestamps=True)
    replacement_char = "\ufffd"

    words = []
    word_tokens = []
    token_indices = []
    current_tokens = []
    current_indices = []
    unicode_offset = 0

    for token_idx, token in enumerate(tokens):
        current_tokens.append(token)
        current_indices.append(token_idx)
        # 使用 tokenizer 解码当前 tokens
        decoded = tokenizer.decode(current_tokens, decode_with_timestamps=True)

        # 判断是否包含替换字符或者完全匹配替换字符
        if (
            replacement_char not in decoded
            or decoded_full[unicode_offset + decoded.index(replacement_char)] == replacement_char
        ):
            words.append(decoded)
            word_tokens.append(current_tokens)
            token_indices.append(current_indices)
            current_tokens = []
            current_indices = []
            unicode_offset += len(decoded)

    return words, word_tokens, token_indices


# 将 tokens 按照空格或标点符号进行分割成单词
def _split_tokens_on_spaces(tokenizer, tokens: List[int]):
    """Combine tokens into words by splitting at whitespace and punctuation tokens."""
    # 调用 _split_tokens_on_unicode，按 Unicode 码点分割 tokens
    subwords, subword_tokens_list, subword_indices_list = _split_tokens_on_unicode(tokenizer, tokens)
    words = []
    word_tokens = []
    token_indices = []
    # 遍历三个列表：subwords, subword_tokens_list, subword_indices_list，同时迭代获取对应的元素
    for subword, subword_tokens, subword_indices in zip(subwords, subword_tokens_list, subword_indices_list):
        # 检查当前子词的第一个标记是否大于或等于tokenizer.eos_token_id，判断是否为特殊标记
        special = subword_tokens[0] >= tokenizer.eos_token_id
        # 检查当前子词是否以空格开头
        with_space = subword.startswith(" ")
        # 检查当前子词去除两端空白后是否是标点符号
        punctuation = subword.strip() in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

        # 如果满足特殊标记、以空格开头、是标点符号或者words列表为空，则将当前子词加入words列表，以及相应的标记和索引
        if special or with_space or punctuation or len(words) == 0:
            words.append(subword)
            word_tokens.append(subword_tokens)
            token_indices.append(subword_indices)
        # 否则，将当前子词连接到words列表的最后一个元素上，并扩展相应的标记和索引
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)
            token_indices[-1].extend(subword_indices)

    # 返回处理后的words列表、word_tokens列表和token_indices列表
    return words, word_tokens, token_indices
# 合并标点符号与相邻单词
def _merge_punctuations(words, tokens, indices, prepended, appended):
    # 在单词列表末尾添加标点符号
    i = len(words) - 2
    j = len(words) - 1
    while i >= 0:
        # 如果前一个单词以空格开头且在预定义的前置标点符号列表中
        if words[i].startswith(" ") and words[i].strip() in prepended:
            # 将当前标点符号与前一个单词合并
            words[j] = words[i] + words[j]
            tokens[j] = tokens[i] + tokens[j]
            indices[j] = indices[i] + indices[j]
            # 清空前一个单词的内容，以及对应的 tokens 和 indices
            words[i] = ""
            tokens[i] = []
            indices[i] = []
        else:
            j = i
        i -= 1

    # 在单词列表开头添加标点符号
    i = 0
    j = 1
    while j < len(words):
        # 如果当前单词不以空格结尾且在预定义的后置标点符号列表中
        if not words[i].endswith(" ") and words[j] in appended:
            # 将当前标点符号与前一个单词合并
            words[i] += words[j]
            tokens[i] += tokens[j]
            indices[i] += indices[j]
            # 清空当前单词的内容，以及对应的 tokens 和 indices
            words[j] = ""
            tokens[j] = []
            indices[j] = []
        else:
            i = j
        j += 1

    # 移除现在为空的元素
    words[:] = [word for word in words if word]
    tokens[:] = [token for token in tokens if token]
    indices[:] = [idx for idx in indices if idx]
```