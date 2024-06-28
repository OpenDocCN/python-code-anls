# `.\models\deprecated\transfo_xl\tokenization_transfo_xl.py`

```
# 设置编码格式为 UTF-8
# 版权声明和许可信息，详细说明了使用该代码的条件和限制
# 导入所需的模块和库
# 引入正则表达式、计数器和有序字典等工具
# 导入 NumPy 库，并命名为 np

# 如果 sacremoses 可用，则导入 sacremoses 库
# 如果 Torch 可用，则导入 Torch 库
# 获取 logging 模块的记录器对象，并命名为 logger

# 定义词汇文件名的映射
# 定义预训练词汇文件的映射
# 定义预训练位置嵌入的大小映射
# 定义预训练语料库的映射和语料库名称

# 定义匹配数字的正则表达式和替换方式的元组
# 定义将已经被 token 化的数字进行重新组合的规则
    # 对给定的文本进行数字解标记化处理
    for reg, sub in DETOKENIZE_NUMBERS:
        # 使用正则表达式替换文本中匹配到的模式为指定的替换字符串
        text = re.sub(reg, sub, text)
    # 返回处理后的文本
    return text
class TransfoXLTokenizer(PreTrainedTokenizer):
    """
    Construct a Transformer-XL tokenizer adapted from Vocab class in [the original
    code](https://github.com/kimiyoung/transformer-xl). The Transformer-XL tokenizer is a word-level tokenizer (no
    sub-word tokenization).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        special (`List[str]`, *optional*):
            A list of special tokens (to be treated by the original implementation of this tokenizer).
        min_freq (`int`, *optional*, defaults to 0):
            The minimum number of times a token has to be present in order to be kept in the vocabulary (otherwise it
            will be mapped to `unk_token`).
        max_size (`int`, *optional*):
            The maximum size of the vocabulary. If left unset, it will default to the size of the vocabulary found
            after excluding the tokens according to the `min_freq` rule.
        lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        delimiter (`str`, *optional*):
            The delimiter used between tokens.
        vocab_file (`str`, *optional*):
            File containing the vocabulary (from the original implementation).
        pretrained_vocab_file (`str`, *optional*):
            File containing the vocabulary as saved with the `save_pretrained()` method.
        never_split (`List[str]`, *optional*):
            List of tokens that should never be split. If no list is specified, will simply use the existing special
            tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (`str`, *optional*, defaults to `"<eos>"`):
            The end of sequence token.
        additional_special_tokens (`List[str]`, *optional*, defaults to `['<formula>']`):
            A list of additional special tokens (for the HuggingFace functionality).
        language (`str`, *optional*, defaults to `"en"`):
            The language of this tokenizer (used for mose preprocessing).
    """

    # 定义一些类属性，用于管理词汇文件和模型输入的最大尺寸
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids"]

    def __init__(
        self,
        special=None,
        min_freq=0,
        max_size=None,
        lower_case=False,
        delimiter=None,
        vocab_file=None,
        pretrained_vocab_file: str = None,
        never_split=None,
        unk_token="<unk>",
        eos_token="<eos>",
        additional_special_tokens=["<formula>"],
        language="en",
        **kwargs,
    ):
        # 初始化方法，用于创建一个新的Tokenizer对象
        # 参数说明如上述文档所述
        super().__init__(
            unk_token=unk_token,
            eos_token=eos_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    # 定义一个属性方法，用于访问类属性
    # 返回当前对象的小写字母状态
    def do_lower_case(self):
        return self.lower_case

    # 编译用于匹配标点符号周围空格的正则表达式模式
    def _compile_space_around_punctuation_pattern(self):
        # 创建正向预查以匹配特殊标点符号之前的位置
        look_ahead_for_special_token = f"(?=[{self.punctuation_symbols}])"
        # 创建正向预查以匹配除空格外所有字符之前的位置
        look_ahead_to_match_all_except_space = r"(?=[^\s])"
        # 返回编译后的正则表达式模式对象
        return re.compile(r"" + look_ahead_for_special_token + look_ahead_to_match_all_except_space)

    # 统计文件中的符号并返回句子列表
    def count_file(self, path, verbose=False, add_eos=False):
        # 如果启用详细模式，则记录文件计数过程
        if verbose:
            logger.info(f"counting file {path} ...")
        # 断言文件路径存在，否则抛出异常
        assert os.path.exists(path), f"Input file {path} not found"

        # 初始化句子列表
        sents = []
        # 打开文件进行读取
        with open(path, "r", encoding="utf-8") as f:
            # 逐行读取文件内容
            for idx, line in enumerate(f):
                # 如果启用详细模式并且达到指定的行数间隔，则记录当前行数
                if verbose and idx > 0 and idx % 500000 == 0:
                    logger.info(f"    line {idx}")
                # 对当前行进行符号化处理，可选择在末尾添加结束符
                symbols = self.tokenize(line, add_eos=add_eos)
                # 更新符号计数器
                self.counter.update(symbols)
                # 将处理后的符号列表添加到句子列表中
                sents.append(symbols)

        # 返回处理后的句子列表
        return sents

    # 统计符号列表中的符号
    def count_sents(self, sents, verbose=False):
        """
        sents : a list of sentences, each a list of tokenized symbols
        """
        # 如果启用详细模式，则记录句子计数过程
        if verbose:
            logger.info(f"counting {len(sents)} sents ...")
        # 遍历句子列表并统计符号
        for idx, symbols in enumerate(sents):
            # 如果启用详细模式并且达到指定的行数间隔，则记录当前行数
            if verbose and idx > 0 and idx % 500000 == 0:
                logger.info(f"    line {idx}")
            # 更新符号计数器
            self.counter.update(symbols)

    # 从文件构建词汇表
    def _build_from_file(self, vocab_file):
        # 初始化索引到符号的映射列表
        self.idx2sym = []
        # 初始化符号到索引的有序字典
        self.sym2idx = OrderedDict()

        # 打开词汇文件进行读取
        with open(vocab_file, "r", encoding="utf-8") as f:
            # 逐行读取文件内容
            for line in f:
                # 剥离行末尾的空白字符并按空格分割取第一个符号
                symb = line.strip().split()[0]
                # 添加符号到词汇表中
                self.add_symbol(symb)
        # 如果词汇表中存在"<UNK>"符号，则设置其索引为unk_idx
        if "<UNK>" in self.sym2idx:
            self.unk_idx = self.sym2idx["<UNK>"]
        # 否则如果词汇表中存在"<unk>"符号，则设置其索引为unk_idx
        elif "<unk>" in self.sym2idx:
            self.unk_idx = self.sym2idx["<unk>"]
        # 否则抛出异常，表示找不到用于替换的未知符号
        else:
            raise ValueError("Token not in vocabulary and no <unk> token in vocabulary for replacement.")

    # 保存词汇表到指定目录中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录已存在，则设置词汇文件路径
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["pretrained_vocab_file"],
            )
        # 否则设置词汇文件路径为指定的文件名前缀
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开词汇文件并使用pickle将当前对象的字典形式保存到文件中
        with open(vocab_file, "wb") as f:
            pickle.dump(self.__dict__, f)
        # 返回保存的词汇文件路径
        return (vocab_file,)
    # 构建词汇表的方法
    def build_vocab(self):
        # 如果指定了词汇文件，从文件中构建词汇表
        if self.vocab_file:
            logger.info(f"building vocab from {self.vocab_file}")
            self._build_from_file(self.vocab_file)
            logger.info(f"Final vocab size {len(self.sym2idx)}")
        else:
            # 如果没有指定词汇文件，根据设定的最小频率和最大大小构建词汇表
            logger.info(f"building vocab with min_freq={self.min_freq}, max_size={self.max_size}")
            # 初始化索引到符号的列表和符号到索引的有序字典
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            # 添加特殊符号到词汇表中
            for sym in self.special:
                self.add_special(sym)

            # 根据计数器中的频率最高的符号构建词汇表，直到达到最大大小或者低于最小频率
            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq:
                    break
                self.add_symbol(sym)

            # 打印构建后的词汇表大小和原始符号的唯一标记数
            logger.info(f"Final vocab size {len(self.sym2idx)} from {len(self.counter)} unique tokens")

    # 使用 PyTorch 的方法进行文件编码
    @torch_only_method
    def encode_file(self, path, ordered=False, verbose=False, add_eos=True, add_double_eos=False):
        # 如果启用详细模式，记录编码文件的信息
        if verbose:
            logger.info(f"encoding file {path} ...")
        # 断言检查路径是否存在
        assert os.path.exists(path), f"Output file {path} not found"
        # 初始化编码结果列表
        encoded = []
        # 打开文件并逐行处理
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                # 如果启用详细模式并且处理了一定数量的行数，记录当前处理的行数
                if verbose and idx > 0 and idx % 500000 == 0:
                    logger.info(f"    line {idx}")
                # 对每一行进行分词和编码成张量，并添加到编码结果列表中
                symbols = self.tokenize(line, add_eos=add_eos, add_double_eos=add_double_eos)
                encoded.append(self.convert_to_tensor(symbols))

        # 如果需要按顺序返回编码结果，将所有张量拼接成一个张量
        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    # 使用 PyTorch 的方法进行句子编码
    @torch_only_method
    def encode_sents(self, sents, ordered=False, verbose=False):
        # 如果启用详细模式，记录编码句子的信息
        if verbose:
            logger.info(f"encoding {len(sents)} sents ...")
        # 初始化编码结果列表
        encoded = []
        # 遍历每一个句子进行编码
        for idx, symbols in enumerate(sents):
            # 如果启用详细模式并且处理了一定数量的句子，记录当前处理的句子数
            if verbose and idx > 0 and idx % 500000 == 0:
                logger.info(f"    line {idx}")
            # 将每个句子的符号序列转换成张量，并添加到编码结果列表中
            encoded.append(self.convert_to_tensor(symbols))

        # 如果需要按顺序返回编码结果，将所有张量拼接成一个张量
        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    # 向词汇表中添加特殊符号
    def add_special(self, sym):
        # 如果符号不在词汇表中，则将其添加到词汇表中，并为其设置索引属性
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, f"{sym.strip('<>')}_idx", self.sym2idx[sym])

    # 向词汇表中添加普通符号
    def add_symbol(self, sym):
        # 如果符号不在词汇表中，则将其添加到词汇表中，并为其设置索引属性
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
    def move_added_token(self, token: str, target_idx: int):
        """
        Moves an added token to a specific position in the vocab. This method should be used when resizing an embedding
        layer other than the last one in the `AdaptiveEmbedding` in order to move the token in the tokenizer from the
        default position (at the very end) to the desired one.

        Args:
            token: The token to move to a specific position in the vocab.
            target_idx: The position where the token should be moved to.
        """
        # 确保要移动的 token 是已添加的 token
        assert token in self.added_tokens_encoder, "Token which should be moved has to be an added token"
        # 确保要移动的 token 不在词汇表中
        assert token not in self.idx2sym, "Token which should be moved is already in vocab"

        # 将 token 插入到目标位置
        self.idx2sym.insert(target_idx, token)
        # 更新 token 对应的索引位置
        self.sym2idx[token] = target_idx

        # 调整后续 token 在 sym2idx 中的索引位置
        for idx in range(target_idx + 1, len(self.idx2sym)):
            current_sym = self.idx2sym[idx]
            self.sym2idx[current_sym] = idx

        # 从 added_tokens 中删除 token
        old_index = self._added_tokens_encoder.pop(token)
        self._added_tokens_decoder.pop(old_index)

    def moses_punct_norm(self, text):
        """
        Normalize punctuation in the text using MosesPunctNormalizer.

        Args:
            text: Input text to be normalized.

        Returns:
            Normalized text with standardized punctuation.
        """
        return self.moses_punct_normalizer.normalize(text)

    def moses_tokenize(self, text):
        """
        Tokenizes text using MosesTokenizer.

        Args:
            text: Input text to be tokenized.

        Returns:
            List of tokens extracted from the input text.
        """
        return self.moses_tokenizer.tokenize(
            text, aggressive_dash_splits=True, return_str=False, escape=False, protected_patterns=self.never_split
        )

    def moses_pipeline(self, text: str) -> List[str]:
        """
        Performs a pipeline of basic text preprocessing tasks using MosesPunctNormalizer and MosesTokenizer. Also handles
        splitting of large comma-separated numbers and floating point values.

        Args:
            text: Text to be tokenized and preprocessed.

        Returns:
            A list of tokenized strings.
        """
        text = self.moses_punct_norm(text)
        text = self.moses_tokenize(text)
        text = tokenize_numbers(text)  # Assuming tokenize_numbers is defined elsewhere
        return text

    def _convert_id_to_token(self, idx):
        """
        Converts an index to a token using the vocabulary.

        Args:
            idx: Index to be converted into a token.

        Returns:
            Corresponding token based on the index.
        """
        assert 0 <= idx < len(self), f"Index {idx} out of vocabulary range"
        return self.idx2sym[idx]
    def _convert_token_to_id(self, sym):
        """Converts a token (str) into an id using the vocabulary."""
        # 如果符号在符号到索引的映射中存在，则返回其对应的索引
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # 如果符号不在映射中
            # logger.info(f'encounter unk {sym}')
            # assert '<eos>' not in sym
            # 如果对象具有 unk_idx 属性，则返回其映射值，否则根据预训练模型的后向兼容性返回默认的未知标记索引
            if hasattr(self, "unk_idx"):
                return self.sym2idx.get(sym, self.unk_idx)
            elif "<unk>" in self.sym2idx:
                return self.sym2idx["<unk>"]
            elif "<UNK>" in self.sym2idx:
                return self.sym2idx["<UNK>"]
            else:
                # 如果符号既不在映射中，也没有未知标记，则抛出异常
                raise ValueError("Token not in vocabulary and no <unk> token in vocabulary for replacement.")

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (string) into a single string.
        Additionally, converts split numbers back to their original form.
        """
        # 使用 Moses detokenizer 将 tokens 转换为字符串
        out_string = self.moses_detokenizer.detokenize(tokens)
        # 对字符串中的数字进行反转识别并返回处理后的字符串
        return detokenize_numbers(out_string).strip()

    @torch_only_method
    def convert_to_tensor(self, symbols):
        """Converts a list of symbols into a PyTorch tensor of Long type."""
        return torch.LongTensor(self.convert_tokens_to_ids(symbols))

    @property
    def vocab_size(self):
        """Returns the size of the vocabulary."""
        return len(self.idx2sym)

    def get_vocab(self):
        """Returns a dictionary containing the entire vocabulary."""
        vocab = self.sym2idx.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, line, add_eos=False, add_double_eos=False):
        """
        Tokenizes a line of text with optional end-of-sequence tokens.
        """
        # 去除行首尾空白字符
        line = line.strip()
        # 若需要，将字符串转换为小写
        if self.lower_case:
            line = line.lower()

        # 如果分隔符为空，则直接使用整个行作为 symbols
        if self.delimiter == "":
            symbols = line
        else:
            # 使用 Moses pipeline 对行进行分词处理
            symbols = self.moses_pipeline(line)

        # 根据参数决定是否添加特定的结束符号
        if add_double_eos:  # lm1b
            return ["<S>"] + symbols + ["<S>"]
        elif add_eos:
            return symbols + ["<eos>"]
        else:
            return symbols
class LMOrderedIterator(object):
    def __init__(self, data, bsz, bptt, device="cpu", ext_len=None):
        """
        data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i + 1 : i + 1 + seq_len]

        data_out = data.transpose(0, 1).contiguous().to(self.device)
        target_out = target.transpose(0, 1).contiguous().to(self.device)

        return data_out, target_out, seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.0
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()


class LMShuffledIterator(object):
    def __init__(self, data, bsz, bptt, device="cpu", ext_len=None, shuffle=False):
        """
        data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        # index iterator
        epoch_indices = np.random.permutation(len(self.data)) if self.shuffle else np.array(range(len(self.data)))

        # sentence iterator
        for idx in epoch_indices:
            yield self.data[idx]

    @torch_only_method
    def stream_iterator(self, sent_stream):
        # streams for each data in the batch
        streams = [None] * self.bsz  # 初始化一个列表，用于存储每个批次中数据的流

        data = torch.LongTensor(self.bptt, self.bsz)  # 创建一个大小为 (bptt x bsz) 的长整型张量，用于存储数据
        target = torch.LongTensor(self.bptt, self.bsz)  # 创建一个大小为 (bptt x bsz) 的长整型张量，用于存储目标

        n_retain = 0  # 初始化保留数据的数量

        while True:
            # data   : [n_retain+bptt x bsz]
            # target : [bptt x bsz]
            data[n_retain:].fill_(-1)  # 将数据张量中从索引 n_retain 开始的所有元素填充为 -1
            target.fill_(-1)  # 将目标张量中所有元素填充为 -1

            valid_batch = True  # 标志位，表示当前批次是否有效

            for i in range(self.bsz):
                n_filled = 0  # 初始化已填充数据的数量
                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)  # 获取下一个句子流并存储在 streams[i] 中
                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)  # 计算需要填充的新令牌数量
                        # first n_retain tokens are retained from last batch
                        data[n_retain + n_filled : n_retain + n_filled + n_new, i] = streams[i][:n_new]  # 将流中的令牌填充到数据张量中
                        target[n_filled : n_filled + n_new, i] = streams[i][1 : n_new + 1]  # 将流中的目标填充到目标张量中
                        streams[i] = streams[i][n_new:]  # 更新 streams[i]，去除已填充的令牌
                        n_filled += n_new  # 更新已填充的数量
                except StopIteration:
                    valid_batch = False  # 如果出现 StopIteration 异常，则当前批次无效
                    break

            if not valid_batch:
                return  # 如果当前批次无效，结束循环

            data_out = data.transpose(0, 1).contiguous().to(self.device)  # 转置并确保张量在内存中连续，然后移到指定设备上
            target_out = target.transpose(0, 1).contiguous().to(self.device)  # 转置并确保张量在内存中连续，然后移到指定设备上

            yield data_out, target_out, self.bptt  # 生成当前批次的数据、目标和 bptt 值

            n_retain = min(data.size(0), self.ext_len)  # 更新保留数据的数量
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]  # 将数据张量中最后 n_retain 行的数据复制到开头
            data.resize_(n_retain + self.bptt, data.size(1))  # 调整数据张量的大小，以便容纳更多的数据

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()  # 获取句子流的迭代器

        for batch in self.stream_iterator(sent_stream):
            yield batch  # 生成批次数据
class LMMultiFileIterator(LMShuffledIterator):
    def __init__(self, paths, vocab, bsz, bptt, device="cpu", ext_len=None, shuffle=False):
        self.paths = paths  # 初始化文件路径列表
        self.vocab = vocab  # 初始化词汇表对象

        self.bsz = bsz  # 批量大小
        self.bptt = bptt  # 每个时间步长
        self.ext_len = ext_len if ext_len is not None else 0  # 扩展长度，默认为0

        self.device = device  # 设备类型，默认为CPU
        self.shuffle = shuffle  # 是否随机化顺序

    def get_sent_stream(self, path):
        sents = self.vocab.encode_file(path, add_double_eos=True)  # 使用词汇表编码文件内容，并添加双端标记
        if self.shuffle:
            np.random.shuffle(sents)  # 如果需要，随机打乱句子顺序
        sent_stream = iter(sents)  # 创建句子流迭代器

        return sent_stream  # 返回句子流迭代器

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.paths)  # 如果需要，随机打乱文件路径顺序

        for path in self.paths:
            # sent_stream 是一个迭代器
            sent_stream = self.get_sent_stream(path)  # 获取当前文件路径的句子流迭代器
            for batch in self.stream_iterator(sent_stream):  # 对句子流进行批量迭代
                yield batch  # 生成批量数据


class TransfoXLCorpus(object):
    @classmethod
    @torch_only_method
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a pre-processed corpus.
        """
        vocab = TransfoXLTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)  # 从预训练模型名或路径实例化词汇表
        is_local = os.path.isdir(pretrained_model_name_or_path)  # 检查是否为本地路径
        # 如果需要，重定向到缓存
        try:
            resolved_corpus_file = cached_file(pretrained_model_name_or_path, CORPUS_NAME, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                f"Corpus '{pretrained_model_name_or_path}' was not found in corpus list"
                f" ({', '.join(PRETRAINED_CORPUS_ARCHIVE_MAP.keys())}. We assumed '{pretrained_model_name_or_path}'"
                f" was a path or url but couldn't find files {CORPUS_NAME} at this path or url."
            )
            return None  # 如果出错，返回空值
        if is_local:
            logger.info(f"loading corpus file {resolved_corpus_file}")  # 如果是本地路径，记录日志加载语料文件
        else:
            logger.info(f"loading corpus file {CORPUS_NAME} from cache at {resolved_corpus_file}")  # 否则，从缓存加载语料文件

        # 实例化语料对象
        corpus = cls(*inputs, **kwargs)
        corpus_dict = torch.load(resolved_corpus_file)  # 加载语料文件内容到字典
        for key, value in corpus_dict.items():
            corpus.__dict__[key] = value  # 将加载的内容赋值给语料对象的属性
        corpus.vocab = vocab  # 设置语料对象的词汇表属性
        if corpus.train is not None:
            corpus.train = torch.tensor(corpus.train, dtype=torch.long)  # 将训练数据转换为长整型张量
        if corpus.valid is not None:
            corpus.valid = torch.tensor(corpus.valid, dtype=torch.long)  # 将验证数据转换为长整型张量
        if corpus.test is not None:
            corpus.test = torch.tensor(corpus.test, dtype=torch.long)  # 将测试数据转换为长整型张量
        return corpus  # 返回语料对象实例

    def __init__(self, *args, **kwargs):
        self.vocab = TransfoXLTokenizer(*args, **kwargs)  # 初始化语料的词汇表
        self.dataset = None  # 初始化数据集属性为空
        self.train = None  # 初始化训练数据为空
        self.valid = None  # 初始化验证数据为空
        self.test = None  # 初始化测试数据为空
    # 构建语料库的方法，根据指定路径和数据集名称来设置语料库
    def build_corpus(self, path, dataset):
        # 将数据集名称存储到实例变量中
        self.dataset = dataset

        # 根据数据集类型执行相应的操作
        if self.dataset in ["ptb", "wt2", "enwik8", "text8"]:
            # 统计训练、验证和测试文件中的词频
            self.vocab.count_file(os.path.join(path, "train.txt"))
            self.vocab.count_file(os.path.join(path, "valid.txt"))
            self.vocab.count_file(os.path.join(path, "test.txt"))
        elif self.dataset == "wt103":
            # 对于 wt103 数据集，只统计训练文件中的词频
            self.vocab.count_file(os.path.join(path, "train.txt"))
        elif self.dataset == "lm1b":
            # 构建训练文件路径的模式，并获取匹配的文件路径列表
            train_path_pattern = os.path.join(
                path,
                "1-billion-word-language-modeling-benchmark-r13output",
                "training-monolingual.tokenized.shuffled",
                "news.en-*",
            )
            train_paths = glob.glob(train_path_pattern)
            # 在调用 build_vocab() 方法时，从文件中加载词汇表

        # 构建词汇表
        self.vocab.build_vocab()

        # 根据数据集类型编码训练、验证和测试文件
        if self.dataset in ["ptb", "wt2", "wt103"]:
            self.train = self.vocab.encode_file(os.path.join(path, "train.txt"), ordered=True)
            self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"), ordered=True)
            self.test = self.vocab.encode_file(os.path.join(path, "test.txt"), ordered=True)
        elif self.dataset in ["enwik8", "text8"]:
            self.train = self.vocab.encode_file(os.path.join(path, "train.txt"), ordered=True, add_eos=False)
            self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"), ordered=True, add_eos=False)
            self.test = self.vocab.encode_file(os.path.join(path, "test.txt"), ordered=True, add_eos=False)
        elif self.dataset == "lm1b":
            self.train = train_paths
            self.valid = self.vocab.encode_file(os.path.join(path, "valid.txt"), ordered=False, add_double_eos=True)
            self.test = self.vocab.encode_file(os.path.join(path, "test.txt"), ordered=False, add_double_eos=True)

    # 获取数据迭代器的方法，根据指定的分割（训练、验证、测试）返回相应的数据迭代器
    def get_iterator(self, split, *args, **kwargs):
        if split == "train":
            if self.dataset in ["ptb", "wt2", "wt103", "enwik8", "text8"]:
                # 使用 LMOrderedIterator 创建有序数据迭代器
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
            elif self.dataset == "lm1b":
                # 对于 lm1b 数据集，设置 shuffle 参数为 True，使用 LMMultiFileIterator 创建多文件数据迭代器
                kwargs["shuffle"] = True
                data_iter = LMMultiFileIterator(self.train, self.vocab, *args, **kwargs)
        elif split in ["valid", "test"]:
            # 获取验证或测试数据集
            data = self.valid if split == "valid" else self.test
            if self.dataset in ["ptb", "wt2", "wt103", "enwik8", "text8"]:
                # 使用 LMOrderedIterator 创建有序数据迭代器
                data_iter = LMOrderedIterator(data, *args, **kwargs)
            elif self.dataset == "lm1b":
                # 对于 lm1b 数据集，使用 LMShuffledIterator 创建打乱顺序的数据迭代器
                data_iter = LMShuffledIterator(data, *args, **kwargs)
        else:
            data_iter = None
            # 如果分割类型未识别，则抛出异常
            raise ValueError(f"Split not recognized: {split}")

        return data_iter
# 仅限于 Torch 方法的装饰器，标志着这是一个专门为 Torch 框架设计的方法
@torch_only_method
def get_lm_corpus(datadir, dataset):
    # 构建缓存文件路径
    fn = os.path.join(datadir, "cache.pt")
    # 构建另一个缓存文件路径
    fn_pickle = os.path.join(datadir, "cache.pkl")
    
    # 如果存在 cache.pt 文件
    if os.path.exists(fn):
        # 记录日志，提示正在加载缓存数据集
        logger.info("Loading cached dataset...")
        # 使用 Torch 加载 cache.pkl 文件作为数据集
        corpus = torch.load(fn_pickle)
    
    # 如果存在 cache.pt 文件（这个条件似乎重复了，因为它与上一个条件相同）
    elif os.path.exists(fn):
        # 记录日志，提示正在从 pickle 文件加载缓存数据集
        logger.info("Loading cached dataset from pickle...")
        # 如果未设置 TRUST_REMOTE_CODE 环境变量为 True，则抛出 ValueError
        if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
            raise ValueError(
                "This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially "
                "malicious. It's recommended to never unpickle data that could have come from an untrusted source, or "
                "that could have been tampered with. If you already verified the pickle data and decided to use it, "
                "you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it."
            )
        # 以二进制只读模式打开 cache.pt 文件，并使用 pickle 加载数据集
        with open(fn, "rb") as fp:
            corpus = pickle.load(fp)
    
    # 如果以上两个条件均不满足
    else:
        # 记录日志，提示正在生成指定数据集的数据
        logger.info(f"Producing dataset {dataset}...")
        kwargs = {}
        # 根据数据集类型设置不同的参数
        if dataset in ["wt103", "wt2"]:
            kwargs["special"] = ["<eos>"]
            kwargs["lower_case"] = False
        elif dataset == "ptb":
            kwargs["special"] = ["<eos>"]
            kwargs["lower_case"] = True
        elif dataset == "lm1b":
            kwargs["special"] = []
            kwargs["lower_case"] = False
            kwargs["vocab_file"] = os.path.join(datadir, "1b_word_vocab.txt")
        elif dataset in ["enwik8", "text8"]:
            # 如果数据集是 enwik8 或 text8，则不设置任何参数
            pass
        
        # 使用 TransfoXLCorpus 类构建数据集 corpus，传入指定的数据目录和参数 kwargs
        corpus = TransfoXLCorpus(datadir, dataset, **kwargs)
        # 使用 Torch 将生成的数据集 corpus 保存到 cache.pt 文件中
        torch.save(corpus, fn)

    # 返回生成或加载的数据集 corpus
    return corpus
```