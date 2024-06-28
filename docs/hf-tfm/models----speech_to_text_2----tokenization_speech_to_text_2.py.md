# `.\models\speech_to_text_2\tokenization_speech_to_text_2.py`

```
# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",  # 词汇表文件名
    "tokenizer_config_file": "tokenizer_config.json",  # 分词器配置文件名
    "merges_file": "merges.txt",  # 分词器合并文件名
}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/s2t-wav2vec2-large-en-de": (
            "https://huggingface.co/facebook/s2t-wav2vec2-large-en-de/resolve/main/vocab.json"
        ),
    },
    "tokenizer_config_file": {
        "facebook/s2t-wav2vec2-large-en-de": (
            "https://huggingface.co/facebook/s2t-wav2vec2-large-en-de/resolve/main/tokenizer_config.json"
        ),
    },
    "merges_file": {
        "facebook/s2t-wav2vec2-large-en-de": (
            "https://huggingface.co/facebook/s2t-wav2vec2-large-en-de/resolve/main/merges.txt"
        ),
    },
}

# BPE（字节对编码）模型的特殊标记
BPE_TOKEN_MERGES = "</w>"
BPE_TOKEN_VOCAB = "@@ "

def get_pairs(word):
    """
    返回单词中的符号对集合。单词被表示为符号（符号是长度可变的字符串）的元组。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

# Speech2Text2 模型没有最大输入长度限制
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/s2t-wav2vec2-large-en-de": 1024}

class Speech2Text2Tokenizer(PreTrainedTokenizer):
    """
    构建 Speech2Text2Tokenizer。

    该分词器继承自 `PreTrainedTokenizer`，其中包含一些主要方法。用户应参考超类以获取有关这些方法的更多信息。
    """
    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sentence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sentence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.

        **kwargs
            Additional keyword arguments passed along to `PreTrainedTokenizer`.

    ```
    # 定义预训练模型的词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义预训练模型的词汇文件映射表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义预训练模型的最大输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        do_lower_case=False,
        merges_file=None,
        **kwargs,
    ):
        # 初始化时设置是否小写化标记
        self.do_lower_case = do_lower_case

        # 从文件中加载词汇表到编码器中
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建解码器，将编码器中的键值对颠倒
        self.decoder = {v: k for k, v in self.encoder.items()}

        # 如果没有提供合并文件，记录日志并设置相关属性为None
        if merges_file is None:
            logger.info(f"No merges files provided. {self.__class__.__name__} can only be used for decoding.")
            self.bpe_ranks = None
            self.cache = None
        else:
            # 从文件中读取并处理合并规则，创建BPE合并的排名字典和缓存
            with open(merges_file, encoding="utf-8") as merges_handle:
                merges = merges_handle.read().split("\n")[:-1]

            merges = [tuple(merge.split()[:2]) for merge in merges]
            self.bpe_ranks = dict(zip(merges, range(len(merges))))
            self.cache = {}

        # 调用父类的初始化方法，传递必要的参数和额外的关键字参数
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        # 返回词汇表大小
        return len(self.decoder)

    def get_vocab(self) -> Dict:
        # 返回编码器和额外添加编码器合并后的词汇表
        return dict(self.encoder, **self.added_tokens_encoder)
    def bpe(self, token):
        # 构建包含特殊 BPE 标记的元组 word
        word = tuple(token[:-1]) + (token[-1] + BPE_TOKEN_MERGES,)
        # 如果 token 已经被缓存，直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 获取 token 的所有可能的字符对
        pairs = get_pairs(word)

        # 如果没有字符对，直接返回 token
        if not pairs:
            return token

        # 开始 BPE 算法的主循环
        while True:
            # 选择当前权重最小的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果选择的字符对不在 bpe_ranks 中，则跳出循环
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 根据字符对重组单词
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
            # 如果单词长度为 1，则结束循环
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        # 将单词转换为字符串形式
        word = " ".join(word)
        # 处理特殊情况下的 BPE 标记
        if word == "\n  " + BPE_TOKEN_MERGES:
            word = "\n" + BPE_TOKEN_MERGES

        if word.endswith(BPE_TOKEN_MERGES):
            word = word.replace(BPE_TOKEN_MERGES, "")

        word = word.replace(" ", BPE_TOKEN_VOCAB)
        # 将处理后的单词缓存起来
        self.cache[token] = word
        # 返回处理后的单词
        return word

    def _tokenize(self, text):
        """Tokenize a string."""

        # 检查是否有 BPE 等级文件
        if self.bpe_ranks is None:
            raise ValueError(
                "This tokenizer was instantiated without a `merges.txt` file, so"
                " that it can only be used for decoding, not for encoding. "
                "Make sure to provide `merges.txt` file at instantiation to enable "
                "encoding."
            )

        # 如果设置为小写，则将文本转换为小写
        if self.do_lower_case:
            text = text.lower()

        # 拆分文本为单词列表
        text = text.split()

        split_tokens = []
        # 对每个单词进行 BPE 分词处理
        for token in text:
            if token:
                split_tokens.extend(list(self.bpe(token).split(" ")))

        # 返回处理后的分词列表
        return split_tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an index (integer) using the vocab."""
        # 使用词汇表将 token 转换为对应的 ID
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用词汇表将 ID 转换为对应的 token
        result = self.decoder.get(index, self.unk_token)
        return result

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a list of output tokens into a single string.
        """
        # 将 tokens 列表合并成一个字符串
        string = " ".join(tokens)

        # 确保 @@ 标记被正确连接
        string = "".join(string.split(BPE_TOKEN_VOCAB))

        # 返回合并后的字符串
        return string
    # 保存词汇表和词汇合并文件到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构造词汇表文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        
        # 构造词汇合并文件路径
        merges_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将编码器的内容写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 初始化索引为0
        index = 0
        # 如果词分片（bpe_ranks）为空，则返回仅包含词汇表文件路径的元组
        if self.bpe_ranks is None:
            return (vocab_file,)

        # 打开词汇合并文件，以UTF-8编码写入内容
        with open(merges_file, "w", encoding="utf-8") as writer:
            # 对self.bpe_ranks按token_index进行排序，并迭代处理
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                # 如果索引不等于token_index，则记录警告信息
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merges_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                # 将BPE标记写入合并文件
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回包含词汇表文件路径和词汇合并文件路径的元组
        return (vocab_file, merges_file)
```