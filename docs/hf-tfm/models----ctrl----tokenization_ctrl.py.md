# `.\models\ctrl\tokenization_ctrl.py`

```
# 设置编码格式为 UTF-8
# 版权声明
# 根据 Apache 许可证版本 2.0 授权
# 获取许可证的网址
# 如果根据适用法律需要或书面同意，需要满足许可证
# 根据许可证分发的软件基于“现状”分发，没有任何明示或暗示的担保或条件
# 请查看许可证以了解特定语言管理权限和限制
"""Salesforce Ctrl""" 的 Tokenization 类
导入JSON、OS、可选和元祖类型
导入正则表达式作为 re
导入 tokenization_utils 中的 PreTrainedTokenizer 和 logging 中的工具
得到 logger 工具
词汇文件名，包括词汇表文件和合并文件
预训练词汇文件映射
预训练位置嵌入大小
控制代码映射
定义一个函数用来返回单词的符号对
定义一个 CTRLTokenizer 类，继承 PreTrainedTokenizer
    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """



    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    control_codes = CONTROL_CODES



    def __init__(self, vocab_file, merges_file, unk_token="<unk>", **kwargs):
        # 打开词汇文件，使用utf-8编码读取，并加载为编码器
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建词汇解码器
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 打开融合文件，使用utf-8编码读取后分割为列表
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[1:-1]
        # 解析融合元组并映射为融合等级
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # 创建缓存
        self.cache = {}
        # 调用父类初始化方法
        super().__init__(unk_token=unk_token, **kwargs)



    @property
    def vocab_size(self):
        return len(self.encoder)



    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)



    def bpe(self, token):
        # 如果token已在缓存中，则直接返回缓存中的值
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        word = tuple(list(word[:-1]) + [word[-1] + "</w>"])
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
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
        word = "@@ ".join(word)
        word = word[:-4]
        self.cache[token] = word
        return word
    # 将字符串进行分词处理
    def _tokenize(self, text):
        """Tokenize a string."""
        split_tokens = []

        # 使用正则表达式找出非空白字符的序列
        words = re.findall(r"\S+\n?", text)

        # 对每个 token 进行 BPE 处理，并扩展到 split_tokens 列表中
        for token in words:
            split_tokens.extend(list(self.bpe(token).split(" ")))
        # 返回分词后的结果
        return split_tokens

    # 将 token 转换为对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将 id 转换为对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    # 将 tokens 转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将 tokens 组合成字符串，并去除特殊标记
        out_string = " ".join(tokens).replace("@@ ", "").strip()
        return out_string

    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将 encoder 写入到文件中
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 将 BPE merges 写入到文件中
        index = 0
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

    # 将 token ids 解码为字符串
    # def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
    #     filtered_tokens = ' '.join(self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens))
    #     tokens_generated_so_far = re.sub('(@@ )', '', string=filtered_tokens)
    #     tokens_generated_so_far = re.sub('(@@ ?$)', '', string=tokens_generated_so_far)
    #     return ''.join(tokens_generated_so_far)
```