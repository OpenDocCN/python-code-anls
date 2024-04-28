# `.\models\code_llama\tokenization_code_llama.py`

```py
# 定义系统起始和结束标记
B_SYS, E_SYS = "<<SYS>>\n", "\n\n"

# 定义默认的系统提示信息
# fmt: off
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
# fmt: on
    # 初始化函数，接收各种参数来配置分词器
    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        prefix_token="▁<PRE>",
        middle_token="▁<MID>",
        suffix_token="▁<SUF>",
        eot_token="▁<EOT>",
        fill_token="<FILL_ME>",
        suffix_first=False,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        additional_special_tokens=None,
        use_default_system_prompt=False,
        **kwargs,
    ):
        # 检查是否安装了 protobuf 后端
        requires_backends(self, "protobuf")
        # 如果没有提供 sp_model_kwargs，则设置为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # 如果传入的 bos_token 是字符串，则封装成 AddedToken 对象
        bos_token = AddedToken(bos_token, normalized=False, special=True) if isinstance(bos_token, str) else bos_token
        # 如果传入的 eos_token 是字符串，则封装成 AddedToken 对象
        eos_token = AddedToken(eos_token, normalized=False, special=True) if isinstance(eos_token, str) else eos_token
        # 如果传入的 unk_token 是字符串，则封装成 AddedToken 对象
        unk_token = AddedToken(unk_token, normalized=False, special=True) if isinstance(unk_token, str) else unk_token

        # 设置是否使用默认系统提示
        self.use_default_system_prompt = use_default_system_prompt
        # 标记特殊的标记以跳过它们
        additional_special_tokens = additional_special_tokens or []
        for token in [prefix_token, middle_token, suffix_token, eot_token]:
            additional_special_tokens += [token] if token is not None else []

        # 保存参数到实例变量
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self._prefix_token = prefix_token
        self._middle_token = middle_token
        self._suffix_token = suffix_token
        self._eot_token = eot_token
        self.fill_token = fill_token
        self.suffix_first = suffix_first
        # 获取 SentencePiece 分词器
        self.sp_model = self.get_spm_processor()

        # 调用父类的初始化方法，传入相应参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            prefix_token=prefix_token,
            middle_token=middle_token,
            suffix_token=suffix_token,
            eot_token=eot_token,
            fill_token=fill_token,
            sp_model_kwargs=self.sp_model_kwargs,
            suffix_first=suffix_first,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            additional_special_tokens=additional_special_tokens,
            use_default_system_prompt=use_default_system_prompt,
            **kwargs,
        )

    # 获取 unk_token 的长度
    @property
    def unk_token_length(self):
        return len(self.sp_model.encode(str(self.unk_token)))
    # 获取 spm 处理器
    def get_spm_processor(self):
        # 使用参数初始化 spm.SentencePieceProcessor 对象
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 从 vocab_file 中读取 sp_model 数据
        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            # 导入 protobuf 模块中的 import_protobuf 函数
            model_pb2 = import_protobuf()
            # 使用 sp_model 字节流创建 model_pb2.ModelProto 对象
            model = model_pb2.ModelProto.FromString(sp_model)
            # 创建 normalizer_spec 对象
            normalizer_spec = model_pb2.NormalizerSpec()
            # 设置 normalizer_spec 属性
            normalizer_spec.add_dummy_prefix = False
            # 合并 normalizer_spec 到 model 的 normalizer_spec 属性
            model.normalizer_spec.MergeFrom(normalizer_spec)
            # 将 model 序列化为字节流
            sp_model = model.SerializeToString()
            # 从序列化的 proto 字符串中加载 tokenizer
            tokenizer.LoadFromSerializedProto(sp_model)
        # 返回 tokenizer 对象
        return tokenizer
    
    # 获取 prefix_token 属性
    @property
    def prefix_token(self):
        return self._prefix_token
    
    # 获取 prefix_id 属性
    @property
    def prefix_id(self):
        if self._prefix_token is None:
            return None
        return self.convert_tokens_to_ids(self.prefix_token)
    
    # 获取 middle_token 属性
    @property
    def middle_token(self):
        return self._middle_token
    
    # 获取 middle_id 属性
    @property
    def middle_id(self):
        if self._middle_token is None:
            return None
        return self.convert_tokens_to_ids(self.middle_token)
    
    # 获取 suffix_token 属性
    @property
    def suffix_token(self):
        return self._suffix_token
    
    # 获取 suffix_id 属性
    @property
    def suffix_id(self):
        if self._suffix_token is None:
            return None
        return self.convert_tokens_to_ids(self.suffix_token)
    
    # 获取 eot_token 属性
    @property
    def eot_token(self):
        return self._eot_token
    
    # 获取 eot_id 属性
    @property
    def eot_id(self):
        if self._eot_token is None:
            return None
        return self.convert_tokens_to_ids(self.eot_token)
    
    # 获取 vocab_size 属性
    @property
    def vocab_size(self):
        """返回词汇表大小"""
        return self.sp_model.get_piece_size()
    
    # 从 transformers.models.llama.tokenization_llama.LlamaTokenizer.get_vocab 复制的方法
    def get_vocab(self):
        """返回词汇表"""
        # 创建字典 vocab，并将词汇表索引映射到词汇的对应关系
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 更新 vocab 字典，将 self.added_tokens_encoder 中的数据添加到 vocab 中
        vocab.update(self.added_tokens_encoder)
        # 返回 vocab 字典
        return vocab
    def tokenize(self, prefix, suffix=None, suffix_first=False, **kwargs) -> List[int]:
        # 对prefix进行分词，返回token的索引列表
        # 如果填充标记不为空，并且填充标记出现在prefix中并且suffix为空
        if self.fill_token is not None and self.fill_token in prefix and suffix is None:
            # 用填充标记分隔prefix和suffix
            prefix, suffix = prefix.split(self.fill_token)

        # 如果prefix的长度大于0
        if len(prefix) > 0:
            # 在prefix前添加一个空格并用下划线替换前缀中的特殊标记
            prefix = SPIECE_UNDERLINE + prefix.replace(SPIECE_UNDERLINE, " ")

        # 如果suffix为空或长度小于1
        if suffix is None or len(suffix) < 1:
            # 对prefix进行分词
            tokens = super().tokenize(prefix, **kwargs)
            # 如果tokens长度大于1并且第一个token是下划线并且第二个token在所有特殊token里
            if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
                # 从第一个token开始返回tokens列表
                tokens = tokens[1:]
            return tokens

        # 把prefix进行分词，前缀有一个额外的`SPIECE_UNDERLINE`
        prefix_tokens = self._tokenize(prefix)

        # 如果self.prefix_id、self.middle_id、self.suffix_id有任何一个是None
        if None in (self.prefix_id, self.middle_id, self.suffix_id):
            # 抛出数值错误异常
            raise ValueError(
                "The input either includes a `prefix` and a `suffix` used for the infilling task,"
                f"  or can be split on the {self.fill_token} token, creating a suffix and prefix,"
                " but the model does not support `infilling`."
            )
        # 把suffix进行分词，确保CodeLlama sp模型不会出问题
        suffix_tokens = self._tokenize(suffix)

        # 如果suffix_first不为None，并且suffix_first为真
        if suffix_first:
            # 格式化为" <PRE> <SUF>{suf} <MID> {pre}"
            return [self.prefix_token, self.suffix_token] + suffix_tokens + [self.middle_token] + prefix_tokens
        else:
            # 格式化为" <PRE> {pre} <SUF>{suf} <MID>"
            return [self.prefix_token] + prefix_tokens + [self.suffix_token] + suffix_tokens + [self.middle_token]

    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        # 对文本进行分词
        tokens = self.sp_model.encode(text, out_type=str)
        # 如果文本不是以SPIECE_UNDERLINE或空格开头
        if not text.startswith((SPIECE_UNDERLINE, " ")):
            # 返回tokens列表
            return tokens
        # 1. 编码字符串+前缀，例如: "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. 从 ['<','unk','>', '▁Hey'] 中移除unk_token
        return tokens[self.unk_token_length :] if len(tokens) >= self.unk_token_length else tokens

    # 从transformers.models.llama.tokenization_llama.LlamaTokenizer._convert_token_to_id复制而来
    def _convert_token_to_id(self, token):
        """将token（字符串）转换为id使用词汇表。"""
        return self.sp_model.piece_to_id(token)
    # 从transformers.models.llama.tokenization_llama.LlamaTokenizer._convert_id_to_token复制而来，将索引（整数）转换为标记（字符串）使用字典
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用vocab将索引转换为标记
        token = self.sp_model.IdToPiece(index)
        return token

    # 将标记序列（字符串）转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 因为我们手动添加了前缀空格，所以在解码时必须将其删除
        if tokens[0].startswith(SPIECE_UNDERLINE):
            tokens[0] = tokens[0][1:]

        current_sub_tokens = []
        out_string = ""
        for _, token in enumerate(tokens):
            # 确保特殊标记不使用sentencepiece模型进行解码
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    # 从transformers.models.llama.tokenization_llama.LlamaTokenizer.save_vocabulary复制而来，将词汇表和特殊标记文件保存到目录中
    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    # 从transformers.models.llama.tokenization_llama.LlamaTokenizer.build_inputs_with_special_tokens复制而来，构建带有特殊标记的输入
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    # 从transformers.models.llama.tokenization_llama.LlamaTokenizer.get_special_tokens_mask
    # 定义一个方法，用于获取没有添加特殊令牌的令牌列表中的序列 ID。当使用分词器的 `prepare_for_model` 方法添加特殊令牌时，会调用该方法。
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
            `List[int`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        # 如果已经存在特殊令牌，则直接返回特殊令牌掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
    
        bos_token_id = [1] if self.add_bos_token else []  # 如果存在 bos 令牌，则用 [1] 表示，否则为空列表
        eos_token_id = [1] if self.add_eos_token else []  # 如果存在 eos 令牌，则用 [1] 表示，否则为空列表
    
        # 如果 token_ids_1 为空，则只返回第一个序列的特殊令牌掩码
        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        # 否则，返回两个序列的特殊令牌掩码
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )
    
    # 从序列中创建令牌类型 ID，用于用于序列对分类任务
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:
    
        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```py
    
        if token_ids_1 is None, only returns the first portion of the mask (0s).
    
        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
    
        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []  # 如果存在 bos 令牌，则用 [bos_token_id] 表示，否则为空列表
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []  # 如果存在 eos 令牌，则用 [eos_token_id] 表示，否则为空列表
    
        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)  # 创建初始输出，长度为 bos_token_id + token_ids_0 + eos_token_id 的长度，值均为 0
    
        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)  # 如果存在 token_ids_1，则将输出扩展，长度为 bos_token_id + token_ids_1 + eos_token_id 的长度，值均为 1
    
        return output  # 返回创建的令牌类型 ID 列表
    
    @property
    # 从 transformers.models.llama.tokenization_llama.LlamaTokenizer.default_chat_template 复制过来的
    # 定义对象的状态获取方法
    def __getstate__(self):
        # 复制对象的字典状态
        state = self.__dict__.copy()
        # 将特定属性设置为 None
        state["sp_model"] = None
        # 将特定属性设置为序列化的模型协议
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        # 返回状态
        return state

    # 定义对象的状态设置方法
    def __setstate__(self, d):
        # 用给定的字典状态来设置对象的字典属性
        self.__dict__ = d
        # 根据特定的参数重新创建 sp_model 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 从序列化协议中加载 sp_model 对象
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)
```