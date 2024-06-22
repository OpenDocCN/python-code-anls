# `.\models\layoutlmv2\tokenization_layoutlmv2.py`

```py
# 设置编码格式为 UTF-8
# 版权声明
# 引入所需依赖库
# 设置类型别名的引入
# 引入工具类
# 日志记录
# 定义词汇文件名
# 预训练模型相关文件
# 预训练模型位置编码大小
# 预训练模型初始化配置

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    # 使用有序字典存储加载的词汇文件
    vocab = collections.OrderedDict()
    # 以 UTF-8 编码方式打开词汇文件
    with open(vocab_file, "r", encoding="utf-8") as reader:
        # 逐行读取词汇文件中的内容
        tokens = reader.readlines()
    # 遍历词汇文件中的内容，将词汇按索引加入到有序字典 vocab 中
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    # 返回加载后的词汇字典
    return vocab

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # 去除文本两端空白字符
    text = text.strip()
    # 如果文本为空，返回空列表
    if not text:
        return []
    # 以空格分割文本，得到词汇列表
    tokens = text.split()
    # 返回词汇列表
    return tokens

# 生成标点符号词典
# 查找子序列
    # 构造一个 LayoutLMv2 的分词器。基于 WordPiece。LayoutLMv2Tokenizer 可以用于将单词、单词级边界框和可选的单词标签转换为标记级的 input_ids、attention_mask、token_type_ids、bbox 和可选的标签（用于标记分类）。

    # 这个分词器继承自 PreTrainedTokenizer，其中包含大部分主要方法。用户应参考该超类获取有关这些方法的更多信息。

    # LayoutLMv2Tokenizer 运行端到端的分词：标点符号分割和wordpiece。它还将单词级边界框转换为标记级边界框。
    """

    # 词汇文件的名称
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇文件的映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练位置嵌入的最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 预训练初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION

    # 初始化方法
    def __init__(
        # 词汇文件
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
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_label=-100,
        only_label_first_subword=True,
        tokenize_chinese_chars=True,
        strip_accents=None,
        model_max_length: int = 512,
        additional_special_tokens: Optional[List[str]] = None,
        **kwargs,
        # 如果 sep_token 是字符串，则将其转换为特殊标记
        sep_token = AddedToken(sep_token, special=True) if isinstance(sep_token, str) else sep_token
        # 如果 unk_token 是字符串，则将其转换为特殊标记
        unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
        # 如果 pad_token 是字符串，则将其转换为特殊标记
        pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token
        # 如果 cls_token 是字符串，则将其转换为特殊标记
        cls_token = AddedToken(cls_token, special=True) if isinstance(cls_token, str) else cls_token
        # 如果 mask_token 是字符串，则将其转换为特殊标记
        mask_token = AddedToken(mask_token, special=True) if isinstance(mask_token, str) else mask_token

        # 如果词汇表文件不存在，则引发值错误异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 从词汇表文件加载词汇
        self.vocab = load_vocab(vocab_file)
        # 将词汇表转换为从标识符到词汇的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 设置是否进行基本分词的标志
        self.do_basic_tokenize = do_basic_tokenize
        # 如果设置了基本分词，则初始化基本分词器
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 初始化 WordPiece 分词器
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 设置额外的属性
        self.cls_token_box = cls_token_box
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword
        # 调用父类初始化方法，传递参数
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            cls_token_box=cls_token_box,
            sep_token_box=sep_token_box,
            pad_token_box=pad_token_box,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            model_max_length=model_max_length,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    def do_lower_case(self):
        # 返回基本分词器的小写标志
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        # 返回词汇表大小
        return len(self.vocab)

    def get_vocab(self):
        # 返回词汇表和已添加标记的编码器的组合
        return dict(self.vocab, **self.added_tokens_encoder)
    def _tokenize(self, text):
        split_tokens = []
        # 如果需要进行基本的分词处理
        if self.do_basic_tokenize:
            # 使用基本分词器对文本进行分词，不分割特殊标记
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # 如果token在不分割的特殊标记中
                if token in self.basic_tokenizer.never_split:
                    # 将token添加到分词结果中
                    split_tokens.append(token)
                else:
                    # 使用WordPiece分词器对token进行分词，将结果添加到分词结果中
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 使用WordPiece分词器对文本进行分词，将结果赋给split_tokens
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        # 返回分词结果
        return split_tokens

    def _convert_token_to_id(self, token):
        """将token转换为其对应的id"""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """将id转换为其对应的token"""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """将一系列token转换为单个字符串"""
        # 将tokens连接成字符串，并去除" ##"，然后去除两侧的空格
        out_string = " ".join(tokens).replace(" ##", "").strip()
        # 返回结果字符串
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        根据序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记。
        BERT序列的格式如下：

        - 单序列：`[CLS] X [SEP]`
        - 一对序列：`[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的ID列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个序列的可选ID列表。

        Returns:
            `List[int]`: 具有适当特殊标记的输入ID列表。
        """
        # 如果没有第二个序列，则返回添加了CLS和SEP的token_ids_0
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 添加CLS和SEP到token_ids_0和token_ids_1之间，并返回结果
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        返回一个mask，标记哪些token是特殊标记。

        Args:
            token_ids_0 (`List[int]`):
                第一个序列的token ID列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个序列的可选token ID列表。
            already_has_special_tokens (bool):
                如果token已经包含特殊标记，则为True。

        Returns:
            `List[int]`: 特殊标记的mask列表。
        """
```  
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列 ID。在使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                序列对的第二个 ID 列表，可选。
            already_has_special_tokens (`bool`, *可选*, 默认为 `False`):
                标记列表是否已经使用特殊标记格式化。

        Returns:
            `List[int]`: 一个整数列表，范围为 [0, 1]：1 表示特殊标记，0 表示序列标记。
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
    ) -> List[int]:
        """
        从传递的两个序列创建一个掩码，用于在序列对分类任务中使用。BERT 序列对掩码的格式如下: :: 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 | 第一个序列 | 第二个序列 | 如果 `token_ids_1` 为 `None`，则此方法仅返回掩码的第一部分（0s）。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                序列对的第二个 ID 列表，可选。

        Returns:
            `List[int]`: 根据给定序列返回的 [标记类型 ID](../glossary#token-type-ids) 列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
```py  
    # 保存词汇表到指定目录下，并返回词汇表文件名
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引值
        index = 0
        # 如果保存目录存在
        if os.path.isdir(save_directory):
            # 合并保存目录和文件名前缀（如果存在）与词汇表文件名，得到完整的词汇表文件路径
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 否则，直接将文件名前缀与保存目录（或文件名）拼接得到完整的词汇表文件路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开词汇表文件，准备写入
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表，按索引排序，依次写入词汇
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 如果索引值不连续，警告提示
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 写入词汇
                writer.write(token + "\n")
                index += 1
        # 返回保存的词汇表文件名
        return (vocab_file,)

    # 对象调用方法，对输入文本或文本对进行编码
    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义一个方法，对批量文本或文本对进行编码处理，并返回结果
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,  # 指示是否文本对
        boxes: Optional[List[List[List[int]]]] = None,  # 文字框的位置信息
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,  # 单词标签
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略
        truncation: Union[bool, str, TruncationStrategy] = None,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度
        stride: int = 0,  # 步长
        pad_to_multiple_of: Optional[int] = None,  # 对齐到的长度
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型ID
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的标记
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回长度
        verbose: bool = True,  # 是否详细显示信息
        **kwargs,  # 可选的关键字参数
    ) -> BatchEncoding:  # 返回BatchEncoding对象
        # 为了向后兼容 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略，最大长度和关键字参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用_batch_encode_plus方法进行编码处理
        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            is_pair=is_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )
    # 定义一个私有方法，用于批量编码文本或者文本对
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],                     # 批量文本输入
            List[TextInputPair],                 # 批量文本对输入
            List[PreTokenizedInput],            # 预先标记输入
        ],
        is_pair: bool = None,                    # 是否是文本对
        boxes: Optional[List[List[List[int]]]] = None,       # 文字框坐标
        word_labels: Optional[List[List[int]]] = None,        # 单词标签
        add_special_tokens: bool = True,         # 是否添加特殊标记
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,    # 填充策略
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,    # 截断策略
        max_length: Optional[int] = None,       # 最大长度
        stride: int = 0,                        # 步幅
        pad_to_multiple_of: Optional[int] = None,     # 填充到的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,    # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,        # 是否返回标记类型ID
        return_attention_mask: Optional[bool] = None,        # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,     # 是否返回溢出的标记
        return_special_tokens_mask: bool = False,     # 是否返回特殊标记掩码
        return_offsets_mapping: bool = False,         # 是否返回偏移映射
        return_length: bool = False,                  # 是否返回长度
        verbose: bool = True,                         # 是否详细输出
        **kwargs,                             # 其他参数
    ) -> BatchEncoding:                          # 返回值为批处理编码
        # 如果要返回偏移映射，则抛出未实现错误
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        # 对模型进行准备以进行批处理
        batch_outputs = self._batch_prepare_for_model(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            is_pair=is_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        # 返回批处理编码结果
        return BatchEncoding(batch_outputs)

    # 添加关于LAYOUTLMV2编码参数和额外参数的文档字符串
    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 对输入文本或文本对进行批量准备，为模型生成输入数据
    def _batch_prepare_for_model(
        self,
        batch_text_or_text_pairs,
        is_pair: bool = None,  # 是否是文本对
        boxes: Optional[List[List[int]]] = None,  # 文本框的坐标
        word_labels: Optional[List[List[int]]] = None,  # 单词标签
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度
        stride: int = 0,  # 步幅
        pad_to_multiple_of: Optional[int] = None,  # 填充到的倍数
        return_tensors: Optional[str] = None,  # 返回张量类型
        return_token_type_ids: Optional[bool] = None,  # 返回标记类型 ID
        return_attention_mask: Optional[bool] = None,  # 返回注意力掩码
        return_overflowing_tokens: bool = False,  # 返回溢出的标记
        return_special_tokens_mask: bool = False,  # 返回特殊标记掩码
        return_length: bool = False,  # 返回长度
        verbose: bool = True,  # 详细输出
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens.

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        batch_outputs = {}
        # 遍历批量文本或文本对及它们的框，准备输入给模型
        for idx, example in enumerate(zip(batch_text_or_text_pairs, boxes)):
            batch_text_or_text_pair, boxes_example = example
            # 调用prepare_for_model方法预处理文本或文本对，生成模型输入
            outputs = self.prepare_for_model(
                batch_text_or_text_pair[0] if is_pair else batch_text_or_text_pair,
                batch_text_or_text_pair[1] if is_pair else None,
                boxes_example,
                word_labels=word_labels[idx] if word_labels is not None else None,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            # 将输出存入batch_outputs字典
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # 对输出进行填充
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # 返回包装好的BatchEncoding对象
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING)
    # encode 方法用于将文本编码成模型输入的整数列表
    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 输入文本，可以是字符串或预分词输入
        text_pair: Optional[PreTokenizedInput] = None,  # 可选的第二个文本输入，用于生成文本对编码
        boxes: Optional[List[List[int]]] = None,  # 文本框的坐标，用于布局感知模型的输入
        word_labels: Optional[List[int]] = None,  # 单词标签，用于识别文本中的标签
        add_special_tokens: bool = True,  # 是否添加特殊令牌（如CLS、SEP等）
        padding: Union[bool, str, PaddingStrategy] = False,  # 是否进行填充，如果是字符串或填充策略则使用该策略
        truncation: Union[bool, str, TruncationStrategy] = None,  # 是否截断输入文本，如果是字符串或截断策略则使用该策略
        max_length: Optional[int] = None,  # 输入序列的最大长度
        stride: int = 0,  # 截断策略中的步幅
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的长度倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回令牌类型的ID
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的令牌
        return_special_tokens_mask: bool = False,  # 是否返回特殊令牌的掩码
        return_offsets_mapping: bool = False,  # 是否返回字符偏移映射
        return_length: bool = False,  # 是否返回编码后的长度
        verbose: bool = True,  # 是否显示详细信息
        **kwargs,
    ) -> List[int]:  # 返回值是编码后的整数列表
        # 调用 encode_plus 方法进行编码，获取编码后的结果字典
        encoded_inputs = self.encode_plus(
            text=text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )
        # 返回编码后的输入令牌ID列表
        return encoded_inputs["input_ids"]

    # encode_plus 方法用于将文本编码成模型输入的字典形式，包括更多的详细信息
    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 输入文本，可以是字符串或预分词输入
        text_pair: Optional[PreTokenizedInput] = None,  # 可选的第二个文本输入，用于生成文本对编码
        boxes: Optional[List[List[int]]] = None,  # 文本框的坐标，用于布局感知模型的输入
        word_labels: Optional[List[int]] = None,  # 单词标签，用于识别文本中的标签
        add_special_tokens: bool = True,  # 是否添加特殊令牌（如CLS、SEP等）
        padding: Union[bool, str, PaddingStrategy] = False,  # 是否进行填充，如果是字符串或填充策略则使用该策略
        truncation: Union[bool, str, TruncationStrategy] = None,  # 是否截断输入文本，如果是字符串或截断策略则使用该策略
        max_length: Optional[int] = None,  # 输入序列的最大长度
        stride: int = 0,  # 截断策略中的步幅
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的长度倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回令牌类型的ID
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的令牌
        return_special_tokens_mask: bool = False,  # 是否返回特殊令牌的掩码
        return_offsets_mapping: bool = False,  # 是否返回字符偏移映射
        return_length: bool = False,  # 是否返回编码后的长度
        verbose: bool = True,  # 是否显示详细信息
        **kwargs,
    ):  
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences. .. warning:: This method is deprecated,
        `__call__` should be used instead.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The first sequence to be encoded. This can be a string, a list of strings, or a list of lists of strings.
            text_pair (`List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a list of strings (words of a single example) or a
                list of lists of strings (words of a batch of examples).
        """

        # 获取填充和截断策略以及其他参数，用于向后兼容性
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用_encode_plus方法，进行编码并返回结果
        return self._encode_plus(
            text=text,  # 第一个序列，可以是字符串、字符串列表或字符串列表的列表
            boxes=boxes,  # 包含文本框位置信息的列表
            text_pair=text_pair,  # 第二个序列（可选），可以是字符串列表或字符串列表的列表
            word_labels=word_labels,  # 单词标签（可选），用于识别每个单词的标签
            add_special_tokens=add_special_tokens,  # 是否添加特殊令牌
            padding_strategy=padding_strategy,  # 填充策略
            truncation_strategy=truncation_strategy,  # 截断策略
            max_length=max_length,  # 最大长度
            stride=stride,  # 滑动窗口步长
            pad_to_multiple_of=pad_to_multiple_of,  # 填充到的长度的倍数
            return_tensors=return_tensors,  # 返回张量类型
            return_token_type_ids=return_token_type_ids,  # 是否返回令牌类型ID
            return_attention_mask=return_attention_mask,  # 是否返回注意力掩码
            return_overflowing_tokens=return_overflowing_tokens,  # 是否返回溢出的令牌
            return_special_tokens_mask=return_special_tokens_mask,  # 是否返回特殊令牌掩码
            return_offsets_mapping=return_offsets_mapping,  # 是否返回偏移映射
            return_length=return_length,  # 是否返回编码后序列的长度
            verbose=verbose,  # 是否打印详细信息
            **kwargs,  # 其他参数
        )

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 第一个序列，可以是TextInput或PreTokenizedInput类型
        text_pair: Optional[PreTokenizedInput] = None,  # 第二个序列（可选），可以是PreTokenizedInput类型
        boxes: Optional[List[List[int]]] = None,  # 包含文本框位置信息的列表（可选）
        word_labels: Optional[List[int]] = None,  # 单词标签列表（可选）
        add_special_tokens: bool = True,  # 是否添加特殊令牌
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 最大长度（可选）
        stride: int = 0,  # 滑动窗口步长，默认为0
        pad_to_multiple_of: Optional[int] = None,  # 填充到的长度的倍数（可选）
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量类型（可选）
        return_token_type_ids: Optional[bool] = None,  # 是否返回令牌类型ID（可选）
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码（可选）
        return_overflowing_tokens: bool = False,  # 是否返回溢出的令牌，默认为False
        return_special_tokens_mask: bool = False,  # 是否返回特殊令牌掩码，默认为False
        return_offsets_mapping: bool = False,  # 是否返回偏移映射，默认为False
        return_length: bool = False,  # 是否返回编码后序列的长度，默认为False
        verbose: bool = True,  # 是否打印详细信息，默认为True
        **kwargs,  # 其他参数
    # 函数签名，指定了函数返回类型为 BatchEncoding 类型
    ) -> BatchEncoding:
        # 如果需要返回偏移映射，则抛出 NotImplementedError
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # 调用 prepare_for_model 方法准备输入数据并编码
        return self.prepare_for_model(
            text=text,  # 输入文本
            text_pair=text_pair,  # 第二段输入文本（可选）
            boxes=boxes,  # 文本框边界框（可选）
            word_labels=word_labels,  # 单词标签（可选）
            add_special_tokens=add_special_tokens,  # 是否添加特殊标记
            padding=padding_strategy.value,  # 填充策略
            truncation=truncation_strategy.value,  # 截断策略
            max_length=max_length,  # 最大长度
            stride=stride,  # 步幅
            pad_to_multiple_of=pad_to_multiple_of,  # 填充到指定长度的倍数（可选）
            return_tensors=return_tensors,  # 返回的张量类型（可选）
            prepend_batch_axis=True,  # 是否在返回的张量上增加批次维度
            return_attention_mask=return_attention_mask,  # 是否返回注意力掩码（可选）
            return_token_type_ids=return_token_type_ids,  # 是否返回标记类型 ID（可选）
            return_overflowing_tokens=return_overflowing_tokens,  # 是否返回溢出的标记（可选）
            return_special_tokens_mask=return_special_tokens_mask,  # 是否返回特殊标记掩码（可选）
            return_length=return_length,  # 是否返回编码长度（可选）
            verbose=verbose,  # 是否打印详细信息（可选）
        )

    # 函数装饰器，添加文档字符串
    @add_end_docstrings(LAYOUTLMV2_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV2_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 函数定义，准备输入数据并编码
    def prepare_for_model(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 输入文本或预分词输入
        text_pair: Optional[PreTokenizedInput] = None,  # 第二段输入文本（可选）
        boxes: Optional[List[List[int]]] = None,  # 文本框边界框（可选）
        word_labels: Optional[List[int]] = None,  # 单词标签（可选）
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略
        truncation: Union[bool, str, TruncationStrategy] = None,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度
        stride: int = 0,  # 步幅
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定长度的倍数（可选）
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型（可选）
        return_token_type_ids: Optional[bool] = None,  # 是否返回标记类型 ID（可选）
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码（可选）
        return_overflowing_tokens: bool = False,  # 是否返回溢出的标记（可选）
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记掩码（可选）
        return_offsets_mapping: bool = False,  # 是否返回偏移映射（可选）
        return_length: bool = False,  # 是否返回编码长度（可选）
        verbose: bool = True,  # 是否打印详细信息（可选）
        prepend_batch_axis: bool = False,  # 是否在返回的张量上增加批次维度（可选）
        **kwargs,  # 其他关键字参数
    # 截断序列
    def truncate_sequences(
        self,
        ids: List[int],  # 输入 ID 序列
        token_boxes: List[List[int]],  # 输入标记框边界框
        pair_ids: Optional[List[int]] = None,  # 第二序列输入 ID 序列（可选）
        pair_token_boxes: Optional[List[List[int]]] = None,  # 第二序列输入标记框边界框（可选）
        labels: Optional[List[int]] = None,  # 标签（可选）
        num_tokens_to_remove: int = 0,  # 要移除的标记数量
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",  # 截断策略
        stride: int = 0,  # 步幅
    # 定义内部方法 _pad，用于填充输入序列的长度
    def _pad(
        # 输入参数 encoded_inputs，可以是字典形式的 EncodedInput 或者 BatchEncoding 对象
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        # 最大长度，用于指定填充后序列的最大长度，默认为 None
        max_length: Optional[int] = None,
        # 填充策略，指定如何进行填充，默认为 DO_NOT_PAD，即不填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 将序列填充到指定的倍数，默认为 None
        pad_to_multiple_of: Optional[int] = None,
        # 是否返回注意力掩码，默认为 None
        return_attention_mask: Optional[bool] = None,
# 从 transformers.models.bert.tokenization_bert.BasicTokenizer 复制而来的 BasicTokenizer 类定义
class BasicTokenizer(object):
    """
    构造一个 BasicTokenizer 对象，用于执行基本的分词操作（标点符号拆分、转换为小写等）。

    参数：
        do_lower_case（`bool`，*可选*，默认为 `True`）：
            是否在分词时将输入转换为小写。
        never_split（`Iterable`，*可选*）：
            在分词过程中永远不会被拆分的 token 集合。仅在 `do_basic_tokenize=True` 时生效。
        tokenize_chinese_chars（`bool`，*可选*，默认为 `True`）：
            是否对中文字符进行分词。

            对于日文，这可能需要关闭（参见此 [issue](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents（`bool`，*可选*）：
            是否去除所有重音符号。如果未指定此选项，则会根据 `lowercase` 的值来确定（与原始的 BERT 一致）。
        do_split_on_punc（`bool`，*可选*，默认为 `True`）：
            在某些情况下，我们希望跳过基本的标点符号拆分，以便后续的分词可以捕获单词的完整上下文，例如缩略词。

    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果 never_split 未指定，设置为一个空列表
        if never_split is None:
            never_split = []
        # 初始化对象属性
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 将输入的 never_split 参数转换成集合，并与实例中的 never_split 集合取并集
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清洗文本，例如去除空格等
        text = self._clean_text(text)

        # 以下代码段是用来处理中文字符的，将中文字符加工成标记
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 标准化unicode编码，防止使用不同unicode编码的相同字符被视为不同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 以空白字符为分隔符将文本分割成原始token
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        # 初始化分割后的token列表
        split_tokens = []
        for token in orig_tokens:
            # 如果token不在never_split中，则处理token
            if token not in never_split:
                # 如果需要小写化token，则进行处理
                if self.do_lower_case:
                    token = token.lower()
                    # 如果需要去除重音符号，则进行处理
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果需要去除重音符号，则进行处理
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 将处理后的token添加到split_tokens列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 以空白字符为分隔符将split_tokens列表中的token组合成输出的token列表
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            # 如果字符的分类是Mn，则跳过该字符
            if cat == "Mn":
                continue
            # 否则将字符添加到output列表中
            output.append(char)
        return "".join(output)
    # 在给定文本上根据标点符号进行分隔
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要根据标点符号进行分隔，或者文本在不分隔列表中，则返回原始文本
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    # 在文本中的中文字符周围添加空格
    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    # 检查给定代码点是否为CJK字符的代码点
    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 定义"中文字符"为CJK Unicode块中的任何字符
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

    # 清理文本中的非法字符和空白字符
    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
``` 
# 从transformers.models.bert.tokenization_bert中复制的WordpieceTokenizer类
class WordpieceTokenizer(object):
    """运行WordPiece标记化。"""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化WordpieceTokenizer对象，接收一个词汇表、未知标记和每个单词的最大输入字符数
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        将文本分词为其词块。使用贪心最长匹配算法利用给定的词汇进行标记化。

        例如，`input = "unaffable"` 将返回输出 `["un", "##aff", "##able"]`.

        参数:
            text: 单个标记或以空格分隔的标记。这应该已经被传递给*BasicTokenizer*。

        返回:
            一个词块标记列表。
        """

        output_tokens = []
        # 对输入进行空格分割后的每个标记处理
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                # 如果标记长度大于最大输入字符数，则将未知标记添加到输出列表
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
                # 如果存在无法匹配的子字符串，则将未知标记添加到输出列表
                output_tokens.append(self.unk_token)
            else:
                # 将匹配的子字符串添加到输出列表
                output_tokens.extend(sub_tokens)
        return output_tokens
```