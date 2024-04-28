# `.\models\deprecated\retribert\tokenization_retribert_fast.py`

```
# 表示使用 utf-8 编码
# 版权信息
# 导入需要的模块和类
# 导入用于规范化的标准化器
# 导入日志工具
# 定义词汇文件和标记器文件的名称
# 定义预训练模型的词汇文件和标记器文件的位置
# 定义预训练模型的位置编码数量
# 定义预训练模型的初始化配置
# 定义 "RetriBertTokenizerFast" 类，继承自 "PreTrainedTokenizerFast"
#   一个用于构造 "fast" RetriBERT tokenizer 的类
#   表示此类与 BertTokenizerFast 完全相同，可以进行端到端的标记操作：标点符号分割和 WordPiece 序列化
#   继承自 PreTrainedTokenizerFast 类，包含大部分主要方法，用户可以参考超类中的方法来了解更多信息
    # 参数说明:
    # vocab_file (`str`): 词汇表文件路径。
    # do_lower_case (`bool`, *optional*, defaults to `True`): 在进行标记化时是否将输入转换为小写。
    # unk_token (`str`, *optional*, defaults to `"[UNK]"`): 未知标记。词汇表中没有的标记无法转换为 ID，而是设置为此标记。
    # sep_token (`str`, *optional*, defaults to `"[SEP]"`): 分隔符标记，用于从多个序列构建序列时使用，例如用于序列分类或用于文本和问题的问答。还用作使用特殊标记构建的序列的最后一个标记。
    # pad_token (`str`, *optional*, defaults to `"[PAD]"`): 用于填充的标记，例如在对不同长度的序列进行批处理时。
    # cls_token (`str`, *optional*, defaults to `"[CLS]"`): 分类器标记，用于进行序列分类（对整个序列进行分类，而不是对每个标记进行分类）。这是使用特殊标记构建的序列的第一个标记。
    # mask_token (`str`, *optional*, defaults to `"[MASK]"`): 用于屏蔽值的标记。这是在进行屏蔽语言建模训练时使用的标记。这是模型将尝试预测的标记。
    # clean_text (`bool`, *optional*, defaults to `True`): 在标记化之前是否清理文本，通过删除任何控制字符并将所有空格替换为经典空格。
    # tokenize_chinese_chars (`bool`, *optional*, defaults to `True`): 是否标记化中文字符。对于日语，可能应该取消激活此选项（参见此问题）。
    # strip_accents (`bool`, *optional*): 是否剥离所有重音。如果未指定此选项，则将由 `lowercase` 的值确定（与原始 BERT 相同）。
    # wordpieces_prefix (`str`, *optional*, defaults to `"##"`): 子词的前缀。
    """

    # VOCAB_FILES_NAMES、PRETRAINED_VOCAB_FILES_MAP、PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES、PRETRAINED_INIT_CONFIGURATION 是预定义的常量

    # 从 transformers 包中导入的常量，这些常量包含有关预训练模型和词汇表的信息
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION

    # 定义慢速分词器的类，该类用于处理 RetriBertTokenizer 分词器
    slow_tokenizer_class = RetriBertTokenizer

    # 模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]

    # 从 transformers.models.bert.tokenization_bert_fast.BertTokenizerFast.__init__ 中复制的代码段，未提供具体内容
    # 初始化函数，设置各种参数
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        # 调用父类的初始化函数，传入参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        # 获取normalizer_state并根据参数修改normalizer_state
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        # 设置self.do_lower_case 参数
        self.do_lower_case = do_lower_case

    # 创建模型输入时，添加特殊token
    # 来自transformers.models.bert.tokenization_bert_fast.BertTokenizerFast.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        构建用于序列分类任务的模型输入，通过连接和添加特殊token。BERT序列的格式如下：

        - 单个序列: `[CLS] X [SEP]`
        - 两个序列: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊token的ID列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个序列的ID列表，可选。

        Returns:
            `List[int]`: 带有适当特殊token的输入ID列表。
        """
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    # 从序列中创建token type IDs
    # 来自transformers.models.bert.tokenization_bert_fast.BertTokenizerFast.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 定义一个函数，用于创建用于序列对分类任务的掩码。掩码的格式是将两个序列拼接在一起，前半部分为0，后半部分为1。
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # 分隔符
        sep = [self.sep_token_id]
        # 分类符
        cls = [self.cls_token_id]
        # 如果token_ids_1为空，则返回前半部分掩码
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 返回拼接的两个序列掩码
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # 从transformers.models.bert.tokenization_bert_fast.BertTokenizerFast.save_vocabulary中复制而来
    # 保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用tokenizer.model.save保存模型
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回保存的文件名
        return tuple(files)
```