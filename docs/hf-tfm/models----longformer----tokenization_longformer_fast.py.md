# `.\transformers\models\longformer\tokenization_longformer_fast.py`

```py
# 定义了字符编码格式
# 版权声明
# 导入依赖
# 导入类型提示
# 导入日志模块
# 定义了词汇文件名称
# 预训练词汇文件对照表
    "tokenizer_file": {   # 创建名为tokenizer_file的字典
        "allenai/longformer-base-4096": (   # 键为"allenai/longformer-base-4096"，值为包含长字符串的元组
            # 标识符"allenai/longformer-base-4096"所对应的值是指向tokenizer.json文件的链接
            "https://huggingface.co/allenai/longformer-base-4096/resolve/main/tokenizer.json"
        ),
        "allenai/longformer-large-4096": (   # 键为"allenai/longformer-large-4096"，值为包含长字符串的元组
            # 标识符"allenai/longformer-large-4096"所对应的值是指向tokenizer.json文件的链接
            "https://huggingface.co/allenai/longformer-large-4096/resolve/main/tokenizer.json"
        ),
        "allenai/longformer-large-4096-finetuned-triviaqa": (   # 键为"allenai/longformer-large-4096-finetuned-triviaqa"，值为包含长字符串的元组
            # 标识符"allenai/longformer-large-4096-finetuned-triviaqa"所对应的值是指向tokenizer.json文件的链接
            "https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa/resolve/main/tokenizer.json"
        ),
        "allenai/longformer-base-4096-extra.pos.embd.only": (   # 键为"allenai/longformer-base-4096-extra.pos.embd.only"，值为包含长字符串的元组
            # 标识符"allenai/longformer-base-4096-extra.pos.embd.only"所对应的值是指向tokenizer.json文件的链接
            "https://huggingface.co/allenai/longformer-base-4096-extra.pos.embd.only/resolve/main/tokenizer.json"
        ),
        "allenai/longformer-large-4096-extra.pos.embd.only": (   # 键为"allenai/longformer-large-4096-extra.pos.embd.only"，值为包含长字符串的元组
            # 标识符"allenai/longformer-large-4096-extra.pos.embd.only"所对应的值是指向tokenizer.json文件的链接
            "https://huggingface.co/allenai/longformer-large-4096-extra.pos.embd.only/resolve/main/tokenizer.json"
        ),
    },
# 结尾的大括号，可能是代码块缺失或多余，需要检查并确认代码块的完整性

# 预训练位置嵌入的尺寸字典，包含不同模型名称和相应的嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "allenai/longformer-base-4096": 4096,
    "allenai/longformer-large-4096": 4096,
    "allenai/longformer-large-4096-finetuned-triviaqa": 4096,
    "allenai/longformer-base-4096-extra.pos.embd.only": 4096,
    "allenai/longformer-large-4096-extra.pos.embd.only": 4096,
}

# 从 transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast 复制得到的类，将 RoBERTa 字段替换为 Longformer 字段
class LongformerTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”Longformer分词器（由HuggingFace的*tokenizers*库支持），源自GPT-2分词器，使用字节级的字节对编码。

    该分词器已经训练过，将空格视为标记的一部分（有点像sentencepiece），因此一个单词将根据它是否在句子开头（没有空格）而被编码成不同的方式：

    ```python
    >>> from transformers import LongformerTokenizerFast

    >>> tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    ```py

    通过在实例化分词器或在对一些文本调用它时传递 `add_prefix_space=True`，可以避免这种行为，但由于模型没有使用这种方式进行预训练，这可能会降低性能。

    <Tip>

    当使用`is_split_into_words=True`时，需要使用`add_prefix_space=True`来实例化该分词器。

    </Tip>

    该分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大部分主要方法。用户应该参考该超类，了解这些方法的更多信息。
    """
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (Longformer tokenizer detect beginning of words by the preceding space).
        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether the post processing step should trim offsets to avoid including whitespaces.
    """

    # Vocabulary files names defined by the model
    vocab_files_names = VOCAB_FILES_NAMES
    # Pretrained vocabulary files map defined by the model
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # Maximum model input sizes defined by the model
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 指定慢速标记器的类为LongformerTokenizer
    
    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        trim_offsets=True,
        **kwargs,
    ):
        # 如果mask_token是字符串，则创建一个AddedToken对象，并指定其属性
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )
        # 调用父类的初始化方法，传入参数
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            **kwargs,
        )
    
        # 将预处理状态转换为JSON格式
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        # 如果add_prefix_space与参数add_prefix_space不一致，则修改预处理器的add_prefix_space属性
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)
    
        # 设置实例的add_prefix_space属性
        self.add_prefix_space = add_prefix_space
    
        # 获取后处理器实例，并对其状态进行修改
        tokenizer_component = "post_processor"
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)
        if tokenizer_component_instance:
            state = json.loads(tokenizer_component_instance.__getstate__())
    
            # 如果状态中存在'sep'和'cls'，则将它们转换为元组
            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            if "cls" in state:
                state["cls"] = tuple(state["cls"])
    
            changes_to_apply = False
    
            # 如果add_prefix_space与参数add_prefix_space不一致，则修改状态中的add_prefix_space属性
            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                state["add_prefix_space"] = add_prefix_space
                changes_to_apply = True
    
            # 如果trim_offsets与参数trim_offsets值不一致，则修改状态中的trim_offsets属性
            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                state["trim_offsets"] = trim_offsets
                changes_to_apply = True
    
            # 如果有需要应用的更改，则创建新的组件实例并应用
            if changes_to_apply:
                component_class = getattr(processors, state.pop("type"))
                new_value = component_class(**state)
                setattr(self.backend_tokenizer, tokenizer_component, new_value)
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.

        Longformer tokenizer has a special mask token to be usable in the fill-mask pipeline. The mask token will greedily
        comprise the space before the *
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. Longformer does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # 分隔符标记
        sep = [self.sep_token_id]
        # 类别标记
        cls = [self.cls_token_id]

        # 如果第二个序列为空，则返回由0组成的列表，其长度等于类别标记、第一个序列、分隔符的总长度
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则，返回由0组成的列表，其长度等于类别标记、第一个序列、两个分隔符、第二个序列、分隔符的总长度
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
```