# `.\models\led\tokenization_led_fast.py`

```
# 定义 LEDTokenizerFast 类，继承自 PreTrainedTokenizerFast 类
class LEDTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" LED tokenizer (backed by HuggingFace's *tokenizers* library), derived from the GPT-2 tokenizer,
    using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import LEDTokenizerFast

    >>> tokenizer = LEDTokenizerFast.from_pretrained("allenai/led-base-16384")
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    """
    refer to this superclass for more information regarding those methods.

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
            other word. (LED tokenizer detect beginning of words by the preceding space).
        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether the post processing step should trim offsets to avoid including whitespaces.
    """

    # 设置两个常量变量
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 设置最大模型输入尺寸为预训练位置嵌入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 指定慢速分词器的类为 LEDTokenizer
    slow_tokenizer_class = LEDTokenizer
    # 模型输入的名称列表，包括 input_ids 和 attention_mask
    model_input_names = ["input_ids", "attention_mask"]

    # 以下内容是从 transformers.models.bart.tokenization_bart_fast.BartTokenizerFast.__init__ 中复制过来的
    # 初始化方法
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
        # 如果 `mask_token` 是字符串，创建一个带有特殊标志的 AddedToken 对象，用于表示特殊的 MASK 标记
        mask_token = (
            AddedToken(mask_token, lstrip=True, normalized=True, special=True)
            if isinstance(mask_token, str)
            else mask_token
        )
        # 调用父类的初始化方法，初始化 LEDTokenizerFast 对象
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

        # 获取当前前置处理器的状态，并检查是否需要更新 `add_prefix_space` 属性
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            # 如果前置处理器的 `add_prefix_space` 属性不匹配当前设定，更新前置处理器的状态
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        self.add_prefix_space = add_prefix_space

        # 检查后处理器的状态，并更新 `sep` 和 `cls` 标记为元组，以便与 LED 的 `post_processor` 兼容
        tokenizer_component = "post_processor"
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)
        if tokenizer_component_instance:
            state = json.loads(tokenizer_component_instance.__getstate__())

            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            if "cls" in state:
                state["cls"] = tuple(state["cls"])

            changes_to_apply = False

            # 检查后处理器的状态是否需要更新 `add_prefix_space` 和 `trim_offsets` 属性
            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                state["add_prefix_space"] = add_prefix_space
                changes_to_apply = True

            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                state["trim_offsets"] = trim_offsets
                changes_to_apply = True

            # 如果有更改需要应用，则创建新的后处理器实例并更新到 LEDTokenizerFast 对象中
            if changes_to_apply:
                component_class = getattr(processors, state.pop("type"))
                new_value = component_class(**state)
                setattr(self.backend_tokenizer, tokenizer_component, new_value)
    def mask_token(self) -> str:
        """
        `str`: 获取掩码标记，用于训练掩码语言建模的模型。如果尚未设置，则记录错误信息。
        
        LED 分词器具有特殊的掩码标记，用于填充掩码管道中的空白。掩码标记将贪婪地包括在 *<mask>* 前的空格。
        """
        # 如果掩码标记未设置，则记录错误信息并返回 None
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        # 返回掩码标记的字符串表示
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        """
        设置掩码标记的默认行为，使其在之前包含空格。

        这是为了与所有先前使用的基于 LED 的模型保持向后兼容所必需的。
        """
        # 如果值是字符串类型，则创建 AddedToken 对象，并设置 lstrip=True，rstrip=False，使掩码标记行为类似普通词
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    # 从 transformers.models.bart.tokenization_bart_fast.BartTokenizerFast._batch_encode_plus 复制而来
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)

        # 如果输入被预分词且没有添加前缀空格，则抛出 ValueError
        if is_split_into_words and not self.add_prefix_space:
            raise ValueError(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
                "to use it with pretokenized inputs."
            )

        # 调用父类的 _batch_encode_plus 方法进行批处理编码
        return super()._batch_encode_plus(*args, **kwargs)

    # 从 transformers.models.bart.tokenization_bart_fast.BartTokenizerFast._encode_plus 复制而来
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)

        # 如果输入被预分词且没有添加前缀空格，则抛出 ValueError
        if is_split_into_words and not self.add_prefix_space:
            raise ValueError(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
                "to use it with pretokenized inputs."
            )

        # 调用父类的 _encode_plus 方法进行编码
        return super()._encode_plus(*args, **kwargs)

    # 从 transformers.models.bart.tokenization_bart_fast.BartTokenizerFast.save_vocabulary 复制而来
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用内部的 tokenizer.model.save 方法保存词汇表到指定目录下
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    # 从 transformers.models.bart.tokenization_bart_fast.BartTokenizerFast.build_inputs_with_special_tokens 复制而来
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 构建带有特殊标记的输入，包括起始标记、终止标记
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        # 如果存在第二个输入序列，添加终止标记，并连接第二个输入序列及其终止标记
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
    # 从 BART -> LED 的转换中复制的方法，用于根据输入序列创建token类型ID
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        创建用于序列对分类任务的掩码。LED 不使用token类型ID，因此返回一个由零组成的列表。
    
        Args:
            token_ids_0 (`List[int]`):
                第一个序列的ID列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个序列的ID列表，用于序列对。
    
        Returns:
            `List[int]`: 全零列表。
        """
        sep = [self.sep_token_id]  # 分隔符的token ID列表
        cls = [self.cls_token_id]  # 类别标记的token ID列表
    
        if token_ids_1 is None:
            # 如果只有一个输入序列，则返回一个由零填充的列表
            return len(cls + token_ids_0 + sep) * [0]
        # 如果有两个输入序列，则返回一个由零填充的列表，包括两个分隔符
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
    
    # 从 transformers.models.led.tokenization_led.LEDTokenizer._pad 复制的方法
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        encoded_inputs = super()._pad(
            encoded_inputs=encoded_inputs,
            max_length=max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

使用 `super()._pad` 方法对输入进行填充操作，返回填充后的编码输入字典。


        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

如果 `return_attention_mask` 为 `None`，则检查模型输入名称中是否包含 `"attention_mask"`，将其赋值给 `return_attention_mask`。


        if return_attention_mask and "global_attention_mask" in encoded_inputs:
            required_input = encoded_inputs[self.model_input_names[0]]
            # `global_attention_mask` need to have the same length as other (sequential) inputs.
            needs_to_be_padded = len(encoded_inputs["global_attention_mask"]) != len(required_input)

如果 `return_attention_mask` 为真且 `encoded_inputs` 中包含 `"global_attention_mask"`：
- 获取第一个模型输入的名称，并检查 `"global_attention_mask"` 的长度是否与该输入的长度相同。


            if needs_to_be_padded:
                difference = len(required_input) - len(encoded_inputs["global_attention_mask"])

                if self.padding_side == "right":
                    # Use `-1` since `0` in `global_attention_mask` means `local attention` instead of `not to attend`
                    encoded_inputs["global_attention_mask"] = (
                        encoded_inputs["global_attention_mask"] + [-1] * difference
                    )
                elif self.padding_side == "left":
                    encoded_inputs["global_attention_mask"] = [-1] * difference + encoded_inputs[
                        "global_attention_mask"
                    ]
                else:
                    raise ValueError("Invalid padding strategy:" + str(self.padding_side))

如果需要进行填充：
- 计算差异，确定填充方向（右侧或左侧），将 `-1` 添加到 `global_attention_mask` 以保持与其他输入相同的长度。


        return encoded_inputs

返回填充后的编码输入字典 `encoded_inputs`。
```