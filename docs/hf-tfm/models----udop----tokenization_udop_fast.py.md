# `.\models\udop\tokenization_udop_fast.py`

```py
"""
定义一个 UdopTokenizerFast 类，继承自 PreTrainedTokenizerFast 类，用于实现快速的 UDOP 分词器，基于 HuggingFace 的 tokenizers 库。

该类提供了从 LayoutXLMTokenizer 和 T5Tokenizer 中适配的功能，并基于 BPE 模型实现。

继承自 PreTrainedTokenizerFast 类，包含了大部分主要方法，用户可以参考其超类以获取更多关于这些方法的信息。
"""
class UdopTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" UDOP tokenizer (backed by HuggingFace's *tokenizers* library). Adapted from
    [`LayoutXLMTokenizer`] and [`T5Tokenizer`]. Based on
    [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """
    # 定义一个类，用于处理特定任务的标记器
    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file. 词汇表文件的路径。

        tokenizer_file (`str`, *optional*):
            Path to the tokenizer file. 标记器文件的路径。

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token. 序列结束标记，默认为 `"</s>"`。

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
            分隔符标记，在构建多个序列时使用，例如用于序列分类或问题回答中的文本和问题。还用作使用特殊标记构建的序列的最后一个标记。

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
            未知标记，词汇表中不存在的标记会被设置为此标记。

        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
            用于填充的标记，例如在批处理具有不同长度序列时使用。

        sep_token_box (`List[int]`, *optional*, defaults to `[1000, 1000, 1000, 1000]`):
            The bounding box to use for the special [SEP] token.
            用于特殊 [SEP] 标记的边界框。

        pad_token_box (`List[int]`, *optional*, defaults to `[0, 0, 0, 0]`):
            The bounding box to use for the special [PAD] token.
            用于特殊 [PAD] 标记的边界框。

        pad_token_label (`int`, *optional*, defaults to -100):
            The label to use for padding tokens. Defaults to -100, which is the `ignore_index` of PyTorch's
            CrossEntropyLoss.
            用于填充标记的标签。默认为 -100，这是 PyTorch CrossEntropyLoss 的 `ignore_index`。

        only_label_first_subword (`bool`, *optional*, defaults to `True`):
            Whether or not to only label the first subword, in case word labels are provided.
            是否仅标记第一个子词，如果提供了单词标签。

        additional_special_tokens (`List[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`):
            Additional special tokens used by the tokenizer.
            标记器使用的额外特殊标记。
    """

    # 定义用于加载预训练模型的相关常量和类
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = UdopTokenizer

    # 初始化方法，用于设置类的属性
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        eos_token="</s>",
        sep_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_label=-100,
        only_label_first_subword=True,
        additional_special_tokens=None,
        **kwargs,
    ):
    ):
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token_box=sep_token_box,
            pad_token_box=pad_token_box,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )


        # 调用父类的初始化方法，传递必要的参数和关键字参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token_box=sep_token_box,
            pad_token_box=pad_token_box,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )


        self.vocab_file = vocab_file

        # 添加额外的属性
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword


    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查是否可以保存慢速的分词器，需要检查词汇文件是否存在
        return os.path.isfile(self.vocab_file) if self.vocab_file else False


    @add_end_docstrings(UDOP_ENCODE_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        **kwargs,
    ) -> BatchEncoding:
        # 检查输入参数，确保至少有 `text` 或 `text_target` 被指定
        if text is None and text_target is None:
            raise ValueError("You need to specify either `text` or `text_target`.")
        if text is not None:
            # 如果没有处于目标文本模式，则切换到输入文本模式
            if not self._in_target_context_manager:
                self._switch_to_input_mode()
            # 调用 `call_boxes` 方法处理文本、文本对、框和词标签等参数
            encodings = self.call_boxes(text=text, text_pair=text_pair, boxes=boxes, word_labels=word_labels, **kwargs)
        if text_target is not None:
            # 切换到目标文本模式
            self._switch_to_target_mode()
            # 调用 `_call_one` 方法处理目标文本、目标文本对等参数
            target_encodings = self._call_one(text=text_target, text_pair=text_pair_target, **kwargs)
        # 回到输入文本模式
        self._switch_to_input_mode()

        # 根据是否有目标文本，返回相应的编码结果
        if text_target is None:
            return encodings
        elif text is None:
            return target_encodings
        else:
            # 将目标文本的 `input_ids` 放入编码结果的 `labels` 键中
            encodings["labels"] = target_encodings["input_ids"]
            return encodings


    @add_end_docstrings(UDOP_ENCODE_KWARGS_DOCSTRING)
    # 定义一个方法用于处理文本、文本对、文本列表或预分词输入，同时接收盒子坐标和词标签等参数
    def call_boxes(
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
    ):
        # 从文本和文本对（如果存在）创建批处理输入
        batched_input = [(text, text_pair)] if text_pair else [text]
        # 使用预定义的 tokenizer 对批处理输入进行编码
        encodings = self._tokenizer.encode_batch(
            batched_input, add_special_tokens=add_special_tokens, is_pretokenized=False, **kwargs
        )
        # 返回编码结果的第一个样本的 token 列表
        return encodings[0].tokens

    # 定义一个方法用于将文本或文本对列表批量编码并处理盒子坐标和词标签等参数
    def batch_encode_plus_boxes(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,
        boxes: Optional[List[List[List[int]]]] = None,
        word_labels: Optional[List[List[int]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
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
    ):
        # 创建一个包含文本或文本对的批处理输入
        batched_input = batch_text_or_text_pairs
        # 使用预定义的 tokenizer 对批处理输入进行编码
        encodings = self._tokenizer.encode_batch(
            batched_input, add_special_tokens=add_special_tokens, is_pretokenized=False, **kwargs
        )
        # 返回编码结果
        return encodings
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.

        <Tip warning={true}>

        This method is deprecated, `__call__` should be used instead.

        </Tip>

        Args:
            batch_text_or_text_pairs (`List[str]`, `List[Tuple[str, str]]`, `List[List[str]]`, `List[Tuple[List[str], List[str]]]`, and for not-fast tokenizers, also `List[List[int]]`, `List[Tuple[List[int], List[int]]]`):
                Batch of sequences or pair of sequences to be encoded. This can be a list of
                string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see
                details in `encode_plus`).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略以及其他相关参数，以确保向后兼容性
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用底层方法 `_batch_encode_plus_boxes` 进行批量编码
        return self._batch_encode_plus_boxes(
            batch_text_or_text_pairs=batch_text_or_text_pairs,  # 待编码的文本或文本对
            is_pair=is_pair,  # 是否是文本对
            boxes=boxes,  # 区域框
            word_labels=word_labels,  # 单词标签
            add_special_tokens=add_special_tokens,  # 是否添加特殊标记
            padding_strategy=padding_strategy,  # 填充策略
            truncation_strategy=truncation_strategy,  # 截断策略
            max_length=max_length,  # 最大长度
            stride=stride,  # 步长
            is_split_into_words=is_split_into_words,  # 是否已拆分为单词
            pad_to_multiple_of=pad_to_multiple_of,  # 填充至倍数长度
            return_tensors=return_tensors,  # 是否返回张量
            return_token_type_ids=return_token_type_ids,  # 是否返回 token 类型 id
            return_attention_mask=return_attention_mask,  # 是否返回注意力掩码
            return_overflowing_tokens=return_overflowing_tokens,  # 是否返回溢出的 token
            return_special_tokens_mask=return_special_tokens_mask,  # 是否返回特殊 token 掩码
            return_offsets_mapping=return_offsets_mapping,  # 是否返回偏移映射
            return_length=return_length,  # 是否返回长度
            verbose=verbose,  # 是否详细输出
            **kwargs,  # 其他参数
        )
    # 定义一个方法用于批量编码文本或文本对，支持多种输入类型
    def _batch_encode_plus_boxes(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],         # 输入为单个文本
            List[TextInputPair],     # 输入为文本对
            List[PreTokenizedInput], # 输入为预分词文本
        ],
        is_pair: bool = None,        # 标志是否为文本对
        boxes: Optional[List[List[List[int]]]] = None,    # 相关文本的边框坐标
        word_labels: Optional[List[List[int]]] = None,    # 文本中单词的标签
        add_special_tokens: bool = True,                   # 是否添加特殊标记
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,   # 填充策略
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
        max_length: Optional[int] = None,                  # 最大长度限制
        stride: int = 0,                                   # 截断和填充时的步长
        pad_to_multiple_of: Optional[int] = None,          # 填充到倍数长度
        return_tensors: Optional[str] = None,              # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,      # 是否返回token类型id
        return_attention_mask: Optional[bool] = None,      # 是否返回attention mask
        return_overflowing_tokens: bool = False,           # 是否返回超出最大长度的token
        return_special_tokens_mask: bool = False,          # 是否返回特殊token的mask
        return_offsets_mapping: bool = False,              # 是否返回偏移映射
        return_length: bool = False,                       # 是否返回编码后的长度
        verbose: bool = True,                              # 是否输出详细信息
        **kwargs,                                           # 其他关键字参数
    ):
        # TODO: 实现批量编码文本及边框的功能
        pass

    # 定义一个方法用于编码单个文本或文本对，支持多种输入类型
    def _encode_plus_boxes(
        self,
        text: Union[TextInput, PreTokenizedInput],          # 输入的文本
        text_pair: Optional[PreTokenizedInput] = None,     # 可选的第二个文本
        boxes: Optional[List[List[int]]] = None,           # 相关文本的边框坐标
        word_labels: Optional[List[int]] = None,           # 文本中单词的标签
        add_special_tokens: bool = True,                   # 是否添加特殊标记
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,   # 填充策略
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
        max_length: Optional[int] = None,                  # 最大长度限制
        stride: int = 0,                                   # 截断和填充时的步长
        pad_to_multiple_of: Optional[int] = None,          # 填充到倍数长度
        return_tensors: Optional[bool] = None,             # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,      # 是否返回token类型id
        return_attention_mask: Optional[bool] = None,      # 是否返回attention mask
        return_overflowing_tokens: bool = False,           # 是否返回超出最大长度的token
        return_special_tokens_mask: bool = False,          # 是否返回特殊token的mask
        return_offsets_mapping: bool = False,              # 是否返回偏移映射
        return_length: bool = False,                       # 是否返回编码后的长度
        verbose: bool = True,                              # 是否输出详细信息
        **kwargs,                                           # 其他关键字参数
    ):
        # TODO: 实现编码单个文本及边框的功能
        pass
    ) -> BatchEncoding:
        # 将输入组成批处理输入
        # 两种选项：
        # 1) 只有文本，如果文本必须是一个字符串列表
        # 2) 文本 + 文本对，此时文本是字符串，text_pair 是字符串列表
        batched_input = [(text, text_pair)] if text_pair else [text]
        batched_boxes = [boxes]  # 将盒子坐标转为批处理列表
        batched_word_labels = [word_labels] if word_labels is not None else None  # 将单词标签转为批处理列表，如果不存在则为 None
        batched_output = self._batch_encode_plus_boxes(
            batched_input,
            is_pair=bool(text_pair is not None),  # 如果存在 text_pair 则设置为 True
            boxes=batched_boxes,
            word_labels=batched_word_labels,
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

        # 如果返回的张量为 None，并且不返回溢出的 tokens，则移除前导的批处理轴
        # 在这种情况下，溢出的 tokens 作为批处理输出返回
        if return_tensors is None and not return_overflowing_tokens:
            batched_output = BatchEncoding(
                {
                    key: value[0] if len(value) > 0 and isinstance(value[0], list) else value
                    for key, value in batched_output.items()
                },
                batched_output.encodings,  # 将编码添加到批处理输出中
            )

        # 检查是否需要提醒序列过长
        self._eventual_warn_about_too_long_sequence(batched_output["input_ids"], max_length, verbose)

        return batched_output
    ) -> List[int]:
        """
        Args:
            Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary. Same as doing
            `self.convert_tokens_to_ids(self.tokenize(text))`.
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """
        # 使用 `encode_plus_boxes` 方法对输入文本及其可选的文本对进行编码，同时处理其他参数
        encoded_inputs = self.encode_plus_boxes(
            text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
            **kwargs,
        )

        # 返回编码后的输入文本的 `input_ids` 列表
        return encoded_inputs["input_ids"]

    def encode_plus_boxes(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[List[int]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
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
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences.

        <Tip warning={true}>

        This method is deprecated, `__call__` should be used instead.

        </Tip>

        Args:
            text (`str`, `List[str]` or `List[int]` (the latter only for not-fast tokenizers)):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略，以及其他参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用内部方法 `_encode_plus_boxes` 进行编码，并返回结果
        return self._encode_plus_boxes(
            text=text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
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

    # Copied from transformers.models.layoutxlm.tokenization_layoutxlm_fast.LayoutXLMTokenizerFast._pad
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ):
        # 方法 `_pad` 负责对编码后的输入进行填充操作，根据传入的参数进行相应的处理
        pass

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLM-RoBERTa sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        # If only one sequence is provided, append the separator token to the end of token_ids_0
        if token_ids_1 is None:
            return token_ids_0 + [self.sep_token_id]
        
        # Define the separator token as a list
        sep = [self.sep_token_id]
        
        # Concatenate token_ids_0, separator, token_ids_1, and another separator
        return token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
        not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.

        """

        # Define the separator token as a list
        sep = [self.sep_token_id]

        # If only one sequence is provided, return a list of zeros of length equal to token_ids_0 + separator
        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0]
        
        # If two sequences are provided, return a list of zeros of length equal to token_ids_0 + separator + token_ids_1 + separator
        return len(token_ids_0 + sep + token_ids_1 + sep) * [0]

    # Copied from transformers.models.layoutxlm.tokenization_layoutxlm_fast.LayoutXLMTokenizerFast.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary to a directory. This method is adapted from the LayoutXLMTokenizerFast class.

        Args:
            save_directory (`str`):
                Directory where the vocabulary will be saved.
            filename_prefix (`str`, *optional*):
                Optional prefix to prepend to the vocabulary filename.

        Returns:
            `Tuple[str]`: Tuple containing the path to the saved vocabulary file.

        Raises:
            ValueError: If the fast tokenizer cannot save the vocabulary.
        """

        # Check if the fast tokenizer has the capability to save the vocabulary
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # Ensure save_directory exists and is a directory; log an error if not
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory.")
            return
        
        # Define the output vocabulary file path
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # If the current vocabulary file path is different from the desired output path, copy the vocabulary file
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # Return the path to the saved vocabulary file
        return (out_vocab_file,)
```