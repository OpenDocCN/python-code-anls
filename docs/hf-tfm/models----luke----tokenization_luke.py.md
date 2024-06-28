# `.\models\luke\tokenization_luke.py`

```py
@lru_cache()
# 使用 functools 模块的 lru_cache 装饰器，用于缓存函数的返回值，提高函数的执行效率
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    """
    # 返回 utf-8 字节列表和映射到 Unicode 字符串的映射
    return list(bytes(range(256))) + [chr(i) for i in range(256, 65536)]
    """
    # 创建一个字典，用于将 UTF-8 字节与 Unicode 字符之间建立映射关系
    bs = (
        # bs 列表包含可打印的 ASCII 字符的 Unicode 编码范围
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    # 复制 bs 到 cs 列表
    cs = bs[:]
    n = 0
    # 遍历所有可能的字节值
    for b in range(2**8):
        # 如果字节值不在 bs 中，将其添加到 bs 和 cs 列表，并分配一个新的 Unicode 字符给它
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    # 将 cs 列表中的数字转换为相应的 Unicode 字符
    cs = [chr(n) for n in cs]
    # 返回一个将 bs 中的字节映射到 cs 中 Unicode 字符的字典
    return dict(zip(bs, cs))
    ```
# 从 transformers.models.roberta.tokenization_roberta.get_pairs 复制而来的函数定义，用于获取单词中的符号对集合
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    # 遍历单词中的每个字符（从第二个字符开始），形成前一个字符和当前字符的符号对，加入集合中
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


# LUKE tokenizer 类，继承自 PreTrainedTokenizer
class LukeTokenizer(PreTrainedTokenizer):
    """
    Constructs a LUKE tokenizer, derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```
    >>> from transformers import LukeTokenizer

    >>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods. It also creates entity sequences, namely
    `entity_ids`, `entity_attention_mask`, `entity_token_type_ids`, and `entity_position_ids` to be used by the LUKE
    model.

    """

    # 定义了一些类属性
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化方法，接受多个参数，设置 LUKE tokenizer 的各种配置
    def __init__(
        self,
        vocab_file,
        merges_file,
        entity_vocab_file,
        task=None,
        max_entity_length=32,
        max_mention_length=30,
        entity_token_1="<ent>",
        entity_token_2="<ent2>",
        entity_unk_token="[UNK]",
        entity_pad_token="[PAD]",
        entity_mask_token="[MASK]",
        entity_mask2_token="[MASK2]",
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        **kwargs,
    ):
        # 继承父类的初始化方法
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    @property
    # 从 transformers.models.roberta.tokenization_roberta.RobertaTokenizer.vocab_size 复制而来，用于返回 LUKE tokenizer 的词汇表大小
    def vocab_size(self):
        return len(self.encoder)

    # 从 transformers.models.roberta.tokenization_roberta.RobertaTokenizer.get_vocab 复制而来，用于获取 LUKE tokenizer 的词汇表
    # 获取词汇表，复制编码器中的内容并更新添加的特殊标记编码器的内容，返回合并后的词汇表字典
    def get_vocab(self):
        vocab = dict(self.encoder).copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 从 transformers.models.roberta.tokenization_roberta.RobertaTokenizer.bpe 复制过来，修改为使用 LUKE 和 Luke 替代 RoBERTa 和 Roberta
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            # 选择具有最小 bpe 排名的双字母对作为 bigram
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

                # 如果找到 bigram，则将其替换为一个单一的 token
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
        word = " ".join(word)
        self.cache[token] = word
        return word

    # 从 transformers.models.roberta.tokenization_roberta.RobertaTokenizer._tokenize 复制过来，修改为使用 LUKE 和 Luke 替代 RoBERTa 和 Roberta
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # 将所有字节映射为 Unicode 字符串，避免 BPE 的控制标记（在我们的情况下是空格）
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # 从 transformers.models.roberta.tokenization_roberta.RobertaTokenizer._convert_token_to_id 复制过来，修改为使用 LUKE 和 Luke 替代 RoBERTa 和 Roberta
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 从 transformers.models.roberta.tokenization_roberta.RobertaTokenizer._convert_id_to_token 复制过来，修改为使用 LUKE 和 Luke 替代 RoBERTa 和 Roberta
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    # 从 transformers.models.roberta.tokenization_roberta.RobertaTokenizer.convert_tokens_to_string 复制过来，修改为使用 LUKE 和 Luke 替代 RoBERTa 和 Roberta
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将一系列的 tokens（字符串）连接成一个字符串
        text = "".join(tokens)
        # 使用 byte_decoder 将字符串解码为 UTF-8 编码的文本
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        # 返回解码后的文本
        return text

    # Copied from transformers.models.roberta.tokenization_roberta.RobertaTokenizer.build_inputs_with_special_tokens with Roberta->Luke, RoBERTa->LUKE
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A LUKE sequence has the following format:

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
        if token_ids_1 is None:
            # 返回单个序列的 input IDs，加上特殊 token `<s>` 和 `</s>`
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 返回序列对的 input IDs，加上特殊 token `<s>`, `</s>` 和 `</s>` 以及第二个序列的 tokens
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    # Copied from transformers.models.roberta.tokenization_roberta.RobertaTokenizer.get_special_tokens_mask with Roberta->Luke, RoBERTa->LUKE
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
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            # 如果已经有特殊 token，则调用父类方法获取特殊 token 的 mask
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            # 返回单个序列的特殊 token 的 mask：首位为 1，其余为 0，末尾再加 1
            return [1] + ([0] * len(token_ids_0)) + [1]
        # 返回序列对的特殊 token 的 mask：首位为 1，其余为 0，中间再加两个 1，再加上第二个序列的 mask
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]
    # 从 transformers.models.roberta.tokenization_roberta.RobertaTokenizer.create_token_type_ids_from_sequences 复制并修改，RoBERTa->LUKE, Roberta->Luke
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        创建用于序列对分类任务的掩码。LUKE 不使用 token type ids，因此返回一个全零列表。

        Args:
            token_ids_0 (`List[int]`):
                第一个序列的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个序列的 ID 列表（可选）。

        Returns:
            `List[int]`: 全零列表。
        """
        sep = [self.sep_token_id]  # 分隔符的 token ID
        cls = [self.cls_token_id]  # 类别开始的 token ID

        if token_ids_1 is None:
            # 如果只有一个序列，则返回长度为序列长度加上特殊 token 的全零列表
            return len(cls + token_ids_0 + sep) * [0]
        # 如果有两个序列，则返回长度为两个序列加上多个特殊 token 的全零列表
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    # 从 transformers.models.roberta.tokenization_roberta.RobertaTokenizer.prepare_for_tokenization 复制并修改，RoBERTa->LUKE, Roberta->Luke
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # 如果文本已经分成单词或需要在文本前加空格，并且文本长度大于0且第一个字符不是空白，则在文本前加空格
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)

    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(
        self,
        text: Union[TextInput, List[TextInput]],
        text_pair: Optional[Union[TextInput, List[TextInput]]] = None,
        entity_spans: Optional[Union[EntitySpanInput, List[EntitySpanInput]]] = None,
        entity_spans_pair: Optional[Union[EntitySpanInput, List[EntitySpanInput]]] = None,
        entities: Optional[Union[EntityInput, List[EntityInput]]] = None,
        entities_pair: Optional[Union[EntityInput, List[EntityInput]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: Optional[bool] = False,
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
        # 这是一个装饰器，将 ENCODE_KWARGS_DOCSTRING 和 ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING 作为参数添加到该方法中的文档字符串中
        pass
    # 定义一个方法 `_encode_plus`，用于处理文本编码及其相关特征的生成
    def _encode_plus(
        self,
        text: Union[TextInput],  # 输入参数：文本或文本对，可以是字符串或列表形式的字符串
        text_pair: Optional[Union[TextInput]] = None,  # 可选参数：第二个文本或文本对
        entity_spans: Optional[EntitySpanInput] = None,  # 可选参数：实体跨度信息
        entity_spans_pair: Optional[EntitySpanInput] = None,  # 可选参数：第二个文本的实体跨度信息
        entities: Optional[EntityInput] = None,  # 可选参数：单个文本的实体信息
        entities_pair: Optional[EntityInput] = None,  # 可选参数：第二个文本的实体信息
        add_special_tokens: bool = True,  # 是否添加特殊标记（如CLS和SEP）
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 最大长度限制
        max_entity_length: Optional[int] = None,  # 单个实体的最大长度限制
        stride: int = 0,  # 步长，默认为0
        is_split_into_words: Optional[bool] = False,  # 输入是否已分词
        pad_to_multiple_of: Optional[int] = None,  # 填充到某个整数倍
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型IDs
        return_attention_mask: Optional[bool] = None,  # 是否返回attention mask
        return_overflowing_tokens: bool = False,  # 是否返回溢出的token
        return_special_tokens_mask: bool = False,  # 是否返回特殊token的mask
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回编码后的长度
        verbose: bool = True,  # 是否打印详细信息
        **kwargs,  # 其他未列出的关键字参数
    ):
    ) -> BatchEncoding:
        # 如果 return_offsets_mapping 为真，则抛出未实现的错误
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )

        # 如果 is_split_into_words 为真，则抛出未实现的错误
        if is_split_into_words:
            raise NotImplementedError("is_split_into_words is not supported in this tokenizer.")

        # 调用 _create_input_sequence 方法生成输入序列的各部分
        (
            first_ids,
            second_ids,
            first_entity_ids,
            second_entity_ids,
            first_entity_token_spans,
            second_entity_token_spans,
        ) = self._create_input_sequence(
            text=text,
            text_pair=text_pair,
            entities=entities,
            entities_pair=entities_pair,
            entity_spans=entity_spans,
            entity_spans_pair=entity_spans_pair,
            **kwargs,
        )

        # prepare_for_model 方法将创建 attention_mask 和 token_type_ids
        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            entity_ids=first_entity_ids,
            pair_entity_ids=second_entity_ids,
            entity_token_spans=first_entity_token_spans,
            pair_entity_token_spans=second_entity_token_spans,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            max_entity_length=max_entity_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )
    # 定义一个方法用于批量编码文本和实体信息，并返回编码后的结果
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair]],
        batch_entity_spans_or_entity_spans_pairs: Optional[
            Union[List[EntitySpanInput], List[Tuple[EntitySpanInput, EntitySpanInput]]]
        ] = None,
        batch_entities_or_entities_pairs: Optional[
            Union[List[EntityInput], List[Tuple[EntityInput, EntityInput]]]
        ] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: Optional[bool] = False,
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
        # 检查实体输入的格式是否正确
        def _check_entity_input_format(self, entities: Optional[EntityInput], entity_spans: Optional[EntitySpanInput]):
            # 如果实体 spans 不是 list 类型，抛出数值错误异常
            if not isinstance(entity_spans, list):
                raise ValueError("entity_spans should be given as a list")
            # 如果实体 spans 的长度大于零且第一个元素不是元组，抛出数值错误异常
            elif len(entity_spans) > 0 and not isinstance(entity_spans[0], tuple):
                raise ValueError(
                    "entity_spans should be given as a list of tuples containing the start and end character indices"
                )

            # 如果 entities 不是 None
            if entities is not None:
                # 如果 entities 不是 list 类型，抛出数值错误异常
                if not isinstance(entities, list):
                    raise ValueError("If you specify entities, they should be given as a list")

                # 如果 entities 的长度大于零且第一个元素不是字符串，抛出数值错误异常
                if len(entities) > 0 and not isinstance(entities[0], str):
                    raise ValueError("If you specify entities, they should be given as a list of entity names")

                # 如果 entities 的长度和 entity_spans 的长度不相等，抛出数值错误异常
                if len(entities) != len(entity_spans):
                    raise ValueError("If you specify entities, entities and entity_spans must be the same length")

        # 创建输入序列的方法，接受文本、实体信息以及其他关键字参数
        def _create_input_sequence(
            self,
            text: Union[TextInput],
            text_pair: Optional[Union[TextInput]] = None,
            entities: Optional[EntityInput] = None,
            entities_pair: Optional[EntityInput] = None,
            entity_spans: Optional[EntitySpanInput] = None,
            entity_spans_pair: Optional[EntitySpanInput] = None,
            **kwargs,
        ):
            # 使用 ENCODE_KWARGS_DOCSTRING 和 ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING 的注释添加到方法
            @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义一个方法 `_batch_prepare_for_model`，用于准备模型输入数据的批处理
    def _batch_prepare_for_model(
        # 批次中的每个样本由一个 ID 列表和一个空值组成的元组组成
        self,
        batch_ids_pairs: List[Tuple[List[int], None]],
        # 批次中的每个样本由两个可选的实体 ID 列表组成的元组组成
        batch_entity_ids_pairs: List[Tuple[Optional[List[int]], Optional[List[int]]]],
        # 批次中的每个样本由两个可选的实体标记跨度列表组成的元组组成
        batch_entity_token_spans_pairs: List[Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]],
        # 是否添加特殊标记
        add_special_tokens: bool = True,
        # 填充策略，默认不填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 截断策略，默认不截断
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        # 最大长度限制
        max_length: Optional[int] = None,
        # 最大实体长度限制
        max_entity_length: Optional[int] = None,
        # 步幅大小，默认为0
        stride: int = 0,
        # 填充到某个倍数，默认不填充到倍数
        pad_to_multiple_of: Optional[int] = None,
        # 返回的张量类型，默认不返回张量
        return_tensors: Optional[str] = None,
        # 返回的 token_type_ids 是否可选
        return_token_type_ids: Optional[bool] = None,
        # 返回的 attention_mask 是否可选
        return_attention_mask: Optional[bool] = None,
        # 是否返回溢出的 token
        return_overflowing_tokens: bool = False,
        # 是否返回特殊 token 掩码
        return_special_tokens_mask: bool = False,
        # 是否返回长度信息
        return_length: bool = False,
        # 是否详细输出信息，默认为 True
        verbose: bool = True,
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens


        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
            batch_entity_ids_pairs: list of entity ids or entity ids pairs
            batch_entity_token_spans_pairs: list of entity spans or entity spans pairs
            max_entity_length: The maximum length of the entity sequence.
        """

        # Initialize an empty dictionary to store batch outputs
        batch_outputs = {}

        # Iterate over input sequences and corresponding entity information
        for input_ids, entity_ids, entity_token_span_pairs in zip(
            batch_ids_pairs, batch_entity_ids_pairs, batch_entity_token_spans_pairs
        ):
            # Unpack input sequences into first and second parts
            first_ids, second_ids = input_ids
            # Unpack entity ids into first and second parts
            first_entity_ids, second_entity_ids = entity_ids
            # Unpack entity token spans into first and second parts
            first_entity_token_spans, second_entity_token_spans = entity_token_span_pairs

            # Prepare inputs for the model using specified parameters
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                entity_ids=first_entity_ids,
                pair_entity_ids=second_entity_ids,
                entity_token_spans=first_entity_token_spans,
                pair_entity_token_spans=second_entity_token_spans,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # Specify padding strategy
                truncation=truncation_strategy.value,  # Specify truncation strategy
                max_length=max_length,  # Maximum length of the sequences
                max_entity_length=max_entity_length,  # Maximum length of the entity sequence
                stride=stride,  # Stride for handling overflowing tokens
                pad_to_multiple_of=None,  # We pad in batch afterward
                return_attention_mask=False,  # We pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # Convert batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
            )

            # Aggregate outputs from each batch iteration
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # Perform padding on batch outputs using specified parameters
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,  # Specify padding strategy
            max_length=max_length,  # Maximum length of the sequences
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # Convert batch outputs to BatchEncoding format
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        # Return the processed batch outputs
        return batch_outputs
    # 准备输入数据以供模型使用，根据参数进行处理和转换
    def prepare_for_model(
        self,
        ids: List[int],
        pair_ids: Optional[List[int]] = None,
        entity_ids: Optional[List[int]] = None,
        pair_entity_ids: Optional[List[int]] = None,
        entity_token_spans: Optional[List[Tuple[int, int]]] = None,
        pair_entity_token_spans: Optional[List[Tuple[int, int]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
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
        prepend_batch_axis: bool = False,
        **kwargs,
    ):
        # 对输入数据进行预处理，包括添加特殊标记、填充、截断等操作
        ...

    # 对编码后的输入进行填充处理，确保输入数据的长度一致性
    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ):
        # 对编码后的输入进行填充，使得它们具有相同的长度或满足指定的填充要求
        ...

    # 内部方法：对编码后的输入进行低级别的填充操作
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        ...
    # 保存词汇表到指定目录中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，若不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建词汇表文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        
        # 构建合并文件路径
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 初始化索引
        index = 0
        # 写入合并文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            # 遍历并排序BPE词汇表，按索引写入文件
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                # 检查BPE合并索引是否连续，记录警告信息
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                # 写入BPE token到文件
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 构建实体词汇表文件路径
        entity_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["entity_vocab_file"]
        )

        # 写入实体词汇表文件
        with open(entity_vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.entity_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 返回保存的文件路径元组
        return vocab_file, merge_file, entity_vocab_file
```