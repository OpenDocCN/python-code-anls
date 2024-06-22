# `.\transformers\models\luke\tokenization_luke.py`

```py
# 使用 lru_cache 装饰器缓存函数的结果，避免重复计算
@lru_cache()
# 从 transformers.models.roberta.tokenization_roberta.bytes_to_unicode 复制的函数
def bytes_to_unicode():
    """
    返回 utf-8 字节的列表和一个到 Unicode 字符串的映射。
    我们特意避免映射到空格/控制字符，因为 BPE 编码会出错。

    可逆的 BPE 编码在 Unicode 字符串上工作。这意味着如果想要避免 UNKs，需要在词汇表中包含大量的 Unicode 字符。
    当你的数据集达到约 100 亿标记时，你最终需要大约 5 千个字符。
    """
    # 返回空白字符和控制字符之外的 UTF-8 字节列表和对应的 Unicode 字符映射
    """
    生成一个字典，用于将 utf-8 字节和 Unicode 字符串互相转换。
    """
    # bs 为包含 ASCII 可见字符、特殊字符和扩展 ASCII 字符的列表
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    # 复制 bs 列表到 cs 列表中
    cs = bs[:]
    # n 用于追踪新字符的索引
    n = 0
    # 循环迭代 0 到 255 的整数
    for b in range(2**8):
        # 如果当前整数不在 bs 中，则将其添加到 bs 和 cs 中
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    # 将 cs 列表中的整数转换为对应的 Unicode 字符
    cs = [chr(n) for n in cs]
    # 返回将 bs 和 cs 中元素一一对应的字典
    return dict(zip(bs, cs))
# 从 transformers.models.roberta.tokenization_roberta.get_pairs 复制而来的函数
# 返回单词中的符号对集合
# 单词表示为符号元组（符号是可变长度的字符串）
def get_pairs(word):
    # 创建一个空的符号对集合
    pairs = set()
    # 保存前一个字符
    prev_char = word[0]
    # 遍历单词中的每个字符
    for char in word[1:]:
        # 将前一个字符和当前字符组成的符号对加入集合
        pairs.add((prev_char, char))
        # 更新前一个字符
        prev_char = char
    # 返回符号对集合
    return pairs


# 定义一个名为 LukeTokenizer 的类，继承自 PreTrainedTokenizer
class LukeTokenizer(PreTrainedTokenizer):
    # 构造函数
    # 构建了一个 LUKE 分词器，派生自 GPT-2 分词器，使用字节级字节对编码
    # 这个分词器已经训练成将空格视为标记的一部分，所以一个单词会根据它是否在句子开头（无空格）而编码不同
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
    @property
    # 从 transformers.models.roberta.tokenization_roberta.RobertaTokenizer.vocab_size 复制而来的属性
    # 返回词汇表大小
    def vocab_size(self):
        return len(self.encoder)

    # 从 transformers.models.roberta.tokenization_roberta.RobertaTokenizer.get_vocab 复制而来的方法
    # 返回词汇表
    # 获取词汇表，包括编码器和添加的 tokens 的编码器
    def get_vocab(self):
        vocab = dict(self.encoder).copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 将 token 进行 BPE (Byte Pair Encoding) 处理
    # 如果 token 在缓存中已经存在，则直接返回缓存中的处理结果
    # 否则，对 token 按照 BPE 算法进行处理
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        # 不断地应用 BPE 算法，直到无法继续合并
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
        word = " ".join(word)
        self.cache[token] = word
        return word

    # 将文本进行 tokenize 处理，返回 BPE 处理后的 token 列表
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # 将 token 转换为对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将 id 转换为对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    # 将 tokens 列表转换为字符串
    # 将一系列的标记（字符串）转换成一个字符串
    def convert_tokens_to_string(self, tokens):
        text = "".join(tokens)  # 将标记连接成一个字符串
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)  # 解码字符串为utf-8格式
        return text  # 返回解码后的字符串

    # 从token_ids_0和optional的token_ids_1构建包含特殊token的模型输入
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
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep  # 返回包含特殊token的模型输入

    # 获取token_ids_0和optional的token_ids_1列表的特殊token掩码
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
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )  # 如果已经包含特殊token，则直接调用父类的方法

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]  # 返回特殊token掩码
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]  # 返回特殊token掩码
    # 从两个序列创建 token type ids，用于序列对分类任务。LUKE 不使用 token type ids，因此返回一个全为零的列表。
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从两个序列创建 token type ids，用于序列对分类任务。LUKE 不使用 token type ids，因此返回一个全为零的列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个序列的可选 ID 列表，用于序列对。

        Returns:
            `List[int]`: 全为零的列表。
        """
        # 分隔符的 ID
        sep = [self.sep_token_id]
        # 分类符的 ID
        cls = [self.cls_token_id]

        # 如果没有第二个序列，则返回长度为 cls + token_ids_0 + sep 的列表，其中每个元素都是 0
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则，返回长度为 cls + token_ids_0 + sep + sep + token_ids_1 + sep 的列表，其中每个元素都是 0
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    # 准备文本进行标记化
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        """
        准备文本进行标记化。如果文本没有以空格开头，并且参数 add_prefix_space 为 True，则在文本前加一个空格。

        Args:
            text:
                文本内容。
            is_split_into_words:
                是否已经分成单词。
            **kwargs:
                其它参数，包括 add_prefix_space。

        Returns:
            Tuple[str, Dict]: 包含处理后的文本和参数的元组。
        """
        # 弹出 add_prefix_space 参数，如果没有提供，默认为 self.add_prefix_space
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # 如果文本没有以空格开头，并且参数 add_prefix_space 为 True，则在文本前加一个空格
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)

    # 调用 Tokenizer 进行编码
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
    # 定义一个方法，用于处理文本和实体标记，并生成模型输入
    def _encode_plus(
        self,
        text: Union[TextInput],  # 主要文本输入，可以是单个文本或文本对
        text_pair: Optional[Union[TextInput]] = None,  # 第二个文本输入，可选，用于处理文本对任务
        entity_spans: Optional[EntitySpanInput] = None,  # 主要文本中的实体标记位置，可选
        entity_spans_pair: Optional[EntitySpanInput] = None,  # 第二个文本中的实体标记位置，可选
        entities: Optional[EntityInput] = None,  # 主要文本的实体标记，可选
        entities_pair: Optional[EntityInput] = None,  # 第二个文本的实体标记，可选
        add_special_tokens: bool = True,  # 是否添加特殊标记，例如 [CLS] 和 [SEP]
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略，默认不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略，默认不截断
        max_length: Optional[int] = None,  # 最大长度限制，超过则截断
        max_entity_length: Optional[int] = None,  # 实体标记的最大长度限制，超过则截断
        stride: int = 0,  # 滑动窗口的步幅
        is_split_into_words: Optional[bool] = False,  # 输入是否已经分词
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定长度的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量类型，如 'pt' 表示 PyTorch 张量
        return_token_type_ids: Optional[bool] = None,  # 是否返回 token 类型 ID
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的 token
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记的掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射，用于提取原始文本和 token 之间的对应关系
        return_length: bool = False,  # 是否返回编码后的长度
        verbose: bool = True,  # 是否输出详细信息
        **kwargs,  # 其它未指定参数，作为关键字参数传递给下游函数
    # 设置函数的输入参数和返回类型为 BatchEncoding
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

        # 如果文本已经被分成单词，则抛出 NotImplementedError
        if is_split_into_words:
            raise NotImplementedError("is_split_into_words is not supported in this tokenizer.")

        # 使用 _create_input_sequence 方法对输入进行处理，返回相应的序列数据
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

        # 使用 prepare_for_model 方法为模型准备输入数据，包括创建注意力掩码和标记类型ID
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
    # 批量编码输入文本或文本对，同时处理实体标记和实体位置
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair],   # 定义批量输入文本或文本对的类型
        batch_entity_spans_or_entity_spans_pairs: Optional[
            Union[List[EntitySpanInput], List[Tuple[EntitySpanInput, EntitySpanInput]]]
        ] = None,   # 定义批量实体位置或实体位置对的类型，可选
        batch_entities_or_entities_pairs: Optional[
            Union[List[EntityInput], List[Tuple[EntityInput, EntityInput]]]
        ] = None,   # 定义批量实体标记或实体标记对的类型，可选
        add_special_tokens: bool = True,   # 是否添加特殊标记，默认为True
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,   # 填充策略，默认不填充
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,   # 截断策略，默认不截断
        max_length: Optional[int] = None,   # 最大长度，可选
        max_entity_length: Optional[int] = None,   # 最大实体长度，可选
        stride: int = 0,   # 步长，默认为0
        is_split_into_words: Optional[bool] = False,   # 是否已分词，默认为False
        pad_to_multiple_of: Optional[int] = None,   # 填充至最近的某个数的倍数，可选
        return_tensors: Optional[Union[str, TensorType]] = None,   # 返回的张量类型，可选
        return_token_type_ids: Optional[bool] = None,   # 是否返回token类型id，可选
        return_attention_mask: Optional[bool] = None,   # 是否返回注意力掩码，可选
        return_overflowing_tokens: bool = False,   # 是否返回溢出的标记，默认为False
        return_special_tokens_mask: bool = False,   # 是否返回特殊标记掩码，默认为False
        return_offsets_mapping: bool = False,   # 是否返回偏移映射，默认为False
        return_length: bool = False,   # 是否返回长度，默认为False
        verbose: bool = True,   # 是否冗长地输出信息，默认为True
        **kwargs,   # 其他关键字参数
    def _check_entity_input_format(self, entities: Optional[EntityInput], entity_spans: Optional[EntitySpanInput]):
        if not isinstance(entity_spans, list):   # 如果实体位置不是列表类型，抛出异常
            raise ValueError("entity_spans should be given as a list")
        elif len(entity_spans) > 0 and not isinstance(entity_spans[0], tuple):   # 如果实体位置不为空且不是元组类型，抛出异常
            raise ValueError(
                "entity_spans should be given as a list of tuples containing the start and end character indices"
            )

        if entities is not None:   # 如果存在实体标记
            if not isinstance(entities, list):   # 如果实体标记不是列表类型，抛出异常
                raise ValueError("If you specify entities, they should be given as a list")

            if len(entities) > 0 and not isinstance(entities[0], str):   # 如果实体标记不为空且不是字符串类型，抛出异常
                raise ValueError("If you specify entities, they should be given as a list of entity names")

            if len(entities) != len(entity_spans):   # 如果实体标记和实体位置长度不相等，抛出异常
                raise ValueError("If you specify entities, entities and entity_spans must be the same length")

    def _create_input_sequence(
        self,
        text: Union[TextInput],   # 定义文本的类型
        text_pair: Optional[Union[TextInput]] = None,   # 定义文本对的类型，可选
        entities: Optional[EntityInput] = None,   # 定义实体标记的类型，可选
        entities_pair: Optional[EntityInput] = None,   # 定义实体标记对的类型，可选
        entity_spans: Optional[EntitySpanInput] = None,   # 定义实体位置的类型，可选
        entity_spans_pair: Optional[EntitySpanInput] = None,   # 定义实体位置对的类型，可选
        **kwargs,   # 其他关键字参数
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)   # 添加文档字符串，包含编码的关键字参数和附加的关键字参数
    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Tuple[List[int], None]],  # 一个列表，包含批处理的输入样本的id对和实体id对
        batch_entity_ids_pairs: List[Tuple[Optional[List[int]], Optional[List[int]]]],  # 一个列表，包含批处理的实体id对和实体类型id对
        batch_entity_token_spans_pairs: List[Tuple[Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]],  # 一个列表，包含批处理的实体的token跨度对
        add_special_tokens: bool = True,  # 是否在序列的开头和结尾添加特殊的标记
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
        max_length: Optional[int] = None,  # 最大序列长度
        max_entity_length: Optional[int] = None,  # 最大实体序列长度
        stride: int = 0,  # 与流式处理相关的跨度
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的多个长度
        return_tensors: Optional[str] = None,  # 返回输出的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回token类型id
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的tokens
        return_special_tokens_mask: bool = False,  # 是否返回特殊tokens的掩码
        return_length: bool = False,  # 是否返回序列长度
        verbose: bool = True,  # 是否输出冗长信息
    # 定义一个方法，用于准备输入 id 序列或者输入 id 序列对，使其可以被模型使用。它会添加特殊标记，根据特殊标记截断序列，并在考虑特殊标记的情况下管理移动窗口（使用用户定义的步幅）以处理溢出的标记
    
    def __call__(self, batch_ids_pairs: List[Union[List[int], Tuple[List[int], List[int]]]],
                     batch_entity_ids_pairs: List[Union[List[int], Tuple[List[int], List[int]]]],
                     batch_entity_token_spans_pairs: List[Union[List[Tuple[int, int]], Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]],
                     max_entity_length: Optional[int] = None
                     ) -> BatchEncoding:
    
            # 创建一个空的 batch_outputs 字典，用于存储输出
            batch_outputs = {}
    
            # 遍历输入 id、实体 id 和实体标记范围的列表，依次处理每个 batch
            for input_ids, entity_ids, entity_token_span_pairs in zip(
                batch_ids_pairs, batch_entity_ids_pairs, batch_entity_token_spans_pairs
            ):
                # 分别获取第一个和第二个输入 id
                first_ids, second_ids = input_ids
                # 分别获取第一个和第二个实体 id
                first_entity_ids, second_entity_ids = entity_ids
                # 分别获取第一个和第二个实体标记范围
                first_entity_token_spans, second_entity_token_spans = entity_token_span_pairs
    
                # 调用 prepare_for_model 方法，处理输入 id 和实体 id，获取输出
                outputs = self.prepare_for_model(
                    first_ids,
                    second_ids,
                    entity_ids=first_entity_ids,
                    pair_entity_ids=second_entity_ids,
                    entity_token_spans=first_entity_token_spans,
                    pair_entity_token_spans=second_entity_token_spans,
                    add_special_tokens=add_special_tokens,
                    padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                    truncation=truncation_strategy.value,
                    max_length=max_length,
                    max_entity_length=max_entity_length,
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
    
                # 将输出存储到 batch_outputs 中
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
    
            # 创建 BatchEncoding 对象并返回
            batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)
    
            return batch_outputs
    
        # 添加文档注释
        @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 为模型准备输入数据，对给定的参数进行处理
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
        
    # 对输入数据进行填充，以满足模型输入要求
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
        
    # 内部方法，对输入数据进行填充，以满足模型输入要求
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        max_entity_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    # 保存词汇表到指定目录下的文件
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
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

        # 将编码器（encoder）以 JSON 格式写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 初始化索引为 0
        index = 0
        # 将 BPE 标记和对应的索引按索引排序并写入合并文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            # 写入合并文件的版本信息
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                # 检查 BPE 合并索引是否连续，若不连续则记录警告
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                # 将 BPE 标记写入合并文件
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 构建实体词汇表文件路径
        entity_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["entity_vocab_file"]
        )

        # 将实体词汇表以 JSON 格式写入文件
        with open(entity_vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.entity_vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 返回词汇表文件路径、合并文件路径和实体词汇表文件路径的元组
        return vocab_file, merge_file, entity_vocab_file
```