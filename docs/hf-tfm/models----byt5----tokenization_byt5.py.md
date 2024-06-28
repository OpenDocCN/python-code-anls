# `.\models\byt5\tokenization_byt5.py`

```py
# 设置编码声明为 UTF-8，确保源文件以 UTF-8 格式解析
# 版权声明和许可信息
# 引入警告模块，用于处理警告信息
# 引入类型提示模块中的 List、Optional 和 Tuple
# 引入 tokenization_utils 模块中的 AddedToken 和 PreTrainedTokenizer 类
# 引入 logging 模块，获取当前模块的日志记录器对象
# 获取当前模块的日志记录器对象并赋值给 logger 变量

# 定义一个名为 ByT5Tokenizer 的类，继承自 PreTrainedTokenizer 类
class ByT5Tokenizer(PreTrainedTokenizer):

    # 定义模型输入的名称列表，包含 "input_ids" 和 "attention_mask"
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化方法，用于创建一个 ByT5 分词器对象
    def __init__(
        self,
        eos_token="</s>",  # 结束序列的特殊标记，默认为 "</s>"
        unk_token="<unk>",  # 未知标记，表示词汇表中不存在的标记，默认为 "<unk>"
        pad_token="<pad>",  # 用于填充序列的特殊标记，默认为 "<pad>"
        extra_ids=125,  # 额外添加到词汇表末尾的特殊标记数目，默认为 125
        additional_special_tokens=None,  # 额外的特殊标记列表，默认为 None
        **kwargs,  # 接收其他未指定参数的关键字参数
    ) -> None:
        # 将额外的特殊标记添加到特殊标记列表中
        if extra_ids > 0 and additional_special_tokens is None:
            # 如果额外的特殊标记数大于0且未提供额外特殊标记，则创建默认的额外特殊标记列表
            additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None and len(additional_special_tokens) > 0:
            # 如果额外的特殊标记数大于0且提供了额外特殊标记，并且列表中有条目，则验证特殊标记数是否正确
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in str(x)), additional_special_tokens)))
            if extra_tokens != extra_ids:
                # 如果额外的特殊标记数与提供的额外特殊标记列表不匹配，则引发值错误
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are"
                    " provided to ByT5Tokenizer. In this case the additional_special_tokens must include the"
                    " extra_ids tokens"
                )

        pad_token = AddedToken(pad_token, lstrip=True, rstrip=True) if isinstance(pad_token, str) else pad_token
        # 对于向后兼容性，强制左右修剪。byt5tests 依赖于此。
        eos_token = AddedToken(eos_token, lstrip=True, rstrip=True) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=True, rstrip=True) if isinstance(unk_token, str) else unk_token
        # unk 标记需要在词汇表中以正确的索引存在
        self._added_tokens_decoder = {0: pad_token, 1: eos_token, 2: unk_token}
        self.offset = len(self._added_tokens_decoder)
        self._utf_vocab_size = 2**8  # utf 是 8 位
        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=0,
            additional_special_tokens=additional_special_tokens,  # TODO extra ids are not used :sweatywmile:
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回 UTF 编码的词汇表大小
        return self._utf_vocab_size

    def get_vocab(self):
        # 返回词汇表，将词汇 ID 映射到对应的标记
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size + self.offset)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
        # 返回一个布尔掩码，指示哪些标记是特殊标记
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
        # 如果已经包含特殊token，则调用父类的方法获取特殊token的掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 普通情况：一些特殊token
        if token_ids_1 is None:
            # 对于单个序列，返回一个由0组成的列表，并在末尾添加一个1表示特殊token
            return ([0] * len(token_ids_0)) + [1]
        else:
            # 对于序列对，返回一个由0组成的列表，每个序列末尾添加一个1表示特殊token
            return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        # 如果最后一个token已经是eos_token_id，则不再添加eos_token_id
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        else:
            # 否则，在token_ids末尾添加eos_token_id并返回
            return token_ids + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. ByT5 does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # 创建一个用于序列对分类任务的掩码，ByT5模型不使用token type ids，因此返回一个全为0的列表
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            # 对于单个序列，返回一个由0组成的列表，长度为token_ids_0和eos的总和
            return len(token_ids_0 + eos) * [0]
        else:
            # 对于序列对，返回一个由0组成的列表，长度为token_ids_0、eos、token_ids_1和eos的总和
            return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of input IDs with special tokens added in the appropriate positions.
        """
    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens.
        """
        # Ensure token_ids_0 ends with an end-of-sequence token if not already present
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        
        if token_ids_1 is None:
            # If only token_ids_0 is provided, return it as the final input IDs
            return token_ids_0
        else:
            # Ensure token_ids_1 ends with an end-of-sequence token if not already present
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            # Concatenate token_ids_0 and token_ids_1 to form the complete input IDs for pair sequences
            return token_ids_0 + token_ids_1

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        # Convert each character in the input text into tokens
        tokens = [chr(i) for i in text.encode("utf-8")]
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) into an ID using the vocabulary."""
        if len(token) != 1:
            token_id = None  # If the token length is not 1, set token_id to None
        else:
            token_id = ord(token) + self.offset  # Calculate the token ID based on ASCII value and offset
        return token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) into a token (str) using the vocabulary."""
        # Convert index back to token using ASCII value and offset
        token = chr(index - self.offset)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings) into a single string."""
        bstring = b""  # Initialize a byte string
        for token in tokens:
            if token in self.added_tokens_decoder:
                # If token exists in added_tokens_decoder, encode it and append to byte string
                tok_string = self.added_tokens_decoder[token].encode("utf-8")
            elif token in self.added_tokens_encoder:
                # If token exists in added_tokens_encoder, encode it and append to byte string
                tok_string = token.encode("utf-8")
            else:
                # Otherwise, convert token to byte and append to byte string
                tok_string = bytes([ord(token)])
            bstring += tok_string
        # Decode byte string into a string and return
        string = bstring.decode("utf-8", errors="ignore")
        return string

    # ByT5Tokenizer has no vocab file
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # This method is intended to save vocabulary, but currently returns an empty tuple
        return ()
```