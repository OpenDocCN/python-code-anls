# `.\models\seamless_m4t\tokenization_seamless_m4t.py`

```
# 导入标准库和第三方库
import os  # 导入操作系统相关的模块
from shutil import copyfile  # 从 shutil 模块导入 copyfile 函数
from typing import Any, Dict, List, Optional, Tuple, Union  # 导入类型提示相关的模块

import sentencepiece as spm  # 导入 sentencepiece 库，用于分词处理

# 导入本地自定义模块
from ...convert_slow_tokenizer import import_protobuf  # 从本地路径导入 convert_slow_tokenizer 模块
from ...tokenization_utils import (  # 导入本地路径下的 tokenization_utils 模块中的相关类和函数
    BatchEncoding,
    PreTokenizedInput,
    PreTrainedTokenizer,
    TextInput,
)
from ...tokenization_utils_base import AddedToken  # 导入 tokenization_utils_base 模块中的 AddedToken 类
from ...utils import PaddingStrategy, logging  # 从本地路径导入 utils 模块中的 PaddingStrategy 和 logging 类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 预训练词汇文件映射，包含了不同预训练模型的词汇文件链接
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/hf-seamless-m4t-medium": (
            "https://huggingface.co/facebook/hf-seamless-m4t-medium/blob/main/sentencepiece.bpe.model"
        ),
    }
}

SPIECE_UNDERLINE = "▁"  # 定义一个特殊的符号，用于表示词间的分隔

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}  # 定义预训练词汇文件的名称

# 预训练位置嵌入大小映射，指定了不同预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/hf-seamless-m4t-medium": 2048,
}


class SeamlessM4TTokenizer(PreTrainedTokenizer):
    """
    Construct a SeamlessM4T tokenizer.

    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    The tokenization method is `<language code> <tokens> <eos>` for source language documents, and `<eos> <language
    code> <tokens> <eos>` for target language documents.

    Examples:

    ```python
    >>> from transformers import SeamlessM4TTokenizer

    >>> tokenizer = SeamlessM4TTokenizer.from_pretrained(
    ...     "facebook/hf-seamless-m4t-medium", src_lang="eng", tgt_lang="fra"
    ... )
    >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
    >>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
    >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
    ```
    """

    # 初始化方法，构造一个 SeamlessM4TTokenizer 对象
    def __init__(
        self,
        vocab_file,
        src_lang=None,
        tgt_lang=None,
        unk_token="<unk>",
        sep_token="<sep>",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        """
        Args:
            vocab_file (str): 文件路径，包含预训练词汇表
            src_lang (Optional[str]): 源语言代码，默认为 None
            tgt_lang (Optional[str]): 目标语言代码，默认为 None
            unk_token (str): 未知标记，默认为 "<unk>"
            sep_token (str): 分隔符标记，默认为 "<sep>"
            cls_token (str): 类别标记，默认为 "<cls>"
            pad_token (str): 填充标记，默认为 "<pad>"
            mask_token (str): 掩码标记，默认为 "<mask>"
            **kwargs: 其他关键字参数
        """
        # 调用父类的初始化方法，初始化预训练分词器
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs
        )
        self.vocab_file = vocab_file  # 保存预训练词汇表文件路径
        self.src_lang = src_lang  # 保存源语言代码
        self.tgt_lang = tgt_lang  # 保存目标语言代码
        self.sp_model = spm.SentencePieceProcessor(model_file=self.vocab_file)  # 使用 sentencepiece 加载词汇表模型文件

    def _tokenize(self, text_input):
        """
        执行实际的分词过程，根据源语言和目标语言代码进行不同的分词处理。

        Args:
            text_input (str): 输入文本字符串

        Returns:
            List[str]: 分词后的字符串列表
        """
        # 如果存在源语言代码
        if self.src_lang:
            # 将源语言代码加入到分词结果中，并在末尾添加特殊符号
            tokenized = [self.src_lang] + self.sp_model.encode(text_input, out_type=str) + [self.eos_token]
        # 如果存在目标语言代码
        elif self.tgt_lang:
            # 将特殊符号加入到分词结果中，并在末尾添加目标语言代码和特殊符号
            tokenized = [self.eos_token] + [self.tgt_lang] + self.sp_model.encode(text_input, out_type=str) + [self.eos_token]
        else:
            raise ValueError("Either src_lang or tgt_lang must be specified.")  # 抛出数值错误异常

        return tokenized  # 返回分词结果列表

    def _convert_token_to_id(self, token):
        """
        将单个标记转换为其对应的 ID。

        Args:
            token (str): 输入的单个标记字符串

        Returns:
            int: 标记对应的 ID
        """
        return self.sp_model.piece_to_id(token)  # 使用 sentencepiece 将标记转换为 ID

    def _convert_id_to_token(self, index):
        """
        将单个 ID 转换为其对应的标记。

        Args:
            index (int): 输入的单个 ID

        Returns:
            str: ID 对应的标记字符串
        """
        return self.sp_model.id_to_piece(index)  # 使用 sentencepiece 将 ID 转换为标记字符串

    def convert_tokens_to_string(self, tokens):
        """
        将分词后的标记列表转换为原始字符串。

        Args:
            tokens (List[str]): 输入的分词后的标记列表

        Returns:
            str: 原始字符串
        """
        # 如果列表中第一个元素是特殊符号，则从第二个元素开始连接为字符串
        if tokens[0] == self.eos_token:
            tokens = tokens[1:]
        return self.sp_model.decode(tokens)  # 使用 sentencepiece 解码为原始字符串

    @property
    def vocab_size(self):
        """
        返回词汇表大小。

        Returns:
            int: 词汇表大小
        """
        return len(self.sp_model)

    @property
    def vocab(self):
        """
        返回词汇表内容。

        Returns:
            List[str]: 词汇表中的所有标记列表
        """
        return [self.sp_model.id_to_piece(i) for i in range(self.vocab_size)]  # 使用 sentencepiece 返回所有标记列表

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        从预训练模型加载分词器。

        Args:
            pretrained_model_name_or_path (str): 预训练模型的名称或路径
            *inputs: 不定长参数列表
            **kwargs: 关键字参数

        Returns:
            PreTrainedTokenizer: 加载后的预训练分词器对象
        """
        # 调用父类的 from_pretrained 方法，加载预训练模型
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        return tokenizer  # 返回加载后的分词器对象
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
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
        tokenizer_file (`str`, *optional*):
            The path to a tokenizer file to use instead of the vocab file.
        src_lang (`str`, *optional*, defaults to `"eng"`):
            The language to use as source language for translation.
        tgt_lang (`str`, *optional*, defaults to `"fra"`):
            The language to use as target language for translation.
        sp_model_kwargs (`Dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the model initialization.
        additional_special_tokens (tuple or list of `str` or `tokenizers.AddedToken`, *optional*):
            A tuple or a list of additional special tokens. Can be used to specify the list of languages that will be
            supported by the tokenizer.
        add_prefix_space (`bool`, *optional*, defaults to `True`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word.
    """

    # 从外部导入的全局变量，包含预训练模型所需的词汇表文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 从外部导入的全局变量，包含预训练模型支持的最大输入序列长度
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 从外部导入的全局变量，包含预训练模型所需的词汇表文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        tokenizer_file=None,
        src_lang="eng",
        tgt_lang="fra",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        additional_special_tokens=None,
        add_prefix_space=True,
        **kwargs,
    ):
        # 初始化函数，用于初始化对象的各种属性和设置
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # 保留了一个未使用的参数以保持某些重要的“Copied from”语句
        self.legacy = False
        self.vocab_file = vocab_file

        # 获取 SentencePiece 处理器对象
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))

        # 下面是为了保持 fairseq 与 SentencePiece 词汇对齐的逻辑说明
        self._added_tokens_decoder = {
            0: AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token,
            1: AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token,
            2: AddedToken(bos_token, special=True) if isinstance(bos_token, str) else bos_token,
            3: AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token,
        }

        # fairseq 词汇与 SentencePiece 词汇的偏移量，用于对齐
        self.fairseq_offset = 1

        # 获取 SentencePiece 词汇表大小
        self.sp_model_size = len(self.sp_model)

        # 设置源语言和目标语言的特殊标记
        self._src_lang = f"__{src_lang}__" if "__" not in src_lang else src_lang
        self._tgt_lang = f"__{tgt_lang}__" if "__" not in tgt_lang else tgt_lang
        self.add_prefix_space = add_prefix_space

        # 调用父类的初始化方法，初始化基本的特殊标记和其他参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            tokenizer_file=tokenizer_file,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        # 设置源语言特殊标记
        self.set_src_lang_special_tokens(self._src_lang)
        # 设置目标语言特殊标记
        self.set_tgt_lang_special_tokens(self._tgt_lang)

    # 从 transformers.models.nllb.tokenization_nllb.NllbTokenizer.__getstate__ 复制而来的方法
    # 返回对象的序列化状态，包括所有实例变量
    def __getstate__(self):
        # 复制对象的字典形式作为状态
        state = self.__dict__.copy()
        # 将 sp_model 设置为 None，不包含在序列化状态中
        state["sp_model"] = None
        # 使用 sp_model 的 serialized_model_proto 方法获取其序列化模型协议
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        # 返回状态字典
        return state

    # 从给定状态恢复对象的状态
    # 从 transformers.models.nllb.tokenization_nllb.NllbTokenizer.__setstate__ 复制而来
    def __setstate__(self, d):
        # 直接使用给定的状态字典来恢复对象的实例变量
        self.__dict__ = d

        # 为了向后兼容
        # 如果对象中没有 sp_model_kwargs 属性，则创建空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用 sp_model_kwargs 中的参数创建 SentencePieceProcessor 对象赋给 sp_model
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 从 sp_model_proto 中加载序列化的模型协议
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    # 返回 sp_model 的词汇表大小
    @property
    def vocab_size(self):
        return len(self.sp_model)

    # 定义对象的调用方法，允许将对象像函数一样调用
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        pad_to_multiple_of: Optional[int] = 2,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        **kwargs,
    ):
        pass

    # 返回 src_lang 属性值，从 transformers.models.nllb.tokenization_nllb.NllbTokenizer.src_lang 复制而来
    @property
    def src_lang(self) -> str:
        return self._src_lang

    # 设置 src_lang 属性，从 transformers.models.nllb.tokenization_nllb.NllbTokenizer.src_lang 复制而来
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        # 如果 new_src_lang 中不包含 "__"，则添加前后缀 "__"
        if "__" not in new_src_lang:
            self._src_lang = f"__{new_src_lang}__"
        else:
            self._src_lang = new_src_lang
        # 调用 set_src_lang_special_tokens 方法，设置特殊标记的 src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    # 返回 tgt_lang 属性值
    @property
    def tgt_lang(self) -> str:
        return self._tgt_lang

    # 设置 tgt_lang 属性
    @tgt_lang.setter
    def tgt_lang(self, new_tgt_lang: str) -> None:
        # 如果 new_tgt_lang 中不包含 "__"，则添加前后缀 "__"
        if "__" not in new_tgt_lang:
            self._tgt_lang = f"__{new_tgt_lang}__"
        else:
            self._tgt_lang = new_tgt_lang
        # 调用 set_tgt_lang_special_tokens 方法，设置特殊标记的 tgt_lang
        self.set_tgt_lang_special_tokens(self._tgt_lang)

    # 返回特殊标记的掩码，从 transformers.models.nllb.tokenization_nllb.NllbTokenizer.get_special_tokens_mask 复制而来
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
            )

        # Create lists filled with 1s for the prefix and suffix tokens
        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)

        # If token_ids_1 is not provided, return the prefix tokens, token_ids_0 with 0s inserted, and suffix tokens
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        
        # If token_ids_1 is provided, concatenate prefix tokens, token_ids_0, token_ids_1, and suffix tokens
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    # Copied from transformers.models.nllb.tokenization_nllb.NllbTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An NLLB sequence has the following format, where `X` represents the sequence:

        - `input_ids` (for encoder) `X [eos, src_lang_code]`
        - `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            # Return prefix tokens, token_ids_0, and suffix tokens if token_ids_1 is not provided
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        
        # Return prefix tokens, token_ids_0, token_ids_1, and suffix tokens if token_ids_1 is provided
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    # Copied from transformers.models.nllb.tokenization_nllb.NllbTokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs tensor from given sequences, indicating which sequence a token belongs to (first or second).
        This function is used in sequence pair tasks, such as question answering or sequence classification.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs representing the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional list of IDs representing the second sequence for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs where each ID corresponds to its respective token in token_ids_0 and token_ids_1.
        """
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. nllb does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.

        """

        # Define special tokens for separation and classification
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If only one sequence is provided
        if token_ids_1 is None:
            # Return a list of zeros based on the length of cls + token_ids_0 + sep
            return len(cls + token_ids_0 + sep) * [0]
        
        # If two sequences are provided
        # Return a list of zeros based on the length of cls + token_ids_0 + sep + sep + token_ids_1 + sep
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        # Check if source and target languages are specified
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model.")
        
        # Set the source language
        self.src_lang = src_lang
        
        # Generate inputs by calling the model with additional special tokens and other kwargs
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        
        # Adjust target language format if not already formatted
        if "__" not in tgt_lang:
            tgt_lang = f"__{tgt_lang}__"
        
        # Convert the target language to its corresponding ID
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        
        # Add the target language ID to inputs
        inputs["forced_bos_token_id"] = tgt_lang_id
        
        return inputs

    def get_vocab(self):
        # Create a vocabulary dictionary mapping token IDs to tokens
        vocab = {
            self.convert_ids_to_tokens(i): i for i in range(self.fairseq_offset, self.vocab_size + self.fairseq_offset)
        }
        
        # Update vocabulary with any additional tokens
        vocab.update(self.added_tokens_encoder)
        
        return vocab

    @property
    def unk_token_length(self):
        # Return the length of the encoded representation of the unknown token
        return len(self.sp_model.encode(str(self.unk_token)))

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_spm_processor
    def get_spm_processor(self, from_slow=False):
        # Initialize SentencePiece tokenizer with specified parameters
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        
        # Load vocabulary and return tokenizer instance for legacy or slow mode
        if self.legacy or from_slow:  # no dependency on protobuf
            tokenizer.Load(self.vocab_file)
            return tokenizer
        
        # Load serialized SentencePiece model for faster processing
        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            model_pb2 = import_protobuf(f"The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)")
            model = model_pb2.ModelProto.FromString(sp_model)
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            sp_model = model.SerializeToString()
            tokenizer.LoadFromSerializedProto(sp_model)
        
        return tokenizer

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.tokenize
    def tokenize(self, text: "TextInput", **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the
        first token is special.
        """
        # 如果 self.legacy 为真或者输入文本为空，则调用父类的tokenize方法处理
        if self.legacy or len(text) == 0:
            return super().tokenize(text, **kwargs)

        # 将特殊的 SPIECE_UNDERLINE 替换为空格
        text = text.replace(SPIECE_UNDERLINE, " ")
        # 如果设置了 add_prefix_space，则在文本前加上 SPIECE_UNDERLINE
        if self.add_prefix_space:
            text = SPIECE_UNDERLINE + text

        # 使用父类的tokenize方法对处理后的文本进行分词
        tokens = super().tokenize(text, **kwargs)

        # 如果分词后的tokens长度大于1，并且第一个token是 SPIECE_UNDERLINE，第二个token是特殊标记中的一个，则去掉第一个token
        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return tokens

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._tokenize
    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        # 使用 sentencepiece 模型对文本进行编码，返回字符串形式的tokens列表
        tokens = self.sp_model.encode(text, out_type=str)
        # 如果 self.legacy 为真或者文本不以 SPIECE_UNDERLINE 或空格开头，则直接返回tokens
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens

        # 1. 对字符串进行编码并加上前缀，例如 "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. 从编码结果中移除 unk_token
        return tokens[self.unk_token_length :] if len(tokens) >= self.unk_token_length else tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用 vocab 将 token 转换为对应的 id
        spm_id = self.sp_model.PieceToId(token)

        # 如果 spm_id 为0，则返回未知token的id
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用 vocab 将 index 转换为对应的 token
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        # 如果 tokens 的第一个元素以 SPIECE_UNDERLINE 开头，并且设置了 add_prefix_space，则去掉第一个字符
        if tokens[0].startswith(SPIECE_UNDERLINE) and self.add_prefix_space:
            tokens[0] = tokens[0][1:]

        # 将 tokens 中的 SPIECE_UNDERLINE 替换为空格，然后连接成字符串并去除首尾空格
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    # Copied from transformers.models.nllb.tokenization_nllb.NllbTokenizer.save_vocabulary
    # 将词汇表保存到指定目录下的文件中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出词汇表文件的路径，根据可选的前缀和默认文件名构成
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与输出路径不同且当前词汇表文件存在，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将序列化后的模型内容写入输出路径
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回输出词汇表文件路径的元组
        return (out_vocab_file,)

    # 从 transformers.models.nllb.tokenization_nllb.NllbTokenizer.prepare_seq2seq_batch 复制，配置序列到序列任务的批处理
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "eng",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "fra",
        **kwargs,
    ) -> BatchEncoding:
        # 设置源语言和目标语言
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        # 调用父类方法，准备序列到序列的批处理数据
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    # 从 transformers.models.nllb.tokenization_nllb.NllbTokenizer._switch_to_input_mode 复制，切换到输入模式
    def _switch_to_input_mode(self):
        # 设置特殊令牌以适应源语言
        return self.set_src_lang_special_tokens(self.src_lang)

    # 从 transformers.models.nllb.tokenization_nllb.NllbTokenizer._switch_to_target_mode 复制，切换到目标模式
    def _switch_to_target_mode(self):
        # 设置特殊令牌以适应目标语言
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    # 设置特殊令牌以适应源语言
    def set_src_lang_special_tokens(self, src_lang) -> None:
        """重设特殊令牌以适应源语言设定。
        前缀=[src_lang_code]，后缀=[eos]
        """
        # 将当前语言代码转换为对应的标识符
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)
        self.init_kwargs["src_lang"] = src_lang

        # 如果当前语言代码等于未知标记的标识符，则记录警告
        if self.cur_lang_code == self.unk_token_id:
            logger.warning_once(
                f"`src_lang={src_lang}` has not be found in the vocabulary. Behaviour will probably be unexpected because the language token id will be replaced by the unknown token id."
            )

        # 设置前缀特殊令牌为当前语言代码，后缀特殊令牌为结束符标识符
        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

    # https://github.com/facebookresearch/fairseq2/blob/c53f18e6be6b8b46b722f2249b8397b7eccd7ad3/src/fairseq2/models/nllb/tokenizer.py#L112-L116
    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target lang setting.
        Prefix=[eos, tgt_lang_code] and suffix=[eos].
        """
        # 将当前语言代码转换为对应的词汇表中的 ID
        self.cur_lang_code = self.convert_tokens_to_ids(lang)
        
        # 将目标语言设置更新到初始化参数中
        self.init_kwargs["tgt_lang"] = lang

        # 如果当前语言代码等于未知标记的 ID，则记录警告日志
        if self.cur_lang_code == self.unk_token_id:
            logger.warning_once(
                f"`tgt_lang={lang}` has not be found in the vocabulary. Behaviour will probably be unexpected because the language token id will be replaced by the unknown token id."
            )

        # 设置前缀特殊标记为 [eos, 当前语言代码]
        self.prefix_tokens = [self.eos_token_id, self.cur_lang_code]
        
        # 设置后缀特殊标记为 [eos]
        self.suffix_tokens = [self.eos_token_id]
```