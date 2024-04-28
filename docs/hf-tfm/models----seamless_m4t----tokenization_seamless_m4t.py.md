# `.\transformers\models\seamless_m4t\tokenization_seamless_m4t.py`

```py
# 导入所需的模块和库
import os  # 导入操作系统模块
from shutil import copyfile  # 导入文件复制函数
from typing import Any, Dict, List, Optional, Tuple, Union  # 导入类型提示相关的类和函数

import sentencepiece as spm  # 导入sentencepiece模块

from ...convert_slow_tokenizer import import_protobuf  # 导入Protobuf转换函数
from ...tokenization_utils import (  # 导入分词相关的函数和类
    BatchEncoding,  # 批量编码类
    PreTokenizedInput,  # 预分词输入类
    PreTrainedTokenizer,  # 预训练分词器基类
    TextInput,  # 文本输入类
)
from ...tokenization_utils_base import AddedToken  # 导入附加标记类
from ...utils import PaddingStrategy, logging  # 导入填充策略和日志相关函数

# 获取logger对象
logger = logging.get_logger(__name__)

# 预训练模型词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/hf-seamless-m4t-medium": (
            "https://huggingface.co/facebook/hf-seamless-m4t-medium/blob/main/sentencepiece.bpe.model"
        ),
    }
}

# SentencePiece标记连接符
SPIECE_UNDERLINE = "▁"

# 词汇文件名称
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

# 预训练位置嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/hf-seamless-m4t-medium": 2048,
}


class SeamlessM4TTokenizer(PreTrainedTokenizer):
    """
    构建 SeamlessM4T 分词器。

    从 `RobertaTokenizer` 和 `XLNetTokenizer` 改编而来。基于
    [SentencePiece](https://github.com/google/sentencepiece)。

    分词方法为 `<language code> <tokens> <eos>` 用于源语言文档，以及 `<eos> <language
    code> <tokens> <eos>` 用于目标语言文档。

    示例:

    ```python
    >>> from transformers import SeamlessM4TTokenizer

    >>> tokenizer = SeamlessM4TTokenizer.from_pretrained(
    ...     "facebook/hf-seamless-m4t-medium", src_lang="eng", tgt_lang="fra"
    ... )
    >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
    >>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
    >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
    ```py
    """
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
    """

    # List of names of vocabulary files
    vocab_files_names = VOCAB_FILES_NAMES
    # List of maximum model input sizes
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # Dictionary mapping pretrained vocabulary files
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # List of model input names
    model_input_names = ["input_ids", "attention_mask"]

    # List of prefix tokens
    prefix_tokens: List[int] = []
    # List of suffix tokens
    suffix_tokens: List[int] = []
    # 定义一个名为 __init__ 的初始化函数
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
        **kwargs,
    ):
        # 如果 sp_model_kwargs 为 None，则将其设为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # 将 legacy 设为 False，这是一个未使用的参数
        self.legacy = False
        # 保存 vocab_file 属性
        self.vocab_file = vocab_file
    
        # 获取 spm 处理器
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))
    
        # 定义了一个 _added_tokens_decoder 字典，用于存储 padding、未知、开始、结束标记的索引及值
        self._added_tokens_decoder = {
            0: AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token,
            1: AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token,
            2: AddedToken(bos_token, special=True) if isinstance(bos_token, str) else bos_token,
            3: AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token,
        }
    
        # 定义 fairseq 偏移量为 1
        self.fairseq_offset = 1
    
        # 获取 sp_model 的大小
        self.sp_model_size = len(self.sp_model)
    
        # 设置源语言和目标语言标记
        self._src_lang = f"__{src_lang}__" if "__" not in src_lang else src_lang
        self._tgt_lang = f"__{tgt_lang}__" if "__" not in tgt_lang else tgt_lang
    
        # 调用父类的初始化方法
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
            **kwargs,
        )
    
        # 设置源语言和目标语言的特殊标记
        self.set_src_lang_special_tokens(self._src_lang)
        self.set_tgt_lang_special_tokens(self._tgt_lang)
    
    # 定义一个 __getstate__ 方法，用于在序列化时保存模型状态
    def __getstate__(self):
        # 复制当前对象的字典
        state = self.__dict__.copy()
        # 将 sp_model 设为 None，并将序列化后的模型原型保存在 sp_model_proto 中
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state
    
    # 定义一个 __setstate__ 方法，用于在反序列化时恢复模型状态
    def __setstate__(self, state):
        # 将状态字典赋值给当前对象
        self.__dict__ = state
        # 从 sp_model_proto 中恢复 sp_model
        self.sp_model = self.get_spm_processor()
    # 重写对象的状态，将字典参数赋值给对象的 __dict__ 属性
    def __setstate__(self, d):
        self.__dict__ = d

        # 为了向后兼容，如果对象没有 sp_model_kwargs 属性，则将其赋值为空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 根据 sp_model_kwargs 创建 SentencePieceProcessor 对象并加载序列化的 sp_model_proto
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    # 返回 sp_model 的词汇量大小
    @property
    def vocab_size(self):
        return len(self.sp_model)

    # 对象的调用方法，接受多个参数，具体作用需要根据实际使用场景来解释
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        ...

    # 返回对象的 src_lang 属性
    @property
    def src_lang(self) -> str:
        return self._src_lang

    # 设置对象的 src_lang 属性，如果新值不含有 "__"，则在新值两边加上 "__"
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        if "__" not in new_src_lang:
            self._src_lang = f"__{new_src_lang}__"
        else:
            self._src_lang = new_src_lang
        # 调用方法设置 src_lang 的特殊标记
        self.set_src_lang_special_tokens(self._src_lang)

    # 返回对象的 tgt_lang 属性
    @property
    def tgt_lang(self) -> str:
        return self._tgt_lang

    # 设置对象的 tgt_lang 属性，如果新值不含有 "__"，则在新值两边加上 "__"
    @tgt_lang.setter
    def tgt_lang(self, new_tgt_lang: str) -> None:
        if "__" not in new_tgt_lang:
            self._tgt_lang = f"__{new_tgt_lang}__"
        else:
            self._tgt_lang = new_tgt_lang
        # 调用方法设置 tgt_lang 的特殊标记
        self.set_tgt_lang_special_tokens(self._tgt_lang)

    # 获取特殊标记的掩码，具体作用需要根据实际使用场景来解释
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
            # If the token list already has special tokens, return the special token mask using the superclass method
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Create lists of ones for prefix and suffix tokens based on their lengths
        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)

        if token_ids_1 is None:
            # If there is no second list of token IDs, return a list with prefix tokens, sequence tokens, and suffix tokens
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        
        # If there are two lists of token IDs, return a list with prefix tokens, sequence tokens from both lists, and suffix tokens
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
            # If there is no second list of token IDs, return a list with prefix tokens, token_ids_0, and suffix tokens
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        
        # If there are two lists of token IDs, return a list with prefix tokens, token_ids_0, token_ids_1, and suffix tokens
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    # Copied from transformers.models.nllb.tokenization_nllb.NllbTokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def create_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
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

        # Define the separator token
        sep = [self.sep_token_id]
        # Define the classification token
        cls = [self.cls_token_id]

        # If only one sequence is provided
        if token_ids_1 is None:
            # Return a list of zeros of the length of the combined tokens and special tokens
            return len(cls + token_ids_0 + sep) * [0]
        # If two sequences are provided
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model.")
        # Set the source language and prepare inputs for the generation function
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        if "__" not in tgt_lang:
            tgt_lang = f"__{tgt_lang}__"
        # Get the token ID for the target language
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def get_vocab(self):
        # Create a vocabulary by mapping token IDs to tokens
        vocab = {
            self.convert_ids_to_tokens(i): i for i in range(self.fairseq_offset, self.vocab_size + self.fairseq_offset)
        }
        # Add any additional tokens to the vocabulary
        vocab.update(self.added_tokens_encoder)
        return vocab

    @property
    def unk_token_length(self):
        # Calculate the length of the unknown token by encoding it using the SentencePiece model
        return len(self.sp_model.encode(str(self.unk_token)))

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_spm_processor
    def get_spm_processor(self, from_slow=False):
        # Get the SentencePiece processor with given model parameters
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        if self.legacy or from_slow:  # no dependency on protobuf
            # Load the vocabulary file for the tokenizer
            tokenizer.Load(self.vocab_file)
            return tokenizer

        with open(self.vocab_file, "rb") as f:
            # Read the contents of the vocabulary file
            sp_model = f.read()
            model_pb2 = import_protobuf(f"The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)")
            model = model_pb2.ModelProto.FromString(sp_model)
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            sp_model = model.SerializeToString()
            # Load the serialized model into the tokenizer
            tokenizer.LoadFromSerializedProto(sp_model)
        return tokenizer

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.tokenize
    # 将文本转换为标记列表。如果 self.legacy 被设置为 False，除非第一个标记是特殊标记，否则会添加前缀标记。
    def tokenize(self, text: "TextInput", add_special_tokens=False, **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the
        first token is special.
        """
        # 如果 self.legacy 被设置为 True 或者文本长度为 0，则调用超类的 tokenize 方法
        if self.legacy or len(text) == 0:
            return super().tokenize(text, **kwargs)

        # 在文本中的每个标记之前添加 SPIECE_UNDERLINE，并调用超类的 tokenize 方法
        tokens = super().tokenize(SPIECE_UNDERLINE + text.replace(SPIECE_UNDERLINE, " "), **kwargs)

        # 如果标记数量大于 1 并且第一个标记是 SPIECE_UNDERLINE，并且第二个标记是特殊标记，则去除第一个标记
        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return tokens

    # 从 transformers.models.t5.tokenization_t5.T5Tokenizer._tokenize 复制而来
    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        # 使用 sentencepiece 模型将文本编码为标记序列
        tokens = self.sp_model.encode(text, out_type=str)
        # 如果 self.legacy 为 True 或者文本不以 SPIECE_UNDERLINE 或空格开头，则直接返回标记序列
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens

        # 1. 编码字符串 + 前缀，例如："<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. 从 ['
    # 保存词汇表到指定目录下
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则报错
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出词汇表文件路径，如果有前缀则添加前缀，否则直接使用文件名
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件与输出文件不在同一路径下并且当前词汇表文件存在，则复制当前词汇表文件到输出文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将序列化的词汇模型内容写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回输出文件路径
        return (out_vocab_file,)

    # 为序列到序列任务准备批处理数据
    # src_texts：源文本列表，src_lang：源语言，默认为英语，tgt_texts：目标文本列表，tgt_lang：目标语言，默认为法语
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "eng",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "fra",
        **kwargs,
    ) -> BatchEncoding:
        # 设置源语言和目标语言，并调用父类方法准备批处理数据
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    # 切换到输入模式，设置特殊标记为源语言标记
    def _switch_to_input_mode(self):
        return self.set_src_lang_special_tokens(self.src_lang)

    # 切换到目标模式，设置特殊标记为目标语言标记
    def _switch_to_target_mode(self):
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    # 设置特殊标记为源语言标记
    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting.
        Prefix=[src_lang_code], suffix = [eos]
        """
        # 将当前语言编码设置为源语言编码，并更新初始化参数中的源语言设置
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)
        self.init_kwargs["src_lang"] = src_lang

        # 如果当前语言编码在词汇表中找不到，则发出警告
        if self.cur_lang_code == self.unk_token_id:
            logger.warning_once(
                f"`src_lang={src_lang}` has not be found in the vocabulary. Behaviour will probably be unexpected because the language token id will be replaced by the unknown token id."
            )

        # 设置前缀和后缀特殊标记
        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

    # https://github.com/facebookresearch/fairseq2/blob/c53f18e6be6b8b46b722f2249b8397b7eccd7ad3/src/fairseq2/models/nllb/tokenizer.py#L112-L116
    # 设置目标语言专用标记
    # 前缀=[eos, tgt_lang_code]，后缀=[eos]
    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target lang setting.
        Prefix=[eos, tgt_lang_code] and suffix=[eos].
        """
        # 将当前语言代码设置为目标语言的标记 id
        self.cur_lang_code = self.convert_tokens_to_ids(lang)
        # 更新初始化参数中的目标语言
        self.init_kwargs["tgt_lang"] = lang

        # 如果当前语言代码等于未知标记 id，则记录警告信息
        if self.cur_lang_code == self.unk_token_id:
            logger.warning_once(
                f"`tgt_lang={lang}` has not be found in the vocabulary. Behaviour will probably be unexpected because the language token id will be replaced by the unknown token id."
            )

        # 设置前缀标记为[eos, tgt_lang_code]
        self.prefix_tokens = [self.eos_token_id, self.cur_lang_code]
        # 设置后缀标记为[eos]
        self.suffix_tokens = [self.eos_token_id]
```