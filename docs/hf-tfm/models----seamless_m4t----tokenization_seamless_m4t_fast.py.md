# `.\models\seamless_m4t\tokenization_seamless_m4t_fast.py`

```py
# 导入必要的模块和库
import os  # 导入操作系统模块
from shutil import copyfile  # 导入文件复制函数
from typing import List, Optional, Tuple, Union  # 导入类型提示相关的类和函数

from tokenizers import processors  # 导入 tokenizers 库中的 processors 模块

# 导入 tokenization_utils_fast 模块中的一些类和函数
from ...tokenization_utils import (
    BatchEncoding,  # 批编码类
    PreTokenizedInput,  # 预分词输入类
    TextInput,  # 文本输入类
)
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入优化的预训练 tokenizer 类
from ...utils import PaddingStrategy, is_sentencepiece_available, logging  # 导入填充策略、检查 sentencepiece 可用性和日志模块

# 如果 sentencepiece 可用，则导入 SeamlessM4TTokenizer 类，否则设为 None
if is_sentencepiece_available():
    from .tokenization_seamless_m4t import SeamlessM4TTokenizer
else:
    SeamlessM4TTokenizer = None

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 定义用于 SeamlessM4TTokenizerFast 类的词汇文件名称映射
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/hf-seamless-m4t-medium": "https://huggingface.co/facebook/hf-seamless-m4t-medium/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "facebook/hf-seamless-m4t-medium": "https://huggingface.co/facebook/hf-seamless-m4t-medium/resolve/main/tokenizer.json",
    },
}

# 预训练位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/hf-seamless-m4t-medium": 2048,
}


class SeamlessM4TTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速” SeamlessM4T tokenizer（基于 HuggingFace 的 tokenizers 库）。基于 BPE 模型。
    该 tokenizer 继承自 PreTrainedTokenizerFast，包含大多数主要方法。用户应参考超类以获取更多关于这些方法的信息。

    分词方法为：<language code> <tokens> <eos> 用于源语言文档，
    和 <eos> <language code> <tokens> <eos> 用于目标语言文档。

    示例：

    ```
    >>> from transformers import SeamlessM4TTokenizerFast

    >>> tokenizer = SeamlessM4TTokenizerFast.from_pretrained(
    ...     "facebook/hf-seamless-m4t-medium", src_lang="eng", tgt_lang="fra"
    ... )
    >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
    >>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
    >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
    ```
    """
    pass  # 类定义暂无额外代码，只是简单继承了 PreTrainedTokenizerFast
    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        tokenizer_file (`str`, *optional*):
            The path to a tokenizer file to use instead of the vocab file.
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
        src_lang (`str`, *optional*, defaults to `"eng"`):
            The language to use as source language for translation.
        tgt_lang (`str`, *optional*, defaults to `"fra"`):
            The language to use as target language for translation.
        additional_special_tokens (tuple or list of `str` or `tokenizers.AddedToken`, *optional*):
            A tuple or a list of additional special tokens.

    Define constants and variables related to tokenizer and vocabulary files.
    These constants are typically provided by the library and tailored for specific models.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    # Assigns predefined constant `VOCAB_FILES_NAMES` to `vocab_files_names`

    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # Assigns predefined constant `PRETRAINED_VOCAB_FILES_MAP` to `pretrained_vocab_files_map`

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # Assigns predefined constant `PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES` to `max_model_input_sizes`

    slow_tokenizer_class = SeamlessM4TTokenizer
    # Assigns `SeamlessM4TTokenizer` class to `slow_tokenizer_class`, presumably for tokenization tasks

    model_input_names = ["input_ids", "attention_mask"]
    # Defines a list of strings representing model input names

    prefix_tokens: List[int] = []
    # Initializes an empty list `prefix_tokens` intended to hold integer values

    suffix_tokens: List[int] = []
    # Initializes an empty list `suffix_tokens` intended to hold integer values
    # 初始化方法，设置各种参数和特殊标记，继承自父类Tokenizer
    def __init__(
        self,
        vocab_file=None,  # 词汇文件路径，默认为None
        tokenizer_file=None,  # 分词器文件路径，默认为None
        bos_token="<s>",  # 开始标记，默认为"<s>"
        eos_token="</s>",  # 结束标记，默认为"</s>"
        sep_token="</s>",  # 分隔标记，默认为"</s>"
        cls_token="<s>",  # 类别标记，默认为"<s>"
        unk_token="<unk>",  # 未知标记，默认为"<unk>"
        pad_token="<pad>",  # 填充标记，默认为"<pad>"
        src_lang="eng",  # 源语言，默认为"eng"
        tgt_lang="fra",  # 目标语言，默认为"fra"
        additional_special_tokens=None,  # 额外的特殊标记列表，默认为None
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化方法，传入所有参数
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        # 设置实例的词汇文件路径
        self.vocab_file = vocab_file
        # 根据源语言设置私有属性，以双下划线包围
        self._src_lang = f"__{src_lang}__" if "__" not in src_lang else src_lang
        # 根据目标语言设置私有属性，以双下划线包围
        self._tgt_lang = f"__{tgt_lang}__" if "__" not in tgt_lang else tgt_lang
        # 调用方法设置源语言特殊标记
        self.set_src_lang_special_tokens(self._src_lang)
        # 调用方法设置目标语言特殊标记
        self.set_tgt_lang_special_tokens(self._tgt_lang)

    @property
    # 返回是否可以保存慢速分词器的布尔值
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    @property
    # 返回源语言的私有属性
    # 摘录自transformers.models.nllb.tokenization_nllb.NllbTokenizer.src_lang
    def src_lang(self) -> str:
        return self._src_lang

    @src_lang.setter
    # 设置新的源语言，更新私有属性和特殊标记
    def src_lang(self, new_src_lang: str) -> None:
        # 如果新的源语言不含双下划线，则用双下划线包围
        if "__" not in new_src_lang:
            self._src_lang = f"__{new_src_lang}__"
        else:
            self._src_lang = new_src_lang
        # 更新源语言的特殊标记
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    # 返回目标语言的私有属性
    def tgt_lang(self) -> str:
        return self._tgt_lang

    @tgt_lang.setter
    # 设置新的目标语言，更新私有属性和特殊标记
    def tgt_lang(self, new_tgt_lang: str) -> None:
        # 如果新的目标语言不含双下划线，则用双下划线包围
        if "__" not in new_tgt_lang:
            self._tgt_lang = f"__{new_tgt_lang}__"
        else:
            self._tgt_lang = new_tgt_lang
        # 更新目标语言的特殊标记
        self.set_tgt_lang_special_tokens(self._tgt_lang)

    # 构建带有特殊标记的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. The special tokens depend on calling set_lang.

        An SeamlessM4T sequence has the following format, where `X` represents the sequence:

        - `input_ids` (for encoder) `[src_lang_code] X [eos]`
        - `decoder_input_ids`: (for decoder) `[eos, tgt_lang_code] X [eos]`

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            # If only one sequence is provided, concatenate with prefix and suffix tokens
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # If two sequences are provided, concatenate with prefix, both sequences, and suffix tokens
        # Note: This is for API consistency; in practice, pairs of sequences are not expected.
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    # Copied from transformers.models.nllb.tokenization_nllb_fast.NllbTokenizerFast.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
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

        if token_ids_1 is None:
            # Return a list of zeros corresponding to the length of cls + token_ids_0 + sep
            return len(cls + token_ids_0 + sep) * [0]
        # Return a list of zeros corresponding to the length of cls + token_ids_0 + sep + sep + token_ids_1 + sep
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""

        if src_lang is None or tgt_lang is None:
            # Raise an error if either source language or target language is missing
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")

        # Set the source language attribute
        self.src_lang = src_lang

        # Generate model inputs with special tokens added
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)

        if "__" not in tgt_lang:
            # Ensure target language has proper formatting with double underscores
            tgt_lang = f"__{tgt_lang}__"

        # Convert target language name to token ID
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)

        # Add forced beginning-of-sequence token ID to inputs
        inputs["forced_bos_token_id"] = tgt_lang_id

        return inputs
    # 从 transformers.models.nllb.tokenization_nllb_fast.NllbTokenizerFast.prepare_seq2seq_batch 复制而来，用于准备序列到序列的批次数据
    # 将源语言和目标语言设置为默认值"eng"和"fra"，并调用父类方法准备序列到序列的批次数据
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "eng",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "fra",
        **kwargs,
    ) -> BatchEncoding:
        self.src_lang = src_lang  # 设置源语言
        self.tgt_lang = tgt_lang  # 设置目标语言
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    # 从 transformers.models.nllb.tokenization_nllb_fast.NllbTokenizerFast._switch_to_input_mode 复制而来
    # 切换为输入模式，设置当前语言的特殊标记
    def _switch_to_input_mode(self):
        return self.set_src_lang_special_tokens(self.src_lang)

    # 从 transformers.models.nllb.tokenization_nllb_fast.NllbTokenizerFast._switch_to_target_mode 复制而来
    # 切换为目标模式，设置当前语言的特殊标记
    def _switch_to_target_mode(self):
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    # 设置源语言的特殊标记
    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting.
        Prefix=[src_lang_code], suffix = [eos]
        """
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)  # 将源语言代码转换为对应的标记 ID

        # 如果当前语言代码等于未知标记 ID，则发出警告
        if self.cur_lang_code == self.unk_token_id:
            logger.warning_once(
                f"`tgt_lang={src_lang}` has not be found in the `vocabulary`. Behaviour will probably be unexpected because the language token id will be replaced by the unknown token id."
            )

        self.init_kwargs["src_lang"] = src_lang  # 更新初始化参数中的源语言设置

        self.prefix_tokens = [self.cur_lang_code]  # 设置前缀特殊标记为当前语言代码
        self.suffix_tokens = [self.eos_token_id]  # 设置后缀特殊标记为终止标记 ID

        # 将标记 ID 转换为对应的字符串形式
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        # 更新分词器的后处理器，使用模板处理方式
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,  # 单句模板
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,  # 双句模板
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),  # 特殊标记映射
        )
    # 重设特殊标记为目标语言设置。
    # 前缀=[eos, tgt_lang_code]，后缀=[eos]。
    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        # 将当前语言代码转换为对应的标记 ID
        self.cur_lang_code = self.convert_tokens_to_ids(lang)

        # 如果当前语言代码等于未知标记 ID，则记录警告日志
        if self.cur_lang_code == self.unk_token_id:
            logger.warning_once(
                f"`tgt_lang={lang}` has not be found in the `vocabulary`. Behaviour will probably be unexpected because the language token id will be replaced by the unknown token id."
            )

        # 更新初始化参数中的目标语言设置
        self.init_kwargs["tgt_lang"] = lang

        # 设置前缀标记为 [eos, cur_lang_code]
        self.prefix_tokens = [self.eos_token_id, self.cur_lang_code]
        # 设置后缀标记为 [eos]
        self.suffix_tokens = [self.eos_token_id]

        # 将前缀和后缀标记 ID 转换为对应的字符串表示
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        # 设置 Tokenizer 的后处理器为 TemplateProcessing 对象
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            # 设置特殊标记列表，将标记字符串与其对应的标记 ID 组合成元组
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    # 从 transformers 库中的 NllbTokenizerFast 类的方法 save_vocabulary 复制而来
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果无法保存慢速 Tokenizer 的词汇表，则引发 ValueError 异常
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 检查保存目录是否存在，若不存在则记录错误日志并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory.")
            return
        
        # 定义输出的词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与目标路径不同，则复制当前词汇表文件到目标路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回输出词汇表文件路径的元组形式
        return (out_vocab_file,)
    
    # 类方法，用于从预训练模型或路径加载模型的初始化方法
    @classmethod
    def _from_pretrained(
        cls,
        resolved_vocab_files,
        pretrained_model_name_or_path,
        init_configuration,
        *init_inputs,
        token=None,
        cache_dir=None,
        local_files_only=False,
        _commit_hash=None,
        _is_local=False,
        **kwargs,
    ):
        # 方法主体部分包含复杂的初始化逻辑，此处省略不进行详细解释
        pass
    ):
        # 调用父类的_from_pretrained方法，初始化tokenizer对象
        tokenizer = super()._from_pretrained(
            resolved_vocab_files,
            pretrained_model_name_or_path,
            init_configuration,
            *init_inputs,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            _commit_hash=_commit_hash,
            _is_local=_is_local,
            **kwargs,
        )

        # 在从预训练模型加载后，确保设置源语言特殊标记
        tokenizer.set_src_lang_special_tokens(tokenizer._src_lang)
        # 在从预训练模型加载后，确保设置目标语言特殊标记
        tokenizer.set_tgt_lang_special_tokens(tokenizer._tgt_lang)

        # 返回初始化后的tokenizer对象
        return tokenizer

    def __call__(
        self,
        # 输入的文本可以是单一文本、预分词输入或其列表形式
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        # 可选的文本对，可以是单一文本、预分词输入或其列表形式
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        # 目标文本，可以是单一文本、预分词输入或其列表形式
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        # 可选的目标文本对，可以是单一文本、预分词输入或其列表形式
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        # 填充策略，可以是布尔值、字符串或填充策略对象
        padding: Union[bool, str, PaddingStrategy] = True,
        # 填充至的倍数，可选的整数值
        pad_to_multiple_of: Optional[int] = 2,
        # 源语言标识符，可选的字符串
        src_lang: Optional[str] = None,
        # 目标语言标识符，可选的字符串
        tgt_lang: Optional[str] = None,
        # 其它关键字参数
        **kwargs,
```