# `.\transformers\models\mbart\tokenization_mbart_fast.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 Facebook AI Research Team 作者和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入所需的库
import os
from shutil import copyfile
from typing import List, Optional, Tuple

from tokenizers import processors

# 导入自定义的模块
from ...tokenization_utils import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging

# 如果 sentencepiece 可用，则导入 MBartTokenizer，否则设为 None
if is_sentencepiece_available():
    from .tokenization_mbart import MBartTokenizer
else:
    MBartTokenizer = None

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/mbart-large-en-ro": (
            "https://huggingface.co/facebook/mbart-large-en-ro/resolve/main/sentencepiece.bpe.model"
        ),
        "facebook/mbart-large-cc25": (
            "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/sentencepiece.bpe.model"
        ),
    },
    "tokenizer_file": {
        "facebook/mbart-large-en-ro": "https://huggingface.co/facebook/mbart-large-en-ro/resolve/main/tokenizer.json",
        "facebook/mbart-large-cc25": "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/tokenizer.json",
    },
}

# 预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/mbart-large-en-ro": 1024,
    "facebook/mbart-large-cc25": 1024,
}

# Fairseq 语言代码列表
FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"]  # fmt: skip

# MBartTokenizerFast 类，继承自 PreTrainedTokenizerFast
class MBartTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" MBART tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    The tokenization method is `<tokens> <eos> <language code>` for source language documents, and `<language code>
    <tokens> <eos>` for target language documents.

    Examples:

    ```python
    >>> from transformers import MBartTokenizerFast

    >>> tokenizer = MBartTokenizerFast.from_pretrained(
    # 设置模型的配置参数，包括模型名称、源语言和目标语言
    ...     "facebook/mbart-large-en-ro", src_lang="en_XX", tgt_lang="ro_RO"
    ...
    # 定义示例的英语短语和期望的罗马尼亚翻译
    >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
    >>> expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"
    # 使用tokenizer处理输入，返回PyTorch张量
    >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_romanian, return_tensors="pt")
    ```

    # 定义一些常量和变量
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = MBartTokenizer

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    # 初始化函数，设置特殊标记和语言相关参数
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        src_lang=None,
        tgt_lang=None,
        additional_special_tokens=None,
        **kwargs,
    ):
        # 处理mask_token，使其在词之前包含空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 复制FAIRSEQ_LANGUAGE_CODES到_additional_special_tokens
        _additional_special_tokens = FAIRSEQ_LANGUAGE_CODES.copy()

        # 如果有额外的特殊标记，添加到_additional_special_tokens中
        if additional_special_tokens is not None:
            _additional_special_tokens.extend(
                [t for t in additional_special_tokens if t not in _additional_special_tokens]
            )

        # 调用父类的初始化函数，设置参数
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=_additional_special_tokens,
            **kwargs,
        )

        # 设置vocab_file和lang_code_to_id
        self.vocab_file = vocab_file
        self.lang_code_to_id = {
            lang_code: self.convert_tokens_to_ids(lang_code) for lang_code in FAIRSEQ_LANGUAGE_CODES
        }

        # 设置_src_lang、cur_lang_code和tgt_lang
        self._src_lang = src_lang if src_lang is not None else "en_XX"
        self.cur_lang_code = self.convert_tokens_to_ids(self._src_lang)
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    # 判断是否可以保存慢速tokenizer
    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    # 获取源语言
    @property
    def src_lang(self) -> str:
        return self._src_lang

    # 设置源语言
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或序列对构建模型输入，用于序列分类任务，通过连接和添加特殊标记。特殊标记取决于调用 set_lang。

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 具有适当特殊标记的输入 ID 列表。
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传递的两个序列创建用于序列对分类任务的掩码。mBART 不使用 token type ids，因此返回一个零列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 零列表。
        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """用于翻译流水线，为生成函数准备输入"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "ro_RO",
        **kwargs,
    ) -> BatchEncoding:
        # 设置源语言和目标语言
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        # 调用父类方法准备序列到序列的批处理
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        # 切换到输入模式，设置源语言特殊标记
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        # 切换到目标模式，设置目标语言特殊标记
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
        # 将当前语言代码设置为源语言代码
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)
        # 设置前缀和后缀标记
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]

        # 将前缀和后缀标记转换为字符串形式
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        # 更新分词器的后处理器，设置特殊标记
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
        # 将当前语言代码设置为目标语言代码
        self.cur_lang_code = self.convert_tokens_to_ids(lang)
        # 设置前缀和后缀标记
        self.prefix_tokens = []
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]

        # 将前缀和后缀标记转换为字符串形式
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        # 更新分词器的后处理器，设置特殊标记
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )
    # 保存词汇表到指定目录下，返回保存的文件路径
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果无法保存慢速分词器的词汇表，则抛出数值错误
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存目录不存在，则记录错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory.")
            return
        # 构建输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与输出词汇表文件路径不同，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回保存的文件路径
        return (out_vocab_file,)
```