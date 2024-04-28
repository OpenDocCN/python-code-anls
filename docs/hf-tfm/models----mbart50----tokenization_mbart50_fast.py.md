# `.\transformers\models\mbart50\tokenization_mbart50_fast.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归属于 Facebook AI Research Team 作者和 HuggingFace Inc. 团队
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入所需的库
import os
from shutil import copyfile
from typing import List, Optional, Tuple

from tokenizers import processors

# 导入自定义的模块
from ...tokenization_utils import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging

# 如果 sentencepiece 可用，则导入 MBart50Tokenizer，否则设为 None
if is_sentencepiece_available():
    from .tokenization_mbart50 import MBart50Tokenizer
else:
    MBart50Tokenizer = None

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/mbart-large-50-one-to-many-mmt": (
            "https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt/resolve/main/sentencepiece.bpe.model"
        ),
    },
    "tokenizer_file": {
        "facebook/mbart-large-50-one-to-many-mmt": (
            "https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt/resolve/main/tokenizer.json"
        ),
    },
}

# 预训练位置嵌入的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/mbart-large-50-one-to-many-mmt": 1024,
}

# Fairseq 语言代码列表
FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL", "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF", "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA", "gl_ES", "sl_SI"]  # fmt: skip

# MBart50TokenizerFast 类，继承自 PreTrainedTokenizerFast
class MBart50TokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" MBART tokenizer for mBART-50 (backed by HuggingFace's *tokenizers* library). Based on
    [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    # 定义类的初始化方法，用于初始化各种参数
    Args:
        vocab_file (`str`):
            词汇表文件的路径。
        src_lang (`str`, *optional*):
            表示源语言的字符串。
        tgt_lang (`str`, *optional*):
            表示目标语言的字符串。
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            结束序列的标记。
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            分隔符标记，在构建多个序列的序列时使用，例如用于序列分类或用于文本和问题的序列。还用作使用特殊标记构建的序列的最后一个标记。
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            分类器标记，用于进行序列分类（对整个序列进行分类，而不是对每个标记进行分类）。在使用特殊标记构建序列时，它是序列的第一个标记。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知标记。词汇表中不存在的标记无法转换为 ID，并将设置为此标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            用于填充的标记，例如在批处理不同长度的序列时使用。
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            用于屏蔽值的标记。这是在使用掩码语言建模训练此模型时使用的标记。这是模型将尝试预测的标记。

    Examples:

    ```python
    >>> from transformers import MBart50TokenizerFast

    >>> tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")
    >>> src_text = " UN Chief Says There Is No Military Solution in Syria"
    >>> tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"
    >>> model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")
    >>> # model(**model_inputs) should work
    ```"""

    # 定义类的属性
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = MBart50Tokenizer

    # 初始化方法，设置前缀和后缀标记为空列表
    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file=None,
        src_lang=None,
        tgt_lang=None,
        tokenizer_file=None,
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs,
    ):
        # 如果 mask token 是字符串，则创建一个 AddedToken 对象，保留前面的空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 设置额外的特殊 token，如果不存在则创建一个空列表
        kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", []) or []
        # 添加 Fairseq 语言代码中未包含的代码到额外的特殊 token 列表中
        kwargs["additional_special_tokens"] += [
            code for code in FAIRSEQ_LANGUAGE_CODES if code not in kwargs["additional_special_tokens"]
        ]

        # 调用父类的初始化方法，传入参数
        super().__init__(
            vocab_file,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        # 设置属性 vocab_file 为传入的 vocab_file
        self.vocab_file = vocab_file

        # 创建 Fairseq 语言代码到 ID 的映射
        self.lang_code_to_id = {
            lang_code: self.convert_tokens_to_ids(lang_code) for lang_code in FAIRSEQ_LANGUAGE_CODES
        }

        # 设置属性 _src_lang 为传入的 src_lang 或默认值 "en_XX"
        self._src_lang = src_lang if src_lang is not None else "en_XX"
        # 设置属性 tgt_lang 为传入的 tgt_lang
        self.tgt_lang = tgt_lang
        # 设置属性 cur_lang_code_id 为 _src_lang 对应的 ID
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
        # 调用方法设置 src_lang 的特殊 token
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查是否可以保存慢速 tokenizer，依据是否存在 vocab_file
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    @property
    def src_lang(self) -> str:
        # 返回属性 _src_lang
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        # 设置属性 _src_lang 为新的 src_lang，并调用方法设置 src_lang 的特殊 token
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. The special tokens depend on calling set_lang.

        An MBART-50 sequence has the following format, where `X` represents the sequence:

        - `input_ids` (for encoder) `[src_lang_code] X [eos]`
        - `labels`: (for decoder) `[tgt_lang_code] X [eos]`

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
            # 如果没有第二个 token 列表，则返回添加了特殊 token 的第一个 token 列表
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # 如果有第二个 token 列表，则返回添加了特殊 token 的两个 token 列表
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens
    # 准备用于序列到序列模型的批处理数据
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
        # 调用父类方法准备序列到序列模型的批处理数据
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    # 切换到输入模式
    def _switch_to_input_mode(self):
        return self.set_src_lang_special_tokens(self.src_lang)

    # 切换到目标模式
    def _switch_to_target_mode(self):
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    # 设置源语言的特殊标记
    def set_src_lang_special_tokens(self, src_lang: str) -> None:
        """Reset the special tokens to the source lang setting. prefix=[src_lang_code] and suffix=[eos]."""
        # 将当前语言代码转换为 ID
        self.cur_lang_code_id = self.convert_tokens_to_ids(src_lang)
        # 设置前缀和后缀标记
        self.prefix_tokens = [self.cur_lang_code_id]
        self.suffix_tokens = [self.eos_token_id]

        # 将前缀和后缀标记转换为字符串
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        # 设置后处理器
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    # 设置目标语言的特殊标记
    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        """Reset the special tokens to the target language setting. prefix=[src_lang_code] and suffix=[eos]."""
        # 将当前语言代码转换为 ID
        self.cur_lang_code_id = self.convert_tokens_to_ids(tgt_lang)
        # 设置前缀和后缀标记
        self.prefix_tokens = [self.cur_lang_code_id]
        self.suffix_tokens = [self.eos_token_id]

        # 将前缀和后缀标记转换为字符串
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        # 设置后处理器
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    # 构建翻译输入
    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        # 检查是否提供了源语言和目标语言
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        # 设置源语言
        self.src_lang = src_lang
        # 调用模型处理输入数据
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        # 将目标语言转换为 ID
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        # 设置强制开始标记的 ID
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs
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
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与输出词汇表文件路径不同，则复制当前词汇表文件到输出词汇表文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回保存的文件路径
        return (out_vocab_file,)
```