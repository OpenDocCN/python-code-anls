# `.\transformers\models\mbart50\tokenization_mbart50.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 Facebook AI Research Team 作者和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言
# 导入所需的库
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

# 导入自定义的模块
from ...tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 SentencePiece 分词的下划线符号
SPIECE_UNDERLINE = "▁"

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/mbart-large-50-one-to-many-mmt": (
            "https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt/resolve/main/sentencepiece.bpe.model"
        ),
    }
}

# 预训练位置嵌入的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/mbart-large-50-one-to-many-mmt": 1024,
}

# Fairseq 语言代码列表
FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL", "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF", "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA", "gl_ES", "sl_SI"]  # fmt: skip

# MBart50Tokenizer 类，继承自 PreTrainedTokenizer
class MBart50Tokenizer(PreTrainedTokenizer):
    """
    Construct a MBart50 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            Path to the vocabulary file. 词汇表文件的路径
        src_lang (`str`, *optional*):
            A string representing the source language. 表示源语言的字符串
        tgt_lang (`str`, *optional*):
            A string representing the target language. 表示目标语言的字符串
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token. 序列结束标记
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens. 分隔符标记，用于从多个序列构建序列时使用
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens. 分类器标记，用于进行序列分类
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. 未知标记，用于处理不在词汇表中的标记
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths. 用于填充的标记
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict. 用于掩码值的标记
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout. 用于传递给 `SentencePieceProcessor.__init__()` 方法的参数
    Examples:

    ```python
    >>> from transformers import MBart50Tokenizer

    >>> tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")
    >>> src_text = " UN Chief Says There Is No Military Solution in Syria"
    >>> tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"
    >>> model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")
    # 定义一些变量和常量，用于模型初始化
    # VOCAB_FILES_NAMES: 词汇表文件名
    # PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES: 预训练位置嵌入的最大模型输入大小
    # PRETRAINED_VOCAB_FILES_MAP: 预训练词汇文件映射
    # model_input_names: 模型输入的名称列表
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]

    # 前缀和后缀的标记列表
    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    # 初始化函数，用于初始化模型
    def __init__(
        self,
        vocab_file,
        src_lang=None,
        tgt_lang=None,
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
        # 定义一个方法，用于设置 Mask token 的行为，使其像普通单词一样，即包含前面的空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 设置 SentencePiece 模型的参数，如果未提供则为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 添加额外的特殊 token 到参数中
        kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", []) or []
        kwargs["additional_special_tokens"] += [
            code for code in FAIRSEQ_LANGUAGE_CODES if code not in kwargs["additional_special_tokens"]
        ]

        # 使用 SentencePieceProcessor 加载词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        # fairseq 和 spm 的词汇必须对齐
        # 设置 fairseq token 到 id 的对应关系
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        self.fairseq_offset = 1

        # 设置语言代码到 id 的映射关系
        self.sp_model_size = len(self.sp_model)
        self.lang_code_to_id = {
            code: self.sp_model_size + i + self.fairseq_offset for i, code in enumerate(FAIRSEQ_LANGUAGE_CODES)
        }
        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset

        # 更新 fairseq token 到 id 的映射关系
        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        # 调用父类的初始化方法
        super().__init__(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

        # 设置当前源语言和对应的语言代码 id
        self._src_lang = src_lang if src_lang is not None else "en_XX"
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    # 定义 vocab_size 属性，返回词汇大小
    @property
    def vocab_size(self) -> int:
        return len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset + 1  # 加 1 是为了 mask token

    # 定义 src_lang 属性，返回源语言
    @property
    def src_lang(self) -> str:
        return self._src_lang

    # 设置 src_lang 属性
    @src_lang.setter
    # 设置源语言属性为新的源语言
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        # 调用方法设置源语言的特殊标记
        self.set_src_lang_special_tokens(self._src_lang)

    # 序列化对象状态
    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    # 反序列化对象状态
    def __setstate__(self, d: Dict) -> None:
        self.__dict__ = d

        # 为了向后兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用参数初始化 SentencePieceProcessor 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载词汇文件
        self.sp_model.Load(self.vocab_file)

    # 获取词汇表
    def get_vocab(self) -> Dict:
        # 创建词汇表，将 id 映射到 token
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 更新词汇表，添加额外的 token 编码器
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 对文本进行分词
    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)

    # 将 token 转换为 id
    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an id using the vocab."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # 如果 SP 模型返回 0，则返回未知 token id
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    # 将 id 转换为 token
    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    # 将 tokens 序列转换为字符串
    # 从 transformers.models.albert.tokenization_albert.AlbertTokenizer.convert_tokens_to_string 复制而来
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # 确保特殊标记不使用 sentencepiece 模型解码
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与输出路径不同且当前词汇表文件存在，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将序列化后的 sp_model 内容写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

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

        # 如果已经有特殊标记，则调用父类方法返回特殊标记掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 构建前缀和后缀的特殊标记掩码
        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        # 如果没有第二个序列，则返回前缀 + 序列0 + 后缀的特殊标记掩码
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        # 如果有第二个序列，则返回前缀 + 序列0 + 序列1 + 后缀的特殊标记掩码
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或一个序列对构建模型输入，用于序列分类任务，通过连接和添加特殊标记。一个 MBART-50 序列具有以下格式，其中 `X` 表示序列：

        - `input_ids`（用于编码器）`[src_lang_code] X [eos]`
        - `labels`:（用于解码器）`[tgt_lang_code] X [eos]`

        BOS 从不被使用。序列对不是预期的使用情况，但它们将被处理而不使用分隔符。

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 具有适当特殊标记的 [input IDs](../glossary#input-ids) 列表。
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # 我们不希望处理序列对，但为了 API 一致性保留了对序列对的逻辑
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """由翻译管道使用，为生成函数准备输入"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("此模型需要 `src_lang` 和 `tgt_lang` 进行翻译")
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
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang: str) -> None:
        """将特殊标记重置为源语言设置。前缀=[src_lang_code] 和后缀=[eos]。"""
        self.cur_lang_code_id = self.lang_code_to_id[src_lang]
        self.prefix_tokens = [self.cur_lang_code_id]
        self.suffix_tokens = [self.eos_token_id]
    # 设置目标语言的特殊标记，包括前缀和后缀
    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        """Reset the special tokens to the target language setting. prefix=[tgt_lang_code] and suffix=[eos]."""
        # 将当前语言代码设置为目标语言对应的 ID
        self.cur_lang_code_id = self.lang_code_to_id[tgt_lang]
        # 将前缀标记设置为当前语言代码的 ID
        self.prefix_tokens = [self.cur_lang_code_id]
        # 将后缀标记设置为结束标记的 ID
        self.suffix_tokens = [self.eos_token_id]
```