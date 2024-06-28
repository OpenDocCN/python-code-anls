# `.\models\mbart50\tokenization_mbart50.py`

```
# coding=utf-8
# 设置日志记录器以获取当前模块的日志记录器对象
import os
# 导入文件复制函数
from shutil import copyfile
# 导入类型提示相关库
from typing import Any, Dict, List, Optional, Tuple

# 导入 SentencePiece 库
import sentencepiece as spm

# 从 tokenization_utils 模块中导入必要的类和函数
from ...tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer
# 导入 logging 函数
from ...utils import logging

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# SentencePiece 使用的特殊字符
SPIECE_UNDERLINE = "▁"

# 词汇文件的命名映射，这里指定了词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

# 预训练模型的词汇文件映射，指定了预训练模型及其对应的词汇文件下载链接
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/mbart-large-50-one-to-many-mmt": (
            "https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt/resolve/main/sentencepiece.bpe.model"
        ),
    }
}

# 预训练模型的位置编码大小映射，指定了每个预训练模型的位置编码大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/mbart-large-50-one-to-many-mmt": 1024,
}

# Fairseq 使用的语言代码列表，包含多种语言代码
FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL", "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF", "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA", "gl_ES", "sl_SI"]  # fmt: skip

# MBart50Tokenizer 类，继承自 PreTrainedTokenizer 类
class MBart50Tokenizer(PreTrainedTokenizer):
    """
    Construct a MBart50 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        src_lang (`str`, *optional*):
            A string representing the source language.
        tgt_lang (`str`, *optional*):
            A string representing the target language.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
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
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
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
              BPE-dropout.

    Examples:

    ```python
    >>> from transformers import MBart50Tokenizer

    >>> tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")
    >>> src_text = " UN Chief Says There Is No Military Solution in Syria"
    >>> tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"
    >>> model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")
    # 定义模型的词汇文件名列表，从全局常量中获取
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义模型的最大输入大小列表，从全局常量中获取
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义预训练词汇文件映射字典，从全局常量中获取
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义模型的输入名称列表
    model_input_names = ["input_ids", "attention_mask"]

    # 前缀令牌列表的初始化
    prefix_tokens: List[int] = []
    # 后缀令牌列表的初始化
    suffix_tokens: List[int] = []

    # 初始化方法，用于创建模型对象
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
    ) -> None:
        # Mask token behave like a normal word, i.e. include the space before it
        # 如果 mask_token 是字符串，则设置为一个添加的标记，去除左侧空格，保留右侧空格；否则直接使用给定的 mask_token
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 初始化 sp_model_kwargs，如果未提供则为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 更新 kwargs 中的 additional_special_tokens，确保包含所有 FAIRSEQ_LANGUAGE_CODES 中的代码
        kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", []) or []
        kwargs["additional_special_tokens"] += [
            code for code in FAIRSEQ_LANGUAGE_CODES if code not in kwargs["additional_special_tokens"]
        ]

        # 使用给定的 sp_model_kwargs 创建 SentencePieceProcessor 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        
        # 加载指定路径的词汇文件到 sp_model 中
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        # 原始 fairseq 词汇和 spm 词汇必须是对齐的，初始化 fairseq_tokens_to_ids
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # fairseq 的偏移量，用于模仿 fairseq 与 spm 的对齐
        self.fairseq_offset = 1

        # 计算 sp_model 的大小
        self.sp_model_size = len(self.sp_model)

        # 创建语言代码到 ID 的映射，以及 ID 到语言代码的映射
        self.lang_code_to_id = {
            code: self.sp_model_size + i + self.fairseq_offset for i, code in enumerate(FAIRSEQ_LANGUAGE_CODES)
        }
        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}

        # 添加 <mask> 到 fairseq_tokens_to_ids 中
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset

        # 更新 fairseq_tokens_to_ids，将语言代码映射添加进去
        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)

        # 创建 fairseq_ids_to_tokens，将 fairseq_tokens_to_ids 的键值对颠倒
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        # 调用父类的初始化方法，设置各种特殊的语言标记和参数
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

        # 设置当前源语言的语言代码，如果未指定则默认为 "en_XX"
        self._src_lang = src_lang if src_lang is not None else "en_XX"
        
        # 获取当前源语言代码的 ID
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]

        # 设置源语言的特殊标记
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def vocab_size(self) -> int:
        # 返回词汇大小，包括 sp_model、语言代码和 fairseq 偏移量，再加上一个用于 mask token
        return len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset + 1  # Plus 1 for the mask token

    @property
    def src_lang(self) -> str:
        # 返回当前源语言代码
        return self._src_lang

    @src_lang.setter
    # 设置新的源语言，并更新特殊标记
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    # 返回对象的状态字典表示，排除 sp_model
    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    # 根据给定的状态字典恢复对象的状态
    def __setstate__(self, d: Dict) -> None:
        self.__dict__ = d

        # 兼容旧版本
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 根据 sp_model_kwargs 创建 SentencePieceProcessor 对象，并加载词汇表文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    # 返回词汇表，包括从 ID 到 token 的映射
    def get_vocab(self) -> Dict:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 使用 SentencePieceProcessor 对文本进行分词处理，返回 token 列表
    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)

    # 根据 token 获取对应的 ID，若 token 未知则返回未知 token 的 ID
    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an id using the vocab."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # 如果 SP 模型返回 0，则返回未知 token 的 ID
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    # 根据 ID 获取对应的 token，考虑 fairseq 偏移量
    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    # 将 tokens 序列转换为单个字符串
    # 保证特殊标记不通过 SentencePiece 模型解码
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
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
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 组合输出的词汇文件路径，考虑可选的文件名前缀
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇文件路径与输出路径不同且当前词汇文件存在，则复制文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇文件不存在，则将序列化后的词汇模型写入输出路径
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回输出路径的元组形式
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

        # 如果已经有特殊标记，则调用父类方法获取特殊标记的掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 创建前缀和后缀的特殊标记掩码列表
        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        
        # 如果只有一个 token_ids 列表，则返回前缀 + 序列 token + 后缀 的掩码
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        
        # 如果有两个 token_ids 列表，则返回前缀 + 第一个序列 token + 第二个序列 token + 后缀 的掩码
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        # 该方法用于构建包含特殊标记的输入 token 列表
        
        # 如果只有一个 token_ids 列表，则返回前缀 + 第一个序列 token + 后缀 的列表
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        
        # 如果有两个 token_ids 列表，则返回前缀 + 第一个序列 token + 第二个序列 token + 后缀 的列表
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An MBART-50 sequence has the following format, where `X` represents the sequence:

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
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # If only one sequence (`token_ids_1` is None), concatenate with prefix and suffix tokens
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # If processing a pair of sequences, concatenate both sequences with prefix and suffix tokens
        # Although pairs are not the expected use case, handle it for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        # Check if source language and target language are provided
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = src_lang
        # Generate model inputs by adding special tokens to raw inputs
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        # Convert target language to its corresponding token ID
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        # Set the forced beginning-of-sequence token ID for decoding
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
        # Set the source and target languages for the batch
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        # Call the superclass method to prepare the sequence-to-sequence batch
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        # Set the special tokens for the current source language
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        # Set the special tokens for the current target language
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang: str) -> None:
        """Reset the special tokens to the source lang setting. prefix=[src_lang_code] and suffix=[eos]."""
        # Set the current language code ID to the provided source language
        self.cur_lang_code_id = self.lang_code_to_id[src_lang]
        # Set the prefix tokens to start with the source language code ID
        self.prefix_tokens = [self.cur_lang_code_id]
        # Set the suffix tokens to end with the end-of-sequence token ID
        self.suffix_tokens = [self.eos_token_id]
    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        """重设特殊标记以适应目标语言设置。前缀=[tgt_lang_code] 和 后缀=[eos]。"""
        # 将当前语言代码ID设置为目标语言对应的ID
        self.cur_lang_code_id = self.lang_code_to_id[tgt_lang]
        # 将前缀标记设为包含当前语言代码ID的列表
        self.prefix_tokens = [self.cur_lang_code_id]
        # 将后缀标记设为包含结束符标记ID的列表
        self.suffix_tokens = [self.eos_token_id]
```