# `.\models\nllb\tokenization_nllb.py`

```
# 导入必要的模块
import os  # 导入操作系统模块
from shutil import copyfile  # 从 shutil 模块中导入 copyfile 函数
from typing import Any, Dict, List, Optional, Tuple  # 导入类型提示相关的类和函数

import sentencepiece as spm  # 导入 sentencepiece 库

from ...tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer  # 导入特定的类和函数
from ...utils import logging  # 导入日志记录工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

SPIECE_UNDERLINE = "▁"  # 定义特殊符号“▁”，用于处理语料中的词片段

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}  # 指定词汇表文件名的映射字典

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/nllb-200-distilled-600M": (
            "https://huggingface.co/facebook/nllb-200-distilled-600M/blob/main/sentencepiece.bpe.model"
        ),
    }
}  # 预训练模型与其词汇文件的映射，包含下载链接

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/nllb-200-distilled-600M": 1024,
}  # 预训练模型的位置嵌入尺寸映射表
# 定义了一个包含多种语言和脚本组合的列表，用于表示Fairseq支持的语言代码
FAIRSEQ_LANGUAGE_CODES = ['ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn', 'ajp_Arab', 'aka_Latn', 'amh_Ethi', 'apc_Arab', 'arb_Arab', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'asm_Beng', 'ast_Latn', 'awa_Deva', 'ayr_Latn', 'azb_Arab', 'azj_Latn', 'bak_Cyrl', 'bam_Latn', 'ban_Latn', 'bel_Cyrl', 'bem_Latn', 'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn', 'bod_Tibt', 'bos_Latn', 'bug_Latn', 'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn', 'cjk_Latn', 'ckb_Arab', 'crh_Latn', 'cym_Latn', 'dan_Latn', 'deu_Latn', 'dik_Latn', 'dyu_Latn', 'dzo_Tibt', 'ell_Grek', 'eng_Latn', 'epo_Latn', 'est_Latn', 'eus_Latn', 'ewe_Latn', 'fao_Latn', 'pes_Arab', 'fij_Latn', 'fin_Latn', 'fon_Latn', 'fra_Latn', 'fur_Latn', 'fuv_Latn', 'gla_Latn', 'gle_Latn', 'glg_Latn', 'grn_Latn', 'guj_Gujr', 'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hne_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ibo_Latn', 'ilo_Latn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan', 'kab_Latn', 'kac_Latn', 'kam_Latn', 'kan_Knda', 'kas_Arab', 'kas_Deva', 'kat_Geor', 'knc_Arab', 'knc_Latn', 'kaz_Cyrl', 'kbp_Latn', 'kea_Latn', 'khm_Khmr', 'kik_Latn', 'kin_Latn', 'kir_Cyrl', 'kmb_Latn', 'kon_Latn', 'kor_Hang', 'kmr_Latn', 'lao_Laoo', 'lvs_Latn', 'lij_Latn', 'lim_Latn', 'lin_Latn', 'lit_Latn', 'lmo_Latn', 'ltg_Latn', 'ltz_Latn', 'lua_Latn', 'lug_Latn', 'luo_Latn', 'lus_Latn', 'mag_Deva', 'mai_Deva', 'mal_Mlym', 'mar_Deva', 'min_Latn', 'mkd_Cyrl', 'plt_Latn', 'mlt_Latn', 'mni_Beng', 'khk_Cyrl', 'mos_Latn', 'mri_Latn', 'zsm_Latn', 'mya_Mymr', 'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva', 'nso_Latn', 'nus_Latn', 'nya_Latn', 'oci_Latn', 'gaz_Latn', 'ory_Orya', 'pag_Latn', 'pan_Guru', 'pap_Latn', 'pol_Latn', 'por_Latn', 'prs_Arab', 'pbt_Arab', 'quy_Latn', 'ron_Latn', 'run_Latn', 'rus_Cyrl', 'sag_Latn', 'san_Deva', 'sat_Beng', 'scn_Latn', 'shn_Mymr', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'smo_Latn', 'sna_Latn', 'snd_Arab', 'som_Latn', 'sot_Latn', 'spa_Latn', 'als_Latn', 'srd_Latn', 'srp_Cyrl', 'ssw_Latn', 'sun_Latn', 'swe_Latn', 'swh_Latn', 'szl_Latn', 'tam_Taml', 'tat_Cyrl', 'tel_Telu', 'tgk_Cyrl', 'tgl_Latn', 'tha_Thai', 'tir_Ethi', 'taq_Latn', 'taq_Tfng', 'tpi_Latn', 'tsn_Latn', 'tso_Latn', 'tuk_Latn', 'tum_Latn', 'tur_Latn', 'twi_Latn', 'tzm_Tfng', 'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab', 'uzn_Latn', 'vec_Latn', 'vie_Latn', 'war_Latn', 'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'yue_Hant', 'zho_Hans', 'zho_Hant', 'zul_Latn']  # fmt: skip

# 定义一个NllbTokenizer类，继承自PreTrainedTokenizer类
class NllbTokenizer(PreTrainedTokenizer):
    """
    构建一个NLLB分词器。

    从RobertaTokenizer和XLNetTokenizer进行了适配。
    基于SentencePiece（https://github.com/google/sentencepiece）。

    分词方法对于源语言文档是'<tokens> <eos> <language code>'，对于目标语言文档是'<language code> <tokens> <eos>'。

    示例：
    
    ```python
    >>> from transformers import NllbTokenizer

    >>> tokenizer = NllbTokenizer.from_pretrained(
    ```
    """
    # 定义函数参数和默认值，用于初始化一个特定的tokenizer对象
    vocab_file (`str`):
        # 词汇表文件的路径
        Path to the vocabulary file.
    
    bos_token (`str`, *optional*, defaults to `"<s>"`):
        # 在预训练期间用作序列开头的特殊token。在构建序列时，实际用于序列开头的是`cls_token`。
        The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
    
    eos_token (`str`, *optional*, defaults to `"</s>"`):
        # 序列结束的特殊token。在构建序列时，实际用于序列结尾的是`sep_token`。
        The end of sequence token.
    
    sep_token (`str`, *optional*, defaults to `"</s>"`):
        # 分隔符token，用于从多个序列构建一个序列，例如用于序列分类或文本与问题回答中。还用作使用特殊token构建序列的最后一个token。
        The separator token, which is used when building a sequence from multiple sequences.
    
    cls_token (`str`, *optional*, defaults to `"<s>"`):
        # 分类器token，在进行序列分类（整个序列的分类而不是每个token的分类）时使用。在使用特殊token构建序列时是序列的第一个token。
        The classifier token which is used when doing sequence classification.
    
    unk_token (`str`, *optional*, defaults to `"<unk>"`):
        # 未知token。如果一个token不在词汇表中，无法转换为ID，则会被设置为该token。
        The unknown token.
    
    pad_token (`str`, *optional*, defaults to `"<pad>"`):
        # 填充token，在批处理不同长度的序列时使用。
        The token used for padding.
    
    mask_token (`str`, *optional*, defaults to `"<mask>"`):
        # 掩码值token。在进行掩码语言建模训练时使用，模型将尝试预测此token。
        The token used for masking values.
    
    tokenizer_file (`str`, *optional*):
        # 要使用的分词器文件的路径，用于替代词汇表文件。
        The path to a tokenizer file to use instead of the vocab file.
    
    src_lang (`str`, *optional*):
        # 用作翻译的源语言。
        The language to use as source language for translation.
    
    tgt_lang (`str`, *optional*):
        # 用作翻译的目标语言。
        The language to use as target language for translation.
    
    sp_model_kwargs (`Dict[str, str]`):
        # 传递给模型初始化的额外关键字参数。
        Additional keyword arguments to pass to the model initialization.
    # 从预训练的位置编码大小中获取最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 从预训练的词汇文件映射中获取预训练的词汇文件
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 模型的输入名称列表，包括input_ids和attention_mask
    model_input_names = ["input_ids", "attention_mask"]

    # 前缀标记和后缀标记的初始化为空列表
    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    # 初始化函数，接受多个参数，包括词汇文件、特殊标记（如bos_token、eos_token等）、序列化的分词器文件等
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        tokenizer_file=None,
        src_lang=None,
        tgt_lang=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        additional_special_tokens=None,
        legacy_behaviour=False,
        **kwargs,
    ):
    
    # 获取对象状态的函数，返回对象的字典形式状态
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None  # 状态中的sp_model置为None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()  # 将sp_model的序列化模型保存到状态中
        return state

    # 设置对象状态的函数，接受状态字典d，并设置对象的状态
    def __setstate__(self, d):
        self.__dict__ = d

        # 为了向后兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用sp_model_kwargs创建spm.SentencePieceProcessor对象，并从序列化的proto中加载模型
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    # 返回词汇大小，包括sp_model的长度和fairseq_offset
    @property
    def vocab_size(self):
        return len(self.sp_model) + self.fairseq_offset

    # 返回源语言代码_src_lang
    @property
    def src_lang(self) -> str:
        return self._src_lang

    # 返回语言代码到id的映射，同时发出警告提示属性即将移除
    @property
    def lang_code_to_id(self):
        logger.warning_once(
            "the `lang_code_to_id` attribute is deprecated. The logic is natively handled in the `tokenizer.adder_tokens_decoder`"
            " this attribute will be removed in `transformers` v4.38"
        )
        return self._lang_code_to_id

    # 返回fairseq中tokens到ids的映射，同时发出警告提示属性即将移除
    @property
    def fairseq_tokens_to_ids(self):
        logger.warning_once(
            "the `fairseq_tokens_to_ids` attribute is deprecated. The logic is natively handled in the `tokenizer.adder_tokens_decoder`"
            " this attribute will be removed in `transformers` v4.38"
        )
        return self._fairseq_tokens_to_ids

    # 返回id到语言代码的映射，同时发出警告提示属性即将移除
    @property
    def id_to_lang_code(self):
        logger.warning_once(
            "the `id_to_lang_code` attribute is deprecated. The logic is natively handled in the `tokenizer.adder_tokens_decoder`"
            " this attribute will be removed in `transformers` v4.38"
        )
        return self._id_to_lang_code

    # 返回fairseq中ids到tokens的映射，同时发出警告提示属性即将移除
    @property
    def fairseq_ids_to_tokens(self):
        logger.warning_once(
            "the `_fairseq_ids_to_tokens` attribute is deprecated. The logic is natively handled in the `tokenizer.adder_tokens_decoder`"
            " this attribute will be removed in `transformers` v4.38"
        )
        return self._fairseq_ids_to_tokens

    # 设置源语言_src_lang，同时更新特殊标记
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)
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

        # Check if the token list already has special tokens
        if already_has_special_tokens:
            # If yes, delegate the computation to the superclass method
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Create lists of 1s for the prefix and suffix tokens
        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)

        # If there's no token_ids_1 (single sequence case), return with special tokens added
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        
        # For sequence pairs, return with special tokens added for both sequences
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

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

        # If there's no token_ids_1 (single sequence case), return input_ids with added special tokens
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        
        # If there are token_ids_1 (sequence pairs case), concatenate both sequences with special tokens
        # Note: This case is for API consistency and not expected to be a common use case.
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        """
        Create token type IDs tensor from token id lists.

        Args:
            token_ids_0 (`List[int]`):
                List of token IDs representing the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional list of token IDs representing the second sequence (for sequence pairs).

        Returns:
            `List[int]`: A list of token type IDs based on the input sequences.
        """

        # Create a list of zeros representing token type IDs for token_ids_0
        token_type_ids = [0] * len(token_ids_0)
        
        # If token_ids_1 is provided, extend the token_type_ids list with ones for token_ids_1
        if token_ids_1 is not None:
            token_type_ids += [1] * len(token_ids_1)
        
        # Return the token_type_ids list
        return token_type_ids
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

        # Initialize separator and classification tokens
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If token_ids_1 is None, return a list of zeros of the appropriate length
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]

        # If token_ids_1 is provided, calculate the length of the resulting sequence with additional separators
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        # Check if source language and target language are provided
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        
        # Set source language for the instance
        self.src_lang = src_lang
        
        # Prepare inputs by invoking the model with special tokens and additional keyword arguments
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        
        # Convert target language token to its corresponding ID
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        
        # Add the target language ID as a forced beginning-of-sequence token ID
        inputs["forced_bos_token_id"] = tgt_lang_id
        
        return inputs

    def get_vocab(self):
        # Create a dictionary mapping token strings to their corresponding IDs across the entire vocabulary
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        
        # Update the vocabulary dictionary with any additional tokens introduced
        vocab.update(self.added_tokens_encoder)
        
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        # Tokenize a given text using the SentencePiece model and return a list of token strings
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) into an ID using the vocabulary."""
        # Convert a token string into its corresponding ID using the SentencePiece model
        spm_id = self.sp_model.PieceToId(token)
        
        # Return the ID adjusted by fairseq offset for unknown tokens
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) into a token (str) using the vocabulary."""
        # Convert an index into its corresponding token string using the SentencePiece model and fairseq offset
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) into a single string."""
        # Concatenate tokens into a single string and replace special token underscore with a space
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 根据提供的前缀（如果有的话）和文件名字典中的键值，构建输出的词汇文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇文件路径不同于输出路径，并且当前词汇文件存在，则复制当前词汇文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇文件不存在，则将序列化后的特殊模型写入输出路径
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回输出路径的元组
        return (out_vocab_file,)

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "eng_Latn",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "fra_Latn",
        **kwargs,
    ) -> BatchEncoding:
        # 设置源语言和目标语言属性
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        # 调用父类方法，准备序列到序列的批次编码并返回结果
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        # 切换到输入模式，调用设置源语言特殊标记方法并返回结果
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        # 切换到目标模式，调用设置目标语言特殊标记方法并返回结果
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting.
        - In legacy mode: No prefix and suffix=[eos, src_lang_code].
        - In default mode: Prefix=[src_lang_code], suffix = [eos]
        """
        # 将当前语言代码转换为对应的 ID
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)
        # 根据传统模式与默认模式设置前缀和后缀特殊标记
        if self.legacy_behaviour:
            self.prefix_tokens = []
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.prefix_tokens = [self.cur_lang_code]
            self.suffix_tokens = [self.eos_token_id]

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target lang setting.
        - In legacy mode: No prefix and suffix=[eos, tgt_lang_code].
        - In default mode: Prefix=[tgt_lang_code], suffix = [eos]
        """
        # 将当前语言代码转换为对应的 ID
        self.cur_lang_code = self.convert_tokens_to_ids(lang)
        # 根据传统模式与默认模式设置前缀和后缀特殊标记
        if self.legacy_behaviour:
            self.prefix_tokens = []
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.prefix_tokens = [self.cur_lang_code]
            self.suffix_tokens = [self.eos_token_id]
```