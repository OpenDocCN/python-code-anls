# `.\transformers\models\nllb\tokenization_nllb_fast.py`

```py
# 导入所需的模块和函数
import os
# 从 shutil 模块导入 copyfile 函数，用于复制文件
from shutil import copyfile
# 从 typing 模块导入 List、Optional 和 Tuple 类型
from typing import List, Optional, Tuple
# 从 tokenizers 模块导入 processors 模块
from tokenizers import processors
# 从 tokenization_utils 模块导入 AddedToken 和 BatchEncoding 类
from ...tokenization_utils import AddedToken, BatchEncoding
# 从 tokenization_utils_fast 模块导入 PreTrainedTokenizerFast 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 从 ...utils 模块导入 is_sentencepiece_available 和 logging 函数
from ...utils import is_sentencepiece_available, logging

# 判断是否安装了 SentencePiece 库
if is_sentencepiece_available():
    # 若安装了 SentencePiece 库，则从 tokenization_nllb 模块导入 NllbTokenizer 类
    from .tokenization_nllb import NllbTokenizer
else:
    # 若未安装 SentencePiece 库，则将 NllbTokenizer 设为 None
    NllbTokenizer = None

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/nllb-200-distilled-600M": (
            "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/sentencepiece.bpe.model"
        ),
    },
    "tokenizer_file": {
        "facebook/nllb-200-distilled-600M": (
            "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/tokenizer.json"
        ),
    },
}

# 定义预训练模型的位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/nllb-large-en-ro": 1024,
    "facebook/nllb-200-distilled-600M": 1024,
}
# 定义一个列表，包含FAIRSEQ模型支持的语言代码
FAIRSEQ_LANGUAGE_CODES = ['ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn', 'ajp_Arab', 'aka_Latn', 'amh_Ethi', 'apc_Arab', 'arb_Arab', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'asm_Beng', 'ast_Latn', 'awa_Deva', 'ayr_Latn', 'azb_Arab', 'azj_Latn', 'bak_Cyrl', 'bam_Latn', 'ban_Latn', 'bel_Cyrl', 'bem_Latn', 'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn', 'bod_Tibt', 'bos_Latn', 'bug_Latn', 'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn', 'cjk_Latn', 'ckb_Arab', 'crh_Latn', 'cym_Latn', 'dan_Latn', 'deu_Latn', 'dik_Latn', 'dyu_Latn', 'dzo_Tibt', 'ell_Grek', 'eng_Latn', 'epo_Latn', 'est_Latn', 'eus_Latn', 'ewe_Latn', 'fao_Latn', 'pes_Arab', 'fij_Latn', 'fin_Latn', 'fon_Latn', 'fra_Latn', 'fur_Latn', 'fuv_Latn', 'gla_Latn', 'gle_Latn', 'glg_Latn', 'grn_Latn', 'guj_Gujr', 'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hne_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ibo_Latn', 'ilo_Latn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan', 'kab_Latn', 'kac_Latn', 'kam_Latn', 'kan_Knda', 'kas_Arab', 'kas_Deva', 'kat_Geor', 'knc_Arab', 'knc_Latn', 'kaz_Cyrl', 'kbp_Latn', 'kea_Latn', 'khm_Khmr', 'kik_Latn', 'kin_Latn', 'kir_Cyrl', 'kmb_Latn', 'kon_Latn', 'kor_Hang', 'kmr_Latn', 'lao_Laoo', 'lvs_Latn', 'lij_Latn', 'lim_Latn', 'lin_Latn', 'lit_Latn', 'lmo_Latn', 'ltg_Latn', 'ltz_Latn', 'lua_Latn', 'lug_Latn', 'luo_Latn', 'lus_Latn', 'mag_Deva', 'mai_Deva', 'mal_Mlym', 'mar_Deva', 'min_Latn', 'mkd_Cyrl', 'plt_Latn', 'mlt_Latn', 'mni_Beng', 'khk_Cyrl', 'mos_Latn', 'mri_Latn', 'zsm_Latn', 'mya_Mymr', 'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva', 'nso_Latn', 'nus_Latn', 'nya_Latn', 'oci_Latn', 'gaz_Latn', 'ory_Orya', 'pag_Latn', 'pan_Guru', 'pap_Latn', 'pol_Latn', 'por_Latn', 'prs_Arab', 'pbt_Arab', 'quy_Latn', 'ron_Latn', 'run_Latn', 'rus_Cyrl', 'sag_Latn', 'san_Deva', 'sat_Beng', 'scn_Latn', 'shn_Mymr', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'smo_Latn', 'sna_Latn', 'snd_Arab', 'som_Latn', 'sot_Latn', 'spa_Latn', 'als_Latn', 'srd_Latn', 'srp_Cyrl', 'ssw_Latn', 'sun_Latn', 'swe_Latn', 'swh_Latn', 'szl_Latn', 'tam_Taml', 'tat_Cyrl', 'tel_Telu', 'tgk_Cyrl', 'tgl_Latn', 'tha_Thai', 'tir_Ethi', 'taq_Latn', 'taq_Tfng', 'tpi_Latn', 'tsn_Latn', 'tso_Latn', 'tuk_Latn', 'tum_Latn', 'tur_Latn', 'twi_Latn', 'tzm_Tfng', 'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab', 'uzn_Latn', 'vec_Latn', 'vie_Latn', 'war_Latn', 'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'yue_Hant', 'zho_Hans', 'zho_Hant', 'zul_Latn']  # fmt: skip



# 定义一个类NllbTokenizerFast，继承自PreTrainedTokenizerFast类
class NllbTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”NLLB分词器（由HuggingFace的“tokenizers”库支持）。基于[BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models)。

    此分词器继承自PreTrainedTokenizerFast类，该类包含大部分主要方法。用户可以在这个超类中找到更多关于这些方法的信息。
    # 定义了源语言文档和目标语言文档的分词方法
    # 源语言文档的分词方法是 `<tokens> <eos> <language code>`
    # 目标语言文档的分词方法是 `<language code> <tokens> <eos>`
    
    # 示例代码
    # 导入 NllbTokenizerFast 类
    # 使用预训练模型 "facebook/nllb-200-distilled-600M"，设置源语言为"eng_Latn"，目标语言为"fra_Latn"
    # 定义一个例句 example_english_phrase，设置期望的法语翻译 expected_translation_french
    # 使用 tokenizer 对 example_english_phrase 进行分词，并指定目标文本及返回的张量类型为"pt"
    Args:
        vocab_file (`str`):
            # 词汇表文件的路径
            Path to the vocabulary file.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            # 在预训练期间使用的序列开头标记。可用于序列分类器标记。
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            # 在构建序列时使用特殊标记，这不是用于序列开头的标记。用于序列开头的标记是 `cls_token`。
            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            # 序列结束的标记
            The end of sequence token.

            <Tip>

            # 在构建序列时使用特殊标记，这不是用于序列结束的标记。用于序列结束的标记是 `sep_token`。
            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            # 分隔符标记，用于从多个序列构建序列，例如，用于序列分类的两个序列，或用于文本和问题的序列。还用作使用特殊标记构建的序列的最后一个标记。
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            # 用于序列分类（整个序列而不是每个标记的分类）时使用的分类器标记。当使用特殊标记构建时，它是序列的第一个标记。
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            # 未知标记。词汇表中没有的标记无法转换为ID，并设置为此标记。
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            # 用于填充的标记，例如，当批处理不同长度的序列时。
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            # 用于掩码值的标记。这是用于使用掩码语言模型进行模型训练的标记。这是模型将尝试预测的标记。
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        tokenizer_file (`str`, *optional*):
            # 要使用的分词器文件的路径，而不是词汇表文件。
            The path to a tokenizer file to use instead of the vocab file.
        src_lang (`str`, *optional*):
            # 用作翻译源语言的语言。
            The language to use as source language for translation.
        tgt_lang (`str`, *optional*):
            # 用作翻译目标语言的语言。
            The language to use as target language for translation.
    """

    # 词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练位置嵌入大小列表
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 预训练词汇文件映射字典
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 慢分词器类
    slow_tokenizer_class = NllbTokenizer

    # 前缀标记列表初始化
    prefix_tokens: List[int] = []
    # 后缀标记列表初始化
    suffix_tokens: List[int] = []
```  
    # 初始化方法，用于初始化Tokenizer对象
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        bos_token="<s>",  # 开始标记
        eos_token="</s>",  # 结束标记
        sep_token="</s>",  # 分隔标记
        cls_token="<s>",   # 类别标记
        unk_token="<unk>",  # 未知标记
        pad_token="<pad>",  # 填充标记
        mask_token="<mask>",  # 掩码标记
        src_lang=None,  # 源语言
        tgt_lang=None,  # 目标语言
        additional_special_tokens=None,  # 额外的特殊标记
        legacy_behaviour=False,  # 是否使用旧版行为
        **kwargs,  # 其他参数
    ):
        # 如果 mask_token 是字符串，则将其转换为 AddedToken 对象
        mask_token = (
            AddedToken(mask_token, normalized=True, lstrip=True, special=True)
            if isinstance(mask_token, str)
            else mask_token
        )
        # 保存是否使用旧版行为的标志
        self.legacy_behaviour = legacy_behaviour

        # 复制 FAIRSEQ_LANGUAGE_CODES 到 _additional_special_tokens
        _additional_special_tokens = FAIRSEQ_LANGUAGE_CODES.copy()

        # 如果有额外的特殊标记，则添加到 _additional_special_tokens 中
        if additional_special_tokens is not None:
            # 仅在 _additional_special_tokens 中不存在时才添加
            _additional_special_tokens.extend(
                [t for t in additional_special_tokens if t not in _additional_special_tokens]
            )

        # 调用父类的初始化方法，传入各种参数
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
            legacy_behaviour=legacy_behaviour,
            **kwargs,
        )

        # 保存词汇文件路径
        self.vocab_file = vocab_file

        # 创建从语言代码到词汇 ID 的映射
        self.lang_code_to_id = {
            lang_code: self.convert_tokens_to_ids(lang_code) for lang_code in FAIRSEQ_LANGUAGE_CODES
        }

        # 设置源语言，默认为 "eng_Latn"
        self._src_lang = src_lang if src_lang is not None else "eng_Latn"
        # 将源语言转换为其对应的词汇 ID
        self.cur_lang_code = self.convert_tokens_to_ids(self._src_lang)
        # 设置源语言的特殊标记
        self.set_src_lang_special_tokens(self._src_lang)

    # 是否可以保存慢速分词器
    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    # 获取源语言属性
    @property
    def src_lang(self) -> str:
        return self._src_lang

    # 设置源语言属性
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        # 根据新的源语言设置特殊标记
        self.set_src_lang_special_tokens(self._src_lang)

    # 构建带有特殊标记的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. The special tokens depend on calling set_lang.

        An NLLB sequence has the following format, where `X` represents the sequence:

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
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # If only one list of IDs is provided, concatenate it with special tokens and return
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # If two lists of IDs are provided, concatenate both with special tokens and return
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

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

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If only one list of IDs is provided, create a mask with zeros based on the concatenated sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # If two lists of IDs are provided, create a mask with zeros based on the concatenated sequences
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        # Check if both source language and target language are provided
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        # Set the source language for the model
        self.src_lang = src_lang
        # Prepare inputs by adding special tokens and setting forced BOS token ID
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "eng_Latn",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "fra_Latn",
        **kwargs,
    ) -> BatchEncoding:
        # 设置源语言和目标语言
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        # 调用父类方法准备序列到序列的批处理
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        # 切换到输入模式，设置特殊标记以匹配源语言
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        # 切换到目标模式，设置特殊标记以匹配目标语言
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting.
        - In legacy mode: No prefix and suffix=[eos, src_lang_code].
        - In default mode: Prefix=[src_lang_code], suffix = [eos]
        """
        # 将当前语言代码转换为对应的 ID
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)

        # 根据遗留模式和默认模式设置前缀和后缀特殊标记
        if self.legacy_behaviour:
            self.prefix_tokens = []
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.prefix_tokens = [self.cur_lang_code]
            self.suffix_tokens = [self.eos_token_id]

        # 将前缀和后缀标记转换为字符串
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        # 更新分词器的后处理器，根据模板设置特殊标记
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target lang setting.
        - In legacy mode: No prefix and suffix=[eos, tgt_lang_code].
        - In default mode: Prefix=[tgt_lang_code], suffix = [eos]
        """
        # 将当前语言代码转换为对应的 ID
        self.cur_lang_code = self.convert_tokens_to_ids(lang)
        # 根据遗留模式和默认模式设置前缀和后缀特殊标记
        if self.legacy_behaviour:
            self.prefix_tokens = []
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.prefix_tokens = [self.cur_lang_code]
            self.suffix_tokens = [self.eos_token_id]

        # 将前缀和后缀标记转换为字符串
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        # 更新分词器的后处理器，根据模板设置特殊标记
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查当前的快速分词器是否能够保存慢速分词器的词汇表
        if not self.can_save_slow_tokenizer:
            # 如果不能保存，则抛出异常
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 检查保存目录是否是一个目录
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory.")
            # 如果不是目录，则返回空值
            return
        # 创建保存词汇表的文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前的词汇表文件路径和目标文件路径不一致，则将当前文件复制到目标文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回保存的词汇表文件路径
        return (out_vocab_file,)
```