# `.\models\nllb\tokenization_nllb_fast.py`

```py
# 导入标准库和第三方库
import os  # 导入操作系统相关的功能模块
from shutil import copyfile  # 从 shutil 模块中导入 copyfile 函数
from typing import List, Optional, Tuple  # 导入类型提示相关的模块

# 导入自定义模块和函数
from tokenizers import processors  # 从 tokenizers 库导入 processors 对象
from ...tokenization_utils import AddedToken, BatchEncoding  # 导入相对路径中的 tokenization_utils 模块中的类和函数
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入相对路径中的 tokenization_utils_fast 模块中的 PreTrainedTokenizerFast 类
from ...utils import is_sentencepiece_available, logging  # 从相对路径中的 utils 模块导入 is_sentencepiece_available 和 logging 函数

# 根据 sentencepiece 库的可用性选择性地导入 NllbTokenizer 类
if is_sentencepiece_available():
    from .tokenization_nllb import NllbTokenizer  # 从当前包中的 tokenization_nllb 模块导入 NllbTokenizer 类
else:
    NllbTokenizer = None  # 如果 sentencepiece 不可用，则将 NllbTokenizer 设为 None

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# 定义 VOCAB_FILES_NAMES 字典，指定词汇表和分词器文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

# 定义 PRETRAINED_VOCAB_FILES_MAP 字典，指定预训练模型的词汇表和分词器文件的下载链接
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

# 定义 PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES 字典，指定预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/nllb-large-en-ro": 1024,
    "facebook/nllb-200-distilled-600M": 1024,
}
# 支持Fairseq使用的语言代码列表，每个元素表示一个语言代码，格式为语言代码_脚本
FAIRSEQ_LANGUAGE_CODES = ['ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab', 'aeb_Arab', 'afr_Latn', 'ajp_Arab', 'aka_Latn', 'amh_Ethi', 'apc_Arab', 'arb_Arab', 'ars_Arab', 'ary_Arab', 'arz_Arab', 'asm_Beng', 'ast_Latn', 'awa_Deva', 'ayr_Latn', 'azb_Arab', 'azj_Latn', 'bak_Cyrl', 'bam_Latn', 'ban_Latn', 'bel_Cyrl', 'bem_Latn', 'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn', 'bod_Tibt', 'bos_Latn', 'bug_Latn', 'bul_Cyrl', 'cat_Latn', 'ceb_Latn', 'ces_Latn', 'cjk_Latn', 'ckb_Arab', 'crh_Latn', 'cym_Latn', 'dan_Latn', 'deu_Latn', 'dik_Latn', 'dyu_Latn', 'dzo_Tibt', 'ell_Grek', 'eng_Latn', 'epo_Latn', 'est_Latn', 'eus_Latn', 'ewe_Latn', 'fao_Latn', 'pes_Arab', 'fij_Latn', 'fin_Latn', 'fon_Latn', 'fra_Latn', 'fur_Latn', 'fuv_Latn', 'gla_Latn', 'gle_Latn', 'glg_Latn', 'grn_Latn', 'guj_Gujr', 'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hne_Deva', 'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ibo_Latn', 'ilo_Latn', 'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan', 'kab_Latn', 'kac_Latn', 'kam_Latn', 'kan_Knda', 'kas_Arab', 'kas_Deva', 'kat_Geor', 'knc_Arab', 'knc_Latn', 'kaz_Cyrl', 'kbp_Latn', 'kea_Latn', 'khm_Khmr', 'kik_Latn', 'kin_Latn', 'kir_Cyrl', 'kmb_Latn', 'kon_Latn', 'kor_Hang', 'kmr_Latn', 'lao_Laoo', 'lvs_Latn', 'lij_Latn', 'lim_Latn', 'lin_Latn', 'lit_Latn', 'lmo_Latn', 'ltg_Latn', 'ltz_Latn', 'lua_Latn', 'lug_Latn', 'luo_Latn', 'lus_Latn', 'mag_Deva', 'mai_Deva', 'mal_Mlym', 'mar_Deva', 'min_Latn', 'mkd_Cyrl', 'plt_Latn', 'mlt_Latn', 'mni_Beng', 'khk_Cyrl', 'mos_Latn', 'mri_Latn', 'zsm_Latn', 'mya_Mymr', 'nld_Latn', 'nno_Latn', 'nob_Latn', 'npi_Deva', 'nso_Latn', 'nus_Latn', 'nya_Latn', 'oci_Latn', 'gaz_Latn', 'ory_Orya', 'pag_Latn', 'pan_Guru', 'pap_Latn', 'pol_Latn', 'por_Latn', 'prs_Arab', 'pbt_Arab', 'quy_Latn', 'ron_Latn', 'run_Latn', 'rus_Cyrl', 'sag_Latn', 'san_Deva', 'sat_Beng', 'scn_Latn', 'shn_Mymr', 'sin_Sinh', 'slk_Latn', 'slv_Latn', 'smo_Latn', 'sna_Latn', 'snd_Arab', 'som_Latn', 'sot_Latn', 'spa_Latn', 'als_Latn', 'srd_Latn', 'srp_Cyrl', 'ssw_Latn', 'sun_Latn', 'swe_Latn', 'swh_Latn', 'szl_Latn', 'tam_Taml', 'tat_Cyrl', 'tel_Telu', 'tgk_Cyrl', 'tgl_Latn', 'tha_Thai', 'tir_Ethi', 'taq_Latn', 'taq_Tfng', 'tpi_Latn', 'tsn_Latn', 'tso_Latn', 'tuk_Latn', 'tum_Latn', 'tur_Latn', 'twi_Latn', 'tzm_Tfng', 'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab', 'uzn_Latn', 'vec_Latn', 'vie_Latn', 'war_Latn', 'wol_Latn', 'xho_Latn', 'ydd_Hebr', 'yor_Latn', 'yue_Hant', 'zho_Hans', 'zho_Hant', 'zul_Latn']  # fmt: skip

# 定义了一个新的类 NllbTokenizerFast，它继承自 PreTrainedTokenizerFast 类
class NllbTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”NLLB分词器，使用HuggingFace的 *tokenizers* 库作为后端。基于[BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models)。
    
    这个分词器继承自 `PreTrainedTokenizerFast`，该类包含大部分主要方法。用户应参考这个超类以获取关于这些方法更多信息。
    """
    # 在源语言文档中，标记化方法为 `<tokens> <eos> <language code>`；在目标语言文档中，标记化方法为 `<language code> <tokens> <eos>`。
    # 
    # 示例:
    # 
    # ```
    # >>> from transformers import NllbTokenizerFast
    # 
    # >>> tokenizer = NllbTokenizerFast.from_pretrained(
    # ...     "facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn"
    # ... )
    # >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
    # >>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
    # >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
    # ```
    # 获取预定义的词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 获取预训练模型的最大输入大小列表
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 获取预训练词汇文件映射字典
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 指定一个较慢的分词器类
    slow_tokenizer_class = NllbTokenizer

    # 前缀令牌的初始整数列表
    prefix_tokens: List[int] = []
    # 后缀令牌的初始整数列表
    suffix_tokens: List[int] = []
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
        legacy_behaviour=False,
        **kwargs,
    ):
        # 如果未提供额外的特殊标记，使用默认的FAIRSEQ_LANGUAGE_CODES
        if additional_special_tokens is None:
            additional_special_tokens = FAIRSEQ_LANGUAGE_CODES

        self.vocab_file = vocab_file
        # 如果mask_token是字符串，创建一个AddedToken对象，处理其属性
        mask_token = (
            AddedToken(mask_token, normalized=True, lstrip=True, special=True)
            if isinstance(mask_token, str)
            else mask_token
        )
        self.legacy_behaviour = legacy_behaviour
        # 调用父类的初始化方法，设置实例变量
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
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            legacy_behaviour=legacy_behaviour,
            **kwargs,
        )

        # 创建语言代码到其对应ID的映射字典
        self._lang_code_to_id = {
            lang_code: self.convert_tokens_to_ids(str(lang_code)) for lang_code in additional_special_tokens
        }

        # 设置源语言，默认为"eng_Latn"，如果未提供则使用默认值
        self._src_lang = src_lang if src_lang is not None else "eng_Latn"
        # 将当前语言代码转换为其对应的ID
        self.cur_lang_code = self.convert_tokens_to_ids(self._src_lang)
        self.tgt_lang = tgt_lang
        # 设置源语言特殊标记
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def lang_code_to_id(self):
        # 提示警告，该属性即将被废弃
        logger.warning_once(
            "the `lang_code_to_id` attribute is deprecated. The logic is natively handled in the `tokenizer.adder_tokens_decoder`"
            " this attribute will be removed in `transformers` v4.38"
        )
        return self._lang_code_to_id

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查是否可以保存缓慢的分词器，检查词汇文件是否存在
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    @property
    def src_lang(self) -> str:
        # 返回源语言
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        # 设置新的源语言，并更新特殊标记
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def set_lang(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
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
        if token_ids_1 is None:
            # Return concatenated list of tokens with prefix and suffix tokens
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # Handle the case of sequence pairs by concatenating both sequences with prefix and suffix tokens
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

        # Define special tokens for separator and classification respectively
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            # Return list of zeros with length corresponding to the total tokens including special tokens
            return len(cls + token_ids_0 + sep) * [0]
        # Handle sequence pairs by computing the length of tokens including additional separators
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        # Ensure both source and target languages are provided
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        
        # Set the source language for further processing
        self.src_lang = src_lang
        
        # Generate inputs for the model with special tokens added
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        
        # Convert target language to token ID and assign as forced beginning-of-sequence token
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
    ):
        """
        Prepare a batch of inputs for sequence-to-sequence tasks, including source and target texts and languages.

        Args:
            src_texts (`List[str]`):
                List of source texts.
            src_lang (`str`):
                Source language identifier.
            tgt_texts (`List[str]`, *optional*):
                List of target texts.
            tgt_lang (`str`):
                Target language identifier.
            **kwargs:
                Additional keyword arguments for further customization.

        Returns:
            Dictionary containing prepared inputs for the model.
        """
        # Implementation of this function's details would go here, but the provided snippet does not include its full body.
        pass
    def set_src_lang_special_tokens(self, src_lang) -> None:
        """设置特殊标记以适应源语言设置。
        - 在传统模式下：无前缀，后缀=[eos, src_lang_code]。
        - 在默认模式下：前缀=[src_lang_code]，后缀=[eos]。
        """
        # 将当前语言代码转换为对应的 ID
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)

        # 根据 legacy_behaviour 设置特殊标记列表
        if self.legacy_behaviour:
            self.prefix_tokens = []
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.prefix_tokens = [self.cur_lang_code]
            self.suffix_tokens = [self.eos_token_id]

        # 将 ID 转换为对应的 token 字符串
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        # 更新 tokenizer 的后处理器以包含特殊标记
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """设置特殊标记以适应目标语言设置。
        - 在传统模式下：无前缀，后缀=[eos, tgt_lang_code]。
        - 在默认模式下：前缀=[tgt_lang_code]，后缀=[eos]。
        """
        # 将当前语言代码转换为对应的 ID
        self.cur_lang_code = self.convert_tokens_to_ids(lang)

        # 根据 legacy_behaviour 设置特殊标记列表
        if self.legacy_behaviour:
            self.prefix_tokens = []
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.prefix_tokens = [self.cur_lang_code]
            self.suffix_tokens = [self.eos_token_id]

        # 将 ID 转换为对应的 token 字符串
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        # 更新 tokenizer 的后处理器以包含特殊标记
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )
    # 保存词汇表到指定目录中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查是否可以保存慢速分词器的词汇表，如果不能则抛出值错误异常
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 检查保存目录是否存在，如果不存在则记录错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory.")
            return

        # 构建输出词汇表文件的完整路径，如果有前缀则加在文件名前面
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件的绝对路径不等于输出文件的绝对路径，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回输出文件路径的元组
        return (out_vocab_file,)
```