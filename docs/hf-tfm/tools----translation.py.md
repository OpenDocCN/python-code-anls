# `.\tools\translation.py`

```
#!/usr/bin/env python
# coding=utf-8

# 导入必要的模块和类
from ..models.auto import AutoModelForSeq2SeqLM, AutoTokenizer
from .base import PipelineTool

# 定义语言代码的映射关系，将语言名映射到其对应的缩写代码
LANGUAGE_CODES = {
    "Acehnese Arabic": "ace_Arab",
    "Acehnese Latin": "ace_Latn",
    "Mesopotamian Arabic": "acm_Arab",
    "Ta'izzi-Adeni Arabic": "acq_Arab",
    "Tunisian Arabic": "aeb_Arab",
    "Afrikaans": "afr_Latn",
    "South Levantine Arabic": "ajp_Arab",
    "Akan": "aka_Latn",
    "Amharic": "amh_Ethi",
    "North Levantine Arabic": "apc_Arab",
    "Modern Standard Arabic": "arb_Arab",
    "Modern Standard Arabic Romanized": "arb_Latn",
    "Najdi Arabic": "ars_Arab",
    "Moroccan Arabic": "ary_Arab",
    "Egyptian Arabic": "arz_Arab",
    "Assamese": "asm_Beng",
    "Asturian": "ast_Latn",
    "Awadhi": "awa_Deva",
    "Central Aymara": "ayr_Latn",
    "South Azerbaijani": "azb_Arab",
    "North Azerbaijani": "azj_Latn",
    "Bashkir": "bak_Cyrl",
    "Bambara": "bam_Latn",
    "Balinese": "ban_Latn",
    "Belarusian": "bel_Cyrl",
    "Bemba": "bem_Latn",
    "Bengali": "ben_Beng",
    "Bhojpuri": "bho_Deva",
    "Banjar Arabic": "bjn_Arab",
    "Banjar Latin": "bjn_Latn",
    "Standard Tibetan": "bod_Tibt",
    "Bosnian": "bos_Latn",
    "Buginese": "bug_Latn",
    "Bulgarian": "bul_Cyrl",
    "Catalan": "cat_Latn",
    "Cebuano": "ceb_Latn",
    "Czech": "ces_Latn",
    "Chokwe": "cjk_Latn",
    "Central Kurdish": "ckb_Arab",
    "Crimean Tatar": "crh_Latn",
    "Welsh": "cym_Latn",
    "Danish": "dan_Latn",
    "German": "deu_Latn",
    "Southwestern Dinka": "dik_Latn",
    "Dyula": "dyu_Latn",
    "Dzongkha": "dzo_Tibt",
    "Greek": "ell_Grek",
    "English": "eng_Latn",
    "Esperanto": "epo_Latn",
    "Estonian": "est_Latn",
    "Basque": "eus_Latn",
    "Ewe": "ewe_Latn",
    "Faroese": "fao_Latn",
    "Fijian": "fij_Latn",
    "Finnish": "fin_Latn",
    "Fon": "fon_Latn",
    "French": "fra_Latn",
    "Friulian": "fur_Latn",
    "Nigerian Fulfulde": "fuv_Latn",
    "Scottish Gaelic": "gla_Latn",
    "Irish": "gle_Latn",
    "Galician": "glg_Latn",
    "Guarani": "grn_Latn",
    "Gujarati": "guj_Gujr",
    "Haitian Creole": "hat_Latn",
    "Hausa": "hau_Latn",
    "Hebrew": "heb_Hebr",
    "Hindi": "hin_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Croatian": "hrv_Latn",
    "Hungarian": "hun_Latn",
    "Armenian": "hye_Armn",
    "Igbo": "ibo_Latn",
    "Ilocano": "ilo_Latn",
    "Indonesian": "ind_Latn",
    # 印尼语使用拉丁字母表

    "Icelandic": "isl_Latn",
    # 冰岛语使用拉丁字母表

    "Italian": "ita_Latn",
    # 意大利语使用拉丁字母表

    "Javanese": "jav_Latn",
    # 爪哇语使用拉丁字母表

    "Japanese": "jpn_Jpan",
    # 日语使用日本汉字

    "Kabyle": "kab_Latn",
    # 卡比尔语使用拉丁字母表

    "Jingpho": "kac_Latn",
    # 景颇语使用拉丁字母表

    "Kamba": "kam_Latn",
    # 坎巴语使用拉丁字母表

    "Kannada": "kan_Knda",
    # 卡纳达语使用卡纳达字母表

    "Kashmiri Arabic": "kas_Arab",
    # 克什米尔语使用阿拉伯字母表

    "Kashmiri Devanagari": "kas_Deva",
    # 克什米尔语使用梵文字母表

    "Georgian": "kat_Geor",
    # 格鲁吉亚语使用格鲁吉亚字母表

    "Central Kanuri Arabic": "knc_Arab",
    # 中夸努里语使用阿拉伯字母表

    "Central Kanuri Latin": "knc_Latn",
    # 中夸努里语使用拉丁字母表

    "Kazakh": "kaz_Cyrl",
    # 哈萨克语使用西里尔字母表

    "Kabiyè": "kbp_Latn",
    # 卡比语使用拉丁字母表

    "Kabuverdianu": "kea_Latn",
    # 佛得角克里奥尔语使用拉丁字母表

    "Khmer": "khm_Khmr",
    # 高棉语使用高棉字母表

    "Kikuyu": "kik_Latn",
    # 基库尤语使用拉丁字母表

    "Kinyarwanda": "kin_Latn",
    # 卢旺达语使用拉丁字母表

    "Kyrgyz": "kir_Cyrl",
    # 吉尔吉斯语使用西里尔字母表

    "Kimbundu": "kmb_Latn",
    # 金本杜语使用拉丁字母表

    "Northern Kurdish": "kmr_Latn",
    # 北库尔德语使用拉丁字母表

    "Kikongo": "kon_Latn",
    # 基孔戈语使用拉丁字母表

    "Korean": "kor_Hang",
    # 韩语使用朝鲜字母

    "Lao": "lao_Laoo",
    # 老挝语使用老挝字母表

    "Ligurian": "lij_Latn",
    # 利古里亚语使用拉丁字母表

    "Limburgish": "lim_Latn",
    # 林堡语使用拉丁字母表

    "Lingala": "lin_Latn",
    # 林加拉语使用拉丁字母表

    "Lithuanian": "lit_Latn",
    # 立陶宛语使用拉丁字母表

    "Lombard": "lmo_Latn",
    # 伦巴第语使用拉丁字母表

    "Latgalian": "ltg_Latn",
    # 拉特加利亚语使用拉丁字母表

    "Luxembourgish": "ltz_Latn",
    # 卢森堡语使用拉丁字母表

    "Luba-Kasai": "lua_Latn",
    # 卢巴卡萨语使用拉丁字母表

    "Ganda": "lug_Latn",
    # 干达语使用拉丁字母表

    "Luo": "luo_Latn",
    # 卢奥语使用拉丁字母表

    "Mizo": "lus_Latn",
    # 米佐语使用拉丁字母表

    "Standard Latvian": "lvs_Latn",
    # 标准拉脱维亚语使用拉丁字母表

    "Magahi": "mag_Deva",
    # 马加希语使用梵文字母表

    "Maithili": "mai_Deva",
    # 麦蒂利语使用梵文字母表

    "Malayalam": "mal_Mlym",
    # 马拉雅拉姆语使用马拉雅拉姆字母表

    "Marathi": "mar_Deva",
    # 马拉地语使用梵文字母表

    "Minangkabau Arabic ": "min_Arab",
    # 苏门答腊语使用阿拉伯字母表

    "Minangkabau Latin": "min_Latn",
    # 苏门答腊语使用拉丁字母表

    "Macedonian": "mkd_Cyrl",
    # 马其顿语使用西里尔字母表

    "Plateau Malagasy": "plt_Latn",
    # 马达加斯加高原语使用拉丁字母表

    "Maltese": "mlt_Latn",
    # 马耳他语使用拉丁字母表

    "Meitei Bengali": "mni_Beng",
    # 曼尼普尔语使用孟加拉字母表

    "Halh Mongolian": "khk_Cyrl",
    # 哈尔哈蒙古语使用西里尔字母表

    "Mossi": "mos_Latn",
    # 莫西语使用拉丁字母表

    "Maori": "mri_Latn",
    # 毛利语使用拉丁字母表

    "Burmese": "mya_Mymr",
    # 缅甸语使用缅甸字母表

    "Dutch": "nld_Latn",
    # 荷兰语使用拉丁字母表

    "Norwegian Nynorsk": "nno_Latn",
    # 挪威尼诺斯克语使用拉丁字母表

    "Norwegian Bokmål": "nob_Latn",
    # 挪威博克马尔语使用拉丁字母表

    "Nepali": "npi_Deva",
    # 尼泊尔语使用梵文字母表

    "Northern Sotho": "nso_Latn",
    # 北索托语使用拉丁字母表

    "Nuer": "nus_Latn",
    # 努埃尔语使用拉丁字母表

    "Nyanja": "nya_Latn",
    # 尼昂加语使用拉丁字母表

    "Occitan": "oci_Latn",
    # 奥克语使用拉丁字母表

    "West Central Oromo": "gaz_Latn",
    # 西中奥罗莫语使用拉丁字母表

    "Odia": "ory_Orya",
    # 奥里雅语使用奥里雅字母表

    "Pangasinan": "pag_Latn",
    # 潘加西南语使用拉丁字母表

    "Eastern Panjabi": "pan_Guru",
    # 东旁遮普语使用古尔穆基字母表

    "Papiamento": "pap_Latn",
    # 帕皮亚门托语使用拉丁字母表

    "Western Persian": "pes_Arab",
    # 西部波斯语
    # 字符串到语言代码的映射，其中键为语言名称，值为语言代码
    {
        "Tamasheq Latin": "taq_Latn",
        "Tamasheq Tifinagh": "taq_Tfng",
        "Tok Pisin": "tpi_Latn",
        "Tswana": "tsn_Latn",
        "Tsonga": "tso_Latn",
        "Turkmen": "tuk_Latn",
        "Tumbuka": "tum_Latn",
        "Turkish": "tur_Latn",
        "Twi": "twi_Latn",
        "Central Atlas Tamazight": "tzm_Tfng",
        "Uyghur": "uig_Arab",
        "Ukrainian": "ukr_Cyrl",
        "Umbundu": "umb_Latn",
        "Urdu": "urd_Arab",
        "Northern Uzbek": "uzn_Latn",
        "Venetian": "vec_Latn",
        "Vietnamese": "vie_Latn",
        "Waray": "war_Latn",
        "Wolof": "wol_Latn",
        "Xhosa": "xho_Latn",
        "Eastern Yiddish": "ydd_Hebr",
        "Yoruba": "yor_Latn",
        "Yue Chinese": "yue_Hant",
        "Chinese Simplified": "zho_Hans",
        "Chinese Traditional": "zho_Hant",
        "Standard Malay": "zsm_Latn",
        "Zulu": "zul_Latn",
    }
    }



class TranslationTool(PipelineTool):
    """
    Example:

    ```py
    from transformers.tools import TranslationTool

    translator = TranslationTool()
    translator("This is a super nice API!", src_lang="English", tgt_lang="French")
    ```
    """

    default_checkpoint = "facebook/nllb-200-distilled-600M"
    description = (
        "This is a tool that translates text from a language to another. It takes three inputs: `text`, which should "
        "be the text to translate, `src_lang`, which should be the language of the text to translate and `tgt_lang`, "
        "which should be the language for the desired ouput language. Both `src_lang` and `tgt_lang` are written in "
        "plain English, such as 'Romanian', or 'Albanian'. It returns the text translated in `tgt_lang`."
    )
    name = "translator"
    pre_processor_class = AutoTokenizer
    model_class = AutoModelForSeq2SeqLM
    lang_to_code = LANGUAGE_CODES

    inputs = ["text", "text", "text"]
    outputs = ["text"]

    def encode(self, text, src_lang, tgt_lang):
        # 检查源语言是否在支持的语言列表中
        if src_lang not in self.lang_to_code:
            raise ValueError(f"{src_lang} is not a supported language.")
        # 检查目标语言是否在支持的语言列表中
        if tgt_lang not in self.lang_to_code:
            raise ValueError(f"{tgt_lang} is not a supported language.")
        # 将源语言和目标语言转换为对应的语言代码
        src_lang = self.lang_to_code[src_lang]
        tgt_lang = self.lang_to_code[tgt_lang]
        # 使用预处理器构建翻译输入
        return self.pre_processor._build_translation_inputs(
            text, return_tensors="pt", src_lang=src_lang, tgt_lang=tgt_lang
        )

    def forward(self, inputs):
        # 使用模型生成翻译结果
        return self.model.generate(**inputs)

    def decode(self, outputs):
        # 使用后处理器解码输出结果，跳过特殊标记
        return self.post_processor.decode(outputs[0].tolist(), skip_special_tokens=True)
```