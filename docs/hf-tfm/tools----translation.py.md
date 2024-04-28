# `.\transformers\tools\translation.py`

```
# 导入所需模块
from ..models.auto import AutoModelForSeq2SeqLM, AutoTokenizer
from .base import PipelineTool

# 定义语言代码映射表
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
    "Indonesian": "ind_Latn",  # 印尼语使用拉丁字母
    "Icelandic": "isl_Latn",  # 冰岛语使用拉丁字母
    "Italian": "ita_Latn",  # 意大利语使用拉丁字母
    "Javanese": "jav_Latn",  # 爪哇语使用拉丁字母
    "Japanese": "jpn_Jpan",  # 日语使用日文
    "Kabyle": "kab_Latn",  # 卡比尔语使用拉丁字母
    "Jingpho": "kac_Latn",  # 景颇语使用拉丁字母
    "Kamba": "kam_Latn",  # 坎巴语使用拉丁字母
    "Kannada": "kan_Knda",  # 卡纳达语使用卡纳达字母
    "Kashmiri Arabic": "kas_Arab",  # 克什米尔语使用阿拉伯字母
    "Kashmiri Devanagari": "kas_Deva",  # 克什米尔语使用天城字母
    "Georgian": "kat_Geor",  # 格鲁吉亚语使用格鲁吉亚字母
    "Central Kanuri Arabic": "knc_Arab",  # 中央卡努里语使用阿拉伯字母
    "Central Kanuri Latin": "knc_Latn",  # 中央卡努里语使用拉丁字母
    "Kazakh": "kaz_Cyrl",  # 哈萨克语使用西里尔字母
    "Kabiyè": "kbp_Latn",  # 卡比耶语使用拉丁字母
    "Kabuverdianu": "kea_Latn",  # 佛得角克里奥尔语使用拉丁字母
    "Khmer": "khm_Khmr",  # 高棉语使用高棉字母
    "Kikuyu": "kik_Latn",  # 基库尤语使用拉丁字母
    "Kinyarwanda": "kin_Latn",  # 卢旺达语使用拉丁字母
    "Kyrgyz": "kir_Cyrl",  # 吉尔吉斯语使用西里尔字母
    "Kimbundu": "kmb_Latn",  # 金本杜语使用拉丁字母
    "Northern Kurdish": "kmr_Latn",  # 北库尔德语使用拉丁字母
    "Kikongo": "kon_Latn",  # 金贡戈语使用拉丁字母
    "Korean": "kor_Hang",  # 韩语使用韩文
    "Lao": "lao_Laoo",  # 老挝语使用老挝字母
    "Ligurian": "lij_Latn",  # 利古里亚语使用拉丁字母
    "Limburgish": "lim_Latn",  # 林堡语使用拉丁字母
    "Lingala": "lin_Latn",  # 林加拉语使用拉丁字母
    "Lithuanian": "lit_Latn",  # 立陶宛语使用拉丁字母
    "Lombard": "lmo_Latn",  # 伦巴第语使用拉丁字母
    "Latgalian": "ltg_Latn",  # 拉特加利亚语使用拉丁字母
    "Luxembourgish": "ltz_Latn",  # 卢森堡语使用拉丁字母
    "Luba-Kasai": "lua_Latn",  # 卢巴卡萨伊语使用拉丁字母
    "Ganda": "lug_Latn",  # 干达语使用拉丁字母
    "Luo": "luo_Latn",  # 罗语使用拉丁字母
    "Mizo": "lus_Latn",  # 米佐语使用拉丁字母
    "Standard Latvian": "lvs_Latn",  # 拉脱维亚语使用拉丁字母
    "Magahi": "mag_Deva",  # 马加希语使用天城字母
    "Maithili": "mai_Deva",  # 麦蒂利语使用天城字母
    "Malayalam": "mal_Mlym",  # 马拉雅拉姆语使用马拉雅拉姆字母
    "Marathi": "mar_Deva",  # 马拉地语使用天城字母
    "Minangkabau Arabic ": "min_Arab",  # 明古巴语使用阿拉伯字母
    "Minangkabau Latin": "min_Latn",  # 明古巴语使用拉丁字母
    "Macedonian": "mkd_Cyrl",  # 马其顿语使用西里尔字母
    "Plateau Malagasy": "plt_Latn",  # 马达加斯加高原语使用拉丁字母
    "Maltese": "mlt_Latn",  # 马耳他语使用拉丁字母
    "Meitei Bengali": "mni_Beng",  # 曼尼普尔语使用孟加拉字母
    "Halh Mongolian": "khk_Cyrl",  # 哈尔哈蒙古语使用西里尔字母
    "Mossi": "mos_Latn",  # 莫西语使用拉丁字母
    "Maori": "mri_Latn",  # 毛利语使用拉丁字母
    "Burmese": "mya_Mymr",  # 缅甸语使用缅甸字母
    "Dutch": "nld_Latn",  # 荷兰语使用拉丁字母
    "Norwegian Nynorsk": "nno_Latn",  # 挪威尼诺斯克语使用拉丁字母
    "Norwegian Bokmål": "nob_Latn",  # 挪威博克马尔语使用拉丁字母
    "Nepali": "npi_Deva",  # 尼泊尔语使用天城字母
    "Northern Sotho": "nso_Latn",  # 北索托语使用拉丁字母
    "Nuer": "nus_Latn",  # 努埃尔语使用拉丁字母
    "Nyanja": "nya_Latn",  # 尼扬扎语使用拉丁字母
    "Occitan": "oci_Latn",  # 奥克语使用拉丁字母
    "West Central Oromo": "gaz_Latn",  # 西中奥罗莫语使用拉丁字母
    "Odia": "ory_Orya",  # 奥里亚语使用奥里亚字母
    "Pangasinan": "pag_Latn",  # 庞加西南语使用拉丁字母
    "Eastern Panjabi": "pan_Guru",  # 东旁遮普语使用古尔穆基字母
    "Papiamento": "pap_Latn",  # 帕皮亚门托语使用拉丁字母
    "Western Persian": "pes_Arab",  # 波斯语使用阿拉伯字母
    "Polish": "pol_Latn",  # 波兰语使用拉丁字母
    "Portuguese": "por_Latn",  # 葡萄牙语使用拉丁字母
    "Dari": "prs_Arab",  # 达里语使用阿拉伯字母
    "Southern Pashto": "pbt_Arab",  # 南普什图语使用阿拉伯字母
    "Ayacucho Quechua": "quy_Latn",  # 阿亚库乔克丘亚语使用拉丁字母
    "Romanian": "ron_Latn",  # 罗马尼亚语使用拉丁字母
    "Rundi": "run_Latn",  # 基隆迪语使用拉丁字母
    "Russian": "rus_Cyrl",  # 俄语使用西里尔字母
    "Sango": "sag_Latn",  # 桑戈语使用拉丁字母
    "Sanskrit": "san_Deva",  # 梵语使用天城字母
    "Santali": "sat_Olck",  # 桑塔利语使用奥尔查字母
    "Sicilian": "scn_Latn",  # 西西里语使用拉丁字母
    "Shan": "shn_Mymr",  # 掸语使用缅甸字母
    "Sinhala": "sin_Sinh",  # 僧伽罗语使用僧伽罗字母
    "Slovak": "slk_Latn",  # 斯洛伐克语使用拉丁字母
    "Slovenian": "slv_Latn",  # 斯洛文尼亚语使用拉丁字母
    "Samoan": "smo_Latn",  # 萨摩亚语使用拉丁字母
    "Shona": "sna_Latn",  # 绍纳语使用拉丁字母
    "Sindhi": "snd_Arab",  # 信德语使用阿拉伯字母
    "Somali": "som_Latn",  # 索马里语使用拉丁字母
    "Southern Sotho": "sot_Latn",  # 南索托语使用拉丁字母
    "Spanish": "spa_Latn",  # 西班牙语使用拉丁字母
    "Tosk Albanian": "als_Latn",  # 托斯克阿尔巴尼亚语使用拉丁字母
    "Sardinian": "srd_Latn",  # 萨丁尼亚语使用拉丁字母
    "Serbian": "srp_Cyrl",  # 塞尔维亚语使用西里尔字母
    "Swati": "ssw_Latn",  # 斯瓦蒂语使用拉丁字母
    "Sundanese": "sun_Latn",  # 巽他语使用拉丁字母
    "Swedish": "swe_Latn",  # 瑞典语使用拉丁字母
    "Swahili": "swh_Latn",  # 斯瓦希里语使用拉丁字母
    "Silesian": "szl_Latn",  # 西利西亚语使用拉丁字母
    "Tamil": "tam_Taml",  # 泰米尔语使用泰米尔字母
    "Tatar": "tat_Cyrl",  # 鞑靼语��用西里尔字母
    "Telugu": "tel_Telu",  # 泰卢固语使用泰卢固字母
    "Tajik": "tgk_Cyrl",  # 塔吉克语使用西里尔字母
    "Tagalog": "tgl_Latn",  # 塔加拉语使用拉丁字母
    "Thai": "tha_Thai",  # 泰语使用泰文
    "Tigrinya": "tir_Ethi",  # 提格利尼亚语使用埃塞俄比亚字母
    "Tamasheq Latin": "taq_Latn",  # 拉丁字母版塔马舍克语
    "Tamasheq Tifinagh": "taq_Tfng",  # 提非纳字母版塔马舍克语
    "Tok Pisin": "tpi_Latn",  # 托克皮辛语
    "Tswana": "tsn_Latn",  # 塞茨瓦纳语
    "Tsonga": "tso_Latn",  # 宗加语
    "Turkmen": "tuk_Latn",  # 土库曼语
    "Tumbuka": "tum_Latn",  # 通布卡语
    "Turkish": "tur_Latn",  # 土耳其语
    "Twi": "twi_Latn",  # 特威语
    "Central Atlas Tamazight": "tzm_Tfng",  # 中央阿特拉斯塔马齐格特语
    "Uyghur": "uig_Arab",  # 维吾尔语
    "Ukrainian": "ukr_Cyrl",  # 乌克兰语
    "Umbundu": "umb_Latn",  # 温布杜语
    "Urdu": "urd_Arab",  # 乌尔都语
    "Northern Uzbek": "uzn_Latn",  # 北部乌兹别克语
    "Venetian": "vec_Latn",  # 威尼斯语
    "Vietnamese": "vie_Latn",  # 越南语
    "Waray": "war_Latn",  # 瓦赖语
    "Wolof": "wol_Latn",  # 沃洛夫语
    "Xhosa": "xho_Latn",  # 科萨语
    "Eastern Yiddish": "ydd_Hebr",  # 东部意第绪语
    "Yoruba": "yor_Latn",  # 约鲁巴语
    "Yue Chinese": "yue_Hant",  # 粤语
    "Chinese Simplified": "zho_Hans",  # 简体中文
    "Chinese Traditional": "zho_Hant",  # 繁体中文
    "Standard Malay": "zsm_Latn",  # 标准马来语
    "Zulu": "zul_Latn",  # 祖鲁语
# 定义一个翻译工具类，继承自PipelineTool
class TranslationTool(PipelineTool):
    """
    Example:

    ```py
    from transformers.tools import TranslationTool

    translator = TranslationTool()
    translator("This is a super nice API!", src_lang="English", tgt_lang="French")
    ```
    """

    # 默认的检查点
    default_checkpoint = "facebook/nllb-200-distilled-600M"
    # 描述翻译工具的功能和用法
    description = (
        "This is a tool that translates text from a language to another. It takes three inputs: `text`, which should "
        "be the text to translate, `src_lang`, which should be the language of the text to translate and `tgt_lang`, "
        "which should be the language for the desired ouput language. Both `src_lang` and `tgt_lang` are written in "
        "plain English, such as 'Romanian', or 'Albanian'. It returns the text translated in `tgt_lang`."
    )
    # 工具的名称
    name = "translator"
    # 预处理器类
    pre_processor_class = AutoTokenizer
    # 模型类
    model_class = AutoModelForSeq2SeqLM
    # 语言到代码的映射
    lang_to_code = LANGUAGE_CODES

    # 输入参数
    inputs = ["text", "text", "text"]
    # 输出参数
    outputs = ["text"]

    # 编码方法，将输入文本编码成模型可接受的格式
    def encode(self, text, src_lang, tgt_lang):
        # 检查源语言是否支持
        if src_lang not in self.lang_to_code:
            raise ValueError(f"{src_lang} is not a supported language.")
        # 检查目标语言是否支持
        if tgt_lang not in self.lang_to_code:
            raise ValueError(f"{tgt_lang} is not a supported language.")
        # 将源语言和目标语言转换成对应的代码
        src_lang = self.lang_to_code[src_lang]
        tgt_lang = self.lang_to_code[tgt_lang]
        # 构建翻译输入
        return self.pre_processor._build_translation_inputs(
            text, return_tensors="pt", src_lang=src_lang, tgt_lang=tgt_lang
        )

    # 前向传播方法，调用模型生成翻译结果
    def forward(self, inputs):
        return self.model.generate(**inputs)

    # 解码方法，将模型输出解码成文本
    def decode(self, outputs):
        return self.post_processor.decode(outputs[0].tolist(), skip_special_tokens=True)
```