# `.\transformers\models\marian\convert_marian_tatoeba_to_pytorch.py`

```py
# 版权声明和许可证信息
# 代码版权归 The HuggingFace Team 所有，保留所有权利
#
# 根据 Apache 许可证 2.0 版本进行许可；
# 除非遵守许可证的规定，否则您不能使用此文件
# 您可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按 "原样" 分发，
# 没有任何形式的担保或条件，包括但不限于明示或暗示的担保或条件。
# 请查阅许可证了解详细信息。

# 导入所需模块和库
import argparse  # 用于解析命令行参数的模块
import datetime  # 用于处理日期和时间的模块
import json  # 用于处理 JSON 格式数据的模块
import os  # 提供与操作系统交互的功能
import re  # 提供正则表达式匹配操作的功能
from pathlib import Path  # 用于处理文件和目录路径的模块
from typing import Tuple  # 用于类型注解的模块

import yaml  # 用于解析和生成 YAML 格式数据的模块
from tqdm import tqdm  # 用于创建进度条的模块

# 从 transformers 包中导入 marian 模块下的转换函数和工具函数
from transformers.models.marian.convert_marian_to_pytorch import (
    FRONT_MATTER_TEMPLATE,
    convert,
    convert_opus_name_to_hf_name,
    download_and_unzip,
    get_system_metadata,
)

# 默认的 Tatoeba 仓库路径
DEFAULT_REPO = "Tatoeba-Challenge"
# 默认的模型目录路径
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_REPO, "models")
# 语言代码数据源的 URL
LANG_CODE_URL = "https://datahub.io/core/language-codes/r/language-codes-3b2.csv"
# ISO 语言代码数据源的 URL
ISO_URL = "https://cdn-datasets.huggingface.co/language_codes/iso-639-3.csv"
# ISO 语言代码数据的本地路径
ISO_PATH = "lang_code_data/iso-639-3.csv"
# 语言代码数据的本地路径
LANG_CODE_PATH = "lang_code_data/language-codes-3b2.csv"
# Tatoeba 模型的下载链接
TATOEBA_MODELS_URL = "https://object.pouta.csc.fi/Tatoeba-MT-models"

# TatoebaConverter 类，用于将 Tatoeba-Challenge 模型转换为 huggingface 格式
class TatoebaConverter:
    """
    Convert Tatoeba-Challenge models to huggingface format.

    Steps:

        1. Convert numpy state dict to hf format (same code as OPUS-MT-Train conversion).
        2. Rename opus model to huggingface format. This means replace each alpha3 code with an alpha2 code if a unique
           one exists. e.g. aav-eng -> aav-en, heb-eng -> he-en
        3. Select the best model for a particular pair, parse the yml for it and write a model card. By default the
           best model is the one listed first in released-model-results, but it's also possible to specify the most
           recent one.
    """

    # 初始化方法
    def __init__(self, save_dir="marian_converted"):
        # 检查是否已经克隆了 Tatoeba-Challenge 仓库
        assert Path(DEFAULT_REPO).exists(), "need git clone git@github.com:Helsinki-NLP/Tatoeba-Challenge.git"
        # 下载语言信息数据
        self.download_lang_info()
        # 加载模型结果数据
        self.model_results = json.load(open("Tatoeba-Challenge/models/released-model-results.json"))
        # 初始化 alpha3 到 alpha2 语言代码的映射字典
        self.alpha3_to_alpha2 = {}
        # 从 ISO 语言代码数据中读取并构建 alpha3 到 alpha2 的映射
        for line in open(ISO_PATH):
            parts = line.split("\t")
            if len(parts[0]) == 3 and len(parts[3]) == 2:
                self.alpha3_to_alpha2[parts[0]] = parts[3]
        # 从语言代码数据中读取并构建 alpha3 到 alpha2 的映射
        for line in LANG_CODE_PATH:
            parts = line.split(",")
            if len(parts[0]) == 3 and len(parts[1]) == 2:
                self.alpha3_to_alpha2[parts[0]] = parts[1]
        # 初始化模型卡片目录路径
        self.model_card_dir = Path(save_dir)
        # 初始化标签到名称的映射字典
        self.tag2name = {}
        # 从 GROUP_MEMBERS 中构建标签到名称的映射
        for key, value in GROUP_MEMBERS.items():
            self.tag2name[key] = value[0]
``` 
    # 将给定的 Tatoeba 句子对 ID 转换为模型元数据列表
    def convert_models(self, tatoeba_ids, dry_run=False):
        # 解析每个 Tatoeba 句子对的元数据
        models_to_convert = [self.parse_metadata(x) for x in tatoeba_ids]
        # 定义保存转换后模型的目录
        save_dir = Path("marian_ckpt")
        # 定义目标目录，用于保存模型卡片
        dest_dir = Path(self.model_card_dir)
        # 如果目标目录不存在，则创建
        dest_dir.mkdir(exist_ok=True)
        # 遍历待转换的模型列表并显示进度条
        for model in tqdm(models_to_convert):  # k, prepro, download, test_set_url in tqdm(model_list):
            # 如果模型预处理中不包含 SentencePiece，则跳过该模型的转换
            if "SentencePiece" not in model["pre-processing"]:
                print(f"Skipping {model['release']} because it doesn't appear to use SentencePiece")
                continue
            # 如果保存目录中不存在该模型，则下载并解压缩模型文件
            if not os.path.exists(save_dir / model["_name"]):
                download_and_unzip(f"{TATOEBA_MODELS_URL}/{model['release']}", save_dir / model["_name"])
            # 将模型转换为 PyTorch 格式
            opus_language_groups_to_hf = convert_opus_name_to_hf_name
            pair_name = opus_language_groups_to_hf(model["_name"])
            convert(save_dir / model["_name"], dest_dir / f"opus-mt-{pair_name}")
            # 写入模型卡片信息
            self.write_model_card(model, dry_run=dry_run)

    # 将群组名称扩展为两字母代码
    def expand_group_to_two_letter_codes(self, grp_name):
        return [self.alpha3_to_alpha2.get(x, x) for x in GROUP_MEMBERS[grp_name][1]]

    # 判断给定代码和名称是否为群组
    def is_group(self, code, name):
        return "languages" in name or len(GROUP_MEMBERS.get(code, [])) > 1

    # 获取给定代码对应的标签列表
    def get_tags(self, code, name):
        # 如果代码为两字母代码，则直接返回该代码作为标签
        if len(code) == 2:
            assert "languages" not in name, f"{code}: {name}"
            return [code]
        # 如果代码为群组代码，则扩展成包含该群组内所有语言的标签列表
        elif self.is_group(code, name):
            group = self.expand_group_to_two_letter_codes(code)
            group.append(code)
            return group
        # 否则，输出警告信息，返回原始代码作为标签
        else:  # zho-> zh
            print(f"Three letter monolingual code: {code}")
            return [code]

    # 解析源语言和目标语言的代码，返回对应的标签列表
    def resolve_lang_code(self, src, tgt) -> Tuple[str, str]:
        src_tags = self.get_tags(src, self.tag2name[src])
        tgt_tags = self.get_tags(tgt, self.tag2name[tgt])
        return src_tags, tgt_tags

    # 从模型名称中提取模型类型信息
    @staticmethod
    def model_type_info_from_model_name(name):
        info = {"_has_backtranslated_data": False}
        # 根据模型名称判断模型是否包含后向翻译数据
        if "1m" in name:
            info["_data_per_pair"] = str(1e6)
        if "2m" in name:
            info["_data_per_pair"] = str(2e6)
        if "4m" in name:
            info["_data_per_pair"] = str(4e6)
        if "+bt" in name:
            info["_has_backtranslated_data"] = True
        if "tuned4" in name:
            info["_tuned"] = re.search(r"tuned4[^-]+", name).group()
        return info
### {model_dict['_name']}

# 输出模型名称信息
* source language name: {self.tag2name[a3_src]}
# 输出源语言名称
* target language name: {self.tag2name[a3_tgt]}
# 输出目标语言名称
* OPUS readme: [README.md]({readme_url})
# 输出 OPUS readme 的链接

"""

# 生成 content 字符串，包含模型相关信息
content = (
    f"""
* model: {model_dict['modeltype']}
# 输出模型类型
* source language code{src_multilingual*'s'}: {', '.join(a2_src_tags)}
# 输出源语言代码（多语言情况下使用复数形式）
* target language code{tgt_multilingual*'s'}: {', '.join(a2_tgt_tags)}
# 输出目标语言代码（多语言情况下使用复数形式）
* dataset: opus {backtranslated_data}
# 输出数据集信息，包括 opus 和反向翻译数据
* release date: {model_dict['release-date']}
# 输出模型发布日期
* pre-processing: {model_dict['pre-processing']}
# 输出预处理信息
"""
    + multilingual_data
    + tuned
    + download
    + langtoken
    + datainfo
    + testset
    + testscores
    + scorestable
)

# 拼接 front matter 模板和额外的 markdown 内容
content = FRONT_MATTER_TEMPLATE.format(lang_tags) + extra_markdown + content

# 生成 metadata 字符串
items = "\n".join([f"* {k}: {v}" for k, v in metadata.items()])
# 生成 System Info 部分的字符串
sec3 = "\n### System Info: \n" + items
# 将 System Info 部分添加到 content 中
content += sec3

# 如果是 dry_run 模式，则打印 content 和 metadata 并返回
if dry_run:
    print("CONTENT:")
    print(content)
    print("METADATA:")
    print(metadata)
    return

# 创建模型卡片的子目录，并写入 README.md 文件和 metadata.json 文件
sub_dir = self.model_card_dir / model_dict["_hf_model_id"]
sub_dir.mkdir(exist_ok=True)
dest = sub_dir / "README.md"
dest.open("w").write(content)
for k, v in metadata.items():
    if isinstance(v, datetime.date):
        metadata[k] = datetime.datetime.strftime(v, "%Y-%m-%d")
with open(sub_dir / "metadata.json", "w", encoding="utf-8") as writeobj:
    json.dump(metadata, writeobj)

def download_lang_info(self):
    # 创建语言代码目录的父目录（如果不存在）
    Path(LANG_CODE_PATH).parent.mkdir(exist_ok=True)
    import wget

    # 下载 ISO 文件
    if not os.path.exists(ISO_PATH):
        wget.download(ISO_URL, ISO_PATH)
    # 下载语言代码文件
    if not os.path.exists(LANG_CODE_PATH):
        wget.download(LANG_CODE_URL, LANG_CODE_PATH)
    # 解析模型元数据，根据模型名称、存储库路径和方法返回元数据字典
    def parse_metadata(self, model_name, repo_path=DEFAULT_MODEL_DIR, method="best"):
        # 根据存储库路径和模型名称创建路径对象
        p = Path(repo_path) / model_name

        def url_to_name(url):
            # 从 URL 中提取文件名并去掉扩展名，作为名称返回
            return url.split("/")[-1].split(".")[0]

        if model_name not in self.model_results:
            # 如果模型名称不在结果中，则模型结果不明确，按照最新模型处理
            method = "newest"

        if method == "best":
            # 按照模型结果中下载链接出现的顺序排序
            results = [url_to_name(model["download"]) for model in self.model_results[model_name]]
            # 获取文件夹中以 .yml 结尾且在结果中的文件名列表
            ymls = [f for f in os.listdir(p) if f.endswith(".yml") and f[:-4] in results]
            # 按照结果中的顺序排序 ymls
            ymls.sort(key=lambda x: results.index(x[:-4]))
            # 加载第一个 yml 文件的元数据，并根据模型名称获取模型类型信息
            metadata = yaml.safe_load(open(p / ymls[0]))
            metadata.update(self.model_type_info_from_model_name(ymls[0][:-4]))
        elif method == "newest":
            # 获取文件夹中以 .yml 结尾的文件名列表
            ymls = [f for f in os.listdir(p) if f.endswith(".yml")]
            # 按照日期排序 ymls
            ymls.sort(
                key=lambda x: datetime.datetime.strptime(re.search(r"\d\d\d\d-\d\d?-\d\d?", x).group(), "%Y-%m-%d")
            )
            # 加载最新的 yml 文件的元数据，并根据模型名称获取模型类型信息
            metadata = yaml.safe_load(open(p / ymls[-1]))
            metadata.update(self.model_type_info_from_model_name(ymls[-1][:-4]))
        else:
            # 抛出未实现的错误，提示不认识的 method 参数值
            raise NotImplementedError(f"Don't know argument method='{method}' to parse_metadata()")
        # 将模型名称作为键添加到元数据字典中
        metadata["_name"] = model_name
        # 返回处理后的元数据字典
        return metadata
GROUP_MEMBERS = {
    # 语言代码对应语言组/语言名称，以及该语言组/语言所包含的子语言代码
    # 如果该语言是目标语言端，子语言代码可以被用作目标语言代码
    # 如果该语言是源语言端，子语言代码可以被原生支持，无需特殊代码
    "aav": ("Austro-Asiatic languages", {"hoc", "hoc_Latn", "kha", "khm", "khm_Latn", "mnw", "vie", "vie_Hani"}),
    "afa": (
        "Afro-Asiatic languages",
        {
            "acm",
            "afb",
            "amh",
            "apc",
            "ara",
            "arq",
            "ary",
            "arz",
            "hau_Latn",
            "heb",
            "kab",
            "mlt",
            "rif_Latn",
            "shy_Latn",
            "som",
            "thv",
            "tir",
        },
    ),
    "afr": ("Afrikaans", {"afr"}),
    "alv": (
        "Atlantic-Congo languages",
        {
            "ewe",
            "fuc",
            "fuv",
            "ibo",
            "kin",
            "lin",
            "lug",
            "nya",
            "run",
            "sag",
            "sna",
            "swh",
            "toi_Latn",
            "tso",
            "umb",
            "wol",
            "xho",
            "yor",
            "zul",
        },
    ),
    "ara": ("Arabic", {"afb", "apc", "apc_Latn", "ara", "ara_Latn", "arq", "arq_Latn", "arz"}),
    "art": (
        "Artificial languages",
        {
            "afh_Latn",
            "avk_Latn",
            "dws_Latn",
            "epo",
            "ido",
            "ido_Latn",
            "ile_Latn",
            "ina_Latn",
            "jbo",
            "jbo_Cyrl",
            "jbo_Latn",
            "ldn_Latn",
            "lfn_Cyrl",
            "lfn_Latn",
            "nov_Latn",
            "qya",
            "qya_Latn",
            "sjn_Latn",
            "tlh_Latn",
            "tzl",
            "tzl_Latn",
            "vol_Latn",
        },
    ),
    "aze": ("Azerbaijani", {"aze_Latn"}),
    "bat": ("Baltic languages", {"lit", "lav", "prg_Latn", "ltg", "sgs"}),
    "bel": ("Belarusian", {"bel", "bel_Latn"}),
    "ben": ("Bengali", {"ben"}),
    "bnt": (
        "Bantu languages",
        {"kin", "lin", "lug", "nya", "run", "sna", "swh", "toi_Latn", "tso", "umb", "xho", "zul"},
    ),
    "bul": ("Bulgarian", {"bul", "bul_Latn"}),
    "cat": ("Catalan", {"cat"}),
    "cau": ("Caucasian languages", {"abk", "kat", "che", "ady"}),
    "ccs": ("South Caucasian languages", {"kat"}),
    "ceb": ("Cebuano", {"ceb"}),
    "cel": ("Celtic languages", {"gla", "gle", "bre", "cor", "glv", "cym"}),
    "ces": ("Czech", {"ces"}),
    "cpf": ("Creoles and pidgins, French‑based", {"gcf_Latn", "hat", "mfe"}),
    "cpp": (
        "Creoles and pidgins, Portuguese-based",
        {"zsm_Latn", "ind", "pap", "min", "tmw_Latn", "max_Latn", "zlm_Latn"},
    ),
    "cus": ("Cushitic languages", {"som"}),
    "dan": ("Danish", {"dan"}),
    "deu": ("German", {"deu"}),
    # "dra" 对应 "Dravidian languages"，包含语言代码为 {"tam", "kan", "mal", "tel"}
    "dra": ("Dravidian languages", {"tam", "kan", "mal", "tel"}),
    # "ell" 对应 "Modern Greek (1453-)"，包含语言代码为 {"ell"}
    "ell": ("Modern Greek (1453-)", {"ell"}),
    # "eng" 对应 "English"，包含语言代码为 {"eng"}
    "eng": ("English", {"eng"}),
    # "epo" 对应 "Esperanto"，包含语言代码为 {"epo"}
    "epo": ("Esperanto", {"epo"}),
    # "est" 对应 "Estonian"，包含语言代码为 {"est"}
    "est": ("Estonian", {"est"}),
    # "euq" 对应 "Basque (family)"，包含语言代码为 {"eus"}
    "euq": ("Basque (family)", {"eus"}),
    # "eus" 对应 "Basque"，包含语言代码为 {"eus"}
    "eus": ("Basque", {"eus"}),
    # "fin" 对应 "Finnish"，包含语言代码为 {"fin"}
    "fin": ("Finnish", {"fin"}),
    # "fiu" 对应 "Finno-Ugrian languages"，包含语言代码为以下集合
    # {"est", "fin", "fkv_Latn", "hun", "izh", "kpv", "krl", "liv_Latn", "mdf", "mhr", "myv", "sma", "sme", "udm", "vep", "vro"}
    "fiu": (
        "Finno-Ugrian languages",
        {
            "est",
            "fin",
            "fkv_Latn",
            "hun",
            "izh",
            "kpv",
            "krl",
            "liv_Latn",
            "mdf",
            "mhr",
            "myv",
            "sma",
            "sme",
            "udm",
            "vep",
            "vro",
        },
    ),
    # "fra" 对应 "French"，包含语言代码为 {"fra"}
    "fra": ("French", {"fra"}),
    # "gem" 对应 "Germanic languages"，包含语言代码为以下集合
    # {"afr", "ang_Latn", "dan", "deu", "eng", "enm_Latn", "fao", "frr", "fry", "gos", "got_Goth", "gsw", "isl", "ksh", "ltz", "nds", "nld", "nno", "nob", "nob_Hebr", "non_Latn", "pdc", "sco", "stq", "swe", "swg", "yid"}
    "gem": (
        "Germanic languages",
        {
            "afr",
            "ang_Latn",
            "dan",
            "deu",
            "eng",
            "enm_Latn",
            "fao",
            "frr",
            "fry",
            "gos",
            "got_Goth",
            "gsw",
            "isl",
            "ksh",
            "ltz",
            "nds",
            "nld",
            "nno",
            "nob",
            "nob_Hebr",
            "non_Latn",
            "pdc",
            "sco",
            "stq",
            "swe",
            "swg",
            "yid",
        },
    ),
    # "gle" 对应 "Irish"，包含语言代码为 {"gle"}
    "gle": ("Irish", {"gle"}),
    # "glg" 对应 "Galician"，包含语言代码为 {"glg"}
    "glg": ("Galician", {"glg"}),
    # "gmq" 对应 "North Germanic languages"，包含语言代码为以下集合
    # {"dan", "nob", "nob_Hebr", "swe", "isl", "nno", "non_Latn", "fao"}
    "gmq": ("North Germanic languages", {"dan", "nob", "nob_Hebr", "swe", "isl", "nno", "non_Latn", "fao"}),
    # "gmw" 对应 "West Germanic languages"，包含语言代码为以下集合
    # {"afr", "ang_Latn", "deu", "eng", "enm_Latn", "frr", "fry", "gos", "gsw", "ksh", "ltz", "nds", "nld", "pdc", "sco", "stq", "swg", "yid"}
    "gmw": (
        "West Germanic languages",
        {
            "afr",
            "ang_Latn",
            "deu",
            "eng",
            "enm_Latn",
            "frr",
            "fry",
            "gos",
            "gsw",
            "ksh",
            "ltz",
            "nds",
            "nld",
            "pdc",
            "sco",
            "stq",
            "swg",
            "yid",
        },
    ),
    # "grk" 对应 "Greek languages"，包含语言代码为 {"grc_Grek", "ell"}
    "grk": ("Greek languages", {"grc_Grek", "ell"}),
    # "hbs" 对应 "Serbo-Croatian"，包含语言代码为 {"hrv", "srp_Cyrl", "bos_Latn", "srp_Latn"}
    "hbs": ("Serbo-Croatian", {"hrv", "srp_Cyrl", "bos_Latn", "srp_Latn"}),
    # "heb" 对应 "Hebrew"，包含语言代码为 {"heb"}
    "heb": ("Hebrew", {"heb"}),
    # "hin" 对应 "Hindi"，包含语言代码为 {"hin"}
    "hin": ("Hindi", {"hin"}),
    # "hun" 对应 "Hungarian"，包含语言代码为 {"hun"}
    "hun": ("Hungarian", {"hun"}),
    # "hye" 对应 "Armenian"，包含语言代码为 {"hye", "hye_Latn"}
    "hye": ("Armenian", {"hye", "hye_Latn"}),
    # "iir" 对应 "Indo-Iranian languages"，包含语言代码为以下集合
    # {"asm", "awa", "ben", "bho", "
    # "inc" 键对应的值是一个元组，包含两个元素：一个字符串和一个集合
    "inc": (
        # 字符串 "Indic languages" 描述了集合中语言的类别
        "Indic languages",
        # 集合包含了各种印度语言的标识符
        {
            "asm",        # 阿萨姆语
            "awa",        # 阿瓦德语
            "ben",        # 孟加拉语
            "bho",        # 博杰普尔语
            "gom",        # 孟加拉古阿姆孟语
            "guj",        # 古吉拉特语
            "hif_Latn",   # 斐济希腊语言（拉丁文）
            "hin",        # 北印度语
            "mai",        # 马蒂利语
            "mar",        # 马拉地语
            "npi",        # 尼泊尔语
            "ori",        # 奥里亚语
            "pan_Guru",   # 旁遮普语（古鲁穆基文）
            "pnb",        # 西旁遮普语
            "rom",        # 罗姆语
            "san_Deva",   # 梵语（天城文）
            "sin",        # 锡兰语
            "snd_Arab",   # 信德语（阿拉伯文）
            "urd",        # 乌尔都语
        },
    ),
    # 创建一个名为 "ine" 的元组，包含一个字符串和一个集合
    "ine": (
        "Indo-European languages",  # 字符串，描述该语言族
        {  # 集合，包含多个语言代码
            "afr",  # 南非荷兰语
            "afr_Arab",  # 南非荷兰语（使用阿拉伯字母）
            "aln",  # 盲文
            "ang_Latn",  # 古英语
            ...
            "zsm_Latn",  # 马来语
            "zza",  # 扎扎语
        },
    ),
    # 创建一个名为 "isl" 的元组，包含一个字符串和一个集合
    "isl": ("Icelandic", {"isl"}),  # 冰岛语
    # 创建一个名为 "ita" 的元组，包含一个字符串和一个集合
    "ita": ("Italian", {"ita"}),  # 意大利语
    # "itc" 代表 Italic 语系相关语言
    "itc": (
        # 意大利语族中包含的语言
        "Italic languages",
        {
            "arg",
            "ast",
            "bjn",
            "cat",
            "cos",
            "egl",
            "ext",
            "fra",
            "frm_Latn",
            "gcf_Latn",
            "glg",
            "hat",
            "ind",
            "ita",
            "lad",
            "lad_Latn",
            "lat_Grek",
            "lat_Latn",
            "lij",
            "lld_Latn",
            "lmo",
            "max_Latn",
            "mfe",
            "min",
            "mwl",
            "oci",
            "pap",
            "pcd",
            "pms",
            "por",
            "roh",
            "ron",
            "scn",
            "spa",
            "srd",
            "tmw_Latn",
            "vec",
            "wln",
            "zlm_Latn",
            "zsm_Latn",
        },
    ),
    # "jpn" 代表日语
    "jpn": ("Japanese", {"jpn", "jpn_Bopo", "jpn_Hang", "jpn_Hani", "jpn_Hira", "jpn_Kana", "jpn_Latn", "jpn_Yiii"}),
    # "jpx" 代表日语家族
    "jpx": ("Japanese (family)", {"jpn"}),
    # "kat" 代表格鲁吉亚语
    "kat": ("Georgian", {"kat"}),
    # "kor" 代表韩语
    "kor": ("Korean", {"kor_Hani", "kor_Hang", "kor_Latn", "kor"}),
    # "lav" 代表拉脱维亚语
    "lav": ("Latvian", {"lav"}),
    # "lit" 代表立陶宛语
    "lit": ("Lithuanian", {"lit"}),
    # "mkd" 代表马其顿语
    "mkd": ("Macedonian", {"mkd"}),
    # "mkh" 代表孟高棉语系
    "mkh": ("Mon-Khmer languages", {"vie_Hani", "mnw", "vie", "kha", "khm_Latn", "khm"}),
    # "msa" 代表马来语（宏观语言）
    "msa": ("Malay (macrolanguage)", {"zsm_Latn", "ind", "max_Latn", "zlm_Latn", "min"}),
    # "nic" 代表尼日科多法尼亚语系
    "nic": (
        # 日尔曼诸语族中包含的语言
        "Niger-Kordofanian languages",
        {
            "bam_Latn",
            "ewe",
            "fuc",
            "fuv",
            "ibo",
            "kin",
            "lin",
            "lug",
            "nya",
            "run",
            "sag",
            "sna",
            "swh",
            "toi_Latn",
            "tso",
            "umb",
            "wol",
            "xho",
            "yor",
            "zul",
        },
    ),
    # "nld" 代表荷兰语
    "nld": ("Dutch", {"nld"}),
    # "nor" 代表挪威语
    "nor": ("Norwegian", {"nob", "nno"}),
    # "phi" 代表菲��宾语系
    "phi": ("Philippine languages", {"ilo", "akl_Latn", "war", "hil", "pag", "ceb"}),
    # "pol" 代表波兰语
    "pol": ("Polish", {"pol"}),
    # "por" 代表葡萄牙语
    "por": ("Portuguese", {"por"}),
    # "pqe" 代表东马来波利尼西亚语系
    "pqe": (
        # 东马来波利尼西亚语系中包含的语言
        "Eastern Malayo-Polynesian languages",
        {"fij", "gil", "haw", "mah", "mri", "nau", "niu", "rap", "smo", "tah", "ton", "tvl"},
    ),
    "roa": (
        "Romance languages",  # 键为"roa"，值是元组，代表罗曼语言
        {
            "arg",          # 阿根廷语
            "ast",          # 阿斯图里亚斯语
            "cat",          # 加泰罗尼亚语
            "cos",          # 科西嘉语
            "egl",          # 埃米利亚-罗马涅语
            "ext",          # 埃斯特雷马杜拉语
            "fra",          # 法语
            "frm_Latn",     # 拉丁字母法语
            "gcf_Latn",     # 拉丁字母德国法语
            "glg",          # 加利西亚语
            "hat",          # 海地克里奥尔语
            "ind",          # 印度尼西亚语
            "ita",          # 意大利语
            "lad",          # 莱迪诺语
            "lad_Latn",     # 拉丁字母莱迪诺语
            "lij",          # 利古里亚语
            "lld_Latn",     # 拉丁字母吕达语
            "lmo",          # 伦巴第语
            "max_Latn",     # 拉丁字母马勒加什尼特语
            "mfe",          # 毛里求斯克里奥尔语
            "min",          # 明亚语
            "mwl",          # 米兰德斯语
            "oci",          # 奥克语
            "pap",          # 路易斯语
            "pms",          # 皮埃蒙特语
            "por",          # 葡萄牙语
            "roh",          # 罗曼什语
            "ron",          # 罗马尼亚语
            "scn",          # 西西里语
            "spa",          # 西班牙语
            "tmw_Latn",     # 拉丁字母特莫斯语
            "vec",          # 威尼斯语
            "wln",          # 瓦隆语
            "zlm_Latn",     # 拉丁字母马来语
            "zsm_Latn",     # 拉丁字母马来西亚语
        },
    ),
    "ron": ("Romanian", {"ron"}),  # 键为"ron"，值为元组，代表罗马尼亚语
    "run": ("Rundi", {"run"}),  # 键为"run"，值为元组，代表隆迪语
    "rus": ("Russian", {"rus"}),  # 键为"rus"，值为元组，代表俄语
    "sal": ("Salishan languages", {"shs_Latn"}),  # 键为"sal"，值为元组，代表萨利什语言
    "sem": ("Semitic languages", {"acm", "afb", "amh", "apc", "ara", "arq", "ary", "arz", "heb", "mlt", "tir"}),  # 键为"sem"，值为元组，代表闪米特语言
    "sla": (
        "Slavic languages",  # 键为"sla"，值为元组，代表斯拉夫语言
        {
            "bel",          # 白俄罗斯语
            "bel_Latn",     # 拉丁字母白俄罗斯语
            "bos_Latn",     # 拉丁字母波斯尼亚语
            "bul",          # 保加利亚语
            "bul_Latn",     # 拉丁字母保加利亚语
            "ces",          # 捷克语
            "csb_Latn",     # 拉丁字母卡舒比亚语
            "dsb",          # 下索布诺语
...
    # 定义语种代码 "zho"
    "zho": (
        # 中文的含义
        "Chinese",
        # 中文的子类语种代码集合
        {
            "cjy_Hans",
            "cjy_Hant",
            "cmn",
            "cmn_Bopo",
            "cmn_Hang",
            "cmn_Hani",
            "cmn_Hans",
            "cmn_Hant",
            "cmn_Hira",
            "cmn_Kana",
            "cmn_Latn",
            "cmn_Yiii",
            "gan",
            "hak_Hani",
            "lzh",
            "lzh_Bopo",
            "lzh_Hang",
            "lzh_Hani",
            "lzh_Hans",
            "lzh_Hira",
            "lzh_Kana",
            "lzh_Yiii",
            "nan",
            "nan_Hani",
            "wuu",
            "wuu_Bopo",
            "wuu_Hani",
            "wuu_Latn",
            "yue",
            "yue_Bopo",
            "yue_Hang",
            "yue_Hani",
            "yue_Hans",
            "yue_Hant",
            "yue_Hira",
            "yue_Kana",
            "zho",
            "zho_Hans",
            "zho_Hant",
        },
    ),
    # 定义语种代码 "zle"
    "zle": ("East Slavic languages", {"bel", "orv_Cyrl", "bel_Latn", "rus", "ukr", "rue"}),
    # 定义语种代码 "zls"
    "zls": ("South Slavic languages", {"bos_Latn", "bul", "bul_Latn", "hrv", "mkd", "slv", "srp_Cyrl", "srp_Latn"}),
    # 定义语种代码 "zlw"
    "zlw": ("West Slavic languages", {"csb_Latn", "dsb", "hsb", "pol", "ces"}),
def l2front_matter(langs):
    # 返回一个包含语言列表中每个语言的前缀的字符串
    return "".join(f"- {l}\n" for l in langs)


def dedup(lst):
    """Preservers order"""
    # 创建一个新的列表，保留原始顺序
    new_lst = []
    # 遍历原始列表中的每个元素
    for item in lst:
        # 如果元素为空或者已经存在于新列表中，则跳过
        if not item or item in new_lst:
            continue
        else:
            # 否则将元素添加到新列表中
            new_lst.append(item)
    # 返回去重后的列表
    return new_lst


if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数：models，要求参数必须存在，类型为列表，接受多个值
    parser.add_argument(
        "-m", "--models", action="append", help="<Required> Set flag", required=True, nargs="+", dest="models"
    )
    # 添加命令行参数：save_dir，指定默认值为"marian_converted"，用于保存转换后的模型
    parser.add_argument("-save_dir", "--save_dir", default="marian_converted", help="where to save converted models")
    # 解析命令行参数
    args = parser.parse_args()
    # 创建TatoebaConverter对象，指定保存目录
    resolver = TatoebaConverter(save_dir=args.save_dir)
    # 转换模型
    resolver.convert_models(args.models[0])
```py  
```