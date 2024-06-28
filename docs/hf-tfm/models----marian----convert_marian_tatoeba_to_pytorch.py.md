# `.\models\marian\convert_marian_tatoeba_to_pytorch.py`

```
# 导入必要的模块和库
import argparse  # 解析命令行参数的库
import datetime  # 处理日期和时间的库
import json  # 处理 JSON 数据的库
import os  # 提供与操作系统交互的功能
import re  # 提供正则表达式操作的库
from pathlib import Path  # 提供处理文件路径的类
from typing import Tuple  # 提供类型提示支持

import yaml  # 处理 YAML 格式的库
from tqdm import tqdm  # 提供进度条功能

# 从 transformers 库中导入相关模块和函数
from transformers.models.marian.convert_marian_to_pytorch import (
    FRONT_MATTER_TEMPLATE,  # 导入一个变量：Marian 模型转换时使用的前置模板
    convert,  # 导入一个函数：用于转换模型
    convert_opus_name_to_hf_name,  # 导入一个函数：用于转换 OPUS 模型名称为 HF 模型名称
    download_and_unzip,  # 导入一个函数：用于下载并解压文件
    get_system_metadata,  # 导入一个函数：获取系统元数据
)

# 设置默认的仓库名称和模型目录路径
DEFAULT_REPO = "Tatoeba-Challenge"
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_REPO, "models")

# 定义语言代码信息的 URL
LANG_CODE_URL = "https://datahub.io/core/language-codes/r/language-codes-3b2.csv"
# 定义 ISO 语言代码的 URL
ISO_URL = "https://cdn-datasets.huggingface.co/language_codes/iso-639-3.csv"
# 定义存储 ISO 语言代码的本地路径
ISO_PATH = "lang_code_data/iso-639-3.csv"
# 定义存储语言代码信息的本地路径
LANG_CODE_PATH = "lang_code_data/language-codes-3b2.csv"
# 定义 Tatoeba 模型下载 URL
TATOEBA_MODELS_URL = "https://object.pouta.csc.fi/Tatoeba-MT-models"


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

    def __init__(self, save_dir="marian_converted"):
        # 检查默认仓库是否存在，否则给出错误提示
        assert Path(DEFAULT_REPO).exists(), "need git clone git@github.com:Helsinki-NLP/Tatoeba-Challenge.git"

        # 下载语言信息
        self.download_lang_info()

        # 加载模型结果数据
        self.model_results = json.load(open("Tatoeba-Challenge/models/released-model-results.json"))

        # 初始化 alpha3 到 alpha2 映射字典
        self.alpha3_to_alpha2 = {}
        # 从 ISO 文件中读取 alpha3 到 alpha2 的映射关系
        for line in open(ISO_PATH):
            parts = line.split("\t")
            if len(parts[0]) == 3 and len(parts[3]) == 2:
                self.alpha3_to_alpha2[parts[0]] = parts[3]

        # 从语言代码文件中读取 alpha3 到 alpha2 的映射关系
        for line in open(LANG_CODE_PATH):
            parts = line.split(",")
            if len(parts[0]) == 3 and len(parts[1]) == 2:
                self.alpha3_to_alpha2[parts[0]] = parts[1]

        # 设置模型卡片输出目录
        self.model_card_dir = Path(save_dir)

        # 初始化标签到名称的映射字典
        self.tag2name = {}
        # 从 GROUP_MEMBERS 中获取标签和名称的映射关系
        for key, value in GROUP_MEMBERS.items():
            self.tag2name[key] = value[0]
    # 将给定的 Tatoeba IDs 转换为模型元数据列表，如果 dry_run 为 True，则仅进行试运行
    def convert_models(self, tatoeba_ids, dry_run=False):
        # 解析每个 Tatoeba ID 对应的模型元数据，形成列表
        models_to_convert = [self.parse_metadata(x) for x in tatoeba_ids]
        # 设置保存目录为 "marian_ckpt"
        save_dir = Path("marian_ckpt")
        # 设置目标目录为指定的模型卡片目录，并确保目录存在
        dest_dir = Path(self.model_card_dir)
        dest_dir.mkdir(exist_ok=True)
        # 遍历待转换的模型元数据列表，显示进度条
        for model in tqdm(models_to_convert):  # k, prepro, download, test_set_url in tqdm(model_list):
            # 如果模型的预处理步骤中不包含 "SentencePiece"，则跳过转换
            if "SentencePiece" not in model["pre-processing"]:
                print(f"Skipping {model['release']} because it doesn't appear to use SentencePiece")
                continue
            # 如果保存目录中不存在当前模型的文件夹，则下载并解压对应的模型文件
            if not os.path.exists(save_dir / model["_name"]):
                download_and_unzip(f"{TATOEBA_MODELS_URL}/{model['release']}", save_dir / model["_name"])
            # 将模型从 Marian 转换为 PyTorch 格式，并保存到目标目录
            # 模型名称转换为适合 HF 格式的名称
            opus_language_groups_to_hf = convert_opus_name_to_hf_name
            pair_name = opus_language_groups_to_hf(model["_name"])
            convert(save_dir / model["_name"], dest_dir / f"opus-mt-{pair_name}")
            # 将模型的元数据写入模型卡片，如果 dry_run 为 True，则仅进行试运行
            self.write_model_card(model, dry_run=dry_run)

    # 根据组名扩展为其成员的两字母代码列表
    def expand_group_to_two_letter_codes(self, grp_name):
        return [self.alpha3_to_alpha2.get(x, x) for x in GROUP_MEMBERS[grp_name][1]]

    # 判断给定的代码和名称是否代表一个语言组
    def is_group(self, code, name):
        return "languages" in name or len(GROUP_MEMBERS.get(code, [])) > 1

    # 根据代码和名称获取标签列表
    def get_tags(self, code, name):
        if len(code) == 2:
            # 对于两字母代码，名称中不应包含 "languages"
            assert "languages" not in name, f"{code}: {name}"
            return [code]
        elif self.is_group(code, name):
            # 如果是语言组，则将组成员的两字母代码列表返回，并加入原始代码
            group = self.expand_group_to_two_letter_codes(code)
            group.append(code)
            return group
        else:  # zho-> zh
            # 对于三字母单一语言代码，输出警告信息
            print(f"Three letter monolingual code: {code}")
            return [code]

    # 解析语言代码，将源语言和目标语言转换为标签列表
    def resolve_lang_code(self, src, tgt) -> Tuple[str, str]:
        src_tags = self.get_tags(src, self.tag2name[src])
        tgt_tags = self.get_tags(tgt, self.tag2name[tgt])
        return src_tags, tgt_tags

    # 从模型名称中获取模型类型信息，返回一个字典
    @staticmethod
    def model_type_info_from_model_name(name):
        info = {"_has_backtranslated_data": False}
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
        content = (
            f"""
* model: {model_dict['modeltype']}
* source language code{src_multilingual*'s'}: {', '.join(a2_src_tags)}
* target language code{tgt_multilingual*'s'}: {', '.join(a2_tgt_tags)}
* dataset: opus {backtranslated_data}
* release date: {model_dict['release-date']}
* pre-processing: {model_dict['pre-processing']}
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
        # 构建模型卡片的内容，包括模型类型、源语言和目标语言代码、数据集信息等

        content = FRONT_MATTER_TEMPLATE.format(lang_tags) + extra_markdown + content
        # 将模型卡片的前置模板和额外的 markdown 内容插入到卡片内容开头

        items = "\n".join([f"* {k}: {v}" for k, v in metadata.items()])
        # 构建元数据字典的字符串表示，每个键值对形如 "* key: value"

        sec3 = "\n### System Info: \n" + items
        # 构建系统信息部分的标题和元数据内容

        content += sec3
        # 将系统信息部分添加到模型卡片的内容末尾

        if dry_run:
            # 如果 dry_run 为 True，则打印内容和元数据并返回，不执行后续操作
            print("CONTENT:")
            print(content)
            print("METADATA:")
            print(metadata)
            return

        sub_dir = self.model_card_dir / model_dict["_hf_model_id"]
        sub_dir.mkdir(exist_ok=True)
        # 创建模型卡片的存储子目录，如果不存在则创建

        dest = sub_dir / "README.md"
        dest.open("w").write(content)
        # 将构建好的模型卡片内容写入 README.md 文件中

        for k, v in metadata.items():
            if isinstance(v, datetime.date):
                metadata[k] = datetime.datetime.strftime(v, "%Y-%m-%d")
        # 将元数据中的日期对象转换成字符串形式 "%Y-%m-%d"

        with open(sub_dir / "metadata.json", "w", encoding="utf-8") as writeobj:
            json.dump(metadata, writeobj)
        # 将元数据以 JSON 格式写入 metadata.json 文件中

    def download_lang_info(self):
        Path(LANG_CODE_PATH).parent.mkdir(exist_ok=True)
        # 确保存储语言代码文件的目录存在，如果不存在则创建

        import wget
        # 导入 wget 模块用于下载文件

        if not os.path.exists(ISO_PATH):
            wget.download(ISO_URL, ISO_PATH)
        # 如果 ISO 文件不存在，则使用 wget 下载 ISO 文件

        if not os.path.exists(LANG_CODE_PATH):
            wget.download(LANG_CODE_URL, LANG_CODE_PATH)
        # 如果语言代码文件不存在，则使用 wget 下载语言代码文件
    # 解析模型元数据的方法，给定模型名称、存储库路径和解析方法
    def parse_metadata(self, model_name, repo_path=DEFAULT_MODEL_DIR, method="best"):
        # 构建模型在存储库中的路径
        p = Path(repo_path) / model_name

        # 定义一个函数，从URL中提取文件名（不含扩展名）
        def url_to_name(url):
            return url.split("/")[-1].split(".")[0]

        # 如果模型名称不在模型结果中，则模型结果不明确，使用最新的模型
        if model_name not in self.model_results:
            method = "newest"

        # 如果解析方法为“best”
        if method == "best":
            # 根据下载链接提取模型文件名列表
            results = [url_to_name(model["download"]) for model in self.model_results[model_name]]
            # 在路径p中查找所有以".yml"结尾且名称在results列表中的文件
            ymls = [f for f in os.listdir(p) if f.endswith(".yml") and f[:-4] in results]
            # 根据results列表中模型文件名的顺序排序ymls列表
            ymls.sort(key=lambda x: results.index(x[:-4]))
            # 加载第一个符合条件的YAML文件的元数据
            metadata = yaml.safe_load(open(p / ymls[0]))
            # 更新元数据，添加模型类型信息
            metadata.update(self.model_type_info_from_model_name(ymls[0][:-4]))
        # 如果解析方法为“newest”
        elif method == "newest":
            # 找到所有以".yml"结尾的文件
            ymls = [f for f in os.listdir(p) if f.endswith(".yml")]
            # 按日期排序
            ymls.sort(
                key=lambda x: datetime.datetime.strptime(re.search(r"\d\d\d\d-\d\d?-\d\d?", x).group(), "%Y-%m-%d")
            )
            # 加载最新的YAML文件的元数据
            metadata = yaml.safe_load(open(p / ymls[-1]))
            # 更新元数据，添加模型类型信息
            metadata.update(self.model_type_info_from_model_name(ymls[-1][:-4]))
        else:
            # 抛出未实现的错误，指明不支持的解析方法
            raise NotImplementedError(f"Don't know argument method='{method}' to parse_metadata()")
        
        # 添加模型名称作为元数据的一个字段
        metadata["_name"] = model_name
        # 返回解析得到的元数据
        return metadata
GROUP_MEMBERS = {
    # 三字母代码 -> (语言组/语言名称, {成员...}
    # 如果语言在目标端，成员可以作为目标语言代码使用。
    # 如果语言在源端，它们可以在没有特殊代码的情况下被本地支持。
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
}
    "dra": ("Dravidian languages", {"tam", "kan", "mal", "tel"}),  # 定义键为"dra"的元组，包含语言族名称和语言代码集合
    "ell": ("Modern Greek (1453-)", {"ell"}),  # 定义键为"ell"的元组，包含语言名称和单一的语言代码集合
    "eng": ("English", {"eng"}),  # 定义键为"eng"的元组，包含语言名称和单一的语言代码集合
    "epo": ("Esperanto", {"epo"}),  # 定义键为"epo"的元组，包含语言名称和单一的语言代码集合
    "est": ("Estonian", {"est"}),  # 定义键为"est"的元组，包含语言名称和单一的语言代码集合
    "euq": ("Basque (family)", {"eus"}),  # 定义键为"euq"的元组，包含语言家族名称和单一的语言代码集合
    "eus": ("Basque", {"eus"}),  # 定义键为"eus"的元组，包含语言名称和单一的语言代码集合
    "fin": ("Finnish", {"fin"}),  # 定义键为"fin"的元组，包含语言名称和单一的语言代码集合
    "fiu": (  # 定义键为"fiu"的元组，包含语言家族名称和语言代码集合
        "Finno-Ugrian languages",
        {
            "est",  # 爱沙尼亚语代码
            "fin",  # 芬兰语代码
            "fkv_Latn",  # 科瓦林语的拉丁字母代码
            "hun",  # 匈牙利语代码
            "izh",  # 苏里奥语代码
            "kpv",  # 科米语代码
            "krl",  # 卡累利阿语代码
            "liv_Latn",  # 利沃尼亚语的拉丁字母代码
            "mdf",  # 莫克沙语代码
            "mhr",  # 马里语代码
            "myv",  # 厄尔茨亚语代码
            "sma",  # 南萨米语代码
            "sme",  # 北萨米语代码
            "udm",  # 乌德穆尔特语代码
            "vep",  # 维普尔语代码
            "vro",  # 维兰语代码
        },
    ),
    "fra": ("French", {"fra"}),  # 定义键为"fra"的元组，包含语言名称和单一的语言代码集合
    "gem": (  # 定义键为"gem"的元组，包含语言家族名称和语言代码集合
        "Germanic languages",
        {
            "afr",  # 南非荷兰语代码
            "ang_Latn",  # 古英语的拉丁字母代码
            "dan",  # 丹麦语代码
            "deu",  # 德语代码
            "eng",  # 英语代码
            "enm_Latn",  # 中古英语的拉丁字母代码
            "fao",  # 法罗语代码
            "frr",  # 北弗里西语代码
            "fry",  # 弗里西语代码
            "gos",  # 弗兰克-萨克逊语代码
            "got_Goth",  # 哥特语代码
            "gsw",  # 瑞士德语代码
            "isl",  # 冰岛语代码
            "ksh",  # 科隆语代码
            "ltz",  # 卢森堡语代码
            "nds",  # 下地德语代码
            "nld",  # 荷兰语代码
            "nno",  # 新挪威语代码
            "nob",  # 书面挪威语代码
            "nob_Hebr",  # 书面挪威语的希伯来字母代码
            "non_Latn",  # 古挪威语的非拉丁字母代码
            "pdc",  # 宾夕法尼亚德语代码
            "sco",  # 苏格兰语代码
            "stq",  # 萨特弗里斯兰语代码
            "swe",  # 瑞典语代码
            "swg",  # 沃特兰弗兰克语代码
            "yid",  # 意第绪语代码
        },
    ),
    "gle": ("Irish", {"gle"}),  # 定义键为"gle"的元组，包含语言名称和单一的语言代码集合
    "glg": ("Galician", {"glg"}),  # 定义键为"glg"的元组，包含语言名称和单一的语言代码集合
    "gmq": (  # 定义键为"gmq"的元组，包含语言家族名称和语言代码集合
        "North Germanic languages",
        {
            "dan",  # 丹麦语代码
            "nob",  # 书面挪威语代码
            "nob_Hebr",  # 书面挪威语的希伯来字母代码
            "swe",  # 瑞典语代码
            "isl",  # 冰岛语代码
            "nno",  # 新挪威语代码
            "non_Latn",  # 古挪威语的非拉丁字母代码
            "fao",  # 法罗语代码
        },
    ),
    "gmw": (  # 定义键为"gmw"的元组，包含语言家族名称和语言代码集合
        "West Germanic languages",
        {
            "afr",  # 南非荷兰语代码
            "ang_Latn",  # 古英语的拉丁字母代码
            "deu",  # 德语代码
            "eng",  # 英语代码
            "enm_Latn",  # 中古英语的拉丁字母代码
            "frr",  # 北弗里西语代码
            "fry",  # 弗里西语代码
            "gos",  # 弗兰克-萨克逊语代码
            "gsw",  # 瑞士德语代码
            "ksh",  # 科隆语代码
            "ltz",  # 卢森堡语代码
            "nds",  # 下地德语代码
            "nld",  # 荷兰语代码
            "pdc",  # 宾夕法尼亚德语代码
            "sco",  # 苏格兰语代码
            "stq",  # 萨特弗里斯兰语代码
            "swg",  # 沃特兰弗兰克语代码
            "yid",  # 意第绪语代码
        },
    ),
    "grk": ("Greek languages", {"grc_Grek", "ell"}),  # 定义键为"grk"的元组，包含语言族名称和语言代码集合
    "hbs": ("Serbo-Croatian", {"hrv", "srp_Cyrl", "bos_Latn", "srp_Latn"}),  # 定义键为"hbs"的元组，包含语言名称和语言代码集合
    "heb": ("Hebrew", {"heb"}),  # 定义键为"heb"的元组，包含语言名称和单一的语言代码集合
    "hin": ("Hindi", {"hin"}),  # 定义键为"hin"的元组，包
    "inc": (
        "Indic languages",  # "inc" 键对应的值是一个元组，包含了 "Indic languages" 和一个集合
        {
            "asm",          # 集合中包含 "asm"，代表阿萨姆语
            "awa",          # 集合中包含 "awa"，代表阿瓦德语
            "ben",          # 集合中包含 "ben"，代表孟加拉语
            "bho",          # 集合中包含 "bho"，代表博杰普尔语
            "gom",          # 集合中包含 "gom"，代表孔卡尼语
            "guj",          # 集合中包含 "guj"，代表古吉拉特语
            "hif_Latn",     # 集合中包含 "hif_Latn"，代表斐济印地语（拉丁字母）
            "hin",          # 集合中包含 "hin"，代表印地语
            "mai",          # 集合中包含 "mai"，代表迈蒂利语
            "mar",          # 集合中包含 "mar"，代表马拉地语
            "npi",          # 集合中包含 "npi"，代表尼泊尔文
            "ori",          # 集合中包含 "ori"，代表奥里亚语
            "pan_Guru",     # 集合中包含 "pan_Guru"，代表旁遮普语（古鲁穆基字母）
            "pnb",          # 集合中包含 "pnb"，代表西旁遮普语
            "rom",          # 集合中包含 "rom"，代表罗姆语
            "san_Deva",     # 集合中包含 "san_Deva"，代表梵语（天城文）
            "sin",          # 集合中包含 "sin"，代表僧伽罗语
            "snd_Arab",     # 集合中包含 "snd_Arab"，代表信德语（阿拉伯字母）
            "urd",          # 集合中包含 "urd"，代表乌尔都语
        },
    ),
    "ine": (
        "Indo-European languages",  # 定义键值对 "ine"，表示印欧语系语言，值为元组
        {
            "afr", "afr_Arab", "aln", "ang_Latn", "arg", "asm", "ast", "awa", "bel",  # 定义一个包含多个字符串的集合，表示不同印欧语系语言的标识符
            "bel_Latn", "ben", "bho", "bjn", "bos_Latn", "bre", "bul", "bul_Latn", "cat",
            "ces", "cor", "cos", "csb_Latn", "cym", "dan", "deu", "dsb", "egl", "ell",
            "eng", "enm_Latn", "ext", "fao", "fra", "frm_Latn", "frr", "fry", "gcf_Latn",
            "gla", "gle", "glg", "glv", "gom", "gos", "got_Goth", "grc_Grek", "gsw",
            "guj", "hat", "hif_Latn", "hin", "hrv", "hsb", "hye", "hye_Latn", "ind",
            "isl", "ita", "jdt_Cyrl", "ksh", "kur_Arab", "kur_Latn", "lad", "lad_Latn",
            "lat_Grek", "lat_Latn", "lav", "lij", "lit", "lld_Latn", "lmo", "ltg", "ltz",
            "mai", "mar", "max_Latn", "mfe", "min", "mkd", "mwl", "nds", "nld", "nno",
            "nob", "nob_Hebr", "non_Latn", "npi", "oci", "ori", "orv_Cyrl", "oss",
            "pan_Guru", "pap", "pcd", "pdc", "pes", "pes_Latn", "pes_Thaa", "pms",
            "pnb", "pol", "por", "prg_Latn", "pus", "roh", "rom", "ron", "rue", "rus",
            "rus_Latn", "san_Deva", "scn", "sco", "sgs", "sin", "slv", "snd_Arab",
            "spa", "sqi", "srd", "srp_Cyrl", "srp_Latn", "stq", "swe", "swg", "tgk_Cyrl",
            "tly_Latn", "tmw_Latn", "ukr", "urd", "vec", "wln", "yid", "zlm_Latn",
            "zsm_Latn", "zza"
        },  # 这些字符串代表各种不同印欧语系语言的标识符
    ),
    "isl": ("Icelandic", {"isl"}),  # 定义键值对 "isl"，表示冰岛语，值为包含字符串 "isl" 的集合
    "ita": ("Italian", {"ita"}),  # 定义键值对 "ita"，表示意大利语，值为包含字符串 "ita" 的集合
    "itc": (
        "Italic languages",  # 键 'itc'，代表意大利语族的语言
        {  # 值是一个集合，包含多个字符串，代表具体的语言码
            "arg",  # 阿拉贡语
            "ast",  # 阿斯图里亚斯语
            "bjn",  # 班亚尔语
            "cat",  # 加泰罗尼亚语
            "cos",  # 科西嘉语
            "egl",  # 艾米利安语
            "ext",  # 埃斯特雷马杜拉语
            "fra",  # 法语
            "frm_Latn",  # 中古法语（拉丁字母版）
            "gcf_Latn",  # 古典法罗语（拉丁字母版）
            "glg",  # 加利西亚语
            "hat",  # 海地克里奥尔语
            "ind",  # 印度尼西亚语
            "ita",  # 意大利语
            "lad",  # 罗马尼亚吉普赛语
            "lad_Latn",  # 罗马尼亚吉普赛语（拉丁字母版）
            "lat_Grek",  # 拉丁语（希腊字母版）
            "lat_Latn",  # 拉丁语（拉丁字母版）
            "lij",  # 利古里亚语
            "lld_Latn",  # 皮德蒙特语（拉丁字母版）
            "lmo",  # 伦巴第语
            "max_Latn",  # 马萨伊语（拉丁字母版）
            "mfe",  # 毛里求斯克里奥尔语
            "min",  # 明边语
            "mwl",  # 米兰德语
            "oci",  # 奥克语
            "pap",  # 比道语
            "pcd",  # 皮卡第语
            "pms",  # 皮埃蒙特语
            "por",  # 葡萄牙语
            "roh",  # 罗曼什语
            "ron",  # 罗马尼亚语
            "scn",  # 西西里语
            "spa",  # 西班牙语
            "srd",  # 萨丁语
            "tmw_Latn",  # 提姆西语（拉丁字母版）
            "vec",  # 威尼斯语
            "wln",  # 瓦隆语
            "zlm_Latn",  # 马来语（拉丁字母版）
            "zsm_Latn",  # 马来语（拉丁字母版）
        },
    ),
    "jpn": (  # 键 'jpn'，代表日语
        "Japanese",  # 日语的全称
        {  # 值是一个集合，包含多个字符串，代表具体的日语方言或使用不同字母表的形式
            "jpn",  # 日语
            "jpn_Bopo",  # 日语（注音符号版）
            "jpn_Hang",  # 日语（朝鲜字母版）
            "jpn_Hani",  # 日语（汉字版）
            "jpn_Hira",  # 日语（平假名版）
            "jpn_Kana",  # 日语（假名版）
            "jpn_Latn",  # 日语（拉丁字母版）
            "jpn_Yiii",  # 日语（纳西字母版）
        },
    ),
    "jpx": (  # 键 'jpx'，代表日语的家族
        "Japanese (family)",  # 日语的家族名
        {"jpn"},  # 包含日语
    ),
    "kat": (  # 键 'kat'，代表格鲁吉亚语
        "Georgian",  # 格鲁吉亚语的全称
        {"kat"},  # 包含格鲁吉亚语
    ),
    "kor": (  # 键 'kor'，代表韩语
        "Korean",  # 韩语的全称
        {  # 值是一个集合，包含多个字符串，代表具体的韩语方言或使用不同字母表的形式
            "kor_Hani",  # 韩语（汉字版）
            "kor_Hang",  # 韩语（朝鲜字母版）
            "kor_Latn",  # 韩语（拉丁字母版）
            "kor",  # 韩语
        },
    ),
    "lav": (  # 键 'lav'，代表拉脱维亚语
        "Latvian",  # 拉脱维亚语的全称
        {"lav"},  # 包含拉脱维亚语
    ),
    "lit": (  # 键 'lit'，代表立陶宛语
        "Lithuanian",  # 立陶宛语的全称
        {"lit"},  # 包含立陶宛语
    ),
    "mkd": (  # 键 'mkd'，代表马其顿语
        "Macedonian",  # 马其顿语的全称
        {"mkd"},  # 包含马其顿语
    ),
    "mkh": (  # 键 'mkh'，代表蒙高—湄语族
        "Mon-Khmer languages",  # 蒙高—湄语族的全称
        {  # 值是一个集合，包含多个字符串，代表具体的蒙高—湄语族语言或使用不同字母表的形式
            "vie_Hani",  # 越南语（汉字版）
            "mnw",  # 孟语
            "vie",  # 越南语
            "kha",  # 卡西语
            "khm_Latn",  # 高棉语（拉丁字母版）
            "khm",  # 高棉语
        },
    ),
    "msa": (  # 键 'msa'，代表马来语（宏语言）
        "Malay (macrolanguage)",  # 马来语（宏语言）的全称
        {  # 值是一个集合，包含多个字符串，代表具体的马来语及其变体
            "zsm_Latn",  # 马来语（马来文拉丁字母版）
            "ind",  # 印度尼西亚语
            "max_Latn",  # 马德佩勒马语（拉丁字母版）
            "zlm_Latn",  # 马来语（马来亚文拉丁字母版）
            "min",  # 明边语
        },
    ),
    "nic": (  # 键 'nic'，代表尼日尔—科尔多凡语族
        "Niger-Kordofanian languages",  # 尼日尔—科尔多凡语族的全称
        {  # 值是一个集合，包含多个字符串，代表具体的尼日尔—科尔多凡语族语言
            "bam_Latn",  # 班巴拉语（拉丁字母版）
            "ewe",  # 埃维语
            "fuc",  # 富拉语
            "fuv",  # 富拉语
            "ibo",  # 伊博语
            "kin",  # 卢安达语
    "roa": (
        "Romance languages",
        {  # 这是一个集合，包含多种罗曼语系的语言代码
            "arg",  # 阿拉贡语
            "ast",  # 阿斯图里亚斯语
            "cat",  # 加泰罗尼亚语
            "cos",  # 科西嘉语
            "egl",  # 埃米利亚-罗马涅语
            "ext",  # 埃斯特雷马杜拉语
            "fra",  # 法语
            "frm_Latn",  # 中古法语（拉丁文书写）
            "gcf_Latn",  # 海地克里奥尔法语（拉丁文书写）
            "glg",  # 加利西亚语
            "hat",  # 海地克里奥尔语
            "ind",  # 印尼语
            "ita",  # 意大利语
            "lad",  # 犹太西班牙语
            "lad_Latn",  # 犹太西班牙语（拉丁文书写）
            "lij",  # 利古里亚语
            "lld_Latn",  # 皮德蒙特语（拉丁文书写）
            "lmo",  # 里米尼语
            "max_Latn",  # 里诺罗曼语（拉丁文书写）
            "mfe",  # 毛里求斯克里奥尔语
            "min",  # 明亚克语
            "mwl",  # 米兰达语
            "oci",  # 奥克语
            "pap",  # 帕皮亚门托语
            "pms",  # 皮埃蒙特语
            "por",  # 葡萄牙语
            "roh",  # 罗曼什语
            "ron",  # 罗马尼亚语
            "scn",  # 西西里语
            "spa",  # 西班牙语
            "tmw_Latn",  # 特米纳语（拉丁文书写）
            "vec",  # 威尼斯语
            "wln",  # 瓦隆语
            "zlm_Latn",  # 马来语（拉丁文书写）
            "zsm_Latn",  # 马来语（新加坡拉丁文书写）
        },
    ),
    "ron": ("Romanian", {"ron"}),  # 罗马尼亚语
    "run": ("Rundi", {"run"}),  # 鲁恩迪语
    "rus": ("Russian", {"rus"}),  # 俄语
    "sal": ("Salishan languages", {"shs_Latn"}),  # 沙利什语系
    "sem": (
        "Semitic languages",
        {  # 这是一个集合，包含多种闪米特语系的语言代码
            "acm",  # 中阿拉伯语
            "afb",  # 南布尔语
            "amh",  # 阿姆哈拉语
            "apc",  # 联合阿拉伯语
            "ara",  # 阿拉伯语
            "arq",  # 阿尔及利亚阿拉伯语
            "ary",  # 摩洛哥阿拉伯语
            "arz",  # 埃及阿拉伯语
            "heb",  # 希伯来语
            "mlt",  # 马耳他语
            "tir",  # 提格利尼亚语
        },
    ),
    "sla": (
        "Slavic languages",
        {  # 这是一个集合，包含多种斯拉夫语系的语言代码
            "bel",  # 白俄罗斯语
            "bel_Latn",  # 白俄罗斯语（拉丁文书写）
            "bos_Latn",  # 波斯尼亚语（拉丁文书写）
            "bul",  # 保加利亚语
            "bul_Latn",  # 保加利亚语（拉丁文书写）
            "ces",  # 捷克语
            "csb_Latn",  # 卡舒比亚语（拉丁文书写）
            "dsb",  # 下索布语
            "hrv",  # 克罗地亚语
            "hsb",  # 上索布语
            "mkd",  # 马其顿语
            "orv_Cyrl",  # 古教会斯拉夫语（西里尔文书写）
            "pol",  # 波兰语
            "rue",  # 卢森尼亚语
            "rus",  # 俄语
            "slv",  # 斯洛文尼亚语
            "srp_Cyrl",  # 塞尔维亚语（西里尔文书写）
            "srp_Latn",  # 塞尔维亚语（拉丁文书写）
            "ukr",  # 乌克兰语
        },
    ),
    "slv": ("Slovenian", {"slv"}),  # 斯洛文尼亚语
    "spa": ("Spanish", {"spa"}),  # 西班牙语
    "swe": ("Swedish", {"swe"}),  # 瑞典语
    "taw": ("Tai", {"lao", "tha"}),  # 泰语系
    "tgl": ("Tagalog", {"tgl_Latn"}),  # 菲律宾语
    "tha": ("Thai", {"tha"}),  # 泰语
    "trk": (
        "Turkic languages",
        {  # 这是一个集合，包含多种突厥语系的语言代码
            "aze_Latn",  # 阿塞拜疆语（拉丁文书写）
            "bak",  # 巴什基尔语
            "chv",  # 楚瓦什语
            "crh",  # 克里米亚土耳其语
            "crh_Latn",  # 克里米亚土耳其语（拉丁文书写）
            "kaz_Cyrl",  # 哈萨克语（西里尔文书写）
            "kaz_Latn",  # 哈萨克语（拉丁文书写）
            "kir_Cyrl",  # 柯尔克孜语（西里尔文书写）
            "kjh",  # 喀尔巴阡罗姆语
            "kum",  # 库梅克语
            "ota_Arab",  # 奥斯曼土耳其语（阿拉伯文书写）
            "ota_Latn",  # 奥斯曼土耳其语（拉丁文书写）
            "sah",  # 萨哈语
            "tat",  # 塔塔尔语
            "tat_Arab",  # 塔塔尔语（阿拉伯文书写）
            "tat_Latn",  # 塔塔尔语（拉丁文书写）
            "tuk",  # 土库曼语
            "tuk_Latn",  # 土库曼语（拉丁文书写）
            "tur",  # 土耳其语
            "tyv",  # 图瓦语
            "uig_Arab",  # 维吾尔语（阿拉伯文书写）
            "uig_Cyrl",  # 维吾尔语（西里尔文书写）
            "uzb_Cyrl",
    "zho": (
        "Chinese",
        {  # 定义一个包含多个元素的集合，表示中文相关的语言代码
            "cjy_Hans",  # 简体中文
            "cjy_Hant",  # 繁体中文
            "cmn",       # 普通话（中文）
            "cmn_Bopo",  # 普通话拼音
            "cmn_Hang",  # 普通话汉字
            "cmn_Hani",  # 普通话汉字
            "cmn_Hans",  # 普通话简体字
            "cmn_Hant",  # 普通话繁体字
            "cmn_Hira",  # 普通话平假名
            "cmn_Kana",  # 普通话假名
            "cmn_Latn",  # 普通话拉丁字母
            "cmn_Yiii",  # 普通话伊语
            "gan",       # 赣语
            "hak_Hani",  # 客家话汉字
            "lzh",       # 文言文
            "lzh_Bopo",  # 文言文拼音
            "lzh_Hang",  # 文言文汉字
            "lzh_Hani",  # 文言文汉字
            "lzh_Hans",  # 文言文简体字
            "lzh_Hira",  # 文言文平假名
            "lzh_Kana",  # 文言文假名
            "lzh_Yiii",  # 文言文伊语
            "nan",       # 台湾闽南语
            "nan_Hani",  # 台湾闽南语汉字
            "wuu",       # 吴语
            "wuu_Bopo",  # 吴语拼音
            "wuu_Hani",  # 吴语汉字
            "wuu_Latn",  # 吴语拉丁字母
            "yue",       # 粤语
            "yue_Bopo",  # 粤语拼音
            "yue_Hang",  # 粤语汉字
            "yue_Hani",  # 粤语汉字
            "yue_Hans",  # 粤语简体字
            "yue_Hant",  # 粤语繁体字
            "yue_Hira",  # 粤语平假名
            "yue_Kana",  # 粤语假名
            "zho",       # 中文
            "zho_Hans",  # 中文简体字
            "zho_Hant",  # 中文繁体字
        },
    ),
    "zle": (
        "East Slavic languages",
        {  # 定义一个包含多个元素的集合，表示东斯拉夫语族的语言代码
            "bel",       # 白俄罗斯语
            "orv_Cyrl",  # 古教会斯拉夫语（西里尔字母）
            "bel_Latn",  # 白俄罗斯语拉丁字母
            "rus",       # 俄语
            "ukr",       # 乌克兰语
            "rue",       # 卢森堡文
        },
    ),
    "zls": (
        "South Slavic languages",
        {  # 定义一个包含多个元素的集合，表示南斯拉夫语族的语言代码
            "bos_Latn",  # 波斯尼亚语拉丁字母
            "bul",       # 保加利亚语
            "bul_Latn",  # 保加利亚语拉丁字母
            "hrv",       # 克罗地亚语
            "mkd",       # 马其顿语
            "slv",       # 斯洛文尼亚语
            "srp_Cyrl",  # 塞尔维亚语（西里尔字母）
            "srp_Latn",  # 塞尔维亚语拉丁字母
        },
    ),
    "zlw": (
        "West Slavic languages",
        {  # 定义一个包含多个元素的集合，表示西斯拉夫语族的语言代码
            "csb_Latn",  # 卡舒比语拉丁字母
            "dsb",       # 下索布语
            "hsb",       # 上索布语
            "pol",       # 波兰语
            "ces",       # 捷克语
        },
    ),
}

# l2front_matter 函数：接受一个语言列表，返回一个包含每种语言前缀的字符串
def l2front_matter(langs):
    return "".join(f"- {l}\n" for l in langs)

# dedup 函数：移除列表中的重复项，并保持原有顺序
def dedup(lst):
    """Preservers order"""
    new_lst = []
    for item in lst:
        if not item or item in new_lst:
            continue
        else:
            new_lst.append(item)
    return new_lst

# 程序主入口，用于命令行参数解析和调用相关功能
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数选项：models，要求必须提供，可以多次指定
    parser.add_argument(
        "-m", "--models", action="append", help="<Required> Set flag", required=True, nargs="+", dest="models"
    )
    # 添加命令行参数选项：save_dir，用于指定模型转换后的保存目录，默认为"marian_converted"
    parser.add_argument("-save_dir", "--save_dir", default="marian_converted", help="where to save converted models")
    # 解析命令行参数
    args = parser.parse_args()
    # 创建 TatoebaConverter 的实例，保存目录由命令行参数 save_dir 指定
    resolver = TatoebaConverter(save_dir=args.save_dir)
    # 调用 TatoebaConverter 实例的 convert_models 方法，传入命令行参数 models 的第一个参数作为模型列表
    resolver.convert_models(args.models[0])
```