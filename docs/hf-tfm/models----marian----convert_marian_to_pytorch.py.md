# `.\transformers\models\marian\convert_marian_to_pytorch.py`

```
# 导入所需的模块和库
import argparse                                   # 用于处理命令行参数
import json                                       # 用于处理 JSON 数据
import os                                         # 用于访问操作系统功能
import socket                                     # 用于通过网络进行通信
import time                                       # 用于计时和时间操作
import warnings                                   # 用于忽略警告
from pathlib import Path                          # 用于处理文件路径
from typing import Dict, List, Union              # 用于类型注解
from zipfile import ZipFile                       # 用于处理 ZIP 文件

import numpy as np                                # 用于科学计算
import torch                                      # 用于构建神经网络
from huggingface_hub.hf_api import list_models    # 用于列出模型
from torch import nn                              # 用于构建神经网络
from tqdm import tqdm                            # 用于显示进度条

from transformers import MarianConfig, MarianMTModel, MarianTokenizer   # 用于处理多语种翻译

# 定义一个函数，删除字符串的后缀
def remove_suffix(text: str, suffix: str):
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text

# 定义一个函数，删除字符串的前缀
def remove_prefix(text: str, prefix: str):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text

# 定义一个函数，将编码器层从 Opus 模型转换为 PyTorch 模型
def convert_encoder_layer(opus_dict, layer_prefix: str, converter: dict):
    sd = {}
    for k in opus_dict:
        if not k.startswith(layer_prefix):
            continue
        stripped = remove_prefix(k, layer_prefix)
        v = opus_dict[k].T  # 除了嵌入层之外，其他都需要转置
        sd[converter[stripped]] = torch.tensor(v).squeeze()
    return sd

# 定义一个函数，加载编码器/解码器的每一层的参数
def load_layers_(layer_lst: nn.ModuleList, opus_state: dict, converter, is_decoder=False):
    for i, layer in enumerate(layer_lst):
        layer_tag = f"decoder_l{i + 1}_" if is_decoder else f"encoder_l{i + 1}_"
        sd = convert_encoder_layer(opus_state, layer_tag, converter)
        layer.load_state_dict(sd, strict=False)

# 定义一个函数，查找支持指定源语言和目标语言的预训练模型
def find_pretrained_model(src_lang: str, tgt_lang: str) -> List[str]:
    prefix = "Helsinki-NLP/opus-mt-"
    model_list = list_models()
    model_ids = [x.modelId for x in model_list if x.modelId.startswith("Helsinki-NLP")]
    src_and_targ = [
        remove_prefix(m, prefix).lower().split("-") for m in model_ids if "+" not in m
    ]  # + 不能加载
    matching = [f"{prefix}{a}-{b}" for (a, b) in src_and_targ if src_lang in a and tgt_lang in b]
    return matching

# 定义一个函数，添加嵌入层的条目
def add_emb_entries(wemb, final_bias, n_special_tokens=1):
    vsize, d_model = wemb.shape
    embs_to_add = np.zeros((n_special_tokens, d_model))
    new_embs = np.concatenate([wemb, embs_to_add])
    bias_to_add = np.zeros((n_special_tokens, 1))
    new_bias = np.concatenate((final_bias, bias_to_add), axis=1)
    return new_embs, new_bias

# 定义一个函数，将 YAML 字符串转换为对应的类型
def _cast_yaml_str(v):
    bool_dct = {"true": True, "false": False}
    # 如果输入值不是字符串，则直接返回该值
    if not isinstance(v, str):
        return v
    # 如果输入值在布尔字典中，则返回对应的布尔值
    elif v in bool_dct:
        return bool_dct[v]
    # 尝试将输入值转换为整数，如果成功则返回转换后的整数
    try:
        return int(v)
    # 如果转换为整数时出现异常（TypeError 或 ValueError），则返回原始输入值
    except (TypeError, ValueError):
        return v
# 转换原始配置字典中的值为字典类型
def cast_marian_config(raw_cfg: Dict[str, str]) -> Dict:
    return {k: _cast_yaml_str(v) for k, v in raw_cfg.items()}


# 定义用于访问配置的特殊键
CONFIG_KEY = "special:model.yml"


# 从给定的状态字典中加载配置
def load_config_from_state_dict(opus_dict):
    # 导入 yaml 模块
    import yaml
    # 将 opus_dict[CONFIG_KEY] 中的字符转换为字符串
    cfg_str = "".join([chr(x) for x in opus_dict[CONFIG_KEY]])
    # 解析 YAML 字符串
    yaml_cfg = yaml.load(cfg_str[:-1], Loader=yaml.BaseLoader)
    return cast_marian_config(yaml_cfg)


# 查找指定目录下的模型文件
def find_model_file(dest_dir):  # this one better
    # 列出目录下所有的 .npz 文件
    model_files = list(Path(dest_dir).glob("*.npz"))
    # 如果文件数量不等于 1，抛出值错误
    if len(model_files) != 1:
        raise ValueError(f"Found more than one model file: {model_files}")
    # 返回找到的模型文件
    model_file = model_files[0]
    return model_file


# 定义一些语言组合和与之对应的组名
ROM_GROUP = (
    "fr+fr_BE+fr_CA+fr_FR+wa+frp+oc+ca+rm+lld+fur+lij+lmo+es+es_AR+es_CL+es_CO+es_CR+es_DO+es_EC+es_ES+es_GT"
    "+es_HN+es_MX+es_NI+es_PA+es_PE+es_PR+es_SV+es_UY+es_VE+pt+pt_br+pt_BR+pt_PT+gl+lad+an+mwl+it+it_IT+co"
    "+nap+scn+vec+sc+ro+la"
)
# 定义不同语言组合的名称
GROUPS = [
    ("cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh", "ZH"),
    (ROM_GROUP, "ROMANCE"),
    ("de+nl+fy+af+da+fo+is+no+nb+nn+sv", "NORTH_EU"),
    ("da+fo+is+no+nb+nn+sv", "SCANDINAVIA"),
    ("se+sma+smj+smn+sms", "SAMI"),
    ("nb_NO+nb+nn_NO+nn+nog+no_nb+no", "NORWAY"),
    ("ga+cy+br+gd+kw+gv", "CELTIC"),  # https://en.wikipedia.org/wiki/Insular_Celtic_languages
]
# 定义不同组合到 opus 名称的映射关系
GROUP_TO_OPUS_NAME = {
    "opus-mt-ZH-de": "cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-de",
    "opus-mt-ZH-fi": "cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-fi",
    "opus-mt-ZH-sv": "cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-sv",
    "opus-mt-SCANDINAVIA-SCANDINAVIA": "da+fo+is+no+nb+nn+sv-da+fo+is+no+nb+nn+sv",
    "opus-mt-NORTH_EU-NORTH_EU": "de+nl+fy+af+da+fo+is+no+nb+nn+sv-de+nl+fy+af+da+fo+is+no+nb+nn+sv",
    "opus-mt-de-ZH": "de-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh",
    "opus-mt-en_el_es_fi-en_el_es_fi": "en+el+es+fi-en+el+es+fi",
    "opus-mt-en-ROMANCE": (
        "en-fr+fr_BE+fr_CA+fr_FR+wa+frp+oc+ca+rm+lld+fur+lij+lmo+es+es_AR+es_CL+es_CO+es_CR+es_DO"
        "+es_EC+es_ES+es_GT+es_HN+es_MX+es_NI+es_PA+es_PE+es_PR+es_SV+es_UY+es_VE+pt+pt_br+pt_BR"
        "+pt_PT+gl+lad+an+mwl+it+it_IT+co+nap+scn+vec+sc+ro+la"
    ),
    "opus-mt-en-CELTIC": "en-ga+cy+br+gd+kw+gv",
    "opus-mt-es-NORWAY": "es-nb_NO+nb+nn_NO+nn+nog+no_nb+no",
    "opus-mt-fi_nb_no_nn_ru_sv_en-SAMI": "fi+nb+no+nn+ru+sv+en-se+sma+smj+smn+sms",
    "opus-mt-fi-ZH": "fi-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh",
    "opus-mt-fi-NORWAY": "fi-nb_NO+nb+nn_NO+nn+nog+no_nb+no",
    "opus-mt-ROMANCE-en": (
        "fr+fr_BE+fr_CA+fr_FR+wa+frp+oc+ca+rm+lld+fur+lij+lmo+es+es_AR+es_CL+es_CO+es_CR+es_DO"
        "+es_EC+es_ES+es_GT+es_HN+es_MX+es_NI+es_PA+es_PE+es_PR+es_SV+es_UY+es_VE+pt+pt_br+pt_BR"
        "+pt_PT+gl+lad+an+mwl+it+it_IT+co+nap+scn+vec+sc+ro+la-en"
    ),
    # 指定了名称为"opus-mt-CELTIC-en"的语言对，对应的目标语言包括爱尔兰语、威尔士语、布列塔尼语、苏格兰盖尔语、凯尔特语、曼岛盖尔语以及英语
    "opus-mt-CELTIC-en": "ga+cy+br+gd+kw+gv-en",
    # 指定了名称为"opus-mt-sv-ZH"的语言对，对应的目标语言包括瑞典语、普通话、粤语、泽西法语、中文简体、中国大陆中文、香港中文、台湾中文、中国台湾中文、粤语、简体中文、繁体中文、中文
    "opus-mt-sv-ZH": "sv-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh",
    # 指定了名称为"opus-mt-sv-NORWAY"的语言对，对应的目标语言包括瑞典语、挪威语（诺尔斯克）、挪威语（诺斯克）(尼诺斯克)、挪威语（诺斯克）(尼诺斯克)、挪威语（诺尔斯克）+挪威语（诺尔斯克）+挪威语（尼诺尔斯克）+挪威语（尼诺尔斯克）+挪威语（诺尔斯克）+挪威语（诺尔斯克）
    "opus-mt-sv-NORWAY": "sv-nb_NO+nb+nn_NO+nn+nog+no_nb+no",
}
这是代码的结尾，为了保持代码结构完整，注释此行

OPUS_GITHUB_URL = "https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/models/"
定义一个变量OPUS_GITHUB_URL，用于存储GitHub的链接

ORG_NAME = "Helsinki-NLP/"
定义一个变量ORG_NAME，用于存储组织名称"Helsinki-NLP/"

def convert_opus_name_to_hf_name(x):
    """For OPUS-MT-Train/ DEPRECATED"""
    定义一个名为convert_opus_name_to_hf_name的函数，将OPUS-MT-Train名称转换为hf_name

    for substr, grp_name in GROUPS:
        x = x.replace(substr, grp_name)
    用循环遍历GROUPS列表中的元素，使用replace方法将x中的substr替换为grp_name，并重新赋值给x

    return x.replace("+", "_")
    使用replace方法将x中的加号"+"替换为下划线"_"，并返回新的字符串x

def convert_hf_name_to_opus_name(hf_model_name):
    """
    Relies on the assumption that there are no language codes like pt_br in models that are not in GROUP_TO_OPUS_NAME.
    """
    定义一个名为convert_hf_name_to_opus_name的函数，将hf_model_name转换为opus_name

    hf_model_name = remove_prefix(hf_model_name, ORG_NAME)
    使用remove_prefix函数将hf_model_name中的ORG_NAME前缀移除，并重新赋值给hf_model_name

    if hf_model_name in GROUP_TO_OPUS_NAME:
        判断hf_model_name是否在GROUP_TO_OPUS_NAME中
        opus_w_prefix = GROUP_TO_OPUS_NAME[hf_model_name]
        如果在其中，从GROUP_TO_OPUS_NAME字典中获取hf_model_name对应的值，赋值给opus_w_prefix
    else:
        opus_w_prefix = hf_model_name.replace("_", "+")
        如果不在其中，使用replace方法将hf_model_name中的下划线"_"替换为加号"+"，并重新赋值给opus_w_prefix

    return remove_prefix(opus_w_prefix, "opus-mt-")
    使用remove_prefix函数将opus_w_prefix中的"opus-mt-"前缀移除，并返回移除后的字串

def get_system_metadata(repo_root):
    import git

    return {
        "helsinki_git_sha": git.Repo(path=repo_root, search_parent_directories=True).head.object.hexsha,
        "transformers_git_sha": git.Repo(path=".", search_parent_directories=True).head.object.hexsha,
        "port_machine": socket.gethostname(),
        "port_time": time.strftime("%Y-%m-%d-%H:%M"),
    }
定义一个名为get_system_metadata的函数，用于获取系统元数据，包括'helsinki_git_sha', 'transformers_git_sha', 'port_machine'和'port_time'

# docstyle-ignore
FRONT_MATTER_TEMPLATE = """---
language:
{}
tags:
- translation

license: apache-2.0
---
"""
定义一个名为FRONT_MATTER_TEMPLATE的字符串模板，包含了markdown文档前部分的metadata部分的格式

DEFAULT_REPO = "Tatoeba-Challenge"
定义一个名为DEFAULT_REPO的变量，用于存储默认的仓库名"Tatoeba-Challenge"

DEFAULT_MODEL_DIR = os.path.join(DEFAULT_REPO, "models")
定义一个名为DEFAULT_MODEL_DIR的变量，用于存储DEFAULT_REPO和"models"拼接后的字符串

def write_model_card(
    hf_model_name: str,
    repo_root=DEFAULT_REPO,
    save_dir=Path("marian_converted"),
    dry_run=False,
    extra_metadata={},
) -> str:
    """
    Copy the most recent model's readme section from opus, and add metadata. upload command: aws s3 sync model_card_dir
    s3://models.huggingface.co/bert/Helsinki-NLP/ --dryrun
    """
    定义一个名为write_model_card的函数，用于将最新模型的readme部分从opus中复制，并添加metadata

    import pandas as pd

    hf_model_name = remove_prefix(hf_model_name, ORG_NAME)
    使用remove_prefix函数将hf_model_name中的ORG_NAME前缀移除，并重新赋值给hf_model_name

    opus_name: str = convert_hf_name_to_opus_name(hf_model_name)
    将hf_model_name传递给convert_hf_name_to_opus_name函数，并将返回值赋值给opus_name

    if repo_root not in ("OPUS-MT-train", "Tatoeba-Challenge"):
        如果repo_root不是"OPUS-MT-train"和"Tatoeba-Challenge"中的任意一个
        raise ValueError(f"Repos root is {repo_root}. Expected either OPUS-MT-train or Tatoeba-Challenge")
        抛出ValueError异常，提示仓库根目录不是"OPUS-MT-train"或"Tatoeba-Challenge"

    opus_readme_path = Path(repo_root).joinpath("models", opus_name, "README.md")
    根据repo_root、opus_name和"README.md"生成opus_readme_path路径对象

    if not (opus_readme_path.exists()):
        如果opus_readme_path路径不存在
        raise ValueError(f"Readme file {opus_readme_path} not found")
        抛出ValueError异常，提示找不到readme文件

    opus_src, opus_tgt = [x.split("+") for x in opus_name.split("-")]
    将opus_name按"-"分割，然后按"+"分割，得到opus_src和opus_tgt列表

    readme_url = f"https://github.com/Helsinki-NLP/{repo_root}/tree/master/models/{opus_name}/README.md"
    根据repo_root、opus_name和"README.md"生成readme_url

    s, t = ",".join(opus_src), ",".join(opus_tgt)
    将opus_src和opus_tgt列表中的元素使用","连接成字符串，并分别赋值给s和t

    metadata = {
        "hf_name": hf_model_name,
        "source_languages": s,
        "target_languages": t,
        "opus_readme_url": readme_url,
        "original_repo": repo_root,
        "tags": ["translation"],
    }
    构建metadata字典，包含"hf_name"、"source_languages"、"target_languages"、"opus_readme_url"、"original_repo"和"tags"字段

    metadata.update(extra_metadata)
    将extra_metadata中的键值对更新到metadata字典中

    metadata.update(get_system_metadata(repo_root))
    获取repo_root的系统元数据，并将其更新到metadata字典中

    # combine with opus markdown
    将opus_readme_path的内容读取出来，然后与metadata进行组合并返回该字符串


完成。
    # 生成额外的 Markdown 内容，包括模型名称、源组和目标组信息以及相关链接
    extra_markdown = (
        f"### {hf_model_name}\n\n* source group: {metadata['src_name']} \n* target group: "
        f"{metadata['tgt_name']} \n*  OPUS readme: [{opus_name}]({readme_url})\n"
    )

    # 读取 OPUS README 文件的内容
    content = opus_readme_path.open().read()
    # 获取 README 中最低级别的头部（即最近的模型）
    content = content.split("\n# ")[-1]
    # 根据特定的标记符号对内容进行分割
    splat = content.split("*")[2:]
    # 打印分割后的内容的第四个元素
    print(splat[3])
    # 重新组合分割后的内容
    content = "*".join(splat)
    # 拼接生成新的 Markdown 内容
    content = (
        FRONT_MATTER_TEMPLATE.format(metadata["src_alpha2"])
        + extra_markdown
        + "\n* "
        + content.replace("download", "download original weights")
    )

    # 构建包含元数据项的字符串，每个项包括键和值
    items = "\n\n".join([f"- {k}: {v}" for k, v in metadata.items()])
    # 生成系统信息的Markdown内容
    sec3 = "\n### System Info: \n" + items
    # 将系统信息添加到内容中
    content += sec3
    # 如果是 dry run 模式，返回生成的内容和元数据
    if dry_run:
        return content, metadata
    # 创建子目录用于保存文件
    sub_dir = save_dir / f"opus-mt-{hf_model_name}"
    sub_dir.mkdir(exist_ok=True)
    # 创建并写入 README 文件
    dest = sub_dir / "README.md"
    dest.open("w").write(content)
    # 将 metadata 转换成 JSON 格式并写入文件
    pd.Series(metadata).to_json(sub_dir / "metadata.json")

    # 如果是 dry run 模式，返回生成的内容和元数据
    # if dry_run:
    return content, metadata
# 创建一个函数用于构建模型注册表，指定默认的仓库路径为"Opus-MT-train/models"
def make_registry(repo_path="Opus-MT-train/models"):
    # 检查指定路径下是否存在特定的 README.md 文件，若不存在则引发异常
    if not (Path(repo_path) / "fr-en" / "README.md").exists():
        raise ValueError(
            f"repo_path:{repo_path} does not exist: "
            "You must run: git clone git@github.com:Helsinki-NLP/Opus-MT-train.git before calling."
        )
    # 初始化结果字典
    results = {}
    # 遍历指定路径下的所有文件和目录
    for p in Path(repo_path).iterdir():
        # 计算文件名中短横线的数量
        n_dash = p.name.count("-")
        # 如果短横线数量为零，则跳过当前文件或目录
        if n_dash == 0:
            continue
        else:
            # 读取当前文件的 README.md 内容为列表
            lns = list(open(p / "README.md").readlines())
            # 解析 README.md 内容，将结果存储到结果字典中
            results[p.name] = _parse_readme(lns)
    # 返回结果字典的键值对列表，其中包含了模型名、预处理信息、下载链接以及测试集链接
    return [(k, v["pre-processing"], v["download"], v["download"][:-4] + ".test.txt") for k, v in results.items()]


# 创建一个函数用于转换所有的 SentencePiece 模型
def convert_all_sentencepiece_models(model_list=None, repo_path=None, dest_dir=Path("marian_converted")):
    """Requires 300GB"""
    # 设置保存路径
    save_dir = Path("marian_ckpt")
    # 创建目标目录
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)
    # 初始化保存路径列表
    save_paths = []
    # 如果模型列表为空，则调用 make_registry 函数生成模型列表
    if model_list is None:
        model_list: list = make_registry(repo_path=repo_path)
    # 遍历模型列表
    for k, prepro, download, test_set_url in tqdm(model_list):
        # 如果预处理信息中不包含 "SentencePiece"，则跳过当前模型
        if "SentencePiece" not in prepro:  # dont convert BPE models.
            continue
        # 如果保存目录中不存在当前模型的目录，则下载并解压当前模型
        if not os.path.exists(save_dir / k):
            download_and_unzip(download, save_dir / k)
        # 将 Opus 模型名转换为 Hugging Face 模型名
        pair_name = convert_opus_name_to_hf_name(k)
        # 将当前模型转换为 Hugging Face 格式
        convert(save_dir / k, dest_dir / f"opus-mt-{pair_name}")
        # 将转换后的模型路径添加到保存路径列表中
        save_paths.append(dest_dir / f"opus-mt-{pair_name}")
    # 返回转换后的模型保存路径列表
    return save_paths


# 创建一个函数用于将函数应用到列表的每个元素，并返回结果列表
def lmap(f, x) -> List:
    return list(map(f, x))


# 创建一个函数用于获取测试集数据
def fetch_test_set(test_set_url):
    # 导入 wget 模块
    import wget
    # 使用 wget 下载测试集文件
    fname = wget.download(test_set_url, "opus_test.txt")
    # 读取下载的测试集文件内容为列表
    lns = Path(fname).open().readlines()
    # 提取源语言句子、目标语言句子和参考译文，并以列表形式返回
    src = lmap(str.strip, lns[::4])
    gold = lmap(str.strip, lns[1::4])
    mar_model = lmap(str.strip, lns[2::4])
    # 如果源语言句子、目标语言句子和参考译文长度不一致，则引发异常
    if not (len(gold) == len(mar_model) == len(src)):
        raise ValueError(f"Gold, marian and source lengths {len(gold)}, {len(mar_model)}, {len(src)} mismatched")
    # 删除下载的测试集文件
    os.remove(fname)
    # 返回源语言句子、目标语言句子和参考译文列表
    return src, mar_model, gold


# 创建一个函数用于转换指定目录下的所有模型
def convert_whole_dir(path=Path("marian_ckpt/")):
    # 遍历指定目录下的所有子目录
    for subdir in tqdm(list(path.ls())):
        # 设置目标目录路径
        dest_dir = f"marian_converted/{subdir.name}"
        # 如果目标目录中不存在模型文件，则进行模型转换
        if (dest_dir / "pytorch_model.bin").exists():
            continue
        convert(source_dir, dest_dir)


# 创建一个函数用于解析 README.md 内容，提取链接和元数据
def _parse_readme(lns):
    """Get link and metadata from opus model card equivalent."""
    # 初始化子结果字典
    subres = {}
    # 遍历 README.md 中的每一行
    for ln in [x.strip() for x in lns]:
        # 如果当前行不以 "*" 开头，则跳过
        if not ln.startswith("*"):
            continue
        # 去除行首的 "*" 符号并去除首尾空格
        ln = ln[1:].strip()

        # 查找包含特定关键字的行，并提取关键字和对应值
        for k in ["download", "dataset", "models", "model", "pre-processing"]:
            if ln.startswith(k):
                break
        else:
            continue
        if k in ["dataset", "model", "pre-processing"]:
            splat = ln.split(":")
            _, v = splat
            subres[k] = v
        elif k == "download":
            v = ln.split("(")[-1][:-1]
            subres[k] = v
    # 返回解析得到的子结果字典
    return subres


# 创建一个函数用于保存分词器配置文件
def save_tokenizer_config(dest_dir: Path, separate_vocabs=False):
    # 根据目标目录的名称分割字符串，并将结果存储在列表中
    dname = dest_dir.name.split("-")
    # 创建一个字典，包含目标语言、源语言和是否分开词汇表的键值对
    dct = {"target_lang": dname[-1], "source_lang": "-".join(dname[:-1]), "separate_vocabs": separate_vocabs}
    # 将字典保存为 JSON 格式，并写入目标目录下的 "tokenizer_config.json" 文件
    save_json(dct, dest_dir / "tokenizer_config.json")
# 将特殊标记添加到词汇表中
def add_to_vocab_(vocab: Dict[str, int], special_tokens: List[str]):
    # 确定新标记的起始值
    start = max(vocab.values()) + 1
    added = 0
    # 遍历每个特殊标记
    for tok in special_tokens:
        if tok in vocab:
            continue
        # 将新的特殊标记添加到词汇表中
        vocab[tok] = start + added
        added += 1
    return added

# 查找模型目录中的词汇文件
def find_vocab_file(model_dir):
    return list(model_dir.glob("*vocab.yml"))[0]

# 查找模型目录中源语言词汇文件
def find_src_vocab_file(model_dir):
    return list(model_dir.glob("*src.vocab.yml"))[0]

# 查找模型目录中目标语言词汇文件
def find_tgt_vocab_file(model_dir):
    return list(model_dir.glob("*trg.vocab.yml"))[0]

# 将特殊标记添加到词汇表中
def add_special_tokens_to_vocab(model_dir: Path, separate_vocab=False) -> None:
    # 如果使用独立的词汇表
    if separate_vocab:
        # 加载源语言词汇表
        vocab = load_yaml(find_src_vocab_file(model_dir))
        vocab = {k: int(v) for k, v in vocab.items()}
        # 将新的特殊标记添加到词汇表中
        num_added = add_to_vocab_(vocab, ["<pad>"])
        # 保存新的词汇表
        save_json(vocab, model_dir / "vocab.json")

        # 加载目标语言词汇表
        vocab = load_yaml(find_tgt_vocab_file(model_dir))
        vocab = {k: int(v) for k, v in vocab.items()}
        # 将新的特殊标记添加到词汇表中
        num_added = add_to_vocab_(vocab, ["<pad>"])
        # 保存新的目标词汇表
        save_json(vocab, model_dir / "target_vocab.json")
        # 保存分词器配置
        save_tokenizer_config(model_dir, separate_vocabs=separate_vocab)
    # 如果不使用独立的词汇表
    else:
        # 加载词汇表
        vocab = load_yaml(find_vocab_file(model_dir))
        vocab = {k: int(v) for k, v in vocab.items()}
        # 将新的特殊标记添加到词汇表中
        num_added = add_to_vocab_(vocab, ["<pad>"])
        # 输出添加了多少个新的标记到词汇表中
        print(f"added {num_added} tokens to vocab")
        # 保存词汇表
        save_json(vocab, model_dir / "vocab.json")
        # 保存分词器配置
        save_tokenizer_config(model_dir)

# 检查两个配置项是否相等
def check_equal(marian_cfg, k1, k2):
    v1, v2 = marian_cfg[k1], marian_cfg[k2]
    if v1 != v2:
        raise ValueError(f"hparams {k1},{k2} differ: {v1} != {v2}")

# 检查Marian配置的假设
def check_marian_cfg_assumptions(marian_cfg):
    assumed_settings = {
        # 一系列Marian配置的假设
    }
    # 对于每个假设，检查实际配置值与假设值是否相等
    for k, v in assumed_settings.items():
        actual = marian_cfg[k]
        if actual != v:
            raise ValueError(f"Unexpected config value for {k} expected {v} got {actual}")

BIAS_KEY = "decoder_ff_logit_out_b"
BART_CONVERTER = {  # for each encoder and decoder layer
    # BART模型转换器
    # 对于每个编码器和解码器层
    # 将每个参数的旧名称映射为新名称
    "self_Wq": "self_attn.q_proj.weight",
    "self_Wk": "self_attn.k_proj.weight",
    "self_Wv": "self_attn.v_proj.weight",
    "self_Wo": "self_attn.out_proj.weight",
    "self_bq": "self_attn.q_proj.bias",
    "self_bk": "self_attn.k_proj.bias",
    "self_bv": "self_attn.v_proj.bias",
    "self_bo": "self_attn.out_proj.bias",
    "self_Wo_ln_scale": "self_attn_layer_norm.weight",
    "self_Wo_ln_bias": "self_attn_layer_norm.bias",
    "ffn_W1": "fc1.weight",
    "ffn_b1": "fc1.bias",
    # 定义了一系列键值对，表示模型中各个部分的权重和偏置的对应关系
    "ffn_W2": "fc2.weight",
    "ffn_b2": "fc2.bias",
    "ffn_ffn_ln_scale": "final_layer_norm.weight",
    "ffn_ffn_ln_bias": "final_layer_norm.bias",
    # 下面是Decoder Cross Attention部分的权重和偏置
    "context_Wk": "encoder_attn.k_proj.weight",
    "context_Wo": "encoder_attn.out_proj.weight",
    "context_Wq": "encoder_attn.q_proj.weight",
    "context_Wv": "encoder_attn.v_proj.weight",
    "context_bk": "encoder_attn.k_proj.bias",
    "context_bo": "encoder_attn.out_proj.bias",
    "context_bq": "encoder_attn.q_proj.bias",
    "context_bv": "encoder_attn.v_proj.bias",
    "context_Wo_ln_scale": "encoder_attn_layer_norm.weight",
    "context_Wo_ln_bias": "encoder_attn_layer_norm.bias",
# 定义 OpusState 类，用于管理 Opus 模型的状态信息
class OpusState:
    # 检查各层的条目数量是否符合预期
    def _check_layer_entries(self):
        # 获取编码器层的键列表
        self.encoder_l1 = self.sub_keys("encoder_l1")
        # 获取解码器第一层的键列表
        self.decoder_l1 = self.sub_keys("decoder_l1")
        # 获取解码器第二层的键列表
        self.decoder_l2 = self.sub_keys("decoder_l2")
        # 检查编码器第一层键的数量是否为 16
        if len(self.encoder_l1) != 16:
            # 发出警告，显示实际的键数量
            warnings.warn(f"Expected 16 keys for each encoder layer, got {len(self.encoder_l1)}")
        # 检查解码器第一层键的数量是否为 26
        if len(self.decoder_l1) != 26:
            # 发出警告，显示实际的键数量
            warnings.warn(f"Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}")
        # 检查解码器第二层键的数量是否为 26
        if len(self.decoder_l2) != 26:
            # 发出警告，显示实际的键数量（这里使用了错误的键数量，应为 len(self.decoder_l2)）
            warnings.warn(f"Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}")

    # 计算额外的键
    @property
    def extra_keys(self):
        # 初始化额外键列表
        extra = []
        # 遍历状态键列表
        for k in self.state_keys:
            # 如果键以 "encoder_l" 或 "decoder_l" 开头，或者是预定义的特殊键，则跳过
            if (
                k.startswith("encoder_l")
                or k.startswith("decoder_l")
                or k in [CONFIG_KEY, "Wemb", "encoder_Wemb", "decoder_Wemb", "Wpos", "decoder_ff_logit_out_b"]
            ):
                continue
            # 否则将键添加到额外键列表中
            else:
                extra.append(k)
        # 返回额外键列表
        return extra

    # 获取指定层的键列表
    def sub_keys(self, layer_prefix):
        # 使用给定前缀获取子键列表
        return [remove_prefix(k, layer_prefix) for k in self.state_dict if k.startswith(layer_prefix)]

    # 载入分词器
    def load_tokenizer(self):
        # 将特殊标记添加到词汇表中，根据需要共享编码器和解码器的嵌入
        add_special_tokens_to_vocab(self.source_dir, not self.share_encoder_decoder_embeddings)
        # 从预训练模型源目录加载 MarianTokenizer 对象，并返回
        return MarianTokenizer.from_pretrained(str(self.source_dir))
    # 加载MarianMTModel模型，返回加载后的模型对象
    def load_marian_model(self) -> MarianMTModel:
        # 获取模型状态字典和配置信息
        state_dict, cfg = self.state_dict, self.hf_config

        # 如果配置中static_position_embeddings不为True，则引发错误
        if not cfg.static_position_embeddings:
            raise ValueError("config.static_position_embeddings should be True")
        # 创建MarianMTModel模型对象
        model = MarianMTModel(cfg)

        # 如果配置中存在hidden_size，则引发错误
        if "hidden_size" in cfg.to_dict():
            raise ValueError("hidden_size is in config")
        # 加载编码器层参数
        load_layers_(
            model.model.encoder.layers,
            state_dict,
            BART_CONVERTER,
        )
        # 加载解码器层参数
        load_layers_(model.model.decoder.layers, state_dict, BART_CONVERTER, is_decoder=True)

        # 处理与层无关的张量
        if self.cfg["tied-embeddings-src"]:
            # 创建编码器嵌入权重张量和偏置张量
            wemb_tensor = nn.Parameter(torch.FloatTensor(self.wemb))
            bias_tensor = nn.Parameter(torch.FloatTensor(self.final_bias))
            # 设置共享的嵌入权重
            model.model.shared.weight = wemb_tensor
            model.model.encoder.embed_tokens = model.model.decoder.embed_tokens = model.model.shared
        else:
            # 创建编码器嵌入权重张量
            wemb_tensor = nn.Parameter(torch.FloatTensor(self.wemb))
            model.model.encoder.embed_tokens.weight = wemb_tensor

            # 创建解码器嵌入权重张量和偏置张量
            decoder_wemb_tensor = nn.Parameter(torch.FloatTensor(self.dec_wemb))
            bias_tensor = nn.Parameter(torch.FloatTensor(self.final_bias))
            # 设置解码器嵌入权重
            model.model.decoder.embed_tokens.weight = decoder_wemb_tensor

        # 设置最终的logits偏置张量
        model.final_logits_bias = bias_tensor

        # 如果状态字典中存在Wpos，则设置编码器和解码器的位置嵌入权重
        if "Wpos" in state_dict:
            print("Unexpected: got Wpos")
            wpos_tensor = torch.tensor(state_dict["Wpos"])
            model.model.encoder.embed_positions.weight = wpos_tensor
            model.model.decoder.embed_positions.weight = wpos_tensor

        # 如果配置中normalize_embedding为True，则需要实现对嵌入层的归一化
        if cfg.normalize_embedding:
            if "encoder_emb_ln_scale_pre" not in state_dict:
                raise ValueError("encoder_emb_ln_scale_pre is not in state dictionary")
            raise NotImplementedError("Need to convert layernorm_embedding")

        # 如果存在额外的键，则引发错误
        if self.extra_keys:
            raise ValueError(f"Failed to convert {self.extra_keys}")

        # 如果模型输入嵌入的填充索引与指定的填充标记不匹配，则引发错误
        if model.get_input_embeddings().padding_idx != self.pad_token_id:
            raise ValueError(
                f"Padding tokens {model.get_input_embeddings().padding_idx} and {self.pad_token_id} mismatched"
            )
        # 返回加载后的模型
        return model
# 定义函数download_and_unzip，用于从给定URL下载文件并解压到目标目录
def download_and_unzip(url, dest_dir):
    # 尝试导入wget模块，若导入失败则抛出ImportError异常提示安装wget模块
    try:
        import wget
    except ImportError:
        raise ImportError("you must pip install wget")

    # 使用wget下载文件，并将文件名保存到filename变量中
    filename = wget.download(url)
    # 调用unzip函数解压下载的文件到目标目录
    unzip(filename, dest_dir)
    # 删除下载的文件
    os.remove(filename)


# 定义函数convert，用于转换模型
def convert(source_dir: Path, dest_dir):
    # 将目标目录路径转换为Path对象
    dest_dir = Path(dest_dir)
    # 如果目标目录不存在，则创建目标目录
    dest_dir.mkdir(exist_ok=True)

    # 创建OpusState对象，传入源目录路径
    opus_state = OpusState(source_dir)

    # 保存tokenizer到目标目录
    opus_state.tokenizer.save_pretrained(dest_dir)

    # 加载Marian模型
    model = opus_state.load_marian_model()
    # 将模型转换为半精度浮点数表示
    model = model.half()
    # 保存模型到目标目录
    model.save_pretrained(dest_dir)
    # 从目标目录加载模型进行检查
    model.from_pretrained(dest_dir)  # sanity check


# 定义函数load_yaml，用于加载YAML文件
def load_yaml(path):
    # 导入yaml模块
    import yaml

    # 打开指定路径的文件，并使用yaml模块加载其中内容，返回解析后的内容
    with open(path) as f:
        return yaml.load(f, Loader=yaml.BaseLoader)


# 定义函数save_json，用于保存JSON格式的内容到文件
def save_json(content: Union[Dict, List], path: str) -> None:
    # 打开指定路径的文件，将内容以JSON格式写入文件
    with open(path, "w") as f:
        json.dump(content, f)


# 定义函数unzip，用于解压ZIP文件到目标目录
def unzip(zip_path: str, dest_dir: str) -> None:
    # 使用ZipFile对象打开ZIP文件
    with ZipFile(zip_path, "r") as zipObj:
        # 解压ZIP文件中的所有内容到目标目录
        zipObj.extractall(dest_dir)


# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    """
    Tatoeba conversion instructions in scripts/tatoeba/README.md
    """
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument("--src", type=str, help="path to marian model sub dir", default="en-de")
    parser.add_argument("--dest", type=str, default=None, help="Path to the output PyTorch model.")
    # 解析命令行参数
    args = parser.parse_args()

    # 将源目录路径转换为Path对象
    source_dir = Path(args.src)
    # 如果源目录不存在，则抛出ValueError异常
    if not source_dir.exists():
        raise ValueError(f"Source directory {source_dir} not found")
    # 如果未指定目标目录，则使用源目录名作为转换后的目录名
    dest_dir = f"converted-{source_dir.name}" if args.dest is None else args.dest
    # 调用convert函数进行模型转换
    convert(source_dir, dest_dir)
```