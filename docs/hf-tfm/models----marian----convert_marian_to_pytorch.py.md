# `.\models\marian\convert_marian_to_pytorch.py`

```py
# 导入必要的库
import argparse  # 用于命令行参数解析
import json  # 用于处理 JSON 数据
import os  # 提供与操作系统交互的功能
import socket  # 提供网络通信的功能
import time  # 提供时间相关的功能
import warnings  # 用于处理警告信息
from pathlib import Path  # 提供操作文件路径的功能
from typing import Dict, List, Union  # 提供类型提示支持
from zipfile import ZipFile  # 用于处理 ZIP 文件

import numpy as np  # 提供数值计算支持
import torch  # 提供深度学习框架支持
from huggingface_hub.hf_api import list_models  # 用于获取模型列表的功能
from torch import nn  # 提供神经网络模块的支持
from tqdm import tqdm  # 提供进度条功能

from transformers import MarianConfig, MarianMTModel, MarianTokenizer  # 导入 Hugging Face 的模型相关组件


def remove_suffix(text: str, suffix: str):
    # 如果文本以指定后缀结尾，则移除后缀并返回
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text  # 如果没有匹配的后缀，则返回原始文本


def remove_prefix(text: str, prefix: str):
    # 如果文本以指定前缀开头，则移除前缀并返回
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # 如果没有匹配的前缀，则返回原始文本


def convert_encoder_layer(opus_dict, layer_prefix: str, converter: dict):
    # 将 OPUS 字典中特定前缀的层转换成 PyTorch 可用的状态字典
    sd = {}
    for k in opus_dict:
        if not k.startswith(layer_prefix):
            continue
        stripped = remove_prefix(k, layer_prefix)
        v = opus_dict[k].T  # 除了嵌入层，所有内容都需要转置
        sd[converter[stripped]] = torch.tensor(v).squeeze()
    return sd


def load_layers_(layer_lst: nn.ModuleList, opus_state: dict, converter, is_decoder=False):
    # 加载 OPUS 状态字典中的编码器或解码器层到模型的指定层列表中
    for i, layer in enumerate(layer_lst):
        layer_tag = f"decoder_l{i + 1}_" if is_decoder else f"encoder_l{i + 1}_"
        sd = convert_encoder_layer(opus_state, layer_tag, converter)
        layer.load_state_dict(sd, strict=False)


def find_pretrained_model(src_lang: str, tgt_lang: str) -> List[str]:
    """查找可以接受指定源语言并输出目标语言的模型列表。"""
    prefix = "Helsinki-NLP/opus-mt-"
    model_list = list_models()  # 获取模型列表信息
    model_ids = [x.modelId for x in model_list if x.modelId.startswith("Helsinki-NLP")]
    src_and_targ = [
        remove_prefix(m, prefix).lower().split("-") for m in model_ids if "+" not in m
    ]  # 只选择不含有 "+" 的模型
    matching = [f"{prefix}{a}-{b}" for (a, b) in src_and_targ if src_lang in a and tgt_lang in b]
    return matching  # 返回匹配的模型列表


def add_emb_entries(wemb, final_bias, n_special_tokens=1):
    # 添加特殊的嵌入条目和偏置项到词嵌入和偏置中
    vsize, d_model = wemb.shape
    embs_to_add = np.zeros((n_special_tokens, d_model))
    new_embs = np.concatenate([wemb, embs_to_add])
    bias_to_add = np.zeros((n_special_tokens, 1))
    new_bias = np.concatenate((final_bias, bias_to_add), axis=1)
    return new_embs, new_bias


def _cast_yaml_str(v):
    bool_dct = {"true": True, "false": False}
    # 检查变量 v 是否不是字符串类型，如果是其他类型则直接返回 v
    if not isinstance(v, str):
        return v
    # 如果 v 是布尔值字典 bool_dct 中的键，返回其对应的值
    elif v in bool_dct:
        return bool_dct[v]
    # 尝试将 v 转换为整数类型，如果成功则返回转换后的整数值
    try:
        return int(v)
    # 如果转换失败（TypeError 或 ValueError），则返回原始的 v 值
    except (TypeError, ValueError):
        return v
# 将原始配置字典中的每个值转换为 YAML 字符串，并返回新的字典
def cast_marian_config(raw_cfg: Dict[str, str]) -> Dict:
    return {k: _cast_yaml_str(v) for k, v in raw_cfg.items()}

# 定义配置文件的键名
CONFIG_KEY = "special:model.yml"

# 从给定的字典中加载配置信息，并返回转换后的配置字典
def load_config_from_state_dict(opus_dict):
    import yaml
    
    # 将从状态字典中取得的配置信息转换为字符串
    cfg_str = "".join([chr(x) for x in opus_dict[CONFIG_KEY]])
    # 使用 YAML 解析器加载配置字符串，使用 BaseLoader 作为加载器
    yaml_cfg = yaml.load(cfg_str[:-1], Loader=yaml.BaseLoader)
    # 调用 cast_marian_config 函数对 YAML 配置进行类型转换，并返回结果
    return cast_marian_config(yaml_cfg)

# 根据目标目录查找模型文件，并确保只有一个模型文件存在，返回该模型文件路径
def find_model_file(dest_dir):  # this one better
    model_files = list(Path(dest_dir).glob("*.npz"))
    if len(model_files) != 1:
        raise ValueError(f"Found more than one model file: {model_files}")
    model_file = model_files[0]
    return model_file

# 定义 ROMANCE 组的语言列表
ROM_GROUP = (
    "fr+fr_BE+fr_CA+fr_FR+wa+frp+oc+ca+rm+lld+fur+lij+lmo+es+es_AR+es_CL+es_CO+es_CR+es_DO"
    "+es_EC+es_ES+es_GT+es_HN+es_MX+es_NI+es_PA+es_PE+es_PR+es_SV+es_UY+es_VE+pt+pt_br+pt_BR"
    "+pt_PT+gl+lad+an+mwl+it+it_IT+co+nap+scn+vec+sc+ro+la"
)

# 定义语言组的列表，每个元组包含语言列表和对应的组名
GROUPS = [
    ("cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh", "ZH"),
    (ROM_GROUP, "ROMANCE"),
    ("de+nl+fy+af+da+fo+is+no+nb+nn+sv", "NORTH_EU"),
    ("da+fo+is+no+nb+nn+sv", "SCANDINAVIA"),
    ("se+sma+smj+smn+sms", "SAMI"),
    ("nb_NO+nb+nn_NO+nn+nog+no_nb+no", "NORWAY"),
    ("ga+cy+br+gd+kw+gv", "CELTIC"),  # https://en.wikipedia.org/wiki/Insular_Celtic_languages
]

# 定义从组名到 OPUS 模型名称的映射字典
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
    "opus-mt-CELTIC-en": "ga+cy+br+gd+kw+gv-en",
    # 为键 "opus-mt-CELTIC-en" 添加值 "ga+cy+br+gd+kw+gv-en"
    "opus-mt-sv-ZH": "sv-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh",
    # 为键 "opus-mt-sv-ZH" 添加值 "sv-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh"
    "opus-mt-sv-NORWAY": "sv-nb_NO+nb+nn_NO+nn+nog+no_nb+no",
    # 为键 "opus-mt-sv-NORWAY" 添加值 "sv-nb_NO+nb+nn_NO+nn+nog+no_nb+no"
}
# OPUS-GitHub 项目的 URL
OPUS_GITHUB_URL = "https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/models/"
# 组织名称
ORG_NAME = "Helsinki-NLP/"


def convert_opus_name_to_hf_name(x):
    """将 OPUS-MT-Train 名称转换为 Hugging Face 模型名称（已弃用）"""
    # 根据 GROUPS 中的替换规则，将 x 中的子字符串替换为对应的组名称
    for substr, grp_name in GROUPS:
        x = x.replace(substr, grp_name)
    return x.replace("+", "_")


def convert_hf_name_to_opus_name(hf_model_name):
    """
    根据假设，假设在不在 GROUP_TO_OPUS_NAME 中的模型中没有像 pt_br 这样的语言代码。
    将 Hugging Face 模型名称转换为 OPUS-MT-Train 名称
    """
    # 去除模型名称中的 ORG_NAME 前缀
    hf_model_name = remove_prefix(hf_model_name, ORG_NAME)
    if hf_model_name in GROUP_TO_OPUS_NAME:
        opus_w_prefix = GROUP_TO_OPUS_NAME[hf_model_name]
    else:
        opus_w_prefix = hf_model_name.replace("_", "+")
    return remove_prefix(opus_w_prefix, "opus-mt-")


def get_system_metadata(repo_root):
    import git

    # 返回系统元数据字典，包括 Helsinki 的 Git SHA、transformers 的 Git SHA、运行机器名、当前时间
    return {
        "helsinki_git_sha": git.Repo(path=repo_root, search_parent_directories=True).head.object.hexsha,
        "transformers_git_sha": git.Repo(path=".", search_parent_directories=True).head.object.hexsha,
        "port_machine": socket.gethostname(),
        "port_time": time.strftime("%Y-%m-%d-%H:%M"),
    }


# docstyle-ignore
# 前置内容模板，用于生成模型卡片的前置元数据
FRONT_MATTER_TEMPLATE = """---
language:
{}
tags:
- translation

license: apache-2.0
---
"""
# 默认仓库名称
DEFAULT_REPO = "Tatoeba-Challenge"
# 默认模型目录路径
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_REPO, "models")


def write_model_card(
    hf_model_name: str,
    repo_root=DEFAULT_REPO,
    save_dir=Path("marian_converted"),
    dry_run=False,
    extra_metadata={},
) -> str:
    """
    复制最新模型的 readme 部分来自 OPUS，并添加元数据。上传命令: aws s3 sync model_card_dir
    s3://models.huggingface.co/bert/Helsinki-NLP/ --dryrun
    """
    import pandas as pd

    # 去除模型名称中的 ORG_NAME 前缀
    hf_model_name = remove_prefix(hf_model_name, ORG_NAME)
    # 将 Hugging Face 模型名称转换为 OPUS-MT-Train 名称
    opus_name: str = convert_hf_name_to_opus_name(hf_model_name)
    if repo_root not in ("OPUS-MT-train", "Tatoeba-Challenge"):
        raise ValueError(f"Repos root is {repo_root}. Expected either OPUS-MT-train or Tatoeba-Challenge")
    # 构建 OPUS readme 文件路径
    opus_readme_path = Path(repo_root).joinpath("models", opus_name, "README.md")
    if not (opus_readme_path.exists()):
        raise ValueError(f"Readme file {opus_readme_path} not found")

    # 分离 OPUS 名称中的源语言和目标语言
    opus_src, opus_tgt = [x.split("+") for x in opus_name.split("-")]

    # 构建 OPUS README 在 GitHub 上的 URL
    readme_url = f"https://github.com/Helsinki-NLP/{repo_root}/tree/master/models/{opus_name}/README.md"

    s, t = ",".join(opus_src), ",".join(opus_tgt)
    # 构建元数据字典
    metadata = {
        "hf_name": hf_model_name,
        "source_languages": s,
        "target_languages": t,
        "opus_readme_url": readme_url,
        "original_repo": repo_root,
        "tags": ["translation"],
    }
    metadata.update(extra_metadata)
    # 添加系统元数据到元数据字典中
    metadata.update(get_system_metadata(repo_root))

    # 合并 OPUS readme 的 markdown 内容
    extra_markdown = (
        f"### {hf_model_name}\n\n* source group: {metadata['src_name']} \n* target group: "
        f"{metadata['tgt_name']} \n*  OPUS readme: [{opus_name}]({readme_url})\n"
    )
    # 构建额外的 Markdown 格式字符串，包含模型名称、源语言组、目标语言组和 OPUS readme 链接

    content = opus_readme_path.open().read()
    # 读取 OPUS readme 文件的内容

    content = content.split("\n# ")[-1]  # Get the lowest level 1 header in the README -- the most recent model.
    # 通过分割文本获取 README 中最底层的一级标题，即最近的模型信息

    splat = content.split("*")[2:]
    # 使用星号分割内容，从第三个星号开始获取后面的所有部分

    print(splat[3])
    # 打印第四个分割后的部分，假设这里是输出特定信息的调试步骤

    content = "*".join(splat)
    # 将分割后的内容重新连接起来，使用星号作为连接符

    content = (
        FRONT_MATTER_TEMPLATE.format(metadata["src_alpha2"])
        + extra_markdown
        + "\n* "
        + content.replace("download", "download original weights")
    )
    # 构建最终的内容字符串，包括前置模板、额外的 Markdown 信息和处理后的内容部分

    items = "\n\n".join([f"- {k}: {v}" for k, v in metadata.items()])
    # 将元数据中的键值对格式化为列表项

    sec3 = "\n### System Info: \n" + items
    # 构建系统信息部分的 Markdown 标题和元数据列表

    content += sec3
    # 将系统信息部分添加到最终的内容字符串中

    if dry_run:
        return content, metadata
    # 如果是 dry_run 模式，则返回内容字符串和元数据

    sub_dir = save_dir / f"opus-mt-{hf_model_name}"
    # 构建保存子目录路径，包括模型名称

    sub_dir.mkdir(exist_ok=True)
    # 创建保存子目录，如果已存在则忽略

    dest = sub_dir / "README.md"
    # 构建 README 文件路径

    dest.open("w").write(content)
    # 将最终的内容写入 README 文件

    pd.Series(metadata).to_json(sub_dir / "metadata.json")
    # 将元数据以 JSON 格式保存到子目录的 metadata.json 文件中

    # if dry_run:
    return content, metadata
    # 返回最终的内容字符串和元数据
# 创建注册表函数，用于处理特定路径下的模型注册
def make_registry(repo_path="Opus-MT-train/models"):
    # 检查指定路径下的 README.md 文件是否存在，如果不存在则抛出数值错误
    if not (Path(repo_path) / "fr-en" / "README.md").exists():
        raise ValueError(
            f"repo_path:{repo_path} does not exist: "
            "You must run: git clone git@github.com:Helsinki-NLP/Opus-MT-train.git before calling."
        )
    # 初始化结果字典
    results = {}
    # 遍历指定路径下的所有子目录和文件
    for p in Path(repo_path).iterdir():
        # 统计当前路径名称中 "-" 的数量
        n_dash = p.name.count("-")
        # 如果没有 "-"，则跳过当前路径
        if n_dash == 0:
            continue
        else:
            # 读取当前路径下的 README.md 文件的所有行
            lns = list(open(p / "README.md").readlines())
            # 使用解析函数处理 README.md 的内容，并存入结果字典
            results[p.name] = _parse_readme(lns)
    # 返回结果列表，包含每个模型的关键信息
    return [(k, v["pre-processing"], v["download"], v["download"][:-4] + ".test.txt") for k, v in results.items()]


# 批量转换所有 SentencePiece 模型
def convert_all_sentencepiece_models(model_list=None, repo_path=None, dest_dir=Path("marian_converted")):
    """Requires 300GB"""
    # 设置保存目录和目标目录
    save_dir = Path("marian_ckpt")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)
    # 初始化保存路径列表
    save_paths = []
    # 如果未指定模型列表，则调用 make_registry 函数获取模型列表
    if model_list is None:
        model_list: list = make_registry(repo_path=repo_path)
    # 遍历模型列表
    for k, prepro, download, test_set_url in tqdm(model_list):
        # 如果预处理中不包含 "SentencePiece"，则跳过当前模型
        if "SentencePiece" not in prepro:  # dont convert BPE models.
            continue
        # 如果保存目录中不存在当前模型的文件夹，则下载并解压缩模型文件
        if not os.path.exists(save_dir / k):
            download_and_unzip(download, save_dir / k)
        # 将 Opus 模型名转换为 Hugging Face 模型名
        pair_name = convert_opus_name_to_hf_name(k)
        # 执行模型转换操作
        convert(save_dir / k, dest_dir / f"opus-mt-{pair_name}")

        # 将转换后的模型保存路径加入保存路径列表
        save_paths.append(dest_dir / f"opus-mt-{pair_name}")
    # 返回所有转换后模型的保存路径列表
    return save_paths


# 自定义列表映射函数，对输入列表中的每个元素应用给定函数
def lmap(f, x) -> List:
    return list(map(f, x))


# 下载测试集并返回源语言、金标准和模型输出列表
def fetch_test_set(test_set_url):
    import wget

    # 使用 wget 下载测试集文件到本地
    fname = wget.download(test_set_url, "opus_test.txt")
    # 读取下载的文件的所有行
    lns = Path(fname).open().readlines()
    # 提取源语言、金标准和模型输出的列表，并进行字符串修剪
    src = lmap(str.strip, lns[::4])
    gold = lmap(str.strip, lns[1::4])
    mar_model = lmap(str.strip, lns[2::4])
    # 检查三个列表的长度是否相等，如果不相等则抛出数值错误
    if not (len(gold) == len(mar_model) == len(src)):
        raise ValueError(f"Gold, marian and source lengths {len(gold)}, {len(mar_model)}, {len(src)} mismatched")
    # 删除下载的测试集文件
    os.remove(fname)
    # 返回源语言列表、模型输出列表和金标准列表
    return src, mar_model, gold


# 批量转换指定目录下的所有模型文件
def convert_whole_dir(path=Path("marian_ckpt/")):
    # 遍历指定路径下的所有子目录
    for subdir in tqdm(list(path.ls())):
        # 设置目标目录路径
        dest_dir = f"marian_converted/{subdir.name}"
        # 如果目标目录中已存在 pytorch_model.bin 文件，则跳过当前子目录
        if (dest_dir / "pytorch_model.bin").exists():
            continue
        # 执行模型转换操作
        convert(source_dir, dest_dir)


# 解析 README.md 文件内容，获取 Opus 模型的链接和元数据
def _parse_readme(lns):
    """Get link and metadata from opus model card equivalent."""
    # 初始化子结果字典
    subres = {}
    # 遍历所有行
    for ln in [x.strip() for x in lns]:
        # 如果行不以 "*" 开头，则跳过当前行
        if not ln.startswith("*"):
            continue
        # 去掉首部的 "*" 符号
        ln = ln[1:].strip()

        # 遍历关键词列表，识别关键词并提取对应的值
        for k in ["download", "dataset", "models", "model", "pre-processing"]:
            if ln.startswith(k):
                break
        else:
            continue
        # 根据关键词类型处理对应的值
        if k in ["dataset", "model", "pre-processing"]:
            splat = ln.split(":")
            _, v = splat
            subres[k] = v
        elif k == "download":
            v = ln.split("(")[-1][:-1]
            subres[k] = v
    # 返回子结果字典，包含从 README.md 中提取的所有信息
    return subres


# 保存分词器配置到指定目录
def save_tokenizer_config(dest_dir: Path, separate_vocabs=False):
    # 将目标目录的名称按照 "-" 分割成列表
    dname = dest_dir.name.split("-")
    # 构建包含目标语言、源语言和是否分开词汇表的字典
    dct = {"target_lang": dname[-1], "source_lang": "-".join(dname[:-1]), "separate_vocabs": separate_vocabs}
    # 将字典保存为 JSON 文件，文件名为 "tokenizer_config.json"，保存在目标目录中
    save_json(dct, dest_dir / "tokenizer_config.json")
# 向词汇表中添加特殊标记，如果需要分开处理词汇表，则加载源和目标词汇表并分别处理
def add_special_tokens_to_vocab(model_dir: Path, separate_vocab=False) -> None:
    if separate_vocab:
        # 加载源语言词汇表并转换为整数键值对
        vocab = load_yaml(find_src_vocab_file(model_dir))
        vocab = {k: int(v) for k, v in vocab.items()}
        # 向词汇表中添加特殊标记"<pad>"，返回添加的标记数目
        num_added = add_to_vocab_(vocab, ["<pad>"])
        # 将更新后的词汇表保存为 JSON 文件
        save_json(vocab, model_dir / "vocab.json")

        # 加载目标语言词汇表并转换为整数键值对
        vocab = load_yaml(find_tgt_vocab_file(model_dir))
        vocab = {k: int(v) for k, v in vocab.items()}
        # 向词汇表中添加特殊标记"<pad>"，返回添加的标记数目
        num_added = add_to_vocab_(vocab, ["<pad>"])
        # 将更新后的目标语言词汇表保存为 JSON 文件
        save_json(vocab, model_dir / "target_vocab.json")
        # 保存分词器配置
        save_tokenizer_config(model_dir, separate_vocabs=separate_vocab)
    else:
        # 加载统一词汇表并转换为整数键值对
        vocab = load_yaml(find_vocab_file(model_dir))
        vocab = {k: int(v) for k, v in vocab.items()}
        # 向词汇表中添加特殊标记"<pad>"，返回添加的标记数目
        num_added = add_to_vocab_(vocab, ["<pad>"])
        # 打印添加的标记数目
        print(f"added {num_added} tokens to vocab")
        # 将更新后的词汇表保存为 JSON 文件
        save_json(vocab, model_dir / "vocab.json")
        # 保存分词器配置
        save_tokenizer_config(model_dir)



# 检查两个键对应的值是否相等，若不相等则抛出 ValueError 异常
def check_equal(marian_cfg, k1, k2):
    v1, v2 = marian_cfg[k1], marian_cfg[k2]
    if v1 != v2:
        raise ValueError(f"hparams {k1},{k2} differ: {v1} != {v2}")



# 检索指定目录下的第一个以 "*vocab.yml" 结尾的文件并返回其路径
def find_vocab_file(model_dir):
    return list(model_dir.glob("*vocab.yml"))[0]



# 检索指定目录下的第一个以 "*src.vocab.yml" 结尾的文件并返回其路径
def find_src_vocab_file(model_dir):
    return list(model_dir.glob("*src.vocab.yml"))[0]



# 检索指定目录下的第一个以 "*trg.vocab.yml" 结尾的文件并返回其路径
def find_tgt_vocab_file(model_dir):
    return list(model_dir.glob("*trg.vocab.yml"))[0]



# 向词汇表中添加特殊标记，根据词汇表中最大的值确定起始位置
def add_to_vocab_(vocab: Dict[str, int], special_tokens: List[str]):
    start = max(vocab.values()) + 1  # 确定新添加标记的起始位置
    added = 0  # 初始化添加的标记数目
    for tok in special_tokens:
        if tok in vocab:
            continue
        vocab[tok] = start + added  # 将特殊标记添加到词汇表中
        added += 1  # 更新添加的标记数目
    return added  # 返回添加的标记数目



# 检查 marian_cfg 中指定的配置项是否符合预期设置
def check_marian_cfg_assumptions(marian_cfg):
    assumed_settings = {
        "layer-normalization": False,
        "right-left": False,
        "transformer-ffn-depth": 2,
        "transformer-aan-depth": 2,
        "transformer-no-projection": False,
        "transformer-postprocess-emb": "d",
        "transformer-postprocess": "dan",  # Dropout, add, normalize
        "transformer-preprocess": "",
        "type": "transformer",
        "ulr-dim-emb": 0,
        "dec-cell-base-depth": 2,
        "dec-cell-high-depth": 1,
        "transformer-aan-nogate": False,
    }
    for k, v in assumed_settings.items():
        actual = marian_cfg[k]
        if actual != v:
            raise ValueError(f"Unexpected config value for {k} expected {v} got {actual}")



# BART 模型的配置映射，将不同的层权重映射到对应的键
BIAS_KEY = "decoder_ff_logit_out_b"
BART_CONVERTER = {  # 用于每个编码器和解码器层
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
}
    # 权重矩阵和偏置向量对应于神经网络的第二个全连接层
    "ffn_W2": "fc2.weight",
    "ffn_b2": "fc2.bias",
    
    # 最终层归一化的缩放因子和偏置项
    "ffn_ffn_ln_scale": "final_layer_norm.weight",
    "ffn_ffn_ln_bias": "final_layer_norm.bias",
    
    # 解码器交叉注意力机制中的权重矩阵和偏置向量
    "context_Wk": "encoder_attn.k_proj.weight",
    "context_Wo": "encoder_attn.out_proj.weight",
    "context_Wq": "encoder_attn.q_proj.weight",
    "context_Wv": "encoder_attn.v_proj.weight",
    "context_bk": "encoder_attn.k_proj.bias",
    "context_bo": "encoder_attn.out_proj.bias",
    "context_bq": "encoder_attn.q_proj.bias",
    "context_bv": "encoder_attn.v_proj.bias",
    
    # 编码器注意力层归一化的缩放因子和偏置项
    "context_Wo_ln_scale": "encoder_attn_layer_norm.weight",
    "context_Wo_ln_bias": "encoder_attn_layer_norm.bias",
    }

class OpusState:
    # 检查层条目的有效性，初始化编码器和解码器的第一层的键列表
    def _check_layer_entries(self):
        self.encoder_l1 = self.sub_keys("encoder_l1")  # 获取编码器第一层的键列表
        self.decoder_l1 = self.sub_keys("decoder_l1")  # 获取解码器第一层的键列表
        self.decoder_l2 = self.sub_keys("decoder_l2")  # 获取解码器第二层的键列表
        # 检查编码器第一层键的数量是否为16，如果不是则发出警告
        if len(self.encoder_l1) != 16:
            warnings.warn(f"Expected 16 keys for each encoder layer, got {len(self.encoder_l1)}")
        # 检查解码器第一层键的数量是否为26，如果不是则发出警告
        if len(self.decoder_l1) != 26:
            warnings.warn(f"Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}")
        # 检查解码器第二层键的数量是否为26，如果不是则发出警告
        if len(self.decoder_l2) != 26:
            warnings.warn(f"Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}")

    @property
    # 获取额外的键列表，排除特定的键
    def extra_keys(self):
        extra = []
        # 遍历状态键列表，排除特定的键，生成额外的键列表
        for k in self.state_keys:
            if (
                k.startswith("encoder_l")
                or k.startswith("decoder_l")
                or k in [CONFIG_KEY, "Wemb", "encoder_Wemb", "decoder_Wemb", "Wpos", "decoder_ff_logit_out_b"]
            ):
                continue
            else:
                extra.append(k)
        return extra

    # 获取给定层前缀的子键列表
    def sub_keys(self, layer_prefix):
        return [remove_prefix(k, layer_prefix) for k in self.state_dict if k.startswith(layer_prefix)]

    # 加载分词器，根据源目录加载Marian分词器
    def load_tokenizer(self):
        add_special_tokens_to_vocab(self.source_dir, not self.share_encoder_decoder_embeddings)  # 将特殊标记添加到词汇表中
        return MarianTokenizer.from_pretrained(str(self.source_dir))  # 返回从预训练模型加载的Marian分词器
    # 加载 MarianMTModel 模型的方法，返回一个 MarianMTModel 对象
    def load_marian_model(self) -> MarianMTModel:
        # 获取状态字典和 HF 配置
        state_dict, cfg = self.state_dict, self.hf_config

        # 如果配置中 static_position_embeddings 不为 True，则抛出数值错误异常
        if not cfg.static_position_embeddings:
            raise ValueError("config.static_position_embeddings should be True")

        # 根据配置创建 MarianMTModel 模型对象
        model = MarianMTModel(cfg)

        # 如果配置中包含 "hidden_size" 键，抛出数值错误异常
        if "hidden_size" in cfg.to_dict():
            raise ValueError("hidden_size is in config")

        # 加载编码器层的状态字典到模型中，使用 BART_CONVERTER 转换
        load_layers_(
            model.model.encoder.layers,
            state_dict,
            BART_CONVERTER,
        )

        # 加载解码器层的状态字典到模型中，使用 BART_CONVERTER 转换，并指定为解码器层
        load_layers_(model.model.decoder.layers, state_dict, BART_CONVERTER, is_decoder=True)

        # 处理与层无关的张量
        if self.cfg["tied-embeddings-src"]:
            # 如果源语言嵌入被绑定，创建源语言嵌入张量和偏置张量，并分配给模型共享的权重
            wemb_tensor = nn.Parameter(torch.FloatTensor(self.wemb))
            bias_tensor = nn.Parameter(torch.FloatTensor(self.final_bias))
            model.model.shared.weight = wemb_tensor
            model.model.encoder.embed_tokens = model.model.decoder.embed_tokens = model.model.shared
        else:
            # 如果未绑定源语言嵌入，创建源语言嵌入张量，并分配给编码器的嵌入权重
            wemb_tensor = nn.Parameter(torch.FloatTensor(self.wemb))
            model.model.encoder.embed_tokens.weight = wemb_tensor

            # 创建解码器嵌入张量、偏置张量，并分配给解码器的嵌入权重和最终偏置
            decoder_wemb_tensor = nn.Parameter(torch.FloatTensor(self.dec_wemb))
            bias_tensor = nn.Parameter(torch.FloatTensor(self.final_bias))
            model.model.decoder.embed_tokens.weight = decoder_wemb_tensor

        # 将最终偏置张量分配给模型的最终对数偏置
        model.final_logits_bias = bias_tensor

        # 如果状态字典中存在 "Wpos" 键，打印警告信息
        if "Wpos" in state_dict:
            print("Unexpected: got Wpos")
            # 创建 Wpos 张量并分配给编码器和解码器的位置嵌入权重
            wpos_tensor = torch.tensor(state_dict["Wpos"])
            model.model.encoder.embed_positions.weight = wpos_tensor
            model.model.decoder.embed_positions.weight = wpos_tensor

        # 如果配置中启用了嵌入归一化
        if cfg.normalize_embedding:
            # 如果状态字典中缺少 "encoder_emb_ln_scale_pre" 键，抛出数值错误异常
            if "encoder_emb_ln_scale_pre" not in state_dict:
                raise ValueError("encoder_emb_ln_scale_pre is not in state dictionary")
            # 抛出未实现错误，需要转换 layernorm_embedding
            raise NotImplementedError("Need to convert layernorm_embedding")

        # 如果存在额外的键，抛出数值错误异常
        if self.extra_keys:
            raise ValueError(f"Failed to convert {self.extra_keys}")

        # 如果模型的输入嵌入的填充索引与 self.pad_token_id 不匹配，抛出数值错误异常
        if model.get_input_embeddings().padding_idx != self.pad_token_id:
            raise ValueError(
                f"Padding tokens {model.get_input_embeddings().padding_idx} and {self.pad_token_id} mismatched"
            )

        # 返回加载完成的 MarianMTModel 模型对象
        return model
    """
    Tatoeba conversion instructions in scripts/tatoeba/README.md
    """
    # 导入 argparse 模块，用于处理命令行参数
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument("--src", type=str, help="path to marian model sub dir", default="en-de")
    parser.add_argument("--dest", type=str, default=None, help="Path to the output PyTorch model.")
    # 解析命令行参数
    args = parser.parse_args()

    # 将源目录路径转换为 Path 对象
    source_dir = Path(args.src)
    # 如果源目录不存在，则抛出 ValueError 异常
    if not source_dir.exists():
        raise ValueError(f"Source directory {source_dir} not found")
    # 将目标目录转换为路径字符串，默认情况下是在源目录名前加上 'converted-'
    dest_dir = f"converted-{source_dir.name}" if args.dest is None else args.dest
    # 调用 convert 函数，进行模型转换
    convert(source_dir, dest_dir)
```