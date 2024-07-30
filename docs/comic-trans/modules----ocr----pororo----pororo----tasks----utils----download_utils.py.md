# `.\comic-translate\modules\ocr\pororo\pororo\tasks\utils\download_utils.py`

```py
"""Module download related function from. Tenth"""

import logging
import os
#import platform
import sys
import zipfile
from dataclasses import dataclass
from typing import Tuple, Union

import wget

from ....pororo.tasks.utils.config import CONFIGS

DEFAULT_PREFIX = {
    "model": "https://twg.kakaocdn.net/pororo/{lang}/models",
    "dict": "https://twg.kakaocdn.net/pororo/{lang}/dicts",
}


@dataclass
class TransformerInfo:
    r"Dataclass for transformer-based model"
    path: str
    dict_path: str
    src_dict: str
    tgt_dict: str
    src_tok: Union[str, None]
    tgt_tok: Union[str, None]


@dataclass
class DownloadInfo:
    r"Download information such as defined directory, language and model name"
    n_model: str
    lang: str
    root_dir: str


def get_save_dir(save_dir: str = None) -> str:
    """
    Get default save directory

    Args:
        savd_dir(str): User-defined save directory

    Returns:
        str: Set save directory

    """
    # If user wants to manually define save directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    #pf = platform.system()

    # if pf == "Windows":
    #     save_dir = "C:\\pororo"
    # else:
    #     home_dir = os.path.expanduser("~")
    #     save_dir = os.path.join(home_dir, ".pororo")

    # Default save directory if not provided by user
    save_dir = "models/ocr/pororo"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    return save_dir


def get_download_url(n_model: str, key: str, lang: str) -> str:
    """
    Get download url using default prefix

    Args:
        n_model (str): model name
        key (str): key name either `model` or `dict`
        lang (str): language name

    Returns:
        str: generated download url

    """
    # Constructing download URL based on model, key, and language
    default_prefix = DEFAULT_PREFIX[key].format(lang=lang)
    return f"{default_prefix}/{n_model}"


def download_or_load_bert(info: DownloadInfo) -> str:
    """
    Download fine-tuned BrainBert & BrainSBert model and dict

    Args:
        info (DownloadInfo): download information

    Returns:
        str: downloaded bert & sbert path

    """
    # Construct full path where the model will be stored
    model_path = os.path.join(info.root_dir, info.n_model)

    # Check if model path already exists
    if not os.path.exists(model_path):
        # Append '.zip' extension to model name
        info.n_model += ".zip"
        zip_path = os.path.join(info.root_dir, info.n_model)

        # Download the model zip file from URL
        type_dir = download_from_url(
            info.n_model,
            zip_path,
            key="model",
            lang=info.lang,
        )

        # Extract the downloaded zip file to the specified directory
        zip_file = zipfile.ZipFile(zip_path)
        zip_file.extractall(type_dir)
        zip_file.close()

    return model_path


def download_or_load_transformer(info: DownloadInfo) -> TransformerInfo:
    """
    Download pre-trained Transformer model and corresponding dict

    Args:
        info (DownloadInfo): download information

    Returns:
        TransformerInfo: information dataclass for transformer construction

    """
    # Retrieve configuration details for the specified model
    config = CONFIGS[info.n_model.split("/")[-1]]

    # Initialize paths from the configuration
    src_dict_in = config.src_dict
    # 从配置中获取目标字典路径
    tgt_dict_in = config.tgt_dict
    # 从配置中获取源语言的标记器
    src_tok = config.src_tok
    # 从配置中获取目标语言的标记器
    tgt_tok = config.tgt_tok

    # 将模型名称后缀改为".pt"
    info.n_model += ".pt"
    # 构建完整的模型路径
    model_path = os.path.join(info.root_dir, info.n_model)

    # 下载或加载Transformer模型
    model_type_dir = "/".join(model_path.split("/")[:-1])
    if not os.path.exists(model_path):
        # 如果模型路径不存在，则下载对应的模型文件
        model_type_dir = download_from_url(
            info.n_model,
            model_path,
            key="model",
            lang=info.lang,
        )

    dict_type_dir = str()
    src_dict, tgt_dict = str(), str()

    # 下载或加载对应的字典文件
    if src_dict_in:
        # 构建源语言字典文件名
        src_dict = f"{src_dict_in}.txt"
        src_dict_path = os.path.join(info.root_dir, f"dicts/{src_dict}")
        dict_type_dir = "/".join(src_dict_path.split("/")[:-1])
        if not os.path.exists(src_dict_path):
            # 如果字典文件不存在，则下载对应的字典文件
            dict_type_dir = download_from_url(
                src_dict,
                src_dict_path,
                key="dict",
                lang=info.lang,
            )

    if tgt_dict_in:
        # 构建目标语言字典文件名
        tgt_dict = f"{tgt_dict_in}.txt"
        tgt_dict_path = os.path.join(info.root_dir, f"dicts/{tgt_dict}")
        if not os.path.exists(tgt_dict_path):
            # 如果字典文件不存在，则下载对应的字典文件
            download_from_url(
                tgt_dict,
                tgt_dict_path,
                key="dict",
                lang=info.lang,
            )

    # 下载或加载对应的标记器文件
    src_tok_path, tgt_tok_path = None, None
    if src_tok:
        src_tok_path = download_or_load(
            f"tokenizers/{src_tok}.zip",
            lang=info.lang,
        )
    if tgt_tok:
        tgt_tok_path = download_or_load(
            f"tokenizers/{tgt_tok}.zip",
            lang=info.lang,
        )

    # 返回Transformer模型、字典和标记器的信息
    return TransformerInfo(
        path=model_type_dir,
        dict_path=dict_type_dir,
        # 去除前缀"dict."和后缀".txt"
        src_dict=".".join(src_dict.split(".")[1:-1]),
        # 遵循fairseq的字典加载过程
        tgt_dict=".".join(tgt_dict.split(".")[1:-1]),
        src_tok=src_tok_path,
        tgt_tok=tgt_tok_path,
    )
def download_or_load_misc(info: DownloadInfo) -> str:
    """
    Download (pre-trained) miscellaneous model

    Args:
        info (DownloadInfo): download information

    Returns:
        str: miscellaneous model path

    """
    # 如果模型名称中包含 "sentencepiece"，则添加 ".model" 后缀
    if "sentencepiece" in info.n_model:
        info.n_model += ".model"

    # 使用根目录和模型名称生成目标模型路径
    model_path = os.path.join(info.root_dir, info.n_model)
    # 如果目标路径不存在
    if not os.path.exists(model_path):
        # 从指定 URL 下载模型文件，并保存到 model_path
        type_dir = download_from_url(
            info.n_model,
            model_path,
            key="model",
            lang=info.lang,
        )

        # 如果模型名称中包含 ".zip"
        if ".zip" in info.n_model:
            # 打开模型路径的 ZIP 文件
            zip_file = zipfile.ZipFile(model_path)
            # 解压缩 ZIP 文件中的所有内容到 type_dir
            zip_file.extractall(type_dir)
            # 关闭 ZIP 文件
            zip_file.close()

    # 如果模型名称中包含 ".zip"，则更新模型路径为去除 ".zip" 后缀的版本
    if ".zip" in info.n_model:
        model_path = model_path[:model_path.rfind(".zip")]
    # 返回最终的模型路径
    return model_path


def download_or_load_bart(info: DownloadInfo) -> Union[str, Tuple[str, str]]:
    """
    Download BART model

    Args:
        info (DownloadInfo): download information

    Returns:
        Union[str, Tuple[str, str]]: BART model path (with corresponding SentencePiece)

    """
    # 给模型名称添加 ".pt" 后缀
    info.n_model += ".pt"

    # 使用根目录和模型名称生成目标模型路径
    model_path = os.path.join(info.root_dir, info.n_model)
    # 如果目标路径不存在
    if not os.path.exists(model_path):
        # 从指定 URL 下载模型文件，并保存到 model_path
        download_from_url(
            info.n_model,
            model_path,
            key="model",
            lang=info.lang,
        )

    # 返回最终的模型路径
    return model_path


def download_from_url(
    n_model: str,
    model_path: str,
    key: str,
    lang: str,
) -> str:
    """
    Download specified model from Tenth

    Args:
        n_model (str): model name
        model_path (str): pre-defined model path
        key (str): type key (either model or dict)
        lang (str): language name

    Returns:
        str: default type directory

    """
    # 获取默认的类型目录路径
    type_dir = "/".join(model_path.split("/")[:-1])
    # 创建目录，如果不存在则自动创建
    os.makedirs(type_dir, exist_ok=True)

    # 获取下载链接
    url = get_download_url(n_model, key=key, lang=lang)

    # 记录下载信息
    logging.info("Downloading user-selected model...")
    # 使用 wget 下载指定 URL 的文件到 type_dir
    wget.download(url, type_dir)
    # 输出换行符到标准错误流
    sys.stderr.write("\n")
    sys.stderr.flush()

    # 返回默认类型目录路径
    return type_dir


def download_or_load(
    n_model: str,
    lang: str,
    custom_save_dir: str = None,
) -> Union[TransformerInfo, str, Tuple[str, str]]:
    """
    Download or load model based on model information

    Args:
        n_model (str): model name
        lang (str): language information
        custom_save_dir (str, optional): user-defined save directory path. defaults to None.

    Returns:
        Union[TransformerInfo, str, Tuple[str, str]]

    """
    # 获取保存模型的根目录路径
    root_dir = get_save_dir(save_dir=custom_save_dir)
    # 创建 DownloadInfo 对象，包含模型名称、语言和根目录
    info = DownloadInfo(n_model, lang, root_dir)

    # 如果模型名称中包含 "transformer"，则调用 download_or_load_transformer 函数
    if "transformer" in n_model:
        return download_or_load_transformer(info)
    # 如果模型名称中包含 "bert"，则调用 download_or_load_bert 函数
    if "bert" in n_model:
        return download_or_load_bert(info)
    # 检查字符串 "bart" 是否在 n_model 中，并且字符串 "bpe" 不在 n_model 中
    if "bart" in n_model and "bpe" not in n_model:
        # 如果条件成立，调用 download_or_load_bart 函数并返回其结果
        return download_or_load_bart(info)

    # 如果条件不成立，则调用 download_or_load_misc 函数并返回其结果
    return download_or_load_misc(info)
```