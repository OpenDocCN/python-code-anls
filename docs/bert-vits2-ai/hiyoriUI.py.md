# `Bert-VITS2\hiyoriUI.py`

```

"""
api服务，网页后端 多版本多模型 fastapi实现
原 server_fastapi
"""
# 导入所需的库
import logging  # 日志记录
import gc  # 垃圾回收
import random  # 随机数生成
import librosa  # 音频处理库
import gradio  # 用于构建交互式界面的库
import numpy as np  # 数组处理库
import utils  # 自定义工具库
from fastapi import FastAPI, Query, Request, File, UploadFile, Form  # FastAPI框架相关
from fastapi.responses import Response, FileResponse  # FastAPI框架相关
from fastapi.staticfiles import StaticFiles  # FastAPI框架相关
from io import BytesIO  # 用于处理二进制数据的库
from scipy.io import wavfile  # 读取和写入音频文件的库
import uvicorn  # ASGI服务器
import torch  # PyTorch深度学习库
import webbrowser  # 控制浏览器的库
import psutil  # 获取系统信息的库
import GPUtil  # 获取GPU信息的库
from typing import Dict, Optional, List, Set, Union, Tuple  # 类型提示相关
import os  # 系统相关操作的库
from tools.log import logger  # 自定义日志记录器
from urllib.parse import unquote  # URL解析相关的库

from infer import infer, get_net_g, latest_version  # 导入推理相关的函数和变量
import tools.translate as trans  # 导入翻译工具
from tools.sentence import split_by_language  # 导入句子分割工具
from re_matching import cut_sent  # 导入句子匹配工具

from config import config  # 导入配置文件

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 设置环境变量

# 定义模型类
class Model:
    """模型封装类"""

    def __init__(self, config_path: str, model_path: str, device: str, language: str):
        # 初始化模型类
        self.config_path: str = os.path.normpath(config_path)  # 规范化配置文件路径
        self.model_path: str = os.path.normpath(model_path)  # 规范化模型路径
        self.device: str = device  # 设备
        self.language: str = language  # 语言
        self.hps = utils.get_hparams_from_file(config_path)  # 从配置文件中获取超参数
        self.spk2id: Dict[str, int] = self.hps.data.spk2id  # spk - id 映射字典
        self.id2spk: Dict[int, str] = dict()  # id - spk 映射字典
        for speaker, speaker_id in self.hps.data.spk2id.items():
            self.id2spk[speaker_id] = speaker
        self.version: str = (
            self.hps.version if hasattr(self.hps, "version") else latest_version
        )  # 获取模型版本
        self.net_g = get_net_g(
            model_path=model_path,
            version=self.version,
            device=device,
            hps=self.hps,
        )  # 获取生成器网络

    def to_dict(self) -> Dict[str, any]:
        return {
            "config_path": self.config_path,
            "model_path": self.model_path,
            "device": self.device,
            "language": self.language,
            "spk2id": self.spk2id,
            "id2spk": self.id2spk,
            "version": self.version,
        }  # 将模型信息转换为字典格式


# 定义模型集合类
class Models:
    def __init__(self):
        self.models: Dict[int, Model] = dict()  # 模型字典
        self.num = 0  # 模型数量
        # spkInfo[角色名][模型id] = 角色id
        self.spk_info: Dict[str, Dict[int, int]] = dict()  # 角色信息字典
        self.path2ids: Dict[str, Set[int]] = dict()  # 路径指向的model的id

    def init_model(
        self, config_path: str, model_path: str, device: str, language: str
    ) -> int:
        """
        初始化并添加一个模型

        :param config_path: 模型config.json路径
        :param model_path: 模型路径
        :param device: 模型推理使用设备
        :param language: 模型推理默认语言
        """
        # 若文件不存在则不进行加载
        if not os.path.isfile(model_path):
            if model_path != "":
                logger.warning(f"模型文件{model_path} 不存在，不进行初始化")
            return self.num
        if not os.path.isfile(config_path):
            if config_path != "":
                logger.warning(f"配置文件{config_path} 不存在，不进行初始化")
            return self.num

        # 若路径中的模型已存在，则不添加模型，若不存在，则进行初始化。
        model_path = os.path.realpath(model_path)
        if model_path not in self.path2ids.keys():
            self.path2ids[model_path] = {self.num}
            self.models[self.num] = Model(
                config_path=config_path,
                model_path=model_path,
                device=device,
                language=language,
            )
            logger.success(f"添加模型{model_path}，使用配置文件{os.path.realpath(config_path)}")
        else:
            # 获取一个指向id
            m_id = next(iter(self.path2ids[model_path]))
            self.models[self.num] = self.models[m_id]
            self.path2ids[model_path].add(self.num)
            logger.success("模型已存在，添加模型引用。")
        # 添加角色信息
        for speaker, speaker_id in self.models[self.num].spk2id.items():
            if speaker not in self.spk_info.keys():
                self.spk_info[speaker] = {self.num: speaker_id}
            else:
                self.spk_info[speaker][self.num] = speaker_id
        # 修改计数
        self.num += 1
        return self.num - 1

    def del_model(self, index: int) -> Optional[int]:
        """删除对应序号的模型，若不存在则返回None"""
        if index not in self.models.keys():
            return None
        # 删除角色信息
        for speaker, speaker_id in self.models[index].spk2id.items():
            self.spk_info[speaker].pop(index)
            if len(self.spk_info[speaker]) == 0:
                # 若对应角色的所有模型都被删除，则清除该角色信息
                self.spk_info.pop(speaker)
        # 删除路径信息
        model_path = os.path.realpath(self.models[index].model_path)
        self.path2ids[model_path].remove(index)
        if len(self.path2ids[model_path]) == 0:
            self.path2ids.pop(model_path)
            logger.success(f"删除模型{model_path}, id = {index}")
        else:
            logger.success(f"删除模型引用{model_path}, id = {index}")
        # 删除模型
        self.models.pop(index)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return index

    def get_models(self):
        """获取所有模型"""
        return self.models

```