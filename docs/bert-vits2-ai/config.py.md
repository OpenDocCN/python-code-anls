# `Bert-VITS2\config.py`

```py
"""
@Desc: 全局配置文件读取
"""
# 导入必要的库
import argparse  # 用于解析命令行参数
import yaml  # 用于读取和写入 YAML 文件
from typing import Dict, List  # 用于类型提示
import os  # 用于操作系统相关功能
import shutil  # 用于高级文件操作
import sys  # 提供对 Python 解释器的访问和一些与解释器交互的函数

# 定义重采样配置类
class Resample_config:
    """重采样配置"""

    def __init__(self, in_dir: str, out_dir: str, sampling_rate: int = 44100):
        # 目标采样率
        self.sampling_rate: int = sampling_rate
        # 待处理音频目录路径
        self.in_dir: str = in_dir
        # 重采样输出路径
        self.out_dir: str = out_dir

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        """从字典中生成实例"""

        # 不检查路径是否有效，此逻辑在resample.py中处理
        data["in_dir"] = os.path.join(dataset_path, data["in_dir"])
        data["out_dir"] = os.path.join(dataset_path, data["out_dir"])

        return cls(**data)


# 定义数据预处理配置类
class Preprocess_text_config:
    """数据预处理配置"""

    def __init__(
        self,
        transcription_path: str,
        cleaned_path: str,
        train_path: str,
        val_path: str,
        config_path: str,
        val_per_lang: int = 5,
        max_val_total: int = 10000,
        clean: bool = True,
    ):
        # 原始文本文件路径，文本格式应为{wav_path}|{speaker_name}|{language}|{text}。
        self.transcription_path: str = transcription_path
        # 数据清洗后文本路径，可以不填。不填则将在原始文本目录生成
        self.cleaned_path: str = cleaned_path
        # 训练集路径，可以不填。不填则将在原始文本目录生成
        self.train_path: str = train_path
        # 验证集路径，可以不填。不填则将在原始文本目录生成
        self.val_path: str = val_path
        # 配置文件路径
        self.config_path: str = config_path
        # 每个speaker的验证集条数
        self.val_per_lang: int = val_per_lang
        # 验证集最大条数，多于的会被截断并放到训练集中
        self.max_val_total: int = max_val_total
        # 是否进行数据清洗
        self.clean: bool = clean

    @classmethod
    # 从字典中生成实例的类方法
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        """从字典中生成实例"""

        # 将数据中的transcription_path与dataset_path拼接成完整路径
        data["transcription_path"] = os.path.join(
            dataset_path, data["transcription_path"]
        )
        # 如果cleaned_path为空或None，则将其赋值为None，否则拼接成完整路径
        if data["cleaned_path"] == "" or data["cleaned_path"] is None:
            data["cleaned_path"] = None
        else:
            data["cleaned_path"] = os.path.join(dataset_path, data["cleaned_path"])
        # 将train_path、val_path、config_path与dataset_path拼接成完整路径
        data["train_path"] = os.path.join(dataset_path, data["train_path"])
        data["val_path"] = os.path.join(dataset_path, data["val_path"])
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        # 使用拼接后的数据生成实例并返回
        return cls(**data)
class Bert_gen_config:
    """bert_gen 配置"""

    def __init__(
        self,
        config_path: str,
        num_processes: int = 2,
        device: str = "cuda",
        use_multi_device: bool = False,
    ):
        # 初始化函数，设置配置路径、进程数、设备类型、是否使用多设备
        self.config_path = config_path
        self.num_processes = num_processes
        self.device = device
        self.use_multi_device = use_multi_device

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        # 从字典中创建配置对象，设置配置路径为数据集路径和配置文件名的组合
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        return cls(**data)


class Emo_gen_config:
    """emo_gen 配置"""

    def __init__(
        self,
        config_path: str,
        num_processes: int = 2,
        device: str = "cuda",
        use_multi_device: bool = False,
    ):
        # 初始化函数，设置配置路径、进程数、设备类型、是否使用多设备
        self.config_path = config_path
        self.num_processes = num_processes
        self.device = device
        self.use_multi_device = use_multi_device

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        # 从字典中创建配置对象，设置配置路径为数据集路径和配置文件名的组合
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        return cls(**data)


class Train_ms_config:
    """训练配置"""

    def __init__(
        self,
        config_path: str,
        env: Dict[str, any],
        base: Dict[str, any],
        model: str,
        num_workers: int,
        spec_cache: bool,
        keep_ckpts: int,
    ):
        # 初始化函数，设置配置路径、环境变量、底模配置、训练模型存储目录、worker数量、是否启用spec缓存、ckpt数量
        self.env = env  # 需要加载的环境变量
        self.base = base  # 底模配置
        self.model = model  # 训练模型存储目录，该路径为相对于dataset_path的路径，而非项目根目录
        self.config_path = config_path  # 配置文件路径
        self.num_workers = num_workers  # worker数量
        self.spec_cache = spec_cache  # 是否启用spec缓存
        self.keep_ckpts = keep_ckpts  # ckpt数量

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        # 从字典中创建配置对象，设置模型路径为数据集路径和模型文件名的组合，配置路径为数据集路径和配置文件名的组合
        # data["model"] = os.path.join(dataset_path, data["model"])
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        return cls(**data)
class Webui_config:
    """webui 配置"""

    def __init__(
        self,
        device: str,
        model: str,
        config_path: str,
        language_identification_library: str,
        port: int = 7860,
        share: bool = False,
        debug: bool = False,
    ):
        # 初始化 Webui_config 类的属性
        self.device: str = device
        self.model: str = model  # 端口号
        self.config_path: str = config_path  # 是否公开部署，对外网开放
        self.port: int = port  # 是否开启debug模式
        self.share: bool = share  # 模型路径
        self.debug: bool = debug  # 配置文件路径
        self.language_identification_library: str = (
            language_identification_library  # 语种识别库
        )

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        # 将配置文件路径和模型路径添加到数据中
        data["config_path"] = os.path.join(dataset_path, data["config_path"])
        data["model"] = os.path.join(dataset_path, data["model"])
        # 使用数据创建并返回 Webui_config 实例
        return cls(**data)


class Server_config:
    def __init__(
        self, models: List[Dict[str, any]], port: int = 5000, device: str = "cuda"
    ):
        # 初始化 Server_config 类的属性
        self.models: List[Dict[str, any]] = models  # 需要加载的所有模型的配置
        self.port: int = port  # 端口号
        self.device: str = device  # 模型默认使用设备

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        # 使用数据创建并返回 Server_config 实例
        return cls(**data)


class Translate_config:
    """翻译api配置"""

    def __init__(self, app_key: str, secret_key: str):
        # 初始化 Translate_config 类的属性
        self.app_key = app_key
        self.secret_key = secret_key

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        # 使用数据创建并返回 Translate_config 实例
        return cls(**data)


class Config:
    parser = argparse.ArgumentParser()
    # 为避免与以前的config.json起冲突，将其更名如下
    parser.add_argument("-y", "--yml_config", type=str, default="config.yml")
    args, _ = parser.parse_known_args()
    config = Config(args.yml_config)
```