# `d:/src/tocomm/Bert-VITS2\config.py`

```
"""
@Desc: 全局配置文件读取
"""
# 导入必要的模块
import argparse  # 用于解析命令行参数
import yaml  # 用于读取和写入 YAML 文件
from typing import Dict, List  # 用于类型提示
import os  # 用于操作文件和目录
import shutil  # 用于高级文件操作
import sys  # 用于访问与 Python 解释器交互的变量和函数


class Resample_config:
    """重采样配置"""

    def __init__(self, in_dir: str, out_dir: str, sampling_rate: int = 44100):
        self.sampling_rate: int = sampling_rate  # 目标采样率
        self.in_dir: str = in_dir  # 待处理音频目录路径
        self.out_dir: str = out_dir  # 重采样输出路径

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        """从字典中生成实例"""
        # 从给定的字典中生成一个实例
        # 不检查路径是否有效，此逻辑在resample.py中处理
        data["in_dir"] = os.path.join(dataset_path, data["in_dir"])  # 将数据字典中的"in_dir"路径与dataset_path拼接成完整路径
        data["out_dir"] = os.path.join(dataset_path, data["out_dir"])  # 将数据字典中的"out_dir"路径与dataset_path拼接成完整路径
        return cls(**data)  # 返回使用给定数据字典生成的实例


class Preprocess_text_config:
    """数据预处理配置"""

    def __init__(
        self,
        transcription_path: str,
        cleaned_path: str,
        train_path: str,
        val_path: str,
        config_path: str,
```
- `def from_dict(cls, dataset_path: str, data: Dict[str, any]):`：定义一个类方法，从字典中生成实例，参数包括类本身、数据集路径和数据字典。
- `"""从字典中生成实例"""`：方法的文档字符串，解释了方法的作用。
- `data["in_dir"] = os.path.join(dataset_path, data["in_dir"])`：将数据字典中的"in_dir"路径与dataset_path拼接成完整路径。
- `data["out_dir"] = os.path.join(dataset_path, data["out_dir"])`：将数据字典中的"out_dir"路径与dataset_path拼接成完整路径。
- `return cls(**data)`：返回使用给定数据字典生成的实例。
        val_per_lang: int = 5,  # 每个语种的验证集条数，默认为5
        max_val_total: int = 10000,  # 验证集最大条数，默认为10000
        clean: bool = True,  # 是否进行数据清洗，默认为True
    ):
        self.transcription_path: str = transcription_path  # 原始文本文件路径，文本格式应为{wav_path}|{speaker_name}|{language}|{text}。
        self.cleaned_path: str = cleaned_path  # 数据清洗后文本路径，可以不填。不填则将在原始文本目录生成
        self.train_path: str = train_path  # 训练集路径，可以不填。不填则将在原始文本目录生成
        self.val_path: str = val_path  # 验证集路径，可以不填。不填则将在原始文本目录生成
        self.config_path: str = config_path  # 配置文件路径
        self.val_per_lang: int = val_per_lang  # 每个speaker的验证集条数
        self.max_val_total: int = max_val_total  # 验证集最大条数，多于的会被截断并放到训练集中
        self.clean: bool = clean  # 是否进行数据清洗

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        """从字典中生成实例"""

        data["transcription_path"] = os.path.join(
            dataset_path, data["transcription_path"]
        )
        # 如果数据中的 cleaned_path 为空或者为 None，则将 cleaned_path 设置为 None
        if data["cleaned_path"] == "" or data["cleaned_path"] is None:
            data["cleaned_path"] = None
        else:
            # 否则，将 cleaned_path 设置为 dataset_path 和 data["cleaned_path"] 的组合路径
            data["cleaned_path"] = os.path.join(dataset_path, data["cleaned_path"])
        # 将 train_path 设置为 dataset_path 和 data["train_path"] 的组合路径
        data["train_path"] = os.path.join(dataset_path, data["train_path"])
        # 将 val_path 设置为 dataset_path 和 data["val_path"] 的组合路径
        data["val_path"] = os.path.join(dataset_path, data["val_path"])
        # 将 config_path 设置为 dataset_path 和 data["config_path"] 的组合路径
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        # 返回一个新的 Bert_gen_config 实例，使用传入的 data 参数
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
        # 初始化 Emo_gen_config 类的实例变量
        self.config_path = config_path  # 设置配置文件路径
        self.num_processes = num_processes  # 设置进程数量
        self.device = device  # 设置设备
        self.use_multi_device = use_multi_device  # 设置是否使用多设备

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        # 将配置文件路径添加到数据字典中
        data["config_path"] = os.path.join(dataset_path, data["config_path"])

        # 使用数据字典创建 Emo_gen_config 类的实例
        return cls(**data)


class Emo_gen_config:
    """emo_gen 配置"""

    def __init__(
        self,
        config_path: str,
        num_processes: int = 2,
```
在这段代码中，我们定义了一个类Emo_gen_config，其中包含了初始化方法__init__和一个类方法from_dict。在__init__方法中，我们初始化了类的实例变量config_path、num_processes、device和use_multi_device。在from_dict方法中，我们从数据字典中获取配置文件路径，并使用数据字典创建Emo_gen_config类的实例。
        device: str = "cuda",  # 设置默认设备为 "cuda"
        use_multi_device: bool = False,  # 设置默认使用单设备

    ):
        self.config_path = config_path  # 初始化配置路径
        self.num_processes = num_processes  # 初始化进程数
        self.device = device  # 初始化设备
        self.use_multi_device = use_multi_device  # 初始化是否使用多设备

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        data["config_path"] = os.path.join(dataset_path, data["config_path"])  # 将配置路径拼接到数据集路径

        return cls(**data)  # 返回根据数据创建的类实例


class Train_ms_config:
    """训练配置"""

    def __init__(
        config_path: str,  # 配置文件路径，数据类型为字符串
        env: Dict[str, any],  # 环境变量，数据类型为字典，键为字符串，值为任意类型
        base: Dict[str, any],  # 底模配置，数据类型为字典，键为字符串，值为任意类型
        model: str,  # 训练模型存储目录，数据类型为字符串
        num_workers: int,  # worker数量，数据类型为整数
        spec_cache: bool,  # 是否启用spec缓存，数据类型为布尔值
        keep_ckpts: int,  # ckpt数量，数据类型为整数
    ):
        self.env = env  # 将传入的环境变量赋值给类的环境变量属性
        self.base = base  # 将传入的底模配置赋值给类的底模配置属性
        self.model = model  # 将传入的训练模型存储目录赋值给类的训练模型存储目录属性
        self.config_path = config_path  # 将传入的配置文件路径赋值给类的配置文件路径属性
        self.num_workers = num_workers  # 将传入的worker数量赋值给类的worker数量属性
        self.spec_cache = spec_cache  # 将传入的是否启用spec缓存赋值给类的是否启用spec缓存属性
        self.keep_ckpts = keep_ckpts  # 将传入的ckpt数量赋值给类的ckpt数量属性

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        # data["model"] = os.path.join(dataset_path, data["model"])  # 将数据字典中的"model"键对应的值与dataset_path拼接成完整的模型路径
        data["config_path"] = os.path.join(dataset_path, data["config_path"])  # 将数据字典中的"config_path"键对应的值与dataset_path拼接成完整的配置文件路径
        return cls(**data)  # 使用传入的数据创建并返回一个类实例


class Webui_config:
    """webui 配置"""

    def __init__(
        self,
        device: str,
        model: str,
        config_path: str,
        language_identification_library: str,
        port: int = 7860,  # 默认端口号为 7860
        share: bool = False,  # 默认不公开部署
        debug: bool = False,  # 默认不开启调试模式
    ):
        self.device: str = device  # 设置设备属性
        self.model: str = model  # 设置模型属性
        self.config_path: str = config_path  # 设置配置文件路径属性
        self.port: int = port  # 设置类的属性port，并指定类型为整数，表示端口号
        self.share: bool = share  # 设置类的属性share，并指定类型为布尔值，表示是否开启共享
        self.debug: bool = debug  # 设置类的属性debug，并指定类型为布尔值，表示是否开启调试模式
        self.language_identification_library: str = (
            language_identification_library  # 设置类的属性language_identification_library，并指定类型为字符串，表示语种识别库
        )

    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        data["config_path"] = os.path.join(dataset_path, data["config_path"])  # 将配置文件路径拼接到数据集路径下
        data["model"] = os.path.join(dataset_path, data["model"])  # 将模型路径拼接到数据集路径下
        return cls(**data)  # 返回一个类的实例，使用传入的数据作为参数

class Server_config:
    def __init__(
        self, models: List[Dict[str, any]], port: int = 5000, device: str = "cuda"
    ):
        self.models: List[Dict[str, any]] = models  # 设置类的属性models，并指定类型为字典列表，表示需要加载的所有模型的配置
        self.port: int = port  # 设置类的属性port，并指定类型为整数，表示端口号
        self.device: str = device  # 模型默认使用设备
        # 设置模型默认使用的设备

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)
        # 从字典数据创建一个类实例

class Translate_config:
    """翻译api配置"""
    # 翻译API的配置类

    def __init__(self, app_key: str, secret_key: str):
        self.app_key = app_key
        self.secret_key = secret_key
        # 初始化翻译API的app_key和secret_key属性

    @classmethod
    def from_dict(cls, data: Dict[str, any]):
        return cls(**data)
        # 从字典数据创建一个翻译配置类实例

class Config:
    # 配置类
    # 初始化函数，接受一个配置文件路径作为参数
    def __init__(self, config_path: str):
        # 如果配置文件路径不是一个文件，并且存在默认配置文件"default_config.yml"
        if not os.path.isfile(config_path) and os.path.isfile("default_config.yml"):
            # 复制默认配置文件"default_config.yml"到指定的配置文件路径
            shutil.copy(src="default_config.yml", dst=config_path)
            # 打印提示信息
            print(
                f"已根据默认配置文件default_config.yml生成配置文件{config_path}。请按该配置文件的说明进行配置后重新运行。"
            )
            print("如无特殊需求，请勿修改default_config.yml或备份该文件。")
            # 退出程序
            sys.exit(0)
        # 打开配置文件
        with open(file=config_path, mode="r", encoding="utf-8") as file:
            # 加载配置文件内容为字典
            yaml_config: Dict[str, any] = yaml.safe_load(file.read())
            # 从配置文件中获取数据集路径和OpenI令牌
            dataset_path: str = yaml_config["dataset_path"]
            openi_token: str = yaml_config["openi_token"]
            # 将数据集路径和OpenI令牌保存为对象的属性
            self.dataset_path: str = dataset_path
            self.mirror: str = yaml_config["mirror"]
            self.openi_token: str = openi_token
            # 从配置文件中获取并创建Resample_config对象和Preprocess_text_config对象
            self.resample_config: Resample_config = Resample_config.from_dict(
                dataset_path, yaml_config["resample"]
            )
            self.preprocess_text_config: Preprocess_text_config = (
                Preprocess_text_config.from_dict(
            dataset_path, yaml_config["preprocess_text"]
        )
        # 从给定的字典中创建一个Bert_gen_config对象
        self.bert_gen_config: Bert_gen_config = Bert_gen_config.from_dict(
            dataset_path, yaml_config["bert_gen"]
        )
        # 从给定的字典中创建一个Emo_gen_config对象
        self.emo_gen_config: Emo_gen_config = Emo_gen_config.from_dict(
            dataset_path, yaml_config["emo_gen"]
        )
        # 从给定的字典中创建一个Train_ms_config对象
        self.train_ms_config: Train_ms_config = Train_ms_config.from_dict(
            dataset_path, yaml_config["train_ms"]
        )
        # 从给定的字典中创建一个Webui_config对象
        self.webui_config: Webui_config = Webui_config.from_dict(
            dataset_path, yaml_config["webui"]
        )
        # 从给定的字典中创建一个Server_config对象
        self.server_config: Server_config = Server_config.from_dict(
            yaml_config["server"]
        )
        # 从给定的字典中创建一个Translate_config对象
        self.translate_config: Translate_config = Translate_config.from_dict(
            yaml_config["translate"]
parser = argparse.ArgumentParser()
# 创建一个参数解析器对象

parser.add_argument("-y", "--yml_config", type=str, default="config.yml")
# 添加一个命令行参数，用于指定配置文件的名称，默认为"config.yml"

args, _ = parser.parse_known_args()
# 解析命令行参数，并返回一个包含解析结果的命名空间对象

config = Config(args.yml_config)
# 使用解析结果中指定的配置文件名称创建一个配置对象
```