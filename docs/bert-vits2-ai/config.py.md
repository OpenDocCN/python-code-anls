# `Bert-VITS2\config.py`

```

# 导入必要的库
import argparse  # 用于解析命令行参数
import yaml  # 用于读取和写入YAML文件
from typing import Dict, List  # 用于类型提示
import os  # 用于操作文件和目录
import shutil  # 用于复制文件
import sys  # 用于访问与Python解释器交互的变量和函数

# 定义一个重采样配置类
class Resample_config:
    # 初始化方法，设置默认参数
    def __init__(self, in_dir: str, out_dir: str, sampling_rate: int = 44100):
        self.sampling_rate: int = sampling_rate  # 目标采样率
        self.in_dir: str = in_dir  # 待处理音频目录路径
        self.out_dir: str = out_dir  # 重采样输出路径

    # 从字典中生成实例的类方法
    @classmethod
    def from_dict(cls, dataset_path: str, data: Dict[str, any]):
        # 对路径进行处理
        data["in_dir"] = os.path.join(dataset_path, data["in_dir"])
        data["out_dir"] = os.path.join(dataset_path, data["out_dir"])
        return cls(**data)

# 其他配置类的定义和注释与上述相似，这里不再重复注释

# 定义一个配置类
class Config:
    # 初始化方法，根据配置文件路径读取配置信息
    def __init__(self, config_path: str):
        # 如果配置文件不存在，复制默认配置文件并提示用户
        if not os.path.isfile(config_path) and os.path.isfile("default_config.yml"):
            shutil.copy(src="default_config.yml", dst=config_path)
            print("已根据默认配置文件default_config.yml生成配置文件{config_path}。请按该配置文件的说明进行配置后重新运行。")
            print("如无特殊需求，请勿修改default_config.yml或备份该文件。")
            sys.exit(0)
        # 读取配置文件内容
        with open(file=config_path, mode="r", encoding="utf-8") as file:
            yaml_config: Dict[str, any] = yaml.safe_load(file.read())
            dataset_path: str = yaml_config["dataset_path"]
            openi_token: str = yaml_config["openi_token"]
            # 设置配置信息
            self.dataset_path: str = dataset_path
            self.mirror: str = yaml_config["mirror"]
            self.openi_token: str = openi_token
            self.resample_config: Resample_config = Resample_config.from_dict(dataset_path, yaml_config["resample"])
            # 其他配置信息的设置与上述相似，这里不再重复注释

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
# 添加命令行参数选项
parser.add_argument("-y", "--yml_config", type=str, default="config.yml")
# 解析命令行参数
args, _ = parser.parse_known_args()
# 根据命令行参数指定的配置文件路径创建配置对象
config = Config(args.yml_config)

```