# `.\PaddleOCR\test_tipc\supplementary\config.py`

```py
# 导入必要的库
import numpy as np
import os
import sys
import platform
import yaml
import time
import shutil
import paddle
import paddle.distributed as dist
from tqdm import tqdm
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from utils import get_logger, print_dict

# 定义参数解析类，继承自ArgumentParser
class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        # 添加命令行参数
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")
        self.add_argument(
            '-p',
            '--profiler_options',
            type=str,
            default=None,
            help='The option of profiler, which should be in format \"key1=value1;key2=value2;key3=value3\".'
        )

    def parse_args(self, argv=None):
        # 解析命令行参数
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config

# 定义属性字典类，继承自dict
class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))

# 全局配置字典
global_config = AttrDict()

# 默认配置
default_config = {'Global': {'debug': False, }}

# 加载配置文件
def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    # 合并默认配置和全局配置
    merge_config(default_config)
    # 获取文件路径的扩展名
    _, ext = os.path.splitext(file_path)
    # 确保文件扩展名为.yml或.yaml，否则抛出异常
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    # 读取文件内容并将其与全局配置合并
    merge_config(yaml.load(open(file_path, 'rb'), Loader=yaml.Loader))
    # 返回全局配置
    return global_config
# 合并配置到全局配置中
def merge_config(config):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    # 遍历配置字典中的键值对
    for key, value in config.items():
        # 如果键中不包含"."，则直接更新全局配置
        if "." not in key:
            # 如果值是字典类型且键在全局配置中存在，则更新对应键的值
            if isinstance(value, dict) and key in global_config:
                global_config[key].update(value)
            else:
                global_config[key] = value
        else:
            # 如果键中包含"."，则按层级更新全局配置
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in global_config
            ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                global_config.keys(), sub_keys[0])
            cur = global_config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]


# 预处理函数，根据是否训练参数加载配置
def preprocess(is_train=False):
    # 解析命令行参数
    FLAGS = ArgsParser().parse_args()
    profiler_options = FLAGS.profiler_options
    # 加载配置文件
    config = load_config(FLAGS.config)
    # 合并命令行参数中的配置
    merge_config(FLAGS.opt)
    profile_dic = {"profiler_options": FLAGS.profiler_options}
    # 合并性能分析配置
    merge_config(profile_dic)

    if is_train:
        # 如果是训练模式，保存配置文件
        save_model_dir = config['save_model_dir']
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, 'config.yml'), 'w') as f:
            yaml.dump(
                dict(config), f, default_flow_style=False, sort_keys=False)
        log_file = '{}/train.log'.format(save_model_dir)
    else:
        log_file = None
    logger = get_logger(log_file=log_file)

    # 检查是否在CPU版本中设置了use_gpu=True
    use_gpu = config['use_gpu']

    # 打印配置信息
    print_dict(config, logger)

    return config, logger


if __name__ == "__main__":
    # 预处理并获取配置和日志对象
    config, logger = preprocess(is_train=False)
    # print(config)
```