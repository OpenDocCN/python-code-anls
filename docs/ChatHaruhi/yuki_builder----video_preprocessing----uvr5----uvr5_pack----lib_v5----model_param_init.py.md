# `.\Chat-Haruhi-Suzumiya\yuki_builder\video_preprocessing\uvr5\uvr5_pack\lib_v5\model_param_init.py`

```py
# 导入必要的模块：json 用于 JSON 数据处理，os 用于操作系统相关功能，pathlib 用于处理路径操作
import json
import os
import pathlib

# 定义默认参数字典
default_param = {}

# 设置默认参数的值
default_param["bins"] = 768
default_param["unstable_bins"] = 9  # 仅用于训练
default_param["reduction_bins"] = 762  # 仅用于训练
default_param["sr"] = 44100
default_param["pre_filter_start"] = 757
default_param["pre_filter_stop"] = 768
default_param["band"] = {}

# 设置 band 参数中的第一个子字典
default_param["band"][1] = {
    "sr": 11025,
    "hl": 128,
    "n_fft": 960,
    "crop_start": 0,
    "crop_stop": 245,
    "lpf_start": 61,  # 仅用于推断
    "res_type": "polyphase",
}

# 设置 band 参数中的第二个子字典
default_param["band"][2] = {
    "sr": 44100,
    "hl": 512,
    "n_fft": 1536,
    "crop_start": 24,
    "crop_stop": 547,
    "hpf_start": 81,  # 仅用于推断
    "res_type": "sinc_best",
}

# 定义一个函数 int_keys，将参数字典中的键转换为整数类型
def int_keys(d):
    r = {}
    for k, v in d:
        if k.isdigit():
            k = int(k)
        r[k] = v
    return r

# 定义一个类 ModelParameters
class ModelParameters(object):
    # 初始化方法，根据配置文件路径加载参数
    def __init__(self, config_path=""):
        # 如果配置文件路径以 ".pth" 结尾，表示这是一个压缩文件
        if ".pth" == pathlib.Path(config_path).suffix:
            import zipfile

            # 使用 zipfile 模块打开压缩文件
            with zipfile.ZipFile(config_path, "r") as zip:
                # 从压缩文件中读取并解析 JSON 数据，使用 int_keys 函数处理键
                self.param = json.loads(
                    zip.read("param.json"), object_pairs_hook=int_keys
                )
        # 如果配置文件路径以 ".json" 结尾，表示这是一个 JSON 文件
        elif ".json" == pathlib.Path(config_path).suffix:
            # 直接打开 JSON 文件并解析其中的内容，使用 int_keys 函数处理键
            with open(config_path, "r") as f:
                self.param = json.loads(f.read(), object_pairs_hook=int_keys)
        else:
            # 如果配置文件路径不是以上两种格式，则使用默认参数
            self.param = default_param

        # 针对一些特定的键检查其是否存在于参数字典中，如果不存在则添加默认值 False
        for k in [
            "mid_side",
            "mid_side_b",
            "mid_side_b2",
            "stereo_w",
            "stereo_n",
            "reverse",
        ]:
            if not k in self.param:
                self.param[k] = False
```