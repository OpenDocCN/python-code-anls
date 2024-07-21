# `.\pytorch\docs\source\scripts\build_quantization_configs.py`

```py
"""
This script will generate default values of quantization configs.
These are for use in the documentation.
"""

# 导入必要的库
import os.path
import torch
from torch.ao.quantization.backend_config import get_native_backend_config_dict
from torch.ao.quantization.backend_config.utils import (
    entry_to_pretty_str,
    remove_boolean_dispatch_from_name,
)

# 创建存放图片的目录，如果不存在的话
QUANTIZATION_BACKEND_CONFIG_IMAGE_PATH = os.path.join(
    os.path.realpath(os.path.join(__file__, "..")), "quantization_backend_configs"
)

if not os.path.exists(QUANTIZATION_BACKEND_CONFIG_IMAGE_PATH):
    os.mkdir(QUANTIZATION_BACKEND_CONFIG_IMAGE_PATH)

# 指定输出文件路径
output_path = os.path.join(
    QUANTIZATION_BACKEND_CONFIG_IMAGE_PATH, "default_backend_config.txt"
)

# 打开文件准备写入
with open(output_path, "w") as f:
    # 获取本地后端配置字典
    native_backend_config_dict = get_native_backend_config_dict()
    configs = native_backend_config_dict["configs"]

    # 定义排序函数，用于对配置进行排序
    def _sort_key_func(entry):
        pattern = entry["pattern"]
        while isinstance(pattern, tuple):
            pattern = pattern[-1]

        # 移除布尔分发的名称部分
        pattern = remove_boolean_dispatch_from_name(pattern)
        if not isinstance(pattern, str):
            pattern = torch.typename(pattern)

        # 标准化模式字符串以进行比较
        pattern_str_normalized = pattern.lower().replace("_", "")
        key = pattern_str_normalized.split(".")[-1]
        return key

    # 根据排序函数对配置进行排序
    configs.sort(key=_sort_key_func)

    entries = []
    # 转换每个配置条目为格式化字符串
    for entry in configs:
        entries.append(entry_to_pretty_str(entry))
    
    # 将格式化后的配置条目以逗号分隔写入文件
    entries = ",\n".join(entries)
    f.write(entries)
```