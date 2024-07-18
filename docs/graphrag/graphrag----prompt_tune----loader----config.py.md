# `.\graphrag\graphrag\prompt_tune\loader\config.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Config loading, parsing and handling module."""

# 从 pathlib 库中导入 Path 类
from pathlib import Path

# 从 graphrag.config 模块中导入 create_graphrag_config 函数
from graphrag.config import create_graphrag_config

# 从 graphrag.index.progress.types 模块中导入 ProgressReporter 类
from graphrag.index.progress.types import ProgressReporter


# 定义函数 read_config_parameters，用于读取配置参数
def read_config_parameters(root: str, reporter: ProgressReporter):
    """Read the configuration parameters from the settings file or environment variables.

    Parameters
    ----------
    - root: The root directory where the parameters are.
    - reporter: The progress reporter.
    """
    # 将 root 转换为 Path 对象
    _root = Path(root)
    
    # 设置 settings.yaml 文件路径
    settings_yaml = _root / "settings.yaml"
    
    # 如果 settings.yaml 文件不存在，则尝试使用 settings.yml 文件
    if not settings_yaml.exists():
        settings_yaml = _root / "settings.yml"
    
    # 设置 settings.json 文件路径
    settings_json = _root / "settings.json"

    # 如果存在 settings_yaml 文件，则读取其中的配置
    if settings_yaml.exists():
        # 使用 ProgressReporter 对象记录日志，指示正在从 settings_yaml 文件中读取配置
        reporter.info(f"Reading settings from {settings_yaml}")
        # 打开 settings_yaml 文件进行读取
        with settings_yaml.open("r") as file:
            import yaml
            # 使用 yaml.safe_load 安全加载 YAML 文件内容
            data = yaml.safe_load(file)
            # 调用 create_graphrag_config 函数，传入读取的数据和 root 目录，返回配置对象
            return create_graphrag_config(data, root)

    # 如果存在 settings_json 文件，则读取其中的配置
    if settings_json.exists():
        # 使用 ProgressReporter 对象记录日志，指示正在从 settings_json 文件中读取配置
        reporter.info(f"Reading settings from {settings_json}")
        # 打开 settings_json 文件进行读取
        with settings_json.open("r") as file:
            import json
            # 使用 json.loads 将 JSON 文件内容解析为 Python 对象
            data = json.loads(file.read())
            # 调用 create_graphrag_config 函数，传入读取的数据和 root 目录，返回配置对象
            return create_graphrag_config(data, root)

    # 如果以上文件都不存在，则从环境变量中读取配置
    reporter.info("Reading settings from environment variables")
    # 调用 create_graphrag_config 函数，传入 root 目录，返回配置对象
    return create_graphrag_config(root_dir=root)
```