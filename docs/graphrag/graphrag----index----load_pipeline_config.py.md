# `.\graphrag\graphrag\index\load_pipeline_config.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing read_dotenv, load_pipeline_config, _parse_yaml and _create_include_constructor methods definition."""

import json
from pathlib import Path

import yaml
from pyaml_env import parse_config as parse_config_with_env

from graphrag.config import create_graphrag_config, read_dotenv
from graphrag.index.config import PipelineConfig

from .create_pipeline_config import create_pipeline_config


def load_pipeline_config(config_or_path: str | PipelineConfig) -> PipelineConfig:
    """Load a pipeline config from a file path or a config object."""
    # 如果传入的参数是 PipelineConfig 类型的对象，则直接使用
    if isinstance(config_or_path, PipelineConfig):
        config = config_or_path
    # 如果传入的参数是字符串 "default"，则创建一个默认的 pipeline 配置
    elif config_or_path == "default":
        config = create_pipeline_config(create_graphrag_config(root_dir="."))
    else:
        # 检查配置文件所在目录是否有 .env 文件，如果有则读取
        read_dotenv(str(Path(config_or_path).parent))

        # 根据配置文件后缀选择加载方式
        if config_or_path.endswith(".json"):
            with Path(config_or_path).open(encoding="utf-8") as f:
                config = json.load(f)
        elif config_or_path.endswith((".yml", ".yaml")):
            config = _parse_yaml(config_or_path)
        else:
            # 抛出异常，指出配置文件类型不合法
            msg = f"Invalid config file type: {config_or_path}"
            raise ValueError(msg)

        # 对配置对象进行模型验证
        config = PipelineConfig.model_validate(config)
        
        # 如果配置中未指定 root_dir，则使用配置文件所在目录的绝对路径
        if not config.root_dir:
            config.root_dir = str(Path(config_or_path).parent.resolve())

    # 如果配置文件有继承关系，则递归加载继承的配置文件，并进行合并
    if config.extends is not None:
        if isinstance(config.extends, str):
            config.extends = [config.extends]
        for extended_config in config.extends:
            extended_config = load_pipeline_config(extended_config)
            merged_config = {
                **json.loads(extended_config.model_dump_json()),
                **json.loads(config.model_dump_json(exclude_unset=True)),
            }
            config = PipelineConfig.model_validate(merged_config)

    return config


def _parse_yaml(path: str):
    """Parse a yaml file, with support for !include directives."""
    # 使用安全加载器加载 YAML 文件
    loader_class = yaml.SafeLoader

    # 如果加载器中不存在 !include 构造器，则添加
    if "!include" not in loader_class.yaml_constructors:
        loader_class.add_constructor("!include", _create_include_constructor())

    # 调用 pyaml_env 库解析配置文件，并支持环境变量替换
    return parse_config_with_env(path, loader=loader_class, default_value="")


def _create_include_constructor():
    """Create a constructor for !include directives."""

    def handle_include(loader: yaml.Loader, node: yaml.Node):
        """Include file referenced at node."""
        # 计算被引用文件的完整路径
        filename = str(Path(loader.name).parent / node.value)
        
        # 如果被引用文件是 YAML 文件，则调用 _parse_yaml 函数解析
        if filename.endswith((".yml", ".yaml")):
            return _parse_yaml(filename)

        # 否则，直接读取文件内容并返回
        with Path(filename).open(encoding="utf-8") as f:
            return f.read()

    return handle_include
```