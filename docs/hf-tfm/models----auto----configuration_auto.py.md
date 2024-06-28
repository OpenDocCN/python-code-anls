# `.\models\auto\configuration_auto.py`

```
# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Auto Config class."""

# 导入标准库和第三方模块
import importlib
import os
import re
import warnings
from collections import OrderedDict
from typing import List, Union

# 导入自定义模块
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...utils import CONFIG_NAME, logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义用于配置映射、模型映射和归档映射的有序字典
CONFIG_MAPPING_NAMES = OrderedDict(
    []
)

CONFIG_ARCHIVE_MAP_MAPPING_NAMES = OrderedDict(
    []
)

MODEL_NAMES_MAPPING = OrderedDict(
    []
)

# 被废弃的模型类型列表，需要将 "-" 转换为 "_"
DEPRECATED_MODELS = [
    "bort",
    "mctct",
    "mmbt",
    "open_llama",
    "retribert",
    "tapex",
    "trajectory_transformer",
    "transfo_xl",
    "van",
]

# 特殊模型类型到模块名的映射
SPECIAL_MODEL_TYPE_TO_MODULE_NAME = OrderedDict(
    [
        ("openai-gpt", "openai"),
        ("data2vec-audio", "data2vec"),
        ("data2vec-text", "data2vec"),
        ("data2vec-vision", "data2vec"),
        ("donut-swin", "donut"),
        ("kosmos-2", "kosmos2"),
        ("maskformer-swin", "maskformer"),
        ("xclip", "x_clip"),
        ("clip_vision_model", "clip"),
        ("siglip_vision_model", "siglip"),
        ("chinese_clip_vision_model", "chinese_clip"),
    ]
)

def model_type_to_module_name(key):
    """Converts a config key to the corresponding module."""
    # 特殊模型类型的特殊处理
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        return SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]

    # 将 "-" 转换为 "_"，处理被废弃的模型类型
    key = key.replace("-", "_")
    if key in DEPRECATED_MODELS:
        key = f"deprecated.{key}"

    return key

def config_class_to_model_type(config):
    """Converts a config class name to the corresponding model type"""
    # 在 CONFIG_MAPPING_NAMES 中查找与 config 类相匹配的键
    for key, cls in CONFIG_MAPPING_NAMES.items():
        if cls == config:
            return key
    # 如果在 CONFIG_MAPPING_NAMES 中找不到，则在额外内容中查找
    for key, cls in CONFIG_MAPPING._extra_content.items():
        if cls.__name__ == config:
            return key
    return None

class _LazyConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    # 这里可以添加 LazyConfigMapping 类的其他方法，但未提供在这个片段中
    # 定义魔术方法 __getitem__()，实现索引操作
    def __getitem__(self, key):
        # 如果键存在于额外内容中，则返回额外内容中对应的值
        if key in self._extra_content:
            return self._extra_content[key]
        # 如果键不存在于映射中，则引发 KeyError 异常
        if key not in self._mapping:
            raise KeyError(key)
        # 获取键在映射中对应的值
        value = self._mapping[key]
        # 根据键获取模型类型对应的模块名
        module_name = model_type_to_module_name(key)
        # 如果模块名不在已加载的模块集合中，则动态导入对应模块
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        # 如果模块中存在对应的属性，则返回该属性值
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

        # 某些映射可能指向另一个模型类型的配置对象，此时尝试获取顶层对象
        transformers_module = importlib.import_module("transformers")
        return getattr(transformers_module, value)

    # 返回映射的键列表和额外内容的键列表的合并
    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    # 返回映射的值列表和额外内容的值列表的合并
    def values(self):
        return [self[k] for k in self._mapping.keys()] + list(self._extra_content.values())

    # 返回映射的键值对列表和额外内容的键值对列表的合并
    def items(self):
        return [(k, self[k]) for k in self._mapping.keys()] + list(self._extra_content.items())

    # 返回一个迭代器，迭代器包含映射的键和额外内容的键
    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    # 检查给定的项是否存在于映射或额外内容中
    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    # 将新的配置注册到映射中
    def register(self, key, value, exist_ok=False):
        """
        Register a new configuration in this mapping.
        """
        # 如果键已经存在于映射中且不允许覆盖，则引发 ValueError 异常
        if key in self._mapping.keys() and not exist_ok:
            raise ValueError(f"'{key}' is already used by a Transformers config, pick another name.")
        # 否则将键值对添加到额外内容中
        self._extra_content[key] = value
CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)
# 创建一个懒加载的配置映射对象，根据给定的配置映射名称列表 CONFIG_MAPPING_NAMES

class _LazyLoadAllMappings(OrderedDict):
    """
    A mapping that will load all pairs of key values at the first access (either by indexing, requestions keys, values,
    etc.)

    Args:
        mapping: The mapping to load.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._initialized = False  # 初始化标志位，表示映射是否已经初始化
        self._data = {}  # 存储加载后的映射数据的字典

    def _initialize(self):
        if self._initialized:  # 如果已经初始化过，则直接返回
            return
        warnings.warn(
            "ALL_PRETRAINED_CONFIG_ARCHIVE_MAP is deprecated and will be removed in v5 of Transformers. "
            "It does not contain all available model checkpoints, far from it. Checkout hf.co/models for that.",
            FutureWarning,
        )

        # 遍历配置映射，加载模块并更新数据字典
        for model_type, map_name in self._mapping.items():
            module_name = model_type_to_module_name(model_type)  # 获取模块名称
            module = importlib.import_module(f".{module_name}", "transformers.models")  # 动态导入模块
            mapping = getattr(module, map_name)  # 获取模块中的映射
            self._data.update(mapping)  # 更新数据字典

        self._initialized = True  # 设置初始化标志为 True，表示已完成初始化

    def __getitem__(self, key):
        self._initialize()  # 确保初始化完成
        return self._data[key]  # 返回指定键的值

    def keys(self):
        self._initialize()  # 确保初始化完成
        return self._data.keys()  # 返回所有键的视图

    def values(self):
        self._initialize()  # 确保初始化完成
        return self._data.values()  # 返回所有值的视图

    def items(self):
        self._initialize()  # 确保初始化完成
        return self._data.keys()  # 返回所有键-值对的视图

    def __iter__(self):
        self._initialize()  # 确保初始化完成
        return iter(self._data)  # 返回迭代器，用于迭代所有键

    def __contains__(self, item):
        self._initialize()  # 确保初始化完成
        return item in self._data  # 检查指定项是否在数据字典中


ALL_PRETRAINED_CONFIG_ARCHIVE_MAP = _LazyLoadAllMappings(CONFIG_ARCHIVE_MAP_MAPPING_NAMES)
# 创建一个懒加载的预训练配置存档映射对象，根据给定的配置映射名称列表 CONFIG_ARCHIVE_MAP_MAPPING_NAMES


def _get_class_name(model_class: Union[str, List[str]]):
    if isinstance(model_class, (list, tuple)):
        return " or ".join([f"[`{c}`]" for c in model_class if c is not None])
    return f"[`{model_class}`]"
# 返回格式化的模型类名称字符串，接受字符串或字符串列表参数


def _list_model_options(indent, config_to_class=None, use_model_types=True):
    if config_to_class is None and not use_model_types:
        raise ValueError("Using `use_model_types=False` requires a `config_to_class` dictionary.")
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {model_type: f"[`{config}`]" for model_type, config in CONFIG_MAPPING_NAMES.items()}
        else:
            model_type_to_name = {
                model_type: _get_class_name(model_class)
                for model_type, model_class in config_to_class.items()
                if model_type in MODEL_NAMES_MAPPING
            }
        lines = [
            f"{indent}- **{model_type}** -- {model_type_to_name[model_type]} ({MODEL_NAMES_MAPPING[model_type]} model)"
            for model_type in sorted(model_type_to_name.keys())
        ]
        # 构建模型选项列表，包括模型类型名称和关联的模型类名称
    else:
        # 创建一个字典，将配置映射到类名
        config_to_name = {
            CONFIG_MAPPING_NAMES[config]: _get_class_name(clas)
            for config, clas in config_to_class.items()
            if config in CONFIG_MAPPING_NAMES
        }
        # 创建另一个字典，将配置映射到模型名称
        config_to_model_name = {
            config: MODEL_NAMES_MAPPING[model_type] for model_type, config in CONFIG_MAPPING_NAMES.items()
        }
        # 生成包含配置信息的行列表
        lines = [
            # 每行格式为："- [`配置名`] configuration class: 类名 (模型名称 model)"
            f"{indent}- [`{config_name}`] configuration class: {config_to_name[config_name]} ({config_to_model_name[config_name]} model)"
            for config_name in sorted(config_to_name.keys())
        ]
    # 返回以换行符连接的行字符串
    return "\n".join(lines)
# 定义一个装饰器函数，用于替换函数的文档字符串中的特定部分，以生成新的文档字符串
def replace_list_option_in_docstrings(config_to_class=None, use_model_types=True):
    # 实际的装饰器函数，接受被装饰的函数 fn 作为参数
    def docstring_decorator(fn):
        # 获取函数 fn 的文档字符串
        docstrings = fn.__doc__
        # 将文档字符串按行分割成列表
        lines = docstrings.split("\n")
        i = 0
        # 查找以指定格式开始的行，以定位到“List options”部分
        while i < len(lines) and re.search(r"^(\s*)List options\s*$", lines[i]) is None:
            i += 1
        # 如果找到了符合格式的行
        if i < len(lines):
            # 提取缩进信息，用于替换“List options”部分
            indent = re.search(r"^(\s*)List options\s*$", lines[i]).groups()[0]
            # 如果 use_model_types 为真，追加额外的缩进
            if use_model_types:
                indent = f"{indent}    "
            # 替换文档字符串中的“List options”部分为具体内容
            lines[i] = _list_model_options(indent, config_to_class=config_to_class, use_model_types=use_model_types)
            # 更新修改后的文档字符串
            docstrings = "\n".join(lines)
        else:
            # 如果未找到符合格式的行，则抛出异常
            raise ValueError(
                f"The function {fn} should have an empty 'List options' in its docstring as placeholder, current"
                f" docstring is:\n{docstrings}"
            )
        # 将更新后的文档字符串赋回给函数的 __doc__ 属性
        fn.__doc__ = docstrings
        # 返回经装饰后的函数
        return fn

    # 返回装饰器函数本身
    return docstring_decorator


# 定义一个配置类 AutoConfig
class AutoConfig:
    r"""
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the [`~AutoConfig.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    # 禁止直接实例化该类，抛出环境错误异常
    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    # 类方法，根据 model_type 返回相应的配置类实例
    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        # 如果 model_type 在 CONFIG_MAPPING 中注册过，则返回相应的配置类实例
        if model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        # 如果 model_type 未注册，则抛出值错误异常
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
        )

    # 静态方法，用于注册新的配置
    @classmethod
    @replace_list_option_in_docstrings()  # 应用装饰器，替换文档字符串中的“List options”部分
    def register(model_type, config, exist_ok=False):
        """
        Register a new configuration for this class.

        Args:
            model_type (`str`): The model type like "bert" or "gpt".
            config ([`PretrainedConfig`]): The config to register.
        """
        # 如果 config 是 PretrainedConfig 的子类且其 model_type 不与传入的 model_type 一致，则抛出值错误异常
        if issubclass(config, PretrainedConfig) and config.model_type != model_type:
            raise ValueError(
                "The config you are passing has a `model_type` attribute that is not consistent with the model type "
                f"you passed (config has {config.model_type} and you passed {model_type}. Fix one of those so they "
                "match!"
            )
        # 调用 CONFIG_MAPPING 的 register 方法注册新的配置
        CONFIG_MAPPING.register(model_type, config, exist_ok=exist_ok)
```