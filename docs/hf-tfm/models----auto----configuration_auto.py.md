# `.\transformers\models\auto\configuration_auto.py`

```py
# 设定脚本编码格式为 UTF-8
# 版权声明，使用 Apache 许可证版本 2.0，详见链接
# 引入必要的模块和库
""" Auto Config class."""
import importlib  # 导入模块动态加载所需的库
import os  # 导入操作系统相关功能的库
import re  # 导入正则表达式功能的库
import warnings  # 导入警告处理相关的库
from collections import OrderedDict  # 导入有序字典的类
from typing import List, Union  # 导入类型提示的相关功能

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code  # 导入动态模块加载相关的函数
from ...utils import CONFIG_NAME, logging  # 导入配置文件名和日志记录相关的功能

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

CONFIG_MAPPING_NAMES = OrderedDict(  # 定义模型配置映射的有序字典，用于存储模型类型到配置类的映射关系
    # 空字典，待填充
)

CONFIG_ARCHIVE_MAP_MAPPING_NAMES = OrderedDict(  # 定义模型配置档案映射的有序字典，用于存储配置档案到模型类型的映射关系
    # 空字典，待填充
)

MODEL_NAMES_MAPPING = OrderedDict(  # 定义模型名称映射的有序字典，用于存储模型类型到模型名称的映射关系
    # 空字典，待填充
)

# 下列模型类型会经过处理，将 "-" 替换为 "_"
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

# 某些特殊模型类型的处理方式
SPECIAL_MODEL_TYPE_TO_MODULE_NAME = OrderedDict(
    [
        ("openai-gpt", "openai"),  # "openai-gpt" 映射为 "openai"
        ("data2vec-audio", "data2vec"),  # "data2vec-audio" 映射为 "data2vec"
        ("data2vec-text", "data2vec"),  # "data2vec-text" 映射为 "data2vec"
        ("data2vec-vision", "data2vec"),  # "data2vec-vision" 映射为 "data2vec"
        ("donut-swin", "donut"),  # "donut-swin" 映射为 "donut"
        ("kosmos-2", "kosmos2"),  # "kosmos-2" 映射为 "kosmos2"
        ("maskformer-swin", "maskformer"),  # "maskformer-swin" 映射为 "maskformer"
        ("xclip", "x_clip"),  # "xclip" 映射为 "x_clip"
        ("clip_vision_model", "clip"),  # "clip_vision_model" 映射为 "clip"
        ("siglip_vision_model", "siglip"),  # "siglip_vision_model" 映射为 "siglip"
    ]
)


def model_type_to_module_name(key):
    """Converts a config key to the corresponding module."""
    # 特殊处理
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        return SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]

    # 将 "-" 替换为 "_"
    key = key.replace("-", "_")
    if key in DEPRECATED_MODELS:  # 检查是否为已弃用模型
        key = f"deprecated.{key}"  # 弃用模型添加前缀 "deprecated."

    return key


def config_class_to_model_type(config):
    """Converts a config class name to the corresponding model type"""
    for key, cls in CONFIG_MAPPING_NAMES.items():  # 遍历模型配置映射字典
        if cls == config:  # 检查配置类是否与给定配置相同
            return key  # 返回对应的模型类型
    # 如果在主字典中找不到对应的配置类，则在额外内容中查找
    for key, cls in CONFIG_MAPPING._extra_content.items():
        if cls.__name__ == config:  # 检查额外内容中的类名是否与给定配置相同
            return key  # 返回对应的模型类型
    return None  # 如果找不到对应的模型类型，则返回 None


class _LazyConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping):
        self._mapping = mapping  # 存储原始映射字典
        self._extra_content = {}  # 存储额外的内容
        self._modules = {}  # 存储模块信息
    # 定义特殊方法 __getitem__，用于实现类的索引功能
    def __getitem__(self, key):
        # 如果键在额外内容中，则返回额外内容中的值
        if key in self._extra_content:
            return self._extra_content[key]
        # 如果键不在映射中，则引发 KeyError
        if key not in self._mapping:
            raise KeyError(key)
        # 从映射中获取值
        value = self._mapping[key]
        # 根据键获取对应模型类型的模块名称
        module_name = model_type_to_module_name(key)
        # 如果模块名称不在已加载的模块字典中，则动态导入对应模块
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        # 如果模块中存在对应的属性，则返回该属性的值
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

        # 某些映射具有条目模型类型 -> 另一个模型类型的配置。在这种情况下，我们尝试获取顶层对象。
        # 动态导入 transformers 模块
        transformers_module = importlib.import_module("transformers")
        # 返回指定模块中的指定属性的值
        return getattr(transformers_module, value)

    # 返回映射中所有键的列表，包括额外内容的键
    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    # 返回映射中所有值的列表，包括额外内容的值
    def values(self):
        return [self[k] for k in self._mapping.keys()] + list(self._extra_content.values())

    # 返回映射中所有键值对的列表，包括额外内容的键值对
    def items(self):
        return [(k, self[k]) for k in self._mapping.keys()] + list(self._extra_content.items())

    # 实现迭代功能，返回映射中所有键的迭代器，包括额外内容的键
    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    # 检查给定元素是否存在于映射中或额外内容中
    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    # 向映射中注册新的配置
    def register(self, key, value, exist_ok=False):
        """
        Register a new configuration in this mapping.
        """
        # 如果键已存在于映射中且不允许覆盖，则引发 ValueError
        if key in self._mapping.keys() and not exist_ok:
            raise ValueError(f"'{key}' is already used by a Transformers config, pick another name.")
        # 将键值对添加到额外内容中
        self._extra_content[key] = value
# 使用 LazyConfigMapping 对象初始化 CONFIG_MAPPING 变量，该变量用于将配置映射到模型名称
CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)

# _LazyLoadAllMappings 类定义了一个延迟加载所有映射的字典，它会在首次访问时加载所有键值对
class _LazyLoadAllMappings(OrderedDict):
    """
    A mapping that will load all pairs of key values at the first access (either by indexing, requestions keys, values,
    etc.)

    Args:
        mapping: The mapping to load.
    """

    def __init__(self, mapping):
        # 初始化 LazyLoadAllMappings 类的实例，接受一个映射作为参数
        self._mapping = mapping
        self._initialized = False
        self._data = {}

    def _initialize(self):
        # 如果已经初始化过，则直接返回
        if self._initialized:
            return
        # 发出关于 ALL_PRETRAINED_CONFIG_ARCHIVE_MAP 已过时的警告
        warnings.warn(
            "ALL_PRETRAINED_CONFIG_ARCHIVE_MAP is deprecated and will be removed in v5 of Transformers. "
            "It does not contain all available model checkpoints, far from it. Checkout hf.co/models for that.",
            FutureWarning,
        )

        # 遍历映射，加载所有映射的键值对
        for model_type, map_name in self._mapping.items():
            module_name = model_type_to_module_name(model_type)
            module = importlib.import_module(f".{module_name}", "transformers.models")
            mapping = getattr(module, map_name)
            self._data.update(mapping)

        # 标记已初始化
        self._initialized = True

    def __getitem__(self, key):
        # 初始化后访问字典的值
        self._initialize()
        return self._data[key]

    def keys(self):
        # 初始化后访问字典的键
        self._initialize()
        return self._data.keys()

    def values(self):
        # 初始化后访问字典的值
        self._initialize()
        return self._data.values()

    def items(self):
        # 初始化后访问字典的键值对
        self._initialize()
        return self._data.keys()

    def __iter__(self):
        # 初始化后返回字典的迭代器
        self._initialize()
        return iter(self._data)

    def __contains__(self, item):
        # 初始化后检查字典是否包含指定项
        self._initialize()
        return item in self._data

# ALL_PRETRAINED_CONFIG_ARCHIVE_MAP 是一个 LazyLoadAllMappings 对象，用于加载所有预训练配置的映射
ALL_PRETRAINED_CONFIG_ARCHIVE_MAP = _LazyLoadAllMappings(CONFIG_ARCHIVE_MAP_MAPPING_NAMES)

# _get_class_name 函数用于获取模型类名称的字符串表示
def _get_class_name(model_class: Union[str, List[str]]):
    # 如果模型类是一个列表或元组，则返回列表中每个元素的类名的字符串表示
    if isinstance(model_class, (list, tuple)):
        return " or ".join([f"[`{c}`]" for c in model_class if c is not None])
    # 如果模型类是一个字符串，则返回该字符串的类名的字符串表示
    return f"[`{model_class}`]"

# _list_model_options 函数用于列出模型选项
def _list_model_options(indent, config_to_class=None, use_model_types=True):
    # 如果不使用模型类型且未提供 config_to_class 参数，则抛出 ValueError 异常
    if config_to_class is None and not use_model_types:
        raise ValueError("Using `use_model_types=False` requires a `config_to_class` dictionary.")
    if use_model_types:
        # 如果使用模型类型，则根据 CONFIG_MAPPING_NAMES 创建模型类型到模型类名的映射
        if config_to_class is None:
            model_type_to_name = {model_type: f"[`{config}`]" for model_type, config in CONFIG_MAPPING_NAMES.items()}
        else:
            model_type_to_name = {
                model_type: _get_class_name(model_class)
                for model_type, model_class in config_to_class.items()
                if model_type in MODEL_NAMES_MAPPING
            }
        # 构建列出模型选项的行列表
        lines = [
            f"{indent}- **{model_type}** -- {model_type_to_name[model_type]} ({MODEL_NAMES_MAPPING[model_type]} model)"
            for model_type in sorted(model_type_to_name.keys())
        ]
    # 如果条件不成立，执行以下代码块
    else:
        # 创建一个字典，将配置映射名称与类名对应起来
        config_to_name = {
            CONFIG_MAPPING_NAMES[config]: _get_class_name(clas)
            for config, clas in config_to_class.items()
            if config in CONFIG_MAPPING_NAMES
        }
        # 创建一个字典，将配置与模型名称对应起来
        config_to_model_name = {
            config: MODEL_NAMES_MAPPING[model_type] for model_type, config in CONFIG_MAPPING_NAMES.items()
        }
        # 生成包含配置信息的列表
        lines = [
            f"{indent}- [`{config_name}`] configuration class:"
            f" {config_to_name[config_name]} ({config_to_model_name[config_name]} model)"
            for config_name in sorted(config_to_name.keys())
        ]
    # 将列表中的元素用换行符连接成一个字符串并返回
    return "\n".join(lines)
# 用于替换文档字符串中的列表选项内容，并根据给定参数生成新的装饰器函数
def replace_list_option_in_docstrings(config_to_class=None, use_model_types=True):
    # 定义装饰器函数
    def docstring_decorator(fn):
        # 获取函数的文档字符串
        docstrings = fn.__doc__
        # 将文档字符串按行分割成列表
        lines = docstrings.split("\n")
        # 初始化索引
        i = 0
        # 查找以List options开头的行
        while i < len(lines) and re.search(r"^(\s*)List options\s*$", lines[i]) is None:
            i += 1
        # 如果找到了以List options开头的行
        if i < len(lines):
            # 获取该行的缩进
            indent = re.search(r"^(\s*)List options\s*$", lines[i]).groups()[0]
            # 如果使用模型类型，则在缩进后添加额外的缩进
            if use_model_types:
                indent = f"{indent}    "
            # 将该行替换为新的列表选项内容
            lines[i] = _list_model_options(indent, config_to_class=config_to_class, use_model_types=use_model_types)
            # 更新文档字符串
            docstrings = "\n".join(lines)
        else:
            # 如果未找到以List options开头的行，则抛出异常
            raise ValueError(
                f"The function {fn} should have an empty 'List options' in its docstring as placeholder, current"
                f" docstring is:\n{docstrings}"
            )
        # 更新函数的文档字符串
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


# 自动配置类
class AutoConfig:
    r"""
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the [`~AutoConfig.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    # 初始化方法，抛出环境错误异常
    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    # 根据模型类型注册配置的类方法
    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        # 检查给定模型类型是否在配置映射中
        if model_type in CONFIG_MAPPING:
            # 获取对应的配置类
            config_class = CONFIG_MAPPING[model_type]
            # 使用给定参数实例化配置对象并返回
            return config_class(*args, **kwargs)
        # 如果模型类型未在配置映射中，则抛出异常
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
        )

    # 注册新配置的静态方法，并替换文档字符串中的列表选项内容
    @classmethod
    @replace_list_option_in_docstrings()
    @staticmethod
    def register(model_type, config, exist_ok=False):
        """
        Register a new configuration for this class.

        Args:
            model_type (`str`): The model type like "bert" or "gpt".
            config ([`PretrainedConfig`]): The config to register.
        """
        # 检查注册的配置类是否是预训练配置的子类，并且其模型类型与给定模型类型一致
        if issubclass(config, PretrainedConfig) and config.model_type != model_type:
            raise ValueError(
                "The config you are passing has a `model_type` attribute that is not consistent with the model type "
                f"you passed (config has {config.model_type} and you passed {model_type}. Fix one of those so they "
                "match!"
            )
        # 将配置类注册到配置映射中
        CONFIG_MAPPING.register(model_type, config, exist_ok=exist_ok)
```