# `.\models\auto\auto_factory.py`

```
# 设置编码为 UTF-8，确保可以正确处理中文和其他特殊字符
# Copyright 2021 The HuggingFace Inc. team.
# 根据 Apache License, Version 2.0 授权许可，进行版权声明和许可信息的设置

# 导入必要的模块和函数
import copy  # 导入 copy 模块，用于对象的深拷贝操作
import importlib  # 导入 importlib 模块，用于动态导入模块和类
import json  # 导入 json 模块，用于 JSON 数据的序列化和反序列化
import os  # 导入 os 模块，用于操作系统相关功能的访问
import warnings  # 导入 warnings 模块，用于警告的处理
from collections import OrderedDict  # 从 collections 模块导入 OrderedDict 类，用于有序字典的创建

# 从其他模块中导入必要的函数和类
from ...configuration_utils import PretrainedConfig  # 导入 PretrainedConfig 类，用于预训练模型配置管理
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code  # 导入动态模块工具函数
from ...utils import (
    CONFIG_NAME,  # 从 utils 模块导入 CONFIG_NAME 常量，用于配置文件名
    cached_file,  # 导入 cached_file 函数，用于缓存文件处理
    copy_func,  # 导入 copy_func 函数，用于函数的复制
    extract_commit_hash,  # 导入 extract_commit_hash 函数，用于提取提交哈希值
    find_adapter_config_file,  # 导入 find_adapter_config_file 函数，用于查找适配器配置文件
    is_peft_available,  # 导入 is_peft_available 函数，用于检查 PEFT 是否可用
    logging,  # 导入 logging 模块，用于日志记录
    requires_backends,  # 导入 requires_backends 装饰器，用于声明后端依赖
)

# 从当前模块的子模块中导入必要的类和函数
from .configuration_auto import (
    AutoConfig,  # 导入 AutoConfig 类，用于自动配置模型
    model_type_to_module_name,  # 导入 model_type_to_module_name 函数，用于模型类型到模块名的映射
    replace_list_option_in_docstrings,  # 导入 replace_list_option_in_docstrings 函数，用于替换文档字符串中的列表选项
)

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义类的文档字符串，描述了一个通用模型类的用途和创建方法
CLASS_DOCSTRING = """
    This is a generic model class that will be instantiated as one of the model classes of the library when created
    with the [`~BaseAutoModelClass.from_pretrained`] class method or the [`~BaseAutoModelClass.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
"""
# 多行字符串，包含用于从配置文件实例化模型类的文档字符串
FROM_CONFIG_DOCSTRING = """
        Instantiates one of the model classes of the library from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights. It only affects the
            model's configuration. Use [`~BaseAutoModelClass.from_pretrained`] to load the model weights.

        Args:
            config ([`PretrainedConfig`]):
                The model class to instantiate is selected based on the configuration class:

                List options
            attn_implementation (`str`, *optional*):
                The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

        Examples:

        ```python
        >>> from transformers import AutoConfig, BaseAutoModelClass

        >>> # Download configuration from huggingface.co and cache.
        >>> config = AutoConfig.from_pretrained("checkpoint_placeholder")
        >>> model = BaseAutoModelClass.from_config(config)
        ```
"""

# 空行

# 空行

"""
BaseAutoModelClass is a base class for auto models. It provides functionality to select and instantiate a model class based on a configuration.
"""

def _get_model_class(config, model_mapping):
    # 获取与给定配置相对应的模型类
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    # 创建模型名称到模型类的映射字典
    name_to_model = {model.__name__: model for model in supported_models}
    # 从配置中获取架构信息
    architectures = getattr(config, "architectures", [])
    # 遍历架构信息，尝试匹配模型类
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]
        elif f"TF{arch}" in name_to_model:
            return name_to_model[f"TF{arch}"]
        elif f"Flax{arch}" in name_to_model:
            return name_to_model[f"Flax{arch}"]

    # 如果配置中未设置架构或未匹配到支持的模型类，则返回元组的第一个元素作为默认模型类
    return supported_models[0]

# 空行

class _BaseAutoModelClass:
    # BaseAutoModelClass 是自动模型的基类。

    # 类变量，用于存储模型映射信息
    _model_mapping = None

    def __init__(self, *args, **kwargs):
        # 抛出环境错误，提示应使用 from_pretrained 或 from_config 方法实例化模型类
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    # 从配置中创建一个模型实例的类方法
    def from_config(cls, config, **kwargs):
        # 从 kwargs 中弹出 trust_remote_code 参数，若无则设为 None
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        # 检查 config 是否具有 auto_map 属性，并且 cls.__name__ 是否在 auto_map 中
        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        # 检查 config 的类型是否在 cls._model_mapping 字典的键中
        has_local_code = type(config) in cls._model_mapping.keys()
        # 解析 trust_remote_code 参数，确定是否信任远程代码
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, config._name_or_path, has_local_code, has_remote_code
        )

        # 如果存在远程代码并且信任远程代码
        if has_remote_code and trust_remote_code:
            # 从 config.auto_map 中获取类引用
            class_ref = config.auto_map[cls.__name__]
            # 若 class_ref 包含 "--" 分隔符，则分割出 repo_id 和 class_ref
            if "--" in class_ref:
                repo_id, class_ref = class_ref.split("--")
            else:
                # 否则 repo_id 设为 config.name_or_path
                repo_id = config.name_or_path
            # 通过动态模块获取类对象 model_class
            model_class = get_class_from_dynamic_module(class_ref, repo_id, **kwargs)
            # 如果 config._name_or_path 是目录，则将 model_class 注册为自动类
            if os.path.isdir(config._name_or_path):
                model_class.register_for_auto_class(cls.__name__)
            else:
                # 否则使用 cls.register 方法注册 model_class
                cls.register(config.__class__, model_class, exist_ok=True)
            # 从 kwargs 中弹出 code_revision 参数，但不使用其值
            _ = kwargs.pop("code_revision", None)
            # 调用 model_class 的 _from_config 方法，返回结果
            return model_class._from_config(config, **kwargs)
        # 如果 config 的类型在 cls._model_mapping 字典的键中
        elif type(config) in cls._model_mapping.keys():
            # 从 _model_mapping 中获取对应的 model_class 类对象
            model_class = _get_model_class(config, cls._model_mapping)
            # 调用 model_class 的 _from_config 方法，返回结果
            return model_class._from_config(config, **kwargs)

        # 如果以上条件都不满足，则抛出 ValueError 异常
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    # 用于注册新模型类的类方法
    def register(cls, config_class, model_class, exist_ok=False):
        """
        Register a new model for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            model_class ([`PreTrainedModel`]):
                The model to register.
        """
        # 如果 model_class 具有 config_class 属性且不等于 config_class 参数，则引发 ValueError 异常
        if hasattr(model_class, "config_class") and model_class.config_class != config_class:
            raise ValueError(
                "The model class you are passing has a `config_class` attribute that is not consistent with the "
                f"config class you passed (model has {model_class.config_class} and you passed {config_class}. Fix "
                "one of those so they match!"
            )
        # 调用 _model_mapping 的 register 方法注册 config_class 和 model_class
        cls._model_mapping.register(config_class, model_class, exist_ok=exist_ok)
class _BaseAutoBackboneClass(_BaseAutoModelClass):
    # Base class for auto backbone models.
    _model_mapping = None

    @classmethod
    def _load_timm_backbone_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Ensure required backends are available
        requires_backends(cls, ["vision", "timm"])
        # Import TimmBackboneConfig from specific module
        from ...models.timm_backbone import TimmBackboneConfig

        # Set default configuration or use provided `config`
        config = kwargs.pop("config", TimmBackboneConfig())

        # Check for disallowed arguments
        if kwargs.get("out_features", None) is not None:
            raise ValueError("Cannot specify `out_features` for timm backbones")
        if kwargs.get("output_loading_info", False):
            raise ValueError("Cannot specify `output_loading_info=True` when loading from timm")

        # Set configuration parameters based on kwargs or defaults
        num_channels = kwargs.pop("num_channels", config.num_channels)
        features_only = kwargs.pop("features_only", config.features_only)
        use_pretrained_backbone = kwargs.pop("use_pretrained_backbone", config.use_pretrained_backbone)
        out_indices = kwargs.pop("out_indices", config.out_indices)

        # Create TimmBackboneConfig object with specified parameters
        config = TimmBackboneConfig(
            backbone=pretrained_model_name_or_path,
            num_channels=num_channels,
            features_only=features_only,
            use_pretrained_backbone=use_pretrained_backbone,
            out_indices=out_indices,
        )

        # Call superclass method `from_config` with the constructed config
        return super().from_config(config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Determine if timm backbone should be used
        use_timm_backbone = kwargs.pop("use_timm_backbone", False)
        
        # If `use_timm_backbone` is True, invoke `_load_timm_backbone_from_pretrained`
        if use_timm_backbone:
            return cls._load_timm_backbone_from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Otherwise, call superclass method `from_pretrained`
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


def insert_head_doc(docstring, head_doc=""):
    # Replace part of the docstring based on presence of `head_doc`
    if len(head_doc) > 0:
        return docstring.replace(
            "one of the model classes of the library ",
            f"one of the model classes of the library (with a {head_doc} head) ",
        )
    else:
        return docstring.replace(
            "one of the model classes of the library ", "one of the base model classes of the library "
        )


def auto_class_update(cls, checkpoint_for_example="google-bert/bert-base-cased", head_doc=""):
    # Create a new class with updated documentation based on `head_doc`
    model_mapping = cls._model_mapping
    name = cls.__name__
    class_docstring = insert_head_doc(CLASS_DOCSTRING, head_doc=head_doc)
    
    # Replace `BaseAutoModelClass` with current class name in class docstring
    cls.__doc__ = class_docstring.replace("BaseAutoModelClass", name)

    # Copy `from_config` method from superclass `_BaseAutoModelClass`
    from_config = copy_func(_BaseAutoModelClass.from_config)
    # Update docstring of `from_config` method based on `head_doc`
    from_config_docstring = insert_head_doc(FROM_CONFIG_DOCSTRING, head_doc=head_doc)
    from_config_docstring = from_config_docstring.replace("BaseAutoModelClass", name)
    # 将 from_config_docstring 中的 "checkpoint_placeholder" 替换为示例中的 checkpoint_for_example 变量值
    from_config_docstring = from_config_docstring.replace("checkpoint_placeholder", checkpoint_for_example)
    # 将 from_config 的文档字符串设为替换后的 from_config_docstring
    from_config.__doc__ = from_config_docstring
    # 使用 model_mapping._model_mapping 中的信息替换 from_config 中的列表选项
    from_config = replace_list_option_in_docstrings(model_mapping._model_mapping, use_model_types=False)(from_config)
    # 将 from_config 方法设为类方法
    cls.from_config = classmethod(from_config)

    # 根据模型名称选择合适的 from_pretrained_docstring
    if name.startswith("TF"):
        from_pretrained_docstring = FROM_PRETRAINED_TF_DOCSTRING
    elif name.startswith("Flax"):
        from_pretrained_docstring = FROM_PRETRAINED_FLAX_DOCSTRING
    else:
        from_pretrained_docstring = FROM_PRETRAINED_TORCH_DOCSTRING
    # 复制 _BaseAutoModelClass.from_pretrained 方法为 from_pretrained
    from_pretrained = copy_func(_BaseAutoModelClass.from_pretrained)
    # 插入 head_doc 到 from_pretrained_docstring 的头部
    from_pretrained_docstring = insert_head_doc(from_pretrained_docstring, head_doc=head_doc)
    # 替换 from_pretrained_docstring 中的类名和占位符
    from_pretrained_docstring = from_pretrained_docstring.replace("BaseAutoModelClass", name)
    from_pretrained_docstring = from_pretrained_docstring.replace("checkpoint_placeholder", checkpoint_for_example)
    # 从 checkpoint_for_example 的路径中提取快捷名称
    shortcut = checkpoint_for_example.split("/")[-1].split("-")[0]
    from_pretrained_docstring = from_pretrained_docstring.replace("shortcut_placeholder", shortcut)
    # 将 from_pretrained 方法的文档字符串设为替换后的 from_pretrained_docstring
    from_pretrained.__doc__ = from_pretrained_docstring
    # 使用 model_mapping._model_mapping 中的信息替换 from_pretrained 中的列表选项
    from_pretrained = replace_list_option_in_docstrings(model_mapping._model_mapping)(from_pretrained)
    # 将 from_pretrained 方法设为类方法
    cls.from_pretrained = classmethod(from_pretrained)
    # 返回修改后的类对象
    return cls
# 定义函数 `get_values`，接收一个映射 `model_mapping`，返回所有值的列表
def get_values(model_mapping):
    # 初始化一个空列表 `result` 用于存放结果
    result = []
    # 遍历 `model_mapping` 中的所有值
    for model in model_mapping.values():
        # 如果值是列表或元组，则将其扁平化后加入 `result`
        if isinstance(model, (list, tuple)):
            result += list(model)
        else:
            # 否则直接将值添加到 `result` 中
            result.append(model)

    # 返回处理后的结果列表
    return result


# 定义函数 `getattribute_from_module`，根据模块和属性获取属性值
def getattribute_from_module(module, attr):
    # 如果属性为 None，则返回 None
    if attr is None:
        return None
    # 如果属性是元组，则递归获取每个元素的属性值并返回元组
    if isinstance(attr, tuple):
        return tuple(getattribute_from_module(module, a) for a in attr)
    # 如果模块具有指定属性，则返回该属性的值
    if hasattr(module, attr):
        return getattr(module, attr)
    
    # 如果以上条件都不满足，则尝试从 `transformers` 模块中导入相应的模块
    transformers_module = importlib.import_module("transformers")
    if module != transformers_module:
        try:
            # 尝试在 `transformers` 模块中查找属性的值
            return getattribute_from_module(transformers_module, attr)
        except ValueError:
            # 如果无法找到属性，则抛出 ValueError 异常
            raise ValueError(f"Could not find {attr} neither in {module} nor in {transformers_module}!")
    else:
        # 如果模块是 `transformers` 且仍然找不到属性，则抛出 ValueError 异常
        raise ValueError(f"Could not find {attr} in {transformers_module}!")


# 定义类 `_LazyAutoMapping`，继承自 `OrderedDict`
class _LazyAutoMapping(OrderedDict):
    """
    一个映射配置到对象（例如模型或分词器），在访问时加载键和值的类。

    Args:
        - config_mapping: 模型类型到配置类的映射
        - model_mapping: 模型类型到模型（或分词器）类的映射
    """

    # 初始化方法，接收配置映射和模型映射作为参数
    def __init__(self, config_mapping, model_mapping):
        # 初始化 `_config_mapping` 和 `_reverse_config_mapping` 属性
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        # 初始化 `_model_mapping` 属性，并将 `_model_mapping` 的 `_model_mapping` 属性设置为当前对象自身
        self._model_mapping = model_mapping
        self._model_mapping._model_mapping = self
        # 初始化 `_extra_content` 和 `_modules` 属性
        self._extra_content = {}
        self._modules = {}

    # 返回映射的长度
    def __len__(self):
        # 计算 `_config_mapping` 和 `_model_mapping` 公共键的数量，并加上 `_extra_content` 的长度
        common_keys = set(self._config_mapping.keys()).intersection(self._model_mapping.keys())
        return len(common_keys) + len(self._extra_content)

    # 根据键获取值的方法
    def __getitem__(self, key):
        # 如果键在 `_extra_content` 中，则返回其对应的值
        if key in self._extra_content:
            return self._extra_content[key]
        # 根据键获取模型类型
        model_type = self._reverse_config_mapping[key.__name__]
        # 如果模型类型在 `_model_mapping` 中，则获取相应模型的属性值
        if model_type in self._model_mapping:
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)

        # 如果一个配置关联了多个模型类型，则尝试获取每个模型类型对应的属性值
        model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        # 如果未找到匹配的键，则抛出 KeyError 异常
        raise KeyError(key)

    # 根据模型类型和属性名从模块中加载属性值的私有方法
    def _load_attr_from_module(self, model_type, attr):
        # 获取模型类型对应的模块名称
        module_name = model_type_to_module_name(model_type)
        # 如果模块名称不在 `_modules` 中，则导入该模块
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        # 调用 `getattribute_from_module` 函数获取模块中属性的值并返回
        return getattribute_from_module(self._modules[module_name], attr)
    def keys(self):
        # 从配置映射中加载属性，形成映射键列表
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping.keys()
        ]
        # 返回映射键列表加上额外内容的键列表
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key, default):
        try:
            # 调用 __getitem__ 方法获取键对应的值
            return self.__getitem__(key)
        except KeyError:
            # 如果键不存在，则返回默认值
            return default

    def __bool__(self):
        # 返回映射键的布尔值
        return bool(self.keys())

    def values(self):
        # 从模型映射中加载属性，形成映射值列表
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping.keys()
        ]
        # 返回映射值列表加上额外内容的值列表
        return mapping_values + list(self._extra_content.values())

    def items(self):
        # 从模型映射和配置映射中加载属性，形成映射项列表
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        # 返回映射项列表加上额外内容的项列表
        return mapping_items + list(self._extra_content.items())

    def __iter__(self):
        # 返回迭代器，迭代映射键
        return iter(self.keys())

    def __contains__(self, item):
        # 检查额外内容中是否包含指定项
        if item in self._extra_content:
            return True
        # 检查项是否具有 "__name__" 属性且其名称不在反向配置映射中
        if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping:
            return False
        # 获取项的模型类型
        model_type = self._reverse_config_mapping[item.__name__]
        # 检查模型类型是否在模型映射中
        return model_type in self._model_mapping

    def register(self, key, value, exist_ok=False):
        """
        Register a new model in this mapping.
        """
        # 如果键具有 "__name__" 属性且其名称在反向配置映射中
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            # 获取键对应的模型类型
            model_type = self._reverse_config_mapping[key.__name__]
            # 如果模型类型在模型映射中且不允许覆盖，则引发值错误异常
            if model_type in self._model_mapping.keys() and not exist_ok:
                raise ValueError(f"'{key}' is already used by a Transformers model.")
        # 向额外内容映射中注册新的键值对
        self._extra_content[key] = value
```