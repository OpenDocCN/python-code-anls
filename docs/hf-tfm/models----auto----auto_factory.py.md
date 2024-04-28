# `.\transformers\models\auto\auto_factory.py`

```
# 导入所需模块和库
import copy  # 导入copy模块，用于深拷贝对象
import importlib  # 导入importlib模块，用于动态导入模块
import json  # 导入json模块，用于处理JSON格式数据
import os  # 导入os模块，提供了与操作系统交互的功能
import warnings  # 导入warnings模块，用于处理警告信息
from collections import OrderedDict  # 从collections模块中导入OrderedDict类，用于创建有序字典

# 从相应模块中导入函数和类
from ...configuration_utils import PretrainedConfig  # 从configuration_utils模块中导入PretrainedConfig类
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code  # 从dynamic_module_utils模块中导入函数
from ...utils import (  # 从utils模块中导入多个函数和常量
    CONFIG_NAME,  # 导入CONFIG_NAME常量
    cached_file,  # 导入cached_file函数，用于缓存文件
    copy_func,  # 导入copy_func函数，用于复制函数对象
    extract_commit_hash,  # 导入extract_commit_hash函数，用于提取提交哈希值
    find_adapter_config_file,  # 导入find_adapter_config_file函数，用于查找适配器配置文件
    is_peft_available,  # 导入is_peft_available函数，用于检查是否可用PEFT
    logging,  # 导入logging模块，用于日志记录
    requires_backends,  # 导入requires_backends装饰器，用于指定后端依赖
)

# 从当前模块中导入模块
from .configuration_auto import AutoConfig, model_type_to_module_name, replace_list_option_in_docstrings  # 从configuration_auto模块中导入类和函数

# 获取logger对象
logger = logging.get_logger(__name__)


# 定义类文档字符串
CLASS_DOCSTRING = """
    This is a generic model class that will be instantiated as one of the model classes of the library when created
    with the [`~BaseAutoModelClass.from_pretrained`] class method or the [`~BaseAutoModelClass.from_config`] class
    method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
"""

# 定义从配置加载模型的文档字符串
FROM_CONFIG_DOCSTRING = """
        Instantiates one of the model classes of the library from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights. It only affects the
            model's configuration. Use [`~BaseAutoModelClass.from_pretrained`] to load the model weights.

        Args:
            config ([`PretrainedConfig`]):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples:

        ```python
        >>> from transformers import AutoConfig, BaseAutoModelClass

        >>> # Download configuration from huggingface.co and cache.
        >>> config = AutoConfig.from_pretrained("checkpoint_placeholder")
        >>> model = BaseAutoModelClass.from_config(config)
        ```
"""

# 定义_get_model_class函数
def _get_model_class(config, model_mapping):
    # 获取与配置相对应的支持模型列表
    supported_models = model_mapping[type(config)]
    # 如果支持的模型不是列表或元组，则直接返回支持的模型
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    # 构建模型名称到模型类的映射字典
    name_to_model = {model.__name__: model for model in supported_models}
    # 获取配置中的架构列表，如果不存在则为空列表
    architectures = getattr(config, "architectures", [])
    # 遍历给定的 architectures 列表
    for arch in architectures:
        # 如果当前架构在 name_to_model 字典中
        if arch in name_to_model:
            # 返回 name_to_model 字典中当前架构对应的模型
            return name_to_model[arch]
        # 如果以 "TF{arch}" 形式的架构在 name_to_model 字典中
        elif f"TF{arch}" in name_to_model:
            # 返回 name_to_model 字典中以 "TF{arch}" 形式命名的模型
            return name_to_model[f"TF{arch}"]
        # 如果以 "Flax{arch}" 形式的架构在 name_to_model 字典中
        elif f"Flax{arch}" in name_to_model:
            # 返回 name_to_model 字典中以 "Flax{arch}" 形式命名的模型
            return name_to_model[f"Flax{arch}"]

    # 如果配置中未设置架构或未匹配到支持的模型，则返回支持模型列表中的第一个元素作为默认值
    return supported_models[0]
class _BaseAutoModelClass:
    # Base class for auto models.
    _model_mapping = None

    def __init__(self, *args, **kwargs):
        # 抛出环境错误，提示使用特定方法实例化对象
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
            f"`{self.__class__.__name__}.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        # 从参数中获取 trust_remote_code
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        # 检查是否有远程代码，并且是否信任远程代码
        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        # 检查是否有本地代码
        has_local_code = type(config) in cls._model_mapping.keys()
        # 解析是否信任远程代码
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, config._name_or_path, has_local_code, has_remote_code
        )

        if has_remote_code and trust_remote_code:
            # 获取远程代码的类引用
            class_ref = config.auto_map[cls.__name__]
            if "--" in class_ref:
                repo_id, class_ref = class_ref.split("--")
            else:
                repo_id = config.name_or_path
            # 从动态模块中获取类
            model_class = get_class_from_dynamic_module(class_ref, repo_id, **kwargs)
            if os.path.isdir(config._name_or_path):
                model_class.register_for_auto_class(cls.__name__)
            else:
                cls.register(config.__class__, model_class, exist_ok=True)
            _ = kwargs.pop("code_revision", None)
            return model_class._from_config(config, **kwargs)
        elif type(config) in cls._model_mapping.keys():
            # 获取模型类
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class._from_config(config, **kwargs)

        # 抛出数值错误，提示配置类不被识别
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    @classmethod
    def register(cls, config_class, model_class, exist_ok=False):
        """
        Register a new model for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            model_class ([`PreTrainedModel`]):
                The model to register.
        """
        if hasattr(model_class, "config_class") and model_class.config_class != config_class:
            raise ValueError(
                "The model class you are passing has a `config_class` attribute that is not consistent with the "
                f"config class you passed (model has {model_class.config_class} and you passed {config_class}. Fix "
                "one of those so they match!"
            )
        cls._model_mapping.register(config_class, model_class, exist_ok=exist_ok)


class _BaseAutoBackboneClass(_BaseAutoModelClass):
    # Base class for auto backbone models.
    _model_mapping = None

    @classmethod
    # 从预训练模型名称或路径加载 Timm backbone
    def _load_timm_backbone_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 检查是否需要 vision 和 timm 后端
        requires_backends(cls, ["vision", "timm"])
        # 导入 TimmBackboneConfig 类
        from ...models.timm_backbone import TimmBackboneConfig

        # 获取配置，如果没有则使用默认配置
        config = kwargs.pop("config", TimmBackboneConfig())

        # 检查是否使用 Timm backbone，如果不是则抛出异常
        use_timm = kwargs.pop("use_timm_backbone", True)
        if not use_timm:
            raise ValueError("`use_timm_backbone` must be `True` for timm backbones")

        # 检查是否指定了 `out_features`，如果指定了则抛出异常
        if kwargs.get("out_features", None) is not None:
            raise ValueError("Cannot specify `out_features` for timm backbones")

        # 检查是否指定了 `output_loading_info`，如果指定了则抛出异常
        if kwargs.get("output_loading_info", False):
            raise ValueError("Cannot specify `output_loading_info=True` when loading from timm")

        # 获取或设置一些参数
        num_channels = kwargs.pop("num_channels", config.num_channels)
        features_only = kwargs.pop("features_only", config.features_only)
        use_pretrained_backbone = kwargs.pop("use_pretrained_backbone", config.use_pretrained_backbone)
        out_indices = kwargs.pop("out_indices", config.out_indices)
        
        # 创建 TimmBackboneConfig 对象
        config = TimmBackboneConfig(
            backbone=pretrained_model_name_or_path,
            num_channels=num_channels,
            features_only=features_only,
            use_pretrained_backbone=use_pretrained_backbone,
            out_indices=out_indices,
        )
        # 从配置中创建实例并返回
        return super().from_config(config, **kwargs)

    # 从预训练模型名称或路径加载模型
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 如果指定了使用 Timm backbone，则调用 _load_timm_backbone_from_pretrained 方法
        if kwargs.get("use_timm_backbone", False):
            return cls._load_timm_backbone_from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # 否则调用父类的 from_pretrained 方法
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
# 插入头部文档到给定文档字符串中
def insert_head_doc(docstring, head_doc=""):
    # 如果头部文档不为空，则在文档字符串中插入头部文档
    if len(head_doc) > 0:
        return docstring.replace(
            "one of the model classes of the library ",
            f"one of the model classes of the library (with a {head_doc} head) ",
        )
    # 否则在文档字符串中插入默认头部文档
    return docstring.replace(
        "one of the model classes of the library ", "one of the base model classes of the library "
    )

# 自动更新类
def auto_class_update(cls, checkpoint_for_example="bert-base-cased", head_doc=""):
    # 创建一个新的类，从基类中获取正确的名称
    model_mapping = cls._model_mapping
    name = cls.__name__
    class_docstring = insert_head_doc(CLASS_DOCSTRING, head_doc=head_doc)
    # 替换类文档字符串中的占位符为类名
    cls.__doc__ = class_docstring.replace("BaseAutoModelClass", name)

    # 复制并重新注册 `from_config` 和 `from_pretrained` 作为类方法，否则无法为它们设置特定的文档字符串
    from_config = copy_func(_BaseAutoModelClass.from_config)
    from_config_docstring = insert_head_doc(FROM_CONFIG_DOCSTRING, head_doc=head_doc)
    from_config_docstring = from_config_docstring.replace("BaseAutoModelClass", name)
    from_config_docstring = from_config_docstring.replace("checkpoint_placeholder", checkpoint_for_example)
    from_config.__doc__ = from_config_docstring
    from_config = replace_list_option_in_docstrings(model_mapping._model_mapping, use_model_types=False)(from_config)
    cls.from_config = classmethod(from_config)

    # 根据类名前缀选择不同的 `from_pretrained` 文档字符串
    if name.startswith("TF"):
        from_pretrained_docstring = FROM_PRETRAINED_TF_DOCSTRING
    elif name.startswith("Flax"):
        from_pretrained_docstring = FROM_PRETRAINED_FLAX_DOCSTRING
    else:
        from_pretrained_docstring = FROM_PRETRAINED_TORCH_DOCSTRING
    from_pretrained = copy_func(_BaseAutoModelClass.from_pretrained)
    from_pretrained_docstring = insert_head_doc(from_pretrained_docstring, head_doc=head_doc)
    from_pretrained_docstring = from_pretrained_docstring.replace("BaseAutoModelClass", name)
    from_pretrained_docstring = from_pretrained_docstring.replace("checkpoint_placeholder", checkpoint_for_example)
    shortcut = checkpoint_for_example.split("/")[-1].split("-")[0]
    from_pretrained_docstring = from_pretrained_docstring.replace("shortcut_placeholder", shortcut)
    from_pretrained.__doc__ = from_pretrained_docstring
    from_pretrained = replace_list_option_in_docstrings(model_mapping._model_mapping)(from_pretrained)
    cls.from_pretrained = classmethod(from_pretrained)
    return cls

# 获取模型映射中的值
def get_values(model_mapping):
    result = []
    for model in model_mapping.values():
        # 如果值是列表或元组，则将其展开后添加到结果中
        if isinstance(model, (list, tuple)):
            result += list(model)
        else:
            result.append(model)
    return result

# 从模块中获取属性
def getattribute_from_module(module, attr):
    # 如果属性为空，则返回 None
    if attr is None:
        return None
    # 如果属性是元组，则递归获取每个属性
    if isinstance(attr, tuple):
        return tuple(getattribute_from_module(module, a) for a in attr)
    # 如果模块中存在该属性，则返回属性值
    if hasattr(module, attr):
        return getattr(module, attr)
    # 导入模块 importlib 中的 import_module 函数，用于动态导入模块
    transformers_module = importlib.import_module("transformers")

    # 如果当前模块不是 transformers 模块，则执行以下操作
    if module != transformers_module:
        # 尝试从 transformers 模块中获取属性 attr
        try:
            # 调用自定义函数 getattribute_from_module 从 transformers 模块获取属性 attr
            return getattribute_from_module(transformers_module, attr)
        # 如果在 transformers 模块中找不到指定的属性，则抛出 ValueError 异常
        except ValueError:
            raise ValueError(f"Could not find {attr} neither in {module} nor in {transformers_module}!")
    # 如果当前模块是 transformers 模块，则执行以下操作
    else:
        # 抛出 ValueError 异常，指示在 transformers 模块中找不到指定的属性
        raise ValueError(f"Could not find {attr} in {transformers_module}!")
class _LazyAutoMapping(OrderedDict):
    """
    " A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    # 初始化 LazyAutoMapping 类
    def __init__(self, config_mapping, model_mapping):
        # 存储 config_mapping
        self._config_mapping = config_mapping
        # 创建反向映射，将 config_mapping 的值作为键，键作为值
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        # 存储 model_mapping
        self._model_mapping = model_mapping
        # 将 LazyAutoMapping 对象作为 model_mapping 的属性
        self._model_mapping._model_mapping = self
        # 存储额外内容
        self._extra_content = {}
        # 存储模块
        self._modules = {}

    # 返回 LazyAutoMapping 的长度
    def __len__(self):
        # 获取 config_mapping 和 model_mapping 的交集，并返回长度
        common_keys = set(self._config_mapping.keys()).intersection(self._model_mapping.keys())
        return len(common_keys) + len(self._extra_content)

    # 获取 LazyAutoMapping 中指定键的值
    def __getitem__(self, key):
        # 如果键在额外内容中，则返回额外内容中的值
        if key in self._extra_content:
            return self._extra_content[key]
        # 获取模型类型
        model_type = self._reverse_config_mapping[key.__name__]
        # 如果模型类型在 model_mapping 中，则加载模块属性
        if model_type in self._model_mapping:
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)

        # 如果一个配置关联了多个模型类型
        model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        raise KeyError(key)

    # 从模块加载属性
    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        return getattribute_from_module(self._modules[module_name], attr)

    # 返回 LazyAutoMapping 的键
    def keys(self):
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping.keys()
        ]
        return mapping_keys + list(self._extra_content.keys())

    # 获取指定键的值，如果不存在则返回默认值
    def get(self, key, default):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    # 返回 LazyAutoMapping 是否为真
    def __bool__(self):
        return bool(self.keys())

    # 返回 LazyAutoMapping 的值
    def values(self):
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping.keys()
        ]
        return mapping_values + list(self._extra_content.values())
    # 返回映射项的迭代器
    def items(self):
        # 生成一个包含映射项的列表，每个映射项由配置映射和模型映射中对应键的属性组成
        mapping_items = [
            (
                # 从模块中加载指定键对应的属性值，并加入配置映射中
                self._load_attr_from_module(key, self._config_mapping[key]),
                # 从模块中加载指定键对应的属性值，并加入模型映射中
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            # 对模型映射中的每个键进行遍历
            for key in self._model_mapping.keys()
            # 如果该键也在配置映射中
            if key in self._config_mapping.keys()
        ]
        # 返回映射项列表加上额外内容字典中的所有项
        return mapping_items + list(self._extra_content.items())

    # 返回键的迭代器
    def __iter__(self):
        # 返回键的迭代器
        return iter(self.keys())

    # 检查映射中是否包含指定项
    def __contains__(self, item):
        # 如果项在额外内容字典中，则返回 True
        if item in self._extra_content:
            return True
        # 如果项没有 '__name__' 属性，或者该属性的值不在反向配置映射中，则返回 False
        if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping:
            return False
        # 获取项的名称，查找其是否在反向配置映射中，如果在，则获取对应的模型类型
        model_type = self._reverse_config_mapping[item.__name__]
        # 返回模型类型是否在模型映射中
        return model_type in self._model_mapping

    # 在映射中注册新的模型
    def register(self, key, value, exist_ok=False):
        """
        Register a new model in this mapping.
        """
        # 如果键具有 '__name__' 属性，并且该属性的值在反向配置映射中
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            # 获取键对应的模型类型
            model_type = self._reverse_config_mapping[key.__name__]
            # 如果模型类型在模型映射中，并且不允许覆盖，则引发 ValueError
            if model_type in self._model_mapping.keys() and not exist_ok:
                raise ValueError(f"'{key}' is already used by a Transformers model.")

        # 在额外内容字典中注册键值对
        self._extra_content[key] = value
```