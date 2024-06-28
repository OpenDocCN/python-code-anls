# `.\configuration_utils.py`

```
# coding=utf-8
# 版权声明及许可证信息

""" Configuration base class and utilities."""
# 导入必要的库和模块
import copy  # 用于对象的深拷贝操作
import json  # 用于 JSON 数据的处理
import os  # 提供与操作系统相关的功能
import re  # 提供正则表达式的支持
import warnings  # 用于发出警告信息
from typing import Any, Dict, List, Optional, Tuple, Union  # 引入类型提示功能

from packaging import version  # 用于版本号处理

from . import __version__  # 导入当前模块的版本信息
from .dynamic_module_utils import custom_object_save  # 导入自定义对象保存函数
from .utils import (  # 导入一些工具函数和常量
    CONFIG_NAME,  # 配置文件名常量
    PushToHubMixin,  # 提供向 Hub 推送功能的混合类
    add_model_info_to_auto_map,  # 将模型信息添加到自动映射的函数
    cached_file,  # 缓存文件的函数
    copy_func,  # 函数复制的工具函数
    download_url,  # 下载 URL 资源的函数
    extract_commit_hash,  # 提取提交哈希的函数
    is_remote_url,  # 判断是否是远程 URL 的函数
    is_torch_available,  # 判断是否可用 PyTorch 的函数
    logging,  # 日志记录模块
)

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_re_configuration_file = re.compile(r"config\.(.*)\.json")  # 编译用于匹配配置文件名的正则表达式


class PretrainedConfig(PushToHubMixin):
    # no-format
    r"""
    Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
    methods for loading/downloading/saving configurations.

    <Tip>

    A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
    initialize a model does **not** load the model weights. It only affects the model's configuration.

    </Tip>

    Class attributes (overridden by derived classes):

    - **model_type** (`str`) -- An identifier for the model type, serialized into the JSON file, and used to recreate
      the correct object in [`~transformers.AutoConfig`].
    - **is_composition** (`bool`) -- Whether the config class is composed of multiple sub-configs. In this case the
      config has to be initialized from two or more configs of type [`~transformers.PretrainedConfig`] like:
      [`~transformers.EncoderDecoderConfig`] or [`~RagConfig`].
    - **keys_to_ignore_at_inference** (`List[str]`) -- A list of keys to ignore by default when looking at dictionary
      outputs of the model during inference.
    - **attribute_map** (`Dict[str, str]`) -- A dict that maps model specific attribute names to the standardized
      naming of attributes.

    Common attributes (present in all subclasses):

    - **vocab_size** (`int`) -- The number of tokens in the vocabulary, which is also the first dimension of the
      embeddings matrix (this attribute may be missing for models that don't have a text modality like ViT).
    - **hidden_size** (`int`) -- The hidden size of the model.

    """
    # 定义了一个预训练配置类 PretrainedConfig，是所有配置类的基类，包含了通用的模型配置参数和加载/保存配置的方法
    # 没有额外的代码需要注释
    # `model_type` 是模型的类型描述字符串
    model_type: str = ""
    # `is_composition` 表示模型是否是一个组合模型，默认为 False
    is_composition: bool = False
    # `attribute_map` 是一个映射，用于属性重命名
    attribute_map: Dict[str, str] = {}
    # `_auto_class` 是一个私有属性，用于存储自动类的名称，可选为 None
    _auto_class: Optional[str] = None

    def __setattr__(self, key, value):
        # 自定义的属性设置方法，用于根据 `attribute_map` 重命名属性
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        # 自定义的属性获取方法，用于根据 `attribute_map` 重命名属性
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)

    @property
    def name_or_path(self) -> str:
        # 返回模型的名称或路径，作为 `_name_or_path` 的值
        return getattr(self, "_name_or_path", None)

    @name_or_path.setter
    def name_or_path(self, value):
        # 设置模型的名称或路径，确保为字符串类型（用于 JSON 编码）
        self._name_or_path = str(value)

    @property
    def use_return_dict(self) -> bool:
        """
        `bool`: 是否返回 [`~utils.ModelOutput`] 而不是元组。
        """
        # 如果设置了 torchscript，强制 `return_dict=False` 以避免 JIT 错误
        return self.return_dict and not self.torchscript

    @property
    def num_labels(self) -> int:
        """
        `int`: 分类模型的标签数量。
        """
        # 返回模型的标签数量，基于 `id2label` 的长度
        return len(self.id2label)

    @num_labels.setter
    def num_labels(self, num_labels: int):
        # 设置模型的标签数量，如果 `id2label` 不存在或长度不符合，则重新生成标签映射
        if not hasattr(self, "id2label") or self.id2label is None or len(self.id2label) != num_labels:
            self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
            self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

    @property
    def _attn_implementation(self):
        """
        `str`: 注意力机制的实现方式。
        """
        # 私有属性，返回注意力机制的实现方式，默认为 "eager"
        if hasattr(self, "_attn_implementation_internal"):
            if self._attn_implementation_internal is None:
                return "eager"
            else:
                return self._attn_implementation_internal
        else:
            return "eager"

    @_attn_implementation.setter
    def _attn_implementation(self, value):
        # 设置注意力机制的实现方式
        self._attn_implementation_internal = value
    @classmethod
    def _set_token_in_kwargs(kwargs, token=None):
        """在 kwargs 中设置 `token` 参数。

        这个方法是为了避免在所有模型配置类中重复应用相同的更改，这些类重写了 `from_pretrained` 方法。

        需要在随后的 PR 中清理 `use_auth_token`。
        """
        # 一些模型配置类（如 CLIP）定义了自己的 `from_pretrained` 方法，但还没有新参数 `token`。
        if token is None:
            token = kwargs.pop("token", None)
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        # 如果存在 token，则将其添加到 kwargs 中
        if token is not None:
            kwargs["token"] = token

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
        """
        从 `pretrained_model_name_or_path` 解析出参数字典，用于通过 `from_dict` 实例化 `PretrainedConfig`。

        参数：
            pretrained_model_name_or_path (`str` 或 `os.PathLike`):
                想要获取参数字典的预训练检查点的标识符。

        返回：
            `Tuple[Dict, Dict]`: 将用于实例化配置对象的字典。

        """
        # 调用 `_set_token_in_kwargs` 方法，设置 `token` 参数
        cls._set_token_in_kwargs(kwargs)

        original_kwargs = copy.deepcopy(kwargs)
        # 获取与基本配置文件关联的配置字典
        config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "_commit_hash" in config_dict:
            original_kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # 可能会指向另一个要使用的配置文件。
        if "configuration_files" in config_dict:
            configuration_file = get_configuration_file(config_dict["configuration_files"])
            config_dict, kwargs = cls._get_config_dict(
                pretrained_model_name_or_path, _configuration_file=configuration_file, **original_kwargs
            )

        return config_dict, kwargs

    @classmethod
    # 定义类方法 `_get_config_dict`，用于获取配置信息的字典
    def _get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ):
    # 类方法：从给定的配置字典中实例化一个预训练配置对象 [`PretrainedConfig`]。

    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        """
        Instantiates a [`PretrainedConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the [`~PretrainedConfig.get_config_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from those parameters.
        """
        # 从 kwargs 中弹出 "return_unused_kwargs" 参数，如果没有则默认为 False
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # 从 kwargs 中移除 "_from_auto" 和 "_from_pipeline" 参数，避免它们出现在 `return_unused_kwargs` 中
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        
        # 如果配置字典中包含 "_commit_hash"，则更新 kwargs 中的 "_commit_hash"，以防 kwargs 覆盖这个更新
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # 将 kwargs 中的 "attn_implementation" 参数移除，并将其设置为 config_dict 中的值
        config_dict["attn_implementation"] = kwargs.pop("attn_implementation", None)

        # 使用 config_dict 实例化一个 cls 类型的配置对象 config
        config = cls(**config_dict)

        # 如果配置对象 config 有 "pruned_heads" 属性，则将其键转换为整数
        if hasattr(config, "pruned_heads"):
            config.pruned_heads = {int(key): value for key, value in config.pruned_heads.items()}

        # 如果 kwargs 中包含 "num_labels" 和 "id2label"，则验证它们是否兼容
        if "num_labels" in kwargs and "id2label" in kwargs:
            num_labels = kwargs["num_labels"]
            id2label = kwargs["id2label"] if kwargs["id2label"] is not None else []
            if len(id2label) != num_labels:
                raise ValueError(
                    f"You passed along `num_labels={num_labels}` with an incompatible id to label map: "
                    f"{kwargs['id2label']}. Since those arguments are inconsistent with each other, you should remove "
                    "one of them."
                )

        # 准备从配置对象 config 中移除的参数列表
        to_remove = []
        # 遍历 kwargs 中的键值对
        for key, value in kwargs.items():
            # 如果 config 中有对应的属性 key，则将其设置为 value
            if hasattr(config, key):
                current_attr = getattr(config, key)
                # 如果当前属性是 PretrainedConfig 类型且 value 是字典，则将其转换为相应的子配置
                if isinstance(current_attr, PretrainedConfig) and isinstance(value, dict):
                    value = current_attr.__class__(**value)
                setattr(config, key, value)
                # 将 key 添加到待移除列表中（除了 "torch_dtype"）
                if key != "torch_dtype":
                    to_remove.append(key)
        # 从 kwargs 中移除已处理的键值对
        for key in to_remove:
            kwargs.pop(key, None)

        # 记录配置对象 config 的信息
        logger.info(f"Model config {config}")
        # 如果需要返回未使用的 kwargs，则返回配置对象和剩余的 kwargs
        if return_unused_kwargs:
            return config, kwargs
        else:
            # 否则只返回配置对象
            return config

    @classmethod
    # 从 JSON 文件中读取配置并实例化一个 PretrainedConfig 对象
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "PretrainedConfig":
        """
        Instantiates a [`PretrainedConfig`] from the path to a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from that JSON file.

        """
        # 从 JSON 文件中读取配置信息并转换成字典形式
        config_dict = cls._dict_from_json_file(json_file)
        # 使用字典中的配置参数实例化一个 PretrainedConfig 对象
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        """
        Reads and parses a JSON file into a dictionary.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file.

        Returns:
            dict: Dictionary containing the parsed JSON content.

        """
        # 打开 JSON 文件，读取其中的文本内容
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        # 将读取的 JSON 文本解析为字典对象
        return json.loads(text)

    # 定义相等性比较方法，用于比较两个 PretrainedConfig 对象是否相等
    def __eq__(self, other):
        return isinstance(other, PretrainedConfig) and (self.__dict__ == other.__dict__)

    # 定义对象的字符串表示方法，返回包含 JSON 字符串的对象表示形式
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"
    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        # Serialize current configuration instance to a dictionary
        config_dict = self.to_dict()

        # Get default configuration dictionary
        default_config_dict = PretrainedConfig().to_dict()

        # Get class-specific configuration dictionary
        class_config_dict = self.__class__().to_dict() if not self.is_composition else {}

        serializable_config_dict = {}

        # Iterate over each key-value pair in the current configuration dictionary
        for key, value in config_dict.items():
            # Check if the attribute is a PretrainedConfig instance and differs from class-specific config
            if (
                isinstance(getattr(self, key, None), PretrainedConfig)
                and key in class_config_dict
                and isinstance(class_config_dict[key], dict)
            ):
                # Recursive diff for nested configurations
                diff = recursive_diff_dict(value, class_config_dict[key], config_obj=getattr(self, key, None))
                # Ensure model_type is set even if not in the diff
                if "model_type" in value:
                    diff["model_type"] = value["model_type"]
                # Include in serializable dictionary if there are differences
                if len(diff) > 0:
                    serializable_config_dict[key] = diff
            # Include if key not in default config, or values differ from default or class-specific configs
            elif (
                key not in default_config_dict
                or key == "transformers_version"
                or value != default_config_dict[key]
                or (key in class_config_dict and value != class_config_dict[key])
            ):
                serializable_config_dict[key] = value

        # Handle special case for quantization_config
        if hasattr(self, "quantization_config"):
            if isinstance(self.quantization_config, dict):
                serializable_config_dict["quantization_config"] = self.quantization_config
            else:
                serializable_config_dict["quantization_config"] = self.quantization_config.to_dict()

            # Remove _pre_quantization_dtype as it's not serializable
            _ = serializable_config_dict.pop("_pre_quantization_dtype", None)

        # Convert torch dtypes to strings in the dictionary
        self.dict_torch_dtype_to_str(serializable_config_dict)

        # Remove internal implementation detail if present
        if "_attn_implementation_internal" in serializable_config_dict:
            del serializable_config_dict["_attn_implementation_internal"]

        return serializable_config_dict
    def to_dict(self) -> Dict[str, Any]:
        """
        将当前实例序列化为一个 Python 字典。

        Returns:
            `Dict[str, Any]`: 包含构成该配置实例的所有属性的字典。
        """
        # 深拷贝实例的所有属性到输出字典
        output = copy.deepcopy(self.__dict__)
        # 如果类定义了 model_type 属性，则将其加入输出字典
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        # 删除输出字典中的特定内部属性
        if "_auto_class" in output:
            del output["_auto_class"]
        if "_commit_hash" in output:
            del output["_commit_hash"]
        if "_attn_implementation_internal" in output:
            del output["_attn_implementation_internal"]

        # 添加 Transformers 的版本信息到输出字典
        output["transformers_version"] = __version__

        # 处理嵌套的配置（例如 CLIP），将其转换为字典形式
        for key, value in output.items():
            if isinstance(value, PretrainedConfig):
                value = value.to_dict()
                # 移除嵌套配置中的 Transformers 版本信息
                del value["transformers_version"]
            output[key] = value

        # 如果实例有 quantization_config 属性，将其转换为字典形式并加入输出字典
        if hasattr(self, "quantization_config"):
            output["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )

            # 移除输出字典中的 _pre_quantization_dtype 属性，因为 torch.dtypes 不可序列化
            _ = output.pop("_pre_quantization_dtype", None)

        # 对输出字典中的 torch 数据类型进行转换处理
        self.dict_torch_dtype_to_str(output)

        # 返回最终的输出字典
        return output

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        将当前实例序列化为 JSON 字符串。

        Args:
            use_diff (`bool`, *optional*, 默认为 `True`):
                如果设置为 `True`，则只序列化配置实例与默认 `PretrainedConfig()` 之间的差异。

        Returns:
            `str`: 包含构成该配置实例的所有属性的 JSON 格式字符串。
        """
        # 根据 use_diff 参数决定是否只序列化差异部分
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        # 将字典转换为 JSON 字符串，缩进为 2，按键排序
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        将当前实例保存为 JSON 文件。

        Args:
            json_file_path (`str` 或 `os.PathLike`):
                保存配置实例参数的 JSON 文件路径。
            use_diff (`bool`, *optional*, 默认为 `True`):
                如果设置为 `True`，则只序列化配置实例与默认 `PretrainedConfig()` 之间的差异。
        """
        # 打开指定路径的 JSON 文件，将实例转换为 JSON 字符串并写入文件
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))
    def update(self, config_dict: Dict[str, Any]):
        """
        Updates attributes of this class with attributes from `config_dict`.

        Args:
            config_dict (`Dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        # 遍历传入的字典，将每个键值对应用到当前类的属性上
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update_from_string(self, update_str: str):
        """
        Updates attributes of this class with attributes from `update_str`.

        The expected format is ints, floats and strings as is, and for booleans use `true` or `false`. For example:
        "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"

        The keys to change have to already exist in the config object.

        Args:
            update_str (`str`): String with attributes that should be updated for this class.

        """
        # 将传入的字符串按逗号分割成键值对，构建字典
        d = dict(x.split("=") for x in update_str.split(","))
        # 遍历字典中的每个键值对
        for k, v in d.items():
            # 检查当前类是否存在名为 k 的属性
            if not hasattr(self, k):
                raise ValueError(f"key {k} isn't in the original config dict")

            # 获取当前属性的旧值
            old_v = getattr(self, k)
            # 根据旧值的类型转换新值 v 的类型，并设置为当前类的属性
            if isinstance(old_v, bool):
                if v.lower() in ["true", "1", "y", "yes"]:
                    v = True
                elif v.lower() in ["false", "0", "n", "no"]:
                    v = False
                else:
                    raise ValueError(f"can't derive true or false from {v} (key {k})")
            elif isinstance(old_v, int):
                v = int(v)
            elif isinstance(old_v, float):
                v = float(v)
            elif not isinstance(old_v, str):
                raise ValueError(
                    f"You can only update int, float, bool or string values in the config, got {v} for key {k}"
                )

            setattr(self, k, v)

    def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        # 检查传入的字典是否包含名为 torch_dtype 的键，并且其值不为 None
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            # 将 torch.dtype 转换为只包含类型的字符串，例如将 torch.float32 转换为 "float32"
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        # 递归处理字典中的每个值，如果值是字典，则继续调用该方法
        for value in d.values():
            if isinstance(value, dict):
                self.dict_torch_dtype_to_str(value)

    @classmethod
    def`
# 注册自动配置类方法，用于将当前类注册到指定的自动配置类中
def register_for_auto_class(cls, auto_class="AutoConfig"):
    """
    Register this class with a given auto class. This should only be used for custom configurations as the ones in
    the library are already mapped with `AutoConfig`.

    <Tip warning={true}>
    This API is experimental and may have some slight breaking changes in the next releases.
    </Tip>

    Args:
        auto_class (`str` or `type`, *optional*, defaults to `"AutoConfig"`):
            The auto class to register this new configuration with.
    """
    # 如果 auto_class 不是字符串，将其转换为类名字符串
    if not isinstance(auto_class, str):
        auto_class = auto_class.__name__

    # 导入 transformers.models.auto 模块
    import transformers.models.auto as auto_module

    # 如果 auto_class 在 auto_module 中不存在，抛出 ValueError
    if not hasattr(auto_module, auto_class):
        raise ValueError(f"{auto_class} is not a valid auto class.")

    # 将 auto_class 赋值给当前类的 _auto_class 属性
    cls._auto_class = auto_class

@staticmethod
# 返回默认的生成参数字典
def _get_generation_defaults() -> Dict[str, Any]:
    return {
        "max_length": 20,
        "min_length": 0,
        "do_sample": False,
        "early_stopping": False,
        "num_beams": 1,
        "num_beam_groups": 1,
        "diversity_penalty": 0.0,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "typical_p": 1.0,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "bad_words_ids": None,
        "num_return_sequences": 1,
        "output_scores": False,
        "return_dict_in_generate": False,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "remove_invalid_values": False,
        "exponential_decay_length_penalty": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
    }

# 判断当前实例是否具有非默认生成参数
def _has_non_default_generation_parameters(self) -> bool:
    """
    Whether or not this instance holds non-default generation parameters.
    """
    # 获取默认的生成参数字典
    defaults = self._get_generation_defaults()

    # 遍历生成参数字典，检查当前实例是否有非默认值的生成参数
    for parameter_name, default_value in defaults.items():
        if hasattr(self, parameter_name) and getattr(self, parameter_name) != default_value:
            return True
    return False
# 获取用于此版本 transformers 的配置文件。
def get_configuration_file(configuration_files: List[str]) -> str:
    """
    Get the configuration file to use for this version of transformers.

    Args:
        configuration_files (`List[str]`): The list of available configuration files.

    Returns:
        `str`: The configuration file to use.
    """
    # 初始化一个空字典，用于存储版本号与配置文件名的映射关系
    configuration_files_map = {}
    # 遍历每个配置文件名
    for file_name in configuration_files:
        # 使用正则表达式搜索文件名中的版本号信息
        search = _re_configuration_file.search(file_name)
        # 如果找到匹配项
        if search is not None:
            # 提取版本号信息并存储到字典中
            v = search.groups()[0]
            configuration_files_map[v] = file_name

    # 对版本号进行排序
    available_versions = sorted(configuration_files_map.keys())

    # 默认使用 FULL_CONFIGURATION_FILE，然后尝试使用一些更新的版本
    configuration_file = CONFIG_NAME
    transformers_version = version.parse(__version__)
    # 遍历所有可用版本
    for v in available_versions:
        # 如果当前版本小于等于 transformers 的版本
        if version.parse(v) <= transformers_version:
            # 更新配置文件为对应版本的配置文件
            configuration_file = configuration_files_map[v]
        else:
            # 因为版本已排序，所以不再继续查找
            break

    # 返回选择的配置文件名
    return configuration_file


# 递归比较两个嵌套字典的差异，返回仅包含 dict_a 中不同于 dict_b 的值的字典
def recursive_diff_dict(dict_a, dict_b, config_obj=None):
    """
    Helper function to recursively take the diff between two nested dictionaries. The resulting diff only contains the
    values from `dict_a` that are different from values in `dict_b`.
    """
    # 初始化一个空字典，用于存储差异
    diff = {}
    # 如果传入了 config_obj 参数，则获取其默认配置的字典表示
    default = config_obj.__class__().to_dict() if config_obj is not None else {}
    # 遍历 dict_a 的每一个键值对
    for key, value in dict_a.items():
        # 尝试从 config_obj 中获取与当前键对应的值
        obj_value = getattr(config_obj, str(key), None)
        # 如果 obj_value 是 PretrainedConfig 类型，并且 dict_b 中存在当前键，并且 dict_b 中的值也是字典
        if isinstance(obj_value, PretrainedConfig) and key in dict_b and isinstance(dict_b[key], dict):
            # 递归调用自身，比较当前值与 dict_b[key] 的差异
            diff_value = recursive_diff_dict(value, dict_b[key], config_obj=obj_value)
            # 如果有差异，则将其存储到 diff 字典中
            if len(diff_value) > 0:
                diff[key] = diff_value
        # 如果当前键不在 dict_b 中，或者当前值与 dict_b[key] 的值不同，或者当前键在 default 中但值不同于 default 中的值
        elif key not in dict_b or value != dict_b[key] or key not in default or value != default[key]:
            # 将当前键值对存储到 diff 字典中
            diff[key] = value
    # 返回差异字典
    return diff


# 将 PretrainedConfig 类的 push_to_hub 方法复制给 PretrainedConfig.push_to_hub
PretrainedConfig.push_to_hub = copy_func(PretrainedConfig.push_to_hub)
# 如果 PretrainedConfig.push_to_hub 方法有文档字符串
if PretrainedConfig.push_to_hub.__doc__ is not None:
    # 使用格式化字符串，将文档字符串中的占位符替换为实际值
    PretrainedConfig.push_to_hub.__doc__ = PretrainedConfig.push_to_hub.__doc__.format(
        object="config", object_class="AutoConfig", object_files="configuration file"
    )
```