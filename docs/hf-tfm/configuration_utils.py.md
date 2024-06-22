# `.\transformers\configuration_utils.py`

```py
# 设置文件编码格式为 UTF-8
# 版权声明，版权归 Google AI 语言团队作者和 HuggingFace Inc. 团队所有
# 版权归 NVIDIA 公司所有
# 根据 Apache 许可证 2.0 版进行许可
# 除非符合许可证的要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，不附带任何形式的担保或条件
# 有关特定语言的权限，请参阅许可证
"""配置基类和工具。"""

# 导入必要的库
import copy  # 复制对象的标准库
import json  # JSON 序列化和反序列化的标准库
import os  # 提供了许多与操作系统交互的函数的标准库
import re  # 提供了正则表达式功能的标准库
import warnings  # 用于警告处理的标准库
from typing import Any, Dict, List, Optional, Tuple, Union  # 强类型提示的标准库

from packaging import version  # 版本处理的第三方库

# 导入本地模块
from . import __version__  # 导入当前包的版本信息
from .dynamic_module_utils import custom_object_save  # 导入自定义对象保存函数
from .utils import (  # 导入本地工具函数和类
    CONFIG_NAME,  # 配置文件名的常量
    PushToHubMixin,  # 推送到 Hub 的混合类
    add_model_info_to_auto_map,  # 将模型信息添加到自动映射中的函数
    cached_file,  # 缓存文件的函数
    copy_func,  # 复制函数的函数
    download_url,  # 下载 URL 的函数
    extract_commit_hash,  # 提取提交哈希的函数
    is_remote_url,  # 检查是否为远程 URL 的函数
    is_torch_available,  # 检查 Torch 是否可用的函数
    logging,  # 日志记录模块
)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义用于匹配配置文件名的正则表达式
_re_configuration_file = re.compile(r"config\.(.*)\.json")


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
    # 定义一个类，用于存储模型的相关属性和方法
    class PretrainedConfig:
        """
        - **num_attention_heads** (`int`) -- The number of attention heads used in the multi-head attention layers of the
          model.
        - **num_hidden_layers** (`int`) -- The number of blocks in the model.
        """

        # 模型类型
        model_type: str = ""
        # 是否为组合模型
        is_composition: bool = False
        # 属性映射字典
        attribute_map: Dict[str, str] = {}
        # 自动类
        _auto_class: Optional[str] = None

        # 设置属性值时的方法
        def __setattr__(self, key, value):
            if key in super().__getattribute__("attribute_map"):
                key = super().__getattribute__("attribute_map")[key]
            super().__setattr__(key, value)

        # 获取属性值时的方法
        def __getattribute__(self, key):
            if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
                key = super().__getattribute__("attribute_map")[key]
            return super().__getattribute__(key)

        # 获取属性值 name_or_path 的方法
        @property
        def name_or_path(self) -> str:
            return getattr(self, "_name_or_path", None)

        # 设置属性值 name_or_path 的方法
        @name_or_path.setter
        def name_or_path(self, value):
            self._name_or_path = str(value)  # 确保 name_or_path 是一个字符串（用于 JSON 编码）

        # 获取属性值 use_return_dict 的方法
        @property
        def use_return_dict(self) -> bool:
            """
            `bool`: Whether or not return [`~utils.ModelOutput`] instead of tuples.
            """
            # 如果设置了 torchscript，则强制 `return_dict=False` 以避免 jit 错误
            return self.return_dict and not self.torchscript

        # 获取属性值 num_labels 的方法
        @property
        def num_labels(self) -> int:
            """
            `int`: The number of labels for classification models.
            """
            return len(self.id2label)

        # 设置属性值 num_labels 的方法
        @num_labels.setter
        def num_labels(self, num_labels: int):
            if not hasattr(self, "id2label") or self.id2label is None or len(self.id2label) != num_labels:
                self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
                self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

        # 获取属性值 _attn_implementation 的方法
        @property
        def _attn_implementation(self):
            # 这个属性暂时设为私有（因为它不能被更改，需要实现一个 PreTrainedModel.use_attn_implementation 方法）
            if hasattr(self, "_attn_implementation_internal"):
                if self._attn_implementation_internal is None:
                    # `config.attn_implementation` 永远不应该为 None，为了向后兼容性。
                    return "eager"
                else:
                    return self._attn_implementation_internal
            else:
                return "eager"

        # 设置属性值 _attn_implementation 的方法
        @_attn_implementation.setter
        def _attn_implementation(self, value):
            self._attn_implementation_internal = value

        # 静态方法
        @staticmethod
    def _set_token_in_kwargs(kwargs, token=None):
        """临时方法用于处理 `token` 和 `use_auth_token`。

        这个方法是为了避免在所有覆盖 `from_pretrained` 的模型配置类中应用相同的更改。

        需要在后续 PR 中清理 `use_auth_token`。
        """
        # 一些模型配置类（如 CLIP）定义了自己的 `from_pretrained`，但尚未包含新参数 `token`。
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
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        从 `pretrained_model_name_or_path` 解析出一个参数字典，用于实例化一个 [`PretrainedConfig`] 使用 `from_dict`。

        参数:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                我们想要参数字典的预训练检查点的标识符。

        返回:
            `Tuple[Dict, Dict]`: 将用于实例化配置对象的字典。

        """
        cls._set_token_in_kwargs(kwargs)

        original_kwargs = copy.deepcopy(kwargs)
        # 获取与基本配置文件关联的配置字典
        config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "_commit_hash" in config_dict:
            original_kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # 那个配置文件可能会指向我们使用的另一个配置文件。
        if "configuration_files" in config_dict:
            configuration_file = get_configuration_file(config_dict["configuration_files"])
            config_dict, kwargs = cls._get_config_dict(
                pretrained_model_name_or_path, _configuration_file=configuration_file, **original_kwargs
            )

        return config_dict, kwargs

    @classmethod
    # 定义一个类方法，用于获取配置字典
    def _get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    @classmethod
    # 从字典参数实例化一个预训练配置对象
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
        # 从 kwargs 中弹出 "return_unused_kwargs" 参数，用于控制是否返回未使用的参数
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # 以下参数可能会被传递给内部遥测，将其从 kwargs 中移除，以免出现在 `return_unused_kwargs` 中
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # 如果 `config_dict` 和 kwargs 中都有 "_commit_hash" 参数，则保留 `config_dict` 中的值
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # 将 kwargs 中的 "attn_implementation" 参数移除，以免出现在 `return_unused_kwargs` 中
        config_dict["attn_implementation"] = kwargs.pop("attn_implementation", None)

        # 使用 config_dict 实例化一个配置对象
        config = cls(**config_dict)

        # 如果配置对象有 "pruned_heads" 属性，则将其转换为整数键的字典
        if hasattr(config, "pruned_heads"):
            config.pruned_heads = {int(key): value for key, value in config.pruned_heads.items()}

        # 如果 kwargs 中包含 "num_labels" 和 "id2label" 参数，则进行一些验证和处理
        if "num_labels" in kwargs and "id2label" in kwargs:
            num_labels = kwargs["num_labels"]
            id2label = kwargs["id2label"] if kwargs["id2label"] is not None else []
            if len(id2label) != num_labels:
                raise ValueError(
                    f"You passed along `num_labels={num_labels }` with an incompatible id to label map: "
                    f"{kwargs['id2label']}. Since those arguments are inconsistent with each other, you should remove "
                    "one of them."
                )
        
        # 处理 kwargs 中的参数，更新配置对象
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                current_attr = getattr(config, key)
                # 如果当前属性是预训练配置对象且值是字典，则实例化一个新的子配置对象
                if isinstance(current_attr, PretrainedConfig) and isinstance(value, dict):
                    value = current_attr.__class__(**value)
                setattr(config, key, value)
                if key != "torch_dtype":
                    to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        # 记录配置对象信息
        logger.info(f"Model config {config}")
        # 根据 return_unused_kwargs 决定返回结果
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    # 从 JSON 文件中实例化一个 PretrainedConfig 对象
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "PretrainedConfig":
        """
        Instantiates a [`PretrainedConfig`] from the path to a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from that JSON file.

        """
        # 从 JSON 文件中读取配置参数并转换为字典
        config_dict = cls._dict_from_json_file(json_file)
        # 使用参数字典实例化一个 PretrainedConfig 对象
        return cls(**config_dict)

    @classmethod
    # 从 JSON 文件中读取配置参数并返回字典
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            # 读取 JSON 文件内容
            text = reader.read()
        # 将 JSON 字符串转换为字典并返回
        return json.loads(text)

    # 判断两个 PretrainedConfig 对象是否相等
    def __eq__(self, other):
        return isinstance(other, PretrainedConfig) and (self.__dict__ == other.__dict__)

    # 返回 PretrainedConfig 对象的字符串表示形式
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"
    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        # 将配置中与默认配置属性对应的所有属性删除，以提高可读性，并序列化为 Python 字典

        config_dict = self.to_dict()

        # 获取默认配置字典
        default_config_dict = PretrainedConfig().to_dict()

        # 获取特定类的配置字典
        class_config_dict = self.__class__().to_dict() if not self.is_composition else {}

        serializable_config_dict = {}

        # 只序列化与默认配置不同的值
        for key, value in config_dict.items():
            if (
                isinstance(getattr(self, key, None), PretrainedConfig)
                and key in class_config_dict
                and isinstance(class_config_dict[key], dict)
            ):
                # 对于嵌套配置，需要递归清除差异
                diff = recursive_diff_dict(value, class_config_dict[key], config_obj=getattr(self, key, None))
                if "model_type" in value:
                    # 即使不在差异中，也需要设置
                    diff["model_type"] = value["model_type"]
                if len(diff) > 0:
                    serializable_config_dict[key] = diff
            elif (
                key not in default_config_dict
                or key == "transformers_version"
                or value != default_config_dict[key]
                or (key in class_config_dict and value != class_config_dict[key])
            ):
                serializable_config_dict[key] = value

        if hasattr(self, "quantization_config"):
            serializable_config_dict["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )

            # 移除 `_pre_quantization_dtype`，因为 torch.dtypes 不可序列化
            _ = serializable_config_dict.pop("_pre_quantization_dtype", None)

        self.dict_torch_dtype_to_str(serializable_config_dict)

        if "_attn_implementation_internal" in serializable_config_dict:
            del serializable_config_dict["_attn_implementation_internal"]

        return serializable_config_dict
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        # 深拷贝实例的属性字典
        output = copy.deepcopy(self.__dict__)
        # 如果类有 "model_type" 属性，则添加到输出字典中
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        # 删除特定的属性
        if "_auto_class" in output:
            del output["_auto_class"]
        if "_commit_hash" in output:
            del output["_commit_hash"]
        if "_attn_implementation_internal" in output:
            del output["_attn_implementation_internal"]

        # 将 Transformers 版本添加到输出字典中
        output["transformers_version"] = __version__

        for key, value in output.items():
            # 处理嵌套配置，如 CLIP
            if isinstance(value, PretrainedConfig):
                value = value.to_dict()
                del value["transformers_version"]

            output[key] = value

        if hasattr(self, "quantization_config"):
            # 将量化配置添加到输出字典中
            output["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )

            # 移除 "_pre_quantization_dtype"，因为 torch.dtypes 无法序列化
            _ = output.pop("_pre_quantization_dtype", None)

        # 转换 torch 数据类型为字符串
        self.dict_torch_dtype_to_str(output)

        return output

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            # 如果 use_diff 为 True，则返回差异字典
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        # 将配置字典转换为 JSON 字符串并返回
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON file.
        """
        # 将配置实例保存到 JSON 文件中
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))
    def update(self, config_dict: Dict[str, Any]):
        """
        Updates attributes of this class with attributes from `config_dict`.

        Args:
            config_dict (`Dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        # 遍历传入的字典，将每个键值对应用于当前类的属性
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
        # 将传入的字符串解析为字典，以 "=" 分割键值对，以 "," 分割不同的键值对
        d = dict(x.split("=") for x in update_str.split(","))
        # 遍历解析后的字典
        for k, v in d.items():
            # 检查当前类是否具有键 k 对应的属性
            if not hasattr(self, k):
                # 如果当前类没有该属性，则抛出 ValueError 异常
                raise ValueError(f"key {k} isn't in the original config dict")

            # 获取当前属性的旧值
            old_v = getattr(self, k)
            # 根据旧值的类型，将新值转换为相应类型
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

            # 使用新值更新当前类的属性
            setattr(self, k, v)

    def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        # 检查传入的字典及其嵌套字典是否具有名为 "torch_dtype" 的键，如果不是 None，则将 torch.dtype 转换为只包含类型的字符串
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        # 递归遍历字典的值，如果值是字典，则继续进行检查和转换
        for value in d.values():
            if isinstance(value, dict):
                self.dict_torch_dtype_to_str(value)

    @classmethod
    # 注册自定义配置类到指定的自动配置类中
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
        # 如果 auto_class 不是字符串类型，则获取其类名
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        # 导入自动配置模块
        import transformers.models.auto as auto_module

        # 检查是否存在指定的自动配置类
        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        # 将自动配置类名赋值给当前类的 _auto_class 属性
        cls._auto_class = auto_class

    # 获取生成默认参数的字典
    @staticmethod
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

    # 检查是否存在非默认生成参数
    def _has_non_default_generation_parameters(self) -> bool:
        """
        Whether or not this instance holds non-default generation parameters.
        """
        # 遍历默认生成参数字典，检查是否存在非默认值
        for parameter_name, default_value in self._get_generation_defaults().items():
            if hasattr(self, parameter_name) and getattr(self, parameter_name) != default_value:
                return True
        return False
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
    # 遍历提供的配置文件列表
    for file_name in configuration_files:
        # 使用正则表达式匹配版本号
        search = _re_configuration_file.search(file_name)
        # 如果匹配成功
        if search is not None:
            # 获取匹配到的版本号
            v = search.groups()[0]
            # 将版本号与配置文件名加入映射关系字典
            configuration_files_map[v] = file_name
    # 将版本号排序，确保较新的版本在前面
    available_versions = sorted(configuration_files_map.keys())

    # 默认使用 FULL_CONFIGURATION_FILE，然后尝试查看一些更新的版本
    configuration_file = CONFIG_NAME
    # 获取当前 transformers 版本号
    transformers_version = version.parse(__version__)
    # 遍历可用版本号
    for v in available_versions:
        # 如果当前版本号小于等于 transformers 版本号
        if version.parse(v) <= transformers_version:
            # 更新配置文件为该版本对应的配置文件
            configuration_file = configuration_files_map[v]
        else:
            # 由于版本号已排序，没有必要继续查找
            break

    # 返回选定的配置文件
    return configuration_file


def recursive_diff_dict(dict_a, dict_b, config_obj=None):
    """
    Helper function to recursively take the diff between two nested dictionaries. The resulting diff only contains the
    values from `dict_a` that are different from values in `dict_b`.
    """
    # 初始化差异字典
    diff = {}
    # 如果提供了配置对象，则获取其默认配置字典，否则初始化为空字典
    default = config_obj.__class__().to_dict() if config_obj is not None else {}
    # 遍历第一个字典的键值对
    for key, value in dict_a.items():
        # 获取配置对象中与当前键对应的属性值
        obj_value = getattr(config_obj, str(key), None)
        # 如果当前值为预训练配置对象并且在第二个字典中存在，并且第二个字典中对应值为字典
        if isinstance(obj_value, PretrainedConfig) and key in dict_b and isinstance(dict_b[key], dict):
            # 递归地计算当前键的差异
            diff_value = recursive_diff_dict(value, dict_b[key], config_obj=obj_value)
            # 如果差异字典不为空
            if len(diff_value) > 0:
                # 将当前键的差异加入差异字典
                diff[key] = diff_value
        # 如果当前键不在第二个字典中，或者当前值与第二个字典中对应值不同，或者当前键在默认配置中不存在，或者当前值与默认配置中对应值不同
        elif key not in dict_b or value != dict_b[key] or key not in default or value != default[key]:
            # 将当前键值对加入差异字典
            diff[key] = value
    # 返回差异字典
    return diff


# 使 PretrainedConfig 类的 push_to_hub 方法可以被复制
PretrainedConfig.push_to_hub = copy_func(PretrainedConfig.push_to_hub)
# 如果 push_to_hub 方法有文档字符串
if PretrainedConfig.push_to_hub.__doc__ is not None:
    # 格式化文档字符串，替换占位符
    PretrainedConfig.push_to_hub.__doc__ = PretrainedConfig.push_to_hub.__doc__.format(
        object="config", object_class="AutoConfig", object_files="configuration file"
    )
```