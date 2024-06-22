# `.\transformers\generation\configuration_utils.py`

```py
# 引入所需的模块和库
import copy  # 引入 copy 模块，用于复制对象
import json  # 引入 json 模块，用于处理 JSON 格式数据
import os  # 引入 os 模块，用于操作系统相关功能
import warnings  # 引入 warnings 模块，用于警告处理
from typing import Any, Dict, Optional, Union  # 从 typing 模块引入特定类型

# 从当前包中引入的模块
from .. import __version__  # 引入当前包的 __version__ 变量
from ..configuration_utils import PretrainedConfig  # 从当前包的 configuration_utils 模块引入 PretrainedConfig 类
from ..utils import (  # 从当前包的 utils 模块引入以下函数和变量
    GENERATION_CONFIG_NAME,  # 引入 GENERATION_CONFIG_NAME 变量
    PushToHubMixin,  # 引入 PushToHubMixin 类
    cached_file,  # 引入 cached_file 函数
    download_url,  # 引入 download_url 函数
    extract_commit_hash,  # 引入 extract_commit_hash 函数
    is_remote_url,  # 引入 is_remote_url 函数
    logging,  # 引入 logging 模块
)

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# 元数据字段，用于记录配置信息的元数据
METADATA_FIELDS = ("_from_model_config", "_commit_hash", "_original_object_hash", "transformers_version")


class GenerationConfig(PushToHubMixin):
    # no-format
    r"""
    Class that holds a configuration for a generation task. A `generate` call supports the following generation methods
    for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

        - *greedy decoding* by calling [`~generation.GenerationMixin.greedy_search`] if `num_beams=1` and
            `do_sample=False`
        - *contrastive search* by calling [`~generation.GenerationMixin.contrastive_search`] if `penalty_alpha>0.`
            and `top_k>1`
        - *multinomial sampling* by calling [`~generation.GenerationMixin.sample`] if `num_beams=1` and
            `do_sample=True`
        - *beam-search decoding* by calling [`~generation.GenerationMixin.beam_search`] if `num_beams>1` and
            `do_sample=False`
        - *beam-search multinomial sampling* by calling [`~generation.GenerationMixin.beam_sample`] if
            `num_beams>1` and `do_sample=True`
        - *diverse beam-search decoding* by calling [`~generation.GenerationMixin.group_beam_search`], if
            `num_beams>1` and `num_beam_groups>1`
        - *constrained beam-search decoding* by calling [`~generation.GenerationMixin.constrained_beam_search`], if
            `constraints!=None` or `force_words_ids!=None`
        - *assisted decoding* by calling [`~generation.GenerationMixin.assisted_decoding`], if
            `assistant_model` is passed to `.generate()`

    You do not need to call any of the above methods directly. Pass custom parameter values to '.generate()'. To learn
    more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).

    <Tip>

    A large number of these flags control the logits or the stopping criteria of the generation. Make sure you check
    # 定义 GenerationConfig 类，用于配置生成文本时的参数
    """
    the [generate-related classes](https://huggingface.co/docs/transformers/internal/generation_utils) for a full
    description of the possible manipulations, as well as examples of their usage.
    """

    # 定义 __hash__ 方法，返回对象的哈希值
    def __hash__(self):
        return hash(self.to_json_string(ignore_metadata=True))

    # 定义 __eq__ 方法，用于比较两个 GenerationConfig 对象是否相等
    def __eq__(self, other):
        if not isinstance(other, GenerationConfig):
            return False

        # 将当前对象转换成 JSON 字符串，忽略元数据信息
        self_without_metadata = self.to_json_string(use_diff=False, ignore_metadata=True)
        # 将另一个对象转换成 JSON 字符串，忽略元数据信息
        other_without_metadata = other.to_json_string(use_diff=False, ignore_metadata=True)
        # 比较两个 JSON 字符串是否相等
        return self_without_metadata == other_without_metadata

    # 定义 __repr__ 方法，返回对象的字符串表示形式
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string(ignore_metadata=True)}"

    # 定义 save_pretrained 方法，用于保存配置到指定目录
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
    # 定义 from_pretrained 方法，用于从预训练模型加载配置
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: Union[str, os.PathLike],
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
    # 定义 _dict_from_json_file 方法，从 JSON 文件中加载字典
    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        # 打开 JSON 文件并读取其内容
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        # 将 JSON 文本转换成字典并返回
        return json.loads(text)

    # 定义另一个 @classmethod 方法
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "GenerationConfig":
        """
        从 Python 字典参数实例化一个 [`GenerationConfig`]。

        Args:
            config_dict (`Dict[str, Any]`):
                用于实例化配置对象的字典。
            kwargs (`Dict[str, Any]`):
                用于初始化配置对象的额外参数。

        Returns:
            [`GenerationConfig`]: 从这些参数实例化的配置对象。
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # 这些参数可能会被传递给内部遥测。
        # 我们移除它们，以便它们不会出现在 `return_unused_kwargs` 中。
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # 提交哈希可能已在 `config_dict` 中更新，我们不希望 kwargs 擦除该更新。
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # 下面的行允许通过 kwargs 加载特定于模型的配置，带有安全检查。
        # 参见 https://github.com/huggingface/transformers/pull/21269
        config = cls(**{**config_dict, **kwargs})
        unused_kwargs = config.update(**kwargs)

        logger.info(f"Generate config {config}")
        if return_unused_kwargs:
            return config, unused_kwargs
        else:
            return config

    def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        检查传递的字典及其嵌套字典是否具有 *torch_dtype* 键，如果不是 None，则将 torch.dtype 转换为仅类型的字符串。
        例如，`torch.float32` 被转换为 *"float32"* 字符串，然后可以存储在 json 格式中。
        """
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_torch_dtype_to_str(value)
    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        # 将配置中与默认配置属性相对应的所有属性删除，以提高可读性，并序列化为 Python 字典
        config_dict = self.to_dict()

        # 获取默认配置字典
        default_config_dict = GenerationConfig().to_dict()

        serializable_config_dict = {}

        # 仅序列化与默认配置不同的值
        for key, value in config_dict.items():
            if key not in default_config_dict or key == "transformers_version" or value != default_config_dict[key]:
                serializable_config_dict[key] = value

        # 将 torch 数据类型转换为字符串
        self.dict_torch_dtype_to_str(serializable_config_dict)
        return serializable_config_dict

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)

        # 在序列化时忽略的字段
        if "_commit_hash" in output:
            del output["_commit_hash"]
        if "_original_object_hash" in output:
            del output["_original_object_hash"]

        # 序列化此文件时的 Transformers 版本
        output["transformers_version"] = __version__

        # 将 torch 数据类型转换为字符串
        self.dict_torch_dtype_to_str(output)
        return output

    def to_json_string(self, use_diff: bool = True, ignore_metadata: bool = False) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON string.
            ignore_metadata (`bool`, *optional*, defaults to `False`):
                Whether to ignore the metadata fields present in the instance

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        if ignore_metadata:
            for metadata_field in METADATA_FIELDS:
                config_dict.pop(metadata_field, None)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON file.
        """
        # 打开指定路径的 JSON 文件，以写入模式，编码为 utf-8
        with open(json_file_path, "w", encoding="utf-8") as writer:
            # 将配置实例转换为 JSON 字符串并写入文件
            writer.write(self.to_json_string(use_diff=use_diff))

    @classmethod
    def from_model_config(cls, model_config: PretrainedConfig) -> "GenerationConfig":
        """
        Instantiates a [`GenerationConfig`] from a [`PretrainedConfig`]. This function is useful to convert legacy
        [`PretrainedConfig`] objects, which may contain generation parameters, into a stand-alone [`GenerationConfig`].

        Args:
            model_config (`PretrainedConfig`):
                The model config that will be used to instantiate the generation config.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
        # 将模型配置转换为字典
        config_dict = model_config.to_dict()
        # 移除特定键
        config_dict.pop("_from_model_config", None)
        # 从字典创建配置对象
        config = cls.from_dict(config_dict, return_unused_kwargs=False, _from_model_config=True)

        # 特殊情况：某些模型在解码器中设置了生成属性。如果在生成配置中仍未设置，使用它们。
        for decoder_name in ("decoder", "generator", "text_config"):
            if decoder_name in config_dict:
                default_generation_config = GenerationConfig()
                decoder_config = config_dict[decoder_name]
                for attr in config.to_dict().keys():
                    if attr in decoder_config and getattr(config, attr) == getattr(default_generation_config, attr):
                        setattr(config, attr, decoder_config[attr])

        # 计算配置对象的哈希值，用于检测实例是否被修改
        config._original_object_hash = hash(config)
        return config
    # 定义一个方法用于更新类实例的属性，使用 `kwargs` 中与现有属性匹配的属性进行更新，返回所有未使用的 kwargs
    def update(self, **kwargs):
        # 创建一个空列表，用于存储待删除的属性名
        to_remove = []
        # 遍历 kwargs 中的键值对
        for key, value in kwargs.items():
            # 检查当前类实例是否具有与键名匹配的属性
            if hasattr(self, key):
                # 如果存在匹配的属性，则使用 setattr 方法更新该属性的值为 kwargs 中对应的值
                setattr(self, key, value)
                # 将该属性名添加到待删除列表中
                to_remove.append(key)

        # 创建一个字典，存储未使用的 kwargs 中的键值对，即未被更新到类实例中的属性
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        # 返回未使用的 kwargs 字典
        return unused_kwargs
```