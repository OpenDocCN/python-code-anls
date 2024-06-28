# `.\generation\configuration_utils.py`

```py
# coding=utf-8
# 声明编码格式为UTF-8，确保文件中可以包含非ASCII字符
# Copyright 2022 The HuggingFace Inc. team.
# 版权声明，指出代码的版权归属于HuggingFace Inc.团队。

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据Apache License, Version 2.0许可证授权，使用该文件需要遵守许可证规定。
# you may not use this file except in compliance with the License.
# 除非符合许可证的规定，否则不得使用此文件。
# You may obtain a copy of the License at
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 除非适用法律要求或书面同意，否则依照“原样”分发本软件，不附带任何明示或暗示的担保或条件。
# 可以在许可证下查看特定语言的权限和限制。

""" Generation configuration class and utilities."""
# 生成配置类和实用程序的说明文档。

import copy
# 导入copy模块，用于复制对象
import json
# 导入json模块，用于JSON数据的处理
import os
# 导入os模块，提供与操作系统交互的功能
import warnings
# 导入warnings模块，用于管理警告信息
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
# 导入类型提示相关的模块和类型

from .. import __version__
# 从父级目录导入__version__，用于获取当前模块的版本信息
from ..configuration_utils import PretrainedConfig
# 从父级目录导入PretrainedConfig类，用于处理预训练配置相关的功能
from ..utils import (
    GENERATION_CONFIG_NAME,
    ExplicitEnum,
    PushToHubMixin,
    cached_file,
    download_url,
    extract_commit_hash,
    is_remote_url,
    logging,
)
# 从父级目录的utils模块导入各种工具函数和类

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    # 如果在类型检查模式下，导入预训练模型相关的模块

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器对象
METADATA_FIELDS = ("_from_model_config", "_commit_hash", "_original_object_hash", "transformers_version")
# 元数据字段元组，包含模型配置来源、提交哈希、原始对象哈希和transformers版本信息


class GenerationMode(ExplicitEnum):
    """
    Possible generation modes, downstream of the [`~generation.GenerationMixin.generate`] method.
    """
    # 生成模式枚举类，表示`generate`方法的可能生成模式

    # Non-beam methods
    CONTRASTIVE_SEARCH = "contrastive_search"
    # 对比搜索方法
    GREEDY_SEARCH = "greedy_search"
    # 贪婪搜索方法
    SAMPLE = "sample"
    # 随机采样方法
    ASSISTED_GENERATION = "assisted_generation"
    # 辅助生成方法

    # Beam methods
    BEAM_SEARCH = "beam_search"
    # Beam搜索方法
    BEAM_SAMPLE = "beam_sample"
    # Beam采样方法
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
    # 限制Beam搜索方法
    GROUP_BEAM_SEARCH = "group_beam_search"
    # 分组Beam搜索方法


class GenerationConfig(PushToHubMixin):
    # no-format
    r"""
    Class that holds a configuration for a generation task. A `generate` call supports the following generation methods
    ```
    # 生成任务配置类，支持以下生成方法
    """
    Defines special methods for hash, equality comparison, and representation of GenerationConfig objects.
    """

    # 计算对象的哈希值，基于忽略元数据的 JSON 字符串表示
    def __hash__(self):
        return hash(self.to_json_string(ignore_metadata=True))

    # 判断两个 GenerationConfig 对象是否相等，忽略元数据进行比较
    def __eq__(self, other):
        # 如果 other 不是 GenerationConfig 类型，直接返回 False
        if not isinstance(other, GenerationConfig):
            return False
        
        # 分别获取去除元数据后的 JSON 字符串
        self_without_metadata = self.to_json_string(use_diff=False, ignore_metadata=True)
        other_without_metadata = other.to_json_string(use_diff=False, ignore_metadata=True)
        
        # 比较两个 JSON 字符串是否相等
        return self_without_metadata == other_without_metadata

    # 返回 GenerationConfig 对象的字符串表示，包括忽略元数据的 JSON 字符串
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string(ignore_metadata=True)}"
    def get_generation_mode(self, assistant_model: Optional["PreTrainedModel"] = None) -> GenerationMode:
        """
        Returns the generation mode triggered by the [`GenerationConfig`] instance.

        Arg:
            assistant_model (`PreTrainedModel`, *optional*):
                The assistant model to be used for assisted generation. If set, the generation mode will be
                assisted generation.

        Returns:
            `GenerationMode`: The generation mode triggered by the instance.
        """
        # Determine generation mode based on various configuration parameters
        if self.constraints is not None or self.force_words_ids is not None:
            generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
        elif self.num_beams == 1:
            if self.do_sample is False:
                if (
                    self.top_k is not None
                    and self.top_k > 1
                    and self.penalty_alpha is not None
                    and self.penalty_alpha > 0
                ):
                    generation_mode = GenerationMode.CONTRASTIVE_SEARCH
                else:
                    generation_mode = GenerationMode.GREEDY_SEARCH
            else:
                generation_mode = GenerationMode.SAMPLE
        else:
            if self.num_beam_groups > 1:
                generation_mode = GenerationMode.GROUP_BEAM_SEARCH
            elif self.do_sample is True:
                generation_mode = GenerationMode.BEAM_SAMPLE
            else:
                generation_mode = GenerationMode.BEAM_SEARCH

        # Modify generation mode if assistant model is specified for assisted generation
        if assistant_model is not None or self.prompt_lookup_num_tokens is not None:
            if generation_mode in (GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE):
                generation_mode = GenerationMode.ASSISTED_GENERATION
            else:
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generation. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )
        # Return the determined generation mode
        return generation_mode

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Saves the current configuration to the specified directory.

        Args:
            save_directory (Union[str, os.PathLike]): Directory where the configuration should be saved.
            config_file_name (Optional[Union[str, os.PathLike]], *optional*):
                Name for the configuration file. If not provided, a default name will be used.
            push_to_hub (bool, *optional*):
                Whether to push the saved configuration to the model hub (if applicable).
            **kwargs:
                Additional keyword arguments for future expansion.
        """

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
        """
        Creates an instance of the class from a pretrained model.

        Args:
            pretrained_model_name (Union[str, os.PathLike]): Name or path of the pretrained model.
            config_file_name (Optional[Union[str, os.PathLike]], *optional*):
                Name for the configuration file. If not provided, a default name will be used.
            cache_dir (Optional[Union[str, os.PathLike]], *optional*):
                Directory to cache downloaded files (if applicable).
            force_download (bool, *optional*):
                Whether to force re-download of the model files, ignoring any cached versions.
            local_files_only (bool, *optional*):
                Whether to only consider local files as sources for the model, ignoring any remote repositories.
            token (Optional[Union[str, bool]], *optional*):
                Access token for private model repositories (if applicable).
            revision (str, *optional*):
                Revision or version of the model to load.
            **kwargs:
                Additional keyword arguments for future expansion.

        Returns:
            Instance of the class loaded from the pretrained model.
        """
    # 从给定的 JSON 文件中读取内容并将其解析为 Python 字典
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "GenerationConfig":
        """
        从一个 Python 字典参数实例化一个 GenerationConfig 对象。

        Args:
            config_dict (`Dict[str, Any]`):
                将用于实例化配置对象的字典。
            kwargs (`Dict[str, Any]`):
                用于初始化配置对象的额外参数。

        Returns:
            [`GenerationConfig`]: 从这些参数实例化的配置对象。
        """
        # 是否返回未使用的关键字参数，默认为 False
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # 移除内部遥测用的参数，以防止它们出现在 `return_unused_kwargs` 中
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # 如果 `_commit_hash` 在 kwargs 中且在 config_dict 中，则更新 `_commit_hash`
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # 下面的语句允许通过 kwargs 加载特定于模型的配置，并进行安全检查。
        # 参考：https://github.com/huggingface/transformers/pull/21269
        config = cls(**{**config_dict, **kwargs})
        # 更新配置，并返回未使用的关键字参数
        unused_kwargs = config.update(**kwargs)

        # 记录生成的配置信息
        logger.info(f"Generate config {config}")
        if return_unused_kwargs:
            return config, unused_kwargs
        else:
            return config

    # 将字典及其嵌套字典中的 `torch_dtype` 键转换为字符串形式，例如 `torch.float32` 转换为 `"float32"`
    def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
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
        # 将当前配置转换为字典形式
        config_dict = self.to_dict()

        # 获取默认配置的字典形式
        default_config_dict = GenerationConfig().to_dict()

        # 初始化一个空字典，用于存储与默认配置不同的配置项
        serializable_config_dict = {}

        # 只序列化与默认配置不同的值
        for key, value in config_dict.items():
            # 如果配置项不在默认配置中，或者是特定例外项，或者值不同，则加入序列化字典中
            if key not in default_config_dict or key == "transformers_version" or value != default_config_dict[key]:
                serializable_config_dict[key] = value

        # 转换字典中的 torch 数据类型为字符串表示
        self.dict_torch_dtype_to_str(serializable_config_dict)
        return serializable_config_dict

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        # 深拷贝对象的 __dict__ 属性，得到一个副本
        output = copy.deepcopy(self.__dict__)

        # 在序列化时忽略的字段
        if "_commit_hash" in output:
            del output["_commit_hash"]
        if "_original_object_hash" in output:
            del output["_original_object_hash"]

        # 序列化时记录 Transformers 版本信息
        output["transformers_version"] = __version__

        # 转换字典中的 torch 数据类型为字符串表示
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
        # 根据 use_diff 参数决定是否只序列化配置实例与默认 GenerationConfig() 之间的差异
        if use_diff is True:
            config_dict = self.to_diff_dict()  # 调用实例方法获取配置实例与默认配置之间的差异字典
        else:
            config_dict = self.to_dict()  # 调用实例方法获取完整的配置实例字典

        # 如果 ignore_metadata 参数为 True，则移除配置字典中的元数据字段
        if ignore_metadata:
            for metadata_field in METADATA_FIELDS:
                config_dict.pop(metadata_field, None)

        # 定义一个函数，将字典中的键转换为字符串类型
        def convert_keys_to_string(obj):
            if isinstance(obj, dict):
                return {str(key): convert_keys_to_string(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys_to_string(item) for item in obj]
            else:
                return obj

        # 转换配置字典中所有键为字符串类型
        config_dict = convert_keys_to_string(config_dict)

        # 将转换后的配置字典转换为带缩进、按键排序的 JSON 格式字符串，并添加换行符
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
        # 打开指定路径的 JSON 文件，并将实例转换为 JSON 字符串后写入文件
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    @classmethod
    def from_model_config(cls, model_config: PretrainedConfig) -> "GenerationConfig":
        """
        从一个预训练配置 (`PretrainedConfig`) 实例化一个生成配置 (`GenerationConfig`)。
        这个函数用于将可能包含生成参数的旧式预训练配置对象转换为独立的生成配置对象。

        Args:
            model_config (`PretrainedConfig`):
                将用于实例化生成配置的模型配置。

        Returns:
            [`GenerationConfig`]: 从这些参数实例化的配置对象。
        """
        # 将模型配置转换为字典
        config_dict = model_config.to_dict()
        # 移除特定的属性，这些属性不应该用于构建生成配置
        config_dict.pop("_from_model_config", None)
        # 通过字典创建生成配置对象，确保不返回未使用的关键字参数
        config = cls.from_dict(config_dict, return_unused_kwargs=False, _from_model_config=True)

        # 特殊情况：某些模型在解码器中设置了生成属性。如果生成配置中仍未设置这些属性，则使用解码器中的值。
        for decoder_name in ("decoder", "generator", "text_config"):
            if decoder_name in config_dict:
                default_generation_config = GenerationConfig()
                decoder_config = config_dict[decoder_name]
                # 检查生成配置中的每个属性，如果属性在解码器配置中存在且生成配置中未设置，则设置为解码器中的值
                for attr in config.to_dict().keys():
                    if attr in decoder_config and getattr(config, attr) == getattr(default_generation_config, attr):
                        setattr(config, attr, decoder_config[attr])

        # 计算对象的哈希值，用于检测实例是否已修改
        config._original_object_hash = hash(config)
        return config

    def update(self, **kwargs):
        """
        使用 `kwargs` 中的属性更新该类实例的属性，如果属性匹配现有属性，则返回所有未使用的 kwargs。

        Args:
            kwargs (`Dict[str, Any]`):
                尝试更新此类的属性的属性字典。

        Returns:
            `Dict[str, Any]`: 包含所有未用于更新实例的键值对的字典。
        """
        to_remove = []
        # 遍历传入的关键字参数
        for key, value in kwargs.items():
            # 如果类实例具有这个属性，则更新为传入的值，并记录已更新的属性名
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # 确保更新后的实例仍然有效
        self.validate()

        # 返回所有未使用的关键字参数，即未更新到类实例的参数
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs
```