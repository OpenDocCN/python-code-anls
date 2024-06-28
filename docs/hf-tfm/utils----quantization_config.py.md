# `.\utils\quantization_config.py`

```
# 指定 Python 解释器位置和文件编码
#!/usr/bin/env python
# coding=utf-8

# 版权声明和许可信息
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

# 导入所需模块和库
import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# 导入 packaging 库中的 version 模块
from packaging import version

# 导入相对路径的模块和函数
from ..utils import is_auto_awq_available, is_torch_available, logging

# 如果 torch 可用，则导入 torch 模块
if is_torch_available():
    import torch

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义量化方法的枚举类
class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = "bitsandbytes"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    QUANTO = "quanto"

# 定义 AWQ 线性版本的枚举类
class AWQLinearVersion(str, Enum):
    GEMM = "gemm"
    GEMV = "gemv"
    EXLLAMA = "exllama"

    @staticmethod
    def from_str(version: str):
        # 将版本字符串转换为 AWQLinearVersion 枚举成员
        version = version.lower()
        if version == "gemm":
            return AWQLinearVersion.GEMM
        elif version == "gemv":
            return AWQLinearVersion.GEMV
        elif version == "exllama":
            return AWQLinearVersion.EXLLAMA
        else:
            raise ValueError(f"Unknown AWQLinearVersion {version}")

# 定义 AWQ 后端打包方法的枚举类
class AwqBackendPackingMethod(str, Enum):
    AUTOAWQ = "autoawq"
    LLMAWQ = "llm-awq"

# 数据类，用于量化配置的混合类
@dataclass
class QuantizationConfigMixin:
    """
    Mixin class for quantization config
    """

    # 量化方法
    quant_method: QuantizationMethod

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """
        从一个Python字典参数实例化一个`QuantizationConfigMixin`对象。

        Args:
            config_dict (`Dict[str, Any]`):
                用于实例化配置对象的字典。
            return_unused_kwargs (`bool`, *optional*, 默认为 `False`):
                是否返回未使用的关键字参数列表。用于`PreTrainedModel`中的`from_pretrained`方法。
            kwargs (`Dict[str, Any]`):
                其他用于初始化配置对象的参数。

        Returns:
            [`QuantizationConfigMixin`]: 从这些参数实例化的配置对象。
        """

        # 使用给定的config_dict参数实例化一个cls类的对象
        config = cls(**config_dict)

        # 准备存储要移除的关键字参数列表
        to_remove = []
        # 遍历kwargs中的关键字参数
        for key, value in kwargs.items():
            # 如果config对象具有名为key的属性，则设置该属性为value
            if hasattr(config, key):
                setattr(config, key, value)
                # 将已处理的关键字参数加入移除列表
                to_remove.append(key)
        # 从kwargs中移除已处理的关键字参数
        for key in to_remove:
            kwargs.pop(key, None)

        # 如果return_unused_kwargs为True，则返回config对象和未使用的kwargs
        if return_unused_kwargs:
            return config, kwargs
        else:
            # 否则，只返回config对象
            return config

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        将该实例保存到一个JSON文件中。

        Args:
            json_file_path (`str` or `os.PathLike`):
                要保存配置实例参数的JSON文件路径。
        """
        # 打开指定路径的JSON文件，以写入模式，使用UTF-8编码
        with open(json_file_path, "w", encoding="utf-8") as writer:
            # 获取当前配置实例的字典表示
            config_dict = self.to_dict()
            # 将配置字典转换为格式化的JSON字符串，按键排序，并添加换行符
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            # 将JSON字符串写入文件
            writer.write(json_string)

    def to_dict(self) -> Dict[str, Any]:
        """
        将该实例序列化为一个Python字典。

        Returns:
            `Dict[str, Any]`: 包含该配置实例所有属性的字典。
        """
        # 使用深拷贝获取该实例的所有属性字典并返回
        return copy.deepcopy(self.__dict__)

    def __iter__(self):
        """允许对该对象进行`dict(obj)`操作，适用于obj可能是字典或QuantizationConfigMixin的情况。"""
        # 对该实例的每个属性进行迭代，并返回属性名和对应的值
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    def __repr__(self):
        # 返回该实例的字符串表示，包括调用to_json_string()方法的结果
        return f"{self.__class__.__name__} {self.to_json_string()}"
    def to_json_string(self, use_diff: bool = True) -> str:
        """
        将当前实例序列化为 JSON 字符串。

        Args:
            use_diff (`bool`, *可选*, 默认为 `True`):
                如果设置为 `True`，则只序列化配置实例与默认 `PretrainedConfig()` 之间的差异。

        Returns:
            `str`: 包含此配置实例所有属性的 JSON 格式字符串。
        """
        if use_diff is True:
            # 使用 to_diff_dict() 方法获取差异化的配置字典
            config_dict = self.to_diff_dict()
        else:
            # 使用 to_dict() 方法获取完整的配置字典
            config_dict = self.to_dict()
        # 将配置字典转换为 JSON 格式字符串，设置缩进为2，按键排序，并加上换行符
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def update(self, **kwargs):
        """
        使用 `kwargs` 中的属性更新此类实例的属性，如果它们与现有属性匹配，则返回所有未使用的 kwargs。

        Args:
            kwargs (`Dict[str, Any]`):
                要更新此类的属性的字典。

        Returns:
            `Dict[str, Any]`: 包含所有未用于更新实例的键值对的字典。
        """
        to_remove = []
        for key, value in kwargs.items():
            # 检查当前实例是否具有要更新的属性
            if hasattr(self, key):
                # 更新实例的属性
                setattr(self, key, value)
                # 记录已更新的属性名
                to_remove.append(key)

        # 构建包含未使用的 kwargs 中未更新的所有属性的字典
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs
@dataclass
class BitsAndBytesConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `bitsandbytes`.

    This replaces `load_in_8bit` or `load_in_4bit`therefore both options are mutually exclusive.

    Currently only supports `LLM.int8()`, `FP4`, and `NF4` quantization. If more methods are added to `bitsandbytes`,
    then more arguments will be added to this class.
    """

    def __init__(
        self,
        load_in_8bit=False,
        load_in_4bit=False,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=None,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage=None,
        **kwargs,
    ):
        # 设置量化方法为 BITS_AND_BYTES
        self.quant_method = QuantizationMethod.BITS_AND_BYTES

        # 检查是否同时设置了 load_in_4bit 和 load_in_8bit，如果是则抛出错误
        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")

        # 设置 load_in_8bit 和 load_in_4bit 的属性
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit

        # 设置 LLM.int8() 的阈值
        self.llm_int8_threshold = llm_int8_threshold
        # 设置需要跳过的模块列表
        self.llm_int8_skip_modules = llm_int8_skip_modules
        # 设置是否启用 FP32 CPU 卸载
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        # 设置是否具有 FP16 权重
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        # 设置 bnb_4bit 的量化类型，默认为 "fp4"
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        # 设置是否使用双量化
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

        # 如果未指定 bnb_4bit_compute_dtype，默认设置为 torch.float32
        if bnb_4bit_compute_dtype is None:
            self.bnb_4bit_compute_dtype = torch.float32
        # 如果指定的 bnb_4bit_compute_dtype 是字符串，则转换为对应的 torch.dtype
        elif isinstance(bnb_4bit_compute_dtype, str):
            self.bnb_4bit_compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        # 如果指定的 bnb_4bit_compute_dtype 已经是 torch.dtype 类型，则直接使用
        elif isinstance(bnb_4bit_compute_dtype, torch.dtype):
            self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        else:
            raise ValueError("bnb_4bit_compute_dtype must be a string or a torch.dtype")

        # 如果未指定 bnb_4bit_quant_storage，默认设置为 torch.uint8
        if bnb_4bit_quant_storage is None:
            self.bnb_4bit_quant_storage = torch.uint8
        # 如果指定的 bnb_4bit_quant_storage 是字符串，则转换为对应的 torch.dtype
        elif isinstance(bnb_4bit_quant_storage, str):
            self.bnb_4bit_quant_storage = getattr(torch, bnb_4bit_quant_storage)
        # 如果指定的 bnb_4bit_quant_storage 已经是 torch.dtype 类型，则直接使用
        elif isinstance(bnb_4bit_quant_storage, torch.dtype):
            self.bnb_4bit_quant_storage = bnb_4bit_quant_storage
        else:
            raise ValueError("bnb_4bit_quant_storage must be a string or a torch.dtype")

        # 执行初始化后的操作
        self.post_init()

    @property
    def load_in_4bit(self):
        # 返回 load_in_4bit 的属性值
        return self._load_in_4bit

    @load_in_4bit.setter
    def load_in_4bit(self, value: bool):
        # 如果同时设置了 load_in_4bit 和 load_in_8bit，则抛出错误
        if self.load_in_8bit and value:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")
        # 设置 load_in_4bit 的属性值
        self._load_in_4bit = value

    @property
    def load_in_8bit(self):
        # 返回 load_in_8bit 的属性值
        return self._load_in_8bit
    @load_in_8bit.setter
    def load_in_8bit(self, value: bool):
        # 如果同时设置了 load_in_4bit 和 load_in_8bit，则抛出数值错误异常
        if self.load_in_4bit and value:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")
        # 设置 load_in_8bit 属性的值
        self._load_in_8bit = value

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        # 检查 llm_int8_threshold 是否为浮点数，若不是则抛出数值错误异常
        if not isinstance(self.llm_int8_threshold, float):
            raise ValueError("llm_int8_threshold must be a float")

        # 如果 llm_int8_skip_modules 不为 None 且不是字符串列表，则抛出数值错误异常
        if self.llm_int8_skip_modules is not None and not isinstance(self.llm_int8_skip_modules, list):
            raise ValueError("llm_int8_skip_modules must be a list of strings")
        # 检查 llm_int8_enable_fp32_cpu_offload 是否为布尔值，若不是则抛出数值错误异常
        if not isinstance(self.llm_int8_enable_fp32_cpu_offload, bool):
            raise ValueError("llm_int8_enable_fp32_cpu_offload must be a boolean")

        # 检查 llm_int8_has_fp16_weight 是否为布尔值，若不是则抛出数值错误异常
        if not isinstance(self.llm_int8_has_fp16_weight, bool):
            raise ValueError("llm_int8_has_fp16_weight must be a boolean")

        # 如果 bnb_4bit_compute_dtype 不为 None 且不是 torch.dtype 类型，则抛出数值错误异常
        if self.bnb_4bit_compute_dtype is not None and not isinstance(self.bnb_4bit_compute_dtype, torch.dtype):
            raise ValueError("bnb_4bit_compute_dtype must be torch.dtype")

        # 检查 bnb_4bit_quant_type 是否为字符串，若不是则抛出数值错误异常
        if not isinstance(self.bnb_4bit_quant_type, str):
            raise ValueError("bnb_4bit_quant_type must be a string")

        # 检查 bnb_4bit_use_double_quant 是否为布尔值，若不是则抛出数值错误异常
        if not isinstance(self.bnb_4bit_use_double_quant, bool):
            raise ValueError("bnb_4bit_use_double_quant must be a boolean")

        # 如果 load_in_4bit 为 True 且 bitsandbytes 版本小于 0.39.0，则抛出数值错误异常
        if self.load_in_4bit and not version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse(
            "0.39.0"
        ):
            raise ValueError(
                "4 bit quantization requires bitsandbytes>=0.39.0 - please upgrade your bitsandbytes version"
            )

    def is_quantizable(self):
        r"""
        Returns `True` if the model is quantizable, `False` otherwise.
        """
        # 返回模型是否可量化，即 load_in_8bit 或 load_in_4bit 是否为 True
        return self.load_in_8bit or self.load_in_4bit

    def quantization_method(self):
        r"""
        This method returns the quantization method used for the model. If the model is not quantizable, it returns
        `None`.
        """
        # 返回模型使用的量化方法，若模型不可量化则返回 None
        if self.load_in_8bit:
            return "llm_int8"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "fp4":
            return "fp4"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "nf4":
            return "nf4"
        else:
            return None
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        # 创建一个深拷贝，包含当前实例的所有属性
        output = copy.deepcopy(self.__dict__)
        # 将特定属性转换为字符串并提取小数点后的部分
        output["bnb_4bit_compute_dtype"] = str(output["bnb_4bit_compute_dtype"]).split(".")[1]
        output["bnb_4bit_quant_storage"] = str(output["bnb_4bit_quant_storage"]).split(".")[1]
        # 将当前实例中的特定属性直接赋给输出字典
        output["load_in_4bit"] = self.load_in_4bit
        output["load_in_8bit"] = self.load_in_8bit

        return output

    def __repr__(self):
        # 将实例转换为字典表示
        config_dict = self.to_dict()
        # 返回类名及其属性的 JSON 格式字符串表示，带缩进和按键排序
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # 获取默认配置的字典表示
        default_config_dict = BitsAndBytesConfig().to_dict()

        serializable_config_dict = {}

        # 只序列化与默认配置不同的值
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict
class ExllamaVersion(int, Enum):
    # 定义一个枚举类，表示Exllama的版本，继承自int和Enum
    ONE = 1  # 枚举项：版本一，对应数值1
    TWO = 2  # 枚举项：版本二，对应数值2


@dataclass
class GPTQConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `optimum` api for gptq quantization relying on auto_gptq backend.
    """

    def __init__(
        self,
        bits: int,
        tokenizer: Any = None,
        dataset: Optional[Union[List[str], str]] = None,
        group_size: int = 128,
        damp_percent: float = 0.1,
        desc_act: bool = False,
        sym: bool = True,
        true_sequential: bool = True,
        use_cuda_fp16: bool = False,
        model_seqlen: Optional[int] = None,
        block_name_to_quantize: Optional[str] = None,
        module_name_preceding_first_block: Optional[List[str]] = None,
        batch_size: int = 1,
        pad_token_id: Optional[int] = None,
        use_exllama: Optional[bool] = None,
        max_input_length: Optional[int] = None,
        exllama_config: Optional[Dict[str, Any]] = None,
        cache_block_outputs: bool = True,
        modules_in_block_to_quantize: Optional[List[List[str]]] = None,
        **kwargs,
    ):
        # 初始化函数，设置类的各个属性
        self.quant_method = QuantizationMethod.GPTQ  # 设置量化方法为GPTQ
        self.bits = bits  # 量化的比特数
        self.tokenizer = tokenizer  # 分词器
        self.dataset = dataset  # 数据集
        self.group_size = group_size  # 分组大小
        self.damp_percent = damp_percent  # 阻尼百分比
        self.desc_act = desc_act  # 描述行为
        self.sym = sym  # 是否对称
        self.true_sequential = true_sequential  # 真实顺序
        self.use_cuda_fp16 = use_cuda_fp16  # 是否使用CUDA FP16
        self.model_seqlen = model_seqlen  # 模型序列长度
        self.block_name_to_quantize = block_name_to_quantize  # 要量化的块名称
        self.module_name_preceding_first_block = module_name_preceding_first_block  # 第一个块之前的模块名称列表
        self.batch_size = batch_size  # 批处理大小
        self.pad_token_id = pad_token_id  # 填充token的ID
        self.use_exllama = use_exllama  # 是否使用Exllama
        self.max_input_length = max_input_length  # 最大输入长度
        self.exllama_config = exllama_config  # Exllama配置
        self.disable_exllama = kwargs.pop("disable_exllama", None)  # 禁用Exllama的选项
        self.cache_block_outputs = cache_block_outputs  # 缓存块输出
        self.modules_in_block_to_quantize = modules_in_block_to_quantize  # 要量化的块中的模块列表
        self.post_init()  # 调用初始化后的操作

    def get_loading_attributes(self):
        # 获取用于加载的属性字典
        attibutes_dict = copy.deepcopy(self.__dict__)
        loading_attibutes = ["disable_exllama", "use_exllama", "exllama_config", "use_cuda_fp16", "max_input_length"]
        loading_attibutes_dict = {i: j for i, j in attibutes_dict.items() if i in loading_attibutes}
        return loading_attibutes_dict

    def to_dict(self):
        # 转换为字典形式
        config_dict = super().to_dict()
        config_dict.pop("disable_exllama", None)  # 删除禁用Exllama的配置项
        return config_dict

    def to_dict_optimum(self):
        """
        Get compatible dict for optimum gptq config
        获取适用于最优gptq配置的兼容字典
        """
        quant_dict = self.to_dict()
        # 使其与最优配置兼容
        quant_dict["disable_exllama"] = not self.use_exllama  # 如果不使用Exllama，则禁用Exllama
        return quant_dict

    @classmethod
    # 类方法
    def from_dict_optimum(cls, config_dict):
        """
        从字典中创建最佳配置类
        
        检查配置字典中是否存在 "disable_exllama" 键
        """
        # 如果配置字典中有 "disable_exllama" 键，则根据其值设置 "use_exllama" 键
        if "disable_exllama" in config_dict:
            config_dict["use_exllama"] = not config_dict["disable_exllama"]
            # 将 "disable_exllama" 键设为 None，以避免触发警告
            config_dict["disable_exllama"] = None
        
        # 使用配置字典创建一个新的类实例
        config = cls(**config_dict)
        return config
# 使用 dataclass 装饰器定义 AwqConfig 类，用于包装通过 auto-awq 库加载的模型的所有可能属性和功能。
# 继承自 QuantizationConfigMixin 类。
@dataclass
class AwqConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `auto-awq` library awq quantization relying on auto_awq backend.

    Args:
        bits (`int`, *optional*, defaults to 4):
            The number of bits to quantize to.
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        zero_point (`bool`, *optional*, defaults to `True`):
            Whether to use zero point quantization.
        version (`AWQLinearVersion`, *optional*, defaults to `AWQLinearVersion.GEMM`):
            The version of the quantization algorithm to use. GEMM is better for big batch_size (e.g. >= 8) otherwise,
            GEMV is better (e.g. < 8 ). GEMM models are compatible with Exllama kernels.
        backend (`AwqBackendPackingMethod`, *optional*, defaults to `AwqBackendPackingMethod.AUTOAWQ`):
            The quantization backend. Some models might be quantized using `llm-awq` backend. This is useful for users
            that quantize their own models using `llm-awq` library.
        do_fuse (`bool`, *optional*, defaults to `False`):
            Whether to fuse attention and mlp layers together for faster inference
        fuse_max_seq_len (`int`, *optional*):
            The Maximum sequence length to generate when using fusing.
        modules_to_fuse (`dict`, *optional*, default to `None`):
            Overwrite the natively supported fusing scheme with the one specified by the users.
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
            Note you cannot quantize directly with transformers, please refer to `AutoAWQ` documentation for quantizing HF models.
        exllama_config (`Dict[str, Any]`, *optional*):
            You can specify the version of the exllama kernel through the `version` key, the maximum sequence
            length through the `max_input_len` key, and the maximum batch size through the `max_batch_size` key.
            Defaults to `{"version": 2, "max_input_len": 2048, "max_batch_size": 8}` if unset.
    """

    # 初始化方法，设置 AwqConfig 类的各种配置选项
    def __init__(
        self,
        bits: int = 4,  # 默认量化位数为 4 位
        group_size: int = 128,  # 默认分组大小为 128，-1 表示每列量化
        zero_point: bool = True,  # 是否使用零点量化，默认为 True
        version: AWQLinearVersion = AWQLinearVersion.GEMM,  # 量化算法版本，默认为 GEMM
        backend: AwqBackendPackingMethod = AwqBackendPackingMethod.AUTOAWQ,  # 量化后端，默认为 AUTOAWQ
        do_fuse: Optional[bool] = None,  # 是否融合注意力和 MLP 层以加快推理速度，默认为 None
        fuse_max_seq_len: Optional[int] = None,  # 使用融合时生成的最大序列长度，默认为 None
        modules_to_fuse: Optional[dict] = None,  # 覆盖默认支持的融合方案，默认为 None
        modules_to_not_convert: Optional[List] = None,  # 不进行量化的模块列表，默认为 None
        exllama_config: Optional[Dict[str, int]] = None,  # Exllama 内核的配置选项，默认为 None
        **kwargs,
    ):
        # 设置量化方法为 AWQ
        self.quant_method = QuantizationMethod.AWQ

        # 设置量化的比特数
        self.bits = bits
        # 设置量化的组大小
        self.group_size = group_size
        # 设置零点
        self.zero_point = zero_point
        # 设置版本号
        self.version = version
        # 设置后端
        self.backend = backend
        # 设置融合的最大序列长度
        self.fuse_max_seq_len = fuse_max_seq_len
        # 设置不转换的模块列表
        self.modules_to_not_convert = modules_to_not_convert
        # 设置 exllama 配置
        self.exllama_config = exllama_config

        # 设置要融合的模块列表
        self.modules_to_fuse = modules_to_fuse
        # 如果未指定是否进行融合，则根据 modules_to_fuse 的有无决定
        if do_fuse is None:
            self.do_fuse = modules_to_fuse is not None and len(modules_to_fuse) > 0
        else:
            self.do_fuse = do_fuse
        # 再次设置融合的最大序列长度（可能是重复设置，需注意）
        self.fuse_max_seq_len = fuse_max_seq_len

        # 调用后初始化方法
        self.post_init()

    def get_loading_attributes(self):
        # 深拷贝对象的字典属性
        attibutes_dict = copy.deepcopy(self.__dict__)
        # 指定需要加载的属性列表
        loading_attibutes = ["version", "do_fuse", "modules_to_fuse", "fuse_max_seq_len"]
        # 从深拷贝的属性字典中筛选出需要加载的属性构成新的字典
        loading_attibutes_dict = {i: j for i, j in attibutes_dict.items() if i in loading_attibutes}
        # 返回需要加载的属性字典
        return loading_attibutes_dict
@dataclass
class AqlmConfig(QuantizationConfigMixin):
    """
    This is a dataclass that defines configuration parameters for the AQLM quantization method.

    Args:
        in_group_size (`int`, *optional*, defaults to 8):
            The group size along the input dimension.
        out_group_size (`int`, *optional*, defaults to 1):
            The group size along the output dimension. It's recommended to always use 1.
        num_codebooks (`int`, *optional*, defaults to 1):
            Number of codebooks for the Additive Quantization procedure.
        nbits_per_codebook (`int`, *optional*, defaults to 16):
            Number of bits encoding a single codebook vector. Codebooks size is 2**nbits_per_codebook.
        linear_weights_not_to_quantize (`Optional[List[str]]`, *optional*):
            List of full paths of `nn.Linear` weight parameters that shall not be quantized.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional parameters from which to initialize the configuration object.
    """

    def __init__(
        self,
        in_group_size: int = 8,
        out_group_size: int = 1,
        num_codebooks: int = 1,
        nbits_per_codebook: int = 16,
        linear_weights_not_to_quantize: Optional[List[str]] = None,
        **kwargs,
    ):
        # 设置量化方法为 AQLM
        self.quant_method = QuantizationMethod.AQLM
        # 设置输入维度的组大小
        self.in_group_size = in_group_size
        # 设置输出维度的组大小
        self.out_group_size = out_group_size
        # 设置代码书的数量
        self.num_codebooks = num_codebooks
        # 设置每个代码书的比特数
        self.nbits_per_codebook = nbits_per_codebook
        # 设置不需要量化的线性权重列表
        self.linear_weights_not_to_quantize = linear_weights_not_to_quantize

        # 调用后续初始化方法
        self.post_init()

    def post_init(self):
        r"""
        检查参数的正确性 - 替换一些 NoneType 参数为它们的默认值。
        """
        # 如果 in_group_size 不是整数，则抛出错误
        if not isinstance(self.in_group_size, int):
            raise ValueError("in_group_size must be an integer")
        # 如果 out_group_size 不是整数，则抛出错误
        if not isinstance(self.out_group_size, int):
            raise ValueError("out_group_size must be an integer")
        # 如果 num_codebooks 不是整数，则抛出错误
        if not isinstance(self.num_codebooks, int):
            raise ValueError("num_codebooks must be an integer")
        # 如果 nbits_per_codebook 不是整数，则抛出错误
        if not isinstance(self.nbits_per_codebook, int):
            raise ValueError("nbits_per_codebook must be an integer")

        # 如果 linear_weights_not_to_quantize 不为 None 且不是字符串列表，则抛出错误
        if self.linear_weights_not_to_quantize is not None and not isinstance(
            self.linear_weights_not_to_quantize, list
        ):
            raise ValueError("linear_weights_not_to_quantize must be a list of strings")

        # 如果 linear_weights_not_to_quantize 是 None，则将其设为空列表
        if self.linear_weights_not_to_quantize is None:
            self.linear_weights_not_to_quantize = []
    Args:
        weights (`str`, *optional*, defaults to `"int8"`):
            The target dtype for the weights after quantization. Supported values are ("float8","int8","int4","int2")
        activations (`str`, *optional*):
            The target dtype for the activations after quantization. Supported values are (None,"int8","float8")
        modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have
            some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
    """
    # 初始化方法，用于设置量化器的参数和执行后续的初始化步骤
    def __init__(
        self,
        weights="int8",  # 设置权重的目标数据类型，默认为 "int8"
        activations=None,  # 设置激活函数的目标数据类型，默认为 None
        modules_to_not_convert: Optional[List] = None,  # 不需要量化的模块列表，默认为 None
        **kwargs,  # 允许接收任意额外的关键字参数
    ):
        self.quant_method = QuantizationMethod.QUANTO  # 设置量化方法为 QUANTO
        self.weights = weights  # 初始化权重目标数据类型
        self.activations = activations  # 初始化激活函数目标数据类型
        self.modules_to_not_convert = modules_to_not_convert  # 初始化不需量化的模块列表
        self.post_init()  # 调用后续初始化方法

    # 后续初始化方法，用于检查参数是否合法
    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        accepted_weights = ["float8", "int8", "int4", "int2"]  # 支持的权重数据类型列表
        accepted_activations = [None, "int8", "float8"]  # 支持的激活函数数据类型列表
        if self.weights not in accepted_weights:  # 检查权重数据类型是否在支持列表中
            raise ValueError(f"Only support weights in {accepted_weights} but found {self.weights}")
        if self.activations not in accepted_activations:  # 检查激活函数数据类型是否在支持列表中
            raise ValueError(f"Only support weights in {accepted_activations} but found {self.activations}")
```