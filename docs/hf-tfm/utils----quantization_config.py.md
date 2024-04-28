# `.\transformers\utils\quantization_config.py`

```
#!/usr/bin/env python
# coding=utf-8

# 版权声明
# 版权所有 © 2023 年 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言的权限和限制。
import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from packaging import version

from ..utils import is_auto_awq_available, is_torch_available, logging

# 如果 torch 可用
if is_torch_available():
    import torch

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义量化方法的枚举类
class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = "bitsandbytes"
    GPTQ = "gptq"
    AWQ = "awq"

# 定义 AWQ 线性版本的枚举类
class AWQLinearVersion(str, Enum):
    GEMM = "gemm"
    GEMV = "gemv"

    @staticmethod
    def from_str(version: str):
        version = version.lower()
        if version == "gemm":
            return AWQLinearVersion.GEMM
        elif version == "gemv":
            return AWQLinearVersion.GEMV
        else:
            raise ValueError(f"Unknown AWQLinearVersion {version}")

# 定义 AWQ 后端打包方法的枚举类
class AwqBackendPackingMethod(str, Enum):
    AUTOAWQ = "autoawq"
    LLMAWQ = "llm-awq"

# 定义量化配置的数据类
@dataclass
class QuantizationConfigMixin:
    """
    Mixin class for quantization config
    """

    quant_method: QuantizationMethod

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """
        Instantiates a [`QuantizationConfigMixin`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`,*optional*, defaults to `False`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`QuantizationConfigMixin`]: The configuration object instantiated from those parameters.
        """

        config = cls(**config_dict)

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config
    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `QuantizationConfig()` is serialized to JSON file.
        """
        # 打开 JSON 文件，准备写入
        with open(json_file_path, "w", encoding="utf-8") as writer:
            # 将配置实例转换为字典
            config_dict = self.to_dict()
            # 将字典转换为 JSON 字符串，带有缩进和排序
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            # 写入 JSON 字符串到文件
            writer.write(json_string)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        # 深拷贝实例的属性字典
        return copy.deepcopy(self.__dict__)

    def __repr__(self):
        # 返回实例的类名和 JSON 字符串表示
        return f"{self.__class__.__name__} {self.to_json_string()}"

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
        # 根据 use_diff 参数选择要序列化的字典
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        # 将字典转换为 JSON 字符串，带有缩进和排序
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
# 定义一个数据类，包含了所有可以在使用 `bitsandbytes` 加载的模型中进行操作的属性和特性的包装类
@dataclass
class BitsAndBytesConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `bitsandbytes`.

    This replaces `load_in_8bit` or `load_in_4bit`therefore both options are mutually exclusive.

    Currently only supports `LLM.int8()`, `FP4`, and `NF4` quantization. If more methods are added to `bitsandbytes`,
    then more arguments will be added to this class.

    """

    # 初始化方法，接受一系列参数
    def __init__(
        self,
        load_in_8bit=False,  # 是否以8位加载模型，默认为False
        load_in_4bit=False,  # 是否以4位加载模型，默认为False
        llm_int8_threshold=6.0,  # LLM.int8() 的阈值，默认为6.0
        llm_int8_skip_modules=None,  # 跳过的模块列表，默认为None
        llm_int8_enable_fp32_cpu_offload=False,  # 是否启用 FP32 CPU 卸载，默认为False
        llm_int8_has_fp16_weight=False,  # 是否有FP16权重，默认为False
        bnb_4bit_compute_dtype=None,  # 4位计算的数据类型，默认为None
        bnb_4bit_quant_type="fp4",  # 4位量化类型，默认为"fp4"
        bnb_4bit_use_double_quant=False,  # 是否使用双重量化，默认为False
        **kwargs,  # 其他关键字参数
    ):
        # 设置量化方法为 BITS_AND_BYTES
        self.quant_method = QuantizationMethod.BITS_AND_BYTES
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.llm_int8_threshold = llm_int8_threshold
        self.llm_int8_skip_modules = llm_int8_skip_modules
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

        # 根据不同情况设置 4位计算的数据类型
        if bnb_4bit_compute_dtype is None:
            self.bnb_4bit_compute_dtype = torch.float32
        elif isinstance(bnb_4bit_compute_dtype, str):
            self.bnb_4bit_compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        elif isinstance(bnb_4bit_compute_dtype, torch.dtype):
            self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        else:
            raise ValueError("bnb_4bit_compute_dtype must be a string or a torch.dtype")

        # 调用初始化后的方法
        self.post_init()
    # 在初始化之后执行的方法，用于检查参数是否正确，并将一些 NoneType 参数替换为它们的默认值
    def post_init(self):
        """
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        # 检查 llm_int8_threshold 是否为 float 类型
        if not isinstance(self.llm_int8_threshold, float):
            raise ValueError("llm_int8_threshold must be a float")

        # 检查 llm_int8_skip_modules 是否为字符串列表类型
        if self.llm_int8_skip_modules is not None and not isinstance(self.llm_int8_skip_modules, list):
            raise ValueError("llm_int8_skip_modules must be a list of strings")
        # 检查 llm_int8_enable_fp32_cpu_offload 是否为布尔类型
        if not isinstance(self.llm_int8_enable_fp32_cpu_offload, bool):
            raise ValueError("llm_int8_enable_fp32_cpu_offload must be a boolean")

        # 检查 llm_int8_has_fp16_weight 是否为布尔类型
        if not isinstance(self.llm_int8_has_fp16_weight, bool):
            raise ValueError("llm_int8_has_fp16_weight must be a boolean")

        # 检查 bnb_4bit_compute_dtype 是否为 torch.dtype 类型
        if self.bnb_4bit_compute_dtype is not None and not isinstance(self.bnb_4bit_compute_dtype, torch.dtype):
            raise ValueError("bnb_4bit_compute_dtype must be torch.dtype")

        # 检查 bnb_4bit_quant_type 是否为字符串类型
        if not isinstance(self.bnb_4bit_quant_type, str):
            raise ValueError("bnb_4bit_quant_type must be a string")

        # 检查 bnb_4bit_use_double_quant 是否为布尔类型
        if not isinstance(self.bnb_4bit_use_double_quant, bool):
            raise ValueError("bnb_4bit_use_double_quant must be a boolean")

        # 如果 load_in_4bit 为 True，且 bitsandbytes 版本小于 0.39.0，则抛出异常
        if self.load_in_4bit and not version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse(
            "0.39.0"
        ):
            raise ValueError(
                "4 bit quantization requires bitsandbytes>=0.39.0 - please upgrade your bitsandbytes version"
            )

    # 返回模型是否可量化
    def is_quantizable(self):
        """
        Returns `True` if the model is quantizable, `False` otherwise.
        """
        return self.load_in_8bit or self.load_in_4bit

    # 返回模型使用的量化方法，如果模型不可量化，则返回 None
    def quantization_method(self):
        """
        This method returns the quantization method used for the model. If the model is not quantizable, it returns
        `None`.
        """
        if self.load_in_8bit:
            return "llm_int8"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "fp4":
            return "fp4"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "nf4":
            return "nf4"
        else:
            return None

    # 将实例序列化为 Python 字典
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        # 将 bnb_4bit_compute_dtype 转换为字符串形式
        output["bnb_4bit_compute_dtype"] = str(output["bnb_4bit_compute_dtype"]).split(".")[1]

        return output

    # 返回实例的字符串表示形式
    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"
    # 将所有与默认配置属性对应的属性从配置中移除，以提高可读性，并序列化为Python字典
    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        # 将配置转换为字典
        config_dict = self.to_dict()

        # 获取默认配置字典
        default_config_dict = BitsAndBytesConfig().to_dict()

        serializable_config_dict = {}

        # 只序列化与默认配置不同的值
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict
# 定义一个枚举类 ExllamaVersion，继承自 int 和 Enum
class ExllamaVersion(int, Enum):
    # 定义枚举值 ONE，值为 1
    ONE = 1
    # 定义枚举值 TWO，值为 2

# 使用 dataclass 装饰器定义类 GPTQConfig，继承自 QuantizationConfigMixin
@dataclass
class GPTQConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `optimum` api for gptq quantization relying on auto_gptq backend.
    """

    # 初始化方法，接收多个参数
    def __init__(
        self,
        bits: int,  # 位数
        tokenizer: Any = None,  # 分词器
        dataset: Optional[Union[List[str], str]] = None,  # 数据集
        group_size: int = 128,  # 组大小
        damp_percent: float = 0.1,  # 阻尼百分比
        desc_act: bool = False,  # 描述行为
        sym: bool = True,  # 是否对称
        true_sequential: bool = True,  # 真正的顺序
        use_cuda_fp16: bool = False,  # 使用 CUDA FP16
        model_seqlen: Optional[int] = None,  # 模型序列长度
        block_name_to_quantize: Optional[str] = None,  # 要量化的块名称
        module_name_preceding_first_block: Optional[List[str]] = None,  # 第一个块之前的模块名称
        batch_size: int = 1,  # 批量大小
        pad_token_id: Optional[int] = None,  # 填充标记 ID
        use_exllama: Optional[bool] = None,  # 使用 Exllama
        max_input_length: Optional[int] = None,  # 最大输入长度
        exllama_config: Optional[Dict[str, Any]] = None,  # Exllama 配置
        cache_block_outputs: bool = True,  # 缓存块输出
        modules_in_block_to_quantize: Optional[List[List[str]]] = None,  # 要量化的块中的模块
        **kwargs,  # 其他关键字参数
    ):
        # 设置量化方法为 GPTQ
        self.quant_method = QuantizationMethod.GPTQ
        self.bits = bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.desc_act = desc_act
        self.sym = sym
        self.true_sequential = true_sequential
        self.use_cuda_fp16 = use_cuda_fp16
        self.model_seqlen = model_seqlen
        self.block_name_to_quantize = block_name_to_quantize
        self.module_name_preceding_first_block = module_name_preceding_first_block
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.use_exllama = use_exllama
        self.max_input_length = max_input_length
        self.exllama_config = exllama_config
        self.disable_exllama = kwargs.pop("disable_exllama", None)
        self.cache_block_outputs = cache_block_outputs
        self.modules_in_block_to_quantize = modules_in_block_to_quantize
        self.post_init()  # 调用 post_init 方法

    # 获取加载属性的方法
    def get_loading_attributes(self):
        attibutes_dict = copy.deepcopy(self.__dict__)
        loading_attibutes = ["disable_exllama", "use_exllama", "exllama_config", "use_cuda_fp16", "max_input_length"]
        loading_attibutes_dict = {i: j for i, j in attibutes_dict.items() if i in loading_attibutes}
        return loading_attibutes_dict

    # 转换为字典的方法
    def to_dict(self):
        config_dict = super().to_dict()
        config_dict.pop("disable_exllama", None)
        return config_dict

    # 转换为适用于 optimum 的字典的方法
    def to_dict_optimum(self):
        """
        Get compatible dict for optimum gptq config
        """
        quant_dict = self.to_dict()
        # 使其与 optimum 配置兼容
        quant_dict["disable_exllama"] = not self.use_exllama
        return quant_dict

    @classmethod
    # 从给定的配置字典中获取与最佳 gptq 配置字典兼容的类
    def from_dict_optimum(cls, config_dict):
        """
        Get compatible class with optimum gptq config dict
        """

        # 如果配置字典中包含 "disable_exllama" 键
        if "disable_exllama" in config_dict:
            # 将 "use_exllama" 键设置为与 "disable_exllama" 相反的布尔值
            config_dict["use_exllama"] = not config_dict["disable_exllama"]
            # 将 "disable_exllama" 键设置为 None，以避免触发警告
            config_dict["disable_exllama"] = None

        # 使用配置字典创建一个新的配置对象
        config = cls(**config_dict)
        # 返回新的配置对象
        return config
# 使用 dataclass 装饰器定义 AwqConfig 类，该类包含了所有可以用于已加载模型的属性和特性的包装器
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
            GEMV is better (e.g. < 8 )
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
    """

    # 初始化方法，设置各个参数的默认值
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        version: AWQLinearVersion = AWQLinearVersion.GEMM,
        backend: AwqBackendPackingMethod = AwqBackendPackingMethod.AUTOAWQ,
        do_fuse: Optional[bool] = None,
        fuse_max_seq_len: Optional[int] = None,
        modules_to_fuse: Optional[dict] = None,
        modules_to_not_convert: Optional[List] = None,
        **kwargs,
    # 初始化量化方法为 AWQ
    ):
        self.quant_method = QuantizationMethod.AWQ

        # 设置位数、组大小、零点、版本、后端、融合最大序列长度、不转换的模块列表等属性
        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version
        self.backend = backend
        self.fuse_max_seq_len = fuse_max_seq_len
        self.modules_to_not_convert = modules_to_not_convert

        # 设置需要融合的模块列表
        self.modules_to_fuse = modules_to_fuse
        # 如果未指定是否融合，则根据是否存在需要融合的模块来确定是否融合
        if do_fuse is None:
            self.do_fuse = modules_to_fuse is not None and len(modules_to_fuse) > 0
        else:
            self.do_fuse = do_fuse
        # 设置融合最大序列长度
        self.fuse_max_seq_len = fuse_max_seq_len

        # 调用后初始化方法
        self.post_init()

    # 获取加载属性的方法
    def get_loading_attributes(self):
        # 深拷贝当前对象的属性字典
        attibutes_dict = copy.deepcopy(self.__dict__)
        # 需要加载的属性列表
        loading_attibutes = ["do_fuse", "modules_to_fuse", "fuse_max_seq_len"]
        # 从属性字典中筛选出需要加载的属性
        loading_attibutes_dict = {i: j for i, j in attibutes_dict.items() if i in loading_attibutes}
        # 返回需要加载的属性字典
        return loading_attibutes_dict
```