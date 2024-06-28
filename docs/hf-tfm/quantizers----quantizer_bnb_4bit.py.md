# `.\quantizers\quantizer_bnb_4bit.py`

```py
# 导入必要的模块和库

# 版权声明和许可证信息
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入 importlib 模块，用于动态加载模块
import importlib

# 导入类型检查相关模块和类型
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# 导入版本控制相关模块
from packaging import version

# 导入基础量化类 HfQuantizer
from .base import HfQuantizer

# 导入工具函数 get_module_from_name
from .quantizers_utils import get_module_from_name

# 如果是类型检查模式，则导入 PreTrainedModel 类
if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

# 导入加速库是否可用检查函数
from ..utils import is_accelerate_available, is_bitsandbytes_available, is_torch_available, logging

# 如果 torch 可用，则导入 torch 相关模块和类
if is_torch_available():
    import torch
    from ..pytorch_utils import Conv1D

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义 Bnb4BitHfQuantizer 类，继承自 HfQuantizer
class Bnb4BitHfQuantizer(HfQuantizer):
    """
    从 bitsandbytes.py 量化方法中实现的 4 位量化:
        在加载之前: 将 transformer 层转换为 Linear4bit
        在加载期间: 加载 16 位权重并传递给层对象
        在量化后: 在第一次 .cuda() 调用时将 Linear4bit 中的单个权重量化为 4 位
        保存:
            从状态字典中，像往常一样; 保存权重和 `quant_state` 组件
        加载:
            需要定位 `quant_state` 组件并传递给 Param4bit 构造函数
    """

    # 使用 keep_in_fp32 模块
    use_keep_in_fp32_modules = True

    # 需要参数量化
    requires_parameters_quantization = True

    # 不需要校准
    requires_calibration = False

    # 必需的软件包
    required_packages = ["bitsandbytes", "accelerate"]

    # 初始化方法，接受量化配置和其他关键字参数
    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        # 如果量化配置中定义了 llm_int8_skip_modules，则使用它来指定不转换的模块
        if self.quantization_config.llm_int8_skip_modules is not None:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

# 结束 Bnb4BitHfQuantizer 类定义
    # 检查是否安装了加速库和 bitsandbytes 库，如果没有则引发 ImportError 异常
    if not (is_accelerate_available() and is_bitsandbytes_available()):
        raise ImportError(
            "Using `bitsandbytes` 8-bit quantization requires Accelerate: `pip install accelerate` "
            "and the latest version of bitsandbytes: `pip install -i https://pypi.org/simple/ bitsandbytes`"
        )

    # 检查是否从 TensorFlow 或 Flax 来源转换权重，这种情况下不支持转换，需使用 PyTorch 格式的权重
    if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
        raise ValueError(
            "Converting into 4-bit or 8-bit weights from tf/flax weights is currently not supported, please make"
            " sure the weights are in PyTorch format."
        )

    # 检查是否有可用的 CUDA 设备，如果没有找到 GPU 则引发 RuntimeError 异常
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. A GPU is needed for quantization.")

    # 获取参数中的 device_map，检查其类型为 dict，并且未开启 llm_int8_enable_fp32_cpu_offload 选项
    device_map = kwargs.get("device_map", None)
    if (
        device_map is not None
        and isinstance(device_map, dict)
        and not self.quantization_config.llm_int8_enable_fp32_cpu_offload
    ):
        # 剔除 self.modules_to_not_convert 中的模块，生成不包含 lm_head 的新 device_map
        device_map_without_lm_head = {
            key: device_map[key] for key in device_map.keys() if key not in self.modules_to_not_convert
        }
        # 如果 device_map_without_lm_head 中的值包含 "cpu" 或 "disk"，则引发 ValueError 异常
        if "cpu" in device_map_without_lm_head.values() or "disk" in device_map_without_lm_head.values():
            raise ValueError(
                """
                Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the
                quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules
                in 32-bit, you need to set `load_in_8bit_fp32_cpu_offload=True` and pass a custom `device_map` to
                `from_pretrained`. Check
                https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu
                for more details.
                """
            )

    # 检查 bitsandbytes 库的版本是否小于 0.39.0，如果是则引发 ValueError 异常
    if version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.39.0"):
        raise ValueError(
            "You have a version of `bitsandbytes` that is not compatible with 4bit inference and training"
            " make sure you have the latest version of `bitsandbytes` installed"
        )
    # 调整目标数据类型，确保与加速库的版本兼容
    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        # 检查加速库版本是否大于0.19.0
        if version.parse(importlib.metadata.version("accelerate")) > version.parse("0.19.0"):
            # 如果是最新版本，导入加速库的自定义数据类型
            from accelerate.utils import CustomDtype

            # 如果目标数据类型不是 torch.int8
            if target_dtype != torch.int8:
                # 记录日志，说明目标数据类型被替换为 CustomDtype.INT4，用于4位BnB量化
                logger.info(f"target_dtype {target_dtype} is replaced by `CustomDtype.INT4` for 4-bit BnB quantization")
            # 返回 CustomDtype.INT4 作为新的目标数据类型
            return CustomDtype.INT4
        else:
            # 如果加速库版本过低，抛出数值错误，提示用户升级加速库以支持自动设备映射计算
            raise ValueError(
                "You are using `device_map='auto'` on a 4bit loaded version of the model. To automatically compute"
                " the appropriate device map, you should upgrade your `accelerate` library,"
                "`pip install --upgrade accelerate` or install it from source to support fp4 auto device map"
                "calculation. You may encounter unexpected behavior, or pass your own device map"
            )

    # 检查量化参数是否符合预期
    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        # 导入 bitsandbytes 库作为 bnb
        import bitsandbytes as bnb

        # 从模型中获取指定参数的模块和张量名称
        module, tensor_name = get_module_from_name(model, param_name)
        
        # 如果参数是 bnb.nn.Params4bit 类型，返回 True
        if isinstance(module._parameters.get(tensor_name, None), bnb.nn.Params4bit):
            # TODO: 添加序列化实现后，添加加载组件的数据类型检查
            return True
        # 如果模块是 bnb.nn.Linear4bit 并且张量名称是 "bias"，返回 True
        elif isinstance(module, bnb.nn.Linear4bit) and tensor_name == "bias":
            # bias 可能被 accelerate 的 regular set_module_tensor_to_device() 加载，
            # 但在那里会错误地使用未初始化的权重。
            return True
        else:
            # 其他情况返回 False
            return False

    # 创建量化参数
    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
    ):
        # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer.adjust_max_memory
        # 调整最大内存限制，确保在量化过程中有足够的空间
        def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
            # 遍历每个键值对，将值乘以0.90以腾出更多的空间用于量化过程中创建的缓冲区
            max_memory = {key: val * 0.90 for key, val in max_memory.items()}
            # 返回调整后的最大内存限制字典
            return max_memory

        # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer.update_torch_dtype
    # 更新 torch 数据类型，以及返回更新后的 torch 数据类型
    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        # 如果 torch 数据类型为空
        if torch_dtype is None:
            # 强制将 `dtype` 设置为 float16，这是 `bitsandbytes` 的要求，以便在8位或4位下启用模型加载
            logger.info(
                "Overriding torch_dtype=%s with `torch_dtype=torch.float16` due to "
                "requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. "
                "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
                " torch_dtype=torch.float16 to remove this warning.",
                torch_dtype,
            )
            # 将 torch 数据类型强制设为 float16
            torch_dtype = torch.float16
        # 返回更新后的 torch 数据类型
        return torch_dtype

    # 从 transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer.update_device_map 复制而来
    # 更新设备映射
    def update_device_map(self, device_map):
        # 如果设备映射为空
        if device_map is None:
            # 将设备映射设为当前 CUDA 设备
            device_map = {"": torch.cuda.current_device()}
            # 输出日志信息
            logger.info(
                "The device_map was not initialized. "
                "Setting device_map to {'':torch.cuda.current_device()}. "
                "If you want to use the model for inference, please set device_map ='auto' "
            )
        # 返回更新后的设备映射
        return device_map

    # 从 transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer._process_model_before_weight_loading 复制而来
    # 在加载权重之前处理模型
    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
        from ..integrations import get_keys_to_not_convert, replace_with_bnb_linear
        # 从上层模块导入函数 `get_keys_to_not_convert` 和 `replace_with_bnb_linear`

        load_in_8bit_fp32_cpu_offload = self.quantization_config.llm_int8_enable_fp32_cpu_offload
        # 从量化配置中获取是否启用 8 位整数加载到 FP32 CPU 卸载的设置

        # 将一些模块（如 lm_head）保持在其原始数据类型中，以确保数值稳定性
        if self.quantization_config.llm_int8_skip_modules is None:
            self.modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules
        # 如果未指定不转换的模块列表，则使用 get_keys_to_not_convert 函数从模型中获取
        # 否则，使用量化配置中指定的不转换的模块列表

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]
        # 如果不转换的模块不是列表类型，则转换为列表类型

        self.modules_to_not_convert.extend(keep_in_fp32_modules)
        # 将 keep_in_fp32_modules 中的模块添加到 self.modules_to_not_convert 中

        # 扩展 `self.modules_to_not_convert` 到需要卸载到 `cpu` 或 `disk` 的键
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]
            # 在 device_map 中查找值为 "disk" 或 "cpu" 的键

            if len(keys_on_cpu) > 0 and not load_in_8bit_fp32_cpu_offload:
                raise ValueError(
                    "If you want to offload some keys to `cpu` or `disk`, you need to set "
                    "`llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be "
                    " converted to 8-bit but kept in 32-bit."
                )
            # 如果有键被卸载到 `cpu` 或 `disk` 但未设置加载到 FP32 CPU 卸载，则抛出 ValueError 异常

            self.modules_to_not_convert.extend(keys_on_cpu)
            # 将键添加到 self.modules_to_not_convert 中

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        # 使用 replace_with_bnb_linear 函数替换模型中的模块，传入不转换的模块列表和量化配置

        # TODO: consider bringing replace_with_bnb_linear() code from ..integrations/bitsandbyter.py to here
        # TODO：考虑将来自 ..integrations/bitsandbyter.py 的 replace_with_bnb_linear() 代码引入到此处

        model.config.quantization_config = self.quantization_config
        # 设置模型配置的量化配置为当前的量化配置

    # 从 transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer._process_model_after_weight_loading 复制，将 8bit->4bit
    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        model.is_loaded_in_4bit = True
        model.is_4bit_serializable = self.is_serializable
        # 将模型标记为已加载 4 位，并根据可序列化性设置 4 位模型是否可序列化
        return model

    @property
    def is_serializable(self):
        _is_4bit_serializable = version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.41.3")
        # 检查当前安装的 bitsandbytes 版本是否支持 4 位模型的序列化

        if not _is_4bit_serializable:
            logger.warning(
                "You are calling `save_pretrained` to a 4-bit converted model, but your `bitsandbytes` version doesn't support it. "
                "If you want to save 4-bit models, make sure to have `bitsandbytes>=0.41.3` installed."
            )
            return False
        # 如果不支持 4 位模型的序列化，记录警告信息并返回 False

        return True
        # 否则返回 True，表示支持 4 位模型的序列化

    @property
    def is_trainable(self) -> bool:
        return True
    # 表示当前对象是可训练的，始终返回 True
```