# `.\quantizers\quantizer_bnb_8bit.py`

```py
# 导入 importlib 模块，用于动态加载其他模块
import importlib
# 导入类型提示相关模块：TYPE_CHECKING, Any, Dict, List, Optional, Union
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# 导入版本控制相关模块
from packaging import version

# 从当前包中导入 HfQuantizer 基类
from .base import HfQuantizer

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从 ..modeling_utils 中导入 PreTrainedModel 类型
    from ..modeling_utils import PreTrainedModel

# 从 ..utils 模块导入加速相关函数：is_accelerate_available, is_bitsandbytes_available, is_torch_available, logging
from ..utils import is_accelerate_available, is_bitsandbytes_available, is_torch_available, logging
# 从当前目录下 quantizers_utils 模块中导入 get_module_from_name 函数
from .quantizers_utils import get_module_from_name

# 如果 Torch 可用
if is_torch_available():
    # 导入 Torch 模块
    import torch
    # 从 ..pytorch_utils 模块导入 Conv1D 类

    from ..pytorch_utils import Conv1D

# 获取 logger 对象，用于记录日志
logger = logging.get_logger(__name__)


class Bnb8BitHfQuantizer(HfQuantizer):
    """
    从 bitsandbytes 量化方法中获得 8 位量化：
        加载前：将 transformer 层转换为 Linear8bitLt
        加载中：加载 16 位权重并传递给层对象
        加载后：在首次 .cuda() 调用时将 Linear8bitLt 中的单个权重量化为 8 位
    保存：
        与通常一样，从状态字典中保存权重和 'SCB' 组件
    加载：
        需要定位 'SCB' 组件并传递给 Linear8bitLt 对象
    """

    # 是否保持在 FP32 模块中
    use_keep_in_fp32_modules = True
    # 是否需要参数量化
    requires_parameters_quantization = True
    # 是否需要校准
    requires_calibration = False

    # 必需的包
    required_packages = ["bitsandbytes", "accelerate"]

    def __init__(self, quantization_config, **kwargs):
        # 调用父类的构造函数
        super().__init__(quantization_config, **kwargs)

        # 如果配置中指定了 llm_int8_skip_modules
        if self.quantization_config.llm_int8_skip_modules is not None:
            # 将其赋值给不需要转换的模块列表
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules
    # 验证运行环境是否支持加速库和 bitsandbytes 库
    def validate_environment(self, *args, **kwargs):
        # 检查是否安装了必要的加速库和 bitsandbytes 库
        if not (is_accelerate_available() and is_bitsandbytes_available()):
            raise ImportError(
                "Using `bitsandbytes` 8-bit quantization requires Accelerate: `pip install accelerate` "
                "and the latest version of bitsandbytes: `pip install -i https://pypi.org/simple/ bitsandbytes`"
            )

        # 检查是否从 TensorFlow 或 Flax 权重进行转换，目前不支持这种转换
        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Converting into 4-bit or 8-bit weights from tf/flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

        # 检查是否有可用的 GPU，量化需要 GPU 支持
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")

        # 获取并验证传入的设备映射信息
        device_map = kwargs.get("device_map", None)
        if (
            device_map is not None
            and isinstance(device_map, dict)
            and not self.quantization_config.llm_int8_enable_fp32_cpu_offload
        ):
            # 创建一个不包含指定模块的设备映射副本
            device_map_without_lm_head = {
                key: device_map[key] for key in device_map.keys() if key not in self.modules_to_not_convert
            }
            # 如果设备映射中包含 CPU 或 Disk，则抛出错误
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

        # 检查安装的 bitsandbytes 版本是否支持8位推断和训练
        if version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.37.2"):
            raise ValueError(
                "You have a version of `bitsandbytes` that is not compatible with 8bit inference and training"
                " make sure you have the latest version of `bitsandbytes` installed"
            )

    # 调整最大内存配置以供量化期间使用
    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        # 将最大内存配置按 90% 缩放，以便在量化期间创建的缓冲区有足够的空间
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory
    # 更新 Torch 张量数据类型为指定的 `torch.dtype`
    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        # 如果输入的 torch_dtype 为 None，则强制设置为 float16，这是 `bitsandbytes` 的要求
        logger.info(
            "Overriding torch_dtype=%s with `torch_dtype=torch.float16` due to "
            "requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. "
            "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
            " torch_dtype=torch.float16 to remove this warning.",
            torch_dtype,
        )
        torch_dtype = torch.float16
        return torch_dtype

    # 更新设备映射表，确保 device_map 不为 None
    def update_device_map(self, device_map):
        # 如果 device_map 为 None，则设置为当前 CUDA 设备的空映射
        if device_map is None:
            device_map = {"": torch.cuda.current_device()}
            logger.info(
                "The device_map was not initialized. "
                "Setting device_map to {'':torch.cuda.current_device()}. "
                "If you want to use the model for inference, please set device_map ='auto' "
            )
        return device_map

    # 调整目标数据类型为 torch.int8
    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        # 如果目标数据类型不是 torch.int8，则替换为 torch.int8，用于 8-bit BnB 量化
        if target_dtype != torch.int8:
            logger.info("target_dtype {target_dtype} is replaced by `torch.int8` for 8-bit BnB quantization")
        return torch.int8

    # 检查是否为量化参数
    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ):
        import bitsandbytes as bnb

        # 获取模型和参数名称对应的模块
        module, tensor_name = get_module_from_name(model, param_name)
        # 检查参数是否为 Int8Params 类型（来自 bitsandbytes 库）
        if isinstance(module._parameters.get(tensor_name, None), bnb.nn.Int8Params):
            # 如果预先量化已启用
            if self.pre_quantized:
                # 如果参数名中不包含 `weight` 的替代项 `SCB`，则抛出异常
                if param_name.replace("weight", "SCB") not in state_dict.keys():
                    raise ValueError("Missing quantization component `SCB`")
                # 如果参数值的数据类型不是 torch.int8，则抛出异常
                if param_value.dtype != torch.int8:
                    raise ValueError(
                        f"Incompatible dtype `{param_value.dtype}` when loading 8-bit prequantized weight. Expected `torch.int8`."
                    )
            return True
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
        # 此方法的具体实现在此省略，需要根据功能进一步补充
        """
        组合来自 _load_state_dict_into_meta_model 和 .integrations.bitsandbytes.py::set_module_quantized_tensor_to_device() 的逻辑
        需要从状态字典中获取辅助项，如果找到的话，将其从 unexpected_keys 中移除
        """
        # 导入 bitsandbytes 库作为 bnb
        import bitsandbytes as bnb

        # 根据 param_name 构造 fp16 统计数据的键名
        fp16_statistics_key = param_name.replace("weight", "SCB")
        # 从 state_dict 中获取 fp16 统计数据
        fp16_statistics = state_dict.get(fp16_statistics_key, None)

        # 根据 param_name 获取模型中的模块和张量名
        module, tensor_name = get_module_from_name(model, param_name)
        # 检查张量名是否存在于模块的参数中
        if tensor_name not in module._parameters:
            raise ValueError(f"{module} 没有名为 {tensor_name} 的参数或缓冲区.")

        # 获取旧值
        old_value = getattr(module, tensor_name)

        # 检查模块的参数类型是否为 bnb.nn.Int8Params
        if not isinstance(module._parameters[tensor_name], bnb.nn.Int8Params):
            raise ValueError(f"参数 `{tensor_name}` 应该是 `bnb.nn.Int8Params` 的实例.")

        # 检查旧值的设备是否为 "meta"，并且目标设备不是 "meta" 或 torch.device("meta")，且 param_value 为 None
        if (
            old_value.device == torch.device("meta")
            and target_device not in ["meta", torch.device("meta")]
            and param_value is None
        ):
            raise ValueError(f"{tensor_name} 在 meta 设备上，需要在 {target_device} 上放置一个 `value`.")

        # 将 param_value 转移到 CPU
        new_value = param_value.to("cpu")

        # 如果 self.pre_quantized 为真且 self.is_serializable 为假，则抛出异常
        if self.pre_quantized and not self.is_serializable:
            raise ValueError(
                "检测到 int8 权重，但 bitsandbytes 的版本不兼容 int8 序列化。"
                "请确保下载最新的 `bitsandbytes` 版本。`pip install --upgrade bitsandbytes`."
            )

        # 如果模块的源类是 Conv1D，则在量化之前对权重矩阵进行转置
        if issubclass(module.source_cls, Conv1D):
            if fp16_statistics is None:
                new_value = new_value.T

        # 将旧值的关键字参数赋给 kwargs
        kwargs = old_value.__dict__
        # 使用 bitsandbytes 创建一个新的 Int8Params 实例并将其移至目标设备
        new_value = bnb.nn.Int8Params(new_value, requires_grad=False, **kwargs).to(target_device)

        # 更新模块中的参数值
        module._parameters[tensor_name] = new_value

        # 如果存在 fp16_statistics，则将其设置为新值的 SCB 属性，并从 unexpected_keys 中移除 fp16_statistics_key
        if fp16_statistics is not None:
            setattr(module.weight, "SCB", fp16_statistics.to(target_device))
            if unexpected_keys is not None:
                unexpected_keys.remove(fp16_statistics_key)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        # 将模型标记为已加载 8 位
        model.is_loaded_in_8bit = True
        # 设置模型的 8 位序列化属性为当前对象的 is_serializable 属性
        model.is_8bit_serializable = self.is_serializable
        return model

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
        from ..integrations import get_keys_to_not_convert, replace_with_bnb_linear
        # 从模块导入必要的函数和类，用于不转换的模块获取和替换操作

        load_in_8bit_fp32_cpu_offload = self.quantization_config.llm_int8_enable_fp32_cpu_offload
        # 从量化配置中获取是否启用在 8 位情况下的 FP32 CPU 卸载加载

        # 由于数值稳定性原因，保持某些模块（如 lm_head）在其原始 dtype 下不转换
        if self.quantization_config.llm_int8_skip_modules is None:
            self.modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules
        # 根据配置决定不转换的模块列表，如果未指定，则从模型中获取

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]
        # 如果不转换的模块不是列表类型，则转换为列表

        self.modules_to_not_convert.extend(keep_in_fp32_modules)
        # 将需要保持在 FP32 的模块列表扩展到不转换的模块列表中

        # 将需要卸载到 CPU 或磁盘的键扩展到 `self.modules_to_not_convert`
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]

            if len(keys_on_cpu) > 0 and not load_in_8bit_fp32_cpu_offload:
                raise ValueError(
                    "If you want to offload some keys to `cpu` or `disk`, you need to set "
                    "`llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be "
                    " converted to 8-bit but kept in 32-bit."
                )
            self.modules_to_not_convert.extend(keys_on_cpu)
        # 如果设备映射是字典类型且键数大于1，则根据设备映射中的值为 "disk" 或 "cpu" 的键添加到不转换的模块列表中

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        # 使用指定的替换函数替换模型中的部分模块，传入不转换的模块列表和量化配置

        # TODO: 考虑将 `replace_with_bnb_linear()` 函数从 ..integrations/bitsandbyter.py 文件中移到这里

        model.config.quantization_config = self.quantization_config
        # 设置模型配置的量化配置属性

    @property
    def is_serializable(self):
        _bnb_supports_8bit_serialization = version.parse(importlib.metadata.version("bitsandbytes")) > version.parse(
            "0.37.2"
        )
        # 检查当前安装的 bitsandbytes 版本是否支持 8 位序列化

        if not _bnb_supports_8bit_serialization:
            logger.warning(
                "You are calling `save_pretrained` to a 8-bit converted model, but your `bitsandbytes` version doesn't support it. "
                "If you want to save 8-bit models, make sure to have `bitsandbytes>0.37.2` installed. You will most likely face errors or"
                " unexpected behaviours."
            )
            return False
        # 如果不支持 8 位序列化，则发出警告并返回 False

        return True
        # 如果支持 8 位序列化，则返回 True

    @property
    def is_trainable(self) -> bool:
        return version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.37.0")
    # 检查当前安装的 bitsandbytes 版本是否支持模型训练
```