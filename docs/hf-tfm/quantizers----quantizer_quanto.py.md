# `.\quantizers\quantizer_quanto.py`

```py
# 导入模块 importlib，用于动态导入模块
import importlib
# 导入类型检查标记 TYPE_CHECKING、Any、Dict、List、Optional、Union
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# 导入版本比较模块 version，来自 packaging 包
from packaging import version

# 从当前包中导入 base 模块中的 HfQuantizer 类
from .base import HfQuantizer
# 从 quantizers_utils 模块中导入 get_module_from_name 函数
from .quantizers_utils import get_module_from_name

# 如果是类型检查状态，从 modeling_utils 模块中导入 PreTrainedModel 类
if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

# 从 utils 模块中导入 is_accelerate_available、is_quanto_available、is_torch_available、logging 函数和类
from ..utils import is_accelerate_available, is_quanto_available, is_torch_available, logging
# 从 utils.quantization_config 模块中导入 QuantoConfig 类
from ..utils.quantization_config import QuantoConfig

# 如果 torch 可用，导入 torch 模块
if is_torch_available():
    import torch

# 从 logging 模块中获取 logger 对象
logger = logging.get_logger(__name__)

# 定义 QuantoHfQuantizer 类，继承自 HfQuantizer 类
class QuantoHfQuantizer(HfQuantizer):
    """
    Quantizer for the quanto library
    """

    # 定义 required_packages 列表，指明需要的依赖包
    required_packages = ["quanto", "accelerate"]
    # 指明是否需要参数量化
    requires_parameters_quantization = True
    # 指明是否需要校准
    requires_calibration = False

    # 初始化方法，接收 quantization_config 参数和其他关键字参数
    def __init__(self, quantization_config: QuantoConfig, **kwargs):
        # 调用父类 HfQuantizer 的初始化方法
        super().__init__(quantization_config, **kwargs)
        # 调用 post_init 方法
        self.post_init()

    # 定义 post_init 方法，用于安全检查
    def post_init(self):
        # 如果 quantization_config.activations 不为空且未预量化
        if self.quantization_config.activations is not None and not self.pre_quantized:
            # 抛出值错误异常，提示不支持对激活进行量化
            raise ValueError(
                "We don't support quantizing the activations with transformers library."
                "Use quanto library for more complex use cases such as activations quantization, calibration and quantization aware training."
            )

    # 定义 validate_environment 方法，用于验证环境是否支持 quanto 库和 accelerate 库
    def validate_environment(self, *args, **kwargs):
        # 如果 quanto 库不可用，抛出导入错误异常
        if not is_quanto_available():
            raise ImportError("Loading a quanto quantized model requires quanto library (`pip install quanto`)")
        # 如果 accelerate 库不可用，抛出导入错误异常
        if not is_accelerate_available():
            raise ImportError("Loading a quanto quantized model requires accelerate library (`pip install quanto`)")

    # 定义 update_device_map 方法，用于更新设备映射
    def update_device_map(self, device_map):
        # 如果 device_map 为 None，则初始化为 {'': 'cpu'}
        if device_map is None:
            device_map = {"": "cpu"}
            # 记录日志信息，提示设备映射未初始化，将其设置为 {'': 'cpu'}
            logger.info(
                "The device_map was not initialized. "
                "Setting device_map to {'':'cpu'}. "
                "If you want to use the model for inference, please set device_map ='auto'"
            )
        # 返回更新后的 device_map
        return device_map

    # 定义 update_torch_dtype 方法，用于更新 torch 数据类型
    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        # 如果 torch_dtype 为 None
        if torch_dtype is None:
            # 记录日志信息，提示在 from_pretrained 中未指定 torch_dtype，默认设置为 torch.float32
            logger.info("You did not specify `torch_dtype` in `from_pretrained`. Setting it to `torch.float32`.")
            # 将 torch_dtype 设置为 torch.float32
            torch_dtype = torch.float32
        # 返回更新后的 torch_dtype
        return torch_dtype
    # 更新模型中缺失的键列表，返回更新后未缺失的键列表
    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        import quanto  # 导入quanto模块

        not_missing_keys = []  # 初始化未缺失的键列表
        # 遍历模型中的命名模块
        for name, module in model.named_modules():
            # 如果模块是quanto.QModuleMixin的实例
            if isinstance(module, quanto.QModuleMixin):
                # 遍历缺失的键列表
                for missing in missing_keys:
                    # 如果模块名称在缺失键中或者在以prefix开头的缺失键中
                    if (
                        (name in missing or name in f"{prefix}.{missing}")
                        and not missing.endswith(".weight")  # 排除以.weight结尾的键
                        and not missing.endswith(".bias")    # 排除以.bias结尾的键
                    ):
                        not_missing_keys.append(missing)  # 将该键添加到未缺失的键列表中
        # 返回更新后的未缺失的键列表
        return [k for k in missing_keys if k not in not_missing_keys]

    # 检查是否需要量化参数
    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        """
        Check if a parameter needs to be quantized.
        """
        import quanto  # 导入quanto模块

        device_map = kwargs.get("device_map", None)  # 获取device_map参数
        param_device = kwargs.get("param_device", None)  # 获取param_device参数
        # 如果模块将要被离线到cpu上，则不进行模型量化
        if device_map is not None and param_device is not None:
            device_map_values = set(device_map.values())  # 获取device_map的所有值集合
            if param_device == "cpu" and len(device_map_values) > 1:
                if not (device_map_values == {"cpu"} or device_map_values == {"cpu", "disk"}):
                    return False  # 如果条件满足，返回False，不进行模型量化

        module, tensor_name = get_module_from_name(model, param_name)  # 获取参数所在的模块和张量名
        # 只量化权重，不量化偏置
        if isinstance(module, quanto.QModuleMixin) and "weight" in tensor_name:
            # 如果权重已经量化，不需要使用`create_quantized_param`重新创建
            return not module.frozen
        else:
            return False  # 其他情况下不进行模型量化

    # 调整最大内存限制
    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        # 将最大内存限制减少10%
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory  # 返回调整后的最大内存限制字典

    # 创建量化参数
    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        *args,
        **kwargs,
    ):
        """
        Create the quantized parameter by calling .freeze() after setting it to the module.
        """
        from accelerate.utils import set_module_tensor_to_device  # 从accelerate.utils模块导入set_module_tensor_to_device函数

        set_module_tensor_to_device(model, param_name, target_device, param_value)  # 将参数设置到模块并移动到目标设备
        module, _ = get_module_from_name(model, param_name)  # 获取参数所在的模块
        module.freeze()  # 冻结模块，使其无法再修改
        module.weight.requires_grad = False  # 设置权重张量不需要梯度计算
    # 调整目标数据类型以匹配加速库版本的需求
    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        # 检查当前加速库版本是否大于 0.27.0
        if version.parse(importlib.metadata.version("accelerate")) > version.parse("0.27.0"):
            # 导入加速库中的自定义数据类型
            from accelerate.utils import CustomDtype

            # 定义数据类型映射关系
            mapping = {
                "int8": torch.int8,
                "float8": CustomDtype.FP8,
                "int4": CustomDtype.INT4,
                "int2": CustomDtype.INT2,
            }
            # 根据量化配置中的权重类型选择目标数据类型
            target_dtype = mapping[self.quantization_config.weights]
            return target_dtype
        else:
            # 抛出数值错误，提示升级加速库版本
            raise ValueError(
                "You are using `device_map='auto'` on a quanto quantized model. To automatically compute"
                " the appropriate device map, you should upgrade your `accelerate` library,"
                "`pip install --upgrade accelerate` or install it from source."
            )

    # 在加载权重前处理模型
    def _process_model_before_weight_loading(
        self, model: "PreTrainedModel", keep_in_fp32_modules: List[str] = [], **kwargs
    ):
        # 导入必要的函数以及类
        from ..integrations import get_keys_to_not_convert, replace_with_quanto_layers

        # 如果未设置不转换的模块列表，则根据模型获取不转换模块的键
        if self.quantization_config.modules_to_not_convert is None:
            self.modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            self.modules_to_not_convert = self.quantization_config.modules_to_not_convert

        # 确保不转换的模块列表为一个列表类型
        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        # 将需要保持在 FP32 精度的模块添加到不转换的模块列表中
        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        # 使用自定义函数替换量化层并更新模型
        model, _ = replace_with_quanto_layers(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        # 更新模型的量化配置
        model.config.quantization_config = self.quantization_config

    # 在加载完权重后处理模型
    def _process_model_after_weight_loading(self, model):
        return model

    # 返回模型是否可训练的属性
    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return False

    # 返回模型是否可序列化的属性
    @property
    def is_serializable(self):
        return False
```