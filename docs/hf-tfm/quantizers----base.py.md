# `.\quantizers\base.py`

```
# 引入 ABC 类和类型检查相关的模块
from abc import ABC, abstractmethod
# 引入类型检查模块，用于检查是否支持特定类型的操作
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# 从相对路径的模块中导入 is_torch_available 函数
from ..utils import is_torch_available
# 从相对路径的模块中导入 QuantizationConfigMixin 类
from ..utils.quantization_config import QuantizationConfigMixin

# 如果在类型检查模式下，导入 PreTrainedModel 类
if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

# 如果 Torch 可用，则导入 Torch 模块
if is_torch_available():
    import torch

# 定义 HfQuantizer 抽象类
class HfQuantizer(ABC):
    """
    HuggingFace 量化器的抽象类。目前支持对 HF transformers 模型进行推断和/或量化。
    这个类仅用于 transformers.PreTrainedModel.from_pretrained 方法的范围内，无法在该方法范围外轻松使用。

    Attributes:
        quantization_config (`transformers.utils.quantization_config.QuantizationConfigMixin`):
            定义要量化的模型的量化参数的配置。
        modules_to_not_convert (`List[str]`, *optional*):
            在量化模型时不希望转换的模块名称列表。
        required_packages (`List[str]`, *optional*):
            使用量化器之前需要安装的必需 pip 包列表。
        requires_calibration (`bool`):
            使用量化方法是否需要在使用模型之前进行校准。
        requires_parameters_quantization (`bool`):
            使用量化方法是否需要创建新的参数。例如，对于 bitsandbytes，需要创建新的 xxxParameter 来正确量化模型。
    """

    # 标识量化方法是否需要校准
    requires_calibration = False
    # 用于存储需要安装的必需 pip 包的列表
    required_packages = None
    # 标识量化方法是否需要对参数进行量化
    requires_parameters_quantization = False
    # 初始化函数，接受量化配置和其他关键字参数
    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        # 将量化配置保存到实例变量中
        self.quantization_config = quantization_config

        # 处理额外的关键字参数
        self.modules_to_not_convert = kwargs.pop("modules_to_not_convert", [])
        self.pre_quantized = kwargs.pop("pre_quantized", True)

        # 如果未预量化但需要校准，引发值错误异常
        if not self.pre_quantized and self.requires_calibration:
            raise ValueError(
                f"The quantization method {quantization_config.quant_method} does require the model to be pre-quantized."
                f" You explicitly passed `pre_quantized=False` meaning your model weights are not quantized. Make sure to "
                f"pass `pre_quantized=True` while knowing what you are doing."
            )

    # 更新 Torch 数据类型的方法，通常由子类重写以确保行为一致性
    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        """
        Some quantization methods require to explicitly set the dtype of the model to a
        target dtype. You need to override this method in case you want to make sure that behavior is
        preserved

        Args:
            torch_dtype (`torch.dtype`):
                The input dtype that is passed in `from_pretrained`
        """
        return torch_dtype

    # 更新设备映射的方法，通常由子类重写以允许传递新的设备映射
    def update_device_map(self, device_map: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Override this method if you want to pass a override the existing device map with a new
        one. E.g. for bitsandbytes, since `accelerate` is a hard requirement, if no device_map is
        passed, the device_map is set to `"auto"``

        Args:
            device_map (`Union[dict, str]`, *optional*):
                The device_map that is passed through the `from_pretrained` method.
        """
        return device_map

    # 调整目标 Torch 数据类型的方法，通常由子类重写以适应特定的量化需求
    def adjust_target_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        """
        Override this method if you want to adjust the `target_dtype` variable used in `from_pretrained`
        to compute the device_map in case the device_map is a `str`. E.g. for bitsandbytes we force-set `target_dtype`
        to `torch.int8` and for 4-bit we pass a custom enum `accelerate.CustomDtype.int4`.

        Args:
            torch_dtype (`torch.dtype`, *optional*):
                The torch_dtype that is used to compute the device_map.
        """
        return torch_dtype

    # 更新缺失键列表的方法，通常由子类重写以适应特定的模型加载需求
    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        """
        Override this method if you want to adjust the `missing_keys`.

        Args:
            missing_keys (`List[str]`, *optional*):
                The list of missing keys in the checkpoint compared to the state dict of the model
        """
        return missing_keys
    def get_special_dtypes_update(self, model, torch_dtype: "torch.dtype") -> Dict[str, "torch.dtype"]:
        """
        返回未量化模块的数据类型字典 - 用于在传递字符串作为 device_map 时计算 device_map。
        该方法将使用 `_process_model_before_weight_loading` 中修改的 `modules_to_not_convert`。
        
        Args:
            model (`~transformers.PreTrainedModel`):
                要量化的模型
            torch_dtype (`torch.dtype`):
                在 `from_pretrained` 方法中传递的数据类型
        """
        return {
            name: torch_dtype
            for name, _ in model.named_parameters()
            if any(m in name for m in self.modules_to_not_convert)
        }

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        """根据需要为量化调整 max_memory 参数，用于 infer_auto_device_map()。"""
        return max_memory

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        """
        检查加载的 state_dict 组件是否是量化参数的一部分，同时进行一些验证；
        只有在 requires_parameters_quantization == True 时才会定义，用于需要为量化方法创建新参数的情况。
        """
        return False

    def create_quantized_param(self, *args, **kwargs) -> "torch.nn.Parameter":
        """
        从 state_dict 中获取必要的组件并创建量化参数；
        只有在 requires_parameters_quantization == True 时适用。
        如果不支持 requires_parameters_quantization，则会引发 AttributeError。
        """
        if not self.requires_parameters_quantization:
            raise AttributeError(
                f"`.create_quantized_param()` 方法不受量化器类 {self.__class__.__name__} 支持。"
            )

    def validate_environment(self, *args, **kwargs):
        """
        该方法用于潜在地检查在 `from_pretrained` 中传递的参数是否存在冲突。
        对于所有未来与 transformers 集成的量化器，都需要定义它。
        如果不需要显式检查，则简单返回即可。
        """
        return
    # 定义一个方法，用于预处理模型，在加载权重之前设置模型属性或转换模型。
    def preprocess_model(self, model: "PreTrainedModel", **kwargs):
        """
        设置模型属性和/或在加载权重之前对模型进行转换。此时模型应在元设备上初始化，
        因此可以自由地操纵模型的骨架以替换模块。确保覆盖抽象方法 `_process_model_before_weight_loading`。

        Args:
            model (`~transformers.PreTrainedModel`):
                要量化的模型
            kwargs (`dict`, *optional*):
                被传递到 `_process_model_before_weight_loading` 的关键字参数。
        """
        # 设置模型的量化标志为True
        model.is_quantized = True
        # 设置模型的量化方法为配置文件中指定的量化方法
        model.quantization_method = self.quantization_config.quant_method
        # 调用 `_process_model_before_weight_loading` 方法进行进一步处理
        return self._process_model_before_weight_loading(model, **kwargs)

    # 定义一个方法，用于在加载权重后对模型进行后处理。
    def postprocess_model(self, model: "PreTrainedModel", **kwargs):
        """
        在加载权重后对模型进行后处理。确保覆盖抽象方法 `_process_model_after_weight_loading`。

        Args:
            model (`~transformers.PreTrainedModel`):
                要量化的模型
            kwargs (`dict`, *optional*):
                被传递到 `_process_model_after_weight_loading` 的关键字参数。
        """
        # 调用 `_process_model_after_weight_loading` 方法进行后处理
        return self._process_model_after_weight_loading(model, **kwargs)

    # 抽象方法，用于在加载权重之前处理模型，需要在子类中实现
    @abstractmethod
    def _process_model_before_weight_loading(self, model, **kwargs):
        ...

    # 抽象方法，用于在加载权重之后处理模型，需要在子类中实现
    @abstractmethod
    def _process_model_after_weight_loading(self, model, **kwargs):
        ...

    # 抽象属性，用于指示对象是否可序列化，需要在子类中实现
    @property
    @abstractmethod
    def is_serializable(self):
        ...

    # 抽象属性，用于指示对象是否可训练，需要在子类中实现
    @property
    @abstractmethod
    def is_trainable(self):
        ...
```