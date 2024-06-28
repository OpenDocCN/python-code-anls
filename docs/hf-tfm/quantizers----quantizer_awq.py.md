# `.\quantizers\quantizer_awq.py`

```
# 版权声明及许可信息，指明HuggingFace Inc.团队拥有版权
#
# 根据Apache许可证2.0版（"许可证"）授权，除非符合许可证要求，
# 否则不得使用本文件。您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件按"原样"分发，不提供任何明示或
# 含示的担保或条件。详细信息请参阅许可证。
import importlib.metadata
from typing import TYPE_CHECKING

from packaging import version

# 导入基础的HfQuantizer类
from .base import HfQuantizer

# 如果类型检查开启，则导入PreTrainedModel类
if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

# 导入一些工具和依赖的模块
from ..utils import is_accelerate_available, is_auto_awq_available, is_torch_available, logging
from ..utils.quantization_config import AWQLinearVersion

# 如果torch可用，则导入torch库
if is_torch_available():
    import torch

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# AwqQuantizer类继承自HfQuantizer类，提供Activation-aware Weight Quantization(AWQ)的4位量化支持
class AwqQuantizer(HfQuantizer):
    """
    4-bit quantization for Activation-aware Weight Quantization(AWQ) (https://arxiv.org/abs/2306.00978)
    """

    # AWQ需要数据校准 - 我们只支持推断（inference）
    requires_calibration = True

    # 必需的包名称列表，包括"awq"和"accelerate"
    required_packages = ["awq", "accelerate"]

    # 初始化方法，接受quantization_config和其他关键字参数
    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    # 验证运行环境的方法，检查GPU是否可用以及必需的库是否已安装
    def validate_environment(self, device_map, **kwargs):
        # 如果没有CUDA设备可用，则抛出运行时错误
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required to run AWQ quantized model.")

        # 如果未安装auto-awq库，则抛出导入错误
        if not is_auto_awq_available():
            raise ImportError("Loading an AWQ quantized model requires auto-awq library (`pip install autoawq`)")

        # 如果未安装accelerate库，则抛出导入错误
        if not is_accelerate_available():
            raise ImportError("Loading an AWQ quantized model requires accelerate (`pip install accelerate`)")

        # 如果device_map为None，则发出警告，建议在GPU设备上运行模型
        if device_map is None:
            logger.warning_once(
                "You have loaded an AWQ model on CPU and have a CUDA device available, make sure to set "
                "your model on a GPU device in order to run your model."
            )
        # 如果device_map不为None，则检查是否包含CPU或磁盘设备，如果是则抛出数值错误
        elif device_map is not None:
            if isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
                raise ValueError(
                    "You are attempting to load an AWQ model with a device_map that contains a CPU or disk device."
                    " This is not supported. Please remove the CPU or disk device from the device_map."
                )

    # 更新torch数据类型的方法，如果未提供torch_dtype，则使用torch.float16
    def update_torch_dtype(self, torch_dtype):
        if torch_dtype is None:
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16:
            logger.warning("We suggest you to set `torch_dtype=torch.float16` for better efficiency with AWQ.")
        return torch_dtype
    # 在加载权重前处理模型的方法，用于量化感知训练模型
    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        # 导入必要的集成模块：获取不需转换的模块键和替换 AWQ 线性模块
        from ..integrations import get_keys_to_not_convert, replace_with_awq_linear
        
        # 获取不需要转换的模块列表
        self.modules_to_not_convert = get_keys_to_not_convert(model)
        
        # 如果配置中有指定不需要转换的模块，则扩展已有的列表
        if self.quantization_config.modules_to_not_convert is not None:
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)
        
        # 替换模型中的线性层为 AWQ 线性层，并检查是否有替换发生
        model, has_been_replaced = replace_with_awq_linear(
            model, quantization_config=self.quantization_config, modules_to_not_convert=self.modules_to_not_convert
        )
        
        # 如果没有进行替换，则发出警告信息
        if not has_been_replaced:
            logger.warning(
                "You are loading an AWQ model but no linear modules were found in your model."
                " Please double check your model architecture, or submit an issue on github if you think this is a bug."
            )
    
    # 在加载权重后处理模型的方法
    def _process_model_after_weight_loading(self, model):
        # 如果配置要求进行模块融合
        if self.quantization_config.do_fuse:
            # 导入模块：融合 AWQ 模块
            from ..integrations import fuse_awq_modules
            
            # 融合模型中的 AWQ 模块
            model = fuse_awq_modules(model, self.quantization_config)
            # 设置 AWQ 被融合的标志为真，考虑将此标志存储在 model.config 中
            model._awq_is_fused = True  # TODO: consider storing this flag in model.config instead
        
        # 如果使用的 AWQ 版本为 EXLLAMA
        if self.quantization_config.version == AWQLinearVersion.EXLLAMA:
            # 导入模块：初始化 AWQ EXLLAMA 后端模块
            from ..integrations import post_init_awq_exllama_modules
            
            # 对模型进行 AWQ EXLLAMA 后端模块的初始化
            model = post_init_awq_exllama_modules(model, self.quantization_config.exllama_config)
    
    # 判断模型是否可序列化的属性
    @property
    def is_serializable(self):
        # 如果配置要求进行模块融合，则不可保存
        if self.quantization_config.do_fuse:
            logger.warning("You cannot save an AWQ model that uses fused modules!")
            return False
        
        # 如果使用的 AWQ 版本为 EXLLAMA，则不可保存
        if self.quantization_config.version == AWQLinearVersion.EXLLAMA:
            logger.warning("You cannot save an AWQ model that uses Exllama backend!")
            return False
        
        # 否则可保存
        return True
    
    # 判断模型是否可训练的属性
    @property
    def is_trainable(self):
        # 定义 PEFT 微调所需的最小 AWQ 版本
        MIN_AWQ_VERSION_FOR_PEFT = "0.2.0"
        
        # 检查当前 autoawq 模块的版本是否支持 PEFT 微调
        return version.parse(importlib.metadata.version("autoawq")) >= version.parse(MIN_AWQ_VERSION_FOR_PEFT)
```