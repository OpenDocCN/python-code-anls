# `.\diffusers\loaders\single_file_model.py`

```py
# 版权声明，说明该文件的所有权归 HuggingFace 团队所有，保留所有权利
# 
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵循该许可证，否则您不能使用此文件。
# 您可以在以下网址获取许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是按“现状”基础提供的，
# 不提供任何形式的明示或暗示的担保或条件。
# 请参阅许可证以了解有关权限和限制的具体信息。
import importlib  # 导入动态模块加载功能的标准库
import inspect  # 导入用于获取对象信息的标准库
import re  # 导入正则表达式模块
from contextlib import nullcontext  # 从上下文管理器模块导入 nullcontext，用于无操作的上下文管理
from typing import Optional  # 从类型提示模块导入 Optional，表示可选类型

from huggingface_hub.utils import validate_hf_hub_args  # 从 Hugging Face Hub 工具导入验证函数

from ..utils import deprecate, is_accelerate_available, logging  # 从上级模块导入相关的工具和日志模块
from .single_file_utils import (  # 从当前包的单文件工具模块导入多个函数和类
    SingleFileComponentError,  # 导入单文件组件错误类
    convert_animatediff_checkpoint_to_diffusers,  # 导入将 Animatediff 检查点转换为 Diffusers 的函数
    convert_controlnet_checkpoint,  # 导入将 ControlNet 检查点转换的函数
    convert_flux_transformer_checkpoint_to_diffusers,  # 导入将 Flux Transformer 检查点转换为 Diffusers 的函数
    convert_ldm_unet_checkpoint,  # 导入将 LDM UNet 检查点转换的函数
    convert_ldm_vae_checkpoint,  # 导入将 LDM VAE 检查点转换的函数
    convert_sd3_transformer_checkpoint_to_diffusers,  # 导入将 SD3 Transformer 检查点转换为 Diffusers 的函数
    convert_stable_cascade_unet_single_file_to_diffusers,  # 导入将 Stable Cascade UNet 单文件转换为 Diffusers 的函数
    create_controlnet_diffusers_config_from_ldm,  # 导入从 LDM 创建 ControlNet Diffusers 配置的函数
    create_unet_diffusers_config_from_ldm,  # 导入从 LDM 创建 UNet Diffusers 配置的函数
    create_vae_diffusers_config_from_ldm,  # 导入从 LDM 创建 VAE Diffusers 配置的函数
    fetch_diffusers_config,  # 导入获取 Diffusers 配置的函数
    fetch_original_config,  # 导入获取原始配置的函数
    load_single_file_checkpoint,  # 导入加载单文件检查点的函数
)

logger = logging.get_logger(__name__)  # 创建一个日志记录器，使用当前模块的名称

if is_accelerate_available():  # 检查是否可用加速库
    from accelerate import init_empty_weights  # 从 accelerate 导入初始化空权重的功能

    from ..models.modeling_utils import load_model_dict_into_meta  # 从上级模型工具模块导入将模型字典加载到元数据的函数

# 定义一个包含可加载类及其相关配置的字典
SINGLE_FILE_LOADABLE_CLASSES = {
    "StableCascadeUNet": {  # 对应 StableCascadeUNet 类的配置
        "checkpoint_mapping_fn": convert_stable_cascade_unet_single_file_to_diffusers,  # 检查点映射函数
    },
    "UNet2DConditionModel": {  # 对应 UNet2DConditionModel 类的配置
        "checkpoint_mapping_fn": convert_ldm_unet_checkpoint,  # 检查点映射函数
        "config_mapping_fn": create_unet_diffusers_config_from_ldm,  # 配置映射函数
        "default_subfolder": "unet",  # 默认子文件夹名称
        "legacy_kwargs": {  # 旧参数的映射
            "num_in_channels": "in_channels",  # 旧参数映射到新参数的例子
        },
    },
    "AutoencoderKL": {  # 对应 AutoencoderKL 类的配置
        "checkpoint_mapping_fn": convert_ldm_vae_checkpoint,  # 检查点映射函数
        "config_mapping_fn": create_vae_diffusers_config_from_ldm,  # 配置映射函数
        "default_subfolder": "vae",  # 默认子文件夹名称
    },
    "ControlNetModel": {  # 对应 ControlNetModel 类的配置
        "checkpoint_mapping_fn": convert_controlnet_checkpoint,  # 检查点映射函数
        "config_mapping_fn": create_controlnet_diffusers_config_from_ldm,  # 配置映射函数
    },
    "SD3Transformer2DModel": {  # 对应 SD3Transformer2DModel 类的配置
        "checkpoint_mapping_fn": convert_sd3_transformer_checkpoint_to_diffusers,  # 检查点映射函数
        "default_subfolder": "transformer",  # 默认子文件夹名称
    },
    "MotionAdapter": {  # 对应 MotionAdapter 类的配置
        "checkpoint_mapping_fn": convert_animatediff_checkpoint_to_diffusers,  # 检查点映射函数
    },
    "SparseControlNetModel": {  # 对应 SparseControlNetModel 类的配置
        "checkpoint_mapping_fn": convert_animatediff_checkpoint_to_diffusers,  # 检查点映射函数
    },
}
    # 定义一个包含 FluxTransformer2DModel 配置的字典
        "FluxTransformer2DModel": {
            # 指定一个函数，用于将 FluxTransformer 的检查点映射到 diffusers 格式
            "checkpoint_mapping_fn": convert_flux_transformer_checkpoint_to_diffusers,
            # 设置默认子文件夹名称为 "transformer"
            "default_subfolder": "transformer",
        },
# 结束上一个代码块
}


# 定义获取单一文件可加载映射类的函数
def _get_single_file_loadable_mapping_class(cls):
    # 导入当前模块的父级模块
    diffusers_module = importlib.import_module(__name__.split(".")[0])
    # 遍历所有单文件可加载类的字符串名称
    for loadable_class_str in SINGLE_FILE_LOADABLE_CLASSES:
        # 获取对应的可加载类
        loadable_class = getattr(diffusers_module, loadable_class_str)

        # 检查给定的类是否是可加载类的子类
        if issubclass(cls, loadable_class):
            # 如果是，则返回该可加载类的字符串名称
            return loadable_class_str

    # 如果没有找到合适的可加载类，返回 None
    return None


# 定义获取映射函数关键字参数的函数
def _get_mapping_function_kwargs(mapping_fn, **kwargs):
    # 获取映射函数的参数签名
    parameters = inspect.signature(mapping_fn).parameters

    # 创建一个字典以存储匹配的关键字参数
    mapping_kwargs = {}
    # 遍历所有参数
    for parameter in parameters:
        # 如果参数在提供的关键字参数中，则存入字典
        if parameter in kwargs:
            mapping_kwargs[parameter] = kwargs[parameter]

    # 返回匹配的关键字参数字典
    return mapping_kwargs


# 定义一个混合类，用于从原始模型加载预训练权重
class FromOriginalModelMixin:
    """
    加载保存为 `.ckpt` 或 `.safetensors` 格式的预训练权重到模型中。
    """

    # 声明该方法为类方法
    @classmethod
    # 应用参数验证装饰器
    @validate_hf_hub_args
```