# `.\diffusers\loaders\ip_adapter.py`

```py
# 版权声明，2024年HuggingFace团队保留所有权利
# 
# 根据Apache许可证第2.0版（“许可证”）授权；
# 除非遵守许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，软件在“原样”基础上分发，
# 不附带任何形式的明示或暗示的担保或条件。
# 请参阅许可证以了解有关权限和
# 限制的具体条款。

# 从pathlib模块导入Path类，用于路径操作
from pathlib import Path
# 从typing模块导入各种类型提示
from typing import Dict, List, Optional, Union

# 导入torch库
import torch
# 导入torch的功能模块，用于神经网络操作
import torch.nn.functional as F
# 从huggingface_hub.utils导入验证函数，用于验证HF Hub参数
from huggingface_hub.utils import validate_hf_hub_args
# 从safetensors导入安全打开函数
from safetensors import safe_open

# 从本地模型工具导入低CPU内存使用默认值和加载状态字典的函数
from ..models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_state_dict
# 从本地工具导入多个实用函数和变量
from ..utils import (
    USE_PEFT_BACKEND,  # 是否使用PEFT后端的标志
    _get_model_file,  # 获取模型文件的函数
    is_accelerate_available,  # 检查加速库是否可用
    is_torch_version,  # 检查Torch版本的函数
    is_transformers_available,  # 检查Transformers库是否可用
    logging,  # 导入日志模块
)
# 从unet_loader_utils模块导入可能扩展LoRA缩放的函数
from .unet_loader_utils import _maybe_expand_lora_scales

# 如果Transformers库可用，则导入相关的类和函数
if is_transformers_available():
    # 导入CLIP图像处理器和带投影的CLIP视觉模型
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
    )

    # 导入注意力处理器类
    from ..models.attention_processor import (
        AttnProcessor,  # 注意力处理器
        AttnProcessor2_0,  # 版本2.0的注意力处理器
        IPAdapterAttnProcessor,  # IP适配器注意力处理器
        IPAdapterAttnProcessor2_0,  # 版本2.0的IP适配器注意力处理器
    )

# 获取日志记录器实例，使用当前模块的名称
logger = logging.get_logger(__name__)

# 定义一个处理IP适配器的Mixin类
class IPAdapterMixin:
    """处理IP适配器的Mixin类。"""

    # 使用装饰器验证HF Hub参数
    @validate_hf_hub_args
    def load_ip_adapter(
        # 定义加载IP适配器所需的参数，包括模型名称、子文件夹、权重名称等
        self,
        pretrained_model_name_or_path_or_dict: Union[str, List[str], Dict[str, torch.Tensor]],
        subfolder: Union[str, List[str]],
        weight_name: Union[str, List[str]],
        # 可选参数，默认值为“image_encoder”
        image_encoder_folder: Optional[str] = "image_encoder",
        **kwargs,  # 其他关键字参数
    # 定义方法以设置 IP-Adapter 的缩放比例，输入参数 scale 可为单个配置或多个配置的列表
    def set_ip_adapter_scale(self, scale):
        # 文档字符串，提供使用示例和配置说明
        """
        Set IP-Adapter scales per-transformer block. Input `scale` could be a single config or a list of configs for
        granular control over each IP-Adapter behavior. A config can be a float or a dictionary.
    
        Example:
    
        ```py
        # To use original IP-Adapter
        scale = 1.0
        pipeline.set_ip_adapter_scale(scale)
    
        # To use style block only
        scale = {
            "up": {"block_0": [0.0, 1.0, 0.0]},
        }
        pipeline.set_ip_adapter_scale(scale)
    
        # To use style+layout blocks
        scale = {
            "down": {"block_2": [0.0, 1.0]},
            "up": {"block_0": [0.0, 1.0, 0.0]},
        }
        pipeline.set_ip_adapter_scale(scale)
    
        # To use style and layout from 2 reference images
        scales = [{"down": {"block_2": [0.0, 1.0]}}, {"up": {"block_0": [0.0, 1.0, 0.0]}}]
        pipeline.set_ip_adapter_scale(scales)
        ```py
        """
        # 根据名称获取 UNet 对象，如果没有则使用已有的 unet 属性
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        # 如果 scale 不是列表，将其转换为列表
        if not isinstance(scale, list):
            scale = [scale]
        # 调用辅助函数以展开缩放配置，默认缩放为 0.0
        scale_configs = _maybe_expand_lora_scales(unet, scale, default_scale=0.0)
    
        # 遍历 UNet 的注意力处理器字典
        for attn_name, attn_processor in unet.attn_processors.items():
            # 检查处理器是否为 IPAdapter 类型
            if isinstance(attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)):
                # 验证缩放配置数量与处理器的数量匹配
                if len(scale_configs) != len(attn_processor.scale):
                    raise ValueError(
                        f"Cannot assign {len(scale_configs)} scale_configs to "
                        f"{len(attn_processor.scale)} IP-Adapter."
                    )
                # 如果只有一个缩放配置，复制到每个处理器
                elif len(scale_configs) == 1:
                    scale_configs = scale_configs * len(attn_processor.scale)
                # 遍历每个缩放配置
                for i, scale_config in enumerate(scale_configs):
                    # 如果配置是字典，则根据名称匹配进行设置
                    if isinstance(scale_config, dict):
                        for k, s in scale_config.items():
                            if attn_name.startswith(k):
                                attn_processor.scale[i] = s
                    # 否则直接将缩放配置赋值
                    else:
                        attn_processor.scale[i] = scale_config
    # 定义一个方法来卸载 IP 适配器的权重
    def unload_ip_adapter(self):
        """
        卸载 IP 适配器的权重

        示例：

        ```python
        >>> # 假设 `pipeline` 已经加载了 IP 适配器的权重。
        >>> pipeline.unload_ip_adapter()
        >>> ...
        ```py
        """
        # 移除 CLIP 图像编码器
        if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is not None:
            # 将图像编码器设为 None
            self.image_encoder = None
            # 更新配置，移除图像编码器的相关信息
            self.register_to_config(image_encoder=[None, None])

        # 仅当 safety_checker 为 None 时移除特征提取器，因为 safety_checker 后续会使用 feature_extractor
        if not hasattr(self, "safety_checker"):
            if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is not None:
                # 将特征提取器设为 None
                self.feature_extractor = None
                # 更新配置，移除特征提取器的相关信息
                self.register_to_config(feature_extractor=[None, None])

        # 移除隐藏编码器
        self.unet.encoder_hid_proj = None
        # 将编码器的隐藏维度类型设为 None
        self.unet.config.encoder_hid_dim_type = None

        # Kolors: 使用 `text_encoder_hid_proj` 恢复 `encoder_hid_proj`
        if hasattr(self.unet, "text_encoder_hid_proj") and self.unet.text_encoder_hid_proj is not None:
            # 将 encoder_hid_proj 设置为 text_encoder_hid_proj
            self.unet.encoder_hid_proj = self.unet.text_encoder_hid_proj
            # 将 text_encoder_hid_proj 设为 None
            self.unet.text_encoder_hid_proj = None
            # 更新编码器的隐藏维度类型为 "text_proj"
            self.unet.config.encoder_hid_dim_type = "text_proj"

        # 恢复原始 Unet 注意力处理器层
        attn_procs = {}
        # 遍历 Unet 的注意力处理器
        for name, value in self.unet.attn_processors.items():
            # 根据 F 是否具有 scaled_dot_product_attention 选择注意力处理器的类
            attn_processor_class = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else AttnProcessor()
            )
            # 将注意力处理器添加到字典中，若是 IPAdapter 的类则使用新的类，否则使用原类
            attn_procs[name] = (
                attn_processor_class
                if isinstance(value, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0))
                else value.__class__()
            )
        # 设置 Unet 的注意力处理器为新生成的处理器字典
        self.unet.set_attn_processor(attn_procs)
```