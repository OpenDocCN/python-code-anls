# `.\diffusers\models\transformers\pixart_transformer_2d.py`

```
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件
# 按“原样”提供，没有任何形式的明示或暗示的担保或条件。
# 请参阅许可证以了解有关权限和
# 限制的具体条款。
from typing import Any, Dict, Optional, Union  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 导入神经网络模块

from ...configuration_utils import ConfigMixin, register_to_config  # 导入配置相关的混合类和注册函数
from ...utils import is_torch_version, logging  # 导入工具函数：检查 PyTorch 版本和日志记录
from ..attention import BasicTransformerBlock  # 导入基础 Transformer 块
from ..attention_processor import Attention, AttentionProcessor, FusedAttnProcessor2_0  # 导入注意力相关的处理器
from ..embeddings import PatchEmbed, PixArtAlphaTextProjection  # 导入嵌入相关的模块
from ..modeling_outputs import Transformer2DModelOutput  # 导入模型输出相关的类
from ..modeling_utils import ModelMixin  # 导入模型混合类
from ..normalization import AdaLayerNormSingle  # 导入自适应层归一化类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器；pylint 禁用命名检查

class PixArtTransformer2DModel(ModelMixin, ConfigMixin):  # 定义 PixArt 2D Transformer 模型类，继承自 ModelMixin 和 ConfigMixin
    r"""  # 文档字符串：描述模型及其来源
    A 2D Transformer model as introduced in PixArt family of models (https://arxiv.org/abs/2310.00426,
    https://arxiv.org/abs/2403.04692).
    """

    _supports_gradient_checkpointing = True  # 设置支持梯度检查点
    _no_split_modules = ["BasicTransformerBlock", "PatchEmbed"]  # 指定不进行分割的模块

    @register_to_config  # 使用装饰器将初始化函数注册到配置中
    def __init__(  # 定义初始化函数
        self,
        num_attention_heads: int = 16,  # 注意力头的数量，默认为 16
        attention_head_dim: int = 72,  # 每个注意力头的维度，默认为 72
        in_channels: int = 4,  # 输入通道数，默认为 4
        out_channels: Optional[int] = 8,  # 输出通道数，默认为 8，可选
        num_layers: int = 28,  # 层数，默认为 28
        dropout: float = 0.0,  # dropout 比例，默认为 0.0
        norm_num_groups: int = 32,  # 归一化的组数，默认为 32
        cross_attention_dim: Optional[int] = 1152,  # 交叉注意力的维度，默认为 1152，可选
        attention_bias: bool = True,  # 是否使用注意力偏置，默认为 True
        sample_size: int = 128,  # 样本尺寸，默认为 128
        patch_size: int = 2,  # 每个补丁的尺寸，默认为 2
        activation_fn: str = "gelu-approximate",  # 激活函数类型，默认为近似 GELU
        num_embeds_ada_norm: Optional[int] = 1000,  # 自适应归一化的嵌入数量，默认为 1000，可选
        upcast_attention: bool = False,  # 是否提高注意力精度，默认为 False
        norm_type: str = "ada_norm_single",  # 归一化类型，默认为单一自适应归一化
        norm_elementwise_affine: bool = False,  # 是否使用逐元素仿射变换，默认为 False
        norm_eps: float = 1e-6,  # 归一化的 epsilon 值，默认为 1e-6
        interpolation_scale: Optional[int] = None,  # 插值尺度，可选
        use_additional_conditions: Optional[bool] = None,  # 是否使用额外条件，可选
        caption_channels: Optional[int] = None,  # 说明通道数，可选
        attention_type: Optional[str] = "default",  # 注意力类型，默认为默认类型
    ):
        # 初始化函数参数设置
        ...

    def _set_gradient_checkpointing(self, module, value=False):  # 定义设置梯度检查点的方法
        if hasattr(module, "gradient_checkpointing"):  # 检查模块是否具有梯度检查点属性
            module.gradient_checkpointing = value  # 设置梯度检查点的值

    @property  # 定义一个属性
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors 复制的属性
    # 定义一个方法，返回模型中所有注意力处理器的字典，键为权重名称
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # 创建一个空字典，用于存储注意力处理器
        processors = {}

        # 定义一个递归函数，用于添加处理器到字典中
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 如果模块具有获取处理器的方法，则将其添加到字典中
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            # 遍历子模块，递归调用该函数以添加处理器
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            # 返回更新后的处理器字典
            return processors

        # 遍历当前模块的所有子模块，调用递归函数以填充处理器字典
        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        # 返回所有注意力处理器的字典
        return processors

    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor 复制
    # 定义一个方法，用于设置计算注意力的处理器
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        # 获取当前注意力处理器的数量
        count = len(self.attn_processors.keys())

        # 检查传入的处理器字典的长度是否与注意力层的数量匹配
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        # 定义一个递归函数，用于设置处理器
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 如果模块具有设置处理器的方法，则进行设置
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)  # 设置单一处理器
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))  # 从字典中移除并设置处理器

            # 遍历子模块，递归调用以设置处理器
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        # 遍历当前模块的所有子模块，调用递归函数以设置处理器
        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections 复制
    # 定义融合 QKV 投影的函数
    def fuse_qkv_projections(self):
        # 启用融合的 QKV 投影，对自注意力模块进行融合查询、键、值矩阵
        # 对交叉注意力模块则仅融合键和值投影矩阵
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.
    
        <Tip warning={true}>
    
        This API is 🧪 experimental.
    
        </Tip>
        """
        # 初始化原始注意力处理器为 None
        self.original_attn_processors = None
    
        # 遍历所有注意力处理器
        for _, attn_processor in self.attn_processors.items():
            # 检查处理器类名中是否包含 "Added"
            if "Added" in str(attn_processor.__class__.__name__):
                # 如果存在，则抛出错误，说明不支持此融合操作
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")
    
        # 保存当前的注意力处理器
        self.original_attn_processors = self.attn_processors
    
        # 遍历模型中的所有模块
        for module in self.modules():
            # 如果模块是 Attention 类型
            if isinstance(module, Attention):
                # 执行投影融合
                module.fuse_projections(fuse=True)
    
        # 设置新的融合注意力处理器
        self.set_attn_processor(FusedAttnProcessor2_0())
    
    # 从 UNet2DConditionModel 中复制的函数，用于取消融合 QKV 投影
    def unfuse_qkv_projections(self):
        # 禁用已启用的融合 QKV 投影
        """Disables the fused QKV projection if enabled.
    
        <Tip warning={true}>
    
        This API is 🧪 experimental.
    
        </Tip>
    
        """
        # 检查原始注意力处理器是否存在
        if self.original_attn_processors is not None:
            # 恢复到原始注意力处理器
            self.set_attn_processor(self.original_attn_processors)
    
    # 定义前向传播函数
    def forward(
        # 输入隐藏状态的张量
        hidden_states: torch.Tensor,
        # 编码器隐藏状态（可选）
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 时间步长（可选）
        timestep: Optional[torch.LongTensor] = None,
        # 添加的条件关键字参数（字典类型，可选）
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        # 交叉注意力关键字参数（字典类型，可选）
        cross_attention_kwargs: Dict[str, Any] = None,
        # 注意力掩码（可选）
        attention_mask: Optional[torch.Tensor] = None,
        # 编码器注意力掩码（可选）
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # 是否返回字典（默认值为 True）
        return_dict: bool = True,
```