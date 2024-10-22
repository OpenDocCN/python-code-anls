# `.\diffusers\models\transformers\prior_transformer.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 导入字典、可选值和联合类型的定义
from typing import Dict, Optional, Union

# 导入 PyTorch 及其功能模块
import torch
import torch.nn.functional as F
# 从 PyTorch 导入神经网络模块
from torch import nn

# 导入配置和注册功能的相关类
from ...configuration_utils import ConfigMixin, register_to_config
# 导入 PeftAdapter 和 UNet2DConditionLoader 的相关类
from ...loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
# 导入基础输出工具类
from ...utils import BaseOutput
# 导入基本变换器块
from ..attention import BasicTransformerBlock
# 导入注意力处理器的相关组件
from ..attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
# 导入时间步嵌入和时间步类
from ..embeddings import TimestepEmbedding, Timesteps
# 导入模型混合工具类
from ..modeling_utils import ModelMixin

# 定义 PriorTransformerOutput 数据类，继承自 BaseOutput
@dataclass
class PriorTransformerOutput(BaseOutput):
    """
    [`PriorTransformer`] 的输出。

    Args:
        predicted_image_embedding (`torch.Tensor` 的形状为 `(batch_size, embedding_dim)`):
            基于 CLIP 文本嵌入输入的预测 CLIP 图像嵌入。
    """

    # 定义预测的图像嵌入属性
    predicted_image_embedding: torch.Tensor


# 定义 PriorTransformer 类，继承多个混合类
class PriorTransformer(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin):
    """
    一种 Prior Transformer 模型。
    # 参数说明部分
    Parameters:
        # 用于多头注意力的头数量，默认为 32
        num_attention_heads (`int`, *optional*, defaults to 32): The number of heads to use for multi-head attention.
        # 每个头的通道数量，默认为 64
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        # Transformer 块的层数，默认为 20
        num_layers (`int`, *optional*, defaults to 20): The number of layers of Transformer blocks to use.
        # 模型输入 `hidden_states` 的维度，默认为 768
        embedding_dim (`int`, *optional*, defaults to 768): The dimension of the model input `hidden_states`
        # 模型输入 `hidden_states` 的嵌入数量，默认为 77
        num_embeddings (`int`, *optional*, defaults to 77):
            The number of embeddings of the model input `hidden_states`
        # 附加令牌的数量，默认为 4，追加到投影的 `hidden_states`
        additional_embeddings (`int`, *optional*, defaults to 4): The number of additional tokens appended to the
            projected `hidden_states`. The actual length of the used `hidden_states` is `num_embeddings +
            additional_embeddings`.
        # 用于 dropout 的概率，默认为 0.0
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        # 创建时间步嵌入时使用的激活函数，默认为 'silu'
        time_embed_act_fn (`str`, *optional*, defaults to 'silu'):
            The activation function to use to create timestep embeddings.
        # 在传递给 Transformer 块之前应用的归一化层，默认为 None
        norm_in_type (`str`, *optional*, defaults to None): The normalization layer to apply on hidden states before
            passing to Transformer blocks. Set it to `None` if normalization is not needed.
        # 输入 `proj_embedding` 上应用的归一化层，默认为 None
        embedding_proj_norm_type (`str`, *optional*, defaults to None):
            The normalization layer to apply on the input `proj_embedding`. Set it to `None` if normalization is not
            needed.
        # 输入 `encoder_hidden_states` 上应用的投影层，默认为 `linear`
        encoder_hid_proj_type (`str`, *optional*, defaults to `linear`):
            The projection layer to apply on the input `encoder_hidden_states`. Set it to `None` if
            `encoder_hidden_states` is `None`.
        # 条件模型的附加嵌入类型，默认为 `prd`
        added_emb_type (`str`, *optional*, defaults to `prd`): Additional embeddings to condition the model.
            Choose from `prd` or `None`. if choose `prd`, it will prepend a token indicating the (quantized) dot
            product between the text embedding and image embedding as proposed in the unclip paper
            https://arxiv.org/abs/2204.06125 If it is `None`, no additional embeddings will be prepended.
        # 时间步嵌入的维度，默认为 None，如果为 None，则设置为 `num_attention_heads * attention_head_dim`
        time_embed_dim (`int, *optional*, defaults to None): The dimension of timestep embeddings.
            If None, will be set to `num_attention_heads * attention_head_dim`
        # `proj_embedding` 的维度，默认为 None，如果为 None，则设置为 `embedding_dim`
        embedding_proj_dim (`int`, *optional*, default to None):
            The dimension of `proj_embedding`. If None, will be set to `embedding_dim`.
        # 输出的维度，默认为 None，如果为 None，则设置为 `embedding_dim`
        clip_embed_dim (`int`, *optional*, default to None):
            The dimension of the output. If None, will be set to `embedding_dim`.
    """

    # 注册到配置中
    @register_to_config
    # 初始化类的构造函数，设置默认参数
        def __init__(
            # 注意力头的数量，默认值为32
            self,
            num_attention_heads: int = 32,
            # 每个注意力头的维度，默认值为64
            attention_head_dim: int = 64,
            # 层的数量，默认值为20
            num_layers: int = 20,
            # 嵌入的维度，默认值为768
            embedding_dim: int = 768,
            # 嵌入的数量，默认值为77
            num_embeddings=77,
            # 额外嵌入的数量，默认值为4
            additional_embeddings=4,
            # dropout的比率，默认值为0.0
            dropout: float = 0.0,
            # 时间嵌入激活函数的类型，默认值为"silu"
            time_embed_act_fn: str = "silu",
            # 输入归一化类型，默认为None
            norm_in_type: Optional[str] = None,  # layer
            # 嵌入投影归一化类型，默认为None
            embedding_proj_norm_type: Optional[str] = None,  # layer
            # 编码器隐藏投影类型，默认值为"linear"
            encoder_hid_proj_type: Optional[str] = "linear",  # linear
            # 添加的嵌入类型，默认值为"prd"
            added_emb_type: Optional[str] = "prd",  # prd
            # 时间嵌入维度，默认为None
            time_embed_dim: Optional[int] = None,
            # 嵌入投影维度，默认为None
            embedding_proj_dim: Optional[int] = None,
            # 裁剪嵌入维度，默认为None
            clip_embed_dim: Optional[int] = None,
        # 定义一个属性，获取注意力处理器
        @property
        # 从diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors复制而来
        def attn_processors(self) -> Dict[str, AttentionProcessor]:
            r"""
            返回值：
                `dict`类型的注意力处理器：一个字典，包含模型中使用的所有注意力处理器，并按其权重名称索引。
            """
            # 创建一个空字典用于存储处理器
            processors = {}
    
            # 定义递归函数，添加处理器到字典
            def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
                # 如果模块有获取处理器的方法，添加到字典中
                if hasattr(module, "get_processor"):
                    processors[f"{name}.processor"] = module.get_processor()
    
                # 遍历模块的所有子模块，递归调用自身
                for sub_name, child in module.named_children():
                    fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
    
                # 返回更新后的处理器字典
                return processors
    
            # 遍历当前类的所有子模块，调用递归函数添加处理器
            for name, module in self.named_children():
                fn_recursive_add_processors(name, module, processors)
    
            # 返回最终的处理器字典
            return processors
    
        # 从diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor复制而来
    # 设置用于计算注意力的处理器
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的处理器。
    
        参数：
            processor（`dict` 或 `AttentionProcessor`）：
                实例化的处理器类或处理器类的字典，将作为 **所有** `Attention` 层的处理器。
    
                如果 `processor` 是一个字典，键需要定义相应的交叉注意力处理器的路径。建议在设置可训练注意力处理器时使用此方法。
    
        """
        # 计算当前注意力处理器的数量
        count = len(self.attn_processors.keys())
    
        # 如果传入的是字典且其长度与注意力层数量不匹配，抛出错误
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"传入了处理器的字典，但处理器的数量 {len(processor)} 与注意力层的数量 {count} 不匹配。请确保传入 {count} 个处理器类。"
            )
    
        # 定义递归设置注意力处理器的函数
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 如果模块具有 set_processor 方法，则设置处理器
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
    
            # 遍历子模块并递归调用
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
    
        # 遍历所有子模块，调用递归函数设置处理器
        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)
    
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel 复制的设置默认注意力处理器的方法
    def set_default_attn_processor(self):
        """
        禁用自定义注意力处理器并设置默认的注意力实现。
        """
        # 如果所有处理器都是添加的 KV 注意力处理器，则设置为添加的 KV 处理器
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        # 如果所有处理器都是交叉注意力处理器，则设置为普通的注意力处理器
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"当注意力处理器的类型为 {next(iter(self.attn_processors.values()))} 时，无法调用 `set_default_attn_processor`"
            )
    
        # 调用设置处理器的方法
        self.set_attn_processor(processor)
    
    # 前向传播方法定义
    def forward(
        self,
        hidden_states,
        timestep: Union[torch.Tensor, float, int],
        proj_embedding: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        return_dict: bool = True,
    ):
        # 处理传入的潜在变量
        def post_process_latents(self, prior_latents):
            # 将潜在变量进行标准化处理
            prior_latents = (prior_latents * self.clip_std) + self.clip_mean
            return prior_latents
```