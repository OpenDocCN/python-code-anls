# `.\diffusers\models\transformers\dit_transformer_2d.py`

```py
# 版权声明，表示本代码的版权归 HuggingFace 团队所有，保留所有权利
# 
# 根据 Apache 许可证 2.0 版（“许可证”）进行许可；
# 除非遵守许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非根据适用法律或书面协议另有约定，软件
# 按照“现状”分发，不提供任何形式的保证或条件，
# 明示或暗示。
# 查看许可证以获取有关许可和
# 限制的具体信息。
from typing import Any, Dict, Optional  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的函数式 API
from torch import nn  # 导入 PyTorch 的神经网络模块

from ...configuration_utils import ConfigMixin, register_to_config  # 从配置工具中导入混入类和注册功能
from ...utils import is_torch_version, logging  # 从工具中导入版本检查和日志记录功能
from ..attention import BasicTransformerBlock  # 从注意力模块导入基本变换块
from ..embeddings import PatchEmbed  # 从嵌入模块导入补丁嵌入类
from ..modeling_outputs import Transformer2DModelOutput  # 从建模输出模块导入 2D 变换器模型输出类
from ..modeling_utils import ModelMixin  # 从建模工具中导入模型混入类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，使用 pylint 禁用无效名称警告


class DiTTransformer2DModel(ModelMixin, ConfigMixin):  # 定义一个 2D 变换器模型类，继承自模型混入和配置混入类
    r"""  # 开始文档字符串，描述模型的功能和来源
    A 2D Transformer model as introduced in DiT (https://arxiv.org/abs/2212.09748).  # 描述模型为 DiT 中引入的 2D 变换器模型
    # 定义参数的文档字符串，说明每个参数的作用及默认值
    Parameters:
        # 使用的多头注意力的头数，默认为 16
        num_attention_heads (int, optional, defaults to 16): The number of heads to use for multi-head attention.
        # 每个头的通道数，默认为 72
        attention_head_dim (int, optional, defaults to 72): The number of channels in each head.
        # 输入的通道数，默认为 4
        in_channels (int, defaults to 4): The number of channels in the input.
        # 输出的通道数，如果与输入的通道数不同，需要指定该参数
        out_channels (int, optional):
            The number of channels in the output. Specify this parameter if the output channel number differs from the
            input.
        # Transformer 块的层数，默认为 28
        num_layers (int, optional, defaults to 28): The number of layers of Transformer blocks to use.
        # Transformer 块内使用的 dropout 概率，默认为 0.0
        dropout (float, optional, defaults to 0.0): The dropout probability to use within the Transformer blocks.
        # Transformer 块内组归一化的组数，默认为 32
        norm_num_groups (int, optional, defaults to 32):
            Number of groups for group normalization within Transformer blocks.
        # 配置 Transformer 块的注意力是否包含偏置参数，默认为 True
        attention_bias (bool, optional, defaults to True):
            Configure if the Transformer blocks' attention should contain a bias parameter.
        # 潜在图像的宽度，训练期间该参数是固定的，默认为 32
        sample_size (int, defaults to 32):
            The width of the latent images. This parameter is fixed during training.
        # 模型处理的补丁大小，与处理非序列数据的架构相关，默认为 2
        patch_size (int, defaults to 2):
            Size of the patches the model processes, relevant for architectures working on non-sequential data.
        # 在 Transformer 块内前馈网络中使用的激活函数，默认为 "gelu-approximate"
        activation_fn (str, optional, defaults to "gelu-approximate"):
            Activation function to use in feed-forward networks within Transformer blocks.
        # AdaLayerNorm 的嵌入数量，训练期间固定，影响推理时的最大去噪步骤，默认为 1000
        num_embeds_ada_norm (int, optional, defaults to 1000):
            Number of embeddings for AdaLayerNorm, fixed during training and affects the maximum denoising steps during
            inference.
        # 如果为真，提升注意力机制维度以潜在改善性能，默认为 False
        upcast_attention (bool, optional, defaults to False):
            If true, upcasts the attention mechanism dimensions for potentially improved performance.
        # 指定使用的归一化类型，可以是 'ada_norm_zero'，默认为 "ada_norm_zero"
        norm_type (str, optional, defaults to "ada_norm_zero"):
            Specifies the type of normalization used, can be 'ada_norm_zero'.
        # 如果为真，启用归一化层中的逐元素仿射参数，默认为 False
        norm_elementwise_affine (bool, optional, defaults to False):
            If true, enables element-wise affine parameters in the normalization layers.
        # 在归一化层中添加的一个小常数，以防止除以零，默认为 1e-5
        norm_eps (float, optional, defaults to 1e-5):
            A small constant added to the denominator in normalization layers to prevent division by zero.
    """

    # 支持梯度检查点，以减少内存使用
    _supports_gradient_checkpointing = True

    # 用于注册配置的装饰器
    @register_to_config
    def __init__(
        # 初始化时使用的多头注意力的头数，默认为 16
        num_attention_heads: int = 16,
        # 初始化时每个头的通道数，默认为 72
        attention_head_dim: int = 72,
        # 初始化时输入的通道数，默认为 4
        in_channels: int = 4,
        # 初始化时输出的通道数，默认为 None（可选）
        out_channels: Optional[int] = None,
        # 初始化时 Transformer 块的层数，默认为 28
        num_layers: int = 28,
        # 初始化时使用的 dropout 概率，默认为 0.0
        dropout: float = 0.0,
        # 初始化时组归一化的组数，默认为 32
        norm_num_groups: int = 32,
        # 初始化时注意力的偏置参数，默认为 True
        attention_bias: bool = True,
        # 初始化时潜在图像的宽度，默认为 32
        sample_size: int = 32,
        # 初始化时模型处理的补丁大小，默认为 2
        patch_size: int = 2,
        # 初始化时使用的激活函数，默认为 "gelu-approximate"
        activation_fn: str = "gelu-approximate",
        # 初始化时 AdaLayerNorm 的嵌入数量，默认为 1000（可选）
        num_embeds_ada_norm: Optional[int] = 1000,
        # 初始化时提升注意力机制维度，默认为 False
        upcast_attention: bool = False,
        # 初始化时使用的归一化类型，默认为 "ada_norm_zero"
        norm_type: str = "ada_norm_zero",
        # 初始化时启用归一化层的逐元素仿射参数，默认为 False
        norm_elementwise_affine: bool = False,
        # 初始化时用于归一化层的小常数，默认为 1e-5
        norm_eps: float = 1e-5,
    # 初始化父类
        ):
            super().__init__()
    
            # 验证输入参数是否有效
            if norm_type != "ada_norm_zero":
                # 如果规范类型不正确，抛出未实现错误
                raise NotImplementedError(
                    f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
                )
            elif norm_type == "ada_norm_zero" and num_embeds_ada_norm is None:
                # 当规范类型为 'ada_norm_zero' 且嵌入数为 None 时，抛出值错误
                raise ValueError(
                    f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
                )
    
            # 设置通用变量
            self.attention_head_dim = attention_head_dim
            # 计算内部维度为注意力头数量乘以注意力头维度
            self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
            # 设置输出通道数，如果未指定，则使用输入通道数
            self.out_channels = in_channels if out_channels is None else out_channels
            # 初始化梯度检查点为 False
            self.gradient_checkpointing = False
    
            # 2. 初始化位置嵌入和变换器块
            self.height = self.config.sample_size
            self.width = self.config.sample_size
    
            # 获取补丁大小
            self.patch_size = self.config.patch_size
            # 初始化补丁嵌入对象
            self.pos_embed = PatchEmbed(
                height=self.config.sample_size,
                width=self.config.sample_size,
                patch_size=self.config.patch_size,
                in_channels=self.config.in_channels,
                embed_dim=self.inner_dim,
            )
    
            # 创建变换器块的模块列表
            self.transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        self.inner_dim,
                        self.config.num_attention_heads,
                        self.config.attention_head_dim,
                        dropout=self.config.dropout,
                        activation_fn=self.config.activation_fn,
                        num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                        attention_bias=self.config.attention_bias,
                        upcast_attention=self.config.upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=self.config.norm_elementwise_affine,
                        norm_eps=self.config.norm_eps,
                    )
                    # 根据层数创建多个变换器块
                    for _ in range(self.config.num_layers)
                ]
            )
    
            # 3. 输出层
            # 初始化层归一化
            self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
            # 第一层线性变换，将维度从 inner_dim 扩展到 2 * inner_dim
            self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
            # 第二层线性变换，输出维度为补丁大小的平方乘以输出通道数
            self.proj_out_2 = nn.Linear(
                self.inner_dim, self.config.patch_size * self.config.patch_size * self.out_channels
            )
    
        # 设置梯度检查点的功能
        def _set_gradient_checkpointing(self, module, value=False):
            # 如果模块有梯度检查点属性，则设置其值
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = value
    
        # 前向传播函数定义
        def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            return_dict: bool = True,
```