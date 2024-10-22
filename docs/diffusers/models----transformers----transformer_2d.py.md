# `.\diffusers\models\transformers\transformer_2d.py`

```py
# 版权声明，指明版权归 2024 年 HuggingFace 团队所有
# 
# 根据 Apache 许可证第 2.0 版（"许可证"）进行许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面协议另有约定，根据许可证分发的软件是以 "按现状" 的基础进行分发，
# 不提供任何形式的明示或暗示的担保或条件。
# 请参阅许可证以获取有关权限和
# 限制的特定语言。
from typing import Any, Dict, Optional  # 导入用于类型注释的 Any、Dict 和 Optional 模块

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的函数式神经网络模块，通常用于激活函数等
from torch import nn  # 从 PyTorch 中导入 nn 模块，用于构建神经网络

from ...configuration_utils import LegacyConfigMixin, register_to_config  # 从配置工具导入遗留配置混合类和注册配置函数
from ...utils import deprecate, is_torch_version, logging  # 从工具模块导入弃用函数、PyTorch 版本检查函数和日志功能
from ..attention import BasicTransformerBlock  # 从注意力模块导入基础变换器块
from ..embeddings import ImagePositionalEmbeddings, PatchEmbed, PixArtAlphaTextProjection  # 从嵌入模块导入图像位置嵌入、补丁嵌入和 PixArt Alpha 文本投影
from ..modeling_outputs import Transformer2DModelOutput  # 从建模输出模块导入 2D 变换器模型输出类
from ..modeling_utils import LegacyModelMixin  # 从建模工具模块导入遗留模型混合类
from ..normalization import AdaLayerNormSingle  # 从归一化模块导入 AdaLayerNormSingle 类


logger = logging.get_logger(__name__)  # 创建一个与当前模块名称相关的日志记录器，禁用 pylint 的无效名称警告


class Transformer2DModelOutput(Transformer2DModelOutput):  # 定义 Transformer2DModelOutput 类，继承自 Transformer2DModelOutput
    def __init__(self, *args, **kwargs):  # 构造函数，接受任意参数
        deprecation_message = "Importing `Transformer2DModelOutput` from `diffusers.models.transformer_2d` is deprecated and this will be removed in a future version. Please use `from diffusers.models.modeling_outputs import Transformer2DModelOutput`, instead."  # 设置弃用信息
        deprecate("Transformer2DModelOutput", "1.0.0", deprecation_message)  # 调用弃用函数，记录弃用信息
        super().__init__(*args, **kwargs)  # 调用父类的构造函数，传递参数


class Transformer2DModel(LegacyModelMixin, LegacyConfigMixin):  # 定义 Transformer2DModel 类，继承遗留模型混合类和遗留配置混合类
    """
    A 2D Transformer model for image-like data.  # 类文档字符串，说明这是一个用于图像类数据的 2D 变换器模型。
    # 定义参数部分，用于描述模型配置
    Parameters:
        # 多头注意力中使用的头数，默认为16
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        # 每个头中的通道数，默认为88
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        # 输入和输出中的通道数（如果输入是**连续**，则需指定）
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        # 使用的Transformer块的层数，默认为1
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        # 使用的丢弃概率，默认为0.0
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        # 使用的`encoder_hidden_states`维度数
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        # 潜在图像的宽度（如果输入是**离散**，则需指定）
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            # 在训练期间固定使用，以学习多个位置嵌入
            This is fixed during training since it is used to learn a number of position embeddings.
        # 潜在像素的向量嵌入类数（如果输入是**离散**，则需指定）
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            # 包含用于掩蔽潜在像素的类
            Includes the class for the masked latent pixel.
        # 前馈中的激活函数，默认为"geglu"
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        # 训练期间使用的扩散步骤数。如果至少有一个norm_layers是
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            # `AdaLayerNorm`。在训练期间固定使用，以学习多个嵌入
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            # 添加到隐藏状态。
            added to the hidden states.

            # 在推理期间，可以去噪的步骤数最多不超过`num_embeds_ada_norm`
            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        # 配置是否`TransformerBlocks`的注意力应包含偏差参数
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    # 支持梯度检查点功能的标志
    _supports_gradient_checkpointing = True
    # 不进行拆分的模块列表
    _no_split_modules = ["BasicTransformerBlock"]

    # 注册到配置的装饰器
    @register_to_config
    # 初始化方法，用于设置模型的参数
    def __init__(
        # 设置注意力头的数量，默认为16
        self,
        num_attention_heads: int = 16,
        # 每个注意力头的维度，默认为88
        attention_head_dim: int = 88,
        # 输入通道数，默认为None，表示未指定
        in_channels: Optional[int] = None,
        # 输出通道数，默认为None，表示未指定
        out_channels: Optional[int] = None,
        # 模型层数，默认为1
        num_layers: int = 1,
        # dropout比率，默认为0.0
        dropout: float = 0.0,
        # 归一化时的组数，默认为32
        norm_num_groups: int = 32,
        # 交叉注意力维度，默认为None，表示未指定
        cross_attention_dim: Optional[int] = None,
        # 是否使用注意力偏差，默认为False
        attention_bias: bool = False,
        # 采样大小，默认为None，表示未指定
        sample_size: Optional[int] = None,
        # 向量嵌入的数量，默认为None，表示未指定
        num_vector_embeds: Optional[int] = None,
        # patch大小，默认为None，表示未指定
        patch_size: Optional[int] = None,
        # 激活函数类型，默认为"geglu"
        activation_fn: str = "geglu",
        # 自适应归一化嵌入的数量，默认为None，表示未指定
        num_embeds_ada_norm: Optional[int] = None,
        # 是否使用线性投影，默认为False
        use_linear_projection: bool = False,
        # 是否仅使用交叉注意力，默认为False
        only_cross_attention: bool = False,
        # 是否使用双重自注意力，默认为False
        double_self_attention: bool = False,
        # 是否提高注意力精度，默认为False
        upcast_attention: bool = False,
        # 归一化类型，默认为"layer_norm"
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        # 归一化时是否使用元素级仿射，默认为True
        norm_elementwise_affine: bool = True,
        # 归一化的epsilon值，默认为1e-5
        norm_eps: float = 1e-5,
        # 注意力类型，默认为"default"
        attention_type: str = "default",
        # 说明通道数，默认为None，表示未指定
        caption_channels: int = None,
        # 插值缩放因子，默认为None，表示未指定
        interpolation_scale: float = None,
        # 是否使用额外条件，默认为None，表示未指定
        use_additional_conditions: Optional[bool] = None,
    # 初始化连续输入的方法，接受归一化类型作为参数
    def _init_continuous_input(self, norm_type):
        # 创建归一化层，使用组归一化，设置组数、通道数和epsilon
        self.norm = torch.nn.GroupNorm(
            num_groups=self.config.norm_num_groups, num_channels=self.in_channels, eps=1e-6, affine=True
        )
        # 如果使用线性投影，则创建线性层进行输入投影
        if self.use_linear_projection:
            self.proj_in = torch.nn.Linear(self.in_channels, self.inner_dim)
        # 否则，创建卷积层进行输入投影
        else:
            self.proj_in = torch.nn.Conv2d(self.in_channels, self.inner_dim, kernel_size=1, stride=1, padding=0)

        # 创建变换器块的模块列表
        self.transformer_blocks = nn.ModuleList(
            [
                # 对于每一层，初始化一个基本变换器块
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                )
                # 重复上面的块，根据模型层数
                for _ in range(self.config.num_layers)
            ]
        )

        # 如果使用线性投影，则创建线性层进行输出投影
        if self.use_linear_projection:
            self.proj_out = torch.nn.Linear(self.inner_dim, self.out_channels)
        # 否则，创建卷积层进行输出投影
        else:
            self.proj_out = torch.nn.Conv2d(self.inner_dim, self.out_channels, kernel_size=1, stride=1, padding=0)
    # 初始化向量化输入的方法，接收规范类型作为参数
        def _init_vectorized_inputs(self, norm_type):
            # 确保配置中的样本大小不为 None，否则抛出错误信息
            assert self.config.sample_size is not None, "Transformer2DModel over discrete input must provide sample_size"
            # 确保配置中的向量嵌入数量不为 None，否则抛出错误信息
            assert (
                self.config.num_vector_embeds is not None
            ), "Transformer2DModel over discrete input must provide num_embed"
    
            # 从配置中获取样本大小并赋值给高度
            self.height = self.config.sample_size
            # 从配置中获取样本大小并赋值给宽度
            self.width = self.config.sample_size
            # 计算潜在像素的总数量，等于高度乘以宽度
            self.num_latent_pixels = self.height * self.width
    
            # 创建图像位置嵌入对象，用于处理向量嵌入和图像维度
            self.latent_image_embedding = ImagePositionalEmbeddings(
                num_embed=self.config.num_vector_embeds, embed_dim=self.inner_dim, height=self.height, width=self.width
            )
    
            # 创建一个包含基本变换块的模块列表
            self.transformer_blocks = nn.ModuleList(
                [
                    # 为每一层创建一个基本变换块
                    BasicTransformerBlock(
                        self.inner_dim,
                        self.config.num_attention_heads,
                        self.config.attention_head_dim,
                        dropout=self.config.dropout,
                        cross_attention_dim=self.config.cross_attention_dim,
                        activation_fn=self.config.activation_fn,
                        num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                        attention_bias=self.config.attention_bias,
                        only_cross_attention=self.config.only_cross_attention,
                        double_self_attention=self.config.double_self_attention,
                        upcast_attention=self.config.upcast_attention,
                        norm_type=norm_type,
                        norm_elementwise_affine=self.config.norm_elementwise_affine,
                        norm_eps=self.config.norm_eps,
                        attention_type=self.config.attention_type,
                    )
                    # 通过配置中的层数决定变换块的数量
                    for _ in range(self.config.num_layers)
                ]
            )
    
            # 创建输出层归一化层
            self.norm_out = nn.LayerNorm(self.inner_dim)
            # 创建线性层，将内部维度映射到向量嵌入数量减一
            self.out = nn.Linear(self.inner_dim, self.config.num_vector_embeds - 1)
    
        # 设置梯度检查点的方法，接收模块和布尔值作为参数
        def _set_gradient_checkpointing(self, module, value=False):
            # 检查模块是否具有梯度检查点属性
            if hasattr(module, "gradient_checkpointing"):
                # 设置模块的梯度检查点属性
                module.gradient_checkpointing = value
    
        # 前向传播方法，处理输入的隐藏状态及其他可选参数
        def forward(
            self,
            hidden_states: torch.Tensor,  # 隐藏状态张量
            encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态，默认为 None
            timestep: Optional[torch.LongTensor] = None,  # 时间步长，默认为 None
            added_cond_kwargs: Dict[str, torch.Tensor] = None,  # 额外条件的字典，默认为 None
            class_labels: Optional[torch.LongTensor] = None,  # 类标签，默认为 None
            cross_attention_kwargs: Dict[str, Any] = None,  # 交叉注意力参数字典，默认为 None
            attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，默认为 None
            encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器注意力掩码，默认为 None
            return_dict: bool = True,  # 是否返回字典格式的结果，默认为 True
    # 对连续输入进行操作，处理隐藏状态
        def _operate_on_continuous_inputs(self, hidden_states):
            # 获取隐藏状态的批次大小、高度和宽度
            batch, _, height, width = hidden_states.shape
            # 对隐藏状态进行归一化处理
            hidden_states = self.norm(hidden_states)
    
            # 如果不使用线性投影
            if not self.use_linear_projection:
                # 通过输入投影层处理隐藏状态
                hidden_states = self.proj_in(hidden_states)
                # 获取内部维度
                inner_dim = hidden_states.shape[1]
                # 调整隐藏状态的维度
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            else:
                # 获取内部维度
                inner_dim = hidden_states.shape[1]
                # 调整隐藏状态的维度
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                # 通过输入投影层处理隐藏状态
                hidden_states = self.proj_in(hidden_states)
    
            # 返回处理后的隐藏状态和内部维度
            return hidden_states, inner_dim
    
        # 对修补输入进行操作，处理隐藏状态和编码器隐藏状态
        def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, added_cond_kwargs):
            # 获取批次大小
            batch_size = hidden_states.shape[0]
            # 对隐藏状态进行位置嵌入
            hidden_states = self.pos_embed(hidden_states)
            # 初始化嵌入时间步
            embedded_timestep = None
    
            # 如果自适应归一化单元存在
            if self.adaln_single is not None:
                # 如果使用额外条件且未提供额外参数，抛出错误
                if self.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError(
                        "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                    )
                # 处理时间步和嵌入时间步
                timestep, embedded_timestep = self.adaln_single(
                    timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
                )
    
            # 如果存在标题投影
            if self.caption_projection is not None:
                # 对编码器隐藏状态进行投影
                encoder_hidden_states = self.caption_projection(encoder_hidden_states)
                # 调整编码器隐藏状态的维度
                encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
    
            # 返回处理后的隐藏状态、编码器隐藏状态、时间步和嵌入时间步
            return hidden_states, encoder_hidden_states, timestep, embedded_timestep
    
        # 获取连续输入的输出，处理隐藏状态和残差
        def _get_output_for_continuous_inputs(self, hidden_states, residual, batch_size, height, width, inner_dim):
            # 如果不使用线性投影
            if not self.use_linear_projection:
                # 调整隐藏状态的维度
                hidden_states = (
                    hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                )
                # 通过输出投影层处理隐藏状态
                hidden_states = self.proj_out(hidden_states)
            else:
                # 通过输出投影层处理隐藏状态
                hidden_states = self.proj_out(hidden_states)
                # 调整隐藏状态的维度
                hidden_states = (
                    hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                )
    
            # 将处理后的隐藏状态与残差相加
            output = hidden_states + residual
            # 返回最终输出
            return output
    
        # 获取向量化输入的输出，处理隐藏状态
        def _get_output_for_vectorized_inputs(self, hidden_states):
            # 对隐藏状态进行归一化处理
            hidden_states = self.norm_out(hidden_states)
            # 通过输出层处理隐藏状态，得到 logits
            logits = self.out(hidden_states)
            # 调整 logits 的维度
            logits = logits.permute(0, 2, 1)
            # 对 logits 应用 log_softmax，获取最终输出
            output = F.log_softmax(logits.double(), dim=1).float()
            # 返回最终输出
            return output
    
        # 获取修补输入的输出，处理隐藏状态和时间步
        def _get_output_for_patched_inputs(
            self, hidden_states, timestep, class_labels, embedded_timestep, height=None, width=None
    ):
        # 检查配置中的归一化类型是否不是 "ada_norm_single"
        if self.config.norm_type != "ada_norm_single":
            # 使用第一个变换块的归一化层对时间步和类别标签进行嵌入处理
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            # 将条件信息通过线性变换获得偏移和缩放因子
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            # 对隐藏状态进行归一化和调整，应用偏移和缩放
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            # 将调整后的隐藏状态通过第二个线性变换
            hidden_states = self.proj_out_2(hidden_states)
        # 检查配置中的归一化类型是否是 "ada_norm_single"
        elif self.config.norm_type == "ada_norm_single":
            # 从缩放偏移表中获得偏移和缩放因子
            shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
            # 对隐藏状态进行归一化处理
            hidden_states = self.norm_out(hidden_states)
            # 调整隐藏状态，应用偏移和缩放
            hidden_states = hidden_states * (1 + scale) + shift
            # 将调整后的隐藏状态通过线性变换
            hidden_states = self.proj_out(hidden_states)
            # 压缩维度，去掉多余的维度
            hidden_states = hidden_states.squeeze(1)

        # 取消补丁化处理
        if self.adaln_single is None:
            # 计算高度和宽度，基于隐藏状态的形状
            height = width = int(hidden_states.shape[1] ** 0.5)
        # 重新调整隐藏状态的形状
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        )
        # 使用爱因斯坦求和约定重排维度
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        # 再次调整输出的形状
        output = hidden_states.reshape(
            shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        )
        # 返回最终的输出结果
        return output
```