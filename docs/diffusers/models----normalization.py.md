# `.\diffusers\models\normalization.py`

```py
# 指定文件编码为 UTF-8
# copyright 信息，标识版权所有者及年份
# 许可证声明，指明使用的许可证类型及条件
# 提供许可证的获取链接
# 声明在适用情况下，软件是以“原样”方式分发的，且不提供任何形式的担保或条件
# 引用许可证中关于权限和限制的具体条款

# 导入 numbers 模块，用于处理数值相关的操作
from typing import Dict, Optional, Tuple  # 导入类型提示所需的类型

# 导入 PyTorch 相关模块和功能
import torch
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入功能性神经网络操作模块

# 导入工具函数以检查 PyTorch 版本
from ..utils import is_torch_version
# 导入激活函数获取方法
from .activations import get_activation
# 导入嵌入层相关类
from .embeddings import (
    CombinedTimestepLabelEmbeddings,
    PixArtAlphaCombinedTimestepSizeEmbeddings,
)


class AdaLayerNorm(nn.Module):  # 定义自定义的层归一化类，继承自 nn.Module
    r"""  # 文档字符串，描述此类的功能和参数
    Norm layer modified to incorporate timestep embeddings.  # 说明此层归一化是为了支持时间步嵌入

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.  # 嵌入向量的维度
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.  # 嵌入字典的大小（可选）
        output_dim (`int`, *optional*):  # 输出维度（可选）
        norm_elementwise_affine (`bool`, defaults to `False):  # 是否应用元素级仿射变换（默认 False）
        norm_eps (`bool`, defaults to `False`):  # 归一化时的小常数（默认 1e-5）
        chunk_dim (`int`, defaults to `0`):  # 分块维度（默认 0）
    """

    def __init__(  # 初始化方法，定义类的构造函数
        self,
        embedding_dim: int,  # 嵌入维度
        num_embeddings: Optional[int] = None,  # 嵌入字典的大小（可选）
        output_dim: Optional[int] = None,  # 输出维度（可选）
        norm_elementwise_affine: bool = False,  # 是否应用元素级仿射变换
        norm_eps: float = 1e-5,  # 归一化时的小常数
        chunk_dim: int = 0,  # 分块维度
    ):
        super().__init__()  # 调用父类构造函数

        self.chunk_dim = chunk_dim  # 保存分块维度
        output_dim = output_dim or embedding_dim * 2  # 如果未指定输出维度，则计算输出维度

        if num_embeddings is not None:  # 如果指定了嵌入字典大小
            self.emb = nn.Embedding(num_embeddings, embedding_dim)  # 初始化嵌入层
        else:
            self.emb = None  # 嵌入层为 None

        self.silu = nn.SiLU()  # 初始化 SiLU 激活函数
        self.linear = nn.Linear(embedding_dim, output_dim)  # 初始化线性层
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)  # 初始化层归一化

    def forward(  # 定义前向传播方法
        self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None, temb: Optional[torch.Tensor] = None  # 输入张量及可选时间步和嵌入
    ) -> torch.Tensor:  # 返回类型为张量
        if self.emb is not None:  # 如果嵌入层存在
            temb = self.emb(timestep)  # 通过嵌入层计算时间步的嵌入

        temb = self.linear(self.silu(temb))  # 应用激活函数并通过线性层处理嵌入

        if self.chunk_dim == 1:  # 如果分块维度为 1
            # 对于 CogVideoX 的特殊情况，分割嵌入为偏移量和缩放量
            shift, scale = temb.chunk(2, dim=1)  # 按照维度 1 分块
            shift = shift[:, None, :]  # 扩展偏移量维度
            scale = scale[:, None, :]  # 扩展缩放量维度
        else:  # 如果分块维度不是 1
            scale, shift = temb.chunk(2, dim=0)  # 按照维度 0 分块

        x = self.norm(x) * (1 + scale) + shift  # 进行层归一化，并应用缩放和偏移
        return x  # 返回结果


class FP32LayerNorm(nn.LayerNorm):  # 定义 FP32 层归一化类，继承自 nn.LayerNorm
    # 定义前向传播方法，接受输入张量并返回输出张量
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            # 保存输入张量的数据类型
            origin_dtype = inputs.dtype
            # 进行层归一化处理，并将结果转换回原始数据类型
            return F.layer_norm(
                # 将输入张量转换为浮点型进行归一化
                inputs.float(),
                # 归一化的形状
                self.normalized_shape,
                # 如果权重存在，将其转换为浮点型；否则为 None
                self.weight.float() if self.weight is not None else None,
                # 如果偏置存在，将其转换为浮点型；否则为 None
                self.bias.float() if self.bias is not None else None,
                # 设置一个小的数值以避免除零
                self.eps,
            ).to(origin_dtype)  # 将归一化后的结果转换回原始数据类型
# 定义自适应层归一化零层的类
class AdaLayerNormZero(nn.Module):
    r"""
    自适应层归一化零层 (adaLN-Zero)。

    参数：
        embedding_dim (`int`): 每个嵌入向量的大小。
        num_embeddings (`int`): 嵌入字典的大小。
    """

    # 初始化方法，接收嵌入维度和可选的嵌入数量及归一化类型
    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        # 调用父类初始化方法
        super().__init__()
        # 如果提供了嵌入数量，初始化嵌入层
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            # 否则，嵌入层设置为 None
            self.emb = None

        # 初始化 SiLU 激活函数
        self.silu = nn.SiLU()
        # 初始化线性变换层，输出维度为 6 倍的嵌入维度
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        # 根据提供的归一化类型，初始化归一化层
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            # 如果提供了不支持的归一化类型，抛出错误
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    # 定义前向传播方法
    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 如果嵌入层不为 None，则计算嵌入
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        # 先经过 SiLU 激活函数再经过线性变换
        emb = self.linear(self.silu(emb))
        # 将嵌入切分为 6 个部分
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        # 对输入 x 应用归一化，并结合缩放和偏移
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        # 返回处理后的 x 及其他信息
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# 定义自适应层归一化零层单一版本的类
class AdaLayerNormZeroSingle(nn.Module):
    r"""
    自适应层归一化零层 (adaLN-Zero)。

    参数：
        embedding_dim (`int`): 每个嵌入向量的大小。
        num_embeddings (`int`): 嵌入字典的大小。
    """

    # 初始化方法，接收嵌入维度和归一化类型
    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        # 调用父类初始化方法
        super().__init__()

        # 初始化 SiLU 激活函数
        self.silu = nn.SiLU()
        # 初始化线性变换层，输出维度为 3 倍的嵌入维度
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        # 根据提供的归一化类型，初始化归一化层
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            # 如果提供了不支持的归一化类型，抛出错误
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    # 定义前向传播方法
    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    # 定义一个函数的返回类型为五个张量的元组
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 通过线性层和Silu激活函数处理嵌入向量
            emb = self.linear(self.silu(emb))
        # 将处理后的嵌入向量分割成三个部分：shift_msa, scale_msa 和 gate_msa
            shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        # 对输入x进行归一化，并结合scale和shift进行变换
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        # 返回变换后的x和gate_msa
            return x, gate_msa
# 定义 LuminaRMSNormZero 类，继承自 nn.Module
class LuminaRMSNormZero(nn.Module):
    """
    Norm layer adaptive RMS normalization zero.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    # 初始化方法，设置嵌入维度、正则化参数和元素级偏置
    def __init__(self, embedding_dim: int, norm_eps: float, norm_elementwise_affine: bool):
        # 调用父类构造函数
        super().__init__()
        # 初始化 SiLU 激活函数
        self.silu = nn.SiLU()
        # 初始化线性变换层，输入为 embedding_dim 或 1024 中的较小值，输出为 4 倍的 embedding_dim
        self.linear = nn.Linear(
            min(embedding_dim, 1024),
            4 * embedding_dim,
            bias=True,
        )
        # 初始化 RMSNorm 层
        self.norm = RMSNorm(embedding_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

    # 前向传播方法
    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 对 emb 应用线性变换和 SiLU 激活
        emb = self.linear(self.silu(emb))
        # 将嵌入分块为四个部分
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        # 对输入 x 应用 RMSNorm 并与 scale_msa 相乘
        x = self.norm(x) * (1 + scale_msa[:, None])

        # 返回处理后的 x 以及门控和缩放值
        return x, gate_msa, scale_mlp, gate_mlp


# 定义 AdaLayerNormSingle 类，继承自 nn.Module
class AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    # 初始化方法，设置嵌入维度和是否使用额外条件
    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False):
        # 调用父类构造函数
        super().__init__()

        # 初始化 PixArtAlphaCombinedTimestepSizeEmbeddings，用于时间步嵌入
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
        )

        # 初始化 SiLU 激活函数
        self.silu = nn.SiLU()
        # 初始化线性变换层，输出为 6 倍的嵌入维度
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    # 前向传播方法
    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 嵌入时间步，可能使用额外的条件
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        # 返回线性变换后的嵌入和嵌入结果
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


# 定义 AdaGroupNorm 类，继承自 nn.Module
class AdaGroupNorm(nn.Module):
    r"""
    GroupNorm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
        num_groups (`int`): The number of groups to separate the channels into.
        act_fn (`str`, *optional*, defaults to `None`): The activation function to use.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
    """
    # 初始化方法，用于设置类的基本属性
        def __init__(
            # 嵌入向量的维度
            self, embedding_dim: int, 
            # 输出向量的维度
            out_dim: int, 
            # 组的数量
            num_groups: int, 
            # 激活函数名称（可选）
            act_fn: Optional[str] = None, 
            # 防止除零错误的微小值
            eps: float = 1e-5
        ):
            # 调用父类初始化方法
            super().__init__()
            # 设置组的数量
            self.num_groups = num_groups
            # 设置用于数值稳定性的微小值
            self.eps = eps
    
            # 如果没有提供激活函数，则设置为 None
            if act_fn is None:
                self.act = None
            else:
                # 根据激活函数名称获取激活函数
                self.act = get_activation(act_fn)
    
            # 创建一个线性层，将嵌入维度映射到输出维度的两倍
            self.linear = nn.Linear(embedding_dim, out_dim * 2)
    
        # 前向传播方法，定义输入数据的处理方式
        def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
            # 如果存在激活函数，则对嵌入进行激活
            if self.act:
                emb = self.act(emb)
            # 将嵌入传递通过线性层
            emb = self.linear(emb)
            # 扩展嵌入的维度，以适配后续操作
            emb = emb[:, :, None, None]
            # 将嵌入分割为缩放因子和偏移量
            scale, shift = emb.chunk(2, dim=1)
    
            # 对输入数据进行分组归一化
            x = F.group_norm(x, self.num_groups, eps=self.eps)
            # 使用缩放因子和偏移量调整归一化后的数据
            x = x * (1 + scale) + shift
            # 返回处理后的数据
            return x
# 定义一个自定义的神经网络模块，继承自 nn.Module
class AdaLayerNormContinuous(nn.Module):
    # 初始化方法，接受多个参数以配置层的特性
    def __init__(
        self,
        embedding_dim: int,  # 嵌入维度
        conditioning_embedding_dim: int,  # 条件嵌入维度
        # 注释：规范层可以配置缩放和偏移参数有点奇怪，因为输出会被投影的条件嵌入立即缩放和偏移。
        # 注意，AdaLayerNorm 不允许规范层有缩放和偏移参数。
        # 但是这是原始代码中的实现，您应该将 `elementwise_affine` 设置为 False。
        elementwise_affine=True,  # 是否允许元素级的仿射变换
        eps=1e-5,  # 防止除零错误的小值
        bias=True,  # 是否在全连接层中使用偏置
        norm_type="layer_norm",  # 规范化类型
    ):
        super().__init__()  # 调用父类构造函数
        self.silu = nn.SiLU()  # 定义 SiLU 激活函数
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)  # 全连接层，输出两倍嵌入维度
        # 根据指定的规范类型初始化规范层
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)  # 层规范化
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)  # RMS 规范化
        else:
            raise ValueError(f"unknown norm_type {norm_type}")  # 抛出错误，若规范类型未知

    # 前向传播方法，定义如何计算输出
    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # 将条件嵌入转换为与输入 x 相同的数据类型
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))  # 应用激活函数和全连接层
        scale, shift = torch.chunk(emb, 2, dim=1)  # 将输出拆分为缩放和偏移
        # 规范化输入 x，并进行缩放和偏移操作
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]  # 返回处理后的输出
        return x  # 返回最终结果


# 定义另一个自定义的神经网络模块，继承自 nn.Module
class LuminaLayerNormContinuous(nn.Module):
    # 初始化方法，接受多个参数以配置层的特性
    def __init__(
        self,
        embedding_dim: int,  # 嵌入维度
        conditioning_embedding_dim: int,  # 条件嵌入维度
        # 注释：规范层可以配置缩放和偏移参数有点奇怪，因为输出会被投影的条件嵌入立即缩放和偏移。
        # 注意，AdaLayerNorm 不允许规范层有缩放和偏移参数。
        # 但是这是原始代码中的实现，您应该将 `elementwise_affine` 设置为 False。
        elementwise_affine=True,  # 是否允许元素级的仿射变换
        eps=1e-5,  # 防止除零错误的小值
        bias=True,  # 是否在全连接层中使用偏置
        norm_type="layer_norm",  # 规范化类型
        out_dim: Optional[int] = None,  # 可选的输出维度
    ):
        super().__init__()  # 调用父类构造函数
        # AdaLN
        self.silu = nn.SiLU()  # 定义 SiLU 激活函数
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)  # 全连接层，将条件嵌入映射到嵌入维度
        # 根据指定的规范类型初始化规范层
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)  # 层规范化
        else:
            raise ValueError(f"unknown norm_type {norm_type}")  # 抛出错误，若规范类型未知
        # 如果指定了输出维度，则创建第二个全连接层
        if out_dim is not None:
            self.linear_2 = nn.Linear(
                embedding_dim,  # 输入维度为嵌入维度
                out_dim,  # 输出维度
                bias=bias,  # 是否使用偏置
            )

    # 前向传播方法，定义如何计算输出
    def forward(
        self,
        x: torch.Tensor,  # 输入张量
        conditioning_embedding: torch.Tensor,  # 条件嵌入张量
    # 返回一个张量，类型为 torch.Tensor
    ) -> torch.Tensor:
        # 将条件嵌入转换回原始数据类型，以防止其被提升为 float32（用于 hunyuanDiT）
        emb = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        # 将嵌入值赋给 scale
        scale = emb
        # 对输入 x 进行规范化，并乘以（1 + scale），同时在新维度上扩展
        x = self.norm(x) * (1 + scale)[:, None, :]
    
        # 如果 linear_2 存在，则对 x 应用 linear_2
        if self.linear_2 is not None:
            x = self.linear_2(x)
    
        # 返回处理后的张量 x
        return x
# 定义一个自定义的层，继承自 nn.Module
class CogVideoXLayerNormZero(nn.Module):
    # 初始化方法，定义该层的参数
    def __init__(
        self,
        conditioning_dim: int,  # 输入的条件维度
        embedding_dim: int,  # 嵌入的维度
        elementwise_affine: bool = True,  # 是否启用逐元素仿射变换
        eps: float = 1e-5,  # 防止除零的一个小常数
        bias: bool = True,  # 是否添加偏置
    ) -> None:
        # 调用父类的初始化方法
        super().__init__()

        # 使用 SiLU 激活函数
        self.silu = nn.SiLU()
        # 线性变换，将条件维度映射到 6 倍的嵌入维度
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        # 归一化层，使用层归一化
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    # 前向传播方法，定义输入和输出
    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 通过线性层处理 temb，并分成 6 个部分
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        # 对隐藏状态进行归一化并应用缩放和平移
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        # 对编码器隐藏状态进行相同处理
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        # 返回处理后的隐藏状态和编码器隐藏状态，以及门控信号
        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]


# 根据 PyTorch 版本决定是否使用标准 LayerNorm
if is_torch_version(">=", "2.1.0"):
    # 使用标准的 LayerNorm
    LayerNorm = nn.LayerNorm
else:
    # 定义自定义的 LayerNorm 类，兼容旧版本 PyTorch
    # Has optional bias parameter compared to torch layer norm
    # TODO: replace with torch layernorm once min required torch version >= 2.1
    class LayerNorm(nn.Module):
        # 初始化方法
        def __init__(self, dim, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True):
            # 调用父类的初始化方法
            super().__init__()

            # 设置小常数以避免除零
            self.eps = eps

            # 如果维度是整数，则转为元组
            if isinstance(dim, numbers.Integral):
                dim = (dim,)

            # 保存维度信息
            self.dim = torch.Size(dim)

            # 如果启用逐元素仿射，则初始化权重和偏置
            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
            else:
                self.weight = None
                self.bias = None

        # 前向传播方法
        def forward(self, input):
            # 应用层归一化
            return F.layer_norm(input, self.dim, self.weight, self.bias, self.eps)


# 定义 RMSNorm 类，继承自 nn.Module
class RMSNorm(nn.Module):
    # 初始化方法
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        # 调用父类的初始化方法
        super().__init__()

        # 设置小常数以避免除零
        self.eps = eps

        # 如果维度是整数，则转为元组
        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        # 保存维度信息
        self.dim = torch.Size(dim)

        # 如果启用逐元素仿射，则初始化权重
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    # 前向传播方法
    def forward(self, hidden_states):
        # 保存输入数据类型
        input_dtype = hidden_states.dtype
        # 计算输入的方差
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 对隐藏状态进行缩放
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # 如果有权重，则进行进一步处理
        if self.weight is not None:
            # 如果需要，将隐藏状态转换为半精度
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            # 应用权重
            hidden_states = hidden_states * self.weight
        else:
            # 将隐藏状态转换回原数据类型
            hidden_states = hidden_states.to(input_dtype)

        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个全局响应归一化的类，继承自 nn.Module
class GlobalResponseNorm(nn.Module):
    # 初始化方法，接受一个维度参数 dim
    def __init__(self, dim):
        # 调用父类构造函数
        super().__init__()
        # 初始化可学习参数 gamma，形状为 (1, 1, 1, dim)
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        # 初始化可学习参数 beta，形状为 (1, 1, 1, dim)
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    # 定义前向传播方法，接受输入 x
    def forward(self, x):
        # 计算输入 x 在 (1, 2) 维度上的 L2 范数，保持维度
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        # 归一化 gx，计算每个样本的均值并防止除以零
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        # 返回归一化后的结果，加上可学习的 gamma 和 beta
        return self.gamma * (x * nx) + self.beta + x
```