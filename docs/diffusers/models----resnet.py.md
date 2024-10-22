# `.\diffusers\models\resnet.py`

```py
# 版权声明，指定版权持有者及保留所有权利
# `TemporalConvLayer` 的版权，指定相关团队及保留所有权利
#
# 根据 Apache License 2.0 版本授权使用本文件；
# 除非遵守许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件按“原样”提供，不提供任何形式的担保或条件。
# 查看许可证以获取特定权限和限制的信息。

# 从 functools 模块导入 partial 函数，用于部分应用
from functools import partial
# 导入类型提示的 Optional、Tuple 和 Union
from typing import Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 导入 PyTorch 的功能性操作模块
import torch.nn.functional as F

# 从工具模块导入 deprecate 装饰器
from ..utils import deprecate
# 从激活函数模块导入获取激活函数的工具
from .activations import get_activation
# 从注意力处理模块导入空间归一化类
from .attention_processor import SpatialNorm
# 从下采样模块导入相关的下采样类和函数
from .downsampling import (  # noqa
    Downsample1D,  # 一维下采样类
    Downsample2D,  # 二维下采样类
    FirDownsample2D,  # FIR 二维下采样类
    KDownsample2D,  # K 下采样类
    downsample_2d,  # 二维下采样函数
)
# 从归一化模块导入自适应组归一化类
from .normalization import AdaGroupNorm
# 从上采样模块导入相关的上采样类和函数
from .upsampling import (  # noqa
    FirUpsample2D,  # FIR 二维上采样类
    KUpsample2D,  # K 上采样类
    Upsample1D,  # 一维上采样类
    Upsample2D,  # 二维上采样类
    upfirdn2d_native,  # 原生的二维上采样函数
    upsample_2d,  # 二维上采样函数
)

# 定义一个使用条件归一化的 ResNet 块类，继承自 nn.Module
class ResnetBlockCondNorm2D(nn.Module):
    r"""
    使用包含条件信息的归一化层的 Resnet 块。
    # 参数说明
        Parameters:
            # 输入通道的数量
            in_channels (`int`): The number of channels in the input.
            # 第一层 conv2d 的输出通道数量，默认为 None 表示与输入通道相同
            out_channels (`int`, *optional*, default to be `None`):
                The number of output channels for the first conv2d layer. If None, same as `in_channels`.
            # 使用的 dropout 概率，默认为 0.0
            dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
            # 时间步嵌入的通道数量，默认为 512
            temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
            # 第一层归一化使用的组数量，默认为 32
            groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
            # 第二层归一化使用的组数量，默认为 None 表示与 groups 相同
            groups_out (`int`, *optional*, default to None):
                The number of groups to use for the second normalization layer. if set to None, same as `groups`.
            # 归一化使用的 epsilon 值，默认为 1e-6
            eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
            # 使用的激活函数类型，默认为 "swish"
            non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
            # 时间嵌入的归一化层，当前只支持 "ada_group" 或 "spatial"
            time_embedding_norm (`str`, *optional*, default to `"ada_group"` ):
                The normalization layer for time embedding `temb`. Currently only support "ada_group" or "spatial".
            # FIR 滤波器，见相关文档
            kernel (`torch.Tensor`, optional, default to None): FIR filter, see
                [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
            # 输出的缩放因子，默认为 1.0
            output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
            # 如果为 True，则添加 1x1 的 nn.conv2d 层作为跳跃连接
            use_in_shortcut (`bool`, *optional*, default to `True`):
                If `True`, add a 1x1 nn.conv2d layer for skip-connection.
            # 如果为 True，则添加上采样层
            up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
            # 如果为 True，则添加下采样层
            down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
            # 如果为 True，则为 `conv_shortcut` 输出添加可学习的偏置
            conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
                `conv_shortcut` output.
            # 输出的通道数量，默认为 None 表示与输出通道相同
            conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
                If None, same as `out_channels`.
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels
        # 如果没有指定输出通道数，则设置为输入通道数
        out_channels = in_channels if out_channels is None else out_channels
        # 保存输出通道数
        self.out_channels = out_channels
        # 保存卷积快捷方式的使用状态
        self.use_conv_shortcut = conv_shortcut
        # 保存上采样的标志
        self.up = up
        # 保存下采样的标志
        self.down = down
        # 保存输出缩放因子
        self.output_scale_factor = output_scale_factor
        # 保存时间嵌入的归一化方式
        self.time_embedding_norm = time_embedding_norm

        # 如果没有指定输出组数，则设置为输入组数
        if groups_out is None:
            groups_out = groups

        # 根据时间嵌入归一化方式选择不同的归一化层
        if self.time_embedding_norm == "ada_group":  # ada_group
            # 使用 AdaGroupNorm 进行归一化
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        elif self.time_embedding_norm == "spatial":
            # 使用 SpatialNorm 进行归一化
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            # 如果归一化方式不支持，抛出错误
            raise ValueError(f" unsupported time_embedding_norm: {self.time_embedding_norm}")

        # 创建第一层卷积，输入通道数为 in_channels，输出通道数为 out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # 根据时间嵌入归一化方式选择第二个归一化层
        if self.time_embedding_norm == "ada_group":  # ada_group
            # 使用 AdaGroupNorm 进行归一化
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        elif self.time_embedding_norm == "spatial":  # spatial
            # 使用 SpatialNorm 进行归一化
            self.norm2 = SpatialNorm(out_channels, temb_channels)
        else:
            # 如果归一化方式不支持，抛出错误
            raise ValueError(f" unsupported time_embedding_norm: {self.time_embedding_norm}")

        # 创建 dropout 层以防止过拟合
        self.dropout = torch.nn.Dropout(dropout)

        # 如果没有指定 2D 卷积的输出通道数，则设置为输出通道数
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        # 创建第二层卷积，输入通道数为 out_channels，输出通道数为 conv_2d_out_channels
        self.conv2 = nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        # 获取激活函数
        self.nonlinearity = get_activation(non_linearity)

        # 初始化上采样和下采样的变量
        self.upsample = self.downsample = None
        # 如果需要上采样，则创建上采样层
        if self.up:
            self.upsample = Upsample2D(in_channels, use_conv=False)
        # 如果需要下采样，则创建下采样层
        elif self.down:
            self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        # 判断是否使用输入快捷方式，默认根据通道数决定
        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        # 初始化卷积快捷方式
        self.conv_shortcut = None
        # 如果使用输入快捷方式，则创建对应的卷积层
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )
    # 定义前向传播方法，接收输入张量和时间嵌入，返回输出张量
        def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            # 检查是否有额外的位置参数或关键字参数中的 scale
            if len(args) > 0 or kwargs.get("scale", None) is not None:
                # 设置弃用消息，提醒用户 scale 参数已弃用并将来会引发错误
                deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
                # 调用弃用函数，记录 scale 的弃用情况
                deprecate("scale", "1.0.0", deprecation_message)
    
            # 将输入张量赋值给隐藏状态
            hidden_states = input_tensor
    
            # 对隐藏状态进行归一化处理，使用时间嵌入
            hidden_states = self.norm1(hidden_states, temb)
    
            # 应用非线性激活函数
            hidden_states = self.nonlinearity(hidden_states)
    
            # 检查是否存在上采样操作
            if self.upsample is not None:
                # 如果批次大小大于等于 64，确保输入张量和隐藏状态是连续的
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                # 对输入张量和隐藏状态进行上采样
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
    
            # 检查是否存在下采样操作
            elif self.downsample is not None:
                # 对输入张量和隐藏状态进行下采样
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)
    
            # 对隐藏状态进行卷积操作
            hidden_states = self.conv1(hidden_states)
    
            # 再次对隐藏状态进行归一化处理，使用时间嵌入
            hidden_states = self.norm2(hidden_states, temb)
    
            # 应用非线性激活函数
            hidden_states = self.nonlinearity(hidden_states)
    
            # 应用 dropout 操作，防止过拟合
            hidden_states = self.dropout(hidden_states)
            # 再次对隐藏状态进行卷积操作
            hidden_states = self.conv2(hidden_states)
    
            # 如果存在 shortcut 卷积，则对输入张量进行 shortcut 卷积
            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)
    
            # 将输入张量和隐藏状态相加，并按输出缩放因子进行缩放
            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
    
            # 返回输出张量
            return output_tensor
# 定义一个名为 ResnetBlock2D 的类，继承自 nn.Module
class ResnetBlock2D(nn.Module):
    r"""
    一个 Resnet 块的文档字符串。

    参数：
        in_channels (`int`): 输入的通道数。
        out_channels (`int`, *可选*, 默认为 `None`):
            第一个 conv2d 层的输出通道数。如果为 None，则与 `in_channels` 相同。
        dropout (`float`, *可选*, 默认为 `0.0`): 使用的 dropout 概率。
        temb_channels (`int`, *可选*, 默认为 `512`): 时间步嵌入的通道数。
        groups (`int`, *可选*, 默认为 `32`): 第一个归一化层使用的组数。
        groups_out (`int`, *可选*, 默认为 None):
            第二个归一化层使用的组数。如果设为 None，则与 `groups` 相同。
        eps (`float`, *可选*, 默认为 `1e-6`): 用于归一化的 epsilon。
        non_linearity (`str`, *可选*, 默认为 `"swish"`): 使用的激活函数。
        time_embedding_norm (`str`, *可选*, 默认为 `"default"` ): 时间缩放平移配置。
            默认情况下，通过简单的平移机制应用时间步嵌入条件。选择 "scale_shift" 以获得
            更强的条件作用，包含缩放和平移。
        kernel (`torch.Tensor`, *可选*, 默认为 None): FIR 滤波器，见
            [`~models.resnet.FirUpsample2D`] 和 [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *可选*, 默认为 `1.0`): 输出使用的缩放因子。
        use_in_shortcut (`bool`, *可选*, 默认为 `True`):
            如果为 `True`，为跳跃连接添加 1x1 的 nn.conv2d 层。
        up (`bool`, *可选*, 默认为 `False`): 如果为 `True`，添加一个上采样层。
        down (`bool`, *可选*, 默认为 `False`): 如果为 `True`，添加一个下采样层。
        conv_shortcut_bias (`bool`, *可选*, 默认为 `True`): 如果为 `True`，为
            `conv_shortcut` 输出添加可学习的偏置。
        conv_2d_out_channels (`int`, *可选*, 默认为 `None`): 输出的通道数。
            如果为 None，则与 `out_channels` 相同。
    """

    # 定义初始化方法，接收各参数以设置 Resnet 块的属性
    def __init__(
        self,
        *,
        in_channels: int,  # 输入通道数
        out_channels: Optional[int] = None,  # 输出通道数，默认为 None
        conv_shortcut: bool = False,  # 是否使用卷积快捷连接
        dropout: float = 0.0,  # dropout 概率
        temb_channels: int = 512,  # 时间步嵌入的通道数
        groups: int = 32,  # 归一化层的组数
        groups_out: Optional[int] = None,  # 第二个归一化层的组数
        pre_norm: bool = True,  # 是否在激活之前进行归一化
        eps: float = 1e-6,  # 归一化使用的 epsilon
        non_linearity: str = "swish",  # 使用的激活函数类型
        skip_time_act: bool = False,  # 是否跳过时间激活
        time_embedding_norm: str = "default",  # 时间嵌入的归一化方式
        kernel: Optional[torch.Tensor] = None,  # FIR 滤波器
        output_scale_factor: float = 1.0,  # 输出缩放因子
        use_in_shortcut: Optional[bool] = None,  # 是否在快捷连接中使用
        up: bool = False,  # 是否添加上采样层
        down: bool = False,  # 是否添加下采样层
        conv_shortcut_bias: bool = True,  # 是否添加可学习的偏置
        conv_2d_out_channels: Optional[int] = None,  # 输出通道数
    # 前向传播方法，接受输入张量和时间嵌入，返回输出张量
        def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            # 检查是否有额外参数或已弃用的 scale 参数
            if len(args) > 0 or kwargs.get("scale", None) is not None:
                # 生成弃用信息提示
                deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
                # 调用弃用函数记录弃用信息
                deprecate("scale", "1.0.0", deprecation_message)
    
            # 将输入张量赋值给隐藏状态
            hidden_states = input_tensor
    
            # 对隐藏状态进行规范化
            hidden_states = self.norm1(hidden_states)
            # 应用非线性激活函数
            hidden_states = self.nonlinearity(hidden_states)
    
            # 如果存在上采样层
            if self.upsample is not None:
                # 当批量大小较大时，确保张量连续存储
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                # 对输入和隐藏状态进行上采样
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            # 如果存在下采样层
            elif self.downsample is not None:
                # 对输入和隐藏状态进行下采样
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)
    
            # 对隐藏状态进行卷积操作
            hidden_states = self.conv1(hidden_states)
    
            # 如果存在时间嵌入投影层
            if self.time_emb_proj is not None:
                # 如果不跳过时间激活
                if not self.skip_time_act:
                    # 对时间嵌入应用非线性激活
                    temb = self.nonlinearity(temb)
                # 进行时间嵌入投影，并增加维度
                temb = self.time_emb_proj(temb)[:, :, None, None]
    
            # 根据时间嵌入的规范化方式处理隐藏状态
            if self.time_embedding_norm == "default":
                if temb is not None:
                    # 将时间嵌入加到隐藏状态上
                    hidden_states = hidden_states + temb
                # 对隐藏状态进行第二次规范化
                hidden_states = self.norm2(hidden_states)
            elif self.time_embedding_norm == "scale_shift":
                # 如果时间嵌入为 None，抛出错误
                if temb is None:
                    raise ValueError(
                        f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                    )
                # 将时间嵌入分割为缩放和偏移
                time_scale, time_shift = torch.chunk(temb, 2, dim=1)
                # 对隐藏状态进行第二次规范化
                hidden_states = self.norm2(hidden_states)
                # 应用缩放和偏移
                hidden_states = hidden_states * (1 + time_scale) + time_shift
            else:
                # 直接对隐藏状态进行第二次规范化
                hidden_states = self.norm2(hidden_states)
    
            # 应用非线性激活函数
            hidden_states = self.nonlinearity(hidden_states)
    
            # 应用 dropout 以增加正则化
            hidden_states = self.dropout(hidden_states)
            # 对隐藏状态进行第二次卷积操作
            hidden_states = self.conv2(hidden_states)
    
            # 如果存在卷积快捷连接
            if self.conv_shortcut is not None:
                # 对输入进行快捷卷积
                input_tensor = self.conv_shortcut(input_tensor)
    
            # 计算输出张量，结合输入和隐藏状态，并按输出缩放因子归一化
            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
    
            # 返回输出张量
            return output_tensor
# unet_rl.py
# 定义一个函数，用于重新排列张量的维度
def rearrange_dims(tensor: torch.Tensor) -> torch.Tensor:
    # 如果张量的维度是 2，则在最后添加一个新维度
    if len(tensor.shape) == 2:
        return tensor[:, :, None]
    # 如果张量的维度是 3，则在第二维后添加一个新维度
    if len(tensor.shape) == 3:
        return tensor[:, :, None, :]
    # 如果张量的维度是 4，则取出第三维的第一个元素
    elif len(tensor.shape) == 4:
        return tensor[:, :, 0, :]
    # 如果维度不在 2, 3 或 4 之间，则抛出错误
    else:
        raise ValueError(f"`len(tensor)`: {len(tensor)} has to be 2, 3 or 4.")


# unet_rl.py
# 定义一个卷积块类，包含 1D 卷积、分组归一化和激活函数
class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish

    Parameters:
        inp_channels (`int`): 输入通道数。
        out_channels (`int`): 输出通道数。
        kernel_size (`int` or `tuple`): 卷积核的大小。
        n_groups (`int`, default `8`): 将通道分成的组数。
        activation (`str`, defaults to `mish`): 激活函数的名称。
    """

    # 初始化函数，定义卷积块的各个层
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        n_groups: int = 8,
        activation: str = "mish",
    ):
        super().__init__()

        # 创建 1D 卷积层，设置填充以保持输出尺寸
        self.conv1d = nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2)
        # 创建分组归一化层
        self.group_norm = nn.GroupNorm(n_groups, out_channels)
        # 获取指定的激活函数
        self.mish = get_activation(activation)

    # 前向传播函数，定义数据流经网络的方式
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 通过卷积层处理输入
        intermediate_repr = self.conv1d(inputs)
        # 重新排列维度
        intermediate_repr = rearrange_dims(intermediate_repr)
        # 通过分组归一化处理
        intermediate_repr = self.group_norm(intermediate_repr)
        # 再次重新排列维度
        intermediate_repr = rearrange_dims(intermediate_repr)
        # 应用激活函数
        output = self.mish(intermediate_repr)
        # 返回最终输出
        return output


# unet_rl.py
# 定义一个残差时序块类，包含时序卷积
class ResidualTemporalBlock1D(nn.Module):
    """
    Residual 1D block with temporal convolutions.

    Parameters:
        inp_channels (`int`): 输入通道数。
        out_channels (`int`): 输出通道数。
        embed_dim (`int`): 嵌入维度。
        kernel_size (`int` or `tuple`): 卷积核的大小。
        activation (`str`, defaults `mish`): 可以选择合适的激活函数。
    """

    # 初始化函数，定义残差块的各个层
    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        embed_dim: int,
        kernel_size: Union[int, Tuple[int, int]] = 5,
        activation: str = "mish",
    ):
        super().__init__()
        # 创建输入卷积块
        self.conv_in = Conv1dBlock(inp_channels, out_channels, kernel_size)
        # 创建输出卷积块
        self.conv_out = Conv1dBlock(out_channels, out_channels, kernel_size)

        # 获取指定的激活函数
        self.time_emb_act = get_activation(activation)
        # 创建线性层，将嵌入维度映射到输出通道数
        self.time_emb = nn.Linear(embed_dim, out_channels)

        # 创建残差卷积，如果输入通道数不等于输出通道数，则使用 1x1 卷积
        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()
        )
    # 定义前向传播函数，接收输入张量和时间嵌入张量，返回输出张量
    def forward(self, inputs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 参数说明：inputs是输入数据，t是时间嵌入
        """
        Args:
            inputs : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
    
        returns:
            out : [ batch_size x out_channels x horizon ]
        """
        # 对时间嵌入应用激活函数
        t = self.time_emb_act(t)
        # 将时间嵌入进行进一步处理
        t = self.time_emb(t)
        # 将输入经过初始卷积处理并与重排后的时间嵌入相加
        out = self.conv_in(inputs) + rearrange_dims(t)
        # 对合并后的结果进行输出卷积处理
        out = self.conv_out(out)
        # 返回卷积结果与残差卷积的和
        return out + self.residual_conv(inputs)
# 定义一个时间卷积层，适用于视频（图像序列）输入，主要代码来源于指定 GitHub 地址
class TemporalConvLayer(nn.Module):
    """
    时间卷积层，用于视频（图像序列）输入。代码主要复制自：
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016

    参数：
        in_dim (`int`): 输入通道数。
        out_dim (`int`): 输出通道数。
        dropout (`float`, *可选*, 默认值为 `0.0`): 使用的丢弃概率。
    """

    # 初始化方法，设置输入输出维度和其他参数
    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 如果没有提供 out_dim，则将其设为 in_dim
        out_dim = out_dim or in_dim
        # 保存输入和输出通道数
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 卷积层构建
        self.conv1 = nn.Sequential(
            # 对输入通道进行分组归一化
            nn.GroupNorm(norm_num_groups, in_dim),
            # 应用 SiLU 激活函数
            nn.SiLU(),
            # 创建 3D 卷积层
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv2 = nn.Sequential(
            # 对输出通道进行分组归一化
            nn.GroupNorm(norm_num_groups, out_dim),
            # 应用 SiLU 激活函数
            nn.SiLU(),
            # 应用丢弃层
            nn.Dropout(dropout),
            # 创建 3D 卷积层
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv3 = nn.Sequential(
            # 对输出通道进行分组归一化
            nn.GroupNorm(norm_num_groups, out_dim),
            # 应用 SiLU 激活函数
            nn.SiLU(),
            # 应用丢弃层
            nn.Dropout(dropout),
            # 创建 3D 卷积层
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv4 = nn.Sequential(
            # 对输出通道进行分组归一化
            nn.GroupNorm(norm_num_groups, out_dim),
            # 应用 SiLU 激活函数
            nn.SiLU(),
            # 应用丢弃层
            nn.Dropout(dropout),
            # 创建 3D 卷积层
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )

        # 将最后一层的参数归零，使卷积块成为恒等映射
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    # 前向传播方法，定义数据如何通过网络流动
    def forward(self, hidden_states: torch.Tensor, num_frames: int = 1) -> torch.Tensor:
        # 重塑输入的隐藏状态以适应卷积层的要求
        hidden_states = (
            hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
        )

        # 保存输入的恒等映射
        identity = hidden_states
        # 通过第一个卷积层处理
        hidden_states = self.conv1(hidden_states)
        # 通过第二个卷积层处理
        hidden_states = self.conv2(hidden_states)
        # 通过第三个卷积层处理
        hidden_states = self.conv3(hidden_states)
        # 通过第四个卷积层处理
        hidden_states = self.conv4(hidden_states)

        # 将处理后的隐藏状态与恒等映射相加
        hidden_states = identity + hidden_states

        # 重塑输出以便返回
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
        )
        # 返回最终的隐藏状态
        return hidden_states


# 定义一个 Resnet 块
class TemporalResnetBlock(nn.Module):
    r"""
    一个 Resnet 块。
    # 参数文档
    Parameters:
        # 输入的通道数
        in_channels (`int`): The number of channels in the input.
        # 第一层 conv2d 的输出通道数，可选，默认为 None，表示与输入通道相同
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        # 时间步嵌入的通道数，可选，默认为 512
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        # 归一化使用的 epsilon，可选，默认为 1e-6
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
    """

    # 初始化方法
    def __init__(
        self,
        # 输入的通道数
        in_channels: int,
        # 输出的通道数，可选，默认为 None
        out_channels: Optional[int] = None,
        # 时间步嵌入的通道数，默认为 512
        temb_channels: int = 512,
        # 归一化使用的 epsilon，默认为 1e-6
        eps: float = 1e-6,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置输入通道数
        self.in_channels = in_channels
        # 如果输出通道数为 None，则使用输入通道数
        out_channels = in_channels if out_channels is None else out_channels
        # 设置输出通道数
        self.out_channels = out_channels

        # 定义卷积核大小
        kernel_size = (3, 1, 1)
        # 计算填充大小
        padding = [k // 2 for k in kernel_size]

        # 创建第一层的归一化层
        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps, affine=True)
        # 创建第一层的卷积层
        self.conv1 = nn.Conv3d(
            # 输入通道数
            in_channels,
            # 输出通道数
            out_channels,
            # 卷积核大小
            kernel_size=kernel_size,
            # 步幅
            stride=1,
            # 填充大小
            padding=padding,
        )

        # 如果时间步嵌入通道数不为 None，则创建对应的线性层
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            # 否则设置为 None
            self.time_emb_proj = None

        # 创建第二层的归一化层
        self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=eps, affine=True)

        # 创建 Dropout 层，比例为 0.0
        self.dropout = torch.nn.Dropout(0.0)
        # 创建第二层的卷积层
        self.conv2 = nn.Conv3d(
            # 输入通道数
            out_channels,
            # 输出通道数
            out_channels,
            # 卷积核大小
            kernel_size=kernel_size,
            # 步幅
            stride=1,
            # 填充大小
            padding=padding,
        )

        # 获取激活函数，这里使用的是 "silu"
        self.nonlinearity = get_activation("silu")

        # 判断是否需要使用输入的 shortcut
        self.use_in_shortcut = self.in_channels != out_channels

        # 初始化 shortcut 卷积层为 None
        self.conv_shortcut = None
        # 如果需要使用输入的 shortcut，则创建对应的卷积层
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(
                # 输入通道数
                in_channels,
                # 输出通道数
                out_channels,
                # 卷积核大小为 1
                kernel_size=1,
                # 步幅
                stride=1,
                # 填充为 0
                padding=0,
            )

    # 前向传播方法
    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # 将输入张量赋值给隐藏状态
        hidden_states = input_tensor

        # 进行第一次归一化
        hidden_states = self.norm1(hidden_states)
        # 应用非线性激活函数
        hidden_states = self.nonlinearity(hidden_states)
        # 进行第一次卷积操作
        hidden_states = self.conv1(hidden_states)

        # 如果时间步嵌入投影层存在
        if self.time_emb_proj is not None:
            # 应用非线性激活函数
            temb = self.nonlinearity(temb)
            # 通过线性层处理时间嵌入
            temb = self.time_emb_proj(temb)[:, :, :, None, None]
            # 调整维度顺序
            temb = temb.permute(0, 2, 1, 3, 4)
            # 将时间嵌入添加到隐藏状态
            hidden_states = hidden_states + temb

        # 进行第二次归一化
        hidden_states = self.norm2(hidden_states)
        # 应用非线性激活函数
        hidden_states = self.nonlinearity(hidden_states)
        # 应用 Dropout
        hidden_states = self.dropout(hidden_states)
        # 进行第二次卷积操作
        hidden_states = self.conv2(hidden_states)

        # 如果 shortcut 卷积层存在
        if self.conv_shortcut is not None:
            # 对输入张量应用 shortcut 卷积
            input_tensor = self.conv_shortcut(input_tensor)

        # 将输入张量与隐藏状态相加，得到输出张量
        output_tensor = input_tensor + hidden_states

        # 返回输出张量
        return output_tensor
# VideoResBlock
# 定义一个时空残差块的类，继承自 nn.Module
class SpatioTemporalResBlock(nn.Module):
    r"""
    一个时空残差网络块。

    参数：
        in_channels (`int`): 输入通道的数量。
        out_channels (`int`, *可选*, 默认为 `None`):
            第一个 conv2d 层的输出通道数量。如果为 None，和 `in_channels` 相同。
        temb_channels (`int`, *可选*, 默认为 `512`): 时间步嵌入的通道数量。
        eps (`float`, *可选*, 默认为 `1e-6`): 用于空间残差网络的 epsilon。
        temporal_eps (`float`, *可选*, 默认为 `eps`): 用于时间残差网络的 epsilon。
        merge_factor (`float`, *可选*, 默认为 `0.5`): 用于时间混合的合并因子。
        merge_strategy (`str`, *可选*, 默认为 `learned_with_images`):
            用于时间混合的合并策略。
        switch_spatial_to_temporal_mix (`bool`, *可选*, 默认为 `False`):
            如果为 `True`，则切换空间和时间混合。
    """

    # 初始化方法，定义类的属性
    def __init__(
        self,
        in_channels: int,  # 输入通道数量
        out_channels: Optional[int] = None,  # 输出通道数量，可选
        temb_channels: int = 512,  # 时间步嵌入通道数量，默认值为512
        eps: float = 1e-6,  # epsilon的默认值
        temporal_eps: Optional[float] = None,  # 时间残差网络的epsilon，默认为None
        merge_factor: float = 0.5,  # 合并因子的默认值
        merge_strategy="learned_with_images",  # 合并策略的默认值
        switch_spatial_to_temporal_mix: bool = False,  # 切换标志，默认为False
    ):
        # 调用父类初始化方法
        super().__init__()

        # 创建一个空间残差块实例
        self.spatial_res_block = ResnetBlock2D(
            in_channels=in_channels,  # 输入通道数量
            out_channels=out_channels,  # 输出通道数量
            temb_channels=temb_channels,  # 时间步嵌入通道数量
            eps=eps,  # epsilon的值
        )

        # 创建一个时间残差块实例
        self.temporal_res_block = TemporalResnetBlock(
            in_channels=out_channels if out_channels is not None else in_channels,  # 输入通道数量，依据输出通道数量决定
            out_channels=out_channels if out_channels is not None else in_channels,  # 输出通道数量
            temb_channels=temb_channels,  # 时间步嵌入通道数量
            eps=temporal_eps if temporal_eps is not None else eps,  # epsilon的值
        )

        # 创建一个时间混合器实例
        self.time_mixer = AlphaBlender(
            alpha=merge_factor,  # 合并因子
            merge_strategy=merge_strategy,  # 合并策略
            switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,  # 切换标志
        )

    # 前向传播方法，定义如何处理输入数据
    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态的张量输入
        temb: Optional[torch.Tensor] = None,  # 可选的时间步嵌入张量
        image_only_indicator: Optional[torch.Tensor] = None,  # 可选的图像指示张量
    ):
        # 获取图像帧的数量，即最后一个维度的大小
        num_frames = image_only_indicator.shape[-1]
        # 通过空间残差块处理隐藏状态
        hidden_states = self.spatial_res_block(hidden_states, temb)

        # 获取当前隐藏状态的批次大小、通道数、高度和宽度
        batch_frames, channels, height, width = hidden_states.shape
        # 计算每个批次的大小，即总帧数除以每个批次的帧数
        batch_size = batch_frames // num_frames

        # 重新调整隐藏状态的形状并进行维度转换，以便于后续处理
        hidden_states_mix = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )
        # 同样的调整隐藏状态的形状并进行维度转换
        hidden_states = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )

        # 如果时间嵌入不为空，则调整其形状以匹配批次大小和帧数
        if temb is not None:
            temb = temb.reshape(batch_size, num_frames, -1)

        # 通过时间残差块处理隐藏状态
        hidden_states = self.temporal_res_block(hidden_states, temb)
        # 将空间和时间的隐藏状态混合
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # 重新排列维度并调整形状，以恢复到原始的隐藏状态格式
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个名为 AlphaBlender 的类，继承自 nn.Module
class AlphaBlender(nn.Module):
    r"""
    一个模块，用于混合空间和时间特征。

    参数:
        alpha (`float`): 混合因子的初始值。
        merge_strategy (`str`, *可选*, 默认值为 `learned_with_images`):
            用于时间混合的合并策略。
        switch_spatial_to_temporal_mix (`bool`, *可选*, 默认值为 `False`):
            如果为 `True`，则交换空间和时间混合。
    """

    # 定义可用的合并策略列表
    strategies = ["learned", "fixed", "learned_with_images"]

    # 初始化方法，设置参数和合并策略
    def __init__(
        self,
        alpha: float,  # 混合因子的初始值
        merge_strategy: str = "learned_with_images",  # 合并策略的默认值
        switch_spatial_to_temporal_mix: bool = False,  # 是否交换混合方式的标志
    ):
        # 调用父类构造函数
        super().__init__()
        # 保存合并策略
        self.merge_strategy = merge_strategy
        # 保存空间和时间混合的交换标志
        self.switch_spatial_to_temporal_mix = switch_spatial_to_temporal_mix  # 用于 TemporalVAE

        # 检查合并策略是否在可用策略中
        if merge_strategy not in self.strategies:
            raise ValueError(f"merge_strategy needs to be in {self.strategies}")

        # 如果合并策略为 "fixed"，则注册固定混合因子
        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))  # 使用缓冲区注册固定值
        # 如果合并策略为 "learned" 或 "learned_with_images"，则注册可学习的混合因子
        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))  # 使用可学习参数注册
        else:
            # 如果合并策略未知，抛出错误
            raise ValueError(f"Unknown merge strategy {self.merge_strategy}")

    # 获取当前的 alpha 值，基于合并策略和输入
    def get_alpha(self, image_only_indicator: torch.Tensor, ndims: int) -> torch.Tensor:
        # 如果合并策略为 "fixed"，直接使用 mix_factor
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor

        # 如果合并策略为 "learned"，使用 sigmoid 函数处理 mix_factor
        elif self.merge_strategy == "learned":
            alpha = torch.sigmoid(self.mix_factor)

        # 如果合并策略为 "learned_with_images"，根据图像指示器计算 alpha
        elif self.merge_strategy == "learned_with_images":
            # 如果没有提供图像指示器，则抛出错误
            if image_only_indicator is None:
                raise ValueError("Please provide image_only_indicator to use learned_with_images merge strategy")

            # 根据 image_only_indicator 的布尔值选择 alpha 的值
            alpha = torch.where(
                image_only_indicator.bool(),  # 使用布尔索引
                torch.ones(1, 1, device=image_only_indicator.device),  # 图像对应的 alpha 为 1
                torch.sigmoid(self.mix_factor)[..., None],  # 其他情况下使用 sigmoid 处理后的值
            )

            # (batch, channel, frames, height, width)
            if ndims == 5:
                alpha = alpha[:, None, :, None, None]  # 调整维度以适应 5D 输入
            # (batch*frames, height*width, channels)
            elif ndims == 3:
                alpha = alpha.reshape(-1)[:, None, None]  # 重塑为 3D 输入
            else:
                # 如果维度不符合预期，抛出错误
                raise ValueError(f"Unexpected ndims {ndims}. Dimensions should be 3 or 5")

        else:
            # 如果合并策略未实现，抛出错误
            raise NotImplementedError

        # 返回计算得到的 alpha 值
        return alpha

    # 前向传播方法，用于处理输入数据
    def forward(
        self,
        x_spatial: torch.Tensor,  # 空间特征输入
        x_temporal: torch.Tensor,  # 时间特征输入
        image_only_indicator: Optional[torch.Tensor] = None,  # 可选的图像指示器
    # 定义一个函数的返回类型为 torch.Tensor
        ) -> torch.Tensor:
        # 获取 alpha 值，依据图像指示器和空间维度
            alpha = self.get_alpha(image_only_indicator, x_spatial.ndim)
        # 将 alpha 转换为与 x_spatial 相同的数据类型
            alpha = alpha.to(x_spatial.dtype)
    
        # 如果开启空间到时间混合的切换
            if self.switch_spatial_to_temporal_mix:
        # 将 alpha 值取反
                alpha = 1.0 - alpha
    
        # 根据 alpha 值进行空间和时间数据的加权组合
            x = alpha * x_spatial + (1.0 - alpha) * x_temporal
        # 返回合成后的数据
            return x
```