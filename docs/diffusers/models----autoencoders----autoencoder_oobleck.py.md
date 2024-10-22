# `.\diffusers\models\autoencoders\autoencoder_oobleck.py`

```py
# 版权声明，表示该代码属于 HuggingFace 团队，所有权利保留
# 根据 Apache 2.0 许可证进行授权
# 用户在合规的情况下可以使用该文件
# 许可证的获取地址
# 如果没有适用的法律或书面协议，软件是按“现状”提供的
# 免责声明，表示不提供任何形式的保证或条件
import math  # 导入数学库，提供数学函数和常数
from dataclasses import dataclass  # 导入数据类装饰器，用于简化类的定义
from typing import Optional, Tuple, Union  # 导入类型提示的必要工具

import numpy as np  # 导入 NumPy 库，提供数组和矩阵操作
import torch  # 导入 PyTorch 库，提供深度学习框架
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch.nn.utils import weight_norm  # 导入权重归一化工具

from ...configuration_utils import ConfigMixin, register_to_config  # 导入配置相关工具
from ...utils import BaseOutput  # 导入基础输出工具
from ...utils.accelerate_utils import apply_forward_hook  # 导入加速工具
from ...utils.torch_utils import randn_tensor  # 导入随机张量生成工具
from ..modeling_utils import ModelMixin  # 导入模型混合工具


class Snake1d(nn.Module):
    """
    一个 1 维的 Snake 激活函数模块。
    """

    def __init__(self, hidden_dim, logscale=True):  # 初始化方法，接收隐藏维度和对数缩放标志
        super().__init__()  # 调用父类初始化方法
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))  # 定义 alpha 参数，初始为 0
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))  # 定义 beta 参数，初始为 0

        self.alpha.requires_grad = True  # 允许 alpha 参数更新
        self.beta.requires_grad = True  # 允许 beta 参数更新
        self.logscale = logscale  # 设置对数缩放标志

    def forward(self, hidden_states):  # 前向传播方法，接收隐藏状态
        shape = hidden_states.shape  # 获取隐藏状态的形状

        alpha = self.alpha if not self.logscale else torch.exp(self.alpha)  # 计算 alpha 值
        beta = self.beta if not self.logscale else torch.exp(self.beta)  # 计算 beta 值

        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)  # 重塑隐藏状态
        hidden_states = hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)  # 更新隐藏状态
        hidden_states = hidden_states.reshape(shape)  # 恢复隐藏状态形状
        return hidden_states  # 返回更新后的隐藏状态


class OobleckResidualUnit(nn.Module):
    """
    一个由 Snake1d 和带扩张的权重归一化 Conv1d 层组成的残差单元。
    """

    def __init__(self, dimension: int = 16, dilation: int = 1):  # 初始化方法，接收维度和扩张因子
        super().__init__()  # 调用父类初始化方法
        pad = ((7 - 1) * dilation) // 2  # 计算填充大小

        self.snake1 = Snake1d(dimension)  # 创建第一个 Snake1d 实例
        self.conv1 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad))  # 创建第一个卷积层并应用权重归一化
        self.snake2 = Snake1d(dimension)  # 创建第二个 Snake1d 实例
        self.conv2 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=1))  # 创建第二个卷积层并应用权重归一化
    # 定义前向传播函数，接收隐藏状态作为输入
    def forward(self, hidden_state):
        """
        前向传播通过残差单元。
    
        参数:
            hidden_state (`torch.Tensor` 形状为 `(batch_size, channels, time_steps)`):
                输入张量。
    
        返回:
            output_tensor (`torch.Tensor` 形状为 `(batch_size, channels, time_steps)`)
                通过残差单元处理后的输入张量。
        """
        # 将输入隐藏状态赋值给输出张量
        output_tensor = hidden_state
        # 通过第一个卷积层和激活函数处理输出张量
        output_tensor = self.conv1(self.snake1(output_tensor))
        # 通过第二个卷积层和激活函数处理输出张量
        output_tensor = self.conv2(self.snake2(output_tensor))
    
        # 计算填充量，以对齐输出张量和输入张量的时间步长
        padding = (hidden_state.shape[-1] - output_tensor.shape[-1]) // 2
        # 如果需要填充，则对隐藏状态进行切片
        if padding > 0:
            hidden_state = hidden_state[..., padding:-padding]
        # 将处理后的输出张量与切片后的隐藏状态相加，实现残差连接
        output_tensor = hidden_state + output_tensor
        # 返回最终的输出张量
        return output_tensor
# Oobleck编码器块的定义，继承自nn.Module
class OobleckEncoderBlock(nn.Module):
    """Encoder block used in Oobleck encoder."""

    # 初始化函数，定义输入维度、输出维度和步幅
    def __init__(self, input_dim, output_dim, stride: int = 1):
        # 调用父类的初始化函数
        super().__init__()

        # 定义第一个残差单元，膨胀率为1
        self.res_unit1 = OobleckResidualUnit(input_dim, dilation=1)
        # 定义第二个残差单元，膨胀率为3
        self.res_unit2 = OobleckResidualUnit(input_dim, dilation=3)
        # 定义第三个残差单元，膨胀率为9
        self.res_unit3 = OobleckResidualUnit(input_dim, dilation=9)
        # 定义一维蛇形结构
        self.snake1 = Snake1d(input_dim)
        # 定义卷积层，使用权重归一化
        self.conv1 = weight_norm(
            nn.Conv1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))
        )

    # 前向传播函数
    def forward(self, hidden_state):
        # 通过第一个残差单元处理隐状态
        hidden_state = self.res_unit1(hidden_state)
        # 通过第二个残差单元处理隐状态
        hidden_state = self.res_unit2(hidden_state)
        # 通过第三个残差单元和蛇形结构处理隐状态
        hidden_state = self.snake1(self.res_unit3(hidden_state))
        # 通过卷积层处理隐状态
        hidden_state = self.conv1(hidden_state)

        # 返回处理后的隐状态
        return hidden_state


# Oobleck解码器块的定义，继承自nn.Module
class OobleckDecoderBlock(nn.Module):
    """Decoder block used in Oobleck decoder."""

    # 初始化函数，定义输入维度、输出维度和步幅
    def __init__(self, input_dim, output_dim, stride: int = 1):
        # 调用父类的初始化函数
        super().__init__()

        # 定义一维蛇形结构
        self.snake1 = Snake1d(input_dim)
        # 定义转置卷积层，使用权重归一化
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        )
        # 定义第一个残差单元，膨胀率为1
        self.res_unit1 = OobleckResidualUnit(output_dim, dilation=1)
        # 定义第二个残差单元，膨胀率为3
        self.res_unit2 = OobleckResidualUnit(output_dim, dilation=3)
        # 定义第三个残差单元，膨胀率为9
        self.res_unit3 = OobleckResidualUnit(output_dim, dilation=9)

    # 前向传播函数
    def forward(self, hidden_state):
        # 通过蛇形结构处理隐状态
        hidden_state = self.snake1(hidden_state)
        # 通过转置卷积层处理隐状态
        hidden_state = self.conv_t1(hidden_state)
        # 通过第一个残差单元处理隐状态
        hidden_state = self.res_unit1(hidden_state)
        # 通过第二个残差单元处理隐状态
        hidden_state = self.res_unit2(hidden_state)
        # 通过第三个残差单元处理隐状态
        hidden_state = self.res_unit3(hidden_state)

        # 返回处理后的隐状态
        return hidden_state


# Oobleck对角高斯分布的定义
class OobleckDiagonalGaussianDistribution(object):
    # 初始化函数，定义参数和确定性标志
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        # 保存输入的参数
        self.parameters = parameters
        # 将参数分解为均值和尺度
        self.mean, self.scale = parameters.chunk(2, dim=1)
        # 计算标准差，确保为正值
        self.std = nn.functional.softplus(self.scale) + 1e-4
        # 计算方差
        self.var = self.std * self.std
        # 计算对数方差
        self.logvar = torch.log(self.var)
        # 保存确定性标志
        self.deterministic = deterministic

    # 采样函数，生成高斯样本
    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # 确保生成的样本在与参数相同的设备和数据类型上
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        # 根据均值和标准差生成样本
        x = self.mean + self.std * sample
        # 返回生成的样本
        return x
    # 计算 Kullback-Leibler 散度，返回一个张量
        def kl(self, other: "OobleckDiagonalGaussianDistribution" = None) -> torch.Tensor:
            # 如果是确定性分布，返回零的张量
            if self.deterministic:
                return torch.Tensor([0.0])
            else:
                # 如果没有提供其他分布，计算本分布的 KL 散度
                if other is None:
                    return (self.mean * self.mean + self.var - self.logvar - 1.0).sum(1).mean()
                else:
                    # 计算均值差的平方归一化
                    normalized_diff = torch.pow(self.mean - other.mean, 2) / other.var
                    # 计算方差比
                    var_ratio = self.var / other.var
                    # 计算对数方差差值
                    logvar_diff = self.logvar - other.logvar
    
                    # 计算 KL 散度的各个部分
                    kl = normalized_diff + var_ratio + logvar_diff - 1
    
                    # 计算 KL 散度的平均值
                    kl = kl.sum(1).mean()
                    return kl
    
        # 返回分布的众数
        def mode(self) -> torch.Tensor:
            return self.mean
# 定义一个数据类，表示自编码器的输出
@dataclass
class AutoencoderOobleckOutput(BaseOutput):
    """
    AutoencoderOobleck 编码方法的输出。

    Args:
        latent_dist (`OobleckDiagonalGaussianDistribution`):
            表示 `Encoder` 编码输出的均值和标准差，
            `OobleckDiagonalGaussianDistribution` 允许从分布中采样潜在变量。
    """

    latent_dist: "OobleckDiagonalGaussianDistribution"  # noqa: F821


# 定义一个数据类，表示解码器的输出
@dataclass
class OobleckDecoderOutput(BaseOutput):
    r"""
    解码方法的输出。

    Args:
        sample (`torch.Tensor` of shape `(batch_size, audio_channels, sequence_length)`):
            从模型最后一层解码的输出样本。
    """

    sample: torch.Tensor


# 定义 Oobleck 编码器类，继承自 nn.Module
class OobleckEncoder(nn.Module):
    """Oobleck 编码器"""

    # 初始化编码器参数
    def __init__(self, encoder_hidden_size, audio_channels, downsampling_ratios, channel_multiples):
        super().__init__()

        # 设置下采样比例
        strides = downsampling_ratios
        # 为通道倍数添加一个起始值
        channel_multiples = [1] + channel_multiples

        # 创建第一个卷积层
        self.conv1 = weight_norm(nn.Conv1d(audio_channels, encoder_hidden_size, kernel_size=7, padding=3))

        self.block = []
        # 创建编码块，随着下采样通过 `stride` 加倍通道
        for stride_index, stride in enumerate(strides):
            self.block += [
                OobleckEncoderBlock(
                    input_dim=encoder_hidden_size * channel_multiples[stride_index],  # 输入维度
                    output_dim=encoder_hidden_size * channel_multiples[stride_index + 1],  # 输出维度
                    stride=stride,  # 下采样步幅
                )
            ]

        # 将编码块转换为模块列表
        self.block = nn.ModuleList(self.block)
        # 计算模型的最终维度
        d_model = encoder_hidden_size * channel_multiples[-1]
        # 创建 Snake1d 模块
        self.snake1 = Snake1d(d_model)
        # 创建第二个卷积层
        self.conv2 = weight_norm(nn.Conv1d(d_model, encoder_hidden_size, kernel_size=3, padding=1))

    # 定义前向传播方法
    def forward(self, hidden_state):
        # 将输入通过第一个卷积层
        hidden_state = self.conv1(hidden_state)

        # 通过每个编码块进行处理
        for module in self.block:
            hidden_state = module(hidden_state)

        # 通过 Snake1d 模块处理
        hidden_state = self.snake1(hidden_state)
        # 将结果通过第二个卷积层
        hidden_state = self.conv2(hidden_state)

        # 返回最终的隐藏状态
        return hidden_state


# 定义 Oobleck 解码器类，继承自 nn.Module
class OobleckDecoder(nn.Module):
    """Oobleck 解码器"""
    # 初始化方法，设置模型参数
        def __init__(self, channels, input_channels, audio_channels, upsampling_ratios, channel_multiples):
            # 调用父类的初始化方法
            super().__init__()
    
            # 将上采样比例赋值给 strides
            strides = upsampling_ratios
            # 在 channel_multiples 列表前添加 1
            channel_multiples = [1] + channel_multiples
    
            # 添加第一个卷积层
            self.conv1 = weight_norm(nn.Conv1d(input_channels, channels * channel_multiples[-1], kernel_size=7, padding=3))
    
            # 添加上采样 + MRF 块
            block = []
            # 遍历 strides 列表，构建 OobleckDecoderBlock
            for stride_index, stride in enumerate(strides):
                block += [
                    OobleckDecoderBlock(
                        # 设置输入和输出维度
                        input_dim=channels * channel_multiples[len(strides) - stride_index],
                        output_dim=channels * channel_multiples[len(strides) - stride_index - 1],
                        stride=stride,
                    )
                ]
    
            # 将构建的块列表转为 nn.ModuleList
            self.block = nn.ModuleList(block)
            # 设置输出维度
            output_dim = channels
            # 创建 Snake1d 实例
            self.snake1 = Snake1d(output_dim)
            # 添加第二个卷积层，不使用偏置
            self.conv2 = weight_norm(nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False))
    
        # 前向传播方法
        def forward(self, hidden_state):
            # 通过第一个卷积层处理输入
            hidden_state = self.conv1(hidden_state)
    
            # 遍历每个块，依次处理隐藏状态
            for layer in self.block:
                hidden_state = layer(hidden_state)
    
            # 通过 Snake1d 处理隐藏状态
            hidden_state = self.snake1(hidden_state)
            # 通过第二个卷积层处理隐藏状态
            hidden_state = self.conv2(hidden_state)
    
            # 返回最终的隐藏状态
            return hidden_state
# 定义一个自动编码器类，用于将波形编码为潜在表示并解码为波形
class AutoencoderOobleck(ModelMixin, ConfigMixin):
    r"""
    自动编码器，用于将波形编码为潜在向量并将潜在表示解码为波形。首次引入于 Stable Audio。

    此模型继承自 [`ModelMixin`]。请查阅超类文档以获取所有模型的通用方法（例如下载或保存）。

    参数：
        encoder_hidden_size (`int`, *可选*, 默认值为 128):
            编码器的中间表示维度。
        downsampling_ratios (`List[int]`, *可选*, 默认值为 `[2, 4, 4, 8, 8]`):
            编码器中下采样的比率。这些比率在解码器中以反向顺序用于上采样。
        channel_multiples (`List[int]`, *可选*, 默认值为 `[1, 2, 4, 8, 16]`):
            用于确定隐藏层隐藏尺寸的倍数。
        decoder_channels (`int`, *可选*, 默认值为 128):
            解码器的中间表示维度。
        decoder_input_channels (`int`, *可选*, 默认值为 64):
            解码器的输入维度。对应于潜在维度。
        audio_channels (`int`, *可选*, 默认值为 2):
            音频数据中的通道数。1 表示单声道，2 表示立体声。
        sampling_rate (`int`, *可选*, 默认值为 44100):
            音频波形应数字化的采样率，以赫兹（Hz）表示。
    """

    # 指示模型是否支持梯度检查点
    _supports_gradient_checkpointing = False

    # 注册到配置的构造函数
    @register_to_config
    def __init__(
        self,
        encoder_hidden_size=128,
        downsampling_ratios=[2, 4, 4, 8, 8],
        channel_multiples=[1, 2, 4, 8, 16],
        decoder_channels=128,
        decoder_input_channels=64,
        audio_channels=2,
        sampling_rate=44100,
    ):
        # 调用父类的构造函数
        super().__init__()

        # 设置编码器的隐藏层大小
        self.encoder_hidden_size = encoder_hidden_size
        # 设置编码器中的下采样比率
        self.downsampling_ratios = downsampling_ratios
        # 设置解码器的通道数
        self.decoder_channels = decoder_channels
        # 将下采样比率反转以用于解码器的上采样
        self.upsampling_ratios = downsampling_ratios[::-1]
        # 计算跳长，作为下采样比率的乘积
        self.hop_length = int(np.prod(downsampling_ratios))
        # 设置音频采样率
        self.sampling_rate = sampling_rate

        # 创建编码器实例，传入必要参数
        self.encoder = OobleckEncoder(
            encoder_hidden_size=encoder_hidden_size,
            audio_channels=audio_channels,
            downsampling_ratios=downsampling_ratios,
            channel_multiples=channel_multiples,
        )

        # 创建解码器实例，传入必要参数
        self.decoder = OobleckDecoder(
            channels=decoder_channels,
            input_channels=decoder_input_channels,
            audio_channels=audio_channels,
            upsampling_ratios=self.upsampling_ratios,
            channel_multiples=channel_multiples,
        )

        # 设置是否使用切片，初始值为假
        self.use_slicing = False
    # 启用切片 VAE 解码的功能
    def enable_slicing(self):
        r"""
        启用切片 VAE 解码。当启用此选项时，VAE 将会将输入张量分割为多个切片以
        分步计算解码。这对于节省内存和允许更大的批量大小非常有用。
        """
        # 设置标志以启用切片
        self.use_slicing = True
    
    # 禁用切片 VAE 解码的功能
    def disable_slicing(self):
        r"""
        禁用切片 VAE 解码。如果之前启用了 `enable_slicing`，则该方法将恢复为一步
        计算解码。
        """
        # 设置标志以禁用切片
        self.use_slicing = False
    
    @apply_forward_hook
    # 编码函数，将一批图像编码为潜在表示
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderOobleckOutput, Tuple[OobleckDiagonalGaussianDistribution]]:
        """
        将一批图像编码为潜在表示。
    
        参数:
            x (`torch.Tensor`): 输入图像批。
            return_dict (`bool`, *可选*, 默认为 `True`):
                是否返回 [`~models.autoencoder_kl.AutoencoderKLOutput`] 而不是普通元组。
    
        返回:
                编码图像的潜在表示。如果 `return_dict` 为 True，则返回
                [`~models.autoencoder_kl.AutoencoderKLOutput`]，否则返回普通 `tuple`。
        """
        # 检查是否启用切片且输入批量大于 1
        if self.use_slicing and x.shape[0] > 1:
            # 对输入进行切片编码
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            # 将所有编码结果连接成一个张量
            h = torch.cat(encoded_slices)
        else:
            # 对整个输入进行编码
            h = self.encoder(x)
    
        # 创建潜在分布
        posterior = OobleckDiagonalGaussianDistribution(h)
    
        # 检查是否返回字典格式
        if not return_dict:
            return (posterior,)
    
        # 返回潜在表示的输出
        return AutoencoderOobleckOutput(latent_dist=posterior)
    
    # 解码函数，将潜在向量解码为图像
    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[OobleckDecoderOutput, torch.Tensor]:
        # 使用解码器解码潜在向量
        dec = self.decoder(z)
    
        # 检查是否返回字典格式
        if not return_dict:
            return (dec,)
    
        # 返回解码结果的输出
        return OobleckDecoderOutput(sample=dec)
    
    @apply_forward_hook
    # 解码函数，解码一批潜在向量为图像
    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[OobleckDecoderOutput, torch.FloatTensor]:
        """
        解码一批图像。
    
        参数:
            z (`torch.Tensor`): 输入潜在向量批。
            return_dict (`bool`, *可选*, 默认为 `True`):
                是否返回 [`~models.vae.OobleckDecoderOutput`] 而不是普通元组。
    
        返回:
            [`~models.vae.OobleckDecoderOutput`] 或 `tuple`:
                如果 return_dict 为 True，则返回 [`~models.vae.OobleckDecoderOutput`]，否则返回普通 `tuple`。
        """
        # 检查是否启用切片且输入批量大于 1
        if self.use_slicing and z.shape[0] > 1:
            # 对输入进行切片解码
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            # 将所有解码结果连接成一个张量
            decoded = torch.cat(decoded_slices)
        else:
            # 对整个输入进行解码
            decoded = self._decode(z).sample
    
        # 检查是否返回字典格式
        if not return_dict:
            return (decoded,)
    
        # 返回解码结果的输出
        return OobleckDecoderOutput(sample=decoded)
    # 定义一个前向传播的方法，接受样本输入和其他参数
        def forward(
            self,
            sample: torch.Tensor,
            sample_posterior: bool = False,  # 是否从后验分布中采样的标志，默认值为 False
            return_dict: bool = True,  # 是否返回 OobleckDecoderOutput 对象而非普通元组，默认值为 True
            generator: Optional[torch.Generator] = None,  # 可选的随机数生成器
        ) -> Union[OobleckDecoderOutput, torch.Tensor]:  # 返回类型为 OobleckDecoderOutput 或 torch.Tensor
            r"""
            Args:
                sample (`torch.Tensor`): 输入样本。
                sample_posterior (`bool`, *optional*, defaults to `False`):
                    是否从后验分布中采样。
                return_dict (`bool`, *optional*, defaults to `True`):
                    是否返回一个 [`OobleckDecoderOutput`] 而不是普通元组。
            """
            x = sample  # 将输入样本赋值给变量 x
            posterior = self.encode(x).latent_dist  # 对输入样本进行编码，获取潜在分布
            if sample_posterior:  # 如果选择从后验分布中采样
                z = posterior.sample(generator=generator)  # 从潜在分布中采样
            else:
                z = posterior.mode()  # 否则取潜在分布的众数
            dec = self.decode(z).sample  # 解码采样得到的潜在变量 z，并获取样本
    
            if not return_dict:  # 如果不需要返回字典
                return (dec,)  # 返回解码样本作为元组
    
            return OobleckDecoderOutput(sample=dec)  # 返回 OobleckDecoderOutput 对象
```