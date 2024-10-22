# `.\diffusers\models\autoencoders\autoencoder_kl_temporal_decoder.py`

```py
# 版权声明，指明该文件由 HuggingFace 团队创建，版权归其所有
# 
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面协议另有约定，软件
# 在许可证下以“原样”方式分发，不附带任何明示或暗示的担保或条件。
# 有关许可证下的具体权限和
# 限制，请参见许可证。
# 导入所需的类型定义
from typing import Dict, Optional, Tuple, Union

# 导入 PyTorch 和神经网络模块
import torch
import torch.nn as nn

# 从配置和工具模块中导入必要的功能
from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import is_torch_version
from ...utils.accelerate_utils import apply_forward_hook
# 从注意力处理模块中导入相关处理器
from ..attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
# 从模型输出模块中导入 Autoencoder 的输出类型
from ..modeling_outputs import AutoencoderKLOutput
# 从模型工具模块中导入模型的混合功能
from ..modeling_utils import ModelMixin
# 从 3D U-Net 模块中导入解码器块
from ..unets.unet_3d_blocks import MidBlockTemporalDecoder, UpBlockTemporalDecoder
# 从变分自编码器模块中导入解码器输出和相关分布
from .vae import DecoderOutput, DiagonalGaussianDistribution, Encoder


# 定义时间解码器类，继承自 PyTorch 的 nn.Module
class TemporalDecoder(nn.Module):
    # 初始化时间解码器，设置输入输出通道及块参数
    def __init__(
        self,
        in_channels: int = 4,  # 输入通道数，默认值为 4
        out_channels: int = 3,  # 输出通道数，默认值为 3
        block_out_channels: Tuple[int] = (128, 256, 512, 512),  # 每个块的输出通道数，默认值为指定的元组
        layers_per_block: int = 2,  # 每个块的层数，默认值为 2
    ):
        # 初始化父类
        super().__init__()
        # 设置每个块的层数
        self.layers_per_block = layers_per_block

        # 创建输入卷积层，接受 in_channels 通道，输出 block_out_channels[-1] 通道
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)
        # 创建中间块的时间解码器，传入层数、输入通道、输出通道和注意力头维度
        self.mid_block = MidBlockTemporalDecoder(
            num_layers=self.layers_per_block,
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            attention_head_dim=block_out_channels[-1],
        )

        # 创建上采样块的列表
        self.up_blocks = nn.ModuleList([])
        # 反转输出通道列表
        reversed_block_out_channels = list(reversed(block_out_channels))
        # 获取第一个输出通道
        output_channel = reversed_block_out_channels[0]
        # 遍历每个输出通道
        for i in range(len(block_out_channels)):
            # 保存前一个输出通道
            prev_output_channel = output_channel
            # 更新当前输出通道
            output_channel = reversed_block_out_channels[i]

            # 判断是否为最后一个块
            is_final_block = i == len(block_out_channels) - 1
            # 创建上采样块的时间解码器
            up_block = UpBlockTemporalDecoder(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
            )
            # 将上采样块添加到列表中
            self.up_blocks.append(up_block)
            # 更新前一个输出通道
            prev_output_channel = output_channel

        # 创建输出的归一化卷积层
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-6)

        # 创建激活函数层，使用 SiLU 激活函数
        self.conv_act = nn.SiLU()
        # 创建输出卷积层，将输入通道转换为输出通道
        self.conv_out = torch.nn.Conv2d(
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        # 定义卷积输出的核大小
        conv_out_kernel_size = (3, 1, 1)
        # 计算填充
        padding = [int(k // 2) for k in conv_out_kernel_size]
        # 创建 3D 卷积层，进行时间卷积
        self.time_conv_out = torch.nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=conv_out_kernel_size,
            padding=padding,
        )

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False

    def forward(
        # 定义前向传播方法的输入参数
        self,
        sample: torch.Tensor,
        image_only_indicator: torch.Tensor,
        num_frames: int = 1,
    ) -> torch.Tensor:
        r"""`Decoder` 类的前向传播方法。"""

        # 对输入样本进行初始卷积处理
        sample = self.conv_in(sample)

        # 获取上采样块参数的 dtype，用于后续转换
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        
        # 如果处于训练模式并启用了梯度检查点
        if self.training and self.gradient_checkpointing:

            # 创建自定义前向传播函数
            def create_custom_forward(module):
                # 定义自定义前向传播，接受输入并返回模块的输出
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # 如果 PyTorch 版本大于等于 1.11.0
            if is_torch_version(">=", "1.11.0"):
                # 中间处理
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),  # 使用自定义前向传播处理中间块
                    sample,  # 输入样本
                    image_only_indicator,  # 指示符
                    use_reentrant=False,  # 不使用可重入的检查点
                )
                # 转换样本的 dtype
                sample = sample.to(upscale_dtype)

                # 上采样处理
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),  # 使用自定义前向传播处理每个上采样块
                        sample,  # 当前样本
                        image_only_indicator,  # 指示符
                        use_reentrant=False,  # 不使用可重入的检查点
                    )
            else:
                # 中间处理
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),  # 使用自定义前向传播处理中间块
                    sample,  # 输入样本
                    image_only_indicator,  # 指示符
                )
                # 转换样本的 dtype
                sample = sample.to(upscale_dtype)

                # 上采样处理
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),  # 使用自定义前向传播处理每个上采样块
                        sample,  # 当前样本
                        image_only_indicator,  # 指示符
                    )
        else:
            # 如果不在训练模式
            # 中间处理
            sample = self.mid_block(sample, image_only_indicator=image_only_indicator)  # 处理样本
            # 转换样本的 dtype
            sample = sample.to(upscale_dtype)

            # 上采样处理
            for up_block in self.up_blocks:
                sample = up_block(sample, image_only_indicator=image_only_indicator)  # 处理样本

        # 后处理步骤
        sample = self.conv_norm_out(sample)  # 正则化输出样本
        sample = self.conv_act(sample)  # 应用激活函数
        sample = self.conv_out(sample)  # 生成最终输出样本

        # 获取样本的形状信息
        batch_frames, channels, height, width = sample.shape
        # 计算批大小
        batch_size = batch_frames // num_frames
        # 重新排列样本的形状以适应时间维度
        sample = sample[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        # 应用时间卷积层
        sample = self.time_conv_out(sample)

        # 还原样本的维度
        sample = sample.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)

        # 返回最终处理后的样本
        return sample
# 定义一个类 AutoencoderKLTemporalDecoder，继承自 ModelMixin 和 ConfigMixin
class AutoencoderKLTemporalDecoder(ModelMixin, ConfigMixin):
    r"""
    一个具有 KL 损失的 VAE 模型，用于将图像编码为潜在表示，并将潜在表示解码为图像。

    该模型继承自 [`ModelMixin`]。有关所有模型的通用方法（如下载或保存）的详细信息，请查阅超类文档。

    参数：
        in_channels (int, *可选*, 默认为 3): 输入图像的通道数。
        out_channels (int, *可选*, 默认为 3): 输出的通道数。
        down_block_types (`Tuple[str]`, *可选*, 默认为 `("DownEncoderBlock2D",)`):
            下采样块类型的元组。
        block_out_channels (`Tuple[int]`, *可选*, 默认为 `(64,)`):
            块输出通道的元组。
        layers_per_block: (`int`, *可选*, 默认为 1): 每个块的层数。
        latent_channels (`int`, *可选*, 默认为 4): 潜在空间中的通道数。
        sample_size (`int`, *可选*, 默认为 32): 输入样本大小。
        scaling_factor (`float`, *可选*, 默认为 0.18215):
            使用训练集的第一批计算的训练潜在空间的逐分量标准差。用于将潜在空间缩放到单位方差，当训练扩散模型时，潜在变量按公式 `z = z * scaling_factor` 缩放。解码时，潜在变量通过公式 `z = 1 / scaling_factor * z` 缩放回原始比例。有关详细信息，请参见 [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) 论文的第 4.3.2 节和 D.1 节。
        force_upcast (`bool`, *可选*, 默认为 `True`):
            如果启用，将强制 VAE 在 float32 中运行，以适应高图像分辨率管道，例如 SD-XL。VAE 可以进行微调/训练到较低范围，而不会失去太多精度，在这种情况下 `force_upcast` 可以设置为 `False` - 参见: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    # 设置支持梯度检查点的标志为真
    _supports_gradient_checkpointing = True

    # 注册到配置的方法，初始化类的实例
    @register_to_config
    def __init__(
        # 输入通道的数量，默认为 3
        self,
        in_channels: int = 3,
        # 输出通道的数量，默认为 3
        out_channels: int = 3,
        # 下采样块的类型，默认为一个包含 "DownEncoderBlock2D" 的元组
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        # 块输出通道的数量，默认为一个包含 64 的元组
        block_out_channels: Tuple[int] = (64,),
        # 每个块的层数，默认为 1
        layers_per_block: int = 1,
        # 潜在通道的数量，默认为 4
        latent_channels: int = 4,
        # 样本输入大小，默认为 32
        sample_size: int = 32,
        # 缩放因子，默认为 0.18215
        scaling_factor: float = 0.18215,
        # 强制使用 float32 的标志，默认为 True
        force_upcast: float = True,
    ):
        # 调用父类的构造函数进行初始化
        super().__init__()

        # 将初始化参数传递给编码器（Encoder）
        self.encoder = Encoder(
            # 输入通道数
            in_channels=in_channels,
            # 潜在通道数
            out_channels=latent_channels,
            # 下采样块的类型
            down_block_types=down_block_types,
            # 每个块的输出通道数
            block_out_channels=block_out_channels,
            # 每个块的层数
            layers_per_block=layers_per_block,
            # 是否双重潜在变量
            double_z=True,
        )

        # 将初始化参数传递给解码器（Decoder）
        self.decoder = TemporalDecoder(
            # 潜在通道数作为输入
            in_channels=latent_channels,
            # 输出通道数
            out_channels=out_channels,
            # 每个块的输出通道数
            block_out_channels=block_out_channels,
            # 每个块的层数
            layers_per_block=layers_per_block,
        )

        # 创建一个卷积层，用于量化
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)

        # 获取样本大小，支持列表或元组形式
        sample_size = (
            self.config.sample_size[0]  # 如果是列表或元组，取第一个元素
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size  # 否则直接使用样本大小
        )
        # 计算最小的平铺潜在大小
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        # 设置平铺重叠因子
        self.tile_overlap_factor = 0.25

    # 定义一个私有方法，用于设置梯度检查点
    def _set_gradient_checkpointing(self, module, value=False):
        # 如果模块是编码器或解码器，设置其梯度检查点
        if isinstance(module, (Encoder, TemporalDecoder)):
            module.gradient_checkpointing = value

    # 使用@property装饰器定义一个属性
    @property
    # 从 UNet2DConditionModel 的 attn_processors 复制而来
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        返回:
            `dict` 类型的注意力处理器：一个包含模型中所有注意力处理器的字典，按权重名称索引。
        """
        # 初始化一个空字典以递归存储处理器
        processors = {}

        # 定义一个递归函数，用于添加处理器
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 如果模块有获取处理器的方法，则添加到字典中
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            # 遍历模块的所有子模块
            for sub_name, child in module.named_children():
                # 递归调用自身以处理子模块
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors  # 返回处理器字典

        # 遍历当前模块的所有子模块
        for name, module in self.named_children():
            # 调用递归函数添加处理器
            fn_recursive_add_processors(name, module, processors)

        return processors  # 返回所有处理器的字典

    # 从 UNet2DConditionModel 的 set_attn_processor 复制而来
    # 定义一个方法用于设置注意力处理器，接受处理器参数
        def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
            r"""
            设置用于计算注意力的处理器。
    
            参数：
                processor (`dict` 或 `AttentionProcessor`):
                    实例化的处理器类或将被设置为**所有**`Attention`层的处理器类字典。
    
                    如果 `processor` 是字典，则键需要定义相应交叉注意力处理器的路径。强烈建议在设置可训练的注意力处理器时使用。
    
            """
            # 计算当前注意力处理器的数量
            count = len(self.attn_processors.keys())
    
            # 如果传入的是字典且其长度不匹配注意力层的数量，则抛出异常
            if isinstance(processor, dict) and len(processor) != count:
                raise ValueError(
                    f"传入了处理器字典，但处理器数量 {len(processor)} 与注意力层数量 {count} 不匹配。请确保传入 {count} 个处理器类。"
                )
    
            # 定义一个递归函数以设置每个子模块的处理器
            def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
                # 如果模块具有设置处理器的方法，则进行设置
                if hasattr(module, "set_processor"):
                    if not isinstance(processor, dict):
                        module.set_processor(processor)  # 设置单个处理器
                    else:
                        # 从字典中移除并设置对应处理器
                        module.set_processor(processor.pop(f"{name}.processor"))
    
                # 遍历模块的子模块并递归调用
                for sub_name, child in module.named_children():
                    fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
    
            # 遍历当前对象的所有子模块，设置处理器
            for name, module in self.named_children():
                fn_recursive_attn_processor(name, module, processor)
    
        # 定义一个方法用于设置默认的注意力处理器
        def set_default_attn_processor(self):
            """
            禁用自定义注意力处理器并设置默认的注意力实现。
            """
            # 如果所有处理器都是交叉注意力处理器，则创建默认处理器
            if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
                processor = AttnProcessor()  # 创建默认处理器实例
            else:
                # 否则抛出异常，说明当前处理器类型不兼容
                raise ValueError(
                    f"当注意力处理器的类型为 {next(iter(self.attn_processors.values()))} 时，无法调用 `set_default_attn_processor`。"
                )
    
            # 调用设置处理器的方法
            self.set_attn_processor(processor)
    
        # 应用前向钩子，定义编码方法
        @apply_forward_hook
        def encode(
            self, x: torch.Tensor, return_dict: bool = True
    # 定义编码器输出的返回类型，包含两种可能的输出格式
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        将一批图像编码为潜在表示。
    
        参数：
            x (`torch.Tensor`): 输入的图像批次。
            return_dict (`bool`, *可选*, 默认值为 `True`):
                是否返回 [`~models.autoencoders.autoencoder_kl.AutoencoderKLOutput`] 而不是普通元组。
    
        返回：
                编码图像的潜在表示。如果 `return_dict` 为 True，则返回
                [`~models.autoencoders.autoencoder_kl.AutoencoderKLOutput`]，否则返回普通 `tuple`。
        """
        # 使用编码器对输入图像进行编码，得到中间表示
        h = self.encoder(x)
        # 对编码结果进行量化，得到矩（均值和方差）
        moments = self.quant_conv(h)
        # 根据矩生成对角高斯分布的后验
        posterior = DiagonalGaussianDistribution(moments)
    
        # 检查是否需要返回普通元组
        if not return_dict:
            # 返回后验分布
            return (posterior,)
    
        # 返回封装后的潜在分布
        return AutoencoderKLOutput(latent_dist=posterior)
    
    # 应用前向钩子装饰器
    @apply_forward_hook
    def decode(
        # 输入的潜在向量
        z: torch.Tensor,
        # 输入帧的数量
        num_frames: int,
        # 是否返回字典格式的结果
        return_dict: bool = True,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        解码一批图像。
    
        参数：
            z (`torch.Tensor`): 输入的潜在向量批次。
            return_dict (`bool`, *可选*, 默认值为 `True`):
                是否返回 [`~models.vae.DecoderOutput`] 而不是普通元组。
    
        返回：
            [`~models.vae.DecoderOutput`] 或 `tuple`:
                如果 return_dict 为 True，返回 [`~models.vae.DecoderOutput`]，否则返回普通 `tuple`。
        """
        # 计算批次大小
        batch_size = z.shape[0] // num_frames
        # 创建图像指示器，初始为零
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=z.dtype, device=z.device)
        # 解码潜在向量，生成图像
        decoded = self.decoder(z, num_frames=num_frames, image_only_indicator=image_only_indicator)
    
        # 检查是否需要返回普通元组
        if not return_dict:
            # 返回解码结果
            return (decoded,)
    
        # 返回解码后的结果封装
        return DecoderOutput(sample=decoded)
    
    def forward(
        # 输入样本
        sample: torch.Tensor,
        # 是否从后验分布中采样
        sample_posterior: bool = False,
        # 是否返回字典格式的结果
        return_dict: bool = True,
        # 随机生成器
        generator: Optional[torch.Generator] = None,
        # 输入帧的数量
        num_frames: int = 1,
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        参数：
            sample (`torch.Tensor`): 输入样本。
            sample_posterior (`bool`, *可选*, 默认值为 `False`):
                是否从后验分布中采样。
            return_dict (`bool`, *可选*, 默认值为 `True`):
                是否返回 [`DecoderOutput`] 而不是普通元组。
        """
        # 直接将样本赋值给 x
        x = sample
        # 编码样本以获取潜在分布
        posterior = self.encode(x).latent_dist
        # 判断是否需要从后验中采样
        if sample_posterior:
            # 从后验中采样潜在向量
            z = posterior.sample(generator=generator)
        else:
            # 使用后验的模值作为潜在向量
            z = posterior.mode()
    
        # 解码潜在向量以生成图像
        dec = self.decode(z, num_frames=num_frames).sample
    
        # 检查是否需要返回普通元组
        if not return_dict:
            # 返回解码结果
            return (dec,)
    
        # 返回解码结果的封装
        return DecoderOutput(sample=dec)
```