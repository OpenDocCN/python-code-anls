# `.\diffusers\models\autoencoders\consistency_decoder_vae.py`

```py
# 版权所有 2024 The HuggingFace Team. 保留所有权利。
#
# 根据 Apache License 2.0 版（“许可证”）授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，否则根据许可证分发的软件是按“原样”基础分发的，
# 不提供任何形式的保证或条件，无论是明示或暗示的。
# 有关许可证下权限和限制的具体条款，请参见许可证。
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import Dict, Optional, Tuple, Union  # 从 typing 模块导入类型提示

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的函数式神经网络模块
from torch import nn  # 从 PyTorch 导入 nn 模块

from ...configuration_utils import ConfigMixin, register_to_config  # 从配置工具模块导入混合类和注册函数
from ...schedulers import ConsistencyDecoderScheduler  # 从调度器模块导入一致性解码器调度器
from ...utils import BaseOutput  # 从工具模块导入基础输出类
from ...utils.accelerate_utils import apply_forward_hook  # 从加速工具模块导入前向钩子应用函数
from ...utils.torch_utils import randn_tensor  # 从 PyTorch 工具模块导入随机张量函数
from ..attention_processor import (  # 从注意力处理器模块导入所需类
    ADDED_KV_ATTENTION_PROCESSORS,  # 导入添加键值注意力处理器
    CROSS_ATTENTION_PROCESSORS,  # 导入交叉注意力处理器
    AttentionProcessor,  # 导入注意力处理器基类
    AttnAddedKVProcessor,  # 导入添加键值注意力处理器类
    AttnProcessor,  # 导入注意力处理器类
)
from ..modeling_utils import ModelMixin  # 从建模工具模块导入模型混合类
from ..unets.unet_2d import UNet2DModel  # 从 2D U-Net 模块导入 U-Net 模型类
from .vae import DecoderOutput, DiagonalGaussianDistribution, Encoder  # 从 VAE 模块导入解码器输出、对角高斯分布和编码器

@dataclass  # 将该类标记为数据类
class ConsistencyDecoderVAEOutput(BaseOutput):  # 定义一致性解码器 VAE 输出类，继承自基础输出类
    """
    编码方法的输出。

    参数：
        latent_dist (`DiagonalGaussianDistribution`):
            表示为均值和对数方差的编码器输出。
            `DiagonalGaussianDistribution` 允许从分布中采样潜变量。
    """

    latent_dist: "DiagonalGaussianDistribution"  # 定义潜在分布属性，类型为对角高斯分布


class ConsistencyDecoderVAE(ModelMixin, ConfigMixin):  # 定义一致性解码器 VAE 类，继承自模型混合类和配置混合类
    r"""
    与 DALL-E 3 一起使用的一致性解码器。

    示例：
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline, ConsistencyDecoderVAE

        >>> vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16)  # 从预训练模型加载一致性解码器 VAE
        >>> pipe = StableDiffusionPipeline.from_pretrained(  # 从预训练模型加载稳定扩散管道
        ...     "runwayml/stable-diffusion-v1-5", vae=vae, torch_dtype=torch.float16
        ... ).to("cuda")  # 将管道移动到 CUDA 设备

        >>> image = pipe("horse", generator=torch.manual_seed(0)).images[0]  # 生成图像
        >>> image  # 输出生成的图像
        ```py
    """

    @register_to_config  # 将该方法注册到配置中
    # 初始化方法，用于创建类的实例
    def __init__(
            # 缩放因子，默认为 0.18215
            scaling_factor: float = 0.18215,
            # 潜在通道数，默认为 4
            latent_channels: int = 4,
            # 样本尺寸，默认为 32
            sample_size: int = 32,
            # 编码器激活函数，默认为 "silu"
            encoder_act_fn: str = "silu",
            # 编码器输出通道数的元组，默认为 (128, 256, 512, 512)
            encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
            # 编码器是否使用双重 Z，默认为 True
            encoder_double_z: bool = True,
            # 编码器下采样块类型的元组，默认为多个 "DownEncoderBlock2D"
            encoder_down_block_types: Tuple[str, ...] = (
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            # 编码器输入通道数，默认为 3
            encoder_in_channels: int = 3,
            # 每个编码器块的层数，默认为 2
            encoder_layers_per_block: int = 2,
            # 编码器归一化组数，默认为 32
            encoder_norm_num_groups: int = 32,
            # 编码器输出通道数，默认为 4
            encoder_out_channels: int = 4,
            # 解码器是否添加注意力机制，默认为 False
            decoder_add_attention: bool = False,
            # 解码器输出通道数的元组，默认为 (320, 640, 1024, 1024)
            decoder_block_out_channels: Tuple[int, ...] = (320, 640, 1024, 1024),
            # 解码器下采样块类型的元组，默认为多个 "ResnetDownsampleBlock2D"
            decoder_down_block_types: Tuple[str, ...] = (
                "ResnetDownsampleBlock2D",
                "ResnetDownsampleBlock2D",
                "ResnetDownsampleBlock2D",
                "ResnetDownsampleBlock2D",
            ),
            # 解码器下采样填充，默认为 1
            decoder_downsample_padding: int = 1,
            # 解码器输入通道数，默认为 7
            decoder_in_channels: int = 7,
            # 每个解码器块的层数，默认为 3
            decoder_layers_per_block: int = 3,
            # 解码器归一化的 epsilon 值，默认为 1e-05
            decoder_norm_eps: float = 1e-05,
            # 解码器归一化组数，默认为 32
            decoder_norm_num_groups: int = 32,
            # 解码器训练时长的时间步数，默认为 1024
            decoder_num_train_timesteps: int = 1024,
            # 解码器输出通道数，默认为 6
            decoder_out_channels: int = 6,
            # 解码器时间缩放偏移类型，默认为 "scale_shift"
            decoder_resnet_time_scale_shift: str = "scale_shift",
            # 解码器时间嵌入类型，默认为 "learned"
            decoder_time_embedding_type: str = "learned",
            # 解码器上采样块类型的元组，默认为多个 "ResnetUpsampleBlock2D"
            decoder_up_block_types: Tuple[str, ...] = (
                "ResnetUpsampleBlock2D",
                "ResnetUpsampleBlock2D",
                "ResnetUpsampleBlock2D",
                "ResnetUpsampleBlock2D",
            ),
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化编码器，传入各类参数以配置其行为
        self.encoder = Encoder(
            act_fn=encoder_act_fn,  # 激活函数
            block_out_channels=encoder_block_out_channels,  # 编码器每个块的输出通道数
            double_z=encoder_double_z,  # 是否使用双Z向量
            down_block_types=encoder_down_block_types,  # 编码器下采样块的类型
            in_channels=encoder_in_channels,  # 输入通道数
            layers_per_block=encoder_layers_per_block,  # 每个块的层数
            norm_num_groups=encoder_norm_num_groups,  # 归一化的组数
            out_channels=encoder_out_channels,  # 输出通道数
        )

        # 初始化解码器UNet模型，配置其参数
        self.decoder_unet = UNet2DModel(
            add_attention=decoder_add_attention,  # 是否添加注意力机制
            block_out_channels=decoder_block_out_channels,  # 解码器每个块的输出通道数
            down_block_types=decoder_down_block_types,  # 解码器下采样块的类型
            downsample_padding=decoder_downsample_padding,  # 下采样的填充方式
            in_channels=decoder_in_channels,  # 输入通道数
            layers_per_block=decoder_layers_per_block,  # 每个块的层数
            norm_eps=decoder_norm_eps,  # 归一化中的epsilon值
            norm_num_groups=decoder_norm_num_groups,  # 归一化的组数
            num_train_timesteps=decoder_num_train_timesteps,  # 训练时的时间步数
            out_channels=decoder_out_channels,  # 输出通道数
            resnet_time_scale_shift=decoder_resnet_time_scale_shift,  # ResNet时间尺度偏移
            time_embedding_type=decoder_time_embedding_type,  # 时间嵌入类型
            up_block_types=decoder_up_block_types,  # 解码器上采样块的类型
        )
        # 初始化一致性解码器调度器
        self.decoder_scheduler = ConsistencyDecoderScheduler()
        # 注册编码器的输出通道数到配置中
        self.register_to_config(block_out_channels=encoder_block_out_channels)
        # 注册强制上采样的配置
        self.register_to_config(force_upcast=False)
        # 注册均值的缓冲区，形状为(1, C, 1, 1)
        self.register_buffer(
            "means",
            torch.tensor([0.38862467, 0.02253063, 0.07381133, -0.0171294])[None, :, None, None],  # 均值张量
            persistent=False,  # 不持久化保存
        )
        # 注册标准差的缓冲区，形状为(1, C, 1, 1)
        self.register_buffer(
            "stds", torch.tensor([0.9654121, 1.0440036, 0.76147926, 0.77022034])[None, :, None, None], persistent=False  # 标准差张量
        )

        # 初始化量化卷积层，输入和输出通道数相同
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)

        # 设置切片和拼接的使用标志为假
        self.use_slicing = False
        self.use_tiling = False

        # 仅在启用 VAE 切片时相关
        self.tile_sample_min_size = self.config.sample_size  # 最小样本大小
        # 判断样本大小类型，并设置样本大小
        sample_size = (
            self.config.sample_size[0]  # 如果样本大小是列表或元组，取第一个元素
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size  # 否则直接使用样本大小
        )
        # 计算最小的切片潜在大小
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        # 设置切片重叠因子
        self.tile_overlap_factor = 0.25

    # 从 diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL.enable_tiling 复制的方法
    def enable_tiling(self, use_tiling: bool = True):
        r"""
        启用切片 VAE 解码。启用此选项时，VAE 将输入张量拆分为切片，以
        分步骤计算解码和编码。这有助于节省大量内存，并允许处理更大图像。
        """
        # 设置是否使用切片标志
        self.use_tiling = use_tiling

    # 从 diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL.disable_tiling 复制的方法
    # 定义一个方法来禁用平铺的 VAE 解码
    def disable_tiling(self):
        r"""
        禁用平铺的 VAE 解码。如果之前启用了 `enable_tiling`，该方法将恢复为一步计算解码。
        """
        # 调用 enable_tiling 方法并传入 False 参数以禁用平铺
        self.enable_tiling(False)

    # 从 diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL.enable_slicing 复制而来
    # 定义一个方法来启用切片的 VAE 解码
    def enable_slicing(self):
        r"""
        启用切片的 VAE 解码。当启用此选项时，VAE 将把输入张量分成多个切片进行解码计算。
        这对于节省一些内存和允许更大的批处理大小非常有用。
        """
        # 设置 use_slicing 为 True，以启用切片
        self.use_slicing = True

    # 从 diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL.disable_slicing 复制而来
    # 定义一个方法来禁用切片的 VAE 解码
    def disable_slicing(self):
        r"""
        禁用切片的 VAE 解码。如果之前启用了 `enable_slicing`，该方法将恢复为一步计算解码。
        """
        # 设置 use_slicing 为 False，以禁用切片
        self.use_slicing = False

    @property
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors 复制而来
    # 定义一个属性方法，用于返回注意力处理器
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        返回:
            `dict` 类型的注意力处理器：一个包含模型中所有注意力处理器的字典，按权重名称索引。
        """
        # 创建一个空字典用于存储处理器
        processors = {}

        # 定义一个递归函数以添加处理器
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 检查模块是否有 get_processor 方法
            if hasattr(module, "get_processor"):
                # 将处理器添加到字典中，键为模块名称加上 ".processor"
                processors[f"{name}.processor"] = module.get_processor()

            # 遍历模块的子模块
            for sub_name, child in module.named_children():
                # 递归调用该函数以添加子模块的处理器
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            # 返回更新后的处理器字典
            return processors

        # 遍历当前对象的所有子模块
        for name, module in self.named_children():
            # 调用递归函数以添加所有处理器
            fn_recursive_add_processors(name, module, processors)

        # 返回最终的处理器字典
        return processors

    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor 复制而来
    # 设置用于计算注意力的处理器
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的处理器。

        参数：
            processor（`dict` of `AttentionProcessor` 或仅 `AttentionProcessor`）：
                实例化的处理器类或一个处理器类的字典，将作为所有 `Attention` 层的处理器设置。

                如果 `processor` 是一个字典，键需要定义对应的交叉注意力处理器的路径。强烈建议在设置可训练的注意力处理器时使用此方式。

        """
        # 计算当前注意力处理器的数量
        count = len(self.attn_processors.keys())

        # 检查传入的处理器字典长度是否与当前注意力层数量匹配
        if isinstance(processor, dict) and len(processor) != count:
            # 抛出值错误，提示处理器数量不匹配
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        # 定义递归函数，用于设置每个子模块的处理器
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 检查子模块是否有设置处理器的方法
            if hasattr(module, "set_processor"):
                # 如果处理器不是字典，则直接设置
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    # 从字典中弹出对应的处理器并设置
                    module.set_processor(processor.pop(f"{name}.processor"))

            # 遍历子模块，递归调用自身
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        # 遍历当前模块的所有子模块
        for name, module in self.named_children():
            # 调用递归函数设置处理器
            fn_recursive_attn_processor(name, module, processor)

    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor 复制而来
    def set_default_attn_processor(self):
        """
        禁用自定义注意力处理器并设置默认的注意力实现。
        """
        # 检查所有处理器是否属于新增的 KV 注意力处理器
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 设置处理器为新增的 KV 注意力处理器
            processor = AttnAddedKVProcessor()
        # 检查所有处理器是否属于交叉注意力处理器
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 设置处理器为标准的注意力处理器
            processor = AttnProcessor()
        else:
            # 抛出值错误，提示无法设置默认处理器
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        # 调用设置处理器的方法
        self.set_attn_processor(processor)

    # 应用前向钩子修饰器
    @apply_forward_hook
    def encode(
        # 输入张量 x，返回字典的标志
        self, x: torch.Tensor, return_dict: bool = True
    # 定义一个方法，返回编码后的图像的潜在表示
    ) -> Union[ConsistencyDecoderVAEOutput, Tuple[DiagonalGaussianDistribution]]:
            """
            将一批图像编码为潜在表示。
    
            参数：
                x (`torch.Tensor`): 输入图像批次。
                return_dict (`bool`, *可选*, 默认为 `True`):
                    是否返回 [`~models.autoencoders.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] 
                    而不是普通的元组。
    
            返回：
                    编码图像的潜在表示。如果 `return_dict` 为 True，则返回 
                    [`~models.autoencoders.consistency_decoder_vae.ConsistencyDecoderVAEOutput`]，否则返回 
                    普通的 `tuple`。
            """
            # 检查是否使用分块编码，并且图像尺寸超过最小样本大小
            if self.use_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
                # 调用分块编码方法处理输入
                return self.tiled_encode(x, return_dict=return_dict)
    
            # 检查是否使用切片，并且输入批次大于1
            if self.use_slicing and x.shape[0] > 1:
                # 对输入的每个切片进行编码，并将结果收集到列表中
                encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
                # 将所有编码切片连接成一个张量
                h = torch.cat(encoded_slices)
            else:
                # 对整个输入进行编码
                h = self.encoder(x)
    
            # 通过量化卷积获取潜在表示的统计量
            moments = self.quant_conv(h)
            # 创建一个对角高斯分布作为后验分布
            posterior = DiagonalGaussianDistribution(moments)
    
            # 如果不需要返回字典格式
            if not return_dict:
                # 返回包含后验分布的元组
                return (posterior,)
    
            # 返回包含潜在分布的输出对象
            return ConsistencyDecoderVAEOutput(latent_dist=posterior)
    
        # 应用前向钩子装饰器
        @apply_forward_hook
        def decode(
            # 定义解码方法，输入潜在变量
            z: torch.Tensor,
            # 可选的随机数生成器
            generator: Optional[torch.Generator] = None,
            # 是否返回字典格式，默认为 True
            return_dict: bool = True,
            # 推理步骤的数量，默认为 2
            num_inference_steps: int = 2,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        """
        解码输入的潜在向量 `z`，使用一致性解码器 VAE 模型。

        Args:
            z (torch.Tensor): 输入的潜在向量。
            generator (Optional[torch.Generator]): 随机数生成器，默认为 None。
            return_dict (bool): 是否以字典形式返回输出，默认为 True。
            num_inference_steps (int): 推理步骤的数量，默认为 2。

        Returns:
            Union[DecoderOutput, Tuple[torch.Tensor]]: 解码后的输出。

        """
        # 对潜在向量 `z` 进行标准化处理
        z = (z * self.config.scaling_factor - self.means) / self.stds

        # 计算缩放因子，基于输出通道的数量
        scale_factor = 2 ** (len(self.config.block_out_channels) - 1)
        # 将潜在向量 `z` 进行最近邻插值缩放
        z = F.interpolate(z, mode="nearest", scale_factor=scale_factor)

        # 获取当前张量的批量大小、高度和宽度
        batch_size, _, height, width = z.shape

        # 设置解码器调度器的时间步长
        self.decoder_scheduler.set_timesteps(num_inference_steps, device=self.device)

        # 初始化噪声张量 `x_t`，用于后续的解码过程
        x_t = self.decoder_scheduler.init_noise_sigma * randn_tensor(
            (batch_size, 3, height, width), generator=generator, dtype=z.dtype, device=z.device
        )

        # 对每个时间步进行解码
        for t in self.decoder_scheduler.timesteps:
            # 将当前噪声与潜在向量 `z` 组合成模型输入
            model_input = torch.concat([self.decoder_scheduler.scale_model_input(x_t, t), z], dim=1)
            # 获取模型输出的样本
            model_output = self.decoder_unet(model_input, t).sample[:, :3, :, :]
            # 更新当前样本 `x_t`
            prev_sample = self.decoder_scheduler.step(model_output, t, x_t, generator).prev_sample
            x_t = prev_sample

        # 将最后的样本赋值给 `x_0`
        x_0 = x_t

        # 如果不返回字典，直接返回样本
        if not return_dict:
            return (x_0,)

        # 返回解码后的输出，封装成 DecoderOutput 对象
        return DecoderOutput(sample=x_0)

    # 从 diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL.blend_v 复制的函数
    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        # 计算混合范围，确保不超过输入张量的维度
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        # 对于每个混合范围内的高度像素进行加权混合
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        # 返回混合后的张量
        return b

    # 从 diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL.blend_h 复制的函数
    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        # 计算混合范围，确保不超过输入张量的维度
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        # 对于每个混合范围内的宽度像素进行加权混合
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        # 返回混合后的张量
        return b
    # 定义一个使用平铺编码器对图像批次进行编码的方法
    def tiled_encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[ConsistencyDecoderVAEOutput, Tuple]:
        r"""使用平铺编码器编码一批图像。

        当启用此选项时，VAE将输入张量分割成平铺，以进行多个步骤的编码。这有助于保持内存使用量在任何图像大小下都是恒定的。平铺编码的最终结果与非平铺编码不同，因为每个平铺使用不同的编码器。为了避免平铺伪影，平铺之间会重叠并进行混合，以形成平滑的输出。您仍然可能会看到平铺大小的变化，但应该不那么明显。

        参数：
            x (`torch.Tensor`): 输入图像批次。
            return_dict (`bool`, *可选*, 默认为 `True`):
                是否返回 [`~models.autoencoders.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] 
                而不是一个普通的元组。

        返回：
            [`~models.autoencoders.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] 或 `tuple`:
                如果 return_dict 为 True，则返回一个 [`~models.autoencoders.consistency_decoder_vae.ConsistencyDecoderVAEOutput`]
                ，否则返回一个普通的 `tuple`。
        """
        # 计算重叠区域的大小
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        # 计算混合范围的大小
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        # 计算行限制的大小
        row_limit = self.tile_latent_min_size - blend_extent

        # 将图像分割成512x512的平铺并分别编码
        rows = []  # 存储每一行的编码结果
        for i in range(0, x.shape[2], overlap_size):  # 遍历图像的高度
            row = []  # 存储当前行的编码结果
            for j in range(0, x.shape[3], overlap_size):  # 遍历图像的宽度
                # 提取当前平铺
                tile = x[:, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                # 使用编码器对平铺进行编码
                tile = self.encoder(tile)
                # 进行量化处理
                tile = self.quant_conv(tile)
                # 将编码后的平铺添加到当前行
                row.append(tile)
            # 将当前行添加到所有行的列表中
            rows.append(row)
        
        result_rows = []  # 存储最终结果的行
        for i, row in enumerate(rows):  # 遍历每一行的编码结果
            result_row = []  # 存储当前行的最终结果
            for j, tile in enumerate(row):  # 遍历当前行的每个平铺
                # 将上方的平铺与当前平铺进行混合
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                # 将左侧的平铺与当前平铺进行混合
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                # 将处理后的平铺裁剪并添加到结果行
                result_row.append(tile[:, :, :row_limit, :row_limit])
            # 将当前行的结果合并到结果行列表中
            result_rows.append(torch.cat(result_row, dim=3))

        # 将所有结果行在高度维度上进行合并
        moments = torch.cat(result_rows, dim=2)
        # 创建对角高斯分布对象
        posterior = DiagonalGaussianDistribution(moments)

        # 如果不返回字典，则返回一个包含后验分布的元组
        if not return_dict:
            return (posterior,)

        # 返回包含后验分布的ConsistencyDecoderVAEOutput对象
        return ConsistencyDecoderVAEOutput(latent_dist=posterior)
    # 定义前向传播方法，处理输入样本并返回解码结果
    def forward(
            self,
            sample: torch.Tensor,  # 输入样本，类型为 torch.Tensor
            sample_posterior: bool = False,  # 是否从后验分布采样，默认为 False
            return_dict: bool = True,  # 是否返回 DecoderOutput 对象，默认为 True
            generator: Optional[torch.Generator] = None,  # 可选的随机数生成器，用于采样
        ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:  # 返回类型可以是 DecoderOutput 或元组
            r"""
            Args:
                sample (`torch.Tensor`): Input sample.  # 输入样本
                sample_posterior (`bool`, *optional*, defaults to `False`):  # 是否采样后验的标志
                    Whether to sample from the posterior.
                return_dict (`bool`, *optional*, defaults to `True`):  # 返回类型的标志
                    Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
                generator (`torch.Generator`, *optional*, defaults to `None`):  # 随机数生成器的说明
                    Generator to use for sampling.
    
            Returns:
                [`DecoderOutput`] or `tuple`:  # 返回类型的说明
                    If return_dict is True, a [`DecoderOutput`] is returned, otherwise a plain `tuple` is returned.
            """
            x = sample  # 将输入样本赋值给变量 x
            posterior = self.encode(x).latent_dist  # 使用编码器对样本进行编码，获取后验分布
            if sample_posterior:  # 检查是否需要从后验分布采样
                z = posterior.sample(generator=generator)  # 从后验分布中采样，使用指定的生成器
            else:  # 如果不从后验分布采样
                z = posterior.mode()  # 选择后验分布的众数作为 z
            dec = self.decode(z, generator=generator).sample  # 解码 z，并获取解码后的样本
    
            if not return_dict:  # 如果不需要返回字典
                return (dec,)  # 返回解码样本的元组
    
            return DecoderOutput(sample=dec)  # 返回 DecoderOutput 对象，包含解码样本
```