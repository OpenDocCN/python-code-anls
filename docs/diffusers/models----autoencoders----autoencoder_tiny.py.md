# `.\diffusers\models\autoencoders\autoencoder_tiny.py`

```py
# 版权所有 2024 Ollin Boer Bohan 和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，按照许可证分发的软件是在“按原样”基础上分发的，
# 不提供任何种类的保证或条件，无论是明示还是暗示。
# 请参阅许可证以获取有关权限和限制的具体信息。


# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入可选类型、元组和联合类型
from typing import Optional, Tuple, Union

# 导入 PyTorch 库
import torch

# 从配置相关的模块导入 ConfigMixin 和 register_to_config
from ...configuration_utils import ConfigMixin, register_to_config
# 从工具模块导入 BaseOutput
from ...utils import BaseOutput
# 从加速工具模块导入 apply_forward_hook 函数
from ...utils.accelerate_utils import apply_forward_hook
# 从模型工具模块导入 ModelMixin
from ..modeling_utils import ModelMixin
# 从 VAE 模块导入 DecoderOutput、DecoderTiny 和 EncoderTiny
from .vae import DecoderOutput, DecoderTiny, EncoderTiny


@dataclass
class AutoencoderTinyOutput(BaseOutput):
    """
    AutoencoderTiny 编码方法的输出。

    参数：
        latents (`torch.Tensor`): `Encoder` 的编码输出。
    """

    # 定义编码输出的张量属性
    latents: torch.Tensor


class AutoencoderTiny(ModelMixin, ConfigMixin):
    r"""
    一个小型的蒸馏变分自编码器（VAE）模型，用于将图像编码为潜在表示并将潜在表示解码为图像。

    [`AutoencoderTiny`] 是对 `TAESD` 原始实现的封装。

    此模型继承自 [`ModelMixin`]。有关其为所有模型实现的通用方法的文档，请查看超类文档
    （例如下载或保存）。

    """

    # 指示该模型支持梯度检查点
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        # 定义输入通道的默认值，默认为3（RGB图像）
        self,
        in_channels: int = 3,
        # 定义输出通道的默认值，默认为3（RGB图像）
        out_channels: int = 3,
        # 定义编码器块输出通道的元组，默认为四个64
        encoder_block_out_channels: Tuple[int, ...] = (64, 64, 64, 64),
        # 定义解码器块输出通道的元组，默认为四个64
        decoder_block_out_channels: Tuple[int, ...] = (64, 64, 64, 64),
        # 定义激活函数的默认值，默认为 ReLU
        act_fn: str = "relu",
        # 定义上采样函数的默认值，默认为最近邻插值
        upsample_fn: str = "nearest",
        # 定义潜在通道的默认值
        latent_channels: int = 4,
        # 定义上采样缩放因子的默认值
        upsampling_scaling_factor: int = 2,
        # 定义编码器块数量的元组，默认为 (1, 3, 3, 3)
        num_encoder_blocks: Tuple[int, ...] = (1, 3, 3, 3),
        # 定义解码器块数量的元组，默认为 (3, 3, 3, 1)
        num_decoder_blocks: Tuple[int, ...] = (3, 3, 3, 1),
        # 定义潜在幅度的默认值
        latent_magnitude: int = 3,
        # 定义潜在偏移量的默认值
        latent_shift: float = 0.5,
        # 定义是否强制上溯的布尔值，默认为 False
        force_upcast: bool = False,
        # 定义缩放因子的默认值
        scaling_factor: float = 1.0,
        # 定义偏移因子的默认值
        shift_factor: float = 0.0,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 检查编码器块输出通道数量与编码器块数量是否一致
        if len(encoder_block_out_channels) != len(num_encoder_blocks):
            # 抛出异常，提示编码器块输出通道数量与编码器块数量不匹配
            raise ValueError("`encoder_block_out_channels` should have the same length as `num_encoder_blocks`.")
        # 检查解码器块输出通道数量与解码器块数量是否一致
        if len(decoder_block_out_channels) != len(num_decoder_blocks):
            # 抛出异常，提示解码器块输出通道数量与解码器块数量不匹配
            raise ValueError("`decoder_block_out_channels` should have the same length as `num_decoder_blocks`.")

        # 创建编码器实例
        self.encoder = EncoderTiny(
            # 输入通道数
            in_channels=in_channels,
            # 潜在通道数
            out_channels=latent_channels,
            # 编码块数量
            num_blocks=num_encoder_blocks,
            # 编码块输出通道
            block_out_channels=encoder_block_out_channels,
            # 激活函数
            act_fn=act_fn,
        )

        # 创建解码器实例
        self.decoder = DecoderTiny(
            # 输入潜在通道数
            in_channels=latent_channels,
            # 输出通道数
            out_channels=out_channels,
            # 解码块数量
            num_blocks=num_decoder_blocks,
            # 解码块输出通道
            block_out_channels=decoder_block_out_channels,
            # 上采样缩放因子
            upsampling_scaling_factor=upsampling_scaling_factor,
            # 激活函数
            act_fn=act_fn,
            # 上采样函数
            upsample_fn=upsample_fn,
        )

        # 潜在幅度
        self.latent_magnitude = latent_magnitude
        # 潜在偏移量
        self.latent_shift = latent_shift
        # 缩放因子
        self.scaling_factor = scaling_factor

        # 切片使用标志
        self.use_slicing = False
        # 瓦片使用标志
        self.use_tiling = False

        # 仅在启用 VAE 瓦片时相关
        # 空间缩放因子
        self.spatial_scale_factor = 2**out_channels
        # 瓦片重叠因子
        self.tile_overlap_factor = 0.125
        # 瓦片样本最小大小
        self.tile_sample_min_size = 512
        # 瓦片潜在最小大小
        self.tile_latent_min_size = self.tile_sample_min_size // self.spatial_scale_factor

        # 注册解码器块输出通道到配置
        self.register_to_config(block_out_channels=decoder_block_out_channels)
        # 注册强制上升到配置
        self.register_to_config(force_upcast=False)

    # 设置梯度检查点的方法
    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        # 如果模块是 EncoderTiny 或 DecoderTiny 类型
        if isinstance(module, (EncoderTiny, DecoderTiny)):
            # 设置梯度检查点标志
            module.gradient_checkpointing = value

    # 缩放潜在变量的方法
    def scale_latents(self, x: torch.Tensor) -> torch.Tensor:
        """raw latents -> [0, 1]"""
        # 将潜在变量缩放到 [0, 1] 范围
        return x.div(2 * self.latent_magnitude).add(self.latent_shift).clamp(0, 1)

    # 反缩放潜在变量的方法
    def unscale_latents(self, x: torch.Tensor) -> torch.Tensor:
        """[0, 1] -> raw latents"""
        # 将 [0, 1] 范围的潜在变量反缩放回原始值
        return x.sub(self.latent_shift).mul(2 * self.latent_magnitude)

    # 启用切片解码的方法
    def enable_slicing(self) -> None:
        r"""
        启用切片 VAE 解码。当启用此选项时，VAE 将输入张量切片以
        分几步计算解码。这有助于节省一些内存并允许更大的批量大小。
        """
        # 设置切片使用标志为真
        self.use_slicing = True

    # 禁用切片解码的方法
    def disable_slicing(self) -> None:
        r"""
        禁用切片 VAE 解码。如果之前启用了 `enable_slicing`，此方法将回到
        一步计算解码。
        """
        # 设置切片使用标志为假
        self.use_slicing = False
    # 启用分块 VAE 解码的函数，默认为启用
    def enable_tiling(self, use_tiling: bool = True) -> None:
        r"""
        启用分块 VAE 解码。启用后，VAE 将把输入张量拆分成多个块，以
        分步计算解码和编码。这有助于节省大量内存，并允许处理更大的图像。
        """
        # 将实例变量设置为传入的布尔值以启用或禁用分块
        self.use_tiling = use_tiling

    # 禁用分块 VAE 解码的函数
    def disable_tiling(self) -> None:
        r"""
        禁用分块 VAE 解码。如果之前启用了 `enable_tiling`，则该方法将
        返回到一次计算解码的方式。
        """
        # 调用 enable_tiling 方法以禁用分块
        self.enable_tiling(False)

    # 使用分块编码器对图像批次进行编码的私有方法
    def _tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""使用分块编码器编码图像批次。

        启用该选项后，VAE 将把输入张量拆分成多个块，以分步计算编码。
        这有助于保持内存使用量在固定范围内，不受图像大小影响。为了避免
        块之间的伪影，块之间会重叠并进行混合，以形成平滑的输出。

        Args:
            x (`torch.Tensor`): 输入的图像批次。

        Returns:
            `torch.Tensor`: 编码后的图像批次。
        """
        # 编码器输出相对于输入的缩放比例
        sf = self.spatial_scale_factor
        # 分块的最小样本尺寸
        tile_size = self.tile_sample_min_size

        # 计算混合和遍历之间的像素数量
        blend_size = int(tile_size * self.tile_overlap_factor)
        traverse_size = tile_size - blend_size

        # 计算分块的索引（上/左）
        ti = range(0, x.shape[-2], traverse_size)
        tj = range(0, x.shape[-1], traverse_size)

        # 创建用于混合的掩码
        blend_masks = torch.stack(
            torch.meshgrid([torch.arange(tile_size / sf) / (blend_size / sf - 1)] * 2, indexing="ij")
        )
        # 将掩码限制在 0 到 1 之间并转移到相应的设备
        blend_masks = blend_masks.clamp(0, 1).to(x.device)

        # 初始化输出数组
        out = torch.zeros(x.shape[0], 4, x.shape[-2] // sf, x.shape[-1] // sf, device=x.device)
        # 遍历分块索引
        for i in ti:
            for j in tj:
                # 获取当前分块的输入张量
                tile_in = x[..., i : i + tile_size, j : j + tile_size]
                # 获取当前分块的输出位置
                tile_out = out[..., i // sf : (i + tile_size) // sf, j // sf : (j + tile_size) // sf]
                # 对输入分块进行编码
                tile = self.encoder(tile_in)
                h, w = tile.shape[-2], tile.shape[-1]
                # 将当前块的结果与输出进行混合
                blend_mask_i = torch.ones_like(blend_masks[0]) if i == 0 else blend_masks[0]
                blend_mask_j = torch.ones_like(blend_masks[1]) if j == 0 else blend_masks[1]
                # 计算总混合掩码
                blend_mask = blend_mask_i * blend_mask_j
                # 将块和混合掩码裁剪到一致的形状
                tile, blend_mask = tile[..., :h, :w], blend_mask[..., :h, :w]
                # 更新输出数组中的当前块
                tile_out.copy_(blend_mask * tile + (1 - blend_mask) * tile_out)
        # 返回最终的编码输出
        return out
    # 定义一个私有方法，用于对一批图像进行分块解码
    def _tiled_decode(self, x: torch.Tensor) -> torch.Tensor:
        r"""使用分块编码器对图像批进行编码。

        启用此选项时，VAE 将把输入张量分割成多个块，以分步计算编码。这有助于保持内存使用在
        常数范围内，无论图像大小如何。为了避免分块伪影，块之间重叠并融合在一起形成平滑输出。

        参数：
            x (`torch.Tensor`): 输入图像批。

        返回：
            `torch.Tensor`: 编码后的图像批。
        """
        # 解码器输出相对于输入的缩放因子
        sf = self.spatial_scale_factor
        # 定义每个块的最小大小
        tile_size = self.tile_latent_min_size

        # 计算用于混合和在块之间遍历的像素数量
        blend_size = int(tile_size * self.tile_overlap_factor)
        # 计算块之间的遍历大小
        traverse_size = tile_size - blend_size

        # 创建块的索引（上/左）
        ti = range(0, x.shape[-2], traverse_size)
        tj = range(0, x.shape[-1], traverse_size)

        # 创建混合掩码
        blend_masks = torch.stack(
            torch.meshgrid([torch.arange(tile_size * sf) / (blend_size * sf - 1)] * 2, indexing="ij")
        )
        # 将混合掩码限制在0到1之间，并移动到输入张量所在的设备
        blend_masks = blend_masks.clamp(0, 1).to(x.device)

        # 创建输出数组，初始化为零
        out = torch.zeros(x.shape[0], 3, x.shape[-2] * sf, x.shape[-1] * sf, device=x.device)
        # 遍历每个块的索引
        for i in ti:
            for j in tj:
                # 获取当前块的输入数据
                tile_in = x[..., i : i + tile_size, j : j + tile_size]
                # 获取当前块的输出位置
                tile_out = out[..., i * sf : (i + tile_size) * sf, j * sf : (j + tile_size) * sf]
                # 使用解码器对当前块进行解码
                tile = self.decoder(tile_in)
                h, w = tile.shape[-2], tile.shape[-1]
                # 将当前块的结果混合到输出中
                blend_mask_i = torch.ones_like(blend_masks[0]) if i == 0 else blend_masks[0]
                blend_mask_j = torch.ones_like(blend_masks[1]) if j == 0 else blend_masks[1]
                # 计算最终的混合掩码
                blend_mask = (blend_mask_i * blend_mask_j)[..., :h, :w]
                # 将解码后的块与输出进行混合
                tile_out.copy_(blend_mask * tile + (1 - blend_mask) * tile_out)
        # 返回最终的输出结果
        return out

    # 使用装饰器应用前向钩子，定义编码方法
    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderTinyOutput, Tuple[torch.Tensor]]:
        # 如果使用切片且输入批量大于1
        if self.use_slicing and x.shape[0] > 1:
            # 对每个切片进行编码，使用分块编码或普通编码
            output = [
                self._tiled_encode(x_slice) if self.use_tiling else self.encoder(x_slice) for x_slice in x.split(1)
            ]
            # 将输出合并成一个张量
            output = torch.cat(output)
        else:
            # 对整个输入进行编码，使用分块编码或普通编码
            output = self._tiled_encode(x) if self.use_tiling else self.encoder(x)

        # 如果不返回字典格式，返回输出张量的元组
        if not return_dict:
            return (output,)

        # 返回包含编码结果的自定义输出对象
        return AutoencoderTinyOutput(latents=output)

    # 使用装饰器应用前向钩子，定义解码方法
    @apply_forward_hook
    def decode(
        self, x: torch.Tensor, generator: Optional[torch.Generator] = None, return_dict: bool = True
    # 函数返回类型为 DecoderOutput 或者元组，处理解码后的输出
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        # 如果使用切片并且输入的第一维大于1
        if self.use_slicing and x.shape[0] > 1:
            # 根据是否使用平铺解码，将切片解码结果存入列表中
            output = [self._tiled_decode(x_slice) if self.use_tiling else self.decoder(x) for x_slice in x.split(1)]
            # 将列表中的张量沿着第0维连接成一个张量
            output = torch.cat(output)
        else:
            # 直接对输入进行解码，根据是否使用平铺解码
            output = self._tiled_decode(x) if self.use_tiling else self.decoder(x)
    
        # 如果不需要返回字典格式
        if not return_dict:
            # 返回解码结果作为元组
            return (output,)
    
        # 返回解码结果作为 DecoderOutput 对象
        return DecoderOutput(sample=output)
    
    # forward 方法定义，处理输入样本并返回解码输出
    def forward(
        self,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        r"""
        Args:
            sample (`torch.Tensor`): 输入样本。
            return_dict (`bool`, *optional*, defaults to `True`):
                是否返回一个 [`DecoderOutput`] 对象而不是普通元组。
        """
        # 对输入样本进行编码，提取潜在表示
        enc = self.encode(sample).latents
    
        # 将潜在表示缩放到 [0, 1] 范围，并量化为字节张量，
        # 类似于将潜在表示存储为 RGBA uint8 图像
        scaled_enc = self.scale_latents(enc).mul_(255).round_().byte()
    
        # 将量化后的潜在表示反量化回 [0, 1] 范围，并恢复到原始范围，
        # 类似于从 RGBA uint8 图像中加载潜在表示
        unscaled_enc = self.unscale_latents(scaled_enc / 255.0)
    
        # 对反量化后的潜在表示进行解码
        dec = self.decode(unscaled_enc)
    
        # 如果不需要返回字典格式
        if not return_dict:
            # 返回解码结果作为元组
            return (dec,)
        # 返回解码结果作为 DecoderOutput 对象
        return DecoderOutput(sample=dec)
```