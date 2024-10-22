# `.\diffusers\pipelines\free_noise_utils.py`

```py
# 版权所有 2024 HuggingFace 团队，保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）授权；
# 除非遵守许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议另有约定，
# 否则根据许可证分发的软件按“原样”提供，
# 不提供任何形式的保证或条件，无论是明示或暗示的。
# 有关许可证下的特定语言及其权限和限制，请参阅许可证。

from typing import Optional, Union  # 导入可选和联合类型，用于类型注解

import torch  # 导入 PyTorch 库，以便进行深度学习操作

from ..models.attention import BasicTransformerBlock, FreeNoiseTransformerBlock  # 从上级目录导入注意力模型的基础和自由噪声变换器块
from ..models.unets.unet_motion_model import (  # 从上级目录导入 UNet 动作模型的相关模块
    CrossAttnDownBlockMotion,  # 导入交叉注意力下采样块
    DownBlockMotion,  # 导入下采样块
    UpBlockMotion,  # 导入上采样块
)
from ..utils import logging  # 从上级目录导入日志工具
from ..utils.torch_utils import randn_tensor  # 从上级目录导入生成随机张量的工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，便于记录日志信息

class AnimateDiffFreeNoiseMixin:  # 定义一个混合类，用于实现自由噪声相关功能
    r"""混合类用于 [FreeNoise](https://arxiv.org/abs/2310.15169) 的实现。"""  # 文档字符串，提供该类的用途说明
    # 启用变换器块中的 FreeNoise 功能的辅助函数
    def _enable_free_noise_in_block(self, block: Union[CrossAttnDownBlockMotion, DownBlockMotion, UpBlockMotion]):
        # 文档字符串，说明该函数的目的
    
        for motion_module in block.motion_modules:
            # 遍历每个运动模块
            num_transformer_blocks = len(motion_module.transformer_blocks)
            # 获取当前运动模块中变换器块的数量
    
            for i in range(num_transformer_blocks):
                # 遍历每个变换器块
                if isinstance(motion_module.transformer_blocks[i], FreeNoiseTransformerBlock):
                    # 检查当前块是否为 FreeNoise 变换器块
                    motion_module.transformer_blocks[i].set_free_noise_properties(
                        # 设置 FreeNoise 属性
                        self._free_noise_context_length,
                        self._free_noise_context_stride,
                        self._free_noise_weighting_scheme,
                    )
                else:
                    # 确保当前块是基本变换器块
                    assert isinstance(motion_module.transformer_blocks[i], BasicTransformerBlock)
                    basic_transfomer_block = motion_module.transformer_blocks[i]
                    # 获取基本变换器块的引用
    
                    motion_module.transformer_blocks[i] = FreeNoiseTransformerBlock(
                        # 创建新的 FreeNoise 变换器块，复制基本块的参数
                        dim=basic_transfomer_block.dim,
                        num_attention_heads=basic_transfomer_block.num_attention_heads,
                        attention_head_dim=basic_transfomer_block.attention_head_dim,
                        dropout=basic_transfomer_block.dropout,
                        cross_attention_dim=basic_transfomer_block.cross_attention_dim,
                        activation_fn=basic_transfomer_block.activation_fn,
                        attention_bias=basic_transfomer_block.attention_bias,
                        only_cross_attention=basic_transfomer_block.only_cross_attention,
                        double_self_attention=basic_transfomer_block.double_self_attention,
                        positional_embeddings=basic_transfomer_block.positional_embeddings,
                        num_positional_embeddings=basic_transfomer_block.num_positional_embeddings,
                        context_length=self._free_noise_context_length,
                        context_stride=self._free_noise_context_stride,
                        weighting_scheme=self._free_noise_weighting_scheme,
                    ).to(device=self.device, dtype=self.dtype)
                    # 将新创建的块赋值到当前变换器块的位置，并设置设备和数据类型
    
                    motion_module.transformer_blocks[i].load_state_dict(
                        # 加载基本变换器块的状态字典到新的块
                        basic_transfomer_block.state_dict(), strict=True
                    )
    # 定义一个辅助函数，用于禁用变换器块中的 FreeNoise
    def _disable_free_noise_in_block(self, block: Union[CrossAttnDownBlockMotion, DownBlockMotion, UpBlockMotion]):
        r"""辅助函数，用于禁用变换器块中的 FreeNoise。"""
    
        # 遍历给定块中的所有运动模块
        for motion_module in block.motion_modules:
            # 计算当前运动模块中的变换器块数量
            num_transformer_blocks = len(motion_module.transformer_blocks)
    
            # 遍历每个变换器块
            for i in range(num_transformer_blocks):
                # 检查当前变换器块是否为 FreeNoiseTransformerBlock 类型
                if isinstance(motion_module.transformer_blocks[i], FreeNoiseTransformerBlock):
                    # 获取当前的 FreeNoise 变换器块
                    free_noise_transfomer_block = motion_module.transformer_blocks[i]
    
                    # 用 BasicTransformerBlock 替换 FreeNoise 变换器块，保持相应参数
                    motion_module.transformer_blocks[i] = BasicTransformerBlock(
                        dim=free_noise_transfomer_block.dim,
                        num_attention_heads=free_noise_transfomer_block.num_attention_heads,
                        attention_head_dim=free_noise_transfomer_block.attention_head_dim,
                        dropout=free_noise_transfomer_block.dropout,
                        cross_attention_dim=free_noise_transfomer_block.cross_attention_dim,
                        activation_fn=free_noise_transfomer_block.activation_fn,
                        attention_bias=free_noise_transfomer_block.attention_bias,
                        only_cross_attention=free_noise_transfomer_block.only_cross_attention,
                        double_self_attention=free_noise_transfomer_block.double_self_attention,
                        positional_embeddings=free_noise_transfomer_block.positional_embeddings,
                        num_positional_embeddings=free_noise_transfomer_block.num_positional_embeddings,
                    ).to(device=self.device, dtype=self.dtype)
    
                    # 加载 FreeNoise 变换器块的状态字典到新的 BasicTransformerBlock
                    motion_module.transformer_blocks[i].load_state_dict(
                        free_noise_transfomer_block.state_dict(), strict=True
                    )
    
    # 定义准备自由噪声潜在变量的函数
    def _prepare_latents_free_noise(
        self,
        batch_size: int,
        num_channels_latents: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        # 此处省略函数的具体实现
    
    # 定义启用自由噪声的函数
    def enable_free_noise(
        self,
        context_length: Optional[int] = 16,
        context_stride: int = 4,
        weighting_scheme: str = "pyramid",
        noise_type: str = "shuffle_context",
    ):
        # 此处省略函数的具体实现
    ) -> None:
        r"""
        # 启用使用 FreeNoise 生成长视频的功能

        Args:
            context_length (`int`, defaults to `16`, *optional*):
                # 一次处理的视频帧数量。推荐设置为运动适配器训练时的最大帧数（通常为 16/24/32）。如果为 `None`，将使用运动适配器配置中的默认值。
            context_stride (`int`, *optional*):
                # 通过处理多个帧生成长视频。FreeNoise 使用滑动窗口处理这些帧，窗口大小为 `context_length`。上下文步幅允许指定每个窗口之间跳过多少帧。例如，`context_length` 为 16，`context_stride` 为 4 时将处理 24 帧为：[0, 15], [4, 19], [8, 23]（基于 0 的索引）。
            weighting_scheme (`str`, defaults to `pyramid`):
                # 加权方案，用于在 FreeNoise 块中累积后平均潜在变量。目前支持以下加权方案：
                    - "pyramid"
                        # 使用类似金字塔的权重模式进行加权平均：[1, 2, 3, 2, 1]。
            noise_type (`str`, defaults to "shuffle_context"):
                # TODO
        """

        # 定义允许的加权方案列表
        allowed_weighting_scheme = ["pyramid"]
        # 定义允许的噪声类型列表
        allowed_noise_type = ["shuffle_context", "repeat_context", "random"]

        # 检查上下文长度是否超过最大序列长度
        if context_length > self.motion_adapter.config.motion_max_seq_length:
            logger.warning(
                # 记录警告，表示上下文长度设置过大可能导致生成结果不佳
                f"You have set {context_length=} which is greater than {self.motion_adapter.config.motion_max_seq_length=}. This can lead to bad generation results."
            )
        # 验证加权方案是否在允许的选项中
        if weighting_scheme not in allowed_weighting_scheme:
            raise ValueError(
                # 抛出错误，表示加权方案不合法
                f"The parameter `weighting_scheme` must be one of {allowed_weighting_scheme}, but got {weighting_scheme=}"
            )
        # 验证噪声类型是否在允许的选项中
        if noise_type not in allowed_noise_type:
            raise ValueError(f"The parameter `noise_type` must be one of {allowed_noise_type}, but got {noise_type=}")

        # 设置 FreeNoise 的上下文长度，使用提供值或最大序列长度的默认值
        self._free_noise_context_length = context_length or self.motion_adapter.config.motion_max_seq_length
        # 设置 FreeNoise 的上下文步幅
        self._free_noise_context_stride = context_stride
        # 设置 FreeNoise 的加权方案
        self._free_noise_weighting_scheme = weighting_scheme
        # 设置 FreeNoise 的噪声类型
        self._free_noise_noise_type = noise_type

        # 获取 UNet 的所有块，准备启用 FreeNoise
        blocks = [*self.unet.down_blocks, self.unet.mid_block, *self.unet.up_blocks]
        # 对每个块启用 FreeNoise 功能
        for block in blocks:
            self._enable_free_noise_in_block(block)

    # 禁用 FreeNoise 功能的方法
    def disable_free_noise(self) -> None:
        # 将上下文长度设置为 None，表示禁用
        self._free_noise_context_length = None

        # 获取 UNet 的所有块，准备禁用 FreeNoise
        blocks = [*self.unet.down_blocks, self.unet.mid_block, *self.unet.up_blocks]
        # 对每个块禁用 FreeNoise 功能
        for block in blocks:
            self._disable_free_noise_in_block(block)

    # 属性装饰器
    @property
    # 检查是否启用了自由噪声功能
        def free_noise_enabled(self):
            # 检查对象是否具有属性"_free_noise_context_length"且其值不为None
            return hasattr(self, "_free_noise_context_length") and self._free_noise_context_length is not None
```