# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\temporal_ae.py`

```py
# 从 typing 模块导入 Callable, Iterable, Union 类型注解
from typing import Callable, Iterable, Union

# 导入 PyTorch 库
import torch
# 从 einops 导入 rearrange 和 repeat 函数，用于张量重排和重复
from einops import rearrange, repeat

# 从自定义模块中导入所需的类和变量
from sgm.modules.diffusionmodules.model import (
    # 检查 XFORMERS 库是否可用
    XFORMERS_IS_AVAILABLE,
    # 导入注意力块、解码器和其他模块
    AttnBlock,
    Decoder,
    MemoryEfficientAttnBlock,
    ResnetBlock,
)
# 从 openaimodel 模块导入 ResBlock 和时间步嵌入函数
from sgm.modules.diffusionmodules.openaimodel import ResBlock, timestep_embedding
# 从 video_attention 模块导入视频变换块
from sgm.modules.video_attention import VideoTransformerBlock
# 从 util 模块导入 partialclass 函数
from sgm.util import partialclass


# 定义一个新的类 VideoResBlock，继承自 ResnetBlock
class VideoResBlock(ResnetBlock):
    # 初始化函数，接收多个参数
    def __init__(
        self,
        out_channels,  # 输出通道数
        *args,  # 额外参数
        dropout=0.0,  # dropout 概率
        video_kernel_size=3,  # 视频卷积核大小
        alpha=0.0,  # 混合因子
        merge_strategy="learned",  # 合并策略
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类构造函数进行初始化
        super().__init__(out_channels=out_channels, dropout=dropout, *args, **kwargs)
        # 如果未指定 video_kernel_size，则默认设置
        if video_kernel_size is None:
            video_kernel_size = [3, 1, 1]
        # 创建时间堆栈，使用 ResBlock
        self.time_stack = ResBlock(
            channels=out_channels,  # 通道数
            emb_channels=0,  # 嵌入通道数
            dropout=dropout,  # dropout 概率
            dims=3,  # 数据维度
            use_scale_shift_norm=False,  # 是否使用缩放平移归一化
            use_conv=False,  # 是否使用卷积
            up=False,  # 是否向上采样
            down=False,  # 是否向下采样
            kernel_size=video_kernel_size,  # 卷积核大小
            use_checkpoint=False,  # 是否使用检查点
            skip_t_emb=True,  # 是否跳过时间嵌入
        )

        # 设置合并策略
        self.merge_strategy = merge_strategy
        # 根据合并策略注册混合因子
        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))  # 固定混合因子
        elif self.merge_strategy == "learned":
            self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))  # 学习混合因子
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")  # 抛出未知合并策略错误

    # 获取 alpha 值的函数
    def get_alpha(self, bs):
        # 根据合并策略返回混合因子
        if self.merge_strategy == "fixed":
            return self.mix_factor  # 固定策略
        elif self.merge_strategy == "learned":
            return torch.sigmoid(self.mix_factor)  # 学习策略
        else:
            raise NotImplementedError()  # 抛出未实现错误

    # 前向传播函数
    def forward(self, x, temb, skip_video=False, timesteps=None):
        # 如果未提供时间步，则使用类中的 timesteps
        if timesteps is None:
            timesteps = self.timesteps

        # 获取输入张量的形状
        b, c, h, w = x.shape

        # 调用父类的前向传播方法
        x = super().forward(x, temb)

        # 如果不跳过视频处理
        if not skip_video:
            # 重排张量，将其调整为视频格式
            x_mix = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

            # 重排当前张量
            x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)

            # 通过时间堆栈进行处理
            x = self.time_stack(x, temb)

            # 获取 alpha 值
            alpha = self.get_alpha(bs=b // timesteps)
            # 按比例混合两个张量
            x = alpha * x + (1.0 - alpha) * x_mix

            # 再次重排张量
            x = rearrange(x, "b c t h w -> (b t) c h w")
        return x  # 返回处理后的张量


# 定义一个新的类 AE3DConv，继承自 torch.nn.Conv2d
class AE3DConv(torch.nn.Conv2d):
    # 初始化方法，设置输入和输出通道及卷积核大小等参数
        def __init__(self, in_channels, out_channels, video_kernel_size=3, *args, **kwargs):
            # 调用父类的初始化方法，传递输入和输出通道及其他参数
            super().__init__(in_channels, out_channels, *args, **kwargs)
            # 检查 video_kernel_size 是否为可迭代对象
            if isinstance(video_kernel_size, Iterable):
                # 如果是可迭代对象，计算每个核的填充大小
                padding = [int(k // 2) for k in video_kernel_size]
            else:
                # 否则，计算单个核的填充大小
                padding = int(video_kernel_size // 2)
    
            # 创建一个 3D 卷积层，用于处理视频数据
            self.time_mix_conv = torch.nn.Conv3d(
                in_channels=out_channels,  # 输入通道数
                out_channels=out_channels,  # 输出通道数
                kernel_size=video_kernel_size,  # 卷积核大小
                padding=padding,  # 填充大小
            )
    
        # 前向传播方法，处理输入数据并返回输出
        def forward(self, input, timesteps, skip_video=False):
            # 调用父类的前向传播方法，处理输入数据
            x = super().forward(input)
            # 如果跳过视频处理，直接返回处理后的数据
            if skip_video:
                return x
            # 调整张量形状，以适应 3D 卷积的输入要求
            x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)
            # 通过 3D 卷积层处理数据
            x = self.time_mix_conv(x)
            # 调整输出张量的形状，返回到原始格式
            return rearrange(x, "b c t h w -> (b t) c h w")
# 定义一个视频块类，继承自注意力块基类
class VideoBlock(AttnBlock):
    # 初始化函数，接收输入通道数、混合因子和合并策略
    def __init__(self, in_channels: int, alpha: float = 0, merge_strategy: str = "learned"):
        # 调用基类初始化函数
        super().__init__(in_channels)
        # 创建视频转换块，使用单头注意力机制
        self.time_mix_block = VideoTransformerBlock(
            dim=in_channels,
            n_heads=1,
            d_head=in_channels,
            checkpoint=False,
            ff_in=True,
            attn_mode="softmax",
        )

        # 计算时间嵌入维度
        time_embed_dim = self.in_channels * 4
        # 构建视频时间嵌入的神经网络结构
        self.video_time_embed = torch.nn.Sequential(
            torch.nn.Linear(self.in_channels, time_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim, self.in_channels),
        )

        # 设置合并策略
        self.merge_strategy = merge_strategy
        # 根据合并策略注册混合因子为缓冲区
        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", torch.Tensor([alpha]))
        # 注册混合因子为可学习参数
        elif self.merge_strategy == "learned":
            self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))
        # 如果合并策略未知，抛出错误
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    # 前向传播函数，接收输入、时间步和跳过视频标志
    def forward(self, x, timesteps, skip_video=False):
        # 如果跳过视频，调用基类的前向传播
        if skip_video:
            return super().forward(x)

        # 保存输入数据
        x_in = x
        # 进行注意力计算
        x = self.attention(x)
        # 获取输出的高度和宽度
        h, w = x.shape[2:]
        # 重新排列数据形状
        x = rearrange(x, "b c h w -> b (h w) c")

        # 初始化混合输入
        x_mix = x
        # 创建时间步的序列
        num_frames = torch.arange(timesteps, device=x.device)
        # 重复时间步以匹配批大小
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        # 重新排列时间步
        num_frames = rearrange(num_frames, "b t -> (b t)")
        # 生成时间嵌入
        t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False)
        # 计算时间嵌入的输出
        emb = self.video_time_embed(t_emb)  # b, n_channels
        # 增加一个维度
        emb = emb[:, None, :]
        # 将时间嵌入与输入混合
        x_mix = x_mix + emb

        # 获取当前的混合因子
        alpha = self.get_alpha()
        # 进行时间混合块计算
        x_mix = self.time_mix_block(x_mix, timesteps=timesteps)
        # 根据混合因子合并输入和混合输出
        x = alpha * x + (1.0 - alpha) * x_mix  # alpha merge

        # 重新排列输出形状
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        # 投影到输出空间
        x = self.proj_out(x)

        # 返回输入与输出的和
        return x_in + x

    # 获取当前的混合因子
    def get_alpha(
        self,
    ):
        # 如果合并策略是固定，返回固定的混合因子
        if self.merge_strategy == "fixed":
            return self.mix_factor
        # 如果合并策略是学习的，返回经过 sigmoid 函数处理的混合因子
        elif self.merge_strategy == "learned":
            return torch.sigmoid(self.mix_factor)
        # 如果合并策略未知，抛出错误
        else:
            raise NotImplementedError(f"unknown merge strategy {self.merge_strategy}")


# 定义一个内存高效的视频块类，继承自内存高效注意力块
class MemoryEfficientVideoBlock(MemoryEfficientAttnBlock):
    # 初始化类，设置输入通道、混合因子和合并策略
        def __init__(self, in_channels: int, alpha: float = 0, merge_strategy: str = "learned"):
            # 调用父类构造函数，传递输入通道数
            super().__init__(in_channels)
            # 创建视频变换块，设置相关参数
            self.time_mix_block = VideoTransformerBlock(
                dim=in_channels,
                n_heads=1,
                d_head=in_channels,
                checkpoint=False,
                ff_in=True,
                attn_mode="softmax-xformers",
            )
    
            # 计算时间嵌入维度
            time_embed_dim = self.in_channels * 4
            # 定义时间嵌入序列，包含两层线性变换和SiLU激活
            self.video_time_embed = torch.nn.Sequential(
                torch.nn.Linear(self.in_channels, time_embed_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(time_embed_dim, self.in_channels),
            )
    
            # 保存合并策略
            self.merge_strategy = merge_strategy
            # 如果合并策略是固定，则注册混合因子为缓冲区
            if self.merge_strategy == "fixed":
                self.register_buffer("mix_factor", torch.Tensor([alpha]))
            # 如果合并策略是学习的，则注册混合因子为可学习参数
            elif self.merge_strategy == "learned":
                self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor([alpha])))
            # 否则抛出错误
            else:
                raise ValueError(f"unknown merge strategy {self.merge_strategy}")
    
        # 前向传播函数
        def forward(self, x, timesteps, skip_time_block=False):
            # 如果跳过时间块，调用父类的前向传播
            if skip_time_block:
                return super().forward(x)
    
            # 保存输入数据
            x_in = x
            # 应用注意力机制
            x = self.attention(x)
            # 获取输出的高度和宽度
            h, w = x.shape[2:]
            # 重排张量以便于处理
            x = rearrange(x, "b c h w -> b (h w) c")
    
            # 初始化混合输入
            x_mix = x
            # 创建时间帧的张量
            num_frames = torch.arange(timesteps, device=x.device)
            # 重复时间帧以匹配批次大小
            num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
            # 重排张量
            num_frames = rearrange(num_frames, "b t -> (b t)")
            # 获取时间嵌入
            t_emb = timestep_embedding(num_frames, self.in_channels, repeat_only=False)
            # 应用视频时间嵌入
            emb = self.video_time_embed(t_emb)  # b, n_channels
            # 在第二维插入新的维度
            emb = emb[:, None, :]
            # 将嵌入加到混合输入
            x_mix = x_mix + emb
    
            # 获取混合因子
            alpha = self.get_alpha()
            # 应用时间混合块
            x_mix = self.time_mix_block(x_mix, timesteps=timesteps)
            # 根据alpha进行混合
            x = alpha * x + (1.0 - alpha) * x_mix  # alpha merge
    
            # 重新排列张量以恢复原始维度
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            # 应用输出投影
            x = self.proj_out(x)
    
            # 返回输入与输出的和
            return x_in + x
    
        # 获取混合因子的函数
        def get_alpha(
            self,
        ):
            # 如果合并策略是固定，则返回混合因子
            if self.merge_strategy == "fixed":
                return self.mix_factor
            # 如果合并策略是学习的，则返回经过sigmoid处理的混合因子
            elif self.merge_strategy == "learned":
                return torch.sigmoid(self.mix_factor)
            # 否则抛出未实现错误
            else:
                raise NotImplementedError(f"unknown merge strategy {self.merge_strategy}")
# 创建时空注意力机制的函数，接受多个参数以配置注意力类型和其他设置
def make_time_attn(
    in_channels,  # 输入通道数
    attn_type="vanilla",  # 注意力类型，默认为'vanilla'
    attn_kwargs=None,  # 额外的注意力参数，默认为None
    alpha: float = 0,  # 参数alpha，默认为0
    merge_strategy: str = "learned",  # 合并策略，默认为'learned'
):
    # 检查注意力类型是否在支持的选项中
    assert attn_type in [
        "vanilla",
        "vanilla-xformers",
    ], f"attn_type {attn_type} not supported for spatio-temporal attention"
    # 打印当前创建的注意力类型及其输入通道数
    print(f"making spatial and temporal attention of type '{attn_type}' with {in_channels} in_channels")
    # 如果不支持xformers，且当前注意力类型为'vanilla-xformers'，则回退到'vanilla'
    if not XFORMERS_IS_AVAILABLE and attn_type == "vanilla-xformers":
        print(
            f"Attention mode '{attn_type}' is not available. Falling back to vanilla attention. "
            f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
        )
        attn_type = "vanilla"

    # 如果注意力类型为'vanilla'，则返回部分类VideoBlock
    if attn_type == "vanilla":
        assert attn_kwargs is None  # 确保没有提供额外的参数
        return partialclass(VideoBlock, in_channels, alpha=alpha, merge_strategy=merge_strategy)
    # 如果注意力类型为'vanilla-xformers'，则返回部分类MemoryEfficientVideoBlock
    elif attn_type == "vanilla-xformers":
        print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
        return partialclass(
            MemoryEfficientVideoBlock,
            in_channels,
            alpha=alpha,
            merge_strategy=merge_strategy,
        )
    else:
        return NotImplementedError()  # 如果不支持的类型，返回未实现错误


# 自定义的卷积层包装器，继承自torch.nn.Conv2d
class Conv2DWrapper(torch.nn.Conv2d):
    # 前向传播方法，调用父类的前向传播
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return super().forward(input)


# 视频解码器类，继承自Decoder
class VideoDecoder(Decoder):
    available_time_modes = ["all", "conv-only", "attn-only"]  # 可用的时间模式列表

    # 初始化方法，设置视频解码器的各种参数
    def __init__(
        self,
        *args,
        video_kernel_size: Union[int, list] = 3,  # 视频卷积核大小，默认为3
        alpha: float = 0.0,  # alpha参数，默认为0.0
        merge_strategy: str = "learned",  # 合并策略，默认为'learned'
        time_mode: str = "conv-only",  # 时间模式，默认为'conv-only'
        **kwargs,
    ):
        self.video_kernel_size = video_kernel_size  # 设置视频卷积核大小
        self.alpha = alpha  # 设置alpha参数
        self.merge_strategy = merge_strategy  # 设置合并策略
        self.time_mode = time_mode  # 设置时间模式
        # 确保时间模式在可用选项内
        assert (
            self.time_mode in self.available_time_modes
        ), f"time_mode parameter has to be in {self.available_time_modes}"
        super().__init__(*args, **kwargs)  # 调用父类的初始化方法

    # 获取最后一层的权重，支持跳过时间混合的选项
    def get_last_layer(self, skip_time_mix=False, **kwargs):
        # 如果时间模式为'attn-only'，则抛出未实现错误
        if self.time_mode == "attn-only":
            raise NotImplementedError("TODO")
        else:
            # 返回适当的权重，基于跳过时间混合的选项
            return self.conv_out.time_mix_conv.weight if not skip_time_mix else self.conv_out.weight

    # 创建注意力机制的方法
    def _make_attn(self) -> Callable:
        # 根据时间模式返回适当的部分类
        if self.time_mode not in ["conv-only", "only-last-conv"]:
            return partialclass(
                make_time_attn,
                alpha=self.alpha,
                merge_strategy=self.merge_strategy,
            )
        else:
            return super()._make_attn()  # 否则调用父类的方法

    # 创建卷积层的方法
    def _make_conv(self) -> Callable:
        # 根据时间模式返回适当的部分类或卷积包装器
        if self.time_mode != "attn-only":
            return partialclass(AE3DConv, video_kernel_size=self.video_kernel_size)
        else:
            return Conv2DWrapper  # 返回卷积包装器
    # 定义一个私有方法，用于创建残差块，返回一个可调用对象
        def _make_resblock(self) -> Callable:
            # 检查当前的时间模式是否不在指定的两种模式中
            if self.time_mode not in ["attn-only", "only-last-conv"]:
                # 返回一个部分应用的类，用于创建 VideoResBlock 实例
                return partialclass(
                    VideoResBlock,
                    video_kernel_size=self.video_kernel_size,
                    alpha=self.alpha,
                    merge_strategy=self.merge_strategy,
                )
            else:
                # 否则，调用父类的方法以创建残差块
                return super()._make_resblock()
```