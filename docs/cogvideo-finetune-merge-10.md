# CogVideo & CogVideoX 微调代码源码解析（十一）



# `.\cogvideo-finetune\sat\sgm\modules\encoders\__init__.py`

```py
请提供需要注释的代码，我将为您添加注释。
```

# `.\cogvideo-finetune\sat\sgm\modules\video_attention.py`

```py
# 导入 PyTorch 库
import torch

# 从上级模块导入必要的组件
from ..modules.attention import *
from ..modules.diffusionmodules.util import AlphaBlender, linear, timestep_embedding

# 定义一个新的类 TimeMixSequential，继承自 nn.Sequential
class TimeMixSequential(nn.Sequential):
    # 重写 forward 方法
    def forward(self, x, context=None, timesteps=None):
        # 遍历所有层，将输入 x 逐层传递
        for layer in self:
            x = layer(x, context, timesteps)

        # 返回最终的输出
        return x

# 定义 VideoTransformerBlock 类，继承自 nn.Module
class VideoTransformerBlock(nn.Module):
    # 定义注意力模式的字典
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # 使用软最大注意力
        "softmax-xformers": MemoryEfficientCrossAttention,  # 使用内存高效的软最大注意力
    }

    # 初始化方法
    def __init__(
        self,
        dim,  # 输入特征的维度
        n_heads,  # 注意力头的数量
        d_head,  # 每个头的维度
        dropout=0.0,  # dropout 的比率
        context_dim=None,  # 上下文特征的维度
        gated_ff=True,  # 是否使用门控前馈网络
        checkpoint=True,  # 是否启用检查点功能
        timesteps=None,  # 时间步
        ff_in=False,  # 是否使用前馈网络输入
        inner_dim=None,  # 内部特征维度
        attn_mode="softmax",  # 注意力模式
        disable_self_attn=False,  # 是否禁用自注意力
        disable_temporal_crossattention=False,  # 是否禁用时间交叉注意力
        switch_temporal_ca_to_sa=False,  # 是否将时间交叉注意力切换为自注意力
    ):
        # 调用父类构造函数
        super().__init__()

        # 根据指定的注意力模式选择注意力类
        attn_cls = self.ATTENTION_MODES[attn_mode]

        # 确定是否使用前馈网络输入
        self.ff_in = ff_in or inner_dim is not None
        # 如果没有指定内部维度，则将其设为输入维度
        if inner_dim is None:
            inner_dim = dim

        # 确保头的数量乘以每个头的维度等于内部维度
        assert int(n_heads * d_head) == inner_dim

        # 确定是否使用残差连接
        self.is_res = inner_dim == dim

        # 如果使用前馈网络输入，定义输入层的归一化和前馈网络
        if self.ff_in:
            self.norm_in = nn.LayerNorm(dim)  # 定义层归一化
            self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff)  # 定义前馈网络

        # 设置时间步数
        self.timesteps = timesteps
        # 设置是否禁用自注意力
        self.disable_self_attn = disable_self_attn
        # 如果禁用自注意力，使用交叉注意力
        if self.disable_self_attn:
            self.attn1 = attn_cls(
                query_dim=inner_dim,  # 查询的维度
                heads=n_heads,  # 注意力头的数量
                dim_head=d_head,  # 每个头的维度
                context_dim=context_dim,  # 上下文维度
                dropout=dropout,  # dropout 的比率
            )  # 这是一个交叉注意力
        else:
            # 否则，使用自注意力
            self.attn1 = attn_cls(
                query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
            )  # 这是一个自注意力

        # 定义前馈网络
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)

        # 根据设置决定是否禁用时间交叉注意力
        if disable_temporal_crossattention:
            # 如果要切换交叉注意力为自注意力，抛出异常
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None  # 不使用时间交叉注意力
        else:
            # 定义内部归一化层
            self.norm2 = nn.LayerNorm(inner_dim)
            # 根据设置选择交叉注意力或自注意力
            if switch_temporal_ca_to_sa:
                self.attn2 = attn_cls(
                    query_dim=inner_dim, heads=n_heads, dim_head=d_head, dropout=dropout
                )  # 这是一个自注意力
            else:
                self.attn2 = attn_cls(
                    query_dim=inner_dim,
                    context_dim=context_dim,  # 上下文维度
                    heads=n_heads,  # 注意力头的数量
                    dim_head=d_head,  # 每个头的维度
                    dropout=dropout,  # dropout 的比率
                )  # 如果上下文为空，则为自注意力

        # 定义层归一化
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        # 保存切换自注意力和交叉注意力的设置
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

        # 保存检查点设置
        self.checkpoint = checkpoint
        # 如果启用检查点，打印信息
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")
    # 定义前向传播方法，接收输入张量 x 和可选的上下文及时间步数
        def forward(self, x: torch.Tensor, context: torch.Tensor = None, timesteps: int = None) -> torch.Tensor:
            # 如果启用检查点，使用检查点机制来执行前向传播
            if self.checkpoint:
                return checkpoint(self._forward, x, context, timesteps)
            # 否则，直接调用内部前向传播方法
            else:
                return self._forward(x, context, timesteps=timesteps)
    
    # 定义内部前向传播方法，处理输入张量及可选的上下文和时间步数
        def _forward(self, x, context=None, timesteps=None):
            # 确保有时间步数，如果未传入则使用对象属性
            assert self.timesteps or timesteps
            # 确保时间步数一致性，若两者都提供则必须相等
            assert not (self.timesteps and timesteps) or self.timesteps == timesteps
            # 设置时间步数
            timesteps = self.timesteps or timesteps
            # 获取输入张量的批次、序列长度和通道数
            B, S, C = x.shape
            # 调整输入张量的形状以适应后续计算
            x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)
    
            # 如果启用前馈输入，则进行前馈层处理
            if self.ff_in:
                x_skip = x  # 保存输入以便残差连接
                x = self.ff_in(self.norm_in(x))  # 归一化后通过前馈层
                # 如果使用残差连接，则将输入加回
                if self.is_res:
                    x += x_skip
    
            # 如果禁用自注意力，则仅使用上下文进行处理
            if self.disable_self_attn:
                x = self.attn1(self.norm1(x), context=context) + x  # 计算自注意力并加回原输入
            else:
                x = self.attn1(self.norm1(x)) + x  # 计算自注意力并加回原输入
    
            # 如果存在第二个自注意力层
            if self.attn2 is not None:
                # 根据标志决定是否使用上下文
                if self.switch_temporal_ca_to_sa:
                    x = self.attn2(self.norm2(x)) + x  # 仅计算自注意力
                else:
                    x = self.attn2(self.norm2(x), context=context) + x  # 使用上下文计算自注意力
            x_skip = x  # 保存当前状态以便残差连接
            x = self.ff(self.norm3(x))  # 归一化后通过前馈层
            # 如果使用残差连接，则将之前的状态加回
            if self.is_res:
                x += x_skip
    
            # 调整输出张量的形状为原始格式
            x = rearrange(x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps)
            return x  # 返回处理后的张量
    
    # 定义获取最后一层的权重的方法
        def get_last_layer(self):
            return self.ff.net[-1].weight  # 返回前馈网络最后一层的权重
# 定义数据类型与 PyTorch 数据类型的映射字典
str_to_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

# 定义空间视频变换器类，继承自空间变换器
class SpatialVideoTransformer(SpatialTransformer):
    # 初始化方法，接收多个参数以配置变换器
    def __init__(
        self,
        in_channels,  # 输入通道数
        n_heads,      # 注意力头数
        d_head,       # 每个头的维度
        depth=1,      # 网络深度
        dropout=0.0,  # dropout 率
        use_linear=False,  # 是否使用线性层
        context_dim=None,  # 上下文维度
        use_spatial_context=False,  # 是否使用空间上下文
        timesteps=None,  # 时间步数
        merge_strategy: str = "fixed",  # 合并策略
        merge_factor: float = 0.5,  # 合并因子
        time_context_dim=None,  # 时间上下文维度
        ff_in=False,  # 前馈网络输入标志
        checkpoint=False,  # 是否启用检查点
        time_depth=1,  # 时间深度
        attn_mode="softmax",  # 注意力模式
        disable_self_attn=False,  # 是否禁用自注意力
        disable_temporal_crossattention=False,  # 是否禁用时间交叉注意力
        max_time_embed_period: int = 10000,  # 最大时间嵌入周期
        dtype="fp32",  # 数据类型
    ):
        # 调用父类初始化方法，传递部分参数
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            use_checkpoint=checkpoint,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
        )
        # 设置类属性
        self.time_depth = time_depth
        self.depth = depth
        self.max_time_embed_period = max_time_embed_period

        # 初始化时间混合头维度与数量
        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        # 计算时间混合内部维度
        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        # 计算内部维度
        inner_dim = n_heads * d_head
        # 如果使用空间上下文，更新时间上下文维度
        if use_spatial_context:
            time_context_dim = context_dim

        # 创建时间堆栈，包含多个视频变换器块
        self.time_stack = nn.ModuleList(
            [
                VideoTransformerBlock(
                    inner_dim,  # 内部维度
                    n_time_mix_heads,  # 时间混合头数
                    time_mix_d_head,  # 时间混合头维度
                    dropout=dropout,  # dropout 率
                    context_dim=time_context_dim,  # 时间上下文维度
                    timesteps=timesteps,  # 时间步数
                    checkpoint=checkpoint,  # 检查点标志
                    ff_in=ff_in,  # 前馈网络输入标志
                    inner_dim=time_mix_inner_dim,  # 时间混合内部维度
                    attn_mode=attn_mode,  # 注意力模式
                    disable_self_attn=disable_self_attn,  # 禁用自注意力标志
                    disable_temporal_crossattention=disable_temporal_crossattention,  # 禁用时间交叉注意力标志
                )
                for _ in range(self.depth)  # 创建指定深度的块
            ]
        )

        # 确保时间堆栈与变换器块的数量一致
        assert len(self.time_stack) == len(self.transformer_blocks)

        # 设置类属性
        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        # 计算时间嵌入维度
        time_embed_dim = self.in_channels * 4
        # 定义时间位置嵌入序列
        self.time_pos_embed = nn.Sequential(
            linear(self.in_channels, time_embed_dim),  # 从输入通道到时间嵌入维度的线性变换
            nn.SiLU(),  # 应用 SiLU 激活函数
            linear(time_embed_dim, self.in_channels),  # 从时间嵌入维度回到输入通道的线性变换
        )

        # 初始化时间混合器
        self.time_mixer = AlphaBlender(alpha=merge_factor, merge_strategy=merge_strategy)
        # 设置数据类型
        self.dtype = str_to_dtype[dtype]

    # 前向传播方法，接收输入张量和上下文
    def forward(
        self,
        x: torch.Tensor,  # 输入张量
        context: Optional[torch.Tensor] = None,  # 可选的上下文张量
        time_context: Optional[torch.Tensor] = None,  # 可选的时间上下文张量
        timesteps: Optional[int] = None,  # 可选的时间步数
        image_only_indicator: Optional[torch.Tensor] = None,  # 可选的图像指示器张量
    ) -> torch.Tensor:  # 函数返回一个 torch.Tensor 类型的结果
        _, _, h, w = x.shape  # 解包输入张量 x 的形状，获取高度 h 和宽度 w
        x_in = x  # 保存输入张量 x 的原始值，便于后续操作
        spatial_context = None  # 初始化空间上下文为 None
        if exists(context):  # 如果上下文存在
            spatial_context = context  # 将上下文赋值给空间上下文

        if self.use_spatial_context:  # 如果使用空间上下文
            assert context.ndim == 3, f"n dims of spatial context should be 3 but are {context.ndim}"  # 检查上下文的维度是否为 3

            time_context = context  # 将上下文赋值给时间上下文
            time_context_first_timestep = time_context[::timesteps]  # 获取时间上下文的第一时间步
            time_context = repeat(time_context_first_timestep, "b ... -> (b n) ...", n=h * w)  # 重复时间上下文以适应高度和宽度的大小
        elif time_context is not None and not self.use_spatial_context:  # 如果时间上下文存在且不使用空间上下文
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)  # 重复时间上下文以适应高度和宽度的大小
            if time_context.ndim == 2:  # 如果时间上下文的维度为 2
                time_context = rearrange(time_context, "b c -> b 1 c")  # 将时间上下文的维度重新排列

        x = self.norm(x)  # 对输入张量 x 进行归一化处理
        if not self.use_linear:  # 如果不使用线性变换
            x = self.proj_in(x)  # 对 x 进行输入投影
        x = rearrange(x, "b c h w -> b (h w) c")  # 将 x 的维度重新排列为 (batch_size, height*width, channels)
        if self.use_linear:  # 如果使用线性变换
            x = self.proj_in(x)  # 对 x 进行输入投影

        num_frames = torch.arange(timesteps, device=x.device)  # 生成一个包含时间步数的张量 num_frames
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)  # 重复 num_frames 以适应 batch 大小
        num_frames = rearrange(num_frames, "b t -> (b t)")  # 将 num_frames 的维度重新排列为一维

        t_emb = timestep_embedding(  # 计算时间步嵌入
            num_frames,  # 输入时间步
            self.in_channels,  # 输入通道数
            repeat_only=False,  # 不仅仅重复
            max_period=self.max_time_embed_period,  # 最大周期
            dtype=self.dtype,  # 数据类型
        )
        emb = self.time_pos_embed(t_emb)  # 通过时间步嵌入计算位置嵌入
        emb = emb[:, None, :]  # 在位置维度上增加一个维度

        for it_, (block, mix_block) in enumerate(zip(self.transformer_blocks, self.time_stack)):  # 遍历 transformer 块和时间堆栈
            x = block(  # 通过 transformer 块处理 x
                x,  # 输入张量
                context=spatial_context,  # 传入空间上下文
            )

            x_mix = x  # 复制处理后的 x 为 x_mix
            x_mix = x_mix + emb  # 将位置嵌入加到 x_mix 上

            x_mix = mix_block(x_mix, context=time_context, timesteps=timesteps)  # 通过混合块处理 x_mix
            x = self.time_mixer(  # 通过时间混合器整合空间和时间特征
                x_spatial=x,  # 输入空间特征
                x_temporal=x_mix,  # 输入时间特征
                image_only_indicator=image_only_indicator,  # 图像标识符
            )
        if self.use_linear:  # 如果使用线性变换
            x = self.proj_out(x)  # 对 x 进行输出投影
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)  # 将 x 的维度重新排列为 (batch_size, channels, height, width)
        if not self.use_linear:  # 如果不使用线性变换
            x = self.proj_out(x)  # 对 x 进行输出投影
        out = x + x_in  # 将输出张量与原始输入 x_in 相加
        return out  # 返回最终结果
```

# `.\cogvideo-finetune\sat\sgm\modules\__init__.py`

```py
# 从相对路径导入 GeneralConditioner 模块
from .encoders.modules import GeneralConditioner

# 定义一个无条件配置的字典
UNCONDITIONAL_CONFIG = {
    # 设置目标为 GeneralConditioner 模块
    "target": "sgm.modules.GeneralConditioner",
    # 定义参数，初始化 emb_models 为一个空列表
    "params": {"emb_models": []},
}
```

# `.\cogvideo-finetune\sat\sgm\util.py`

```py
# 导入 functools 模块以支持高阶函数功能
import functools
# 导入 importlib 模块以动态导入模块
import importlib
# 导入 os 模块以支持操作系统相关功能
import os
# 从 functools 导入 partial 函数以便于函数部分应用
from functools import partial
# 从 inspect 导入 isfunction 函数用于检查对象是否为函数
from inspect import isfunction

# 导入 fsspec 库以支持文件系统规范
import fsspec
# 导入 numpy 库用于科学计算
import numpy as np
# 导入 torch 库以支持深度学习功能
import torch
# 从 PIL 导入 Image、ImageDraw 和 ImageFont 以支持图像处理
from PIL import Image, ImageDraw, ImageFont
# 从 safetensors.torch 导入 load_file 函数以加载安全张量
from safetensors.torch import load_file as load_safetensors
# 导入 torch.distributed 模块以支持分布式训练
import torch.distributed

# 定义全局变量以存储并行组的上下文
_CONTEXT_PARALLEL_GROUP = None
# 定义全局变量以存储并行组的大小
_CONTEXT_PARALLEL_SIZE = None


# 定义函数以检查上下文并行是否已初始化
def is_context_parallel_initialized():
    # 检查上下文并行组是否为 None
    if _CONTEXT_PARALLEL_GROUP is None:
        return False
    else:
        return True


# 定义函数以设置上下文并行组和大小
def set_context_parallel_group(size, group):
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_SIZE
    # 设置上下文并行组
    _CONTEXT_PARALLEL_GROUP = group
    # 设置上下文并行大小
    _CONTEXT_PARALLEL_SIZE = size


# 定义函数以初始化上下文并行
def initialize_context_parallel(context_parallel_size):
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_SIZE

    # 断言上下文并行组未被初始化
    assert _CONTEXT_PARALLEL_GROUP is None, "context parallel group is already initialized"
    # 设置上下文并行大小
    _CONTEXT_PARALLEL_SIZE = context_parallel_size

    # 获取当前进程的 rank
    rank = torch.distributed.get_rank()
    # 获取全局进程的数量
    world_size = torch.distributed.get_world_size()

    # 按上下文并行大小遍历所有进程以创建新的并行组
    for i in range(0, world_size, context_parallel_size):
        ranks = range(i, i + context_parallel_size)
        # 创建新的分组
        group = torch.distributed.new_group(ranks)
        # 如果当前 rank 在创建的 ranks 中，则设置上下文并行组
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            break


# 定义函数以获取当前上下文并行组
def get_context_parallel_group():
    # 断言上下文并行组已初始化
    assert _CONTEXT_PARALLEL_GROUP is not None, "context parallel group is not initialized"

    return _CONTEXT_PARALLEL_GROUP


# 定义函数以获取当前上下文并行的世界大小
def get_context_parallel_world_size():
    # 断言上下文并行大小已初始化
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    return _CONTEXT_PARALLEL_SIZE


# 定义函数以获取当前上下文并行的 rank
def get_context_parallel_rank():
    # 断言上下文并行大小已初始化
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    # 获取当前进程的 rank
    rank = torch.distributed.get_rank()
    # 计算当前上下文并行的 rank
    cp_rank = rank % _CONTEXT_PARALLEL_SIZE
    return cp_rank


# 定义函数以获取当前上下文并行组的 rank
def get_context_parallel_group_rank():
    # 断言上下文并行大小已初始化
    assert _CONTEXT_PARALLEL_SIZE is not None, "context parallel size is not initialized"

    # 获取当前进程的 rank
    rank = torch.distributed.get_rank()
    # 计算当前上下文并行组的 rank
    cp_group_rank = rank // _CONTEXT_PARALLEL_SIZE

    return cp_group_rank


# 定义 SafeConv3d 类，继承自 torch.nn.Conv3d
class SafeConv3d(torch.nn.Conv3d):
    # 定义前向传播函数，接收输入数据
    def forward(self, input):
        # 计算输入数据的内存占用（以 GB 为单位），乘以 2 是因为需要考虑反向传播的内存
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / 1024**3
        # 如果内存占用超过 2GB，则进行内存优化处理
        if memory_count > 2:
            # kernel_size 取自实例属性，表示卷积核的大小
            kernel_size = self.kernel_size[0]
            # 计算需要将输入分成的部分数量，以控制内存占用
            part_num = int(memory_count / 2) + 1
            # 将输入数据按时间维度进行分块处理
            input_chunks = torch.chunk(input, part_num, dim=2)  # NCTHW
            # 如果卷积核大小大于 1，则需要处理相邻块的拼接
            if kernel_size > 1:
                # 将第一个块保留，后续块与前一个块拼接
                input_chunks = [input_chunks[0]] + [
                    torch.cat((input_chunks[i - 1][:, :, -kernel_size + 1 :], input_chunks[i]), dim=2)
                    for i in range(1, len(input_chunks))
                ]

            # 初始化输出块列表
            output_chunks = []
            # 对每个输入块进行前向传播，并将结果存储到输出块列表中
            for input_chunk in input_chunks:
                output_chunks.append(super(SafeConv3d, self).forward(input_chunk))
            # 将所有输出块在时间维度上拼接成最终输出
            output = torch.cat(output_chunks, dim=2)
            # 返回拼接后的输出
            return output
        else:
            # 如果内存占用不超过 2GB，直接调用父类的前向传播
            return super(SafeConv3d, self).forward(input)
# 禁用训练模式，确保模型的训练/评估模式不会再更改
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self  # 返回当前对象，保持状态不变


# 从元组字符串中提取字符串
def get_string_from_tuple(s):
    try:
        # 检查字符串是否以括号开始和结束
        if s[0] == "(" and s[-1] == ")":
            # 将字符串转换为元组
            t = eval(s)
            # 检查 t 的类型是否为元组
            if type(t) == tuple:
                return t[0]  # 返回元组的第一个元素
            else:
                pass  # 如果不是元组则不做处理
    except:
        pass  # 捕获异常，防止程序崩溃
    return s  # 返回原始字符串


# 检查一个数是否是 2 的幂
def is_power_of_two(n):
    """
    chat.openai.com/chat
    Return True if n is a power of 2, otherwise return False.

    The function is_power_of_two takes an integer n as input and returns True if n is a power of 2, otherwise it returns False.
    The function works by first checking if n is less than or equal to 0. If n is less than or equal to 0, it can't be a power of 2, so the function returns False.
    If n is greater than 0, the function checks whether n is a power of 2 by using a bitwise AND operation between n and n-1. If n is a power of 2, then it will have only one bit set to 1 in its binary representation. When we subtract 1 from a power of 2, all the bits to the right of that bit become 1, and the bit itself becomes 0. So, when we perform a bitwise AND between n and n-1, we get 0 if n is a power of 2, and a non-zero value otherwise.
    Thus, if the result of the bitwise AND operation is 0, then n is a power of 2 and the function returns True. Otherwise, the function returns False.

    """
    if n <= 0:
        return False  # 如果 n 小于或等于 0，返回 False
    return (n & (n - 1)) == 0  # 返回 n 是否是 2 的幂


# 自动混合精度处理函数
def autocast(f, enabled=True):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
            enabled=enabled,  # 启用或禁用自动混合精度
            dtype=torch.get_autocast_gpu_dtype(),  # 获取 GPU 的自动混合精度数据类型
            cache_enabled=torch.is_autocast_cache_enabled(),  # 检查缓存是否启用
        ):
            return f(*args, **kwargs)  # 调用原始函数并返回结果

    return do_autocast  # 返回自动混合精度处理的封装函数


# 从配置加载部分对象
def load_partial_from_config(config):
    return partial(get_obj_from_str(config["target"]), **config.get("params", dict()))  # 返回部分应用的对象


# 将文本作为图像进行记录
def log_txt_as_img(wh, xc, size=10):
    # wh 是一个包含 (宽度, 高度) 的元组
    # xc 是要绘制的标题列表
    b = len(xc)  # 获取标题列表的长度
    txts = list()  # 初始化文本图像列表
    for bi in range(b):  # 遍历每个标题
        txt = Image.new("RGB", wh, color="white")  # 创建一个白色背景的图像
        draw = ImageDraw.Draw(txt)  # 创建可绘制对象
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)  # 加载指定字体
        nc = int(40 * (wh[0] / 256))  # 计算每行最多字符数
        if isinstance(xc[bi], list):  # 如果标题是列表
            text_seq = xc[bi][0]  # 获取第一个元素
        else:
            text_seq = xc[bi]  # 否则直接使用标题
        lines = "\n".join(text_seq[start : start + nc] for start in range(0, len(text_seq), nc))  # 将文本分行

        try:
            draw.text((0, 0), lines, fill="black", font=font)  # 绘制文本
        except UnicodeEncodeError:  # 如果编码错误
            print("Cant encode string for logging. Skipping.")  # 打印错误信息并跳过

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0  # 转换图像为数组并归一化
        txts.append(txt)  # 添加到列表中
    txts = np.stack(txts)  # 堆叠所有文本图像
    txts = torch.tensor(txts)  # 转换为 PyTorch 张量
    # 返回包含文本数据的列表或字符串
        return txts
# 定义一个部分类，用于传递参数到初始化方法
def partialclass(cls, *args, **kwargs):
    # 创建一个新类，继承自原始类
    class NewCls(cls):
        # 使用 functools.partialmethod 将原始类的初始化方法与参数绑定
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    # 返回新创建的类
    return NewCls


# 将给定路径转换为绝对路径
def make_path_absolute(path):
    # 解析路径并获取文件系统和路径
    fs, p = fsspec.core.url_to_fs(path)
    # 如果文件系统协议为文件，则返回绝对路径
    if fs.protocol == "file":
        return os.path.abspath(p)
    # 否则返回原始路径
    return path


# 判断输入是否为地图类型的张量
def ismap(x):
    # 如果输入不是张量，则返回 False
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量是否为四维且第二维大于3
    return (len(x.shape) == 4) and (x.shape[1] > 3)


# 判断输入是否为图像类型的张量
def isimage(x):
    # 如果输入不是张量，则返回 False
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量是否为四维且第二维为3或1
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


# 判断输入是否为热图类型的张量
def isheatmap(x):
    # 如果输入不是张量，则返回 False
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量是否为二维
    return x.ndim == 2


# 判断输入是否为邻接类型的张量
def isneighbors(x):
    # 如果输入不是张量，则返回 False
    if not isinstance(x, torch.Tensor):
        return False
    # 检查张量是否为五维且第三维为3或1
    return x.ndim == 5 and (x.shape[2] == 3 or x.shape[2] == 1)


# 检查输入是否存在
def exists(x):
    # 判断输入是否不为 None
    return x is not None


# 扩展张量的维度，使其与目标张量的维度相同
def expand_dims_like(x, y):
    # 当 x 的维度不等于 y 的维度时，持续扩展 x 的维度
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    # 返回扩展后的 x
    return x


# 返回存在的值或默认值
def default(val, d):
    # 如果 val 存在，则返回 val
    if exists(val):
        return val
    # 返回默认值，如果 d 是函数则调用它
    return d() if isfunction(d) else d


# 计算张量的平坦均值
def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    对所有非批次维度进行均值计算。
    """
    # 计算张量在所有非批次维度上的均值
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# 计算模型的参数总数
def count_params(model, verbose=False):
    # 计算模型所有参数的总数量
    total_params = sum(p.numel() for p in model.parameters())
    # 如果 verbose 为真，则打印参数数量
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    # 返回参数总数
    return total_params


# 根据配置实例化对象
def instantiate_from_config(config, **extra_kwargs):
    # 检查配置中是否包含目标键
    if not "target" in config:
        # 如果配置为 "__is_first_stage__"，返回 None
        if config == "__is_first_stage__":
            return None
        # 如果配置为 "__is_unconditional__"，返回 None
        elif config == "__is_unconditional__":
            return None
        # 抛出缺少目标键的异常
        raise KeyError("Expected key `target` to instantiate.")
    # 根据目标字符串实例化对象，并传递参数
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **extra_kwargs)


# 从字符串获取对象
def get_obj_from_str(string, reload=False, invalidate_cache=True):
    # 分割模块和类名
    module, cls = string.rsplit(".", 1)
    # 如果需要，失效缓存
    if invalidate_cache:
        importlib.invalidate_caches()
    # 如果需要重载模块，导入并重载
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # 返回模块中的类对象
    return getattr(importlib.import_module(module, package=None), cls)


# 在张量末尾添加零
def append_zero(x):
    # 将零张量与输入张量拼接
    return torch.cat([x, x.new_zeros([1])])


# 添加维度到张量以达到目标维度
def append_dims(x, target_dims):
    """将维度添加到张量末尾，直到其具有目标维度。"""
    # 计算需要添加的维度数量
    dims_to_append = target_dims - x.ndim
    # 如果输入维度大于目标维度，抛出异常
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    # 返回添加了维度的张量
    return x[(...,) + (None,) * dims_to_append]


# 从配置加载模型
def load_model_from_config(config, ckpt, verbose=True, freeze=True):
    # 打印加载模型的检查点路径
    print(f"Loading model from {ckpt}")
    # 检查检查点文件名是否以 "ckpt" 结尾
    if ckpt.endswith("ckpt"):
        # 从检查点文件加载状态字典到 CPU
        pl_sd = torch.load(ckpt, map_location="cpu")
        # 如果状态字典中包含 "global_step" 键
        if "global_step" in pl_sd:
            # 打印全局步数
            print(f"Global Step: {pl_sd['global_step']}")
        # 提取状态字典
        sd = pl_sd["state_dict"]
    # 检查检查点文件名是否以 "safetensors" 结尾
    elif ckpt.endswith("safetensors"):
        # 从 safetensors 文件加载状态字典
        sd = load_safetensors(ckpt)
    # 如果文件名不符合上述格式，则抛出未实现的错误
    else:
        raise NotImplementedError

    # 根据配置实例化模型
    model = instantiate_from_config(config.model)

    # 加载模型状态字典，允许非严格匹配
    m, u = model.load_state_dict(sd, strict=False)

    # 如果有缺失的键且详细模式开启
    if len(m) > 0 and verbose:
        # 打印缺失的键
        print("missing keys:")
        print(m)
    # 如果有意外的键且详细模式开启
    if len(u) > 0 and verbose:
        # 打印意外的键
        print("unexpected keys:")
        print(u)

    # 如果冻结标志为真
    if freeze:
        # 遍历模型参数
        for param in model.parameters():
            # 设置参数为不需要梯度
            param.requires_grad = False

    # 将模型设置为评估模式
    model.eval()
    # 返回已加载的模型
    return model
# 定义一个获取配置路径的函数，返回字符串类型
def get_configs_path() -> str:
    # 函数文档说明：获取 `configs` 目录的位置
    this_dir = os.path.dirname(__file__)  # 获取当前文件所在目录的路径
    # 创建候选路径，包含当前目录下的 configs 目录和上级目录下的 configs 目录
    candidates = (
        os.path.join(this_dir, "configs"),
        os.path.join(this_dir, "..", "configs"),
    )
    # 遍历每个候选路径
    for candidate in candidates:
        candidate = os.path.abspath(candidate)  # 将候选路径转换为绝对路径
        if os.path.isdir(candidate):  # 检查该路径是否为一个目录
            return candidate  # 如果是，返回该路径
    # 如果没有找到有效的 configs 目录，抛出文件未找到错误
    raise FileNotFoundError(f"Could not find SGM configs in {candidates}")


# 定义一个获取嵌套属性的函数
def get_nested_attribute(obj, attribute_path, depth=None, return_key=False):
    # 函数文档说明：递归获取对象的嵌套属性
    attributes = attribute_path.split(".")  # 根据 '.' 分割属性路径
    if depth is not None and depth > 0:  # 如果指定深度且大于零
        attributes = attributes[:depth]  # 限制属性列表到指定深度
    assert len(attributes) > 0, "At least one attribute should be selected"  # 确保至少有一个属性
    current_attribute = obj  # 初始化当前属性为对象
    current_key = None  # 初始化当前键
    # 遍历每个属性
    for level, attribute in enumerate(attributes):
        current_key = ".".join(attributes[: level + 1])  # 生成当前键的字符串
        try:
            id_ = int(attribute)  # 尝试将属性转换为整数
            current_attribute = current_attribute[id_]  # 使用索引访问属性
        except ValueError:  # 如果转换失败
            current_attribute = getattr(current_attribute, attribute)  # 使用 getattr 获取属性

    # 返回当前属性和当前键的元组，或者只返回当前属性
    return (current_attribute, current_key) if return_key else current_attribute


# 从 math 模块导入平方根函数
from math import sqrt


# 定义一个 SeededNoise 类
class SeededNoise:
    # 初始化方法，接受种子和权重
    def __init__(self, seeds, weights):
        self.seeds = seeds  # 存储种子
        self.weights = weights  # 存储权重
        weight_square_sum = 0  # 初始化权重平方和
        # 遍历每个权重
        for weight in weights:
            weight_square_sum += weight**2  # 计算权重的平方和
        self.weight_square_sum_sqrt = sqrt(weight_square_sum)  # 计算权重平方和的平方根
        self.cnt = 0  # 初始化计数器

    # 定义可调用方法
    def __call__(self, x):
        self.cnt += 1  # 计数器加一
        randn_combined = torch.zeros_like(x)  # 创建与 x 同形状的零张量
        # 遍历种子和权重
        for seed, weight in zip(self.seeds, self.weights):
            randn = np.random.RandomState(seed + self.cnt).randn(*x.shape)  # 生成正态分布随机数
            randn = torch.from_numpy(randn, dtype=x.dtype, device=x.device)  # 将随机数转换为张量
            randn_combined += randn * weight  # 将加权随机数累加到组合中
        randn_combined /= self.weight_square_sum_sqrt  # 将组合随机数归一化
        return randn_combined  # 返回最终的组合随机数
```

# `.\cogvideo-finetune\sat\sgm\webds.py`

```py
# 导入所需的标准库
import sys  # 系统相关的功能
import io  # 输入输出操作
import os  # 操作系统功能
import re  # 正则表达式操作
import json  # JSON 数据处理
import tarfile  # TAR 文件处理
from functools import partial  # 偏函数应用

# 导入 webdataset 库的相关模块
import webdataset as wds  # webdataset 的主模块
from webdataset import ResampledShards, DataPipeline, tarfile_to_samples  # 导入特定功能
from webdataset.filters import pipelinefilter  # 导入过滤功能
from webdataset.tariterators import url_opener, group_by_keys  # 导入 TAR 迭代器相关功能
from webdataset.handlers import reraise_exception  # 导入异常处理功能
from webdataset.gopen import gopen_schemes, gopen  # 导入打开函数和方案

def pytorch_worker_info(group=None):  # sourcery skip: use-contextlib-suppress
    """返回 PyTorch 和一些分布式环境的节点和工作者信息。"""
    rank = 0  # 初始化节点秩
    world_size = 1  # 初始化世界大小
    worker = 0  # 初始化工作者 ID
    num_workers = 1  # 初始化工作者数量
    try:
        import torch.distributed  # 导入分布式 PyTorch 模块

        # 检查分布式模块是否可用和已初始化
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            group = group or torch.distributed.group.WORLD  # 设置组为 WORLD
            rank = torch.distributed.get_rank(group=group)  # 获取节点秩
            world_size = torch.distributed.get_world_size(group=group)  # 获取世界大小
    except ModuleNotFoundError:
        pass  # 如果未找到模块，则跳过
    try:
        import torch.utils.data  # 导入数据工具模块

        worker_info = torch.utils.data.get_worker_info()  # 获取工作者信息
        if worker_info is not None:  # 如果工作者信息存在
            worker = worker_info.id  # 获取工作者 ID
            num_workers = worker_info.num_workers  # 获取工作者总数
    except ModuleNotFoundError:
        pass  # 如果未找到模块，则跳过

    return rank, world_size, worker, num_workers  # 返回节点信息

def pytorch_worker_seed(group=None):
    """为每个工作者和节点计算唯一且确定性的随机种子。"""
    rank, world_size, worker, num_workers = pytorch_worker_info(group=group)  # 获取工作者信息
    return rank * 1000 + worker  # 计算并返回随机种子

def worker_seed_sat(group=None, seed=0):
    return pytorch_worker_seed(group=group) + seed * 23  # 计算工作者的随机种子并增加偏移

class ConfiguredResampledShards(ResampledShards):
    def __init__(self, urls, seed, nshards=sys.maxsize, deterministic=True):
        from sat.helpers import print_rank0  # 导入打印功能

        try:
            from megatron.core.parallel_state import get_data_parallel_group  # 尝试导入 Megatron 数据并行组

            group = get_data_parallel_group()  # 获取数据并行组
            print_rank0("Using megatron data parallel group.")  # 打印使用的组信息
        except:
            from sat.mpu import get_data_parallel_group  # 导入备用的数据并行组

            try:
                group = get_data_parallel_group()  # 获取数据并行组
                print_rank0("Using sat data parallel group.")  # 打印使用的组信息
            except AssertionError:
                group = None  # 如果没有指定组，则设置为 None
                print_rank0("No data parallel group is specified!")  # 打印警告信息
        worker_seed_sat_this = partial(worker_seed_sat, group=group, seed=seed)  # 创建偏函数
        super().__init__(urls, nshards, worker_seed_sat_this, deterministic)  # 调用父类构造函数

class SimpleDistributedWebDataset(DataPipeline):  # 定义简单的分布式 Web 数据集类
    # 初始化方法，接收路径、处理函数、种子以及可选的洗牌缓冲区大小
    def __init__(self, path, process_fn, seed, *, shuffle_buffer=1000):
        # 如果将 shuffle_buffer 设置为 1，则禁用洗牌，模型并行将会有所不同
        try:
            # 从 sat.mpu 模块导入获取模型并行世界大小的函数
            from sat.mpu import get_model_parallel_world_size
    
            # 检查模型并行世界大小，如果大于 1，则将洗牌缓冲区设置为 1
            if get_model_parallel_world_size() > 1:
                shuffle_buffer = 1
        except Exception:
            # 捕获异常并忽略
            pass
        # 调用父类构造函数，初始化配置的重采样分片及相关参数
        super().__init__(
            ConfiguredResampledShards(path, seed),  # 使用指定路径和种子初始化重采样分片，推荐使用多个分片以避免不均匀
            tarfile_to_samples(),  # 将 tar 文件转换为样本
            wds.shuffle(shuffle_buffer),  # 使用洗牌函数，并传入洗牌缓冲区大小
            process_fn,  # 传入处理函数
        )
# 定义一个迭代器函数，用于遍历 tar 文件，生成文件名和内容的对
def tar_file_iterator_with_meta(
    # 输入的字节流对象，适用于 tarfile
    fileobj, 
    # 元数据文件中不同项的键
    meta_names, 
    # 用于跳过某些键的正则表达式（默认值为 r"__[^/]*__($|/)"）
    skip_meta=r"__[^/]*__($|/)", 
    # 文件后缀名（可选）
    suffix=None, 
    # 异常处理的处理程序（默认为 reraise_exception）
    handler=reraise_exception, 
    # 元数据流（可选）
    meta_stream=None
):
    """遍历 tar 文件，返回给定 tar 流的文件名和内容对。

    :param fileobj: 适用于 tarfile 的字节流
    :param meta_names: 元数据文件中不同项的键
    :param skip_meta: 完全跳过的键的正则表达式（默认值 = r"__[^/]*__($|/)"）
    """
    # 打开 tar 文件流，以读取模式
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    # 从文件对象中提取数据目录和文件名
    data_dir, filename = fileobj.name.rsplit("/", 1)
    # 初始化元数据字典，用于存储元数据
    meta_data = {}  # {id: {meta_name: meta_value, meta_name2: meta_value2, ...}}

    # 如果没有提供元数据流
    if meta_stream is None:
        # 生成元数据文件名，使用文件名的前缀加上 ".meta.jsonl"
        meta_file_name = filename.split(".")[0] + ".meta.jsonl"
        # 构建元数据文件的完整路径
        meta_path = os.path.join(data_dir, meta_file_name)
        # 如果元数据文件存在，则打开文件
        if os.path.exists(meta_path):
            meta_stream = open(meta_path, "r")
    else:
        # 如果提供了元数据流，则使用其名称
        meta_file_name = meta_stream.name

    # 如果元数据流存在
    if meta_stream is not None:
        # 遍历元数据流的每一行，记录行号和内容
        for lineno, line in enumerate(meta_stream):
            meta_list = []
            try:
                # 尝试将行内容解析为 JSON 对象
                meta_list.append(json.loads(line))
            except Exception as exn:
                # 导入帮助函数以打印错误
                from sat.helpers import print_rank0

                # 打印解析 JSONL 时的错误信息
                print_rank0(f"Error in loading jsonl {meta_file_name}, lineno {lineno}: {line}", level="DEBUG")
                # 继续下一行
                continue
            # 遍历解析出的每个元数据项
            for item in meta_list:
                # 如果元数据项的键不存在于元数据字典中，则初始化它
                if not item["key"] in meta_data:
                    meta_data[item["key"]] = {}
                # 遍历所有指定的元数据名称
                for meta_name in meta_names:
                    # 如果元数据项中包含该元数据名称，则将其值存储在元数据字典中
                    if meta_name in item:
                        meta_data[item["key"]][meta_name] = item[meta_name]
        # 关闭元数据流
        meta_stream.close()
    # 尝试处理流中的每个项目
        try:
            # 遍历流中的每个 tarinfo 对象
            for tarinfo in stream:
                # 获取当前项目的文件名
                fname = tarinfo.name
                try:
                    # 如果不是常规文件，则跳过
                    if not tarinfo.isreg():
                        continue
                    # 如果文件名为空，则跳过
                    if fname is None:
                        continue
                    # 跳过以双下划线开头和结尾的元数据文件
                    if "/" not in fname and fname.startswith("__") and fname.endswith("__"):
                        # 目前跳过元数据
                        continue
                    # 如果指定了 skip_meta，且文件名匹配，则跳过
                    if skip_meta is not None and re.match(skip_meta, fname):
                        continue
                    # 如果文件是 txt 类型且有后缀，则读取内容并附加后缀
                    if fname.endswith(".txt") and suffix is not None:
                        data = (stream.extractfile(tarinfo).read().decode() + suffix).encode()
                    else:
                        # 否则仅读取文件内容
                        data = stream.extractfile(tarinfo).read()
                    # 创建包含文件名和数据的字典
                    result = dict(fname=fname, data=data)
                    # 生成结果字典
                    yield result
    
                    # 如果文件名以 .id 结尾
                    if fname.endswith(".id"):
                        # 获取文件 ID，去掉扩展名
                        fid = fname.split(".")[0]
                        # 检查文件 ID 是否包含特定字符串并处理
                        if "-$#%@&" in fid:
                            sfid = fid.split("-$#%@&")[0]
                        else:
                            sfid = fid
                        # 从元数据中获取相关数据
                        meta_data_fid = meta_data.get(sfid, {})
                        # 遍历元数据名称
                        for meta_name in meta_names:
                            # 构建元数据文件名
                            meta_fname = fid + "." + meta_name
                            # 获取元数据内容
                            meta = meta_data_fid.get(meta_name, None)
                            # 生成包含元数据的字典
                            yield dict(fname=meta_fname, data=meta)
                    # 清空流的成员列表
                    stream.members = []
                except Exception as exn:
                    # 如果异常有参数，则附加当前文件对象信息
                    if hasattr(exn, "args") and len(exn.args) > 0:
                        exn.args = (exn.args[0] + " @ " + str(fileobj),) + exn.args[1:]
                    # 处理异常，如果处理成功则继续
                    if handler(exn):
                        continue
                    else:
                        # 否则跳出循环
                        break
        except Exception as exn:
            # 打印外层异常信息
            print(exn)
        # 删除流对象以释放资源
        del stream
# 扩展一个打开的 tar 文件流，并返回包含文件内容的迭代器
def tar_file_expander_with_meta(data, meta_names, handler=reraise_exception):
    """Expand a stream of open tar files into a stream of tar file contents.

    This returns an iterator over (filename, file_contents).
    """
    # 遍历输入数据中的每个源
    for source in data:
        # 从源字典中获取 URL
        url = source["url"]
        try:
            # 确保源是字典类型
            assert isinstance(source, dict)
            # 确保源字典中包含 "stream" 键
            assert "stream" in source
            # 遍历 tar 文件内容生成器
            for sample in tar_file_iterator_with_meta(source["stream"], meta_names, meta_stream=source["meta_stream"]):
                # 确保样本是字典并包含 "data" 和 "fname"
                assert isinstance(sample, dict) and "data" in sample and "fname" in sample
                # 将 URL 添加到样本字典中
                sample["__url__"] = url
                # 生成样本
                yield sample
        except Exception as exn:
            # 追加流和 URL 到异常参数中
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            # 如果处理异常的函数返回真，继续循环
            if handler(exn):
                continue
            else:
                # 否则，退出循环
                break


# 打开 URL 并返回 URL 和流的配对迭代器
def url_opener(
    data,
    handler,
    **kw,
):
    """Open URLs and yield a stream of url+stream pairs.

    Args:
        data: iterator over dict(url=...)
        handler: exception handler.
        kw: keyword arguments for gopen.gopen.

    Yields:
        a stream of url+stream pairs.
    """
    # 遍历输入数据中的每个样本
    for sample in data:
        # 确保样本是字典类型
        assert isinstance(sample, dict), sample
        # 确保样本字典中包含 "url" 键
        assert "url" in sample
        # 从样本中获取 URL
        url = sample["url"]
        try:
            # 打开 URL 并获取流
            stream = gopen(url, **kw)
            # 检查流是否有 meta_stream 属性
            if hasattr(stream, "meta_stream"):
                # 获取 meta_stream，并删除该属性
                meta_stream = stream.meta_stream
                del stream.meta_stream
            else:
                # 如果没有，则设为 None
                meta_stream = None
            # 更新样本字典，包含流和 meta_stream
            sample.update(stream=stream, meta_stream=meta_stream)
            # 生成样本
            yield sample
        except Exception as exn:
            # 追加 URL 到异常参数中
            exn.args = exn.args + (url,)
            # 如果处理异常的函数返回真，继续循环
            if handler(exn):
                continue
            else:
                # 否则，退出循环
                break


# 使用元数据扩展 tar 文件样本
def tarfile_samples_with_meta(src, meta_names, handler=reraise_exception):
    # 使用 URL 打开器获取流
    streams = url_opener(src, handler=handler)
    # 扩展 tar 文件流并获取文件样本
    files = tar_file_expander_with_meta(streams, meta_names, handler)
    # 按键对样本进行分组
    samples = group_by_keys(files, handler=handler)
    # 返回样本
    return samples


# 定义带有元信息文件的分布式 Web 数据集类
class MetaDistributedWebDataset(DataPipeline):
    """WebDataset with meta information files
    Extra Format:
        in webdataset (tar), for each sample there is a '.id';
        for each tar file, there is a '.meta.jsonl' file with the same name;
        The '.meta.jsonl' file contains lines of json objects, each with a 'key' field to match '.id'.
    """

    # 初始化方法，设置数据集参数
    def __init__(
        self, path, process_fn, seed, *, meta_names=[], nshards=sys.maxsize, shuffle_buffer=1000, include_dirs=None
    ):
        # 设置环境变量，控制是否显示种子（注释掉）
        # os.environ['WDS_SHOW_SEED'] = '1'
        # 导入 PyTorch 库
        import torch

        # 检查当前进程是否为主进程
        if torch.distributed.get_rank() == 0:
            # 如果包含的目录不为 None
            if include_dirs is not None:  # /webdatasets/A,/webdatasets/C
                # 初始化其他路径列表
                other_paths = []
                # 将包含的目录字符串按逗号分割
                include_dirs = include_dirs.split(",")
                # 遍历每个包含的目录
                for include_dir in include_dirs:
                    # 如果目录名中包含通配符 "*"
                    if "*" in include_dir:
                        # 分割目录名和数量
                        include_dir, n = include_dir.split("*")
                        n = int(n)  # 转换数量为整数
                    else:
                        n = 1  # 默认数量为 1
                    # 遍历当前目录及其子目录
                    for cur_dir, dirs, files in os.walk(include_dir):
                        # 遍历所有文件
                        for f in files:
                            # 检查文件是否以 "tar" 结尾且文件大小大于 0
                            if f.endswith("tar") and os.path.getsize(os.path.join(cur_dir, f)) > 0:
                                # 将符合条件的文件路径添加到其他路径列表中
                                # other_paths.append(os.path.join(cur_dir,f))
                                other_paths.extend([os.path.join(cur_dir, f)] * n)  # 根据数量扩展列表
                # print(f'Adding dataset paths {",".join(other_paths)}')
                # 从 braceexpand 库导入
                from braceexpand import braceexpand

                # 如果路径字符串不为空
                if len(path) > 0:  # not ""
                    # 扩展路径并与其他路径合并
                    path = list(braceexpand(path)) + other_paths
                else:
                    # 如果路径为空，仅使用其他路径
                    path = other_paths
            # 将路径包装成列表
            path = [path]
        else:
            # 如果不是主进程，将路径设置为 None
            path = [
                None,
            ]
        # 广播路径列表到所有进程
        torch.distributed.broadcast_object_list(path, src=0)
        # 选择第一个路径
        path = path[0]

        # 生成带元数据的 tar 文件样本处理函数
        tarfile_samples = partial(tarfile_samples_with_meta, meta_names=meta_names)
        # 对 tar 文件样本进行管道过滤
        tarfile_to_samples = pipelinefilter(tarfile_samples)

        # 如果模型并行，设置打乱缓冲区大小为 1 以禁用打乱
        try:
            # 从 sat.mpu 模块导入获取模型并行世界大小的函数
            from sat.mpu import get_model_parallel_world_size

            # 检查模型并行世界大小是否大于 1
            if get_model_parallel_world_size() > 1:
                shuffle_buffer = 1  # 设置打乱缓冲区大小
        except Exception:
            pass  # 忽略导入错误

        # 调用父类初始化方法，传入配置好的重采样分片、样本处理管道及其他参数
        super().__init__(
            ConfiguredResampledShards(path, seed, nshards=nshards),
            tarfile_to_samples(),
            wds.shuffle(shuffle_buffer),
            process_fn,
        )
# rclone 支持
from webdataset.gopen import Pipe  # 从 webdataset.gopen 导入 Pipe 类


def gopen_rclone(url, mode="rb", bufsize=1024 * 1024 * 32):
    """使用 `curl` 打开一个 URL。

    :param url: rclone URL，例如 data:bucket1/foo.tar，数据需要被配置。
    :param mode: 文件模式
    :param bufsize: 缓冲区大小
    """
    # 去掉 URL 前缀 "rclone://"
    url = url.replace("rclone://", "")
    # 如果模式以 "r" 开头，准备读取命令
    if mode[0] == "r":
        cmd = f"rclone cat '{url}'"  # 生成 rclone 读取命令
        return Pipe(  # 返回 Pipe 对象用于读取
            cmd,
            mode=mode,  # 设置文件模式
            shell=True,  # 通过 shell 执行命令
            bufsize=bufsize,  # 设置缓冲区大小
            ignore_status=[141, 23],  # 忽略特定的返回状态
        )  # skipcq: BAN-B604
    # 如果模式以 "w" 开头，准备写入命令
    elif mode[0] == "w":
        cmd = f"rclone cp - '{url}'"  # 生成 rclone 写入命令
        return Pipe(  # 返回 Pipe 对象用于写入
            cmd,
            mode=mode,  # 设置文件模式
            shell=True,  # 通过 shell 执行命令
            bufsize=bufsize,  # 设置缓冲区大小
            ignore_status=[141, 26],  # 忽略特定的返回状态
        )  # skipcq: BAN-B604
    else:
        # 如果模式未知，抛出错误
        raise ValueError(f"{mode}: unknown mode")


def gopen_boto3(url, mode="rb", bufsize=8192 * 2):
    """使用 boto3 API 打开一个 URL。

    :param url: boto3 URL，例如 boto3://bucket1/foo.tar，数据需要被配置。
    :param mode: 文件模式
    :param bufsize: 缓冲区大小
    """
    import boto3  # 导入 boto3 库

    # boto3.set_stream_logger('botocore', level='DEBUG')  # 设置日志记录（已注释）
    # 如果 URL 以 "boto3://" 开头，去掉前缀并标记是否需要元数据
    if url.startswith("boto3://"):
        url = url.replace("boto3://", "")
        need_meta = False
    else:
        url = url.replace("metaboto3://", "")  # 去掉 "metaboto3://" 前缀
        need_meta = True  # 需要元数据

    # 从环境变量获取 S3 配置
    endpoint_url = os.environ.get("S3_ENDPOINT_URL", None)  # S3 端点 URL
    access_key = os.environ.get("S3_ACCESS_KEY_ID", None)  # 访问密钥 ID
    secret_key = os.environ.get("S3_SECRET_ACCESS_KEY", None)  # 秘密访问密钥

    # 如果模式以 "r" 开头，准备读取 S3 对象
    if mode[0] == "r":
        # 创建 S3 客户端
        s3_client = boto3.client(
            "s3", endpoint_url=endpoint_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key
        )
        # 分割桶名和对象键
        bucket, key = url.split("/", 1)

        # 如果需要元数据，下载相应的元数据文件
        if need_meta:
            # 下载一个元数据 JSON 文件
            meta_file_key = key.split(".")[0] + ".meta.jsonl"  # 元数据文件名
            meta_stream = io.BytesIO()  # 创建一个字节流用于存储元数据
            s3_client.download_fileobj(bucket, meta_file_key, meta_stream)  # 从 S3 下载元数据
            meta_stream.seek(0)  # 重置字节流位置
            meta_stream.name = meta_file_key  # 设置字节流名称
        else:
            meta_stream = None  # 不需要元数据时设置为 None

        # 获取数据流对象
        response = s3_client.get_object(Bucket=bucket, Key=key)  # 获取 S3 对象
        response["Body"].name = key  # 设置对象的名称（实际未使用）
        response["Body"].meta_stream = meta_stream  # 将元数据流关联到对象
        return response["Body"]  # 返回数据流对象
    else:
        # 如果模式未知，抛出错误
        raise ValueError(f"{mode}: unknown mode")


# 注册 gopen_rclone 和 gopen_boto3 到 gopen_schemes 字典
gopen_schemes["rclone"] = gopen_rclone
gopen_schemes["boto3"] = gopen_boto3
gopen_schemes["metaboto3"] = gopen_boto3
```

# `.\cogvideo-finetune\sat\sgm\__init__.py`

```py
# 从当前模块导入 AutoencodingEngine 类
from .models import AutoencodingEngine
# 从当前模块导入获取配置路径和根据配置实例化对象的函数
from .util import get_configs_path, instantiate_from_config
# 定义当前模块的版本号
__version__ = "0.1.0"
```

# `.\cogvideo-finetune\sat\train_video.py`

```py
# 导入操作系统模块
import os
# 导入命令行参数解析模块
import argparse
# 从 functools 模块导入 partial 函数，用于部分应用
from functools import partial
# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 的分布式模块
import torch.distributed
# 导入 OmegaConf，用于配置管理
from omegaconf import OmegaConf
# 导入 imageio，用于读取和写入视频
import imageio

# 导入 PyTorch 库
import torch

# 从 sat 模块导入 mpu
from sat import mpu
# 从 sat.training.deepspeed_training 导入 training_main
from sat.training.deepspeed_training import training_main

# 从 sgm.util 导入工具函数
from sgm.util import get_obj_from_str, isheatmap

# 从 diffusion_video 导入 SATVideoDiffusionEngine 类
from diffusion_video import SATVideoDiffusionEngine
# 从 arguments 导入获取命令行参数的函数
from arguments import get_args

# 从 einops 导入 rearrange 函数，用于重新排列张量
from einops import rearrange

# 尝试导入 wandb 库，用于实验跟踪
try:
    import wandb
# 如果 wandb 未安装，打印警告信息
except ImportError:
    print("warning: wandb not installed")


# 打印调试信息的函数
def print_debug(args, s):
    # 如果启用了调试模式
    if args.debug:
        # 添加当前进程的排名到输出字符串
        s = f"RANK:[{torch.distributed.get_rank()}]:" + s
        # 打印调试信息
        print(s)


# 保存文本列表到指定目录的函数
def save_texts(texts, save_dir, iterations):
    # 构建输出文件的路径
    output_path = os.path.join(save_dir, f"{str(iterations).zfill(8)}")
    # 打开输出文件，设置编码为 UTF-8
    with open(output_path, "w", encoding="utf-8") as f:
        # 遍历文本列表
        for text in texts:
            # 将每个文本写入文件，每个文本后换行
            f.write(text + "\n")


# 将视频批次保存为网格和 MP4 格式的函数
def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, T: int, fps: int = 5, args=None, key=None):
    # 创建保存路径，如果不存在则创建
    os.makedirs(save_path, exist_ok=True)

    # 遍历视频批次
    for i, vid in enumerate(video_batch):
        gif_frames = []  # 用于存储帧的列表
        # 遍历视频中的每一帧
        for frame in vid:
            # 重新排列帧的维度，从 (c, h, w) 转为 (h, w, c)
            frame = rearrange(frame, "c h w -> h w c")
            # 将帧数据缩放到 [0, 255] 并转换为 uint8 类型
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            # 将处理后的帧添加到列表中
            gif_frames.append(frame)
        # 构建当前视频保存的路径
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        # 使用 imageio 创建视频写入器
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            # 将每一帧写入视频文件
            for frame in gif_frames:
                writer.append_data(frame)
        # 如果 args 存在且 wandb 被启用
        if args is not None and args.wandb:
            # 记录视频到 wandb
            wandb.log(
                {key + f"_video_{i}": wandb.Video(now_save_path, fps=fps, format="mp4")}, step=args.iteration + 1
            )


# 日志视频的函数
def log_video(batch, model, args, only_log_video_latents=False):
    # 获取文本数据
    texts = batch["txt"]
    # 构建保存文本的目录
    text_save_dir = os.path.join(args.save, "video_texts")
    # 创建保存文本目录
    os.makedirs(text_save_dir, exist_ok=True)
    # 保存文本数据到文件
    save_texts(texts, text_save_dir, args.iteration)

    # 准备 GPU 自动混合精度的参数
    gpu_autocast_kwargs = {
        "enabled": torch.is_autocast_enabled(),
        "dtype": torch.get_autocast_gpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }
    # 在不计算梯度的上下文中，启用自动混合精度
    with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
        # 使用模型记录视频，是否只记录视频的潜在表示
        videos = model.log_video(batch, only_log_video_latents=only_log_video_latents)
    # 检查当前进程是否是主进程（rank 为 0）
    if torch.distributed.get_rank() == 0:
        # 设置视频保存的根目录
        root = os.path.join(args.save, "video")

        # 如果只记录视频潜在变量
        if only_log_video_latents:
            # 创建潜在变量的子目录
            root = os.path.join(root, "latents")
            # 格式化文件名，包含当前迭代次数
            filename = "{}_gs-{:06}".format("latents", args.iteration)
            # 生成完整的文件路径
            path = os.path.join(root, filename)
            # 创建保存路径的目录（如果不存在的话）
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            # 创建文件夹（如果不存在的话）
            os.makedirs(path, exist_ok=True)
            # 保存潜在变量数据到指定路径
            torch.save(videos["latents"], os.path.join(path, "latent.pt"))
        else:
            # 遍历所有视频数据
            for k in videos:
                # 获取当前视频的帧数
                N = videos[k].shape[0]
                # 如果不是热图，裁剪视频数据
                if not isheatmap(videos[k]):
                    videos[k] = videos[k][:N]
                # 如果视频数据是张量
                if isinstance(videos[k], torch.Tensor):
                    # 将张量分离、转换为浮点数并移动到 CPU
                    videos[k] = videos[k].detach().float().cpu()
                    # 如果不是热图，限制张量值范围
                    if not isheatmap(videos[k]):
                        videos[k] = torch.clamp(videos[k], -1.0, 1.0)

            # 获取当前批次的视频帧数
            num_frames = batch["num_frames"][0]
            # 获取当前批次的帧率
            fps = batch["fps"][0].cpu().item()
            # 如果只记录视频潜在变量
            if only_log_video_latents:
                # 创建潜在变量的子目录
                root = os.path.join(root, "latents")
                # 格式化文件名，包含当前迭代次数
                filename = "{}_gs-{:06}".format("latents", args.iteration)
                # 生成完整的文件路径
                path = os.path.join(root, filename)
                # 创建保存路径的目录（如果不存在的话）
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                # 创建文件夹（如果不存在的话）
                os.makedirs(path, exist_ok=True)
                # 保存潜在变量数据到指定路径
                torch.save(videos["latents"], os.path.join(path, "latents.pt"))
            else:
                # 遍历所有视频数据
                for k in videos:
                    # 将视频数据标准化到 [0, 1] 范围
                    samples = (videos[k] + 1.0) / 2.0
                    # 格式化文件名，包含当前迭代次数
                    filename = "{}_gs-{:06}".format(k, args.iteration)

                    # 生成完整的文件路径
                    path = os.path.join(root, filename)
                    # 创建保存路径的目录（如果不存在的话）
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    # 保存视频为网格和 MP4 格式
                    save_video_as_grid_and_mp4(samples, path, num_frames // fps, fps, args, k)
# 定义广播批处理函数
def broad_cast_batch(batch):
    # 获取模型并行世界的大小
    mp_size = mpu.get_model_parallel_world_size()
    # 获取全局进程的排名
    global_rank = torch.distributed.get_rank() // mp_size
    # 计算源进程的索引
    src = global_rank * mp_size

    # 如果批次中存在 mp4 数据，则获取各数据的形状
    if batch["mp4"] is not None:
        broadcast_shape = [batch["mp4"].shape, batch["fps"].shape, batch["num_frames"].shape]
    else:
        # 否则设置为 None
        broadcast_shape = None

    # 准备要广播的对象列表，包括文本和形状信息
    txt = [batch["txt"], broadcast_shape]
    # 广播对象列表到指定源
    torch.distributed.broadcast_object_list(txt, src=src, group=mpu.get_model_parallel_group())
    # 从广播结果中提取文本数据
    batch["txt"] = txt[0]

    # 获取广播后各数据的形状
    mp4_shape = txt[1][0]
    fps_shape = txt[1][1]
    num_frames_shape = txt[1][2]

    # 如果当前模型并行进程不是 0，则初始化对应的张量
    if mpu.get_model_parallel_rank() != 0:
        batch["mp4"] = torch.zeros(mp4_shape, device="cuda")
        batch["fps"] = torch.zeros(fps_shape, device="cuda", dtype=torch.long)
        batch["num_frames"] = torch.zeros(num_frames_shape, device="cuda", dtype=torch.long)

    # 广播 mp4 数据
    torch.distributed.broadcast(batch["mp4"], src=src, group=mpu.get_model_parallel_group())
    # 广播 fps 数据
    torch.distributed.broadcast(batch["fps"], src=src, group=mpu.get_model_parallel_group())
    # 广播 num_frames 数据
    torch.distributed.broadcast(batch["num_frames"], src=src, group=mpu.get_model_parallel_group())
    # 返回处理后的批次数据
    return batch


# 定义前向评估步骤函数
def forward_step_eval(data_iterator, model, args, timers, only_log_video_latents=False, data_class=None):
    # 如果当前模型并行进程是 0，开始数据加载
    if mpu.get_model_parallel_rank() == 0:
        timers("data loader").start()  # 启动计时器
        batch_video = next(data_iterator)  # 获取下一个批次数据
        timers("data loader").stop()  # 停止计时器

        # 如果 mp4 数据的维度为 6，重新调整其形状
        if len(batch_video["mp4"].shape) == 6:
            b, v = batch_video["mp4"].shape[:2]  # 提取批次和视频维度
            batch_video["mp4"] = batch_video["mp4"].view(-1, *batch_video["mp4"].shape[2:])  # 扁平化 mp4 数据
            txt = []  # 初始化文本列表
            # 遍历批次和视频维度，构建文本列表
            for i in range(b):
                for j in range(v):
                    txt.append(batch_video["txt"][j][i])
            batch_video["txt"] = txt  # 更新批次中的文本数据

        # 将批次中的每个张量转移到 GPU
        for key in batch_video:
            if isinstance(batch_video[key], torch.Tensor):
                batch_video[key] = batch_video[key].cuda()  # 转移张量到 CUDA 设备
    else:
        # 如果当前进程不是 0，初始化空的批次数据
        batch_video = {"mp4": None, "fps": None, "num_frames": None, "txt": None}
    # 调用广播函数以同步批次数据
    broad_cast_batch(batch_video)
    # 如果数据并行进程是 0，记录视频数据
    if mpu.get_data_parallel_rank() == 0:
        log_video(batch_video, model, args, only_log_video_latents=only_log_video_latents)

    # 在批次中添加全局步骤信息
    batch_video["global_step"] = args.iteration
    # 进行共享步骤并计算损失
    loss, loss_dict = model.shared_step(batch_video)
    # 将损失字典中的 bfloat16 类型转换为 float32
    for k in loss_dict:
        if loss_dict[k].dtype == torch.bfloat16:
            loss_dict[k] = loss_dict[k].to(torch.float32)  # 转换数据类型
    return loss, loss_dict  # 返回损失和损失字典


# 定义前向步骤函数
def forward_step(data_iterator, model, args, timers, data_class=None):
    # 检查当前模型并行进程的排名是否为0
    if mpu.get_model_parallel_rank() == 0:
        # 启动计时器以记录数据加载时间
        timers("data loader").start()
        # 从数据迭代器中获取下一个批次的数据
        batch = next(data_iterator)
        # 停止计时器
        timers("data loader").stop()
        # 遍历批次中的每个键
        for key in batch:
            # 检查当前键对应的值是否为 PyTorch 张量
            if isinstance(batch[key], torch.Tensor):
                # 将张量移到 GPU 上
                batch[key] = batch[key].cuda()

        # 检查当前进程的分布式排名是否为0
        if torch.distributed.get_rank() == 0:
            # 检查保存目录下是否存在训练配置文件
            if not os.path.exists(os.path.join(args.save, "training_config.yaml")):
                # 加载基础配置文件，并将其存储在列表中
                configs = [OmegaConf.load(cfg) for cfg in args.base]
                # 合并所有基础配置
                config = OmegaConf.merge(*configs)
                # 创建保存目录（如果不存在）
                os.makedirs(args.save, exist_ok=True)
                # 将合并后的配置保存为 YAML 文件
                OmegaConf.save(config=config, f=os.path.join(args.save, "training_config.yaml"))
    else:
        # 如果当前不是模型并行进程0，则创建一个空的批次字典
        batch = {"mp4": None, "fps": None, "num_frames": None, "txt": None}

    # 在批次字典中添加全局步数
    batch["global_step"] = args.iteration

    # 广播批次数据到所有进程
    broad_cast_batch(batch)

    # 执行模型的共享步骤，计算损失和损失字典
    loss, loss_dict = model.shared_step(batch)

    # 返回计算的损失和损失字典
    return loss, loss_dict
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 检查环境变量中是否存在 OMPI_COMM_WORLD_LOCAL_RANK
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        # 将 OMPI_COMM_WORLD_LOCAL_RANK 的值赋给 LOCAL_RANK 环境变量
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        # 将 OMPI_COMM_WORLD_SIZE 的值赋给 WORLD_SIZE 环境变量
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        # 将 OMPI_COMM_WORLD_RANK 的值赋给 RANK 环境变量
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

    # 创建一个不带帮助信息的命令行参数解析器
    py_parser = argparse.ArgumentParser(add_help=False)
    # 解析已知和未知的命令行参数
    known, args_list = py_parser.parse_known_args()
    # 根据解析得到的参数列表获取具体参数
    args = get_args(args_list)
    # 将已知参数和具体参数合并到一个命名空间中
    args = argparse.Namespace(**vars(args), **vars(known))

    # 根据目标字符串获取数据类对象
    data_class = get_obj_from_str(args.data_config["target"])
    # 使用部分应用来创建数据集函数
    create_dataset_function = partial(data_class.create_dataset_function, **args.data_config["params"])

    # 导入 YAML 库
    import yaml

    # 初始化配置列表
    configs = []
    # 遍历基础配置文件列表
    for config in args.base:
        # 以只读模式打开基础配置文件
        with open(config, "r") as f:
            # 安全加载 YAML 文件内容为字典
            base_config = yaml.safe_load(f)
        # 将加载的基础配置添加到配置列表中
        configs.append(base_config)
    # 将加载的配置列表赋给 args.log_config
    args.log_config = configs

    # 调用训练主函数，传入相关参数和函数
    training_main(
        args,
        model_cls=SATVideoDiffusionEngine,
        forward_step_function=partial(forward_step, data_class=data_class),
        forward_step_eval=partial(
            forward_step_eval, data_class=data_class, only_log_video_latents=args.only_log_video_latents
        ),
        create_dataset_function=create_dataset_function,
    )
```

# `.\cogvideo-finetune\sat\vae_modules\attention.py`

```py
# 导入数学库
import math
# 从 inspect 模块导入 isfunction 函数，用于检查对象是否为函数
from inspect import isfunction
# 导入 Any 和 Optional 类型注解
from typing import Any, Optional

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的功能性操作
import torch.nn.functional as F
# 从 einops 库导入 rearrange 和 repeat 函数
from einops import rearrange, repeat
# 导入版本控制工具
from packaging import version
# 导入 PyTorch 的神经网络模块
from torch import nn

# 检查 PyTorch 版本是否大于或等于 2.0.0
if version.parse(torch.__version__) >= version.parse("2.0.0"):
    # 设置 SDP_IS_AVAILABLE 为 True，表示 SDP 后端可用
    SDP_IS_AVAILABLE = True
    # 从 CUDA 后端导入 SDPBackend 和 sdp_kernel
    from torch.backends.cuda import SDPBackend, sdp_kernel

    # 定义后端配置映射
    BACKEND_MAP = {
        SDPBackend.MATH: {
            "enable_math": True,  # 启用数学模式
            "enable_flash": False,  # 禁用闪电模式
            "enable_mem_efficient": False,  # 禁用内存高效模式
        },
        SDPBackend.FLASH_ATTENTION: {
            "enable_math": False,  # 禁用数学模式
            "enable_flash": True,  # 启用闪电模式
            "enable_mem_efficient": False,  # 禁用内存高效模式
        },
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False,  # 禁用数学模式
            "enable_flash": False,  # 禁用闪电模式
            "enable_mem_efficient": True,  # 启用内存高效模式
        },
        None: {"enable_math": True, "enable_flash": True, "enable_mem_efficient": True},  # 默认情况下启用所有模式
    }
else:
    # 从上下文管理模块导入 nullcontext
    from contextlib import nullcontext

    # 设置 SDP_IS_AVAILABLE 为 False，表示 SDP 后端不可用
    SDP_IS_AVAILABLE = False
    # 将 sdp_kernel 设置为 nullcontext
    sdp_kernel = nullcontext
    # 打印警告信息，提示用户升级 PyTorch 版本
    print(
        f"No SDP backend available, likely because you are running in pytorch versions < 2.0. In fact, "
        f"you are using PyTorch {torch.__version__}. You might want to consider upgrading."
    )

# 尝试导入 xformers 和 xformers.ops 模块
try:
    import xformers
    import xformers.ops

    # 如果导入成功，设置 XFORMERS_IS_AVAILABLE 为 True
    XFORMERS_IS_AVAILABLE = True
# 如果导入失败，设置 XFORMERS_IS_AVAILABLE 为 False，并打印提示信息
except:
    XFORMERS_IS_AVAILABLE = False
    print("no module 'xformers'. Processing without...")

# 从 modules.utils 模块导入 checkpoint 函数
from modules.utils import checkpoint


# 定义一个检查值是否存在的函数
def exists(val):
    # 返回值是否不为 None
    return val is not None


# 定义一个去重函数
def uniq(arr):
    # 将数组转换为字典以去重，然后返回字典的键
    return {el: True for el in arr}.keys()


# 定义一个返回默认值的函数
def default(val, d):
    # 如果 val 存在，返回 val
    if exists(val):
        return val
    # 否则返回 d 的结果，如果 d 是函数则调用它
    return d() if isfunction(d) else d


# 定义一个返回最大负值的函数
def max_neg_value(t):
    # 返回指定数据类型的最大负值
    return -torch.finfo(t.dtype).max


# 定义一个初始化张量的函数
def init_(tensor):
    # 获取张量的最后一个维度
    dim = tensor.shape[-1]
    # 计算标准差
    std = 1 / math.sqrt(dim)
    # 在[-std, std]范围内均匀初始化张量
    tensor.uniform_(-std, std)
    # 返回初始化后的张量
    return tensor


# 定义一个前馈神经网络类
class GEGLU(nn.Module):
    # 初始化方法
    def __init__(self, dim_in, dim_out):
        super().__init__()  # 调用父类的初始化方法
        # 创建线性变换层，输出维度为输入维度的两倍
        self.proj = nn.Linear(dim_in, dim_out * 2)

    # 前向传播方法
    def forward(self, x):
        # 将输入 x 通过线性层投影并拆分为两部分
        x, gate = self.proj(x).chunk(2, dim=-1)
        # 返回 x 和经过 GELU 激活的 gate 的乘积
        return x * F.gelu(gate)


# 定义一个前馈神经网络类
class FeedForward(nn.Module):
    # 初始化方法
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()  # 调用父类的初始化方法
        # 计算内部维度
        inner_dim = int(dim * mult)
        # 如果 dim_out 为空，设置为 dim
        dim_out = default(dim_out, dim)
        # 根据 glu 参数决定使用哪种输入变换
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        # 创建包含输入变换、dropout 和线性变换的序列模型
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    # 前向传播方法
    def forward(self, x):
        # 返回网络的输出
        return self.net(x)


# 定义一个将模块参数归零的函数
def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    # 遍历模块的所有参数
    for p in module.parameters():
        # 断开与计算图的联系并将参数归零
        p.detach().zero_()
    # 返回修改后的模块
    return module


# 定义一个归一化函数
def Normalize(in_channels):
    # 返回 GroupNorm 实例，用于对输入通道进行归一化
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# 定义一个线性注意力类
class LinearAttention(nn.Module):
    # 初始化方法，设置维度、头数和每个头的维度
        def __init__(self, dim, heads=4, dim_head=32):
            # 调用父类的初始化方法
            super().__init__()
            # 存储头数
            self.heads = heads
            # 计算隐藏层的维度，即头数与每个头的维度的乘积
            hidden_dim = dim_head * heads
            # 创建一个卷积层，将输入通道数转换为三倍的隐藏维度，用于生成查询、键和值
            self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
            # 创建一个卷积层，将隐藏维度转换回原始的输入通道数
            self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    
        # 前向传播方法，处理输入数据
        def forward(self, x):
            # 获取输入的批次大小、通道数、高度和宽度
            b, c, h, w = x.shape
            # 通过卷积层生成查询、键和值的组合
            qkv = self.to_qkv(x)
            # 重新排列 qkv 数据，使其分开为 q、k 和 v，并按头数分组
            q, k, v = rearrange(qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3)
            # 对键进行 softmax 操作，计算注意力权重
            k = k.softmax(dim=-1)
            # 使用爱因斯坦求和约定计算上下文，即加权后的值
            context = torch.einsum("bhdn,bhen->bhde", k, v)
            # 根据查询和上下文计算输出
            out = torch.einsum("bhde,bhdn->bhen", context, q)
            # 重新排列输出数据，恢复到原始形状
            out = rearrange(out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w)
            # 返回最终的输出结果
            return self.to_out(out)
# 定义一个空间自注意力类，继承自 nn.Module
class SpatialSelfAttention(nn.Module):
    # 初始化方法，接受输入通道数
    def __init__(self, in_channels):
        # 调用父类构造函数
        super().__init__()
        # 保存输入通道数
        self.in_channels = in_channels

        # 创建归一化层
        self.norm = Normalize(in_channels)
        # 创建查询卷积层
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建键卷积层
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建值卷积层
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 创建输出投影卷积层
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播方法
    def forward(self, x):
        # 初始化 h_ 为输入 x
        h_ = x
        # 对输入进行归一化
        h_ = self.norm(h_)
        # 计算查询
        q = self.q(h_)
        # 计算键
        k = self.k(h_)
        # 计算值
        v = self.v(h_)

        # 计算注意力
        b, c, h, w = q.shape  # 获取批量大小、通道数、高度和宽度
        # 重新排列查询以便进行矩阵乘法
        q = rearrange(q, "b c h w -> b (h w) c")
        # 重新排列键以便进行矩阵乘法
        k = rearrange(k, "b c h w -> b c (h w)")
        # 计算注意力权重
        w_ = torch.einsum("bij,bjk->bik", q, k)

        # 缩放注意力权重
        w_ = w_ * (int(c) ** (-0.5))
        # 应用 softmax 得到注意力分布
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # 处理值
        v = rearrange(v, "b c h w -> b c (h w)")  # 重新排列值
        w_ = rearrange(w_, "b i j -> b j i")  # 重新排列注意力权重
        # 根据注意力权重对值进行加权
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        # 重新排列以匹配输出形状
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        # 通过输出卷积层进行投影
        h_ = self.proj_out(h_)

        # 返回原始输入与输出的和
        return x + h_


# 定义一个交叉注意力类，继承自 nn.Module
class CrossAttention(nn.Module):
    # 初始化方法，接受多个参数
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        backend=None,
    ):
        # 调用父类构造函数
        super().__init__()
        # 计算内部维度
        inner_dim = dim_head * heads
        # 设置上下文维度的默认值
        context_dim = default(context_dim, query_dim)

        # 计算缩放因子
        self.scale = dim_head**-0.5
        # 保存头的数量
        self.heads = heads

        # 创建查询线性层
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # 创建键线性层
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        # 创建值线性层
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 创建输出线性层及 dropout
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        # 保存后端
        self.backend = backend

    # 前向传播方法
    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        h = self.heads  # 获取头的数量，用于后续的多头注意力机制

        if additional_tokens is not None:  # 检查是否有额外的标记
            # 获取输出序列开头的掩蔽标记数量
            n_tokens_to_mask = additional_tokens.shape[1]  
            # 将额外的标记添加到输入序列前面
            x = torch.cat([additional_tokens, x], dim=1)  

        # 将输入 x 转换为查询向量 q
        q = self.to_q(x)  
        # 默认上下文为 x
        context = default(context, x)  
        # 将上下文转换为键向量 k
        k = self.to_k(context)  
        # 将上下文转换为值向量 v
        v = self.to_v(context)  

        if n_times_crossframe_attn_in_self:  # 检查是否需要进行跨帧注意力机制
            # 按照文献中的方法重新编程跨帧注意力
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0  # 确保输入长度是跨帧次数的整数倍
            n_cp = x.shape[0] // n_times_crossframe_attn_in_self  # 计算每个跨帧的大小
            # 根据跨帧次数重复键向量
            k = repeat(k[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)  
            # 根据跨帧次数重复值向量
            v = repeat(v[::n_times_crossframe_attn_in_self], "b ... -> (b n) ...", n=n_cp)  

        # 将 q, k, v 重新排列为适合多头注意力的形状
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))  

        # old
        """
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale  # 计算查询和键的相似度
        del q, k  # 删除不再需要的 q 和 k

        if exists(mask):  # 检查是否存在掩蔽
            mask = rearrange(mask, 'b ... -> b (...)')  # 重新排列掩蔽形状
            max_neg_value = -torch.finfo(sim.dtype).max  # 获取最大负值，用于掩蔽
            mask = repeat(mask, 'b j -> (b h) () j', h=h)  # 将掩蔽重复到多头
            sim.masked_fill_(~mask, max_neg_value)  # 将不需要的部分填充为最大负值

        # 计算注意力分布
        sim = sim.softmax(dim=-1)  

        out = einsum('b i j, b j d -> b i d', sim, v)  # 计算最终输出
        """
        # new
        with sdp_kernel(**BACKEND_MAP[self.backend]):  # 使用指定后端的 SDP 核心
            # print("dispatching into backend", self.backend, "q/k/v shape: ", q.shape, k.shape, v.shape)  # 调试信息，打印 q, k, v 的形状
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # 计算缩放点积注意力，默认缩放因子为 dim_head ** -0.5

        del q, k, v  # 删除 q, k, v，释放内存
        out = rearrange(out, "b h n d -> b n (h d)", h=h)  # 将输出重新排列为合适的形状

        if additional_tokens is not None:  # 如果存在额外的标记
            # 移除额外的标记
            out = out[:, n_tokens_to_mask:]  
        return self.to_out(out)  # 将输出转换为最终输出格式并返回
# 定义一个内存高效的交叉注意力模块，继承自 nn.Module
class MemoryEfficientCrossAttention(nn.Module):
    # 初始化方法，设置参数并打印相关信息
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs):
        # 调用父类的初始化方法
        super().__init__()
        # 打印类名及查询维度、上下文维度、头数和维度信息
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
            f"{heads} heads with a dimension of {dim_head}."
        )
        # 计算每个头的内部维度
        inner_dim = dim_head * heads
        # 如果上下文维度为 None，则默认为查询维度
        context_dim = default(context_dim, query_dim)

        # 保存头数和每个头的维度
        self.heads = heads
        self.dim_head = dim_head

        # 定义查询、键、值的线性变换，无偏置项
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # 定义最终输出的线性变换和 dropout 层
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        # 初始化注意力操作为 None
        self.attention_op: Optional[Any] = None

    # 前向传播方法，接收输入数据和可选参数
    def forward(
        self,
        x,
        context=None,
        mask=None,
        additional_tokens=None,
        n_times_crossframe_attn_in_self=0,
    ):
        # 检查是否有额外的令牌
        if additional_tokens is not None:
            # 获取输出序列开头被遮蔽的令牌数量
            n_tokens_to_mask = additional_tokens.shape[1]
            # 将额外的令牌与输入张量在列上拼接
            x = torch.cat([additional_tokens, x], dim=1)
        # 将输入张量转换为查询张量
        q = self.to_q(x)
        # 默认上下文为输入张量
        context = default(context, x)
        # 将上下文转换为键张量
        k = self.to_k(context)
        # 将上下文转换为值张量
        v = self.to_v(context)

        # 检查是否需要在自注意力中进行跨帧注意力
        if n_times_crossframe_attn_in_self:
            # 按照文献重新编程跨帧注意力
            assert x.shape[0] % n_times_crossframe_attn_in_self == 0
            # k的形状将被重复n_times_crossframe_attn_in_self次
            k = repeat(
                k[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )
            # v的形状将被重复n_times_crossframe_attn_in_self次
            v = repeat(
                v[::n_times_crossframe_attn_in_self],
                "b ... -> (b n) ...",
                n=n_times_crossframe_attn_in_self,
            )

        # 获取查询张量的批次大小和其他维度
        b, _, _ = q.shape
        # 对q, k, v进行处理以匹配注意力机制的要求
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # 实际计算注意力机制
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        # TODO: 将这个直接用于注意力操作作为偏差
        if exists(mask):
            # 如果存在掩码，抛出未实现异常
            raise NotImplementedError
        # 调整输出张量的形状以符合后续处理
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        # 如果有额外的令牌，去除它们
        if additional_tokens is not None:
            out = out[:, n_tokens_to_mask:]
        # 将输出张量转换为最终输出
        return self.to_out(out)
# 基础变换器块，继承自 nn.Module
class BasicTransformerBlock(nn.Module):
    # 定义注意力模式的映射
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # 标准注意力
        "softmax-xformers": MemoryEfficientCrossAttention,  # 节省内存的注意力
    }

    # 初始化方法，接收多个参数
    def __init__(
        self,
        dim,  # 特征维度
        n_heads,  # 注意力头数量
        d_head,  # 每个注意力头的维度
        dropout=0.0,  # dropout比率
        context_dim=None,  # 上下文维度
        gated_ff=True,  # 是否使用门控前馈网络
        checkpoint=True,  # 是否使用检查点
        disable_self_attn=False,  # 是否禁用自注意力
        attn_mode="softmax",  # 注意力模式
        sdp_backend=None,  # 后端配置
    ):
        # 调用父类初始化方法
        super().__init__()
        # 确保指定的注意力模式有效
        assert attn_mode in self.ATTENTION_MODES
        # 如果选择的模式不可用，则回退到默认模式
        if attn_mode != "softmax" and not XFORMERS_IS_AVAILABLE:
            print(
                f"Attention mode '{attn_mode}' is not available. Falling back to native attention. "
                f"This is not a problem in Pytorch >= 2.0. FYI, you are running with PyTorch version {torch.__version__}"
            )
            attn_mode = "softmax"
        # 如果选择的是标准模式，但不再支持，则进行适当处理
        elif attn_mode == "softmax" and not SDP_IS_AVAILABLE:
            print("We do not support vanilla attention anymore, as it is too expensive. Sorry.")
            # 确保 xformers 可用
            if not XFORMERS_IS_AVAILABLE:
                assert False, "Please install xformers via e.g. 'pip install xformers==0.0.16'"
            else:
                print("Falling back to xformers efficient attention.")
                attn_mode = "softmax-xformers"
        # 根据选择的模式获取注意力类
        attn_cls = self.ATTENTION_MODES[attn_mode]
        # 检查 PyTorch 版本以确保后端有效
        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            assert sdp_backend is None or isinstance(sdp_backend, SDPBackend)
        else:
            assert sdp_backend is None
        # 设置自注意力禁用标志
        self.disable_self_attn = disable_self_attn
        # 创建第一个注意力层
        self.attn1 = attn_cls(
            query_dim=dim,  # 查询维度
            heads=n_heads,  # 注意力头数量
            dim_head=d_head,  # 每个头的维度
            dropout=dropout,  # dropout比率
            context_dim=context_dim if self.disable_self_attn else None,  # 上下文维度，若禁用自注意力则为None
            backend=sdp_backend,  # 后端配置
        )  # 如果未禁用自注意力，则为自注意力层
        # 创建前馈网络
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # 创建第二个注意力层
        self.attn2 = attn_cls(
            query_dim=dim,  # 查询维度
            context_dim=context_dim,  # 上下文维度
            heads=n_heads,  # 注意力头数量
            dim_head=d_head,  # 每个头的维度
            dropout=dropout,  # dropout比率
            backend=sdp_backend,  # 后端配置
        )  # 如果上下文为None，则为自注意力层
        # 创建层归一化层
        self.norm1 = nn.LayerNorm(dim)  # 第一层归一化
        self.norm2 = nn.LayerNorm(dim)  # 第二层归一化
        self.norm3 = nn.LayerNorm(dim)  # 第三层归一化
        # 设置检查点标志
        self.checkpoint = checkpoint
        # 如果启用检查点，则打印提示
        if self.checkpoint:
            print(f"{self.__class__.__name__} is using checkpointing")
    # 前向传播方法，接收输入和可选的上下文、附加标记及其他参数
    def forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        # 初始化参数字典，包含输入 x
        kwargs = {"x": x}
    
        # 如果上下文不为 None，将其添加到参数字典
        if context is not None:
            kwargs.update({"context": context})
    
        # 如果附加标记不为 None，将其添加到参数字典
        if additional_tokens is not None:
            kwargs.update({"additional_tokens": additional_tokens})
    
        # 如果跨帧自注意力次数大于 0，将其添加到参数字典
        if n_times_crossframe_attn_in_self:
            kwargs.update({"n_times_crossframe_attn_in_self": n_times_crossframe_attn_in_self})
    
        # 调用检查点函数，执行前向传播，返回计算结果
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
    
    # 定义实际的前向传播逻辑
    def _forward(self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0):
        # 对输入 x 进行归一化处理并通过第一个注意力层，结合上下文和附加标记
        x = (
            self.attn1(
                self.norm1(x),
                context=context if self.disable_self_attn else None,
                additional_tokens=additional_tokens,
                n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self if not self.disable_self_attn else 0,
            )
            + x  # 将原始输入加到输出中，形成残差连接
        )
        # 对处理后的 x 进行归一化处理，并通过第二个注意力层
        x = self.attn2(self.norm2(x), context=context, additional_tokens=additional_tokens) + x  # 残差连接
        # 通过前馈层处理，完成归一化后加到输入上
        x = self.ff(self.norm3(x)) + x  # 残差连接
        # 返回最终的输出
        return x
# 定义一个单层基本变换器块，继承自 nn.Module
class BasicTransformerSingleLayerBlock(nn.Module):
    # 定义注意力模式的字典，映射名称到对应的注意力类
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # 标准注意力
        "softmax-xformers": MemoryEfficientCrossAttention,  # 在 A100 上速度不如上面的版本
        # (待办：可能依赖于 head_dim，检查，降级为针对 dim!=[16,32,64,128] 的半优化内核)
    }

    # 初始化函数，定义构造函数的参数
    def __init__(
        self,
        dim,  # 输入特征维度
        n_heads,  # 注意力头的数量
        d_head,  # 每个注意力头的维度
        dropout=0.0,  # dropout 概率
        context_dim=None,  # 上下文维度（可选）
        gated_ff=True,  # 是否使用门控前馈网络
        checkpoint=True,  # 是否使用检查点以节省内存
        attn_mode="softmax",  # 使用的注意力模式
    ):
        # 调用父类的构造函数
        super().__init__()
        # 确保给定的注意力模式在允许的模式中
        assert attn_mode in self.ATTENTION_MODES
        # 根据模式获取对应的注意力类
        attn_cls = self.ATTENTION_MODES[attn_mode]
        # 初始化注意力层
        self.attn1 = attn_cls(
            query_dim=dim,  # 查询维度
            heads=n_heads,  # 头数量
            dim_head=d_head,  # 每个头的维度
            dropout=dropout,  # dropout 概率
            context_dim=context_dim,  # 上下文维度
        )
        # 初始化前馈网络
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # 初始化层归一化
        self.norm1 = nn.LayerNorm(dim)
        # 初始化第二个层归一化
        self.norm2 = nn.LayerNorm(dim)
        # 保存检查点标志
        self.checkpoint = checkpoint

    # 前向传播函数
    def forward(self, x, context=None):
        # 使用检查点机制调用 _forward 方法
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    # 实际的前向传播逻辑
    def _forward(self, x, context=None):
        # 先应用注意力层，结合残差连接
        x = self.attn1(self.norm1(x), context=context) + x
        # 应用前馈网络，再结合残差连接
        x = self.ff(self.norm2(x)) + x
        # 返回输出
        return x


# 定义用于图像数据的变换器块，继承自 nn.Module
class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    # 初始化函数，定义构造函数的参数
    def __init__(
        self,
        in_channels,  # 输入通道数
        n_heads,  # 注意力头的数量
        d_head,  # 每个头的维度
        depth=1,  # 堆叠的层数
        dropout=0.0,  # dropout 概率
        context_dim=None,  # 上下文维度（可选）
        disable_self_attn=False,  # 是否禁用自注意力
        use_linear=False,  # 是否使用线性层以提高效率
        attn_type="softmax",  # 使用的注意力类型
        use_checkpoint=True,  # 是否使用检查点以节省内存
        # sdp_backend=SDPBackend.FLASH_ATTENTION  # 备用的 sdp 后端
        sdp_backend=None,  # 指定 sdp 后端（默认为 None）
    ):
        # 调用父类的构造函数，初始化父类属性
        super().__init__()
        # 打印当前类的名称、深度、输入通道数和头的数量
        print(f"constructing {self.__class__.__name__} of depth {depth} w/ {in_channels} channels and {n_heads} heads")
        # 从 omegaconf 库导入 ListConfig 类
        from omegaconf import ListConfig

        # 如果 context_dim 存在且不是列表或 ListConfig 类型，则将其转换为列表
        if exists(context_dim) and not isinstance(context_dim, (list, ListConfig)):
            context_dim = [context_dim]
        # 如果 context_dim 存在且是列表类型
        if exists(context_dim) and isinstance(context_dim, list):
            # 如果给定的深度与 context_dim 的长度不匹配
            if depth != len(context_dim):
                # 打印警告信息，提示深度与上下文维度不匹配，并重设上下文维度
                print(
                    f"WARNING: {self.__class__.__name__}: Found context dims {context_dim} of depth {len(context_dim)}, "
                    f"which does not match the specified 'depth' of {depth}. Setting context_dim to {depth * [context_dim[0]]} now."
                )
                # 断言所有上下文维度相同，以便自动匹配深度
                assert all(
                    map(lambda x: x == context_dim[0], context_dim)
                ), "need homogenous context_dim to match depth automatically"
                # 将上下文维度设置为深度乘以第一个上下文维度
                context_dim = depth * [context_dim[0]]
        # 如果上下文维度为 None，创建一个包含 None 的列表，长度为深度
        elif context_dim is None:
            context_dim = [None] * depth
        # 设置输入通道数的属性
        self.in_channels = in_channels
        # 计算内部维度，等于头的数量乘以每个头的维度
        inner_dim = n_heads * d_head
        # 创建归一化层
        self.norm = Normalize(in_channels)
        # 根据是否使用线性层选择不同的输入投影层
        if not use_linear:
            # 使用卷积层进行输入投影
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            # 使用线性层进行输入投影
            self.proj_in = nn.Linear(in_channels, inner_dim)

        # 创建变换器块的模块列表
        self.transformer_blocks = nn.ModuleList(
            [
                # 对于每一层深度，创建基本变换器块
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    sdp_backend=sdp_backend,
                )
                for d in range(depth)  # 遍历深度范围
            ]
        )
        # 根据是否使用线性层选择不同的输出投影层
        if not use_linear:
            # 使用零初始化的卷积层作为输出投影层
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            # 使用零初始化的线性层作为输出投影层
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        # 设置是否使用线性层的属性
        self.use_linear = use_linear
    # 前向传播函数，接受输入 x 和可选的上下文参数
        def forward(self, x, context=None):
            # 如果没有给定上下文，交叉注意力默认使用自注意力
            if not isinstance(context, list):
                context = [context]  # 确保上下文为列表形式
            # 获取输入 x 的形状信息
            b, c, h, w = x.shape
            # 保存原始输入以便后续使用
            x_in = x
            # 对输入 x 进行归一化处理
            x = self.norm(x)
            # 如果不使用线性变换，则进行输入投影
            if not self.use_linear:
                x = self.proj_in(x)
            # 重新排列 x 的形状，以适应后续处理
            x = rearrange(x, "b c h w -> b (h w) c").contiguous()
            # 如果使用线性变换，则进行输入投影
            if self.use_linear:
                x = self.proj_in(x)
            # 遍历每个变换块
            for i, block in enumerate(self.transformer_blocks):
                # 如果不是第一个块且上下文只有一个，则使用相同的上下文
                if i > 0 and len(context) == 1:
                    i = 0  # 每个块使用相同的上下文
                # 将 x 和对应的上下文传入当前块
                x = block(x, context=context[i])
            # 如果使用线性变换，则进行输出投影
            if self.use_linear:
                x = self.proj_out(x)
            # 重新排列 x 的形状，恢复到原始输入的形状
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
            # 如果不使用线性变换，则进行输出投影
            if not self.use_linear:
                x = self.proj_out(x)
            # 返回输出与原始输入的和
            return x + x_in
```