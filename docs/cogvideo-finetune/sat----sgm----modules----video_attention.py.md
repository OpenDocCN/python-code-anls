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