# `.\diffusers\models\attention_flax.py`

```py
# 版权声明，表明该代码的版权归 HuggingFace 团队所有
# 根据 Apache 2.0 许可证授权使用该文件，未遵守许可证不得使用
# 许可证获取链接
# 指出该软件是以“现状”分发，不附带任何明示或暗示的保证
# 具体的权限和限制请参见许可证

# 导入 functools 模块，用于函数式编程工具
import functools
# 导入 math 模块，提供数学相关的功能
import math

# 导入 flax.linen 模块，作为神经网络构建的工具
import flax.linen as nn
# 导入 jax 库，用于加速计算
import jax
# 导入 jax.numpy 模块，提供类似于 NumPy 的数组功能
import jax.numpy as jnp


def _query_chunk_attention(query, key, value, precision, key_chunk_size: int = 4096):
    """多头点积注意力，查询数目有限的实现。"""
    # 获取 key 的维度信息，包括 key 的数量、头数和特征维度
    num_kv, num_heads, k_features = key.shape[-3:]
    # 获取 value 的特征维度
    v_features = value.shape[-1]
    # 确保 key_chunk_size 不超过 num_kv
    key_chunk_size = min(key_chunk_size, num_kv)
    # 对查询进行缩放，防止数值溢出
    query = query / jnp.sqrt(k_features)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(query, key, value):
        # 计算查询和键之间的注意力权重
        attn_weights = jnp.einsum("...qhd,...khd->...qhk", query, key, precision=precision)

        # 获取每个查询的最大得分，用于数值稳定性
        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        # 计算最大得分的梯度不更新
        max_score = jax.lax.stop_gradient(max_score)
        # 计算经过 softmax 的注意力权重
        exp_weights = jnp.exp(attn_weights - max_score)

        # 计算加权后的值
        exp_values = jnp.einsum("...vhf,...qhv->...qhf", value, exp_weights, precision=precision)
        # 获取每个查询的最大得分
        max_score = jnp.einsum("...qhk->...qh", max_score)

        return (exp_values, exp_weights.sum(axis=-1), max_score)

    def chunk_scanner(chunk_idx):
        # 动态切片获取键的部分数据
        key_chunk = jax.lax.dynamic_slice(
            operand=key,
            start_indices=[0] * (key.ndim - 3) + [chunk_idx, 0, 0],  # [...,k,h,d]
            slice_sizes=list(key.shape[:-3]) + [key_chunk_size, num_heads, k_features],  # [...,k,h,d]
        )

        # 动态切片获取值的部分数据
        value_chunk = jax.lax.dynamic_slice(
            operand=value,
            start_indices=[0] * (value.ndim - 3) + [chunk_idx, 0, 0],  # [...,v,h,d]
            slice_sizes=list(value.shape[:-3]) + [key_chunk_size, num_heads, v_features],  # [...,v,h,d]
        )

        return summarize_chunk(query, key_chunk, value_chunk)

    # 对每个键块进行注意力计算
    chunk_values, chunk_weights, chunk_max = jax.lax.map(f=chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size))

    # 计算全局最大得分
    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    # 计算每个块与全局最大得分的差异
    max_diffs = jnp.exp(chunk_max - global_max)

    # 更新值和权重以便于归一化
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    # 计算所有块的总值和总权重
    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)

    # 返回归一化后的总值
    return all_values / all_weights


def jax_memory_efficient_attention(
    query, key, value, precision=jax.lax.Precision.HIGHEST, query_chunk_size: int = 1024, key_chunk_size: int = 4096
):
    r"""
    # Flax 实现的内存高效多头点积注意力机制，相关文献链接
    Flax Memory-efficient multi-head dot product attention. https://arxiv.org/abs/2112.05682v2
    # 相关 GitHub 项目链接
    https://github.com/AminRezaei0x443/memory-efficient-attention

    # 参数说明：
    # query: 输入的查询张量，形状为 (batch..., query_length, head, query_key_depth_per_head)
    Args:
        query (`jnp.ndarray`): (batch..., query_length, head, query_key_depth_per_head)
        # key: 输入的键张量，形状为 (batch..., key_value_length, head, query_key_depth_per_head)
        key (`jnp.ndarray`): (batch..., key_value_length, head, query_key_depth_per_head)
        # value: 输入的值张量，形状为 (batch..., key_value_length, head, value_depth_per_head)
        value (`jnp.ndarray`): (batch..., key_value_length, head, value_depth_per_head)
        # precision: 计算时的数值精度，默认值为 jax.lax.Precision.HIGHEST
        precision (`jax.lax.Precision`, *optional*, defaults to `jax.lax.Precision.HIGHEST`):
            numerical precision for computation
        # query_chunk_size: 将查询数组划分的块大小，必须能整除 query_length
        query_chunk_size (`int`, *optional*, defaults to 1024):
            chunk size to divide query array value must divide query_length equally without remainder
        # key_chunk_size: 将键和值数组划分的块大小，必须能整除 key_value_length
        key_chunk_size (`int`, *optional*, defaults to 4096):
            chunk size to divide key and value array value must divide key_value_length equally without remainder

    # 返回值为形状为 (batch..., query_length, head, value_depth_per_head) 的数组
    Returns:
        (`jnp.ndarray`) with shape of (batch..., query_length, head, value_depth_per_head)
    """
    # 获取查询张量的最后三个维度的大小
    num_q, num_heads, q_features = query.shape[-3:]

    # 定义一个函数，用于扫描处理每个查询块
    def chunk_scanner(chunk_idx, _):
        # 从查询数组中切片出当前块
        query_chunk = jax.lax.dynamic_slice(
            # 操作的对象是查询张量
            operand=query,
            # 起始索引，保持前面的维度不变，从 chunk_idx 开始切片
            start_indices=([0] * (query.ndim - 3)) + [chunk_idx, 0, 0],  # [...,q,h,d]
            # 切片的大小，前面的维度不变，后面根据块大小取最小值
            slice_sizes=list(query.shape[:-3]) + [min(query_chunk_size, num_q), num_heads, q_features],  # [...,q,h,d]
        )

        return (
            # 返回未使用的下一个块索引
            chunk_idx + query_chunk_size,  # unused ignore it
            # 调用注意力函数处理当前查询块
            _query_chunk_attention(
                query=query_chunk, key=key, value=value, precision=precision, key_chunk_size=key_chunk_size
            ),
        )

    # 使用 jax.lax.scan 进行块的扫描处理
    _, res = jax.lax.scan(
        f=chunk_scanner,  # 处理函数
        init=0,  # 初始化块索引为 0
        xs=None,  # 不需要额外的输入数据
        # 根据查询块大小计算要处理的块数
        length=math.ceil(num_q / query_chunk_size),  # start counter  # stop counter
    )

    # 将所有块的结果在第 -3 维度拼接在一起
    return jnp.concatenate(res, axis=-3)  # fuse the chunked result back
# 定义一个 Flax 的多头注意力模块，遵循文献中的描述
class FlaxAttention(nn.Module):
    r"""
    Flax多头注意力模块，详见： https://arxiv.org/abs/1706.03762

    参数：
        query_dim (:obj:`int`):
            输入隐藏状态的维度
        heads (:obj:`int`, *optional*, defaults to 8):
            注意力头的数量
        dim_head (:obj:`int`, *optional*, defaults to 64):
            每个头内隐藏状态的维度
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            dropout比率
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            启用内存高效注意力 https://arxiv.org/abs/2112.05682
        split_head_dim (`bool`, *optional*, defaults to `False`):
            是否将头维度拆分为自注意力计算的新轴。通常情况下，启用该标志可以加快Stable Diffusion 2.x和Stable Diffusion XL的计算速度。
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            参数的 `dtype`

    """

    # 定义输入参数的类型和默认值
    query_dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False
    dtype: jnp.dtype = jnp.float32

    # 设置模块的初始化函数
    def setup(self):
        # 计算内部维度为每个头的维度与头的数量的乘积
        inner_dim = self.dim_head * self.heads
        # 计算缩放因子
        self.scale = self.dim_head**-0.5

        # 创建权重矩阵，使用旧的命名 {to_q, to_k, to_v, to_out}
        self.query = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_q")
        # 创建键的权重矩阵
        self.key = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_k")
        # 创建值的权重矩阵
        self.value = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_v")

        # 创建输出的权重矩阵
        self.proj_attn = nn.Dense(self.query_dim, dtype=self.dtype, name="to_out_0")
        # 创建dropout层
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    # 将张量的头部维度重塑为批次维度
    def reshape_heads_to_batch_dim(self, tensor):
        # 解构张量的形状
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        # 重塑张量形状以分离头维度
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        # 转置张量的维度
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        # 进一步重塑为批次与头维度合并
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    # 将张量的批次维度重塑为头部维度
    def reshape_batch_dim_to_heads(self, tensor):
        # 解构张量的形状
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        # 重塑张量形状以合并批次与头维度
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        # 转置张量的维度
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        # 进一步重塑为合并批次与头维度
        tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

# 定义一个 Flax 基础变换器块层，使用 GLU 激活函数，详见：
class FlaxBasicTransformerBlock(nn.Module):
    r"""
    Flax 变换器块层，使用 `GLU` (门控线性单元) 激活函数，详见：
    https://arxiv.org/abs/1706.03762
    # 参数说明部分
    Parameters:
        dim (:obj:`int`):  # 内部隐藏状态的维度
            Inner hidden states dimension
        n_heads (:obj:`int`):  # 注意力头的数量
            Number of heads
        d_head (:obj:`int`):  # 每个头内部隐藏状态的维度
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):  # 随机失活率
            Dropout rate
        only_cross_attention (`bool`, defaults to `False`):  # 是否仅应用交叉注意力
            Whether to only apply cross attention.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):  # 参数数据类型
            Parameters `dtype`
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):  # 启用内存高效注意力
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        split_head_dim (`bool`, *optional*, defaults to `False`):  # 是否将头维度拆分为新轴
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
    """

    dim: int  # 内部隐藏状态维度的类型声明
    n_heads: int  # 注意力头数量的类型声明
    d_head: int  # 每个头的隐藏状态维度的类型声明
    dropout: float = 0.0  # 随机失活率的默认值
    only_cross_attention: bool = False  # 默认不只应用交叉注意力
    dtype: jnp.dtype = jnp.float32  # 默认数据类型为 jnp.float32
    use_memory_efficient_attention: bool = False  # 默认不启用内存高效注意力
    split_head_dim: bool = False  # 默认不拆分头维度

    def setup(self):
        # 设置自注意力（如果 only_cross_attention 为 True，则为交叉注意力）
        self.attn1 = FlaxAttention(
            self.dim,  # 传入的内部隐藏状态维度
            self.n_heads,  # 传入的注意力头数量
            self.d_head,  # 传入的每个头的隐藏状态维度
            self.dropout,  # 传入的随机失活率
            self.use_memory_efficient_attention,  # 是否使用内存高效注意力
            self.split_head_dim,  # 是否拆分头维度
            dtype=self.dtype,  # 传入的数据类型
        )
        # 设置交叉注意力
        self.attn2 = FlaxAttention(
            self.dim,  # 传入的内部隐藏状态维度
            self.n_heads,  # 传入的注意力头数量
            self.d_head,  # 传入的每个头的隐藏状态维度
            self.dropout,  # 传入的随机失活率
            self.use_memory_efficient_attention,  # 是否使用内存高效注意力
            self.split_head_dim,  # 是否拆分头维度
            dtype=self.dtype,  # 传入的数据类型
        )
        # 设置前馈网络
        self.ff = FlaxFeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)  # 前馈网络初始化
        # 设置第一个归一化层
        self.norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)  # 归一化层初始化
        # 设置第二个归一化层
        self.norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)  # 归一化层初始化
        # 设置第三个归一化层
        self.norm3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)  # 归一化层初始化
        # 设置丢弃层
        self.dropout_layer = nn.Dropout(rate=self.dropout)  # 丢弃层初始化
    # 定义可调用对象，接收隐藏状态、上下文和确定性标志
    def __call__(self, hidden_states, context, deterministic=True):
        # 保存输入的隐藏状态以供后续残差连接使用
        residual = hidden_states
        # 如果仅执行交叉注意力，进行相关的处理
        if self.only_cross_attention:
            hidden_states = self.attn1(self.norm1(hidden_states), context, deterministic=deterministic)
        else:
            # 否则执行自注意力处理
            hidden_states = self.attn1(self.norm1(hidden_states), deterministic=deterministic)
        # 将自注意力的输出与输入的残差相加
        hidden_states = hidden_states + residual
    
        # 交叉注意力处理
        residual = hidden_states
        # 处理交叉注意力
        hidden_states = self.attn2(self.norm2(hidden_states), context, deterministic=deterministic)
        # 将交叉注意力的输出与输入的残差相加
        hidden_states = hidden_states + residual
    
        # 前馈网络处理
        residual = hidden_states
        # 应用前馈网络
        hidden_states = self.ff(self.norm3(hidden_states), deterministic=deterministic)
        # 将前馈网络的输出与输入的残差相加
        hidden_states = hidden_states + residual
    
        # 返回经过 dropout 处理的最终隐藏状态
        return self.dropout_layer(hidden_states, deterministic=deterministic)
# 定义一个二维的 Flax Transformer 模型，继承自 nn.Module
class FlaxTransformer2DModel(nn.Module):
    r"""
    A Spatial Transformer layer with Gated Linear Unit (GLU) activation function as described in:
    https://arxiv.org/pdf/1506.02025.pdf

    文档字符串，描述该类的功能和参数。

    Parameters:
        in_channels (:obj:`int`):
            Input number of channels
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        depth (:obj:`int`, *optional*, defaults to 1):
            Number of transformers block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_linear_projection (`bool`, defaults to `False`): tbd
        only_cross_attention (`bool`, defaults to `False`): tbd
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
    """
    
    # 定义输入通道数
    in_channels: int
    # 定义头的数量
    n_heads: int
    # 定义每个头的隐藏状态维度
    d_head: int
    # 定义 Transformer 块的数量，默认为 1
    depth: int = 1
    # 定义 Dropout 率，默认为 0.0
    dropout: float = 0.0
    # 定义是否使用线性投影，默认为 False
    use_linear_projection: bool = False
    # 定义是否仅使用交叉注意力，默认为 False
    only_cross_attention: bool = False
    # 定义参数的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32
    # 定义是否使用内存高效注意力，默认为 False
    use_memory_efficient_attention: bool = False
    # 定义是否将头维度拆分为新的轴，默认为 False
    split_head_dim: bool = False

    # 设置模型的组件
    def setup(self):
        # 使用 Group Normalization 规范化层，分组数为 32，epsilon 为 1e-5
        self.norm = nn.GroupNorm(num_groups=32, epsilon=1e-5)

        # 计算内部维度为头的数量乘以每个头的维度
        inner_dim = self.n_heads * self.d_head
        # 根据是否使用线性投影选择输入层
        if self.use_linear_projection:
            # 创建一个线性投影层，输出维度为 inner_dim，数据类型为 self.dtype
            self.proj_in = nn.Dense(inner_dim, dtype=self.dtype)
        else:
            # 创建一个卷积层，输出维度为 inner_dim，卷积核大小为 (1, 1)，步幅为 (1, 1)，填充方式为 "VALID"，数据类型为 self.dtype
            self.proj_in = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

        # 创建一系列 Transformer 块，数量为 depth
        self.transformer_blocks = [
            FlaxBasicTransformerBlock(
                inner_dim,
                self.n_heads,
                self.d_head,
                dropout=self.dropout,
                only_cross_attention=self.only_cross_attention,
                dtype=self.dtype,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                split_head_dim=self.split_head_dim,
            )
            for _ in range(self.depth)  # 循环生成每个 Transformer 块
        ]

        # 根据是否使用线性投影选择输出层
        if self.use_linear_projection:
            # 创建一个线性投影层，输出维度为 inner_dim，数据类型为 self.dtype
            self.proj_out = nn.Dense(inner_dim, dtype=self.dtype)
        else:
            # 创建一个卷积层，输出维度为 inner_dim，卷积核大小为 (1, 1)，步幅为 (1, 1)，填充方式为 "VALID"，数据类型为 self.dtype
            self.proj_out = nn.Conv(
                inner_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

        # 创建一个 Dropout 层，Dropout 率为 self.dropout
        self.dropout_layer = nn.Dropout(rate=self.dropout)
    # 定义可调用对象的方法，接收隐藏状态、上下文和确定性标志
    def __call__(self, hidden_states, context, deterministic=True):
        # 解构隐藏状态的形状，获取批量大小、高度、宽度和通道数
        batch, height, width, channels = hidden_states.shape
        # 保存原始隐藏状态以用于残差连接
        residual = hidden_states
        # 对隐藏状态进行归一化处理
        hidden_states = self.norm(hidden_states)
        # 如果使用线性投影，则重塑隐藏状态
        if self.use_linear_projection:
            # 将隐藏状态重塑为(batch, height * width, channels)的形状
            hidden_states = hidden_states.reshape(batch, height * width, channels)
            # 应用输入投影
            hidden_states = self.proj_in(hidden_states)
        else:
            # 直接应用输入投影
            hidden_states = self.proj_in(hidden_states)
            # 将隐藏状态重塑为(batch, height * width, channels)的形状
            hidden_states = hidden_states.reshape(batch, height * width, channels)
    
        # 遍历每个变换块，更新隐藏状态
        for transformer_block in self.transformer_blocks:
            # 通过变换块处理隐藏状态和上下文
            hidden_states = transformer_block(hidden_states, context, deterministic=deterministic)
    
        # 如果使用线性投影，则先应用输出投影
        if self.use_linear_projection:
            hidden_states = self.proj_out(hidden_states)
            # 将隐藏状态重塑回原来的形状
            hidden_states = hidden_states.reshape(batch, height, width, channels)
        else:
            # 先重塑隐藏状态
            hidden_states = hidden_states.reshape(batch, height, width, channels)
            # 再应用输出投影
            hidden_states = self.proj_out(hidden_states)
    
        # 将隐藏状态与原始状态相加，实现残差连接
        hidden_states = hidden_states + residual
        # 返回经过dropout层处理后的隐藏状态
        return self.dropout_layer(hidden_states, deterministic=deterministic)
# 定义一个 Flax 的前馈神经网络模块，继承自 nn.Module
class FlaxFeedForward(nn.Module):
    r"""
    Flax 模块封装了两个线性层，中间由一个非线性激活函数分隔。它是 PyTorch 的
    [`FeedForward`] 类的对应物，具有以下简化：
    - 激活函数目前硬编码为门控线性单元，来自：
    https://arxiv.org/abs/2002.05202
    - `dim_out` 等于 `dim`。
    - 隐藏维度的数量硬编码为 `dim * 4` 在 [`FlaxGELU`] 中。

    参数：
        dim (:obj:`int`):
            内部隐藏状态的维度
        dropout (:obj:`float`, *可选*, 默认为 0.0):
            丢弃率
        dtype (:obj:`jnp.dtype`, *可选*, 默认为 jnp.float32):
            参数的数据类型
    """

    # 定义类属性 dim、dropout 和 dtype，分别表示维度、丢弃率和数据类型
    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    # 设置方法，初始化网络层
    def setup(self):
        # 第二个线性层暂时称为 net_2，以匹配顺序层的索引
        self.net_0 = FlaxGEGLU(self.dim, self.dropout, self.dtype)  # 初始化 FlaxGEGLU 网络
        self.net_2 = nn.Dense(self.dim, dtype=self.dtype)  # 初始化线性层

    # 定义前向传播方法
    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.net_0(hidden_states, deterministic=deterministic)  # 通过 net_0 处理隐藏状态
        hidden_states = self.net_2(hidden_states)  # 通过 net_2 处理隐藏状态
        return hidden_states  # 返回处理后的隐藏状态


# 定义 Flax 的 GEGLU 激活层，继承自 nn.Module
class FlaxGEGLU(nn.Module):
    r"""
    Flax 实现的线性层后跟门控线性单元激活函数变体，来自
    https://arxiv.org/abs/2002.05202。

    参数：
        dim (:obj:`int`):
            输入隐藏状态的维度
        dropout (:obj:`float`, *可选*, 默认为 0.0):
            丢弃率
        dtype (:obj:`jnp.dtype`, *可选*, 默认为 jnp.float32):
            参数的数据类型
    """

    # 定义类属性 dim、dropout 和 dtype
    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    # 设置方法，初始化网络层
    def setup(self):
        inner_dim = self.dim * 4  # 计算内部维度
        self.proj = nn.Dense(inner_dim * 2, dtype=self.dtype)  # 初始化线性层
        self.dropout_layer = nn.Dropout(rate=self.dropout)  # 初始化丢弃层

    # 定义前向传播方法
    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.proj(hidden_states)  # 通过线性层处理隐藏状态
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)  # 将输出分为两个部分
        return self.dropout_layer(hidden_linear * nn.gelu(hidden_gelu), deterministic=deterministic)  # 返回带丢弃的激活输出
```