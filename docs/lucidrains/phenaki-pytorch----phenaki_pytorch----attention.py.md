# `.\lucidrains\phenaki-pytorch\phenaki_pytorch\attention.py`

```py
        # 初始化注意力机制模块
        def __init__(
            self,
            dim,
            dim_context = None,
            dim_head = 64,
            heads = 8,
            causal = False,
            num_null_kv = 0,
            norm_context = True,
            dropout = 0.,
            scale = 8
        ):
            # 调用父类初始化方法
            super().__init__()
            # 设置注意力头数
            self.heads = heads
            # 是否为因果注意力
            self.causal = causal
            # 缩放因子
            self.scale = scale
            # 内部维度
            inner_dim = dim_head * heads
            # 如果未指定上下文维度，则默认为输入维度
            dim_context = default(dim_context, dim)

            # 如果是因果注意力，则使用AlibiPositionalBias初始化相对位置偏置
            if causal:
                self.rel_pos_bias = AlibiPositionalBias(heads = heads)

            # 注意力机制的dropout层
            self.attn_dropout = nn.Dropout(dropout)

            # 输入的LayerNorm层
            self.norm = LayerNorm(dim)
            # 上下文的LayerNorm层（如果需要规范化上下文）
            self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

            # 空键值对的数量
            self.num_null_kv = num_null_kv
            # 空键值对参数
            self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))

            # 查询转换层
            self.to_q = nn.Linear(dim, inner_dim, bias = False)
            # 键值对转换层
            self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias = False)

            # 查询缩放参数
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            # 键缩放参数
            self.k_scale = nn.Parameter(torch.ones(dim_head))

            # 输出转换层
            self.to_out = nn.Linear(inner_dim, dim, bias = False)
    # 获取输入张量 x 的批量大小、设备和数据类型
    batch, device, dtype = x.shape[0], x.device, x.dtype

    # 如果上下文存在，则对上下文进行归一化处理
    if exists(context):
        context = self.context_norm(context)

    # 对输入张量 x 进行归一化处理
    x = self.norm(x)

    # 将输入张量 x 转换为查询（q）、键（k）、值（v）张量
    q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

    # 将查询（q）、键（k）、值（v）张量按照指定维度重新排列
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

    # 重复空键值对（null_kv）以匹配批量大小和维度
    nk, nv = repeat(self.null_kv, 'h (n r) d -> b h n r d', b = batch, r = 2).unbind(dim = -2)

    # 将键（k）和值（v）张量与空键值对（nk、nv）进行拼接
    k = torch.cat((nk, k), dim = -2)
    v = torch.cat((nv, v), dim = -2)

    # 对查询（q）和键（k）进行 L2 归一化处理
    q, k = map(l2norm, (q, k))
    q = q * self.q_scale
    k = k * self.k_scale

    # 计算查询（q）和键（k）之间的相似度
    sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

    i, j = sim.shape[-2:]

    # 如果存在注意力偏置（attn_bias），则对相似度矩阵进行加权
    if exists(attn_bias):
        attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value = 0.)
        sim = sim + attn_bias

    # 如果存在掩码（mask），则对掩码进行处理
    if exists(mask):
        mask = F.pad(mask, (self.num_null_kv, 0), value = True)
        mask = rearrange(mask, 'b j -> b 1 1 j')
        sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

    # 如果启用因果注意力，则对相似度矩阵进行处理
    if self.causal:
        sim = sim + self.rel_pos_bias(sim)

        causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

    # 对相似度矩阵进行 softmax 操作
    attn = sim.softmax(dim = -1)
    attn = self.attn_dropout(attn)

    # 计算输出张量
    out = einsum('b h i j, b h j d -> b h i d', attn, v)

    # 重新排列输出张量的维度
    out = rearrange(out, 'b h n d -> b n (h d)')
    return self.to_out(out)
# 定义一个名为 AlibiPositionalBias 的类，用于处理位置偏差
class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        # 初始化斜率参数
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        # 注册斜率参数和偏差参数
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    # 获取偏差值
    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    # 获取斜率参数
    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    # 前向传播函数
    def forward(self, sim):
        h, i, j, device = *sim.shape[-3:], sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent = False)

        return self.bias

# 定义一个名为 ContinuousPositionBias 的类，用于处理连续位置偏差
class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims = 2, # 2 for images, 3 for video
        layers = 2,
        log_dist = True,
        cache_rel_pos = False
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads)

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer('rel_pos', None, persistent = False)

    # 前向传播函数
    def forward(self, *dimensions, device = torch.device('cpu')):

        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device = device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing = 'ij'))
            grid = rearrange(grid, 'c ... -> (...) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer('rel_pos', rel_pos, persistent = False)

        rel_pos = self.rel_pos.float()

        for layer in self.net:
            rel_pos = layer(rel_pos)

        return rearrange(rel_pos, 'i j h -> h i j')

# 定义一个名为 Transformer 的类，用于实现 Transformer 模型
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_context = None,
        causal = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        peg = False,
        peg_causal = False,
        attn_num_null_kv = 2,
        has_cross_attn = False,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化一个空的神经网络模块列表
        self.layers = nn.ModuleList([])

        # 循环depth次，向神经网络模块列表中添加不同的模块
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 如果peg为真，则添加一个PEG模块，否则添加None
                PEG(dim = dim, causal = peg_causal) if peg else None,
                # 添加一个Attention模块
                Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, dropout = attn_dropout),
                # 如果has_cross_attn为真，则添加一个带有跨注意力的Attention模块，否则添加None
                Attention(dim = dim, dim_head = dim_head, dim_context = dim_context, heads = heads, causal = False, num_null_kv = attn_num_null_kv, dropout = attn_dropout) if has_cross_attn else None,
                # 添加一个FeedForward模块
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        # 初始化一个LayerNorm模块
        self.norm_out = LayerNorm(dim)

    @beartype
    def forward(
        self,
        x,
        video_shape: Tuple[int, int, int, int] = None,
        attn_bias = None,
        context = None,
        self_attn_mask = None,
        cross_attn_context_mask = None
    ):

        # 遍历神经网络模块列表中的不同模块
        for peg, self_attn, cross_attn, ff in self.layers:
            # 如果存在PEG模块，则对输入进行处理并与原始输入相加
            if exists(peg):
                x = peg(x, shape = video_shape) + x

            # 对输入进行自注意力处理并与原始输入相加
            x = self_attn(x, attn_bias = attn_bias, mask = self_attn_mask) + x

            # 如果存在跨注意力模块且存在上下文信息，则对输入进行处理并与原始输入相加
            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context = context, mask = cross_attn_context_mask) + x

            # 对输入进行前馈处理并与原始输入相加
            x = ff(x) + x

        # 对处理后的结果进行LayerNorm处理并返回
        return self.norm_out(x)
```