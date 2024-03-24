# `.\lucidrains\quartic-transformer\quartic_transformer\multi_stream_transformer.py`

```
        """
        实现注意力机制的模块
        参数：
            dim - 输入特征的维度
            num_streams - 流的数量
            dim_head - 每个头的维度
            heads - 头的数量
            dropout - 丢弃率
            causal - 是否使用因果注意力
            pre_talking_heads - 是否使用预对话头
            post_talking_heads - 是否使用后对话头
            non_linear_talking_heads - 是否使用非线性对话头
        """
        super().__init__()
        dim_inner = dim_head * heads
        all_heads = num_streams * heads

        self.num_streams = num_streams

        # 将输入转换为查询、键、值
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)
        )

        # 生成门控值
        self.to_gates = nn.Sequential(
            nn.Linear(dim, heads),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        )

        # RMSNorm 归一化
        self.rmsnorm = einn.Norm('b... [d]', mean = False, bias = False)

        self.scale = dim_head ** 0.5
        self.causal = causal
        self.dropout = nn.Dropout(dropout)

        self.pre_talking_heads = None
        self.post_talking_heads = None

        # 根据参数选择是否使用非线性对话头
        if non_linear_talking_heads:
            self.pre_talking_heads = TalkingHeadsFeedForward(all_heads) if pre_talking_heads else None
            self.post_talking_heads = TalkingHeadsFeedForward(all_heads) if post_talking_heads else None
        else:
            # 根据参数选择是否使用卷积对话头
            self.pre_talking_heads = nn.Conv2d(all_heads, all_heads, 1, bias = False) if pre_talking_heads else None
            self.post_talking_heads = nn.Conv2d(all_heads, all_heads, 1, bias = False) if post_talking_heads else None

            # 初始化卷积对话头的权重
            nn.init.dirac_(self.pre_talking_heads.weight)
            nn.init.dirac_(self.post_talking_heads.weight)

        # 输出层
        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )
        ):
            # 获取输入张量 x 的流数
            s = self.num_streams
            # 对输入张量 x 进行均方根归一化
            x = self.rmsnorm(x)

            # 将输入张量 x 转换为查询、键、值张量
            q, k, v = self.to_qkv(x)

            # 对查询张量 q 进行缩放
            q = q * self.scale
            # 计算注意力矩阵
            sim = einsum('b h i d, b h j d -> b h i j', q, k)

            # 计算掩码值
            mask_value = -torch.finfo(sim.dtype).max

            # 如果存在预处理头部函数
            if exists(self.pre_talking_heads):
                # 重排注意力矩阵的维度
                sim = rearrange(sim, '(b s) h n d -> b (s h) n d', s = s)
                # 对注意力矩阵进行预处理
                sim = self.pre_talking_heads(sim)
                # 恢复注意力矩阵的维度
                sim = rearrange(sim, 'b (s h) n d -> (b s) h n d', s = s)

            # 如果存在掩码
            if exists(mask):
                # 根据掩码值对注意力矩阵进行处理
                sim = einx.where('b j, b ... j, ', mask, sim, mask_value)

            # 如果是因果注意力
            if self.causal:
                i, j = sim.shape[-2:]
                # 创建因果掩码
                causal_mask = torch.ones((i, j), dtype = torch.bool).triu(j - i + 1)
                sim = sim.masked_fill(causal_mask, mask_value)

            # 对注意力矩阵进行 softmax 操作
            attn = einx.softmax('b h i [j]', sim)

            # 保存 softmax 操作后的注意力矩阵
            post_softmax_attn = attn

            # 对注意力矩阵进行 dropout 操作
            attn = self.dropout(attn)

            # 如果存在后处理头部函数
            if exists(self.post_talking_heads):
                # 重排注意力矩阵的维度
                attn = rearrange(attn, '(b s) h n d -> b (s h) n d', s = s)
                # 对注意力矩阵进行后处理
                attn = self.post_talking_heads(attn)
                # 恢复注意力矩阵的维度
                attn = rearrange(attn, 'b (s h) n d -> (b s) h n d', s = s)

            # 计算输出张量
            out = einsum('b h i j, b h j d -> b h i d', attn, v)

            # 对输出张量进行门控操作
            out = out * self.to_gates(x)
            # 对输出张量进行输出转换
            out = self.to_out(out)

            # 返回输出张量和 softmax 操作后的注意力矩阵
            return out, post_softmax_attn
# 定义一个前馈神经网络模块
def FeedForward(dim, mult = 4, dropout = 0.):
    # 计算内部维度
    dim_inner = int(dim * mult)
    # 返回一个包含多个层的神经网络模块
    return nn.Sequential(
        # 归一化层，对输入进行归一化处理
        einn.Norm('b... [d]', mean = False, bias = False),
        # 全连接层，将输入维度转换为内部维度
        nn.Linear(dim, dim_inner, bias = False),
        # GELU激活函数
        nn.GELU(),
        # Dropout层，以一定概率丢弃部分神经元
        nn.Dropout(dropout),
        # 全连接层，将内部维度转换为输出维度
        nn.Linear(dim_inner, dim, bias = False)
    )

# 定义一个TalkingHeads前馈神经网络模块
def TalkingHeadsFeedForward(dim, mult = 2, dropout = 0.):
    # 计算内部维度
    dim_inner = int(dim * mult)
    # 创建一个包含多个层的神经网络模块
    net = nn.Sequential(
        # 归一化层，对输入进行归一化处理
        einn.Norm('b [c] ...', mean = False, bias = False),
        # 二维卷积层，将输入维度转换为内部维度
        nn.Conv2d(dim, dim_inner, 1, bias = False),
        # GELU激活函数
        nn.GELU(),
        # Dropout层，以一定概率丢弃部分神经元
        nn.Dropout(dropout),
        # 二维卷积层，将内部维度转换为输出维度
        nn.Conv2d(dim_inner, dim, 1, bias = False)
    )

    # 初始化最后一层的权重为零
    nn.init.zeros_(net[-1].weight)
    # 返回一个残差连接的神经网络模块
    return Residual(net)

# 定义TokenAndPosEmb类，用于处理共享的Token和位置嵌入
class TokenAndPosEmb(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        num_streams
    ):
        super().__init__()
        # 创建Token嵌入层
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 创建位置嵌入层
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        # 创建流嵌入参数
        self.stream_emb = nn.Parameter(torch.zeros(num_streams, dim))
        # 初始化流嵌入参数
        nn.init.normal_(self.stream_emb, std = 0.02)

    def forward(self, x):
        # 生成序列长度
        seq_len = torch.arange(x.shape[-1], device = x.device)
        # 获取Token嵌入
        token_emb = self.token_emb(x)
        # 获取位置嵌入
        pos_emb = self.pos_emb(seq_len)
        # 返回Token、位置和流嵌入的加和结果
        return einx.add('b n d, n d, s d -> (b s) n d', token_emb, pos_emb, self.stream_emb)

# 定义SeparateTokenAndPosEmb类，用于处理独立的Token和位置嵌入
class SeparateTokenAndPosEmb(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        max_seq_len,
        num_streams
    ):
        super().__init__()
        # 创建独立的Token嵌入参数
        self.token_emb = nn.Parameter(torch.zeros(num_streams, num_tokens, dim))
        # 创建独立的位置嵌入参数
        self.pos_emb = nn.Parameter(torch.zeros(num_streams, max_seq_len, dim))
        # 初始化Token嵌入参数和位置嵌入参数
        nn.init.normal_(self.token_emb, std = 0.02)
        nn.init.normal_(self.pos_emb, std = 0.02)

    def forward(self, x):
        # 生成序列长度
        seq_len = torch.arange(x.shape[-1], device = x.device)
        # 获取Token嵌入
        token_emb = get_at('s [e] d, b n -> b s n d', self.token_emb, x)
        # 获取位置嵌入
        pos_emb = get_at('s [e] d, n -> s n d', self.pos_emb, x)
        # 返回Token和位置嵌入的加和结果
        return einx.add('b s n d, s n d -> (b s) n d', token_emb, pos_emb)

# 定义MultiStreamTransformer类，用于多流Transformer模型
class MultiStreamTransformer(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        num_streams = 2,
        dim_head = 64,
        heads = 8,
        max_seq_len = 2048,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4.,
        ablate_cross_stream_talking_heads = False,
        pre_talking_heads = True,
        post_talking_heads = True,
        separate_stream_emb = True,
        non_linear_talking_heads = False
    ):
        # 调用父类的构造函数
        super().__init__()
        # 根据是否需要分离流嵌入选择不同的嵌入类
        embed_klass = SeparateTokenAndPosEmb if separate_stream_emb else TokenAndPosEmb

        # 初始化嵌入层
        self.emb = embed_klass(
            dim = dim,
            num_tokens = num_tokens,
            num_streams = num_streams,
            max_seq_len = max_seq_len
        )

        # 设置流的数量
        self.num_streams = num_streams
        # 初始化层列表
        self.layers = ModuleList([])

        # 根据是否禁用跨流的交谈头选择不同的流数量
        talking_heads_num_streams = 2 if not ablate_cross_stream_talking_heads else 1

        # 根据深度循环创建多个注意力层和前馈层
        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    dropout = attn_dropout,
                    num_streams = talking_heads_num_streams,
                    pre_talking_heads = pre_talking_heads,
                    post_talking_heads = post_talking_heads,
                    non_linear_talking_heads = non_linear_talking_heads
                ),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        # 定义输出层
        self.to_logits = nn.Sequential(
            Reduce('(b s) n d -> b n d', 'sum', s = num_streams),
            einn.Norm('b... [d]', mean = False, bias = False),
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(
        self,
        x,
        mask = None,
        stream_attn_diversity_loss = False
    ):
        # 获取输入张量的形状和设备信息
        b, n, s, device = *x.shape, self.num_streams, x.device

        # 如果流的数量大于1，则计算流的注意力多样性损失
        stream_attn_diversity_loss &= s > 1

        # 对输入张量进行嵌入
        x = self.emb(x)

        # 存储每个注意力层的注意力矩阵
        attn_matrices = []

        # 遍历每个注意力层和前馈层
        for attn, ff in self.layers:
            # 计算注意力层的输出和后softmax的注意力矩阵
            attn_out, post_softmax_attn = attn(x, mask = mask)

            # 将后softmax的注意力矩阵添加到列表中
            attn_matrices.append(post_softmax_attn)

            # 更新输入张量
            x = x + attn_out
            x = ff(x) + x

        # 如果需要计算流的注意力多样性损失，则计算辅助损失
        if stream_attn_diversity_loss:
            aux_loss = sum([calc_stream_loss(attn_matrix, s).mean() for attn_matrix in attn_matrices])

        # 计算最终输出
        logits = self.to_logits(x)

        # 如果不需要计算流的注意力多样性损失，则直接返回logits
        if not stream_attn_diversity_loss:
            return logits

        # 如果需要计算流的注意力多样性损失，则返回logits和辅助损失
        return logits, aux_loss
```