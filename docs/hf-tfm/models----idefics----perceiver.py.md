# `.\models\idefics\perceiver.py`

```
    def __init__(
        self, config: IdeficsConfig, embed_dim: int, depth: int, n_heads: int, head_dim: int, n_latents: int
    ):
        """
        初始化函数，创建一个 IdeficsPerceiverResampler 对象。

        参数:
        - config: IdeficsConfig 对象，包含了模型的配置信息
        - embed_dim: 整数，嵌入维度，用于定义输入的特征维度
        - depth: 整数，表示模型的深度或层数
        - n_heads: 整数，注意力头的数量，用于多头注意力机制
        - head_dim: 整数，每个注意力头的维度
        - n_latents: 整数，指定要生成的潜变量的数量
        """
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        
        # 将传入的配置信息保存为类的属性
        self.config = config
        
        # 初始化一个线性层，用于将输入嵌入特征维度映射到 latent 潜变量的数量维度
        self.project_in = nn.Linear(embed_dim, n_latents * head_dim)
        
        # 保存配置中的参数作为类属性
        self.depth = depth
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_latents = n_latents


这段代码定义了一个名为 `IdeficsPerceiverResampler` 的 PyTorch 模型类，用于实现 Perceiver Resampler 架构。
    ) -> None:
        """
        Instantiates a Perceiver Resampler that operates over a sequence of embeddings (say from a ResNet or ViT or
        MAE) of a given dimension, performs `depth` blocks of cross-attention with a fixed `n_latents` inputs, then
        returns a Tensor of shape [bsz, n_latents, embed_dim]. :param embed_dim: Dimensionality of embeddings being fed
        to the Perceiver Resampler (also dimensionality of latent embeddings *returned* by the Perceiver Resampler.
        Could be e.g., VIT embed_dim, ResNet pool dim, and so on.

        Args:
            config (`IdeficsConfig`): config object
            embed_dim (`int`): The size of each embedding vector
            depth (`int`): Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (< 3).
            n_heads (`int`): Number of heads in each Transformer block (for multi-headed self-attention).
            head_dim (`int`): Dimensionality of each head projection in the Transformer block.
            n_latents (`int`):
                Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).

        """
        super().__init__()
        # 设置类的属性 embed_dim, n_heads, head_dim, n_latents
        self.embed_dim, self.n_heads, self.head_dim, self.n_latents = embed_dim, n_heads, head_dim, n_latents
        # 获取配置文件中的 qk_layer_norms_perceiver 设置
        self.qk_layer_norms = config.perceiver_config.qk_layer_norms_perceiver

        # 创建 Perceiver 的潜变量（latent embeddings）
        self.latents = nn.Parameter(torch.randn(self.n_latents, self.embed_dim), requires_grad=True)

        # 确定中间层的维度，根据是否存在 vision_config.embed_dim 进行选择
        self.intermediate_dim = (
            self.embed_dim * 4
            if not hasattr(config.vision_config, "embed_dim")
            else config.vision_config.embed_dim * 4
        )
        # 创建包含 depth 个 Transformer 块的模块列表
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        IdeficsPerceiverAttention(self.embed_dim, self.n_heads, self.head_dim, self.qk_layer_norms),
                        IdeficsMLP(self.intermediate_dim, config),
                    ]
                )
                for _ in range(depth)
            ]
        )
        # 创建用于归一化输出的 LayerNorm
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Resample arbitrary length context & *compress* down to self.n_latents latent embeddings"""
        # 重复潜变量以匹配上下文张量的批次大小
        latents = self.latents.repeat(context.shape[0], 1, 1)

        # 通过每个 Transformer 块的注意力机制和 MLP 前馈网络
        for attn, ff in self.blocks:
            # 执行注意力机制
            latents = attn(context, latents) + latents
            # 执行前馈网络
            latents = ff(latents) + latents

        # 对最终的输出进行 LayerNorm 处理
        return self.layer_norm(latents)
class IdeficsPerceiverAttention(nn.Module):
    # 定义 Perceiver 注意力模块，用于处理跨注意力的计算
    def __init__(self, embed_dim: int, n_heads: int, head_dim: int, qk_layer_norms: bool) -> None:
        """Perceiver Cross-Attention Module --> let long-form inputs be `context`, resampled embeddings be `latents`"""
        # 初始化函数，设置模块的参数和层
        super().__init__()
        self.embed_dim, self.n_heads, self.head_dim = embed_dim, n_heads, head_dim
        self.qk_layer_norms = qk_layer_norms
        # 对上下文向量和潜在向量进行层标准化
        self.context_layer_norm = nn.LayerNorm(self.embed_dim)
        self.latents_layer_norm = nn.LayerNorm(self.embed_dim)
        if self.qk_layer_norms:
            # 如果需要，对查询和键进行单独的层标准化
            self.q_layer_norm = nn.LayerNorm(self.head_dim)
            self.k_layer_norm = nn.LayerNorm(self.head_dim)

        # 缩放因子，用于缩放 Q 和 K 的点积计算
        self.qk_scale = self.head_dim**-0.5

        # Q, K, V 投影层 (无偏置 -- 根据 Perceiver/Flamingo 论文中的详细说明)
        self.q_proj = nn.Linear(self.embed_dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.n_heads * self.head_dim, bias=False)

        # 输出投影层，将多头注意力结果投影到最终输出维度
        self.output_proj = nn.Linear(self.n_heads * self.head_dim, embed_dim, bias=False)
    def forward(self, context: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Runs Perceiver Self-Attention, with special (context, latents) appended along the `seq` dimension!

        Args:
            context (`torch.Tensor`):
                Tensor of shape `[bsz, seq, embed_dim]` representing long-form context to resample.
            latents (`torch.Tensor`):
                Tensor of shape `[bsz, n_latents, embed_dim]` representing fixed length latents to compress to.

        Returns:
            `torch.Tensor`: Tensor of shape `[bsz, n_latents, embed_dim]` representing attention over latents w/ cross
            from context.
        """
        # 对上下文进行 layer normalization
        context = self.context_layer_norm(context)
        # 对潜变量进行 layer normalization
        latents = self.latents_layer_norm(latents)
        # 获取 batch_size, seq_length, embed_dim 的值
        batch_size, seq_length, embed_dim = context.shape[:3]

        # 查询、键、值的投影 --> 注意，在 Flamingo 中，潜变量会 *连接* 到上下文之前进行注意力操作！
        # 注意：这导致查询具有 `seq = n_latents`，键和值具有 `seq = len(context) + n_latents`
        q = self.q_proj(latents)
        k = self.k_proj(torch.cat([context, latents], dim=-2))
        v = self.v_proj(torch.cat([context, latents], dim=-2))

        # 多头自注意力机制，使用稳定的 softmax（在调用 softmax 前减去每行的最大值）
        #   =>> `attn` 应该是形状为 [n_latents x (context + n_latents)] 的二维矩阵
        q, k, v = [x.reshape(batch_size, x.shape[1], self.n_heads, self.head_dim).transpose(1, 2) for x in (q, k, v)]

        # 如果启用了 qk_layer_norms，对查询和键进行 layer normalization
        if self.qk_layer_norms:
            q = self.q_layer_norm(q)
            k = self.k_layer_norm(k)

        # 计算注意力分数
        scores = torch.einsum("... i d, ... j d -> ... i j", q * self.qk_scale, k)
        # 对分数进行稳定化处理（减去每行的最大值）
        stabilized_scores = scores - (scores.amax(dim=-1, keepdim=True).detach())
        # 应用 softmax 获取注意力权重
        attn = stabilized_scores.softmax(dim=-1)

        # 注意力加权平均并投影回输出空间...
        resampled = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        # 对 resampled 进行重新排列，以便恢复形状为 "bsz seq (heads embed)"
        return self.output_proj(resampled.transpose(1, 2).flatten(-2))
class IdeficsMLP(nn.Module):
    def __init__(self, intermediate_size, config: IdeficsConfig):
        """Simple MLP block with intermediate_size and embedding size"""
        # 初始化函数，定义一个简单的MLP模块，使用给定的intermediate_size和配置的embedding大小
        super().__init__()
        # 从配置中获取视觉配置的嵌入维度
        self.embed_dim = config.vision_config.embed_dim
        # LayerNorm层，对输入进行归一化处理
        self.ln = nn.LayerNorm(self.embed_dim)
        # 全连接层，将输入的嵌入维度映射到intermediate_size维度，无偏置项
        self.fc = nn.Linear(self.embed_dim, intermediate_size, bias=False)
        # 激活函数ReLU
        self.act = nn.ReLU()
        # 进一步映射intermediate_size维度到嵌入维度，无偏置项
        self.c_proj = nn.Linear(intermediate_size, self.embed_dim, bias=False)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        # 前向传播函数，接收隐藏状态作为输入，返回张量类型的隐藏状态
        hidden_states = self.ln(hidden_states)  # LayerNorm处理隐藏状态
        hidden_states = self.fc(hidden_states)  # 全连接层处理隐藏状态
        hidden_states = self.act(hidden_states)  # ReLU激活函数处理隐藏状态
        hidden_states = self.c_proj(hidden_states)  # 进一步映射处理隐藏状态

        return hidden_states
```