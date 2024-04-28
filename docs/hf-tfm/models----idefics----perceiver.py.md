# `.\models\idefics\perceiver.py`

```py
# 导入必要的库和模块
from typing import Optional, Tuple
import torch
import torch.nn as nn
from .configuration_idefics import IdeficsConfig

# 定义一个名为IdeficsPerceiverResampler的类，继承自nn.Module
class IdeficsPerceiverResampler(nn.Module):
    # 初始化方法，接受配置、嵌入维度、深度、头数、头维度和潜变量数等参数
    def __init__(
        self, config: IdeficsConfig, embed_dim: int, depth: int, n_heads: int, head_dim: int, n_latents: int
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
        self.embed_dim, self.n_heads, self.head_dim, self.n_latents = embed_dim, n_heads, head_dim, n_latents
        self.qk_layer_norms = config.perceiver_config.qk_layer_norms_perceiver

        # Create Latents for Perceiver
        self.latents = nn.Parameter(torch.randn(self.n_latents, self.embed_dim), requires_grad=True)

        self.intermediate_dim = (
            self.embed_dim * 4
            if not hasattr(config.vision_config, "embed_dim")
            else config.vision_config.embed_dim * 4
        )
        # Create Transformer Blocks
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
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Resample arbitrary length context & *compress* down to self.n_latents latent embeddings"""
        # einsum.repeat(self.latents, "seq embed -> bsz seq embed", bsz=context.shape[0])
        latents = self.latents.repeat(context.shape[0], 1, 1)

        # Feed through Perceiver Attention blocks...
        for attn, ff in self.blocks:
            latents = attn(context, latents) + latents
            latents = ff(latents) + latents

        return self.layer_norm(latents)
class IdeficsPerceiverAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, head_dim: int, qk_layer_norms: bool) -> None:
        """Perceiver Cross-Attention Module --> let long-form inputs be `context`, resampled embeddings be `latents`"""
        # 初始化 Perceiver 注意力模块，接受嵌入维度、头数、头维度和是否对 Q、K 进行层归一化作为参数
        super().__init__()
        self.embed_dim, self.n_heads, self.head_dim = embed_dim, n_heads, head_dim
        self.qk_layer_norms = qk_layer_norms
        # 初始化嵌入维度、头数和头维度
        # 是否对 Q、K 进行层归一化
        # 归一化和缩放
        self.context_layer_norm = nn.LayerNorm(self.embed_dim)
        self.latents_layer_norm = nn.LayerNorm(self.embed_dim)
        if self.qk_layer_norms:
            self.q_layer_norm = nn.LayerNorm(self.head_dim)
            self.k_layer_norm = nn.LayerNorm(self.head_dim)

        self.qk_scale = self.head_dim**-0.5

        # Q, K, V 投影（无偏置 -- 来自 Perceiver/Flamingo 论文的细节）。
        self.q_proj = nn.Linear(self.embed_dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.n_heads * self.head_dim, bias=False)

        self.output_proj = nn.Linear(self.n_heads * self.head_dim, embed_dim, bias=False)
    def forward(self, context: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        运行 Perceiver Self-Attention，特殊地将 (context, latents) 沿着 `seq` 维度附加！

        Args:
            context (`torch.Tensor`):
                形状为 `[bsz, seq, embed_dim]` 的张量，表示要重新采样的长形上下文。
            latents (`torch.Tensor`):
                形状为 `[bsz, n_latents, embed_dim]` 的张量，表示要压缩到的固定长度潜变量。

        Returns:
            `torch.Tensor`: 形状为 `[bsz, n_latents, embed_dim]` 的张量，表示对潜变量进行的注意力与来自上下文的交叉。
        """
        # 对上下文进行 Layer Normalization
        context = self.context_layer_norm(context)
        # 对潜变量进行 Layer Normalization
        latents = self.latents_layer_norm(latents)
        # 获取批量大小、序列长度、嵌入维度
        batch_size, seq_length, embed_dim = context.shape[:3]

        # 查询、键、值的投影 --> 注意，在 Flamingo 中，潜变量在注意力之前与上下文*连接*！
        #   注意：这会导致查询的 `seq = n_latents`，键和值的 `seq = len(context) + n_latents`
        q = self.q_proj(latents)
        k = self.k_proj(torch.cat([context, latents], dim=-2))
        v = self.v_proj(torch.cat([context, latents], dim=-2))

        # 多头自注意力与稳定的 softmax（在 softmax 调用之前减去每行的最大值 -- `amax`）
        #   =>> `attn` 应该是形状为 [n_latents x (context + n_latents)] 的 2D 矩阵
        # einsum.rearrange(x, "bsz seq (heads embed) -> bsz heads seq embed", heads=self.n_heads)
        q, k, v = [x.reshape(batch_size, x.shape[1], self.n_heads, self.head_dim).transpose(1, 2) for x in (q, k, v)]

        if self.qk_layer_norms:
            q = self.q_layer_norm(q)
            k = self.k_layer_norm(k)

        scores = torch.einsum("... i d, ... j d -> ... i j", q * self.qk_scale, k)
        stabilized_scores = scores - (scores.amax(dim=-1, keepdim=True).detach())
        attn = stabilized_scores.softmax(dim=-1)

        # 注意并投影回输出...
        resampled = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        # einsum.rearrange(resampled, "bsz heads seq embed -> bsz seq (heads embed)", heads=self.n_heads)
        return self.output_proj(resampled.transpose(1, 2).flatten(-2))
class IdeficsMLP(nn.Module):
    # 定义一个名为IdeficsMLP的类，继承自nn.Module
    def __init__(self, intermediate_size, config: IdeficsConfig):
        # 初始化函数，接受intermediate_size和config作为参数
        """Simple MLP block with intermediate_size and embedding size"""
        # 简单的MLP块，具有intermediate_size和嵌入大小的注释
        super().__init__()
        # 调用父类的初始化函数
        self.embed_dim = config.vision_config.embed_dim
        # 设置embed_dim为config中vision_config的embed_dim属性
        self.ln = nn.LayerNorm(self.embed_dim)
        # 初始化LayerNorm层，输入维度为embed_dim
        self.fc = nn.Linear(self.embed_dim, intermediate_size, bias=False)
        # 初始化全连接层，输入维度为embed_dim，输出维度为intermediate_size，不使用偏置
        self.act = nn.ReLU()
        # 初始化ReLU激活函数
        self.c_proj = nn.Linear(intermediate_size, self.embed_dim, bias=False)
        # 初始化全连接层，输入维度为intermediate_size，输出维度为embed_dim，不使用偏置

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        # 前向传播函数，接受hidden_states作为参数，返回torch.FloatTensor类型的数据
        hidden_states = self.ln(hidden_states)
        # 对hidden_states进行LayerNorm操作
        hidden_states = self.fc(hidden_states)
        # 对hidden_states进行全连接操作
        hidden_states = self.act(hidden_states)
        # 对hidden_states进行ReLU激活
        hidden_states = self.c_proj(hidden_states)
        # 对hidden_states进行全连接操作

        return hidden_states
        # 返回处理后的hidden_states
```