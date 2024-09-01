# `.\flux\src\flux\model.py`

```py
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass

# 导入 PyTorch 和相关模块
import torch
from torch import Tensor, nn

# 从 flux.modules.layers 模块导入特定的类
from flux.modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)

# 定义包含模型参数的类
@dataclass
class FluxParams:
    # 输入通道数
    in_channels: int
    # 输入向量维度
    vec_in_dim: int
    # 上下文输入维度
    context_in_dim: int
    # 隐藏层大小
    hidden_size: int
    # MLP 比例
    mlp_ratio: float
    # 头数
    num_heads: int
    # 网络深度
    depth: int
    # 单流块的深度
    depth_single_blocks: int
    # 轴维度列表
    axes_dim: list[int]
    # theta 参数
    theta: int
    # 是否使用 QKV 偏置
    qkv_bias: bool
    # 是否使用引导嵌入
    guidance_embed: bool

# 定义 Flux 模型类
class Flux(nn.Module):
    """
    Transformer 模型用于序列上的流匹配。
    """

    # 初始化方法
    def __init__(self, params: FluxParams):
        super().__init__()

        # 保存参数
        self.params = params
        # 输入通道数
        self.in_channels = params.in_channels
        # 输出通道数与输入通道数相同
        self.out_channels = self.in_channels
        # 确保隐藏层大小可以被头数整除
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        # 计算位置编码维度
        pe_dim = params.hidden_size // params.num_heads
        # 确保轴维度总和与位置编码维度匹配
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        # 隐藏层大小
        self.hidden_size = params.hidden_size
        # 头数
        self.num_heads = params.num_heads
        # 初始化位置嵌入层
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        # 初始化图像输入线性层
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        # 初始化时间嵌入层
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        # 初始化向量嵌入层
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        # 初始化引导嵌入层（如果需要的话）
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        # 初始化文本输入线性层
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        # 创建双流块的模块列表
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        # 创建单流块的模块列表
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        # 初始化最终层
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    # 前向传播方法
    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:  # 定义返回类型为 Tensor 的函数
        # 检查 img 和 txt 张量是否都具有 3 个维度
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # 对输入的 img 张量进行初步处理
        img = self.img_in(img)
        # 计算时间步嵌入向量，并通过 self.time_in 处理
        vec = self.time_in(timestep_embedding(timesteps, 256))
        # 如果启用了指导嵌入，则处理指导嵌入
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            # 将指导嵌入向量添加到 vec 中
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        # 将其他向量添加到 vec 中
        vec = vec + self.vector_in(y)
        # 对 txt 张量进行处理
        txt = self.txt_in(txt)

        # 将 txt_ids 和 img_ids 按维度 1 拼接
        ids = torch.cat((txt_ids, img_ids), dim=1)
        # 计算位置编码
        pe = self.pe_embedder(ids)

        # 对 double_blocks 中的每个块进行处理
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        # 将 txt 和 img 张量按维度 1 拼接
        img = torch.cat((txt, img), 1)
        # 对 single_blocks 中的每个块进行处理
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        # 截取 img 张量，去掉前面的 txt 部分
        img = img[:, txt.shape[1] :, ...]

        # 最终处理 img 张量，返回结果
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
```