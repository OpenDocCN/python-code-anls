# `.\lucidrains\glom-pytorch\glom_pytorch\glom_pytorch.py`

```
# 从 math 模块中导入 sqrt 函数
from math import sqrt
# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn 和 functional 模块
import torch.nn.functional as F
# 从 torch 模块中导入 einsum 函数
from torch import nn, einsum
# 从 einops 模块中导入 rearrange 和 repeat 函数，以及 torch 模块中的 Rearrange 类
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 常量定义

# 定义 TOKEN_ATTEND_SELF_VALUE 常量为 -5e-4
TOKEN_ATTEND_SELF_VALUE = -5e-4

# 辅助函数

# 定义 exists 函数，判断值是否存在
def exists(val):
    return val is not None

# 定义 default 函数，如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 类定义

# 定义 GroupedFeedForward 类
class GroupedFeedForward(nn.Module):
    def __init__(self, *, dim, groups, mult = 4):
        super().__init__()
        total_dim = dim * groups # 计算总维度
        # 定义神经网络结构
        self.net = nn.Sequential(
            Rearrange('b n l d -> b (l d) n'),
            nn.Conv1d(total_dim, total_dim * mult, 1, groups = groups),
            nn.GELU(),
            nn.Conv1d(total_dim * mult, total_dim, 1, groups = groups),
            Rearrange('b (l d) n -> b n l d', l = groups)
        )

    # 前向传播函数
    def forward(self, levels):
        return self.net(levels)

# 定义 ConsensusAttention 类
class ConsensusAttention(nn.Module):
    def __init__(self, num_patches_side, attend_self = True, local_consensus_radius = 0):
        super().__init__()
        self.attend_self = attend_self
        self.local_consensus_radius = local_consensus_radius

        # 如果存在局部一致性半径
        if self.local_consensus_radius > 0:
            # 生成坐标网格
            coors = torch.stack(torch.meshgrid(
                torch.arange(num_patches_side),
                torch.arange(num_patches_side)
            )).float()

            coors = rearrange(coors, 'c h w -> (h w) c')
            dist = torch.cdist(coors, coors)
            mask_non_local = dist > self.local_consensus_radius
            mask_non_local = rearrange(mask_non_local, 'i j -> () i j')
            self.register_buffer('non_local_mask', mask_non_local)

    # 前向传播函数
    def forward(self, levels):
        _, n, _, d, device = *levels.shape, levels.device
        q, k, v = levels, F.normalize(levels, dim = -1), levels

        sim = einsum('b i l d, b j l d -> b l i j', q, k) * (d ** -0.5)

        if not self.attend_self:
            self_mask = torch.eye(n, device = device, dtype = torch.bool)
            self_mask = rearrange(self_mask, 'i j -> () () i j')
            sim.masked_fill_(self_mask, TOKEN_ATTEND_SELF_VALUE)

        if self.local_consensus_radius > 0:
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(self.non_local_mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b l i j, b j l d -> b i l d', attn, levels)
        return out

# 主类定义

# 定义 Glom 类
class Glom(nn.Module):
    def __init__(
        self,
        *,
        dim = 512,
        levels = 6,
        image_size = 224,
        patch_size = 14,
        consensus_self = False,
        local_consensus_radius = 0
    ):
        super().__init__()
        # 计算每个边上的补丁数量
        num_patches_side = (image_size // patch_size)
        num_patches =  num_patches_side ** 2
        self.levels = levels

        # 图像转换为标记的神经网络结构
        self.image_to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_size ** 2 * 3, dim)
        )
        self.pos_emb = nn.Embedding(num_patches, dim)

        # 列的所有级别的初始嵌入
        self.init_levels = nn.Parameter(torch.randn(levels, dim))

        # 自下而上和自上而下
        self.bottom_up = GroupedFeedForward(dim = dim, groups = levels)
        self.top_down = GroupedFeedForward(dim = dim, groups = levels - 1)

        # 一致性注意力
        self.attention = ConsensusAttention(num_patches_side, attend_self = consensus_self, local_consensus_radius = local_consensus_radius)
    # 定义前向传播函数，接受输入图像和可选参数，返回处理后的结果
    def forward(self, img, iters = None, levels = None, return_all = False):
        # 获取输入图像的形状和设备信息
        b, device = img.shape[0], img.device
        # 如果未提供迭代次数，则设置为默认值（层级数的两倍），以便信息在上下传播时能够传播
        iters = default(iters, self.levels * 2)

        # 将图像转换为 tokens
        tokens = self.image_to_tokens(img)
        n = tokens.shape[1]

        # 生成位置编码
        pos_embs = self.pos_emb(torch.arange(n, device = device))
        pos_embs = rearrange(pos_embs, 'n d -> () n () d')

        # 初始化底层 tokens
        bottom_level = tokens
        bottom_level = rearrange(bottom_level, 'b n d -> b n () d')

        # 如果未提供层级信息，则使用初始层级信息
        if not exists(levels):
            levels = repeat(self.init_levels, 'l d -> b n l d', b = b, n = n)

        # 存储每次迭代后的隐藏层信息
        hiddens = [levels]

        # 初始化每个层级的贡献次数
        num_contributions = torch.empty(self.levels, device = device).fill_(4)
        num_contributions[-1] = 3  # 顶层不会得到来自顶部的贡献，因此需要考虑这一点在计算加权平均时

        # 迭代处理
        for _ in range(iters):
            # 将原始输入附加到最底层，用于自底向上
            levels_with_input = torch.cat((bottom_level, levels), dim = -2)

            # 底部向上处理
            bottom_up_out = self.bottom_up(levels_with_input[..., :-1, :])

            # 顶部向下处理，加上位置编码
            top_down_out = self.top_down(levels_with_input[..., 2:, :] + pos_embs)
            top_down_out = F.pad(top_down_out, (0, 0, 0, 1), value = 0.)

            # 计算共识信息
            consensus = self.attention(levels)

            # 计算加权平均值
            levels_sum = torch.stack((levels, bottom_up_out, top_down_out, consensus)).sum(dim = 0)
            levels_mean = levels_sum / rearrange(num_contributions, 'l -> () () l ()')

            # 更新层级信息，用于下一次迭代
            levels = levels_mean
            hiddens.append(levels)

        # 如果需要返回所有隐藏层信息，则返回整个列表
        if return_all:
            return torch.stack(hiddens)

        # 否则，只返回最终的层级信息
        return levels
```