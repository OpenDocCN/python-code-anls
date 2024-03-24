# `.\lucidrains\global-self-attention-network\gsa_pytorch\gsa_pytorch.py`

```
# 导入 torch 库
import torch
# 导入 torch.nn.functional 模块，并重命名为 F
import torch.nn.functional as F
# 从 torch 中导入 nn 和 einsum 模块
from torch import nn, einsum
# 从 einops 中导入 rearrange 函数
from einops import rearrange
# 从 inspect 中导入 isfunction 函数

# 辅助函数

# 如果 val 存在则返回 val，否则返回 d()
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# 判断 val 是否存在
def exists(val):
    return val is not None

# 计算重新索引张量
def calc_reindexing_tensor(l, L, device):
    """
    Appendix B - (5)
    """
    # 创建 x 张量
    x = torch.arange(l, device = device)[:, None, None]
    # 创建 i 张量
    i = torch.arange(l, device = device)[None, :, None]
    # 创建 r 张量
    r = torch.arange(-(L - 1), L, device = device)[None, None, :]
    # 创建 mask 张量
    mask = ((i - x) == r) & ((i - x).abs() <= L)
    return mask.float()

# 类

# GSA 类
class GSA(nn.Module):
    # 初始化函数
    def __init__(self, dim, *, rel_pos_length = None, dim_out = None, heads = 8, dim_key = 64, norm_queries = False, batch_norm = True):
        super().__init__()
        dim_out = default(dim_out, dim)
        dim_hidden = dim_key * heads

        self.heads = heads
        self.dim_out = dim_out
        self.rel_pos_length = rel_pos_length
        self.norm_queries = norm_queries

        # 创建卷积层，用于将输入转换为查询、键和值
        self.to_qkv = nn.Conv2d(dim, dim_hidden * 3, 1, bias = False)
        # 创建卷积层，用于将隐藏层转换为输出维度
        self.to_out = nn.Conv2d(dim_hidden, dim_out, 1)

        self.rel_pos_length = rel_pos_length
        if exists(rel_pos_length):
            num_rel_shifts = 2 * rel_pos_length - 1
            self.norm = nn.BatchNorm2d(dim_key) if batch_norm else None
            self.rel_rows = nn.Parameter(torch.randn(num_rel_shifts, dim_key))
            self.rel_columns = nn.Parameter(torch.randn(num_rel_shifts, dim_key))

    # 前向传播函数
    def forward(self, img):
        # 获取输入张量的形状信息
        b, c, x, y, h, c_out, L, device = *img.shape, self.heads, self.dim_out, self.rel_pos_length, img.device

        # 将输入张量通过 to_qkv 卷积层得到查询、键和值
        qkv = self.to_qkv(img).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) c (x y)', h = h), qkv)

        # 对键进行 softmax 操作
        k = k.softmax(dim = -1)
        # 计算上下文信息
        context = einsum('ndm,nem->nde', k, v)

        # 如果需要对查询进行归一化，则进行 softmax 操作
        content_q = q if not self.norm_queries else q.softmax(dim=-2)

        # 计算内容输出
        content_out = einsum('nde,ndm->nem', context, content_q)
        content_out = rearrange(content_out, 'n d (x y) -> n d x y', x = x, y = y)

        # 根据附录 B (6) - (8) 中的数学实现细节进行处理
        if exists(self.rel_pos_length):
            q, v = map(lambda t: rearrange(t, 'n c (x y) -> n c x y', x = x, y = y), (q, v))

            Ix = calc_reindexing_tensor(x, L, device)
            Px = einsum('xir,rd->xid', Ix, self.rel_rows)
            Sx = einsum('ndxy,xid->nixy', q, Px)
            Yh = einsum('nixy,neiy->nexy', Sx, v)

            if exists(self.norm):
                Yh = self.norm(Yh)

            Iy = calc_reindexing_tensor(y, L, device)
            Py = einsum('yir,rd->yid', Iy, self.rel_columns)
            Sy = einsum('ndxy,yid->nixy', q, Py)
            rel_pos_out = einsum('nixy,nexi->nexy', Sy, Yh)

            content_out = content_out + rel_pos_out.contiguous()

        content_out = rearrange(content_out, '(b h) c x y -> b (h c) x y', h = h)
        return self.to_out(content_out)
```