# `.\flux\src\flux\math.py`

```py
# 导入 PyTorch 库和 einops 的 rearrange 函数
import torch
from einops import rearrange
from torch import Tensor


# 注意力机制函数
def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    # 对 q 和 k 应用相对位置编码
    q, k = apply_rope(q, k, pe)

    # 使用缩放点积注意力计算输出
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    # 重新排列输出张量的维度
    x = rearrange(x, "B H L D -> B L (H D)")

    # 返回处理后的张量
    return x


# 相对位置编码函数
def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    # 确保维度是偶数
    assert dim % 2 == 0
    # 计算尺度因子
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    # 计算 omega 值
    omega = 1.0 / (theta**scale)
    # 通过爱因斯坦求和计算输出
    out = torch.einsum("...n,d->...nd", pos, omega)
    # 创建旋转矩阵
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    # 重新排列旋转矩阵的维度
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    # 转换为 float 类型并返回
    return out.float()


# 应用相对位置编码的辅助函数
def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    # 重新排列 q 和 k 的维度并转换为 float 类型
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    # 计算 q 和 k 的编码输出
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    # 恢复原始维度并返回
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
```