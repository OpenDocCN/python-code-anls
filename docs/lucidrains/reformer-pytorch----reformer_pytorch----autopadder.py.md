# `.\lucidrains\reformer-pytorch\reformer_pytorch\autopadder.py`

```py
# 导入数学库和 PyTorch 库
import math
import torch
from torch import nn
import torch.nn.functional as F

# 导入自定义模块
from reformer_pytorch.reformer_pytorch import Reformer, ReformerLM, LSHSelfAttention

# 定义函数，用于将张量填充到指定的倍数
def pad_to_multiple(tensor, seqlen, multiple, dim=-1):
    # 计算倍数
    m = seqlen / multiple
    # 如果是整数倍则直接返回张量
    if m.is_integer():
        return tensor
    # 计算需要填充的长度
    remainder = math.ceil(m) * multiple - seqlen
    # 计算填充的偏移量
    pad_offset = (0,) * (-1 - dim) * 2
    # 对张量进行填充
    return F.pad(tensor, (*pad_offset, 0, remainder), value=0)

# 定义自动填充器类
class Autopadder(nn.Module):
    def __init__(self, net):
        super().__init__()
        # 检查输入的网络类型是否符合要求
        assert isinstance(net, (LSHSelfAttention, Reformer, ReformerLM)), 'only modules LSHSelfAttention, Reformer, ReformerLM accepted'
        self.net = net

        # 获取 Reformer 对象
        reformer = net.reformer if isinstance(net, ReformerLM) else net
        # 根据网络类型确定填充的维度
        self.pad_dim = -1 if isinstance(net, ReformerLM) else -2

        # 获取 Reformer 的参数
        self.bucket_size = reformer.bucket_size
        self.num_mem_kv = reformer.num_mem_kv
        self.full_attn_thres = reformer.full_attn_thres

    def forward(self, x, **kwargs):
        # 获取输入张量的形状信息
        b, t, m, device = *x.shape[:2], self.num_mem_kv, x.device

        # 获取关键信息和输入掩码
        keys = kwargs.get('keys')
        input_mask = kwargs.get('input_mask')
        input_attn_mask = kwargs.get('input_attn_mask')

        # 计算关键信息的长度
        k_len = 0 if keys is None else keys.shape[1]
        # 计算序列长度
        seqlen = t + m + k_len

        # 如果序列长度超过全局注意力阈值
        if seqlen > self.full_attn_thres:
            # 如果输入掩码为空，则创建全为 True 的掩码
            if input_mask is None:
                input_mask = torch.full((b, t), True, device=x.device, dtype=torch.bool)

            # 对输入张量进行填充
            x = pad_to_multiple(x, seqlen, self.bucket_size * 2, dim=self.pad_dim)

            # 如果输入掩码不为空，则对其进行填充
            if input_mask is not None:
                new_mask = F.pad(input_mask, (0, x.shape[1] - input_mask.shape[1]), value=False)
                kwargs.update(input_mask=new_mask)

            # 如果输入注意力掩码不为空，则对其进行填充
            if input_attn_mask is not None:
                offset = x.shape[1] - input_attn_mask.shape[1]
                new_mask = F.pad(input_attn_mask, (0, offset, 0, offset), value=False)
                kwargs.update(input_attn_mask=new_mask)

        # 对输入进行网络前向传播
        out = self.net(x, **kwargs)
        # 返回前 t 个时间步的输出
        return out[:, 0:t]
```