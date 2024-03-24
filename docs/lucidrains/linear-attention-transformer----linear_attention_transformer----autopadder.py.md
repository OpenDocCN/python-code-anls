# `.\lucidrains\linear-attention-transformer\linear_attention_transformer\autopadder.py`

```
# 导入数学库
import math
# 导入 PyTorch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.nn 模块中导入 F 函数
import torch.nn.functional as F
# 从 linear_attention_transformer.linear_attention_transformer 模块中导入 LinearAttentionTransformer 和 LinearAttentionTransformerLM 类
from linear_attention_transformer.linear_attention_transformer import LinearAttentionTransformer, LinearAttentionTransformerLM

# 定义一个函数，用于查找指定类型的模块
def find_module(nn_module, type):
    # 遍历 nn_module 中的所有模块
    for module in nn_module.modules():
        # 如果找到指定类型的模块，则返回该模块
        if isinstance(module, type):
            return module
    # 如果未找到指定类型的模块，则返回 None
    return None

# 定义一个函数，用于将张量填充到指定的倍数
def pad_to_multiple(tensor, multiple, dim=-1, pad_left = False):
    # 获取张量在指定维度上的长度
    seqlen = tensor.shape[dim]
    # 计算需要填充的数量
    m = seqlen / multiple
    # 如果 m 是整数，则不需要填充
    if m.is_integer():
        return tensor, 0

    # 计算填充前的偏移量
    pre_pad_offset = (0,) * (-1 - dim) * 2
    # 计算需要填充的数量
    padding = math.ceil(m) * multiple - seqlen
    # 根据填充方式进行填充
    offset = (padding, 0) if pad_left else (0, padding)
    # 对张量进行填充操作
    padded_tensor = F.pad(tensor, (*pre_pad_offset, *offset), value=0)
    return padded_tensor, padding

# 定义一个类 Autopadder，继承自 nn.Module 类
class Autopadder(nn.Module):
    # 初始化方法
    def __init__(self, net, pad_left=False):
        super().__init__()
        # 断言 net 是 LinearAttentionTransformer 或 LinearAttentionTransformerLM 类的实例
        assert isinstance(net, (LinearAttentionTransformer, LinearAttentionTransformerLM)), 'only modules SinkhornTransformer and SinkhornTransformerLM accepted'
        self.net = net

        # 判断 net 是否为 LinearAttentionTransformerLM 类的实例
        is_lm = isinstance(net, LinearAttentionTransformerLM)
        # 查找 net 中的 LinearAttentionTransformer 模块
        transformer = find_module(net, LinearAttentionTransformer)
        # 设置填充的倍数
        self.pad_to = transformer.pad_to_multiple
        # 设置填充的维度
        self.pad_dim = -1 if is_lm else -2
        # 设置填充的方式
        self.pad_left = pad_left

    # 前向传播方法
    def forward(self, x, **kwargs):
        # 如果不需要填充，则直接调用 net 的前向传播方法
        if self.pad_to <= 1:
            return self.net(x, **kwargs)

        # 获取输入张量 x 的形状和设备信息
        b, t, device = *x.shape[:2], x.device

        # 获取输入参数中的 input_mask，如果不存在则创建全为 True 的 mask
        input_mask = kwargs.get('input_mask')
        if input_mask is None:
            input_mask = torch.full((b, t), True, device=x.device, dtype=torch.bool)

        # 对输入张量 x 进行填充操作
        x, padding = pad_to_multiple(x, self.pad_to, dim=self.pad_dim, pad_left=self.pad_left)

        # 如果有填充操作，则更新 mask
        if padding != 0:
            offset = (0, padding) if not self.pad_left else (padding, 0)
            new_mask = F.pad(input_mask, offset, value=False)
            kwargs.update(input_mask=new_mask)

        # 调用 net 的前向传播方法
        out = self.net(x, **kwargs)

        # 根据填充方式获取输出张量的切片
        output_slice = slice(0, t) if not self.pad_left else slice(padding, None)
        return out[:, output_slice]
```