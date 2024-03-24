# `.\lucidrains\sinkhorn-transformer\sinkhorn_transformer\autopadder.py`

```py
# 导入数学库
import math
# 导入 PyTorch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.nn.functional 中导入 F 模块
import torch.nn.functional as F
# 从 sinkhorn_transformer.sinkhorn_transformer 中导入 SinkhornTransformer 和 SinkhornTransformerLM 类
from sinkhorn_transformer.sinkhorn_transformer import SinkhornTransformer, SinkhornTransformerLM

# 定义一个函数，用于查找指定类型的模块
def find_module(nn_module, type):
    # 遍历 nn_module 中的所有模块
    for module in nn_module.modules():
        # 如果模块是指定类型的实例，则返回该模块
        if isinstance(module, type):
            return module
    # 如果没有找到指定类型的模块，则返回 None
    return None

# 定义一个函数，用于将张量填充到指定的倍数
def pad_to_multiple(tensor, multiple, dim=-1, pad_left = False):
    # 获取张量在指定维度上的长度
    seqlen = tensor.shape[dim]
    # 计算需要填充的长度
    m = seqlen / multiple
    # 如果 m 是整数，则不需要填充
    if m.is_integer():
        return tensor, 0

    # 计算填充前的偏移量
    pre_pad_offset = (0,) * (-1 - dim) * 2
    # 计算需要填充的长度
    padding = math.ceil(m) * multiple - seqlen
    # 根据填充方式进行填充
    offset = (padding, 0) if pad_left else (0, padding)
    # 对张量进行填充操作
    padded_tensor = F.pad(tensor, (*pre_pad_offset, *offset), value=0)
    return padded_tensor, padding

# 定义一个自动填充器类
class Autopadder(nn.Module):
    def __init__(self, net, pad_left=False):
        super().__init__()
        # 断言 net 是 SinkhornTransformer 或 SinkhornTransformerLM 类的实例
        assert isinstance(net, (SinkhornTransformer, SinkhornTransformerLM)), 'only modules SinkhornTransformer and SinkhornTransformerLM accepted'
        self.net = net

        # 判断 net 是否为 SinkhornTransformerLM 类的实例
        is_lm = isinstance(net, SinkhornTransformerLM)
        # 查找 net 中的 SinkhornTransformer 模块
        sinkhorn = find_module(net, SinkhornTransformer)
        # 获取填充到桶大小的值
        self.bucket_size = sinkhorn.pad_to_bucket_size
        # 获取上下文桶大小的值
        self.context_bucket_size = sinkhorn.context_bucket_size

        # 根据 net 的类型确定填充的维度
        self.pad_dim = -1 if is_lm else -2
        # 设置填充的方式
        self.pad_left = pad_left

    # 定义前向传播函数
    def forward(self, x, **kwargs):
        # 获取输入张��的 batch 大小和时间步长
        b, t, device = *x.shape[:2], x.device

        # 获取关键字参数中的上下文和输入掩码
        context = kwargs.get('context')
        input_mask = kwargs.get('input_mask')
        context_mask = kwargs.get('context_mask')

        # 如果输入掩码为空，则创建一个全为 True 的掩码张量
        if input_mask is None:
            input_mask = torch.full(x.shape[:2], True, device=x.device, dtype=torch.bool)

        # 如果存在上下文且上下文掩码为空，则创建一个全为 True 的上下文掩码张量
        if context is not None and context_mask is None:
            context_mask = torch.full(context.shape[0:2], True, device=x.device, dtype=torch.bool)

        # 对输入张量进行填充操作
        x, padding = pad_to_multiple(x, self.bucket_size, dim=self.pad_dim, pad_left=self.pad_left)

        # 如果有填充操作，则更新输入掩码
        if padding != 0:
            offset = (0, padding) if not self.pad_left else (padding, 0)
            new_mask = F.pad(input_mask, offset, value=False)
            kwargs.update(input_mask=new_mask)

        # 如果存在上下文，则对上下文进行填充操作
        if context is not None:
            context, context_padding = pad_to_multiple(context, self.context_bucket_size, dim=-2)

            # 如果有填充操作，则更新上下文掩码
            if context_padding != 0:
                new_mask = F.pad(context_mask, (0, context_padding), value=False)
                kwargs.update(context_mask=new_mask)

            # 更新关键字参数中的上下文
            kwargs.update(context=context)

        # 调用 net 的前向传播函数
        out = self.net(x, **kwargs)

        # 根据填充方式获取输出切片
        output_slice = slice(0, t) if not self.pad_left else slice(padding, None)
        return out[:, output_slice]
```