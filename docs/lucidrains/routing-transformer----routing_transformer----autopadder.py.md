# `.\lucidrains\routing-transformer\routing_transformer\autopadder.py`

```
# 导入数学库和 PyTorch 库
import math
import torch
# 从 torch 模块中导入 nn 模块
from torch import nn
# 从 routing_transformer 模块中导入 RoutingTransformer 类
from routing_transformer.routing_transformer import RoutingTransformer
# 从 torch.nn.functional 模块中导入 F 别名
import torch.nn.functional as F

# 定义一个函数，用于查找指定类型的模块
def find_module(nn_module, type):
    # 遍历 nn_module 中的所有模块
    for module in nn_module.modules():
        # 如果模块是指定类型的实例，则返回该模块
        if isinstance(module, type):
            return module
    # 如果未找到指定类型的模块，则返回 None
    return None

# 定义一个函数，用于将张量填充到指定的倍数
def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    # 获取张量在指定维度上的长度
    seqlen = tensor.shape[dim]
    # 计算需要填充的长度
    m = seqlen / multiple
    # 如果 m 是整数，则无需填充，直接返回原张量
    if m.is_integer():
        return tensor

    # 计算填充前的偏移量和填充长度
    pre_pad_offset = (0,) * (-1 - dim) * 2
    padding = math.ceil(m) * multiple - seqlen
    # 对张量进行填充操作
    padded_tensor = F.pad(tensor, (*pre_pad_offset, *(0, padding)), value=value)
    return padded_tensor

# 定义一个自动填充器类，继承自 nn.Module
class Autopadder(nn.Module):
    def __init__(self, net):
        super().__init__()
        # 查找 RoutingTransformer 类型的模块
        transformer = find_module(net, RoutingTransformer)
        self.net = net
        # 获取 RoutingTransformer 模块的 pad_to_multiple 属性
        self.pad_multiple = transformer.pad_to_multiple

    def forward(self, x, **kwargs):
        # 如果 pad_multiple 小于等于 0，则直接调用网络的 forward 方法
        if self.pad_multiple <= 0:
            return self.net(x, **kwargs)

        # 获取输入张量 x 的形状和设备信息
        b, t, device = *x.shape, x.device

        # 获取输入参数中的 input_mask，如果不存在则创建全为 True 的 mask 张量
        input_mask = kwargs.get('input_mask')
        if input_mask is None:
            input_mask = torch.full((b, t), True, device=device, dtype=torch.bool)

        # 对输入张量和 mask 张量进行填充操作
        x = pad_to_multiple(x, self.pad_multiple, dim=1)
        new_mask = pad_to_multiple(input_mask, self.pad_multiple, dim=1, value=False)
        kwargs.update(input_mask=new_mask)

        # 调用网络的 forward 方法，���返回结果
        out, loss = self.net(x, **kwargs)
        return out[:, 0:t], loss
```