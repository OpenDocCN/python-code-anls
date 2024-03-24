# `.\lucidrains\lie-transformer-pytorch\lie_transformer_pytorch\reversible.py`

```
# 导入 torch 库
import torch
# 导入 torch 中的神经网络模块
import torch.nn as nn
# 从 torch.autograd.function 中导入 Function 类
from torch.autograd.function import Function
# 从 torch.utils.checkpoint 中导入 get_device_states 和 set_device_states 函数

# 辅助函数

# 对元组中指定维度的元素求和
def sum_tuple(x, y, dim = 1):
    x = list(x)
    x[dim] += y[dim]
    return tuple(x)

# 对元组中指定维度的元素求差
def subtract_tuple(x, y, dim = 1):
    x = list(x)
    x[dim] -= y[dim]
    return tuple(x)

# 设置元组中指定维度的值
def set_tuple(x, dim, value):
    x = list(x).copy()
    x[dim] = value
    return tuple(x)

# 对元组中指定维度的元素应用函数
def map_tuple(fn, x, dim = 1):
    x = list(x)
    x[dim] = fn(x[dim])
    return tuple(x)

# 对元组中指定维度的元素进行分块
def chunk_tuple(fn, x, dim = 1):
    x = list(x)
    value = x[dim]
    chunks = fn(value)
    return tuple(map(lambda t: set_tuple(x, 1, t), chunks))

# 将两个元组在指定维度进行拼接
def cat_tuple(x, y, dim = 1, cat_dim = -1):
    x = list(x)
    y = list(y)
    x[dim] = torch.cat((x[dim], y[dim]), dim = cat_dim)
    return tuple(x)

# 删除元组中的元素
def del_tuple(x):
    for el in x:
        if el is not None:
            del el

# 根据 https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html 中的示例，实现保存和设置随机数生成器状态的类
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng = False, set_rng = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

# 受 https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py 启发，实现可逆块类
# 一旦多 GPU 确认工作正常，重构并将 PR 发回源代码
class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args = {}, g_args = {}):
        training = self.training
        x1, x2 = chunk_tuple(lambda t: torch.chunk(t, 2, dim=2), x)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = sum_tuple(self.f(x2, record_rng = training, **f_args), x1)
            y2 = sum_tuple(self.g(y1, record_rng = training, **g_args), x2)

        return cat_tuple(y1, y2, cat_dim = 2)

    def backward_pass(self, y, dy, f_args = {}, g_args = {}):
        y1, y2 = chunk_tuple(lambda t: torch.chunk(t, 2, dim=2), y)
        del_tuple(y)

        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy

        with torch.enable_grad():
            y1[1].requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1[1], dy2)

        with torch.no_grad():
            x2 = subtract_tuple(y2, gy1)
            del_tuple(y2)
            del gy1

            dx1 = dy1 + y1[1].grad
            del dy1
            y1[1].grad = None

        with torch.enable_grad():
            x2[1].requires_grad = True
            fx2 = self.f(x2, set_rng = True, **f_args)
            torch.autograd.backward(fx2[1], dx1)

        with torch.no_grad():
            x1 = subtract_tuple(y1, fx2)
            del fx2
            del_tuple(y1)

            dx2 = dy2 + x2[1].grad
            del dy2
            x2[1].grad = None

            x2 = map_tuple(lambda t: t.detach(), x2)
            x = cat_tuple(x1, x2, cat_dim = -1)
            dx = torch.cat((dx1, dx2), dim=2)

        return x, dx

class _ReversibleFunction(Function):
    # 定义一个静态方法，用于前向传播
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        # 将传入的参数保存在上下文中
        ctx.kwargs = kwargs
        # 将传入的参数重新组合
        x = (kwargs.pop('coords'), x, kwargs.pop('mask'), kwargs.pop('edges'))
    
        # 遍历每个块并进行前向传播
        for block in blocks:
            x = block(x, **kwargs)
    
        # 将计算结果保存在上下文中，并将梯度分离
        ctx.y = map_tuple(lambda t: t.detach(), x, dim=1)
        ctx.blocks = blocks
        # 返回计算结果的第二个元素
        return x[1]
    
    # 定义一个静态方法，用于反向传播
    @staticmethod
    def backward(ctx, dy):
        # 从上下文中获取保存的数据
        y = ctx.y
        kwargs = ctx.kwargs
    
        # 反向遍历每个块并进行反向传播
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        # 返回计算结果的梯度
        return dy, None, None
class SequentialSequence(nn.Module):
    # 定义一个顺序执行的序列模块
    def __init__(self, blocks):
        # 初始化函数，接受一个包含多个块的列表作为参数
        super().__init__()
        # 调用父类的初始化函数
        self.blocks = blocks
        # 将传入的块列表保存在当前对象的属性中

    def forward(self, x):
        # 前向传播函数，接受输入参数 x
        for (f, g) in self.blocks:
            # 遍历块列表中的每个块，每个块包含两个函数 f 和 g
            x = sum_tuple(f(x), x, dim = 1)
            # 将 f 函数作用在输入 x 上，然后与 x 求和，指定维度为 1
            x = sum_tuple(g(x), x, dim = 1)
            # 将 g 函数作用在上一步的结果 x 上，然后与 x 求和，指定维度为 1
        return x
        # 返回最终结果 x

class ReversibleSequence(nn.Module):
    # 定义一个可逆执行的序列模块
    def __init__(self, blocks):
        # 初始化函数，接受一个包含多个块的列表作为参数
        super().__init__()
        # 调用父类的初始化函数
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for (f, g) in blocks])
        # 将传入的块列表中的每个块转换为 ReversibleBlock 对象，并保存在当前对象的属性中

    def forward(self, x, **kwargs):
        # 前向传播函数，接受输入参数 x 和关键字参数 kwargs
        x = map_tuple(lambda t: torch.cat((t, t), dim = -1), x)
        # 对输入 x 中的每个元素应用 lambda 函数，将其在最后一个维度上进行拼接

        blocks = self.blocks
        # 将当前对象的块列表保存在变量 blocks 中

        coords, values, mask, edges = x
        # 将输入 x 拆分为 coords、values、mask 和 edges 四部分
        kwargs = {'coords': coords, 'mask': mask, 'edges': edges, **kwargs}
        # 将 coords、mask、edges 和 kwargs 合并为一个字典
        x = _ReversibleFunction.apply(values, blocks, kwargs)
        # 调用自定义的 _ReversibleFunction 类的 apply 方法，传入 values、blocks 和 kwargs，得到结果 x

        x = (coords, x, mask, edges)
        # 将 x 重新组合为一个元组
        return map_tuple(lambda t: sum(t.chunk(2, dim = -1)) * 0.5, x)
        # 对 x 中的每个元素应用 lambda 函数，将其在最后一个维度上进行拆分并求和，然后乘以 0.5
```