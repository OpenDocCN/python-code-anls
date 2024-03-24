# `.\lucidrains\equiformer-pytorch\equiformer_pytorch\reversible.py`

```
import torch
from torch.nn import Module
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

from beartype import beartype
from beartype.typing import List, Tuple

from einops import rearrange, reduce

from equiformer_pytorch.utils import to_order

# helpers

# 将函数 fn 应用于字典 x 中的每个值，并返回新的字典
def map_values(fn, x):
    out = {}
    for (k, v) in x.items():
        out[k] = fn(v)
    return out

# 将字典 x 中的值按照指定维度 dim 进行分块，返回两个新的字典
def dict_chunk(x, chunks, dim):
    out1 = {}
    out2 = {}
    for (k, v) in x.items():
        c1, c2 = v.chunk(chunks, dim = dim)
        out1[k] = c1
        out2[k] = c2
    return out1, out2

# 将两个字典 x 和 y 中的对应值相加，并返回新的字典
def dict_sum(x, y):
    out = {}
    for k in x.keys():
        out[k] = x[k] + y[k]
    return out

# 将两个字典 x 和 y 中的对应值相减，并返回新的字典
def dict_subtract(x, y):
    out = {}
    for k in x.keys():
        out[k] = x[k] - y[k]
    return out

# 将两个字典 x 和 y 中的对应值在指定维度 dim 上进行拼接，并返回新的字典
def dict_cat(x, y, dim):
    out = {}
    for k, v1 in x.items():
        v2 = y[k]
        out[k] = torch.cat((v1, v2), dim = dim)
    return out

# 设置字典 x 中所有值的指定属性 key 为指定值 value
def dict_set_(x, key, value):
    for k, v in x.items():
        setattr(v, key, value)

# 对字典 outputs 中的值进行反向传播，使用 grad_tensors 中的梯度
def dict_backwards_(outputs, grad_tensors):
    for k, v in outputs.items():
        torch.autograd.backward(v, grad_tensors[k], retain_graph = True)

# 删除字典 x 中的所有值
def dict_del_(x):
    for k, v in x.items():
        del v
    del x

# 返回字典 d 中所有值的列表
def values(d):
    return [v for _, v in d.items()]

# following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html

# 定义一个继承自 Module 的类 Deterministic
class Deterministic(Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    # 记录随机数生成器状态
    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    # 前向传播函数
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

# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source

# 定义一个继承自 Module 的类 ReversibleBlock
class ReversibleBlock(Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    # 前向传播函数
    def forward(self, x, **kwargs):
        training = self.training
        x1, x2 = dict_chunk(x, 2, dim = -1)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = dict_sum(x1, self.f(x2, record_rng = training, **kwargs))
            y2 = dict_sum(x2, self.g(y1, record_rng = training))

        return dict_cat(y1, y2, dim = -1)
    # 定义反向传播函数，接收输入 y、梯度 dy 和其他参数
    def backward_pass(self, y, dy, **kwargs):
        # 将 y 按照指定维度分成两部分 y1 和 y2
        y1, y2 = dict_chunk(y, 2, dim = -1)
        # 删除原始 y 字典
        dict_del_(y)

        # 将 dy 按照指定维度分成两部分 dy1 和 dy2
        dy1, dy2 = dict_chunk(dy, 2, dim = -1)
        # 删除原始 dy 字典
        dict_del_(dy)

        # 开启梯度追踪
        with torch.enable_grad():
            # 设置 y1 的 requires_grad 为 True
            dict_set_(y1, 'requires_grad', True)
            # 计算 y1 的梯度 gy1
            gy1 = self.g(y1, set_rng = True)
            # 对 gy1 进行反向传播，传入 dy2
            dict_backwards_(gy1, dy2)

        # 关闭梯度追踪
        with torch.no_grad():
            # 计算 x2，即 y2 减去 gy1
            x2 = dict_subtract(y2, gy1)
            # 删除 y2 和 gy1
            dict_del_(y2)
            dict_del_(gy1)

            # 计算 dx1，即 dy1 加上 y1 中各张量的梯度
            dx1 = dict_sum(dy1, map_values(lambda t: t.grad, y1))
            # 删除 dy1，并将 y1 的梯度设为 None
            dict_del_(dy1)
            dict_set_(y1, 'grad', None)

        # 开启梯度追踪
        with torch.enable_grad():
            # 设置 x2 的 requires_grad 为 True
            dict_set_(x2, 'requires_grad', True)
            # 计算 fx2，即对 x2 进行操作并计算梯度
            fx2 = self.f(x2, set_rng = True, **kwargs)
            # 对 fx2 进行反向传播，传入 dx1
            dict_backwards_(fx2, dx1)

        # 关闭梯度追踪
        with torch.no_grad():
            # 计算 x1，即 y1 减去 fx2
            x1 = dict_subtract(y1, fx2)
            # 删除 y1 和 fx2
            dict_del_(y1)
            dict_del_(fx2)

            # 计算 dx2，即 dy2 加上 x2 中各张量的梯度
            dx2 = dict_sum(dy2, map_values(lambda t: t.grad, x2))
            # 删除 dy2，并将 x2 的梯度设为 None
            dict_del_(dy2)
            dict_set_(x2, 'grad', None)

            # 将 x2 中的张量都 detach，即不再追踪梯度
            x2 = map_values(lambda t: t.detach(), x2)

            # 将 x1 和 x2 按照指定维度拼接成 x
            x = dict_cat(x1, x2, dim = -1)
            # 将 dx1 和 dx2 按照指定维度拼接成 dx
            dx = dict_cat(dx1, dx2, dim = -1)

        # 返回拼接后的 x 和 dx
        return x, dx
class _ReversibleFunction(Function):
    # 定义一个继承自Function的类_ReversibleFunction
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        # 定义静态方法forward，接受输入x、blocks和kwargs
        input_keys = kwargs.pop('input_keys')
        # 从kwargs中弹出键为'input_keys'的值，赋给input_keys
        split_dims = kwargs.pop('split_dims')
        # 从kwargs中弹出键为'split_dims'的值，赋给split_dims
        input_values = x.split(split_dims, dim = -1)
        # 将输入x按照split_dims在最后一个维度上分割，得到input_values

        x = dict(zip(input_keys, input_values))
        # 将input_keys和input_values打包成字典，赋给x

        ctx.kwargs = kwargs
        ctx.split_dims = split_dims
        ctx.input_keys = input_keys
        # 将kwargs、split_dims和input_keys保存在ctx中

        x = {k: rearrange(v, '... (d m) -> ... d m', m = to_order(k) * 2) for k, v in x.items()}
        # 对x中的每个键值对进行重排列操作，重新赋值给x

        for block in blocks:
            x = block(x, **kwargs)
        # 遍历blocks中的每个块，对x进行处理

        ctx.y = map_values(lambda t: t.detach(), x)
        ctx.blocks = blocks
        # 将x中的值进行detach操作后保存在ctx.y中，保存blocks到ctx中

        x = map_values(lambda t: rearrange(t, '... d m -> ... (d m)'), x)
        x = torch.cat(values(x), dim = -1)
        # 对x中的值进行重排列和拼接操作

        return x
        # 返回处理后的x

    @staticmethod
    def backward(ctx, dy):
        # 定义静态方法backward，接受输入dy
        y = ctx.y
        kwargs = ctx.kwargs
        input_keys = ctx.input_keys
        split_dims = ctx.split_dims
        # 从ctx中获取y、kwargs、input_keys和split_dims

        dy = dy.split(split_dims, dim = -1)
        dy = dict(zip(input_keys, dy))
        # 将dy按照split_dims在最后一个维度上分割，打包成字典

        dy = {k: rearrange(v, '... (d m) -> ... d m', m = to_order(k) * 2) for k, v in dy.items()}
        # 对dy中的每个键值对进行重排列操作

        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        # 逆序遍历ctx.blocks中的每个块，对y和dy进行处理

        dy = map_values(lambda t: rearrange(t, '... d m -> ... (d m)'), dy)
        dy = torch.cat(values(dy), dim = -1)
        # 对dy中的值进行重排列和拼接操作

        return dy, None, None
        # 返回处理后的dy和两个None值

# sequential

def residual_fn(x, residual):
    # 定义一个函数residual_fn，接受输入x和residual
    out = {}
    # ��始化一个空字典out

    for degree, tensor in x.items():
        # 遍历x中的每个键值对
        out[degree] = tensor
        # 将键值对中的值赋给out对应的键

        if degree not in residual:
            continue
        # 如果degree不在residual中，则继续下一次循环

        if not any(t.requires_grad for t in (out[degree], residual[degree])):
            out[degree] += residual[degree]
        else:
            out[degree] = out[degree] + residual[degree]
        # 如果out[degree]和residual[degree]中有任意一个张量需要梯度，则相加，否则直接赋值相加

    return out
    # 返回处理后的out字典

class SequentialSequence(Module):
    # 定义一个继承自Module的类SequentialSequence

    @beartype
    def __init__(
        self,
        blocks: List[Tuple[Module, Module]]
    ):
        # 初始化方法，接受blocks参数，类型为包含元组的列表
        super().__init__()
        # 调用父类的初始化方法

        self.blocks = nn.ModuleList([nn.ModuleList([f, g]) for f, g in blocks])
        # 将blocks中的每个元组(f, g)转换为nn.ModuleList，再转换为nn.ModuleList，赋给self.blocks

    def forward(self, x, **kwargs):
        # 定义前向传播方法，接受输入x和kwargs

        for attn, ff in self.blocks:
            # 遍历self.blocks中的每个元组(attn, ff)
            x = residual_fn(attn(x, **kwargs), x)
            # 对attn(x, **kwargs)和x进行残差连接后赋给x
            x = residual_fn(ff(x), x)
            # 对ff(x)和x进行残差连接后赋给x

        return x
        # 返回处理后的x

# reversible

class ReversibleSequence(Module):
    # 定义一个继承自Module的类ReversibleSequence

    @beartype
    def __init__(
        self,
        blocks: List[Tuple[Module, Module]]
    ):
        # 初始化方法，接受blocks参数，类型为包含元组的列表
        super().__init__()
        # 调用父类的初始化方法

        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for (f, g) in blocks])
        # 将blocks中的每个元组(f, g)转换为ReversibleBlock，再转换为nn.ModuleList，赋给self.blocks

    def forward(self, x, **kwargs):
        # 定义前向传播方法，接受输入x和kwargs
        blocks = self.blocks
        # 将self.blocks赋给blocks

        # merge into single tensor

        x = map_values(lambda t: torch.cat((t, t), dim = -1), x)
        # 对x中的每个值进行拼接操作
        x = map_values(lambda t: rearrange(t, '... d m -> ... (d m)'), x)
        # 对x中的每个值进行重排列操作

        input_keys = x.keys()
        # 获取x的键集合

        split_dims = tuple(map(lambda t: t.shape[-1], x.values()))
        # 获取x中每个值在最后一个维度上的大小，转换为元组赋给split_dims
        block_kwargs = {'input_keys': input_keys, 'split_dims': split_dims, **kwargs}
        # 构建块的参数字典

        x = torch.cat(values(x), dim = -1)
        # 对x中的值进行拼接操作

        # reversible function, tailored for equivariant network

        x = _ReversibleFunction.apply(x, blocks, block_kwargs)
        # 调用_ReversibleFunction的apply方法处理x

        # reconstitute

        x = dict(zip(input_keys, x.split(split_dims, dim = -1)))
        # 将x按照split_dims在最后一个维度上分割，打包成字典

        x = {k: reduce(v, '... (d r m) -> ... d m', 'mean', r = 2, m = to_order(k)) for k, v in x.items()}
        # 对x中的每个键值对进行降维操作

        return x
        # 返回处理后的x
```