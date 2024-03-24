# `.\lucidrains\h-transformer-1d\h_transformer_1d\reversible.py`

```
# 导入 torch 库
import torch
# 导入 torch 中的神经网络模块
import torch.nn as nn
# 从 operator 模块中导入 itemgetter 函数
from operator import itemgetter
# 从 torch.autograd.function 模块中导入 Function 类
from torch.autograd.function import Function
# 从 torch.utils.checkpoint 模块中导入 get_device_states 和 set_device_states 函数

# 用于将参数路由到可逆层函数中的函数
def route_args(router, args, depth):
    # 初始化路由后的参数列表
    routed_args = [(dict(), dict()) for _ in range(depth)]
    # 获取参数中与路由器匹配的键
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args

# 根据概率丢弃层的函数
def layer_drop(layers, prob):
    to_drop = torch.empty(len(layers)).uniform_(0, 1) < prob
    blocks = [block for block, drop in zip(layers, to_drop) if not drop]
    blocks = layers[:1] if len(blocks) == 0 else blocks
    return blocks

# 保存和设置随机数种子的类
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

# 可逆块类，受启发于 https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# 一旦多 GPU 工作正常，重构并将 PR 发回源代码
class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args = {}, g_args = {}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=2)

    def backward_pass(self, y, dy, f_args = {}, g_args = {}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)

        return x, dx

# 可逆函数类
class _ReversibleFunction(Function):
    @staticmethod
    # 前向传播函数，接收上下文对象 ctx，输入数据 x，模块列表 blocks 和参数列表 args
    def forward(ctx, x, blocks, args):
        # 将参数列表 args 存储到上下文对象 ctx 中
        ctx.args = args
        # 遍历模块列表 blocks 和参数列表 args，对输入数据 x 进行处理
        for block, kwarg in zip(blocks, args):
            x = block(x, **kwarg)
        # 将处理后的数据 x 分离出来，并存储到上下文对象 ctx 中
        ctx.y = x.detach()
        # 将模块列表 blocks 存储到上下文对象 ctx 中
        ctx.blocks = blocks
        # 返回处理后的数据 x
        return x

    # 反向传播函数，接收上下文对象 ctx 和梯度 dy
    @staticmethod
    def backward(ctx, dy):
        # 获取上下文对象 ctx 中存储的处理后的数据 y 和参数列表 args
        y = ctx.y
        args = ctx.args
        # 反向遍历模块列表 blocks 和参数列表 args，对梯度 dy 进行处理
        for block, kwargs in zip(ctx.blocks[::-1], args[::-1]):
            # 调用模块的反向传播函数，更新梯度 dy 和数据 y
            y, dy = block.backward_pass(y, dy, **kwargs)
        # 返回更新后的梯度 dy
        return dy, None, None
class SequentialSequence(nn.Module):
    # 定义一个顺序执行的神经网络模块
    def __init__(self, layers, args_route = {}, layer_dropout = 0.):
        super().__init__()
        # 断言每个参数路由映射的深度与顺序层的数量相同
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    def forward(self, x, **kwargs):
        # 根据参数路由和关键字参数获取参数
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        if self.training and self.layer_dropout > 0:
            # 如果处于训练状态且存在层丢弃率，则执行层丢弃
            layers_and_args = layer_drop(layers_and_args, self.layer_dropout)

        for (f, g), (f_args, g_args) in layers_and_args:
            # 依次执行每个顺序层的前向传播
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x

class ReversibleSequence(nn.Module):
    # 定义一个可逆的序列神经网络模块
    def __init__(self, blocks, args_route = {}, layer_dropout = 0.):
        super().__init__()
        self.args_route = args_route
        self.layer_dropout = layer_dropout
        # 创建包含可逆块的模块列表
        self.blocks = nn.ModuleList([ReversibleBlock(f=f, g=g) for f, g in blocks])

    def forward(self, x, **kwargs):
        # 在最后一个维度上连接输入张量的副本
        x = torch.cat([x, x], dim=-1)

        blocks = self.blocks
        # 根据参数路由和关键字参数获取参数
        args = route_args(self.args_route, kwargs, len(blocks))
        args = list(map(lambda x: {'f_args': x[0], 'g_args': x[1]}, args))

        layers_and_args = list(zip(blocks, args))

        if self.training and self.layer_dropout > 0:
            # 如果处于训练状态且存在层丢弃率，则执行层丢弃
            layers_and_args = layer_drop(layers_and_args, self.layer_dropout)
            blocks, args = map(lambda ind: list(map(itemgetter(ind), layers_and_args)), (0, 1))

        # 调用自定义的可逆函数进行前向传播
        out =  _ReversibleFunction.apply(x, blocks, args)
        # 在最后一个维度上分割输出并求和
        return torch.stack(out.chunk(2, dim=-1)).sum(dim=0)
```