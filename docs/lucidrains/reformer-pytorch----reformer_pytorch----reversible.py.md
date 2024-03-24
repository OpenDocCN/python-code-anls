# `.\lucidrains\reformer-pytorch\reformer_pytorch\reversible.py`

```
import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states



# 创建一个继承自 nn.Module 的 Deterministic 类，用于记录和设置随机数生成器状态
# 参考链接：https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
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



# 创建一个继承自 nn.Module 的 ReversibleBlock 类，用于实现可逆块
# 受 https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py 启发
# 一旦多 GPU 工作正常，重构并向源发送 PR
class ReversibleBlock(nn.Module):
    def __init__(self, f, g, depth=None, send_signal = False):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

        self.depth = depth
        self.send_signal = send_signal

    def forward(self, x, f_args = {}, g_args = {}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None

        if self.send_signal:
            f_args['_reverse'] = g_args['_reverse'] = False
            f_args['_depth'] = g_args['_depth'] = self.depth

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=2)

    def backward_pass(self, y, dy, f_args = {}, g_args = {}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy

        if self.send_signal:
            f_args['_reverse'] = g_args['_reverse'] = True
            f_args['_depth'] = g_args['_depth'] = self.depth

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



# 创建一个继承自 nn.Module 的 IrreversibleBlock 类，用于实现不可逆块
class IrreversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x, f_args, g_args):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1 = x1 + self.f(x2, **f_args)
        y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim=2)



# 创建一个继承自 Function 的 _ReversibleFunction 类，用于实现可逆函数
class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    # 定义一个反向传播函数，接收上下文和梯度作为参数
    def backward(ctx, dy):
        # 从上下文中获取 y 值
        y = ctx.y
        # 从上下文中获取关键字参数
        kwargs = ctx.kwargs
        # 反向遍历上下文中的块列表
        for block in ctx.blocks[::-1]:
            # 调用每个块的反向传播方法，更新 y 和 dy
            y, dy = block.backward_pass(y, dy, **kwargs)
        # 返回更新后的梯度
        return dy, None, None
# 定义一个可逆序列的神经网络模块
class ReversibleSequence(nn.Module):
    # 初始化函数，接受一些参数用于构建可逆序列
    def __init__(self, blocks, layer_dropout = 0., reverse_thres = 0, send_signal = False):
        super().__init__()
        # 设置层级丢弃率和反转阈值
        self.layer_dropout = layer_dropout
        self.reverse_thres = reverse_thres

        # 创建可逆块的模块列表，根据是否需要反转选择不同的块
        self.blocks = nn.ModuleList([ReversibleBlock(f, g, depth, send_signal) for depth, (f, g) in enumerate(blocks)])
        self.irrev_blocks = nn.ModuleList([IrreversibleBlock(f=f, g=g) for f, g in blocks])

    # 前向传播函数，接受输入和一些参数，根据是否需要反转选择不同的块进行处理
    def forward(self, x, arg_route = (True, False), **kwargs):
        # 判断是否需要反转
        reverse = x.shape[1] > self.reverse_thres
        blocks = self.blocks if reverse else self.irrev_blocks

        # 如果处于训练状态且设置了层级丢弃率
        if self.training and self.layer_dropout > 0:
            # 随机选择是否丢弃某些块
            to_drop = torch.empty(len(self.blocks)).uniform_(0, 1) < self.layer_dropout
            blocks = [block for block, drop in zip(self.blocks, to_drop) if not drop]
            blocks = self.blocks[:1] if len(blocks) == 0 else blocks

        # 根据参数路由设置不同的参数
        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {'f_args': f_args, 'g_args': g_args}

        # 如果不需要反转，则依次对每个块进行处理
        if not reverse:
            for block in blocks:
                x = block(x, **block_kwargs)
            return x

        # 如果需要反转，则调用自定义的可逆函数进行处理
        return _ReversibleFunction.apply(x, blocks, block_kwargs)
```