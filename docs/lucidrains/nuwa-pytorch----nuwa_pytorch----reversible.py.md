# `.\lucidrains\nuwa-pytorch\nuwa_pytorch\reversible.py`

```py
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

    # 遍历匹配的键
    for key in matched_keys:
        val = args[key]
        # 遍历路由后的参数列表和路由器中的路由
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            # 根据路由将参数添加到对应的函数参数中
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args

# 参考示例 https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html 中的保存和设置随机数生成器
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

# 受 https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py 启发
# 一旦多 GPU 确认工作正常，重构并将 PR 发回源代码
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

class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, args):
        ctx.args = args
        for block, kwarg in zip(blocks, args):
            x = block(x, **kwarg)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    # 定义反向传播函数，接收上下文和梯度作为参数
    def backward(ctx, dy):
        # 获取上下文中的 y 和 args
        y = ctx.y
        args = ctx.args
        # 反向遍历上下文中的 blocks 和 args
        for block, kwargs in zip(ctx.blocks[::-1], args[::-1]):
            # 调用每个 block 的反向传播函数，更新 y 和 dy
            y, dy = block.backward_pass(y, dy, **kwargs)
        # 返回更新后的梯度
        return dy, None, None
# 定义一个可逆序列的神经网络模块
class ReversibleSequence(nn.Module):
    # 初始化函数，接受一组块和参数路由作为输入
    def __init__(self, blocks, args_route = {}):
        super().__init__()
        # 将参数路由保存在对象中
        self.args_route = args_route
        # 创建一个包含多个可逆块的模块列表
        self.blocks = nn.ModuleList([ReversibleBlock(f=f, g=g) for f, g in blocks])

    # 前向传播函数
    def forward(self, x, **kwargs):
        # 在最后一个维度上将输入张量 x 进行拼接
        x = torch.cat([x, x], dim=-1)

        # 获取模块列表和参数路由
        blocks = self.blocks
        args = route_args(self.args_route, kwargs, len(blocks))
        # 将参数转换为字典形式
        args = list(map(lambda x: {'f_args': x[0], 'g_args': x[1]}, args))

        # 将块和参数组成元组列表
        layers_and_args = list(zip(blocks, args))

        # 调用自定义的可逆函数 _ReversibleFunction 的前向传播方法
        out =  _ReversibleFunction.apply(x, blocks, args)
        # 在最后一个维度上将输出张量拆分成两部分，然后对它们进行求和
        return torch.stack(out.chunk(2, dim=-1)).sum(dim=0)
```