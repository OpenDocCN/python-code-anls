# `.\lucidrains\routing-transformer\routing_transformer\reversible.py`

```
import torch
import torch.nn as nn
from operator import itemgetter
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

# 用于将参数路由到可逆层函数中的函数

def route_args(router, args, depth):
    # 初始化路由后的参数列表
    routed_args = [(dict(), dict()) for _ in range(depth)]
    # 获取参数中与路由器匹配的键
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            # 根据路由将参数分配到对应的函数中
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args

def layer_drop(layers, prob):
    # 根据概率丢弃层
    to_drop = torch.empty(len(layers)).uniform_(0, 1) < prob
    blocks = [block for block, drop in zip(layers, to_drop) if not drop]
    blocks = layers[:1] if len(blocks) == 0 else blocks
    return blocks

def cast_return(ret, requires_grad = True):
    # 将返回值转换为元组形式，用于梯度计算
    if type(ret) is not tuple:
        loss = torch.tensor(0., device=ret.device, dtype=ret.dtype, requires_grad=requires_grad)
        return (ret, loss)
    return ret

# 参考示例 https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html 进行保存和设置随机数生成器
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        # 记录随机数生成器状态
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
# 一旦多GPU工作正常，重构并将PR发送回源代码
class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args = {}, g_args = {}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None

        f_args['_reverse'] = g_args['_reverse'] = False

        with torch.no_grad():
            f_out, f_loss = cast_return(self.f(x2, record_rng=self.training, **f_args), requires_grad = False)
            y1 = x1 + f_out

            g_out, g_loss = cast_return(self.g(y1, record_rng=self.training, **g_args), requires_grad = False)
            y2 = x2 + g_out

        return torch.cat([y1, y2], dim=2), f_loss, g_loss
    # 定义反向传播函数，接收输入 y、梯度 dy、损失函数 dl_f 和 dl_g，以及额外参数 f_args 和 g_args
    def backward_pass(self, y, dy, dl_f, dl_g, f_args = {}, g_args = {}):
        # 将 y 沿着第二维度分成两部分 y1 和 y2
        y1, y2 = torch.chunk(y, 2, dim=2)
        # 释放 y 变量的内存
        del y

        # 将 dy 沿着第二维度分成两部分 dy1 和 dy2
        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        # 释放 dy 变量的内存
        del dy

        # 设置 f_args 和 g_args 中的 '_reverse' 参数为 True
        f_args['_reverse'] = g_args['_reverse'] = True

        # 启用梯度计算环境
        with torch.enable_grad():
            # 设置 y1 可以计算梯度
            y1.requires_grad = True
            # 调用 self.g 函数计算 gy1 和 g_loss
            gy1, g_loss = cast_return(self.g(y1, set_rng=True, **g_args))
            # 反向传播计算梯度
            torch.autograd.backward((gy1, g_loss), (dy2, dl_g))

        # 禁用梯度计算环境
        with torch.no_grad():
            # 计算 x2
            x2 = y2 - gy1
            # 释放 y2 和 gy1 变量的内存
            del y2, gy1

            # 计算 dx1
            dx1 = dy1 + y1.grad
            # 释放 dy1 变量的内存
            del dy1
            # 清空 y1 的梯度
            y1.grad = None

        # 再次启用梯度计算环境
        with torch.enable_grad():
            # 设置 x2 可以计算梯度
            x2.requires_grad = True
            # 调用 self.f 函数计算 fx2 和 f_loss
            fx2, f_loss = cast_return(self.f(x2, set_rng=True, **f_args))
            # 反向传播计算梯度，保留计算图
            torch.autograd.backward((fx2, f_loss), (dx1, dl_f), retain_graph=True)

        # 禁用梯度计算环境
        with torch.no_grad():
            # 计算 x1
            x1 = y1 - fx2
            # 释放 y1 和 fx2 变量的内存
            del y1, fx2

            # 计算 dx2
            dx2 = dy2 + x2.grad
            # 释放 dy2 变量的内存
            del dy2
            # 清空 x2 的梯度
            x2.grad = None

            # 拼接 x1 和去除梯度的 x2，沿着第二维度
            x = torch.cat([x1, x2.detach()], dim=2)
            # 拼接 dx1 和 dx2，沿着第二维度
            dx = torch.cat([dx1, dx2], dim=2)

        # 返回拼接后的 x 和 dx
        return x, dx
class _ReversibleFunction(Function):
    # 静态方法，定义前向传播逻辑
    @staticmethod
    def forward(ctx, x, blocks, args):
        # 保存参数
        ctx.args = args

        # 初始化辅助损失列表
        f_aux_loss = []
        g_aux_loss = []

        # 遍历每个块并执行前向传播
        for block, kwarg in zip(blocks, args):
            x, f_loss, g_loss = block(x, **kwarg)
            f_aux_loss.append(f_loss)
            g_aux_loss.append(g_loss)

        # 保存中间结果和块信息
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x, torch.stack(f_aux_loss), torch.stack(g_aux_loss)

    # 静态方法，定义反向传播逻辑
    @staticmethod
    def backward(ctx, dy, dl_f, dl_g):
        # 获取保存的中间结果和参数
        y = ctx.y
        args = ctx.args
        # 反向遍历每个块并执行反向传播
        for block, kwargs, ind in zip(ctx.blocks[::-1], args[::-1], range(len(ctx.blocks))[::-1]):
            y, dy = block.backward_pass(y, dy, dl_f[ind], dl_g[ind], **kwargs)
        return dy, None, None

class SequentialSequence(nn.Module):
    # 初始化顺序序列模块
    def __init__(self, layers, args_route = {}, layer_dropout = 0.):
        super().__init__()
        # 断言每个参数路由映射的深度与顺序层的数量相同
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    # 前向传播逻辑
    def forward(self, x, **kwargs):
        # 根据参数路由获取参数
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        # 如果处于训练状态且存在层丢弃率，则执行层丢弃
        if self.training and self.layer_dropout > 0:
            layers_and_args = layer_drop(layers_and_args, self.layer_dropout)

        # 初始化辅助损失
        aux_loss = torch.zeros(1, device=x.device, dtype=x.dtype)

        # 遍历每个层并执行前向传播
        for (f, g), (f_args, g_args) in layers_and_args:
            res, loss = cast_return(f(x, **f_args))
            aux_loss += loss
            x = x + res

            res, loss = cast_return(g(x, **g_args))
            aux_loss += loss
            x = x + res
        return x, aux_loss

class ReversibleSequence(nn.Module):
    # 初始化可逆序列模块
    def __init__(self, blocks, args_route = {}, layer_dropout = 0.):
        super().__init__()
        self.args_route = args_route
        self.layer_dropout = layer_dropout
        # 创建可逆块模块列表
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for f, g in blocks])

    # 前向传播逻辑
    def forward(self, x, **kwargs):
        # 将输入张量在最后一个维度上进行拼接
        x = torch.cat([x, x], dim=-1)

        blocks = self.blocks
        # 根据参数路由获取参数
        args = route_args(self.args_route, kwargs, len(blocks))
        args = list(map(lambda x: {'f_args': x[0], 'g_args': x[1]}, args))

        layers_and_args = list(zip(blocks, args))

        # 如果处于训练状态且存在层丢弃率，则执行层丢弃
        if self.training and self.layer_dropout > 0:
            layers_and_args = layer_drop(layers_and_args, self.layer_dropout)
            blocks, args = map(lambda ind: list(map(itemgetter(ind), layers_and_args)), (0, 1))

        # 调用_ReversibleFunction的apply方法执行前向传播
        out, f_loss, g_loss =  _ReversibleFunction.apply(x, blocks, args)
        # 将输出张量在最后一个维度上分割成两部分并取平均
        out = torch.stack(out.chunk(2, dim=-1)).mean(dim=0)
        # 计算辅助损失
        aux_loss = f_loss.sum() + g_loss.sum()
        return out, aux_loss
```