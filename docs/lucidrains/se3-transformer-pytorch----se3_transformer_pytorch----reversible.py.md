# `.\lucidrains\se3-transformer-pytorch\se3_transformer_pytorch\reversible.py`

```
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
from torch.autograd.function import Function  # 导入 PyTorch 中的自动微分函数
from torch.utils.checkpoint import get_device_states, set_device_states  # 导入 PyTorch 中的检查点函数

# 辅助函数

def map_values(fn, x):  # 定义一个函数，对字典中的值应用给定函数
    out = {}
    for (k, v) in x.items():
        out[k] = fn(v)
    return out

def dict_chunk(x, chunks, dim):  # 定义一个函数，将字典中的值按给定维度和块数进行分块
    out1 = {}
    out2 = {}
    for (k, v) in x.items():
        c1, c2 = v.chunk(chunks, dim=dim)
        out1[k] = c1
        out2[k] = c2
    return out1, out2

def dict_sum(x, y):  # 定义一个函数，对两个字典中的值进行相加
    out = {}
    for k in x.keys():
        out[k] = x[k] + y[k]
    return out

def dict_subtract(x, y):  # 定义一个函数，对两个字典中的值进行相减
    out = {}
    for k in x.keys():
        out[k] = x[k] - y[k]
    return out

def dict_cat(x, y, dim):  # 定义一个函数，对两个字典中的值按给定维度进行拼接
    out = {}
    for k, v1 in x.items():
        v2 = y[k]
        out[k] = torch.cat((v1, v2), dim=dim)
    return out

def dict_set_(x, key, value):  # 定义一个函数，设置字典中所有值的指定属性为给定值
    for k, v in x.items():
        setattr(v, key, value)

def dict_backwards_(outputs, grad_tensors):  # 定义一个函数，对字典中的值进行反向传播
    for k, v in outputs.items():
        torch.autograd.backward(v, grad_tensors[k], retain_graph=True)

def dict_del_(x):  # 定义一个函数，删除字典中的所有值
    for k, v in x.items():
        del v
    del x

def values(d):  # 定义一个函数，返回字典中所有值的列表
    return [v for _, v in d.items()]

# 参考以下示例保存和设置随机数生成器 https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
class Deterministic(nn.Module):  # 定义一个类，用于确定性计算
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):  # 记录随机数生成器状态
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):  # 前向传播函数
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
# 一旦多 GPU 工作正常，重构并将 PR 发回源代码
class ReversibleBlock(nn.Module):  # 定义一个可逆块
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, **kwargs):  # 前向传播函数
        training = self.training
        x1, x2 = dict_chunk(x, 2, dim=-1)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = dict_sum(x1, self.f(x2, record_rng=training, **kwargs))
            y2 = dict_sum(x2, self.g(y1, record_rng=training))

        return dict_cat(y1, y2, dim=-1)
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
        # 从kwargs中弹出键为'input_keys'的值
        split_dims = kwargs.pop('split_dims')
        # 从kwargs中弹出键为'split_dims'的值
        input_values = x.split(split_dims, dim = -1)
        # 将输入x按照split_dims进行分割，得到输入值列表
        x = dict(zip(input_keys, input_values))
        # 将输入键和值列表组合成字典

        ctx.kwargs = kwargs
        ctx.split_dims = split_dims
        ctx.input_keys = input_keys
        # 将kwargs、split_dims和input_keys保存在上下文对象ctx中

        for block in blocks:
            x = block(x, **kwargs)
        # 遍历blocks中的每个块，对输入x进行处理

        ctx.y = map_values(lambda t: t.detach(), x)
        # 将x中的值进行detach操作，保存在ctx.y中
        ctx.blocks = blocks
        # 将blocks保存在ctx.blocks中

        x = torch.cat(values(x), dim = -1)
        # 将x中的值按照dim = -1进行拼接
        return x
        # 返回处理后的x

    @staticmethod
    def backward(ctx, dy):
        # 定义静态方法backward，接受输入dy
        y = ctx.y
        kwargs = ctx.kwargs
        input_keys = ctx.input_keys
        split_dims = ctx.split_dims
        # 从上下文对象ctx中获取y、kwargs、input_keys和split_dims

        dy = dy.split(split_dims, dim = -1)
        # 将dy按照split_dims进行分割
        dy = dict(zip(input_keys, dy))
        # 将分割后的dy与input_keys组合成字典

        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        # 逆序遍历ctx.blocks中的每个块，对y和dy进行反向传播

        dy = torch.cat(values(dy), dim = -1)
        # 将dy中的值按照dim = -1进行拼接
        return dy, None, None
        # 返回处理后的dy，以及None值



class SequentialSequence(nn.Module):
    # 定义一个继承自nn.Module的类SequentialSequence
    def __init__(self, blocks):
        # 初始化方法，接受blocks作为参数
        super().__init__()
        self.blocks = blocks
        # 调用父类的初始化方法，并将blocks保存在self.blocks中

    def forward(self, x, **kwargs):
        # 前向传播方法，接受输入x和kwargs
        for (attn, ff) in self.blocks:
            x = attn(x, **kwargs)
            x = ff(x)
        # 遍历self.blocks中的每个元素，对输入x进行处理
        return x
        # 返回处理后的x



class ReversibleSequence(nn.Module):
    # 定义一个继承自nn.Module的类ReversibleSequence
    def __init__(self, blocks):
        # 初始化方法，接受blocks作为参数
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for (f, g) in blocks])
        # 调用父类的初始化方法，并将blocks中的每个元素(f, g)构建成ReversibleBlock对象保存在self.blocks中

    def forward(self, x, **kwargs):
        # 前向传播方法，接受输入x和kwargs
        blocks = self.blocks

        x = map_values(lambda t: torch.cat((t, t), dim = -1), x)
        # 对输入x中的值进行操作，将每个值与自身拼接

        input_keys = x.keys()
        split_dims = tuple(map(lambda t: t.shape[-1], x.values()))
        # 获取输入x的键和每个值的最后一个维度大小，保存在split_dims中
        block_kwargs = {'input_keys': input_keys, 'split_dims': split_dims, **kwargs}
        # 构建块的参数字典，包括input_keys、split_dims和kwargs

        x = torch.cat(values(x), dim = -1)
        # 将输入x中的值按照dim = -1进行拼接

        x = _ReversibleFunction.apply(x, blocks, block_kwargs)
        # 调用_ReversibleFunction的apply方法进行处理

        x = dict(zip(input_keys, x.split(split_dims, dim = -1)))
        # 将处理后的x按照split_dims进行分割，组合成字典
        x = map_values(lambda t: torch.stack(t.chunk(2, dim = -1)).mean(dim = 0), x)
        # 对x中的值进行操作，拆分成两部分，取平均值
        return x
        # 返回处理后的x
```