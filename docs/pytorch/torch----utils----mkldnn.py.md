# `.\pytorch\torch\utils\mkldnn.py`

```
# mypy: allow-untyped-defs
import torch  # 导入 PyTorch 模块


class MkldnnLinear(torch.jit.ScriptModule):
    def __init__(self, dense_module, dtype):
        super().__init__()
        # 将输入的 dense_module 的权重转换为 MKL-DNN 格式并注册为缓冲区
        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))
        if dense_module.bias is not None:
            # 如果存在偏置，将其转换为 MKL-DNN 格式并注册为缓冲区
            # 偏置可以是 fp32 或 bf16，但为了获得更好的准确性，我们使用 fp32 类型
            self.register_buffer('bias', dense_module.bias.to_mkldnn())
        else:
            # 如果不存在偏置，创建一个全零的 Tensor 并转换为 MKL-DNN 格式后注册为缓冲区
            # TODO: 一旦 ScriptModule 支持注册 None 缓冲区，应该删除此部分
            self.register_buffer(
                'bias',
                torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())

    @torch.jit.script_method
    def __getstate__(self):
        # 返回模型状态的方法，包括权重、偏置和训练状态
        return (self.weight.to_dense(), self.bias.to_dense(), self.training)

    @torch.jit.script_method
    def __setstate__(self, state):
        # 设置模型状态的方法，将传入的状态恢复为权重、偏置和训练状态
        self.weight = state[0].to_mkldnn()
        self.bias = state[1].to_mkldnn()
        self.training = state[2]

    @torch.jit.script_method
    def forward(self, x):
        # 前向传播方法，将输入 x 转换为 MKL-DNN 格式（如果不是的话），然后执行 MKL-DNN 线性层操作
        x_mkldnn = x if x.is_mkldnn else x.to_mkldnn()
        y_mkldnn = torch._C._nn.mkldnn_linear(x_mkldnn, self.weight, self.bias)
        y = y_mkldnn if x.is_mkldnn else y_mkldnn.to_dense()
        return y


class _MkldnnConvNd(torch.jit.ScriptModule):
    """MkldnnConv1d 和 MkldnnConv2d 的共同基类."""

    __constants__ = ['stride', 'padding', 'dilation', 'groups']

    def __init__(self, dense_module):
        super().__init__()

        # 初始化卷积参数：步幅、填充、膨胀、分组
        self.stride = dense_module.stride
        self.padding = dense_module.padding
        self.dilation = dense_module.dilation
        self.groups = dense_module.groups

        if dense_module.bias is not None:
            # 如果存在偏置，将其转换为 MKL-DNN 格式并注册为缓冲区
            self.register_buffer('bias', dense_module.bias.to_mkldnn())
        else:
            # 如果不存在偏置，创建一个全零的 Tensor 并转换为 MKL-DNN 格式后注册为缓冲区
            # TODO: 一旦 ScriptModule 支持注册 None 缓冲区，应该删除此部分
            self.register_buffer(
                'bias',
                torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())

    @torch.jit.script_method
    def __getstate__(self):
        # 返回模型状态的方法，包括权重、偏置和训练状态
        return (self.weight.to_dense(), self.bias.to_dense(), self.training)

    @torch.jit.script_method
    def forward(self, x):
        # 前向传播方法，执行 MKL-DNN 卷积操作
        return torch.mkldnn_convolution(
            x,
            self.weight,
            self.bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups)


class MkldnnConv1d(_MkldnnConvNd):
    def __init__(self, dense_module, dtype):
        super().__init__(dense_module)

        # 将输入的 dense_module 的权重转换为 MKL-DNN 格式并注册为缓冲区
        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))

    @torch.jit.script_method
    def __setstate__(self, state):
        # 设置模型状态的方法，将传入的状态恢复为权重、偏置和训练状态
        self.weight = state[0].to_mkldnn()
        self.bias = state[1].to_mkldnn()
        self.training = state[2]
class MkldnnConv2d(_MkldnnConvNd):
    def __init__(self, dense_module, dtype):
        super().__init__(dense_module)

        # 将dense_module的权重转换为MKLDNN格式，并重新排序以适应MKLDNN卷积的需要
        self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv2d_weight(
            dense_module.weight.to_mkldnn(dtype),
            self.padding,
            self.stride,
            self.dilation,
            self.groups))

    @torch.jit.script_method
    def __setstate__(self, state):
        # 根据状态state重新设置权重、偏置和训练状态，转换为MKLDNN格式
        self.weight = torch._C._nn.mkldnn_reorder_conv2d_weight(
            state[0].to_mkldnn(),
            self.padding,
            self.stride,
            self.dilation,
            self.groups)
        self.bias = state[1].to_mkldnn()
        self.training = state[2]

class MkldnnConv3d(_MkldnnConvNd):
    def __init__(self, dense_module, dtype):
        super().__init__(dense_module)

        # 将dense_module的权重转换为MKLDNN格式，并重新排序以适应MKLDNN 3D卷积的需要
        self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv3d_weight(
            dense_module.weight.to_mkldnn(dtype),
            self.padding,
            self.stride,
            self.dilation,
            self.groups))

    @torch.jit.script_method
    def __setstate__(self, state):
        # 根据状态state重新设置权重、偏置和训练状态，转换为MKLDNN格式
        self.weight = torch._C._nn.mkldnn_reorder_conv3d_weight(
            state[0].to_mkldnn(),
            self.padding,
            self.stride,
            self.dilation,
            self.groups)
        self.bias = state[1].to_mkldnn()
        self.training = state[2]

class MkldnnBatchNorm(torch.jit.ScriptModule):
    __constants__ = ['exponential_average_factor', 'eps']

    def __init__(self, dense_module):
        super().__init__()

        # 断言dense_module不处于训练状态，并且追踪运行统计信息并进行仿射变换
        assert not dense_module.training
        assert dense_module.track_running_stats
        assert dense_module.affine

        # 如果dense_module的动量参数为None，则指定指数平均因子为0.0
        if dense_module.momentum is None:
            self.exponential_average_factor = 0.0
        else:
            self.exponential_average_factor = dense_module.momentum
        self.eps = dense_module.eps

        # 将dense_module的权重、偏置、运行均值和运行方差转换为MKLDNN格式并注册为缓冲区
        self.register_buffer('weight', dense_module.weight.to_mkldnn())
        self.register_buffer('bias', dense_module.bias.to_mkldnn())
        self.register_buffer('running_mean', dense_module.running_mean.to_mkldnn())
        self.register_buffer('running_var', dense_module.running_var.to_mkldnn())

    @torch.jit.script_method
    def __getstate__(self):
        # 将当前状态以包含密集数据的形式返回，用于序列化
        weight = self.weight.to_dense()
        bias = self.bias.to_dense()
        running_mean = self.running_mean.to_dense()
        running_var = self.running_var.to_dense()
        return (weight, bias, running_mean, running_var, self.training)

    @torch.jit.script_method
    def __setstate__(self, state):
        # 根据状态state重新设置权重、偏置、运行均值、运行方差和训练状态，转换为MKLDNN格式
        self.weight = state[0].to_mkldnn()
        self.bias = state[1].to_mkldnn()
        self.running_mean = state[2].to_mkldnn()
        self.running_var = state[3].to_mkldnn()
        self.training = state[4]
    # 定义一个前向传播方法，接受输入张量 x
    def forward(self, x):
        # 调用 PyTorch 的批归一化函数 torch.batch_norm
        # 输入参数依次为：输入张量 x，权重 self.weight，偏置 self.bias，
        # 运行时均值 self.running_mean，运行时方差 self.running_var，
        # training 参数设为 False 表示不处于训练模式，
        # 指数加权平均因子 self.exponential_average_factor，
        # eps 参数用于数值稳定性，cuda_enabled 设为 False 表示不使用 CUDA 加速
        return torch.batch_norm(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            False,  # training
            self.exponential_average_factor,
            self.eps,
            False,  # cuda_enabled
        )
# 定义一个自定义的 Torch Script 模块 MkldnnPrelu，继承自 torch.jit.ScriptModule
class MkldnnPrelu(torch.jit.ScriptModule):
    # 初始化方法，接受一个 dense_module 和一个 dtype 参数
    def __init__(self, dense_module, dtype):
        # 调用父类的初始化方法
        super().__init__()
        # 将 dense_module 的权重转换成 MKLDNN 张量并注册为 buffer 'weight'
        self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))

    # Torch Script 方法，用于序列化对象状态
    @torch.jit.script_method
    def __getstate__(self):
        # 返回当前对象的状态，包括权重转换为 dense tensor 和训练状态
        return (self.weight.to_dense(), self.training)

    # Torch Script 方法，用于反序列化对象状态
    @torch.jit.script_method
    def __setstate__(self, state):
        # 根据传入的状态恢复对象的权重为 MKLDNN 张量和训练状态
        self.weight = state[0].to_mkldnn()
        self.training = state[1]

    # Torch Script 方法，前向传播函数
    @torch.jit.script_method
    def forward(self, x):
        # 如果输入 x 不是 MKLDNN 张量，则转换为 MKLDNN 张量
        x_mkldnn = x if x.is_mkldnn else x.to_mkldnn()
        # 使用 MKLDNN 加速的 PReLU 操作，得到输出 y_mkldnn
        y_mkldnn = torch.prelu(x_mkldnn, self.weight)
        # 如果输入 x 不是 MKLDNN 张量，则将输出 y_mkldnn 转换为 dense tensor
        y = y_mkldnn if x.is_mkldnn else y_mkldnn.to_dense()
        # 返回前向传播的结果 y
        return y

# 将输入模块 module 转换为 MKLDNN 加速模块的函数
def to_mkldnn(module, dtype=torch.float):
    # 断言 dtype 必须是 torch.float, torch.bfloat16, torch.half 中的一种
    assert dtype in [torch.float, torch.bfloat16, torch.half], \
        "MKLDNN only support float, bfloat16, and half path now"

    # 定义转换模块的内部函数 m_fn
    def m_fn(m, d):
        # 根据不同的模块类型返回相应的 MKLDNN 加速模块
        if isinstance(m, torch.nn.Linear):
            return MkldnnLinear(m, d)
        elif isinstance(m, torch.nn.Conv1d):
            return MkldnnConv1d(m, d)
        elif isinstance(m, torch.nn.Conv2d):
            return MkldnnConv2d(m, d)
        elif isinstance(m, torch.nn.Conv3d):
            return MkldnnConv3d(m, d)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            # 对于 BatchNorm，使用 MKLDNN 加速的 BatchNorm 模块
            return MkldnnBatchNorm(m)
        elif isinstance(m, torch.nn.PReLU):
            # 对于 PReLU，使用自定义的 MKLDNN PReLU 模块
            return MkldnnPrelu(m, d)
        else:
            # 其他类型的模块保持不变
            return m

    # 递归地对 module 及其子模块应用 m_fn 转换函数的内部递归函数
    def m_fn_rec(m, d):
        # 对当前模块 m 应用 m_fn 转换为 MKLDNN 加速模块
        new_m = m_fn(m, d)
        # 递归地对 m 的每个子模块 sub_m 应用 m_fn_rec 函数
        for name, sub_m in m.named_children():
            setattr(new_m, name, m_fn_rec(sub_m, d))
        # 返回转换后的新模块 new_m
        return new_m

    # 对输入模块 module 及其子模块应用 m_fn_rec 函数进行转换，并返回结果
    return m_fn_rec(module, dtype)
```