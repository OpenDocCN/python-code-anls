# `.\pytorch\benchmarks\tensorexpr\rnn_eltwise.py`

```py
# 导入 PyTorch 库
import torch

# 从当前目录导入 benchmark 模块
from . import benchmark

# 定义 RNNEltwise 类，继承自 benchmark.Benchmark 类
class RNNEltwise(benchmark.Benchmark):
    # 初始化方法，接受 mode, device, dtype, b, hs 作为参数
    def __init__(self, mode, device, dtype, b, hs):
        # 调用父类 benchmark.Benchmark 的初始化方法
        super().__init__(mode, device, dtype)
        # 设置对象的属性：batch size
        self.b = b
        # 设置对象的属性：hidden size
        self.hs = hs
        # 使用 self.rand 方法生成随机张量作为输入，并赋给 self.input 属性
        self.input = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        # 使用 self.rand 方法生成随机张量作为初始隐藏状态，并赋给 self.hx 属性
        self.hx = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        # 使用 self.rand 方法生成随机张量作为初始细胞状态，并赋给 self.cx 属性
        self.cx = self.rand(
            [b, hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        # 使用 self.rand 方法生成随机张量作为输入到隐藏状态的权重，并赋给 self.b_ih 属性
        self.b_ih = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        # 使用 self.rand 方法生成随机张量作为隐藏状态到隐藏状态的权重，并赋给 self.b_hh 属性
        self.b_hh = self.rand(
            [b, 4 * hs], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        # 将生成的张量作为列表赋给 self.inputs 属性，用于存储所有输入张量
        self.inputs = [
            self.input,
            self.hx,
            self.cx,
            self.b_ih,
            self.b_hh,
        ]

    # 前向传播方法，接受 input, hx, cx, b_ih, b_hh 作为输入
    def forward(self, input, hx, cx, b_ih, b_hh):
        # 将输入张量和权重张量相加，形成门控信号张量 gates
        gates = input + hx + b_ih + b_hh

        # 使用 chunk 方法将 gates 张量分割成四个部分，分别代表输入门、遗忘门、细胞状态和输出门
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # 对每个门控信号进行激活函数处理
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        # 根据 LSTM 公式计算新的细胞状态 cy 和隐藏状态 hy
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        # 返回隐藏状态和细胞状态
        return hy, cy

    # 返回配置信息，包括 batch size 和 hidden size
    def config(self):
        return [self.b, self.hs]

    # 静态方法，返回模块名称 "rnn_eltwise"
    @staticmethod
    def module():
        return "rnn_eltwise"

    # 计算内存工作量的方法
    def memory_workload(self):
        # 定义计算张量内存大小的辅助函数 memsize
        def memsize(t):
            return t.numel() * t.element_size()

        # 计算输入张量总内存大小
        input_size = sum(memsize(t) for t in self.inputs)
        # 计算输出张量总内存大小
        output_size = 2 * memsize(self.cx)
        # 计算总的 I/O 操作内存大小
        io_size = input_size + output_size
        # 返回内存工作量字典，包括顺序读写 (sol) 和算法计算 (algorithmic) 的内存消耗
        return {"sol": io_size, "algorithmic": io_size}

    # 静态方法，返回默认配置参数列表
    @staticmethod
    def default_configs():
        return [[64, 512]]

# 将 RNNEltwise 类注册到 benchmark 模块中
benchmark.register_benchmark_class(RNNEltwise)

# 定义 DynamicLSTM 类，继承自 benchmark.DynamicShape 和 RNNEltwise 类
class DynamicLSTM(benchmark.DynamicShape, RNNEltwise):
    # 初始化方法，接受 mode, device, dtype, b, hs 作为参数
    def __init__(self, mode, device, dtype, b, hs):
        # 调用 benchmark.DynamicShape 的初始化方法
        benchmark.DynamicShape.__init__(self)
        # 调用 RNNEltwise 的初始化方法
        RNNEltwise.__init__(self, mode, device, dtype, b, hs)

# DynamicLSTM 类定义结束
    # 定义一个方法，用于初始化输入数据
    def instantiate_input(self):
        # 调用随机形状生成器，生成b和hs的随机形状
        b, hs = self.rand_shape([self.b, self.hs])

        # 初始化self.input，随机生成形状为[b, 4 * hs]的张量，设备为self.device，数据类型为self.dtype，是否需要梯度取决于self.requires_grad
        self.input = self.rand(
            [b, 4 * hs],
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )

        # 初始化self.hx，随机生成形状为[b, 4 * hs]的张量，设备、数据类型、梯度需求与self.input相同
        self.hx = self.rand(
            [b, 4 * hs],
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )

        # 初始化self.cx，随机生成形状为[b, hs]的张量，设备、数据类型、梯度需求与self.input相同
        self.cx = self.rand(
            [b, hs],
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )

        # 初始化self.b_ih，随机生成形状为[b, 4 * hs]的张量，设备、数据类型、梯度需求与self.input相同
        self.b_ih = self.rand(
            [b, 4 * hs],
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )

        # 初始化self.b_hh，随机生成形状为[b, 4 * hs]的张量，设备、数据类型、梯度需求与self.input相同
        self.b_hh = self.rand(
            [b, 4 * hs],
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )

        # 将所有初始化后的张量放入self.inputs列表中
        self.inputs = [
            self.input,
            self.hx,
            self.cx,
            self.b_ih,
            self.b_hh,
        ]

    @staticmethod
    # 返回字符串"dynamic_lstm"，表示这是一个静态方法，用于标识这个类是一个动态LSTM模块
    def module():
        return "dynamic_lstm"
# 注册 DynamicLSTM 类到 benchmark 中，使其可以被 benchmark 模块调用
benchmark.register_benchmark_class(DynamicLSTM)
```