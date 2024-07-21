# `.\pytorch\benchmarks\tensorexpr\broadcast.py`

```py
import itertools  # 导入 itertools 模块，用于高效的迭代工具
import operator   # 导入 operator 模块，提供了一些常见的运算符函数

import numpy as np   # 导入 NumPy 库，用于科学计算
import torch         # 导入 PyTorch 深度学习框架

from . import benchmark   # 从当前包中导入 benchmark 模块

class BroadcastMulBench(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, case, M, N, K):
        super().__init__(mode, device, dtype)
        self.case = case   # 设置案例类型
        self.M = M         # 设置 M 维度大小
        self.N = N         # 设置 N 维度大小
        self.K = K         # 设置 K 维度大小

        if case == "row":   # 如果案例是 "row"
            self.d1 = self.rand(   # 生成一个随机张量 d1，形状为 [M, N, 1]
                [M, N, 1], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
            self.d2 = self.rand(   # 生成一个随机张量 d2，形状为 [M, 1, K]
                [M, 1, K], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
        elif case == "mid":   # 如果案例是 "mid"
            self.d1 = self.rand(   # 生成一个随机张量 d1，形状为 [M, N, 1]
                [M, N, 1], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
            self.d2 = self.rand(   # 生成一个随机张量 d2，形状为 [1, N, K]
                [1, N, K], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
        elif case == "col":   # 如果案例是 "col"
            self.d1 = self.rand(   # 生成一个随机张量 d1，形状为 [M, 1, K]
                [M, 1, K], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
            self.d2 = self.rand(   # 生成一个随机张量 d2，形状为 [1, N, K]
                [1, N, K], device=device, dtype=dtype, requires_grad=self.requires_grad
            )
        else:
            raise ValueError(f"invalid case: {case}")   # 如果案例不是 "row", "mid", "col" 中的任意一个，抛出值错误异常

        self.inputs = [self.d1, self.d2]   # 将生成的张量 d1 和 d2 放入 inputs 列表中

    def forward(self, d1, d2):
        y = d1 + d2   # 计算输入张量 d1 和 d2 的加法
        return y   # 返回结果张量 y

    def reference(self):
        return self.numpy(self.d1) + self.numpy(self.d2)   # 返回参考结果，使用 NumPy 计算 d1 和 d2 的加法

    def config(self):
        return [self.M, self.N, self.K]   # 返回配置信息，包括 M、N、K 的值

    @staticmethod
    def default_configs():
        return [[128, 256, 128]]   # 返回默认配置，固定为 [[128, 256, 128]]

    def memory_workload(self):
        if self.mode == "fwd":   # 如果模式是 "fwd"
            sol_count = 1   # 单解算的数量为 1
            algorithmic_count = 1   # 算法计算的数量为 1
        else:
            sol_count = (1) + (1)   # 解算的数量为 (1) + (1)
            algorithmic_count = 1 + (1 + 1)   # 算法计算的数量为 1 + (1 + 1)

        buffer_size = self.M * self.N * self.K   # 计算缓冲区大小
        return {
            "sol": buffer_size * sol_count,   # 返回解算内存消耗
            "algorithmic": buffer_size * algorithmic_count,   # 返回算法计算内存消耗
        }


class BroadcastRowBench(BroadcastMulBench):
    def __init__(self, mode, device, dtype, M, N, K):
        super().__init__(mode, device, dtype, "row", M, N, K)   # 调用父类构造函数，并设置案例为 "row"

    @staticmethod
    def module():
        return "broadcast_row"   # 返回模块名称 "broadcast_row"


class BroadcastMidBench(BroadcastMulBench):
    def __init__(self, mode, device, dtype, M, N, K):
        super().__init__(mode, device, dtype, "mid", M, N, K)   # 调用父类构造函数，并设置案例为 "mid"

    @staticmethod
    def module():
        return "broadcast_mid"   # 返回模块名称 "broadcast_mid"


class BroadcastColBench(BroadcastMulBench):
    def __init__(self, mode, device, dtype, M, N, K):
        super().__init__(mode, device, dtype, "col", M, N, K)   # 调用父类构造函数，并设置案例为 "col"

    @staticmethod
    def module():
        return "broadcast_col"   # 返回模块名称 "broadcast_col"


class BroadcastThreeArgs(benchmark.Benchmark):
    pass   # BroadcastThreeArgs 类继承自 benchmark.Benchmark 类，无额外功能
    # 初始化函数，用于设置模式、设备、数据类型、以及维度参数 M, N, K, L
    def __init__(self, mode, device, dtype, M, N, K, L):
        # 调用父类的初始化方法，设置模式、设备、数据类型
        super().__init__(mode, device, dtype)
        # 初始化对象的维度参数
        self.M = M
        self.N = N
        self.K = K
        self.L = L

        # 使用 rand 方法生成随机数据 self.d1，形状为 [M, N]，存储在指定设备上，可指定是否需要梯度
        self.d1 = self.rand(
            [M, N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        # 使用 rand 方法生成随机数据 self.d2，形状为 [K, M, 1]，存储在指定设备上，可指定是否需要梯度
        self.d2 = self.rand(
            [K, M, 1], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        # 使用 rand 方法生成随机数据 self.d3，形状为 [L, K, 1, 1]，存储在指定设备上，可指定是否需要梯度
        self.d3 = self.rand(
            [L, K, 1, 1], device=device, dtype=dtype, requires_grad=self.requires_grad
        )

        # 将生成的数据存储在 inputs 列表中，用于后续的前向计算
        self.inputs = [self.d1, self.d2, self.d3]

    # 前向计算函数，接受三个输入参数 d1, d2, d3，返回它们的和
    def forward(self, d1, d2, d3):
        y = d1 + d2 + d3
        return y

    # 返回当前对象中存储数据的 numpy 格式的总和 self.d1 + self.d2 + self.d3
    def reference(self):
        return self.numpy(self.d1) + self.numpy(self.d2) + self.numpy(self.d3)

    # 返回模型的配置信息，包括维度参数 M, N, K, L
    def config(self):
        return [self.M, self.N, self.K, self.L]

    # 静态方法，返回默认的配置信息，包括固定的维度参数列表 [[32, 16, 64, 128]]
    @staticmethod
    def default_configs():
        return [[32, 16, 64, 128]]

    # 计算内存工作量的函数，根据模式选择不同的计算数量并返回内存使用情况的字典
    def memory_workload(self):
        if self.mode == "fwd":
            sol_count = 1
            algorithmic_count = 1
        else:
            sol_count = (1) + (1)
            algorithmic_count = 1 + (1 + 1 + 1)

        # 计算所需的缓冲区大小，单位为字节
        buffer_size = self.M * self.N * self.K * self.L * 4
        # 返回内存工作量的字典，包括解算和算法计算部分的内存使用量
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    # 静态方法，返回模块名称字符串 "broadcast_3args"
    @staticmethod
    def module():
        return "broadcast_3args"
# 注册 BroadcastRowBench 类到 benchmark 中
# benchmark.register_benchmark_class(BroadcastRowBench)
# 注册 BroadcastMidBench 类到 benchmark 中
# benchmark.register_benchmark_class(BroadcastMidBench)
# 注册 BroadcastColBench 类到 benchmark 中
# benchmark.register_benchmark_class(BroadcastColBench)
# 注册 BroadcastThreeArgs 类到 benchmark 中
# benchmark.register_benchmark_class(BroadcastThreeArgs)

# TODO: 将此部分与 elementwise bench 合并
# 用于元素级操作的模板类。
# 派生类将重写类实例以自定义其行为。
class BroadcastBench(benchmark.Benchmark):
    # 自定义类变量列表。
    op_str = None  # 操作字符串
    binary_op_pt_func = None  # PyTorch 二元操作函数
    binary_op_np_func = None  # NumPy 二元操作函数
    unary_op_pt_func = None  # PyTorch 一元操作函数
    unary_op_np_func = None  # NumPy 一元操作函数
    split_input = True  # 是否分割输入数据

    def __init__(self, mode, device, dtype, M, N, K):
        super().__init__(mode, device, dtype)
        self.M = M  # 维度 M
        self.N = N  # 维度 N
        self.K = K  # 维度 K
        # 随机生成数据 d1, d2, d3, d4，并设置是否需要梯度
        self.d1 = self.rand(
            [M, N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.d2 = self.rand(
            [K, 1, N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.d3 = self.rand(
            [M, N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.d4 = self.rand(
            [K, M, 1], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.inputs = [self.d1, self.d2, self.d3, self.d4]  # 输入数据列表

    def _eval(self, d1, d2, d3, d4, binary_op, unary_op):
        if not binary_op:
            # 如果未定义二元操作函数，则定义为加法操作
            def binary_op(x, y):
                return x + y

        if not unary_op:
            # 如果未定义一元操作函数，则定义为恒等操作
            def unary_op(x):
                return x

        if self.split_input:
            # 如果需要分割输入数据，则对每个数据进行一元操作
            d1 = unary_op(d1)
            d2 = unary_op(d2)
            d3 = unary_op(d3)
            d4 = unary_op(d4)
        else:
            # 否则，对输入数据执行一元操作，并进行加法操作
            d1, d2, d3, d4 = (
                unary_op(d1),
                unary_op(d2),
                unary_op(d1 + 0.001),
                unary_op(d4),
            )
        a = binary_op(d1, d2)  # 对 d1 和 d2 执行二元操作
        b = binary_op(d3, d4)  # 对 d3 和 d4 执行二元操作
        c = a + b  # 对 a 和 b 执行加法操作
        return c  # 返回结果 c

    def forward(self, d1, d2, d3, d4):
        # 调用 _eval 方法进行前向计算
        binary_op = self.__class__.binary_op_pt_func
        unary_op = self.__class__.unary_op_pt_func
        return self._eval(d1, d2, d3, d4, binary_op, unary_op)

    def reference(self):
        # 返回参考结果，使用 NumPy 函数进行计算
        binary_op = self.__class__.binary_op_np_func
        unary_op = self.__class__.unary_op_np_func
        [d1, d2, d3, d4] = [self.numpy(d) for d in [self.d1, self.d2, self.d3, self.d4]]
        return self._eval(d1, d2, d3, d4, binary_op, unary_op)

    def config(self):
        # 返回配置信息 [M, N, K]
        return [self.M, self.N, self.K]

    @classmethod
    def module(cls):
        # 返回模块名称 "broadcast_" + cls.op_str
        return "broadcast_" + cls.op_str
    # 定义一个实例方法 `memory_workload`，用于计算内存工作负载
    def memory_workload(self):
        # 计算输入的数量
        input_count = len(self.inputs)
        
        # 根据模式选择不同的计算方式
        if self.mode == "fwd":
            # 如果模式为前向传播并且分割输入
            if self.split_input:
                sol_count = 1  # 解决方案数量为1
                algorithmic_count = 1  # 算法数量为1
            else:
                sol_count = 1  # 解决方案数量为1
                algorithmic_count = 1  # 算法数量为1
        else:
            # 如果模式不是前向传播
            if self.split_input:
                sol_count = 1  # 解决方案数量为1
                algorithmic_count = input_count  # 算法数量为输入数量
            else:
                sol_count = 1  # 解决方案数量为1
                algorithmic_count = input_count  # 算法数量为输入数量
        
        # 计算缓冲区大小，单位为字节
        buffer_size = self.M * self.N * self.K * 4
        
        # 返回一个字典，包含解决方案和算法内存负载的计算结果
        return {
            "sol": buffer_size * sol_count,  # 解决方案内存负载
            "algorithmic": buffer_size * algorithmic_count,  # 算法内存负载
        }

    @staticmethod
    # 定义一个静态方法 `default_configs`，返回一个包含默认配置信息的列表
    def default_configs():
        return [[1 << 8, 1 << 7, 1 << 9]]
def register_broadcast_ops():
    # 定义二元操作列表
    binary_op_list = [
        ["mul", operator.mul],  # 乘法操作
        ["add", operator.add],  # 加法操作
        ["sub", operator.sub],  # 减法操作
        ["div", lambda a, b: a / (b + 1e-4)],  # 除法操作，处理除零情况
        [
            "pow",
            torch.pow,
            np.power,
        ],  # 指数操作，包括 torch 和 numpy 的实现
        ["max", torch.max, np.maximum],  # 最大值操作，包括 torch 和 numpy 的实现
        ["min", torch.min, np.minimum],  # 最小值操作，包括 torch 和 numpy 的实现
    ]

    # 定义一元操作列表
    unary_op_list = [
        ["erf", torch.erf, np.erf],  # 误差函数操作，包括 torch 和 numpy 的实现
        ["exp", torch.exp, np.exp],  # 指数函数操作，包括 torch 和 numpy 的实现
        ["sin", torch.sin, np.sin],  # 正弦函数操作，包括 torch 和 numpy 的实现
        ["cos", torch.cos, np.cos],  # 余弦函数操作，包括 torch 和 numpy 的实现
    ]

    # 遍历所有的二元操作和它们的输入方式（分割或共享）
    for split_input, binary_op in itertools.product([True, False], binary_op_list):
        # 复制 BroadcastBench 类
        if len(binary_op) == 2:
            [op_str, op_pt_func] = binary_op
            op_np_func = op_pt_func
        elif len(binary_op) == 3:
            [op_str, op_pt_func, op_np_func] = binary_op
        split_str = "split" if split_input else "shared"
        op_str = split_str + "_" + op_str
        bm_cls = type("BroadcastBench_" + op_str, (BroadcastBench,), {})
        bm_cls.op_str = op_str
        bm_cls.binary_op_pt_func = op_pt_func
        bm_cls.binary_op_np_func = op_np_func
        bm_cls.split_input = split_input
        benchmark.register_benchmark_class(bm_cls)

    # 遍历所有的一元操作和它们的输入方式（分割或共享）
    for split_input, unary_op in itertools.product([True, False], unary_op_list):
        # 复制 BroadcastBench 类
        if len(unary_op) == 2:
            [op_str, op_pt_func] = unary_op
            op_np_func = op_pt_func
        elif len(unary_op) == 3:
            [op_str, op_pt_func, op_np_func] = unary_op
        split_str = "split" if split_input else "shared"
        op_str = split_str + "_" + op_str
        bm_cls = type("BroadcastBench_" + op_str, (BroadcastBench,), {})
        bm_cls.op_str = op_str
        bm_cls.unary_op_pt_func = op_pt_func
        bm_cls.unary_op_np_func = op_np_func
        bm_cls.split_input = split_input
        benchmark.register_benchmark_class(bm_cls)


register_broadcast_ops()
```