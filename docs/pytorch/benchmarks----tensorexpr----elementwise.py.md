# `.\pytorch\benchmarks\tensorexpr\elementwise.py`

```
# 导入 itertools 库，用于高效迭代操作
# 导入 operator 库，提供操作符和函数，用于元素级操作
import itertools
import operator

# 导入 numpy 库，提供对多维数组的支持
import numpy as np
# 导入 scipy.special 库，提供特殊数学函数的支持
import scipy.special

# 导入 torch 库，用于构建和训练深度学习模型
import torch

# 从当前目录中导入 benchmark 模块
from . import benchmark


# 元素级操作的模板类
# 派生类将覆盖类实例以定制其行为
class ElementBench(benchmark.Benchmark):
    # 自定义类变量的列表
    op_str = None  # 操作字符串
    binary_op_pt_func = None  # PyTorch 二元操作函数
    binary_op_np_func = None  # NumPy 二元操作函数
    unary_op_pt_func = None   # PyTorch 一元操作函数
    unary_op_np_func = None   # NumPy 一元操作函数
    split_input = True         # 是否分割输入

    def __init__(self, mode, device, dtype, N):
        # 调用基类的构造函数初始化模式、设备和数据类型
        super().__init__(mode, device, dtype)
        self.N = N
        # 生成随机数填充数组 d1 - d4，并设置是否需要梯度
        self.d1 = self.rand(
            [N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.d2 = self.rand(
            [N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.d3 = self.rand(
            [N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        self.d4 = self.rand(
            [N], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        # 将生成的随机数组成列表输入
        self.inputs = [self.d1, self.d2, self.d3, self.d4]
        # 确定是否为非随机操作
        self.deterministic = "rand" not in self.op_str

    def _eval(self, d1, d2, d3, d4, binary_op, unary_op):
        # 如果未提供二元操作函数，则定义一个默认的二元操作函数
        if not binary_op:

            def binary_op(x, y):
                return x + y

        # 如果未提供一元操作函数，则定义一个默认的一元操作函数
        if not unary_op:

            def unary_op(x):
                return x

        # 如果设定了分割输入，则对输入数组应用一元操作函数
        if self.split_input:
            d1 = unary_op(d1)
            d2 = unary_op(d2)
            d3 = unary_op(d3)
            d4 = unary_op(d4)
        else:
            # 否则，对 d1 加上一个小量后应用一元操作函数
            d2 = unary_op(d1 + 0.001)
            d3 = unary_op(d1 + 0.002)
            d4 = unary_op(d1 + 0.003)
            d1 = unary_op(d1)
        
        # 使用二元操作函数计算 a 和 b
        a = binary_op(d1, d2)
        b = binary_op(d3, d4)
        # 计算 c，将 a 和 b 相加
        c = a + b
        return c

    def forward(self, d1, d2, d3, d4):
        # 获取当前类的 PyTorch 二元和一元操作函数，然后调用 _eval 方法
        binary_op = self.__class__.binary_op_pt_func
        unary_op = self.__class__.unary_op_pt_func
        return self._eval(d1, d2, d3, d4, binary_op, unary_op)

    def reference(self):
        # 获取当前类的 NumPy 二元和一元操作函数，然后调用 _eval 方法
        binary_op = self.__class__.binary_op_np_func
        unary_op = self.__class__.unary_op_np_func
        [d1, d2, d3, d4] = [self.numpy(d) for d in [self.d1, self.d2, self.d3, self.d4]]
        return self._eval(d1, d2, d3, d4, binary_op, unary_op)

    def config(self):
        # 返回类的 N 属性，作为配置的一部分
        return [self.N]

    @classmethod
    def module(cls):
        # 返回模块名称，以 element_ 开头，接着是 op_str
        return "element_" + cls.op_str
    # 定义一个方法用于计算内存工作量
    def memory_workload(self):
        # 计算输入的数量
        input_count = len(self.inputs)
        
        # 根据模式选择计算分支
        if self.mode == "fwd":
            # 如果是前向模式并且需要分割输入
            if self.split_input:
                # 计算解决方案所需内存量
                sol_count = input_count + 1
                # 计算算法所需内存量
                algorithmic_count = input_count + 1
            else:
                # 否则固定的内存量计算
                sol_count = 1 + 1
                algorithmic_count = 1 + 1
            # 如果操作字符串包含随机指令，则固定内存量为1
            if "rand" in self.op_str:
                sol_count = 1
                algorithmic_count = 1
        else:
            # 如果是反向模式并且需要分割输入
            if self.split_input:
                # 计算解决方案所需内存量，复杂计算
                sol_count = (input_count + 1) + (1 + input_count)
                # 计算算法所需内存量，复杂计算
                algorithmic_count = (input_count + 1) + ((2 + 1) * input_count)
            else:
                # 否则固定的内存量计算
                sol_count = 1 + 1
                algorithmic_count = 1 + 1
            # 如果操作字符串包含随机指令，则固定内存量为1
            if "rand" in self.op_str:
                sol_count = 1
                algorithmic_count = 1
        
        # 缓冲区大小设为 N
        buffer_size = self.N
        # 返回解决方案和算法所需内存量的乘积
        return {
            "sol": buffer_size * sol_count,
            "algorithmic": buffer_size * algorithmic_count,
        }

    # 静态方法：返回一个默认配置的列表，包含一个元素 1 << 25
    @staticmethod
    def default_configs():
        return [[1 << 25]]
def register_element_ops():
    # 定义二元操作列表，包括操作名和对应的函数
    binary_op_list = [
        ["mul", operator.mul],  # 乘法操作
        ["add", operator.add],  # 加法操作
        ["sub", operator.sub],  # 减法操作
        ["div", lambda a, b: a / (b + 1e-4)],  # 除法操作，防止除数为零
        [
            "pow",
            torch.pow,
            np.power,
        ],  # 指数操作，包括PyTorch和NumPy的实现
        ["max", torch.max, np.maximum],  # 最大值操作，包括PyTorch和NumPy的实现
        ["min", torch.min, np.minimum],  # 最小值操作，包括PyTorch和NumPy的实现
    ]

    # 定义一元操作列表，包括操作名和对应的函数
    unary_op_list = [
        ["erf", torch.erf, scipy.special.erf],  # 误差函数操作，包括PyTorch和SciPy的实现
        ["exp", torch.exp, np.exp],  # 指数函数操作，包括PyTorch和NumPy的实现
        ["sin", torch.sin, np.sin],  # 正弦函数操作，包括PyTorch和NumPy的实现
        ["cos", torch.cos, np.cos],  # 余弦函数操作，包括PyTorch和NumPy的实现
        ["rand_like", torch.rand_like, lambda x: np.random.rand(*x.shape)],  # 类似随机数操作，包括PyTorch和NumPy的实现
    ]

    # 遍历所有的分割输入和二元操作，创建并注册基准测试类
    for split_input, binary_op in itertools.product([True, False], binary_op_list):
        # 复制 ElementBench 创建新的基准测试类
        if len(binary_op) == 2:
            [op_str, op_pt_func] = binary_op
            op_np_func = op_pt_func
        elif len(binary_op) == 3:
            [op_str, op_pt_func, op_np_func] = binary_op
        split_str = "split" if split_input else "shared"
        op_str = split_str + "_" + op_str
        bm_cls = type("ElementBench_" + op_str, (ElementBench,), {})
        bm_cls.op_str = op_str
        bm_cls.binary_op_pt_func = op_pt_func
        bm_cls.binary_op_np_func = op_np_func
        bm_cls.split_input = split_input
        benchmark.register_benchmark_class(bm_cls)

    # 遍历所有的分割输入和一元操作，创建并注册基准测试类
    for split_input, unary_op in itertools.product([True, False], unary_op_list):
        # 复制 ElementBench 创建新的基准测试类
        if len(unary_op) == 2:
            [op_str, op_pt_func] = unary_op
            op_np_func = op_pt_func
        elif len(unary_op) == 3:
            [op_str, op_pt_func, op_np_func] = unary_op
        split_str = "split" if split_input else "shared"
        op_str = split_str + "_" + op_str
        bm_cls = type("ElementBench_" + op_str, (ElementBench,), {})
        bm_cls.op_str = op_str
        bm_cls.unary_op_pt_func = op_pt_func
        bm_cls.unary_op_np_func = op_np_func
        bm_cls.split_input = split_input
        benchmark.register_benchmark_class(bm_cls)
    # 定义一个静态方法 `module`，返回固定字符串 "simple_element"
    def module(cls):
        return "simple_element"

    # 定义一个实例方法 `memory_workload`，计算内存工作负载
    def memory_workload(self):
        # 计算输入的数量
        input_count = len(self.inputs)
        
        # 根据模式 `mode` 的不同设置解算器和算法的数量
        if self.mode == "fwd":
            sol_count = 2
            algorithmic_count = 2
        else:
            sol_count = 2
            algorithmic_count = 2
        
        # 设置缓冲区大小为 `N`
        buffer_size = self.N
        
        # 返回一个字典，包含解算器和算法内存占用的估算值
        return {
            "sol": buffer_size * sol_count,        # 解算器内存占用
            "algorithmic": buffer_size * algorithmic_count,   # 算法内存占用
        }

    # 定义一个静态方法 `default_configs`，返回一个包含单个列表 `[1 << 25]` 的列表
    @staticmethod
    def default_configs():
        return [[1 << 25]]
# 注册简单元素基准类到基准测试框架中
benchmark.register_benchmark_class(SimpleElementBench)

# 定义动态简单元素基准类，继承自benchmark.DynamicShape和SimpleElementBench类
class DynamicSimpleElementBench(benchmark.DynamicShape, SimpleElementBench):
    
    # 初始化方法，接受mode、device、dtype和N作为参数
    def __init__(self, mode, device, dtype, N):
        # 调用benchmark.DynamicShape的初始化方法
        benchmark.DynamicShape.__init__(self)
        # 调用SimpleElementBench的初始化方法
        SimpleElementBench.__init__(self, mode, device, dtype, N)

    # 类方法，返回字符串"simple_dynamic_element"，表示模块名称
    @classmethod
    def module(cls):
        return "simple_dynamic_element"

    # 实例化输入数据的方法
    def instantiate_input(self):
        # 生成长度为self.N的随机形状数组，并解包得到N
        (N,) = self.rand_shape([self.N])
        # 生成随机数据数组data，设定其设备、数据类型和是否需要梯度
        data = self.rand(
            [N], device=self.device, dtype=self.dtype, requires_grad=self.requires_grad
        )
        # 将生成的data数组作为输入存储在self.inputs中
        self.inputs = [data]

# 注册动态简单元素基准类到基准测试框架中
benchmark.register_benchmark_class(DynamicSimpleElementBench)
```