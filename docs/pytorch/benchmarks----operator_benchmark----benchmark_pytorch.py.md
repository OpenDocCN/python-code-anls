# `.\pytorch\benchmarks\operator_benchmark\benchmark_pytorch.py`

```
import json  # 导入json模块，用于处理JSON格式数据
import time  # 导入time模块，用于时间相关操作

import benchmark_cpp_extension  # noqa: F401 导入benchmark_cpp_extension模块，用于C++扩展的性能基准测试

import torch  # 导入PyTorch库


"""PyTorch performance microbenchmarks.

This module contains PyTorch-specific functionalities for performance
microbenchmarks.
"""


class TorchBenchmarkBase(torch.nn.Module):
    """This is a base class used to create Pytorch operator benchmark.
    module_name is the name of the operator being benchmarked.
    test_name is the name (it's created by concatenating all the
    inputs) of a specific test
    """

    def __init__(self):
        super().__init__()
        self.user_given_name = None  # 初始化用户指定的名称为None
        self._pass_count = 0  # 初始化_pass_count为0，用于计数
        self._num_inputs_require_grads = 0  # 初始化需要梯度的输入数量为0

    def _set_backward_test(self, is_backward):
        self._is_backward = is_backward  # 设置是否为反向传播测试的标志

    def auto_set(self):
        """This is used to automatically set the require_grad for the backward patch.
        It is implemented based on two counters. One counter to save the number of
        times init has been called. The other counter to save the number of times
        this function itself has been called. In the very first time init is called,
        this function counts how many inputs require gradient. In each of the
        following init calls, this function will return only one true value.
        Here is an example:
            ...
            self.v1 = torch.rand(M, N, K, requires_grad=self.auto_set())
            self.v2 = torch.rand(M, N, K, requires_grad=self.auto_set())
            ...
        """
        if not self._is_backward:  # 如果不是反向传播测试，则返回False
            return False

        if self._pass_count == 0:  # 如果_pass_count为0，表示第一次调用
            self._num_inputs_require_grads += 1  # 统计需要梯度的输入数量
            return True
        else:
            self._auto_set_counter += 1  # 计数器加一，表示函数自身调用的次数
            return self._pass_count == self._auto_set_counter

    def extract_inputs_tuple(self):
        self.inputs_tuple = tuple(self.inputs.values())  # 提取输入值，转换为元组形式

    @torch.jit.export
    def get_inputs(self):
        # Need to convert the inputs to tuple outside of JIT so that
        # JIT can infer the size of the inputs.
        return self.inputs_tuple  # 返回输入值的元组形式，供JIT推断输入大小使用

    @torch.jit.export
    def forward_impl(self):
        # This is to supply the inputs to the forward function which
        # will be called in both the eager and JIT mode of local runs
        return self.forward(*self.get_inputs())  # 调用forward函数并传入输入值

    @torch.jit.export
    def forward_consume(self, iters: int):
        #  _consume is used to avoid the dead-code-elimination optimization
        for _ in range(iters):
            torch.ops.operator_benchmark._consume(self.forward_impl())  # 调用forward_impl函数以避免优化

    def module_name(self):
        """this is used to label the operator being benchmarked"""
        if self.user_given_name:  # 如果有用户给定的名称，则返回该名称
            return self.user_given_name
        return self.__class__.__name__  # 否则返回类名作为模块名称

    def set_module_name(self, name):
        self.user_given_name = name  # 设置用户指定的模块名称
    # 定义一个测试名称生成函数，接受任意关键字参数
    def test_name(self, **kargs):
        """this is a globally unique name which can be used to
        label a specific test
        """
        
        # 要跳过不包含在测试名称中的属性列表
        skip_key_list = ["device"]

        # 初始化测试名称字符串列表
        test_name_str = []
        
        # 遍历关键字参数中的每个键值对
        for key in kargs:
            # 获取当前键的值
            value = kargs[key]
            # 如果值是布尔型，转换为整数后转换为字符串，否则直接转换为字符串
            test_name_str.append(
                ("" if key in skip_key_list else key)
                + str(value if type(value) != bool else int(value))
            )
        
        # 根据模块名称和拼接的测试名称字符串，生成最终的测试名称
        name = (self.module_name() + "_" + "_".join(test_name_str)).replace(" ", "")
        
        # 返回生成的测试名称
        return name
# 这个类包含了用于基准测试操作符的所有信息
class PyTorchOperatorTestCase:
    """This class includes all the information needed to benchmark an operator.
    op_bench: it's a user-defined class (child of TorchBenchmarkBase)
    which includes input and operator, .etc
    test_config: a namedtuple includes test_name, input_shape, tag, run_backward.
    When run_backward is false, the run_forward method will be executed,
    When run_backward is true, run_forward_eager and _output_mean will be
    executed to generate output. Then, run_backward will be executed.
    """

    def __init__(self, op_bench, test_config):
        # 初始化函数，保存测试配置和操作符基准对象
        self.test_config = test_config
        self.op_bench = op_bench
        self.place_holder_tensor = torch.ones(1)  # 创建一个大小为1的张量，用作占位符
        self.framework = "PyTorch"  # 指定测试所用的框架为PyTorch
        self.time_series = []  # 初始化一个空列表，用于存储时间序列数据
        self._jit_forward_graph = None  # 初始化为None，用于存储JIT编译后的前向图

    def _generate_jit_forward_graph(self):
        """generate a graph for the forward function via scripting"""
        # 通过脚本化生成操作符基准的前向函数图形
        scripted_op_bench = torch.jit.script(self.op_bench)
        return scripted_op_bench.forward_consume

    def run_jit_forward(self, num_runs, print_per_iter=False, cuda_sync=False):
        """Run the forward path of an op with JIT mode"""
        # 使用JIT模式运行操作符的前向路径
        if self._jit_forward_graph is None:
            self._jit_forward_graph = self._generate_jit_forward_graph()
        self._jit_forward_graph(num_runs)

    def _print_per_iter(self):
        # 打印最后50个值
        length = min(len(self.time_series), 50)
        for i in range(length):
            print(
                "PyTorchObserver "
                + json.dumps(
                    {
                        "type": self.test_config.test_name,
                        "metric": "latency",
                        "unit": "ms",
                        "value": str(self.time_series[length - i - 1]),
                    }
                )
            )

    def run_forward(self, num_runs, print_per_iter, cuda_sync):
        """Run the forward path of an op with eager mode"""
        # 使用即时执行（eager）模式运行操作符的前向路径
        if print_per_iter:
            for _ in range(num_runs):
                start_time = time.time()
                self.output = self.op_bench.forward_impl()  # 执行操作符的前向实现方法
                if cuda_sync:
                    torch.cuda.synchronize(torch.cuda.current_device())  # 如果需要，同步CUDA设备
                end_time = time.time()
                self.time_series.append((end_time - start_time) * 1e3)  # 记录操作的执行时间
        else:
            for _ in range(num_runs):
                self.output = self.op_bench.forward_impl()  # 执行操作符的前向实现方法
            if cuda_sync:
                torch.cuda.synchronize(torch.cuda.current_device())  # 如果需要，同步CUDA设备
    def _output_mean(self):
        """TODO (mingzhe): it is not necessary to sum up everything by myself,
        torch.autograd.backward do take a gradient tensor. By default, it
        is the same shape as your output tensor, with all 1s.
        Mathematically, it is the same as if the output is summed together.
        So we should be able to get ride of this method.
        dummy function for gradient calculation
        """
        # 计算输出张量的均值
        self.mean = self.output.mean()

    def run_backward(self, num_runs, print_per_iter=False):
        """Run the backward path of an op in many iterations"""
        # TODO: can we use JIT here to reduce python overhead?
        # 循环执行反向传播多次
        for _ in range(num_runs):
            # 对均值进行反向传播，保留计算图以便多次反向传播
            self.mean.backward(retain_graph=True)
# 创建一个用于生成 PyTorch 操作测试用例的函数
def create_pytorch_op_test_case(op_bench, test_config):
    """This method is used to generate est. func_name is a global unique
    string. For PyTorch add operator with M=8, N=2, K=1, tag = long, here
    are the values for the members in test_case:
    op.module_name: add
    framework: PyTorch
    test_config: TestConfig(test_name='add_M8_N2_K1', input_config='M: 8, N: 2, K: 1',
        tag='long', run_backward=False)
    func_name: addPyTorchTestConfig(test_name='add_M8_N2_K1', input_config='M: 8, N: 2, K: 1',
                                    tag='long', run_backward=False)
    """
    # 创建一个 PyTorchOperatorTestCase 的实例，传入 op_bench 和 test_config 作为参数
    test_case = PyTorchOperatorTestCase(op_bench, test_config)
    # 从 test_case 中获取测试配置
    test_config = test_case.test_config
    # 从 test_case 中获取操作对象 op_bench
    op = test_case.op_bench
    # 生成一个全局唯一的函数名 func_name，格式为 "<操作模块名><框架名><测试配置的字符串表示>"
    func_name = f"{op.module_name()}{test_case.framework}{str(test_config)}"
    # 返回一个元组，包含 func_name 和 test_case 对象
    return (func_name, test_case)
```