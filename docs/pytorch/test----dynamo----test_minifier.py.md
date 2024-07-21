# `.\pytorch\test\dynamo\test_minifier.py`

```
# Owner(s): ["module: dynamo"]
import unittest  # 导入单元测试模块

import torch._dynamo  # 导入torch._dynamo模块
from torch._dynamo.test_minifier_common import MinifierTestBase  # 从torch._dynamo.test_minifier_common模块导入MinifierTestBase类
from torch.testing._internal.common_utils import skipIfNNModuleInlined  # 从torch.testing._internal.common_utils模块导入skipIfNNModuleInlined函数

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")  # 条件装饰器，如果CUDA可用，则跳过测试


class MinifierTests(MinifierTestBase):
    # Test that compile, runtime, and accuracy errors after dynamo can be repro'd (both CPU and CUDA)
    def _test_after_dynamo(self, device, backend, expected_error):
        run_code = f"""\
@torch._dynamo.optimize({backend!r})
def inner(x):
    for _ in range(10):
        x = torch.sin(x)
    x = torch.relu(x)
    for _ in range(10):
        x = torch.cos(x)
    return x

inner(torch.randn(20, 20).to("{device}"))
"""
        self._run_full_test(run_code, "dynamo", expected_error, isolate=False)  # 运行完整的测试代码，验证预期的错误，不进行隔离

    def test_after_dynamo_cpu_compile_error(self):
        self._test_after_dynamo(
            "cpu", "relu_compile_error_TESTING_ONLY", "ReluCompileError"
        )

    def test_after_dynamo_cpu_runtime_error(self):
        self._test_after_dynamo(
            "cpu", "relu_runtime_error_TESTING_ONLY", "ReluRuntimeError"
        )

    def test_after_dynamo_cpu_accuracy_error(self):
        self._test_after_dynamo(
            "cpu", "relu_accuracy_error_TESTING_ONLY", "AccuracyError"
        )

    @requires_cuda
    def test_after_dynamo_cuda_compile_error(self):
        self._test_after_dynamo(
            "cuda", "relu_compile_error_TESTING_ONLY", "ReluCompileError"
        )

    @requires_cuda
    def test_after_dynamo_cuda_runtime_error(self):
        self._test_after_dynamo(
            "cuda", "relu_runtime_error_TESTING_ONLY", "ReluRuntimeError"
        )

    @requires_cuda
    def test_after_dynamo_cuda_accuracy_error(self):
        self._test_after_dynamo(
            "cuda", "relu_accuracy_error_TESTING_ONLY", "AccuracyError"
        )

    def test_after_dynamo_non_leaf_compile_error(self):
        run_code = """\
@torch._dynamo.optimize("non_leaf_compile_error_TESTING_ONLY")
def inner(x):
    return x + 1

inner(torch.randn(20, 20, requires_grad=True) + 1)
"""
        self._run_full_test(
            run_code, "dynamo", "TestingOnlyCompileError", isolate=False
        )  # 运行完整的测试代码，验证预期的编译错误，不进行隔离

    # Ensure that the testing backends pass when relu is not present.
    def _test_after_dynamo_backend_passes(self, device, backend):
        @torch._dynamo.optimize(backend)
        def inner(x):
            for _ in range(10):
                x = torch.sin(x)
            for _ in range(10):
                x = torch.cos(x)
            return x

        inner(torch.randn(20, 20).to(device))

    def test_after_dynamo_cpu_compile_backend_passes(self):
        self._test_after_dynamo_backend_passes("cpu", "relu_compile_error_TESTING_ONLY")

    def test_after_dynamo_cpu_runtime_backend_passes(self):
        self._test_after_dynamo_backend_passes("cpu", "relu_runtime_error_TESTING_ONLY")
    # 测试在 Dynamo 后 CPU 精度后端是否通过
    def test_after_dynamo_cpu_accuracy_backend_passes(self):
        self._test_after_dynamo_backend_passes(
            "cpu", "relu_accuracy_error_TESTING_ONLY"
        )

    # 使用 CUDA 的前提下测试在 Dynamo 后 CUDA 编译后端是否通过
    @requires_cuda
    def test_after_dynamo_cuda_compile_backend_passes(self):
        self._test_after_dynamo_backend_passes(
            "cuda", "relu_compile_error_TESTING_ONLY"
        )

    # 使用 CUDA 的前提下测试在 Dynamo 后 CUDA 运行时后端是否通过
    @requires_cuda
    def test_after_dynamo_cuda_runtime_backend_passes(self):
        self._test_after_dynamo_backend_passes(
            "cuda", "relu_runtime_error_TESTING_ONLY"
        )

    # 使用 CUDA 的前提下测试在 Dynamo 后 CUDA 精度后端是否通过
    @requires_cuda
    def test_after_dynamo_cuda_accuracy_backend_passes(self):
        self._test_after_dynamo_backend_passes(
            "cuda", "relu_accuracy_error_TESTING_ONLY"
        )

    # 测试一个模块，该模块包含混合的 CPU/CUDA 部分，并且在 Dynamo 之后出现错误能否重现
    @skipIfNNModuleInlined()
    @requires_cuda
    def test_cpu_cuda_module_after_dynamo(self):
        # 后端名称为 "relu_compile_error_TESTING_ONLY"
        backend_name = "relu_compile_error_TESTING_ONLY"
        # 定义运行的代码字符串
        run_code = f"""\
class CpuCudaModule(torch.nn.Module):
    # 定义一个继承自torch.nn.Module的模块类CpuCudaModule
    def __init__(self):
        super().__init__()
        # 初始化模块的成员变量
        self.m_x = torch.nn.Linear(20, 20).cuda()  # 在CUDA上创建一个线性层对象
        self.m_y = torch.nn.Linear(20, 20)  # 创建一个普通的线性层对象
        self.p_x = torch.nn.Parameter(torch.randn(20, 20).cuda())  # 在CUDA上创建一个模型参数对象
        self.p_y = torch.nn.Parameter(torch.randn(20, 20))  # 创建一个普通的模型参数对象
        self.register_buffer("b_x", torch.ones(20, 20).cuda())  # 在CUDA上注册一个缓冲区对象
        self.register_buffer("b_y", torch.ones(20, 20))  # 注册一个普通的缓冲区对象

    def forward(self, x, y):
        # 前向传播函数定义
        return self.m_x(x) + self.p_x + self.b_x, self.m_y(y) + self.p_y + self.b_y
        # 返回两个元组，分别是模型计算后的结果
        # 注意：这里的操作涉及模型参数、缓冲区和CUDA的使用

mod = CpuCudaModule()  # 创建一个CpuCudaModule实例对象

@torch._dynamo.optimize({backend_name!r})
# 使用torch._dynamo.optimize装饰器优化后的函数定义
def inner(x1, y1):
    # 内部函数inner的定义，接受两个输入参数x1和y1
    x2 = torch.randn(20, 20).cuda()  # 在CUDA上创建一个20x20的随机张量
    y2 = torch.randn(20, 20)  # 创建一个普通的20x20的随机张量
    x3, y3 = mod(x1 + x2, y1 + y2)  # 调用模块对象mod的forward方法，传入加和后的输入参数
    return torch.relu(x3.cpu() + y3)
    # 对模型输出进行ReLU激活并返回结果
    # 注意：涉及到CUDA和CPU之间的张量传输

inner(torch.randn(20, 20).cuda(), torch.randn(20, 20))
# 调用inner函数，传入两个20x20的CUDA张量作为参数
# 注意：这里进行了CUDA张量的创建和函数调用
    # 执行一个循环，循环20次
    for _ in range(20):
        # 使用 PyTorch 的余弦函数计算当前 x 的余弦值，并将结果赋给 x
        x = torch.cos(x)
    # 返回最后一次循环后的 x 值
    return x
"""

        res = self._run_full_test(run_code, "dynamo", "ReluCompileError", isolate=False)
        
        # 使用 self._run_full_test 方法执行完整测试，获取测试结果
        self.assertExpectedInline(
            res.repro_module(),
            """\
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_19):
        x_20 = torch.relu(x_19);  x_19 = None
        return (x_20,)""",
        )
        
        # 使用 self.assertExpectedInline 方法断言内联代码是否符合预期
        # 检查生成的模块，验证是否与预期的类和方法定义匹配
        

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    # 运行测试套件，确保模块的行为符合预期
    run_tests()
```