# `.\pytorch\test\inductor\test_decompose_mem_bound_mm.py`

```py
# Owner(s): ["module: inductor"]

# 引入日志模块
import logging
# 引入单元测试模块
import unittest

# 引入PyTorch相关模块
import torch
import torch._inductor
# 引入计数器工具类
from torch._dynamo.utils import counters
# 引入测试用例类和运行测试的函数
from torch._inductor.test_case import run_tests, TestCase
# 引入运行和获取代码的工具函数
from torch._inductor.utils import run_and_get_code
# 引入文件检查工具
from torch.testing import FileCheck

# 引入内部共用测试工具类和函数
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
# 引入自定义的Inductor工具类，检查是否支持CUDA
from torch.testing._internal.inductor_utils import HAS_CUDA

# 根据是否有CUDA支持跳过测试
requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")


# 定义一个简单的神经网络模块，包含一个线性层
class MyModule(torch.nn.Module):
    def __init__(
        self, n_input: int, n_output: int, has_bias: bool, device="cuda"
    ) -> None:
        super().__init__()
        # 使用输入的参数初始化一个线性层
        self.linear = torch.nn.Linear(n_input, n_output, bias=has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 执行前向传播，返回线性层的输出
        return self.linear(x)


# 定义一个执行矩阵乘法的模块
class MyModule2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        # 执行矩阵乘法操作
        output = torch.bmm(input1, input2)
        return output


# 定义一个执行矩阵乘法的模块
class MyModule3(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        # 执行矩阵乘法操作
        output = torch.mm(input1, input2)
        return output


# 在具有CUDA支持的环境下执行测试，同时应用特定的配置补丁
@requires_cuda
@torch._inductor.config.patch(
    post_grad_fusion_options={
        "decompose_mm_pass": {},
    }
)
# 实例化参数化测试
@instantiate_parametrized_tests
class TestDecomposeMemMM(TestCase):
    # 比较两个字典形式的张量，检查它们是否接近
    def compare_dict_tensors(self, ref_dict, res_dict, rtol=1e-3, atol=1e-3):
        if len(set(ref_dict.keys())) != len(set(res_dict.keys())):
            return False
        for key1 in ref_dict.keys():
            key2 = "_orig_mod." + key1
            assert key2 in res_dict, f"{key1} does not exist in traced module"
            if not torch.allclose(ref_dict[key1], res_dict[key2], rtol=rtol, atol=atol):
                return False
        return True

    # 比较模块的预测输出是否一致
    def compare_pred(self, module, traced, input, rtol=1e-3, atol=1e-3):
        ref = module(*input)
        res = traced(*input)
        self.assertEqual(ref, res, rtol=rtol, atol=atol)

    # 比较模块的参数是否一致
    def compare_parameters(self, module, traced, rtol=1e-3, atol=1e-3):
        ref_params = dict(module.named_parameters())
        res_params = dict(traced.named_parameters())
        self.assertTrue(self.compare_dict_tensors(ref_params, res_params, rtol, atol))

    # 比较模块的梯度是否一致
    def compare_gradients(self, module, traced, rtol=1e-3, atol=1e-3):
        ref_grad = {key: param.grad for key, param in module.named_parameters()}
        res_grad = {key: param.grad for key, param in traced.named_parameters()}
        self.assertTrue(
            self.compare_dict_tensors(ref_grad, res_grad, rtol=rtol, atol=atol)
        )

    # 参数化测试用例，测试矩阵乘法的分解
    @parametrize(
        "b,m,k,n,should_decompose",
        [(10240, 2, 2, 2, True), (10240, 2, 32, 32, False), (2000, 2, 2, 2, False)],
    )
    def test_decompose_bmm(self, b, m, n, k, should_decompose):
        # 设置日志级别为 DEBUG
        torch._logging.set_logs(inductor=logging.DEBUG)
        
        # 创建随机张量 mat1 和 mat2，存储在 GPU 上，并设置需要梯度计算
        mat1 = torch.randn(b, m, k, device="cuda").requires_grad_(True)
        mat2 = torch.randn(b, k, n, device="cuda").requires_grad_(True)

        # 清空计数器
        counters.clear()

        # 在 GPU 上创建 MyModule2 实例
        module = MyModule2().to("cuda")
        # 编译追踪模块
        traced = torch.compile(module)
        # 准备输入数据列表
        input = [mat1, mat2]
        # 调用模块前向传播
        ref = module(*input)
        # 调用追踪后的模块前向传播
        res = traced(*input)

        # 比较预测结果
        self.compare_pred(module, traced, input)

        # 根据 should_decompose 的值设置期望的计数值
        expected_val = 1 if should_decompose else 0
        # 断言计数器中 decompose_bmm 的值与期望值相等
        self.assertEqual(
            counters["inductor"]["decompose_bmm"],
            expected_val,
        )

        # 对追踪前的模块和追踪后的模块进行反向传播
        ref.sum().backward()
        res.sum().backward()
        # 比较模块的参数
        self.compare_parameters(module, traced)
        # 比较模块的梯度
        self.compare_gradients(module, traced)

        # 根据 should_decompose 的值设置期望的计数值
        expected_val = 3 if should_decompose else 0
        # 断言计数器中 decompose_bmm 的值与期望值相等
        self.assertEqual(
            counters["inductor"]["decompose_bmm"],
            expected_val,
        )
        # 清空计数器
        counters.clear()

    @parametrize(
        "m,k,n, should_decompose",
        [(20480, 5, 2, True), (20480, 32, 2, False), (2048, 2, 2, False)],
    )
    @parametrize("has_bias", [True, False])
    def test_decompose_linear(self, m, n, k, has_bias, should_decompose):
        # 设置日志级别为 DEBUG
        torch._logging.set_logs(inductor=logging.DEBUG)
        
        # 创建随机张量 input，存储在 GPU 上，并设置需要梯度计算
        input = torch.randn(m, k, device="cuda").requires_grad_(True)

        # 清空计数器
        counters.clear()

        # 在 GPU 上创建 MyModule 实例
        module = MyModule(k, n, has_bias).to("cuda")
        # 编译追踪模块
        traced = torch.compile(module)
        # 准备输入数据列表
        input = [input]
        # 调用模块前向传播
        ref = module(*input)
        # 调用追踪后的模块前向传播
        res = traced(*input)

        # 比较预测结果
        self.compare_pred(module, traced, input)

        # 根据 should_decompose 和 has_bias 的值设置期望的计数值
        expected_val = 1 if should_decompose else 0
        if has_bias:
            # 断言计数器中 decompose_addmm 的值与期望值相等
            self.assertEqual(
                counters["inductor"]["decompose_addmm"],
                expected_val,
            )
        else:
            # 断言计数器中 decompose_mm 的值与期望值相等
            self.assertEqual(
                counters["inductor"]["decompose_mm"],
                expected_val,
            )
        # 保存 decompose_mm 的当前值
        decompose_mm_fwd = counters["inductor"]["decompose_mm"]

        # 对模块进行反向传播
        ref.sum().backward()
        res.sum().backward()

        # 比较模块的参数
        self.compare_parameters(module, traced)
        # 比较模块的梯度
        self.compare_gradients(module, traced)

        # 计算 decompose_mm 的值变化，并与期望的计数值进行断言
        self.assertEqual(
            counters["inductor"]["decompose_mm"] - decompose_mm_fwd,
            expected_val,
        )
        # 清空计数器
        counters.clear()
    # 定义一个测试函数，用于测试矩阵乘法分解的功能
    def test_decompose_mm(self, m, n, k, has_bias, should_decompose):
        # 设置日志级别为 DEBUG
        torch._logging.set_logs(inductor=logging.DEBUG)
        # 创建随机张量 mat1 和 mat2，存储在 CUDA 设备上，并且需要计算梯度
        mat1 = torch.randn(m, k, device="cuda").requires_grad_(True)
        mat2 = torch.randn(k, n, device="cuda").requires_grad_(True)

        # 清空计数器
        counters.clear()

        # 在 CUDA 设备上创建 MyModule3 的实例 module
        module = MyModule3().to("cuda")
        # 编译模块 module
        traced = torch.compile(module)
        input = [mat1, mat2]
        # 获取模块的输出结果 ref
        ref = module(*input)
        # 获取编译后模块的输出结果 res
        res = traced(*input)

        # 比较模块和编译后模块的预测结果
        self.compare_pred(module, traced, input)

        # 根据是否应该进行矩阵乘法分解，设置预期值 expected_val
        expected_val = 1 if should_decompose else 0
        # 断言计数器中矩阵乘法分解的调用次数
        self.assertEqual(
            counters["inductor"]["decompose_mm"],
            expected_val,
        )
        # 记录前向矩阵乘法分解的计数值
        decompose_mm_fwd = counters["inductor"]["decompose_mm"]

        # 计算 ref 的和，并对其进行反向传播
        ref.sum().backward()
        # 计算 res 的和，并对其进行反向传播
        res.sum().backward()
        # 比较模块和编译后模块的参数
        self.compare_parameters(module, traced)
        # 比较模块和编译后模块的梯度
        self.compare_gradients(module, traced)

        # 再次根据是否应该进行矩阵乘法分解，设置预期值 expected_val
        expected_val = 1 if should_decompose else 0
        # 断言计数器中矩阵乘法分解的调用次数与前向分解的差值
        self.assertEqual(
            counters["inductor"]["decompose_mm"] - decompose_mm_fwd,
            expected_val,
        )
        # 清空计数器
        counters.clear()

    @parametrize("m,k,n, should_decompose", [(20480, 5, 2, True)])
    @parametrize("has_bias", [True, False])
    # 定义一个测试函数，用于测试动态形状输入的功能
    def test_dynamic_shape(self, m, n, k, has_bias, should_decompose):
        # 设置日志级别为 DEBUG
        torch._logging.set_logs(inductor=logging.DEBUG)
        # 创建随机输入张量 input，存储在 CUDA 设备上，并且需要计算梯度
        input = torch.randn(m, k, device="cuda").requires_grad_(True)

        # 清空计数器
        counters.clear()

        # 在 CUDA 设备上创建 MyModule 的实例 module
        module = MyModule(k, n, has_bias).to("cuda")
        # 编译模块 module，支持动态形状输入
        traced = torch.compile(module, dynamic=True)
        input = [input]
        # 获取模块的输出结果 ref
        ref = module(*input)
        # 获取编译后模块的输出结果 res
        res = traced(*input)

        # 比较模块和编译后模块的预测结果
        self.compare_pred(module, traced, input)

        # 根据是否应该进行矩阵乘法分解，设置预期值 expected_val
        expected_val = 1 if should_decompose else 0
        # 如果存在偏置，则断言计数器中矩阵加法乘法分解的调用次数
        if has_bias:
            self.assertEqual(
                counters["inductor"]["decompose_addmm"],
                expected_val,
            )

        # 计算 ref 的和，并对其进行反向传播
        ref.sum().backward()
        # 计算 res 的和，并对其进行反向传播
        res.sum().backward()

        # 比较模块和编译后模块的参数
        self.compare_parameters(module, traced)
        # 比较模块和编译后模块的梯度
        self.compare_gradients(module, traced)

        # 断言计数器中矩阵乘法分解的调用次数
        self.assertEqual(
            counters["inductor"]["decompose_mm"],
            1 if has_bias else 2,
        )
        # 清空计数器
        counters.clear()

    # 定义一个测试函数，用于测试实现输入的功能
    def test_realize_input(self):
        # 定义张量的维度 m, k, n
        m = 20480
        k = 5
        n = 2
        # 设置日志级别为 DEBUG
        torch._logging.set_logs(inductor=logging.DEBUG)
        # 创建随机输入张量 input1 和 input2，存储在 CUDA 设备上
        input1 = torch.randn(m, k, device="cuda").T.contiguous()
        input2 = torch.randn(k, n, device="cuda")

        # 使用装饰器 @torch.compile() 编译函数 foo
        @torch.compile()
        def foo(x, y):
            # 返回输入张量 x 的转置乘以张量 y 的结果
            return x.T.contiguous() @ y

        # 运行函数 foo，获取输出结果 out 和生成的代码 code
        out, code = run_and_get_code(foo, input1, input2)

        # 断言代码中 ".run(" 的出现次数为 2 次
        FileCheck().check_count(".run(", 2, exactly=True).run(code[0])
# 如果当前脚本作为主程序执行（而非被导入到其他脚本中），则执行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```