# `.\pytorch\test\inductor\test_compiled_autograd.py`

```py
# Owner(s): ["module: inductor"]
# 导入必要的库和模块
import functools  # 导入 functools 库
import io  # 导入 io 库
import logging  # 导入 logging 库
import re  # 导入 re 库
import sys  # 导入 sys 库
import unittest  # 导入 unittest 库
from importlib.machinery import SourceFileLoader  # 导入 SourceFileLoader 类
from pathlib import Path  # 导入 Path 类
from unittest import mock  # 导入 mock 模块

import torch  # 导入 torch 库
import torch.nn as nn  # 导入 torch.nn 模块
import torch.nn.functional as F  # 导入 torch.nn.functional 模块
from torch import _inductor as inductor  # 导入 _inductor 模块作为 inductor
from torch._dynamo import compiled_autograd, config  # 导入 compiled_autograd 和 config 模块
from torch._dynamo.utils import counters  # 导入 counters 工具
from torch._inductor import config as inductor_config  # 导入 inductor_config 模块
from torch._inductor.test_case import run_tests, TestCase  # 导入 run_tests 和 TestCase 类
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA  # 导入 HAS_CPU 和 HAS_CUDA 变量
from torch.testing._internal.logging_utils import logs_to_string  # 导入 logs_to_string 函数

# note: these tests are not run on windows due to inductor_utils.HAS_CPU

# 定义函数 make_compiler_fn，返回一个编译器函数
def make_compiler_fn(fullgraph=True, dynamic=True):
    def _compiler_fn(gm):
        """Same as torch.compile() but counts number of compiles"""
        # 内部编译器函数，增加编译计数并调用 inductor.compile 进行编译
        def _inner_compiler(gm_, example_inputs_):
            counters["compiled_autograd"]["compiles"] += 1
            return inductor.compile(gm_, example_inputs_)

        return torch.compile(
            gm, backend=_inner_compiler, fullgraph=fullgraph, dynamic=dynamic
        )

    return _compiler_fn


compiler_fn = make_compiler_fn()

# TODO(jansel): hooks as lambdas creates recompiles in dynamo, we should fix that

# 定义 hook1 函数，简单地将梯度乘以2
def hook1(grad):
    return grad * 2

# 定义 hook2 函数，对输入的梯度元组进行处理，返回第一个元素加1
def hook2(grads):
    return (grads[0] + 1,)

# 定义 hook3 函数，对输入的梯度元组进行处理，返回输入的正弦值加上输出的第一个元素
def hook3(gI, gO):
    return (torch.sin(gI[0]) + gO[0],)

# 定义测试类 TestCompiledAutograd，继承自 TestCase 类
class TestCompiledAutograd(TestCase):

    # 设置测试前的准备工作
    def setUp(self) -> None:
        super().setUp()
        torch._logging.set_logs(compiled_autograd_verbose=False)
        config.compiled_autograd = False
        compiled_autograd.reset()

    # 设置测试后的清理工作
    def tearDown(self) -> None:
        super().tearDown()
        torch._logging.set_logs(compiled_autograd_verbose=False)
        config.compiled_autograd = False
        compiled_autograd.reset()

    # 检查输出和重新编译的方法
    def check_output_and_recompiles(
        self, fn, count=1, compiler_fn=compiler_fn, compile_fn=False
    ):
        if isinstance(count, list):
            captures, compiles = count
        else:
            captures, compiles = count, count
        with torch.autograd.set_multithreading_enabled(False):
            torch._dynamo.reset()
            counters["compiled_autograd"].clear()
            torch.manual_seed(123)
            expected = list(fn())
            torch.manual_seed(123)
            with compiled_autograd.enable(compiler_fn):
                opt_fn = torch.compile(fn) if compile_fn else fn
                actual = list(opt_fn())
            self.assertEqual(expected, actual)
            self.assertEqual(counters["compiled_autograd"]["captures"], captures)
            self.assertEqual(counters["compiled_autograd"]["compiles"], compiles)

    # 测试动态自动微分的不稳定性导致段错误的问题
    def test_dynamo_flaky_segfault(self):
        import os
        import subprocess

        script = """
import torch

def main():
    def compiler_fn(gm):
        return torch.compile(gm, backend="eager")
    # 定义内部函数 `inner`
    def inner():
        # 创建大小为 1000x3000 的随机张量 `x`
        x = torch.randn(1000, 3000)
        # 创建大小为 1000x3000 的随机张量 `w`，并设置为需要梯度计算
        w = torch.randn(1000, 3000, requires_grad=True)
        
        # 定义模型函数 `model`，使用 `w` 进行线性计算
        def model(i):
            return torch.nn.functional.linear(i, w)
        
        # 使用模型进行计算，得到输出 `out`
        out = model(x)
        
        # 计算输出的和作为损失
        loss = out.sum()
        
        # 使用动态图框架中的编译自动求导功能，并执行反向传播
        with torch._dynamo.compiled_autograd.enable(compiler_fn):
            loss.backward()
        
        # 断言张量 `w` 的梯度不为 None
        assert(w.grad is not None)

    # 调用内部函数 `inner` 第一次
    inner()
    
    # 重置动态图框架的状态，以清除之前的编译自动求导设置
    torch._dynamo.reset()
    
    # 再次调用内部函数 `inner`，此时不应用编译自动求导功能
    inner()
    """
    # Run the main function three times to ensure reliability in catching
    # bad dynamo state resets.
    def main():
        for _ in range(3):
            try:
                # Execute a subprocess running a Python script with the given script content
                # Ensure subprocess's current working directory is set to the directory of this script
                subprocess.check_output(
                    [sys.executable, "-c", script],
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(os.path.realpath(__file__)),
                )
            except subprocess.CalledProcessError as e:
                if e.returncode < 0:
                    self.fail("Subprocess exited with a fatal signal")

    # Define a test case for basic functionality
    def test_basic(self):
        def fn():
            # Create a simple neural network model with two linear layers and ReLU activations
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            x = torch.randn([2, 4])  # Generate random input tensor
            result = model(x).sum()  # Forward pass and sum the output
            result.backward()  # Perform backward propagation
            yield model[0].weight.grad  # Yield gradient of the first linear layer's weights
            yield model[0].bias.grad  # Yield gradient of the first linear layer's bias
            yield model[2].weight.grad  # Yield gradient of the second linear layer's weights
            yield model[2].bias.grad  # Yield gradient of the second linear layer's bias

        self.check_output_and_recompiles(fn)  # Run the function and check output

    # Define a test case to check cache hit scenario
    def test_cache_hit(self):
        def fn():
            for _ in range(3):
                # Create a neural network model similar to test_basic
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([2, 4])  # Generate random input tensor
                result = model(x).sum()  # Forward pass and sum the output
                result.backward()  # Perform backward propagation
                yield model[0].weight.grad  # Yield gradient of the first linear layer's weights
                yield model[0].bias.grad  # Yield gradient of the first linear layer's bias
                yield model[2].weight.grad  # Yield gradient of the second linear layer's weights
                yield model[2].bias.grad  # Yield gradient of the second linear layer's bias

        self.check_output_and_recompiles(fn)  # Run the function and check output

    # Define a test case to test tensor gradient hooks
    def test_tensor_grad_hook1(self):
        def fn():
            for _ in range(3):
                # Create a neural network model with a single linear layer and ReLU activation
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([2, 4])  # Generate random input tensor

                # Register a hook function 'hook1' to the gradient computation of the first linear layer's weights
                model[0].weight.register_hook(hook1)

                result = model(x).sum()  # Forward pass and sum the output
                result.backward()  # Perform backward propagation
                yield model[0].weight.grad  # Yield gradient of the first linear layer's weights
                yield model[0].bias.grad  # Yield gradient of the first linear layer's bias

        self.check_output_and_recompiles(fn)  # Run the function and check output

    # Define a test case to test tensor gradient hooks using result.grad_fn.register_prehook
    def test_tensor_grad_hook2(self):
        def fn():
            for _ in range(3):
                # Create a neural network model with a single linear layer and ReLU activation
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([1, 4])  # Generate random input tensor

                result = model(x).sum()  # Forward pass and sum the output
                result.grad_fn.register_prehook(hook2)  # Register 'hook2' as a pre-hook to the gradient function
                result.backward()  # Perform backward propagation
                yield model[0].weight.grad  # Yield gradient of the first linear layer's weights
                yield model[0].bias.grad  # Yield gradient of the first linear layer's bias

        self.check_output_and_recompiles(fn)  # Run the function and check output
    # 定义一个测试函数 test_tensor_grad_hook3，用于测试梯度钩子的功能
    def test_tensor_grad_hook3(self):
        # 定义一个内部函数 fn，用于执行测试逻辑
        def fn():
            # 循环3次
            for _ in range(3):
                # 创建一个包含线性层和ReLU激活函数的模型
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),  # 输入维度为4，输出维度为4的线性层
                    torch.nn.ReLU(),        # ReLU激活函数
                )
                # 创建一个形状为[1, 4]的随机输入张量
                x = torch.randn([1, 4])

                # 对模型的输出求和，并将结果设置为可微张量
                result = model(x).sum()
                # 注册一个名为 hook3 的梯度钩子到 result 的梯度函数中
                result.grad_fn.register_hook(hook3)
                # 执行反向传播
                result.backward()
                # 生成模型第一个层的权重梯度
                yield model[0].weight.grad
                # 生成模型第一个层的偏置梯度
                yield model[0].bias.grad

        # 调用自定义的辅助方法 check_output_and_recompiles，验证输出并重新编译
        self.check_output_and_recompiles(fn)

    # 定义一个测试函数 test_torch_compile，用于测试 torch.compile 的功能
    def test_torch_compile(self):
        # 定义一个内部函数 fn，用于执行测试逻辑
        def fn():
            # 创建一个包含线性层和Sigmoid激活函数的模型
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),  # 输入维度为4，输出维度为4的线性层
                torch.nn.Sigmoid(),     # Sigmoid激活函数
            )
            # 使用 fullgraph=True 编译模型
            opt_model = torch.compile(model, fullgraph=True)

            # 循环3次
            for _ in range(3):
                # 创建一个形状为[1, 4]的随机输入张量
                x = torch.randn([1, 4])

                # 对优化后的模型的输出求和，并将结果设置为可微张量
                result = opt_model(x).sum()
                # 执行反向传播
                result.backward()
                # 生成模型第一个层的权重梯度
                yield model[0].weight.grad
                # 生成模型第一个层的偏置梯度
                yield model[0].bias.grad
                # 清空模型的梯度信息
                model.zero_grad()

        # 调用自定义的辅助方法 check_output_and_recompiles，验证输出并重新编译
        self.check_output_and_recompiles(fn)

    # 定义一个测试函数 test_torch_compile_api_inductor，用于测试 torch.compile 的 API 模式
    def test_torch_compile_api_inductor(self):
        # 定义一个内部函数 fn，用于执行测试逻辑
        def fn():
            # 设置随机种子为123
            torch.manual_seed(123)
            # 创建一个包含线性层和Sigmoid激活函数的模型
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),  # 输入维度为4，输出维度为4的线性层
                torch.nn.Sigmoid(),     # Sigmoid激活函数
            )

            res = []
            # 循环3次
            for _ in range(3):
                # 创建一个形状为[1, 4]的随机输入张量
                x = torch.randn([1, 4])

                # 对模型的输出求和，并将结果设置为可微张量
                result = model(x).sum()
                # 执行反向传播
                result.backward()
                # 将模型第一个层的权重梯度添加到结果列表中
                res.append(model[0].weight.grad)
                # 将模型第一个层的偏置梯度添加到结果列表中
                res.append(model[0].bias.grad)
                # 清空模型的梯度信息
                model.zero_grad()
            # 返回结果列表
            return res

        # 调用 fn 函数获取预期结果
        expected = fn()
        # 使用 config.patch(compiled_autograd=True) 打开编译自动求导的配置
        with config.patch(compiled_autograd=True):
            # 使用 torch.compile 编译 fn 函数
            compiled_fn = torch.compile(fn)
        # 执行编译后的函数获取实际结果
        actual = compiled_fn()
        # 断言预期结果和实际结果相等
        self.assertEqual(expected, actual)
        # 断言编译自动求导的捕获次数为1
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    # 定义一个测试函数 test_torch_compile_api_aot_eager，用于测试 torch.compile 的 AOT Eager 模式
    def test_torch_compile_api_aot_eager(self):
        # 定义一个内部函数 fn，用于执行测试逻辑
        def fn():
            # 设置随机种子为123
            torch.manual_seed(123)
            # 创建一个包含线性层和Sigmoid激活函数的模型
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),  # 输入维度为4，输出维度为4的线性层
                torch.nn.Sigmoid(),     # Sigmoid激活函数
            )

            res = []
            # 循环3次
            for _ in range(3):
                # 创建一个形状为[1, 4]的随机输入张量
                x = torch.randn([1, 4])

                # 对模型的输出求和，并将结果设置为可微张量
                result = model(x).sum()
                # 执行反向传播
                result.backward()
                # 将模型第一个层的权重梯度添加到结果列表中
                res.append(model[0].weight.grad)
                # 将模型第一个层的偏置梯度添加到结果列表中
                res.append(model[0].bias.grad)
                # 清空模型的梯度信息
                model.zero_grad()
            # 返回结果列表
            return res

        # 调用 fn 函数获取预期结果
        expected = fn()
        # 使用 config.patch(compiled_autograd=True) 打开编译自动求导的配置
        with config.patch(compiled_autograd=True):
            # 使用 torch.compile 编译 fn 函数，使用 backend="aot_eager" 参数指定 AOT Eager 模式
            compiled_fn = torch.compile(fn, backend="aot_eager")
        # 执行编译后的函数获取实际结果
        actual = compiled_fn()
        # 断言预期结果和实际结果相等
        self.assertEqual(expected, actual)
        # 断言编译自动求导的捕获次数为1
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
    # 定义一个测试函数，测试使用 Torch 的编译 API 在 eager 模式下的行为
    def test_torch_compile_api_eager(self):
        # 定义一个内部函数 fn，用于创建模型并执行多次前向传播和反向传播
        def fn():
            # 设置随机种子
            torch.manual_seed(123)
            # 创建一个简单的神经网络模型，包含一个线性层和一个 Sigmoid 激活函数
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.Sigmoid(),
            )

            res = []
            # 执行三次迭代
            for _ in range(3):
                # 生成一个随机输入张量
                x = torch.randn([1, 4])

                # 执行模型的前向传播，并对输出结果求和
                result = model(x).sum()
                # 对结果进行反向传播
                result.backward()
                # 收集第一个线性层的权重梯度和偏置梯度
                res.append(model[0].weight.grad)
                res.append(model[0].bias.grad)
                # 清空模型的梯度缓存
                model.zero_grad()
            return res

        # 生成预期结果
        expected = fn()
        
        # 使用编译后的自动求导功能进行函数编译，设置编译后端为 "eager"
        with config.patch(compiled_autograd=True):
            compiled_fn = torch.compile(fn, backend="eager")
        
        # 执行编译后的函数
        actual = compiled_fn()
        
        # 断言预期结果和实际结果相等
        self.assertEqual(expected, actual)
        
        # 断言编译自动求导的捕获计数为 1
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    # 定义测试函数，测试多次使用 Torch 编译功能的行为
    def test_multiple_torch_compile(self):
        # 创建一个简单的神经网络模型
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        x = torch.randn([1, 4])

        # 定义函数 fn，执行模型的前向传播和反向传播
        def fn():
            result = model(x).sum()
            result.backward()

        # 创建另一个线性模型和输入数据
        model2 = torch.nn.Linear(4, 4)
        x2 = torch.randn([1, 4])

        # 定义函数 fn2，执行第二个模型的前向传播和反向传播
        def fn2():
            result = model2(x2).sum()
            result.backward()

        # 编译 fn 函数，不启用编译自动求导
        no_ca1 = torch.compile(fn)
        no_ca1()
        
        # 断言编译自动求导的捕获计数为 0
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)
        counters.clear()

        # 启用编译自动求导，并编译 fn2 函数
        with config.patch(compiled_autograd=True):
            with_ca = torch.compile(fn2)
            with_ca()
            # 断言编译自动求导的捕获计数为 1
            self.assertEqual(counters["compiled_autograd"]["captures"], 1)
            counters.clear()

        # 再次编译 fn 函数，不启用编译自动求导
        no_ca2 = torch.compile(fn)
        no_ca2()
        
        # 断言编译自动求导的捕获计数为 0
        self.assertEqual(counters["compiled_autograd"]["captures"], 0)

    # 定义测试函数，测试使用 Torch 编译功能时可能的图结构破坏情况
    def test_torch_compile_graph_break(self):
        # 创建一个简单的神经网络模型和输入数据
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        x = torch.randn([1, 4])

        # 使用装饰器禁用 Torch 的动态图功能，定义函数 fn
        @torch._dynamo.disable()
        def fn():
            # 执行模型的前向传播和反向传播
            result = model(x).sum()
            result.backward()

        # 启用编译自动求导，并编译 fn 函数
        with config.patch(compiled_autograd=True):
            opt_fn = torch.compile(fn)
            opt_fn()

        # 断言编译自动求导的捕获计数为 1
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    # 定义测试函数，测试使用 Torch 编译功能时可能的图结构破坏情况（第二个场景）
    def test_torch_compile_graph_break2(self):
        # 创建一个简单的神经网络模型和输入数据
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        x = torch.randn([1, 4])

        # 定义内部函数 inner_fn，执行梯度反向传播
        @torch._dynamo.disable()
        def inner_fn(loss):
            loss.backward()

        # 定义函数 fn，执行模型的前向传播和内部函数的反向传播
        def fn():
            result = model(x).sum()
            inner_fn(result)

        # 启用编译自动求导，并编译 fn 函数
        with config.patch(compiled_autograd=True):
            opt_fn = torch.compile(fn)
            opt_fn()

        # 断言编译自动求导的捕获计数为 1
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
    # 定义一个测试方法，用于测试仅编译后向调用的情况
    def test_torch_compile_only_backward_call(self):
        # 创建一个包含线性层和Sigmoid激活函数的序列模型
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Sigmoid(),
        )
        # 生成一个形状为[1, 4]的随机张量作为输入
        x = torch.randn([1, 4])

        # 对模型进行前向传播并计算输出的和
        result = model(x).sum()
        
        # 使用配置，开启编译自动求导功能
        with config.patch(compiled_autograd=True):
            # 使用torch.compile将result.backward()编译为优化后的反向传播函数
            opt_bwd = torch.compile(lambda: result.backward())
            # 调用编译后的优化反向传播函数
            opt_bwd()

        # 断言编译自动求导的捕获次数为1
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)

    # 定义一个测试方法，用于测试带有嵌套函数的情况
    def test_dynamo_boxed(self):
        # 定义获取图中占位符节点的函数
        def get_placeholders(gm_):
            placeholders = []
            # 遍历图中的每个节点
            for node in gm_.graph.nodes:
                # 如果节点的操作是"placeholder"，则将其添加到占位符列表中
                if node.op == "placeholder":
                    placeholders.append(node)
            return placeholders

        # 定义一个包含内部编译器的函数，根据is_bwd参数检查占位符数量
        def eager_with_check(gm, is_bwd):
            def inner_compiler(gm_, example_inputs_):
                # 获取图中的占位符节点
                placeholders = get_placeholders(gm_)
                if is_bwd:
                    # 对于反向传播，占位符应为被包装的输入
                    assert len(placeholders) == 1
                    pass
                else:
                    # 对于前向传播，占位符数量应大于1
                    assert len(placeholders) > 1

                return gm_

            # 使用torch.compile将gm编译，并指定内部编译器为inner_compiler
            return torch.compile(gm, backend=inner_compiler)

        # 创建前向编译器函数，使用eager_with_check函数检查前向传播
        fwd_compiler_fn = functools.partial(eager_with_check, is_bwd=False)
        # 创建反向编译器函数，使用eager_with_check函数检查反向传播
        bwd_compiler_fn = functools.partial(eager_with_check, is_bwd=True)

        # 定义一个函数fn，接受inputs作为参数
        def fn(inputs):
            args_0, args_1, args_2 = inputs
            # 执行两次矩阵乘法
            out = torch.mm(args_0, args_1)
            out = torch.mm(out, args_2)
            # 计算输出的和作为损失
            loss = out.sum()
            # 使用编译自动求导，开启反向编译器函数bwd_compiler_fn
            with compiled_autograd.enable(bwd_compiler_fn):
                # 对损失进行反向传播
                loss.backward()
            # 返回args_0, args_1, args_2的梯度
            yield args_0.grad
            yield args_1.grad
            yield args_2.grad

        # 创建一个包含三个随机张量的inputs列表
        inputs = [
            torch.randn([1, 2], requires_grad=True),
            torch.randn([2, 3], requires_grad=True),
            torch.randn([3, 4], requires_grad=True),
        ]

        # 使用eager_with_check函数将fn编译为compiled_fn，检查前向传播
        compiled_fn = eager_with_check(fn, is_bwd=False)
        # 调用compiled_fn并将结果存储在grads列表中
        grads = list(compiled_fn(inputs))
        # 断言grads的长度为3
        self.assertEqual(len(grads), 3)
        # 断言grads中每个元素不为None
        self.assertNotEqual(grads[0], None)
        self.assertNotEqual(grads[1], None)
        self.assertNotEqual(grads[2], None)
    def test_inputs_aliasing_bytecode_attr_mutations(self):
        # Freeze compiled autograd graph
        # 创建编译后的自动微分图编译器实例
        compiler = torch._dynamo.compiled_autograd.AutogradCompilerInstance(compiler_fn)
        # 创建大小为100的全1张量 param
        param = torch.ones(100)
        # 创建大小为100的全2张量 activ
        activ = torch.ones(100) * 2
        # 将 param 和 activ 放入输入列表
        inputs = [param, activ]
        # 开始捕获编译后的图，获取代理和未使用的结果
        proxies, _ = compiler.begin_capture(inputs=inputs, sizes=[])
        # 将 param_proxy 和 activ_proxy 分别设置为 proxies 列表的第一和第二个元素
        param_proxy, activ_proxy = proxies
        # 计算 activ_proxy 的两倍，并存储在 buf 中
        buf = activ_proxy * 2
        # 在 param_proxy 上调用 inductor 库中的累积梯度操作
        torch.ops.inductor.accumulate_grad_.default(param_proxy, buf)
        # 结束捕获，获取运行时包装器和编译后的函数
        runtime_wrapper, compiled_fn = compiler.end_capture(buf)

        def bytecode_hook(code, out_code):
            import dis
            import sys

            # 确定调用操作的正确版本
            if sys.version_info < (3, 11):
                call_op = "CALL_FUNCTION"
            else:
                call_op = "CALL"

            # 获取编译后代码的指令列表
            insts = list(dis.get_instructions(out_code))
            # 找到 CALL 或 CALL_FUNCTION 操作的索引
            call_graph_idx = next(
                i for i, inst in enumerate(insts) if inst.opname == call_op
            )
            # 在调用图之前应该存在别名：inputs_ref_0 = inputs[0]
            # 查找存储 inputs[0] 的 STORE_FAST 指令
            matches = [
                inst
                for inst in insts[:call_graph_idx]
                if inst.opname == "STORE_FAST" and inst.argval == "inputs_ref_0"
            ]
            self.assertTrue(len(matches) == 1)
            # 在调用图之后应该访问 inputs_ref_0 而不是 inputs
            # 查找使用 inputs 的指令
            matches = [
                inst for inst in insts[call_graph_idx:] if inst.argval == "inputs"
            ]
            self.assertTrue(len(matches) == 0)
            # 在调用图之后应该加载 inputs_ref_0 而不是 inputs
            # 查找加载 inputs_ref_0 的 LOAD_FAST 指令
            matches = [
                inst
                for inst in insts[call_graph_idx:]
                if inst.opname == "LOAD_FAST" and inst.argval == "inputs_ref_0"
            ]
            self.assertTrue(len(matches) == 1)

        # 重置 Torch 的动态计算图
        torch._dynamo.reset()
        # 注册字节码钩子函数
        handle = torch._dynamo.convert_frame.register_bytecode_hook(bytecode_hook)
        try:
            # 运行编译后的函数，传入参数 param 和 activ
            runtime_wrapper(
                compiled_fn=compiled_fn, inputs=[param, activ], sizes=(), hooks=()
            )
        finally:
            # 移除字节码钩子函数
            handle.remove()
    # 设置全局日志级别为警告
    logging.getLogger().setLevel(logging.WARNING)
    # 导入 LoggingTensor 类
    from torch.testing._internal.logging_tensor import LoggingTensor

    # 定义一个允许输入窃取的图形
    def forward(inputs):
        # 对第一个输入执行加法操作
        add = inputs[0] + 1
        # 将上一步结果与第二个输入相加，处理张量子类的后缀
        add_1 = add + inputs[1]
        # 将结果转移到 CPU
        out = add_1.cpu()
        return (out,)

    # 对 forward 函数进行符号化跟踪
    gm = torch.fx.symbolic_trace(forward)
    # 设置允许本地变量窃取的输入列表为 ["inputs"]
    torch._dynamo.utils.set_locals_to_steal(gm, ["inputs"])
    # 编译符号化图形
    compiled_fn = torch.compile(gm)

    # 创建输入列表，包括一个大张量和一个 LoggingTensor 实例
    inputs = [
        torch.ones(1000000, dtype=torch.float32),
        LoggingTensor(torch.ones(1)),
    ]

    # 定义字节码钩子函数
    def bytecode_hook(code, out_code):
        import dis
        import sys

        # 根据 Python 版本选择调用操作符名称
        if sys.version_info < (3, 11):
            call_op = "CALL_FUNCTION"
        else:
            call_op = "CALL"

        # 获取字节码指令列表
        insts = list(dis.get_instructions(out_code))
        # 查找调用操作的索引
        call_graph_idx = next(
            i for i, inst in enumerate(insts) if inst.opname == call_op
        )
        # 在调用图之前应该别名化：inputs_ref_0 = inputs[0]
        matches = [
            inst
            for inst in insts[:call_graph_idx]
            if inst.opname == "STORE_FAST" and inst.argval == "inputs_ref_0"
        ]
        self.assertTrue(len(matches) == 1)  # 断言确保只有一个匹配项
        # 在调用图之后应该访问 inputs_ref_0 而不是 inputs
        matches = [
            inst for inst in insts[call_graph_idx:] if inst.argval == "inputs"
        ]
        self.assertTrue(len(matches) == 0)  # 断言确保没有匹配项
        matches = [
            inst
            for inst in insts[call_graph_idx:]
            if inst.opname == "LOAD_FAST" and inst.argval == "inputs_ref_0"
        ]
        self.assertTrue(len(matches) == 1)  # 断言确保只有一个匹配项

    # 重置 torch._dynamo
    torch._dynamo.reset()
    # 注册字节码钩子函数
    handle = torch._dynamo.convert_frame.register_bytecode_hook(bytecode_hook)
    try:
        # 调用编译后的函数，处理输入列表
        out = compiled_fn(inputs)
        self.assertTrue(len(inputs) == 0)  # 断言确保输入列表已清空
    finally:
        # 移除字节码钩子函数
        handle.remove()
    # 定义测试方法，验证所有叶子节点的输出
    def test_output_nodes_all_leaves(self):
        # 定义内部函数 fn
        def fn():
            # 创建随机张量 y 和 z，要求计算梯度
            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)

            # 定义模型函数 model，接受输入 x，并返回经过复杂计算后的结果
            def model(x):
                return torch.sigmoid(x * z + torch.sin(y) + torch.cos(y))

            # 循环三次，生成随机输入 x
            for _ in range(3):
                x = torch.randn([1, 4])

                # 对模型输出结果求和
                result = model(x).sum()
                
                # 对结果分别对 y 和 z 求梯度
                gy, gz = torch.autograd.grad(result, inputs=[y, z])
                
                # 断言 y 和 z 的梯度应为空
                assert y.grad is None
                assert z.grad is None
                
                # 返回生成的梯度 gy 和 gz
                yield gy
                yield gz

        # 调用类中的检查输出和重新编译函数，验证函数 fn 的输出
        self.check_output_and_recompiles(fn)

    # 定义测试方法，验证部分叶子节点的输出
    def test_output_nodes_some_leaves(self):
        # 定义内部函数 fn
        def fn():
            # 定义自定义反向传播函数 UnreachableBwd
            class UnreachableBwd(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    # 在反向传播时抛出运行时错误
                    raise RuntimeError

            # 创建随机张量 y 和 z，要求计算梯度
            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)

            # 定义模型函数 model，接受输入 x，并返回经过复杂计算后的结果
            def model(x):
                return torch.sigmoid(UnreachableBwd.apply(y) * z)

            # 循环三次，生成随机输入 x
            for _ in range(3):
                x = torch.randn([1, 4])

                # 对模型输出结果求和
                result = model(x).sum()
                
                # 对结果对 z 求梯度
                gz = torch.autograd.grad(result, inputs=[z])
                
                # 断言 y 和 z 的梯度应为空
                assert y.grad is None
                assert z.grad is None
                
                # 返回生成的梯度 gz
                yield gz

        # 调用类中的检查输出和重新编译函数，验证函数 fn 的输出
        self.check_output_and_recompiles(fn)

    # 定义测试方法，验证所有叶子节点的输出（无输出节点）
    def test_no_output_nodes_all_leaves(self):
        # 定义内部函数 fn
        def fn():
            # 创建随机张量 y 和 z，要求计算梯度
            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)

            # 定义模型函数 model，接受输入 x，并返回经过复杂计算后的结果
            def model(x):
                return torch.sigmoid(x * z + torch.sin(y) + torch.cos(y))

            # 循环三次，生成随机输入 x
            for _ in range(3):
                x = torch.randn([1, 4])
                
                # 对模型输出结果求和
                result = model(x).sum()

                # 对结果执行反向传播，此处期望返回空值
                out = result.backward()

                # 断言反向传播输出为空，y 和 z 的梯度应为非空
                assert out is None
                assert y.grad is not None
                assert z.grad is not None
                
                # 返回生成的梯度 y.grad 和 z.grad，并重置它们为 None
                yield y.grad
                yield z.grad
                y.grad = None
                z.grad = None

        # 调用类中的检查输出和重新编译函数，验证函数 fn 的输出
        self.check_output_and_recompiles(fn)
    # 定义测试函数 test_no_output_nodes_some_leaves，该函数测试没有输出节点但有一些叶子节点的情况
    def test_no_output_nodes_some_leaves(self):
        # 定义内部函数 fn
        def fn():
            # 定义一个自定义的 PyTorch 自动求导函数 UnreachableBwd
            class UnreachableBwd(torch.autograd.Function):
                # 前向传播方法，直接返回输入 x
                @staticmethod
                def forward(ctx, x):
                    return x

                # 反向传播方法，抛出 RuntimeError
                @staticmethod
                def backward(ctx, gO):
                    raise RuntimeError

            # 随机初始化张量 y, z, a，并设置 requires_grad=True，使其支持梯度计算
            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)
            a = torch.randn(1, 4, requires_grad=True)

            # 定义模型函数 model，其计算结果为 torch.sigmoid(x * y * z * UnreachableBwd.apply(a))
            def model(x):
                return torch.sigmoid(x * y * z * UnreachableBwd.apply(a))

            # 循环执行3次
            for _ in range(3):
                # 随机初始化输入张量 x，形状为 [1, 4]
                x = torch.randn([1, 4])
                # 计算模型的输出结果的和
                result = model(x).sum()
                # 对结果进行反向传播，仅针对 y 和 z
                out = result.backward(inputs=[y, z])
                # 断言输出为 None
                assert out is None
                # 断言 y 和 z 的梯度不为 None
                assert y.grad is not None
                assert z.grad is not None
                # 断言 a 的梯度为 None
                assert a.grad is None
                # 返回 y 和 z 的梯度作为生成器的结果
                yield y.grad
                yield z.grad
                # 清空 y 和 z 的梯度
                y.grad = None
                z.grad = None

        # 调用 self.check_output_and_recompiles 方法，检查输出并重新编译函数 fn
        self.check_output_and_recompiles(fn)

    # 定义测试函数 test_no_output_nodes_different_leaves_will_recompile，该函数测试不同叶子节点将重新编译的情况
    def test_no_output_nodes_different_leaves_will_recompile(self):
        # 定义内部函数 fn
        def fn():
            # 定义前向传播函数 fwd，计算 x * y * z 的和
            def fwd(x, y, z):
                out = x * y  # MulBackward0
                out2 = out * z  # MulBackward0
                return out2.sum()  # SumBackward0

            # 随机初始化张量 x, y, z，并设置 requires_grad=True，使其支持梯度计算
            x = torch.randn(5, requires_grad=True)
            y = torch.randn(5, requires_grad=True)
            z = torch.randn(5, requires_grad=True)
            # 计算前向传播的结果 loss
            loss = fwd(x, y, z)
            # 使用 torch.compile 重新编译梯度计算函数，并执行
            torch.compile(lambda: torch.autograd.backward(loss, inputs=[x]))()
            # 返回 x 的梯度作为生成器的结果
            yield x.grad
            # 清空 x 的梯度
            x.grad = None

            # 再次计算前向传播的结果 loss
            loss = fwd(x, y, z)
            # 使用 torch.compile 重新编译梯度计算函数，并执行
            torch.compile(lambda: torch.autograd.backward(loss, inputs=[y]))()
            # 返回 y 的梯度作为生成器的结果
            yield y.grad

        # 调用 self.check_output_and_recompiles 方法，检查输出并重新编译函数 fn，期望执行两次
        self.check_output_and_recompiles(fn, 2)

    # 定义测试函数 test_dynamic_shapes，该函数测试动态形状的情况
    def test_dynamic_shapes(self):
        # 定义内部函数 fn
        def fn():
            # 构建一个包含线性层和激活函数的序列模型
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            # 使用 torch.compile 对模型进行重新编译，支持动态输入形状
            opt_model = torch.compile(model, dynamic=True)

            # 循环遍历不同的批次大小 b
            for b in range(10, 100, 10):
                # 随机初始化输入张量 x，形状为 [b, 4]
                x = torch.randn([b, 4])
                # 计算优化后模型的输出结果的和
                result = opt_model(x).sum()
                # 对结果进行反向传播
                result.backward()
                # 返回模型第一层权重和偏置的梯度作为生成器的结果
                yield model[0].weight.grad
                yield model[0].bias.grad
                # 返回模型第三层权重和偏置的梯度作为生成器的结果
                yield model[2].weight.grad
                yield model[2].bias.grad
                # 清空模型所有层的梯度
                model.zero_grad()

        # TODO(jansel): we should be able to get this count to 1
        # 调用 self.check_output_and_recompiles 方法，检查输出并重新编译函数 fn，期望执行两次，count 参数暂时为 2
        self.check_output_and_recompiles(fn, count=2)
    # 测试在不包含零值的情况下进行累积
    def test_accumulate_without_zero(self):
        # 定义一个函数
        def fn():
            # 创建一个包含线性层和激活函数的神经网络模型
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            # 使用动态编译方式编译模型
            opt_model = torch.compile(model, dynamic=True)

            # 进行10次迭代
            for _ in range(10):
                # 生成一个随机输入张量
                x = torch.randn([10, 4])
                # 对模型进行前向传播并求和
                result = opt_model(x).sum()
                # 反向传播
                result.backward()
                # 生成模型第一层权重的梯度
                yield model[0].weight.grad.clone()
                # 生成模型第一层偏置的梯度
                yield model[0].bias.grad.clone()
                # 生成模型第三层权重的梯度
                yield model[2].weight.grad.clone()
                # 生成模型第三层偏置的梯度
                yield model[2].bias.grad.clone()

        # 检查输出并重新编译模型
        self.check_output_and_recompiles(fn, count=2)

    # 测试原地梯度更新
    def test_inplace_grad_update(self):
        # 定义一个函数
        def fn():
            # 创建一个包含线性层和激活函数的神经网络模型
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            # 使用动态编译方式编译模型
            opt_model = torch.compile(model, dynamic=True)

            # 进行10次迭代
            for _ in range(10):
                # 生成一个与模型第一层权重相同形状的随机梯度张量
                w_grad = torch.rand_like(model[0].weight)
                # 生成一个与模型第一层偏置相同形状的随机梯度张量
                b_grad = torch.rand_like(model[0].bias)
                # 将模型第一层权重的梯度更新为w_grad
                model[0].weight.grad = w_grad
                # 将模型第一层偏置的梯度更新为b_grad
                model[0].bias.grad = b_grad

                # 生成一个随机输入张量
                x = torch.randn([10, 4])
                # 对模型进行前向传播并求和
                result = opt_model(x).sum()
                # 反向传播
                result.backward()
                # 断言模型第一层权重的梯度与w_grad相同
                assert model[0].weight.grad is w_grad
                # 断言模型第一层偏置的梯度与b_grad相同
                assert model[0].bias.grad is b_grad
                # 生成w_grad的克隆
                yield w_grad.clone()
                # 生成b_grad的克隆
                yield b_grad.clone()

        # 检查输出并重新编译模型
        self.check_output_and_recompiles(fn, count=1)

    # 如果没有CUDA，则跳过测试
    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    def test_issue106555(self):
        # 设置运行在 cuda:0 设备上
        DEVICE = torch.device("cuda:0")
        # 定义特征数量
        NUM_FEATURES = 256

        def bias_sigmoid_mul(x1, x2, bias):
            # 对 x2 添加偏置并应用 sigmoid 函数
            x2 = torch.sigmoid(x2 + bias)
            # 计算 x1 和经过 sigmoid 的 x2 的乘积
            y = x1 * x2
            return y

        # 编译 JIT 版本的 bias_sigmoid_mul 函数
        bias_sigmoid_mul_jit = torch.compile(bias_sigmoid_mul)

        class ModuleWithJit(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个具有偏置的线性层
                self.linear_1 = nn.Linear(NUM_FEATURES, NUM_FEATURES, bias=True)
                # 创建一个不带偏置的线性层
                self.linear_2 = nn.Linear(NUM_FEATURES, NUM_FEATURES, bias=False)
                # 定义线性层 2 的偏置参数
                self.linear_2_bias = nn.Parameter(torch.zeros(NUM_FEATURES))

            def forward(self, input_tensor):
                # 计算线性层 1 的输出
                x1 = self.linear_1(input_tensor)
                # 计算线性层 2 的输出，并应用 JIT 版本的 bias_sigmoid_mul 函数
                x2 = self.linear_2(input_tensor)
                output = bias_sigmoid_mul_jit(x1, x2, self.linear_2_bias)
                return output

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建两个具有 JIT 支持的 ModuleWithJit 实例
                self.module_with_jit_1 = ModuleWithJit()
                self.module_with_jit_2 = ModuleWithJit()

            def forward(self, x, gradient_checkpointing: bool):
                # 如果使用梯度检查点，则调用 checkpoint 函数
                if gradient_checkpointing:
                    y = torch.utils.checkpoint.checkpoint(
                        self._forward, x, use_reentrant=True
                    )
                else:
                    # 否则直接调用 _forward 函数
                    y = self._forward(x)
                return y

            def _forward(self, x):
                # 将输入 x 和 module_with_jit_1 的输出相加
                x = x + self.module_with_jit_1(x)
                # 对 x 进行转置后，将其与 module_with_jit_2 的输出相加，并再次转置
                x = x + self.module_with_jit_2(x.transpose(-2, -3)).transpose(-2, -3)
                return x

        # 设置当前 CUDA 设备
        torch.cuda.set_device(device=DEVICE)
        # 设置随机数种子
        torch.manual_seed(1234567890)
        # 创建模型实例
        model = Model()
        # 将模型设置为训练模式
        model.train()
        # 将模型移动到指定设备上
        model.to(device=DEVICE)
        # 获取模型的所有参数列表
        model_parameters = list(model.parameters())

        # 重新设置随机数种子
        torch.manual_seed(1234567890)
        # 创建输入张量，并将其移动到指定设备上，并设置其需要计算梯度
        input_tensor = torch.randn(1, 128, 256, NUM_FEATURES).to(device=DEVICE)
        input_tensor.requires_grad = True
        # 创建目标张量，并将其移动到指定设备上，类型与输入张量一致
        target_tensor = torch.randn(1, 128, 256, NUM_FEATURES).to(
            dtype=input_tensor.dtype, device=DEVICE
        )

        # 进行 10 轮迭代
        for iteration in range(10):
            # 清空所有参数的梯度
            for param in model_parameters:
                param.grad = None
            # 使用模型进行前向传播，使用梯度检查点
            output_tensor = model(
                x=input_tensor.clone(),
                gradient_checkpointing=True,
            )
            # 计算损失，使用目标张量与输出张量之间的绝对误差的平均值
            loss = torch.mean(torch.abs(target_tensor - output_tensor))
            # 反向传播计算梯度
            loss.backward()
    def test_keep_graph_simple(self):
        # 创建一个张量 x，并指定需要计算梯度
        x = torch.tensor([2.0], requires_grad=True)
        # 计算 y = x^2
        y = x**2

        # 第一次反向传播，保留计算图
        y.backward(retain_graph=True)
        # 断言 x 的梯度为 4，因为在 x=2 时，dy/dx = 4
        self.assertEqual(x.grad, torch.Tensor([4]))  # dy/dx at x=2 is 4

        # 注意 - 这段代码会在即时执行和编译后的模式下运行
        def fn():
            # 重置梯度
            x.grad = torch.tensor([0.0])
            # 第二次和第三次反向传播，保留计算图
            y.backward(retain_graph=True)
            # 断言 x 的梯度为 4，因为在 x=2 时，dy/dx = 4
            self.assertEqual(x.grad, torch.Tensor([4]))  # dy/dx at x=2 is 4
            return x.grad

        # 调用辅助函数检查输出并重新编译
        self.check_output_and_recompiles(fn, count=1)

    def test_keep_graph_usage_after_compiled(self):
        # 创建一个张量 x，并指定需要计算梯度
        x = torch.tensor([2.0], requires_grad=True)
        # 计算 y = x^2
        y = x**2

        # 第一次反向传播，保留计算图
        def eager_check():
            y.backward(retain_graph=True)
            # 断言 x 的梯度为 4，因为在 x=2 时，dy/dx = 4
            self.assertEqual(x.grad, torch.Tensor([4]))  # dy/dx at x=2 is 4
            x.grad = torch.tensor([0.0])

        eager_check()

        for i in range(0, 5):
            # 使用编译后自动求导
            with compiled_autograd.enable(compiler_fn):
                eager_check()

            eager_check()

    def test_custom_fn_saved_tensors(self):
        def fn():
            # 定义自定义的 autograd 函数 MySin
            class MySin(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    # 保存输入张量 x 到上下文中
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    # 从上下文中读取保存的张量 x
                    (x,) = ctx.saved_tensors
                    # 返回梯度 gO * cos(x)
                    return gO * torch.cos(x)

            # 遍历不同的范围值 i
            for i in [10, 100, 10, 15, 20, 25]:
                # 创建张量 x，指定需要计算梯度
                x = torch.arange(0.0, i, requires_grad=True)
                # 调用 MySin 函数进行前向传播
                out = MySin.apply(x)
                # 计算损失函数，这里是输出的和
                loss = out.sum()
                # 反向传播计算梯度
                loss.backward()
                # 生成 x 的梯度
                yield x.grad

        # 调用辅助函数检查输出并重新编译
        self.check_output_and_recompiles(fn, count=2)

    def test_custom_fn_saved_multiple_tensors(self):
        def fn():
            # 定义自定义的 autograd 函数 MyFn
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, y):
                    # 保存输入张量 x 和 y 到上下文中
                    ctx.save_for_backward(x, y)
                    # 返回 sin(x) 和 sin(y)
                    return torch.sin(x), torch.sin(y)

                @staticmethod
                def backward(ctx, gO_x, gO_y):
                    # 从上下文中读取保存的张量 x 和 y
                    (x, y) = ctx.saved_tensors
                    # 返回梯度 gO_x * cos(x), gO_y * cos(y)
                    return gO_x * torch.cos(x), gO_y * torch.cos(y)

            # 遍历不同的范围值 i
            for i in [10, 100, 10, 15, 20, 25]:
                # 创建张量 x 和 y，指定需要计算梯度
                x = torch.arange(0.0, i, requires_grad=True)
                y = torch.arange(0.0, i, requires_grad=True)
                # 调用 MyFn 函数进行前向传播
                out1, out2 = MyFn.apply(x, y)
                # 计算损失函数，这里是输出的乘积的和
                loss = (out1 * out2).sum()
                # 反向传播计算梯度
                loss.backward()
                # 生成 x 的梯度
                yield x.grad

        # 调用辅助函数检查输出并重新编译
        self.check_output_and_recompiles(fn, count=2)
    def test_custom_fn_saved_multiple_tensors_dedup(self):
        # 定义一个测试函数，用于测试自定义函数在保存多个张量并去重时的功能
        def fn():
            # 定义一个自定义的 PyTorch 函数类 MyFn
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    # 在前向传播中保存张量 x 两次
                    ctx.save_for_backward(x, x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    # 在反向传播中获取保存的张量 x1 和 x2
                    (x1, x2) = ctx.saved_tensors
                    return gO * torch.cos(x1) * torch.cos(x2)

            # 对于一组不同的长度进行迭代
            for i in [10, 100, 10, 15, 20, 25]:
                # 创建一个张量 x，设置 requires_grad=True 以便计算梯度
                x = torch.arange(0.0, i, requires_grad=True)
                # 应用自定义函数 MyFn 的前向传播
                out = MyFn.apply(x)
                # 计算输出的和作为损失
                loss = out.sum()
                # 对损失进行反向传播
                loss.backward()
                # 生成当前张量 x 的梯度
                yield x.grad

        # 调用测试辅助函数，验证输出并重新编译
        self.check_output_and_recompiles(fn, count=2)

    def test_custom_fn_saved_shape_tensor(self):
        # 定义一个测试函数，用于测试自定义函数在保存张量形状时的功能
        def fn():
            # 定义一个自定义的 PyTorch 函数类 MyFn
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    # 在前向传播中保存张量 x
                    ctx.save_for_backward(x)
                    return x

                @staticmethod
                def backward(ctx, gO):
                    # 在反向传播中获取保存的张量 x，并返回梯度乘以张量 x 的长度
                    (x,) = ctx.saved_tensors
                    return gO * x.shape[0]

            # 对于一组不同的长度进行迭代
            for i in [10, 100, 10, 15, 20, 25]:
                # 创建一个张量 x，设置 requires_grad=True 以便计算梯度
                x = torch.arange(0.0, i, requires_grad=True)
                # 应用自定义函数 MyFn 的前向传播
                out = MyFn.apply(x)
                # 计算输出的和作为损失
                loss = out.sum()
                # 对损失进行反向传播
                loss.backward()
                # 生成当前张量 x 的梯度
                yield x.grad

        # 调用测试辅助函数，验证输出并重新编译
        self.check_output_and_recompiles(fn, count=2)

    def test_custom_fn_saved_attr(self):
        # 定义一个测试函数，用于测试自定义函数在保存张量形状属性时的功能
        def fn():
            # 定义一个自定义的 PyTorch 函数类 MyFn
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    # 在前向传播中保存张量 x 的形状属性
                    ctx.shape = x.shape
                    return x

                @staticmethod
                def backward(ctx, gO):
                    # 在反向传播中获取保存的张量形状属性，并返回梯度乘以形状的第一个维度
                    x_shape = ctx.shape[0]
                    return gO * x_shape

            # 对于一组不同的长度进行迭代
            for i in [10, 100, 10, 15, 20, 25]:
                # 创建一个张量 x，设置 requires_grad=True 以便计算梯度
                x = torch.arange(0.0, i, requires_grad=True)
                # 应用自定义函数 MyFn 的前向传播
                out = MyFn.apply(x)
                # 计算输出的和作为损失
                loss = out.sum()
                # 对损失进行反向传播
                loss.backward()
                # 生成当前张量 x 的梯度
                yield x.grad

        # 调用测试辅助函数，验证输出并重新编译，使用特定的编译器函数
        self.check_output_and_recompiles(
            fn, count=2, compiler_fn=make_compiler_fn(fullgraph=False)
        )

    def test_custom_fn_multiple_grads(self):
        # 定义一个测试函数，用于测试自定义函数在处理多个梯度时的功能
        def fn():
            # 定义一个自定义的 PyTorch 函数类 MyFn
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, y):
                    # 前向传播中返回两个张量的和以及第二个张量
                    return x + y, y

                @staticmethod
                def backward(ctx, gO_1, gO_2):
                    # 反向传播中直接返回两个梯度
                    return gO_1, gO_2

            # 对于一组不同的长度进行迭代
            for i in [10, 100, 10, 15, 20, 25]:
                # 创建两个张量 x 和 y，设置 requires_grad=True 以便计算梯度
                x = torch.arange(0.0, i, requires_grad=True)
                y = torch.arange(0.0, i, requires_grad=True)
                # 应用自定义函数 MyFn 的前向传播
                out1, out2 = MyFn.apply(x, y)
                # 计算输出的和作为损失
                loss = (out1 + out2).sum()
                # 对损失进行反向传播
                loss.backward()
                # 生成当前张量 x 和 y 的梯度
                yield x.grad
                yield y.grad

        # 调用测试辅助函数，验证输出并重新编译
        self.check_output_and_recompiles(fn, count=2)
    def test_custom_fn_non_variable_input(self):
        # 定义一个测试函数，测试自定义函数对非变量输入的处理
        def fn():
            # 定义一个自定义的 torch.autograd.Function 类
            class MyFn(torch.autograd.Function):
                @staticmethod
                # 前向传播函数，接收输入参数 x, y, z
                def forward(ctx, x, y, z):
                    return x * 2, y * 3, z * 4

                @staticmethod
                # 反向传播函数，接收梯度 gO_1, gO_2, gO_3
                def backward(ctx, gO_1, gO_2, gO_3):
                    return gO_1, gO_2, gO_3

            # 对于一系列的数值进行迭代
            for i in [10, 100, 10, 15, 20, 25]:
                # 创建一个 tensor x，从 0 到 i-1，需要计算梯度
                x = torch.arange(0.0, i, requires_grad=True)
                # 设置常量 y 为 1
                y = 1
                # 创建一个 tensor z，从 0 到 i-1，需要计算梯度
                z = torch.arange(0.0, i, requires_grad=True)
                # 调用 MyFn 的前向传播计算
                out1, out2, out3 = MyFn.apply(x, y, z)
                # 计算损失，将输出结果求和
                loss = (out1 + out2 + out3).sum()
                # 反向传播计算梯度
                loss.backward()
                # 生成器函数，每次返回 x, y, z 中的一个
                yield x
                yield y
                yield z

        # 调用测试辅助函数，检查输出并重新编译
        self.check_output_and_recompiles(fn, count=2)

    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    def test_logging_tensor_flaky(self) -> None:
        # 当首次运行某些包含 Triton 的测试，然后运行 test_inputs_aliasing_bytecode_stack_restore 时可能出现以下问题:
        #   - pytest: `TypeError: unsupported operand type(s) for +: 'Tensor' and 'LoggingTensor'`
        #   - python: `TypeError: not all arguments converted during string formatting`

        # 1. 某些涉及 Triton 的测试
        def fn():
            # 内部函数 _fn 返回输入 x
            def _fn(x):
                return x

            # 创建一个在 CUDA 设备上的 float16 类型的 tensor x，需要计算梯度
            x = torch.arange(
                1, 10, requires_grad=True, dtype=torch.float16, device="cuda"
            )
            # 调用 _fn 函数
            out = _fn(x)
            # 计算损失，对输出结果进行求和
            loss = out.sum()
            # 反向传播计算梯度
            loss.backward()

        # 启用编译自动求导功能，并调用 fn 函数
        with compiled_autograd.enable(compiler_fn):
            fn()

        # 将日志记录器的日志级别设置为 WARNING，因为 Triton 设置过程中将其覆盖为 INFO
        logging.getLogger().setLevel(logging.WARNING)

        # 2. test_inputs_aliasing_bytecode_stack_restore
        from torch.testing._internal.logging_tensor import LoggingTensor

        # 定义一个前向传播函数，接收输入 inputs
        def forward(inputs):
            add = inputs[0] + 1
            add_1 = add + inputs[1]
            out = add_1.cpu()
            return (out,)

        # 使用 symbolic_trace 对 forward 函数进行符号化追踪
        gm = torch.fx.symbolic_trace(forward)
        # 打印可读的图形表示
        print(gm.print_readable())
        # 设置 gm 中的 locals 变量以进行优化
        torch._dynamo.utils.set_locals_to_steal(gm, ["inputs"])
        # 编译符号化追踪后的函数
        compiled_fn = torch.compile(gm)

        # 创建输入 inputs，包括一个大尺寸的 float32 类型的 tensor 和一个 LoggingTensor
        inputs = [
            torch.ones(1000000, dtype=torch.float32),
            LoggingTensor(torch.ones(1)),
        ]

        # 调用编译后的函数并传入 inputs
        compiled_fn(inputs)

    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    # 测试自定义函数输出元数据的方法
    def test_custom_fn_output_metadata(self):
        # 定义一个编译器函数，接收参数 gm（图模型）
        def my_compiler_fn(gm):
            # 遍历图模型中的每个节点
            for node in gm.graph.nodes:
                # 检查节点的目标是否为 torch._ops.OpOverload 类型
                if isinstance(node.target, torch._ops.OpOverload):
                    # 断言节点目标的名称不是 "aten::_to_copy"
                    assert (
                        node.target._name != "aten::_to_copy"
                    ), "there should be no implicit copies (e.g. dtype casting)"

            # 定义内部编译器函数，接收图模型 gm_ 和示例输入 example_inputs_
            def inner_compiler(gm_, example_inputs_):
                # 计数器中编译自动梯度的计数器增加 1
                counters["compiled_autograd"]["compiles"] += 1
                # 使用 inductor 编译图模型 gm_ 和示例输入 example_inputs_
                return inductor.compile(gm_, example_inputs_)

            # 调用 torch.compile 函数，传入 gm 和自定义的内部编译器函数
            return torch.compile(
                gm, backend=inner_compiler, fullgraph=True, dynamic=True
            )

        # 定义一个函数 fn
        def fn():
            # 定义一个名为 MyFn 的自动求导函数类
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return gO

            # 使用 torch.arange 创建一个张量 x，设置 requires_grad=True，dtype=torch.float16，device="cuda"
            x = torch.arange(
                1, 10, requires_grad=True, dtype=torch.float16, device="cuda"
            )
            # 调用 x.view(3, 3) 生成 x 的视图 x_view
            x_view = x.view(3, 3)
            # 使用 MyFn.apply 对 x_view 应用自动求导函数 MyFn，得到输出 out
            out = MyFn.apply(x_view)
            # 计算 out 的和作为损失值 loss
            loss = out.sum()
            # 对损失值 loss 进行反向传播
            loss.backward()
            # 生成器函数，依次产生张量 x 的 dtype、device 和梯度 x.grad
            yield x.dtype
            yield x.device
            yield x.grad

        # 调用 self.check_output_and_recompiles 函数，传入 fn 和 count=1 进行检查输出和重新编译

    # 测试具有相同图形的自定义函数的方法
    def test_custom_fn_with_same_graph(self):
        # 定义一个函数 fn
        def fn():
            # 定义一个名为 MyFn1 的自动求导函数类
            class MyFn1(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return gO

            # 定义一个名为 MyFn2 的自动求导函数类，与 MyFn1 相同，但具有不同的自动求导函数 ID
            class MyFn2(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return gO

            # 遍历列表中的 MyFn1 和 MyFn2
            for myfn in [MyFn1, MyFn2, MyFn1, MyFn2]:
                # 使用 torch.arange 创建一个浮点数张量 x，设置 requires_grad=True
                x = torch.arange(0.0, 10, requires_grad=True)
                # 对 x 应用 myfn 函数，得到输出 out
                out = myfn.apply(x)
                # 计算 out 的和作为损失值 loss
                loss = out.sum()
                # 对损失值 loss 进行反向传播
                loss.backward()
                # 生成器函数，产生张量 x 的梯度 x.grad
                yield x.grad

        # 调用 self.check_output_and_recompiles 函数，传入 fn 和 count=2 进行检查输出和重新编译，应对 MyFn1 和 MyFn2 每个编译一次
    def test_custom_fn_dynamically_defined_class(self):
        # 定义一个测试方法，用于测试动态定义的函数类
        def fn():
            # 定义内部函数create_class，用于创建一个具有动态定义函数的类
            def create_class(multiplier: int):
                # 定义一个继承自torch.autograd.Function的动态函数类DynamicFn
                class DynamicFn(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, x):
                        # 前向传播方法，返回输入x乘以multiplier后的结果
                        return x * multiplier

                    @staticmethod
                    def backward(ctx, gO):
                        # 反向传播方法，返回梯度gO乘以multiplier后的结果
                        return gO * multiplier

                return DynamicFn

            # 遍历multiplier列表，每个迭代中创建一个新的tensor x，并应用动态函数类，计算损失和反向传播
            for multiplier in [10, 20, 30]:
                x = torch.arange(0.0, 10, requires_grad=True)
                out = create_class(multiplier).apply(x)
                loss = out.sum()
                loss.backward()
                # 生成当前x的梯度，并返回
                yield x.grad

        # 调用测试辅助函数，验证输出并重新编译，执行三次测试函数fn
        self.check_output_and_recompiles(fn, count=3)

    def test_custom_fn_bw_graph_break(self):
        # 定义一个测试方法，用于测试自定义函数的反向传播和图断裂情况
        def fn():
            # 定义一个继承自torch.autograd.Function的自定义函数类MySin
            class MySin(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    # 前向传播方法，保存输入x到上下文中，并返回sin(x)的结果
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    # 反向传播方法，输出“graph break”信息，计算梯度并返回
                    print("graph break")
                    (x,) = ctx.saved_tensors
                    return gO * torch.cos(x)

            # 遍历列表i，每个迭代中创建一个新的tensor x，并应用自定义函数类，计算损失和反向传播
            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MySin.apply(x)
                loss = out.sum()
                loss.backward()
                # 生成当前x的梯度，并返回
                yield x.grad

        # 调用测试辅助函数，验证输出并重新编译，执行fn函数，期望输出次数为[2, 6]
        self.check_output_and_recompiles(
            fn, count=[2, 6], compiler_fn=make_compiler_fn(fullgraph=False)
        )

    def test_custom_fn_compiled_fw_graph_break(self):
        # 定义一个测试方法，用于测试编译优化的自定义函数的前向传播和图断裂情况
        def fn():
            # 定义一个继承自torch.autograd.Function的自定义函数类MySin
            class MySin(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    # 前向传播方法，输出“graph break”信息，保存输入x到上下文中，并返回sin(x)的结果
                    print("graph break")
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    # 反向传播方法，从上下文中获取保存的tensor x，计算梯度并返回
                    (x,) = ctx.saved_tensors
                    return gO * torch.cos(x)

            # 使用torch.compile优化MySin.apply方法，返回优化后的模型opt_model
            opt_model = torch.compile(MySin.apply)
            # 遍历列表i，每个迭代中创建一个新的tensor x，并应用优化后的模型，计算损失和反向传播
            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = opt_model(x)
                loss = out.sum()
                loss.backward()
                # 生成当前x的梯度，并返回
                yield x.grad

        # 调用测试辅助函数，验证输出并重新编译，执行fn函数，期望输出次数为2，并断言unique_graphs的数量为5
        self.check_output_and_recompiles(
            fn, count=2, compiler_fn=make_compiler_fn(fullgraph=False)
        )
        # 断言统计结果中的unique_graphs数量为5
        self.assertEqual(counters["stats"]["unique_graphs"], 5)
    def test_custom_fn_compiled_fw_bw_graph_break(self):
        def fn():
            # 定义自定义函数 MySin，继承自 torch.autograd.Function
            class MySin(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    # 前向传播函数，计算正弦并保存输入张量 x
                    print("graph break")
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    # 反向传播函数，计算梯度并返回
                    print("graph break")
                    (x,) = ctx.saved_tensors
                    return gO * torch.cos(x)

            # 使用 MySin.apply 编译优化模型
            opt_model = torch.compile(MySin.apply)
            # 遍历多个长度，创建张量 x，并进行前向传播、反向传播计算
            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = opt_model(x)
                loss = out.sum()
                loss.backward()
                yield x.grad

        # 调用 check_output_and_recompiles 函数检查输出及重新编译情况
        self.check_output_and_recompiles(
            fn, count=[2, 6], compiler_fn=make_compiler_fn(fullgraph=False)
        )
        # 断言唯一图形计算次数
        self.assertEqual(counters["stats"]["unique_graphs"], 9)  # 3 fw, 6 bw

    def test_mismatch_fake_tensor_mode(self, dynamic_shape=False):
        """
        Repro the failure of training nanogpt with both compiled-autograd
        and _LazyGraphModule. Check https://github.com/pytorch/pytorch/pull/118981
        for more context.
        """
        # 定义张量 x 和 y，用于模拟编译和动态形状
        B = 8
        x = torch.rand(B, 16)
        y = torch.rand(B, 16, requires_grad=True)

        # 如果动态形状为真，则标记张量 x 和 y 为动态
        if dynamic_shape:
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(y, 0)

        def f():
            # 清空梯度，计算张量相加 out，并确保编译反向图时不触发错误
            y.grad = None
            out = x + y
            out.sum().backward()
            return out, y.grad

        # 调用 check_output_and_recompiles 函数检查输出及重新编译情况
        self.check_output_and_recompiles(f, compile_fn=True)

    def test_mismatch_fake_tensor_mode_dynamic_shape(self):
        # 调用 test_mismatch_fake_tensor_mode 函数并传入动态形状为真
        self.test_mismatch_fake_tensor_mode(dynamic_shape=True)

    def test_accumulate_grad_accuracy(self):
        def fn():
            # 定义模型，计算输出和损失，并计算梯度
            model = torch.nn.Sequential(
                torch.nn.Linear(2, 1, bias=False),
                torch.nn.Linear(1, 2, bias=False),
            )
            x = torch.randn(2, 2)

            out = model(x)
            loss = out.sum()
            torch.manual_seed(0)
            loss.backward()

            # 生成模型第一层和第二层的梯度
            yield model[0].weight.grad
            yield model[1].weight.grad

        # 调用 check_output_and_recompiles 函数检查输出及重新编译情况
        self.check_output_and_recompiles(fn, 1)

    def test_autograd_cpp_node(self):
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  // 定义静态成员变量，用于指示是否支持跟踪
  static constexpr bool is_traceable = true;

  // 前向传播函数的实现
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    // 前向传播直接返回输入张量 x
    return x;
  }

  // 反向传播函数的实现
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    // 反向传播时直接返回梯度输出 grad_output
    return grad_output;
  }
};

// 自定义的操作函数，调用了自定义的自动求导函数 CustomOpAutogradFunction 的 apply 方法
torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}

// 定义了一个 Torch 扩展库的注册块
TORCH_LIBRARY(test_autograd_cpp_node, m) {
    // 注册自定义操作函数 custom_op_backed_by_autograd_fn
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """

        # 使用 torch.utils.cpp_extension.load_inline 函数加载内联编写的 C++ 源码作为扩展模块
        module = torch.utils.cpp_extension.load_inline(
            name="test_autograd_cpp_node_id",  # 扩展模块的名称
            cpp_sources=cpp_source,  # C++ 源码
            functions="custom_op_backed_by_autograd_fn",  # 指定要加载的函数名称
            verbose=True,  # 输出详细的加载信息
        )

        # 定义同一个自动求导函数的生成器函数
        def same_autograd_fn():
            def fn():
                x = torch.ones(10, 10, requires_grad=True)  # 创建一个需要梯度的张量
                out = (
                    torch.ops.test_autograd_cpp_node_id.custom_op_backed_by_autograd_fn(
                        x
                    )  # 调用 C++ 扩展中的自定义自动求导函数
                )
                loss = out.sum()  # 计算输出的和作为损失
                loss.backward()  # 反向传播计算梯度
                yield x.grad  # 返回梯度

            yield from fn()  # 编译生成器函数
            yield from fn()  # 重复使用生成器函数
            yield from fn()  # 重复使用生成器函数
            yield from fn()  # 重复使用生成器函数

        self.check_output_and_recompiles(same_autograd_fn, 1)  # 检查输出并重新编译

        # 定义不同自动求导函数的生成器函数
        def different_autograd_fn():
            def fn(op):
                x = torch.ones(10, 10, requires_grad=True)  # 创建一个需要梯度的张量
                out = op(x)  # 调用传入的自动求导函数
                loss = out.sum()  # 计算输出的和作为损失
                loss.backward()  # 反向传播计算梯度
                yield x.grad  # 返回梯度

            op1 = torch.ops.test_autograd_cpp_node_id.custom_op_backed_by_autograd_fn  # 第一个自定义自动求导函数
            op2 = torch.ops.test_autograd_cpp_node_id.custom_op_backed_by_autograd_fn2  # 第二个自定义自动求导函数
            yield from fn(op1)  # 编译使用第一个自动求导函数
            yield from fn(op2)  # 编译使用第二个自动求导函数
            yield from fn(op1)  # 重复使用第一个自动求导函数
            yield from fn(op2)  # 重复使用第二个自动求导函数

        self.check_output_and_recompiles(different_autograd_fn, 2)  # 检查输出并重新编译

    # 测试保存节点的自动求导函数
    def test_autograd_cpp_node_saved(self):
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      const torch::Tensor& y,
      const torch::Tensor& fixed) {
    ctx->save_for_backward({x, y});  // 保存张量 x 和 y 作为反向传播时需要的变量
    ctx->saved_data["fixed_tensor"] = fixed;  // 保存张量 fixed 到上下文中的 saved_data
    ctx->saved_data["bool"] = true;  // 保存布尔值到 saved_data
    ctx->saved_data["int"] = 1;  // 保存整数到 saved_data
    c10::List<std::string> list({"string"});  // 创建包含一个字符串的列表
    ctx->saved_data["list"] = std::move(list);  // 保存列表到 saved_data
    c10::Dict<std::string, double> dict;  // 创建一个字符串到双精度浮点数的字典
    dict.insert("string", 1.0);  // 向字典中插入键值对
    ctx->saved_data["dict"] = std::move(dict);  // 保存字典到 saved_data
    return x;  // 返回输入张量 x
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    const auto& saved_variables = ctx->get_saved_variables();  // 获取保存的张量变量列表
    assert(saved_variables.size() == 2);  // 断言保存的张量变量数量为 2
    torch::Tensor x = saved_variables[0];  // 获取保存的第一个张量变量
    torch::Tensor y = saved_variables[1];  // 获取保存的第二个张量变量
    torch::Tensor fixed = ctx->saved_data["fixed_tensor"].toTensor();  // 获取保存的张量 fixed
    assert(ctx->saved_data["bool"].isBool());  // 断言 saved_data 中的 bool 值为布尔类型
    int i = ctx->saved_data["int"].toInt();  // 获取 saved_data 中的整数值
    c10::List<c10::IValue> list = ctx->saved_data["list"].toList();  // 获取 saved_data 中的列表
    assert(list.size() == 1);  // 断言列表大小为 1
    assert(list.get(0).toStringRef() == "string");  // 断言列表中的字符串值为 "string"
    c10::Dict<c10::IValue, c10::IValue> dict = ctx->saved_data["dict"].toGenericDict();  // 获取 saved_data 中的字典
    # 断言字典的大小是否为1，用于检查字典的预期大小
    assert(dict.size() == 1);

    # 断言字典中键为"string"的值是否为1.0，用于验证特定键的预期值
    assert(dict.at("string") == 1.0);

    # 创建一个包含3个元素的自动求导变量列表
    torch::autograd::variable_list grad_inputs(3);

    # 将四个表达式相加并赋值给列表中的第一个元素，表达式包括变量 x、y，张量 fixed 的总和以及变量 i
    grad_inputs[0] = x + y + torch::sum(fixed) + i;

    # 返回填充了梯度输入的变量列表
    return grad_inputs;
        """

        // 加载内联定义的 C++ 源码作为扩展模块，并定义自定义的 Torch 操作函数
        module = torch.utils.cpp_extension.load_inline(
            name="test_autograd_cpp_node_saved",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",  // 加载定义的自定义操作函数
            verbose=True,
        )

        // 定义一个 Python 函数 fn，用于测试加载的 C++ 扩展模块功能
        def fn():
            fixed = torch.ones(2, 2)
            // 遍历不同的尺寸值 i 进行测试
            for i in [10, 100, 10, 20, 10]:
                // 创建一个大小为 i x i 的张量 x，并启用梯度计算
                x = torch.ones(i, i, requires_grad=True)
                // 创建一个大小为 i x i 的随机张量 y
                y = torch.randn(i, i)
                // 调用自定义的 Torch 操作函数 custom_op_backed_by_autograd_fn，传入参数 x, y, fixed
                out = torch.ops.test_autograd_cpp_node_saved.custom_op_backed_by_autograd_fn(
                    x, y, fixed
                )
                // 计算输出张量 out 的元素和作为损失值
                loss = out.sum()
                // 对损失值进行反向传播
                loss.backward()
                // 生成张量 x 的梯度，并使用 yield 返回
                yield x.grad

        // 调用测试函数 fn，并检查输出结果及重新编译模块
        self.check_output_and_recompiles(fn, 2)

    // 定义测试函数 test_autograd_cpp_node_saved_dynamic
    def test_autograd_cpp_node_saved_dynamic(self):
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  static constexpr bool is_traceable = true;

  // 前向传播函数定义
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    // 保存张量 x 到 AutogradContext 中
    ctx->save_for_backward({x});
    // 将张量 x 按照指定维度展开，并保存到 saved_data 字典中
    ctx->saved_data["dynamic"] = x.view(-1);
    // 返回张量 x 作为前向传播的结果
    return x;
  }

  // 反向传播函数定义
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    // 获取保存在 AutogradContext 中的变量
    const auto& saved_variables = ctx->get_saved_variables();
    // 断言保存的变量数量为 1
    assert(saved_variables.size() == 1);
    // 从保存的变量中取出张量 x
    torch::Tensor x = saved_variables[0];
    // 从 saved_data 字典中获取保存的动态数据，并转换为张量 z
    torch::Tensor z = ctx->saved_data["dynamic"].toTensor();

    // 创建一个梯度输入列表，长度为 1
    torch::autograd::variable_list grad_inputs(1);
    // 计算梯度输入，为张量 x 加上动态数据 z 的和
    grad_inputs[0] = x + torch::sum(z);
    // 返回梯度输入列表
    return grad_inputs;
  }
};

// 定义自定义的 Torch 操作函数 custom_op_backed_by_autograd_fn，接收一个张量 x 作为参数
torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x) {
  // 调用自定义的 AutogradFunction 中的 apply 方法，传入张量 x
  return CustomOpAutogradFunction::apply(x);
}

// 注册自定义 Torch 操作函数到指定的 Torch 库中
TORCH_LIBRARY(test_autograd_cpp_node_saved_dynamic, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}
        """

        // 加载内联定义的 C++ 源码作为扩展模块，并定义自定义的 Torch 操作函数
        module = torch.utils.cpp_extension.load_inline(
            name="test_autograd_cpp_node_saved_dynamic",
            cpp_sources=cpp_source,
            functions="custom_op_backed_by_autograd_fn",  // 加载定义的自定义操作函数
            verbose=True,
        )

        // 定义一个 Python 函数 fn，用于测试加载的 C++ 扩展模块功能
        def fn():
            // 遍历不同的尺寸值 i 进行测试
            for i in [10, 100, 10, 20, 10]:
                // 创建一个大小为 i x i 的张量 x，并启用梯度计算
                x = torch.ones(i, i, requires_grad=True)
                // 调用自定义的 Torch 操作函数 custom_op_backed_by_autograd_fn，传入参数 x
                out = torch.ops.test_autograd_cpp_node_saved_dynamic.custom_op_backed_by_autograd_fn(
                    x
                )
                // 计算输出张量 out 的元素和作为损失值
                loss = out.sum()
                // 对损失值进行反向传播
                loss.backward()
                // 生成张量 x 的梯度，并使用 yield 返回
                yield x.grad

        // 调用测试函数 fn，并检查输出结果及重新编译模块
        self.check_output_and_recompiles(fn, 5)
    # 定义测试方法 test_autograd_cpp_node_data_dependent，用于测试自动求导相关的 C++ 节点数据依赖性
    def test_autograd_cpp_node_data_dependent(self):
        # C++ 源代码作为字符串赋值给变量 cpp_source
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  // 定义静态常量 is_traceable，表示该自定义操作可追踪
  static constexpr bool is_traceable = true;
  // 定义静态整数 iteration，用于追踪执行次数
  static int iteration;

  // 前向传播函数
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x,
      const torch::Tensor& y) {
    // 保存张量 x 和 y 到上下文中，以便后向传播使用
    ctx->save_for_backward({x, y});
    // 保存布尔类型数据到上下文中
    ctx->saved_data["bool"] = true;
    // 保存整数数据 1 到上下文中
    ctx->saved_data["int"] = 1;

    // 根据迭代次数进行不同的处理
    switch (iteration) {
        case 0: {
            // 迭代次数为 0 时不进行特定操作
            break;
        }
        case 1: {
            // 迭代次数为 1 时设置 saved_data 中的 forces_recompile
            ctx->saved_data["forces_recompile"] = iteration;
            break;
        }
        case 2: {
            // 迭代次数为 2 时设置不生成梯度信息
            ctx->set_materialize_grads(false);
            break;
        }
        case 3: {
            // 迭代次数为 3 时不进行特定操作，即重用之前的结果
            break;
        }
        default: {
            // 抛出运行时异常，表示不期望的迭代次数
            throw std::runtime_error("unexpected iteration");
        }
    }
    // 迭代次数加一
    iteration++;
    // 返回输入张量列表，即返回 x 和 y
    return {x, y};
  }

  // 后向传播函数
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    // 获取保存的变量列表
    const auto& saved_variables = ctx->get_saved_variables();
    // 断言保存的变量数为 2
    assert(saved_variables.size() == 2);
    // 获取保存的张量 x 和 y
    torch::Tensor x = saved_variables[0];
    torch::Tensor y = saved_variables[1];
    // 断言 saved_data 中的 bool 和 int 数据类型分别为布尔型和整型
    assert(ctx->saved_data["bool"].isBool());
    assert(ctx->saved_data["int"].isInt());
    // 将 saved_data 中的 int 数据转换为整数
    int i = ctx->saved_data["int"].toInt();

    // 初始化梯度输入列表，包含两个元素
    torch::autograd::variable_list grad_inputs(2);
    // 计算梯度输入的第一个元素，即 x + y + i
    grad_inputs[0] = x + y + i;
    // 返回梯度输入列表
    return grad_inputs;
  }
};

// 初始化静态整数 iteration
int CustomOpAutogradFunction::iteration = 0;

// 定义自定义操作的包装函数
torch::autograd::variable_list custom_op_backed_by_autograd_fn(const torch::Tensor& x, const torch::Tensor& y) {
  // 调用自定义操作函数的 apply 方法，返回前向传播的结果
  return CustomOpAutogradFunction::apply(x, y);
}

// 重置迭代计数器的函数
void reset() {
    // 将静态整数 iteration 重置为 0
    CustomOpAutogradFunction::iteration = 0;
}

// 定义 Torch 库的扩展接口
TORCH_LIBRARY(test_autograd_cpp_node_data_dependent, m) {
    // 注册自定义操作的包装函数 custom_op_backed_by_autograd_fn
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
    // 注册重置迭代计数器的函数 reset
    m.def("reset", reset);
}
    def test_free_activation_memory(self):
        self.assertTrue(torch.cuda.memory_allocated() == 0)

        # Use an op to check that the memory is freed by the time the op is executed
        def assertion_impl(to_clone):
            mem_allocated = torch.cuda.memory_allocated()
            self.assertTrue(
                mem_allocated < 4000000, "activations should have been freed"
            )
            return to_clone.clone()

        with torch.library._scoped_library("test_compiled_autograd", "FRAGMENT") as lib:
            # Define a new custom operator "assertion_op" in the "test_compiled_autograd" library
            lib.define(
                "assertion_op(Tensor x) -> Tensor", tags=(torch.Tag.pt2_compliant_tag,)
            )
            # Implement the "assertion_op" function using assertion_impl, running on CPU
            lib.impl("assertion_op", assertion_impl, "CPU")
            # Implement a default implementation of "assertion_op" using lambda for the "Meta" backend

            # Create a graph using symbolic tracing to prepare for compilation
            def forward(activations):
                # Perform addition operation on the first activation tensor
                add = activations[0] + 1
                # Move the result to CPU
                out = add.cpu()
                # Apply the custom assertion operation "assertion_op" to out
                cloned_out = torch.ops.test_compiled_autograd.assertion_op(out)
                return (cloned_out,)

            # Symbolically trace the forward function to create a graph module
            gm = torch.fx.symbolic_trace(forward)
            # Allow inputs to be stolen (moved) during optimization
            torch._dynamo.utils.set_locals_to_steal(gm, ["activations"])
            # Compile the graph module into a callable function
            compiled_fn = torch.compile(gm)

            # Allocate a tensor on CUDA device with at least 4,000,000 bytes (1,000,000 * 4 bytes)
            activations = [torch.ones(1000000, dtype=torch.float32, device="cuda")]
            self.assertTrue(torch.cuda.memory_allocated() > 4000000)

            # Execute the compiled function with activations as input
            out = compiled_fn(activations)
            # Ensure all activations have been consumed by the function
            self.assertTrue(len(activations) == 0)

    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    def test_free_activation_memory_subclass(self):
        # 检查当AOT输入具有子类时，会导致不同的运行时包装情况
        self.assertTrue(torch.cuda.memory_allocated() == 0)

        # 使用一个操作来检查在操作执行时内存是否被释放
        def assertion_impl(to_clone):
            # 获取当前CUDA已分配的内存量
            mem_allocated = torch.cuda.memory_allocated()
            # 断言：应该有一些激活被释放了，因此内存量应小于1200000
            self.assertTrue(
                mem_allocated < 1200000, "some activations should have been freed"
            )
            # 断言：目前子类似乎没有在感应器中被释放，因此内存量应大于800000
            self.assertTrue(
                mem_allocated > 800000,
                "currently subclasses don't seem to be freed in inductor",
            )
            # 返回输入的克隆
            return to_clone.clone()

        # 使用torch.library._scoped_library上下文管理器来定义和实现操作
        with torch.library._scoped_library("test_compiled_autograd", "FRAGMENT") as lib:
            # 定义带有特定标签的断言操作
            lib.define(
                "assertion_op(Tensor x) -> Tensor", tags=(torch.Tag.pt2_compliant_tag,)
            )
            # 在不同的上下文（CPU、Meta、NestedTensor）中实现断言操作
            lib.impl("assertion_op", assertion_impl, "CPU")
            lib.impl("assertion_op", lambda x: x.clone(), "Meta")
            lib.impl("assertion_op", lambda x: x.clone(), "NestedTensor")

            # 定义一个函数fn，输入是一个元组inputs，其中第二个元素y将在CPU上执行
            def fn(inputs):
                _, y = inputs
                out = y.cpu()
                # 调用torch.ops.test_compiled_autograd.assertion_op操作
                cloned_out = torch.ops.test_compiled_autograd.assertion_op(out)
                return cloned_out

            # 对函数fn进行符号化跟踪，生成图模型gm
            gm = torch.fx.symbolic_trace(fn)
            # 设置要窃取的局部变量为["inputs"]
            torch._dynamo.utils.set_locals_to_steal(gm, ["inputs"])
            # 编译图模型gm，生成compiled_fn
            compiled_fn = torch.compile(gm)

            # 导入jagged_from_list函数
            from torch.nested._internal.nested_tensor import jagged_from_list

            # 创建激活张量列表activations
            activations = [
                jagged_from_list(
                    [
                        torch.ones((1, 100000), device="cuda"),  # 400,000 bytes
                        torch.ones((1, 100000), device="cuda"),  # 400,000 bytes
                    ],
                    None,
                )[0],  # NestedTensor
                torch.ones((1, 100000), device="cuda"),  # 400,000 bytes
            ]
            # 断言：当前CUDA已分配的内存量大于1200000，即1,200,000字节
            self.assertTrue(torch.cuda.memory_allocated() > 1200000)

            # 执行编译后的函数compiled_fn，并将activations作为输入
            out = compiled_fn(activations)
            # 断言：activations列表已为空
            self.assertTrue(len(activations) == 0)
    def test_callback_graph_break_throws_error(self):
        called = [0]

        def callback_final():
            called[0] += 1  # 定义一个回调函数，每次调用增加计数器值

        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return input  # 前向传播函数，直接返回输入

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, grad):
                torch.autograd.Variable._execution_engine.queue_callback(callback_final)  # 将回调函数添加到执行队列中
                torch._dynamo.graph_break()  # 在反向传播过程中调用图断裂函数
                return grad  # 返回梯度

        a = torch.rand((3, 3), requires_grad=True)  # 创建一个需要梯度的随机张量
        with self.assertRaisesRegex(
            AssertionError,
            "only supported when Compiled Autograd is enabled with fullgraph=True",
        ):
            with compiled_autograd.enable(make_compiler_fn(fullgraph=False)):  # 启用编译自动求导，并配置不完整图
                b = MyFunc.apply(a)  # 应用自定义的函数 MyFunc
                b.sum().backward()  # 对结果求和并进行反向传播

    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    def test_cudagraphs_cpu_division(self):
        from torch._dynamo.testing import reduce_to_scalar_loss

        model = torch.nn.Linear(10, 10, dtype=torch.float16).cuda()  # 创建一个在 CUDA 上运行的线性模型
        inputs = torch.randn(10, 10, dtype=torch.float16).cuda()  # 创建 CUDA 上的随机输入张量
        out = model(inputs)  # 使用模型进行前向传播
        loss = reduce_to_scalar_loss(out)  # 计算输出的标量损失

        stderr_msgs = io.StringIO()  # 创建一个字符串IO对象，用于捕获stderr输出
        with mock.patch("sys.stderr", stderr_msgs), compiled_autograd.enable(
            compiler_fn
        ):
            torch._inductor.config.triton.cudagraphs = True  # 启用 CUDA 图模式
            loss.backward()  # 对损失进行反向传播
            torch._inductor.config.triton.cudagraphs = False  # 关闭 CUDA 图模式

        self.assertFalse("skipping cudagraphs" in stderr_msgs.getvalue())  # 检查stderr消息中是否包含跳过CUDA图模式的信息

    def test_cudagraphs_cpu_graph(self):
        from torch._dynamo.testing import reduce_to_scalar_loss

        model = torch.nn.Linear(10, 10, dtype=torch.float16)  # 创建一个线性模型
        inputs = torch.randn(10, 10, dtype=torch.float16)  # 创建随机输入张量
        out = model(inputs)  # 使用模型进行前向传播
        loss = reduce_to_scalar_loss(out)  # 计算输出的标量损失

        with compiled_autograd.enable(compiler_fn):  # 启用编译自动求导
            torch._inductor.config.triton.cudagraphs = True  # 启用 CUDA 图模式
            loss.backward()  # 对损失进行反向传播
            torch._inductor.config.triton.cudagraphs = False  # 关闭 CUDA 图模式

        self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)  # 检查跳过的 CUDA 图模式计数器

    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    # 定义一个测试方法，用于测试CUDA图形的Scaled Dot-Product Attention功能
    def test_cudagraphs_sdpa(self):
        # 创建一个随机的查询张量，在CUDA上进行操作，需要梯度计算
        query = torch.rand(
            32, 8, 128, 64, dtype=torch.float16, device="cuda", requires_grad=True
        )
        # 创建一个随机的键张量，在CUDA上进行操作
        key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        # 创建一个随机的值张量，在CUDA上进行操作
        value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device="cuda")
        # 使用函数式API执行Scaled Dot-Product Attention操作
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)

        # 使用配置管理器进行设置，开启编译自动求导功能
        with config.patch(compiled_autograd=True), inductor_config.patch(
            "triton.cudagraphs", True
        ):
            # 编译计算梯度反向传播的函数
            opt_bwd = torch.compile(lambda: out.sum().backward())
            opt_bwd()

        # 断言编译自动求导捕获的次数为1
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        # 断言CUDA图形的跳过次数为0
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

    # 如果没有CUDA，则跳过此测试
    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    # 定义测试方法，用于测试CPU标量在Python自定义操作中的使用情况
    def test_cudagraphs_cpu_scalar_used_in_python_custom_op(self):
        # 定义一个自定义的PyTorch函数类
        class MyFn(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，保存输入张量和一个CPU张量，并返回求和结果
            def forward(ctx, x):
                # 创建一个CPU张量
                cpu_tensor = torch.tensor(5)
                # 保存张量用于反向传播
                ctx.save_for_backward(x, cpu_tensor)  # visible to c++/autograd
                # 设置上下文中的CPU标量，用于C++/autograd不透明
                ctx.cpu_scalar = 5  # opaque to c++/autograd
                return x.sum()

            @staticmethod
            # 反向传播函数，根据保存的张量计算梯度
            def backward(ctx, gO):
                x, cpu_tensor = ctx.saved_tensors
                # 扩展梯度
                expand = gO * torch.ones_like(x)
                # 返回计算的梯度
                return expand * cpu_tensor * ctx.cpu_scalar

        # 创建一个随机张量，在CUDA上进行操作，需要梯度计算
        x = torch.randn(10, requires_grad=True, device="cuda")
        # 应用自定义的函数
        out = MyFn.apply(x)
        
        # 使用配置管理器进行设置，开启编译自动求导功能
        with config.patch(compiled_autograd=True), inductor_config.patch(
            "triton.cudagraphs", True
        ):
            # 编译计算梯度反向传播的函数
            opt_bwd = torch.compile(lambda: out.backward())
            opt_bwd()

        # 断言编译自动求导捕获的次数为1
        self.assertEqual(counters["compiled_autograd"]["captures"], 1)
        # 断言CUDA图形的跳过次数为1，因为不确定CPU标量是否仅在ATen/prim操作中使用
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

    # 如果没有CUDA，则跳过此测试
    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    # 定义测试方法，用于测试CPU标量在C++自定义操作中的使用情况
    def test_cudagraphs_cpu_scalar_used_in_cpp_custom_op(self):
        # 定义一个C++源码字符串，暂未提供
# 定义一个自定义的 Torch 自动微分函数 CustomOpAutogradFunction，继承自 torch::autograd::Function<CustomOpAutogradFunction>
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  // 声明静态常量 is_traceable 为 true
  static constexpr bool is_traceable = true;

  // 前向传播函数的实现，接收 AutogradContext 指针和输入张量 x
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    // 创建一个 CPU 上的张量 cpu_tensor，值为 1
    const auto& cpu_tensor = torch::tensor(1);
    // 将输入 x 和 cpu_tensor 保存在 AutogradContext 中，以便反向传播使用
    ctx->save_for_backward({x, cpu_tensor});
    // 在 saved_data 中保存名为 "cpu_scalar" 的标量值 1
    ctx->saved_data["cpu_scalar"] = 1;
    // 前向传播直接返回输入张量 x
    return x;
  }

  // 反向传播函数的实现，接收 AutogradContext 指针和梯度输出 grad_output 的变量列表
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    // 获取之前保存的所有变量
    const auto& saved_variables = ctx->get_saved_variables();
    // 断言保存的变量数量为 2
    assert(saved_variables.size() == 2);
    // 分别获取保存的张量 x 和 cpu_tensor
    torch::Tensor x = saved_variables[0];
    torch::Tensor cpu_tensor = saved_variables[1];
    // 从 saved_data 中获取名为 "cpu_scalar" 的整数值
    int cpu_scalar = ctx->saved_data["cpu_scalar"].toInt();
    // 计算梯度的扩展部分，即 grad_output 乘以与 x 形状相同的全 1 张量
    auto expand = grad_output[0] * torch::ones_like(x);
    // 创建一个变量列表 grad_inputs，包含一个元素
    torch::autograd::variable_list grad_inputs(1);
    // 将梯度计算结果存入 grad_inputs 中，计算公式为 expand * cpu_tensor * cpu_scalar
    grad_inputs[0] = expand * cpu_tensor * cpu_scalar;  // autograd engine asserts that tensors are on same device
    // 返回梯度输入列表
    return grad_inputs;
  }
};

// 定义一个函数 custom_op_backed_by_autograd_fn，调用 CustomOpAutogradFunction 的 apply 方法
torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x) {
  return CustomOpAutogradFunction::apply(x);
}

// 注册自定义操作的 Torch 库，包含 custom_op_backed_by_autograd_fn 函数
TORCH_LIBRARY(test_cudagraphs_cpu_scalar_used_in_cpp_custom_op, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}

// 加载内联的 C++ 代码，定义名称为 "test_cudagraphs_cpu_scalar_used_in_cpp_custom_op" 的 Torch 扩展模块
module = torch.utils.cpp_extension.load_inline(
    name="test_cudagraphs_cpu_scalar_used_in_cpp_custom_op",
    cpp_sources=cpp_source,
    functions="custom_op_backed_by_autograd_fn",
    verbose=True,
)

// 创建一个随机数张量 x，形状为 (2, 2)，在 CUDA 设备上，需要计算梯度
x = torch.randn(2, 2, requires_grad=True, device="cuda")

// 进入上下文，开启编译后自动微分的功能，确保编译成功
with config.patch(compiled_autograd=True), inductor_config.patch(
    "triton.cudagraphs", True
):
    // 调用自定义操作 custom_op_backed_by_autograd_fn，并传入张量 x
    out = torch.ops.test_cudagraphs_cpu_scalar_used_in_cpp_custom_op.custom_op_backed_by_autograd_fn(
        x
    )
    // 编译自动生成的反向传播函数，计算对 out 求和后的梯度
    opt_bwd = torch.compile(lambda: out.sum().backward())
    // 执行编译后的反向传播函数
    opt_bwd()

// 断言编译自动微分捕获的次数为 1
self.assertEqual(counters["compiled_autograd"]["captures"], 1)
// 断言导入器跳过 cudagraph 的次数为 0
self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)
    # 定义一个测试方法，验证详细日志记录图形
    def test_verbose_logs_graph(self):
        # 设置 Torch 的日志记录，启用编译自动微分详细信息
        torch._logging.set_logs(compiled_autograd_verbose=True)

        # 定义内部函数 fn，用于构建简单的神经网络模型并执行反向传播
        def fn():
            # 创建一个包含两个线性层和 ReLU 激活函数的序列模型
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            # 生成一个随机输入张量 x
            x = torch.randn([2, 4])
            # 执行模型前向传播，并对结果求和
            result = model(x).sum()
            # 执行反向传播
            result.backward()
            # 生成器函数，依次返回模型第一层权重和偏置的梯度，以及第二层权重和偏置的梯度
            yield model[0].weight.grad
            yield model[0].bias.grad
            yield model[2].weight.grad
            yield model[2].bias.grad

        # 将编译自动微分模块的日志及上下文保存到 logs 和 ctx 变量中
        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )
        # 进入日志记录上下文
        with ctx():
            # 调用 self.check_output_and_recompiles 函数，验证输出并重新编译
            self.check_output_and_recompiles(fn)

        # 预期的日志消息列表，用于验证日志内容
        expected_logs = [
            "SumBackward0 (NodeCall 1)",
            "ReluBackward0 (NodeCall 2)",
            "AddmmBackward0 (NodeCall 3)",
            "TBackward0 (NodeCall 4)",
            "torch::autograd::AccumulateGrad (NodeCall 5)",
            "ReluBackward0 (NodeCall 6)",
            "AddmmBackward0 (NodeCall 7)",
            "TBackward0 (NodeCall 8)",
            "torch::autograd::AccumulateGrad (NodeCall 9)",
            "torch::autograd::AccumulateGrad (NodeCall 10)",
            "torch::autograd::AccumulateGrad (NodeCall 11)",
        ]

        # 使用断言验证预期日志消息是否存在于实际日志中
        self.assertEqual(
            sum(1 for e in expected_logs if e in logs.getvalue()), len(expected_logs)
        )
    def test_verbose_logs_cpp(self):
        # 设置 Torch 的编译自动求导模块的详细日志
        torch._logging.set_logs(compiled_autograd_verbose=True)

        def fn():
            # 创建一个包含线性层和ReLU激活函数的神经网络模型
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            # 对于不同长度的输入数据进行测试
            for i in [10, 11, 12]:
                # 模型梯度归零
                model.zero_grad()
                # 生成随机输入张量
                x = torch.randn([i, 4])
                # 计算模型输出的和，并进行反向传播
                result = model(x).sum()
                result.backward()
                # 返回第一个线性层的权重梯度和偏置梯度
                yield model[0].weight.grad
                yield model[0].bias.grad
                # 返回第二个线性层的权重梯度和偏置梯度
                yield model[2].weight.grad
                yield model[2].bias.grad

        # 将编译自动求导的详细日志转换为字符串
        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )
        # 进入上下文管理器，检查输出并重新编译函数两次
        with ctx():
            self.check_output_and_recompiles(fn, count=2)

        # 第一组模式，用于匹配详细日志中的缓存未命中情况
        patterns1 = [
            r".*Cache miss due to new autograd node: torch::autograd::GraphRoot \(NodeCall 0\) with key size (\d+), "
            r"previous key sizes=\[\]\n",
        ]

        # 第二组模式，用于匹配详细日志中形状变化导致的缓存未命中情况
        patterns2 = [
            r".*Cache miss due to changed shapes: marking size idx (\d+) of torch::autograd::GraphRoot \(NodeCall 0\) as dynamic\n",
            r".*Cache miss due to changed shapes: marking size idx (\d+) of SumBackward0 \(NodeCall 1\) as dynamic\n",
            r".*Cache miss due to changed shapes: marking size idx (\d+) of SumBackward0 \(NodeCall 1\) as dynamic\n",
            r".*Cache miss due to changed shapes: marking size idx (\d+) of ReluBackward0 \(NodeCall 2\) as dynamic\n",
            r".*Cache miss due to changed shapes: marking size idx (\d+) of AddmmBackward0 \(NodeCall 3\) as dynamic\n",
            r".*Cache miss due to changed shapes: marking size idx (\d+) of torch::autograd::AccumulateGrad "
            r"\(NodeCall 5\) as dynamic\n",
            r".*Cache miss due to changed shapes: marking size idx (\d+) of ReluBackward0 \(NodeCall 6\) as dynamic\n",
        ]

        # 获取所有日志的字符串
        all_logs = logs.getvalue()

        # 编译第一组模式并进行匹配
        pattern1 = r"".join(patterns1)
        matches1 = re.findall(pattern1, all_logs)
        # 断言第一组模式的匹配数为1
        self.assertEqual(len(matches1), 1)
        # 断言匹配结果为字符串类型
        assert isinstance(
            matches1[0], str
        )  # for a single match: matches1=['match'], for multiple matches: matches1=[('match1', 'match2')]...
        # 断言第一组模式的匹配数与模式列表长度相等
        self.assertEqual(len(matches1), len(patterns1))

        # 编译第二组模式并进行匹配
        pattern2 = r"".join(patterns2)
        matches2 = re.findall(pattern2, all_logs)
        # 断言第二组模式的匹配数为1
        self.assertEqual(len(matches2), 1)
        # 断言第二组模式中每个模式的匹配数与模式列表长度相等
        self.assertEqual(len(matches2[0]), len(patterns2))
    # 定义一个测试方法，测试详细日志标志位
    def test_snapshot_verbose_logs_flag(self):
        # 定义内部函数fn，用于构建简单的神经网络模型并执行反向传播
        def fn():
            # 创建一个包含两个线性层和ReLU激活函数的序列模型
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            # 生成一个形状为[2, 4]的随机张量作为输入
            x = torch.randn([2, 4])
            # 将输入x传递给模型并计算输出的总和
            result = model(x).sum()
            # 对模型输出的结果进行反向传播
            result.backward()
            # 依次生成第一个线性层的权重梯度、偏置梯度、第二个线性层的权重梯度、偏置梯度
            yield model[0].weight.grad
            yield model[0].bias.grad
            yield model[2].weight.grad
            yield model[2].bias.grad

        # 调用logs_to_string函数获取日志输出流和上下文管理器
        logs, ctx = logs_to_string(
            torch._dynamo.compiled_autograd.__name__, "compiled_autograd_verbose"
        )
        # 进入日志记录的上下文
        with ctx():
            # 启用编译后自动求导，并使用指定的编译器函数
            with compiled_autograd.enable(compiler_fn):
                # 设置torch._logging的编译后自动求导详细日志为True
                torch._logging.set_logs(compiled_autograd_verbose=True)
                # 调用内部函数fn，执行模型操作

        # 定义预期的未期望日志条目列表
        unexpected_logs = [
            "SumBackward0 (NodeCall 1)",
            "ReluBackward0 (NodeCall 2)",
            "AddmmBackward0 (NodeCall 3)",
            "TBackward0 (NodeCall 4)",
            "torch::autograd::AccumulateGrad (NodeCall 5)",
            "ReluBackward0 (NodeCall 6)",
            "AddmmBackward0 (NodeCall 7)",
            "TBackward0 (NodeCall 8)",
            "torch::autograd::AccumulateGrad (NodeCall 9)",
            "torch::autograd::AccumulateGrad (NodeCall 10)",
            "torch::autograd::AccumulateGrad (NodeCall 11)",
        ]

        # 使用self.assertEqual断言，检查未期望日志条目在日志中出现的次数是否为0
        self.assertEqual(sum(1 for e in unexpected_logs if e in logs.getvalue()), 0)
# 根据给定的模块名加载测试模块
def load_test_module(name):
    # 获取当前文件的绝对路径的父目录的父目录作为测试目录
    testdir = Path(__file__).absolute().parent.parent
    # 使用 mock.patch 将测试目录添加到 sys.path 中
    with mock.patch("sys.path", [*sys.path, str(testdir)]):
        # 使用 SourceFileLoader 加载指定名字的模块
        return SourceFileLoader(
            name, str(testdir / f"{name.replace('.', '/')}.py")
        ).load_module()


# 创建一个装饰器函数，用于包装测试函数
def make_wrapped(fn, fullgraph):
    @functools.wraps(fn)
    def wrapped(self):
        # 重置 torch._dynamo 状态
        torch._dynamo.reset()
        # 启用编译后自动求导，并传入完整图形参数
        with compiled_autograd.enable(make_compiler_fn(fullgraph=fullgraph)):
            # 调用原始函数 fn
            out = fn(self)

        return out

    return wrapped


# 对原始类进行包装，返回新的带编译自动求导的类
def wrap_test_class(orig_cls):
    # 拷贝原始类的字典属性
    dct = orig_cls.__dict__.copy()
    for name in list(dct.keys()):
        fn = dct[name]
        # 如果属性不是可调用的函数，继续下一个循环
        if not callable(fn):
            continue
        # 如果函数名符合已知失败的测试用例的正则表达式或在已知失败的测试集合中
        elif known_failures_re.match(name) or name in known_failing_tests:
            # 将该测试函数标记为预期失败
            dct[name] = unittest.expectedFailure
        # 如果函数名以 "test_" 开头
        elif name.startswith("test_"):
            # 根据函数名是否在已知图形断裂测试的集合中来决定是否包装该函数
            fullgraph = name not in known_graph_breaks_tests
            dct[name] = make_wrapped(fn, fullgraph)

    # 创建新的类，继承自原始类，但使用新的字典属性 dct
    cls = type(
        orig_cls.__name__ + "WithCompiledAutograd",
        orig_cls.__bases__,
        dct,
    )
    # 设置新类的文件属性为当前文件的文件名
    cls.__file__ = __file__
    return cls


# 已知会导致图形断裂的测试用例集合
known_graph_breaks_tests = {
    "test_hook_none",  # 使用钩子中的断言
    "test_post_accumulate_grad_hook_e2e",  # 手动破坏 optim.Adam
    "test_tensor_hooks_inplace",  # 使用钩子中的断言
    "test_tensor_hooks_inplace_over_view",  # 使用钩子中的断言
    "test_grad_fn_prehooks",  # 使用钩子中的断言
    "test_grad_fn_prehooks_multiple_outputs",  # 使用钩子中的断言
    "test_grad_fn_prehooks_remove_hooks",  # 在钩子中使用 handle.remove()
    "test_tensor_hooks_inplace_multiple_outputs",  # 使用钩子中的断言
    "test_hooks",  # 使用钩子中的断言
    "test_accumulate_grad_posthooks_can_observe_tensor_prehook",  # allclose
}

# 已知暂不支持的测试用例类型的正则表达式
known_failures_re = re.compile(
    r"^test_(sparse|profiler|gradcheck|checkpoint|named_tensor)"
)

# 需要进一步调查的已知失败测试用例集合
known_failing_tests = {
    "test_current_graph_task_execution_order",  # torch._dynamo.exc.TorchRuntimeError: Failed running call_function <
    "test_input_buffer_accum",  # RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
    "test_graph_save_on_cpu_cuda",  # AssertionError: 0 not greater than 0
    "test_graph_save_on_cpu",  # torch._dynamo.exc.BackendCompilerFailed: backend='inner_compiler' raised:
    "test_reentrant_with_leaf_variable_hook",  # torch._dynamo.exc.Unsupported: inline in skipfiles: RemovableHandle.
    "test_reentrant_with_non_leaf_variable_hook",  # torch._dynamo.exc.Unsupported: inline in skipfiles: RemovableHan
    "test_saved_variable_saved_original_inplace_detach",  # AssertionError: RuntimeError not raised
    "test_saving_variable_to_disk",  # Cannot call numel() on tensor with symbolic sizes/strides
    "test_setitem_mask",  # torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode: It appears that you're
}
    "test_wrapped_number_saved_variable_hooks",  # RuntimeError: this hook should not be called
    # 测试函数名称："test_wrapped_number_saved_variable_hooks"
    # 异常类型：RuntimeError
    # 异常信息："this hook should not be called"
    
    "test_save_tensor_hook_version_counter_not_shared",  # raise UnsupportedInputs
    # 测试函数名称："test_save_tensor_hook_version_counter_not_shared"
    # 异常类型：UnsupportedInputs
    # 异常信息："raise UnsupportedInputs"
    
    "test_accumulate_grad_tensor_reference",  # backend='inner_compiler' raised:
    # 测试函数名称："test_accumulate_grad_tensor_reference"
    # 异常信息："backend='inner_compiler' raised"
    
    "test_anomaly_grad_warnings",  # "one of the variables needed for gradient computation has been modified by an...
    # 测试函数名称："test_anomaly_grad_warnings"
    # 异常信息："one of the variables needed for gradient computation has been modified by an operation or function call"
    
    "test_autograd_inplace_views_cross_dtype",  # view_fn not supported by compiled autograd
    # 测试函数名称："test_autograd_inplace_views_cross_dtype"
    # 异常信息："view_fn not supported by compiled autograd"
    
    "test_current_node",  # TorchDispatchMode not yet implemented for compiled autograd
    # 测试函数名称："test_current_node"
    # 异常信息："TorchDispatchMode not yet implemented for compiled autograd"
    
    "test_custom_function_exception",  # "Simulate error on backward pass" does not match "type object 'SimulateBackwa...
    # 测试函数名称："test_custom_function_exception"
    # 异常信息："Simulate error on backward pass" does not match "type object 'SimulateBackwardError'"
    
    "test_grad_batched_grad",  # Cannot access storage of BatchedTensorImpl
    # 测试函数名称："test_grad_batched_grad"
    # 异常信息："Cannot access storage of BatchedTensorImpl"
    
    "test_index_backward_does_not_save_tensor",  # dynamic shape operator: aten.nonzero.default
    # 测试函数名称："test_index_backward_does_not_save_tensor"
    # 异常信息："dynamic shape operator: aten.nonzero.default"
    
    "test_post_accumulate_grad_hook_ordering",  # tensor_post_acc_grad_hooks not implemented for compiled autograd
    # 测试函数名称："test_post_accumulate_grad_hook_ordering"
    # 异常信息："tensor_post_acc_grad_hooks not implemented for compiled autograd"
    
    "test_post_accumulate_grad_hook_returns_not_None",  # "hooks should return None." does not match
    # 测试函数名称："test_post_accumulate_grad_hook_returns_not_None"
    # 异常信息："hooks should return None." does not match
    
    "test_reentrant_child_error",  # "Simulate error" does not match "type object 'ReentrantFunc' has no attribute...
    # 测试函数名称："test_reentrant_child_error"
    # 异常信息："Simulate error" does not match "type object 'ReentrantFunc' has no attribute..."
    
    "test_retain_grad_cycle",  # retains_grad_hooks not implemented for compiled autograd
    # 测试函数名称："test_retain_grad_cycle"
    # 异常信息："retains_grad_hooks not implemented for compiled autograd"
    
    "test_retain_grad_inplace",  # retains_grad_hooks not implemented for compiled autograd
    # 测试函数名称："test_retain_grad_inplace"
    # 异常信息："retains_grad_hooks not implemented for compiled autograd"
    
    "test_retain_grad_inplace_over_view",  # retains_grad_hooks not implemented for compiled autograd
    # 测试函数名称："test_retain_grad_inplace_over_view"
    # 异常信息："retains_grad_hooks not implemented for compiled autograd"
    
    "test_retains_grad_can_always_observe_tensor_prehook",  # retains_grad_hooks not implemented for compiled autograd
    # 测试函数名称："test_retains_grad_can_always_observe_tensor_prehook"
    # 异常信息："retains_grad_hooks not implemented for compiled autograd"
    
    "test_retains_grad_inplace_multiple_outputs",  # retains_grad_hooks not implemented for compiled autograd
    # 测试函数名称："test_retains_grad_inplace_multiple_outputs"
    # 异常信息："retains_grad_hooks not implemented for compiled autograd"
    
    "test_to_sparse_backward",  # backend='inner_compiler' raised:
    # 测试函数名称："test_to_sparse_backward"
    # 异常信息："backend='inner_compiler' raised"
    
    "test_accumulate_grad",  # RuntimeError: compiled_autograd does not support create_graph
    # 测试函数名称："test_accumulate_grad"
    # 异常类型：RuntimeError
    # 异常信息："compiled_autograd does not support create_graph"
    
    "test_anomaly_assign_parent_cleanup",  # RuntimeError: compiled_autograd does not support create_graph
    # 测试函数名称："test_anomaly_assign_parent_cleanup"
    # 异常类型：RuntimeError
    # 异常信息："compiled_autograd does not support create_graph"
    
    "test_anomaly_mode_no_check_nan",  # RuntimeError: compiled_autograd does not support AnomalyMode
    # 测试函数名称："test_anomaly_mode_no_check_nan"
    # 异常类型：RuntimeError
    # 异常信息："compiled_autograd does not support AnomalyMode"
    
    "test_backward_create_graph_warns",  # RuntimeError: compiled_autograd does not support create_graph
    # 测试函数名称："test_backward_create_graph_warns"
    # 异常类型：RuntimeError
    # 异常信息："compiled_autograd does not support create_graph"
    
    "test_backward_with_nonleaf_inputs",  # RuntimeError: compiled_autograd does not support create_graph
    # 测试函数名称："test_backward_with_nonleaf_inputs"
    # 异常类型：RuntimeError
    # 异常信息："compiled_autograd does not support create_graph"
    
    "test_create_graph_and_full_backward_hook_cycle",  # RuntimeError: compiled_autograd does not support create_graph
    # 测试函数名称："test_create_graph_and_full_backward_hook_cycle"
    # 异常类型：RuntimeError
    # 异常信息："compiled_autograd does not support create_graph"
    
    "test_current_graph_task_id",  # torch._dynamo.exc.Unsupported: torch.* op returned non-Tensor int
    # 测试函数名称："test_current_graph_task_id"
    # 异常类型：torch._dynamo.exc.Unsupported
    # 异常信息："torch.* op returned non-Tensor int"
    
    "test_custom_autograd_repeated_grad_grad",  # RuntimeError: compiled_autograd does not support create_graph
    # 测试函数名称："test_custom_autograd_repeated_grad_grad"
    # 异常类型：RuntimeError
    # 异常信息："compiled_autograd does not support create_graph"
    
    "test_custom_function_forward_mode_forward_is_no_op",  # AttributeError: type object 'MyFn'
    # 测试函数名称："test_custom_function_forward_mode_forward_is_no_op"
    # 异常类型：AttributeError
    # 异常信息："type object 'MyFn'"
    
    "test_custom_function_forward_mode_inplace_checks",  # AttributeError: type object 'InplaceFn'
    # 测试函数名称："test_custom_function_forward_mode_inplace_checks"
    # 异常类型：AttributeError
    # 异常信息："type object 'InplaceFn'"
    
    "test_custom_function_forward_mode_view_checks",  # AttributeError: type object 'ViewFn'
    # 测试函数名称："test_custom_function_forward_mode_view_checks"
    # 异常类型：AttributeError
    # 异常信息："type object 'ViewFn'"
    
    "test_custom_function_forward_mode_wrong_formula",  # AttributeError: type object 'UserFn'
    # 测试函数名称："test_custom_function_forward_mode_wrong_formula"
    # 异常类型：AttributeError
    # 异常信息："type object 'UserFn'"
    "test_default_saved_variable_hooks_double_backward",  # RuntimeError: compiled_autograd does not support create_graph
    "test_full_backward_hook_double_backward",  # RuntimeError: compiled_autograd does not support create_graph
    "test_function",  # RuntimeError: compiled_autograd does not support create_graph
    "test_grad",  # RuntimeError: compiled_autograd does not support create_graph
    "test_grad_materialize_grads",  # RuntimeError: compiled_autograd does not support create_graph
    "test_grad_nonleaf",  # RuntimeError: compiled_autograd does not support create_graph
    "test_grad_nonleaf_many_outputs",  # RuntimeError: compiled_autograd does not support create_graph
    "test_hessian_vector",  # RuntimeError: compiled_autograd does not support create_graph
    "test_hook_closure_cycle_use_custom_function_True_use_tensor_hook_False",  # AttributeError: type object
    "test_hook_closure_cycle_use_custom_function_True_use_tensor_hook_True",  # AttributeError: type object
    "test_hook_edge_case_when_called_with_grad",  # retains_grad_hooks NYI
    "test_inplace_on_view_backward",  # RuntimeError: compiled_autograd does not support create_graph
    "test_multi_grad_any_hooks",  # register_multi_grad_hook NYI
    "test_multi_grad_all_hooks",  # retains_grad_hooks NYI
    "test_nested_anomaly_detect_nan",  # RuntimeError: compiled_autograd does not support create_graph
    "test_nested_anomaly_printstack_cleanup",  # RuntimeError: compiled_autograd does not support create_graph
    "test_once_differentiable",  # RuntimeError: compiled_autograd does not support create_graph
    "test_prehook_ordering",  # retains_grad_hooks NYI
    "test_retain_grad",  # RuntimeError: retains_grad_hooks not implemented for compiled autograd
    "test_saved_variable_packing_unpacking_saved_original_with_hooks",  # RuntimeError: compiled_autograd
    "test_select_sum",  # torch.autograd.gradcheck.GradcheckError: While computing batched gradients
    "test_unrelated_inputs",  # torch.autograd.gradcheck.GradcheckError: While computing batched gradients
    "test_will_engine_execute_node",  # retains_grad_hooks NYI
    "test_backward_to_node",  # retains_grad_hooks NYI
    "test_anomaly_detect_nan",  # torch._dynamo.exc.TorchRuntimeError: Failed running call_function aten.add.Tensor(
    "test_autograd_multiple_views_python",  # torch._dynamo.exc.Unsupported: call_function args: TensorVariable(
    "test_autograd_node_isinstance",  # torch._dynamo.exc.Unsupported: 'inline in skipfiles: TestCase.assertIsInstance
    "test_autograd_simple_views_python",  # torch._dynamo.exc.TorchRuntimeError: Failed running call_function
    "test_callback_propagates_errors_from_device_thread",  # AssertionError: "blah" does not match "call_method
    "test_custom_autograd_no_early_free",  # torch.autograd.gradcheck.GradcheckError: While computing batched gradients
    "test_custom_function_cycle",  # torch._dynamo.exc.Unsupported: call_function UserDefinedClassVariable() [] {}
    "test_custom_function_error",  # AssertionError: "must implement either the backward" does not match "call_function
    "test_custom_function_non_tensor_inputs_outputs",  # torch._dynamo.exc.Unsupported: call_function
    "test_custom_function_save_for_forward",  # torch._dynamo.exc.Unsupported: call_function
    "test_custom_function_setup_context_multi_input",  # torch._dynamo.exc.Unsupported: call_function args
    "test_custom_function_setup_context_multi_output",  # torch._dynamo.exc.Unsupported: call_function args
    "test_deep_reentrant",  # torch._dynamo.exc.InternalTorchDynamoError: '<' not supported between instances of
    "test_dont_materialize_grads",  # torch._dynamo.exc.Unsupported: 'inline in skipfiles: TestCase.assertIsNone
    "test_function_returns_undefined_tensor",  # torch._dynamo.exc.TorchRuntimeError: Failed running call_function
    "test_grad_mode_restored_reentrant",  # torch._dynamo.exc.Unsupported: 'inline in skipfiles: TestCase.assertTrue
    "test_invalid_gradients",  # AssertionError: "expected shape" does not match "The size of tensor a (5) must match
    "test_mark_non_differentiable_mixed",  # torch._dynamo.exc.Unsupported: 'inline in skipfiles: TestCase.assertTrue
    "test_materialize_grads",  # torch._dynamo.exc.Unsupported: call_function UserDefinedClassVariable() [] {}
    "test_naughty_autograd_function_stashing_ctx",  # torch._dynamo.exc.TorchRuntimeError: Failed running call_function
    "test_no_grad_copy",  # torch._dynamo.exc.Unsupported: call_function args: TensorVariable() SkipFunctionVariable()
    "test_no_grad_copy_sparse",  # torch._dynamo.exc.Unsupported: Tensor.data_ptr
    "test_reentrant_priority",  # torch._dynamo.exc.InternalTorchDynamoError: '<' not supported between instances of
    "test_reentrant_with_callbacks_both_depths",  # torch._dynamo.exc.Unsupported: call_method UserDefinedObjectVariable
    "test_reentrant_with_callbacks_depth_0",  # torch._dynamo.exc.Unsupported: call_method UserDefinedObjectVariable
    "test_reentrant_with_callbacks_depth_1",  # torch._dynamo.exc.Unsupported: Tensor.requires_grad_
    "test_return_duplicate",  # torch.autograd.gradcheck.GradcheckError: While computing batched gradients
    "test_return_duplicate_inplace",  # torch.autograd.gradcheck.GradcheckError: While computing batched gradients
    "test_return_leaf",  # torch._dynamo.exc.Unsupported: call_function UserDefinedClassVariable() [] {}
    "test_save_none_for_backward",  # AssertionError:
    "test_save_output_nr",  # torch._dynamo.exc.Unsupported: call_function UserDefinedClassVariable() [] {}
    "test_saved_variables_deprecated",  # torch._dynamo.exc.Unsupported: UNPACK_SEQUENCE SkipFunctionVariable()
    "test_set_materialize_non_diff_grads",  # torch._dynamo.exc.Unsupported: 'inline in skipfiles: TestCase.assertIsNone



    # 错误：调用自定义函数导致不支持的异常
    "test_custom_function_cycle",
    # 错误：断言失败，期望的错误信息不匹配
    "test_custom_function_error",
    # 错误：调用自定义函数导致不支持的异常
    "test_custom_function_non_tensor_inputs_outputs",
    # 错误：调用自定义函数导致不支持的异常
    "test_custom_function_save_for_forward",
    # 错误：调用自定义函数的参数不正确导致不支持的异常
    "test_custom_function_setup_context_multi_input",
    # 错误：调用自定义函数的参数不正确导致不支持的异常
    "test_custom_function_setup_context_multi_output",
    # 错误：深度重新进入时引发内部错误
    "test_deep_reentrant",
    # 错误：跳过梯度材料化时的断言错误
    "test_dont_materialize_grads",
    # 错误：调用函数返回未定义的张量时引发运行时错误
    "test_function_returns_undefined_tensor",
    # 错误：在重新进入时，恢复梯度模式时的断言错误
    "test_grad_mode_restored_reentrant",
    # 错误：无效梯度时的断言错误
    "test_invalid_gradients",
    # 错误：混合标记为非可区分时的断言错误
    "test_mark_non_differentiable_mixed",
    # 错误：调用自定义函数导致不支持的异常
    "test_materialize_grads",
    # 错误：调用函数失败导致运行时错误
    "test_naughty_autograd_function_stashing_ctx",
    # 错误：在复制梯度时参数不正确导致不支持的异常
    "test_no_grad_copy",
    # 错误：稀疏张量数据指针错误导致不支持的异常
    "test_no_grad_copy_sparse",
    # 错误：重新进入时优先级引发内部错误
    "test_reentrant_priority",
    # 错误：回调深度为0或1时调用自定义对象时的异常
    "test_reentrant_with_callbacks_both_depths",
    "test_reentrant_with_callbacks_depth_0",
    "test_reentrant_with_callbacks_depth_1",
    # 错误：返回重复张量时的梯度检查错误
    "test_return_duplicate",
    "test_return_duplicate_inplace",
    # 错误：调用自定义函数导致不支持的异常
    "test_return_leaf",
    # 错误：保存为后向传播时错误
    "test_save_none_for_backward",
    # 错误：调用自定义函数导致不支持的异常
    "test_save_output_nr",
    # 错误：已弃用的保存变量导致不支持的异常
    "test_saved_variables_deprecated",
    # 错误：设置材化为非差异梯度时的断言错误
    "test_set_materialize_non_diff_grads"
    "test_setup_context_when_forward_has_default_args",  # torch._dynamo.exc.Unsupported: call_function args
    "test_simple_reentrant",  # torch._dynamo.exc.Unsupported: call_method SkipFunctionVariable() sum [] {}
    "test_lobpcg",  # torch._dynamo.exc.Unsupported: 'call_function LOBPCGAutogradFunction.backward in skip_files
    "test_backward_dict_grad_for_nontensor",  # AssertionError: "non-Tensor-like types" does not match "'skip function
    "test_backward_dict_invalid_keys",  # AssertionError: "to have keys {'x'}" does not match "'skip function
    "test_backward_dict_requires_keys_for_input_optional_tensors",  # AssertionError: "to have keys {.*'y'.*}"
    "test_backward_dict_requires_keys_for_input_tensors",  # AssertionError: "to have keys {.*'y'.*}" does not
    "test_backward_grads_are_tensor_or_none",  # AssertionError: "either None or a Tensor" does not match "'
    "test_backward_impl_on_existing_op",  # torch._dynamo.exc.Unsupported: 'skip function
    "test_backward_returns_dict",  # AssertionError: "to be a dict" does not match "'skip function
    "test_backward_tensorlist_input_requires_list_grads",  # AssertionError: "list of gradients" does not
    "test_backward_tensorlist_input_requires_list_grads_none_or_Tensor",  # AssertionError: "None or Tensor"
    "test_backward_tensorlist_input_requires_list_grads_with_same_numel",  # AssertionError: "3 gradients
    "test_save_for_backward_inputs_are_namedtuple",  # torch._dynamo.exc.Unsupported: 'skip function
    "test_setitem",  # AssertionError: Tensor-likes are not close!
    "test_grad_nonleaf_register_hook",  # IndexError: list index out of range (NB: x.grad = y where both x and y are input tensors)
    "test_unpack_hooks_exec_count",  # pack/unpack saved tensor hooks firing more than once
    "test_scalar_grad_mixed_device",  # Fake Tensors aren't propagating device properly for 0-dim grads


注释：

    "test_setup_context_when_forward_has_default_args",  # 测试函数：设置前向传播默认参数上下文，遇到 torch._dynamo.exc.Unsupported 异常，可能是调用函数参数问题
    "test_simple_reentrant",  # 测试函数：简单的可重入调用，遇到 torch._dynamo.exc.Unsupported 异常，可能是调用方法 SkipFunctionVariable() 的问题
    "test_lobpcg",  # 测试函数：LOBPCG，遇到 torch._dynamo.exc.Unsupported 异常，可能是调用 LOBPCGAutogradFunction.backward 中的问题
    "test_backward_dict_grad_for_nontensor",  # 测试函数：对于非张量的字典梯度，遇到 AssertionError 异常，期望非张量类型不匹配
    "test_backward_dict_invalid_keys",  # 测试函数：无效键的字典梯度，遇到 AssertionError 异常，期望键为 {'x'}，实际上不匹配
    "test_backward_dict_requires_keys_for_input_optional_tensors",  # 测试函数：需要输入可选张量的键的字典梯度，遇到 AssertionError 异常，期望键包含 {'y'}
    "test_backward_dict_requires_keys_for_input_tensors",  # 测试函数：需要输入张量的键的字典梯度，遇到 AssertionError 异常，期望键包含 {'y'}，实际上不匹配
    "test_backward_grads_are_tensor_or_none",  # 测试函数：梯度应为张量或 None，遇到 AssertionError 异常，期望是 None 或张量
    "test_backward_impl_on_existing_op",  # 测试函数：现有操作的反向实现，遇到 torch._dynamo.exc.Unsupported 异常，可能是跳过函数的问题
    "test_backward_returns_dict",  # 测试函数：反向传播应返回字典，遇到 AssertionError 异常，期望返回的是字典
    "test_backward_tensorlist_input_requires_list_grads",  # 测试函数：张量列表输入需要列表梯度，遇到 AssertionError 异常，期望是梯度列表
    "test_backward_tensorlist_input_requires_list_grads_none_or_Tensor",  # 测试函数：张量列表输入需要列表梯度为 None 或张量，遇到 AssertionError 异常，期望是 None 或张量
    "test_backward_tensorlist_input_requires_list_grads_with_same_numel",  # 测试函数：张量列表输入需要具有相同 numel 的列表梯度，遇到 AssertionError 异常，期望是 3 个梯度
    "test_save_for_backward_inputs_are_namedtuple",  # 测试函数：保存反向传播输入应为命名元组，遇到 torch._dynamo.exc.Unsupported 异常，可能是跳过函数的问题
    "test_setitem",  # 测试函数：setitem 操作，遇到 AssertionError 异常，期望张量类型不匹配
    "test_grad_nonleaf_register_hook",  # 测试函数：非叶节点的梯度注册钩子，遇到 IndexError 异常，列表索引超出范围（例如，x.grad = y，其中 x 和 y 都是输入张量）
    "test_unpack_hooks_exec_count",  # 测试函数：解包保存的张量钩子，触发多次执行
    "test_scalar_grad_mixed_device",  # 测试函数：标量梯度混合设备传播错误，Fake Tensors 在处理零维梯度时未正确传播设备信息
}

# 如果没有CUDA支持，则向known_failing_tests集合中添加测试名称
if not HAS_CUDA:
    known_failing_tests.add("test_type_conversions")

# 加载并获取测试模块test_autograd和test_custom_ops
test_autograd = load_test_module("test_autograd")
test_custom_ops = load_test_module("test_custom_ops")

# 包装测试类TestAutogradWithCompiledAutograd和TestCustomOpWithCompiledAutograd
TestAutogradWithCompiledAutograd = wrap_test_class(test_autograd.TestAutograd)
TestCustomOpWithCompiledAutograd = wrap_test_class(test_custom_ops.TestCustomOp)

# 如果作为主程序运行
if __name__ == "__main__":
    # 如果有CPU支持，则运行测试，并指定需要文件锁定（filelock）
    if HAS_CPU:
        run_tests(needs="filelock")
```