# `.\pytorch\test\test_stateless.py`

```
# Owner(s): ["module: nn"]

# 引入上下文管理模块，用于创建上下文管理器
import contextlib
import os  # 引入操作系统相关功能模块
import re  # 引入正则表达式模块
import subprocess  # 引入子进程管理模块
import sys  # 引入系统相关功能模块
import unittest  # 引入单元测试框架模块

import torch  # 引入PyTorch深度学习框架
import torch.nn.utils.stateless as stateless  # 引入PyTorch的无状态工具模块
from torch.testing._internal.common_cuda import TEST_MULTIGPU  # 引入测试多GPU的相关功能
from torch.testing._internal.common_utils import run_tests, TestCase, parametrize, instantiate_parametrized_tests, \
    subtest  # 引入测试运行相关函数和类


class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 1)  # 创建一个包含1个输入和1个输出的线性层
        self.register_buffer('buffer', torch.ones(1))  # 注册一个包含值为1的缓冲区
        self.foo = 0.0  # 初始化一个属性foo并赋值为0.0

    def forward(self, x):
        return self.l1(x) + self.buffer  # 在前向传播中返回线性层和缓冲区的和


class MockTiedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 1)  # 创建一个包含1个输入和1个输出的线性层
        self.tied_bias = self.l1.bias  # 将线性层的偏置作为tied_bias属性
        self.register_buffer('buffer', torch.ones(1))  # 注册一个包含值为1的缓冲区
        self.register_buffer('tied_buffer', self.buffer)  # 注册一个与buffer相同值的缓冲区作为tied_buffer属性

    def forward(self, x):
        return self.l1(x) + self.tied_bias + self.buffer + self.tied_buffer  # 在前向传播中返回线性层、偏置和缓冲区的和


class TestStatelessFunctionalAPI(TestCase):
    def _run_call_with_mock_module(self, module, functional_call, device='cpu', prefix=''):
        # 准备测试所需的输入数据和参数
        x = torch.rand((1, 1)).to(device)  # 创建一个随机张量并将其移动到指定设备上
        weight = torch.tensor([[1.0]], device=device)  # 创建一个包含值为1.0的权重张量并将其移动到指定设备上
        bias = torch.tensor([0.0], device=device)  # 创建一个包含值为0.0的偏置张量并将其移动到指定设备上
        buffer = torch.tensor([0.0], device=device)  # 创建一个包含值为0.0的缓冲区张量并将其移动到指定设备上

        if prefix != '':
            parameters = {f'{prefix}.l1.weight': weight,  # 如果有前缀，使用前缀为参数命名空间
                          f'{prefix}.l1.bias': bias,
                          f'{prefix}.buffer': buffer}
        else:
            parameters = {'l1.weight': weight,  # 否则使用默认的参数命名空间
                          'l1.bias': bias,
                          'buffer': buffer}

        to_check = module
        if prefix != '':
            to_check = getattr(module, prefix)  # 获取指定前缀的模块或属性

        prev_weight = to_check.l1.weight.clone()  # 克隆当前模块的权重张量
        prev_buffer = to_check.buffer.clone()  # 克隆当前模块的缓冲区张量

        # 调用被测函数，并检查结果是否与输入一致
        res = functional_call(module, parameters, x)
        self.assertEqual(x, res)

        # 检查权重和缓冲区是否保持不变
        cur_weight = to_check.l1.weight
        cur_buffer = to_check.buffer
        self.assertEqual(cur_weight, prev_weight)
        self.assertEqual(cur_buffer, prev_buffer)

    @contextlib.contextmanager
    # 确保模块在测试期间未更改，检查参数和缓冲区是否保持不变
    def _ensure_module_unchanged(self, module, message):
        # 获取模块的原始参数和缓冲区，并转换为元组
        orig_parameters, orig_buffers = tuple(module.parameters()), tuple(module.buffers())
        # 组合原始参数和缓冲区形成原始张量列表
        orig_tensors = orig_parameters + orig_buffers
        # 对每个原始张量进行克隆，形成原始张量值的元组
        orig_tensors_values = tuple(t.clone() for t in orig_tensors)
        try:
            # 返回模块对象，允许测试函数继续执行
            yield module
        finally:
            # 获取当前模块的参数和缓冲区，并转换为元组
            parameters, buffers = tuple(module.parameters()), tuple(module.buffers())
            # 断言保证参数和缓冲区的数量与原始一致，并且所有张量在值和引用上都一致
            self.assertTrue(
                len(parameters) == len(orig_parameters)
                and len(buffers) == len(orig_buffers)
                and all(
                    t1 is t2 and torch.allclose(t1, t3)
                    for t1, t2, t3 in zip(
                        orig_tensors,
                        parameters + buffers,
                        orig_tensors_values,
                    )
                ),
                message,
            )

    # 使用参数化测试，测试给定的函数调用与模块的交互
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_functional_call(self, functional_call):
        # 创建 MockModule 实例
        module = MockModule()
        # 在模拟模块上运行测试函数
        self._run_call_with_mock_module(module, functional_call)

    # 使用参数化测试，测试 JIT 编译后的模块与给定函数调用的交互
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_functional_call_with_jit(self, functional_call):
        # 创建 MockModule 实例
        module = MockModule()
        # 对模块进行 JIT 编译
        jit_module = torch.jit.script(module)
        # 断言检查在 Jitted 模块上调用特定函数时是否抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            r'used with Jitted modules'
        ):
            self._run_call_with_mock_module(jit_module, functional_call)
        # 创建输入张量 x
        x = torch.rand((1, 1))
        # 对模块进行追踪
        traced_module = torch.jit.trace(module, x)
        # 断言检查在追踪模块上调用特定函数时是否抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            r'used with Jitted modules'
        ):
            self._run_call_with_mock_module(traced_module, functional_call)

    # 使用参数化测试，测试多 GPU 环境下，数据并行模块与给定函数调用的交互
    @unittest.skipIf(not TEST_MULTIGPU, 'multi-GPU not supported')
    @unittest.skip("This doesn't work right now")
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_functional_call_with_data_parallel(self, functional_call):
        # 创建 MockModule 实例
        module = MockModule()
        # 将模块移到 GPU 上
        module.cuda()
        # 创建数据并行模块，指定 GPU 设备列表
        dp_module = torch.nn.DataParallel(module, [0, 1])
        # 在数据并行模块上运行测试函数
        self._run_call_with_mock_module(dp_module, functional_call, device='cuda', prefix='module')

    # 使用参数化测试，测试多 GPU 环境下，数据并行模块与给定函数调用的交互
    @unittest.skipIf(not TEST_MULTIGPU, 'multi-GPU not supported')
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    # 定义测试函数，用于测试在数据并行错误情况下的功能调用
    def test_functional_call_with_data_parallel_error(self, functional_call):
        # 创建 MockModule 实例
        module = MockModule()
        # 将模块移动到 CUDA 设备
        module.cuda()
        # 使用 DataParallel 封装模块，使用 GPU 0 和 1
        dp_module = torch.nn.DataParallel(module, [0, 1])
        # 断言运行时错误中包含特定字符串，用于检测在 nn.DataParallel 模块中使用的错误
        with self.assertRaisesRegex(RuntimeError, r'used with nn.DataParallel module'):
            functional_call(
                dp_module,
                {'module.weight': torch.zeros(5, device='cuda')},
                (torch.ones(2, 5, device='cuda'),))

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    # 定义测试函数，用于测试带梯度的功能调用
    def test_functional_call_with_gradient(self, functional_call):
        # 创建 MockModule 实例
        module = MockModule()
        # 创建随机输入张量 x
        x = torch.rand((1, 1))
        # 创建需要梯度的权重和偏置张量
        weight = torch.tensor([[1.0]], requires_grad=True)
        bias = torch.tensor([0.0], requires_grad=True)
        # 创建不需要梯度的缓冲张量
        buffer = torch.tensor([0.0])
        # 将参数封装成字典
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        # 调用功能函数，并传递模块、参数和输入数据 x
        res = functional_call(module, parameters, x)
        # 执行反向传播以计算参数的梯度
        res.backward()
        # 断言权重和偏置张量的梯度不为 None
        self.assertIsNotNone(weight.grad)
        self.assertIsNotNone(bias.grad)
        # 断言缓冲张量的梯度为 None
        self.assertIsNone(buffer.grad)
        # 断言模块中其他参数的梯度为 None
        self.assertIsNone(module.l1.weight.grad)
        self.assertIsNone(module.l1.bias.grad)
        self.assertIsNone(module.buffer.grad)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    # 定义测试函数，用于测试功能调用中的批归一化
    def test_functional_batch_norm(self, functional_call):
        # 创建 BatchNorm1d 模块实例
        module = torch.nn.BatchNorm1d(10)
        # 设置模块处于训练模式，以便更新统计信息
        module.train()
        # 创建输入张量 x，并用值 128.0 填充
        x = torch.full((20, 10), 128.0)
        # 创建用于替换 running_mean 缓冲区的张量 rm
        rm = torch.zeros(10)
        # 将参数封装成字典
        parameters = {'running_mean': rm}
        # 备份当前的 running_mean
        prev_rm = module.running_mean.clone()
        # 调用功能函数，并传递模块、参数和输入数据 x
        res = functional_call(module, parameters, x)
        # 检查 running_mean 是否未被更新
        cur_rm = module.running_mean
        self.assertEqual(cur_rm, prev_rm)
        # 断言 rm 是否已更新为全为 12.8 的张量
        self.assertEqual(rm, torch.full((10,), 12.8))
        # 再次调用功能函数，不重新参数化，并检查模块是否已更新
        res = functional_call(module, {}, x)
        self.assertEqual(module.running_mean, torch.full((10,), 12.8))
    # 定义一个测试函数，用于测试循环引用的情况
    def test_circular_references(self, functional_call):
        # 创建一个 MockModule 对象，模拟一个模块
        module = MockModule()
        # 添加一个循环引用，将模块自身赋值给其属性
        module.l1.m = module
        # 创建一个大小为 (1, 1) 的随机张量 x
        x = torch.rand((1, 1))
        # 创建权重张量，偏置张量，缓冲张量
        weight = torch.tensor([[1.0]])
        bias = torch.tensor([0.0])
        buffer = torch.tensor([0.0])
        # 构建参数字典，指定不同模块属性的张量参数
        parameters = {'l1.m.l1.weight': weight,
                      'l1.bias': bias,
                      'l1.m.buffer': buffer}
        # 备份模块的权重和缓冲张量
        prev_weight = module.l1.weight.clone()
        prev_buffer = module.buffer.clone()
        # 调用功能函数进行测试，传入模块、参数、输入张量 x，不共享权重
        res = functional_call(module, parameters, x, tie_weights=False)
        # 断言函数返回的结果与输入张量 x 相等
        self.assertEqual(x, res)
        # 检查权重是否保持不变且正确访问
        cur_weight = module.l1.weight
        cur_buffer = module.buffer
        self.assertEqual(cur_weight, prev_weight)
        self.assertEqual(cur_buffer, prev_buffer)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    # 定义测试函数，验证重新参数化模块后是否正确返回原始参数
    def test_reparametrized_module_change_parametrization_original(self, functional_call):
        # 创建一个 MockModule 对象
        module = MockModule()
        # 对模块 l1 应用谱归一化
        torch.nn.utils.parametrizations.spectral_norm(module.l1)
        # 断言 l1.parametrizations.weight.original 是否在模块的命名参数中
        self.assertTrue('l1.parametrizations.weight.original' in dict(module.named_parameters()))
        # 备份谱归一化后的权重
        orig_sn_weight = module.l1.weight.clone()
        # 创建一个大小为 (1, 1) 的随机张量 x
        x = torch.rand((1, 1))
        # 构建参数字典，替换参数化过程中的原始张量
        parameters = {'l1.parametrizations.weight.original': torch.nn.Parameter(torch.tensor([[1.0]])),
                      'l1.bias': torch.tensor([0.0]),
                      'buffer': torch.tensor([0.0])}
        # 调用功能函数进行测试，传入模块、参数、输入张量 x
        res = functional_call(module, parameters, x)
        # 断言函数返回的结果与输入张量 x 相等
        self.assertEqual(x, res)
        # 验证谱归一化是否仍然应用在权重上
        self.assertTrue('l1.parametrizations.weight.original' in dict(module.named_parameters()))
        self.assertEqual(orig_sn_weight, module.l1.weight)

    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    # 定义测试方法，用于验证重新参数化模块并将其重置为原始状态时的行为
    def test_reparametrize_module_fail_reset_to_original(self, functional_call):
        # 创建模拟模块对象
        module = MockModule()
        
        # 对模块的l1层应用谱归一化参数化
        torch.nn.utils.parametrizations.spectral_norm(module.l1)
        
        # 断言模块的参数字典中包含原始权重参数的标识符
        self.assertTrue('l1.parametrizations.weight.original' in dict(module.named_parameters()))
        
        # 备份原始的谱归一化权重参数
        orig_sn_weight = module.l1.weight.clone()
        
        # 准备参数字典，包括替换谱归一化权重参数的过程
        parameters = {'l1.parametrizations.weight.original': torch.nn.Parameter(torch.tensor([[1.0]])),
                      'l1.bias': torch.tensor([0.0]),
                      'buffer': torch.tensor([0.0])}
        
        # 使用断言捕获运行时异常，确保在函数调用时会出现"shapes cannot be multiplied"的错误信息
        with self.assertRaisesRegex(RuntimeError, "shapes cannot be multiplied"):
            @torch._dynamo.disable
            def _error_case():
                x = torch.rand((4, 5))  # 这里应该是大小为(1, 1)的张量
                functional_call(module, parameters, x)  # 因为x的大小错误，此调用将失败
            _error_case()

        # 验证谱归一化是否仍然应用在模块的参数中
        self.assertTrue('l1.parametrizations.weight.original' in dict(module.named_parameters()))
        
        # 验证模块的l1层的权重是否与原始的谱归一化权重相等
        self.assertEqual(orig_sn_weight, module.l1.weight)

    # 使用参数化装饰器测试部分权重重新参数化的行为
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_reparametrize_some_weights(self, functional_call):
        # 创建模拟模块对象
        module = MockModule()
        
        # 设置权重、偏置、缓冲区和额外参数的初始值
        weight = torch.tensor([[2.0]])
        bias = torch.tensor([5.0])
        buffer = torch.tensor([3.0])
        extra = torch.tensor([1.0])

        # 准备包含仅权重参数的参数字典，并生成输入张量x
        parameters = {'l1.weight': weight}
        x = torch.randn(1, 1)
        
        # 调用功能函数，并验证输出是否符合预期
        out = functional_call(module, parameters, x)
        self.assertEqual(out, x * weight + module.l1.bias + module.buffer)

        # 准备包含权重和额外参数的参数字典，并生成新的输入张量x
        parameters = {'l1.weight': weight,
                      'extra': extra}
        x = torch.randn(1, 1)
        
        # 再次调用功能函数，并验证输出是否符合预期
        out = functional_call(module, parameters, x)
        self.assertEqual(out, x * weight + module.l1.bias + module.buffer)

    # 使用参数化装饰器测试不同功能函数的行为
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    # 定义测试方法，测试严格重参数化功能
    def test_reparametrize_strict(self, functional_call):
        # 创建一个模拟的模块对象
        module = MockModule()
        # 创建张量表示权重，偏置，缓冲区和额外参数
        weight = torch.tensor([[2.0]])
        bias = torch.tensor([5.0])
        buffer = torch.tensor([3.0])
        extra = torch.tensor([1.0])

        # 设置参数字典，包含权重，偏置和缓冲区
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        # 创建输入张量 x
        x = torch.randn(1, 1)
        # 使用上下文管理器确保模块未被修改
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a successful call',
        ):
            # 调用函数式调用，期望没有错误
            out = functional_call(module, parameters, x, strict=True)
            # 断言输出与期望值相等
            self.assertEqual(out, x * weight + bias + buffer)

        # 测试缺少部分权重的情况
        parameters = {'l1.weight': weight}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            # 使用上下文管理器断言引发了特定的运行时错误
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape("Missing key(s): 'buffer', 'l1.bias'."),
            ):
                # 调用函数式调用，期望抛出异常
                out = functional_call(module, parameters, x, strict=True)

        # 测试包含额外键的情况
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer,
                      'extra': extra}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            # 使用上下文管理器断言引发了特定的运行时错误
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape("Unexpected key(s): 'extra'."),
            ):
                # 调用函数式调用，期望抛出异常
                out = functional_call(module, parameters, x, strict=True)

        # 测试缺少部分权重和包含额外键的情况
        parameters = {'l1.weight': weight,
                      'extra': extra}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            # 使用上下文管理器断言引发了特定的运行时错误
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape("Unexpected key(s): 'extra'.") + r'\s+' + re.escape("Missing key(s): 'buffer', 'l1.bias'."),
            ):
                # 调用函数式调用，期望抛出异常
                out = functional_call(module, parameters, x, strict=True)

    # 参数化装饰器，使用不同的功能调用函数来测试
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    # 定义测试函数 test_reparametrize_special，用于测试参数重参数化的特殊情况
    def test_reparametrize_special(self, functional_call):
        # 定义一个非张量对象，用于模拟参数字典中的非张量情况
        class NonTensor:
            def __repr__(self):
                return f'<{self.__class__.__name__}>'

        # 创建一个 MockModule 的模拟对象
        module = MockModule()
        # 创建张量 weight，bias，buffer 以及一个非张量对象 non_tensor
        weight = torch.tensor([[2.0]])
        bias = torch.tensor([5.0])
        buffer = torch.tensor([3.0])
        non_tensor = NonTensor()

        # 第一个测试用例：设置参数字典中 'l1.weight' 对应的值为 weight，
        # 'l1.bias' 对应的值为 None，'buffer' 对应的值为 buffer
        parameters = {'l1.weight': weight,
                      'l1.bias': None,
                      'buffer': buffer}
        x = torch.randn(1, 1)
        # 使用 _ensure_module_unchanged 上下文管理器确保模块未被修改
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a successful call',
        ):
            # 调用 functional_call 函数，期望输出与预期一致
            out = functional_call(module, parameters, x)
            self.assertEqual(out, x * weight + buffer)

        # 第二个测试用例：设置参数字典中 'l1.weight' 对应的值为 non_tensor，
        # 预期调用失败并抛出 TypeError 异常
        parameters = {'l1.weight': non_tensor}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            with self.assertRaisesRegex(
                TypeError,
                re.escape("<NonTensor> is not an instance of torch.Tensor"),
            ):
                out = functional_call(module, parameters, x)

        # 第三个测试用例：设置参数字典中 'l1.weight' 对应的值为 weight，
        # 'foo' 对应的值为 torch.tensor([1.0])，预期调用失败并抛出 TypeError 异常
        parameters = {'l1.weight': weight, 'foo': torch.tensor([1.0])}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            with self.assertRaisesRegex(
                TypeError,
                re.escape("attribute `foo`: 0.0 is not an instance of torch.Tensor"),
            ):
                out = functional_call(module, parameters, x)

        # 第四个测试用例：设置参数字典中 'l1.weight' 对应的值为 weight，
        # 'l2.bias' 对应的值为 bias，预期调用失败并抛出 AttributeError 异常
        parameters = {'l1.weight': weight,
                      'l2.bias': bias}
        x = torch.randn(1, 1)
        with self._ensure_module_unchanged(
            module,
            'the module should not have been modified by a failed call',
        ):
            with self.assertRaisesRegex(
                AttributeError,
                re.escape("MockModule has no attribute `l2`"),
            ):
                out = functional_call(module, parameters, x)

    # 使用 parametrize 装饰器为 test_tied_weights_warns 函数添加参数化测试
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    # 定义测试函数 test_tied_weights_warns，用于测试权重绑定时的警告
    def test_tied_weights_warns(self, functional_call):
        # 创建一个 MockModule 的模拟对象
        module = MockModule()
        # 将 module 的 tied_bias 属性设置为 module.l1.bias 的值
        module.tied_bias = module.l1.bias
        # 将 module 的 tied_buffer 属性注册为 module.buffer 的值
        module.register_buffer("tied_buffer", module.buffer)

    # 使用 parametrize 装饰器为 test_reparametrize_special 函数添加参数化测试
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    # 测试函数，用于验证在重参数化时是否正确处理了权重绑定的情况
    def test_reparametrize_tie_weights(self, functional_call):
        # 创建一个模拟的绑定模块对象
        module = MockTiedModule()
        # 设置权重张量和偏置张量
        weight = torch.tensor([[2.0]])
        bias = torch.tensor([5.0])
        # 设置缓冲区张量
        buffer = torch.tensor([3.0])
        # 设置额外的张量
        extra = torch.tensor([1.0])

        # 构建参数字典，包含权重、偏置和缓冲区
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        # 生成随机输入张量
        x = torch.randn(1, 1)
        # 调用功能函数，传递模块、参数、输入数据和指定权重绑定的标志
        out = functional_call(module, parameters, x, tie_weights=True)
        # 断言输出是否符合预期的计算结果
        self.assertEqual(out, x * weight + bias + bias + buffer + buffer)

        # 更新参数字典，包含额外的张量
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer,
                      'extra': extra}
        # 生成新的随机输入张量
        x = torch.randn(1, 1)
        # 再次调用功能函数，传递更新后的参数和输入数据，并指定权重绑定
        out = functional_call(module, parameters, x, tie_weights=True)
        # 断言输出是否符合预期的计算结果
        self.assertEqual(out, x * weight + bias + bias + buffer + buffer)

    # 使用参数化装饰器，测试部分权重绑定的重参数化情况
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    def test_reparametrize_tie_some_weights(self, functional_call):
        # 创建一个模拟的绑定模块对象
        module = MockTiedModule()
        # 设置权重张量和缓冲区张量
        weight = torch.tensor([[2.0]])
        buffer = torch.tensor([3.0])

        # 构建参数字典，只包含权重和缓冲区
        parameters = {'l1.weight': weight,
                      'buffer': buffer}
        # 生成随机输入张量
        x = torch.randn(1, 1)
        # 调用状态无关的功能函数，传递模块、参数、输入数据和指定权重绑定的标志
        out = stateless.functional_call(module, parameters, x, tie_weights=True)
        # 断言输出是否符合预期的计算结果
        self.assertEqual(out, x * 2. + module.l1.bias + module.tied_bias + buffer + buffer)

    # 使用参数化装饰器，测试不同功能调用函数的重参数化情况
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless._functional_call, "stateless")
    ])
    # 定义一个测试函数，用于测试绑定权重时的错误处理
    def test_tied_weights_errors(self, functional_call):
        # 创建一个模拟的绑定模块
        module = MockTiedModule()
        # 创建张量表示权重、偏置和缓冲区
        weight = torch.tensor([[1.0]])
        bias = torch.tensor([0.0])
        buffer = torch.tensor([0.0])

        # 初始化参数字典，包括权重、偏置和缓冲区
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        
        # 生成一个随机张量 x
        x = torch.randn(1, 1)
        
        # 调用 functional_call 函数，并断言不应该有警告，测试绑定权重是否生效
        self.assertNotWarn(lambda: functional_call(module, parameters, x, tie_weights=True))

        # 如果绑定的值是相同的张量，不应该有警告
        parameters['tied_bias'] = bias
        parameters['tied_buffer'] = buffer
        self.assertNotWarn(lambda: functional_call(module, parameters, x, tie_weights=True))
        
        # 清除测试中添加的绑定的偏置和缓冲区参数
        del parameters['tied_bias']
        del parameters['tied_buffer']

        # 测试当多次使用同一键时是否会引发 ValueError
        with self.assertRaisesRegex(
            ValueError,
            re.escape("functional_call got multiple values for keys ['l1.bias', 'tied_bias']"),
        ):
            parameters['tied_bias'] = torch.tensor([5.0])
            functional_call(module, parameters, x, tie_weights=True)
        del parameters['tied_bias']

        # 测试当多次使用同一键时是否会引发 ValueError
        with self.assertRaisesRegex(
            ValueError,
            re.escape("functional_call got multiple values for keys ['buffer', 'tied_buffer']"),
        ):
            parameters['tied_buffer'] = torch.tensor([5.0])
            functional_call(module, parameters, x, tie_weights=True)

    # 定义一个测试函数，用于测试在不设置绑定权重标志的情况下是否没有错误
    def test_tied_weights_no_error_without_flag(self):
        # 创建一个模拟的绑定模块
        module = MockTiedModule()
        # 创建张量表示权重、偏置和缓冲区
        weight = torch.tensor([[1.0]])
        bias = torch.tensor([0.0])
        buffer = torch.tensor([0.0])

        # 初始化参数字典，包括权重、偏置和缓冲区
        parameters = {'l1.weight': weight,
                      'l1.bias': bias,
                      'buffer': buffer}
        
        # 生成一个随机张量 x
        x = torch.randn(1, 1)
        
        # 调用 stateless._functional_call 函数，并断言不应该有警告，测试绑定权重是否生效
        self.assertNotWarn(lambda: stateless._functional_call(module, parameters, x, tie_weights=False))
        
        # 添加一个绑定的偏置参数，并断言不应该有警告
        parameters['tied_bias'] = torch.tensor([5.0])
        self.assertNotWarn(lambda: stateless._functional_call(module, parameters, x, tie_weights=False))
        
        # 清除测试中添加的绑定的偏置参数
        del parameters['tied_bias']
        
        # 添加一个绑定的缓冲区参数，并断言不应该有警告
        parameters['tied_buffer'] = torch.tensor([5.0])
        self.assertNotWarn(lambda: stateless._functional_call(module, parameters, x, tie_weights=False))

    # 使用参数化装饰器定义多个测试用例，测试不同 functional_call 实现的 functional_call 函数
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    # 定义一个测试方法，用于测试属性设置的行为
    def test_setattr(self, functional_call):
        # 定义一个继承自 torch.nn.Module 的 Foo 类
        class Foo(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 注册一个名为 'foo' 的缓冲区，初始化为 [0.0]
                self.register_buffer('foo', torch.tensor([0.0]))

            # 前向传播方法
            def forward(self, x):
                # 修改 self.foo 的值，使其加上 1
                self.foo = self.foo + 1
                return x + self.foo

        # 定义一个名为 foo 的张量，值为 [2.0]
        foo = torch.tensor([2.0])
        # 生成一个形状为 (1,) 的随机张量 x
        x = torch.randn(1)
        # 构建一个字典 a，包含一个名为 'foo' 的键值对
        a = {'foo': foo}
        # 创建一个 Foo 类的实例 mod
        mod = Foo()
        # 调用给定的 functional_call 函数，传入 mod, a, x 进行功能调用
        functional_call(mod, a, x)
        # 断言 mod.foo 的值为 [0.0]
        self.assertEqual(mod.foo, torch.tensor([0.0]))
        # 断言 a['foo'] 的值为 [3.0]
        self.assertEqual(a['foo'], torch.tensor([3.0]))
        # 断言 foo 的值仍为 [2.0]
        self.assertEqual(foo, torch.tensor([2.0]))
        # 断言 a['foo'] 和 foo 引用的不是同一个对象
        self.assertTrue(a['foo'] is not foo)

    # 使用参数化装饰器 parametrize，测试不同的 functional_call 函数
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    # 定义测试方法，测试原地操作符的行为
    def test_in_place_operator(self, functional_call):
        # 定义一个继承自 torch.nn.Module 的 Foo 类
        class Foo(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 注册一个名为 'foo' 的缓冲区，初始化为 [0.0]
                self.register_buffer('foo', torch.tensor([0.0]))

            # 前向传播方法
            def forward(self, x):
                # 原地增加 self.foo 的值
                self.foo.add_(1)
                return x + self.foo

        # 定义一个名为 foo 的张量，值为 [2.0]
        foo = torch.tensor([2.0])
        # 生成一个形状为 (1,) 的随机张量 x
        x = torch.randn(1)
        # 构建一个字典 a，包含一个名为 'foo' 的键值对
        a = {'foo': foo}
        # 创建一个 Foo 类的实例 mod
        mod = Foo()
        # 调用给定的 functional_call 函数，传入 mod, a, x 进行功能调用
        functional_call(mod, a, x)
        # 断言 mod.foo 的值为 [0.0]
        self.assertEqual(mod.foo, torch.tensor([0.0]))
        # 断言 a['foo'] 的值为 [3.0]
        self.assertEqual(a['foo'], torch.tensor([3.0]))
        # 断言 foo 的值更新为 [3.0]，因为原地操作修改了原始的 foo 引用
        self.assertEqual(foo, torch.tensor([3.0]))
        # 断言 a['foo'] 和 foo 引用的是同一个对象
        self.assertTrue(a['foo'] is foo)

    # 使用参数化装饰器 parametrize，测试不同的 functional_call 函数
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
    # 测试设置严格模式下的setattr方法，使用functional_call作为函数参数
    def test_setattr_strict(self, functional_call):
        # 定义一个名为Bar的子类，继承自torch.nn.Module
        class Bar(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 断言确保self对象没有名为'extra'的属性
                assert not hasattr(self, 'extra')

            # 前向传播方法
            def forward(self, x):
                # 返回输入x加上self.extra的结果
                return x + self.extra

        # 创建一个字典a，包含键为'extra'，值为torch.zeros(())的项
        a = {'extra': torch.zeros(())}
        # 创建Bar类的实例mod
        mod = Bar()
        # 断言确保mod对象没有名为'extra'的属性
        self.assertTrue(not hasattr(mod, 'extra'))
        # 使用functional_call调用mod对象的forward方法，传入a和torch.ones(())作为参数，返回结果out
        out = functional_call(mod, a, torch.ones(()))
        # 断言确保out的值等于torch.ones(())
        self.assertEqual(out, torch.ones(()))
        # 再次断言确保mod对象没有名为'extra'的属性
        self.assertTrue(not hasattr(mod, 'extra'))

        # 更新字典a，包含键为'extra'，值为torch.zeros(())的项
        a = {'extra': torch.zeros(())}
        # 使用上下文管理器确保以下操作引发RuntimeError异常，异常消息包含"Unexpected key(s): 'extra'."
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("Unexpected key(s): 'extra'."),
        ):
            # 使用functional_call调用mod对象的forward方法，传入a和torch.ones(())，并且严格模式为True，返回结果out
            out = functional_call(mod, a, torch.ones(()), strict=True)
        # 断言确保mod对象没有名为'extra'的属性
        self.assertTrue(not hasattr(mod, 'extra'))

        # 更新字典a为空字典
        a = {}
        # 使用上下文管理器确保以下操作引发AttributeError异常，异常消息包含"'Bar' object has no attribute 'extra'"
        with self.assertRaisesRegex(
            AttributeError,
            re.escape("'Bar' object has no attribute 'extra'"),
        ):
            # 使用functional_call调用mod对象的forward方法，传入a和torch.ones(())作为参数，返回结果out
            out = functional_call(mod, a, torch.ones(()))
        # 断言确保mod对象没有名为'extra'的属性
        self.assertTrue(not hasattr(mod, 'extra'))

        # 更新字典a为空字典
        a = {}
        # 使用上下文管理器确保以下操作引发AttributeError异常，异常消息包含"'Bar' object has no attribute 'extra'"
        with self.assertRaisesRegex(
            AttributeError,
            re.escape("'Bar' object has no attribute 'extra'"),
        ):
            # 使用functional_call调用mod对象的forward方法，传入a和torch.ones(())，并且严格模式为True，返回结果out
            out = functional_call(mod, a, torch.ones(()), strict=True)
        # 断言确保mod对象没有名为'extra'的属性
        self.assertTrue(not hasattr(mod, 'extra'))

    # 使用@parametrize装饰器，参数为"functional_call"和子测试列表
    @parametrize("functional_call", [
        # 使用subtest函数调用torch.func.functional_call作为函数参数，名称为"torch_func"
        subtest(torch.func.functional_call, "torch_func"),
        # 使用subtest函数调用stateless.functional_call作为函数参数，名称为"stateless"
        subtest(stateless.functional_call, "stateless")
    ])
    # 测试带关键字参数的functional_call方法
    def test_functional_call_with_kwargs(self, functional_call):
        # 定义一个名为Foo的子类，继承自torch.nn.Module
        class Foo(torch.nn.Module):
            # 初始化方法，接受参数x
            def __init__(self, x):
                super().__init__()
                # 将参数x保存到self.x属性
                self.x = x

            # 前向传播方法，接受inp和其他关键字参数other_inp
            def forward(self, inp, *, other_inp):
                # 返回inp乘以self.x再加上other_inp的结果
                return inp * self.x + other_inp

        # 创建一个字典a，包含键为'x'，值为torch.zeros(2, 3)的项
        a = {'x': torch.zeros(2, 3)}
        # 创建Foo类的实例mod，传入torch.randn(2, 3)作为参数
        mod = Foo(torch.randn(2, 3))
        # 创建inp和other_inp，均为torch.randn(2, 3)
        inp, other_inp = torch.randn(2, 3), torch.randn(2, 3)
        # 使用上下文管理器确保以下操作引发TypeError异常，异常消息包含"missing 1 required keyword-only argument: 'other_inp'"
        with self.assertRaisesRegex(TypeError, "missing 1 required keyword-only argument: 'other_inp'"):
            # 使用functional_call调用mod对象的forward方法，传入a和inp作为参数，返回结果res
            functional_call(mod, a, inp)
        # 使用functional_call调用mod对象的forward方法，传入a、inp和关键字参数{'other_inp': other_inp}，返回结果res
        res = functional_call(mod, a, inp, {'other_inp': other_inp})
        # 断言确保res的值等于other_inp
        self.assertEqual(res, other_inp)
        # 使用functional_call调用mod对象的forward方法，传入a、空元组()和关键字参数{'inp': inp, 'other_inp': other_inp}，返回结果res_1
        res_1 = functional_call(mod, a, (), {'inp': inp, 'other_inp': other_inp})
        # 断言确保res和res_1的值相等
        self.assertEqual(res, res_1)
    # 定义测试函数 `test_functional_call_tuple_dicts`，用于测试函数 `functional_call` 的多种调用情况
    def test_functional_call_tuple_dicts(self):
        # 创建一个 MockModule 的模拟对象
        mod = MockModule()
        # 生成一个形状为 (1, 1) 的随机张量 x
        x = torch.rand((1, 1))
        # 生成一个字典，包含模块 `mod` 中每个参数的张量，张量值为对应参数张量的全 1 张量
        parameters = {k: torch.ones_like(v) for k, v in mod.named_parameters()}
        # 生成一个字典，包含模块 `mod` 中每个缓冲区的张量，张量值为对应缓冲区张量的全 0 张量
        buffers = {k: torch.zeros_like(v) for k, v in mod.named_buffers()}

        # 调用 torch.func.functional_call 函数，传入模块 `mod`、参数和缓冲区字典，以及输入张量 x
        res = torch.func.functional_call(mod, (parameters, buffers), x)
        # 断言结果张量 res 等于输入张量 x 加 1
        self.assertEqual(res, x + 1)

        # 再次调用 torch.func.functional_call 函数，但这次不传入任何字典
        res = torch.func.functional_call(mod, (), x)
        # 断言结果张量 res 等于模块 `mod` 对输入张量 x 的函数调用结果
        self.assertEqual(res, mod(x))

        # 创建一个包含三个字典的元组 a，每个字典分别包含 'l1.weight'、'l1.bias' 和 'buffer' 的张量
        a = ({'l1.weight': torch.ones(1, 1)}, {'l1.bias': torch.ones(1)}, {'buffer': torch.zeros(1)})
        # 调用 torch.func.functional_call 函数，传入模块 `mod` 和元组 a，以及输入张量 x
        res = torch.func.functional_call(mod, a, x)
        # 断言结果张量 res 等于输入张量 x 加 1
        self.assertEqual(res, x + 1)

    # 定义测试函数 `test_functional_call_multiple_dicts_error`，用于测试函数 `functional_call` 处理多个重复字典时的错误情况
    def test_functional_call_multiple_dicts_error(self):
        # 创建一个 MockModule 的模拟对象
        mod = MockModule()
        # 生成一个字典，包含 'l1.weight' 和 'l1.bias' 的张量，张量值分别为全 0 张量
        parameters = {'l1.weight': torch.zeros((1, 1)), 'l1.bias': torch.zeros((1, 1))}
        # 生成一个字典，包含 'l1.weight' 的张量，张量值为全 1 张量
        repeated_parameters = {'l1.weight': torch.ones((1, 1))}
        
        # 使用断言检查以下代码块是否引发了 ValueError 异常，并验证异常消息是否包含指定的字符串
        with self.assertRaisesRegex(
            ValueError,
            re.escape("['l1.weight'] appeared in multiple dictionaries"),
        ):
            # 调用 torch.func.functional_call 函数，传入模块 `mod`、参数和重复参数字典，以及输入张量 x
            torch.func.functional_call(mod, (parameters, repeated_parameters), x)

    # 定义参数化测试，测试不同的 functional_call 函数和名称
    @parametrize("functional_call", [
        subtest(torch.func.functional_call, "torch_func"),
        subtest(stateless.functional_call, "stateless")
    ])
class TestStatelessDeprecation(TestCase):
    def test_private_stateless_warns(self):
        # 定义一个包含引发警告的 Python 脚本
        script = """
import torch
import warnings

# 捕获所有警告
with warnings.catch_warnings(record=True) as w:
    # 导入 torch.nn.utils._stateless 模块
    from torch.nn.utils import _stateless

# 返回捕获到的警告数量
exit(len(w))
"""
        try:
            # 执行指定脚本，并捕获其输出
            subprocess.check_output(
                [sys.executable, '-W', 'all', '-c', script],
                stderr=subprocess.STDOUT,
                # 在 Windows 平台上，使用默认的当前工作目录会导致 `import torch` 失败，
                # 因此设置当前工作目录为当前脚本所在目录
                cwd=os.path.dirname(os.path.realpath(__file__)),)
        except subprocess.CalledProcessError as e:
            # 检查子进程返回的状态码是否为 1（表示有警告）
            self.assertEqual(e.returncode, 1)
        else:
            # 如果没有引发警告，断言失败并输出错误信息
            self.assertTrue(False, "No warning was raised.")

    def test_stateless_functional_call_warns(self):
        # 创建一个 torch.nn.Linear 模型实例
        m = torch.nn.Linear(1, 1)
        # 获取模型的参数字典
        params = dict(m.named_parameters())
        # 创建输入张量 x，形状为 (3, 1)
        x = torch.randn(3, 1)
        # 使用 assertWarnsRegex 断言捕获 FutureWarning 警告，检查是否包含特定提示信息
        with self.assertWarnsRegex(FutureWarning, "Please use `torch.func.functional_call`"):
            # 调用 stateless.functional_call 函数
            stateless.functional_call(m, params, x)

class TestPythonOptimizeMode(TestCase):
    def test_runs_with_optimize_flag(self):
        # 定义一个包含导入 torch 和 torch._functorch.deprecated 的 Python 脚本
        script = "import torch; import torch._functorch.deprecated"
        try:
            # 执行指定脚本，并捕获其输出
            subprocess.check_output(
                [sys.executable, "-OO", "-c", script],
                stderr=subprocess.STDOUT,
                # 在 Windows 平台上，使用默认的当前工作目录会导致 `import torch` 失败，
                # 因此设置当前工作目录为当前脚本所在目录
                cwd=os.path.dirname(os.path.realpath(__file__)),)
        except subprocess.CalledProcessError as e:
            # 检查子进程返回的状态码是否为 0（表示成功）
            self.assertFalse(e.returncode, "Import failed while running python in optimized mode")


# 实例化 TestStatelessFunctionalAPI 中的参数化测试
instantiate_parametrized_tests(
    TestStatelessFunctionalAPI,
)

# 如果脚本作为主程序运行，执行所有测试
if __name__ == '__main__':
    run_tests()
```