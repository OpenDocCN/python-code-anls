# `.\pytorch\test\test_utils.py`

```
# mypy: allow-untyped-defs
# Owner(s): ["module: unknown"]

# 引入必要的标准库和第三方库
import os                     # 操作系统接口模块
import random                 # 随机数生成模块
import re                     # 正则表达式模块
import shutil                 # 文件操作模块
import subprocess             # 子进程管理模块
import sys                    # Python运行时环境模块
import tempfile               # 创建临时文件和目录模块
import textwrap               # 文本包装和填充模块
import traceback              # 异常跟踪模块
import unittest               # 单元测试框架模块
import warnings               # 警告模块
from typing import Any, Dict, List   # 类型提示模块

import torch                          # PyTorch深度学习框架
import torch.cuda                     # PyTorch CUDA支持模块
import torch.nn as nn                 # PyTorch神经网络模块
import torch.utils.cpp_extension      # PyTorch C++扩展支持模块
import torch.utils.data               # PyTorch数据加载和处理模块
from torch.autograd._functions.utils import check_onnx_broadcast  # PyTorch自动微分功能模块
from torch.onnx.symbolic_opset9 import _prepare_onnx_paddings     # PyTorch ONNX符号操作模块
from torch.testing._internal.common_cuda import TEST_MULTIGPU     # PyTorch测试CUDA多GPU支持模块
from torch.testing._internal.common_device_type import (          # PyTorch测试设备类型支持模块
    instantiate_device_type_tests,
    onlyCPU,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db  # PyTorch测试方法调用模块
from torch.testing._internal.common_utils import (  # PyTorch测试常用工具模块
    IS_FBCODE,
    IS_SANDCASTLE,
    IS_WINDOWS,
    load_tests,
)

from torch.utils._device import set_device     # PyTorch设备设置模块
from torch.utils._pytree import tree_all_only, tree_any  # PyTorch树结构操作模块
from torch.utils._traceback import (                      # PyTorch追踪错误模块
    CapturedTraceback,
    format_traceback_short,
    report_compile_source_on_error,
)
from torch.utils.checkpoint import (            # PyTorch模型checkpoint管理模块
    _infer_device_type,
    checkpoint,
    checkpoint_sequential,
    get_device_states,
)
from torch.utils.data import DataLoader         # PyTorch数据加载器模块

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

# 检查系统是否支持CUDA
HAS_CUDA = torch.cuda.is_available()


from torch.testing._internal.common_utils import run_tests, TestCase  # 导入测试运行和测试案例类


class RandomDatasetMock(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return torch.tensor([torch.rand(1).item(), random.uniform(0, 1)])

    def __len__(self):
        return 1000


class TestCheckpoint(TestCase):
    # This runs checkpoint_sequential on each of the nets in
    # module_lists_to_compare, and compares them against the uncheckpointed model.
    # To compare, it checks outputs as well as input gradients and parameter gradients
    # 测试checkpoint_sequential在module_lists_to_compare中的每个网络上的效果，并将其与未checkpoint的模型进行比较。
    # 为了比较，检查输出以及输入梯度和参数梯度。
    def _check_checkpoint_sequential(
        self,
        model,
        module_lists_to_compare,
        num_chunks,
        input,
        use_reentrant,
    ):
        # not checkpointed
        out = model(input)
        out_not_checkpointed = out.detach().clone()
        model.zero_grad()
        out.sum().backward()
        grad_not_checkpointed = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
        }
        input_grad_not_checkpointed = input.grad.detach().clone()
        for model_to_compare in module_lists_to_compare:
            # checkpointed model by passing list of modules

            # Detach input tensor and set requires_grad=True for checkpointing
            detached = input.detach()
            detached.requires_grad = True

            # pass list of modules to checkpoint
            out = checkpoint_sequential(
                model_to_compare, num_chunks, detached, use_reentrant=use_reentrant
            )
            out_checkpointed = out.detach().clone()
            model.zero_grad()
            out.sum().backward()
            grad_checkpointed = {
                name: param.grad.detach().clone()
                for name, param in model.named_parameters()
            }
            input_grad_checkpointed = detached.grad.detach().clone()
            # compare outputs as well as the gradients of input and parameters
            self.assertEqual(out_checkpointed, out_not_checkpointed)
            self.assertEqual(input_grad_not_checkpointed, input_grad_checkpointed)
            for name in grad_checkpointed:
                self.assertEqual(grad_checkpointed[name], grad_not_checkpointed[name])

    # Test whether checkpoint is being triggered or not. For this, we check
    # the number of times forward pass happens
    def test_checkpoint_trigger(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def forward(self, input_var):
                self.counter += 1
                # For reentrant, need to have autograd actually
                # pack a tensor to trigger recomp
                ret = input_var * torch.tensor(2.0)
                return ret

        # checkpointed
        for use_reentrant in [True, False]:
            with self.subTest(use_reentrant=use_reentrant):
                modules = [Net() for _ in range(10)]
                for m in modules:
                    self.assertEqual(m.counter, 0)
                input_var = torch.randn(3, 4, requires_grad=True)
                out = checkpoint_sequential(
                    modules, 2, input_var, use_reentrant=use_reentrant
                )
                for m in modules:
                    self.assertEqual(m.counter, 1)
                out.sum().backward()
                for m in modules[: (len(modules) // 2)]:
                    self.assertEqual(m.counter, 2)
                for m in modules[(len(modules) // 2) :]:
                    self.assertEqual(m.counter, 1)
    # 定义一个测试函数，用于验证模型在使用检查点时的有效性
    def test_checkpoint_valid(self):
        # 创建一个包含线性层和ReLU激活函数的神经网络模型
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU(),
        )

        # 生成一个形状为(1, 100)的随机输入张量，并要求计算梯度
        input_var = torch.randn(1, 100, requires_grad=True)

        # 使用检查点技术，将模型分块为2个块
        chunks = 2
        modules = list(model.children())
        out = checkpoint_sequential(modules, chunks, input_var, use_reentrant=True)

        # 断言检查，验证在使用可重入模式时是否抛出预期的运行时错误
        with self.assertRaisesRegex(
            RuntimeError, "torch.utils.checkpoint is incompatible"
        ):
            torch.autograd.grad(
                outputs=[out],
                grad_outputs=[torch.ones(1, 5)],
                inputs=[input_var],
                create_graph=True,
            )

        # 当 use_reentrant=False 时，模型正常运行，并且梯度相同
        out = model(input_var)
        grads_no_checkpoint = torch.autograd.grad(
            outputs=[out],
            grad_outputs=[torch.ones(1, 5)],
            inputs=[input_var],
            create_graph=True,
        )

        # 使用不同的检查点设置再次运行模型
        out_checkpoint = checkpoint_sequential(
            modules, chunks, input_var, use_reentrant=False
        )

        # 检查输出结果是否一致
        self.assertEqual(out_checkpoint, out)

        # 计算使用检查点技术时的梯度
        grads_checkpoint = torch.autograd.grad(
            outputs=[out_checkpoint],
            grad_outputs=[torch.ones(1, 5)],
            inputs=[input_var],
            create_graph=True,
        )

        # 断言检查，验证使用检查点和不使用检查点时的梯度是否相同
        self.assertEqual(grads_no_checkpoint, grads_checkpoint)

    # 定义一个测试函数，用于验证不同检查点设置下模型的运行情况
    def test_checkpoint(self):
        # 遍历不同的可重入模式参数值进行子测试
        for use_reentrant in [True, False]:
            with self.subTest(use_reentrant=use_reentrant):
                # 创建一个包含线性层和ReLU激活函数的神经网络模型
                model = nn.Sequential(
                    nn.Linear(100, 50),
                    nn.ReLU(),
                    nn.Linear(50, 20),
                    nn.ReLU(),
                    nn.Linear(20, 5),
                    nn.ReLU(),
                )

                # 在不同的模型设置下，验证未经检查点和经检查点的运行结果是否一致
                self._check_checkpoint_sequential(
                    model,
                    [list(model.children()), model],
                    2,
                    torch.randn(1, 100, requires_grad=True),
                    use_reentrant=use_reentrant,
                )
    def test_checkpoint_module_list(self):
        class ModuleListNet(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个包含多个模块的列表
                module_list = [
                    nn.Linear(100, 50),  # 添加线性层，输入大小100，输出大小50
                    nn.ReLU(),           # 添加ReLU激活函数层
                    nn.Linear(50, 20),   # 添加另一个线性层，输入大小50，输出大小20
                    nn.ReLU(),           # 添加ReLU激活函数层
                    nn.Linear(20, 5),    # 添加最后一个线性层，输入大小20，输出大小5
                    nn.ReLU(),           # 添加ReLU激活函数层
                ]
                # 将模块列表转换为ModuleList对象
                self.module_list = nn.ModuleList(module_list)

            def forward(self, input):
                # 遍历模块列表中的每个模块并依次应用
                for layer in self.module_list:
                    input = layer(input)
                return input

        for use_reentrant in [True, False]:
            with self.subTest(use_reentrant=use_reentrant):
                model = ModuleListNet()

                # 使用_check_checkpoint_sequential方法比较未检查点模型和检查点模型的效果
                self._check_checkpoint_sequential(
                    model,
                    [list(model.module_list.children()), model.module_list],
                    2,  # 每隔2个模块进行检查点
                    torch.randn(1, 100, requires_grad=True),  # 创建一个大小为1x100的张量，需要梯度
                    use_reentrant=use_reentrant,
                )

    def test_checkpoint_sequential_deprecated_multiple_args(self):
        class Two(nn.Module):
            def forward(self, a, b):
                return a, b

        model = nn.Sequential(Two())
        a = torch.randn(1, 100, requires_grad=True)  # 创建一个大小为1x100的张量，需要梯度
        b = torch.randn(1, 100, requires_grad=True)  # 创建一个大小为1x100的张量，需要梯度

        for use_reentrant in [True, False]:
            with self.subTest(use_reentrant=use_reentrant):
                with self.assertRaises(TypeError):
                    # 使用checkpoint_sequential方法尝试对包含两个输入的模型进行检查点操作（已废弃）
                    checkpoint_sequential(model, 1, a, b)  # type: ignore[call-arg]

    def test_checkpoint_sequential_deprecated_no_args(self):
        class Noop(nn.Module):
            def forward(self):
                pass

        model = nn.Sequential(Noop())
        for use_reentrant in [True, False]:
            with self.subTest(use_reentrant=use_reentrant):
                with self.assertRaises(TypeError):
                    # 使用checkpoint_sequential方法尝试对不包含输入的模型进行检查点操作（已废弃）
                    checkpoint_sequential(model, 1)  # type: ignore[call-arg]

    def test_checkpoint_rng_cpu(self):
        for _ in range(5):
            inp = torch.randn(20000, device="cpu").requires_grad_()
            phase1 = torch.nn.Dropout()
            phase2 = torch.nn.Dropout()

            def run_fn(input):
                return phase2(input)

            state = torch.get_rng_state()

            out = phase1(inp)
            # 使用checkpoint函数对run_fn进行检查点操作，应用于out
            out = checkpoint(run_fn, out, use_reentrant=True)
            out.sum().backward()
            grad_with_checkpointing = inp.grad

            torch.set_rng_state(state)

            inp.grad = None

            out = phase1(inp)
            out = run_fn(out)
            out.sum().backward()
            grad_no_checkpointing = inp.grad

            # 断言使用检查点和不使用检查点的梯度值相等
            self.assertEqual(grad_with_checkpointing, grad_no_checkpointing)

    @unittest.skipIf(not HAS_CUDA, "No CUDA")
    # 定义一个测试函数，用于测试带有 CUDA 支持的检查点功能
    def test_checkpoint_rng_cuda(self):
        # 执行5次循环
        for _ in range(5):
            # 在 CUDA 设备上生成一个具有梯度的随机张量
            inp = torch.randn(20000, device="cuda").requires_grad_()
            # 定义两个 Dropout 层
            phase1 = torch.nn.Dropout()
            phase2 = torch.nn.Dropout()

            # 定义一个运行函数，用于在检查点中调用
            def run_fn(input):
                return phase2(input)

            # 获取当前 CUDA 随机数生成器的状态
            state = torch.cuda.get_rng_state()

            # 应用第一个 Dropout 层到输入张量
            out = phase1(inp)
            # 使用检查点运行 run_fn 函数，启用重入模式
            out = checkpoint(run_fn, out, use_reentrant=True)
            # 对输出进行求和并反向传播
            out.sum().backward()
            # 获取使用检查点获得的梯度
            grad_with_checkpointing = inp.grad

            # 恢复之前保存的 CUDA 随机数生成器状态
            torch.cuda.set_rng_state(state)

            # 将输入张量的梯度置空
            inp.grad = None

            # 再次应用第一个 Dropout 层到输入张量
            out = phase1(inp)
            # 直接运行 run_fn 函数
            out = run_fn(out)
            # 对输出进行求和并反向传播
            out.sum().backward()
            # 获取没有使用检查点获得的梯度
            grad_no_checkpointing = inp.grad

            # 断言使用检查点和未使用检查点获得的梯度相等
            self.assertEqual(grad_with_checkpointing, grad_no_checkpointing)

    # 如果没有 CUDA 支持，则跳过此测试
    @unittest.skipIf(not HAS_CUDA, "No CUDA")
    def test_checkpoint_not_preserve_rng_state_and_without_reentrant(self):
        # 在 CUDA 设备上生成一个具有梯度的随机张量
        inp = torch.randn(2, device="cuda").requires_grad_()
        # 定义一个 Dropout 层
        layer = torch.nn.Dropout()

        # 定义一个运行函数，用于在检查点中调用
        def run_fn(input):
            return layer(input)

        # 使用检查点运行 run_fn 函数，不保存 RNG 状态并禁用重入模式
        out = checkpoint(run_fn, inp, use_reentrant=False, preserve_rng_state=False)
        # 对输出进行求和并反向传播
        out.sum().backward()
        # 此处应当可以正常运行，不抛出错误

    # 测试检查点功能处理非张量输入的情况
    def test_checkpoint_non_tensor(self):
        # 定义一个运行函数，接受两个张量输入并返回它们的和
        def run_fn(tensor1, tensor2):
            if tensor2 is None:
                return tensor1
            return tensor1 + tensor2

        # 生成一个具有梯度的随机张量
        input_var = torch.randn(1, 100, requires_grad=True)
        # 使用检查点运行 run_fn 函数，允许重入模式
        out = checkpoint(run_fn, input_var, None, use_reentrant=True)
        # 对输出进行求和并反向传播
        out.sum().backward()
    # 定义一个测试函数，用于测试 checkpoint 函数处理非张量输入和输出的情况
    def test_checkpoint_non_tensor_inputs_outputs(self):
        # 定义内部函数 foo，接收四个参数 t1, t2, scale, t3，并返回多个值
        def foo(t1, t2, scale, t3):
            # 计算 t4，是 t1 加上 t2 乘以 t3 的结果
            t4 = t1 + t2 * t3
            # 计算 t5，是 t1 乘以 t2 加上 t3 的结果
            t5 = t1 * t2 + t3
            # 将 t4 和 t5 分别乘以 scale
            t4 *= scale
            t5 *= scale
            # 返回多个值：scale, t4, None, True, t5, "bar", t1
            return scale, t4, None, True, t5, "bar", t1
        
        # 生成一个 10x1 的随机张量 t1，并要求计算其梯度
        t1 = torch.rand(10, requires_grad=True)
        # 生成一个 10x1 的随机张量 t2，并要求计算其梯度
        t2 = torch.rand(10, requires_grad=True)
        # 生成一个 10x1 的随机张量 t3
        t3 = torch.rand(10)
        # 生成一个 0 到 10 之间的随机整数 scale
        scale = random.randint(0, 10)
        # 调用 checkpoint 函数，传入 foo 函数和其他参数，并指定 use_reentrant=True
        res = checkpoint(foo, t1, t2, scale, t3, use_reentrant=True)
        
        # 断言 scale 和 res 的第一个元素相等
        self.assertEqual(scale, res[0])
        # 断言 (t1 + t2 * t3) * scale 和 res 的第二个元素相等
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        # 断言 None 和 res 的第三个元素相等
        self.assertEqual(None, res[2])
        # 断言 True 和 res 的第四个元素相等
        self.assertEqual(True, res[3])
        # 断言 (t1 * t2 + t3) * scale 和 res 的第五个元素相等
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        # 断言 "bar" 和 res 的第六个元素相等
        self.assertEqual("bar", res[5])
        # 断言 t1 和 res 的第七个元素相等
        self.assertEqual(t1, res[6])
        
        # 验证反向传播的执行
        # 对 res 的第二个元素求和，并执行反向传播，保留计算图
        res[1].sum().backward(retain_graph=True)
        # 对 res 的第五个元素求和，并执行反向传播，保留计算图
        res[4].sum().backward(retain_graph=True)
        # 对 res 的第七个元素求和，并执行反向传播
        res[6].sum().backward()
        
        # 使用断言验证 RuntimeError 中是否包含 "Trying to backward through the graph a second time"
        with self.assertRaisesRegex(
            RuntimeError, "Trying to backward through the graph a second time"
        ):
            # 对 res 的第七个元素再次求和，并执行反向传播
            res[6].sum().backward()
        
        # 保存 t1 和 t2 的梯度
        t1_grad = t1.grad
        t2_grad = t2.grad
        
        # 重置 t1 和 t2 的梯度为 None
        t1.grad = None
        t2.grad = None
        
        # 再次调用 foo 函数，传入 t1, t2, scale, t3，但不使用 checkpoint
        res = foo(t1, t2, scale, t3)
        # 对 res 的第二个、第五个和第七个元素的梯度进行反向传播
        torch.autograd.backward([res[1].sum(), res[4].sum(), res[6].sum()])
        
        # 使用断言验证 t1 和 t1_grad 是否相等
        self.assertEqual(t1.grad, t1_grad)
        # 使用断言验证 t2 和 t2_grad 是否相等
        self.assertEqual(t2.grad, t2_grad)

    # 定义一个测试函数，用于测试 checkpoint 函数处理无张量输入的情况
    def test_checkpoint_no_tensors(self):
        # 定义内部函数 foo，接收四个参数 t1, t2, scale, t3，并返回多个值
        def foo(t1, t2, scale, t3):
            # 计算 t4，是 t1 加上 t2 乘以 t3 的结果
            t4 = t1 + t2 * t3
            # 计算 t5，是 t1 乘以 t2 加上 t3 的结果
            t5 = t1 * t2 + t3
            # 将 t4 和 t5 分别乘以 scale
            t4 *= scale
            t5 *= scale
            # 返回多个值：scale, t4, None, True, t5, "bar", t1
            return scale, t4, None, True, t5, "bar", t1
        
        # 生成一个随机数 t1
        t1 = random.random()
        # 生成一个随机数 t2
        t2 = random.random()
        # 生成一个随机数 t3
        t3 = random.random()
        # 生成一个 0 到 10 之间的随机整数 scale
        scale = random.randint(0, 10)
        # 调用 checkpoint 函数，传入 foo 函数和其他参数，并指定 use_reentrant=True
        res = checkpoint(foo, t1, t2, scale, t3, use_reentrant=True)
        
        # 断言 scale 和 res 的第一个元素相等
        self.assertEqual(scale, res[0])
        # 断言 (t1 + t2 * t3) * scale 和 res 的第二个元素相等
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        # 断言 None 和 res 的第三个元素相等
        self.assertEqual(None, res[2])
        # 断言 True 和 res 的第四个元素相等
        self.assertEqual(True, res[3])
        # 断言 (t1 * t2 + t3) * scale 和 res 的第五个元素相等
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        # 断言 "bar" 和 res 的第六个元素相等
        self.assertEqual("bar", res[5])
        # 断言 t1 和 res 的第七个元素相等
        self.assertEqual(t1, res[6])
    # 定义一个测试函数，用于测试支持部分梯度检查点的功能
    def test_checkpoint_partial_grad(self):
        # 定义一个运行函数，接受两个张量作为输入，并返回它们
        def run_fn(tensor1, tensor2):
            # tensor2 用于其他应用逻辑
            return tensor1, tensor2

        # 创建一个形状为 (1, 4) 的张量，并标记为需要梯度
        input_var = torch.randn(1, 4, requires_grad=True)
        # 创建一个形状为 (1, 4) 的张量，并标记为不需要梯度
        input_var2 = torch.randn(1, 4, requires_grad=False)
        # 使用检查点函数运行 run_fn，并启用重入模式，返回结果 out
        out = checkpoint(run_fn, input_var, input_var2, use_reentrant=True)
        # 对 out 的第一个张量的所有元素求和，并进行反向传播
        out[0].sum().backward()

        # 定义第二个运行函数，接受两个张量作为输入，并只返回第一个张量
        def run_fn2(tensor1, tensor2):
            return tensor1

        # 创建一个形状为 (1, 4) 的张量，并标记为不需要梯度
        input_var = torch.randn(1, 4, requires_grad=False)
        # 创建一个形状为 (1, 4) 的张量，并标记为需要梯度
        input_var2 = torch.randn(1, 4, requires_grad=True)
        
        # 使用断言检测运行 run_fn2 函数时的运行时错误，确保输出张量中至少有一个需要梯度
        with self.assertRaisesRegex(
            RuntimeError,
            r"none of output has requires_grad=True, this checkpoint\(\) is not necessary",
        ):
            # 使用检查点函数运行 run_fn2，并启用重入模式，返回结果 out
            out = checkpoint(run_fn2, input_var, input_var2, use_reentrant=True)
            # 对 out 的所有元素求和，并进行反向传播
            out.sum().backward()

    # 如果 CUDA 不可用，则跳过该测试
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    def test_checkpointing_without_reentrant_early_free(self):
        # 测试不使用重新进入时，是否可以提前释放临时保存的变量缓冲区
        # 使用 CUDA 内存使用量作为代理来检查

        def _do_test(fn, should_free):
            stats: List[int] = []

            def track(x, idx):
                # 跟踪每次反向传播时释放的 Tensor（对应每步清空检查点存储）
                def hook(_unused):
                    self.assertEqual(len(stats), idx)
                    torch.cuda.synchronize()
                    stats.append(torch.cuda.memory_allocated())
                    if idx > 0:
                        if should_free:
                            self.assertLess(stats[idx], stats[idx - 1])
                        else:
                            self.assertEqual(stats[idx], stats[idx - 1])

                x.register_hook(hook)

            def test_fn(x):
                # 此函数的主要属性是包含多个操作，这些操作在链中保存梯度
                x = x**2
                track(x, 2)
                x = x**2
                track(x, 1)
                x = x**2
                track(x, 0)
                x = x**2
                return x.sum()

            fn(test_fn)

            return stats

        x = torch.zeros(10, device="cuda", requires_grad=True)
        x.grad = torch.zeros_like(x)

        # 在常规反向传播中，缓冲区会被及时释放
        non_retain_stats = _do_test(lambda fn: fn(x).backward(), True)

        # 在保留梯度的反向传播中，缓冲区会被保留
        _unused_retain_stats = _do_test(
            lambda fn: fn(x).backward(retain_graph=True), False
        )

        # 在带检查点的常规反向传播中，缓冲区会被及时释放
        checkpoint_non_retain_stats = _do_test(
            lambda fn: checkpoint(fn, x, use_reentrant=False).backward(), True
        )

        # 在带检查点的保留梯度反向传播中，缓冲区会被及时释放
        checkpoint_retain_stats = _do_test(
            lambda fn: checkpoint(fn, x, use_reentrant=False).backward(
                retain_graph=True
            ),
            True,
        )

        self.assertEqual(non_retain_stats, checkpoint_non_retain_stats)
        self.assertEqual(non_retain_stats, checkpoint_retain_stats)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_get_device_states_recursive(self):
        # 准备输入数据，包括两个张量，分别在不同的 CUDA 设备上
        inp = {
            "foo": torch.rand(10, device="cuda:0"),
            "bar": [torch.rand(10, device="cuda:1")],
        }
        # 调用函数获取设备状态信息
        device_ids, device_states = get_device_states(inp)
        # 断言设备数量为两个
        self.assertEqual(2, len(device_ids))
        # 断言设备状态也为两个
        self.assertEqual(2, len(device_states))
        # 第一个设备的 ID 应为 0
        self.assertEqual(0, device_ids[0])
        # 第二个设备的 ID 应为 1
        self.assertEqual(1, device_ids[1])
        # 断言第一个设备状态是 torch.Tensor 类型
        self.assertTrue(isinstance(device_states[0], torch.Tensor))
        # 断言第二个设备状态是 torch.Tensor 类型
        self.assertTrue(isinstance(device_states[1], torch.Tensor))

    def test_infer_device_state_recursive_meta(self):
        # 准备输入数据，包括一个张量，设备类型为 "meta"
        inp = {"foo": torch.rand(10, device="meta")}
        # 调用函数推断设备类型
        device_type = _infer_device_type(inp)
        # 断言推断出的设备类型为 "meta"
        self.assertEqual("meta", device_type)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_infer_device_state_recursive_multi_cuda(self):
        # 测试对于 "cuda:0", "cuda:1" 和 "cuda:0", "cuda:0" 情况下不会发出警告
        inp = {
            "foo": torch.rand(10, device="cuda:0"),
            "bar": [torch.rand(10, device="cuda:1")],
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # 调用函数推断设备类型
            device_type = _infer_device_type(inp)
            # 断言推断出的设备类型为 "cuda"
            self.assertEqual("cuda", device_type)
        
        inp = {
            "foo": torch.rand(10, device="cuda:0"),
            "bar": [torch.rand(10, device="cuda:0")],
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # 调用函数推断设备类型
            device_type = _infer_device_type(inp)
            # 断言推断出的设备类型为 "cuda"
            self.assertEqual("cuda", device_type)
        
        # 测试对于 "cuda:0", "meta" 情况下会发出警告，并检查警告消息内容
        inp = {
            "foo": torch.rand(10, device="cuda:0"),
            "bar": [torch.rand(10, device="meta")],
        }
        with warnings.catch_warnings(record=True) as w:
            # 调用函数推断设备类型
            device_type = _infer_device_type(inp)
            # 断言推断出的设备类型为 "cuda"
            self.assertEqual("cuda", device_type)
        
        # 断言警告列表长度为 1
        self.assertEqual(len(w), 1)
        warning_msg = str(w[-1].message)
        # 断言警告消息中包含特定的警告信息
        self.assertTrue(
            "Tensor arguments, excluding CPU tensors, are detected on at least two types of devices"
            in warning_msg
        )
        self.assertTrue("Device types: ['cuda', 'meta']" in warning_msg)
        self.assertTrue("first device type: cuda" in warning_msg)
class TestDataLoaderUtils(TestCase):
    MAX_TIMEOUT_IN_SECOND = 300  # 设置最大超时时间为300秒

    def setUp(self):
        super().setUp()
        self.dataset = torch.randn(5, 3, 3, 2)  # 创建一个5x3x3x2的随机张量作为数据集
        self.batch_size = 3  # 设置批处理大小为3

    def test_random_seed(self):
        def run():
            dataloader = torch.utils.data.DataLoader(
                RandomDatasetMock(),  # 使用随机数据集模拟器创建数据加载器
                batch_size=2,  # 设置批处理大小为2
                num_workers=4,  # 设置工作进程数为4
                shuffle=True,  # 打开数据随机洗牌
                timeout=self.MAX_TIMEOUT_IN_SECOND,  # 设置超时时间
            )
            return next(iter(dataloader))  # 返回数据加载器的一个批次数据

        torch.manual_seed(2018)  # 设置随机种子为2018
        x1 = run()  # 运行数据加载器以获取第一个批次数据
        torch.manual_seed(2018)  # 重新设置随机种子为2018（确保每次运行一致性）
        x2 = run()  # 再次运行数据加载器以获取第二个批次数据
        self.assertEqual(x1, x2)  # 断言两次运行的结果应当相等

    def test_single_keep(self):
        # self.dataset 是一个张量，在此处不是 Dataset 的子类，但需要保持工作状态以便与 mypy 的类型检查忽略其类型检查
        dataloader: DataLoader = DataLoader(
            self.dataset,  # 使用数据集张量创建数据加载器
            batch_size=self.batch_size,  # 设置批处理大小为预设值
            num_workers=0,  # 不使用多进程加载数据
            drop_last=False,  # 不丢弃最后一个不足批次的数据
        )
        dataiter = iter(dataloader)  # 创建数据加载器的迭代器
        self.assertEqual(len(list(dataiter)), 2)  # 断言数据加载器的迭代次数为2

    def test_single_drop(self):
        dataloader: DataLoader = DataLoader(
            self.dataset,  # 使用数据集张量创建数据加载器
            batch_size=self.batch_size,  # 设置批处理大小为预设值
            num_workers=0,  # 不使用多进程加载数据
            drop_last=True,  # 丢弃最后一个不足批次的数据
        )
        dataiter = iter(dataloader)  # 创建数据加载器的迭代器
        self.assertEqual(len(list(dataiter)), 1)  # 断言数据加载器的迭代次数为1

    @unittest.skip(
        "FIXME: Intermittent CUDA out-of-memory error on Windows and time-out under ASAN"
    )
    def test_multi_keep(self):
        dataloader: DataLoader = DataLoader(
            self.dataset,  # 使用数据集张量创建数据加载器
            batch_size=self.batch_size,  # 设置批处理大小为预设值
            num_workers=2,  # 使用2个工作进程加载数据
            drop_last=False,  # 不丢弃最后一个不足批次的数据
            timeout=self.MAX_TIMEOUT_IN_SECOND,  # 设置超时时间
        )
        dataiter = iter(dataloader)  # 创建数据加载器的迭代器
        self.assertEqual(len(list(dataiter)), 2)  # 断言数据加载器的迭代次数为2

    def test_multi_drop(self):
        dataloader: DataLoader = DataLoader(
            self.dataset,  # 使用数据集张量创建数据加载器
            batch_size=self.batch_size,  # 设置批处理大小为预设值
            num_workers=2,  # 使用2个工作进程加载数据
            drop_last=True,  # 丢弃最后一个不足批次的数据
            timeout=self.MAX_TIMEOUT_IN_SECOND,  # 设置超时时间
        )
        dataiter = iter(dataloader)  # 创建数据加载器的迭代器
        self.assertEqual(len(list(dataiter)), 1)  # 断言数据加载器的迭代次数为1


test_dir = os.path.abspath(os.path.dirname(str(__file__)))

@unittest.skipIf(
    "SKIP_TEST_BOTTLENECK" in os.environ.keys(), "SKIP_TEST_BOTTLENECK is set"
)
class TestBottleneck(TestCase):
    def _run(self, command, timeout=30):
        """Runs a command in the shell and captures its output and error.

        Args:
            command (str): The command to be executed.
            timeout (int): Timeout duration in seconds (default is 30).

        Returns:
            tuple: A tuple containing return code, stdout as a string, and stderr as a string.
        """
        import subprocess

        # Start a subprocess to execute the command
        p = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        try:
            # Communicate with the subprocess, capturing stdout and stderr
            output, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # If timeout occurs, kill the subprocess and communicate again
            p.kill()
            output, err = p.communicate()
        
        # Get the return code of the subprocess
        rc = p.returncode
        
        # Decode output and error from bytes to ASCII strings
        output_str = output.decode("ascii")
        err_str = err.decode("ascii")
        
        # Return a tuple containing return code, stdout, and stderr
        return (rc, output_str, err_str)

    def _run_bottleneck(self, test_file, scriptargs=""):
        """Runs the Python 'torch.utils.bottleneck' module on a given test file.

        Args:
            test_file (str): The relative path of the test file.
            scriptargs (str): Additional arguments to pass to the script (default is '').

        Returns:
            tuple: A tuple containing return code, stdout as a string, and stderr as a string.
        """
        import os
        import sys

        # Get the current directory of this script
        curdir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the full file path of the test file
        filepath = f"{curdir}/{test_file}"
        
        # Append script arguments if provided
        if scriptargs != "":
            scriptargs = f" {scriptargs}"
        
        # Execute the '_run' method with the constructed command
        rc, out, err = self._run(
            f"{sys.executable} -m torch.utils.bottleneck {filepath}{scriptargs}"
        )
        
        # Return a tuple containing return code, stdout, and stderr
        return rc, out, err

    def _check_run_args(self):
        """Checks the behavior of running '_run_bottleneck' with different arguments."""
        
        # Check the case where script execution fails due to missing arguments
        rc, out, err = self._run_bottleneck("bottleneck_test/test_args.py")
        self.assertEqual(
            rc,
            2,
            atol=0,
            rtol=0,
            msg=self._fail_msg("Missing args should error", out + err),
        )

        # Check the case where script execution succeeds with valid arguments
        rc, out, err = self._run_bottleneck(
            "bottleneck_test/test_args.py", "--foo foo --bar bar"
        )
        self.assertEqual(
            rc,
            0,
            atol=0,
            rtol=0,
            msg=self._fail_msg("Should pass args to script", out + err),
        )

    def _fail_msg(self, msg, output):
        """Generates a failure message with specific output.

        Args:
            msg (str): The message describing the failure.
            output (str): The output relevant to the failure.

        Returns:
            str: A formatted failure message string.
        """
        return f"{msg}, output was:\n{output}"

    def _check_environment_summary(self, output):
        """Checks if the output contains expected information about environment summary.

        Args:
            output (str): The output to be checked.

        Raises:
            AssertionError: If the expected information is not found in the output.
        """
        import re

        # Search for 'Environment Summary' in the output
        results = re.search("Environment Summary", output)
        self.assertIsNotNone(
            results, self._fail_msg("Should have Environment Summary", output)
        )

        # Search for 'PyTorch version' within five lines after 'Environment Summary'
        results = re.search(
            r"Environment Summary.*(\n.*){,5}\nPyTorch \d+\.\d+", output
        )
        self.assertIsNotNone(
            results, self._fail_msg("Should have PyTorch version", output)
        )

    def _check_cprof_summary(self, output):
        """Checks if the output contains expected information about cProfile summary.

        Args:
            output (str): The output to be checked.

        Raises:
            AssertionError: If the expected information is not found in the output.
        """
        import re

        # Search for 'cProfile output' in the output
        results = re.search("cProfile output", output)
        self.assertIsNotNone(
            results, self._fail_msg("Should have cProfile output", output)
        )

        # Search for the distance between 'cProfile output' and 'autograd profiler output'
        # The distance should be between 6 to 50 lines
        results = re.search(
            r"cProfile output.*(\n.*){6,50}\n.*autograd profiler output", output
        )
        self.assertIsNotNone(
            results,
            self._fail_msg(
                "Distance between cProfile and autograd prof out not in [6, 50] lines",
                output,
            ),
        )
    # 检查输出中是否包含 "autograd profiler output"，返回匹配对象
    results = re.search("autograd profiler output", output)
    # 断言匹配对象不为 None，否则输出自定义失败消息
    self.assertIsNotNone(
        results, self._fail_msg("Should have autograd profiler output", output)
    )

    # 假设 autograd profiler output 后是输出的末尾，检查这段输出
    results = re.search(r"autograd profiler output.*(\n.*){6,100}", output)
    # 断言匹配对象不为 None，否则输出自定义失败消息
    self.assertIsNotNone(
        results,
        self._fail_msg(
            "Distance between autograd prof output and end of output not in [6, 100] lines",
            output,
        ),
    )

# 检查输出中是否包含 "CUDA mode"，返回匹配对象
results = re.search("CUDA mode", output)
# 如果有 CUDA 支持，断言匹配对象不为 None，否则输出自定义失败消息
self.assertIsNotNone(
    results, self._fail_msg("Should tell users CUDA", output)
)
# 如果没有 CUDA 支持，断言匹配对象为 None，否则输出自定义失败消息
self.assertIsNone(
    results, self._fail_msg("Should not tell users about CUDA", output)
)

# 如果有 CUDA 支持，跳过该测试，输出消息 "CPU-only test"
@unittest.skipIf(HAS_CUDA, "CPU-only test")
def test_bottleneck_cpu_only(self):
    # 运行性能瓶颈测试，并获取返回码、输出和错误信息
    rc, out, err = self._run_bottleneck("bottleneck_test/test.py")
    # 断言返回码为 0，否则输出运行失败的详细错误信息
    self.assertEqual(rc, 0, msg=f"Run failed with\n{err}")

    # 分别检查运行参数、环境概要、自动微分概要、性能分析概要和 CUDA 支持情况
    self._check_run_args()
    self._check_environment_summary(out)
    self._check_autograd_summary(out)
    self._check_cprof_summary(out)
    self._check_cuda(out)

# 如果没有 CUDA 支持，跳过该测试，输出消息 "No CUDA"
@unittest.skipIf(not HAS_CUDA, "No CUDA")
def test_bottleneck_cuda(self):
    # 运行 CUDA 版本的性能瓶颈测试，并获取返回码、输出和错误信息
    rc, out, err = self._run_bottleneck("bottleneck_test/test_cuda.py")
    # 断言返回码为 0，否则输出运行失败的详细错误信息
    self.assertEqual(rc, 0, msg=f"Run failed with\n{err}")

    # 分别检查运行参数、环境概要、自动微分概要、性能分析概要和 CUDA 支持情况
    self._check_run_args()
    self._check_environment_summary(out)
    self._check_autograd_summary(out)
    self._check_cprof_summary(out)
    self._check_cuda(out)
from torch.utils.collect_env import get_pretty_env_info  # 导入获取环境信息的函数

@unittest.skipIf(IS_FBCODE, "runs pip which is not available internally")
class TestCollectEnv(TestCase):
    def test_smoke(self):
        info_output = get_pretty_env_info()  # 调用获取环境信息的函数
        self.assertTrue(info_output.count("\n") >= 17)  # 断言环境信息输出的行数至少为17行

class TestONNXUtils(TestCase):
    def test_prepare_onnx_paddings(self):
        sizes = [2, 3, 4]  # 给定的大小列表
        pad = [1, 2, 3, 4]  # 给定的填充列表
        paddings = _prepare_onnx_paddings(len(sizes), pad)  # 调用准备 ONNX 填充的函数
        self.assertEqual(paddings, [0, 3, 1, 0, 4, 2])  # 断言填充结果是否符合预期

    def test_check_onnx_broadcast(self):
        def try_check_onnx_broadcast(dims1, dims2, expect_broadcast, expect_fail):
            broadcast = True  # 默认广播标志为真
            fail = False  # 默认失败标志为假
            try:
                broadcast = check_onnx_broadcast(dims1, dims2)  # 尝试检查 ONNX 广播情况
            except ValueError:
                fail = True  # 如果出现值错误，标记为失败
            self.assertEqual(broadcast, expect_broadcast)  # 断言广播结果是否符合预期
            self.assertEqual(fail, expect_fail)  # 断言失败结果是否符合预期

        # Case 1, 检查 dims1 长度小于 dims2 且 dims2 的元素数大于1的情况
        dims1 = [3, 4]
        dims2 = [2, 3, 4]
        try_check_onnx_broadcast(dims1, dims2, True, True)

        # Case 2, 检查 dims1 长度小于 dims2 且 dims2 的元素数等于1的情况
        dims1 = [3, 4]
        dims2 = [1, 1, 1]
        try_check_onnx_broadcast(dims1, dims2, True, False)

        # Case 3, 检查 dims1 长度大于 dims2 且 dims2 的元素数等于1的情况
        dims1 = [1, 1]
        dims2 = [1]
        try_check_onnx_broadcast(dims1, dims2, True, False)

        # Case 4, 检查 dims1 长度大于 dims2 且 dims1 的后缀与 dims2 相同的情况
        dims1 = [2, 3, 4]
        dims2 = [3, 4]
        try_check_onnx_broadcast(dims1, dims2, True, False)

        # Case 5, 检查 dims1 长度大于 dims2 但 dims1 的后缀与 dims2 不同的情况
        dims1 = [2, 3, 4]
        dims2 = [1, 4]
        try_check_onnx_broadcast(dims1, dims2, True, True)

        # Case 6, 检查 dims1 与 dims2 相等的情况，无需广播
        dims1 = [3, 4]
        dims2 = [3, 4]
        try_check_onnx_broadcast(dims1, dims2, False, False)

        # Case 7, 检查 dims1 长度等于 dims2 但 dims1 与 dims2 不同的情况
        dims1 = [3, 4]
        dims2 = [1, 4]
        try_check_onnx_broadcast(dims1, dims2, True, True)

        # Case 8, 检查 dims1 长度等于 dims2 且 dims2 的元素数等于1的情况
        dims1 = [3, 4]
        dims2 = [1, 1]
        try_check_onnx_broadcast(dims1, dims2, True, False)

class TestHipify(TestCase):
    def test_import_hipify(self):
        from torch.utils.hipify import hipify_python  # 导入 hipify_python 函数，禁止 F401 提示

class TestHipifyTrie(TestCase):
    def setUp(self):
        self.trie = torch.utils.hipify.hipify_python.Trie()  # 设置测试用例时初始化 Trie 对象

    def test_add_and_search_trie(self):
        self.trie.add("banana")  # 向 Trie 中添加字符串 "banana"
        self.assertTrue(self.trie.search("banana"))  # 断言 Trie 中能找到 "banana"
        self.assertFalse(self.trie.search("ban"))  # 断言 Trie 中不能找到 "ban"
        self.assertFalse(self.trie.search("dog"))  # 断言 Trie 中不能找到 "dog"
    # 测试向 Trie 数据结构添加多个单词并进行搜索功能
    def test_add_multiple_and_search_trie(self):
        # 要添加到 Trie 中的单词列表
        words_to_add = ["banana", "apple", "orange"]
        # 将每个单词逐个添加到 Trie 中
        for word in words_to_add:
            self.trie.add(word)

        # 确保每个添加的单词可以在 Trie 中被搜索到
        for word in words_to_add:
            self.assertTrue(self.trie.search(word))

        # 确保 Trie 中未添加的单词无法被搜索到
        for word in ["ban", "dog", "okay", "app"]:
            self.assertFalse(self.trie.search(word))

    # 测试字符转义功能
    def test_quote_escape(self):
        # 原始字符列表
        orig_chars = ["*", "[", ".", "+", "a", "z", "-"]
        # 预期转义后的字符串列表
        quoted_strs = ["\\*", "\\[", "\\.", "\\+", "a", "z", "\\-"]
        # 对每个字符进行转义并验证结果
        for i in range(len(orig_chars)):
            self.assertEqual(self.trie.quote(orig_chars[i]), quoted_strs[i])

    # 测试将 Trie 导出为正则表达式
    def test_export_trie_to_regex(self):
        # 要添加到 Trie 中的单词列表
        words_to_add = [
            "__CUDACC__",
            "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
            "CUDA_ERROR_ARRAY_IS_MAPPED",
            "CUDA_ERROR_NOT_MAPPED",
            "CUDA_ERROR_INVALID_SOURCE",
        ]
        # 将每个单词逐个添加到 Trie 中
        for word in words_to_add:
            self.trie.add(word)
        # 导出 Trie 为正则表达式
        regex = self.trie.export_to_regex()
        # 预期的正则表达式结果
        expected_regex = r"(?:CUDA_ERROR_(?:ARRAY_IS_MAPPED|CONTEXT_ALREADY_CURRENT|INVALID_SOURCE|NOT_MAPPED)|__CUDACC__)"
        self.assertEqual(regex, expected_regex)

    # 测试导出 Trie 包含前缀的单词为正则表达式
    def test_prefix_words_export_trie_to_regex(self):
        # 要添加到 Trie 中的单词列表
        words_to_add = ["apple", "app", "ban", "banana"]
        # 将每个单词逐个添加到 Trie 中
        for word in words_to_add:
            self.trie.add(word)
        # 导出 Trie 为正则表达式
        regex = self.trie.export_to_regex()
        # 预期的正则表达式结果
        expected_regex = r"(?:app(?:le)?|ban(?:ana)?)"
        self.assertEqual(regex, expected_regex)

    # 测试导出 Trie 只含有一个单词为正则表达式
    def test_single_export_trie_to_regex(self):
        words_to_add = ["cudaErrorInvalidMemcpyDirection"]
        # 将单词添加到 Trie 中
        for word in words_to_add:
            self.trie.add(word)
        # 导出 Trie 为正则表达式
        regex = self.trie.export_to_regex()
        # 预期的正则表达式结果
        expected_regex = "cudaErrorInvalidMemcpyDirection"
        self.assertEqual(regex, expected_regex)

    # 测试导出 Trie 包含字符为正则表达式
    def test_char_export_trie_to_regex(self):
        # 添加单个字符到 Trie 中
        self.trie.add("a")
        # 验证导出的正则表达式结果
        self.assertEqual(self.trie.export_to_regex(), "a")
        # 再添加另一个字符到 Trie 中
        self.trie.add("b")
        # 验证导出的正则表达式结果
        self.assertEqual(self.trie.export_to_regex(), "[ab]")

    # 测试导出 Trie 包含特殊字符为正则表达式
    def test_special_char_export_trie_to_regex(self):
        # 添加包含特殊字符的字符串到 Trie 中
        self.trie.add(r"c*")
        # 验证导出的正则表达式结果
        self.assertEqual(self.trie.export_to_regex(), r"c\*")
class TestAssert(TestCase):
    def test_assert_true(self):
        # verify assertions work as expected
        # bool argument
        # 调用 torch._assert 函数，验证布尔参数为 True 时无异常抛出
        torch._assert(True, "foo")
        # 使用 assertRaisesRegex 检查当布尔参数为 False 时是否抛出 AssertionError 异常，异常消息为 "bar"
        with self.assertRaisesRegex(AssertionError, "bar"):
            torch._assert(False, "bar")
        # tensor argument
        # 使用 torch.tensor 创建布尔张量，验证其为 True 时无异常
        torch._assert(torch.tensor([True], dtype=torch.bool), "foo")
        # 检查布尔张量为 False 时是否抛出 AssertionError 异常，异常消息为 "bar"
        with self.assertRaisesRegex(AssertionError, "bar"):
            torch._assert(torch.tensor([False], dtype=torch.bool), "bar")

    def test_assert_scriptable(self):
        class M(torch.nn.Module):
            def forward(self, x):
                # 在 forward 方法中使用 torch._assert，验证 x.sum() > 0
                torch._assert(x.sum() > 0, "foo")
                return x

        m = M()
        # 使用 torch.jit.script 将模块 m 脚本化
        ms = torch.jit.script(m)
        # 传递数据 x 到脚本化模块 ms，验证没有错误抛出
        x = torch.randn(4, 4).fill_(1.0)
        ms(x)
        # 检查传递布尔张量时是否抛出 torch.jit.Error 异常，异常消息为 "foo"
        with self.assertRaisesRegex(torch.jit.Error, "foo"):
            ms(torch.tensor([False], dtype=torch.bool))


@unittest.skipIf(IS_SANDCASTLE, "cpp_extension is OSS only")
class TestStandaloneCPPJIT(TestCase):
    def test_load_standalone(self):
        build_dir = tempfile.mkdtemp()
        try:
            src_path = os.path.join(build_dir, "main.cpp")
            src = textwrap.dedent(
                """\
                #include <iostream>
                #include <torch/torch.h>
                int main() {
                    auto x = torch::eye(3);
                    std::cout << x << std::endl;
                }
            """
            )
            # 将 C++ 源码写入 main.cpp 文件
            with open(src_path, "w") as f:
                f.write(src)

            # 使用 torch.utils.cpp_extension.load 加载独立的 C++ JIT 模块
            exec_path = torch.utils.cpp_extension.load(
                "standalone_load_test",
                src_path,
                build_directory=build_dir,
                is_python_module=False,
                is_standalone=True,
            )

            ext = ".exe" if IS_WINDOWS else ""
            # 验证加载后的执行路径与预期路径一致
            self.assertEqual(
                exec_path, os.path.join(build_dir, f"standalone_load_test{ext}")
            )

            # 分别以交互式（shell=True）和非交互式（shell=False）方式运行执行路径
            for shell in [True, False]:
                r = subprocess.run(
                    [exec_path],
                    shell=shell,
                    stdout=subprocess.PIPE,
                )
                # 验证执行结果返回码为 0
                self.assertEqual(r.returncode, 0)
                # 验证输出与预期输出一致，处理 Windows 下的换行符问题
                self.assertEqual(
                    textwrap.dedent(r.stdout.decode("utf-8")).replace("\r\n", "\n"),
                    textwrap.dedent(
                        """\
                         1  0  0
                         0  1  0
                         0  0  1
                        [ CPUFloatType{3,3} ]
                        """
                    ),
                )

        finally:
            # 清理临时构建目录
            shutil.rmtree(build_dir)


class DummyPrivateUse1Module:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_autocast_enabled():
        return True

    @staticmethod
    # 返回自动类型转换的数据类型，在这里是 torch.float16
    def get_autocast_dtype():
        return torch.float16

    # 静态方法：设置是否启用自动类型转换
    @staticmethod
    def set_autocast_enabled(enable):
        pass

    # 静态方法：设置自动类型转换的数据类型
    @staticmethod
    def set_autocast_dtype(dtype):
        pass

    # 静态方法：返回支持自动混合精度（Automatic Mixed Precision, AMP）的数据类型列表
    @staticmethod
    def get_amp_supported_dtype():
        return [torch.float16]
class TestExtensionUtils(TestCase):
    def tearDown(self):
        # Clean up
        # 获取私有使用1后端的名称
        backend_name = torch._C._get_privateuse1_backend_name()
        # 如果 torch 模块有该后端的属性，删除之
        if hasattr(torch, backend_name):
            delattr(torch, backend_name)
        # 如果 "torch.{backend_name}" 在 sys.modules 中，从中删除
        if f"torch.{backend_name}" in sys.modules:
            del sys.modules[f"torch.{backend_name}"]

    def test_external_module_register(self):
        # Built-in module
        # 当尝试注册一个已经存在的设备模块时，抛出运行时错误
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("cuda", torch.cuda)

        # Wrong device type
        # 当尝试注册一个错误的设备类型时，抛出运行时错误
        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
            torch._register_device_module("dummmy", DummyPrivateUse1Module)

        # 当私有使用1模块没有可用的属性时，抛出属性错误
        with self.assertRaises(AttributeError):
            torch.privateuseone.is_available()  # type: ignore[attr-defined]

        # 注册私有使用1模块
        torch._register_device_module("privateuseone", DummyPrivateUse1Module)

        # 验证私有使用1模块是否可用
        torch.privateuseone.is_available()  # type: ignore[attr-defined]

        # No supporting for override
        # 当尝试重写注册私有使用1模块时，抛出运行时错误
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("privateuseone", DummyPrivateUse1Module)

    def test_external_module_register_with_renamed_backend(self):
        # 重命名私有使用1后端为 "foo"
        torch.utils.rename_privateuse1_backend("foo")
        # 当尝试再次重命名私有使用1后端时，抛出运行时错误
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            torch.utils.rename_privateuse1_backend("dummmy")

        # 获取自定义后端名称
        custom_backend_name = torch._C._get_privateuse1_backend_name()
        # 断言自定义后端名称为 "foo"
        self.assertEqual(custom_backend_name, "foo")

        # 当 foo 模块没有可用的属性时，抛出属性错误
        with self.assertRaises(AttributeError):
            torch.foo.is_available()  # type: ignore[attr-defined]

        # 当使用 AMP 与自定义后端时，抛出断言错误
        with self.assertRaisesRegex(AssertionError, "Tried to use AMP with the"):
            with torch.autocast(device_type=custom_backend_name):
                pass

        # 注册自定义后端模块
        torch._register_device_module("foo", DummyPrivateUse1Module)

        # 验证自定义后端模块是否可用
        torch.foo.is_available()  # type: ignore[attr-defined]

        # 使用自定义后端进行 autocast 操作
        with torch.autocast(device_type=custom_backend_name):
            pass

        # 获取 "foo:1" 的设备索引，断言为 1
        self.assertEqual(torch._utils._get_device_index("foo:1"), 1)
        # 获取 torch.device("foo:2") 的设备索引，断言为 2
        self.assertEqual(torch._utils._get_device_index(torch.device("foo:2")), 2)


class TestRenderUtils(TestCase):
    def test_basic(self):
        # 断言 torch.sum 的渲染输出符合预期
        self.assertExpectedInline(
            torch._utils.render_call(torch.sum, [torch.randn(100)], {"dim": 0}),
            """torch.sum(tensor([...], size=(100,)), dim=0)""",
        )
        # 断言 torch.sum 的渲染输出符合预期
        self.assertExpectedInline(
            torch._utils.render_call(torch.sum, [torch.randn(100, 100)], {"dim": 0}),
            """torch.sum(tensor([...], size=(100, 100)), dim=0)""",
        )


class TestDeviceUtils(TestCase):
    def test_basic(self):
        # 使用 "meta" 设备上下文管理器创建张量 x
        with torch.device("meta") as dev:
            x = torch.empty(3, 3)
        # 断言张量 x 的设备类型为 "meta"
        self.assertEqual(x.device.type, "meta")
        # 断言设备上下文管理器 dev 与 torch.device("meta") 相等
        self.assertEqual(dev, torch.device("meta"))
    # 定义测试装饰器的方法
    def test_decorator(self):
        # 使用 @set_device("meta") 装饰器设置函数 f 的设备为 "meta"
        @set_device("meta")
        def f():
            # 返回一个形状为 (3, 3) 的空张量
            return torch.empty(3, 3)

        # 断言函数 f 返回的张量的设备类型为 "meta"
        self.assertEqual(f().device.type, "meta")

    # 定义测试生成器函数装饰器的方法
    def test_decorator_generator(self):
        # 使用 @set_device("meta") 装饰器设置生成器函数 f 的设备为 "meta"
        @set_device("meta")
        def f():
            # 生成两个形状为 (3, 3) 的空张量
            yield torch.empty(3, 3)
            yield torch.empty(3, 3)

        # 获取生成器的两个结果
        r1, r2 = list(f())
        # 断言生成器生成的第一个张量的设备类型为 "meta"
        self.assertEqual(r1.device.type, "meta")
        # 断言生成器生成的第二个张量的设备类型为 "meta"
        self.assertEqual(r2.device.type, "meta")

    # 定义测试 nn.Module 的方法
    def test_nn_module(self):
        # 在 "meta" 设备上创建一个线性层，输入维度为 40，输出维度为 50
        with torch.device("meta"):
            m = nn.Linear(40, 50)
        # 断言线性层的权重张量的设备类型为 "meta"
        self.assertEqual(m.weight.device.type, "meta")

    # 定义设置默认设备的测试方法
    def test_set_default_device(self):
        try:
            # 设置默认设备为 "meta"
            torch.set_default_device("meta")
            # 创建一个形状为 (2, 2) 的空张量
            r = torch.empty(2, 2)
        finally:
            # 最终将默认设备设置为 None
            torch.set_default_device(None)

        # 断言张量 r 的设备类型为 "meta"
        self.assertEqual(r.device.type, "meta")

    # 定义获取默认设备的测试方法
    def test_get_default_device(self):
        # 设置默认设备为 "meta"
        torch.set_default_device("meta")
        # 断言获取的默认设备的设备类型为 "meta"
        self.assertEqual(torch.get_default_device().type, "meta")
        # 将默认设备设置为 None
        torch.set_default_device(None)

    # 定义更多获取默认设备的测试方法，这些方法要求支持多GPU
    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_get_default_device_more(self):
        # 设置默认设备为 "cuda"
        torch.set_default_device("cuda")
        # 断言获取的默认设备为一个空张量的设备
        self.assertEqual(torch.get_default_device(), torch.tensor([]).device)
        # 将默认设备设置为 None
        torch.set_default_device(None)

        # 设置默认设备为 "cuda"，并指定第一个 GPU
        torch.set_default_device("cuda")
        torch.cuda.set_device("cuda:1")
        # 断言获取的默认设备为一个空张量的设备
        self.assertEqual(torch.get_default_device(), torch.tensor([]).device)
        # 将默认设备设置为 None
        torch.set_default_device(None)

        # 设置默认设备为 "cuda:1"
        torch.set_default_device("cuda:1")
        # 断言获取的默认设备为一个空张量的设备
        self.assertEqual(torch.get_default_device(), torch.tensor([]).device)
        # 将默认设备设置为 None
        torch.set_default_device(None)

    # 定义测试设备模式操作的方法，使用了自定义的装饰器
    @onlyCPU
    @ops(op_db)
    def test_device_mode_ops(self, device, dtype, op):
        # 获取操作函数
        func = op.get_op()
        # 从操作对象中获取设备相关的样本输入，不需要梯度
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample in samples:
            # 只测试没有张量输入的样本。但不测试 OpInfo 的工厂属性，因为它非常不完整。
            if tree_any(
                lambda x: isinstance(x, torch.Tensor),
                (sample.input, sample.args, sample.kwargs),
            ):
                continue
            # 如果样本中显式传入了设备参数，则在这种情况下需要去掉设备参数以测试 DeviceContext。
            # 注意：不能向 sample_inputs 传递 None，该函数无法处理 None。
            kwargs = sample.kwargs.copy()
            kwargs.pop("device", None)
            # 在 "meta" 设备上执行操作函数
            with torch.device("meta"):
                r = func(sample.input, *sample.args, **kwargs)

            # 定义一个函数判断张量是否使用了 "meta" 设备
            def is_meta_device(x: torch.Tensor) -> bool:
                return x.device.type == "meta"

            # 断言所有返回的张量 r 均只使用了 "meta" 设备
            self.assertTrue(tree_all_only(torch.Tensor, is_meta_device, r))
# 调用函数实例化设备类型测试，传入测试类 TestDeviceUtils 和全局命名空间 globals()
instantiate_device_type_tests(TestDeviceUtils, globals())

# 定义测试类 TestCppExtensionUtils，继承自 TestCase
class TestCppExtensionUtils(TestCase):

    # 测试函数：验证 C++ 编译器是否可用
    def test_cpp_compiler_is_ok(self):
        self.assertTrue(torch.utils.cpp_extension.check_compiler_ok_for_platform("c++"))

    # 测试函数：验证 CC 编译器是否可用
    def test_cc_compiler_is_ok(self):
        self.assertTrue(torch.utils.cpp_extension.check_compiler_ok_for_platform("cc"))


# 定义测试类 TestTraceback，继承自 TestCase
class TestTraceback(TestCase):

    # 测试函数：基本回溯测试
    def test_basic(self):
        # 定义源代码字符串
        source = """\
def f(x):
    def g(x):
        raise RuntimeError  # HEYA

    x = x * 3
    return g(x) + 1
"""
        # 定义输出字典
        out: Dict[str, Any] = {}
        # 定义作用域字典，包含源代码字符串
        scope = {"__compile_source__": source}
        # 执行源代码，将结果写入 out 字典
        exec(source, scope, out)

        try:
            # 使用 report_compile_source_on_error 上下文管理器，调用 out 字典中的函数 f(1)
            with report_compile_source_on_error():
                out["f"](1        except RuntimeError as e:
            # 断言异常消息中包含 "HEYA"，并将格式化的回溯信息转换为字符串进行断言
            self.assertIn("HEYA", "".join(traceback.format_tb(e.__traceback__)))

    # 测试函数：格式化短回溯信息测试
    def test_format_traceback_short(self):
        try:
            # 抛出 RuntimeError 异常
            raise RuntimeError
        except RuntimeError as e:
            # 断言格式化后的短回溯信息匹配指定模式
            self.assertRegex(
                format_traceback_short(e.__traceback__),
                r".*test_utils.py:\d+ in test_format_traceback_short",
            )

    # 测试函数：捕获的回溯信息测试
    def test_captured_traceback(self):
        # 断言 "test_captured_traceback" 存在于捕获的回溯信息格式化后的字符串中
        self.assertIn(
            "test_captured_traceback", "".join(CapturedTraceback.extract().format())
        )

    # 测试函数：格式化全部捕获的回溯信息测试
    def test_captured_traceback_format_all(self):
        # 调用 CapturedTraceback.format_all 函数，格式化传入的捕获的回溯信息列表
        rs = CapturedTraceback.format_all(
            [CapturedTraceback.extract(), CapturedTraceback.extract()]
        )
        # 断言返回结果列表长度为 2
        self.assertEqual(len(rs), 2)
        # 断言 "test_captured_traceback_format_all" 存在于第一个结果字符串中
        self.assertIn("test_captured_traceback_format_all", "".join(rs[0]))

    # 测试函数：格式化全部捕获的回溯信息（缓存）测试
    def test_captured_traceback_format_all_cached(self):
        # 提取捕获的回溯信息
        tb = CapturedTraceback.extract()
        # 格式化回溯信息（缓存）
        tb.format()
        # 调用 CapturedTraceback.format_all 函数，格式化传入的捕获的回溯信息列表
        rs = CapturedTraceback.format_all([tb, CapturedTraceback.extract()])
        # 断言返回结果列表长度为 2
        self.assertEqual(len(rs), 2)
        # 断言 "test_captured_traceback_format_all" 存在于第一个结果字符串中
        self.assertIn("test_captured_traceback_format_all", "".join(rs[0]))


# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```