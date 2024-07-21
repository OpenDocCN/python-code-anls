# `.\pytorch\test\test_autograd.py`

```py
# Owner(s): ["module: autograd"]

# 导入需要的标准库和第三方库
import collections
import contextlib
import functools
import gc
import io
import math
import operator
import os
import pickle
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
import warnings
import weakref
from collections import OrderedDict
from copy import deepcopy
from functools import partial, reduce
from itertools import product
from operator import mul
from typing import List, Tuple

# 导入 PyTorch 相关模块
import torch
import torch.autograd._functions
import torch.autograd.forward_ad as fwAD

# 导入 PyTorch 中的一些重要组件和功能
from torch import inf, nan, nn
from torch.autograd import (
    _calculate_shape,
    detect_anomaly,
    Function,
    kineto_available,
    Variable,
)
from torch.autograd.function import InplaceFunction, once_differentiable
from torch.autograd.graph import GradientEdge
from torch.autograd.profiler import emit_itt, emit_nvtx, profile, record_function
from torch.autograd.profiler_util import (
    _format_time,
    EventList,
    FunctionEvent,
    FunctionEventAvg,
)
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    dtypes,
    dtypesIfCUDA,
    dtypesIfMPS,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    skipMeta,
)
from torch.testing._internal.common_dtype import floating_types_and
from torch.testing._internal.common_methods_invocations import mask_not_all_zeros
from torch.testing._internal.common_utils import (
    disable_gc,
    gradcheck,
    gradgradcheck,
    instantiate_parametrized_tests,
    IS_MACOS,
    IS_WINDOWS,
    parametrize,
    run_tests,
    set_warn_always_context,
    skipIfMps,
    skipIfNoLapack,
    skipIfTorchDynamo,
    slowTest,
    TestCase,
    xfailIfTorchDynamo,
)
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import (
    checkpoint,
    checkpoint_sequential,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)
from torch.utils.cpp_extension import load_inline
from torch.utils.flop_counter import FlopCounterMode
from torch.utils.hooks import RemovableHandle  # noqa: TCH001

# 定义函数 graph_desc 用于描述函数的计算图结构
def graph_desc(fn):
    # 如果传入的函数对象是 None，则返回字符串 "None"
    if fn is None:
        return "None"
    # 构建函数对象类型的描述字符串，包含函数名和其后续函数的描述
    result = type(fn).__name__ + "("
    next_functions = fn.next_functions
    for next_fn, _ in next_functions:
        # 递归获取每个后续函数的描述，并添加到结果字符串中
        result += graph_desc(next_fn)
        result += ", "
    if next_functions:
        result = result[:-2]  # 去除最后多余的逗号和空格
    return result + ")"  # 返回完整的描述字符串

# 定义测试类 TestAutograd，继承自 unittest.TestCase
class TestAutograd(TestCase):
    pass  # 空的测试类，待添加测试方法
    def test_copy_slices_graph_task_updates(self):
        # 定义函数 f1，接受两个张量 x 和 y 作为输入
        def f1(x, y):
            # 克隆张量 x，并将其视图变形为一维张量，存储在 out 中
            out = x.clone().view(-1)
            # 将 y 加到 out 上
            out += y
            return out

        # 定义函数 f2，接受两个张量 x 和 y 作为输入
        def f2(x, y):
            # 克隆张量 x，并将其视图变形为一维张量，存储在 out 中
            out = x.clone().view(-1)
            # 将 out 乘以 2，并存储在 b 中
            b = out * 2
            # 将 y 加到 out 上，然后再将 b 加到 out 上，并返回结果
            out += y
            return out + b

        # 创建两个随机张量 x 和 y，并启用它们的梯度跟踪
        x = torch.rand(2, requires_grad=True)
        y = torch.rand(2, requires_grad=True)

        # 创建一个延迟错误的 y 安全版本
        y_safe = torch._C._functions.DelayedError("Boom!", 1)(y)

        # 遍历函数列表 [f1, f2]
        for f in [f1, f2]:
            # 确保错误节点起作用
            out = f(x, y_safe)
            # 断言捕获 RuntimeError 并包含 "Boom!" 错误信息
            with self.assertRaisesRegex(RuntimeError, "Boom!"):
                out.sum().backward()

            out = f(x, y_safe)
            # 断言捕获 RuntimeError 并包含 "Boom!" 错误信息
            with self.assertRaisesRegex(RuntimeError, "Boom!"):
                torch.autograd.grad(out.sum(), y)

            # 确保没有请求 y 时不会崩溃
            out = f(x, y_safe)
            torch.autograd.grad(out.sum(), x)

            out = f(x, y_safe)
            torch.autograd.grad(out.sum(), y_safe)

            out = f(x, y_safe)
            torch.autograd.grad(out.sum(), (x, y_safe))

        # 确保不运行额外的视图节点
        def f3(x, y):
            # 克隆张量 x，并将其视图变形为一维张量，存储在 out 中
            out = x.clone().view(-1)

            def hook(*args):
                # 这里不应该被调用！断言失败
                self.assertTrue(False)

            # 注册 hook 函数到 out 上
            out.register_hook(hook)

            # 将 y 加到 out 上，并将其存储在 b 中
            b = out + y
            # 将 y 加到 out 上，并返回 out 和 b 的和
            out += y
            return out + b, b

        # 调用函数 f3，得到 out 和 b
        out, b = f3(x, y_safe)
        # 计算 out 的和关于 (b, y_safe) 的梯度
        torch.autograd.grad(out.sum(), (b, y_safe))

    def test_grad_mode_class_decoration(self):
        # 警告：类装饰已弃用，不应使用
        with self.assertWarnsRegex(FutureWarning, "Decorating classes is deprecated"):

            # 使用 @torch.no_grad() 装饰类 Foo
            @torch.no_grad()
            class Foo:
                def __init__(self):
                    # 断言梯度未启用
                    assert not torch.is_grad_enabled()

                def foo(self):
                    # 方法 foo 中梯度应启用
                    assert torch.is_grad_enabled()

            # 构造类 Foo 的实例
            foo = Foo()
            # 调用实例的方法 foo
            foo.foo()

        # 函数或方法装饰是可以的
        with warnings.catch_warnings(record=True) as w:

            # 使用 @torch.no_grad() 装饰函数 foo
            @torch.no_grad()
            def foo():
                # 断言梯度未启用
                assert not torch.is_grad_enabled()

            # 调用函数 foo
            foo()

            # 定义类 Foo2
            class Foo2:
                # 使用 @torch.no_grad() 装饰构造函数
                @torch.no_grad()
                def __init__(self):
                    # 断言梯度未启用
                    assert not torch.is_grad_enabled()

                # 使用 @torch.no_grad() 装饰方法 foo
                @torch.no_grad()
                def foo(self):
                    # 断言梯度未启用
                    assert not torch.is_grad_enabled()

            # 构造类 Foo2 的实例
            foo2 = Foo2()
            # 调用实例 foo2 的方法 foo
            foo2.foo()

        # 确保没有警告被记录
        self.assertEqual(len(w), 0)
    def test_tensor_grad_warnings(self):
        dummy = torch.empty(1)  # 创建一个大小为1的空张量

        with warnings.catch_warnings(record=True) as w:
            # 捕获警告并记录
            dummy.requires_grad_()  # 设置张量为需要梯度
            foo = dummy.grad  # 访问叶子节点的 .grad 属性
            self.assertEqual(len(w), 0)  # 断言未发出警告

            # 访问非叶子节点的 .grad 属性
            dummy = dummy.clone()  # 克隆张量
            foo = dummy.grad  # 访问 .grad 属性
            self.assertEqual(len(w), 1)  # 断言发出了一个警告

            # 访问保留梯度的非叶子节点的 .grad 属性
            dummy.retain_grad()  # 保留梯度
            foo = dummy.grad  # 访问 .grad 属性
            self.assertEqual(len(w), 1)  # 断言发出了一个警告

    def _function_test(self, cls):
        x = torch.randn(5, 5, requires_grad=True)  # 创建一个大小为5x5的张量，并设置需要梯度
        y = torch.randn(5, 5, requires_grad=True)  # 创建一个大小为5x5的张量，并设置需要梯度
        result = cls.apply(x, 2, y)  # 调用自定义函数的静态方法进行计算
        go = torch.ones((), requires_grad=True)  # 创建一个标量张量，并设置需要梯度
        result.sum().backward(go, create_graph=True)  # 计算结果的和的梯度，创建计算图

        self.assertEqual(x.grad, y + torch.ones(5, 5))  # 断言 x 的梯度符合预期
        self.assertEqual(y.grad, x + torch.ones(5, 5) * 2)  # 断言 y 的梯度符合预期
        self.assertIsNotNone(x.grad.grad_fn)  # 断言 x 的梯度函数不为空
        self.assertIsNotNone(y.grad.grad_fn)  # 断言 y 的梯度函数不为空

        return x, y  # 返回 x 和 y 张量

    def test_function(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, tensor1, pyscalar, tensor2):
                ctx.pyscalar = pyscalar  # 在上下文中存储标量
                ctx.save_for_backward(tensor1, tensor2)  # 在上下文中保存张量
                return tensor1 + pyscalar * tensor2 + tensor1 * tensor2  # 计算前向传播结果

            @staticmethod
            def backward(ctx, grad_output):
                var1, var2 = ctx.saved_tensors  # 从上下文中获取保存的张量
                # 注意：这里的 self 是测试用例本身
                self.assertIsInstance(var1, torch.Tensor)  # 断言 var1 是 torch.Tensor 类型
                self.assertIsInstance(var2, torch.Tensor)  # 断言 var2 是 torch.Tensor 类型
                self.assertIsInstance(grad_output, torch.Tensor)  # 断言 grad_output 是 torch.Tensor 类型
                return (
                    grad_output + grad_output * var2,  # 返回 var1 的梯度
                    None,  # 返回 pyscalar 的梯度（这里为 None）
                    grad_output * ctx.pyscalar + grad_output * var1,  # 返回 var2 的梯度
                )

        x, y = self._function_test(MyFunction)  # 调用 _function_test 方法进行测试

        x_grad_desc = graph_desc(x.grad.grad_fn)  # 获取 x 的梯度函数的描述
        y_grad_desc = graph_desc(y.grad.grad_fn)  # 获取 y 的梯度函数的描述
        self.assertExpected(x_grad_desc, "x_grad_desc")  # 断言 x 的梯度函数描述符符合预期
        self.assertExpected(y_grad_desc, "y_grad_desc")  # 断言 y 的梯度函数描述符符合预期
    def test_once_differentiable(self):
        # 定义一个继承自 Function 的自定义函数 MyFunction，用于测试反向传播
        class MyFunction(Function):
            # 静态方法：定义前向传播
            @staticmethod
            def forward(ctx, tensor1, pyscalar, tensor2):
                # 保存 Python 标量到上下文，保存张量 tensor1 和 tensor2 以备反向传播使用
                ctx.pyscalar = pyscalar
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + pyscalar * tensor2 + tensor1 * tensor2

            # 静态方法：定义反向传播
            @staticmethod
            @once_differentiable
            def backward(ctx, grad_output):
                # 断言梯度是否被启用
                self.assertFalse(torch.is_grad_enabled())
                # 从上下文中获取保存的张量
                t1, t2 = ctx.saved_tensors
                # 返回计算的梯度
                return (
                    grad_output + grad_output * t2,
                    None,
                    grad_output * ctx.pyscalar + grad_output * t1,
                )

        # 对自定义函数 MyFunction 进行测试，并获取测试结果
        x, y = self._function_test(MyFunction)
        # 断言 x 的梯度函数描述符
        self.assertEqual(
            graph_desc(x.grad.grad_fn),
            "CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))",
        )
        # 断言 y 的梯度函数描述符
        self.assertEqual(
            graph_desc(y.grad.grad_fn),
            "CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))",
        )

    def test_function_returns_input(self):
        # 定义一个继承自 Function 的自定义函数 MyFunction，用于测试返回输入张量的情况
        class MyFunction(Function):
            # 静态方法：定义前向传播
            @staticmethod
            def forward(ctx, x):
                return x

            # 静态方法：定义反向传播
            @staticmethod
            def backward(ctx, grad):
                # 返回输入梯度乘以2
                return grad * 2

        # 遍历不同的张量形状进行测试
        for shape in [(1,), ()]:
            # 创建一个全为1的张量，要求计算梯度
            v = torch.ones(shape, requires_grad=True)
            # 应用自定义函数 MyFunction 进行前向和反向传播
            MyFunction.apply(v).backward()
            # 断言梯度的正确性
            self.assertEqual(v.grad, torch.full(shape, 2.0))

            # 使用 torch.no_grad 上下文清零梯度
            with torch.no_grad():
                v.grad.zero_()
            # 使用克隆的张量再次应用 MyFunction 进行前向和反向传播
            MyFunction.apply(v.clone()).backward()
            # 断言梯度的正确性
            self.assertEqual(v.grad, torch.full(shape, 2.0))

    def test_function_returns_undefined_tensor(self):
        # 定义一个继承自 Function 的自定义函数 MyFunction，用于测试返回未定义张量的情况
        class MyFunction(Function):
            # 静态方法：定义前向传播
            @staticmethod
            def forward(ctx, x):
                return x * 2

            # 静态方法：定义反向传播
            @staticmethod
            def backward(ctx, grad):
                # 返回 None 表示未定义梯度
                return None

        # 测试从自定义反向传播函数返回未定义张量的情况
        x = torch.ones(1, requires_grad=True)

        MyFunction.apply(x).backward()
        # 断言梯度为 None
        self.assertIsNone(x.grad)

        MyFunction.apply(x**2).backward()
        # 断言梯度为 None
        self.assertIsNone(x.grad)

        MyFunction.apply(x).sum().backward()
        # 断言梯度为 None
        self.assertIsNone(x.grad)

        # 使用 torch.autograd.grad 函数测试从自定义函数返回未定义张量的情况
        self.assertIsNone(
            torch.autograd.grad(MyFunction.apply(x), x, allow_unused=True)[0]
        )

    def test_materialize_grads(self):
        # 定义一个继承自 Function 的自定义函数 MyFunction，用于测试梯度是否正确地被传递
        class MyFunction(Function):
            # 静态方法：定义前向传播
            @staticmethod
            def forward(ctx, x):
                return x

            # 静态方法：定义反向传播
            @staticmethod
            def backward(ctx, grad):
                # 断言梯度为全零张量
                self.assertEqual(grad, torch.zeros(1))
                return grad

        # 创建一个全为1的张量，要求计算梯度
        x = torch.ones(1, requires_grad=True)
        # 使用 UndefinedGrad 函数应用 MyFunction 进行前向和反向传播
        torch._C._functions.UndefinedGrad()(MyFunction.apply(x)).backward()
    def test_dont_materialize_grads(self):
        class MyFunction(Function):
            @staticmethod
            # 前向传播函数的实现
            def forward(ctx, x):
                # 禁用梯度的生成
                ctx.set_materialize_grads(False)
                return x

            @staticmethod
            # 反向传播函数的实现
            def backward(ctx, grad):
                # 断言梯度应为None
                self.assertIsNone(grad)
                return grad

        x = torch.ones(1, requires_grad=True)
        # 使用自定义的MyFunction进行前向传播和反向传播
        torch._C._functions.UndefinedGrad()(MyFunction.apply(x)).backward()

    def test_set_materialize_non_diff_grads(self):
        class Func(torch.autograd.Function):
            @staticmethod
            # 前向传播函数的实现
            def forward(ctx, x):
                # 克隆输入张量为两个输出
                out0 = x.clone()
                out1 = x.clone()
                # 标记out1为不可微分
                ctx.mark_non_differentiable(out1)
                # 设置不生成非可微分部分的梯度
                ctx._materialize_non_diff_grads = False
                return out0, out1

            @staticmethod
            # 反向传播函数的实现
            def backward(ctx, g0, g1):
                # 断言g1梯度应为None
                self.assertIsNone(g1)
                return g0

        a = torch.tensor(1.0, requires_grad=True)
        # 使用自定义的Func进行前向传播和反向传播
        out = Func.apply(a)[0]
        out.backward()

    def test_legacy_function_deprecation_exception(self):
        # 触发异常测试
        class MyFunction(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output

        # 检查是否触发了异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Legacy autograd function with non-static forward method is deprecated",
        ):
            # 创建MyFunction实例并传入参数进行计算
            MyFunction()(torch.randn(3, 4))

    class SimulateBackwardError(Function):
        @staticmethod
        # 前向传播函数的实现
        def forward(ctx, input):
            return input.clone()

        @staticmethod
        @once_differentiable
        # 反向传播函数的实现
        def backward(ctx, input):
            # 模拟反向传播时的错误
            raise Exception("Simulate error on backward pass")  # noqa: TRY002

    def test_custom_function_exception(self):
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)

        tmp = (t1 + t2) * (t1 + t2)
        # 使用SimulateBackwardError自定义函数处理tmp
        t3 = TestAutograd.SimulateBackwardError.apply(tmp)
        # 断言在计算t3的梯度时会抛出异常
        with self.assertRaisesRegex(Exception, "Simulate error on backward pass"):
            t3.sum().backward()
    def test_custom_function_non_tensor_inputs_outputs(self):
        # 定义一个自定义的 PyTorch 函数类 MyFunction，继承自 Function 类
        class MyFunction(Function):
            # 静态方法：前向传播函数
            @staticmethod
            def forward(ctx, t1, t2, scale, t3):
                # 计算 t4 和 t5
                t4 = t1 + t2 * t3
                t5 = t1 * t2 + t3
                # 缩放 t4 和 t5
                t4 *= scale
                t5 *= scale

                # 保存 scale 到上下文对象 ctx
                ctx.scale = scale
                # 保存 t1, t2, t3 到上下文对象 ctx，以备反向传播使用
                ctx.save_for_backward(t1, t2, t3)
                # 返回多个值作为前向传播结果
                return scale, t4, None, True, t5, "bar", t1

            # 静态方法：反向传播函数
            @staticmethod
            @once_differentiable
            def backward(ctx, *grads):
                # 验证梯度数目和是否为 None
                self.assertEqual(7, len(grads))
                self.assertIsNone(grads[0])
                self.assertIsNone(grads[2])
                self.assertIsNone(grads[3])
                self.assertIsNone(grads[5])

                # 从上下文对象 ctx 中恢复保存的值
                scale = ctx.scale
                var1, var2, var3 = ctx.saved_tensors
                # 返回计算得到的梯度值
                return (
                    grads[1] * scale + grads[4] * var2 * scale + grads[6],
                    grads[1] * var3 * scale + grads[4] * var1 * scale,
                    None,
                    grads[1] * var2 * scale + grads[4] * scale,
                )

        # 生成随机张量作为输入
        t1 = torch.rand(10, dtype=torch.double, requires_grad=True)
        t2 = torch.rand(10, dtype=torch.double, requires_grad=True)
        t3 = torch.rand(10, dtype=torch.double)
        scale = random.randint(0, 10)
        # 调用自定义函数 MyFunction 的 apply 方法进行前向传播计算
        res = MyFunction.apply(t1, t2, scale, t3)
        # 验证前向传播结果
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

        # 运行反向传播以计算梯度
        torch.autograd.backward([res[1].sum(), res[4].sum(), res[6].sum()])
        # 验证梯度是否计算成功
        self.assertIsNotNone(t1.grad)
        self.assertIsNotNone(t2.grad)
        self.assertIsNone(t3.grad)

        # 测试梯度检查函数
        def foo(t1, t2, t3):
            res = MyFunction.apply(t1, t2, scale, t3)
            return res[1], res[4], res[6]

        gradcheck(foo, (t1, t2, t3))
    def test_custom_function_no_tensors(self):
        class MyFunction(Function):
            @staticmethod
            # 定义前向传播函数，接受四个张量作为输入参数
            def forward(ctx, t1, t2, scale, t3):
                # 计算 t1 + t2 * t3 并赋值给 t4
                t4 = t1 + t2 * t3
                # 计算 t1 * t2 + t3 并赋值给 t5
                t5 = t1 * t2 + t3
                # t4 乘以 scale
                t4 *= scale
                # t5 乘以 scale
                t5 *= scale
                # 返回结果，包括 scale, t4, None, True, t5, "bar", t1
                return scale, t4, None, True, t5, "bar", t1

            @staticmethod
            @once_differentiable
            # 定义反向传播函数，接受任意数量的参数
            def backward(ctx, *args):
                # 返回前向传播函数中的部分参数作为梯度
                return (args[0], args[1], None, args[2])

        # 随机生成 t1, t2, t3 和 scale
        t1 = random.random()
        t2 = random.random()
        t3 = random.random()
        scale = random.randint(0, 10)
        # 调用自定义函数 MyFunction 的前向传播，并传入 t1, t2, scale, t3
        res = MyFunction.apply(t1, t2, scale, t3)
        # 断言：确保返回的结果与预期的 scale 相等
        self.assertEqual(scale, res[0])
        # 断言：确保返回的结果中 t1 + t2 * t3 * scale 与预期相等
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        # 断言：确保返回的结果中第三个元素为 None
        self.assertEqual(None, res[2])
        # 断言：确保返回的结果中第四个元素为 True
        self.assertEqual(True, res[3])
        # 断言：确保返回的结果中 t1 * t2 + t3 * scale 与预期相等
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        # 断言：确保返回的结果中第六个元素为 "bar"
        self.assertEqual("bar", res[5])
        # 断言：确保返回的结果中第七个元素与输入的 t1 相等
        self.assertEqual(t1, res[6])

    def test_invalid_gradients(self):
        class MyFunction(Function):
            @staticmethod
            # 定义前向传播函数，接受一个张量作为输入参数
            def forward(ctx, x):
                # 返回输入张量的两倍
                return x * 2

            @staticmethod
            # 定义反向传播函数，接受一个梯度输出
            def backward(ctx, grad_output):
                # 返回一个形状为 (10,) 的随机张量作为梯度
                return torch.randn(10, dtype=torch.float)

        # 使用断言确保在执行梯度计算时引发 RuntimeError，并包含特定错误信息
        with self.assertRaisesRegex(RuntimeError, "expected shape"):
            input = torch.randn(5, 5, dtype=torch.float, requires_grad=True)
            # 调用自定义函数 MyFunction 的前向传播，然后对结果求和并进行反向传播
            MyFunction.apply(input).sum().backward()

    def test_unrelated_inputs(self):
        # 测试以确保 gradcheck 在存在不相关但可微分的输入时运行成功

        def my_function(x, y):
            # 计算 x 的平方
            return x * x

        # 创建两个形状为 (10,) 的双精度张量，并声明需要梯度计算
        x = torch.rand(10, dtype=torch.double, requires_grad=True)
        y = torch.rand(10, dtype=torch.double, requires_grad=True)

        # 调用 gradcheck 和 gradgradcheck 来检查 my_function 在 x, y 上的梯度计算
        gradcheck(my_function, (x, y))
        gradgradcheck(my_function, (x, y))

    def test_not_implemented_grad(self):
        # 创建一个形状为 (2,) 的随机张量，并声明需要梯度计算
        a = torch.rand(2, requires_grad=True)
        # 使用 torch.nextafter 对 a 进行操作，并对结果进行求和
        y = torch.nextafter(a, a).sum()
        # 使用断言确保在执行反向传播时引发 NotImplementedError，并包含特定错误信息
        with self.assertRaisesRegex(
            NotImplementedError, "the derivative for .* is not implemented"
        ):
            y.backward()

    def test_not_implemented_fwad(self):
        # 创建一个形状为 (3,) 的随机张量
        x = torch.randn(3)
        # 创建一个形状为 (3,) 的随机张量
        v = torch.rand(3)

        # 使用 forward AD 模块
        with fwAD.dual_level():
            # 将 x, v 转换为支持 forward AD 的双重张量
            dual_x = fwAD.make_dual(x, v)

            # 定义错误信息和提示信息
            err_msg = r"Trying to use forward AD with .* that does not support it"
            hint_msg = "Running forward AD for an OP that does not implement it should raise a NotImplementedError"

            # 使用断言确保在执行 torch.igamma 时引发 NotImplementedError，并包含特定错误信息
            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                torch.igamma(dual_x, dual_x)
    # 定义一个测试方法，测试自定义函数在执行过程中是否按预期执行
    def test_will_engine_execute_node(self):
        # 计数器，用于记录函数调用次数
        counter = [0]

        # 定义一个自定义函数类 MyFunction，继承自 torch.autograd.Function
        class MyFunction(Function):
            @staticmethod
            # 前向传播函数，对输入 x 执行乘以 2 的操作
            def forward(ctx, x):
                return x * 2

            @staticmethod
            # 反向传播函数，对梯度 gO 执行乘以 2 的操作
            def backward(ctx, gO):
                return gO * 2

        # 定义一个获取梯度函数的辅助函数
        def get_grad_fn(t):
            # 如果张量 t 需要梯度且没有梯度函数，则返回 t 的梯度函数的下一个函数
            if t.requires_grad and t.grad_fn is None:
                return t.clone().grad_fn.next_functions[0][0]
            else:
                return t.grad_fn

        # 创建两个随机张量 a 和 a2，均为 2x3x4 的形状，并且需要梯度计算
        a = torch.randn(2, 3, 4, requires_grad=True)
        a2 = torch.randn(2, 3, 4, requires_grad=True)
        # 计算张量 b，为 a 和 a2 的逐元素乘积
        b = a * a2
        # 计算张量 b2，为 b 的余弦值
        b2 = b.cos()
        # 计算张量 c，通过自定义函数 MyFunction 的 apply 方法对 b 执行操作
        c = MyFunction.apply(b)

        # 获取应该执行和不应该执行的梯度函数列表
        should_execute = list(map(get_grad_fn, (a, b, c)))
        should_not_execute = list(map(get_grad_fn, (a2, b2)))

        # 定义一个函数 fn，用作注册的钩子函数
        def fn(x):
            # 计数器加一
            counter[0] += 1

            # 对于应该执行的梯度函数列表中的每个函数 g，断言 torch._C._will_engine_execute_node(g) 返回 True
            for g in should_execute:
                self.assertTrue(torch._C._will_engine_execute_node(g))

            # 对于不应该执行的梯度函数列表中的每个函数 g，断言 torch._C._will_engine_execute_node(g) 返回 False
            for g in should_not_execute:
                self.assertFalse(torch._C._will_engine_execute_node(g))

        # 注册钩子函数 fn 到张量 b 和 c
        b.register_hook(fn)
        c.register_hook(fn)

        # 对 c 的和进行反向传播计算，同时指定输入为 a 和 b，并保留计算图
        out = c.sum()
        torch.autograd.backward(out, inputs=(a, b), retain_graph=True)
        # 断言计数器的值为 2，表示 fn 函数被调用了两次
        self.assertEqual(counter[0], 2)

        # 更新应该执行和不应该执行的梯度函数列表
        should_execute = list(map(get_grad_fn, (a, a2, b, c)))
        should_not_execute = list(map(get_grad_fn, (b2,)))
        # 再次进行反向传播计算，但此时不指定输入，仅保留计算图
        torch.autograd.backward(out, retain_graph=True)

        # 当传入的张量是叶子节点时，调用 torch.autograd.grad(out, (a,)) 会引发 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError, "are currently running autograd.grad()"
        ):
            torch.autograd.grad(out, (a,))

        # 当传入的张量不是叶子节点时，调用 torch.autograd.grad(b.sum(), (a,)) 应该正常执行
        a = torch.randn(1, 2, 3, requires_grad=True) * 2
        b = a * 2

        # 定义一个新的钩子函数 fn，用于检查非叶子节点 b 的梯度函数
        def fn(x):
            # 检查非叶子节点 b 的梯度函数是否能被执行
            counter[0] += 1
            self.assertTrue(torch._C._will_engine_execute_node(b.grad_fn))

        # 注册钩子函数 fn 到张量 b
        b.register_hook(fn)
        counter[0] = 0
        # 对 b 的和进行反向传播计算，指定输入为 a，保留计算图
        torch.autograd.grad(b.sum(), (a,))
        # 断言计数器的值为 1，表示 fn 函数被调用了一次
        self.assertEqual(counter[0], 1)

        # 验证当尝试访问 out 的梯度函数时，会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "during the backward pass"):
            torch._C._will_engine_execute_node(out.grad_fn)

        # 验证当尝试访问 out 本身的梯度函数时，会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "expects an grad_fn"):
            torch._C._will_engine_execute_node(out)
    def test_custom_function_vmap_defaults(self):
        # 定义一个自定义的 Torch 函数 MySquare，继承自 Function 类
        class MySquare(Function):
            @staticmethod
            # 前向传播函数，计算输入的平方
            def forward(x):
                return x**2

            @staticmethod
            # 设置上下文函数，保存输入张量 x 到上下文中
            def setup_context(ctx, inputs, output):
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            # 反向传播函数，计算梯度，这里返回对输入 x 的梯度
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                return gO * 2 * x

        # 断言 MySquare 类的 generate_vmap_rule 属性为 False
        self.assertFalse(MySquare.generate_vmap_rule)
        # 断言 MySquare 类包含 vmap 方法
        self.assertTrue(hasattr(MySquare, "vmap"))

    def test_custom_function_setup_context_simple(self):
        # 定义一个自定义的 Torch 函数 MySquare，继承自 Function 类
        class MySquare(Function):
            @staticmethod
            # 前向传播函数，计算输入的平方
            def forward(x):
                return x**2

            @staticmethod
            # 设置上下文函数，保存输入张量 x 到上下文中
            def setup_context(ctx, inputs, output):
                (x,) = inputs
                ctx.save_for_backward(x)

            @staticmethod
            # 反向传播函数，计算梯度，这里返回对输入 x 的梯度
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                return gO * 2 * x

        # 生成一个随机张量 x，并设置其需要梯度计算
        x = torch.randn([], requires_grad=True)
        # 调用 MySquare 的 apply 方法，对 x 进行计算得到 y
        y = MySquare.apply(x)
        # 使用 autograd.grad 计算 y 对 x 的梯度
        (gx,) = torch.autograd.grad(y, x)
        # 断言计算得到的梯度 gx 等于 2 * x
        self.assertEqual(gx, 2 * x)

    def test_custom_function_setup_context_multi_output(self):
        # 定义一个自定义的 Torch 函数 MySquare，继承自 Function 类
        # 此函数有多个输出，包括一个非张量的输出
        class MySquare(Function):
            @staticmethod
            # 前向传播函数，计算输入的平方，并额外返回输入的两倍
            def forward(x):
                two_x = x.item() * 2
                return x**2, two_x

            @staticmethod
            # 设置上下文函数，保存输入张量 x 和输出中的两倍值到上下文中
            def setup_context(ctx, inputs, output):
                (x,) = inputs
                _, two_x = output
                ctx.two_x = two_x

            @staticmethod
            # 反向传播函数，计算梯度，这里返回 gO 乘以保存在上下文中的两倍值
            @once_differentiable
            def backward(ctx, gO, _):
                return gO * ctx.two_x

        # 生成一个随机张量 x，并设置其需要梯度计算
        x = torch.randn([], requires_grad=True)
        # 调用 MySquare 的 apply 方法，对 x 进行计算得到 y，并忽略第二个输出
        y, _ = MySquare.apply(x)
        # 使用 autograd.grad 计算 y 对 x 的梯度
        (gx,) = torch.autograd.grad(y, x)
        # 断言计算得到的梯度 gx 等于 2 * x
        self.assertEqual(gx, 2 * x)
    def test_custom_function_setup_context_multi_input(self):
        class MyReshape(Function):
            @staticmethod
            def forward(x, shape, scale_forward, scale_backward):
                # 前向传播：对输入张量 x 进行形状重塑，并乘以 scale_forward
                return x.reshape(shape) * scale_forward

            @staticmethod
            def setup_context(ctx, inputs, output):
                # 设置上下文：从输入中提取参数并保存到上下文对象 ctx 中
                x, shape, scale_forward, scale_backward = inputs
                ctx.scale_backward = scale_backward
                ctx.x_shape = x.shape

            @staticmethod
            def backward(ctx, gO):
                # 反向传播：根据保存在上下文中的信息计算梯度
                return gO.reshape(ctx.x_shape) * ctx.scale_backward, None, None, None

        class MyReshapeRef(Function):
            @staticmethod
            def forward(ctx, x, shape, scale_forward, scale_backward):
                # 前向传播：对输入张量 x 进行形状重塑，并乘以 scale_forward，并在上下文中保存相关信息
                ctx.scale_backward = scale_backward
                ctx.x_shape = x.shape
                return x.reshape(shape) * scale_forward

            @staticmethod
            def backward(ctx, gO):
                # 反向传播：根据保存在上下文中的信息计算梯度
                return gO.reshape(ctx.x_shape) * ctx.scale_backward, None, None, None

        def test(x, shape, scale_forward, scale_backward):
            # 测试函数：使用自定义的 MyReshape 和 MyReshapeRef 函数进行前向和后向传播测试
            y = MyReshape.apply(x, shape, scale_forward, scale_backward).sum()
            (gx,) = torch.autograd.grad(y, x)

            y_expected = MyReshapeRef.apply(
                x, shape, scale_forward, scale_backward
            ).sum()
            (gx_expected,) = torch.autograd.grad(y_expected, x)

            self.assertEqual(y_expected, y)
            self.assertEqual(gx_expected, gx)

        # 对两组输入进行测试
        test(torch.randn(24, requires_grad=True), (3, 8), 7, 11)
        test(torch.randn(2, 3, 4, requires_grad=True), (6, 4), -1, 2)

    def test_accumulate_grad(self):
        # 测试累积梯度的函数
        grad_output = torch.ones(5, 5)

        def compute_grad(create_graph):
            # 计算梯度的内部函数，接受 create_graph 参数用于控制是否创建计算图
            x = torch.randn(5, 5, requires_grad=True)
            y = x + 2
            y.backward(grad_output, retain_graph=True)  # 计算梯度并保留计算图
            x_grad = x.grad
            x_grad_clone = x.grad.clone()
            y.backward(grad_output, create_graph=create_graph)  # 再次计算梯度，根据 create_graph 参数决定是否创建新的计算图
            return x_grad, x_grad_clone

        # 当 create_graph=False 时，测试原地累积梯度
        x_grad, x_grad_clone = compute_grad(create_graph=False)
        self.assertEqual(x_grad, x_grad_clone * 2)

        # 当 create_graph=True 时，测试非原地累积梯度
        x_grad, x_grad_clone = compute_grad(create_graph=True)
        self.assertEqual(x_grad, x_grad_clone)
    def test_accumulate_grad_tensor_reference(self):
        # 定义内部函数 `_test_grad_tensor`，用于测试梯度张量引用的行为
        def _test_grad_tensor(
            params_grad_tensor,
            backward_grad_tensor,
            should_preserve_reference,
            create_graph,
        ):
            # 创建一个包含两个元素的张量 `params`，并要求其梯度计算
            params = torch.tensor([1.5, 1.5]).requires_grad_()
            # 将 `params_grad_tensor` 赋值给 `params` 的梯度张量
            params.grad = params_grad_tensor
            # 保存 `params.grad` 的引用
            grad_saved = params.grad
            # 对 `params` 进行反向传播，使用 `backward_grad_tensor` 作为梯度
            params.backward(backward_grad_tensor, create_graph=create_graph)
            # 断言 `params.grad` 的引用是否保持不变
            self.assertEqual(
                id(grad_saved) == id(params.grad), should_preserve_reference
            )

        # 循环测试两种情况：`create_graph` 为 False 和 True
        for create_graph in (False, True):
            # 测试稀疏梯度累积到稠密梯度是否会改变 `params.grad` 的引用
            _test_grad_tensor(
                torch.sparse_coo_tensor(
                    torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])
                ),
                torch.tensor([1.5, 1.5]),
                False,  # 永远不会原地累积
                create_graph,
            )

            # 测试稠密梯度累积到稠密梯度是否会保持 `params.grad` 的引用，但仅当 `create_graph=False` 时
            _test_grad_tensor(
                torch.tensor([1.5, 1.5]),
                torch.tensor([1.5, 1.5]),
                not create_graph,
                create_graph,
            )

            # 测试稀疏梯度累积到稀疏梯度是否会保持 `params.grad` 的引用，但仅当 `create_graph=False` 时
            _test_grad_tensor(
                torch.sparse_coo_tensor(
                    torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])
                ),
                torch.sparse_coo_tensor(
                    torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])
                ),
                not create_graph,
                create_graph,
            )

    def test_accumulate_grad_with_zero_numel_grad(self):
        # 创建具有零元素梯度的张量 `a` 和 `b`
        a = torch.rand(4, 0, requires_grad=True)
        b = torch.rand(4, 1, requires_grad=True)
        # 计算张量 `c` 作为 `a` 和 `b` 的和，并断言其形状为 (4, 0)
        c = a + b
        assert c.shape == (4, 0)
        # 对 `c` 求和并进行反向传播
        c.sum().backward()

        # 断言 `b` 的梯度为全零张量
        self.assertEqual(b.grad, torch.zeros(4, 1))
        # 断言 `a` 的梯度为全零张量
        self.assertEqual(a.grad, torch.zeros(4, 0))

    def test_hessian_vector(self):
        # 创建两个具有随机值并要求梯度的张量 `x` 和 `y`
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)

        # 定义张量 `z`，进行数学运算并求其对张量的一阶导数
        z = x**2 + y * x + y**2
        z.backward(torch.ones(2, 2), create_graph=True)

        # 在没有梯度更新的情况下，计算 `x` 和 `y` 的梯度
        with torch.no_grad():
            x_grad = 2 * x + y
            y_grad = x + 2 * y
        # 断言 `x` 的梯度等于预期的一阶导数
        self.assertEqual(x.grad, x_grad)
        # 断言 `y` 的梯度等于预期的一阶导数
        self.assertEqual(y.grad, y_grad)

        # 计算 `x` 和 `y` 的梯度和，并进行反向传播
        grad_sum = 2 * x.grad + y.grad
        grad_sum.backward(torch.ones(2, 2))
        # 定义 `x_hv` 和 `y_hv` 作为 Hessian-Vector 乘积的结果
        x_hv = torch.ones(2, 2) * 5
        y_hv = torch.ones(2, 2) * 4
        # 断言 `x` 的梯度等于预期的一阶导数加上 Hessian-Vector 乘积的结果
        self.assertEqual(x.grad, x_grad + x_hv)
        # 断言 `y` 的梯度等于预期的一阶导数加上 Hessian-Vector 乘积的结果
        self.assertEqual(y.grad, y_grad + y_hv)
    # 定义一个测试函数，用于验证梯度计算和自动微分的功能
    def test_grad(self):
        # 创建两个随机张量，需要计算梯度
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)
        # 定义一个复杂的张量操作
        z = x**2 + y * x + y**2
        # 对张量 z 进行反向传播，计算梯度，并且创建一个计算图
        z.backward(torch.ones(2, 2), create_graph=True)

        # 计算期望的梯度
        x_grad = 2 * x + y
        y_grad = x + 2 * y
        # 验证计算得到的梯度与期望的梯度是否相等
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        # 计算二阶导数的一部分
        grad_sum = 2 * x.grad + y.grad
        x_hv = torch.autograd.grad(
            outputs=[grad_sum],
            grad_outputs=[torch.ones(2, 2)],
            inputs=[x],
            create_graph=True,
        )
        # 期望的二阶导数值
        expected_x_hv = torch.ones(2, 2) * 5
        expected_y_hv = torch.ones(2, 2) * 4

        # 验证计算得到的二阶导数与期望的二阶导数是否相等
        self.assertEqual(x_hv[0], expected_x_hv)
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

        # 测试 grad_outputs 和 outputs 的形状是否匹配
        grad_out = torch.ones(2)
        try:
            torch.autograd.grad(
                outputs=[grad_sum],
                grad_outputs=[grad_out],
                inputs=[x],
                create_graph=True,
            )
            # 如果形状不匹配，应该抛出 RuntimeError 异常
            self.assertFail()
        except RuntimeError as error:
            self.assertEqual(
                str(error),
                "Mismatch in shape: grad_output[0] has a shape of "
                + str(grad_out.shape)
                + " and output[0] has a shape of "
                + str(grad_sum.shape)
                + ".",
            )

    # 测试自动微分图中节点的匹配性
    def test_grad_to_node(self):
        def check_matches(out, inp):
            # 计算参考的梯度
            ref = torch.autograd.grad(out.sum(), inp)

            # 获取输入张量的梯度边缘
            edge = torch.autograd.graph.get_gradient_edge(inp)
            # 计算新的梯度
            new = torch.autograd.grad(out.sum(), edge)
            # 验证参考梯度和新计算梯度是否匹配
            self.assertEqual(ref, new)

        # 测试不同类型的节点工作情况（常规 cpp 节点，累积梯度节点和自定义函数）
        x = torch.rand(2, requires_grad=True)
        out = x.clone()
        check_matches(out, x)

        x = x.clone()
        out = x.clone()
        check_matches(out, x)

        x = torch.autograd._functions.Resize.apply(x, (2,))
        out = x.clone()
        check_matches(out, x)

        x = torch.var_mean(x)[1]
        out = x.clone()
        check_matches(out, x)

    # 测试自动微分图中节点集的情况
    def test_grad_to_node_set(self):
        x = torch.rand(2, requires_grad=True)
        x_edge = torch.autograd.graph.get_gradient_edge(x)
        out = x.clone()

        # 在没有梯度追踪的情况下修改张量 x
        with torch.no_grad():
            x.set_(torch.rand_like(x))

        # 预期抛出 RuntimeError 异常，因为 x 已经在计算图中使用过
        with self.assertRaisesRegex(RuntimeError, "to not have been used in the graph"):
            torch.autograd.grad(out.sum(), x)

        # 正常情况下，应该可以计算梯度
        torch.autograd.grad(out.sum(), x_edge)
    # 测试函数：测试在原地操作时梯度传播到节点
    def test_grad_to_node_inplace(self):
        # 创建一个形状为 (2,) 的张量 x，并设置 requires_grad=True，使其可以计算梯度
        x = torch.rand(2, requires_grad=True).clone()
        # 获取张量 x 的梯度边缘信息
        x_edge = torch.autograd.graph.get_gradient_edge(x)
        # 将张量 x 的值乘以 2，进行原地操作
        x *= 2

        # 计算 out = x.sum() 对 x_edge 和 x 的梯度
        g_old, g_new = torch.autograd.grad(x.sum(), (x_edge, x))
        # 断言 g_old 的梯度应为 2 倍的 x 的全 1 张量
        self.assertEqual(g_old, 2 * torch.ones_like(x))
        # 断言 g_new 的梯度应为 x 的全 1 张量
        self.assertEqual(g_new, torch.ones_like(x))

    # 测试函数：测试多个节点上的梯度传播
    def test_grad_to_node_multi(self):
        # 创建两个形状为 (2,) 的张量 x 和 y，并设置 requires_grad=True，使其可以计算梯度
        x = torch.rand(2, requires_grad=True).clone()
        y = torch.rand(2, requires_grad=True).clone()

        # 计算张量 x 和 y 的和
        out = x + y

        # 计算 out.sum() 对 x 和 y 的梯度
        ref = torch.autograd.grad(out.sum(), (x, y))

        # 创建输入边缘 inp_edges，分别为 x 和 y 的梯度边缘信息
        inp_edges = (
            GradientEdge(x.grad_fn, x.output_nr),
            GradientEdge(y.grad_fn, y.output_nr),
        )
        # 计算 out.sum() 对 inp_edges 的梯度
        new = torch.autograd.grad(out.sum(), inp_edges)

        # 断言 ref 和 new 的梯度相等
        self.assertEqual(ref, new)

    # 测试函数：测试在梯度材料化时梯度传播到节点
    def test_grad_to_node_materialize(self):
        # 创建形状为 (2,) 的张量 x 和 y，并设置 requires_grad=True，使其可以计算梯度
        x = torch.rand(2, requires_grad=True).clone()
        edge_x = GradientEdge(x.grad_fn, x.output_nr)
        y = torch.rand(2, requires_grad=True).clone()
        edge_y = GradientEdge(y.grad_fn, y.output_nr)

        # 将张量 x 的克隆赋值给 out
        out = x.clone()

        # 测试 materialize_grads=True 的情况
        torch.autograd.grad(
            out.sum(), (edge_x, y), allow_unused=True, materialize_grads=True
        )
        torch.autograd.grad(
            out.sum(), (x, y), allow_unused=True, materialize_grads=True
        )
        torch.autograd.grad(out.sum(), (x, edge_y), allow_unused=True)

        # 断言在给定输入为 GradientEdge 时，使用 materialize_grads 会抛出 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError,
            "materialize_grads cannot be used when the given input is a GradientEdge",
        ):
            torch.autograd.grad(
                out.sum(), (x, edge_y), allow_unused=True, materialize_grads=True
            )

    # 测试函数：测试反向传播到节点
    def test_backward_to_node(self):
        # 创建形状为 (2,) 的张量 x 和 y，并设置 requires_grad=True，使其可以计算梯度
        x = torch.rand(2, requires_grad=True).clone()
        edge_x = GradientEdge(x.grad_fn, x.output_nr)
        y = torch.rand(2, requires_grad=True).clone()
        edge_y = GradientEdge(y.grad_fn, y.output_nr)

        # 将张量 x 的克隆赋值给 out
        out = x.clone()

        # 测试不同输入情况下的反向传播
        torch.autograd.backward(out.sum(), inputs=(edge_x, y))
        torch.autograd.backward(out.sum(), inputs=(x, y))
        torch.autograd.backward(out.sum(), inputs=(x, edge_y))
        torch.autograd.backward(out.sum(), inputs=(edge_x, edge_y))
    # 定义一个测试函数，用于测试非叶子张量的梯度计算
    def test_grad_nonleaf(self):
        # 创建一个形状为 (2, 2) 的张量，要求计算其梯度
        x_init = torch.randn(2, 2, requires_grad=True)
        # 将 x_init 赋值给 x
        x = x_init
        # 创建一个形状为 (2, 2) 的张量 y，并要求计算其梯度
        y = torch.randn(2, 2, requires_grad=True)
        # 创建一个形状为 (2, 2) 的张量 grad_output，其值全为 1，用于梯度计算
        grad_output = torch.ones(2, 2)

        # 定义一个函数 fn，接受一个张量 x，并返回 x 的平方加上 y 与 x 的乘积再加上 y 的平方
        def fn(x):
            return x**2 + y * x + y**2

        # 循环5次
        for _ in range(5):
            # 计算 fn(x) 对 x 的梯度，同时创建计算图以便二阶导数的计算
            (grad_x,) = torch.autograd.grad(
                fn(x), x, grad_outputs=grad_output, create_graph=True
            )

            # 计算预期的 grad_x
            grad_x_expected = 2 * x + y
            # 断言 y 的梯度为 None
            self.assertIsNone(y.grad)
            # 断言 x 的梯度为 None
            self.assertIsNone(x.grad)
            # 断言计算得到的 grad_x 与预期的 grad_x 相等
            self.assertEqual(grad_x, grad_x_expected)

            # 更新 x 的值，使用梯度下降法
            x = x + 0.05 * grad_x

        # 计算初始状态下和最终状态下 fn(x) 的总和，并进行断言判断最终状态下总和大于初始状态下的总和
        val_init = fn(x_init).sum()
        val_final = fn(x).sum()
        self.assertGreater(val_final, val_init)

        # 对 x 进行反向传播，计算所有梯度
        x.backward(grad_output)
        # 断言 y 的梯度不为 None
        self.assertIsNotNone(y.grad)
        # 断言 x_init 的梯度不为 None
        self.assertIsNotNone(x_init.grad)

    # 定义一个测试函数，用于测试非叶子张量并输出多个梯度的情况
    def test_grad_nonleaf_many_outputs(self):
        # 这里检查一个函数回调的边缘情况
        # 我们希望捕获一个函数的两个梯度，但只能注册一个回调函数。
        x = torch.randn(4, 2, requires_grad=True)
        # 将 x 分割成两部分，分别命名为 a 和 b
        a, b = x.chunk(2)

        # 定义一个回调函数 hook，用于捕获梯度
        def hook(*grads):
            hook_called[0] = True

        # 初始化一个标志位用于判断回调函数是否被调用
        hook_called = [False]
        # 注册 hook 函数到 x 上
        x.register_hook(hook)

        # 创建形状为 (2, 2) 的张量 go，用作梯度的输出
        go = torch.randn(2, 2)
        # 计算 (a + 2 * b) 对 a 和 b 的梯度，并创建计算图以便二阶导数的计算
        grad_a, grad_b = torch.autograd.grad(
            (a + 2 * b), [a, b], grad_outputs=go, create_graph=True
        )

        # 断言 grad_a 的值与 go 相等
        self.assertEqual(grad_a, go)
        # 断言 grad_b 的值与 go * 2 相等
        self.assertEqual(grad_b, go * 2)
        # 断言 hook_called[0] 为 False，即 hook 函数未被调用
        self.assertFalse(hook_called[0])
        # 断言 x 的梯度为 None
        self.assertIsNone(x.grad)

    # 定义一个测试函数，用于测试 register_hook 的边缘情况
    def test_grad_nonleaf_register_hook(self):
        # 这里检查 register_hook 的一个边缘情况
        # 我们希望捕获一个非叶子张量的梯度，
        # 但在反向传播其他非叶子张量时要避免段错误（segfault）。
        x = torch.randn(5, requires_grad=True)
        # 将 x 拆分为一个张量列表 x_list
        x_list = x.unbind()

        # 获取 x_list 中的第一个张量 x0
        x0 = x_list[0]
        # 定义一个列表 hook_results，用于存储 hook 函数的结果
        hook_results = [None]

        # 定义一个 hook 函数，用于捕获梯度
        def hook(grad):
            hook_results[0] = grad

        # 将 hook 函数注册到 x0 上
        x0.register_hook(hook)

        # 对 x_list[0] 进行反向传播
        x_list[0].backward()
        # 断言 hook_results[0] 的值为 tensor([1.0])
        self.assertEqual(hook_results[0], torch.tensor(1.0))
        # 预期的梯度值
        expected_grad = torch.tensor([1.0, 0, 0, 0, 0])
        # 断言 x 的梯度等于 expected_grad
        self.assertEqual(x.grad, expected_grad)
        # 断言 x_list[0] 的梯度为 None
        self.assertIsNone(x_list[0].grad)

        # 循环处理 x_list 中的其它张量
        for i in range(1, 5, 1):
            # 对 x_list[i] 进行反向传播
            x_list[i].backward()
            # 断言 hook_results[0] 的值为 None
            self.assertEqual(hook_results[0], None)
            # 更新 expected_grad 的值
            expected_grad[i] = 1.0
            # 断言 x 的梯度等于 expected_grad
            self.assertEqual(x.grad, expected_grad)
            # 断言 x_list[i] 的梯度为 None
            self.assertIsNone(x_list[i].grad)
    # 测试梯度材料化和自动求导功能
    def test_grad_materialize_grads(self):
        # 创建一个张量 x，指定需要计算其梯度
        x = torch.tensor(0.5, requires_grad=True)
        # 创建一个张量 a，指定需要计算其梯度
        a = torch.tensor(1.0, requires_grad=True)
        # 计算张量 y = x * a
        y = x * a
        # 计算 dy/dx，并创建一个允许构建更高阶导数图的上下文
        dydx = torch.autograd.grad(y, x, create_graph=True)
        # 计算 d^2y/dx^2，并允许梯度材料化
        d2ydx2_none = torch.autograd.grad(dydx, x, create_graph=True, allow_unused=True)
        d2ydx2 = torch.autograd.grad(
            dydx, x, create_graph=True, allow_unused=True, materialize_grads=True
        )
        # 计算 d^3y/dx^3，并允许梯度材料化
        d3ydx3 = torch.autograd.grad(d2ydx2, x, materialize_grads=True)
        # 断言 d2ydx2_none 的值应为 None
        self.assertIsNone(d2ydx2_none[0])
        # 断言 d2ydx2 的值应为 0
        self.assertEqual(d2ydx2[0].item(), 0)
        # 断言 d3ydx3 的值应为 0
        self.assertEqual(d3ydx3[0].item(), 0)
        # 使用断言捕获预期的 ValueError 异常，验证 allow_unused 设置为 False 时出错
        with self.assertRaisesRegex(
            ValueError, "Expected allow_unused to be True or not passed when"
        ):
            torch.autograd.grad(y, x, allow_unused=False, materialize_grads=True)

    # 测试在非叶节点上注册后累积梯度钩子的行为
    def test_post_accumulate_grad_hook_on_non_leaf(self):
        # 定义一个钩子函数，将张量减去 1.0
        def hook(tensor):
            tensor.sub_(1.0)

        # 创建一个需要梯度的随机张量 leaf
        leaf = torch.rand(3, requires_grad=True)
        # 创建一个非叶节点张量 non_leaf，它是 leaf 的两倍
        non_leaf = 2.0 * leaf

        # 断言捕获预期的 RuntimeError 异常，验证无法在非叶节点上注册后累积梯度钩子
        with self.assertRaisesRegex(
            RuntimeError,
            "post accumulate grad hooks cannot be registered on non-leaf tensors",
        ):
            non_leaf.register_post_accumulate_grad_hook(hook)

    # 测试在单个张量上注册多个后累积梯度钩子的行为
    def test_post_accumulate_grad_hook_multiple_hooks(self):
        # 定义第一个钩子函数，将张量减去其梯度
        def hook1(tensor):
            tensor.sub_(tensor.grad)

        # 定义第二个钩子函数，将张量乘以 4.0
        def hook2(tensor):
            tensor.mul_(4.0)

        # 创建一个需要梯度的随机张量 tensor
        tensor = torch.rand(3, requires_grad=True)
        # 创建 tensor 的克隆副本 tensor_ref
        tensor_ref = tensor.clone().detach()
        # 分别注册 hook1 和 hook2 两个后累积梯度钩子到 tensor 上
        tensor.register_post_accumulate_grad_hook(hook1)
        tensor.register_post_accumulate_grad_hook(hook2)
        # 计算张量 tensor 的和，并反向传播梯度
        sum = tensor.sum()
        sum.backward()
        # 断言 tensor 的值应为 tensor_ref - 1.0 乘以 4.0
        self.assertEqual(4.0 * (tensor_ref - 1.0), tensor)

    # 测试在多个张量上注册后累积梯度钩子的行为
    def test_post_accumulate_grad_hook_multiple_tensors(self):
        # 定义一个钩子函数，将张量减去其梯度
        def hook(tensor):
            tensor.sub_(tensor.grad)

        # 创建两个需要梯度的随机张量 tensor1 和 tensor2
        tensor1 = torch.rand(3, requires_grad=True)
        tensor1_ref = tensor1.clone().detach()
        tensor2 = torch.rand(5, requires_grad=True)
        tensor2_ref = tensor2.clone().detach()
        # 分别注册 hook 钩子到 tensor1 和 tensor2 上
        tensor1.register_post_accumulate_grad_hook(hook)
        tensor2.register_post_accumulate_grad_hook(hook)
        # 计算张量 tensor1 和 tensor2 的和，并反向传播梯度
        tensor1.sum().backward()
        tensor2.sum().backward()
        # 断言 tensor1 和 tensor2 的值应为 tensor1_ref - 1.0 和 tensor2_ref - 1.0
        self.assertEqual(tensor1_ref - 1.0, tensor1)
        self.assertEqual(tensor2_ref - 1.0, tensor2)

    # 测试后累积梯度钩子返回值不是 None 的行为
    def test_post_accumulate_grad_hook_returns_not_None(self):
        # 定义一个错误的钩子函数，返回张量的梯度
        def bad_hook(tensor):
            return tensor.grad

        # 创建一个需要梯度的随机张量 tensor
        tensor = torch.rand(2, 3, requires_grad=True)
        # 注册 bad_hook 钩子到 tensor 上
        tensor.register_post_accumulate_grad_hook(bad_hook)
        # 断言捕获预期的 RuntimeError 异常，验证钩子函数应该返回 None
        with self.assertRaisesRegex(RuntimeError, "hooks should return None."):
            tensor.sum().backward()
    def test_post_accumulate_grad_hook_e2e(self):
        # 定义一个函数，用于在反向传播过程中设置优化器
        def setup_optim_in_bwd(model):
            optims = {}  # 创建一个空字典，用于存储不同参数对应的优化器
            handles = []  # 创建一个空列表，用于存储注册的钩子句柄

            # 定义一个优化步骤的钩子函数，用于在梯度累积后执行优化器的步骤和清空梯度
            def optim_step_hook(param):
                optims[param].step()  # 执行参数对应的优化器的步骤
                optims[param].zero_grad()  # 清空参数对应的优化器的梯度

            # 遍历模型的参数，为每个参数创建一个Adam优化器，并注册优化步骤钩子
            for p in model.parameters():
                optims[p] = torch.optim.Adam([p])  # 使用Adam优化器优化当前参数
                handles.append(p.register_post_accumulate_grad_hook(optim_step_hook))  # 注册优化步骤钩子

            return handles  # 返回所有注册的钩子句柄列表

        model = torch.nn.Linear(3, 2)  # 创建一个线性模型
        input = torch.rand(2, 3)  # 创建一个随机输入张量
        handles = setup_optim_in_bwd(model)  # 设置模型中参数的优化器，并获取钩子句柄列表

        # 复制模型和优化器的副本用于后续对比
        model_copy = deepcopy(model)  # 深度复制原始模型
        optim_copy = torch.optim.Adam(model_copy.parameters())  # 创建副本模型的Adam优化器

        iters = 5  # 定义迭代次数

        # 在迭代过程中执行优化器步骤和反向传播
        for _ in range(iters):
            loss = model(input).sum()  # 计算模型输出的损失值
            loss.backward()  # 执行反向传播

            loss_copy = model_copy(input).sum()  # 计算模型副本输出的损失值
            loss_copy.backward()  # 执行模型副本的反向传播
            optim_copy.step()  # 执行副本模型的优化步骤
            optim_copy.zero_grad()  # 清空副本模型的梯度

        params_copy = []  # 创建一个空列表，用于存储参数的副本

        # 比较模型和模型副本的参数，确保它们在每次迭代中保持一致
        for p_reference, p in zip(model_copy.parameters(), model.parameters()):
            self.assertEqual(p_reference, p)  # 断言模型和模型副本的参数相等
            params_copy.append(p_reference.clone().detach())  # 将模型副本的参数加入列表，确保其为不可变对象

        # 移除钩子句柄后，模型不应再更新
        for h in handles:
            h.remove()  # 移除每个钩子句柄

        # 再次迭代并验证模型是否保持不变
        for _ in range(iters):
            loss = model(input).sum()  # 计算模型输出的损失值
            loss.backward()  # 执行反向传播

            loss_copy = model_copy(input).sum()  # 计算模型副本输出的损失值
            loss_copy.backward()  # 执行模型副本的反向传播
            optim_copy.step()  # 执行副本模型的优化步骤
            optim_copy.zero_grad()  # 清空副本模型的梯度

        # 比较模型和模型副本的静态参数，确保它们在每次迭代中保持一致
        for p_static, p_reference, p in zip(
            params_copy, model_copy.parameters(), model.parameters()
        ):
            self.assertEqual(p_static, p)  # 断言模型和模型副本的静态参数相等
            self.assertNotEqual(p_reference, p)  # 断言模型和模型副本的参数不相等

    def test_post_accumulate_grad_hook_gets_cleaned_up(self):
        # 定义一个函数，用于测试钩子在垃圾回收后是否被正确清除
        def fun_stuff_with_hook():
            thing_to_put_in_hook = torch.rand(3)  # 创建一个张量用于在钩子中使用

            # 定义一个钩子函数，对张量进行操作
            def hook(tensor):
                tensor.sub_(tensor.grad)  # 张量减去其梯度
                tensor.add_(thing_to_put_in_hook)  # 添加预先定义的张量

            tensor = torch.rand(3, requires_grad=True)  # 创建一个需要梯度的随机张量
            tensor.register_post_accumulate_grad_hook(hook)  # 注册钩子函数到张量
            tensor.sum().backward()  # 计算张量的和并执行反向传播
            ref = weakref.ref(thing_to_put_in_hook)  # 创建弱引用检查张量是否被垃圾回收
            gc.collect()  # 手动触发垃圾回收
            return tensor, ref  # 返回张量和对thing_to_put_in_hook的弱引用

        with disable_gc():  # 禁用垃圾回收器
            tensor, ref = fun_stuff_with_hook()  # 执行钩子测试函数并获取返回结果
            self.assertIsNotNone(
                ref()
            )  # 断言thing_to_put_in_hook应该被tensor保持活跃

            del tensor  # 删除张量
            gc.collect()  # 手动触发垃圾回收
            self.assertIsNone(ref())  # 断言thing_to_put_in_hook应该被正确清理
    def test_post_accumulate_grad_hook_ordering(self):
        # 创建一个具有梯度的随机张量
        tensor = torch.rand(3, requires_grad=True)

        # 定义一个前置钩子函数，减去2.0
        def pre_hook(grad):
            return grad.sub(2.0)

        # 定义一个累积梯度节点的前置钩子函数，将梯度除以5.0
        def acc_grad_node_pre_hook(grad_out):
            return (grad_out[0].div(5.0),)

        # 定义一个张量后置累积梯度钩子函数，将梯度加上0.5
        def post_acc_grad_hook(tensor):
            tensor.grad.add_(0.5)

        # 定义一个累积梯度节点的后置钩子函数，将输出梯度乘以10
        def acc_grad_node_post_hook(grad_in, grad_out):
            tensor.grad = grad_out[0].mul(10)

        # 获取累积梯度节点，并注册前置钩子
        acc_grad = tensor.view_as(tensor).grad_fn.next_functions[0][0]
        tensor.register_hook(pre_hook)
        acc_grad.register_prehook(acc_grad_node_pre_hook)
        tensor.register_post_accumulate_grad_hook(post_acc_grad_hook)
        acc_grad.register_hook(acc_grad_node_post_hook)

        # 计算张量的和，并反向传播
        tensor.sum().backward()

        # 断言，钩子应按顺序运行
        #   1. 张量的前置钩子
        #   2. 累积梯度节点的前置钩子
        #   3. 张量的后置累积梯度钩子
        #   4. 累积梯度节点的后置钩子
        self.assertEqual(torch.tensor([3.0, 3.0, 3.0]), tensor.grad)

    def test_hook_with_no_name(self):
        # 创建一个没有 __name__ 属性的钩子类
        class MyHookClass:
            def __call__(self, grad):
                return grad.clone()

        # 创建一个随机张量，并注册钩子类的实例
        x = torch.randn(5, requires_grad=True).clone()
        x.register_hook(MyHookClass())
        x.sum().backward()
        # 应当正常运行

    def test_prehook_ordering(self):
        # 张量上注册的钩子在 grad_fn 上注册的钩子之前执行
        log = []

        # 定义第一个钩子函数，乘以3，并记录日志
        def hook1(g):
            log.append(1)
            return g * 3

        # 定义第二个钩子函数，每个梯度乘以2，并记录日志
        def hook2(gs):
            log.append(2)
            return tuple(g * 2 for g in gs)

        # 创建张量 a 和其克隆 b，并在 b 上注册钩子函数
        a = torch.tensor(1.0, requires_grad=True)
        b = a.clone()

        # 在 b 的 grad_fn 上注册第二个钩子函数
        b.grad_fn.register_prehook(hook2)
        # 在 b 上注册第一个钩子函数
        b.register_hook(hook1)
        # 在 b 的 grad_fn 上再次注册第二个钩子函数
        b.grad_fn.register_prehook(hook2)

        # 获取累积梯度节点，并在其上注册钩子函数
        acc = b.grad_fn.next_functions[0][0]
        # 在 a 上注册第一个钩子函数
        a.register_hook(hook1)
        # 在累积梯度节点上注册第二个钩子函数
        acc.register_prehook(hook2)
        # 在 a 上再次注册第一个钩子函数
        a.register_hook(hook1)

        # 对 b 进行和操作，并进行反向传播，保留计算图
        b.sum().backward(retain_graph=True)
        # 断言，日志应按顺序记录
        self.assertEqual(log, [1, 2, 2, 1, 1, 2])

        # grad 运行钩子函数在累积梯度节点上，即使累积梯度节点实际上不执行
        log = []
        torch.autograd.grad(b.sum(), inputs=(a,), retain_graph=True)
        # 断言，日志应按顺序记录
        self.assertEqual(log, [1, 2, 2, 1, 1])

        log = []
        b.sum().backward(inputs=(b,))
        # 断言，日志应按顺序记录
        self.assertEqual(log, [1, 2, 2])
        # 保留梯度钩子将不会观察到所有前置钩子的修改，因为它们在后执行
        self.assertEqual(b.grad.item(), 3)
    def test_retains_grad_can_always_observe_tensor_prehook(self):
        # 定义一个张量预处理函数，将梯度乘以2
        def tensor_prehook(g):
            return g * 2

        # 创建一个浮点张量a，需要计算梯度
        a = torch.tensor(1.0, requires_grad=True)
        # 克隆张量a，得到张量b
        b = a.clone()
        # 注册张量预处理函数到张量b
        b.register_hook(tensor_prehook)
        # 保留张量b的梯度信息
        b.retain_grad()
        # 再次注册张量预处理函数到张量b
        b.register_hook(tensor_prehook)

        # 对克隆的张量b执行反向传播
        b.clone().backward()
        # 断言张量b的梯度值为4
        self.assertEqual(b.grad.item(), 4)

        # 重新创建一个浮点张量a，需要计算梯度
        a = torch.tensor(1.0, requires_grad=True)
        # 克隆张量a，得到张量b
        b = a.clone()
        # 保留张量b的梯度信息
        b.retain_grad()
        # 注册张量预处理函数到张量b
        b.register_hook(tensor_prehook)

        # 对克隆的张量b执行反向传播
        b.clone().backward()
        # 断言张量b的梯度值为2
        self.assertEqual(b.grad.item(), 2)

    def test_accumulate_grad_posthooks_can_observe_tensor_prehook(self):
        # 累积梯度后处理函数应能观察张量预处理函数对梯度的影响
        a = torch.tensor(1.0, requires_grad=True)

        # 定义一个张量预处理函数，将梯度乘以2
        def tensor_prehook(g):
            return g * 2

        # 定义一个后处理函数，检查输入梯度与原张量a乘以2是否接近，输出梯度为空
        def posthook(gO, gI):
            self.assertTrue(torch.allclose(gI[0], a * 2))
            self.assertEqual(len(gO), 0)

        # 定义一个前处理函数，检查输入梯度与原张量a乘以2是否接近，输入梯度长度为1
        def prehook(gI):
            self.assertTrue(torch.allclose(gI[0], a * 2))
            self.assertEqual(len(gI), 1)

        # 克隆张量a，得到张量b
        b = a.clone()
        # 获取累积梯度的函数，并注册后处理函数与前处理函数
        acc = b.grad_fn.next_functions[0][0]
        acc.register_hook(posthook)
        acc.register_prehook(prehook)
        # 注册张量预处理函数到张量a
        a.register_hook(tensor_prehook)

        # 对克隆的张量b执行反向传播
        b.backward()

    def test_accumulate_grad_posthooks_should_not_execute(self):
        # 累积梯度后处理函数不应执行
        # 定义一个张量预处理函数，引发运行时错误
        def tensor_prehook(g):
            raise RuntimeError

        # 定义一个后处理函数，引发运行时错误
        def posthook(gO, gI):
            raise RuntimeError

        # 创建一个浮点张量a，需要计算梯度
        a = torch.tensor(1.0, requires_grad=True)
        # 注册张量预处理函数到张量a
        a.register_hook(tensor_prehook)
        # 创建一个浮点张量b，需要计算梯度
        b = torch.tensor(1.0, requires_grad=True)
        # 克隆张量a，得到张量c
        c = a.clone()
        # 获取累积梯度的函数，并注册后处理函数到acc
        acc = c.grad_fn.next_functions[0][0]
        acc.register_hook(posthook)

        # 计算张量a、b和c的和，然后对和进行反向传播，只计算b的梯度
        out = a + b + c
        out.sum().backward(inputs=[b])
    def test_hook_edge_case_when_called_with_grad(self):
        # 当调用 grad 时，它执行下一个节点的张量钩子，但不执行 grad_fn 的前钩子或后钩子
        a = torch.tensor(1.0, requires_grad=True)  # 创建一个张量 a，并设置需要梯度计算
        b = a * 2  # 创建张量 b，是 a 的两倍
        c = b * 2  # 创建张量 c，是 b 的两倍

        tensor_hook_count = [0]  # 用于统计张量钩子调用次数的列表
        prehook_count = [0]  # 用于统计前钩子调用次数的列表
        posthook_count = [0]  # 用于统计后钩子调用次数的列表

        def reset_counts():
            nonlocal tensor_hook_count, prehook_count, posthook_count
            tensor_hook_count = [0]  # 重置张量钩子计数
            prehook_count = [0]  # 重置前钩子计数
            posthook_count = [0]  # 重置后钩子计数

        def tensor_prehook(g):
            tensor_hook_count[0] += 1  # 张量钩子函数，每调用一次计数加一

        def prehook(g):
            prehook_count[0] += 1  # 前钩子函数，每调用一次计数加一

        def posthook(gI, gO):
            posthook_count[0] += 1  # 后钩子函数，每调用一次计数加一

        a.register_hook(tensor_prehook)  # 注册张量 a 的张量钩子函数
        b.register_hook(tensor_prehook)  # 注册张量 b 的张量钩子函数
        acc = b.grad_fn.next_functions[0][0]  # 获取 b 的梯度函数的下一个函数
        acc.register_hook(posthook)  # 注册 acc 的后钩子函数
        acc.register_prehook(prehook)  # 注册 acc 的前钩子函数
        b.grad_fn.register_hook(posthook)  # 注册 b 的梯度函数的后钩子函数
        b.grad_fn.register_prehook(prehook)  # 注册 b 的梯度函数的前钩子函数

        torch.autograd.grad(c, inputs=(b), retain_graph=True)  # 计算 c 对 b 的梯度，保留计算图
        self.assertEqual(tensor_hook_count[0], 1)  # 断言张量钩子调用次数为1
        self.assertEqual(posthook_count[0], 0)  # 断言后钩子调用次数为0
        self.assertEqual(prehook_count[0], 0)  # 断言前钩子调用次数为0
        reset_counts()  # 重置计数器

        torch.autograd.grad(c, inputs=(a, b), retain_graph=True)  # 计算 c 对 (a, b) 的梯度，保留计算图
        self.assertEqual(tensor_hook_count[0], 2)  # 断言张量钩子调用次数为2
        self.assertEqual(posthook_count[0], 1)  # 断言后钩子调用次数为1
        self.assertEqual(prehook_count[0], 1)  # 断言前钩子调用次数为1
        reset_counts()  # 重置计数器

        c.backward(retain_graph=True)  # 对 c 进行反向传播，保留计算图
        self.assertEqual(tensor_hook_count[0], 2)  # 断言张量钩子调用次数为2
        self.assertEqual(posthook_count[0], 2)  # 断言后钩子调用次数为2
        self.assertEqual(prehook_count[0], 2)  # 断言前钩子调用次数为2
        reset_counts()  # 重置计数器

        c.backward(inputs=(a, b), retain_graph=True)  # 对 c 进行反向传播，指定输入为 (a, b)，保留计算图
        self.assertEqual(tensor_hook_count[0], 2)  # 断言张量钩子调用次数为2
        self.assertEqual(posthook_count[0], 2)  # 断言后钩子调用次数为2
        self.assertEqual(prehook_count[0], 2)  # 断言前钩子调用次数为2

    def test_sharded_grad(self):
        leaves = [torch.zeros(5, 5, requires_grad=True) for _ in range(10)]  # 创建包含10个需要梯度的 5x5 零张量的列表
        intermediates = [l * i + l * l for i, l in enumerate(leaves)]  # 计算中间值张量列表
        loss = sum(v * i for i, v in enumerate(intermediates)).sum()  # 计算损失值

        def group(l, group_size):
            return (l[i : i + group_size] for i in range(0, len(l), group_size))  # 分组函数，将列表按照指定大小分组

        shard_size = 2  # 分片大小
        d_intermediates = [
            d_i
            for intermediates_batch in group(intermediates, shard_size)  # 遍历分组后的中间值列表
            for d_i in torch.autograd.grad(loss, intermediates_batch)  # 计算损失对每个分组的中间值的梯度
        ]
        torch.autograd.backward(intermediates, d_intermediates)  # 反向传播中间值和中间值的梯度

        for i, l in enumerate(leaves):
            self.assertEqual(l.grad, i * i * (1 + l))  # 断言每个叶子张量的梯度
    # 测试不正确的 torch.autograd.grad 调用，确保在不支持情况下引发 RuntimeError 异常
    def test_grad_badcalls(self):
        x = torch.ones(1)
        y = x**2
        # 确保在不支持情况下引发 RuntimeError 异常，提示不需要梯度
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            torch.autograd.grad(x, y)
        # 确保在不支持情况下引发 RuntimeError 异常，提示不需要梯度
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            torch.autograd.grad(y, x)

        x = torch.ones(1, requires_grad=True)
        y = x**2
        torch.autograd.grad(y, x)  # 现在应该成功

    # 测试空输入情况下的 torch.autograd.grad 调用，确保引发 ValueError 异常
    def test_grad_empty_inputs(self):
        x = torch.tensor([1.0], requires_grad=True)
        # 确保在空输入情况下引发 ValueError 异常，提示需要非空输入
        with self.assertRaisesRegex(ValueError, "grad requires non-empty inputs."):
            torch.autograd.grad(2 * x, [], grad_outputs=torch.tensor([1.0]))

    # 测试不正确的 torch.autograd.grad_fn 调用，确保在不支持情况下引发 TypeError 异常
    def test_grad_fn_badcalls(self):
        error_regex = "expected .* arguments, got .* instead"
        x = torch.ones(1, requires_grad=True)
        y = x**2
        # 确保在不支持情况下引发 TypeError 异常，提示参数数量错误
        with self.assertRaisesRegex(TypeError, error_regex):
            y.grad_fn(x.detach(), x.detach())  # 参数过多
        # 确保在不支持情况下引发 TypeError 异常，提示参数数量错误
        with self.assertRaisesRegex(TypeError, error_regex):
            y.grad_fn()  # 参数过少

        y.grad_fn(x.detach())  # 现在应该成功

    # 测试不可达的梯度计算情况，确保在允许未使用情况下正常运行，且在不允许时引发异常
    def test_grad_unreachable(self):
        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        # 确保 x 和 y 有梯度累加器分配
        z = x * 2
        w = y * 2

        # 确保正确计算 x 的梯度，y 的梯度应为 None
        grad_x, grad_y = torch.autograd.grad(x * 2, [x, y], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_y)

        # 这与上述情况略有不同，因为 z 甚至没有分配梯度累加器
        z = torch.ones(1, requires_grad=True)
        grad_x, grad_z = torch.autograd.grad(x * 2, [x, z], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_z)

        # allow_unused=False，但梯度包含 None，应该引发异常
        with self.assertRaisesRegex(RuntimeError, "Set allow_unused=True"):
            grad_x, grad_y = torch.autograd.grad(x * 2, [x, y], allow_unused=False)
    def test_grad_unreachable_discovery(self):
        # 测试当输入不可达时，确保某些节点不会被错误执行。参见 issue #39784

        # 定义一个自定义的 autograd 函数 MyFunc
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, x):
                # 断言，此节点不应该被执行到
                self.fail("This node should not be executed!")

        # 创建一个 MyFunc 对象并应用到随机生成的张量 x 上
        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        # 创建一个随机张量 y，要求梯度
        y = torch.randn(1, requires_grad=True)
        # 计算 x 对 y 的梯度，允许未使用的梯度
        (gY,) = torch.autograd.grad(x, (y,), allow_unused=True)
        # 断言 gY 应该是 None
        self.assertIsNone(gY)

        # 再次应用 MyFunc 到另一个随机生成的张量 x 上
        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        # 创建两个随机张量 y 和 z，要求梯度
        y = torch.randn(1, requires_grad=True)
        z = torch.randn(1, requires_grad=True)
        # 计算 x + z 对 y 和 z 的梯度，允许未使用的梯度
        (gY, gZ) = torch.autograd.grad(x + z, (y, z), allow_unused=True)
        # 断言 gY 应该是 None，而 gZ 应该不为 None
        self.assertIsNone(gY)
        self.assertIsNotNone(gZ)

        # 再次应用 MyFunc 到另一个随机生成的张量 x 上
        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        # 创建一个随机张量 y，要求梯度
        y = torch.randn(1, requires_grad=True)
        # 对 x 的梯度计算，inputs 参数为 (y,)，自动允许未使用的梯度
        torch.autograd.backward(x, inputs=(y,))
        # 断言 y.grad 应该是 None
        self.assertIsNone(y.grad)

    def test_grad_batched_grad(self):
        # 创建一个大小为 [2, 2] 的随机张量 x，要求梯度
        x = torch.randn(2, 2, requires_grad=True)

        # 克隆张量 x，并命名为 out，大小为 [2, 2]
        out = x.clone()  # Size([2, 2])
        # 创建一个大小为 [3, 2, 2] 的 batched_grad 张量
        batched_grad = (
            torch.arange(3).expand(2, 2, 3).transpose(0, 2)
        )  # Size([3, 2, 2])
        # 计算 out 对 x 的梯度，同时传入 batched_grad，并标记为 is_grads_batched=True
        (grad,) = torch.autograd.grad(out, (x,), (batched_grad,), is_grads_batched=True)
        # 断言 grad 应该与 torch.arange(3).expand(2, 2, 3).transpose(0, 2) 相等
        self.assertEqual(
            grad, torch.arange(3).expand(2, 2, 3).transpose(0, 2).to(dtype=grad.dtype)
        )

        # 检测形状不匹配的情况
        grad_out = torch.ones(2, 2)
        # 使用断言捕获 RuntimeError 异常，确保错误消息包含指定的文本
        with self.assertRaisesRegex(
            RuntimeError, "If `is_grads_batched=True`, we interpret the first"
        ):
            # 计算 out 对 x 的梯度，同时传入 grad_out 和 inputs=(x,)，并标记为 is_grads_batched=True
            torch.autograd.grad(
                outputs=out,
                grad_outputs=(grad_out,),
                inputs=(x,),
                is_grads_batched=True,
            )

        # 创建一个大小为 [] 的标量输出 out
        out = x.sum()  # Size([])
        # 创建一个大小为 [3] 的 batched_grad 张量
        batched_grad = torch.arange(3)  # Size([3])
        # 计算 out 对 x 的梯度，同时传入 batched_grad，并标记为 is_grads_batched=True
        (grad,) = torch.autograd.grad(out, (x,), (batched_grad,), is_grads_batched=True)
        # 断言 grad 应该与 torch.arange(3).expand(2, 2, 3).transpose(0, 2) 相等
        self.assertEqual(
            grad, torch.arange(3).expand(2, 2, 3).transpose(0, 2).to(dtype=grad.dtype)
        )

        # 创建一个大小为 [2] 的张量 grad_out
        grad_out = torch.ones(2).unsqueeze(1)
        # 使用断言捕获 RuntimeError 异常，确保错误消息包含指定的文本
        with self.assertRaisesRegex(
            RuntimeError, "If `is_grads_batched=True`, we interpret the first"
        ):
            # 计算 out 对 x 的梯度，同时传入 grad_out 和 inputs=(x,)，并标记为 is_grads_batched=True
            torch.autograd.grad(
                outputs=out,
                grad_outputs=(grad_out,),
                inputs=(x,),
                is_grads_batched=True,
            )
    def test_hooks(self):
        # 创建一个 5x5 的张量 x，并设置 requires_grad=True 以便进行梯度计算
        x = torch.ones(5, 5, requires_grad=True)
        # 创建一个 5x5 的张量 y，并将其每个元素设置为 4，不需要梯度计算
        y = torch.ones(5, 5) * 4
        y.requires_grad_(True)

        # 定义一个计数器列表，用于记录 backward hook 被调用的次数
        counter = [0]

        # 定义一个 backward hook 函数 bw_hook，用于接收梯度并更新计数器
        def bw_hook(inc, grad):
            self.assertIsInstance(grad, torch.Tensor)  # 检查梯度类型是否为 torch.Tensor
            counter[0] += inc  # 更新计数器

        # 定义计算结果张量 z
        z = x**2 + x * 2 + x * y + y
        # 注册一个 lambda 函数作为 x 的 backward hook，调用 bw_hook(0, *args)
        x.register_hook(lambda *args: bw_hook(0, *args))
        # 注册一个 lambda 函数作为 z 的 backward hook，调用 bw_hook(1, *args)
        test = z.register_hook(lambda *args: bw_hook(1, *args))
        # 执行反向传播，传入全为1的张量作为梯度，保留计算图
        z.backward(torch.ones(5, 5), retain_graph=True)
        # 断言计数器值为1
        self.assertEqual(counter[0], 1)

        # 注册一个 lambda 函数作为 z 的另一个 backward hook，调用 bw_hook(2, *args)
        test2 = z.register_hook(lambda *args: bw_hook(2, *args))
        # 再次执行反向传播，传入全为1的张量作为梯度，保留计算图
        z.backward(torch.ones(5, 5), retain_graph=True)
        # 断言计数器值为4
        self.assertEqual(counter[0], 4)

        # 移除 test2 hook
        test2.remove()
        # 再次执行反向传播，传入全为1的张量作为梯度，保留计算图
        z.backward(torch.ones(5, 5), retain_graph=True)
        # 断言计数器值为5
        self.assertEqual(counter[0], 5)

        # 定义一个修改梯度的函数 bw_hook_modify
        def bw_hook_modify(grad):
            return grad.mul(2)

        # 移除 test hook
        test.remove()
        # 注册 bw_hook_modify 函数作为 z 的新的 backward hook
        z.register_hook(bw_hook_modify)
        # 使用 torch.no_grad() 块清零 y 的梯度
        with torch.no_grad():
            y.grad.zero_()
        # 再次执行反向传播，传入全为1的张量作为梯度，保留计算图
        z.backward(torch.ones(5, 5), retain_graph=True)
        # 断言 y 的梯度为 (x + 1) * 2
        self.assertEqual(y.grad, (x + 1) * 2)

        # 注册 bw_hook_modify 函数作为 y 的新的 backward hook
        y.register_hook(bw_hook_modify)
        # 使用 torch.no_grad() 块清零 y 的梯度
        with torch.no_grad():
            y.grad.zero_()
        # 再次执行反向传播，传入全为1的张量作为梯度，不保留计算图
        z.backward(torch.ones(5, 5))
        # 断言 y 的梯度为 (x + 1) * 4
        self.assertEqual(y.grad, (x + 1) * 4)

    def _get_mul2(self, use_custom_function):
        if use_custom_function:
            # 定义一个自定义的 Function 类 Mul2，实现对输入张量乘以2的前向和反向传播
            class Mul2(Function):
                @staticmethod
                def forward(ctx, x):
                    return x * 2

                @staticmethod
                def backward(ctx, gO):
                    return gO * 2

            # 返回 Mul2 类的 apply 方法
            return Mul2.apply
        else:
            # 返回一个 lambda 函数，实现对输入张量乘以2
            return lambda x: x * 2
    # 定义测试函数 test_grad_fn_prehooks，用于测试梯度函数前钩子
    def test_grad_fn_prehooks(self):
        # 针对是否使用自定义函数进行循环测试
        for use_custom_function in (True, False):
            # 获取乘以2的函数 mul2
            mul2 = self._get_mul2(use_custom_function)

            # 创建一个张量 a，设置 requires_grad=True，表示需要计算梯度
            a = torch.tensor([1.0], requires_grad=True)
            # 对张量 a 应用乘以2的函数得到张量 b
            b = mul2(a)

            # 定义后钩子的计数器 post_counter 和前钩子的计数器 pre_counter
            post_counter = [0]
            pre_counter = [0]

            # 定义后钩子函数 posthook
            def posthook(grad_input, grad_output):
                # 断言前钩子计数器值为3
                self.assertEqual(pre_counter[0], 3)
                # 断言梯度输出与期望值接近
                self.assertTrue(torch.allclose(grad_output[0], torch.ones(1) * 8))
                # 断言梯度输入与期望值接近
                self.assertTrue(torch.allclose(grad_input[0], torch.ones(1) * 16))
                # 后钩子计数器加1
                post_counter[0] += 1
                return grad_input

            # 定义前钩子函数 prehook
            def prehook(grad_output):
                # 前钩子计数器加1
                pre_counter[0] += 1
                # 返回梯度输出乘以2的结果作为输入梯度
                return (grad_output[0] * 2,)

            # 注册两个后钩子函数
            b.grad_fn.register_hook(posthook)
            b.grad_fn.register_hook(posthook)
            # 注册三个前钩子函数
            b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(lambda x: None)
            b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(prehook)
            b.grad_fn.register_prehook(lambda x: x)
            b.grad_fn.register_prehook(lambda x: None)

            # 对张量 b 进行求和操作并反向传播
            b.sum().backward()

            # 断言后钩子计数器值为2
            self.assertEqual(post_counter[0], 2)
            # 断言前钩子计数器值为3
            self.assertEqual(pre_counter[0], 3)

            # 创建一个随机张量 a，设置 requires_grad=True，表示需要计算梯度
            a = torch.rand(3, 3, requires_grad=True)
            # 对张量 a 应用乘以2的函数得到张量 b
            b = mul2(a)

            # 重新定义前钩子函数 prehook
            def prehook(grad_output):
                # 前钩子计数器加1
                pre_counter[0] += 1
                # 返回 None 表示不修改梯度输入
                return None

            # 注册前钩子函数 prehook
            b.grad_fn.register_prehook(prehook)
            # 对张量 b 进行求和操作并反向传播
            b.sum().backward()

            # 断言前钩子计数器值为4
            self.assertEqual(pre_counter[0], 4)
            # 断言张量 a 的梯度与期望值接近，即全为2
            self.assertTrue(torch.allclose(a.grad, torch.ones(3, 3) * 2))
    def test_grad_fn_prehooks_multiple_outputs(self):
        # Compute gradients without hooks
        b = torch.rand(3, 3, requires_grad=True)  # 创建一个随机张量b，需要计算梯度
        var, mean = torch.var_mean(b, dim=0)  # 计算张量b在dim=0维度上的方差和均值
        (var + mean).sum().backward()  # 对方差和均值之和求和，并反向传播梯度

        # Compute gradients with hooks
        a = b.detach().requires_grad_()  # 创建张量a作为b的分离版本，并指定需要计算梯度
        counter = [0]

        def prehook(grad_output):
            gvar, gmean = grad_output  # 获取梯度输出中的方差梯度和均值梯度
            counter[0] += 1
            return (gvar * 2, gmean * 2)  # 返回修改后的方差和均值的梯度

        var, mean = torch.var_mean(a, dim=0)  # 计算张量a在dim=0维度上的方差和均值
        mean.grad_fn.register_prehook(prehook)  # 将prehook函数注册为均值的前钩子
        (var + mean).sum().backward()  # 对方差和均值之和求和，并反向传播梯度

        self.assertEqual(counter[0], 1)  # 断言前钩子调用次数为1
        # Compare
        self.assertTrue(torch.allclose(a.grad, b.grad * 2))  # 断言张量a的梯度是否等于张量b的梯度乘以2

        # Test with custom Function
        class DoubleMul2(Function):
            @staticmethod
            def forward(ctx, x, a, y):
                ctx.a = a
                return a * x * 2, a, a * y * 2  # 返回a * x * 2, a, a * y * 2

            @staticmethod
            def backward(ctx, g1, _a, g2):
                return ctx.a * g1 * 2, None, ctx.a * g2 * 2  # 返回ctx.a * g1 * 2, None, ctx.a * g2 * 2

        counter = [0]

        def prehook(grad_output):
            g1, ga, g2 = grad_output  # 获取梯度输出中的g1, ga, g2
            self.assertIsNone(ga)  # 断言ga为None
            counter[0] += 1
            return (g1 * 2, None, g2 * 2)  # 返回修改后的梯度输出

        a = torch.randn(3, 3, requires_grad=True)  # 创建一个随机张量a，需要计算梯度
        b = torch.randn(3, 3, requires_grad=True)  # 创建一个随机张量b，需要计算梯度
        k = 3
        c, _, d = DoubleMul2.apply(a, k, b)  # 使用DoubleMul2的apply方法计算张量a, k, b的结果
        c.grad_fn.register_prehook(prehook)  # 将prehook函数注册为c的前钩子
        (c + d).sum().backward()  # 对c和d之和求和，并反向传播梯度

        self.assertEqual(counter[0], 1)  # 断言前钩子调用次数为1
        self.assertTrue(torch.allclose(a.grad, torch.ones(1) * 4 * k))  # 断言张量a的梯度是否等于torch.ones(1) * 4 * k
        self.assertTrue(torch.allclose(b.grad, torch.ones(1) * 4 * k))  # 断言张量b的梯度是否等于torch.ones(1) * 4 * k
    def test_grad_fn_prehooks_remove_hooks(self):
        for use_custom_function in (True, False):
            mul2 = self._get_mul2`
    def test_grad_fn_prehooks_remove_hooks(self):
        # 遍历两个不同的布尔值，决定是否使用自定义函数
        for use_custom_function in (True, False):
            # 调用方法获取 mul2 函数
            mul2 = self._get_mul2(use_custom_function)

            # 创建一个张量 a，并设置 requires_grad 为 True，表示需要计算梯度
            a = torch.rand(3, 3, requires_grad=True)
            # 调用 mul2 函数对张量 a 进行运算，得到结果 b
            b = mul2(a)
            # 初始化计数器，初始值为 0
            counter = [0]

            # 定义一个前向钩子函数 prehook，增加计数器的值
            def prehook(grad_output):
                counter[0] += 1
                return None

            # 注册 prehook 钩子到 b.grad_fn 上，并返回句柄 handle
            handle = b.grad_fn.register_prehook(prehook)
            # 再次注册 prehook 钩子到 b.grad_fn 上
            b.grad_fn.register_prehook(prehook)
            # 移除句柄 handle，取消前向钩子的注册
            handle.remove()
            # 执行 b 的求和，并进行反向传播
            b.sum().backward()
            # 检查张量 a 的梯度是否等于 2，且所有元素值都为 2
            self.assertTrue(torch.allclose(a.grad, torch.ones(3, 3) * 2))
            # 检查计数器的值是否为 1，验证前向钩子被调用了一次
            self.assertEqual(counter[0], 1)

            # 创建张量 a，并设置 requires_grad 为 True，表示需要计算梯度
            a = torch.rand(3, 3, requires_grad=True)
            # 调用 mul2 函数对张量 a 进行运算，得到结果 b
            b = mul2(a)
            # 初始化计数器，初始值为 0
            counter = [0]

            # 定义前向钩子函数 prehook1，移除句柄 handle2 和 handle3
            def prehook1(grad_output):
                handle2.remove()
                # 移除已移除的钩子不会报错
                handle3.remove()
                return None

            # 定义另一个前向钩子函数 prehook2，增加计数器的值
            def prehook2(grad_output):
                counter[0] += 1
                return None

            # 注册前向钩子 prehook1 到 b.grad_fn 上
            b.grad_fn.register_prehook(prehook1)
            # 注册前向钩子 prehook2 到 b.grad_fn 上，返回句柄 handle2
            handle2 = b.grad_fn.register_prehook(prehook2)
            # 再次注册前向钩子 prehook2 到 b.grad_fn 上，返回句柄 handle3
            handle3 = b.grad_fn.register_prehook(prehook2)
            # 移除句柄 handle3，取消 prehook2 的注册
            handle3.remove()
            # 执行 b 的求和，并进行反向传播
            b.sum().backward()
            # 检查张量 a 的梯度是否等于 2，且所有元素值都为 2
            self.assertTrue(torch.allclose(a.grad, torch.ones(3, 3) * 2))
            # 检查计数器的值是否为 1，验证前向钩子被调用了一次
            self.assertEqual(counter[0], 1)

    def test_hooks_cpp(self):
        # 测试 C++ 实现的 autograd 函数的钩子
        bn = torch.nn.BatchNorm1d(5, affine=False)
        bn.double()
        bn.eval()

        # 初始化计数器，初始值为 0
        counter = [0]

        # 定义后向钩子函数 bw_hook，增加计数器的值，并返回梯度的两倍
        def bw_hook(grad):
            counter[0] += 1
            return grad * 2

        # 创建张量 x，数据类型为 double，并设置 requires_grad 为 True
        x = torch.ones(5, 5, dtype=torch.double, requires_grad=True)
        # 对张量 x 进行 BatchNorm 操作，得到结果 z
        z = bn(x)
        # 注册后向钩子 bw_hook 到 z 上
        z.register_hook(bw_hook)
        # 执行 z 的求和，并进行反向传播
        z.sum().backward()

        # 检查计数器的值是否为 1，验证后向钩子函数 bw_hook 是否被调用
        self.assertEqual(counter[0], 1, msg="bw_hook not called")
        # 检查张量 x 的梯度是否等于 2 倍的全 1 张量，容忍误差 1e-5
        self.assertEqual(
            x.grad, torch.ones(5, 5, dtype=torch.double) * 2, atol=1e-5, rtol=0
        )
    def test_hook_none(self):
        # WARNING: this is a test for autograd internals.
        # You should never have to use such things in your code.
        # 定义一个测试用的 Function 类，用于处理梯度计算中的特殊情况
        class NoneGradientFunction(Function):
            @staticmethod
            # 前向传播函数，接收输入 x 和 y，并返回它们本身
            def forward(ctx, x, y):
                assert ctx.needs_input_grad[0]
                assert not ctx.needs_input_grad[1]
                return x, y

            @staticmethod
            # 反向传播函数，接收梯度 grad_x 和 grad_y，并返回 grad_x 和 None
            def backward(ctx, grad_x, grad_y):
                return grad_x, None

        was_called = [False]

        # 定义一个钩子函数 hook，用于检查梯度是否为 None，并设置回调标志 was_called
        def hook(grad):
            self.assertIsNotNone(grad)
            was_called[0] = True

        # 创建随机张量 x 和 y，其中 x 需要计算梯度
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5)
        # 应用定义的 NoneGradientFunction 来计算 rx 和 ry
        rx, ry = NoneGradientFunction.apply(x, y)
        # 对 rx 和 ry 注册钩子函数 hook
        rx.register_hook(hook)
        ry.register_hook(hook)
        # 对 rx + ry 的结果进行求和，并进行反向传播
        sum(rx, ry).sum().backward()
        # 断言钩子函数 hook 被调用过
        self.assertTrue(was_called[0])

    def test_retain_grad(self):
        # 创建一个随机张量 input，需要计算梯度
        input = torch.rand(1, 3, requires_grad=True)
        # 对 input 进行乘法操作，并计算 out
        h1 = input * 3
        out = (h1 * h1).sum()

        # 调用 retain_grad() 函数两次，确保梯度保留
        h1.retain_grad()
        h1.retain_grad()

        # 对 out 进行反向传播，并断言 h1 的梯度应该是 h1 的两倍
        out.backward(retain_graph=True)
        self.assertEqual(h1 * 2, h1.grad)
        out.backward(retain_graph=True)
        self.assertEqual(h1 * 4, h1.grad)

        with torch.no_grad():
            input.grad.zero_()
        # 对于叶子节点 input，调用 retain_grad() 应该无效
        input.retain_grad()
        input.retain_grad()
        out.backward()
        self.assertEqual(input * 18, input.grad)

    # NB: See test/cpp/api/autograd.cpp for more tests on the interaction between
    #     retains_grad and hooks in cpp
    # 测试 inplace 操作下的梯度保留
    def test_retain_grad_inplace(self):
        # 创建一个张量 a，需要计算梯度，并克隆它
        a = torch.tensor([1.0], requires_grad=True).clone()
        # 调用 retain_grad() 保留梯度，并进行 inplace 乘法操作
        a.retain_grad()
        a.mul_(2)
        a.sum().backward()
        # 断言 a 的梯度应该是 [1.0]
        self.assertEqual(a.grad, torch.tensor([1.0]))

        # 再次克隆张量 a，进行多次 inplace 操作
        a = torch.tensor([1.0], requires_grad=True).clone()
        a.retain_grad()
        # inplace 操作可以多次进行
        a.mul_(2)
        a.mul_(2)
        a.sum().backward()
        # 断言 a 的梯度应该是 [1.0]
        self.assertEqual(a.grad, torch.tensor([1.0]))

        # 当对视图进行 inplace 操作时，梯度保留钩子应该从原始的 grad_fn 移动到复制的节点上
        x = torch.tensor([1.0], requires_grad=True).clone()
        x.retain_grad()
        x_view = x[:]
        x_view *= 2
        x *= 2
        x.sum().backward()
        # 由于我们计算最新版本的 x 的梯度，因此梯度应该是 1.0，而不是 4.0
        self.assertEqual(a.grad, torch.tensor([1.0]))

        # 如果基础张量原本不需要梯度，则不应有钩子可以移动。确保这种情况可以正常运行而不会出错。
        x = torch.zeros(4)
        y = x.view(2, 2)
        y.add_(torch.randn(2, 2, requires_grad=True))
    # 定义一个测试类方法，用于测试多输出情况下保留梯度的功能
    def test_retains_grad_inplace_multiple_outputs(self):
        # 定义一个继承自Function的类DoubleMul，实现前向传播和反向传播方法
        class DoubleMul(Function):
            @staticmethod
            # 前向传播方法，输入参数x，返回x乘以2和x乘以3的结果
            def forward(ctx, x):
                return x * 2, x * 3

            @staticmethod
            # 反向传播方法，输入参数g1和g2，返回g1乘以2加上g2乘以3的结果
            def backward(ctx, g1, g2):
                return g1 * 2 + g2 * 3

        # 使用partial函数创建一个var_mean函数，用于计算输入张量沿dim=0维度的方差和均值
        var_mean = partial(torch.var_mean, dim=0)

        # 遍历DoubleMul.apply和var_mean两个函数
        for fn in (DoubleMul.apply, var_mean):
            # 创建一个3x3的随机张量b，并要求计算其梯度
            b = torch.rand(3, 3, requires_grad=True)
            # 调用函数fn计算b的结果，分别赋值给var和mean
            var, mean = fn(b)
            # 保留var和mean的梯度
            var.retain_grad()
            mean.retain_grad()
            # 对var执行原地操作，乘以2
            var.mul_(2)
            # 计算(var + mean)的和并进行反向传播
            (var + mean).sum().backward()
            # 获取var和mean的梯度
            gvar = var.grad
            gmean = mean.grad

            # 对于从a克隆的b，重新设置requires_grad=True并计算fn(a)的结果
            a = b.detach().requires_grad_(True)
            var, mean = fn(a)
            # 对var执行原地操作，乘以2
            var.mul_(2)
            # 计算(var + mean)的和作为输出
            out = (var + mean).sum()
            # 计算out关于(var, mean)的梯度
            gvar_expected, gmean_expected = torch.autograd.grad(out, inputs=(var, mean))
            # 使用torch.allclose检查计算得到的梯度gvar和gmean与期望的gvar_expected和gmean_expected是否相似
            self.assertTrue(torch.allclose(gvar, gvar_expected))
            self.assertTrue(torch.allclose(gmean, gmean_expected))

    # 定义一个测试类方法，用于测试在视图操作中保留梯度的功能
    def test_retain_grad_inplace_over_view(self):
        # 创建一个requires_grad=True的基础张量base，并克隆视图view和view2
        base = torch.tensor([1.0], requires_grad=True).clone()
        view = base[:]
        view2 = base[:]
        # 保留view和view2的梯度
        view.retain_grad()
        view2.retain_grad()
        # 对view执行原地操作，乘以2
        view.mul_(2)
        # 计算(view + view2)的和并进行反向传播
        (view + view2).sum().backward()

        # 在反向传播期间，旧的grad_fn“slice”不会在图中，如果retain_grad未正确更新到新的grad_fn，则grad仍为None
        # 使用torch.assertEqual检查view.grad和view2.grad是否相等，并且等于torch.tensor([1.0])
        self.assertEqual(view.grad, view2.grad)
        self.assertEqual(view.grad, torch.tensor([1.0]))

    # 定义一个测试类方法，用于测试张量inplace操作中的钩子函数注册
    def test_tensor_hooks_inplace(self):
        # 检查第二个钩子函数是否注册到新版本的张量上
        count1 = [0]
        count2 = [0]

        # 定义钩子函数fn1，用于乘以2
        def fn1(grad):
            count1[0] += 1
            # 预期grad应为torch.tensor([4.0])
            self.assertEqual(grad, torch.tensor([4.0]))
            return grad * 2

        # 定义钩子函数fn2，用于乘以2
        def fn2(grad):
            count2[0] += 1
            # 预期grad应为torch.tensor([1.0])
            self.assertEqual(grad, torch.tensor([1.0]))
            return grad * 2

        # 创建一个requires_grad=True的张量a，并克隆为b
        a = torch.tensor([1.0], requires_grad=True)
        b = a.clone()
        # 注册fn1作为b的钩子函数
        b.register_hook(fn1)
        # 对b执行原地操作，乘以2
        b.mul_(2)
        # 注册fn2作为b的钩子函数
        b.register_hook(fn2)
        # 计算b的和并进行反向传播
        b.sum().backward()
        # 使用torch.assertEqual检查count1和count2是否分别为1
        self.assertEqual(count1[0], 1)
        self.assertEqual(count2[0], 1)
        # 使用torch.assertEqual检查a.grad是否等于torch.tensor([8.0])
        self.assertEqual(a.grad, torch.tensor([8.0]))

        count3 = [0]

        # 定义钩子函数fn3，用于乘以2
        def fn3(grad):
            count3[0] += 1
            # 预期grad应为torch.tensor([4.0])
            self.assertEqual(grad, torch.tensor([4.0]))
            return grad * 2

        # 重新创建一个requires_grad=True的张量a，并克隆为b
        a = torch.tensor([1.0], requires_grad=True)
        b = a.clone()
        # 注册fn3作为b的钩子函数
        b.register_hook(fn3)
        # 对b执行原地操作，多次乘以2
        b.mul_(2)
        b.mul_(2)
        # 计算b的和并进行反向传播
        b.sum().backward()
        # 使用torch.assertEqual检查count1是否仍然为1
        self.assertEqual(count1[0], 1)
        # 使用torch.assertEqual检查a.grad是否等于torch.tensor([8.0])
        self.assertEqual(a.grad, torch.tensor([8.0]))
    def test_tensor_hooks_inplace_multiple_outputs(self):
        # 定义一个自定义的函数类 DoubleMul，继承自 Function
        class DoubleMul(Function):
            # 静态方法：前向传播计算，返回两个结果 x*2 和 x*3
            @staticmethod
            def forward(ctx, x):
                return x * 2, x * 3

            # 静态方法：反向传播计算，接收两个梯度 g1 和 g2，返回组合后的梯度
            @staticmethod
            def backward(ctx, g1, g2):
                return g1 * 2 + g2 * 3

        # 部分应用 var_mean 函数，指定维度为 0
        var_mean = partial(torch.var_mean, dim=0)

        # 对于 DoubleMul.apply 和 var_mean 函数，分别进行测试
        for fn in (DoubleMul.apply, var_mean):
            # 初始化计数器列表 counts
            counts = [0, 0, 0]

            # 定义函数 fn0，用于处理梯度，增加 counts[0]，并进行梯度断言
            def fn0(grad):
                counts[0] += 1
                self.assertEqual(grad, torch.ones_like(out1) * 2)

            # 定义函数 fn1，用于处理梯度，增加 counts[1]，并进行梯度断言
            def fn1(grad):
                counts[1] += 1
                self.assertEqual(grad, torch.ones_like(out1) * 3)

            # 定义函数 fn2，用于处理梯度，增加 counts[2]，并进行梯度断言
            def fn2(grad):
                counts[2] += 1
                self.assertEqual(grad, torch.ones_like(out1))

            # 创建一个随机张量 b，并设置 requires_grad=True
            b = torch.rand(3, 3, requires_grad=True)
            # 调用 fn 函数（DoubleMul.apply 或 var_mean），得到 out1 和 out2
            out1, out2 = fn(b)
            # 将 fn0 注册到 out1 的 hook 上
            out1.register_hook(fn0)
            # 将 fn1 注册到 out2 的 hook 上
            out2.register_hook(fn1)
            # 对 out1 执行 in-place 操作，乘以 2
            out1.mul_(2)
            # 将 fn2 注册到 out1 的新 hook 上
            out1.register_hook(fn2)
            # 执行 (out1 + out2 * 3).sum() 的反向传播
            (out1 + out2 * 3).sum().backward()
            # 断言 counts 是否为 [1, 1, 1]
            self.assertEqual(counts, [1, 1, 1])

    def test_tensor_hooks_inplace_over_view(self):
        # 提示：这里可能有更好的用户体验，但目前是这样的方式
        # 初始化计数器列表 count
        count = [0]

        # 定义函数 fn0，用于处理梯度，失败断言
        def fn0(grad):
            self.fail()

        # 定义函数 fn1，用于处理梯度，失败断言
        def fn1(grad):
            self.fail()

        # 定义函数 fn2，用于处理梯度，增加 count[0]，并进行梯度断言
        def fn2(grad):
            count[0] += 1
            self.assertEqual(grad, torch.tensor([1.0]))

        # 创建一个张量 base，设置 requires_grad=True，并进行克隆操作
        base = torch.tensor([1.0], requires_grad=True).clone()
        # 创建一个视图 view，共享 base 的数据
        view = base[:]
        # 创建另一个视图 view2，同样共享 base 的数据
        view2 = base[:]
        # 将 fn0 注册到 view 的 hook 上
        view.register_hook(fn0)
        # 将 fn1 注册到 view2 的 hook 上
        view2.register_hook(fn1)
        # 对 view 执行 in-place 操作，乘以 2
        view.mul_(2)
        # 需要显式触发 view 的更新，以更新其 grad_fn
        view2.grad_fn
        # 将 fn2 注册到 view2 的 hook 上
        view2.register_hook(fn2)
        # 执行 (view + view2).sum() 的反向传播
        (view + view2).sum().backward()
        # 断言 count[0] 是否为 1
        self.assertEqual(count[0], 1)

    def test_retain_grad_cycle(self):
        # 创建一个形状为 (5, 5) 的张量 x，设置 requires_grad=True
        x = torch.ones(5, 5, requires_grad=True)

        # 定义函数 run_test，返回 x*2/y 和 torch._C._WeakTensorRef(y)
        def run_test():
            y = x * 2
            # 保留 y 的梯度
            y.retain_grad()
            return y / 2, torch._C._WeakTensorRef(y)

        # 调用 run_test，得到 z 和 ref
        z, ref = run_test()
        # 断言 ref 是否已过期
        self.assertTrue(ref.expired())
        # 对 z 执行 sum() 的反向传播
        z.sum().backward()
    def test_backward(self):
        # 创建一个5x5的张量，要求计算梯度
        v = torch.randn(5, 5, requires_grad=True)
        # 创建一个5x5的张量，要求计算梯度
        x = torch.randn(5, 5, requires_grad=True)
        # 创建一个5x5的随机张量，加上0.1，并要求计算梯度
        y = (torch.rand(5, 5) + 0.1).requires_grad_(True)
        # 创建一个5x5的张量，要求计算梯度
        z = torch.randn(5, 5, requires_grad=True)
        # 创建一个5x5的随机梯度输出张量
        grad_output = torch.randn(5, 5)

        # 对张量v进行反向传播，并断言其梯度与给定梯度grad_output相等
        v.backward(grad_output)
        self.assertEqual(v.grad, grad_output)

        # 计算复杂表达式a的梯度
        a = x + (y * z) + 4 * z**2 * x / y
        a.backward(grad_output)
        # 计算各个变量的梯度
        x_grad = 4 * z.pow(2) / y + 1
        y_grad = z - 4 * x * z.pow(2) / y.pow(2)
        z_grad = 8 * x * z / y + y
        # 断言计算得到的梯度与预期的梯度grad_output乘以各自的梯度相等
        self.assertEqual(x.grad, x_grad * grad_output)
        self.assertEqual(y.grad, y_grad * grad_output)
        self.assertEqual(z.grad, z_grad * grad_output)

    def test_to_sparse_backward(self):
        # 定义转换操作名称和对应的参数
        to_attr_names = (
            "to_dense",
            "to_sparse",
            "to_sparse_csr",
            "to_sparse_csc",
            "to_sparse_bsr",
            "to_sparse_bsc",
        )
        to_params = ((), (), (), (), (2,), (2,))
        to_attr_names_params = dict(zip(to_attr_names, to_params))

        def check_inversion_possible(
            t, layout1, layout1_params, layout2, layout2_params
        ):
            # 将布局和参数组合成元组
            l = (layout1, layout2)
            p = (layout1_params, layout2_params)
            # 尝试执行两种布局之间的转换操作，如果失败则返回False
            for l1, l2, p1, p2 in ((*l, *p), (*l[::-1], *p[::-1])):
                try:
                    to_l1 = getattr(t, l1)(*p1)
                    to_l2 = getattr(to_l1, l2)(*p2)
                except RuntimeError:
                    return False

            return True

        # 创建一个要求梯度的双精度4x4随机张量
        self_strided = torch.rand(4, 4, dtype=torch.double) + 1
        # 创建一个要求梯度的双精度4x4随机张量
        grad_strided = torch.rand(4, 4, dtype=torch.double) + 1

        # 遍历每一种转换操作
        for from_to_attr in to_attr_names:
            from_params = to_attr_names_params[from_to_attr]
            # 执行相应的转换操作，并要求计算梯度
            self_from = getattr(self_strided, from_to_attr)(
                *from_params
            ).requires_grad_(True)

            # 对于每一种目标转换操作，检查是否可以进行逆操作
            for to_to_attr in to_attr_names[1:]:
                to_params = to_attr_names_params[to_to_attr]

                if check_inversion_possible(
                    self_strided, from_to_attr, from_params, to_to_attr, to_params
                ):
                    # 执行目标转换操作，并要求计算梯度
                    self_to = getattr(self_from, to_to_attr)(*to_params)
                    grad_to = getattr(grad_strided, to_to_attr)(*to_params)

                    # 使用autograd.grad检查转换后的梯度grad_to是否正确
                    grad_res = torch.autograd.grad(self_to, self_from, grad_to)[0]

                    # 断言转换后的梯度的布局和to_dense()结果与grad_strided相等
                    self.assertEqual(grad_res.layout, self_from.layout)
                    self.assertEqual(grad_res.to_dense(), grad_strided)
    def test_sparse_mm_backward(self):
        size = (3, 3)

        # 生成所有可能的稀疏矩阵乘法测试用例
        mm_test_cases = product(*(([False, True],) * 4))

        for a_req_grad, a_is_sparse, b_req_grad, b_is_sparse in mm_test_cases:
            # 只测试包含稀疏输入的情况，并且至少一个输入需要求梯度以进行反向传播测试
            if not ((a_is_sparse or b_is_sparse) and (a_req_grad or b_req_grad)):
                continue
            a = torch.randn(size)
            if a_is_sparse:
                # 将稠密张量 `a` 转换为稀疏张量，并断开梯度关系以保证其为叶节点
                a = a.to_sparse().detach()
            b = torch.randn(size)
            if b_is_sparse:
                # 将稠密张量 `b` 转换为稀疏张量，并断开梯度关系以保证其为叶节点
                b = b.to_sparse().detach()

            a = a.requires_grad_(a_req_grad)
            b = b.requires_grad_(b_req_grad)

            r = a.mm(b)
            # 对结果进行求和并进行反向传播
            s = r.sum().backward()
            # 复制并断开梯度关系以获取梯度值
            a_grad = None if a.grad is None else a.grad.clone().detach()
            b_grad = None if b.grad is None else b.grad.clone().detach()

            # 使用稠密张量重复操作
            a = (
                (a.to_dense() if a.is_sparse else a)
                .clone()
                .detach()
                .requires_grad_(a_req_grad)
            )
            b = (
                (b.to_dense() if b.is_sparse else b)
                .clone()
                .detach()
                .requires_grad_(b_req_grad)
            )

            r = a.mm(b)
            # 对结果进行求和并进行反向传播
            r.sum().backward()

            # 断言梯度值是否正确
            self.assertEqual(a_grad, a.grad)
            self.assertEqual(b_grad, b.grad)

    def test_multi_backward(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        q = torch.randn(5, 5, requires_grad=True)

        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)

        q2 = q * 2
        z = x + y + q2
        c = a * b + q2
        grad_z = torch.randn(5, 5)
        grad_c = torch.randn(5, 5)
        # 对多个张量进行反向传播，传入对应的梯度值
        torch.autograd.backward([z, c], [grad_z, grad_c])

        # 断言每个张量的梯度是否正确
        self.assertEqual(x.grad, grad_z)
        self.assertEqual(y.grad, grad_z)
        self.assertEqual(a.grad, grad_c * b)
        self.assertEqual(b.grad, grad_c * a)
        self.assertEqual(q.grad, (grad_c + grad_z) * 2)

    def test_multi_backward_no_grad(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=False)

        z = x + y
        q = y * 2

        # 注意：如果任何传递给 backward 函数的张量 requires_grad=False 且没有 grad_fn，则会抛出异常。
        def call_backwards():
            torch.autograd.backward([z, q], [torch.ones(5, 5), torch.ones(5, 5)])

        # 断言是否会抛出 RuntimeError 异常
        self.assertRaises(RuntimeError, call_backwards)
    def test_backward_with_inputs(self):
        # 创建具有梯度的随机张量 x 和 y，形状为 2x2，数据类型为双精度浮点数
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

        # 定义计算函数 fn，返回 x^2 + y * x + y^2
        def fn():
            return x**2 + y * x + y**2

        # 设置梯度的期望值为全为1的张量
        gradient = torch.ones(2, 2)
        # 预期的 x 和 y 的梯度
        x_grad_expected = 2 * x + y
        y_grad_expected = x + 2 * y

        # 定义一个使用 @torch.no_grad 装饰的函数 reset_grad，用于重置梯度
        @torch.no_grad()
        def reset_grad():
            x.grad.zero_()
            y.grad.zero_()

        # 对 fn() 进行反向传播，计算梯度，inputs 包含 x 和 y
        torch.autograd.backward(fn(), gradient, inputs=[x, y])
        # 断言 x 的梯度与预期值 x_grad_expected 相等
        self.assertEqual(x.grad, x_grad_expected)
        # 断言 y 的梯度与预期值 y_grad_expected 相等
        self.assertEqual(y.grad, y_grad_expected)

        # 重置梯度
        reset_grad()
        # 对 fn() 进行反向传播，计算梯度，inputs 包含 x
        torch.autograd.backward(fn(), gradient, inputs=[x])
        # 断言 x 的梯度与预期值 x_grad_expected 相等
        self.assertEqual(x.grad, x_grad_expected)
        # 断言 y 的梯度为全零张量，数据类型可能不完全匹配
        self.assertEqual(y.grad, torch.zeros(2, 2), exact_dtype=False)

        # 重置梯度
        reset_grad()
        # 对 fn() 进行反向传播，计算梯度，inputs 包含 y
        torch.autograd.backward(fn(), gradient, inputs=[y])
        # 断言 y 的梯度与预期值 y_grad_expected 相等
        self.assertEqual(y.grad, y_grad_expected)
        # 断言 x 的梯度为全零张量，数据类型可能不完全匹配
        self.assertEqual(x.grad, torch.zeros(2, 2), exact_dtype=False)

        # 重置梯度
        reset_grad()
        # 对 fn() 进行反向传播，计算梯度，inputs 只包含 y
        torch.autograd.backward(fn(), gradient, inputs=y)
        # 断言 y 的梯度与预期值 y_grad_expected 相等
        self.assertEqual(y.grad, y_grad_expected)
        # 断言 x 的梯度为全零张量，数据类型可能不完全匹配
        self.assertEqual(x.grad, torch.zeros(2, 2), exact_dtype=False)

        # 重置梯度
        reset_grad()
        # 使用 lambda 函数捕获 RuntimeError，检测空输入时的反向传播
        self.assertRaisesRegex(
            RuntimeError,
            "cannot be empty",
            lambda: torch.autograd.backward(fn(), gradient, inputs=[]),
        )

    def test_backward_with_nonleaf_inputs(self):
        # 创建具有梯度的随机张量 x、x_nonleaf、y 和 z，形状为 2x2，数据类型为双精度浮点数
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        x_nonleaf = x * 1  # x 的一个非叶张量副本
        y = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        z = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

        # 计算 out = x_nonleaf^2 + y * x_nonleaf + y^2
        out = x_nonleaf**2 + y * x_nonleaf + y**2

        # 对 out 进行反向传播，计算梯度，inputs 包含 x、y 和 x_nonleaf
        out.backward(
            torch.ones(2, 2, dtype=torch.double),
            create_graph=True,
            inputs=[x, y, x_nonleaf],
        )
        # 预期的 x、y 和 x_nonleaf 的梯度
        x_grad_expected = 2 * x + y
        y_grad_expected = x + 2 * y
        x_non_leaf_expected = 2 * x_nonleaf + y

        # 断言 y 的梯度与预期值 y_grad_expected 相等
        self.assertEqual(y.grad, y_grad_expected)
        # 断言 x 的梯度与预期值 x_grad_expected 相等
        self.assertEqual(x.grad, x_grad_expected)
        # 断言 x_nonleaf 的梯度与预期值 x_non_leaf_expected 相等
        self.assertEqual(x_nonleaf.grad, x_non_leaf_expected)

        # 对于非叶张量 z，由于 backward 不会处理不在计算图中的变量，预期其梯度为 None
        out.backward(
            torch.ones(2, 2, dtype=torch.double), create_graph=True, inputs=[z]
        )
        self.assertIsNone(z.grad)

    def test_dependent_backward(self):
        # 创建具有梯度的随机张量 x，形状为 10，数据类型为默认浮点数，requires_grad=True
        x = torch.randn(10, requires_grad=True)
        y = x**2  # 计算 y = x^2
        z = y**3  # 计算 z = y^3

        # 创建与 y 和 z 形状相同的随机张量 go_y 和 go_z
        go_y = torch.randn(10)
        go_z = torch.randn(10)
        # 对 y 和 z 进行反向传播，计算梯度，gradient 分别为 go_y 和 go_z
        torch.autograd.backward([y, z], [go_y, go_z])

        # 用 xd 表示 x，计算预期的 x 的梯度
        xd = x
        x_grad_expected = 2 * xd * go_y + 6 * xd.pow(5) * go_z
        # 断言 x 的梯度与预期值 x_grad_expected 相等
        self.assertEqual(x.grad, x_grad_expected)
    def test_save_output_nr(self):
        # 创建一个形状为 (10,) 的张量 x，且要求计算其梯度
        x = torch.randn(10, requires_grad=True)

        # 定义一个自定义的 Torch 函数 MultiOutputFn
        class MultiOutputFn(Function):
            @staticmethod
            def forward(ctx, x):
                # 前向传播函数，返回张量 x 的前半部分和后半部分
                return x[:5], x[5:]

            @staticmethod
            def backward(ctx, *grad):
                # 反向传播函数，将输入的多个梯度张量连接起来返回
                return torch.cat(grad)

        # 调用自定义函数 MultiOutputFn 的 forward 方法，返回结果为 a 和 b
        a, b = MultiOutputFn.apply(x)
        # 断言 b 的输出编号为 1
        self.assertEqual(b.output_nr, 1)

        # 定义另一个自定义的 Torch 函数 TestFn
        class TestFn(Function):
            @staticmethod
            def forward(ctx, b):
                # 前向传播函数，保存张量 b 并返回其乘以 2 的结果
                ctx.save_for_backward(b)
                return b * 2

            @staticmethod
            def backward(ctx, grad_b):
                # 反向传播函数，断言保存的张量 b 的输出编号为 1
                (b,) = ctx.saved_tensors
                self.assertEqual(b.output_nr, 1)

        # 调用自定义函数 TestFn 的 apply 方法，对其结果求和并进行反向传播
        TestFn.apply(b).sum().backward()

    def test_first_grad_fn_access_in_no_grad_mode(self):
        # 创建一个实部为 1+1j 的张量 a，且要求计算其梯度
        a = torch.tensor([1 + 1j], requires_grad=True).clone()
        # 获取张量 a 的实部作为张量 v
        v = a.real
        # 对张量 a 添加 1
        a.add_(1)
        # 进入无梯度计算模式，访问张量 v 的梯度函数

    @skipIfTorchDynamo("too slow")
    def test_free_deep_graph(self):
        def scope():
            # 设置深度为 150000
            depth = 150000
            # 创建一个形状为 (1,) 的张量 x，且要求计算其梯度
            x = torch.randn(1, requires_grad=True)
            # 克隆张量 x 作为张量 y
            y = x.clone()

            # 构建一个“链式”计算图
            for _ in range(depth):
                y = y + y * 0.000001

            # 当上述局部变量退出作用域时，图的删除发生。
            # 在这种情况下，删除 y 将触发删除，但更容易让 Python 删除这些局部变量。

        # 不应该发生堆栈溢出
        scope()

    @skipIfTorchDynamo("too slow")
    def test_free_deep_graph_complicated(self):
        def scope():
            # 设置深度为 100000
            depth = 100000
            # 从 [0, 2) 的整数中随机选择深度个数作为 randchoice 张量的形状
            randchoice = torch.randint(2, [depth, 2])
            # 创建一个形状为 (1,) 的张量 x，且要求计算其梯度
            x = torch.randn(1, requires_grad=True)
            # 克隆张量 x 作为张量 y
            y = x.clone()

            # 保存前两个值
            prev_values = [None, None]

            # 构建一个带有跳跃连接的“链式”计算图
            for _ in range(depth):
                # 从前值列表中选择不为 None 的张量作为 prev_tensors
                prev_tensors = [
                    tensor for tensor in prev_values[:-1] if tensor is not None
                ]
                # 将 y 添加到 prev_values 中并弹出第一个值
                prev_values.append(y)
                prev_values.pop(0)

                # 必然选择一个张量进行加法
                y += y * 0.000001

                # 可能添加其他张量
                nprev = len(prev_tensors)
                if nprev == 2:
                    y += randchoice[depth].mul(torch.cat(prev_tensors)).sum()

            # 当上述局部变量退出作用域时，图的删除发生。

        # 不应该发生堆栈溢出
        scope()

    @skipIfTorchDynamo("too slow")
    # 定义测试用例，验证深度计算图中自定义函数的行为
    def test_free_deep_graph_pyfunction(self):
        # 定义自定义操作类 MyOp，继承自 Function
        class MyOp(Function):
            # 静态方法：定义前向传播逻辑，接受两个张量作为输入，返回它们的和
            @staticmethod
            def forward(ctx, tensor1, tensor2):
                return tensor1 + tensor2

            # 静态方法：定义反向传播逻辑，接受梯度输出作为输入，返回相同的梯度作为输入
            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, grad_output

        # 定义局部函数 scope
        def scope():
            # 设定深度为 150000
            depth = 150000
            # 创建一个随机张量 x，并设置 requires_grad=True，使其可以计算梯度
            x = torch.randn(1, requires_grad=True)
            # 克隆张量 x，并赋值给 y
            y = x.clone()

            # 构建深度嵌套的计算图
            for _ in range(depth):
                # 调用自定义操作 MyOp 的 apply 方法，进行 y = MyOp.apply(y, y) 的运算
                y = MyOp.apply(y, y)

            # 当上述局部变量离开作用域时，计算图将被删除

        # 调用局部函数 scope，验证不会发生堆栈溢出
        scope()

    # 定义测试用例，验证不保存不必要的张量状态
    def test_no_unnecessary_save(self):
        # 创建一个 requires_grad=True 的张量 mu，并赋值为 1
        mu = torch.ones(1, requires_grad=True)
        # 创建一个空张量 x
        x = torch.empty(1)
        # 初始化损失值为 0
        loss = 0
        # 循环三次
        for i in range(3):
            # 将 x 分离出计算图
            x.detach_()
            # 将 mu + i 的值复制给 x
            x.copy_(mu + i)
            # 创建一个 float 类型的张量 ft，其值为 i
            ft = torch.tensor([float(i)])
            # 计算 x 与 ft 的乘积
            multiplied = x * ft
            # 对乘积结果求和
            s = multiplied.sum()
            # 累加到损失值上
            loss += s
        # 反向传播损失
        loss.backward()

    # 定义测试用例，验证 torch.no_grad 上下文管理器的行为
    def test_no_grad(self):
        # 创建一个 requires_grad=True 的 5x5 的张量 x，所有元素均为 1
        x = torch.ones(5, 5, requires_grad=True)
        # 创建一个 5x5 的张量 y，所有元素均为 4
        y = torch.ones(5, 5) * 4
        # 进入 torch.no_grad 上下文管理器
        with torch.no_grad():
            # 计算 x + y，并将结果赋给 w
            w = x + y

        # 定义函数 adder，接受两个输入并返回它们的和
        def adder(x, y):
            return x + y

        # 创建一个包含两个使用 torch.no_grad 包装的 adder 函数的列表
        adders = [torch.no_grad()(adder), torch.no_grad(adder)]

        # 遍历 adders 列表
        for adder in adders:
            # 使用 adder 计算 x + y，并将结果赋给 z
            z = adder(x, y)

            # 断言 w 不需要梯度
            self.assertFalse(w.requires_grad)
            # 断言尝试在不允许梯度的情况下调用 w.backward 会引发 RuntimeError
            self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))
            # 断言 w 没有梯度函数
            self.assertIsNone(w.grad_fn)
            # 断言 z 不需要梯度
            self.assertFalse(z.requires_grad)
            # 断言尝试在不允许梯度的情况下调用 z.backward 会引发 RuntimeError
            self.assertRaises(RuntimeError, lambda: z.backward(torch.ones(5, 5)))
            # 断言 z 没有梯度函数
            self.assertIsNone(z.grad_fn)

        # 在 torch.no_grad 上下文管理器中测试嵌套的装饰器和 with 语句行为
        with torch.no_grad():
            # 断言当前不允许梯度计算
            self.assertFalse(torch.is_grad_enabled())
            # 使用 adder 计算 x + y，并将结果赋给 w
            w = adder(x, y)
            # 断言当前不允许梯度计算
            self.assertFalse(torch.is_grad_enabled())

    # 定义测试用例，验证 torch.enable_grad 装饰器的行为（无括号形式）
    def test_enable_grad_decorator_no_paren(self):
        # 创建一个 requires_grad=True 的张量 x，并赋值为 1
        x = torch.ones(1, requires_grad=True)

        # 定义使用 torch.enable_grad 装饰器包装的 doubler 函数，实现对输入张量的乘以 2 操作
        @torch.enable_grad
        def doubler(x):
            return x * 2

        # 进入 torch.no_grad 上下文管理器
        with torch.no_grad():
            # 调用 doubler 函数计算 x 的两倍，并将结果赋给 z
            z = doubler(x)
        # 断言 z 需要梯度
        self.assertTrue(z.requires_grad)
    def test_set_grad_generator_functions(self):
        # 定义一个测试函数，用于测试生成器函数在设置梯度追踪状态时的行为

        @torch.no_grad()
        def gen_no_grad():
            # 定义一个使用 torch.no_grad() 装饰器修饰的生成器函数，生成器内部梯度不追踪
            for i in range(10):
                self.assertEqual(torch.is_grad_enabled(), False)
                yield i

        with torch.enable_grad():
            # 使用 torch.enable_grad() 上下文管理器，确保生成器函数内部梯度追踪开启
            for _ in gen_no_grad():
                self.assertEqual(torch.is_grad_enabled(), True)

        @torch.enable_grad()
        def gen_enable_grad():
            # 定义一个使用 torch.enable_grad() 装饰器修饰的生成器函数，生成器内部梯度追踪开启
            for i in range(10):
                self.assertEqual(torch.is_grad_enabled(), True)
                yield i

        with torch.no_grad():
            # 使用 torch.no_grad() 上下文管理器，确保生成器函数内部梯度不追踪
            for _ in gen_enable_grad():
                self.assertEqual(torch.is_grad_enabled(), False)

    def test_set_grad_generator_functions_recursive(self):
        # 测试递归调用情况下，装饰器保持调用者的梯度追踪状态

        @torch.enable_grad()
        def enable_grad_decorator_recursive(depth):
            # 使用 torch.enable_grad() 装饰器定义的递归函数，确保梯度追踪开启
            self.assertTrue(torch.is_grad_enabled())
            if depth > 0:
                no_grad_decorator_recursive(depth - 1)
                self.assertTrue(torch.is_grad_enabled())

        @torch.no_grad()
        def no_grad_decorator_recursive(depth):
            # 使用 torch.no_grad() 装饰器定义的递归函数，确保梯度追踪关闭
            self.assertFalse(torch.is_grad_enabled())
            if depth > 0:
                enable_grad_decorator_recursive(depth - 1)
                self.assertFalse(torch.is_grad_enabled())

        # enable_grad_context_manager_recursive and no_grad_context_manager_recursive call
        # each other recursively, to ensure that the decorators preserve the caller's setting
        def enable_grad_context_manager_recursive(depth):
            # 使用 torch.enable_grad() 上下文管理器定义的递归函数，确保梯度追踪开启
            with torch.enable_grad():
                self.assertTrue(torch.is_grad_enabled())
                if depth > 0:
                    no_grad_context_manager_recursive(depth - 1)
                    self.assertTrue(torch.is_grad_enabled())

        def no_grad_context_manager_recursive(depth):
            # 使用 torch.no_grad() 上下文管理器定义的递归函数，确保梯度追踪关闭
            with torch.no_grad():
                self.assertFalse(torch.is_grad_enabled())
                if depth > 0:
                    enable_grad_context_manager_recursive(depth - 1)
                    self.assertFalse(torch.is_grad_enabled())

        with torch.enable_grad():
            # 在 torch.enable_grad() 上下文中，确保梯度追踪开启
            self.assertTrue(torch.is_grad_enabled())
            enable_grad_decorator_recursive(10)
            self.assertTrue(torch.is_grad_enabled())
            enable_grad_context_manager_recursive(10)
            self.assertTrue(torch.is_grad_enabled())

        with torch.no_grad():
            # 在 torch.no_grad() 上下文中，确保梯度追踪关闭
            self.assertFalse(torch.is_grad_enabled())
            enable_grad_decorator_recursive(10)
            self.assertFalse(torch.is_grad_enabled())
            enable_grad_context_manager_recursive(10)
            self.assertFalse(torch.is_grad_enabled())
    # 定义一个测试函数，用于测试协程中的梯度设置情况
    def test_set_grad_coroutines(self):
        
        # 声明一个不启用梯度的协程函数
        @torch.no_grad()
        def coro_no_grad(n=10):
            # 断言当前梯度是否已禁用
            self.assertFalse(torch.is_grad_enabled())
            # 循环执行 n 次
            for i in range(n):
                # 断言当前梯度是否已禁用
                self.assertFalse(torch.is_grad_enabled())
                # 向协程发送 i，并接收返回值
                r = yield i
                # 断言当前梯度是否已禁用
                self.assertFalse(torch.is_grad_enabled())
                # 断言接收到的返回值与 i 相等
                self.assertEqual(i, r)
            # 断言当前梯度是否已禁用
            self.assertFalse(torch.is_grad_enabled())

        # 声明一个启用梯度的协程函数
        @torch.enable_grad()
        def coro_enable_grad(n=10):
            # 断言当前梯度是否已启用
            self.assertTrue(torch.is_grad_enabled())
            # 循环执行 n 次
            for i in range(n):
                # 断言当前梯度是否已启用
                self.assertTrue(torch.is_grad_enabled())
                # 向协程发送 i，并接收返回值
                r = yield i
                # 断言当前梯度是否已启用
                self.assertTrue(torch.is_grad_enabled())
                # 断言接收到的返回值与 i 相等
                self.assertEqual(i, r)
            # 断言当前梯度是否已启用
            self.assertTrue(torch.is_grad_enabled())

        # 在启用梯度的上下文中执行以下代码块
        with torch.enable_grad():
            # 断言当前梯度是否已启用
            self.assertTrue(torch.is_grad_enabled())
            # 实例化不启用梯度的协程，并初始化 r 为 None
            coro, r = coro_no_grad(), None
            try:
                # 无限循环直至协程抛出 StopIteration 异常
                while True:
                    # 断言当前梯度是否已启用
                    self.assertTrue(torch.is_grad_enabled())
                    # 向协程发送 r，并接收返回值
                    r = coro.send(r)
                    # 断言当前梯度是否已启用
                    self.assertTrue(torch.is_grad_enabled())

            except StopIteration:
                pass

        # 在不启用梯度的上下文中执行以下代码块
        with torch.no_grad():
            # 断言当前梯度是否已禁用
            self.assertFalse(torch.is_grad_enabled())
            # 实例化启用梯度的协程，并初始化 r 为 None
            coro, r = coro_enable_grad(), None
            try:
                # 无限循环直至协程抛出 StopIteration 异常
                while True:
                    # 断言当前梯度是否已禁用
                    self.assertFalse(torch.is_grad_enabled())
                    # 向协程发送 r，并接收返回值
                    r = coro.send(r)
                    # 断言当前梯度是否已禁用
                    self.assertFalse(torch.is_grad_enabled())

            except StopIteration:
                pass
    def test_set_grad_coroutines_benign_exceptions(self):
        class RecoverableException(Exception):
            pass

        # 使用torch.no_grad()装饰器定义没有梯度的协程函数
        @torch.no_grad()
        def coro_no_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    # 断言当前梯度不可用
                    self.assertFalse(torch.is_grad_enabled())
                    # 如果已经抛出异常，返回负数；否则返回正数
                    yield (-i if has_raised else i)

                except RecoverableException:
                    # 断言当前梯度不可用
                    self.assertFalse(torch.is_grad_enabled())
                    has_raised = True

        # 使用torch.enable_grad()装饰器定义可以启用梯度的协程函数
        @torch.enable_grad()
        def coro_enable_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    # 断言当前梯度可用
                    self.assertTrue(torch.is_grad_enabled())
                    # 如果已经抛出异常，返回负数；否则返回正数
                    yield (-i if has_raised else i)

                except RecoverableException:
                    # 断言当前梯度可用
                    self.assertTrue(torch.is_grad_enabled())
                    has_raised = True

        # 在启用梯度的上下文中执行协程
        with torch.enable_grad():
            coro = coro_no_grad()
            # 断言第一个值为0
            assert 0 == next(coro)
            try:
                # 不断抛出RecoverableException直到StopIteration异常
                while True:
                    r = coro.throw(RecoverableException)
                    # 断言r小于0
                    self.assertLess(r, 0)

            except StopIteration:
                pass

        # 在没有梯度的上下文中执行协程
        with torch.no_grad():
            coro = coro_enable_grad()
            # 断言第一个值为0
            assert 0 == next(coro)
            try:
                # 不断抛出RecoverableException直到StopIteration异常
                while True:
                    r = coro.throw(RecoverableException)
                    # 断言r小于0
                    self.assertLess(r, 0)

            except StopIteration:
                pass

    def test_set_grad_coroutines_critical_exceptions(self):
        class UnrecoverableException(Exception):
            pass

        class SecondaryException(Exception):
            pass

        # 使用torch.no_grad()装饰器定义没有梯度的协程函数
        @torch.no_grad()
        def coro_no_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    # 断言当前梯度不可用
                    self.assertFalse(torch.is_grad_enabled())
                    # 如果抛出不可恢复异常，引发次要异常（SecondaryException）
                    yield (-i if has_raised else i)

                except UnrecoverableException:
                    # 断言当前梯度不可用
                    self.assertFalse(torch.is_grad_enabled())
                    raise SecondaryException from None

        # 使用torch.enable_grad()装饰器定义可以启用梯度的协程函数
        @torch.enable_grad()
        def coro_enable_grad(n=10):
            has_raised = False
            for i in range(n):
                try:
                    # 断言当前梯度可用
                    self.assertTrue(torch.is_grad_enabled())
                    # 如果抛出不可恢复异常，引发次要异常（SecondaryException）
                    yield (-i if has_raised else i)

                except UnrecoverableException:
                    # 断言当前梯度可用
                    self.assertTrue(torch.is_grad_enabled())
                    raise SecondaryException from None

        # 在启用梯度的上下文中执行协程
        with torch.enable_grad():
            coro = coro_no_grad()
            # 断言第一个值为0
            assert 0 == next(coro)
            # 断言引发次要异常（SecondaryException）
            with self.assertRaises(SecondaryException):
                coro.throw(UnrecoverableException)

        # 在没有梯度的上下文中执行协程
        with torch.no_grad():
            coro = coro_enable_grad()
            # 断言第一个值为0
            assert 0 == next(coro)
            # 断言引发次要异常（SecondaryException）
            with self.assertRaises(SecondaryException):
                coro.throw(UnrecoverableException)
    # 定义测试函数，用于验证在协程退出时设置梯度情况
    def test_set_grad_coroutines_exit(self):
        
        # 声明一个装饰器，用于在协程执行期间禁用梯度
        @torch.no_grad()
        def coro_no_grad(state):
            # 迭代10次
            for i in range(10):
                try:
                    # 断言当前梯度是否已禁用
                    self.assertFalse(torch.is_grad_enabled())
                    # 生成器返回当前迭代值
                    yield i

                except GeneratorExit:
                    # 断言当前梯度是否已禁用
                    self.assertFalse(torch.is_grad_enabled())
                    # 将"GeneratorExit"添加到状态集合中
                    state.add("GeneratorExit")
                    # 引发 GeneratorExit 异常
                    raise

        # 声明一个装饰器，用于在协程执行期间启用梯度
        @torch.enable_grad()
        def coro_enable_grad(state):
            # 迭代10次
            for i in range(10):
                try:
                    # 断言当前梯度是否已启用
                    self.assertTrue(torch.is_grad_enabled())
                    # 生成器返回当前迭代值
                    yield i

                except GeneratorExit:
                    # 断言当前梯度是否已启用
                    self.assertTrue(torch.is_grad_enabled())
                    # 将"GeneratorExit"添加到状态集合中
                    state.add("GeneratorExit")
                    # 引发 GeneratorExit 异常
                    raise

        # 初始化状态集合
        state = set()
        # 在启用梯度的上下文中执行
        with torch.enable_grad():
            # 创建禁用梯度的协程
            coro = coro_no_grad(state)
            # 迭代5次协程
            for i in range(5):
                next(coro)

            # 关闭协程
            coro.close()
        # 断言状态集合中包含"GeneratorExit"
        self.assertTrue("GeneratorExit" in state)

        # 重新初始化状态集合
        state = set()
        # 在禁用梯度的上下文中执行
        with torch.no_grad():
            # 创建启用梯度的协程
            coro = coro_enable_grad(state)
            # 迭代5次协程
            for i in range(5):
                next(coro)

            # 关闭协程
            coro.close()
        # 断言状态集合中包含"GeneratorExit"
        self.assertTrue("GeneratorExit" in state)

    # 定义测试函数，验证Python函数是否遵循梯度模式
    def test_no_grad_python_function(self):
        """Python Functions should respect grad mode."""
        # 创建一个需要梯度的张量
        x = torch.ones(5, 5, requires_grad=True)

        # 定义自定义操作类，继承自Function类
        class MyOp(Function):
            @staticmethod
            def forward(self, x):
                # 前向传播函数，返回输入张量加1
                return x + 1

            @staticmethod
            def backward(self, dy):
                # 反向传播函数，返回梯度
                return dy

        # 在禁用梯度的上下文中执行
        with torch.no_grad():
            # 应用自定义操作到张量x上，得到输出y
            y = MyOp.apply(x)
        # 断言输出y不需要梯度
        self.assertFalse(y.requires_grad)
    def test_indexing_duplicates(self):
        # 创建一个 4x4 的张量 x，值为从 1 到 16
        x = torch.arange(1.0, 17).view(4, 4)
        # 创建一个 Variable y，其值与 x 相同，并且要求计算梯度
        y = Variable(x, requires_grad=True)

        # 定义索引 idx，包含重复元素
        idx = torch.LongTensor([1, 1, 3, 2, 1, 2])
        # 使用索引 idx 访问张量 y，计算其和，并执行反向传播
        y[idx].sum().backward()
        # 创建预期梯度张量 expected_grad，初始化为全零
        expected_grad = torch.zeros(4, 4)
        # 根据索引 idx 更新预期梯度
        for i in idx:
            expected_grad[i] += 1
        # 断言计算得到的梯度与预期梯度相等
        self.assertEqual(y.grad, expected_grad)

        # 使用高级索引
        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)

        # 定义包含嵌套列表的索引 idx
        idx = [[1, 1, 3, 2, 1, 2], [0]]
        # 使用索引 idx 访问张量 y，计算其和，并执行反向传播
        y[idx].sum().backward()
        # 创建预期梯度张量 expected_grad，初始化为全零
        expected_grad = torch.zeros(4, 4)
        # 根据索引 idx 更新预期梯度
        for i in idx[0]:
            for j in idx[1]:
                expected_grad[i][j] += 1

        # 断言计算得到的梯度与预期梯度相等
        self.assertEqual(y.grad, expected_grad)

        # 更复杂的索引示例
        x = torch.arange(1.0, 17).view(4, 4)
        y = Variable(x, requires_grad=True)
        # 定义更深层次的嵌套列表索引 idx
        idx = [[[1, 2], [0, 0]], [[0, 1], [1, 1]]]
        # 使用索引 idx 访问张量 y，计算其和，并执行反向传播
        y[idx].sum().backward()
        # 创建预期梯度张量 expected_grad，根据具体索引位置填充对应的值
        expected_grad = torch.tensor(
            [
                [0.0, 2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        # 断言计算得到的梯度与预期梯度相等
        self.assertEqual(y.grad, expected_grad)

        # 更复杂的索引示例
        x = torch.arange(1.0, 65).view(4, 4, 4)
        y = Variable(x, requires_grad=True)

        # 定义使用切片的索引 idx
        idx = [[1, 1, 1], slice(None), slice(None)]
        # 使用索引 idx 访问张量 y，计算其和，并执行反向传播
        y[idx].sum().backward()
        # 创建预期梯度张量 expected_grad，根据具体索引位置填充对应的值
        expected_grad = torch.empty(4, 4, 4).zero_()
        expected_grad[1].fill_(3)
        # 断言计算得到的梯度与预期梯度相等
        self.assertEqual(y.grad, expected_grad)

    def test_index_backward_does_not_save_tensor(self):
        # 示例来自 https://github.com/pytorch/pytorch/issues/24853.
        # 如果 `index(tensor, indices)` 在反向传播时保存了 `tensor`，则会触发版本检查，可能导致错误
        a = torch.tensor([1.0, 0, 0])
        b = torch.zeros(3, requires_grad=True)
        tensor = b + 0
        # 根据索引条件更新 tensor，此行代码可能影响 `tensor` 的版本检查
        tensor[a != 0] = tensor[a != 0]
        # 执行反向传播，传入全零的梯度
        tensor.backward(torch.zeros_like(tensor))

    def test_volatile_deprecated(self):
        v = torch.autograd.torch.randn(3, 3)
        with warnings.catch_warnings(record=True) as w:
            # 检查张量 v 是否标记为 'volatile'，这是不推荐的做法
            self.assertFalse(v.volatile)
        # 断言警告信息中包含 'volatile' 字样
        self.assertIn("volatile", str(w[0].message))
    # 定义一个测试方法，用于测试已废弃的保存变量功能
    def test_saved_variables_deprecated(self):
        # 定义一个自定义的函数类 MyFunction，继承自 torch.autograd.Function
        class MyFunction(Function):
            # 静态方法：前向传播函数，接收 ctx 上下文和两个张量 tensor1, tensor2
            @staticmethod
            def forward(ctx, tensor1, tensor2):
                # 在上下文 ctx 中保存 tensor1 和 tensor2 的变量
                ctx.save_for_backward(tensor1, tensor2)
                # 返回 tensor1 + tensor2 的结果
                return tensor1 + tensor2

            # 静态方法：反向传播函数，接收 ctx 上下文和梯度 grad_output
            @staticmethod
            def backward(ctx, grad_output):
                # 从 ctx 中获取已保存的变量 var1 和 var2
                var1, var2 = ctx.saved_variables
                # 返回梯度 grad_output 的两倍作为反向传播的梯度
                return (grad_output, grad_output)

        # 使用 warnings 模块捕获所有警告信息
        with warnings.catch_warnings(record=True) as warns:
            # 设定警告的筛选条件为 "always"
            warnings.simplefilter("always")
            # 创建两个随机张量 x 和 y，并标记需要计算梯度
            x = torch.randn((3, 3), requires_grad=True)
            y = torch.randn((3, 3), requires_grad=True)
            # 调用 MyFunction 的前向传播并对其结果求和后进行反向传播
            MyFunction.apply(x, y).sum().backward()

            # 检查是否有 "deprecated" 和 "saved_variables" 的警告信息
            has_deprecated = (
                "deprecated" in str(warn) and "saved_variables" in str(warn)
                for warn in warns
            )
            # 使用 reduce 函数将警告信息中是否存在特定警告的判断结果聚合为一个布尔值
            has_deprecated = reduce(lambda x, y: x or y, has_deprecated)
            # 使用断言验证 has_deprecated 是否为 True
            self.assertTrue(has_deprecated)

    # 定义一个测试方法，用于测试张量的 requires_grad 属性设置
    def test_requires_grad(self):
        # 创建三个随机张量 x, y, z
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        z = torch.randn(5, 5, requires_grad=True)
        # 计算张量 x 和 y 的和 a，并验证 a 的 requires_grad 属性为 False
        a = x + y
        self.assertFalse(a.requires_grad)
        # 计算张量 a 和 z 的和 b，并验证 b 的 requires_grad 属性为 True
        b = a + z
        self.assertTrue(b.requires_grad)

        # 定义一个引发 RuntimeError 异常的函数 error
        def error():
            raise RuntimeError

        # 确保这些张量的 backward 操作不会被调用
        a._backward_hooks = OrderedDict()
        x._backward_hooks = OrderedDict()
        y._backward_hooks = OrderedDict()
        a._backward_hooks["test"] = error
        x._backward_hooks["test"] = error
        y._backward_hooks["test"] = error
        # 对张量 b 执行反向传播，期望会引发错误
        b.backward(torch.ones(5, 5))

    # 定义一个测试方法，用于测试张量的 requires_grad_() 方法
    def test_requires_grad_(self):
        # 创建两个随机张量 x 和 y，并验证它们调用 requires_grad_() 后的结果是否正确
        x = torch.randn(5, 5)
        y = torch.randn(5, 5, requires_grad=True)
        self.assertIs(x, x.requires_grad_())
        self.assertTrue(x.requires_grad)
        self.assertIs(y, y.requires_grad_())
        self.assertTrue(y.requires_grad)
        self.assertIs(x, x.requires_grad_(True))
        self.assertTrue(x.requires_grad)
        self.assertIs(y, y.requires_grad_(True))
        self.assertTrue(y.requires_grad)
        # 创建张量 z，为 x 和 y 的逐元素乘积，并验证尝试取消其 requires_grad 属性会引发 RuntimeError
        z = x * y
        self.assertRaises(RuntimeError, lambda: z.requires_grad_(False))
        self.assertIs(z, z.requires_grad_())
        self.assertTrue(z.requires_grad)
        self.assertIs(z, z.requires_grad_(True))
        self.assertTrue(z.requires_grad)
        # 将张量 x 和 y 的 requires_grad 属性设置为 False，并验证设置后的结果
        self.assertIs(x, x.requires_grad_(False))
        self.assertFalse(x.requires_grad)
        self.assertIs(y, y.requires_grad_(False))
        self.assertFalse(y.requires_grad)

    # 定义一个测试方法，用于测试张量的 inplace 操作后的 requires_grad 属性
    def test_requires_grad_inplace(self):
        # 创建两个随机张量 a 和 b，并验证其 inplace 操作后的 requires_grad 属性是否为 True
        a = torch.randn(5, 5)
        b = torch.randn(5, 5, requires_grad=True)
        a += b
        self.assertTrue(a.requires_grad)

        # 创建两个非叶子节点的张量 a 和 b，并验证其 inplace 操作后的 requires_grad 属性是否为 True
        a = torch.randn(5, 5) + 0
        b = torch.randn(5, 5, requires_grad=True)
        a += b
        self.assertTrue(a.requires_grad)
    # 测试函数：测试不允许在 requires_grad 为 True 时进行原地修改
    def test_no_requires_grad_inplace(self):
        # 创建一个形状为 (2, 3) 的张量 a，其值为随机生成的数值
        a = torch.randn(2, 3)
        # 在 requires_grad 为 False 的情况下，应能原地修改张量的值
        a.add_(5)
        # 将 requires_grad 设置为 True
        a.requires_grad = True
        # 对张量进行求和并进行反向传播
        a.sum().backward()
        # 断言梯度计算的结果与预期一致，应为形状为 (2, 3) 的全为1的张量
        self.assertEqual(a.grad, torch.ones(2, 3))

        # 同样的操作，但是使用视图 b
        a = torch.randn(2, 3)
        b = a[:]
        b.add_(5)
        a.requires_grad = True
        a.sum().backward()
        self.assertEqual(a.grad, torch.ones(2, 3))

        # 如果在进行原地修改时 requires_grad 为 True，则应该抛出 RuntimeError
        a = torch.randn(2, 3)
        b = a[:]
        a.requires_grad = True
        with self.assertRaises(RuntimeError):
            a.add_(5)
        with self.assertRaises(RuntimeError):
            b.add_(5)

    # 测试函数：测试属性删除
    def test_attribute_deletion(self):
        # 创建一个形状为 (5, 5) 的张量 x，并将 requires_grad 设置为 True
        x = torch.randn((5, 5), requires_grad=True)
        # 删除梯度信息，预期梯度应该为 None
        del x.grad
        self.assertIsNone(x.grad)
        # 尝试删除其他关键属性，预期应该抛出 RuntimeError 或 TypeError
        with self.assertRaises(RuntimeError):
            del x.data
        with self.assertRaises(TypeError):
            x.data = None
        with self.assertRaises(RuntimeError):
            del x.requires_grad
        with self.assertRaises(RuntimeError):
            del x._grad_fn
        with self.assertRaises(RuntimeError):
            del x._backward_hooks

    # 测试函数：测试重复进行反向传播时的情况
    def test_duplicate_backward_root(self):
        # 创建两个形状为 (5, 5) 的张量 a 和 b，并将 requires_grad 设置为 True
        a = torch.randn(5, 5, requires_grad=True)
        b = torch.randn(5, 5, requires_grad=True)

        # 计算张量 x，其为 a 和 b 的乘积
        x = a * b
        # 创建与 x 形状相同的随机梯度张量
        grad_output = torch.randn_like(x)
        # 进行反向传播
        torch.autograd.backward([x, x], [grad_output, grad_output])

        # 断言 a 和 b 的梯度计算结果与预期一致
        self.assertEqual(a.grad, b * grad_output * 2)
        self.assertEqual(b.grad, a * grad_output * 2)

    # 测试函数：测试在不允许梯度的张量上进行反向传播
    def test_backward_no_grad(self):
        # 创建一个形状为 (5, 5) 的张量 a，并将 requires_grad 设置为 True
        a = torch.randn(5, 5, requires_grad=True)
        # 创建新的张量 b，其为 a 加上常数 2
        b = a + 2
        # 在不允许梯度的张量上进行反向传播，预期应该抛出 RuntimeError
        with self.assertRaises(RuntimeError):
            torch.autograd.backward([b], [None])

    # 测试函数：测试在保留图的情况下进行重复反向传播，并保存值的情况
    def test_backward_twice_with_saved_values(self):
        # 创建一个形状为 (3,) 的双精度张量 b，并将 requires_grad 设置为 True
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        # 创建一个形状为 (3,) 的零张量 c
        c = torch.zeros(3, dtype=torch.double)
        # 通过索引赋值，将 b 的部分值复制到 c 中
        c[[1, 2]] = b[[1, 1]]
        # 进行反向传播，并指定梯度张量
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))
        # 第二次反向传播时，未指定 retain_graph=True，预期应该抛出错误信息
        self.assertRaisesRegex(
            RuntimeError,
            "Specify retain_graph=True",
            lambda: c.backward(torch.tensor([1, 1, 1], dtype=torch.double)),
        )

    # 测试函数：测试在保留图的情况下进行重复反向传播，并不保存值的情况
    def test_backward_twice_retained_graph_with_saved_values(self):
        # 创建一个形状为 (3,) 的双精度张量 b，并将 requires_grad 设置为 True
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        # 创建一个形状为 (3,) 的零张量 c
        c = torch.zeros(3, dtype=torch.double)
        # 通过索引赋值，将 b 的部分值复制到 c 中
        c[[1, 2]] = b[[1, 1]]
        # 进行反向传播，并指定梯度张量以及 retain_graph=True
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double), retain_graph=True)
        # 再次进行反向传播，预期不会抛出错误
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    # 测试函数：测试在不保存值的情况下进行重复反向传播
    def test_backward_twice_without_saved_values(self):
        # 创建一个形状为 (3,) 的双精度张量 b，并将 requires_grad 设置为 True
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        # 创建一个新的张量 c，其为 b 加上常数 1
        c = b + 1
        # 进行反向传播，并指定梯度张量
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))
        # 再次进行反向传播，预期不会抛出错误
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))
    # 定义一个测试函数，用于测试反向传播在不保留保存值的情况下两次进行时的行为
    def test_backward_twice_retained_graph_without_saved_values(self):
        # 创建一个形状为 (3,) 的随机张量 b，并设置其需要梯度计算，数据类型为双精度
        b = torch.randn(3, requires_grad=True, dtype=torch.double)
        # 创建一个形状为 (3,) 的零张量 c，数据类型为双精度
        c = torch.zeros(3, dtype=torch.double)
        # 从张量 b 的索引 1 和 2 处获取值并赋给张量 c 的相应位置，此处涉及到张量的索引操作
        c[[1, 2]] = b[[1, 1]]
        # 对张量 c 进行反向传播，梯度为 [1, 1, 1]，并保留计算图
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double), retain_graph=True)
        # 再次对张量 c 进行反向传播，梯度为 [1, 1, 1]
        c.backward(torch.tensor([1, 1, 1], dtype=torch.double))

    # 定义一个测试函数，用于测试在创建计算图时 backward 函数是否会发出警告
    def test_backward_create_graph_warns(self):
        # 打开警告上下文，设置在该上下文中始终发出警告
        with set_warn_always_context(True):
            # 创建一个形状为 (3,) 的随机张量 b，并设置其需要梯度计算，数据类型为双精度
            b = torch.randn(3, requires_grad=True, dtype=torch.double)
            # 计算张量 b 的平方，结果为张量 c
            c = b * b
            # 使用 backward 函数对张量 c 进行反向传播，梯度为张量 c 的形状相同的全1张量，并创建计算图
            with warnings.catch_warnings(record=True) as ws:
                c.backward(torch.ones_like(c), create_graph=True)
            # 清空张量 b 的梯度
            b.grad = None
            # 断言是否有警告信息包含指定字符串
            self.assertTrue(
                any(
                    "Using backward() with create_graph=True" in str(w.message)
                    for w in ws
                )
            )

            # 对于 torch.autograd.grad 函数的调用，使用 create_graph=True，不应发出警告
            with warnings.catch_warnings(record=True) as ws:
                torch.autograd.grad(c, b, torch.ones_like(c), create_graph=True)
            # 断言不存在警告信息包含指定字符串
            self.assertFalse(
                any(
                    "Using backward() with create_graph=True" in str(w.message)
                    for w in ws
                )
            )

    # 定义一个测试函数，用于测试张量的梯度函数的 next_functions 属性
    def test_next_functions(self):
        # 创建形状为 (5, 5) 的随机张量 x 和 y，并设置其需要梯度计算
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        # 计算张量 a 为张量 x 和 y 的元素级加法结果
        a = x + y
        # 断言张量 a 的梯度函数不为空
        self.assertIsNotNone(a.grad_fn)
        # 获取张量 a 的梯度函数的 next_functions 属性
        next_functions = a.grad_fn.next_functions
        # 断言 next_functions 的长度为 2
        self.assertEqual(len(next_functions), 2)
        # 断言 next_functions 的第一个元素是 AccumulateGrad 函数类型
        self.assertIsInstance(next_functions[0][0], torch._C._functions.AccumulateGrad)
        # 断言 next_functions 的第一个元素的索引为 0
        self.assertEqual(next_functions[0][1], 0)
        # 断言 next_functions 的第二个元素是 AccumulateGrad 函数类型
        self.assertIsInstance(next_functions[1][0], torch._C._functions.AccumulateGrad)
        # 断言 next_functions 的第二个元素的索引为 0
        self.assertEqual(next_functions[1][1], 0)

        # 计算张量 b 为张量 a 加上标量 5
        b = a + 5
        # 获取张量 b 的梯度函数的 next_functions 属性
        next_functions = b.grad_fn.next_functions
        # 断言 next_functions 的长度为 2
        self.assertEqual(len(next_functions), 2)
        # 断言 next_functions 的第一个元素的梯度函数为张量 a 的梯度函数
        self.assertIs(next_functions[0][0], a.grad_fn)
        # 断言 next_functions 的第二个元素的梯度函数为空
        self.assertIs(next_functions[1][0], None)
    # 定义一个测试函数，用于测试张量操作的就地修改和自动求导功能
    def test_inplace(self):
        # 创建一个大小为5x5的张量x，所有元素为1，并开启梯度追踪
        x = torch.ones(5, 5, requires_grad=True)
        # 创建一个大小为5x5的Variable y，所有元素为4，并开启梯度追踪
        y = Variable(torch.ones(5, 5) * 4, requires_grad=True)

        # 计算张量乘法 z = x * y
        z = x * y
        # 计算张量加法 q = z + y
        q = z + y
        # 计算张量乘法 w = z * y
        w = z * y
        # 对 z 进行就地修改，添加常数值2
        z.add_(2)
        # 对 q 执行反向传播，使用全1张量作为梯度，并保留计算图
        q.backward(torch.ones(5, 5), retain_graph=True)
        # 对 w 执行反向传播，预期会引发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))

        # 重新计算张量乘法 z = x * y
        z = x * y
        # 计算张量乘法 q = z * y
        q = z * y
        # 计算张量加法 r = z + y
        r = z + y
        # 对 z 执行就地加法操作，并将结果赋给 w
        w = z.add_(y)
        # 对 w 执行反向传播，使用全1张量作为梯度，并保留计算图
        w.backward(torch.ones(5, 5), retain_graph=True)
        # 对 r 执行反向传播，使用全1张量作为梯度，并保留计算图
        r.backward(torch.ones(5, 5), retain_graph=True)
        # 对 q 执行反向传播，预期会引发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        # 在没有梯度追踪的环境中操作
        with torch.no_grad():
            # 将张量 x 的梯度置零
            x.grad.zero_()
        # 计算张量 m = x / 2
        m = x / 2
        # 计算张量 z = m + y / 8
        z = m + y / 8
        # 计算张量乘法 q = z * y
        q = z * y
        # 计算张量加法 r = z + y
        r = z + y
        # 记录 z 的当前版本
        prev_version = z._version
        # 对 z 执行指数函数的就地操作
        w = z.exp_()
        # 检查 z 的版本是否有变化
        self.assertNotEqual(z._version, prev_version)
        # 对 r 执行反向传播，使用全1张量作为梯度，并保留计算图
        r.backward(torch.ones(5, 5), retain_graph=True)
        # 检查张量 x 的梯度是否为大小为5x5的全1张量除以2
        self.assertEqual(x.grad, torch.ones(5, 5) / 2)
        # 对 w 执行反向传播，预期会引发 RuntimeError 异常
        w.backward(torch.ones(5, 5), retain_graph=True)
        # 检查张量 x 的梯度是否为大小为5x5的空张量，填充值为 (1 + e) / 2
        self.assertEqual(x.grad, torch.empty(5, 5).fill_((1 + math.e) / 2))
        # 对 q 执行反向传播，预期会引发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: q.backward(torch.ones(5, 5)))

        # 创建一个大小为5x5的需要梯度追踪的张量 leaf，所有元素为1
        leaf = torch.ones(5, 5, requires_grad=True)
        # 克隆张量 leaf 到 x
        x = leaf.clone()
        # 对 x 执行就地加法操作，添加常数值10
        x.add_(10)
        # 检查张量 x 是否等于大小为5x5的全1张量乘以11
        self.assertEqual(x, torch.ones(5, 5) * 11)
        # 创建一个新的张量 y，作为 x 加 2
        y = x + 2
        # 对 y 执行反向传播，使用全1张量作为梯度
        y.backward(torch.ones(5, 5))
        # 检查张量 leaf 的梯度是否为大小为5x5的全1张量
        self.assertEqual(leaf.grad, torch.ones(5, 5))
        # 计算张量 z = x * y
        z = x * y
        # 对 x 执行就地加法操作，添加常数值2
        x.add_(2)
        # 对 z 执行反向传播，预期会引发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: z.backward(torch.ones(5, 5)))

    # 定义一个测试类内部的静态函数 MyFunction，继承自 torch.autograd.Function
    def test_mark_non_differentiable(self):
        class MyFunction(Function):
            @staticmethod
            # 定义前向传播函数，计算 input 大于0的布尔掩码，并标记为不可求导
            def forward(ctx, input):
                output = input > 0
                ctx.mark_non_differentiable(output)
                return output

            @staticmethod
            # 定义反向传播函数，梯度输出为与输入同型的零张量
            def backward(ctx, grad_output):
                return (grad_output * 0).to(torch.double)

        # 创建一个大小为5x5的随机张量 x，并开启梯度追踪
        x = torch.randn(5, 5, requires_grad=True)
        # 应用自定义函数 MyFunction 到张量 x，生成掩码张量 mask
        mask = MyFunction.apply(x)
        # 检查掩码张量 mask 是否不需要梯度追踪
        self.assertFalse(mask.requires_grad)
        # 创建一个新的张量 y，使用掩码 mask 进行填充，将满足掩码条件的元素置为0
        y = x.masked_fill(mask, 0)
        # 计算张量 y 所有元素的和，并执行反向传播
        y.sum().backward()
    def test_mark_non_differentiable_mixed(self):
        # 定义一个继承自Function的子类MyFunction，用于测试非可微标记
        class MyFunction(Function):
            @staticmethod
            # 前向传播函数，接收输入input，并生成a和b
            def forward(ctx, input):
                # a是input加1
                a = input + 1
                # b是input加2
                b = input + 2
                # 在上下文中标记a为非可微
                ctx.mark_non_differentiable(a)
                return a, b

            @staticmethod
            # 反向传播函数，接收梯度grad_a和grad_b
            def backward(ctx, grad_a, grad_b):
                # 断言grad_a全为0
                self.assertTrue((grad_a == 0).all())
                # 断言grad_b全为1
                self.assertTrue((grad_b == 1).all())
                # 返回grad_b作为反向传播的梯度
                return grad_b

        # 生成一个5x5的随机张量x，要求计算梯度
        x = torch.randn(5, 5, requires_grad=True)
        # 调用MyFunction的apply方法，得到a和b
        a, b = MyFunction.apply(x)
        # 断言a不需要梯度
        self.assertFalse(a.requires_grad)
        # 断言b需要梯度
        self.assertTrue(b.requires_grad)
        # 对b的所有元素求和并进行反向传播
        b.sum().backward()
        # 断言x的梯度为一个5x5的全1张量
        self.assertEqual(x.grad, torch.ones(5, 5))

    def test_mark_non_differentiable_none(self):
        # 测试当返回None时的非可微标记
        # 这段代码之前会由于MyFunction返回空梯度导致段错误，
        # 因为MulBackward函数在C++中实现，期望输入的grad_outputs不为空。
        class MyFunction(Function):
            @staticmethod
            # 前向传播函数，接收输入input，并克隆生成output
            def forward(ctx, input):
                output = input.clone()
                # 在上下文中标记output为非可微
                ctx.mark_non_differentiable(output)
                return output

            @staticmethod
            # 反向传播函数，接收梯度grad_output
            def backward(ctx, grad_output):
                # 返回空梯度
                return None

        # 生成一个5x5的随机张量x，要求计算梯度
        x = torch.randn(5, 5, requires_grad=True)
        # 对x的平方应用MyFunction，得到r
        r = MyFunction.apply(x * x)
        # 对(r * x)的所有元素求和并进行反向传播
        (r * x).sum().backward()

    def test_return_duplicate(self):
        # 测试返回重复输出的情况
        class DoubleDuplicate(Function):
            @staticmethod
            # 前向传播函数，接收输入x，并生成两个相同的output
            def forward(ctx, x):
                output = x * 2
                return output, output

            @staticmethod
            # 反向传播函数，接收两个梯度grad1和grad2
            def backward(ctx, grad1, grad2):
                # 返回grad1和grad2各自乘以2后的和作为反向传播的梯度
                return grad1 * 2 + grad2 * 2

        # 定义一个函数fn，接收输入x，并应用DoubleDuplicate
        def fn(x):
            # 调用DoubleDuplicate的apply方法，得到a和b
            a, b = DoubleDuplicate.apply(x)
            # 断言a和b是同一个对象
            self.assertIs(a, b)
            # 返回a和b的和
            return a + b

        # 生成一个5x5的双精度随机张量x，要求计算梯度
        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        # 对函数fn进行梯度检查
        gradcheck(fn, [x])
        # 对函数fn进行双重梯度检查
        gradgradcheck(fn, [x])

    def test_return_duplicate_inplace(self):
        # 测试返回重复输出的情况（原地操作版本）
        class DoubleInplace(Function):
            @staticmethod
            # 前向传播函数，接收输入x，将x乘以2，标记x为脏数据，并返回x的两个拷贝
            def forward(ctx, x):
                x.mul_(2)
                ctx.mark_dirty(x)
                return x, x

            @staticmethod
            # 反向传播函数，接收两个梯度grad1和grad2
            def backward(ctx, grad1, grad2):
                # 返回grad1和grad2各自乘以2后的和作为反向传播的梯度
                return grad1 * 2 + grad2 * 2

        # 定义一个函数inplace_fn，接收输入x，并应用DoubleInplace
        def inplace_fn(x):
            # 调用DoubleInplace的apply方法，得到a和b
            a, b = DoubleInplace.apply(x.clone())
            # 断言a和b是同一个对象
            self.assertIs(a, b)
            # 返回a和b的和
            return a + b

        # 生成一个5x5的双精度随机张量x，要求计算梯度
        x = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        # 对函数inplace_fn进行梯度检查
        gradcheck(inplace_fn, [x])
        # 对函数inplace_fn进行双重梯度检查
        gradgradcheck(inplace_fn, [x])

        # 无法原地修改叶子变量
        self.assertRaises(RuntimeError, lambda: InplaceFunction.apply(x))
        # 原地修改视图的函数必须只返回一个输出
        self.assertRaises(RuntimeError, lambda: InplaceFunction.apply(x.clone()[0]))
    # 测试函数，用于验证张量的索引赋值操作
    def _test_setitem(self, size, index):
        # 创建一个所有元素为1的张量，并要求计算梯度
        x = torch.ones(*size, requires_grad=True)
        # 对张量进行加法操作
        y = x + 2
        # 记录当前张量的版本号
        y_version = y._version
        # 对张量的指定索引位置赋值为2
        y[index] = 2
        # 断言，验证版本号已经更新
        self.assertNotEqual(y._version, y_version)
        # 反向传播
        y.backward(torch.ones(*size))
        # 创建一个预期的梯度张量，指定索引位置为0
        expected_grad = torch.ones(*size)
        expected_grad[index] = 0
        # 断言，验证梯度计算是否正确
        self.assertEqual(x.grad, expected_grad)
    
    # 测试函数，用于验证张量的索引赋值操作（赋值为张量）
    def _test_setitem_tensor(self, size, index):
        # 创建一个所有元素为1的张量，并要求计算梯度
        x = torch.ones(*size, requires_grad=True)
        # 对张量进行加法操作
        y = x + 2
        # 记录当前张量的版本号
        y_version = y._version
        # 创建一个与x[index]相同形状的张量，所有元素为7，并要求计算梯度
        value = x.new(x[index].size()).fill_(7)
        value.requires_grad = True
        # 对张量的指定索引位置赋值为value
        y[index] = value
        # 断言，验证版本号已经更新
        self.assertNotEqual(y._version, y_version)
        # 反向传播
        y.backward(torch.ones(*size))
        # 创建一个预期的输入张量的梯度，指定索引位置为0
        expected_grad_input = torch.ones(*size)
        expected_grad_input[index] = 0
        # 断言，验证输入张量的梯度计算是否正确
        self.assertEqual(x.grad, expected_grad_input)
        # 断言，验证value张量的梯度是否为全1
        self.assertEqual(value.grad, torch.ones_like(value))
    
        # 情况：当x广播到y[1]时
        x = torch.randn(4, requires_grad=True)
        y = torch.zeros(2, 3, 4)
        # 将x赋值给y的索引1位置
        y[1] = x
        # 反向传播
        y.backward(torch.randn(2, 3, 4))
        # 断言，验证x的大小与其梯度大小是否相同
        self.assertEqual(x.size(), x.grad.size())
    
    # 测试函数，对不同的大小和索引组合进行索引赋值测试
    def test_setitem(self):
        self._test_setitem((5, 5), 1)
        self._test_setitem((5,), 1)
        self._test_setitem((1,), 0)
        self._test_setitem((10,), [[0, 4, 2]])
        self._test_setitem((5, 5), [[0, 4], [2, 2]])
        self._test_setitem((5, 5, 5), [slice(None), slice(None), [1, 3]])
        self._test_setitem((5, 5, 5), [slice(None), [1, 3], slice(None)])
        self._test_setitem((5, 5, 5), [[1, 3], slice(None), slice(None)])
        self._test_setitem((5, 5, 5), [slice(None), [2, 4], [1, 3]])
        self._test_setitem((5, 5, 5), [[1, 3], [2, 4], slice(None)])
        self._test_setitem_tensor((5, 5), 3)
        self._test_setitem_tensor((5, 5), [[0, 1], [1, 0]])
        self._test_setitem_tensor((5,), 3)
        self._test_setitem_tensor((5,), Variable(torch.LongTensor([3]), requires_grad=False).sum())
        self._test_setitem_tensor((5,), [[0, 1, 2, 3]])
        self._test_setitem_tensor((5, 5, 5), [slice(None), slice(None), [1, 3]])
        self._test_setitem_tensor((5, 5, 5), [slice(None), [1, 3], slice(None)])
        self._test_setitem_tensor((5, 5, 5), [[1, 3], slice(None), slice(None)])
        self._test_setitem_tensor((5, 5, 5), [slice(None), [2, 4], [1, 3]])
        self._test_setitem_tensor((5, 5, 5), [[1, 3], [2, 4], slice(None)])
        self._test_setitem_tensor(
            (5, 5, 5),
            [
                Variable(torch.LongTensor([1, 3]), requires_grad=False),
                [2, 4],
                slice(None),
            ],
        )
    def test_setitem_mask(self):
        # 创建一个大小为 5x5 的随机布尔张量，表示掩码
        mask = torch.BoolTensor(5, 5).bernoulli_()
        # 调用 _test_setitem 方法，测试在给定掩码的情况下的操作
        self._test_setitem((5, 5), Variable(mask))
        # 调用 _test_setitem 方法，测试在给定掩码的情况下的操作（针对第一行）
        self._test_setitem((5,), Variable(mask[0]))
        # 调用 _test_setitem 方法，测试在给定掩码的情况下的操作（针对第一个元素）
        self._test_setitem((1,), Variable(mask[0, 0:1]))
        # 调用 _test_setitem_tensor 方法，测试在给定掩码的情况下的操作
        self._test_setitem_tensor((5, 5), Variable(mask))
        # 调用 _test_setitem_tensor 方法，测试在给定掩码的情况下的操作（针对第一行）
        self._test_setitem_tensor((5,), Variable(mask[0]))

    def test_select_sum(self):
        # 创建一个形状为 10x1 的双精度浮点数张量 x，并启用梯度计算
        x = torch.randn(10, dtype=torch.double, requires_grad=True)

        def func(x):
            # 选取张量 x 的第 1 列并求和
            return x.select(0, 1).sum()

        # 对 func 函数进行梯度检查
        gradcheck(func, [x])
        # 对 func 函数进行二阶梯度检查
        gradgradcheck(func, [x])

    def test_diagonal_expanded_v(self):
        # 创建一个随机数值的标量 value
        value = torch.rand([])
        # 创建一个形状为 10 的张量 v_expanded，值全部为 value
        v_expanded = torch.tensor(value).expand(10)
        # 创建一个形状为 10x10 的双精度浮点数张量 a，并启用梯度计算
        a = torch.rand(10, 10, dtype=torch.double, requires_grad=True)
        # 计算 a 对角线的梯度，并使用 v_expanded 作为梯度的值
        (result,) = torch.autograd.grad(a.diagonal(), a, v_expanded)
        # 断言梯度结果与单位矩阵乘以 value 相等
        self.assertEqual(result, torch.eye(10, dtype=torch.double) * value)

    def test_select_expanded_v(self):
        # 创建一个形状为 10x10 的随机张量 v_expanded，用于梯度计算
        v_expanded = torch.rand(10).expand(10, 10)
        # 创建一个形状为 10x10x10 的张量 a，并启用梯度计算
        a = torch.rand(10, 10, 10, requires_grad=True)
        # 计算 a[0] 对 a 的梯度，并使用 v_expanded 作为梯度的值
        (result,) = torch.autograd.grad(a[0], a, v_expanded)
        # 创建一个形状为 10x10x10 的零张量 expected，并将 v_expanded 赋值给第一行
        expected = torch.zeros(10, 10, 10)
        expected[0] = v_expanded
        # 断言梯度结果与预期张量 expected 相等
        self.assertEqual(result, expected)

    def test_slice_expanded_v(self):
        # 创建一个形状为 2x10x10 的随机张量 v_expanded，用于梯度计算
        v_expanded = torch.rand(10, 1).expand(2, 10, 10)
        # 创建一个形状为 10x10x10 的张量 a，并启用梯度计算
        a = torch.rand(10, 10, 10, requires_grad=True)
        # 计算 a[3:5] 对 a 的梯度，并使用 v_expanded 作为梯度的值
        (result,) = torch.autograd.grad(a[3:5], a, v_expanded)
        # 创建一个形状为 10x10x10 的零张量 expected，并将 v_expanded 赋值给第 3 到第 4 行
        expected = torch.zeros(10, 10, 10)
        expected[3:5] = v_expanded
        # 断言梯度结果与预期张量 expected 相等
        self.assertEqual(result, expected)

    def test_unused_output(self):
        # 创建一个形状为 10x10 的随机张量 x，并启用梯度计算
        x = torch.randn(10, 10, requires_grad=True)
        # 将张量 x 切分成 5 个张量组成的列表 outputs
        outputs = x.chunk(5)
        # 获取 outputs 列表中的第 2 个张量 o
        o = outputs[2]
        # 对 o 进行乘法和加法运算
        o = o * 4 + 2
        # 对 o 的和进行反向传播
        o.sum().backward()
        # 创建一个形状为 10x10 的零张量 expected_grad，并在第 4 到第 5 行赋值为 4
        expected_grad = torch.zeros(10, 10)
        expected_grad[4:6] = 4
        # 断言张量 x 的梯度与预期张量 expected_grad 相等
        self.assertEqual(x.grad, expected_grad)

        with torch.no_grad():
            x.grad.zero_()
        # 创建一个形状为 2x10 的随机张量 grad_output
        grad_output = torch.randn(2, 10)
        # 再次将张量 x 切分成 5 个张量组成的列表 outputs
        outputs = x.chunk(5)
        # 对 outputs[0] 执行反向传播，使用 grad_output 作为梯度
        outputs[0].backward(grad_output)
        # 创建一个形状为 10x10 的零张量 expected_grad，并在前 2 行赋值为 grad_output
        expected_grad = torch.zeros(10, 10)
        expected_grad[:2] = grad_output
        # 断言张量 x 的梯度与预期张量 expected_grad 相等
        self.assertEqual(x.grad, expected_grad)

    # TODO: opinfo this or move to the sparse test suite
    def _test_sparse_gather(self, size_x, size_ind, dim):
        # 创建一个形状为 size_x 的随机张量 x，并启用梯度计算
        x = torch.randn(size_x, requires_grad=True)
        # 如果 size_ind 的长度大于 0 且 size_x 的长度大于 0，则创建一个随机整数张量 ind
        if len(size_ind) > 0 and len(size_x) > 0:
            ind = torch.randint(x.size(dim), size_ind)
        else:
            ind = torch.zeros(size_ind, dtype=torch.int64)
        # 使用 torch.gather 对张量 x 在指定维度 dim 上进行聚合操作，不使用稀疏梯度
        out = torch.gather(x, dim, ind, sparse_grad=False)
        # 创建一个形状与 out 相同的随机张量 grad，并对 out 执行反向传播
        grad = torch.rand_like(out)
        out.backward(grad)
        # 将稠密梯度保存到 grad_dense
        grad_dense = x.grad.clone()
        x.grad = None
        # 使用 torch.gather 对张量 x 在指定维度 dim 上进行聚合操作，使用稀疏梯度
        out = torch.gather(x, dim, ind, sparse_grad=True)
        out.backward(grad)
        # 断言稠密梯度与稀疏梯度的结果相等
        self.assertEqual(grad_dense, x.grad.to_dense())
    # 测试稀疏数据的聚合功能，按照指定的维度进行测试，这里是维度0的测试
    def test_sparse_gather_dim0(self):
        self._test_sparse_gather((10, 10), (5, 10), 0)

    # 测试稀疏数据的聚合功能，按照指定的维度进行测试，这里是维度1的测试
    def test_sparse_gather_dim1(self):
        self._test_sparse_gather((10, 10, 5), (10, 5, 5), 1)

    # 测试稀疏数据的聚合功能，按照指定的维度进行测试，这里是负数维度的测试
    def test_sparse_gather_dim_neg(self):
        self._test_sparse_gather((10, 10, 5), (10, 10, 2), -1)

    # 测试稀疏数据的聚合功能，当索引是标量时进行测试
    def test_sparse_gather_ind_scalar(self):
        self._test_sparse_gather((10,), (), 0)

    # 测试稀疏数据的聚合功能，当数据是标量时进行测试
    def test_sparse_gather_x_scalar(self):
        self._test_sparse_gather((), (2,), 0)

    # 测试稀疏数据的聚合功能，当索引和数据都是标量时进行测试
    def test_sparse_gather_both_scalar(self):
        self._test_sparse_gather((), (), 0)

    # 测试在函数析构函数中进行垃圾收集时的情况
    def test_gc_in_destructor(self):
        """
        之前，如果函数的析构函数触发了垃圾收集，变量的 tp_dealloc 处理程序会被调用两次，导致段错误。
        """

        # 定义一个函数类，继承自 Function
        class CollectOnDelete(Function):
            def forward(self, x):
                return x

            def backward(self, grad_output):
                return grad_output

            # 在对象销毁时执行垃圾收集
            def __del__(self):
                gc.collect()

        # 进行多次实例化、前向传播、反向传播的测试
        for _ in range(10):
            CollectOnDelete().forward(torch.randn(1, requires_grad=True)).backward()

    # 测试自动求导函数属性访问时的不当行为
    def test_naughty_autograd_function_attribute_access(self):
        # 定义一个标识函数类，继承自 Function
        class Id(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad_x):
                return grad_x

        # 断言应该发出特定的警告信息
        with self.assertWarnsRegex(DeprecationWarning, "should not be instantiated"):
            f = Id()

        # 确认警告后，依然返回一个实例
        self.assertIsInstance(f, Id)

        # 创建一个需要梯度的零张量
        x = torch.zeros(1, requires_grad=True)

        # 断言在调用非静态前向方法时会触发运行时错误
        with self.assertRaisesRegex(
            RuntimeError, "non-static forward method is deprecated"
        ):
            f(x)

        # 对标识函数应用到张量上，进行后向传播，并检查梯度函数名称
        t = Id.apply(x)
        self.assertEqual(t.grad_fn.name(), "IdBackward")

        # THPFunction 是 grad_fn 和自动求导函数的基类，许多对它们的访问可能导致段错误。测试我们能否正确报错。
        t = torch.ones(1, requires_grad=True)
        t._backward_hooks = {}
        with self.assertRaisesRegex(
            RuntimeError, "Attribute '_register_hook_dict' is invalid"
        ):
            f._register_hook_dict(t)
        with self.assertRaisesRegex(
            RuntimeError, "Attribute 'register_hook' is invalid"
        ):
            f.register_hook(lambda x, y: None)
        with self.assertRaisesRegex(
            RuntimeError, "Attribute 'next_functions' is invalid"
        ):
            f.next_functions
        with self.assertRaisesRegex(RuntimeError, "Attribute 'name' is invalid"):
            f.name()
        with self.assertRaisesRegex(
            RuntimeError, "underlying PyNode has already been deallocated"
        ):
            f.metadata

    @unittest.expectedFailure
    def test_naughty_anomaly_access(self):
        # 定义一个继承自 torch.autograd.Function 的自定义函数 MyFunction
        class MyFunction(Function):
            @staticmethod
            # 前向传播方法：返回输入 x 自身
            def forward(ctx, x):
                return x

            @staticmethod
            # 反向传播方法：返回梯度 g 自身
            def backward(ctx, g):
                return g

        # 创建一个 requires_grad=True 的零张量 x
        x = torch.zeros(1, requires_grad=True)
        # 使用自定义函数 MyFunction 对 x 进行处理得到 y
        y = MyFunction.apply(x)
        # 对 y 进行反向传播
        y.backward()
        # 获取 y 的梯度函数 g
        g = y.grad_fn
        # 删除变量 y
        del y
        # 访问 g 的 metadata 属性，这个操作目前会失败，但不应该
        g.metadata  # this currently fails, but shouldn't

    def test_naughty_autograd_function_stashing_ctx(self):
        # 定义一个保存上下文的列表 saved_ctx
        saved_ctx = []

        # 定义一个继承自 torch.autograd.Function 的自定义函数 Id
        class Id(Function):
            @staticmethod
            # 前向传播方法：保存输入 x 的值到上下文中并返回 x 自身
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            # 反向传播方法：将上下文 ctx 加入 saved_ctx 列表，并返回保存的张量
            def backward(ctx, grad_x):
                saved_ctx.append(ctx)
                return ctx.saved_tensors

        # 创建一个 requires_grad=True 的零张量 p
        p = torch.zeros(1, requires_grad=True)
        # 使用自定义函数 Id 对 p 进行处理得到 loss
        loss = Id.apply(p)
        # 对 loss 进行反向传播，保留计算图
        loss.backward(retain_graph=True)
        # 删除变量 loss
        del loss
        # 这一时刻，会报告计算图已经被释放
        # （虽然这是间接表述问题的方式）
        self.assertRaises(RuntimeError, lambda: saved_ctx[0].saved_tensors)

    def test_custom_autograd_repeated_grad_grad(self):
        # 这个测试在 PR #22983 中的相等性检查失败；这是一个值得保留的有趣和不同的测试案例。
        # mult1 并没有测试什么有趣的内容，但 mult2 是有趣的情况。

        # 定义一个函数 mult1，计算输入张量 x 沿着指定维度的乘积
        def mult1(x):
            return x.prod(dim=-1).prod(dim=-1)

        # 定义一个继承自 torch.autograd.Function 的自定义函数 Mult
        class Mult(torch.autograd.Function):
            @staticmethod
            # 前向传播方法：调用 mult1 计算结果 y，并保存输入 x 和 y 到上下文中
            def forward(ctx, x):
                y = mult1(x)
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            # 反向传播方法：根据保存的 x 和 y 计算梯度并返回
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return (grad_output * y)[:, None, None] / x

        # 定义一个函数 mult2，使用 Mult.apply 对输入张量 x 进行处理
        mult2 = Mult.apply

        # 定义一个函数 check_gradgrad_repeated，检查重复计算梯度的情况
        def check_gradgrad_repeated(x, y):
            # 计算 y 对 x 的梯度，并保留计算图
            (gy,) = torch.autograd.grad(y[0], x, create_graph=True)
            # 计算 gy[0, 0, 0] 对 x 的梯度，并保留计算图
            (ggy_1,) = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            # 再次计算 y 对 x 的梯度，并保留计算图
            (gy,) = torch.autograd.grad(y[0], x, create_graph=True)
            # 再次计算 gy[0, 0, 0] 对 x 的梯度，并保留计算图
            (ggy_2,) = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            # 断言 ggy_1 和 ggy_2 在指定位置上相等
            self.assertEqual(ggy_1[0, 0, 1], ggy_2[0, 0, 1])

        # 创建一个 requires_grad=True 的全为 1 的张量 x
        x = torch.ones(2, 4, 4).requires_grad_()
        # 使用 check_gradgrad_repeated 分别检查 mult1(x) 和 mult2(x)
        check_gradgrad_repeated(x, mult1(x))
        check_gradgrad_repeated(x, mult2(x))
    def test_custom_autograd_no_early_free(self):
        # 此测试失败，报告缓冲区在 #22983 之前已被释放的问题。也是一个非常有趣的测试案例。
        class Double(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x**2
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, _ = ctx.saved_tensors
                return grad_output * 2 * x

        # 这段代码等效于上面的代码，但在 .backward() 中使用了 .forward() 的输出。
        class Double2(Double):
            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output * 2 * y / x

        double = Double.apply
        double2 = Double2.apply

        x = torch.tensor(2).double().requires_grad_()

        self.assertTrue(gradcheck(double, x))
        self.assertTrue(gradgradcheck(double, x))
        self.assertTrue(gradcheck(double2, x))
        self.assertTrue(gradgradcheck(double2, x))

        y = double(x)
        torch.autograd.grad(y, x, create_graph=True)
        torch.autograd.grad(y, x)

        y = double2(x)
        torch.autograd.grad(y, x, create_graph=True)
        torch.autograd.grad(y, x)  # 这里不应该报错！

    def test_detach(self):
        x = torch.randn(10, 10, requires_grad=True)
        y = x + 2
        y = y.detach()
        z = y * 4 + 2
        self.assertFalse(y.requires_grad)
        self.assertFalse(z.requires_grad)

        x = torch.randn(10, 10, requires_grad=True)
        y = x * 2
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)
        z = x + y
        z.sum().backward()
        # 这是一个不正确的梯度计算，但我们假设这是用户想要的。detach() 是一个高级选项。
        self.assertEqual(x.grad, torch.ones(10, 10))

        # 就地执行 detach
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        a = x * 2
        (y + a).sum().backward(retain_graph=True)
        a.detach_()
        self.assertFalse(a.requires_grad)
        (y + a).sum().backward()  # 这不会反向传播到 x
        self.assertEqual(x.grad, torch.ones(10, 10) * 2)
        self.assertEqual(y.grad, torch.ones(10, 10) * 2)

        # 在视图上执行就地 detach 会引发异常
        view = x.narrow(0, 1, 4)
        self.assertRaisesRegex(RuntimeError, "view", lambda: view.detach_())

    def test_detach_base(self):
        "detaching base does not detach view"
        x = torch.randn(10, 10, requires_grad=True)
        view = x.narrow(0, 1, 4)
        x.detach_()
        self.assertFalse(x.requires_grad)
        self.assertTrue(view.requires_grad)
        self.assertIsNotNone(view.grad_fn)
        self.assertIs(view._base, x)
    # 测试在自动求导中分离后立即就地操作会引发异常
    def test_detach_then_inplace_raises_in_autograd(self):
        # 创建一个随机张量 x，并要求进行梯度计算
        x = torch.randn([], requires_grad=True)
        # 对 x 进行分离并克隆，保存原始的 x
        orig_x = x.detach().clone()

        # 计算 y = x**2，这里会保存 x
        y = x**2
        # 对 x 进行分离
        z = x.detach()
        # 将 z 清零（就地操作）
        z.zero_()
        # 使用断言确保在自动求导时会抛出 RuntimeError 异常，异常信息中包含 "has been modified by an inplace"
        with self.assertRaisesRegex(RuntimeError, "has been modified by an inplace"):
            y.backward()

    # 测试类型转换后的反向传播
    def _test_type_conversion_backward(
        self,
        t,
    ):
        # 创建一个浮点数张量，并要求进行梯度计算
        fvar = Variable(t(torch.randn(5, 5).float()), requires_grad=True)
        # 将张量类型转换为 double，并计算其和，然后进行反向传播
        fvar.double().sum().backward()
        # 使用断言确保梯度计算的结果是张量 fvar 形状相同的全一张量
        self.assertEqual(fvar.grad, torch.ones_like(fvar))
        # 使用断言确保梯度的类型与张量 fvar 的类型相同
        self.assertEqual(type(fvar.grad), type(fvar))
        
        # 创建一个双精度张量，并要求进行梯度计算
        dvar = Variable(t(torch.randn(5, 5).double()), requires_grad=True)
        # 将张量类型转换为 float，并计算其和，然后进行反向传播
        dvar.float().sum().backward()
        # 使用断言确保梯度计算的结果是张量 dvar 形状相同的全一张量
        self.assertEqual(dvar.grad, torch.ones_like(dvar))
        # 使用断言确保梯度的类型与张量 dvar 的类型相同
        self.assertEqual(type(dvar.grad), type(dvar))

    # 测试孤立节点的自动求导
    def test_isolated_node(self):
        # 创建两个随机张量 x 和 y，并要求进行梯度计算
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        # 计算张量 a = x + y
        a = x + y
        # 计算张量 b = a 沿第一维度取最大值的索引，并重复为 (1, 5)，类型为 double
        b = torch.max(a, 1, True)[1].repeat(1, 5).double()
        # 计算 o = b + a 的和
        o = (b + a).sum()
        # 对 o 进行反向传播
        o.backward()

    # 测试张量形状的断言
    def test_shape(self):
        # 创建一个形状为 (3, 4) 的随机张量 x
        x = torch.randn(3, 4)
        # 使用断言确保张量 x 的维度是 2
        self.assertEqual(2, len(x.shape))
        # 使用断言确保张量 x 的第一个维度为 3
        self.assertEqual(x.shape[0], 3)
        # 使用断言确保张量 x 的第二个维度为 4
        self.assertEqual(x.shape[1], 4)

    # 测试在需要梯度的张量上调用 numpy 函数会引发异常
    def test_numpy_requires_grad(self):
        # 创建一个形状为 (2, 2) 的随机张量 x，并要求进行梯度计算
        x = torch.randn(2, 2, requires_grad=True)
        # 定义错误信息模式
        err_msg_outputs = r"Can't call numpy\(\) on Tensor that requires grad. Use tensor.detach\(\).numpy\(\) instead."
        # 使用断言确保在调用 x.numpy() 时会抛出 RuntimeError 异常，异常信息匹配 err_msg_outputs
        with self.assertRaisesRegex(RuntimeError, err_msg_outputs):
            x.numpy()

        # 使用 torch.no_grad() 上下文环境，调用 x.numpy()，应该不会抛出异常
        with torch.no_grad():
            x.numpy()

        # 创建一个形状为 (2, 2) 的随机张量 x
        x = torch.randn(2, 2)
        # 调用 x.numpy()，应该不会抛出异常
        x.numpy()

        # 使用 torch.no_grad() 上下文环境，调用 x.numpy()，应该不会抛出异常
        with torch.no_grad():
            x.numpy()

    # 测试返回叶子节点
    def test_return_leaf(self):
        # 定义一个自定义函数 Identity，继承自 Function
        class Identity(Function):
            @staticmethod
            # 前向传播方法，接受两个输入参数 a 和 b
            def forward(ctx, a, b):
                return a, a + b

            @staticmethod
            # 反向传播方法，接受两个梯度参数 grad_a 和 grad_b
            def backward(ctx, grad_a, grad_b):
                return grad_a + grad_b, grad_b

        # 创建两个形状为 (5, 5) 的随机张量 x 和 y，并要求进行梯度计算
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)

        # 使用自定义函数 Identity 的 apply 方法计算 q 和 p
        q, p = Identity.apply(x, y)

        # 定义一个钩子函数 hook，用于验证梯度传播过程中是否只收到了 q 的梯度，而非 x 的
        def hook(grad):
            hook_called[0] = True
            self.assertEqual(grad, torch.ones(5, 5))

        # 注册钩子函数到张量 q 上
        q.register_hook(hook)
        # 计算 (q + p + x) 的和，并进行反向传播
        (q + p + x).sum().backward()
        # 使用断言确保张量 x 的梯度是全一张量乘以 3
        self.assertEqual(x.grad, torch.ones(5, 5) * 3)
        # 使用断言确保张量 y 的梯度是全一张量
        self.assertEqual(y.grad, torch.ones(5, 5))
        # 使用断言确保钩子函数被调用过
        self.assertTrue(hook_called[0])
    def test_return_leaf_inplace(self):
        # 定义一个继承自 InplaceFunction 的子类 Inplace
        class Inplace(InplaceFunction):
            # 静态方法：前向传播计算，标记 a 为脏数据，返回 a 加上 b，以及 b 加 2 的结果
            @staticmethod
            def forward(ctx, a, b):
                ctx.mark_dirty(a)
                return a.add_(b), b + 2

            # 静态方法：反向传播计算，返回 grad_a 和 grad_b 的结果
            @staticmethod
            def backward(ctx, grad_a, grad_b):
                return grad_a, grad_a + grad_b

        # 创建一个大小为 5x5 的随机张量 x
        x = torch.randn(5, 5)
        # 创建一个大小为 5x5、需要梯度的随机张量 y
        y = torch.randn(5, 5, requires_grad=True)

        # 调用 Inplace 的 apply 方法，传入 x 和 y，返回 q 和 p
        q, p = Inplace.apply(x, y)
        # 断言 q 等于 x
        self.assertIs(q, x)
        # 断言 q 的梯度函数的类为 Inplace 的反向传播类
        self.assertIs(q.grad_fn.__class__, Inplace._backward_cls)
        # 断言 q 需要梯度
        self.assertTrue(q.requires_grad)
        # 对 q 的所有元素求和并进行反向传播
        q.sum().backward()
        # 断言 y 的梯度为一个全为 1 的 5x5 张量
        self.assertEqual(y.grad, torch.ones(5, 5))

    def test_leaf_assignment(self):
        # 创建一个大小为 5x5 的随机张量 x
        x = torch.randn(5, 5)
        # 创建一个大小为 5 的随机张量 y，并标记需要梯度
        y = torch.randn(5, requires_grad=True)
        # 创建一个大小为 5 的随机张量 z，并标记需要梯度
        z = torch.randn(5, requires_grad=True)

        # 修改 x 的第一行为 y
        x[0] = y
        # 修改 x 的第二行为 2 倍的 z
        x[1] = 2 * z
        # 断言 x 需要梯度
        self.assertTrue(x.requires_grad)
        # 断言 x 的梯度函数不为 None
        self.assertIsNot(x.grad_fn, None)
        # 对 x 的所有元素求和并进行反向传播
        x.sum().backward()
        # 断言 y 的梯度为一个全为 1 的 5 维张量
        self.assertEqual(y.grad, torch.ones(5))
        # 断言 z 的梯度为一个全为 2 的 5 维张量
        self.assertEqual(z.grad, torch.ones(5) * 2)

    def test_no_grad_assignment(self):
        # 创建一个大小为 5x5、需要梯度的随机张量 x
        x = torch.randn(5, 5, requires_grad=True)
        # 创建一个大小为 5 的随机张量 y
        y = torch.randn(5)
        # 使用 torch.no_grad 上下文环境
        with torch.no_grad():
            # 修改 x 的第一行为 y
            x[0] = y

        # 断言 x 需要梯度
        self.assertTrue(x.requires_grad)
        # 断言 x 的梯度函数为 None
        self.assertIsNone(x.grad_fn)

    def test_no_grad_modifies_version(self):
        # 创建一个大小为 5、需要梯度的随机张量 x
        x = torch.randn(5, requires_grad=True)
        # 创建一个大小为 5、需要梯度的随机张量 y
        y = torch.randn(5, requires_grad=True)
        # 计算 x 和 y 的乘积并求和
        z = (x * y).sum()
        # 使用 torch.no_grad 上下文环境
        with torch.no_grad():
            # 对 x 的所有元素乘以 2
            x *= 2
        # 断言在调用 z 的 backward 方法时会引发 RuntimeError，错误信息为 "modified by an inplace operation"
        self.assertRaisesRegex(
            RuntimeError, "modified by an inplace operation", lambda: z.backward()
        )

    def test_increment_version(self):
        # 创建一个大小为 5、需要梯度的随机张量 a
        a = torch.rand(5, requires_grad=True)
        # 记录 a 的版本号
        v = a._version
        # 增加 a 的版本号
        torch.autograd.graph.increment_version(a)
        # 断言 a 的版本号增加了 1
        self.assertEqual(a._version, v + 1)

        # 创建一个大小为 5、整型零张量 a
        a = torch.zeros(5, dtype=torch.int)
        # 记录 a 的版本号
        v = a._version
        # 增加 a 的版本号
        torch.autograd.graph.increment_version(a)
        # 断言 a 的版本号增加了 1
        self.assertEqual(a._version, v + 1)

        # 使用 torch.inference_mode 上下文环境
        with torch.inference_mode():
            # 创建一个大小为 5、需要梯度的随机张量 a
            a = torch.rand(5, requires_grad=True)
        # 断言调用 torch.autograd.graph.increment_version(a) 时会引发 RuntimeError，错误信息为 "update to inference tensor outside InferenceMode"
        msg = "update to inference tensor outside InferenceMode"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.autograd.graph.increment_version(a)

    def test_no_grad_input(self):
        # 定义一个继承自 Function 的子类 MyFunction
        class MyFunction(Function):
            # 静态方法：前向传播计算，返回输入 x 本身
            @staticmethod
            def forward(self, x):
                return x

            # 静态方法：反向传播计算，返回梯度输出 grad_output 本身
            @staticmethod
            def backward(self, grad_output):
                return grad_output

        # 创建一个大小为 5、需要梯度的随机张量 x
        x = torch.randn(5, requires_grad=True)
        # 使用 torch.no_grad 上下文环境
        with torch.no_grad():
            # 调用 MyFunction 的 apply 方法，传入 x，返回 y
            y = MyFunction.apply(x)

        # 断言 x 需要梯度
        self.assertTrue(x.requires_grad)
        # 断言 y 的梯度函数为 None
        self.assertIsNone(y.grad_fn)
    def test_backward_copy(self):
        # 这个测试用例检查了一个非常微妙的 bug，出现在 autograd 的早期版本中。
        # 梯度张量最初存储在列表中，函数等待所有梯度都被计算。然而，有时一个输出会被多次使用，
        # 因此需要对梯度进行求和。引擎过去保持了一个需要克隆的张量集合，在下一次添加时需要进行克隆，
        # 并在克隆完成后从集合中删除它们。然而，如果同一个梯度张量在图中出现了三次，可能会导致错误的结果：
        # 1. 在其中一个地方累积梯度时，它被克隆并从需要克隆的集合中移除。
        # 2. 在第二个地方累积时，它不在需要克隆的集合中，因此梯度会被简单地原地累积（这已经修改了第三个地方的梯度）。
        # 3. 在第三个地方累积时，它同样不在需要克隆的集合中，因此传入的梯度会被原地求和，导致除第一个外的所有函数产生错误的结果。
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5, requires_grad=True)
        # 模拟我们在图中间的情况
        a = x + 2
        b = y + 2
        c = x + 2
        # 这个操作将在反向传播时两次返回 grad_output
        add1 = a + b
        add2 = add1 + c
        # 模拟一个长分支，使 grad_output 被缓存
        for _ in range(4):
            a = a * 2
            b = b * 2
            c = c * 2
        branch = a + b + c
        out = add2 + branch
        # 预期的梯度分别是：
        # 对于 x: 34（来自最终的 a 的 16，最终的 c 的 16，add2 的 2）
        # 对于 y: 17（来自最终的 b 的 16，add2 的 1）
        grad_output = torch.ones(5, 5)
        out.backward(grad_output)
        self.assertEqual(x.grad, torch.ones(5, 5) * 34)
        self.assertEqual(y.grad, torch.ones(5, 5) * 17)

    def test_save_none_for_backward(self):
        test_case = self

        class MyFn(Function):
            @staticmethod
            def forward(ctx, input):
                # 在 forward 方法中，保存 None, input, None 作为上下文的保存张量
                ctx.save_for_backward(None, input, None)
                return input * input

            @staticmethod
            def backward(ctx, grad_output):
                # 在 backward 方法中，获取保存的张量并验证其中的两个为 None
                n1, input, n2 = ctx.saved_tensors
                test_case.assertIsNone(n1)
                test_case.assertIsNone(n2)
                return 2 * input * grad_output

        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn.apply(x)
        y.sum().backward()
        self.assertEqual(x.grad, 2 * x)
    # 定义一个测试函数，用于测试梯度过多的情况
    def test_too_many_grads(self):
        # 定义一个自定义的函数类 MyFn，继承自 Function
        class MyFn(Function):
            @staticmethod
            # 前向传播函数，接受输入并原样返回
            def forward(ctx, input):
                return input

            @staticmethod
            # 反向传播函数，接受梯度输出并返回原梯度以及空的其他梯度信息
            def backward(ctx, grad_output):
                return grad_output, None, None

        # 生成一个大小为 5x5 的随机张量 x，并要求计算其梯度
        x = torch.randn(5, 5, requires_grad=True)
        # 调用自定义函数 MyFn 的 apply 方法，将 x 作为输入并返回结果 y
        y = MyFn.apply(x)
        # 对 y 的所有元素求和，并进行反向传播
        y.sum().backward()
        # 断言 x 的梯度与与其形状相同且全为 1 的张量相等
        self.assertEqual(x.grad, torch.ones_like(x))

    # 定义一个测试函数，用于测试 pickle 序列化与反序列化
    def test_pickle(self):
        # 生成两个大小为 10x10 的随机张量 x（要求梯度）和 y（不要求梯度）
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=False)

        # 定义一个函数 assert_strict_equal，用于断言两个变量相等且其梯度要求一致
        def assert_strict_equal(var1, var2):
            self.assertEqual(var1, var2)
            self.assertEqual(var1.requires_grad, var2.requires_grad)

        # 使用 pickle 序列化并反序列化包含 x 和 y 的列表，使用不同的协议（0、1、2）
        serialized = [pickle.dumps([x, y], protocol=p) for p in range(3)]
        for dump in serialized:
            xc, yc = pickle.loads(dump)
            # 断言反序列化后的 xc 和 yc 与原始的 x 和 y 相等，并且其梯度要求一致
            assert_strict_equal(xc, x)
            assert_strict_equal(yc, y)

    # 定义一个测试函数，用于测试依赖关系中的不需要梯度的情况
    def test_dep_nograd(self):
        # 定义一个自定义函数类 F1，继承自 Function
        class F1(Function):
            @staticmethod
            # 前向传播函数，生成一个与输入相同大小的随机张量 out，并标记为不需要梯度
            def forward(ctx, input):
                out = torch.randn(input.size())
                ctx.mark_non_differentiable(out)
                return input, out

            @staticmethod
            # 反向传播函数，接受梯度输出和忽略的参数，并返回梯度输出
            def backward(ctx, grad_output, ignored):
                return grad_output

        # 定义一个自定义函数类 F2，继承自 Function
        class F2(Function):
            @staticmethod
            # 前向传播函数，接受输入和忽略的参数，并返回输入
            def forward(ctx, input, ignored):
                return input

            @staticmethod
            # 反向传播函数，接受梯度输出并返回梯度输出以及空的梯度信息
            def backward(ctx, grad_output):
                return grad_output, None

        # 生成一个大小为 5 的随机张量 x，并要求计算其梯度
        x = torch.randn(5, requires_grad=True)
        # 调用自定义函数 F1 的 apply 方法，将 x 作为输入，并获得返回值 a 和 b
        a, b = F1.apply(x)
        # 将 b 增加 1，以区分 F1 和 F2 之间的另一个操作
        b = b + 1
        # 断言 a 需要梯度，而 b 不需要梯度
        self.assertTrue(a.requires_grad)
        self.assertFalse(b.requires_grad)
        # 调用自定义函数 F2 的 apply 方法，将 a 和 b 作为输入，并返回结果 c
        c = F2.apply(a, b)
        # 对 c 进行反向传播，梯度为一个与 c 形状相同且所有元素为 1 的张量
        c.backward(torch.ones(c.size()))
        # 断言 x 的梯度与其形状相同且所有元素为 1 的张量相等
        self.assertEqual(x.grad, torch.ones(x.size()))

    # 定义一个测试函数，用于测试动态开启和关闭梯度跟踪功能
    def test_set_grad_enabled(self):
        # 生成一个大小为 1 的张量 x，并要求计算其梯度
        x = torch.tensor([1.0], requires_grad=True)
        # 使用 torch.set_grad_enabled(False) 包裹的上下文，生成 y = x * 2
        with torch.set_grad_enabled(False):
            y = x * 2
        # 断言 y 不需要梯度
        self.assertFalse(y.requires_grad)
        # 使用 torch.set_grad_enabled(True) 包裹的上下文，生成 y = x * 2
        with torch.set_grad_enabled(True):
            y = x * 2
        # 断言 y 需要梯度
        self.assertTrue(y.requires_grad)
        # 使用 torch.set_grad_enabled(False) 包裹的上下文内部，动态开启梯度跟踪功能
        with torch.set_grad_enabled(False):
            torch.set_grad_enabled(True)
            y = x * 2
        # 断言 y 需要梯度
        self.assertTrue(y.requires_grad)
    def test_set_grad_enabled_wraps(self):
        for decorator in [True, False]:
            with torch.enable_grad():
                self.assertTrue(torch.is_grad_enabled())

                if decorator:
                    # 如果 decorator 为 True，则创建一个函数 inner_func，并使用 @torch.set_grad_enabled(False) 装饰器来禁用梯度计算。
                    # 这不会改变全局的梯度模式！
                    @torch.set_grad_enabled(False)
                    def inner_func(x):
                        return x.sin()

                else:
                    # 如果 decorator 为 False，则定义一个函数 inner_func 来计算输入张量 x 的正弦值。
                    def inner_func(x):
                        return x.sin()

                    # 非惯用用法！
                    # 更惯用的用法是使用 torch.set_grad_enabled(False)(inner_func)
                    # 创建一个 torch.set_grad_enabled(False) 的对象，并验证是否成功禁用了梯度计算。
                    obj = torch.set_grad_enabled(False)
                    self.assertTrue(not torch.is_grad_enabled())

                    # 这会消耗 set_grad_enabled 全局变量的修改！
                    # 将 inner_func 函数重新赋值为 obj(inner_func)，以重新启用梯度计算。
                    inner_func = obj(inner_func)
                    self.assertTrue(torch.is_grad_enabled())

                self.assertTrue(torch.is_grad_enabled())

                # 创建一个 requires_grad=True 的零张量 x。
                x = torch.zeros(1, requires_grad=True)
                # 验证 inner_func(x) 的 requires_grad 属性是否为 False。
                self.assertTrue(not inner_func(x).requires_grad)

    def test_simple_reentrant(self):
        # 创建一个随机的 y_data 张量。
        y_data = torch.randn(2, 2)

        class Reenter(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    # 在 forward 方法中，创建变量 ctx.x 和 ctx.y，并设置其 requires_grad=True。
                    ctx.x = Variable(x, requires_grad=True)
                    ctx.y = Variable(y_data, requires_grad=True)
                    # 计算 ctx.x 与 ctx.y 的乘积，并存储在 ctx.output_var 中。
                    ctx.output_var = ctx.x * ctx.y
                # 返回 ctx.output_var 的副本。
                return ctx.output_var.detach()

            @staticmethod
            def backward(ctx, grad_output):
                with torch.enable_grad():
                    # 在 backward 方法中，对 ctx.output_var 的和进行反向传播。
                    ctx.output_var.sum().backward()
                # 返回 ctx.x 的梯度乘以 grad_output。
                return ctx.x.grad * grad_output

        # 创建一个 requires_grad=True 的随机张量 x。
        x = torch.randn(2, 2, requires_grad=True)
        # 应用 Reenter 的静态方法 apply，计算输出 out。
        out = Reenter.apply(x)
        # 对 out 的和进行反向传播。
        out.sum().backward()
        # 验证 x 的梯度是否等于 y_data。
        self.assertEqual(x.grad, y_data)

    def test_reentrant_child_error(self):
        # 创建一个 requires_grad=True 的随机张量 a，并计算 c = a * a。
        a = torch.rand(3, 3, requires_grad=True)
        c = a * a

        # 创建一个 requires_grad=True 的随机张量 b，并计算 e = b * b。
        b = torch.rand(3, 3, requires_grad=True)
        e = b * b
        # 对 e 应用 TestAutograd.SimulateBackwardError 的 apply 方法，得到 f。
        f = TestAutograd.SimulateBackwardError.apply(e)
        # 计算 f 的和，作为 reentrant_root。
        reentrant_root = f.sum()

        class ReentrantFunc(Function):
            @staticmethod
            def forward(ctx, inp):
                # 在 forward 方法中，返回输入 inp 的克隆。
                return inp.clone()

            @staticmethod
            def backward(ctx, grad):
                # 在 backward 方法中，对 reentrant_root 进行反向传播。
                reentrant_root.backward()
                return grad

        # 对 c 应用 ReentrantFunc 的 apply 方法，得到 d。
        d = ReentrantFunc.apply(c)
        # 使用 self.assertRaisesRegex 来验证在执行 d.sum().backward() 时是否抛出了异常，并且异常信息包含 "Simulate error"。
        with self.assertRaisesRegex(Exception, "Simulate error"):
            d.sum().backward()
    # 定义测试方法，验证 torch.var_mean 的行为是否可微分
    def test_var_mean_differentiable(self):
        # 定义维度列表
        dim = [2, 4]
        # 是否保持维度标志
        keepdim = False
        # 创建具有随机值的张量，标记为需要梯度
        input1 = torch.randn(3, 4, 5, 6, 2, 3, requires_grad=True)
        # 深拷贝 input1，创建 input2
        input2 = deepcopy(input1)
        # 调用 torch.var_mean 计算方差和均值
        var1, mean1 = torch.var_mean(input1, dim=dim, keepdim=keepdim)
        # 使用 input2 计算方差
        var2 = input2.var(dim=dim, keepdim=keepdim)
        # 使用 input2 计算均值
        mean2 = input2.mean(dim=dim, keepdim=keepdim)
        # 创建具有随机梯度的张量
        grad = torch.randn(3, 4, 6, 3, requires_grad=True)

        # 计算 r1 和 r2，以验证它们是否相等
        r1 = var1 * var1 * mean1 * mean1
        r2 = var2 * var2 * mean2 * mean2
        # 断言 r1 和 r2 在指定的相对和绝对容差下相等
        self.assertEqual(r1, r2, rtol=0.01, atol=0.0)

        # 对 r1 和 r2 执行反向传播
        torch.autograd.backward(r1, grad)
        torch.autograd.backward(r2, grad)
        # 断言 input1 和 input2 的梯度是否相等
        self.assertEqual(input1.grad, input2.grad, rtol=0.01, atol=0.0)

    # 使用装饰器 @skipIfNoLapack 进行条件跳过测试
    # 定义一个测试函数，用于测试 torch.lobpcg 的功能
    def test_lobpcg(self):
        # 定义一个内部函数 func，用于执行 lobpcg 算法
        def func(k, A, largest=True, B=None):
            # 获取 A 的形状并转换为列表
            X_shape = list(A.shape)
            # 将 X_shape 的最后一个维度设置为 k
            X_shape[-1] = k
            # 创建一个单位矩阵 X，其行数与 A 的倒数第二个维度相同，列数为 k
            X = torch.eye(A.size(-2), k, dtype=A.dtype, device=A.device)
            # 如果 A 的维度大于 2，则将 X 扩展为与 A 相同的维度
            if A.dim() > 2:
                X = X.expand(X_shape)

            # 调用 torch.lobpcg 函数进行特征值分解，返回特征值 D 和特征向量 U
            D, U = torch.lobpcg(A=A, k=k, B=B, X=X, largest=largest)

            # 对 U 进行处理，消除由于初始非确定性而导致的特征向量符号不确定性
            _, idx = U.abs().max(-2, keepdim=True)
            sign = U.gather(-2, idx).sign()
            U = U * sign
            return D, U

        # 定义一个内部函数 run_symeig_test，用于运行对称特征值分解的测试
        def run_symeig_test(k, sizes, largest=True):
            # 生成指定大小的随机矩阵 A，类型为双精度
            A = torch.rand(*sizes).double()
            # 构造对称矩阵 A，用于测试目的
            A = (A @ A.mT) / 10
            # 设置 A 需要计算梯度
            A.requires_grad_(True)

            # 对 func 函数进行梯度检查，检查是否支持批处理梯度
            gradcheck(lambda A: func(k, A, largest), A, check_batched_grad=False)

            # 自定义梯度向量以提高稳定性，由于 lobpcg 正向传播中存在非确定性
            D_grad = torch.rand(*A.shape[:-2], k) / 100
            U_grad = torch.rand(*A.shape[:-1], k) / 100
            gradgradcheck(
                lambda A: func(k, A, largest),
                A,
                [D_grad, U_grad],
                atol=1e-4,
                check_batched_grad=False,
            )

            # 检查 A.grad 是否是对称矩阵
            A = A.detach().requires_grad_(True)
            D, U = func(k, A, largest)
            (D.sum() + U.sum()).backward()
            self.assertEqual(A.grad, A.grad.mT)

        # 遍历 largest 参数的两种可能取值 True 和 False，分别运行对称特征值分解的测试
        for largest in [True, False]:
            run_symeig_test(1, (6, 6), largest=largest)
            run_symeig_test(1, (2, 6, 6), largest=largest)
            run_symeig_test(1, (2, 2, 6, 6), largest=largest)
            run_symeig_test(2, (6, 6), largest=largest)
            run_symeig_test(2, (2, 6, 6), largest=largest)
            run_symeig_test(2, (2, 2, 6, 6), largest=largest)
            run_symeig_test(3, (9, 9), largest=largest)
            run_symeig_test(3, (2, 9, 9), largest=largest)
            run_symeig_test(3, (2, 2, 9, 9), largest=largest)
    # 定义测试函数 test_variable_traverse(self)
    def test_variable_traverse(self):
        # 定义内部函数 get_out_and_unrefed_cycle()
        def get_out_and_unrefed_cycle():
            # 创建一个形状为 (10,) 的张量，并设置其 requires_grad 为 True
            inp = torch.randn(10, requires_grad=True)
            # 将 inp 张量重塑为 (10, 1)，赋值给 tmp
            tmp = inp.view(10, 1)
            # 将 tmp 张量再次重塑为 (10)，赋值给 out
            out = tmp.view(10)

            # 创建一个包含中间变量的引用循环
            my_list = []
            # 将 tmp 添加到 my_list 中
            my_list.append(tmp)
            # 将 my_list 自身添加到 my_list 中，形成循环引用
            my_list.append(my_list)

            # 返回 out 张量作为函数结果
            return out

        # 调用 get_out_and_unrefed_cycle() 函数，将结果赋给 out
        out = get_out_and_unrefed_cycle()
        # 执行垃圾回收
        gc.collect()
        # 如果误释放了资源，此处将导致段错误
        # 对 out 张量进行反向传播，梯度为随机生成的与 out 大小相同的张量
        out.backward(torch.randn(out.size()))

    # TODO: review porting these to OpInfo tests
    # 定义测试函数 test_pow_zero_tensor_gradient(self)
    def test_pow_zero_tensor_gradient(self):
        # 定义内部函数 run_test(input_size, exponent)
        def run_test(input_size, exponent):
            # 创建一个全零张量，形状为 input_size，并设置 requires_grad 为 True
            input = torch.zeros(*input_size, requires_grad=True)
            # 对 input 张量进行指数运算，求和后进行反向传播
            input.pow(exponent).sum().backward()
            # 断言 input 张量的梯度的绝对值之和为 0
            self.assertEqual(input.grad.abs().sum(), 0)

        # 分别调用 run_test() 函数进行测试
        run_test((10,), torch.zeros(10))
        run_test((10, 10), torch.zeros(10, 10))
        run_test((10,), 0)

    # 定义测试函数 test_current_graph_task_id(self)
    def test_current_graph_task_id(self):
        # 初始化 id 列表，包含一个元素 -1
        id = [-1]

        # 定义钩子函数 hook(_)
        def hook(_):
            # 将当前图任务的 ID 存入 id 列表的第一个元素中
            id[0] = torch._C._current_graph_task_id()

        # 创建一个张量 t，其值为 1.0，设置 requires_grad 为 True，并克隆该张量
        t = torch.tensor(1.0, requires_grad=True).clone()
        # 注册 hook 钩子函数到张量 t 上
        t.register_hook(hook)

        # 对张量 t 进行反向传播，保留计算图
        t.backward(retain_graph=True)
        # 将 base 初始化为 id 列表中的第一个元素
        base = id[0]
        # 再次对张量 t 进行反向传播，保留计算图
        t.backward(retain_graph=True)
        # 断言当前图任务的 ID 与 base 的差为 1
        self.assertEqual(id[0] - base, 1)
        # 再次对张量 t 进行反向传播，保留计算图
        t.backward(retain_graph=True)
        # 断言当前图任务的 ID 与 base 的差为 2
        self.assertEqual(id[0] - base, 2)

        # 断言当前图任务的 ID 为 -1
        self.assertEqual(torch._C._current_graph_task_id(), -1)

    # 定义测试函数 test_current_graph_task_execution_order(self)
    def test_current_graph_task_execution_order(self):
        # 初始化 predicted 列表，包含一个 None 元素
        predicted = [None]

        # 定义钩子函数 hook(_)
        def hook(_):
            # 将当前图任务的执行顺序存入 predicted 列表的第一个元素中
            predicted[0] = torch._C._current_graph_task_execution_order()

        # 定义函数 names(nodes)
        def names(nodes):
            # 返回 nodes 中各节点的名称
            return ", ".join([node.name().split(" ")[-1] for node in nodes]) + "\n"

        # 定义函数 grad_fns(*tensors)
        def grad_fns(*tensors):
            # 返回 tensors 中各张量的梯度函数
            out = []
            for t in tensors:
                if t.requires_grad and t.grad_fn is None:
                    out.append(t.clone().grad_fn.next_functions[0][0])
                else:
                    out.append(t.grad_fn)
            return out

        # 初始化 actual 列表
        actual = []

        # 定义注册记录钩子函数的函数 register_logging_hooks(*tensors)
        def register_logging_hooks(*tensors):
            # 注册记录调用顺序的钩子函数
            def get_hook(i):
                def hook(t_):
                    actual.append(tensors[i])

                return hook

            for i, t in enumerate(tensors):
                t.register_hook(get_hook(i))

        # 创建一个张量 t，其值为 1.0，设置 requires_grad 为 True，然后依次调用 sin() 和 exp() 方法
        t = torch.tensor(1.0, requires_grad=True).clone().sin().exp()
        # 注册 hook 钩子函数到张量 t 上
        t.register_hook(hook)
        # 禁用多线程环境下的自动求导
        with torch.autograd.set_multithreading_enabled(False):
            # 对张量 t 进行反向传播
            t.backward()
        # 断言 predicted[0] 的名称与预期名称一致
        self.assertExpectedInline(
            names(predicted[0]),
            """\
# 定义了一个函数注册日志钩子
register_logging_hooks(a, b, c, d, out)

# 注册了一个钩子函数，用于处理反向传播时的梯度
out.register_hook(hook)

# 设置 Torch 的多线程开关为关闭状态，以便在单线程下执行反向传播
with torch.autograd.set_multithreading_enabled(False):
    # 执行反向传播
    out.backward()

# 断言预期的梯度与实际计算的梯度是否相等
self.assertEqual(predicted[0], grad_fns(*actual))
actual = []

# 定义了一个函数注册日志钩子
register_logging_hooks(a, b, c, out)

# 注册了一个钩子函数，用于处理反向传播时的梯度
out.register_hook(hook)

# 设置 Torch 的多线程开关为关闭状态，以便在单线程下执行反向传播
with torch.autograd.set_multithreading_enabled(False):
    # 执行反向传播
    out.backward()

# 断言预期的梯度与实际计算的梯度是否相等
self.assertEqual(predicted[0], grad_fns(*actual))
actual = []

# 定义了一个函数注册日志钩子
register_logging_hooks(a, b, out)

# 注册了一个钩子函数，用于处理反向传播时的梯度
out.register_hook(hook)

# 设置 Torch 的多线程开关为关闭状态，以便在单线程下执行反向传播
with torch.autograd.set_multithreading_enabled(False):
    # 执行反向传播，传入多个输出节点作为参数
    torch.autograd.grad((out, out3, out2), inputs=(a,))

# 断言预期的梯度名称与实际计算的梯度名称是否相等
self.assertExpectedInline(
    names(predicted[0]),
    """\
CosBackward0, CosBackward0, SinBackward0, MulBackward0, torch::autograd::AccumulateGrad
""",
)

# 注释掉的断言语句，待钩子行为更新后取消注释
# self.assertEqual(predicted[0], grad_fns(*actual))
actual = []

# 定义了一个函数注册日志钩子
register_logging_hooks(a, b, out)

# 注册了一个钩子函数，用于处理反向传播时的梯度
out.register_hook(hook)

# 设置 Torch 的多线程开关为关闭状态，以便在单线程下执行反向传播
with torch.autograd.set_multithreading_enabled(False):
    # 执行反向传播
    out.backward()

# 断言预期的梯度与实际计算的梯度是否相等
self.assertEqual(predicted[0], grad_fns(*actual))
actual = []

# 设置 Torch 的多线程开关为关闭状态，以便在单线程下执行反向传播
with torch.autograd.set_multithreading_enabled(False):
    # 执行反向传播，传入多个输入节点作为参数
    torch.autograd.grad(
        (out,),
        inputs=(
            a,
            b,
        ),
    )

# 断言预期的梯度名称与实际计算的梯度名称是否相等
self.assertEqual(
    names(predicted[0]),
    """\
SinBackward0, MulBackward0, torch::autograd::AccumulateGrad
""",
)
        # TODO: 更新钩子行为后取消注释
        # self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # `inputs` 指定子图的情况
        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(1.0, requires_grad=True)
        c = a * b
        out = c.sin()
        # 注册日志记录钩子函数
        register_logging_hooks(a, b, c, out)
        out.register_hook(hook)
        # 设置多线程禁用，并计算梯度
        with torch.autograd.set_multithreading_enabled(False):
            torch.autograd.grad((out,), inputs=(a,))
        self.assertEqual(
            names(predicted[0]),
            """\
SinBackward0, MulBackward0, torch::autograd::AccumulateGrad
""",
        )
        # TODO: 更新钩子行为后取消注释
        # self.assertEqual(predicted[0], grad_fns(*actual))
        actual = []

        # 非反向传播时出错
        with self.assertRaisesRegex(
            RuntimeError, "should only be called during the backward pass"
        ):
            torch._C._current_graph_task_execution_order()

        # 上下文管理器未启用时出错
        t = torch.tensor(1.0, requires_grad=True).clone().sin().exp()
        t.register_hook(hook)
        with self.assertRaisesRegex(
            RuntimeError,
            "expects the current backward to be executed with multithreading disabled",
        ):
            t.backward()
    # 定义测试函数 test_view_replay_enabled
    def test_view_replay_enabled(self):
        # 定义内部函数 f，接受参数 x，返回一个视图变换后的张量
        def f(x):
            # 克隆张量 x 并将其视图变换为一维
            out = x.clone().view(-1)
            # 修改视图，触发自动求导的视图重播逻辑
            out.add_(1)
            return out

        # 创建一个全为 1 的二维张量 x，并标记为需要梯度计算
        x = torch.ones(2, 2, requires_grad=True)

        # 在关闭视图跟踪的上下文管理器中进行测试
        with torch.autograd._force_original_view_tracking(False):
            out = f(x)
            # 断言是否存在 AsStridedBackward 在梯度函数中，表明未启用视图重播
            self.assertTrue("AsStridedBackward" in str(out.grad_fn))
            # 断言当前未启用视图重播
            self.assertFalse(torch.autograd.is_view_replay_enabled())
        # 再次断言当前未启用视图重播
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        # 在开启视图跟踪的上下文管理器中进行测试
        with torch.autograd._force_original_view_tracking(True):
            out = f(x)
            # 断言是否存在 ViewBackward 在梯度函数中，表明启用了视图重播
            self.assertTrue("ViewBackward" in str(out.grad_fn))
            # 断言当前已启用视图重播
            self.assertTrue(torch.autograd.is_view_replay_enabled())
        # 再次进行一次测试，确保在非启用视图重播的上下文管理器中状态仍然是未启用
        out = f(x)
        self.assertTrue("AsStridedBackward" in str(out.grad_fn))
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        # 在关闭视图跟踪的上下文管理器中进行测试，通过直接设置来切换状态
        with torch.autograd._force_original_view_tracking(False):
            torch.autograd._force_original_view_tracking(True)
            out = f(x)
            # 断言是否存在 ViewBackward 在梯度函数中，表明已启用视图重播
            self.assertTrue("ViewBackward" in str(out.grad_fn))
            # 断言当前已启用视图重播
            self.assertTrue(torch.autograd.is_view_replay_enabled())
        # 再次断言当前未启用视图重播
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        # 作为函数形式进行测试，先关闭视图跟踪
        torch.autograd._force_original_view_tracking(False)
        out = f(x)
        self.assertTrue("AsStridedBackward" in str(out.grad_fn))
        self.assertFalse(torch.autograd.is_view_replay_enabled())

        # 再开启视图跟踪
        torch.autograd._force_original_view_tracking(True)
        out = f(x)
        self.assertTrue("ViewBackward" in str(out.grad_fn))
        self.assertTrue(torch.autograd.is_view_replay_enabled())

    # 定义测试函数 test_unsafe_set_version_counter
    def test_unsafe_set_version_counter(self):
        # 创建一个全为 1 的一维张量 x，并克隆
        x = torch.ones(2, requires_grad=True).clone()
        # 修改张量值
        x.add_(1)
        x.add_(2)
        # 断言张量版本号为 2
        self.assertEqual(2, x._version)

        # 在不安全的版本计数器上下文管理器中进行操作
        with torch.autograd._unsafe_preserve_version_counter(x):
            x.mul_(2)
            x.mul_(3)
        # 断言在上下文管理器中版本计数器未改变
        self.assertEqual(2, x._version)

        # 直接设置版本计数器为 0
        torch._C._autograd._unsafe_set_version_counter(x, 0)
        # 断言版本计数器被设置为 0
        self.assertEqual(0, x._version)

        # 使用断言捕获 RuntimeError 异常，尝试设置版本计数器为 -1
        with self.assertRaisesRegex(RuntimeError, "Cannot set"):
            torch._C._autograd._unsafe_set_version_counter(x, -1)
    def test_current_node(self):
        pr = []  # 创建一个空列表 pr，用于存储输出信息

        class MyMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args, kwargs=None):
                # 获取当前的自动求导节点（Autograd Node）
                node = torch._C._current_autograd_node()
                # 在 Windows 上不稳定，因此使用节点的类名作为名称
                node_name = node.__class__.__name__ if node else "None"
                # 将函数名称、节点名称和执行信息添加到 pr 列表中
                pr.append(f"Running {func} from within {node_name}")
                # 调用原始函数并返回结果
                return func(*args, **(kwargs or {}))

        with MyMode():  # 使用定义的 MyMode 上下文管理器
            pr.append("FW")  # 添加 "FW" 到 pr 列表中，表示前向传播阶段
            a = torch.rand(10, requires_grad=True)  # 创建一个需要梯度的张量 a
            b = a.mul(2).div(3).sum()  # 对张量 a 进行乘法、除法和求和操作，得到张量 b
            pr.append("BW")  # 添加 "BW" 到 pr 列表中，表示反向传播阶段
            b.backward()  # 对张量 b 进行反向传播
            pr.append("Done")  # 添加 "Done" 到 pr 列表中，表示操作完成

        self.assertExpectedInline(
            "\n".join(pr),
            """\
# FW 表示前向传播，表示模型的正向计算过程
# Running aten.rand.default from within None 表示在 None 上运行 aten.rand.default 操作
# Running aten.mul.Tensor from within None 表示在 None 上运行 aten.mul.Tensor 操作
# Running aten.div.Tensor from within None 表示在 None 上运行 aten.div.Tensor 操作
# Running aten.sum.default from within None 表示在 None 上运行 aten.sum.default 操作
# BW 表示反向传播，表示模型的反向计算过程
# Running aten.ones_like.default from within None 表示在 None 上运行 aten.ones_like.default 操作
# Running aten.expand.default from within SumBackward0 表示在 SumBackward0 上运行 aten.expand.default 操作
# Running aten.div.Tensor from within DivBackward0 表示在 DivBackward0 上运行 aten.div.Tensor 操作
# Running aten.mul.Tensor from within MulBackward0 表示在 MulBackward0 上运行 aten.mul.Tensor 操作
# Running aten.detach.default from within AccumulateGrad 表示在 AccumulateGrad 上运行 aten.detach.default 操作
# Running aten.detach.default from within AccumulateGrad 表示在 AccumulateGrad 上运行 aten.detach.default 操作
# Done 表示操作完成

class TestProfiler(TestCase):
    # 测试性能分析器
    def test_profiler(self):
        # 生成一个 10x10 的随机张量
        x = torch.randn(10, 10)

        # 使用性能分析器 p 记录代码块的性能
        with profile(use_kineto=kineto_available()) as p:
            # 断言自动求导性能分析器已启用
            self.assertTrue(torch.autograd._profiler_enabled())
            # 对 x 进行乘法和加法操作
            y = x * 2 + 4

        # 断言自动求导性能分析器已关闭
        self.assertFalse(torch.autograd._profiler_enabled())

        # 定义需要查找的操作名称
        names = ["aten::mul", "aten::add"]
        found_indices = set()
        # 遍历性能分析器记录的函数事件
        for evt in p.function_events:
            # 如果事件名称在定义的操作名称中，则添加到集合中
            if evt.name in names:
                found_indices.add(names.index(evt.name))
        # 断言找到的操作名称数量与定义的操作名称数量相等
        self.assertEqual(len(found_indices), len(names))
    # 定义测试函数 test_profiler_seq_nr，用于测试性能分析器中的序列号功能
    def test_profiler_seq_nr(self):
        # 使用性能分析器并启用 Kineto（如果可用）
        with profile(use_kineto=kineto_available()) as p:
            # 创建两个张量 x 和 y，形状为 10x10，并且要求计算梯度
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            # 计算两个张量的和
            z = x + y
            # 沿着所有维度求和，得到一个标量张量 s
            s = z.sum(dim=None)
            # 对标量张量 s 执行反向传播
            s.backward()
        # 打印性能分析器中的关键指标表格，按照 CPU 时间排序
        print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
        
        # 预期 aten::add 和 aten::sum 操作有序列号，
        # 预期对应的反向操作也具有相同的序列号
        autograd_ops = {
            ("aten::add", "Add"): [],   # 存储前向和后向操作对应的事件
            ("aten::sum", "Sum"): [],   # 存储前向和后向操作对应的事件
        }
        accumulate_ops = []  # 存储累积梯度操作的事件
        found_empty = False  # 标记是否发现空操作

        # 遍历性能分析器中的事件列表
        for e in p.function_events:
            # 将符合前向和后向操作名称条件的事件加入对应的 autograd_ops 中
            for (fwd_name, bwd_name), ops in autograd_ops.items():
                if e.name == fwd_name or (bwd_name in e.name and "Backward" in e.name):
                    ops.append(e)

            # 如果事件名称中包含 "AccumulateGrad"，加入 accumulate_ops
            if "AccumulateGrad" in e.name:
                accumulate_ops.append(e)

            # 检查嵌套操作（例如空操作）是否没有序列号
            if e.name == "aten::empty":
                self.assertEqual(e.sequence_nr, -1)  # 确保序列号为 -1
                found_empty = True

        # 对 autograd_ops 中的每一对进行验证
        for idx, ((fwd_name, bwd_name), ops) in enumerate(autograd_ops.items()):
            # 每对应该有三个事件
            self.assertEqual(len(ops), 3)
            # 第一个事件的名称应为前向操作名称
            self.assertEqual(ops[0].name, fwd_name)
            # 第二个事件的名称应为反向操作名称的评估函数
            self.assertEqual(
                ops[1].name,
                f"autograd::engine::evaluate_function: {bwd_name}Backward{idx}",
            )
            # 第三个事件的名称应为反向操作名称
            self.assertEqual(ops[2].name, f"{bwd_name}Backward{idx}")
            # 确保每个事件的序列号大于等于 0
            self.assertGreaterEqual(ops[0].sequence_nr, 0)
            # 确保第二个事件的序列号与第一个相同
            self.assertEqual(ops[1].sequence_nr, ops[0].sequence_nr)
            # 确保第三个事件的序列号与第一个相同
            self.assertEqual(ops[2].sequence_nr, ops[0].sequence_nr)
            # 确保每个事件的前向线程为 0
            self.assertEqual(ops[0].fwd_thread, 0)
            # 确保第二个事件的前向线程与第一个事件的线程相同
            self.assertEqual(ops[1].fwd_thread, ops[0].thread)
            # 确保第三个事件的前向线程与第一个事件的线程相同
            self.assertEqual(ops[2].fwd_thread, ops[0].thread)
        
        # 断言已找到空操作
        self.assertTrue(found_empty)

    # 定义测试函数 test_profiler_unboxed_only，仅测试未装箱的操作
    def test_profiler_unboxed_only(self):
        # 创建一个形状为 [3, 4] 的张量 x
        x = torch.rand(3, 4)

        # 使用自动求导分析器，并启用 Kineto（如果可用）
        with torch.autograd.profiler.profile(use_kineto=kineto_available()) as prof:
            # 将张量 x 重置为 [3, 2]
            x.resize_([3, 2])
    # 定义一个测试函数，用于测试分析器的传播功能
    def test_profiler_propagation(self):
        # 定义内部函数 foo，接受参数 x
        def foo(x):
            # 使用 record_function 在函数内部创建记录点 "in_foo"，返回 x 的两倍
            with record_function("in_foo") as rf:
                return x * 2

        # 创建一个随机张量 x
        x = torch.rand(3, 4)
        # 使用 torch.jit.trace 对 foo 函数进行跟踪
        traced_foo = torch.jit.trace(foo, x)

        # 定义内部函数 bar，接受参数 x
        def bar(x):
            # 使用 record_function 在函数内部创建记录点 "in_bar"
            with record_function("in_bar") as rf:
                # 在 fork 中期待分析器能够传播
                fut = torch.jit._fork(traced_foo, x)
                y = torch.jit._wait(fut)
                # 注意: continuation (和 rf 的结束) 可能在不同的线程中执行
                with record_function("in_bar_after_wait") as rf2:
                    y = y * 2
                return y

        # 使用 torch.jit.trace 对 bar 函数进行跟踪
        traced_bar = torch.jit.trace(bar, x)

        # 使用 profile 函数开启性能分析，根据 kineto 的可用性决定是否使用
        with profile(use_kineto=kineto_available()) as p:
            # 执行跟踪过的 bar 函数，传入参数 x
            traced_bar(x)

        # 初始化三个布尔值变量，用于检查各记录点是否被找到
        found_foo = False
        found_bar = False
        found_bar_after_wait = False

        # 遍历分析器中的函数事件列表
        for info in p.function_events:
            # 如果事件名为 "in_foo"
            if info.name == "in_foo":
                # 断言 found_foo 为 False
                self.assertFalse(found_foo)
                # 将 found_foo 置为 True
                found_foo = True
            # 如果事件名为 "in_bar"
            elif info.name == "in_bar":
                # 断言 found_bar 为 False
                self.assertFalse(found_bar)
                # 将 found_bar 置为 True
                found_bar = True
            # 如果事件名为 "in_bar_after_wait"
            elif info.name == "in_bar_after_wait":
                # 断言 found_bar_after_wait 为 False
                self.assertFalse(found_bar_after_wait)
                # 将 found_bar_after_wait 置为 True
                found_bar_after_wait = True

        # 断言 found_foo、found_bar、found_bar_after_wait 均为 True
        self.assertTrue(found_foo)
        self.assertTrue(found_bar)
        self.assertTrue(found_bar_after_wait)

    # 定义一个测试函数，用于测试 record_function 的回调功能
    def test_record_function_callbacks(self):
        # 创建一个 10x10 的随机张量 x
        x = torch.randn(10, 10)
        # 使用 profile 函数开启性能分析，根据 kineto 的可用性决定是否使用
        with profile(use_kineto=kineto_available()) as p:
            # 使用 record_function 创建记录点 "foo"
            with record_function("foo"):
                y = x * 2 + 4

        # 获取函数事件列表
        function_events = p.function_events
        # 查找名字中包含 "foo" 的事件
        foo_event = next(event for event in function_events if "foo" in event.name)
        # 断言 "foo" 的事件发生次数为 1
        self.assertEqual(foo_event.count, 1)

    # 定义一个测试函数，用于测试 record_function 的旧版用法
    def test_record_function_legacy(self):
        # 测试新的 _record_function 操作是否起作用
        # 注意: 一旦 record_function 直接使用这些操作，可以删除此部分
        # 创建一个 10x10 的随机张量 x
        x = torch.randn(10, 10)
        # 使用 profile 函数开启性能分析，根据 kineto 的可用性决定是否使用
        with profile(use_kineto=kineto_available()) as p:
            # 使用 torch.ops.profiler._record_function_enter 进入记录点 "bar"，无参数
            handle = torch.ops.profiler._record_function_enter("bar", None)
            try:
                y = x * 2 + 4
            finally:
                # 使用 torch.ops.profiler._record_function_exit 退出记录点
                torch.ops.profiler._record_function_exit(handle)

        # 获取函数事件列表
        function_events = p.function_events
        # 查找名字中包含 "bar" 的事件
        foo_event = next(event for event in function_events if "bar" in event.name)
        # 断言 "bar" 的事件发生次数为 1
        self.assertEqual(foo_event.count, 1)
    def test_profiler_aggregation_fake(self):
        events = EventList()
        id = [0]

        def get_id():
            id[0] = id[0] + 1
            return id[0]

        # [[thread_id, [(start, end, id), ....]], ...]
        # 使用列表而不是字典，以确保在任何 Python 版本中都保证顺序
        threads = [
            [1, [(0, 1, get_id()), (1, 2, get_id())]],
            [0, [(0, 2, get_id()), (1, 2, get_id()), (1, 3, get_id())]],
        ]
        for thread, ranges in threads:
            for range in ranges:
                assert len(range) == 3
                # 将事件添加到事件列表中
                events.append(
                    FunctionEvent(
                        id=range[2],
                        node_id=0,
                        name="",
                        thread=thread,
                        start_us=range[0],
                        end_us=range[1],
                    )
                )

        # 填充 CPU 子事件
        events._populate_cpu_children()

        # [1, 3] 推出 [0, 2]，然后将 [1, 2] 记录为 [1, 3] 的子事件
        res = [[], [], [], [], [4]]

        def get_children_ids(event):
            return [child.id for child in event.cpu_children]

        # 断言事件列表中每个事件的 CPU 子事件 ID 序列与预期结果 res 相同
        assert [get_children_ids(event) for event in events] == res

    def test_profiler_aggregation_table(self):
        """
        Test if the profiling result is aggregated for `str(prof)`

        See: https://github.com/pytorch/pytorch/issues/37500
        """
        
        x = torch.randn(1024)
        # 使用 kineto，生成使用 prof 表示的性能分析结果
        with torch.autograd.profiler.profile(use_kineto=kineto_available()) as prof:
            torch.einsum("i->", x)

        # 将性能分析结果转换为字符串
        prof_str = str(prof)
        # 获取性能分析结果的表格表示
        prof_table = prof.table()

        # 断言性能分析结果的表格表示与字符串表示相等
        self.assertEqual(prof_table, prof_str)

    def test_profiler_function_event_avg(self):
        avg = FunctionEventAvg()
        avg.add(
            FunctionEvent(id=0, node_id=0, name="foo", thread=0, start_us=10, end_us=15)
        )
        avg.add(
            FunctionEvent(id=1, node_id=0, name="foo", thread=0, start_us=20, end_us=30)
        )
        avg.add(avg)  # 将自身添加到平均值计算中

        # 断言平均值计算的关键字为 "foo"
        self.assertEqual(avg.key, "foo")

        # 断言聚合统计信息
        self.assertEqual(avg.count, 4)
        self.assertEqual(avg.cpu_time_total, 30)
        self.assertEqual(avg.self_cpu_time_total, 30)
        self.assertEqual(avg.device_time_total, 0)

        # 断言平均统计信息
        self.assertEqual(avg.cpu_time, 7.5)
        self.assertEqual(avg.device_time_total, 0)
    # 定义测试方法，用于测试神经网络层的形状分析
    def test_profiler_shapes(self):
        # 打印空行
        print("")
        # 创建输入维度为 (128, 20) 的线性层
        layer1 = torch.nn.Linear(20, 30)
        # 创建输入维度为 (128, 30) 的线性层
        layer2 = torch.nn.Linear(30, 40)
        # 生成一个大小为 (128, 20) 的随机输入张量
        input = torch.randn(128, 20)
        # 使用 profiler 进行性能分析，记录操作形状并根据 kineto 的可用性使用
        with profile(record_shapes=True, use_kineto=kineto_available()) as prof:
            # 对输入进行一次层次化处理
            layer2(layer1(input))

        # 打印分析结果中的函数事件
        print(prof.function_events)

        # 期望的线性层输入形状列表
        linear_expected_shapes = [
            [[128, 20], [30, 20], [30]],
            [[128, 30], [40, 30], [40]],
        ]

        # 用于存储找到的事件索引集合
        found_indices = set()
        # 遍历分析结果中的函数事件
        for event in prof.function_events:
            # 如果事件名称为 "aten::linear"
            if event.name == "aten::linear":
                # 断言事件的输入形状在期望的线性层输入形状列表中
                self.assertTrue(event.input_shapes in linear_expected_shapes)
                # 将符合条件的输入形状索引加入到集合中
                found_indices.add(linear_expected_shapes.index(event.input_shapes))
        # 断言找到的事件索引数量与期望的线性层输入形状列表长度相等
        self.assertEqual(len(found_indices), len(linear_expected_shapes))

    # 定义测试方法，用于测试 LSTM 的聚合性能分析
    def test_profiler_aggregation_lstm(self):
        # 打印空行
        print("")
        # 创建一个具有输入大小 (5, 3, 10)，隐藏状态大小 (2, 3, 20) 和记忆单元大小 (2, 3, 20) 的 LSTM
        rnn = torch.nn.LSTM(10, 20, 2)
        # 初始化总时间统计变量
        total_time_s = 0
        # 使用 profiler 进行性能分析，记录操作形状并根据 kineto 的可用性使用
        with profile(record_shapes=True, use_kineto=kineto_available()) as prof:
            # 循环 20 次
            for i in range(20):
                # 创建大小为 (5, 3, 10) 的随机输入张量
                input = torch.randn(5, 3, 10)
                # 创建大小为 (2, 3, 20) 的随机隐藏状态张量
                h = torch.randn(2, 3, 20)
                # 创建大小为 (2, 3, 20) 的随机记忆单元张量
                c = torch.randn(2, 3, 20)
                # 记录开始时间
                start = time.time()
                # 对输入和状态进行 LSTM 处理
                rnn(input, (h, c))
                # 记录结束时间
                end = time.time()
                # 累加每次处理的时间到总时间中
                total_time_s += end - start

        # 打印分析结果中排序后的前 10 条自我 CPU 时间总和最大的事件
        print(prof.table(sort_by="self_cpu_time_total", row_limit=10, header="TEST"))
        # 打印按输入形状分组的键平均统计表，排序方式为自我 CPU 时间总和，打印前 10 条记录
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total", row_limit=10
            )
        )
        # 打印排序后的前 10 条自我 CPU 时间总和最大的事件，顶层事件限制为 True
        print(
            prof.table(
                sort_by="self_cpu_time_total",
                row_limit=10,
                max_src_column_width=300,
                header="TEST",
                top_level_events_only=True,
            )
        )
        # 打印按输入形状分组的键平均统计表，排序方式为自我 CPU 时间总和，顶层事件限制为 True，打印前 10 条记录
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total", row_limit=10, top_level_events_only=True
            )
        )

        # 将总时间转换为微秒，符合分析器的默认单位
        total_time_us = (
            total_time_s * 1000.0 * 1000.0
        )  # make it us which is profiler default
        # 打印基于 Python 测量的总时间
        print("Total time based on python measurements: ", _format_time(total_time_us))
        # 打印 CPU 时间测量 Python 端开销的百分比
        print(
            f"CPU time measurement python side overhead: {(total_time_us / prof.self_cpu_time_total - 1.0) * 100.0:.2f}%"
        )

        # 如果系统平台不是 "win32"
        if sys.platform != "win32":
            # 使用临时命名文件导出 Chrome 跟踪文件
            with tempfile.NamedTemporaryFile() as trace_file:
                prof.export_chrome_trace(trace_file.name)
    # 定义一个测试函数，测试记录函数调用和性能分析的功能
    def test_record_function(self):
        # 创建一个 10x10 的随机张量 x
        x = torch.randn(10, 10)

        # 定义一个内部函数 forward，接收参数 x
        def forward(x):
            # 开始记录名为 "outer" 的函数
            with record_function("outer"):
                # 对输入张量 x 执行乘以 2 加 4的操作
                y = x * 2 + 4
                # 开始记录名为 "inner" 的函数
                with record_function("inner"):
                    # 对结果张量 y 执行减去 1 的操作
                    y = y - 1
            # 对 y 执行除以 1 的操作（实际上不会改变 y 的值）
            y = y / 1

        # 调用 forward 函数，传入张量 x
        forward(x)

        # 使用 kineto 可用时，使用性能分析器进行性能分析
        with profile(use_kineto=kineto_available()) as p:
            forward(x)

        # 获取分析器中的函数事件
        events = p.function_events
        # 定义关键的事件名称列表
        important_events = [
            "outer",
            "aten::mul",
            "aten::add",
            "inner",
            "aten::sub",
            "aten::div",
        ]
        # 初始化索引
        idx = 0
        # 遍历事件列表
        for info in events:
            # 如果当前事件的名称与当前重要事件名称匹配
            if info.name == important_events[idx]:
                # 将索引增加 1
                idx = idx + 1
            # 如果索引等于重要事件列表的长度，跳出循环
            if idx == len(important_events):
                break
        # 断言索引是否等于重要事件列表的长度
        self.assertEqual(idx, len(important_events))

        # 使用 record_function 装饰任意函数，这里装饰函数 f，记录其为 "my_func"
        @record_function("my_func")
        def f(x, y):
            return x + y

        # 使用性能分析器进行性能分析
        with profile(use_kineto=kineto_available()) as p:
            f(1, 2)

        # 断言分析结果中是否包含 "my_func" 的记录
        self.assertTrue("my_func" in str(p))

    # 测试多线程环境下的 record_function 函数
    def test_record_function_multithreaded(self):
        # 记录名为 "outer" 的函数进入
        rf = record_function("outer")
        rf.__enter__()
        # 开始记录名为 "inner" 的函数
        with record_function("inner"):
            # 测试在启动另一个记录函数后退出记录函数不会抛出异常
            rf.__exit__(None, None, None)

        # 开始记录名为 "inner" 的函数
        with record_function("inner"):
            rf.__enter__()
        # 测试在结束另一个记录函数后退出记录函数不会抛出异常
        rf.__exit__(None, None, None)

    # 测试 torch 张量的 dir() 方法
    def test_dir(self):
        # 创建一个 10x10 的随机张量 x
        x = torch.randn(10, 10)
        # 获得张量 x 的所有属性名称
        keys = dir(x)
        # 断言 "shape" 是否在属性名称列表中
        self.assertIn("shape", keys)

        # 创建一个复数类型的张量 y
        y = torch.randn(10, 10, dtype=torch.cfloat)
        # 定义属性名称为 "imag"
        imag_key = "imag"
        # 断言对于实数类型的张量 x，不能调用属性 "imag"，会抛出 RuntimeError
        self.assertRaises(RuntimeError, lambda: hasattr(x, imag_key))
        # 断言对于复数类型的张量 y，可以调用属性 "imag"
        self.assertTrue(hasattr(y, imag_key))
        # 从属性名称列表中移除 "imag"
        keys.remove(imag_key)

        # 遍历剩余的属性名称
        for key in keys:
            # 断言张量 x 具有每个属性名称
            self.assertTrue(hasattr(x, key))

    # 测试在视图上进行就地操作并保存输出
    def test_inplace_on_view_saved_output(self):
        # 定义一个测试函数 test
        def test():
            # 创建一个大小为 3x3 的张量 root，并标记需要梯度计算
            root = torch.randn(3, 3, requires_grad=True)
            # 克隆张量 root
            copy = root.clone()
            # 注册 IncrementOnDelete 类的钩子函数到 copy 的梯度函数
            copy.grad_fn.register_hook(IncrementOnDelete())
            # 创建视图 view，将 copy 变成一维张量
            view = copy.view(9)
            # 在 view 上执行就地操作 torch.nn.functional.relu，设置 inplace=True
            torch.nn.functional.relu(view, inplace=True)

        # 调用测试函数 test
        test()
        # 断言 IncrementOnDelete 类的实例 dealloc[0] 是否为 1，表示其被正确释放
        self.assertEqual(dealloc[0], 1)
    # 测试在视图上进行原地操作时的错误处理
    def test_inplace_on_view_leaf_errors(self):
        # 创建一个具有梯度的零张量
        x = torch.zeros(1, requires_grad=True)
        # 使用 x 的视图 y
        y = x.view_as(x)
        # 断言期望捕获 RuntimeError，指出视图操作中的原地操作问题
        with self.assertRaisesRegex(
            RuntimeError,
            "a view of a leaf Variable that "
            "requires grad is being used in "
            "an in-place operation.",
        ):
            y.add_(1)

    # 测试视图上的原地操作在反向传播中的行为
    def test_inplace_on_view_backward(self):
        # 创建一个包含实例归一化和 ReLU 激活函数的神经网络
        net = nn.Sequential(nn.InstanceNorm2d(2), nn.ReLU(True))

        # 创建一个具有梯度的张量 x
        x = torch.tensor([[[[1.0, 1.0]]]], requires_grad=True)
        # 计算 net(x).pow(2) 的梯度，创建计算图
        (g,) = torch.autograd.grad(
            net(x).pow(2), [x], grad_outputs=x.new_ones(x.shape), create_graph=True
        )
        torch.autograd.grad(g.sum(), [x])
        self.assertEqual(x, torch.tensor([[[[1.0, 1.0]]]]))

        # 创建一个具有梯度的输入张量
        inputs = torch.ones((1, 3, 256, 256), requires_grad=True)

        # 对输入张量进行操作并视图化
        tmp1 = (inputs + 1).view_as(inputs)
        tmp2 = torch.nn.functional.threshold(tmp1, 0.0, 0.0, True)
        prob_interpolated = torch.sigmoid(tmp2)

        # 计算 prob_interpolated 对 inputs 的梯度，保留计算图
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=inputs,
            grad_outputs=torch.ones(prob_interpolated.size()),
            create_graph=True,
            retain_graph=True,
        )[0]

        # 计算梯度惩罚并执行反向传播
        gradient_penalty = gradients.sum()
        gradient_penalty.backward()

        # 获取梯度函数并验证其名称
        fn = gradient_penalty.grad_fn.next_functions[0][0].next_functions[1][0]
        self.assertEqual(fn.name(), "ThresholdBackwardBackward0")

    # 测试视图上的原地操作对弱梯度函数的影响
    def test_inplace_on_view_weak_grad_fn(self):
        # 创建一个具有梯度的张量 a
        a = torch.arange(10.0, requires_grad=True)

        # 创建张量 b，使用 narrow 和 clone 方法，并视图化
        b = a.narrow(0, 0, 2).clone().view(-1)
        b.relu_()

        # 克隆 b 并删除原始张量，进行垃圾回收
        c = b.clone()
        del b
        gc.collect()

        # 计算张量 c 的总和并执行反向传播
        s = c.sum()
        s.backward()
        self.assertEqual(s, torch.tensor(1.0))

        # 创建一个具有梯度的张量 a，并尝试在其上执行 relu_ 原地操作
        a = torch.rand(10, requires_grad=True).narrow(0, 0, 10)
        with self.assertRaises(RuntimeError):
            b = a.relu_()
    # 定义一个测试方法，用于测试在需要梯度的输入下是否抛出异常
    def test_out_variant_raises_when_inputs_require_grad(self):
        # 创建两个形状为 (2, 2) 的随机张量，并指定需要梯度
        a = torch.randn(2, 2, requires_grad=True)
        b = torch.randn(2, 2, requires_grad=True)
        # 创建一个与 a 形状相同的零张量 x
        x = torch.zeros_like(a)

        # out=... 的函数当前不支持自动求导
        self.assertRaisesRegex(RuntimeError, "out=", lambda: torch.mul(a, b, out=x))

        # 如果处于 no_grad() 模式下，输入可以需要梯度
        with torch.no_grad():
            # 执行 torch.mul 操作，使用输出张量 x
            torch.mul(a, b, out=x)
            # 断言 x 和 a*b 相等
            self.assertEqual(x, a * b)

        # 重新生成随机张量 a 和 b，但这次不需要梯度
        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        # 创建一个形状为 (2, 2) 的零张量 x，并指定需要梯度
        x = torch.zeros(2, 2, requires_grad=True)
        # 如果输出需要梯度，应该抛出异常
        self.assertRaisesRegex(RuntimeError, "out=", lambda: torch.mul(a, b, out=x))

    # 定义一个测试方法，用于检测 NaN 异常
    def test_anomaly_detect_nan(self):
        size = 10

        # 定义一个自定义的 Torch 函数 MyFunc
        class MyFunc(Function):
            @staticmethod
            def forward(ctx, inp1, inp2, fail_0th):
                # 在上下文中存储是否故意使第 0 个输出失败的标志
                ctx.fail_0th = fail_0th
                # 返回 inp1 按第 0 维求和后的结果，保持维度
                return inp1.sum(0, keepdim=True)

            @staticmethod
            def backward(ctx, gO):
                # 克隆梯度输出 gO，并扩展到指定 size
                gI = gO.clone().expand(size)
                # 人为生成一个 NaN
                gI[0] = 0
                gI[0] /= 0  # Generate a nan
                if ctx.fail_0th:
                    return gI, None, None
                else:
                    return None, gI, None

        # 创建一个大小为 size 的随机张量 inp，并指定需要梯度
        inp = torch.rand(size, requires_grad=True)
        # 应用 MyFunc 函数，传入 inp 两次和 True，得到输出 out
        out = MyFunc.apply(inp, inp, True)
        # 对 out 进行反向传播，不应该失败
        out.backward()

        # 再次生成随机张量 inp，并指定需要梯度
        inp = torch.rand(size, requires_grad=True)
        # 应用 MyFunc 函数，传入 inp 两次和 True，得到输出 out
        out = MyFunc.apply(inp, inp, True)
        # 使用断言检测是否捕获到了返回 NaN 值的异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Function 'MyFuncBackward' returned nan values in its 0th output.",
        ):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    out.backward()
            # 断言警告消息中包含 "No forward pass information"
            self.assertIn("No forward pass information", str(w[0].message))

        # 再次生成随机张量 inp，并指定需要梯度
        inp = torch.rand(size, requires_grad=True)
        # 使用断言检测是否捕获到了返回 NaN 值的异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Function 'MyFuncBackward' returned nan values in its 1th output.",
        ):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    # 应用 MyFunc 函数，传入 inp 两次和 False，得到输出 out
                    out = MyFunc.apply(inp, inp, False)
                    # 对 out 进行反向传播
                    out.backward()
            # 断言警告消息中包含 "MyFunc.apply"
            self.assertIn("MyFunc.apply", str(w[0].message))
    # 定义一个测试方法，用于验证 _calculate_shape 函数的功能
    def test_calculate_shape_util(self):
        # 创建一个形状为 (10, 5) 的随机张量 out，并设置其需要梯度计算
        out = torch.randn(10, 5, requires_grad=True)
        # 创建一个形状为 (5, 10) 的随机张量 grad，并设置其需要梯度计算
        grad = torch.randn(5, 10, requires_grad=True)
        # 调用 _calculate_shape 函数计算 out 和 grad 的形状，不需要考虑边界情况
        out_shape, grad_shape = _calculate_shape(out, grad, False)

        # 断言 out_shape 应为 torch.Size([10, 5])
        assert out_shape == torch.Size([10, 5])
        # 断言 grad_shape 应为 torch.Size([5, 10])
        assert grad_shape == torch.Size([5, 10])

        # 使用 nested.as_nested_tensor 方法创建一个嵌套张量 out，包含三个形状为 (10, 5) 的随机张量，并设置需要梯度计算
        out = torch.nested.as_nested_tensor(
            [
                torch.randn(10, 5, requires_grad=True),
                torch.randn(10, 5, requires_grad=True),
                torch.randn(10, 5, requires_grad=True),
            ]
        )
        # 使用 nested.as_nested_tensor 方法创建一个嵌套张量 grad，包含两个形状为 (5, 10) 的随机张量，并设置需要梯度计算
        grad = torch.nested.as_nested_tensor(
            [
                torch.randn(5, 10, requires_grad=True),
                torch.randn(5, 10, requires_grad=True),
            ]
        )
        # 调用 _calculate_shape 函数计算嵌套张量 out 和 grad 的形状，不需要考虑边界情况
        out_shape, grad_shape = _calculate_shape(out, grad, False)

        # 断言 out_shape 应与 torch.tensor([[10, 5], [10, 5], [10, 5]]) 相等
        assert torch.equal(out_shape, torch.tensor([[10, 5], [10, 5], [10, 5]]))
        # 断言 grad_shape 应与 torch.tensor([[5, 10], [5, 10]]) 相等
        assert torch.equal(grad_shape, torch.tensor([[5, 10], [5, 10]]))
    # 定义一个测试嵌套异常检测的函数
    def test_nested_anomaly_detect_nan(self):
        # 设置大小为10的变量
        size = 10

        # 定义一个自定义的函数类 MyFunc，继承自 Function 类
        class MyFunc(Function):
            # 静态方法：前向传播函数
            @staticmethod
            def forward(ctx, inp1, fail_0th):
                # 将 fail_0th 存储在上下文中
                ctx.fail_0th = fail_0th
                # 保存 inp1，以便反向传播使用
                ctx.save_for_backward(inp1)
                # 返回 inp1 按列求和的结果
                return inp1.sum(0, keepdim=True)

            # 静态方法：反向传播函数
            @staticmethod
            def backward(ctx, gO):
                # 从上下文中恢复保存的张量
                (inp,) = ctx.saved_tensors
                # 获取 fail_0th
                fail_0th = ctx.fail_0th
                # 创建 g 的克隆并扩展到大小为 size
                g = gO.clone().expand(size)
                # 调用 MyFunc2 的自定义应用方法，并传入相应参数
                gI = MyFunc2.apply(g * inp, g + inp, fail_0th)
                # 返回反向传播的梯度
                return gI, None

        # 定义另一个自定义的函数类 MyFunc2，继承自 Function 类
        class MyFunc2(Function):
            # 静态方法：前向传播函数
            @staticmethod
            def forward(ctx, inp1, inp2, fail_0th):
                # 将 fail_0th 存储在上下文中
                ctx.fail_0th = fail_0th
                # 返回 inp1 * 2.0 + inp2 的结果
                return inp1 * 2.0 + inp2

            # 静态方法：反向传播函数
            @staticmethod
            def backward(ctx, gO):
                # 获取 fail_0th
                fail_0th = ctx.fail_0th
                # 克隆 gO 到 g1 和 g2
                g1 = gO.clone()
                g2 = gO.clone()
                # 将 g1 和 g2 的第一个元素设为 0
                g1[0] = 0
                g2[0] = 0
                # 如果 fail_0th 为真，则将 g1 的第一个元素除以 0（生成 NaN）
                if fail_0th:
                    g1[0] /= 0
                else:
                    # 否则将 g2 的第一个元素除以 0（生成 NaN）
                    g2[0] /= 0
                # 返回反向传播的梯度 g1, g2, None
                return g1, g2, None

        # 创建一个大小为 size 的随机张量 inp，要求梯度跟踪
        inp = torch.rand(size, requires_grad=True)
        # 使用 MyFunc 的自定义应用方法，传入 inp 和 True
        out = MyFunc.apply(inp, True)
        # 计算 out 对 inp 的梯度
        (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
        # 计算 ginp 的元素和
        gsum = ginp.sum()
        # 执行反向传播
        gsum.backward()  # 应该不会失败

        # 创建一个大小为 size 的随机张量 inp，要求梯度跟踪
        inp = torch.rand(size, requires_grad=True)
        # 使用 MyFunc 的自定义应用方法，传入 inp 和 True
        out = MyFunc.apply(inp, True)
        # 计算 out 对 inp 的梯度
        (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
        # 计算 ginp 的元素和
        gsum = ginp.sum()
        # 使用警告捕获上下文记录警告信息
        with warnings.catch_warnings(record=True) as w:
            # 断言引发 RuntimeError，且错误信息中包含指定文本
            with self.assertRaisesRegex(
                RuntimeError,
                "Function 'MyFunc2Backward' returned nan values in its 0th output.",
            ):
                # 使用异常检测，执行 gsum 的反向传播
                with detect_anomaly():
                    gsum.backward()
        # 断言特定的警告信息存在于捕获的警告中
        self.assertIn("No forward pass information", str(w[1].message))

        # 创建一个大小为 size 的随机张量 inp，要求梯度跟踪
        inp = torch.rand(size, requires_grad=True)
        # 使用警告捕获上下文记录警告信息
        with warnings.catch_warnings(record=True) as w:
            # 断言引发 RuntimeError，且错误信息中包含指定文本
            with self.assertRaisesRegex(
                RuntimeError,
                "Function 'MyFunc2Backward' returned nan values in its 1th output.",
            ):
                # 使用异常检测
                with detect_anomaly():
                    # 使用 MyFunc 的自定义应用方法，传入 inp 和 False
                    out = MyFunc.apply(inp, False)
                    # 计算 out 对 inp 的梯度
                    (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)
                    # 计算 ginp 的元素和
                    gsum = ginp.sum()
                    # 执行反向传播
                    gsum.backward()
        # 断言特定的警告信息存在于捕获的警告中
        self.assertIn("MyFunc2.apply", str(w[1].message))
        self.assertIn("MyFunc.apply", str(w[2].message))
    def test_anomaly_grad_warnings(self):
        # 定义测试函数，用于检测梯度异常和警告

        # 嵌套类，用于捕获标准错误流
        class StdErrDiverter:
            def __enter__(self):
                self.stderr_orig = sys.stderr
                self.stderr_new = io.StringIO()
                sys.stderr = self.stderr_new
                return self

            def __exit__(self, *args):
                self.captured = self.stderr_new.getvalue()
                sys.stderr = self.stderr_orig

        # 期望捕获特定的 RuntimeError 异常，并检查其包含的错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            "one of the variables needed for gradient computation has been "
            "modified by an inplace operation",
        ):
            # 捕获所有警告信息
            with warnings.catch_warnings(record=True) as w:
                # 启用异常检测上下文管理器
                with detect_anomaly():
                    a = torch.randn(5, requires_grad=True)
                    d1 = a + 1
                    d2 = d1**2
                    d1 += 1
                    torch.autograd.grad(d2.sum(), a)

        # 断言捕获的警告数量为 2
        self.assertEqual(len(w), 2)
        # 断言第一个警告信息包含特定文本
        self.assertIn("Anomaly Detection has been enabled", str(w[0].message))
        # 断言第二个警告信息包含特定文本
        self.assertIn("Error detected in PowBackward0", str(w[1].message))

        # 再次进行异常捕获，此时期望的 RuntimeError 异常会被打印到标准错误流
        with self.assertRaisesRegex(
            RuntimeError,
            "one of the variables needed for gradient computation has been "
            "modified by an inplace operation",
        ):
            # 捕获所有警告信息
            with warnings.catch_warnings(record=True) as w:
                # 启用异常检测上下文管理器
                with detect_anomaly():
                    # 设置警告为错误级别，使警告成为异常
                    warnings.simplefilter("error")
                    # 使用标准错误流重定向器
                    with StdErrDiverter() as s:
                        a = torch.randn(5, requires_grad=True)
                        d1 = a + 1
                        d2 = d1**2
                        d1 += 1
                        torch.autograd.grad(d2.sum(), a)

        # 断言捕获的警告数量为 1
        self.assertEqual(len(w), 1)
        # 断言捕获的标准错误流包含特定文本
        self.assertIn("Anomaly Detection has been enabled", str(w[0].message))
        self.assertIn("Error detected in PowBackward0", s.captured)
    def test_anomaly_assign_parent_cleanup(self):
        # 测试当调用 assign_parent 时，Python 对象的正确清理

        def get_ref():
            # 使用 torch.exp，但任何在其梯度模式下构造新节点的函数都可以工作
            x = torch.randn(2, 2, requires_grad=True)
            t = x.exp()

            # 当 create_graph=True 时，ExpBackward 调用 mul，创建 MulBackward 节点
            # 在异常模式下，将一个引用 MulBackward 的 "parent" ExpBackward 的 PyObject 添加到
            # MulBackward 的异常元数据字典中，形成以下引用链:
            #
            # grad -> MulBackward -> PyObject -> ExpBackward
            #
            with detect_anomaly():
                grad = torch.autograd.grad(t, x, torch.ones_like(t), create_graph=True)

            # 我们添加一个对新 Foo 对象的弱引用，将其插入到 ExpBackward 的元数据字典中
            #
            # (PyObject) -> ExpBackward -> dict -> *Foo*
            #            t ----^        WeakRef ---^
            #
            # 我们希望通过查看 t 销毁后是否未保持 Foo 的存活状态来测试 PyObject 是否被销毁
            class Foo:
                pass

            my_obj = Foo()
            meta_dict = t.grad_fn.metadata
            meta_dict[0] = my_obj
            ref = weakref.ref(my_obj)
            return t, ref

        t, ref = get_ref()
        # 断言 ref() 不为空
        self.assertIsNotNone(ref())
        # 删除 t
        del t
        # 断言 ref() 为空
        self.assertIsNone(ref())
    def test_nested_anomaly_printstack_cleanup(self):
        # 测试元数据字典 PyObject 是否被正确销毁

        def get_ref():
            # 获取引用
            # 这段代码类似于 test_anomaly_assign_parent_cleanup 中的构造：
            #
            # MyFuncBackward2 -> PyObject -> MyFuncBackward -> dict -> Foo
            #                               out ---^         WeakRef ---^
            #
            # 我们想要检查的是，即使 MyFunc2Backward 的 AnomalyMetadata 调用 printstack，
            # 这会进行一些 Python 对象操作，Foo 仍然能够正确被销毁。
            #
            # 你可能会想为什么我们仍然需要 test_anomaly_assign_parent_cleanup，
            # 因为如果 PyObject 在这里没有被销毁，这个测试也会检测到吗？
            # 答案是，自定义函数的 PyObject（如 THPFunction）实际上只持有 C++ 节点的弱引用！

            class MyFunc(Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return x

                @staticmethod
                def backward(ctx, gO):
                    (x,) = ctx.saved_tensors
                    return MyFunc2.apply(x)

            class MyFunc2(Function):
                @staticmethod
                def forward(ctx, x):
                    return x

                @staticmethod
                def backward(ctx, gO):
                    return gO + float("NaN")

            # 创建一个需要梯度的随机输入
            inp = torch.rand(1, requires_grad=True)
            # 调用 MyFunc 的前向传播
            out = MyFunc.apply(inp)
            # 计算梯度
            (ginp,) = torch.autograd.grad(out, (inp,), create_graph=True)

            # 捕获运行时警告，并断言出现特定异常
            with warnings.catch_warnings(record=True) as w:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Function 'MyFunc2Backward' returned nan values in its 0th output.",
                ):
                    with detect_anomaly():
                        ginp.backward()

            # 定义一个简单的类 Foo
            class Foo:
                pass

            # 创建 Foo 类的一个实例
            my_obj = Foo()
            # 获取 out 的梯度函数的元数据字典
            meta_dict = out.grad_fn.metadata
            # 将 my_obj 存入元数据字典中的第一个位置
            meta_dict[0] = my_obj
            # 创建 my_obj 的弱引用
            ref = weakref.ref(my_obj)
            return out, ref

        # 调用 get_ref 函数，获取 out 和 my_obj 的弱引用 ref
        t, ref = get_ref()
        # 断言 ref 不为 None
        self.assertIsNotNone(ref())
        # 删除 t（可能会触发对象的析构函数）
        del t
        # 断言 ref 为 None，即对象已被正确销毁
        self.assertIsNone(ref())
    def test_anomaly_mode_no_check_nan(self):
        # 定义一个自定义的 Torch 自动求导函数 MyFunc
        class MyFunc(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，简单地返回输入的克隆
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            # 反向传播函数，返回全为 NaN 的 Tensor，大小为 10x10
            def backward(ctx, gO):
                return torch.tensor(float("nan")).expand(10, 10)

        # 定义一个运行 MyFunc 的函数
        def run_fn(a):
            out = MyFunc.apply(a)
            return out.sum()

        # 使用 warnings 模块捕获警告信息
        with warnings.catch_warnings(record=True) as w:
            # 开启 Torch 的异常检测模式，不检查 NaN
            with torch.autograd.detect_anomaly(check_nan=False):
                # 创建一个随机张量，要求梯度
                inp = torch.rand(10, 10, requires_grad=True)
                # 执行运行函数
                out = run_fn(inp)
                # 对输出进行反向传播，保留计算图
                out.backward(retain_graph=True)

                # 进入另一个异常检测模式，检查 NaN
                with torch.autograd.detect_anomaly(check_nan=True):
                    # 断言应该会抛出 RuntimeError 异常，指示 MyFuncBackward 的输出包含 NaN 值
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Function 'MyFuncBackward' returned nan values in its 0th output.",
                    ):
                        out.backward(retain_graph=True)

                # 普通的反向传播调用，不保留计算图
                out.backward()

    def test_no_grad_copy(self):
        # 定义一个自动求导函数 MyFunc，保存梯度指针作为类静态变量
        class MyFunc(Function):
            static_grad_ptr = None

            @staticmethod
            # 前向传播函数，返回两个输入的和
            def forward(ctx, inp1, inp2):
                return inp1 + inp2

            @staticmethod
            # 反向传播函数，保存梯度数据指针并返回梯度本身
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad.data_ptr()
                return grad, grad

        # 定义一个非连续梯度的自动求导函数
        class NonContGradFunc(Function):
            @staticmethod
            # 前向传播函数，返回大小为输入张量的标量 1.0 的张量
            def forward(ctx, inp1):
                ctx.size = inp1.size()
                return torch.tensor([1.0])

            @staticmethod
            # 反向传播函数，返回与输入大小相同的全 1 张量
            def backward(ctx, grad):
                return torch.ones(1).expand(ctx.size)

        # 创建两个随机张量，要求梯度
        a = torch.randn(5, 6, requires_grad=True)
        b = torch.randn(5, 6, requires_grad=True)
        # 非连续梯度应该被复制
        NonContGradFunc.apply(MyFunc.apply(a, b)).backward()
        # 断言 a 和 b 的梯度指针不等于 MyFunc 的静态梯度指针
        self.assertFalse(a.grad.data_ptr() == MyFunc.static_grad_ptr)
        self.assertFalse(b.grad.data_ptr() == MyFunc.static_grad_ptr)
        # 测试用例，应该触发其中一个 a 或 b 不复制的情况
        a.grad = b.grad = None
        MyFunc.apply(a, b)[1][0].backward()
        p_g = MyFunc.static_grad_ptr
        p_a = a.grad.data_ptr()
        p_b = b.grad.data_ptr()
        # 检查 a 和 b 使用了不同的梯度缓冲区
        self.assertFalse(p_a == p_b)
        # 检查其中一个使用了计算出的缓冲区
        self.assertTrue(p_a == p_g or p_b == p_g)
    def test_no_grad_copy_sparse(self):
        # create autograd function that saves grad pointer as class static
        # 定义一个自动求导函数，将梯度指针保存为类的静态属性

        class MyFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2
            # 正向传播函数，返回输入的和

            @staticmethod
            def backward(ctx, grad):
                MyFunc.static_grad_ptr = grad._values().data_ptr()
                return grad, grad
                # 反向传播函数，保存梯度的数据指针，并返回相同的梯度

        class NonContGradFunc(Function):
            static_grad_ptr = None

            @staticmethod
            def forward(ctx, inp1, inp2):
                return inp1 + inp2
            # 正向传播函数，返回输入的和

            @staticmethod
            def backward(ctx, grad):
                # Create a sparse tensor with non-contigous indices and values
                # and return as grad.
                v = torch.rand(1, 3)
                i = torch.ones(1, 1, dtype=torch.long)
                nv = v.expand(8, 3)
                ni = i.expand(1, 8)
                ngrad = torch.sparse_coo_tensor(ni, nv, (10, 3), dtype=torch.float32)
                NonContGradFunc.static_grad_ptr = ngrad._values().data_ptr()
                return ngrad, ngrad
                # 反向传播函数，创建具有非连续索引和值的稀疏张量，并作为梯度返回

        a = torch.randn(10, 3, requires_grad=True)
        b = torch.randn(10, 3, requires_grad=True)
        input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        offsets = torch.tensor([0, 4])
        import torch.nn.functional as F

        # test case that should trigger no copy for one of a,b
        # 测试用例，应该对 a 或 b 中的一个触发不复制操作

        emb_matrix = MyFunc.apply(a, b)
        # 使用 MyFunc 自动求导函数对 a 和 b 进行操作，得到嵌入矩阵

        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        # 计算使用嵌入矩阵进行稀疏嵌入的损失和

        loss.backward(retain_graph=True)
        # 反向传播损失函数，保留计算图

        p_g = MyFunc.static_grad_ptr
        # 获取 MyFunc 类的静态梯度指针

        p_a = a.grad._values().data_ptr()
        # 获取 a 的梯度值的数据指针

        p_b = b.grad._values().data_ptr()
        # 获取 b 的梯度值的数据指针

        # check a,b uses different grad buffer
        # 检查 a 和 b 是否使用不同的梯度缓冲区
        self.assertFalse(p_a == p_b)

        # check one of them is using the computed buffer
        # 检查它们中的一个是否使用了计算得到的缓冲区
        self.assertTrue(p_a == p_g or p_b == p_g)

        # Run backwards multiple times to ensure accumulation works.
        # 多次运行反向传播以确保累积工作正常
        for i in range(10):
            loss.backward(retain_graph=True)

        # non-contiguous indices and value, we should trigger a copy.
        # 非连续的索引和值，应该触发复制操作
        a.grad = b.grad = None

        emb_matrix = NonContGradFunc.apply(a, b)
        # 使用 NonContGradFunc 自动求导函数对 a 和 b 进行操作，得到嵌入矩阵

        loss = F.embedding_bag(emb_matrix, input, offsets, sparse=True).sum()
        # 计算使用嵌入矩阵进行稀疏嵌入的损失和

        loss.backward(retain_graph=True)
        # 反向传播损失函数，保留计算图

        p_g = NonContGradFunc.static_grad_ptr
        # 获取 NonContGradFunc 类的静态梯度指针

        p_a = a.grad._values().data_ptr()
        # 获取 a 的梯度值的数据指针

        p_b = b.grad._values().data_ptr()
        # 获取 b 的梯度值的数据指针

        # check a,b uses different grad buffer
        # 检查 a 和 b 是否使用不同的梯度缓冲区
        self.assertFalse(p_a == p_b)

        # Verify we cloned both grads.
        # 验证我们克隆了两个梯度
        self.assertFalse(p_a == p_g)
        self.assertFalse(p_b == p_g)

        # Run backwards multiple times to ensure accumulation works.
        # 多次运行反向传播以确保累积工作正常
        for i in range(10):
            loss.backward(retain_graph=True)
    # 定义单输入梯度检查的测试函数
    def test_gradcheck_single_input(self):
        # 内部函数，用于检查梯度的计算
        def check(fast_mode):
            # 定义一个简单的函数，对输入乘以5
            def f(inp):
                return inp.mul(5)

            # 对 f 函数进行梯度检查
            gradcheck(
                f,
                torch.rand(10, dtype=torch.float64, requires_grad=True),
                fast_mode=fast_mode,
            )
            # 对 f 函数进行二阶梯度检查
            gradgradcheck(
                f,
                torch.rand(10, dtype=torch.float64, requires_grad=True),
                fast_mode=fast_mode,
            )

        # 分别以快速模式和非快速模式调用 check 函数
        check(fast_mode=True)
        check(fast_mode=False)

    # 使用@parametrize装饰器，测试不同的稀疏张量布局
    @parametrize(
        "layout",
        (
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        ),
    )
    # 定义测试稀疏张量输入的函数
    def test_gradcheck_input(self, layout):
        # 根据布局类型设置不同的块大小和大小
        if layout in {torch.sparse_bsr, torch.sparse_bsc}:
            blocksize = (2, 2)
            size = (4, 8)
        else:
            blocksize = None
            size = (2, 2)

        # 内部函数，用于检查梯度的计算和masked选项
        def check(fast_mode, masked):
            # 定义一个简单的函数，对稀疏张量的所有元素求和
            def fn(sparse):
                return torch.sum(sparse)

            # 对 fn 函数进行梯度检查
            gradcheck(
                fn,
                torch.rand(size, dtype=torch.double)
                .to_sparse(layout=layout, blocksize=blocksize)
                .requires_grad_(),
                masked=masked,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )

        # 使用 product 生成快速模式和非快速模式以及masked选项的组合
        for fast_mode, masked in product(*[(True, False)] * 2):
            # 调用 check 函数
            check(fast_mode=fast_mode, masked=masked)
    # 定义一个测试函数，用于检查非确定性函数的梯度
    def test_gradcheck_nondeterministic(self):
        # 定义一个继承自Function的非确定性函数类NonDetFunc
        class NonDetFunc(Function):
            # 前向传播函数：保存jitter值到上下文对象ctx，并返回输入x
            @staticmethod
            def forward(ctx, x, jitter=0.0):
                ctx._jitter = jitter
                return x

            # 反向传播函数：计算梯度，其中使用了随机数和之前保存的jitter值
            @staticmethod
            def backward(ctx, grad_out):
                return (
                    NonDetFunc.apply(grad_out, ctx._jitter)
                    * (1 + torch.rand_like(grad_out) * ctx._jitter),
                    None,
                )

        # 定义一个检查函数，接收一个布尔参数fast_mode
        def check(fast_mode):
            # 创建一个随机的输入张量inp，类型为双精度浮点型，需要计算梯度
            inp = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
            # 进行梯度检查，使用NonDetFunc的前向传播，忽略批处理梯度检查
            gradcheck(
                lambda x: NonDetFunc.apply(x, 0.0),
                inp,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )
            # 检查是否抛出RuntimeError异常，异常消息为"Backward is not reentrant"
            with self.assertRaisesRegex(RuntimeError, "Backward is not reentrant"):
                gradcheck(
                    lambda x: NonDetFunc.apply(x, 1e-6),
                    inp,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            # 检查是否抛出RuntimeError异常，异常消息为"Backward is not reentrant"
            with self.assertRaisesRegex(RuntimeError, "Backward is not reentrant"):
                gradgradcheck(
                    lambda x: NonDetFunc.apply(x, 1e-12),
                    inp,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            # 进行梯度检查，使用NonDetFunc的前向传播，设置非确定性容忍度为1e-5，忽略批处理梯度检查
            gradcheck(
                lambda x: NonDetFunc.apply(x, 0.0),
                inp,
                nondet_tol=1e-5,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )
            # 进行梯度检查，使用NonDetFunc的前向传播，设置非确定性容忍度为1e-5，忽略批处理梯度检查
            gradcheck(
                lambda x: NonDetFunc.apply(x, 1e-6),
                inp,
                nondet_tol=1e-5,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )
            # 进行二阶梯度检查，使用NonDetFunc的前向传播，设置非确定性容忍度为1e-5，忽略批处理梯度检查
            gradgradcheck(
                lambda x: NonDetFunc.apply(x, 1e-12),
                inp,
                nondet_tol=1e-5,
                check_batched_grad=False,
                fast_mode=fast_mode,
            )

        # 分别以快速模式和非快速模式调用检查函数check
        check(fast_mode=True)
        check(fast_mode=False)
    # 定义测试函数 test_gradcheck_validates_inputs，用于验证 gradcheck 函数对输入的有效性进行检查
    def test_gradcheck_validates_inputs(self):
        
        # 定义内部函数 check，用于具体检查不同参数和情况下的 gradcheck 行为
        def check(fast_mode):
            # 创建一个形状为 (10,) 的随机张量 x，并标记为需要梯度，转换为稀疏张量
            x = torch.rand(10, requires_grad=True).to_sparse()
            
            # 调用 gradcheck 函数，检查是否通过梯度检查
            self.assertTrue(
                gradcheck(
                    lambda x: x.to_dense(),  # 转换为稠密张量的 lambda 函数
                    (x,),  # 参数是 x
                    check_batched_grad=False,  # 禁用批量梯度检查
                    atol=1e-1,  # 允许的绝对误差
                    fast_mode=fast_mode,  # 是否启用快速模式
                    masked=True,  # 是否掩码化
                )
            )
            
            # 再次调用 gradcheck 函数，检查不掩码化情况下是否通过梯度检查
            self.assertFalse(
                gradcheck(
                    lambda x: x.to_dense(),  # 转换为稠密张量的 lambda 函数
                    (x,),  # 参数是 x
                    masked=False,  # 不使用掩码
                    check_batched_grad=False,  # 禁用批量梯度检查
                    raise_exception=False,  # 不抛出异常
                    fast_mode=fast_mode,  # 是否启用快速模式
                )
            )
            
            # 再次调用 gradcheck 函数，检查不掩码化和不使用梯度的情况下是否通过梯度检查
            self.assertTrue(
                gradcheck(
                    lambda x: x.to_dense(masked_grad=False),  # 不使用掩码化梯度的 lambda 函数
                    (x,),  # 参数是 x
                    masked=False,  # 不使用掩码
                    atol=1e-1,  # 允许的绝对误差
                    check_batched_grad=False,  # 禁用批量梯度检查
                    raise_exception=False,  # 不抛出异常
                    fast_mode=fast_mode,  # 是否启用快速模式
                )
            )

            # 当所有输入张量均不需要梯度时（即使 raise_exception=False 也会抛出异常）
            x = torch.rand(10, requires_grad=False)
            with self.assertRaisesRegex(
                ValueError, "at least one input tensor to require gradient"
            ):
                gradcheck(lambda x: x, (x,), raise_exception=False, fast_mode=fast_mode)

            # 当输入不是双精度时发出警告
            x = torch.ones(1, dtype=torch.float32, requires_grad=True)
            with self.assertWarnsRegex(
                UserWarning, "Input #0 requires gradient and is not a double precision"
            ):
                self.assertTrue(
                    gradcheck(lambda x: x, (x,), atol=1e-1, fast_mode=fast_mode)
                )

            # 当布局不是 mkldnn（即具有步长）且输入具有步长为 0 的维度时，总是会抛出异常
            x = torch.ones(1, dtype=torch.float64, requires_grad=True)
            x = x.expand((2, 2))  # 扩展张量形状
            with self.assertRaisesRegex(
                RuntimeError, "The 0th input has a dimension with stride 0"
            ):
                gradcheck(lambda x: x, (x,), raise_exception=False, fast_mode=fast_mode)

        # 分别使用 fast_mode=True 和 fast_mode=False 调用 check 函数
        check(fast_mode=True)
        check(fast_mode=False)

    # 若 MKL-DNN 构建被禁用，则跳过该测试
    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_gradcheck_validates_input_mkldnn(self):
        # 当使用 mkldnn 输入时，不允许进行前向模式测试
        # 更新下面的容差以确保梯度即使在单精度浮点数中也匹配
        # 使用警告断言来隐藏浮点数32位警告
        x = torch.ones(1).to_mkldnn().requires_grad_()
        with self.assertWarnsRegex(
            UserWarning, "Input #0 requires gradient and is not a double precision"
        ):
            with self.assertRaisesRegex(
                ValueError, "MKLDNN inputs are not support for forward AD gradcheck."
            ):
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    raise_exception=False,
                    fast_mode=False,
                    check_forward_ad=True,
                    atol=1e-1,
                    rtol=1e-1,
                )

        with self.assertWarnsRegex(
            UserWarning, "Input #0 requires gradient and is not a double precision"
        ):
            with self.assertRaisesRegex(
                ValueError, "MKLDNN inputs are not support for forward AD gradcheck."
            ):
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    raise_exception=False,
                    fast_mode=True,
                    check_forward_ad=True,
                    atol=1e-1,
                    rtol=1e-1,
                )

    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    def test_gradcheck_test_outputs(self):
        def check(fast_mode):
            # 当输出为稀疏张量时（即使 raise_exception=False 也总是抛出异常）
            x = torch.rand(10, requires_grad=True).to_sparse()
            with self.assertRaisesRegex(
                ValueError, "Sparse output is not supported at gradcheck yet"
            ):
                gradcheck(
                    lambda x: x,
                    (x,),
                    masked=True,
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )

            # 当输出为 mkldnn 张量时（即使 raise_exception=False 也总是抛出异常）
            root = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
            with self.assertRaisesRegex(
                ValueError, "MKLDNN output is not supported at gradcheck yet"
            ):
                gradcheck(
                    lambda x: x.to_mkldnn(),
                    (root,),
                    check_batched_grad=False,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )

        check(fast_mode=True)
        check(fast_mode=False)
    def test_gradcheck_check_no_differentiable_outputs(self):
        def check(fast_mode):
            # 定义一个函数，用于检查梯度检查
            # 当所有输出都不可微时，但数值梯度不为零时抛出运行时错误
            x = torch.ones((1,), requires_grad=True)
            # 使用断言检查是否捕获到预期的运行时错误信息
            with self.assertRaisesRegex(
                RuntimeError, "Numerical gradient for function expected to be zero"
            ):
                gradcheck(lambda x: torch.tensor([x]), x)
            # 使用 gradcheck 函数检查梯度是否为零，不抛出异常时返回 False
            self.assertFalse(
                gradcheck(
                    lambda x: torch.tensor([x]),
                    x,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

            # 当没有任何输出时，gradcheck 应该成功通过检查
            self.assertTrue(gradcheck(lambda x: (), (x,), fast_mode=fast_mode))

        # 调用 check 函数，分别传入 fast_mode=True 和 fast_mode=False 进行检查
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_check_batched_grad(self):
        def check(fast_mode):
            # 定义一个函数，用于检查批处理梯度检查
            x = torch.rand(10, dtype=torch.double, requires_grad=True).to_sparse()
            # 在计算批处理梯度时会抛出运行时错误，并打印大量错误信息
            with self.assertRaisesRegex(
                RuntimeError,
                "gradcheck or gradgradcheck failed while testing batched gradient",
            ):
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    masked=True,
                    check_batched_grad=True,
                    fast_mode=fast_mode,
                )
            # 使用 gradcheck 函数检查批处理梯度是否通过，不抛出异常时返回 False
            self.assertFalse(
                gradcheck(
                    lambda x: x.to_dense(),
                    (x,),
                    masked=True,
                    check_batched_grad=True,
                    raise_exception=False,
                    fast_mode=fast_mode,
                )
            )

        # 调用 check 函数，分别传入 fast_mode=True 和 fast_mode=False 进行检查
        check(fast_mode=True)
        check(fast_mode=False)

    def test_gradcheck_undefined_grad(self):
        def check(fast_mode):
            # 当在反向传播时遇到运行时错误时
            def fn(x):
                def hook(x):
                    if x is None:
                        raise RuntimeError("x is undefined")

                y = x.clone()
                y.register_hook(hook)
                return y

            x = torch.ones(1, dtype=torch.double, requires_grad=True)
            # 使用 assertWarnsRegex 检查是否捕获到预期的用户警告信息
            with self.assertWarnsRegex(
                UserWarning,
                "Backwards compatibility: New undefined gradient support checking feature",
            ):
                # 使用 gradcheck 函数检查是否抛出预期的运行时错误
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Expected backward function to handle undefined output grads",
                ):
                    gradcheck(fn, (x,), fast_mode=fast_mode)
                # 使用 gradcheck 函数检查是否抛出预期的运行时错误，不抛出异常时返回 False
                self.assertFalse(
                    gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode)
                )

        # 调用 check 函数，分别传入 fast_mode=True 和 fast_mode=False 进行检查
        check(fast_mode=True)
        check(fast_mode=False)
    # 定义一个测试函数，用于检查梯度检查中的雅可比矩阵不匹配问题
    def test_gradcheck_jacobian_mismatch(self):
        # 定义内部函数check，用于执行具体的梯度检查
        def check(fast_mode):
            # 定义一个函数fn，接受一个输入x，返回一个与x相同的张量，并注册一个钩子函数用于梯度计算
            def fn(x):  # R -> R, C -> C
                y = x.clone()
                y.register_hook(lambda x: x + 1e-2)
                return y

            # 创建一个2x2的全1张量x，需要计算梯度
            x = torch.ones(2, 2, requires_grad=True)
            # 断言梯度检查时出现运行时错误，错误信息包含"Jacobian mismatch for output 0 with respect to input 0"
            with self.assertRaisesRegex(
                RuntimeError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            # 断言梯度检查返回False，忽略异常，使用fast_mode参数
            self.assertFalse(
                gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode)
            )

            # 创建一个2x2的全1张量x_c，需要计算梯度，数据类型为torch.complex128
            x_c = torch.ones(2, 2, requires_grad=True, dtype=torch.complex128)
            # 断言梯度检查时出现运行时错误，错误信息包含"While considering the imaginary part of complex outputs only"
            with self.assertRaisesRegex(
                RuntimeError,
                "While considering the imaginary part of complex outputs only",
            ):
                gradcheck(fn, (x_c,), fast_mode=False)
            # 断言梯度检查返回False，忽略异常，使用fast_mode参数为False
            self.assertFalse(
                gradcheck(fn, (x_c,), raise_exception=False, fast_mode=False)
            )

            # 定义函数fn2，接受一个输入x，返回一个复数张量，注册一个钩子函数用于梯度计算
            def fn2(x):  # R -> C
                y = torch.complex(x, x)
                y.register_hook(lambda x: x + 1e-2)
                return y

            # 创建一个2x2的全1张量x，需要计算梯度
            x = torch.ones(2, 2, requires_grad=True)
            # 断言梯度检查时出现运行时错误，错误信息包含"While considering the imaginary part of complex outputs only"
            with self.assertRaisesRegex(
                RuntimeError,
                "While considering the imaginary part of complex outputs only",
            ):
                gradcheck(fn2, (x,), fast_mode=False)
            # 断言梯度检查返回False，忽略异常，使用fast_mode参数为False
            self.assertFalse(
                gradcheck(fn2, (x,), raise_exception=False, fast_mode=False)
            )

            # 定义函数fn3，接受一个输入x_c，返回一个实部张量，注册一个钩子函数用于梯度计算
            def fn3(x):  # C -> R
                y = torch.real(x)
                y.register_hook(lambda x: x + 1e-2)
                return y

            # 断言梯度检查时出现运行时错误，错误信息包含"Jacobian mismatch for output 0 with respect to input 0"
            with self.assertRaisesRegex(
                RuntimeError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn3, (x_c,), fast_mode=False)
            # 断言梯度检查返回False，忽略异常，使用fast_mode参数为False
            self.assertFalse(
                gradcheck(fn3, (x_c,), raise_exception=False, fast_mode=False)
            )

        # 调用check函数，分别使用fast_mode=True和fast_mode=False执行梯度检查
        check(fast_mode=True)
        check(fast_mode=False)

    # 定义一个测试函数，用于检查梯度检查中的稠密和稀疏输入
    def test_gradcheck_dense_and_sparse_inputs(self):
        # 定义内部函数check，用于执行具体的梯度检查
        def check(fast_mode):
            # 定义一个函数fn，接受两个输入x和y，返回x和y的乘积的稀疏张量转换为稠密张量后的结果
            def fn(x, y):
                return x * y.coalesce().to_dense()

            # 创建一个2x2的随机张量a，数据类型为torch.double，需要计算梯度
            a = torch.rand(2, 2, dtype=torch.double, requires_grad=True)
            # 创建一个2x2的随机张量b，数据类型为torch.double，转换为稀疏张量，需要计算梯度
            b = (
                torch.rand(
                    2,
                    2,
                    dtype=torch.double,
                )
                .to_sparse()
                .requires_grad_(True)
            )
            # 断言梯度检查通过，使用fn函数，忽略batched梯度检查，masked=True，使用fast_mode参数
            self.assertTrue(
                gradcheck(
                    fn,
                    (a, b),
                    masked=True,
                    check_batched_grad=False,
                    fast_mode=fast_mode,
                )
            )

        # 调用check函数，分别使用fast_mode=True和fast_mode=False执行梯度检查
        check(fast_mode=True)
        check(fast_mode=False)

    # 跳过测试，如果MKL-DNN构建被禁用
    @unittest.skipIf(
        not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled"
    )
    # 定义测试函数 `test_gradcheck_multiple_mkldnn_inputs`，用于检查多个 MKLDNN 输入的梯度检查
    def test_gradcheck_multiple_mkldnn_inputs(self):
        # 内部函数 check，接受参数 fast_mode
        def check(fast_mode):
            # 定义函数 fn，接受两个参数 x 和 y，返回它们的和（y 转为 dense）
            def fn(x, y):
                return x + y.to_dense()

            # 创建随机张量 a，要求计算梯度
            a = torch.rand(10, requires_grad=True)
            # 创建随机张量 b，转为 MKLDNN 格式并要求计算梯度
            b = torch.rand(10, dtype=torch.float32).to_mkldnn().requires_grad_(True)
            # 断言梯度检查通过
            self.assertTrue(
                gradcheck(
                    fn, (a, b), atol=1e-1, check_batched_grad=False, fast_mode=fast_mode
                )
            )

            # 定义函数 fn2，接受两个参数 x 和 y，返回两者的 dense 和
            def fn2(x, y):
                return x.to_dense() + y.to_dense()

            # 创建随机张量 c，转为 MKLDNN 格式并要求计算梯度
            c = torch.rand(10, dtype=torch.float32).to_mkldnn().requires_grad_(True)
            # 断言梯度检查通过
            self.assertTrue(
                gradcheck(
                    fn, (a, c), atol=1e-1, check_batched_grad=False, fast_mode=fast_mode
                )
            )

        # 使用 fast_mode=True 和 fast_mode=False 调用 check 函数
        check(fast_mode=True)
        check(fast_mode=False)

    # 定义测试函数 `test_gradcheck_output_shape_or_dtype_depend_on_values`，用于检查输出形状或数据类型依赖于值的梯度检查
    def test_gradcheck_output_shape_or_dtype_depend_on_values(self):
        # 内部函数 check，接受参数 fast_mode
        def check(fast_mode):
            # 定义函数 fn，接受一个参数 x，如果所有元素大于等于 1，则返回 x 重复连接的结果，否则返回 x 自身
            def fn(x):
                if torch.all(x >= 1):
                    return torch.cat([x, x])
                else:
                    return x

            # 创建全为 1 的双精度张量 a，要求计算梯度
            a = torch.ones(1, dtype=torch.double, requires_grad=True)
            # 断言梯度检查引发断言错误，指明输入扰动时输出应具有相同的形状
            with self.assertRaisesRegex(
                AssertionError,
                "return outputs with the same shape when inputs are perturbed",
            ):
                self.assertTrue(gradcheck(fn, (a,), fast_mode=fast_mode))

            # 定义函数 fn2，接受一个参数 x，如果所有元素大于等于 1，则返回 x 转为单精度浮点型，否则返回 x 自身
            def fn2(x):
                if torch.all(x >= 1):
                    return x.to(torch.float32)
                else:
                    return x

            # 断言梯度检查引发断言错误，指明输入扰动时输出应具有相同的数据类型
            with self.assertRaisesRegex(
                AssertionError,
                "return outputs with the same dtype when inputs are perturbed",
            ):
                self.assertTrue(gradcheck(fn2, (a,), fast_mode=fast_mode))

        # 使用 fast_mode=True 和 fast_mode=False 调用 check 函数
        check(fast_mode=True)
        check(fast_mode=False)

    # 定义测试函数 `test_gradcheck_complex_non_complex_outputs`，用于检查复杂和非复杂输出的梯度检查
    def test_gradcheck_complex_non_complex_outputs(self):
        # 定义函数 fn，接受两个参数 x 和 y，创建复数张量并返回复数张量和 x + 1
        def fn(x, y):
            z = torch.complex(x, y)
            return z, x + 1

        # 创建全为 1 的双精度张量 a 和 b，要求计算梯度
        a = torch.ones(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.ones(2, 2, requires_grad=True, dtype=torch.float64)
        # 断言梯度检查通过
        self.assertTrue(gradcheck(fn, (a, b)))

        # 定义函数 fn2，接受一个参数 z，返回 z 和 z 的实部
        def fn2(z):
            return z, torch.real(z)

        # 创建全为 1 的复数张量 c，要求计算梯度
        c = torch.ones(2, 2, requires_grad=True, dtype=torch.complex128)
        # 断言梯度检查通过
        self.assertTrue(gradcheck(fn2, (c,)))
    # 定义一个测试方法，用于测试梯度检查和数值雅可比矩阵计算
    def test_gradcheck_get_numerical_jacobian(self):
        # 导入已弃用且不再被 gradcheck 内部使用的 get_numerical_jacobian 函数
        from torch.autograd.gradcheck import get_numerical_jacobian
        
        # 定义一个函数 fn，用于计算输入的两个张量的函数值
        def fn(inputs):
            # get_numerical_jacobian 要求 fn 接收一个元组作为输入
            # 并返回相对于第一个输出的雅可比矩阵
            x = inputs[0]
            y = inputs[1]
            return 2 * x + y, x + 2 * y
        
        # 创建两个随机张量 a 和 b，要求计算其梯度，并使用双精度浮点数类型
        a = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        
        # 使用 assertWarnsRegex 检查是否会发出 FutureWarning 警告
        with self.assertWarnsRegex(
            FutureWarning, "`get_numerical_jacobian` was part of PyTorch's private API"
        ):
            # 调用 get_numerical_jacobian 计算函数 fn 的数值雅可比矩阵
            # 目标是张量 a，数值小步长为 1e-6
            jacobian = get_numerical_jacobian(fn, (a, b), target=a, eps=1e-6)
        
        # 使用 assertEqual 检查计算得到的雅可比矩阵是否与期望值相等
        self.assertEqual(jacobian[0], 2 * torch.eye(4, dtype=torch.double))
        
        # 再次使用 assertWarnsRegex 检查 FutureWarning 警告
        with self.assertWarnsRegex(
            FutureWarning, "`get_numerical_jacobian` was part of PyTorch's private API"
        ):
            # 再次调用 get_numerical_jacobian 计算函数 fn 的数值雅可比矩阵
            # 没有指定目标张量，数值小步长为 1e-6
            jacobian = get_numerical_jacobian(fn, (a, b), eps=1e-6)
        
        # 使用 assertEqual 检查计算得到的第一个输出的雅可比矩阵是否与期望值相等
        self.assertEqual(jacobian[0], 2 * torch.eye(4, dtype=torch.double))
        # 使用 assertEqual 检查计算得到的第二个输出的雅可比矩阵是否与期望值相等
        self.assertEqual(jacobian[1], 1 * torch.eye(4, dtype=torch.double))
        
        # 使用 assertRaisesRegex 检查是否会引发 ValueError 异常，并验证异常消息
        with self.assertRaisesRegex(ValueError, "Expected grad_out to be 1.0"):
            # 尝试调用 get_numerical_jacobian 计算函数 fn 的数值雅可比矩阵
            # 但传递的 grad_out 参数为 2.0，预期引发异常
            jacobian = get_numerical_jacobian(fn, (a, b), eps=1e-6, grad_out=2.0)
    # 定义一个测试方法，用于测试梯度检查和获取解析雅可比矩阵
    def test_gradcheck_get_analytical_jacobian(self):
        # 导入需要的函数模块
        from torch.autograd.gradcheck import get_analytical_jacobian
        
        # 定义一个简单的函数 fn(x, y)，返回一个元组，包含两个张量的线性组合
        def fn(x, y):
            return 2 * x + y, x + 2 * y
        
        # 创建两个随机张量 a 和 b，要求计算梯度，数据类型为双精度浮点数
        a = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        b = torch.rand(2, 2, requires_grad=True, dtype=torch.float64)
        
        # 调用 fn 函数，计算输出结果
        outputs = fn(a, b)
        
        # 使用断言检查是否有未来警告，提醒 `get_analytical_jacobian` 已不再是 PyTorch 的私有 API
        with self.assertWarnsRegex(
            FutureWarning, "`get_analytical_jacobian` was part of PyTorch's private API"
        ):
            # 调用 get_analytical_jacobian 函数，计算 fn 对于 (a, b) 的解析雅可比矩阵
            (
                jacobians,
                reentrant,
                correct_grad_sizes,
                correct_grad_types,
            ) = get_analytical_jacobian((a, b), outputs[0])
        
        # 使用断言检查第一个输入张量的雅可比矩阵是否符合预期的值
        self.assertEqual(jacobians[0], 2 * torch.eye(4, dtype=torch.double))
        
        # 使用断言检查第二个输入张量的雅可比矩阵是否符合预期的值
        self.assertEqual(jacobians[1], 1 * torch.eye(4, dtype=torch.double))
        
        # 使用断言检查 reentrant 变量是否为 True
        self.assertTrue(reentrant)

        # 定义一个继承自 Function 的类 NonDetFunc，用于测试非确定性函数的梯度
        class NonDetFunc(Function):
            @staticmethod
            # 前向传播函数，接收输入张量 x 和可选参数 jitter
            def forward(ctx, x, jitter=0.0):
                ctx._jitter = jitter
                return x
            
            @staticmethod
            # 反向传播函数，接收梯度 grad_out，并返回对输入 x 的梯度
            def backward(ctx, grad_out):
                return (
                    NonDetFunc.apply(grad_out, ctx._jitter)
                    * (1 + torch.rand_like(grad_out) * ctx._jitter),
                    None,
                )
        
        # 对 NonDetFunc 类的 apply 方法进行调用，计算输出结果
        outputs = NonDetFunc.apply(a, 1e-6)
        
        # 使用断言检查是否有未来警告，提醒 `get_analytical_jacobian` 已不再是 PyTorch 的私有 API
        with self.assertWarnsRegex(
            FutureWarning, "`get_analytical_jacobian` was part of PyTorch's private API"
        ):
            # 调用 get_analytical_jacobian 函数，计算 NonDetFunc 对于 a 的解析雅可比矩阵
            (
                jacobians,
                reentrant,
                correct_grad_sizes,
                correct_grad_types,
            ) = get_analytical_jacobian((a,), outputs)
        
        # 使用断言检查 reentrant 变量是否为 False
        self.assertFalse(reentrant)
        
        # 使用断言检查是否会抛出 ValueError 异常，提示预期的 grad_out 值应为 1.0
        with self.assertRaisesRegex(ValueError, "Expected grad_out to be 1.0"):
            # 调用 get_analytical_jacobian 函数，此次期望会抛出异常
            jacobians, _, _, _ = get_analytical_jacobian((a,), outputs, grad_out=2.0)
    # 定义一个测试函数，用于测试自定义错误处理情况
    def test_gradcheck_custom_error(self):
        # 导入 GradcheckError 类，用于捕获梯度检查时的特定错误
        from torch.autograd.gradcheck import GradcheckError
        
        # 定义内部函数 check，用于执行具体的梯度检查测试
        def check(fast_mode):
            # 定义一个函数 fn，对输入的张量进行克隆，并注册一个钩子函数
            def fn(x):
                y = x.clone()
                # 注册一个钩子函数，用于对张量进行操作
                y.register_hook(lambda x: x + 1e-2)
                return y
            
            # 创建一个大小为 (2, 2) 的张量，并设置 requires_grad=True
            x = torch.ones(2, 2, requires_grad=True)
            
            # 使用 assertRaisesRegex 断言捕获 GradcheckError 错误，验证梯度检查失败时的异常信息
            with self.assertRaisesRegex(
                GradcheckError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            
            # 使用 assertRaisesRegex 断言捕获 RuntimeError 错误，验证梯度检查失败时的异常信息
            with self.assertRaisesRegex(
                RuntimeError, "Jacobian mismatch for output 0 with respect to input 0"
            ):
                gradcheck(fn, (x,), fast_mode=fast_mode)
            
            # 使用 assertFalse 断言验证梯度检查失败时返回 False
            self.assertFalse(
                gradcheck(fn, (x,), raise_exception=False, fast_mode=fast_mode)
            )
            
            # 定义一个函数 fn2，直接抛出 RuntimeError 异常
            def fn2(x):
                raise RuntimeError("Not a GradcheckError!")
            
            # 使用 assertRaisesRegex 断言捕获 RuntimeError 错误，验证非 GradcheckError 异常抛出时的异常信息
            with self.assertRaisesRegex(RuntimeError, "Not a GradcheckError!"):
                gradcheck(fn2, (x,), fast_mode=fast_mode, raise_exception=False)
        
        # 分别使用 fast_mode=True 和 fast_mode=False 调用 check 函数进行测试
        check(fast_mode=True)
        check(fast_mode=False)
    # 定义一个测试函数，用于检查梯度
    def test_gradcheck_forward_ad(self):
        # 定义一个简单的函数 fn，对输入 x 和 y 求和并返回，同时返回 y
        def fn(x, y):
            return x + y, y

        # 定义一个有问题的函数 bad_fn，用于测试梯度检查的特殊情况
        def bad_fn(x, y):
            # 检查当前是否处于前向自动微分级别，通过检查 fwAD._current_level 是否大于等于 0 来判断
            is_running_forward_ad = fwAD._current_level >= 0

            # 如果当前处于前向自动微分级别，对 y 进行特殊处理
            if is_running_forward_ad:
                # 解包 y 为原始部分和导数部分
                y_p, y_d = fwAD.unpack_dual(y)
                # 构造一个新的双重数，乘以一个因子 1.1
                y = fwAD.make_dual(y_p, y_d * 1.1)

            # 返回 x + y 和修改后的 y
            return x + y, y

        # 错误信息，用于检查梯度时出现的错误情况
        err_msg = "Jacobian computed with forward mode mismatch for output 0 with respect to input 1"

        # 循环测试两种模式：快速模式和非快速模式
        for fast_mode in [True, False]:
            # 测试所有输入和输出都是实数的情况
            x = torch.rand(2, dtype=torch.double, requires_grad=True)
            y = torch.rand(2, dtype=torch.double, requires_grad=True)

            # 对函数 fn 进行梯度检查，同时检查前向自动微分模式是否开启
            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            # 使用断言检查 bad_fn 是否会引发预期的运行时错误
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

            # 定义一个简单的乘法函数 basic_mul，用于测试复数情况
            def basic_mul(x):
                # 将输入 x 视为实部和虚部，返回其共轭解析形式
                return torch.view_as_real(torch.resolve_conj(x * 1j))

            # 对 basic_mul 进行梯度检查，检查前向自动微分模式是否开启
            gradcheck(basic_mul, x, check_forward_ad=True, fast_mode=fast_mode)

            # 测试一个输入为复数的情况，此时 y 为实数
            x = torch.rand(2, dtype=torch.cdouble, requires_grad=True)
            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            # 使用断言检查 bad_fn 是否会引发预期的运行时错误
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)

            # 测试所有输入和输出都是复数的情况
            y = torch.rand(2, dtype=torch.cdouble, requires_grad=True)
            gradcheck(fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
            # 使用断言检查 bad_fn 是否会引发预期的运行时错误
            with self.assertRaisesRegex(RuntimeError, err_msg):
                gradcheck(bad_fn, (x, y), check_forward_ad=True, fast_mode=fast_mode)
    def test_gradcheck_forward_ad_runs_with_no_requires_grad(self):
        # 定义一个测试函数，用于验证在 requires_grad=False 的情况下 gradcheck 是否正常运行
        # requires_grad 当前被用作 gradcheck 知道哪些函数输入需要可微分的简便方法
        # 该测试检查当输入被传递给函数时，它们不应该具有 requires_grad=True，即使它们在传递给 gradcheck 时可能有 requires_grad=True
        class UserFn(Function):
            @staticmethod
            def forward(ctx, x, y):
                # 如果当前级别大于等于 0，断言 x 和 y 不应该要求梯度
                if fwAD._current_level >= 0:
                    self.assertFalse(x.requires_grad)
                    self.assertFalse(y.requires_grad)
                # 返回输入的克隆副本
                return x.clone(), y.clone()

            @staticmethod
            def jvp(ctx, x_t, y_t):
                # 返回 x_t 和 y_t 本身
                return x_t, y_t

        # 创建两个随机张量 x 和 y，类型为 double，需要梯度
        x = torch.rand(2, dtype=torch.double, requires_grad=True)
        y = torch.rand(2, dtype=torch.double, requires_grad=True)

        # 调用 gradcheck 进行前向 AD 检查，确保其它参数设置为 False
        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=False,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=False,
        )

        # 再次调用 gradcheck 进行前向 AD 检查，开启 undefined_grad 检查
        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=False,
        )

        # 再次调用 gradcheck 进行前向 AD 检查，开启 undefined_grad 和 batched_forward_grad 检查
        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,
        )

        # 重新定义张量 y，不需要梯度
        y = torch.rand(2, dtype=torch.double, requires_grad=False)

        # 再次调用 gradcheck 进行前向 AD 检查，开启 undefined_grad 和 batched_forward_grad 检查
        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,
        )
    def test_gradcheck_forward_ad_respects_requires_grad(self):
        # 定义一个测试函数，用于验证梯度检查时对 requires_grad 属性的尊重

        jvp_count = [0]  # 记录 JVP 调用次数的列表

        class UserFn(Function):
            @staticmethod
            def forward(ctx, x, y):
                # 前向传播函数：返回输入张量的克隆
                return x.clone(), y.clone()

            @staticmethod
            def jvp(ctx, x_t, y_t):
                # JVP（Jacobians Vector Products）函数：记录调用次数并返回输入张量
                jvp_count[0] += 1
                return x_t, y_t

        # 在梯度检查中，需要通过循环遍历每个元素以保证快速和慢速梯度检查具有相同的计数
        x = torch.rand(1, dtype=torch.double, requires_grad=True)
        y = torch.rand(1, dtype=torch.double, requires_grad=True)

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,  # 检查前向自动微分
            check_undefined_grad=False,  # 不检查未定义的梯度
            check_backward_ad=False,  # 不检查后向自动微分
            check_batched_grad=False,  # 不检查批次化梯度
            check_batched_forward_grad=False,  # 不检查批次化前向梯度
        )

        self.assertEqual(jvp_count[0], 2)  # 断言 JVP 调用次数为 2 次（每个输入调用一次）
        jvp_count = [0]

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,  # 检查未定义的梯度
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=False,
        )

        self.assertEqual(
            jvp_count[0], 6
        )  # 断言 JVP 调用次数为 6 次（正常 ZT 和高效 ZT 各两次，每个输入调用三次）
        jvp_count = [0]

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,  # 检查批次化前向梯度
        )

        self.assertEqual(
            jvp_count[0], 12
        )  # 断言 JVP 调用次数为 12 次（批次化计算两个输入，使用 vmap 和循环各六次）
        jvp_count = [0]

        # 重复上一个测试，但标记一个输入的 requires_grad=False
        x = torch.rand(1, dtype=torch.double, requires_grad=True)
        y = torch.rand(1, dtype=torch.double, requires_grad=False)

        gradcheck(
            UserFn.apply,
            (x, y),
            check_forward_ad=True,
            check_undefined_grad=True,
            check_backward_ad=False,
            check_batched_grad=False,
            check_batched_forward_grad=True,
        )

        self.assertEqual(jvp_count[0], 5)  # 断言 JVP 调用次数为 5 次（1 + 1 + 3）
    def test_gradcheck_check_forward_or_backward_only(self):
        """根据不同的参数设置，检查前向传播或后向传播的正确性或错误性"""

        fwd_fail_err_msg = "FAIL FWD"  # 前向传播失败时的错误消息
        bwd_fail_err_msg = "FAIL BWD"  # 后向传播失败时的错误消息

        class UserFn(Function):
            @staticmethod
            def forward(ctx, foo, fwd_bad, bwd_bad):
                ctx.fwd_bad = fwd_bad  # 将前向传播失败标志保存在上下文中
                ctx.bwd_bad = bwd_bad  # 将后向传播失败标志保存在上下文中
                return foo * 2  # 返回输入值 foo 的两倍

            @staticmethod
            def vjp(ctx, gO):
                if ctx.bwd_bad:
                    raise RuntimeError(bwd_fail_err_msg)  # 如果后向传播失败，则抛出运行时错误
                else:
                    return 2 * gO, None, None  # 返回 gO 的两倍作为输出，其他返回 None

            @staticmethod
            def jvp(ctx, gI, _1, _2):
                if ctx.fwd_bad:
                    raise RuntimeError(fwd_fail_err_msg)  # 如果前向传播失败，则抛出运行时错误
                else:
                    return 2 * gI  # 返回 gI 的两倍作为输出

        for fast_mode in (True, False):
            for check_forward_ad in (True, False):
                for check_backward_ad in (True, False):
                    for fwd_bad in (True, False):
                        for bwd_bad in (True, False):
                            fwd_should_fail = fwd_bad and check_forward_ad
                            bwd_should_fail = bwd_bad and check_backward_ad

                            def run():
                                gradcheck(
                                    UserFn.apply,
                                    (x, fwd_bad, bwd_bad),
                                    check_forward_ad=check_forward_ad,
                                    check_backward_ad=check_backward_ad,
                                    check_undefined_grad=check_backward_ad,
                                    check_batched_grad=check_backward_ad,
                                    fast_mode=fast_mode,
                                )

                            x = torch.rand(2, dtype=torch.double, requires_grad=True)

                            if not check_forward_ad and not check_backward_ad:
                                with self.assertRaisesRegex(
                                    AssertionError, "Expected at least one of"
                                ):
                                    run()
                                continue

                            if not fwd_should_fail and not bwd_should_fail:
                                run()
                            else:
                                # 如果两者都失败，后向传播 AD 的失败会“隐藏”前向传播 AD 的失败
                                if fwd_should_fail:
                                    fail_msg = fwd_fail_err_msg
                                if bwd_should_fail:
                                    fail_msg = bwd_fail_err_msg
                                with self.assertRaisesRegex(RuntimeError, fail_msg):
                                    run()
    # 定义测试函数，用于验证梯度检查在批处理梯度情况下的前向自动求导
    def test_gradcheck_forward_ad_batched_grad(self):
        # 创建一个形状为 (2,) 的双精度随机张量，并设置需要计算梯度
        x = torch.rand(2, dtype=torch.double, requires_grad=True)

        # 定义一个函数 fn1，接受一个张量 a 和一个整数 b 作为输入，返回 a 的克隆和 a+1
        def fn1(a: torch.Tensor, b: int):
            return a.clone(), a + 1

        # 对 fn1 函数进行梯度检查，传入参数 (x, 1)
        gradcheck(
            fn1,
            (x, 1),
            check_forward_ad=True,  # 检查前向自动求导
            check_backward_ad=False,  # 不检查反向自动求导
            check_batched_grad=False,  # 不检查批处理梯度
            check_undefined_grad=False,  # 不检查未定义梯度
            check_batched_forward_grad=True,  # 检查批处理前向梯度
        )

        # 定义一个函数 fn2，接受一个张量 a 和一个张量 c 作为输入，返回 a 的克隆
        def fn2(a: torch.Tensor, c: torch.Tensor):
            return a.clone()

        # 对 fn2 函数进行梯度检查，传入参数 (x, x.clone())
        gradcheck(
            fn2,
            (x, x.clone()),
            check_forward_ad=True,  # 检查前向自动求导
            check_backward_ad=False,  # 不检查反向自动求导
            check_batched_grad=False,  # 不检查批处理梯度
            check_undefined_grad=False,  # 不检查未定义梯度
            check_batched_forward_grad=True,  # 检查批处理前向梯度
        )

        # 定义一个继承自 Function 的类 Fn
        class Fn(Function):
            @staticmethod
            def forward(ctx, foo):
                return foo * 2

            @staticmethod
            def vjp(ctx, gO):
                return gO * 2

            @staticmethod
            def jvp(ctx, gI):
                torch.randn_like(gI)
                return gI * 2

        # 设置错误消息字符串
        msg = "vmap: We do not yet support calling random operations inside of vmap"
        # 断言在执行 gradcheck 时会抛出 RuntimeError，并且错误消息符合预期
        with self.assertRaisesRegex(RuntimeError, msg):
            gradcheck(
                Fn.apply, (x,), check_forward_ad=True, check_batched_forward_grad=True
            )

    # 测试版本计数器
    def test_version_counter(self):
        # 创建一个形状为 (1, 2) 的张量 x
        x = torch.randn(1, 2)

        # 获取张量 x 的版本号并保存
        x_saved_version = x._version
        # 对 x 执行两次原地操作 add_
        x.add_(1).add_(1)
        # 断言当前的版本号比之前保存的版本号要大，即版本号已被更新
        self.assertTrue(x._version > x_saved_version)

        # 创建 x 的一个可导视图 xz
        xz = x[:]
        # 断言 x 和 xz 共享相同的版本号计数器
        self.assertTrue(x._version == xz._version)
        # 对 xz 执行原地操作 add_
        xz.add_(1)
        # 断言 x 和 xz 仍然共享相同的版本号计数器
        self.assertTrue(x._version == xz._version)

        # 保存 x 的当前版本号
        x_saved_version = x._version
        # 用一个新的张量替换 x 的数据，保持其版本号不变
        x.data = torch.randn(2, 3)
        # 断言 x 的版本号仍然与之前保存的版本号相同
        self.assertTrue(x._version == x_saved_version)
        # 对 x 执行原地操作 add_
        x.add_(1)
        # 断言当前的版本号比之前保存的版本号要大，即版本号已被更新
        self.assertTrue(x._version > x_saved_version)
        # 确保 x 仍然使用与 xz 共享的版本号计数器
        self.assertTrue(x._version == xz._version)

        # 对 xz 执行原地操作 add_，同时更新 x 的版本号计数器
        xz.add_(1)
        # 断言 x 和 xz 仍然共享相同的版本号计数器
        self.assertTrue(x._version == xz._version)

    # 测试设置数据张量实现类型
    def test_set_data_tensorimpl_type(self):
        # 创建一个形状为 (1, 2) 的密集张量 x
        x = torch.randn(1, 2)
        # 创建一个形状为 [1, 1] 的稀疏张量 x_s
        x_s = torch.sparse_coo_tensor(torch.zeros([1, 1]), torch.ones([1]))
        # 断言设置 x 的数据为 x_s 时会抛出 RuntimeError，并且错误消息符合预期
        with self.assertRaisesRegex(RuntimeError, "incompatible tensor type"):
            x.data = x_s
    # 测试方法：验证在设置数据时保留原始对象的标识符
    def test_set_data_preserve_pyobj(self):
        # 创建一个形状为 (1, 2) 的随机张量 a
        a = torch.randn(1, 2)
        # 创建另一个形状为 (1, 2) 的随机张量 b
        b = torch.randn(1, 2)
        # 记录张量 b 的初始标识符
        b_id_saved = id(b)
        # 将张量 a 的数据赋值给张量 b
        b.data = a
        # 断言赋值后张量 b 的标识符与初始时相同
        self.assertTrue(b_id_saved == id(b))
    
    # 测试方法：验证设置数据时需要张量本身需要梯度
    def test_set_data_self_requires_grad(self):
        # 创建一个需要梯度的张量 a，值为 1.0
        a = torch.tensor(1.0, requires_grad=True)
        # 创建一个张量 b，值为 2.0
        b = torch.tensor(2.0)
        # 创建一个整型张量 c，值为 3
        c = torch.tensor(3, dtype=torch.int64)
        # 将张量 b 的数据赋值给张量 a
        a.data = b
        # 使用断言检测当尝试将整型张量 c 的数据赋给张量 a 时是否会抛出 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError, "must be floating point or complex dtype"
        ):
            a.data = c
    
    # 装饰器：如果在 Windows 系统下则跳过此测试
    @unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
    def test_thread_shutdown(self):
        # 多线程关闭测试方法
        code = """import torch
from torch.autograd import Function

# 定义一个自定义的 PyTorch Function，用于计算梯度
class MyFunction(Function):
    @staticmethod
    # 前向传播函数，接受输入张量 x，直接返回 x
    def forward(ctx, x):
        return x

    @staticmethod
    # 反向传播函数，接受输入梯度 grad，直接返回梯度 grad
    def backward(ctx, grad):
        return grad

# 检查是否有可用的 CUDA 设备，以决定使用 "cuda" 还是 "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 遍历不同的张量形状进行测试
for shape in [(1,), ()]:
    # 创建一个 requires_grad=True 的张量 v，根据 device 指定放置位置
    v = torch.ones(shape, requires_grad=True, device=device)
    # 对自定义 Function MyFunction 应用前向传播和反向传播
    MyFunction.apply(v).backward()

"""
以下为测试用例执行部分，验证 PyTorch API 使用情况的 stderr 输出。
检查是否需要关闭自动求导引擎中的工作线程，
仅当使用 CUDA 或者存在其他加速器时才会启用工作线程。
"""

    @unittest.skipIf(
        IS_MACOS,
        "Fails with SIGBUS on macOS; https://github.com/pytorch/pytorch/issues/25941",
    )
    # 测试深度递归函数的堆栈溢出逃逸机制
    def test_deep_reentrant(self):
        # 定义一个深度递归的自定义 Function
        class DeepReentrant(Function):
            @staticmethod
            # 前向传播函数，处理输入张量 x
            def forward(ctx, x):
                with torch.enable_grad():
                    # 将输入张量 x 作为变量，并设置 requires_grad=True
                    ctx.x = Variable(x.detach(), requires_grad=True)
                    # 对 ctx.x 进行减一操作
                    ctx.x = ctx.x - 1
                # 返回 ctx.x 的 detach 结果
                return ctx.x.detach()

            @staticmethod
            # 反向传播函数，处理输入梯度 x
            def backward(ctx, x):
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    # 递归调用 DeepReentrant 自身，对 ctx.x 求和并反向传播
                    DeepReentrant.apply(ctx.x).sum().backward()
                return x

        # 创建一个张量 v，初始值为 2000.0，requires_grad=True
        v = torch.tensor(2000.0, requires_grad=True)
        # 对 DeepReentrant 自定义 Function 应用前向传播和反向传播
        DeepReentrant.apply(v).sum().backward()

        # 再次测试，以确保在池中重复使用工作线程的堆栈溢出逃逸机制正常工作
        v2 = torch.tensor(200.0, requires_grad=True)
        DeepReentrant.apply(v2).sum().backward()
    # 定义一个测试函数，用于测试可重入优先级问题
    def test_reentrant_priority(self):
        # 用于记录函数调用顺序的列表
        order = []

        # 定义一个自定义的函数类 MyFunction，继承自 torch.autograd.Function
        class MyFunction(Function):
            @staticmethod
            # 前向传播函数，返回输入张量本身
            def forward(ctx, x):
                return x

            @staticmethod
            # 反向传播函数，在 order 列表中添加 "MyFunction"，并返回输入张量本身
            def backward(ctx, x):
                order.append("MyFunction")
                return x

        # 定义一个自定义的函数类 Reentrant，继承自 torch.autograd.Function
        class Reentrant(Function):
            @staticmethod
            # 前向传播函数，创建一个带梯度的 Variable，然后返回其去掉梯度信息的张量
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = Variable(x.detach(), requires_grad=True)
                    ctx.x = ctx.x - 1
                return ctx.x.detach()

            @staticmethod
            # 反向传播函数，在 order 列表中添加 "Reentrant"，并根据条件决定是否继续递归调用自身的反向传播
            def backward(ctx, x):
                order.append("Reentrant")
                if ctx.x < 0:
                    return x
                with torch.enable_grad():
                    Reentrant.apply(ctx.x).backward()
                return x

        # 使用 MyFunction 类的 apply 方法创建一个张量 a
        a = MyFunction.apply(torch.tensor(6.0, requires_grad=True))
        # 使用 Reentrant 类的 apply 方法创建一个张量 b
        b = Reentrant.apply(torch.tensor(9.0, requires_grad=True))
        # 计算张量 a 和 b 的乘积 v
        v = a * b
        # 对 v 进行反向传播
        v.backward()
        
        # 断言：确保 order 列表长度为 11
        self.assertEqual(len(order), 11)
        # 断言：确保 order 列表中 "Reentrant" 出现的次数为 10
        self.assertEqual(order.count("Reentrant"), 10)
        # 断言：确保 order 列表中最后一个元素为 "MyFunction"
        self.assertEqual(order[-1], "MyFunction")

    # 一个标记为慢速测试的函数，用于测试模型的检查点功能
    @slowTest
    def test_checkpointing(self):
        # 定义输入数据的维度
        num_inp = 2000
        nz_inp = 10
        nz_out = 10
        nz_bottleneck = 1000

        # 定义一个简单的神经网络模型 module，用于复杂推理
        module = nn.Sequential(
            nn.Linear(nz_inp, nz_bottleneck),
            nn.ReLU(),
            nn.Linear(nz_bottleneck, nz_inp),
        )

        # 用于存储特征向量的列表 feat_combined
        feat_combined = []
        # 循环生成 num_inp 个随机数据，并进行检查点计算
        for r in range(num_inp):
            data_r = torch.empty(1, nz_inp)
            data_r.uniform_()
            data_r.requires_grad = True
            # 调用 checkpoint 函数对 module 和 data_r 进行检查点计算，并将结果添加到 feat_combined 中
            feat_r = checkpoint(module, data_r, use_reentrant=True)
            feat_combined.append(feat_r)

        # 计算 feat_combined 列表中所有张量的均值
        mean_combined = torch.stack(feat_combined).mean()
        # 对均值张量进行反向传播
        mean_combined.backward()
    # 定义测试方法，用于验证非可重入自动转换的检查点功能，接受设备类型作为参数
    def _test_checkpointing_non_reentrant_autocast(self, device_type):
        # 对于每种启用状态，分别进行测试
        for enabled in [True, False]:

            # 定义内部函数 foo，执行矩阵乘法操作，其中 torch.mm 在自动转换列表中
            def foo(x, y, z):
                # 使用自动转换精度运行的 torch.mm 操作
                x = torch.mm(x, y)
                y = torch.mm(x, z)
                z = torch.mm(z, z)
                # 如果未启用自动转换，则预期 dtype 为 torch.float32；否则为 torch.bfloat16
                expected_dtype = torch.float32 if not enabled else torch.bfloat16
                # 断言 z 的 dtype 符合预期
                self.assertEqual(expected_dtype, z.dtype)
                return z

            # 创建随机张量 x, y, z，需要梯度计算
            x = torch.randn(3, 3, requires_grad=True)
            y = torch.randn(3, 3, requires_grad=True)
            z = torch.randn(3, 3, requires_grad=True)
            # 如果设备类型为 "cuda"，将张量移动到 GPU 上
            if device_type == "cuda":
                x = x.cuda()
                y = y.cuda()
                z = z.cuda()

            # 使用 torch.autocast 设置自动转换环境，包括启用状态、设备类型和 dtype
            with torch.autocast(
                enabled=enabled, device_type=device_type, dtype=torch.bfloat16
            ):
                # 调用 checkpoint 函数执行 foo 函数，并传入参数 x, y, z，禁用可重入模式
                loss = checkpoint(foo, x, y, z, use_reentrant=False)
                # 对 loss 进行求和
                loss = loss.sum()

            # 没有保存和重新转换自动转换类型的情况下，autograd 将报错，提示 dtype 不匹配
            loss.backward()  # 触发重新计算以确保在 bfloat16 下运行

    # 测试非可重入自动转换检查点功能在 CPU 上的表现
    def test_checkpointing_non_reentrant_autocast_cpu(self):
        """
        Test that autocast args such as the dtype are preserved during non-reentrant
        checkpoint recomputation on CPU.
        """
        self._test_checkpointing_non_reentrant_autocast(device_type="cpu")

    # 跳过条件：如果 CUDA 不可用或不支持 bf16，则跳过测试
    @unittest.skipIf(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        "Test requires CUDA bf16 support",
    )
    # 测试非可重入自动转换检查点功能在 GPU 上的表现
    def test_checkpointing_non_reentrant_autocast_gpu(self):
        """
        Test that autocast args/kwargs such as the dtype are preserved during
        non-reentrant checkpoint recomputation on GPU.
        """
        self._test_checkpointing_non_reentrant_autocast(device_type="cuda")

    # 跳过条件：如果 CUDA 不可用，则跳过测试
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    @slowTest
    # 定义一个名为 test_checkpointing_without_reentrant_memory_savings 的测试方法，测试模型在不同设置下的内存使用情况
    def test_checkpointing_without_reentrant_memory_savings(self):
        # 定义一个名为 MyModel 的内部类，继承自 nn.Module，用于创建包含多个层的神经网络模型
        class MyModel(nn.Module):
            # 初始化方法，设置模型的参数和层列表
            def __init__(self, n, use_checkpoint, use_reentrant):
                super().__init__()
                self.n = n
                self.use_checkpoint = use_checkpoint
                self.use_reentrant = use_reentrant
                self.layers = nn.ModuleList()
                # 根据 n 的值，循环创建包含三个线性层的序列模块，并添加到层列表中
                for i in range(self.n):
                    layer = nn.Sequential(
                        nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 256)
                    )
                    self.layers.append(layer)
                # 预先分配梯度，使增加的内存使用主要是由激活值引起的
                # 初始化每个线性层的权重和偏置的梯度为与其相同形状的全一张量
                for layer in self.layers:
                    for lin in layer:
                        lin.weight.grad = torch.ones_like(lin.weight)
                        lin.bias.grad = torch.ones_like(lin.bias)

            # 前向传播方法，根据 use_checkpoint 的设置选择是否使用 checkpointing
            def forward(self, x):
                for i in range(self.n):
                    if not self.use_checkpoint:
                        # 若不使用 checkpointing，则直接传递数据到当前层
                        x = self.layers[i](x)
                    else:
                        # 若使用 checkpointing，则对当前层进行 checkpointing 操作
                        x = checkpoint(
                            self.layers[i], x, use_reentrant=self.use_reentrant
                        )

                return x

        # 创建三个不同配置的 MyModel 实例，并将它们放到 GPU 上进行计算
        model_no_checkpoint = MyModel(
            8, use_checkpoint=False, use_reentrant=False
        ).cuda()
        model_reentrant_checkpoint = MyModel(
            8, use_checkpoint=True, use_reentrant=True
        ).cuda()
        model_no_reentrant_checkpoint = MyModel(
            8, use_checkpoint=True, use_reentrant=False
        ).cuda()

        # 创建一个 100x256 大小的随机张量 x，并设置 requires_grad=True，表示在计算梯度时保留计算图
        x = torch.randn(100, 256, requires_grad=True, device="cuda")

        # 重置 GPU 的内存统计信息，开始测试 model_no_checkpoint
        torch.cuda.reset_peak_memory_stats()
        # 使用 model_no_checkpoint 进行前向传播，并计算损失值
        loss = model_no_checkpoint(x.clone()).sum()
        # 反向传播求解梯度
        loss.backward()
        # 记录当前 GPU 上的最大内存使用量
        mem_no_checkpoint = torch.cuda.max_memory_allocated()

        # 重置 GPU 的内存统计信息，开始测试 model_reentrant_checkpoint
        torch.cuda.reset_peak_memory_stats()
        # 使用 model_reentrant_checkpoint 进行前向传播，并计算损失值
        loss = model_reentrant_checkpoint(x.clone()).sum()
        # 反向传播求解梯度
        loss.backward()
        # 记录当前 GPU 上的最大内存使用量
        mem_reentrant_checkpoint = torch.cuda.max_memory_allocated()

        # 重置 GPU 的内存统计信息，开始测试 model_no_reentrant_checkpoint
        torch.cuda.reset_peak_memory_stats()
        # 使用 model_no_reentrant_checkpoint 进行前向传播，并计算损失值
        loss = model_no_reentrant_checkpoint(x.clone()).sum()
        # 反向传播求解梯度
        loss.backward()
        # 记录当前 GPU 上的最大内存使用量
        mem_no_reentrant_checkpoint = torch.cuda.max_memory_allocated()

        # 使用断言确保在有重入检查点时内存使用量小于无检查点时的内存使用量
        self.assertTrue(mem_reentrant_checkpoint < mem_no_checkpoint)
        # 使用断言确保在无重入检查点时内存使用量小于无检查点时的内存使用量
        self.assertTrue(mem_no_reentrant_checkpoint < mem_no_checkpoint)
    # 定义一个测试方法，用于验证不可重入的自定义函数检查点正常工作
    def test_checkpointing_without_reentrant_custom_function_works(self):
        # 错误消息字符串，用于断言异常时的比对
        msg = "Unpack is being triggered for a tensor that was already unpacked once"

        # 定义一个自定义的 PyTorch 函数类 MyFunc，继承自 torch.autograd.Function
        class MyFunc(torch.autograd.Function):
            @staticmethod
            # 前向传播方法，接收输入参数 x, y, z
            def forward(ctx, x, y, z):
                # 计算中间变量 w
                w = x * y * z
                # 计算输出 out
                out = w + w
                # 保存需要在反向传播时用到的张量到上下文对象 ctx 中
                ctx.save_for_backward(x, y, z, w, out)
                return out

            @staticmethod
            # 反向传播方法，接收梯度 grad_out
            def backward(ctx, grad_out):
                # 从上下文对象 ctx 中恢复保存的张量
                x, y, z, w, out = ctx.saved_tensors
                # 断言异常，如果再次访问已保存的张量，会抛出 RuntimeError 异常
                with self.assertRaisesRegex(RuntimeError, msg):
                    x_2, y_2, z_2, w_2, out_2 = ctx.saved_tensors
                # 返回需要计算梯度的张量，这里返回 x, y, z
                return x, y, z

        # 创建三个张量 x, y, z，要求计算梯度
        x = torch.tensor(1.0, requires_grad=True)
        y = torch.tensor(2.0, requires_grad=True)
        z = torch.tensor(3.0, requires_grad=True)

        # 定义一个函数 foo，对输入的 x, y, z 进行一系列操作，并应用自定义函数 MyFunc
        def foo(x, y, z):
            x = x * y * z
            y = y * y * z
            z = z * z
            out = MyFunc.apply(x, y, z)
            return out

        # 使用检查点运行函数 foo，并返回结果 out，不可重入
        out = checkpoint(foo, x, y, z, use_reentrant=False)
        # 对结果 out 求和并进行反向传播
        out.sum().backward()

    # 定义测试方法，验证不可重入检查点与上下文函数的交互
    def test_checkpointing_without_reentrant_with_context_fn(self):
        # 定义一个自定义的 TorchDispatchMode 类，用于记录操作符调用
        class VerboseTorchDispatchMode(TorchDispatchMode):
            def __init__(self):
                self.operators = []

            # 自定义 torch dispatch 方法，记录操作符名并执行
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                self.operators.append(func.__name__)
                return func(*args, **kwargs)

        # 创建一个张量 x，要求计算梯度
        x = torch.tensor(1.0, requires_grad=True)
        # 实例化自定义的 verbose_mode 对象
        verbose_mode = VerboseTorchDispatchMode()

        # 定义上下文函数 context_fn，返回 verbose_mode 和空上下文
        def context_fn():
            return verbose_mode, contextlib.nullcontext()

        # 使用检查点运行 lambda 函数 x.exp()，不可重入，并传入上下文函数 context_fn
        out = checkpoint(
            lambda x: x.exp(), x, use_reentrant=False, context_fn=context_fn
        )
        # 断言 verbose_mode 记录了 "exp.default" 操作符的调用
        self.assertEqual(verbose_mode.operators, ["exp.default"])

        # 清空 verbose_mode 记录的操作符列表
        verbose_mode.operators = []

        # 重新定义上下文函数 context_fn，返回空上下文和 verbose_mode
        def context_fn():
            return contextlib.nullcontext(), verbose_mode

        # 使用检查点运行 lambda 函数 x.exp()，不可重入，并传入上下文函数 context_fn
        out = checkpoint(
            lambda x: x.exp(), x, use_reentrant=False, context_fn=context_fn
        )
        # 对结果 out 进行反向传播
        out.backward()
        # 断言 verbose_mode 记录了 "exp.default" 和两次 "detach.default" 操作符的调用
        self.assertEqual(
            verbose_mode.operators, ["exp.default", "detach.default", "detach.default"]
        )

        # 使用检查点运行 lambda 函数 x.sin()，期望抛出异常并捕获
        with self.assertRaisesRegex(
            Exception, "only supported when use_reentrant=False"
        ):
            out = checkpoint(
                lambda x: x.sin(), x, use_reentrant=True, context_fn=context_fn
            )
    # 定义测试函数，检查是否在未显式传递 use_reentrant 参数时发出警告
    def test_checkpoint_warns_if_use_reentrant_not_passed_explcitly(self):
        # 创建一个具有梯度要求的随机张量
        a = torch.randn(1, requires_grad=True)

        # 显式传递参数不应该发出警告
        self.assertNotWarn(lambda: checkpoint(lambda x: x, a, use_reentrant=False))

        # 未显式传递参数应该发出警告
        with self.assertWarnsOnceRegex(
            UserWarning, ".*the use_reentrant parameter should be passed explicitly.*"
        ):
            checkpoint(lambda x: x, a)

    # 定义测试函数，检查在未显式传递 use_reentrant 参数时是否发出警告（顺序执行模块）
    def test_checkpoint_sequential_warns_if_use_reentrant_not_passed_explcitly(self):
        # 创建一个具有梯度要求的随机张量
        a = torch.randn(3, requires_grad=True)
        # 创建模块列表
        modules_list = [
            torch.nn.Linear(3, 3),
            torch.nn.Linear(3, 3),
            torch.nn.Linear(3, 3),
        ]

        # 显式传递参数不应该发出警告
        self.assertNotWarn(
            lambda: checkpoint_sequential(modules_list, 3, a, use_reentrant=False)
        )

        # 未显式传递参数应该发出警告
        with self.assertWarnsOnceRegex(
            UserWarning, ".*the use_reentrant parameter should be passed explicitly.*"
        ):
            checkpoint_sequential(modules_list, 3, a)

    # 定义测试函数，检查在不重新计算的情况下两次访问保存的张量是否正常工作
    def test_access_saved_tensor_twice_without_recomputation_works(self):
        # 定义计数器，用于记录函数调用次数
        count = [0]

        def foo(a):
            count[0] += 1
            b = a * a
            c = a * b
            d = torch.exp(a)
            return d

        # 创建一个具有梯度要求的随机张量
        a = torch.randn(5, requires_grad=True)
        # 使用 checkpoint 函数计算结果并保存
        d = checkpoint(foo, a, use_reentrant=False)
        # 断言函数 foo 只被调用了一次
        self.assertEqual(count[0], 1)
        
        # 重新计算的变量仅在特定的反向传播调用中保持持久性
        # 如果在反向传播外部访问 _saved_result，将触发重新计算。
        # 然后，重新计算的结果会立即清除。
        d.grad_fn._saved_result
        self.assertEqual(count[0], 2)
        
        # 第二次访问将触发另一个重新计算
        d.grad_fn._saved_result
        self.assertEqual(count[0], 3)
        
        # 反向传播清除了保存的变量
        d.sum().backward()
        self.assertEqual(count[0], 4)
        
        # 现在访问将引发错误
        with self.assertRaisesRegex(
            RuntimeError,
            "or directly access saved tensors after they have already been freed",
        ):
            d.grad_fn._saved_result

    @slowTest
    @parametrize("input_requires_grad", [True, False])
    # 定义一个测试函数，用于测试不可重入自动求导下的检查点功能
    def test_checkpointing_without_reentrant(self, input_requires_grad):
        """
        Basic test for checkpoint without reentrant autograd.
        """
        # 设置输入的数量和各种维度大小
        num_inp = 2000  # 输入数量
        nz_inp = 10      # 输入维度
        nz_out = 10      # 输出维度
        nz_bottleneck = 1000  # 瓶颈层维度

        # 创建一个小型的代理网络，用于进行复杂的推理
        module = nn.Sequential(
            nn.Linear(nz_inp, nz_bottleneck),  # 线性层，输入维度到瓶颈层维度
            nn.ReLU(),                         # ReLU 激活函数
            nn.Linear(nz_bottleneck, nz_inp),  # 线性层，瓶颈层维度到输出维度
        )

        # 用于测试激活检查点和不可重入的模块容器
        class MyModule(nn.Module):
            def __init__(self, mod):
                super().__init__()
                self.module = mod

            def forward(self, data):
                return self.module(data)

        module = MyModule(mod=module)

        # 深拷贝模块，用于后续对比梯度
        module_copy = deepcopy(module)

        # 初始化空列表，用于存储特征向量
        feat_combined = []
        feat_combined_no_checkpoint = []

        # 循环处理每个输入
        for r in range(num_inp):
            data_r = torch.empty(1, nz_inp)
            data_r.uniform_()
            data_r.requires_grad = input_requires_grad
            data_r_copy = data_r.clone()

            # 使用检查点函数计算特征向量，不可重入
            feat_r = checkpoint(module, data=data_r, use_reentrant=False)
            feat_combined.append(feat_r)

            # 直接计算特征向量，用于对比
            feat_r_no_checkpoint = module_copy(data_r)
            feat_combined_no_checkpoint.append(feat_r_no_checkpoint)

        # 计算特征向量的平均值作为联合推理的代理
        mean_combined = torch.stack(feat_combined).mean()
        mean_combined.backward()

        # 计算无检查点情况下的特征向量平均值
        mean_combined_no_checkpoint = torch.stack(feat_combined_no_checkpoint).mean()
        mean_combined_no_checkpoint.backward()

        # 对比两个模块的梯度是否相等
        for checkpoint_param, param in zip(
            module.parameters(), module_copy.parameters()
        ):
            self.assertEqual(checkpoint_param.grad, param.grad)

    # 测试检查点在错误时的有效重置
    def test_checkpoint_valid_reset_on_error(self):
        # 创建一个张量 a，要求其梯度
        a = torch.randn(2, 2, requires_grad=True)

        # 在抛出特定异常时进行断言
        with self.assertRaisesRegex(
            Exception, "torch.utils.checkpoint is incompatible"
        ):
            # 使用检查点函数计算指数，并对其求和
            b = checkpoint(torch.exp, a, use_reentrant=True).sum()
            torch.autograd.grad(b, (a,))

        # 继续使用检查点计算指数并求和，然后进行反向传播
        c = checkpoint(torch.exp, a, use_reentrant=True).sum()
        c.backward()

    # 参数化测试，包括检查点是否可重入
    @parametrize("use_reentrant", [True, False])
    def test_checkpointing_without_reentrant_detached_tensor(self, use_reentrant):
        # 定义一个不带梯度的模块类
        class NoGradModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)  # 定义一个2x2的线性层，无偏置
                self.lin2 = nn.Linear(2, 2, bias=False)    # 定义另一个2x2的线性层，无偏置

            def forward(self, x):
                with torch.no_grad():  # 在前向传播中关闭梯度计算
                    return self.lin2(self.linear(x))  # 返回经过两个线性层的计算结果

        module = NoGradModule()  # 创建一个无梯度计算的模块实例

        # 根据是否使用可重入上下文来设置错误断言上下文
        err_ctx = (
            self.assertRaisesRegex(
                RuntimeError, "none of output has requires_grad=True"
            )
            if use_reentrant
            else contextlib.nullcontext()
        )

        a = torch.randn(2, 2, requires_grad=True)  # 创建一个2x2的张量，要求梯度计算
        for _ in range(3):
            with err_ctx:
                # 使用checkpoint函数对模块和输入a进行检查点操作
                out = checkpoint(module, a, use_reentrant=use_reentrant)
                # 使得损失函数需要梯度计算，否则会遇到梯度函数缺失的问题
                out += a
                out.sum().backward()  # 对输出进行求和并反向传播梯度

    def test_checkpointing_without_reentrant_correct_grad(self):
        """
        Verifies that correct gradients are calculated for checkpoint
        without reentrant autograd, for both backward() and autograd.grad().
        """
        a = torch.randn(2, 2, requires_grad=True)  # 创建一个2x2的张量，要求梯度计算

        b = torch.exp(a).sum()  # 对张量a的指数求和
        b.backward()  # 对b进行反向传播
        b_grad = a.grad  # 获取反向传播得到的梯度

        a.grad = None  # 清空张量a的梯度
        c = checkpoint(torch.exp, a, use_reentrant=False).sum()  # 使用checkpoint函数对torch.exp函数和张量a进行检查点操作
        c.backward()  # 对c进行反向传播
        c_grad = a.grad  # 获取反向传播得到的梯度

        a.grad = None  # 再次清空张量a的梯度
        d = checkpoint(torch.exp, a, use_reentrant=False).sum()  # 使用checkpoint函数对torch.exp函数和张量a进行检查点操作
        (d_grad,) = torch.autograd.grad(d, (a,))  # 使用torch.autograd.grad计算d关于a的梯度

        self.assertEqual(b_grad, c_grad)  # 断言b_grad与c_grad相等
        self.assertEqual(b_grad, d_grad)  # 断言b_grad与d_grad相等

    # PYTORCH_TEST_WITH_DYNAMO=1 test fails on CI but can't repro locally
    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/127115")
    def test_checkpointing_without_reentrant_dataparallel(self):
        """
        Verifies gradient correctness when checkpoint without reentrant autograd
        is used in conjunction with DataParallel.
        """

        class LinearModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)  # 定义一个2x2的线性层，无偏置

            def forward(self, inp):
                return self.linear(inp)  # 返回经过线性层计算的结果

        a = torch.randn(2, 2, requires_grad=True)  # 创建一个2x2的张量，要求梯度计算
        if torch.cuda.is_available():  # 如果CUDA可用，则将张量a移动到GPU上
            a = a.cuda()

        model = LinearModule()  # 创建线性模块实例
        if torch.cuda.is_available():  # 如果CUDA可用，则将模型移动到GPU上
            model = model.cuda()

        b = deepcopy(model)(a).sum()  # 深复制模型并对输入a进行求和操作
        b.backward()  # 对b进行反向传播
        b_grad = a.grad  # 获取反向传播得到的梯度

        a.grad = None  # 清空张量a的梯度

        module = torch.nn.DataParallel(deepcopy(model))  # 使用DataParallel包装深复制的模型
        c = checkpoint(module, a, use_reentrant=False).sum()  # 使用checkpoint函数对模块和输入a进行检查点操作
        c.backward()  # 对c进行反向传播
        c_grad = a.grad  # 获取反向传播得到的梯度

        self.assertEqual(b_grad, c_grad)  # 断言b_grad与c_grad相等
    def test_checkpointing_without_reentrant_parameter_used_in_an_out(self):
        """
        Ensures that gradient hooks are only called once per tensor.
        """
        # 创建一个形状为 (10, 10) 的随机张量 w，并要求计算梯度
        w = torch.randn(10, 10, requires_grad=True)
        # 初始化计数器 count
        count = 0

        # 定义一个梯度 hook 函数，用于统计调用次数
        def hook(grad):
            nonlocal count
            count += 1

        # 将 hook 函数注册到张量 w 上
        w.register_hook(hook)
        # 创建一个形状为 (10, 10) 的随机张量 x，并要求计算梯度
        x = torch.rand(10, 10, requires_grad=True)
        # 计算 h，这里使用了 w（在 checkpoint 外部使用）
        h = w * x  
        # 在 checkpoint 内部使用 w，调用 checkpoint 函数
        out = checkpoint(
            lambda x: w * x, h, use_reentrant=False
        )

        # 对结果张量 out 的所有元素求和，并进行反向传播
        out.sum().backward()
        # 断言 hook 函数只被调用了一次
        self.assertEqual(count, 1)

    # https://github.com/pytorch/pytorch/issues/127115
    @xfailIfTorchDynamo
    def test_checkpointing_without_reentrant_arbitrary_input_output(self):
        """
        Ensures checkpointing without reentrant autograd works with functions
        with arbitrary input/output structures.
        """

        # 定义一个简单的神经网络模型
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(5, 5, bias=False)

            def forward(self, dict_input):
                # 从输入字典中获取张量 tensor
                tensor = dict_input["tensor"]
                # 将 tensor 传递给网络层 self.layer，并将结果包装在字典中返回
                return {"result": self.layer(tensor)}

        # 创建一个没有使用 checkpoint 的模型实例
        model_no_checkpoint = MyModel()
        # 创建一个深拷贝自 model_no_checkpoint 的模型实例
        model_checkpoint_without_reentrant = deepcopy(model_no_checkpoint)

        # 创建一个输入字典 inp，包含一个形状为 (5, 5) 的随机张量 tensor
        inp = {"tensor": torch.randn(5, 5)}

        # 使用没有使用 checkpoint 的模型计算结果 out_no_checkpoint
        out_no_checkpoint = model_no_checkpoint(inp)["result"].sum()

        # 使用 checkpoint 函数计算结果 out_checkpoint，禁止重入 autograd
        out_checkpoint = checkpoint(
            model_checkpoint_without_reentrant, inp, use_reentrant=False
        )["result"].sum()

        # 断言使用 checkpoint 和不使用 checkpoint 的结果一致
        self.assertEqual(out_checkpoint, out_no_checkpoint)

        # 对不使用 checkpoint 的结果进行反向传播
        out_no_checkpoint.backward()
        # 对使用 checkpoint 的结果进行反向传播
        out_checkpoint.backward()

        # 检查模型参数的梯度是否相等
        for param, checkpoint_param in zip(
            model_no_checkpoint.parameters(),
            model_checkpoint_without_reentrant.parameters(),
        ):
            self.assertEqual(param.grad, checkpoint_param.grad)

    def test_callback_adds_callback(self):
        # 初始化一个计数器列表，用于记录回调函数被调用的次数
        called = [0]

        # 定义最终回调函数，用于增加计数器的值
        def callback_final():
            called[0] += 1

        # 定义添加回调函数，用于增加计数器的值，并将最终回调函数添加到执行引擎的队列中
        def callback_adds_callback():
            called[0] += 1
            Variable._execution_engine.queue_callback(callback_final)

        # 定义一个自定义的 PyTorch 函数类 MyFunc
        class MyFunc(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, grad):
                # 在反向传播中添加回调函数 callback_adds_callback 到执行引擎的队列中
                Variable._execution_engine.queue_callback(callback_adds_callback)
                return grad

        # 创建一个形状为 (3, 3) 的随机张量 a，并要求计算梯度
        a = torch.rand((3, 3), requires_grad=True)
        # 使用 MyFunc 类应用 a，得到 b
        b = MyFunc.apply(a)
        # 对张量 b 所有元素求和，并进行反向传播
        b.sum().backward()

        # 断言回调函数被调用了两次
        self.assertEqual(called[0], 2)

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_callback_propagates_errors_from_device_thread(self):
        # 定义一个会抛出运行时错误的回调函数
        def callback():
            raise RuntimeError("blah")

        # 定义一个带有回调函数的钩子函数
        def hook_with_callback(*args):
            torch.autograd.Variable._execution_engine.queue_callback(callback)

        # 创建一个张量，指定其在 CUDA 设备上，并注册一个带回调函数的钩子
        t = torch.tensor([1.0, 2.0], requires_grad=True, device=torch.device("cuda"))
        t.register_hook(hook_with_callback)
        # 计算张量的平方
        output = t**2
        # 计算输出的和作为损失
        loss = output.sum()

        # 断言损失反向传播时会抛出特定的运行时错误信息
        with self.assertRaisesRegex(RuntimeError, "blah"):
            loss.backward()

    def _test_reentrant_with_callbacks(self, install_callbacks_in_depths):
        # 初始化计数器字典，记录内部和外部回调次数
        counter = {}
        counter["inner"] = 0
        counter["outer"] = 0

        # 定义增加内部计数器的函数
        def inc_inner_counter():
            counter["inner"] += 1

        # 定义增加外部计数器的函数
        def inc_outer_counter():
            counter["outer"] += 1

        # 定义一个自定义函数 MyFunc，继承自 torch.autograd.Function
        class MyFunc(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, input):
                # 如果指定深度中包含 1，则添加一个内部回调
                if 1 in install_callbacks_in_depths:
                    Variable._execution_engine.queue_callback(inc_inner_counter)

                return input

        # 定义一个可重入函数 MyReentrantFunc，继承自 torch.autograd.Function
        class MyReentrantFunc(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, input):
                # 如果指定深度中包含 0，则添加一个外部回调
                if 0 in install_callbacks_in_depths:
                    Variable._execution_engine.queue_callback(inc_outer_counter)
                # 可重入的反向传播调用
                tmp_inp = input.detach().requires_grad_()
                with torch.enable_grad():
                    tmp_out = (MyFunc.apply(tmp_inp)).sum()
                tmp_out.backward()
                return input

        # 创建一个随机张量，并应用 MyReentrantFunc
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = MyReentrantFunc.apply(t1)
        t3 = t2.sum()
        # 执行反向传播
        torch.autograd.backward([t3])

        return counter

    def test_reentrant_with_callbacks_depth_0(self):
        # 验证回调函数仅被调用一次
        ret = self._test_reentrant_with_callbacks([0])
        self.assertEqual(1, ret["outer"])
        self.assertEqual(0, ret["inner"])

    def test_reentrant_with_callbacks_depth_1(self):
        # 验证回调函数仅被调用一次
        ret = self._test_reentrant_with_callbacks([1])
        self.assertEqual(0, ret["outer"])
        self.assertEqual(1, ret["inner"])

    def test_reentrant_with_callbacks_both_depths(self):
        # 验证回调函数被调用两次
        ret = self._test_reentrant_with_callbacks([0, 1])
        self.assertEqual(1, ret["outer"])
        self.assertEqual(1, ret["inner"])
    # 定义测试函数，用于测试带有叶子变量钩子的重入性
    def test_reentrant_with_leaf_variable_hook(self):
        # 初始化句柄为 None
        handle = None
        # 创建一个形状为 (10,) 的张量，并标记需要计算梯度
        param = torch.rand(10, requires_grad=True)

        # 定义一个函数，用于在梯度上添加惩罚
        def add_gradient_penalty_to_grad(grad):
            # 移除先前注册的钩子
            handle.remove()
            # 保存旧的参数梯度
            old_param_grad = grad
            # 清空当前参数的梯度
            param.grad = None
            # 使用 torch.enable_grad() 启用梯度计算上下文管理器
            with torch.enable_grad():
                # 分离并标记梯度，创建新的参数张量，也进行标记
                g = grad.detach().requires_grad_()
                new_param = param.detach().requires_grad_()
                # 计算新的输出，并进行反向传播
                out = ((g * 2) + new_param).sum()
                out.backward()
            # 计算最终结果的梯度
            res = g.grad + grad
            # 恢复旧的参数梯度
            param.grad = old_param_grad
            return res

        # 注册钩子到参数 param 上
        handle = param.register_hook(add_gradient_penalty_to_grad)
        # 前向传播
        tmp = param * param
        loss = tmp.sum()
        # 计算损失的梯度
        loss.backward()

    # 定义测试函数，用于测试带有非叶子变量钩子的重入性
    def test_reentrant_with_non_leaf_variable_hook(self):
        # 初始化句柄为 None
        handle = None
        # 创建一个形状为 (10,) 的张量，并标记需要计算梯度
        param = torch.rand(10, requires_grad=True)

        # 定义一个函数，用于手动增加梯度
        def manual_increase_gradient(grad):
            # 移除先前注册的钩子
            handle.remove()
            # 使用 torch.enable_grad() 启用梯度计算上下文管理器
            with torch.enable_grad():
                # 分离并标记梯度
                g = grad.detach().requires_grad_()
                # 计算新的输出，并进行反向传播
                out = ((g * 2) + 5).sum()
                out.backward()
            # 计算最终结果的梯度
            res = g.grad + grad
            return res

        # 前向传播
        tmp = param * param
        # 注册钩子到 tmp 张量上
        handle = tmp.register_hook(manual_increase_gradient)
        loss = tmp.sum()
        # 计算损失的梯度
        loss.backward()
        # 断言参数 param 的梯度为 6 * param
        self.assertEqual(param.grad, 6 * param)

    # 定义测试函数，用于测试无法创建保存张量的情况
    def test_cant_create_saved_tensors(self):
        # 使用上下文管理器断言抛出 RuntimeError，并包含特定的错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            "Trying to create a SavedTensor object from Python is forbidden",
        ):
            # 调用 torch.autograd.SavedTensor() 抛出异常
            torch.autograd.SavedTensor()
    # 定义一个测试函数，用于测试自定义函数中的保存张量
    def test_custom_function_saved_tensors(self):
        # 定义一个内部函数，返回一个自定义的 Function 类
        def getFn(save=True):
            # 定义一个自定义的 Function 类 MyFn
            class MyFn(Function):
                # 前向传播函数，保存输入张量到上下文中并返回输入张量本身
                @staticmethod
                def forward(ctx, x):
                    if save:
                        ctx.save_for_backward(x, None)  # 如果需要保存，将张量 x 和 None 保存在上下文中
                    return x

                # 反向传播函数，直接返回梯度 g
                @staticmethod
                def backward(ctx, g):
                    return g

            return MyFn

        # 创建一个随机张量 a，标记为需要计算梯度
        a = torch.randn(5, requires_grad=True)

        # 调用 getFn 函数，生成一个 MyFn 实例并应用于张量 a
        y = getFn(True).apply(a)

        # 断言保存的张量应为 (a, None)
        self.assertEqual((a, None), y.grad_fn.saved_tensors)

        # 获取保存的原始张量列表
        saved = y.grad_fn._raw_saved_tensors

        # 断言第一个保存的张量是 SavedTensor 类型
        self.assertIsInstance(saved[0], torch._C._autograd.SavedTensor)

        # 第二个保存的张量预期是 SavedTensor 类型，但不能直接验证其内容为 None
        self.assertIsInstance(saved[1], torch._C._autograd.SavedTensor)

        # 当用户调用 register_hooks 时，捕获 None 异常
        with self.assertRaisesRegex(RuntimeError, "None is forbidden"):
            saved[1].register_hooks(lambda x: x, lambda x: x)

        # 验证不兼容的函数参数会引发 TypeError
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            saved[0].register_hooks(lambda x: x)
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            saved[0].register_hooks(1, 1)

        # 注册钩子函数成功后，再次注册会引发 RuntimeError
        saved[0].register_hooks(lambda x: x, lambda x: x)
        with self.assertRaisesRegex(RuntimeError, "already been set"):
            saved[0].register_hooks(lambda x: x, lambda x: x)

        # 对 y 求和并进行反向传播
        y.sum().backward()

        # 删除 saved 变量后，访问保存的张量列表会引发 RuntimeError
        del saved
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            y.grad_fn._raw_saved_tensors
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            y.grad_fn.saved_tensors

        # 创建另一个 MyFn 实例，并应用于张量 a，验证保存张量为空
        y = getFn(False).apply(a)
        self.assertEqual(y.grad_fn.saved_tensors, ())  # 断言保存的张量为空元组
        self.assertEqual(y.grad_fn._raw_saved_tensors, ())  # 断言保存的原始张量也为空元组
    # 定义测试方法，验证 autograd 图中的节点是否为 Node 类的实例
    def test_autograd_node_isinstance(self):
        # Node 是通过代码生成的节点的虚拟基类。这意味着 isinstance 和 issubclass 被重写，但 mro 未更改。
        Node = torch.autograd.graph.Node

        # 创建一个需要梯度的随机张量 a
        a = torch.rand(3, 3, requires_grad=True)
        # 对张量 a 进行指数运算得到张量 b
        b = a.exp()

        # 某些节点通过代码生成向 torch._C._function 模块注册
        self.assertIsInstance(b.grad_fn, Node)
        self.assertTrue(issubclass(type(b.grad_fn), Node))
        self.assertTrue(Node not in type(b.grad_fn).mro())

        # 其它节点通过手动向 torch._C._function 模块注册
        self.assertNotIsInstance(torch._C._functions.AccumulateGrad, Node)
        self.assertTrue(issubclass(torch._C._functions.AccumulateGrad, Node))
        self.assertIsInstance(b.grad_fn.next_functions[0][0], Node)
        self.assertTrue(issubclass(torch._C._functions.DelayedError, Node))

        # 特殊情况
        self.assertNotIsInstance(None, Node)
        self.assertNotIsInstance(1, Node)
        self.assertNotIsInstance(Node, Node)
        self.assertTrue(issubclass(Node, Node))

        # 自定义函数的情况
        self.assertTrue(issubclass(torch.autograd.function.BackwardCFunction, Node))

        # 定义一个自定义的 torch.autograd.Function 类
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # ctx 应当是 Node 类的实例
                self.assertIsInstance(ctx, Node)
                return x

            @staticmethod
            def backward(ctx, x):
                # ctx 应当是 Node 类的实例
                self.assertIsInstance(ctx, Node)
                return x

        # 使用自定义的 Func 类对张量 a 进行操作
        out = Func.apply(a)
        self.assertIsInstance(out.grad_fn, Node)
        self.assertTrue(issubclass(type(out.grad_fn), Node))
        self.assertTrue(Node not in type(out.grad_fn).mro())
        # 对输出张量 out 进行求和并反向传播梯度
        out.sum().backward()
    # 定义一个测试方法，验证在不需要梯度的情况下原地操作不会触发梯度计算
    def test_inplace_not_requires_grad(self):
        # 定义一个继承自torch.autograd.Function的自定义函数类MyFn
        class MyFn(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，返回输入的视图
            def forward(ctx, inp):
                return inp.view_as(inp)

            @staticmethod
            # 反向传播函数，直接返回梯度
            def backward(ctx, grad):
                return grad

        # 创建一个不需要梯度的随机张量a
        a = torch.rand(1, 2)

        # 创建一个需要梯度的随机张量b
        b = torch.rand(1, requires_grad=True)

        # 使用自定义函数MyFn对a进行视图变换，此操作预期会触发异常
        view_a = MyFn.apply(a)

        # 使用assertRaisesRegex断言捕获运行时异常，并验证异常信息包含指定文本
        with self.assertRaisesRegex(
            RuntimeError, "This view was created inside a custom Function"
        ):
            # 在view_a上执行原地加法操作，预期会触发异常
            view_a += b

        # 再次对a进行随机初始化，用MyFn.apply进行视图变换
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        view_a = MyFn.apply(a)

        # 使用assertRaisesRegex再次验证视图变换后的copy_操作会触发异常
        with self.assertRaisesRegex(
            RuntimeError, "This view was created inside a custom Function"
        ):
            # 在view_a上执行copy_操作，预期会触发异常
            view_a.copy_(b)

        # 对a进行随机初始化，取其解绑后的视图view_a
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        view_a = a.unbind()[0]

        # 使用assertRaisesRegex验证解绑后的视图view_a进行copy_操作会触发异常
        with self.assertRaisesRegex(
            RuntimeError,
            "This view is the output of a function that returns multiple views.",
        ):
            view_a.copy_(b)

        # 对a进行随机初始化，取其选择的视图进行copy_操作，验证正常情况下视图操作可成功执行
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        a.select(1, 0).copy_(b)

    # 定义测试自动求导简单视图Python代码的方法
    def test_autograd_simple_views_python(self):
        # 调用内部方法_do_test_autograd_simple_views_python进行测试，使用双精度数据类型
        self._do_test_autograd_simple_views_python(torch.double)
        # 调用内部方法_do_test_autograd_simple_views_python进行测试，使用复数双精度数据类型
        self._do_test_autograd_simple_views_python(torch.cdouble)

    # 定义测试自动求导打印张量的方法
    def test_autograd_print_tensor(self):
        # 创建一个requires_grad=True的全1张量a，并进行克隆操作得到a_clone
        a = torch.ones(1, requires_grad=True)
        a_clone = a.clone()

        # 使用assertEqual断言，验证张量a和a_clone的字符串表示与预期相符
        self.assertEqual(repr(a), "tensor([1.], requires_grad=True)")
        self.assertEqual(repr(a_clone), "tensor([1.], grad_fn=<CloneBackward0>)")

        # 在torch.no_grad环境中，创建b为a的切片并乘以2
        with torch.no_grad():
            b = a[:]
            b *= 2

        # 使用assertEqual断言，验证在no-grad环境中修改后的张量b的字符串表示与预期相符
        self.assertEqual(repr(b), "tensor([2.], grad_fn=<Invalid>)")

        # 定义一个自定义的torch.autograd.Function子类Func
        class Func(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，直接返回输入
            def forward(ctx, x):
                return x

            @staticmethod
            # 反向传播函数，直接返回输入
            def backward(ctx, x):
                return x

        # 对a应用Func函数，得到张量c
        c = Func.apply(a)

        # 使用assertEqual断言，验证张量c的字符串表示与预期相符
        self.assertEqual(repr(c), "tensor([2.], grad_fn=<FuncBackward>)")
    # 定义一个测试方法，用于验证自动求导中对视图的操作
    def test_autograd_inplace_view_of_view(self):
        # 创建一个包含两个元素的零张量
        x = torch.zeros(2)
        # 在 torch.no_grad() 上下文中，创建 x 的视图 y
        with torch.no_grad():
            y = x.view(2)
        # 将 y 标记为需要梯度
        y.requires_grad_(True)
        # 创建 y 的另一个视图 z
        z = y.view(2)
        # 在运行时期间，验证是否会出现 RuntimeError，表明在 no_grad 块内部对视图的视图进行了操作
        with self.assertRaisesRegex(
            RuntimeError, "a view of a view .* is being .* inside the no_grad block"
        ):
            z /= 2

        # 重新初始化 x
        x = torch.zeros(2)
        # 在 torch.inference_mode() 上下文中，创建 x 的视图 y
        with torch.inference_mode():
            y = x.view(2)
        # 将 y 标记为需要梯度
        y.requires_grad_(True)
        # 创建 y 的另一个视图 z
        z = y.view(2)
        # 在运行时期间，验证是否会出现 RuntimeError，表明在 inference_mode 块内部对视图的视图进行了操作
        with self.assertRaisesRegex(
            RuntimeError, "a view of a view .* is being .* inside the inference_mode"
        ):
            z /= 2

    # TODO 这不是正确的行为 -
    # 参考 https://github.com/pytorch/pytorch/issues/49825#issuecomment-794466627
    # 定义一个测试方法，用于验证跨数据类型的自动求导中的视图操作
    def test_autograd_inplace_views_cross_dtype(self):
        # 此测试用例确保对此行为的任何更改都能被检测到，并非默默无声
        # 下面的 TODO 标记了意外行为的位置
        # 创建一个需要梯度的随机复数张量 a_orig
        a_orig = torch.rand(3, 3, requires_grad=True, dtype=torch.complex64)
        # 克隆张量 a_orig
        a = a_orig.clone()
        # 使用 torch.view_as_real() 创建 b 作为 a 的实部视图
        b = torch.view_as_real(a)
        # 转置 b 的维度 0 和 1
        b = b.transpose(0, 1)
        # 对 b 中的元素加 1
        b += 1
        # 对张量 a_orig 进行反向传播，使用 torch.arange() 生成的梯度
        b.backward(torch.arange(0, 18, dtype=torch.float).view(3, 3, 2))
        # 获取非原位操作的梯度
        non_inplace_grad = a_orig.grad

        # 重新初始化 a_orig
        a_orig = torch.rand(3, 3, requires_grad=True, dtype=torch.complex64)
        # 克隆张量 a_orig
        a = a_orig.clone()
        # 使用 torch.view_as_real() 创建 b 作为 a 的实部视图
        b = torch.view_as_real(a)
        # 原位转置 b 的维度 0 和 1
        b.transpose_(0, 1)
        # 对 b 中的元素加 1
        b += 1
        # 对张量 a_orig 进行反向传播，使用 torch.arange() 生成的梯度
        b.backward(torch.arange(0, 18, dtype=torch.float).view(3, 3, 2))
        # 获取原位操作的梯度
        inplace_grad = a_orig.grad

        # TODO: 这是一个 bug!
        # 一旦修复，应删除 transpose 操作：
        # self.assertEqual(non_inplace_grad, inplace_grad)
        # 断言非原位操作梯度的转置与原位操作梯度相等
        self.assertEqual(non_inplace_grad.T, inplace_grad)
    def test_autograd_multiple_views_python(self):
        # 测试多视图情况下的自动求导行为，特别是对于原地操作的影响检查

        # 用于记录反向传播函数被调用的次数
        bw_called = [0]

        class ComplexView(Function):
            @staticmethod
            def forward(ctx, a, idx):
                # 从张量 a 中选择指定位置的元素形成视图
                res = a.narrow(0, idx, 1)
                # 从张量 a 中选择指定位置的元素形成视图
                res = a.select(0, idx)
                # 保存用于反向传播的张量 a 到上下文对象
                ctx.save_for_backward(a)
                ctx.idx = idx
                return res

            @staticmethod
            def backward(ctx, grad):
                # 增加反向传播函数调用计数
                bw_called[0] += 1
                # 从上下文中恢复保存的张量 a
                (a,) = ctx.saved_tensors
                # 创建与 a 形状相同的零张量
                res = torch.zeros_like(a)
                # 将 grad 复制到 res 的指定位置上
                res.select(0, ctx.idx).copy_(grad)
                return res, None

        # 创建一个 requires_grad 为 True 的张量 a
        a = torch.ones(2, requires_grad=True)
        idx = 1

        # 初始化反向传播函数调用计数
        bw_called[0] = 0
        # 应用 ComplexView 自定义函数到张量 a 的克隆和指定的索引上
        out = ComplexView.apply(a.clone(), idx)
        # 对 out 求和并执行反向传播
        out.sum().backward()
        # 断言反向传播函数调用次数为 1
        self.assertTrue(bw_called[0] == 1)

        # 再次应用 ComplexView 自定义函数到张量 a 的克隆和指定的索引上
        out = ComplexView.apply(a.clone(), idx)
        # 使用断言确保执行原地操作会引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Output 0 of ComplexViewBackward is a view and is being modified inplace",
        ):
            out += 1
    # 测试自定义函数 mark_dirty_not_differentiable
    def test_custom_function_mark_dirty_not_differentiable(self):
        # 定义获取自定义函数的方法，根据传入的 jvp_err 参数决定行为
        def get_custom_fn(jvp_err):
            # 定义一个自动求导函数类 InplaceMul
            class InplaceMul(torch.autograd.Function):
                # 前向传播方法，对输入 x 执行就地乘法操作，并标记结果为脏数据
                @staticmethod
                def forward(ctx, x):
                    result = x.mul_(2)
                    ctx.mark_dirty(result)
                    return result

                # 反向传播方法，暂时为空实现
                @staticmethod
                def backward(ctx, grad_output):
                    pass

                # JVP（雅可比向量积）方法，根据 jvp_err 参数决定返回结果
                @staticmethod
                def jvp(ctx, x_t):
                    if jvp_err:
                        return x_t
                    else:
                        return x_t.mul_(2)

            return InplaceMul

        # 对 requires_grad 和 jvp_err 参数进行排列组合测试
        for requires_grad, jvp_err in product([True, False], repeat=2):
            # 获取当前 jvp_err 条件下的自定义函数类 InplaceMul
            InplaceMul = get_custom_fn(jvp_err)
            # 确保在标记为脏数据后张量仍然是原地操作后的结果
            z = torch.tensor(1.0, requires_grad=requires_grad)
            x = z.clone()
            y = InplaceMul.apply(x)
            self.assertTrue(x is y)
            self.assertEqual(x, z * 2)

            # 当 mark_dirty 设置时，确保 JVP 能够正确修改输入的梯度
            with fwAD.dual_level():
                x_tangent = torch.ones_like(x)
                x_dual = fwAD.make_dual(x, x_tangent)

                if jvp_err:
                    bad_mark_dirty_err = (
                        "jvp function must modify the corresponding gradient inplace"
                    )
                    # 当 jvp_err 为 True 时，确保引发 RuntimeError 异常
                    with self.assertRaisesRegex(RuntimeError, bad_mark_dirty_err):
                        InplaceMul.apply(x_dual)
                else:
                    # 否则，验证 JVP 正确修改了输入的梯度
                    out_dual = InplaceMul.apply(x_dual)
                    _, out_tangent = fwAD.unpack_dual(out_dual)
                    self.assertTrue(out_dual is x_dual)
                    self.assertTrue(out_tangent is x_tangent)

    # 测试用于复杂视图的命名张量
    def test_named_tensor_for_complex_views(self):
        # 定义张量的命名维度
        names = ["batch", "height", "width", "complex"]
        # 创建一个张量 z，形状为 (2, 1, 2, 2)，并设置 requires_grad=True
        z = torch.ones((2, 1, 2, 2), requires_grad=True)
        # 将张量 z 添加命名维度，并命名为 z_named
        z_named = z.refine_names(*names)
        # 将 z_named 转换为复数视图，并重新命名，去除最后一个维度的命名
        z_complex = torch.view_as_complex(z_named.rename(None)).refine_names(
            *names[:-1]
        )
        # 计算 z_complex 的和的绝对值，并进行反向传播
        z_complex.sum().abs().backward()
        # 期望的梯度张量，形状与 z_complex 相同
        expected = torch.ones_like(z_complex).rename(None)
        abs_1_1j = abs(1 + 1j)
        expected.fill_(complex(abs_1_1j / 2, abs_1_1j / 2))
        # 断言 z 的梯度与期望的梯度张量相等
        self.assertEqual(z.grad, torch.view_as_real(expected))
    def test_custom_function_return_view_in_nograd(self):
        # 定义一个自定义的 Torch 函数 Alias，继承自 Function 类
        class Alias(Function):
            @staticmethod
            # 前向传播函数的静态方法，返回输入张量的视图
            def forward(ctx, x):
                return x[:]

            @staticmethod
            # 反向传播函数的静态方法，返回梯度 gx
            def backward(ctx, gx):
                return gx

        # 创建一个 requires_grad=True 的随机输入张量
        inp = torch.rand(2, requires_grad=True)

        # 进入 no_grad 上下文
        with torch.no_grad():
            # 调用自定义的 Alias.apply 方法，返回输出张量 output
            output = Alias.apply(inp)

        with torch.no_grad():
            # 期望输出张量是输入张量的视图
            expected_output = inp[:]

        # 断言输出张量和期望输出张量的 requires_grad 属性相同
        self.assertEqual(output.requires_grad, expected_output.requires_grad)

        # 检查在视图上进行原地修改是否引发异常
        leaf_grad_err = (
            "A view was created in no_grad mode and is being modified inplace"
        )
        with self.assertRaisesRegex(RuntimeError, leaf_grad_err):
            output.zero_()

    def test_custom_function_preserve_torch_function_when_return_as_is(self):
        # 定义一个继承自 torch.Tensor 的自定义类 Custom
        class Custom(torch.Tensor):
            def __init__(self, data):
                super().__init__()
                self._data = data

            @classmethod
            # 自定义 __torch_function__ 方法，保留 Torch 函数
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                kwargs = {} if kwargs is None else kwargs
                # 将输入参数转换为内部数据格式
                args = tuple(a._data if isinstance(a, cls) else a for a in args)
                # 调用原始 Torch 函数
                out = func(*args, **kwargs)
                # 如果输出是张量，则包装为 Custom 类型
                if isinstance(out, torch.Tensor):
                    out = cls(out)
                return out

        # 定义一个继承自 torch.autograd.Function 的 Fn 类
        class Fn(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，返回输入
            def forward(ctx, input):
                return input

            @staticmethod
            # 后向传播函数，无操作
            def backward(ctx):
                pass

        # 创建一个 Custom 类型的张量 x
        x = Custom(torch.randn(2, 3))
        # 调用 Fn.apply 方法，返回 y
        y = Fn.apply(x)
        # 断言 y 是 Custom 类型的实例
        self.assertTrue(isinstance(y, Custom))

    def test_grad_mode_restored_reentrant(self):
        # 定义一个继承自 Function 的 MyFunction 类
        class MyFunction(Function):
            @staticmethod
            # 前向传播函数，返回输入的克隆
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            # 后向传播函数，返回计算的梯度
            def backward(ctx, go):
                # 获取当前梯度计算是否启用的原始状态
                original = torch._C.is_grad_enabled()
                # 在启用梯度计算的上下文中
                with torch.enable_grad():
                    self.assertTrue(torch._C.is_grad_enabled())
                    # 创建一个 requires_grad=True 的随机张量 foo
                    foo = torch.rand(go.size(), requires_grad=True)
                    # 计算 foo**3 对 foo 的梯度，使用 go 作为梯度输出
                    (grad,) = torch.autograd.grad(foo**3, foo, grad_outputs=go)
                    self.assertTrue(torch._C.is_grad_enabled())
                # 断言当前梯度计算状态是否与原始状态相同
                self.assertTrue(torch._C.is_grad_enabled() == original)
                return grad

        # 创建一个 requires_grad=True 的随机输入张量
        inp = torch.rand(3, requires_grad=True)

        # 案例1：原始梯度计算状态为 False
        MyFunction.apply(inp).sum().backward()
        # 案例2：原始梯度计算状态为 True
        MyFunction.apply(inp).sum().backward(create_graph=True)
    # 定义一个测试函数，用于测试自定义的幂函数功能
    def test_power_function(self):
        # 创建一个包含三个零张量的张量a
        a = torch.tensor([0.0, 0.0, 0.0])
        # 创建一个包含三个元素的张量b，并设置requires_grad为True，以便计算梯度
        b = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        # 计算张量a的b次幂的和
        c = torch.sum(a**b)
        # 对c进行反向传播，计算b的梯度
        c.backward()
        # 断言b的梯度为负无穷、零、零的张量
        self.assertEqual(b.grad, torch.tensor([-inf, 0.0, 0.0]))

        # 初始化一个变量s为0
        s = 0
        # 重新定义张量b，并设置requires_grad为True
        b = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        # 计算s的b次幂的和（实际上是计算0的b次幂的和，结果应该为0）
        c = torch.sum(s**b)
        # 对c进行反向传播，计算b的梯度
        c.backward()
        # 断言b的梯度为负无穷、零、零的张量
        self.assertEqual(b.grad, torch.tensor([-inf, 0.0, 0.0]))

    # 定义一个测试自定义函数出错处理的函数
    def test_custom_function_error(self):
        # 定义一个错误的前向传播函数BadFw
        class BadFw(Function):
            @staticmethod
            def backward(ctx, foo):
                return foo

        # 定义一个错误的后向传播函数BadBw
        class BadBw(Function):
            @staticmethod
            def forward(ctx, foo):
                return foo.clone()

        # 定义一个同时包含前向和后向传播的错误实现BadBw2
        class BadBw2(Function):
            @staticmethod
            def forward(ctx, foo):
                return foo.clone()

            @staticmethod
            def backward(ctx, foo):
                return foo

            @staticmethod
            def vjp(ctx, foo):
                return foo

        # 定义一个错误的JVP函数BadJvp
        class BadJvp(Function):
            @staticmethod
            def forward(ctx, foo):
                return foo.clone()

        # 生成一个随机张量inp，用于测试
        inp = torch.rand(1, requires_grad=True)
        # 断言在使用BadFw时抛出NotImplementedError异常，要求实现前向传播
        with self.assertRaisesRegex(NotImplementedError, "must implement the forward"):
            BadFw.apply(inp)

        # 断言在使用BadBw时抛出RuntimeError异常，要求实现前向或后向传播
        with self.assertRaisesRegex(RuntimeError, "must implement either the backward"):
            BadBw.apply(inp).sum().backward()

        # 断言在使用BadBw2时抛出RuntimeError异常，同时实现了前向和后向传播
        with self.assertRaisesRegex(
            RuntimeError, "Implementing both 'backward' and 'vjp'"
        ):
            BadBw2.apply(inp).sum().backward()

        # 断言在使用BadJvp时抛出RuntimeError异常，要求实现JVP函数
        with self.assertRaisesRegex(RuntimeError, "must implement the jvp function"):
            with fwAD.dual_level():
                d = fwAD.make_dual(inp, torch.rand_like(inp))
                res = BadJvp.apply(d)
    def test_custom_function_forward_mode_view_checks(self):
        # 定义一个字典，将不同的标志映射到对应的错误消息
        flag_to_error = {
            "ok": None,
            "not_a_view": "jvp is not returning a view",
            "not_a_view_of_inp": "jvp is not returning a view of the given",
            "not_a_view_of_inp_base": "jvp is not returning a view of the same base",
        }

        # 定义一个继承自Function的类ViewFn，用于实现自定义的前向传播、反向传播和Jacobian向量积计算
        class ViewFn(Function):
            @staticmethod
            def forward(ctx, foo, flag):
                # 在上下文对象中存储标志和张量尺寸信息
                ctx.flag = flag
                ctx.size = foo.size()
                # 返回张量的一个视图，从索引0开始，长度为2
                return foo.narrow(0, 0, 2)

            @staticmethod
            def vjp(ctx, gO):
                # 根据上下文对象中的尺寸信息，创建与gO相同尺寸的零张量gI
                gI = gO.new_zeros(ctx.size)
                # 将gO复制到gI的索引0开始，长度为2的部分
                gI.narrow(0, 0, 2).copy_(gO)
                return gI, None

            @staticmethod
            def jvp(ctx, gI, _):
                # 返回gI的一个视图，从索引0开始，长度为2
                res = gI.narrow(0, 0, 2)
                if ctx.flag != "ok":
                    # 如果标志不是"ok"，则克隆res，打破视图关系
                    res = res.clone()
                if ctx.flag in ["not_a_view_of_inp", "not_a_view_of_inp_base"]:
                    # 如果标志表明结果应该是一个视图，但是是错误的视图
                    res = res.view_as(res)
                return res

        # 创建一个4x4的双精度浮点型张量inp，并声明其需要梯度
        inp = torch.rand(4, 4, dtype=torch.double, requires_grad=True)

        # 遍历标志到错误消息的映射字典
        for flag, msg in flag_to_error.items():

            # 定义一个测试函数test_fn，根据不同的标志调用ViewFn的apply方法
            def test_fn(inp):
                if flag == "not_a_view_of_inp_base":
                    # 如果标志表明不是inp的基本视图，则将inp视图化为其本身的视图
                    inp = inp.view_as(inp)
                return ViewFn.apply(inp, flag)

            if msg is None:
                # 如果错误消息为空，则使用gradcheck检查前向传播的梯度
                gradcheck(test_fn, inp, check_forward_ad=True)
            else:
                # 否则，使用assertRaisesRegex检查是否抛出特定的运行时错误消息
                with self.assertRaisesRegex(RuntimeError, msg):
                    gradcheck(test_fn, inp, check_forward_ad=True)

    def test_custom_function_forward_mode_inplace_checks(self):
        # 定义一个继承自Function的类InplaceFn，用于实现自定义的原位（inplace）操作的前向传播、反向传播和Jacobian向量积计算
        class InplaceFn(Function):
            @staticmethod
            def forward(ctx, foo, flag):
                # 标记foo为脏张量，记录标志
                ctx.mark_dirty(foo)
                ctx.flag = flag
                # 将foo乘以2，并返回结果
                foo.mul_(2)
                return foo

            @staticmethod
            def vjp(ctx, gO):
                # 返回2倍于gO的值作为梯度
                return 2 * gO, None

            @staticmethod
            def jvp(ctx, gI, _):
                if ctx.flag:
                    # 如果标志为真，则不原位地修改gI
                    return 2 * gI
                else:
                    # 否则，将gI原位地乘以2并返回结果
                    gI.mul_(2)
                    return gI

        # 创建一个4x4的双精度浮点型张量inp，并声明其需要梯度
        inp = torch.rand(4, 4, dtype=torch.double, requires_grad=True)

        # 定义一个测试函数test_fn，根据标志调用InplaceFn的apply方法
        def test_fn(inp, flag):
            # 克隆inp，以避免对原始张量的影响，并应用InplaceFn
            inp = inp.clone()
            return InplaceFn.apply(inp, flag)

        # 使用gradcheck检查不同标志下的前向传播梯度
        gradcheck(test_fn, (inp, False), check_forward_ad=True)

        # 使用assertRaisesRegex检查特定运行时错误消息是否被抛出
        with self.assertRaisesRegex(
            RuntimeError,
            "inplace custom Function is not modifying the forward mode gradients inplace",
        ):
            gradcheck(test_fn, (inp, True), check_forward_ad=True)
    # 定义一个测试方法，用于测试自定义函数的前向模式下错误的计算公式
    def test_custom_function_forward_mode_wrong_formula(self):
        # 定义一个继承自 torch.autograd.Function 的自定义函数类 UserFn
        class UserFn(Function):
            # 前向传播函数，计算 foo 的两倍，并记录 should_fail 参数到 ctx
            @staticmethod
            def forward(ctx, foo, should_fail):
                ctx.should_fail = should_fail
                return foo * 2

            # 反向传播的 vjp 方法，返回输入梯度的两倍和 None
            @staticmethod
            def vjp(ctx, gO):
                return 2 * gO, None

            # 反向传播的 jvp 方法，根据 should_fail 决定返回不同的梯度值
            @staticmethod
            def jvp(ctx, gI, _):
                if ctx.should_fail:
                    # 如果 should_fail 为真，则返回错误的梯度公式
                    return 3 * gI
                else:
                    # 如果 should_fail 为假，则返回正常的梯度公式
                    return 2 * gI

        # 生成一个随机张量 inp，指定为双精度类型，并需要计算梯度
        inp = torch.rand(10, dtype=torch.double, requires_grad=True)
        # 对自定义函数 UserFn 的 forward 方法进行梯度检查，验证正向自动微分是否正确
        gradcheck(UserFn.apply, (inp, False), check_forward_ad=True)

        # 使用断言验证下面的代码块是否会引发 RuntimeError，且提示信息包含指定文本
        with self.assertRaisesRegex(
            RuntimeError, "Jacobian computed with forward mode mismatch for output 0"
        ):
            # 对自定义函数 UserFn 的 forward 方法再次进行梯度检查，此时传入 should_fail=True
            gradcheck(UserFn.apply, (inp, True), check_forward_ad=True)

    # 定义一个测试方法，用于测试自定义函数的前向模式中非张量参数在张量参数之前的情况
    def test_custom_function_forward_mode_non_tensor_before_tensor_args(self):
        # 定义一个继承自 torch.autograd.Function 的自定义函数类 MyFn
        class MyFn(torch.autograd.Function):
            # 前向传播函数，计算 x 的两倍加上 y 的三倍，并返回结果
            @staticmethod
            def forward(ctx, nt, x, nt2, y):
                return x * 2 + y * 3

            # 前向传播的 jvp 方法，验证 nt 和 nt2 参数是否为 None，然后返回 x_t 的两倍加上 y_t 的三倍
            @staticmethod
            def jvp(ctx, nt, x_t, nt2, y_t):
                self.assertIsNone(nt)
                self.assertIsNone(nt2)
                return x_t * 2 + y_t * 3

        # 创建三个双精度类型的张量 x、t、y，分别赋值为 1.0
        x = torch.tensor(1.0, dtype=torch.double)
        t = torch.tensor(1.0, dtype=torch.double)
        y = torch.tensor(1.0, dtype=torch.double)

        # 使用 fwAD.dual_level() 上下文管理器进行双模式自动微分的设置
        with fwAD.dual_level():
            # 使用 fwAD.make_dual() 将 x 和 y 转换为双模式张量，分别传入 MyFn 的 apply 方法
            dual_x = fwAD.make_dual(x, t)
            MyFn.apply(1, dual_x, 1, y)

        # 对自定义函数 MyFn 的 forward 方法进行梯度检查，验证正向自动微分是否正确
        gradcheck(
            MyFn.apply,
            (1, x.requires_grad_(True), 1, y.requires_grad_(True)),
            check_forward_ad=True,
            check_backward_ad=False,
            check_batched_grad=False,
        )
    def test_custom_function_forward_mode_forward_is_no_op(self):
        error_regex = (
            "A custom Function's forward is returning a view \\(or an input as-is\\)"
        )

        return_lambdas = {
            # 如果在 forward 方法中返回输入本身，会被视为执行 self.view_as(self) 操作。
            # 如果 jvp 返回 x.view_as(x)，这是允许的。
            "view_as": lambda x: x.view_as(x),
            # 预期这会引发错误
            "self": lambda x: x,
            # 预期这会引发相同的错误
            "mul_by_2": lambda x: x * 2,
        }

        for k, fn in return_lambdas.items():

            # 定义一个自定义函数 MyFn，继承自 torch.autograd.Function
            class MyFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, y):
                    # forward 方法返回 x + y 和 x
                    return x + y, x

                @staticmethod
                def vjp(ctx, gO1, gO2):
                    # vjp 方法返回 gO1 + gO2 和 gO1
                    return gO1 + gO2, gO1

                @staticmethod
                def jvp(ctx, x_t, y_t):
                    # jvp 方法返回 x_t + y_t 和 fn(x_t) 的结果
                    return x_t + y_t, fn(x_t)

            # 创建需要计算梯度的张量
            a = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
            t = torch.tensor(1.0, dtype=torch.double)
            b = torch.tensor(1.0, dtype=torch.double, requires_grad=True)

            c = torch.tensor(1.0, dtype=torch.double)
            t2 = torch.tensor(1.0, dtype=torch.double)
            d = torch.tensor(1.0, dtype=torch.double)

            # 使用 fwAD.dual_level() 进行上下文管理
            with fwAD.dual_level():
                # 使用 fwAD.make_dual() 创建双重张量 a_dual 和 c_dual
                a_dual = fwAD.make_dual(a, t)
                c_dual = fwAD.make_dual(c, t2)

                if k == "view_as":
                    # 如果 k 是 "view_as"，调用 MyFn.apply() 方法，并断言其输出的 tangent 与 t 相同
                    _, out2 = MyFn.apply(a_dual, b)
                    self.assertTrue(fwAD.unpack_dual(out2).tangent._base is t)

                    _, out2 = MyFn.apply(c_dual, d)
                    self.assertTrue(fwAD.unpack_dual(out2).tangent._base is t2)
                else:
                    # 如果 k 不是 "view_as"，预期会抛出 RuntimeError，并匹配特定的错误信息 error_regex
                    with self.assertRaisesRegex(RuntimeError, error_regex):
                        MyFn.apply(a_dual, b)

                    with self.assertRaisesRegex(RuntimeError, error_regex):
                        MyFn.apply(c_dual, d)

            # 根据 k 的值进行不同的梯度检查操作
            if k == "view_as":
                gradcheck(MyFn.apply, (a, c), check_forward_ad=True)
            else:
                with self.assertRaisesRegex(RuntimeError, error_regex):
                    gradcheck(MyFn.apply, (a, c), check_forward_ad=True)
    # 定义一个自定义的 Torch 自动求导函数 Func
    def test_custom_function_save_for_forward(self):
        # 定义一个内部类 Func，继承自 torch.autograd.Function
        class Func(torch.autograd.Function):
            # 定义静态方法 forward，用于前向传播
            @staticmethod
            def forward(ctx, x: torch.Tensor, y: torch.Tensor, z: int):
                # 保存 x 和 y 到上下文中，以备反向传播使用
                ctx.save_for_backward(x, y)
                # 保存 x 和 y 用于前向传播
                ctx.save_for_forward(x, y)
                # 将 z 存储在上下文中
                ctx.z = z
                # 计算 x 和 y 的乘积，并保存到上下文中
                ctx.prod = x * y
                # 返回 z 乘以 x 和 y 的乘积作为前向传播的结果
                return z * ctx.prod

            # 定义静态方法 jvp，用于计算 Jacobian-Vector Product
            @staticmethod
            def jvp(ctx, x_t, y_t, _):
                # 从上下文中获取保存的 x 和 y
                x_p, y_p = ctx.saved_tensors
                # 获取保存的 z
                z = ctx.z
                # 计算 JVP 并返回结果
                return z * (y_p * x_t + x_p * y_t)

            # 定义静态方法 vjp，用于计算 Vector-Jacobian Product
            @staticmethod
            def vjp(ctx, grad_out):
                # 从上下文中获取保存的 x 和 y
                x, y = ctx.saved_tensors
                # 获取保存的 z
                z = ctx.z
                # 计算 VJP 并返回结果
                return z * grad_out * y, z * grad_out * x, None

        # 定义几个 Torch 张量 a, b，并设置其是否需要梯度
        a = torch.tensor(1.0, requires_grad=True, dtype=torch.double)
        t = torch.tensor(1.0, dtype=torch.double)
        b = torch.tensor(2.0, requires_grad=True, dtype=torch.double)
        c = 4

        # 进入 fwAD 的双重级别上下文管理
        with fwAD.dual_level():
            # 使用 fwAD.make_dual 创建双重张量 a_dual
            a_dual = fwAD.make_dual(a, t)
            # 调用 Func 类的 apply 方法进行前向传播计算
            out = Func.apply(a_dual, b, c)
            # 对计算结果进行反向传播
            out.backward()

        # 对 Func.apply 方法进行梯度检查
        gradcheck(Func.apply, (a, b, c), check_forward_ad=True)

        # 定义一个新的 Func 类，用于测试保存了反向传播但未保存前向传播的情况
        # 当保存了反向传播，但未保存前向传播时
        class Func(torch.autograd.Function):
            # 定义静态方法 forward，用于前向传播
            @staticmethod
            def forward(ctx, x: torch.Tensor):
                # 仅保存 x 到上下文中，用于反向传播
                ctx.save_for_backward(x)
                # 返回 x 的克隆
                return x.clone()

            # 定义静态方法 jvp，用于计算 Jacobian-Vector Product
            @staticmethod
            def jvp(ctx, x_t):
                # 断言保存的张量数量为 0
                self.assertEqual(len(ctx.saved_tensors), 0)
                # 返回 x_t
                return x_t

            # 定义静态方法 vjp，用于计算 Vector-Jacobian Product
            @staticmethod
            def vjp(ctx, grad_out):
                # 获取保存的张量 x
                (x,) = ctx.saved_tensors
                # 断言保存的张量数量为 1
                self.assertEqual(len(ctx.saved_tensors), 1)
                # 返回梯度 grad_out
                return grad_out

        # 进入 fwAD 的双重级别上下文管理
        with fwAD.dual_level():
            # 使用 fwAD.make_dual 创建双重张量 a_dual
            a_dual = fwAD.make_dual(a, t)
            # 调用 Func 类的 apply 方法进行前向传播计算
            out = Func.apply(a_dual)
            # 对计算结果进行反向传播
            out.backward()

        # 对 Func.apply 方法进行梯度检查
        gradcheck(Func.apply, (a,), check_forward_ad=True)
    def test_custom_function_forward_mode_non_differentiable(self):
        # 定义一个测试函数，用于测试自定义函数在非可微模式下的行为

        # 定义一个自定义的 Torch 自动求导函数
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                # 在前向传播中，复制 y，并标记为非可微
                out = y.clone()
                ctx.mark_non_differentiable(out)
                return x.clone(), out

            @staticmethod
            def jvp(ctx, x_tangent, y_tangent):
                # 定义 Jacobian-Vector Product (JVP)，这里对 y 的导数为 None
                return x_tangent, None

        x = torch.tensor(2.0)
        x_tangent = torch.tensor(1.0)
        y = torch.tensor(3.0)

        # 使用 fwAD 的 dual_level 上下文管理器
        with fwAD.dual_level():
            # 创建双重数 x_dual，其导数为 x_tangent
            x_dual = fwAD.make_dual(x, x_tangent)
            # 调用 Func 自定义函数的 forward 方法
            _, out2_dual = Func.apply(x_dual, y)
            # 断言 out2_dual 的导数为 None
            self.assertEqual(fwAD.unpack_dual(out2_dual).tangent, None)

        y = torch.tensor(3)

        # 定义另一个自定义 Torch 自动求导函数
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                # 在前向传播中，直接返回 x 和 y 的克隆
                return x.clone(), y.clone()

            @staticmethod
            def jvp(ctx, x_tangent, y_tangent):
                # 在 JVP 中，断言 y 的导数为 None
                self.assertIsNone(y_tangent)
                return x_tangent, None

        # 使用 fwAD 的 dual_level 上下文管理器
        with fwAD.dual_level():
            # 创建双重数 x_dual，其导数为 x_tangent
            x_dual = fwAD.make_dual(x, x_tangent)
            # 调用 Func 自定义函数的 forward 方法
            _, out2_dual = Func.apply(x_dual, y)
            # 断言 out2_dual 的导数为 None
            self.assertEqual(fwAD.unpack_dual(out2_dual).tangent, None)

        # 定义一个错误的自定义 Torch 自动求导函数
        class FuncWrong(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                # 在前向传播中，复制 y，并标记为非可微（错误示例）
                out = y.clone()
                ctx.mark_non_differentiable(out)
                return x.clone(), out

            @staticmethod
            def jvp(ctx, x_tangent, y_tangent):
                # 在 JVP 中，返回 x_tangent 的克隆，而不是 None
                return x_tangent, x_tangent.clone()

        # 使用 fwAD 的 dual_level 上下文管理器
        with fwAD.dual_level():
            # 创建双重数 x_dual，其导数为 x_tangent
            x_dual = fwAD.make_dual(x, x_tangent)
            # 断言 FuncWrong 应当抛出运行时错误
            with self.assertRaisesRegex(
                RuntimeError, "You should return None at that position instead"
            ):
                FuncWrong.apply(x_dual, y)

        # 定义一个返回非张量的自定义 Torch 自动求导函数
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 在前向传播中，返回 x 的克隆，一个对象，和 x 的克隆
                return x.clone(), object(), x.clone()

            @staticmethod
            def jvp(ctx, x_tangent):
                # 在 JVP 中，返回 x_tangent，None，和 x_tangent 的克隆
                return x_tangent, None, x_tangent

        # 使用 fwAD 的 dual_level 上下文管理器
        with fwAD.dual_level():
            # 创建双重数 x_dual，其导数为 x_tangent
            x_dual = fwAD.make_dual(x, x_tangent)
            # 调用 Func 自定义函数的 forward 方法
            out_dual, _, out2_dual = Func.apply(x_dual)
            # 断言 out_dual 和 out2_dual 的导数为 x_tangent
            self.assertEqual(fwAD.unpack_dual(out_dual).tangent, x_tangent)
            self.assertEqual(fwAD.unpack_dual(out2_dual).tangent, x_tangent)
    # 定义一个测试函数，用于测试自定义的本地函数（inplace操作）
    def test_custom_function_local_inplace(self):
        # 定义一个继承自torch.autograd.Function的自定义函数类MyFn
        class MyFn(torch.autograd.Function):
            # 前向传播方法，接受输入inp和是否原地操作inplace标志
            @staticmethod
            def forward(ctx, inp, inplace):
                # 克隆输入的部分视图（前三个元素）
                view = inp.clone()[:3]
                # 如果选择原地操作
                if inplace:
                    view += 2  # 将视图中的值加2
                return view  # 返回处理后的视图

            # 反向传播方法，接受梯度grad
            @staticmethod
            def backward(ctx, grad):
                return grad, None  # 返回梯度和None作为后向传播的导数

        # 创建一个需要梯度的10x10的随机张量
        base = torch.rand(10, requires_grad=True)

        # 使用自定义函数MyFn的apply方法，进行前向传播（不原地操作）
        foo = MyFn.apply(base, False)
        # 断言生成的foo的梯度函数的类名为"MyFnBackward"
        self.assertEqual(foo.grad_fn.__class__.__name__, "MyFnBackward")

        # 使用自定义函数MyFn的apply方法，进行前向传播（原地操作）
        foo = MyFn.apply(base, True)
        # 断言生成的foo的梯度函数的类名为"MyFnBackward"
        self.assertEqual(foo.grad_fn.__class__.__name__, "MyFnBackward")

    # 定义一个测试函数，用于测试自定义函数的生命周期
    def test_custom_function_cycle(self):
        # 定义一个继承自Function的自定义函数类MyFn
        class MyFn(Function):
            # 前向传播方法，接受输入x和元数据metadata
            @staticmethod
            def forward(ctx, x, metadata):
                # 克隆输入张量x
                x = x.clone()
                # 将元数据metadata保存在上下文ctx中
                ctx.meta = metadata
                # 将克隆的张量x保存在上下文ctx中，以备后向传播使用
                ctx.save_for_backward(x)
                return x  # 返回克隆的张量x

            # 反向传播方法，接受梯度gO
            @staticmethod
            def backward(ctx, gO):
                # 从上下文中恢复保存的张量x
                (x,) = ctx.saved_tensors
                # 断言张量x的值为3.14
                self.assertEqual(x, 3.14)
                # 断言元数据中的"foo"键对应的值为3.14
                self.assertEqual(ctx.meta["foo"], 3.14)
                # 返回梯度gO乘以张量x，以及None作为后向传播的导数
                return gO * x, None

        # 定义一个内部函数，用于获取引用
        def get_refs(with_backward):
            # 创建一个需要梯度的标量张量a，值为3.14
            a = torch.tensor(3.14, requires_grad=True)

            metadata = {}  # 创建一个空的元数据字典
            # 使用自定义函数MyFn的apply方法，进行前向传播
            out = MyFn.apply(a, metadata)

            # 将out作为"foo"键的值保存到元数据中
            metadata["foo"] = out

            if with_backward:
                # 如果with_backward为True，计算out的和的梯度
                out.sum().backward()
                # 断言张量a的梯度为a本身
                self.assertEqual(a.grad, a)

            # 返回out的弱引用
            return torch._C._WeakTensorRef(out)

        # 禁用垃圾回收上下文，获取不带后向传播的引用
        with disable_gc():
            ref = get_refs(False)
            # 断言引用未过期
            self.assertFalse(ref.expired())
        gc.collect()  # 手动触发垃圾回收
        # 断言引用已过期
        self.assertTrue(ref.expired())

        # 后向传播会清除保存的变量，但不会清除__dict__中的内容
        # 再次禁用垃圾回收上下文，获取带后向传播的引用
        with disable_gc():
            ref = get_refs(True)
            # 断言引用未过期
            self.assertFalse(ref.expired())
        gc.collect()  # 手动触发垃圾回收
        # 断言引用已过期
        self.assertTrue(ref.expired())
    def test_create_graph_and_full_backward_hook_cycle(self):
        # 定义测试函数，验证在使用 create_graph=True 时，FullBackwardHook 是否会导致梯度输出保存并创建循环
        #
        #   grad_output -> grad_output.grad_fn -> graph -> hook -> grad_output
        #

        class TestCls:
            # 用于创建弱引用的虚拟类
            pass

        def get_ref(input_requires_grad, nb_hooks):
            # 创建一个需要梯度的张量 t
            t = torch.randn(10, requires_grad=input_requires_grad)
            # 创建一个需要梯度的标量 a
            a = torch.tensor(1.0, requires_grad=True)

            class Test(nn.Module):
                def forward(self, x):
                    return x**2 * a**2

            mod = Test()

            # 注册指定数量的全局反向传播钩子
            for _ in range(nb_hooks):
                mod.register_full_backward_hook(lambda a, b, c: None)

            tmp = mod(t)

            # 将虚拟对象保存到图中，并获取其弱引用
            test = TestCls()
            ref = weakref.ref(test)
            tmp.grad_fn.metadata["a"] = test

            # 执行带有 create_graph=True 的反向传播，并捕获警告信息
            with set_warn_always_context(True):
                with warnings.catch_warnings(record=True) as w:
                    tmp.exp().sum().backward(create_graph=True)
                    self.assertTrue(len(w) == 1)
                    self.assertTrue(
                        "Using backward() with create_graph=True" in str(w[0].message)
                    )

            # 移除梯度和 create_graph=True 的循环依赖
            a.grad = None
            t.grad = None

            return ref

        # 循环测试不同数量的钩子和是否需要梯度的张量
        for nb_hooks in (1, 2, 3):
            for input_requires_grad in (True, False):
                ref_ = get_ref(
                    input_requires_grad=input_requires_grad,
                    nb_hooks=nb_hooks,
                )
                gc.collect()
                self.assertIsNone(ref_())
    def test_hook_closure_cycle(self, use_custom_function, use_tensor_hook):
        # This function tests a closure cycle involving hooks and grad_fn_b.
        # The cycle is hook -> closure -> grad_fn_b (python) -> grad_fn (cpp) -> hook (cpp)
        # -> dict -> hook
        #
        # This test ensures that grad_fn_b (python) only traverses the dictionary if it's
        # the only reference to the grad_fn_b (cpp) shared_ptr.
        #
        # See: https://github.com/pytorch/pytorch/issues/102174
        
        # Define a custom torch.autograd.Function
        class Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad):
                return grad

        # Placeholder class Test
        class Test:
            pass

        # Counter for hook invocations
        count = [0]

        # Define a scope function
        def scope():
            # Create a tensor with requires_grad=True
            a = torch.tensor(1.0, requires_grad=True)
            # Apply custom function or clone tensor based on flag
            if use_custom_function:
                b = Function.apply(a)
            else:
                b = a.clone()
            # Retrieve grad_fn of tensor b
            grad_fn_b = b.grad_fn
            # Create a Test object instance
            obj = Test()

            # Define hook function
            def hook(*args):
                # Ensure this hook's closure holds onto grad_fn_b
                # This creates a closure cycle with grad_fn_b
                # Also, keep a reference to a sentinel object 'obj'
                grad_fn_b
                obj
                count[0] += 1

            # Register hook depending on use_tensor_hook flag
            if use_tensor_hook:
                b.register_hook(hook)
            else:
                b.grad_fn.register_hook(hook)
            
            # Clone tensor b
            c = b.clone()
            # Create weak reference to object obj
            ref = weakref.ref(obj)
            return c, ref

        # Disable garbage collector temporarily
        with disable_gc():
            # Execute scope function and capture output and weak reference
            out, ref = scope()
            # Perform backward pass retaining graph
            out.backward(retain_graph=True)

            # Explicitly trigger garbage collection
            gc.collect()

            # Ensure the hook cycle persists after garbage collection
            out.backward(retain_graph=True)
            self.assertEqual(count[0], 2)

            # Check that the weak reference is still alive
            # due to the shared_ptr being > 1
            self.assertIsNotNone(ref())

            # Delete the rest of the graph and verify weak reference is dead
            del out
            gc.collect()
            self.assertIsNone(ref())
    # 定义一个测试方法，用于测试完整反向钩子和双向反向传播
    def test_full_backward_hook_double_backward(self):
        # 创建一个随机张量 x，并启用梯度跟踪
        x = torch.rand(1, requires_grad=True)
        # 根据 x 创建一个与其形状相同的随机张量 y
        y = torch.rand_like(x)

        # 创建一个均方误差损失函数对象
        func = torch.nn.MSELoss()
        # 计数器，用于计算完整反向钩子调用次数
        counter = [0]

        # 定义一个钩子函数，用于在完整反向传播过程中调用
        def hook(module, grad_input, grad_output):
            counter[0] += 1

        # 注册完整反向钩子到损失函数对象
        func.register_full_backward_hook(hook)

        # 计算损失函数的值
        f = func(x, y)

        # 计算损失函数关于 x 的梯度，并创建计算图
        (gradx_f,) = torch.autograd.grad(f, x, create_graph=True)
        # 断言完整反向钩子被调用一次
        self.assertEqual(counter[0], 1)
        # 计算 gradx_f 关于 x 的梯度，此时不应增加计数器
        _ = torch.autograd.grad(gradx_f, x)
        # 断言计数器仍为 1，未增加
        self.assertEqual(counter[0], 1)

    # 定义一个测试方法，用于测试输入缓冲区的累积效果
    def test_input_buffer_accum(self):
        # 创建一个随机张量 leaf，并启用梯度跟踪
        leaf = torch.rand(2, 2, requires_grad=True)

        # 创建一个操作，返回稀疏梯度的张量
        ind = torch.tensor([[0, 0]], dtype=torch.long)
        out2 = leaf.gather(0, ind, sparse_grad=True)

        # 创建一个操作，返回原始梯度的张量
        out1 = leaf.clone()

        # 创建与 out1 和 out2 形状相同的随机梯度张量
        grad_out1_original = torch.rand_like(out1)
        grad_out1 = grad_out1_original.clone()
        grad_out2 = torch.rand_like(out2)

        # 对 out1 和 out2 进行反向传播，传入相应的梯度
        torch.autograd.backward((out1, out2), (grad_out1, grad_out2))

        # 断言给定梯度张量未被原地修改
        self.assertEqual(grad_out1, grad_out1_original)

    # 定义一个测试方法，用于测试不必要的解包情况
    def test_no_unnecessary_unwrapping(self):
        # 创建一个具有随机值的张量 a，并启用梯度跟踪
        a = torch.randn(5, requires_grad=True)
        # 创建张量 a 的克隆副本 a_orig
        a_orig = a.detach().clone()
        # 计算张量 b 和 c，以及张量 d 的指数函数
        b = a * a
        c = a * b
        d = torch.exp(a)

        # 断言张量 a 是叶子节点
        self.assertIs(b.grad_fn._saved_self, a)
        self.assertIs(b.grad_fn._saved_other, a)
        self.assertIs(c.grad_fn._saved_self, a)

        # 断言张量 b 不是输出
        self.assertIs(c.grad_fn._saved_other, b)

        # 断言张量 d 是输出
        self.assertEqual(d.grad_fn._saved_result, d)
        self.assertIsNot(d.grad_fn._saved_result, d)

        # 对 c 的和进行反向传播
        c.sum().backward()

        # 断言在已释放之后，不应再访问 c 的 _saved_self
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            c.grad_fn._saved_self

        # 断言张量 a 保持不变
        self.assertEqual(a, a_orig)

    # 定义一个测试方法，用于测试保存的变量版本计数器
    def test_saved_variable_version_counter(self):
        # 创建一个具有随机值的张量 a，并启用梯度跟踪
        a = torch.rand(2, requires_grad=True)

        # 计算张量 b 的指数函数
        b = torch.exp(a)

        # 获取张量 b 的原始保存结果
        b_unpacked = b.grad_fn._saved_result
        # 断言张量 b 与其解包后的结果相等
        self.assertEqual(b, b_unpacked)
        # 断言张量 b 的版本与其解包后的版本相等
        self.assertEqual(b._version, b_unpacked._version)

        # 使用 torch.no_grad() 上下文，增加张量 b 的值
        with torch.no_grad():
            b += 1

        # 断言张量 b 与其解包后的结果仍相等
        self.assertEqual(b, b_unpacked)
        # 断言张量 b 的版本与其解包后的版本仍相等
        self.assertEqual(b._version, b_unpacked._version)
    # 测试函数：测试在保存变量时，对于 inplace 操作的 detach 行为的影响
    def test_saved_variable_saved_original_inplace_detach(self):
        # 创建一个需要梯度的张量，并克隆它
        a = torch.tensor(1.0, requires_grad=True).clone()
        # 对张量应用正弦函数，得到新的张量 b
        b = a.sin()
        # 对原始张量 a 进行 detach 操作，使其与计算图的连接断开
        a.detach_()
        # 使用断开连接后的张量 b 进行反向传播，预期会抛出 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError, "Trying to use a saved tensor that has been detached"
        ):
            b.backward()

        # 创建一个需要梯度的张量，并克隆它
        a = torch.tensor(1.0, requires_grad=True).clone()
        # 对张量应用指数函数，得到新的张量 b
        b = a.exp()
        # 对原始张量 a 进行 detach 操作，使其与计算图的连接断开
        a.detach_()
        # 使用断开连接后的张量 b 进行反向传播
        b.backward()

    # 测试函数：测试在保存变量时，使用自定义钩子函数对 SavedVariable 的打包/解包行为的影响
    def test_saved_variable_packing_unpacking_did_not_save_original_with_hooks(self):
        # 创建一个形状为 (5,) 的随机张量，需要梯度
        a = torch.randn(5, requires_grad=True)
        # 对张量应用指数函数，得到新的张量 y
        y = torch.exp(a)
        # 注册自定义钩子函数到 y.grad_fn 的 _raw_saved_result 上
        y.grad_fn._raw_saved_result.register_hooks(lambda x: x, lambda x: x)
        # 断言 y 和 y.grad_fn._saved_result 是相等的
        self.assertEqual(y, y.grad_fn._saved_result)
        # 断言 y.grad_fn 是 y.grad_fn._saved_result.grad_fn 的引用
        self.assertIs(y.grad_fn, y.grad_fn._saved_result.grad_fn)
        # 对 y 求和并进行反向传播
        y.sum().backward()
        # 断言梯度计算正确
        self.assertEqual(a.grad, y)
    def test_saved_variable_packing_unpacking_saved_original_with_default_hooks(self):
        # 测试默认钩子是否正确注册、使用和重置
        # saved_original / did_not_save_original 的区别对应于 `SavedVariable` 的 `save_original` 属性。
        # 参见:
        #  - test_saved_variable_packing_unpacking_saved_original_with_hooks

        # 定义一个打包函数 pack(x)，发出警告 "pack"
        def pack(x):
            warnings.warn("pack")
            return x

        # 使用 torch.autograd.graph.saved_tensors_hooks 方法注册钩子
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            # 创建一个需要梯度的张量 a
            a = torch.ones(5, requires_grad=True)

            # 捕获警告信息
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # 执行张量操作
                y = a * a
                # 应该会由于 a 被保存两次而产生两个警告
                self.assertEqual(len(w), 2)

        # 使用默认的钩子重置
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            # 创建一个需要梯度的随机张量 a
            a = torch.randn(5, requires_grad=True)
            y = a * a
            self.assertEqual(a, y.grad_fn._saved_self)
            self.assertEqual(a, y.grad_fn._saved_other)
            y.sum().backward()
            self.assertEqual(2 * a, a.grad)

        # 使用自定义的钩子
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x / 2):
            # 创建一个需要梯度的随机张量 a
            a = torch.randn(5, requires_grad=True)
            y = a * a
            self.assertEqual(a, y.grad_fn._saved_self)
            self.assertEqual(a, y.grad_fn._saved_other)
            y.sum().backward()
            self.assertEqual(2 * a, a.grad)

        # 使用自定义的钩子，但是保存了修改后的自身
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            # 创建一个需要梯度的随机张量 a
            a = torch.randn(5, requires_grad=True)
            y = a * a
            self.assertEqual(2 * a, y.grad_fn._saved_self)
            self.assertEqual(2 * a, y.grad_fn._saved_other)
            y.sum().backward()
            self.assertEqual(4 * a, a.grad)

        # 正确退出钩子
        a = torch.randn(5, requires_grad=True)
        y = a * a
        self.assertEqual(a, y.grad_fn._saved_self)
        self.assertEqual(a, y.grad_fn._saved_other)
        y.sum().backward()
        self.assertEqual(2 * a, a.grad)

    def test_saved_variable_packing_unpacking_did_not_save_original_with_default_hooks(
        self,
    ):
        # 参见 test_saved_variable_packing_unpacking_did_not_save_original_with_hooks

        # 使用自定义的钩子
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            # 创建一个需要梯度的随机张量 a
            a = torch.randn(5, requires_grad=True)
            y = torch.exp(a)
            self.assertEqual(y, y.grad_fn._saved_result)
            y.sum().backward()
            self.assertEqual(a.grad, y)

    def test_setting_default_saved_variable_hooks_twice_should_not_fail(self):
        # 使用默认的钩子设置两次不应该失败
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
                pass
    def test_setting_default_saved_variable_hooks_twice_should_use_inner(self):
        # 使用 torch.autograd.graph.saved_tensors_hooks 函数设置默认的 saved tensor hooks，
        # 使用两个 lambda 函数对输入的张量做处理（每个都乘以 3）
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 3 * x, lambda x: 3 * x):
            # 创建一个需要梯度的张量 b，并且形状为 (5,)
            b = torch.randn(5, requires_grad=True)
            # 在设置内部，再次使用 torch.autograd.graph.saved_tensors_hooks 函数
            # 使用两个 lambda 函数对输入的张量做处理（每个都乘以 5）
            with torch.autograd.graph.saved_tensors_hooks(
                lambda x: 5 * x, lambda x: 5 * x
            ):
                # 创建一个需要梯度的张量 a，并且形状为 (5,)
                a = torch.randn(5, requires_grad=True)
                # 计算 a * a
                y = a * a
            # 计算 b * b
            z = b * b
        # 对 y 和 z 求和并反向传播梯度
        y.sum().backward()
        z.sum().backward()
        # 断言 a 和 b 的梯度计算是否正确
        self.assertEqual(2 * 5 * 5 * a, a.grad)
        self.assertEqual(2 * 3 * 3 * b, b.grad)

    def test_disabling_saved_tensor_hooks(self):
        # 禁用 saved tensor hooks 并抛出自定义的 RuntimeError
        with torch.autograd.graph.disable_saved_tensors_hooks("error message"):
            with self.assertRaisesRegex(RuntimeError, "error message"):
                # 在禁用的情况下使用 saved tensor hooks 函数，应该会抛出错误
                with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
                    pass

        # 检查 saved tensor hooks 是否重新启用
        self.assertTrue(torch._C._autograd._saved_tensors_hooks_is_enabled())

        # 使用 saved tensor hooks 函数，并在其内部禁用 saved tensor hooks，并抛出错误
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            with self.assertRaisesRegex(RuntimeError, "error message"):
                with torch.autograd.graph.disable_saved_tensors_hooks("error message"):
                    pass

        # 检查 saved tensor hooks 是否重新启用
        self.assertTrue(torch._C._autograd._saved_tensors_hooks_is_enabled())

    def test_disabling_saved_tensor_hooks_nested(self):
        # 嵌套地禁用 saved tensor hooks，并检查错误的传播
        with torch.autograd.graph.disable_saved_tensors_hooks("outer"):
            with torch.autograd.graph.disable_saved_tensors_hooks("inner"):
                with self.assertRaisesRegex(RuntimeError, "inner"):
                    # 在内部禁用的情况下使用 saved tensor hooks 函数，应该会抛出错误
                    with torch.autograd.graph.saved_tensors_hooks(
                        lambda x: x, lambda x: x
                    ):
                        pass

            # 检查 saved tensor hooks 是否未启用
            self.assertFalse(torch._C._autograd._saved_tensors_hooks_is_enabled())

        # 检查 saved tensor hooks 是否重新启用
        self.assertTrue(torch._C._autograd._saved_tensors_hooks_is_enabled())

    def test_saved_tensor_hooks_custom_error_propagaation(self):
        # 定义一个自定义的异常类 CustomError
        class CustomError(Exception):
            pass

        # 定义一个在 pack_hook 中抛出 CustomError 异常的 saved tensor hooks 子类
        class error_on_pack_hook(torch.autograd.graph.saved_tensors_hooks):
            def __init__(self):
                def pack_hook(x):
                    raise CustomError("pack")

                super().__init__(pack_hook, lambda x: x)

        # 定义一个在 unpack_hook 中抛出 CustomError 异常的 saved tensor hooks 子类
        class error_on_unpack_hook(torch.autograd.graph.saved_tensors_hooks):
            def __init__(self):
                def unpack_hook(x):
                    raise CustomError("unpack")

                super().__init__(lambda x: x, unpack_hook)

        # 创建一个需要梯度的张量 a，并初始化其值为 1.0
        a = torch.tensor(1.0, requires_grad=True)

        # 使用 error_on_pack_hook 类来捕获 pack_hook 中的 CustomError 异常
        with error_on_pack_hook():
            with self.assertRaisesRegex(CustomError, "pack"):
                out = torch.sin(a)

        # 使用 error_on_unpack_hook 类来捕获 unpack_hook 中的 CustomError 异常
        with error_on_unpack_hook():
            out = torch.sin(a)
            with self.assertRaisesRegex(CustomError, "unpack"):
                out.backward()
    def test_saved_tensor_hooks_custom_function_intermediates(self):
        # 定义一个自定义的 Torch 自动求导函数 Func
        class Func(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，计算输入张量的指数，并保存中间结果
            def forward(ctx, x):
                intermediate = x.exp()  # 计算输入张量的指数
                ctx.save_for_backward(
                    intermediate.clone().detach_().requires_grad_(True)
                )  # 在上下文中保存中间结果的克隆，用于反向传播
                return x.exp()  # 返回输入张量的指数

            @staticmethod
            # 反向传播函数，根据保存的中间结果计算梯度
            def backward(ctx, grad_out):
                (intermediate,) = ctx.saved_tensors  # 获取保存的中间结果
                return grad_out * intermediate  # 返回梯度乘以中间结果作为反向传播梯度

        a = torch.tensor(1.0, requires_grad=True)  # 创建一个需要求导的张量

        # 使用 Torch 自动求导的 saved_tensors_hooks 上下文管理器
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            out = Func.apply(a)  # 调用自定义函数 Func 进行前向传播计算
        out.backward()  # 对输出张量进行反向传播

    def test_unpack_hooks_exec_count(self):
        def f(x, y):
            return x * y  # 简单的张量乘法运算

        pack_count = 0  # 计数器，用于记录 pack_hook 调用次数
        unpack_count = 0  # 计数器，用于记录 unpack_hook 调用次数

        def pack_hook(x):
            nonlocal pack_count  # 声明 pack_count 变量为非局部变量
            pack_count += 1  # 每次调用时增加 pack_count 计数
            return x

        # unpack hook 在编译时不应运行，因为我们在追踪前向传播过程
        def unpack_hook(x):
            nonlocal unpack_count  # 声明 unpack_count 变量为非局部变量
            unpack_count += 1  # 每次调用时增加 unpack_count 计数
            return x

        x = torch.ones(4, requires_grad=True)  # 创建一个全为1的张量，需要求导
        y = torch.ones(4, requires_grad=False)  # 创建一个全为1的张量，不需要求导
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            out_test = f(x, y)  # 执行张量乘法运算
            self.assertEqual(pack_count, 1)  # 断言 pack_hook 调用次数为1
            self.assertEqual(unpack_count, 0)  # 断言 unpack_hook 调用次数为0
            out_test.sum().backward()  # 对输出张量求和并进行反向传播
            self.assertEqual(pack_count, 1)  # 断言 pack_hook 调用次数为1
            self.assertEqual(unpack_count, 1)  # 断言 unpack_hook 调用次数为1

    def test_save_tensor_hook_version_counter_not_shared(self):
        # 定义一个自定义 Torch 自动求导函数 Test
        class Test(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，保存输入张量并返回其正弦值
            def forward(ctx, x):
                ctx.save_for_backward(x)  # 在上下文中保存输入张量
                return x.sin()  # 返回输入张量的正弦值

            @staticmethod
            # 反向传播函数，根据保存的输入张量计算梯度
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors  # 获取保存的输入张量
                before = a._version  # 获取当前张量版本号
                x.add_(1)  # 修改输入张量的值
                self.assertEqual(a._version, before)  # 断言版本号未变化
                return grad_output  # 返回梯度输出

        a = torch.tensor(1.0, requires_grad=True)  # 创建一个需要求导的张量
        a_replacement = a.clone()  # 克隆输入张量用于替换

        def pack_hook(x):
            return a_replacement  # 在 hook 中返回克隆的张量

        def unpack_hook(x):
            return x  # unpack hook 直接返回输入张量

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            b = Test.apply(a)  # 调用自定义函数 Test 进行前向传播计算

        b.backward()  # 对输出张量进行反向传播
    # 定义一个测试函数，用于测试在 CPU 上保存和检查点操作
    def test_save_on_cpu_and_checkpoint(self):
        # 创建一个形状为 (2, 2) 的张量 a，并标记为需要梯度计算
        a = torch.randn(2, 2, requires_grad=True)

        # 对张量 a 进行多次幂操作
        b = a.pow(2).pow(2).pow(2).pow(2)
        # 对结果张量 b 的所有元素求和并进行反向传播
        b.sum().backward()
        # 复制张量 a 的梯度
        b_grad = a.grad.clone()
        # 清零张量 a 的梯度
        a.grad.zero_()

        # 使用 save_on_cpu 上下文管理器
        with torch.autograd.graph.save_on_cpu():
            # 对张量 a 进行平方操作，并使用检查点对其进行优化
            h = a.pow(2)
            h = checkpoint(lambda x: x.pow(2).pow(2), h, use_reentrant=False)
            # 对结果张量 h 进行平方操作
            c = h.pow(2)
        # 对结果张量 c 的所有元素求和并进行反向传播
        c.sum().backward()
        # 复制张量 a 的梯度
        c_grad = a.grad.clone()
        # 清零张量 a 的梯度
        a.grad.zero_()

        # 定义一个函数 f，对输入张量进行一系列幂操作，并使用 save_on_cpu 上下文管理器
        def f(a):
            h = a.pow(2)
            with torch.autograd.graph.save_on_cpu():
                h = h.pow(2).pow(2)
            return h.pow(2)

        # 使用检查点优化函数 f，并禁用重入
        d = checkpoint(f, a, use_reentrant=False)
        # 对结果张量 d 的所有元素求和并进行反向传播
        d.sum().backward()
        # 复制张量 a 的梯度
        d_grad = a.grad.clone()

        # 断言三次反向传播的结果应该相等
        self.assertEqual(b_grad, c_grad)
        self.assertEqual(b_grad, d_grad)

    # 定义一个测试函数，测试在进行原位修改时是否会失败
    def test_pack_hook_with_inplace_modification_should_fail(self):
        # 创建一个形状为 (5,) 的张量 a，并标记为需要梯度计算
        a = torch.randn(5, requires_grad=True)

        # 定义一个函数 inc，对输入张量进行原位加一操作
        def inc(x):
            x += 1
            return x

        # 使用 saved_tensors_hooks 上下文管理器，并断言在原位修改时会触发 RuntimeError
        with torch.autograd.graph.saved_tensors_hooks(inc, lambda x: x):
            with self.assertRaisesRegex(
                RuntimeError,
                "A saved tensor pack hook is modifying its input in place.",
            ):
                y = torch.exp(a)

        # 再次对张量 a 进行指数操作，并断言注册原位修改钩子时会触发 RuntimeError
        y = torch.exp(a)
        with self.assertRaisesRegex(
            RuntimeError, "A saved tensor pack hook is modifying its input in place."
        ):
            y.grad_fn._raw_saved_result.register_hooks(inc, lambda x: x)

    # 定义一个测试函数，测试将变量保存到磁盘的操作
    def test_saving_variable_to_disk(self):
        # 使用临时目录 tmp_dir 进行测试
        with tempfile.TemporaryDirectory() as tmp_dir:
            
            # 定义一个函数 pack，将输入张量保存到临时目录中，并返回文件名
            def pack(x):
                name = os.path.join(tmp_dir, str(uuid.uuid4()))
                torch.save(x, name)
                return name

            # 定义一个函数 unpack，从指定文件名中加载张量
            def unpack(name):
                return torch.load(name)

            # 使用 saved_tensors_hooks 上下文管理器，并进行相关测试
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                # 创建一个形状为 (5,) 的张量 a，并标记为需要梯度计算
                a = torch.ones(5, requires_grad=True)
                # 对张量 a 进行乘法操作
                y = a * a
                # 断言张量 a 与其梯度函数的保存结果相等
                self.assertEqual(a, y.grad_fn._saved_self)

                # 对结果张量 y 的所有元素求和并进行反向传播
                y.sum().backward()
                # 断言张量 a 的梯度为其自身乘以 2
                self.assertEqual(2 * a, a.grad)
    # 定义一个测试函数，用于测试默认保存的变量钩子的双向传播
    def test_default_saved_variable_hooks_double_backward(self):
        # 使用torch.autograd.graph.saved_tensors_hooks函数设置保存的张量钩子
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            # 创建一个形状为(5,)的张量a，要求计算梯度
            a = torch.randn(5, requires_grad=True)
            # 计算张量a的立方
            y = a**3
            # 计算张量y的元素和
            s = torch.sum(y)
            # 对张量s相对于张量a进行梯度计算，并创建计算图
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            # 对计算得到的梯度g进行求和，并进行反向传播
            g.sum().backward()
            # 断言梯度计算的结果与期望值6 * a相等
            self.assertEqual(6 * a, a.grad)

        # 使用torch.autograd.graph.saved_tensors_hooks函数再次设置保存的张量钩子
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            # 创建一个形状为(5,)的张量a，要求计算梯度
            a = torch.randn(5, requires_grad=True)
            # 计算张量a的立方
            y = a**3
            # 计算张量y的元素和
            s = torch.sum(y)
        # 对张量s相对于张量a进行梯度计算，并创建计算图
        (g,) = torch.autograd.grad(s, (a,), create_graph=True)
        # 对计算得到的梯度g进行求和，并进行反向传播
        g.sum().backward()
        # 断言梯度计算的结果与期望值6 * 2 * a相等
        # 注意，因为a只保存了一次，所以要乘以2
        self.assertEqual(6 * 2 * a, a.grad)

        # 创建一个形状为(5,)的张量a，要求计算梯度
        a = torch.randn(5, requires_grad=True)
        # 计算张量a的立方
        y = a**3
        # 计算张量y的元素和
        s = torch.sum(y)
        # 使用torch.autograd.graph.saved_tensors_hooks函数设置保存的张量钩子
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            # 对张量s相对于张量a进行梯度计算，并创建计算图
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            # 对计算得到的梯度g进行求和，并进行反向传播
            g.sum().backward()
            # 断言梯度计算的结果与期望值6 * 4 * a相等
            # 注意，因为pow_backward的梯度是grad * (exp * self.pow(exp - 1))
            # 所以grad被保存了，self（即a）也被保存了
            self.assertEqual(6 * 4 * a, a.grad)

        # 使用torch.autograd.graph.saved_tensors_hooks函数再次设置保存的张量钩子
        with torch.autograd.graph.saved_tensors_hooks(lambda x: 2 * x, lambda x: x):
            # 创建一个形状为(5,)的张量a，要求计算梯度
            a = torch.randn(5, requires_grad=True)
            # 计算张量a的立方
            y = a**3
            # 计算张量y的元素和
            s = torch.sum(y)
            # 对张量s相对于张量a进行梯度计算，并创建计算图
            (g,) = torch.autograd.grad(s, (a,), create_graph=True)
            # 对计算得到的梯度g进行求和，并进行反向传播
            g.sum().backward()
            # 断言梯度计算的结果与期望值6 * 8 * a相等
            # 组合上述两个块：2 * 4 = 8
            # 注意，在这种情况下，a被保存了两次
            self.assertEqual(6 * 8 * a, a.grad)

    # 定义一个测试函数，用于测试包装数字的保存变量钩子
    def test_wrapped_number_saved_variable_hooks(self):
        # 定义一个错误的钩子函数，用于报告运行时错误
        def err_hook(x):
            raise RuntimeError("this hook should not be called")

        # 使用torch.autograd.graph.saved_tensors_hooks函数设置错误的保存的张量钩子
        with torch.autograd.graph.saved_tensors_hooks(err_hook, err_hook):
            # 创建一个形状为(5,)的张量a，要求计算梯度
            a = torch.randn(5, requires_grad=True)
            # 计算a * 3的元素和
            out = (a * 3).sum()
            # 3被保存为一个保存的张量，因为它是一个包装数字，
            # 但是包装数字应特殊处理，以避免触发保存变量钩子
            torch.autograd.grad(out, (a,))
    # 定义一个测试方法，用于测试在 CPU 上保存计算图的行为
    def test_graph_save_on_cpu(self):
        # 定义内部测试函数，接收获取输入、是否使用 CUDA、是否使用固定内存等参数
        def test(get_input, cuda, pin_memory):
            # 使用 torch.autograd.graph.save_on_cpu 上下文管理器保存计算图在 CPU 上的行为
            with torch.autograd.graph.save_on_cpu(pin_memory):
                # 获取输入张量 a
                a = get_input()
                # 如果使用 CUDA，将张量 a 移动到 GPU 上
                if cuda:
                    a.cuda()
                # 计算张量 a 的平方
                y = a * a
                # 断言保存的自身张量与 a 相等
                self.assertEqual(a, y.grad_fn._saved_self)
                # 断言保存的另一个张量与 a 相等
                self.assertEqual(a, y.grad_fn._saved_other)
                # 断言保存的自身张量的数据类型与 a 的数据类型相同
                self.assertEqual(a.dtype, y.grad_fn._saved_self.dtype)
                # 断言保存的自身张量的布局与 a 的布局相同
                self.assertEqual(a.layout, y.grad_fn._saved_self.layout)
                # 如果 y 是稀疏张量，则将其转换为密集张量
                if y.is_sparse:
                    y = y.to_dense()
                # 对 y 求和并进行反向传播
                y.sum().backward()

                # 计算预期的梯度
                actual = 2 * a
                expected = a.grad
                # 如果 a 是稀疏张量，则将 actual 和 expected 同样转换为稀疏张量
                if a.is_sparse:
                    actual = actual.coalesce()
                    expected = expected.coalesce()

                # 断言 actual 和 expected 相等
                self.assertEqual(actual, expected)

        # 遍历不同的 CUDA 和固定内存选项进行测试
        for cuda in [False] + ([True] if torch.cuda.is_available() else []):
            for pin_memory in [True, False]:
                # 测试 FloatTensor
                test(lambda: torch.randn(5, requires_grad=True), cuda, pin_memory)
                # 测试 DoubleTensor
                test(
                    lambda: torch.randn(5, requires_grad=True, dtype=torch.double),
                    cuda,
                    pin_memory,
                )
                # 创建稀疏张量并进行测试
                x = torch.sparse_coo_tensor(
                    torch.tensor([[1, 1]]).long(),
                    torch.tensor([1.0, 1.0]),
                    requires_grad=True,
                )
                test(lambda: x, cuda, pin_memory)

    # 根据是否测试 CUDA，决定是否跳过该测试方法
    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_graph_save_on_cpu_cuda(self):
        # 定义函数 f，对输入 x 执行一系列计算
        def f(x):
            a = x + 1
            return a * a

        # 使用具有梯度信息的张量 a 在 CUDA 设备上执行函数 f
        a = torch.ones(1, requires_grad=True, device="cuda")
        y = f(a)
        memory_with_grad = torch.cuda.memory_allocated()

        # 删除张量 a 和 y，释放相关的 CUDA 内存
        del a
        del y

        # 使用没有梯度信息的张量 a 在 CUDA 设备上执行函数 f
        a = torch.ones(1, requires_grad=True, device="cuda")
        with torch.no_grad():
            y = f(a)
        memory_without_grad = torch.cuda.memory_allocated()

        # 断言使用有梯度和无梯度的内存使用情况，有梯度情况下应该比无梯度情况下内存使用更大
        self.assertGreater(memory_with_grad, memory_without_grad)

        # 再次删除张量 a 和 y，释放相关的 CUDA 内存
        del a
        del y

        # 使用 hooks 在 CUDA 设备上执行函数 f
        with torch.autograd.graph.save_on_cpu():
            a = torch.ones(1, requires_grad=True, device="cuda")
            y = f(a)
            memory_with_hooks = torch.cuda.memory_allocated()
            # 断言使用 hooks 和没有梯度时的内存使用情况相等
            self.assertEqual(memory_with_hooks, memory_without_grad)

    # 根据是否测试 CUDA，决定是否跳过该测试方法
    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_scalar_grad_mixed_device(self):
        # 创建一个标量张量 x，需要计算梯度
        x = torch.tensor(1.0, requires_grad=True)
        # 创建一个在 CUDA 设备上的随机张量 y
        y = torch.randn(2, 2, device="cuda")
        # 计算 x 与 y 的乘积
        out = x * y
        # 对乘积的所有元素求和并进行反向传播
        out.sum().backward()
    # 定义一个测试函数，测试多个梯度钩子的行为
    def test_multi_grad_all_hooks(self):
        # 创建四个随机张量，并指定需要计算梯度
        t1 = torch.rand(2, requires_grad=True)
        t2 = torch.rand(2, requires_grad=True)
        t3 = torch.rand(2, requires_grad=True)
        t4 = torch.rand(2, requires_grad=True)

        # 确保我们能够正确检测到所有类型的节点

        # 对张量 t1 执行乘法操作（C++ Node）
        t1 = t1.mul(2)

        # 定义一个 Python 自定义函数类 Foo，继承自 Function
        class Foo(Function):
            @staticmethod
            def forward(ctx, a):
                # 在前向传播中，返回输入张量的克隆
                return a.clone()

            @staticmethod
            def backward(ctx, gO):
                # 在反向传播中，直接返回梯度值
                return gO

        # 对张量 t2 应用自定义函数 Foo（Python custom Function）
        t2 = Foo.apply(t2)

        # 对张量 t3 应用一个未定义梯度的 C++ Node 操作
        t3 = torch._C._functions.UndefinedGrad()(t3)

        # 定义一个 C++ 自定义操作的源代码
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  // 定义静态方法 forward，用于前向传播计算，返回输入张量的克隆
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x.clone();
  }

  // 定义静态方法 backward，用于反向传播计算，返回梯度输出列表
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

// 定义函数 custom_op_backed_by_autograd_fn，调用自定义的 AutogradFunction 的 apply 方法
torch::Tensor custom_op_backed_by_autograd_fn(torch::Tensor x) {
  return CustomOpAutogradFunction::apply(x);
}

// 注册自定义操作到 Torch 脚本中的库 test_autograd_cpp_node
TORCH_LIBRARY(test_autograd_cpp_node, m) {
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}

module = load_inline(
    name="test_autograd_cpp_node",
    cpp_sources=cpp_source,
    functions="custom_op_backed_by_autograd_fn",
    verbose=True,
)

// 调用 Torch 操作，对张量 t4 进行自定义操作
t4 = torch.ops.test_autograd_cpp_node.custom_op_backed_by_autograd_fn(t4)

// 初始化结果和计数变量
res = [None] * 4
count = [0]

// 定义梯度钩子函数 hook，用于在多个张量的梯度计算时触发
def hook(grads):
    nonlocal res
    count[0] += 1
    // 检查每个梯度是否为 None，更新结果列表 res
    res = [g is not None for g in grads]

// 注册多个张量的梯度钩子函数
handle = torch.autograd.graph.register_multi_grad_hook((t1, t2, t3, t4), hook)

// 计算张量 t2 和 t3 的乘积
out = t2 * t3

// 对结果张量进行求和并执行反向传播，保留计算图以便后续计算
out.sum().backward(inputs=(t2, t3), retain_graph=True)
self.assertEqual(count[0], 1)
self.assertEqual(res, [False, True, True, False])

// 再次执行反向传播，验证计数是否增加且部分梯度不为 None
out.sum().backward(inputs=(t1, t4), retain_graph=True)
self.assertEqual(count[0], 1)

// 继续执行反向传播，验证计数增加且特定梯度不为 None
out.sum().backward(inputs=(t1, t3), retain_graph=True)
self.assertEqual(count[0], 2)
self.assertEqual(res, [False, False, True, False])

// 定义一个自定义 Torch 自动求导函数 Func，用于验证异常情况处理
class Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, gO):
        // 抛出运行时错误以模拟异常情况
        raise RuntimeError("error message")

// 执行自定义函数并验证是否抛出预期的运行时异常信息
out = Func.apply(t2) * t3
with self.assertRaisesRegex(RuntimeError, "error message"):
    out.sum().backward(inputs=(t2, t3), retain_graph=True)
self.assertEqual(count[0], 2)

// 移除先前注册的梯度钩子函数
handle.remove()

// 再次执行反向传播，验证计数未增加
out.sum().backward(inputs=(t1, t3), retain_graph=True)
self.assertEqual(count[0], 2)
    def test_multi_grad_any_hooks(self):
        hook_id = 0  # 初始化钩子 ID
        any_hook_handles: List[RemovableHandle] = []  # 初始化任意钩子的句柄列表

        class MultiOutputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(3, 3)

            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                z = self.lin(x)  # 线性层计算
                out = torch.sin(z), torch.cos(z)  # 计算正弦和余弦
                nonlocal hook_id  # 声明 hook_id 为非局部变量
                z.register_hook(partial(hook, hook_id))  # 注册钩子函数到 z
                hook_id += 1  # 更新钩子 ID
                any_hook_handles.append(
                    torch.autograd.graph.register_multi_grad_hook(
                        out, partial(hook, hook_id), mode="any"  # 注册多梯度钩子到 out
                    )
                )
                hook_id += 1  # 更新钩子 ID
                return out  # 返回正弦和余弦值

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.mod1 = MultiOutputModule()  # 初始化多输出模块1
                self.mod2 = MultiOutputModule()  # 初始化多输出模块2

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.mod1(x)  # 模块1前向传播
                z = y[0] + y[1]  # 计算 y[0] 和 y[1] 的和
                return self.mod2(z)  # 模块2前向传播

        hook_order: List[int] = []  # 初始化钩子顺序列表
        hook_count = 0  # 初始化钩子计数

        def hook(hook_id: int, *unused):
            nonlocal hook_count  # 钩子函数内部声明非局部变量
            nonlocal hook_order  # 钩子函数内部声明非局部变量
            hook_count += 1  # 增加钩子计数
            hook_order.append(hook_id)  # 将钩子 ID 添加到钩子顺序列表

        # Any hooks: IDs 1 and 3; regular hooks: IDs 0 and 2
        model = Model()  # 创建模型实例
        inp = torch.randn((2, 3))  # 创建随机输入张量
        out = model(inp)  # 模型前向传播
        (out[0] + out[1]).sum().backward()  # 计算梯度
        # 检查任意钩子是否仅运行一次，并且在常规钩子之前运行
        self.assertEqual(len(any_hook_handles), 2)  # 断言任意钩子句柄列表长度为2
        self.assertEqual(hook_order, [3, 2, 1, 0])  # 断言钩子顺序列表的顺序

        hook_id = 0  # 重置钩子 ID
        hook_order.clear()  # 清空钩子顺序列表
        any_hook_handles.clear()  # 清空任意钩子句柄列表
        out = model(inp)  # 模型再次前向传播
        for handle in any_hook_handles:
            handle.remove()  # 移除任意钩子句柄
        (out[0] + out[1]).sum().backward()  # 计算梯度
        # 检查当移除任意钩子后，任意钩子不再运行
        self.assertEqual(hook_order, [2, 0])  # 断言钩子顺序列表的顺序

    def test_multi_grad_hooks_invalid_mode(self):
        t1 = torch.rand(2, requires_grad=True)  # 创建随机张量 t1，需要梯度
        t2 = torch.rand(2, requires_grad=True)  # 创建随机张量 t2，需要梯度
        regex = r"Expects mode to be one of \('all', 'any'\) but got foo"
        with self.assertRaisesRegex(ValueError, regex):
            torch.autograd.graph.register_multi_grad_hook(
                (t1, t2), lambda _: None, mode="foo"  # 注册多梯度钩子，使用无效的模式 "foo"
            )

    def test_pynode_destruction_deadlock(self):
        script = """
import torch

# 定义一个自定义的 Torch 自动求导函数 Foo
class Foo(torch.autograd.Function):
    # 前向传播函数，ctx 是一个上下文对象，用于保存中间结果和梯度信息
    @staticmethod
    def forward(ctx, x):
        # 返回输入张量的克隆副本
        return x.clone()

    # 后向传播函数，处理梯度传播
    @staticmethod
    def forward(ctx, gO):
        # 返回梯度张量的克隆副本
        return gO.clone()

# 定义一个函数 get_out，用于生成计算图并进行反向传播
def get_out():
    # 创建一个随机张量 inp，并要求计算梯度
    inp = torch.rand(2, requires_grad=True)

    # 调用自定义的 Foo 函数进行前向传播计算，并返回结果 right
    right = Foo.apply(inp)

    # 对 inp 进行克隆操作，生成 left1 张量
    left1 = inp.clone()
    # 对 left1 进行平方操作，生成 left2 张量
    left2 = left1 ** 2

    # 对 left1 进行原地修改，增加其值以触发梯度计算错误
    left1 += 1

    # 将 left2 和 right 张量相加，生成 out 张量作为最终输出
    out = left2 + right

    return out

# 调用 get_out 函数生成计算图并进行反向传播
get_out().sum().backward()

# 以下是一个长时间运行的子进程调用的例子，用于检查错误处理和超时情况
# 使用 subprocess 模块调用 Python 脚本，并设置超时时间为 20 秒
# 如果超时则抛出 TimeoutExpired 异常，如果返回码小于 0 则可能是段错误
# 否则检查输出中是否包含特定的错误信息以判断测试是否通过
    # 定义一个测试方法，用于验证修改状态后的视图重播功能
    def test_view_func_replay_with_modified_state(self):
        # 强制使用原始视图跟踪
        with torch.autograd._force_original_view_tracking(True):
            # 创建一个形状为 (3, 4, 5) 的随机张量 base
            base = torch.randn(3, 4, 5)
            # 选择 base 的第二维度为索引 2 的视图 view
            view = base.select(1, 2)

            # 定义一个访问函数，用于修改保存的索引
            def symint_visitor_fn(x):
                # 修改保存的索引
                return x + 1

            # 确保修改状态会改变视图重播
            new_base = torch.randn_like(base)
            # 使用新的基础张量 new_base 调用视图函数，应用修改函数 symint_visitor_fn
            new_view = view._view_func(new_base, symint_visitor_fn=symint_visitor_fn)
            # 断言新视图等于基础张量 new_base 的索引为 3 的视图
            self.assertEqual(new_view, new_base.select(1, 3))

            # 确保保存的状态在之后会恢复
            self.assertEqual(view._view_func(new_base), new_base.select(1, 2))

            # 检查修改张量状态。当前，slice_inverse() 是唯一保存张量的视图
            base = torch.randn(3, 4, 5)
            # 对 base 进行切片操作，选取第二维度的切片为 [2:3]
            sliced = base[:, 2:3, :].detach()
            # 使用 slice_inverse() 方法创建视图 view
            view = torch.ops.aten.slice_inverse(sliced, base, 1, 2, 3, 1)

            # 替换的形状
            replacement_shape = (1, 2, 3)

            # 定义一个访问函数，返回一个形状比保存的形状小的张量
            def tensor_visitor_fn(x):
                return torch.randn(*replacement_shape)

            # 确保修改状态会改变视图重播
            new_sliced = torch.ones_like(base)[:, 2:3, :].detach()
            new_view = view._view_func(new_sliced, tensor_visitor_fn=tensor_visitor_fn)
            # 断言新视图的形状为替换形状 replacement_shape
            self.assertEqual(new_view.shape, replacement_shape)
            # 断言新视图等于 new_sliced 的扩展视图，形状为 replacement_shape
            self.assertEqual(
                new_view, new_sliced.as_strided(replacement_shape, (6, 3, 1))
            )

            # 确保保存的状态在之后会恢复
            self.assertEqual(view._view_func(sliced), base)
    def test_setup_context_when_forward_has_default_args(self):
        # 定义一个自定义的函数类 PowFunction，继承自 torch.autograd.Function
        class PowFunction(Function):
            # 静态方法：前向传播函数，计算 x 的 y 次幂，默认 y=3
            @staticmethod
            def forward(x, y=3):
                return torch.pow(x, y)

            # 静态方法：设置上下文函数，保存输入和输出到上下文对象 ctx
            @staticmethod
            def setup_context(ctx, inputs, output):
                x, y = inputs
                ctx.save_for_backward(x)
                ctx.y = y

            # 静态方法：反向传播函数，计算梯度
            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                y = ctx.y
                return gO * y * torch.pow(x, y - 1), None

        # 定义另一个自定义函数类 PowFunctionWithClassmethod，使用类方法定义前向传播和设置上下文函数
        class PowFunctionWithClassmethod(Function):
            # 类方法：前向传播函数，计算 x 的 y 次幂，默认 y=3
            @classmethod
            def forward(cls, x, y=3):
                return torch.pow(x, y)

            # 类方法：设置上下文函数，保存输入和输出到上下文对象 ctx
            @classmethod
            def setup_context(cls, ctx, inputs, output):
                x, y = inputs
                ctx.save_for_backward(x)
                ctx.y = y

            # 类方法：反向传播函数，计算梯度
            @classmethod
            def backward(cls, ctx, gO):
                (x,) = ctx.saved_tensors
                y = ctx.y
                return gO * y * torch.pow(x, y - 1), None

        # 创建一个张量 x，设置 requires_grad=True
        x = torch.tensor(2.0, requires_grad=True)

        # 创建一个张量 y，不需要计算梯度
        y = torch.tensor(8.0)
        # 创建一个期望的结果张量 y_expected
        y_expected = torch.tensor(12.0)

        # 使用 PowFunction 类的 apply 方法计算 x 的 3 次幂 y1，并计算 x 的梯度
        y1 = PowFunction.apply(x)
        (y1_expected,) = torch.autograd.grad(y1, x)

        # 使用 PowFunctionWithClassmethod 类的 apply 方法计算 x 的 3 次幂 y2，并计算 x 的梯度
        y2 = PowFunctionWithClassmethod.apply(x)
        (y2_expected,) = torch.autograd.grad(y2, x)

        # 断言 y 等于 y1
        self.assertEqual(y, y1)
        # 断言 y_expected 等于 y1_expected
        self.assertEqual(y_expected, y1_expected)
        # 断言 y 等于 y2
        self.assertEqual(y, y2)
        # 断言 y_expected 等于 y2_expected
        self.assertEqual(y_expected, y2_expected)

    # 如果不支持 CUDA 测试，则跳过该测试用例
    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_gradcheck_default_device_placement_context(self):
        # 在 fast_mode=True 的情况下进行梯度检查，创建一个随机向量 x，使用双精度，在 CUDA 设备上计算
        with torch.device("cuda"):
            x = torch.randn(3, dtype=torch.double, requires_grad=True)

            # 定义一个简单的函数 func，计算输入的平方
            def func(inp):
                return inp**2.0

            # 断言 func 函数的梯度检查通过，使用 fast_mode=True
            self.assertTrue(gradcheck(func, x, fast_mode=True))
# 定义函数，根据给定形状和最大索引生成索引张量
def index_perm_variable(shape, max_indices):
    # 如果形状不是元组，则转换为元组
    if not isinstance(shape, tuple):
        shape = (shape,)

    # 使用随机排列生成长度为 reduce(mul, shape) 的随机索引，并按 shape 形状重塑
    index = torch.randperm(max_indices).narrow(0, 0, reduce(mul, shape)).view(shape)
    return index


# 定义函数，返回一个服从伯努利分布的零维张量
def bernoulli_scalar():
    return torch.tensor(0, dtype=torch.uint8).bernoulli_()


# 定义 TestCase 的子类 TestAutogradForwardModeBatchedGrad
class TestAutogradForwardModeBatchedGrad(TestCase):
    # 定义测试方法，测试基础的 out-of-place 梯度检查
    def test_out_of_place_basic(self):
        # 创建两个随机张量 a 和 b，类型为 double，同时需要计算梯度
        a = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        b = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        # 使用 gradcheck 检查 torch.sin 函数的梯度
        self.assertTrue(
            gradcheck(
                torch.sin,
                a,
                check_forward_ad=True,
                check_batched_grad=True,
                check_batched_forward_grad=True,
            )
        )
        # 使用 gradcheck 检查 torch.add 函数的梯度，传入参数 a 和 b
        self.assertTrue(
            gradcheck(
                torch.add,
                (a, b),
                check_forward_ad=True,
                check_batched_grad=True,
                check_batched_forward_grad=True,
            )
        )

    # 定义测试方法，测试 out-of-place 梯度检查但不同布局的情况
    def test_out_of_place_not_same_layout(self):
        # 创建一个 2x2 的全零张量，并对其进行转置
        input = torch.zeros([2, 2]).transpose(0, 1)
        # 创建一个形状为 [2, 2, 2] 的全零张量
        tangent = torch.zeros([2, 2, 2])

        # 定义函数 jvp，对 tangent 进行 Jacobian 向量积运算
        def jvp(tangent):
            # 使用 fwAD.dual_level 上下文管理器
            with fwAD.dual_level():
                # 使用 fwAD.make_dual 将 input 和 tangent 转换为对偶数（dual number）
                x = fwAD.make_dual(input, tangent)
                # 返回对偶数的第二个元素
                return fwAD.unpack_dual(x)[1]

        # 使用 torch._vmap_internals._vmap 对 jvp 函数进行向量化映射，输入是 tangent
        x_tangent = torch._vmap_internals._vmap(jvp, 0, 0)(tangent)

        # 断言 x_tangent 不等于 tangent，确保不是同一个对象
        self.assertIsNot(x_tangent, tangent)

    # 定义测试方法，测试 inplace 操作但是视图与基础张量布局相同的情况
    def test_inplace_on_view_same_layout(self):
        # 创建一个 2x2 的全零张量 input 和一个形状为 [2, 2, 2] 的全零张量 tangent
        input = torch.zeros([2, 2])
        tangent = torch.zeros([2, 2, 2])
        # 创建一个与 base 张量布局相同的视图 view
        base = torch.zeros([2, 2])
        view = base.view_as(base)

        # 定义函数 jvp，对 tangent 进行 Jacobian 向量积运算
        def jvp(tangent):
            # 使用 fwAD.dual_level 上下文管理器
            with fwAD.dual_level():
                # 使用 fwAD.make_dual 将 input 和 tangent 转换为对偶数（dual number）
                x = fwAD.make_dual(input, tangent)
                # 将 x 的值拷贝到 view 中
                view.copy_(x)
                # 返回 unpack_dual(x) 和 unpack_dual(view) 的第二个元素，以及 view._base 的第二个元素
                return (
                    fwAD.unpack_dual(x)[1],
                    fwAD.unpack_dual(view)[1],
                    fwAD.unpack_dual(view._base)[1],
                )

        # 使用 torch._vmap_internals._vmap 对 jvp 函数进行向量化映射，输入是 tangent
        x_tangent, view_tangent, base_tangent = torch._vmap_internals._vmap(jvp, 0, 0)(
            tangent
        )

        # 断言 view_tangent 不是视图
        self.assertFalse(
            view_tangent._is_view()
        )  # 优化以共享同一张量！
        # 断言 view_tangent 和 base_tangent 是同一个对象
        self.assertIs(view_tangent, base_tangent)
        # 断言 x_tangent 和 tangent 是同一个对象
        self.assertIs(x_tangent, tangent)
    def test_inplace_on_view_not_same_layout(self):
        # 创建一个2x2的全零张量作为输入
        input = torch.zeros([2, 2])
        # 创建一个2x2x2的全零张量作为切线
        tangent = torch.zeros([2, 2, 2])
        # 创建一个2x2的全零张量的转置视图
        view = torch.zeros([2, 2]).transpose(0, 1)

        def jvp(tangent):
            # 进入自动微分的双重模式
            with fwAD.dual_level():
                # 将输入张量和切线转换为双重数
                x = fwAD.make_dual(input, tangent)
                # 将x的值拷贝到视图中（inplace操作）
                view.copy_(x)
                # 返回转换后的值和视图的值及其基础张量的值
                return (
                    fwAD.unpack_dual(x)[1],
                    fwAD.unpack_dual(view)[1],
                    fwAD.unpack_dual(view._base)[1],
                )

        # 对jvp函数进行批处理映射
        x_tangent, view_tangent, base_tangent = torch._vmap_internals._vmap(jvp, 0, 0)(
            tangent
        )

        # 断言视图的基础张量与基础张量相同
        self.assertIs(view_tangent._base, base_tangent)
        # 断言输入切线与x的切线相同
        self.assertIs(x_tangent, tangent)
        # 断言视图的切线与输入切线不同
        self.assertIsNot(view_tangent, tangent)

    def test_metadata_check_for_storage_numel_skipped(self):
        # 参考：test_metadata_check_checks_storage_numel的反向测试
        # 创建一个有5个元素的随机张量，取前4个元素并且分离（detach）
        primal = torch.randn(5)[:4].detach()
        # 断言张量存储的长度为5
        self.assertEqual(len(primal.storage()), 5)
        # 创建一个10x4的随机张量作为切线
        tangent = torch.randn(10, 4)

        def jvp(tangent):
            # 进入自动微分的双重模式
            with fwAD.dual_level():
                # 创建一个双重数
                dual = fwAD.make_dual(primal, tangent)
                # 解包双重数中的切线
                _, unpacked_tangent = fwAD.unpack_dual(dual)

                # 没有进行拷贝操作
                # 断言切线与解包后的切线相同
                self.assertIs(tangent, unpacked_tangent)

                # 使用as_strided会引发异常
                with self.assertRaisesRegex(
                    RuntimeError, "can access memory outside of `tensor`"
                ):
                    dual.as_strided((5,), (1,), 0)
            return unpacked_tangent

        # 对jvp函数进行批处理映射
        torch._vmap_internals._vmap(jvp, 0, 0)(tangent)
    # 定义一个测试类 TestAutogradForwardMode，继承自 TestCase
class TestAutogradForwardMode(TestCase):
    # 定义 tearDown 方法，在每个测试后运行以清理环境
    def tearDown(self):
        # 确保即使某个测试失败，也不会影响其它测试
        while fwAD._current_level >= 0:
            fwAD.exit_dual_level()

        super().tearDown()  # 调用父类的 tearDown 方法进行清理

    # 定义测试方法 test_forward_level_cleanup
    def test_forward_level_cleanup(self):
        # 定义内部函数 get_tensor_and_weak_ref
        def get_tensor_and_weak_ref():
            # 创建一个新的 Tensor 并生成弱引用
            t = torch.rand(2, requires_grad=True)
            return t, torch._C._WeakTensorRef(t)

        # 对内部函数进行基本检查
        t, t_ref = get_tensor_and_weak_ref()
        self.assertFalse(t_ref.expired())  # 确保弱引用有效

        del t  # 删除 Tensor 对象
        self.assertTrue(t_ref.expired())  # 确保弱引用失效

        # 主测试代码
        foo = torch.rand(2)

        # 进入双重模式的上下文
        with fwAD.dual_level():
            tangent, tangent_ref = get_tensor_and_weak_ref()
            self.assertFalse(tangent_ref.expired())  # 确保弱引用有效

            dual = fwAD.make_dual(foo, tangent)
            self.assertFalse(tangent_ref.expired())  # 确保弱引用仍有效

            # 确保我们提供的切线被原样重用
            self.assertTrue(fwAD.unpack_dual(dual)[1] is tangent)

            # 确保 dual 对象保持切线对象的有效性
            del tangent  # 删除切线对象
            self.assertFalse(tangent_ref.expired())  # 确保弱引用仍有效

            # 确保双重模式不会保持切线对象的 C++ 版本的有效性
            del dual  # 删除 dual 对象
            self.assertTrue(tangent_ref.expired())  # 确保弱引用失效

    # 定义测试方法 test_size_check
    def test_size_check(self):
        foo = torch.rand(2)
        tangent = torch.rand(3)

        # 在双重模式中进行上下文管理，预期会抛出 RuntimeError 异常
        with fwAD.dual_level():
            with self.assertRaisesRegex(
                RuntimeError,
                "Trying to set a forward gradient that has a different size",
            ):
                dual = fwAD.make_dual(foo, tangent)

            dual = fwAD.make_dual(foo, tangent[1:])  # 创建 dual 对象

    # 定义测试方法 test_metadata_check_checks_storage_numel
    def test_metadata_check_checks_storage_numel(self):
        primal = torch.randn(5)[:4].detach()
        self.assertEqual(len(primal.storage()), 5)  # 检查原始张量的存储长度为 5
        tangent = torch.randn(4)

        # 在双重模式中进行上下文管理
        with fwAD.dual_level():
            dual = fwAD.make_dual(primal, tangent)  # 创建 dual 对象
            _, unpacked_tangent = fwAD.unpack_dual(dual)

            # 验证解包后的切线对象的修改不会影响原始切线对象
            tangent_clone = tangent.clone()
            unpacked_tangent *= 2
            self.assertTrue(torch.allclose(tangent_clone, tangent))

            # 调用 as_strided 方法，确保没有异常抛出
            dual.as_strided((5,), (1,), 0)
    def test_metadata_check_checks_ignores_size_zero(self):
        # 创建一个大小为零的张量a，使用as_strided方法创建视图，但由于大小为零，实际内容为空
        a = torch.ones(0).as_strided(
            (
                0,
                1,
            ),
            (
                1,
                1,
            ),
            0,
        )
        # 创建另一个大小为零的张量b，同样使用as_strided方法创建视图，但是步长为零
        b = torch.ones(0).as_strided(
            (
                0,
                1,
            ),
            (
                1,
                0,
            ),
            0,
        )

        # 进入fwAD的双重级别上下文管理器
        with fwAD.dual_level():
            # 使用fwAD.make_dual函数创建对应a和b的双重张量dual
            dual = fwAD.make_dual(a, b)
            # 获取dual的主元素对角线
            torch.diagonal(dual, offset=0)

        # 创建一个大小为[0, 1]的随机张量input，数据类型为torch.complex128，需要计算梯度
        input = torch.rand([0, 1], dtype=torch.complex128, requires_grad=True)
        # 定义一个偏函数func，调用torch.diagonal函数，偏移为0
        func = partial(torch.diagonal, offset=0)
        # 使用torch.autograd.gradcheck检查func关于input的梯度，同时检查正向自动求导
        torch.autograd.gradcheck(func, (input,), check_forward_ad=True)

    def test_metadata_check_when_primal_has_conj_bit(self):
        # 确保_has_same_storage_numel是一个fallthrough，这样conj位不会实现。
        # 如果实现了conj位，对于不索引整个存储的视图，布局检查会失败。
        # 创建一个大小为(2, 2)的随机双精度复数张量a，并对其进行共轭操作
        a = torch.randn(2, 2, dtype=torch.cdouble).conj()
        # 创建一个与a形状相同的随机张量b
        b = torch.rand_like(a)

        # 断言a是共轭的
        self.assertTrue(torch.is_conj(a))
        # 断言a和b的存储长度相同
        self.assertEqual(len(a.storage()), len(b.storage()))

        # 进入fwAD的双重级别上下文管理器
        with fwAD.dual_level():
            # 使用fwAD.make_dual函数创建对应a和b的双重张量dual，并获取其子张量[1:]
            dual = fwAD.make_dual(a, b)
            dual[1:]

    def test_metadata_check_when_primal_has_neg_bit(self):
        # 确保_has_same_storage_numel是一个fallthrough，这样conj位不会实现。
        # 如果实现了conj位，对于不索引整个存储的视图，布局检查会失败。
        # 创建一个大小为(2, 2)的随机双精度复数张量a，并获取其虚部
        a = torch.randn(2, 2, dtype=torch.cdouble).conj().imag
        # 创建一个大小为(2, 2)的随机双精度复数张量b，并获取其虚部
        b = torch.randn(2, 2, dtype=torch.cdouble).imag

        # 断言a是负数的
        self.assertTrue(torch.is_neg(a))
        # 断言a和b的存储长度相同
        self.assertEqual(len(a.storage()), len(b.storage()))

        # 进入fwAD的双重级别上下文管理器
        with fwAD.dual_level():
            # 使用fwAD.make_dual函数创建对应a和b的双重张量dual，并获取其子张量[1:]
            dual = fwAD.make_dual(a, b)
            dual[1:]

    def test_metadata_check_check_conj(self):
        # 定义键值对字典keys，包含NEITHER、CONJ、NEG三种键，每种键对应一个操作
        keys = {
            "NEITHER": lambda x: x,
            "CONJ": lambda x: x.conj(),
            "NEG": lambda x: x._neg_view(),
        }

        # 遍历keys字典的键组合，primal_key和tangent_key分别表示主元和切向元操作
        for primal_key, tangent_key in product(keys, keys):
            # 根据primal_key选择对应操作生成随机双精度复数张量x
            x = keys[primal_key](torch.randn(2, 3, 4, dtype=torch.cdouble))
            # 根据tangent_key选择对应操作生成随机双精度复数张量t
            t = keys[tangent_key](torch.randn(2, 3, 4, dtype=torch.cdouble))

            # 如果primal_key和tangent_key相同
            if primal_key == tangent_key:
                # 进入fwAD的双重级别上下文管理器
                with fwAD.dual_level():
                    # 使用fwAD.make_dual函数创建对应x和t的双重张量dual
                    dual = fwAD.make_dual(x, t)
                    # 断言fwAD.unpack_dual(dual).tangent等于t
                    self.assertTrue(fwAD.unpack_dual(dual).tangent is t)
                    # 获取dual的实部和虚部
                    torch.real(dual)
                    torch.imag(dual)
            else:
                # 进入fwAD的双重级别上下文管理器
                with fwAD.dual_level():
                    # 使用fwAD.make_dual函数创建对应x和t的双重张量dual
                    dual = fwAD.make_dual(x, t)
                    # 断言fwAD.unpack_dual(dual).tangent不等于t
                    self.assertTrue(fwAD.unpack_dual(dual).tangent is not t)
                    # 获取dual的实部和虚部
                    torch.real(dual)
                    torch.imag(dual)
    def test_metadata_check_ignore_storage_offset_for_zero_numel_tensor(self):
        # 此测试用例是为了验证处理零元素张量时忽略存储偏移的行为
        a = torch.tensor([1.0]).as_strided((0,), (1,), 1)
        b = torch.tensor([1.0]).as_strided((0,), (1,), 2)

        with fwAD.dual_level():
            # 使用双重级别上下文管理器
            dual_input = fwAD.make_dual(a, b)
            # 检查是否没有进行拷贝操作
            self.assertIs(fwAD.unpack_dual(dual_input).tangent, b)

        a = torch.tensor([1.0]).as_strided((1,), (2,), 0)
        b = torch.tensor([1.0]).as_strided((1,), (1,), 0)

        with fwAD.dual_level():
            # 使用双重级别上下文管理器
            dual_input = fwAD.make_dual(a, b)
            dual_input[1:]

    # 以下测试函数旨在确保以下所有行为：
    #   - 确保 Python 绑定中的默认级别系统工作正常
    #   - 确保只存在级别 0，并且嵌套正确禁用
    #   - 确保打印功能正常工作
    #   - 确保基本打包/解包功能正常工作
    #   - 确保高级打包/解包功能正常工作
    #     - 内存/版本计数器共享
    #     - 后向自动微分（常规操作）
    #   - 确保视图 + 原地操作对两种模式都正常工作
    #   - 确保在退出级别时进行适当的清理

    def test_default_level(self):
        # 创建两个随机张量
        foo = torch.rand(2)
        bar = torch.rand(2)

        with fwAD.dual_level():
            # 在双重级别上下文中创建双重对象
            baz = fwAD.make_dual(foo, bar)
            baz_primal, baz_tangent = fwAD.unpack_dual(baz)
        self.assertEqual(baz_primal, foo)
        # 不需要严格验证这两个对象是否完全相同，未来可以放宽要求
        self.assertIs(baz_tangent, bar)

        baz_primal, baz_tangent = fwAD.unpack_dual(baz)
        self.assertEqual(baz_primal, foo)
        self.assertEqual(baz_tangent, None)

    def test_fwd_grad_enabled(self):
        # 测试一些私有辅助函数以启用/禁用前向梯度模式
        enabled = fwAD._is_fwd_grad_enabled()
        self.assertTrue(enabled)

        try:
            torch._C._set_fwd_grad_enabled(False)
            enabled = fwAD._is_fwd_grad_enabled()
            self.assertFalse(enabled)
        finally:
            torch._C._set_fwd_grad_enabled(True)

        enabled = fwAD._is_fwd_grad_enabled()
        self.assertTrue(enabled)

    def test_set_fwd_grad_enabled(self):
        # 测试一个私有辅助函数
        try:
            torch._C._set_fwd_grad_enabled(False)
            enabled = fwAD._is_fwd_grad_enabled()
            self.assertFalse(enabled)

            with fwAD._set_fwd_grad_enabled(True):
                enabled = fwAD._is_fwd_grad_enabled()
                self.assertTrue(enabled)

            enabled = fwAD._is_fwd_grad_enabled()
            self.assertFalse(enabled)
        finally:
            torch._C._set_fwd_grad_enabled(True)
    def test_nested_level(self):
        with fwAD.dual_level() as level:
            # 检查是否处于双重模式的第一层级
            self.assertEqual(level, 0)

        with fwAD.dual_level():
            # 在双重模式下尝试进入第二层级会引发运行时错误
            with self.assertRaisesRegex(
                RuntimeError, "Nested forward mode AD is not supported at the moment"
            ):
                nest_level = fwAD.enter_dual_level()

    def test_set_fw_grad_having_own_fw_grad_at_same_level(self):
        foo = torch.rand(2)
        bar = torch.rand(2)
        baz = torch.rand(2)

        with fwAD.dual_level():
            # 创建第一个双重模式下的双重张量
            dual = fwAD.make_dual(foo, bar)
            # 尝试在同一层级创建第二个带有前向梯度的双重张量将引发运行时错误
            with self.assertRaisesRegex(
                RuntimeError, "has a forward gradient at the same level"
            ):
                fwAD.make_dual(baz, dual)

    def test_codegen_ignores_undefined_outputs(self):
        # 此测试验证代码生成器在忽略未定义输出时的行为
        # 下面的例子中，grad_input 在 grad_output_mask 中被设为 False，所以卷积反向传播将在该位置返回一个未定义的张量。
        # 注意，为了使此测试正常工作，需要确保 grad_output 或 weight 中至少有一个是双重张量，以便 grad_input 需要前向梯度。
        weight = torch.randn(6, 1, 30, 30)
        inp = torch.rand((1, 1, 32, 32))
        out = torch.nn.functional.conv2d(inp, weight)
        grad_out = torch.ones_like(out)

        with fwAD.dual_level():
            # 创建带有前向梯度的双重张量
            dual_weight = fwAD.make_dual(weight, torch.ones_like(weight))
            # 调用 PyTorch 的 convolution_backward 函数来进行反向传播，测试是否返回 None 作为 grad_input
            grad_input, _, _ = torch.ops.aten.convolution_backward(
                grad_out,
                inp,
                dual_weight,
                (0,),
                (1, 1),
                (0, 0),
                (1, 1),
                False,
                (0, 0),
                1,
                (False, True, False),
            )
        self.assertIsNone(grad_input)

    def test_make_dual_inference_tensor_in_inference_mode(self):
        with torch.inference_mode():
            foo = torch.rand(2)
            bar = torch.rand(2)
            foo_copy = foo.clone()

            with fwAD.dual_level():
                # 在推断模式下创建双重张量将不会是视图
                dual = fwAD.make_dual(foo, bar)
                self.assertFalse(dual._is_view())

                dual += 1
                # 确保对双重张量进行操作不会影响原始张量
                self.assertFalse(torch.allclose(foo, foo_copy))
    def test_make_dual_torch_dispatch(self):
        counter = [0]

        class MySubclass(torch.Tensor):
            def __new__(cls, data=None):
                return torch.Tensor._make_subclass(cls, data)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # 检查是否为 torch.ops.aten.alias 函数重载
                if func.overloadpacket == torch.ops.aten.alias:
                    counter[0] += 1

                    # 确保可以在此处重新启用 autograd
                    with torch.overrides.enable_reentrant_dispatch():
                        # 创建一个随机张量 foo，并启用梯度追踪
                        foo = torch.rand(1, requires_grad=True)
                        self.assertIsNotNone(foo.exp().grad_fn)

                # 使用 no_dispatch 上下文管理器禁用分发
                with no_dispatch():
                    return func(*args, **kwargs)

        a = torch.tensor(1.0)
        s = MySubclass(a)

        with fwAD.dual_level():
            # 只有原始张量会调用 "alias"
            fwAD.make_dual(s, torch.rand_like(s))
            self.assertEqual(counter[0], 1)
            fwAD.make_dual(torch.rand_like(s), s)
            self.assertEqual(counter[0], 1)

    def test_make_dual_forbid_integral_dtype(self):
        primal_f = torch.ones(2, 2, dtype=torch.float)
        primal_l = torch.ones(2, 2, dtype=torch.long)

        tangent_f = torch.ones(2, 2, dtype=torch.float)
        tangent_l = torch.ones(2, 2, dtype=torch.long)

        with fwAD.dual_level():
            # 浮点数原始张量和长整型切线张量
            with self.assertRaisesRegex(
                ValueError, "Expected tangent to be floating point or complex"
            ):
                fwAD.make_dual(primal_f, tangent_l)

            # 长整型原始张量和长整型切线张量
            with self.assertRaisesRegex(
                ValueError, "Expected primal to be floating point or complex"
            ):
                fwAD.make_dual(primal_l, tangent_l)

            # 长整型原始张量和浮点数切线张量
            with self.assertRaisesRegex(
                ValueError, "Expected primal to be floating point or complex"
            ):
                fwAD.make_dual(primal_l, tangent_f)

    def test_print(self):
        with fwAD.dual_level() as level:
            a = torch.rand(3)
            self.assertFalse("tangent=" in str(a))

            b = fwAD.make_dual(a, torch.rand(3))
            self.assertFalse("tangent=" in str(a))
            self.assertTrue("tangent=" in str(b))

            b_primal, b_tangent = fwAD.unpack_dual(b)
            self.assertFalse("tangent=" in str(b_primal))
            self.assertFalse("tangent=" in str(b_tangent))
    # 定义一个测试方法，用于基本的打包和解包操作的测试
    def test_basic_packing_unpacking(self):
        # 创建两个包含随机数据的张量
        foo = torch.rand(2)
        bar = torch.rand(2)

        # 进入双重级别上下文管理器
        with fwAD.dual_level():
            # 使用 make_dual 函数将 foo 和 bar 打包成一个双重张量对象 baz
            baz = fwAD.make_dual(foo, bar)
            # 使用 unpack_dual 函数解包双重张量 baz，得到原始值 baz_primal 和切向值 baz_tangent
            baz_primal, baz_tangent = fwAD.unpack_dual(baz)
            # 断言解包后的原始值与原始张量 foo 相等
            self.assertEqual(baz_primal, foo)
            # 断言解包后的切向值与切向张量 bar 是同一对象
            self.assertIs(baz_tangent, bar)

            # 检查解包后的双重张量返回的是一个命名元组
            # 注意：每次调用 unpack_dual 都会返回一个新的张量视图
            self.assertIsNot(baz_primal, fwAD.unpack_dual(baz).primal)
            self.assertEqual(baz_primal, fwAD.unpack_dual(baz).primal)
            self.assertIs(baz_tangent, fwAD.unpack_dual(baz).tangent)

            # 检查打包和解包操作没有改变输入张量的值
            foo_primal, foo_tangent = fwAD.unpack_dual(foo)
            self.assertEqual(foo_primal, foo)
            self.assertIsNone(foo_tangent)
    # 定义一个测试方法，用于验证非可微视图中的原地操作行为
    def test_view_inplace_non_differentiable_views(self):
        # 创建两个形状为 (2,) 的随机双精度张量
        original_foo = torch.rand(2, dtype=torch.double)
        # 创建两个值全为 1 的双精度张量
        original_bar = torch.ones(2, dtype=torch.double)

        # 对原始张量进行克隆，以便比较原地更新后的值与这些张量的原始内容
        foo = original_foo.clone()
        bar = original_bar.clone()

        # 进入双重自动微分上下文
        with fwAD.dual_level():
            # 在这个测试中，我们使用 "update" 表示计算对偶的右切线
            # 这里所有的原地操作都预期会更新张量的原始值，但不总是会更新它们的切线。
            # "non differentiable view" 所有提到的非可微视图指的是非前向可微视图，除非另有说明。
            # 更多关于这些视图如何工作的详细信息，请参见注释 [Forward Grad View/inplace]。

            # 检查原地操作是否不更新非可微视图
            # 创建对偶对象 dual，包含了 foo 和 bar 的原始值及其切线
            dual = fwAD.make_dual(foo, bar)
            # 原地将 dual 的原始值乘以 2
            dual *= 2
            # 检查非可微视图的切线是否未被更新
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            # 检查计算结果是否正确
            self.assertEqual(bar, original_bar * 2)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 2)
            self.assertEqual(foo, original_foo * 2)
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 2)
            
            # 另一个非可微视图
            # 将 dual 拆解为其原始值和切线
            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            # 检查第一个非可微视图的切线是否未被更新
            self.assertIsNone(fwAD.unpack_dual(dual_primal)[1])
            # 检查第二个非可微视图的切线是否未被更新
            self.assertIsNone(fwAD.unpack_dual(dual_tangent)[1])
            # 原地将 dual_primal 的原始值乘以 2
            dual_primal *= 2
            # 确保 dual 的切线未改变
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 4)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 2)
            # 原地将 dual_tangent 的原始值乘以 2
            dual_tangent *= 2
            # 确保 dual 的原始值未改变
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 4)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 4)
    def test_view_inplace_differentiable_views(self):
        original_foo = torch.rand(2)
        original_bar = torch.ones(2)

        # Do clones to be able to compare the values updated inplace
        # with the original content of these Tensors
        foo = original_foo.clone()
        bar = original_bar.clone()

        with fwAD.dual_level():
            # Check that inplace ops do update differentiable view but stop at non differentiable ones
            # A non differentiable view
            dual = fwAD.make_dual(foo, bar)
            # A differentiable view
            view = dual.narrow(0, 0, 1)
            view *= 2
            # Check that non differentiable view was not updated
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            # Check that differentiable view was updated
            self.assertEqual(fwAD.unpack_dual(dual)[1], torch.tensor([2.0, 1.0]))
            self.assertEqual(fwAD.unpack_dual(view)[1], torch.tensor([2.0]))

            # Check that we track differentiable view even for Tensors that are not dual
            baz = torch.rand(2)
            baz += dual
            # Ensure the sum with a dual Tensor still tracks gradients correctly
            self.assertEqual(fwAD.unpack_dual(baz)[1], fwAD.unpack_dual(dual)[1])
            # Updates on view should as well
            baz = torch.rand(2)
            baz[0] = dual[0]
            # Ensure assigning a dual Tensor element to another Tensor correctly tracks gradients
            self.assertEqual(fwAD.unpack_dual(baz)[1][0], fwAD.unpack_dual(dual)[1][0])
            # Unused values get a gradient of 0
            self.assertEqual(fwAD.unpack_dual(baz)[1][1], 0.0)

            # Check that forward non-differentiable views do prevent gradient update
            baz = torch.rand(2)
            view = baz.detach()
            view += dual
            # Ensure that detaching a Tensor prevents gradient propagation
            self.assertIsNone(fwAD.unpack_dual(baz)[1])
    # 测试函数：验证 inplace 操作总是创建一个视图
    def test_view_inplace_always_creates_a_view(self):
        # 查看 https://github.com/pytorch/pytorch/issues/67800
        # 代码路径可能依赖于操作。在编写时，当 self 不是双重张量时，
        # 对于 self 的结果前向梯度...
        # - add_ 的布局与 self 相同
        # - mul_ 的布局与 other 相同
        # 这种方式有点脆弱，因为上述取决于前向梯度表达式的编写方式。至少对于 add 和 mul，输出继承了左侧的布局。
        # 我们至少希望处理这两种情况。
        
        # 定义 inplace 二元操作列表
        inplace_binary_ops = (
            lambda x, y: x.add_(y),
            lambda x, y: x.mul_(y),
            lambda x, y: x.copy_(y),
        )

        for inplace_binary_op in inplace_binary_ops:
            # 创建基础张量 base，并生成其转置视图 view
            base = torch.randn(2, 2)
            view = base.transpose(0, 1)

            # 创建原始张量 primal 和切线张量 tangent
            primal = torch.randn(2, 2)
            tangent = torch.randn(2, 2)

            # 进入双重自动微分环境
            with fwAD.dual_level():
                # 创建双重张量 dual
                dual = fwAD.make_dual(primal, tangent)
                # 执行 inplace 二元操作于 view 和 dual
                inplace_binary_op(view, dual)

                # 验证原始张量和切线张量均创建了视图关系
                p, t = fwAD.unpack_dual(base)
                p_clone = p.clone()
                t_clone = t.clone()
                view *= 2
                p, t = fwAD.unpack_dual(base)

                # 使用断言验证视图变化是否符合预期
                self.assertTrue(torch.allclose(p_clone * 2, p))
                self.assertTrue(torch.allclose(t_clone * 2, t))

    # 测试函数：验证梯度清理操作
    def test_grad_cleanup(self):
        # 创建张量 foo, bar 和 baz
        foo = torch.rand(2)
        bar = torch.rand(2)
        baz = torch.rand(2)

        # 进入双重自动微分环境
        with fwAD.dual_level():
            # 创建双重张量 dual
            dual = fwAD.make_dual(foo, bar)
            # 验证 foo 的切线张量为 None
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            # 验证 dual 的切线张量为 bar
            self.assertIs(fwAD.unpack_dual(dual)[1], bar)

        # 验证 dual 的切线张量现在为 None
        self.assertIsNone(fwAD.unpack_dual(dual)[1])

        # 再次进入双重自动微分环境
        with fwAD.dual_level():
            # 验证 foo 的切线张量为 None
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            # 创建新的双重张量 new_dual
            new_dual = fwAD.make_dual(foo, baz)

            # 解包原双重张量 dual 和新双重张量 new_dual
            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            new_dual_primal, new_dual_tangent = fwAD.unpack_dual(new_dual)
            # 验证原双重张量和新双重张量的原始张量相同
            self.assertEqual(dual_primal, new_dual_primal)
            # 验证原双重张量的切线张量为 None
            self.assertIsNone(dual_tangent)
            # 验证新双重张量的切线张量为 baz
            self.assertEqual(new_dual_tangent, baz)

    # 测试函数：验证视图追踪的分离操作
    def test_detach_view_tracking(self):
        # 默认的 detach 是前向和反向均不可微分的
        foo = torch.rand(2)
        foo_weak = torch._C._WeakTensorRef(foo)

        # 执行 detach 操作
        out = foo.detach()

        # 删除 foo 引用
        del foo
        # 使用断言验证 foo 是否被成功删除
        self.assertTrue(foo_weak.expired())
    # 测试函数，验证 torch.add 函数在指定 out 参数时的异常情况
    def test_out_variant(self):
        # 进入双重自动微分模式
        with fwAD.dual_level():
            # 创建双重张量 foo 和普通张量 bar
            foo = fwAD.make_dual(torch.rand(2), torch.rand(2))
            bar = torch.rand(2)

            # 断言使用 torch.add 函数指定 out 参数时会抛出 RuntimeError 异常
            with self.assertRaisesRegex(RuntimeError, "out= function"):
                torch.add(bar, bar, out=foo)

            # 断言使用 torch.add 函数指定 out 参数时会抛出 RuntimeError 异常
            with self.assertRaisesRegex(RuntimeError, "out= function"):
                torch.add(foo, bar, out=bar)

    # 测试函数，验证在双重自动微分模式下的非可微输出的情况
    def test_non_differentiable(self):
        # 进入双重自动微分模式
        with fwAD.dual_level():
            # 创建双重张量 foo 和普通张量 bar
            foo = fwAD.make_dual(torch.rand(2), torch.rand(2))
            bar = torch.rand(2)

            # 创建布尔张量 eq，比较 foo 和 bar 是否相等，不会抛出异常
            eq = foo == bar

            # 使用 foo 的 inplace 方法 eq_
            foo.eq_(bar)

    # 测试函数，验证 torch.ops.aten._new_zeros_with_same_feature_meta 函数的行为
    def test_create_new_zeros_with_same_meta(self):
        # 获取 torch.ops.aten._new_zeros_with_same_feature_meta 函数引用
        new_zeroes_fn = torch.ops.aten._new_zeros_with_same_feature_meta

        # 定义内部函数 check，用于检查两个张量的特征元信息是否相同
        def check(a, b):
            # 定义内部函数 assert_same_meta，验证两个张量的特征元信息
            def assert_same_meta(t, target):
                # 遍历 t 张量的批次维度数量
                for num_bdim in range(t.dim()):
                    # 使用 new_zeroes_fn 函数生成新的张量 result
                    result = new_zeroes_fn(t, target, self_num_batch_dims=num_bdim)

                    # 断言 result 的维度等于 target 的维度加上 num_bdim
                    self.assertEqual(result.dim(), target.dim() + num_bdim)

                    # 仅检查特征维度的大小和步长匹配
                    for i in range(num_bdim, result.dim()):
                        self.assertEqual(result.size()[i], target.size()[i - num_bdim])
                        self.assertEqual(result.stride()[i], target.stride()[i - num_bdim])

                    # 检查是否生成了合理的步长
                    if target.is_contiguous():
                        self.assertTrue(result.is_contiguous())

                    # 检查 storage offset 是否一致
                    self.assertEqual(result.storage_offset(), target.storage_offset())

                    # 检查 storage 的长度是否符合预期
                    prod_of_t_bdims = reduce(operator.mul, t.size()[:num_bdim], 1)
                    self.assertEqual(len(result.storage()), len(target.storage()) * prod_of_t_bdims)

                    # 断言结果张量的数据类型与 target 相同
                    self.assertEqual(result.dtype, target.dtype)

            # 分别验证 a 和 b 的特征元信息
            assert_same_meta(a, b)
            assert_same_meta(b, a)

        # 创建不同形状和数据类型的张量 a 和 b，然后调用 check 函数进行验证
        a = torch.randn(5, dtype=torch.float)
        b = torch.randn(2, 3, 4, dtype=torch.double)
        check(a, b)

        # 创建非连续的张量 a 和 b，然后调用 check 函数进行验证
        a = torch.randn(2, 3, 4).transpose(0, 1).contiguous().transpose(0, 1)
        b = torch.randn(2, 3, 4)
        check(a, b)

        # 创建使用 narrow 方法得到的张量 a 和 b，然后调用 check 函数进行验证
        a = torch.randn(5).narrow(0, 1, 2)
        b = torch.randn(2)
        check(a, b)

        # 创建使用 resize_ 方法得到的张量 a 和 b，然后调用 check 函数进行验证
        a = torch.randn(5).resize_(4)
        b = torch.randn(4)
        check(a, b)

        # 创建元素数为零的张量 a 和 b，然后调用 check 函数进行验证
        a = torch.randn(1, 0, 2)
        b = torch.randn(1, 2)
        check(a, b)

        # 创建标量张量 a 和 b，然后调用 check 函数进行验证
        a = torch.tensor(1.0)
        b = torch.randn(1, 2)
        check(a, b)
    # 定义一个测试函数，用于测试反向图的销毁过程
    def test_backward_graph_destruction(self):
        # 定义内部函数fn，用于生成随机张量并进行自动求导设置
        def fn():
            # 创建一个形状为(10,)的随机张量a，并标记为需要梯度计算
            a = torch.rand(10, requires_grad=True)
            
            # 使用fwAD.make_dual函数创建a的对偶对象da
            da = fwAD.make_dual(torch.rand_like(a), a)
            
            # 对da应用指数函数，得到新的对偶对象db
            db = da.exp()

        # 进入fwAD.dual_level的上下文管理器
        with fwAD.dual_level():
            # 调用内部函数fn，执行其中的张量操作和自动求导操作
            fn()
        
        # 此测试确保我们不会在退出上下文管理器时发生死锁。
        # 如果发生死锁，很可能是与前向自动求导级别的锁定有关。
# 测试自动求导功能在各种设备类型上的通用性。
class TestAutogradDeviceType(TestCase):

    # 测试最小、最大、中位数和 NaN 中位数函数对所有值进行反向传播。
    def test_min_max_median_backprops_to_all_values(self, device):
        for f in [torch.min, torch.max, torch.median, torch.nanmedian]:
            # 创建一个张量 x1，设置了设备和需要梯度计算
            x1 = torch.tensor(
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], device=device, requires_grad=True
            )
            # 创建一个张量 x2，包含 NaN 值，需要梯度计算
            x2 = torch.tensor(
                [float("nan"), float("nan"), float("nan")], requires_grad=True
            )
            for x in [x1, x2]:
                # 对函数 f(x) 求值
                y = f(x)
                # 对 y 进行反向传播
                y.backward()
                # 断言梯度的和为 1.0
                self.assertEqual(x.grad.sum(), 1.0)
                # 断言梯度中等于 1/3 的数量为 3
                self.assertEqual((x.grad == 1 / 3).sum(), 3)

    # 测试 scatter_reduce 和 index_reduce 函数在多个最大/最小值时梯度是否均匀分布。
    # 此处测试这种情况，因为对于 test_min_max_median_backprops_to_all_values 中的情况，
    # gradgrad 不可微分，因此没有将其作为 SampleInput 添加。
    def test_scatter_index_reduce_amin_amax_backprops_to_all_values(self, device):
        fns = (torch.scatter_reduce, torch.index_reduce)
        reduces = ("amin", "amax")
        for fn, reduction in product(fns, reduces):
            # 创建一个需要梯度计算的随机张量 input
            input = torch.randn(
                (2, 3), device=device, dtype=torch.float64, requires_grad=True
            )
            # 克隆并分离出需要梯度计算的 src
            src = input.clone().detach_().requires_grad_(True)
            # 创建一个索引张量 idx
            idx = torch.arange(2).to(dtype=torch.long, device=device)
            if fn == torch.scatter_reduce:
                # 如果 fn 是 scatter_reduce，则扩展 idx
                idx = idx.unsqueeze(-1).expand((2, 3))
            
            # 使用 gradcheck 进行梯度检查
            gradcheck(fn, (input, 0, idx, src, reduction), check_batched_grad=False)

    # 测试当 src 中存在两个零值被散布到相同位置时，双向梯度计算会引发错误。
    def test_scatter_index_reduce_prod_gradgrad_error(self, device):
        input = torch.tensor(
            [1.0], device=device, dtype=torch.float64, requires_grad=True
        )
        src = torch.tensor(
            [0.0, 0.0], device=device, dtype=torch.float64, requires_grad=True
        )
        idx = torch.tensor([0, 0], device=device, dtype=torch.long)

        for fn in (torch.scatter_reduce, torch.index_reduce):
            # 检查在 gradcheck 中此案例是否通过
            gradcheck(fn, (input, 0, idx, src, "prod"), check_batched_grad=False)
            # 断言在双向梯度计算时会引发 RuntimeError 错误
            with self.assertRaisesRegex(
                RuntimeError, "Double backward is unsupported for"
            ):
                gradgradcheck(fn, (input, 0, idx, src, "prod"))

    @skipIfMps  # 测试不适用于 MPS 的情况，因为不支持双重类型
    def test_parameter_resize(self, device):
        # 创建一个 torch.nn.Parameter 对象，含有 16 个值，类型为 torch.double
        asd = torch.nn.Parameter(torch.ones(16, dtype=torch.double, device=device))

        for i in range(2):
            with torch.no_grad():
                # 将 asd 的值设置为除了第一个元素外的所有元素，并清除梯度
                asd.set_(asd[1:])
                asd.grad = None

            # 进行张量拼接，并对其求和进行反向传播
            m = torch.cat((asd, asd))
            m.sum().backward()
    # 跳过在 MPS 上不支持双精度类型的测试
    @skipIfMps  
    # 指定测试的数据类型为双精度和复数双精度
    @dtypes(torch.double, torch.cdouble)
    # 测试稀疏张量的构造函数、获取器及反向传播功能
    def test_sparse_ctor_getter_backward(self, device, dtype):
        # 查看此测试的预期行为，请参阅注释 [ Sparse: autograd and API ]
        def _test(size, sparse_dim, nnz, device):
            # 创建稀疏索引矩阵 i，形状为 sparse_dim x nnz
            i = torch.rand(sparse_dim, nnz)
            # 乘以尺寸 size 的相关部分，将其转换为长整型
            i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
            i = i.to(torch.long)

            # 创建具有梯度的随机输入 inp，形状为 v_size，数据类型为双精度，设备为指定设备
            inp = torch.randn(
                [nnz] + list(size[sparse_dim:]), dtype=torch.double, device=device, requires_grad=True
            )
            # 使用 genSparseTensor 方法生成稀疏张量 other
            other = self.genSparseTensor(
                size, sparse_dim, nnz, is_uncoalesced=True, device=device, dtype=dtype
            )[0]

            # 定义函数 fn，构建稀疏 COO 张量并执行相关操作
            def fn(v):
                # 构建稀疏 COO 张量 x
                x = torch.sparse_coo_tensor(i, v, size, dtype=dtype, device=device)
                # 将 x 与 other 相加并合并
                y = (x + other).coalesce()
                # 获取合并后的 y 的值
                yv = y.values()
                # 对 yv 进行 tanh 操作得到 new_v
                new_v = yv.tanh()
                # 构建新的稀疏 COO 张量 z
                z = torch.sparse_coo_tensor(y.indices(), new_v, y.size())
                # 返回合并后的 z 的值
                return z.coalesce().values()

            # 使用 gradcheck 函数检查反向传播
            gradcheck(fn, (inp,), check_batched_grad=False)
            # FIXME: 使 gradgradcheck 生效。
            # gradgradcheck(fn, (inp,), check_batched_grad=False)

            # 断言 _values 是不可微的
            with self.assertRaisesRegex(RuntimeError, "does not have a grad_fn"):
                other.detach().requires_grad_()._values().backward(
                    torch.ones_like(other._values())
                )

        # 针对不同的空条件（空索引、空值、空 nnz）进行测试
        for empty_i, empty_v, empty_nnz in product([True, False], repeat=3):
            sparse_size = [] if empty_i else [2, 1]
            dense_size = [1, 0, 2] if empty_v else [1, 2]
            nnz = 0 if empty_nnz else 5
            _test(sparse_size + dense_size, len(sparse_size), nnz, device)

    # 跳过元测试
    @skipMeta
    # 在 MPS 上跳过
    @skipIfMps
    # 指定测试的数据类型为双精度和复数双精度
    @dtypes(torch.double, torch.cdouble)
    # 定义一个测试方法，用于测试稀疏张量的反向传播功能
    def test_sparse_backward(self, device, dtype):
        # 定义一个自定义的函数类，用于固定梯度的处理
        class FixedGradientFunction(Function):
            @staticmethod
            # 前向传播方法：保存梯度信息并返回输入的原始数据
            def forward(ctx, x, grad_x):
                ctx.save_for_backward(grad_x)
                return x

            @staticmethod
            # 反向传播方法：根据保存的梯度返回梯度值，并且不对输入数据进行梯度传播
            def backward(ctx, grad_x):
                (saved_grad_x,) = ctx.saved_tensors
                return saved_grad_x, None

        # 定义一个大小为 [6, 3, 2] 的张量尺寸
        size = torch.Size([6, 3, 2])
        # 定义稀疏张量的索引 i1 和对应的值 v1
        i1 = torch.tensor(
            [
                [0, 3, 4],
                [0, 2, 2],
            ],
            dtype=torch.long,
        )
        v1 = make_tensor([3, 2], dtype=dtype, device=device)
        # 创建稀疏张量 sparse_grad1
        sparse_grad1 = torch.sparse_coo_tensor(i1, v1, size, dtype=dtype, device=device)
        # 定义第二个稀疏张量的索引 i2 和对应的值 v2
        i2 = torch.tensor(
            [
                [0, 1, 3, 4],
                [0, 1, 2, 2],
            ],
            dtype=torch.long,
        )
        v2 = make_tensor([4, 2], dtype=dtype, device=device)
        # 创建稀疏张量 sparse_grad2
        sparse_grad2 = torch.sparse_coo_tensor(i2, v2, size, dtype=dtype, device=device)
        # 创建密集张量 dense_grad
        dense_grad = torch.rand(size, device=device, dtype=dtype)
        # 获取 FixedGradientFunction 类的引用
        fn = FixedGradientFunction

        # 首先对稀疏张量进行操作
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        # 计算并累加三个函数应用的结果，对结果取绝对值并进行反向传播
        (
            fn.apply(x, sparse_grad1)
            + fn.apply(x, dense_grad)
            + fn.apply(x, sparse_grad2)
        ).sum().abs().backward()
        # 断言梯度值是否正确
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)

        # 接下来对密集张量进行操作
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        # 计算并累加三个函数应用的结果，对结果取绝对值并进行反向传播
        (
            fn.apply(x, dense_grad)
            + fn.apply(x, sparse_grad1)
            + fn.apply(x, sparse_grad2)
        ).sum().abs().backward()
        # 断言梯度值是否正确
        self.assertEqual(x.grad, dense_grad + sparse_grad1 + sparse_grad2)

        # 最后只对稀疏张量进行操作
        x = torch.randn(size, dtype=dtype, device=device, requires_grad=True)
        # 计算并累加两个函数应用的结果，对结果取绝对值并进行反向传播
        (fn.apply(x, sparse_grad1) + fn.apply(x, sparse_grad2)).sum().abs().backward()
        # 断言梯度值是否正确
        self.assertEqual(x.grad, sparse_grad1 + sparse_grad2)

    # 标记一个测试方法为跳过执行，适用于不支持双精度类型的 MPS 环境
    @skipIfMps
    def test_sparse_mask_autograd(self, device):
        # 创建一个随机张量 tensor，并标记其需要计算梯度
        tensor = torch.randn(3, requires_grad=True, device=device)
        # 创建一个掩码 mask，并设置其中一个元素为 0
        mask = torch.ones(3, device=device)
        mask[1] = 0
        # 将掩码转换为稀疏张量
        mask = mask.to_sparse()
        # 使用稀疏掩码 mask 对 tensor 进行掩码操作，并将结果转换为密集张量
        converted = tensor.sparse_mask(mask).to_dense()
        # 对转换后的结果求和并进行反向传播
        converted.sum().backward()
        # 断言 tensor 的梯度是否与 mask 转换为密集张量后的结果相等
        self.assertEqual(tensor.grad, mask.to_dense())

    # 标记一个测试方法为跳过执行的原因说明，不支持双精度类型的 MPS 环境
    @skipIfMps  # the test doesn't work on MPS as double types are not supported
    # 定义测试函数 test_pyscalar_conversions，接受一个设备参数 device
    def test_pyscalar_conversions(self, device):
        
        # 定义内部函数 _test_pyscalar_conversions，接受两个参数 t 和 integral_conv
        def _test_pyscalar_conversions(t, integral_conv):
            # integral -> integral
            # 创建一个长整型的张量 l，并将其赋值为 -12345
            l = t(torch.zeros(1, 1, 1, dtype=torch.long))
            pyscalar = -12345
            # 将 pyscalar 赋值给张量 l 的第一个元素
            l[0] = pyscalar
            # 断言 integral_conv 函数应用在 l 上的结果等于 pyscalar
            self.assertEqual(integral_conv(l), pyscalar)

            # floating point -> floating point
            # 创建一个双精度浮点型的变量 f，并将其赋值为一个随机数
            f = Variable(t(torch.randn(1, 1, dtype=torch.double)))
            pyscalar = -12345.1
            # 将 pyscalar 赋值给 f 的第一个元素
            f[0] = pyscalar
            # 断言将 f 转换为 float 后结果等于 pyscalar
            self.assertEqual(float(f), pyscalar)
            # 将 f 的第一个元素设置为 nan
            f[0] = nan
            # 断言 math.isnan 返回 True
            self.assertTrue(math.isnan(float(f)))
            # 将 f 的第一个元素设置为 inf
            f[0] = inf
            # 断言将 f 转换为 float 后结果等于 inf
            self.assertEqual(float(f), inf)
            # 将 f 的第一个元素设置为 -inf
            f[0] = -inf
            # 断言将 f 转换为 float 后结果等于 -inf
            self.assertEqual(float(f), -inf)

            # integral -> floating point
            # 检查能否转换会损失精度的整数值 pyscalar
            pyscalar = 1234567890123456789
            # 断言转换后的结果不等于原始的整数值
            self.assertNotEqual(pyscalar, integral_conv(float(pyscalar)))
            # 将 pyscalar 赋值给 l 的第一个元素
            l[0] = pyscalar
            # 断言将 l 转换为 float 后结果等于 pyscalar 转换为 float 后的结果
            self.assertEqual(float(l), float(pyscalar))

            # floating point -> integral
            # 将 f 的第一个元素设置为 nan
            f[0] = nan
            # 断言 lambda 函数调用 integral_conv(f[0]) 会抛出 ValueError 异常
            self.assertRaises(ValueError, lambda: integral_conv(f[0]))
            # 将 f 的第一个元素设置为 inf
            f[0] = inf
            # 断言 lambda 函数调用 integral_conv(f[0]) 会抛出 OverflowError 异常
            self.assertRaises(OverflowError, lambda: integral_conv(f[0]))
            # 将 f 的第一个元素设置为 -inf
            f[0] = -inf
            # 断言 lambda 函数调用 integral_conv(f[0]) 会抛出 OverflowError 异常
            self.assertRaises(OverflowError, lambda: integral_conv(f[0]))
            # 将 f 的第一个元素设置为 sys.float_info.max
            f[0] = sys.float_info.max
            # 断言将 f 转换为 integral 后结果等于 sys.float_info.max
            self.assertEqual(integral_conv(f), sys.float_info.max)

            # bool, nonzero
            # 定义内部函数 test_nonzero，接受 tensor, value, expected 三个参数
            def test_nonzero(tensor, value, expected):
                # 将 tensor 的第一个元素设置为 value
                tensor[0] = value
                # 断言 bool(tensor) 的结果等于 expected
                self.assertEqual(expected, bool(tensor))
                # 断言 True if tensor else False 的结果等于 expected

            # 测试整型张量 l，值为 0 时的布尔值，应为 False
            test_nonzero(l, 0, False)
            # 测试整型张量 l，值为 -2 时的布尔值，应为 True
            test_nonzero(l, -2, True)
            # 测试浮点型变量 f，值为 0.0 时的布尔值，应为 False
            test_nonzero(f, 0.0, False)
            # 测试浮点型变量 f，最小浮点数时的布尔值，应为 True
            test_nonzero(f, sys.float_info.min, True)
            # 测试浮点型变量 f，值为 nan 时的布尔值，应为 True
            test_nonzero(f, nan, bool(nan))
            # 测试浮点型变量 f，值为 inf 时的布尔值，应为 True
            test_nonzero(f, inf, bool(inf))
            # 测试浮点型变量 f，值为 -inf 时的布尔值，应为 True
            test_nonzero(f, -inf, bool(-inf))

        # 调用 _test_pyscalar_conversions 函数，传递 lambda 函数用于将参数转移到指定设备，和 lambda 函数用于整数转换
        _test_pyscalar_conversions(lambda x: x.to(device), lambda x: int(x))

    # 应用 @dtypesIfMPS 装饰器，指定参数为 torch.float32 时生效
    @dtypesIfMPS(torch.float32)
    # 应用 @dtypesIfCUDA 装饰器，指定参数为 torch.half, torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64 时生效
    @dtypesIfCUDA(
        torch.half,
        torch.float,
        torch.double,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    # 应用 @dtypes 装饰器，指定参数为 torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64 时生效
    @dtypes(
        torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64
    )
    def test_set_requires_grad_only_for_floats(self, device, dtype):
        # 定义函数 f1，设置张量 a 的值为 1，数据类型为 dtype，存储设备为 device，并标记需要梯度
        def f1():
            a = torch.ones(1, dtype=dtype, device=device)
            a.requires_grad_()

        # 定义函数 f2，设置张量 a 的值为 1，数据类型为 dtype，存储设备为 device，并通过属性方式标记需要梯度
        def f2():
            a = torch.ones(1, dtype=dtype, device=device)
            a.requires_grad = True

        # 定义函数 f3，在创建张量时直接标记需要梯度，设置数据类型为 dtype，存储设备为 device
        def f3():
            torch.ones(1, dtype=dtype, device=device, requires_grad=True)

        # 创建张量 a，设置值为 1，数据类型为 dtype，存储设备为 device，并标记不需要梯度
        a = torch.ones(1, dtype=dtype, device=device)
        a.requires_grad = False  # 总是有效
        a.requires_grad_(False)

        # 遍历函数列表 [f1, f2, f3]
        for f in [f1, f2, f3]:
            # 如果数据类型是浮点数
            if dtype.is_floating_point:
                f()  # 调用函数 f
            else:
                # 否则，使用断言检查 RuntimeError 异常，报错信息包含 "floating point"
                with self.assertRaisesRegex(
                    RuntimeError,
                    "floating point",
                    msg=f"dt: {a.dtype} device: {a.device}",
                ):
                    f()

    @onlyCUDA
    def test_advanced_indexing_backwards_large(self, device):
        # 参考 https://github.com/pytorch/pytorch/issues/22843
        n = 1 << 16
        # 创建形状为 (n, 1) 的随机张量 x，需要梯度，存储设备为 device
        x = torch.rand(n, 1, device=device, requires_grad=True)
        # 使用高级索引获取 x 的第一个维度的所有元素，保留维度，得到张量 a
        a = x[:, [0]]
        # 对张量 a 的所有元素求和并反向传播梯度
        a.sum().backward()
        # 断言 x 的梯度与形状为 (n, 1) 的张量全为 1
        self.assertEqual(x.grad, torch.ones(n, 1, device=device))

    def test_advanced_indexing_backwards_memory_format(self, device):
        # 参考 https://github.com/pytorch/pytorch/issues/36956
        shape = (2, 8, 1, 2)
        # 创建形状为 shape 的随机张量 i，元素值为在 [1, shape) 范围内随机整数，连续内存格式为通道为最后
        i = torch.randint(1, shape, device=device).contiguous(
            memory_format=torch.channels_last
        )
        # 创建形状为 shape 的随机张量 x，需要梯度，存储设备为 device
        x = torch.randn(shape, requires_grad=True, device=device)
        # 使用高级索引根据张量 i 获取 x 的子集，求和并反向传播梯度
        x[i].sum().backward()

    def _test_reentrant_parent_error_on_cpu(self, device):
        # 创建形状为 [3, 3] 的随机张量 t1，需要梯度
        t1 = torch.rand([3, 3], requires_grad=True)
        # 创建形状为 [3, 3] 的随机张量 t2，存储设备为 device，需要梯度
        t2 = torch.rand([3, 3], device=device, requires_grad=True)
        # 创建形状为 [3, 3] 的随机张量 t3，存储设备为 device，需要梯度

        # 创建 t1 的平方作为 t4
        t4 = t1 * t1
        # 调用自定义函数 SimulateBackwardError 的静态方法 apply 处理 t4，返回 t5
        t5 = TestAutograd.SimulateBackwardError.apply(t4)

        # 创建 t2 的平方作为 prev
        prev = t2 * t2
        # 多次迭代更新 prev 的值，通过乘法
        for i in range(10):
            prev = prev * t2
        # 将 prev 赋值给 reentrant_root
        reentrant_root = prev

        class ReentrantFunc(Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, grad):
                # 在子图中进行递归反向传播，处理时间较长
                reentrant_root.backward()
                return grad

        # 调用 ReentrantFunc 的静态方法 apply 处理 t3，返回 t6
        t6 = ReentrantFunc.apply(t3)
        # t6 的平方赋值给 t7
        t7 = t6 * t6

        # 在父图中执行反向传播，捕获 "Simulate error" 异常
        with self.assertRaisesRegex(Exception, "Simulate error"):
            torch.autograd.backward([t5.sum(), t7.sum()])

        # 检查 t2、t1、t3 的梯度是否为 None
        self.assertIsNone(t2.grad)
        self.assertIsNone(t1.grad)
        self.assertIsNone(t3.grad)

    @onlyCUDA
    # 在 CPU 上测试可重入父级错误处理函数
    def test_reentrant_parent_error_on_cpu(self, device):
        # 定义函数：获取 CUDA 内存使用情况
        def _get_cuda_memory_usage():
            # 不需要 CUDA 同步，因为统计信息不是在实际释放时跟踪的，
            # 而是在将块标记为自由时跟踪的。
            num_devices = torch.cuda.device_count()
            gc.collect()  # 执行垃圾收集
            return tuple(torch.cuda.memory_allocated(i) for i in range(num_devices))

        before = _get_cuda_memory_usage()  # 获取测试前的 CUDA 内存使用情况

        # 运行作为独立函数，以便在检查内存使用时，gc 能清理一切。
        self._test_reentrant_parent_error_on_cpu(device)

        # 等待自动求导线程清理失败的任务。
        after = _get_cuda_memory_usage()  # 获取测试后的 CUDA 内存使用情况
        start = time.time()
        while before != after and time.time() - start < 30:
            time.sleep(0.1)
            after = _get_cuda_memory_usage()

        self.assertEqual(before, after)  # 断言前后 CUDA 内存使用情况一致

    @skipIfMps  # 测试在 MPS 上不工作
    # TODO: 看看这些测试是否可以移植到 OpInfos 或移动到其他测试套件的位置
    def test_where_functional(self, device):
        # 创建随机张量 x 和 y，要求梯度
        x = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        # 生成条件张量，不全为零的掩码，并移动到指定设备
        cond = mask_not_all_zeros((5, 5)).to(device=device)

        # 定义 where 函数
        def where(cond, x, y):
            return torch.where(cond, x, y)

        # 检查 where 函数的梯度
        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [torch.randn(5, 5, device=device)])

        # 修改 x 和 y 的形状并再次检查梯度
        x = torch.randn(5, 1, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.randn(5, 5, 1, dtype=torch.double, device=device, requires_grad=True)
        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [torch.randn(5, 5, 5, device=device)])

    @skipIfMps  # 测试在 MPS 上不工作
    def test_where_scalar(self, device):
        # 创建随机张量 x，定义标量 scalar，并生成条件张量的掩码
        x = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        scalar = 4.0
        cond = mask_not_all_zeros((5, 5)).to(device=device)

        # 定义 where_scalar_first 函数
        def where_scalar_first(cond, x):
            return torch.where(cond, scalar, x)

        # 定义 where_scalar_second 函数
        def where_scalar_second(cond, x):
            return torch.where(cond, x, scalar)

        # 检查 where_scalar_first 函数的梯度
        gradcheck(where_scalar_first, (cond, x))
        gradgradcheck(where_scalar_first, (cond, x))

        # 检查 where_scalar_second 函数的梯度
        gradcheck(where_scalar_second, (cond, x))
        gradgradcheck(where_scalar_second, (cond, x))

    @onlyCUDA
    # 测试函数，用于验证释放不需要的张量时的内存行为
    def test_free_unneeded_tensor(self, device):
        # 创建一个形状为 (2, 3, 10, 10) 的张量 x，设备为指定的 device，要求梯度计算
        x = torch.randn(2, 3, 10, 10, device=device, requires_grad=True)
        # 创建一个形状为 (1, 3, 1, 1) 的张量 m，设备同样为指定的 device
        m = torch.randn(1, 3, 1, 1, device=device)

        # 计算张量 x 的所有元素之和，并保存在变量 z 中
        z = x.sum()
        # 记录当前 CUDA 设备上的内存分配情况
        base_mem = torch.cuda.memory_allocated()
        # 计算表达式 ((x + 2) * m) 的所有元素之和，并保存在变量 z 中
        z = ((x + 2) * m).sum()
        # 记录计算完成后 CUDA 设备上的内存分配情况
        end_mem = torch.cuda.memory_allocated()

        # 断言：在计算完成后，内存使用应该保持不变，
        # 因为 (x + 2) 和 ((x + 2) * m) 都不应该保留用于反向传播的数据，
        # 而之前分配的 z 的大小与当前分配的 z 的大小应相同。
        self.assertEqual(base_mem, end_mem)

    # 仅在 CUDA 设备上运行的测试函数
    @onlyCUDA
    def test_pin_memory(self, device):
        # 创建一个形状为 (2, 2)、数据类型为 double 的张量 x，要求梯度计算
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        
        # 断言：张量 x 和 x.pin_memory() 应该是相等的
        self.assertEqual(x, x.pin_memory())
        # 断言：张量 x 和 x.pin_memory() 不应该是同一个对象
        self.assertIsNot(x, x.pin_memory())
        # 断言：x.pin_memory() 应该要求梯度计算
        self.assertTrue(x.pin_memory().requires_grad)
        
        # 对 x 进行梯度检查
        gradcheck(lambda x: x.pin_memory(), [x])
        # 对 x 进行二阶梯度检查
        gradgradcheck(lambda x: x.pin_memory(), [x])

    # 仅在 CUDA 设备上运行的测试函数
    @onlyCUDA
    def test_profiler_emit_nvtx(self, device):
        # 这个测试不旨在确保 nvtx 范围的正确性。
        # 那需要更复杂的东西（你需要在子进程中创建一个 profile，打开它，并以某种方式解析 SQL）。
        # 这个测试只是为了捕捉 emit_nvtx 在构建时是否出现问题。
        a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
        
        # 使用 CUDA profiler 进行性能分析
        with torch.cuda.profiler.profile():
            # 使用 emit_nvtx 进行代码区段的 NVTX（NVIDIA Tools Extension）标记
            with emit_nvtx():
                a.add(1.0)

    # 仅在 CUDA 设备上运行的测试函数
    @onlyCUDA
    def test_rnn_backward_to_input_but_not_parameters(self, device):
        # 这个检查是否可能不需要权重参数，但需要输入的情况，参见 issue #7722
        l = torch.nn.LSTM(2, 3).to(device)
        for p in l.parameters():
            p.requires_grad = False
        
        # 创建一个形状为 (1, 1, 2) 的张量 s，设备为指定的 device，要求梯度计算
        s = torch.randn(1, 1, 2, requires_grad=True, device=device)
        # 将张量 s 输入到 LSTM 中
        out, _ = l(s)
        # 对输出 out 的所有元素求和，并进行反向传播
        out.sum().backward()
        
        # 断言：张量 s 的梯度不能是 None，且其绝对值之和不应为 0
        self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    # 当 ITT 可用时才执行的测试函数
    @unittest.skipIf(not torch.profiler.itt.is_available(), "ITT is required")
    def test_profiler_emit_itt(self, device):
        # 这个测试不旨在确保 itt 范围的正确性。
        # 那需要更复杂的东西（你需要在子进程中创建一个 profile，打开它，并以某种方式解析 SQL）。
        # 这个测试只是为了捕捉 emit_itt 在构建时是否出现问题。
        a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
        
        # 使用 emit_itt 进行代码区段的 Intel ITT（Intel Trace Technology）标记
        with emit_itt():
            a.add(1.0)

    # 跳过 Mps 的测试装饰器，条件是 randn 不支持 long 类型
    @skipIfMps
    @deviceCountAtLeast(1)
    # 测试梯度赋值，使用给定的设备
    def test_grad_assignment(self, devices):
        # 创建一个形状为 (5, 5) 的张量 x，使用 devices 中的第一个设备
        x = torch.randn(5, 5, device=devices[0])

        # 测试错误类型的赋值是否会引发 TypeError 异常
        with self.assertRaisesRegex(TypeError, "expected to be a Tensor or None"):
            x.grad = 0

        # 测试错误形状的赋值是否会引发 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            x.grad = torch.randn(2, 2, device=devices[0])

        # 测试错误数据类型的赋值是否会引发 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            x.grad = torch.randn(5, 5, dtype=torch.long, device=devices[0])

        # 测试自我赋值是否会引发 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            x.grad = x

        # 如果当前设备类型不是 "cpu"，测试设备到 CPU 的梯度赋值是否会引发 RuntimeError 异常
        if self.device_type != "cpu":
            with self.assertRaises(RuntimeError):
                t_cpu = torch.rand(5, 5)
                t_cpu.grad = torch.randn(5, 5, device=devices[0])

        # 如果设备类型是 "cuda"，将张量 x 转换为半精度浮点型，并用全零张量赋值其梯度
        if self.device_type == "cuda":
            x = x.to(dtype=torch.half, device=devices[0])
            x.grad = torch.zeros_like(x)

        # 如果设备数量大于 1，测试跨设备赋值是否会引发 RuntimeError 异常
        if len(devices) > 1:
            x = torch.randn(5, 5, device=devices[0])
            with self.assertRaises(RuntimeError):
                x.grad = torch.randn(5, 5, device=devices[1])
    # 测试函数，验证反向传播设备匹配性
    def test_backward_device(self, devices):
        # 设备变量，用于存储梯度输出设备信息
        device = [None]

        # 自定义的身份函数，继承自torch.autograd.Function
        class Identity(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 前向传播函数，返回输入张量的克隆
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                # 反向传播函数，记录梯度输出的设备信息并返回梯度的克隆
                device[0] = grad_output.device
                return grad_output.clone()

        # 创建一个随机张量v，并应用自定义的身份函数，然后进行反向传播
        v = torch.randn(1, device=devices[1], requires_grad=True)
        Identity.apply(v).backward()
        # 断言当前设备与预期设备相符
        self.assertEqual(str(device[0]), devices[1])

    # 装饰器指定至少有两个设备的测试函数
    @deviceCountAtLeast(2)
    def test_inputbuffer_add_multidevice(self, devices):
        # 创建一个在指定设备上的随机输入张量
        input = torch.randn(1, device=devices[0], requires_grad=True)
        # 在不同设备上执行张量加法，并记录输出
        output = input.to(device=devices[1]) + input.to(device=devices[1])
        # 对输出执行反向传播
        output.backward()

    # 仅在CPU上执行的测试函数
    @onlyCPU
    def test_copy_(self, device):
        # 创建一个在指定设备上的随机张量，并确保其需要梯度
        x = torch.randn(10, device=device, requires_grad=True)
        # 定义浮点类型，包括torch.half和torch.bfloat16
        floating_dt = floating_types_and(torch.half, torch.bfloat16)
        # 针对每种浮点类型，创建一个新的张量y，并使用x的值进行复制
        for dt in floating_dt:
            y = torch.empty(10, device=device, dtype=dt)
            y.copy_(x)
            # 断言新张量y需要梯度
            self.assertTrue(y.requires_grad)
            # 将x转换为torch.bfloat16类型，并确保其需要梯度
            z = x.to(torch.bfloat16)
            self.assertTrue(z.requires_grad)

    # 测试复制和前向自动微分处理不同形状的张量
    def test_copy_forward_ad_broadcasting(self, device):
        # 创建原始张量和切线张量，以及非双重的张量
        primal = torch.rand(3, 3, device=device)
        tangent = torch.rand(3, 3, device=device)
        non_dual = torch.rand(1, 3, 3, device=device)

        # 进入前向自动微分的双重级别
        with fwAD.dual_level():
            # 创建双重张量dual，并将其复制到非双重张量non_dual
            dual = fwAD.make_dual(primal, tangent)
            non_dual.copy_(dual)

    # 测试复制和前向自动微分处理相同布局的张量，复制时不应复制梯度
    def test_copy_forward_ad_same_layout_copies_grad(self, device):
        # 创建原始张量和切线张量
        primal = torch.tensor([[3.0], [4.0]], device=device)
        tangent = torch.tensor([[5.0], [6.0]], device=device)

        # 进入前向自动微分的双重级别
        with fwAD.dual_level():
            # 创建双重张量x_dual，并创建一个非双重的张量non_dual
            x_dual = fwAD.make_dual(primal, tangent)
            non_dual = torch.tensor([[1.0], [2.0]])
            # 将双重张量x_dual的值复制到非双重张量non_dual，确保不复制梯度
            non_dual.copy_(x_dual)
            # 断言非双重的non_dual张量的切线不是初始切线张量tangent
            self.assertTrue(fwAD.unpack_dual(non_dual).tangent is not tangent)
    # 定义一个测试方法，用于测试简单的可重入跨设备操作
    def test_simple_reentrant_cross_device(self, device):
        # 定义一个继承自Function的可重入函数类ReentrantFunc
        class ReentrantFunc(Function):
            _cpu_mode = True  # 静态变量，表示当前模式是否为CPU模式

            @staticmethod
            # 前向传播函数，计算输入张量x的平方乘以(x+2)
            def forward(ctx, x):
                return x * (x + 2)

            @staticmethod
            # 反向传播函数，计算梯度
            def backward(ctx, grad_output):
                # 开启梯度计算上下文
                with torch.enable_grad():
                    if ReentrantFunc._cpu_mode:
                        # 如果是CPU模式，则创建一个在CPU上的新参数张量，要求梯度
                        new_param = torch.randn(2, 2, requires_grad=True)
                        # 对新参数的平方求和并进行反向传播
                        (new_param**2).sum().backward()
                    else:
                        # 如果是GPU模式，则创建一个在指定设备上的新参数张量，要求梯度
                        new_param = torch.randn(2, 2, device=device, requires_grad=True)
                        # 对新参数的平方求和并进行反向传播
                        (new_param**2).sum().backward()
                return grad_output

        # 第一组测试：在GPU线程上开始并结束
        x = torch.randn(2, 2, device=device, requires_grad=True)
        out = ReentrantFunc.apply(x)  # 调用ReentrantFunc的前向传播
        out.sum().backward()  # 对输出进行求和并进行反向传播

        # 第二组测试：在CPU线程上开始，然后切换到GPU线程结束
        x = torch.randn(2, 2, requires_grad=True)
        # 将ReentrantFunc节点设置为GPU模式，以便将任务提交到GPU队列
        ReentrantFunc._cpu_mode = False
        out = ReentrantFunc.apply(x)
        out.sum().backward()

        # 第三组测试：在GPU线程上开始，然后切换到CPU线程结束
        x = torch.randn(2, 2, device=device, requires_grad=True)
        # 将ReentrantFunc节点设置为CPU模式，以便将任务提交到CPU队列
        ReentrantFunc._cpu_mode = True
        out = ReentrantFunc.apply(x)
        out.sum().backward()

    @onlyCUDA
    # 定义一个测试方法，用于测试跨设备的可重入自动求导
    def test_cross_device_reentrant_autograd(self, device):
        # 在GPU上执行函数，确保此任务关联到GPU线程
        def fn_on_gpu(inp):
            # 人为增加下一个操作的优先级，确保在到达它之前就运行
            dummy = inp * 2 * 2 * 2 * 2
            return inp.to(device=device)  # 将输入张量移到指定设备

        # 在CPU上执行父级操作
        def parent_on_cpu(inp):
            # 在GPU上执行的操作分支，以使GPU线程的工作队列不会太快清空
            branch1 = inp.to(device=device)
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            # 对CPU张量执行检查点，确保可重入自动求导的最后一个操作是在CPU线程上运行的累积梯度
            branch2 = checkpoint(fn_on_gpu, inp, use_reentrant=True)
            out = branch2 + branch1
            return out

        inp = torch.rand(2, requires_grad=True)
        out = parent_on_cpu(inp)  # 执行父级操作
        out.sum().backward()  # 对输出进行求和并进行反向传播
    def test_inplace_on_view_backprop_base(self, device):
        # modify view and back-prop through base
        # 创建一个随机张量 `root`，形状为 (2, 2)，在指定设备上，开启梯度追踪
        root = torch.randn(2, 2, device=device, requires_grad=True)
        # 克隆张量 `root`，得到张量 `x`
        x = root.clone()
        # 通过 narrow 方法创建视图 `v1`，选择第 0 维的前 1 个元素
        v1 = x.narrow(0, 0, 1)
        # 将视图 `v1` 中的元素乘以 2（原地操作）
        v1.mul_(2)
        # 对张量 `x` 的所有元素求和，并进行反向传播
        x.sum().backward()
        # 断言 `root` 的梯度与预期值相同
        self.assertEqual(root.grad.tolist(), [[2, 2], [1, 1]])

    def test_inplace_on_view_backprop_view_of_view(self, device):
        # modify view and backprop through view-of-view
        # 创建一个随机张量 `root`，形状为 (2, 2)，在指定设备上，开启梯度追踪
        root = torch.randn(2, 2, device=device, requires_grad=True)
        # 克隆张量 `root`，得到张量 `x`
        x = root.clone()
        # 通过 narrow 方法创建视图 `v1`，选择第 0 维的前 1 个元素
        v1 = x.narrow(0, 0, 1)
        # 再次通过 narrow 方法创建另一个视图 `v2`，选择第 0 维的前 1 个元素
        v2 = x.narrow(0, 0, 1)
        # 将视图 `v1` 中的元素乘以 2（原地操作）
        v1.mul_(2)
        # 对视图 `v2` 的所有元素求和，并进行反向传播
        v2.sum().backward()
        # 断言 `root` 的梯度与预期值相同
        self.assertEqual(root.grad.tolist(), [[2, 2], [0, 0]])

    def test_inplace_on_view_of_view(self, device):
        # modify view-of-view and backprop through base
        # 创建一个随机张量 `root`，形状为 (2, 2)，在指定设备上，开启梯度追踪
        root = torch.randn(2, 2, device=device, requires_grad=True)
        # 克隆张量 `root`，得到张量 `x`
        x = root.clone()
        # 通过 narrow 方法创建视图 `v1`，选择第 0 维的前 1 个元素
        v1 = x.narrow(0, 0, 1)
        # 再通过 narrow 方法在 `v1` 上选择第 1 维的前 1 个元素，创建视图 `v2`
        v2 = v1.narrow(1, 1, 1)
        # 将视图 `v2` 中的元素乘以 2（原地操作）
        v2.mul_(2)
        # 对张量 `x` 的所有元素求和，并进行反向传播
        x.sum().backward()
        # 断言 `root` 的梯度与预期值相同
        self.assertEqual(root.grad.tolist(), [[1, 2], [1, 1]])

    @skipIfMps  # the test doesn't work on MPS as double types are not supported
    def test_inplace_on_view_then_no_grad(self, device):
        # Perform an in-place operation on a view of a non-leaf variable.
        # 在非叶子变量的视图上执行原地操作
        a = torch.ones(3, 1, dtype=torch.double, device=device, requires_grad=True)
        # 将张量 `a` 乘以 2，得到张量 `b`
        b = a * 2
        # 将张量 `b` 视图化为自身
        c = b.view_as(b)
        # 修改视图 `c` 的第一个元素为 3
        c[0][0] = 3

        # 强制进行梯度图更新，但梯度追踪被禁用
        with torch.no_grad():
            c.grad_fn

        # 对视图 `c` 的所有元素求和，并进行反向传播
        c.sum().backward()

    @skipIfMps  # the test doesn't work on MPS as double types are not supported
    def test_inplace_on_view_gradcheck(self, device):
        # gradcheck modifications to views
        # 创建两个随机张量 `a` 和 `b`，在指定设备上，开启梯度追踪
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=True)
        b = torch.randn(2, 2, dtype=torch.double, device=device, requires_grad=True)

        def func(root, b):
            # 克隆张量 `root`，得到张量 `x`
            x = root.clone()
            # 在张量 `x` 上执行两次 narrow 操作，并对选定区域乘以张量 `b`（原地操作）
            x.narrow(1, 2, 2).narrow(0, 1, 2).mul_(b)
            x.narrow(1, 0, 2).narrow(0, 1, 2).mul_(b)
            return x

        # 检查梯度是否正确计算
        gradcheck(func, [a, b], raise_exception=True)
        # 创建与张量 `a` 相同形状的随机张量 `go`，在指定设备上，开启梯度追踪
        go = torch.randn(
            a.size(), dtype=torch.double, device=device, requires_grad=True
        )
        # 检查二阶梯度是否正确计算
        gradgradcheck(func, (a, b), (go,))

    def test_inplace_on_view_multiple_outputs(self, device):
        # 创建一个从 0 到 8 的张量 `root`，形状为 (3, 3)，开启梯度追踪
        root = torch.arange(9.0, dtype=torch.double).reshape(3, 3).requires_grad_()
        # 克隆张量 `root`，得到张量 `x`
        x = root.clone()
        # 使用 unbind 方法分离张量 `x` 的所有维度
        v1 = x.unbind()
        # 期望抛出 RuntimeError 异常，因为试图在非叶子变量的视图上执行原地操作
        with self.assertRaises(RuntimeError):
            v1[0].mul_(2)

    @skipIfMps  # the test doesn't work on MPS as double types are not supported
    def test_inplace_on_view_of_multiple_output_view(self, device):
        # 创建一个随机张量 `a`，形状为 (10,)，在指定设备上，开启梯度追踪
        a = torch.rand(
            10, dtype=torch.double, device=device, requires_grad=True
        ).clone()
        # 使用 unbind 方法分离张量 `a` 的第一个维度
        b = a.unbind(0)
        # 使用 view_as 方法创建张量 `c`，视图化 `b` 的第一个元素
        c = b[0].view_as(b[0])
        # 期望抛出 RuntimeError 异常，因为试图在非叶子变量的视图上执行原地操作
        with self.assertRaises(RuntimeError):
            c.mul_(2)
    @skipIfMps  # MPS backend doesn't support double types
    # 测试函数：测试在视图的视图上进行多输出的原地操作
    def test_inplace_multiple_output_view_of_view(self, device):
        # 创建一个形状为 (10,) 的张量 a，数据类型为 double，存储在指定设备上，允许梯度计算
        a = torch.rand(10, dtype=torch.double, device=device, requires_grad=True).clone()
        # 创建张量 b，与张量 a 的视图相同
        b = a.view_as(a)
        # 对 b 进行解绑定操作，返回张量列表 c
        c = b.unbind(0)
        # 使用断言确保运行时错误抛出异常：不允许在原地操作视图上进行乘法操作
        with self.assertRaises(RuntimeError):
            c[0].mul_(2)

    @skipIfMps  # MPS backend doesn't support double types
    # 测试函数：测试在视图上进行原地操作会使基础张量需要梯度
    def test_inplace_on_view_makes_base_require_grad(self, device):
        # 创建形状为 (4, 4) 的张量 a，数据类型为 double，存储在指定设备上，不允许梯度计算
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=False)
        # 创建形状为 (4, 2) 的张量 b，数据类型为 double，存储在指定设备上，允许梯度计算
        b = torch.randn(4, 2, dtype=torch.double, device=device, requires_grad=True)

        # 定义内部函数 func，用于操作根张量 root 和张量 b
        def func(root, b):
            # 克隆根张量 root
            x = root.clone()
            # 断言：x 不需要梯度
            self.assertFalse(x.requires_grad)
            # 对 x 进行狭窄操作，从索引 2 开始的 2 个元素，进行原地乘法操作
            x.narrow(1, 2, 2).mul_(b)
            # 断言：x 需要梯度
            self.assertTrue(x.requires_grad)
            return x

        # 使用 gradcheck 函数检查 func 函数在给定输入 a 和 b 上的梯度是否正确，抛出异常以便调试
        gradcheck(func, [a, b], raise_exception=True)
        # 创建形状与 a 相同的张量 go，数据类型为 double，存储在指定设备上，允许梯度计算
        go = torch.randn(a.size(), dtype=torch.double, device=device, requires_grad=True)
        # 使用 gradgradcheck 函数检查 func 函数在给定输入 a 和 b 上的梯度是否正确，同时检查二阶梯度
        gradgradcheck(func, (a, b), (go,))

    # 测试函数：测试在视图上进行原地操作，并通过视图进行反向传播
    def test_inplace_on_view_backprop_view(self, device):
        # 创建张量 a，包含值 [2.0, 5.0]，存储在指定设备上，不允许梯度计算
        a = torch.tensor([2.0, 5.0], device=device, requires_grad=False)
        # 创建张量 b，包含值 [3.0]，存储在指定设备上，允许梯度计算
        b = torch.tensor([3.0], device=device, requires_grad=True)
        # 在张量 a 的视图上进行狭窄操作，从索引 1 开始的 1 个元素，进行原地乘法操作
        res = a.narrow(0, 1, 1).mul_(b)
        # 计算 res 的和，并执行反向传播
        res.sum().backward()
        # 使用断言检查 b 的梯度是否正确计算为 [5]
        self.assertEqual(b.grad.tolist(), [5])
        # 使用断言检查 a 的梯度是否为 None
        self.assertIsNone(a.grad)

    @skipIfMps  # MPS backend doesn't support double types
    # 测试函数：测试在视图上进行原地操作会修改基础张量
    def test_inplace_on_view_modify_base(self, device):
        # 创建一个形状为 (1,) 的张量 r，数据类型为 double，存储在指定设备上，允许梯度计算
        r = torch.ones(1, dtype=torch.double, device=device, requires_grad=True)

        # 定义内部函数 fn，用于操作张量 r
        def fn(r):
            # 创建一个形状为 (5,) 的张量 x，数据类型为 double，存储在指定设备上
            x = torch.ones(5, dtype=torch.double, device=device)
            # 选择张量 x 的第 1 维度的第 1 个元素作为视图 v
            v = x.select(0, 1)
            # 断言：v 不需要梯度，且梯度函数为 None
            self.assertFalse(v.requires_grad)
            self.assertIsNone(v.grad_fn)
            # 在张量 x 上进行原地加法操作，使视图 v 依赖于张量 r
            x.add_(r)
            # 断言：v 需要梯度
            self.assertTrue(v.requires_grad)
            return v

        # 使用 gradcheck 函数检查 fn 函数在给定输入 r 上的梯度是否正确
        gradcheck(fn, [r])
        # 使用 gradgradcheck 函数检查 fn 函数在给定输入 r 上的梯度是否正确，并检查二阶梯度
        gradgradcheck(fn, [r])
    # 在 Python-autograd 创建的视图上进行原地修改测试
    def test_inplace_on_view_python(self, device):
        # 创建具有梯度的随机张量 a 和 b
        a = torch.randn(4, 4, dtype=torch.double, device=device, requires_grad=True)
        b = torch.randn(2, 2, dtype=torch.double, device=device, requires_grad=True)

        # 定义一个自定义的 PyAdd 函数，用于张量的原地加法操作
        class PyAdd(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                # 标记输入张量 x 为脏（dirty），需要重新计算梯度
                ctx.mark_dirty(x)
                # 原地修改 x，加上 y
                x.add_(y)
                return x

            @staticmethod
            def backward(ctx, grad):
                # 反向传播时直接返回梯度 grad
                return grad, grad

        # 定义一个测试函数 func，对张量 root 的视图进行原地加法操作
        def func(root, b):
            x = root.clone()  # 克隆张量 root
            # 对 x 的部分视图进行 PyAdd 的原地加法操作
            PyAdd.apply(x.narrow(1, 2, 2).narrow(0, 1, 2), b)
            PyAdd.apply(x.narrow(1, 0, 2).narrow(0, 1, 2), b)
            return x

        # 使用 gradcheck 函数检查 func 的梯度计算是否正确
        gradcheck(func, [a, b], raise_exception=True)

        # 创建一个与 a 相同大小的随机张量 go，用于 gradgradcheck 函数测试
        go = torch.randn(
            a.size(), dtype=torch.double, device=device, requires_grad=True
        )
        gradgradcheck(func, (a, b), (go,))

    # 在非连续视图上进行原地修改测试
    def test_inplace_on_view_non_contig(self, device):
        # 创建一个具有梯度的张量 root，其形状为 [2, 2, 2]
        root = torch.ones(2, 3, 2, device=device).select(2, 1).t().requires_grad_(True)
        x = root.clone()  # 克隆张量 root
        v1 = x.narrow(0, 0, 1)  # 获取 x 的一个非连续视图 v1
        v2 = v1.narrow(1, 1, 1)  # 获取 v1 的一个非连续视图 v2
        v2.mul_(2)  # 对 v2 进行原地乘法操作
        x.sum().backward()  # 对 x 求和并进行反向传播
        self.assertEqual(root.grad.tolist(), [[1, 2], [1, 1], [1, 1]])

    # 在多输出非安全视图上进行原地修改测试
    def test_inplace_on_view_multi_output_unsafe(self, device):
        # 遍历多个函数对张量 b 进行非安全分割和分块操作的测试
        for f in [
            lambda t: t.unsafe_split(1),
            lambda t: t.unsafe_split_with_sizes((1, 1, 1)),
            lambda t: t.unsafe_chunk(3),
        ]:
            a = torch.randn(3, 3, device=device, requires_grad=True)  # 创建具有梯度的随机张量 a
            b = a + a  # 创建张量 b，作为 a 的副本
            s1, s2, s3 = f(b)  # 对 b 应用函数 f，获取多个输出视图 s1, s2, s3
            s1.mul_(s2)  # 对 s1 进行原地乘法操作
            s1.sum().backward()  # 对 s1 求和并进行反向传播

    # 在多输出安全视图上进行原地修改测试
    def test_inplace_on_view_multi_output_safe(self, device):
        # 遍历多个函数对张量 b 进行安全分割和分块操作的测试
        for f in [
            lambda t: t.split(1),
            lambda t: t.split_with_sizes((1, 1, 1)),
            lambda t: t.chunk(3),
        ]:
            a = torch.randn(3, 3, device=device, requires_grad=True)  # 创建具有梯度的随机张量 a
            b = a + a  # 创建张量 b，作为 a 的副本
            s1, s2, s3 = f(b)  # 对 b 应用函数 f，获取多个输出视图 s1, s2, s3
            error_msg = (
                "This view is the output of a function that returns multiple views."
            )
            with self.assertRaisesRegex(RuntimeError, error_msg):
                s1.mul_(s2)  # 试图在安全视图上进行原地乘法操作，应该引发 RuntimeError

    # 在未定义梯度输出的视图上进行原地修改测试
    def test_inplace_on_view_undefined_grad_output(self, device):
        a = torch.tensor([1.0], requires_grad=True)  # 创建具有梯度的张量 a
        c = a.clone()  # 克隆张量 a 为 c
        v = c[:]  # 获取 c 的视图 v
        b = torch.tensor(1.0, requires_grad=True)  # 创建具有梯度的张量 b

        # 定义一个 InplaceFunc 函数，对输入张量 x 和其他张量进行原地乘法操作
        class InplaceFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, other):
                ctx.mark_dirty(x)  # 标记输入张量 x 为脏（dirty），需要重新计算梯度
                return x.mul_(2)  # 对 x 进行原地乘法操作，返回结果

            @staticmethod
            def backward(ctx, grad):
                return grad * 2, None  # 反向传播时返回梯度 grad 的两倍，其他输入梯度为 None

        out = InplaceFunc.apply(v, b)  # 对 v 和 b 应用 InplaceFunc 函数
        out.backward()  # 对 out 进行反向传播
        self.assertIsNone(b.grad)  # 检查张量 b 的梯度是否为 None
        self.assertEqual(a.grad.item(), 2)  # 检查张量 a 的梯度是否为 2
    @skipIfMps  # 如果在MPS上测试会跳过，因为双精度类型不支持
    def test_mv_grad_stride_0(self, device):
        # 参考链接：https://github.com/pytorch/pytorch/issues/38315
        # 创建一个随机的双精度张量 mat，形状为 (2, 2)，在指定设备上
        mat = torch.randn(2, 2, dtype=torch.double, device=device)
        # 创建一个随机的双精度张量 vec，形状为 (1,)，并启用梯度追踪
        vec = torch.randn(1, dtype=torch.double, device=device).requires_grad_(True)

        def fn(vec):
            # 在函数内部展开 vec，以确保传递给 gradcheck 的输入没有重叠的内存
            vec = vec.expand(2)
            return (mat @ vec).sum()

        # 检查 fn 函数的梯度是否正确计算
        gradcheck(fn, (vec))
        # 检查 fn 函数的二阶梯度是否正确计算
        gradgradcheck(fn, (vec))

    @onlyCUDA
    def test_gradcheck_input_output_different_device(self, device):
        # 创建一个双精度张量 x，形状为 (1,)，在 CUDA 设备上，并启用梯度追踪
        x = torch.ones((1,), dtype=torch.double, device="cuda", requires_grad=True)
        # 检查将 x 转移到 CPU 后梯度检查是否正确
        gradcheck(lambda x: x.to("cpu"), (x,))

        # 创建一个双精度张量 x，形状为 (1,)，在 CPU 上，并启用梯度追踪
        x = torch.ones((1,), dtype=torch.double, device="cpu", requires_grad=True)
        # 检查将 x 转移到 CUDA 设备后梯度检查是否正确
        gradcheck(lambda x: x.to("cuda"), (x,))

    def test_strided_leaf_grad_layout(self, device):
        # (1) 如果叶子张量是非重叠且密集的，梯度的布局应该与其叶子张量相匹配。
        for fmt_a in (torch.contiguous_format, torch.channels_last):
            for fmt_b in (torch.contiguous_format, torch.channels_last):
                # 创建形状为 (2, 3, 4, 5) 的随机张量 a 和 b，设备为指定设备，并按给定的内存格式转换
                a = torch.rand((2, 3, 4, 5), device=device).to(memory_format=fmt_a)
                b = torch.rand((2, 3, 4, 5), device=device).to(memory_format=fmt_b)
                a.requires_grad_()
                b.requires_grad_()
                # 对 a 和 b 求和并进行反向传播
                a.sum().backward()
                # 断言 a 的梯度步长与其自身步长相同
                self.assertEqual(a.grad.stride(), a.stride())
                b.sum().backward()
                # 断言 b 的梯度步长与其自身步长相同
                self.assertEqual(b.grad.stride(), b.stride())
                # 对 a 和 b 的乘积求和并进行反向传播
                a.grad = None
                b.grad = None
                (a * b).sum().backward()
                # 断言 a 的梯度步长与其自身步长相同
                self.assertEqual(a.grad.stride(), a.stride())
                # 断言 b 的梯度步长与其自身步长相同
                self.assertEqual(b.grad.stride(), b.stride())

        # (2) 如果叶子张量不是密集的，检查梯度是否是行主要连续的。
        c = torch.empty_strided((2, 2), (4, 2), device=device).copy_(
            torch.rand((2, 2), device=device)
        )
        c.requires_grad_()
        d = torch.rand((2, 2), device=device)
        # 对 c 求和并进行反向传播
        c.sum().backward()
        # 断言 c 的梯度步长为 (2, 1)
        self.assertEqual(c.grad.stride(), (2, 1))
        # 对 c 和 d 的乘积求和并进行反向传播
        c.grad = None
        (c * d).sum().backward()
        # 断言 c 的梯度步长为 (2, 1)
        self.assertEqual(c.grad.stride(), (2, 1))
    def test_copy_r_to_c(self, device):
        # 创建一个空的复数张量，形状为 (3, 2)，数据类型为复双精度，放置在指定设备上
        out_c = torch.empty(3, 2, dtype=torch.cdouble, device=device)
        # 创建一个形状为 (3, 2) 的双精度张量，放置在指定设备上，并且需要梯度计算
        inp_r = torch.randn(3, 2, dtype=torch.double, device=device, requires_grad=True)

        def do_test():
            # 将 inp_r 的数据复制到 out_c 中
            out_c.copy_(inp_r)
            # 计算 out_c 中所有元素的和
            out_c_inter = out_c.sum()
            # 计算 out_c_inter 的绝对值，并进行反向传播
            out_c_inter.abs().backward()
            # 使用 torch.no_grad() 上下文管理器，断言梯度计算结果
            with torch.no_grad():
                self.assertEqual(
                    # 断言 inp_r 的梯度，应为 torch.sgn(out_c_inter).real 的值
                    inp_r.grad, torch.ones_like(inp_r) * torch.sgn(out_c_inter).real
                )

        # 调用自定义测试函数，并确保其中没有警告信息
        self.assertNotWarn(do_test)

    def test_to_r_to_c(self, device):
        def do_test():
            # 创建一个形状为 (3, 2) 的双精度张量，放置在指定设备上，并且需要梯度计算
            inp_r = torch.randn(
                3, 2, dtype=torch.double, device=device, requires_grad=True
            )
            # 将 inp_r 转换为复数类型 torch.complex128
            out = inp_r.to(torch.complex128)
            # 计算 out 中所有元素的和
            out_inter = out.sum()
            # 计算 out_inter 的绝对值，并进行反向传播
            out_inter.abs().backward()
            # 使用 torch.no_grad() 上下文管理器，断言梯度计算结果
            with torch.no_grad():
                self.assertEqual(
                    # 断言 inp_r 的梯度，应为 torch.sgn(out_inter).real 的值
                    inp_r.grad, torch.ones_like(inp_r) * torch.sgn(out_inter).real
                )

        # 调用自定义测试函数，并确保其中没有警告信息
        self.assertNotWarn(do_test)

    def test_non_differentiable_ops(self, device):
        # 只需确保操作不会引发错误，并且生成的张量 requires_grad=False
        x = torch.tensor([[1, 2], [3, 4.0]], requires_grad=True, device=device)
        out = torch.isin(x, torch.tensor([2, 3], device=device))
        self.assertFalse(out.requires_grad)

        x = torch.randn(3, 3, requires_grad=True)
        out = torch.signbit(x)
        self.assertFalse(out.requires_grad)

    def test_warning_in_backward(self, device):
        # 测试在反向传播期间警告会作为 Python 警告传播（gh-50209）
        # 注意：对于 device=cuda，警告会从工作线程传播
        a = torch.zeros((), device=device, requires_grad=True)
        b = torch._C._nn._test_warn_in_autograd(a)

        with self.assertWarnsRegex(UserWarning, "Warn from backward"):
            b.backward()

    def test_complex_scalar_backward(self, device):
        a = torch.zeros(1, device=device, requires_grad=True)
        b = a * 0.5j

        msg = "grad can be implicitly created only for real scalar outputs"
        with self.assertRaisesRegex(RuntimeError, msg):
            b.backward()

        with self.assertRaisesRegex(RuntimeError, msg):
            torch.autograd.grad(b, a)

    def test_pow_real_negative_base_complex_exponent(self, device):
        # OpInfo 不自然地支持混合类型的输入，因此在此进行测试
        base = -torch.ones(2, device=device, dtype=torch.double)
        exponent = torch.randn(
            2, device=device, dtype=torch.cdouble, requires_grad=True
        )

        def fn(exponent):
            return torch.pow(base, exponent)

        # 检查梯度
        torch.autograd.gradcheck(fn, (exponent,))

        def fn(exponent):
            return torch.pow(-1, exponent)

        # 检查梯度
        torch.autograd.gradcheck(fn, (exponent,))
    # 定义一个测试方法，用于测试张量的尺寸调整和版本增加
    def test_resize_version_bump(self, device):
        # 创建一个在指定设备上的随机张量 `x`
        x = torch.rand((1,), device=device)
        # 创建一个在指定设备上的标准正态分布张量 `y`
        y = torch.randn((3,), device=device)
        # 调整张量 `x` 的尺寸为 (1, 2)，此操作会增加张量的版本号
        x.resize_((1, 2))
        # 断言张量 `x` 的版本号为 1
        self.assertEqual(x._version, 1)
        # 根据张量 `y` 的尺寸调整张量 `x` 的尺寸，这同样会增加版本号
        x.resize_as_(y)
        # 断言张量 `x` 的版本号为 2
        self.assertEqual(x._version, 2)

        # 在以下情况下，`resize_` 操作是无效的，因此不会增加版本号。
        # 张量 `x` 的尺寸已经是 (3,)，再次调整不会改变尺寸或版本号。
        x.resize_((3,))
        # 断言张量 `x` 的版本号仍然是 2
        self.assertEqual(x._version, 2)

        # 根据张量 `y` 的尺寸再次调整张量 `x` 的尺寸，同样不会增加版本号。
        x.resize_as_(y)
        # 断言张量 `x` 的版本号依然是 2
        self.assertEqual(x._version, 2)
# 定义一个测试类 TestAllowMutationOnSaved，继承自 TestCase
class TestAllowMutationOnSaved(TestCase):
    # 定义一个辅助方法 assertClonedLenEqual，用于断言 ctx.cloned.items() 的长度是否等于 n
    def assertClonedLenEqual(self, ctx, n):
        self.assertEqual(len(list(ctx.cloned.items())), n)

    # 定义一个辅助方法 assertTIDMapLenEqual，用于断言 ctx.tid_to_weakhandle.items() 的长度是否等于 n
    def assertTIDMapLenEqual(self, ctx, n):
        self.assertEqual(len(list(ctx.tid_to_weakhandle.items())), n)

    # 定义一个测试方法 test_basic
    def test_basic(self):
        # 创建一个随机张量 a，设置 requires_grad 为 True
        a = torch.rand(2, 3, requires_grad=True)

        # 定义一个函数 fn，对输入的张量进行操作并返回梯度
        def fn(a):
            b = a.clone()
            out = (b**2).sum()
            b.sin_()
            out.sum().backward()
            return a.grad

        # 设置错误消息
        msg = (
            "variables needed for gradient computation has been modified by an inplace"
        )
        # 断言运行 fn(a) 会抛出 RuntimeError，并且错误消息符合预期
        with self.assertRaisesRegex(RuntimeError, msg):
            fn(a)

        # 使用 allow_mutation_on_saved_tensors 上下文管理器，保存计算图状态
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            da = fn(a)

        # 断言 a 的梯度是正确的
        self.assertTrue(torch.allclose(a * 2, da))
        # 断言 ctx 中的 cloned 长度为 0
        self.assertClonedLenEqual(ctx, 0)

    # 定义一个测试方法 test_views
    def test_views(self):
        # 创建一个随机张量 a，设置 requires_grad 为 True
        a = torch.rand(2, 3, requires_grad=True)

        # 定义一个函数 fn，对输入的张量进行操作并返回梯度
        def fn(a):
            b = a.clone()
            c = b.view_as(b)
            out = (b**2).sum()
            c.sin_()
            out.sum().backward()
            return a.grad

        # 设置错误消息
        msg = (
            "variables needed for gradient computation has been modified by an inplace"
        )
        # 断言运行 fn(a) 会抛出 RuntimeError，并且错误消息符合预期
        with self.assertRaisesRegex(RuntimeError, msg):
            fn(a)

        # 使用 allow_mutation_on_saved_tensors 上下文管理器，保存计算图状态
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            da = fn(a)

        # 断言 ctx 中的 cloned 长度为 0
        self.assertClonedLenEqual(ctx, 0)
        # 断言 a 的梯度是正确的
        self.assertTrue(torch.allclose(a * 2, da))

    # 定义一个测试方法 test_save_base_and_modify_view
    def test_save_base_and_modify_view(self):
        # 使用 allow_mutation_on_saved_tensors 上下文管理器，保存计算图状态
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            # 创建一个随机张量 a，设置 requires_grad 为 True
            a = torch.rand(2, 3, requires_grad=True)
            b = a.clone()
            c = b[:1]
            out = b**2
            # 修改视图
            c *= 10
            out.sum().backward()
            # 断言 ctx 中的 cloned 长度为 0
            self.assertClonedLenEqual(ctx, 0)

        # 断言 ctx 中的 cloned 长度为 0
        self.assertClonedLenEqual(ctx, 0)
        # 断言 a 的梯度是正确的
        self.assertTrue(torch.allclose(a * 2, a.grad))

    # 定义一个测试方法 test_save_view_modify_base
    def test_save_view_modify_base(self):
        # 使用 allow_mutation_on_saved_tensors 上下文管理器，保存计算图状态
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            # 创建一个随机张量 a，设置 requires_grad 为 True
            a = torch.rand(2, 3, requires_grad=True)
            b = a.clone()
            c = b[:]
            out = (c**2).sum()
            b *= 2
            out.backward()
            self.assertTrue(torch.allclose(a * 2, a.grad))

    # 定义一个测试方法 test_double_backward
    def test_double_backward(self):
        # 使用 allow_mutation_on_saved_tensors 上下文管理器，保存计算图状态
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            # 创建一个随机张量 a，设置 requires_grad 为 True
            a = torch.rand(2, 3, requires_grad=True)
            b = a.clone()
            out = (b**2).sum()
            b.sin_()
            # 计算二阶梯度
            torch.autograd.grad(out, a, create_graph=True)
            (da,) = torch.autograd.grad(out, a, create_graph=True)
            (d2a,) = torch.autograd.grad(da.sum(), a)

        # 断言二阶梯度计算结果正确
        self.assertTrue(torch.allclose(torch.ones_like(a) * 2, d2a))
        # 断言 ctx 中的 cloned 长度为 0
        self.assertClonedLenEqual(ctx, 0)
    def test_saved_but_not_anymore(self):
        # 确保在张量曾经保存过但在进行原地操作时不再保存
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            # 创建一个形状为 (2, 3) 的随机张量 a，并克隆它
            a = torch.randn(2, 3, requires_grad=True).clone()
            # 计算张量 a 的平方和，并赋值给 out
            out = (a**2).sum()
            # 断言在上下文中保存的张量映射长度为 1
            self.assertTIDMapLenEqual(ctx, 1)
            # 断言在上下文中克隆的张量长度为 0
            self.assertClonedLenEqual(ctx, 0)
            # 对 out 进行反向传播
            out.backward()
            # 对张量 a 应用正弦函数的原地操作
            a.sin_()
            # 断言在上下文中克隆的张量长度为 0
            self.assertClonedLenEqual(ctx, 0)
            # 再次计算张量 a 的平方和，并赋值给 out
            out = (a**2).sum()
            # 对张量 a 应用正弦函数的原地操作
            a.sin_()
            # 断言在上下文中克隆的张量长度为 1
            self.assertClonedLenEqual(ctx, 1)
            # 删除 out 引用
            del out
            # 断言在上下文中克隆的张量长度为 0
            self.assertClonedLenEqual(ctx, 0)

    def test_saved_same_tensor_many_times(self):
        # 我们应该只克隆一次
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            # 创建一个形状为 (2, 3) 的随机张量 a，并克隆它
            a = torch.randn(2, 3, requires_grad=True).clone()
            # 计算张量 a 的平方，并赋值给 b 和 c
            b = a**2
            c = a**2
            # 对张量 a 应用正弦函数的原地操作
            a.sin_()
            # 断言在上下文中克隆的张量长度为 1
            self.assertClonedLenEqual(ctx, 1)
            # 删除 b 和 c 的引用
            del b, c
            # 断言在上下文中克隆的张量长度为 0
            self.assertClonedLenEqual(ctx, 0)

    def test_saved_same_tensor_different_versions(self):
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            # 创建一个形状为 (2, 3) 的随机张量 a，并克隆它
            a = torch.randn(2, 3, requires_grad=True).clone()
            # 计算张量 a 的平方，并赋值给 b
            b = a**2
            # 对张量 a 应用正弦函数的原地操作
            a.sin_()
            # 计算张量 a 的平方，并赋值给 c
            c = a**2
            # 对张量 a 应用正弦函数的原地操作
            a.sin_()
            # 断言在上下文中克隆的张量长度为 2
            self.assertClonedLenEqual(ctx, 2)
            # 删除 b 的引用
            del b
            # 断言在上下文中克隆的张量长度为 1
            self.assertClonedLenEqual(ctx, 1)
            # 删除 c 的引用
            del c
            # 断言在上下文中克隆的张量长度为 0
            self.assertClonedLenEqual(ctx, 0)

    def test_with_math_views(self):
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            # 创建一个具有复数值的张量 a，并设置 requires_grad=True
            a = torch.tensor([1 + 1j], requires_grad=True).clone()
            # 计算张量 a 的共轭，并赋值给 b
            b = a.conj()
            # 计算张量 b 的平方和，并赋值给 out
            out = (b**2).sum()
            # 对张量 a 应用正弦函数的原地操作
            a.sin_()
            # 对 out 的绝对值进行反向传播
            out.abs().backward()

            # 重新创建一个具有复数值的张量 a，并设置 requires_grad=True
            a = torch.tensor([1 + 1j], requires_grad=True).clone()
            # 计算张量 a 的共轭，并赋值给 b
            b = a.conj()
            # 计算张量 b 的平方和，并赋值给 out
            out = (b**2).sum()
            # 对张量 b 应用正弦函数的原地操作
            b.sin_()
            # 对 out 的绝对值进行反向传播
            out.abs().backward()

    def test_with_out_variant(self):
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            # 创建一个张量 a，并设置 requires_grad=True
            a = torch.tensor([1.0], requires_grad=True)
            # 创建一个张量 b，并设置值为 1.0
            b = torch.tensor([1.0])
            # 创建一个张量 c，并设置值为 2.0
            c = torch.tensor([2.0])
            # 计算张量 a 和 b 的乘积，并赋值给 out
            out = a * b
            # 断言在上下文中保存的张量映射长度为 1
            self.assertTIDMapLenEqual(ctx, 1)
            # 对张量 c 应用正弦函数，并将结果写入 out
            torch.sin(c, out=b)
            # 断言在上下文中克隆的张量长度为 1
            self.assertClonedLenEqual(ctx, 1)
            # 对 out 进行反向传播
            out.backward()
            # 断言在上下文中克隆的张量长度为 0
            self.assertClonedLenEqual(ctx, 0)
    # 定义测试方法，用于测试在上下文外执行反向传播时的行为
    def test_backward_out_of_context(self):
        # 在 'allow_mutation_on_saved_tensors' 上下文中允许张量变异
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            # 创建一个随机张量 a，形状为 (2, 3)，需要计算梯度
            a = torch.rand(2, 3, requires_grad=True)
            # 计算张量 a 的平方和
            out = (a**2).sum()

        # 定义错误信息，用于检查在 'allow_mutation_on_saved_tensors' 上下文外执行反向传播时的断言错误
        msg = "Trying to backward outside of the 'allow_mutation_on_saved_tensors' context"
        # 断言在 'allow_mutation_on_saved_tensors' 上下文外执行反向传播时会抛出 AssertionError，并输出指定错误信息
        with self.assertRaisesRegex(AssertionError, msg):
            out.backward()

        # 在不同的 'allow_mutation_on_saved_tensors' 上下文中重新执行相同的操作
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            # 创建一个随机张量 a，形状为 (2, 3)，需要计算梯度
            a = torch.rand(2, 3, requires_grad=True)
            # 计算张量 a 的平方和
            out = (a**2).sum()

        # 在新的 'allow_mutation_on_saved_tensors' 上下文中，再次尝试在上下文外执行反向传播时的断言错误
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            # 使用断言检查在上下文外执行反向传播时是否会抛出 AssertionError
            with self.assertRaisesRegex(AssertionError, msg):
                out.backward()

    # 定义测试方法，用于测试不允许嵌套 'allow_mutation_on_saved_tensors' 上下文的行为
    def test_disallow_nesting(self):
        # 在 'allow_mutation_on_saved_tensors' 上下文中执行操作
        with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
            # 定义错误信息，用于检查在嵌套 'allow_mutation_on_saved_tensors' 上下文时的运行时错误
            msg = "allow_mutation_on_saved_tensors contexts cannot be nested"
            # 断言在嵌套 'allow_mutation_on_saved_tensors' 上下文时会抛出 RuntimeError，并输出指定错误信息
            with self.assertRaisesRegex(RuntimeError, msg):
                # 尝试在内部创建另一个 'allow_mutation_on_saved_tensors' 上下文，此处不应成功执行
                with torch.autograd.graph.allow_mutation_on_saved_tensors() as ctx:
                    pass
# 定义一个测试类 TestAutogradInferenceMode，继承自 TestCase
class TestAutogradInferenceMode(TestCase):

    # 定义一个私有方法 _is_inference_tensor，用于检查是否为推断模式张量
    def _is_inference_tensor(self, tensor):
        try:
            # 出现 RuntimeError 时抛出断言错误，并验证错误消息格式
            err_msg = "Inference tensors do not track version counter"
            with self.assertRaisesRegex(RuntimeError, err_msg):
                # 检查张量的 _version 属性是否存在
                tensor._version
            return True
        except AssertionError as e:
            return False

    # 定义测试推断模式上下文管理器的方法
    def test_inference_mode_context_manager(self):
        # 初始时推断模式应该是禁用的
        self.assertFalse(torch.is_inference_mode_enabled())
        
        # 使用 torch.inference_mode() 进入推断模式
        with torch.inference_mode():
            # 确认推断模式已启用
            self.assertTrue(torch.is_inference_mode_enabled())
            
            # 嵌套进入 torch.inference_mode(False)，禁用推断模式
            with torch.inference_mode(False):
                # 确认推断模式已禁用
                self.assertFalse(torch.is_inference_mode_enabled())
            
            # 离开内层推断模式，确认推断模式仍然是启用的
            self.assertTrue(torch.is_inference_mode_enabled())
        
        # 最终确认推断模式已经禁用
        self.assertFalse(torch.is_inference_mode_enabled())

    # 定义测试推断模式装饰器的方法
    def test_inference_mode_decorator(self):
        # 定义一个函数 func，用于测试推断模式是否正确设置
        def func(x):
            self.assertEqual(torch.is_inference_mode_enabled(), mode)
            return x * x

        # 遍历推断模式和使用关键字参数的组合
        for mode, use_kwarg in product((True, False, None), (True, False)):
            if mode is None:
                # 根据 use_kwarg 来选择使用不同的方式装饰 func 函数
                if use_kwarg:
                    decorated = torch.inference_mode(mode=func)
                else:
                    decorated = torch.inference_mode(func)
                mode = True
            else:
                if use_kwarg:
                    decorated = torch.inference_mode(mode=mode)(func)
                else:
                    decorated = torch.inference_mode(mode)(func)

            # 遍历是否需要梯度的情况
            for requires_grad in (True, False):
                # 创建一个张量 c，并应用装饰后的函数 decorated
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)
                d = decorated(c)
                
                # 断言：如果 mode 为 True，则 d 应为推断模式张量
                self.assertTrue(not mode or torch.is_inference(d))
                
                # 断言：d 的 requires_grad 属性应符合 requires_grad 和 not mode 的逻辑
                self.assertEqual(d.requires_grad, requires_grad and not mode)

    # 定义测试推断模式下张量创建的方法
    def test_inference_mode_tensor_creation(self):
        # 进入推断模式
        with torch.inference_mode():
            # 使用构造函数创建的新张量应为推断模式张量
            c = torch.ones(1, 2, 3)
            self.assertFalse(c.requires_grad)  # 确认不需要梯度
            self.assertTrue(torch.is_inference(c))  # 确认为推断模式张量

            # 在推断模式下，requires_grad 不影响推断模式张量的行为
            tmp = torch.ones(1, 2, 3, requires_grad=True)
            self.assertTrue(tmp.requires_grad)  # 确认需要梯度
            self.assertTrue(torch.is_inference(tmp))  # 确认为推断模式张量

            tmp = torch.ones(1, 2, 3).requires_grad_(False)
            self.assertFalse(tmp.requires_grad)  # 确认不需要梯度
            self.assertTrue(torch.is_inference(tmp))  # 确认为推断模式张量
    def test_inference_mode_existing_autograd_session(self):
        # 创建一个形状为 (1, 2, 3) 的张量 `s`，并要求计算其梯度
        s = torch.ones(1, 2, 3, requires_grad=True)
        # 克隆张量 `s` 到 `a`
        a = s.clone()

        # 对 `a` 进行非视图操作，计算 `a * a`
        out = a * a
        # 进入推断模式
        with torch.inference_mode():
            # 对 `a` 执行 inplace 加法操作
            a.add_(2)

        # 断言 `a` 不在推断模式中
        self.assertFalse(torch.is_inference(a))
        # 因为在推断模式外创建的张量不是推断张量，所以它们仍然会被追踪版本计数器
        err_msg = (
            "one of the variables needed for gradient computation has been "
            "modified by an inplace operation"
        )
        # 断言梯度计算时出现 RuntimeError，并检查错误消息
        with self.assertRaisesRegex(RuntimeError, err_msg):
            out.backward(torch.ones_like(out))

    def test_inference_mode_inf_tensor_in_inf_mode_functional_op(self):
        def functional_op(x):
            return x * x

        # 进入推断模式
        with torch.inference_mode():
            for requires_grad in (True, False):
                # 创建形状为 (1, 2, 3) 的张量 `c`，要求是否需要计算梯度
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                # 执行非视图操作，产生不需要梯度的推断张量
                func_out = functional_op(c)
                self.assertTrue(torch.is_inference(func_out))
                self.assertFalse(func_out.requires_grad)

    def test_inference_mode_inf_tensor_in_inf_mode_inplace_op(self):
        @torch.inference_mode()
        def run_test(fn):
            for requires_grad in (True, False):
                # 创建形状为 (1, 2, 3) 的张量 `c`，要求是否需要计算梯度
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                # 执行 inplace 操作后，张量仍然是推断张量
                fn(c)
                self.assertTrue(torch.is_inference(c))
                self.assertEqual(c.requires_grad, requires_grad)

        # 运行测试 lambda 函数，对不同的 inplace 操作进行测试
        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))
        run_test(lambda x: x.resize_(1, 2))
        run_test(lambda x: x.resize_as_(torch.ones(1, 2)))
        run_test(lambda x: x.copy_(torch.ones(1, 2, 3)))

    def test_inference_mode_inf_tensor_in_inf_mode_view_op(self):
        # 进入推断模式
        with torch.inference_mode():
            for requires_grad in (True, False):
                # 创建形状为 (1, 2, 3) 的张量 `c`，要求是否需要计算梯度
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

                # 执行视图操作，产生不需要梯度的推断张量
                view_out = c.view(-1)
                self.assertTrue(torch.is_inference(view_out))
                self.assertFalse(view_out.requires_grad)
    # 测试函数：在推理模式下，对具有不同 requires_grad 值的张量进行功能操作
    def test_inference_mode_inf_tensor_in_normal_mode_functional_op(self):
        
        # 定义一个功能操作函数，计算输入张量的平方
        def functional_op(x):
            return x * x
        
        # 遍历 requires_grad 取值为 True 和 False 的情况
        for requires_grad in (True, False):
            # 进入推理模式上下文
            with torch.inference_mode():
                # 创建一个形状为 (1, 2, 3) 的张量，所有元素为1，根据 requires_grad 设置是否需要梯度
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)
        
        # 对创建的张量 c 进行功能操作
        func_out = functional_op(c)
        
        # 断言功能操作的输出不处于推理模式
        self.assertFalse(torch.is_inference(func_out))
        
        # 断言功能操作的输出不需要梯度
        self.assertFalse(func_out.requires_grad)
        
        # 断言功能操作的输出是叶子节点
        self.assertTrue(func_out.is_leaf)

    # 测试函数：在推理模式下，对具有不同 requires_grad 值的张量进行原地操作
    def test_inference_mode_inf_tensor_in_normal_mode_inplace_op(self):
        
        # 定义一个运行测试的函数，接受一个原地操作函数作为参数
        def run_test(fn):
            # 遍历 requires_grad 取值为 False 和 True 的情况
            for requires_grad in (False, True):
                # 进入推理模式上下文
                with torch.inference_mode():
                    # 创建一个形状为 (1, 2, 3) 的张量，所有元素为1，根据 requires_grad 设置是否需要梯度
                    c = torch.ones(1, 2, 3, requires_grad=requires_grad)
                
                # 如果 requires_grad 为 True
                if requires_grad:
                    # 在原地操作中使用需要梯度的叶子变量
                    pass
                else:
                    # 抛出运行时错误，表明推理张量在推理模式外进行了原地更新
                    err_msg = "Inplace update to inference tensor outside InferenceMode"
                    with self.assertRaisesRegex(RuntimeError, err_msg):
                        fn(c)
        
        # 分别测试加法和转置两种原地操作
        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))

    # 测试函数：在推理模式下，对具有不同 requires_grad 值的张量进行视图操作
    def test_inference_mode_inf_tensor_in_normal_mode_view_op(self):
        
        # 遍历 requires_grad 取值为 True 和 False 的情况
        for requires_grad in (True, False):
            # 进入推理模式上下文
            with torch.inference_mode():
                # 创建一个形状为 (1, 2, 3) 的张量，所有元素为1，根据 requires_grad 设置是否需要梯度
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)
            
            # 对张量 c 进行视图变换，将其展平为一维
            out = c.view(-1)
            
            # 断言视图变换后的输出处于推理模式
            self.assertTrue(torch.is_inference(out))
            
            # 断言视图变换后的输出不需要梯度
            self.assertFalse(out.requires_grad)
            
            # 断言视图变换后的输出不是视图
            self.assertFalse(out._is_view())
            
            # 断言视图变换后的输出是叶子节点
            self.assertTrue(out.is_leaf)

    # 测试函数：在推理模式下，对具有不同 requires_grad 值的普通张量进行原地操作输出
    def test_normal_tensor_inplace_output_in_inference_mode(self):
        
        # 定义一个运行测试的函数，接受一个原地操作函数作为参数
        def run_test(fn):
            # 遍历 requires_grad 取值为 True 和 False 的情况
            for requires_grad in (True, False):
                # 创建一个形状为 (1, 2, 3) 的张量，所有元素为1，根据 requires_grad 设置是否需要梯度
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                
                # 克隆张量 s
                a = s.clone()
                
                # 进入推理模式上下文
                with torch.inference_mode():
                    # 对张量 a 执行原地操作
                    fn(a)
                    
                    # 断言原地操作后张量 a 不处于推理模式
                    self.assertFalse(torch.is_inference(a))
                    
                    # 断言原地操作后张量 a 的梯度需求与初始设定一致
                    self.assertEqual(a.requires_grad, requires_grad)
                    
                    # 连续两次原地操作
                    fn(a)
                    
                    # 断言连续两次原地操作后张量 a 不处于推理模式
                    self.assertFalse(torch.is_inference(a))
                    
                    # 断言连续两次原地操作后张量 a 的梯度需求与初始设定一致
                    self.assertEqual(a.requires_grad, requires_grad)
                    
                    # 对张量 a 进行视图变换
                    view_out = a.view(-1)
                    
                    # 断言视图变换后的输出不处于推理模式
                    self.assertFalse(torch.is_inference(view_out))
                    
                    # 断言视图变换后的输出的梯度需求与初始设定一致
                    self.assertEqual(view_out.requires_grad, requires_grad)
        
        # 分别测试加法和转置两种原地操作
        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))
    def test_normal_tensor_inplace_output_in_normal_mode(self):
        # 定义一个测试函数，测试给定的操作对张量的影响
        def run_test(fn):
            # 针对 requires_grad 为 True 和 False 进行测试
            for requires_grad in (True, False):
                # 创建一个 requires_grad 属性为 requires_grad 的全为 1 的张量
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                # 克隆张量 s 到 a
                a = s.clone()

                # 进入推断模式
                with torch.inference_mode():
                    # 执行操作 fn
                    fn(a)
                    # 断言张量 a 不处于推断模式
                    self.assertFalse(torch.is_inference(a))
                    # 断言张量 a 的 requires_grad 属性与初始设置一致
                    self.assertEqual(a.requires_grad, requires_grad)

                # 再次执行操作 fn
                fn(a)
                # 断言张量 a 不处于推断模式
                self.assertFalse(torch.is_inference(a))
                # 断言张量 a 的 requires_grad 属性与初始设置一致
                self.assertEqual(a.requires_grad, requires_grad)

                # inplace 操作 -> inplace 操作
                fn(a)
                # 断言张量 a 不处于推断模式
                self.assertFalse(torch.is_inference(a))
                # 断言张量 a 的 requires_grad 属性与初始设置一致
                self.assertEqual(a.requires_grad, requires_grad)

                # inplace 操作 -> inplace 操作 -> view 操作
                view_out = a.view(-1)
                # 断言 view_out 不处于推断模式
                self.assertFalse(torch.is_inference(view_out))
                # 断言 view_out 的 requires_grad 属性与初始设置一致
                self.assertEqual(view_out.requires_grad, requires_grad)
            # 使用 add_ 操作运行测试函数
            run_test(lambda x: x.add_(2))
            # 使用 transpose_ 操作运行测试函数
            run_test(lambda x: x.transpose_(0, 1))

    def test_normal_tensor_view_output_in_inference_mode(self):
        # 针对 requires_grad 为 True 和 False 进行测试
        for requires_grad in (True, False):
            # 创建一个 requires_grad 属性为 requires_grad 的全为 1 的张量
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            # 克隆张量 s 到 a
            a = s.clone()

            # 进入推断模式
            with torch.inference_mode():
                # 执行 view 操作
                out = a.view(-1)
                # 断言 out 不处于推断模式
                self.assertFalse(torch.is_inference(out))
                # 断言 out 的 requires_grad 属性与初始设置一致
                self.assertEqual(out.requires_grad, requires_grad)
                # 断言 out 是一个视图
                self.assertTrue(out._is_view())

                # view 操作 -> view 操作
                tmp = out.view(-1)
                # 断言 tmp 不处于推断模式
                self.assertFalse(torch.is_inference(tmp))
                # 断言 tmp 的 requires_grad 属性与初始设置一致
                self.assertEqual(tmp.requires_grad, requires_grad)
                # 断言 tmp 是一个视图
                self.assertTrue(tmp._is_view())
                # 断言 tmp 是叶子节点
                self.assertTrue(tmp.is_leaf)

                # view 操作 -> view 操作 -> inplace 操作
                self.assertTrue(torch.is_inference_mode_enabled())
                tmp.add_(2)
                # 断言 tmp 不处于推断模式
                self.assertFalse(torch.is_inference(tmp))
                # 断言 tmp 的 requires_grad 属性与初始设置一致
                self.assertEqual(tmp.requires_grad, requires_grad)
                # 访问 is_leaf 属性会尝试更新 grad_fn，并引发异常：
                # A view was created in inference mode and its base or
                # another view of its base has been modified inplace in normal mode
                # tmp.is_leaf
                # 断言张量 a 的版本与 tmp 的版本相同
                self.assertEqual(a._version, tmp._version)
    # 定义一个函数，对输入的张量进行平方操作
    def test_normal_tensor_view_output_in_normal_mode(self):
        def functional_op(x):
            return x * x  # 返回输入张量的平方

        # 遍历 requires_grad 参数为 True 和 False 的情况
        for requires_grad in (True, False):
            # 创建一个形状为 (1, 2, 3) 的张量 s，并根据 requires_grad 参数决定是否需要梯度
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            # 克隆张量 s 得到张量 a
            a = s.clone()

            # 进入推理模式
            with torch.inference_mode():
                # 将张量 a 展平为一维
                out = a.view(-1)
                # 断言展平后的张量 out 不处于推理模式
                self.assertFalse(torch.is_inference(out))
                # 断言展平后的张量 out 的梯度需求与输入参数一致
                self.assertEqual(out.requires_grad, requires_grad)
                # 断言展平后的张量 out 是一个视图
                self.assertTrue(out._is_view())
                # 断言展平后的张量 out 是叶子节点
                self.assertTrue(out.is_leaf)

            # 对展平后的张量 out 应用功能性操作 functional_op
            tmp = functional_op(out)
            # 断言 tmp 不处于推理模式
            self.assertFalse(torch.is_inference(tmp))
            # 断言 tmp 的梯度需求与输入参数一致
            self.assertEqual(tmp.requires_grad, requires_grad)

            # 如果 requires_grad 为 True，则测试推理模式下进行原地修改会引发 RuntimeError
            if requires_grad:
                err_msg = (
                    "A view was created in inference mode and is being modified inplace"
                )
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    out.add_(2)
                pass
            else:
                # 如果 requires_grad 为 False，则直接对 out 原地加 2
                out.add_(2)

            # 将 out 再次展平为形状 (2, 3)
            tmp = out.view(2, 3)
            # 断言 tmp 不处于推理模式
            self.assertFalse(torch.is_inference(tmp))
            # 断言 tmp 的梯度需求与输入参数一致
            self.assertEqual(tmp.requires_grad, requires_grad)

    # 测试混合推理模式和普通模式下的张量功能操作
    def test_mix_inference_and_normal_tensor_functional_op(self):
        # 遍历 requires_grad 参数为 True 和 False 的情况
        for requires_grad in (True, False):
            # 创建一个形状为 (1, 2, 3) 的张量 s，并根据 requires_grad 参数决定是否需要梯度
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)

            # 进入推理模式
            with torch.inference_mode():
                # 创建一个形状为 (1, 2, 3) 的张量 c，并根据 requires_grad 参数决定是否需要梯度
                c = torch.ones(1, 2, 3, requires_grad=requires_grad)

            # 使用 add 函数将张量 s 和 c 相加
            out = c.add(s)
            # 断言 out 不处于推理模式
            self.assertFalse(torch.is_inference(out))
            # 断言 out 的梯度需求与输入参数一致
            self.assertEqual(out.requires_grad, requires_grad)

            # 如果 requires_grad 为 True，则进行反向传播计算梯度，并检查 c 的梯度
            if requires_grad:
                # 对 out 进行反向传播
                out.backward(torch.ones_like(out))
                # 断言 c 的梯度与一个全为 1 的张量相等
                self.assertEqual(c.grad, torch.ones_like(c))

            # 如果 requires_grad 为 True，则测试推理模式下不允许保存变量进行反向传播
            if requires_grad:
                err_msg = "Inference tensors cannot be saved for backward"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    c * s

                # TODO: Test this with an autograd.Function when it works
                #       stack stopped capturing a TensorList input
                # # inference tensor in TensorList input
                # inputs = [s, c]
                # with self.assertRaisesRegex(RuntimeError, err_msg):
                #     torch.stack(inputs)
    def test_mix_inference_and_normal_tensor_inplace_op(self):
        # 循环测试是否允许混合推理和普通张量的原地操作
        for requires_grad in (True, False):
            # 创建一个形状为 (1, 2, 3) 的张量 s，根据 requires_grad 决定是否需要梯度
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)
            # 克隆张量 s，得到张量 a
            a = s.clone()

            # 进入推理模式
            with torch.inference_mode():
                # 创建一个形状为 (1, 2, 3) 的推理张量 c
                c = torch.ones(1, 2, 3)

            # 断言张量 c 是推理张量
            self.assertTrue(torch.is_inference(c))
            if requires_grad:
                # 如果张量 s 需要梯度，则检查原地操作 mul_ 是否会引发 RuntimeError
                err_msg = "Inference tensors cannot be saved for backward"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    a.mul_(c)

                # 对于需要梯度的情况，检查 torch.mul 是否会引发 RuntimeError
                err_msg = (
                    "out=... arguments don't support automatic differentiation, "
                    "but one of the arguments requires grad"
                )
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    torch.mul(s, s, out=c)
            else:
                # 如果张量 s 不需要梯度，则允许原地操作 mul_
                a.mul_(c)
                # 检查 torch.mul 是否会引发 RuntimeError
                err_msg = "Inplace update to inference tensor outside InferenceMode is not allowed"
                with self.assertRaisesRegex(RuntimeError, err_msg):
                    torch.mul(s, s, out=c)

    def test_mix_inference_and_normal_tensor_view_op(self):
        # 循环测试混合推理和普通张量的视图操作
        for requires_grad in (True, False):
            # 创建一个形状为 (1, 2, 3) 的张量 s，根据 requires_grad 决定是否需要梯度
            s = torch.ones(1, 2, 3, requires_grad=requires_grad)

            # 进入推理模式
            with torch.inference_mode():
                # 创建一个形状为 (1, 2, 3) 的推理张量 c
                c = torch.ones(1, 2, 3)

            # 使用 s 的形状创建推理张量 tmp1，并断言它是推理张量且不需要梯度
            tmp1 = c.view_as(s)
            self.assertTrue(torch.is_inference(tmp1))
            self.assertFalse(tmp1.requires_grad)

            # 使用 c 的形状创建普通张量 tmp2，并断言它不是推理张量且根据 requires_grad 确定是否需要梯度
            tmp2 = s.view_as(c)
            self.assertFalse(torch.is_inference(tmp2))
            self.assertEqual(tmp2.requires_grad, requires_grad)

    def test_inference_mode_handle_direct_view_on_rebase(self):
        # 测试在推理模式下直接视图重建的处理
        def run_test(fn):
            # 循环测试函数 fn 对推理和普通张量的处理
            for requires_grad in (True, False):
                # 创建一个形状为 (1, 2, 3) 的张量 s，根据 requires_grad 决定是否需要梯度
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                # 克隆张量 s，得到张量 a
                a = s.clone()

                # 进入推理模式，创建视图 view_out
                with torch.inference_mode():
                    view_out = a.view_as(a)

                if requires_grad:
                    # 如果张量 s 需要梯度，则检查是否会引发 RuntimeError
                    err_msg = "A view was created in inference mode and is being modified inplace"
                    with self.assertRaisesRegex(RuntimeError, err_msg):
                        fn(view_out)
                    pass
                else:
                    # 如果张量 s 不需要梯度，则直接调用函数 fn 处理 view_out
                    fn(view_out)

        # 运行测试 lambda 函数对原地操作 add_ 和 transpose_ 进行测试
        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))
    # 定义一个测试函数，测试在重建基础上的间接视图处理
    def test_inference_mode_handle_indirect_view_on_rebase(self):
        # 定义一个运行测试的内部函数，接受一个函数作为参数
        def run_test(fn):
            # 对于是否需要梯度的两种情况进行迭代测试
            for requires_grad in (True, False):
                # 创建一个形状为 (1, 2, 3) 的张量，是否需要梯度由 requires_grad 决定
                s = torch.ones(1, 2, 3, requires_grad=requires_grad)
                # 克隆张量 s，a 是其克隆副本
                a = s.clone()

                # 进入推理模式的上下文
                with torch.inference_mode():
                    # 使用张量 a 创建一个视图，展平成一维
                    view_out = a.view(-1)

                # 调用传入的函数 fn 处理张量 a
                fn(a)

                # 根据是否需要梯度来验证视图的梯度函数是否正确抛出异常
                if requires_grad:
                    # 如果需要梯度，则视图在推理模式下被创建，应该抛出 RuntimeError 异常
                    err_msg = "A view was created in inference mode and its base or another view "
                    with self.assertRaisesRegex(RuntimeError, err_msg):
                        view_out.grad_fn
                    # 占位符，这里保持空行或注释以保持代码结构完整
                    pass
                else:
                    # 如果不需要梯度，则视图的梯度函数应该为 None，不应抛出异常
                    view_out.grad_fn

        # 使用 lambda 函数调用 run_test 函数，分别测试张量的加法和转置操作
        run_test(lambda x: x.add_(2))
        run_test(lambda x: x.transpose_(0, 1))
class TestMultithreadAutograd(TestCase):
    # 测试多线程自动微分功能
    def _run_py_multithread_fn(
        self, fn, args=(), num_threads=10, kwargs=None, pass_idx=False
    ):
        # 定义一个内部辅助类，用于在子线程中传播异常到主线程
        class PropagatingThread(threading.Thread):
            """Helper class to propagate exception from child
            thread to main thread on join.

            Reference: https://stackoverflow.com/a/31614591/5602957
            """

            def run(self):
                self.exception = None
                try:
                    self.ret = super().run()
                except Exception as e:
                    self.exception = e

            def join(self, timeout=None):
                super().join(timeout)
                if self.exception:
                    raise self.exception from self.exception
                return self.ret

        threads = []
        # 创建多个线程并启动
        for idx in range(num_threads):
            p = PropagatingThread(target=fn, args=((idx, *args) if pass_idx else args))
            p.start()
            threads.append(p)

        # 等待所有线程结束
        for p in threads:
            p.join()

    def test_multithreaded_exception_propagation(self):
        # 测试异常在子线程中是否传播到主线程
        def fn():
            self.assertTrue(False)

        # 预期在子线程中出现断言错误异常
        with self.assertRaises(AssertionError):
            self._run_py_multithread_fn(fn)

    def test_simple_backward(self):
        # 简单的多线程反向传播，在训练开始时创建线程，其余的训练操作是分开的，例如输入、操作等。
        def train_fn():
            x = torch.ones(5, 5, requires_grad=True)
            y = (x + 3) * (x + 4) * 0.5
            y.sum().backward()
            self.assertEqual(x.grad, x + 3.5)

        self._run_py_multithread_fn(train_fn)

    def test_simple_backward_same_input(self):
        # 带有共享输入的简单多线程反向传播（例如Hogwild多线程训练）
        def train_fn_backward(x):
            y = (x + 3) * (x + 4) * 0.5
            y.sum().backward()

        x = torch.ones(5, 5, requires_grad=True)
        # 启动多线程运行train_fn_backward函数，传入相同的输入x
        self._run_py_multithread_fn(train_fn_backward, (x,))
        # 由于多个线程共同调用backward，所有线程都会将梯度累积到相同的.grad属性中
        self.assertEqual(x.grad, 10 * (x + 3.5))

        def train_fn_grad(x):
            y = (x + 3) * (x + 4) * 0.5
            grads = torch.autograd.grad(y.sum(), x)
            self.assertEqual(len(grads), 1)
            self.assertEqual(grads[0], x + 3.5)

        # 使用torch.autograd.grad()函数，每个线程的梯度会独立计算，不会累积到同一位置
        self._run_py_multithread_fn(train_fn_grad, (x,))
    def test_multi_grad_all_hooks(self):
        # Multihooks should behave independently per execution of backward
        # Test that the hook fired the number of times we ran backward
        # even if those executions occur concurrently on different threads
        
        # 创建四个张量，都需要梯度信息
        t1 = torch.rand(2, requires_grad=True)
        t2 = torch.rand(2, requires_grad=True)
        t3 = torch.rand(2, requires_grad=True)
        t4 = torch.rand(2, requires_grad=True)

        res = None
        count = [0]
        hook_lock = threading.Lock()

        # 定义一个钩子函数，用于处理梯度信息
        def hook(grads):
            nonlocal res
            with hook_lock:
                count[0] += 1
                # 检查每个梯度是否为None，并记录下来
                grad_is_none = [g is not None for g in grads]
                if res is None:
                    res = grad_is_none
                else:
                    self.assertEqual(res, grad_is_none)

        # 注册多重梯度钩子，监听 t1, t2, t3, t4 四个张量的梯度信息
        torch.autograd.graph.register_multi_grad_hook((t1, t2, t3, t4), hook)

        # 计算张量 t2 和 t3 的乘积之和作为输出
        out = (t2 * t3).sum()

        # 定义一个带有保留计算图的反向传播函数
        def backward_retain_graph(out, t2, t3):
            out.backward(inputs=(t2, t3), retain_graph=True)

        # 在多线程环境下运行上述反向传播函数
        self._run_py_multithread_fn(backward_retain_graph, (out, t2, t3), num_threads=5)

        # 断言钩子函数被调用的次数为5次
        self.assertEqual(count[0], 5)
        # 断言记录的梯度信息，期望为 [False, True, True, False]
        self.assertEqual(res, [False, True, True, False])

        # Leave one hook partially applied
        res = None
        count = [0]
        err_count = [0]
        bw_count = [0]
        bw_count_lock = threading.Lock()
        err_count_lock = threading.Lock()

        # 定义一个带异常的自定义 torch.autograd.Function 类
        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, gO):
                with bw_count_lock:
                    bw_count[0] += 1
                    # 在第一次反向传播时抛出异常
                    if bw_count[0] == 1:
                        raise RuntimeError("error message")
                    else:
                        return gO

        # 使用自定义函数计算张量 t2 和 t3 的乘积之和作为输出
        out = (Func.apply(t2) * t3).sum()

        # 定义一个带有保留计算图的反向传播函数，捕获异常并计数
        def backward_retain_graph(out, t2, t3):
            try:
                out.backward(inputs=(t2, t3), retain_graph=True)
            except RuntimeError:
                with err_count_lock:
                    err_count[0] += 1

        # 在多线程环境下运行上述异常处理的反向传播函数
        self._run_py_multithread_fn(backward_retain_graph, (out, t2, t3), num_threads=5)

        # 断言钩子函数被调用的次数为4次
        self.assertEqual(count[0], 4)
        # 断言捕获到的异常次数为1次
        self.assertEqual(err_count[0], 1)
        # 断言记录的梯度信息，期望为 [False, True, True, False]
        self.assertEqual(res, [False, True, True, False])
    def test_multi_grad_any_hooks(self):
        # 测试多个 hooks 在每次 backward 执行时独立运行
        # 确保 hook 在每次 backward 执行时被调用的次数正确
        # 即使这些执行在不同线程中并发进行
        t1 = torch.rand(2, requires_grad=True)  # 创建一个形状为 (2,) 的张量 t1，要求计算梯度
        t2 = torch.rand(2, requires_grad=True)  # 创建一个形状为 (2,) 的张量 t2，要求计算梯度
        t3 = torch.rand(2, requires_grad=True)  # 创建一个形状为 (2,) 的张量 t3，要求计算梯度
        t4 = torch.rand(2, requires_grad=True)  # 创建一个形状为 (2,) 的张量 t4，要求计算梯度

        res = None  # 初始化一个结果变量 res
        count = [0]  # 初始化一个计数器列表，用于存放计数值
        hook_lock = threading.Lock()  # 创建一个线程锁对象

        def hook(grad):
            nonlocal res  # 使用外部变量 res
            with hook_lock:  # 加锁，确保线程安全
                count[0] += 1  # 计数器加一
                if res is None:
                    res = "foo"  # 如果 res 为 None，设置其为 "foo"
                else:
                    self.assertEqual(res, "foo")  # 否则，断言 res 等于 "foo"

        torch.autograd.graph.register_multi_grad_hook(
            (t1, t2, t3, t4), hook, mode="any"
        )  # 注册一个多个梯度 hook，作用在 t1, t2, t3, t4 上，模式为 "any"

        out = (t2 * t3).sum()  # 计算 t2 和 t3 的乘积之和作为 out

        def backward_retain_graph(out, t2, t3):
            out.backward(inputs=(t2, t3), retain_graph=True)  # 执行反向传播，保留计算图

        self._run_py_multithread_fn(backward_retain_graph, (out, t2, t3), num_threads=5)  # 在多线程环境下运行 backward_retain_graph
        self.assertEqual(count[0], 5)  # 断言 count 的值为 5
        self.assertEqual(res, "foo")  # 断言 res 的值为 "foo"

        # 在一个线程的 backward 中引发错误
        res = None  # 重置 res 为 None
        count = [0]  # 重置计数器列表为 [0]
        err_count = [0]  # 初始化错误计数器列表为 [0]
        bw_count = [0]  # 初始化反向传播计数器列表为 [0]
        bw_count_lock = threading.Lock()  # 创建一个反向传播计数器的线程锁对象
        err_count_lock = threading.Lock()  # 创建一个错误计数器的线程锁对象

        class Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x  # 返回输入的张量 x

            @staticmethod
            def backward(ctx, gO):
                with bw_count_lock:  # 加锁，确保线程安全
                    bw_count[0] += 1  # 反向传播计数器加一
                    if bw_count[0] == 1:
                        raise RuntimeError("error message")  # 如果是第一次反向传播，则抛出运行时错误
                    else:
                        return gO  # 否则返回梯度 gO

        out = (Func.apply(t2) * t3).sum()  # 使用自定义的 Func 对象进行计算，并求和作为 out

        def backward_retain_graph(out, t2, t3):
            try:
                out.backward(inputs=(t2, t3), retain_graph=True)  # 执行反向传播，保留计算图
            except RuntimeError:
                with err_count_lock:  # 捕获 RuntimeError 异常
                    err_count[0] += 1  # 错误计数器加一

        self._run_py_multithread_fn(backward_retain_graph, (out, t2, t3), num_threads=5)  # 在多线程环境下运行 backward_retain_graph

        # 期望所有 5 个线程都会增加 count，因为 hook 在自定义 backward 之前运行
        self.assertEqual(count[0], 5)  # 断言 count 的值为 5
        self.assertEqual(err_count[0], 1)  # 断言 err_count 的值为 1
        self.assertEqual(res, "foo")  # 断言 res 的值为 "foo"
    def test_dataparallel_saved_tensors_hooks(self):
        # 定义一个函数pack，用于包装输入并发出警告信息
        def pack(x):
            warnings.warn("pack")
            return x

        # 将self引用保存到_self变量中
        _self = self

        # 定义一个继承自torch.nn.Module的模型类Model
        class Model(torch.nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x):
                # 进入一个警告上下文管理器，用于捕获警告信息
                with warnings.catch_warnings(record=True) as w:
                    # 计算y = x * x
                    y = x * x
                    # 如果当前CUDA设备数量大于等于2
                    if torch.cuda.device_count() >= 2:
                        # DataParallel在不同线程中调用前向传播，不会传播TLS（线程本地存储），因此这里不应调用hooks
                        _self.assertEqual(len(w), 0)
                    else:
                        # DataParallel只使用一个线程，因此这里应该调用hooks
                        _self.assertGreater(len(w), 0)

        # 创建一个5x5的张量x，要求梯度计算
        x = torch.ones(5, 5, requires_grad=True)
        # 使用DataParallel封装Model模型
        model = torch.nn.DataParallel(Model())

        # 进入一个图保存张量hooks的上下文管理器
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            # 对模型进行前向传播计算
            model(x)
            # 进入一个警告上下文管理器，用于捕获警告信息
            with warnings.catch_warnings(record=True) as w:
                # 计算y = x * x
                y = x * x
                # 这里应该调用hooks
                _self.assertGreater(len(w), 0)
    def test_python_thread_in_middle(self):
        # 用户可能编写一个网络，从一个 CPU 线程开始运行，然后与其他线程（通过 python threading 或 fork/join 调用）并发运行其第二部分，
        # 然后在两个线程上同时调用 backward()/grad()，就像从底部输入到顶部输出的 Y 形状。这种方式会使部分 GraphTask 被跨多个线程共享，
        # 我们需要确保用户指定 retain_graph=True，否则需要报出正确的错误消息。

        # 案例 1：多个 python 线程同时调用 backward，retain_graph=False
        # 应该在某些线程中抛出没有 retain_graph 的错误。
        success_vs_raises = [0, 0]

        def train_fn_no_retain_graph(x):
            y = x + x**2
            try:
                y.sum().backward()
                success_vs_raises[0] += 1
            except RuntimeError as error:
                success_vs_raises[1] += 1
                self.assertRegex(str(error), "Specify retain_graph=True")

        x_no_retain = torch.ones(5, 5, requires_grad=True)
        y_no_retain = x_no_retain + x_no_retain**2
        self._run_py_multithread_fn(
            train_fn_no_retain_graph, (y_no_retain,), num_threads=5
        )
        # 至少有一个线程会成功，此时其他所有线程应该抛出错误，提示用户指定 retain_graph=True
        self.assertTrue(success_vs_raises[0] >= 1)

        # 多个 python 线程同时调用 backward，使用 retain_graph=True 不会出错
        def train_fn_retain_graph(x):
            y = x + x**2
            y.sum().backward(retain_graph=True)

        x_retain = torch.ones(5, 5, requires_grad=True)
        y_retain = x_retain + x_retain**2
        self._run_py_multithread_fn(train_fn_retain_graph, (y_retain,), num_threads=5)
        # 结果应该等于 num_thread * 梯度
        self.assertEqual(
            x_retain.grad,
            5 * (4 * x_retain**3 + 6 * (x_retain**2) + 4 * x_retain + 1),
        )
    def test_fork_join_in_middle(self):
        # 定义测试函数：在中间使用多个 jit 线程进行反向传播（fork/join 原语）
        # 类似于 test_python_thread_in_middle，我们测试 retain_graph=False/True

        # Case 1: multiple grad() calls with jit threads, retain_graph=False
        # 在一些线程中不使用 retain_graph 会抛出错误。
        @torch.jit.script
        def train_fn_jit_no_retain(middle, orig_x):
            y = middle + middle**2
            return torch.autograd.grad([y.sum()], [orig_x])

        @torch.jit.script
        def train_fn_fork_join_calls_no_retain(x):
            y_no_retain = (x + 3) * (x + 4) * 0.5

            # 在 jit 线程中调用 train_fn_jit_no_retain
            fut = torch.jit._fork(train_fn_jit_no_retain, y_no_retain, x)
            grad_hat = train_fn_jit_no_retain(y_no_retain, x)
            grad = torch.jit._wait(fut)
            return grad, grad_hat

        try:
            # 测试不使用 retain_graph=True 时是否抛出 RuntimeError
            train_fn_fork_join_calls_no_retain(torch.randn(5, 5, requires_grad=True))
        except RuntimeError as error:
            self.assertRegex(str(error), "Specify retain_graph=True")

        # Case 2: no error with retain_graph=True
        # 使用 retain_graph=True 时不会抛出错误
        @torch.jit.script
        def train_fn_jit_retain(middle, orig_x):
            y = middle + middle**2
            return torch.autograd.grad([y.sum()], [orig_x], retain_graph=True)

        @torch.jit.script
        def train_fn_fork_join_calls_retain(x):
            y_retain = (x + 3) * (x + 4) * 0.5
            fut1 = torch.jit._fork(train_fn_jit_retain, y_retain, x)
            fut2 = torch.jit._fork(train_fn_jit_retain, y_retain, x)
            grad = train_fn_jit_retain(y_retain, x)
            grad1 = torch.jit._wait(fut1)
            grad2 = torch.jit._wait(fut2)
            return grad, grad1, grad2

        # 调用测试函数，验证多个 fork/join 线程中的梯度是否一致
        grad, grad1, grad2 = train_fn_fork_join_calls_retain(
            torch.randn(5, 5, requires_grad=True)
        )
        self.assertEqual(grad, grad1)
        self.assertEqual(grad, grad2)

    def test_preserve_backtrace(self):
        # 定义测试函数：保留回溯信息
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            def backward(ctx, *grad):
                raise ValueError("something")

        t = torch.rand(10, requires_grad=True)
        try:
            # 调用自定义函数 Foo 的反向传播方法，并期望捕获 ValueError 异常
            Foo.apply(t).sum().backward()
        except Exception:
            import traceback

            # 捕获异常并获取详细的回溯信息
            tb = sys.exc_info()[2]
            tb_str = "\n".join(traceback.format_tb(tb))
            self.assertTrue('raise ValueError("something")' in tb_str)

    # TODO(@anjali411): add an OpInfo based test for torch.cat
    # Issue: https://github.com/pytorch/pytorch/issues/51627
    #        https://github.com/pytorch/pytorch/issues/75852
    # 定义测试函数：测试 torch.cat 在实部为 double 类型、虚部为 cdouble 类型的张量上的操作
    def test_cat_stack_r_to_c(self):
        # 创建实部张量，形状为 (3, 2)，类型为 torch.double，需要计算梯度
        inp_c = torch.rand(3, 2, dtype=torch.cdouble, requires_grad=True)
        # 创建虚部张量，形状为 (3, 2)，类型为 torch.double，需要计算梯度
        inp_r = torch.randn(3, 2, dtype=torch.double, requires_grad=True)

        # 定义函数 fn，将两个张量在最后一个维度上拼接
        def fn(x1, x2):
            return torch.cat((x1, x2), dim=-1)

        # 定义函数 fn2，将两个张量在新创建的维度上堆叠
        def fn2(x1, x2):
            return torch.stack((x1, x2), dim=-1)

        # 使用 gradcheck 验证函数 fn 在不同输入顺序下的梯度计算是否正确
        torch.autograd.gradcheck(fn, [inp_r, inp_c], check_forward_ad=True)
        torch.autograd.gradcheck(fn, [inp_c, inp_r], check_forward_ad=True)

        # 使用 gradcheck 验证函数 fn2 在不同输入顺序下的梯度计算是否正确
        torch.autograd.gradcheck(fn2, [inp_r, inp_c], check_forward_ad=True)
        torch.autograd.gradcheck(fn2, [inp_c, inp_r], check_forward_ad=True)

    # 定义测试函数：测试 torch.autograd.set_multithreading_enabled 的多线程设置
    def test_set_multithreading_enabled_as_context_manager_and_function(self):
        # 测试作为上下文管理器使用时的多线程设置
        with torch.autograd.set_multithreading_enabled(False):
            # 断言多线程未启用
            self.assertFalse(torch.autograd.is_multithreading_enabled())
        # 断言多线程已启用
        self.assertTrue(torch.autograd.is_multithreading_enabled())

        # 测试作为上下文管理器使用时的多线程设置
        with torch.autograd.set_multithreading_enabled(True):
            # 断言多线程已启用
            self.assertTrue(torch.autograd.is_multithreading_enabled())
        # 断言多线程已启用
        self.assertTrue(torch.autograd.is_multithreading_enabled())

        # 测试作为上下文管理器使用时的多线程设置
        with torch.autograd.set_multithreading_enabled(False):
            # 在上下文中手动启用多线程
            torch.autograd.set_multithreading_enabled(True)
            # 断言多线程已启用
            self.assertTrue(torch.autograd.is_multithreading_enabled())
        # 断言多线程已启用
        self.assertTrue(torch.autograd.is_multithreading_enabled())

        # 手动设置多线程为禁用状态
        torch.autograd.set_multithreading_enabled(False)
        # 断言多线程未启用
        self.assertFalse(torch.autograd.is_multithreading_enabled())

        # 手动设置多线程为启用状态
        torch.autograd.set_multithreading_enabled(True)
        # 断言多线程已启用
        self.assertTrue(torch.autograd.is_multithreading_enabled())

    # 标记为跳过测试 CUDA 不可用的情况下，测试自定义函数在设备线程中传播错误
    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_custom_function_propagates_errors_from_device_thread(self):
        # 定义自定义函数 MyFunc 继承自 torch.autograd.Function
        class MyFunc(Function):
            # 前向传播函数：返回输入张量本身
            @staticmethod
            def forward(ctx, x):
                return x

            # 反向传播函数：抛出运行时错误 "blah"
            @staticmethod
            def backward(ctx, gO):
                raise RuntimeError("blah")
                return gO

        # 创建 CUDA 设备上的张量 t，需要计算梯度
        t = torch.tensor([1.0, 2.0], requires_grad=True, device=torch.device("cuda"))
        # 使用自定义函数 MyFunc 对张量 t 进行操作，并求和
        out = MyFunc.apply(t).sum()

        # 预期在反向传播时抛出运行时错误 "blah"
        with self.assertRaisesRegex(RuntimeError, "blah"):
            out.backward()
    # 定义一个名为 TestNestedCheckpoint 的测试用例类，继承自 TestCase
    class TestNestedCheckpoint(TestCase):
        
        # 静态方法 grad，接受一个函数 fn 作为参数，返回一个装饰器
        @staticmethod
        def grad(fn):
            # 内部函数 wrapper 接受参数 x
            def wrapper(x):
                # 启用 Torch 的梯度计算上下文
                with torch.enable_grad():
                    # 调用 fn 函数计算输出
                    out = fn(x)
                    # 使用 create_graph=True 计算输入 x 的梯度
                    (grad_input,) = torch.autograd.grad(out, inputs=(x,), create_graph=True)
                return grad_input

            return wrapper
        
        # 静态方法 sum，接受一个函数 fn 作为参数，返回一个函数 wrapped
        @staticmethod
        def sum(fn):
            # 函数 wrapped 接受参数 x，返回 fn(x) 的和
            def wrapped(x):
                return fn(x).sum()

            return wrapped
        
        # 静态方法 checkpoint，接受一个函数 fn 作为参数，返回一个函数 wrapped
        @staticmethod
        def checkpoint(fn):
            # 函数 wrapped 接受任意位置和关键字参数，通过 Torch 的 checkpoint 功能调用 fn
            def wrapped(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(
                    fn, *args, use_reentrant=False, **kwargs
                )

            return wrapped
        
        # 定义一个方法 get_tests，接受一个函数 fn 作为参数，返回一组测试用例
        def get_tests(self, fn):
            # 获取梯度计算函数 grad 和 checkpoint 函数 c
            grad, c = self.grad, self.checkpoint

            # 定义测试用例元组 tests，包含不同包装方式的函数 fn 和对应的 checkpoint 包装
            tests = (
                (fn, (c(fn), c(c(fn)))),
                (grad(fn), (grad(c(fn)), grad(c(c(fn))))),
                (
                    grad(grad(fn)),
                    (grad(c(grad(fn))), c(grad(grad(c(fn)))), grad(c(grad(c(fn))))),
                ),
                (
                    grad(grad(grad(fn))),
                    (grad(c(grad(grad(c(fn))))), grad(c(grad(c(grad(c(fn))))))),
                ),
            )
            return tests
        
        # 定义一个方法 check_graph_dies，接受一个函数 fn 作为参数，检查计算图中的节点是否释放
        def check_graph_dies(self, fn):
            # 定义内部函数 iter_graph，迭代计算图中的节点，查看其是否被释放
            def iter_graph(roots):
                if not roots:
                    return
                seen = set()
                q = collections.deque()
                for node in roots:
                    if node is not None:
                        seen.add(node)
                        q.append(node)

                while q:
                    node = q.popleft()
                    for fn, _idx in node.next_functions:
                        if fn in seen or fn is None:
                            continue
                        seen.add(fn)
                        q.append(fn)

                    yield node
            
            # 定义内部类 Handle，用于包装节点名称
            class Handle:
                __slot__ = ["node_name"]

                def __init__(self, node_name):
                    self.node_name = node_name
            
            # 定义作用域函数 scope
            def scope():
                # 创建一个随机张量 a，要求计算其梯度
                a = torch.randn((), requires_grad=True)
                # 计算 fn 函数在张量 a 上的输出
                out = fn(a)
                refs = []
                # 遍历计算图中的节点，并为每个节点创建 Handle 对象存入 weakref 列表 refs
                for node in iter_graph([out.grad_fn]):
                    handle = Handle(node.name())
                    refs.append(weakref.ref(handle))
                    node.metadata["blah"] = handle
                return refs
            
            # 在作用域内执行 scope 函数，获取节点的 weakref 引用列表 refs
            refs = scope()
            # 提取所有有效的节点名称列表
            node_names = [ref().node_name for ref in refs if ref() is not None]
            # 如果仍有节点存活，打印其名称
            if len(node_names) > 0:
                print("Nodes still alive:", node_names)

            # 使用断言检查节点列表长度是否为 0
            self.assertEqual(len(node_names), 0)
        
        # 参数化测试方法 early_stop，接受一个布尔值参数，用于控制测试过程中的停止条件
        @parametrize("early_stop", [True, False])
    def test_nested_checkpoint(self, early_stop):
        # 使用给定的早停标志设置检查点
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            # 创建一个随机张量 x，要求计算梯度
            x = torch.randn((), requires_grad=True)

            # 定义函数 f，对输入 x 进行一系列数学操作
            def f(x):
                out = x.sin().exp().sin()
                return out

            # 定义函数 g，对输入 x 进行一系列数学操作，并计算梯度
            def g(x):
                a = x.sin().exp().sin()
                b = x.sin().exp().sin()
                (ga,) = torch.autograd.grad(a, x)
                (gb,) = torch.autograd.grad(b, x)
                return x.sin()

            # 对于函数 f 和 g，获取其预期结果和实际执行结果进行测试
            for fn in (f, g):
                for expected_fn, actual_fns in self.get_tests(fn):
                    expected = expected_fn(x)

                    # 对于每一个实际执行函数，计算其结果并进行断言测试
                    for actual_fn in actual_fns:
                        actual = actual_fn(x)
                        self.assertTrue(torch.allclose(expected, actual))
                        # 检查计算图是否在当前实际执行函数下终止
                        self.check_graph_dies(actual_fn)

    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint_two_children(self, early_stop):
        # 使用给定的早停标志设置检查点
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            grad, sum, c = self.grad, self.sum, self.checkpoint

            # 定义函数 f，对输入 x 进行一系列数学操作
            def f(x):
                return x.sin().exp().sin()

            # 定义函数 g，对输入 x 进行一系列数学操作
            def g(x):
                return x.cos().sin().exp()

            # 定义高阶函数 hc，对 g(f(x)) 应用检查点
            def hc(x):
                return c(g)(c(f)(x))

            # 定义函数 h，对 g(f(x)) 进行直接计算
            def h(x):
                return g(f(x))

            # 创建一个随机张量 a，要求计算梯度
            a = torch.randn(3, 3, requires_grad=True)

            # 计算预期结果
            expected = grad(sum(grad(sum(h))))(a)

            # 使用检查点计算实际结果，并进行断言测试
            actual = grad(sum(grad(sum(c(hc)))))(a)
            self.assertTrue(torch.allclose(expected, actual))

            # 再次使用检查点计算实际结果，并进行断言测试
            actual = grad(sum(c(grad(sum(c(hc))))))(a)
            self.assertTrue(torch.allclose(expected, actual))

            # 检查计算图是否在当前高阶函数 hc 下终止
            self.check_graph_dies(grad(c(hc)))
            self.check_graph_dies(grad(sum(grad(sum(c(hc))))))
            self.check_graph_dies(grad(sum(c(grad(sum(c(hc)))))))

    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint_non_tensor_inputs_and_outputs(self, early_stop):
        # 定义一个函数 fn，接受多个参数并执行特定函数 f，返回非张量结果
        def fn(k, a, b, f):
            return f(k * a * b.exp()), 1, "abcd"

        k = 3
        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)

        # 定义函数 f，对输入 x 进行 sin() 操作
        def f(x):
            return x.sin()

        # 使用给定的早停标志设置检查点
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            # 使用检查点运行函数 fn，并获取输出结果
            out, _unused1, _unused2 = checkpoint(fn, k, a, b, f, use_reentrant=False)
        # 计算实际梯度
        actual_grads = torch.autograd.grad(out, (a, b))

        # 直接运行函数 fn，并获取输出结果
        out, _unused1, _unused2 = fn(k, a, b, f)
        # 计算预期梯度
        expected_grads = torch.autograd.grad(out, (a, b))

        # 对比实际和预期梯度，进行断言测试
        for actual, expected in zip(actual_grads, expected_grads):
            self.assertTrue(torch.allclose(actual, expected))

    @parametrize("early_stop", [True, False])
    # 定义一个测试函数，测试使用嵌套检查点和关键字参数
    def test_nested_checkpoint_kwargs(self, early_stop):
        # 定义一个内部函数 fn，接受参数 a 和可选参数 blah
        def fn(a, blah=None):
            # 计算 a 的 sin 函数和指数函数的组合
            out = a.sin().exp()
            # 如果 blah 不是 None，则将 out 乘以 blah
            if blah is not None:
                out = out * blah
            # 对结果再次应用 sin 函数和指数函数的组合
            return out.sin().exp()

        # 创建两个张量 a 和 b，并标记为需要梯度信息
        a = torch.tensor(2.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)

        # 使用早停策略设置检查点
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            # 使用检查点运行函数 fn，传入参数 a 和命名参数 blah=b，不允许重入
            out = checkpoint(fn, a, blah=b, use_reentrant=False)
            # 计算 out 对 a 和 b 的梯度
            actual_grads = torch.autograd.grad(out, (a, b))

            # 直接调用 fn 函数，传入参数 a 和命名参数 blah=b
            out = fn(a, blah=b)
            # 计算 out 对 a 和 b 的梯度
            expected_grads = torch.autograd.grad(out, (a, b))
            # 断言两组梯度值是否相近
            for actual, expected in zip(actual_grads, expected_grads):
                self.assertTrue(torch.allclose(actual, expected))

    # 参数化测试函数，测试使用相同图的嵌套检查点
    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint_same_graph(self, early_stop):
        # 定义一个计数器，用于记录钩子函数调用次数
        counter = [0]

        # 定义一个钩子函数，将计数器递增
        def hook(*_unused_args):
            counter[0] += 1

        # 定义一个函数 fn，接受参数 a，返回 a 的 sin、cos、sin 函数组合的结果
        def fn(a):
            return a.sin().cos().sin()

        # 创建一个张量 a，并标记为需要梯度信息
        a = torch.tensor(1.0, requires_grad=True)

        # 使用早停策略设置检查点
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            # 使用检查点运行函数 fn，传入参数 a，不允许重入
            out = checkpoint(fn, a, use_reentrant=False)
        
        # 注册钩子函数到原始图的下一个函数的第一个输入
        out.grad_fn.next_functions[0][0].register_hook(hook)
        # 在原始图上执行反向传播
        out.backward()

        # 断言计数器的值为 1
        self.assertEqual(counter[0], 1)

    # 参数化测试函数，测试支持重新进入的嵌套检查点的反向传播
    @parametrize("early_stop", [True, False])
    def test_nested_checkpoint_reentrant_backwards(self, early_stop):
        # 定义一个函数 fn，接受参数 a，执行一系列 sin 和 cos 操作，并返回结果
        def fn(a):
            x = a.sin().cos()
            out = x.sin()
            return x, out

        # 定义一个钩子函数，执行重新进入的反向传播，跳过钩子函数注册的部分
        def hook(*_unused_args):
            x.backward(retain_graph=True)

        # 创建一个张量 a，并标记为需要梯度信息
        a = torch.tensor(1.0, requires_grad=True)
        
        # 使用早停策略设置检查点
        with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
            # 使用检查点运行函数 fn，传入参数 a，不允许重入
            x, out = checkpoint(fn, a, use_reentrant=False)
        
        # 注册钩子函数到 out 的梯度函数
        out.grad_fn.register_hook(hook)
        # 执行重新进入的反向传播
        out.backward(retain_graph=True)
    def test_nested_checkpoint_set_early_stop(self):
        # 定义一个计数器列表，用于在嵌套函数中记录调用次数
        counter = [0]

        # 定义一个内部函数，用于对输入进行克隆并增加计数器
        def clone(x):
            counter[0] += 1
            return x.clone()

        # 定义一个函数，对输入进行一系列操作，并调用克隆函数
        def fn(x):
            # 由于克隆函数不保存任何信息，当启用早停机制时，不会重新计算
            return clone(x.sin().cos())

        # 默认情况下启用早停机制
        a = torch.tensor(1.0, requires_grad=True)
        # 使用checkpoint函数对fn进行计算，返回的结果需要进行反向传播
        out = checkpoint(fn, a, use_reentrant=False)
        out.backward()
        # 断言计数器值为1，表明在早停机制下克隆函数仅被调用一次
        self.assertEqual(counter[0], 1)

        # 尝试使用上下文管理器将早停机制设置为False
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            # 在早停机制被设置为False的上下文中再次调用checkpoint函数
            out = checkpoint(fn, a, use_reentrant=False)

        out.backward()
        # 断言计数器值为2，表明在早停机制被设置为False的上下文中克隆函数被调用了两次
        self.assertEqual(counter[0], 2)
    def test_nested_checkpoint_set_early_stop_no_recompution_needed(self):
        # 定义测试函数：测试在不需要重新计算的情况下设置了嵌套检查点和早停功能

        # Case 1: We have one tensor saved and its the input
        # 案例1：我们保存了一个张量作为输入

        # We have two different counters here because in this case we actually
        # do call into x.sin() at the python level during recomputation whether
        # or not early stop is enabled. This is because the early stopping
        # only happens at the autograd level (preventing us from reaching the
        # backend).
        # 这里有两个不同的计数器，因为在这种情况下，无论早停是否启用，我们实际上都在Python级别调用了 x.sin() 进行重新计算。
        # 这是因为早停仅在自动求导级别发生（防止我们达到后端）。

        python_dispatch_counter = [0]
        # Python 分发计数器，用于统计调用特定函数的次数
        counter = [0]
        # 普通计数器，用于统计函数 fn 被调用的次数

        class SinCounterMode(TorchDispatchMode):
            def __init__(self):
                self.count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = {} if kwargs is None else kwargs
                if func is torch.ops.aten.sin.default:
                    self.count += 1
                return func(*args, **kwargs)

        def fn(x):
            counter[0] += 1
            return x.sin()

        # With early stopping (enabled by default)
        # 使用早停（默认启用）
        a = torch.tensor(1.0, requires_grad=True)
        with SinCounterMode() as python_dispatch_counter:  # noqa: F811
            out = checkpoint(fn, a, use_reentrant=False)
            out.backward()
        self.assertEqual(counter[0], 2)
        self.assertEqual(python_dispatch_counter.count, 1)

        # Without early stopping
        # 禁用早停
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with SinCounterMode() as python_dispatch_counter:
            with torch.utils.checkpoint.set_checkpoint_early_stop(False):
                out = checkpoint(fn, a, use_reentrant=False)
            out.backward()
        self.assertEqual(counter[0], 2)
        self.assertEqual(python_dispatch_counter.count, 2)

        # Case 2: Forward saves no tensors
        # 案例2：前向传播不保存任何张量

        # Since unpack isn't even called, counter is 1 whether or not early stop
        # is enabled!
        # 由于未调用 unpack，无论早停是否启用，counter 都是1！

        counter = [0]

        def fn2(x):
            counter[0] += 1
            return x.clone()

        # With early stopping (enabled by default)
        # 使用早停（默认启用）
        a = torch.tensor(1.0, requires_grad=True)
        out = checkpoint(fn2, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 1)

        # Without early stopping
        # 禁用早停
        counter = [0]
        a = torch.tensor(1.0, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            out = checkpoint(fn2, a, use_reentrant=False)
        out.backward()
        self.assertEqual(counter[0], 1)
class TestSelectiveActivationCheckpoint(TestCase):
    # 如果未开启 CUDA，则跳过此测试
    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    # 在 Torch Dynamo 中跳过这个测试，因为已经在 test/dynamo/test_activation_checkpointing.py 中测试过编译
    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_output_already_has_autograd_meta(self):
        # 定义一个函数 fn，接受两个参数 x 和 y，并返回 x 的视图和 y.sin().cos() 的结果
        def fn(x, y):
            return x.view(-1), y.sin().cos()

        # 创建一个整型张量 x，内容为 [1, 2, 3]，数据类型为 torch.int64
        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        # 创建一个形状为 (3,) 的随机张量 y，并设置 requires_grad=True
        y = torch.randn(3, requires_grad=True)

        # 使用 functools.partial 创建一个 context_fn 函数，传入一个包含 torch.ops.aten.view.default 的列表
        context_fn = functools.partial(
            create_selective_checkpoint_contexts,
            [
                torch.ops.aten.view.default,
            ],
        )
        # 使用 checkpoint 函数对 fn 进行检查点操作，传入 x, y 和一些参数
        out = checkpoint(fn, x, y, use_reentrant=False, context_fn=context_fn)
        # 对 out[1] 的所有元素求和并进行反向传播
        out[1].sum().backward()

    # 在 Torch Dynamo 中跳过这个测试，因为已经在 test/dynamo/test_activation_checkpointing.py 中测试过编译
    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_bad_inputs(self):
        # 创建一个包含数字 2 的列表 bad_op_list1
        bad_op_list1 = [2]

        # 使用 assertRaisesRegex 断言 bad_op_list1 调用 create_selective_checkpoint_contexts 时抛出 ValueError 异常
        with self.assertRaisesRegex(
            ValueError, "Expected op in `op_list` to be an OpOverload"
        ):
            create_selective_checkpoint_contexts(bad_op_list1)

        # 创建一个包含 torch.ops.aten.sin 的列表 bad_op_list2
        bad_op_list2 = [torch.ops.aten.sin]

        # 使用 assertRaisesRegex 断言 bad_op_list2 调用 create_selective_checkpoint_contexts 时抛出 ValueError 异常
        with self.assertRaisesRegex(
            ValueError, "update the OpOverloadPacket to a specific OpOverload"
        ):
            create_selective_checkpoint_contexts(bad_op_list2)

        # 使用 assertRaisesRegex 断言调用 create_selective_checkpoint_contexts 时传入整数 2 会抛出 TypeError 异常
        with self.assertRaisesRegex(TypeError, "either a function or a list of ops."):
            create_selective_checkpoint_contexts(2)

    # Dynamo 失败的各种原因：
    # - 某些测试使用不实现 Fake 的自定义操作
    # - dynamo 尝试跟踪保存的变量 hooks 的 unpack hook，原因不明
    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_policy_with_state(self):
        # 测试带有状态的策略函数
        # 如果我有一个有状态的可调用对象，在原始前向传播和重新计算之间共享状态。

        counters = []

        class Policy:
            def __init__(self):
                self.counter = [0]
                self.recompute_counter = [0]

            def __call__(self, ctx, func, *args, **kwargs):
                # 根据上下文选择计数器
                counter = self.recompute_counter if ctx.is_recompute else self.counter
                counter[0] += 1
                counters.append(counter[0])
                # 如果是第一次计数且函数是 torch.ops.aten.mm.default，则必须保存
                if counter == 1 and func is torch.ops.aten.mm.default:
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            # 进行三次正弦函数嵌套调用
            return x.sin().sin().sin()

        x = torch.randn(3, requires_grad=True)
        # 创建选择性检查点上下文的部分函数
        context_fn = functools.partial(
            create_selective_checkpoint_contexts,
            Policy(),
            allow_cache_entry_mutation=True,
        )
        # 使用检查点进行函数调用，不支持可重入
        out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
        # 对结果进行求和并进行反向传播
        out.sum().backward()
        # 1. 重新计算时计数器正确重置为0
        # 2. 由于提前停止，我们不会重新计算最后一个操作
        self.assertEqual(counters, [1, 2, 3, 1, 2])
    # 定义测试函数 test_storage_lifetime(self)，用于测试存储的生命周期
    def test_storage_lifetime(self):
        # 导入必要的模块和函数
        from torch.utils._python_dispatch import _get_current_dispatch_mode
        from torch.utils.checkpoint import (
            _CachedTorchDispatchMode,
            _CachingTorchDispatchMode,
        )

        # 定义策略函数 policy_fn，始终返回必须保存的检查点策略
        def policy_fn(ctx, op, *args, **kwargs):
            return CheckpointPolicy.MUST_SAVE

        # 初始化引用变量 ref 为 None
        ref = None

        # 定义函数 fn(x)，处理输入张量 x
        def fn(x):
            nonlocal ref

            # 断言当前的调度模式是缓存或者已缓存的 Torch 调度模式之一
            self.assertIsInstance(
                _get_current_dispatch_mode(),
                (_CachingTorchDispatchMode, _CachedTorchDispatchMode),
            )

            # 对输入张量 x 执行 cos() 和 exp() 操作，得到结果 out
            out = x.cos().exp()

            # 如果当前的调度模式是 CachingTorchDispatchMode，则进行如下操作
            if isinstance(_get_current_dispatch_mode(), _CachingTorchDispatchMode):
                # 获取当前调度模式下 exp 操作的默认存储，检索其值
                raw_val = (
                    _get_current_dispatch_mode()
                    .storage[torch.ops.aten.exp.default][0]
                    .val
                )
                # 断言 raw_val 已被分离（detached），以避免图 -> 保存的变量钩子 -> 重新计算上下文 -> 存储 -> 图的引用循环
                self.assertFalse(raw_val.requires_grad)
                # 使用 weakref 创建 raw_val 的弱引用，并赋值给 ref
                ref = weakref.ref(raw_val)

            # 返回 out 的 sin() 操作结果
            return out.sin()

        # 禁用垃圾回收
        with disable_gc():
            # Case 1: 如果图形在没有反向传播的情况下消失，请确保没有保持存储活动的引用循环。
            #         创建一个需要梯度的随机张量 x
            x = torch.randn(3, requires_grad=True)
            # 创建选择性检查点上下文的函数 context_fn，部分应用策略函数 policy_fn
            context_fn = functools.partial(
                create_selective_checkpoint_contexts, policy_fn
            )
            # 调用 checkpoint 函数，传入 fn 函数、输入张量 x、禁用重入的标志 use_reentrant=False 和上下文函数 context_fn
            out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
            # 断言 ref() 不为 None，即保证 ref 引用有效
            self.assertIsNotNone(ref())
            # 删除 out 变量
            del out
            # 断言 ref() 为 None，确保在删除 out 后 ref 引用被释放
            self.assertIsNone(ref())

            # Case 2: 在执行反向传播后，即使 retain_graph=True，存储也应该被释放
            # 创建新的需要梯度的随机张量 x
            x = torch.randn(3, requires_grad=True)
            # 再次创建选择性检查点上下文的函数 context_fn，部分应用策略函数 policy_fn
            context_fn = functools.partial(
                create_selective_checkpoint_contexts, policy_fn
            )
            # 再次调用 checkpoint 函数，传入 fn 函数、输入张量 x、禁用重入的标志 use_reentrant=False 和上下文函数 context_fn
            out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
            # 断言 ref() 不为 None，即保证 ref 引用有效
            self.assertIsNotNone(ref())
            # 对 out 的所有元素求和并执行反向传播，保留计算图 retain_graph=True
            out.sum().backward(retain_graph=True)
            # 断言 ref() 为 None，确保在执行反向传播后，ref 引用被释放
            self.assertIsNone(ref())

    # 跳过 Torch Dynamo 测试的装饰器
    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_version_counter(self):
        # 定义一个用于策略的函数，根据操作选择是否保存检查点
        def policy_fn(ctx, op, *args, **kwargs):
            if op == torch.ops.aten.sin.default:
                return CheckpointPolicy.MUST_SAVE  # 如果是 sin 操作，必须保存检查点
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE  # 否则，推荐重新计算

        # 定义一个操作函数 fn，对输入张量进行一系列数学运算
        def fn(x):
            return x.sin().mul_(2).cos().exp()

        # 创建一个随机张量 x，并要求计算其梯度
        x = torch.randn(3, requires_grad=True)

        # 创建部分函数 context_fn，使用指定的策略函数来生成选择性检查点上下文
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)

        # 使用 checkpoint 函数对 fn 函数进行检查点计算，不支持可重入，并使用指定的上下文函数
        out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)

        # 1) 因为 sin 的输出被保存并被 mul_ 修改导致错误
        with self.assertRaisesRegex(RuntimeError, "has been mutated"):
            out.sum().backward()

        # 重新生成一个随机张量 x，并要求计算其梯度
        x = torch.randn(3, requires_grad=True)

        # 创建部分函数 context_fn，使用指定的策略函数和允许缓存条目变异参数来生成选择性检查点上下文
        context_fn = functools.partial(
            create_selective_checkpoint_contexts,
            policy_fn,
            allow_cache_entry_mutation=True,
        )

        # 使用 checkpoint 函数对 fn 函数进行检查点计算，不支持可重入，并使用指定的上下文函数
        out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)

        # 2) 因为允许缓存条目变异，不再会出现错误
        out.sum().backward()

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_function_with_more_than_one_output(self):
        # 可能有更系统化的方法：
        counter = [0]

        # 定义一个用于策略的函数，根据操作选择是否保存检查点，并增加计数
        def policy_fn(ctx, op, *args, **kwargs):
            if op == torch.ops.aten.var_mean.correction:
                counter[0] += 1
                return CheckpointPolicy.MUST_SAVE  # 如果是 var_mean 操作，必须保存检查点
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE  # 否则，推荐重新计算

        # var_mean 操作有两个输出
        def fn(x):
            a, b = torch.var_mean(x)
            return a * b

        # 创建一个随机张量 x，并要求计算其梯度
        x = torch.randn(3, requires_grad=True)

        # 创建部分函数 context_fn，使用指定的策略函数来生成选择性检查点上下文
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)

        # 使用 checkpoint 函数对 fn 函数进行检查点计算，不支持可重入，并使用指定的上下文函数
        out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)

        # 计算张量 x 的梯度，并参考未使用检查点的 fn(x) 的梯度
        x_grad = torch.autograd.grad(out.sum(), (x,))
        x_grad_ref = torch.autograd.grad(fn(x).sum(), (x,))

        # 断言计数器值和预期的不一致性次数相等
        self.assertEqual(counter[0], 2)

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_function_with_non_tensor_output(self):
        # SAC 启用时，操作不会再次计算
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            counter = [0]

            @torch.library.custom_op("mylib::sin_with_extra", mutates_args=())
            def sin_with_extra(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
                counter[0] += 1
                return x.sin(), 2

            def setup_context(ctx, inputs, output) -> torch.Tensor:
                (x,) = inputs
                ctx.save_for_backward(x)

            def backward(ctx, grad, _unused):
                (x,) = ctx.saved_tensors
                return grad * x.cos()

            torch.library.register_autograd(
                "mylib::sin_with_extra", backward, setup_context=setup_context
            )

            x = torch.randn(3, requires_grad=True)

            def fn(x):
                # 调用自定义操作并对结果进行复杂的张量运算
                return (torch.ops.mylib.sin_with_extra(x)[0] * x.sin().exp()).sin()

            ops_list = [torch.ops.mylib.sin_with_extra.default]

            x = torch.randn(3, requires_grad=True)
            context_fn = functools.partial(
                create_selective_checkpoint_contexts, ops_list
            )
            # 使用检查点技术计算函数输出，避免重复计算
            out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
            # 计算函数输出对输入 x 的梯度
            x_grad = torch.autograd.grad(out.sum(), (x,))
            self.assertEqual(counter[0], 1)
            # 计算参考梯度
            x_grad_ref = torch.autograd.grad(fn(x).sum(), (x,))
            self.assertEqual(x_grad, x_grad_ref)

    @skipIfTorchDynamo("compile tested in test/dynamo/test_activation_checkpointing.py")
    def test_can_only_trigger_recompute_once(self):
        # 目前不支持这个功能，以避免增加额外的复杂性。
        # 如果有需求，可能可以进行某种使用计数跟踪。
        # TODO: 在这里添加一个友好的错误消息。
        def policy_fn(ctx, op, *args, **kwargs):
            if op == torch.ops.aten.sin.default:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            # 执行复杂的张量操作
            return x.sin().cos().exp()

        x = torch.randn(3, requires_grad=True)
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        # 使用检查点技术计算函数输出，避免重复计算
        out = checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)
        # 对函数输出求和进行反向传播
        out.sum().backward(retain_graph=True)

        with self.assertRaisesRegex(RuntimeError, "Trying to backward an extra time"):
            # 尝试再次对输出求和进行反向传播，应该触发异常
            out.sum().backward(retain_graph=True)
class TestAutogradMultipleDispatch(TestCase):
    # 定义测试类 TestAutogradMultipleDispatch，继承自 TestCase

    def test_autograd_multiple_dispatch_registrations(self, device):
        # 定义测试方法 test_autograd_multiple_dispatch_registrations，接受参数 device

        t = torch.randn(3, 3, device=device, requires_grad=True)
        # 创建一个大小为 3x3 的张量 t，在指定设备上，并设置 requires_grad 为 True

        # 使用 _test_autograd_multiple_dispatch.fullcoverage 进行调用，
        # 其中 derivatives.yaml 中有 Default、AutogradCUDA 和 NestedTensorAutograd 的注册
        out = torch._test_autograd_multiple_dispatch(t)
        # 调用 _test_autograd_multiple_dispatch 方法，并传入张量 t，得到输出 out

        grad = torch.randn(3, 3, device=device)
        # 创建一个与 t 同样大小的张量 grad，用于梯度计算

        out.backward(grad)
        # 对 out 进行反向传播，传入梯度 grad

        if "cuda" not in device:
            # 如果设备不包含 "cuda"
            # Autograd 的伪梯度默认为 grad + 1
            self.assertEqual(t.grad, grad + 1)
        else:
            # 如果设备包含 "cuda"
            # AutogradCUDA 的伪梯度默认为 grad * 2
            self.assertEqual(t.grad, grad * 2)

        # 测试注册的 AutogradNestedTensor 公式
        a = (
            torch.arange(6, dtype=torch.float, device=device)
            .reshape(2, 3)
            .requires_grad_(True)
        )
        # 创建一个大小为 2x3 的张量 a，使用设备 device，并设置 requires_grad 为 True

        b = (
            torch.arange(8, dtype=torch.float, device=device)
            .reshape(2, 4)
            .requires_grad_(True)
        )
        # 创建一个大小为 2x4 的张量 b，使用设备 device，并设置 requires_grad 为 True

        nt = torch.nested.as_nested_tensor([a, b], dtype=torch.float, device=device)
        # 将张量 a 和 b 转换为嵌套张量 nt，使用指定的 dtype 和 device

        nt_out = torch._test_autograd_multiple_dispatch(nt)
        # 调用 _test_autograd_multiple_dispatch 方法，并传入嵌套张量 nt，得到 nt_out

        c = torch.randn(2, 3, device=device)
        # 创建一个与 a 相同大小的张量 c，用于梯度计算

        d = torch.randn(2, 4, device=device)
        # 创建一个与 b 相同大小的张量 d，用于梯度计算

        nt_grad = torch.nested.nested_tensor([c, d], dtype=torch.float, device=device)
        # 将张量 c 和 d 转换为嵌套张量 nt_grad，使用指定的 dtype 和 device

        nt_out.backward(nt_grad)
        # 对 nt_out 进行反向传播，传入嵌套张量 nt_grad

        # AutogradNestedTensor 的伪梯度为 grad * grad
        self.assertEqual(a.grad, c * c)
        self.assertEqual(b.grad, d * d)
    # 测试自动微分的复合隐式和调度注册
    def test_autograd_composite_implicit_and_dispatch_registration(self, device):
        # 创建一个形状为 (3, 3) 的张量 t，随机初始化，设置 requires_grad=True
        t = torch.randn(3, 3, device=device, requires_grad=True)
        
        # 使用 _test_autograd_multiple_dispatch.ntonly 进行操作
        # 在 derivatives.yaml 中，_test_autograd_multiple_dispatch.ntonly 注册了 NestedTensorAutograd，
        # 否则是 CompositeImplicit
        out = torch._test_autograd_multiple_dispatch(t, True)
        
        # 创建一个形状为 (3, 3) 的梯度张量 grad
        grad = torch.randn(3, 3, device=device)
        
        # 对 out 进行反向传播，计算梯度
        out.backward(grad)

        # t.grad 应当等于 grad，因为 _test_autograd_multiple_dispatch 是一个克隆操作
        self.assertEqual(t.grad, grad)

        # 测试注册的 AutogradNestedTensor 公式
        # 创建形状为 (2, 3) 的张量 a 和形状为 (2, 4) 的张量 b
        a = (
            torch.arange(6, dtype=torch.float, device=device)
            .reshape(2, 3)
            .requires_grad_(True)
        )
        b = (
            torch.arange(8, dtype=torch.float, device=device)
            .reshape(2, 4)
            .requires_grad_(True)
        )
        
        # 使用 as_nested_tensor 将 a 和 b 转换为嵌套张量 nt
        nt = torch.nested.as_nested_tensor([a, b], dtype=torch.float, device=device)

        # 对 nt 进行 _test_autograd_multiple_dispatch 操作
        nt_out = torch._test_autograd_multiple_dispatch(nt, True)
        
        # 创建形状为 (2, 3) 和 (2, 4) 的随机张量 c 和 d，作为 nt 的梯度
        c = torch.randn(2, 3, device=device)
        d = torch.randn(2, 4, device=device)
        nt_grad = torch.nested.nested_tensor([c, d], dtype=torch.float, device=device)
        
        # 对 nt_out 进行反向传播，计算梯度
        nt_out.backward(nt_grad)

        # AutogradNestedTensor 的虚假梯度应为 grad * grad + grad
        self.assertEqual(a.grad, c * c + c)
        self.assertEqual(b.grad, d * d + d)

    # 测试前向模式自动微分
    def test_foward_mode_AD(self, device):
        # 检查仅对 _test_autograd_multiple_dispatch.fullcoverage 的默认调度注册了前向模式 AD，
        # 而 AutogradCUDA 没有注册
        
        # 创建一个形状为 (3,) 的随机张量 primal 和 tangent
        primal = torch.randn(3, device=device)
        tangent = torch.randn(3, device=device)

        # 使用 fwAD.dual_level() 进入双重级别上下文
        with fwAD.dual_level():
            # 创建一个双重输入 dual_input，使用 primal 和 tangent
            dual_input = fwAD.make_dual(primal, tangent)

            # 如果设备中包含 "cuda"，则应该抛出 NotImplementedError 异常，带有特定的错误消息 err_msg
            # 提示信息 hint_msg 为 "Running forward AD for an OP that does not implement it should raise a NotImplementedError"
            if "cuda" in device:
                with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                    torch._test_autograd_multiple_dispatch(dual_input)
            else:
                # 否则，对 dual_input 进行 _test_autograd_multiple_dispatch 操作
                torch._test_autograd_multiple_dispatch(dual_input)
    def test_view_copy(self, device):
        # tests that view_copy derivative formulas are also generated per dispatch key
        # from their respective view ops in derivatives.yaml
        
        # 创建一个形状为 (2, 2) 的张量，设备为指定的 device，启用梯度计算
        t = torch.randn(2, 2, device=device, requires_grad=True)
        # 克隆张量 t，并分离计算图，同时保留梯度计算
        t_ref = t.clone().detach().requires_grad_()
        
        # 调用 _test_autograd_multiple_dispatch_view 对 t_ref 进行 .view(-1) 操作
        t_view = torch._test_autograd_multiple_dispatch_view(t_ref)
        # 调用 _test_autograd_multiple_dispatch_view_copy 对 t 进行视图复制操作
        t_view_copy = torch._test_autograd_multiple_dispatch_view_copy(t)
        
        # 创建一个形状为 (4,) 的随机梯度张量，设备为指定的 device
        grad = torch.randn(4, device=device)
        # 对 t_view_copy 进行反向传播，传入梯度 grad
        t_view_copy.backward(grad)
        # 对 t_view 进行反向传播，传入梯度 grad 的克隆
        t_view.backward(grad.clone())
        
        # 断言 t_view_copy 和 t_view 的形状和数值相等
        self.assertEqual(t_view_copy, t_view)
        # 断言 t 的梯度和 t_ref 的梯度相等
        self.assertEqual(t.grad, t_ref.grad)
        
        # 根据设备类型检查梯度注册是否符合 derivatives.yaml 中的预期
        if "cuda" in device:
            # 对于 AutogradCUDA，预期梯度是 grad.reshape_as(t) + 1
            self.assertEqual(t.grad, grad.reshape_as(t) + 1)
        else:
            # 默认情况下，预期梯度是 grad.reshape_as(t)
            self.assertEqual(t.grad, grad.reshape_as(t))

    @onlyCPU
    def test_per_dispatch_key_input_saving(self, device):
        # Tests that sum.dim_IntList's input is not saved for regular tensors but is saved for nested tensors
        
        def foo(x):
            # 不直接修改输入 inplace
            x = x.clone()
            # 对输入张量 x 按指定维度 -1 进行求和，并保持维度
            res = x.sum(-1, keepdim=True)
            # 将 x 自身加倍
            x.add_(x)
            return res
        
        # 创建一个形状为 (2,) 的随机张量，设备为指定的 device，启用梯度计算
        inp = torch.rand(2, device=device, requires_grad=True)
        # 对 regular tensors 进行反向传播，预期 sum 的输入不会被保存
        foo(inp).backward()
        
        # 创建一个嵌套张量 nt，包含两个形状为 (2,) 的随机张量，设备为指定的 device，启用梯度计算
        nt = torch.nested.nested_tensor(
            [torch.rand(2), torch.rand(2)], device=device, requires_grad=True
        )
        # 对 nested tensors 进行反向传播，预期 sum 的输入会被保存
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            foo(nt).backward(
                torch.nested.nested_tensor(
                    [torch.rand(1), torch.rand(1)], device=device
                )
            )

    @onlyCUDA
    def test_backward_single_threaded(self):
        threads_eq = None
        
        # 定义一个自定义 Function 类 TestFn
        class TestFn(Function):
            @staticmethod
            def forward(ctx, x, self):
                # 保存自身和线程 ID 到上下文 ctx 中
                ctx.self = self
                ctx.tid = threading.get_ident()
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                nonlocal threads_eq
                # 检查当前线程 ID 是否与保存的线程 ID 相同
                threads_eq = ctx.tid == threading.get_ident()
                return gO, None
        
        # 创建一个形状为 (10,) 的随机张量，设备为 "cuda"，启用梯度计算
        inp = torch.rand(10, device="cuda", requires_grad=True)
        
        # 禁用多线程
        with torch.autograd.set_multithreading_enabled(False):
            # 应用自定义函数 TestFn，对输入 inp 执行求和并反向传播
            TestFn.apply(inp, None).sum().backward()
        # 断言多线程是否相等
        self.assertTrue(threads_eq)
        
        # 默认情况下启用多线程
        TestFn.apply(inp, None).sum().backward()
        # 断言多线程不相等
        self.assertFalse(threads_eq)
    def test_backward_tls_stash(self):
        # 创建一个线程本地存储对象
        local = threading.local()
        # 在线程本地存储对象中创建一个空字典
        local.my_obj = {}
        # 向本地存储对象中的字典添加一个键值对
        local.my_obj[10] = 10
        # 将当前测试对象的引用保存在 test_self 中
        test_self = self
        # 将 local.my_obj 存储在线程本地存储中，键为 "my_obj"
        torch._C._stash_obj_in_tls("my_obj", local.my_obj)

        # 定义一个继承自 Function 的测试函数类
        class TestFn(Function):
            @staticmethod
            def forward(ctx, x, self):
                # 返回输入张量的副本
                return x.clone()

            @staticmethod
            def backward(ctx, gO):
                # 断言 "my_obj" 是否存在于线程本地存储中
                test_self.assertTrue(torch._C._is_key_in_tls("my_obj"))
                # 断言从线程本地存储中获取的 "my_obj" 中的值为 10
                test_self.assertTrue(torch._C._get_obj_in_tls("my_obj")[10] == 10)
                # 修改线程本地存储中 "my_obj" 的值为 5
                torch._C._get_obj_in_tls("my_obj")[10] = 5
                return gO, None

        # 创建一个在 CUDA 设备上的随机张量，并要求计算梯度
        inp = torch.rand(10, device="cuda", requires_grad=True)

        # 调用 TestFn 的 apply 方法进行前向计算和反向传播
        TestFn.apply(inp, None).sum().backward()
        # 断言本地存储中 "my_obj" 的值已经被修改为 5
        self.assertEqual(local.my_obj[10], 5)

    def test_set_sequence_nr(self):
        # 创建三个随机张量，并要求计算梯度
        x = torch.randn((10,), dtype=torch.float32, requires_grad=True)
        y = torch.randn((10,), dtype=torch.float32, requires_grad=True)
        z = torch.randn((10,), dtype=torch.float32, requires_grad=True)

        # 执行张量的加法操作
        a = x + y
        b = y + z
        c = a + b

        # 断言张量 a, b, c 的梯度函数不为空
        self.assertIsNotNone(a.grad_fn)
        self.assertIsNotNone(b.grad_fn)
        self.assertIsNotNone(c.grad_fn)

        # 设置张量 a, b, c 的梯度函数的序列号
        a.grad_fn._set_sequence_nr(100)
        b.grad_fn._set_sequence_nr(99)
        c.grad_fn._set_sequence_nr(98)

        # 断言张量 a, b, c 的梯度函数序列号已经设置成功
        self.assertEqual(a.grad_fn._sequence_nr(), 100)
        self.assertEqual(b.grad_fn._sequence_nr(), 99)
        self.assertEqual(c.grad_fn._sequence_nr(), 98)

        # 定义一个记录梯度计算顺序的函数
        def log_grad_order(grad: torch.Tensor, name: str, order):
            order.append(name)
            return grad

        # 创建一个空列表 order
        order = []
        # 注册张量 a, b, c 的反向传播钩子，记录计算顺序
        a.register_hook(partial(log_grad_order, name="a", order=order))
        b.register_hook(partial(log_grad_order, name="b", order=order))
        c.register_hook(partial(log_grad_order, name="c", order=order))

        # 对张量 c 执行求和并进行反向传播
        c.sum().backward()

        # 断言反向传播的计算顺序为 ["c", "a", "b"]
        self.assertEqual(order, ["c", "a", "b"])

        # 断言张量 x, y, z 的梯度计算正确
        self.assertEqual(x.grad, torch.ones_like(x))
        self.assertEqual(y.grad, 2 * torch.ones_like(x))
        self.assertEqual(z.grad, torch.ones_like(x))
# 从 autograd.test_complex 模块导入 TestAutogradComplex 类，用于测试复杂情况下的自动微分功能
# noqa: F401 表示告知 Flake8 忽略未使用的导入警告
from autograd.test_complex import TestAutogradComplex  # noqa: F401

# 从 autograd.test_functional 模块导入 TestAutogradFunctional 类，用于测试自动微分的函数式功能
# noqa: F401 表示告知 Flake8 忽略未使用的导入警告
from autograd.test_functional import TestAutogradFunctional  # noqa: F401

# 从 autograd.test_logging 模块导入 TestAutogradLogging 类，用于测试自动微分的日志记录功能
# noqa: F401 表示告知 Flake8 忽略未使用的导入警告
from autograd.test_logging import TestAutogradLogging  # noqa: F401

# 实例化 TestAutogradDeviceType 类的测试用例，这些测试用例会根据设备类型（CPU 或 CUDA）自动加载
instantiate_device_type_tests(TestAutogradDeviceType, globals(), except_for=None)

# 实例化 TestAutogradMultipleDispatch 类的测试用例，仅适用于 CPU 和 CUDA 设备类型
instantiate_device_type_tests(
    TestAutogradMultipleDispatch, globals(), only_for=("cpu", "cuda")
)

# 实例化 TestAutograd 类的参数化测试用例
instantiate_parametrized_tests(TestAutograd)

# 实例化 TestNestedCheckpoint 类的参数化测试用例
instantiate_parametrized_tests(TestNestedCheckpoint)

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```