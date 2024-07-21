# `.\pytorch\test\functorch\test_eager_transforms.py`

```py
# 导入标准库和第三方库
import copy  # 导入深拷贝模块
import math  # 导入数学函数模块
import os  # 导入操作系统接口模块
import subprocess  # 导入子进程管理模块
import sys  # 导入系统相关功能模块
import unittest  # 导入单元测试模块
import warnings  # 导入警告处理模块
from functools import partial, wraps  # 导入偏函数和装饰器功能

# 导入第三方库 numpy
import numpy as np  # 导入数值计算库 numpy
from common_utils import expectedFailureIf  # 从本地导入自定义的测试工具函数

# 导入 PyTorch 和 Functorch 相关模块
import functorch  # 导入 functorch 库
import torch  # 导入 PyTorch 深度学习框架
import torch.autograd.forward_ad as fwAD  # 导入自动求导模块
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入神经网络函数模块
from functorch import (  # 从 functorch 导入多个函数和类
    combine_state_for_ensemble,  # 组合集成状态的函数
    grad,  # 梯度函数
    grad_and_value,  # 梯度和值的计算函数
    hessian,  # 海森矩阵函数
    jacfwd,  # 前向雅各比矩阵函数
    jacrev,  # 反向雅各比矩阵函数
    jvp,  # 雅各比向量积函数
    make_functional,  # 创建函数化对象的函数
    make_functional_with_buffers,  # 创建带缓冲区的函数化对象的函数
    make_fx,  # 创建 FX 功能的函数
    vjp,  # 倒向雅各比矩阵函数
    vmap,  # 向量映射函数
)
from functorch.experimental import functionalize, replace_all_batch_norm_modules_  # 导入实验性功能模块
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet  # 导入 Torch C++ 扩展模块相关内容
from torch._dynamo import allow_in_graph  # 导入 Torch 动态图计算相关功能
from torch._functorch.eager_transforms import _slice_argnums  # 导入函数式 Torch 的特定函数
from torch._functorch.make_functional import (  # 导入创建函数化对象的相关函数
    functional_init,  # 初始化函数化对象的函数
    functional_init_with_buffers,  # 初始化带缓冲区的函数化对象的函数
)
from torch._functorch.utils import enable_single_level_autograd_function  # 导入 Torch 功能扩展工具函数
from torch._ops import HigherOrderOperator  # 导入高阶操作器
from torch._subclasses.fake_tensor import FakeTensorMode  # 导入虚张量模式
from torch.func import functional_call, linearize, stack_module_state  # 导入函数式调用相关函数
from torch.testing import make_tensor  # 导入测试相关的张量创建函数
from torch.testing._internal.common_cuda import (  # 导入测试 CUDA 相关的公共模块和函数
    SM70OrLater,  # 标识是否为 SM70 架构或更新版本
    TEST_CUDA,  # 是否进行 CUDA 测试
    tf32_on_and_off,  # 是否开启或关闭 TF32 模式
    with_tf32_off,  # 开启 TF32 模式的上下文管理器
)
from torch.testing._internal.common_device_type import (  # 导入测试设备类型相关模块和函数
    dtypes,  # 支持的数据类型
    instantiate_device_type_tests,  # 实例化设备类型测试的函数
    onlyCPU,  # 仅限 CPU 环境的测试装饰器
    onlyCUDA,  # 仅限 CUDA 环境的测试装饰器
)
from torch.testing._internal.common_dtype import get_all_fp_dtypes  # 导入获取所有浮点数类型的函数
from torch.testing._internal.common_utils import (  # 导入通用测试工具模块和函数
    freeze_rng_state,  # 冻结随机数生成器状态的函数
    instantiate_parametrized_tests,  # 实例化参数化测试的函数
    IS_FBCODE,  # 是否在 Facebook 代码中运行的标志
    IS_WINDOWS,  # 是否在 Windows 操作系统上运行的标志
    markDynamoStrictTest,  # 标记为动态图的严格测试的装饰器
    parametrize,  # 参数化测试的装饰器
    run_tests,  # 运行测试的函数
    skipIfRocm,  # 在 ROCm 环境下跳过测试的装饰器
    skipIfTorchDynamo,  # 在 Torch 动态图环境下跳过测试的装饰器
    subtest,  # 子测试函数
    TEST_WITH_TORCHDYNAMO,  # 是否使用 Torch 动态图进行测试的标志
    TestCase,  # 单元测试的基类
    xfailIfTorchDynamo,  # 在 Torch 动态图环境下标记为失败的装饰器
)

from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten  # 导入 PyTree 模块中的函数

# 设置是否使用 torchvision 标志，默认为 False
USE_TORCHVISION = False
try:
    import torchvision  # 尝试导入 torchvision 库

    USE_TORCHVISION = True  # 若成功导入，则设置使用 torchvision 为 True
except ImportError:
    warnings.warn(
        "Couldn't import torchvision. Some of our tests use it, try "
        "to install it with commands from pytorch.org, post-fixed with "
        "`--no-deps` to avoid overwriting the pytorch installation",
        UserWarning,
    )

# 定义 TestCase 类，用于测试 _slice_argnums 函数的辅助函数
class VmapTearDownMixin:
    # 确保在测试失败时，下一个测试不会因为之前未撤销的 _vmap_increment_nesting 调用而失败
    # 例如，当 PYTORCH_TEST_WITH_DYNAMO=1 时，test_vmap_free_tensor 测试失败且未撤销增加嵌套调用
    if not TEST_WITH_TORCHDYNAMO:
        return

    # 初始化警告标志
    warn = False

    # 检查并撤销解释器堆栈中的 Vmap 变换类型，直到堆栈为空或者遇到非 Vmap 变换
    while ci := torch._C._functorch.peek_interpreter_stack():
        if ci.key() == torch._C._functorch.TransformType.Vmap:
            warn = True
            torch._C._functorch._vmap_decrement_nesting()
        else:
            break

    # 如果有进行过撤销的操作，则发出警告
    if warn:
        msg = (
            "Interpreter stack is not empty. Test should have called "
            "'torch._C._functorch._vmap_decrement_nesting()'"
        )
        warnings.warn(msg)
# 使用装饰器标记为 DynamoStrictTest 的测试类
@markDynamoStrictTest
class TestSliceArgnums(TestCase):
    # 测试无效的 argnum 类型
    def test_invalid_argnum_type(self):
        # 创建一个包含 3 个随机数字的张量
        x = torch.randn(3)
        # 将 x 打包成一个元组
        args = (x,)
        # 断言捕获 RuntimeError 异常并检查是否包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "int or Tuple"):
            _slice_argnums(args, 0.0)
        # 同上，但检查的异常信息不同
        with self.assertRaisesRegex(RuntimeError, "int or Tuple"):
            _slice_argnums(args, [0])
        # 同上，但这次传入的是元组，但元组里面包含了浮点数
        with self.assertRaisesRegex(RuntimeError, "must be int"):
            _slice_argnums(args, (0.0,))

        # 重新赋值 args
        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        # 检查是否抛出指定的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "must be int"):
            _slice_argnums(args, ((0, 1), 2))

    # 测试超出边界的 argnum 值
    def test_out_of_bounds_argnum_values(self):
        # 创建一个包含 3 个随机数字的张量
        x = torch.randn(3)
        # 将 x 打包成一个元组
        args = (x,)
        # 断言捕获 RuntimeError 异常并检查是否包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "positional inputs"):
            _slice_argnums(args, 1)
        # 同上，但检查的异常信息不同
        with self.assertRaisesRegex(RuntimeError, "positional inputs"):
            _slice_argnums(args, -2)
        # 同上，但这次传入的是负数作为元组中的值
        with self.assertRaisesRegex(RuntimeError, "positional inputs"):
            _slice_argnums(args, (-2,))

    # 测试不足的 argnum
    def test_not_enough_argnums(self):
        # 创建一个包含 3 个随机数字的张量
        x = torch.randn(3)
        # 将 x 打包成一个元组
        args = (x,)
        # 断言捕获 RuntimeError 异常并检查是否包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
            _slice_argnums(args, ())

    # 测试重复的 argnum
    def test_duplicate_argnums(self):
        # 创建一个包含 3 个随机数字的张量
        x = torch.randn(3)
        # 将 x 打包成一个元组，但重复使用 x
        args = (x, x)
        # 断言捕获 RuntimeError 异常并检查是否包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            _slice_argnums(args, (0, 0))
        # 同上，但检查的异常信息不同
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            _slice_argnums(args, (0, -2))

    # 测试带有正整数 argnum 的扁平参数
    def test_flat_args_with_positive_int_argnum(self):
        # 创建一个包含浮点数的元组
        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        # 调用 _slice_argnums 函数，传入参数和正整数 argnum
        res = _slice_argnums(args, 0)
        # 断言 res 是否等于预期的结果
        self.assertEqual(res, (0.1,))

        # 同上，但传入另一个正整数 argnum
        res = _slice_argnums(args, 4)
        # 断言 res 是否等于预期的结果
        self.assertEqual(res, (4.1,))

    # 测试带有负整数 argnum 的扁平参数
    def test_flat_args_with_negative_int_argnum(self):
        # 创建一个包含浮点数的元组
        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        # 调用 _slice_argnums 函数，传入参数和负整数 argnum
        res = _slice_argnums(args, -1)
        # 断言 res 是否等于预期的结果
        self.assertEqual(res, (4.1,))

        # 同上，但传入另一个负整数 argnum
        res = _slice_argnums(args, -5)
        # 断言 res 是否等于预期的结果
        self.assertEqual(res, (0.1,))

    # 测试带有元组 argnum 的扁平参数
    def test_flat_args_with_tuple_argnum(self):
        # 创建一个包含浮点数的元组
        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        # 调用 _slice_argnums 函数，传入参数和元组 argnum
        res = _slice_argnums(args, (0, 1, 2, 3, 4))
        # 断言 res 是否等于预期的结果
        self.assertEqual(res, args)

        # 同上，但传入另一个元组 argnum
        res = _slice_argnums(args, (0, -3))
        # 断言 res 是否等于预期的结果
        self.assertEqual(res, (0.1, 2.1))

    # 测试带有 pytree 参数的情况
    def test_pytree_args(self):
        # 创建一个包含元组、浮点数和列表的元组
        args = ((0.1, 1.1), 2.0, [3.1])

        # 调用 _slice_argnums 函数，传入参数和正整数 argnum
        res = _slice_argnums(args, 0)
        # 断言 res 是否等于预期的结果，即 args 的第一个元素
        self.assertEqual(res, args[0:1])

        # 同上，但传入一个包含一个元素的元组 argnum
        res = _slice_argnums(args, (0,))
        # 断言 res 是否等于预期的结果，即 args 的第一个元素
        self.assertEqual(res, args[0:1])

        # 同上，但传入负整数 argnum
        res = _slice_argnums(args, -1)
        # 断言 res 是否等于预期的结果，即 args 的最后一个元素
        self.assertEqual(res, args[-1:])

        # 同上，但传入一个包含两个整数的元组 argnum
        res = _slice_argnums(args, (0, -2))
        # 断言 res 是否等于预期的结果，即 args 的前两个元素
        self.assertEqual(res, args[0:2])

    # 测试 argnum 的重新排序
    def test_argnums_reorders(self):
        # 创建一个包含元组和浮点数的元组
        args = ((0.1, 1.1, 2.1), 3.1, 4.1)

        # 调用 _slice_argnums 函数，传入参数和元组 argnum
        res = _slice_argnums(args, (1, 0))
        # 断言 res 是否等于预期的结果，即 args 的第二个和第一个元素
        self.assertEqual(res, (args[1], args[0]))


# 定义一个用于获取权重和函数调用的私有函数，参数为网络和机制
def _get_weights_and_functional_call(net, mechanism):
    # 如果 mechanism 参数为 "make_functional"，则调用 make_functional 函数并返回其结果
    if mechanism == "make_functional":
        return make_functional(net)
    else:
        # 如果 mechanism 参数为 "functional_call"，进行断言确认
        assert mechanism == "functional_call"
        # 下面的定义确保从 make_functional 函数生成的函数和此处的调用具有相同的签名

        # 定义一个新的函数 net_func，接受 weights 和 data 两个参数，
        # 调用 functional_call 函数，并将 net、weights 和 data 作为参数传递进去
        def net_func(weights, data):
            return functional_call(net, weights, (data,))

        # 返回定义的 net_func 函数及 net 模型的命名参数组成的字典
        return net_func, dict(net.named_parameters())
# 定义一个函数，根据给定的网络和机制返回权重和使用缓冲区的功能调用或者一个函数
def _get_weights_and_functional_call_with_buffers(net, mechanism):
    # 如果机制为 "make_functional"，则调用 make_functional_with_buffers 函数并返回结果
    if mechanism == "make_functional":
        return make_functional_with_buffers(net)
    else:
        # 否则，确保机制为 "functional_call"
        assert mechanism == "functional_call"

        # 定义一个函数 net_func，使其具有与 make_functional 中函数相同的签名
        def net_func(weights, buffers, data):
            return functional_call(net, (weights, buffers), (data,))

        # 返回 net_func 函数、网络的参数字典和缓冲区字典
        return net_func, dict(net.named_parameters()), dict(net.named_buffers())


# 使用 markDynamoStrictTest 标记的测试类 TestGradTransform，继承自 TestCase
@markDynamoStrictTest
class TestGradTransform(TestCase):
    
    # 测试计算 torch.sin 梯度的基本函数
    def test_primitive(self, device):
        x = torch.randn([], device=device)
        result = grad(torch.sin)(x)
        self.assertEqual(result, torch.cos(x))

    # 测试计算 lambda 函数梯度的简单复合函数
    def test_composite_simple(self, device):
        x = torch.randn(2, 3, 4, device=device)
        result = grad(lambda x: torch.flatten(x).sum())(x)
        self.assertEqual(result, torch.ones_like(x))

    # 测试带有关键字参数的函数梯度
    def test_fn_with_kwargs(self, device):
        def foo(x, y):
            return (x * y).sum()

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        expected = grad(foo)(x, y)
        result = grad(foo)(x, y=y)
        self.assertEqual(result, expected)

    # 测试复合复杂函数的梯度
    def test_composite_complicated(self, device):
        x = torch.randn(3, device=device)
        y = torch.randn(3, 5, device=device)

        def foo(x, y):
            result = x @ y
            return result.sum()

        result = grad(foo)(x, y)

        x.requires_grad_()
        out = foo(x, y)
        (expected,) = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    # 测试两个操作的复合函数的梯度
    def test_composite_two_ops(self, device):
        N, C = 2, 5
        y = torch.randn(N, C, device=device)
        targets = torch.randint(0, C, (N,), device=device)

        def foo(y, targets):
            return F.cross_entropy(y, targets)

        result = grad(foo)(y, targets)

        y.requires_grad_()
        (expected,) = torch.autograd.grad(foo(y, targets), y)

        self.assertEqual(result, expected)

    # 测试用于获取属性的内部函数
    def _test_attributes(self, get_attr_lambda, device):
        x = torch.randn(2, 3, 5, dtype=torch.double, device=device)
        expected = get_attr_lambda(x)

        def foo(x):
            self.assertEqual(get_attr_lambda(x), expected)
            return x.sum()

        grad(foo)(x)

    # 测试张量形状的属性
    def test_shape(self, device):
        self._test_attributes(lambda x: x.shape, device)

    # 测试张量数据类型的属性
    def test_dtype(self, device):
        self._test_attributes(lambda x: x.dtype, device)

    # 测试张量是否在 CUDA 上的属性
    def test_is_cuda(self, device):
        self._test_attributes(lambda x: x.is_cuda, device)

    # 测试张量元素数量的属性
    def test_numel(self, device):
        self._test_attributes(lambda x: x.numel(), device)

    # 测试原地操作的函数
    def test_inplace(self, device):
        x = torch.randn([], device=device)

        def foo(x):
            return x.clone().sin_()

        result = grad(foo)(x)
        self.assertEqual(result, x.cos())
    # 在视图上进行原地操作的测试函数，传入设备参数
    def test_inplace_on_view(self, device):
        # 创建一个形状为 (3,) 的张量 x，使用设备参数指定设备
        x = torch.randn(3, device=device)

        # 定义内部函数 foo，接受一个参数 x
        def foo(x):
            # 克隆张量 x，并赋值给 y
            y = x.clone()
            # 取 y 的第一个元素 y[0]，对其应用 sin_() 原地操作
            y0 = y[0]
            y0.sin_()
            # 返回 y 所有元素的和
            return y.sum()

        # 计算 foo 函数相对于 x 的梯度
        result = grad(foo)(x)

        # 将 x 标记为需要计算梯度
        x.requires_grad_()
        # 计算 foo 函数的输出
        out = foo(x)
        # 计算 foo 函数相对于 x 的梯度
        (expected,) = torch.autograd.grad(out, x)

        # 断言计算得到的梯度与预期的梯度相等
        self.assertEqual(result, expected)

    # 在视图上进行原地操作的基础测试函数，传入设备参数
    def test_inplace_on_view_base(self, device):
        # 创建一个形状为 (3,) 的张量 x，使用设备参数指定设备
        x = torch.randn(3, device=device)

        # 定义内部函数 foo，接受一个参数 x
        def foo(x):
            # 克隆张量 x，并赋值给 y
            y = x.clone()
            # 取 y 的第一个元素 y[0]，对 y 应用 sin_() 原地操作
            y0 = y[0]
            y.sin_()
            # 返回 y 的第一个元素
            return y0

        # 计算 foo 函数相对于 x 的梯度
        result = grad(foo)(x)

        # 将 x 标记为需要计算梯度
        x.requires_grad_()
        # 计算 foo 函数的输出
        out = foo(x)
        # 计算 foo 函数相对于 x 的梯度
        (expected,) = torch.autograd.grad(out, x)

        # 断言计算得到的梯度与预期的梯度相等
        self.assertEqual(result, expected)

    # 在捕获的张量上进行原地操作的测试函数，传入设备参数
    def test_inplace_on_captures(self, device):
        # 创建一个形状为 (3,) 的张量 x，使用设备参数指定设备
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        # 创建一个形状为 (3,) 的张量 captured，使用设备参数指定设备
        captured = torch.randn(3, device=device)

        # 定义内部函数 foo，接受一个参数 x
        def foo(x):
            # 将 x 的值复制到 captured 张量中
            captured.copy_(x)
            # 返回 x 与 captured 的元素对应相乘后的和
            return (x * captured).sum()

        # 断言调用 grad(foo)(x) 时会抛出 RuntimeError 异常，提示“mutate a captured Tensor”
        with self.assertRaisesRegex(RuntimeError, "mutate a captured Tensor"):
            grad(foo)(x)

    # 简单嵌套的测试函数，传入设备参数
    def test_nesting_simple(self, device):
        # 创建一个标量张量 x，使用设备参数指定设备
        x = torch.randn([], device=device)
        # 计算两次 sin 函数的梯度
        result = grad(grad(torch.sin))(x)
        # 断言计算得到的结果与 -sin(x) 相等
        self.assertEqual(result, -torch.sin(x))

    # 测试逃逸的包装器被标记为无效的测试函数，传入设备参数
    @skipIfTorchDynamo("Ref: https://github.com/pytorch/pytorch/issues/103613")
    def test_escaped_wrappers_are_marked_as_dead(self, device):
        # 创建一个标量张量 x，使用设备参数指定设备
        x = torch.randn([], device=device)
        # 创建一个空列表 escaped
        escaped = []

        # 定义内部函数 foo，接受一个参数 x
        def foo(x):
            # 计算 x 的 sin 函数值，并将结果添加到 escaped 列表中
            y = x.sin()
            escaped.append(y)
            return y

        # 计算 foo 函数相对于 x 的梯度
        grad(foo)(x)
        # 断言 escaped 列表中第一个元素的梯度级别为 -1
        self.assertEqual(torch._C._functorch.dlevel(escaped[0]), -1)

    # 忽略逃逸的包装器的测试函数，传入设备参数
    @skipIfTorchDynamo("Ref: https://github.com/pytorch/pytorch/issues/103613")
    def test_escaped_wrappers_are_ignored(self, device):
        # 创建一个标量张量 x，使用设备参数指定设备
        x = torch.randn([], device=device)
        # 创建一个空列表 escaped
        escaped = []

        # 定义内部函数 foo，接受一个参数 x
        def foo(x):
            # 计算 x 的 sin 函数值，并将结果添加到 escaped 列表中
            y = x.sin()
            escaped.append(y)
            return y

        # 计算 foo 函数相对于 x 的梯度
        grad(foo)(x)

        # 对 escaped 列表中的第一个元素求和
        something = escaped[0].sum()
        # 断言 something 的梯度级别为 0
        self.assertEqual(torch._C._functorch.dlevel(something), 0)
        # 断言 something 的值与 x.sin().sum() 相等
        self.assertEqual(something, x.sin().sum())

    # 在梯度计算内部手动设置随机种子的测试函数，传入设备参数
    def test_manual_seed_inside_grad(self, device):
        # 创建一个标量张量 x，使用设备参数指定设备
        x = torch.randn([], device=device)

        # 定义内部函数 f，接受一个参数 x
        def f(x):
            # 设置随机种子为 0
            torch.manual_seed(0)
            # 返回 x 与形状与 x 相同的随机张量元素对应相乘的结果
            return x * torch.randn_like(x)

        # 冻结随机数生成器状态
        with freeze_rng_state():
            # 计算 f 函数相对于 x 的梯度
            result = grad(f)(x)
            # 将 x 标记为需要计算梯度
            x.requires_grad_()
            # 计算 f 函数相对于 x 的梯度
            (expected,) = torch.autograd.grad(f(x), x)
            # 断言计算得到的梯度与预期的梯度相等
            self.assertEqual(result, expected)

    # 测试 VJP（向量雅可比乘积）的函数，传入设备参数
    def test_vjp(self, device):
        # 创建一个标量张量 x，使用设备参数指定设备
        x = torch.randn([], device=device)
        # 计算 sin(x) 的 VJP，并返回结果及其对应的 VJP 函数
        out, vjp_fn = vjp(torch.sin, x)
        # 断言计算得到的结果与 sin(x) 相等
        self.assertEqual(out, x.sin())

        # 创建一个与 x 形状相同的标量张量 v
        v = torch.randn([], device=device)
        # 计算 VJP 函数对于 v 的结果
        (result,) = vjp_fn(v)
        # 断言计算得到的结果与 v * cos(x) 相等
        self.assertEqual(result, v * x.cos())

    # 测试返回两个输出的 VJP 函数，传入设备参数
    def test_vjp_two_outputs(self, device):
        # 定义函数 f，接受一个参数 x
        def f(x):
            # 返回 x 本身及其本身
            return x, x

        # 计算函数 f 在 x = 1.0 处的
    def test_conj_bit(self):
        # 创建一个复数张量，1 + 1j
        x = torch.tensor(1 + 1j)

        def foo(x):
            # 断言张量 x 不是共轭的
            assert not x.is_conj()
            # 计算 x 的共轭
            y = x.conj()
            # 断言 y 是共轭的
            assert y.is_conj()
            # 返回 y 的绝对值
            return y.abs()

        # 计算 foo 函数关于 x 的梯度
        res = grad(foo)(x)
        # 使用 torch.no_grad() 上下文管理器，确保不进行梯度计算
        with torch.no_grad():
            # 断言 res 等于一个与 x 的符号函数值相同大小的张量
            self.assertEqual(res, torch.ones_like(res) * torch.sgn(x))

    def test_composed_with_autograd(self, device):
        # 创建一个在指定设备上的随机张量，需要梯度信息
        x = torch.randn([], requires_grad=True, device=device)

        # 计算 torch.sin(x) 关于 x 的梯度
        y = grad(torch.sin)(x)
        # 计算 y 关于 x 的梯度
        (result,) = torch.autograd.grad(y, x)
        # 断言结果等于 -x.sin()
        self.assertEqual(result, -x.sin())

    def test_grad_of_vjp_composition(self, device):
        # 创建两个在指定设备上的随机张量
        x = torch.randn([], device=device)
        y = torch.randn([], device=device)

        def foo(x, y):
            # 计算 torch.sin(x) 的值和其对应的 vjp 函数
            out, vjp_fn = vjp(torch.sin, x)
            # 计算 vjp_fn(y)[0] 的梯度
            return grad(lambda y: vjp_fn(y)[0])(y)

        # 调用 foo 函数，计算结果
        result = foo(x, y)
        # 预期结果为 x.cos()
        expected = x.cos()
        # 断言结果与预期相等
        self.assertEqual(result, expected)

    def test_vjp_of_grad_composition(self, device):
        # 创建两个在指定设备上的随机张量
        x = torch.randn([], device=device)
        y = torch.randn([], device=device)

        def foo(x, y):
            # 计算 grad(torch.sin)(x) 的值和其对应的 vjp 函数
            out, vjp_fn = vjp(grad(torch.sin), x)
            # 计算 vjp_fn(y)[0] 的值
            return vjp_fn(y)[0]

        # 调用 foo 函数，计算结果
        result = foo(x, y)
        # 预期结果为 -y * x.sin()
        expected = -y * x.sin()
        # 断言结果与预期相等
        self.assertEqual(result, expected)

    def test_grad_of_vjp_of_grad_composition(self, device):
        # 创建两个在指定设备上的随机张量
        x = torch.randn([], device=device)
        y = torch.randn([], device=device)

        def foo(x, y):
            # 计算 grad(lambda x: -torch.cos(x)) 的值和其对应的 vjp 函数
            df, vjp_fn = vjp(grad(lambda x: -torch.cos(x)), x)
            # 计算 vjp_fn(y)[0] 的梯度
            return grad(lambda y: vjp_fn(y)[0])(y)

        # 调用 foo 函数，计算结果
        result = foo(x, y)
        # 预期结果为 x.cos()
        expected = x.cos()
        # 断言结果与预期相等
        self.assertEqual(result, expected)

    def test_views(self, device):
        # 创建两个在指定设备上需要梯度信息的随机张量
        x = torch.randn([], requires_grad=True, device=device)
        y = torch.randn([], requires_grad=True, device=device)

        def silly_sin(x):
            # 将 x 视图为标量，并计算其正弦
            x = x.view([])
            x = x.sin()
            return x

        def foo(x, y):
            # 计算 silly_sin(x) 关于 x 的梯度
            z1 = grad(silly_sin)(x)
            # 计算 torch.cos(y)
            z2 = torch.cos(y)
            # 返回 z1 + z2
            return z1 + z2

        # 调用 foo 函数，计算结果
        result = foo(x, y)
        # 计算 result 关于 x 和 y 的梯度
        grads = torch.autograd.grad(result, [x, y])
        # 断言关于 x 的梯度等于 -x.sin()
        self.assertEqual(grads[0], -x.sin())
        # 断言关于 y 的梯度等于 -y.sin()
        self.assertEqual(grads[1], -y.sin())

    def test_view_inplace_simple(self, device):
        def foo(x):
            # 克隆输入张量 x
            x = x.clone()
            # 将 x 视图为标量，并计算其正弦的原地操作
            x.view([]).sin_()
            return x

        # 创建一个在指定设备上需要梯度信息的随机张量
        x = torch.randn([], requires_grad=True, device=device)
        # 计算 foo 函数关于 x 的梯度
        result = grad(foo)(x)
        # 断言结果等于 x 的余弦值
        self.assertEqual(result, x.cos())
    # 测试在给定的设备上使用无效的 argnums 参数来调用 grad 函数，预期引发 RuntimeError 异常
    def test_invalid_argnums(self, device):
        # 创建随机张量 x 和 y
        x = torch.randn([])
        y = torch.randn([])
        # 测试使用 argnums=-3 调用 grad 函数，预期引发异常并包含指定错误信息
        with self.assertRaisesRegex(RuntimeError, "but only"):
            grad(torch.mul, argnums=-3)(x, y)
        # 测试使用 argnums=2 调用 grad 函数，预期引发异常并包含指定错误信息
        with self.assertRaisesRegex(RuntimeError, "but only"):
            grad(torch.mul, argnums=2)(x, y)
        # 测试使用 argnums=[0] 调用 grad 函数，预期引发异常并包含指定错误信息
        with self.assertRaisesRegex(RuntimeError, "int or Tuple"):
            grad(torch.mul, argnums=[0])(x, y)
        # 测试使用 argnums=("0",) 调用 grad 函数，预期引发异常并包含指定错误信息
        with self.assertRaisesRegex(RuntimeError, "must be int"):
            grad(torch.mul, argnums=("0",))(x, y)
        # 测试使用 argnums=(0, 0) 调用 grad 函数，预期引发异常并包含指定错误信息
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            grad(torch.mul, argnums=(0, 0))(x, y)
        # 测试使用 argnums=(0, -2) 调用 grad 函数，预期引发异常并包含指定错误信息
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            grad(torch.mul, argnums=(0, -2))(x, y)

    # 测试在给定的设备上使用有效的 argnums 参数来调用 grad 函数，验证梯度计算的正确性
    def test_argnums(self, device):
        # 创建随机张量 x 和 y
        x = torch.randn([])
        y = torch.randn([])
        # 计算 torch.mul 函数关于第一个参数 x 的梯度，预期值为 y
        gx = grad(torch.mul, argnums=0)(x, y)
        self.assertEqual(gx, y)
        # 计算 torch.mul 函数关于第二个参数 y 的梯度，预期值为 x
        gy = grad(torch.mul, argnums=1)(x, y)
        self.assertEqual(gy, x)
        # 使用 argnums=(0,) 计算 torch.mul 函数关于第一个参数 x 的梯度，预期值为 y
        (gx,) = grad(torch.mul, argnums=(0,))(x, y)
        self.assertEqual(gx, y)
        # 使用 argnums=(0, 1) 同时计算 torch.mul 函数关于参数 x 和 y 的梯度，预期 gx=y, gy=x
        gx, gy = grad(torch.mul, argnums=(0, 1))(x, y)
        self.assertEqual(gx, y)
        self.assertEqual(gy, x)

    # 测试在给定的设备上使用有效的 argnums 参数来调用 grad 函数，验证参数顺序对梯度计算的影响
    def test_out_of_order_argnums(self, device):
        # 创建随机张量 x 和 y
        x = torch.randn([])
        y = torch.randn([])
        # 使用 argnums=(1, 0) 计算 torch.mul 函数关于参数 y 和 x 的梯度，预期 gy=x, gx=y
        gy, gx = grad(torch.mul, argnums=(1, 0))(x, y)
        self.assertEqual(gx, y)
        self.assertEqual(gy, x)

    # 测试在给定的设备上使用负数的 argnums 参数来调用 grad 函数，验证梯度计算的正确性
    def test_negative_argnums(self, device):
        # 创建随机张量 x 和 y
        x = torch.randn([])
        y = torch.randn([])
        # 计算 torch.mul 函数关于倒数第二个参数的梯度，预期值为 y
        gx = grad(torch.mul, argnums=-2)(x, y)
        self.assertEqual(gx, y)
        # 计算 torch.mul 函数关于最后一个参数的梯度，预期值为 x
        gy = grad(torch.mul, argnums=-1)(x, y)
        self.assertEqual(gy, x)
        # 使用 argnums=(-2,) 计算 torch.mul 函数关于倒数第二个参数的梯度，预期值为 y
        (gx,) = grad(torch.mul, argnums=(-2,))(x, y)
        self.assertEqual(gx, y)
        # 使用 argnums=(-2, -1) 同时计算 torch.mul 函数关于倒数第二个和最后一个参数的梯度，预期 gx=y, gy=x
        gx, gy = grad(torch.mul, argnums=(-2, -1))(x, y)
        self.assertEqual(gx, y)
        self.assertEqual(gy, x)

    # 测试在给定的设备上使用复杂数据结构作为输入参数，验证 grad 函数对于复杂输入的梯度计算能力
    def test_grad_pytree_inputs(self, device):
        # 创建在指定设备上的随机张量 x
        x = torch.randn([], device=device)

        # 定义一个接受复杂结构参数的函数 f
        def f(a, b):
            x, y = a
            return 1 * x + 2 * y + 3 * b["foo"]

        # 准备复杂结构的输入参数 args
        args = ((x, x), {"foo": x})

        # 计算函数 f 对第一个参数 x 的梯度 gx 和对第二个参数 y 的梯度 gy
        gx, gy = grad(f)(*args)
        self.assertEqual(gx, torch.tensor(1.0, device=device))
        self.assertEqual(gy, torch.tensor(2.0, device=device))

        # 使用 argnums=(0,) 计算函数 f 对第一个参数 x 的梯度 gx 和对第二个参数 y 的梯度 gy
        ((gx, gy),) = grad(f, argnums=(0,))(*args)
        self.assertEqual(gx, torch.tensor(1.0, device=device))
        self.assertEqual(gy, torch.tensor(2.0, device=device))

        # 使用 argnums=(0, 1) 同时计算函数 f 对第一个参数 x、第二个参数 y 的梯度 gx, gy 和对参数 b["foo"] 的梯度 gz
        (gx, gy), gz = grad(f, argnums=(0, 1))(*args)
        self.assertEqual(gx, torch.tensor(1.0, device=device))
        self.assertEqual(gy, torch.tensor(2.0, device=device))
        self.assertEqual(gz["foo"], torch.tensor(3.0, device=device))
    # 测试梯度辅助张量函数，使用给定的设备
    def test_grad_aux_tensor(self, device):
        # 创建一个在指定设备上的形状为 (3,) 的随机张量 x
        x = torch.randn(3, device=device)

        # 断言以下代码块会引发 RuntimeError 异常，且异常信息包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError,
            r"grad_and_value\(f\)\(\*args\): output of function f should be a tuple",
        ):
            # 调用 grad 函数，传入 lambda 函数和 has_aux=True，用于计算梯度
            grad(lambda t: [t, t], has_aux=True)(x)

        # 同上，验证另一个 lambda 函数输出不是 tuple 的情况
        with self.assertRaisesRegex(
            RuntimeError,
            r"grad_and_value\(f\)\(\*args\): output of function f should be a tuple",
        ):
            grad(lambda t: (t, t + 2, t + 3), has_aux=True)(x)

        # 定义一个函数 f，接受参数 t，计算其正弦的和与余弦值
        def f(t):
            y = t.sin()  # 计算 t 的正弦
            return y.sum(), t.cos()  # 返回正弦的和与 t 的余弦值

        # 使用 grad 函数计算函数 f 在输入 x 上的梯度及辅助信息
        out, aux = grad(f, has_aux=True)(x)
        # 断言辅助信息 aux 等于 x 的余弦值
        self.assertEqual(aux, x.cos())
        # 断言输出 out 等于 x 的余弦值
        self.assertEqual(out, x.cos())

    # 测试梯度辅助 pytree 的函数，使用给定的设备
    def test_grad_aux_pytree(self, device):
        # 定义一个函数 f，接受参数 x，计算其正弦的和与一个字典作为辅助信息
        def f(x):
            y = x.sin()  # 计算 x 的正弦
            return y.sum(), {"a": x.cos(), "b": [x.tan()]}  # 返回正弦的和与字典

        # 创建一个在指定设备上的形状为 (3,) 的随机张量 x
        x = torch.randn(3, device=device)

        # 使用 grad 函数计算函数 f 在输入 x 上的梯度及辅助信息
        out, aux = grad(f, has_aux=True)(x)
        # 获取函数 f 在输入 x 上预期的辅助信息
        _, expected_aux = f(x)
        # 断言计算得到的辅助信息 aux 等于预期的辅助信息 expected_aux
        self.assertEqual(aux, expected_aux)
        # 断言输出 out 等于 x 的余弦值
        self.assertEqual(out, x.cos())

        # 验证对于不支持的类型（非张量），会引发 RuntimeError 异常
        for aux in [1, 1.0, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                # 调用 grad 函数，传入 lambda 函数和 has_aux=True，其中包含不支持的类型
                _ = grad(lambda x: (x.sum(), aux), has_aux=True)(x)
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                # 同上，但是输出包含列表，其中包含不支持的类型
                _ = grad(lambda x: (x.sum(), [x, aux]), has_aux=True)(x)

    # 测试零梯度函数，使用给定的设备
    def test_zero_grad(self, device):
        # 定义一个函数 f，接受参数 x，计算 x["a"] 的平方和
        def f(x):
            return (x["a"] ** 2.0).sum()

        # 创建一个输入字典 inps，包含键为 "a" 和 "b" 的张量
        inps = {
            "a": torch.randn(10, device=device) + 3,  # 形状为 (10,) 的随机张量，加上常数 3
            "b": torch.randn(10, device=device),  # 形状为 (10,) 的随机张量
        }
        # 计算函数 f 在输入 inps 上的梯度
        grads = grad(f)(inps)
        # 断言 "a" 的梯度之和不等于 0.0
        self.assertNotEqual(grads["a"].sum(), 0.0)
        # 断言 "b" 的梯度之和等于 0.0
        self.assertEqual(grads["b"].sum(), 0.0)

    # 测试不相关梯度函数，使用给定的设备
    def test_unrelated_grad(self, device):
        # 创建一个值为 1.0 的张量 x，使用指定的设备
        x = torch.tensor(1.0, device=device)
        # 创建一个值为 2.0 的张量 y，使用指定的设备
        y = torch.tensor(2.0, device=device)

        # 定义一个不相关函数 unrelated，接受参数 x，返回常数 y
        def unrelated(x):
            return y

        # 使用 grad 函数计算函数 unrelated 在输入 x 上的梯度
        result = grad(unrelated)(x)
        # 断言计算得到的结果 result 等于形状与 x 相同且值全为 0 的张量
        self.assertEqual(result, torch.zeros_like(x))

    # 测试不相关 vjp 函数，使用给定的设备
    def test_unrelated_vjp(self, device):
        # 创建一个值为 1.0 的张量 x，使用指定的设备
        x = torch.tensor(1.0, device=device)
        # 创建一个值为 2.0 的张量 y，使用指定的设备
        y = torch.tensor(2.0, device=device)
        # 创建一个值为 1.0 的张量 v，使用指定的设备
        v = torch.tensor(1.0, device=device)

        # 定义一个不相关函数 unrelated，接受参数 x，返回常数 y
        def unrelated(x):
            return y

        # 使用 vjp 函数计算函数 unrelated 在输入 x 上的输出和反向传播函数
        out, vjp_fn = vjp(unrelated, x)
        # 调用反向传播函数 vjp_fn，传入张量 v，计算其结果
        result = vjp_fn(v)
        # 定义预期结果 expected 为一个包含形状与 x 相同且值全为 0 的元组
        expected = (torch.zeros_like(x),)
        # 断言计算得到的结果 result 等于预期结果 expected
        self.assertEqual(result, expected)
    # 测试不相关的向量雅可比乘积（VJP）函数，接受多个输入和输出
    def test_unrelated_vjp_multiple_inputs_outputs(self, device):
        # 创建张量 w, x, y, v，分别赋值为 3.0, 4.0, 2.0, 1.0，并指定设备
        w = torch.tensor(3.0, device=device)
        x = torch.tensor(4.0, device=device)
        y = torch.tensor(2.0, device=device)
        v = torch.tensor(1.0, device=device)

        # 定义不相关的函数 unrelated，接受 w, x 作为参数，返回 (y, y, x)
        def unrelated(w, x):
            return y, y, x

        # 调用 VJP 函数，计算 unrelated 函数在 (w, x) 处的输出和 VJP 函数
        out, vjp_fn = vjp(unrelated, w, x)
        # 调用 VJP 函数的结果，传入参数 (v, v, v)
        result = vjp_fn((v, v, v))
        # 期望输出是一个与 x 形状相同的全零张量和一个与 x 形状相同的全一张量
        expected = (torch.zeros_like(x), torch.ones_like(x))
        # 断言结果与期望相等
        self.assertEqual(result, expected)

    # TODO: https://github.com/zou3519/functorch/issues/12
    # 测试不相关的海森矩阵函数，仅在 CPU 上运行
    @onlyCPU
    def test_unrelated_hessian(self, device):
        # 创建形状为 N x M 的随机张量 W，并指定设备
        N = 5
        M = 3
        W = torch.randn(N, M, device=device)

        # 定义函数 f(x)，返回 W @ x
        def f(x):
            return W @ x

        # 创建形状为 M 的随机张量 x
        x = torch.randn(M)
        # 计算函数 f 的二阶雅可比矩阵（Jacobian matrix of Jacobian matrix）
        result = jacrev(jacrev(f))(x)
        # 期望输出是形状为 N x M x M 的零张量
        expected = torch.zeros(N, M, M, device=device)
        # 断言结果与期望相等
        self.assertEqual(result, expected)

    # 测试 VJP 函数接受 Pytree 输入
    def test_vjp_pytree_input(self, device):
        # 定义函数 f(x)，返回 x[0] * x[1][0]
        def f(x):
            return x[0] * x[1][0]

        # 创建空的随机张量 x 和 v，并指定设备
        x = torch.randn([], device=device)
        v = torch.randn([], device=device)
        # 调用 VJP 函数，计算 f 在 (x, (x, x)) 处的输出和 VJP 函数
        out, vjp_fn = vjp(f, (x, (x, x)))
        # 断言输出结果等于 x * x
        self.assertEqual(out, x * x)
        # 调用 VJP 函数的结果，传入参数 v
        result = vjp_fn(v)
        # 断言结果与期望相等
        self.assertEqual(result, ((x * v, (x * v, 0.0)),))

    # 测试 VJP 函数返回 Pytree 输出
    def test_vjp_pytree_output(self, device):
        # 定义函数 f(x)，返回 (x, (x, x))
        def f(x):
            return x, (x, x)

        # 创建空的随机张量 x 和 v1, v2, v3，并指定设备
        x = torch.randn([], device=device)
        v1 = torch.randn([], device=device)
        v2 = torch.randn([], device=device)
        v3 = torch.randn([], device=device)
        # 调用 VJP 函数，计算 f 在 x 处的输出和 VJP 函数
        _, vjp_fn = vjp(f, x)
        # 调用 VJP 函数的结果，传入参数 (v1, (v2, v3))
        (result,) = vjp_fn((v1, (v2, v3)))
        # 断言结果与 v1 + v2 + v3 相等
        self.assertEqual(result, v1 + v2 + v3)
    # 测试函数，验证对任意 PyTree 结构的 vjp 输出
    def test_vjp_outputs_can_any_pytree(self, device):
        # 创建随机张量 x 和 t，指定设备
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        # 对于空输出和输出为元组的情况，验证是否引发 RuntimeError 异常
        for output in [None, ()]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"vjp\(f, \*primals\): Expected f to be a function that has non-empty output",
            ):
                # 调用 vjp 函数，使用 lambda 函数作为输入，捕获 vjp 函数
                _, vjp_fn = vjp(lambda _: output, x)
                # 调用 vjp_fn 函数，传入 t 张量作为参数
                vjp_fn(t)

        # 对于输出为非张量的情况（整数、布尔值、浮点数、字符串），验证是否引发 RuntimeError 异常
        for output in [1, True, 12.2, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"vjp\(f, \*primals\): expected f\(\*primals\) to return only tensors",
            ):
                # 调用 vjp 函数，使用 lambda 函数作为输入，捕获 vjp 函数
                _, vjp_fn = vjp(lambda _: output, x)
                # 调用 vjp_fn 函数，传入 t 张量作为参数
                vjp_fn(t)

        # 验证列表输出的情况
        output, vjp_fn = vjp(lambda x: [x, x.sum()], x)
        # 调用 vjp_fn 函数，传入列表 [t, t.sum()] 作为参数
        (vjp_out,) = vjp_fn([t, t.sum()])
        assert isinstance(output, list) and len(output) == 2
        assert isinstance(vjp_out, torch.Tensor)

        # 验证字典输出的情况
        output, vjp_fn = vjp(lambda x: {"x": x, "xsum": x.sum()}, x)
        # 调用 vjp_fn 函数，传入字典 {"x": t, "xsum": t.sum()} 作为参数
        (vjp_out,) = vjp_fn({"x": t, "xsum": t.sum()})
        assert isinstance(output, dict) and len(output) == 2 and "xsum" in output
        assert isinstance(vjp_out, torch.Tensor)

        # 验证复合输出的情况
        def composite_output(x):
            out = x.sum()
            return [
                (out, {"a": x, "out": [x, out]}),
            ]

        output, vjp_fn = vjp(composite_output, x)
        # 调用 vjp_fn 函数，传入 [(t.sum(), {"a": t, "out": [t, t.sum()]}),] 作为参数
        (vjp_out,) = vjp_fn(
            [
                (t.sum(), {"a": t, "out": [t, t.sum()]}),
            ]
        )
        assert isinstance(output, list)
        assert isinstance(output[0], tuple) and isinstance(output[0][1], dict)
        assert isinstance(vjp_out, torch.Tensor)

    # 测试函数，验证 vjp 函数在不符合 PyTree 结构时是否引发异常
    def test_vjp_pytree_error(self, device):
        # 定义函数 f，返回输入 x 和元组 (x, x)
        def f(x):
            return x, (x, x)

        # 创建随机张量 x 和 v1, v2, v3，指定设备
        x = torch.randn([], device=device)
        v1 = torch.randn([], device=device)
        v2 = torch.randn([], device=device)
        v3 = torch.randn([], device=device)
        # 调用 vjp 函数，捕获 vjp_fn 函数
        _, vjp_fn = vjp(f, x)
        # 验证传入的参数 ((v1, (v2, v3)),) 是否引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Expected pytree structure"):
            (result,) = vjp_fn(((v1, (v2, v3)),))

    # 测试函数，验证 vjp 函数在具有辅助张量时的行为
    def test_vjp_aux_tensor(self, device):
        # 创建随机张量 x，指定设备
        x = torch.randn(3, device=device)

        # 验证当函数 f 输出为列表时是否引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError, r"vjp\(f, \*primals\): output of function f should be a tuple"
        ):
            vjp(lambda t: [t, t], x, has_aux=True)

        # 验证当函数 f 输出为元组但长度超过两个时是否引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError, r"vjp\(f, \*primals\): output of function f should be a tuple"
        ):
            vjp(lambda t: (t, t + 2, t + 3), x, has_aux=True)

        # 定义函数 f，返回输入张量 t 的 sin 值和 cos 值
        def f(t):
            y = t.sin()
            return y, t.cos()

        # 调用 vjp 函数，捕获返回值 out, vjp_fn 和 aux
        out, vjp_fn, aux = vjp(f, x, has_aux=True)
        # 验证 aux 是否等于 x 的 cos 值
        self.assertEqual(aux, x.cos())
        # 验证 out 是否等于 x 的 sin 值
        self.assertEqual(out, x.sin())

        # 创建随机张量 v，指定设备
        v = torch.randn(3, device=device)
        # 调用 vjp_fn 函数，传入 v 作为参数，捕获返回值 grad_x
        (grad_x,) = vjp_fn(v)
        # 验证 grad_x 是否等于 v 乘以 x 的 cos 值
        self.assertEqual(grad_x, v * x.cos())
    # 定义一个测试函数，测试带有辅助信息的反向传播
    def test_vjp_aux_pytree(self, device):
        # 定义一个函数 f，对输入 x 执行 sin() 操作，返回结果 y 和辅助信息字典
        def f(x):
            y = x.sin()
            return y, {"a": x.cos(), "b": [x.tan()]}

        # 生成一个在指定设备上的随机张量 x
        x = torch.randn(3, device=device)

        # 调用 vjp 函数计算 f 在 x 处的结果 out，并返回对应的 vjp 函数和辅助信息 aux
        out, vjp_fn, aux = vjp(f, x, has_aux=True)

        # 计算函数 f 在 x 处的预期输出和辅助信息
        expected_out, expected_aux = f(x)

        # 使用断言验证计算结果是否与预期一致
        self.assertEqual(out, expected_out)
        self.assertEqual(aux, expected_aux)

        # 生成一个在指定设备上的随机张量 v
        v = torch.randn(3, device=device)

        # 调用 vjp_fn 计算梯度 grad_x
        (grad_x,) = vjp_fn(v)

        # 使用断言验证计算得到的梯度是否与预期一致
        self.assertEqual(grad_x, v * x.cos())

        # 遍历不同类型的辅助信息 aux，使用断言验证 vjp 函数在处理不支持类型时是否能够引发 RuntimeError 异常
        for aux in [1, 1.0, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = vjp(lambda x: (x, aux), x, has_aux=True)
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                _ = vjp(lambda x: (x, [x, aux]), x, has_aux=True)

    # 定义一个测试函数，测试初始化函数的功能
    def test_functional_init(self, device):
        # 定义一个简单的 MLP 分类器模型
        class MLPClassifier(nn.Module):
            def __init__(self, hidden_dim=32, n_classes=2):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes

                self.fc1 = nn.Linear(2, self.hidden_dim)
                self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                x = F.log_softmax(x, -1)
                return x

        # 设置批次大小 B
        B = 10

        # 使用 functional_init 初始化 MLPClassifier 模型，获取权重、初始化函数和其他信息
        weights, fn, _ = functional_init(MLPClassifier, (B,), device=device)(32, 2)

        # 生成一个在指定设备上的随机输入张量 inputs
        inputs = torch.randn(B, 7, 2, device=device)

        # 使用 vmap 执行 fn 函数映射，对权重和输入进行操作
        vmap(fn)(weights, (inputs,))

    # 定义一个测试函数，测试带缓冲区的初始化函数的功能
    def test_functional_init_with_buffers(self, device):
        # 定义一个带有 BatchNorm 层的 MLP 分类器模型
        class MLPClassifier(nn.Module):
            def __init__(self, hidden_dim=32, n_classes=2):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes

                self.fc1 = nn.Linear(2, self.hidden_dim)
                self.bn = nn.BatchNorm1d(self.hidden_dim, affine=True)
                self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.bn(x)
                x = self.fc2(x)
                x = F.log_softmax(x, -1)
                return x

        # 设置批次大小 B
        B = 10

        # 使用 functional_init_with_buffers 初始化 MLPClassifier 模型，获取权重、缓冲区、初始化函数和其他信息
        weights, buffers, fn, _, _ = functional_init_with_buffers(
            MLPClassifier, [B], device=device
        )(32, 2)

        # 生成一个在指定设备上的随机输入张量 inputs
        inputs = torch.randn(B, 7, 2, device=device)

        # 使用 vmap 执行 fn 函数映射，对权重、缓冲区和输入进行操作
        vmap(fn)(weights, buffers, (inputs,))
    # 定义一个测试函数，用于测试在指定设备上的高级索引功能
    def test_advanced_indexing(self, device):
        
        # 定义函数 f，接受一个参数 value
        def f(value):
            # 初始化一个在指定设备上的空的对数概率张量
            log_prob = torch.ones((), device=device)
            # 创建一个布尔张量 val，其值为 False
            val = torch.zeros(()) > 0
            # 如果 val 为 True，则将对数概率张量对应位置设为 0
            log_prob[val] = 0
            return value
        
        # 计算函数 f 在随机数值处的梯度
        result = grad(f)(torch.randn((), device=device))
        # 断言计算出的梯度与全 1 张量形状相同
        self.assertEqual(result, torch.ones_like(result))
        
        # 定义函数 f2，接受一个参数 value
        def f2(value):
            # 克隆输入值，确保不会修改原始值
            value = value.clone()
            # 将大于 0 的值设为 0
            value[value > 0] = 0
            # 返回修改后的值的和
            return value.sum()
        
        # 生成一个包含随机数的张量 x
        x = torch.randn(100, device=device)
        # 计算函数 f2 在张量 x 上的梯度
        result = grad(f2)(x)
        # 断言计算出的梯度与小于等于 0 的张量形状相同
        self.assertEqual(result, (x <= 0).type_as(x))

    # 定义测试函数，用于测试在梯度计算中在张量构造函数内部的行为
    def test_tensor_ctor_inside_grad(self, device):
        # 定义函数 foo，接受一个参数 x
        def foo(x):
            # 返回输入张量 x 乘以一个在指定设备上的常数张量 2.0
            return x * torch.tensor(2.0, device=device)
        
        # 创建一个包含随机数的张量 x
        x = torch.tensor(3.14, device=device)
        # 计算函数 foo 在张量 x 上的梯度
        functorch.grad(foo)(x)

    # 参数化装饰器，用于多次运行相同的测试函数，每次使用不同的输入数据
    @parametrize(
        "op_list_data",
        [
            # 子测试 1：使用 vmap 函数，输入为两个不同形状的元组
            subtest(
                (
                    [
                        vmap,
                    ],
                    [(4, 2), (64, 3, 32, 32)],
                ),
                name="vmap",
            ),
            # 子测试 2：连续两次使用 vmap 函数，输入为不同形状的元组
            subtest(([vmap, vmap], [(4, 3, 2), (64, 3, 32, 32)]), name="vmap_vmap"),
            # 子测试 3：使用 grad 函数，输入为不同长度的元组和空元组
            subtest(
                (
                    [
                        grad,
                    ],
                    [(0,), [], (4, 2), (64, 3, 32, 32)],
                ),
                name="grad",
            ),
            # 子测试 4：连续两次使用 grad 函数，输入为空元组
            subtest(
                (
                    [grad, grad],
                    [
                        [],
                    ],
                ),
                name="grad_grad",
            ),
            # 子测试 5：先使用 vmap 函数再使用 grad 函数，输入为一个元组
            subtest(([vmap, grad], [(4, 2)]), name="vmap_grad"),
        ],
    )
    # 测试函数：测试在给定设备和操作列表数据下的张量打印功能
    def test_tensor_print(self, device, op_list_data):
        # 解包操作列表和形状数据
        op_list, shapes = op_list_data

        # 针对所有浮点数数据类型进行迭代
        for dt in get_all_fp_dtypes():
            # 创建具有指定数据类型和形状的数据张量列表
            data = [torch.randn(s, dtype=dt, device=device) for s in shapes]

            # 遍历每一个数据张量
            for x in data:
                buf = None

                # 定义一个内部函数 foo，用于获取张量的字符串表示并返回其均值
                def foo(t):
                    nonlocal buf
                    buf = repr(t)  # 将张量 t 的字符串表示存储在 buf 中
                    return t.mean()

                fn = foo
                bdim = 0
                # 反向遍历操作列表
                for op in reversed(op_list):
                    if op == vmap:
                        fn = op(fn, in_dims=bdim)  # 应用 vmap 操作，并更新批处理维度
                        bdim += 1
                    else:
                        fn = op(fn)  # 应用当前操作到函数 fn 上

                # 预期结果为当前数据张量 x 的字符串表示
                expected = f"{repr(x)}"
                level = 0
                # 再次遍历操作列表
                for op in op_list:
                    level += 1
                    if op == grad:
                        expected = f"GradTrackingTensor(lvl={level}, value={expected})"
                    elif op == vmap:
                        bdim -= 1
                        expected = (
                            f"BatchedTensor(lvl={level}, bdim={bdim}, value={expected})"
                        )

                # 调用 fn 函数，将 x 作为参数传入
                fn(x)
                # 清理 buf 字符串中的换行和多余空格
                buf = buf.replace("\n", "").replace("  ", "")
                expected = expected.replace("\n", "").replace("  ", "")
                # 断言 buf 和 expected 字符串相等
                self.assertEqual(expected, buf)

    # 测试函数：测试在转换内部打印捕获的张量功能
    def test_print_captured_tensor_inside_transform(self, device):
        # 创建一个张量 x
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        out = None

        # 定义一个函数 f，用于捕获并存储张量 x 的字符串表示
        def f(y):
            nonlocal out
            out = repr(x)
            return y

        # 调用 vjp 函数，并将 f 函数和一个随机张量作为参数
        vjp(f, torch.randn(4, device=device))
        # 断言捕获的张量字符串表示与预期的 x 的字符串表示相等
        self.assertEqual(out, repr(x))

    # 测试函数：测试在无梯度上下文外部的操作
    def test_no_grad_outside(self, device):
        # 创建一个具有梯度的随机张量 x
        x = torch.randn([], device=device, requires_grad=True)
        # 在无梯度上下文中计算 torch.sin(x) 的梯度
        with torch.no_grad():
            y = grad(torch.sin)(x)
        # 断言 y 等于 x.cos()
        self.assertEqual(y, x.cos())
        # 断言 y 不需要梯度
        self.assertFalse(y.requires_grad)

    # 测试函数：测试在无梯度上下文内部的操作
    def test_no_grad_inside(self, device):
        # 定义一个函数 f，该函数在内部使用无梯度上下文计算张量的平方和差
        def f(x):
            with torch.no_grad():
                shift = x**2
            return x**2 - shift

        # 创建一个随机张量 x
        x = torch.randn([], device=device)
        # 计算函数 f 对 x 的梯度
        y = grad(f)(x)
        # 断言计算的梯度等于 2 * x
        self.assertEqual(y, 2 * x)
        # 计算 f 的二阶导数
        y = grad(grad(f))(x)
        # 断言二阶导数的值等于 2
        self.assertEqual(y, 2)

        # 创建一个具有梯度的随机张量 x
        x = torch.randn([], device=device, requires_grad=True)
        # 计算函数 f 对 x 的梯度
        y = grad(f)(x)
        # 使用 torch.autograd.grad 计算 y 对 x 的梯度
        (z,) = torch.autograd.grad(y, x)
        # 断言计算的梯度等于 2
        self.assertEqual(z, 2)

    # 测试函数：测试混合使用无梯度上下文的操作
    def test_no_grad_mixed(self, device):
        # 定义一个函数 f，该函数在混合的无梯度上下文中计算张量的平方和差
        def f(x):
            with torch.no_grad():
                shift = x**2
            return x**2 - shift

        # 创建一个具有梯度的随机张量 x
        x = torch.randn([], device=device, requires_grad=True)
        # 在无梯度上下文中计算函数 f 对 x 的梯度
        with torch.no_grad():
            y = grad(f)(x)

        # 断言计算的梯度等于 2 * x
        self.assertEqual(y, 2 * x)
        # 断言 y 不需要梯度
        self.assertFalse(y.requires_grad)
    # 定义一个测试函数，测试在嵌套情况下的使用 torch.no_grad()
    def test_no_grad_nested_simple(self, device):
        # 定义函数 h(x)，其中包含一个使用 torch.no_grad() 的上下文管理器
        def h(x):
            with torch.no_grad():
                # 在 torch.no_grad() 下计算梯度的偏移量
                shift = grad(lambda x: 0.25 * x**4)(x)
            # 返回 x^3 减去偏移量的结果
            return x**3 - shift

        # 创建一个张量 x，并且要求计算其梯度
        x = torch.tensor(1.5, device=device, requires_grad=True)
        # 计算 h(x) 的梯度
        y = grad(h)(x)
        # 断言计算的梯度与预期值相等
        self.assertEqual(y, 3 * x**2)

        # 计算 y 对 x 的梯度
        (z,) = torch.autograd.grad(y, x)
        # 断言计算的梯度与预期值相等
        self.assertEqual(z, 6 * x)

    # 定义一个测试函数，测试在复杂嵌套情况下的使用 torch.no_grad()
    def test_no_grad_nested_complicated(self, device):
        # 定义函数 f(x)，其中包含一个使用 torch.no_grad() 的上下文管理器
        def f(x):
            with torch.no_grad():
                # 计算 x^3 并赋值给 shift
                shift = x**3
            # 返回 x^3 减去 shift 的结果
            return x**3 - shift

        # 定义函数 g(x)，包含一个调用 f(x) 的梯度计算
        def g(x):
            # 计算 f(x) 的梯度
            r1 = grad(f)(x)
            with torch.no_grad():
                # 在 torch.no_grad() 下计算 f(x) 的梯度
                shift = grad(f)(x)
            # 返回 r1 减去 shift 的结果
            return r1 - shift

        # 创建一个随机张量 x，并且要求计算其梯度
        x = torch.randn([], requires_grad=True, device=device)
        # 计算 g(x) 的梯度
        y = grad(g)(x)
        # 断言计算的梯度与预期值相等，即 6 * x
        self.assertEqual(y, 6 * x)

        # 计算 y 对 x 的梯度
        (z,) = torch.autograd.grad(y, x)
        # 断言计算的梯度与预期值相等，即 6
        self.assertEqual(z, 6)

    # 定义一个测试函数，测试在 torch.no_grad() 中使用多个返回值
    def test_no_grad_value(self, device):
        # 定义函数 h(x)，其中包含一个使用 torch.no_grad() 的上下文管理器
        def h(x):
            with torch.no_grad():
                # 调用 grad_and_value 函数，获取梯度和值
                gvalue, value = grad_and_value(lambda x: x**3)(x)
            # 返回 x^3 减去 value 的结果
            return x**3 - value

        # 创建一个张量 x，并且要求计算其梯度
        x = torch.tensor(1.6, device=device, requires_grad=True)
        # 计算 h(x) 的梯度
        y = grad(h)(x)
        # 断言计算的梯度与预期值相等
        self.assertEqual(y, 3 * x**2)

        # 计算 y 对 x 的梯度
        (z,) = torch.autograd.grad(y, x)
        # 断言计算的梯度与预期值相等
        self.assertEqual(z, 6 * x)

    # 定义一个测试函数，测试在 torch.no_grad() 中使用 vjp 函数
    def test_no_grad_outside_vjp(self, device):
        # 定义函数 h(x)，直接返回 x^2
        def h(x):
            return x**2

        # 创建一个张量 x，并且要求计算其梯度
        x = torch.tensor(2.0, requires_grad=True, device=device)
        with torch.no_grad():
            # 调用 vjp 函数，获取输出和 vjp_fn
            out, vjp_fn = vjp(h, x)
            # 对 vjp_fn 进行调用，得到 y
            (y,) = vjp_fn(torch.tensor(1.0, device=device))

        # 断言 y 的值与预期相等，即 2 * x
        self.assertEqual(y, 2 * x)
        # 断言 y 不需要梯度
        self.assertFalse(y.requires_grad)
        # 断言 out 不需要梯度
        self.assertFalse(out.requires_grad)

    # 定义一个测试函数，测试在 torch.no_grad() 中使用 vjp 函数和梯度计算
    def test_no_grad_outside_vjp_fn(self, device):
        # 定义函数 h(x)，直接返回 x^2
        def h(x):
            return x**2

        # 创建一个张量 x，并且要求计算其梯度
        x = torch.tensor(3.14, requires_grad=True, device=device)
        # 调用 vjp 函数，获取输出和 vjp_fn
        out, vjp_fn = vjp(h, x)
        with torch.no_grad():
            # 对 vjp_fn 进行调用，得到 y
            (y,) = vjp_fn(torch.tensor(1.0, device=device))

        # 断言 y 的值与预期相等，即 2 * x
        self.assertEqual(y, 2 * x)
        # 断言 y 不需要梯度
        self.assertFalse(y.requires_grad)
        # 断言 out 需要梯度
        self.assertTrue(out.requires_grad)

        # 计算 out 对 x 的梯度
        (z,) = torch.autograd.grad(out, x)
        # 断言计算的梯度与预期值相等，即 2 * x
        self.assertEqual(z, 2 * x)

    # 定义一个测试函数，测试在 torch.no_grad() 中使用 vjp 函数和后续梯度计算
    def test_no_grad_outside_vjp_only(self, device):
        # 定义函数 h(x)，直接返回 x^2
        def h(x):
            return x**2

        # 创建一个张量 x，并且要求计算其梯度
        x = torch.tensor(3.14, requires_grad=True, device=device)
        with torch.no_grad():
            # 调用 vjp 函数，获取输出和 vjp_fn
            out, vjp_fn = vjp(h, x)
        # 对 vjp_fn 进行调用，得到 y
        (y,) = vjp_fn(torch.tensor(1.0, device=device))

        # 断言 y 的值与预期相等，即 2 * x
        self.assertEqual(y, 2 * x)
        # 断言 out 不需要梯度
        self.assertFalse(out.requires_grad)

        # 注意：这里的 y.requires_grad 为 True，这是因为 y 是通过 vjp_fn 返回的结果，其计算过程涉及到梯度
        # 断言 y 需要梯度
        self.assertTrue(y.requires_grad)

        # 计算 y 对 x 的梯度
        (z,) = torch.autograd.grad(y, x)
        # 断言计算的梯度与预期值相等，即 2
        self.assertEqual(z, 2)
# 声明一个测试类 TestAutogradFunction，用于测试自动微分功能
@markDynamoStrictTest
class TestAutogradFunction(TestCase):

    # 定义测试函数 test_set_materialize_grads，接受一个设备参数 device
    def test_set_materialize_grads(self, device):

        # 定义一个自定义的 torch.autograd.Function 类 A
        class A(torch.autograd.Function):

            # 静态方法：前向传播，接受输入 x, y，直接返回这两个输入
            @staticmethod
            def forward(x, y):
                return x, y

            # 静态方法：设置上下文 ctx，用于控制梯度材料化（materialize_grads），这里设为 False
            @staticmethod
            def setup_context(ctx, inputs, output):
                ctx.set_materialize_grads(False)

            # 静态方法：反向传播，接受梯度 gx, gy，并进行断言确保 gx 不为空，gy 为空，然后返回这两个梯度
            @staticmethod
            def backward(ctx, gx, gy):
                self.assertIsNotNone(gx)
                self.assertIsNone(gy)
                return gx, gy

        # 定义一个函数 f，接受参数 y, x
        def f(y, x):
            # 调用自定义函数 A 的 apply 方法，对 x, y 进行前向传播
            x, y = A.apply(x, y)
            # 返回 x 的平方
            return x**2

        # 创建一个张量 x，值为 2.0，使用给定的设备
        x = torch.tensor(2.0, device=device)
        # 创建一个张量 y，值为 3.0，使用给定的设备
        y = torch.tensor(3.0, device=device)
        # 对 f 函数进行梯度计算，对第一个参数（默认情况下是第一个参数）进行求导
        grad(f)(y, x)
        # 对 f 函数的梯度再次进行计算，这次对第一个参数的梯度再求导一次
        grad(grad(f))(y, x)

    # 参数化测试函数，测试多个参数组合
    @parametrize("inner_requires_grad", [True, False])
    @parametrize("save_for", ["jvp", "vjp"])
    @parametrize("save_tensors", ["input", "output", "neither"])
    @parametrize("mark_dirty", [True, False])
    def test_function_returns_input(
        self, device, inner_requires_grad, save_for, save_tensors, mark_dirty
        ):
        # 定义一个类 A，继承自 torch.autograd.Function
        class A(torch.autograd.Function):
            # 静态方法：前向传播，直接返回输入 x
            @staticmethod
            def forward(x):
                return x

            # 静态方法：设置上下文，根据 save_for 参数选择保存函数，并标记输入是否脏
            @staticmethod
            def setup_context(ctx, inputs, output):
                if save_for == "jvp":
                    save_fn = ctx.save_for_forward
                else:
                    save_fn = ctx.save_for_backward

                if mark_dirty:
                    ctx.mark_dirty(inputs[0])

                if save_tensors == "input":
                    save_fn(inputs[0])
                elif save_tensors == "output":
                    save_fn(output)
                elif save_tensors == "neither":
                    pass

            # 静态方法：反向传播，直接返回梯度输出 grad_output
            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

            # 静态方法：JVP（Jacobian-Vector Product）方法，根据 mark_dirty 选择返回值处理
            @staticmethod
            def jvp(ctx, x_t):
                # 注意：在到达此处之前，检查 ctx.save_for_forward 的逻辑已经执行
                if mark_dirty:
                    ret = x_t.add_(0)
                else:
                    ret = x_t.view_as(x_t)
                return ret

        # 定义函数 fn，调用 A 类的 apply 方法并克隆输入 x
        def fn(x):
            return A.apply(x.clone())

        # 错误信息字符串
        err_msg = "A input that has been returned as-is"

        # 创建张量 a 和 a_t，设备为 device，是否需要梯度 inner_requires_grad
        a = torch.tensor(2.0, device=device, requires_grad=inner_requires_grad)
        a_t = torch.tensor(2.0, device=device, requires_grad=inner_requires_grad)

        # 如果 save_tensors 是 "input" 或 "output"，且 mark_dirty 不为真，则使用断言捕获 RuntimeError
        if save_tensors in ("input", "output") and not mark_dirty:
            with self.assertRaisesRegex(RuntimeError, err_msg):
                grad(fn)(a)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                jvp(fn, (a,), (a_t,))
        else:
            grad(fn)(a)
            jvp(fn, (a,), (a_t,))

        # 克隆张量 a 和 a_t，并在设备 device 上创建
        a = torch.tensor(2.0, device=device, requires_grad=inner_requires_grad).clone()
        a_t = torch.tensor(
            2.0, device=device, requires_grad=inner_requires_grad
        ).clone()

        # 如果 save_tensors 是 "input" 或 "output"，且 mark_dirty 不为真，则使用断言捕获 RuntimeError
        if save_tensors in ("input", "output") and not mark_dirty:
            with self.assertRaisesRegex(RuntimeError, err_msg):
                A.apply(a)
            with self.assertRaisesRegex(RuntimeError, err_msg):
                with fwAD.dual_level():
                    A.apply(fwAD.make_dual(a, a_t))
        else:
            # 使用 A 类的 apply 方法处理张量 a，并根据 mark_dirty 进行验证
            b = A.apply(a)
            if mark_dirty:
                self.assertTrue(a is b)
            if not (
                mark_dirty and save_for == "vjp" and save_tensors in ("input", "output")
            ):
                # TODO: 访问链接 https://github.com/pytorch/pytorch/issues/97827
                with fwAD.dual_level():
                    # 创建双重级别 a_dual 和 b_dual，并使用 A 类的 apply 方法处理
                    a_dual = fwAD.make_dual(a, a_t)
                    b_dual = A.apply(a_dual)
                if mark_dirty:
                    self.assertTrue(a_dual is b_dual)
    # 定义一个测试函数，验证输入梯度需求
    def test_needs_input_grads(self, device):
        # 定义一个继承自torch.autograd.Function的类A
        class A(torch.autograd.Function):
            # 前向传播函数，返回输入张量的乘积
            @staticmethod
            def forward(x, y):
                return x * y

            # 设置上下文的静态方法，这里未实际使用
            @staticmethod
            def setup_context(ctx, inputs, output):
                return

            # 反向传播函数，验证输入梯度需求
            @staticmethod
            def backward(ctx, grad_output):
                # 断言第一个输入需要梯度
                self.assertTrue(ctx.needs_input_grad[0])
                # 断言第二个输入不需要梯度
                self.assertFalse(ctx.needs_input_grad[1])
                return None, None

        # 创建两个张量，分别用于测试
        x = torch.tensor(2.0, device=device)
        y = torch.tensor(3.0, device=device)
        # 调用自定义函数A的forward方法，计算梯度
        grad(A.apply)(x, y)
        # 调用两次自定义函数A的forward方法，进一步计算梯度
        grad(grad(A.apply))(x, y)

    # 获取一个不可组合的NumpyCubeNotComposable类
    def _get_NumpyCubeNotComposable(self):
        # 定义一个继承自torch.autograd.Function的类NumpyCubeNotComposable
        class NumpyCubeNotComposable(torch.autograd.Function):
            # 前向传播函数，将输入转为numpy并返回其立方与原始numpy数组
            @staticmethod
            def forward(input):
                input_np = input.cpu().numpy()
                return torch.tensor(input_np**3, device=input.device), input_np

            # 设置上下文的静态方法，存储输入numpy数组和设备信息
            @staticmethod
            def setup_context(ctx, inputs, output):
                ctx.input_np = output[1]
                ctx.device = inputs[0].device

            # 反向传播函数，计算梯度
            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, grad_output, grad_saved):
                result_np = 3 * (ctx.input_np**2)
                return torch.tensor(result_np, device=ctx.device)

        return NumpyCubeNotComposable

    # 测试一次可微函数的自动梯度变换
    def test_once_differentiable_autograd_vjp(self, device):
        # 获取一个不可组合的NumpyCubeNotComposable类
        NumpyCubeNotComposable = self._get_NumpyCubeNotComposable()

        # 定义一个函数f，调用NumpyCubeNotComposable的前向传播
        def f(x):
            y, _ = NumpyCubeNotComposable.apply(x)
            return y

        # 创建一个随机张量x，并设置其需要梯度
        x = torch.randn([], requires_grad=True, device=device)
        grad_y = torch.randn_like(x, requires_grad=True)
        # 获取f关于x的雅可比向量积函数
        _, vjp_fn = vjp(f, x)
        # 计算雅可比向量积
        (gx,) = vjp_fn(grad_y)

        # 断言运行时错误，提示“标记为@once_differentiable”
        with self.assertRaisesRegex(RuntimeError, "marked with @once_differentiable"):
            gx.backward()

    # TODO: support torch.autograd.function.once_differentiable
    # (or, if impossible, figure out how to raise a nice error)
    # https://github.com/pytorch/pytorch/issues/90224
    # 预期测试失败，以支持torch.autograd.function.once_differentiable
    @unittest.expectedFailure
    def test_once_differentiable_grad_vjp(self, device):
        # 获取一个不可组合的NumpyCubeNotComposable类
        NumpyCubeNotComposable = self._get_NumpyCubeNotComposable()

        # 定义一个函数h，用于计算梯度
        def h(x, grad_y):
            _, vjp_fn = vjp(f, x)  # noqa: F821
            (gx,) = vjp_fn(grad_y)
            return gx

        # 创建一个随机张量x
        x = torch.randn([], device=device)
        grad_y = torch.randn_like(x)

        # 计算h关于x和grad_y的梯度
        grad(h, argnums=(0, 1))(x, grad_y)
    # 定义一个测试方法，用于测试梯度函数的名称
    def test_grad_fn_name(self, device):
        # 创建一个空列表用于存储梯度函数的名称
        names = []

        # 定义一个自定义的 PyTorch 自动求导函数 FooBar
        class FooBar(torch.autograd.Function):
            @staticmethod
            def forward(x):
                # 前向传播函数，简单地返回输入的克隆
                return x.clone()

            @staticmethod
            def setup_context(ctx, inputs, output):
                # 设置上下文的静态方法，这里没有实际操作，返回空
                return

            @staticmethod
            def backward(ctx, grad_output):
                # 反向传播函数，直接返回梯度输出
                return grad_output

        # 定义一个简单的函数 f，其中调用了自定义的 FooBar 函数
        def f(x):
            y = FooBar.apply(x)  # 调用 FooBar 的前向传播函数
            names.append(type(y.grad_fn).__name__)  # 将梯度函数的名称添加到 names 列表中
            return y

        # 创建一个张量 x，并传入函数 f，以进行梯度计算
        x = torch.tensor(1.0)
        grad(f)(x)  # 计算函数 f 在 x 上的梯度
        # 断言 names 列表中的内容，确保其包含 "FooBarGeneratedBackward"
        self.assertEqual(names, ["FooBarGeneratedBackward"])
# 使用装饰器标记为 DynamoStrictTest 的测试类 TestAutogradFunctionVmapAPI
@markDynamoStrictTest
class TestAutogradFunctionVmapAPI(TestCase):

    # 测试函数：测试当没有 vmap 静态方法和没有生成 vmap 规则时的情况
    def test_no_vmap_staticmethod_and_no_generate_vmap_rule(self, device):
        # 定义一个名为 NumpyCube 的自定义 torch.autograd.Function 类
        class NumpyCube(torch.autograd.Function):

            # 静态方法：前向传播
            @staticmethod
            def forward(input):
                # 转换输入为 numpy 格式
                input_np = to_numpy(input)  # noqa: F821
                # 计算梯度
                dinput = torch.tensor(3 * input_np**2, device=input.device)
                # 返回前向传播结果和梯度
                return torch.tensor(input_np**3, device=input.device), dinput

            # 静态方法：设置上下文
            @staticmethod
            def setup_context(ctx, inputs, output):
                # 保存输入和输出的梯度信息
                ctx.save_for_backward(inputs, output[1])

            # 静态方法：反向传播
            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                # 抛出运行时错误
                raise RuntimeError("foobar")

        # 生成一个设备上的随机张量
        x = torch.randn(3, device=device)
        # 使用断言检查是否抛出预期的运行时错误信息
        with self.assertRaisesRegex(RuntimeError, "does not have vmap support"):
            # 对 NumpyCube.apply 进行 vmap 操作
            vmap(NumpyCube.apply)(x)

    # 测试函数：测试有 vmap 静态方法和有生成 vmap 规则时的情况
    def test_has_vmap_staticmethod_and_has_generate_vmap_rule(self, device):
        # 定义一个名为 NumpyCube 的自定义 torch.autograd.Function 类
        class NumpyCube(torch.autograd.Function):
            
            # 标志为生成 vmap 规则为 True
            generate_vmap_rule = True

            # 静态方法：前向传播
            @staticmethod
            def forward(input):
                # 转换输入为 numpy 格式
                input_np = to_numpy(input)  # noqa: F821
                # 计算梯度
                dinput = torch.tensor(3 * input_np**2, device=input.device)
                # 返回前向传播结果和梯度
                return torch.tensor(input_np**3, device=input.device), dinput

            # 静态方法：设置上下文
            @staticmethod
            def setup_context(ctx, outputs, input):
                # 保存输入和输出的梯度信息
                ctx.save_for_backward(input, outputs[1])

            # 静态方法：反向传播
            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                # 抛出运行时错误
                raise RuntimeError("foobar")

            # 静态方法：vmap 方法
            @staticmethod
            def vmap(infos, in_dims, x):
                # 抛出运行时错误
                raise RuntimeError("foobar")

        # 生成一个设备上的随机张量
        x = torch.randn(3, device=device)
        # 使用断言检查是否抛出预期的运行时错误信息
        with self.assertRaisesRegex(RuntimeError, "generate_vmap_rule=True and"):
            # 对 NumpyCube.apply 进行 vmap 操作
            vmap(NumpyCube.apply)(x)

    # 测试函数：测试 info 对象
    def test_info_object(self, device):
        # 批处理大小
        batch_size = 10

        # 定义一个名为 Id 的自定义 torch.autograd.Function 类
        class Id(torch.autograd.Function):
            
            # 静态方法：前向传播
            @staticmethod
            def forward(input):
                pass

            # 静态方法：设置上下文
            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            # 静态方法：反向传播
            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                pass

            # 静态方法：vmap 方法
            @staticmethod
            def vmap(info, in_dims, input):
                # 使用断言检查 info.batch_size 是否等于预设的批处理大小
                self.assertEqual(info.batch_size, batch_size)
                # 使用断言检查 info.randomness 是否等于随机性参数
                self.assertEqual(info.randomness, randomness)
                # 返回输入和输入维度的元组
                return input, in_dims[0]

        # 生成一个设备上的随机张量
        x = torch.randn(batch_size, 3, device=device)

        # 遍历不同的随机性参数
        for randomness in ("error", "different", "same"):
            # 对 Id.apply 进行 vmap 操作，期望的随机性参数为当前遍历的值
            vmap(Id.apply, randomness=randomness)(x)
    def test_in_dims_single_input(self, device):
        class Id(torch.autograd.Function):
            @staticmethod
            def forward(input):
                pass
            @staticmethod
            def setup_context(ctx, inputs, output):
                pass
            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                pass
            @staticmethod
            def vmap(info, in_dims, input):
                # 断言输入维度为 (1,)
                self.assertEqual(in_dims, (1,))
                return input, in_dims[0]

        B = 10
        x = torch.randn(3, B, device=device)  # 生成一个形状为 (3, B) 的张量
        vmap(Id.apply, in_dims=1)(x)  # 对 Id.apply 进行 vmap 映射，指定输入维度为 1
        vmap(Id.apply, in_dims=(1,))(x)  # 同上，使用元组形式指定输入维度为 (1,)

    def test_in_dims_multiple_inputs(self, device):
        class Id(torch.autograd.Function):
            @staticmethod
            def forward(x, y):
                pass
            @staticmethod
            def setup_context(ctx, inputs, output):
                pass
            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                pass
            @staticmethod
            def vmap(info, in_dims, x, y):
                # 断言输入维度为 (0, [0, 0])
                self.assertEqual(in_dims, (0, [0, 0]))
                # 断言 in_dims 是一个元组，且第二个元素是一个列表
                self.assertTrue(isinstance(in_dims, tuple))
                self.assertTrue(isinstance(in_dims[1], list))
                return (x, y), in_dims

        x = torch.randn(2, device=device)  # 生成一个形状为 (2,) 的张量
        vmap(Id.apply)(x, [x, x])  # 对 Id.apply 进行 vmap 映射，指定多个输入并给出输入维度

    def test_skips_empty_layer(self, device):
        class Id(torch.autograd.Function):
            @staticmethod
            def forward(input):
                return input
            @staticmethod
            def setup_context(ctx, inputs, output):
                pass
            @staticmethod
            def backward(ctx, grad_output, grad_saved):
                pass
            @staticmethod
            def vmap(info, in_dims, input):
                # 抛出运行时错误，预期不会被调用
                raise RuntimeError("expected to not be called")

        def f(x):
            y = torch.tensor(1.0)  # 创建一个标量张量 y
            y = Id.apply(y)  # 对 y 应用 Id 函数
            return x * 1  # 返回 x 的复制

        x = torch.randn(2, 3)  # 生成一个形状为 (2, 3) 的张量
        vmap(f)(x)  # 对函数 f 进行 vmap 映射
    def test_none_returns(self, device):
        # 定义一个名为 Zeros 的自定义 PyTorch 函数
        class Zeros(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，接收输入并返回形状相同的零张量
            def forward(input):
                return torch.zeros(input.shape, device=input.device)

            @staticmethod
            # 用于设置上下文的静态方法，这里没有实际操作
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            # vmap 方法的实现，断言输入维度为 (0,)，然后返回零张量和 None
            def vmap(info, in_dims, input):
                assert in_dims == (0,)
                return torch.zeros(input.shape[1:], device=input.device), None

        B = 2
        # 生成一个形状为 (B, 3) 的随机张量 x
        x = torch.randn(B, 3)
        # 对 Zeros 函数应用 vmap，生成结果 y，期望结果是形状相同的零张量
        y = vmap(Zeros.apply)(x)
        # 断言 y 与形状相同的零张量相等
        self.assertEqual(y, torch.zeros_like(x))

        # 定义一个名为 TwoZeros 的自定义 PyTorch 函数
        class TwoZeros(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，返回两个相同形状的零张量
            def forward(input):
                r = torch.zeros(input.shape, device=input.device)
                return r, r

            @staticmethod
            # 用于设置上下文的静态方法，这里没有实际操作
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            # vmap 方法的实现，断言输入维度为 (0,)，然后返回两个相同形状的零张量和 None
            def vmap(info, in_dims, input):
                assert in_dims == (0,)
                r = torch.zeros(input.shape[1:], device=input.device)
                return (r, r), None

        B = 2
        # 生成一个形状为 (B, 3) 的随机张量 x
        x = torch.randn(B, 3)
        # 对 TwoZeros 函数应用 vmap，生成结果 result，期望结果是两个形状相同的零张量的元组
        result = vmap(TwoZeros.apply)(x)

        # 断言 result 是元组类型
        self.assertTrue(isinstance(result, tuple))
        # 将 result 解包为 y 和 z
        y, z = result
        # 断言 y 与形状相同的零张量相等
        self.assertEqual(y, torch.zeros_like(x))
        # 断言 z 与形状相同的零张量相等
        self.assertEqual(z, torch.zeros_like(x))

    def test_should_have_two_returns(self, device):
        # 定义一个名为 Zeros 的自定义 PyTorch 函数
        class Zeros(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，返回一个形状相同的零张量
            def forward(input):
                r = torch.zeros(input.shape, device=input.device)
                return r

            @staticmethod
            # 用于设置上下文的静态方法，这里没有实际操作
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            # vmap 方法的实现，返回一个形状相同的零张量
            def vmap(info, in_dims, input):
                r = torch.zeros(input.shape[1:], device=input.device)
                return r

        B = 2
        # 生成一个形状为 (B, 3) 的随机张量 x
        x = torch.randn(B, 3)
        # 使用 Zeros 函数应用 vmap，预期会引发 RuntimeError，错误信息包含 "to have two returns"
        with self.assertRaisesRegex(RuntimeError, "to have two returns"):
            result = vmap(Zeros.apply)(x)

        # 定义一个名为 TwoZeros 的自定义 PyTorch 函数
        class TwoZeros(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，返回两个相同形状的零张量
            def forward(input):
                r = torch.zeros(input.shape, device=input.device)
                return r, r

            @staticmethod
            # 用于设置上下文的静态方法，这里没有实际操作
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            # vmap 方法的实现，返回两个形状相同的零张量和两个额外的零标量
            def vmap(info, in_dims, input):
                r = torch.zeros(input.shape[1:], device=input.device)
                return r, r, 0, 0

        B = 2
        # 生成一个形状为 (B, 3) 的随机张量 x
        x = torch.randn(B, 3)
        # 使用 Zeros 函数应用 vmap，预期会引发 RuntimeError，错误信息包含 "to have two returns"
        with self.assertRaisesRegex(RuntimeError, "to have two returns"):
            result = vmap(Zeros.apply)(x)
    # 定义测试类中的一个方法，用于测试不兼容的输出维度错误信息
    def test_incompatible_out_dims_error_msg(self, device):
        # 定义一个继承自 torch.autograd.Function 的类 Zeros
        class Zeros(torch.autograd.Function):
            # 静态方法：前向传播函数，接受输入并返回一个全零张量
            @staticmethod
            def forward(input):
                r = torch.zeros(input.shape, device=input.device)
                return r

            # 静态方法：设置上下文，这里不做任何操作
            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            # 静态方法：处理 vmap 操作的函数，返回一个全零张量和一个 None 的元组
            @staticmethod
            def vmap(info, in_dims, input):
                r = torch.zeros(input.shape[1:], device=input.device)
                return r, (None,)

        # 设置批处理大小 B
        B = 2
        # 生成一个形状为 (B, 3) 的随机张量 x
        x = torch.randn(B, 3)
        # 使用断言检查调用 vmap(Zeros.apply) 后是否会抛出 RuntimeError 并包含指定的错误信息
        with self.assertRaisesRegex(RuntimeError, "returned an incompatible"):
            result = vmap(Zeros.apply)(x)

        # 定义另一个继承自 torch.autograd.Function 的类 Zeros
        class Zeros(torch.autograd.Function):
            # 静态方法：前向传播函数，接受输入并返回一个包含全零张量的列表
            @staticmethod
            def forward(input):
                r = torch.zeros(input.shape, device=input.device)
                return [r]

            # 静态方法：设置上下文，这里不做任何操作
            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            # 静态方法：处理 vmap 操作的函数，返回一个包含全零张量的列表和一个 None 的元组
            @staticmethod
            def vmap(info, in_dims, input):
                r = torch.zeros(input.shape[1:], device=input.device)
                return [r], (None,)

        # 设置批处理大小 B
        B = 2
        # 生成一个形状为 (B, 3) 的随机张量 x
        x = torch.randn(B, 3)
        # 使用断言检查调用 vmap(Zeros.apply) 后是否会抛出 RuntimeError 并包含指定的错误信息
        with self.assertRaisesRegex(RuntimeError, "returned an incompatible"):
            result = vmap(Zeros.apply)(x)
# 使用装饰器将该测试类标记为 DynamoStrict 测试
@markDynamoStrictTest
class TestVmapOfGrad(TestCase):
    
    # 定义一个测试函数，用于测试 per_sample_grads_inplace_view 方法
    def test_per_sample_grads_inplace_view(self, device):
        # 定义计算损失函数的函数 compute_loss
        def compute_loss(weight, x, t):
            # 计算输入 x 与权重 weight 的矩阵乘积
            x = x.mm(weight)
            # 将 x 压缩维度（squeeze）并赋给 y
            y = x.squeeze_(0)
            # 返回损失值，即预测值 y 与目标值 t 之差的和
            return (y - t).sum()

        # 生成一个随机的权重 weight，维度为 (16, 2)，并且位于指定的设备上
        weight = torch.randn(16, 2, device=device)
        # 生成一个随机的输入 x，维度为 (64, 1, 16)，并且位于指定的设备上
        x = torch.randn(64, 1, 16, device=device)
        # 生成一个随机的目标值 t，维度为 (64, 2)，并且位于指定的设备上
        t = torch.randn(64, 2, device=device)
        # 使用 vmap 函数对 compute_loss 函数的梯度进行向量化映射计算，partial(grad(compute_loss), weight) 表示对 compute_loss 函数关于 weight 的梯度
        result = vmap(partial(grad(compute_loss), weight))(x, t)
        # 计算期望的梯度值，即对每个样本分别计算 compute_loss 函数关于 weight 的梯度
        expected = [grad(compute_loss)(weight, x[i], t[i]) for i in range(64)]
        expected = torch.stack(expected)
        # 断言测试结果与期望结果相等，设置相对误差容差 rtol 为 5e-4
        self.assertEqual(result, expected, atol=0, rtol=5e-4)

    # 定义一个测试函数，用于测试 new_zeros_materializes_tensor 方法
    def test_new_zeros_materializes_tensor(self, device):
        N = 3
        C = 5

        # 定义一个函数 foo，用于生成一个新的全零张量，并复制 y 到该张量中，最后返回求和结果
        def foo(y, x):
            result = x.new_zeros((C,))
            result.copy_(y)
            return result.sum()

        # 生成一个随机的输入 x，维度为 (N,)，并且位于指定的设备上
        x = torch.randn(N, device=device)
        # 生成一个随机的输入 y，维度为 (N, C)，并且位于指定的设备上
        y = torch.randn(N, C, device=device)
        # 使用 vmap 函数对 foo 函数的梯度进行向量化映射计算
        result = vmap(grad(foo))(y, x)
        # 断言测试结果与全一张量 torch.ones_like(y) 相等
        self.assertEqual(result, torch.ones_like(y))

    # 定义一个测试函数，用于测试 new_empty_materializes_tensor 方法
    def test_new_empty_materializes_tensor(self, device):
        N = 3
        C = 5

        # 定义一个函数 foo，用于生成一个新的空张量，并复制 y 到该张量中，最后返回求和结果
        def foo(y, x):
            result = x.new_empty((C,))
            result.copy_(y)
            return result.sum()

        # 生成一个随机的输入 x，维度为 (N,)，并且位于指定的设备上
        x = torch.randn(N, device=device)
        # 生成一个随机的输入 y，维度为 (N, C)，并且位于指定的设备上
        y = torch.randn(N, C, device=device)
        # 使用 vmap 函数对 foo 函数的梯度进行向量化映射计算
        result = vmap(grad(foo))(y, x)
        # 断言测试结果与全一张量 torch.ones_like(y) 相等
        self.assertEqual(result, torch.ones_like(y))

    # 定义一个测试函数，用于测试 per_sample_grads_simple 方法
    def test_per_sample_grads_simple(self, device):
        # 定义计算损失函数的函数 compute_loss
        def compute_loss(weight, x, t):
            # 计算输入 x 与权重 weight 的矩阵乘积
            y = x @ weight
            # 返回损失值，即预测值 y 与目标值 t 的平方差的和
            return ((y - t) ** 2).sum()

        # 生成一个随机的权重 weight，维度为 (16, 2)，并且位于指定的设备上
        weight = torch.randn(16, 2, device=device)
        # 生成一个随机的输入 x，维度为 (64, 16)，并且位于指定的设备上
        x = torch.randn(64, 16, device=device)
        # 生成一个随机的目标值 t，维度为 (64, 2)，并且位于指定的设备上
        t = torch.randn(64, 2, device=device)
        # 使用 vmap 函数对 compute_loss 函数的梯度进行向量化映射计算，partial(grad(compute_loss), weight) 表示对 compute_loss 函数关于 weight 的梯度
        result = vmap(partial(grad(compute_loss), weight))(x, t)
        # 计算期望的梯度值，即对每个样本分别计算 compute_loss 函数关于 weight 的梯度
        expected = [grad(compute_loss)(weight, x[i], t[i]) for i in range(64)]
        expected = torch.stack(expected)
        # 断言测试结果与期望结果相等，设置相对误差容差 rtol 为 5e-4
        self.assertEqual(result, expected, atol=0, rtol=5e-4)

    # 定义一个比较预期结果和实际结果的私有方法
    def _compare_expected_and_result(self, expected, result, mechanism):
        # 如果 mechanism 参数为 "make_functional"
        if mechanism == "make_functional":
            # 将 expected 结果进行转置，并对每个元素进行 torch.stack 操作
            expected = zip(*expected)
            expected = tuple(torch.stack(shards) for shards in expected)
            # 对每个结果 r 和预期结果 e 进行断言，设置相对误差容差 rtol 为 1.5e-3
            for r, e in zip(result, expected):
                self.assertEqual(r, e, atol=0, rtol=1.5e-3)
        else:
            # 否则，断言 mechanism 参数为 "functional_call"
            assert mechanism == "functional_call"
            # 对 expected 结果进行字典化处理，对每个键进行 torch.stack 操作
            expected = {
                k: tuple(d[k] for d in expected) for k, v in expected[0].items()
            }
            expected = {k: torch.stack(shards) for k, shards in expected.items()}
            # 对每个键的结果进行断言，设置相对误差容差 rtol 为 1.5e-3
            for key in result:
                self.assertEqual(result[key], expected[key], atol=0, rtol=1.5e-3)

    # 使用 tf32_on_and_off 装饰器设定相对误差容差 rtol 为 0.005，并对 "make_functional" 和 "functional_call" 两种机制进行参数化
    @tf32_on_and_off(0.005)
    @parametrize("mechanism", ["make_functional", "functional_call"])
    def test_per_sample_grads_embeddingnet(self, device, mechanism):
        # 定义一个名为 SampleNet 的内部类，继承自 nn.Module
        class SampleNet(nn.Module):
            # 构造函数，初始化模型的各个层
            def __init__(self, vocab_size: int):
                super().__init__()
                # Embedding 层，将词汇表中的单词映射为长度为 16 的向量
                self.emb = nn.Embedding(vocab_size, 16)
                # 全连接层 fc1，输入大小为 16，输出大小为 16
                self.fc1 = nn.Linear(16, 16)
                # 全连接层 fc2，输入大小为 16，输出大小为 2
                self.fc2 = nn.Linear(16, 2)

            # 前向传播函数，接受输入 x，返回预测结果
            def forward(self, x):
                # 将输入 x 映射到词嵌入空间
                x = self.emb(x)
                # 转置张量 x 的最后两个维度
                x = torch.transpose(x, -1, -2)
                # 沿着最后一个维度计算平均值
                x = torch.mean(x, -1)
                # 输入到第一个全连接层 fc1
                x = self.fc1(x)
                # 应用 ReLU 激活函数
                x = F.relu(x)
                # 输入到第二个全连接层 fc2，得到最终输出
                x = self.fc2(x)
                return x

            # 返回模型名称
            def name(self):
                return "SampleNet"

        # 创建模型的输入数据
        vocab_size = 1000
        batch_shape = [64]
        words_per_sentence = 5
        # 生成随机整数张量作为输入数据
        data = torch.randint(
            0, vocab_size, (*batch_shape, words_per_sentence), device=device
        )
        # 生成随机整数张量作为目标数据
        targets = torch.randint(0, 1, (*batch_shape,), device=device)

        # 构建 SampleNet 模型并移动到指定的设备上
        net = SampleNet(vocab_size).to(device=device)
        # 定义交叉熵损失函数
        criterion = nn.CrossEntropyLoss()

        # 获取权重和模型函数调用
        net_func, weights = _get_weights_and_functional_call(net, mechanism)

        # 定义计算损失的函数
        def compute_loss(weights, data, target):
            # 使用模型函数进行前向传播
            output = net_func(weights, data)
            # 计算交叉熵损失
            result = criterion(output, target)
            return result

        # 计算期望的梯度值
        expected = [grad(compute_loss)(weights, data[i], targets[i]) for i in range(64)]
        # 使用 vmap 函数对 compute_loss 进行矢量化计算
        result = vmap(partial(grad(compute_loss), weights))(data, targets)
        # 比较期望结果和计算结果
        self._compare_expected_and_result(expected, result, mechanism)

    def test_log_softmax(self, device):
        # 创建一个形状为 (3, 5) 的随机张量 x
        x = torch.randn(3, 5, device=device)
        # 创建一个形状为 (5,) 的随机张量 v
        v = torch.randn(5, device=device)

        # 定义一个函数 foo，返回 log_softmax 操作的反向传播值
        def foo(x, v):
            # 使用 vjp 函数获取 log_softmax 操作的 VJP（Vector-Jacobian Product）
            _, vjp_fn = vjp(partial(torch.log_softmax, dim=-1), x)
            # 计算 v 的 VJP
            return vjp_fn(v)[0]

        # 使用 vmap 函数对 foo 进行矢量化计算
        result = vmap(foo, (0, None))(x, v)

        # 将 v 广播成与 x 相同的形状
        v = v.expand_as(x)
        # 设置 x 为需要计算梯度
        x.requires_grad_()
        # 对 x 应用 log_softmax 操作
        output = torch.log_softmax(x, dim=-1)
        # 对 log_softmax 的结果进行反向传播
        output.backward(v)
        # 断言结果与计算的梯度相等
        self.assertEqual(result, x.grad)
# 使用 parametrize 装饰器为 jacrev 和 jacfwd 两个子测试参数化
jacrev_and_jacfwd = parametrize(
    "jacapi", [subtest(jacrev, name="jacrev"), subtest(jacfwd, name="jacfwd")]
)

# 使用 FIXME_jacrev_only 装饰器只针对 jacrev 进行参数化
FIXME_jacrev_only = parametrize("jacapi", [subtest(jacrev, name="jacrev")])

# 使用 markDynamoStrictTest 装饰器定义一个测试类 TestJac，并继承 VmapTearDownMixin 和 TestCase
@markDynamoStrictTest
class TestJac(VmapTearDownMixin, TestCase):

    # 使用 jacrev_and_jacfwd 装饰器定义一个测试方法 test_simple，参数为 device 和 jacapi
    @jacrev_and_jacfwd
    def test_simple(self, device, jacapi):
        # 生成一个形状为 (3,) 的随机张量 x，分配在指定的设备上
        x = torch.randn(3, device=device)
        # 对 x 中的每个元素应用 jacapi(torch.sin)，并将结果保存在 y 中
        y = jacapi(torch.sin)(x)
        # 创建一个对角矩阵，对角线元素为 x 中每个元素应用 torch.cos() 的结果
        expected = torch.diagflat(x.cos())
        # 断言 y 和 expected 在数值上接近
        assert torch.allclose(y, expected)

    # 使用 jacrev_and_jacfwd 装饰器定义一个测试方法 test_simple_not_flat，参数为 device 和 jacapi
    @jacrev_and_jacfwd
    def test_simple_not_flat(self, device, jacapi):
        # 生成一个形状为 (2, 3) 的随机张量 x，分配在指定的设备上
        x = torch.randn(2, 3, device=device)
        # 对 x 中的每个元素应用 jacapi(torch.sin)，并将结果保存在 y 中
        y = jacapi(torch.sin)(x)
        # 创建一个形状为 (2, 3, 2, 3) 的张量，对角线元素为 x.view(-1) 中每个元素应用 torch.cos() 的结果
        expected = torch.diagflat(x.view(-1).cos())
        # 将 expected 的形状变为 (2, 3, 2, 3)
        expected = expected.view(2, 3, 2, 3)
        # 断言 y 和 expected 在数值上接近
        assert torch.allclose(y, expected)

    # 使用 jacrev_and_jacfwd 装饰器定义一个测试方法 test_take，参数为 device 和 jacapi
    @jacrev_and_jacfwd
    def test_take(self, device, jacapi):
        # 生成一个包含 5 个随机数的张量 x
        x = torch.rand(5)

        # 定义一个函数 func，接受 x 作为输入
        def func(x):
            # 生成一个 dtype 为 torch.long 的全 1 张量 y
            y = torch.ones(3, dtype=torch.long)
            # 从 x 中取出索引为 y 的元素，并将结果保存在 z 中
            z = torch.take(x, y)
            return z

        # 断言 jacrev(func)(x) 和 torch.autograd.functional.jacobian(func, x) 相等
        self.assertEqual(jacrev(func)(x), torch.autograd.functional.jacobian(func, x))

    # 使用 jacrev_and_jacfwd 装饰器定义一个测试方法 test_diff_numel，参数为 device 和 jacapi
    @jacrev_and_jacfwd
    def test_diff_numel(self, device, jacapi):
        # 生成一个形状为 (2, 4) 的随机张量 x，分配在指定的设备上
        x = torch.randn(2, 4, device=device)

        # 定义一个函数 f，接受 x 作为输入
        def f(x):
            # 返回 x[0, 1:] 扩展一个维度后的结果
            return x[0, 1:].unsqueeze(-1)

        # 对 f(x) 应用 jacapi，将结果保存在 y 中
        y = jacapi(f)(x)
        # 断言 y 的形状为 (3, 1, 2, 4)
        self.assertEqual(y.shape, (3, 1, 2, 4))

        # 生成一个形状和数据类型与 x 相同的全 0 张量 expected
        expected = x.new_zeros(3, 1, 2, 4)
        # 设置 expected 的特定位置为 1
        expected[0, 0, 0, 1] = 1
        expected[1, 0, 0, 2] = 1
        expected[2, 0, 0, 3] = 1
        # 断言 y 和 expected 相等
        self.assertEqual(y, expected)

    # 使用 jacrev_and_jacfwd 装饰器定义一个测试方法 test_vmap_on_jac_simple，参数为 device 和 jacapi
    @jacrev_and_jacfwd
    def test_vmap_on_jac_simple(self, device, jacapi):
        # 生成一个形状为 (2, 3) 的随机张量 x，分配在指定的设备上
        x = torch.randn(2, 3, device=device)
        # 对 jacapi(torch.sin) 应用 vmap，将结果保存在 y 中
        y = vmap(jacapi(torch.sin))(x)
        # 生成一个形状和数据类型与 x 相同的张量，每个子张量都是对应 x[i] 的对角矩阵 torch.diagflat(x[i].cos()) 的结果
        expected = torch.stack([torch.diagflat(x[i].cos()) for i in range(2)])
        # 断言 y 和 expected 在数值上接近
        assert torch.allclose(y, expected)

    # 使用 jacrev_and_jacfwd 装饰器定义一个测试方法 test_nested_jac_simple，参数为 device 和 jacapi
    @jacrev_and_jacfwd
    def test_nested_jac_simple(self, device, jacapi):
        # 定义一个函数 foo，接受 x 作为输入
        def foo(x):
            # 返回 x.sin().sum() 的结果
            return x.sin().sum()

        # 生成一个形状为 (3,) 的随机张量 x，分配在指定的设备上
        x = torch.randn(3, device=device)
        # 对 jacapi(jacapi(foo)) 应用，将结果保存在 y 中
        y = jacapi(jacapi(foo))(x)
        # 生成一个对角矩阵，对角线元素为 -x.sin() 的结果
        expected = torch.diagflat(-x.sin())
        # 断言 y 和 expected 在数值上接近
        assert torch.allclose(y, expected)

    # 使用 jacrev_and_jacfwd 装饰器定义一个测试方法 test_multiple_args，参数为 device 和 jacapi
    @jacrev_and_jacfwd
    def test_multiple_args(self, device, jacapi):
        # 生成两个形状为 (3,) 的随机张量 x 和 y，分配在指定的设备上
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        # 对 jacapi(torch.multiply, argnums=1) 应用，将结果保存在 z 中
        z = jacapi(torch.multiply, argnums=1)(x, y)
        # 生成一个对角矩阵，对角线元素为 x 的结果
        expected = torch.diagflat(x)
        # 断言 z 和 expected 在数值上接近
        assert torch.allclose(z, expected)
    @jacrev_and_jacfwd
    def test_multiple_outputs_multiple_argnums(self, device, jacapi):
        # 定义一个函数 f，接受两个参数 x 和 y，返回两个值的元组
        def f(x, y):
            return 2 * x + 3 * y, 4 * x + 5 * y

        # 生成一个形状为 (3,) 的随机张量 x，设备为给定的 device
        x = torch.randn(3, device=device)
        # 生成一个形状为 (3,) 的随机张量 y，设备为给定的 device
        y = torch.randn(3, device=device)
        # 使用 jacapi 对函数 f 进行自动求导，指定对参数 (0, 1) 求导
        z = jacapi(f, argnums=(0, 1))(x, y)
        # 生成一个对角线矩阵，对角线元素为 x 中每个元素乘以 2
        expected_out0_x = torch.diagflat(torch.full_like(x, 2))
        # 生成一个对角线矩阵，对角线元素为 y 中每个元素乘以 3
        expected_out0_y = torch.diagflat(torch.full_like(y, 3))
        # 生成一个对角线矩阵，对角线元素为 x 中每个元素乘以 4
        expected_out1_x = torch.diagflat(torch.full_like(x, 4))
        # 生成一个对角线矩阵，对角线元素为 y 中每个元素乘以 5
        expected_out1_y = torch.diagflat(torch.full_like(y, 5))

        # 断言 z 的长度为 2
        self.assertEqual(len(z), 2)
        # 断言 z 是一个元组
        self.assertTrue(isinstance(z, tuple))
        # 断言 z 的第一个元素是一个长度为 2 的元组
        self.assertEqual(len(z[0]), 2)
        # 断言 z 的第一个元素是一个元组
        self.assertTrue(isinstance(z[0], tuple))
        # 断言 z 的第一个元素的第一个子元素等于 expected_out0_x
        self.assertEqual(z[0][0], expected_out0_x)
        # 断言 z 的第一个元素的第二个子元素等于 expected_out0_y
        self.assertEqual(z[0][1], expected_out0_y)
        # 断言 z 的第二个元素的第一个子元素等于 expected_out1_x
        self.assertEqual(z[1][0], expected_out1_x)
        # 断言 z 的第二个元素的第二个子元素等于 expected_out1_y
        self.assertEqual(z[1][1], expected_out1_y)
    # 定义一个测试函数，使用 pytree 版本的自动微分库进行测试
    def test_multiple_inputs_pytree(self, device, jacapi):
        # 定义一个简单的函数 f，接受三个参数，并对其进行数学运算
        def f(a, b, c):
            # 解构 a 参数为 a0 和 a1，然后返回根据这些参数计算得出的结果
            a0, a1 = a
            return a0 + a1 * 2 + b * 3 + c * 4

        # 在给定设备上生成一个随机张量 x
        x = torch.randn([], device=device)
        # 构造函数 f 的参数列表 args，包含多个输入变量
        args = ((x, x), x, x)

        # 使用 jacapi 对函数 f 进行自动微分，指定参数的索引为 (0, 1, 2)，然后传入 args 运行并获取结果
        result = jacapi(f, argnums=(0, 1, 2))(*args)
        # 预期的结果 expected，用来与 result 进行比较
        expected = (
            (torch.tensor(1.0, device=device), torch.tensor(2.0, device=device)),
            torch.tensor(3.0, device=device),
            torch.tensor(4.0, device=device),
        )
        # 断言 result 和 expected 相等
        self.assertEqual(result, expected)

        # 使用 jacapi 对函数 f 进行自动微分，指定参数的索引为 (0,)，然后传入 args 运行并获取结果
        result = jacapi(f, argnums=(0,))(*args)
        # 预期的结果 expected，用来与 result 进行比较
        expected = (
            (torch.tensor(1.0, device=device), torch.tensor(2.0, device=device)),
        )
        # 断言 result 和 expected 相等
        self.assertEqual(result, expected)

        # 使用 jacapi 对函数 f 进行自动微分，不指定特定参数索引，直接传入 args 运行并获取结果
        result = jacapi(f)(*args)
        # 预期的结果 expected，用来与 result 进行比较
        expected = (torch.tensor(1.0, device=device), torch.tensor(2.0, device=device))
        # 断言 result 和 expected 相等
        self.assertEqual(result, expected)
    @jacrev_and_jacfwd
    def test_outputs_can_any_pytree(self, device, jacapi):
        # 创建一个形状为 (2, 3) 的随机张量 x，设备为 device
        x = torch.randn(2, 3, device=device)

        # 测试 output 为 None 或空元组时的异常情况
        for output in [None, ()]:
            # 使用断言检查是否会抛出 RuntimeError 异常，并验证异常信息中包含特定字符串
            with self.assertRaisesRegex(
                RuntimeError,
                r"(vjp|jvp).+: Expected f to be a function that has non-empty output",
            ):
                jacapi(lambda _: output)(x)

        # 测试 output 类型为整数、布尔值、浮点数、字符串时的异常情况
        for output in [1, True, 12.2, "abc"]:
            # 使用断言检查是否会抛出 RuntimeError 异常，并验证异常信息中包含特定字符串
            with self.assertRaisesRegex(
                RuntimeError,
                r"(vjp|jvp).+: expected f\(\*primals\) to return only tensors",
            ):
                jacapi(lambda _: output)(x)

        # 检查列表类型的输出
        out = jacapi(lambda x: [x, x.sum()])(x)
        # 使用断言验证 out 是一个列表且长度为 2
        assert isinstance(out, list) and len(out) == 2

        # 检查字典类型的输出
        out = jacapi(lambda x: {"x": x, "xsum": x.sum()})(x)
        # 使用断言验证 out 是一个字典且长度为 2，且包含键 "xsum"
        assert isinstance(out, dict) and len(out) == 2 and "xsum" in out

        # 定义一个复合输出的函数
        def composite_output(x):
            out = x.sum()
            return [
                (out, {"a": x, "out": [x, out]}),
            ]

        # 测试复合输出函数
        out = jacapi(composite_output)(x)
        # 使用断言验证 out 是一个列表，并且第一个元素是一个元组，其中第二个元素是一个字典
        assert isinstance(out, list)
        assert isinstance(out[0], tuple) and isinstance(out[0][1], dict)
    # 测试对于不相关的输入是否能正确计算 Jacobian 矩阵
    def test_unrelated_input(self, device, jacapi):
        # 定义一个函数 f(x, y)，返回 x
        def f(x, y):
            return x
        
        # 生成随机张量 x 和 y，设备为指定设备
        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)
        
        # 使用 jacapi 计算 f 的 Jacobian 矩阵，对应参数为 (0, 1)
        result = jacapi(f, argnums=(0, 1))(x, y)
        
        # 生成预期的结果 expected0 和 expected1
        expected0 = torch.eye(6, 6, device=device).view(2, 3, 2, 3)
        expected1 = y.new_zeros(2, 3, 2, 3)
        expected = (expected0, expected1)
        
        # 断言 result 是一个元组
        self.assertTrue(isinstance(result, tuple))
        # 断言 result 等于预期的 expected 结果
        self.assertEqual(result, expected)

    # 使用 jacrev_and_jacfwd 修饰器测试对于不相关的输出是否能正确计算 Jacobian 矩阵
    @jacrev_and_jacfwd
    def test_unrelated_output(self, device, jacapi):
        # 生成随机张量 y，设备为指定设备
        y = torch.randn(2, 3, device=device)
        
        # 定义一个函数 f(x)，始终返回 y
        def f(x):
            return y
        
        # 生成随机张量 x，设备为指定设备
        x = torch.randn(2, 3, device=device)
        
        # 使用 jacapi 计算 f 的 Jacobian 矩阵
        result = jacapi(f)(x)
        
        # 生成预期的结果 expected
        expected = x.new_zeros(2, 3, 2, 3)
        
        # 断言 result 等于预期的 expected 结果
        self.assertEqual(result, expected)

    # 使用 jacrev_and_jacfwd 修饰器测试空输出时是否能引发 RuntimeError
    @jacrev_and_jacfwd
    def test_empty_output(self, device, jacapi):
        # 生成随机张量 x 和 y，设备为指定设备
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        
        # 定义一个函数 f(x, y)，返回空元组
        def f(x, y):
            return ()
        
        # 使用 jacapi 计算 f 的 Jacobian 矩阵，预期会引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "xpected"):
            jacapi(f)(x, y)

    # 使用 jacrev_and_jacfwd 修饰器测试 argnums 参数为元组时是否正确计算 Jacobian 矩阵
    @jacrev_and_jacfwd
    def test_argnums_tuple(self, device, jacapi):
        # 生成随机张量 x 和 y，设备为指定设备
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        
        # 使用 jacapi 计算 torch.multiply 函数的 Jacobian 矩阵，对应参数为 (0, 1)
        z = jacapi(torch.multiply, argnums=(0, 1))(x, y)
        
        # 生成预期的结果 expected0 和 expected1
        expected0 = torch.diagflat(y)
        expected1 = torch.diagflat(x)
        
        # 断言 z 是一个长度为 2 的列表
        assert len(z) == 2
        # 断言 z[0] 等于预期的 expected0 结果
        assert torch.allclose(z[0], expected0)
        # 断言 z[1] 等于预期的 expected1 结果
        assert torch.allclose(z[1], expected1)

    # 使用 jacrev_and_jacfwd 修饰器测试 argnums 参数影响返回值的情况
    @jacrev_and_jacfwd
    def test_argnums_effect_on_return(self, device, jacapi):
        # 生成随机张量 x 和 y，设备为指定设备
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        
        # 使用 jacapi 计算 torch.multiply 函数的 Jacobian 矩阵，对应参数为 (0,)
        z = jacapi(torch.multiply, argnums=(0,))(x, y)
        
        # 生成预期的结果 expected0
        expected0 = torch.diagflat(y)
        
        # 断言 z 是一个元组
        assert isinstance(z, tuple)
        # 断言 z 的长度为 1
        assert len(z) == 1
        # 断言 z[0] 等于预期的 expected0 结果
        assert torch.allclose(z[0], expected0)
        
        # 使用 jacapi 计算 torch.multiply 函数的 Jacobian 矩阵，对应参数为 0
        z = jacapi(torch.multiply, argnums=0)(x, y)
        
        # 断言 z 是一个 torch.Tensor
        assert isinstance(z, torch.Tensor)
        # 断言 z 等于预期的 expected0 结果
        assert torch.allclose(z, expected0)

    # 使用 jacrev_and_jacfwd 修饰器测试 argnums 参数默认为 0 的情况
    @jacrev_and_jacfwd
    def test_argnums_defaults_to_zero(self, device, jacapi):
        # 定义一个函数 f(x, y)，返回 x * 2 + y * 3
        def f(x, y):
            return x * 2 + y * 3
        
        # 生成随机张量 x 和 y，设备为指定设备
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        
        # 使用 jacapi 计算 f 的 Jacobian 矩阵
        z = jacapi(f)(x, y)
        
        # 生成预期的结果 expected
        expected = torch.diagflat(torch.full_like(x, 2))
        
        # 断言 z 等于预期的 expected 结果
        self.assertEqual(z, expected)

    # 使用 jacrev_and_jacfwd 修饰器测试 argnums 参数为空元组时是否能引发 RuntimeError
    @jacrev_and_jacfwd
    def test_empty_argnums(self, device, jacapi):
        # 生成随机张量 x，设备为指定设备
        x = torch.randn(3, device=device)
        
        # 使用 jacapi 计算 torch.sin 函数的 Jacobian 矩阵，argnums 参数为空元组
        with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
            jacapi(torch.sin, argnums=())(x)

    # 使用 jacrev_and_jacfwd 修饰器测试 argnums 参数超出边界时是否能引发 RuntimeError
    @jacrev_and_jacfwd
    def test_out_of_bounds_argnums(self, device, jacapi):
        # 生成随机张量 x，设备为指定设备
        x = torch.randn(3, device=device)
        
        # 使用 jacapi 计算 torch.sin 函数的 Jacobian 矩阵，argnums 参数为 2
        with self.assertRaisesRegex(RuntimeError, "only 1 positional inputs"):
            jacapi(torch.sin, argnums=2)(x)
    # 测试负的 argnums 参数是否引发异常
    def test_negative_argnums(self, device, jacapi):
        # 创建一个在指定设备上的随机张量
        x = torch.randn(3, device=device)
        # 使用 assertRaisesRegex 断言上下文，检查是否引发 RuntimeError 异常，并验证异常消息是否包含指定字符串
        with self.assertRaisesRegex(RuntimeError, "only 1 positional inputs"):
            # 调用 jacapi 函数，并尝试传入负数的 argnums 参数来计算梯度
            jacapi(torch.sin, argnums=-2)(x)

    # 测试重复的 argnums 参数是否引发异常
    @jacrev_and_jacfwd
    def test_repeated_argnums(self, device, jacapi):
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            # 调用 jacapi 函数，传入重复的 argnums 参数来计算梯度
            jacapi(torch.sin, argnums=(0, 0))(x)

    # 测试浮点数作为 argnums 参数是否引发异常
    @jacrev_and_jacfwd
    def test_float_argnums(self, device, jacapi):
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be int or Tuple"):
            # 调用 jacapi 函数，并传入浮点数作为 argnums 参数来计算梯度
            jacapi(torch.sin, argnums=0.0)(x)
        with self.assertRaisesRegex(RuntimeError, "must be int"):
            # 调用 jacapi 函数，并传入元组中包含浮点数的 argnums 参数来计算梯度
            jacapi(torch.multiply, argnums=(1, 0.0))(x, x)

    # 测试简单的 Hessian 矩阵计算
    def test_hessian_simple(self, device):
        # 定义一个简单的函数 f(x)，返回 x 的正弦值
        def f(x):
            return x.sin()

        # 创建一个在指定设备上的随机张量
        x = torch.randn(3, device=device)
        # 使用 hessian 函数计算 f 在 x 处的 Hessian 矩阵
        hessian(f)(x)

    # 对比测试函数输出和预期输出是否一致的辅助函数
    def _test_against_reference(self, f, inputs, jacapi):
        # 定义一个辅助函数 foo，调用原始函数 f，并传入 inputs 元组作为参数
        def foo(inputs):
            return f(*inputs)

        # 使用 torch.autograd.functional.jacobian 计算函数 f 在给定输入 inputs 上的 Jacobian 矩阵的预期输出
        expected = torch.autograd.functional.jacobian(f, inputs)
        # 使用 jacapi 函数计算函数 foo 在给定输入 inputs 上的 Jacobian 矩阵的实际输出
        result = jacapi(foo)(inputs)
        # 使用 self.assertEqual 检查计算结果和预期结果是否相等
        self.assertEqual(result, expected)

    # 测试简单函数与参考函数在 Jacobian 矩阵上的计算结果是否一致
    @jacrev_and_jacfwd
    def test_against_reference_simple(self, device, jacapi):
        # 定义一个简单的函数 f(x)，返回 3 * x^2
        def f(x):
            return 3 * x**2

        # 创建一个在指定设备上的随机张量
        x = torch.randn(2, 3, 5, device=device)
        # 调用 _test_against_reference 函数，比较 f 在 x 上的计算结果与预期结果
        self._test_against_reference(f, (x,), jacapi)

    # 测试多输入函数与参考函数在 Jacobian 矩阵上的计算结果是否一致
    @jacrev_and_jacfwd
    def test_against_reference_multi_input(self, device, jacapi):
        # 定义一个多输入函数 f(x, y)，返回 (x.cos() * x) @ y.sin()
        def f(x, y):
            return (x.cos() * x) @ y.sin()

        # 创建在指定设备上的随机张量 x 和 y
        x = torch.randn(2, 3, device=device)
        y = torch.randn(3, 5, device=device)
        # 调用 _test_against_reference 函数，比较 f 在 (x, y) 上的计算结果与预期结果
        self._test_against_reference(f, (x, y), jacapi)

    # 测试多输入多输出函数与参考函数在 Jacobian 矩阵上的计算结果是否一致
    @jacrev_and_jacfwd
    def test_against_reference_multi_input_multi_output(self, device, jacapi):
        # 定义一个多输入多输出函数 f(x, y)，返回三个值：(x * x) @ y, x @ (x.sum(1) * y), y.sum()
        def f(x, y):
            return (x * x) @ y, x @ (x.sum(1) * y), y.sum()

        # 创建在指定设备上的随机张量 x 和 y
        x = torch.randn(5, 3, device=device)
        y = torch.randn(3, 5, device=device)
        # 调用 _test_against_reference 函数，比较 f 在 (x, y) 上的计算结果与预期结果
        self._test_against_reference(f, (x, y), jacapi)

    # 测试多输出函数与参考函数在 Jacobian 矩阵上的计算结果是否一致
    @jacrev_and_jacfwd
    def test_against_reference_unrelated_outputs(self, device, jacapi):
        # 定义一个多输出函数 f(x, y)，返回四个值：x, y, x, y
        def f(x, y):
            return x, y, x, y

        # 创建在指定设备上的随机张量 x 和 y
        x = torch.randn(2, device=device)
        y = torch.randn(3, device=device)
        # 调用 _test_against_reference 函数，比较 f 在 (x, y) 上的计算结果与预期结果
        self._test_against_reference(f, (x, y), jacapi)

    # 下面的测试函数未完整提供，需要继续编写
    def test_against_reference_zero_dim(self, device, jacapi):
        # 定义函数f，计算两个张量的和、乘积，返回元组
        def f(x, y):
            return x.sum(), y.sum(), x * y

        # 生成一个形状为(3,)的随机张量x，设备为device
        x = torch.randn(3, device=device)
        # 生成一个形状为(3,)的随机张量y，设备为device
        y = torch.randn(3, device=device)
        # 调用_test_against_reference方法，测试函数f在输入(x, y)下的结果
        self._test_against_reference(f, (x, y), jacapi)

        # 定义函数g，将零维张量x堆叠三次
        def g(x):
            return torch.stack([x, x, x])

        # 生成一个零维张量x，设备为device
        x = torch.randn([], device=device)
        # 调用_test_against_reference方法，测试函数g在输入(x,)下的结果
        self._test_against_reference(g, (x,), jacapi)

        # 定义函数h，计算一个张量y的和及乘积，返回元组
        def h(x, y):
            return y.sum(), x * y

        # 生成一个零维张量x，设备为device
        x = torch.randn([], device=device)
        # 生成一个形状为(1,)的随机张量y，设备为device
        y = torch.randn(1, device=device)
        # 调用_test_against_reference方法，测试函数h在输入(x, y)下的结果
        self._test_against_reference(h, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_correctness_different_devices(self, device, jacapi):
        # 定义函数f，计算两个张量的乘积，并将结果转移到指定设备上
        def f(x, y):
            return x * y, (x * y).to(device=device)

        # 生成一个形状为(3,)的随机张量x
        x = torch.randn(3)
        # 生成一个形状为(3,)的随机张量y
        y = torch.randn(3)
        # 调用_test_against_reference方法，测试函数f在输入(x, y)下的结果
        self._test_against_reference(f, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_default_arg(self, device, jacapi):
        # 定义函数f，计算两个张量的乘积并乘以默认值z
        def f(x, y, z=3.0):
            return x * y * z

        # 生成一个形状为(3,)的随机张量x，设备为device
        x = torch.randn(3, device=device)
        # 生成一个形状为(3,)的随机张量y，设备为device
        y = torch.randn(3, device=device)
        # 调用_test_against_reference方法，测试函数f在输入(x, y)下的结果
        self._test_against_reference(f, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_inplace(self, device, jacapi):
        # 定义函数f，将张量x的值复制给张量y并返回y
        def f(x, y):
            y.copy_(x)
            return y

        # 使用jacapi对函数f求导，指定argnums=0表示对x进行求导
        out = jacapi(f, argnums=0)
        # 生成两个形状为(2,)的随机张量x和y，设备为device
        x, y = torch.randn(2, device=device), torch.randn(2, device=device)
        # 断言调用out函数对x和y的结果等于单位矩阵
        self.assertEqual(out(x, y), torch.eye(y.shape[0]))

        # 定义函数g，将张量x的前两个元素替换为张量y的值，并返回两个元素的平方和及张量z的立方和
        def g(x, y, z):
            x[:2] = y
            return torch.vstack([(x**2).sum(), (z**3).sum()])

        # 使用jacapi对函数g进行求导，指定argnums=(1, 2)表示对y和z进行求导
        out = jacapi(g, argnums=(1, 2))
        # 生成三个形状为(3,)的随机张量x、y、z，设备为device
        x, y, z = (
            torch.randn(3, device=device),
            torch.randn(2, device=device),
            torch.randn(2, device=device),
        )

        # 生成期望输出，包含两个形状为(2, 1, 2)的零张量，设备为device
        expected_out = (
            torch.zeros(2, 1, 2, device=device),
            torch.zeros(2, 1, 2, device=device),
        )
        # 设置期望输出的值
        expected_out[0][0][0] = 2 * y  # 左上角
        expected_out[1][1][0] = 3 * (z**2)  # 右下角

        # 调用out函数计算x、y、z的结果，断言结果等于期望输出
        out_val = out(x, y, z)
        self.assertEqual(out_val, expected_out)

    @parametrize("_preallocate_and_copy", (True, False))
    # 定义一个测试函数，用于测试带有不同参数的 `jacrev` 函数
    def test_chunk_jacrev(self, device, _preallocate_and_copy):
        # 生成一个在指定设备上的随机张量 x，形状为 (10, 2)
        x = torch.randn(10, 2, device=device)
        # 生成一个在指定设备上的随机张量 y，形状为 (1, 2)
        y = torch.randn(1, 2, device=device)

        # 定义一个嵌套函数 f，接受 x 和 y 作为参数，返回两个元组
        def f(x, y):
            # 第一个元组中包含 x 的正弦值和 x 加上 y
            return (x.sin(), x + y), (x + 2, x.sum())

        # 遍历不同的 chunk_size 值进行测试
        for chunk_size in (1, 2, 3, 4, 7, 10, 1000):
            # 计算使用 jacrev 函数求取的梯度期望值
            expected = jacrev(f, argnums=(0, 1))(x, y)
            # 计算使用 jacrev 函数（带有 chunk_size 参数）求取的梯度实际值
            actual = jacrev(
                f,
                argnums=(0, 1),
                chunk_size=chunk_size,
                _preallocate_and_copy=_preallocate_and_copy,
            )(x, y)
            # 断言实际值与期望值相等
            self.assertEqual(actual, expected)

        # 当 chunk_size 小于等于 0 时，期望引发 ValueError 异常
        err_msg = "jacrev: `chunk_size` should be greater than 0."
        with self.assertRaisesRegex(ValueError, err_msg):
            jacrev(f, argnums=(0,), chunk_size=0)(x, y)

        with self.assertRaisesRegex(ValueError, err_msg):
            jacrev(f, argnums=(0,), chunk_size=-2)(x, y)

    # 通过参数化装饰器指定不同的 _preallocate_and_copy 值进行测试
    @parametrize("_preallocate_and_copy", (True, False))
    def test_chunk_jacrev_composition(self, device, _preallocate_and_copy):
        # 生成一个在指定设备上的随机张量 x，形状为 (10, 2)
        x = torch.randn(10, 2, device=device)
        # 设置 chunk_size 值为 3
        chunk_size = 3

        # 定义一个嵌套函数 f，接受 x 作为参数，返回两个元组
        def f(x):
            # 第一个元组中包含 x 的正弦值和 x 本身
            return (x.sin(), x), (x + 2, x.sum())

        # 计算嵌套使用 jacrev 函数两次的梯度期望值
        expected = vmap(jacrev(jacrev(f)))(x)
        # 计算嵌套使用 jacrev 函数两次（带有 chunk_size 参数）的梯度实际值
        actual = vmap(
            jacrev(
                jacrev(
                    f,
                    chunk_size=chunk_size,
                    _preallocate_and_copy=_preallocate_and_copy,
                ),
                chunk_size=chunk_size,
            )
        )(x)
        # 断言实际值与期望值相等
        self.assertEqual(actual, expected)

    # 用于标记一个已知的问题链接，这里是关于 PyTorch 的 GitHub 问题页面
    @xfailIfTorchDynamo
    @parametrize("_preallocate_and_copy", (True, False))
    def test_chunk_jacrev_chunksize_one(self, device, _preallocate_and_copy):
        # 使用设备上的随机张量创建一个 3x3 的张量 x
        x = torch.randn(3, 3, device=device)

        # 具有动态操作的反向传播函数
        # 这应该导致 jacrev/vmap(vjp) 失败。
        class IdentityWithDynamicBackwardOp(torch.autograd.Function):
            @staticmethod
            def forward(input):
                return input

            @staticmethod
            def setup_context(ctx, inputs, output):
                pass

            @staticmethod
            def backward(ctx, grad_output):
                # 在反向传播中使用动态操作。
                grad_output.nonzero()
                return grad_output

        def f(x):
            return IdentityWithDynamicBackwardOp.apply(x)

        # 使用 `chunk_size=1`，我们不使用 vmap。因此以下代码应该可以工作。
        jacfn = jacrev(f, chunk_size=1, _preallocate_and_copy=_preallocate_and_copy)
        actual = jacfn(x)
        expected = torch.autograd.functional.jacobian(f, x, vectorize=False)
        self.assertEqual(actual, expected)

        # 使用 `chunk_size=2` 应该会失败。
        msg = (
            r"vmap: We do not support batching operators that can output dynamic shape."
        )
        with self.assertRaisesRegex(RuntimeError, msg):
            jacrev(f, chunk_size=2, _preallocate_and_copy=_preallocate_and_copy)(x)

    def test_complex_error(self, device):
        # 验证复杂输入会引发错误
        # C -> C
        def fn(x):
            return x.conj()

        x = torch.randn(1, device=device, dtype=torch.cfloat)

        with self.assertRaisesRegex(RuntimeError, "jacrev: Expected all inputs"):
            jacrev(fn)(x)

        with self.assertRaisesRegex(RuntimeError, "jacfwd: Expected all inputs"):
            jacfwd(fn)(x)

        # 验证复杂输出会引发错误
        # R -> C
        def fn(x):
            return torch.conj(x * 0.5j)

        x = torch.randn(1, device=device, dtype=torch.float)

        with self.assertRaisesRegex(RuntimeError, "jacrev: Expected all outputs"):
            jacrev(fn)(x)

        with self.assertRaisesRegex(RuntimeError, "jacfwd: Expected all outputs"):
            jacfwd(fn)(x)

    @jacrev_and_jacfwd
    def test_jac_with_non_tensor_args(self, device, jacapi):
        def f(t, int_x):
            return t + int_x

        t = torch.randn(3, 3, device=device)

        actual = jacapi(f)(t, 3)
        expected = torch.autograd.functional.jacobian(partial(f, int_x=3), t)
        self.assertEqual(actual, expected)
@markDynamoStrictTest
class TestHessian(TestCase):
    # 测试类标记为 DynamoStrictTest

    def _test_against_reference(self, f, inputs):
        # 定义一个用于测试函数 f 的私有方法，参数为 f 和输入 inputs
        def foo(inputs):
            return f(*inputs)
        # 定义内部函数 foo，用于调用 f 并返回结果

        expected = torch.autograd.functional.hessian(f, inputs)
        # 计算函数 f 的输入 inputs 的 Hessian 矩阵作为期望结果
        result = hessian(foo)(inputs)
        # 使用自定义的 hessian 函数计算 foo 的 Hessian 矩阵作为实际结果
        self.assertEqual(result, expected)
        # 断言计算结果与期望结果相等

    def test_hessian_vectorize_correctness_simple(self, device):
        # 测试简单的 Hessian 向量化正确性

        def f(x):
            return (3 * x**2).sum()
        # 定义一个简单的函数 f，计算给定输入 x 的平方乘以 3 的和

        x = torch.randn(2, 3, 5, device=device)
        # 创建一个随机张量 x，形状为 (2, 3, 5)，使用指定的设备
        self._test_against_reference(f, (x,))
        # 调用私有方法 _test_against_reference 测试函数 f，传入参数 x 的元组

    def test_hessian_vectorize_correctness_multi_input(self, device):
        # 测试多输入的 Hessian 向量化正确性

        def f(x, y, z):
            return ((x.relu() * x) @ y.sin() @ z).sum()
        # 定义一个函数 f，包含多个输入 x, y, z，计算它们的一系列操作的和

        x = torch.randn(2, 3, device=device)
        y = torch.randn(3, 5, device=device)
        z = torch.randn(5, 5, device=device)
        # 创建随机张量 x, y, z，分别使用指定的设备
        self._test_against_reference(f, (x, y, z))
        # 调用私有方法 _test_against_reference 测试函数 f，传入参数 x, y, z 的元组

    def test_hessian_vectorize_correctness_unrelated_outputs(self, device):
        # 测试输出与输入无关的 Hessian 向量化正确性

        # 输出与其中一个输入无关
        def f(x, y):
            return (x**2).sum()
        # 定义一个函数 f，计算输入 x 的平方和

        x = torch.randn(2, device=device)
        y = torch.randn(3, device=device)
        # 创建随机张量 x, y，使用指定的设备
        self._test_against_reference(f, (x, y))
        # 调用私有方法 _test_against_reference 测试函数 f，传入参数 x, y 的元组

        # 输出与所有输入无关
        def f(x, y):
            return torch.ones([])
        # 定义一个函数 f，返回一个标量张量 1.0

        x = torch.randn(2, device=device)
        y = torch.randn(3, device=device)
        # 创建随机张量 x, y，使用指定的设备
        self._test_against_reference(f, (x, y))
        # 调用私有方法 _test_against_reference 测试函数 f，传入参数 x, y 的元组

    def test_jacfwd_different_levels(self, device):
        # 测试不同级别的 Jacobian 向前自动求导

        # 从以下链接的测试案例：
        # https://github.com/pytorch/functorch/issues/597
        b = 8
        n = 100
        d = 2
        x1 = torch.randn(b, n, d, device=device)
        x2 = x1
        A = 0.1 * torch.randn(b, d, d, device=device)
        # 创建随机张量 b, n, d, A，使用指定的设备

        def loss(A, x1, x2):
            x2_hat = (A @ (x1.T)).T
            res = x2 - x2_hat
            res_sqr = res**2
            return res_sqr.sum()
        # 定义一个损失函数 loss，接受 A, x1, x2 作为参数，并返回损失值的平方和

        hess1 = vmap(jacrev(jacrev(loss)))(A, x1, x2)
        hess2 = vmap(hessian(loss))(A, x1, x2)
        # 使用 vmap 计算 loss 函数的二阶导数 Hessian 矩阵，分别使用双向和 Hessian 函数
        self.assertEqual(hess2, hess1)
        # 断言两种计算方法得到的结果相等


@markDynamoStrictTest
class TestJvp(TestCase):
    # 测试类标记为 DynamoStrictTest

    def test_inplace_on_captures(self, device):
        # 测试捕获对象的原地操作

        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        # 创建一个张量 x，值为 [1.0, 2.0, 3.0]，使用指定的设备
        captured = torch.randn(3, device=device)
        # 创建一个随机张量 captured，形状为 (3)，使用指定的设备

        def foo(x):
            captured.copy_(x)
            return (x * captured).sum()
        # 定义一个函数 foo，执行 captured 与 x 的乘积之和，同时将 x 复制给 captured

        with self.assertRaisesRegex(RuntimeError, "mutate a captured Tensor"):
            grad(foo)(x)
        # 使用 grad 函数尝试对 foo 函数求导，预期捕获的张量会引发运行时错误提示


    def test_simple(self, device):
        # 测试简单情况

        x = torch.randn(2, 3, device=device)
        # 创建一个随机张量 x，形状为 (2, 3)，使用指定的设备
        t = torch.randn(2, 3, device=device)
        # 创建一个随机张量 t，形状为 (2, 3)，使用指定的设备
        result = jvp(torch.sin, (x,), (t,))
        # 使用 jvp 函数计算 sin 函数在 x 处的结果与 t 的乘积
        expected = (x.sin(), x.cos() * t)
        # 计算期望结果，包括 sin 函数在 x 处的结果和 x 的 cos 函数结果与 t 的乘积
        self.assertTrue(isinstance(result, tuple))
        # 断言结果为元组
        self.assertEqual(result, expected)
        # 断言计算结果与期望结果相等
    # 测试多个输入情况的函数
    def test_multiple_inputs(self, device):
        # 生成随机张量 x, y, tx, ty，形状为 (2, 3)，在指定设备上
        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)
        ty = torch.randn(2, 3, device=device)

        # 定义一个函数 f，计算两个输入张量的乘积
        def f(x, y):
            return x * y

        # 对函数 f 使用 jvp 函数，计算其在 (x, y) 和 (tx, ty) 处的 Jacobian-Vector Product (JVP)
        result = jvp(f, (x, y), (tx, ty))
        # 预期的输出结果是 (x * y, y * tx + x * ty)
        expected = (x * y, y * tx + x * ty)
        # 断言 result 是一个元组，并且与 expected 相等
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    # 测试 Pytree 输入情况的函数
    def test_pytree_inputs(self, device):
        # 定义一个函数 f，接受三个参数，其中第一个参数是一个元组
        def f(x, y, z):
            a, b = x
            return a + 2 * b + 3 * y + 4 * z

        # 创建 torch.Tensor 类型的标量 one，放在指定设备上
        one = torch.tensor(1.0, device=device)
        # 对函数 f 使用 jvp 函数，计算其在 ((one, one), one, one) 和 ((one, one), one, one) 处的 JVP
        primal_outs, tangent_outs = jvp(
            f, ((one, one), one, one), ((one, one), one, one)
        )
        # 断言 primal_outs 和 tangent_outs 分别等于 one * 10
        self.assertEqual(primal_outs, one * 10)
        self.assertEqual(tangent_outs, one * 10)

    # 测试 Pytree 输入情况下的错误情况
    def test_pytree_inputs_error_cases(self, device):
        # 定义一个简单的函数 f，返回输入值 x
        def f(x):
            return x

        # 创建 torch.Tensor 类型的标量 one，放在指定设备上
        one = torch.tensor(1.0, device=device)

        # 测试异常情况：primals 不是一个元组
        with self.assertRaisesRegex(RuntimeError, "Expected primals to be a tuple"):
            jvp(f, one, one)
        # 测试异常情况：primals 和 tangents 结构不同
        with self.assertRaisesRegex(RuntimeError, "same python structure"):
            jvp(f, ((one, one), one), (one, one))
        # 测试异常情况：primals 中包含非 Tensor 类型
        with self.assertRaisesRegex(RuntimeError, "only contain Tensors"):
            jvp(f, ((one, one), 1), ((one, one), one))
        # 测试异常情况：tangents 中包含非 Tensor 类型
        with self.assertRaisesRegex(RuntimeError, "only contain Tensors"):
            jvp(f, ((one, one), 1), ((1, one), one))
        # 测试异常情况：primals 和 tangents 中至少有一个为空
        with self.assertRaisesRegex(RuntimeError, "at least one Tensor"):
            jvp(f, ((),), ((),))

    # 测试无关输入的函数
    def test_unrelated_input(self, device):
        # 定义一个函数 f，返回第一个输入 x
        def f(x, y):
            return x

        # 生成随机张量 x, y, tx, ty，形状为 (2, 3)，在指定设备上
        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)
        ty = torch.randn(2, 3, device=device)

        # 对函数 f 使用 jvp 函数，计算其在 (x, y) 和 (tx, ty) 处的 JVP
        result = jvp(f, (x, y), (tx, ty))
        # 预期的输出结果是 (x, tx)
        expected = (x, tx)
        # 断言 result 是一个元组，并且与 expected 相等
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    # 测试无关输出的函数
    def test_unrelated_output(self, device):
        # 生成随机张量 y，形状为 (2, 3)，在指定设备上
        y = torch.randn(2, 3, device=device)

        # 定义一个函数 f，返回固定的张量 y
        def f(x):
            return y

        # 生成随机张量 x, tx，形状为 (2, 3)，在指定设备上
        x = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)

        # 对函数 f 使用 jvp 函数，计算其在 (x,) 和 (tx,) 处的 JVP
        result = jvp(f, (x,), (tx,))
        # 预期的输出结果是 (y, zeros_like(y))
        expected = (y, torch.zeros_like(y))
        # 断言 result 是一个元组，并且与 expected 相等
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    # 测试 strict 模式的函数
    def test_strict_mode(self, device):
        # 生成随机张量 y，形状为 (2, 3)，在指定设备上
        y = torch.randn(2, 3, device=device)

        # 定义一个函数 f，返回输入张量 x 和固定张量 y 的元组
        def f(x):
            return x, y

        # 生成随机张量 x, tx，形状为 (2, 3)，在指定设备上
        x = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)

        # 测试在 strict 模式下对函数 f 使用 jvp 函数的行为
        with self.assertRaisesRegex(RuntimeError, "strict"):
            jvp(f, (x,), (tx,), strict=True)
    # 定义一个测试方法，验证多个输出的情况，使用给定的设备
    def test_multiple_outputs(self, device):
        # 创建一个形状为 (2, 3) 的随机张量 x，使用指定设备
        x = torch.randn(2, 3, device=device)
        # 创建一个形状为 (2, 3) 的随机张量 t，使用指定设备
        t = torch.randn(2, 3, device=device)

        # 定义一个函数 f(x)，返回 sin(x) 和 cos(x)
        def f(x):
            return torch.sin(x), torch.cos(x)

        # 调用 jvp 函数，计算 f 在 x 处关于 t 的 Jacobian 向量积
        result = jvp(f, (x,), (t,))
        # 计算预期结果，包括 f(x) 和其对应的导数关于 t 的计算结果
        expected = (f(x), (x.cos() * t, -x.sin() * t))
        # 断言 result 是一个元组
        self.assertTrue(isinstance(result, tuple))
        # 断言 result 等于预期结果 expected
        self.assertEqual(result, expected)

    # 定义一个测试方法，验证多个输入和输出的情况，使用给定的设备
    def test_multiple_inputs_outputs(self, device):
        # 创建形状为 (2, 3) 的随机张量 x 和 y，使用指定设备
        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)
        # 创建形状为 (2, 3) 的随机张量 tx 和 ty，使用指定设备
        tx = torch.randn(2, 3, device=device)
        ty = torch.randn(2, 3, device=device)

        # 定义一个函数 f(x, y)，返回 2*x + 3*y 和 4*x + 5*y
        def f(x, y):
            return 2 * x + 3 * y, 4 * x + 5 * y

        # 调用 jvp 函数，计算 f 在 (x, y) 处关于 (tx, ty) 的 Jacobian 向量积
        result = jvp(f, (x, y), (tx, ty))
        # 计算预期结果，包括 f(x, y) 和 f(tx, ty)
        expected = (f(x, y), f(tx, ty))
        # 断言 result 是一个元组
        self.assertTrue(isinstance(result, tuple))
        # 断言 result 等于预期结果 expected
        self.assertEqual(result, expected)

    # 定义一个测试方法，验证当原始值和导数长度不匹配时的情况，使用给定的设备
    def test_primals_tangents_length_mismatch(self, device):
        # 创建形状为 (2, 3) 的随机张量 x 和 t，使用指定设备
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        # 设置异常信息
        msg = "same python structure"
        # 断言调用 jvp 函数时会引发 RuntimeError 异常，异常信息包含指定消息
        with self.assertRaisesRegex(RuntimeError, msg):
            jvp(torch.sin, (x,), (t, t))
        # 断言调用 jvp 函数时会引发 RuntimeError 异常，异常信息包含指定消息
        with self.assertRaisesRegex(RuntimeError, msg):
            jvp(torch.sin, (x, x), (t, t, t))

    # 定义一个测试方法，验证当原始值和导数为空时的情况，使用给定的设备
    def test_nonempty_primals_and_tangents(self, device):
        # 设置异常信息
        with self.assertRaisesRegex(RuntimeError, "at least one Tensor"):
            # 断言调用 jvp 函数时会引发 RuntimeError 异常，异常信息包含指定消息
            jvp(torch.sin, (), ())

    # 定义一个测试方法，验证输入必须是张量元组的情况，使用给定的设备
    def test_inputs_are_tuples_of_tensors(self, device):
        # 创建形状为 (2, 3) 的随机张量 x 和 t，使用指定设备
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        # 断言调用 jvp 函数时会引发 RuntimeError 异常，异常信息包含指定消息
        with self.assertRaisesRegex(RuntimeError, "be a tuple"):
            jvp(torch.sin, x, (t,))
        # 断言调用 jvp 函数时会引发 RuntimeError 异常，异常信息包含指定消息
        with self.assertRaisesRegex(RuntimeError, "same python structure"):
            jvp(torch.sin, (x,), t)
        # 断言调用 jvp 函数时会引发 RuntimeError 异常，异常信息包含指定消息
        with self.assertRaisesRegex(RuntimeError, "same python structure"):
            jvp(torch.sin, (x,), [t])
        # 断言调用 jvp 函数时会引发 RuntimeError 异常，异常信息包含指定消息
        with self.assertRaisesRegex(RuntimeError, "only contain Tensors"):
            jvp(torch.sin, (1.0,), (t,))
        # 断言调用 jvp 函数时会引发 RuntimeError 异常，异常信息包含指定消息
        with self.assertRaisesRegex(RuntimeError, "only contain Tensors"):
            jvp(torch.sin, (x,), (1.0,))
    # 测试函数，验证输出可以是任何 PyTree 结构
    def test_outputs_can_any_pytree(self, device):
        # 创建随机张量 x 和 t，指定设备
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        # 对于空输出或 None 的情况，验证是否触发 RuntimeError 异常
        for output in [None, ()]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"jvp\(f, primals, tangents\): Expected f to be a function that has non-empty output",
            ):
                jvp(lambda _: output, (x,), (t,))

        # 对于输出不是张量的情况（如整数、布尔值、浮点数、字符串），验证是否触发 RuntimeError 异常
        for output in [1, True, 12.2, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError,
                r"jvp\(f, primals, tangents\): expected f\(\*primals\) to return only tensors",
            ):
                jvp(lambda _: output, (x,), (t,))

        # 验证列表输出的结构
        out = jvp(lambda x: [x, x.sum()], (x,), (t,))
        for i in range(2):
            assert isinstance(out[i], list) and len(out[i]) == 2

        # 验证字典输出的结构
        out = jvp(lambda x: {"x": x, "xsum": x.sum()}, (x,), (t,))
        for i in range(2):
            assert isinstance(out[i], dict) and len(out[i]) == 2 and "xsum" in out[i]

        # 验证复合输出结构的函数
        def composite_output(x):
            out = x.sum()
            return [
                (out, {"a": x, "out": [x, out]}),
            ]

        out = jvp(composite_output, (x,), (t,))
        for i in range(2):
            assert isinstance(out[i], list)
            assert isinstance(out[i][0], tuple) and isinstance(out[i][0][1], dict)

    # 测试辅助张量的函数
    def test_aux_tensor(self, device):
        # 创建随机张量 x 和 t，指定设备
        x = torch.randn(3, device=device)
        t = torch.randn(3, device=device)

        # 对于输出不是元组的情况，验证是否触发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            r"jvp\(f, primals, tangents\): output of function f should be a tuple",
        ):
            jvp(lambda t: [t, t], (x,), (t,), has_aux=True)

        # 对于输出元组长度不符合预期的情况，验证是否触发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            r"jvp\(f, primals, tangents\): output of function f should be a tuple",
        ):
            jvp(lambda t: (t, t + 2, t + 3), (x,), (t,), has_aux=True)

        # 验证包含辅助张量的函数的正确性
        def f(z):
            y = z.sin()
            return y, z.cos()

        out, jvp_out, aux = jvp(f, (x,), (t,), has_aux=True)
        # 验证辅助张量的值
        self.assertEqual(aux, x.cos())
        # 验证主输出的值
        self.assertEqual(out, x.sin())
        # 验证 JVP 输出的值
        self.assertEqual(jvp_out, t * x.cos())
    # 定义测试函数 `test_aux_pytree`，接受一个设备参数 `device`
    def test_aux_pytree(self, device):
        # 定义内部函数 `f(x)`，对输入 `x` 进行数学运算
        def f(x):
            # 计算 x 的正弦值
            y = x.sin()
            # 返回一个元组，包含 x 的余弦值和一个字典
            return y, {"a": x.cos(), "b": [x.tan()]}

        # 生成一个随机张量 x，使用给定的设备
        x = torch.randn(3, device=device)
        # 生成一个随机张量 t，使用给定的设备
        t = torch.randn(3, device=device)

        # 调用 `jvp` 函数，计算函数 `f` 在输入 x 和 t 上的 JVP（雅可比向量积）
        out, jvp_out, aux = jvp(f, (x,), (t,), has_aux=True)
        # 计算函数 `f` 在输入 x 上的预期输出
        expected_out, expected_aux = f(x)
        # 断言输出值 `out` 与预期输出值 `expected_out` 相等
        self.assertEqual(out, expected_out)
        # 断言辅助值 `aux` 与预期辅助值 `expected_aux` 相等
        self.assertEqual(aux, expected_aux)
        # 断言雅可比向量积的输出 `jvp_out` 与预期值 `t * x.cos()` 相等
        self.assertEqual(jvp_out, t * x.cos())

        # 针对不同类型的辅助值进行断言，应该引发运行时错误并包含特定的错误信息
        for aux in [1, 1.0, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                # 调用 `jvp` 函数，使用一个 lambda 函数作为输入函数，期望引发异常
                _ = jvp(lambda x: (x, aux), (x,), (t,), has_aux=True)
            with self.assertRaisesRegex(
                RuntimeError, r"Expected tensors, got unsupported type"
            ):
                # 调用 `jvp` 函数，使用一个 lambda 函数返回包含不支持类型的列表，期望引发异常
                _ = jvp(lambda x: (x, [x, aux]), (x,), (t,), has_aux=True)

    # 定义测试函数 `test_autograd_function_disables_fwd_grad`，接受一个设备参数 `device`
    def test_autograd_function_disables_fwd_grad(self, device):
        # 用于验证的自定义 Torch 自动求导函数 `MySquare`
        class MySquare(torch.autograd.Function):
            # 静态方法，定义前向传播计算
            @staticmethod
            def forward(ctx, x):
                # 检查前向梯度是否启用
                enabled = fwAD._is_fwd_grad_enabled()
                self.assertFalse(enabled)
                # 返回输入张量 x 的平方
                return x * x

            # 静态方法，定义反向传播计算
            @staticmethod
            def backward(ctx, gx):
                # 反向传播直接返回输入的梯度 gx
                return gx

        # 生成一个随机张量 x，需要梯度计算
        x = torch.randn(3, requires_grad=True)
        # 应用自定义的 `MySquare` 自动求导函数到张量 x 上
        MySquare.apply(x)

    # 定义测试函数 `test_disable_fwd_grad_outside`，接受一个设备参数 `device`
    def test_disable_fwd_grad_outside(self, device):
        # 生成一个随机标量张量 x，使用给定的设备
        x = torch.randn([], device=device)
        # 生成一个和 x 形状相同的全 1 张量 t
        t = torch.ones_like(x)
        
        # 在禁用前向梯度的上下文中执行代码块
        with fwAD._set_fwd_grad_enabled(False):
            # 计算 sin 函数在 x 上的 JVP，忽略返回的 out，只关注 y
            _, y = jvp(torch.sin, (x,), (t,))
        # 断言 y 的值等于 x 的余弦值
        self.assertEqual(y, x.cos())

    # 定义测试函数 `test_disable_fwd_grad_inside`，接受一个设备参数 `device`
    def test_disable_fwd_grad_inside(self, device):
        # 定义函数 `f(x)`，在禁用前向梯度的上下文中计算
        def f(x):
            with fwAD._set_fwd_grad_enabled(False):
                shift = x**2
            return x**2 - shift

        # 生成一个随机标量张量 x，使用给定的设备
        x = torch.randn([], device=device)
        # 生成一个和 x 形状相同的全 1 张量 t
        t = torch.ones_like(x)
        
        # 计算函数 `f` 在输入 x 上的 JVP，忽略返回的 out，只关注 y
        _, y = jvp(f, (x,), (t,))
        # 断言 y 的值等于 2 * x
        self.assertEqual(y, 2 * x)
        
        # 对于嵌套调用，计算函数 `f` 在输入 x 上的 JVP，并只返回其辅助部分
        _, y = jvp(lambda x: jvp(f, (x,), (t,))[1], (x,), (t,))
        # 断言 y 的值等于 2
        self.assertEqual(y, 2)

    # 定义测试函数 `test_disable_fwd_grad_mixed`，接受一个设备参数 `device`
    def test_disable_fwd_grad_mixed(self, device):
        # 定义函数 `f(x)`，在禁用前向梯度的上下文中计算
        def f(x):
            with fwAD._set_fwd_grad_enabled(False):
                shift = x**2
            return x**2 - shift

        # 生成一个随机标量张量 x，使用给定的设备
        x = torch.randn([], device=device)
        # 生成一个和 x 形状相同的全 1 张量 t
        t = torch.ones_like(x)
        
        # 在启用前向梯度的上下文中执行代码块
        with fwAD._set_fwd_grad_enabled(True):
            # 计算函数 `f` 在输入 x 上的 JVP，忽略返回的 out，只关注 y
            _, y = jvp(f, (x,), (t,))
        
        # 断言 y 的值等于 2 * x
        self.assertEqual(y, 2 * x)
    # 定义一个测试方法，用于测试自动求导函数内的 JVP（雅可比向量积）操作
    def test_jvp_inside_autograd_function(self, device):
        # 定义一个自定义的 PyTorch 自动求导函数 MySin
        class MySin(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，计算负正弦值并返回
            def forward(ctx, x):
                # 创建与 x 相同大小的全为 1 的张量 t
                t = torch.ones_like(x)
                # 使用 jvp 函数计算 torch.cos 在 x 处的 JVP，得到负的 sin(x) 的 JVP
                _, neg_sin_x = jvp(torch.cos, (x,), (t,))
                # 保存 x 到上下文 ctx 中以备后用
                ctx.save_for_backward(x)
                # 返回负的 sin(x)
                return -neg_sin_x

            @staticmethod
            # 反向传播函数，根据输入的梯度 gx 计算导数
            def backward(ctx, gx):
                # 从上下文 ctx 中获取保存的 x
                (x,) = ctx.saved_tensors
                # 创建与 x 相同大小的全为 1 的张量 t
                t = torch.ones_like(x)
                # 使用 jvp 函数计算 torch.sin 在 x 处的 JVP，得到 cos(x) 的 JVP
                _, cos_x = jvp(torch.sin, (x,), (t,))
                # 返回 gx 乘以 cos(x) 的 JVP，作为反向传播的结果
                return gx * cos_x

        # 生成一个随机张量 x，要求其梯度信息
        x = torch.randn([], device=device, requires_grad=True)
        # 使用自定义的 MySin.apply 方法计算 sin(x) 的负值
        y = MySin.apply(x)
        # 断言计算结果 y 等于 x 的正弦值的负值
        self.assertEqual(y, x.sin())

        # 计算 y 对 x 的梯度
        (gx,) = torch.autograd.grad(y, x)
        # 断言计算的梯度 gx 等于 x 的余弦值
        self.assertEqual(gx, x.cos())

    # 定义一个测试方法，用于测试零张量和 vmap/jvp 的交互
    def test_zerotensor_vmapjvp_interaction(self, device):
        # 创建一个形状为 (4, 1) 的全为 1 的张量 dummy
        dummy = torch.ones(4, 1)
        # 创建一个形状为 (4, 2) 的随机张量 x
        x = torch.randn(4, 2)
        # 创建一个形状为 (2) 的随机张量 x_tangent
        x_tangent = torch.randn(2)

        # 定义一个 push_jvp 函数，用于应用 jvp 到 torch.cov 上
        def push_jvp(dummy, x):
            # 调用 jvp 函数计算 torch.cov 在 x 处的 JVP，并返回结果
            result = jvp(torch.cov, (x,), (x_tangent,))
            return result

        # 使用 vmap(vmap(push_jvp, (0, None)))(dummy, x) 执行 vmap 的双重映射
        # 应保证不出现错误
        vmap(vmap(push_jvp, (0, None)))(dummy, x)
# 标记为 DynamoStrict 测试用例类，可能使用某种测试框架的装饰器
@markDynamoStrictTest
# 定义 TestLinearize 类，继承自 TestCase，用于测试 linearize 函数
class TestLinearize(TestCase):

    # 使用装饰器指定测试函数的数据类型为 torch.float
    @dtypes(torch.float)
    # 测试基本的 linearize 功能
    def test_linearize_basic(self, device, dtype):
        # 创建设备上指定形状和数据类型的张量 x_p 和 x_t
        x_p = make_tensor((3, 1), device=device, dtype=dtype)
        x_t = make_tensor((3, 1), device=device, dtype=dtype)

        # 定义一个函数 fn，计算输入张量 x 的余弦值
        def fn(x):
            return x.cos()

        # 调用 linearize 函数，获取实际输出和 jvp 函数
        actual_output, jvp_fn = linearize(fn, x_p)
        # 使用 jvp 函数计算 x_t 的结果
        actual_jvp = jvp_fn(x_t)
        # 使用 jvp 辅助函数计算期望输出和 jvp
        expected_output, expected_jvp = jvp(fn, (x_p,), (x_t,))
        # 断言实际输出与期望输出相等
        self.assertEqual(actual_output, expected_output)
        # 断言实际 jvp 结果与期望 jvp 结果相等
        self.assertEqual(actual_jvp, expected_jvp)

    @dtypes(torch.float)
    # 测试返回多个值的 linearize 功能
    def test_linearize_return(self, device, dtype):
        # 创建设备上指定形状和数据类型的张量 x_p 和 x_t
        x_p = make_tensor((3, 1), device=device, dtype=dtype)
        x_t = make_tensor((3, 1), device=device, dtype=dtype)

        # 定义一个函数 fn，返回输入张量 x 的余弦值和求和结果
        def fn(x):
            return (x.cos(), x.sum())

        # 调用 linearize 函数，获取实际输出和 jvp 函数
        actual_output, jvp_fn = linearize(fn, x_p)
        # 使用 jvp 函数计算 x_t 的结果
        actual_jvp = jvp_fn(x_t)
        # 使用 jvp 辅助函数计算期望输出和 jvp
        expected_output, expected_jvp = jvp(fn, (x_p,), (x_t,))
        # 断言实际输出与期望输出相等
        self.assertEqual(actual_output, expected_output)
        # 断言实际 jvp 结果与期望 jvp 结果相等
        self.assertEqual(actual_jvp, expected_jvp)

    @dtypes(torch.float)
    # 测试组合 linearize 函数的功能
    def test_linearize_composition(self, device, dtype):
        # 创建设备上指定形状和数据类型的张量 x_p 和 x_t
        x_p = make_tensor((3, 1), device=device, dtype=dtype)
        x_t = make_tensor((3, 3, 1), device=device, dtype=dtype)

        # 定义一个函数 fn，返回输入张量 x 的余弦值和求和结果
        def fn(x):
            return (x.cos(), x.sum())

        # 调用 linearize 函数，获取不关心的实际输出和 jvp 函数
        _, jvp_fn = linearize(fn, x_p)
        # 使用 vmap 函数对 jvp_fn 进行批处理
        actual_batched_jvp = vmap(jvp_fn)(x_t)

        # 定义一个新的 jvp_fn 函数，计算 fn 的 jvp
        def jvp_fn(x_t):
            return jvp(fn, (x_p,), (x_t,))[1]

        # 使用 vmap 函数计算期望的批处理 jvp
        expected_batched_jvp = vmap(jvp_fn)(x_t)

        # 断言实际批处理 jvp 与期望批处理 jvp 相等
        self.assertEqual(actual_batched_jvp, expected_batched_jvp)

    @dtypes(torch.float)
    # 测试嵌套输入和嵌套输出的 linearize 函数的功能
    def test_linearize_nested_input_nested_output(self, device, dtype):
        # 创建设备上指定形状和数据类型的张量 x_p, x_t, y_p, y_t, z_p, z_t
        x_p = make_tensor((3, 1), device=device, dtype=dtype)
        x_t = make_tensor((3, 1), device=device, dtype=dtype)
        y_p = make_tensor((3, 1), device=device, dtype=dtype)
        y_t = make_tensor((3, 1), device=device, dtype=dtype)
        z_p = make_tensor((3, 1), device=device, dtype=dtype)
        z_t = make_tensor((3, 1), device=device, dtype=dtype)

        # 定义一个函数 fn，接受一个字典 arg，包含键为 'x' 和 'yz' 的张量
        def fn(arg):
            x = arg["x"]
            y = arg["yz"][0]
            z = arg["yz"][1]

            # 返回一个字典，包含键为 'a' 和 'b' 的结果
            return {"a": x.sum(), "b": {"c": y + z, "d": (x * z, y.exp())}}

        # 创建输入和目标字典，用于测试 linearize 函数
        inp_p = {"x": x_p, "yz": (y_p, z_p)}
        inp_t = {"x": x_t, "yz": (y_t, z_t)}
        # 调用 linearize 函数，获取实际输出和 jvp 函数
        actual_output, jvp_fn = linearize(fn, inp_p)
        # 使用 jvp 函数计算 inp_t 的结果
        actual_jvp = jvp_fn(inp_t)

        # 使用 jvp 辅助函数计算期望输出和 jvp
        expected_output, expected_jvp = jvp(fn, (inp_p,), (inp_t,))
        # 断言实际输出与期望输出相等
        self.assertEqual(actual_output, expected_output)
        # 断言实际 jvp 结果与期望 jvp 结果相等
        self.assertEqual(actual_jvp, expected_jvp)

    # 仅在 CUDA 环境下运行的测试用例
    @onlyCUDA
    # 定义一个测试函数，用于测试线性化函数的错误情况
    def test_linearize_errors(self):
        # 定义张量的数据类型为浮点型
        dtype = torch.float
        # 将张量放在 CPU 设备上
        device = torch.device("cpu")
        # 创建一个主张量 x_p 和一个目标张量 x_t，形状为 (3, 1)，在指定设备和数据类型上创建
        x_p = make_tensor((3, 1), device=device, dtype=dtype)
        x_t = make_tensor((3, 1), device=device, dtype=dtype)

        # 定义一个函数 fn，对输入张量求正弦函数
        def fn(x):
            return x.sin()

        # 调用 linearize 函数，返回的第二个返回值 jvp_fn 是一个求偏导数的函数
        _, jvp_fn = linearize(fn, x_p)

        # 使用 assertRaisesRegex 断言，期望在执行 jvp_fn((x_t, x_t)) 时抛出 RuntimeError 异常，并且异常消息包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError, "to have the same argspec as the primals"
        ):
            jvp_fn((x_t, x_t))

        # 使用 assertRaisesRegex 断言，期望在执行 jvp_fn(x_t.unsqueeze(0)) 时抛出 RuntimeError 异常，并且异常消息包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError, "in flattened pytree doesn't match the shape"
        ):
            jvp_fn(x_t.unsqueeze(0))

        # 使用 assertRaisesRegex 断言，期望在执行 jvp_fn(x_t.to(torch.double)) 时抛出 RuntimeError 异常，并且异常消息包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError, "in flattened pytree doesn't match the dtype"
        ):
            jvp_fn(x_t.to(torch.double))

        # 使用 assertRaisesRegex 断言，期望在执行 jvp_fn(x_t.to(torch.device("cuda"))) 时抛出 RuntimeError 异常，并且异常消息包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError, "in flattened pytree doesn't match the device"
        ):
            jvp_fn(x_t.to(torch.device("cuda")))
# 在这里的测试中遵循了 [Forward Grad View/inplace] 中的案例
# https://github.com/pytorch/pytorch/blob/master/torch/csrc/autograd/autograd_meta.cpp#L18-L43
@markDynamoStrictTest
class TestVmapJvpInplaceView(TestCase):
    # Case 1 in [Forward Grad View/inplace]
    def test_all_dual_no_view(self, device):
        B = 2

        # 定义一个函数，用于创建带有 JVP 的函数
        def push_jvp(f):
            def inner(x, xt, y, yt):
                return jvp(f, (x, y), (xt, yt))
            return inner

        # 定义一个函数 f，实现 inplace 操作，并返回 x
        def f(x, y):
            x.copy_(y)
            return x

        # 生成随机数据张量
        x = torch.randn(3, B, device=device)
        xt = torch.randn(3, B, device=device)
        y = torch.randn(3, B, device=device)
        yt = torch.randn(3, B, device=device)

        # 使用 vmap 对 push_jvp(f) 进行向量化映射
        out, out_tangent = vmap(push_jvp(f), in_dims=1)(x, xt, y, yt)

        # 断言输出结果
        self.assertEqual(out, x.movedim(1, 0))
        self.assertEqual(out_tangent, yt.movedim(1, 0))

        # 重新生成数据张量
        x = torch.randn(3, B, device=device)
        xt = torch.randn(3, B, device=device)
        y = torch.randn(3, 3, device=device)[:, 1]
        yt = torch.randn(6, device=device)[::2]

        # 使用 vmap 对 push_jvp(f) 进行向量化映射，使用不同的 in_dims
        out, out_tangent = vmap(push_jvp(f), in_dims=(1, 1, None, None))(x, xt, y, yt)

        # 断言输出结果
        self.assertEqual(out, x.movedim(1, 0))
        self.assertEqual(out_tangent, yt.expand(B, 3))

    # Case 2 in [Forward Grad View/inplace]
    def test_all_dual_base_view_inplace(self, device):
        B = 2

        # 定义一个函数，用于创建带有 JVP 的函数
        def push_jvp(f):
            def inner(x, xt, y, yt):
                return jvp(f, (x, y), (xt, yt))
            return inner

        # 实现带有视图的 inplace 操作，并返回视图和原始 x
        def f(x, y):
            view = x[:, ::2]
            view.copy_(y)
            return view, x

        # 生成原始数据张量的副本
        orig_x = torch.randn(2, 6, B, device=device)
        orig_xt = torch.randn(2, 6, B, device=device)
        x = orig_x.clone()
        xt = orig_xt.clone()
        y = torch.randn(2, B, 3, device=device)
        yt = torch.randn(2, B, 3, device=device)

        # 使用 vmap 对 push_jvp(f) 进行向量化映射，指定 in_dims
        out, out_tangent = vmap(push_jvp(f), in_dims=(2, 2, 1, 1))(x, xt, y, yt)

        # 期望的输出结果
        expected_out = vmap(f, in_dims=(2, 1))(orig_x.clone(), y)

        # 断言输出结果
        self.assertEqual(out[0], expected_out[0])
        self.assertEqual(out[1], expected_out[1])

        self.assertEqual(out_tangent[0], yt.movedim(1, 0))

        expected_x_tangent = orig_xt.movedim(-1, 0).clone()
        expected_x_tangent[:, :, ::2].copy_(yt.movedim(1, 0))

        # 断言输出结果
        self.assertEqual(out_tangent[1], expected_x_tangent)

        expected = orig_x.movedim(2, 0).clone()
        expected[:, :, ::2] = y.movedim(1, 0)

        # 断言输出结果
        self.assertEqual(x.movedim(2, 0), expected)

    # Case 3 in [Forward Grad View/inplace]
    # 定义一个测试函数，测试在给定设备上进行双基推导的所有情况
    def test_all_dual_base_inplace(self, device):
        B = 2

        # 定义一个装饰器函数，用于将某个函数转化为支持 JVP 的函数
        def push_jvp(f):
            def inner(x, xt, y, yt):
                return jvp(f, (x, y), (xt, yt))
            return inner

        # Case 3: 带有视图的情况，从基张量传播到视图
        def f(x, y):
            view = x[0, ::2]  # 创建一个张量视图
            x.copy_(y)  # 将 y 的值复制到 x 中
            return x, view  # 返回修改后的 x 和视图

        # 创建四个张量变量，用于测试
        x = torch.randn(2, B, 6, device=device)
        xt = torch.randn(2, 6, B, device=device)
        y = torch.randn(2, B, 6, device=device)
        yt = torch.randn(2, B, 6, device=device)

        # 使用 vmap 将 push_jvp 应用到 f 上，进行 JVP 变换
        out, out_tangent = vmap(push_jvp(f), in_dims=(1, 2, 1, 1))(x.clone(), xt, y, yt)

        # 期望的输出是将 f 函数应用到 x 和 y 上得到的结果
        expected_out = vmap(f, in_dims=(1, 1))(x.clone(), y)
        self.assertEqual(out[0], expected_out[0])  # 检查第一个输出的正确性
        self.assertEqual(out[1], expected_out[1])  # 检查第二个输出的正确性

        # 检查 out_tangent 的正确性，这里使用了 torch 的 movedim 方法来移动维度
        self.assertEqual(out_tangent[0], yt.movedim(1, 0))  # 检查第一个 out_tangent 的正确性
        self.assertEqual(out_tangent[1], yt.movedim(1, 0)[:, 0, ::2])  # 检查第二个 out_tangent 的正确性

    # Case 4 in [Forward Grad View/inplace]
    # 测试右偶视图传播
    def test_right_dual_view_prop(self, device):
        B = 2

        # 定义一个函数 f，x 是普通张量，y 是双基张量，测试视图上的更改是否传播到其基张量
        def f(x, y):
            x = x.clone()  # 克隆输入张量 x
            view = x[0]  # 创建一个视图 view
            view.copy_(y)  # 将 y 的值复制到 view 中
            return view, x  # 返回修改后的 view 和 x

        # 定义一个推导函数 push_jvp，用于对 f 进行 JVP 变换
        def push_jvp(x, y, yt):
            return jvp(partial(f, x), (y,), (yt,))

        # 创建三个张量变量，用于测试
        x = torch.randn(2, B, 6, device=device)
        y = torch.randn(6, B, device=device)
        yt = torch.randn(6, B, device=device)

        # 使用 vmap 将 push_jvp 应用到 x, y, yt 上，进行 JVP 变换
        outs, tangents = vmap(push_jvp, in_dims=(1, 1, 1))(x, y, yt)

        # 期望的输出是将 f 函数应用到 x 和 y 上得到的结果
        expected_out = vmap(f, in_dims=(1, 1))(x.clone(), y)
        self.assertEqual(outs[0], expected_out[0])  # 检查第一个输出的正确性
        self.assertEqual(outs[1], expected_out[1])  # 检查第二个输出的正确性

        # 检查 tangents 的正确性，这里使用了 torch 的 movedim 方法来移动维度
        self.assertEqual(tangents[0], yt.movedim(1, 0))  # 检查第一个 tangent 的正确性

        # 生成预期的 tangents[1]，这里使用了 torch 的 zeros_like 方法和 movedim 方法
        expected_tangent_1 = torch.zeros_like(x).movedim(1, 0)
        expected_tangent_1[:, 0].copy_(yt.movedim(1, 0))
        self.assertEqual(tangents[1], expected_tangent_1)  # 检查第二个 tangent 的正确性

    # Case 5 in [Forward Grad View/inplace]
    # 测试右偶基传播
    def test_right_dual_base_prop(self, device):
        B = 2

        # 定义一个函数 f，x 是普通张量，y 是双基张量，测试基张量上的更改是否传播到其所有视图
        def f(x, y):
            x = x.clone()  # 克隆输入张量 x
            view = x[0]  # 创建一个视图 view
            x.copy_(y)  # 将 y 的值复制到 x 中
            return view, x  # 返回修改后的 view 和 x

        # 定义一个推导函数 push_jvp，用于对 f 进行 JVP 变换
        def push_jvp(x, y, yt):
            return jvp(partial(f, x), (y,), (yt,))

        # 创建三个张量变量，用于测试
        x = torch.randn(2, B, 6)
        y = torch.randn(2, 6, B)
        yt = torch.randn(2, 6, B)

        # 使用 vmap 将 push_jvp 应用到 x, y, yt 上，进行 JVP 变换
        outs, tangents = vmap(push_jvp, in_dims=(1, 2, 2))(x, y, yt)

        # 期望的输出是将 f 函数应用到 x 和 y 上得到的结果
        expected_out = vmap(f, in_dims=(1, 2))(x, y)
        self.assertEqual(outs[0], expected_out[0])  # 检查第一个输出的正确性
        self.assertEqual(outs[1], expected_out[1])  # 检查第二个输出的正确性

        # 检查 tangents 的正确性，这里使用了 torch 的 movedim 方法来移动维度
        self.assertEqual(tangents[0], yt.movedim(2, 0)[:, 0])  # 检查第一个 tangent 的正确性
        self.assertEqual(tangents[1], yt.movedim(2, 0))  # 检查第二个 tangent 的正确性
# 用于测试各种辅助函数的情况
@markDynamoStrictTest
class TestHelpers(TestCase):
    def test_CtxWithSavedTensors_error_if_name_collision(self, device):
        # 创建两个随机张量 x 和 y，设备为指定设备，并且需要计算梯度
        x = torch.randn([], device=device, requires_grad=True)
        y = torch.randn([], device=device, requires_grad=True)

        # 定义自定义的 PyTorch 自动求导函数 A
        class A(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 在上下文中保存一个内部变量 _pt_inner_ctx，并返回输入张量 x
                ctx._pt_inner_ctx = 1
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gy):
                # 使用 CtxWithSavedTensors 包装上下文 ctx 和张量 y
                wrapped = torch._functorch.autograd_function.CtxWithSavedTensors(
                    ctx, (y,)
                )
                return gy

        # 定义自定义的 PyTorch 自动求导函数 B
        class B(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 在上下文中保存一个新的张量 _pt_new_saved_tensors，并返回输入张量 x
                ctx._pt_new_saved_tensors = 1
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gy):
                # 使用 CtxWithSavedTensors 包装上下文 ctx 和张量 y
                wrapped = torch._functorch.autograd_function.CtxWithSavedTensors(
                    ctx, (y,)
                )
                return gy

        # 使用自定义函数 A 对张量 x 进行操作
        out = A.apply(x)
        # 验证在反向传播时是否抛出 RuntimeError，提示名称冲突
        with self.assertRaisesRegex(RuntimeError, "name collision"):
            out.backward()

        # 使用自定义函数 B 对张量 x 进行操作
        out = B.apply(x)
        # 验证在反向传播时是否抛出 RuntimeError，提示名称冲突
        with self.assertRaisesRegex(RuntimeError, "name collision"):
            out.backward()

    def test_CtxWithSavedTensors_nesting(self, device):
        # 获取 CtxWithSavedTensors 类的引用
        CtxWithSavedTensors = torch._functorch.autograd_function.CtxWithSavedTensors
        # 创建三个随机张量 x, y 和 z，设备为指定设备
        x = torch.randn([], device=device, requires_grad=True)
        y = torch.randn([], device=device)
        z = torch.randn([], device=device)

        # 定义自定义的 PyTorch 自动求导函数 A
        class A(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 在上下文中保存输入张量 x，并返回该张量
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gy):
                # 使用 CtxWithSavedTensors 包装上下文 ctx 和张量 y
                ctx_y = CtxWithSavedTensors(ctx, (y,))
                # 使用断言验证 wrapped.saved_tensors 中张量的数量为 1
                assert len(ctx_y.saved_tensors) == 1
                # 使用断言验证保存的张量与 y 的值近似相等
                assert torch.allclose(ctx_y.saved_tensors[0], y)

                # 使用 CtxWithSavedTensors 包装 ctx_y 和张量 z
                wrapped = CtxWithSavedTensors(ctx_y, (z,))
                # 使用断言验证 wrapped.saved_tensors 中张量的数量为 1
                assert len(wrapped.saved_tensors) == 1
                # 使用断言验证保存的张量与 z 的值近似相等
                assert torch.allclose(wrapped.saved_tensors[0], z)

                # 再次使用断言验证 ctx_y.saved_tensors 中张量的数量为 1
                assert len(ctx_y.saved_tensors) == 1
                # 再次使用断言验证保存的张量与 y 的值近似相等
                assert torch.allclose(ctx_y.saved_tensors[0], y)

                # 返回梯度 gy 乘以 wrapped.saved_tensors 中的值
                return gy * wrapped.saved_tensors[0]

        # 使用自定义函数 A 对张量 x 进行操作
        out = A.apply(x)
        # 对结果进行反向传播
        out.backward()
        # 使用断言验证计算得到的梯度与张量 z 的梯度值相等
        self.assertEqual(x.grad, z)
    # 定义一个测试函数，测试在具有保存张量上下文的情况下，是否能正确覆盖保存的张量

    def test_CtxWithSavedTensors_overrides_saved_tensors(self, device):
        # 创建一个在指定设备上的随机张量，要求计算梯度
        x = torch.randn([], device=device, requires_grad=True)

        # 定义一个继承自torch.autograd.Function的类A
        class A(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 在前向传播中保存张量x到上下文对象ctx中
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gy):
                # 覆盖可以是任意值
                override = (1, 2, 3)
                # 使用CtxWithSavedTensors包装上下文ctx和覆盖值override
                wrapped = torch._functorch.autograd_function.CtxWithSavedTensors(
                    ctx, override
                )
                # 断言包装后的saved_tensors与override相同
                assert wrapped.saved_tensors == override
                return gy

        # 对函数A进行调用
        out = A.apply(x)
        # 反向传播
        out.backward()

    # 定义另一个测试函数，测试在具有保存张量上下文的情况下，能否正确地传递参数

    def test_CtxWithSavedTensors_passthrough(self, device):
        # 创建两个在指定设备上的随机张量，其中x要求计算梯度，y不需要
        x = torch.randn([], device=device, requires_grad=True)
        y = torch.randn([], device=device)

        # 定义一个继承自torch.autograd.Function的类A
        class A(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                # 在前向传播中保存张量x和y到上下文对象ctx中
                ctx.save_for_backward(x, y)
                # 返回张量x和y的乘积
                return x * y

            @staticmethod
            def backward(ctx, gz):
                # 覆盖可以是任意值
                override = (1, 2, 3)
                # 使用CtxWithSavedTensors包装上下文ctx和覆盖值override
                wrapped = torch._functorch.autograd_function.CtxWithSavedTensors(
                    ctx, override
                )

                # 断言wrapped的needs_input_grad与ctx的相同
                assert wrapped.needs_input_grad[0] == ctx.needs_input_grad[0]
                assert wrapped.needs_input_grad[1] == ctx.needs_input_grad[1]

                # 在wrapped上设置一个新属性foo，并断言ctx上也能访问到该属性
                wrapped.foo = "bar"
                assert wrapped.foo == "bar"
                assert ctx.foo == "bar"

                # 返回梯度gz，并传递相同的梯度给输入张量x和y
                return gz, gz

        # 对函数A进行调用
        out = A.apply(x, y)
        # 反向传播
        out.backward()
    # 定义测试函数 test_reductify_leaf，接受一个设备参数 device
    def test_reductify_leaf(self, device):
        # 从 torch._functorch.autograd_function 中导入 reductify_leaf 函数
        reductify_leaf = torch._functorch.autograd_function.reductify_leaf
        # 设置常量 B 的值为 2
        B = 2

        # 测试 grad_input 为 None 的情况
        output = reductify_leaf(None, None, 0, B)
        # 断言输出为 None
        self.assertIsNone(output)
        output = reductify_leaf(None, None, None, B)
        # 断言输出为 None
        self.assertIsNone(output)

        # 测试 grad_input 具有 bdim，而 input 不具有 bdim 的情况
        grad_input = torch.randn([B, 3, 4], device=device)
        # 调用 reductify_leaf 函数，对 grad_input 在第 0 维度上求和
        output = reductify_leaf(grad_input, 0, None, B)
        # 断言输出与 grad_input 在第 0 维度上求和的结果相等
        self.assertEqual(output, grad_input.sum(0))

        grad_input = torch.randn([3, B, 4], device=device)
        # 调用 reductify_leaf 函数，对 grad_input 在第 1 维度上求和，指定额外的 size 参数 (3,)
        output = reductify_leaf(grad_input, 1, None, B, (3,))
        # 断言输出与 grad_input 在第 1 维度上求和的结果相等
        self.assertEqual(output, grad_input.sum(1))

        # 测试 grad_input 不具有 bdim，而 input 具有 bdim 的情况
        # 这种情况可能发生在用户从反向传播中返回一个与输入无关的新张量
        grad_input = torch.randn([3, 4], device=device)
        # 调用 reductify_leaf 函数，将 grad_input 在第 1 维度上扩展为 B 个副本
        output = reductify_leaf(grad_input, None, 1, B)
        # 断言输出与 grad_input 在第 1 维度上扩展为 B 个副本的结果相等
        self.assertEqual(output, grad_input.view(3, 1, 4).expand(3, B, 4))

        grad_input = torch.randn([3, 4], device=device)
        # 调用 reductify_leaf 函数，将 grad_input 在第 1 维度上扩展为 4 个副本，再在第 2 维度上求和
        output = reductify_leaf(grad_input, None, 1, B, (4,))
        # 断言输出与 grad_input 在第 1 维度上扩展为 4 个副本，再在第 2 维度上求和的结果相等
        self.assertEqual(output, grad_input.view(3, 4, 1).expand(3, 4, B).sum(0))

        # 测试 grad_input 和 input 均具有 bdim 的情况
        grad_input = torch.randn([B, 3, 4], device=device)
        # 调用 reductify_leaf 函数，将 grad_input 的第 0 维度移动到第 1 维度
        output = reductify_leaf(grad_input, 0, 1, B)
        # 断言输出与 grad_input 的第 0 维度移动到第 1 维度的结果相等
        self.assertEqual(output, grad_input.movedim(0, 1))

        grad_input = torch.randn([3, 4, 5, B], device=device)
        # 调用 reductify_leaf 函数，将 grad_input 的第 3 维度移动到第 2 维度，然后在新的第 2 维度上求和，再在第 0 维度上求和
        output = reductify_leaf(grad_input, 3, 0, B, (5,))
        # 断言输出与 grad_input 的第 3 维度移动到第 2 维度，然后在新的第 2 维度上求和，再在第 0 维度上求和的结果相等
        self.assertEqual(output, grad_input.movedim(-1, 2).sum(0).sum(0))
# 标记为 DynamoStrict 测试用例
@markDynamoStrictTest
class TestComposability(TestCase):
    
    # 测试 vmap 函数的弃用警告
    def test_deprecation_vmap(self, device):
        # 生成一个设备上的随机张量
        x = torch.randn(3, device=device)

        # 对 functorch 版本的 API 发出弃用警告
        with self.assertWarnsRegex(FutureWarning, "Please use `torch.vmap`"):
            vmap(torch.sin)

        # 非 functorch 版本的 API 未被弃用
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            torch.vmap(torch.sin)

    # 测试不同变换函数的弃用警告
    @parametrize(
        "transform",
        ["grad", "jacrev", "jacfwd", "grad_and_value", "hessian", "functionalize"],
    )
    def test_deprecation_transforms(self, device, transform):
        # 获取 functorch 和 torch.func 中对应的 API
        api = getattr(functorch, transform)
        new_api = getattr(torch.func, transform)

        # 对 functorch 版本的 API 发出弃用警告
        with self.assertWarnsRegex(
            FutureWarning, f"Please use `torch.func.{transform}`"
        ):
            api(torch.sin)

        # 非 functorch 版本的 API 未被弃用
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            new_api(torch.sin)

    # 测试双重梯度计算
    def test_grad_grad(self, device):
        # 生成一个设备上的随机标量张量
        x = torch.randn([], device=device)
        # 计算 sin 函数的二阶导数
        y = grad(grad(torch.sin))(x)
        self.assertEqual(y, -x.sin())

    # 测试梯度与 vmap 的结合使用
    def test_grad_vmap(self, device):
        # 定义一个函数 foo，应用 vmap 计算 sin 函数的梯度并求和
        def foo(x):
            y = vmap(torch.sin)(x)
            return y.sum()

        # 生成一个设备上的随机向量张量
        x = torch.randn(3, device=device)
        # 计算 foo 函数的梯度
        y = grad(foo)(x)
        self.assertEqual(y, x.cos())

    # 测试 vjp 函数的梯度计算
    def test_grad_vjp(self, device):
        # 生成一个设备上的随机向量张量
        x = torch.randn(3, device=device)

        # 定义一个函数 foo，使用 vjp 计算 sin 函数的梯度并求和
        def foo(x):
            _, vjp_fn = vjp(torch.sin, x)
            return vjp_fn(x)[0].sum()

        # 计算 foo 函数的梯度
        y = grad(foo)(x)
        # 计算预期的梯度
        expected = grad(lambda x: (x * x.cos()).sum())(x)
        self.assertEqual(y, expected)

    # 测试 vmap 与梯度计算的结合使用
    def test_vmap_grad(self, device):
        # 生成一个设备上的随机向量张量
        x = torch.randn(3, device=device)
        # 对 sin 函数应用 vmap 计算其梯度
        y = vmap(grad(torch.sin))(x)
        self.assertEqual(y, x.cos())

    # 测试嵌套 vmap 的使用
    def test_vmap_vmap(self, device):
        # 生成一个设备上的随机矩阵张量
        x = torch.randn(2, 3, device=device)
        # 对 sin 函数进行嵌套的 vmap 操作
        y = vmap(vmap(torch.sin))(x)
        self.assertEqual(y, x.sin())

    # 测试 vmap 与 vjp 的结合使用
    def test_vmap_vjp(self, device):
        # 生成一个设备上的随机向量张量
        x = torch.randn(3, device=device)
        # 计算 sin 函数的 vjp
        _, vjp_fn = vjp(torch.sin, x)

        # 定义一个函数 foo，对 sin 函数的 vjp 应用 vmap
        def foo(x):
            _, vjp_fn = vjp(torch.sin, x)
            return vjp_fn(x)

        # 对 foo 函数应用 vmap
        y = vmap(foo)(x)
        self.assertEqual(y, vjp_fn(x))

        # 在 CPU 上存在一个有趣的错误消息
        xs = torch.randn(5, 3, device=device)
        # 计算预期的结果
        expected = torch.stack([vjp_fn(x)[0] for x in xs])
        # 对 vjp_fn 的嵌套 vmap 操作
        result = vmap(lambda x: vjp_fn(x)[0])(xs)
        self.assertEqual(result, expected)

    # 测试 vjp 与梯度计算的结合使用
    def test_vjp_grad(self, device):
        # 生成一个设备上的随机标量张量
        x = torch.randn([], device=device)
        # 计算 sin 函数的 vjp
        y, vjp_fn = vjp(grad(torch.sin), x)
        self.assertEqual(y, x.cos())

        # 生成一个随机标量张量
        v = torch.randn([])
        # 对 vjp_fn 的应用
        self.assertEqual(vjp_fn(v)[0], -x.sin() * v)
    # 测试函数，用于测试自动微分中的 VJP（反向传播函数）和 VMAP（映射函数）功能
    def test_vjp_vmap(self, device):
        # 创建一个在指定设备上随机初始化的张量 x
        x = torch.randn(3, device=device)
        # 使用 vmap 函数对 torch.sin 函数进行矢量化，并返回结果 y 和反向传播函数 vjp_fn
        y, vjp_fn = vjp(vmap(torch.sin), x)
        # 断言 y 的值等于 x 求正弦函数的结果
        self.assertEqual(y, x.sin())

        # 创建一个与 x 同设备的随机张量 v
        v = torch.randn(3, device=device)
        # 使用 vjp_fn 对 v 进行反向传播，并断言结果与 x 求余弦函数再乘以 v 的结果相等
        self.assertEqual(vjp_fn(v)[0], x.cos() * v)

    # 测试函数，测试自动微分中的双重反向传播函数 VJP 的功能
    def test_vjp_vjp(self, device):
        # 创建一个在指定设备上随机初始化的张量 x
        x = torch.randn(3, device=device)
        # 使用 VJP 函数对 torch.sin 函数进行反向传播，并返回结果 y 和反向传播函数 vjp_fn
        y, vjp_fn = vjp(torch.sin, x)
        # 断言 y 的值等于 x 求正弦函数的结果
        self.assertEqual(y, x.sin())

        # 使用 lambda 函数实现对 vjp_fn 的双重反向传播，并返回结果 y 和反向传播函数 vjp_fn
        y, vjp_fn = vjp(lambda x: vjp_fn(x)[0], x)
        # 断言 y 的值等于 x 乘以 x 的余弦函数的结果
        self.assertEqual(y, x * x.cos())

        # 对 vjp_fn 应用于 x 的反向传播，并未指定具体的断言，但至少能成功运行

    # 测试函数，用于测试 make_fx 函数中对 vmap 的使用
    def test_make_fx_vmap(self, device):
        # 定义一个函数 f，计算输入张量的正弦值
        def f(x):
            return torch.sin(x)

        # 创建一个大小为 5x3 的随机输入张量
        inp = torch.randn(5, 3)
        # 对函数 f 应用 vmap 函数
        f = vmap(f)
        # 使用 make_fx 函数将 f 封装为 fx_f 函数，并对新的输入张量进行计算
        fx_f = make_fx(f)(inp)
        # 创建一个新的大小为 5x3 的随机输入张量，并断言 fx_f 的结果与 f 的结果相等
        new_inp = torch.randn(5, 3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    # 测试函数，用于测试 make_fx 函数中对 JacRev 的使用
    def test_make_fx_jacrev(self, device):
        # 定义一个函数 f，计算输入张量的正弦值的和
        def f(x):
            return x.sin().sum()

        # 创建一个大小为 3 的随机输入张量
        inp = torch.randn(3)
        # 对函数 f 进行两次 JacRev 反向传播计算
        f = jacrev(jacrev(f))
        # 使用 make_fx 函数将 f 封装为 fx_f 函数，并对新的输入张量进行计算
        fx_f = make_fx(f)(inp)
        # 创建一个新的大小为 3 的随机输入张量，并断言 fx_f 的结果与 f 的结果相等
        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    # 测试函数，用于测试 make_fx 函数中对 VJP 的使用
    def test_make_fx_vjp(self, device):
        # 定义一个函数 f，计算输入张量的正弦值的总和
        def f(x):
            return torch.sin(x).sum()

        # 创建一个大小为 3 的随机输入张量 primals
        primals = torch.randn(3)
        # 使用 VJP 函数计算函数 f 的值和对应的反向传播函数 vjp_fn
        _, vjp_fn = vjp(f, primals)
        # 创建一个大小为 () 的随机输入张量 cotangent
        cotangent = torch.randn(())
        # 使用 make_fx 函数将 vjp_fn 封装为 fx_f 函数，并对新的输入张量进行计算
        fx_f = make_fx(vjp_fn)(cotangent, True, True)
        # 创建一个新的大小为 () 的随机输入张量 new_cotangent，并断言 fx_f 的结果与 vjp_fn 的结果相等
        new_cotangent = torch.randn(())
        self.assertEqual(fx_f(new_cotangent, True, True), vjp_fn(new_cotangent))

    # 如果在 FBCODE 中，跳过测试；用于测试 funtortch 导入时不产生警告
    @unittest.skipIf(IS_FBCODE, "can't subprocess in fbcode")
    # 在 CPU 上运行的装饰器，防止在已经有 GPU 的机器上重复运行此测试
    @onlyCPU
    def test_no_warning_on_import_functorch(self, device):
        # 执行子进程命令，检查导入 functorch 是否会产生警告，并断言输出为空字符串
        out = subprocess.check_output(
            [sys.executable, "-W", "all", "-c", "import functorch"],
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.realpath(__file__)),
        ).decode("utf-8")
        self.assertEqual(out, "")

    # 测试函数，用于测试在转换内部要求梯度的情况
    def test_requires_grad_inside_transform(self, device):
        # 定义一个函数 f，对输入张量 x 要求梯度并计算正弦值的和
        def f(x):
            x.requires_grad_()
            return x.sin().sum()

        # 创建一个大小为 3 的随机输入张量 x
        x = torch.randn(3)

        # 使用 assertRaisesRegex 断言，在使用 vmap(f) 时会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Tensor.requires_grad_()"):
            vmap(f)(x)
        # 使用 assertRaisesRegex 断言，在使用 grad(f) 时会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Tensor.requires_grad_()"):
            grad(f)(x)
        # 使用 assertRaisesRegex 断言，在使用 vmap(grad(f)) 时会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Tensor.requires_grad_()"):
            vmap(grad(f))(x)

        # 创建一个大小为 () 的随机输入张量 x
        x = torch.randn([])
        # 使用 assertRaisesRegex 断言，在使用 grad(grad(f)) 时会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Tensor.requires_grad_()"):
            grad(grad(f))(x)

    # 测试函数，用于测试在转换内部要求保留梯度的情况
    def test_retain_grad_inside_transform(self, device):
        # 定义一个函数 f，对输入张量 x 计算正弦值并要求保留梯度
        def f(x):
            y = x.sin()
            y.retain_grad()
            return y.sum()

        # 创建一个大小为 3 的随机输入张量 x
        x = torch.randn(3)

        # 使用 assertRaisesRegex 断言，在使用 grad(f) 时会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Tensor.retain_grad()"):
            grad(f)(x)
    # 定义一个测试方法，用于测试自动微分库中函数在变换中的行为
    def test_autograd_functional_jacrev_inside_transform(self, device):
        # 定义函数 f(x)，计算 x.sin().sum() 的雅可比矩阵
        def f(x):
            y = torch.autograd.functional.jacobian(lambda x: x.sin().sum(), x)
            return y

        # 设定批次大小 B，并生成形状为 (B, 3) 的随机张量 x
        B = 5
        x = torch.randn(B, 3)
        # 使用 vmap 函数对 f(x) 进行批处理，期望抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "torch.autograd.functional"):
            vmap(f)(x)

        # 重新生成一个标量张量 x，并再次使用 grad 函数调用 f(x)，期望抛出 RuntimeError 异常
        x = torch.randn([])
        with self.assertRaisesRegex(RuntimeError, "torch.autograd.functional"):
            grad(f)(x)

    # 定义一个测试方法，用于测试自动微分库中函数在变换中的行为
    def test_autograd_functional_vjp_inside_transform(self, device):
        # 定义函数 f(x)，计算 x.sin().sum() 的 VJP（向量-雅可比积）
        def f(x):
            y = torch.autograd.functional.vjp(lambda x: x.sin().sum(), x)
            return y

        # 设定批次大小 B，并生成形状为 (B, 3) 的随机张量 x
        B = 5
        x = torch.randn(B, 3)
        # 使用 vmap 函数对 f(x) 进行批处理，期望抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "torch.autograd.functional"):
            vmap(f)(x)

        # 重新生成一个标量张量 x，并再次使用 grad 函数调用 f(x)，期望抛出 RuntimeError 异常
        x = torch.randn([])
        with self.assertRaisesRegex(RuntimeError, "torch.autograd.functional"):
            grad(f)(x)

    # 定义一个测试方法，用于测试自动微分库中函数在变换中的行为
    def test_autograd_functional_jvp_inside_transform(self, device):
        # 定义函数 f(x)，计算 x.sin().sum() 的 JVP（雅可比向量积）
        def f(x):
            t = torch.ones_like(x)
            y = torch.autograd.functional.jvp(lambda x: x.sin().sum(), (x,), (t,))
            return y

        # 设定批次大小 B，并生成形状为 (B, 3) 的随机张量 x
        B = 5
        x = torch.randn(B, 3)
        # 使用 vmap 函数对 f(x) 进行批处理，期望抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "torch.autograd.functional"):
            vmap(f)(x)

        # 重新生成一个标量张量 x，并再次使用 grad 函数调用 f(x)，期望抛出 RuntimeError 异常
        x = torch.randn([])
        with self.assertRaisesRegex(RuntimeError, "torch.autograd.functional"):
            grad(f)(x)

    # 定义一个测试方法，用于测试自动微分库中函数在变换中的行为
    def test_autograd_functional_jacfwd_inside_transform(self, device):
        # 定义函数 f(x)，使用前向模式计算 x.sin().sum() 的雅可比矩阵
        def f(x):
            y = torch.autograd.functional.jacobian(
                lambda x: x.sin().sum(), x, strategy="forward-mode", vectorize=True
            )
            return y

        # 设定批次大小 B，并生成形状为 (B, 3) 的随机张量 x
        B = 5
        x = torch.randn(B, 3)
        # 使用 vmap 函数对 f(x) 进行批处理，期望抛出特定错误信息的 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError, "Batching rule not implemented for aten::_make_dual"
        ):
            vmap(f)(x)

    # 使用 parametrize 装饰器进行参数化测试，测试自动微分库中不同函数在变换中的行为
    @parametrize(
        "transform",
        [
            "vmap",
            "grad",
            "jacrev",
            "jacfwd",
            "grad_and_value",
            "hessian",
            "functionalize",
        ],
    )
    # 定义一个测试方法，用于测试自动微分库中函数在无设置上下文情况下的行为
    def test_autograd_function_no_setup_context(self, device, transform):
        # 定义一个自定义的 torch.autograd.Function 子类 MySin，实现 sin 函数的自动求导
        class MySin(torch.autograd.Function):
            # 前向传播函数，保存输入张量，并返回其 sin 值
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.sin()

            # 反向传播函数，计算输入梯度
            @staticmethod
            def backward(ctx, gy):
                (x,) = ctx.saved_tensors
                return gy * x.cos()

        # 生成一个随机张量 x，并根据 transform 参数获取对应的函数进行变换
        x = torch.randn(3, device=device)
        transform = getattr(functorch, transform)
        # 使用 transform 对 MySin.apply(x) 进行变换，期望抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "must override the setup_context"):
            transform(MySin.apply)(x)

    # 一些测试通过，一些测试不通过
    @parametrize(
        "transform",
        [  # 参数化测试函数，测试不同的变换方法
            "vmap",  # 使用vmap变换
            "grad",  # 计算梯度
            "jacrev",  # 使用反向模式的雅可比矩阵向量积
            "jacfwd",  # 使用前向模式的雅可比矩阵向量积
            "grad_and_value",  # 计算梯度和函数值
            "hessian",  # 计算Hessian矩阵
            "functionalize",  # 将函数功能化
        ],
    )
    def test_transforms_dont_support_saved_tensor_hooks(self, device, transform):
        # 定义函数f，计算输入张量的正弦值之和
        def f(x):
            return torch.sin(x).sum()

        # 定义函数g，使用torch.autograd.graph.save_on_cpu()保存张量钩子
        def g(x):
            with torch.autograd.graph.save_on_cpu():
                return f(x)

        # 生成一个随机张量x，用于测试
        x = torch.randn(3, device=device)

        # 根据参数transform选择对应的变换函数
        if transform == "functionalize":
            transform = functorch.experimental.functionalize
        else:
            transform = getattr(functorch, transform)

        # 断言调用transform(f)(x)时会抛出RuntimeError异常，异常信息包含"saved tensor hooks"
        with self.assertRaisesRegex(RuntimeError, "saved tensor hooks"):
            with torch.autograd.graph.save_on_cpu():
                transform(f)(x)

        # 断言调用transform(g)(x)时会抛出RuntimeError异常，异常信息包含"saved tensor hooks"
        with self.assertRaisesRegex(RuntimeError, "saved tensor hooks"):
            transform(g)(x)

    # 测试函数，验证vjp函数不支持保存的张量钩子
    def test_vjp_doesnt_support_saved_tensor_hooks(self, device):
        # 定义函数f，计算输入张量的正弦值之和
        def f(x):
            return torch.sin(x).sum()

        # 定义函数g，使用torch.autograd.graph.save_on_cpu()保存张量钩子
        def g(x):
            with torch.autograd.graph.save_on_cpu():
                return f(x)

        # 生成一个随机张量x，用于测试
        x = torch.randn(3, device=device)

        # 断言调用vjp(f, x)时会抛出RuntimeError异常，异常信息包含"saved tensor hooks"
        with self.assertRaisesRegex(RuntimeError, "saved tensor hooks"):
            with torch.autograd.graph.save_on_cpu():
                vjp(f, x)

        # 断言调用vjp(g, x)时会抛出RuntimeError异常，异常信息包含"saved tensor hooks"
        with self.assertRaisesRegex(RuntimeError, "saved tensor hooks"):
            vjp(g, x)

    # 测试函数，验证jvp函数不支持保存的张量钩子
    def test_jvp_doesnt_support_saved_tensor_hooks(self, device):
        # 定义函数f，计算输入张量的正弦值之和
        def f(x):
            return torch.sin(x).sum()

        # 定义函数g，使用torch.autograd.graph.save_on_cpu()保存张量钩子
        def g(x):
            with torch.autograd.graph.save_on_cpu():
                return f(x)

        # 生成一个随机张量x和t，用于测试
        x = torch.randn(3, device=device)
        t = torch.randn(3, device=device)

        # 断言调用jvp(f, (x,), (t,))时会抛出RuntimeError异常，异常信息包含"saved tensor hooks"
        with self.assertRaisesRegex(RuntimeError, "saved tensor hooks"):
            with torch.autograd.graph.save_on_cpu():
                jvp(f, (x,), (t,))

        # 断言调用jvp(g, (x,), (t,))时会抛出RuntimeError异常，异常信息包含"saved tensor hooks"
        with self.assertRaisesRegex(RuntimeError, "saved tensor hooks"):
            jvp(g, (x,), (t,))

    # 测试函数，验证当排除特定的调度键时，可以使用functionalize
    def test_can_use_functionalize_when_key_is_excluded(self, device):
        # 定义函数f，对输入张量进行sin操作，并返回结果
        def f(x):
            y = x.clone()
            y.sin_()
            return y

        # 生成一个随机张量x，用于测试
        x = torch.randn([], device=device)
        expected = f(x)

        # 使用_ExcludeDispatchKeyGuard排除DispatchKey.Functionalize键的调度
        with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
            # 使用make_fx和functionalize(f)，构建GraphModule
            gm = make_fx(functorch.functionalize(f))(x)
            # 断言"sin_"不在生成的GraphModule的代码中
            self.assertTrue("sin_" not in gm.code)
            # 断言gm(x)的计算结果与预期的expected一致
            self.assertEqual(gm(x), expected)

            # 获取本地的排除集合，验证其中包含DispatchKey.Functionalize键
            local_exclude_set = torch._C._dispatch_tls_local_exclude_set()
            self.assertTrue(local_exclude_set.has(DispatchKey.Functionalize))
    # 定义测试函数，验证在排除特定调度键时是否可以使用 vmap 函数
    def test_can_use_vmap_when_key_is_excluded(self, device):
        # 定义一个函数 f，对输入张量 x 按第一个维度求和
        def f(x):
            return x.sum(0)

        # 创建一个指定设备上的随机张量 x
        x = torch.randn(3, device=device)
        # 使用 vmap 函数对函数 f 进行向量化映射，得到期望结果
        expected = vmap(f)(x)

        # 使用 _ExcludeDispatchKeyGuard 上下文管理器，排除 FuncTorchBatched 调度键
        with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.FuncTorchBatched)):
            # 使用 vmap 函数对函数 f 进行向量化映射，得到结果
            result = vmap(f)(x)
            # 断言结果与期望值相等
            self.assertEqual(result, expected)
            # 获取当前线程局部的调度键排除集合
            local_exclude_set = torch._C._dispatch_tls_local_exclude_set()
            # 断言 FuncTorchBatched 调度键在局部排除集合中
            self.assertTrue(local_exclude_set.has(DispatchKey.FuncTorchBatched))

    # 定义测试函数，验证在排除特定调度键时是否可以使用 grad 函数
    def test_can_use_grad_when_key_is_excluded(self, device):
        # 定义一个函数 f，对输入张量 x 求正弦函数
        def f(x):
            return x.sin()

        # 创建一个指定设备上的随机张量 x，空维度
        x = torch.randn([], device=device)
        # 使用 grad 函数计算函数 f 在 x 处的梯度，得到期望结果
        expected = grad(f)(x)

        # 使用 _ExcludeDispatchKeyGuard 上下文管理器，排除 Autograd 调度键
        with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Autograd)):
            # 使用 grad 函数计算函数 f 在 x 处的梯度，得到结果
            result = grad(f)(x)
            # 断言结果与期望值相等
            self.assertEqual(result, expected)
            # 获取当前线程局部的调度键排除集合
            local_exclude_set = torch._C._dispatch_tls_local_exclude_set()
            # 断言 Autograd 调度键在局部排除集合中
            self.assertTrue(local_exclude_set.has(DispatchKey.Autograd))
# 使用装饰器标记为 DynamoStrictTest 的测试类
@markDynamoStrictTest
class TestMakeFunctional(TestCase):

    # 参数化测试函数，测试禁用自动梯度追踪
    @parametrize("disable_autograd_tracking", [True, False])
    def test_disable_autograd_tracking(self, disable_autograd_tracking):
        
        # 定义一个简单的神经网络模型 Foo
        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 3)

            # 前向传播函数
            def forward(self, x):
                x = self.linear(x)
                return x

        # 创建 Foo 类的实例 mod
        mod = Foo()
        
        # 调用 make_functional 函数，获取返回的模型函数及其参数
        _, params = make_functional(
            mod, disable_autograd_tracking=disable_autograd_tracking
        )
        
        # 断言返回的参数列表长度为 2
        self.assertEqual(len(params), 2)
        
        # 遍历参数列表，检查每个参数的梯度追踪属性与 disable_autograd_tracking 的关系
        for param in params:
            self.assertEqual(param.requires_grad, not disable_autograd_tracking)

    # 测试参数绑定的功能
    def test_parameter_tying(self):
        
        # 定义一个包含参数绑定的神经网络模型 Foo
        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.randn(3))
                self.linear = nn.Linear(3, 3)
                self.linear.bias = self.bias
                self.linear_tied = self.linear

            # 前向传播函数
            def forward(self, x):
                x = self.linear(x)
                x = self.linear_tied(x)
                x = x + self.bias
                return x

        # 设置随机种子
        torch.manual_seed(1)
        
        # 创建 Foo 类的实例 mod
        mod = Foo()
        
        # 调用 make_functional 函数，获取返回的模型函数及其参数
        func, _ = make_functional(mod)

        # 重置随机种子
        torch.manual_seed(0)
        
        # 再次创建 Foo 类的实例 mod
        mod = Foo()
        
        # 调用 make_functional 函数，获取返回的模型函数及其参数
        _, params = make_functional(mod)
        
        # 断言返回的参数列表长度为 2
        self.assertEqual(len(params), 2)

        # 创建输入张量 x
        x = torch.randn(2, 3)
        
        # 使用返回的函数 func 计算结果
        result = func(params, x)
        
        # 使用原始模型计算期望结果
        expected = mod(x)
        
        # 断言结果相等
        self.assertEqual(result, expected)

    # 测试缓冲区绑定的功能
    def test_buffer_tying(self):
        
        # 定义一个包含缓冲区绑定的神经网络模型 Foo
        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.randn(3))
                self.linear = nn.Linear(3, 3)
                self.register_buffer("buffer", torch.randn(3))
                self.register_buffer("buffer_tied", self.buffer)

            # 前向传播函数
            def forward(self, x):
                x = self.linear(x)
                x = x + self.bias
                x = x + self.buffer
                x = x + self.buffer_tied
                return x

        # 设置随机种子
        torch.manual_seed(1)
        
        # 创建 Foo 类的实例 mod
        mod = Foo()
        
        # 调用 make_functional_with_buffers 函数，获取返回的模型函数、参数和缓冲区
        func, _, _ = make_functional_with_buffers(mod)

        # 重置随机种子
        torch.manual_seed(0)
        
        # 再次创建 Foo 类的实例 mod
        mod = Foo()
        
        # 调用 make_functional_with_buffers 函数，获取返回的模型函数、参数和缓冲区
        _, params, buffers = make_functional_with_buffers(mod)
        
        # 断言返回的参数列表长度为 3，缓冲区列表长度为 1
        self.assertEqual(len(params), 3)
        self.assertEqual(len(buffers), 1)

        # 创建输入张量 x
        x = torch.randn(2, 3)
        
        # 使用返回的函数 func 计算结果
        result = func(params, buffers, x)
        
        # 使用原始模型计算期望结果
        expected = mod(x)
        
        # 断言结果相等
        self.assertEqual(result, expected)
    def test_with_buffers_disable_autograd_tracking(self, disable_autograd_tracking):
        # 定义一个内部的 nn.Module 类 Foo，用于测试
        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个线性层，输入维度为3，输出维度为3
                self.linear = nn.Linear(3, 3)
                # 注册一个缓冲区，内容为形状为 (3,) 的随机张量
                self.register_buffer("buffer", torch.randn(3))

            def forward(self, x):
                # 前向传播函数
                x = self.linear(x)
                # 加上注册的缓冲区内容
                x = x + self.buffer
                return x

        # 创建 Foo 的实例
        mod = Foo()
        # 调用 make_functional_with_buffers 函数，获取模型的函数化版本以及参数和缓冲区
        _, params, buffers = make_functional_with_buffers(
            mod, disable_autograd_tracking=disable_autograd_tracking
        )
        # 断言参数的数量为2
        self.assertEqual(len(params), 2)
        # 断言缓冲区的数量为1
        self.assertEqual(len(buffers), 1)
        # 遍历参数列表，检查参数的 requires_grad 属性是否符合预期
        for param in params:
            self.assertEqual(param.requires_grad, not disable_autograd_tracking)

    @parametrize("detach_params", [True, False])
    def test_using_detach_functional_call(self, detach_params):
        # 定义一个内部的 nn.Module 类 Foo，用于测试
        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个线性层，输入维度为3，输出维度为3
                self.linear = nn.Linear(3, 3)
                # 注册一个缓冲区，内容为形状为 (3,) 的随机张量
                self.register_buffer("buffer", torch.randn(3))

            def forward(self, x):
                # 前向传播函数
                x = self.linear(x)
                # 加上注册的缓冲区内容
                x = x + self.buffer
                return x

        # 定义一个函数 params_dict，用于获取模型参数字典
        def params_dict(mod):
            named_params = mod.named_parameters()
            return (
                {k: v.detach() for k, v in named_params}
                if detach_params
                else dict(named_params)
            )

        # 创建 Foo 的实例
        mod = Foo()
        # 生成一个形状为 (3, 3) 的随机张量
        x = torch.randn(3, 3)
        # 获取参数字典和缓冲区字典
        d = (params_dict(mod), dict(mod.named_buffers()))
        # 调用 functional_call 函数，进行函数化调用
        out = functional_call(mod, d, x)
        # 断言输出的梯度函数是否为 None，根据 detach_params 的值来确定
        self.assertEqual(out.grad_fn is None, detach_params)

    def test_parameter_tying_grad(self):
        # 定义一个内部的 nn.Module 类 Foo，用于测试
        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个线性层，输入维度为3，输出维度为3
                self.linear = nn.Linear(3, 3)
                # 将线性层的权重和偏置分别赋给 weight 和 bias 属性
                self.weight = self.linear.weight
                self.bias = self.linear.bias

            def forward(self, x):
                # 前向传播函数
                x = self.linear(x)
                # 使用 F.linear 函数再次计算线性操作，传入 weight 和 bias
                x = F.linear(x, self.weight, self.bias)
                return x

        # 生成一个形状为 (2, 3) 的随机张量
        x = torch.randn(2, 3)
        # 设定随机数种子为 0
        torch.manual_seed(0)
        # 创建 Foo 的实例
        mod = Foo()
        # 计算模型输出的损失和
        loss = mod(x).sum()
        # 使用 autograd 计算损失对模型参数的梯度
        expected = torch.autograd.grad(loss, mod.parameters())

        # 重新创建 Foo 的实例
        mod = Foo()
        # 调用 make_functional_with_buffers 函数，获取模型的函数化版本以及参数和缓冲区
        fmod, _, _ = make_functional_with_buffers(mod)
        # 设定随机数种子为 0
        torch.manual_seed(0)
        # 再次创建 Foo 的实例
        mod = Foo()
        # 调用 grad 函数，计算函数化模型的梯度
        result = grad(compute_loss)(params, buffers, x)

        # 断言计算得到的梯度与预期的梯度是否一致
        self.assertEqual(result, expected)
    # 定义测试方法，用于测试参数绑定的集成
    def test_parameter_tying_ensemble(self):
        # 定义一个名为 Foo 的神经网络模块
        class Foo(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个线性层，输入和输出都是 3
                self.linear = nn.Linear(3, 3)
                # 将线性层的权重赋值给变量 weight
                self.weight = self.linear.weight
                # 将线性层的偏置赋值给变量 bias
                self.bias = self.linear.bias
                # 创建一个随机张量，并注册为模块的缓冲区 buffer
                self.register_buffer("buffer", torch.randn(3))
                # 将 buffer 缓冲区的内容赋值给另一个缓冲区 buffer_tied
                self.register_buffer("buffer_tied", self.buffer)

            # 前向传播方法
            def forward(self, x):
                # 使用线性层进行前向计算
                x = self.linear(x)
                # 使用 F.linear 函数进行线性变换，传入权重和偏置
                x = F.linear(x, self.weight, self.bias)
                # 添加模块的缓冲区 buffer 到 x 上
                x = x + self.buffer
                # 添加模块的缓冲区 buffer_tied 到 x 上
                x = x + self.buffer_tied
                return x

        # 定义模型的数量
        num_models = 2
        # 生成输入数据 xs，大小为 (num_models, 64, 3)
        xs = torch.randn(num_models, 64, 3)
        # 创建多个 Foo 类的实例，组成模型列表 models
        models = [Foo() for _ in range(num_models)]
        # 调用 combine_state_for_ensemble 函数，获取集成模型和其它状态
        fmodel, _, _ = combine_state_for_ensemble(models)

        # 设定随机种子为 0，重新创建模型列表 models
        torch.manual_seed(0)
        models = [Foo() for _ in range(num_models)]
        # 调用 combine_state_for_ensemble 函数，获取集成模型的参数和缓冲区
        _, params, buffers = combine_state_for_ensemble(models)
        # 使用 vmap 函数对 fmodel 进行矢量化映射，传入参数、缓冲区和输入数据 xs
        result = vmap(fmodel)(params, buffers, xs)

        # 设定随机种子为 0，重新创建模型列表 models
        torch.manual_seed(0)
        models = [Foo() for _ in range(num_models)]
        # 创建期望的输出结果 expected，通过对每个模型和输入数据进行前向传播计算
        expected = torch.stack([model(x) for model, x in zip(models, xs)])

        # 使用 self.assertEqual 检验 result 和 expected 是否相等
        self.assertEqual(result, expected)

    # 使用 parametrize 装饰器为测试方法传递参数 mechanism
    @parametrize("mechanism", ["make_functional", "functional_call"])
    # 定义测试方法，用于测试 MNIST 模型的正确性
    def test_correctness_mnist(self, mechanism):
        # 定义一个名为 Net 的神经网络模块
        class Net(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 第一个卷积层，输入通道数为 1，输出通道数为 10，卷积核大小为 5
                self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
                # 第二个卷积层，输入通道数为 10，输出通道数为 20，卷积核大小为 5
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                # 二维 Dropout 层
                self.conv2_drop = nn.Dropout2d()
                # 第一个全连接层，输入特征数为 320，输出特征数为 50
                self.fc1 = nn.Linear(320, 50)
                # 第二个全连接层，输入特征数为 50，输出特征数为 10
                self.fc2 = nn.Linear(50, 10)

            # 前向传播方法
            def forward(self, x):
                # 使用 ReLU 函数和最大池化对第一个卷积层的输出进行处理
                x = F.relu(F.max_pool2d(self.conv1(x), 2))
                # 使用 ReLU 函数、最大池化和 Dropout 层对第二个卷积层的输出进行处理
                x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
                # 将 x 进行形状变换，展平成一维向量
                x = x.view(-1, 320)
                # 使用 ReLU 函数对第一个全连接层的输出进行处理
                x = F.relu(self.fc1(x))
                # 使用 Dropout 层对第一个全连接层的输出进行处理
                x = F.dropout(x, training=self.training)
                # 使用第二个全连接层进行分类，输出 log_softmax
                x = self.fc2(x)
                return F.log_softmax(x)

        # 生成随机输入数据 x，大小为 (64, 1, 32, 32)
        x = torch.randn(64, 1, 32, 32)
        # 设定随机种子为 301，创建 Net 类的实例并获取权重和函数调用方式
        torch.manual_seed(301)
        fnet, _ = _get_weights_and_functional_call(Net(), mechanism)

        # 设定随机种子为 0，重新创建 Net 类的实例并获取权重和函数调用方式
        torch.manual_seed(0)
        _, params = _get_weights_and_functional_call(Net(), mechanism)
        # 使用 fnet 函数进行函数调用，传入权重和输入数据 x
        result = fnet(params, x)

        # 设定随机种子为 0，重新创建 Net 类的实例并获取期望的输出结果
        torch.manual_seed(0)
        net = Net()
        expected = net(x)

        # 使用 self.assertEqual 检验 result 和 expected 是否相等
        self.assertEqual(result, expected)
    def test_combine_state_for_ensemble_error(self):
        # 定义输入特征数和输出特征数
        in_features = 2
        out_features = 2

        # 初始化模型列表
        models = []
        # 使用断言检查是否引发了预期的 RuntimeError 异常，提示至少需要一个模型
        with self.assertRaisesRegex(RuntimeError, "Expected at least one model"):
            _ = combine_state_for_ensemble(models)

        # 设定模型数量
        num_models = 3
        # 创建包含指定数量线性层模型的列表，并将第二个模型设为评估模式
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        models[1].eval()
        # 使用断言检查是否引发了预期的 RuntimeError 异常，提示模型需要保持相同的训练/评估模式
        with self.assertRaisesRegex(RuntimeError, "same training/eval mode"):
            _ = combine_state_for_ensemble(models)

        # 再次创建模型列表，但将第二个模型替换为一个卷积层模型
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        models[1] = torch.nn.Conv2d(3, 3, (3, 3))
        # 使用断言检查是否引发了预期的 RuntimeError 异常，提示模型需要属于同一类别
        with self.assertRaisesRegex(RuntimeError, "models to be of the same class"):
            _ = combine_state_for_ensemble(models)

    def test_combine_state_for_ensemble_smoke(self):
        # 定义输入特征数和输出特征数
        in_features = 2
        out_features = 2
        # 设定模型数量
        num_models = 3
        # 创建包含指定数量线性层模型的列表
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        # 调用函数测试组合模型状态的功能
        _ = combine_state_for_ensemble(models)

    def test_stack_module_state_smoke(self):
        # 定义输入特征数和输出特征数
        in_features = 2
        out_features = 2
        # 设定模型数量
        num_models = 3
        # 创建包含指定数量线性层模型的列表
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        # 调用函数测试堆叠模块状态的功能
        _ = stack_module_state(models)

    def test_stack_module_state_leaf(self):
        # 定义输入特征数和输出特征数
        in_features = 2
        out_features = 2
        # 设定模型数量
        num_models = 3
        # 创建包含指定数量线性层模型的列表
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        # 调用函数获取堆叠模块状态的参数和缓冲区
        params, buffers = stack_module_state(models)
        # 遍历参数字典中的每个参数，确保每个参数要求梯度且为叶子节点
        for param in params.values():
            self.assertTrue(param.requires_grad)
            self.assertTrue(param.is_leaf)

    def test_stack_module_state_mismatch_error(self):
        # 定义输入特征数和输出特征数
        in_features = 2
        out_features = 2
        # 设定模型数量
        num_models = 3
        # 创建包含指定数量线性层模型的列表，并将第一个模型的权重设为不需要梯度
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        models[0].weight.requires_grad_(False)
        # 使用断言检查是否引发了预期的 RuntimeError 异常，提示模型参数需要具有相同的 requires_grad 属性
        with self.assertRaisesRegex(RuntimeError, "same .requires_grad"):
            params, buffers = stack_module_state(models)

    def test_stack_module_state_error(self):
        # 定义输入特征数和输出特征数
        in_features = 2
        out_features = 2

        # 初始化模型列表
        models = []
        # 使用断言检查是否引发了预期的 RuntimeError 异常，提示至少需要一个模型
        with self.assertRaisesRegex(
            RuntimeError, "stack_module_state:.* Expected at least one model"
        ):
            _ = stack_module_state(models)

        # 设定模型数量
        num_models = 3
        # 创建包含指定数量线性层模型的列表，并将第二个模型设为评估模式
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        models[1].eval()
        # 使用断言检查是否引发了预期的 RuntimeError 异常，提示模型需要保持相同的训练/评估模式
        with self.assertRaisesRegex(
            RuntimeError, "stack_module_state:.* same training/eval mode."
        ):
            _ = stack_module_state(models)

        # 再次创建模型列表，但将第二个模型替换为一个卷积层模型
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        models[1] = torch.nn.Conv2d(3, 3, (3, 3))
        # 使用断言检查是否引发了预期的 RuntimeError 异常，提示模型需要属于同一类别
        with self.assertRaisesRegex(
            RuntimeError, "stack_module_state:.* models to be of the same class"
        ):
            _ = stack_module_state(models)
    @parametrize("mechanism", ["make_functional", "functional_call"])
    def test_make_functional_state_correctly_returned_after_forward(self, mechanism):
        # 定义一个简单的神经网络模型
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 3)

            def forward(self, x):
                # 在前向传播中应用线性层
                x = self.linear(x)
                return x

        # 定义一个函数，根据机制返回模块信息
        def get_module_info(mod):
            if mechanism == "make_functional":
                # 如果机制是 "make_functional"，则返回功能化后的模块
                return make_functional(mod)
            else:
                # 否则机制应为 "functional_call"，返回模块本身和其参数的字典
                assert mechanism == "functional_call"
                return mod, dict(mod.named_parameters())

        # 创建一个神经网络模型实例
        mod = Net()
        # 调用函数获取模块信息
        func_mod, params = get_module_info(mod)

        # 获取线性层的旧状态（权重和偏置）
        # 在 func_mod 中访问 stateless_model 或者直接访问 func_mod 取决于机制
        mod = func_mod.stateless_model if mechanism == "make_functional" else func_mod
        old_state_linear_weight = mod.linear.weight
        old_state_linear_bias = mod.linear.bias

        # 断言旧状态的权重和偏置不为空
        self.assertIsNotNone(old_state_linear_weight)
        self.assertIsNotNone(old_state_linear_bias)

        # 创建输入张量 x，形状为 (4, 3)
        x = torch.randn(4, 3)
        # 根据不同的机制应用模块
        if mechanism == "make_functional":
            func_mod(params, x)
        else:
            assert mechanism == "functional_call"
            functional_call(func_mod, params, x)

        # 再次获取线性层的新状态（权重和偏置）
        # 在 func_mod 中访问 stateless_model 或者直接访问 func_mod 取决于机制
        mod = func_mod.stateless_model if mechanism == "make_functional" else func_mod
        new_state_linear_weight = mod.linear.weight
        new_state_linear_bias = mod.linear.bias

        # 断言新状态的权重和偏置不为空
        self.assertIsNotNone(new_state_linear_weight)
        self.assertIsNotNone(new_state_linear_bias)

        # 断言新状态的权重和偏置与旧状态相等
        self.assertEqual(old_state_linear_weight, new_state_linear_weight)
        self.assertEqual(old_state_linear_bias, new_state_linear_bias)
@markDynamoStrictTest
class TestExamplesCorrectness(TestCase):
    # 测试类标记为 DynamoStrictTest
    def _update_params(self, params, grads, alpha, mechanism):
        # 更新参数函数，根据机制进行不同操作
        if mechanism == "make_functional":
            # 如果机制是 make_functional，则返回更新后的参数列表
            return [(params[i] - alpha * grads[i]) for i in range(len(params))]
        else:
            # 否则，机制必须是 functional_call
            assert mechanism == "functional_call"
            # 返回更新后的参数字典
            return {k: params[k] - alpha * grads[k] for k in params}

    @parametrize("mechanism", ["make_functional", "functional_call"])
    @parametrize("mechanism", ["make_functional", "functional_call"])
    @parametrize("mechanism", ["make_functional", "functional_call"])
    @parametrize("originally_track_running_stats", [True, False])
    def test_update_batch_norm(self, device, originally_track_running_stats, mechanism):
        # 测试更新批归一化操作
        dtype = torch.double
        inplace_relu = False
        classes = 5
        num_batches = 2
        # 构建神经网络模型
        net = (
            nn.Sequential(
                nn.Conv2d(64, 64, 3),
                nn.BatchNorm2d(
                    64, affine=True, track_running_stats=originally_track_running_stats
                ),
                nn.ReLU(inplace=inplace_relu),
                nn.Flatten(),
                nn.Linear(43264, classes),
            )
            .to(device)
            .to(dtype)
        )

        # 替换网络中所有批归一化模块
        replace_all_batch_norm_modules_(net)
        transformed_net = net
        # 获取网络权重和功能调用时的参数及缓存
        fnet, params, buffers = _get_weights_and_functional_call_with_buffers(
            transformed_net, mechanism
        )
        criterion = nn.CrossEntropyLoss()

        def compute_loss(x, y, params, buffers):
            # 计算损失函数
            return criterion(fnet(params, buffers, x), y)

        # 获取一些示例输入数据
        x = torch.randn(num_batches, 1, 64, 28, 28, device=device, dtype=dtype)
        y = torch.randint(0, classes, (num_batches, 1), device=device)

        # 使用 vmap + grad 计算每个样本的梯度
        result_grads = vmap(grad(compute_loss, argnums=2), in_dims=(0, 0, None, None))(
            x, y, params, buffers
        )

        # 不使用 vmap + grad 计算每个样本的梯度
        fnet, params, buffers = _get_weights_and_functional_call_with_buffers(
            transformed_net, mechanism
        )
        flat_params, spec = tree_flatten(params)
        expected_grads = [
            torch.autograd.grad(compute_loss(x[i], y[i], params, buffers), flat_params)
            for i in range(num_batches)
        ]
        expected_grads = [torch.stack(shards) for shards in zip(*expected_grads)]
        expected_grads = tree_unflatten(expected_grads, spec)

        # 断言计算得到的梯度与预期的梯度相等
        self.assertEqual(result_grads, expected_grads)

    @parametrize("jac", ["jacfwd", "jacrev"])
    # 定义测试函数，测试批处理雅可比矩阵计算
    def test_lennard_jones_batched_jac(self, device, jac):
        # 设置 Lennard-Jones 势能函数的参数
        sigma = 0.5
        epsilon = 4.0

        # 获取指定的 functorch 中的函数
        jac = getattr(functorch, jac)

        # 定义 Lennard-Jones 势能公式
        def lennard_jones(r):
            return epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

        # 定义 Lennard-Jones 势能的力公式
        def lennard_jones_force(r):
            """Get magnitude of LJ force"""
            return -epsilon * (
                (-12 * sigma**12 / r**13) + (6 * sigma**6 / r**7)
            )

        # 生成一系列 r 值，表示原子之间的距离，使用 Torch 张量表示
        r = torch.linspace(0.5, 2 * sigma, steps=100, requires_grad=True, device=device)
        # 创建 r 的单位方向向量 drs
        drs = torch.outer(r, torch.tensor([1.0, 0, 0], device=device))
        # 计算 drs 向量的范数 norms
        norms = torch.norm(drs, dim=1).reshape(-1, 1)
        # 计算训练用的能量值，使用 Lennard-Jones 势能函数
        training_energies = torch.stack(list(map(lennard_jones, norms))).reshape(-1, 1)
        # 计算训练用的力，使用 Lennard-Jones 势能的力函数
        training_forces = torch.stack(
            [force * dr for force, dr in zip(map(lennard_jones_force, norms), drs)]
        )

        # 创建神经网络模型，使用了四个隐藏层和一个输出层
        model = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        ).to(device)

        # 定义预测函数，根据模型预测能量和力
        def make_prediction(model, drs, use_functorch):
            norms = torch.norm(drs, dim=1).reshape(-1, 1)
            energies = model(norms)

            if use_functorch:
                # 使用 functorch 提供的自动雅可比矩阵计算
                network_derivs = vmap(jac(model))(norms).squeeze(-1)
                forces = -network_derivs * drs / norms
            else:
                # 使用 PyTorch 自带的自动微分计算
                forces = []
                for r, dr in zip(norms, drs):
                    network_deriv = torch.autograd.functional.jacobian(
                        model, r, create_graph=True
                    )
                    force = -network_deriv * dr / r
                    forces.append(force)
                forces = torch.cat(forces)
            return energies, forces

        # 定义损失函数，包括能量损失和力损失
        def loss_fn(energies, forces, predicted_energies, predicted_forces):
            return (
                F.mse_loss(energies, predicted_energies)
                + 0.01 * F.mse_loss(forces, predicted_forces) / 3
            )

        # 使用 functorch 计算预测的能量和力
        energies, forces = make_prediction(model, drs, use_functorch=True)
        # 计算使用 functorch 计算得到的损失
        loss = loss_fn(training_energies, training_forces, energies, forces)
        # 计算损失相对于模型参数的梯度
        result = torch.autograd.grad(loss, model.parameters())

        # 使用 PyTorch 自带的自动微分计算预测的能量和力
        energies, forces = make_prediction(model, drs, use_functorch=False)
        # 计算使用 PyTorch 自带自动微分计算得到的损失
        loss = loss_fn(training_energies, training_forces, energies, forces)
        # 计算损失相对于模型参数的梯度
        expected = torch.autograd.grad(loss, model.parameters())

        # 断言计算得到的梯度与期望的梯度相等
        self.assertEqual(result, expected)
    # 设置测试函数的装饰器，用于关闭 TF32 模式，以解决特定的 PyTorch 问题
    @with_tf32_off  # https://github.com/pytorch/pytorch/issues/86798
    # 如果未安装 torchvision，则跳过测试
    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    # 参数化测试函数，使用不同的机制名称进行参数化
    @parametrize("mechanism", ["make_functional", "functional_call"])
    # 定义测试函数，测试 ResNet-18 模型每个样本的梯度
    def test_resnet18_per_sample_grads(self, device, mechanism):
        # 导入 torchvision 中的 models 模块
        import torchvision.models as models

        # 创建一个未经预训练的 ResNet-18 模型，并移到指定的设备上
        model = models.__dict__["resnet18"](
            pretrained=False, norm_layer=(lambda c: nn.GroupNorm(min(32, c), c))
        ).to(device)
        
        # 定义交叉熵损失函数，设置 reduction="sum" 以避免跨批次的比较
        criterion = nn.CrossEntropyLoss(
            reduction="sum"
        )  # avoid cross batch reductions for for loop comparison

        # 获取权重和功能调用后的模型
        func_model, weights = _get_weights_and_functional_call(model, mechanism)

        # 定义计算损失的函数
        def compute_loss(weights, image, target):
            # 在图像维度上扩展单个图像和目标
            image = image.unsqueeze(0)
            target = target.unsqueeze(0)
            # 使用功能调用模型计算输出
            output = func_model(weights, image)
            # 计算交叉熵损失
            loss = criterion(output, target)
            return loss

        # 设置批量大小
        batch_size = 3
        # 生成随机图像数据和目标标签，移到指定设备上
        images = torch.randn(batch_size, 3, 32, 32, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)

        # 使用 vmap 函数在权重、图像和目标标签的输入维度上运行梯度函数
        result_grads = vmap(grad(compute_loss), in_dims=(None, 0, 0))(
            weights, images, targets
        )

        # 展平权重，并获取结构描述
        flat_weights, spec = tree_flatten(weights)
        # 计算每个样本的预期梯度
        expected_grads = [
            torch.autograd.grad(
                compute_loss(weights, images[i], targets[i]), flat_weights
            )
            for i in range(batch_size)
        ]
        # 将结果堆叠为张量
        expected_grads = [torch.stack(shards) for shards in zip(*expected_grads)]
        # 将展平后的梯度结构化为原始形状
        expected_grads = tree_unflatten(expected_grads, spec)

        # 断言结果梯度与预期梯度的近似相等性
        self.assertEqual(result_grads, expected_grads, atol=1e-3, rtol=1.0)
# 根据输入的图形对象对设备进行规范化处理
def normalize_devices(fx_g):
    # 遍历图中的每个节点
    for node in fx_g.graph.nodes:
        # 将节点的参数列表转换为列表形式
        args = list(node.args)
        # 遍历节点的参数列表
        for idx, arg in enumerate(args):
            # 如果参数是 torch.device 类型，则替换为字符串 "cpu"
            if isinstance(arg, torch.device):
                args[idx] = "cpu"
        # 将处理后的参数列表重新赋值给节点
        node.args = tuple(args)
        
        # 创建一个新的关键字参数字典
        new_kwargs = {}
        # 遍历节点的关键字参数
        for k, v in node.kwargs.items():
            # 如果参数值是 torch.device 类型，则替换为字符串 "cpu"
            if isinstance(v, torch.device):
                v = "cpu"
            # 更新新的关键字参数字典
            new_kwargs[k] = v
        # 将更新后的关键字参数字典赋值给节点
        node.kwargs = new_kwargs

    # 重新编译处理后的图形对象
    fx_g.recompile()

    # 返回处理后的图形对象
    return fx_g


@markDynamoStrictTest
class TestFunctionalize(TestCase):
    # 检查函数功能化的正确性
    def _check_functionalize_correctness(self, f, inpt, *, skip_vmap=False):
        # 克隆输入张量，准备进行测试
        inpt1 = inpt.clone()
        inpt2 = inpt.clone()
        inpt3 = inpt.clone()

        # 计算期望输出
        expected_outputs = f(inpt1)
        
        # 根据条件决定是否跳过 vmap
        if skip_vmap:
            actual_outputs = functionalize(f)(inpt2)
        else:
            actual_outputs = vmap(functionalize(f))(inpt2.unsqueeze(0))[0].squeeze()

        # 使用 functionalize 函数处理的实际输出（暂时不包含 view 操作）
        actual_outputs_view_copy = functionalize(f, remove="mutations_and_views")(inpt3)

        # 断言两种方式得到的输出相同
        self.assertEqual(actual_outputs, expected_outputs)
        self.assertEqual(actual_outputs_view_copy, expected_outputs)

        # 检查输入张量是否被 f 函数正确修改
        self.assertEqual(inpt1, inpt2)
        self.assertEqual(inpt1, inpt3)

    # 测试简单的 view 操作
    def test_simple_view(self, device):
        # 定义一个操作函数 f
        def f(x: torch.Tensor) -> torch.Tensor:
            tmp = torch.ones(2, device=device)
            y = x.view(4, 2)
            y.add_(tmp)
            return x

        # 调用 _check_functionalize_correctness 进行测试
        self._check_functionalize_correctness(f, torch.zeros(4, 2, device=device))

    # 测试多输出 view 操作
    def test_multioutput_view(self, device):
        # 定义一个操作函数 f
        def f(x: torch.Tensor) -> torch.Tensor:
            tmp = torch.ones(2, device=device)
            y1, y2 = x.split(2)
            y1_view = y1.diagonal()
            y1_view.add_(tmp)
            return x

        # 调用 _check_functionalize_correctness 进行测试
        self._check_functionalize_correctness(f, torch.zeros(4, 2, device=device))

    # 测试原地 view 操作
    def test_inplace_view(self, device):
        # 定义一个操作函数 f
        def f(x: torch.Tensor) -> torch.Tensor:
            tmp = torch.ones(4, device=device)
            y = x + x
            y2 = y.transpose(1, 0)
            z = y2[0]
            z.add_(tmp)
            return y

        # 调用 _check_functionalize_correctness 进行测试，跳过 vmap
        self._check_functionalize_correctness(
            f, torch.zeros(4, 2, device=device), skip_vmap=True
        )

    # 参见 https://github.com/pytorch/functorch/issues/780
    # 定义测试函数，用于测试 torch._C._nn.linear 的功能
    def test_linear(self, device):
        # 定义一个简单的函数 f，返回 torch.Tensor
        def f(x, y, z) -> torch.Tensor:
            return torch._C._nn.linear(x, y, z)

        # 生成随机张量 x，形状为 (14, 1, 384)，在指定设备上
        x = torch.randn(14, 1, 384, device=device)
        # 生成随机张量 y，形状为 (96, 384)，在指定设备上
        y = torch.randn(96, 384, device=device)
        # 生成随机张量 z，形状为 (96)，在指定设备上
        z = torch.randn(96, device=device)

        # 使用函数 f 计算预期输出
        out_expected = f(x, y, z)
        # 使用 functionalize 包装函数 f，并计算实际输出
        out_actual = functionalize(f)(x, y, z)
        # 断言预期输出与实际输出相等
        self.assertEqual(out_expected, out_actual)

    # 测试多输出、原地操作和切片视图的函数
    def test_multioutput_inplace_slice_view(self, device):
        # 定义函数 f，接受一个 torch.Tensor 参数，返回 torch.Tensor
        def f(x: torch.Tensor) -> torch.Tensor:
            # 创建临时张量 tmp，形状为 (2, 2)，在指定设备上
            tmp = torch.ones(2, 2, device=device)
            # 将输入张量 x 视图为 1 维张量 y
            y = x.view(8)
            # 将 y 重塑为形状为 (2, 4) 的张量 z0
            z0 = y.reshape(2, 4)
            # 将 z0 转置为形状为 (4, 2) 的张量 z1
            z1 = z0.transpose(1, 0)
            # 在 z1 张量上增加一个维度
            z1.unsqueeze_(0)
            # 在 z1 张量上压缩第一个维度
            z1.squeeze_()
            # 在 z1 张量上分割为 z2 和 z3，每部分形状为 (2, 2)
            z2, z3 = z1.split(2)
            # 将 tmp 加到 z2 上
            z2.add_(tmp)
            # 返回输入张量 x
            return x

        # 调用 _check_functionalize_correctness 函数验证 functionalize 是否正确处理函数 f
        self._check_functionalize_correctness(
            f, torch.zeros(4, 2, device=device), skip_vmap=True
        )

    # 确保 functionalize 能处理 List[Optional[Tensor]] 类型的参数
    # 参考 https://github.com/pytorch/pytorch/pull/76085 的修复/讨论
    def test_functionalize_opt_tensor_list(self, device):
        # 定义函数 f，接受两个参数 x 和 indices，返回 x[indices] 的结果
        def f(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            return x[indices]

        # 创建输入张量 inpta，形状为 (4)，在指定设备上
        inpta = torch.ones(4, device=device)
        # 创建输入张量 inptb，形状为 (2)，在指定设备上
        inptb = torch.arange(2, device=device)
        
        # 使用函数 f 计算输出 out1
        out1 = f(inpta, inptb)
        # 使用 functionalize 包装函数 f，并计算输出 out2
        out2 = functionalize(f)(inpta, inptb)
        # 断言 out1 与 out2 相等
        self.assertEqual(out1, out2)
        
        # 使用 make_fx 创建 functionalize(f) 的函数，并计算输出 out
        out = make_fx(functionalize(f))(inpta, inptb)
        # 断言输出 out 的行为符合预期
        self.assertExpectedInline(
            (out.code),
            """\
# 定义一个方法 forward，接受参数 self, x_1, indices_1，并返回一个 torch.Tensor 对象
def forward(self, x_1, indices_1) -> torch.Tensor:
    # 使用 torch.ops.aten.index.Tensor 方法创建一个索引对象 index，将 x_1 和 indices_1 作为参数传入；然后清空 x_1 和 indices_1 的引用
    index = torch.ops.aten.index.Tensor(x_1, [indices_1]);  x_1 = indices_1 = None
    # 返回创建的索引对象 index
    return index
    """
        )

    # 确保 grad(functionalize(f)) 正常工作
    def test_functionalize_grad(self, device):
        # 定义一个函数 f，接受参数 x，并返回一个 torch.Tensor 对象
        def f(x: torch.Tensor) -> torch.Tensor:
            # 创建一个形状为 (2,) 的全为 1 的张量 tmp，放在指定的设备上
            tmp = torch.ones(2, device=device)
            # 计算 y = x + x
            y = x + x
            # 将 y 重塑为形状为 (4, 2) 的张量 z
            z = y.view(4, 2)
            # 将 tmp 添加到 y 中
            y.add_(tmp)
            # 返回 z 的所有元素之和
            return z.sum()

        # 创建一个形状为 (4, 2) 的全为 1 的张量 inpt1，并放在指定的设备上
        inpt1 = torch.ones(4, 2, device=device)
        # 创建一个形状为 (4, 2) 的全为 1 的张量 inpt2，并放在指定的设备上
        inpt2 = torch.ones(4, 2, device=device)
        # 计算函数 f 对输入张量 inpt1 的梯度
        out1 = grad(f)(inpt1)
        # 使用 functionalize(f) 计算函数 f 对输入张量 inpt2 的梯度
        out2 = grad(functionalize(f))(inpt2)
        # 断言两个梯度计算结果相等
        self.assertEqual(out1, out2)
        # 断言输入张量 inpt1 和 inpt2 相等
        self.assertEqual(inpt1, inpt2)

    @unittest.skipIf(IS_FBCODE, "fails in fbcode")
    def test_vmap_functionalize_jvp(self, device):
        # 定义一个函数 f，接受参数 x，并返回一个 torch.Tensor 对象
        def f(x: torch.Tensor) -> torch.Tensor:
            # 计算 y = x + x
            y = x + x
            # 将 y 重塑为一个扁平的张量 z
            z = y.view(-1)
            # 将 1 添加到 y 中
            y.add_(1)
            # 返回 z
            return z

        # 定义一个包装函数 jvp_wrapper，接受参数 x 和 t
        def jvp_wrapper(x, t):
            # 调用 jvp 函数，对函数 f 和输入 x, t 进行求偏导数操作
            return jvp(
                f,
                (x,),
                (t,),
            )

        # 创建一个形状为 (2, 3) 的随机张量 x，并放在指定的设备上
        x = torch.randn(2, 3, device=device)
        # 创建一个形状为 (2, 3) 的随机张量 t，并放在指定的设备上
        t = torch.randn(2, 3, device=device)

        # 使用 vmap 对 jvp_wrapper 函数进行矢量化操作，输入 x 和 t
        out1 = vmap(jvp_wrapper)(x, t)
        # 使用 functionalize(jvp_wrapper) 对 jvp_wrapper 函数进行函数化操作，输入 x 和 t
        out2 = vmap(functionalize(jvp_wrapper))(x, t)
        # 断言两者结果相等
        self.assertEqual(out1, out2)

    # TODO: 将此测试移到 test_fake_tensor.py，一旦 functionalize() 能在核心测试中使用。
    def test_functionalize_fake_tensors(self, device):
        # 定义一个函数 f，接受参数 x，并返回一个 torch.Tensor 对象
        def f(x: torch.Tensor) -> torch.Tensor:
            # 创建一个张量 y，与 x 分离，即不再依赖于 x
            y = x.detach()
            # 返回 y + y 的结果
            return y + y

        # 进入 FakeTensorMode 上下文
        with FakeTensorMode() as mode:
            # 创建一个形状为 (2,) 的全为 1 的张量 x，并将其放在指定的设备上，同时要求计算梯度
            x = torch.ones(2, device=device, requires_grad=True)
            # 使用 functionalize(f) 对函数 f 进行函数化，输入 x
            out = functionalize(f)(x)
        # 断言输出张量 out 的形状为 (2,)
        self.assertEqual(x.size(), (2,))

    def test_functionalize_fx_simple(self, device):
        # 定义一个函数 f，接受参数 x，并返回一个 torch.Tensor 对象
        def f(x: torch.Tensor) -> torch.Tensor:
            # 创建一个形状为 (2,) 的全为 1 的张量 tmp，放在指定的设备上
            tmp = torch.ones(2, device=device)
            # 将 x 重塑为形状为 (4, 2) 的张量 y
            y = x.view(4, 2)
            # 将 tmp 添加到 y 中
            y.add_(tmp)
            # 返回输入张量 x
            return x

        # 使用 functionalize(f, remove="mutations_and_views") 创建 fx 操作
        fn = make_fx(functionalize(f, remove="mutations_and_views"))
        # 对形状为 (4, 2) 的全为 0 的张量进行 fx 操作，并将结果进行设备归一化
        out = fn(torch.zeros(4, 2, device=device))
        # 断言行内预期结果
        out = normalize_devices(out)
        self.assertExpectedInline(
            (out.code),
            """



def forward(self, x_1) -> torch.Tensor:
    ones = torch.ops.aten.ones.default([2], device = 'cpu', pin_memory = False)
    view_copy = torch.ops.aten.view_copy.default(x_1, [4, 2])
    add = torch.ops.aten.add.Tensor(view_copy, ones);  view_copy = ones = None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2]);  add = None
    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [4, 2])
    copy_ = torch.ops.aten.copy_.default(x_1, view_copy_1);  x_1 = None
    return view_copy_1
    """
        )
        # 定义一个嵌套函数f，接受一个torch.Tensor类型的参数x，并返回其转置后的结果
        def f(x: torch.Tensor) -> torch.Tensor:
            return x.transpose(1, 0)

        # 使用functionalize函数将f转化为FX，并移除“mutations_and_views”中的内容
        fn = make_fx(functionalize(f, remove="mutations_and_views"))
        
        # 对一个设备上的4x2的全零张量进行fn函数的运算
        out = fn(torch.zeros(4, 2, device=device))
        
        # 对输出结果进行设备规范化处理
        out = normalize_devices(out)
        
        # 使用self.assertExpectedInline函数验证out.code的预期输出
        self.assertExpectedInline(
            out.code,
            """\
def forward(self, inpt_1) -> torch.Tensor:
    # 创建一个空的张量，不指定形状，数据类型为 float32，在 CPU 上，不使用 pin memory
    empty = torch.ops.aten.empty.memory_format([], dtype=torch.float32, device='cpu', pin_memory=False)
    # 对输入张量进行加法操作，结果保存在新的张量中，同时清空原始输入张量的引用
    add = torch.ops.aten.add.Tensor(inpt_1, inpt_1);  inpt_1 = None
    # 使用指定的形状对加法结果进行视图变换
    view_copy = torch.ops.aten.view_copy.default(add, [4])
    # 再次对加法结果进行视图变换，指定形状为 [4]，同时清空前一个变量的引用
    view_copy_1 = torch.ops.aten.view_copy.default(add, [4]);  add = None
    # 对视图变换后的张量进行加法操作，增加值为标量 1，同时清空前一个变量的引用
    add_1 = torch.ops.aten.add.Tensor(view_copy_1, 1);  view_copy_1 = None
    # 对加法操作后的结果再次进行视图变换，指定形状为 [4]，同时清空前一个变量的引用
    view_copy_2 = torch.ops.aten.view_copy.default(add_1, [4]);  add_1 = None
    # 对最终的视图变换结果再次进行视图变换，指定形状为 [4]
    view_copy_3 = torch.ops.aten.view_copy.default(view_copy_2, [4])
    # 返回最终的视图变换结果作为函数的输出
    return view_copy_2
    # 定义一个内部函数 f，接受一个 torch.Tensor 类型的参数 x，并返回一个 torch.Tensor 类型的结果
    def f(x: torch.Tensor) -> torch.Tensor:
        # 在指定设备上创建一个形状为 (2,) 的全为 1 的张量 tmp
        tmp = torch.ones(2, device=device)
        # 将输入张量 x 重新视图为形状 (4, 2)，结果保存在变量 y 中
        y = x.view(4, 2)
        # 将 tmp 加到 y 上，直接修改 y 的值
        y.add_(tmp)
        # 返回输入张量 x，未经任何修改
        return x

    # 使用 functionalize 函数将 f 函数转换为可序列化的 FX 函数，再调用 make_fx 进行处理
    out = make_fx(functionalize(f))(torch.zeros(4, 2, device=device))
    # 将输出 out 进行设备标准化处理
    out = normalize_devices(out)
    # 使用 self.assertExpectedInline 方法验证 out.code 是否符合预期输出
    self.assertExpectedInline(
        out.code,
        """\
def forward(self, x_1) -> torch.Tensor:
    # 创建一个包含两个元素的全1张量，设备为CPU，不需要固定在内存中
    ones = torch.ops.aten.ones.default([2], device='cpu', pin_memory=False)
    # 将输入张量 x_1 视图重塑为 4x2 的张量
    view = torch.ops.aten.view.default(x_1, [4, 2])
    # 将两个张量相加，结果保存在 add 中；释放 view 和 ones 的引用
    add = torch.ops.aten.add.Tensor(view, ones); view = ones = None
    # 将 add 张量视图重塑为 4x2 的张量；释放 add 的引用
    view_1 = torch.ops.aten.view.default(add, [4, 2]); add = None
    # 将 view_1 的内容复制回输入张量 x_1 中；释放 x_1 的引用
    copy_ = torch.ops.aten.copy_.default(x_1, view_1); x_1 = None
    # 返回重塑后的张量 view_1
    return view_1



def test_functionalize_nonfunctional_output(self, device):
    global_out = torch.ones(2, device=device)

    def f() -> torch.Tensor:
        return global_out

    # 对函数进行功能化处理后，再次标准化设备输出并进行断言
    out = make_fx(functionalize(f))()
    out = normalize_devices(out)
    self.assertExpectedInline(
        out.code,
        """
def forward(self) -> torch.Tensor:
    _tensor_constant0 = self._tensor_constant0
    return _tensor_constant0
""",
    )



def test_functionalize_optional_tensorlist1(self, device):
    def f(a, b) -> torch.Tensor:
        # at::index 具有 OptionalTensorList 参数，这里进行测试
        return a[b]

    a = torch.arange(4).reshape(2, 2)
    b = torch.ones(2, dtype=torch.long)
    # 对函数进行功能化处理后，标准化设备输出并进行断言
    out = make_fx(functionalize(f))(a, b)
    out = normalize_devices(out)
    self.assertExpectedInline(
        out.code,
        """
def forward(self, a_1, b_1) -> torch.Tensor:
    index = torch.ops.aten.index.Tensor(a_1, [b_1]); a_1 = b_1 = None
    return index
""",
    )



@unittest.skipIf(IS_FBCODE, "fails in fbcode")
def test_functionalize_optional_tensorlist2(self, device):
    def f(a, b) -> torch.Tensor:
        # 参考 https://github.com/pytorch/pytorch/pull/77846
        return torch.ops.aten.index(a, b)

    a = torch.arange(4).reshape(2, 2)
    b = torch.ones(2, dtype=torch.long)
    # 对函数进行功能化处理后，标准化设备输出并进行断言
    out = make_fx(functionalize(f))(a, b)
    self.assertExpectedInline(
        out.code,
        """
def forward(self, a_1, b_1) -> torch.Tensor:
    unbind = torch.ops.aten.unbind.int(b_1); b_1 = None
    getitem = unbind[0]
    getitem_1 = unbind[1]; unbind = None
    index = torch.ops.aten.index.Tensor(a_1, [getitem, getitem_1]); a_1 = getitem = getitem_1 = None
    return index
""",
    )



def test_resize_program_inputs(self, device):
    def f(x):
        x.resize_(10)
        x.fill_(2)

    fn = make_fx(functionalize(f))
    out = fn(torch.zeros(0, device=device))
    out = normalize_devices(out)
    self.assertExpectedInline(
        (out.code),
        """
def forward(self, x_1):
    resize = torch.ops.aten.resize.default(x_1, [10])
    fill = torch.ops.aten.fill.Scalar(resize, 2); resize = None
    resize_ = torch.ops.aten.resize_.default(x_1, [10]); x_1 = None
    copy_ = torch.ops.aten.copy_.default(resize_, fill); resize_ = fill = None
    return None
""",
    )
    # 创建一个 HigherOrderOperator 对象，并命名为 "mysum"
    mysum = HigherOrderOperator("mysum")

    # 使用装饰器将下面定义的函数注册为 "mysum" 操作的 Vmap 实现
    @mysum.py_impl(torch._C._functorch.TransformType.Vmap)
    def mysum_batch_rule(interpreter, x, dim):
        # 检查输入张量 x 是否为批处理张量，如果不是，则直接调用 mysum 函数
        if not torch._C._functorch.is_batchedtensor(x):
            with interpreter.lower():
                x = x.view_as(x)  # 无必要，只是用来测试分发
                return mysum(x, dim)

        # 获取批处理维度并解包输入张量的值
        bdim = torch._C._functorch.maybe_get_bdim(x)
        value = torch._C._functorch.get_unwrapped(x)

        # 在降低层次上下文中执行以下代码块
        with interpreter.lower():
            # 将批处理维度移动到索引 0 的位置
            value = value.movedim(bdim, 0)
            # 对移动后的张量应用 mysum 函数，并在 dim+1 维度上求和
            result = mysum(value, dim + 1)

        # 将结果张量添加批处理维度，返回结果
        return torch._C._functorch._add_batch_dim(result, 0, interpreter.level())

    # 使用装饰器将下面定义的函数注册为 "mysum" 操作的 Grad 实现
    @mysum.py_impl(torch._C._functorch.TransformType.Grad)
    def mysum_grad_rule(interpreter, x, dim):
        # 获取解释器的层级
        level = interpreter.level()

        # 定义一个继承自 torch.autograd.function._SingleLevelFunction 的内部类 MySum
        class MySum(torch.autograd.function._SingleLevelFunction):
            @staticmethod
            def forward(ctx, x, dim):
                # 保存输入张量的形状和维度信息到上下文对象
                ctx.x_shape = x.shape
                ctx.dim = dim
                # 使用 level 级别的解包函数解包输入张量 x
                x = torch._C._functorch._unwrap_for_grad(x, level)
                # 在开启梯度计算的上下文中执行以下代码块
                with torch.enable_grad(), interpreter.lower():
                    x = x.view_as(x)  # 无必要，只是用来测试分发
                    # 调用 mysum 函数计算结果 y
                    y = mysum(x, dim)

                # 使用 level 级别的封装函数封装输出结果 y
                y = torch._C._functorch._wrap_for_grad(y, level)
                return y

            @staticmethod
            def backward(ctx, gy):
                # 返回梯度 gy 在 ctx.dim 维度上增加一个维度，再根据 ctx.x_shape 进行扩展
                return gy.unsqueeze(ctx.dim).expand(ctx.x_shape), None

        # 在启用单层级自动求导函数的上下文中，调用 MySum 类的 apply 方法
        with enable_single_level_autograd_function():
            return MySum.apply(x, dim)

    # 使用装饰器将下面定义的函数注册为 "mysum" 操作的 AutogradCPU 实现
    @mysum.py_impl(torch._C.DispatchKey.AutogradCPU)
    def mysum_autograd_cpu(x, dim):
        # 返回输入张量 x 在 dim 维度上的求和结果
        return torch.sum(x, dim)

    # 使用装饰器将下面定义的函数注册为 "mysum" 操作的 AutogradCUDA 实现
    @mysum.py_impl(torch._C.DispatchKey.AutogradCUDA)
    def mysum_autograd_cuda(x, dim):
        # 返回输入张量 x 在 dim 维度上的求和结果
        return torch.sum(x, dim)

    # 返回注册完成的 mysum 操作
    return mysum
sum_pyop = construct_sum_pyop()


@markDynamoStrictTest
class TestHigherOrderOperatorInteraction(TestCase):
    # 测试类，用于测试高阶操作符的交互

    def test_basic_sum(self, device):
        # 测试基本的求和操作
        x = torch.randn(2, 3, 4, device=device)
        result = sum_pyop(x, 1)
        self.assertEqual(result, torch.sum(x, 1))

    def test_vmap_sum(self, device):
        # 测试 vmap 函数与求和操作的交互
        x = torch.randn(2, 3, 4, device=device)
        result = vmap(sum_pyop, (0, None))(x, 0)
        self.assertEqual(result, torch.sum(x, 1))

        result = vmap(vmap(sum_pyop, (0, None)), (0, None))(x, 0)
        self.assertEqual(result, torch.sum(x, 2))

    def test_grad_sum(self, device):
        # 测试对求和操作的梯度计算
        x = torch.randn(3, device=device)
        gx = grad(sum_pyop)(x, 0)
        self.assertEqual(gx, torch.ones_like(x))

    def test_grad_grad_sum(self, device):
        # 测试对求和操作的二阶梯度计算
        x = torch.randn(3, requires_grad=True, device=device)

        def f(x):
            # 高阶梯度计算。需要一个非线性操作
            return sum_pyop(x.sin(), 0)

        def grad_f_sum(x):
            return grad(f)(x).sum()

        ggx = grad(grad_f_sum)(x)
        self.assertEqual(ggx, -x.sin())

    def test_vmap_grad_sum(self, device):
        # 测试 vmap 函数与求和操作的梯度计算
        x = torch.randn(2, 3, device=device)
        gx = vmap(grad(sum_pyop), (0, None))(x, 0)
        self.assertEqual(gx, torch.ones_like(x))

    def test_no_grad_outside_grad(self, device):
        # 测试在梯度计算外部使用 no_grad 上下文
        x = torch.randn(3, device=device, requires_grad=True)
        with torch.no_grad():
            y = grad(sum_pyop)(x, 0)
        self.assertEqual(y, torch.ones_like(x))
        self.assertFalse(y.requires_grad)

    def test_no_grad_inside_grad(self, device):
        # 测试在梯度计算内部使用 no_grad 上下文
        def f(x):
            with torch.no_grad():
                shift = sum_pyop(x**2, 0)
            return sum_pyop(x**2, 0) - shift

        x = torch.randn(3, device=device)
        y = grad(f)(x)
        self.assertEqual(y, 2 * x)
        y = grad(lambda x: grad(f)(x).sum())(x)
        self.assertEqual(y, torch.full_like(x, 2))

        x = torch.randn(3, device=device, requires_grad=True)
        y = grad(f)(x)
        (z,) = torch.autograd.grad(y.sum(), x)
        self.assertEqual(z, torch.full_like(x, 2))

    def test_grad_name_wrapping(self, device):
        # 测试梯度函数的名称包装
        def my_fn(x):
            return x.sum()

        grad_fn = grad(my_fn)
        self.assertEqual(grad_fn.__name__, "my_fn")

    def test_functional_call_multiple_dicts(self):
        # 测试使用多个字典参数的函数调用
        mod = nn.Linear(1, 1)
        x = torch.randn((1, 1))
        params = ({"weight": torch.zeros(1, 1)}, {"bias": torch.ones(1)})
        functional_call(mod, params, x)


def traceable(f):
    # 函数修饰器，允许函数在图中运行
    f = allow_in_graph(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@markDynamoStrictTest
class TestCompileTransforms(TestCase):
    # 测试编译转换的类

    @skipIfRocm(msg="test leaks memory on ROCm")
    # 在 ROCm 上会内存泄漏的测试跳过
    # 在 Windows 上不支持 torch.compile
    # Triton 只支持 SM70 或更高版本的 GPU
    @expectedFailureIf(IS_WINDOWS or (TEST_CUDA and not SM70OrLater))
    # 定义一个测试方法，用于测试编译 vmap_hessian 函数
    def test_compile_vmap_hessian(self, device):
        # 设置输入数据维度 D 和 batch 大小 B
        D = 2
        B = 4

        # 生成一个随机张量 x，形状为 (B, D)，并指定设备为 device
        x = torch.randn(B, D, device=device)

        # 创建一个包含线性层和 ReLU 激活函数的序列模型，并将其移到指定设备
        model = nn.Sequential(nn.Linear(D, D), nn.ReLU()).to(device)

        # 获取模型中的参数和缓冲区，分别存放在两个字典中
        params_and_buffers = (
            dict(model.named_parameters()),
            dict(model.named_buffers()),
        )

        # 定义一个预测函数，使用 torch.func.functional_call 调用模型
        def predict(params_and_buffers, x):
            out = torch.func.functional_call(model, params_and_buffers, x)
            return out, out

        # 使用 vmap 包装预测函数，通过 jacobian forward 和 backward 运算求取函数的高阶导数
        fn = vmap(
            jacfwd(jacrev(predict, argnums=1, has_aux=True), argnums=1, has_aux=True),
            in_dims=(None, 0),
        )

        # 计算预期输出
        expected = fn(params_and_buffers, x)

        # 使用 torch.compile 对函数进行优化编译，生成优化后的函数
        opt_fn = torch.compile(traceable(fn))
        # 计算优化后函数的实际输出
        actual = opt_fn(params_and_buffers, x)
        # 断言优化后函数的输出与预期输出相等
        self.assertEqual(actual, expected)

    # 当运行环境为 Windows 时，torch.compile 不受支持，所以预期测试失败
    @expectedFailureIf(IS_WINDOWS)
    @torch._dynamo.config.patch(suppress_errors=False)
    # 定义一个测试方法，测试使用过时的 API 进行梯度计算
    def test_grad_deprecated_api(self, device):
        # 生成两个随机张量 x 和 y，并指定设备为 device
        x = torch.randn((), device=device)
        y = torch.randn((), device=device)

        # 定义一个包装函数，调用 functorch.grad 计算两个张量的乘积梯度
        def wrapper_fn(x, y):
            return functorch.grad(torch.mul)(x, y)

        # 计算实际输出
        actual = wrapper_fn(x, y)
        # 使用 torch.compile 对包装函数进行优化编译，生成优化后的函数，并计算其输出
        expected = torch.compile(wrapper_fn, backend="eager", fullgraph=True)(x, y)
        # 断言优化后函数的输出与预期输出相等
        self.assertEqual(actual, expected)

        # 重新定义包装函数，调用 functorch.grad 计算两个张量乘积的梯度，指定参数索引
        def wrapper_fn(x, y):
            return functorch.grad(torch.mul, argnums=(0, 1))(x, y)

        # 计算实际输出
        actual = wrapper_fn(x, y)
        # 使用 torch.compile 对包装函数进行优化编译，生成优化后的函数，并计算其输出
        expected = torch.compile(wrapper_fn, backend="eager", fullgraph=True)(x, y)
        # 断言优化后函数的输出与预期输出相等
        self.assertEqual(actual, expected)
# 使用只适用于 "cpu" 和 "cuda" 的设备类型来实例化 TestGradTransform 测试类的设备类型测试
instantiate_device_type_tests(
    TestGradTransform,
    globals(),
    only_for=only_for,
)

# 使用只适用于 "cpu" 和 "cuda" 的设备类型来实例化 TestVmapOfGrad 测试类的设备类型测试
instantiate_device_type_tests(
    TestVmapOfGrad,
    globals(),
    only_for=only_for,
)

# 使用只适用于 "cpu" 和 "cuda" 的设备类型来实例化 TestJac 测试类的设备类型测试
instantiate_device_type_tests(
    TestJac,
    globals(),
    only_for=only_for,
)

# 使用只适用于 "cpu" 和 "cuda" 的设备类型来实例化 TestJvp 测试类的设备类型测试
instantiate_device_type_tests(
    TestJvp,
    globals(),
    only_for=only_for,
)

# 使用只适用于 "cpu" 和 "cuda" 的设备类型来实例化 TestLinearize 测试类的设备类型测试
instantiate_device_type_tests(
    TestLinearize,
    globals(),
    only_for=only_for,
)

# 使用只适用于 "cpu" 和 "cuda" 的设备类型来实例化 TestVmapJvpInplaceView 测试类的设备类型测试
instantiate_device_type_tests(
    TestVmapJvpInplaceView,
    globals(),
    only_for=only_for,
)

# 使用只适用于 "cpu" 和 "cuda" 的设备类型来实例化 TestHessian 测试类的设备类型测试
instantiate_device_type_tests(
    TestHessian,
    globals(),
    only_for=only_for,
)

# 使用只适用于 "cpu" 和 "cuda" 的设备类型来实例化 TestComposability 测试类的设备类型测试
instantiate_device_type_tests(
    TestComposability,
    globals(),
    only_for=only_for,
)

# 使用只适用于 "cpu" 和 "cuda" 的设备类型来实例化 TestExamplesCorrectness 测试类的设备类型测试
instantiate_device_type_tests(
    TestExamplesCorrectness,
    globals(),
    only_for=only_for,
)

# 使用只适用于 "cpu" 和 "cuda" 的设备类型来实例化 TestHigherOrderOperatorInteraction 测试类的设备类型测试
instantiate_device_type_tests(
    TestHigherOrderOperatorInteraction,
    globals(),
    only_for=only_for,
)

# 使用只适用于 "cpu" 和 "cuda" 的设备类型来实例化 TestFunctionalize 测试类的设备类型测试
instantiate_device_type_tests(
    TestFunctionalize,
    globals(),
    only_for=only_for,
)

# 使用只适用于 "cpu" 和 "cuda" 的设备类型来实例化 TestAutogradFunction 测试类的设备类型测试
instantiate_device_type_tests(
    TestAutogradFunction,
    globals(),
    only_for=only_for,
)

# 使用只适用于 "cpu" 和 "cuda" 的设备类型来实例化 TestAutogradFunctionVmapAPI 测试类的设备类型测试
instantiate_device_type_tests(
    TestAutogradFunctionVmapAPI,
    globals(),
    only_for=only_for,
)

# 使用只适用于 "cpu" 和 "cuda" 的设备类型来实例化 TestHelpers 测试类的设备类型测试
instantiate_device_type_tests(
    TestHelpers,
    globals(),
    only_for=only_for,
)

# 实例化 TestMakeFunctional 测试类的参数化测试
instantiate_parametrized_tests(
    TestMakeFunctional,
)

# 使用只适用于 "cpu" 和 "cuda" 的设备类型来实例化 TestCompileTransforms 测试类的设备类型测试
instantiate_device_type_tests(
    TestCompileTransforms,
    globals(),
    only_for=only_for,
)

# 如果当前脚本作为主程序运行，则执行所有测试
if __name__ == "__main__":
    run_tests()
```