# `.\pytorch\test\test_fake_tensor.py`

```
# 导入所需的模块和库
import contextlib                 # 提供上下文管理工具的模块
import copy                       # 提供对象复制功能的模块
import dataclasses                # 支持数据类的模块
import inspect                    # 提供有关活动对象的信息的模块
import itertools                  # 提供用于创建和操作迭代器的函数的模块
import pickle                     # 提供 Python 对象的序列化和反序列化的功能
import unittest                   # Python 内置的单元测试框架
import weakref                    # 提供弱引用对象的支持的模块
from unittest.mock import patch  # 提供模拟和打桩功能的模块

import numpy as np                # 数值计算库，支持大量的维度数组与矩阵运算
import torch                      # PyTorch 深度学习框架
import torch._dynamo             # PyTorch 的私有模块
import torch._functorch.config    # PyTorch Functorch 的配置模块
import torch._prims as prims      # PyTorch 私有模块，提供了一些基本操作的原语
import torch.testing._internal.optests as optests  # PyTorch 内部测试模块
import torch.utils._pytree as pytree  # PyTorch 内部的树形数据结构工具

from torch import distributed as dist  # PyTorch 分布式模块
from torch._C._functorch import _add_batch_dim, get_unwrapped, is_batchedtensor  # PyTorch Functorch 的私有模块
from torch._dynamo.testing import make_test_cls_with_patches, rand_strided  # PyTorch 私有测试模块
from torch._guards import tracing, TracingContext  # PyTorch 保护模块，提供追踪和上下文保护功能
from torch._subclasses.fake_tensor import (  # PyTorch 私有模块，提供了假张量相关的类和异常
    DynamicOutputShapeException,
    extract_tensor_metadata,
    FakeTensor,
    FakeTensorConverter,
    FakeTensorMode,
    unset_fake_temporarily,
    UnsupportedOperatorException,
)
from torch.fx.experimental.proxy_tensor import make_fx  # PyTorch FX 实验性模块，提供了代理张量的功能
from torch.fx.experimental.symbolic_shapes import (  # PyTorch FX 实验性模块，提供了符号形状相关的类和函数
    DimDynamic,
    free_symbols,
    ShapeEnv,
    ShapeEnvSettings,
    StatelessSymbolicContext,
    statically_known_true,
)
from torch.fx.passes.fake_tensor_prop import FakeTensorProp  # PyTorch FX 模块，提供了假张量传播的功能
from torch.testing import FileCheck  # PyTorch 测试模块，提供文件检查的工具
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION  # PyTorch 内部 CUDA 相关模块
from torch.testing._internal.common_device_type import (  # PyTorch 内部设备类型模块
    instantiate_device_type_tests,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_utils import (  # PyTorch 内部通用工具模块
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfCrossRef,
    skipIfRocm,
    skipIfTorchDynamo,
    TemporaryFileName,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)

# 获取 PyTorch ATen 操作的引用
aten = torch.ops.aten

# 启用假张量缓存和跨检查功能
torch._dynamo.config.fake_tensor_cache_enabled = True
torch._dynamo.config.fake_tensor_cache_crosscheck_enabled = True

# 定义一个装饰器，标记预期失败会传播真实张量的测试函数
def expectedFailurePropagateRealTensors(fn):
    fn._expected_failure_propagate_real_tensors = True
    return fn

# FakeTensorTest 类，继承自 TestCase 类，用于假张量的测试
class FakeTensorTest(TestCase):
    
    # 检查张量类型的方法
    def checkType(self, t, device_str, size):
        self.assertTrue(isinstance(t, FakeTensor))  # 断言 t 是 FakeTensor 类的实例
        self.assertEqual(t.device.type, device_str)  # 断言 t 的设备类型为 device_str
        self.assertEqual(list(t.size()), size)  # 断言 t 的尺寸为 size

    # CUDA 初始化测试，当 CUDA 可用时执行
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_cuda_initialized(self):
        # 不会引发错误
        with FakeTensorMode():  # 使用假张量模式
            p = torch.randn(4, 2, requires_grad=True, device="cuda")  # 在 CUDA 设备上创建随机张量 p
            x = torch.randn(8, 4, device="cuda")  # 在 CUDA 设备上创建随机张量 x
            y = torch.mm(x, p).square().sum()  # 执行矩阵乘法并计算平方和
            y.backward()  # 反向传播
    # 定义一个测试方法，测试基本的张量操作
    def test_basic(self):
        # 创建一个空的 2x2 张量在 CPU 上
        x = torch.empty(2, 2, device="cpu")
        # 创建一个空的 4x2x2 张量在 CPU 上
        y = torch.empty(4, 2, 2, device="cpu")
        # 使用 FakeTensorMode 上下文管理器，模拟张量操作
        with FakeTensorMode() as mode:
            # 将普通张量 x 转换为 FakeTensor
            x = mode.from_tensor(x)
            # 将普通张量 y 转换为 FakeTensor
            y = mode.from_tensor(y)
            # 对两个 FakeTensor 进行加法操作
            z = x + y
            # 断言加法结果的形状为 (4, 2, 2)
            self.assertEqual(z.shape, (4, 2, 2))
            # 断言加法结果的设备为 CPU
            self.assertEqual(z.device, torch.device("cpu"))
            # 断言 z 是 FakeTensor 的实例
            self.assertTrue(isinstance(z, FakeTensor))

    # 定义测试自定义操作的回退情况
    def test_custom_op_fallback(self):
        # 导入相关库和模块
        from torch.library import impl, Library
        
        try:
            # 创建一个自定义库对象 my_test_op
            test_lib = Library("my_test_op", "DEF")  # noqa: TOR901
            # 定义一个 foo 操作
            test_lib.define("foo(Tensor self) -> Tensor")
            
            # 实现 foo 操作的具体逻辑
            @impl(test_lib, "foo", "CPU")
            def foo_impl(self):
                return self.cos()
            
            # 创建一个空的 2x2 张量在 CPU 上
            x = torch.empty(2, 2, device="cpu")
            # 在 FakeTensorMode 下，测试使用自定义操作
            with self.assertRaisesRegex(
                UnsupportedOperatorException, "my_test_op.foo.default"
            ):
                with FakeTensorMode(allow_fallback_kernels=True) as mode:
                    # 将普通张量 x 转换为 FakeTensor
                    x = mode.from_tensor(x)
                    # 调用自定义操作 foo
                    torch.ops.my_test_op.foo(x)
        
        finally:
            # 清理 test_lib 资源
            test_lib._destroy()

    # 定义测试参数实例化的方法
    def test_parameter_instantiation(self):
        # 在 FakeTensorMode 下
        with FakeTensorMode():
            # 创建一个随机张量 x
            x = torch.rand([4])
            # 将 x 转换为 Parameter 对象 y
            y = torch.nn.parameter.Parameter(x)
            # 断言 y 是 torch.nn.Parameter 的实例
            self.assertTrue(isinstance(y, torch.nn.Parameter))

    # 如果分布式环境可用，测试分布式 FSDP 的平坦参数
    @unittest.skipIf(not dist.is_available(), "requires distributed")
    def test_fsdp_flat_param(self):
        # 导入相关模块
        from torch.distributed.fsdp._flat_param import FlatParameter
        
        # 在 FakeTensorMode 下
        with FakeTensorMode() as m:
            # 创建一个随机数据张量
            data = torch.randn(2, 2)
            # 创建一个 FlatParameter 对象 param
            param = FlatParameter(data, requires_grad=True)
        # 断言 param 是 FlatParameter 的实例
        self.assertIsInstance(param, FlatParameter)
        # 断言 param 同时是 torch.nn.Parameter 的实例
        self.assertIsInstance(param, torch.nn.Parameter)
        # 断言 param 同时是 FakeTensor 的实例
        self.assertIsInstance(param, FakeTensor)

    # 测试非参数梯度的情况
    def test_non_parameter_grad(self):
        # 创建一个 FakeTensorMode 实例
        mode = FakeTensorMode()
        # 创建一个随机张量 t，需要梯度
        t = torch.rand([4], requires_grad=True)
        # 将普通张量 t 转换为 FakeTensor
        fake_t = mode.from_tensor(t)
        # 断言 fake_t 的 requires_grad 属性与 t 相同
        self.assertEqual(fake_t.requires_grad, t.requires_grad)

    # 如果 CUDA 可用，测试在 CPU 上索引 CUDA 张量的情况
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_index_cuda_with_cpu(self):
        # 在 FakeTensorMode 下
        with FakeTensorMode():
            # 创建一个在 CUDA 上的随机张量 x
            x = torch.rand([2048], device="cuda")
            # 使用 CPU 上的索引对 x 进行切片操作
            out = x[torch.zeros([36], dtype=torch.int64)]
            # 检查输出 out 的类型为 CUDA 张量
            self.checkType(out, "cuda", [36])

    # 如果 CUDA 可用，测试在 CPU 上调整形状的情况
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_shape_take_not_device(self):
        # 在 FakeTensorMode 下
        with FakeTensorMode():
            # 创建一个空的 1 维张量在 CPU 上
            x = torch.empty(1, device="cpu")
            # 创建一个空的 8x8 张量在 CUDA 上
            y = torch.empty(8, 8, device="cuda")
            # 调整 x 的形状与 y 相同，并替换原来的张量
            out = x.resize_as_(y)
            # 断言 out 的形状为 (8, 8)
            self.assertEqual(out.shape, (8, 8))
            # 断言 out 的设备类型为 CPU
            self.assertEqual(out.device.type, "cpu")
            # 断言 out 是 FakeTensor 的实例
            self.assertTrue(isinstance(out, FakeTensor))
    # 定义测试函数 test_repr，用于测试对象的字符串表示形式
    def test_repr(self):
        # 进入 FakeTensorMode 上下文环境
        with FakeTensorMode():
            # 创建一个空的 Torch 张量，设备为 CPU
            x = torch.empty(2, 2, device="cpu")
            # 断言张量的字符串表示形式符合预期
            self.assertEqual(repr(x), "FakeTensor(..., size=(2, 2))")
            # 创建一个空的 Torch 张量，设备为 meta
            x = torch.empty(2, 2, device="meta")
            # 断言张量的字符串表示形式符合预期，包含设备信息
            self.assertEqual(repr(x), "FakeTensor(..., device='meta', size=(2, 2))")

    # 标记为跳过测试，如果未启用 CUDA
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义测试函数 test_zero_dim，测试零维张量的行为
    def test_zero_dim(self):
        # 进入 FakeTensorMode 上下文环境，并将其赋值给 mode
        with FakeTensorMode() as mode:
            # 创建一个值为 0.0 的 Torch 张量
            x = torch.tensor(0.0)
            # 创建一个随机初始化的 Torch 张量，设备为 cuda
            y = torch.rand([4, 4], device="cuda")
            # 执行张量加法运算
            out = x + y
            # 断言输出张量的形状为 (4, 4)
            self.assertEqual(out.shape, (4, 4))
            # 断言输出张量的设备与 y 相同
            self.assertEqual(out.device, y.device)
            # 断言输出张量属于 FakeTensor 类型
            self.assertTrue(isinstance(out, FakeTensor))

    # 定义测试函数 test_nan_to_num，测试 torch.nan_to_num 函数的行为
    def test_nan_to_num(self):
        # 进入 FakeTensorMode 上下文环境
        with FakeTensorMode():
            # 遍历指定的浮点数数据类型列表
            for dtype in [torch.float16, torch.float32]:
                # 创建一个指定数据类型和形状的随机初始化 Torch 张量
                x = torch.rand([4], dtype=dtype)
                # 将 x 中的 NaN 替换为 None
                y = torch.nan_to_num(x, nan=None)
                # 将 x 中的 NaN 替换为 0.0
                z = torch.nan_to_num(x, 0.0)
                # 断言 y 的数据类型与指定的数据类型相同
                self.assertEqual(dtype, y.dtype)
                # 断言 z 的数据类型与指定的数据类型相同
                self.assertEqual(dtype, z.dtype)

    # 标记为跳过测试，如果未启用 CUDA
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义测试函数 test_throw，测试在 FakeTensorMode 下的异常处理行为
    def test_throw(self):
        # 创建一个值为 0.0 的 Torch 张量，并添加 TODO 注释
        x = torch.tensor(0.0)  # TODO: tensor() errors
        # 进入 FakeTensorMode 上下文环境，并将其赋值给 mode
        with FakeTensorMode() as mode:
            # 将 x 转换为 FakeTensor 对象
            x_conv = mode.from_tensor(x)
            # 创建一个随机初始化的 Torch 张量，设备为 cuda
            y = torch.rand([4, 4], device="cuda")
            # 创建一个随机初始化的 Torch 张量，设备为 cpu
            z = torch.rand([4, 4], device="cpu")
            # 断言 torch.lerp 操作引发异常
            self.assertRaises(Exception, lambda: torch.lerp(x_conv, y, z))

    # 标记为跳过测试，如果未启用 CUDA
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义测试函数 test_type_as，测试 torch.Tensor.type_as 方法的行为
    def test_type_as(self):
        # 进入 FakeTensorMode 上下文环境
        with FakeTensorMode():
            # 创建一个随机初始化的 Torch 张量，形状为 [16, 1]，设备为 cpu
            x = torch.rand([16, 1], device="cpu")
            # 创建一个随机初始化的 Torch 张量，形状为 [4, 4]，设备为 cuda
            y = torch.rand([4, 4], device="cuda")
            # 将 x 转换为与 y 相同设备类型的张量
            out = x.type_as(y)
            # 断言 out 的设备类型为 cuda
            self.assertEqual(out.device.type, "cuda")
            # 断言 out 属于 FakeTensor 类型
            self.assertTrue(isinstance(out, FakeTensor))

    # 标记为跳过测试，如果未启用 CUDA
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义测试函数 test_setitem，测试 Torch 张量的切片赋值操作
    def test_setitem(self):
        # 遍历设备列表，包括 cpu 和 cuda
        for device in ["cpu", "cuda"]:
            # 进入 FakeTensorMode 上下文环境
            with FakeTensorMode():
                # 创建一个随机初始化的 Torch 张量，形状为 [16, 1]，设备为指定的 device
                x = torch.rand([16, 1], device=device)
                # 对 x 的所有行的第一列进行赋值操作，设置为 0
                x[..., 0] = 0

    # 标记为跳过测试，如果未启用 CUDA
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义测试函数 test_device_inplace_copy，测试 Torch 张量的 in-place 复制操作
    def test_device_inplace_copy(self):
        # 进入 FakeTensorMode 上下文环境
        with FakeTensorMode():
            # 创建一个随机初始化的 Torch 张量，形状为 [8, 8]，设备为 cpu
            x = torch.rand([8, 8], device="cpu")
            # 创建一个随机初始化的 Torch 张量，形状为 [8, 8]，设备为 cuda
            y = torch.rand([8, 8], device="cuda")
            # 断言 x 在复制 y 的同时，其设备类型为 cpu
            assert x.copy_(y).device.type == "cpu"
            # 断言 y 在复制 x 的同时，其设备类型为 cuda
            assert y.copy_(x).device.type == "cuda"
    def test_fake_dispatch_keys(self):
        # 使用 FakeTensorMode 上下文环境进行测试
        with FakeTensorMode():
            # 创建一个形状为 [4] 的随机张量 x
            x = torch.rand([4])
            # 创建一个 FileCheck 对象 f，并设置检查条件
            f = (
                FileCheck()
                .check("CPU")
                .check("ADInplaceOrView")
                .check("AutogradCPU")
                .check("AutocastCPU")
            )
            # 运行 FileCheck 对象 f，检查 x 的分发键集合
            f.run(torch._C._dispatch_key_set(x))

            # 进入 torch 推断模式上下文环境
            with torch.inference_mode():
                # 创建一个形状为 [4] 的随机张量 x
                x = torch.rand([4])
                # 计算 x + x 得到 y
                y = x + x
                # 使用 FileCheck 对象检查 y 的 CPU 和 AutocastCPU 分发键集合
                FileCheck().check("CPU").check("AutocastCPU").run(
                    torch._C._dispatch_key_set(y)
                )
                # 使用 FileCheck 对象检查 y，排除 ADInplaceOrView 和 Autograd 分发键集合
                FileCheck().check_not("ADInplaceOrView").check_not("Autograd").run(
                    torch._C._dispatch_key_set(y)
                )

    def test_batch_tensor(self):
        # 创建一个形状为 (3, 4, 5) 的随机张量 x
        x = torch.rand((3, 4, 5))
        # 将 x 添加批次维度，返回新的张量 b
        b = _add_batch_dim(x, 0, 0)
        # 创建 FakeTensorMode 上下文模式对象 mode
        mode = FakeTensorMode()
        # 使用 mode 将 b 转换成伪张量 fake_b
        fake_b = mode.from_tensor(b)
        # 比较张量 b 和 fake_b 的元数据，包括检查步长是否一致
        prims.utils.compare_tensor_meta(b, fake_b, check_strides=True)

        # 将 x 添加两个批次维度，返回新的张量 b2
        b1 = _add_batch_dim(x, 1, 1)
        b2 = _add_batch_dim(b1, 0, 2)
        # 使用 mode 将 b2 转换成伪张量 fake_b2
        fake_b2 = mode.from_tensor(b2)
        # 比较张量 b2 和 fake_b2 的元数据，包括检查步长是否一致
        prims.utils.compare_tensor_meta(b2, fake_b2, check_strides=True)
        # 断言 fake_b2 是否为批次张量
        self.assertTrue(is_batchedtensor(fake_b2))
        # 获取 fake_b2 的内部张量 fake_b1
        fake_b1 = get_unwrapped(fake_b2)
        # 断言 fake_b1 是否为批次张量
        self.assertTrue(is_batchedtensor(fake_b1))
        # 获取 fake_b1 的内部张量 fake_tensor
        fake_tensor = get_unwrapped(fake_b1)
        # 断言 fake_tensor 是 FakeTensor 类型的对象
        self.assertIsInstance(fake_tensor, FakeTensor)

    def test_constructor(self):
        # 使用 FakeTensorMode 上下文环境创建形状为 [4, 4] 的随机张量 x，设备为 CPU
        with FakeTensorMode():
            x = torch.rand([4, 4], device="cpu")

        # 断言 x 是 FakeTensor 类型的对象
        self.assertTrue(isinstance(x, FakeTensor))
        # 断言 x 的设备类型是 CPU
        self.assertTrue(x.device.type == "cpu")

    def test_mode(self):
        # 使用 FakeTensorMode 上下文环境创建形状为 [4] 的随机张量 y，设备为 CPU
        with FakeTensorMode():
            y = torch.rand([4], device="cpu")
            # 计算 y + y 得到 out
            out = y + y

        # 断言 out 是 FakeTensor 类型的对象
        self.assertTrue(isinstance(out, FakeTensor))

    def test_full(self):
        # 测试 torch.full 返回具有正确数据类型的张量
        with torch._subclasses.CrossRefFakeMode():
            # 创建形状为 (4, 4) 的张量 y，每个元素值为 1
            y = torch.full((4, 4), 1)

    def check_function_with_fake(self, fn):
        # 调用给定函数 fn，获取其返回值 out
        out = fn()
        # 使用 torch._subclasses.FakeTensorMode 上下文环境
        with torch._subclasses.FakeTensorMode():
            # 再次调用函数 fn，获取其返回值 out_fake
            out_fake = fn()

        # 遍历 out 和 out_fake 的叶子节点，并比较其张量元数据，包括检查步长是否一致
        for a, b in zip(pytree.tree_leaves(out), pytree.tree_leaves(out_fake)):
            if not isinstance(a, torch.Tensor):
                # 如果 a 不是 torch.Tensor 类型，断言 b 也不是 torch.Tensor 类型
                self.assertTrue(not isinstance(b, torch.Tensor))
                continue

            # 比较张量 a 和 b 的元数据，包括检查步长是否一致
            prims.utils.compare_tensor_meta(a, b, check_strides=True)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_non_kwarg_device(self):
        # 使用 FakeTensorMode 上下文环境创建形状为 [16, 1] 的随机张量 x，设备为 CPU
        with FakeTensorMode():
            x = torch.rand([16, 1], device="cpu")
            # 将 x 转换到设备为 CPU 的张量 y
            y = x.to(torch.device("cpu"))
            # 断言 x 和 y 是同一个张量对象
            self.assertIs(x, y)
            # 将 x 转换到设备为 CUDA 的张量 z
            z = x.to(torch.device("cuda"))
            # 断言 z 的设备类型是 CUDA
            self.assertEqual(z.device.type, "cuda")

    def test_non_overlapping_stride_zero(self):
        # 定义函数 foo
        def foo():
            # 创建步长不重叠的形状为 [1, 3, 427, 640] 的空张量 x
            x = torch.empty_strided([1, 3, 427, 640], (0, 1, 1920, 3))
            # 将 x 转换成半精度张量，并返回结果
            return x.half()

        # 使用 check_function_with_fake 函数检查 foo 函数
        self.check_function_with_fake(foo)
    # 定义测试方法，用于测试在伪张量模式下引发错误
    def test_fake_mode_error(self):
        # 创建一个形状为 [4, 4] 的随机张量 x
        x = torch.rand([4, 4])

        # 使用断言检查是否引发异常，并验证异常消息
        with self.assertRaisesRegex(Exception, "Please convert all Tensors"):
            # 进入伪张量模式
            with FakeTensorMode():
                # 尝试访问张量 x 的第一个元素，此处应该引发异常
                y = x[0]

    # 根据条件跳过测试，如果使用 Torch Dynamo，则跳过测试
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    # 定义测试方法，用于测试伪梯度复制功能
    def test_fake_grad_copy(self):
        # 创建一个形状为 [4, 4] 的随机张量 x，并设置 requires_grad=True
        x = torch.rand([4, 4], requires_grad=True)
        # 创建一个形状为 [4, 4] 的随机张量作为梯度值，并赋给 x.grad
        x.grad = torch.rand([4, 4])
        # 创建伪张量模式对象
        mode = FakeTensorMode()
        # 从真实张量 x 转换为伪张量 fake_x
        fake_x = mode.from_tensor(x)
        # 使用 prims.utils.compare_tensor_meta 比较 fake_x 和 x 的元数据
        prims.utils.compare_tensor_meta(fake_x, x)
        # 使用 prims.utils.compare_tensor_meta 比较 fake_x.grad 和 x.grad 的元数据
        prims.utils.compare_tensor_meta(fake_x.grad, x.grad)

        # 使用断言检查 fake_x.grad 是否为 FakeTensor 类型
        self.assertTrue(isinstance(fake_x.grad, FakeTensor))

    # 根据条件跳过测试，如果未运行 CUDA，则跳过测试
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义测试方法，用于测试索引赋值时的错误情况
    def test_index_put_error(self):
        # 创建伪张量模式对象
        mode = FakeTensorMode()
        # 使用 contextlib.nullcontext 创建上下文管理器，对应的代码块为空
        for context in [contextlib.nullcontext, lambda: mode]:
            # 进入上下文管理器
            with context():
                # 创建形状为 [2, 2, 3] 的随机张量 y
                y = torch.randn(2, 2, 3)
                # 创建形状为 [2, 2, 3] 的随机张量 x，并将其移动到 CUDA 设备
                x = torch.randn(2, 2, 3).to("cuda")
                # 使用断言检查在索引赋值时是否引发 RuntimeError 异常
                with self.assertRaises(RuntimeError):
                    x[[1, 1]] = y

                # 使用断言检查在索引赋值时是否引发 RuntimeError 异常
                with self.assertRaises(RuntimeError):
                    torch.ops.aten.index_put(x, torch.tensor([1, 1], device="cuda"), y)

                # 没有错误发生的情况下执行下列索引赋值操作
                torch.ops.aten.index_put(
                    x, torch.tensor([1, 1], device="cuda"), torch.tensor(5.0)
                )
                torch.ops.aten.index_put_(
                    x, torch.tensor([1, 1], device="cuda"), torch.tensor(5.0)
                )

    # 根据条件跳过测试，如果未运行 CUDA，则跳过测试
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义测试方法，用于测试 torch.ones_like 构造函数的行为
    def test_like_constructor(self):
        # 进入伪张量模式
        with FakeTensorMode():
            # 创建形状为 [4, 4] 的随机张量 x
            x = torch.rand([4, 4])
            # 使用 torch.ones_like 生成一个形状与 x 相同且元素全为 1 的张量 y
            y = torch.ones_like(x)
            # 使用断言检查 y 是否为 FakeTensor 类型
            self.assertTrue(isinstance(y, FakeTensor))
            # 使用断言检查 y 的设备类型是否为 "cpu"
            self.assertEqual(y.device.type, "cpu")
            # 使用 torch.ones_like 在 CUDA 设备上生成一个形状与 x 相同且元素全为 1 的张量 z
            z = torch.ones_like(x, device="cuda")
            # 使用断言检查 z 是否为 FakeTensor 类型
            self.assertTrue(isinstance(z, FakeTensor))
            # 使用断言检查 z 的设备类型是否为 "cuda"
            self.assertEqual(z.device.type, "cuda")

    # 定义测试方法，用于测试二元操作中类型提升的行为
    def test_binary_op_type_promotion(self):
        # 进入伪张量模式
        with FakeTensorMode():
            # 创建一个形状为 [2, 2]、dtype 为 torch.float 的空张量 x
            x = torch.empty([2, 2], dtype=torch.float)
            # 创建一个形状为 [2, 2]、dtype 为 torch.int64 的空张量 y
            y = torch.empty([2, 2], dtype=torch.int64)
            # 执行 x / y 的二元操作，得到结果张量 out
            out = x / y
            # 使用断言检查 out 的 dtype 是否为 torch.float
            self.assertEqual(out.dtype, torch.float)
            # 使用断言检查 out 的设备类型是否为 "cpu"
            self.assertEqual(out.device.type, "cpu")

    # 根据条件跳过测试，如果使用 Torch Dynamo，则跳过测试
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    # 定义测试方法，用于测试从 numpy 数组创建张量的行为
    def test_from_numpy(self):
        # 进入伪张量模式
        with FakeTensorMode():
            # 使用 numpy 创建一个形状为 [4, 4] 的零张量，并将其转换为 torch.tensor
            x = torch.tensor(np.zeros([4, 4]))
            # 调用辅助函数 checkType，验证 x 的设备类型为 "cpu"、形状为 [4, 4]
            self.checkType(x, "cpu", [4, 4])

    # 定义测试方法，用于测试 torch.randperm 函数的行为
    def test_randperm(self):
        # 创建一个长度为 10 的随机排列张量 x
        x = torch.randperm(10)
        # 创建一个长度为 5 的随机排列张量 y，并将其放置在 "cpu" 设备上
        y = torch.randperm(5, device="cpu")
        # 进入伪张量模式
        with FakeTensorMode():
            # 创建一个长度为 10 的随机排列张量 x1
            x1 = torch.randperm(10)
            # 使用 prims.utils.compare_tensor_meta 比较 x 和 x1 的元数据
            prims.utils.compare_tensor_meta(x, x1)
            # 创建一个长度为 5 的随机排列张量 y1，并将其放置在 "cpu" 设备上
            y1 = torch.randperm(5, device="cpu")
            # 使用 prims.utils.compare_tensor_meta 比较 y 和 y1 的元数据
            prims.utils.compare_tensor_meta(y, y1)
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义一个测试函数，用于测试在假张量模式下的打印行为
    def test_print_in_fake_mode(self):
        # 创建一个形状为 (2,) 的全零张量 x
        x = torch.zeros(2)
        # 在 FakeTensorMode 上下文中执行，确保转换为字符串不会失败
        with FakeTensorMode():
            # 将张量 x 转换为字符串
            out = str(x)
        # 断言输出中不包含 "FakeTensor"
        assert "FakeTensor" not in out

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义一个测试函数，用于测试小通道数情况下的双线性上采样
    def test_upsample_bilinear_small_channels(self):
        # 初始化一个空列表 out
        out = []
        # 创建一个 FakeTensorMode 实例 mode
        mode = FakeTensorMode()
        # 遍历两种上下文环境：正常环境和 mode 所定义的环境
        for i, context in enumerate([contextlib.nullcontext, lambda: mode]):
            with context():
                # 创建一个形状为 (3, 427, 640) 的带有步幅的空张量 arg0_1，在 CUDA 设备上
                arg0_1 = torch.empty_strided(
                    (3, 427, 640), (1, 1920, 3), dtype=torch.float32, device="cuda"
                )
                # 对 arg0_1 进行 unsqueeze 操作，添加一个维度
                unsqueeze = torch.ops.aten.unsqueeze.default(arg0_1, 0)
                # 执行双线性 2D 上采样操作
                out.append(
                    torch.ops.aten.upsample_bilinear2d.default(
                        unsqueeze, [800, 1199], False
                    )
                )

        # 断言 out[1] 是连续的张量
        self.assertTrue(out[1].is_contiguous())
        # 检查 out[0] 和 out[1] 的元数据属性
        self.checkMetaProps(out[0], out[1])

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义一个测试函数，用于测试 CPU 回退情况
    def test_cpu_fallback(self):
        # 在不允许回退内核的 FakeTensorMode 下执行
        with FakeTensorMode(allow_fallback_kernels=False):
            # 创建形状为 (8, 4, 3, 3) 的随机张量 filters，并放置在 CUDA 设备上
            filters = torch.randn(8, 4, 3, 3).cuda()
            # 创建形状为 (1, 4, 5, 5) 的随机张量 inputs，并放置在 CUDA 设备上
            inputs = torch.randn(1, 4, 5, 5).cuda()
            # 执行 2D 卷积操作
            out = torch.nn.functional.conv2d(inputs, filters, padding=1)
            # 断言输出张量的设备类型为 "cuda"
            self.assertEqual(out.device.type, "cuda")
            # 断言输出张量的形状为 [1, 8, 5, 5]

        # 在允许回退内核的 FakeTensorMode 下执行
        with FakeTensorMode(allow_fallback_kernels=True):
            # 故意使用不良输入创建 filters 张量
            filters = torch.randn(8, 20, 3, 3).cuda()
            # 故意使用不良输入创建 inputs 张量
            inputs = torch.randn(1, 7, 10, 5).cuda()
            # 断言抛出 RuntimeError 异常
            with self.assertRaises(RuntimeError):
                torch.nn.functional.conv2d(inputs, filters, padding=1)

        # 再次在允许回退内核的 FakeTensorMode 下执行
        with FakeTensorMode(allow_fallback_kernels=True):
            # 创建形状为 (8, 4, 3, 3) 的随机张量 filters，并放置在 CUDA 设备上
            filters = torch.randn(8, 4, 3, 3).cuda()
            # 创建形状为 (1, 4, 5, 5) 的随机张量 inputs，并放置在 CUDA 设备上
            inputs = torch.randn(1, 4, 5, 5).cuda()
            # 执行 2D 卷积操作
            out = torch.nn.functional.conv2d(inputs, filters, padding=1)
            # 断言输出张量的设备类型为 "cuda"
            self.assertEqual(out.device.type, "cuda")
            # 断言输出张量的形状为 [1, 8, 5, 5]

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义一个测试函数，用于测试多设备输出
    def test_out_multi_device(self):
        # 在 FakeTensorMode 下执行
        with FakeTensorMode():
            # 创建一个形状为 [4] 的随机张量 x
            x = torch.rand([4])
            # 创建一个形状为 [4] 的随机张量 y，并放置在 CUDA 设备上
            y = torch.rand([4], device="cuda")

            # 断言执行 torch.sin 操作时抛出异常，指示找到了两个不同设备的张量
            with self.assertRaisesRegex(Exception, "found.+two.+devices"):
                torch.sin(x, out=y)

            # 断言执行 x.add_(y) 操作时抛出异常，指示找到了两个不同设备的张量
            with self.assertRaisesRegex(Exception, "found.+two.+devices"):
                x.add_(y)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义一个测试函数，用于测试张量在 FakeTensorMode 下的正常化行为
    def test_normalize_device(self):
        # 在 FakeTensorMode 下执行
        with FakeTensorMode():
            # 创建一个形状为 [1] 的空张量 x，并放置在 CUDA 设备上
            x = torch.empty(1, device="cuda")
            # 创建一个与当前 CUDA 设备对应的形状为 [1] 的空张量 y
            y = torch.empty(1, device=f"cuda:{torch.cuda.current_device()}")
            # 执行张量加法操作
            out = x + y
        # 检查 out 张量的类型和设备信息
        self.checkType(out, "cuda", [1])
    # 定义一个测试方法，用于测试递归调用
    def test_recursive_invocation(self):
        # 创建一个 FakeTensorMode 的实例
        mode = FakeTensorMode()
        # 进入模式上下文，确保资源被正确释放
        with mode:
            # 创建一个值为2的张量
            x = torch.tensor(2)
            # 设置模式中的内核调用标志为真
            mode.in_kernel_invocation = True
            # 对张量进行加法运算
            y = x + x
            # 断言模式中的内核调用标志为真
            self.assertTrue(mode.in_kernel_invocation)

    # 如果使用 TorchDynamo 进行测试，则跳过该测试
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    # 如果在 ROCm 平台下，则跳过该测试
    @skipIfRocm
    # 参数化装饰器，测试不同的参数配置
    @parametrize(
        "allow_fallback_kernels",
        [False, True],
        lambda a: "with_fallback" if a else "without_fallback",
    )
    # 如果没有 CUDA 支持，则跳过该测试
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义一个测试方法，用于测试 cudnn_rnn 函数，接受一个布尔值参数 allow_fallback_kernels
    def test_cudnn_rnn(self, allow_fallback_kernels):
        # 定义一个内部函数 fn，接受多个参数 a0 到 a5 和 b0 到 b15
        def fn(
            a0,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            b8,
            b9,
            b10,
            b11,
            b12,
            b13,
            b14,
            b15,
            a3,
            a4,
            a5,
        ):
            # 将 b0 到 b15 放入列表 a1 中
            a1 = [
                b0,
                b1,
                b2,
                b3,
                b4,
                b5,
                b6,
                b7,
                b8,
                b9,
                b10,
                b11,
                b12,
                b13,
                b14,
                b15,
            ]
            # 调用 torch.ops.aten._cudnn_rnn 函数，传入多个参数，返回计算结果
            return torch.ops.aten._cudnn_rnn(
                a0,
                a1,
                4,
                a3,
                a4,
                a5,
                2,
                2048,
                0,
                2,
                False,
                0.0,
                False,
                True,
                [],
                None,
            )

        # 创建一个 FakeTensorMode 对象 mode，用于控制测试中的模拟环境
        mode = FakeTensorMode(allow_fallback_kernels=allow_fallback_kernels)
        # 遍历两个上下文管理器，contextlib.nullcontext 和 mode 对象的 lambda 表达式
        for i, context in enumerate([contextlib.nullcontext, lambda: mode]):
            # 使用当前的上下文管理器进行测试环境的设置
            with context():
                # 创建输入张量列表 inps1 和 inps2，每个张量都使用 torch.randn 生成，并移到 GPU 上
                inps1 = [
                    torch.randn([92, 8, 2048]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192, 4096]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192, 4096]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([167837696]).cuda(),
                    torch.randn([4, 8, 2048]).cuda(),
                    torch.randn([4, 8, 2048]).cuda(),
                ]
                # 复制 inps1 到 inps2，并将 inps2 的最后一个元素设置为 None，用于测试的特定情况
                inps2 = inps1
                inps2[len(inps2) - 1] = None  # 参数 `cx` 可以为 None

                # 对于 inps1 和 inps2 中的每一个列表进行测试
                for inps in [inps1, inps2]:
                    # 调用 fn 函数，传入 inps 列表中的所有参数，获取返回结果 out
                    out = fn(*inps)
                    # 断言 out 的第五个元素等于 inps 倒数第三个元素
                    self.assertIs(out[4], inps[-3])
                    # 对于 out 中的每个张量 ten 进行遍历
                    for ten in out:
                        # 如果 i 为 1，则断言 ten 是 FakeTensor 类型
                        if i == 1:
                            self.assertTrue(isinstance(ten, FakeTensor))
                        # 断言 ten 的设备类型为 "cuda"
                        self.assertEqual(ten.device.type, "cuda")

    # 根据是否支持 CUDA，跳过测试（如果不支持 CUDA 则跳过）
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 测试 CUDA 上的 LSTM 实现，使用虚拟张量确保非 cuDNN 实现成功运行。
    def test_cuda_lstm(self):
        # 禁用 cuDNN 后的上下文
        with torch.backends.cudnn.flags(enabled=False):
            # 使用 FakeTensorMode，禁止回退内核
            fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=False)
            with fake_tensor_mode:
                # 定义 LSTM 的参数
                N = 5   # batch size
                L = 4   # sequence length
                H_in = 2    # 输入特征大小
                hidden_size = 3    # 隐藏层大小
                proj_size = 2   # 投影层大小
                num_layers = 2  # LSTM 层数
                bidir = False   # 是否双向
                D = 2 if bidir else 1   # 方向乘积
                H_out = proj_size if proj_size > 0 else hidden_size   # 输出大小取决于是否有投影层

                # 创建 LSTM 模型
                lstm = torch.nn.LSTM(
                    input_size=H_in,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    proj_size=proj_size,
                    batch_first=False,
                    bias=True,
                    bidirectional=bidir,
                    device="cuda",
                )

                # 初始化 LSTM 的隐藏状态和细胞状态
                h_0 = torch.randn((num_layers * D, N, H_out), device="cuda")
                c_0 = torch.randn((num_layers * D, N, hidden_size), device="cuda")
                # 创建输入张量
                inp = torch.randn((L, N, H_in), device="cuda")
                # 执行 LSTM 计算
                (output, (h_n, c_n)) = lstm(inp, (h_0, c_0))
                # 对输出进行反向传播
                output.sum().backward()

                # 断言输出的形状符合预期
                self.assertEqual(output.shape, (L, N, D * H_out))
                self.assertEqual(h_n.shape, (D * num_layers, N, H_out))
                self.assertEqual(c_n.shape, (D * num_layers, N, hidden_size))

    # 测试依赖数据的运算符
    def test_data_dependent_operator(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            # 创建随机张量 x
            x = torch.rand([10, 10])
            # 断言 torch.nonzero 在此模式下会引发 DynamicOutputShapeException 异常
            self.assertRaises(DynamicOutputShapeException, lambda: torch.nonzero(x))

    # 测试参数视图
    def test_parameter_view(self):
        # 创建参数张量 x
        x = torch.nn.Parameter(torch.randn(4))
        # 对参数进行视图变换
        x_view = x.view(4)
        # 创建 FakeTensorMode 实例
        mode = FakeTensorMode()
        # 将视图转换为模拟张量
        fake_x_view = mode.from_tensor(x_view)
        # 将参数转换为模拟张量
        fake_x = mode.from_tensor(x)
        # 断言视图不是 torch.nn.Parameter 类型
        self.assertFalse(isinstance(fake_x_view, torch.nn.Parameter))
        # 断言参数是 torch.nn.Parameter 类型
        self.assertTrue(isinstance(fake_x, torch.nn.Parameter))

    # 测试 tolist 方法
    def test_tolist(self):
        # 创建 ShapeEnv 实例
        shape_env = ShapeEnv()
        with FakeTensorMode(allow_fallback_kernels=False, shape_env=shape_env):
            # 创建随机张量 x
            x = torch.rand([10])
            # 调用 tolist 方法
            x.tolist()

    # 预期真实张量传播失败
    @expectedFailurePropagateRealTensors
    # 定义一个测试函数，验证在相同形状环境下张量的保留
    def test_same_shape_env_preserved(self):
        # 创建一个形状环境对象
        shape_env = ShapeEnv()
        # 创建一个虚拟张量模式，使用上述形状环境
        mode1 = FakeTensorMode(shape_env=shape_env)
        # 创建张量 t1，从标准正态分布中随机生成的大小为 10 的张量开始
        t1 = mode1.from_tensor(
            torch.randn(10),
            # 使用无状态符号上下文，动态尺寸为 [DimDynamic.DYNAMIC]，约束尺寸为空
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC], constraint_sizes=[None]
            ),
        )
        # 创建另一个虚拟张量模式，使用相同的形状环境
        mode2 = FakeTensorMode(shape_env=shape_env)
        # 使用 t1 创建张量 t2
        t2 = mode2.from_tensor(t1)
        # 断言 t2 的对象不同于 t1
        self.assertIsNot(t2, t1)
        # 断言 t1 的虚拟模式是 mode1
        self.assertIs(t1.fake_mode, mode1)
        # 断言 t2 的虚拟模式是 mode2
        self.assertIs(t2.fake_mode, mode2)
        # 断言 t2 的第一个维度的形状环境与 t1 的第一个维度的形状环境相同
        self.assertIs(t2.size(0).node.shape_env, t1.size(0).node.shape_env)
        # 断言 t2 的第一个维度的字符串表示与 t1 的第一个维度的字符串表示相同
        self.assertEqual(str(t2.size(0)), str(t1.size(0)))

    # TODO: Support NJT.  There's also some funny business with dynamic shapes
    # which would need to be dealt with as well
    # 标记为预期的失败，传播真实张量的测试函数，用于验证伪造张量之间的保留性
    @expectedFailurePropagateRealTensors
    def test_jagged_fake_to_fake_preserved(self):
        # 从 torch.nested._internal.nested_tensor 中导入 jagged_from_list 函数
        from torch.nested._internal.nested_tensor import jagged_from_list

        # 设置三个不同尺寸的张量 S0, S1, S2 和一个维度 D
        S0, S1, S2 = 3, 4, 5
        D = 4
        # 创建三个随机张量 a, b, c，每个大小为 (Si, D)，需要梯度，数据类型为 float64
        a = torch.randn(S0, D, requires_grad=True, dtype=torch.float64)
        b = torch.randn(S1, D, requires_grad=True, dtype=torch.float64)
        c = torch.randn(S2, D, requires_grad=True, dtype=torch.float64)
        offsets = None
        # 使用 jagged_from_list 函数创建嵌套张量 jt，并返回
        jt, _ = jagged_from_list([a, b, c], offsets)
        # 创建一个形状环境对象
        shape_env = ShapeEnv()
        # 创建一个虚拟张量模式，使用上述形状环境
        mode1 = FakeTensorMode(shape_env=shape_env)
        # 使用 jt 创建张量 t1
        t1 = mode1.from_tensor(jt)
        # 创建另一个虚拟张量模式，使用相同的形状环境
        mode2 = FakeTensorMode(shape_env=shape_env)
        # 使用 t1 创建张量 t2
        t2 = mode2.from_tensor(t1)
        # 断言 t1 的自由符号不为空
        self.assertTrue(free_symbols(t1.size()))
        # 断言 t2 的对象不同于 t1
        self.assertIsNot(t2, t1)
        # 断言 t1 的偏移量方法的虚拟模式是 mode1
        self.assertIs(t1.offsets().fake_mode, mode1)
        # 断言 t2 的偏移量方法的虚拟模式是 mode2
        self.assertIs(t2.offsets().fake_mode, mode2)
        # 断言 t2 的第二个维度的形状环境与 t1 的第二个维度的形状环境相同
        self.assertIs(t2.size(1).node.shape_env, t1.size(1).node.shape_env)
        # 断言 t2 的第二个维度的字符串表示与 t1 的第二个维度的字符串表示相同
        self.assertEqual(str(t2.size(1)), str(t1.size(1)))

    # 检查两个张量的元数据属性
    def checkMetaProps(self, t1, t2):
        prims.utils.compare_tensor_meta(t1, t2, check_strides=True)

    # 根据跨引用情况决定是否跳过该测试
    @skipIfCrossRef
    # 定义名为 test_deepcopy 的测试方法
    def test_deepcopy(self):
        # 使用 FakeTensorMode 上下文管理器模拟环境
        with FakeTensorMode() as mode:
            pass
        # 创建一个包含 10 个通道的 BatchNorm2d 模块
        mod = torch.nn.BatchNorm2d(10)
        # 使用 torch._subclasses.fake_tensor.FakeCopyMode 上下文管理器进行深拷贝模式
        with torch._subclasses.fake_tensor.FakeCopyMode(mode):
            # 对 mod 进行深度拷贝
            mod_copied = copy.deepcopy(mod)

        # 定义函数 check_copy，用于验证拷贝后的模块与原始模块的属性
        def check_copy(mod, mod_copied):
            # 遍历原始模块和拷贝模块的参数和缓冲区
            for name, param in itertools.chain(
                mod.named_parameters(), mod.named_buffers()
            ):
                # 获取拷贝模块中同名的参数或缓冲区
                param_copied = getattr(mod_copied, name)
                # 检查参数的元数据属性
                self.checkMetaProps(param, param_copied)
                # 断言拷贝后的对象是 FakeTensor 类的实例
                self.assertTrue(isinstance(param_copied, FakeTensor))
                # 检查参数是否为 torch.nn.Parameter 的实例，拷贝是否一致
                self.assertEqual(
                    isinstance(param, torch.nn.Parameter),
                    isinstance(param_copied, torch.nn.Parameter),
                )
                # 检查参数的 requires_grad 属性是否一致
                self.assertEqual(param.requires_grad, param_copied.requires_grad)

        # 对 mod 和 mod_copied 进行属性验证
        check_copy(mod, mod_copied)

        # 定义一个新的 ModuleNew 类，继承自 torch.nn.Module
        class ModuleNew(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个形状为 [10, 2] 的随机张量 a
                self.a = torch.rand([10, 2])
                # 将 b 引用指向 a
                self.b = self.a
                # 将 c 引用指向 a 的第一个元素
                self.c = self.a[0]

        # 创建 ModuleNew 类的实例 mod
        mod = ModuleNew()
        # 使用 torch._subclasses.fake_tensor.FakeCopyMode 上下文管理器进行深拷贝模式
        with torch._subclasses.fake_tensor.FakeCopyMode(mode):
            # 对 mod 进行深度拷贝
            mod_copied = copy.deepcopy(mod)

        # 断言 mod_copied 的属性 a 和 b 是同一个对象
        self.assertIs(mod_copied.a, mod_copied.b)
        # 断言 mod_copied 的属性 a 和 b 的存储数据指针相同
        self.assertEqual(mod_copied.b.storage()._cdata, mod_copied.a.storage()._cdata)

    # 使用 unittest.skipIf 装饰器，若条件 TEST_WITH_TORCHDYNAMO 为真，则跳过此测试
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    # 使用 unittest.skipIf 装饰器，若条件 RUN_CUDA 为假，则跳过此测试
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义名为 test_new 的测试方法
    def test_new(self):
        # 使用 FakeTensorMode 上下文管理器模拟环境
        with FakeTensorMode():
            # 创建一个形状为 [16, 1] 的随机张量 a
            a = torch.rand([16, 1])
            # 使用 self.checkType 验证 a.new(10, 10) 返回的张量类型及其形状
            self.checkType(a.new(10, 10), "cpu", [10, 10])
            # 使用 self.checkType 验证 a.new([1, 2, 3, 4]) 返回的张量类型及其形状
            self.checkType(a.new([1, 2, 3, 4]), "cpu", [4])
            # 创建一个形状为 [4, 4]，设备为 "cuda" 的随机张量 b
            b = torch.rand([4, 4], device="cuda")
            # 使用 self.checkType 验证 b.new(device="cuda") 返回的张量类型及其形状
            self.checkType(b.new(device="cuda"), "cuda", [0])
            # 使用 self.checkType 验证 a.new(torch.rand([1])) 返回的张量类型及其形状
            self.checkType(a.new(torch.rand([1])), "cpu", [1])

    # 使用 unittest.skipIf 装饰器，若条件 TEST_WITH_TORCHDYNAMO 为真，则跳过此测试
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    # 定义名为 test_scalar_inputs 的测试方法
    def test_scalar_inputs(self):
        # 使用 FakeTensorMode 上下文管理器模拟环境
        with FakeTensorMode():
            # 使用 self.checkType 验证 torch.div(3, 2) 返回的张量类型及其形状
            self.checkType(torch.div(3, 2), "cpu", [])
            # 创建一个形状为 [2]，数据类型为 torch.int32 的零张量 ten
            ten = torch.zeros(2, dtype=torch.int32) * 2.0
            # 断言 ten 的数据类型为 torch.float
            self.assertEqual(ten.dtype, torch.float)
            # 使用 self.checkType 验证 ten 返回的张量类型及其形状
            self.checkType(ten, "cpu", [2])

    # 使用 unittest.skipIf 装饰器，若条件 TEST_WITH_TORCHDYNAMO 为真，则跳过此测试
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    # 定义名为 test_allow_meta 的测试方法
    def test_allow_meta(self):
        # 定义内部函数 run_meta
        def run_meta():
            # 使用 FakeTensorMode 上下文管理器模拟环境
            with FakeTensorMode():
                # 创建一个形状为 [4]，设备为 "meta" 的随机张量 x
                x = torch.rand([4], device="meta")
                # 返回 x 与自身的加法操作结果
                return x + x

        # 使用 self.checkType 验证 run_meta() 返回的张量类型及其形状
        self.checkType(run_meta(), "meta", [4])

        # 使用 patch.object 修改 torch._functorch.config.fake_tensor_allow_meta 属性为 False
        with patch.object(torch._functorch.config, "fake_tensor_allow_meta", False):
            # 使用 self.assertRaises 验证 run_meta() 抛出异常
            self.assertRaises(Exception, run_meta)
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义测试函数，测试在多设备上进行 ATen 复制操作
    def test_aten_copy_multi_device(self):
        # 进入伪张量模式
        with FakeTensorMode():
            # 创建一个在 CPU 上的随机张量 x1
            x1 = torch.rand(4, device="cpu")
            # 创建一个在 CUDA 上的随机张量 x2
            x2 = torch.rand(4, device="cuda")
            # 使用默认的 ATen 复制操作，将 x1 复制到 x2
            copy1 = torch.ops.aten.copy.default(x1, x2)
            # 使用默认的 ATen 复制操作，将 x2 复制到 x1
            copy2 = torch.ops.aten.copy.default(x2, x1)
            # 创建一个空张量 out，在 CPU 上
            out = torch.empty(4, device="cpu")
            # 使用 ATen 复制操作，将 x1 复制到 x2，并将结果存储在 out 中
            torch.ops.aten.copy.out(x1, x2, out=out)
        # 检查复制操作后的张量类型和设备
        self.checkType(copy1, "cpu", (4,))
        self.checkType(copy2, "cuda", (4,))
        self.checkType(out, "cpu", (4,))
    # 测试函数，用于测试多设备上的索引操作
    def test_aten_index_multi_device(self):
        # 进入 FakeTensorMode 上下文
        with FakeTensorMode():
            # 在 CPU 设备上创建随机张量 x1
            x1 = torch.rand(4, 4, device="cpu")
            # 在 CUDA 设备上创建随机张量 x2
            x2 = torch.rand(4, 4, device="cuda")
            # 在 CUDA 设备上创建索引张量 i1
            i1 = torch.tensor([0, 1], device="cuda")
            # 在 CPU 设备上创建索引张量 i2
            i2 = torch.tensor([0, 1], device="cpu")
            # 使用 torch.ops.aten.index 函数尝试用 i1 在 x1 上进行索引（不可行的情况）
            # r1 = torch.ops.aten.index(x1, i1)
            # 使用 torch.ops.aten.index 函数在 x2 上使用 i2 进行索引
            r2 = torch.ops.aten.index(x2, i2)

            # 在 CPU 设备上创建随机张量 y1
            y1 = torch.rand(4, device="cpu")
            # 在 CUDA 设备上创建随机张量 y2
            y2 = torch.rand(4, device="cuda")
            # 在 CUDA 设备上创建索引张量 j1
            j1 = torch.tensor([2], device="cuda")
            # 在 CPU 设备上创建索引张量 j2
            j2 = torch.tensor([2], device="cpu")
            # 使用 torch.ops.aten.index_put.default 函数，在 x1 上用 j1 进行索引赋值
            r3 = torch.ops.aten.index_put.default(x1, j1, y1)
            # 使用 torch.ops.aten.index_put.default 函数，在 x2 上用 j2 进行索引赋值
            r4 = torch.ops.aten.index_put.default(x2, j2, y2)
        # 检查 r1 的类型和设备，期望为 "cpu" 和空元组
        # self.checkType(r1, "cpu", ())
        # 检查 r2 的类型和设备，期望为 "cuda" 和空元组
        self.checkType(r2, "cuda", ())
        # 检查 r3 的类型和形状，期望为 "cpu" 和 (4, 4)
        self.checkType(r3, "cpu", (4, 4))
        # 检查 r4 的类型和形状，期望为 "cuda" 和 (4, 4)
        self.checkType(r4, "cuda", (4, 4))

    # 使用 unittest.skipIf 装饰器标记，用于条件跳过测试
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile"
    )
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 测试函数，用于测试多设备上的切片散步操作
    def test_aten_slice_scatter_multi_device(self):
        # 进入 FakeTensorMode 上下文
        with FakeTensorMode():
            # 在 CPU 设备上创建随机张量 x1
            x1 = torch.rand(4, 4, device="cpu")
            # 在 CUDA 设备上创建随机张量 y1
            y1 = torch.rand(2, 4, device="cuda")
            # 在 CUDA 设备上创建随机张量 x2
            x2 = torch.rand(4, 4, device="cuda")
            # 在 CPU 设备上创建随机张量 y2
            y2 = torch.rand(2, 4, device="cpu")
            # 创建一个空的张量 out 在 CPU 设备上
            out = torch.empty(4, 4, device="cpu")
            # 使用 torch.ops.aten.slice_scatter.default 函数，对 x1 使用 y1 进行切片散步，从索引 2 开始
            r1 = torch.ops.aten.slice_scatter.default(x1, y1, start=2)
            # 使用 torch.ops.aten.slice_scatter.default 函数，对 x2 使用 y2 进行切片散步，从索引 2 开始
            r2 = torch.ops.aten.slice_scatter.default(x2, y2, start=2)
            # 使用 torch.ops.aten.slice_scatter.out 函数，对 x1 使用 y1 进行切片散步，将结果写入 out 张量中，从索引 2 开始
            r3 = torch.ops.aten.slice_scatter.out(x1, y1, out=out, start=2)
        # 检查 r1 的类型和形状，期望为 "cpu" 和 (4, 4)
        self.checkType(r1, "cpu", (4, 4))
        # 检查 r2 的类型和形状，期望为 "cuda" 和 (4, 4)
        self.checkType(r2, "cuda", (4, 4))
        # 检查 r3 的类型和形状，期望为 "cpu" 和 (4, 4)
        self.checkType(r3, "cpu", (4, 4))
        # 检查 out 的类型和形状，期望为 "cpu" 和 (4, 4)
        self.checkType(out, "cpu", (4, 4))

    # 测试函数，用于测试 _adaptive_avg_pool2d_backward 函数的反向传播
    def test__adaptive_avg_pool2d_backward(self):
        # 进入 FakeTensorMode 上下文
        with FakeTensorMode():
            # 创建一个随机梯度输出张量 grad_out
            grad_out = torch.rand(2, 3, 4, 4)
            # 创建一个随机输入张量 inp，并设置其内存格式为通道最后格式
            inp = torch.rand(2, 3, 4, 4).to(memory_format=torch.channels_last)
            # 使用 torch.ops.aten._adaptive_avg_pool2d_backward 函数计算梯度输入张量 grad_in
            grad_in = torch.ops.aten._adaptive_avg_pool2d_backward(grad_out, inp)
            # 断言梯度输入张量的推荐内存格式为通道最后格式
            self.assertTrue(
                torch._prims_common.suggest_memory_format(grad_in)
                == torch.channels_last
            )

    # 测试函数，用于测试将 PyTorch 模型导出为 NumPy 的功能
    def test_export_numpy(self):
        # 定义一个简单的 PyTorch 模型 MyNumpyModel
        class MyNumpyModel(torch.nn.Module):
            # 前向传播函数，将输入转换为 NumPy 数组并加上随机扰动
            def forward(self, input):
                input = input.numpy()
                return input + np.random.randn(*input.shape)

        # 进入 FakeTensorMode 上下文
        with FakeTensorMode():
            # 使用 MyNumpyModel 模型导出一个程序 ep，传入一个随机张量作为参数
            ep = torch.export.export(MyNumpyModel(), args=(torch.randn(1000),))
            # 断言导出的程序 ep 是 torch.export.ExportedProgram 类的实例
            self.assertTrue(isinstance(ep, torch.export.ExportedProgram))
    # 定义测试方法，用于测试unsqueeze_copy函数的行为
    def test_unsqueeze_copy(self):
        # 创建形状环境对象
        shape_env = ShapeEnv()
        # 创建一个形状为(2, 2, 768)的全1张量
        t1 = torch.ones(2, 2, 768)
        # 使用FakeTensorMode上下文管理器，将张量t1转换为FakeTensor对象t
        with FakeTensorMode(shape_env=shape_env) as fake_mode:
            t = fake_mode.from_tensor(
                t1,
                # 使用StatelessSymbolicContext创建符号上下文，指定动态维度
                symbolic_context=StatelessSymbolicContext(
                    dynamic_sizes=[
                        DimDynamic.DYNAMIC,
                        DimDynamic.STATIC,
                        DimDynamic.STATIC,
                    ],
                ),
            )

        # 断言t的第0维的大小等于在第1维度上调用unsqueeze_copy后的张量的第0维的大小
        self.assertEqual(t.shape[0], torch.ops.aten.unsqueeze_copy(t, 1).shape[0])

    # 定义测试方法，用于测试alias_call函数的行为
    def test_alias_call(self):
        # 获取torch.autograd.forward_ad别名fwAD
        fwAD = torch.autograd.forward_ad

        # 定义函数f，对输入x乘以常数4312491并返回结果
        def f(x):
            return 4312491 * x

        # 使用FakeTensorMode上下文管理器
        with torch._subclasses.fake_tensor.FakeTensorMode():
            # 使用fwAD.dual_level()上下文管理器
            with fwAD.dual_level():
                # 在CPU上生成一个形状为(3,)的随机张量x
                x = torch.randn(3, device="cpu")
                # 根据x生成一个形状相同的全1张量y
                y = torch.ones_like(x)
                # 使用fwAD.make_dual函数将x和y转换为双重数（dual数）
                dual = fwAD.make_dual(x, y)
                # 调用函数f，对dual进行计算，得到结果r
                r = f(dual)

        # 断言r的类型为FakeTensor
        self.assertIsInstance(r, FakeTensor)
        # 断言r的大小为[3]
        self.assertEqual(r.size(), [3])
# 实例化具有参数化测试的 FakeTensorTest 测试类
instantiate_parametrized_tests(FakeTensorTest)


def make_propagate_real_tensors_cls(cls):
    # 创建一个带有补丁的测试类，并命名为 PropagateRealTensors
    # 使用 torch._functorch.config 进行修补，设置 fake_tensor_propagate_real_tensors 为 True
    cls = make_test_cls_with_patches(
        cls,
        "PropagateRealTensors",
        "_propagate_real_tensors",
        (torch._functorch.config, "fake_tensor_propagate_real_tensors", True),
        xfail_prop="_expected_failure_propagate_real_tensors",
        decorator=skipIfTorchDynamo("propagate_real_tensors affects Dynamo"),
    )
    # 设置类的 __file__ 和 __module__ 属性
    cls.__file__ = __file__
    cls.__module__ = __name__
    # 将类添加到全局命名空间中
    globals()[cls.__name__] = cls


# 使用 FakeTensorTest 类创建 PropagateRealTensors 测试类
make_propagate_real_tensors_cls(FakeTensorTest)


class FakeTensorConstHandling(TestCase):
    def assertConst(self, *args):
        # 断言参数中的每个对象都具有 constant 属性，且不为 None
        for arg in args:
            self.assertTrue(arg.constant is not None)

    def assertNotConst(self, *args):
        # 断言参数中的每个对象都具有 constant 属性，且为 None
        for arg in args:
            self.assertTrue(arg.constant is None)

    def test_simple(self):
        # 在 FakeTensorMode 下进行测试
        with FakeTensorMode():
            # 创建一个值为 4.0 的 torch.tensor 对象 x
            x = torch.tensor(4.0)
            # 断言 x 的值为 4.0
            self.assertEqual(x.item(), 4.0)

    def test_inplace_add(self):
        # 在 FakeTensorMode 下进行测试
        with FakeTensorMode():
            # 创建一个值为 4.0 的 torch.tensor 对象 x
            x = torch.tensor(4.0)
            # 对 x 执行 inplace 加法操作，并将结果赋给 y
            y = x.add_(1)
            # 断言 x 和 y 的值都为 5.0
            self.assertEqual(x.item(), 5.0)
            self.assertEqual(y.item(), 5.0)
            # 断言 x 和 y 均为常量张量
            self.assertConst(x, y)

    def test_shared_storages(self):
        # 在 FakeTensorMode 下进行测试
        with FakeTensorMode():
            # 创建一个包含单个元素 4.0 的 torch.tensor 对象 x
            x = torch.tensor([4.0])
            # 通过切片创建对象 y，与 x 共享存储
            y = x[:]
            # 断言 x 和 y 的存储地址相同
            self.assertEqual(x.storage()._cdata, y.storage()._cdata)
            # 断言 x 的常量属性和 y 的常量属性的存储地址也相同
            self.assertEqual(x.constant.storage()._cdata, y.constant.storage()._cdata)

    def test_constant_invalidation(self):
        # 在 FakeTensorMode 下进行测试
        with FakeTensorMode():
            # 创建一个包含单个元素 1.0 的 torch.tensor 对象 x，并断言其为常量张量
            x = torch.tensor([1.0])
            self.assertConst(x)
            # 创建一个随机数张量 y，并将其加到 x 上
            y = torch.rand([1])
            x.add_(y)
            # 断言 x 不再是常量张量
            self.assertNotConst(x)

    def test_inplace_view_invalidation(self):
        # 在 FakeTensorMode 下进行测试
        with FakeTensorMode():
            # 创建一个包含单个元素 1 的 torch.tensor 对象 x，并断言其为常量张量
            x = torch.tensor([1])
            self.assertConst(x)
            # 调整 x 的形状为 [2]
            x.resize_([2])
            # 断言 x 的大小为 2，并且不再是常量张量
            self.assertEqual(x.size(0), 2)
            self.assertNotConst(x)

    def test_fake_tensor_in_intlist_repro(self):
        # 定义一个函数 fn，接受张量列表作为输入，返回一个填充为 0.0 的张量
        def fn(tensors):
            max_size = torch.tensor([800, 1216], dtype=torch.int64)
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
            return tensors[0].new_full(batch_shape, 0.0)

        # 断言在使用 FakeTensorMode 下执行时会抛出 DataDependentOutputException 异常
        with self.assertRaises(
            torch._subclasses.fake_tensor.DataDependentOutputException
        ):
            with torch._subclasses.fake_tensor.FakeTensorMode():
                # 创建两个随机张量 a 和 b
                a = torch.randn(3, 800, 1199)
                b = torch.randn(3, 800, 800)
                inputs = [a, b]
                # 调用函数 fn，并将结果存储在 ref 中
                ref = fn(inputs)

    def test_fake_tensor_batch_norm_cpu(self):
        # 使用 CrossRefFakeMode 下的测试
        with torch._subclasses.CrossRefFakeMode():
            # 创建一个包含批归一化层和 ReLU 激活函数的序列模型 m
            m = torch.nn.Sequential(
                torch.nn.BatchNorm2d(10),
                torch.nn.ReLU(),
            )
            # 将模型设置为评估模式
            m.eval()
            # 将随机输入张量输入模型 m 中，并获取输出
            out = m(torch.randn([2, 10, 8, 8]))
    # 定义一个测试方法，用于验证共享存储的失效
    def test_shared_storage_invalidation(self):
        # 进入模拟张量模式的上下文
        with FakeTensorMode():
            # 创建一个张量 x，包含单个浮点数 1.0
            x = torch.tensor([1.0])
            # 创建一个 y 张量，其内容与 x 相同，共享存储
            y = x[:]
            # 断言 x 和 y 共享存储
            self.assertConst(x, y)
            # 修改 y 的内容，向其中添加随机浮点数
            y.add_(torch.rand([1]))
            # 断言 x 和 y 不再共享存储
            self.assertNotConst(x, y)

    # 定义一个测试方法，用于验证别名的常量写入
    def test_aliased_const_write(self):
        # 进入模拟张量模式的上下文
        with FakeTensorMode():
            # 创建一个张量 x，包含单个整数 1
            x = torch.tensor([1])
            # 创建一个 y 张量，通过扩展 x 而得到，其内容与 x 部分共享存储
            y = x.expand([4])
            # 断言 y 不是常量
            self.assertNotConst(y)
            # 修改 y 的第一个元素为整数 1
            y[0] = 1
            # 断言 x 不是常量
            self.assertNotConst(x)

    # 定义一个测试方法，用于验证常量在函数传递中的传播
    def test_constant_propagate_through_functions(self):
        # 进入模拟张量模式的上下文
        with FakeTensorMode():
            # 使用 torch.div 函数创建一个常量张量 y，值为 1
            y = torch.div(4, 4, rounding_mode="trunc")
            # 断言 y 是常量
            self.assertConst(y)
# 使用 FakeTensorConstHandling 创建并注册 propagate_real_tensors_cls
make_propagate_real_tensors_cls(FakeTensorConstHandling)


def contains_type(type: torch.Type, maybe_contained_type: torch.Type):
    # 检查 maybe_contained_type 是否是 type 的子类型，或者检查 type 的 containedTypes 中是否存在 maybe_contained_type 的子类型
    return maybe_contained_type.isSubtypeOf(type) or any(
        contains_type(e, maybe_contained_type) for e in type.containedTypes()
    )


class FakeTensorOpInfoTest(TestCase):
    @ops(custom_op_db, dtypes=OpDTypes.any_one)
    def test_fake(self, device, dtype, op):
        # 生成 op 的样本输入，返回一个迭代器
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in sample_inputs_itr:
            args = (sample_input.input,) + sample_input.args
            kwargs = sample_input.kwargs
            # 使用 optests 模块检查 op 的行为
            optests.fake_check(op, args, kwargs)


# 使用 FakeTensorOpInfoTest 创建并注册 propagate_real_tensors_cls
make_propagate_real_tensors_cls(FakeTensorOpInfoTest)

# 在全局范围内实例化 FakeTensorOpInfoTest 类的设备类型相关测试，仅限于 "cpu" 和 "cuda" 设备
instantiate_device_type_tests(FakeTensorOpInfoTest, globals(), only_for=("cpu", "cuda"))

# 在全局范围内实例化 PropagateRealTensorsFakeTensorOpInfoTest 类的设备类型相关测试，仅限于 "cpu" 设备
instantiate_device_type_tests(
    PropagateRealTensorsFakeTensorOpInfoTest, globals(), only_for=("cpu",)  # noqa: F821
)


class FakeTensorConverterTest(TestCase):
    def test_memoized_conversion_to_meta(self):
        # 创建一个随机张量 x
        x = torch.rand(2, 2, 2)
        mode = FakeTensorMode()
        # 断言 meta 模式下的张量转换为 meta 后，对象是同一实例
        self.assertTrue(mode.from_tensor(x) is mode.from_tensor(x))

    def test_memoized_conversion_from_meta(self):
        # 创建一个随机张量 x，并设置其设备为 "meta"
        x = torch.rand(2, 2).to(device="meta")
        mode = FakeTensorMode()
        converter = mode.fake_tensor_converter
        # 断言从 meta 和设备转换后的对象是同一实例
        self.assertTrue(
            converter.from_meta_and_device(mode, x, "cpu")
            is converter.from_meta_and_device(mode, x, "cpu")
        )

    def test_separate_tensor_storages_view(self):
        # 创建一个随机张量 x 和它的一个视图 y
        x = torch.rand(2, 2, 2)
        y = x[0]
        mode = FakeTensorMode()
        converter = mode.fake_tensor_converter
        # 将实际张量 x 和其视图 y 转换为 fake 张量，然后比较它们的 storage_id
        x_conv = converter.from_real_tensor(mode, x)
        y_conv = converter.from_real_tensor(mode, y)
        self.assertEqual(torch._C._storage_id(x_conv), torch._C._storage_id(y_conv))

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1991")
    def test_separate_tensor_storages_non_view(self):
        # 创建两个随机张量 x 和 y，并将 y 的存储设置为 x 的存储
        x = torch.rand(2, 2, 2)
        y = torch.rand(4, 2)
        y.set_(x.storage())
        mode = FakeTensorMode()
        converter = mode.fake_tensor_converter
        # 将实际张量 x 和 y 转换为 fake 张量，然后比较它们的 storage_id
        x_conv = converter.from_real_tensor(mode, x)
        y_conv = converter.from_real_tensor(mode, y)
        stor_id = torch._C._storage_id(x_conv)
        self.assertEqual(stor_id, torch._C._storage_id(y_conv))
        # 清除变量并检查转换器的内部状态
        del x
        del x_conv
        self.assertEqual(len(converter.tensor_memo), 1)
        self.assertEqual(len(converter.meta_converter.storage_memo), 1)
        del y
        del y_conv
        self.assertEqual(len(converter.tensor_memo), 0)
        self.assertEqual(len(converter.meta_converter.storage_memo), 0)

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1991")
    # 测试函数，用于验证在没有引用循环的情况下的行为
    def test_no_ref_cycle(self):
        # 创建一个包含4个随机数的张量
        x = torch.rand([4])
        # 创建一个 FakeTensorMode 对象
        mode = FakeTensorMode()
        # 将张量 x 转换为 FakeTensor
        y = mode.from_tensor(x)
        # 验证 tensor_memo 中包含一个对象
        self.assertEqual(len(mode.fake_tensor_converter.tensor_memo), 1)
        # 创建 mode 的弱引用
        mode_weak = weakref.ref(mode)
        # 创建 y 的弱引用
        y_weak = weakref.ref(mode)
        # 删除 mode 和 y 的引用
        del mode
        del y
        # 断言 mode 的弱引用指向 None
        assert mode_weak() is None
        # 断言 y 的弱引用指向 None
        assert y_weak() is None
# 为给定的类添加所有属性和方法，确保在测试中覆盖所有的转换案例
make_propagate_real_tensors_cls(FakeTensorConverterTest)

# 定义一个测试类 FakeTensorOperatorInvariants，继承自 TestCase
class FakeTensorOperatorInvariants(TestCase):

    # 获取 ATen 操作的函数
    def get_aten_op(self, schema):
        # 将 schema.name 按 "::" 分割为 namespace 和 name
        namespace, name = schema.name.split("::")
        # 如果有重载名称，则使用重载名称，否则使用 "default"
        overload = schema.overload_name if schema.overload_name else "default"
        # 断言 namespace 必须是 "aten"
        assert namespace == "aten"
        # 返回 torch.ops.aten 中对应的操作对象
        return getattr(getattr(torch.ops.aten, name), overload)

    # 获取所有 ATen 模式的生成器函数
    def get_all_aten_schemas(self):
        # 遍历所有的 ATen schemas
        for schema in torch._C._jit_get_all_schemas():
            # 将 schema.name 按 "::" 分割为 namespace 和 name
            namespace = schema.name.split("::")[0]
            # 如果 namespace 不是 "aten"，则继续下一个 schema
            if namespace != "aten":
                continue
            # 生成当前 schema
            yield schema

    # 测试函数，验证是否有非关键字参数的设备类型
    def test_non_kwarg_only_device(self):
        # 遍历所有 ATen schemas
        for schema in self.get_all_aten_schemas():
            # 获取 Tensor 类型
            ten_type = torch._C.TensorType.get()
            # 如果没有任何参数或返回值的类型包含 ten_type，则继续下一个 schema
            if not any(
                contains_type(arg.type, ten_type)
                for arg in itertools.chain(schema.arguments, schema.returns)
            ):
                continue

            # 定义可选的设备类型
            opt_device = torch._C.OptionalType(torch._C.DeviceObjType.get())
            # 检查是否存在非关键字参数的设备类型
            has_non_kwarg_device = any(
                not arg.kwarg_only and arg.type.isSubtypeOf(opt_device)
                for arg in schema.arguments
            )
            # 如果有非关键字参数的设备类型，则断言该 ATen 操作存在于 fake_tensor._device_not_kwarg_ops 中
            if has_non_kwarg_device:
                self.assertTrue(
                    self.get_aten_op(schema)
                    in torch._subclasses.fake_tensor._device_not_kwarg_ops
                )

    # 测试函数，验证所有张量构造函数是否都有关键字参数设备
    def test_tensor_constructors_all_have_kwarg_device(self):
        # 遍历所有 ATen schemas
        for schema in self.get_all_aten_schemas():
            # 获取 ATen 操作
            op = self.get_aten_op(schema)
            # 如果不是张量构造函数，则继续下一个 schema
            if not torch._subclasses.fake_tensor._is_tensor_constructor(op):
                continue

            # 定义可选的设备类型
            opt_device = torch._C.OptionalType(torch._C.DeviceObjType.get())
            # 检查是否存在关键字参数设备类型
            has_kwarg_device = any(
                arg.kwarg_only and arg.type.isSubtypeOf(opt_device)
                for arg in schema.arguments
            )

            # 断言所有张量构造函数都有关键字参数设备或者 op 是 _list_to_tensor.default
            self.assertTrue(
                has_kwarg_device or op == torch.ops.aten._list_to_tensor.default
            )

    # 标记为预期失败的测试函数，测试稀疏张量的创建
    @unittest.expectedFailure
    def test_sparse_new(self):
        with FakeTensorMode():
            indices = torch.randn(1, 1, dtype=torch.int64)
            values = torch.randn(1)
            extra = (2,)
            sparse = torch.randn(1).to_sparse()
            # 这曾经会导致段错误，现在不会了，但仍然会引发错误
            sparse2 = sparse.new(indices, values, extra)

    # 测试函数，验证张量的创建
    def test_tensor_new(self):
        with FakeTensorMode():
            x = torch.Tensor([1, 2, 3])
        # 断言 x 的类型是 FakeTensor
        self.assertIsInstance(x, FakeTensor)

    # 测试函数，验证 _like 操作
    def test_like_ops(self):
        # 遍历所有 ATen schemas
        for schema in self.get_all_aten_schemas():
            # 如果操作名称以 "_like" 结尾，则继续下一个 schema
            if "_like" == schema.name[-5:]:
                op = self.get_aten_op(schema)
                # 断言操作在 fake_tensor._like_tensor_constructors 中
                self.assertIn(
                    op, torch._subclasses.fake_tensor._like_tensor_constructors
                )
    # 定义一个测试方法，用于测试字符串存储功能
    def test_str_storage(self):
        # 创建一个包含三个元素的全零张量
        x = torch.zeros(3)
        # 使用 FakeTensorMode 上下文环境
        with FakeTensorMode() as m:
            # 将张量 x 转换为 m 的表示形式，并赋值给 y
            y = m.from_tensor(x)
            # 使用 self.assertExpectedInline 方法断言 x.storage() 的字符串表示是否符合预期
            self.assertExpectedInline(
                str(x.storage()),
                """\
 0.0
 0.0
 0.0
                """  # 期望 x.storage() 的字符串表示
            )
# 对于给定的输入张量，调用其存储（storage）方法，返回存储对象的字符串表示
self.assertExpectedInline(
    str(y.storage()),
    """\
...
[torch.storage.TypedStorage(dtype=torch.float32, device=cpu) of size 3]""",
)

# 对于给定的输入张量，调用其存储（storage）方法，返回存储对象的字符串表示
self.assertExpectedInline(
    str(y.storage()),
    """\
...
[torch.storage.TypedStorage(dtype=torch.float32, device=meta) of size 3]""",
)

# 使用特定的参数配置对 `at::_embedding_bag` 进行私有方法的测试
# 该方法不返回操作信息，并且返回额外的张量，这些张量由 `at::embedding_bag` 丢弃
def test_embedding_bag_private(self):
    # 准备测试参数
    args = [
        torch.ones(6, 1),
        torch.ones(6, dtype=torch.int64),
        torch.arange(2, dtype=torch.int64),
        False,
        2,  # mode = max
    ]

    # 调用原始的 `at::_embedding_bag` 方法
    ref_out = torch.ops.aten._embedding_bag(*args)

    # 使用 FakeTensorMode 上下文环境测试 `at::_embedding_bag` 方法
    with FakeTensorMode() as m:
        # 转换参数以适应元数据环境
        meta_args = [
            m.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args
        ]
        # 调用元数据环境下的 `at::_embedding_bag` 方法
        meta_out = torch.ops.aten._embedding_bag(*meta_args)

    # 断言原始输出和元数据输出的长度相等
    self.assertEqual(len(ref_out), len(meta_out))
    # 对比每对原始输出和元数据输出的大小
    for ref_o, meta_o in zip(ref_out, meta_out):
        self.assertEqual(ref_o.size(), meta_o.size())

# 对 `torch.nn.functional.cross_entropy` 方法进行测试
def test_cross_entropy_loss(self):
    # 准备输入数据
    inp = torch.randn(3, 5)
    target = torch.randint(5, (3,), dtype=torch.long)
    weight = torch.rand(5)
    fn = torch.nn.functional.cross_entropy

    # 针对不同权重参数进行测试
    for w in (weight, None):
        args = (inp, target, w)
        # 调用原始的 `torch.nn.functional.cross_entropy` 方法
        ref = fn(*args)

        # 使用 FakeTensorMode 上下文环境测试 `torch.nn.functional.cross_entropy` 方法
        with FakeTensorMode() as m:
            # 转换参数以适应元数据环境
            meta_args = [
                m.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args
            ]
            # 调用元数据环境下的 `torch.nn.functional.cross_entropy` 方法
            meta_out = torch.nn.functional.cross_entropy(
                *meta_args, label_smoothing=0.5
            )

        # 断言原始输出和元数据输出的大小相等
        self.assertEqual(ref.size(), meta_out.size())

# 跳过 ROCm 平台的测试
@skipIfRocm
# 在不支持 Flash Attention 的平台上跳过测试
@unittest.skipIf(
    not PLATFORM_SUPPORTS_FLASH_ATTENTION,
    "Does not support SDPA or pre-SM80 hardware",
)
    def test_flash_attention(self):
        # 定义一个内部类 Repro，继承自 torch.nn.Module
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 实现 forward 方法，执行特定的注意力计算操作
            def forward(self, arg1, arg2, arg3):
                torch.ops.aten._scaled_dot_product_flash_attention(
                    arg1, arg2, arg3, scale=0.17677669529663687
                )

        # 定义输入参数列表 args_new
        args_new = [
            [
                # 第一组参数元组列表
                ((1, 48, 64, 64), (0, 4096, 64, 1), torch.float16, "cuda"),
                ((1, 48, 64, 64), (0, 4096, 64, 1), torch.float16, "cuda"),
                ((1, 48, 64, 64), (0, 4096, 64, 1), torch.float16, "cuda"),
            ],
            [
                # 第二组参数元组列表
                ((4, 2, 16, 32), (1024, 512, 32, 1), torch.float16, "cuda"),
                ((4, 2, 16, 32), (1024, 512, 32, 1), torch.float16, "cuda"),
                ((4, 2, 16, 32), (1024, 512, 32, 1), torch.float16, "cuda"),
            ],
        ]

        # 遍历参数列表 args_new
        for args_list in args_new:
            # 根据每个元组参数列表生成参数 args
            args = [
                rand_strided(bsz, num_heads, seq_len, head_dim)
                for (bsz, num_heads, seq_len, head_dim) in args_list
            ]
            try:
                # 使用 CrossRefFakeMode 上下文管理器
                with torch._subclasses.CrossRefFakeMode():
                    # 创建 Repro 实例并传入参数执行 forward 方法
                    Repro()(*args)
            except RuntimeError as e:
                # 捕获 RuntimeError 异常
                # 确保输出的第一个输出不包含 "output[0]"，即预期的交叉引用成功但第一个输出失败
                self.assertTrue("output[0]" not in str(e))
                # 确保异常信息包含特定的错误描述，表示找到不匹配的张量元数据
                self.assertTrue(
                    "found mismatched tensor metadata for output[6]: Devices cpu and cuda:0 are not equal!"
                    in str(e)
                )

    # IMPORTANT!!! Always run even if CUDA is not available
    def test_fake_cuda_no_init(self):
        # 如果 torch._functorch.config.fake_tensor_propagate_real_tensors 为 True，则跳过测试
        if torch._functorch.config.fake_tensor_propagate_real_tensors:
            return
        # 使用 FakeTensorMode 上下文管理器
        with FakeTensorMode():
            # 在 CUDA 设备上创建各种张量并执行操作
            torch.empty(10, device="cuda")
            torch.ones(10, device="cuda")
            torch.zeros(10, device="cuda")
            torch.rand(10, device="cuda")
            torch.tensor(3.14, device="cuda")
            torch.tensor([[3.14, 2], [1, 2]], device="cuda")

    @skipIfRocm
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 定义一个名为 test_conv_c1_backward 的测试方法
    def test_conv_c1_backward(self):
        # 定义一个名为 Repro 的内部类，继承自 torch.nn.Module
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 定义 forward 方法，接受三个参数 arg1, arg2, arg3
            def forward(self, arg1, arg2, arg3):
                # 调用 torch.ops.aten.convolution_backward.default 执行卷积反向传播计算
                torch.ops.aten.convolution_backward.default(
                    arg1,  # 输入的梯度
                    arg2,  # 卷积核
                    arg3,  # 前向传播的输出
                    [1],  # groups
                    [1, 1],  # strides
                    [1, 1],  # padding
                    [1, 1],  # dilation
                    False,  # transposed
                    [0, 0],  # output_padding
                    1,  # groups
                    [True, True, False],  # output_mask
                )

        # 定义参数列表 args_new
        args_new = [
            ((16, 1, 128, 128), (16384, 16384, 128, 1), torch.float16, "cuda"),
            ((16, 64, 128, 128), (1048576, 1, 8192, 64), torch.float16, "cuda"),
            ((1, 64, 3, 3), (576, 9, 3, 1), torch.float16, "cuda"),
        ]
        # 使用 rand_strided 函数为每组参数生成具体值，构成列表 args
        args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args_new]

        # 使用 FakeMode 模拟跨引用假模式
        with torch._subclasses.CrossRefFakeMode():
            # 创建 Repro 实例并调用其 __call__ 方法，传入 args 中的参数
            Repro()(*args)

    # 定义一个名为 test_no_dispatch_with_like_function 的测试方法
    def test_no_dispatch_with_like_function(self):
        # 定义 CountingMode 类，继承自 TorchDispatchMode
        class CountingMode(TorchDispatchMode):
            def __init__(self):
                self.count = 0

            # 实现 __torch_dispatch__ 方法，用于记录调用次数并执行函数
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.count += 1
                return func(*args, **kwargs)

        # 使用 FakeTensorMode 模拟张量假模式
        with FakeTensorMode():
            # 创建随机张量 x
            x = torch.randn(2)
            # 创建 CountingMode 实例 mode，并在其作用域内执行以下操作
            with CountingMode() as mode:
                # 在 no_dispatch 上下文中，调用 torch.zeros_like(x) 函数
                with no_dispatch():
                    torch.zeros_like(x)

        # 断言 mode.count 的值为 0
        self.assertEqual(mode.count, 0)
# 调用函数 make_propagate_real_tensors_cls，传入 FakeTensorOperatorInvariants 作为参数
make_propagate_real_tensors_cls(FakeTensorOperatorInvariants)

# 定义 TestCase 的子类 FakeTensorPropTest
class FakeTensorPropTest(TestCase):
    
    # 定义测试方法 test_fake_tensor_prop_on_nn_module
    def test_fake_tensor_prop_on_nn_module(self):
        
        # 定义一个简单的神经网络模型 ToyNnModuleWithParameters
        class ToyNnModuleWithParameters(torch.nn.Module):
            
            # 初始化方法
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(4, 3)  # 第一层线性模块
                self.layer2 = torch.nn.Linear(3, 2)  # 第二层线性模块
            
            # 前向传播方法
            def forward(self, value):
                value = self.layer1(value)  # 第一层线性变换
                value = torch.relu(value)   # ReLU 激活函数
                value = self.layer2(value)  # 第二层线性变换
                return value
        
        # 创建 ToyNnModuleWithParameters 的实例 model
        model = ToyNnModuleWithParameters()
        value = torch.randn(5, 4)  # 创建一个随机张量作为输入数据
        
        # 将 nn.Module 转换为 GraphModule 以便运行 FakeTensorProp
        graph_model = torch.fx.symbolic_trace(model, (value,))
        
        # 下面的代码块在同一个 FakeTensorMode 下运行 FakeTensorProp
        with FakeTensorMode() as fake_tensor_mode:
            
            # 定义一个函数 to_fake_tensor，用于将普通张量转换为 FakeTensor
            def to_fake_tensor(x):
                if isinstance(x, torch.Tensor) and not isinstance(x, FakeTensor):
                    return fake_tensor_mode.from_tensor(x)
                return x
            
            # 创建 fake_parameters_and_buffers 字典，其中包含模型的参数和缓冲区的 FakeTensor
            fake_parameters_and_buffers = {
                k: to_fake_tensor(v)
                for k, v in itertools.chain(
                    graph_model.named_parameters(), graph_model.named_buffers()
                )
            }
            
            # 使用 stateless._reparametrize_module 函数重参数化模型
            with torch.nn.utils.stateless._reparametrize_module(
                graph_model, fake_parameters_and_buffers
            ):
                # 在 graph_model 上运行 FakeTensorProp，并传入 value 作为输入
                result = FakeTensorProp(graph_model, fake_tensor_mode).propagate(value)
                
                # 断言 result 是 FakeTensor 类型
                self.assertTrue(isinstance(result, FakeTensor))
                # 断言 result 的形状为 (5, 2)
                self.assertEqual(result.shape, (5, 2))
                
                # 使用不同的 FakeTensorMode 下运行 FakeTensorProp，预期会失败
                failed = False
                try:
                    FakeTensorProp(graph_model).propagate(value)
                except AssertionError:
                    # 预期捕获到 AssertionError: tensor's device must be `meta`, got cpu instead
                    failed = True
                self.assertTrue(failed)
    def test_fake_tensor_prop_on_nn_module_with_optional_args(self):
        class OptionalArgumentInBetween(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化神经网络模块，包括两个线性层
                self.layer1 = torch.nn.Linear(4, 3)
                self.layer2 = torch.nn.Linear(3, 2)

            def forward(self, value, another_value=None, another_optional_value=None):
                # 模仿 huggingface 的 `forward` 方法，支持多个可选参数
                # 例如，GPT 接受 forward(self, input_ids, None, attention_mask, ...) 形式
                # 为了应用 FakeTensorProp，需要使 from_real_tensor(...) 方法能接受 None
                if another_value is None:
                    another_value = torch.rand_like(value)
                if another_optional_value is None:
                    another_optional_value = torch.rand_like(value)
                # 对输入值进行运算
                value = value + another_value + another_optional_value
                return value * value

        # 设置 FakeTensorMode，允许非 FakeTensor 输入，但不允许回退到基本内核
        fake_mode = FakeTensorMode(
            allow_non_fake_inputs=True, allow_fallback_kernels=False
        )
        with fake_mode:
            # 创建模型实例
            model = OptionalArgumentInBetween()
            value = torch.randn(5, 4)
            another_optional_value = torch.randn(5, 4)
            # 对模型进行符号跟踪
            graph_model = torch.fx.symbolic_trace(
                model, (value, None, another_optional_value)
            )
            # 使用 FakeTensorProp 进行传播
            FakeTensorProp(graph_model, fake_mode).propagate(
                value, None, another_optional_value
            )

    def test_unbacked_shape_realloc(self):
        # 定义一个简单函数 f，返回输入的非零元素的索引
        def f(x):
            return x.nonzero()

        # 初始化 ShapeEnv，用于存储形状信息
        shape_env = ShapeEnv()
        # 设置 FakeTensorMode，使用指定的 shape_env
        fake_mode = FakeTensorMode(shape_env=shape_env)
        with fake_mode:
            value = torch.randn(5)
            # 使用 make_fx 将函数 f 转换为图模型
            gm = make_fx(f)(value)
        # 找到所有使用 torch.ops.aten.nonzero.default 的节点
        nonzero_nodes = [
            n for n in gm.graph.nodes if n.target is torch.ops.aten.nonzero.default
        ]
        # 断言只有一个节点使用了 nonzero 操作
        self.assertEqual(len(nonzero_nodes), 1)
        # 断言该节点的 meta 中的 shape[0] 类型为 torch.SymInt
        self.assertIsInstance(nonzero_nodes[0].meta["val"].shape[0], torch.SymInt)
        u0 = nonzero_nodes[0].meta["val"].shape[0]
        # 使用 FakeTensorProp 进行传播
        FakeTensorProp(gm, fake_mode).propagate(value)
        u1 = nonzero_nodes[0].meta["val"].shape[0]
        # 测试该测试用例是否有效，确保 FakeTensorProp 确实触发了重新分配
        # 如果该断言失败，可能是因为我们开始对 nonzero 的 nnz 计数进行了记忆化，
        # 这在某种意义上是好的（没有重新分配），但对于此测试是没有帮助的。
        # 如果如此，请尝试使此示例更复杂（例如，在将输入传递给 nonzero 之前进行非平凡的计算，
        # 或者引入某种随机性）
        self.assertIsNot(u0, u1)
        self.assertTrue(statically_known_true(u0 == u1))
    def test_torch_load_with_fake_mode(self):
        # 定义一个用于测试的神经网络模型类
        class TheModelClass(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(5, 10)  # 创建一个线性层，输入大小为5，输出大小为10

            def forward(self, x):
                return self.fc1(x)  # 前向传播函数，将输入x传递给fc1线性层并返回输出

        with TemporaryFileName() as state_dict_file:
            # 使用临时文件名state_dict_file创建一个模型实例并保存其状态字典
            model = TheModelClass()
            torch.save(model.state_dict(), state_dict_file)

            # 创建一个FakeTensorMode上下文环境
            fake_mode = FakeTensorMode()
            with fake_mode:
                torch.load(state_dict_file)  # 加载状态字典文件（场景1）
                torch.load(state_dict_file, map_location="cpu")  # 加载状态字典文件，指定映射位置为CPU（场景2）
# 导入一个名为 FakeTensorPropTest 的类，并用其创建一个名为 make_propagate_real_tensors_cls 的函数
make_propagate_real_tensors_cls(FakeTensorPropTest)

# 创建一个名为 FakeTensorSerialization 的测试类，继承自 TestCase
class FakeTensorSerialization(TestCase):

    # 测试序列化功能
    def test_serialization(self):
        # 创建一个在 CPU 上的 torch tensor x
        x = torch.tensor([0], device="cpu")
        
        # 进入 FakeTensorMode 上下文
        with FakeTensorMode():
            # 使用 pickle 序列化和反序列化 x，得到 y
            y = pickle.loads(pickle.dumps(x))
            # 断言 y 的类型为 FakeTensor
            self.assertEqual(type(y), FakeTensor)
            # 断言 y 的设备类型为 "meta"
            self.assertEqual(y.device.type, "meta")

            # 暂时取消 FakeTensor 模式
            with unset_fake_temporarily():
                # 再次序列化和反序列化 x，得到 y
                y = pickle.loads(pickle.dumps(x))
                # 断言 x 和 y 的设备相同
                self.assertEqual(x.device, y.device)

    # 测试带跟踪的序列化功能
    def test_serialization_with_tracing(self):
        # 创建一个在 CPU 上的 torch tensor x
        x = torch.tensor([0], device="cpu")
        
        # 使用跟踪上下文（TracingContext(FakeTensorMode())）
        with tracing(TracingContext(FakeTensorMode())):
            # 使用 pickle 序列化和反序列化 x，得到 y
            y = pickle.loads(pickle.dumps(x))
            # 断言 x 和 y 的设备相同
            self.assertEqual(x.device, y.device)


# 创建一个名为 FakeTensorDispatchCache 的测试类，继承自 TestCase
class FakeTensorDispatchCache(TestCase):

    # 测试 ShapeEnv 设置
    def test_shape_env_settings(self):
        """
        验证 ShapeEnv 中的任何布尔设置是否存在于 ShapeEnvSettings 中。
        我们希望确保任何可能影响 FakeTensor 分发的新设置都包含在缓存键计算中。
        如果此测试失败，请考虑更新 ShapeEnvSettings 或更改此测试以排除对新字段的检查。
        """
        # 获取 ShapeEnv._init 方法的参数签名
        init_sig = inspect.signature(ShapeEnv._init)
        # 提取所有默认类型为布尔型的参数名
        args = [
            name
            for name, param in init_sig.parameters.items()
            if type(param.default) is bool
        ]
        
        # 获取 ShapeEnvSettings 类中的所有字段名
        settings = [f.name for f in dataclasses.fields(ShapeEnvSettings)]
        # 对于每个参数名，断言其在 settings 中
        for arg in args:
            self.assertTrue(arg in settings)

    # 辅助函数，用于所有 test_cache_key_* 测试
    def _test_cache_key(self, fm, x, y, z):
        """
        Helper for all test_cache_key_* tests below. Assert that the
        cache keys for inputs x and y are the same, but z is different.
        """
        # 定义要测试的函数
        func = aten.add.Tensor
        # 计算输入 x, y, z 对应的缓存键
        key_x = fm._cache_key(func, [x], {})
        key_y = fm._cache_key(func, [y], {})
        key_z = fm._cache_key(func, [z], {})

        # 断言 x 和 y 的缓存键相同
        self.assertEqual(key_x, key_y)
        # 断言 x 和 z 的缓存键不同
        self.assertNotEqual(key_x, key_z)

    # 测试缓存键 - dtype
    def test_cache_key_dtype(self):
        # 进入 FakeTensorMode 上下文
        with FakeTensorMode() as fm:
            # 创建 torch float16 类型的 tensor x 和 y
            x = torch.randn(4, 3, dtype=torch.float16)
            y = torch.randn(4, 3, dtype=torch.float16)
            # 将 x 转换为 torch float32 类型的 tensor z
            z = x.to(dtype=torch.float32)
            # 调用 _test_cache_key 辅助函数进行测试
            self._test_cache_key(fm, x, y, z)

    # 测试缓存键 - shape
    def test_cache_key_shape(self):
        # 进入 FakeTensorMode 上下文
        with FakeTensorMode() as fm:
            # 创建形状为 (4, 3) 的 torch tensor x 和 y
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)
            # 创建形状为 (4, 2) 的 torch tensor z
            z = torch.randn(4, 2)
            # 调用 _test_cache_key 辅助函数进行测试
            self._test_cache_key(fm, x, y, z)

    # 测试缓存键 - stride
    def test_cache_key_stride(self):
        # 进入 FakeTensorMode 上下文
        with FakeTensorMode() as fm:
            # 创建步幅为 (1, 2) 的 torch tensor x 和 y
            x = torch.randn(4, 2)
            y = torch.randn(4, 2)
            # 使用 as_strided 方法创建步幅为 (1, 2) 的 torch tensor z
            z = x.as_strided((4, 2), (1, 2))
            # 调用 _test_cache_key 辅助函数进行测试
            self._test_cache_key(fm, x, y, z)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    # 测试缓存键与设备相关性
    def test_cache_key_device(self):
        # 使用 FakeTensorMode 上下文
        with FakeTensorMode() as fm:
            # 创建随机张量 x 和 y
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)
            # 将张量 x 转移到 CUDA 设备上，并赋给 z
            z = x.to(device="cuda")
            # 调用 _test_cache_key 方法测试缓存键
            self._test_cache_key(fm, x, y, z)

    # 测试缓存键与内存格式相关性
    def test_cache_key_memory_format(self):
        # 使用 FakeTensorMode 上下文
        with FakeTensorMode() as fm:
            # 创建随机张量 x 和 y，指定内存格式为 channels_last
            x = torch.randn(1, 2, 3, 4)
            y = torch.randn(1, 2, 3, 4)
            z = x.to(memory_format=torch.channels_last)
            # 调用 _test_cache_key 方法测试缓存键
            self._test_cache_key(fm, x, y, z)

    # 测试缓存键与存储偏移相关性
    def test_cache_key_storage_offset(self):
        # 使用 FakeTensorMode 上下文
        with FakeTensorMode() as fm:
            # 创建张量 x 和 y，从索引 1 开始，不包括首元素
            x = torch.randn(3)[1:]
            y = torch.randn(3)[1:]
            z = torch.randn(2)
            # 调用 _test_cache_key 方法测试缓存键
            self._test_cache_key(fm, x, y, z)

    # 测试缓存键与 requires_grad 属性相关性
    def test_cache_key_requires_grad(self):
        # 使用 FakeTensorMode 上下文
        with FakeTensorMode() as fm:
            # 创建随机张量 x 和 y，其中 z 需要梯度
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)
            z = torch.randn(4, 3, requires_grad=True)
            # 调用 _test_cache_key 方法测试缓存键
            self._test_cache_key(fm, x, y, z)

    # 测试缓存键与共轭属性相关性
    def test_cache_key_is_conj(self):
        # 使用 FakeTensorMode 上下文
        with FakeTensorMode() as fm:
            # 创建随机复数张量 x, y 和 z，并设置 z 的共轭属性
            x = torch.randn(4, 3, dtype=torch.complex64)
            y = torch.randn(4, 3, dtype=torch.complex64)
            z = torch.randn(4, 3, dtype=torch.complex64)
            torch._C._set_conj(z, not z.is_conj())
            # 调用 _test_cache_key 方法测试缓存键
            self._test_cache_key(fm, x, y, z)

    # 测试缓存键与负数属性相关性
    def test_cache_key_is_neg(self):
        # 使用 FakeTensorMode 上下文
        with FakeTensorMode() as fm:
            # 创建随机复数张量 x, y 和 z，并设置 z 的负数属性
            x = torch.randn(4, 3, dtype=torch.complex64)
            y = torch.randn(4, 3, dtype=torch.complex64)
            z = torch.randn(4, 3, dtype=torch.complex64)
            torch._C._set_neg(z, not z.is_neg())
            # 调用 _test_cache_key 方法测试缓存键
            self._test_cache_key(fm, x, y, z)

    # 测试缓存键与推理模式相关性
    def test_cache_key_is_inference(self):
        # 在推理模式下创建张量 t
        with torch.inference_mode(True):
            t = torch.randn(4, 3)
        # 使用 FakeTensorMode 上下文
        with FakeTensorMode() as fm:
            # 创建随机张量 x 和 y，使用 fm.from_tensor 方法创建张量 z
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)
            z = fm.from_tensor(t)
            # 调用 _test_cache_key 方法测试缓存键
            self._test_cache_key(fm, x, y, z)

    # 测试缓存键与常量相关性
    def test_cache_key_constants(self):
        # 使用 FakeTensorMode 上下文
        with FakeTensorMode() as fm:
            # 测试浮点数 1.0 和整数 1 的缓存键不同
            self._test_cache_key(fm, 1.0, 1.0, 1)
            # 测试浮点数 0.0 的缓存键
            self._test_cache_key(fm, 0.0, 0.0, 0)

    # 辅助方法，用于断言记录的命中和未命中次数
    def assertHitsMisses(self, hits, misses):
        """
        Helper to assert on the number of recorded hits and misses.
        """
        # 获取 FakeTensorMode 缓存信息
        info = FakeTensorMode.cache_info()
        # 断言命中和未命中次数与预期值相等
        self.assertEqual(info.hits, hits)
        self.assertEqual(info.misses, misses)

    # 辅助方法，用于断言记录的绕过次数
    def assertBypasses(self, reason, count):
        """
        Helper to assert on the number of recorded bypasses.
        """
        # 获取 FakeTensorMode 缓存信息
        info = FakeTensorMode.cache_info()
        # 如果绕过次数大于 0，则断言绕过原因在记录中且次数正确
        if count > 0:
            self.assertIn(reason, info.bypasses)
            self.assertEqual(info.bypasses[reason], count)
        # 如果绕过次数为 0，则断言绕过原因不在记录中
        else:
            self.assertNotIn(reason, info.bypasses)
    def test_cache_hit(self):
        """
        Test that cache hit/miss counters are updated correctly.
        """
        # 进入 FakeTensorMode 上下文环境
        with FakeTensorMode():
            # 创建两个随机张量 x 和 y
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)

            # 清空缓存计数器
            FakeTensorMode.cache_clear()
            # 断言缓存命中和未命中的次数为 0
            self.assertHitsMisses(0, 0)
            # 执行第一次张量加法
            res1 = x + y
            # 断言缓存命中次数为 0，未命中次数为 1
            self.assertHitsMisses(0, 1)
            # 执行第二次张量加法
            res2 = x + y
            # 断言缓存命中次数为 1，未命中次数为 1
            self.assertHitsMisses(1, 1)

            # 断言两次操作的结果张量元数据相同
            self.assertEqual(
                extract_tensor_metadata(res1),
                extract_tensor_metadata(res2),
            )

    def test_cache_bypass(self):
        """
        Test that cache bypass counters are updated correctly.
        """
        # 进入 FakeTensorMode 上下文环境
        with FakeTensorMode():
            # 创建一个随机张量 x
            x = torch.randn(1, 2)

            # 清空缓存计数器
            FakeTensorMode.cache_clear()
            # 断言 "inplace view" 的缓存绕过次数为 0
            self.assertBypasses("inplace view", 0)

            # 执行张量的原地视图操作
            x.unsqueeze_(0)
            # 断言 "inplace view" 的缓存绕过次数为 1
            self.assertBypasses("inplace view", 1)

    def test_cache_default_dtype(self):
        """
        Test that the default dtype is respected when serving cached results.
        """
        # 进入 FakeTensorMode 上下文环境
        with FakeTensorMode():
            # 创建一个 torch.int32 类型的张量 x
            x = torch.tensor([1, 2], dtype=torch.int32)
            # 设置默认数据类型为 torch.float32
            torch.set_default_dtype(torch.float32)

            # 清空缓存计数器
            FakeTensorMode.cache_clear()
            # 断言缓存命中和未命中的次数为 0
            self.assertHitsMisses(0, 0)

            # 执行张量 x 加 1.0 的操作
            y = x + 1.0
            # 断言结果张量的数据类型为 torch.float32
            self.assertEqual(y.dtype, torch.float32)
            # 断言缓存命中次数为 0，未命中次数为 1
            self.assertHitsMisses(0, 1)

            # 设置默认数据类型为 torch.float16
            torch.set_default_dtype(torch.float16)
            # 再次执行张量 x 加 1.0 的操作
            y = x + 1.0
            # 断言结果张量的数据类型为 torch.float16
            self.assertEqual(y.dtype, torch.float16)
            # 断言缓存命中次数为 0，未命中次数为 2
            self.assertHitsMisses(0, 2)

            # 设置默认数据类型为 torch.float32
            torch.set_default_dtype(torch.float32)
            # 再次执行张量 x 加 1.0 的操作
            y = x + 1.0
            # 断言结果张量的数据类型为 torch.float32
            self.assertEqual(y.dtype, torch.float32)
            # 断言缓存命中次数为 1，未命中次数为 2
            self.assertHitsMisses(1, 2)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_cache_default_device(self):
        """
        Test that the default device is respected when serving cached results.
        """
        # 进入 FakeTensorMode 上下文环境
        with FakeTensorMode():
            # 清空缓存计数器
            FakeTensorMode.cache_clear()
            # 断言缓存命中和未命中的次数为 0
            self.assertHitsMisses(0, 0)

            # 设置默认设备为 "cpu"，创建张量 x
            torch.set_default_device("cpu")
            x = torch.tensor([1, 2])
            # 执行张量 x 加 1.0 的操作
            y = x + 1.0
            # 断言结果张量的设备类型为 "cpu"
            self.assertEqual(y.device.type, "cpu")
            # 断言缓存命中次数为 0，未命中次数为 1
            self.assertHitsMisses(0, 1)

            # 设置默认设备为 "cuda"，再次创建张量 x
            torch.set_default_device("cuda")
            x = torch.tensor([1, 2])
            # 执行张量 x 加 1.0 的操作
            y = x + 1.0
            # 断言结果张量的设备类型为 "cuda"
            self.assertEqual(y.device.type, "cuda")
            # 断言缓存命中次数为 0，未命中次数为 2
            self.assertHitsMisses(0, 2)

            # 设置默认设备为 "cpu"，再次创建张量 x
            torch.set_default_device("cpu")
            x = torch.tensor([1, 2])
            # 执行张量 x 加 1.0 的操作
            y = x + 1.0
            # 断言结果张量的设备类型为 "cpu"
            self.assertEqual(y.device.type, "cpu")
            # 断言缓存命中次数为 1，未命中次数为 2
            self.assertHitsMisses(1, 2)
    def test_cache_inplace_op(self):
        """
        Test that inplace ops served from the cache correctly reference the
        input parameter.
        """
        # 进入 FakeTensorMode 上下文环境，用于模拟张量操作
        with FakeTensorMode():
            # 创建两个随机张量 x 和 y
            x = torch.randn(1, 2)
            y = torch.randn(1, 2)

            # 清空 FakeTensorMode 的缓存，断言缓存命中数为 0，未命中数为 0
            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            # 执行 inplace 加法操作，并断言缓存命中数为 0，未命中数为 1
            z = x.add_(y)
            self.assertHitsMisses(0, 1)
            # 断言 z 和 x 是同一个对象
            self.assertEqual(id(x), id(z))

            # 再次执行 inplace 加法操作，并断言缓存命中数为 1，未命中数为 1
            w = x.add_(y)
            self.assertHitsMisses(1, 1)
            # 断言 w 和 x 是同一个对象
            self.assertEqual(id(x), id(w))

    def test_cache_view_op(self):
        """
        Test that view ops are handled correctly when served from the cache.
        """
        # 进入 FakeTensorMode 上下文环境，用于模拟张量操作
        with FakeTensorMode():
            # 创建两个需要梯度的张量，并克隆第二个张量并进行 view 操作
            x1 = torch.ones(2, requires_grad=True).clone()
            x2 = torch.ones(2, requires_grad=True).clone()
            y2 = x2.view(-1)

            # 测试对非 view 张量执行操作，然后对 view 张量执行相同操作，断言 view 属性设置正确
            z1 = x1.mul_(2)
            self.assertFalse(z1._is_view())

            z2 = y2.mul_(2)
            self.assertTrue(z2._is_view())

            # 现在反过来测试：先对 view 张量执行操作，然后对非 view 张量执行相同操作
            z2 = y2.mul_(2)
            self.assertTrue(z2._is_view())

            z1 = x1.mul_(2)
            self.assertFalse(z1._is_view())

    def test_cache_dispatch_key_set(self):
        """
        Test that operations that change the dispatch key set bypass caching.
        """
        # 进入 FakeTensorMode 上下文环境，用于模拟张量操作
        with FakeTensorMode():
            # 清空 FakeTensorMode 的缓存，断言绕过缓存操作次数为 0
            FakeTensorMode.cache_clear()
            self.assertBypasses("dispatch_key_set mismatch", 0)

            # 调用 torch._efficientzerotensor(3)，断言返回的张量是零张量，绕过缓存操作次数为 1
            x = torch._efficientzerotensor(3)
            self.assertTrue(x._is_zerotensor())
            self.assertBypasses("dispatch_key_set mismatch", 1)

            # 再次调用 torch._efficientzerotensor(3)，断言返回的张量是零张量，绕过缓存操作次数为 2
            y = torch._efficientzerotensor(3)
            self.assertTrue(y._is_zerotensor())
            self.assertBypasses("dispatch_key_set mismatch", 2)
    # 定义一个测试方法，用于测试缓存是否正确处理推断模式
    def test_inference_mode(self):
        """
        Test that caching handles inference mode correctly.
        """
        # 使用 FakeTensorMode 上下文
        with FakeTensorMode():
            # 创建两个随机张量
            x = torch.randn(4, 3)
            y = torch.randn(4, 3)

            # 清空 FakeTensorMode 的缓存并断言命中和未命中次数为 0
            FakeTensorMode.cache_clear()
            self.assertHitsMisses(0, 0)

            # 当推断模式不同时，预期会未命中缓存
            res1 = x + y
            with torch.inference_mode():
                res2 = x + y

            # 断言此时的命中和未命中次数为 0 和 2
            self.assertHitsMisses(0, 2)
            # 断言 res1 不处于推断模式
            self.assertFalse(res1.is_inference())
            # 断言 res2 处于推断模式
            self.assertTrue(res2.is_inference())

            # 第二次操作应命中缓存
            res3 = x + y

            # 断言此时的命中和未命中次数为 1 和 2
            self.assertHitsMisses(1, 2)
            # 断言 res3 不处于推断模式，并且与 res1 元数据相同
            self.assertFalse(res3.is_inference())
            self.assertEqual(
                extract_tensor_metadata(res1),
                extract_tensor_metadata(res3),
            )

            # 在推断模式下再次执行操作
            with torch.inference_mode():
                res4 = x + y

            # 断言此时的命中和未命中次数为 2 和 2
            self.assertHitsMisses(2, 2)
            # 断言 res4 处于推断模式，并且与 res2 元数据相同
            self.assertTrue(res4.is_inference())
            self.assertEqual(
                extract_tensor_metadata(res2),
                extract_tensor_metadata(res4),
            )
# 如果当前脚本被直接执行（而不是被导入到其他模块中），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```