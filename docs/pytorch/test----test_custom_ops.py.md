# `.\pytorch\test\test_custom_ops.py`

```py
# Owner(s): ["module: custom-operators"]

# 从 torch.testing._internal.common_utils 导入所有内容，禁止 F403 错误提示
from torch.testing._internal.common_utils import *  # noqa: F403
# 从 torch.testing._internal.common_device_type 导入所有内容，禁止 F403 错误提示
from torch.testing._internal.common_device_type import *  # noqa: F403
import collections  # 导入 Python 标准库 collections

import itertools  # 导入 Python 标准库 itertools
import os  # 导入 Python 标准库 os
import re  # 导入 Python 标准库 re
import typing  # 导入 Python 标准库 typing

# 导入 torch._custom_ops 模块，并赋值给 custom_ops
import torch._custom_ops as custom_ops

# 导入 torch.testing._internal.optests 模块，并赋值给 optests
import torch.testing._internal.optests as optests
# 导入 torch.utils.cpp_extension 模块
import torch.utils.cpp_extension

# 从 functorch 模块导入 make_fx 函数
from functorch import make_fx
# 导入 Tensor 类型
from torch import Tensor
# 从 torch._custom_op.impl 导入 custom_op, CustomOp, infer_schema
from torch._custom_op.impl import custom_op, CustomOp, infer_schema
# 从 torch._library.infer_schema 导入 tuple_to_list 函数
from torch._library.infer_schema import tuple_to_list
# 从 torch._utils_internal 导入 get_file_path_2 函数
from torch._utils_internal import get_file_path_2
# 从 torch.testing._internal 导入 custom_op_db 模块
from torch.testing._internal import custom_op_db
# 从 torch.testing._internal.common_cuda 导入 TEST_CUDA 常量
from torch.testing._internal.common_cuda import TEST_CUDA
# 从 torch.testing._internal.custom_op_db 导入 numpy_nonzero 函数
from torch.testing._internal.custom_op_db import numpy_nonzero
# 导入 typing 模块所有内容，禁止 F403 错误提示
from typing import *  # noqa: F403
# 导入 numpy 模块，并赋值给 np
import numpy as np


# 定义一个装饰器函数 requires_compile，用于跳过在 Windows 下不兼容的测试
def requires_compile(fun):
    fun = unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")(fun)
    return fun


# 定义一个测试类 CustomOpTestCaseBase，继承自 TestCase
class CustomOpTestCaseBase(TestCase):
    test_ns = "_test_custom_op"  # 类变量 test_ns 设置为 "_test_custom_op"

    # 设置测试环境，在每个测试方法执行前调用
    def setUp(self):
        super().setUp()  # 调用父类的 setUp 方法
        self.libraries = []  # 初始化实例变量 libraries 为空列表

    # 清理测试环境，在每个测试方法执行后调用
    def tearDown(self):
        super().tearDown()  # 调用父类的 tearDown 方法
        import torch._custom_op  # 导入 torch._custom_op 模块

        keys = list(torch._custom_op.impl.global_registry.keys())  # 获取全局注册表中的所有键
        # 遍历全局注册表中的键
        for key in keys:
            # 如果键不以指定的命名空间开头，则跳过
            if not key.startswith(f"{self.test_ns}::"):
                continue
            # 销毁对应键的全局注册对象
            torch._custom_op.impl.global_registry[key]._destroy()
        # 删除 torch.ops 中指定命名空间的属性
        if hasattr(torch.ops, self.test_ns):
            delattr(torch.ops, self.test_ns)
        # 销毁 libraries 列表中的所有库对象
        for lib in self.libraries:
            lib._destroy()
        del self.libraries  # 删除实例变量 libraries


    # 返回当前测试类的命名空间
    def ns(self):
        return getattr(torch.ops, self.test_ns)


    # 创建并返回一个自定义库对象，命名空间为 test_ns，类型为 "FRAGMENT"
    def lib(self):
        result = torch.library.Library(self.test_ns, "FRAGMENT")  # noqa: TOR901
        self.libraries.append(result)  # 将创建的库对象添加到实例变量 libraries 中
        return result


    # 根据操作的完全限定名 qualname，获取并返回对应的自定义操作对象
    def get_op(self, qualname):
        return torch._custom_op.impl.get_op(qualname)


# 使用 requires_compile 装饰器修饰 TestCustomOpTesting 类，
# 在 Windows 环境下跳过 torch.compile 不兼容的测试
@requires_compile
class TestCustomOpTesting(CustomOpTestCaseBase):
    # 参数化测试方法 test_aot_autograd_check_degenerate_cases
    @parametrize("check_gradients", (False, "auto"))
    @parametrize("dynamic", (True, False))
    def test_aot_autograd_check_degenerate_cases(
        self, device, dynamic, check_gradients
    ):
        # 定义一个简单的函数 simple，用于克隆输入张量并返回
        def simple(x):
            return x.clone()

        # 使用指定设备生成一个随机张量 x
        x = torch.randn(3, device=device)
        # 调用 optests.aot_autograd_check 进行自动求导的检查，期望不会引发异常
        optests.aot_autograd_check(
            simple, (x,), {}, dynamic=dynamic, check_gradients=check_gradients
        )

        # 定义一个函数 outputs_dont_require_grad，用于返回输入张量的去梯度化版本
        def outputs_dont_require_grad(x):
            return x.detach()

        # 使用指定设备生成一个随机张量 y，并设置 requires_grad=True
        y = torch.randn(3, device=device, requires_grad=True)
        # 调用 optests.aot_autograd_check 进行自动求导的检查，期望不会引发异常
        optests.aot_autograd_check(
            simple, (y,), {}, dynamic=dynamic, check_gradients=check_gradients
        )

        # 定义一个函数 no_outputs，用于返回输入张量的去梯度化版本
        def no_outputs(x):
            return x.detach()

        # 使用指定设备生成一个随机张量 x，并设置 requires_grad=True
        x = torch.randn(3, device=device, requires_grad=True)
        # 使用指定设备生成一个随机张量 y，并设置 requires_grad=False
        y = torch.randn(3, device=device, requires_grad=False)
        # 分别调用 optests.aot_autograd_check 进行自动求导的检查，期望不会引发异常
        optests.aot_autograd_check(
            no_outputs, (x,), {}, dynamic=dynamic, check_gradients=check_gradients
        )
        optests.aot_autograd_check(
            no_outputs, (y,), {}, dynamic=dynamic, check_gradients=check_gradients
        )

    # 定义测试函数 test_incorrect_schema_mutation，输入设备信息 device
    def test_incorrect_schema_mutation(self, device):
        # 调用 self.lib() 返回一个库对象 lib
        lib = self.lib()
        # 定义一个函数 foo，其输入为 Tensor x，返回值为 Tensor
        lib.define("foo(Tensor x) -> Tensor")
        # 获取 self.ns().foo.default 对象，并赋值给 op
        op = self.ns().foo.default

        # 定义一个自定义的 Torch 自动求导函数 Foo
        class Foo(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，接受输入 x
            def forward(ctx, x):
                # 创建一个保护对象 guard
                guard = torch._C._AutoDispatchBelowAutograd()
                try:
                    # 调用 op(x) 执行前向传播
                    return op(x)
                finally:
                    # 在 finally 中删除 guard 对象
                    del guard

            @staticmethod
            # 后向传播函数，接受梯度 gx
            def backward(ctx, gx):
                return gx

        # 定义 foo_impl 函数，实现对输入 x 执行 sin_() 操作并返回克隆版本
        def foo_impl(x):
            x.sin_()
            return x.clone()

        # 将 Foo.apply 作为 "Autograd" 类型的实现注册到 lib 中的 foo 函数
        lib.impl("foo", Foo.apply, "Autograd")
        # 将 foo_impl 作为 "CPU" 类型的实现注册到 lib 中的 foo 函数
        lib.impl("foo", foo_impl, "CPU")
        # 将 foo_impl 作为 "CUDA" 类型的实现注册到 lib 中的 foo 函数
        lib.impl("foo", foo_impl, "CUDA")

        # 使用指定设备生成一个张量 x，并设置 requires_grad=True
        x = torch.tensor(3.14159 / 3, requires_grad=True, device=device)
        # 使用 self.assertRaisesRegex 检查 op 在给定参数 x 下的行为是否符合预期
        with self.assertRaisesRegex(
            optests.OpCheckError, "Argument x is not defined as mutable but was mutated"
        ):
            # 调用 torch.library.opcheck 检查 op 在输入 (x,) 下的行为
            torch.library.opcheck(op, (x,), {})
    # 测试不正确的模式视图
    def test_incorrect_schema_view(self, device):
        # 创建库对象
        lib = self.lib()
        # 定义函数签名
        lib.define("foo(Tensor x) -> Tensor")
        # 获取操作符
        op = self.ns().foo.default

        # 定义一个继承自torch.autograd.Function的类
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 模拟AutoDispatchBelowADInplaceOrView，它未绑定到Python中
                with torch._C._AutoDispatchBelowAutograd():
                    # 排除torch._C.DispatchKey.ADInplaceOrView，以免影响
                    with torch._C._ExcludeDispatchKeyGuard(
                        torch._C.DispatchKeySet(torch._C.DispatchKey.ADInplaceOrView)
                    ):
                        # 调用操作符的前向传播
                        return op(x)

            @staticmethod
            def backward(ctx, gx):
                # 反向传播直接返回梯度
                return gx

        # 实现CPU函数foo_impl，将输入张量x按其自身形状重塑
        def foo_impl(x):
            return x.view_as(x)

        # 定义元函数foo_meta，将输入张量x按其自身形状重塑
        def foo_meta(x):
            return x.view_as(x)

        # 注册Autograd版本的foo函数实现到库中
        lib.impl("foo", Foo.apply, "Autograd")
        # 注册CPU版本的foo_impl函数实现到库中
        lib.impl("foo", foo_impl, "CPU")
        # 注册Meta版本的foo_meta函数实现到库中
        lib.impl("foo", foo_meta, "Meta")

        # 创建一个需要梯度的张量x
        x = torch.tensor(3.14159 / 3, requires_grad=True)
        
        # 断言调用opcheck时抛出特定异常，并检查异常信息
        with self.assertRaisesRegex(
            optests.OpCheckError,
            "Argument x is not defined to alias output but was aliasing",
        ):
            # 检查foo操作在输入x上的行为是否符合预期
            torch.library.opcheck(op, (x,), {})

    # 测试缺少抽象实现
    def test_missing_abstract_impl(self, device):
        # 创建库对象
        lib = self.lib()
        # 定义函数签名
        lib.define("foo(Tensor x) -> Tensor")
        # 获取操作符
        op = self.ns().foo.default

        # 定义一个继承自torch.autograd.Function的类
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 模拟AutoDispatchBelowAutograd，用于处理前向传播
                with torch._C._AutoDispatchBelowAutograd():
                    # 调用操作符的前向传播
                    return op(x)

            @staticmethod
            def backward(ctx, gx):
                # 返回梯度的2倍
                return 2 * gx

        # 实现CPU函数foo_impl，对输入张量x执行平方操作并返回
        def foo_impl(x):
            return torch.tensor(x.cpu().numpy() ** 2, device=x.device)

        # 注册Autograd版本的foo函数实现到库中
        lib.impl("foo", Foo.apply, "Autograd")
        # 注册CPU版本的foo_impl函数实现到库中
        lib.impl("foo", foo_impl, "CPU")
        # 注册CUDA版本的foo_impl函数实现到库中
        lib.impl("foo", foo_impl, "CUDA")

        # 创建一个需要梯度的张量x
        x = torch.tensor([0, 1.0], requires_grad=True)
        
        # 断言调用opcheck时抛出特定异常，并检查异常信息
        with self.assertRaisesRegex(
            optests.OpCheckError,
            "_test_custom_op.foo.default",
        ):
            # 检查foo操作在输入x上的行为是否符合预期
            torch.library.opcheck(op, (x,), {})

    # 如果是Torch Dynamo测试，则跳过该测试用例
    def test_incorrect_abstract_impl(self, device):
        # 创建库实例
        lib = self.lib()
        # 定义函数签名
        lib.define("foo(Tensor x) -> Tensor")
        # 获取操作符
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 模拟 AutoDispatchBelowADInplaceOrView，该模块未绑定到 Python 中
                # 创建 AutoDispatchBelowAutograd 的实例
                guard = torch._C._AutoDispatchBelowAutograd()
                # 创建 ExcludeDispatchKeyGuard 的实例，限制 ADInplaceOrView DispatchKey
                guard2 = torch._C.ExcludeDispatchKeyGuard(
                    torch._C.DispatchKeySet(torch._C.DispatchKey.ADInplaceOrView)
                )
                try:
                    # 调用操作符进行前向传播计算
                    return op(x)
                finally:
                    # 清除 guard 和 guard2 实例
                    del guard
                    del guard2

            @staticmethod
            def backward(ctx, gx):
                # 反向传播直接返回梯度 gx
                return gx

        def foo_impl(x):
            # 实现 CPU 和 CUDA 上的具体操作
            return x**2

        def foo_meta(x):
            # 实现 Meta 操作
            return x.unsqueeze(1) ** 2

        # 将不同实现注册到库中
        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")
        lib.impl("foo", foo_meta, "Meta")

        # 创建输入张量 x，要求计算梯度
        x = torch.tensor([0, 1.0], requires_grad=True)
        # 断言操作符的行为符合预期
        with self.assertRaisesRegex(optests.OpCheckError, "Shapes .* are not equal"):
            torch.library.opcheck(op, (x,), {})

    def test_missing_functionalization(self, device):
        # 创建库实例
        lib = self.lib()
        # 定义带有 alias annotation 的函数签名
        lib.define("foo(Tensor(a!) x) -> Tensor(a!)")
        # 获取操作符
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 标记输入张量 x 为脏数据
                ctx.mark_dirty(x)
                # 使用 AutoDispatchBelowAutograd 自动分发环境
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            def backward(ctx, gx):
                # 反向传播直接返回梯度 gx
                return gx

        def foo_impl(x):
            # CPU 上的具体操作实现
            return x.sin_()

        def foo_meta(x):
            # Meta 操作实现
            return x

        # 将不同实现注册到库中
        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")
        lib.impl("foo", foo_meta, "Meta")

        # 创建输入张量 x 和其克隆 y
        x = torch.tensor([0, 1.0])
        y = x.clone()
        # 断言操作符的行为符合预期
        with self.assertRaisesRegex(
            optests.OpCheckError,
            "We only support functionalizing operators whose outputs do not have alias annotations",
        ):
            torch.library.opcheck(op, (y,), {})
    # 测试自动求导是否在后端注册
    def test_autograd_registered_at_backend(self, device):
        # 获取当前库
        lib = self.lib()
        # 定义 foo 函数签名，输入为 Tensor x，输出为 Tensor
        lib.define("foo(Tensor x) -> Tensor")
        # 获取命名空间下的 foo 函数默认实现
        op = self.ns().foo.default

        # 定义一个继承自 torch.autograd.Function 的 Foo 类
        class Foo(torch.autograd.Function):
            @staticmethod
            # 前向传播函数，返回输入的克隆
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            # 反向传播函数，返回输入的梯度乘以 0.5
            def backward(ctx, gx):
                return gx * 0.5

        # 将 Foo 类的 apply 方法注册为 CPU 和 CUDA 上的 foo 函数实现
        lib.impl("foo", Foo.apply, "CPU")
        lib.impl("foo", Foo.apply, "CUDA")
        # 在 Meta 上使用 lambda 函数注册 foo 函数实现
        lib.impl("foo", lambda x: x.clone(), "Meta")

        # 创建一个随机张量 x，并设置 requires_grad=True
        x = torch.randn([], requires_grad=True)

        # 使用 assertRaisesRegex 检测是否抛出 OpCheckError 异常，
        # 提示 "does not have an autograd kernel"
        with self.assertRaisesRegex(
            torch.testing._internal.optests.OpCheckError,
            "does not have an autograd kernel",
        ):
            # 调用 torch.library.opcheck 检查 op 函数在 x 上的操作
            torch.library.opcheck(op, (x,), {})

        # 删除库实例 lib
        del lib

    # 测试全局状态变异
    def test_global_state_mutation(self, device):
        # 获取当前库
        lib = self.lib()
        # 定义 foo 函数签名，输入为 Tensor x，输出为 Tensor
        lib.define("foo(Tensor x) -> Tensor")
        # 获取命名空间下的 foo 函数默认实现
        op = self.ns().foo.default

        # 定义一个继承自 torch.autograd.Function 的 Foo 类
        class Foo(torch.autograd.Function):
            invoked = 0

            @staticmethod
            # 前向传播函数，返回输入的克隆乘以 invoked 的增量
            def forward(ctx, x):
                Foo.invoked += 1
                return x.clone() * Foo.invoked

            @staticmethod
            # 反向传播函数，返回输入的梯度
            def backward(ctx, gx):
                return gx

        # 将 Foo 类的 apply 方法注册为 CompositeImplicitAutograd 上的 foo 函数实现
        lib.impl("foo", Foo.apply, "CompositeImplicitAutograd")

        # 创建一个张量 x，并设置 requires_grad=True
        x = torch.tensor(3.14159 / 3, requires_grad=True)
        
        # 使用 assertRaisesRegex 检测是否抛出 OpCheckError 异常，
        # 提示 "eager-mode PyTorch vs AOTAutograd"
        with self.assertRaisesRegex(
            optests.OpCheckError, "eager-mode PyTorch vs AOTAutograd"
        ):
            # 调用 torch.library.opcheck 检查 op 函数在 x 上的操作
            torch.library.opcheck(op, (x,), {})

    # 测试 opcheck_opinfo 函数
    @ops(custom_op_db.custom_op_db, dtypes=OpDTypes.any_one)
    def test_opcheck_opinfo(self, device, dtype, op):
        # 遍历 op 的样本输入
        for sample_input in op.sample_inputs(
            device, dtype, requires_grad=op.supports_autograd
        ):
            # 准备 op 的参数
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            # 调用 torch.library.opcheck 检查 op.op 函数在 args 和 kwargs 上的操作
            torch.library.opcheck(
                op.op,
                args,
                kwargs,
            )

    # 测试 opcheck 失败基本情况
    def test_opcheck_fails_basic(self, device):
        # 定义名为 foo 的自定义操作
        @custom_op(f"{self.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            ...

        # 将 foo 函数实现注册为 "cpu" 和 "cuda" 上的 foo 函数
        @foo.impl(["cpu", "cuda"])
        def foo_impl(x):
            return x.sum()

        # 创建一个随机张量 x，并设置 requires_grad=True
        x = torch.randn(3, device=device, requires_grad=True)
        
        # 使用 assertRaisesRegex 检测是否抛出 OpCheckError 异常，
        # 提示 "Autograd has not been implemented for operator"
        with self.assertRaisesRegex(
            optests.OpCheckError, "Autograd has not been implemented for operator"
        ):
            # 调用 self.get_op 获取 f"{self.test_ns}::foo" 对应的操作，并检查在 x 上的操作
            torch.library.opcheck(self.get_op(f"{self.test_ns}::foo"), (x,), {})
    def test_autograd_registration_check_autograd_kernel(self, device):
        # 获取当前库实例
        lib = self.lib()
        # 定义名为 "foo" 的函数签名，接受一个 Tensor 参数并返回一个 Tensor
        lib.define("foo(Tensor x) -> Tensor")
        # 获取默认实现函数对象 op
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 在自动求导下设置自动调度
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            def backward(ctx, gx):
                return gx

        def foo_impl(x):
            # 返回输入张量的正弦值
            return x.sin()

        # 注册 Autograd 实现给 "foo" 函数
        lib.impl("foo", Foo.apply, "Autograd")
        # 注册 CPU 上的实现给 "foo" 函数
        lib.impl("foo", foo_impl, "CPU")
        # 注册 CUDA 上的实现给 "foo" 函数
        lib.impl("foo", foo_impl, "CUDA")

        x = torch.randn(3, requires_grad=True, device=device)
        # 应该不会引发异常
        optests.autograd_registration_check(op, (x,), {})



    def test_autograd_registration_check_compositeimplicitautograd(self, device):
        # 获取当前库实例
        lib = self.lib()
        # 定义名为 "foo" 的函数签名，接受一个 Tensor 参数并返回一个 Tensor
        lib.define("foo(Tensor x) -> Tensor")
        # 获取默认实现函数对象 op
        op = self.ns().foo.default

        def foo_impl(x):
            # 返回输入张量的正弦值再求余弦值
            return x.sin().cos()

        # 注册 CompositeImplicitAutograd 实现给 "foo" 函数
        lib.impl("foo", foo_impl, "CompositeImplicitAutograd")

        x = torch.randn(3, requires_grad=True, device=device)
        # 应该不会引发异常
        optests.autograd_registration_check(op, (x,), {})



    def test_autograd_registration_check_incorrect_composite(self, device):
        # 获取当前库实例
        lib = self.lib()
        # 定义名为 "foo" 的函数签名，接受一个 Tensor 参数并返回一个 Tensor
        lib.define("foo(Tensor x) -> Tensor")
        # 获取默认实现函数对象 op
        op = self.ns().foo.default

        def foo_impl(x):
            # 返回输入张量的正弦值再求余弦值
            return x.sin().cos()

        # 注册 CompositeExplicitAutograd 实现给 "foo" 函数
        lib.impl("foo", foo_impl, "CompositeExplicitAutograd")

        x = torch.randn(3, requires_grad=True, device=device)
        # 应该引发 AssertionError，提示注册不正确
        with self.assertRaisesRegex(AssertionError, "incorrectly registered"):
            optests.autograd_registration_check(op, (x,), {})



    def test_autograd_registration_check_incorrect(self, device):
        # 获取当前库实例
        lib = self.lib()
        # 定义名为 "foo" 的函数签名，接受一个 Tensor 参数并返回一个 Tensor
        lib.define("foo(Tensor x) -> Tensor")
        # 获取默认实现函数对象 op
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 返回输入张量的正弦值
                return torch.sin(x)

            @staticmethod
            def backward(ctx, gx):
                return gx

        # 注册 CPU 上的 Foo 类实现给 "foo" 函数
        lib.impl("foo", Foo.apply, "CPU")
        # 注册 CUDA 上的 Foo 类实现给 "foo" 函数
        lib.impl("foo", Foo.apply, "CUDA")

        x = torch.randn(3, requires_grad=True, device=device)
        # 应该引发 AssertionError，提示注册不正确
        with self.assertRaisesRegex(AssertionError, "incorrectly registered"):
            optests.autograd_registration_check(op, (x,), {})
    # 定义一个测试方法，用于测试断言引发异常的正则表达式匹配情况
    def test_assert_raises_regex(self, device):
        # 从 torch.testing._internal.optests.aot_autograd 导入 assert_raises_regex 函数
        from torch.testing._internal.optests.aot_autograd import assert_raises_regex

        # 断言抛出 RuntimeError 异常，并且异常信息中包含字符串 "c"
        with assert_raises_regex(RuntimeError, "c"):
            raise RuntimeError("abcd")

        # 断言抛出 RuntimeError 异常，并且异常信息中以 "c" 开头
        with assert_raises_regex(RuntimeError, "c.*"):
            raise RuntimeError("abcd")

        # 断言抛出 AssertionError 异常，并且异常信息中包含 "instead got"
        with self.assertRaisesRegex(AssertionError, "instead got"):
            # 内部嵌套的断言，期望抛出 RuntimeError 异常，且异常信息匹配 "c.*"
            with assert_raises_regex(RuntimeError, "c.*"):
                raise ValueError("abcd")

        # 断言抛出 AssertionError 异常，并且异常信息中包含 "Expected exception"
        with self.assertRaisesRegex(AssertionError, "Expected exception"):
            # 内部嵌套的断言，期望抛出 RuntimeError 异常，但是这里没有引发异常
            with assert_raises_regex(RuntimeError, "c.*"):
                pass

        # 断言抛出 AssertionError 异常，并且异常信息中包含 "to match regex"
        with self.assertRaisesRegex(AssertionError, "to match regex"):
            # 断言抛出 RuntimeError 异常，但是异常信息中不包含 "f"
            with assert_raises_regex(RuntimeError, "f"):
                raise RuntimeError("abcd")
class TestCustomOp(CustomOpTestCaseBase):
    test_ns = "_test_custom_op"

    @requires_compile
    def test_functionalize_error(self):
        # 使用指定的命名空间和版本定义库
        with torch.library._scoped_library(self.test_ns, "FRAGMENT") as lib:
            # 定义函数 foo 的签名及其输入参数的 alias annotation
            lib.define("foo(Tensor(a!) x) -> Tensor(a!)")

            def foo(x):
                # 返回输入张量 x 的正弦函数
                return x.sin_()

            # 实现函数 foo，采用 CompositeExplicitAutograd 模式
            lib.impl("foo", foo, "CompositeExplicitAutograd")
            # 获取 foo 操作的实例
            foo_op = self.get_op(f"{self.test_ns}::foo")

            # 定义函数 bar 的签名
            lib.define("bar(Tensor(a) x) -> Tensor(a)")

            def bar(x):
                # 返回将输入张量 x 视图重塑为一维的结果
                return x.view(-1)

            # 实现函数 bar，采用 CompositeExplicitAutograd 模式
            lib.impl("bar", bar, "CompositeExplicitAutograd")
            # 获取 bar 操作的实例
            bar_op = self.get_op(f"{self.test_ns}::bar")

            # 定义错误消息的正则表达式模式
            msg = r".*We only support functionalizing operators whose outputs do not have alias annotations"

            # 创建一个随机张量 x，包含三个元素
            x = torch.randn(3)

            # 编译函数 f，使用 AOT eager 模式，生成完整图
            @torch.compile(backend="aot_eager", fullgraph=True)
            def f(x):
                return foo_op(x)

            # 编译函数 g，使用 AOT eager 模式，生成完整图
            @torch.compile(backend="aot_eager", fullgraph=True)
            def g(x):
                return bar_op(x)

            # 断言调用函数 f(x) 和 g(x) 时抛出 RuntimeError 异常，并匹配错误消息
            with self.assertRaisesRegex(RuntimeError, msg):
                f(x)
            with self.assertRaisesRegex(RuntimeError, msg):
                g(x)

    def test_invalid_schemas(self):
        # 断言调用 custom_op 函数时抛出 AssertionError 异常，并匹配错误消息
        with self.assertRaisesRegex(AssertionError, "Invalid function schema: foo"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", "(")

    def test_invalid_qualname(self):
        # 断言调用 custom_op 函数时抛出 ValueError 异常，并匹配错误消息
        with self.assertRaisesRegex(ValueError, "overload"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo.Tensor", "() -> ()")

    def test_name_must_match(self):
        # 断言自定义操作的名称不匹配时抛出 ValueError 异常，并匹配错误消息
        with self.assertRaisesRegex(ValueError, "to have name"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def baz(x: Tensor) -> Tensor:
                raise NotImplementedError

    def test_unsupported_schemas(self):
        # 断言调用 custom_op 函数时抛出 ValueError 异常，并匹配错误消息
        with self.assertRaisesRegex(ValueError, "only supports functional"):
            custom_ops.custom_op(
                f"{TestCustomOp.test_ns}::foo", "(Tensor(a!) x) -> Tensor(a)"
            )(foo)
        with self.assertRaisesRegex(ValueError, "only supports functional"):
            custom_ops.custom_op(
                f"{TestCustomOp.test_ns}::foo", "(Tensor(a) x) -> Tensor(a)"
            )(foo)
        with self.assertRaisesRegex(ValueError, "only supports functional"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", "(Tensor x) -> ()")(
                foo
            )
        with self.assertRaisesRegex(ValueError, "self"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", "(Tensor self) -> ()")(
                foo
            )

    # Tests for the older custom_op API
    def test_schema_matches_signature(self):
        # 测试自定义操作的签名是否匹配特定的错误信息
        with self.assertRaisesRegex(ValueError, "signature to match"):
            # 定义自定义操作 'blah'，期望抛出数值错误，提示签名不匹配
            @custom_op(f"{TestCustomOp.test_ns}::blah", "(Tensor y) -> Tensor")
            def blah(x):
                pass

        with self.assertRaisesRegex(ValueError, "signature to match"):
            # 定义自定义操作 'blah2'，期望抛出数值错误，提示签名不匹配
            @custom_op(
                f"{TestCustomOp.test_ns}::blah2", "(Tensor x, *, Tensor y) -> Tensor"
            )
            def blah2(x, y):
                pass

        with self.assertRaisesRegex(ValueError, "signature to match"):
            # 定义自定义操作 'blah3'，期望抛出数值错误，提示签名不匹配
            @custom_op(
                f"{TestCustomOp.test_ns}::blah3",
                "(Tensor x, *, Tensor w, Tensor z) -> Tensor",
            )
            def blah3(x, *, y, z):
                pass

        with self.assertRaisesRegex(ValueError, "signature to match"):
            # 定义自定义操作 'blah4'，期望抛出数值错误，提示签名不匹配
            @custom_op(
                f"{TestCustomOp.test_ns}::blah4",
                "(Tensor x, *, Tensor z, Tensor y) -> Tensor",
            )
            def blah4(x, *, y, z):
                pass

        with self.assertRaisesRegex(ValueError, "not supported"):
            # 定义自定义操作 'blah5'，期望抛出数值错误，提示不支持的操作
            @custom_op(f"{TestCustomOp.test_ns}::blah5", "(Tensor x) -> Tensor")
            def blah5(*args):
                pass

        with self.assertRaisesRegex(ValueError, "not supported"):
            # 定义自定义操作 'blah6'，期望抛出数值错误，提示不支持的操作
            @custom_op(
                f"{TestCustomOp.test_ns}::blah6", "(*, Tensor z, Tensor y) -> Tensor"
            )
            def blah6(**kwargs):
                pass

        with self.assertRaisesRegex(ValueError, "default arguments"):
            # 定义自定义操作 'blah7'，期望抛出数值错误，提示不支持默认参数
            @custom_op(
                f"{TestCustomOp.test_ns}::blah7", "(Tensor x, *, Tensor y) -> Tensor"
            )
            def blah7(x=1, *, y):
                pass

        with self.assertRaisesRegex(ValueError, "default arguments"):
            # 定义自定义操作 'blah8'，期望抛出数值错误，提示不支持默认参数
            @custom_op(
                f"{TestCustomOp.test_ns}::blah8", "(Tensor x, *, Tensor y) -> Tensor"
            )
            def blah8(x, *, y=1):
                pass

        # 测试关键字参数在自定义操作 'blah9' 中是否工作
        @custom_op(
            f"{TestCustomOp.test_ns}::blah9", "(Tensor x, *, Tensor y) -> Tensor"
        )
        def blah9(x, *, y):
            pass
    def test_infer_schema_supported(self):
        # 定义函数 a，接受一个 Tensor 类型参数 x，并返回一个空的 Tensor
        def a(x: Tensor) -> Tensor:
            return torch.empty([])

        # 调用 infer_schema 函数，验证函数 a 的推断类型为 (Tensor x) -> Tensor
        self.assertExpectedInline(infer_schema(a), """(Tensor x) -> Tensor""")

        # 定义函数 kwonly1，接受一个 Tensor 类型参数 x，以及必须的关键字参数 y (整数) 和 z (浮点数)，返回一个空的 Tensor
        def kwonly1(x: Tensor, *, y: int, z: float) -> Tensor:
            return torch.empty([])

        # 调用 infer_schema 函数，验证函数 kwonly1 的推断类型为 (Tensor x, *, SymInt y, float z) -> Tensor
        self.assertExpectedInline(
            infer_schema(kwonly1), """(Tensor x, *, SymInt y, float z) -> Tensor"""
        )

        # 定义函数 kwonly2，只接受关键字参数 y (Tensor 类型)，返回一个空的 Tensor
        def kwonly2(*, y: Tensor) -> Tensor:
            return torch.empty([])

        # 调用 infer_schema 函数，验证函数 kwonly2 的推断类型为 (*, Tensor y) -> Tensor
        self.assertExpectedInline(infer_schema(kwonly2), """(*, Tensor y) -> Tensor""")

        # 定义函数 b，接受多个参数：x (Tensor 类型)，y (整数)，z (布尔值)，a (浮点数)，b (Tensor 类型)，c (torch.device 类型)，d (torch.types.Number 类型)
        # 返回一个包含 Tensor, int, float, bool 类型的元组
        def b(
            x: Tensor,
            y: int,
            z: bool,
            a: float,
            b: torch.dtype,
            c: torch.device,
            d: torch.types.Number,
        ) -> Tuple[Tensor, int, float, bool]:
            return torch.empty([]), 1, 0.1, True

        # 调用 infer_schema 函数，验证函数 b 的推断类型为 (Tensor x, SymInt y, bool z, float a, ScalarType b, Device c, Scalar d) -> (Tensor, SymInt, float, bool)
        self.assertExpectedInline(
            infer_schema(b),
            """(Tensor x, SymInt y, bool z, float a, ScalarType b, Device c, Scalar d) -> (Tensor, SymInt, float, bool)""",
        )

        # 定义函数 c，接受多个参数：x (Tensor 类型)，y (Tensor 列表)，z (可选的 Tensor 类型)，w (可选的 Tensor 列表)
        # 返回一个 Tensor 列表
        def c(
            x: Tensor,
            y: Sequence[Tensor],
            z: Optional[Tensor],
            w: Sequence[Optional[Tensor]],
        ) -> List[Tensor]:
            return [torch.empty([])]

        # 调用 infer_schema 函数，验证函数 c 的推断类型为 (Tensor x, Tensor[] y, Tensor? z, Tensor?[] w) -> Tensor[]
        self.assertExpectedInline(
            infer_schema(c),
            """(Tensor x, Tensor[] y, Tensor? z, Tensor?[] w) -> Tensor[]""",
        )

        # 定义函数 d，接受一个 Tensor 类型参数 x，返回一个包含 Tensor 列表和一个 Tensor 的元组
        def d(x: Tensor) -> Tuple[List[Tensor], Tensor]:
            return [torch.empty([])], torch.empty([])

        # 调用 infer_schema 函数，验证函数 d 的推断类型为 (Tensor x) -> (Tensor[], Tensor)
        self.assertExpectedInline(
            infer_schema(d), """(Tensor x) -> (Tensor[], Tensor)"""
        )

        # 定义函数 e，不接受任何参数，返回一个空的 Tensor
        def e() -> Tensor:
            return torch.empty([])

        # 调用 infer_schema 函数，验证函数 e 的推断类型为 () -> Tensor
        self.assertExpectedInline(infer_schema(e), """() -> Tensor""")

        # 定义函数 f，接受一个 Tensor 类型参数 x，没有返回值
        def f(x: Tensor) -> None:
            pass

        # 调用 infer_schema 函数，验证函数 f 的推断类型为 (Tensor x) -> ()
        self.assertExpectedInline(infer_schema(f), """(Tensor x) -> ()""")

        # 定义函数 g，接受多个参数：x (Tensor 类型)，y (Tensor 列表)，z (Tensor 列表)，w (可选的 Tensor 列表)
        # 没有返回值
        def g(
            x: Tensor, y: List[Tensor], z: List[Tensor], w: List[Optional[Tensor]]
        ) -> None:
            pass

        # 调用 infer_schema 函数，验证函数 g 的推断类型为 (Tensor x, Tensor[] y, Tensor[] z, Tensor?[] w) -> ()
        self.assertExpectedInline(
            infer_schema(g), """(Tensor x, Tensor[] y, Tensor[] z, Tensor?[] w) -> ()"""
        )

        # 调用 infer_schema 函数，验证函数 g 在特定参数上的推断类型为 (Tensor(a0!) x, Tensor(a2!)[] z, Tensor(a3!)?[] w) -> ()
        self.assertExpectedInline(
            infer_schema(g, mutates_args={"x", "w", "z"}),
            """(Tensor(a0!) x, Tensor(a2!)[] z, Tensor(a3!)?[] w) -> ()""",
        )
    # 测试函数：测试当函数参数不支持的情况下，是否能正确抛出异常
    def test_infer_schema_unsupported(self):
        # 测试当函数使用可变位置参数时，抛出值错误异常
        with self.assertRaisesRegex(ValueError, "varargs"):
            # 定义一个未实现的函数 foo，接受任意数量的位置参数
            def foo(*args):
                raise NotImplementedError

            # 调用 infer_schema 函数，期望抛出值错误异常
            infer_schema(foo)

        # 测试当函数使用可变关键字参数时，抛出值错误异常
        with self.assertRaisesRegex(ValueError, "varkwargs"):
            # 定义一个未实现的函数 foo，接受任意数量的关键字参数
            def foo(**kwargs):
                raise NotImplementedError

            # 调用 infer_schema 函数，期望抛出值错误异常
            infer_schema(foo)

        # 测试当函数参数缺少类型注解时，抛出值错误异常
        with self.assertRaisesRegex(ValueError, "must have a type annotation"):
            # 定义一个未实现的函数 foo，只接受一个参数 x，但没有类型注解
            def foo(x):
                raise NotImplementedError

            # 调用 infer_schema 函数，期望抛出值错误异常
            infer_schema(foo)

        # 测试当函数参数类型不支持时，抛出值错误异常
        with self.assertRaisesRegex(ValueError, "unsupported"):
            # 定义一个未实现的函数 foo，接受一个参数 x，返回值类型为 Tensor 元组
            def foo(x: Tensor) -> Tuple[Tensor, ...]:
                raise NotImplementedError

            # 调用 infer_schema 函数，期望抛出值错误异常
            infer_schema(foo)

        # 测试当函数参数声明为不可变时，但传递的参数可以被修改时，抛出值错误异常
        with self.assertRaisesRegex(ValueError, "can be mutated"):
            # 定义一个未实现的函数 foo，接受一个 Tensor 类型的参数 x 和一个整数参数 y，返回值为 Tensor
            def foo(x: Tensor, y: int) -> Tensor:
                raise NotImplementedError

            # 调用 infer_schema 函数，传递 mutates_args={"y"} 表示参数 y 可以被修改，期望抛出值错误异常
            infer_schema(foo, mutates_args={"y"})

    # 辅助函数：根据输入的类型生成示例数据
    def _generate_examples(self, typ):
        # 如果 typ 是 int 类型，则返回一个整数示例列表 [17]
        if typ is int:
            return [17]
        # 如果 typ 是 float 类型，则返回一个浮点数示例列表 [3.14]
        if typ is float:
            return [3.14]
        # 如果 typ 是 bool 类型，则返回一个布尔值示例列表 [True]
        if typ is bool:
            return [True]
        # 如果 typ 是 str 类型，则返回一个字符串示例列表 ["foo"]
        if typ is str:
            return ["foo"]
        # 如果 typ 是 torch.dtype 类型，则返回一个 torch.dtype 示例列表 [torch.float32]
        if typ is torch.dtype:
            return [torch.float32]
        # 如果 typ 是 torch.device 类型，则返回一个 torch.device 示例列表 [torch.device("cpu")]
        if typ is torch.device:
            return [torch.device("cpu")]
        # 如果 typ 是 torch.types.Number 类型，则返回一个数字示例列表 [2.718]
        if typ == torch.types.Number:
            return [2.718]
        # 如果 typ 是 torch.Tensor 类型，则返回一个 torch.Tensor 示例列表 [torch.tensor(3)]
        if typ is torch.Tensor:
            return [torch.tensor(3)]
        # 如果 typ 是 Optional[torch.types.Number] 类型，则返回一个包含 None 和数字示例的列表 [None, 2.718]
        if typ == Optional[torch.types.Number]:
            return [None, 2.718]
        # 如果 typ 是 Union 类型，则递归生成其中元素类型的示例，并添加 None 到列表中
        origin = typing.get_origin(typ)
        if origin is Union:
            args = typing.get_args(typ)
            assert len(args) == 2 and (args[0] is type(None) or args[1] is type(None))
            elt = args[0] if args[1] is type(None) else args[1]
            return self._generate_examples(elt) + [None]
        # 如果 typ 是 list 类型，则递归生成列表中元素类型的示例，并返回一个包含三个列表示例的列表
        if origin is list:
            args = typing.get_args(typ)
            assert len(args) == 1
            elt = args[0]
            return [
                self._generate_examples(elt),
                self._generate_examples(elt),
                self._generate_examples(elt),
            ]
        # 如果 typ 是 collections.abc.Sequence 类型，则递归生成序列中元素类型的示例，并返回其笛卡尔积
        if origin is collections.abc.Sequence:
            args = typing.get_args(typ)
            assert len(args) == 1
            examples = self._generate_examples(args[0])
            return list(itertools.product(examples, examples)) + []
        # 如果没有匹配到以上任何类型，则抛出未实现错误，提示无法生成指定类型的实例
        raise NotImplementedError(
            f"testrunner cannot generate instanstance of type {typ}"
        )
    # 测试支持的单返回类型
    def test_supported_return_types_single_return(self):
        # 遍历所有支持的返回类型
        for typ in torch._library.infer_schema.SUPPORTED_RETURN_TYPES:
            # 对于每种返回类型生成示例
            for example in self._generate_examples(typ):
                try:
                    # 定义自定义操作函数 foo，声明其返回类型为 typ
                    @custom_ops.custom_op(f"{self.test_ns}::foo")
                    def foo(x: Tensor) -> typ:
                        raise NotImplementedError

                    # 实现 foo 函数的具体操作，返回示例数据
                    @custom_ops.impl(f"{self.test_ns}::foo")
                    def foo_impl(x: Tensor) -> typ:
                        return example

                    # 获取操作对象 op
                    op = self.get_op(f"{self.test_ns}::foo")
                    # 执行操作并获取结果
                    result = op(torch.randn([]))
                    # 断言操作结果与示例数据相等
                    self.assertEqual(result, example, msg=f"{typ} {example}")
                finally:
                    # 清理自定义操作
                    custom_ops._destroy(f"{self.test_ns}::foo")

    # 测试支持的多返回类型
    def test_supported_return_types_multi_return(self):
        # 遍历所有支持的返回类型
        for typ in torch._library.infer_schema.SUPPORTED_RETURN_TYPES:
            # 对于每种返回类型生成示例
            for example in self._generate_examples(typ):
                try:
                    # 定义自定义操作函数 foo，声明其返回类型为 Tuple[typ, typ]
                    @custom_ops.custom_op(f"{self.test_ns}::foo")
                    def foo(x: Tensor) -> Tuple[typ, typ]:
                        raise NotImplementedError

                    # 实现 foo 函数的具体操作，返回示例数据的元组
                    @custom_ops.impl(f"{self.test_ns}::foo")
                    def foo_impl(x: Tensor) -> Tuple[typ, typ]:
                        return (example, example)

                    # 获取操作对象 op
                    op = self.get_op(f"{self.test_ns}::foo")
                    # 执行操作并获取结果
                    result = op(torch.randn([]))
                    expected = (example, example)
                    # 断言操作结果与期望的元组相等
                    self.assertEqual(result, expected, msg=f"{typ} {example}")
                finally:
                    # 清理自定义操作
                    custom_ops._destroy(f"{self.test_ns}::foo")

    # 测试支持的参数类型
    def test_supported_param_types(self):
        # 遍历所有支持的参数类型
        for typ in torch._library.infer_schema.SUPPORTED_PARAM_TYPES:

            # 定义自定义操作函数 foo，声明其参数类型为 (Tensor, typ) -> Tensor
            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: typ) -> Tensor:
                raise NotImplementedError

            yeet = None

            # 实现 foo 函数的具体操作，只在 CPU 设备上可用，记录参数 y 的值，并返回 x 的克隆
            @custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types=["cpu"])
            def foo_cpu(x, y):
                nonlocal yeet
                yeet = y
                return x.clone()

            try:
                # 对于每种参数类型生成示例
                for example in self._generate_examples(typ):
                    # 获取操作对象 op
                    op = self.get_op(f"{self.test_ns}::foo")
                    # 执行操作并传入示例参数
                    op(torch.randn([]), example)
                    # 断言函数 foo 在 CPU 实现中记录的参数 y 与示例参数相等
                    self.assertEqual(yeet, example, msg=f"{typ} {example}")
                    yeet = None
            finally:
                # 清理自定义操作
                custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
    def test_sequences(self):
        # Sequence[int] gets automagically turned into int[] in the schema.
        # This test checks that we actually do support arbitrary sequence types.
        
        # 定义一个自定义的序列类 MySequence，继承自 collections.abc.Sequence
        class MySequence(collections.abc.Sequence):
            def __init__(self):
                self._container = [1, 2, 3]

            def __getitem__(self, idx):
                return self._container[idx]

            def __len__(self):
                return len(self._container)

        # 自定义操作装饰器 custom_ops.custom_op，注册一个名为 f"{self.test_ns}::foo" 的自定义操作
        @custom_ops.custom_op(f"{self.test_ns}::foo")
        def foo(x: torch.Tensor, sizes: Sequence[int]) -> torch.Tensor:
            raise NotImplementedError

        called = 0

        # 实现 foo 操作的具体函数 foo_cpu，用于 CPU 设备，对 sizes 进行规范化处理
        @custom_ops.impl(f"{self.test_ns}::foo", device_types="cpu")
        def foo_cpu(x, sizes):
            nonlocal called
            called += 1
            # Dispatcher 将 sizes 的类型规范化为 List
            self.assertEqual(sizes, [1, 2, 3])
            return x.clone()

        # 生成一个随机张量 x
        x = torch.randn([])
        # 创建 MySequence 的实例 seq
        seq = MySequence()
        # 获取名为 f"{self.test_ns}::foo" 的操作
        op = self.get_op(f"{self.test_ns}::foo")
        # 调用操作 op，传入 x 和 seq 作为参数
        op(x, seq)
        # 断言 called 变量的值为 1
        self.assertEqual(called, 1)

    def test_unsupported_param_types(self):
        # Not comprehensive (it doesn't need to be), just a check that our mechanism works
        # 使用 assertRaisesRegex 检测是否抛出特定异常，并包含特定字符串 "unsupported type"
        
        with self.assertRaisesRegex(ValueError, "unsupported type"):
            # 定义一个不支持的参数类型的自定义操作 foo
            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: List[Optional[int]]) -> Tensor:
                raise NotImplementedError

            del foo

        with self.assertRaisesRegex(ValueError, "unsupported type"):
            # 同上，定义一个不支持的参数类型的自定义操作 foo
            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: Tuple[int, int]) -> Tensor:
                raise NotImplementedError

            del foo

        with self.assertRaisesRegex(ValueError, r"For example, typing.List\[int\]"):
            # 检查是否提出了正确且支持的类型建议
            @torch.library.custom_op(f"{TestCustomOp.test_ns}::foo", mutates_args={})
            def foo(x: Tensor, y: Tuple[int, int]) -> Tensor:
                raise NotImplementedError

            del foo

        with self.assertRaises(ValueError) as cm:
            # 检查是否抛出了特定的异常类型
            @torch.library.custom_op(f"{TestCustomOp.test_ns}::foo", mutates_args={})
            def foo(x: Tensor, y: Tuple[int, float]) -> Tensor:
                raise NotImplementedError

            del foo

            # 确保异常信息中不包含 "example"
            self.assertNotIn("example", str(cm.exception), "")

        with self.assertRaisesRegex(ValueError, "unsupported type"):
            # 定义一个不支持的参数类型的自定义操作 foo
            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: Callable) -> Tensor:
                raise NotImplementedError

            del foo
    def test_supported_schemas(self):
        # 所有这些模式应该已经被 PyTorch 代码生成工具测试过
        # （我们共享相同的机制），但这里是一个健全性检查。
        schemas = [
            "(Tensor x) -> Tensor",
            "(Tensor x) -> Tensor y",
            "(Tensor[] x) -> Tensor y",
            "(Tensor x) -> (Tensor, Tensor)",
            "(Tensor x) -> (Tensor y, Tensor z)",
            "(Tensor x) -> (Tensor y, Tensor z)",
        ]
        other_schemas = [
            "(Tensor x, Tensor w) -> (Tensor y, Tensor z)",
            "(Tensor x, Tensor w) -> (Tensor, Tensor)",
            "(Tensor x, Tensor w) -> Tensor",
            "(Tensor? x, Tensor w) -> Tensor",
            "(Tensor? x, Tensor[] w) -> Tensor",
            "(Tensor x, int[] w) -> Tensor",
            "(Tensor x, SymInt[] w) -> Tensor",
            "(Tensor x, Scalar w) -> Tensor",
            "(Tensor x, float w) -> Tensor",
            "(Tensor x, float? w) -> Tensor",
            "(Tensor x, bool[] w) -> Tensor",
        ]

        for schema in schemas:
            # 调用自定义操作的方法，使用给定的模式注册操作
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", schema)
            # 销毁已注册的操作
            custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
        for schema in other_schemas:
            # 调用自定义操作的方法，使用给定的模式注册操作
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::bar", schema)
            # 销毁已注册的操作
            custom_ops._destroy(f"{TestCustomOp.test_ns}::bar")

    def test_reserved_ns(self):
        # 导入预定义的命名空间列表
        from torch._custom_op.impl import RESERVED_NS

        for ns in RESERVED_NS:
            # 使用预定义命名空间创建自定义操作，预期抛出值错误异常
            with self.assertRaisesRegex(ValueError, "is a reserved namespace"):
                custom_ops.custom_op(f"{ns}::foo", "(Tensor x) -> Tensor")

            # 使用预定义命名空间创建自定义操作函数，预期抛出值错误异常
            with self.assertRaisesRegex(ValueError, "is a reserved namespace"):
                @custom_ops.custom_op(f"{ns}::foo2")
                def foo2(x: torch.Tensor) -> torch.Tensor:
                    raise NotImplementedError

    def test_private_ctor(self):
        # 使用私有构造函数创建 CustomOp 实例，预期抛出运行时错误异常
        with self.assertRaisesRegex(RuntimeError, "CustomOp constructor is private"):
            CustomOp(None, None, None, None, None)

    def test_lifetime(self):
        # 使用给定的模式注册自定义操作函数
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        # 获取已注册的自定义操作
        custom_op = torch._custom_op.impl.get_op(f"{TestCustomOp.test_ns}::foo")

        # 试图重复定义同一操作，预期抛出运行时错误异常
        with self.assertRaisesRegex(RuntimeError, "multiple times"):
            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
                raise NotImplementedError

        # 销毁已注册的自定义操作
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

        # 烟雾测试：重新注册已销毁的自定义操作
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            raise NotImplementedError

        # 销毁重新注册的自定义操作
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
    # 定义测试函数，用于测试未实现自动求导的情况
    def test_autograd_notimplemented(self):
        # 定义装饰器，将函数 foo 注册为自定义操作
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        # 定义函数 foo，接收一个 torch.Tensor 类型参数 x，返回一个 torch.Tensor
        def foo(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            # 抛出未实现错误，提示自动求导尚未实现
            raise NotImplementedError

        # 创建一个随机张量 x，要求其进行梯度跟踪
        x = torch.randn(3, requires_grad=True)
        # 获取名为 foo 的操作对象
        op = self.get_op(f"{self.test_ns}::foo")
        # 断言在运行 op(x) 时会抛出 RuntimeError，且错误信息包含 "Autograd has not been implemented"
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            op(x)
        # 销毁自定义操作 foo
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
        # 删除 foo 函数

        del foo

        # 定义另一个函数 foo，接收一个 Sequence[torch.Tensor] 类型参数 x，返回一个 torch.Tensor
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: Sequence[torch.Tensor]) -> torch.Tensor:
            # 抛出未实现错误，提示自动求导尚未实现
            raise NotImplementedError

        # 创建随机张量 x 和 y
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        # 获取名为 foo 的操作对象
        op = self.get_op(f"{self.test_ns}::foo")
        # 断言在运行 op([y, x]) 时会抛出 RuntimeError，且错误信息包含 "Autograd has not been implemented"
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            op([y, x])
        # 销毁自定义操作 foo
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
        # 删除 foo 函数

        del foo

        # 定义另一个函数 foo，接收两个 torch.Tensor 类型参数 x 和 y，返回一个 torch.Tensor
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # 抛出未实现错误，提示自动求导尚未实现
            raise NotImplementedError

        # 创建随机张量 x 和 y
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        # 获取名为 foo 的操作对象
        op = self.get_op(f"{self.test_ns}::foo")
        # 断言在运行 op(y, x) 时会抛出 RuntimeError，且错误信息包含 "Autograd has not been implemented"
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            op(y, x)
        # 销毁自定义操作 foo
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

    # 定义测试函数，用于测试未实现自动求导的情况（处于 no_grad 模式）
    def test_autograd_notimplemented_gradmode(self):
        # 定义装饰器，将函数 foo 注册为自定义操作
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        # 定义函数 foo，接收两个 torch.Tensor 类型参数 x 和 y，返回一个 torch.Tensor
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # 抛出未实现错误，提示自动求导尚未实现
            raise NotImplementedError

        # 定义 foo 的实现函数，接收参数 x 和 y，并返回它们的乘积
        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, y):
            return x * y

        # 创建随机张量 x 和 y
        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        # 获取名为 foo 的操作对象
        op = self.get_op(f"{self.test_ns}::foo")
        # 在 no_grad 模式下运行 op(y, x)，不应该抛出异常，因为不会进行梯度跟踪
        with torch.no_grad():
            op(y, x)

    # 定义测试函数，用于测试在 CPU 上的实现情况
    def test_impl_cpu(self):
        # 定义装饰器，将函数 foo 注册为自定义操作
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        # 定义函数 foo，接收一个 torch.Tensor 类型参数 x，返回一个 torch.Tensor
        def foo(x: torch.Tensor) -> torch.Tensor:
            # 抛出未实现错误，提示自动求导尚未实现
            raise NotImplementedError

        # 定义在 CPU 上的 foo 实现函数，接收参数 x，并返回其正弦值
        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types="cpu")
        def foo_cpu(x):
            return x.sin()

        # 创建随机张量 x
        x = torch.randn(3)
        # 获取名为 foo 的操作对象
        op = self.get_op(f"{self.test_ns}::foo")
        # 运行 op(x)，得到结果
        result = op(x)
        # 断言运行结果与在 CPU 上的实现结果 foo_cpu(x) 相等
        self.assertEqual(result, foo_cpu(x))
    def test_impl_invalid_devices(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError
        定义一个名为 foo 的自定义操作，抛出未实现错误

        def foo_impl(x):
            return x.sin()
        定义一个函数 foo_impl，对输入张量 x 执行正弦函数

        from torch._custom_op.impl import SUPPORTED_DEVICE_TYPE_TO_KEY
        导入支持设备类型到键的映射

        for device_type in SUPPORTED_DEVICE_TYPE_TO_KEY.keys():
            # Smoke test: should not raise error
            custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types=device_type)(
                foo_impl
            )
            使用给定的 device_type 对 foo_impl 进行自定义操作的实现

        # Not supported by this API: we can either support them in the future
        # or provide some other CustomOp.def_* function. This depends on how
        # common the use cases are.
        针对不受此 API 支持的设备类型，可以在将来支持它们或提供其他 CustomOp.def_* 函数。这取决于使用案例的普遍程度。

        for invalid_type in ["hip", "xla", "mkldnn", ["cpu", "hip"]]:
            with self.assertRaisesRegex(ValueError, "we only support device_type"):
                custom_ops.impl(
                    f"{TestCustomOp.test_ns}::foo", device_types=invalid_type
                )(foo_impl)
            使用不受支持的设备类型 invalid_type 调用 custom_ops.impl，并期望抛出 ValueError 异常

    def test_backward_partially_registered(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError
        定义一个名为 foo 的自定义操作，抛出未实现错误

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.sin()
        定义一个函数 foo_impl，对输入张量 x 执行正弦函数

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return grad * saved.cos()
        定义一个名为 foo_backward 的反向实现函数，接受上下文 ctx、保存的值 saved 和梯度 grad，并返回梯度乘以 saved 的余弦值

        x = torch.randn([], requires_grad=True)
        创建一个具有梯度信息的随机张量 x

        op = self.get_op(f"{self.test_ns}::foo")
        获取操作符 op，其名称为 self.test_ns::foo

        with self.assertRaisesRegex(
            RuntimeError, "unable to find a 'save_for_backward'"
        ):
            y = op(x)
            y.backward()
        使用操作符 op 对 x 进行操作，并期望在执行反向传播时抛出 RuntimeError 异常，提示找不到 'save_for_backward'

    def test_save_for_backward_inputs_are_namedtuple(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError
        定义一个名为 foo 的自定义操作，抛出未实现错误

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.sin()
        定义一个函数 foo_impl，对输入张量 x 执行正弦函数

        hit = 0
        初始化 hit 计数器为 0

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            nonlocal hit
            hit += 1
            self.assertTrue(isinstance(inputs, tuple))
            self.assertEqual(list(inputs._asdict().keys()), ["x"])
            return inputs.x
        定义一个名为 foo_save_for_backward 的保存梯度函数，接受输入 inputs 和输出 output，增加 hit 计数器并返回 inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos()}
        定义一个名为 foo_backward 的反向实现函数，接受上下文 ctx、保存的值 saved 和梯度 grad，并返回一个字典，键为 'x'，值为 grad 乘以 saved 的余弦值

        x = torch.randn([], requires_grad=True)
        创建一个具有梯度信息的随机张量 x

        op = self.get_op(f"{self.test_ns}::foo")
        获取操作符 op，其名称为 self.test_ns::foo

        y = op(x)
        对 x 应用操作符 op，得到 y

        self.assertEqual(hit, 1)
        断言 hit 的值为 1

        y.backward()
        对 y 执行反向传播
        self.assertEqual(hit, 1)
        断言 hit 的值仍为 1
    def test_backward_returns_dict(self):
        # 定义自定义操作 'foo'，抛出未实现错误
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        # 实现 'foo' 的具体功能，返回输入张量的正弦值
        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.sin()

        # 保存 'foo' 的反向传播所需的输入
        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        # 实现 'foo' 的反向传播，计算梯度乘以保存的值的余弦值
        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return grad * saved.cos()

        # 创建随机输入张量 x，并标记为需要梯度
        x = torch.randn([], requires_grad=True)
        # 获取操作 'foo' 的实现
        op = self.get_op(f"{self.test_ns}::foo")
        # 执行操作 'foo'，得到输出 y
        y = op(x)
        # 断言在调用反向传播时抛出运行时错误，并期望错误信息包含 "to be a dict"
        with self.assertRaisesRegex(RuntimeError, "to be a dict"):
            y.backward()

    def test_backward_dict_invalid_keys(self):
        # 定义自定义操作 'foo'，抛出未实现错误
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        # 实现 'foo' 的具体功能，返回输入张量的正弦值
        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.sin()

        # 保存 'foo' 的反向传播所需的输入
        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        # 实现 'foo' 的反向传播，返回带有无效键 'y' 的字典
        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos(), "y": None}

        # 创建随机输入张量 x，并标记为需要梯度
        x = torch.randn([], requires_grad=True)
        # 获取操作 'foo' 的实现
        op = self.get_op(f"{self.test_ns}::foo")
        # 执行操作 'foo'，得到输出 y
        y = op(x)
        # 断言在调用反向传播时抛出运行时错误，并期望错误信息包含 "to have keys {'x'}"
        with self.assertRaisesRegex(RuntimeError, "to have keys {'x'}"):
            y.backward()

    def test_backward_dict_grad_for_nontensor(self):
        # 定义自定义操作 'foo'，抛出未实现错误
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            raise NotImplementedError

        # 实现 'foo' 的具体功能，返回输入张量的正弦值
        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, dim):
            return x.sin()

        # 保存 'foo' 的反向传播所需的输入
        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        # 实现 'foo' 的反向传播，返回带有非张量类型 'dim' 的字典
        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos(), "dim": None}

        # 创建随机输入张量 x，并标记为需要梯度，以及整数类型 dim
        x = torch.randn([], requires_grad=True)
        # 获取操作 'foo' 的实现
        op = self.get_op(f"{self.test_ns}::foo")
        # 执行操作 'foo'，得到输出 y
        y = op(x, 32)
        # 断言在调用反向传播时抛出运行时错误，并期望错误信息包含 "non-Tensor-like types"
        with self.assertRaisesRegex(RuntimeError, "non-Tensor-like types"):
            y.backward()
    # 测试函数：验证反向传播中输入张量必须有指定的键
    def test_backward_dict_requires_keys_for_input_tensors(self):
        # 自定义操作 'foo' 的装饰器，声明输入参数为两个 torch 张量 x 和 y，返回一个 torch 张量
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        # 实现 'foo' 操作的函数，对输入 x 进行正弦运算并返回结果
        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, y):
            return x.sin()

        # 保存反向传播所需的信息，选择保存输入张量 x
        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        # 'foo' 操作的反向传播函数，接收上下文 ctx、保存的张量 saved 和梯度 grad，返回梯度字典
        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos()}

        # 生成一个随机张量 x，设置其需要梯度计算
        x = torch.randn([], requires_grad=True)
        # 获取 'foo' 操作对象
        op = self.get_op(f"{self.test_ns}::foo")
        # 执行 'foo' 操作，并用 y 接收结果
        y = op(x, x)
        # 断言捕获 RuntimeError，并验证错误消息中包含指定的键 'y'
        with self.assertRaisesRegex(RuntimeError, r"to have keys {.*'y'.*}"):
            # 对 y 执行反向传播
            y.backward()

    # 测试函数：验证反向传播中可选输入张量必须有指定的键
    def test_backward_dict_requires_keys_for_input_optional_tensors(self):
        # 自定义操作 'foo' 的装饰器，声明输入参数为一个 torch 张量 x 和一个可选的 torch 张量 y，返回一个 torch 张量
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
            raise NotImplementedError

        # 实现 'foo' 操作的函数，对输入 x 进行正弦运算并返回结果
        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, y):
            return x.sin()

        # 保存反向传播所需的信息，选择保存输入张量 x
        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        # 'foo' 操作的反向传播函数，接收上下文 ctx、保存的张量 saved 和梯度 grad，返回梯度字典
        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos()}

        # 生成一个随机张量 x，设置其需要梯度计算
        x = torch.randn([], requires_grad=True)
        # 获取 'foo' 操作对象
        op = self.get_op(f"{self.test_ns}::foo")
        # 执行 'foo' 操作，此时传入的 y 为 None
        y = op(x, None)
        # 断言捕获 RuntimeError，并验证错误消息中包含指定的键 'y'
        with self.assertRaisesRegex(RuntimeError, r"to have keys {.*'y'.*}"):
            # 对 y 执行反向传播
            y.backward()

    # 测试函数：验证反向传播中返回的梯度必须是张量或 None
    def test_backward_grads_are_tensor_or_none(self):
        # 自定义操作 'foo' 的装饰器，声明输入参数为一个 torch 张量 x，返回一个 torch 张量
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        # 实现 'foo' 操作的函数，对输入 x 进行正弦运算并返回结果
        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.sin()

        # 保存反向传播所需的信息，选择保存输入张量 x
        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        # 'foo' 操作的反向传播函数，接收上下文 ctx、保存的张量 saved 和梯度 grad，返回梯度字典
        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            # 返回梯度字典，其中 'x' 键对应的值为 grad * saved.cos()
            return {"x": grad * saved.cos()}

        # 生成一个随机张量 x，设置其需要梯度计算
        x = torch.randn([], requires_grad=True)
        # 获取 'foo' 操作对象
        op = self.get_op(f"{self.test_ns}::foo")
        # 执行 'foo' 操作，并用 y 接收结果
        y = op(x)
        # 断言捕获 RuntimeError，并验证错误消息中包含指定的文本 "either None or a Tensor"
        with self.assertRaisesRegex(RuntimeError, "either None or a Tensor"):
            # 对 y 执行反向传播
            y.backward()
    def test_backward_tensorlist_input_requires_list_grads_with_same_numel(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            raise NotImplementedError
        # 定义自定义操作 'foo'，接受一个 Torch 张量序列，返回一个 Torch 张量，抛出未实现错误

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(xs):
            return xs[0].sin()
        # 实现 'foo' 操作的具体逻辑，返回第一个张量的正弦值

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.xs[0]
        # 保存 'foo' 操作的输入，用于反向传播时的梯度计算

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"xs": [grad * saved.cos(), None]}
        # 实现 'foo' 操作的反向传播，计算并返回输入张量列表的梯度字典

        xs = [torch.randn([], requires_grad=True) for _ in range(3)]
        # 创建一个包含三个随机张量的列表，每个张量需要梯度信息
        op = self.get_op(f"{self.test_ns}::foo")
        # 获取自定义操作 'foo' 的操作对象
        y = op(xs)
        # 执行操作 'foo' 并传入张量列表 xs，得到输出张量 y
        with self.assertRaisesRegex(RuntimeError, "3 gradients but got 2"):
            y.backward()
        # 使用反向传播计算梯度，并断言抛出运行时错误，指示期望 3 个梯度但实际得到 2 个

    def test_backward_tensorlist_input_requires_list_grads_none_or_Tensor(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            raise NotImplementedError
        # 同上，定义 'foo' 操作，抛出未实现错误

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(xs):
            return xs[0].sin()
        # 同上，实现 'foo' 操作的具体逻辑

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.xs[0]
        # 同上，保存 'foo' 操作的输入

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"xs": [grad * saved.cos(), None, (None,)]}
        # 实现 'foo' 操作的反向传播，返回包含 None 或张量的元组

        xs = [torch.randn([], requires_grad=True) for _ in range(3)]
        # 创建一个包含三个随机张量的列表，每个张量需要梯度信息
        op = self.get_op(f"{self.test_ns}::foo")
        # 获取自定义操作 'foo' 的操作对象
        y = op(xs)
        # 执行操作 'foo' 并传入张量列表 xs，得到输出张量 y
        with self.assertRaisesRegex(RuntimeError, "None or Tensor"):
            y.backward()
        # 使用反向传播计算梯度，并断言抛出运行时错误，指示梯度应为 None 或张量

    def test_backward_tensorlist_input_requires_list_grads(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            raise NotImplementedError
        # 同上，定义 'foo' 操作，抛出未实现错误

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(xs):
            return xs[0].sin()
        # 同上，实现 'foo' 操作的具体逻辑

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.xs[0]
        # 同上，保存 'foo' 操作的输入

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"xs": None}
        # 实现 'foo' 操作的反向传播，返回 None 表示没有梯度信息

        xs = [torch.randn([], requires_grad=True) for _ in range(3)]
        # 创建一个包含三个随机张量的列表，每个张量需要梯度信息
        op = self.get_op(f"{self.test_ns}::foo")
        # 获取自定义操作 'foo' 的操作对象
        y = op(xs)
        # 执行操作 'foo' 并传入张量列表 xs，得到输出张量 y
        with self.assertRaisesRegex(RuntimeError, "list of gradients"):
            y.backward()
        # 使用反向传播计算梯度，并断言抛出运行时错误，指示期望返回梯度列表
    def test_backward_output_differentiability_type(self):
        # 定义一个自定义操作 'foo'，其输出不可微
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            raise NotImplementedError

        # 断言运行时错误中包含 'output_differentiability' 字符串
        with self.assertRaisesRegex(RuntimeError, "output_differentiability"):
            # 实现 'foo' 的反向传播，声明其输出可微性为真
            @custom_ops.impl_backward(
                f"{TestCustomOp.test_ns}::foo", output_differentiability=True
            )
            def foo_backward(ctx, saved, grad):
                return {"xs": None}

    def test_backward_output_differentiability_numel(self):
        # 定义一个自定义操作 'foo'，其输出不可微
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(xs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
            raise NotImplementedError

        # 断言运行时错误中包含 'output_differentiability' 字符串
        with self.assertRaisesRegex(RuntimeError, "output_differentiability"):
            # 实现 'foo' 的反向传播，声明其输出可微性为真
            @custom_ops.impl_backward(
                f"{TestCustomOp.test_ns}::foo", output_differentiability=[True]
            )
            def foo_backward(ctx, saved, grad):
                return {"xs": None}

    def test_backward_output_differentiability_tensorlist(self):
        # 定义一个自定义操作 'foo'，其输入为张量 'x'
        @custom_ops.custom_op(f"{self.test_ns}::foo")
        def foo(x: Tensor) -> Tuple[List[Tensor], Tensor]:
            raise NotImplementedError

        # 实现 'foo' 的具体实现函数
        @custom_ops.impl(f"{self.test_ns}::foo")
        def foo_impl(x):
            return [x.clone(), x.clone()], x.clone()

        # 实现 'foo' 的保存梯度函数
        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return []

        # 实现 'foo' 的反向传播，声明其输出可微性为 [False, True]
        @custom_ops.impl_backward(
            f"{TestCustomOp.test_ns}::foo", output_differentiability=[False, True]
        )
        def foo_backward(ctx, saved, grad_lst, grad):
            return {"x": grad}

        # 获取操作 'foo' 的对象
        op = self.get_op(f"{self.test_ns}::foo")
        # 创建一个随机张量 'x'，要求其梯度跟踪
        x = torch.randn(3, requires_grad=True)
        # 执行操作 'foo'，获取输出 [a, b], c
        [a, b], c = op(x)
        # 断言 a 和 b 不要求梯度跟踪，而 c 要求梯度跟踪
        self.assertFalse(a.requires_grad)
        self.assertFalse(b.requires_grad)
        self.assertTrue(c.requires_grad)

    def test_backward_output_differentiability_non_tensor(self):
        # 定义一个自定义操作 'foo'，其输入为张量 'x'
        @custom_ops.custom_op(f"{self.test_ns}::foo")
        def foo(x: Tensor) -> Tuple[Tensor, int]:
            raise NotImplementedError

        # 实现 'foo' 的具体实现函数
        @custom_ops.impl(f"{self.test_ns}::foo")
        def foo_impl(x):
            return x.clone(), 3

        # 实现 'foo' 的保存梯度函数
        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return []

        # 实现 'foo' 的反向传播，声明其输出可微性为 [True, True]
        @custom_ops.impl_backward(
            f"{TestCustomOp.test_ns}::foo", output_differentiability=[True, True]
        )
        def foo_backward(ctx, saved, grad0, grad1):
            return {"x": grad0}

        # 获取操作 'foo' 的对象
        op = self.get_op(f"{self.test_ns}::foo")
        # 创建一个随机张量 'x'，要求其梯度跟踪
        x = torch.randn(3, requires_grad=True)
        # 断言运行时错误中包含 'is not a Tensor' 字符串
        with self.assertRaisesRegex(RuntimeError, "is not a Tensor"):
            op(x)

    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    # 定义测试方法：测试单独实现
    def test_impl_separate(self):
        # 定义名为 `foo` 的自定义操作，接受 torch.Tensor 类型参数并返回 torch.Tensor 类型
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # 未实现的占位符
            raise NotImplementedError

        # 定义在 CPU 设备上执行 `foo` 操作的具体实现
        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types="cpu")
        def foo_cpu(x):
            return x.sin()

        # 定义在 CUDA 设备上执行 `foo` 操作的具体实现
        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types="cuda")
        def foo_cuda(x):
            return x.cos()

        # 生成一个形状为 (3,) 的随机输入张量 x
        x = torch.randn(3)
        # 获取操作 `foo` 的实现
        op = self.get_op(f"{self.test_ns}::foo")
        # 执行在 CPU 上的 `foo` 操作，并得到结果
        result = op(x)
        # 断言操作的结果与在 CPU 上定义的 `foo_cpu` 函数的结果相等
        self.assertEqual(result, foo_cpu(x))

        # 将张量 x 移到 CUDA 设备上
        x_cuda = x.cuda()
        # 再次获取操作 `foo` 的实现
        op = self.get_op(f"{self.test_ns}::foo")
        # 执行在 CUDA 上的 `foo` 操作，并得到结果
        result = op(x_cuda)
        # 断言操作的结果与在 CUDA 上定义的 `foo_cuda` 函数的结果相等
        self.assertEqual(result, foo_cuda(x_cuda))

    # 如果没有 CUDA 支持，则跳过该测试
    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    # 定义测试方法：测试多个实现
    def test_impl_multiple(self):
        # 定义名为 `foo` 的自定义操作，接受 torch.Tensor 类型参数并返回 torch.Tensor 类型
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # 未实现的占位符
            raise NotImplementedError

        # 定义默认实现 `foo_impl`，在任何设备上都执行 x 的余弦运算
        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.cos()

        # 获取操作 `foo` 的实现
        op = self.get_op(f"{self.test_ns}::foo")
        # 生成一个形状为 (3,) 的随机输入张量 x
        x = torch.randn(3)
        # 执行操作 `foo`，得到结果
        result = op(x)
        # 断言操作的结果与默认实现 `foo_impl` 函数的结果相等
        self.assertEqual(result, foo_impl(x))

        # 将张量 x 移到 CUDA 设备上
        x_cuda = x.cuda()
        # 再次执行操作 `foo`，得到结果
        result = op(x_cuda)
        # 断言操作的结果与默认实现 `foo_impl` 函数在 CUDA 上的结果相等
        self.assertEqual(result, foo_impl(x_cuda))

    # 定义测试方法：测试抽象重载实现
    def test_impl_abstract_overload(self):
        # 获取库实例
        lib = self.lib()
        # 定义一个具有指定签名的新函数
        lib.define("sin.blah(Tensor x) -> Tensor")

        # 将 torch.empty_like 作为抽象实现注册到 `sin.blah`，使用给定的库
        torch.library.impl_abstract(
            f"{self.test_ns}::sin.blah", torch.empty_like, lib=lib
        )

        # 获取操作 `sin.blah` 的实现
        op = self.ns().sin.blah
        # 生成一个形状为 (3,) 的随机输入张量 x，指定设备为 "meta"
        x = torch.randn(3, device="meta")
        # 执行操作 `sin.blah`，没有返回值的期望
        op(x)

    # 定义测试方法：测试元操作实现
    def test_impl_meta(self):
        # 定义名为 `foo` 的自定义操作，接受 torch.Tensor 类型和 int 类型参数，并返回 torch.Tensor 类型
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            # 未实现的占位符
            raise NotImplementedError

        # 将 torch.new_empty 作为抽象实现注册到 `foo`，使用给定的库
        @torch.library.impl_abstract(f"{TestCustomOp.test_ns}::foo", lib=self.lib())
        def foo_meta(x, dim):
            # 创建一个新的空张量，形状与 x 相同，但删除指定维度 dim
            output_shape = list(x.shape)
            del output_shape[dim]
            return x.new_empty(output_shape)

        # 生成一个形状为 (2, 3) 的随机输入张量 x，指定设备为 "meta"
        x = torch.randn(2, 3, device="meta")
        # 获取操作 `foo` 的实现
        op = self.get_op(f"{self.test_ns}::foo")
        # 执行操作 `foo`，并得到结果，删除第 1 维度后的张量
        result = op(x, 1)
        # 断言操作的结果形状与元操作 `foo_meta` 函数的结果形状相等
        self.assertEqual(result.shape, foo_meta(x, 1).shape)
    def test_duplicate_impl(self):
        # 定义一个名为 foo 的函数，使用 custom_op 装饰器自定义操作
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            # 未实现，抛出未实现错误
            raise NotImplementedError

        # 定义一个名为 foo_meta 的抽象实现函数，使用 torch.library.impl_abstract 函数注册
        @torch.library.impl_abstract(f"{TestCustomOp.test_ns}::foo", lib=self.lib())
        def foo_meta(x, dim):
            # 计算输出张量的形状，删除指定维度
            output_shape = list(x.shape)
            del output_shape[dim]
            return x.new_empty(output_shape)

        # 断言捕获到 RuntimeError，并且错误消息中包含当前文件名和行号
        with self.assertRaisesRegex(RuntimeError, r"test_custom_ops.py:\d+"):
            
            # 定义另一个名为 foo_meta2 的抽象实现函数，使用 torch.library.impl_abstract 函数注册
            @torch.library.impl_abstract(f"{TestCustomOp.test_ns}::foo", lib=self.lib())
            def foo_meta2(x, dim):
                # 计算输出张量的形状，删除指定维度
                output_shape = list(x.shape)
                del output_shape[dim]
                return x.new_empty(output_shape)

    def test_new_data_dependent_symint(self):
        # 定义一个名为 foo 的函数，使用 custom_op 装饰器自定义操作
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # 未实现，抛出未实现错误
            raise NotImplementedError

        # 定义一个名为 foo_meta 的抽象实现函数，使用 torch.library.impl_abstract 函数注册
        def foo_meta(x):
            # 获取当前上下文并创建一个动态大小的符号整数
            ctx = torch.library.get_ctx()
            r = ctx.new_dynamic_size(min=1)
            # 断言捕获到 ValueError，并且错误消息中包含 "greater than or equal to 0"
            with self.assertRaisesRegex(ValueError, "greater than or equal to 0"):
                ctx.new_dynamic_size(min=-1)
            # 断言捕获到 ValueError，并且错误消息中包含 "SymInt"
            with self.assertRaisesRegex(ValueError, "SymInt"):
                ctx.new_dynamic_size(max=x.numel())
            # 注意：必须返回动态大小的值！
            return x.new_empty(r)

        x = torch.randn(2, 3, device="cpu")
        # 获取名为 foo 的操作，生成符号化的函数图
        op = self.get_op(f"{self.test_ns}::foo")
        make_fx(op, tracing_mode="symbolic")(x)

    def test_meta_for_data_dependent_shape_operation(self):
        x = torch.randn(10, device="meta")
        # 断言捕获到 RuntimeError，并且错误消息中包含 "data-dependent output shape"
        with self.assertRaisesRegex(RuntimeError, "data-dependent output shape"):
            numpy_nonzero(x)

    def test_basic_make_fx(self):
        # 更严格的测试在我们的 CustomOp opinfo db 中进行，
        # 这个测试只是一个健全性检查。
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # 未实现，抛出未实现错误
            raise NotImplementedError

        # 定义一个名为 foo_meta 的抽象实现函数，使用 torch.library.impl_abstract 函数注册
        def foo_meta(x):
            # 返回输入张量的元素的和
            return x.sum()

        x = torch.randn(3)
        # 获取名为 foo 的操作，生成符号化的函数图
        op = self.get_op(f"{self.test_ns}::foo")
        gm = make_fx(op, tracing_mode="symbolic")(x)
        # 断言确保 "TestCustomOp.test_ns.foo" 存在于生成的代码中
        self.assertTrue(f"{TestCustomOp.test_ns}.foo" in gm.code)
    def test_not_implemented_error(self):
        # 定义一个名为 foo 的自定义操作，使用 custom_ops.custom_op 装饰器注册
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            # 抛出未实现错误
            raise NotImplementedError

        # 创建一个形状为 (3,) 的随机张量 x
        x = torch.randn(3)
        # 获取名称为 "{self.test_ns}::foo" 的操作对象
        op = self.get_op(f"{self.test_ns}::foo")
        # 断言调用 op(x) 会抛出 NotImplementedError，并且错误信息包含 "cpu impl registered"
        with self.assertRaisesRegex(NotImplementedError, "cpu impl registered"):
            op(x)

        # 创建一个设备为 "meta" 的形状为 (3,) 的随机张量 x
        x = torch.randn(3, device="meta")
        # 断言调用 op(x) 会抛出 NotImplementedError，并且错误信息包含 "no fake impl or Meta kernel"
        with self.assertRaisesRegex(NotImplementedError, "no fake impl or Meta kernel"):
            op(x)

        # 定义一个名为 bar 的自定义操作，使用 custom_ops.custom_op 装饰器注册
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::bar")
        def bar(sizes: Sequence[int]) -> torch.Tensor:
            # 抛出未实现错误
            raise NotImplementedError

        # 获取名称为 "{self.test_ns}::bar" 的操作对象
        op = self.get_op(f"{self.test_ns}::bar")
        # 断言调用 op((1, 2, 3)) 会抛出 NotImplementedError，并且错误信息包含 "no Tensor inputs"
        with self.assertRaisesRegex(NotImplementedError, "no Tensor inputs"):
            op((1, 2, 3))

    def test_data_dependent_basic(self):
        # 创建一个形状为 (5, 5) 的随机张量 x
        x = torch.randn(5, 5)
        # 对 numpy_nonzero 函数进行符号化追踪，返回追踪后的图模块 gm
        gm = make_fx(numpy_nonzero, tracing_mode="symbolic")(x)
        # 断言 gm.code 中包含字符串 "nonzero"
        self.assertTrue("nonzero" in gm.code)

    def test_data_dependent_fake_tracing(self):
        # 创建一个形状为 (5, 5) 的随机张量 x
        x = torch.randn(5, 5)
        # 使用 "fake" 追踪模式进行符号化追踪 numpy_nonzero 函数
        make_fx(numpy_nonzero, tracing_mode="fake")(x)

    def test_symints(self):
        # 定义一个函数 f，调用 torch.ops._torch_testing.numpy_view_copy 对 x 进行操作
        def f(x):
            return torch.ops._torch_testing.numpy_view_copy(x, x.shape)

        # 创建一个形状为 (2, 3, 4) 的随机张量 x
        x = torch.randn(2, 3, 4)
        # 对函数 f 进行符号化追踪，返回追踪后的图模块 gm
        gm = make_fx(f, tracing_mode="symbolic")(x)
        # 计算 gm(x) 的结果
        result = gm(x)
        # 断言 gm(x) 的结果等于 f(x) 的结果
        self.assertEqual(result, f(x))
        # 断言 gm.code.strip() 的值符合预期结果
        self.assertExpectedInline(
            gm.code.strip(),
            """\
# 定义一个类中的方法，处理输入张量 x_1，并进行前向计算
def forward(self, x_1):
    # 调用 torch.ops.aten.sym_size.int 函数获取张量 x_1 在维度 0、1、2 上的大小
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1)
    sym_size_int_2 = torch.ops.aten.sym_size.int(x_1, 2)
    # 调用 torch.ops._torch_testing.numpy_view_copy.default 函数，创建一个形状与 x_1 相同的张量 numpy_view_copy
    numpy_view_copy = torch.ops._torch_testing.numpy_view_copy.default(x_1, [sym_size_int, sym_size_int_1, sym_size_int_2]);  x_1 = sym_size_int = sym_size_int_1 = sym_size_int_2 = None
    # 返回处理后的张量 numpy_view_copy
    return numpy_view_copy

@unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work on windows")
def test_data_dependent_compile(self):
    import torch._dynamo.testing
    from torch._dynamo.utils import counters

    # 清空计数器 counters
    counters.clear()
    # 创建编译计数器 cnt
    cnt = torch._dynamo.testing.CompileCounter()

    @torch.compile(backend=cnt)
    def f(x):
        # 调用 numpy_nonzero 函数对输入张量 x 进行操作，并返回结果的克隆
        return numpy_nonzero(x.clone()).clone()

    # 调用函数 f，传入随机生成的张量，触发编译操作
    f(torch.randn(10))

    # 断言 graph_break 计数器中记录的图断点数量为 1
    self.assertEqual(len(counters["graph_break"]), 1)
    # 断言 graph_break 计数器中记录的第一个图断点的值为 1
    self.assertEqual(next(iter(counters["graph_break"].values())), 1)
    # 断言行内预期结果与实际值相符
    self.assertExpectedInline(
        # 获取 graph_break 计数器中的第一个键，并将其中的分号替换为换行符
        next(iter(counters["graph_break"].keys())).replace(";", "\n"),
        """\
dynamic shape operator: _torch_testing.numpy_nonzero.default
 to enable, set torch._dynamo.config.capture_dynamic_output_shape_ops = True""",
    )

# 预先存在的问题：torch.compile(dynamic=True) 默认会在数据相关操作上进行图断点。
# 最终我们会使其在数据相关操作上不进行图断点。
@unittest.expectedFailure
def test_data_dependent_nms_dynamic_compile(self):
    import torch._dynamo.testing
    from torch._dynamo.utils import counters

    # 清空计数器 counters
    counters.clear()
    # 创建编译计数器 cnt
    cnt = torch._dynamo.testing.CompileCounter()

    @torch.compile(backend=cnt, dynamic=True)
    def f(x, s, i):
        # 调用 torch.ops._torch_testing.numpy_nms 函数对输入张量 x 进行操作，并返回结果的克隆
        return torch.ops._torch_testing.numpy_nms(x.clone(), s, i).clone()

    # 调用函数 f，传入随机生成的张量和参数，触发编译操作
    f(torch.randn(20, 4), torch.randn(20), 0.1)

    # 断言 graph_break 计数器中记录的图断点数量为 0
    self.assertEqual(len(counters["graph_break"]), 0)

def test_impl_on_existing_op(self):
    # 获取当前类实例的库
    lib = self.lib()
    # 定义一个名为 foo 的操作，接受一个张量 x 并返回一个张量
    lib.define("foo(Tensor x) -> Tensor")
    # 构造完全限定名 qualname
    qualname = f"{self.test_ns}::foo"

    @torch._custom_ops.impl(qualname)
    def foo_impl(x):
        # 对输入张量 x 执行 sin 函数操作，并返回结果
        return x.sin()

    # 获取操作 op
    op = self.get_op(qualname)
    # 生成一个形状为 (3,) 的随机张量 x
    x = torch.randn(3)
    # 调用操作 op，传入张量 x，并断言其结果等于 x 的 sin 值
    result = op(x)
    self.assertEqual(result, x.sin())

@parametrize(
    # 参数化测试用例，测试不同的 key 值
    "key", ["CPU", "CUDA", "CompositeImplicitAutograd", "CompositeExplicitAutograd"]
)
def test_impl_on_existing_op_with_cpu_registration(self, key):
    # 获取当前类实例的库
    lib = self.lib()
    # 定义一个名为 foo 的操作，接受一个张量 x 并返回一个张量
    lib.define("foo(Tensor x) -> Tensor")
    # 构造完全限定名 qualname
    qualname = f"{self.test_ns}::foo"

    def foo_impl(x):
        # 对输入张量 x 执行 sin 函数操作，并返回结果
        return x.sin()

    # 将 foo_impl 函数注册到库 lib 中的 foo 操作上，并使用指定的 key
    lib.impl("foo", foo_impl, key)
    # 获取操作 op
    op = self.get_op(qualname)

    # 使用断言验证：尝试为已经有实现的操作 foo 注册新的实现，应抛出 RuntimeError 异常
    with self.assertRaisesRegex(RuntimeError, "already has an implementation"):
        custom_ops.impl(qualname, func=foo_impl)
    # 定义测试函数，用于验证在已存在操作上的抽象实现
    def test_abstract_impl_on_existing_op(self):
        # 创建库实例
        lib = self.lib()
        # 在库中定义一个函数签名
        lib.define("foo(Tensor x) -> Tensor")
        # 构建完整的函数限定名
        qualname = f"{self.test_ns}::foo"

        # 使用装饰器将抽象实现注册到 Torch 库中的指定函数上
        @torch.library.impl_abstract(qualname, lib=self.lib())
        def foo_impl(x):
            return x.sin()

        # 获取操作对象
        op = self.get_op(qualname)

        # 进入 Torch 的 FakeTensorMode 上下文
        with torch._subclasses.FakeTensorMode():
            # 创建一个形状为 (3,) 的随机张量
            x = torch.randn(3)
            # 执行操作
            result = op(x)
            # 验证操作结果的形状与输入张量 x 的形状相同
            self.assertEqual(result.shape, x.shape)
            # 验证操作结果的步长与输入张量 x 的步长相同
            self.assertEqual(result.stride(), x.stride())

    # 测试在已存在操作上的抽象实现，带有元数据
    def test_abstract_impl_on_existing_op_with_meta(self):
        # 创建库实例
        lib = self.lib()
        # 在库中定义一个函数签名
        lib.define("foo(Tensor x) -> Tensor")
        # 构建完整的函数限定名
        qualname = f"{self.test_ns}::foo"

        # 定义一个简单的实现函数
        def foo_impl(x):
            return x.sin()

        # 将实现函数注册到库中的指定函数上，带有元数据 "Meta"
        lib.impl("foo", foo_impl, "Meta")
        # 获取操作对象
        op = self.get_op(qualname)

        # 预期引发 RuntimeError，因为函数 foo 已经有了 "Meta" 实现
        with self.assertRaisesRegex(RuntimeError, r"already has .*Meta implementation"):
            torch.library.impl_abstract(qualname, func=foo_impl, lib=self.lib())

    # 测试在已存在操作上的抽象实现，带有 "CompositeImplicitAutograd" 实现
    def test_abstract_impl_on_existing_op_with_CompositeImplicitAutograd(self):
        # 创建库实例
        lib = self.lib()
        # 在库中定义一个函数签名
        lib.define("foo(Tensor x) -> Tensor")
        # 构建完整的函数限定名
        qualname = f"{self.test_ns}::foo"

        # 定义一个简单的实现函数
        def foo_impl(x):
            return x.sin()

        # 将实现函数注册到库中的指定函数上，带有 "CompositeImplicitAutograd" 元数据
        lib.impl("foo", foo_impl, "CompositeImplicitAutograd")
        # 获取操作对象
        op = self.get_op(qualname)

        # 预期引发 RuntimeError，因为函数 foo 已经有了 "CompositeImplicitAutograd" 实现
        with self.assertRaisesRegex(RuntimeError, "CompositeImplicitAutograd"):
            torch.library.impl_abstract(qualname, func=foo_impl, lib=self.lib())

    # 测试在已存在操作上的抽象实现，带有 "CompositeExplicitAutograd" 实现
    def test_abstract_impl_on_existing_op_with_CompositeExplicitAutograd(self):
        # 创建库实例
        lib = self.lib()
        # 在库中定义一个函数签名
        lib.define("foo(Tensor x) -> Tensor")
        # 构建完整的函数限定名
        qualname = f"{self.test_ns}::foo"

        # 定义一个简单的实现函数
        def foo_impl(x):
            return x.sin()

        # 将实现函数注册到库中的指定函数上，带有 "CompositeExplicitAutograd" 元数据
        lib.impl("foo", foo_impl, "CompositeExplicitAutograd")
        # 获取操作对象
        op = self.get_op(qualname)

        # 使用 lambda 函数注册抽象实现，对输入张量进行求和
        torch.library.impl_abstract(qualname, func=lambda x: x.sum(), lib=self.lib())

        # 进入 Torch 的 FakeTensorMode 上下文
        with torch._subclasses.FakeTensorMode():
            # 创建一个形状为 (10,) 的随机张量
            x = torch.randn(10)
            # 执行操作
            result = op(x)
            # 验证操作结果的形状为标量
            self.assertEqual(result.shape, ())

    # 辅助函数，测试反向实现是否引发指定的异常
    def _test_backward_impl_raises(self, qualname, err_regex):
        # 预期引发 RuntimeError，因为没有保存梯度的操作
        with self.assertRaisesRegex(RuntimeError, err_regex):
            # 使用装饰器定义一个保存梯度的实现函数
            @custom_ops.impl_save_for_backward(qualname)
            def foo2(x):
                return

        # 预期引发 RuntimeError，因为没有反向梯度的操作
        with self.assertRaisesRegex(RuntimeError, err_regex):
            # 使用装饰器定义一个反向梯度的实现函数
            @custom_ops.impl_backward(qualname)
            def foo3(x):
                return

    # 测试在已存在操作上的抽象反向实现，验证不正确的视图方案
    def test_backward_impl_on_existing_op_incorrect_schema_views(self):
        # 创建库实例
        lib = self.lib()
        # 在库中定义一个带有视图的函数签名
        lib.define("foo(Tensor(a) x) -> Tensor(a)")
        # 构建完整的函数限定名
        qualname = f"{self.test_ns}::foo"
        # 调用辅助函数，验证是否引发指定异常
        self._test_backward_impl_raises(qualname, "operator that returns views")
    # 定义测试方法：测试在现有操作上使用不正确的模式（可变参数类型`a!`）时是否引发异常
    def test_backward_impl_on_existing_op_incorrect_schema_mutable(self):
        # 创建库对象
        lib = self.lib()
        # 定义操作 `foo`，接受类型为 `a!` 的张量并返回张量
        lib.define("foo(Tensor(a!) x) -> Tensor")
        # 获取完全限定名称
        qualname = f"{self.test_ns}::foo"
        # 调用方法测试是否引发“non-functional”异常
        self._test_backward_impl_raises(qualname, "non-functional")

    # 定义测试方法：测试在现有操作上使用不正确的模式（无输出）时是否引发异常
    def test_backward_impl_on_existing_op_incorrect_schema_no_output(self):
        # 创建库对象
        lib = self.lib()
        # 定义操作 `foo`，接受类型为 `Tensor` 的参数但不返回任何值
        lib.define("foo(Tensor x) -> ()")
        # 获取完全限定名称
        qualname = f"{self.test_ns}::foo"
        # 调用方法测试是否引发“no returns”异常
        self._test_backward_impl_raises(qualname, "no returns")

    # 定义测试方法：测试在现有操作上使用 CompositeImplicitAutograd 模式时是否引发异常
    def test_backward_impl_on_existing_op_CompositeImplicitAutograd(self):
        # 创建库对象
        lib = self.lib()
        # 定义操作 `foo`，接受类型为 `Tensor` 的参数并返回 `Tensor`
        lib.define("foo(Tensor x) -> Tensor")
        # 获取完全限定名称
        qualname = f"{self.test_ns}::foo"
        # 实现操作 `foo` 使用 lambda 函数 x.sin().cos()，模式为 CompositeImplicitAutograd
        lib.impl("foo", lambda x: x.sin().cos(), "CompositeImplicitAutograd")
        # 调用方法测试是否引发“CompositeImplicitAutograd”异常
        self._test_backward_impl_raises(qualname, "CompositeImplicitAutograd")

    # 参数化测试方法：测试在现有操作上使用不同的 `key`（例如 "Autograd", "AutogradCPU", "AutogradCUDA"）时是否引发异常
    @parametrize("key", ["Autograd", "AutogradCPU", "AutogradCUDA"])
    def test_backward_impl_on_existing_op_with_key(self, key):
        # 创建库对象
        lib = self.lib()
        # 定义操作 `foo`，接受类型为 `Tensor` 的参数并返回 `Tensor`
        lib.define("foo(Tensor x) -> Tensor")
        # 获取完全限定名称
        qualname = f"{self.test_ns}::foo"
        # 实现操作 `foo` 使用 lambda 函数 x.sin().cos()，模式为给定的 `key`
        lib.impl("foo", lambda x: x.sin().cos(), key)
        # 调用方法测试是否引发给定 `key` 异常
        self._test_backward_impl_raises(qualname, key)

    # 定义测试方法：测试 `torch._library.utils.is_functional_schema` 方法的功能
    def test_is_functional_schema(self):
        # 测试用例字典，包含不同的函数模式字符串及预期结果
        tests = {
            "foo(Tensor x) -> Tensor": True,
            "foo(Tensor(a) x) -> Tensor": True,
            "foo(Tensor(a!) x) -> Tensor": False,
            "foo(Tensor(a) x) -> Tensor(a)": False,
            "foo(Tensor x) -> ()": False,
        }
        # 遍历测试用例
        for schema_str, expected in tests.items():
            # 调用 `torch._library.utils.is_functional_schema` 方法，验证其结果是否与预期一致
            res = torch._library.utils.is_functional_schema(schema_str)
            self.assertEqual(res, expected)

            # 导入 `FunctionSchema` 类
            from torchgen.model import FunctionSchema
            # 使用 `FunctionSchema.parse` 解析函数模式字符串
            schema = FunctionSchema.parse(schema_str)
            # 再次调用 `torch._library.utils.is_functional_schema` 方法，验证其结果是否与预期一致
            res = torch._library.utils.is_functional_schema(schema)
            self.assertEqual(res, expected)

            # 使用 `torch._C.parse_schema` 解析函数模式字符串
            schema = torch._C.parse_schema(schema_str)
            # 最后一次调用 `torch._library.utils.is_functional_schema` 方法，验证其结果是否与预期一致
            res = torch._library.utils.is_functional_schema(schema)
            self.assertEqual(res, expected)

    # 定义测试方法：测试在定义操作时使用不正确的模式类型时是否引发适当的异常
    def test_incorrect_schema_types(self):
        # 使用 `torch.library._scoped_library` 创建库对象，作用范围为 "mylib" 和 "FRAGMENT"
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            # 使用 `lib.define` 定义操作 `foo12`，接受类型为 `asdfasdf` 的参数并返回 `Tensor`
            with self.assertRaisesRegex(RuntimeError, "unknown type specifier"):
                lib.define("foo12(Tensor a) -> asdfasdf")
            # 使用 `lib.define` 定义操作 `foo12`，接受类型为 `Tensor` 的参数并返回类型为 `asdf` 的值
            with self.assertRaisesRegex(RuntimeError, "unknown type specifier"):
                lib.define("foo12(asdf a) -> Tensor")
            # 使用 `lib.define` 定义操作 `foo12`，接受类型为 `int64_t` 的参数并返回 `Tensor`
            with self.assertRaisesRegex(RuntimeError, "Use `SymInt` or `int`"):
                lib.define("foo12(int64_t a) -> Tensor")
            # 使用 `lib.define` 定义操作 `foo12`，接受类型为 `double` 的参数并返回 `Tensor`
            with self.assertRaisesRegex(RuntimeError, "Use `float`"):
                lib.define("foo12(double a) -> Tensor")
    def test_is_tensorlist_like_type(self):
        # 定义包含 Tensor[] 类型的列表
        tensorlists = [
            torch.ops.aten.where.default._schema.returns[0].type,
            # 定义包含 Tensor?[] 类型的列表
            torch.ops.aten.index.Tensor._schema.arguments[1].type,
            # 解析 foo 函数的参数，返回 Tensor[]? 类型
            torch._C.parse_schema("foo(Tensor[]? x) -> ()").arguments[0].type,
            # 解析 foo 函数的参数，返回 Tensor?[]? 类型
            torch._C.parse_schema("foo(Tensor?[]? x) -> ()").arguments[0].type,
        ]
        # 定义不包含 Tensor[] 类型的列表
        non_tensorlists = [
            # 定义包含 Tensor 类型的列表
            torch.ops.aten.sin.default._schema.arguments[0].type,
            # 定义包含 IntList 类型的列表
            torch.ops.aten.sum.dim_IntList._schema.arguments[1].type,
        ]
        # 对 tensorlists 中的每个元素执行断言，验证是否为 tensorlist 类型
        for a in tensorlists:
            self.assertTrue(torch._library.utils.is_tensorlist_like_type(a))
        # 对 non_tensorlists 中的每个元素执行断言，验证是否不为 tensorlist 类型
        for a in non_tensorlists:
            self.assertFalse(torch._library.utils.is_tensorlist_like_type(a))

    def test_backward_impl_on_existing_op(self):
        # 创建一个库实例
        lib = self.lib()
        # 定义 foo 函数的签名
        lib.define("foo(Tensor x) -> Tensor")
        # 构建函数的全限定名
        qualname = f"{self.test_ns}::foo"

        # 自定义 foo 函数的实现
        @custom_ops.impl(qualname)
        def foo_impl(x):
            # 在计算梯度时禁用梯度计算
            with torch.no_grad():
                return x.sin()

        # 保存 foo 函数的梯度计算方式
        @custom_ops.impl_save_for_backward(qualname)
        def foo_save_for_backward(inputs, output):
            return inputs.x

        # 实现 foo 函数的反向传播
        @custom_ops.impl_backward(qualname)
        def foo_backward(ctx, saved, grad_out):
            return {"x": grad_out * saved.cos()}

        # 获取 foo 函数的操作对象
        op = self.get_op(qualname)
        # 创建一个随机张量 x，并要求计算其梯度
        x = torch.randn([], requires_grad=True)
        # 调用 op 函数计算 y = foo(x)
        y = op(x)
        # 计算 y 对 x 的梯度
        (gx,) = torch.autograd.grad(y, x)
        # 断言计算得到的梯度 gx 等于 x.cos()
        self.assertEqual(gx, x.cos())

    @parametrize(
        "tags",
        [
            # 测试不同方式定义的标签
            subtest(torch.Tag.pointwise, "single"),
            subtest((torch.Tag.pointwise,), "tuple"),
            subtest([torch.Tag.pointwise], "list"),
        ],
    )
    def test_define_with_tags(self, tags):
        # 创建一个库实例
        lib = self.lib()
        # 定义 foo 函数，指定类型和标签
        torch.library.define(
            f"{self.test_ns}::foo", "(Tensor x) -> Tensor", lib=lib, tags=tags
        )
        # 获取 foo 函数的实际标签
        actual = self.ns().foo.default.tags
        # 断言实际标签类型为列表
        self.assertTrue(isinstance(actual, list))
        # 断言实际标签与预期标签列表相等
        self.assertEqual(actual, list(tags))

    def test_builtin_aten_ops_are_pt2_compliant(self):
        # 遍历内置的 aten 操作，验证其是否符合 pt2 标准
        for op in [torch.ops.aten.sin.default, torch.ops.aten.sum.dim_IntList]:
            self.assertIn(torch.Tag.pt2_compliant_tag, op.tags)

    def test_builtin_torchscript_ops(self):
        # 遍历内置的 TorchScript 操作，验证其是否符合 pt2 标准
        for op in [torch.ops.aten.sub.complex, torch.ops.aten.mul.complex]:
            self.assertIn(torch.Tag.pt2_compliant_tag, op.tags)

    def test_autogen_aten_ops_are_pt2_compliant(self):
        # 遍历自动生成的 aten 操作，验证其是否符合 pt2 标准
        for op in [
            torch.ops.aten.fill.Tensor_out,
        ]:
            # 断言操作标签中包含 generated 和 pt2_compliant_tag
            self.assertIn(torch.Tag.generated, op.tags)
            self.assertIn(torch.Tag.pt2_compliant_tag, op.tags)
    # 测试解析数据包功能
    def test_resolve_packet(self):
        # 创建一个大小为3的随机张量
        x = torch.randn(3)
        # 解析 "aten::sum" 操作的数据包
        result = torch._C._jit_resolve_packet("aten::sum", x)
        # 断言结果与预期的默认值相等
        self.assertEqual(result, "default")

        # 解析带有指定维度参数的 "aten::sum" 操作的数据包
        result = torch._C._jit_resolve_packet("aten::sum", x, dim=1)
        # 断言结果与预期的维度相关值相等
        self.assertEqual(result, "dim_IntList")

        # 使用多个张量参数尝试解析 "aten::sum" 操作的数据包，预期抛出 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "failed to match any schema"):
            result = torch._C._jit_resolve_packet("aten::sum", x, x, x)

    # 测试定义错误模式的架构
    def test_define_bad_schema(self):
        # 获取测试库
        lib = self.lib()
        # 定义一个错误的架构，预期抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "expected schema to look like"):
            torch.library.define(f"{self.test_ns}::foo", "foo(Tensor x) -> Tensor")

    # 测试定义和实现自定义函数库
    def test_define_and_impl(self):
        # 获取测试库
        lib = self.lib()
        # 定义一个简单的函数库函数 "foo"，将其定义为使用 CPU 计算
        torch.library.define(f"{self.test_ns}::foo", "(Tensor x) -> Tensor", lib=lib)

        @torch.library.impl(f"{self.test_ns}::foo", "CPU", lib=lib)
        def f(x):
            return torch.from_numpy(np.sin(x.numpy()))

        # 创建一个随机张量 x
        x = torch.randn(3)
        # 调用实现的函数库函数，计算结果保存为 y
        y = self.ns().foo(x)
        # 断言计算结果 y 与预期结果 x.sin() 非常接近
        assert torch.allclose(y, x.sin())

    # 测试定义过程中的验证
    def test_define_validation(self):
        # 预期在命名空间中定义时抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "namespace"):
            torch.library.define("foo", "(Tensor x) -> Tensor")

    # 测试传统的函数库定义方式
    def test_legacy_define(self):
        # 获取测试库
        lib = self.lib()

        # 使用传统装饰器定义一个函数库函数 "foo"
        @torch.library.define(lib, "foo(Tensor x) -> Tensor")
        def f(x):
            return torch.from_numpy(np.sin(x.numpy()))

        # 创建一个随机张量 x
        x = torch.randn(3)
        # 调用函数库中的函数 "foo"，计算结果保存为 y
        y = self.ns().foo(x)
        # 断言计算结果 y 与预期结果 x.sin() 非常接近
        assert torch.allclose(y, x.sin())

    # 测试实现自定义函数
    def test_impl_function(self):
        # 获取测试库
        lib = self.lib()
        # 定义一个简单的函数库函数 "foo"，将其定义为使用 CPU 计算
        torch.library.define(f"{self.test_ns}::foo", "(Tensor x) -> Tensor", lib=lib)

        # 定义一个简单的函数实现，实现为使用 CPU 计算
        def f(x):
            return torch.from_numpy(np.sin(x.numpy()))

        # 将实现的函数注册为函数库函数 "foo" 的实现
        torch.library.impl(f"{self.test_ns}::foo", "CPU", f, lib=lib)

        # 创建一个随机张量 x
        x = torch.randn(3)
        # 调用实现的函数库函数，计算结果保存为 y
        y = self.ns().foo(x)
        # 断言计算结果 y 与预期结果 x.sin() 非常接近
        assert torch.allclose(y, x.sin())

    # 测试传统的函数库实现方式
    def test_legacy_impl(self):
        # 获取测试库
        lib = self.lib()
        # 定义一个简单的函数库函数 "foo"
        torch.library.define(f"{self.test_ns}::foo", "(Tensor x) -> Tensor", lib=lib)

        # 使用传统装饰器定义一个函数库函数 "foo" 的实现，使用 CPU 计算
        @torch.library.impl(lib, "foo", "CPU")
        def f(x):
            return torch.from_numpy(np.sin(x.numpy()))

        # 创建一个随机张量 x
        x = torch.randn(3)
        # 调用函数库中的函数 "foo"，计算结果保存为 y
        y = self.ns().foo(x)
        # 断言计算结果 y 与预期结果 x.sin() 非常接近
        assert torch.allclose(y, x.sin())

    # 测试在 Python 中定义的函数库是否成功注册
    def test_defined_in_python(self):
        # 断言默认情况下 sin 操作未在 Python 中定义
        self.assertFalse(torch.ops.aten.sin.default._defined_in_python)
        # 断言 dim_IntList 操作未在 Python 中定义
        self.assertFalse(torch.ops.aten.sum.dim_IntList._defined_in_python)

        # 获取测试库
        lib = self.lib()
        # 定义一个简单的函数库函数 "foo"，将其定义为使用指定库 lib
        torch.library.define("{self._test_ns}::foo", "(Tensor x) -> Tensor", lib=lib)
        # 获取当前命名空间的函数库命名空间
        ns = self.ns()
        # 断言 foo.default 操作已在 Python 中定义
        self.assertTrue(ns.foo.default._defined_in_python)

        # 定义一个重载函数库 "bar.overload"，将其定义为使用指定库 lib
        torch.library.define(
            "{self._test_ns}::bar.overload", "(Tensor x) -> Tensor", lib=lib
        )
        # 断言 bar.overload 操作已在 Python 中定义
        self.assertTrue(ns.bar.overload._defined_in_python)
    # 定义一个私有方法，用于测试实现特定设备的功能
    def _test_impl_device(self, name, types, device):
        # 获取当前库对象
        lib = self.lib()
        # 定义 Torch 库函数，指定函数签名和库对象
        torch.library.define(f"{self.test_ns}::{name}", "(Tensor x) -> Tensor", lib=lib)

        # 使用装饰器将函数注册为 Torch 库的实现
        @torch.library.impl(f"{self.test_ns}::{name}", types)
        def f(x):
            # 将输入张量转换为 NumPy 数组
            x_np = x.cpu().numpy()
            # 计算 NumPy 数组中每个元素的正弦值，并将结果转换为 Torch 张量
            y = torch.from_numpy(np.sin(x_np))
            # 将结果张量移到指定设备上，并返回
            return y.to(device=x.device)

        # 创建一个指定设备上的随机张量
        x = torch.randn(3, device=device)
        # 调用通过 Torch 库实现的函数，获取计算结果
        y = getattr(self.ns(), name)(x)
        # 断言计算结果与输入张量的正弦值相等
        assert torch.allclose(y, x.sin())

    # 测试在 CPU 设备上实现的函数
    def test_impl_device_cpu(self):
        self._test_impl_device("foo1", "default", "cpu")
        self._test_impl_device("foo2", ["cpu"], "cpu")
        self._test_impl_device("foo3", ["cpu", "cuda"], "cpu")

    # 根据条件跳过 CUDA 设备测试
    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    # 测试在 CUDA 设备上实现的函数
    def test_impl_device_cuda(self):
        self._test_impl_device("foo4", "default", "cuda")
        self._test_impl_device("foo5", ["cuda"], "cuda")
        self._test_impl_device("foo6", ["cpu", "cuda"], "cuda")

    # 测试在默认函数命名空间下定义和实现特定设备函数
    def test_impl_device_function(self):
        # 获取当前库对象
        lib = self.lib()
        # 定义 Torch 库函数，指定函数签名和库对象
        torch.library.define(f"{self.test_ns}::foo", "(Tensor x) -> Tensor", lib=lib)

        # 定义处理输入张量的函数
        def f(x):
            # 将输入张量转换为 NumPy 数组
            x_np = x.cpu().numpy()
            # 计算 NumPy 数组中每个元素的正弦值，并将结果转换为 Torch 张量
            y = torch.from_numpy(np.sin(x_np))
            # 将结果张量移到输入张量所在设备上，并返回
            return y.to(device=x.device)

        # 将函数 f 注册为 Torch 库函数的实现，指定默认设备和库对象
        torch.library.impl(f"{self.test_ns}::foo", "default", f, lib=lib)
        # 创建一个随机张量
        x = torch.randn(3)
        # 调用通过 Torch 库实现的函数，获取计算结果
        y = self.ns().foo(x)
        # 断言计算结果与输入张量的正弦值相等
        assert torch.allclose(y, x.sin())

    # 测试在未知设备类型上定义实现函数时引发的异常
    def test_impl_device_invalid(self):
        # 使用断言捕获预期的运行时异常，确保错误消息包含指定内容
        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu, cuda"):
            torch.library.impl("blah::blah", "somethingsomething")

    # 测试自动微分函数支持的后端操作
    def test_autograd_function_backed_op(self):
        cpp_source = """
struct CustomOpAutogradFunction : public torch::autograd::Function<CustomOpAutogradFunction> {
  // 定义静态常量，指示此自定义操作是否可以追踪
  static constexpr bool is_traceable = true;

  // 前向传播函数，接受 AutogradContext 和输入张量 x，直接返回 x
  static torch::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const torch::Tensor& x) {
    return x;
  }

  // 反向传播函数，接受 AutogradContext 和梯度输出列表 grad_output，直接返回 grad_output
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_output) {
    return grad_output;
  }
};

// 自定义操作函数 custom_op_backed_by_autograd_fn，接受输入张量 x，并调用自定义函数 CustomOpAutogradFunction::apply(x)
torch::Tensor custom_op_backed_by_autograd_fn(const torch::Tensor& x) {
  return CustomOpAutogradFunction::apply(x);
}

// 定义 Torch 扩展库 mylib
TORCH_LIBRARY(mylib, m) {
    // 将 custom_op_backed_by_autograd_fn 注册为 mylib 库的操作
    m.def("custom_op_backed_by_autograd_fn", custom_op_backed_by_autograd_fn);
}



        """

        // 使用内联方式加载 C++ 源码作为 Torch 扩展
        module = torch.utils.cpp_extension.load_inline(
            name="mylib",
            cpp_sources=cpp_source,
            // 指定加载的函数为 custom_op_backed_by_autograd_fn
            functions="custom_op_backed_by_autograd_fn",
            verbose=True,
        )

        // 创建一个需要梯度的 2x2 全 1 张量 x
        x = torch.ones(2, 2, requires_grad=True)
        // 克隆 x 并分离计算图，作为参考梯度
        temp = x.clone().detach()
        // 调用自定义操作 custom_op_backed_by_autograd_fn
        out = torch.ops.mylib.custom_op_backed_by_autograd_fn(x)
        // 计算输出张量 out 的和作为损失
        loss = out.sum()
        // 反向传播损失
        loss.backward()
        // 断言原始张量 x 的梯度与参考梯度 temp 相等
        self.assertEqual(x.grad, temp)



def op_with_incorrect_schema(testcase, name):
    // 获取测试用例的库
    lib = testcase.lib()
    // 定义具有特定输入输出签名的操作
    lib.define(f"{name}(Tensor x) -> Tensor")
    // 构建操作的完整限定名称
    qualname = f"{testcase.test_ns}::{name}"
    // 实现操作，使用 lambda 函数实现操作逻辑并指定 Autograd 类型
    lib.impl(name, lambda x: x[:], "CompositeExplicitAutograd")
    // 返回操作对象
    return testcase.get_op(qualname)



class MiniOpTest(CustomOpTestCaseBase):
    test_ns = "mini_op_test"

    // 初始化具有延迟反向传播错误的操作
    def _init_op_delayed_backward_error(self):
        name = "delayed_error"
        qualname = f"{self.test_ns}::{name}"
        // 获取测试用例的库
        lib = self.lib()
        // 定义具有特定输入输出签名的操作
        lib.define(f"{name}(Tensor x) -> Tensor")
        // 实现操作，使用 lambda 函数克隆输入张量，并指定 CompositeExplicitAutograd 类型
        lib.impl(name, lambda x: x.clone(), "CompositeExplicitAutograd")
        // 获取操作对象
        op = self.get_op(qualname)

        // 定义具有 Autograd 功能的操作类 Op
        class Op(torch.autograd.Function):
            @staticmethod
            // 前向传播函数，调用 op 对象处理输入张量 x
            def forward(ctx, x):
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            // 后向传播函数，抛出未实现错误
            def backward(ctx, grad):
                raise NotImplementedError

        // 定义 Autograd 实现，调用 Op 类的 apply 方法
        def autograd_impl(x):
            return Op.apply(x)

        // 重新实现操作，使用 Autograd 类型
        lib.impl(name, autograd_impl, "Autograd")
        // 返回操作对象
        return op

    // 初始化没有抽象实现的操作
    def _init_op_with_no_abstract_impl(self):
        name = "no_abstract"
        qualname = f"{self.test_ns}::{name}"
        // 获取测试用例的库
        lib = self.lib()
        // 定义具有特定输入输出签名的操作，并标记为 pt2_compliant_tag
        lib.define(f"{name}(Tensor x) -> Tensor", tags=(torch.Tag.pt2_compliant_tag,))
        // 实现操作，使用 lambda 函数克隆输入张量，并指定 CPU 类型
        lib.impl(name, lambda x: x.clone(), "CPU")
        // 查询操作对象
        return torch._library.utils.lookup_op(qualname)

    // 设置测试前的初始化操作
    def setUp(self):
        super().setUp()
        // 初始化没有抽象实现的操作和延迟反向传播错误的操作
        self._op_with_no_abstract_impl = self._init_op_with_no_abstract_impl()
        self._op_delayed_backward_error = self._init_op_delayed_backward_error()

    // 测试函数不生成操作检查测试
    @optests.dontGenerateOpCheckTests("Testing this API")
    def test_dont_generate(self):
        // 调用函数创建具有错误模式的操作，并获取操作对象
        op = op_with_incorrect_schema(self, "incorrect_schema")
        // 创建大小为 3 的随机张量 x
        x = torch.randn(3)
        // 调用操作处理输入张量 x
        op(x)
    # 定义测试方法 test_mm，测试 torch.mm 函数的默认实现
    def test_mm(self):
        # 创建一个大小为 (2, 3) 的随机张量 x，要求梯度跟踪
        x = torch.randn(2, 3, requires_grad=True)
        # 创建一个大小为 (3, 5) 的随机张量 y
        y = torch.randn(3, 5)
        # 使用 torch.ops.aten.mm.default 执行矩阵乘法，计算结果
        result = torch.ops.aten.mm.default(x, y)
        # 断言计算结果与 x @ y 相等
        self.assertEqual(result, x @ y)

    # 定义测试方法 test_mm_meta，测试带有 meta 设备的情况下 torch.mm 函数的默认实现
    def test_mm_meta(self):
        # 创建一个大小为 (2, 3) 的随机张量 x，要求梯度跟踪，使用 meta 设备
        x = torch.randn(2, 3, requires_grad=True, device="meta")
        # 创建一个大小为 (3, 5) 的随机张量 y，使用 meta 设备
        y = torch.randn(3, 5, device="meta")
        # 使用 torch.ops.aten.mm.default 执行矩阵乘法，计算结果
        result = torch.ops.aten.mm.default(x, y)
        # 断言计算结果的形状与 x @ y 的形状相等
        self.assertEqual(result.shape, (x @ y).shape)

    # 定义测试方法 test_mm_fake，测试使用虚拟张量模式下 torch.mm 函数的默认实现
    def test_mm_fake(self):
        # 进入虚拟张量模式
        with torch._subclasses.fake_tensor.FakeTensorMode():
            # 创建一个大小为 (2, 3) 的随机张量 x，要求梯度跟踪，使用 CPU 设备
            x = torch.randn(2, 3, requires_grad=True, device="cpu")
            # 创建一个大小为 (3, 5) 的随机张量 y，使用 CPU 设备
            y = torch.randn(3, 5, device="cpu")
            # 使用 torch.ops.aten.mm.default 执行矩阵乘法，计算结果
            result = torch.ops.aten.mm.default(x, y)
            # 断言计算结果的形状与 x @ y 的形状相等
            self.assertEqual(result.shape, (x @ y).shape)

    # 定义测试方法 test_mm_errors，测试当张量尺寸不匹配时 torch.mm 函数的默认实现
    def test_mm_errors(self):
        # 创建一个大小为 (2, 3) 的随机张量 x，要求梯度跟踪
        x = torch.randn(2, 3, requires_grad=True)
        # 创建一个大小为 (4, 5) 的随机张量 y
        y = torch.randn(4, 5)
        # 断言在执行 torch.ops.aten.mm.default(x, y) 时抛出 RuntimeError 异常，异常信息包含 "cannot be multiplied"
        with self.assertRaisesRegex(RuntimeError, "cannot be multiplied"):
            result = torch.ops.aten.mm.default(x, y)

    # 定义测试方法 test_nonzero，测试 torch.nonzero 函数的默认实现
    def test_nonzero(self):
        # 创建一个张量 x，包含元素 [0, 1, 2, 0, 0]
        x = torch.tensor([0, 1, 2, 0, 0])
        # 使用 torch.ops.aten.nonzero.default 执行非零元素索引查找
        y = torch.ops.aten.nonzero.default(x)
        # 断言结果 y 与预期结果 torch.tensor([[1], [2]]) 相等
        self.assertEqual(y, torch.tensor([[1], [2]]))

    # 定义测试方法 test_inplace，测试 torch.sin_ 函数的默认实现
    def test_inplace(self):
        # 创建一个大小为 (3,) 的随机张量 x
        x = torch.randn(3)
        # 克隆张量 x 得到 x_clone
        x_clone = x.clone()
        # 使用 torch.ops.aten.sin_ 执行就地正弦计算
        y = torch.ops.aten.sin_(x)
        # 断言计算结果与 x_clone.sin() 的结果相等
        self.assertEqual(x, x_clone.sin())

    # 定义测试方法 test_incorrect_schema，测试具有不正确模式的操作函数
    def test_incorrect_schema(self):
        # 调用带有不正确模式的操作函数 op_with_incorrect_schema，并断言其抛出异常
        op = op_with_incorrect_schema(self, "incorrect_schema")
        x = torch.randn(3)
        op(x)

    # 定义测试方法 test_no_abstract，测试没有抽象实现的操作函数
    def test_no_abstract(self):
        # 获取没有抽象实现的操作函数 op，并调用该函数
        op = self._op_with_no_abstract_impl
        x = torch.randn(3)
        op(x)

    # 定义测试方法 test_delayed_error，测试延迟错误处理的操作函数
    def test_delayed_error(self):
        # 获取具有延迟反向传播错误的操作函数 op，并调用该函数
        op = self._op_delayed_backward_error
        x = torch.randn([], requires_grad=True)
        y = op(x)
        # 断言在调用 y.sum().backward() 时抛出 NotImplementedError 异常
        with self.assertRaises(NotImplementedError):
            y.sum().backward()

    # 定义测试方法 test_delayed_error_no_requires_grad，测试非梯度跟踪模式下的延迟错误处理
    def test_delayed_error_no_requires_grad(self):
        # 获取具有延迟反向传播错误的操作函数 op，并调用该函数
        op = self._op_delayed_backward_error
        x = torch.randn([])
        y = op(x)
        # 在不要求梯度跟踪的情况下，调用 y.sum().backward()，不应抛出异常
        # （因为没有梯度信息，不会尝试执行反向传播）
class TestCustomOpAPI(TestCase):
    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_basic(self):
        # 定义自定义操作函数 add，将 x 与浮点数 y 相加
        @torch.library.custom_op("_torch_testing::add", mutates_args=())
        def add(x: Tensor, y: float) -> Tensor:
            # 将输入张量 x 转换为 NumPy 数组
            x_np = x.numpy(force=True)
            # 执行加法操作
            out_np = x_np + y
            # 将 NumPy 数组转换为 Torch 张量，并确保在与 x 相同的设备上
            return torch.from_numpy(out_np).to(x.device)

        # 创建一个随机张量 x
        x = torch.randn(3)
        # 定义一个浮点数 y
        y = 3.14
        # 使用自定义操作 add 对 x 和 y 进行加法操作，得到结果 z
        z = add(x, y)
        # 断言 z 等于标准 Torch 加法操作 x + y 的结果
        self.assertEqual(z, x + y)

        # 声明一个变量用于检查是否在 CPU 上调用了自定义函数
        cpu_called = False

        # 注册 CPU 上的 add 内核
        @add.register_kernel("cpu")
        def _(x, y):
            nonlocal cpu_called
            cpu_called = True
            # 将输入张量 x 转换为 NumPy 数组
            x_np = x.numpy()
            # 执行加法操作
            out_np = x_np + y
            # 将 NumPy 数组转换为 Torch 张量
            return torch.from_numpy(out_np)

        # 再次使用 add 进行加法操作
        z = add(x, y)
        # 断言 z 等于标准 Torch 加法操作 x + y 的结果
        self.assertEqual(z, x + y)
        # 断言在 CPU 上调用了自定义函数
        self.assertTrue(cpu_called)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_no_grad_skips_autograd(self):
        # 定义自定义操作函数 add，将 x 与浮点数 y 相加
        @torch.library.custom_op("_torch_testing::add", mutates_args=())
        def add(x: Tensor, y: float) -> Tensor:
            # 将输入张量 x 转换为 NumPy 数组
            x_np = x.numpy(force=True)
            # 执行加法操作
            out_np = x_np + y
            # 将 NumPy 数组转换为 Torch 张量，并确保在与 x 相同的设备上
            return torch.from_numpy(out_np).to(x.device)

        # 记录调用次数的变量
        called = 0

        # 设置上下文函数 setup_context
        def setup_context(ctx, inputs, output):
            nonlocal called
            called += 1

        # 反向传播函数 backward，用于在自动求导时抛出异常
        def backward(ctx, grad):
            raise AssertionError("should not be reached")

        # 注册自动求导函数和上下文设置函数
        add.register_autograd(backward, setup_context=setup_context)

        # 创建一个随机张量 x，并声明其需要梯度
        x = torch.randn(3, requires_grad=True)
        # 使用 torch.no_grad() 上下文，调用自定义操作 add
        with torch.no_grad():
            y = add(x, 2.0)
        # 断言 setup_context 函数未被调用
        self.assertEqual(called, 0)
        # 断言 y 等于标准 Torch 加法操作 x + 2.0 的结果
        self.assertEqual(y, x + 2.0)

        # 将 x 的 requires_grad 属性设置为 False
        x.requires_grad_(False)
        # 再次调用自定义操作 add
        y = add(x, 2.0)
        # 断言 setup_context 函数未被调用
        self.assertEqual(called, 0)
        # 断言 y 等于标准 Torch 加法操作 x + 2.0 的结果
        self.assertEqual(y, x + 2.0)

        # 创建一个随机张量 x，并声明其需要梯度
        x = torch.randn(3, requires_grad=True)
        # 再次调用自定义操作 add
        y = add(x, 2.0)
        # 断言 setup_context 函数被调用了一次
        self.assertEqual(called, 1)
        # 断言 y 等于标准 Torch 加法操作 x + 2.0 的结果
        self.assertEqual(y, x + 2.0)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_manual_schema(self):
        # 定义自定义操作函数 add，将 x 与浮点数 y 相加
        @torch.library.custom_op(
            "_torch_testing::add",
            mutates_args=(),
            schema="(Tensor x, float y) -> Tensor",
        )
        def add(x, y):
            # 将输入张量 x 转换为 NumPy 数组
            x_np = x.numpy(force=True)
            # 执行加法操作
            out_np = x_np + y
            # 将 NumPy 数组转换为 Torch 张量，并确保在与 x 相同的设备上
            return torch.from_numpy(out_np).to(x.device)

        # 创建一个随机张量 x
        x = torch.randn(3)
        # 定义一个浮点数 y
        y = 3.14
        # 使用自定义操作 add 对 x 和 y 进行加法操作，得到结果 z
        z = add(x, y)
        # 断言 z 等于标准 Torch 加法操作 x + y 的结果
        self.assertEqual(z, x + y)

        # 定义自定义操作函数 sin_，对输入张量 x 执行正弦函数操作
        @torch.library.custom_op(
            "_torch_testing::sin_",
            mutates_args=["x"],
            schema="(Tensor(a!) x) -> ()",
        )
        def sin_(x):
            # 将输入张量 x 转换为 NumPy 数组
            x_np = x.numpy()
            # 在 NumPy 数组上执行正弦函数
            np.sin(x_np, out=x_np)

        # 创建一个随机张量 x
        x = torch.randn(3)
        # 计算标准 Torch 正弦函数的期望值
        expected = x.sin()
        # 调用自定义操作 sin_ 对 x 执行正弦函数操作
        sin_(x)
        # 断言自定义操作后的张量 x 等于标准 Torch 正弦函数操作的结果 expected
        self.assertEqual(x, expected)
    # 定义测试方法，用于测试只接受关键字参数的张量操作函数的行为
    def test_kwarg_only_tensors(self):
        # 断言捕获 NotImplementedError 异常，并验证异常消息中是否包含 "kwarg-only Tensor args"
        with self.assertRaisesRegex(NotImplementedError, "kwarg-only Tensor args"):
            
            # 使用装饰器定义自定义 Torch 操作 "_torch_testing::foo"，不改变任何参数
            @torch.library.custom_op("_torch_testing::foo", mutates_args=())
            def foo(x: Tensor, *, y: int, z: Tensor) -> Tensor:
                pass

        with self.assertRaisesRegex(NotImplementedError, "kwarg-only Tensor args"):
            # 使用装饰器定义自定义 Torch 操作 "_torch_testing::foo"，其中第三个参数可选
            @torch.library.custom_op("_torch_testing::foo", mutates_args=())
            def foo2(x: Tensor, *, y: int, z: Optional[Tensor]) -> Tensor:
                pass

        with self.assertRaisesRegex(NotImplementedError, "kwarg-only Tensor args"):
            # 使用装饰器定义自定义 Torch 操作 "_torch_testing::foo"，其中第三个参数是张量列表
            @torch.library.custom_op("_torch_testing::foo", mutates_args=())
            def foo3(x: Tensor, *, y: int, z: List[Tensor]) -> Tensor:
                pass

        # 进入 Torch 库的命名空间 "_torch_testing"，在其中注册新的操作 "foo"
        with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
            # 定义操作 "foo" 的签名，指定仅接受一个张量参数和一个关键字参数张量
            lib.define("foo(Tensor x, *, Tensor y) -> Tensor")
            # 验证在注册自动求导时是否抛出 "kwarg-only Tensor args" 的 NotImplementedError
            with self.assertRaisesRegex(NotImplementedError, "kwarg-only Tensor args"):
                torch.library.register_autograd(
                    "_torch_testing::foo",
                    lambda grad: grad,
                    setup_context=lambda ctx, inputs, keyword_only_inputs, output: None,
                )

    # 跳过 Torch Dynamo 测试（因为没有 FakeTensor 支持，不是 bug ）
    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    # 测试注册低级别的关键字参数张量操作的自动求导行为
    def test_register_autograd_kwargonly_low_level(self):
        # 进入 Torch 库的命名空间 "_torch_testing"，在其中注册新的操作 "foo"
        with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
            # 定义操作 "foo" 的签名，指定接受一个张量参数和一个浮点数关键字参数
            lib.define("foo(Tensor x, *, float y) -> Tensor")
            called = False

            # 定义操作 "foo" 的实现函数，返回输入张量乘以关键字参数的结果
            def foo_impl(x, *, y):
                return x * y

            # 将实现函数注册到 Torch 库中，指定在 CPU 上执行
            lib.impl("foo", foo_impl, "CPU")

            # 定义反向传播函数，根据上下文和梯度计算梯度
            def backward(ctx, grad):
                nonlocal called
                called = True
                return grad * ctx.y

            # 定义设置上下文的函数，验证关键字参数的存在
            def setup_context(ctx, inputs, keyword_only_inputs, output):
                assert tuple(keyword_only_inputs.keys()) == ("y",)
                ctx.y = keyword_only_inputs["y"]

            # 注册自动求导函数 "_torch_testing::foo"，指定反向传播和上下文设置函数
            torch.library.register_autograd(
                "_torch_testing::foo", backward, setup_context=setup_context, lib=lib
            )

            # 创建一个需要梯度的随机张量
            x = torch.randn(3, requires_grad=True)
            # 执行操作 "_torch_testing::foo"，对结果求和并反向传播
            torch.ops._torch_testing.foo(x, y=3.14).sum().backward()
            # 验证反向传播函数是否被调用
            self.assertTrue(called)
            # 验证张量 x 的梯度是否正确计算
            self.assertEqual(x.grad, torch.tensor([3.14, 3.14, 3.14]))
    def test_register_autograd_defaults(self):
        # 使用 torch.library._scoped_library 函数创建名为 "_torch_testing" 的库，并设定作用域为 "FRAGMENT"
        with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
            # 在库中定义一个名为 "foo" 的函数签名，接受参数为 Tensor w, int x = 2, *, int y = 3, int z，返回 Tensor
            lib.define("foo(Tensor w, int x = 2, *, int y = 3, int z) -> Tensor")

            # 定义 Python 函数 foo_impl 实现 foo 函数的具体操作
            def foo_impl(w, x=2, *, y=3, z):
                return w * x * y * z

            # 将 foo_impl 函数注册为 "foo" 函数的实现，限定为在 "CPU" 上运行
            lib.impl("foo", foo_impl, "CPU")

            called = False

            # 定义反向传播函数 backward，用于计算梯度
            def backward(ctx, grad):
                nonlocal called
                called = True
                return grad * ctx.c

            # 定义设置上下文函数 setup_context，用于设置上下文信息
            def setup_context(ctx, inputs, keyword_only_inputs, output):
                assert len(inputs) == 2
                assert inputs[1] == 2
                assert keyword_only_inputs == {"y": 3, "z": 42}
                ctx.c = keyword_only_inputs["y"] * keyword_only_inputs["z"] * inputs[1]

            # 注册自动求导功能，将 backward 函数绑定到 "_torch_testing::foo" 上，同时指定 setup_context 函数和库 lib
            torch.library.register_autograd(
                "_torch_testing::foo", backward, setup_context=setup_context, lib=lib
            )

            # 创建一个需要梯度的随机张量 w
            w = torch.randn(3, requires_grad=True)
            # 调用 "_torch_testing::foo" 操作，并计算其结果的和的反向传播
            torch.ops._torch_testing.foo(w, z=42).sum().backward()
            # 断言 backward 函数被调用
            self.assertTrue(called)
            # 断言 w 的梯度计算结果符合预期，即全为 2 * 3 * 42
            self.assertEqual(w.grad, torch.full_like(w, 2 * 3 * 42))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_manual_schema_error(self):
        # 使用 self.assertRaisesRegex 检查是否引发 ValueError 异常，异常信息包含 "the op mutates {'x'}"
        with self.assertRaisesRegex(ValueError, "the op mutates {'x'}"):

            # 使用 torch.library.custom_op 自定义操作 "_torch_testing::sin_"，设置其不改变参数
            # 操作接受 Tensor 类型的参数 x，并不返回任何值
            @torch.library.custom_op(
                "_torch_testing::sin_",
                mutates_args=(),
                schema="(Tensor(a!) x) -> ()",
            )
            # 定义 sin_ 函数实现，调用 numpy 库计算输入张量 x 的正弦值
            def sin_(x):
                x_np = x.numpy()
                np.sin(x_np, out=x_np)
    # 定义一个测试方法，用于测试是否支持张量列表操作
    def test_supports_tensorlist(self):
        # 声明一个自定义的 Torch 自动求导函数 Stack，标记为支持张量列表操作
        @torch._library.autograd.supports_tensorlist
        class Stack(torch.autograd.Function):
            # 前向传播方法：接收上下文 ctx 和张量列表 xs，计算堆叠后的张量并返回
            @staticmethod
            def forward(ctx, xs):
                # 记录张量列表的长度到上下文 ctx 中
                ctx.num_xs = len(xs)
                return torch.stack(xs)

            # 反向传播方法：接收上下文 ctx 和梯度 grad，执行反向传播计算
            @staticmethod
            def backward(ctx, grad):
                # 期望的输入梯度是否需要计算的标志
                expected = ([True] * ctx.num_xs,)
                # 断言当前输入梯度计算标志与期望值相同
                self.assertEqual(ctx.needs_input_grad, expected)
                # 解绑梯度张量 grad，返回一个未绑定的张量列表
                return list(grad.unbind(0))

        # 调用两次 apply 方法，对第一个 apply 的结果执行反向传播
        def t():
            # 生成一个随机张量，设置 requires_grad=True 启用自动求导
            return torch.randn([], requires_grad=True)

        # 创建两个张量列表
        xs0 = [t(), t(), t()]
        xs1 = [t(), t(), t(), t()]
        # 对两个张量列表分别调用 Stack 自动求导函数的 apply 方法
        y0 = Stack.apply(xs0)
        y1 = Stack.apply(xs1)
        # 计算 y0 的总和的梯度，并断言梯度值为 [1.0, 1.0, 1.0]
        grads = torch.autograd.grad(y0.sum(), xs0)
        self.assertEqual(grads, [torch.tensor(1.0) for _ in range(3)])

        # 对一个张量列表执行一次 apply 方法，然后多次进行反向传播
        xs = [t(), t(), t()]
        y = Stack.apply(xs)
        _ = torch.autograd.grad(y.sum(), xs, retain_graph=True)
        _ = torch.autograd.grad(y.sum(), xs, retain_graph=True)
        grads = torch.autograd.grad(y.sum(), xs, retain_graph=True)
        # 断言最后一次梯度计算的结果为 [1.0, 1.0, 1.0]
        self.assertEqual(grads, [torch.tensor(1.0) for _ in range(3)])

        # 错误：尝试直接访问 forward 和 backward 方法，预期抛出 NotImplementedError 异常
        with self.assertRaisesRegex(NotImplementedError, "Function.forward directly"):
            Stack.forward(None, xs)
        with self.assertRaisesRegex(NotImplementedError, "Function.backward directly"):
            Stack.backward(None, xs)

        # 递归情况下的自动求导函数 Foo 的定义，标记为支持张量列表操作
        @torch._library.autograd.supports_tensorlist
        class Foo(torch.autograd.Function):
            # 前向传播方法：接收上下文 ctx 和张量列表 xs，如果长度大于1则递归调用自身，否则计算正弦值并返回
            @staticmethod
            def forward(ctx, xs):
                if len(xs) > 1:
                    return Foo.apply(xs[1:])
                # 记录张量列表的长度到上下文 ctx 中
                ctx.len_xs = len(xs)
                return xs[0].sin()

            # 反向传播方法：接收上下文 ctx 和梯度 grad，计算反向传播梯度并返回
            @staticmethod
            def backward(ctx, grad):
                # 创建一个与输入张量列表长度相同的结果列表，最后一个梯度为 grad 的余弦值
                result = [None] * ctx.len_xs
                result[-1] = grad.cos()
                return result

        # 应该正常工作：对张量列表 xs 执行 Foo 自动求导函数的 apply 方法
        result = Foo.apply(xs)
        expected = xs[-1].sin()
        # 断言结果与预期的正弦值相同
        self.assertEqual(result, expected)

        # 递归调用的反向传播情况下的自动求导函数 Bar 的定义，标记为支持张量列表操作
        @torch._library.autograd.supports_tensorlist
        class Bar(torch.autograd.Function):
            # 前向传播方法：接收上下文 ctx 和张量列表 xs，返回一个列表，每个元素为输入张量加上其索引值
            @staticmethod
            def forward(ctx, xs):
                return [xs[i] + i for i in range(len(xs))]

            # 反向传播方法：接收上下文 ctx 和梯度列表 grads，递归调用自身对梯度列表进行反向传播并返回
            @staticmethod
            def backward(ctx, grads):
                f1 = Bar.apply(grads[:2])
                f2 = Bar.apply(grads[2:])
                return f1 + f2

        # 创建包含 5 个需要梯度的零张量的列表 xs
        xs = [torch.tensor(0.0, requires_grad=True) for _ in range(5)]
        # 对列表 xs 执行 Bar 自动求导函数的 apply 方法
        ys = Bar.apply(xs)
        # 计算 ys 中所有元素的和的梯度
        sum(ys).backward()
        # 获取每个输入张量的梯度结果
        result = [xi.grad for xi in xs]
        # 断言梯度结果与预期的 [1.0, 2.0, 1.0, 2.0, 3.0] 相同
        self.assertEqual(result, torch.tensor([1.0, 2, 1, 2, 3]).unbind(0))

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    # 定义一个测试方法，用于测试默认参数值
    def test_default_values(self):
        # 初始化一个空列表，用于收集默认参数值
        defaults = []

        # 定义一个自定义 Torch 操作函数，注册为 "_torch_testing::f"
        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        # 函数 f 接收多个参数，包括张量 x 和多个具有默认值的参数
        def f(
            x: Tensor,
            a: Optional[int] = None,
            b: float = 3.14,
            c: bool = True,
            d: int = 3,
            e: str = "foo",
            f: torch.dtype = torch.float,
            g: torch.dtype = torch.float32,
            h: torch.dtype = torch.int,
        ) -> Tensor:
            # 将默认参数值依次添加到 defaults 列表中
            defaults.extend([a, b, c, d, e, f, g, h])
            # 返回张量 x 的克隆
            return x.clone()

        # 创建一个随机张量 x
        x = torch.randn(3)
        # 调用函数 f，传入张量 x
        f(x)
        # 断言 defaults 列表中的值与预期的默认参数值列表相同
        self.assertEqual(
            defaults,
            [None, 3.14, True, 3, "foo", torch.float, torch.float32, torch.int],
        )

    # 定义一个测试方法，用于测试带有 mutates_args 的错误情况
    def test_mutated_error(self):
        # 使用断言检测是否抛出预期的 ValueError 异常
        with self.assertRaisesRegex(
            ValueError, r".*{'y'} in mutates_args were not found"
        ):
            # 定义一个自定义 Torch 操作函数，注册为 "_torch_testing::numpy_sin_inplace"
            @torch.library.custom_op(
                "_torch_testing::numpy_sin_inplace",
                mutates_args={"y"},
                device_types="cpu",
            )
            # 函数 numpy_sin_inplace 接收一个张量 x，并在其上执行就地 numpy.sin 操作
            def numpy_sin_inplace(x: Tensor) -> None:
                x_np = x.numpy()
                np.sin(x_np, out=x_np)

    # 定义一个测试方法，用于测试带有 mutates_args 的情况
    def test_mutated(self):
        # 定义一个自定义 Torch 操作函数，注册为 "_torch_testing::numpy_sin_inplace"
        @torch.library.custom_op(
            "_torch_testing::numpy_sin_inplace", mutates_args={"x"}, device_types="cpu"
        )
        # 函数 numpy_sin_inplace 接收一个张量 x，并在其上执行就地 numpy.sin 操作
        def numpy_sin_inplace(x: Tensor) -> None:
            x_np = x.numpy()
            np.sin(x_np, out=x_np)

        # 创建一个随机张量 x
        x = torch.randn(3)
        # 记录张量 x 的版本号
        version = x._version
        # 计算张量 x 的正弦值，作为预期结果
        expected = x.sin()
        # 调用函数 numpy_sin_inplace，在张量 x 上执行就地 sin 操作
        numpy_sin_inplace(x)
        # 断言张量 x 的当前值与预期值相等
        self.assertEqual(x, expected)
        # 断言张量 x 的版本号已增加
        self.assertGreater(x._version, version)

        # 定义一个自定义 Torch 操作函数，注册为 "_torch_testing::f"，带有多个 mutates_args
        @torch.library.custom_op("_torch_testing::f", mutates_args={"y", "z", "w"})
        # 函数 f 接收张量 x 和其他张量参数 y, z, w，并无返回值
        def f(
            x: Tensor, y: Optional[Tensor], z: List[Tensor], w: List[Optional[Tensor]]
        ) -> None:
            return

        # 创建多个随机张量 x, y, z, w
        x = torch.randn(3)
        y = torch.randn(3)
        z = [torch.randn(3), torch.randn(3)]
        w = [torch.randn(3), None, torch.randn(3)]
        # 记录所有张量的初始版本号
        initial_versions = pytree.tree_map_only(
            torch.Tensor, lambda x: x._version, (x, y, z, w)
        )
        # 调用函数 f，传入张量 x, y, z, w
        f(x, y, z, w)
        # 记录调用后所有张量的新版本号
        new_versions = pytree.tree_map_only(
            torch.Tensor, lambda x: x._version, (x, y, z, w)
        )

        # 断言张量 x 的版本号未改变
        self.assertEqual(initial_versions[0], new_versions[0])
        # 只比较张量 y, z, w 的版本号，确保它们都有增加
        initial_versions, _ = pytree.tree_flatten(initial_versions[1:])
        new_versions, _ = pytree.tree_flatten(new_versions[1:])
        for prev, after in zip(initial_versions, new_versions):
            if prev is None and after is None:
                continue
            # 断言新版本号比旧版本号大
            self.assertGreater(after, prev)

    # 跳过 Torch Dynamo 测试，因为 FakeTensor 支持不可用，不是错误
    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    # 参数化测试，使用不同的索引值进行多次测试
    @parametrize("idx", [0, 1, 2, 3, 4, 5])
    # 定义一个测试方法，用于测试注册虚拟操作源的功能
    def test_library_register_fake_source(self, idx):
        # 根据索引生成虚拟操作名称
        opname = f"source{idx}"
        # 使用 getattr 获取 torch._torch_testing 模块中名称为 opname 的对象，并取其默认值
        op = getattr(torch.ops._torch_testing, opname).default
        # 从简单注册表中查找操作 op 对应的条目
        entry = torch._library.simple_registry.singleton.find(op._name)
        # 获取条目中的虚拟实现的内核源代码
        source = entry.fake_impl.kernel.source
        # 断言确保源代码不为 None
        assert source is not None
        # 断言确保 "custom_op_db.py" 存在于源代码中
        self.assertTrue("custom_op_db.py" in source)

    # 装饰器，如果 Torch Dynamo 模式下跳过测试，因为没有 FakeTensor 支持，这不是一个 bug
    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    # 定义测试方法，用于测试注册虚拟操作的功能
    def test_library_register_fake(self):
        # 遍历三种模式：function, qualname, opoverload
        for mode in ["function", "qualname", "opoverload"]:

            # 根据 mode 不同定义不同的虚拟操作函数
            @torch.library.custom_op("_torch_testing::add", mutates_args=())
            def add(x: Tensor, y: float) -> Tensor:
                # 将张量 x 转移到 CPU，再转换成 numpy 数组
                x_np = x.cpu().numpy()
                # 执行简单的加法操作
                out_np = x_np + y
                # 将结果转换成 Torch 张量并移动到与 x 相同的设备上
                return torch.from_numpy(out_np).to(x.device)

            called = False

            # 根据 mode 不同选择注册虚拟操作的方式
            if mode == "function":
                # 使用 torch.library.register_fake 注册虚拟操作函数 add
                dec = torch.library.register_fake(add)
                self.assertIsNotNone(dec)
            elif mode == "qualname":
                # 使用 torch.library.register_fake 注册指定名称的虚拟操作 "_torch_testing::add"
                dec = torch.library.register_fake("_torch_testing::add")
                self.assertIsNotNone(dec)
            elif mode == "opoverload":
                # 使用 torch.library.register_fake 注册指定的操作对象 torch.ops._torch_testing.add.default
                dec = torch.library.register_fake(torch.ops._torch_testing.add.default)
                self.assertIsNotNone(dec)
            else:
                # 如果 mode 不在预期的三种模式中，抛出断言错误
                raise AssertionError("should not get here")

            # 匿名函数装饰器，用于替换注册的虚拟操作函数
            @dec
            def _(x, y):
                # 使用 nonlocal 标记变量 called，在函数内部修改外部变量
                nonlocal called
                called = True
                # 返回一个与输入 x 形状相同的空张量
                return torch.empty_like(x)

            # 在 FakeTensor 模式下执行以下代码块
            with torch._subclasses.fake_tensor.FakeTensorMode():
                # 生成一个随机张量 x
                x = torch.randn(3)
                # 定义一个浮点数 y
                y = 3.14
                # 调用虚拟操作函数 add
                z = add(x, y)
                # 断言虚拟操作的输出张量 z 的形状与输入张量 x 的形状相同
                self.assertEqual(z.shape, x.shape)
                # 断言确保虚拟操作函数被调用过
                self.assertTrue(called)

    # 装饰器，如果 Torch Dynamo 模式下跳过测试，因为没有 FakeTensor 支持，这不是一个 bug
    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    # 定义一个测试函数，用于测试注册自定义操作的不同模式和调用方式
    def test_library_register_kernel(self):
        # 定义模式列表：函数模式、限定名称模式、操作重载模式
        modes = ["function", "qualname", "opoverload"]
        # 定义调用方式列表：装饰器调用、普通函数调用
        calls = ["decorator", "function"]
        # 定义设备类型选项列表：CPU、无指定设备类型
        device_types_options = ["cpu", None]

        # 使用 itertools.product 生成所有模式、调用方式、设备类型的组合
        for mode, call, device_types in itertools.product(
            modes, calls, device_types_options
        ):

            # 定义一个名为 add 的自定义操作函数
            @torch.library.custom_op(
                "_torch_testing::add", mutates_args=(), device_types="cuda"
            )
            def add(x: Tensor, y: float) -> Tensor:
                # 将输入张量 x 转移到 CPU，再转换为 NumPy 数组
                x_np = x.cpu().numpy()
                # 执行加法操作
                out_np = x_np + y
                # 将结果转换为 PyTorch 张量，并保持在原始设备上
                return torch.from_numpy(out_np).to(x.device)

            # 根据模式选择要注册的操作对象
            if mode == "function":
                op = add  # 如果模式为函数，则使用 add 函数对象
            elif mode == "qualname":
                op = "_torch_testing::add"  # 如果模式为限定名称，则使用指定的操作名称
            else:
                assert mode == "opoverload"
                op = torch.ops._torch_testing.add.default  # 操作重载模式下使用默认操作对象

            # 初始化一个标志来表示是否调用了注册的内核函数
            called = False

            # 根据调用方式选择注册内核函数的方式
            if call == "decorator":
                # 使用装饰器方式注册内核函数
                @torch.library.register_kernel(op, device_types)
                def _(x, y):
                    nonlocal called
                    called = True
                    x_np = x.numpy()
                    out_np = x_np + y
                    return torch.from_numpy(out_np)

            else:
                assert call == "function"

                # 定义一个普通函数 add_cpu 来作为内核函数的实现
                def add_cpu(x, y):
                    nonlocal called
                    called = True
                    x_np = x.numpy()
                    out_np = x_np + y
                    return torch.from_numpy(out_np)

                # 使用普通函数方式注册内核函数
                torch.library.register_kernel(op, device_types, add_cpu)

            # 生成一个形状为 (3,) 的随机张量 x 和一个标量 y
            x = torch.randn(3)
            y = 3.14
            # 调用 add 函数执行操作
            z = add(x, y)
            # 断言操作的结果正确
            self.assertEqual(z, x + y)
            # 断言内核函数被调用
            self.assertTrue(called)

    # 跳过测试，因为 TorchDynamo 不支持 FakeTensor，预期会失败，这不是一个 bug
    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    # 定义一个测试方法，用于低级别注册库函数
    def test_library_register_kernel_low_level(self):
        # 定义三种不同的模式
        modes = ["qualname", "opoverload"]
        # 定义两种不同的调用方式
        calls = ["decorator", "function"]
        # 定义设备类型选项，包括元组和单个设备类型的字符串
        device_types_options = [("cpu", "cuda"), "cpu", None]

        # 对模式、调用方式和设备类型选项进行笛卡尔积迭代
        for mode, call, device_types in itertools.product(
            modes, calls, device_types_options
        ):
            # 进入 Torch 库的测试上下文管理器
            with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
                # 定义一个叫做 "add9" 的库函数
                lib.define("add9(Tensor x, float y) -> Tensor")

                # 根据模式选择使用不同的操作符
                if mode == "qualname":
                    op = "_torch_testing::add9"
                else:
                    assert mode == "opoverload"
                    op = torch.ops._torch_testing.add9.default

                # 初始化一个变量来标记是否调用了库函数
                called = False

                # 根据调用方式选择注册方式
                if call == "decorator":
                    # 使用装饰器注册库函数
                    @torch.library.register_kernel(op, device_types, lib=lib)
                    def _(x, y):
                        nonlocal called
                        called = True
                        # 将 Tensor 转换为 NumPy 数组
                        x_np = x.numpy()
                        # 执行加法操作
                        out_np = x_np + y
                        # 将 NumPy 数组转换回 Tensor
                        return torch.from_numpy(out_np)

                else:
                    assert call == "function"

                    # 定义一个普通的 Python 函数作为库函数的实现
                    def add_cpu(x, y):
                        nonlocal called
                        called = True
                        x_np = x.numpy()
                        out_np = x_np + y
                        return torch.from_numpy(out_np)

                    # 使用函数注册库函数
                    torch.library.register_kernel(op, device_types, add_cpu, lib=lib)

                # 生成一个随机的 Tensor
                x = torch.randn(3)
                y = 3.14
                # 调用注册的库函数进行计算
                z = torch.ops._torch_testing.add9.default(x, y)
                # 断言计算结果与预期相符
                self.assertEqual(z, x + y)
                # 断言库函数被调用过
                self.assertTrue(called)

    # 如果是 Torch Dynamo 模式，则跳过此测试
    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_library_register_autograd(self):
        # 循环遍历三种模式：function、qualname、opoverload
        for mode in ["function", "qualname", "opoverload"]:

            # 定义一个自定义操作函数，注册到 torch 库中
            @torch.library.custom_op("mylib::numpy_sin", mutates_args=())
            def numpy_sin(x: Tensor) -> Tensor:
                # 将输入张量 x 转换为 NumPy 数组
                x_np = x.cpu().numpy()
                # 计算 NumPy 数组的 sin 函数
                y_np = np.sin(x_np)
                # 将 NumPy 数组转换回 Torch 张量，保留在原设备上
                return torch.from_numpy(y_np).to(device=x.device)

            # 定义一个设置上下文的函数，用于保存输入和输出
            def setup_context(ctx, inputs, output) -> Tensor:
                # 解包输入
                (x,) = inputs
                # 在上下文中保存输入张量 x
                ctx.save_for_backward(x)

            # 标志变量，用于检查反向传播是否被调用
            called = False

            # 定义反向传播函数
            def backward(ctx, grad):
                nonlocal called
                called = True
                # 从上下文中加载保存的张量 x
                (x,) = ctx.saved_tensors
                # 返回梯度乘以 x 的余弦值
                return grad * x.cos()

            # 根据不同的模式注册自定义操作的反向传播函数
            if mode == "function":
                torch.library.register_autograd(
                    numpy_sin, backward, setup_context=setup_context
                )
            elif mode == "qualname":
                torch.library.register_autograd(
                    "mylib::numpy_sin", backward, setup_context=setup_context
                )
            elif mode == "opoverload":
                torch.library.register_autograd(
                    torch.ops.mylib.numpy_sin.default,
                    backward,
                    setup_context=setup_context,
                )

            # 创建一个随机张量 x，并启用其梯度
            x = torch.randn(3, requires_grad=True)
            # 对张量 x 应用自定义操作 numpy_sin
            y = numpy_sin(x)
            # 计算 y 对 x 的梯度
            (grad_x,) = torch.autograd.grad(y, x, torch.ones_like(y))
            # 断言反向传播函数被调用
            self.assertTrue(called)
            # 断言计算得到的梯度与预期的 x 的余弦值相等
            self.assertEqual(grad_x, x.cos())

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    # 定义一个测试方法，用于测试自动微分注册的低级接口
    def test_library_register_autograd_low_level(self):
        # 遍历两种模式：qualname 和 opoverload
        for mode in ["qualname", "opoverload"]:
            # 使用 torch.library._scoped_library 创建临时库 "_torch_testing"，模式为 "FRAGMENT"
            with torch.library._scoped_library("_torch_testing", "FRAGMENT") as lib:
                # 定义一个名为 "sin5" 的函数签名，接受一个名为 x 的 Tensor 输入，返回一个 Tensor 输出
                lib.define("sin5(Tensor x) -> Tensor")

                # 定义一个 numpy_sin 函数，将输入 x 转为 numpy 数组，计算 sin 函数，再转回 Tensor 对象
                def numpy_sin(x: Tensor) -> Tensor:
                    x_np = x.cpu().detach().numpy()
                    y_np = np.sin(x_np)
                    return torch.from_numpy(y_np).to(device=x.device)

                # 定义 setup_context 函数，用于保存输入数据 x
                def setup_context(ctx, inputs, output) -> Tensor:
                    (x,) = inputs
                    ctx.save_for_backward(x)

                called = False

                # 定义反向传播函数 backward，设置局部变量 called 为 True，计算梯度 grad
                def backward(ctx, grad):
                    nonlocal called
                    called = True
                    (x,) = ctx.saved_tensors
                    return grad * x.cos()

                # 将 numpy_sin 函数注册为 "sin5" 的实现，使用 CPU 运行
                lib.impl("sin5", numpy_sin, "CPU")

                called = False

                # 根据模式注册自动微分函数
                if mode == "qualname":
                    # 使用 qualname 模式注册自动微分函数 "_torch_testing::sin5"
                    torch.library.register_autograd(
                        "_torch_testing::sin5",
                        backward,
                        setup_context=setup_context,
                        lib=lib,
                    )
                elif mode == "opoverload":
                    # 使用 opoverload 模式注册自动微分函数 torch.ops._torch_testing.sin5.default
                    torch.library.register_autograd(
                        torch.ops._torch_testing.sin5.default,
                        backward,
                        setup_context=setup_context,
                        lib=lib,
                    )

                # 创建一个需要梯度的随机输入张量 x
                x = torch.randn(3, requires_grad=True)
                # 调用自定义操作 "_torch_testing.sin5"，计算输出 y
                y = torch.ops._torch_testing.sin5(x)
                # 计算 y 对 x 的梯度 grad_x
                (grad_x,) = torch.autograd.grad(y, x, torch.ones_like(y))
                # 断言 called 变量为 True，表示反向传播函数被调用
                self.assertTrue(called)
                # 断言计算得到的梯度 grad_x 等于 x 的余弦值
                self.assertEqual(grad_x, x.cos())

    # 如果 Torch Dynamo 不支持 FakeTensor，测试将会失败，这是预期的，不是 bug
    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_fake(self):
        # 定义一个自定义操作 "_torch_testing::add"，不会修改参数
        @torch.library.custom_op("_torch_testing::add", mutates_args=())
        def add(x: Tensor, y: float) -> Tensor:
            # 将输入 x 转为 numpy 数组，执行加法操作，返回 Tensor 对象
            x_np = x.cpu().numpy()
            out_np = x_np + y
            return torch.from_numpy(out_np).to(x.device)

        # 创建一个随机张量 x 和一个浮点数 y
        x = torch.randn(3)
        y = 3.14
        # 调用自定义操作 add，计算结果 z
        z = add(x, y)
        # 断言 z 等于 x + y
        self.assertEqual(z, x + y)

        # 使用 FakeTensorMode 上下文管理器测试
        try:
            with torch._subclasses.fake_tensor.FakeTensorMode():
                # 创建一个新的随机张量 x
                x = torch.randn(3)
                # 调用 add 函数
                add(x, y)
            # 如果代码执行到这里，抛出 AssertionError
            raise AssertionError("should not be hit")
        except RuntimeError as e:
            abstract_impl_error_msg = str(e)
        # 替换错误消息中的地址信息，使其更具可读性
        abstract_impl_error_msg = re.sub(
            r"0x.*>\)>", "0xDEADBEEF>)>", abstract_impl_error_msg
        ).replace(". ", ".\n")
        # 断言错误消息符合预期
        self.assertExpectedInline(
            abstract_impl_error_msg,
            """\
        There was no fake impl registered for <CustomOpDef(_torch_testing::add)>.
        This is necessary for torch.compile/export/fx tracing to work.
        Please use `add.register_fake` to add an fake impl.""",
        )

        # 如果未注册 _torch_testing::add 的虚拟实现，则会出现警告消息

        if not IS_WINDOWS:
            # 仅当不在 Windows 环境下时执行以下代码块

            @torch.compile(backend="eager")
            def f(x, y):
                return add(x, y)

            # 定义一个使用 torch.compile 注册的函数 f，尝试调用 add 函数
            # 如果未注册 add 的虚拟实现，会抛出 RuntimeError 异常

            x = torch.randn(3)
            with self.assertRaisesRegex(RuntimeError, "no fake impl"):
                f(x, y)

        abstract_called = False

        # 定义一个匿名函数作为 add 的虚拟实现
        @add.register_fake
        def _(x, y):
            nonlocal abstract_called
            abstract_called = True
            return torch.empty_like(x)

        # 使用 torch._subclasses.fake_tensor.FakeTensorMode() 上下文管理器
        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.randn(3)
            z = add(x, y)
            # 确保 add 的虚拟实现被调用，并检查 abstract_called 变量
            self.assertEqual(z.shape, x.shape)
            self.assertTrue(abstract_called)

    @skipIfTorchDynamo("recursive dynamo")
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work on windows")
    def test_compile(self):
        called_impl = False
        called_abstract = False

        # 定义一个自定义操作 custom_linear
        @torch.library.custom_op("_torch_testing::linear", mutates_args=())
        def custom_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
            nonlocal called_impl
            called_impl = True
            x_np = x.numpy()
            w_np = weight.numpy()
            b_np = bias.numpy()
            out_np = np.add(x_np @ w_np.T, bias)
            return out_np

        # 定义 custom_linear 的虚拟实现
        @custom_linear.register_fake
        def _(x, weight, bias):
            nonlocal called_abstract
            called_abstract = True
            assert x.dim() == 2
            assert weight.dim() == 2
            assert bias.dim() == 1
            assert x.shape[1] == weight.shape[1]
            assert weight.shape[0] == bias.shape[0]
            assert x.device == weight.device
            return x.new_empty(x.size(0), weight.size(0))

        x = torch.randn(2, 2)
        weight = torch.randn(2, 2)
        bias = torch.randn(2)
        # 使用 torch.compile 调用 custom_linear 函数
        out = torch.compile(custom_linear, backend="eager", fullgraph=True)(
            x, weight, bias
        )
        self.assertEqual(out, torch.nn.functional.linear(x, weight, bias))
        self.assertTrue(called_impl)
        self.assertTrue(called_abstract)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_register_autograd_error_cases(self):
        # 定义一个自定义操作 g
        @torch.library.custom_op("_torch_testing::g", mutates_args=())
        def g(x: Tensor) -> Tensor:
            return x.sin()

        x = torch.randn(3, requires_grad=True)
        y = g(x)
        # 对使用自定义操作 g 的张量进行反向传播，预期抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "no autograd formula"):
            y.sum().backward()

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_replacement(self):
        # 定义自定义操作函数 f，使用 Torch 库的装饰器声明，无变异参数
        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            # 返回张量 x 的正弦值
            return x.sin()

        # 创建一个形状为 (3,) 的随机张量 x
        x = torch.randn(3)
        # 调用自定义操作函数 f，计算 y = sin(x)
        y = f(x)
        # 断言 y 与 sin(x) 的值相等
        self.assertEqual(y, x.sin())

        # 重新定义自定义操作函数 f，返回张量 x 的余弦值
        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            return x.cos()

        # 再次调用自定义操作函数 f，计算 y = cos(x)
        y = f(x)
        # 断言 y 与 cos(x) 的值相等
        self.assertEqual(y, x.cos())

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_split_device(self):
        # 初始化 CPU 和 CUDA 调用计数器
        cpu_call_count = 0
        cuda_call_count = 0

        # 定义自定义操作函数 f，设备类型为 CPU
        @torch.library.custom_op(
            "_torch_testing::f", mutates_args=(), device_types="cpu"
        )
        def f(x: Tensor) -> Tensor:
            nonlocal cpu_call_count
            cpu_call_count += 1
            # 将张量 x 转换为 numpy 数组
            x_np = x.numpy()
            # 计算 x_np 的正弦值
            out_np = np.sin(x_np)
            # 将结果转换为 Torch 张量
            return torch.from_numpy(out_np)

        # 在 CPU 上测试自定义操作函数 f
        x = torch.randn(3)
        y = f(x)
        # 断言 y 与 sin(x) 的值相等
        self.assertEqual(y, x.sin())
        # 断言 CPU 调用计数器为 1
        self.assertEqual(cpu_call_count, 1)
        # 断言 CUDA 调用计数器为 0
        self.assertEqual(cuda_call_count, 0)

        # 将张量 x 移动到 CUDA 设备上
        x = x.cuda()
        # 再次调用自定义操作函数 f 在 CUDA 上
        y = f(x)
        # 断言 y 与 sin(x) 的值相等
        self.assertEqual(y, x.sin())
        # 断言 CPU 调用计数器为 1（未增加）
        self.assertEqual(cpu_call_count, 1)
        # 断言 CUDA 调用计数器为 1
        self.assertEqual(cuda_call_count, 1)

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    @unittest.skipIf(not TEST_CUDA, "requires CUDA")
    def test_multi_types(self):
        # 定义自定义操作函数 f，支持 CPU 和 CUDA 设备类型
        @torch.library.custom_op(
            "_torch_testing::f", mutates_args=(), device_types=("cpu", "cuda")
        )
        def f(x: Tensor) -> Tensor:
            # 将张量 x 转换为 numpy 数组
            x_np = x.cpu().numpy()
            # 计算 x_np 的正弦值
            out_np = np.sin(x_np)
            # 将结果转换为 Torch 张量，并保持在原设备上
            return torch.from_numpy(out_np).to(x.device)

        # 在 CPU 上测试自定义操作函数 f
        x = torch.randn(3)
        y = f(x)
        # 断言 y 与 sin(x) 的值相等
        self.assertEqual(y, x.sin())
        
        # 将张量 x 移动到 CUDA 设备上
        x = x.cuda()
        # 再次调用自定义操作函数 f 在 CUDA 上
        y = f(x)
        # 断言 y 与 sin(x) 的值相等
        self.assertEqual(y, x.sin())

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_overloading(self):
        # 初始化函数调用计数器
        called_f = 0
        called_f1 = 0

        # 定义自定义操作函数 f，无变异参数
        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            nonlocal called_f
            called_f += 1
            # 返回张量 x 的克隆
            return x.clone()

        # 创建一个形状为 (2, 3) 的随机张量 x
        x = torch.randn(2, 3)
        # 调用函数 f，不修改参数，计数器增加
        torch.ops._torch_testing.f(x)
        # 断言函数 f 的调用次数为 1
        self.assertEqual(called_f, 1)

        # 定义自定义操作函数 f1，无变异参数，重载版本
        @torch.library.custom_op("_torch_testing::f.overload", mutates_args=())
        def f1(x: Tensor, y: Tensor) -> Tensor:
            nonlocal called_f1
            called_f1 += 1
            # 返回张量 x 的克隆
            return x.clone()

        # 调用函数 f，重载版本，计数器增加
        torch.ops._torch_testing.f(x, x)
        # 断言函数 f1 的调用次数为 1
        self.assertEqual(called_f1, 1)
    # 定义一个测试方法，用于验证不允许输出参数别名的情况
    def test_disallows_output_aliasing(self):
        # 定义一个名为f的自定义操作函数，不修改参数
        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            # 返回张量x的视图，展平为一维
            return x.view(-1)

        # 创建一个形状为(3,)的随机张量x
        x = torch.randn(3)
        # 使用断言检查调用f(x)是否会引发RuntimeError，错误信息包含"may not alias"
        with self.assertRaisesRegex(RuntimeError, "may not alias"):
            f(x)

        # 重新定义名为f的自定义操作函数，不修改参数
        @torch.library.custom_op("_torch_testing::f", mutates_args=())
        def f(x: Tensor) -> Tensor:
            # 直接返回张量x
            return x

        # 重新创建一个形状为(3,)的随机张量x
        x = torch.randn(3)
        # 使用断言再次检查调用f(x)是否会引发RuntimeError，错误信息包含"may not alias"
        with self.assertRaisesRegex(RuntimeError, "may not alias"):
            f(x)

        # 定义一个名为numpy_sin_inplace的自定义操作函数，在CPU设备上修改参数x
        @torch.library.custom_op(
            "_torch_testing::f", mutates_args={"x"}, device_types="cpu"
        )
        def numpy_sin_inplace(x: Tensor) -> Tensor:
            # 将张量x转换为NumPy数组x_np
            x_np = x.numpy()
            # 对NumPy数组x_np执行正弦函数操作，结果写回x_np
            np.sin(x_np, out=x_np)
            # 返回修改后的张量x
            return x

        # 重新创建一个形状为(3,)的随机张量x
        x = torch.randn(3)
        # 使用断言再次检查调用numpy_sin_inplace(x)是否会引发RuntimeError，错误信息包含"may not alias"
        with self.assertRaisesRegex(RuntimeError, "may not alias"):
            numpy_sin_inplace(x)
class MiniOpTestOther(CustomOpTestCaseBase):
    test_ns = "mini_op_test"  # 设置测试命名空间为 "mini_op_test"

    def test_nonzero_again(self):
        x = torch.tensor([0, 1, 2, 0, 0])  # 创建一个张量 x
        y = torch.ops.aten.nonzero.default(x)  # 使用 ATen 操作计算张量 x 的非零元素索引
        self.assertEqual(y, torch.tensor([[1], [2]]))  # 断言 y 的结果为 [[1], [2]]


optests.generate_opcheck_tests(
    MiniOpTest,  # 生成 MiniOpTest 类的操作检查测试
    ["aten", "mini_op_test"],  # 测试涵盖的命名空间列表
    get_file_path_2(
        os.path.dirname(__file__),  # 获取当前文件所在目录的路径
        "minioptest_failures_dict.json",  # 指定的 JSON 文件路径
    ),
    additional_decorators={
        "test_pt2_compliant_tag_mini_op_test_no_abstract": [unittest.expectedFailure]  # 添加额外的装饰器，标记为预期失败的测试
    },
    test_utils=optests.generate_tests.DEPRECATED_DEFAULT_TEST_UTILS,  # 指定测试工具
)

optests.generate_opcheck_tests(
    MiniOpTestOther,  # 生成 MiniOpTestOther 类的操作检查测试
    ["aten", "mini_op_test"],  # 测试涵盖的命名空间列表
    get_file_path_2(
        os.path.dirname(__file__),  # 获取当前文件所在目录的路径
        "minioptest_failures_dict.json",  # 指定的 JSON 文件路径
    ),
    test_utils=optests.generate_tests.DEPRECATED_DEFAULT_TEST_UTILS,  # 指定测试工具
)


class TestGenerateOpcheckTests(CustomOpTestCaseBase):
    def test_MiniOpTest(self):
        for orig_test in ["test_mm", "test_nonzero"]:
            for (
                test
            ) in torch.testing._internal.optests.generate_tests.DEFAULT_TEST_UTILS:
                expected_test = f"{test}__{orig_test}"  # 构建期望的测试方法名
                self.assertTrue(hasattr(MiniOpTest, expected_test), msg=expected_test)  # 断言 MiniOpTest 类中存在期望的测试方法

    def test_generate_repro_save_data(self):
        from torch.testing._internal.optests.generate_tests import generate_repro

        args = (torch.ones(2, 2),)  # 创建输入参数 args
        kwargs = {"mat2": torch.zeros(2, 2)}  # 创建输入关键字参数 kwargs
        actual = generate_repro(
            "test_schema",  # 指定测试模式
            torch.ops.aten.sin.default,  # 指定要生成 Repro 脚本的操作
            args,  # 输入参数
            kwargs,  # 输入关键字参数
            save_data=True,  # 是否保存数据
            dry_run=True,  # 是否仅测试运行，不实际执行
        )
        actual = re.sub(r"torch.load\(\".*\.pt\"\)", 'torch.load("repro.pt")', actual)  # 替换加载数据的语句
        self.assertExpectedInline(
            actual,
            """\
# =========================================================
# BEGIN REPRO SCRIPT
# =========================================================
import torch
from torch.testing._internal.optests import opcheck

# Make sure you have loaded the library that contains the op
# via an import or torch.ops.load_library(...)
op = torch.ops.aten.sin.default

args, kwargs = torch.load("repro.pt")
opcheck(op, args, kwargs, test_utils="test_schema")
# =========================================================
# END REPRO SCRIPT
# =========================================================
""",
        )

    def test_generate_repro_no_save_data(self):
        from torch.testing._internal.optests.generate_tests import generate_repro

        args = (torch.ones(2, 2),)  # 创建输入参数 args
        kwargs = {"mat2": torch.zeros(2, 2)}  # 创建输入关键字参数 kwargs
        actual = generate_repro(
            "test_schema",  # 指定测试模式
            torch.ops.aten.sin.default,  # 指定要生成 Repro 脚本的操作
            args,  # 输入参数
            kwargs,  # 输入关键字参数
            save_data=False,  # 不保存数据
            dry_run=True,  # 是否仅测试运行，不实际执行
        )
        self.assertExpectedInline(
            actual,
            """\
# =========================================================
# BEGIN REPRO SCRIPT
# =========================================================

# 导入 torch 库
import torch
# 从 torch.testing._internal.optests 模块中导入 opcheck 函数
from torch.testing._internal.optests import opcheck

# 确保已加载包含操作的库，可以通过 import 或 torch.ops.load_library(...) 实现
op = torch.ops.aten.sin.default

# 定义空元组 args 和空字典 kwargs，作为操作符的参数
args = ()  # 操作符的参数
kwargs = {}  # 操作符的关键字参数

# 调用 opcheck 函数来测试操作符
opcheck(op, args, kwargs, test_utils="test_schema")

# =========================================================
# END REPRO SCRIPT
# =========================================================

# =========================================================
# BEGIN test_failures_dict_validation
# =========================================================

# 从 torch.testing._internal.optests.generate_tests 模块中导入 FailuresDict 和 validate_failures_dict_structure 函数
def test_failures_dict_validation(self):
    from torch.testing._internal.optests.generate_tests import (
        FailuresDict,
        validate_failures_dict_structure,
    )

    # 定义测试用例的失败字典
    failures = {
        "mini_op_test::incorrect_schema": {
            "MiniOpTest.test_aot_dispatch_dynamic__test_delayed_error": {
                "comment": "",
                "status": "success",
            }
        }
    }

    # 断言运行时异常，确保失败字典结构验证时抛出指定错误信息
    with self.assertRaisesRegex(RuntimeError, "got status=success"):
        validate_failures_dict_structure(
            FailuresDict("", failures),
            torch.testing._internal.optests.generate_tests.DEFAULT_TEST_UTILS,
            MiniOpTest,
        )

    # 更新测试用例的失败字典
    failures = {
        "mini_op_test::incorrect_schema": {
            "MiniOpTest.test_aot_dispatch__test_delayed_error": {
                "comment": "",
                "status": "xfail",
            },
        }
    }

    # 断言运行时异常，确保失败字典结构验证时抛出指定错误信息
    with self.assertRaisesRegex(RuntimeError, "should begin with one of"):
        validate_failures_dict_structure(
            FailuresDict("", failures),
            torch.testing._internal.optests.generate_tests.DEFAULT_TEST_UTILS,
            MiniOpTest,
        )

    # 更新测试用例的失败字典
    failures = {
        "mini_op_test::incorrect_schema": {
            "MiniOpTest.test_aot_dispatch_dynamic__test_delayed_error_nopenopenope": {
                "comment": "",
                "status": "xfail",
            },
        }
    }

    # 断言运行时异常，确保失败字典结构验证时抛出指定错误信息
    with self.assertRaisesRegex(RuntimeError, "does not exist on the TestCase"):
        validate_failures_dict_structure(
            FailuresDict("", failures),
            torch.testing._internal.optests.generate_tests.DEFAULT_TEST_UTILS,
            MiniOpTest,
        )

# =========================================================
# END test_failures_dict_validation
# =========================================================

# =========================================================
# BEGIN test_dont_generate_decorator
# =========================================================

# 确认 MiniOpTest 类具有 test_dont_generate 方法
self.assertTrue(hasattr(MiniOpTest, "test_dont_generate"))
# 确认 MiniOpTest 类不具有 test_schema__test_dont_generate 方法
self.assertFalse(hasattr(MiniOpTest, "test_schema__test_dont_generate"))

# =========================================================
# END test_dont_generate_decorator
# =========================================================
    # 定义一个测试函数，用于测试 torch 库中的自定义操作
    def test_opcheck(self):
        # 创建一个形状为 (3,) 的张量 x，并标记为需要梯度计算
        x = torch.randn(3, requires_grad=True)
        
        # 测试在调用操作时是否会抛出指定异常信息
        with self.assertRaisesRegex(ValueError, "OpOverload"):
            torch.library.opcheck(torch.sin, (x,))
        
        # 测试在调用操作时是否会抛出指定异常信息，并验证 test_utils 参数是否为预期的子集
        with self.assertRaisesRegex(ValueError, "test_utils to be subset of"):
            torch.library.opcheck(torch.ops.aten.sin.default, (x,), test_utils="blah")
        
        # 调用操作并获取结果
        result = torch.library.opcheck(torch.ops.aten.sin.default, (x,))
        
        # 断言操作检查的结果是否符合预期
        self.assertEqual(
            result,
            {
                "test_schema": "SUCCESS",
                "test_autograd_registration": "SUCCESS",
                "test_faketensor": "SUCCESS",
                "test_aot_dispatch_dynamic": "SUCCESS",
            },
        )
        
        # 使用特定的 test_utils 参数再次调用操作并获取结果
        result = torch.library.opcheck(
            torch.ops.aten.sin.default, (x,), test_utils="test_schema"
        )
        
        # 断言操作检查的结果是否符合预期
        self.assertEqual(
            result,
            {
                "test_schema": "SUCCESS",
            },
        )
        
        # 使用列表形式的 test_utils 参数再次调用操作并获取结果
        result = torch.library.opcheck(
            torch.ops.aten.sin.default,
            (x,),
            test_utils=["test_schema", "test_faketensor"],
        )
        
        # 断言操作检查的结果是否符合预期
        self.assertEqual(
            result,
            {
                "test_schema": "SUCCESS",
                "test_faketensor": "SUCCESS",
            },
        )

    # 测试自定义操作的 opcheck 函数
    def test_opcheck_customopdef(self):
        # 定义多组输入样例，包括不同形式的张量和可能的 CUDA 张量
        sample_inputs = [
            (torch.randn(3),),
            (torch.randn(3, requires_grad=True),),
        ]
        
        # 如果 CUDA 可用，添加更多的样例输入
        if torch.cuda.is_available():
            sample_inputs.extend(
                [
                    (torch.randn(3, device="cuda"),),
                    (torch.randn(3, device="cuda", requires_grad=True),),
                ]
            )
        
        # 遍历所有样例输入并调用 opcheck 函数
        for args in sample_inputs:
            torch.library.opcheck(custom_op_db.numpy_cube, args)

    # 测试 opcheck 模式是否被正确设置
    def test_is_inside_opcheck_mode(self):
        # 断言当前不在 opcheck 模式中
        self.assertFalse(optests.is_inside_opcheck_mode())
        
        # 进入 opcheck 模式并执行一些断言验证
        with optests.generate_tests.OpCheckMode(
            ["foo"], "bar", lambda x: x, None, "baz", "brr"
        ):
            self.assertTrue(optests.is_inside_opcheck_mode())

    # 测试当操作不正确时 opcheck 函数的行为
    def test_opcheck_bad_op(self):
        # 创建一个具有错误模式的操作对象
        op = op_with_incorrect_schema(self, "foo")
        
        # 创建一个形状为 (3,) 的张量 x
        x = torch.randn(3)
        
        # 断言调用操作时会抛出指定异常信息
        with self.assertRaisesRegex(Exception, "is not defined to alias output"):
            torch.library.opcheck(op, (x,))
        
        # 调用操作并获取结果，但不抛出异常
        result = torch.library.opcheck(op, (x,), raise_exception=False)
        
        # 断言结果中的 test_schema 是否为 RuntimeError 类型
        self.assertTrue(isinstance(result["test_schema"], RuntimeError))
        
        # 从结果中删除 test_schema 条目
        del result["test_schema"]
        
        # 断言剩余的结果是否符合预期
        self.assertEqual(
            result,
            {
                "test_autograd_registration": "SUCCESS",
                "test_faketensor": "SUCCESS",
                "test_aot_dispatch_dynamic": "SUCCESS",
            },
        )
    def test_opcheck_does_not_require_extra_deps(self):
        # 测试函数，验证 opcheck 是否在不需要额外依赖项的情况下可用
        # torch.testing._internal.common_utils 包含许多额外的测试时依赖项。
        # 由于 opcheck 是公共 API，应仅与 pytorch 安装时的依赖项一起使用。
        
        # 定义一个命令列表，运行在当前 Python 解释器下
        cmd = [
            sys.executable,  # 使用当前 Python 解释器
            "-c",  # 执行后续的 Python 代码
            "import torch; import sys; \
               x = torch.randn(3, requires_grad=True); \
               torch.library.opcheck(torch.ops.aten.sin.default, (x,)); \
               assert 'expecttest' not in sys.modules; \
               assert 'torch.testing._internal.common_utils' not in sys.modules",
        ]
        
        # 执行命令，并捕获其输出（不使用 shell）
        subprocess.check_output(cmd, shell=False)
class TestTypeConversion(TestCase):
    """In infer_schema(), we try to suggest a correct type when the type annotation is wrong."""

    def setUp(self):
        # 设置支持的基本类型列表
        self.supported_base_types = [
            int,
            float,
            bool,
            str,
            torch.device,
            torch.Tensor,
            torch.dtype,
            torch.types.Number,
        ]

    def test_simple_tuple(self):
        # 测试简单元组的转换，期望返回列表
        self.assertEqual(List, tuple_to_list(Tuple))

    def test_supported_types(self):
        # 测试支持的基本类型
        for t in self.supported_base_types:
            # 将元组转换为列表并验证结果类型是否符合预期
            result_type = tuple_to_list(Tuple[t, t, t])
            self.assertEqual(result_type, List[t])

            result_type = tuple_to_list(Tuple[t])
            self.assertEqual(result_type, List[t])

    def test_optional(self):
        # 测试可选类型的处理
        for t in self.supported_base_types:
            result_type = tuple_to_list(Tuple[t, Optional[t]])
            self.assertEqual(result_type, List[Optional[t]])

            result_type = tuple_to_list(Tuple[t, t, Optional[t]])
            self.assertEqual(result_type, List[Optional[t]])

            result_type = tuple_to_list(Tuple[t, ...])
            self.assertEqual(result_type, List[t])

    def test_mixed_types(self):
        # 测试混合类型的处理
        result_type = tuple_to_list(Tuple[int, float])
        self.assertEqual(result_type, List[typing.Union[int, float]])

        result_type = tuple_to_list(Tuple[int, float, str])
        self.assertEqual(result_type, List[typing.Union[int, float, str]])


# 仅适用于 "cpu" 和 "cuda"，实例化 TestCustomOpTesting 类的设备类型测试
only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestCustomOpTesting, globals(), only_for=only_for)

# 实例化参数化测试用例 TestCustomOp 的所有实例
instantiate_parametrized_tests(TestCustomOp)

# 实例化参数化测试用例 TestCustomOpAPI 的所有实例
instantiate_parametrized_tests(TestCustomOpAPI)

if __name__ == "__main__":
    # 运行测试
    run_tests()
```