# `.\pytorch\test\test_autograd_fallback.py`

```
# Owner(s): ["module: autograd"]

# 导入必要的模块和库
import contextlib  # 上下文管理模块，用于创建上下文管理器
import warnings  # 用于处理警告信息

import numpy as np  # 导入 NumPy 库

import torch  # 导入 PyTorch 库
from torch.library import _scoped_library, Library  # 导入 Torch 库中的 _scoped_library 和 Library 类
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入用于实例化参数化测试的函数
    parametrize,  # 导入用于参数化的装饰器
    run_tests,  # 导入运行测试的函数
    TestCase,  # 导入测试用例基类
)


@contextlib.contextmanager
def autograd_fallback_mode(mode):
    prev = torch._C._get_autograd_fallback_mode()  # 获取当前自动求导回退模式
    try:
        torch._C._set_autograd_fallback_mode(mode)  # 设置自动求导回退模式为指定模式
        yield  # 执行被装饰代码块
    finally:
        torch._C._set_autograd_fallback_mode(prev)  # 恢复之前的自动求导回退模式


class TestAutogradFallback(TestCase):
    test_ns = "_test_autograd_fallback"

    def tearDown(self):
        if hasattr(torch.ops, self.test_ns):  # 如果 torch.ops 中有指定命名空间的属性
            delattr(torch.ops, self.test_ns)  # 删除该属性
        if hasattr(self, "lib"):  # 如果当前对象有 'lib' 属性
            del self.lib.m  # 删除 'm' 属性
            del self.lib  # 删除 'lib' 属性

    def get_op(self, name):
        return getattr(getattr(torch.ops, self.test_ns), name).default  # 获取指定操作的默认实现

    def get_lib(self):
        lib = Library(self.test_ns, "FRAGMENT")  # 创建一个名为 test_ns 的 Library 对象
        self.lib = lib  # 将创建的 Library 对象存储在当前对象的属性中
        return lib  # 返回创建的 Library 对象

    @parametrize("mode", ("nothing", "warn"))
    def test_no_grad(self, mode):
        with autograd_fallback_mode(mode):  # 使用指定的自动求导回退模式
            lib = self.get_lib()  # 获取 Library 对象
            lib.define("foo(Tensor a, Tensor b, int c) -> Tensor")  # 定义名为 'foo' 的操作签名
            lib.impl("foo", lambda a, b, c: a + b + c, "CPU")  # 实现 'foo' 操作的具体功能和计算设备
            op = self.get_op("foo")  # 获取 'foo' 操作的默认实现

            with warnings.catch_warnings():  # 捕获警告信息
                warnings.simplefilter("error")  # 设置警告过滤器，将警告转换为错误
                with torch.no_grad():  # 使用无梯度上下文管理器
                    a = torch.randn([], requires_grad=True)  # 创建一个需要梯度的随机张量
                    b = torch.randn([], requires_grad=True)  # 创建一个需要梯度的随机张量
                    out = op(a, b, 1)  # 执行 'foo' 操作
                self.assertFalse(out.requires_grad)  # 检查输出张量是否不需要梯度

            with warnings.catch_warnings():  # 再次捕获警告信息
                warnings.simplefilter("error")  # 设置警告过滤器
                a = torch.randn([])  # 创建一个不需要梯度的随机张量
                b = torch.randn([])  # 创建一个不需要梯度的随机张量
                out = op(a, b, 1)  # 执行 'foo' 操作
                self.assertFalse(out.requires_grad)  # 检查输出张量是否不需要梯度

    @parametrize("mode", ("nothing", "warn"))
    def test_no_autograd_kernel(self, mode):
        with autograd_fallback_mode(mode):  # 使用指定的自动求导回退模式
            lib = self.get_lib()  # 获取 Library 对象
            lib.define("foo(Tensor a, Tensor b, int c) -> Tensor")  # 定义名为 'foo' 的操作签名
            op = self.get_op("foo")  # 获取 'foo' 操作的默认实现

            def foo_impl(a, b, c):
                result = a.detach().numpy() + b.detach().numpy() + c  # 实现 'foo' 操作的具体功能
                return torch.tensor(result)  # 返回结果张量

            lib.impl("foo", foo_impl, "CPU")  # 实现 'foo' 操作的具体功能和计算设备

            # Some inputs requiring grad
            a = torch.randn([], requires_grad=False)  # 创建一个不需要梯度的随机张量
            b = torch.randn([], requires_grad=True)  # 创建一个需要梯度的随机张量
            out = op(a, b, 1).sum()  # 执行 'foo' 操作并对结果求和
            with self._check_ctx(mode, mode_nothing_raises=True):  # 使用自定义的上下文管理器进行检查
                out.backward()  # 反向传播计算梯度
            self.assertIsNone(b.grad)  # 断言张量 b 的梯度为空
    def _check_ctx(self, mode, *, mode_nothing_raises=False):
        # 检查上下文模式，根据模式返回相应的上下文管理器
        if mode == "warn":
            # 如果模式为 "warn"，则断言会发出 UserWarning，指示未注册自动微分内核
            return self.assertWarnsRegex(
                UserWarning, "an autograd kernel was not registered"
            )
        assert mode == "nothing"
        # 如果模式为 "nothing"，确保不会发出警告
        if mode_nothing_raises:
            # 如果指定 mode_nothing_raises 为 True，则断言会引发 RuntimeError，指示不需要梯度
            return self.assertRaisesRegex(RuntimeError, "does not require grad")
        # 默认情况下返回一个空的上下文管理器
        return contextlib.nullcontext()

    @parametrize("mode", ("nothing", "warn"))
    def test_no_autograd_kernel_inplace(self, mode):
        # 使用 parametrize 装饰器，将 mode 参数化为 "nothing" 和 "warn"
        with autograd_fallback_mode(mode):
            # 使用 autograd_fallback_mode 上下文管理器，设置自动微分的回退模式
            # 在这个上下文中，对输入进行原地修改，并将其作为输出返回
            lib = self.get_lib()
            # 获取库对象并定义函数签名
            lib.define("foo(Tensor(a!) self, Tensor(b!) y) -> (Tensor(a!), Tensor(b!))")
            # 获取名为 "foo" 的操作
            op = self.get_op("foo")

            def foo_impl(x, y):
                # 定义函数 foo_impl，对 x 和 y 进行原地操作，使用 torch.no_grad 来关闭梯度追踪
                with torch.no_grad():
                    x.sin_()
                    y.cos_()
                return x, y

            # 将 foo_impl 注册为 "foo" 操作的实现，指定在 CPU 上执行
            lib.impl("foo", foo_impl, "CPU")

            # 创建需要梯度的张量 x
            x = torch.randn(3, requires_grad=True)
            w = x.clone()
            v = x.clone()
            y0 = w[0]
            y1 = v[1]
            # 调用操作 op 处理 y0 和 y1
            z0, z1 = op(y0, y1)
            # 对一组张量进行循环处理
            for tensor in [w, v, z0, z1, y0, y1]:
                # 使用 self._check_ctx(mode) 来检查上下文模式
                with self._check_ctx(mode):
                    # 计算张量的和的梯度，保留计算图以便多次反向传播
                    tensor.sum().backward(retain_graph=True)

            # 没有输出：我们什么也不做。也许将来应该做点什么。
            # 这不是常见的失败模式。
            lib.define("bar(Tensor(a!) self) -> ()")
            # 定义一个没有输出的函数 bar
            op = self.get_op("bar")

            def bar_impl(x):
                # 定义函数 bar_impl，对 x 进行原地操作，使用 torch.no_grad 来关闭梯度追踪
                with torch.no_grad():
                    x.sin_()

            # 将 bar_impl 注册为 "bar" 操作的实现，指定在 CPU 上执行
            lib.impl("bar", bar_impl, "CPU")

            # 使用 warnings.catch_warnings() 上下文管理器捕获所有警告
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                # 创建需要梯度的标量张量 x
                x = torch.randn([], requires_grad=True)
                y = x.clone()
                # 调用操作 op 处理 y
                z = op(y)
                # 对 y 进行反向传播
                y.backward()
                # 断言计算的梯度等于 torch.ones_like(x)
                self.assertEqual(x.grad, torch.ones_like(x))
    # 定义测试方法，用于测试 CPU 模式下函数自身返回情况
    def test_cpu_return_self(self, mode):
        # 使用指定的自动求导回退模式
        with autograd_fallback_mode(mode):
            # 在当前命名空间下创建临时库作用域，类型为 "FRAGMENT"
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                # 定义一个名为 foo 的函数，接受一个 Tensor 类型参数并返回 Tensor
                lib.define("foo(Tensor self) -> Tensor")
                # 实现 foo 函数，实现为输入参数自身返回，CPU 模式
                lib.impl("foo", lambda x: x, "CPU")
                # 获取函数 foo 的操作对象
                op = self.get_op("foo")

                # 创建一个形状为 (3,) 的随机张量 x，并要求其梯度计算
                x = torch.randn(3, requires_grad=True)
                # 对操作对象 op 对 x 的结果求和
                y = op(x).sum()
                # 使用 _check_ctx 方法验证当前模式下的上下文
                with self._check_ctx(mode):
                    # 对 y 进行反向传播
                    y.backward()
                    # 断言 x 的梯度应与形状与 x 相同的全 1 张量相等
                    self.assertEqual(x.grad, torch.ones_like(x))

                # 重新定义名为 bar 的函数，接受一个具有梯度属性的 Tensor 类型参数，并返回具有梯度属性的 Tensor
                lib.define("bar(Tensor(a!) self) -> Tensor(a!)")
                # 实现 bar 函数，实现为输入参数自身返回，CPU 模式
                lib.impl("bar", lambda x: x, "CPU")
                # 获取函数 bar 的操作对象
                op = self.get_op("bar")

                # 创建一个形状为 (3,) 的随机张量 x，并要求其梯度计算
                x = torch.randn(3, requires_grad=True)
                # 对操作对象 op 对 x 的结果求和
                y = op(x).sum()
                # 使用 _check_ctx 方法验证当前模式下的上下文
                with self._check_ctx(mode):
                    # 对 y 进行反向传播
                    y.backward()
                    # 断言 x 的梯度应与形状与 x 相同的全 1 张量相等
                    self.assertEqual(x.grad, torch.ones_like(x))

    # 参数化测试方法，测试复合函数注册到 CPU 上的情况
    @parametrize("mode", ("nothing", "warn"))
    def test_composite_registered_to_cpu(self, mode):
        # 使用指定的自动求导回退模式
        with autograd_fallback_mode(mode):
            # 在当前命名空间下创建临时库作用域，类型为 "FRAGMENT"
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                # 定义一个名为 foo 的函数，接受一个 Tensor 类型参数并返回 Tensor
                lib.define("foo(Tensor self) -> Tensor")
                # 实现 foo 函数，实现为输入参数求 sin 函数后求和，CPU 模式
                lib.impl("foo", lambda x: x.sin().sum(), "CPU")
                # 获取函数 foo 的操作对象
                op = self.get_op("foo")

                # 创建一个形状为 (3,) 的随机张量 x，并要求其梯度计算
                x = torch.randn(3, requires_grad=True)
                # 对操作对象 op 对 x 的结果求值
                y = op(x)
                # 使用 _check_ctx 方法验证当前模式下的上下文
                with self._check_ctx(mode):
                    # 对 y 进行反向传播
                    y.backward()
                    # 断言 x 的梯度应与 x 的 cos 函数的结果相等
                    self.assertEqual(x.grad, x.cos())

    # 参数化测试方法，测试自动求导函数注册到 CPU 上的情况
    @parametrize("mode", ("nothing", "warn"))
    def test_autograd_function_registered_to_cpu(self, mode):
        # 使用指定的自动求导回退模式
        with autograd_fallback_mode(mode):
            # 在当前命名空间下创建临时库作用域，类型为 "FRAGMENT"
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                # 定义一个名为 foo 的函数，接受一个 Tensor 类型参数并返回 Tensor
                lib.define("foo(Tensor self) -> Tensor")

                # 定义一个继承自 torch.autograd.Function 的 NumpySin 类
                class NumpySin(torch.autograd.Function):
                    # 实现前向传播方法，保存输入张量并返回其 sin 函数的结果
                    @staticmethod
                    def forward(ctx, x):
                        ctx.save_for_backward(x)
                        return torch.tensor(np.sin(x.cpu().numpy()))

                    # 实现反向传播方法，计算输入梯度
                    @staticmethod
                    def backward(ctx, gx):
                        (x,) = ctx.saved_tensors
                        return gx * x.cos()

                # 实现 foo 函数，使用 NumpySin 的 apply 方法，CPU 模式
                lib.impl("foo", NumpySin.apply, "CPU")
                # 获取函数 foo 的操作对象
                op = self.get_op("foo")

                # 创建一个形状为 (3,) 的随机张量 x，并要求其梯度计算
                x = torch.randn(3, requires_grad=True)
                # 对操作对象 op 对 x 的结果求和
                y = op(x).sum()
                # 使用 _check_ctx 方法验证当前模式下的上下文
                with self._check_ctx(mode):
                    # 对 y 进行反向传播
                    y.backward()
                    # 断言 x 的梯度应与 x 的 cos 函数的结果相等
                    self.assertEqual(x.grad, x.cos())
    # 定义一个测试方法，用于测试 inplace autograd 函数在 CPU 上的注册和使用
    def test_inplace_autograd_function_registered_to_cpu(self, mode):
        # 使用 autograd_fallback_mode 进入指定的模式上下文
        with autograd_fallback_mode(mode):
            # 在当前测试命名空间下创建一个名为 "FRAGMENT" 的作用域库
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                # 在作用域库中定义一个名为 "foo" 的函数，该函数接受一个标记为 inplace 的 Tensor，并返回一个标记为 inplace 的 Tensor
                lib.define("foo(Tensor(a!) self) -> Tensor(a!)")

                # 定义一个继承自 torch.autograd.Function 的类 NumpySin_
                class NumpySin_(torch.autograd.Function):
                    # 静态方法：前向传播函数，接受一个上下文 ctx 和输入张量 x
                    @staticmethod
                    def forward(ctx, x):
                        # 保存输入张量 x 的克隆版本到上下文 ctx 中
                        ctx.save_for_backward(x.clone())
                        # 将 x 分离出来并转换为 numpy 数组 x_np，然后对 x_np 执行正弦函数操作
                        x_np = x.detach().numpy()
                        np.sin(x_np, out=x_np)
                        # 标记输入张量 x 为脏数据，表示已经被修改
                        ctx.mark_dirty(x)
                        # 返回修改后的 x
                        return x

                    # 静态方法：反向传播函数，接受一个上下文 ctx 和梯度 gx
                    @staticmethod
                    def backward(ctx, gx):
                        # 从上下文 ctx 中获取保存的张量 x
                        (x,) = ctx.saved_tensors
                        # 返回梯度 gx 乘以 x 的余弦值作为反向传播结果
                        return gx * x.cos()

                # 在作用域库 lib 中实现函数 "foo"，使用 NumpySin_.apply 作为实现函数，并指定运行在 "CPU" 上
                lib.impl("foo", NumpySin_.apply, "CPU")
                # 获取名称为 "foo" 的操作符 op
                op = self.get_op("foo")

                # 创建一个随机张量 x，并标记为需要计算梯度
                x = torch.randn(3, requires_grad=True)
                # 克隆张量 x 到 z
                z = x.clone()
                # 取 z 的第一个元素并赋值给 w
                w = z[0]
                # 使用 op 计算 w，并赋值给 y
                y = op(w)

                # 创建一个与 x 形状相同的零张量 expected
                expected = torch.zeros_like(x)
                # 将 expected 的第一个元素设置为 x 的第一个元素的余弦值
                expected[0] = x[0].cos()
                # 在指定的上下文 mode 下执行以下代码块
                with self._check_ctx(mode):
                    # 计算 y 对 x 的梯度 gx，使用 torch.autograd.grad 进行计算，retain_graph=True 表示保留计算图
                    (gx,) = torch.autograd.grad(
                        y, x, torch.ones_like(y), retain_graph=True
                    )
                    # 断言计算得到的梯度 gx 等于预期的 expected
                    self.assertEqual(gx, expected)

                # 创建一个与 x 形状相同的全一张量 expected
                expected = torch.ones_like(x)
                # 将 expected 的第一个元素设置为 x 的第一个元素的余弦值
                expected[0] = x[0].cos()
                # 在指定的上下文 mode 下执行以下代码块
                with self._check_ctx(mode):
                    # 计算 z 对 x 的梯度 gx，使用 torch.autograd.grad 进行计算
                    (gx,) = torch.autograd.grad(z, x, torch.ones_like(z))
                    # 断言计算得到的梯度 gx 等于预期的 expected
                    self.assertEqual(gx, expected)
    def test_inplace_on_tensor_that_does_not_require_grad(self, mode):
        # 对于不需要梯度的张量，测试原地操作
        # 详见 NOTE [autograd fallback and in-place operations] 了解为何

        # 使用自动求导回退模式进行上下文管理
        with autograd_fallback_mode(mode):
            # 使用 _scoped_library 进入指定的命名空间
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                # 正确使用 (a!)：声明函数 foo，接受两个张量参数，并且返回第一个张量 (a!)
                lib.define("foo(Tensor(a!) self, Tensor other) -> Tensor(a!)")

                def foo_impl(x, y):
                    # 对输入张量 x 进行分离操作，不影响梯度计算
                    x_d = x.detach()
                    y = y.detach()
                    # 原地加法操作
                    x_d.add_(y)
                    return x

                # 将 foo_impl 函数注册到 lib 中的 foo 函数
                lib.impl("foo", foo_impl, "CPU")
                # 获取 foo 函数的操作句柄
                foo = self.get_op("foo")

                # 不正确使用 (a!)：用户没有返回原始张量
                lib.define("bar(Tensor(a!) self, Tensor other) -> Tensor(a!)")

                def bar_impl(x, y):
                    x_d = x.detach()
                    y = y.detach()
                    x_d.add_(y)
                    # 返回 x_d 的克隆而不是原始张量 x
                    return x_d.clone()

                lib.impl("bar", bar_impl, "CPU")
                bar = self.get_op("bar")

                # 用户修改了输入张量但没有返回它
                lib.define("baz(Tensor(a!) self, Tensor other) -> ()")

                def baz_impl(x, y):
                    x_d = x.detach()
                    y = y.detach()
                    x_d.add_(y)

                lib.impl("baz", baz_impl, "CPU")
                baz = self.get_op("baz")

                # 在非视图上测试原地操作
                for op in (foo, bar, baz):
                    x = torch.randn(3)
                    y = torch.randn(3, requires_grad=True)
                    with self.assertRaisesRegex(RuntimeError, "does not require grad"):
                        z = x.clone()
                        op(z, y)
                        torch.autograd.grad(z, y, torch.ones_like(z), allow_unused=True)

                # 在视图上测试原地操作
                for op in (foo, bar, baz):
                    x = torch.randn(3)
                    y = torch.randn(3, requires_grad=True)
                    with self.assertRaisesRegex(RuntimeError, "does not require grad"):
                        z = x[:]
                        op(z, y)
                        torch.autograd.grad(z, x, torch.ones_like(z), allow_unused=True)
    # 使用给定的执行模式进行测试，自动选择是否回退到自动微分模式
    def test_undefined_inputs_outputs(self, mode):
        with autograd_fallback_mode(mode):  # 进入自动微分回退模式
            # 获取当前库的实例
            lib = self.get_lib()
            # 定义名为 "foo" 的操作，指定输入输出的张量类型
            lib.define("foo(Tensor a, Tensor b) -> (Tensor, Tensor)")
            # 获取名为 "foo" 的操作
            op = self.get_op("foo")

            # 定义操作 "foo" 的实现，这里定义了当第一个输入为 None 时返回 None，第二个输入复制自身
            def foo_impl(a, b):
                return None, b.clone()

            # 将操作 "foo" 的实现注册到库中，并指定在 CPU 上执行
            lib.impl("foo", foo_impl, "CPU")

            # 创建一个随机张量 x，并指定需要计算其梯度
            x = torch.randn(3, requires_grad=True)
            # 注意：PyTorch 分发器将 None 视为未定义的张量
            # 调用操作 "foo"，传入 None 和张量 x，返回结果为 y 和 z
            y, z = op(None, x)
            with self._check_ctx(mode):  # 进入上下文检查模式
                # 对 z 张量的所有元素求和并进行反向传播计算梯度
                z.sum().backward()

    @parametrize("mode", ("nothing", "warn"))
    # 测试未定义梯度的情况
    def test_undefined_grads(self, mode):
        with autograd_fallback_mode(mode):  # 进入自动微分回退模式
            # 获取当前库的实例
            lib = self.get_lib()
            # 定义名为 "foo" 的操作，指定输入输出的张量类型
            lib.define("foo(Tensor a, Tensor b) -> (Tensor, Tensor)")
            # 获取名为 "foo" 的操作
            op = self.get_op("foo")

            # 定义操作 "foo" 的实现，这里定义了对第一个输入取 sin 函数，对第二个输入取 cos 函数
            def foo_impl(a, b):
                return a.sin(), b.cos()

            # 将操作 "foo" 的实现注册到库中，并指定在 CPU 上执行
            lib.impl("foo", foo_impl, "CPU")

            # 创建一个随机张量 x，并指定需要计算其梯度
            x = torch.randn(3, requires_grad=True)
            # 创建另一个随机张量 y
            y = torch.randn(3)
            # 调用操作 "foo"，传入张量 x 和 y，返回结果为 w 和 z
            w, z = op(x, y)
            # 将 w 和 z 张量标记为未定义的梯度
            w = torch._C._functions.UndefinedGrad()(w)
            z = torch._C._functions.UndefinedGrad()(z)
            with self._check_ctx(mode):  # 进入上下文检查模式
                # 对 (z + w) 张量的所有元素求和并进行反向传播计算梯度
                (z + w).sum().backward()

    @parametrize("mode", ("nothing", "warn"))
    # 测试基本张量不需要梯度的情况
    def test_base_does_not_require_grad(self, mode):
        with autograd_fallback_mode(mode):  # 进入自动微分回退模式
            # 获取当前库的实例
            lib = self.get_lib()
            # 定义名为 "foo" 的操作，指定输入输出的张量类型，并标记输入张量 a 不需要梯度
            lib.define("foo(Tensor(a!) x) -> Tensor(a!)")
            # 获取名为 "foo" 的操作
            op = self.get_op("foo")

            # 定义操作 "foo" 的实现，使用 torch.no_grad() 上下文将输入张量 a 清零
            def foo_impl(a):
                with torch.no_grad():
                    return a.zero_()

            # 将操作 "foo" 的实现注册到库中，并指定在 CPU 上执行
            lib.impl("foo", foo_impl, "CPU")
            # 创建一个随机张量 x
            x = torch.randn(3)
            # 将 x 复制给 y，并标记 y 需要计算梯度
            y = x[:]
            y.requires_grad_()
            # 将 y 复制给 w
            w = y[:]
            # 断言 w 的基础张量是 x
            self.assertTrue(w._base is x)

            # 在 w 上注册钩子，但不在 w._base 上注册
            op(w)
            with self._check_ctx(mode):  # 进入上下文检查模式
                # 对 w 张量的所有元素求和并进行反向传播计算梯度
                w.sum().backward()
    # 测试函数，验证 autograd 返回的张量中是否包含 requires_grad=True 和 requires_grad=False 的张量混合
    def test_post_autograd_returns_mix_of_requires_grad_tensors(self, mode):
        # 使用 autograd_fallback_mode 上下文管理器设置 autograd 的模式
        with autograd_fallback_mode(mode):
            # 获取当前的库对象
            lib = self.get_lib()
            # 定义库函数签名
            lib.define("foo(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)")
            # 获取操作符函数
            op = self.get_op("foo")

            # 定义 foo_impl 函数实现
            def foo_impl(a, b):
                # 在无梯度追踪的上下文中进行操作
                with torch.no_grad():
                    # 克隆张量 a 和 b，并赋值给 x 和 z
                    x = a.clone()
                    z = b.clone()
                # 计算张量 a 和 b 的乘积，赋值给 y
                y = a * b
                # 返回克隆的张量 x, 乘积张量 y, 克隆的张量 z
                return x, y, z

            # 将 foo_impl 函数实现注册到库中
            lib.impl("foo", foo_impl, "CPU")

            # 创建 requires_grad=True 的随机张量 a 和 b
            a = torch.randn(3, requires_grad=True)
            b = torch.randn(3, requires_grad=True)

            # 调用操作符 op，传入张量 a 和 b，并接收返回的张量 x, y, z
            x, y, z = op(a, b)

            # 使用 _check_ctx 上下文管理器验证 autograd 的行为
            with self._check_ctx(mode, mode_nothing_raises=True):
                # 对张量 x 进行反向传播
                torch.autograd.grad(
                    x, (a, b), torch.ones_like(x), allow_unused=True, retain_graph=True
                )

            # 再次使用 _check_ctx 上下文管理器验证 autograd 的行为
            with self._check_ctx(mode, mode_nothing_raises=False):
                # 对张量 y 进行反向传播
                torch.autograd.grad(
                    y, (a, b), torch.ones_like(y), allow_unused=True, retain_graph=True
                )

            # 第三次使用 _check_ctx 上下文管理器验证 autograd 的行为
            with self._check_ctx(mode, mode_nothing_raises=True):
                # 对张量 z 进行反向传播
                torch.autograd.grad(
                    z, (a, b), torch.ones_like(z), allow_unused=True, retain_graph=True
                )

    # 参数化测试函数，测试张量列表的支持情况
    @parametrize("mode", ("nothing", "warn"))
    def test_supports_tensor_lists(self, mode):
        # 使用 autograd_fallback_mode 上下文管理器设置 autograd 的模式
        with autograd_fallback_mode(mode):
            # 获取当前的库对象
            lib = self.get_lib()
            # 定义库函数签名
            lib.define("foo(Tensor[] a) -> Tensor[]")
            # 获取操作符函数
            op = self.get_op("foo")

            # 定义 foo_impl 函数实现
            def foo_impl(a):
                # 解包输入的张量列表 a 为 x, y, z
                x, y, z = a
                # 在无梯度追踪的上下文中进行操作
                with torch.no_grad():
                    # 返回 x+y+z 和 x*y*z 的结果
                    return x + y + z, x * y * z

            # 将 foo_impl 函数实现注册到库中
            lib.impl("foo", foo_impl, "CPU")

            # 创建 requires_grad=True 的随机张量 x, y, z
            x = torch.randn(3, requires_grad=True)
            y = torch.randn(1, requires_grad=True)
            z = torch.randn(2, 1, requires_grad=True)

            # 调用操作符 op，传入张量列表 [x, y, z]，并接收返回的张量列表 a, b
            a, b = op([x, y, z])

            # 使用 _check_ctx 上下文管理器验证 autograd 的行为
            with self._check_ctx(mode, mode_nothing_raises=True):
                # 对张量 a 进行反向传播
                torch.autograd.grad(
                    a,
                    (x, y, z),
                    torch.ones_like(a),
                    allow_unused=True,
                    retain_graph=True,
                )

            # 再次使用 _check_ctx 上下文管理器验证 autograd 的行为
            with self._check_ctx(mode, mode_nothing_raises=True):
                # 对张量 b 进行反向传播
                torch.autograd.grad(
                    b,
                    (x, y, z),
                    torch.ones_like(b),
                    allow_unused=True,
                    retain_graph=True,
                )
# 调用函数 `instantiate_parametrized_tests`，并传入参数 `TestAutogradFallback`，用于实例化带参数的测试。
instantiate_parametrized_tests(TestAutogradFallback)

# 检查当前脚本是否作为主程序运行
if __name__ == "__main__":
    # 如果是主程序，则执行函数 `run_tests()`，用于运行测试。
    run_tests()
```