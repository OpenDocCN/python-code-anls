# `.\pytorch\test\jit\test_exception.py`

```py
# Owner(s): ["oncall: jit"]
import torch  # 导入 PyTorch 库
from torch import nn  # 导入神经网络模块
from torch.testing._internal.common_utils import TestCase  # 导入测试用例基类

r"""
Test TorchScript exception handling.
"""


class TestException(TestCase):  # 定义测试异常处理的测试类
    def test_pyop_exception_message(self):  # 测试自定义操作的异常消息
        class Foo(torch.jit.ScriptModule):  # 定义一个继承自 ScriptModule 的类 Foo
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 10, kernel_size=5)  # 创建一个二维卷积层

            @torch.jit.script_method
            def forward(self, x):
                return self.conv(x)  # 返回卷积层处理后的结果

        foo = Foo()  # 创建 Foo 类的实例
        # 测试确保正确的错误消息传播
        with self.assertRaisesRegex(
            RuntimeError, r"Expected 3D \(unbatched\) or 4D \(batched\) input to conv2d"
        ):
            foo(torch.ones([123]))  # 错误的输入尺寸

    def test_builtin_error_messsage(self):  # 测试内置错误消息
        with self.assertRaisesRegex(RuntimeError, "Arguments for call are not valid"):
            @torch.jit.script
            def close_match(x):
                return x.masked_fill(True)  # 使用错误的方法调用

        with self.assertRaisesRegex(
            RuntimeError,
            "This op may not exist or may not be currently " "supported in TorchScript",
        ):
            @torch.jit.script
            def unknown_op(x):
                torch.set_anomaly_enabled(True)  # 启用异常检测
                return x
    # 定义一个测试异常处理的函数
    def test_exceptions(self):
        # 使用 torch.jit.CompilationUnit 创建一个包含函数定义的编译单元
        cu = torch.jit.CompilationUnit(
            """
            def foo(cond):
                if bool(cond):
                    raise ValueError(3)
                return 1
        """
        )

        # 调用 cu 中定义的 foo 函数，传入条件为 torch.tensor(0)，不会引发异常
        cu.foo(torch.tensor(0))
        # 使用 assertRaisesRegex 断言在调用 cu.foo(torch.tensor(1)) 时会引发 torch.jit.Error 异常，并且异常消息包含 "3"
        with self.assertRaisesRegex(torch.jit.Error, "3"):
            cu.foo(torch.tensor(1))

        # 定义一个名为 foo 的函数，接受一个条件参数 cond
        def foo(cond):
            a = 3
            # 如果条件 cond 为真，抛出一个名为 ArbitraryError 的异常，异常消息为 "hi"
            if bool(cond):
                raise ArbitraryError(a, "hi")  # noqa: F821
                # 不会执行到这里，因为前一行已经抛出异常
                if 1 == 2:
                    raise ArbitraryError  # noqa: F821
            # 返回变量 a 的值
            return a

        # 使用 assertRaisesRegex 断言在对 torch.jit.script(foo) 脚本化时会引发 RuntimeError 异常，并且异常消息包含 "undefined value ArbitraryError"
        with self.assertRaisesRegex(RuntimeError, "undefined value ArbitraryError"):
            torch.jit.script(foo)

        # 定义一个将异常对象 Exception 赋值给变量 a 的函数
        def exception_as_value():
            a = Exception()
            print(a)

        # 使用 assertRaisesRegex 断言在对 torch.jit.script(exception_as_value) 脚本化时会引发 RuntimeError 异常，并且异常消息包含 "cannot be used as a value"
        with self.assertRaisesRegex(RuntimeError, "cannot be used as a value"):
            torch.jit.script(exception_as_value)

        # 使用 torch.jit.script 装饰器创建一个总是抛出 RuntimeError 异常的函数 foo_no_decl_always_throws
        @torch.jit.script
        def foo_no_decl_always_throws():
            raise RuntimeError("Hi")

        # 获取 foo_no_decl_always_throws 函数的输出类型，并断言其为 NoneType
        output_type = next(foo_no_decl_always_throws.graph.outputs()).type()
        self.assertTrue(str(output_type) == "NoneType")

        # 使用 torch.jit.script 装饰器创建一个声明总是抛出异常的函数 foo_decl_always_throws
        @torch.jit.script
        def foo_decl_always_throws():
            # type: () -> Tensor
            raise Exception("Hi")  # noqa: TRY002

        # 获取 foo_decl_always_throws 函数的输出类型，并断言其为 Tensor 类型
        output_type = next(foo_decl_always_throws.graph.outputs()).type()
        self.assertTrue(str(output_type) == "Tensor")

        # 定义一个会抛出异常的函数 foo，抛出一个数学运算异常
        def foo():
            raise 3 + 4

        # 使用 assertRaisesRegex 断言在对 torch.jit.script(foo) 脚本化时会引发 RuntimeError 异常，并且异常消息包含 "must derive from BaseException"
        with self.assertRaisesRegex(RuntimeError, "must derive from BaseException"):
            torch.jit.script(foo)

        # 使用 torch.jit.script 装饰器创建一个函数 foo，根据条件返回不同的值或抛出异常
        @torch.jit.script
        def foo():
            # 如果条件为真，设置变量 a 为 1
            if 1 == 1:
                a = 1
            else:
                # 如果条件为假，进入 else 分支，根据条件抛出不同的异常
                if 1 == 1:
                    raise Exception("Hi")  # noqa: TRY002
                else:
                    raise Exception("Hi")  # noqa: TRY002
            # 返回变量 a 的值
            return a

        # 使用断言验证调用 foo() 函数返回的结果为 1
        self.assertEqual(foo(), 1)

        # 使用 torch.jit.script 装饰器创建一个总是抛出带有多个消息参数的 RuntimeError 异常的函数 tuple_fn
        @torch.jit.script
        def tuple_fn():
            raise RuntimeError("hello", "goodbye")

        # 使用 assertRaisesRegex 断言在调用 tuple_fn() 函数时会引发 torch.jit.Error 异常，并且异常消息包含 "hello, goodbye"
        with self.assertRaisesRegex(torch.jit.Error, "hello, goodbye"):
            tuple_fn()

        # 使用 torch.jit.script 装饰器创建一个总是抛出 RuntimeError 异常但没有指定消息的函数 no_message
        @torch.jit.script
        def no_message():
            raise RuntimeError

        # 使用 assertRaisesRegex 断言在调用 no_message() 函数时会引发 torch.jit.Error 异常，并且异常消息为 "RuntimeError"
        with self.assertRaisesRegex(torch.jit.Error, "RuntimeError"):
            no_message()
    def test_assertions(self):
        # 创建 TorchScript 编译单元 cu，包含一个函数 foo，其中包含条件断言
        cu = torch.jit.CompilationUnit(
            """
            def foo(cond):
                assert bool(cond), "hi"
                return 0
        """
        )

        # 调用 foo 函数，传入条件为 True 的张量
        cu.foo(torch.tensor(1))
        # 使用断言检查调用 foo 函数时条件为 False 的情况
        with self.assertRaisesRegex(torch.jit.Error, "AssertionError: hi"):
            cu.foo(torch.tensor(0))

        # 定义一个 TorchScript 函数 foo，包含条件断言
        @torch.jit.script
        def foo(cond):
            assert bool(cond), "hi"

        # 调用 foo 函数，传入条件为 True 的张量
        foo(torch.tensor(1))
        # 使用断言检查调用 foo 函数时条件为 False 的情况
        # 目前不验证异常的名称
        with self.assertRaisesRegex(torch.jit.Error, "AssertionError: hi"):
            foo(torch.tensor(0))

    def test_python_op_exception(self):
        # 定义一个被 TorchScript 忽略的 Python 函数 python_op
        @torch.jit.ignore
        def python_op(x):
            # 抛出一个异常 "bad!"
            raise Exception("bad!")  # noqa: TRY002

        # 定义一个 TorchScript 函数 fn，调用 python_op 函数
        @torch.jit.script
        def fn(x):
            return python_op(x)

        # 使用断言检查调用 fn 函数时是否抛出了 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError, "operation failed in the TorchScript interpreter"
        ):
            fn(torch.tensor(4))

    def test_dict_expansion_raises_error(self):
        # 定义一个 Python 函数 fn，试图返回字典 d 的扩展版本
        def fn(self):
            d = {"foo": 1, "bar": 2, "baz": 3}
            return {**d}

        # 使用断言检查尝试编译 fn 函数时是否抛出了 NotSupportedError 异常
        with self.assertRaisesRegex(
            torch.jit.frontend.NotSupportedError, "Dict expansion "
        ):
            torch.jit.script(fn)

    def test_custom_python_exception(self):
        # 定义一个自定义异常类 MyValueError，继承自 ValueError
        class MyValueError(ValueError):
            pass

        # 定义一个 TorchScript 函数 fn，抛出 MyValueError 异常
        @torch.jit.script
        def fn():
            raise MyValueError("test custom exception")

        # 使用断言检查调用 fn 函数时是否抛出了 MyValueError 异常
        with self.assertRaisesRegex(
            torch.jit.Error, "jit.test_exception.MyValueError: test custom exception"
        ):
            fn()

    def test_custom_python_exception_defined_elsewhere(self):
        # 从外部模块 jit.myexception 导入自定义异常类 MyKeyError
        from jit.myexception import MyKeyError

        # 定义一个 TorchScript 函数 fn，抛出 MyKeyError 异常
        @torch.jit.script
        def fn():
            raise MyKeyError("This is a user defined key error")

        # 使用断言检查调用 fn 函数时是否抛出了 MyKeyError 异常，并验证异常消息
        with self.assertRaisesRegex(
            torch.jit.Error,
            "jit.myexception.MyKeyError: This is a user defined key error",
        ):
            fn()
```