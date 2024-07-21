# `.\pytorch\test\test_native_functions.py`

```py
# 导入必要的模块和类
from typing import Optional, List
import torch
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfTorchDynamo

# End-to-end tests of features in native_functions.yaml

# 定义一个继承自 torch.nn.Module 的类，用于测试 optional float list
class FloatListWrapperModule(torch.nn.Module):
    def forward(self, values, incr: Optional[List[float]]):
        return torch._C._nn._test_optional_floatlist(values, incr)

# 定义一个继承自 torch.nn.Module 的类，用于测试 optional int list
class IntListWrapperModule(torch.nn.Module):
    def forward(self, values, incr: Optional[List[int]]):
        return torch._C._nn._test_optional_intlist(values, incr)

# 测试用例类，继承自 TestCase
class TestNativeFunctions(TestCase):

    # 返回一个包含不同类型参数的列表，用于测试类型错误抛出
    def _lists_with_str(self):
        return [
            ("foo",),
            (2, "foo"),
            ("foo", 3),
            ["foo"],
            [2, "foo"],
            ["foo", 3],
            "foo",
        ]

    # 测试函数，检查传入的函数是否会抛出 TypeError 异常
    def _test_raises_str_typeerror(self, fn):
        for arg in self._lists_with_str():
            self.assertRaisesRegex(TypeError, "str", lambda: fn(arg))
            try:
                fn(arg)
            except TypeError as e:
                print(e)

    # 测试函数，测试 torch._C._nn.pad 函数对 str 类型参数是否会抛出异常
    def test_symintlist_error(self):
        x = torch.randn(1)
        self._test_raises_str_typeerror(lambda arg: torch._C._nn.pad(x, arg))

    # 测试函数，测试 torch.rand 函数对 str 类型参数是否会抛出异常
    def test_vararg_symintlist_error(self):
        self._test_raises_str_typeerror(lambda arg: torch.rand(arg))
        self._test_raises_str_typeerror(lambda arg: torch.rand(*arg))

    # 测试函数，测试 x.set_ 函数对 str 类型参数是否会抛出异常
    def test_symintlist_error_with_overload_but_is_unique(self):
        x = torch.randn(1)
        y = torch.randn(1)
        self._test_raises_str_typeerror(lambda arg: x.set_(y, 0, arg))

    # 测试函数，测试 x.view 函数对 str 类型参数是否会抛出异常
    def test_symintlist_error_with_overload(self):
        x = torch.randn(1)
        self._test_raises_str_typeerror(lambda arg: x.view(arg))

    # 测试函数，测试 torch._C._nn.pad 函数对 str 类型参数是否会抛出异常
    def test_intlist_error_with_overload(self):
        x = torch.randn(1)
        self._test_raises_str_typeerror(lambda arg: torch._C._nn.pad(x, arg))

    #
    # optional float list
    #

    # 测试函数，使用 module 对象测试 optional float list
    def do_test_optional_floatlist_with_module(self, module):
        values = torch
    def test_optional_floatlist(self):
        # 使用 FloatListWrapperModule 测试可选的浮点数列表处理
        self.do_test_optional_floatlist_with_module(FloatListWrapperModule())
        # 使用 torch.jit.script 将 FloatListWrapperModule 脚本化后测试可选的浮点数列表处理
        self.do_test_optional_floatlist_with_module(torch.jit.script(FloatListWrapperModule()))

        # 跟踪传入 None 的情况并返回结果
        traced_none = self.trace_optional_floatlist(None)
        # 跟踪传入 [5.1, 4.1] 的情况并返回结果
        traced_list = self.trace_optional_floatlist([5.1, 4.1])

        # 定义一个伪模块，用于处理特定情况：传入 None 和 [5.1, 4.1]
        def fake_module(values, const):
            if const is None:
                return traced_none(values)
            if const == [5.1, 4.1]:
                return traced_list(values)
            # 如果传入了非预期的参数，则抛出异常
            raise Exception("Invalid argument")  # noqa: TRY002

        # 使用 fake_module 测试可选的浮点数列表处理
        self.do_test_optional_floatlist_with_module(fake_module)

    def test_optional_floatlist_invalid(self):
        # 测试 FloatListWrapperModule 处理不合法的参数时是否会抛出 TypeError 异常
        with self.assertRaisesRegex(TypeError, "must be tuple of floats, not list"):
            FloatListWrapperModule()(torch.zeros(1), ["hi"])

        # 测试 torch.jit.script(FloatListWrapperModule()) 处理不合法的参数时是否会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "value of type .* instead found type"):
            torch.jit.script(FloatListWrapperModule())(torch.zeros(1), ["hi"])

        # 测试 FloatListWrapperModule 处理不合法的参数时是否会抛出 TypeError 异常
        with self.assertRaisesRegex(TypeError, "must be .* Tensor"):
            FloatListWrapperModule()(torch.zeros(1), torch.zeros(1))

        # 测试 torch.jit.script(FloatListWrapperModule()) 处理不合法的参数时是否会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "value of type .* instead found type"):
            torch.jit.script(FloatListWrapperModule())(torch.zeros(1), torch.zeros(1))

    #
    # optional int list
    #

    def do_test_optional_intlist_with_module(self, module):
        # 创建一个包含整数 [1, 2] 的张量
        values = torch.tensor([1, 2], dtype=torch.int)

        # 测试 module 处理传入 None 的情况
        returned = module(values, None)
        self.assertEqual(values, returned)
        # 确保 values 和 returned 是同一对象，表明运算符看到了 nullopt
        values[0] = 3
        self.assertEqual(values, returned)

        # 测试 module 处理传入 [5, 4] 的情况
        returned = module(values, [5, 4])
        self.assertEqual(values, torch.tensor([3, 2], dtype=torch.int))
        self.assertEqual(returned, torch.tensor([8, 6], dtype=torch.int))

    def trace_optional_intlist(self, const):
        # 创建一个包装器函数，用于跟踪调用 torch._C._nn._test_optional_intlist 处理整数列表
        def wrapper(values):
            return torch._C._nn._test_optional_intlist(values, const)
        return torch.jit.trace(wrapper, torch.tensor([1, 2], dtype=torch.int))

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    # 测试可选的整数列表情况
    def test_optional_intlist(self):
        # 使用普通模块进行测试
        self.do_test_optional_intlist_with_module(IntListWrapperModule())
        # 使用 Torch 脚本化模块进行测试
        self.do_test_optional_intlist_with_module(torch.jit.script(IntListWrapperModule()))

        # 跟踪 None 参数的情况
        traced_none = self.trace_optional_intlist(None)
        # 跟踪 [5, 4] 参数的情况
        traced_list = self.trace_optional_intlist([5, 4])

        # 这不是真正的模块，只是让我们使用两个跟踪函数来处理传递 None 和 [5, 4] 的特定情况。
        def fake_module(values, const):
            if const is None:
                return traced_none(values)
            if const == [5, 4]:
                return traced_list(values)
            # 抛出异常，表示参数无效
            raise Exception("Invalid argument")  # noqa: TRY002

        # 使用 fake_module 进行测试
        self.do_test_optional_intlist_with_module(fake_module)

    # 测试不合法的可选整数列表情况
    def test_optional_intlist_invalid(self):
        # 测试期望是整数列表但传递了浮点数列表的情况
        with self.assertRaisesRegex(TypeError, "must be .* but found"):
            IntListWrapperModule()(torch.zeros(1), [0.5])

        # 使用 Torch 脚本化模块测试期望是整数列表但传递了浮点数列表的情况
        with self.assertRaisesRegex(RuntimeError, "value of type .* instead found type"):
            torch.jit.script(IntListWrapperModule())(torch.zeros(1), [0.5])

        # 测试期望是整数 Tensor 但传递了浮点数 Tensor 的情况
        with self.assertRaisesRegex(TypeError, "must be .* Tensor"):
            IntListWrapperModule()(torch.zeros(1), torch.zeros(1))

        # 使用 Torch 脚本化模块测试期望是整数 Tensor 但传递了浮点数 Tensor 的情况
        with self.assertRaisesRegex(RuntimeError, "value of type .* instead found type"):
            torch.jit.script(IntListWrapperModule())(torch.zeros(1), torch.zeros(1))

    #
    # optional filled int list
    #

    # 使用模块进行测试，期望填充的整数列表情况
    def do_test_optional_filled_intlist_with_module(self, module):
        # 创建整数张量 [1, 2]
        values = torch.tensor([1, 2], dtype=torch.int)

        # 测试传递 None 参数的情况
        returned = module(values, None)
        self.assertEqual(values, returned)
        # 确保它是一个别名，表示操作符看到了一个 nullopt。
        values[0] = 3
        self.assertEqual(values, returned)

        # 测试传递 10 参数的情况
        returned = module(values, 10)
        self.assertEqual(values, torch.tensor([3, 2], dtype=torch.int))
        self.assertEqual(returned, torch.tensor([13, 12], dtype=torch.int))

    # 跟踪填充的整数列表情况
    def trace_optional_filled_intlist(self, const):
        def wrapper(values):
            return torch._C._nn._test_optional_filled_intlist(values, const)
        return torch.jit.trace(wrapper, torch.tensor([1, 2], dtype=torch.int))

    # 如果是 TorchDynamo 环境，则跳过该测试
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    # 定义一个测试方法，用于测试具有可选填充整数列表的函数
    def test_optional_filled_intlist(self):

        # 定义内部函数 f，接受一个整数 n 作为参数
        def f(n: int):
            # 调用 torch._C._nn._test_optional_filled_intlist 函数，传入一个包含两个整数的张量和一个元组 (n, n)
            x = torch._C._nn._test_optional_filled_intlist(torch.tensor([1, 1], dtype=torch.int), (n, n))
            # 调用 torch._C._nn._test_optional_filled_intlist 函数，传入一个包含一个整数的张量和整数 n
            y = torch._C._nn._test_optional_filled_intlist(torch.tensor([1, 1], dtype=torch.int), n)
            return x, y

        # 在 eager 模式下运行 f 函数，返回的两个结果应相等
        returned = f(10)
        self.assertEqual(returned[0], returned[1])

        # 将函数 f 脚本化
        s = torch.jit.script(f)
        # 在脚本化的函数下运行，返回的两个结果应相等
        returned = s(10)
        self.assertEqual(returned[0], returned[1])

        # 调用 self.trace_optional_filled_intlist 方法，传入 None，获得 traced_none 对象
        traced_none = self.trace_optional_filled_intlist(None)
        # 调用 self.trace_optional_filled_intlist 方法，传入整数 10，获得 traced_int 对象
        traced_int = self.trace_optional_filled_intlist(10)

        # 定义一个假的模块函数 fake_module，用于处理传入 None 和整数 10 的特定情况
        def fake_module(values, const):
            if const is None:
                return traced_none(values)
            if const == 10:
                return traced_int(values)
            # 如果传入的 const 不是 None 也不是 10，则抛出异常
            raise Exception("Invalid argument")  # noqa: TRY002

        # 使用 fake_module 方法来测试处理可选填充整数列表的情况
        self.do_test_optional_filled_intlist_with_module(fake_module)

    # 定义一个测试方法，用于测试字符串默认值的情况
    def test_string_defaults(self):
        # 创建一个随机张量 dummy
        dummy = torch.rand(1)
        # 获取 torch._C._nn._test_string_default 函数的引用，并调用它，传入 dummy
        fn = torch._C._nn._test_string_default
        fn(dummy)

        # 测试当传入空字符串时，是否会抛出包含 "A" 的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "A"):
            fn(dummy, a="")

        # 测试当传入空字符串时，是否会抛出包含 "B" 的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "B"):
            fn(dummy, b="")

        # 定义函数 f，接受一个参数 x，调用 torch._C._nn._test_string_default 函数，传入 x
        def f(x):
            torch._C._nn._test_string_default(x)
        # 将函数 f 脚本化
        scripted_fn = torch.jit.script(f)
        # 在脚本化的函数下运行，传入 dummy 作为参数
        scripted_fn(dummy)
# 如果当前模块被直接执行（而不是被导入到其他模块中），则执行以下代码
if __name__ == '__main__':
    # 调用名为 run_tests 的函数，通常用于执行测试用例
    run_tests()
```