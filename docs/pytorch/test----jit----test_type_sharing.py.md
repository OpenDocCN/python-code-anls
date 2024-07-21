# `.\pytorch\test\jit\test_type_sharing.py`

```py
# Owner(s): ["oncall: jit"]

# 引入必要的库
import io
import os
import sys

import torch

# 将测试目录中的辅助文件变为可导入状态
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 导入测试所需的函数和类
from torch.testing._internal.common_utils import suppress_warnings
from torch.testing._internal.jit_utils import JitTestCase

# 如果此脚本作为主程序运行，抛出运行时错误提示信息
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类，继承自JitTestCase
class TestTypeSharing(JitTestCase):

    # 辅助函数：验证两个模块是否具有相同的类型
    def assertSameType(self, m1, m2):
        if not isinstance(m1, torch.jit.ScriptModule):
            m1 = torch.jit.script(m1)
        if not isinstance(m2, torch.jit.ScriptModule):
            m2 = torch.jit.script(m2)
        self.assertEqual(m1._c._type(), m2._c._type())

    # 辅助函数：验证两个模块是否具有不同的类型
    def assertDifferentType(self, m1, m2):
        if not isinstance(m1, torch.jit.ScriptModule):
            m1 = torch.jit.script(m1)
        if not isinstance(m2, torch.jit.ScriptModule):
            m2 = torch.jit.script(m2)
        self.assertNotEqual(m1._c._type(), m2._c._type())

    # 测试函数：验证基本情况下模块类型共享
    def test_basic(self):
        # 定义简单的模块类
        class M(torch.nn.Module):
            def __init__(self, a, b, c):
                super().__init__()
                self.a = a
                self.b = b
                self.c = c

            def forward(self, x):
                return x

        # 创建输入张量
        a = torch.rand(2, 3)
        b = torch.rand(2, 3)
        c = torch.rand(2, 3)
        # 创建两个相同参数的模块实例
        m1 = M(a, b, c)
        m2 = M(a, b, c)
        # 验证两个模块具有相同的类型
        self.assertSameType(m1, m2)

    # 测试函数：验证属性值不同但类型依然共享
    def test_diff_attr_values(self):
        """
        Types should be shared even if attribute values differ
        """
        # 定义带有不同属性值的模块类
        class M(torch.nn.Module):
            def __init__(self, a, b, c):
                super().__init__()
                self.a = a
                self.b = b
                self.c = c

            def forward(self, x):
                return x

        # 创建输入张量
        a = torch.rand(2, 3)
        b = torch.rand(2, 3)
        c = torch.rand(2, 3)
        # 创建两个属性值不同的模块实例
        m1 = M(a, b, c)
        m2 = M(a * 2, b * 3, c * 4)
        # 验证两个模块具有相同的类型
        self.assertSameType(m1, m2)

    # 测试函数：验证常量值相同则类型共享，不同则类型不同
    def test_constants(self):
        """
        Types should be shared for identical constant values, and different for different constant values
        """
        # 定义带有常量的模块类
        class M(torch.nn.Module):
            __constants__ = ["const"]

            def __init__(self, attr, const):
                super().__init__()
                self.attr = attr
                self.const = const

            def forward(self):
                return self.const

        # 创建输入张量
        attr = torch.rand(2, 3)
        # 创建两个常量相同的模块实例
        m1 = M(attr, 1)
        m2 = M(attr, 1)
        # 验证两个模块具有相同的类型
        self.assertSameType(m1, m2)

        # 创建一个常量不同的模块实例
        m3 = M(attr, 2)
        # 验证这两个模块具有不同的类型
        self.assertDifferentType(m1, m3)
    def test_linear(self):
        """
        Simple example with a real nn Module
        """
        # 创建线性模型 a, b, c
        a = torch.nn.Linear(5, 5)
        b = torch.nn.Linear(5, 5)
        c = torch.nn.Linear(10, 10)
        # 将模型 a, b, c 转换为 Torch Script
        a = torch.jit.script(a)
        b = torch.jit.script(b)
        c = torch.jit.script(c)

        # 断言 a 和 b 的类型相同
        self.assertSameType(a, b)
        # 断言 a 和 c 的类型不同
        self.assertDifferentType(a, c)

    def test_submodules(self):
        """
        If submodules differ, the types should differ.
        """

        class M(torch.nn.Module):
            def __init__(self, in1, out1, in2, out2):
                super().__init__()
                # 创建两个子模块，每个模块包含一个线性层
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)

            def forward(self, x):
                x = self.submod1(x)
                x = self.submod2(x)
                return x

        # 创建模型实例 a 和 b，具有相同的子模块
        a = M(1, 1, 2, 2)
        b = M(1, 1, 2, 2)
        # 断言 a 和 b 的类型相同
        self.assertSameType(a, b)
        # 断言 a 和 c 的第一个子模块的类型相同，即子模块1
        self.assertSameType(a.submod1, b.submod1)
        # 创建模型实例 c，与 a 的第一个子模块类型不同
        c = M(2, 2, 2, 2)
        # 断言 a 和 c 的类型不同
        self.assertDifferentType(a, c)

        # 断言 b 的第二个子模块和 c 的第一个子模块类型相同
        self.assertSameType(b.submod2, c.submod1)
        # 断言 a 的第一个子模块和 b 的第二个子模块类型不同
        self.assertDifferentType(a.submod1, b.submod2)

    def test_param_vs_attribute(self):
        """
        The same module with an `foo` as a parameter vs. attribute shouldn't
        share types
        """

        class M(torch.nn.Module):
            def __init__(self, foo):
                super().__init__()
                # 设置模块的属性为 foo
                self.foo = foo

            def forward(self, x):
                return x + self.foo

        # 创建一个参数化张量和一个属性张量
        as_param = torch.nn.Parameter(torch.ones(2, 2))
        as_attr = torch.ones(2, 2)
        # 使用参数化张量创建模型 param_mod
        param_mod = M(as_param)
        # 使用属性张量创建模型 attr_mod
        attr_mod = M(as_attr)
        # 断言 attr_mod 和 param_mod 的类型不同
        self.assertDifferentType(attr_mod, param_mod)

    def test_same_but_different_classes(self):
        """
        Even if everything about the module is the same, different originating
        classes should prevent type sharing.
        """

        class A(torch.nn.Module):
            __constants__ = ["const"]

            def __init__(self, in1, out1, in2, out2):
                super().__init__()
                # 创建两个子模块，每个模块包含一个线性层，并且定义了一个常量
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)
                self.const = 5

            def forward(self, x):
                x = self.submod1(x)
                x = self.submod2(x)
                return x * self.const

        class B(torch.nn.Module):
            __constants__ = ["const"]

            def __init__(self, in1, out1, in2, out2):
                super().__init__()
                # 创建两个子模块，每个模块包含一个线性层，并且定义了一个常量
                self.submod1 = torch.nn.Linear(in1, out1)
                self.submod2 = torch.nn.Linear(in2, out2)
                self.const = 5

            def forward(self, x):
                x = self.submod1(x)
                x = self.submod2(x)
                return x * self.const

        # 创建两个不同的模型 A 和 B，它们具有相同的结构和常量
        a = A(1, 1, 2, 2)
        b = B(1, 1, 2, 2)
        # 断言 a 和 b 的类型不同
        self.assertDifferentType(a, b)
    def test_mutate_attr_value(self):
        """
        Mutating the value of an attribute should not change type sharing
        """

        class M(torch.nn.Module):
            def __init__(self, in1, out1, in2, out2):
                super().__init__()
                # 定义神经网络模块的子模块，线性层1
                self.submod1 = torch.nn.Linear(in1, out1)
                # 定义神经网络模块的子模块，线性层2
                self.submod2 = torch.nn.Linear(in2, out2)
                # 定义一个名为foo的属性，用torch.ones初始化
                self.foo = torch.ones(in1, in1)

            def forward(self, x):
                # 在前向传播中使用第一个子模块
                x = self.submod1(x)
                # 继续使用第二个子模块
                x = self.submod2(x)
                # 返回处理结果和属性foo的加和
                return x + self.foo

        # 创建两个模型实例a和b，分别使用不同的参数初始化
        a = M(1, 1, 2, 2)
        b = M(1, 1, 2, 2)
        # 修改a的属性foo为全1的2x2张量
        a.foo = torch.ones(2, 2)
        # 修改b的属性foo为随机初始化的2x2张量
        b.foo = torch.rand(2, 2)
        # 断言a和b的类型相同
        self.assertSameType(a, b)

    def test_assign_python_attr(self):
        """
        Assigning a new (python-only) attribute should not change type sharing
        """

        class M(torch.nn.Module):
            def __init__(self, in1, out1, in2, out2):
                super().__init__()
                # 定义神经网络模块的子模块，线性层1
                self.submod1 = torch.nn.Linear(in1, out1)
                # 定义神经网络模块的子模块，线性层2
                self.submod2 = torch.nn.Linear(in2, out2)
                # 定义一个名为foo的属性，用torch.ones初始化
                self.foo = torch.ones(in1, in1)

            def forward(self, x):
                # 在前向传播中使用第一个子模块
                x = self.submod1(x)
                # 继续使用第二个子模块
                x = self.submod2(x)
                # 返回处理结果和属性foo的加和
                return x + self.foo

        # 将模型M用torch.jit.script()转换为脚本模型，保持类型不变
        a = torch.jit.script(M(1, 1, 2, 2))
        b = torch.jit.script(M(1, 1, 2, 2))
        # 给模型a添加一个新的属性new_attr
        a.new_attr = "foo bar baz"
        # 断言a和b的类型相同
        self.assertSameType(a, b)

        # 但是如果我们在调用script()之前就给模型赋予属性，
        # 类型应该不同，因为new_attr应该被转换为脚本属性
        a = M(1, 1, 2, 2)
        b = M(1, 1, 2, 2)
        # 在调用script()之前给模型a添加一个新的属性new_attr
        a.new_attr = "foo bar baz"
        # 断言a和b的类型不同
        self.assertDifferentType(a, b)

    def test_failed_attribute_compilation(self):
        """
        Attributes whose type cannot be inferred should fail cleanly with nice hints
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 将一个无法转换为TorchScript的Python类型赋值给属性foo
                self.foo = object

            def forward(self):
                # 尝试在前向传播中使用属性foo
                return self.foo

        # 创建模型实例m
        m = M()
        # 断言运行时错误中包含特定的错误消息，指出无法转换Python类型
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "failed to convert Python type", "self.foo"
        ):
            # 尝试将模型m转换为脚本模型，应该会抛出错误
            torch.jit.script(m)
    def test_script_function_attribute_different(self):
        """
        Different functions passed in should lead to different types
        """

        # 定义一个使用 torch.jit.script 装饰器修饰的脚本函数 fn1，计算输入的两倍
        @torch.jit.script
        def fn1(x):
            return x + x

        # 定义另一个使用 torch.jit.script 装饰器修饰的脚本函数 fn2，计算输入的零
        @torch.jit.script
        def fn2(x):
            return x - x

        # 定义一个继承自 torch.nn.Module 的模块 M，其构造函数接受一个函数 fn 并保存为属性
        class M(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            # 实现前向传播函数，调用保存的函数 fn 处理输入 x
            def forward(self, x):
                return self.fn(x)

        # 创建两个 M 类的实例，分别使用 fn1 和 fn2
        fn1_mod = M(fn1)
        fn2_mod = M(fn2)

        # 断言 fn1_mod 和 fn2_mod 的类型不同
        self.assertDifferentType(fn1_mod, fn2_mod)

    def test_builtin_function_same(self):
        # 定义一个继承自 torch.nn.Module 的模块 Caller，其构造函数接受一个内置函数 fn 并保存为属性
        class Caller(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            # 实现前向传播函数，调用保存的函数 fn 处理输入 input
            def forward(self, input):
                return self.fn(input, input)

        # 创建两个 Caller 类的实例，均使用 torch.add 函数
        c1 = Caller(torch.add)
        c2 = Caller(torch.add)

        # 断言 c1 和 c2 的类型相同
        self.assertSameType(c1, c2)

    def test_builtin_function_different(self):
        # 定义一个继承自 torch.nn.Module 的模块 Caller，其构造函数接受一个内置函数 fn 并保存为属性
        class Caller(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            # 实现前向传播函数，调用保存的函数 fn 处理输入 input
            def forward(self, input):
                return self.fn(input, input)

        # 创建两个 Caller 类的实例，分别使用 torch.add 和 torch.sub 函数
        c1 = Caller(torch.add)
        c2 = Caller(torch.sub)

        # 断言 c1 和 c2 的类型不同
        self.assertDifferentType(c1, c2)

    def test_script_function_attribute_same(self):
        """
        Same functions passed in should lead to same types
        """

        # 定义一个使用 torch.jit.script 装饰器修饰的脚本函数 fn，计算输入的两倍
        @torch.jit.script
        def fn(x):
            return x + x

        # 定义一个继承自 torch.nn.Module 的模块 M，其构造函数接受一个函数 fn 并保存为属性
        class M(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            # 实现前向传播函数，调用保存的函数 fn 处理输入 x
            def forward(self, x):
                return self.fn(x)

        # 创建两个 M 类的实例，均使用相同的 fn 函数
        fn1_mod = M(fn)
        fn2_mod = M(fn)

        # 断言 fn1_mod 和 fn2_mod 的类型相同
        self.assertSameType(fn1_mod, fn2_mod)

    def test_python_function_attribute_different(self):
        """
        Different functions passed in should lead to different types
        """

        # 定义一个简单的 Python 函数 fn1，计算输入的两倍
        def fn1(x):
            return x + x

        # 定义另一个简单的 Python 函数 fn2，计算输入的零
        def fn2(x):
            return x - x

        # 定义一个继承自 torch.nn.Module 的模块 M，其构造函数接受一个函数 fn 并保存为属性
        class M(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            # 实现前向传播函数，调用保存的函数 fn 处理输入 x
            def forward(self, x):
                return self.fn(x)

        # 创建两个 M 类的实例，分别使用 fn1 和 fn2
        fn1_mod = M(fn1)
        fn2_mod = M(fn2)

        # 断言 fn1_mod 和 fn2_mod 的类型不同
        self.assertDifferentType(fn1_mod, fn2_mod)

    def test_python_function_attribute_same(self):
        """
        Same functions passed in should lead to same types
        """

        # 定义一个简单的 Python 函数 fn，计算输入的两倍
        def fn(x):
            return x + x

        # 定义一个继承自 torch.nn.Module 的模块 M，其构造函数接受一个函数 fn 并保存为属性
        class M(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            # 实现前向传播函数，调用保存的函数 fn 处理输入 x
            def forward(self, x):
                return self.fn(x)

        # 创建两个 M 类的实例，均使用相同的 fn 函数
        fn1_mod = M(fn)
        fn2_mod = M(fn)

        # 断言 fn1_mod 和 fn2_mod 的类型相同
        self.assertSameType(fn1_mod, fn2_mod)

    @suppress_warnings
    def test_loaded_modules_work(self):
        """
        测试加载的模块是否正常工作。
        """

        class AB(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 1
                self.b = 1

            def forward(self):
                # 返回 a 和 b 的和
                return self.a + self.b

        class A(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 1

            def forward(self):
                # 返回属性 a 的值
                return self.a

        class Wrapper(torch.nn.Module):
            def __init__(self, sub):
                super().__init__()
                self.sub = sub

            def forward(self):
                # 调用包装的子模块的 forward 方法
                return self.sub()

        def package(x):
            # 创建一个字节流缓冲区
            buffer = io.BytesIO()
            # 将模块 x 脚本化并保存到缓冲区
            torch.jit.save(torch.jit.script(x), buffer)
            buffer.seek(0)
            # 从缓冲区加载脚本化的模块并封装成 Wrapper 类型的模块
            return torch.jit.script(Wrapper(torch.jit.load(buffer)))

        # 脚本化 AB 模块并包装成 Wrapper
        a = package(AB())
        # 调用 a 的 forward 方法
        a()
        # 脚本化 A 模块并包装成 Wrapper
        b = package(A())
        # 调用 b 的 forward 方法
        b()
    def test_module_dict_same_type_different_name(self):
        """
        We should be able to differentiate between two ModuleDict instances
        that have different keys but the same value types.
        """

        class A(torch.nn.Module):
            def forward(self, x):
                return x

        class Foo(torch.nn.Module):
            def __init__(self, s):
                super().__init__()
                # 初始化 ModuleDict 对象，使用传入的字典 s
                self.dict = torch.nn.ModuleDict(s)

            def forward(self, x):
                return x

        # 创建三个 Foo 类的实例，每个实例包含不同的 ModuleDict 键值对
        a = Foo({"foo": A()})
        b = Foo({"bar": A()})
        c = Foo({"bar": A()})
        # 断言 a 和 b 的类型不同
        self.assertDifferentType(a, b)
        # 断言 b 和 c 的类型相同
        self.assertSameType(b, c)

    def test_type_sharing_define_in_init(self):
        """
        Tests that types between instances of a ScriptModule
        subclass that defines methods in its __init__ are not
        shared.
        """

        class A(torch.jit.ScriptModule):
            def __init__(self, val):
                super().__init__()
                # 在初始化时定义一个脚本方法，使用传入的 val
                self.define(
                    f"""
                def forward(self) -> int:
                    return {val}
                """
                )

        # 创建两个 A 类的实例，每个实例使用不同的 val 值
        one = A(1)
        two = A(2)

        # 断言两个实例的执行结果分别为 1 和 2
        self.assertEqual(one(), 1)
        self.assertEqual(two(), 2)

    def test_type_sharing_disabled(self):
        """
        Test that type sharing can be disabled.
        """

        class A(torch.nn.Module):
            def __init__(self, sub):
                super().__init__()
                # 初始化时传入一个子模块 sub
                self.sub = sub

            def forward(self, x):
                return x

        class B(torch.nn.Module):
            def forward(self, x):
                return x

        # 创建两个 A 类的实例，每个实例中包含 A 类的实例 sub，其又包含一个 B 类的实例
        top1 = A(A(B()))
        top2 = A(A(B()))

        # 使用 torch.jit._recursive.create_script_module 创建两个脚本模块实例，
        # 分别传入不同的 top-level 模块 top1 和 top2，并禁用类型共享
        top1_s = torch.jit._recursive.create_script_module(
            top1,
            torch.jit._recursive.infer_methods_to_compile,
            share_types=False,
        )
        top2_s = torch.jit._recursive.create_script_module(
            top2,
            torch.jit._recursive.infer_methods_to_compile,
            share_types=False,
        )

        # 断言两个脚本模块实例的类型不同
        self.assertDifferentType(top1_s, top2_s)
        # 断言 top1_s 和 top1_s.sub 的类型不同
        self.assertDifferentType(top1_s, top1_s.sub)
        # 断言 top1_s 和 top2_s.sub 的类型不同
        self.assertDifferentType(top1_s, top2_s.sub)
        # 断言 top2_s 和 top2_s.sub 的类型不同
        self.assertDifferentType(top2_s, top2_s.sub)
    def test_type_shared_ignored_attributes(self):
        """
        Test that types are shared if the exclusion of their
        ignored attributes makes them equal.
        """

        class A(torch.nn.Module):
            __jit_ignored_attributes__ = ["a"]  # 定义被忽略的属性列表，这些属性不会影响类型的比较

            def __init__(self, a, b):
                super().__init__()
                self.a = a  # 初始化实例属性 a
                self.b = b  # 初始化实例属性 b

            def forward(self, x):
                return x

        a_with_linear = A(torch.nn.Linear(5, 5), 5)  # 创建一个 A 类型的实例，a 是 torch.nn.Linear 对象，b 是整数 5
        a_with_string = A("string", 10)  # 创建另一个 A 类型的实例，a 是字符串 "string"，b 是整数 10

        # Both should have the same type because the attribute
        # that differs in type is ignored and the common attribute
        # has the same type.
        self.assertSameType(a_with_linear, a_with_string)  # 断言两个实例的类型相同

    def test_type_not_shared_ignored_attributes(self):
        """
        Test that types are not shared if the exclusion of their
        ignored attributes makes them not equal.
        """

        class A(torch.nn.Module):
            __jit_ignored_attributes__ = ["a"]  # 定义被忽略的属性列表，这些属性不会影响类型的比较

            def __init__(self, a, b, c):
                super().__init__()
                self.a = a  # 初始化实例属性 a
                self.b = b  # 初始化实例属性 b
                self.c = c  # 初始化实例属性 c

            def forward(self, x):
                return x

        mod = A(torch.nn.Linear(5, 5), 5, "string")  # 创建一个 A 类型的实例，a 是 torch.nn.Linear 对象，b 是整数 5，c 是字符串 "string"
        s1 = torch.jit.script(mod)  # 对实例进行脚本化，生成 s1
        A.__jit_ignored_attributes__ = ["a", "b"]  # 修改类 A 的被忽略属性列表
        s2 = torch.jit.script(mod)  # 对同一个实例再次进行脚本化，生成 s2

        # The types of s1 and s2 should differ. Although they are instances
        # of A, __jit_ignored_attributes__ was modified before scripting s2,
        # so the set of ignored attributes is different between s1 and s2.
        self.assertDifferentType(s1, s2)  # 断言 s1 和 s2 的类型不同
```