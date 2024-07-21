# `.\pytorch\test\jit\test_class_type.py`

```py
# Owner(s): ["oncall: jit"]

# 导入必要的库和模块
import io  # 输入输出操作模块
import os  # 系统相关操作模块
import sys  # 系统相关操作模块
import unittest  # 单元测试框架
from typing import Any  # 类型提示模块

import torch  # PyTorch 主库
import torch.nn as nn  # PyTorch 神经网络模块
from torch.testing import FileCheck  # PyTorch 测试工具

# 将 test/ 目录下的辅助文件设为可导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from typing import Dict, Iterable, List, Optional, Tuple  # 类型提示模块

import torch.testing._internal.jit_utils  # PyTorch 内部测试工具
from torch.testing._internal.common_utils import IS_SANDCASTLE, skipIfTorchDynamo  # PyTorch 内部常用工具
from torch.testing._internal.jit_utils import JitTestCase, make_global  # PyTorch JIT 测试工具

# 如果以主程序运行则抛出错误提示
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestClassType(JitTestCase):
    def test_reference_semantics(self):
        """
        Test that modifications made to a class instance in TorchScript
        are visible in eager.
        """

        class Foo:
            def __init__(self, a: int):
                self.a = a

            def set_a(self, value: int):
                self.a = value

            def get_a(self) -> int:
                return self.a

            @property
            def attr(self):
                return self.a

        make_global(Foo)  # 将 Foo 类设为全局变量，见 [local resolution in python]

        def test_fn(obj: Foo):
            obj.set_a(2)

        scripted_fn = torch.jit.script(test_fn)  # 对 test_fn 函数进行 TorchScript 编译
        obj = torch.jit.script(Foo(1))  # 对 Foo 类进行 TorchScript 编译并实例化
        self.assertEqual(obj.get_a(), 1)  # 断言初始状态下 get_a 方法返回值为 1
        self.assertEqual(obj.attr, 1)  # 断言初始状态下 attr 属性值为 1

        scripted_fn(obj)  # 执行 TorchScript 编译后的函数对 obj 进行修改

        self.assertEqual(obj.get_a(), 2)  # 断言修改后 get_a 方法返回值为 2
        self.assertEqual(obj.attr, 2)  # 断言修改后 attr 属性值为 2

    def test_get_with_method(self):
        class FooTest:
            def __init__(self, x):
                self.foo = x

            def getFooTest(self):
                return self.foo

        def fn(x):
            foo = FooTest(x)
            return foo.getFooTest()

        input = torch.ones(2, 3)
        self.assertEqual(fn(input), input)  # 断言函数 fn 返回值与输入相同

    def test_get_attr(self):
        class FooTest:  # noqa: B903
            def __init__(self, x):
                self.foo = x

        @torch.jit.script
        def fn(x):
            foo = FooTest(x)
            return foo.foo

        input = torch.ones(2, 3)
        self.assertEqual(fn(input), input)  # 断言 TorchScript 函数 fn 返回值与输入相同

    def test_in(self):
        class FooTest:  # noqa: B903
            def __init__(self):
                pass

            def __contains__(self, key: str) -> bool:
                return key == "hi"

        @torch.jit.script
        def fn():
            foo = FooTest()
            return "hi" in foo, "no" in foo

        self.assertEqual(fn(), (True, False))  # 断言 TorchScript 函数 fn 返回值为 (True, False)
    def test_set_attr_in_method(self):
        # 定义一个测试方法，测试属性设置在方法中的情况
        class FooTest:
            def __init__(self, x: int) -> None:
                self.foo = x  # 初始化实例属性 foo

            def incFooTest(self, y: int) -> None:
                self.foo = self.foo + y  # 在方法中更新实例属性 foo 的值

        @torch.jit.script
        def fn(x: int) -> int:
            foo = FooTest(x)  # 创建 FooTest 类的实例
            foo.incFooTest(2)  # 调用方法修改 foo 属性值
            return foo.foo  # 返回修改后的 foo 属性值

        self.assertEqual(fn(1), 3)  # 断言 foo.foo 的最终值为 3

    def test_set_attr_type_mismatch(self):
        # 测试属性赋值时类型不匹配的情况
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Wrong type for attribute assignment", "self.foo = 10"
        ):

            @torch.jit.script
            class FooTest:
                def __init__(self, x):
                    self.foo = x  # 初始化实例属性 foo
                    self.foo = 10  # 尝试用整数赋值给 Tensor 类型的属性，应该引发错误

    def test_get_attr_not_initialized(self):
        # 测试访问未初始化的属性的情况
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "object has no attribute or method", "self.asdf"
        ):

            @torch.jit.script
            class FooTest:
                def __init__(self, x):
                    self.foo = x  # 初始化实例属性 foo

                def get_non_initialized(self):
                    return self.asdf  # 尝试访问未定义的属性 asdf，应该引发错误

    def test_set_attr_non_initialized(self):
        # 测试给未初始化的属性赋值的情况
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.bar = y"
        ):

            @torch.jit.script
            class FooTest:
                def __init__(self, x):
                    self.foo = x  # 初始化实例属性 foo

                def set_non_initialized(self, y):
                    self.bar = y  # 尝试给未定义的属性 bar 赋值，应该引发错误

    def test_schema_human_readable(self):
        """
        确保模式参数在错误信息中显示为人类可读的形式，例如应该显示为 "nearest" 而不是八进制形式
        """
        with self.assertRaisesRegexWithHighlight(RuntimeError, "nearest", ""):

            @torch.jit.script
            def FooTest(x):
                return torch.nn.functional.interpolate(x, "bad")  # 调用函数时传递错误的参数值

    def test_type_annotations(self):
        # 测试类型注解不匹配的情况
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Expected a value of type 'bool", ""
        ):

            @torch.jit.script  # noqa: B903
            class FooTest:  # noqa: B903
                def __init__(self, x: bool) -> None:
                    self.foo = x  # 初始化实例属性 foo

            @torch.jit.script
            def fn(x):
                FooTest(x)  # 创建 FooTest 类的实例，传递错误类型的参数值

            fn(2)  # 调用函数时传递错误类型的参数值
    def test_conditional_set_attr(self):
        # 使用 assertRaisesRegexWithHighlight 来测试 RuntimeError，期望出现 "assignment cannot be in a control-flow block" 错误信息
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "assignment cannot be in a control-flow block", ""
        ):
            # 将类 FooTest 转换为 Torch 脚本
            @torch.jit.script
            class FooTest:
                def __init__(self, x):
                    # 如果条件满足，则设置实例属性 self.attr 为 x
                    if 1 == 1:
                        self.attr = x

    def test_class_type_as_param(self):
        # 定义类 FooTest，用于测试
        class FooTest:  # noqa: B903
            def __init__(self, x):
                # 设置实例属性 self.attr 为 x
                self.attr = x

        # 将 FooTest 类注册为全局对象，以便 Torch 脚本使用
        make_global(FooTest)  # see [local resolution in python]

        # 定义 Torch 脚本函数 fn，参数为 FooTest 类型的实例，返回其属性 attr
        @torch.jit.script
        def fn(foo: FooTest) -> torch.Tensor:
            return foo.attr

        # 定义 Torch 脚本函数 fn2，创建 FooTest 实例，调用 fn 函数返回属性值
        @torch.jit.script
        def fn2(x):
            foo = FooTest(x)
            return fn(foo)

        # 测试输入为全 1 的 Tensor，期望 fn2 输出与输入相等
        input = torch.ones(1)
        self.assertEqual(fn2(input), input)

    def test_out_of_order_methods(self):
        # 定义类 FooTest，测试构造函数中调用实例方法的情况
        class FooTest:
            def __init__(self, x):
                # 设置实例属性 self.x 为 x，并调用 get_stuff 方法
                self.x = x
                self.x = self.get_stuff(x)

            # 定义实例方法 get_stuff，返回 self.x + y 的结果
            def get_stuff(self, y):
                return self.x + y

        # 定义 Torch 脚本函数 fn，创建 FooTest 实例，并返回其属性 x
        @torch.jit.script
        def fn(x):
            f = FooTest(x)
            return f.x

        # 测试输入为全 1 的 Tensor，期望 fn 输出为输入的两倍
        input = torch.ones(1)
        self.assertEqual(fn(input), input + input)

    def test_save_load_with_classes(self):
        # 定义类 FooTest，包含获取属性值的方法 get_x
        class FooTest:
            def __init__(self, x):
                self.x = x

            def get_x(self):
                return self.x

        # 定义 Torch 脚本模块 MyMod，包含前向传播方法 forward，使用 FooTest 类
        class MyMod(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                foo = FooTest(a)
                return foo.get_x()

        # 创建 MyMod 实例 m
        m = MyMod()

        # 创建一个字节流缓冲区，将 MyMod 实例 m 保存到其中
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)

        # 由于类被全局注册，清除 JIT 注册表以便加载新模型
        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)

        # 测试输入为随机生成的 2x3 Tensor，期望模型输出与输入相等
        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(input, output)

    def test_save_load_with_classes_returned(self):
        # 定义类 FooTest，包含克隆方法 clone，用于返回新的 FooTest 实例
        class FooTest:
            def __init__(self, x):
                self.x = x

            def clone(self):
                clone = FooTest(self.x)
                return clone

        # 定义 Torch 脚本模块 MyMod，包含前向传播方法 forward，使用 FooTest 类
        class MyMod(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                foo = FooTest(a)
                foo_clone = foo.clone()
                return foo_clone.x

        # 创建 MyMod 实例 m
        m = MyMod()

        # 创建一个字节流缓冲区，将 MyMod 实例 m 保存到其中
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)

        # 清除类注册表以模拟加载新模型
        torch.testing._internal.jit_utils.clear_class_registry()

        # 从缓冲区加载模型 m_loaded
        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)

        # 测试输入为随机生成的 2x3 Tensor，期望模型输出与输入相等
        input = torch.rand(2, 3)
        output = m_loaded(input)
        self.assertEqual(input, output)
    def test_save_load_with_classes_nested(self):
        class FooNestedTest:  # 定义嵌套类 FooNestedTest
            def __init__(self, y):
                self.y = y  # 初始化实例变量 y

        class FooNestedTest2:  # 定义嵌套类 FooNestedTest2
            def __init__(self, y):
                self.y = y  # 初始化实例变量 y
                self.nested = FooNestedTest(y)  # 创建嵌套对象 FooNestedTest

        class FooTest:  # 定义主测试类 FooTest
            def __init__(self, x):
                self.class_attr = FooNestedTest(x)  # 创建 FooNestedTest 实例并赋值给 class_attr
                self.class_attr2 = FooNestedTest2(x)  # 创建 FooNestedTest2 实例并赋值给 class_attr2
                self.x = self.class_attr.y + self.class_attr2.y  # 计算属性 x

        class MyMod(torch.jit.ScriptModule):  # 定义 PyTorch 脚本模块 MyMod
            @torch.jit.script_method
            def forward(self, a):
                foo = FooTest(a)  # 创建 FooTest 实例 foo
                return foo.x  # 返回 foo 的属性 x

        m = MyMod()  # 创建 MyMod 实例 m

        buffer = io.BytesIO()  # 创建字节流对象 buffer
        torch.jit.save(m, buffer)  # 将模型 m 保存到 buffer 中

        # 由于类在全局注册，需要清除 JIT 注册表以模拟加载新模型
        torch.testing._internal.jit_utils.clear_class_registry()

        buffer.seek(0)  # 将 buffer 的指针移动到开头
        m_loaded = torch.jit.load(buffer)  # 从 buffer 中加载模型并赋值给 m_loaded

        input = torch.rand(2, 3)  # 创建输入张量 input
        output = m_loaded(input)  # 使用加载的模型 m_loaded 进行推理
        self.assertEqual(2 * input, output)  # 断言输出结果是否符合预期

    def test_python_interop(self):
        class Foo:  # 定义类 Foo
            def __init__(self, x, y):
                self.x = x  # 初始化实例变量 x
                self.y = y  # 初始化实例变量 y

        make_global(Foo)  # 将 Foo 类全局化，参考 [local resolution in python]

        @torch.jit.script
        def use_foo(foo: Foo) -> Foo:
            return foo  # 使用 Foo 类在 TorchScript 中进行操作

        # 从 Python 中创建 Foo 对象
        x = torch.ones(2, 3)  # 创建张量 x
        y = torch.zeros(2, 3)  # 创建张量 y
        f = Foo(x, y)  # 创建 Foo 类实例 f

        self.assertEqual(x, f.x)  # 断言 f 的属性 x 是否等于 x
        self.assertEqual(y, f.y)  # 断言 f 的属性 y 是否等于 y

        # 在 TorchScript 中传递对象
        f2 = use_foo(f)  # 调用 TorchScript 函数 use_foo

        self.assertEqual(x, f2.x)  # 断言 f2 的属性 x 是否等于 x
        self.assertEqual(y, f2.y)  # 断言 f2 的属性 y 是否等于 y

    def test_class_specialization(self):
        class Foo:  # 定义类 Foo
            def __init__(self, x, y):
                self.x = x  # 初始化实例变量 x
                self.y = y  # 初始化实例变量 y

        make_global(Foo)  # 将 Foo 类全局化，参考 [local resolution in python]

        def use_foo(foo: Foo, foo2: Foo, tup: Tuple[Foo, Foo]) -> torch.Tensor:
            a, b = tup  # 解包元组 tup
            return foo.x + foo2.y + a.x + b.y  # 返回计算结果

        # 从 Python 中创建 Foo 对象
        x = torch.ones(2, 3)  # 创建张量 x
        y = torch.zeros(2, 3)  # 创建张量 y
        f = Foo(x, y)  # 创建 Foo 类实例 f
        f2 = Foo(x * 2, y * 3)  # 创建另一个 Foo 类实例 f2
        f3 = Foo(x * 4, y * 4)  # 创建另一个 Foo 类实例 f3

        input = (f, f2, (f, f3))  # 创建输入元组 input
        sfoo = self.checkScript(use_foo, input)  # 使用 TorchScript 检查函数 use_foo
        graphstr = str(sfoo.graph_for(*input))  # 获取输入参数的计算图字符串表示
        FileCheck().check_count("prim::GetAttr", 4).run(graphstr)  # 检查计算图中 "prim::GetAttr" 的数量是否为 4

    def test_class_inheritance(self):
        @torch.jit.script
        class Base:
            def __init__(self):
                self.b = 2  # 初始化实例变量 b

            def two(self, x):
                return x + self.b  # 返回计算结果

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "does not support inheritance", ""
        ):
            @torch.jit.script
            class Derived(Base):
                def two(self, x):
                    return x + self.b + 2  # 返回计算结果
    def test_class_inheritance_implicit(self):
        """
        Test that inheritance is detected in
        implicit scripting codepaths (e.g. try_ann_to_type).
        """

        # 定义类 A
        class A:
            def __init__(self, t):
                self.t = t

            @staticmethod
            # 静态方法 f，接收一个 torch.Tensor 对象 a，并返回 A 类的实例
            def f(a: torch.Tensor):
                return A(a + 1)

        # 定义类 B，继承自 A
        class B(A):
            def __init__(self, t):
                self.t = t + 10

            @staticmethod
            # 静态方法 f，接收一个 torch.Tensor 对象 a，并返回 A 类的实例
            def f(a: torch.Tensor):
                return A(a + 1)

        # 创建 A 类的实例 x，传入一个包含数字 3 的 torch.Tensor 对象
        x = A(torch.tensor([3]))

        # 定义函数 fun，接收任意类型 x，并根据其类型调用相应的类方法 f
        def fun(x: Any):
            if isinstance(x, A):
                return A.f(x.t)
            else:
                return B.f(x.t)

        # 使用 assertRaisesRegexWithHighlight 上下文管理器测试 torch.jit.script(fun) 是否会抛出特定异常
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "object has no attribute or method", ""
        ):
            sc = torch.jit.script(fun)

    @skipIfTorchDynamo("Test does not work with TorchDynamo")
    @unittest.skipIf(IS_SANDCASTLE, "Importing like this doesn't work in fbcode")
    def test_imported_classes(self):
        # 导入需要的模块
        import jit._imported_class_test.bar
        import jit._imported_class_test.foo
        import jit._imported_class_test.very.very.nested

        # 定义一个继承自 torch.jit.ScriptModule 的子类 MyMod
        class MyMod(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 覆写 forward 方法，接收参数 a，并创建不同命名空间下的对象实例，返回它们的属性之和
            def forward(self, a):
                foo = jit._imported_class_test.foo.FooSameName(a)
                bar = jit._imported_class_test.bar.FooSameName(a)
                three = jit._imported_class_test.very.very.nested.FooUniqueName(a)
                return foo.x + bar.y + three.y

        # 创建 MyMod 类的实例 m
        m = MyMod()

        # 创建一个字节流对象 buffer，并将 MyMod 对象 m 保存到其中
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)

        # 由于类是全局注册的，需要清除 JIT 注册表以模拟加载新模型
        torch.testing._internal.jit_utils.clear_class_registry()

        # 将字节流对象 buffer 重置并加载模型
        buffer.seek(0)
        m_loaded = torch.jit.load(buffer)

        # 创建一个形状为 (2, 3) 的随机 tensor 输入
        input = torch.rand(2, 3)
        # 使用加载后的模型 m_loaded 对输入进行预测
        output = m_loaded(input)
        # 断言输出结果与输入的三倍相等
        self.assertEqual(3 * input, output)
    # 定义一个测试函数，测试类型转换的重载情况
    def test_cast_overloads(self):
        # 使用 torch.jit.script 装饰器定义一个脚本类 Foo
        @torch.jit.script
        class Foo:
            # 构造函数，初始化一个 float 类型的值
            def __init__(self, val: float) -> None:
                self.val = val

            # 将对象转换为整数类型
            def __int__(self):
                return int(self.val)

            # 将对象转换为浮点数类型
            def __float__(self):
                return self.val

            # 将对象转换为布尔类型
            def __bool__(self):
                return bool(self.val)

            # 将对象转换为字符串类型
            def __str__(self):
                return str(self.val)

        make_global(Foo)  # 将 Foo 类设置为全局可见，详见 [local resolution in python]

        # 定义一个测试函数，参数为一个 Foo 类型的对象，返回一个元组包含整数、浮点数和布尔值
        def test(foo: Foo) -> Tuple[int, float, bool]:
            # 如果 foo 被视为真值（即 __bool__ 方法返回 True）
            if foo:
                pass
            # 返回 foo 的整数、浮点数和布尔值表示
            return int(foo), float(foo), bool(foo)

        # 使用 torch.jit.script 装饰器将 test 函数编译为脚本函数
        fn = torch.jit.script(test)
        # 断言编译后的脚本函数返回值与直接调用 test 函数返回值相同
        self.assertEqual(fn(Foo(0.5)), test(0.5))
        self.assertEqual(fn(Foo(0.0)), test(0.0))
        # 对于字符串的格式化有轻微差异
        self.assertTrue("0.5" in (str(Foo(0.5))))
        self.assertTrue("0." in (str(Foo(0.0))))

        # 定义一个脚本类 BadBool
        @torch.jit.script
        class BadBool:
            # 构造函数，不做任何初始化操作
            def __init__(self):
                pass

            # 错误实现的 __bool__ 方法，返回一个元组而不是布尔值
            def __bool__(self):
                return (1, 2)

        # 使用 self.assertRaisesRegexWithHighlight 断言捕获 RuntimeError 异常
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "expected a bool expression for condition", ""
        ):
            # 使用 torch.jit.script 装饰器定义一个测试函数
            @torch.jit.script
            def test():
                # 如果 BadBool() 对象被视为真值，应该抛出异常
                if BadBool():
                    print(1)
                    pass

    # 定义一个测试函数，测试编译先行的情况
    def test_init_compiled_first(self):
        # 使用 torch.jit.script 装饰器定义一个脚本类 Foo
        @torch.jit.script  # noqa: B903
        class Foo:  # noqa: B903
            # __before_init__ 方法，访问 self.x 不应该抛出异常，因为 __init__ 应该已经被编译
            def __before_init__(self):
                return self.x

            # 构造函数，接受两个参数 x 和 y
            def __init__(self, x, y):
                self.x = x
                self.y = y

    # 定义一个测试函数，测试类在其方法内部构造自身的情况
    def test_class_constructs_itself(self):
        # 使用 torch.jit.script 装饰器定义一个脚本类 LSTMStateStack
        @torch.jit.script  # noqa: B903
        class LSTMStateStack:  # noqa: B903
            # 构造函数，接受两个整型参数 num_layers 和 hidden_size
            def __init__(self, num_layers: int, hidden_size: int) -> None:
                self.num_layers = num_layers
                self.hidden_size = hidden_size
                # 初始化最后状态为包含全零张量的元组
                self.last_state = (
                    torch.zeros(num_layers, 1, hidden_size),
                    torch.zeros(num_layers, 1, hidden_size),
                )
                # 初始化栈，包含最后状态的最后一个元素的拷贝
                self.stack = [(self.last_state[0][-1], self.last_state[0][-1])]

            # copy 方法，构造一个新的 LSTMStateStack 对象并返回
            def copy(self):
                # 应能在其方法内部构造一个类
                other = LSTMStateStack(self.num_layers, self.hidden_size)
                other.stack = list(self.stack)
                return other

    # 定义一个测试函数，测试可选类型的提升情况
    def test_optional_type_promotion(self):
        # 使用 torch.jit.script 装饰器定义一个脚本类 Leaf
        @torch.jit.script
        class Leaf:
            # 构造函数，初始化一个属性 x 为整数 1
            def __init__(self):
                self.x = 1

        # 不应该抛出异常
        # 使用 torch.jit.script 装饰器定义一个脚本类 Tree
        @torch.jit.script  # noqa: B903
        class Tree:  # noqa: B903
            # 构造函数，初始化一个可选类型的 Leaf 对象为 None
            def __init__(self):
                self.child = torch.jit.annotate(Optional[Leaf], None)

            # add_child 方法，接受一个 Leaf 类型的参数 child，将其赋给 self.child
            def add_child(self, child: Leaf) -> None:
                self.child = child
    def test_recursive_class(self):
        """
        Recursive class types not yet supported. We should give a good error message.
        """
        # 使用 assertRaises 方法验证运行时错误是否被抛出
        with self.assertRaises(RuntimeError):
            # 使用 torch.jit.script 注释标记一个 TorchScript 脚本
            @torch.jit.script  # noqa: B903
            # 定义一个名为 Tree 的类
            class Tree:  # noqa: B903
                # 类的初始化方法
                def __init__(self):
                    # 初始化 self.parent 属性，类型为 Optional[Tree]
                    self.parent = torch.jit.annotate(Optional[Tree], None)

    def test_class_constant(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 声明该类的常量属性列表
            __constants__ = ["w"]

            # 类的初始化方法
            def __init__(self, w):
                super().__init__()
                # 初始化类的实例变量 self.w
                self.w = w

            # 类的前向传播方法
            def forward(self, x):
                # 确保类常量在方法中可访问
                y = self.w
                return x, y

        # 测试类常量的序列化和反序列化
        for c in (2, 1.0, None, True, "str", (2, 3), [5.9, 7.3]):
            # 使用 torch.jit.script 将 M 类实例转换为 TorchScript 脚本
            m = torch.jit.script(M(c))
            buffer = io.BytesIO()
            # 将 TorchScript 脚本保存到缓冲区
            torch.jit.save(m, buffer)

            buffer.seek(0)
            # 从缓冲区加载 TorchScript 脚本
            m_loaded = torch.jit.load(buffer)
            input = torch.rand(2, 3)
            # 确保类常量在模型中可访问
            self.assertEqual(m(input), m_loaded(input))
            # 确保类常量在模块中可访问
            self.assertEqual(m.w, m_loaded.w)

    def test_py_class_to_ivalue_missing_attribute(self):
        # 定义一个 Python 类 Foo
        class Foo:
            i: int
            f: float

            # 类的初始化方法
            def __init__(self, i: int, f: float):
                self.i = i
                self.f = f

        make_global(Foo)  # see [local resolution in python]

        # 使用 torch.jit.script 将 test_fn 函数转换为 TorchScript 脚本
        @torch.jit.script
        def test_fn(x: Foo) -> float:
            return x.i + x.f

        test_fn(Foo(3, 4.0))

        # 使用 assertRaisesRegexWithHighlight 方法验证运行时错误是否被抛出，并包含特定信息
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "missing attribute i", ""
        ):
            # 调用 test_fn 函数并传入一个随机张量作为参数
            test_fn(torch.rand(3, 4))
    def test_unused_method(self):
        """
        Test unused methods on scripted classes.
        """

        @torch.jit.script
        class Unused:
            def __init__(self):
                self.count: int = 0
                self.items: List[int] = []

            def used(self):
                # 增加计数并返回
                self.count += 1
                return self.count

            @torch.jit.unused
            def unused(self, x: int, y: Iterable[int], **kwargs) -> int:
                # 未使用的方法，尝试从 self.items 中获取下一个元素，但未使用
                a = next(self.items)
                return a

            def uses_unused(self) -> int:
                # 调用未使用的方法，返回其结果
                return self.unused(y="hi", x=3)

        class ModuleWithUnused(nn.Module):
            def __init__(self):
                super().__init__()
                # 实例化 Unused 类
                self.obj = Unused()

            def forward(self):
                # 调用已使用的方法
                return self.obj.used()

            @torch.jit.export
            def calls_unused(self):
                # 调用未使用的方法，应当引发错误
                return self.obj.unused(3, "hi")

            @torch.jit.export
            def calls_unused_indirectly(self):
                # 调用使用了未使用的方法的方法，应当引发错误
                return self.obj.uses_unused()

        python_module = ModuleWithUnused()
        script_module = torch.jit.script(ModuleWithUnused())

        # Forward 应当工作，因为它没有使用任何标记为未使用的方法。
        self.assertEqual(python_module.forward(), script_module.forward())

        # 调用标记为未使用的方法应当引发错误。
        with self.assertRaises(torch.jit.Error):
            script_module.calls_unused()

        with self.assertRaises(torch.jit.Error):
            script_module.calls_unused_indirectly()

    def test_self_referential_method(self):
        """
        Test that a scripted class can have a method that refers to the class itself
        in its type annotations.
        """

        @torch.jit.script
        class Meta:
            def __init__(self, a: int):
                self.a = a

            def method(self, other: List["Meta"]) -> "Meta":
                # 返回一个新的 Meta 对象，其属性 a 是 other 列表的长度
                return Meta(len(other))

        class ModuleWithMeta(torch.nn.Module):
            def __init__(self, a: int):
                super().__init__()
                # 实例化 Meta 类
                self.meta = Meta(a)

            def forward(self):
                # 调用 Meta 类的 method 方法，并返回其属性 a
                new_obj = self.meta.method([self.meta])
                return new_obj.a

        self.checkModule(ModuleWithMeta(5), ())

    def test_type_annotation(self):
        """
        Test that annotating container attributes with types works correctly
        """

        @torch.jit.script
        class CompetitiveLinkingTokenReplacementUtils:
            def __init__(self):
                # 初始化一个列表，包含元组作为元素
                self.my_list: List[Tuple[float, int, int]] = []
                # 初始化一个空字典
                self.my_dict: Dict[int, int] = {}

        @torch.jit.script
        def foo():
            y = CompetitiveLinkingTokenReplacementUtils()
            # 创建一个新字典并赋值给 y 的 my_dict 属性
            new_dict: Dict[int, int] = {1: 1, 2: 2}
            y.my_dict = new_dict

            # 创建一个新列表并赋值给 y 的 my_list 属性
            new_list: List[Tuple[float, int, int]] = [(1.0, 1, 1)]
            y.my_list = new_list
            return y
    # 定义一个测试静态方法的方法
    def test_staticmethod(self):
        """
        Test static methods on class types.
        """

        # 使用 Torch 的脚本装饰器定义一个类，并且标记其为脚本类
        @torch.jit.script
        class ClassWithStaticMethod:
            # 类的初始化方法，接受两个整数参数 a 和 b
            def __init__(self, a: int, b: int):
                self.a: int = a  # 设置实例变量 a
                self.b: int = b  # 设置实例变量 b

            # 返回实例变量 a 的方法
            def get_a(self):
                return self.a

            # 返回实例变量 b 的方法
            def get_b(self):
                return self.b

            # 定义对象相等的魔法方法，比较两个对象的 a 和 b 是否相等
            def __eq__(self, other: "ClassWithStaticMethod"):
                return self.a == other.a and self.b == other.b

            # 静态方法，调用构造函数创建一个类对象
            @staticmethod
            def create(args: List["ClassWithStaticMethod"]) -> "ClassWithStaticMethod":
                return ClassWithStaticMethod(args[0].a, args[0].b)

            # 静态方法，调用另一个静态方法创建一个类对象
            @staticmethod
            def create_from(a: int, b: int) -> "ClassWithStaticMethod":
                a = ClassWithStaticMethod(a, b)
                return ClassWithStaticMethod.create([a])

        # 脚本函数调用静态方法
        def test_function(a: int, b: int) -> "ClassWithStaticMethod":
            return ClassWithStaticMethod.create_from(a, b)

        # 将 ClassWithStaticMethod 类注册为全局对象
        make_global(ClassWithStaticMethod)

        # 使用 checkScript 方法检查 test_function 的输出
        self.checkScript(test_function, (1, 2))

    # 定义一个测试类方法的方法
    def test_classmethod(self):
        """
        Test classmethods on class types.
        """

        # 使用 Torch 的脚本装饰器定义一个类，并且标记其为脚本类
        @torch.jit.script
        class ClassWithClassMethod:
            # 类的初始化方法，接受一个整数参数 a
            def __init__(self, a: int):
                self.a: int = a  # 设置实例变量 a

            # 定义对象相等的魔法方法，比较两个对象的 a 是否相等
            def __eq__(self, other: "ClassWithClassMethod"):
                return self.a == other.a

            # 类方法，使用类本身来创建一个类对象
            @classmethod
            def create(cls, a: int) -> "ClassWithClassMethod":
                return cls(a)

        # 将 ClassWithClassMethod 类注册为全局对象
        make_global(ClassWithClassMethod)

        # 脚本函数调用类方法
        def test_function(a: int) -> "ClassWithClassMethod":
            x = ClassWithClassMethod(a)
            # 支持使用实例调用类方法
            # 不支持使用类本身调用类方法
            return x.create(a)

        # 使用 checkScript 方法检查 test_function 的输出
        self.checkScript(test_function, (1,))

    # 根据 TorchDynamo 的状态来决定是否跳过测试
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_properties(self):
        """
        Test that a scripted class can make use of the @property decorator.
        """

        def free_function(x: int) -> int:
            return x + 1

        @torch.jit.script
        class Properties:
            __jit_unused_properties__ = ["unsupported"]

            def __init__(self, a: int):
                self.a = a

            @property
            def attr(self) -> int:
                # Getter method for 'attr' property
                return self.a - 1

            @property
            def unsupported(self) -> int:
                # Getter method for 'unsupported' property (not used in this script)
                return sum([self.a])

            @torch.jit.unused
            @property
            def unsupported_2(self) -> int:
                # Getter method for 'unsupported_2' property (marked as unused)
                return sum([self.a])

            @unsupported_2.setter
            def unsupported_2(self, value):
                # Setter method for 'unsupported_2' property (sets 'a' to a sum)
                self.a = sum([self.a])

            @attr.setter
            def attr(self, value: int):
                # Setter method for 'attr' property (sets 'a' to 'value' plus 3)
                self.a = value + 3

        @torch.jit.script
        class NoSetter:
            def __init__(self, a: int):
                self.a = a

            @property
            def attr(self) -> int:
                # Getter method for 'attr' property, using free function 'free_function'
                return free_function(self.a)

        @torch.jit.script
        class MethodThatUsesProperty:
            def __init__(self, a: int):
                self.a = a

            @property
            def attr(self) -> int:
                # Getter method for 'attr' property
                return self.a - 2

            @attr.setter
            def attr(self, value: int):
                # Setter method for 'attr' property (sets 'a' to 'value' plus 4)
                self.a = value + 4

            def forward(self):
                # Method that uses the 'attr' property
                return self.attr

        class ModuleWithProperties(torch.nn.Module):
            def __init__(self, a: int):
                super().__init__()
                self.props = Properties(a)

            def forward(self, a: int, b: int, c: int, d: int):
                # Set 'attr' property of self.props to 'a'
                self.props.attr = a
                # Create Properties instance with 'b'
                props = Properties(b)
                # Create NoSetter instance with 'c'
                no_setter = NoSetter(c)
                # Create MethodThatUsesProperty instance with 'a + b'
                method_uses_property = MethodThatUsesProperty(a + b)

                # Set 'attr' property of props to 'c'
                props.attr = c
                # Set 'attr' property of method_uses_property to 'd'
                method_uses_property.attr = d

                # Return sum of 'attr' properties from self.props, no_setter, and method_uses_property
                return self.props.attr + no_setter.attr + method_uses_property.forward()

        self.checkModule(
            ModuleWithProperties(5),
            (
                5,
                6,
                7,
                8,
            ),
        )
    def test_custom_delete(self):
        """
        Test that del can be called on an instance of a class that
        overrides __delitem__.
        """

        # 定义一个示例类 Example，演示如何重载 __delitem__ 方法
        class Example:
            def __init__(self):
                # 初始化一个字典 _data，包含一个键 "1" 和对应的 torch.Tensor 对象
                self._data: Dict[str, torch.Tensor] = {"1": torch.tensor(1.0)}

            def check(self, key: str) -> bool:
                # 检查给定的键是否在 _data 字典中
                return key in self._data

            def __delitem__(self, key: str):
                # 删除 _data 字典中指定键的条目
                del self._data[key]

        # 定义一个测试函数 fn，测试 Example 类的 del 操作和 check 方法
        def fn() -> bool:
            example = Example()
            del example["1"]  # 删除 Example 实例中的键 "1"
            return example.check("1")  # 检查 "1" 是否仍在 Example 实例的 _data 字典中

        # 使用 self.checkScript 方法测试 fn 函数
        self.checkScript(fn, ())

        # 测试当类未定义 __delitem__ 方法时的情况
        class NoDelItem:
            def __init__(self):
                # 初始化一个字典 _data，包含一个键 "1" 和对应的 torch.Tensor 对象
                self._data: Dict[str, torch.Tensor] = {"1": torch.tensor(1.0)}

            def check(self, key: str) -> bool:
                # 检查给定的键是否在 _data 字典中
                return key in self._data

        # 定义一个测试函数 fn，试图删除 NoDelItem 实例中的键 "1"
        def fn() -> bool:
            example = NoDelItem()
            key = "1"
            del example[key]  # 尝试删除 NoDelItem 实例中的键 "1"
            return example.check(key)  # 检查 "1" 是否仍在 NoDelItem 实例的 _data 字典中

        # 使用 self.assertRaisesRegexWithHighlight 检查运行时错误
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Class does not define __delitem__", "example[key]"
        ):
            self.checkScript(fn, ())

    def test_recursive_script_builtin_type_resolution(self):
        """
        Test resolution of built-in torch types(e.g. torch.Tensor, torch.device) when a class is recursively compiled.
        """
        
        # 定义 A 类，演示递归编译时内置 torch 类型的解析
        class A:
            def __init__(self):
                pass

            def f(self, x: torch.Tensor, y: torch.device) -> torch.Tensor:
                # 使用输入的 x Tensor 并转换为指定的设备 y
                return x.to(device=y)

            def g(self, x: torch.device) -> torch.device:
                # 直接返回输入的设备类型 x
                return x

            def h(self, a: "A") -> "A":
                # 返回一个新的 A 类实例
                return A()

            def i(self, a: List[int]) -> int:
                # 返回列表 a 的第一个整数元素
                return a[0]

            def j(self, l: List[torch.device]) -> torch.device:
                # 返回列表 l 的第一个设备类型元素
                return l[0]

        # 定义多个测试函数，检查 A 类的不同方法调用是否通过脚本化检查
        def call_f():
            a = A()
            return a.f(torch.tensor([1]), torch.device("cpu"))

        def call_g():
            a = A()
            return a.g(torch.device("cpu"))

        def call_i():
            a = A()
            return a.i([3])

        def call_j():
            a = A()
            return a.j([torch.device("cpu"), torch.device("cpu")])

        # 遍历所有测试函数，使用 self.checkScript 方法检查脚本化后的执行结果
        for fn in [call_f, call_g, call_i, call_j]:
            self.checkScript(fn, ())
            # 获取脚本化后的函数的导入导出副本，并验证其与原始函数的结果是否相同
            s = self.getExportImportCopy(torch.jit.script(fn))
            self.assertEqual(s(), fn())
    def test_recursive_script_module_builtin_type_resolution(self):
        """
        Test resolution of built-in torch types(e.g. torch.Tensor, torch.device) when a class is recursively compiled
        when compiling a module.
        """
        # 定义一个测试函数，用于验证在编译模块时，解析内置的 torch 类型（如 torch.Tensor, torch.device）
        
        class Wrapper:
            def __init__(self, t):
                self.t = t

            def to(self, l: List[torch.device], device: Optional[torch.device] = None):
                return self.t.to(device=device)
                # 将包装的对象转移到指定设备

        class A(nn.Module):
            def forward(self):
                return Wrapper(torch.rand(4, 4))
                # 在前向传播中返回一个 Wrapper 类的实例，其内部包含一个随机生成的 4x4 的 Tensor

        scripted = torch.jit.script(A())
        self.getExportImportCopy(scripted)
        # 对模块 A 进行脚本化，并调用测试辅助方法 getExportImportCopy

    def test_class_attribute_wrong_type(self):
        """
        Test that the error message displayed when convering a class type
        to an IValue that has an attribute of the wrong type.
        """
        # 测试当将类类型转换为具有错误类型属性的 IValue 时显示的错误消息
        
        @torch.jit.script  # noqa: B903
        class ValHolder:  # noqa: B903
            def __init__(self, val):
                self.val = val
                # 初始化方法，接受一个值作为属性

        class Mod(nn.Module):
            def __init__(self):
                super().__init__()
                self.mod1 = ValHolder("1")
                self.mod2 = ValHolder("2")
                # 初始化方法，创建两个 ValHolder 的实例作为模块的属性

            def forward(self, cond: bool):
                if cond:
                    mod = self.mod1
                else:
                    mod = self.mod2
                return mod.val
                # 根据条件返回不同模块的 val 属性值

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Could not cast attribute 'val' to type Tensor", ""
        ):
            torch.jit.script(Mod())
            # 断言捕获 RuntimeError 异常，并验证错误消息是否包含特定文本

    def test_recursive_scripting(self):
        """
        Test that class types are recursively scripted when an Python instance of one
        is encountered as a module attribute.
        """
        # 测试当遇到 Python 实例作为模块属性时，类类型是否会递归进行脚本化
        
        class Class:
            def __init__(self, a: int):
                self.a = a
                # 初始化方法，接受一个整数作为属性

            def get_a(self) -> int:
                return self.a
                # 返回属性 a 的值

        class M(torch.nn.Module):
            def __init__(self, obj):
                super().__init__()
                self.obj = obj
                # 初始化方法，接受一个 Class 类的实例作为模块的属性

            def forward(self) -> int:
                return self.obj.get_a()
                # 在前向传播中调用属性对象的方法获取属性值

        self.checkModule(M(Class(4)), ())
        # 调用测试辅助方法 checkModule，验证给定 M 类的实例是否符合预期
    def test_recursive_scripting_failed(self):
        """
        Test that class types module attributes that fail to script
        are added as failed attributes and do not cause compilation itself
        to fail unless they are used in scripted code.
        """

        class UnscriptableClass:
            def __init__(self, a: int):
                self.a = a

            def get_a(self) -> bool:
                return issubclass(self.a, int)

        # This Module has an attribute of type UnscriptableClass
        # and tries to use it in scripted code. This should fail.
        class ShouldNotCompile(torch.nn.Module):
            def __init__(self, obj):
                super().__init__()
                self.obj = obj

            def forward(self) -> bool:
                return self.obj.get_a()

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "failed to convert Python type", ""
        ):
            # 尝试对包含 UnscriptableClass 类型的对象进行脚本化，预期会失败
            torch.jit.script(ShouldNotCompile(UnscriptableClass(4)))

        # This Module has an attribute of type UnscriptableClass
        # and does not try to use it in scripted code. This should not fail.
        class ShouldCompile(torch.nn.Module):
            def __init__(self, obj):
                super().__init__()
                self.obj = obj

            @torch.jit.ignore
            def ignored_method(self) -> bool:
                return self.obj.get_a()

            def forward(self, x: int) -> int:
                return x + x

        # 对包含 UnscriptableClass 类型的对象进行脚本化，但在脚本代码中未使用该对象，预期不会失败
        self.checkModule(ShouldCompile(UnscriptableClass(4)), (4,))

    def test_unresolved_class_attributes(self):
        class UnresolvedAttrClass:
            def __init__(self):
                pass

            # Define class attributes with complex unpacking and typing
            (attr_a, attr_b), [attr_c, attr_d] = ("", ""), ["", ""]
            attr_e: int = 0

        # Define functions to access each class attribute
        def fn_a():
            u = UnresolvedAttrClass()
            return u.attr_a

        def fn_b():
            u = UnresolvedAttrClass()
            return u.attr_b

        def fn_c():
            u = UnresolvedAttrClass()
            return u.attr_c

        def fn_d():
            u = UnresolvedAttrClass()
            return u.attr_d

        def fn_e():
            u = UnresolvedAttrClass()
            return u.attr_e

        error_message_regex = (
            "object has no attribute or method.*is defined as a class attribute"
        )
        # Iterate over functions and expect RuntimeError for accessing class attributes
        for fn in (fn_a, fn_b, fn_c, fn_d, fn_e):
            with self.assertRaisesRegex(RuntimeError, error_message_regex):
                torch.jit.script(fn)
```