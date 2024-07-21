# `.\pytorch\test\jit\test_torchbind.py`

```py
# Owner(s): ["oncall: jit"]

# 导入所需的模块和库
import copy  # 导入 copy 模块
import io  # 导入 io 模块
import os  # 导入 os 模块
import sys  # 导入 sys 模块
import unittest  # 导入 unittest 模块
from typing import Optional  # 导入 Optional 类型提示

import torch  # 导入 torch 库
from torch.testing._internal.common_utils import skipIfTorchDynamo  # 从 common_utils 模块中导入 skipIfTorchDynamo 装饰器

# Make the helper files in test/ importable
# 将 test/ 中的辅助文件添加到模块搜索路径中
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing import FileCheck  # 导入 FileCheck 工具
from torch.testing._internal.common_utils import (
    find_library_location,  # 从 common_utils 模块中导入 find_library_location 函数
    IS_FBCODE,  # 导入 IS_FBCODE 常量
    IS_MACOS,  # 导入 IS_MACOS 常量
    IS_SANDCASTLE,  # 导入 IS_SANDCASTLE 常量
    IS_WINDOWS,  # 导入 IS_WINDOWS 常量
)
from torch.testing._internal.jit_utils import JitTestCase  # 导入 JitTestCase 类

if __name__ == "__main__":
    # 如果此脚本作为主程序运行，则抛出运行时错误提示
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


@skipIfTorchDynamo("skipping as a precaution")
class TestTorchbind(JitTestCase):
    def setUp(self):
        # 如果运行环境为 Sandcastle、macOS 或者 FBCODE，则跳过测试
        if IS_SANDCASTLE or IS_MACOS or IS_FBCODE:
            raise unittest.SkipTest("non-portable load_library call used in test")
        
        # 查找并加载 libtorchbind_test.so 或 torchbind_test.dll 库文件
        lib_file_path = find_library_location("libtorchbind_test.so")
        if IS_WINDOWS:
            lib_file_path = find_library_location("torchbind_test.dll")
        torch.ops.load_library(str(lib_file_path))
    def test_torchbind(self):
        def test_equality(f, cmp_key):
            obj1 = f()  # 调用函数 f，获取对象 obj1
            obj2 = torch.jit.script(f)()  # 使用 Torch JIT 对函数 f 进行脚本化，获取对象 obj2
            return (cmp_key(obj1), cmp_key(obj2))  # 返回 obj1 和 obj2 经过 cmp_key 处理后的比较结果

        def f():
            val = torch.classes._TorchScriptTesting._Foo(5, 3)  # 创建 _Foo 类的对象 val
            val.increment(1)  # 调用 val 的 increment 方法
            return val  # 返回 val

        test_equality(f, lambda x: x)  # 测试函数 f 的执行结果和其脚本化版本的执行结果是否相等

        with self.assertRaisesRegex(RuntimeError, "Expected a value of type 'int'"):
            val = torch.classes._TorchScriptTesting._Foo(5, 3)  # 创建 _Foo 类的对象 val
            val.increment("foo")  # 尝试用字符串调用 increment 方法，预期抛出 RuntimeError 异常

        def f():
            ss = torch.classes._TorchScriptTesting._StackString(["asdf", "bruh"])  # 创建 _StackString 类的对象 ss
            return ss.pop()  # 调用 ss 的 pop 方法

        test_equality(f, lambda x: x)  # 测试函数 f 的执行结果和其脚本化版本的执行结果是否相等

        def f():
            ss1 = torch.classes._TorchScriptTesting._StackString(["asdf", "bruh"])  # 创建 _StackString 类的对象 ss1
            ss2 = torch.classes._TorchScriptTesting._StackString(["111", "222"])  # 创建 _StackString 类的对象 ss2
            ss1.push(ss2.pop())  # 将 ss2 中弹出的值推送到 ss1 中
            return ss1.pop() + ss2.pop()  # 返回 ss1 和 ss2 弹出的值的拼接结果

        test_equality(f, lambda x: x)  # 测试函数 f 的执行结果和其脚本化版本的执行结果是否相等

        # test nn module with prepare_scriptable function
        class NonJitableClass:
            def __init__(self, int1, int2):
                self.int1 = int1
                self.int2 = int2

            def return_vals(self):
                return self.int1, self.int2

        class CustomWrapper(torch.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            def forward(self) -> None:
                self.foo.increment(1)  # 调用 foo 的 increment 方法
                return

            def __prepare_scriptable__(self):
                int1, int2 = self.foo.return_vals()  # 调用 foo 的 return_vals 方法获取 int1 和 int2
                foo = torch.classes._TorchScriptTesting._Foo(int1, int2)  # 创建 _Foo 类的对象 foo
                return CustomWrapper(foo)  # 返回包装后的 CustomWrapper 对象

        foo = CustomWrapper(NonJitableClass(1, 2))  # 创建 CustomWrapper 类的对象 foo
        jit_foo = torch.jit.script(foo)  # 对 foo 进行 Torch JIT 脚本化处理

    def test_torchbind_take_as_arg(self):
        global StackString  # 声明 StackString 变量为全局变量
        StackString = torch.classes._TorchScriptTesting._StackString  # 将 _StackString 类赋值给 StackString 变量

        def foo(stackstring):
            # type: (StackString)
            stackstring.push("lel")  # 调用 stackstring 的 push 方法
            return stackstring  # 返回 stackstring 对象

        script_input = torch.classes._TorchScriptTesting._StackString([])  # 创建 _StackString 类的对象 script_input
        scripted = torch.jit.script(foo)  # 对 foo 函数进行 Torch JIT 脚本化处理
        script_output = scripted(script_input)  # 对 script_input 进行脚本化处理后调用脚本化函数 scripted
        self.assertEqual(script_output.pop(), "lel")  # 断言 script_output 的 pop 方法返回值为 "lel"

    def test_torchbind_return_instance(self):
        def foo():
            ss = torch.classes._TorchScriptTesting._StackString(["hi", "mom"])  # 创建 _StackString 类的对象 ss
            return ss  # 返回 ss 对象

        scripted = torch.jit.script(foo)  # 对 foo 函数进行 Torch JIT 脚本化处理
        # Ensure we are creating the object and calling __init__
        # rather than calling the __init__wrapper nonsense
        fc = (
            FileCheck()
            .check("prim::CreateObject()")  # 检查是否创建了对象
            .check('prim::CallMethod[name="__init__"]')  # 检查是否调用了 __init__ 方法
        )
        fc.run(str(scripted.graph))  # 在脚本化的图形中运行文件检查
        out = scripted()  # 调用脚本化后的函数获取输出对象
        self.assertEqual(out.pop(), "mom")  # 断言输出对象的 pop 方法返回值为 "mom"
        self.assertEqual(out.pop(), "hi")  # 断言输出对象的 pop 方法返回值为 "hi"
    # 测试 TorchScript 脚本化的返回实例的方法
    def test_torchbind_return_instance_from_method(self):
        # 定义内部函数 foo
        def foo():
            # 使用 TorchScriptTesting 模块中的 _StackString 类创建对象 ss，初始化包含字符串列表
            ss = torch.classes._TorchScriptTesting._StackString(["hi", "mom"])
            # 克隆 ss 对象
            clone = ss.clone()
            # 调用 ss 对象的 pop 方法，弹出栈顶元素，并返回弹出的元素值
            ss.pop()
            # 返回 ss 和 clone 两个对象
            return ss, clone

        # 对 foo 函数进行 TorchScript 脚本化
        scripted = torch.jit.script(foo)
        # 执行脚本化的函数，获取返回结果
        out = scripted()
        # 断言：验证第一个返回对象 ss 的 pop 方法返回 "hi"
        self.assertEqual(out[0].pop(), "hi")
        # 断言：验证第二个返回对象 clone 的 pop 方法返回 "mom"
        self.assertEqual(out[1].pop(), "mom")
        # 断言：再次验证 clone 的 pop 方法返回 "hi"，验证克隆不影响原对象 ss
        self.assertEqual(out[1].pop(), "hi")

    # 测试 TorchScript 脚本化的属性 getter 和 setter 方法
    def test_torchbind_def_property_getter_setter(self):
        # 定义内部函数 foo_getter_setter_full
        def foo_getter_setter_full():
            # 使用 TorchScriptTesting 模块中的 _FooGetterSetter 类创建对象 fooGetterSetter，初始化属性值
            fooGetterSetter = torch.classes._TorchScriptTesting._FooGetterSetter(5, 6)
            # 获取属性 x 的旧值
            old = fooGetterSetter.x
            # 设置属性 x 的新值，通过方法修改属性值
            fooGetterSetter.x = old + 4
            # 获取属性 x 的新值
            new = fooGetterSetter.x
            # 返回属性 x 的旧值和新值
            return old, new

        # 使用测试工具函数 checkScript 对 foo_getter_setter_full 函数进行 TorchScript 脚本化
        self.checkScript(foo_getter_setter_full, ())

        # 定义内部函数 foo_getter_setter_lambda
        def foo_getter_setter_lambda():
            # 使用 TorchScriptTesting 模块中的 _FooGetterSetterLambda 类创建对象 foo，初始化属性值
            foo = torch.classes._TorchScriptTesting._FooGetterSetterLambda(5)
            # 获取属性 x 的旧值
            old = foo.x
            # 设置属性 x 的新值，通过 lambda 函数修改属性值
            foo.x = old + 4
            # 获取属性 x 的新值
            new = foo.x
            # 返回属性 x 的旧值和新值
            return old, new

        # 使用测试工具函数 checkScript 对 foo_getter_setter_lambda 函数进行 TorchScript 脚本化
        self.checkScript(foo_getter_setter_lambda, ())

    # 测试 TorchScript 脚本化的属性仅 getter 方法
    def test_torchbind_def_property_just_getter(self):
        # 定义内部函数 foo_just_getter
        def foo_just_getter():
            # 使用 TorchScriptTesting 模块中的 _FooGetterSetter 类创建对象 fooGetterSetter，初始化属性值
            fooGetterSetter = torch.classes._TorchScriptTesting._FooGetterSetter(5, 6)
            # 获取属性 y 的值
            return fooGetterSetter, fooGetterSetter.y

        # 对 foo_just_getter 函数进行 TorchScript 脚本化
        scripted = torch.jit.script(foo_just_getter)
        # 执行脚本化的函数，获取返回结果
        out, result = scripted()
        # 断言：验证属性 y 的值为 10
        self.assertEqual(result, 10)
        
        # 预期属性 y 是只读的，尝试设置属性 y 的值，预期抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "can't set attribute"):
            out.y = 5

        # 定义内部函数 foo_not_setter
        def foo_not_setter():
            # 使用 TorchScriptTesting 模块中的 _FooGetterSetter 类创建对象 fooGetterSetter，初始化属性值
            fooGetterSetter = torch.classes._TorchScriptTesting._FooGetterSetter(5, 6)
            # 获取属性 y 的旧值
            old = fooGetterSetter.y
            # 尝试设置属性 y 的值，预期抛出 RuntimeError 异常
            fooGetterSetter.y = old + 4
            # 返回属性 y 的值
            return fooGetterSetter.y

        # 预期属性 y 是只读的，尝试对 foo_not_setter 函数进行 TorchScript 脚本化，预期抛出 RuntimeError 异常
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Tried to set read-only attribute: y",
            "fooGetterSetter.y = old + 4",
        ):
            scripted = torch.jit.script(foo_not_setter)

    # 测试 TorchScript 脚本化的属性读写方法
    def test_torchbind_def_property_readwrite(self):
        # 定义内部函数 foo_readwrite
        def foo_readwrite():
            # 使用 TorchScriptTesting 模块中的 _FooReadWrite 类创建对象 fooReadWrite，初始化属性值
            fooReadWrite = torch.classes._TorchScriptTesting._FooReadWrite(5, 6)
            # 获取属性 x 的旧值
            old = fooReadWrite.x
            # 设置属性 x 的新值
            fooReadWrite.x = old + 4
            # 返回属性 x 和属性 y 的值
            return fooReadWrite.x, fooReadWrite.y

        # 使用测试工具函数 checkScript 对 foo_readwrite 函数进行 TorchScript 脚本化
        self.checkScript(foo_readwrite, ())

        # 定义内部函数 foo_readwrite_error
        def foo_readwrite_error():
            # 使用 TorchScriptTesting 模块中的 _FooReadWrite 类创建对象 fooReadWrite，初始化属性值
            fooReadWrite = torch.classes._TorchScriptTesting._FooReadWrite(5, 6)
            # 尝试设置属性 y 的值，预期抛出 RuntimeError 异常
            fooReadWrite.y = 5
            # 返回 fooReadWrite 对象
            return fooReadWrite

        # 预期属性 y 是只读的，尝试对 foo_readwrite_error 函数进行 TorchScript 脚本化，预期抛出 RuntimeError 异常
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set read-only attribute: y", "fooReadWrite.y = 5"
        ):
            scripted = torch.jit.script(foo_readwrite_error)
    # 测试函数，用于测试将实例作为方法参数传递给 TorchScript
    def test_torchbind_take_instance_as_method_arg(self):
        # 内部定义函数 foo，创建两个 _StackString 实例，并调用 merge 方法
        def foo():
            ss = torch.classes._TorchScriptTesting._StackString(["mom"])
            ss2 = torch.classes._TorchScriptTesting._StackString(["hi"])
            ss.merge(ss2)
            return ss
        
        # 对 foo 函数进行 TorchScript 脚本化
        scripted = torch.jit.script(foo)
        # 执行脚本化的函数，获取返回值 ss
        out = scripted()
        # 断言 ss.pop() 的返回值为 "hi"
        self.assertEqual(out.pop(), "hi")
        # 再次断言 ss.pop() 的返回值为 "mom"
        self.assertEqual(out.pop(), "mom")

    # 测试函数，测试 TorchScript 返回元组的情况
    def test_torchbind_return_tuple(self):
        # 内部定义函数 f，创建一个 _StackString 实例，并调用 return_a_tuple 方法
        def f():
            val = torch.classes._TorchScriptTesting._StackString(["3", "5"])
            return val.return_a_tuple()
        
        # 对 f 函数进行 TorchScript 脚本化
        scripted = torch.jit.script(f)
        # 执行脚本化的函数，获取返回的元组 tup
        tup = scripted()
        # 断言 tup 的值为 (1337.0, 123)
        self.assertEqual(tup, (1337.0, 123))

    # 测试函数，测试 TorchScript 中的保存和加载
    def test_torchbind_save_load(self):
        # 内部定义函数 foo，创建两个 _StackString 实例，并调用 merge 方法
        def foo():
            ss = torch.classes._TorchScriptTesting._StackString(["mom"])
            ss2 = torch.classes._TorchScriptTesting._StackString(["hi"])
            ss.merge(ss2)
            return ss
        
        # 对 foo 函数进行 TorchScript 脚本化
        scripted = torch.jit.script(foo)
        # 调用自定义方法 getExportImportCopy 对脚本化的模型进行保存和加载操作
        self.getExportImportCopy(scripted)

    # 测试函数，测试 TorchScript 中 lambda 方法的使用
    def test_torchbind_lambda_method(self):
        # 内部定义函数 foo，创建一个 _StackString 实例，并调用 top 方法
        def foo():
            ss = torch.classes._TorchScriptTesting._StackString(["mom"])
            return ss.top()
        
        # 对 foo 函数进行 TorchScript 脚本化
        scripted = torch.jit.script(foo)
        # 执行脚本化的函数，断言其返回值为 "mom"
        self.assertEqual(scripted(), "mom")

    # 测试函数，测试 TorchScript 中类属性的递归使用
    def test_torchbind_class_attr_recursive(self):
        # 内部定义类 FooBar，继承自 torch.nn.Module
        class FooBar(torch.nn.Module):
            def __init__(self, foo_model):
                super().__init__()
                self.foo_mod = foo_model

            def forward(self) -> int:
                # 调用 foo_mod 的 info 方法，并返回结果
                return self.foo_mod.info()

            def to_ivalue(self):
                # 创建一个 _Foo 实例，并将其作为参数传递给 FooBar 类的新实例
                torchbind_model = torch.classes._TorchScriptTesting._Foo(
                    self.foo_mod.info(), 1
                )
                return FooBar(torchbind_model)

        # 创建 FooBar 类的实例 inst，传入 _Foo(2, 3) 作为参数
        inst = FooBar(torch.classes._TorchScriptTesting._Foo(2, 3))
        # 对 inst.to_ivalue() 进行 TorchScript 脚本化
        scripted = torch.jit.script(inst.to_ivalue())
        # 执行脚本化的函数，断言其返回值为 6
        self.assertEqual(scripted(), 6)

    # 测试函数，测试 TorchScript 中类属性的使用
    def test_torchbind_class_attribute(self):
        # 内部定义类 FooBar1234，继承自 torch.nn.Module
        class FooBar1234(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个 _StackString 实例赋值给 self.f
                self.f = torch.classes._TorchScriptTesting._StackString(["3", "4"])

            def forward(self):
                # 调用 self.f 的 top 方法，并返回结果
                return self.f.top()

        # 创建 FooBar1234 类的实例 inst
        inst = FooBar1234()
        # 对 inst 进行 TorchScript 脚本化
        scripted = torch.jit.script(inst)
        # 调用自定义方法 getExportImportCopy 对脚本化的模型进行保存和加载操作，并获取结果 eic
        eic = self.getExportImportCopy(scripted)
        # 断言 eic() 的返回值为 "deserialized"
        assert eic() == "deserialized"
        # 循环断言 eic.f.pop() 的返回值依次为 "deserialized", "was", "i"
        for expected in ["deserialized", "was", "i"]:
            assert eic.f.pop() == expected
    def test_torchbind_getstate(self):
        # 定义一个继承自 torch.nn.Module 的类 FooBar4321
        class FooBar4321(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 torch 脚本测试用的 PickleTester 实例，传入列表 [3, 4]
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

            # 前向传播方法，调用 self.f 的 top 方法
            def forward(self):
                return self.f.top()

        # 创建 FooBar4321 的实例
        inst = FooBar4321()
        # 对 inst 进行脚本化
        scripted = torch.jit.script(inst)
        # 获得脚本化后对象的导出/导入复制
        eic = self.getExportImportCopy(scripted)
        # 注意：我们期望值为 {7, 3, 3, 1}，因为 __getstate__ 方法定义为返回 {1, 3, 3, 7}。
        # 尝试根据测试中实例化时的某些转换使其实际依赖于值，但由于似乎我们多次序列化/反序列化，该转换不如预期那样。
        assert eic() == 7
        # 遍历期望的值列表 [7, 3, 3, 1]
        for expected in [7, 3, 3, 1]:
            # 断言从 eic.f 中弹出的值符合预期
            assert eic.f.pop() == expected

    def test_torchbind_deepcopy(self):
        # 定义一个继承自 torch.nn.Module 的类 FooBar4321
        class FooBar4321(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 torch 脚本测试用的 PickleTester 实例，传入列表 [3, 4]
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

            # 前向传播方法，调用 self.f 的 top 方法
            def forward(self):
                return self.f.top()

        # 创建 FooBar4321 的实例
        inst = FooBar4321()
        # 对 inst 进行脚本化
        scripted = torch.jit.script(inst)
        # 对脚本化后的对象进行深拷贝
        copied = copy.deepcopy(scripted)
        # 断言拷贝对象的 forward 方法的返回值为 7
        assert copied.forward() == 7
        # 遍历期望的值列表 [7, 3, 3, 1]
        for expected in [7, 3, 3, 1]:
            # 断言从 copied.f 中弹出的值符合预期
            assert copied.f.pop() == expected

    def test_torchbind_python_deepcopy(self):
        # 定义一个继承自 torch.nn.Module 的类 FooBar4321
        class FooBar4321(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 torch 脚本测试用的 PickleTester 实例，传入列表 [3, 4]
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

            # 前向传播方法，调用 self.f 的 top 方法
            def forward(self):
                return self.f.top()

        # 创建 FooBar4321 的实例
        inst = FooBar4321()
        # 对实例进行 Python 对象的深拷贝
        copied = copy.deepcopy(inst)
        # 断言拷贝对象的调用结果为 7
        assert copied() == 7
        # 遍历期望的值列表 [7, 3, 3, 1]
        for expected in [7, 3, 3, 1]:
            # 断言从 copied.f 中弹出的值符合预期
            assert copied.f.pop() == expected

    def test_torchbind_tracing(self):
        # 定义一个 TryTracing 类，继承自 torch.nn.Module
        class TryTracing(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化一个 torch 脚本测试用的 PickleTester 实例，传入列表 [3, 4]
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

            # 前向传播方法，调用 torch.ops._TorchScriptTesting.take_an_instance 方法
            def forward(self):
                return torch.ops._TorchScriptTesting.take_an_instance(self.f)

        # 对 TryTracing 类进行跟踪
        traced = torch.jit.trace(TryTracing(), ())
        # 使用 self.assertEqual 进行断言，期望 torch.zeros(4, 4) 等于 traced() 的返回值
        self.assertEqual(torch.zeros(4, 4), traced())

    def test_torchbind_pass_wrong_type(self):
        # 使用 self.assertRaisesRegex 断言运行时错误，期望错误信息中包含 "but instead found type 'Tensor'"
        with self.assertRaisesRegex(RuntimeError, "but instead found type 'Tensor'"):
            # 调用 torch.ops._TorchScriptTesting.take_an_instance，传入一个形状为 (3, 4) 的随机张量
            torch.ops._TorchScriptTesting.take_an_instance(torch.rand(3, 4))
    # 定义一个测试函数，用于测试 torch.jit.trace 是否能够正确追踪嵌套类的对象方法调用
    def test_torchbind_tracing_nested(self):
        # 定义一个内部类 TryTracingNest，继承自 torch.nn.Module
        class TryTracingNest(torch.nn.Module):
            # 初始化方法，初始化一个 PickleTester 对象
            def __init__(self):
                super().__init__()
                self.f = torch.classes._TorchScriptTesting._PickleTester([3, 4])

        # 定义另一个内部类 TryTracing123，继承自 torch.nn.Module
        class TryTracing123(torch.nn.Module):
            # 初始化方法，初始化一个 TryTracingNest 的实例
            def __init__(self):
                super().__init__()
                self.nest = TryTracingNest()

            # 前向传播方法
            def forward(self):
                # 调用嵌套实例的 PickleTester 对象的方法
                return torch.ops._TorchScriptTesting.take_an_instance(self.nest.f)

        # 使用 torch.jit.trace 对 TryTracing123 类进行追踪
        traced = torch.jit.trace(TryTracing123(), ())
        # 断言追踪后的结果是否为一个 4x4 的零张量
        self.assertEqual(torch.zeros(4, 4), traced())

    # 定义一个测试函数，测试 torch.classes 对象的 pickle 序列化和反序列化
    def test_torchbind_pickle_serialization(self):
        # 创建一个 PickleTester 的实例 nt，包含数据 [3, 4]
        nt = torch.classes._TorchScriptTesting._PickleTester([3, 4])
        # 创建一个字节流对象 b
        b = io.BytesIO()
        # 将 nt 序列化到字节流 b 中
        torch.save(nt, b)
        # 将字节流 b 定位到起始位置
        b.seek(0)
        # 从字节流 b 中反序列化出一个新的对象 nt_loaded
        nt_loaded = torch.load(b)
        # 遍历期望值列表，依次断言反序列化出的对象 nt_loaded.pop() 是否与每个期望值相等
        for exp in [7, 3, 3, 1]:
            self.assertEqual(nt_loaded.pop(), exp)

    # 定义一个测试函数，测试当尝试实例化不存在的类时是否会抛出 RuntimeError 异常
    def test_torchbind_instantiate_missing_class(self):
        # 使用 assertRaisesRegex 上下文，期望捕获 RuntimeError 异常并检查错误消息
        with self.assertRaisesRegex(
            RuntimeError,
            "Tried to instantiate class 'foo.IDontExist', but it does not exist!",
        ):
            # 尝试实例化一个不存在的类 torch.classes.foo.IDontExist
            torch.classes.foo.IDontExist(3, 4, 5)

    # 定义一个测试函数，测试 TorchBindOptionalExplicitAttr 类中 Optional 类型属性的行为
    def test_torchbind_optional_explicit_attr(self):
        # 定义 TorchBindOptionalExplicitAttr 类，继承自 torch.nn.Module
        class TorchBindOptionalExplicitAttr(torch.nn.Module):
            # 声明一个类型为 Optional[_StackString] 的属性 foo
            foo: Optional[torch.classes._TorchScriptTesting._StackString]

            # 初始化方法
            def __init__(self):
                super().__init__()
                # 初始化 foo 属性为一个 _StackString 对象，包含数据 ["test"]
                self.foo = torch.classes._TorchScriptTesting._StackString(["test"])

            # 前向传播方法，返回 foo 对象的 pop() 结果，如果 foo 为 None 则返回 "<None>"
            def forward(self) -> str:
                foo_obj = self.foo
                if foo_obj is not None:
                    return foo_obj.pop()
                else:
                    return "<None>"

        # 创建 TorchBindOptionalExplicitAttr 类的实例 mod
        mod = TorchBindOptionalExplicitAttr()
        # 使用 torch.jit.script 对 mod 进行脚本化

    # 定义一个测试函数，测试当尝试实例化没有初始化方法的类时是否会抛出 RuntimeError 异常
    def test_torchbind_no_init(self):
        # 使用 assertRaisesRegex 上下文，期望捕获 RuntimeError 异常并检查错误消息
        with self.assertRaisesRegex(RuntimeError, "torch::init"):
            # 尝试实例化一个没有初始化方法的类 torch.classes._TorchScriptTesting._NoInit
            x = torch.classes._TorchScriptTesting._NoInit()

    # 定义一个测试函数，测试自定义操作在 autograd profiler 中是否正确记录
    def test_profiler_custom_op(self):
        # 创建一个 PickleTester 的实例 inst，包含数据 [3, 4]
        inst = torch.classes._TorchScriptTesting._PickleTester([3, 4])

        # 使用 torch.autograd.profiler.profile() 上下文记录性能分析信息
        with torch.autograd.profiler.profile() as prof:
            # 调用自定义操作 take_an_instance，并传入 inst 对象
            torch.ops._TorchScriptTesting.take_an_instance(inst)

        # 遍历性能分析事件，检查是否找到了名称为 "_TorchScriptTesting::take_an_instance" 的事件
        found_event = False
        for e in prof.function_events:
            if e.name == "_TorchScriptTesting::take_an_instance":
                found_event = True
        # 断言是否找到了相应事件
        self.assertTrue(found_event)

    # 定义一个测试函数，测试 torch.classes 对象的 getattr 方法
    def test_torchbind_getattr(self):
        # 创建一个 _StackString 的实例 foo，包含数据 ["test"]
        foo = torch.classes._TorchScriptTesting._StackString(["test"])
        # 使用 getattr 获取 foo 对象的属性 "bar"，如果属性不存在返回 None
        self.assertEqual(None, getattr(foo, "bar", None))

    # 定义一个测试函数，测试 torch.classes 对象的属性异常处理
    def test_torchbind_attr_exception(self):
        # 创建一个 _StackString 的实例 foo，包含数据 ["test"]
        foo = torch.classes._TorchScriptTesting._StackString(["test"])
        # 使用 assertRaisesRegex 上下文，期望捕获 AttributeError 异常并检查错误消息
        with self.assertRaisesRegex(AttributeError, "does not have a field"):
            # 尝试访问 foo 对象的不存在属性 "bar"
            foo.bar
    `
        def test_lambda_as_constructor(self):
            # 使用指定的参数调用 TorchScriptTesting._LambdaInit 构造函数，实例化 obj_no_swap 对象
            obj_no_swap = torch.classes._TorchScriptTesting._LambdaInit(4, 3, False)
            # 验证 obj_no_swap 对象调用 diff 方法的结果是否为 1
            self.assertEqual(obj_no_swap.diff(), 1)
    
            # 使用指定的参数调用 TorchScriptTesting._LambdaInit 构造函数，实例化 obj_swap 对象
            obj_swap = torch.classes._TorchScriptTesting._LambdaInit(4, 3, True)
            # 验证 obj_swap 对象调用 diff 方法的结果是否为 -1
            self.assertEqual(obj_swap.diff(), -1)
    
        def test_staticmethod(self):
            # 定义一个函数 fn，调用 TorchScriptTesting._StaticMethod 类的 staticMethod 静态方法，传入输入参数 inp
            def fn(inp: int) -> int:
                return torch.classes._TorchScriptTesting._StaticMethod.staticMethod(inp)
    
            # 使用 checkScript 方法检查 fn 函数的脚本
            self.checkScript(fn, (1,))
    
        def test_default_args(self):
            # 定义函数 fn，创建 TorchScriptTesting._DefaultArgs 对象，调用对象方法，并返回 increment 方法的结果
            def fn() -> int:
                obj = torch.classes._TorchScriptTesting._DefaultArgs()
                obj.increment(5)          # 调用 increment 方法，传入参数 5
                obj.decrement()            # 调用 decrement 方法，无参数
                obj.decrement(2)           # 调用 decrement 方法，传入参数 2
                obj.divide()               # 调用 divide 方法，无参数
                obj.scale_add(5)           # 调用 scale_add 方法，传入参数 5
                obj.scale_add(3, 2)         # 调用 scale_add 方法，传入参数 3 和 2
                obj.divide(3)              # 调用 divide 方法，传入参数 3
                return obj.increment()     # 调用 increment 方法，返回结果
    
            # 使用 checkScript 方法检查 fn 函数的脚本
            self.checkScript(fn, ())
    
            # 定义函数 gn，创建 TorchScriptTesting._DefaultArgs 对象，调用对象方法，并返回 decrement 方法的结果
            def gn() -> int:
                obj = torch.classes._TorchScriptTesting._DefaultArgs(5)  # 创建 TorchScriptTesting._DefaultArgs 对象，传入参数 5
                obj.increment(3)          # 调用 increment 方法，传入参数 3
                obj.increment()            # 调用 increment 方法，无参数
                obj.decrement(2)           # 调用 decrement 方法，传入参数 2
                obj.divide()               # 调用 divide 方法，无参数
                obj.scale_add(3)           # 调用 scale_add 方法，传入参数 3
                obj.scale_add(3, 2)         # 调用 scale_add 方法，传入参数 3 和 2
                obj.divide(2)              # 调用 divide 方法，传入参数 2
                return obj.decrement()     # 调用 decrement 方法，返回结果
    
            # 使用 checkScript 方法检查 gn 函数的脚本
            self.checkScript(gn, ())
```