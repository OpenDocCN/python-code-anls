# `.\pytorch\test\jit\test_save_load.py`

```
# Owner(s): ["oncall: jit"]

import io  # 导入用于处理字节流的模块
import os  # 导入操作系统相关功能的模块
import sys  # 导入系统相关的功能和参数
from pathlib import Path  # 导入处理路径的模块
from typing import NamedTuple, Optional  # 导入类型提示相关的功能

import torch  # 导入PyTorch深度学习库
from torch import Tensor  # 导入PyTorch张量
from torch.testing._internal.common_utils import skipIfTorchDynamo, TemporaryFileName  # 导入PyTorch测试工具相关功能

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # 获取当前测试文件所在目录的绝对路径
sys.path.append(pytorch_test_dir)  # 将当前测试文件所在目录添加到系统路径中，使得其中的辅助文件可以被导入
from torch.testing._internal.jit_utils import clear_class_registry, JitTestCase  # 从PyTorch测试工具中导入清除类注册表和JitTestCase类


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

class TestSaveLoad(JitTestCase):
    def test_different_modules(self):
        """
        Exercise the situation where we have the same qualified name
        in two different CompilationUnits on save/load.
        """

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)  # 创建一个包含2个输入和2个输出的线性层
                self.bar = torch.nn.Linear(2, 2)  # 创建另一个包含2个输入和2个输出的线性层

            def forward(self, x):
                x = self.foo(x)  # 执行第一个线性层的前向传播
                x = self.bar(x)  # 执行第二个线性层的前向传播
                return x

        first_script_module = torch.jit.script(Foo())  # 对Foo类进行脚本化
        first_saved_module = io.BytesIO()  # 创建一个用于存储字节流的对象
        torch.jit.save(first_script_module, first_saved_module)  # 将第一个脚本化模块保存到字节流中
        first_saved_module.seek(0)  # 将字节流的读取指针移到开头

        clear_class_registry()  # 清除类注册表，以准备加载新的模块

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)  # 创建一个包含2个输入和2个输出的线性层

            def forward(self, x):
                x = self.foo(x)  # 执行线性层的前向传播
                return x

        second_script_module = torch.jit.script(Foo())  # 对新的Foo类进行脚本化
        second_saved_module = io.BytesIO()  # 创建一个用于存储字节流的对象
        torch.jit.save(torch.jit.script(Foo()), second_saved_module)  # 将新的脚本化模块保存到字节流中
        second_saved_module.seek(0)  # 将字节流的读取指针移到开头

        clear_class_registry()  # 再次清除类注册表

        self.assertEqual(
            first_script_module._c.qualified_name,  # 比较第一个脚本化模块的限定名
            second_script_module._c.qualified_name,  # 比较第二个脚本化模块的限定名
        )

        class ContainsBoth(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.add_module("second", torch.jit.load(second_saved_module))  # 加载第二个保存的模块，并将其添加为当前模块的子模块
                self.add_module("first", torch.jit.load(first_saved_module))  # 加载第一个保存的模块，并将其添加为当前模块的子模块

            def forward(self, x):
                x = self.first(x)  # 执行第一个子模块的前向传播
                x = self.second(x)  # 执行第二个子模块的前向传播
                return x

        sm = torch.jit.script(ContainsBoth())  # 对包含两个子模块的类进行脚本化
        contains_both = io.BytesIO()  # 创建一个用于存储字节流的对象
        torch.jit.save(sm, contains_both)  # 将包含两个子模块的模块保存到字节流中
        contains_both.seek(0)  # 将字节流的读取指针移到开头
        sm = torch.jit.load(contains_both)  # 加载保存的模块
    def test_different_functions(self):
        """
        Exercise the situation where we have the same qualified name
        in two different CompilationUnits on save/load.
        """
        
        # 定义一个内部函数 lol，用于返回其输入
        def lol(x):
            return x
        
        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 定义该类的前向传播方法
            def forward(self, x):
                # 调用内部函数 lol 处理输入 x
                return lol(x)
        
        # 使用 torch.jit.script 将 Foo 类实例转换为脚本模块
        first_script_module = torch.jit.script(Foo())
        # 创建一个字节流对象 first_saved_module 用于保存模型
        first_saved_module = io.BytesIO()
        # 将脚本模块保存到字节流对象中
        torch.jit.save(first_script_module, first_saved_module)
        # 将字节流对象的读取位置设置为开头
        first_saved_module.seek(0)
        
        # 清空类注册表，用于后续重新定义类和函数
        clear_class_registry()
        
        # 重新定义内部函数 lol，此时返回固定字符串 "hello"
        def lol(x):  # noqa: F811
            return "hello"
        
        # 重新定义类 Foo，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 重新定义该类的前向传播方法
            def forward(self, x):
                # 调用重新定义的 lol 函数处理输入 x
                return lol(x)
        
        # 使用 torch.jit.script 将重新定义后的 Foo 类实例转换为脚本模块
        second_script_module = torch.jit.script(Foo())
        # 创建一个字节流对象 second_saved_module 用于保存模型
        second_saved_module = io.BytesIO()
        # 将重新定义后的脚本模块保存到字节流对象中
        torch.jit.save(torch.jit.script(Foo()), second_saved_module)
        # 将字节流对象的读取位置设置为开头
        second_saved_module.seek(0)
        
        # 清空类注册表，用于后续重新定义类
        clear_class_registry()
        
        # 断言两个脚本模块的完全限定名称相同
        self.assertEqual(
            first_script_module._c.qualified_name,
            second_script_module._c.qualified_name,
        )
        
        # 定义一个包含两个模型的类 ContainsBoth，继承自 torch.nn.Module
        class ContainsBoth(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加第二个保存的模型作为子模块 "second"
                self.add_module("second", torch.jit.load(second_saved_module))
                # 添加第一个保存的模型作为子模块 "first"
                self.add_module("first", torch.jit.load(first_saved_module))
            
            # 定义前向传播方法
            def forward(self, x):
                # 对输入 x 分别调用两个子模块的前向传播方法
                x = self.first(x)
                x = self.second(x)
                return x
        
        # 使用 torch.jit.script 将 ContainsBoth 类实例转换为脚本模块
        sm = torch.jit.script(ContainsBoth())
        # 创建一个字节流对象 contains_both 用于保存模型
        contains_both = io.BytesIO()
        # 将脚本模块保存到字节流对象中
        torch.jit.save(sm, contains_both)
        # 将字节流对象的读取位置设置为开头
        contains_both.seek(0)
        # 使用 torch.jit.load 加载保存的脚本模块
        sm = torch.jit.load(contains_both)
    def test_different_interfaces(self):
        """
        Exercise the situation where we have the same qualified name
        in two different CompilationUnits on save/load.
        """

        # 定义一个 TorchScript 接口 MyInterface，包含方法 bar
        @torch.jit.interface
        class MyInterface:
            def bar(self, x: Tensor) -> Tensor:
                pass

        # 使用 TorchScript 的 @torch.jit.script 装饰器，定义实现了 MyInterface 接口的类 ImplementInterface
        @torch.jit.script
        class ImplementInterface:
            def __init__(self):
                pass

            def bar(self, x):
                return x

        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 设置类属性 __annotations__，指定接口为 MyInterface
            __annotations__ = {"interface": MyInterface}

            def __init__(self):
                super().__init__()
                # 实例化 ImplementInterface 类，作为 interface 的实现
                self.interface = ImplementInterface()

            # 实现 Module 类的 forward 方法
            def forward(self, x):
                return self.interface.bar(x)

        # 将 Foo 类实例化并编译成 TorchScript 模块
        first_script_module = torch.jit.script(Foo())
        # 创建一个 BytesIO 对象用于保存 TorchScript 模块的序列化结果
        first_saved_module = io.BytesIO()
        # 将 TorchScript 模块保存到 first_saved_module 中
        torch.jit.save(first_script_module, first_saved_module)
        # 将文件指针移动到起始位置
        first_saved_module.seek(0)

        # 清空 TorchScript 类注册表中的缓存
        clear_class_registry()

        # 重新定义 MyInterface 接口，此次定义与之前不同
        @torch.jit.interface
        class MyInterface:
            def not_bar(self, x: Tensor) -> Tensor:
                pass

        # 使用 @torch.jit.script 装饰器，定义实现了新 MyInterface 接口的类 ImplementInterface
        @torch.jit.script  # noqa: F811
        class ImplementInterface:  # noqa: F811
            def __init__(self):
                pass

            def not_bar(self, x):
                return x

        # 定义另一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 设置类属性 __annotations__，指定接口为新定义的 MyInterface
            __annotations__ = {"interface": MyInterface}

            def __init__(self):
                super().__init__()
                # 实例化 ImplementInterface 类，作为 interface 的实现
                self.interface = ImplementInterface()

            # 实现 Module 类的 forward 方法
            def forward(self, x):
                return self.interface.not_bar(x)

        # 将新定义的 Foo 类实例化并编译成 TorchScript 模块
        second_script_module = torch.jit.script(Foo())
        # 创建一个 BytesIO 对象用于保存 TorchScript 模块的序列化结果
        second_saved_module = io.BytesIO()
        # 将 TorchScript 模块保存到 second_saved_module 中
        torch.jit.save(torch.jit.script(Foo()), second_saved_module)
        # 将文件指针移动到起始位置
        second_saved_module.seek(0)

        # 清空 TorchScript 类注册表中的缓存
        clear_class_registry()

        # 断言两个 TorchScript 模块的 qualified_name 属性相同
        self.assertEqual(
            first_script_module._c.qualified_name,
            second_script_module._c.qualified_name,
        )

        # 定义一个继承自 torch.nn.Module 的类 ContainsBoth
        class ContainsBoth(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加第一个 TorchScript 模块到该类中，并命名为 "first"
                self.add_module("second", torch.jit.load(second_saved_module))
                # 添加第二个 TorchScript 模块到该类中，并命名为 "second"
                self.add_module("first", torch.jit.load(first_saved_module))

            # 实现 Module 类的 forward 方法
            def forward(self, x):
                x = self.first(x)
                x = self.second(x)
                return x

        # 将 ContainsBoth 类实例化并编译成 TorchScript 模块
        sm = torch.jit.script(ContainsBoth())
        # 创建一个 BytesIO 对象用于保存 TorchScript 模块的序列化结果
        contains_both = io.BytesIO()
        # 将 TorchScript 模块保存到 contains_both 中
        torch.jit.save(sm, contains_both)
        # 将文件指针移动到起始位置
        contains_both.seek(0)
        # 从 contains_both 中加载 TorchScript 模块
        sm = torch.jit.load(contains_both)
    def test_save_load_with_extra_files(self):
        # 定义一个继承自torch.jit.ScriptModule的子类MyMod，用于测试
        class MyMod(torch.jit.ScriptModule):
            # 定义一个脚本方法forward，接受参数a并返回a
            @torch.jit.script_method
            def forward(self, a):
                return a

        # specifically test binary data
        # 定义一个二进制数据作为测试值
        value = b"bar\x00\xffbaz"

        # 初始化预期的额外文件字典
        expected_extra_files = {}
        expected_extra_files["foo"] = value
        # verify that str to bytes conversion also works
        # 验证字符串到字节转换是否正常工作
        expected_extra_files["foo2"] = "bar"
        m = MyMod()

        # Save to file.
        # 使用临时文件名保存模型
        with TemporaryFileName() as fname:
            m.save(fname, _extra_files=expected_extra_files)
            # values don't matter
            # 初始化额外文件字典，用于加载模型时的比较
            extra_files = {"foo": "", "foo2": None}
            torch.jit.load(fname, _extra_files=extra_files)
            self.assertEqual(value, extra_files["foo"])
            # results come back always as bytes
            # 确保结果始终以字节返回
            self.assertEqual(b"bar", extra_files["foo2"])

            # Use torch.jit API
            # 使用torch.jit API保存模型
            torch.jit.save(m, fname, _extra_files=expected_extra_files)
            extra_files["foo"] = ""
            torch.jit.load(fname, _extra_files=extra_files)
            self.assertEqual(value, extra_files["foo"])

        # Save to buffer.
        # 将模型保存到缓冲区
        buffer = io.BytesIO(m.save_to_buffer(_extra_files=expected_extra_files))
        extra_files = {"foo": ""}
        torch.jit.load(buffer, _extra_files=extra_files)
        self.assertEqual(value, extra_files["foo"])

        # Use torch.jit API
        # 使用torch.jit API保存模型到缓冲区
        buffer = io.BytesIO()
        torch.jit.save(m, buffer, _extra_files=expected_extra_files)
        buffer.seek(0)
        extra_files = {"foo": ""}
        torch.jit.load(buffer, _extra_files=extra_files)
        self.assertEqual(value, extra_files["foo"])

        # Non-existent file 'bar'
        # 测试加载不存在的文件时是否抛出RuntimeError
        with self.assertRaises(RuntimeError):
            extra_files["bar"] = ""
            torch.jit.load(buffer, _extra_files=extra_files)
    def test_save_namedtuple_input_only(self):
        """
        Even if a NamedTuple is only used as an input argument, saving and
        loading should work correctly.
        """
        global FooTuple  # 全局变量声明，用于在局部解析中引用

        class FooTuple(NamedTuple):
            a: int  # 定义一个名为FooTuple的命名元组，包含整型字段a

        class MyModule(torch.nn.Module):
            def forward(self, x: FooTuple) -> torch.Tensor:
                return torch.tensor(3)  # 返回一个值为3的张量

        m_loaded = self.getExportImportCopy(torch.jit.script(MyModule()))
        output = m_loaded(FooTuple(a=5))  # 调用模型，传入FooTuple的实例
        self.assertEqual(output, torch.tensor(3))  # 断言输出张量与预期的张量3相等

    def test_save_namedtuple_input_only_forwardref(self):
        """
        Even if a NamedTuple is only used as an input argument, saving and
        loading should work correctly.
        """
        global FooTuple  # 全局变量声明，用于在局部解析中引用

        class FooTuple(NamedTuple):
            a: "int"  # 定义一个名为FooTuple的命名元组，包含字段a，类型为字符串形式的int

        class MyModule(torch.nn.Module):
            def forward(self, x: FooTuple) -> torch.Tensor:
                return torch.tensor(3)  # 返回一个值为3的张量

        m_loaded = self.getExportImportCopy(torch.jit.script(MyModule()))
        output = m_loaded(FooTuple(a=5))  # 调用模型，传入FooTuple的实例
        self.assertEqual(output, torch.tensor(3))  # 断言输出张量与预期的张量3相等

    def test_save_namedtuple_output_only(self):
        """
        Even if a NamedTuple is only used as an output argument, saving and
        loading should work correctly.
        """
        global FooTuple  # 全局变量声明，用于在局部解析中引用

        class FooTuple(NamedTuple):
            a: int  # 定义一个名为FooTuple的命名元组，包含整型字段a

        class MyModule(torch.nn.Module):
            def forward(self) -> Optional[FooTuple]:
                return None  # 返回空值作为可选的FooTuple类型

        m_loaded = self.getExportImportCopy(torch.jit.script(MyModule()))
        output = m_loaded()  # 调用模型，获取输出
        self.assertEqual(output, None)  # 断言输出为None
    def test_save_load_params_buffers_submodules(self):
        """
        Check that parameters, buffers, and submodules are the same after loading.
        """

        # 定义一个子模块
        class Submodule(torch.nn.Module):
            pass

        # 定义一个测试模块
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加子模块 "submodule_a"
                self.add_module("submodule_a", Submodule())
                # 注册参数 "parameter_a"，并初始化为随机张量
                self.register_parameter(
                    "parameter_a", torch.nn.Parameter(torch.randn(4))
                )
                # 注册缓冲区 "buffer"，并初始化为随机张量
                self.register_buffer("buffer", torch.randn(4))
                # 直接赋值张量 "t"，不是缓冲区
                self.t = torch.rand(4)  # not buffer

                # 直接定义参数 "parameter_b"，并初始化为随机张量
                self.parameter_b = torch.nn.Parameter(torch.randn(4))
                # 添加子模块 "submodule_b"
                self.submodule_b = Submodule()

        # 创建测试模块实例
        m = TestModule()
        # 使用 TorchScript 进行模块的序列化与反序列化
        m_loaded = self.getExportImportCopy(torch.jit.script(m))

        # 检查子模块
        self.assertEqual(
            len(list(m.named_modules())), len(list(m_loaded.named_modules()))
        )
        for m_s, loaded_s in zip(m.named_modules(), m_loaded.named_modules()):
            m_name, _ = m_s
            loaded_name, _ = loaded_s
            self.assertEqual(m_name, loaded_name)

        # 检查参数
        self.assertEqual(len(list(m.parameters())), len(list(m_loaded.parameters())))
        for m_p, loaded_p in zip(m.parameters(), m_loaded.parameters()):
            self.assertEqual(m_p, loaded_p)

        # 检查缓冲区
        self.assertEqual(
            len(list(m.named_buffers())), len(list(m_loaded.named_buffers()))
        )
        for m_b, loaded_b in zip(m.named_buffers(), m_loaded.named_buffers()):
            m_name, m_buffer = m_b
            loaded_name, loaded_buffer = loaded_b
            self.assertEqual(m_name, loaded_name)
            self.assertEqual(m_buffer, loaded_buffer)
    def test_save_load_meta_tensors(self):
        """
        Check that parameters, buffers, and submodules are the same after loading
        for a module with parameters and buffers that are meta tensors
        """

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Define a linear layer with input size 2 and output size 3, marked as a meta tensor
                self.foo = torch.nn.Linear(2, 3, device="meta")
                # Define another linear layer with input size 3 and output size 4
                self.bar = torch.nn.Linear(3, 4)
                # Register a buffer tensor of shape (4,) marked as a meta tensor
                self.register_buffer("buffer", torch.randn(4, device="meta"))

            def forward(self, x):
                # Forward pass through the network: first foo, then bar
                x = self.foo(x)
                x = self.bar(x)
                return x

        # Create an instance of Foo
        m = Foo()
        # Load a copy of Foo using torch.jit.script and the self.getExportImportCopy function
        m_loaded = self.getExportImportCopy(torch.jit.script(m))
        # Check submodules: compare the number and names of submodules in m and m_loaded
        self.assertEqual(
            len(list(m.named_modules())), len(list(m_loaded.named_modules()))
        )
        self.assertEqual(
            {name for name, _ in m.named_modules()},
            {name for name, _ in m_loaded.named_modules()},
        )
        # Check parameters: compare parameters of m and m_loaded
        m_params = dict(m.named_parameters())
        m_loaded_params = dict(m_loaded.named_parameters())
        self.assertEqual(len(m_params), len(m_loaded_params))
        self.assertEqual(m_params, m_loaded_params)
        # Check buffers: compare buffers of m and m_loaded
        m_buffers = dict(m.named_buffers())
        m_loaded_buffers = dict(m_loaded.named_buffers())
        self.assertEqual(len(m_buffers), len(m_loaded_buffers))
        self.assertEqual(m_buffers, m_loaded_buffers)
        # Check meta tensor properties for specific parameters and buffers
        self.assertTrue(m_params["foo.weight"].is_meta)
        self.assertTrue(m_loaded_params["foo.weight"].is_meta)
        self.assertTrue(m_params["foo.bias"].is_meta)
        self.assertTrue(m_loaded_params["foo.bias"].is_meta)
        self.assertFalse(m_params["bar.weight"].is_meta)
        self.assertFalse(m_loaded_params["bar.weight"].is_meta)
        self.assertFalse(m_params["bar.bias"].is_meta)
        self.assertFalse(m_loaded_params["bar.bias"].is_meta)
        self.assertTrue(m_buffers["buffer"].is_meta)
        self.assertTrue(m_loaded_buffers["buffer"].is_meta)
    def test_save_load_meta_tensors_to_device(self):
        """
        检查加载带有元张量到设备的模块时，元张量保持在元设备上，非元张量设置为指定的设备。
        """

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 在初始化中定义神经网络层，指定设备为"meta"
                self.foo = torch.nn.Linear(2, 3, device="meta")
                self.bar = torch.nn.Linear(3, 4)

            def forward(self, x):
                # 前向传播函数
                x = self.foo(x)
                x = self.bar(x)
                return x

        m = Foo()

        # 使用自定义函数getExportImportCopy导出和导入模型，设定目标设备为"cpu"
        m_loaded = self.getExportImportCopy(torch.jit.script(m), map_location="cpu")
        
        # 检查子模块
        self.assertEqual(
            len(list(m.named_modules())), len(list(m_loaded.named_modules()))
        )
        self.assertEqual(
            {name for name, _ in m.named_modules()},
            {name for name, _ in m_loaded.named_modules()},
        )
        
        # 检查参数
        m_params = dict(m.named_parameters())
        m_loaded_params = dict(m_loaded.named_parameters())
        self.assertEqual(len(m_params), len(m_loaded_params))
        self.assertEqual(m_params, m_loaded_params)
        
        # 检查是/否为元张量的参数和缓冲区
        self.assertTrue(m_params["foo.weight"].is_meta)
        self.assertTrue(m_loaded_params["foo.weight"].is_meta)
        self.assertTrue(m_params["foo.bias"].is_meta)
        self.assertTrue(m_loaded_params["foo.bias"].is_meta)
        self.assertTrue(m_params["bar.weight"].is_cpu)
        self.assertTrue(m_loaded_params["bar.weight"].is_cpu)
        self.assertTrue(m_params["bar.bias"].is_cpu)
        self.assertTrue(m_loaded_params["bar.bias"].is_cpu)

    @skipIfTorchDynamo("too slow")
    def test_save_load_large_string_attribute(self):
        """
        检查是否能加载具有大于4GB字符串的模型。
        """
        import psutil

        if psutil.virtual_memory().available < 60 * 1024 * 1024 * 1024:
            # 经过测试执行的分析，得到这个数字可以安全运行测试
            self.skipTest(
                "Doesn't have enough memory to run test_save_load_large_string_attribute"
            )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化时定义一个超过4GB的字符串
                self.x = "x" * (2**32 + 1)

            def forward(self, i) -> int:
                # 前向传播函数返回长度和输入张量元素数之和
                return len(self.x) + i.numel()

        inp = torch.ones(0)
        ts = torch.jit.script(Model())
        ts_output = ts(inp)

        # 将模型保存到字节流中
        b = io.BytesIO(ts.save_to_buffer())
        del ts

        # 从字节流加载模型
        loaded_ts = torch.jit.load(b)
        del b
        
        # 对加载后的模型进行前向传播
        loaded_output = loaded_ts(inp)
        self.assertEqual(ts_output, loaded_output)
# 定义一个函数，将 TorchScript 模块保存到字节流缓冲区中
def script_module_to_buffer(script_module):
    # 调用模块的 _save_to_buffer_for_lite_interpreter 方法，使用 FlatBuffer 格式保存模块到字节流
    module_buffer = io.BytesIO(
        script_module._save_to_buffer_for_lite_interpreter(_use_flatbuffer=True)
    )
    # 将字节流的指针移动到起始位置
    module_buffer.seek(0)
    # 返回字节流缓冲区
    return module_buffer


# 定义一个测试类 TestSaveLoadFlatbuffer，继承自 JitTestCase
class TestSaveLoadFlatbuffer(JitTestCase):

    # 定义测试方法 test_different_modules
    def test_different_modules(self):
        """
        Exercise the situation where we have the same qualified name
        in two different CompilationUnits on save/load.
        """

        # 定义一个名为 Foo 的内部类，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)
                self.bar = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = self.foo(x)
                x = self.bar(x)
                return x

        # 使用 torch.jit.script 将 Foo 类实例化为 TorchScript 模块
        first_script_module = torch.jit.script(Foo())
        # 将第一个脚本化模块保存到字节流中
        first_saved_module = script_module_to_buffer(first_script_module)

        # 清空类注册表
        clear_class_registry()

        # 重新定义名为 Foo 的内部类，与上面不同
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = self.foo(x)
                return x

        # 使用 torch.jit.script 将新的 Foo 类实例化为 TorchScript 模块
        second_script_module = torch.jit.script(Foo())
        # 将第二个脚本化模块保存到字节流中
        second_saved_module = script_module_to_buffer(second_script_module)

        # 清空类注册表
        clear_class_registry()

        # 断言两个脚本化模块的限定名相同
        self.assertEqual(
            first_script_module._c.qualified_name,
            second_script_module._c.qualified_name,
        )

        # 定义一个名为 ContainsBoth 的内部类，继承自 torch.nn.Module
        class ContainsBoth(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加第二个保存的模块作为模块 "second"
                self.add_module("second", torch.jit.load(second_saved_module))
                # 添加第一个保存的模块作为模块 "first"
                self.add_module("first", torch.jit.load(first_saved_module))

            def forward(self, x):
                # 对第一个模块执行前向传播
                x = self.first(x)
                # 对第二个模块执行前向传播
                x = self.second(x)
                return x

        # 使用 torch.jit.script 将 ContainsBoth 类实例化为 TorchScript 模块
        sm = torch.jit.script(ContainsBoth())
        # 将包含两个模块的 TorchScript 模块保存到字节流中
        contains_both = script_module_to_buffer(sm)
        # 加载保存的 TorchScript 模块
        sm = torch.jit.load(contains_both)
    def test_different_functions(self):
        """
        Exercise the situation where we have the same qualified name
        in two different CompilationUnits on save/load.
        """

        # 定义一个简单的函数 lol，返回其输入参数 x
        def lol(x):
            return x

        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 定义类中的 forward 方法，调用上面定义的 lol 函数
            def forward(self, x):
                return lol(x)

        # 使用 torch.jit.script 将 Foo 类实例转换为脚本模块
        first_script_module = torch.jit.script(Foo())
        # 将第一个脚本模块转换为字节流表示
        first_saved_module = script_module_to_buffer(first_script_module)
        # 清空类注册表
        clear_class_registry()

        # 重新定义 lol 函数，此时返回固定字符串 "hello"
        def lol(x):  # noqa: F811
            return "hello"

        # 重新定义类 Foo，其 forward 方法使用新定义的 lol 函数
        class Foo(torch.nn.Module):
            def forward(self, x):
                return lol(x)

        # 使用 torch.jit.script 将新定义的 Foo 类实例转换为脚本模块
        second_script_module = torch.jit.script(Foo())
        # 将第二个脚本模块转换为字节流表示
        second_saved_module = script_module_to_buffer(second_script_module)

        # 再次清空类注册表
        clear_class_registry()

        # 断言第一个脚本模块和第二个脚本模块的 qualified_name 相同
        self.assertEqual(
            first_script_module._c.qualified_name,
            second_script_module._c.qualified_name,
        )

        # 定义一个包含两个脚本模块的类 ContainsBoth
        class ContainsBoth(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加第二个脚本模块到模块中，并命名为 "second"
                self.add_module("second", torch.jit.load(second_saved_module))
                # 添加第一个脚本模块到模块中，并命名为 "first"
                self.add_module("first", torch.jit.load(first_saved_module))

            # 定义类中的 forward 方法，依次调用模块 "first" 和 "second" 的 forward 方法
            def forward(self, x):
                x = self.first(x)
                x = self.second(x)
                return x

        # 使用 torch.jit.script 将 ContainsBoth 类实例转换为脚本模块
        sm = torch.jit.script(ContainsBoth())
        # 将包含两个脚本模块的模块再次转换为字节流表示
        contains_both = script_module_to_buffer(sm)
        # 使用 torch.jit.load 加载最终的脚本模块
        sm = torch.jit.load(contains_both)
    def test_different_interfaces(self):
        """
        Exercise the situation where we have the same qualified name
        in two different CompilationUnits on save/load.
        """

        # 定义一个 TorchScript 接口 MyInterface
        @torch.jit.interface
        class MyInterface:
            # 接口方法 bar，接收一个 Tensor 类型参数 x，返回一个 Tensor
            def bar(self, x: Tensor) -> Tensor:
                pass

        # 通过 TorchScript 的脚本模式定义实现了 MyInterface 接口的类 ImplementInterface
        @torch.jit.script
        class ImplementInterface:
            def __init__(self):
                pass

            # 实现接口方法 bar，参数 x 是一个动态类型
            def bar(self, x):
                return x

        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 注解指定类属性 "interface" 为 MyInterface 类型
            __annotations__ = {"interface": MyInterface}

            def __init__(self):
                super().__init__()
                # 实例化 ImplementInterface 类，并赋值给 interface 属性
                self.interface = ImplementInterface()

            # 前向传播方法，调用 interface 对象的 bar 方法
            def forward(self, x):
                return self.interface.bar(x)

        # 使用 TorchScript 将 Foo 类实例转换为脚本模块
        first_script_module = torch.jit.script(Foo())
        # 将第一个脚本模块转换为二进制缓冲
        first_saved_module = script_module_to_buffer(first_script_module)
        # 清除类注册表中的信息
        clear_class_registry()

        # 重新定义 MyInterface 接口，这次定义的方法名为 not_bar
        @torch.jit.interface
        class MyInterface:
            # 新的接口方法 not_bar，接收一个 Tensor 类型参数 x，返回一个 Tensor
            def not_bar(self, x: Tensor) -> Tensor:
                pass

        # 通过 TorchScript 的脚本模式重新定义实现了 MyInterface 接口的类 ImplementInterface
        @torch.jit.script  # noqa: F811
        class ImplementInterface:  # noqa: F811
            def __init__(self):
                pass

            # 实现接口方法 not_bar，参数 x 是一个动态类型
            def not_bar(self, x):
                return x

        # 定义另一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 注解指定类属性 "interface" 为重新定义的 MyInterface 类型
            __annotations__ = {"interface": MyInterface}

            def __init__(self):
                super().__init__()
                # 实例化重新定义的 ImplementInterface 类，并赋值给 interface 属性
                self.interface = ImplementInterface()

            # 前向传播方法，调用 interface 对象的 not_bar 方法
            def forward(self, x):
                return self.interface.not_bar(x)

        # 使用 TorchScript 将另一个 Foo 类实例转换为脚本模块
        second_script_module = torch.jit.script(Foo())
        # 将第二个脚本模块转换为二进制缓冲
        second_saved_module = script_module_to_buffer(second_script_module)

        # 清除类注册表中的信息
        clear_class_registry()

        # 断言两个脚本模块的 qualified_name 属性相等
        self.assertEqual(
            first_script_module._c.qualified_name,
            second_script_module._c.qualified_name,
        )

        # 定义一个继承自 torch.nn.Module 的类 ContainsBoth
        class ContainsBoth(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加第二个保存的脚本模块为子模块 "second"
                self.add_module("second", torch.jit.load(second_saved_module))
                # 添加第一个保存的脚本模块为子模块 "first"
                self.add_module("first", torch.jit.load(first_saved_module))

            # 前向传播方法，依次调用子模块 first 和 second 的 forward 方法
            def forward(self, x):
                x = self.first(x)
                x = self.second(x)
                return x

        # 使用 TorchScript 将 ContainsBoth 类实例转换为脚本模块
        sm = torch.jit.script(ContainsBoth())
        # 将包含两个子模块的脚本模块转换为二进制缓冲
        contains_both = script_module_to_buffer(sm)
        # 加载包含两个子模块的脚本模块
        sm = torch.jit.load(contains_both)
    def test_many_collisions(self):
        # 定义一个命名元组类型 MyCoolNamedTuple，包含一个整数字段 a
        class MyCoolNamedTuple(NamedTuple):
            a: int

        # 定义一个 TorchScript 接口 MyInterface，包含一个 bar 方法，接受一个张量类型参数 x，返回一个张量类型结果
        @torch.jit.interface
        class MyInterface:
            def bar(self, x: Tensor) -> Tensor:
                pass

        # 使用 TorchScript 来声明一个类 ImplementInterface，实现 MyInterface 接口中的 bar 方法
        @torch.jit.script
        class ImplementInterface:
            def __init__(self):
                pass

            def bar(self, x):
                return x

        # 定义一个普通的函数 lol，接受一个参数 x，返回其本身
        def lol(x):
            return x

        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 类属性 interface，类型为 MyInterface
            interface: MyInterface

            def __init__(self):
                super().__init__()
                # 实例化一个线性层对象 self.foo，输入维度 2，输出维度 2
                self.foo = torch.nn.Linear(2, 2)
                # 实例化一个线性层对象 self.bar，输入维度 2，输出维度 2
                self.bar = torch.nn.Linear(2, 2)
                # 实例化 ImplementInterface 类的对象，并赋值给 self.interface
                self.interface = ImplementInterface()

            # 定义模型的前向传播方法 forward，接受一个输入张量 x
            def forward(self, x):
                # 将输入张量 x 传入 self.foo 线性层，得到输出 x
                x = self.foo(x)
                # 将输出 x 传入 self.bar 线性层，得到输出 x
                x = self.bar(x)
                # 将输出 x 传入 lol 函数，得到输出 x
                x = lol(x)
                # 将输出 x 传入 self.interface 的 bar 方法，得到输出结果，同时返回一个 MyCoolNamedTuple 命名元组对象
                x = self.interface.bar(x)

                return x, MyCoolNamedTuple(a=5)

        # 使用 TorchScript 将 Foo 类实例化为脚本模块对象 first_script_module
        first_script_module = torch.jit.script(Foo())
        # 将 first_script_module 转换为字节缓冲区对象 first_saved_module
        first_saved_module = script_module_to_buffer(first_script_module)

        # 清除 TorchScript 类注册表中的所有类信息
        clear_class_registry()

        # 重新定义 MyInterface 接口，包含一个 not_bar 方法，接受一个张量类型参数 x，返回一个张量类型结果
        @torch.jit.interface
        class MyInterface:
            def not_bar(self, x: Tensor) -> Tensor:
                pass

        # 使用 TorchScript 来声明一个类 ImplementInterface，实现 MyInterface 接口中的 not_bar 方法
        @torch.jit.script  # noqa: F811
        class ImplementInterface:  # noqa: F811
            def __init__(self):
                pass

            def not_bar(self, x):
                return x

        # 重新定义函数 lol，接受一个参数 x，返回字符串 "asdofij"
        def lol(x):  # noqa: F811
            return "asdofij"

        # 重新定义 MyCoolNamedTuple 类型，其字段 a 的类型为字符串
        class MyCoolNamedTuple(NamedTuple):  # noqa: F811
            a: str

        # 定义一个继承自 torch.nn.Module 的类 Foo
        class Foo(torch.nn.Module):
            # 类属性 interface，类型为 MyInterface
            interface: MyInterface

            def __init__(self):
                super().__init__()
                # 实例化一个线性层对象 self.foo，输入维度 2，输出维度 2
                self.foo = torch.nn.Linear(2, 2)
                # 实例化 ImplementInterface 类的对象，并赋值给 self.interface
                self.interface = ImplementInterface()

            # 定义模型的前向传播方法 forward，接受一个输入张量 x
            def forward(self, x):
                # 将输入张量 x 传入 self.foo 线性层，得到输出 x
                x = self.foo(x)
                # 将输出 x 传入 self.interface 的 not_bar 方法，得到输出结果，同时返回一个 MyCoolNamedTuple 命名元组对象
                self.interface.not_bar(x)
                # 将输出 x 传入 lol 函数，得到输出 x
                return x, MyCoolNamedTuple(a="hello")

        # 使用 TorchScript 将 Foo 类实例化为脚本模块对象 second_script_module
        second_script_module = torch.jit.script(Foo())
        # 将 second_script_module 转换为字节缓冲区对象 second_saved_module
        second_saved_module = script_module_to_buffer(second_script_module)

        # 清除 TorchScript 类注册表中的所有类信息
        clear_class_registry()

        # 使用 self.assertEqual 检查两个脚本模块对象的合格名称是否相同
        self.assertEqual(
            first_script_module._c.qualified_name,
            second_script_module._c.qualified_name,
        )

        # 定义一个继承自 torch.nn.Module 的类 ContainsBoth
        class ContainsBoth(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 将 second_saved_module 和 first_saved_module 加载为模块对象，并分别命名为 "second" 和 "first"
                self.add_module("second", torch.jit.load(second_saved_module))
                self.add_module("first", torch.jit.load(first_saved_module))

            # 定义模型的前向传播方法 forward，接受一个输入张量 x
            def forward(self, x):
                # 分别调用 self.first 和 self.second 模块对象的 forward 方法，得到输出 x 和两个命名元组对象
                x, named_tuple_1 = self.first(x)
                x, named_tuple_2 = self.second(x)
                # 返回一个整数结果，表示两个命名元组 a 字段长度之和和 named_tuple_1 的 a 字段值之和
                return len(x + named_tuple_2.a) + named_tuple_1.a

        # 使用 TorchScript 将 ContainsBoth 类实例化为脚本模块对象 sm
        sm = torch.jit.script(ContainsBoth())
        # 将 sm 转换为字节缓冲区对象 contains_both
        contains_both = script_module_to_buffer(sm)
        # 使用 torch.jit.load 将字节缓冲区对象 contains_both 加载为 TorchScript 模块对象 sm
        sm = torch.jit.load(contains_both)
    def test_save_load_using_pathlib(self):
        class MyMod(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                return 2 * a

        m = MyMod()

        # Save then load.
        with TemporaryFileName() as fname:
            # 使用临时文件名创建路径对象
            path = Path(fname)
            # 将模型 m 保存为 FlatBuffer 格式到指定路径
            torch.jit.save_jit_module_to_flatbuffer(m, path)
            # 从路径加载 FlatBuffer 格式的模型到 m2
            m2 = torch.jit.load(path)

        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # 断言 m 和 m2 在相同输入下输出相等的结果张量
        self.assertTrue(torch.equal(m(x), m2(x)))

    def test_save_namedtuple_input_only(self):
        """
        Even if a NamedTuple is only used as an input argument, saving and
        loading should work correctly.
        """
        global FooTuple  # see [local resolution in python]

        class FooTuple(NamedTuple):
            a: int

        class MyModule(torch.nn.Module):
            def forward(self, x: FooTuple) -> torch.Tensor:
                return torch.tensor(3)

        m_loaded = self.getExportImportCopy(torch.jit.script(MyModule()))
        output = m_loaded(FooTuple(a=5))
        # 断言输出结果与预期的张量相等
        self.assertEqual(output, torch.tensor(3))

    def test_save_namedtuple_output_only(self):
        """
        Even if a NamedTuple is only used as an output argument, saving and
        loading should work correctly.
        """
        global FooTuple  # see [local resolution in python]

        class FooTuple(NamedTuple):
            a: int

        class MyModule(torch.nn.Module):
            def forward(self) -> Optional[FooTuple]:
                return None

        m_loaded = self.getExportImportCopy(torch.jit.script(MyModule()))
        output = m_loaded()
        # 断言输出结果为 None
        self.assertEqual(output, None)

    def test_module_info_flatbuffer(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)
                self.bar = torch.nn.Linear(2, 2)

            def forward(self, x):
                x = self.foo(x)
                x = self.bar(x)
                return x

        first_script_module = torch.jit.script(Foo())
        first_saved_module = io.BytesIO()
        # 将脚本化的模型保存为 FlatBuffer 格式到内存中的 BytesIO 对象
        torch.jit.save_jit_module_to_flatbuffer(first_script_module, first_saved_module)
        first_saved_module.seek(0)
        # 获取保存的 FlatBuffer 模型信息
        ff_info = torch.jit._serialization.get_flatbuffer_module_info(
            first_saved_module
        )
        # 断言 FlatBuffer 模型信息的各项属性符合预期
        self.assertEqual(ff_info["bytecode_version"], 9)
        self.assertEqual(ff_info["operator_version"], 1)
        self.assertEqual(ff_info["type_names"], set())
        self.assertEqual(ff_info["opname_to_num_args"], {"aten::linear": 3})

        self.assertEqual(len(ff_info["function_names"]), 1)
        self.assertTrue(next(iter(ff_info["function_names"])).endswith("forward"))
    def test_save_load_params_buffers_submodules(self):
        """
        Check that parameters, buffers, and submodules are the same after loading.
        """

        class Submodule(torch.nn.Module):
            pass

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 添加子模块 "submodule_a"
                self.add_module("submodule_a", Submodule())
                # 注册参数 "parameter_a"，使用随机初始化的参数
                self.register_parameter(
                    "parameter_a", torch.nn.Parameter(torch.randn(4))
                )
                # 注册缓冲区 "buffer"，使用随机初始化的缓冲区
                self.register_buffer("buffer", torch.randn(4))
                # 创建普通张量 "t"，而非缓冲区
                self.t = torch.rand(4)

                # 直接创建参数 "parameter_b"
                self.parameter_b = torch.nn.Parameter(torch.randn(4))
                # 添加子模块 "submodule_b"
                self.submodule_b = Submodule()

        m = TestModule()
        # 使用 Torch Script 脚本化方法获取导出和导入的副本
        m_loaded = self.getExportImportCopy(torch.jit.script(m))

        # 检查子模块
        self.assertEqual(
            len(list(m.named_modules())), len(list(m_loaded.named_modules()))
        )
        for m_s, loaded_s in zip(m.named_modules(), m_loaded.named_modules()):
            m_name, _ = m_s
            loaded_name, _ = loaded_s
            self.assertEqual(m_name, loaded_name)

        # 检查参数
        self.assertEqual(len(list(m.parameters())), len(list(m_loaded.parameters())))
        for m_p, loaded_p in zip(m.parameters(), m_loaded.parameters()):
            self.assertEqual(m_p, loaded_p)

        # 检查缓冲区
        self.assertEqual(
            len(list(m.named_buffers())), len(list(m_loaded.named_buffers()))
        )
        for m_b, loaded_b in zip(m.named_buffers(), m_loaded.named_buffers()):
            m_name, m_buffer = m_b
            loaded_name, loaded_buffer = loaded_b
            self.assertEqual(m_name, loaded_name)
            self.assertEqual(m_buffer, loaded_buffer)

    def test_save_load_with_extra_files(self):
        """
        Check that parameters, buffers, and submodules are the same after loading.
        """

        class Module(torch.nn.Module):
            def forward(self, x: Tensor):
                return x

        module = Module()
        # 使用 Torch Script 对模块进行脚本化
        script_module = torch.jit.script(module)

        # 定义额外文件，例如 "abc.json"
        extra_files = {"abc.json": b"[1,2,3]"}
        # 将脚本化的模块保存到缓冲区，同时传入额外文件和使用 FlatBuffer
        script_module_io = script_module._save_to_buffer_for_lite_interpreter(
            _extra_files=extra_files, _use_flatbuffer=True
        )

        re_extra_files = {}
        # 从缓冲区中获取模型的额外文件信息
        torch._C._get_model_extra_files_from_buffer(script_module_io, re_extra_files)

        # 检查额外文件是否一致
        self.assertEqual(extra_files, re_extra_files)
```