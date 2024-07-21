# `.\pytorch\test\jit\test_backends.py`

```
# Owner(s): ["oncall: jit"]

# 引入必要的库和模块
import io  # 导入 io 模块，提供对文件流和字节流的支持
import os  # 导入 os 模块，提供对操作系统功能的访问
import sys  # 导入 sys 模块，提供对 Python 解释器的访问和控制
import unittest  # 导入 unittest 模块，提供单元测试框架支持

import torch  # 导入 PyTorch 主模块
import torch._C  # 导入 PyTorch C++ 前端模块
from torch.jit.mobile import _load_for_lite_interpreter  # 从移动 JIT 模块中导入 _load_for_lite_interpreter 函数
from torch.testing import FileCheck  # 从测试模块中导入 FileCheck 类

from torch.testing._internal.common_utils import (  # 从内部测试共用工具模块中导入函数和变量
    find_library_location,  # 寻找动态库位置的函数
    IS_FBCODE,  # 判断是否在 Facebook 代码环境中的布尔值
    IS_MACOS,  # 判断是否在 macOS 系统中的布尔值
    IS_SANDCASTLE,  # 判断是否在 Sandcastle 环境中的布尔值
    IS_WINDOWS,  # 判断是否在 Windows 系统中的布尔值
    skipIfRocm,  # 如果是在 ROCm 环境下则跳过测试的装饰器
    TEST_WITH_ROCM,  # 判断是否在 ROCm 环境下的布尔值
)
from torch.testing._internal.jit_utils import JitTestCase  # 从 JIT 测试工具模块中导入 JitTestCase 类

# Make the helper files in test/ importable
# 将 test/ 目录下的辅助文件添加到模块搜索路径中，以便能够被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 如果该文件被直接运行，抛出运行时异常，提示不应直接运行该测试文件
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


# 将模块编译为指定后端运行
def to_test_backend(module, method_compile_spec):
    return torch._C._jit_to_backend(
        "test_backend", module, {"forward": method_compile_spec}
    )


# 将模块编译为支持多个方法的指定后端运行
def to_test_backend_multi(module, method_compile_spec):
    return torch._C._jit_to_backend("test_backend", module, method_compile_spec)


# 将模块编译为选择性后端运行，支持子模块
def to_test_backend_selective(module, method_compile_spec, submodules):
    def _to_test_backend(module):
        return to_test_backend(module, method_compile_spec)

    return torch._C._jit_to_backend_selective(module, _to_test_backend, submodules)


# 用于测试的基本模块，继承自 torch.nn.Module
class BasicModule(torch.nn.Module):
    """
    A simple Module used to test to_backend lowering machinery.
    """

    # 前向传播方法
    def forward(self, x, h):
        return self.accum(x, h), self.sub_accum(x, h)

    # 加法操作
    def accum(self, x, h):
        return x + h

    # 减法操作
    def sub_accum(self, x, h):
        return x - h


# 在 IS_WINDOWS 或 IS_MACOS 情况下忽略这个测试类，而是使用 TestBackends 中的一个
@unittest.skipIf(
    TEST_WITH_ROCM or IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE,
    "Non-portable load_library call used in test",
)
# JIT 后端测试用例的基类，包含输出比较和序列化/反序列化的常用实用函数
class JitBackendTestCase(JitTestCase):
    """
    A common base class for JIT backend tests that contains common utility
    functions for output comparison and serialization/deserialization.
    """

    # 测试初始化方法
    def setUp(self):
        super().setUp()
        # 查找并加载 libjitbackend_test.so 动态库文件
        lib_file_path = find_library_location("libjitbackend_test.so")
        torch.ops.load_library(str(lib_file_path))
        # 子类期望在其 setUp 方法中设置三个变量：
        # module - 正常的 Python 版本模块
        # scripted_module - 脚本化版本的模块
        # lowered_module - 编译到后端的版本模块
    def check_function(self, function_name, input):
        """
        Check that the function named 'function_name' produces the same output using
        Python, regular JIT and the backend for the given 'input'.
        """
        # 获取Python方法的引用
        python_method = self.module.__getattribute__(function_name)
        # 获取JIT编译后方法的引用
        jit_method = self.scripted_module.__getattr__(function_name)
        # 获取后端方法的引用
        backend_method = self.lowered_module.__getattr__(function_name)

        # 运行各个方法
        python_output = python_method(*input)  # 使用Python方法计算输出
        jit_output = jit_method(*input)        # 使用JIT编译后方法计算输出
        backend_output = backend_method(*input)  # 使用后端方法计算输出

        # 检查Python、JIT编译和后端方法的输出是否一致
        self.assertEqual(python_output, backend_output)
        self.assertEqual(jit_output, backend_output)

    def save_load(self):
        """
        Save and load the lowered module.
        """
        # 使用自定义方法保存和加载降级后的模块
        self.lowered_module = self.getExportImportCopy(self.lowered_module)

    def test_execution(self):
        """
        Stub for correctness tests.
        """
        # 测试执行正确性的存根函数，当前为空

    def test_save_load(self):
        """
        Stub for serialization tests.
        """
        # 测试序列化的存根函数，当前为空

    def test_errors(self):
        """
        Stub for testing error checking.
        """
        # 测试错误检查的存根函数，当前为空
class BasicModuleTest(JitBackendTestCase):
    """
    Tests for BasicModule.
    """

    def setUp(self):
        super().setUp()
        # Create Python, JIT and backend versions of BasicModule.
        self.module = BasicModule()  # 创建 BasicModule 的 Python 版本实例
        self.scripted_module = torch.jit.script(BasicModule())  # 创建 BasicModule 的 JIT 版本实例
        self.lowered_module = to_test_backend_multi(
            self.scripted_module,
            {"accum": {"": ""}, "sub_accum": {"": ""}, "forward": {"": ""}},
        )  # 创建 BasicModule 的后端版本实例

    def test_execution(self):
        # Test execution with backend against Python and JIT.
        input = torch.randn(5)

        # Test all three module methods.
        self.check_function("accum", (input, input))  # 测试 'accum' 方法
        self.check_function("sub_accum", (input, input))  # 测试 'sub_accum' 方法
        self.check_function("forward", (input, input))  # 测试 'forward' 方法

    @skipIfRocm
    def test_save_load(self):
        # Lowered module should produce the same outputs.
        self.test_execution()  # 执行测试用例

        # Save the compile spec to compare against the version retrieved after loading.
        pre_compile_spec = self.lowered_module.__getattr__(
            "__loweredModule__"
        ).__getattr__("__method_compile_spec")  # 获取编译规范前的版本

        # Save and load the lowered module.
        self.save_load()  # 保存并加载降低版本的模块

        # Get the compile spec after loading.
        post_compile_spec = self.lowered_module.__getattr__(
            "__loweredModule__"
        ).__getattr__("__method_compile_spec")  # 获取加载后的编译规范

        # Compile specs should match.
        self.assertEqual(pre_compile_spec, post_compile_spec)  # 检查编译规范是否匹配

        # Loaded module should produce the same outputs.
        self.test_execution()  # 再次执行测试用例


class BasicModuleUnavailableTest(JitBackendTestCase):
    """
    Tests for BasicModule with a backend that is not available.
    Fundamentally:
      * _jit_to_backend is successful.
      * Execution fails with an exception.
      * Saving is successful.
      * Loading fails with an exception.
    """

    def setUp(self):
        super().setUp()
        # Create Python, JIT and backend versions of BasicModule.
        self.module = BasicModule()  # 创建 BasicModule 的 Python 版本实例
        self.scripted_module = torch.jit.script(BasicModule())  # 创建 BasicModule 的 JIT 版本实例
        self.lowered_module = torch._C._jit_to_backend(
            "test_backend_unavailable",
            self.scripted_module,
            {"forward": {"": ""}},
        )  # 创建 BasicModule 的后端版本实例，此后端不可用

    def test_execution(self):
        # Test execution with backend fails because the backend that is not available.
        input = torch.randn(5)

        # Test exception is thrown.
        with self.assertRaisesRegexWithHighlight(
            Exception,
            r"Backend is not available.",
            'raise Exception("Backend is not available."',
        ):
            backend_method = self.lowered_module.__getattr__("forward")  # 获取后端方法
            backend_output = backend_method(*(input, input))  # 调用后端方法

    @skipIfRocm
    # 定义一个测试方法，用于测试保存和加载降低模块的行为
    def test_save_load(self):
        # 创建一个字节流对象，用于保存模块的序列化数据
        buffer = io.BytesIO()
        # 将降低后的模块保存到字节流中
        torch.jit.save(self.lowered_module, buffer)
        # 将字节流的读写位置移到开头，以便后续读取
        buffer.seek(0)
        # 使用断言检测在加载过程中是否会引发特定异常
        with self.assertRaisesRegexWithHighlight(
            Exception,  # 异常类型为 Exception
            r"Backend is not available.",  # 异常消息中应包含的字符串
            'raise Exception("Backend is not available."',  # 断言具体的异常引发语句
        ):
            # 从字节流中加载模块，预期加载会引发异常
            imported = torch.jit.load(buffer)
    """
    Tests for NestedModule that check that a module lowered to a backend can be used
    as a submodule.
    """
    
    class NestedModule(torch.nn.Module):
        """
        A Module with one submodule that is used to test that lowered Modules
        can be used as submodules.
        """

        def __init__(self, submodule):
            super().__init__()
            self.submodule = submodule
            # 初始化方法，接受一个子模块作为参数，并将其保存为实例变量

        def forward(self, x, h):
            # 前向传播方法，调用子模块的 forward 方法进行计算
            return self.submodule.forward(x, h)

    def setUp(self):
        super().setUp()
        # 设置测试环境，调用父类的 setUp 方法

        # Create Python, JIT and backend versions of NestedModule.
        # Both modules in self.module are regular Python modules.
        self.module = NestedModuleTest.NestedModule(BasicModule())
        # 创建一个 NestedModule 的实例 self.module，传入一个 BasicModule 的实例作为子模块

        # Both modules in self.scripted_module are ScriptModules.
        self.scripted_module = torch.jit.script(
            NestedModuleTest.NestedModule(BasicModule())
        )
        # 使用 torch.jit.script 方法将 NestedModule 包装成 ScriptModule，并保存在 self.scripted_module 中

        # First, script another instance of NestedModule with share_types=False so that it can be
        # selectively lowered without modifying the type of self.scripted_module.
        lowered_module = to_test_backend_multi(
            torch.jit.script(BasicModule()),
            {"accum": {"": ""}, "sub_accum": {"": ""}, "forward": {"": ""}},
        )
        # 使用 to_test_backend_multi 函数将 BasicModule 脚本化，并进行多个操作，得到 lowered_module

        # self.lowered_module is a ScriptModule, but its submodule is a lowered module.
        self.lowered_module = torch.jit.script(
            NestedModuleTest.NestedModule(lowered_module)
        )
        # 使用 torch.jit.script 方法将 NestedModule 包装成 ScriptModule，并保存在 self.lowered_module 中

    def test_execution(self):
        # Test execution with backend against Python and JIT.
        input = torch.randn(5)
        # 创建一个大小为 5 的随机输入张量 input

        # Test forward.
        self.check_function("forward", (input, input))
        # 调用 self.check_function 方法，测试 forward 方法的执行结果

    def test_save_load(self):
        # Lowered module should produce the same outputs.
        self.test_execution()
        # 调用 self.test_execution 方法，测试 lowered module 的执行结果

        # Save and load the lowered module.
        self.save_load()
        # 调用 self.save_load 方法，保存和加载 lowered module

        # Loaded module should produce the same outputs.
        self.test_execution()
        # 再次调用 self.test_execution 方法，测试加载后的模块的执行结果


class SelectiveLoweringTest(JitBackendTestCase):
    """
    Tests for the selective lowering API.
    """

    class OuterModule(torch.nn.Module):
        def __init__(self, sub1, sub2, other):
            super().__init__()
            self.sub1 = sub1
            self.sub2 = sub2
            self.other = other
            # 初始化方法，接受三个子模块作为参数，并将它们保存为实例变量

        def forward(self, x, y):
            # 前向传播方法，调用各个子模块的 forward 方法进行计算，并返回结果的累加
            a, b = self.sub1.submodule.forward(x, y)
            c, d = self.sub2.forward(x, y)
            e, f = self.other.forward(x, y)
            return a + c + e, b + d + f

    class MiddleModule(torch.nn.Module):
        def __init__(self, submodule):
            super().__init__()
            self.submodule = submodule
            # 初始化方法，接受一个子模块作为参数，并将其保存为实例变量

        def forward(self, x, y):
            # 前向传播方法，调用子模块的 forward 方法进行计算
            return self.submodule.forward(x, y)
    # 设置测试环境，调用父类的 setUp 方法
    def setUp(self):
        super().setUp()
        # 导入测试类中的 OuterModule 和 MiddleModule
        OuterModule = SelectiveLoweringTest.OuterModule
        MiddleModule = SelectiveLoweringTest.MiddleModule

        # 定义一个函数，用于创建没有类型共享的脚本模块
        def script_without_type_sharing(mod):
            return torch.jit._recursive.create_script_module(
                mod, torch.jit._recursive.infer_methods_to_compile, share_types=False
            )

        # 创建一个层次结构的 Python 版本、JIT 版本和后端版本：
        #                 --------- OuterModule --------
        #                 |              |              |
        #           MiddleModule    MiddleModule   MiddleModule
        #                |               |              |
        #           BasicModule     BasicModule    BasicModule
        #
        # 两个 BasicModule 将被降低，第三个不会被降低。
        # 初始化 self.module 为 OuterModule 的实例，包含三个 MiddleModule 实例，每个包含一个 BasicModule 实例。
        self.module = OuterModule(
            MiddleModule(BasicModule()),
            MiddleModule(BasicModule()),
            MiddleModule(BasicModule()),
        )
        # 使用 script_without_type_sharing 函数创建没有类型共享的脚本化模块
        self.scripted_module = script_without_type_sharing(
            OuterModule(
                MiddleModule(BasicModule()),
                MiddleModule(BasicModule()),
                MiddleModule(BasicModule()),
            )
        )
        # 再次使用 script_without_type_sharing 函数创建没有类型共享的脚本化模块，赋值给 lowered_module
        self.lowered_module = script_without_type_sharing(
            OuterModule(
                MiddleModule(BasicModule()),
                MiddleModule(BasicModule()),
                MiddleModule(BasicModule()),
            )
        )
        # 将 lowered_module 应用于测试后端选择性降级，传递的参数为要降级的方法名和模块路径列表
        self.lowered_module = to_test_backend_selective(
            self.lowered_module, {"forward": ""}, ["sub1.submodule", "sub2.submodule"]
        )

    # 执行测试函数，测试前向传播
    def test_execution(self):
        # 生成一个随机张量作为输入
        input = torch.randn(5)
        # 调用 check_function 方法，测试 forward 方法的执行结果
        self.check_function("forward", (input, input))

        # 调用 test_selective_lowering_type_remap 方法进行类型重映射测试

    # 测试保存和加载模型
    def test_save_load(self):
        # 执行 test_execution 方法，测试前向传播
        self.test_execution()
        # 调用 save_load 方法保存和加载模型
        self.save_load()
        # 再次执行 test_execution 方法，测试保存和加载后的前向传播结果

        # 调用 test_selective_lowering_type_remap 方法进行类型重映射测试
        self.test_selective_lowering_type_remap()
    # 定义一个测试方法，用于验证选择性降级过程中的类型重映射和替换
    def test_selective_lowering_type_remap(self):
        """
        Check that type remapping and replacement occurred during selective lowering.
        """
        # 检查 self.lowered_module 是否被降级，但由于它直接调用了 lowered module，所以会包含 test_backendLoweredModule。
        FileCheck().check("OuterModule").check("BasicModule").run(
            self.scripted_module.graph
        )
        FileCheck().check("OuterModule").check_not(
            "__torch__.torch.classes.__backends__.test_backend"
        ).check("LoweredWrapper.test_backend").run(self.lowered_module.graph)

        # 检查 self.lowered_module.sub1/sub2 是否被降级，但 BasicModule 已在它们的图中被替换。
        FileCheck().check("MiddleModule").check("BasicModule").check_not(
            "LoweredWrapper.test_backend"
        ).run(self.scripted_module.sub1.graph)
        FileCheck().check("MiddleModule").check_not(
            "__torch__.torch.classes.__backends__.test_backend"
        ).check("LoweredWrapper.test_backend").run(self.lowered_module.sub1.graph)

        FileCheck().check("MiddleModule").check("BasicModule").check_not(
            "LoweredWrapper.test_backend"
        ).run(self.scripted_module.sub2.graph)
        FileCheck().check("MiddleModule").check_not(
            "__torch__.torch.classes.__backends__.test_backend"
        ).check("LoweredWrapper.test_backend").run(self.lowered_module.sub2.graph)

        # 检查 self.lowered_module.sub1/sub2.submodule 是否被降级。它们应该有一个新属性 __loweredModule__，
        # 其图中应提到 __torch__.torch.classes.__backends__.test_backend，这是用于在测试 JIT 后端上执行函数的 TorchBind 类。
        FileCheck().check("LoweredModule.test_backend").check(
            "__torch__.torch.classes.__backends__.test_backend"
        ).run(self.lowered_module.sub1.submodule.__loweredModule__.graph)

        FileCheck().check("LoweredModule.test_backend").check(
            "__torch__.torch.classes.__backends__.test_backend"
        ).run(self.lowered_module.sub2.submodule.__loweredModule__.graph)

        # 检查 self.other 和 self.other.submodule 是否未被选择性降级过程所影响。
        FileCheck().check("MiddleModule").check("BasicModule").check_not(
            "__torch__.torch.classes.__backends__.test_backend"
        ).check_not("LoweredWrapper.test_backend").run(self.scripted_module.other.graph)
        FileCheck().check("BasicModule").check_not(
            "__torch__.torch.classes.__backends__.test_backend"
        ).check_not("LoweredModule.test_backend").run(
            self.scripted_module.other.submodule.graph
        )
    def test_errors(self):
        """
        Check errors associated with selective lowering.
        """
        # 检查尝试降低非 ScriptModule 对象时抛出的错误消息。
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Object .* is not a ScriptModule", ""
        ):
            to_test_backend_selective(torch.nn.ReLU(), {"forward": ""}, ["submodule"])

        # 将 MiddleModule 类赋值给变量 MiddleModule，用于后续测试。
        MiddleModule = SelectiveLoweringTest.MiddleModule
        # 创建一个 MiddleModule 实例 mod，其内部包含 BasicModule 实例。
        mod = MiddleModule(BasicModule())
        # 给 mod 实例添加一个名为 new_attr 的属性，并赋值为 3。
        mod.new_attr = 3

        # 检查尝试对非 Module 类型的属性 new_attr 进行降低时抛出的错误消息。
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Attribute named new_attr is not a Module", ""
        ):
            to_test_backend_selective(
                torch.jit.script(mod), {"forward": ""}, ["new_attr"]
            )

        # 创建一个 OuterModule 实例 mod，其内部包含三个相同的 MiddleModule 实例。
        OuterModule = SelectiveLoweringTest.OuterModule
        mod = OuterModule(
            MiddleModule(BasicModule()),
            MiddleModule(BasicModule()),
            MiddleModule(BasicModule()),
        )

        # 检查当模块层次结构中存在相同类型的模块时，尝试进行选择性降低时抛出的错误消息。
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"Selective lowering is only supported for module hierarchies with unique types",
            "",
        ):
            to_test_backend_selective(
                torch.jit.script(mod), {"forward": ""}, ["sub1.submodule"]
            )
# 根据条件跳过测试，条件包括 TEST_WITH_ROCM 或 IS_SANDCASTLE 或 IS_WINDOWS 或 IS_MACOS 或 IS_FBCODE
@unittest.skipIf(
    TEST_WITH_ROCM or IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE,
    "Non-portable load_library call used in test",
)
class TestBackends(JitTestCase):
    """
    这个类用于包装并调用 JitBackendTestCase 的所有子类，以便在 test_jit.py 中不需要单独导入每一个类。
    """

    def __init__(self, name):
        super().__init__(name)
        # 初始化各个子测试类实例
        self.basic_module_test = BasicModuleTest(name)
        self.basic_module_unavailable_test = BasicModuleUnavailableTest(name)
        self.nested_module_test = NestedModuleTest(name)
        self.selective_lowering_test = SelectiveLoweringTest(name)

    def setUp(self):
        super().setUp()
        # 如果不是在 ROCm 环境下，执行初始化设置
        if not TEST_WITH_ROCM:
            self.basic_module_test.setUp()
            self.basic_module_unavailable_test.setUp()
            self.nested_module_test.setUp()
            self.selective_lowering_test.setUp()

    @skipIfRocm
    def test_execution(self):
        # 执行各个子测试类的执行测试
        self.basic_module_test.test_execution()
        self.basic_module_unavailable_test.test_execution()
        self.nested_module_test.test_execution()
        self.selective_lowering_test.test_execution()

    @skipIfRocm
    def test_save_load(self):
        # 执行各个子测试类的保存加载测试
        self.basic_module_test.test_save_load()
        self.basic_module_unavailable_test.test_save_load()
        self.nested_module_test.test_save_load()
        self.selective_lowering_test.test_save_load()

    @skipIfRocm
    def test_errors(self):
        # 执行选择性降低错误测试
        self.selective_lowering_test.test_errors()


"""
Backend 带编译器的单元测试
这个测试用例与现有的 TestBackends 是分开的，因为它们涵盖不同的方面。
这个测试中的实际后端实现是不同的。
它有一个简单的演示编译器来测试移动端的端到端流程。
然而，这个测试目前不能覆盖选择性降低，这在 TestBackends 中有覆盖。
"""


class BasicModuleAdd(torch.nn.Module):
    """
    一个简单的加法模块，用于测试后端降低机制。
    """

    def forward(self, x, h):
        return x + h


# 在 IS_WINDOWS 或 IS_MACOS 情况下忽略此部分代码。因此我们需要在 TestBackends 中的那一个。
@unittest.skipIf(
    TEST_WITH_ROCM or IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE,
    "Non-portable load_library call used in test",
)
class JitBackendTestCaseWithCompiler(JitTestCase):
    """
    JIT 后端测试的通用基类，包含用于输出比较的通用实用函数。
    """
    # 设置测试环境，在父类的 setUp 方法基础上进行设置
    def setUp(self):
        super().setUp()
        # 查找并获取名为 "libbackend_with_compiler.so" 的库文件路径
        lib_file_path = find_library_location("libbackend_with_compiler.so")
        # 使用 torch.ops.load_library 加载库文件
        torch.ops.load_library(str(lib_file_path))
        # 子类需要在其 setUp 方法中设置四个变量：
        # module - 普通的 Python 版本模块
        # scripted_module - 脚本化版本的模块
        # lowered_module - 被转换到后端的模块
        # mobile_module - 可以由 Pytorch Mobile 执行的模块

    def check_forward(self, input):
        """
        检查前向函数对于给定的输入 'input' 在 Python、普通 JIT、后端和移动端是否产生相同的输出。
        """

        # 调用各个模块的 forward 方法获取输出
        python_output = self.module.forward(*input)
        jit_output = self.scripted_module.forward(*input)
        backend_output = self.lowered_module(*input)
        mobile_output = self.mobile_module(*input)

        # Python、JIT、后端和移动端返回的结果应该全部匹配。
        self.assertEqual(python_output, backend_output)
        self.assertEqual(jit_output, backend_output)
        self.assertEqual(mobile_output, backend_output)

    def test_execution(self):
        """
        用于正确性测试的桩函数。
        """
        pass

    def test_errors(self):
        """
        用于测试错误检查的桩函数。
        """
        pass
class BasicModuleTestWithCompiler(JitBackendTestCaseWithCompiler):
    """
    Tests for BasicModuleAdd.
    """

    def setUp(self):
        super().setUp()
        # 创建 BasicModuleAdd 的 Python、JIT 和后端版本
        self.module = BasicModuleAdd()
        self.scripted_module = torch.jit.script(BasicModuleAdd())
        compile_spec = {
            "forward": {
                "input_shapes": "((1, 1, 320, 240), (1, 3))",
                "some_other_option": "True",
            },
        }
        # 将 JIT 模块编译为指定后端的 lowered_module
        self.lowered_module = torch._C._jit_to_backend(
            "backend_with_compiler_demo", self.scripted_module, compile_spec
        )
        # 创建 BasicModuleAdd 的移动版本
        buffer = io.BytesIO(self.lowered_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        self.mobile_module = _load_for_lite_interpreter(buffer)

    def test_execution(self):
        # 使用后端测试 Python 和 JIT 的执行
        input = torch.ones(1, dtype=torch.float)
        self.check_forward((input, input))


class ErrorMessagesWithCompiler(JitBackendTestCase):
    """
    Tests for errors that occur with compiler, specifically:
        * an operator is not supported by the backend
    """

    class ModuleNotSupported(torch.nn.Module):
        """
        A module with an operator that is not supported.
        """

        def forward(self, x, h):
            # 返回 x 与 h 的乘积，此处后续代码无法执行
            return x * h
            self._loweredmodule.forward()

    def test_errors(self):
        scripted_module_n = torch.jit.script(
            ErrorMessagesWithCompiler.ModuleNotSupported()
        )
        # 测试在将不支持的操作符模块降低时是否会抛出异常
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            # 特殊转义字符被替换成 '.'
            r"""The node of aten::mul is not supported in this compiler. .*
        def forward.self, x, h.:
            return x . h
                   ~~~~~ <--- HERE
            self._loweredmodule.forward..
""",
            "",
        ):
            lowered_module_n = torch._C._jit_to_backend(
                "backend_with_compiler_demo", scripted_module_n, {"forward": {"": ""}}
            )


class CompModuleTestWithCompiler(JitBackendTestCase):
    """
    Tests for CompModule, which is a module with two lowered submodules
    """

    class BasicModuleSub(torch.nn.Module):
        """
        A simple subtraction Module to be used in CompModule.
        """

        def forward(self, x, h):
            # 返回 x 与 h 的差值
            return x - h
    class CompModule(torch.nn.Module):
        """
        A module with two lowered submodules.
        """

        def __init__(self, addmodule, submodule):
            super().__init__()
            self.lowered_add = addmodule  # 将加法子模块保存到实例变量 lowered_add 中
            self.lowered_sub = submodule  # 将减法子模块保存到实例变量 lowered_sub 中

        def forward(self, a, b, s):
            c = self.lowered_add.forward(a, b)  # 调用加法子模块的前向传播方法
            d = self.lowered_sub.forward(a, b)  # 调用减法子模块的前向传播方法
            y = s * (c * d)  # 计算最终的输出值 y
            return y

    def setUp(self):
        super().setUp()
        # Create Python and JIT versions of CompModule with lowered submodules.
        compile_spec = {
            "forward": {
                "input_shapes": "((1, 1, 320, 240), (1, 3))",
                "some_other_option": "True",
            },
        }
        lowered_add = torch._C._jit_to_backend(
            "backend_with_compiler_demo",
            torch.jit.script(BasicModuleAdd()),  # 通过 Torch 脚本创建加法子模块的 JIT 版本
            compile_spec,
        )
        lowered_sub = torch._C._jit_to_backend(
            "backend_with_compiler_demo",
            torch.jit.script(CompModuleTestWithCompiler.BasicModuleSub()),  # 通过 Torch 脚本创建减法子模块的 JIT 版本
            {"forward": {"": ""}},
        )
        self.module = CompModuleTestWithCompiler.CompModule(lowered_add, lowered_sub)  # 创建 CompModule 实例，使用 JIT 版本的子模块
        self.scripted_module = torch.jit.script(
            CompModuleTestWithCompiler.CompModule(lowered_add, lowered_sub)  # 创建 CompModule 的脚本化版本，使用 JIT 版本的子模块
        )
        # No backend version of CompModule currently, so this is filler.
        self.lowered_module = self.scripted_module  # 将脚本化版本的 CompModule 保存为 lowered_module

        # Create a mobile version of CompModule from JIT version
        buffer = io.BytesIO(self.scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        self.mobile_module = _load_for_lite_interpreter(buffer)  # 从 JIT 版本的 CompModule 创建移动版本

    def test_execution(self):
        # Test execution with backend against Python and JIT.
        input1 = torch.ones(1, dtype=torch.float)
        input2 = torch.ones(1, dtype=torch.float)

        # Test forward.
        self.check_function("forward", (input1, input2, input2))  # 测试 CompModule 的 forward 方法的执行
# IS_SANDCASTLE、IS_WINDOWS 或 IS_MACOS 或 IS_FBCODE 为真时跳过测试，用于平台特定逻辑的测试跳过。
@unittest.skipIf(
    IS_SANDCASTLE or IS_WINDOWS or IS_MACOS or IS_FBCODE,
    "Non-portable load_library call used in test",
)
# 测试类，继承自 JitTestCase，用于测试带编译器的后端逻辑
class TestBackendsWithCompiler(JitTestCase):
    """
    这个类包装并调用所有 JitBackendTestCaseWithCompiler 的子类，
    使得在 test_jit.py 中不需要单独导入每个子类。
    """

    def __init__(self, name):
        super().__init__(name)
        # 初始化基本模块编译器测试、错误消息编译器测试和组合模块编译器测试
        self.basic_module_compiler_test = BasicModuleTestWithCompiler(name)
        self.error_module_compiler_test = ErrorMessagesWithCompiler(name)
        self.comp_module_compiler_test = CompModuleTestWithCompiler(name)

    def setUp(self):
        super().setUp()
        # 设置每个子测试的前置条件
        self.basic_module_compiler_test.setUp()
        self.error_module_compiler_test.setUp()
        self.comp_module_compiler_test.setUp()

    def test_execution(self):
        # 执行基本模块编译器测试和组合模块编译器测试的执行测试
        self.basic_module_compiler_test.test_execution()
        self.comp_module_compiler_test.test_execution()

    def test_errors(self):
        # 执行错误消息编译器测试的错误测试
        self.error_module_compiler_test.test_errors()


class CompModuleTestSameNameWithCompiler(JitBackendTestCase):
    """
    CompModule 的测试，这是一个包含两个同名降级子模块的模块。
    """

    class ModuleAdd(torch.nn.Module):
        """
        用于测试 to_backend 降级机制的简单模块。
        """

        def forward(self, x, h):
            return x + h

    class CompModule(torch.nn.Module):
        """
        包含两个降级子模块的模块。
        """

        def __init__(self):
            super().__init__()
            # 编译规范
            compile_spec = {
                "forward": {
                    "some_other_option": "True",
                },
            }
            # 使用 torch._C._jit_to_backend 将 ModuleAdd 降级并加入到后端
            self.add = torch._C._jit_to_backend(
                "backend_with_compiler_demo",
                torch.jit.script(ModuleAdd()),  # noqa: F821
                compile_spec,
            )
            # 使用 torch._C._jit_to_backend 将 ModuleAdd 降级并加入到后端
            self.sub = torch._C._jit_to_backend(
                "backend_with_compiler_demo",
                torch.jit.script(ModuleAdd()),  # noqa: F821
                compile_spec,
            )

        def forward(self, a, b, s: int):
            # 调用降级后的子模块的 forward 方法
            c = self.add.forward(a, b)
            d = self.sub.forward(a, b)
            y = s * (c * d)
            return y

    def setUp(self):
        super().setUp()
        # 初始化 CompModule 实例
        self.module = CompModule()  # noqa: F821
        # 对 CompModule 进行脚本化
        self.scripted_module = torch.jit.script(self.module)
        # 将脚本化后的模块保存为字节流并加载为轻量级解释器
        buffer = io.BytesIO(self.scripted_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        self.mobile_module = _load_for_lite_interpreter(buffer)

    def test_execution(self):
        # 创建测试数据
        a = torch.ones(1)
        b = 3 * torch.ones(1)
        s = 3
        # 测试 forward 方法的执行
        self.check_function("forward", (a, b, s))


class AddedAttributesTest(JitBackendTestCase):
    """
    测试在降级后向模型添加属性的情况。
    """
    def setUp(self):
        super().setUp()
        # 创建 BasicModule 的 Python、JIT 和后端版本。
        self.module = BasicModule()
        # 对 BasicModule 进行 JIT 编译
        self.scripted_module = torch.jit.script(BasicModule())
        # 使用多个测试后端对 JIT 编译后的模型进行转换
        self.lowered_module = to_test_backend_multi(
            self.scripted_module,
            {"accum": {"": ""}, "sub_accum": {"": ""}, "forward": {"": ""}},
        )

    def test_attribute(self):
        input = [(torch.ones(5),)]
        # 使用下降后的模型预测输入
        pre_bundled = self.lowered_module(*input[0])
        # 将捆绑输入附加到模型，添加多个属性和函数
        self.lowered_module = (
            torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
                lowered_module, input  # noqa: F821
            )
        )
        # 使用捆绑后的输入进行预测
        post_bundled = self.lowered_module(
            *self.lowered_module.get_all_bundled_inputs()[0]
        )
        # 保存和加载下降后的模型
        self.save_load()
        # 在保存和加载后使用捆绑输入以证明其保留
        post_load = self.lowered_module(
            *self.lowered_module.get_all_bundled_inputs()[0]
        )
        # 断言捆绑前后的输出相等
        self.assertEqual(pre_bundled, post_bundled)
        # 再次断言捆绑后加载后的输出与之前的相等
        self.assertEqual(post_bundled, post_load)
```