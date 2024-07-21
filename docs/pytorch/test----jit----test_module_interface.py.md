# `.\pytorch\test\jit\test_module_interface.py`

```py
# Owner(s): ["oncall: jit"]

# 导入必要的库和模块
import os
import sys
from typing import Any, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.testing._internal.jit_utils import JitTestCase, make_global

# 将测试文件夹中的 helper 文件变为可导入状态
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 如果直接运行该脚本，则抛出运行时错误，建议通过测试框架运行
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义一个原始模块 OrigModule，继承自 nn.Module
class OrigModule(nn.Module):
    # 定义方法 one，接受两个 Tensor 类型的输入，返回它们的和加 1
    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        return inp1 + inp2 + 1

    # 定义方法 two，接受一个 Tensor 输入，返回输入加 2 的结果
    def two(self, input: Tensor) -> Tensor:
        return input + 2

    # 定义 forward 方法，接受一个 Tensor 输入，返回输入加上 one 方法对输入两次加 1 的结果
    def forward(self, input: Tensor) -> Tensor:
        return input + self.one(input, input) + 1

# 定义一个新模块 NewModule，继承自 nn.Module
class NewModule(nn.Module):
    # 定义方法 one，接受两个 Tensor 类型的输入，返回它们的乘积加 1
    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        return inp1 * inp2 + 1

    # 定义 forward 方法，接受一个 Tensor 输入，返回调用 one 方法的结果，其中第二个输入为 input + 1
    def forward(self, input: Tensor) -> Tensor:
        return self.one(input, input + 1)

# 定义一个测试模块接口 TestModuleInterface，继承自 JitTestCase
class TestModuleInterface(JitTestCase):
    # 定义测试方法 test_not_submodule_interface_call
    def test_not_submodule_interface_call(self):
        # 定义一个 ModuleInterface 接口，标注有一个方法 one，接受两个 Tensor 类型的输入，返回一个 Tensor
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                pass

        # 定义一个测试类 TestNotModuleInterfaceCall，继承自 nn.Module
        class TestNotModuleInterfaceCall(nn.Module):
            proxy_mod: ModuleInterface

            # 定义初始化方法
            def __init__(self):
                super().__init__()
                # 使用 OrigModule 作为 proxy_mod 的实现
                self.proxy_mod = OrigModule()

            # 定义 forward 方法，接受一个 Tensor 输入，返回 proxy_mod 的 two 方法对输入的结果
            def forward(self, input: Tensor) -> Tensor:
                return self.proxy_mod.two(input)

        # 断言在编译 Torch 脚本时会抛出 RuntimeError，且错误信息包含指定内容和高亮部分
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "object has no attribute or method", "self.proxy_mod.two"
        ):
            torch.jit.script(TestNotModuleInterfaceCall())
    # 定义一个测试方法，用于测试模块的接口
    def test_module_interface(self):
        # 定义一个接口 OneTwoModule，继承自 nn.Module
        @torch.jit.interface
        class OneTwoModule(nn.Module):
            # 定义接口方法 one，接受两个张量参数并返回一个张量
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                pass

            # 定义接口方法 two，接受一个张量参数并返回一个张量
            def two(self, x: Tensor) -> Tensor:
                pass

            # 定义接口方法 forward，接受一个张量参数并返回一个张量
            def forward(self, x: Tensor) -> Tensor:
                pass

        # 定义另一个接口 OneTwoClass
        @torch.jit.interface
        class OneTwoClass:
            # 定义接口方法 one，接受两个张量参数并返回一个张量
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                pass

            # 定义接口方法 two，接受一个张量参数并返回一个张量
            def two(self, x: Tensor) -> Tensor:
                pass

        # 定义类 FooMod，实现了接口 OneTwoModule 的方法
        class FooMod(nn.Module):
            # 实现接口方法 one，返回两个张量的和
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                return x + y

            # 实现接口方法 two，返回输入张量的两倍
            def two(self, x: Tensor) -> Tensor:
                return 2 * x

            # 实现接口方法 forward，先调用 two 方法再调用 one 方法
            def forward(self, x: Tensor) -> Tensor:
                return self.one(self.two(x), x)

        # 定义类 BarMod，同样实现了接口 OneTwoModule 的方法
        class BarMod(nn.Module):
            # 实现接口方法 one，返回两个张量的乘积
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                return x * y

            # 实现接口方法 two，返回 2 除以输入张量
            def two(self, x: Tensor) -> Tensor:
                return 2 / x

            # 实现接口方法 forward，先调用 one 方法再调用 two 方法
            def forward(self, x: Tensor) -> Tensor:
                return self.two(self.one(x, x))

            # 定义一个导出方法 forward2，返回 one 和 two 方法组合后的结果加 1
            @torch.jit.export
            def forward2(self, x: Tensor) -> Tensor:
                return self.two(self.one(x, x)) + 1

        # 将接口 OneTwoModule 和 OneTwoClass 注册为全局可用
        make_global(OneTwoModule, OneTwoClass)

        # 定义一个函数 use_module_interface，接受一个 OneTwoModule 类型的列表和一个张量 x，返回两个模块的 forward 方法结果的和
        def use_module_interface(mod_list: List[OneTwoModule], x: torch.Tensor):
            return mod_list[0].forward(x) + mod_list[1].forward(x)

        # 定义一个函数 use_class_interface，接受一个 OneTwoClass 类型的列表和一个张量 x，返回两个类的方法结果的和
        def use_class_interface(mod_list: List[OneTwoClass], x: Tensor) -> Tensor:
            return mod_list[0].two(x) + mod_list[1].one(x, x)

        # 对 FooMod 和 BarMod 分别进行 Torch 脚本编译
        scripted_foo_mod = torch.jit.script(FooMod())
        scripted_bar_mod = torch.jit.script(BarMod())

        # 调用 self.checkScript，测试 use_module_interface 方法
        self.checkScript(
            use_module_interface,
            (
                [scripted_foo_mod, scripted_bar_mod],  # 使用两个 Torch 脚本化的模块
                torch.rand(3, 4),  # 随机生成一个形状为 (3, 4) 的张量
            ),
        )

        # 调用 self.checkScript，测试 use_class_interface 方法
        self.checkScript(
            use_class_interface,
            (
                [scripted_foo_mod, scripted_bar_mod],  # 使用两个 Torch 脚本化的模块
                torch.rand(3, 4),  # 随机生成一个形状为 (3, 4) 的张量
            ),
        )

        # 定义函数 call_module_interface_on_other_method，接受一个 OneTwoModule 类型的接口和一个张量 x，调用其 forward2 方法
        def call_module_interface_on_other_method(
            mod_interface: OneTwoModule, x: Tensor
        ) -> Tensor:
            return mod_interface.forward2(x)

        # 确保在调用非接口指定的方法时出错，预期抛出 RuntimeError，错误信息包含 "mod_interface.forward2"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "object has no attribute or method", "mod_interface.forward2"
        ):
            # 调用 self.checkScript，测试 call_module_interface_on_other_method 方法
            self.checkScript(
                call_module_interface_on_other_method,
                (
                    scripted_bar_mod,  # 使用 Torch 脚本化的 BarMod 模块
                    torch.rand(3, 4),  # 随机生成一个形状为 (3, 4) 的张量
                ),
            )
    def test_module_doc_string(self):
        @torch.jit.interface
        class TestInterface(nn.Module):
            def one(self, inp1, inp2):
                # type: (Tensor, Tensor) -> Tensor
                # 方法声明：接受两个张量作为输入，返回一个张量
                pass

            def forward(self, input):
                # type: (Tensor) -> Tensor
                # 方法声明：接受一个张量作为输入，返回一个张量
                r"""stuff 1"""  # 文档字符串：描述函数功能或重要信息，这里描述“stuff 1”
                r"""stuff 2"""  # 这行文档字符串并不会被正式记录，因为它在函数定义之外
                pass  # 占位符，表示该函数未实现具体功能
                r"""stuff 3"""  # 这行文档字符串同样不会被正式记录

        class TestModule(nn.Module):
            proxy_mod: TestInterface

            def __init__(self):
                super().__init__()
                self.proxy_mod = OrigModule()  # 创建一个OrigModule实例，赋给proxy_mod属性

            def forward(self, input):
                # type: (Tensor) -> Tensor
                # 方法声明：接受一个张量作为输入，返回一个张量
                return self.proxy_mod.forward(input)  # 调用proxy_mod的forward方法并返回结果

        input = torch.randn(3, 4)  # 创建一个形状为(3, 4)的随机张量
        self.checkModule(TestModule(), (input,))  # 调用self.checkModule检查TestModule的行为
    def test_module_interface_subtype(self):
        # 定义一个 TorchScript 接口 OneTwoModule，要求包含方法 one、two 和 forward
        @torch.jit.interface
        class OneTwoModule(nn.Module):
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                pass

            def two(self, x: Tensor) -> Tensor:
                pass

            def forward(self, x: Tensor) -> Tensor:
                pass

        # 将 OneTwoModule 接口注册为全局接口
        make_global(OneTwoModule)

        # 将 Python 类型 Foo 编译为 TorchScript 类型，并确保其不是 OneTwoModule 接口的子类型
        @torch.jit.script
        def as_module_interface(x: OneTwoModule) -> OneTwoModule:
            return x

        @torch.jit.script
        class Foo:
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                return x + y

            def two(self, x: Tensor) -> Tensor:
                return 2 * x

            def forward(self, x: Tensor) -> Tensor:
                return self.one(self.two(x), x)

        # 检查 Foo 类型不是 OneTwoModule 接口的子类型，期望引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError, "ScriptModule class can be subtype of module interface"
        ):
            as_module_interface(Foo())

        # 定义一个错误的 nn.Module 类型 WrongMod，其中方法 two 的参数类型不匹配接口定义
        class WrongMod(nn.Module):
            def two(self, x: int) -> int:
                return 2 * x

            def forward(self, x: Tensor) -> Tensor:
                return x + torch.randn(3, self.two(3))

        # 将 WrongMod 类型编译为 TorchScript 类型
        scripted_wrong_mod = torch.jit.script(WrongMod())

        # 检查 WrongMod 类型不兼容 OneTwoModule 接口，期望引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "is not compatible with interface"):
            as_module_interface(scripted_wrong_mod)

        # 定义一个 TorchScript 接口 TensorToAny，要求包含一个 forward 方法，参数类型为 torch.Tensor，返回类型为 Any
        @torch.jit.interface
        class TensorToAny(nn.Module):
            def forward(self, input: torch.Tensor) -> Any:
                pass

        # 将 TensorToAny 接口注册为全局接口
        make_global(TensorToAny)

        # 将 Python 类型实现 as_tensor_to_any 编译为 TorchScript 类型
        @torch.jit.script
        def as_tensor_to_any(x: TensorToAny) -> TensorToAny:
            return x

        # 定义一个 TorchScript 接口 AnyToAny，要求包含一个 forward 方法，参数类型和返回类型均为 Any
        @torch.jit.interface
        class AnyToAny(nn.Module):
            def forward(self, input: Any) -> Any:
                pass

        # 将 AnyToAny 接口注册为全局接口
        make_global(AnyToAny)

        # 将 Python 类型实现 as_any_to_any 编译为 TorchScript 类型
        @torch.jit.script
        def as_any_to_any(x: AnyToAny) -> AnyToAny:
            return x

        # 定义一个实现了 TensorToAny 接口的 Python 类型 TensorToAnyImplA
        class TensorToAnyImplA(nn.Module):
            def forward(self, input: Any) -> Any:
                return input

        # 定义一个实现了 TensorToAny 接口的 Python 类型 TensorToAnyImplB，其 forward 方法返回类型为 torch.Tensor
        class TensorToAnyImplB(nn.Module):
            def forward(self, input: Any) -> torch.Tensor:
                return torch.tensor([1])

        # 定义一个实现了 AnyToAny 接口的 Python 类型 AnyToAnyImpl，其 forward 方法返回类型为 torch.Tensor
        class AnyToAnyImpl(nn.Module):
            def forward(self, input: Any) -> torch.Tensor:
                return torch.tensor([1])

        # 将 TensorToAnyImplA 类型编译为 TorchScript 类型，并调用 as_tensor_to_any 进行检查
        as_tensor_to_any(torch.jit.script(TensorToAnyImplA()))
        # 将 TensorToAnyImplB 类型编译为 TorchScript 类型，并调用 as_tensor_to_any 进行检查
        as_tensor_to_any(torch.jit.script(TensorToAnyImplB()))
        # 将 AnyToAnyImpl 类型编译为 TorchScript 类型，并调用 as_any_to_any 进行检查
        as_any_to_any(torch.jit.script(AnyToAnyImpl()))
    # 测试模块接口的继承是否会引发异常
    def test_module_interface_inheritance(self):
        # 使用断言检查是否抛出了预期的运行时错误信息
        with self.assertRaisesRegex(
            RuntimeError, "does not support inheritance yet. Please directly"
        ):
            # 定义一个继承了 nn.ReLU 的 Torch JIT 接口，预期会引发错误
            @torch.jit.interface
            class InheritMod(nn.ReLU):
                # 定义一个方法，但这种继承方式应该不被支持
                def three(self, x: Tensor) -> Tensor:
                    return 3 * x

    # 测试模块替换功能
    def test_module_swap(self):
        # 定义一个 Torch JIT 接口 ModuleInterface，包含必要的方法签名
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                pass

            def forward(self, input: Tensor) -> Tensor:
                pass

        # 定义一个测试模块 TestModule，其包含一个 ModuleInterface 类型的成员变量
        class TestModule(nn.Module):
            proxy_mod: ModuleInterface

            def __init__(self):
                super().__init__()
                # 初始化时将 proxy_mod 设置为 OrigModule 的实例
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                # 调用 proxy_mod 的 forward 方法
                return self.proxy_mod.forward(input)

        # 将 TestModule 脚本化
        scripted_mod = torch.jit.script(TestModule())
        input = torch.randn(3, 4)
        # 断言脚本化模块的输出是否符合预期（3 * input + 2）
        self.assertEqual(scripted_mod(input), 3 * input + 2)

        # 替换模块为具有相同接口的新模块
        scripted_mod.proxy_mod = torch.jit.script(NewModule())
        # 断言脚本化模块的输出是否符合预期（input * (input + 1) + 1）
        self.assertEqual(scripted_mod(input), input * (input + 1) + 1)

        # 替换为非脚本化模块应该引发错误
        with self.assertRaisesRegex(
            RuntimeError, "a ScriptModule with non-scripted module"
        ):
            scripted_mod.proxy_mod = NewModule()

    # 测试模块替换时使用不兼容接口的错误情况
    def test_module_swap_wrong_module(self):
        # 定义一个 Torch JIT 接口 ModuleInterface，包含必要的方法签名
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                pass

            def forward(self, input: Tensor) -> Tensor:
                pass

        # 定义一个不兼容接口的新模块 NewModuleWrong
        class NewModuleWrong(nn.Module):
            def forward(self, input: int) -> int:
                return input + 1

        # 定义一个测试模块 TestModule，其包含一个 ModuleInterface 类型的成员变量
        class TestModule(nn.Module):
            proxy_mod: ModuleInterface

            def __init__(self):
                super().__init__()
                # 初始化时将 proxy_mod 设置为 OrigModule 的实例
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                # 调用 proxy_mod 的 forward 方法
                return self.proxy_mod.forward(input)

        # 将 TestModule 脚本化
        scripted_mod = torch.jit.script(TestModule())
        # 替换模块为不兼容接口的新模块应该引发错误
        with self.assertRaisesRegex(RuntimeError, "is not compatible with interface"):
            scripted_mod.proxy_mod = torch.jit.script(NewModuleWrong())
    def test_module_swap_no_lazy_compile(self):
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                pass

            def forward(self, input: Tensor) -> Tensor:
                pass

        class TestModule(nn.Module):
            proxy_mod: ModuleInterface

            def __init__(self):
                super().__init__()
                # 创建一个原始模块实例并赋给 proxy_mod
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                # 调用 proxy_mod 的 forward 方法
                return self.proxy_mod.forward(input)

        class NewModuleMethodNotLazyCompile(nn.Module):
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                # 实现 one 方法，执行输入张量的元素乘法和加法操作
                return inp1 * inp2 + 1

            def forward(self, input: Tensor) -> Tensor:
                # 实现 forward 方法，对输入张量加1
                return input + 1

        # 对 TestModule 进行 TorchScript 脚本化
        scripted_mod = torch.jit.script(TestModule())
        
        # 尝试用 NewModuleMethodNotLazyCompile 替换 proxy_mod，因为其未在 forward 方法中惰性编译，
        # 所以需要显式地导出该方法以使替换有效
        with self.assertRaisesRegex(RuntimeError, "is not compatible with interface"):
            scripted_mod.proxy_mod = torch.jit.script(NewModuleMethodNotLazyCompile())

        class NewModuleMethodManualExport(nn.Module):
            @torch.jit.export
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                return inp1 * inp2 + 1

            def forward(self, input: Tensor) -> Tensor:
                return input + 1

        # 用 NewModuleMethodManualExport 替换 proxy_mod
        scripted_mod.proxy_mod = torch.jit.script(NewModuleMethodManualExport())
        
        # 创建输入张量并断言替换后的模块行为符合预期
        input = torch.randn(3, 4)
        self.assertEqual(scripted_mod(input), input + 1)

    def test_module_swap_no_module_interface(self):
        # 没有模块接口的模块替换测试
        class TestNoModuleInterface(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个原始模块实例并赋给 proxy_mod
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                # 调用 proxy_mod 的 forward 方法
                return self.proxy_mod(input)

        # 对 TestNoModuleInterface 进行 TorchScript 脚本化
        scripted_no_module_interface = torch.jit.script(TestNoModuleInterface())

        # 用 OrigModule 替换 proxy_mod，两者具有相同的 JIT 类型，应该成功
        scripted_no_module_interface.proxy_mod = torch.jit.script(OrigModule())

        # 尝试用 NewModule 替换 proxy_mod，类型不匹配应该导致失败
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected a value of type '__torch__.jit.test_module_interface.OrigModule \(.*\)' "
            + r"for field 'proxy_mod', but found '__torch__.jit.test_module_interface.NewModule \(.*\)'",
        ):
            scripted_no_module_interface.proxy_mod = torch.jit.script(NewModule())
    def test_script_module_as_interface_swap(self):
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            # 定义接口方法 `one`，接收两个张量输入，返回一个张量
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                pass

            # 定义接口方法 `forward`，接收一个张量输入，返回一个张量
            def forward(self, input: Tensor) -> Tensor:
                pass

        class OrigScriptModule(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 实现接口方法 `one`，返回输入张量的和加 1
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                return inp1 + inp2 + 1

            @torch.jit.script_method
            # 实现接口方法 `forward`，返回输入张量与 `one` 方法结果的和加 1
            def forward(self, input: Tensor) -> Tensor:
                return input + self.one(input, input) + 1

        class NewScriptModule(torch.jit.ScriptModule):
            @torch.jit.script_method
            # 实现接口方法 `one`，返回输入张量元素相乘后加 1
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                return inp1 * inp2 + 1

            @torch.jit.script_method
            # 实现接口方法 `forward`，返回输入张量与 `one` 方法结果的和
            def forward(self, input: Tensor) -> Tensor:
                return self.one(input, input + 1)

        class TestNNModuleWithScriptModule(nn.Module):
            proxy_mod: ModuleInterface

            def __init__(self):
                super().__init__()
                self.proxy_mod = OrigScriptModule()  # 使用原始脚本模块作为代理模块

            def forward(self, input: Tensor) -> Tensor:
                return self.proxy_mod.forward(input)  # 调用代理模块的 `forward` 方法

        input = torch.randn(3, 4)
        scripted_mod = torch.jit.script(TestNNModuleWithScriptModule())
        # 断言脚本化模块对输入张量的计算结果是否等于期望值
        self.assertEqual(scripted_mod(input), 3 * input + 2)

        scripted_mod.proxy_mod = NewScriptModule()  # 替换代理模块为新的脚本模块
        # 断言更新后的脚本化模块对输入张量的计算结果是否等于期望值
        self.assertEqual(scripted_mod(input), input * (input + 1) + 1)

    # The call to forward of proxy_mod cannot be inlined. Making sure
    # Freezing is throwing an error for now.
    def test_freeze_module_with_interface(self):
        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.b = 20

            def forward(self, x):
                return self.b

        class OrigMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 0

            def forward(self, x):
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):
            # 定义接口方法 `forward`，接收一个张量输入，返回一个整数
            def forward(self, x: Tensor) -> int:
                pass

        class TestModule(torch.nn.Module):
            proxy_mod: ModInterface

            def __init__(self):
                super().__init__()
                self.proxy_mod = OrigMod()  # 使用原始模块作为代理模块
                self.sub = SubModule()  # 创建子模块

            def forward(self, x):
                # 返回代理模块和子模块对输入张量的处理结果的和
                return self.proxy_mod(x) + self.sub(x)

        m = torch.jit.script(TestModule())
        m.eval()
        # 冻结模块，确保接口不被内联化，当前阶段抛出错误
        mf = torch._C._freeze_module(m._c)
        # 假设接口没有别名
        mf = torch._C._freeze_module(m._c, freezeInterfaces=True)
        input = torch.tensor([1])
        out_s = m.forward(input)
        out_f = mf.forward(input)
        # 断言脚本化模块与冻结后的模块对相同输入的计算结果是否一致
        self.assertEqual(out_s, out_f)
    def test_freeze_module_with_inplace_mutation_in_interface(self):
        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.b = torch.tensor([1.5])  # 初始化子模块的属性 b，使用张量 [1.5]

            def forward(self, x):
                self.b[0] += 2  # 对属性 b 的第一个元素进行原地加法操作
                return self.b  # 返回更新后的属性 b

            @torch.jit.export
            def getb(self, x):
                return self.b  # 返回属性 b

        class OrigMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.tensor([0.5])  # 初始化原始模块的属性 a，使用张量 [0.5]

            def forward(self, x):
                return self.a  # 返回属性 a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                pass  # 接口定义：forward 方法接受张量 x，并返回张量

        class TestModule(torch.nn.Module):
            proxy_mod: ModInterface

            def __init__(self):
                super().__init__()
                self.proxy_mod = OrigMod()  # 初始化代理模块为 OrigMod 实例
                self.sub = SubModule()  # 初始化子模块为 SubModule 实例

            def forward(self, x):
                y = self.proxy_mod(x)  # 调用代理模块的 forward 方法，计算结果保存在 y
                z = self.sub.getb(x)  # 调用子模块的 getb 方法，获取属性 b 的值，计算结果保存在 z
                return y[0] + z[0]  # 返回 y 和 z 的第一个元素之和作为最终结果

        m = torch.jit.script(TestModule())  # 使用 TorchScript 对 TestModule 进行脚本化
        m.proxy_mod = m.sub  # 将代理模块设置为子模块
        m.sub.b = m.proxy_mod.b  # 更新子模块的属性 b 为代理模块的属性 b
        m.eval()  # 将模型设为评估模式
        mf = torch._C._freeze_module(m._c, freezeInterfaces=True)  # 冻结模型及其接口
    def test_freeze_module_with_interface_and_fork(self):
        # 定义子模块 SubModule，继承自 torch.nn.Module
        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化张量 b，赋值为 [1.5]
                self.b = torch.tensor([1.5])

            def forward(self, x):
                # 修改张量 b 的第一个元素
                self.b[0] += 3.2
                return self.b

        # 定义原始模块 OrigMod，继承自 torch.nn.Module
        class OrigMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化张量 a，赋值为 [0.5]
                self.a = torch.tensor([0.5])

            def forward(self, x):
                return self.a

        # 定义模块接口 ModInterface，继承自 torch.nn.Module
        @torch.jit.interface
        class ModInterface(torch.nn.Module):
            # 声明接口方法 forward，参数 x 是 Tensor 类型，返回值也是 Tensor 类型
            def forward(self, x: Tensor) -> Tensor:
                pass

        # 定义测试模块 TestModule，继承自 torch.nn.Module
        class TestModule(torch.nn.Module):
            proxy_mod: ModInterface

            def __init__(self):
                super().__init__()
                # 初始化 proxy_mod 为 OrigMod 的实例
                self.proxy_mod = OrigMod()
                # 初始化子模块 sub 为 SubModule 的实例
                self.sub = SubModule()

            def forward(self, x):
                # 将 proxy_mod 替换为 sub
                y = self.proxy_mod(x)
                z = self.sub(x)
                return y + z

        # 定义主模块 MainModule，继承自 torch.nn.Module
        class MainModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化 test 为 TestModule 的实例
                self.test = TestModule()

            def forward(self, x):
                # 使用 torch.jit._fork 并行执行 test 模块的 forward 方法
                fut = torch.jit._fork(self.test.forward, x)
                # 执行 test 模块的 forward 方法
                y = self.test(x)
                # 等待并获取并行执行的结果 fut
                z = torch.jit._wait(fut)
                return y + z

        # 创建 MainModule 的 TorchScript 表示并设为评估模式
        m = torch.jit.script(MainModule())
        m.eval()
        # 冻结 TorchScript 模块，包括接口类型的冻结
        mf = torch._C._freeze_module(m._c, freezeInterfaces=True)
    # 定义一个测试类的测试方法，用于验证模块接口的接入性
    def test_module_apis_interface(self):
        # 定义一个接口 ModuleInterface，继承自 nn.Module，包含一个方法 one
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            # 方法 one 接收两个张量输入 inp1 和 inp2，返回一个张量
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                pass

        # 定义一个测试模块 TestModule，继承自 nn.Module
        class TestModule(nn.Module):
            # 定义一个 ModuleInterface 类型的成员变量 proxy_mod
            proxy_mod: ModuleInterface

            # 初始化方法
            def __init__(self):
                super().__init__()
                # 初始化 proxy_mod 成员变量为 OrigModule 的实例
                self.proxy_mod = OrigModule()

            # 前向传播方法
            def forward(self, input):
                # 返回输入 input 的两倍
                return input * 2

            # 导出的方法，使用 torch.jit.export 标记为导出方法
            @torch.jit.export
            def method(self, input):
                # 遍历当前模块及其子模块
                for module in self.modules():
                    # 对 input 逐层进行模块的处理
                    input = module(input)
                # 返回处理后的 input
                return input

        # 使用 assertRaisesRegex 上下文管理器验证脚本化的 TestModule 对象是否抛出异常
        with self.assertRaisesRegex(Exception, "Could not compile"):
            # 将 TestModule 实例化并脚本化
            scripted_mod = torch.jit.script(TestModule())
```