# `.\pytorch\test\jit\test_module_containers.py`

```
# Owner(s): ["oncall: jit"]

# 导入必要的库
import os
import sys
from collections import OrderedDict

# 导入类型提示相关模块
from typing import Any, List, Tuple

# 导入 PyTorch 库
import torch
import torch.nn as nn
from torch.testing._internal.jit_utils import JitTestCase

# 让 test/ 目录下的辅助文件可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 如果直接运行此文件，则抛出运行时错误，建议使用指定方式运行测试
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类 TestModuleContainers，继承自 JitTestCase
class TestModuleContainers(JitTestCase):
    
    # 定义测试方法 test_sequential_intermediary_types
    def test_sequential_intermediary_types(self):
        
        # 定义内部类 A，继承自 torch.nn.Module
        class A(torch.nn.Module):
            
            # A 类的前向传播方法
            def forward(self, x):
                return x + 3

        # 定义内部类 B，继承自 torch.nn.Module
        class B(torch.nn.Module):
            
            # B 类的前向传播方法
            def forward(self, x):
                return {"1": x}

        # 定义内部类 C，继承自 torch.nn.Module
        class C(torch.nn.Module):
            
            # C 类的初始化方法
            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Sequential(A(), B())

            # C 类的前向传播方法
            def forward(self, x):
                return self.foo(x)

        # 使用 JitTestCase 类的 checkModule 方法测试类 C 的前向传播
        self.checkModule(C(), (torch.tensor(1),))
    def test_moduledict(self):
        # 定义内部类 Inner，继承自 torch.nn.Module，重写 forward 方法
        class Inner(torch.nn.Module):
            def forward(self, x):
                return x + 10

        # 定义内部类 Inner2，继承自 torch.nn.Module，重写 forward 方法
        class Inner2(torch.nn.Module):
            def forward(self, x):
                return x * 2

        # 定义内部类 Inner3，继承自 torch.nn.Module，重写 forward 方法
        class Inner3(torch.nn.Module):
            def forward(self, x):
                return (x - 4) * 3

        # 定义类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建有序字典 modules，包含三个键值对，每个值为一个内部类实例
                modules = OrderedDict(
                    [
                        ("one", Inner()),
                        ("two", Inner2()),
                        ("three", Inner3()),
                    ]
                )
                # 使用 nn.ModuleDict 将 modules 转换为 ModuleDict 类型并赋值给 self.moduledict
                self.moduledict = nn.ModuleDict(modules)

            def forward(self, x, skip_name):
                # type: (Tensor, str)
                # 声明一个空列表 names，用于存储模块的名称
                names = torch.jit.annotate(List[str], [])
                # 声明一个空列表 values，用于存储模块处理后的值
                values = []
                # 遍历 self.moduledict 中的键（模块名称），并添加到 names 列表中
                for name in self.moduledict:
                    names.append(name)

                # 遍历 self.moduledict 中的键值对（模块名称和模块实例）
                for name, mod in self.moduledict.items():
                    # 如果当前模块名称不等于 skip_name，则执行以下操作
                    if name != skip_name:
                        # 将当前模块名称添加到 names 列表中
                        names.append(name)
                        # 对输入 x 应用当前模块 mod，将结果添加到 values 列表中
                        x = mod(x)
                        values.append(x)

                # 遍历 self.moduledict 中的模块实例，对输入 x 依次应用每个模块，将结果添加到 values 列表中
                for mod in self.moduledict.values():
                    x = mod(x)
                    values.append(x)

                # 遍历 self.moduledict 中的键（模块名称），并添加到 names 列表中
                for key in self.moduledict.keys():
                    names.append(key)

                # 返回最终处理后的输入 x 和模块名称列表 names
                return x, names

        # 定义类 M2，继承自 M
        class M2(M):
            def forward(self, x, skip_name):
                # type: (Tensor, str)
                # 声明一个空列表 names，用于存储模块的名称
                names = torch.jit.annotate(List[str], [])
                # 声明一个空列表 values，用于存储模块处理后的值
                values = []
                # 将输入 x 复制给 x2
                x2 = x
                # 声明一个整数变量 iter，初始化为 0
                iter = 0
                # 遍历 self.moduledict 中的键（模块名称），并添加到 names 列表中
                for name in self.moduledict:
                    names.append(name)

                # 遍历 self.moduledict 中的键值对（模块名称和模块实例），并获取索引 i 和模块实例 mod
                for i, (name, mod) in enumerate(self.moduledict.items()):
                    iter += i
                    # 如果当前模块名称不等于 skip_name，则执行以下操作
                    if name != skip_name:
                        # 将当前模块名称添加到 names 列表中
                        names.append(name)
                        # 对输入 x 应用当前模块 mod，将结果添加到 values 列表中
                        x = mod(x)
                        values.append(x)

                # 遍历 self.moduledict 中的模块实例，并获取索引 i 和模块实例 mod
                for i, mod in enumerate(self.moduledict.values()):
                    iter += i
                    # 对输入 x 依次应用每个模块，将结果添加到 values 列表中
                    x = mod(x)
                    values.append(x)

                # 遍历 self.moduledict 中的键，并获取索引 i 和键 key
                for i, key in enumerate(self.moduledict.keys()):
                    iter += i
                    # 将当前模块名称添加到 names 列表中
                    names.append(key)

                # 使用 zip 函数同时遍历 self.moduledict.values()，并获取模块实例 mod
                for mod, mod in zip(self.moduledict.values(), self.moduledict.values()):
                    iter += i
                    # 对输入 x2 应用两次当前模块 mod，将结果重新赋值给 x2
                    x2 = mod(mod(x2))

                # 返回处理后的输入 x，x2，模块名称列表 names，以及迭代次数 iter
                return x, x2, names, iter

        # 遍历列表中的字符串元素
        for name in ["", "one", "two", "three"]:
            # 创建一个张量 inp，其值为 1
            inp = torch.tensor(1)
            # 调用 self.checkModule 方法，分别传入 M 类实例和元组 (inp, name)
            self.checkModule(M(), (inp, name))
            # 调用 self.checkModule 方法，分别传入 M2 类实例和元组 (inp, name)
            self.checkModule(M2(), (inp, name))
    def test_custom_container_forward(self):
        class Inner(torch.nn.Module):
            def forward(self, x):
                return x + 10

        # 定义一个自定义的序列容器，继承自 nn.Sequential
        class CustomSequential(nn.Sequential):
            def __init__(self):
                super().__init__(nn.ReLU(), Inner())

            def forward(self, x):
                x = x + 3  # 输入张量加 3
                # 遍历容器中的模块，依次对输入进行处理
                for mod in self:
                    x = mod(x)
                return x - 5  # 返回处理后的张量减去 5

        # 调用自定义序列容器的测试函数
        self.checkModule(CustomSequential(), (torch.tensor(0.5),))

        # 定义一个自定义的模块列表，继承自 nn.ModuleList
        class CustomModuleList(nn.ModuleList):
            def __init__(self):
                super().__init__([nn.ReLU(), Inner()])

            def forward(self, x):
                x = x + 3  # 输入张量加 3
                # 遍历模块列表中的模块，依次对输入进行处理
                for mod in self:
                    x = mod(x)
                return x - 5  # 返回处理后的张量减去 5

        # 调用自定义模块列表的测试函数
        self.checkModule(CustomModuleList(), (torch.tensor(0.5),))

        # 定义一个自定义的模块字典，继承自 nn.ModuleDict
        class CustomModuleDict(nn.ModuleDict):
            def __init__(self):
                super().__init__(
                    OrderedDict(
                        [
                            ("one", Inner()),
                            ("two", nn.ReLU()),
                            ("three", Inner()),
                        ]
                    )
                )

            def forward(self, x):
                x = x + 3  # 输入张量加 3
                names = torch.jit.annotate(List[str], [])  # 创建一个空的字符串列表
                # 遍历模块字典中的键值对，对输入进行处理并记录模块名称
                for name, mod in self.items():
                    x = mod(x)
                    names.append(name)
                return names, x - 5  # 返回模块名称列表和处理后的张量减去 5

        # 调用自定义模块字典的测试函数
        self.checkModule(CustomModuleDict(), (torch.tensor(0.5),))

    # 定义测试脚本模块列表和顺序容器的函数
    def test_script_module_list_sequential(self):
        # 定义一个继承自 torch.jit.ScriptModule 的脚本模块
        class M(torch.jit.ScriptModule):
            def __init__(self, mod_list):
                super().__init__()
                self.mods = mod_list

            @torch.jit.script_method
            def forward(self, v):
                # 对模块列表中的每个模块进行迭代，对输入进行处理
                for m in self.mods:
                    v = m(v)
                return v

        # 关闭优化执行
        with torch.jit.optimized_execution(False):
            # 创建 M 类的实例，传入 nn.Sequential(nn.ReLU()) 作为模块列表
            m = M(nn.Sequential(nn.ReLU()))
            # 调用断言导出导入模块的函数，检查模块是否可以成功导出和导入
            self.assertExportImportModule(m, (torch.randn(2, 2),))
    def test_script_modulelist_index(self):
        # 定义一个名为 test_script_modulelist_index 的测试函数
        class Sub(torch.nn.Module):
            # 定义一个名为 Sub 的子类，继承自 torch.nn.Module
            def __init__(self, i):
                # 初始化方法，接受一个参数 i
                super().__init__()
                self.i = i
                # 将参数 i 赋值给实例变量 self.i

            def forward(self, thing):
                # 前向传播方法，接受一个参数 thing
                return thing - self.i
                # 返回 thing 减去实例变量 self.i 的结果作为输出

        class M(torch.nn.Module):
            # 定义一个名为 M 的子类，继承自 torch.nn.Module
            def __init__(self):
                # 初始化方法
                super().__init__()
                # 调用父类的初始化方法
                self.mods = nn.ModuleList([Sub(i) for i in range(10)])
                # 创建一个包含 10 个 Sub 类实例的 ModuleList 并赋值给实例变量 self.mods

            def forward(self, v):
                # 前向传播方法，接受一个参数 v
                v = self.mods[4].forward(v)
                # 调用 ModuleList 中索引为 4 的 Sub 实例的 forward 方法
                v = self.mods[-1].forward(v)
                # 调用 ModuleList 中最后一个 Sub 实例的 forward 方法
                v = self.mods[-9].forward(v)
                # 调用 ModuleList 中倒数第九个 Sub 实例的 forward 方法
                return v
                # 返回处理后的结果 v

        x = torch.tensor(1)
        # 创建一个值为 1 的 tensor 对象 x
        self.checkModule(M(), (x,))
        # 调用测试类中的 checkModule 方法，传入 M 类的实例和一个包含 x 的元组作为参数

        class MForward(torch.nn.Module):
            # 定义一个名为 MForward 的子类，继承自 torch.nn.Module
            def __init__(self):
                # 初始化方法
                super().__init__()
                # 调用父类的初始化方法
                self.mods = nn.ModuleList([Sub(i) for i in range(10)])
                # 创建一个包含 10 个 Sub 类实例的 ModuleList 并赋值给实例变量 self.mods

            def forward(self, v):
                # 前向传播方法，接受一个参数 v
                v = self.mods[4](v)
                # 调用 ModuleList 中索引为 4 的 Sub 实例的 forward 方法
                v = self.mods[-1](v)
                # 调用 ModuleList 中最后一个 Sub 实例的 forward 方法
                v = self.mods[-9](v)
                # 调用 ModuleList 中倒数第九个 Sub 实例的 forward 方法
                return v
                # 返回处理后的结果 v

        self.checkModule(MForward(), (torch.tensor(1),))
        # 调用测试类中的 checkModule 方法，传入 MForward 类的实例和一个包含 tensor 对象的元组作为参数

        class M2(M):
            # 定义一个名为 M2 的子类，继承自 M 类
            def forward(self, v):
                # 重写的前向传播方法，接受一个参数 v
                return self.mods[-11].forward(v)
                # 调用 ModuleList 中索引为 -11 的 Sub 实例的 forward 方法

        with self.assertRaisesRegexWithHighlight(
            Exception, "Index -11 out of range", "self.mods[-11]"
        ):
            # 使用 with 语句捕获预期的异常
            torch.jit.script(M2())
            # 将 M2 类的实例转换为 Torch 脚本

        class M3(M):
            # 定义一个名为 M3 的子类，继承自 M 类
            def forward(self, v):
                # 重写的前向传播方法，接受一个参数 v
                i = 3
                # 定义局部变量 i，赋值为 3
                return self.mods[i].forward(v)
                # 调用 ModuleList 中索引为变量 i 的 Sub 实例的 forward 方法

        with self.assertRaisesRegexWithHighlight(
            Exception, "Enumeration is supported", "self.mods[i]"
        ):
            # 使用 with 语句捕获预期的异常
            torch.jit.script(M3())
            # 将 M3 类的实例转换为 Torch 脚本

        class M4(M):
            # 定义一个名为 M4 的子类，继承自 M 类
            def forward(self, v):
                # 重写的前向传播方法，接受一个参数 v
                i = 3
                # 定义局部变量 i，赋值为 3
                return self.mods[i].forward(v)
                # 调用 ModuleList 中索引为变量 i 的 Sub 实例的 forward 方法

        with self.assertRaisesRegex(Exception, "will fail because i is not a literal"):
            # 使用 with 语句捕获预期的异常
            torch.jit.script(M4())
            # 将 M4 类的实例转换为 Torch 脚本
    def test_special_method_with_override(self):
        # 定义一个自定义模块接口 CustomModuleInterface，继承自 torch.nn.Module
        class CustomModuleInterface(torch.nn.Module):
            pass

        # 定义一个自定义模块列表 CustomModuleList，继承自 CustomModuleInterface 和 torch.nn.ModuleList
        class CustomModuleList(CustomModuleInterface, torch.nn.ModuleList):
            def __init__(self, modules=None):
                # 调用 CustomModuleInterface 的初始化方法
                CustomModuleInterface.__init__(self)
                # 调用 torch.nn.ModuleList 的初始化方法，传入模块列表 modules
                torch.nn.ModuleList.__init__(self, modules)

            def __len__(self):
                # 重写 __len__ 方法，返回固定值 2
                # 这是任意的值，用来验证 CustomModuleList 中自定义的 __len__ 方法会覆盖 JIT 编译器自动生成的 __len__
                return 2

        # 定义一个包含特定设置的 MyModule 类，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 通过 torch.jit.script 预先脚本化 ReLU，以解决 'is' 操作符的别名问题
                self.submod = torch.jit.script(torch.nn.ReLU())
                # 创建 CustomModuleList 实例，包含已脚本化的 ReLU 模块
                self.modulelist = CustomModuleList([self.submod])

            def forward(self, inputs):
                # 断言检查 self.modulelist 的长度是否为 2，验证 ModuleList 的 __len__ 方法是否正常工作
                assert len(self.modulelist) == 2, "__len__ failing for ModuleList"
                return inputs

        # 创建 MyModule 实例 m
        m = MyModule()
        # 使用 self.checkModule 方法检查模块 m
        self.checkModule(m, [torch.randn(2, 2)])
        # 使用 torch.jit.script 将模块 m 脚本化，赋值给 mm
        mm = torch.jit.script(m)

    def test_moduledict_getitem(self):
        # 定义一个包含特定设置的 MyModule 类，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 通过 torch.jit.script 预先脚本化 ReLU 和 Tanh 模块
                self.relu = torch.jit.script(torch.nn.ReLU())
                self.tanh = torch.jit.script(torch.nn.Tanh())
                # 创建 ModuleDict，包含键 "relu" 和 "tanh" 对应的脚本化模块
                self.moduledict = torch.nn.ModuleDict(
                    {"relu": self.relu, "tanh": self.tanh}
                )

            def forward(self, input):
                # 使用断言检查 ModuleDict 的索引操作是否返回正确的模块对象
                assert self.moduledict["relu"] is self.relu
                assert self.moduledict["tanh"] is self.tanh
                return input

        # 创建 MyModule 实例 m
        m = MyModule()
        # 使用 self.checkModule 方法检查模块 m
        self.checkModule(m, [torch.randn(2, 2)])
    def test_empty_dict_override_contains():
        class CustomModuleInterface(torch.nn.Module):
            pass

        class CustomModuleDict(CustomModuleInterface, torch.nn.ModuleDict):
            def __init__(self, modules=None):
                CustomModuleInterface.__init__(self)
                torch.nn.ModuleDict.__init__(self, modules)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # work around aliasing issue for 'is' operator by scripting ReLU up front
                self.submod = torch.jit.script(torch.nn.ReLU())  # 将 torch.nn.ReLU() 脚本化并赋值给 submod
                self.moduledict = CustomModuleDict()  # 初始化自定义的 ModuleDict 对象

            def forward(self, inputs):
                assert (
                    "submod" not in self.moduledict
                ), "__contains__ fails for ModuleDict"  # 检查是否包含 submod 键，应该为 False
                return inputs

        m = MyModule()
        self.checkModule(m, [torch.randn(2, 2)])


注释：
    def test_typed_module_dict(self):
        """
        Test that a type annotation can be provided for a ModuleDict that allows
        non-static indexing.
        """

        @torch.jit.interface
        class ModuleInterface(torch.nn.Module):
            def forward(self, inp: Any) -> Any:
                pass

        class ImplementsInterface(torch.nn.Module):
            def forward(self, inp: Any) -> Any:
                if isinstance(inp, torch.Tensor):
                    return torch.max(inp, dim=0)

                return inp

        class DoesNotImplementInterface(torch.nn.Module):
            def forward(self, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                return torch.max(inp, dim=0)

        # Test annotation of submodule.
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个 ModuleDict 对象，包含一个实现了 ModuleInterface 接口的子模块
                self.d = torch.nn.ModuleDict({"module": ImplementsInterface()})

            def forward(self, x: torch.Tensor, key: str) -> Any:
                # 指定 d[key] 的类型为 ModuleInterface
                value: ModuleInterface = self.d[key]
                return value.forward(x)

        m = Mod()
        self.checkModule(m, (torch.randn(2, 2), "module"))

        # Test annotation of self.
        class ModDict(torch.nn.ModuleDict):
            def __init__(self):
                super().__init__({"module": ImplementsInterface()})

            def forward(self, x: torch.Tensor, key: str) -> Any:
                # 指定 self[key] 的类型为 ModuleInterface
                submodule: ModuleInterface = self[key]
                return submodule.forward(x)

        m = ModDict()
        self.checkModule(m, (torch.randn(2, 2), "module"))

        # Test error message thrown when annotated attribute does not comply with the
        # annotation.
        class ModWithWrongAnnotation(torch.nn.ModuleDict):
            def __init__(self):
                super().__init__()
                # 创建一个 ModuleDict 对象，包含一个未实现 ModuleInterface 接口的子模块
                self.d = torch.nn.ModuleDict({"module": DoesNotImplementInterface()})

            def forward(self, x: torch.Tensor, key: str) -> Any:
                # 尝试访问 self.d[key]，期望其类型为 ModuleInterface
                submodule: ModuleInterface = self.d[key]
                return submodule.forward(x)

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Attribute module is not of annotated type", "self.d[key]"
        ):
            # 使用脚本化方法尝试运行 ModWithWrongAnnotation 类，预期抛出类型不符的异常信息
            torch.jit.script(ModWithWrongAnnotation())
    # 定义一个测试类方法，用于测试带有类型注解的 ModuleList 是否支持非静态索引。
    def test_typed_module_list(self):
        """
        Test that a type annotation can be provided for a ModuleList that allows
        non-static indexing.
        """

        # 定义一个 TorchScript 接口 ModuleInterface，继承自 torch.nn.Module，要求实现 forward 方法
        @torch.jit.interface
        class ModuleInterface(torch.nn.Module):
            def forward(self, inp: Any) -> Any:
                pass

        # 实现了 ModuleInterface 接口的具体类 ImplementsInterface
        class ImplementsInterface(torch.nn.Module):
            def forward(self, inp: Any) -> Any:
                # 如果输入是 torch.Tensor，则返回其在第 0 维上的最大值及其索引
                if isinstance(inp, torch.Tensor):
                    return torch.max(inp, dim=0)

                return inp

        # 不实现 ModuleInterface 接口的类 DoesNotImplementInterface，forward 方法返回两个 torch.Tensor
        class DoesNotImplementInterface(torch.nn.Module):
            def forward(self, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                return torch.max(inp, dim=0)

        # 测试子模块的注解。
        # 定义一个模块 Mod，继承自 torch.nn.Module，包含一个 ModuleList，其中包含实现了 ModuleInterface 接口的实例
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.ModuleList([ImplementsInterface()])

            def forward(self, x: torch.Tensor, idx: int) -> Any:
                # 声明变量 value 为 ModuleInterface 类型，获取 ModuleList 中索引为 idx 的子模块
                value: ModuleInterface = self.l[idx]
                return value.forward(x)

        # 创建 Mod 的实例 m，并对其进行模块检查
        m = Mod()
        self.checkModule(m, (torch.randn(2, 2), 0))

        # 测试 self 的注解。
        # 定义一个模块 ModList，继承自 torch.nn.ModuleList，包含一个 ModuleList，其中包含实现了 ModuleInterface 接口的实例
        class ModList(torch.nn.ModuleList):
            def __init__(self):
                super().__init__([ImplementsInterface()])

            def forward(self, x: torch.Tensor, idx: int) -> Any:
                # 声明变量 submodule 为 ModuleInterface 类型，获取 ModuleList 中索引为 idx 的子模块
                submodule: ModuleInterface = self[idx]
                return submodule.forward(x)

        # 创建 ModList 的实例 m，并对其进行模块检查
        m = ModList()
        self.checkModule(m, (torch.randn(2, 2), 0))

        # 测试当带注解的属性与注解不符合时抛出错误消息。
        # 定义一个模块 ModWithWrongAnnotation，继承自 torch.nn.ModuleList，包含一个 ModuleList，其中包含未实现 ModuleInterface 接口的实例
        class ModWithWrongAnnotation(torch.nn.ModuleList):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.ModuleList([DoesNotImplementInterface()])

            def forward(self, x: torch.Tensor, idx: int) -> Any:
                # 声明变量 submodule 为 ModuleInterface 类型，获取 ModuleList 中索引为 idx 的子模块
                submodule: ModuleInterface = self.l[idx]
                return submodule.forward(x)

        # 使用 assertRaisesRegexWithHighlight 断言，期望运行时错误抛出指定的错误消息
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Attribute 0 is not of annotated type", "self.l[idx]"
        ):
            torch.jit.script(ModWithWrongAnnotation())
    def test_module_properties(self):
        # 定义一个带属性的模块类，继承自 torch.nn.Module
        class ModuleWithProperties(torch.nn.Module):
            # 忽略的属性列表，在 JIT 编译时不使用的属性
            __jit_unused_properties__ = ["ignored_attr"]

            # 初始化方法，接受一个整数参数 a
            def __init__(self, a: int):
                super().__init__()
                self.a = a  # 设置实例变量 self.a

            # 前向传播方法，接受两个整数参数 a 和 b
            def forward(self, a: int, b: int):
                self.attr = a + b  # 设置实例属性 self.attr 为 a + b
                return self.attr  # 返回属性 self.attr 的值

            # 属性装饰器，返回实例变量 self.a
            @property
            def attr(self):
                return self.a

            # 属性装饰器，返回 self.a 的和
            @property
            def ignored_attr(self):
                return sum([self.a])

            # 被 torch.jit.unused 装饰的属性，返回 self.a 的和
            @torch.jit.unused
            @property
            def ignored_attr_2(self):
                return sum([self.a])

            # ignored_attr_2 的 setter 方法，设置 self.a 为 self.a 的和
            @ignored_attr_2.setter
            def ignored_attr_2(self, value):
                self.a = sum([self.a])

            # attr 的 setter 方法，如果 a 大于 0，则设置 self.a 为 a，否则为 0
            @attr.setter
            def attr(self, a: int):
                if a > 0:
                    self.a = a
                else:
                    self.a = 0

        # 定义一个不带 setter 方法的模块类，继承自 torch.nn.Module
        class ModuleWithNoSetter(torch.nn.Module):
            # 初始化方法，接受一个整数参数 a
            def __init__(self, a: int):
                super().__init__()
                self.a = a  # 设置实例变量 self.a

            # 前向传播方法，接受两个整数参数 a 和 b
            def forward(self, a: int, b: int):
                self.attr + a + b  # 计算属性 self.attr + a + b

            # 属性装饰器，返回 self.a + 1
            @property
            def attr(self):
                return self.a + 1

        # 使用 self.checkModule 方法验证不同模块的输出
        self.checkModule(
            ModuleWithProperties(5),  # 使用 ModuleWithProperties 创建实例
            (
                5,  # 预期输出为 5
                6,  # 预期输出为 6
            ),
        )
        self.checkModule(
            ModuleWithProperties(5),  # 使用 ModuleWithProperties 创建实例
            (
                -5,  # 预期输出为 -5
                -6,  # 预期输出为 -6
            ),
        )
        self.checkModule(
            ModuleWithNoSetter(5),  # 使用 ModuleWithNoSetter 创建实例
            (
                5,  # 预期输出为 5
                6,  # 预期输出为 6
            ),
        )
        self.checkModule(
            ModuleWithNoSetter(5),  # 使用 ModuleWithNoSetter 创建实例
            (
                -5,  # 预期输出为 -5
                -6,  # 预期输出为 -6
            ),
        )

        mod = ModuleWithProperties(3)  # 创建 ModuleWithProperties 实例 mod
        scripted_mod = torch.jit.script(mod)  # 对 mod 进行脚本化

        # 使用 assertRaisesRegex 检查是否抛出 AttributeError 异常
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            scripted_mod.ignored_attr  # 访问脚本化模块的 ignored_attr 属性
    def test_module_inplace_construct(self):
        # 定义一个内嵌的 nn.Module 类 M
        class M(nn.Module):
            # 初始化方法，接受一个整数 start
            def __init__(self, start: int):
                super().__init__()
                # 创建一个线性层，输入和输出都是3维
                self.linear = nn.Linear(3, 3)
                # 设置一个属性 attribute，初始值为 start
                self.attribute = start
                # 创建一个参数，值为3，类型为 float 的张量
                self.parameter = nn.Parameter(torch.tensor(3, dtype=torch.float))

            # 返回属性 attribute 的方法
            def method(self) -> int:
                return self.attribute

            # 未使用的方法，在 Torch 脚本中将被忽略
            @torch.jit.unused
            def unused_method(self):
                return self.attribute + self.attribute

            # 前向传播方法，对输入 x 执行两次线性层操作
            def forward(self, x):
                return self.linear(self.linear(x))

        # 定义一个内嵌的 nn.Module 类 N
        class N(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个线性层，输入和输出都是4维
                self.linear = nn.Linear(4, 4)

            # 在 Torch 脚本中忽略的方法
            @torch.jit.ignore
            def ignored_method(self, x):
                return x

            # 前向传播方法，对输入 x 执行一次线性层操作
            def forward(self, x):
                return self.linear(x)

        # 使用 Torch 脚本对 M 类进行脚本化，传入 start 值为 3
        m = torch.jit.script(M(3))
        # 使用 Torch 脚本对 N 类进行脚本化
        n = torch.jit.script(N())

        # 将 n 模块重构为 m 模块的参数状态
        n._reconstruct(m._c)

        # 创建一个大小为 3 的随机张量作为输入
        inp = torch.rand((3))

        # 使用 torch.no_grad() 上下文，检查 m 和 n 模块产生的输出是否相同
        with torch.no_grad():
            m_out = m(inp)
            n_out = n(inp)
            self.assertEqual(m_out, n_out)

        # 检查忽略方法 ignored_method 是否保持不变
        self.assertEqual(inp, n.ignored_method(inp))

    def test_parameterlist_script_getitem(self):
        # 定义一个内嵌的 nn.Module 类 MyModule
        class MyModule(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建包含 10 个 nn.Linear(1, 1) 模块的 ModuleList
                self.module_list = nn.ModuleList([nn.Linear(1, 1) for _ in range(10)])
                # 创建包含 10 个零张量参数的 ParameterList
                self.parameter_list = nn.ParameterList(
                    [nn.Parameter(torch.zeros(1)) for _ in range(10)]
                )

            # 前向传播方法，简单地返回输入 x
            def forward(self, x):
                # 访问第一个 module_list 中的模块
                self.module_list[0]
                # 访问第一个 parameter_list 中的参数
                self.parameter_list[0]
                return x

        # 使用 checkModule 方法检查 MyModule 的行为
        self.checkModule(MyModule(), (torch.zeros(1)))

    def test_parameterlist_script_iter(self):
        # 定义一个内嵌的 nn.Module 类 MyModule
        class MyModule(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建包含 10 个 nn.Linear(1, 1) 模块的 ModuleList
                self.module_list = nn.ModuleList([nn.Linear(1, 1) for _ in range(10)])
                # 创建包含 10 个零张量参数的 ParameterList
                self.parameter_list = nn.ParameterList(
                    [nn.Parameter(torch.zeros(1)) for _ in range(10)]
                )

            # 前向传播方法，对输入 x 执行一系列操作
            def forward(self, x):
                r = x
                # 遍历 parameter_list 中的参数，并将其加到结果 r 中
                for i, p in enumerate(self.parameter_list):
                    r = r + p + i
                return r

        # 使用 checkModule 方法检查 MyModule 的行为
        self.checkModule(MyModule(), (torch.zeros(1),))
    def test_parameterdict_script_getitem(self):
        # 定义一个名为 MyModule 的子类，继承自 nn.Module
        class MyModule(nn.Module):
            # 构造函数，初始化模块
            def __init__(self):
                # 调用父类的构造函数
                super().__init__()
                # 创建一个参数字典 parameter_dict，包含三个参数 'a', 'b', 'c'，初始值为零的参数
                self.parameter_dict = nn.ParameterDict(
                    {k: nn.Parameter(torch.zeros(1)) for k in ["a", "b", "c"]}
                )

            # 前向传播函数
            def forward(self, x):
                # 返回计算结果，使用参数字典中 'a' 参数乘以输入 x，加上 'b' 参数乘以 'c' 参数
                return (
                    self.parameter_dict["a"] * x
                    + self.parameter_dict["b"] * self.parameter_dict["c"]
                )

        # 调用外部函数 checkModule，检查 MyModule 实例的输出
        self.checkModule(MyModule(), (torch.ones(1),))
```