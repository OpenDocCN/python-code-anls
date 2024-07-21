# `.\pytorch\test\jit\test_hooks_modules.py`

```py
# Owner(s): ["oncall: jit"]

# 导入需要的模块
from typing import List, Tuple

import torch


# 表示没有 forward 方法的子模块
class SubmoduleNoForwardInputs(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self):
        # 断言子模块名称为 "inner_mod_name"
        assert self.name == "inner_mod_name"


# 表示没有 forward 方法的模块
class ModuleNoForwardInputs(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        # 初始化子模块
        self.submodule = SubmoduleNoForwardInputs(submodule_name)

    def forward(self):
        # 调用子模块的 forward 方法
        self.submodule()


# 具有单个输入的子模块，具有一个简单的处理函数 foo
class SubmoduleForwardSingleInput(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def foo(self, input: str):
        return input

    def forward(self, input: str):
        # 对输入进行处理，然后调用 foo 函数处理
        input = input + "_inner_mod"
        input = self.foo(input)
        return input


# 具有单个输入的模块，包含一个子模块 SubmoduleForwardSingleInput
class ModuleForwardSingleInput(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        # 初始化子模块
        self.submodule = SubmoduleForwardSingleInput(submodule_name)

    def forward(self, input: str):
        # 对输入进行处理，然后调用子模块的 forward 方法
        input = input + "_outermod"
        return self.submodule(input)


# 直接调用子模块的 forward 方法的模块
class ModuleDirectforwardSubmodCall(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        # 初始化子模块
        self.submodule = SubmoduleForwardSingleInput(submodule_name)

    def forward(self, input: str):
        # 对输入进行处理，然后直接调用子模块的 forward 方法
        input = input + "_outermod"
        return self.submodule.forward(input)


# 具有多个输入的子模块
class SuboduleForwardMultipleInputs(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, input1: List[str], input2: str):
        # 修改输入列表，返回处理后的结果
        input1.append(self.name)
        output2 = input2 + "_"
        return input1, output2


# 具有多个输入的模块，包含一个子模块 SuboduleForwardMultipleInputs
class ModuleForwardMultipleInputs(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        # 初始化子模块
        self.submodule = SuboduleForwardMultipleInputs(submodule_name)

    def forward(self, input1: List[str], input2: str):
        # 修改输入列表并返回，调用子模块的 forward 方法
        input1.append(self.name)
        return self.submodule(input1, input2)


# 具有元组输入的子模块
class SubmoduleForwardTupleInput(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, input: Tuple[int]):
        # 访问元组的第一个元素并返回新的元组
        input_access = input[0]
        return (1,)


# 具有元组输入的模块，包含一个子模块 SubmoduleForwardTupleInput
class ModuleForwardTupleInput(torch.nn.Module):
    def __init__(self, name: str, submodule_name: str):
        super().__init__()
        self.name = name
        # 初始化子模块
        self.submodule = SubmoduleForwardTupleInput(submodule_name)

    def forward(self, input: Tuple[int]):
        # 访问元组的第一个元素并返回，调用子模块的 forward 方法
        input_access = input[0]
        return self.submodule((1,))


# 用于测试没有 forward 方法的模块级钩子
def create_module_no_forward_input():
    # 用于测试没有 forward 方法的模块级钩子
    # 创建一个 ModuleNoForwardInputs 类的实例，传入外层模块名和内层模块名作为参数
    m = ModuleNoForwardInputs("outer_mod_name", "inner_mod_name")

    # 定义一个前处理钩子函数 pre_hook，确保当前对象的名称为 "outer_mod_name"
    def pre_hook(self, input: Tuple[()]) -> None:
        assert self.name == "outer_mod_name"

    # 定义一个正向传播钩子函数 forward_hook，确保当前对象的名称为 "outer_mod_name"
    def forward_hook(self, input: Tuple[()], output: None):
        assert self.name == "outer_mod_name"

    # 注册前处理钩子函数 pre_hook 到模块 m 上
    m.register_forward_pre_hook(pre_hook)
    # 注册正向传播钩子函数 forward_hook 到模块 m 上
    m.register_forward_hook(forward_hook)

    # 返回注册了钩子函数的模块 m
    return m
def create_submodule_no_forward_input():
    # 用于测试没有前向输入的子模块级别钩子
    m = ModuleNoForwardInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[()]) -> None:
        assert self.name == "inner_mod_name"

    def forward_hook(self, input: Tuple[()], output: None):
        assert self.name == "inner_mod_name"

    # 注册前向预处理钩子函数
    m.submodule.register_forward_pre_hook(pre_hook)
    # 注册前向钩子函数
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_module_forward_multiple_inputs():
    # 用于测试前向具有多个输入和返回的模块级别钩子
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        assert self.name == "outer_mod_name"
        assert input[0][0] == "a"
        return ["pre_hook_override_name"], "pre_hook_override"

    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        assert self.name == "outer_mod_name"
        assert input[0][0] == "pre_hook_override_name"
        output2 = output[1] + "fh"
        return output[0], output2

    # 注册前向预处理钩子函数
    m.register_forward_pre_hook(pre_hook)
    # 注册前向钩子函数
    m.register_forward_hook(forward_hook)

    return m


def create_module_multiple_hooks_multiple_inputs():
    # 用于测试具有多个输入的模块级别钩子的执行顺序，并在它们之间传递正确的信息
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    def pre_hook1(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        assert self.name == "outer_mod_name"
        assert input[0][0] == "a"
        return ["pre_hook_override_name"], "pre_hook_override"

    def pre_hook2(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        assert self.name == "outer_mod_name"
        assert input[0][0] == "pre_hook_override_name"
        return ["pre_hook_override_name2"], "pre_hook_override"

    def forward_hook1(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str]
    ):
        assert self.name == "outer_mod_name"
        assert input[0][0] == "pre_hook_override_name2"
        output2 = output[1] + "fh1"
        return output[0], output2

    def forward_hook2(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str]
    ):
        assert self.name == "outer_mod_name"
        assert input[0][0] == "pre_hook_override_name2"
        assert output[1] == "pre_hook_override_fh1"
        output2 = output[1] + "_fh2"
        return output[0], output2

    # 注册前向预处理钩子函数
    m.register_forward_pre_hook(pre_hook1)
    m.register_forward_pre_hook(pre_hook2)
    # 注册前向钩子函数
    m.register_forward_hook(forward_hook1)
    m.register_forward_hook(forward_hook2)

    return m


def create_module_forward_single_input():
    # 用于测试单输入前向的模块级别钩子
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
    # 定义一个预处理钩子函数，接收一个字符串元组作为输入并返回一个字符串元组
    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        # 断言当前对象的名称为 "outer_mod_name"
        assert self.name == "outer_mod_name"
        # 断言输入的第一个元素为 "a"
        assert input[0] == "a"
        # 返回修改后的元组，包含一个字符串 "pre_hook_override_name"
        return ("pre_hook_override_name",)

    # 定义一个前向钩子函数，接收一个字符串元组作为输入和一个字符串作为输出
    def forward_hook(self, input: Tuple[str], output: str):
        # 断言当前对象的名称为 "outer_mod_name"
        assert self.name == "outer_mod_name"
        # 断言输入为 ("pre_hook_override_name",)
        assert input == ("pre_hook_override_name",)
        # 将输出字符串末尾添加 "_fh" 后返回
        output = output + "_fh"
        return output

    # 将定义好的预处理钩子函数注册到模型 m 上
    m.register_forward_pre_hook(pre_hook)
    # 将定义好的前向钩子函数注册到模型 m 上
    m.register_forward_hook(forward_hook)

    # 返回已经注册了钩子函数的模型 m
    return m
# 创建一个测试函数，用于演示模块可以多次运行相同的前向和前向后钩子
def create_module_same_hook_repeated():
    # 创建一个 ModuleForwardSingleInput 实例，外部模块名为 "outer_mod_name"，内部模块名为 "inner_mod_name"
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    # 定义一个前向前钩子函数，修改输入的第一个元素并返回
    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "outer_mod_name"
        input_change = input[0] + "_ph"
        return (input_change,)

    # 定义一个前向后钩子函数，修改输出并返回
    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "outer_mod_name"
        assert input == ("a_ph_ph",)
        output = output + "_fh"
        return output

    # 注册前向前钩子函数 pre_hook 两次
    m.register_forward_pre_hook(pre_hook)
    m.register_forward_pre_hook(pre_hook)

    # 注册前向后钩子函数 forward_hook 两次
    m.register_forward_hook(forward_hook)
    m.register_forward_hook(forward_hook)

    # 返回创建的模块实例 m
    return m


# 创建一个测试函数，用于测试模块级别的钩子返回空
def create_module_hook_return_nothing():
    # 创建一个 ModuleForwardSingleInput 实例，外部模块名为 "outer_mod_name"，内部模块名为 "inner_mod_name"
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    # 定义一个前向前钩子函数，验证输入并不返回任何内容
    def pre_hook(self, input: Tuple[str]) -> None:
        assert self.name == "outer_mod_name"
        assert input[0] == "a"

    # 定义一个前向后钩子函数，验证输入并不返回任何内容
    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "outer_mod_name"
        assert input == ("a",)

    # 注册前向前钩子函数 pre_hook
    m.register_forward_pre_hook(pre_hook)
    # 注册前向后钩子函数 forward_hook
    m.register_forward_hook(forward_hook)

    # 返回创建的模块实例 m
    return m


# 创建一个测试函数，用于测试模块可以运行多个钩子且每个钩子有单一输入
def create_module_multiple_hooks_single_input():
    # 创建一个 ModuleForwardSingleInput 实例，外部模块名为 "outer_mod_name"，内部模块名为 "inner_mod_name"
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    # 定义第一个前向前钩子函数，修改输入并返回新的输入
    def pre_hook1(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "outer_mod_name"
        assert input[0] == "a"
        return ("pre_hook_override_name1",)

    # 定义第二个前向前钩子函数，修改输入并返回新的输入
    def pre_hook2(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "outer_mod_name"
        assert input[0] == "pre_hook_override_name1"
        return ("pre_hook_override_name2",)

    # 定义第一个前向后钩子函数，修改输出并返回新的输出
    def forward_hook1(self, input: Tuple[str], output: str):
        assert self.name == "outer_mod_name"
        assert input == ("pre_hook_override_name2",)
        assert output == "pre_hook_override_name2_outermod_inner_mod"
        output = output + "_fh1"
        return output, output

    # 定义第二个前向后钩子函数，修改输出并返回新的输出
    def forward_hook2(self, input: Tuple[str], output: Tuple[str, str]):
        assert self.name == "outer_mod_name"
        assert input == ("pre_hook_override_name2",)
        assert output[0] == "pre_hook_override_name2_outermod_inner_mod_fh1"
        output = output[0] + "_fh2"
        return output

    # 注册第一个前向前钩子函数 pre_hook1
    m.register_forward_pre_hook(pre_hook1)
    # 注册第二个前向前钩子函数 pre_hook2
    m.register_forward_pre_hook(pre_hook2)
    # 注册第一个前向后钩子函数 forward_hook1
    m.register_forward_hook(forward_hook1)
    # 注册第二个前向后钩子函数 forward_hook2
    m.register_forward_hook(forward_hook2)

    # 返回创建的模块实例 m
    return m


# 创建一个测试函数，用于测试子模块可以运行具有多个前向输入的钩子
def create_submodule_forward_multiple_inputs():
    # 创建一个 ModuleForwardMultipleInputs 实例，外部模块名为 "outer_mod_name"，内部模块名为 "inner_mod_name"
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")
    # 定义一个方法 `pre_hook`，用于预处理钩子函数
    def pre_hook(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        # 断言当前对象的名称为 "inner_mod_name"
        assert self.name == "inner_mod_name"
        # 断言输入元组的第一个列表的第二个元素为 "outer_mod_name"
        assert input[0][1] == "outer_mod_name"
        # 返回修改后的输入列表和字符串
        return ["pre_hook_override_name"], "pre_hook_override"

    # 定义一个方法 `forward_hook`，用于前向传播钩子函数
    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        # 断言当前对象的名称为 "inner_mod_name"
        assert self.name == "inner_mod_name"
        # 断言输入元组的第一个列表的第一个元素为 "pre_hook_override_name"
        assert input[0][0] == "pre_hook_override_name"
        # 对输出元组的第二个元素进行操作，添加后缀 "fh"
        output2 = output[1] + "fh"
        # 返回原输出的第一个元素和修改后的输出的第二个元素
        return output[0], output2

    # 注册 `pre_hook` 方法为 `m.submodule` 的前处理钩子
    m.submodule.register_forward_pre_hook(pre_hook)
    # 注册 `forward_hook` 方法为 `m.submodule` 的前向传播钩子
    m.submodule.register_forward_hook(forward_hook)

    # 返回修改后的模型 `m`
    return m
# 创建一个测试函数，用于测试子模块能够同时运行多个前向输入的钩子函数
def create_submodule_multiple_hooks_multiple_inputs():
    # 创建一个 ModuleForwardMultipleInputs 实例，用于测试多个钩子函数
    m = ModuleForwardMultipleInputs("outer_mod_name", "inner_mod_name")

    # 定义第一个前向预钩子函数 pre_hook1
    def pre_hook1(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        # 断言子模块的名称为 "inner_mod_name"
        assert self.name == "inner_mod_name"
        # 断言输入的第二个元素为 "no_pre_hook"
        assert input[1] == "no_pre_hook"
        # 返回修改后的输入
        return ["pre_hook_override_name"], "pre_hook_override1"

    # 定义第二个前向预钩子函数 pre_hook2
    def pre_hook2(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        # 断言子模块的名称为 "inner_mod_name"
        assert self.name == "inner_mod_name"
        # 断言输入的第二个元素为 "pre_hook_override1"
        assert input[1] == "pre_hook_override1"
        # 返回修改后的输入
        return ["pre_hook_override_name"], "pre_hook_override2"

    # 定义第一个前向钩子函数 forward_hook1
    def forward_hook1(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str]
    ):
        # 断言子模块的名称为 "inner_mod_name"
        assert self.name == "inner_mod_name"
        # 断言输入的第二个元素为 "pre_hook_override2"
        assert input[1] == "pre_hook_override2"
        # 断言输出的第二个元素为 "pre_hook_override2_"
        assert output[1] == "pre_hook_override2_"
        # 修改输出并返回
        output2 = output[1] + "fh1"
        return output[0], output2, output2

    # 定义第二个前向钩子函数 forward_hook2
    def forward_hook2(
        self, input: Tuple[List[str], str], output: Tuple[List[str], str, str]
    ):
        # 断言子模块的名称为 "inner_mod_name"
        assert self.name == "inner_mod_name"
        # 断言输入的第二个元素为 "pre_hook_override2"
        assert input[1] == "pre_hook_override2"
        # 断言输出的第二个元素为 "pre_hook_override2_fh1"
        assert output[1] == "pre_hook_override2_fh1"
        # 修改输出并返回
        output2 = output[1] + "_fh2"
        return output[0], output2

    # 分别注册前向预钩子函数和前向钩子函数到子模块
    m.submodule.register_forward_pre_hook(pre_hook1)
    m.submodule.register_forward_pre_hook(pre_hook2)
    m.submodule.register_forward_hook(forward_hook1)
    m.submodule.register_forward_hook(forward_hook2)

    # 返回创建好的子模块实例
    return m


# 创建一个测试函数，用于测试子模块能够运行带有单一输入参数的钩子函数
def create_submodule_forward_single_input():
    # 创建一个 ModuleForwardSingleInput 实例，用于测试单一输入参数的钩子函数
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    # 定义前向预钩子函数 pre_hook
    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        # 断言子模块的名称为 "inner_mod_name"
        assert self.name == "inner_mod_name"
        # 断言输入参数为 ("a_outermod",)
        assert input[0] == "a_outermod"
        # 返回修改后的输入参数
        return ("pre_hook_override_name",)

    # 定义前向钩子函数 forward_hook
    def forward_hook(self, input: Tuple[str], output: str):
        # 断言子模块的名称为 "inner_mod_name"
        assert self.name == "inner_mod_name"
        # 断言输入参数为 ("pre_hook_override_name",)
        assert input == ("pre_hook_override_name",)
        # 返回修改后的输出
        return output

    # 注册前向预钩子函数和前向钩子函数到子模块
    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    # 返回创建好的子模块实例
    return m


# 创建一个测试函数，用于测试直接调用子模块时其钩子函数会被调用
def create_submodule_to_call_directly_with_hooks():
    # 创建一个 ModuleForwardSingleInput 实例，用于测试直接调用时的钩子函数
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    # 定义前向预钩子函数 pre_hook
    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        # 断言子模块的名称为 "inner_mod_name"
        assert self.name == "inner_mod_name"
        # 返回修改后的输入参数
        return ("pre_hook_override_name",)

    # 定义前向钩子函数 forward_hook
    def forward_hook(self, input: Tuple[str], output: str):
        # 断言子模块的名称为 "inner_mod_name"
        assert self.name == "inner_mod_name"
        # 断言输入参数为 ("pre_hook_override_name",)
        assert input == ("pre_hook_override_name",)
        # 修改输出并返回
        return output + "_fh"

    # 注册前向预钩子函数和前向钩子函数到子模块
    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    # 返回创建好的子模块实例
    return m


# 创建一个测试函数，用于测试重复使用相同钩子函数的情况
    # 创建一个 ModuleForwardSingleInput 类的实例，用于测试子模块是否可以多次运行相同的钩子
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    # 定义一个预处理钩子函数 pre_hook，接受一个输入元组并返回一个修改后的元组
    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        # 断言当前实例的名称为 "inner_mod_name"
        assert self.name == "inner_mod_name"
        # 修改输入的第一个元素并返回
        changed = input[0] + "_ph"
        return (changed,)

    # 定义一个前向钩子函数 forward_hook，接受输入元组和输出字符串，并返回修改后的输出字符串
    def forward_hook(self, input: Tuple[str], output: str):
        # 断言当前实例的名称为 "inner_mod_name"
        assert self.name == "inner_mod_name"
        # 断言输入元组为 ("a_outermod_ph_ph",)
        assert input == ("a_outermod_ph_ph",)
        # 在输出字符串末尾添加 "_fh" 后返回
        return output + "_fh"

    # 向 m.submodule 注册 pre_hook 钩子函数，可以多次注册
    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_pre_hook(pre_hook)
    # 向 m.submodule 注册 forward_hook 钩子函数，可以多次注册
    m.submodule.register_forward_hook(forward_hook)
    m.submodule.register_forward_hook(forward_hook)

    # 返回创建的 ModuleForwardSingleInput 实例 m
    return m
def create_submodule_hook_return_nothing():
    # 用于测试子模块是否能够运行返回空值的钩子函数
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook(self, input: Tuple[str]) -> None:
        assert self.name == "inner_mod_name"
        assert input[0] == "a_outermod"

    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("a_outermod",)

    # 注册前向预钩子和前向钩子函数到子模块
    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)

    return m


def create_submodule_multiple_hooks_single_input():
    # 用于测试子模块是否能够运行多个只有一个输入的钩子函数
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    def pre_hook1(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "inner_mod_name"
        assert input[0] == "a_outermod"
        return ("pre_hook_override_name",)

    def pre_hook2(self, input: Tuple[str]) -> Tuple[str]:
        assert self.name == "inner_mod_name"
        assert input[0] == "pre_hook_override_name"
        return ("pre_hook_override_name2",)

    def forward_hook1(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("pre_hook_override_name2",)
        assert output == "pre_hook_override_name2_inner_mod"
        return output + "_fwh1"

    def forward_hook2(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("pre_hook_override_name2",)
        assert output == "pre_hook_override_name2_inner_mod_fwh1"
        return output

    # 注册两个前向预钩子和两个前向钩子函数到子模块
    m.submodule.register_forward_pre_hook(pre_hook1)
    m.submodule.register_forward_pre_hook(pre_hook2)
    m.submodule.register_forward_hook(forward_hook1)
    m.submodule.register_forward_hook(forward_hook2)

    return m


def create_forward_tuple_input():
    # 用于测试前向传播只传入单个元组的情况
    # 这种情况与其它情况不同，因为急切模式总是将前向预钩子的返回结果封装成元组
    # 当前向预钩子的返回结果不是元组时，急切模式不会再封装一层元组，这会导致不一致的行为
    # 为了保持单个元组输入和其它前向输入之间的一致行为，前向预钩子需要将单个元组输入的返回结果再封装一层元组
    # 这是模式检查器所强制执行的规则。
    m = ModuleForwardTupleInput("outer_mod_name", "inner_mod_name")

    def pre_hook_outermod(self, input: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
        # 在急切模式下，'return (11,)' 会丢失内部的元组
        return ((11,),)

    def pre_hook_innermod(self, input: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
        # 在急切模式下，'return (22,)' 会丢失内部的元组
        return ((22,),)

    # 注意：这里缺少了后续的代码块，需要继续注释。
    # 定义一个在外部模块中注册的前向钩子函数，接受一个元组作为输入，返回一个包含整数 11 的元组
    def forward_hook_outermod(self, input: Tuple[Tuple[int]], output: int):
        return (11,)

    # 定义一个在内部模块中注册的前向钩子函数，接受一个元组作为输入，返回整数 22
    def forward_hook_innermod(self, input: Tuple[Tuple[int]], output: Tuple[int]):
        return 22

    # 将外部模块的前向钩子函数注册到模型 m 上
    m.register_forward_pre_hook(pre_hook_outermod)
    # 将内部模块的前向钩子函数注册到模型 m 的子模块 submodule 上
    m.submodule.register_forward_pre_hook(pre_hook_innermod)
    # 将外部模块的前向钩子函数注册到模型 m 上
    m.register_forward_hook(forward_hook_outermod)
    # 将内部模块的前向钩子函数注册到模型 m 的子模块 submodule 上
    m.submodule.register_forward_hook(forward_hook_innermod)

    # 返回注册了钩子函数的模型 m
    return m
# 创建一个函数，用于生成一个子模块对象，并设置特定的钩子函数以修改输入和输出行为
def create_submodule_forward_single_input_return_not_tupled():
    # 用于检查子模块是否能够返回修改后的输入，而不是包装成元组（以匹配急切模式的行为）
    m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

    # 定义一个前置钩子函数，修改输入参数，并返回修改后的名称
    def pre_hook(self, input: Tuple[str]) -> str:
        assert self.name == "inner_mod_name"
        assert input[0] == "a_outermod"
        # 在其他测试用例中，返回通常会被包装在元组中
        return "pre_hook_override_name"

    # 定义一个正向钩子函数，修改输入和输出参数，确保输出参数以 "_fh" 结尾
    def forward_hook(self, input: Tuple[str], output: str):
        assert self.name == "inner_mod_name"
        assert input == ("pre_hook_override_name",)
        output = output + "_fh"
        return output

    # 将前置钩子函数注册到子模块中
    m.submodule.register_forward_pre_hook(pre_hook)
    # 将正向钩子函数注册到子模块中
    m.submodule.register_forward_hook(forward_hook)

    # 返回创建的子模块对象
    return m
```