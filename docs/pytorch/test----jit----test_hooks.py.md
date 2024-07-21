# `.\pytorch\test\jit\test_hooks.py`

```
# Owner(s): ["oncall: jit"]

# 导入必要的模块和类
import os
import sys
import unittest
from typing import Tuple

import torch

# 导入测试所需的函数和类
from jit.test_hooks_modules import (
    create_forward_tuple_input,
    create_module_forward_multiple_inputs,
    create_module_forward_single_input,
    create_module_hook_return_nothing,
    create_module_multiple_hooks_multiple_inputs,
    create_module_multiple_hooks_single_input,
    create_module_no_forward_input,
    create_module_same_hook_repeated,
    create_submodule_forward_multiple_inputs,
    create_submodule_forward_single_input,
    create_submodule_forward_single_input_return_not_tupled,
    create_submodule_hook_return_nothing,
    create_submodule_multiple_hooks_multiple_inputs,
    create_submodule_multiple_hooks_single_input,
    create_submodule_no_forward_input,
    create_submodule_same_hook_repeated,
    create_submodule_to_call_directly_with_hooks,
    ModuleDirectforwardSubmodCall,
    ModuleForwardSingleInput,
    ModuleForwardTupleInput,
)

# 将 test/ 目录添加到系统路径，使得其中的辅助文件可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

# 如果该文件被直接运行，抛出错误提示
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 测试 JIT forward hooks 和 pre-hooks 的类
class TestHooks(JitTestCase):
    # 测试模块没有 forward 输入
    def test_module_no_forward_input(self):
        self.checkModule(create_module_no_forward_input(), ())

    # 测试子模块没有 forward 输入
    def test_submodule_no_forward_input(self):
        self.checkModule(create_submodule_no_forward_input(), ())

    # 测试模块多输入的 forward
    def test_module_forward_multiple_inputs(self):
        self.checkModule(
            create_module_forward_multiple_inputs(), (["a"], "no_pre_hook")
        )

    # 测试模块多 hooks 多输入
    def test_module_multiple_hooks_multiple_inputs(self):
        self.checkModule(
            create_module_multiple_hooks_multiple_inputs(), (["a"], "no_pre_hook")
        )

    # 测试模块单输入的 forward
    def test_module_forward_single_input(self):
        self.checkModule(create_module_forward_single_input(), ("a",))

    # 测试模块重复使用相同 hook
    def test_module_same_hook_repeated(self):
        self.checkModule(create_module_same_hook_repeated(), ("a",))

    # 测试模块 hook 返回空
    def test_module_hook_return_nothing(self):
        self.checkModule(create_module_hook_return_nothing(), ("a",))

    # 测试模块多 hooks 单输入
    def test_module_multiple_hooks_single_input(self):
        self.checkModule(create_module_multiple_hooks_single_input(), ("a",))

    # 测试子模块多输入的 forward
    def test_submodule_forward_multiple_inputs(self):
        self.checkModule(
            create_submodule_forward_multiple_inputs(), (["a"], "no_pre_hook")
        )

    # 测试子模块多 hooks 多输入
    def test_submodule_multiple_hooks_multiple_inputs(self):
        self.checkModule(
            create_submodule_multiple_hooks_multiple_inputs(),
            (["a"], "no_pre_hook"),
        )
    def test_submodule_forward_single_input(self):
        # 调用测试方法，检查单输入情况下子模块的前向传播
        self.checkModule(create_submodule_forward_single_input(), ("a",))

    def test_submodule_called_directly_with_hooks(self):
        # 创建一个可以直接调用带钩子的子模块，并进行脚本化
        module = create_submodule_to_call_directly_with_hooks()
        module_scripted = torch.jit.script(module)

        # 获取子模块及其脚本化版本
        submodule = module.submodule
        scripted_submodule = module_scripted.submodule

        # 断言直接调用子模块和脚本化子模块的结果一致
        self.assertEqual(submodule("a"), scripted_submodule("a"))

    def test_submodule_same_hook_repeated(self):
        # 检查使用相同钩子重复的子模块
        self.checkModule(create_submodule_same_hook_repeated(), ("a",))

    def test_submodule_hook_return_nothing(self):
        # 检查钩子返回空值的子模块
        self.checkModule(create_submodule_hook_return_nothing(), ("a",))

    def test_submodule_multiple_hooks_single_input(self):
        # 检查单输入情况下使用多个钩子的子模块
        self.checkModule(create_submodule_multiple_hooks_single_input(), (["a"]))

    def test_forward_tuple_input(self):
        # 检查接受元组输入的前向传播方法
        self.checkModule(create_forward_tuple_input(), ((3,),))

    def test_submodule_forward_single_input_return_not_tupled(self):
        # 检查前向传播返回非元组结果的单输入子模块
        self.checkModule(
            create_submodule_forward_single_input_return_not_tupled(), ("a",)
        )

    def test_hook_method_name_collision(self):
        # 测试钩子方法名与已有方法名冲突的情况
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        def foo(self, input: Tuple[str]) -> Tuple[str]:
            assert self.name == "inner_mod_name"
            assert input[0] == "a_outermod"
            return ("pre_hook_override_name",)

        # 注册前向钩子并断言抛出异常
        m.submodule.register_forward_pre_hook(foo)

        with self.assertRaisesRegex(
            RuntimeError,
            "Can't define hook: foo on class: .+ "
            "because a method or hook with that name already exists.",
        ):
            torch.jit.script(m)
    def test_hook_hook_name_collision(self):
        # 测试两个具有相同名称但不同Python定义的钩子的边缘情况
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        # 定义第一个预钩子函数
        def prehook(self, input: Tuple[str]) -> Tuple[str]:
            return "This is the first hook"

        # 注册第一个预钩子函数到子模块
        m.submodule.register_forward_pre_hook(prehook)

        # 定义第二个预钩子函数
        def prehook(self, input: Tuple[str]) -> Tuple[str]:
            return "This is the second hook"

        # 注册第二个预钩子函数到子模块
        m.submodule.register_forward_pre_hook(prehook)

        # 断言脚本化时引发运行时错误，因为至少有两个不同的Python定义的预钩子函数
        with self.assertRaisesRegex(
            RuntimeError,
            "Pre-hook '.+' on .+ has at least two different python "
            "definitions. Please use unique names for all hooks.",
        ):
            torch.jit.script(m)

        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        # 定义第一个正向钩子函数
        def hook(self, input: Tuple[str], output: str):
            return "This is the first hook"

        # 注册第一个正向钩子函数到子模块
        m.submodule.register_forward_hook(hook)

        # 定义第二个正向钩子函数
        def hook(self, input: Tuple[str]):
            return "This is the second hook"

        # 注册第二个正向钩子函数到子模块
        m.submodule.register_forward_hook(hook)

        # 断言脚本化时引发运行时错误，因为至少有两个不同的Python定义的正向钩子函数
        with self.assertRaisesRegex(
            RuntimeError,
            "Hook '.+' on .+ has at least two different python "
            "definitions. Please use unique names for all hooks.",
        ):
            torch.jit.script(m)

    def test_module_direct_forward_invocation(self):
        # 测试只有在直接调用模块时才会触发钩子，而不是在调用forward时
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        # 定义预钩子函数
        def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
            return ("pre_hook_override_name",)

        # 定义正向钩子函数
        def forward_hook(self, input: Tuple[str], output: str):
            assert self.name == "outer_mod_name"
            assert input == ("pre_hook_override_name",)
            output = output + "_fh"
            return output

        # 注册预钩子函数
        m.register_forward_pre_hook(pre_hook)
        # 注册正向钩子函数
        m.register_forward_hook(forward_hook)

        # 对模块进行脚本化
        m_scripted = torch.jit.script(m)

        # 断言直接调用时前向方法的结果相等
        self.assertEqual(m.forward("a"), m_scripted.forward("a"))
        # 断言脚本化模块的调用方法与forward方法不同
        self.assertNotEqual(m_scripted("a"), m_scripted.forward("a"))
    # 定义测试函数，测试子模块直接调用前向传递
    def test_submodule_direct_forward_invocation(self):
        # 创建 ModuleDirectforwardSubmodCall 实例，传入外部和内部模块名称
        m_submod_forward_call = ModuleDirectforwardSubmodCall(
            "outer_mod_name", "inner_mod_name"
        )
        # 创建 ModuleForwardSingleInput 实例，传入外部和内部模块名称
        m_submod_call = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        # 定义预处理钩子函数，修改输入元组的第一个元素
        def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
            return ("pre_hook_override_name",)

        # 定义前向传递钩子函数，验证输入是否符合预期，并在输出后缀添加 "_fh"
        def forward_hook(self, input: Tuple[str], output: str):
            assert input == ("pre_hook_override_name",)
            return output + "_fh"

        # 注册预处理钩子到 m_submod_forward_call 的子模块
        m_submod_forward_call.submodule.register_forward_pre_hook(pre_hook)
        # 注册前向传递钩子到 m_submod_forward_call 的子模块
        m_submod_forward_call.submodule.register_forward_hook(forward_hook)
        # 注册预处理钩子到 m_submod_call 的子模块
        m_submod_call.submodule.register_forward_pre_hook(pre_hook)
        # 注册前向传递钩子到 m_submod_call 的子模块
        m_submod_call.submodule.register_forward_hook(forward_hook)

        # 对 m_submod_forward_call 进行脚本化
        m_submod_forward_call_scripted = torch.jit.script(m_submod_forward_call)
        # 对 m_submod_call 进行脚本化
        m_submod_call_scripted = torch.jit.script(m_submod_call)

        # 断言脚本化后的 m_submod_forward_call 和原始调用结果相等
        self.assertEqual(
            m_submod_forward_call_scripted("a"), m_submod_forward_call("a")
        )
        # 断言脚本化后的 m_submod_forward_call 和 m_submod_call 结果不相等
        self.assertNotEqual(
            m_submod_forward_call_scripted("a"), m_submod_call_scripted("a")
        )

    # TODO: add this test back once figured out how to print error msg
    # 定义测试钩子编译提示函数，用于测试在架构检查后是否打印出钩子错误信息
    @unittest.skip
    def test_hook_compilation_hint(self):
        # 创建 ModuleForwardSingleInput 实例，传入外部和内部模块名称
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")

        # 定义预处理钩子函数，断言模块名称为 "outer_mod_name"，并故意访问元组超出范围的元素
        def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
            assert self.name == "outer_mod_name"
            assert input[4] == "a"  # out of bounds tuple range
            return ("pre_hook_override_name",)

        # 注册预处理钩子到模块 m
        m.register_forward_pre_hook(pre_hook)

        # 使用 assertRaisesRegex 断言在脚本化预处理钩子 'pre_hook' 时会引发 RuntimeError，并且错误消息包含特定文本
        with self.assertRaisesRegex(
            RuntimeError,
            "This error occurred while scripting the forward pre-hook 'pre_hook'",
        ):
            torch.jit.script(m)
    def test_wrong_pre_hook_signatures(self):
        # 测试不正确的前钩子函数签名
        # 正确的签名应为：pre_hook_c(self, input: Tuple[str])

        # 第一个错误的输入参数类型的前钩子函数
        def pre_hook_wrong_input1(self, input: Tuple[None]) -> Tuple[str]:
            return ("hello",)

        # 创建 ModuleForwardSingleInput 实例
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        # 注册前钩子函数
        m.register_forward_pre_hook(pre_hook_wrong_input1)

        # 断言捕获 RuntimeError 异常，并验证错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong inner types for the input tuple argument",
        ):
            torch.jit.script(m)

        # 第二个错误的输入参数数量的前钩子函数
        def pre_hook_wrong_input2(self, input: Tuple[str], input2: str) -> Tuple[str]:
            return ("hello",)

        # 创建 ModuleForwardSingleInput 实例
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        # 注册前钩子函数
        m.register_forward_pre_hook(pre_hook_wrong_input2)

        # 断言捕获 RuntimeError 异常，并验证错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            "was expected to only have exactly 2 inputs but it had 3 inputs",
        ):
            torch.jit.script(m)

        # 第三个错误的输入参数类型的前钩子函数
        def pre_hook_wrong_input3(self, input: int) -> Tuple[str]:
            return ("hello",)

        # 创建 ModuleForwardSingleInput 实例
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        # 注册前钩子函数
        m.register_forward_pre_hook(pre_hook_wrong_input3)

        # 断言捕获 RuntimeError 异常，并验证错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            "expected the input argument to be typed as a Tuple but found type: 'int' instead",
        ):
            torch.jit.script(m)

        # 错误的输出类型的前钩子函数
        def pre_hook_wrong_output(self, input: Tuple[str]) -> int:
            return 1  # 期望返回 Tuple[str], str 或 None

        # 创建 ModuleForwardSingleInput 实例
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        # 注册前钩子函数
        m.register_forward_pre_hook(pre_hook_wrong_output)

        # 断言捕获 RuntimeError 异常，并验证错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            "returned the wrong type of: 'int'",
        ):
            torch.jit.script(m)

        # 缺少输出注释的前钩子函数
        def pre_hook_no_output_annotation(self, input: Tuple[str]):
            return 1  # 期望返回 Tuple[str], str 或 None

        # 创建 ModuleForwardSingleInput 实例
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        # 注册前钩子函数
        m.register_forward_pre_hook(pre_hook_no_output_annotation)

        # 断言捕获 RuntimeError 异常，并验证错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            "is missing a return annotation. Return annotations are required, please add one.",
        ):
            torch.jit.script(m)

        # 错误的元组返回类型的前钩子函数
        def pre_hook_wrong_tuple_return(self, input: Tuple[Tuple[int]]) -> Tuple[int]:
            return (11,)  # 在 eager 模式下无法工作，内部元组丢失

        # 创建 ModuleForwardTupleInput 实例
        m = ModuleForwardTupleInput("outer_mod_name", "inner_mod_name")
        # 注册前钩子函数
        m.register_forward_pre_hook(pre_hook_wrong_tuple_return)

        # 断言捕获 RuntimeError 异常，并验证错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            "When forward has a single tuple input argument, the return needs to be 'None' or a nested tuple containing forward's input tuple argument as in: 'Tuple\[Tuple\[int\]\]'",
        ):
            torch.jit.script(m)
    # 测试错误的钩子函数签名
    def test_wrong_hook_signatures(self):
        # 正确的签名:
        #   def forward_hook(self, input: Tuple[str], output: str)
        # 钩子函数参数中输入参数类型错误
        def forward_hook_wrong_input1(self, input: Tuple[str, str], output: str):
            return output

        # 创建 ModuleForwardSingleInput 实例
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        # 注册错误输入参数类型的钩子函数
        m.register_forward_hook(forward_hook_wrong_input1)

        # 断言捕获 RuntimeError 异常，验证错误的输入参数类型
        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong number of contained types for the "
            r"input argument's Tuple. Received type: 'Tuple\[str, str\]'",
        ):
            torch.jit.script(m)

        # 钩子函数参数中输入参数类型错误
        def forward_hook_wrong_input2(self, input: str, output: str):
            return output

        # 创建 ModuleForwardSingleInput 实例
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        # 注册错误输入参数类型的钩子函数
        m.register_forward_hook(forward_hook_wrong_input2)

        # 断言捕获 RuntimeError 异常，验证错误的输入参数类型
        with self.assertRaisesRegex(
            RuntimeError,
            "expected the input argument to be typed as a Tuple "
            "but found type: 'str' instead.",
        ):
            torch.jit.script(m)

        # 钩子函数参数中输入参数类型错误
        def forward_hook_wrong_input3(self, input: Tuple[None], output: str):
            return output

        # 创建 ModuleForwardSingleInput 实例
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        # 注册错误输入参数类型的钩子函数
        m.register_forward_hook(forward_hook_wrong_input3)

        # 断言捕获 RuntimeError 异常，验证错误的输入参数类型
        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong inner types for the input tuple"
            r" argument. Received type: 'Tuple\[NoneType\]'",
        ):
            torch.jit.script(m)

        # 钩子函数参数中输出参数类型错误
        def forward_hook_wrong_output(self, input: Tuple[str], output: Tuple[str]):
            return output

        # 创建 ModuleForwardSingleInput 实例
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        # 注册错误输出参数类型的钩子函数
        m.register_forward_hook(forward_hook_wrong_output)

        # 断言捕获 RuntimeError 异常，验证错误的输出参数类型
        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong type for the output argument. Received"
            r" type: 'Tuple\[str\]'. Expected type: 'str'",
        ):
            torch.jit.script(m)

        # 钩子函数参数中输出参数类型正确
        def forward_hook_correct(self, input: Tuple[str], output: str):
            return (output,)

        # 钩子函数中前一个钩子函数的输出参数类型错误
        def forward_hook_wrong_output_from_prev_hook(
            self, input: Tuple[str], output: str
        ):
            return output

        # 创建 ModuleForwardSingleInput 实例
        m = ModuleForwardSingleInput("outer_mod_name", "inner_mod_name")
        # 注册正确输出参数类型的钩子函数
        m.register_forward_hook(forward_hook_correct)
        # 注册错误输出参数类型的钩子函数
        m.register_forward_hook(forward_hook_wrong_output_from_prev_hook)

        # 断言捕获 RuntimeError 异常，验证错误的输出参数类型
        with self.assertRaisesRegex(
            RuntimeError,
            "has the wrong type for the output argument. "
            r"Received type: 'str'. Expected type: 'Tuple\[str\]'",
        ):
            torch.jit.script(m)
```