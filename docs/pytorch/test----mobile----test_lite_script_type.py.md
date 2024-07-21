# `.\pytorch\test\mobile\test_lite_script_type.py`

```
# Owner(s): ["oncall: mobile"]

# 导入所需的模块
import io
import unittest
from collections import namedtuple
from typing import Dict, List, NamedTuple

# 导入 PyTorch 相关模块
import torch
import torch.utils.bundled_inputs

# 导入 TorchScript 相关模块
from torch.jit.mobile import _load_for_lite_interpreter
from torch.testing._internal.common_utils import run_tests, TestCase

# 定义测试类 TestLiteScriptModule，继承自 TestCase
class TestLiteScriptModule(TestCase):

    # 定义测试方法 test_typing_namedtuple
    def test_typing_namedtuple(self):
        # 定义一个 NamedTuple 类型 myNamedTuple，包含名为 'a' 的 List[torch.Tensor] 字段
        myNamedTuple = NamedTuple(
            "myNamedTuple", [("a", List[torch.Tensor])]
        )

        # 定义一个简单的 Torch 模块 MyTestModule，包含一个 forward 方法
        class MyTestModule(torch.nn.Module):
            def forward(self, a: torch.Tensor):
                # 创建 myNamedTuple 实例 p，将输入张量 a 放入列表中
                p = myNamedTuple([a])
                return p

        # 创建一个样本输入张量
        sample_input = torch.tensor(5)

        # 对 MyTestModule 进行 TorchScript 脚本化
        script_module = torch.jit.script(MyTestModule())

        # 使用样本输入张量对脚本化模块进行前向传播，并获取结果的 'a' 属性
        script_module_result = script_module(sample_input).a

        # 将脚本化模块保存到字节流中，用于 Lite 解释器
        buffer = io.BytesIO(
            script_module._save_to_buffer_for_lite_interpreter(
                _save_mobile_debug_info=True
            )
        )
        buffer.seek(0)

        # 从字节流中加载 Lite 解释器可用的移动模块
        mobile_module = _load_for_lite_interpreter(buffer)  # Error here

        # 使用样本输入张量对 Lite 模块进行前向传播，并获取结果的 'a' 属性
        mobile_module_result = mobile_module(sample_input).a

        # 断言脚本化模块结果与 Lite 模块结果的近似性
        torch.testing.assert_close(script_module_result, mobile_module_result)

    # 跳过测试方法，注释指定为 "T137512434"
    @unittest.skip("T137512434")
    # 定义一个测试类，用于测试带有命名元组的字典类型
    def test_typing_dict_with_namedtuple(self):
        # 定义命名元组 Foo，包含一个名为 id 的 torch.Tensor
        class Foo(NamedTuple):
            id: torch.Tensor

        # 定义一个继承自 torch.nn.Module 的类 Bar
        class Bar(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个 Foo 实例，id 为 torch.tensor(1)，赋值给 self.foo
                self.foo = Foo(torch.tensor(1))

            # 前向传播方法
            def forward(self, a: torch.Tensor):
                # 将 self.foo 更新为一个新的 Foo 实例，id 为 a
                self.foo = Foo(a)
                # 创建一个空的字典 re
                re: Dict[str, Foo] = {}
                # 向字典 re 中添加一个键值对，键为 "test"，值为 Foo(a)
                re["test"] = Foo(a)
                # 返回当前的 self.foo 和字典 re 中键 "test" 对应的值
                return self.foo, re["test"]

        # 创建一个 torch.tensor 对象，作为输入数据的示例
        sample_input = torch.tensor(5)
        
        # 使用 torch.jit.script 方法将 Bar 类实例化为一个脚本模块
        script_module = torch.jit.script(Bar())

        # 使用示例输入数据调用脚本模块，获取结果
        script_module_result = script_module(sample_input)

        # 将脚本模块保存为字节流 buffer_mobile
        buffer_mobile = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer_mobile.seek(0)
        
        # 从 buffer_mobile 加载移动端模块
        mobile_module = _load_for_lite_interpreter(buffer_mobile)
        
        # 使用示例输入数据调用移动端模块，获取结果
        mobile_module_result = mobile_module(sample_input)
        
        # 使用 torch.testing.assert_close 方法断言脚本模块和移动端模块的结果近似相等
        torch.testing.assert_close(script_module_result, mobile_module_result)
    def test_typing_namedtuple_custom_classtype(self):
        # 定义名为 Foo 的命名元组，包含一个名为 id 的 torch.Tensor 类型字段
        class Foo(NamedTuple):
            id: torch.Tensor

        # 定义名为 Bar 的 PyTorch 模块
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化 Bar 模块中的 foo 属性为一个 Foo 对象，其中 id 初始化为 torch.tensor(1)
                self.foo = Foo(torch.tensor(1))

            # 前向传播函数，接收一个 torch.Tensor 类型的输入 a，更新 self.foo 并返回其值
            def forward(self, a: torch.Tensor):
                self.foo = Foo(a)
                return self.foo

        # 创建一个样本输入 torch.tensor(5)
        sample_input = torch.tensor(5)
        # 将 Bar 模块转换为 Torch 脚本
        script_module = torch.jit.script(Bar())
        # 使用样本输入运行 Torch 脚本模块，得到输出结果
        script_module_result = script_module(sample_input)

        # 将 Torch 脚本模块保存为移动端模块的字节流
        buffer_mobile = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer_mobile.seek(0)
        # 从字节流中加载移动端模块
        mobile_module = _load_for_lite_interpreter(buffer_mobile)
        # 使用样本输入运行移动端模块，得到输出结果
        mobile_module_result = mobile_module(sample_input)
        # 断言 Torch 脚本模块和移动端模块的输出结果在数值上接近
        torch.testing.assert_close(script_module_result, mobile_module_result)

    def test_return_collections_namedtuple(self):
        # 定义名为 myNamedTuple 的命名元组，包含一个字段 "a"
        myNamedTuple = namedtuple("myNamedTuple", [("a")])

        # 定义名为 MyTestModule 的 PyTorch 模块
        class MyTestModule(torch.nn.Module):
            # 前向传播函数，接收一个 torch.Tensor 类型的输入 a，返回一个 myNamedTuple 对象
            def forward(self, a: torch.Tensor):
                return myNamedTuple(a)

        # 创建一个样本输入 torch.Tensor(1)
        sample_input = torch.Tensor(1)
        # 将 MyTestModule 模块转换为 Torch 脚本
        script_module = torch.jit.script(MyTestModule())
        # 使用样本输入运行 Torch 脚本模块，得到输出结果
        script_module_result = script_module(sample_input)
        # 将 Torch 脚本模块保存为移动端模块的字节流
        buffer_mobile = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer_mobile.seek(0)
        # 从字节流中加载移动端模块
        mobile_module = _load_for_lite_interpreter(buffer_mobile)
        # 使用样本输入运行移动端模块，得到输出结果
        mobile_module_result = mobile_module(sample_input)
        # 断言 Torch 脚本模块和移动端模块的输出结果在数值上接近
        torch.testing.assert_close(script_module_result, mobile_module_result)

    def test_nest_typing_namedtuple_custom_classtype(self):
        # 定义名为 Baz 的命名元组，包含一个名为 di 的 torch.Tensor 类型字段
        class Baz(NamedTuple):
            di: torch.Tensor

        # 定义名为 Foo 的命名元组，包含一个名为 id 的 torch.Tensor 类型字段和一个名为 baz 的 Baz 对象字段
        class Foo(NamedTuple):
            id: torch.Tensor
            baz: Baz

        # 定义名为 Bar 的 PyTorch 模块
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化 Bar 模块中的 foo 属性为一个 Foo 对象，其中 id 初始化为 torch.tensor(1)，baz 初始化为 Baz(torch.tensor(1))
                self.foo = Foo(torch.tensor(1), Baz(torch.tensor(1)))

            # 前向传播函数，接收一个 torch.Tensor 类型的输入 a，更新 self.foo 并返回其值
            def forward(self, a: torch.Tensor):
                self.foo = Foo(a, Baz(torch.tensor(1)))
                return self.foo

        # 创建一个样本输入 torch.tensor(5)
        sample_input = torch.tensor(5)
        # 将 Bar 模块转换为 Torch 脚本
        script_module = torch.jit.script(Bar())
        # 使用样本输入运行 Torch 脚本模块，得到输出结果
        script_module_result = script_module(sample_input)

        # 将 Torch 脚本模块保存为移动端模块的字节流
        buffer_mobile = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer_mobile.seek(0)
        # 从字节流中加载移动端模块
        mobile_module = _load_for_lite_interpreter(buffer_mobile)
        # 使用样本输入运行移动端模块，得到输出结果
        mobile_module_result = mobile_module(sample_input)
        # 断言 Torch 脚本模块和移动端模块的输出结果中的 baz.di 字段在数值上接近
        torch.testing.assert_close(
            script_module_result.baz.di, mobile_module_result.baz.di
        )
# 如果当前脚本作为主程序执行（而不是被导入到其他脚本中），则执行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```