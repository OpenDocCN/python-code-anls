# `.\pytorch\test\jit\test_pdt.py`

```py
# Owner(s): ["oncall: jit"]

# 引入必要的库和模块
import os
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple  # noqa: F401

# 引入 Torch 库
import torch

# 引入需要的测试相关工具和类
from torch.jit._monkeytype_config import _IS_MONKEYTYPE_INSTALLED
from torch.testing._internal.common_utils import NoTest
from torch.testing._internal.jit_utils import JitTestCase, make_global

# 将 test/ 目录下的辅助文件加入 Python 搜索路径
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 如果没有安装 monkeytype，打印警告信息并将 JitTestCase 替换为 NoTest 类型
if not _IS_MONKEYTYPE_INSTALLED:
    print(
        "monkeytype is not installed. Skipping tests for Profile-Directed Typing",
        file=sys.stderr,
    )
    JitTestCase = NoTest  # type: ignore[misc, assignment] # noqa: F811

# 如果当前脚本作为主程序运行，抛出运行时错误，提示正确的运行方法
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义一个测试类 TestPDT，继承自 JitTestCase
class TestPDT(JitTestCase):
    """
    A suite of tests for profile directed typing in TorchScript.
    """

    # 测试 TorchScript 中的 nn.Module
    def test_nn_module(self):
        # 定义一个测试用的 TorchScript 模型 TestPDTModel
        class TestPDTModel(torch.nn.Module):
            # 实现模型的 forward 方法
            def forward(self, x) -> Any:
                # 如果输入 x 是 int 类型，返回 x + 1
                if isinstance(x, int):
                    return x + 1
                # 如果输入 x 是 float 类型，返回 x - 1
                elif isinstance(x, float):
                    return x - 1
                # 其他情况直接返回 x
                else:
                    return x

        # 将 TestPDTModel 模型注册为全局模型
        make_global(TestPDTModel)
        
        # 创建 TestPDTModel 的实例 pdt_model
        pdt_model = TestPDTModel()
        
        # 定义输入 inp 作为 List[Tuple[Any, ...]] 类型
        inp: List[Tuple[Any, ...]] = [
            (20,),    # 包含一个元素为整数 20 的元组
            (2.7,),   # 包含一个元素为浮点数 2.7 的元组
            (False,), # 包含一个元素为布尔值 False 的元组
        ]
        
        # 使用 torch.jit.script 对 pdt_model 进行脚本化，并传入示例输入 inp
        scripted_pdt_model = torch.jit.script(
            pdt_model, example_inputs={pdt_model: inp}
        )
        
        # 断言脚本化模型对输入 50 的输出与原模型对输入 50 的输出相等
        self.assertEqual(scripted_pdt_model(50), pdt_model(50))
        
        # 断言脚本化模型对输入 1.8 的输出与原模型对输入 1.8 的输出相等
        self.assertEqual(scripted_pdt_model(1.8), pdt_model(1.8))
        
        # 断言脚本化模型对输入 True 的输出为原模型对输入 True 的输出
        self.assertTrue(scripted_pdt_model(True), pdt_model(True))

    # 测试嵌套的 nn.Module 类
    def test_nested_nn_module_class(self):
        # 定义一个内部的 TorchScript 模型 NestedPDTInner
        class NestedPDTInner(torch.nn.Module):
            # 实现模型的 forward 方法
            def forward(self, x):
                # 如果输入 x 是 int 类型，返回 x * 10
                if isinstance(x, int):
                    return x * 10
                # 其他情况直接返回 x
                return x

        # 定义一个包装器模型 NestedModulePDTWrapper，接收一个 inner 模型
        class NestedModulePDTWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            # 实现模型的 forward 方法，将输入传递给内部模型 inner
            def forward(self, x):
                return self.inner(x)

        # 将 NestedPDTInner 和 NestedModulePDTWrapper 注册为全局模型
        make_global(NestedPDTInner, NestedModulePDTWrapper)
        
        # 创建 NestedPDTInner 的实例 inner_pdt_model
        inner_pdt_model = NestedPDTInner()
        
        # 使用 inner_pdt_model 初始化 NestedModulePDTWrapper 的实例 wrapped_pdt_model
        wrapped_pdt_model = NestedModulePDTWrapper(inner_pdt_model)
        
        # 定义输入 inp 作为 List[Tuple[Any, ...]] 类型
        inp: List[Tuple[Any, ...]] = [(20,), (False,)]
        
        # 使用 torch.jit.script 对 wrapped_pdt_model 进行脚本化，并传入示例输入 inp
        scripted_pdt_model = torch.jit.script(
            wrapped_pdt_model, example_inputs={wrapped_pdt_model: inp}
        )
        
        # 断言脚本化模型对输入 30 的输出与原模型对输入 30 的输出相等
        self.assertEqual(scripted_pdt_model(30), wrapped_pdt_model(30))
        
        # 断言脚本化模型对输入 1.9 的输出与原模型对输入 1.9 的输出相等
        self.assertEqual(scripted_pdt_model(1.9), wrapped_pdt_model(1.9))
        
        # 断言脚本化模型对输入 True 的输出为原模型对输入 True 的输出
        self.assertTrue(scripted_pdt_model(True), wrapped_pdt_model(True))
    def test_nested_nn_module_class_with_args(self):
        # 定义一个嵌套的内部模块类 NestedModulePDTInner
        class NestedModulePDTInner(torch.nn.Module):
            # 内部模块的前向传播函数，根据输入 x 和 y 返回计算结果
            def forward(self, x, y):
                if isinstance(x, int):
                    return x * 10 + y  # 如果 x 是整数，返回 x * 10 + y 的计算结果
                return x  # 否则返回 x 本身

        # 定义一个嵌套的外部模块类 NestedModulePDTOuter
        class NestedModulePDTOuter(torch.nn.Module):
            # 外部模块的初始化函数，接受一个内部模块实例作为参数
            def __init__(self, inner):
                super().__init__()
                self.inner = inner  # 将传入的内部模块实例保存为内部属性

            # 外部模块的前向传播函数，调用内部模块的 forward 方法，传入参数 x 和固定值 20
            def forward(self, x):
                return self.inner(x, 20)

        # 将内部模块类和外部模块类注册到全局作用域中
        make_global(NestedModulePDTInner, NestedModulePDTOuter)
        # 创建内部模块的实例
        inner_pdt_model = NestedModulePDTInner()
        # 创建外部模块的实例，传入内部模块的实例作为参数
        outer_pdt_model = NestedModulePDTOuter(inner_pdt_model)
        # 定义内部模块的输入列表
        inner_input: List[Tuple[Any, ...]] = [
            (10, 10),   # 整数输入示例
            (1.9, 20),  # 浮点数输入示例
        ]
        # 定义外部模块的输入列表
        outer_input: List[Tuple[Any, ...]] = [(20,), (False,)]
        # 对外部模块进行脚本化，指定输入示例
        scripted_pdt_model = torch.jit.script(
            outer_pdt_model,
            example_inputs={
                inner_pdt_model: inner_input,
                outer_pdt_model: outer_input,
            },
        )
        # 断言脚本化模型和原始外部模块在相同输入下的输出结果相等
        self.assertEqual(scripted_pdt_model(30), outer_pdt_model(30))
        self.assertEqual(scripted_pdt_model(1.9), outer_pdt_model(1.9))
        self.assertTrue(scripted_pdt_model(True), outer_pdt_model(True))

    def test_nested_function_in_forward(self):
        # 定义一个包含嵌套函数的模块类 NestedFunctionInForward
        class NestedFunctionInForward(torch.nn.Module):
            # 前向传播函数，调用内部定义的 fun 函数并返回其结果加上 10
            def forward(self, x):
                return self.fun(x) + 10

            # 内部定义的函数 fun，根据输入 x 返回不同的计算结果
            def fun(self, x):
                if isinstance(x, bool):
                    return 0  # 如果 x 是布尔类型，返回 0
                elif isinstance(x, int):
                    return x + 1  # 如果 x 是整数，返回 x + 1
                return 0  # 其他情况返回 0

        # 将模块类注册到全局作用域中
        make_global(NestedFunctionInForward)
        # 创建模块类的实例
        pdt_model = NestedFunctionInForward()
        # 定义输入示例列表
        inp: List[Tuple[Any, ...]] = [(-1,), (False,)]
        # 对模型进行脚本化，指定输入示例
        scripted_pdt_model = torch.jit.script(
            pdt_model, example_inputs={pdt_model: inp}
        )
        # 断言脚本化模型和原始模型在相同输入下的输出结果相等
        self.assertEqual(scripted_pdt_model(30), pdt_model(30))
        self.assertEqual(scripted_pdt_model(True), pdt_model(True))
    def test_nn_module_with_export_function(self):
        # 定义一个带有导出函数的测试模型类
        class TestModelWithExport(torch.nn.Module):
            # 声明一个导出函数 fn，接受参数 x 和 y，返回任意类型
            @torch.jit.export
            def fn(self, x, y) -> Any:
                # 断言 x 和 y 不同时为布尔型
                assert not (isinstance(x, bool) and isinstance(y, bool))
                # 根据 x 和 y 的类型返回不同的计算结果
                if isinstance(x, int) and isinstance(y, int):
                    return x + y
                elif isinstance(x, float) and isinstance(y, float):
                    return x - y
                else:
                    return -1

        # 将 TestModelWithExport 类作为全局对象
        make_global(TestModelWithExport)
        # 创建 TestModelWithExport 类的实例
        pdt_model = TestModelWithExport()
        # 定义输入的列表，包含两个元组，每个元组包含两个数值
        inp: List[Tuple[Any, ...]] = [
            (
                20,
                10,
            ),
            (
                2.7,
                8.9,
            ),
        ]
        # 使用 torch.jit.script 将 pdt_model 进行脚本化，指定输入示例
        scripted_pdt_model = torch.jit.script(
            pdt_model, example_inputs={pdt_model.fn: inp}
        )
        # 断言脚本化模型和原始模型对相同的输入返回相同的结果
        self.assertEqual(scripted_pdt_model.fn(10, 90), pdt_model.fn(10, 90))
        self.assertEqual(scripted_pdt_model.fn(1.8, 2.2), pdt_model.fn(1.8, 2.2))
        # 断言脚本化模型和原始模型对不同类型的输入返回相同的结果
        self.assertTrue(
            scripted_pdt_model.fn(torch.ones(1), 2), pdt_model.fn(torch.ones(1), 2)
        )

    def test_class_methods(self):
        # 定义一个简单的类 PDTModel
        class PDTModel:
            # 定义类方法 test_sum，计算列表 a 中元素的总和
            def test_sum(self, a):
                return sum(a)

        # 将 PDTModel 类作为全局对象
        make_global(PDTModel)
        # 创建 PDTModel 类的实例
        pdt_model = PDTModel()
        # 定义输入的列表，包含一个元组，元组包含一个列表
        inp: List[Tuple[Any, ...]] = [
            (
                [
                    10,
                    20,
                ],
            ),
        ]
        # 使用 torch.jit.script 将 PDTModel 类进行脚本化，指定输入示例
        scripted_pdt_model = torch.jit.script(
            PDTModel, example_inputs={pdt_model.test_sum: inp}
        )
        # 创建脚本化模型的实例
        script_model = scripted_pdt_model()
        # 断言脚本化模型和原始模型对相同的输入返回相同的结果
        self.assertEqual(
            script_model.test_sum(
                [
                    10,
                    20,
                    30,
                ],
            ),
            pdt_model.test_sum(
                [
                    10,
                    20,
                    30,
                ],
            ),
        )
    # 定义一个测试方法，用于测试具有多个方法的类的功能
    def test_class_with_multiple_methods(self):
        # 定义一个内部类 PDTModelWithManyMethods，用于测试
        class PDTModelWithManyMethods:
            # 定义一个将列表转换为字典的方法
            def test_list_to_dict(self, a):
                # 创建一个空字典，键为浮点数，值为布尔类型
                new_dictionary: Dict[float, bool] = {}
                # 遍历列表中的元素，将每个元素作为键，值设为 True 存入字典
                for element in a:
                    new_dictionary[element] = True
                # 返回生成的字典
                return new_dictionary

            # 定义一个检查字符串 b 是否为字符串 a 的子串的方法
            def test_substring(self, a, b):
                # 返回 b 是否在 a 中的布尔结果
                return b in a

        # 将 PDTModelWithManyMethods 类变为全局类
        make_global(PDTModelWithManyMethods)
        # 创建 PDTModelWithManyMethods 类的实例 pdt_model
        pdt_model = PDTModelWithManyMethods()
        # 定义一个包含元组的列表 list_inp，元组包含单个浮点数列表
        list_inp: List[Tuple[Any, ...]] = [
            (
                [
                    1.2,
                    2.3,
                ],
            ),
        ]
        # 定义一个包含元组的列表 str_inp，元组包含两个字符串
        str_inp: List[Tuple[Any, ...]] = [
            (
                "abc",
                "b",
            ),
        ]
        # 使用 torch.jit.script 将 PDTModelWithManyMethods 类脚本化
        scripted_pdt_model = torch.jit.script(
            PDTModelWithManyMethods,
            example_inputs={
                pdt_model.test_list_to_dict: list_inp,  # 使用 list_inp 作为 test_list_to_dict 方法的示例输入
                pdt_model.test_substring: str_inp,      # 使用 str_inp 作为 test_substring 方法的示例输入
            },
        )
        # 创建脚本化后的模型 script_model 的实例
        script_model = scripted_pdt_model()
        # 断言脚本化模型的 test_list_to_dict 方法与原始 pdt_model 的方法结果相等
        self.assertEqual(
            script_model.test_list_to_dict(
                [
                    1.1,
                    2.2,
                    3.3,
                ],
            ),
            pdt_model.test_list_to_dict(
                [
                    1.1,
                    2.2,
                    3.3,
                ],
            ),
        )
        # 断言脚本化模型的 test_substring 方法与原始 pdt_model 的方法结果相等（检查字符串是否为子串）
        self.assertEqual(
            script_model.test_substring(
                "helloworld",
                "world",
            ),
            pdt_model.test_substring(
                "helloworld",
                "world",
            ),
        )
        # 断言脚本化模型的 test_substring 方法与原始 pdt_model 的方法结果相等（检查字符串是否为子串）
        self.assertEqual(
            script_model.test_substring(
                "helloworld",
                "def",
            ),
            pdt_model.test_substring(
                "helloworld",
                "def",
            ),
        )
    def test_multiple_class_with_same_method(self):
        # 定义第一个类 PDTModelOne，包含方法 test_find，用于检查 b 是否存在于 a 的键中
        class PDTModelOne:
            def test_find(self, a, b):
                return b in a.keys()

        # 定义第二个类 PDTModelTwo，包含方法 test_find，用于检查 b 是否存在于 a 中
        class PDTModelTwo:
            def test_find(self, a, b):
                return b in a

        # 将 PDTModelOne 和 PDTModelTwo 设置为全局可用
        make_global(PDTModelOne, PDTModelTwo)
        
        # 创建 PDTModelOne 和 PDTModelTwo 的实例
        pdt_model_one = PDTModelOne()
        pdt_model_two = PDTModelTwo()
        
        # 定义一个包含元组的列表作为输入，每个元组包含一个字典和一个值
        dict_inp: List[Tuple[Any, ...]] = [
            (
                {
                    1.2: True,
                    2.3: False,
                },
                1.2,
            ),
        ]
        
        # 定义一个包含元组的列表作为输入，每个元组包含一个列表和一个值
        list_inp: List[Tuple[Any, ...]] = [
            (
                [
                    "abc",
                    "b",
                ],
                "c",
            ),
        ]
        
        # 使用 torch.jit.script 对 PDTModelOne 进行脚本化，并指定示例输入
        scripted_pdt_model_one = torch.jit.script(
            PDTModelOne, example_inputs={pdt_model_one.test_find: dict_inp}
        )
        
        # 使用 torch.jit.script 对 PDTModelTwo 进行脚本化，并指定示例输入
        scripted_pdt_model_two = torch.jit.script(
            PDTModelTwo, example_inputs={pdt_model_two.test_find: list_inp}
        )

        # 调用脚本化模型，得到 script_model_one 和 script_model_two
        script_model_one, script_model_two = (
            scripted_pdt_model_one(),
            scripted_pdt_model_two(),
        )
        
        # 断言脚本化模型的 test_find 方法与原始模型的 test_find 方法结果一致
        self.assertEqual(
            script_model_one.test_find(
                {
                    1.1: True,
                    2.2: True,
                    3.3: False,
                },
                4.4,
            ),
            pdt_model_one.test_find(
                {
                    1.1: True,
                    2.2: True,
                    3.3: False,
                },
                4.4,
            ),
        )
        
        # 断言脚本化模型的 test_find 方法与原始模型的 test_find 方法结果一致
        self.assertEqual(
            script_model_two.test_find(
                [
                    "hello",
                    "world",
                ],
                "world",
            ),
            pdt_model_two.test_find(
                [
                    "hello",
                    "world",
                ],
                "world",
            ),
        )
    # 定义测试函数 test_pdt，这是一个单元测试方法
    def test_pdt(self):
        
        # 定义一个简单的加法函数 test_sum，接受两个参数并返回它们的和
        def test_sum(a, b):
            return a + b
        
        # 将 test_sum 函数注册为全局函数，使其可以在 Torch 脚本中调用
        make_global(test_sum)
        # 使用 torch.jit.script 将 test_sum 函数编译成 Torch 脚本，并提供示例输入 (3, 4)
        scripted_fn_add = torch.jit.script(test_sum, example_inputs=[(3, 4)])
        # 断言 Torch 脚本编译后的函数与原始函数在给定参数下的输出相同
        self.assertEqual(scripted_fn_add(10, 2), test_sum(10, 2))
        
        # 定义一个简单的减法函数 test_sub，接受两个参数并返回它们的差
        def test_sub(a, b):
            return a - b
        
        # 将 test_sub 函数注册为全局函数
        make_global(test_sub)
        # 使用 torch.jit.script 将 test_sub 函数编译成 Torch 脚本，并提供示例输入 (3.9, 4.10)
        scripted_fn_sub = torch.jit.script(test_sub, example_inputs=[(3.9, 4.10)])
        # 断言 Torch 脚本编译后的函数与原始函数在给定参数下的输出相同
        self.assertEqual(scripted_fn_sub(6.5, 2.9), test_sub(6.5, 2.9))
        
        # 定义一个简单的乘法函数 test_mul，接受两个参数并返回它们的乘积
        def test_mul(a, b):
            return a * b
        
        # 将 test_mul 函数注册为全局函数
        make_global(test_mul)
        # 使用 torch.jit.script 将 test_mul 函数编译成 Torch 脚本，并提供示例输入 (-10, 9)
        scripted_fn_mul = torch.jit.script(test_mul, example_inputs=[(-10, 9)])
        # 断言 Torch 脚本编译后的函数与原始函数在给定参数下的输出相同
        self.assertEqual(scripted_fn_mul(-1, 3), test_mul(-1, 3))
        
        # 定义一个复杂参数的测试函数 test_args_complex，接受两个参数并返回 Torch 复数
        def test_args_complex(real, img):
            return torch.complex(real, img)
        
        # 将 test_args_complex 函数注册为全局函数
        make_global(test_args_complex)
        # 使用 torch.jit.script 将 test_args_complex 函数编译成 Torch 脚本，
        # 并提供示例输入 (torch.rand(3, 4), torch.rand(3, 4))
        scripted_fn_complex = torch.jit.script(
            test_args_complex, example_inputs=[(torch.rand(3, 4), torch.rand(3, 4))]
        )
        arg1, arg2 = torch.rand(3, 4), torch.rand(3, 4)
        # 断言 Torch 脚本编译后的函数与原始函数在给定参数下的输出相同
        self.assertEqual(scripted_fn_complex(arg1, arg2), test_args_complex(arg1, arg2))
        
        # 定义一个布尔参数的测试函数 test_bool，根据布尔值返回 -1 或 0
        def test_bool(a):
            if a:
                return -1
            else:
                return 0
        
        # 将 test_bool 函数注册为全局函数
        make_global(test_bool)
        # 使用 torch.jit.script 将 test_bool 函数编译成 Torch 脚本，并提供示例输入 (True,)
        scripted_fn_bool = torch.jit.script(test_bool, example_inputs=[(True,)])
        # 断言 Torch 脚本编译后的函数与原始函数在给定参数下的输出相同
        self.assertEqual(scripted_fn_bool(True), test_bool(True))
        
        # 定义一个字符串参数的测试函数 test_str，如果参数为空字符串返回 False，否则返回 True
        def test_str(a):
            if a == "":
                return False
            else:
                return True
        
        # 将 test_str 函数注册为全局函数
        make_global(test_str)
        # 使用 torch.jit.script 将 test_str 函数编译成 Torch 脚本，并提供示例输入 ("",)
        scripted_fn_str = torch.jit.script(test_str, example_inputs=[("",)])
        # 断言 Torch 脚本编译后的函数与原始函数在给定参数下的输出相同
        self.assertEqual(scripted_fn_str("abc"), test_str("abc"))
    # 定义一个测试函数，测试对列表和元组进行求和操作
    def test_pdt_list_and_tuple(self):
        # 定义内部函数 test_list_and_tuple，接收一个参数 a，返回其元素的总和
        def test_list_and_tuple(a):
            return sum(a)

        # 将 test_list_and_tuple 函数注册为全局函数，使其可以在 Torch 脚本中调用
        make_global(test_list_and_tuple)

        # 使用 torch.jit.script 对 test_list_and_tuple 进行脚本化，指定浮点数列表作为示例输入
        scripted_fn_float_list_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[([4.9, 8.9],)]
        )
        # 断言脚本化函数的输出与普通函数对相同输入的输出相等
        self.assertEqual(
            scripted_fn_float_list_input([11.9, 7.6]), test_list_and_tuple([11.9, 7.6])
        )

        # 使用 torch.jit.script 对 test_list_and_tuple 进行脚本化，指定布尔值列表作为示例输入
        scripted_fn_bool_list_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[([True, False, True],)]
        )
        # 断言脚本化函数的输出与普通函数对相同输入的输出相等
        self.assertEqual(
            scripted_fn_bool_list_input([True, True, True]),
            test_list_and_tuple([True, True, True]),
        )

        # 使用 torch.jit.script 对 test_list_and_tuple 进行脚本化，指定整数列表作为示例输入
        scripted_fn_int_list_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[([3, 4, 5],)]
        )
        # 断言脚本化函数的输出与普通函数对相同输入的输出相等
        self.assertEqual(
            scripted_fn_int_list_input([1, 2, 3]), test_list_and_tuple([1, 2, 3])
        )

        # 使用 torch.jit.script 对 test_list_and_tuple 进行脚本化，指定浮点数元组作为示例输入
        scripted_fn_float_tuple_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[((4.9, 8.9),)]
        )
        # 断言脚本化函数的输出与普通函数对相同输入的输出相等
        self.assertEqual(
            scripted_fn_float_tuple_input((11.9, 7.6)), test_list_and_tuple((11.9, 7.6))
        )

        # 使用 torch.jit.script 对 test_list_and_tuple 进行脚本化，指定布尔值元组作为示例输入
        scripted_fn_bool_tuple_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[((True, False, True),)]
        )
        # 断言脚本化函数的输出与普通函数对相同输入的输出相等
        self.assertEqual(
            scripted_fn_bool_tuple_input((True, True, True)),
            test_list_and_tuple((True, True, True)),
        )

        # 使用 torch.jit.script 对 test_list_and_tuple 进行脚本化，指定整数元组作为示例输入
        scripted_fn_int_tuple_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[((3, 4, 5),)]
        )
        # 断言脚本化函数的输出与普通函数对相同输入的输出相等
        self.assertEqual(
            scripted_fn_int_tuple_input((1, 2, 3)), test_list_and_tuple((1, 2, 3))
        )

    # 定义一个测试函数，测试对字典进行操作，获取指定键的值
    def test_pdt_dict(self):
        # 定义内部函数 test_dict，接收一个字典 a，返回其键 "foo" 对应的值
        def test_dict(a):
            return a["foo"]

        # 定义内部函数 test_dict_int_list，接收一个字典 a，返回其索引为 1 的值
        def test_dict_int_list(a):
            return a[1]

        # 将 test_dict 和 test_dict_int_list 函数注册为全局函数，使其可以在 Torch 脚本中调用
        make_global(test_dict, test_dict_int_list)

        # 准备一个字符串和布尔值组成的字典作为输入，使用 torch.jit.script 对 test_dict 进行脚本化
        str_bool_inp = {"foo": True, "bar": False}
        scripted_fn = torch.jit.script(test_dict, example_inputs=[(str_bool_inp,)])
        # 断言脚本化函数的输出与普通函数对相同输入的输出相等
        self.assertEqual(
            scripted_fn(
                {"foo": False, "bar": True},
            ),
            test_dict(
                {"foo": False, "bar": True},
            ),
        )

        # 准备一个整数和布尔值列表组成的字典作为输入，使用 torch.jit.script 对 test_dict_int_list 进行脚本化
        str_list_inp = {0: [True, False], 1: [False, True]}
        scripted_fn = torch.jit.script(
            test_dict_int_list, example_inputs=[(str_list_inp,)]
        )
        # 断言脚本化函数的输出与普通函数对相同输入的输出相等
        self.assertEqual(
            scripted_fn(
                {0: [False, False], 1: [True, True]},
            ),
            test_dict_int_list(
                {0: [False, False], 1: [True, True]},
            ),
        )
    def test_any(self):
        # 定义用于测试多种类型参数的函数
        def test_multiple_types(a):
            # 断言参数 a 不是布尔类型
            assert not isinstance(a, bool)
            # 返回参数 a
            return a

        # 定义用于测试多种类型参数并返回不同结果的函数
        def test_multiple_type_refinement(a):
            # 如果参数 a 是布尔类型，返回 1
            if isinstance(a, bool):
                return 1
            # 如果参数 a 是整数类型，返回 1 加上参数 a
            elif isinstance(a, int):
                return 1 + a
            # 如果参数 a 是浮点数类型，返回 1 加上参数 a 的整数部分
            elif isinstance(a, float):
                return 1 + int(a)
            # 其他情况返回 -1
            else:
                return -1

        # 将定义的函数注册为全局函数
        make_global(test_multiple_types, test_multiple_type_refinement)

        # 使用 torch.jit.script 将函数转换为脚本化函数，提供示例输入以进行类型推断
        scripted_fn = torch.jit.script(
            test_multiple_types, example_inputs=[(1,), ("abc",), (8.9,), ([3, 4, 5],)]
        )
        # 断言脚本化函数的结果与原始函数对相同输入的结果一致
        self.assertEqual(scripted_fn(10), test_multiple_types(10))
        self.assertEqual(scripted_fn("def"), test_multiple_types("def"))
        self.assertEqual(scripted_fn(7.89999), test_multiple_types(7.89999))
        self.assertEqual(scripted_fn([10, 11, 14]), test_multiple_types([10, 11, 14]))

        # 使用 torch.jit.script 将函数转换为脚本化函数，提供示例输入以进行类型推断
        scripted_fn = torch.jit.script(
            test_multiple_type_refinement,
            example_inputs=[
                (1,),
                ("abc",),
                (8.9,),
                ([3, 4, 5],),
                (True,),
                ({"a": True},),
            ],
        )
        # 断言脚本化函数的结果与原始函数对相同输入的结果一致
        self.assertEqual(scripted_fn(10), test_multiple_type_refinement(10))
        self.assertEqual(scripted_fn("def"), test_multiple_type_refinement("def"))
        self.assertEqual(scripted_fn(7.89999), test_multiple_type_refinement(7.89999))
        self.assertEqual(
            scripted_fn([10, 11, 14]), test_multiple_type_refinement([10, 11, 14])
        )
        self.assertEqual(scripted_fn(False), test_multiple_type_refinement(False))
        self.assertEqual(
            scripted_fn({"abc": True, "def": False}),
            test_multiple_type_refinement({"abc": True, "def": False}),
        )
    def test_class_as_profiled_types(self):
        # 定义一个内部类 UserDefinedClass，用于测试的用户自定义类
        class UserDefinedClass:
            # 定义方法 fn，参数 b 的类型为 Any，返回值根据 b 的类型不同进行条件判断
            def fn(self, b) -> Any:
                # 断言 b 不为空
                assert b is not None
                # 如果 b 是整数，返回 b（如果大于 0），否则返回 -1
                if isinstance(b, int):
                    return b if b > 0 else -1
                # 如果 b 是浮点数，返回 b（如果大于 0.0），否则返回 -1.0
                elif isinstance(b, float):
                    return b if b > 0.0 else -1.0
                # 如果 b 不是整数或浮点数，返回 0
                return 0

        # 定义函数 test_model，参数 a 和 m，用于测试模型
        def test_model(a, m):
            # 断言 a 不是布尔类型
            assert not isinstance(a, bool)
            # 调用 m 的 fn 方法，传入参数 a
            return m.fn(a)

        # 将 UserDefinedClass 和 test_model 函数注册为全局函数
        make_global(UserDefinedClass, test_model)

        # 创建 UserDefinedClass 实例
        user_class = UserDefinedClass()
        # 使用 torch.jit.script 对 test_model 进行脚本化，指定例子输入
        scripted_fn = torch.jit.script(
            test_model,
            example_inputs=[
                (
                    10,
                    user_class,
                ),
                (
                    10.9,
                    user_class,
                ),
            ],
        )
        # 断言脚本化函数调用结果与普通函数调用结果相同
        self.assertEqual(
            scripted_fn(
                100,
                user_class,
            ),
            test_model(100, user_class),
        )
        # 断言脚本化函数调用结果与普通函数调用结果相同
        self.assertEqual(
            scripted_fn(
                1.9,
                user_class,
            ),
            test_model(1.9, user_class),
        )

    def test_class_with_args_as_profiled_types(self):
        # 定义一个内部类 ClassWithArgs，带有初始化参数 a，用于测试带参数的类
        class ClassWithArgs:
            def __init__(self, a: bool):
                self.a = a

            # 定义方法 fn，参数 b
            def fn(self, b):
                # 如果 self.a 为真，返回 b，否则返回 -1
                if self.a:
                    return b
                else:
                    return -1

        # 定义函数 test_model_with_args，参数 a 和 m，用于测试带参数的模型
        def test_model_with_args(a, m):
            # 断言 a 不是布尔类型
            assert not isinstance(a, bool)
            # 调用 m 的 fn 方法，传入参数 a
            return m.fn(a)

        # 将 ClassWithArgs 和 test_model_with_args 函数注册为全局函数
        make_global(ClassWithArgs, test_model_with_args)

        # 创建 ClassWithArgs 实例
        user_class = ClassWithArgs(False)
        # 使用 torch.jit.script 对 test_model_with_args 进行脚本化，指定例子输入
        scripted_fn = torch.jit.script(
            test_model_with_args,
            example_inputs=[
                (
                    10,
                    user_class,
                ),
                (
                    10.9,
                    user_class,
                ),
            ],
        )
        # 断言脚本化函数调用结果与普通函数调用结果相同
        self.assertEqual(
            scripted_fn(
                100,
                ClassWithArgs(True),
            ),
            test_model_with_args(100, ClassWithArgs(True)),
        )

    def test_nn_parameter_as_arg(self):
        # 定义一个继承自 torch.nn.Module 的内部类 TestNNParameter，用于测试神经网络参数作为参数的情况
        class TestNNParameter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个形状为 (2, 3) 的神经网络参数 inp，初始值为全 1
                self.inp = torch.nn.Parameter(torch.ones(2, 3))

            # 定义方法 add_nn_parameter_with_int，接受 x 和 y 作为参数，返回 x + y 的结果
            def add_nn_parameter_with_int(self, x, y):
                return torch.add(x, y)

            # 定义 forward 方法，接受参数 y，调用 add_nn_parameter_with_int 方法，传入 self.inp 和 y
            def forward(self, y):
                return self.add_nn_parameter_with_int(self.inp, y)

        # 将 TestNNParameter 类注册为全局函数
        make_global(TestNNParameter)
        # 创建 TestNNParameter 实例 pdt_model
        pdt_model = TestNNParameter()
        # 使用 torch.jit.script 对 pdt_model 进行脚本化，指定例子输入
        scripted_fn = torch.jit.script(
            pdt_model,
            example_inputs={
                pdt_model: [
                    (10,),
                ],
            },
        )
        # 断言脚本化函数调用结果与 pdt_model 调用结果相同
        self.assertEqual(scripted_fn(20), pdt_model(20))
    # 定义一个测试方法，用于验证带有类型提示的 FXModel 的脚本化行为
    def test_fx_tracing_with_typing(self):
        # 定义一个命名元组 FXModelOutput，表示 FXModel 的输出结果是一个整数列表
        class FXModelOutput(NamedTuple):
            result: List[int]

        # 定义一个继承自 torch.nn.Module 的 FXModel 类
        class FXModel(torch.nn.Module):
            # 重写 forward 方法，指定其输入参数 a，并声明其返回类型为 FXModelOutput
            def forward(self, a) -> FXModelOutput:
                # 创建 FXModelOutput 实例，其 result 成员变量使用输入参数 a 初始化
                result = FXModelOutput(result=a)
                return result

        # 将 FXModel 和 FXModelOutput 作为全局变量
        make_global(FXModel, FXModelOutput)

        # 创建 FXModel 的实例 pdt_model
        pdt_model = FXModel()

        # 使用 torch.jit.script 方法将 pdt_model 脚本化，并提供示例输入作为 example_inputs
        scripted_fn = torch.jit.script(
            pdt_model,
            example_inputs={
                pdt_model: [
                    (
                        [
                            10,
                            20,
                        ],
                    ),
                ],
            },
        )

        # 断言脚本化函数的输出与原始函数调用的输出一致
        self.assertEqual(scripted_fn([20]), pdt_model([20]))

    # 定义一个测试方法，用于验证将 NoneType 作为类型 Optional 的行为
    def test_nonetype_as_optional_of_type(self):
        # 定义一个函数 test_none，参数 a 的类型为 Any，返回值也为 Any
        def test_none(a) -> Any:
            # 如果 a 是 None，则返回整数 0
            if a is None:
                return 0
            else:
                # 否则返回 a 加上一个包含单个元素的张量 torch.ones(1)
                return a + torch.ones(1)

        # 将 test_none 函数作为全局变量
        make_global(test_none)

        # 使用 torch.jit.script 方法将 test_none 脚本化，并提供多个示例输入
        scripted_fn = torch.jit.script(test_none, example_inputs=[(None,), (10.6,)])
        # 断言脚本化函数的输出与原始函数调用的输出一致
        self.assertEqual(
            scripted_fn(
                30.9,
            ),
            test_none(
                30.9,
            ),
        )

        # 继续使用 torch.jit.script 方法将 test_none 脚本化，提供不同的示例输入
        scripted_fn = torch.jit.script(test_none, example_inputs=[(None,), (10,)])
        # 断言脚本化函数的输出与原始函数调用的输出一致
        self.assertEqual(
            scripted_fn(
                2,
            ),
            test_none(
                2,
            ),
        )

        # 再次使用 torch.jit.script 方法将 test_none 脚本化，提供包含张量的示例输入
        scripted_fn = torch.jit.script(
            test_none, example_inputs=[(None,), (torch.Tensor(1),)]
        )
        # 断言脚本化函数的输出与原始函数调用的输出一致
        self.assertEqual(
            scripted_fn(
                torch.ones(1),
            ),
            test_none(
                torch.ones(1),
            ),
        )
```