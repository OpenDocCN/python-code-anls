# `.\pytorch\test\mobile\test_lite_script_module.py`

```
# Owner(s): ["oncall: mobile"]

# 导入必要的模块和类
import inspect
import io
from tempfile import TemporaryFileName
from typing import Dict, List

import torch
import torch.utils.bundled_inputs

# 导入相关的 Torch 模块和函数
from torch.jit.mobile import _export_operator_list, _load_for_lite_interpreter
from torch.testing import FileCheck
from torch.testing._internal.common_quantization import (
    AnnotatedNestedModel,
    AnnotatedSingleLayerLinearModel,
    QuantizationLiteTestCase,
    TwoLayerLinearModel,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestLiteScriptModule(TestCase):
    # 定义函数，用于获取脚本化模块的导出和导入副本
    def getScriptExportImportCopy(
        self, m, save_mobile_debug_info=True, also_test_file=False
    ):
        # 对输入模块 m 进行脚本化
        m_scripted = torch.jit.script(m)

        # 如果不需要测试文件也不需要保存移动端调试信息
        if not also_test_file:
            # 将脚本化模块保存到字节流中，供 Lite 解释器使用
            buffer = io.BytesIO(
                m_scripted._save_to_buffer_for_lite_interpreter(
                    _save_mobile_debug_info=save_mobile_debug_info
                )
            )
            buffer.seek(0)
            # 从字节流中加载 Lite 解释器可以使用的移动模块
            mobile_module = _load_for_lite_interpreter(buffer)
            return mobile_module

        # 如果需要同时测试文件，并且需要保存移动端调试信息
        with TemporaryFileName() as fname:
            # 将脚本化模块保存到临时文件中，供 Lite 解释器使用
            m_scripted._save_for_lite_interpreter(
                fname, _save_mobile_debug_info=save_mobile_debug_info
            )
            # 从临时文件中加载 Lite 解释器可以使用的移动模块
            mobile_module = _load_for_lite_interpreter(fname)
            return mobile_module

    # 测试加载移动模块的方法
    def test_load_mobile_module(self):
        # 定义一个简单的测试模块
        class MyTestModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        input = torch.tensor([1])

        # 对测试模块进行脚本化
        script_module = torch.jit.script(MyTestModule())
        # 运行脚本化模块的 forward 方法，获取结果
        script_module_result = script_module(input)

        # 将脚本化模块保存到字节流中
        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        # 从字节流中加载 Lite 解释器可以使用的移动模块
        mobile_module = _load_for_lite_interpreter(buffer)

        # 在移动模块上运行输入数据，并进行结果比较
        mobile_module_result = mobile_module(input)
        torch.testing.assert_close(script_module_result, mobile_module_result)

        # 在移动模块上直接调用 forward 方法，并进行结果比较
        mobile_module_forward_result = mobile_module.forward(input)
        torch.testing.assert_close(script_module_result, mobile_module_forward_result)

        # 在移动模块上调用 run_method 方法执行 forward，并进行结果比较
        mobile_module_run_method_result = mobile_module.run_method("forward", input)
        torch.testing.assert_close(
            script_module_result, mobile_module_run_method_result
        )
    def test_save_mobile_module_with_debug_info_with_trace(self):
        # 定义内部类 A，继承自 torch.nn.Module，用于定义模块的 forward 方法
        class A(torch.nn.Module):
            def forward(self, x, y):
                return x * y

        # 定义内部类 B，继承自 torch.nn.Module
        class B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建 A 类的两个实例 A0 和 A1
                self.A0 = A()
                self.A1 = A()

            def forward(self, x, y, z):
                # 返回 A0 和 A1 实例的前向传播结果的和
                return self.A0(x, y) + self.A1(y, z)

        # 遍历导出方法列表 ["trace", "script"]
        for export_method in ["trace", "script"]:
            # 生成随机输入张量 x, y, z
            x = torch.rand((2, 3))
            y = torch.rand((2, 3))
            z = torch.rand((2, 3))
            # 根据导出方法选择使用 trace 或 script 方法导出模型 B
            if export_method == "trace":
                trace_module = torch.jit.trace(B(), [x, y, z])
            else:
                trace_module = torch.jit.script(B())
            # 将导出的模型保存为适用于 Lite 解释器的字节流，并包含调试信息
            exported_module = trace_module._save_to_buffer_for_lite_interpreter(
                _save_mobile_debug_info=True
            )
            # 将导出的模型字节流包装为 BytesIO 对象
            buffer = io.BytesIO(exported_module)
            buffer.seek(0)

            # 断言导出的模型字节流中包含调用栈调试映射文件的标识
            assert b"callstack_debug_map.pkl" in exported_module

            # 使用自定义函数 _load_for_lite_interpreter 加载 Lite 解释器模型
            mobile_module = _load_for_lite_interpreter(buffer)
            # 断言模型在运行时抛出指定异常，验证调试信息的正确性
            with self.assertRaisesRegex(
                RuntimeError,
                r"Module hierarchy:top\(B\)::<unknown>.A0\(A\)::forward.aten::mul",
            ):
                x = torch.rand((2, 3))
                y = torch.rand((8, 10))
                z = torch.rand((8, 10))
                mobile_module(x, y, z)
            with self.assertRaisesRegex(
                RuntimeError,
                r"Module hierarchy:top\(B\)::<unknown>.A1\(A\)::forward.aten::mul",
            ):
                x = torch.rand((2, 3))
                y = torch.rand((2, 3))
                z = torch.rand((8, 10))
                mobile_module(x, y, z)

    def test_load_mobile_module_with_debug_info(self):
        # 定义简单的测试模块 MyTestModule，继承自 torch.nn.Module
        class MyTestModule(torch.nn.Module):
            def forward(self, x):
                return x + 5

        # 创建输入张量 input
        input = torch.tensor([3])

        # 使用 script 方法对 MyTestModule 进行脚本化
        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(input)

        # 将脚本化后的模型保存为适用于 Lite 解释器的字节流，并包含调试信息
        buffer = io.BytesIO(
            script_module._save_to_buffer_for_lite_interpreter(
                _save_mobile_debug_info=True
            )
        )
        buffer.seek(0)
        # 使用自定义函数 _load_for_lite_interpreter 加载 Lite 解释器模型
        mobile_module = _load_for_lite_interpreter(buffer)

        # 比较脚本化模型和 Lite 解释器加载后的模型在相同输入下的输出是否接近
        mobile_module_result = mobile_module(input)
        torch.testing.assert_close(script_module_result, mobile_module_result)

        # 测试 Lite 解释器模型的 forward 方法的结果是否与脚本化模型一致
        mobile_module_forward_result = mobile_module.forward(input)
        torch.testing.assert_close(script_module_result, mobile_module_forward_result)

        # 测试 Lite 解释器模型的 run_method 方法调用 forward 方法的结果是否与脚本化模型一致
        mobile_module_run_method_result = mobile_module.run_method("forward", input)
        torch.testing.assert_close(
            script_module_result, mobile_module_run_method_result
        )
    # 定义一个测试方法，用于测试查找并运行方法
    def test_find_and_run_method(self):
        # 定义一个简单的测试模块，继承自torch.nn.Module
        class MyTestModule(torch.nn.Module):
            # 实现模块的前向传播方法
            def forward(self, arg):
                return arg
        
        # 准备输入数据，这里是一个包含单个张量的元组
        input = (torch.tensor([1]),)
        
        # 使用torch.jit.script将MyTestModule模块转换为脚本模块
        script_module = torch.jit.script(MyTestModule())
        # 在脚本模块上执行前向传播，得到结果
        script_module_result = script_module(*input)
        
        # 将脚本模块保存到字节流中，以便于lite解释器加载
        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        # 使用_load_for_lite_interpreter函数加载lite解释器可用的模块
        mobile_module = _load_for_lite_interpreter(buffer)
        
        # 在mobile_module中查找名为"get_all_bundled_inputs"的方法
        has_bundled_inputs = mobile_module.find_method("get_all_bundled_inputs")
        # 断言模块中不包含"get_all_bundled_inputs"方法
        self.assertFalse(has_bundled_inputs)
        
        # 使用torch.utils.bundled_inputs.augment_model_with_bundled_inputs为script_module添加捆绑输入
        torch.utils.bundled_inputs.augment_model_with_bundled_inputs(
            script_module, [input], []
        )
        
        # 重新保存带有捆绑输入的script_module到字节流中
        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        # 重新加载mobile_module
        mobile_module = _load_for_lite_interpreter(buffer)
        
        # 再次查找名为"get_all_bundled_inputs"的方法
        has_bundled_inputs = mobile_module.find_method("get_all_bundled_inputs")
        # 断言模块中现在包含"get_all_bundled_inputs"方法
        self.assertTrue(has_bundled_inputs)
        
        # 调用mobile_module的"get_all_bundled_inputs"方法，获取捆绑输入
        bundled_inputs = mobile_module.run_method("get_all_bundled_inputs")
        # 执行模块的前向传播，使用捆绑输入的第一个元素
        mobile_module_result = mobile_module.forward(*bundled_inputs[0])
        # 断言脚本模块和移动模块的前向传播结果接近
        torch.testing.assert_close(script_module_result, mobile_module_result)

    # 定义测试方法，测试带可选参数的方法调用
    def test_method_calls_with_optional_arg(self):
        # 定义A类，继承自torch.nn.Module
        class A(torch.nn.Module):
            def __init__(self):
                super().__init__()
            
            # 实现前向传播方法，包含一个可选参数two，默认值为2
            def forward(self, x, two: int = 2):
                return x + two
        
        # 定义B类，继承自torch.nn.Module
        class B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.A0 = A()
            
            # 实现前向传播方法，包含一个可选参数one，默认值为1
            def forward(self, x, one: int = 1):
                return self.A0(x) + one
        
        # 使用torch.jit.script将B类转换为脚本模块
        script_module = torch.jit.script(B())
        # 将脚本模块保存到字节流中
        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        # 使用_load_for_lite_interpreter函数加载lite解释器可用的模块
        mobile_module = _load_for_lite_interpreter(buffer)
        
        # 准备输入数据，这里是一个包含单个张量的Tensor
        input = torch.tensor([5])
        # 在script_module上执行前向传播，得到结果
        script_module_forward_result = script_module.forward(input)
        # 在mobile_module上执行前向传播，得到结果
        mobile_module_forward_result = mobile_module.forward(input)
        # 断言脚本模块和移动模块的前向传播结果接近
        torch.testing.assert_close(
            script_module_forward_result, mobile_module_forward_result
        )
        
        # 修改reference值
        script_module_forward_result = script_module.forward(input, 2)
        # 断言修改后的脚本模块和移动模块的前向传播结果不完全匹配
        self.assertFalse(
            (script_module_forward_result == mobile_module_forward_result).all().item()
        )
        
        # 再次修改为相匹配的结果
        mobile_module_forward_result = mobile_module.forward(input, 2)
        # 断言现在脚本模块和移动模块的前向传播结果接近
        torch.testing.assert_close(
            script_module_forward_result, mobile_module_forward_result
        )
    # 定义一个测试方法，用于测试不支持的类类型情况
    def test_unsupported_classtype(self):
        # 定义一个名为Foo的类
        class Foo:
            # 构造函数，没有具体实现，直接返回
            def __init__(self):
                return

            # 类方法func，接收两个整数参数并返回它们的和
            def func(self, x: int, y: int):
                return x + y

        # 定义一个继承自torch.nn.Module的测试模块类MyTestModule
        class MyTestModule(torch.nn.Module):
            # 前向传播方法
            def forward(self, arg):
                # 创建Foo类的实例f
                f = Foo()
                # 调用Foo类实例的func方法，传入参数1和2，返回结果
                return f.func(1, 2)

        # 使用torch.jit.script将MyTestModule实例化为script_module
        script_module = torch.jit.script(MyTestModule())
        
        # 使用assertRaisesRegex断言捕获RuntimeError异常，检查异常信息是否符合正则表达式
        with self.assertRaisesRegex(
            RuntimeError,
            r"Workaround: instead of using arbitrary class type \(class Foo\(\)\), "
            r"define a pytorch class \(class Foo\(torch\.nn\.Module\)\)\. "
            r"The problematic type is: ",
        ):
            # 调用_script_to_buffer_for_lite_interpreter方法，期望抛出异常
            script_module._save_to_buffer_for_lite_interpreter()

    # 定义一个测试方法，用于测试返回值为列表且包含模块类的情况
    def test_unsupported_return_list_with_module_class(self):
        # 定义一个继承自torch.nn.Module的Foo类
        class Foo(torch.nn.Module):
            pass

        # 定义一个继承自torch.nn.Module的测试模块类MyTestModuleForListWithModuleClass
        class MyTestModuleForListWithModuleClass(torch.nn.Module):
            # 构造函数
            def __init__(self):
                super().__init__()
                # 创建Foo类的实例self.foo
                self.foo = Foo()

            # 前向传播方法
            def forward(self):
                # 声明一个类型为List[Foo]的变量my_list，包含self.foo作为元素
                my_list: List[Foo] = [self.foo]
                # 返回my_list
                return my_list

        # 使用torch.jit.script将MyTestModuleForListWithModuleClass实例化为script_module
        script_module = torch.jit.script(MyTestModuleForListWithModuleClass())

        # 使用assertRaisesRegex断言捕获RuntimeError异常，检查异常信息是否符合正则表达式
        with self.assertRaisesRegex(
            RuntimeError,
            r"^Returning a list or dictionary with pytorch class type "
            r"is not supported in mobile module "
            r"\(List\[Foo\] or Dict\[int\, Foo\] for class Foo\(torch\.nn\.Module\)\)\. "
            r"Workaround\: instead of using pytorch class as their element type\, "
            r"use a combination of list\, dictionary\, and single types\.$",
        ):
            # 调用_script_to_buffer_for_lite_interpreter方法，期望抛出异常
            script_module._save_to_buffer_for_lite_interpreter()

    # 定义一个测试方法，用于测试返回值为字典且包含模块类的情况
    def test_unsupported_return_dict_with_module_class(self):
        # 定义一个继承自torch.nn.Module的Foo类
        class Foo(torch.nn.Module):
            pass

        # 定义一个继承自torch.nn.Module的测试模块类MyTestModuleForDictWithModuleClass
        class MyTestModuleForDictWithModuleClass(torch.nn.Module):
            # 构造函数
            def __init__(self):
                super().__init__()
                # 创建Foo类的实例self.foo
                self.foo = Foo()

            # 前向传播方法
            def forward(self):
                # 声明一个类型为Dict[int, Foo]的变量my_dict，包含键为1，值为self.foo的项
                my_dict: Dict[int, Foo] = {1: self.foo}
                # 返回my_dict
                return my_dict

        # 使用torch.jit.script将MyTestModuleForDictWithModuleClass实例化为script_module
        script_module = torch.jit.script(MyTestModuleForDictWithModuleClass())

        # 使用assertRaisesRegex断言捕获RuntimeError异常，检查异常信息是否符合正则表达式
        with self.assertRaisesRegex(
            RuntimeError,
            r"^Returning a list or dictionary with pytorch class type "
            r"is not supported in mobile module "
            r"\(List\[Foo\] or Dict\[int\, Foo\] for class Foo\(torch\.nn\.Module\)\)\. "
            r"Workaround\: instead of using pytorch class as their element type\, "
            r"use a combination of list\, dictionary\, and single types\.$",
        ):
            # 调用_script_to_buffer_for_lite_interpreter方法，期望抛出异常
            script_module._save_to_buffer_for_lite_interpreter()
    # 定义一个测试函数，用于测试模块导出操作符列表的功能
    def test_module_export_operator_list(self):
        # 定义一个名为Foo的内嵌类，继承自torch.nn.Module
        class Foo(torch.nn.Module):
            # 构造函数，初始化权重和偏置
            def __init__(self):
                super().__init__()
                self.weight = torch.ones((20, 1, 5, 5))  # 初始化权重为全1的张量
                self.bias = torch.ones(20)  # 初始化偏置为全1的张量

            # 前向传播函数，接受输入input，返回三个张量x1, x2, x3的元组
            def forward(self, input):
                x1 = torch.zeros(2, 2)  # 创建一个全0的2x2张量x1
                x2 = torch.empty_like(torch.empty(2, 2))  # 根据空2x2张量创建一个同类型的空张量x2
                x3 = torch._convolution(  # 执行卷积操作，计算卷积结果x3
                    input,
                    self.weight,
                    self.bias,
                    [1, 1],  # 卷积步长为1
                    [0, 0],  # 补0填充
                    [1, 1],  # 空洞卷积参数
                    False,  # 不使用转置卷积
                    [0, 0],  # 转置卷积填充参数
                    1,  # 分组卷积参数
                    False,  # 不进行混合精度卷积
                    False,  # 不进行混合精度卷积
                    True,  # 输出应为3D
                    True,  # 保存在缓冲区
                )
                return (x1, x2, x3)  # 返回三个张量的元组作为前向传播结果

        # 通过torch.jit.script将Foo类实例化为脚本模块对象m
        m = torch.jit.script(Foo())

        # 将脚本模块对象m保存到字节流buffer中，以备Lite解释器使用
        buffer = io.BytesIO(m._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)

        # 从字节流buffer中加载Lite解释器使用的模块mobile_module
        mobile_module = _load_for_lite_interpreter(buffer)

        # 预期的操作符集合，包含了模块中使用的四种操作符
        expected_ops = {
            "aten::_convolution",
            "aten::empty.memory_format",
            "aten::empty_like",
            "aten::zeros",
        }

        # 调用_export_operator_list函数获取实际操作符集合actual_ops
        actual_ops = _export_operator_list(mobile_module)

        # 使用self.assertEqual断言实际操作符集合和预期操作符集合相等
        self.assertEqual(actual_ops, expected_ops)

    # 定义一个测试函数，用于测试源代码范围异常处理的功能
    def test_source_range_simple(self):
        # 定义一个名为FooTest的torch.jit.ScriptModule子类
        class FooTest(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, w):
                return torch.mm(x, w.t())  # 执行矩阵乘法操作

        # 创建FooTest类的实例ft
        ft = FooTest()

        # 调用self.getScriptExportImportCopy方法获取FooTest类的导出-导入副本loaded
        loaded = self.getScriptExportImportCopy(ft)

        # 获取FooTest类的源代码行号
        _, lineno = inspect.getsourcelines(FooTest)

        # 使用self.assertRaisesRegex断言在调用loaded时抛出RuntimeError，并包含特定错误消息
        with self.assertRaisesRegex(
            RuntimeError, f'test_lite_script_module.py", line {lineno + 3}'
        ):
            loaded(torch.rand(3, 4), torch.rand(30, 40))

    # 定义一个测试函数，用于测试异常处理时的源代码范围处理功能
    def test_source_range_raise_exception(self):
        # 定义一个名为FooTest2的torch.jit.ScriptModule子类
        class FooTest2(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self):
                raise RuntimeError("foo")  # 抛出一个包含"foo"消息的RuntimeError

        # 获取FooTest2类的源代码行号
        _, lineno = inspect.getsourcelines(FooTest2)

        # 使用self.assertRaisesRegex断言在调用loaded时抛出torch.jit.Error，并包含特定错误消息
        with self.assertRaisesRegex(torch.jit.Error, "foo"):
            # 创建FooTest2类的实例ft
            ft = FooTest2()
            # 调用self.getScriptExportImportCopy方法获取FooTest2类的导出-导入副本loaded
            loaded = self.getScriptExportImportCopy(ft)
            # 调用loaded()触发前向传播函数，引发异常
            loaded()
    def test_source_range_function_call(self):
        class FooTest3(torch.jit.ScriptModule):
            @torch.jit.script_method
            def add_method(self, x, w):
                return x + w

            @torch.jit.script_method
            def forward(self, x, y, w):
                x = x * y  # 对输入张量 x 和 y 进行逐元素乘法运算
                x = x + 2  # 对结果进行标量加法运算
                return self.add_method(x, w)  # 调用自定义的 add_method 方法进行张量加法

        ft = FooTest3()  # 创建 FooTest3 类的实例
        loaded = self.getScriptExportImportCopy(ft)  # 获取 FooTest3 实例的导出和导入副本
        _, lineno = inspect.getsourcelines(FooTest3)  # 获取 FooTest3 类定义的源代码行号

        try:
            loaded(torch.rand(3, 4), torch.rand(3, 4), torch.rand(30, 40))  # 调用加载后的模型实例进行前向推理
        except RuntimeError as e:
            error_message = f"{e}"  # 捕获运行时异常的错误信息
        self.assertTrue(
            f'test_lite_script_module.py", line {lineno + 3}' in error_message  # 验证异常信息中是否包含特定行号信息
        )
        self.assertTrue(
            f'test_lite_script_module.py", line {lineno + 9}' in error_message  # 验证异常信息中是否包含特定行号信息
        )
        self.assertTrue("top(FooTest3)" in error_message)  # 验证异常信息中是否包含特定字符串

    def test_source_range_no_debug_info(self):
        class FooTest4(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, w):
                return torch.mm(x, w.t())  # 执行张量 x 和 w 转置矩阵的矩阵乘法

        ft = FooTest4()  # 创建 FooTest4 类的实例
        loaded = self.getScriptExportImportCopy(ft, save_mobile_debug_info=False)  # 获取 FooTest4 实例的导出和导入副本，但不保存移动端调试信息

        try:
            loaded(torch.rand(3, 4), torch.rand(30, 40))  # 调用加载后的模型实例进行前向推理
        except RuntimeError as e:
            error_message = f"{e}"  # 捕获运行时异常的错误信息
        self.assertTrue("test_lite_script_module.py" not in error_message)  # 验证异常信息中不包含特定文件名信息

    def test_source_range_raise_exc(self):
        class FooTest5(torch.jit.ScriptModule):
            def __init__(self, val: int):
                super().__init__()
                self.val = val

            @torch.jit.script_method
            def add_method(self, val: int, x, w):
                if val == self.val:
                    raise RuntimeError("self.val and val are same")  # 如果 val 和 self.val 相等，则抛出运行时异常
                return x + w

            @torch.jit.script_method
            def forward(self, val: int, x, y, w):
                x = x * y  # 对输入张量 x 和 y 进行逐元素乘法运算
                x = x + 2  # 对结果进行标量加法运算
                return self.add_method(val, x, w)  # 调用自定义的 add_method 方法进行张量加法

        ft = FooTest5(42)  # 创建 FooTest5 类的实例，并初始化 self.val 为 42
        loaded = self.getScriptExportImportCopy(ft)  # 获取 FooTest5 实例的导出和导入副本
        _, lineno = inspect.getsourcelines(FooTest5)  # 获取 FooTest5 类定义的源代码行号

        try:
            loaded(42, torch.rand(3, 4), torch.rand(3, 4), torch.rand(30, 40))  # 调用加载后的模型实例进行前向推理
        except torch.jit.Error as e:
            error_message = f"{e}"  # 捕获 Torch JIT 引擎异常的错误信息

        # In C++ code, the type of exception thrown is torch::jit::JITException
        # which does not extend c10::Error, and hence it isn't possible to add
        # additional context to the exception message and preserve the correct
        #  C++ stack trace for symbolication. i.e. it isn't possible to add
        # the debug handle string to show where in the Python code the exception
        # occured w/o first changing
        # torch::jit::JITException to extend c10::Error.
        self.assertTrue("self.val and val are same" in error_message)  # 验证异常信息中是否包含特定字符串
    def test_stacktrace_interface_call(self):
        # 定义一个 TorchScript 接口 Forward，要求实现 forward 方法和 forwardError 方法
        @torch.jit.interface
        class Forward(torch.nn.Module):
            def forward(self, x) -> torch.Tensor:
                pass

            def forwardError(self, x) -> torch.Tensor:
                pass

        # 定义一个继承自 Module 的类 B，实现了 Forward 接口的要求
        class B(torch.nn.Module):
            # 实现 forward 方法，返回输入的张量 x
            def forward(self, x):
                return x

            # 实现 forwardError 方法，调用 self.call() 方法并返回结果
            def forwardError(self, x):
                return self.call() + x

            # 定义一个调用方法 call()，返回一个包含值为 -1 的张量
            def call(self):
                return torch.ones(-1)

        # 定义一个继承自 Module 的类 A，包含一个 Forward 接口类型的成员变量 b
        class A(torch.nn.Module):
            b: Forward

            def __init__(self):
                super().__init__()
                self.b = B()

            # 实现 forward 方法，调用 self.b 的 forward 和 forwardError 方法
            def forward(self):
                self.b.forward(torch.ones(1))
                self.b.forwardError(torch.ones(1))

        # 用 TorchScript 脚本化 A 类的实例 a
        a = torch.jit.script(A())
        # 启用移动设备接口调用导出
        torch._C._enable_mobile_interface_call_export()
        # 将 TorchScript 对象 a 保存为字节流 buffer，并包含调试信息
        buffer = io.BytesIO(
            a._save_to_buffer_for_lite_interpreter(_save_mobile_debug_info=True)
        )
        buffer.seek(0)
        # 从 buffer 中加载用于轻量级解释器的模块 mobile_module
        mobile_module = _load_for_lite_interpreter(buffer)
        try:
            # 调用 mobile_module()，期望抛出 RuntimeError 异常
            mobile_module()
            # 如果没有抛出异常，则断言失败
            self.assertTrue(False)
        except RuntimeError as exp:
            # 使用 FileCheck 检查异常信息，包含特定的错误信息和调用栈位置标记
            FileCheck().check("Trying to create tensor with negative dimension").check(
                "Traceback of TorchScript"
            ).check("self.b.forwardError").check_next(
                "~~~~~~~~~~~~~~~~~~~ <--- HERE"
            ).check(
                "return self.call"
            ).check_next(
                "~~~~~~~~~ <--- HERE"
            ).check(
                "return torch.ones"
            ).check_next(
                "~~~~~~~~~~ <--- HERE"
            ).run(
                str(exp)
            )
class TestLiteScriptQuantizedModule(QuantizationLiteTestCase):
    # 定义测试类 TestLiteScriptQuantizedModule，继承自 QuantizationLiteTestCase
    def test_single_layer(self):
        # 测试单层模型量化的方法
        input = torch.rand(2, 5, dtype=torch.float)
        # 创建随机输入张量
        quantized_model = self._create_quantized_model(
            model_class=AnnotatedSingleLayerLinearModel, qengine="qnnpack"
        )
        # 创建被量化的单层线性模型，使用 qnnpack 引擎
        self._compare_script_and_mobile(model=quantized_model, input=input)
        # 调用方法，比较脚本模型与移动端模型的输出

    def test_two_layer(self):
        # 测试双层模型量化的方法
        input = torch.rand(2, 5, dtype=torch.float)
        # 创建随机输入张量
        quantized_model = self._create_quantized_model(model_class=TwoLayerLinearModel)
        # 创建被量化的双层线性模型
        self._compare_script_and_mobile(model=quantized_model, input=input)
        # 调用方法，比较脚本模型与移动端模型的输出

    def test_annotated_nested(self):
        # 测试嵌套模型量化的方法
        input = torch.rand(2, 5, dtype=torch.float)
        # 创建随机输入张量
        quantized_model = self._create_quantized_model(
            model_class=AnnotatedNestedModel, qengine="qnnpack"
        )
        # 创建被量化的带有注解的嵌套模型，使用 qnnpack 引擎
        self._compare_script_and_mobile(model=quantized_model, input=input)
        # 调用方法，比较脚本模型与移动端模型的输出

    def test_quantization_example(self):
        # 从 PyTorch 官方文档静态量化部分示例中提取的测试例子
        class M(torch.nn.Module):
            # 定义一个继承自 torch.nn.Module 的模型类 M
            def __init__(self):
                super().__init__()
                # 调用父类构造函数
                self.quant = torch.ao.quantization.QuantStub()
                # 添加量化操作的存根
                self.conv = torch.nn.Conv2d(1, 1, 1)
                # 添加卷积层
                self.relu = torch.nn.ReLU()
                # 添加 ReLU 激活层
                self.dequant = torch.ao.quantization.DeQuantStub()
                # 添加反量化操作的存根

            def forward(self, x):
                # 定义模型的前向传播方法
                x = self.quant(x)
                # 对输入 x 进行量化
                x = self.conv(x)
                # 使用卷积层处理量化后的输入
                x = self.relu(x)
                # 使用 ReLU 激活函数处理卷积层的输出
                x = self.dequant(x)
                # 对输出进行反量化
                return x

        model_fp32 = M()
        # 创建一个模型对象 model_fp32
        model_fp32.eval()
        # 设置模型为评估模式
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        # 设置模型的量化配置为 qnnpack 引擎的默认配置
        model_fp32_fused = torch.ao.quantization.fuse_modules(
            model_fp32, [["conv", "relu"]]
        )
        # 融合模型中的卷积层和 ReLU 激活层
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)
        # 准备量化所需的模型
        input_fp32 = torch.randn(4, 1, 4, 4)
        # 创建一个 4x1x4x4 大小的随机输入张量
        model_fp32_prepared(input_fp32)
        # 对准备好的模型使用输入进行前向传播
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
        # 将准备好的模型转换为 INT8 类型

        input = torch.randn(4, 1, 4, 4)
        # 创建一个 4x1x4x4 大小的随机输入张量
        self._compare_script_and_mobile(model=model_int8, input=input)
        # 调用方法，比较脚本模型与移动端模型的输出
    # 定义一个继承自 torch.nn.Module 的模型类 Model
    class Model(torch.nn.Module):
        
        # 定义模型的前向传播方法，接受三个参数，每个参数是一个字典，键为整数，值为 torch.Tensor
        def forward(
            self,
            x: Dict[int, torch.Tensor],
            y: Dict[int, torch.Tensor],
            z: Dict[int, torch.Tensor],
        ):
            # 返回参数 x，即输入的第一个字典
            return x

    # 创建 Model 类的实例
    model = Model()

    # 使用 torch.jit.script 方法对模型进行脚本化，转换为 TorchScript 模块
    script_module = torch.jit.script(model)

    # 构建一个示例输入 sample_input，这里使用了 script_module.forward 作为键，并将一个元组作为其值
    sample_input = {
        script_module.forward: [
            (
                {0: torch.ones(1)},  # 第一个参数 x，包含键为 0 的 torch.Tensor
                {1: torch.ones(1)},  # 第二个参数 y，包含键为 1 的 torch.Tensor
                {2: torch.ones(1)},  # 第三个参数 z，包含键为 2 的 torch.Tensor
            )
        ]
    }

    # 使用 torch.utils.bundled_inputs.bundle_inputs 方法打包 TorchScript 模块和示例输入
    bundled_model = torch.utils.bundled_inputs.bundle_inputs(
        script_module, sample_input
    )

    # 将打包好的模型保存到缓冲区 buf 中，为后续在 Lite 解释器中使用做准备
    buf = bundled_model._save_to_buffer_for_lite_interpreter()

    # 载入 Lite 解释器所需的模块，从缓冲区 buf 中加载
    mobile_module = _load_for_lite_interpreter(io.BytesIO(buf))

    # 运行 Lite 解释器中的方法 "get_all_bundled_inputs"，获取输入
    i = mobile_module.run_method("get_all_bundled_inputs")

    # 使用断言检查 Lite 解释器返回的第一个输入是否与预期的输入相同
    self.assertEqual(
        i[0],
        (
            {0: torch.ones(1)},  # 预期的第一个参数 x
            {1: torch.ones(1)},  # 预期的第二个参数 y
            {2: torch.ones(1)},  # 预期的第三个参数 z
        ),
    )
# 如果当前脚本作为主程序执行（而不是被导入到其他脚本中执行），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```