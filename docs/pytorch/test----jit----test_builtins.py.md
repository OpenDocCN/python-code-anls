# `.\pytorch\test\jit\test_builtins.py`

```
# Owner(s): ["oncall: jit"]

# 导入必要的模块和库
import inspect  # 导入inspect模块，用于获取对象信息
import os  # 导入os模块，提供与操作系统交互的功能
import sys  # 导入sys模块，提供对解释器相关的功能访问
import unittest  # 导入unittest模块，用于编写和运行单元测试
from typing import Dict, List  # 导入类型提示相关的模块

import torch  # 导入PyTorch库
from torch.testing import FileCheck  # 导入FileCheck用于测试框架

# Make the helper files in test/ importable
# 使test/目录中的辅助文件可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, RUN_CUDA  # 导入测试相关的工具类和变量

if __name__ == "__main__":
    # 如果作为主程序运行，抛出运行时错误，提示不直接运行此测试文件
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestBuiltins(JitTestCase):
    """
    Tests for TorchScript support of Python builtin functions.
    """

    def test_has_attr(self):
        class HasA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 0

        class HasB(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.b = 1

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建包含HasA和HasB对象的ModuleList
                self.mods = torch.nn.ModuleList([HasA(), HasB()])

            def forward(self):
                # 用列表l存储每个mod是否有属性a和b的结果
                l = torch.jit.annotate(List[int], [])
                for mod in self.mods:
                    l.append(int(hasattr(mod, "a")))
                    l.append(int(hasattr(mod, "b")))
                    # 如果有属性a，则将其值添加到列表l中
                    if hasattr(mod, "a"):
                        l.append(mod.a)
                    # 如果有属性b，则将其值添加到列表l中
                    if hasattr(mod, "b"):
                        l.append(mod.b)
                return l

        # 检查模型Mod的TorchScript支持
        self.checkModule(Mod(), ())

    def test_has_attr_invalid_args(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个torch.nn.Linear对象作为属性mod
                self.mod = torch.nn.Linear(1, 1)

            def forward(self, name):
                # 不允许使用动态的name参数，抛出运行时错误
                return hasattr(self.mod, name)

        # 测试使用torch.jit.script对Mod进行脚本化，期望抛出特定的运行时错误
        with self.assertRaisesRegexWithHighlight(RuntimeError, "hasattr", "name"):
            torch.jit.script(Mod())

        class Mod(torch.nn.Module):
            def forward(self, name):
                # 不允许使用torch.rand(2, 3)对象，因为它不是类类型，抛出运行时错误
                return hasattr(torch.rand(2, 3), name)

        # 测试使用torch.jit.script对Mod进行脚本化，期望抛出特定的运行时错误
        with self.assertRaisesRegexWithHighlight(RuntimeError, "hasattr", "name"):
            torch.jit.script(Mod())
    def test_del(self):
        # 定义一个函数 fn，参数 x 是一个整数列表，返回一个整数列表
        def fn(x: List[int]) -> List[int]:
            # 将列表 x 扩展为自身的两倍赋值给 a
            a = x * 2
            # 删除变量 a
            del a
            # 返回参数 x 本身
            return x

        # 使用自定义的检查函数 checkScript 测试函数 fn，传入参数 ([1, 2, 3],)
        self.checkScript(fn, ([1, 2, 3],))

        # 测试使用 Torch 的脚本化装饰器，期望抛出 RuntimeError 异常并包含 "undefined value" 和 "a"
        with self.assertRaisesRegexWithHighlight(RuntimeError, "undefined value", "a"):

            @torch.jit.script
            def fn(x):
                # 计算 x 的平方赋值给 a
                a = x**2
                # 删除变量 a
                del a
                # 返回变量 a（这里会引发异常因为 a 已经被删除）
                return a  # noqa: F821

        # 测试使用 Torch 的脚本化装饰器，期望抛出 RuntimeError 异常并包含 "undefined value" 和 "a"
        with self.assertRaisesRegexWithHighlight(RuntimeError, "undefined value", "a"):

            @torch.jit.script
            def fn(x):
                # 计算 x 的平方赋值给 a
                a = x**2
                # 如果 a 存在，则删除变量 a
                if a:
                    del a
                # 返回变量 a（这里会引发异常因为 a 已经被删除）
                return a

        # 测试使用 Torch 的脚本化装饰器，期望抛出 RuntimeError 异常并包含 "undefined value" 和 "b"
        with self.assertRaisesRegexWithHighlight(RuntimeError, "undefined value", "b"):

            @torch.jit.script
            def fn(x):
                # 计算 x 的平方赋值给 a
                a = x**2
                # 删除变量 b（这里会引发异常因为 b 未定义）
                del b  # noqa: F821
                # 返回变量 a
                return a

    def test_del_multiple_operands(self):
        # 定义一个函数 fn，参数 x 是一个整数列表，返回一个整数列表
        def fn(x: List[int]) -> List[int]:
            # 分别取列表 x 的前三个元素分别赋值给变量 a, b, c
            a, b, c = x[0], x[1], x[2]
            # 删除变量 a, b, c
            del a, b, c
            # 返回参数 x 本身
            return x

        # 使用自定义的检查函数 checkScript 测试函数 fn，传入参数 ([1, 2, 3],)
        self.checkScript(fn, ([1, 2, 3],))

        # 定义一个函数，用来删除整数列表的前两个元素
        def del_list_multiple_operands(x: List[int]) -> List[int]:
            # 删除列表 x 的第一个和第二个元素
            del x[0], x[1]
            # 返回删除元素后的列表 x
            return x

        # 测试 Python 环境下 del_list_multiple_operands 函数
        py_out = del_list_multiple_operands([0, 1, 2])
        # 使用 Torch 的脚本化装饰器测试 del_list_multiple_operands 函数
        jit_out = torch.jit.script(del_list_multiple_operands)([0, 1, 2])
        # 断言两者输出相等
        self.assertEqual(py_out, jit_out)

        # 定义一个函数，用来删除字典的两个键值对
        def del_dict_multiple_operands(x: Dict[str, int]) -> Dict[str, int]:
            # 删除字典 x 中键为 "hi" 和 "there" 的键值对
            del x["hi"], x["there"]
            # 返回删除键值对后的字典 x
            return x

        # 测试 Python 环境下 del_dict_multiple_operands 函数
        py_out = del_dict_multiple_operands({"hi": 5, "there": 6})
        # 使用 Torch 的脚本化装饰器测试 del_dict_multiple_operands 函数
        jit_out = torch.jit.script(del_dict_multiple_operands)({"hi": 5, "there": 6})
        # 断言两者输出相等
        self.assertEqual(py_out, jit_out)
class TestTensorBuiltins(JitTestCase):
    # TestTensorBuiltins 类，继承自 JitTestCase，用于测试张量的内置函数和特性

    def test_tensor_properties(self):
        # 测试张量的属性

        def should_keep(tensor, name):
            # 判断是否应该保留该属性的函数
            if inspect.isroutine(getattr(tensor, name)):
                return False
            # 如果属性是一个方法，则不保留
            if name.startswith("_"):
                return False
            # 如果属性名以 "_" 开头，则不保留
            return True
            # 否则保留该属性

        tensor = torch.arange(4, dtype=torch.float).view(2, 2)
        # 创建一个形状为 (2, 2)，数据类型为 float 的张量 tensor

        keys = dir(tensor)
        # 获取张量 tensor 的所有属性名列表

        # real and imag are only implemented for complex tensors.
        self.assertRaises(RuntimeError, lambda: should_keep(tensor, "imag"))
        # 断言调用 should_keep 函数时会引发 RuntimeError 异常，因为 imag 属性仅适用于复数张量
        keys.remove("imag")
        # 移除属性列表中的 "imag" 属性名

        properties = [p for p in keys if should_keep(tensor, p)]
        # 根据 should_keep 函数过滤出应该保留的属性列表

        code_template = """
        def fn(x):
            return x.{}
        """
        # 定义一个代码模板，用于生成函数，获取张量的特定属性值

        EQUALITY_MISMATCH = {
            # TorchScript doesn't have real enums so they return an int instead
            # of the actual value
            "dtype",
            "layout",
        }
        # 定义一个集合，包含 TorchScript 中类型不匹配的属性名

        MISSING_PROPERTIES = {
            "grad_fn",
            # This is an undocumented property so it's not included
            "output_nr",
            # This has a longer implementation, maybe not worth copying to
            # TorchScript if named tensors don't work there anyways
            "names",
        }
        # 定义一个集合，包含未包括在 TorchScript 中的属性名

        for p in properties:
            # 遍历所有应该保留的属性名

            if p in MISSING_PROPERTIES:
                continue
            # 如果属性在 MISSING_PROPERTIES 集合中，则跳过不处理

            code = code_template.format(p)
            # 使用代码模板生成获取属性值的代码

            cu = torch.jit.CompilationUnit()
            # 创建一个 TorchScript 的编译单元对象

            cu.define(code)
            # 定义生成的代码

            if p in EQUALITY_MISMATCH:
                continue
            # 如果属性在 EQUALITY_MISMATCH 集合中，则跳过不处理

            self.assertEqual(getattr(tensor, p), cu.fn(tensor))
            # 断言生成的 TorchScript 函数获取的属性值与原始张量的属性值相等

    def test_tensor_subscript_assign(self):
        # 测试张量的下标赋值操作

        def fn1(x):
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[torch.tensor(0)] = torch.tensor(2, dtype=torch.uint8)
            return a
            # 创建一个张量 a，其形状与 x 相同，数据类型为 uint8，将索引为 0 的位置赋值为 2

        def fn2(x):
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[0] = 2
            return a
            # 创建一个张量 a，其形状与 x 相同，数据类型为 uint8，将索引为 0 的位置赋值为 2

        def fn3(x):
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[torch.tensor(0)] = 2
            return a
            # 创建一个张量 a，其形状与 x 相同，数据类型为 uint8，将索引为 0 的位置赋值为 2

        def fn4(x):
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[0] = torch.tensor(2, dtype=torch.uint8)
            return a
            # 创建一个张量 a，其形状与 x 相同，数据类型为 uint8，将索引为 0 的位置赋值为 2

        def fn5(x):
            a = torch.zeros_like(x, dtype=torch.float32)
            a[torch.tensor(0)] = 2
            return a
            # 创建一个张量 a，其形状与 x 相同，数据类型为 float32，将索引为 0 的位置赋值为 2

        for fn in (fn1, fn2, fn3, fn4, fn5):
            self.checkScript(fn, (torch.zeros(2, dtype=torch.uint8),))
            # 对每个函数 fn 进行 TorchScript 的检查，传入一个形状为 (2,)，数据类型为 uint8 的零张量作为参数

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_tensor_subscript_assign_device(self):
        # 测试在 CUDA 设备上进行张量的下标赋值操作

        def fn6(x):
            a = torch.zeros_like(x, dtype=torch.float32, device="cuda")
            a[torch.tensor(0)] = 2
            return a
            # 创建一个张量 a，在 CUDA 设备上，其形状与 x 相同，数据类型为 float32，将索引为 0 的位置赋值为 2

        self.checkScript(fn6, (torch.zeros(2, dtype=torch.float32, device="cuda"),))
        # 对 fn6 函数进行 TorchScript 的检查，传入一个在 CUDA 设备上形状为 (2,)，数据类型为 float32 的零张量作为参数
    # 测试从张量到标量的转换
    def test_tensor_item(self):
        # 定义一个函数，测试标量转换
        def test_scalar_cast(x):
            # 将张量转换为标量
            scalar = x.item()
            return int(scalar), float(scalar)

        # 生成 test_scalar_cast 函数的图形表示
        graph = torch.jit.script(test_scalar_cast).graph
        # 检查图形中是否包含 "(int, float) = prim::TupleConstruct"，并运行检查
        FileCheck().check("(int, float) = prim::TupleConstruct").run(graph)
        # 检查 test_scalar_cast 函数的脚本化版本
        self.checkScript(test_scalar_cast, (torch.tensor(1.0),))
        self.checkScript(test_scalar_cast, (torch.tensor(1),))

    # 测试数字上的方法
    def test_method_on_number(self):
        # 定义一个函数，测试在数字上调用方法
        def func():
            c = 1
            return c.add(1)

        # 断言运行时错误中是否包含特定信息
        with self.assertRaisesRegex(RuntimeError, "object has no attribute or method"):
            # 将 func 函数脚本化
            torch.jit.script(func)

    # 测试将张量隐式转换为标量以匹配函数参数
    def test_scalar_to_num_conversions(self):
        # 定义一个函数，测试多个定义
        @torch.jit.script
        def multiple_defs(x):
            c = 1
            x = x + c
            return x

        # 断言图形中是否不包含 "ImplicitTensorToNum"
        self.assertTrue("ImplicitTensorToNum" not in str(multiple_defs.graph))

        # 定义一个函数，测试张量转换为整数
        @torch.jit.script
        def tensor_to_int_script(x, tensor):
            return x.unsqueeze(tensor)

        # 断言错误消息中是否包含特定信息
        with self.assertRaisesRegex(RuntimeError, "x.unsqueeze"):
            # 将 tensor_to_int_script 函数脚本化
            tensor_to_int_script(torch.tensor([2]), torch.tensor([2, 2]))

        # 定义一个函数，测试张量转换为浮点数
        def tensor_to_int(x, tensor):
            return x.unsqueeze(tensor)

        # 定义一个函数，测试张量转换为浮点数
        @torch.jit.script
        def tensor_to_float_script(x, tensor):
            return x.addcmul(tensor, tensor, value=tensor)

        # 定义一个函数，测试张量转换为浮点数
        def tensor_to_float(x, tensor):
            return x.addcmul(tensor, tensor, value=tensor)

        # 创建一个全零张量
        x = torch.zeros(10)
        # 不同类型的张量
        tensors = [
            torch.tensor(1.1),
            torch.tensor(1.1, requires_grad=True),
            torch.tensor(0),
            torch.tensor([2]),
        ]

        script_funs = [tensor_to_int_script, tensor_to_float_script]
        funs = [tensor_to_int, tensor_to_float]

        # 返回结果或异常是否被抛出
        def test_func(func, x, tensor):
            try:
                result = func(x, tensor)
            except RuntimeError as e:
                result = True
            except TypeError as e:
                result = True
            return result

        # 对每个（函数，输入）组合进行断言
        for tensor in tensors:
            for i in range(len(script_funs)):
                self.assertEqual(
                    test_func(script_funs[i], x, tensor), test_func(funs[i], x, tensor)
                )
```