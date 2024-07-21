# `.\pytorch\test\jit\test_types.py`

```py
# Owner(s): ["oncall: jit"]

# 导入所需的模块和库
import inspect  # 导入用于检查对象的模块
import os  # 导入操作系统相关功能的模块
import sys  # 导入系统相关功能的模块
from collections import namedtuple  # 导入命名元组相关功能
from textwrap import dedent  # 导入文本缩进处理的模块
from typing import Dict, Iterator, List, Optional, Tuple  # 导入类型提示相关功能

import torch  # 导入 PyTorch 深度学习库
import torch.testing._internal.jit_utils  # 导入 PyTorch 内部测试工具相关功能

from jit.test_module_interface import TestModuleInterface  # 导入自定义模块接口测试相关功能，禁止 Flake8 检查
from torch.testing import FileCheck  # 导入 PyTorch 测试文件检查工具

from torch.testing._internal.jit_utils import JitTestCase  # 导入 PyTorch JIT 测试用例基类

# Make the helper files in test/ importable
# 将 test/ 目录下的辅助文件添加到 Python 搜索路径中，使其可被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 如果当前文件被直接运行，抛出运行时错误并显示使用方法
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类 TestTypesAndAnnotation，继承自 JitTestCase
class TestTypesAndAnnotation(JitTestCase):

    # 定义测试方法 test_pep585_type，用于测试 PEP585 类型注解
    def test_pep585_type(self):
        # 内部函数 fn，接收一个 torch.Tensor 参数 x，返回一个包含 tuple 和 dict 的元组
        def fn(x: torch.Tensor) -> Tuple[Tuple[torch.Tensor], Dict[str, int]]:
            xl: list[tuple[torch.Tensor]] = []  # 定义列表 xl，存储元组
            xd: dict[str, int] = {}  # 定义字典 xd，存储字符串到整数的映射
            xl.append((x,))  # 将包含 x 的元组添加到列表 xl 中
            xd["foo"] = 1  # 在字典 xd 中添加键值对 "foo": 1
            return xl.pop(), xd  # 返回列表 xl 弹出的元组和字典 xd

        # 使用 self.checkScript 方法对 fn 进行脚本化测试
        self.checkScript(fn, [torch.randn(2, 2)])

        x = torch.randn(2, 2)  # 创建一个形状为 (2, 2) 的随机张量 x
        expected = fn(x)  # 调用 fn 函数得到期望结果
        scripted = torch.jit.script(fn)(x)  # 对 fn 进行脚本化并传入张量 x

        self.assertEqual(expected, scripted)  # 断言期望结果与脚本化结果相等

    # 定义测试方法 test_types_as_values，用于测试类型作为值的情况
    def test_types_as_values(self):
        # 内部函数 fn，接收一个 torch.Tensor 参数 m，返回该参数的设备信息
        def fn(m: torch.Tensor) -> torch.device:
            return m.device  # 返回张量 m 的设备信息

        # 使用 self.checkScript 方法对 fn 进行脚本化测试
        self.checkScript(fn, [torch.randn(2, 2)])

        # 定义命名元组 GG，包含字段 "f" 和 "g"
        GG = namedtuple("GG", ["f", "g"])

        # 定义类 Foo，继承自 torch.nn.Module
        class Foo(torch.nn.Module):
            # 忽略该方法的脚本化
            @torch.jit.ignore
            def foo(self, x: torch.Tensor, z: torch.Tensor) -> Tuple[GG, GG]:
                return GG(x, z), GG(x, z)

            # 前向传播方法，调用 foo 方法并返回结果
            def forward(self, x, z):
                return self.foo(x, z)

        foo = torch.jit.script(Foo())  # 对类 Foo 进行脚本化
        y = foo(torch.randn(2, 2), torch.randn(2, 2))  # 使用脚本化模型进行前向传播

        # 重新定义类 Foo，改变 foo 方法的返回类型
        class Foo(torch.nn.Module):
            # 忽略该方法的脚本化
            @torch.jit.ignore
            def foo(self, x, z) -> Tuple[GG, GG]:
                return GG(x, z)

            # 前向传播方法，调用 foo 方法并返回结果
            def forward(self, x, z):
                return self.foo(x, z)

        foo = torch.jit.script(Foo())  # 对类 Foo 进行脚本化
        y = foo(torch.randn(2, 2), torch.randn(2, 2))  # 使用脚本化模型进行前向传播
    def test_ignore_with_types(self):
        # 定义一个被忽略的 TorchScript 函数 fn，接受一个字典参数，值为可选的 torch.Tensor 对象
        @torch.jit.ignore
        def fn(x: Dict[str, Optional[torch.Tensor]]):
            return x + 10

        # 定义一个继承自 torch.nn.Module 的模型类 M
        class M(torch.nn.Module):
            # 前向传播方法，接受一个字典参数 in_batch，其值为可选的 torch.Tensor 对象，返回一个 torch.Tensor 对象
            def forward(
                self, in_batch: Dict[str, Optional[torch.Tensor]]
            ) -> torch.Tensor:
                # 调用模型内部的 dropout_modality 方法处理输入的批次数据
                self.dropout_modality(in_batch)
                # 调用被忽略的 TorchScript 函数 fn 处理输入的批次数据
                fn(in_batch)
                # 返回一个 torch.Tensor 对象
                return torch.tensor(1)

            # 定义一个被忽略的方法 dropout_modality，接受一个字典参数 in_batch，其值为可选的 torch.Tensor 对象，返回同样的字典
            @torch.jit.ignore
            def dropout_modality(
                self, in_batch: Dict[str, Optional[torch.Tensor]]
            ) -> Dict[str, Optional[torch.Tensor]]:
                return in_batch

        # 将模型 M 实例化并转换为 TorchScript
        sm = torch.jit.script(M())
        # 使用 FileCheck 检查 TorchScript 中是否包含 "dropout_modality" 和 "in_batch" 字符串
        FileCheck().check("dropout_modality").check("in_batch").run(str(sm.graph))

    def test_python_callable(self):
        # 定义一个 Python 类 MyPythonClass
        class MyPythonClass:
            # 定义一个被忽略的 __call__ 方法，接受任意数量的参数，并返回第一个参数的类型字符串表示
            @torch.jit.ignore
            def __call__(self, *args) -> str:
                return str(type(args[0]))

        # 创建 MyPythonClass 的实例对象
        the_class = MyPythonClass()

        # 定义一个 TorchScript 函数 fn，接受一个参数 x，调用 the_class 实例对象对 x 进行处理
        @torch.jit.script
        def fn(x):
            return the_class(x)

        # 创建一个 torch.Tensor 对象 x
        x = torch.ones(2)
        # 使用断言验证 fn(x) 的输出结果与 the_class(x) 的输出结果相等
        self.assertEqual(fn(x), the_class(x))

    def test_bad_types(self):
        # 定义一个被忽略的 TorchScript 函数 fn，接受一个参数 my_arg，并返回 my_arg + 10
        @torch.jit.ignore
        def fn(my_arg):
            return my_arg + 10

        # 使用断言验证调用带有非法参数的 other_fn 函数时是否会引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "argument 'my_arg'"):
            # 定义一个 TorchScript 函数 other_fn，接受一个参数 x，并调用被忽略的函数 fn
            @torch.jit.script
            def other_fn(x):
                return fn("2")

    def test_type_annotate_py3(self):
        # 定义一个普通 Python 函数 fn，不接受任何参数，返回包含 List[int]、torch.Tensor 和 Optional[torch.Tensor] 类型的元组
        def fn():
            a: List[int] = []
            b: torch.Tensor = torch.ones(2, 2)
            c: Optional[torch.Tensor] = None
            d: Optional[torch.Tensor] = torch.ones(3, 4)
            for _ in range(10):
                a.append(4)
                c = torch.ones(2, 2)
                d = None
            return a, b, c, d

        # 使用 self.checkScript 方法验证 fn 函数是否能够成功转换为 TorchScript
        self.checkScript(fn, ())

        # 定义一个错误的类型注解的 Python 函数 wrong_type，返回一个包含非法类型注解的列表
        def wrong_type():
            wrong: List[int] = [0.5]
            return wrong

        # 使用断言验证调用 wrong_type 函数时是否会引发特定的 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "List type annotation"
            r" `List\[int\]` did not match the "
            "types of the given list elements",
        ):
            # 将 wrong_type 函数尝试转换为 TorchScript，应该会因类型不匹配而引发异常
            torch.jit.script(wrong_type)
    def test_optional_no_element_type_annotation(self):
        """
        Test that using an optional with no contained types produces an error.
        """

        # 定义一个函数 fn_with_comment，参数 x 应为 torch.Tensor 类型，返回类型为 Optional
        def fn_with_comment(x: torch.Tensor) -> Optional:
            return (x, x)

        # 定义一个函数 annotated_fn，参数 x 应为 torch.Tensor 类型，返回类型为 Optional
        def annotated_fn(x: torch.Tensor) -> Optional:
            return (x, x)

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，检查是否提示使用 Optional 时未指定包含类型
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Optional without a contained type"
        ):
            # 创建 torch.jit.CompilationUnit 实例 cu，并定义其中 fn_with_comment 函数的源码
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，检查是否提示使用 Optional 时未指定包含类型
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Optional without a contained type"
        ):
            # 创建 torch.jit.CompilationUnit 实例 cu，并定义其中 annotated_fn 函数的源码
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，检查是否提示使用 Optional 时未指定包含类型
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Optional without a contained type"
        ):
            # 将 fn_with_comment 函数编译为 TorchScript 代码
            torch.jit.script(fn_with_comment)

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，检查是否提示使用 Optional 时未指定包含类型
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Optional without a contained type"
        ):
            # 将 annotated_fn 函数编译为 TorchScript 代码
            torch.jit.script(annotated_fn)

    def test_tuple_no_element_type_annotation(self):
        """
        Test that using a tuple with no contained types produces an error.
        """

        # 定义一个函数 fn_with_comment，参数 x 应为 torch.Tensor 类型，返回类型为 Tuple
        def fn_with_comment(x: torch.Tensor) -> Tuple:
            return (x, x)

        # 定义一个函数 annotated_fn，参数 x 应为 torch.Tensor 类型，返回类型为 Tuple
        def annotated_fn(x: torch.Tensor) -> Tuple:
            return (x, x)

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，检查是否提示使用 Tuple 时未指定包含类型
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Tuple without a contained type"
        ):
            # 创建 torch.jit.CompilationUnit 实例 cu，并定义其中 fn_with_comment 函数的源码
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，检查是否提示使用 Tuple 时未指定包含类型
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Tuple without a contained type"
        ):
            # 创建 torch.jit.CompilationUnit 实例 cu，并定义其中 annotated_fn 函数的源码
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，检查是否提示使用 Tuple 时未指定包含类型
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Tuple without a contained type"
        ):
            # 将 fn_with_comment 函数编译为 TorchScript 代码
            torch.jit.script(fn_with_comment)

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，检查是否提示使用 Tuple 时未指定包含类型
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Tuple without a contained type"
        ):
            # 将 annotated_fn 函数编译为 TorchScript 代码
            torch.jit.script(annotated_fn)
    def test_ignoring_module_attributes(self):
        """
        Test that module attributes can be ignored.
        """

        # 定义一个继承自 torch.nn.Module 的子类 Sub
        class Sub(torch.nn.Module):
            # 定义 forward 方法，接受整数参数 a，返回整数
            def forward(self, a: int) -> int:
                return sum([a])

        # 定义一个继承自 torch.nn.Module 的 ModuleWithIgnoredAttr 类
        class ModuleWithIgnoredAttr(torch.nn.Module):
            # 声明需要被忽略的属性列表
            __jit_ignored_attributes__ = ["a", "sub"]

            # 初始化方法，接受两个整数参数 a 和 b
            def __init__(self, a: int, b: int):
                super().__init__()
                self.a = a
                self.b = b
                self.sub = Sub()  # 创建 Sub 类的实例

            # 前向传播方法，返回整数
            def forward(self) -> int:
                return self.b

            # 标记为忽略的函数，接受 self 作为参数，返回整数
            @torch.jit.ignore
            def ignored_fn(self) -> int:
                return self.sub.forward(self.a)

        # 创建 ModuleWithIgnoredAttr 类的实例 mod，传入参数 1 和 4
        mod = ModuleWithIgnoredAttr(1, 4)
        # 对 mod 进行脚本化
        scripted_mod = torch.jit.script(mod)
        # 断言脚本化后的 mod() 的返回值为 4
        self.assertEqual(scripted_mod(), 4)
        # 断言脚本化后的 mod.ignored_fn() 的返回值为 1
        self.assertEqual(scripted_mod.ignored_fn(), 1)

        # 测试在使用被忽略属性时的错误消息
        class ModuleUsesIgnoredAttr(torch.nn.Module):
            # 声明需要被忽略的属性列表
            __jit_ignored_attributes__ = ["a", "sub"]

            # 初始化方法，接受一个整数参数 a
            def __init__(self, a: int):
                super().__init__()
                self.a = a
                self.sub = Sub()  # 创建 Sub 类的实例

            # 前向传播方法，返回整数
            def forward(self) -> int:
                return self.sub(self.b)

        # 创建 ModuleUsesIgnoredAttr 类的实例 mod，传入参数 1
        mod = ModuleUsesIgnoredAttr(1)

        # 使用断言捕获 RuntimeError 异常，并检查错误消息中是否包含特定字符串
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"attribute was ignored during compilation", "self.sub"
        ):
            # 对 mod 进行脚本化，预期抛出异常
            scripted_mod = torch.jit.script(mod)

    def test_ignoring_fn_with_nonscriptable_types(self):
        # 定义一个名为 CFX 的类
        class CFX:
            # 初始化方法，接受一个 torch.Tensor 类型的列表作为参数
            def __init__(self, a: List[torch.Tensor]) -> None:
                self.a = a

            # 前向传播方法，接受一个 torch.Tensor 参数，返回一个 torch.Tensor
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sin(x)

            # 标记为 _drop 的方法，返回一个迭代器，但在脚本化时将被忽略
            @torch.jit._drop
            def __iter__(self) -> Iterator[torch.Tensor]:
                return iter(self.a)

            # 标记为 _drop 的方法，接受一个 torch.fx.Tracer 类型的参数 tracer，
            # 返回一个 torch.fx.node.Argument 对象，在脚本化时将被忽略
            @torch.jit._drop
            def __fx_create_arg__(
                self, tracer: torch.fx.Tracer
            ) -> torch.fx.node.Argument:
                # torch.fx 类不可脚本化，直接返回一个创建节点的操作
                return tracer.create_node(
                    "call_function",
                    CFX,
                    args=(tracer.create_arg(self.features),),
                    kwargs={},
                )

        # 对 CFX 类进行脚本化，尽管 __iter__ 和 __fx_create_arg__ 方法被标记为 _drop
        torch.jit.script(CFX)

    def test_unimported_type_resolution(self):
        # 验证从 Python 解析器回退到 C++ 解析器

        # 定义一个名为 fn 的函数，接受一个参数 x
        @torch.jit.script
        def fn(x):
            # 标注 x 的类型为 number，返回值类型为 number
            # 实际上 TorchScript 没有 number 类型，此处应为 int 或 float
            return x + 1

        # 使用 FileCheck 检查 fn.graph 中的 "Scalar"
        FileCheck().check("Scalar").run(fn.graph)

    def test_parser_bug(self):
        # 定义一个名为 parser_bug 的函数，接受一个 Optional 类型的 torch.Tensor 参数 o
        def parser_bug(o: Optional[torch.Tensor]):
            pass

    def test_mismatched_annotation(self):
        # 使用断言捕获 RuntimeError 异常，并检查错误消息中是否包含特定字符串
        with self.assertRaisesRegex(RuntimeError, "annotated with type"):

            # 定义一个名为 foo 的函数，使用 torch.jit.script 进行脚本化
            @torch.jit.script
            def foo():
                # 将 x 标注为 str 类型，但实际赋值为整数 4，类型不匹配
                x: str = 4
                return x
    def test_reannotate(self):
        # 断言捕获到 RuntimeError，并且错误信息包含 "declare and annotate"
        with self.assertRaisesRegex(RuntimeError, "declare and annotate"):
            
            # 使用 torch.jit.script 装饰器创建脚本化函数 foo
            @torch.jit.script
            def foo():
                x = 5
                # 如果条件成立
                if 1 == 1:
                    # 重新声明并注释变量 x 的类型为 Optional[int]
                    x: Optional[int] = 7

    def test_annotate_outside_init(self):
        # 出错信息字符串
        msg = "annotations on instance attributes must be declared in __init__"
        # 出错信息中需要高亮显示的部分
        highlight = "self.x: int"

        # 简单情况下的断言：捕获到 ValueError 异常，错误信息包含 msg，并高亮显示 highlight
        with self.assertRaisesRegexWithHighlight(ValueError, msg, highlight):

            # 使用 torch.jit.script 装饰器创建脚本化类 BadModule
            @torch.jit.script
            class BadModule:
                # 类初始化函数，参数 x 被注释为 int 类型
                def __init__(self, x: int):
                    # 实例属性 self.x 被赋值为 x
                    self.x = x

                # 方法 set，参数 val 被注释为 int 类型
                def set(self, val: int):
                    # 重新声明并注释实例属性 self.x 的类型为 int
                    self.x: int = val

        # 在循环中进行类型注释的断言
        with self.assertRaisesRegexWithHighlight(ValueError, msg, highlight):

            # 使用 torch.jit.script 装饰器创建脚本化类 BadModuleLoop
            @torch.jit.script
            class BadModuleLoop:
                # 类初始化函数，参数 x 被注释为 int 类型
                def __init__(self, x: int):
                    # 实例属性 self.x 被赋值为 x
                    self.x = x

                # 方法 set，参数 val 被注释为 int 类型
                def set(self, val: int):
                    # 在循环中，重新声明并注释实例属性 self.x 的类型为 int
                    for i in range(3):
                        self.x: int = val

        # 在 __init__ 方法中进行类型注释的情况，不应该出错
        @torch.jit.script
        class GoodModule:
            # 类初始化函数，参数 x 被注释为 int 类型，实例属性 self.x 也被注释为 int
            def __init__(self, x: int):
                self.x: int = x

            # 方法 set，参数 val 被注释为 int 类型，但不重新注释 self.x
            def set(self, val: int):
                self.x = val

    def test_inferred_type_error_message(self):
        # 创建一个 InferredType 对象，错误原因为 "ErrorReason"
        inferred_type = torch._C.InferredType("ErrorReason")

        # 断言捕获到 RuntimeError，并且错误信息包含 "Tried to get the type from an InferredType but the type is null."
        with self.assertRaisesRegex(
            RuntimeError,
            "Tried to get the type from an InferredType but the type is null.",
        ):
            # 尝试获取类型信息，但是类型为空时抛出异常
            t = inferred_type.type()

        # 断言捕获到 RuntimeError，并且错误信息包含 "ErrorReason"
        with self.assertRaisesRegex(RuntimeError, "ErrorReason"):
            # 直接访问错误类型信息时抛出异常
            t = inferred_type.type()
```