# `.\pytorch\test\jit\test_union.py`

```
# Owner(s): ["oncall: jit"]

# 导入所需的库和模块
import io  # 导入io库，用于处理文件流
import os  # 导入os库，用于操作系统相关功能
import sys  # 导入sys库，用于访问系统相关的参数和功能
from enum import Enum  # 导入Enum类，用于定义枚举类型
from textwrap import dedent  # 导入dedent函数，用于移除文本块开头的缩进
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示相关的类和装饰器

import torch  # 导入PyTorch库
from torch.testing import FileCheck  # 导入FileCheck类，用于检查文本中的模式

# Make the helper files in test/ importable
# 获取当前脚本的父目录，并将其添加到sys.path中，以便导入test/目录下的辅助文件
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, make_global  # 导入测试工具相关的函数和类

if __name__ == "__main__":
    # 如果脚本被直接运行，抛出异常并提示正确的运行方式
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

class TestUnion(JitTestCase):
    """
    This class tests the functionality of `Union`.

    Note: It's important to be able to refine the type of a `Union` to
    one of its internal types. Currently, there are differences in the
    way Python expects `isinstance` checks and the way TorchScript
    expects `isinstance` checks. This means that we can't use
    `checkScript` in our test cases because either the eager mode or the
    script mode wouldn't run! So, some test cases have separate but
    equivalent functions to emulate `checkScript`.
    """

    def test_check_union_annotation(self):
        # 定义一个测试函数，参数a为Union类型，可以是int或float；参数b为Optional类型，可以是int或None
        def test_func(a: Union[int, float], b: Optional[int]):
            return 0

        # 将测试函数编译为TorchScript脚本
        scripted_func = torch.jit.script(test_func)
        # 获取TorchScript图形的字符串表示形式
        graph_rep = str(scripted_func.graph)
        # 获取TorchScript代码的字符串表示形式
        code_rep = str(scripted_func.code)
        # 使用FileCheck检查TorchScript图形的IR，确保Union类型被正确标注
        FileCheck().check("Union(").check("int?").run(graph_rep)
        # 使用FileCheck检查TorchScript代码，确保Union类型被正确标注
        FileCheck().check("Union[").check("Optional[int]").run(code_rep)
        # 使用self.checkScript方法验证脚本化的函数行为符合预期
        self.checkScript(test_func, (5, 6))
        # 解析TorchScript图形的IR，确保不会出错
        torch._C.parse_ir(str(scripted_func.graph))

    def test_union_with_scalar_values(self):
        # 定义一个接受Union[int, float]参数并返回str的函数
        def fn(x: Union[int, float]) -> str:
            return "foo"

        # 使用self.checkScript方法验证脚本化的函数行为符合预期
        self.checkScript(fn, (1,))
        self.checkScript(fn, (1.0,))

        # 将函数编译为TorchScript脚本
        scripted = torch.jit.script(fn)

        # 使用断言确保调用脚本化函数时传入非预期类型会引发异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[float, int\] but "
            "instead found type str",
        ):
            scripted("1")
    # 定义测试函数，测试 Union 类型在函数参数中的使用
    def test_union_with_collections(self):
        # 定义函数 fn，接受 Union 类型的参数并返回字符串 "foo"
        def fn(x: Union[Dict[str, int], List[int]]) -> str:
            return "foo"

        # 检查 fn 函数在输入为字典 {"foo": 1, "bar": 2, "baz": 3} 时的脚本化结果
        self.checkScript(fn, ({"foo": 1, "bar": 2, "baz": 3},))
        # 检查 fn 函数在输入为列表 [1, 2, 3] 时的脚本化结果
        self.checkScript(fn, ([1, 2, 3],))

        # 对 fn 函数进行脚本化
        scripted = torch.jit.script(fn)

        # 断言脚本化函数调用中，当输入的字典值类型为字符串时，引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[List\[int\], Dict\[str, "
            r"int\]\] but instead found type "
            r"Dict\[str, str\]",
        ):
            scripted({"foo": "bar", "baz": "qux"})

        # 断言脚本化函数调用中，当输入的列表值类型为字符串时，引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[List\[int\], Dict\[str, "
            r"int\]\] but instead found type "
            r"List\[str\]",
        ):
            scripted(["foo", "bar", "baz"])

        # 断言脚本化函数调用中，当输入的值类型为字符串时，引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[List\[int\], Dict\[str, "
            r"int\]\] but instead found type "
            "str",
        ):
            scripted("1")

    # 测试 Union 类型与枚举类型的结合使用
    def test_union_with_enum(self):
        # 定义颜色枚举类 Color
        class Color(Enum):
            RED = 1
            GREEN = 2

        # 将 Color 类设为全局可用
        make_global(Color)

        # 定义函数 fn，接受 Union 类型的参数并返回字符串 "foo"
        def fn(x: Union[str, Color]) -> str:
            return "foo"

        # 检查 fn 函数在输入为 Color.RED 时的脚本化结果
        self.checkScript(fn, (Color.RED,))
        # 检查 fn 函数在输入为字符串 "red" 时的脚本化结果
        self.checkScript(fn, ("red",))

        # 对 fn 函数进行脚本化
        scripted = torch.jit.script(fn)

        # 断言脚本化函数调用中，当输入的值类型为整数时，引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[__torch__.jit.test_union."
            r"Color, str\] but instead found "
            "type int",
        ):
            scripted(1)

    # 测试 Union 类型在类构造函数中的使用
    def test_union_in_class_constructor(self):
        @torch.jit.script  # noqa: B903
        class A:  # noqa: B903
            # 定义类 A 的构造函数，接受 Union 类型的参数并无返回值
            def __init__(self, x: Union[int, str]) -> None:
                self.x = x

        # 定义函数 fn，接受 Union 类型的参数并返回 A 类型的实例
        def fn(x: Union[str, int]) -> A:
            return A(x)

        # 断言 fn 函数返回的 A 类型实例的属性 x 为字符串 "foo"
        self.assertEqual(fn("foo").x, "foo")
        # 断言 fn 函数返回的 A 类型实例的属性 x 为整数 1
        self.assertEqual(fn(1).x, 1)

        # 对 fn 函数进行脚本化
        scripted = torch.jit.script(fn)

        # 断言脚本化函数调用中，当输入的列表值类型为字符串时，引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[int, str\] but instead "
            r"found type List\[str\]",
        ):
            scripted(["foo", "bar", "baz"])

    # 测试 Union 类型作为函数返回类型的使用
    def test_union_return_type(self):
        # 定义函数 fn，接受整数参数并返回 Union 类型的值
        def fn(x: int) -> Union[int, str]:
            return "foo"

        # 检查 fn 函数在输入为整数 1 时的脚本化结果
        self.checkScript(fn, (1,))

    # 测试 Union 类型在变量注释中的使用
    def test_union_as_annotation(self):
        # 定义函数 fn，无参数，返回 Union 类型的值
        def fn() -> Union[int, str]:
            # 定义变量 x，类型为 Union[int, str]，赋值为字符串 "foo"
            x: Union[int, str] = "foo"
            return x

        # 检查 fn 函数在无参数输入时的脚本化结果
        self.checkScript(fn, ())

    # 测试 Union 类型在类型化容器中的使用
    def test_union_as_annotation_in_typed_container(self):
        # 定义函数 fn，无参数，无返回值
        def fn() -> None:
            # 定义列表 l，元素类型为 Union[int, str]
            l: List[Union[int, str]] = []
            # 定义变量 u1，类型为 Union[int, str]，赋值为字符串 "foo"
            u1: Union[int, str] = "foo"
            # 定义变量 u2，类型为 Union[int, str]，赋值为整数 1
            u2: Union[int, str] = 1
            # 向列表 l 中添加元素 u1 和 u2
            l.append(u1)
            l.append(u2)

        # 检查 fn 函数在无参数输入时的脚本化结果
        self.checkScript(fn, ())
    # 定义一个测试方法，用于测试将 Union 类型作为注解在 Python 2 中的行为
    def test_union_as_annotation_py2(self):
        # 定义一个内部函数 fn
        def fn():
            # type 注解表明 x 可以是 int 或 str 类型的 Union
            x: Union[int, str] = "foo"
            return x
        
        # 调用父类的 checkScript 方法来检查 fn 的脚本化结果
        self.checkScript(fn, ())

    # 定义一个测试方法，用于测试将 Union 类型作为内部元组类型的行为
    def test_union_as_internal_tuple_type(self):
        # 定义一个内部函数 fn
        def fn():
            # t 的类型是一个包含两个元素的元组，每个元素可以是 int 或 str 类型的 Union
            t: Tuple[Union[int, str], Union[int, str]] = (1, "foo")
            return t
        
        # 调用父类的 checkScript 方法来检查 fn 的脚本化结果
        self.checkScript(fn, ())

    # 定义一个测试方法，用于测试 Union 变量可以被重新赋值的行为
    def test_union_variable_can_be_reassigned(self):
        # 定义一个 Torch Script 函数 aux1，接受一个 int 类型参数并返回其平方
        @torch.jit.script
        def aux1(i: int):
            return int(i**2)

        # 定义一个 Torch Script 函数 aux2，接受一个 str 类型参数并返回其重复两次的结果
        @torch.jit.script
        def aux2(s: str):
            return s + s

        # 定义一个函数 fn，返回类型是 int 或 str 的 Union
        def fn() -> Union[int, str]:
            # 初始化 x 为 str 类型
            x: Union[int, str] = "foo"
            # 初始化 i 为 int 类型
            i: int = 1
            # 将 x 重新赋值为 i，此时 x 是 int 类型
            x = i
            # 调用 aux1 函数处理 x 的值，y 的类型为 int
            y: int = aux1(x)
            # 将 y 转换为 str，并传递给 aux2 函数处理，z 的类型为 str
            z: str = aux2(str(y))
            # 将 z 赋值给 x，此时 x 是 str 类型
            x = z
            return x
        
        # 调用父类的 checkScript 方法来检查 fn 的脚本化结果
        self.checkScript(fn, ())

    # 定义一个测试方法，用于测试 Union 类型不会替换已有的注解类型的行为
    def test_union_does_not_replace_existing_annotated_type(self):
        # 定义一个函数 fn，其中 x 被注解为 List[int] 类型
        def fn():
            x: List[int] = [1, 2, 3]
            # 向 x 中添加一个 str 类型的元素，预期会抛出 RuntimeError 异常
            x.append("foo")
            return x
        
        # 使用 assertRaisesRegex 来断言 RuntimeError 异常中包含 "Could not match type str"
        with self.assertRaisesRegex(RuntimeError, "Could not match type str"):
            # 对 fn 进行 Torch Script 脚本化，并执行
            scripted = torch.jit.script(fn)
            scripted()

    # 定义一个测试方法，用于测试 Union 类型不会替换已有的注解类型的行为（包含 Union[int, str]）
    def test_union_does_not_replace_existing_annotated_type_union(self):
        # 定义一个函数 fn，其中 x 被注解为 List[Union[int, str]] 类型
        def fn():
            x: List[Union[int, str]] = [1, "foo", 3]
            # 向 x 中添加一个 float 类型的元素，预期会抛出 RuntimeError 异常
            x.append(2.0)
            return x
        
        # 使用 assertRaisesRegex 来断言 RuntimeError 异常中包含 "Could not match type float"
        with self.assertRaisesRegex(RuntimeError, "Could not match type float"):
            # 对 fn 进行 Torch Script 脚本化，并执行
            scripted = torch.jit.script(fn)
            scripted()

    # 定义一个测试方法，用于测试 Union 类型不会替换已有的注解类型的行为（空的 List[int]）
    def test_union_does_not_replace_existing_annotated_type_empty_container(self):
        # 定义一个函数 fn，其中 x 被注解为 List[int] 类型
        def fn():
            x: List[int] = []
            # 向 x 中添加一个 str 类型的元素，预期会抛出 RuntimeError 异常
            x.append("foo")
            return x
        
        # 使用 assertRaisesRegex 来断言 RuntimeError 异常中包含 "Could not match type str"
        with self.assertRaisesRegex(RuntimeError, "Could not match type str"):
            # 对 fn 进行 Torch Script 脚本化，并执行
            scripted = torch.jit.script(fn)
            scripted()

    # 定义一个测试方法，用于测试 Union 的嵌套行为
    def test_unions_of_unions_are_flattened(self):
        # 定义一个 Torch Script 函数 fn，接受一个 Union[Union[int, str], float] 参数，并返回 str 类型结果
        @torch.jit.script
        def fn(x: Union[Union[int, str], float]) -> str:
            return "foo"
        
        # 获取 fn 的图形表示
        s = fn.graph
        
        # 使用 FileCheck 验证图形中的类型信息，检查是否展开了 Union 的嵌套
        FileCheck().check("x : Union(float, int, str)").run(s)

    # 定义一个测试方法，用于测试 Union 类型的单一参数行为
    def test_unions_of_a_single_argument_vanish(self):
        # 定义一个 Torch Script 函数 fn，接受一个 Union[int] 参数，并返回 str 类型结果
        @torch.jit.script
        def fn(x: Union[int]) -> str:
            return "foo"
        
        # 获取 fn 的图形表示
        s = fn.graph
        
        # 使用 FileCheck 验证图形中的类型信息，检查是否只保留了 int 类型
        FileCheck().check("x : int").run(s)

    # 定义一个测试方法，用于测试 Union 类型中的冗余参数被省略的行为
    def test_union_redundant_arguments_are_skipped(self):
        # 定义一个 Torch Script 函数 fn，接受一个 Union[int, str, int] 参数，并返回 str 类型结果
        @torch.jit.script
        def fn(x: Union[int, str, int]) -> str:
            return "foo"
        
        # 获取 fn 的图形表示
        s = fn.graph
        
        # 使用 FileCheck 验证图形中的类型信息，检查是否省略了冗余的 int 类型
        FileCheck().check("x : Union(int, str)").run(s)

    # 定义一个测试方法，用于测试 Union 类型中的冗余参数（包含 Optional 类型）被省略的行为
    def test_union_redundant_arguments_are_skipped_optional(self):
        # 定义一个 Torch Script 函数 fn，接受一个 Union[int, Optional[float], Optional[int]] 参数，并返回 str 类型结果
        @torch.jit.script
        def fn(x: Union[int, Optional[float], Optional[int]]) -> str:
            return "foo"
        
        # 获取 fn 的图形表示
        s = fn.graph
        
        # 使用 FileCheck 验证图形中的类型信息，检查是否省略了冗余的 int 类型和 NoneType
        FileCheck().check("x : Union(float, int, NoneType)").run(s)
    # 定义一个测试函数，用于验证当子类型存在冗余参数时会被忽略
    def test_union_redundant_arguments_are_skipped_subtyping(self):
        # 使用 torch.jit.script 装饰器将函数 fn 转换为脚本
        @torch.jit.script
        def fn(x: Union[str, Tuple[Optional[int], int], Tuple[int, int]]) -> str:
            return "foo"

        # 获取函数 fn 的计算图
        s = fn.graph

        # 使用 FileCheck() 对计算图 s 进行检查，验证是否存在预期的类型注释
        FileCheck().check("x : Union((int?, int), str)").run(s)

    # 定义一个测试函数，用于验证容器类型中存在冗余参数时会被忽略
    def test_union_redundant_arguments_are_skipped_container(self):
        # 使用 torch.jit.script 装饰器将函数 fn 转换为脚本
        @torch.jit.script
        def fn(x: Union[List[str], List[float], List[str]]) -> str:
            return "foo"

        # 获取函数 fn 的计算图
        s = fn.graph

        # 使用 FileCheck() 对计算图 s 进行检查，验证是否存在预期的类型注释
        FileCheck().check("x : Union(float[], str[])").run(s)

    # 定义一个测试函数，用于验证联合类型中参数顺序不影响类型推断
    def test_union_argument_order_is_ignored(self):
        # 使用 torch.jit.script 装饰器将函数 fn1 转换为脚本
        @torch.jit.script
        def fn1(x: Union[int, str]) -> str:
            return "foo"

        # 使用 torch.jit.script 装饰器将函数 fn2 转换为脚本
        @torch.jit.script
        def fn2(x: Union[str, int]) -> str:
            return "foo"

        # 遍历 fn1 和 fn2 的计算图
        for s in (fn1.graph, fn2.graph):
            # 使用 FileCheck() 对计算图 s 进行检查，验证是否存在预期的类型注释
            FileCheck().check("x : Union(int, str)").run(s)

    # 定义一个测试函数，用于验证容器类型中联合类型参数顺序不影响类型推断
    def test_union_argument_order_is_ignored_container(self):
        # 使用 torch.jit.script 装饰器将函数 fn1 转换为脚本
        @torch.jit.script
        def fn1(x: Union[List[str], List[int]]) -> str:
            return "foo"

        # 使用 torch.jit.script 装饰器将函数 fn2 转换为脚本
        @torch.jit.script
        def fn2(x: Union[List[int], List[str]]) -> str:
            return "foo"

        # 遍历 fn1 和 fn2 的计算图
        for s in (fn1.graph, fn2.graph):
            # 使用 FileCheck() 对计算图 s 进行检查，验证是否存在预期的类型注释
            FileCheck().check("x : Union(int[], str[])").run(s)

    # 定义一个测试函数，用于验证 Union[T, None] 等同于 Optional[T]
    def test_union_T_None_is_equivalent_to_optional_T(self):
        # 使用 torch.jit.script 装饰器将函数 inner 转换为脚本
        @torch.jit.script
        def inner(x: Union[int, None]) -> int:
            # 如果 x 不为 None，则返回 x；否则返回 5
            if x is not None:
                return x
            else:
                return 5

        # 使用 torch.jit.script 装饰器将函数 fn1 转换为脚本
        @torch.jit.script
        def fn1() -> int:
            # 声明一个可选类型的整数变量 a 和 b
            a: Optional[int] = 5
            b: Optional[int] = None
            # 调用 inner 函数获取 a 和 b 的返回值，并返回它们的和
            a_ = inner(a)
            b_ = inner(b)
            return a_ + b_

        # 断言 fn1 的返回值等于 10
        self.assertEqual(fn1(), 10)

        # 使用 torch.jit.script 装饰器将函数 inner2 转换为脚本
        @torch.jit.script
        def inner2(x: Optional[int]) -> int:
            # 如果 x 不为 None，则返回 x；否则返回 5
            if x is not None:
                return x
            else:
                return 5

        # 使用 torch.jit.script 装饰器将函数 fn2 转换为脚本
        @torch.jit.script
        def fn2() -> int:
            # 声明一个联合类型的整数变量 a 和 b
            a: Union[int, None] = 5
            b: Union[int, None] = None
            # 调用 inner2 函数获取 a 和 b 的返回值，并返回它们的和
            a_ = inner(a)
            b_ = inner(b)
            return a_ + b_

        # 断言 fn2 的返回值等于 10
        self.assertEqual(fn2(), 10)
    def test_union_optional_of_union_is_flattened(self):
        # 定义一个 TorchScript 函数，接受一个整数参数，返回一个 Union 类型的值（可以是 str, int, None 中的一种）
        @torch.jit.script
        def fn(flag: int) -> Union[str, int, None]:
            # 定义一个 Union 类型的变量 y，初始为字符串 "foo"
            y: Union[int, str, None] = "foo"
            # 根据 flag 的不同取值，确定 x 的类型
            if flag == 0:
                # 如果 flag 为 0，则 x 可能是 int 或 str 类型，取决于 y 的实际类型
                x: Optional[Union[int, str]] = y
            elif flag == 1:
                # 如果 flag 为 1，则 x 是 int 类型
                x: Optional[Union[int, str]] = 1
            else:
                # 否则，flag 不为 0 或 1，则 x 是 NoneType 类型
                x: Optional[Union[int, str]] = None
            # 返回 x
            return x

        # 测试 TorchScript 函数的返回值是否符合预期
        self.assertEqual(fn(0), "foo")
        self.assertEqual(fn(1), 1)
        self.assertEqual(fn(2), None)

        # 将 TorchScript 函数保存到字节流中
        buffer = io.BytesIO()
        torch.jit.save(fn, buffer)
        buffer = io.BytesIO(buffer.getvalue())
        # 从字节流中加载 TorchScript 函数
        l = torch.jit.load(buffer)

        # 获取加载后的 TorchScript 函数的代码
        s = l.code

        # 使用 FileCheck 检查代码中的类型声明是否为 "Union[int, NoneType, str]"
        FileCheck().check("Union[int, NoneType, str]").check(
            "Union[int, NoneType, str]"
        ).run(s)

    def test_union_subclasses_larger_union(self):
        # 定义一个函数，返回一个 Union 类型的值（可以是 int, str, torch.Tensor 中的一种）
        def fn() -> Union[int, str, torch.Tensor]:
            # 定义一个 Union 类型的变量 x，初始为字符串 "foo"
            x: Union[int, str] = "foo"
            # 返回 x
            return x

        # 检查脚本化版本的函数是否符合预期
        self.checkScript(fn, ())

    # TODO: We would like to eventually support this. The issue is being
    # tracked at https://github.com/pytorch/pytorch/issues/58167
    def test_union_as_dict_key(self):
        # 定义一个函数，演示使用 Union 类型作为字典的键
        def fn():
            # 定义一个字典 x，其键为 Union[int, str] 类型，初始为空字典
            x: Dict[Union[int, str], str] = {}
            # 添加键为 "foo"，值为 "bar" 的项到字典中
            x["foo"] = "bar"
            # 添加键为 1，值为 2 的项到字典中
            x[1] = 2
            # 返回字典中键为 1 的值
            return x[1]

        # 检查脚本化版本的函数是否抛出预期的运行时异常
        with self.assertRaisesRegex(
            RuntimeError,
            "only int, float, "
            "complex, Tensor, device and string keys "
            "are supported",
        ):
            # 尝试将函数脚本化
            torch.jit.script(fn)

    def test_union_as_dict_value(self):
        # 定义一个函数，演示使用 Union 类型作为字典的值
        def fn():
            # 定义一个字典 x，其值为 Union[int, str] 类型，初始为空字典
            x: Dict[str, Union[int, str]] = {}
            # 添加键为 "foo"，值为 "bar" 的项到字典中
            x["foo"] = "bar"
            # 添加键为 "baz"，值为 2 的项到字典中
            x["baz"] = 2
            # 返回字典中键为 "baz" 的值
            return x["baz"]

        # 检查脚本化版本的函数是否符合预期
        self.checkScript(fn, ())

    def test_union_module_with_union_instance_variable(self):
        # 定义一个继承自 torch.nn.Module 的类 M，演示在类中使用 Union 类型作为实例变量
        class M(torch.nn.Module):
            # 定义一个 Union 类型的实例变量 x
            x: Union[int, str]

            def __init__(self, x: Union[int, str]):
                super().__init__()
                # 初始化实例变量 x
                self.x: Union[int, str] = x

            def forward(self, y: Union[int, str]):
                # 将参数 y 赋值给实例变量 x，并返回 x
                self.x = y
                return self.x

        # 检查 M 类的模块化版本是否符合预期
        self.checkModule(
            M(
                2,
            ),
            (1,),
        )
        self.checkModule(M("bar"), ("foo",))

    def test_union_module_with_union_class_variable(self):
        # 定义一个继承自 torch.nn.Module 的类 M，演示在类中使用 Union 类型作为类变量
        class M(torch.nn.Module):
            # 定义一个 Union 类型的类变量 x，初始为字符串 "foo"
            x: Union[int, str] = "foo"

            def __init__(self, y: int):
                super().__init__()
                # 初始化局部变量 x
                x = y

            def forward(self, z: str):
                # 将参数 z 赋值给局部变量 x，并返回 x
                x = z
                return x

        # 检查 M 类的模块化版本是否符合预期
        self.checkModule(M(1), ("foo",))
    # 定义一个测试函数，用于测试联合类型的细化
    def test_union_type_refinement(self):
        # 定义一个函数 fn，接受一个联合类型的参数 x，返回一个字符串
        def fn(x: Union[int, str]) -> str:
            # 如果 x 是字符串类型，将 x 后面添加 "bar" 并返回 x 本身
            if isinstance(x, str):
                z = x + "bar"
                return x
            else:
                # 如果 x 不是字符串类型，返回字符串 "baz"
                return "baz"

        # 使用 self.checkScript 方法检查函数 fn 在参数 ("foo",) 上的执行结果
        self.checkScript(fn, ("foo",))
        # 使用 self.checkScript 方法检查函数 fn 在参数 (1,) 上的执行结果
        self.checkScript(fn, (1,))

    # 定义另一个测试函数，测试联合类型细化中的联合右侧类型
    def test_union_type_refinement_union_rhs(self):
        # 定义一个函数 fn，接受一个整数参数 x，返回一个字符串
        def fn(x: int) -> str:
            # 如果 torch.jit.isinstance 判断 x 是 Union[int, str] 类型
            if torch.jit.isinstance(x, Union[int, str]):
                # 返回字符串 "bar"
                return "bar"
            else:
                # 否则返回字符串 "baz"
                return "baz"

        # 使用 self.checkScript 方法检查函数 fn 在参数 (1,) 上的执行结果
        self.checkScript(fn, (1,))

    # 定义测试函数，测试联合类型细化中的元组右侧类型
    def test_union_type_refinement_tuple_rhs(self):
        # 定义一个函数 fn，接受一个联合类型的参数 x，返回一个字符串
        def fn(x: Union[int, float, List[str]]) -> str:
            # 如果 x 是 int 或者 float 类型
            if isinstance(x, (int, float)):
                # 如果 x 是 int 类型，将其转换为字符串并返回
                if isinstance(x, int):
                    return str(x)
                else:
                    # 如果 x 是 float 类型，返回字符串 "foo"
                    return "foo"
            else:
                # 如果 x 是 List[str] 类型且长度不为 0，返回第一个元素
                if len(x):
                    return x[0]
                else:
                    # 否则返回字符串 "bar"
                    return "bar"

        # 使用 self.checkScript 方法检查函数 fn 在参数 (1,) 上的执行结果
        self.checkScript(fn, (1,))
        # 使用 self.checkScript 方法检查函数 fn 在参数 (1.0,) 上的执行结果
        self.checkScript(fn, (1.0,))
        # 使用 self.checkScript 方法检查函数 fn 在参数 (["a", "b", "c"],) 上的执行结果
        self.checkScript(fn, (["a", "b", "c"],))

    # 定义测试函数，测试联合类型细化中的元组右侧类型，其中包含非包含的类型
    def test_union_type_refinement_tuple_rhs_noncontained_type(self):
        # 定义一个函数 fn，接受一个联合类型的参数 x，返回一个字符串
        def fn(x: Union[int, List[str]]) -> str:
            # 如果 x 是 int 或 float 类型
            if isinstance(x, (int, float)):
                # 计算 y = x + x，将结果转换为字符串并返回
                y = x + x
                return str(y)
            else:
                # 如果 x 是 List[str] 类型且长度不为 0，返回第一个元素
                if len(x):
                    return x[0]
                else:
                    # 否则返回字符串 "bar"
                    return "bar"

        # 使用 self.checkScript 方法检查函数 fn 在参数 (1,) 上的执行结果
        self.checkScript(fn, (1,))
        # 使用 self.checkScript 方法检查函数 fn 在参数 (["a", "b", "c"],) 上的执行结果
        self.checkScript(fn, (["a", "b", "c"],))

    # 定义测试函数，测试联合类型细化中的元组右侧联合
    def test_union_type_refinement_tuple_rhs_union(self):
        # 使用 torch.jit.script 装饰器定义一个函数 fn，接受一个整数参数 x，返回一个字符串
        @torch.jit.script
        def fn(x: int) -> str:
            # 如果 torch.jit.isinstance 判断 x 是 Union[int, str] 或 float 类型
            if torch.jit.isinstance(x, (Union[int, str], float)):
                # 计算 y = x + x，将结果转换为字符串并返回
                y = x + x
                return str(y)
            else:
                # 否则返回字符串 "foo"
                return "foo"

        # TODO: There's currently an unrelated bug in
        # `torch.jit.isinstance` that makes it fail for tuple literals.
        # Posted here: https://github.com/pytorch/pytorch/issues/60095
        # Change `assertEqual` to `checkScript` when the bug is fixed
        # 使用 assertEqual 方法验证 fn(1) 的执行结果为 "2"
        self.assertEqual(fn(1), "2")

    # 定义测试函数，测试静态 false 联合类型细化
    def test_union_type_refinement_statically_false(self):
        # 使用 torch.jit.script 装饰器定义一个函数 fn，接受一个整数参数 x，返回一个字符串
        @torch.jit.script
        def fn(x: int) -> str:
            # 如果 torch.jit.isinstance 判断 x 是 Union[str, float] 或 List[str] 或 str 类型
            if torch.jit.isinstance(x, (Union[str, float], List[str], str)):
                # 计算 z = x + "foo"，返回结果
                z = x + "foo"
                return z
            else:
                # 否则返回字符串 "bar"
                return "bar"

        # 获取函数 fn 的图形结构
        s = fn.graph

        # 检查 fn 的图形结构中是否存在分支语句
        FileCheck().check_not("block0()").check_not("block1()").run(s)
    def test_union_type_refinement_statically_true(self):
        # 定义一个 Torch 脚本函数，输入参数 x 可以是 List[int] 或 int 类型，返回值也是这两种类型之一
        @torch.jit.script
        def fn(x: Union[List[int], int]) -> Union[List[int], int]:
            # 如果 x 不是 int 或 List[int] 类型的实例，则直接返回 x
            if not torch.jit.isinstance(x, (int, List[int])):
                return x
            else:
                # 在这个分支中，创建一个列表 [1, 2, 3]
                l = [1, 2, 3]
                # 将 l 赋值给 y，类型为 Union[List[int], int]
                y: Union[List[int], int] = l
                return y

        # 获取函数 fn 的计算图
        s = fn.graph

        # 检查计算图中是否没有分支语句
        FileCheck().check_not("block0()").check_not("block1()").run(s)

    def test_union_type_refinement_partial_static_refinement_tuple_rhs(self):
        # 定义一个函数 fn，输入参数 x 可以是 List[int] 或 int 类型，返回值是 int 类型
        def fn(x: Union[List[int], int]) -> int:
            # 如果 x 是 int 或 float 或 str 类型的实例，则执行以下代码
            if torch.jit.isinstance(x, (int, float, str)):
                # 在这种情况下，我们知道 x 是 int 类型，执行 x + 1
                z = x + 1
                return z
            else:
                # 如果 x 不是上述类型的实例，则返回 100
                return 100

        # 分别对参数为列表和整数的情况进行 Torch 脚本检查
        self.checkScript(fn, ([1, 2, 3],))
        self.checkScript(fn, (1,))

    def test_union_type_refinement_partial_static_refinement_union_rhs(self):
        # 定义一个函数 fn，输入参数 x 可以是 List[int] 或 int 类型，返回值是 int 类型
        def fn(x: Union[List[int], int]) -> int:
            # 如果 x 是 Union[int, float, str] 类型的实例，则执行以下代码
            if torch.jit.isinstance(x, Union[int, float, str]):
                # 在这种情况下，我们知道 x 是 int 类型，执行 x + 1
                z = x + 1
                return z
            else:
                # 如果 x 不是上述类型的实例，则返回 100
                return 100

        # 分别对参数为列表和整数的情况进行 Torch 脚本检查
        self.checkScript(fn, ([1, 2, 3],))
        self.checkScript(fn, (1,))

    def test_union_type_refinement_internal_declaration(self):
        # 定义一个函数 fn，输入参数 flag 是 bool 类型，返回值是 str 类型
        def fn(flag: bool) -> str:
            # 声明变量 x 可以是 Union[int, str, None] 类型的实例，初始化为 None
            x: Union[int, str, None] = None
            # 根据 flag 的值选择不同的分支
            if flag:
                y = "foo"
            else:
                y = 1
            # 如果 x 是 str 类型的实例，则返回 x
            if isinstance(x, str):
                return x
            else:
                # 否则返回字符串 "bar"
                return "bar"

        # 分别对参数为 True 和 False 的情况进行 Torch 脚本检查
        self.checkScript(fn, (True,))
        self.checkScript(fn, (False,))

    def test_union_branching_with_union_return_and_homogenous_types(self):
        # 定义一个函数 fn，输入参数 x 是 int 类型，返回值可以是 int 或 str 类型
        def fn(x: int) -> Union[int, str]:
            # 如果 x 是奇数，则返回字符串 "foo"
            if x % 2:
                return "foo"
            else:
                # 如果 x 是偶数，则返回字符串 "bar"
                return "bar"

        # 分别对参数为奇数和偶数的情况进行 Torch 脚本检查
        self.checkScript(fn, (1,))
        self.checkScript(fn, (8,))

    def test_union_branching_does_not_autoinfer_undeclared_union(self):
        # 定义一个函数 fn，输入参数 x 是 int 类型，返回值是 str 类型
        def fn(x: int) -> str:
            # 根据 x 的奇偶性选择不同的分支
            if x % 2:
                y = "foo"
            else:
                y = x
            # 如果 y 是 str 类型的实例，则返回 y
            if isinstance(y, str):
                return y
            else:
                # 否则抛出异常，指示 y 在不同分支中具有不同类型
                return "bar"

        # 测试函数 fn 是否会抛出特定的异常
        with self.assertRaisesRegex(
            RuntimeError,
            "y is set to type str"
            " in the true branch and type int "
            "in the false branch",
        ):
            torch.jit.script(fn)
    # 测试：联合类型分支不会扩展现有推断类型
    def test_union_branching_does_not_widen_existing_inferred_type(self):
        # 定义一个函数 fn，接受一个整数 x，返回一个字符串
        def fn(x: int) -> str:
            # 初始化 y 为字符串 "foo"
            y = "foo"
            # 如果 x 取模 2 不为 0，即 x 是奇数时，将 y 赋值为 "bar"
            if x % 2:
                y = "bar"
            else:
                # 否则，将 y 赋值为 x
                y = x
            # 如果 y 是字符串类型，返回 y
            if isinstance(y, str):
                return y
            else:
                # 否则返回字符串 "baz"
                return "baz"

        # 使用 assertRaisesRegex 验证在转换为 Torch 脚本时是否会抛出指定异常信息
        with self.assertRaisesRegex(
            RuntimeError,
            "previously had type "
            "str but is now being assigned to a"
            " value of type int",
        ):
            torch.jit.script(fn)

    # 测试：联合类型模式匹配内部类型
    def test_union_schema_matching_on_internal_type(self):
        # 定义一个函数 fn，接受一个联合类型参数 x，返回一个整数
        def fn(x: Union[List[int], Dict[str, int]]) -> int:
            # 如果 x 是 List[int] 类型，返回其第一个元素
            if torch.jit.isinstance(x, List[int]):
                return x[0]
            else:
                # 否则返回字典 x 的第一个值
                return list(x.values())[0]

        # 使用 checkScript 验证脚本转换的正确性
        self.checkScript(fn, ([1, 2, 3],))
        self.checkScript(fn, ({"foo": 1, "bar": 2, "baz": 3},))

    # 测试：联合类型减少细化
    def test_union_subtractive_refinement(self):
        # 定义一个函数 fn，接受一个联合类型参数 x，返回一个整数
        def fn(x: Union[List[int], int]) -> int:
            # 如果 x 不是整数类型，向 x 中添加元素 1，然后返回第一个元素
            if not isinstance(x, int):
                x.append(1)
                return x[0]
            else:
                # 否则直接返回 x
                return x

        # 使用 checkScript 验证脚本转换的正确性
        self.checkScript(fn, (1,))
        self.checkScript(fn, ([1, 2, 3],))

    # 测试：带容器的联合类型减少细化
    def test_union_subtractive_refinement_with_container(self):
        # 定义一个函数 fn，接受一个联合类型参数 x，返回一个整数
        def fn(x: Union[List[int], int]) -> int:
            # 如果 x 不是 List[int] 类型，直接返回 x
            if not torch.jit.isinstance(x, List[int]):
                return x
            else:
                # 否则向 x 中添加元素 1，然后返回第一个元素
                x.append(1)
                return x[0]

        # 使用 checkScript 验证脚本转换的正确性
        self.checkScript(fn, (1,))
        self.checkScript(fn, ([1, 2, 3],))

    # 测试：联合类型内存别名
    def test_union_memory_aliasing(self):
        # 定义一个函数 fn，返回一个 Torch 张量列表 x
        def fn():
            x: List[torch.Tensor] = []
            z: List[Optional[List[torch.Tensor]]] = []
            z.append(x)
            # 将 z 中的第一个元素赋值给 x_alias
            x_alias = z[0]
            # 如果 x_alias 是 List[torch.Tensor] 类型，向其添加一个张量 3
            if torch.jit.isinstance(x_alias, List[torch.Tensor]):
                x_alias.append(torch.tensor(3))
            # 返回 x
            return x

        # 使用 checkScript 验证脚本转换的正确性
        self.checkScript(fn, ())

    # 测试：联合类型序列化保留类型注释
    def test_union_serialization_preserves_type_annotations(self):
        # 此函数经过 torch.jit.save 和 torch.jit.load 后如果类型注释不被保留将会失败
        # 需要 Union[str, int] 注释以确保 y 被类型化为联合类型，而不是一个分支是 str 而另一个是 int
        def fn(x: int) -> str:
            # 如果 x 取模 2 为真，y 被注释为 Union[str, int] 类型，赋值为 "bar"
            if x % 2:
                y: Union[str, int] = "bar"
            else:
                # 否则 y 被注释为 Union[str, int] 类型，赋值为 x
                y: Union[str, int] = x
            # 如果 y 是字符串类型，返回 y
            if isinstance(y, str):
                return y
            else:
                # 否则返回字符串 "baz"
                return "baz"

        # 使用 checkScript 验证脚本转换的正确性
        self.checkScript(fn, (1,))
        self.checkScript(fn, (8,))

    # 私有方法：验证通过，传入模板、注释和左手边值，检查脚本的正确性
    def _assert_passes(self, template: str, ann: str, lhs: str):
        # 生成代码，替换模板中的注释和左手边值
        code = template.format(ann=ann, lhs=lhs)
        # 使用 checkScript 验证脚本转换的正确性
        self.checkScript(code, (), name="fn")
    # 定义一个用于测试异常情况的辅助方法，检查是否引发了指定异常
    def _assert_raises(self, template: str, ann: str, lhs: str, msg: str):
        # 使用模板字符串构建代码，替换注解和左手边表达式
        code = template.format(ann=ann, lhs=lhs)
        # 断言运行时异常中包含特定消息
        with self.assertRaisesRegex(RuntimeError, msg):
            # 使用 Torch 的 JIT 编译单元创建对象，捕获异常
            cu = torch.jit.CompilationUnit(code, _frames_up=1)
            # 获取编译单元对象中的特定前端函数字符串表示，忽略 B009 规范
            string_frontend = getattr(cu, "fn")  # noqa: B009
```