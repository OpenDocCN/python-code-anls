# `.\pytorch\test\jit\test_union_pep604.py`

```
# Owner(s): ["oncall: jit"]

# 导入所需的模块
import io  # 提供用于处理文件流的工具
import os  # 提供与操作系统交互的功能
import sys  # 提供与 Python 解释器交互的功能
import unittest  # 提供单元测试框架
from enum import Enum  # 枚举类型支持
from textwrap import dedent  # 文本缩进处理工具
from typing import Dict, List, Optional, Tuple, Union  # 类型提示支持

import torch  # PyTorch 深度学习框架
from torch.testing import FileCheck  # 用于测试的文件检查工具

# 将 test/ 中的辅助文件路径添加到系统路径中，使其可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, make_global  # 导入 JIT 测试相关的工具

# 如果该脚本被直接运行，则抛出运行时错误，建议使用指定的方式运行
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


# 对于 Python 版本低于 3.10 的环境，跳过测试
@unittest.skipIf(sys.version_info < (3, 10), "Requires Python 3.10")
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
        # 定义一个带有 Union 类型注解的测试函数
        def test_func(a: int | float, b: Optional[int]):
            return 0

        # 对测试函数进行脚本化（转换为 TorchScript）
        scripted_func = torch.jit.script(test_func)
        # 获取脚本化函数的图形表示字符串和代码表示字符串
        graph_rep = str(scripted_func.graph)
        code_rep = str(scripted_func.code)
        # 检查 TorchScript 图形 IR 是否包含 Union() 注解
        FileCheck().check("Union(").check("int?").run(graph_rep)
        # 检查序列化代码是否包含 Union[] 注解
        FileCheck().check("Union[").check("Optional[int]").run(code_rep)
        # 使用 JitTestCase 提供的 checkScript 方法进行测试
        self.checkScript(test_func, (5, 6))
        # 检查 TorchScript 的 IR 是否能够被解析而不引发错误
        torch._C.parse_ir(str(scripted_func.graph))

    def test_union_with_scalar_values(self):
        # 定义一个接受 int 或 float 类型参数并返回 str 类型的函数
        def fn(x: int | float) -> str:
            return "foo"

        # 使用 JitTestCase 提供的 checkScript 方法进行测试
        self.checkScript(fn, (1,))
        self.checkScript(fn, (1.0,))

        # 对函数进行脚本化（转换为 TorchScript）
        scripted = torch.jit.script(fn)

        # 测试脚本化函数是否会引发预期的异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[float, int\] but "
            "instead found type str",
        ):
            scripted("1")
    # 定义测试方法，测试 Union 类型与 collections 模块的结合使用
    def test_union_with_collections(self):
        # 定义函数 fn，参数 x 可以是 Dict[str, int] 或 List[int]，返回类型为 str
        def fn(x: Dict[str, int] | List[int]) -> str:
            return "foo"

        # 使用 self.checkScript 方法验证 fn 函数在给定参数字典 {"foo": 1, "bar": 2, "baz": 3} 下的运行结果
        self.checkScript(fn, ({"foo": 1, "bar": 2, "baz": 3},))
        # 使用 self.checkScript 方法验证 fn 函数在给定参数列表 [1, 2, 3] 下的运行结果
        self.checkScript(fn, ([1, 2, 3],))

        # 将 fn 函数编译为 TorchScript 代码
        scripted = torch.jit.script(fn)

        # 使用 assertRaisesRegex 检测运行时错误，确保传入类型为 Dict[str, str] 的参数时抛出特定异常信息
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[List\[int\], Dict\[str, "
            r"int\]\] but instead found type "
            r"Dict\[str, str\]",
        ):
            scripted({"foo": "bar", "baz": "qux"})

        # 使用 assertRaisesRegex 检测运行时错误，确保传入类型为 List[str] 的参数时抛出特定异常信息
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[List\[int\], Dict\[str, "
            r"int\]\] but instead found type "
            r"List\[str\]",
        ):
            scripted(["foo", "bar", "baz"])

        # 使用 assertRaisesRegex 检测运行时错误，确保传入类型为 str 的参数时抛出特定异常信息
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
        # 定义枚举类型 Color 包含 RED 和 GREEN 两个成员
        class Color(Enum):
            RED = 1
            GREEN = 2

        # 将 Color 类型设为全局可见
        make_global(Color)

        # 定义函数 fn，参数 x 可以是 str 或 Color 类型，返回类型为 str
        def fn(x: str | Color) -> str:
            return "foo"

        # 使用 self.checkScript 方法验证 fn 函数在给定参数 Color.RED 下的运行结果
        self.checkScript(fn, (Color.RED,))
        # 使用 self.checkScript 方法验证 fn 函数在给定参数 "red" 下的运行结果
        self.checkScript(fn, ("red",))

        # 将 fn 函数编译为 TorchScript 代码
        scripted = torch.jit.script(fn)

        # 使用 assertRaisesRegex 检测运行时错误，确保传入类型为 int 的参数时抛出特定异常信息
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[__torch__.jit.test_union_pep604."
            r"Color, str\] but instead found "
            "type int",
        ):
            scripted(1)

    # 测试 Union 类型在类构造函数中的使用
    def test_union_in_class_constructor(self):
        # 使用 torch.jit.script 将类 A 编译为 TorchScript 代码
        @torch.jit.script  # noqa: B903
        class A:  # noqa: B903
            # 类 A 的构造函数，参数 x 可以是 int 或 str 类型，无返回值
            def __init__(self, x: int | str) -> None:
                self.x = x

        # 定义函数 fn，参数 x 可以是 str 或 int 类型，返回类型为 A
        def fn(x: str | int) -> A:
            return A(x)

        # 使用 self.assertEqual 验证 fn("foo").x 的返回值为 "foo"
        self.assertEqual(fn("foo").x, "foo")
        # 使用 self.assertEqual 验证 fn(1).x 的返回值为 1
        self.assertEqual(fn(1).x, 1)

        # 将 fn 函数编译为 TorchScript 代码
        scripted = torch.jit.script(fn)

        # 使用 assertRaisesRegex 检测运行时错误，确保传入类型为 List[str] 的参数时抛出特定异常信息
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a member of"
            r" Union\[int, str\] but instead "
            r"found type List\[str\]",
        ):
            scripted(["foo", "bar", "baz"])

    # 测试 Union 类型作为返回类型的函数
    def test_union_return_type(self):
        # 定义函数 fn，参数 x 是 int 类型，返回类型为 int 或 str
        def fn(x: int) -> int | str:
            return "foo"

        # 使用 self.checkScript 方法验证 fn 函数在给定参数 1 下的运行结果
        self.checkScript(fn, (1,))

    # 测试 Union 类型在注解中的使用
    def test_union_as_annotation(self):
        # 定义函数 fn，无参数，返回类型为 int 或 str
        def fn() -> int | str:
            # 声明变量 x，类型为 int 或 str，赋值为 "foo"
            x: int | str = "foo"
            return x

        # 使用 self.checkScript 方法验证 fn 函数在无参数情况下的运行结果
        self.checkScript(fn, ())

    # 测试 Union 类型在类型化容器中的使用
    def test_union_as_annotation_in_typed_container(self):
        # 定义函数 fn，无参数，无返回值
        def fn() -> None:
            # 声明列表 l，元素类型为 int 或 str
            l: List[int | str] = []
            # 声明变量 u1，类型为 int 或 str，赋值为 "foo"
            u1: int | str = "foo"
            # 声明变量 u2，类型为 int 或 str，赋值为 1
            u2: int | str = 1
            # 将 u1 添加到列表 l 中
            l.append(u1)
            # 将 u2 添加到列表 l 中
            l.append(u2)

        # 使用 self.checkScript 方法验证 fn 函数在无参数情况下的运行结果
        self.checkScript(fn, ())
    def test_union_as_annotation_py2(self):
        def fn():
            # type: () -> int | str  # 定义函数 fn，返回类型为 int 或 str
            x: int | str = "foo"  # 声明变量 x，类型为 int 或 str，初始值为 "foo"
            return x  # 返回变量 x 的值

        self.checkScript(fn, ())

    def test_union_as_internal_tuple_type(self):
        def fn():
            t: Tuple[int | str, int | str] = (1, "foo")  # 声明变量 t，类型为元组，包含两个元素，每个元素类型为 int 或 str
            return t  # 返回变量 t 的值

        self.checkScript(fn, ())

    def test_union_variable_can_be_reassigned(self):
        @torch.jit.script
        def aux1(i: int):
            return int(i**2)

        @torch.jit.script
        def aux2(s: str):
            return s + s

        def fn() -> int | str:
            x: int | str = "foo"  # 声明变量 x，类型为 int 或 str，初始值为 "foo"
            i: int = 1  # 声明变量 i，类型为 int，初始值为 1
            x = i  # 变量 x 被重新赋值为 i 的值
            y: int = aux1(x)  # 声明变量 y，类型为 int，调用 aux1 函数处理 x 的值
            z: str = aux2(str(y))  # 声明变量 z，类型为 str，调用 aux2 函数处理 y 转换为字符串的值
            x = z  # 变量 x 被重新赋值为 z 的值
            return x  # 返回变量 x 的值

        self.checkScript(fn, ())

    def test_union_does_not_replace_existing_annotated_type(self):
        def fn():
            x: List[int] = [1, 2, 3]  # 声明变量 x，类型为列表，包含整数类型的元素
            x.append("foo")  # 向列表 x 中添加字符串类型的元素
            return x  # 返回变量 x 的值

        with self.assertRaisesRegex(RuntimeError, "Could not match type str"):
            scripted = torch.jit.script(fn)  # 对函数 fn 进行脚本化处理
            scripted()

    def test_union_does_not_replace_existing_annotated_type_union(self):
        def fn():
            x: List[int | str] = [1, "foo", 3]  # 声明变量 x，类型为列表，包含 int 或 str 类型的元素
            x.append(2.0)  # 向列表 x 中添加浮点数类型的元素
            return x  # 返回变量 x 的值

        with self.assertRaisesRegex(RuntimeError, "Could not match type float"):
            scripted = torch.jit.script(fn)  # 对函数 fn 进行脚本化处理
            scripted()

    def test_union_does_not_replace_existing_annotated_type_empty_container(self):
        def fn():
            x: List[int] = []  # 声明变量 x，类型为空列表，预期包含整数类型的元素
            x.append("foo")  # 向空列表 x 中添加字符串类型的元素
            return x  # 返回变量 x 的值

        with self.assertRaisesRegex(RuntimeError, "Could not match type str"):
            scripted = torch.jit.script(fn)  # 对函数 fn 进行脚本化处理
            scripted()

    def test_unions_of_unions_are_flattened(self):
        @torch.jit.script
        def fn(x: (int | str) | float) -> str:
            return "foo"

        s = fn.graph

        FileCheck().check("x : Union(float, int, str)").run(s)  # 检查脚本化函数的图形中变量 x 的类型注释

    def test_unions_of_a_single_argument_vanish(self):
        @torch.jit.script
        def fn(x: Union[int]) -> str:
            return "foo"

        s = fn.graph

        FileCheck().check("x : int").run(s)  # 检查脚本化函数的图形中变量 x 的类型注释

    def test_union_redundant_arguments_are_skipped(self):
        @torch.jit.script
        def fn(x: int | str | int) -> str:
            return "foo"

        s = fn.graph

        FileCheck().check("x : Union(int, str)").run(s)  # 检查脚本化函数的图形中变量 x 的类型注释

    def test_union_redundant_arguments_are_skipped_optional(self):
        @torch.jit.script
        def fn(x: int | Optional[float] | Optional[int]) -> str:
            return "foo"

        s = fn.graph

        FileCheck().check("x : Union(float, int, NoneType)").run(s)  # 检查脚本化函数的图形中变量 x 的类型注释
    # 定义一个测试方法，用于验证在子类型化时跳过冗余参数的情况
    def test_union_redundant_arguments_are_skipped_subtyping(self):
        # 定义一个脚本化的函数，参数 x 可以是 str、可选的元组(int, int)或(int?, int)
        @torch.jit.script
        def fn(x: str | Tuple[Optional[int], int] | Tuple[int, int]) -> str:
            return "foo"

        # 获取函数 fn 的计算图
        s = fn.graph

        # 使用 FileCheck 检查计算图中 x 参数的类型声明
        FileCheck().check("x : Union((int?, int), str)").run(s)

    # 定义一个测试方法，用于验证在容器类型中忽略冗余参数的情况
    def test_union_redundant_arguments_are_skipped_container(self):
        # 定义一个脚本化的函数，参数 x 可以是 List[str]、List[float] 或 List[str]
        @torch.jit.script
        def fn(x: List[str] | List[float] | List[str]) -> str:
            return "foo"

        # 获取函数 fn 的计算图
        s = fn.graph

        # 使用 FileCheck 检查计算图中 x 参数的类型声明
        FileCheck().check("x : Union(float[], str[])").run(s)

    # 定义一个测试方法，用于验证在 Union 中参数顺序被忽略的情况
    def test_union_argument_order_is_ignored(self):
        # 定义一个脚本化的函数，参数 x 可以是 int 或 str
        @torch.jit.script
        def fn1(x: int | str) -> str:
            return "foo"

        # 定义一个脚本化的函数，参数 x 可以是 str 或 int
        @torch.jit.script
        def fn2(x: str | int) -> str:
            return "foo"

        # 遍历 fn1 和 fn2 的计算图
        for s in (fn1.graph, fn2.graph):
            # 使用 FileCheck 检查计算图中 x 参数的类型声明
            FileCheck().check("x : Union(int, str)").run(s)

    # 定义一个测试方法，用于验证在容器类型中 Union 的参数顺序被忽略的情况
    def test_union_argument_order_is_ignored_container(self):
        # 定义一个脚本化的函数，参数 x 可以是 List[str] 或 List[int]
        @torch.jit.script
        def fn1(x: List[str] | List[int]) -> str:
            return "foo"

        # 定义一个脚本化的函数，参数 x 可以是 List[int] 或 List[str]
        @torch.jit.script
        def fn2(x: List[int] | List[str]) -> str:
            return "foo"

        # 遍历 fn1 和 fn2 的计算图
        for s in (fn1.graph, fn2.graph):
            # 使用 FileCheck 检查计算图中 x 参数的类型声明
            FileCheck().check("x : Union(int[], str[])").run(s)

    # 定义一个测试方法，用于验证 Optional[T] 和 T? 在 Union 中的等价性
    def test_union_T_None_is_equivalent_to_optional_T(self):
        # 定义一个脚本化的函数 inner，参数 x 可以是 int 或 None
        @torch.jit.script
        def inner(x: int | None) -> int:
            if x is not None:
                return x
            else:
                return 5

        # 定义一个脚本化的函数 fn1，不接受参数，返回一个整数
        @torch.jit.script
        def fn1() -> int:
            # 声明两个 Optional[int] 类型的变量
            a: Optional[int] = 5
            b: Optional[int] = None
            # 调用 inner 函数处理变量 a 和 b，返回结果相加
            a_ = inner(a)
            b_ = inner(b)
            return a_ + b_

        # 断言 fn1 返回值为 10
        self.assertEqual(fn1(), 10)

        # 定义一个脚本化的函数 inner2，参数 x 是 Optional[int] 类型
        @torch.jit.script
        def inner2(x: Optional[int]) -> int:
            if x is not None:
                return x
            else:
                return 5

        # 定义一个脚本化的函数 fn2，不接受参数，返回一个整数
        @torch.jit.script
        def fn2() -> int:
            # 声明两个 int | None 类型的变量
            a: int | None = 5
            b: int | None = None
            # 调用 inner2 函数处理变量 a 和 b，返回结果相加
            a_ = inner(a)
            b_ = inner(b)
            return a_ + b_

        # 断言 fn2 返回值为 10
        self.assertEqual(fn2(), 10)

    # 定义一个预期失败的测试方法，用于验证 Union[None, str, int] 返回类型的处理
    @unittest.expectedFailure
    def test_union_optional_of_union_return(self):
        # 定义一个脚本化的函数 fn，不接受参数，返回 None、str 或 int 类型
        @torch.jit.script
        def fn() -> None | str | int:
            # 声明一个 Optional[int | str] 类型的变量 y
            y: Optional[int | str] = "foo"
            return y
    def test_union_optional_of_union_is_flattened(self):
        @torch.jit.script
        def fn(flag: int) -> str | int | None:
            # 定义变量 y，类型为 int | str | None，初始值为 "foo"
            y: int | str | None = "foo"
            # 根据 flag 的值选择 x 的类型
            if flag == 0:
                # 如果 flag 为 0，x 的类型为 Optional[int | str]，赋值为 y
                x: Optional[int | str] = y
            elif flag == 1:
                # 如果 flag 为 1，x 的类型为 Optional[int | str]，赋值为 1
                x: Optional[int | str] = 1
            else:
                # 否则，flag 不为 0 或 1，x 的类型为 Optional[int | str]，赋值为 None
                x: Optional[int | str] = None
            # 返回 x
            return x

        # 测试函数 fn 的返回值
        self.assertEqual(fn(0), "foo")
        self.assertEqual(fn(1), 1)
        self.assertEqual(fn(2), None)

        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 将函数 fn 保存到字节流缓冲区中
        torch.jit.save(fn, buffer)
        # 将保存的内容加载回来
        buffer = io.BytesIO(buffer.getvalue())
        l = torch.jit.load(buffer)

        # 获取加载后模型的代码
        s = l.code

        # 使用 FileCheck 检查加载后代码中是否包含指定的字符串
        FileCheck().check("Union[int, NoneType, str]").check(
            "Union[int, NoneType, str]"
        ).run(s)

    def test_union_subclasses_larger_union(self):
        def fn() -> int | str | torch.Tensor:
            # 定义变量 x，类型为 int | str，初始值为 "foo"
            x: int | str = "foo"
            return x

        # 检查脚本化后的 fn 函数
        self.checkScript(fn, ())

    # TODO: 我们希望最终支持此功能。问题跟踪地址为 https://github.com/pytorch/pytorch/issues/58167
    def test_union_as_dict_key(self):
        def fn():
            # 定义变量 x，类型为 Dict[int | str, str]，初始为空字典
            x: Dict[int | str, str] = {}
            x["foo"] = "bar"
            x[1] = 2  # 此处可能导致脚本化时出错，因为不支持除 int, float, complex, Tensor, device, string 之外的键类型
            return x[1]

        # 预期在脚本化 fn 函数时抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "only int, float, "
            "complex, Tensor, device and string keys "
            "are supported",
        ):
            torch.jit.script(fn)

    def test_union_as_dict_value(self):
        def fn():
            # 定义变量 x，类型为 Dict[str, int | str]，初始为空字典
            x: Dict[str, int | str] = {}
            x["foo"] = "bar"
            x["baz"] = 2
            return x["baz"]

        # 检查脚本化后的 fn 函数
        self.checkScript(fn, ())

    def test_union_module_with_union_instance_variable(self):
        class M(torch.nn.Module):
            # 类变量 x，类型为 int | str
            x: int | str

            def __init__(self, x: int | str):
                super().__init__()
                # 实例变量 x，类型为 int | str，初始值为构造函数参数 x
                self.x: int | str = x

            def forward(self, y: int | str):
                # 设置实例变量 x 为参数 y 的值
                self.x = y
                return self.x

        # 检查模块 M 的脚本化版本
        self.checkModule(
            M(
                2,
            ),
            (1,),
        )
        self.checkModule(M("bar"), ("foo",))

    def test_union_module_with_union_class_variable(self):
        class M(torch.nn.Module):
            # 类变量 x，类型为 int | str，默认值为 "foo"
            x: int | str = "foo"

            def __init__(self, y: int):
                super().__init__()
                x = y  # 这里应该是 self.x = y，修正为正确的赋值操作

            def forward(self, z: str):
                x = z  # 这里应该是 self.x = z，修正为正确的赋值操作
                return x

        # 检查模块 M 的脚本化版本
        self.checkModule(M(1), ("foo",))
    # 定义一个测试方法，用于测试联合类型的细化
    def test_union_type_refinement(self):
        # 定义一个函数 fn，接受一个联合类型参数 x (可以是 int 或 str)，返回一个 str
        def fn(x: int | str) -> str:
            # 检查 x 是否是 str 类型
            if isinstance(x, str):
                # 如果是 str，将 x 和 "bar" 连接并返回 x 本身（这里应该返回 x + "bar" 而不是 x）
                z = x + "bar"
                return x
            else:
                # 如果不是 str，返回 "baz"
                return "baz"

        # 使用自定义的方法 checkScript 测试 fn，传入参数 ("foo",)
        self.checkScript(fn, ("foo",))
        # 再次使用自定义的方法 checkScript 测试 fn，传入参数 (1,)
        self.checkScript(fn, (1,))

    # 定义一个测试方法，用于测试联合类型的细化，联合类型的右侧是一个联合类型
    def test_union_type_refinement_union_rhs(self):
        # 定义一个函数 fn，接受一个 int 类型参数 x，返回一个 str
        def fn(x: int) -> str:
            # 使用 torch.jit.isinstance 检查 x 是否是 int 或 str 类型
            if torch.jit.isinstance(x, int | str):
                # 如果是，返回 "bar"
                return "bar"
            else:
                # 否则返回 "baz"
                return "baz"

        # 使用自定义的方法 checkScript 测试 fn，传入参数 (1,)
        self.checkScript(fn, (1,))

    # 定义一个测试方法，用于测试联合类型的细化，联合类型的右侧是一个元组
    def test_union_type_refinement_tuple_rhs(self):
        # 定义一个函数 fn，接受一个联合类型参数 x (可以是 int、float 或 List[str])，返回一个 str
        def fn(x: int | float | List[str]) -> str:
            # 检查 x 是否是 int 或 float 类型
            if isinstance(x, (int, float)):
                # 如果是 int，返回其字符串表示
                if isinstance(x, int):
                    return str(x)
                else:
                    # 如果是 float，返回 "foo"
                    return "foo"
            else:
                # 如果不是 int 或 float，检查 x 的长度
                if len(x):
                    # 如果长度大于 0，返回列表 x 的第一个元素
                    return x[0]
                else:
                    # 如果长度为 0，返回 "bar"
                    return "bar"

        # 使用自定义的方法 checkScript 测试 fn，传入参数 (1,)
        self.checkScript(fn, (1,))
        # 再次使用自定义的方法 checkScript 测试 fn，传入参数 (1.0,)
        self.checkScript(fn, (1.0,))
        # 再次使用自定义的方法 checkScript 测试 fn，传入参数 (["a", "b", "c"],)
        self.checkScript(fn, (["a", "b", "c"],))

    # 定义一个测试方法，用于测试联合类型的细化，联合类型的右侧包含不支持的类型
    def test_union_type_refinement_tuple_rhs_noncontained_type(self):
        # 定义一个函数 fn，接受一个联合类型参数 x (可以是 int 或 List[str])，返回一个 str
        def fn(x: int | List[str]) -> str:
            # 检查 x 是否是 int 或 float 类型
            if isinstance(x, (int, float)):
                # 如果是，将 x 与自身相加并转换为字符串返回
                y = x + x
                return str(y)
            else:
                # 如果不是 int 或 float，检查 x 的长度
                if len(x):
                    # 如果长度大于 0，返回列表 x 的第一个元素
                    return x[0]
                else:
                    # 如果长度为 0，返回 "bar"
                    return "bar"

        # 使用自定义的方法 checkScript 测试 fn，传入参数 (1,)
        self.checkScript(fn, (1,))
        # 再次使用自定义的方法 checkScript 测试 fn，传入参数 (["a", "b", "c"],)
        self.checkScript(fn, (["a", "b", "c"],))

    # 定义一个测试方法，用于测试联合类型的细化，联合类型的右侧是一个联合类型，包含不支持的类型
    def test_union_type_refinement_tuple_rhs_union(self):
        # 使用 torch.jit.script 装饰器定义一个函数 fn，接受一个 int 类型参数 x，返回一个 str
        @torch.jit.script
        def fn(x: int) -> str:
            # 使用 torch.jit.isinstance 检查 x 是否是 int 或 str 类型，或者 float 类型
            if torch.jit.isinstance(x, (int | str, float)):
                # 如果是，将 x 与自身相加并转换为字符串返回
                y = x + x
                return str(y)
            else:
                # 如果不是，返回 "foo"
                return "foo"

        # TODO: 目前 `torch.jit.isinstance` 存在一个与元组文字相关的无关错误。
        # 请参考：https://github.com/pytorch/pytorch/issues/60095
        # 当错误修复后，将 `assertEqual` 更改为 `checkScript`
        # 使用 assertEqual 测试 fn(1) 的返回值是否为 "2"
        self.assertEqual(fn(1), "2")

    # 定义一个测试方法，用于测试联合类型的细化，当静态条件为假时
    def test_union_type_refinement_statically_false(self):
        # 使用 torch.jit.script 装饰器定义一个函数 fn，接受一个 int 类型参数 x，返回一个 str
        @torch.jit.script
        def fn(x: int) -> str:
            # 使用 torch.jit.isinstance 检查 x 是否是 str 或 float 类型，或者 List[str] 类型，或者 str 类型
            if torch.jit.isinstance(x, (str | float, List[str], str)):
                # 如果是，将 x 与 "foo" 相加并返回结果
                z = x + "foo"
                return z
            else:
                # 如果不是，返回 "bar"
                return "bar"

        # 获取 fn 的计算图
        s = fn.graph

        # 检查计算图中是否没有任何分支语句
        FileCheck().check_not("block0()").check_not("block1()").run(s)
    def test_union_type_refinement_statically_true(self):
        @torch.jit.script
        def fn(x: List[int] | int) -> List[int] | int:
            # 检查 x 的类型是否为 int 或者 List[int]，如果不是则直接返回 x
            if not torch.jit.isinstance(x, (int, List[int])):
                return x
            else:
                # 创建一个整数列表 [1, 2, 3]
                l = [1, 2, 3]
                # 将变量 y 声明为类型为 List[int] | int 的联合类型，并赋值为 l
                y: List[int] | int = l
                return y

        # 获取 fn 函数的图形表示
        s = fn.graph

        # 检查图形表示中是否没有分支语句
        FileCheck().check_not("block0()").check_not("block1()").run(s)

    def test_union_type_refinement_partial_static_refinement_tuple_rhs(self):
        def fn(x: List[int] | int) -> int:
            # 如果 x 的类型是 int, float 或者 str 中的一种，执行以下代码块
            if torch.jit.isinstance(x, (int, float, str)):
                # 在这里我们可以确定 x 是一个 int 类型
                z = x + 1
                return z
            else:
                return 100

        # 分别对输入 [1, 2, 3] 和 1 调用 checkScript 方法来检查 fn 函数
        self.checkScript(fn, ([1, 2, 3],))
        self.checkScript(fn, (1,))

    def test_union_type_refinement_partial_static_refinement_union_rhs(self):
        def fn(x: List[int] | int) -> int:
            # 如果 x 的类型是 int, float 或者 str 中的一种，执行以下代码块
            if torch.jit.isinstance(x, int | float | str):
                # 在这里我们可以确定 x 是一个 int 类型
                z = x + 1
                return z
            else:
                return 100

        # 分别对输入 [1, 2, 3] 和 1 调用 checkScript 方法来检查 fn 函数
        self.checkScript(fn, ([1, 2, 3],))
        self.checkScript(fn, (1,))

    def test_union_type_refinement_internal_declaration(self):
        def fn(flag: bool) -> str:
            # 声明一个 x 变量，类型为 int | str | None，初始化为 None
            x: int | str | None = None
            # 根据 flag 的值选择不同的分支
            if flag:
                y = "foo"
            else:
                y = 1
            # 如果 x 的类型是 str，则返回 x
            if isinstance(x, str):
                return x
            else:
                return "bar"

        # 分别对输入 True 和 False 调用 checkScript 方法来检查 fn 函数
        self.checkScript(fn, (True,))
        self.checkScript(fn, (False,))

    def test_union_branching_with_union_return_and_homogenous_types(self):
        def fn(x: int) -> int | str:
            # 如果 x 是奇数，返回字符串 "foo"，否则返回字符串 "bar"
            if x % 2:
                return "foo"
            else:
                return "bar"

        # 分别对输入 1 和 8 调用 checkScript 方法来检查 fn 函数
        self.checkScript(fn, (1,))
        self.checkScript(fn, (8,))

    def test_union_branching_does_not_autoinfer_undeclared_union(self):
        def fn(x: int) -> str:
            # 根据 x 的奇偶性选择不同的分支
            if x % 2:
                y = "foo"
            else:
                y = x
            # 如果 y 的类型是 str，则返回 y
            if isinstance(y, str):
                return y
            else:
                return "bar"

        # 预期 fn 函数会抛出 RuntimeError 异常，描述 y 在 true 分支为 str 类型，
        # false 分支为 int 类型的情况
        with self.assertRaisesRegex(
            RuntimeError,
            "y is set to type str"
            " in the true branch and type int "
            "in the false branch",
        ):
            torch.jit.script(fn)
    # 定义一个测试函数，验证联合类型不会扩展现有推断类型
    def test_union_branching_does_not_widen_existing_inferred_type(self):
        # 定义一个函数 fn，接受整数 x 并返回字符串
        def fn(x: int) -> str:
            # 初始化变量 y 为字符串 "foo"
            y = "foo"
            # 如果 x 除以 2 的余数不为 0，则将 y 赋值为 "bar"
            if x % 2:
                y = "bar"
            else:
                # 否则将 y 赋值为 x
                y = x
            # 如果 y 是字符串类型，则返回 y
            if isinstance(y, str):
                return y
            else:
                # 否则返回字符串 "baz"
                return "baz"

        # 使用 self.assertRaisesRegex 验证调用 torch.jit.script(fn) 时抛出的异常信息
        with self.assertRaisesRegex(
            RuntimeError,
            "previously had type "
            "str but is now being assigned to a"
            " value of type int",
        ):
            # 对 fn 进行脚本化处理
            torch.jit.script(fn)

    # 测试函数，验证联合类型的模式匹配在内部类型上
    def test_union_schema_matching_on_internal_type(self):
        # 定义一个函数 fn，接受 List[int] 或 Dict[str, int]，返回一个整数
        def fn(x: List[int] | Dict[str, int]) -> int:
            # 如果 x 是 List[int] 类型，则返回其第一个元素
            if torch.jit.isinstance(x, List[int]):
                return x[0]
            else:
                # 否则返回字典 x 的第一个值
                return list(x.values())[0]

        # 验证 fn 在接受 ([1, 2, 3],) 参数时的脚本化处理结果
        self.checkScript(fn, ([1, 2, 3],))
        # 验证 fn 在接受 ({"foo": 1, "bar": 2, "baz": 3},) 参数时的脚本化处理结果
        self.checkScript(fn, ({"foo": 1, "bar": 2, "baz": 3},))

    # 测试函数，验证联合类型的减法细化
    def test_union_subtractive_refinement(self):
        # 定义一个函数 fn，接受 List[int] 或 int，返回一个整数
        def fn(x: List[int] | int) -> int:
            # 如果 x 不是 int 类型，则在其末尾添加整数 1，并返回第一个元素
            if not isinstance(x, int):
                x.append(1)
                return x[0]
            else:
                # 否则直接返回 x
                return x

        # 验证 fn 在接受 (1,) 参数时的脚本化处理结果
        self.checkScript(fn, (1,))
        # 验证 fn 在接受 ([1, 2, 3],) 参数时的脚本化处理结果
        self.checkScript(fn, ([1, 2, 3],))

    # 测试函数，验证联合类型的减法细化（带容器）
    def test_union_subtractive_refinement_with_container(self):
        # 定义一个函数 fn，接受 List[int] 或 int，返回一个整数
        def fn(x: List[int] | int) -> int:
            # 如果 x 不是 List[int] 类型，则直接返回 x
            if not torch.jit.isinstance(x, List[int]):
                return x
            else:
                # 否则在 x 的末尾添加整数 1，并返回第一个元素
                x.append(1)
                return x[0]

        # 验证 fn 在接受 (1,) 参数时的脚本化处理结果
        self.checkScript(fn, (1,))
        # 验证 fn 在接受 ([1, 2, 3],) 参数时的脚本化处理结果
        self.checkScript(fn, ([1, 2, 3],))

    # 测试函数，验证联合类型的内存别名
    def test_union_memory_aliasing(self):
        # 定义一个函数 fn，初始化空列表 x 和空的可选列表列表 z
        def fn():
            x: List[torch.Tensor] = []
            z: List[Optional[List[torch.Tensor]]] = []
            # 将 x 添加到 z 的末尾
            z.append(x)
            # 创建 x 的别名 x_alias 为 z 的第一个元素
            x_alias = z[0]
            # 如果 x_alias 是 List[torch.Tensor] 类型，则向其添加一个值为 3 的张量
            if torch.jit.isinstance(x_alias, List[torch.Tensor]):
                x_alias.append(torch.tensor(3))
            # 返回列表 x
            return x

        # 验证 fn 在无参数时的脚本化处理结果
        self.checkScript(fn, ())

    # 测试函数，验证联合类型的序列化保留类型注释
    def test_union_serialization_preserves_type_annotations(self):
        # 定义一个函数 fn，接受整数 x，返回一个字符串
        # 该函数在被 torch.jit.save 和 torch.jit.load 之后，如果类型注释没有被保留，将会失败
        def fn(x: int) -> str:
            # 如果 x 除以 2 的余数不为 0，则 y 的类型为 str | int，赋值为 "bar"
            if x % 2:
                y: str | int = "bar"
            else:
                # 否则 y 的类型为 str | int，赋值为 x
                y: str | int = x
            # 如果 y 是字符串类型，则返回 y
            if isinstance(y, str):
                return y
            else:
                # 否则返回字符串 "baz"
                return "baz"

        # 验证 fn 在接受 (1,) 参数时的脚本化处理结果
        self.checkScript(fn, (1,))
        # 验证 fn 在接受 (8,) 参数时的脚本化处理结果
        self.checkScript(fn, (8,))

    # 私有辅助函数，验证模板、注释和左手边的函数调用能够正常处理
    def _assert_passes(self, template: str, ann: str, lhs: str):
        # 使用给定的模板、注释和左手边调用 self.checkScript 进行验证
        code = template.format(ann=ann, lhs=lhs)
        self.checkScript(code, (), name="fn")
    # 定义一个私有方法 `_assert_raises`，用于测试特定条件下是否会引发异常
    def _assert_raises(self, template: str, ann: str, lhs: str, msg: str):
        # 使用传入的模板字符串 `template`，填充参数 `ann` 和 `lhs`，生成具体的代码字符串
        code = template.format(ann=ann, lhs=lhs)
        # 使用 `assertRaisesRegex` 上下文管理器来检查是否会抛出指定类型和消息的异常
        with self.assertRaisesRegex(RuntimeError, msg):
            # 使用 Torch 的 JIT 编译单元 `CompilationUnit`，传入生成的代码字符串 `code`
            cu = torch.jit.CompilationUnit(code, _frames_up=1)
            # 获取编译单元 `cu` 中名为 "fn" 的属性，并忽略 B009 编码规范的警告
            string_frontend = getattr(cu, "fn")  # noqa: B009
```