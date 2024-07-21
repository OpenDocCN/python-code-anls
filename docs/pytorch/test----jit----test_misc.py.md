# `.\pytorch\test\jit\test_misc.py`

```
# Owner(s): ["oncall: jit"]

# 导入所需的模块和库
import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.testing._internal.jit_utils

# 导入自定义的测试模块接口，禁用 F401 警告
from jit.test_module_interface import TestModuleInterface  # noqa: F401
from torch import jit
from torch.testing import FileCheck
from torch.testing._internal.common_utils import freeze_rng_state

# 导入测试中使用到的 JIT 测试工具和标记
from torch.testing._internal.jit_utils import JitTestCase, make_global, RUN_CUDA_HALF

# 将 test/ 目录加入到系统路径中，使得其中的辅助文件可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 如果直接运行该文件，则抛出运行时错误，提醒使用正确的方式运行
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestMisc(JitTestCase):
    def test_joined_str(self):
        # 定义一个测试函数，使用 f-string 进行字符串格式化输出
        def func(x):
            hello, test = "Hello", "test"
            print(f"{hello + ' ' + test}, I'm a {test}")
            print("format blank")
            hi = "hi"
            print(f"stuff before {hi}")
            print(f"{hi} stuff after")
            return x + 1

        x = torch.arange(4.0, requires_grad=True)

        # 使用 `capture_stdout` 上下文管理器捕获标准输出
        with self.capture_stdout() as captured:
            out = func(x)

        # 对函数进行脚本化，同时捕获脚本化后的标准输出
        scripted = torch.jit.script(func)
        with self.capture_stdout() as captured_script:
            out_script = func(x)

        # 断言函数输出和脚本化函数输出的结果相等，以及捕获的标准输出一致
        self.assertEqual(out, out_script)
        self.assertEqual(captured, captured_script)

    def test_kwarg_support(self):
        # 测试不支持可变数量参数的情况，预期抛出特定异常
        with self.assertRaisesRegex(
            torch.jit.frontend.NotSupportedError, "variable number of arguments"
        ):
            # 定义一个带有命名参数的 Module 类，其中一个参数有默认值
            class M(torch.nn.Module):
                def forward(self, *, n_tokens: int, device_name: str = 2):
                    pass

            # 对该 Module 进行脚本化，验证是否会抛出异常
            torch.jit.script(M())

        # 定义另一个 Module 类，测试参数缺失时的异常情况
        class M(torch.nn.Module):
            def forward(self, *, n_tokens: int, device_name: str):
                return n_tokens, device_name

        sm = torch.jit.script(M())

        # 预期缺少 'n_tokens' 参数时抛出异常
        with self.assertRaisesRegex(
            RuntimeError, "missing value for argument 'n_tokens'"
        ):
            sm()

        # 预期使用位置参数时抛出异常
        with self.assertRaisesRegex(RuntimeError, "positional arg"):
            sm(3, "hello")

        # 验证使用正确的命名参数时，Module 正常执行并返回预期结果
        self.assertEqual(sm(n_tokens=3, device_name="hello"), (3, "hello"))

    def test_tuple_subscripted_assign(self):
        # 测试元组元素赋值操作是否会抛出异常
        with self.assertRaisesRegex(RuntimeError, "subscripted assignment"):

            @torch.jit.script
            def foo(a: Tuple[int, int]) -> None:
                a[0] = a[1]

        # 测试元组元素增强赋值操作是否会抛出异常
        with self.assertRaisesRegex(RuntimeError, "augmented assignment"):

            @torch.jit.script
            def bar(a: Tuple[int, int]) -> None:
                a[0] += a[1]
    def test_subexpression_List_Future(self):
        @torch.jit.script
        def fn(x: List[torch.jit.Future[int]]) -> torch.jit.Future[int]:
            # 定义了一个 TorchScript 函数 fn，接受一个 torch.jit.Future[int] 类型的列表作为参数，并返回第一个元素
            return x[0]

        # 使用 FileCheck 检查 TorchScript 函数的图形表示，确保包含了 "Future[int]" 类型的注释
        FileCheck().check("Future[int]").check("Future[int]").run(fn.graph)

    def test_subexpression_Future_annotate(self):
        @torch.jit.script
        def fn() -> torch.jit.Future[int]:
            # 声明一个空的 torch.jit.Future[int] 类型的列表 x
            x: List[torch.jit.Future[int]] = []
            # 返回列表 x 的第一个元素
            return x[0]

        # 使用 FileCheck 检查 TorchScript 函数的图形表示，确保包含了 "Future[int][]" 类型的注释
        FileCheck().check("Future[int][]").run(fn.graph)

    def test_future_isinstance(self):
        @torch.jit.script
        def fn(x: Any) -> torch.jit.Future[int]:
            # 断言输入 x 的类型是 jit.Future[int]
            assert isinstance(x, jit.Future[int])
            # 返回输入 x
            return x

        # 使用 FileCheck 检查 TorchScript 函数的图形表示，确保包含了 "Future[int]" 类型的注释
        FileCheck().check("Future[int]").run(fn.graph)

    def test_str_refine_any(self):
        def forward(x: Any) -> str:
            # 如果输入 x 是字符串类型，则直接返回 x
            if isinstance(x, str):
                return x
            # 否则返回字符串 "foo"
            return "foo"

        # 将函数 forward 转换为 TorchScript 函数
        forward = torch.jit.script(forward)
        # 断言 forward(1) 返回 "foo"
        self.assertEqual(forward(1), "foo")
        # 断言 forward("bar") 返回 "bar"
        self.assertEqual(forward("bar"), "bar")

    def test_subexpression_Tuple_int_int_Future(self):
        @torch.jit.script
        def fn(
            x: Tuple[int, int, torch.jit.Future[int]]
        ) -> Tuple[int, torch.jit.Future[int]]:
            # 返回元组 x 的第一个元素和第三个元素
            return x[0], x[2]

        # 使用 FileCheck 检查 TorchScript 函数的图形表示，确保包含了 "(int, int, Future[int])" 和 "(int, Future[int])" 类型的注释
        FileCheck().check("(int, int, Future[int])").check("(int, Future[int])").run(
            fn.graph
        )

    def test_subexpression_Dict_int_Future(self):
        @torch.jit.script
        def fn(x: Dict[int, torch.jit.Future[int]], y: int) -> torch.jit.Future[int]:
            # 返回字典 x 中键为 y 的值，这个值是一个 torch.jit.Future[int] 类型的对象
            return x[y]

        # 使用 FileCheck 检查 TorchScript 函数的图形表示，确保包含了 "Dict(int, Future(int))" 和 "Future[int]" 类型的注释
        FileCheck().check("Dict(int, Future(int))").check("Future[int]").run(fn.graph)

    def test_subexpression_Optional(self):
        @torch.jit.script
        def fn(
            x: Optional[Dict[int, torch.jit.Future[int]]]
        ) -> Optional[torch.jit.Future[int]]:
            # 如果输入 x 不为空，则返回 x 的第一个值，否则返回 None
            if x is not None:
                return x[0]
            else:
                return None

        # 使用 FileCheck 检查 TorchScript 函数的图形表示，确保包含了 "Dict(int, Future(int))?" 类型的注释
        FileCheck().check("Dict(int, Future(int))?").run(fn.graph)

    def test_if_returning_any(self):
        """
        Check that an if statement can return different
        types early from each branch when the return
        type of the function is Any.
        """

        def if_function(inp: torch.Tensor) -> Any:
            # 如果输入张量 inp 的形状的第一个维度是 1，则返回 inp * inp，否则返回字符串 "str"
            if inp.shape[0] == 1:
                return inp * inp
            else:
                return "str"

        # 使用自定义的测试函数检查 if_function 的行为，验证其在不同分支返回不同类型时的行为
        self.checkScript(if_function, (torch.randn(5),))
    # 定义一个测试函数，用于测试 hacked_twin 函数的行为
    def test_hacked_twin(self):
        # 定义一个生成数据的嵌套函数
        def gen_data():
            # 内部使用 freeze_rng_state 上下文管理器，确保生成的数据是确定性的
            with freeze_rng_state():
                # 生成包含三个张量的元组：大小为 10 的随机张量，大小为 (20,) 的随机整数索引张量，大小为 20 的随机张量
                return torch.randn(10), torch.randint(10, (20,)), torch.randn(20)

        # 调用 gen_data 函数获取三个张量，并分别赋值给 input, index, value 变量
        (
            input,
            index,
            value,
        ) = gen_data()
        # 再次调用 gen_data 函数获取另一组三个张量，并分别赋值给 input1, index1, value1 变量
        (
            input1,
            index1,
            value1,
        ) = gen_data()
        # 使用 hacked_twin 函数对 input 执行索引赋值操作，不累积结果
        out1 = torch.ops.aten.index_put.hacked_twin(
            input, [index], value, accumulate=False
        )
        # 使用 torch 自带的 index_put 函数对 input1 执行索引赋值操作，不累积结果
        out2 = torch.index_put(input1, [index1], value1, accumulate=False)
        # 断言 out1 和 out2 的结果应当相等
        self.assertEqual(out1, out2)

        # 使用 hacked_twin 函数对 input 执行原地索引赋值操作，不累积结果
        torch.ops.aten.index_put_.hacked_twin(input, [index], value, accumulate=False)
        # 使用 torch 自带的 index_put_ 函数对 input1 执行原地索引赋值操作，不累积结果
        torch.index_put_(input1, [index1], value1, accumulate=False)
        # 断言经过原地操作后 input 和 input1 应当相等
        self.assertEqual(input, input1)

    # 定义一个测试函数，用于测试 _unsafe_hacked_twin 函数的行为
    def test_unsafe_hacked_twin(self):
        # 定义一个生成数据的嵌套函数
        def gen_data():
            # 内部使用 freeze_rng_state 上下文管理器，确保生成的数据是确定性的
            with freeze_rng_state():
                # 生成包含三个张量的元组：大小为 10 的随机张量，大小为 (20,) 的随机整数索引张量，大小为 20 的随机张量
                return torch.randn(10), torch.randint(10, (20,)), torch.randn(20)

        # 调用 gen_data 函数获取三个张量，并分别赋值给 input, index, value 变量
        (
            input,
            index,
            value,
        ) = gen_data()
        # 再次调用 gen_data 函数获取另一组三个张量，并分别赋值给 input1, index1, value1 变量
        (
            input1,
            index1,
            value1,
        ) = gen_data()
        # 使用 _unsafe_hacked_twin 函数对 input 执行索引赋值操作，不累积结果
        out1 = torch.ops.aten._unsafe_index_put.hacked_twin(
            input, [index], value, accumulate=False
        )
        # 使用 torch 自带的 index_put 函数对 input1 执行索引赋值操作，不累积结果
        out2 = torch.index_put(input1, [index1], value1, accumulate=False)
        # 断言 out1 和 out2 的结果应当相等
        self.assertEqual(out1, out2)

        # 使用 _unsafe_index.Tensor_hacked_twin 函数对 input 执行索引赋值操作，不累积结果
        torch.ops.aten._unsafe_index.Tensor_hacked_twin(input, [index])
        # 使用 torch 自带的 index_put 函数对 input1 执行索引赋值操作，不累积结果
        torch.index_put(input1, [index1], value1, accumulate=False)
        # 断言经过操作后 input 和 input1 应当相等

        # 定义一个嵌套函数 index_put_fn，用于对输入数据进行索引赋值操作
        def index_put_fn(input, index, value):
            return torch.ops.aten._unsafe_index_put(
                input, [index], value, accumulate=False
            )

        # 生成新的数据 input2, index2, value2，并分别赋值给对应的变量
        input2, index2, value2 = gen_data()
        # 使用 torch.jit.script 对 index_put_fn 函数进行脚本化
        script_index_put_fn = torch.jit.script(index_put_fn)
        # 分别计算使用 index_put_fn 和 script_index_put_fn 函数对 input2 进行操作后的结果
        expect = index_put_fn(input2.clone(), index2, value2)
        actual = script_index_put_fn(input2.clone(), index2, value2)
        # 断言 expect 和 actual 的结果应当相等
        self.assertEqual(expect, actual)

        # 定义一个嵌套函数 index_fn，用于对输入数据进行索引赋值操作
        def index_fn(input, index, value):
            return torch.ops.aten._unsafe_index_put(
                input, [index], value, accumulate=False
            )

        # 使用 torch.jit.script 对 index_fn 函数进行脚本化
        script_index_fn = torch.jit.script(index_fn)
        # 分别计算使用 index_fn 和 script_index_fn 函数对 input2 进行操作后的结果
        expect = index_fn(input2.clone(), index2, value2)
        actual = script_index_fn(input2.clone(), index2, value2)
        # 断言 expect 和 actual 的结果应当相等
        self.assertEqual(expect, actual)
    # 定义一个测试函数，用于测试导出操作名的接口
    def test_export_opnames_interface(self):
        # 定义一个 TorchScript 接口 OneTwoModule，包含两个方法和一个 forward 方法
        @torch.jit.interface
        class OneTwoModule(nn.Module):
            # 方法 one 接收两个张量参数并返回张量
            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                pass

            # 方法 two 接收一个张量参数并返回张量
            def two(self, x: torch.Tensor) -> torch.Tensor:
                pass

            # forward 方法接收一个张量参数并返回张量
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                pass

        # 定义一个实现了接口 OneTwoModule 的 FooMod 类
        class FooMod(nn.Module):
            # 实现接口方法 one，返回输入张量的和
            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

            # 实现接口方法 two，返回输入张量的两倍
            def two(self, x: torch.Tensor) -> torch.Tensor:
                return 2 * x

            # 实现接口方法 forward，先调用 two 方法，再调用 one 方法
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.one(self.two(x), x)

        # 定义一个实现了接口 OneTwoModule 的 BarMod 类
        class BarMod(nn.Module):
            # 实现接口方法 one，返回输入张量的乘积
            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x * y

            # 实现接口方法 two，返回输入张量的倒数
            def two(self, x: torch.Tensor) -> torch.Tensor:
                return 2 / x

            # 实现接口方法 forward，先调用 one 方法，再调用 two 方法
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.two(self.one(x, x))

        # 将 OneTwoModule 接口注册为全局对象
        make_global(OneTwoModule)

        # 定义一个包含 OneTwoModule 类型成员 sub 的类 M
        class M(nn.Module):
            sub: OneTwoModule

            def __init__(self):
                super().__init__()
                self.sub = BarMod()  # 初始化 sub 为 BarMod 类的实例

            # 实现 forward 方法，调用 sub 对象的 forward 方法
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.sub.forward(x)

        # 定义一个函数 use_module_interface，接收一个 OneTwoModule 类型的列表和一个张量参数 x
        def use_module_interface(mod_list: List[OneTwoModule], x: torch.Tensor):
            # 调用列表中第一个元素的 forward 方法，并将结果与列表中第二个元素的 forward 方法结果相加返回
            return mod_list[0].forward(x) + mod_list[1].forward(x)

        # 启用移动端接口调用导出
        torch._C._enable_mobile_interface_call_export()

        # 使用 TorchScript 对象 script 化 M 类的实例
        scripted_M_mod = torch.jit.script(M())

        # 断言导出的操作名集合中包含指定的运算名
        self.assertTrue(
            {"aten::mul.Scalar", "aten::mul.Tensor", "aten::reciprocal"}.issubset(
                set(torch.jit.export_opnames(scripted_M_mod))
            )
        )

        # 将 scripted_M_mod 对象的 sub 成员替换为 FooMod 类的 TorchScript 对象
        scripted_M_mod.sub = torch.jit.script(FooMod())

        # 断言导出的操作名集合中包含指定的运算名
        self.assertTrue(
            {"aten::add.Tensor", "aten::mul.Scalar"}.issubset(
                set(torch.jit.export_opnames(scripted_M_mod))
            )
        )

    # 定义一个测试函数，测试包含无穷大的数学运算
    def test_math_inf(self):
        # 导入 math 模块中的 inf 常量
        from math import inf

        # 定义一个函数 foo，返回无穷大
        def foo():
            return inf

        # 使用 self.checkScript 方法检查 foo 函数
        self.checkScript(foo, ())

    # 定义一个测试函数，测试列表字面量的推断
    def test_list_literal_infer(self):
        # 定义一个函数 expects_intlist，接收一个整数列表参数 x
        def expects_intlist(x: List[int]):
            # 在列表末尾添加一个整数 3
            x.append(3)
            return x

        # 定义一个函数 foo，调用 expects_intlist 函数并传入一个空列表
        def foo():
            return expects_intlist([])

        # 使用 self.checkScript 方法检查 foo 函数
        self.checkScript(foo, ())

        # 定义一个函数 annotated_list_fail，调用 expects_intlist 函数并传入一个标注为张量列表的空列表
        def annotated_list_fail():
            return expects_intlist(torch.jit.annotate([], List[Tensor]))  # noqa: F821

        # 使用 self.assertRaises 方法验证对 annotated_list_fail 函数 TorchScript 化时抛出运行时错误
        with self.assertRaises(RuntimeError):
            torch.jit.script(annotated_list_fail)

        # 定义一个函数 non_temporary_fail，创建一个空列表 a，调用 expects_intlist 函数并传入列表 a
        def non_temporary_fail():
            a = []
            return expects_intlist(a)

        # 使用 self.assertRaises 方法验证对 non_temporary_fail 函数 TorchScript 化时抛出运行时错误
        with self.assertRaises(RuntimeError):
            torch.jit.script(non_temporary_fail)

        # 使用 TorchScript 对象 script 化 test_return 函数
        @torch.jit.script
        def test_return():
            return []

        # 使用 FileCheck 对象检查 test_return 函数的图是否包含特定的字符串
        FileCheck().check("Tensor[] = prim::ListConstruct").run(test_return.graph)
    def test_legacy_tensor_constructor(self):
        # 测试 PyObject 的重载

        def test_all_dtypes():
            # 返回包含不同数据类型的张量
            return (
                torch.BoolTensor([2]),     # 布尔型张量
                torch.LongTensor([3]),     # 长整型张量
                torch.ByteTensor([4]),     # 字节型张量
                torch.CharTensor([5]),     # 字符型张量
                torch.DoubleTensor([6]),   # 双精度浮点型张量
                torch.FloatTensor([7]),    # 浮点型张量
                torch.IntTensor([8]),      # 整型张量
                torch.ShortTensor([1]),    # 短整型张量
                torch.HalfTensor([1]),     # 半精度浮点型张量
            )

        self.checkScript(test_all_dtypes, ())

        # 现在测试空张量的重载
        def empty_overload():
            return torch.LongTensor(2, 3, 4)  # 返回一个指定维度的长整型张量

        eager = empty_overload()
        jit = torch.jit.script(empty_overload)()
        eager[:] = 1
        jit[:] = 1
        self.assertEqual(eager, jit)

        def no_inputs():
            return torch.DoubleTensor()  # 返回一个双精度浮点型张量

        self.checkScript(no_inputs, ())

        # 错误的函数签名
        def multiple_args():
            return torch.LongTensor(1, [2])  # 期望多个位置参数但给出了单个参数列表

        with self.assertRaisesRegex(
            RuntimeError, "multiple positional arguments that were not all integers"
        ):
            torch.jit.script(multiple_args)

        # 错误的关键字参数
        def bad_kwarg():
            return torch.LongTensor(hello="1")  # 不支持关键字参数 "hello"

        with self.assertRaisesRegex(RuntimeError, "hello"):
            torch.jit.script(bad_kwarg)

    def test_broadcasting_list(self):
        """
        测试 BroadcastingList 和 torch.nn._size_N_t 的别名
        """
        from torch._jit_internal import BroadcastingList2
        from torch.nn.common_types import _size_2_t

        def sum_i(x: _size_2_t) -> int:
            return x[0] + x[1]  # 返回两个元素的整数和

        def sum_f(x: BroadcastingList2[float]) -> float:
            return x[0] + x[1]  # 返回两个元素的浮点数和

        self.assertTrue(torch.jit.script(sum_i)(4) == 8)
        self.assertTrue(torch.jit.script(sum_f)(4.5) == 9.0)

    def test_parse_ir_annotate(self):
        ir = """
        graph():
          %3 : int[] = prim::Constant[value=annotate(List[int], [])]()
          return (%3)
        """
        graph = torch._C.parse_ir(ir, True)
        func = torch._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret == [])  # 确保解析后的图返回空列表

    def test_parse_ir_single_element_tensor_positive(self):
        ir = """
        graph():
          %7 : Long(1, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value={0}]()
          return (%7)
        """
        graph = torch._C.parse_ir(ir, True)
        func = torch._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret.numel() == 1)  # 确保返回的张量只有一个元素
        self.assertTrue(len(ret.size()) == 1)  # 确保返回的张量是一维的
    # 定义一个单元测试方法，测试解析包含单个元素张量的 IR（中间表示）代码的情况，验证其负面情况
    def test_parse_ir_single_element_tensor_negative(self):
        # 定义包含 IR 代码的字符串
        ir = """
        graph():
          %7 : Long(1, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value={-17}]()
          return (%7)
        """
        # 解析 IR 代码，生成计算图对象
        graph = torch._C.parse_ir(ir, True)
        # 根据计算图创建一个函数对象
        func = torch._C._create_function_from_graph("forward", graph)
        # 执行函数得到结果
        ret = func()
        # 断言返回的张量元素数量为 1
        self.assertTrue(ret.numel() == 1)
        # 断言返回的张量维度数为 1
        self.assertTrue(len(ret.size()) == 1)

    # 定义一个单元测试方法，测试脚本函数上多个装饰器的情况
    def test_script_many_decorators(self):
        # 定义一个空操作的装饰器函数
        def no_op_decorator(f):
            return f

        # 使用多个装饰器装饰的函数定义
        @no_op_decorator
        @no_op_decorator
        @no_op_decorator
        @no_op_decorator
        @no_op_decorator
        def foo(x, dim: int):
            return x.unsqueeze(dim)

        # 生成一个随机张量 x
        x = torch.randn(
            1,
        )
        # 使用函数 foo 对 x 进行操作得到期望结果
        expected = foo(x, 0)
        # 对函数 foo 进行脚本化
        scripted = torch.jit.script(foo)
        # 使用脚本化后的函数执行得到实际结果
        actual = scripted(x, 0)
        # 使用 PyTorch 的测试工具断言期望结果与实际结果接近
        torch.testing.assert_close(expected, actual)

    # 定义一个单元测试方法，测试在多种数据类型下进行幂运算的情况
    @unittest.skipIf(not RUN_CUDA_HALF, "need CUDA half support")
    def test_pow_multiple_dtype(self):
        # 定义一个函数，接受一个 torch.Tensor p 和一个 gamma 参数，并返回计算结果
        def fn(p: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
            # 对 p 进行 sigmoid 操作
            p = torch.sigmoid(p)
            # 计算 p 的 gamma 次幂
            result = p**gamma
            return result

        # 在 CUDA 半精度数据类型和设备上生成一个随机张量 x
        x = torch.rand((2, 2), dtype=torch.half, device="cuda")
        # 使用函数 fn 计算 x 的结果作为参考结果
        ref = fn(x)
        # 对函数 fn 进行脚本化
        script_fn = torch.jit.script(fn)
        # 多次使用脚本化后的函数计算结果
        for i in range(4):
            res = script_fn(x)
        # 使用 PyTorch 的测试工具断言参考结果与最终结果相等
        self.assertEqual(ref, res)

    # 定义一个单元测试方法，测试获取操作顺序的情况
    def test_jit_get_operation_order(self):
        # 查看 GitHub 上的一个问题链接
        # 根据操作符的注册顺序，可以获取不同的操作顺序
        # 这里验证 _jit_get_operation 始终将 aten 操作放在首位的顺序
        # 确保选择标量重载而不是复杂重载
        ret = torch.ops.aten.add(4, 3.3)
        self.assertFalse("complex" in str(ret.dtype))

        # 获取 "aten::add" 操作的操作对象和重载名称列表
        op, override_names = torch._C._jit_get_operation("aten::add")
        print(override_names)
        # 找出复杂重载的索引
        complex_indices = [
            i for i, name in enumerate(override_names) if name == "complex"
        ]
        # 找出标量重载的索引
        Scalar_indices = [
            i for i, name in enumerate(override_names) if name == "Scalar"
        ]

        # 断言复杂重载的索引比标量重载的索引大
        self.assertTrue(len(complex_indices) > 0)
        self.assertTrue(len(Scalar_indices) > 0)
        self.assertTrue(complex_indices[0] > Scalar_indices[0])
```