# `.\pytorch\test\jit\test_typing.py`

```py
# Owner(s): ["oncall: jit"]

# 导入标准库模块
import os
import sys
# 导入命名元组和类型提示相关模块
from collections import namedtuple
from typing import Dict, List, NamedTuple, Tuple

# 导入 PyTorch 相关模块
import torch
# 导入 PyTorch 内部测试相关模块
from torch.testing._internal.common_utils import IS_WINDOWS
from torch.testing._internal.jit_utils import JitTestCase, make_global

# 将 test/ 目录下的文件加入模块搜索路径
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 如果直接运行该脚本，则抛出异常，提示正确的运行方式
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类 TestTyping，继承自 JitTestCase
class TestTyping(JitTestCase):

    # 定义测试函数 test_dict_in_not_in
    def test_dict_in_not_in(self):

        # 定义内部函数 test_in_dict，接受字典参数 x，返回是否存在 "hi" 键
        def test_in_dict(x):
            # type: (Dict[str, int]) -> bool
            return "hi" in x

        # 使用 JitTestCase 提供的方法 checkScript，对 test_in_dict 进行脚本化测试
        self.checkScript(test_in_dict, ({"hi": 2, "bye": 3},))
        self.checkScript(test_in_dict, ({"bye": 3},))

        # 检查执行顺序的函数
        @torch.jit.script
        def a():
            print("a")
            return 3

        @torch.jit.script
        def b():
            print("b")
            return {3: 2, 4: 1}

        @torch.jit.script
        def fn():
            return a() in b()

        # 使用 capture_stdout 方法捕获标准输出
        with self.capture_stdout() as captured:
            self.assertTrue(fn())
        # 在非 Windows 环境下，验证捕获的标准输出内容
        if not IS_WINDOWS:
            # no stdout capturing on windows
            self.assertEqual(captured[0], "a\nb\n")

        # 定义内部函数 test_not_in_dict，接受字典参数 a，返回是否不存在 "hello" 键
        def test_not_in_dict(a):
            # type: (Dict[str, int]) -> bool
            if "hello" not in a:
                return False
            else:
                return True

        # 使用 JitTestCase 提供的方法 checkScript，对 test_not_in_dict 进行脚本化测试
        self.checkScript(test_not_in_dict, ({"hello": 1, "world": 2},))
        self.checkScript(test_not_in_dict, ({"world": 2},))

        # 定义内部函数 test_dict_tensor_key，接受字典参数 a 和张量参数 t，返回 t 是否存在于字典的键中
        def test_dict_tensor_key(a, t):
            # type: (Dict[Tensor, int], Tensor) -> bool
            if t in a:
                return True
            else:
                return False

        # 创建两个张量作为字典的键
        inp1 = torch.tensor(3)
        inp2 = torch.tensor(5)
        # 创建字典 dict_a，包含 inp1 和 inp2 作为键
        dict_a = {inp1: 1, inp2: 3}
        # 使用 JitTestCase 提供的方法 checkScript，对 test_dict_tensor_key 进行脚本化测试
        self.checkScript(test_dict_tensor_key, (dict_a, torch.tensor(4)))
        self.checkScript(test_dict_tensor_key, (dict_a, torch.tensor(3)))
        self.checkScript(test_dict_tensor_key, (dict_a, inp1))
        self.checkScript(test_dict_tensor_key, (dict_a, inp2))

    # 定义测试函数 test_list_type_refinement_annotation_element_mismatch
    def test_list_type_refinement_annotation_element_mismatch(self):
        # 定义函数 fn
        def fn():
            # 使用类型注解指定列表 l 的类型为 List[int]，但实际包含了不匹配类型的元素
            l: List[int] = [1, 2, "foo", 3]
            return l

        # 使用 assertRaisesRegex 方法验证运行时异常是否包含特定的错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            "List type annotation"
            r" `List\[int\]` did not match the "
            "types of the given list elements",
        ):
            # 对函数 fn 进行脚本化测试，预期会抛出 RuntimeError 异常
            torch.jit.script(fn)
    def test_dict_type_refinement_annotation_key_mismatch(self):
        # 定义测试函数
        def fn():
            # 创建包含不同类型元素的列表
            l1 = [1, 2, "foo", 3]
            l2 = ["foo", "bar", "baz", "qux"]
            # 使用zip函数将两个列表合并成字典，但键类型不匹配类型注解
            d: Dict[int, str] = dict(zip(l1, l2))
            return d

        # 断言捕获运行时错误，并验证错误消息是否匹配预期
        with self.assertRaisesRegex(
            RuntimeError,
            "Dicts may only "
            "contain homogeneous keys, but the "
            "type of the first generated key "
            r"was Union\[int, str\]",
        ):
            # 对fn函数进行脚本化编译
            torch.jit.script(fn)

    def test_dict_type_refinement_annotation_value_mismatch(self):
        # 定义测试函数
        def fn():
            # 创建包含不同类型元素的列表
            l1 = ["foo", "bar", "baz", "qux"]
            l2 = [1, 2, "foo", 3]
            # 使用zip函数将两个列表合并成字典，但值类型不匹配类型注解
            d: Dict[str, int] = dict(zip(l1, l2))
            return d

        # 断言捕获运行时错误，并验证错误消息是否匹配预期
        with self.assertRaisesRegex(
            RuntimeError,
            "Dict type annotation"
            r" `Dict\[str, int\]` did not match"
            " the type of an actual value type"
            r" `Union\[int, str\]`",
        ):
            # 对fn函数进行脚本化编译
            torch.jit.script(fn)

    def test_dict_invalid_annotations(self):
        # 检查值类型注解无效性
        def wrong_value_type(dictionary: Dict[str, torch.jit.ScriptModule]):
            return

        # 断言捕获值错误的异常，并验证错误消息是否匹配预期
        with self.assertRaisesRegex(ValueError, "Unknown type annotation"):
            # 对错误的值类型注解进行脚本化编译
            torch.jit.script(wrong_value_type)

        # 检查键类型注解无效性
        def wrong_key_type(dictionary: Dict[torch.jit.ScriptModule, str]):
            return

        # 断言捕获键错误的异常，并验证错误消息是否匹配预期
        with self.assertRaisesRegex(ValueError, "Unknown type annotation"):
            # 对错误的键类型注解进行脚本化编译
            torch.jit.script(wrong_key_type)

        # 检查键和值类型注解无效性
        def wrong_key_value_type(
            dictionary: Dict[torch.jit.ScriptModule, torch.jit.ScriptModule]
        ):
            return

        # 断言捕获键值类型错误的异常，并验证错误消息是否匹配预期
        with self.assertRaisesRegex(ValueError, "Unknown type annotation"):
            # 对错误的键值类型注解进行脚本化编译
            torch.jit.script(wrong_key_value_type)

    def test_tuple_specialization(self):
        # 使用torch.jit.script装饰器声明脚本函数
        @torch.jit.script
        def f(t, s):
            # type: (Tuple[Tensor, Tuple[int, Tensor]], str) -> Tensor
            # 解构元组t，获取其元素
            x, t2 = t
            _, y = t2
            # 返回两个张量的和
            return x + y

        # 创建一个复杂的元组作为输入
        t = (
            torch.randn(2, 2),
            (1, torch.randn(2, 2)),
        )
        # 调用脚本函数
        f(t, "hi")
        # 获取函数关联的图形对象
        graph = f.graph_for(t, "hi")
        # 获取输入参数的类型列表
        input_types = list(next(graph.inputs()).type().elements())
        w = input_types[0]
        # 断言输入类型的种类是否为张量类型
        self.assertEqual(input_types[0].kind(), "TensorType")
        # 断言输入类型的第二个元素的种类是否为张量类型
        self.assertEqual(input_types[1].elements()[1].kind(), "TensorType")

    def test_tuple_io(self):
        # 定义一个处理元组的函数
        def stuff(x):
            # type: (Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]
            # 解构元组x，交换其元素顺序
            a, b = x
            return b, a

        # 创建一个包含两个张量的元组
        a = (torch.rand(3), torch.rand(3))
        # 使用自定义方法验证脚本函数
        self.checkScript(stuff, (a,))
    # 定义一个测试函数 test_tuple_keyword
    def test_tuple_keyword(self):
        # 定义内部函数 bar
        def bar():
            # 创建一个元组 (1, 2)，此处 noqa: C409 表示忽略某种类型的静态分析警告
            f = tuple((1, 2))  # noqa: C409
            return f
        
        # 使用自定义函数检查 bar 的脚本化结果，并传入空元组作为参数
        self.checkScript(bar, ())

        # 定义内部函数 foo
        def foo():
            # 返回一个元组，但是使用了错误的语法 tuple(1, 2)，会抛出异常
            return tuple(1, 2)
        
        # 使用自定义函数检查 foo 的脚本化结果，预期会抛出异常并包含 "1 argument" 的错误信息
        self.checkScriptRaisesRegex(foo, (), Exception, "1 argument")

        # 定义内部函数 cant_infer_size
        def cant_infer_size():
            # 返回一个元组，其中包含一个列表 [1, 2, 3]，此处 noqa: C409 表示忽略某种类型的静态分析警告
            return tuple([1, 2, 3])  # noqa: C409
        
        # 使用 assertRaisesRegex 上下文管理器验证 torch.jit.script(cant_infer_size) 是否会抛出异常，并包含 "cannot statically infer the expected" 的错误信息
        with self.assertRaisesRegex(Exception, "cannot statically infer the expected"):
            torch.jit.script(cant_infer_size)

    # 定义一个测试函数 test_tuple_create_return
    def test_tuple_create_return(self):
        # 定义内部函数 stuff2，接受一个整数参数 x
        def stuff2(x):
            # type: (int) -> Tuple[Tensor, Tensor]
            # 创建一个包含两个 Tensor 对象的元组 a，分别为全 1 和全 0 的 Tensor
            a = (torch.ones(x), torch.zeros(x))
            return a
        
        # 使用自定义函数检查 stuff2 的脚本化结果，并传入参数 (3,)，表示一个含有一个整数 3 的元组
        self.checkScript(stuff2, (3,))

    # 定义一个测试函数 test_list_io
    def test_list_io(self):
        # 定义内部函数 stuff3，接受一个列表参数 x，其中元素为整数
        def stuff3(x):
            # type: (List[int]) -> Tuple[Tensor, List[int]]
            # 返回一个包含一个 Tensor 对象和参数 x 的列表的元组
            return torch.ones(x), x
        
        # 使用自定义函数检查 stuff3 的脚本化结果，并传入参数 ([3, 2],)，表示一个包含一个整数列表 [3, 2] 的元组
        self.checkScript(stuff3, ([3, 2],))

    # 定义一个测试函数 test_bool_list_io
    def test_bool_list_io(self):
        # 使用 torch.jit.script 装饰器定义内部函数 stuff4，接受一个布尔值列表参数 x
        @torch.jit.script
        def stuff4(x):
            # type: (List[bool]) -> Tuple[List[bool], List[bool], List[List[bool]]]
            # 返回一个元组，包含参数 x 自身、固定的布尔值列表 [True, False]、以及包含一个布尔值列表 [True] 的列表
            return x, [True, False], [[True]]

        # 调用 stuff4 函数，并解包其返回值到 li_1、li_2、li_3
        li_1, li_2, li_3 = stuff4([True])
        # 将 li_3 中的唯一元素解包到 li_3 自身，实质上将其变为一个布尔值列表 [True]
        li_3 = li_3[0]
        # 对于 [li_1, li_2, li_3] 中的每个列表，验证其第一个元素是否为布尔值类型
        for li in [li_1, li_2, li_3]:
            self.assertTrue(type(li[0]) == bool)

    # 定义一个测试函数 test_nested_list
    def test_nested_list(self):
        # 定义内部函数 foo，接受一个元组 z，包含一个整数和一个整数列表的列表
        def foo(z):
            # type: (Tuple[int, List[List[int]]]) -> int
            # 解包元组 z，分别赋值给变量 x 和 y
            x, y = z
            # 返回 y 列表中的第一个子列表的第二个元素
            return y[0][1]

        # 使用自定义函数检查 foo 的脚本化结果，并传入参数 ((1, [[1, 2], [3, 4]]),)，表示一个包含元组 (1, [[1, 2], [3, 4]]) 的元组
        self.checkScript(foo, ((1, [[1, 2], [3, 4]]),))

    # 定义一个测试函数 test_list_sum
    def test_list_sum(self):
        # 定义函数 fn，接受一个整数列表参数 x，返回其所有元素之和
        def fn(x: List[int]) -> int:
            return sum(x)

        # 定义函数 fn1，接受一个浮点数列表参数 x，返回其所有元素之和
        def fn1(x: List[float]):
            return sum(x)

        # 定义函数 fn2，接受一个布尔值列表参数 x，返回其所有元素之和
        def fn2(x: List[bool]):
            return sum(x)

        # 使用自定义函数检查 fn 的脚本化结果，并传入参数 ([1, 2, 3],)，表示一个包含整数列表 [1, 2, 3] 的元组
        self.checkScript(fn, ([1, 2, 3],))
        # 使用自定义函数检查 fn1 的脚本化结果，并传入参数 ([1.0, 2.0, 3.0],)，表示一个包含浮点数列表 [1.0, 2.0, 3.0] 的元组
        self.checkScript(fn1, ([1.0, 2.0, 3.0],))
        # 使用自定义函数检查 fn1 的脚本化结果，并传入参数 ([1, 2.8, 3],)，表示一个包含混合类型列表 [1, 2.8, 3] 的元组
        self.checkScript(fn1, ([1, 2.8, 3],))
        # 使用自定义函数检查 fn2 的脚本化结果，并传入参数 ([True, False, False],)，表示一个包含布尔值列表 [True, False, False] 的元组
        self.checkScript(fn2, ([True, False, False],))
        # 使用自定义函数检查 fn2 的脚本化结果，并传入参数 ([False, False, False],)，表示一个包含布尔值列表 [False, False, False] 的元组
        self.checkScript(fn2, ([False, False, False],))
        # 使用自定义函数检查 fn2 的脚本化结果，并传入参数 ([0, 1, 1, 0],)，表示一个包含整数列表 [0, 1, 1, 0] 的元组
        self.checkScript(fn2, ([0, 1, 1, 0],))

    # 定义一个测试函数 test_list_unification
    def test_list_unification(self):
        # 定义函数 fn，返回一个包含整数和 None 的列表
        def fn():
            return [1, None, 2]

        # 定义函数 fn2，接受一个参数 x，返回一个包含 Tensor 对象、None 和 x 的列表
        def fn2(x):
            return [torch.ones(2, 2), None, x]

        # 使用自定义函数检查 fn 的脚本化结果，并传入空元组作为参数
        self.checkScript(fn, [])
        # 使用自定义函数检查 fn2 的脚本化结果，并传入参数 (torch.ones(2, 2),)，表示一个包含一个 2x2 全 1 Tensor 的元组
        self.checkScript(fn2, (torch.ones(2, 2),))

    # 定义一个函数 get_sum_list_fn
    # 用于避免在多个测试中重复定义 sum_list
    def get_sum_list_fn(self):
        # 定义内部
    def test_sum_list_literal(self):
        # 定义一个函数用于计算列表 [1, 2, 3, 4, 5] 的总和
        def sum_list():
            # 初始化一个变量 sum 用于存储总和
            sum = 0
            # 遍历列表 [1, 2, 3, 4, 5] 中的每个元素并累加到 sum 中
            for i in [1, 2, 3, 4, 5]:
                sum += i

            return sum

        # 调用 self.checkScript 方法检查 sum_list 函数的行为
        self.checkScript(sum_list, ())

    def test_sum_list_wrong_type(self):
        # 使用 self.assertRaisesRegex 检查运行时异常信息是否包含 "'int' object is not iterable"
        with self.assertRaisesRegex(RuntimeError, "'int' object is not iterable"):

            # 使用 @torch.jit.script 装饰器将 sum_list 函数编译成 Torch 脚本
            @torch.jit.script
            def sum_list(a):
                # type: (int) -> int
                # 初始化一个变量 sum 用于存储总和
                sum = 0
                # 遍历参数 a 中的每个元素并累加到 sum 中
                for i in a:  # noqa: T484
                    sum += i

                return sum

            # 调用 sum_list 函数并传入整数参数 1，期望触发异常
            sum_list(1)

    def test_list_iterables(self):
        # 使用 self.assertRaisesRegex 检查运行时异常信息是否包含 "List of iterables is not supported currently"
        with self.assertRaisesRegex(
            RuntimeError, "List of iterables is not supported currently"
        ):
            # 创建一个 Torch 编译单元 cu，内容定义了一个函数 list_iterables，但此处未调用

    def test_for_in_string(self):
        # 定义一个函数 test_strings 用于将输入字符串 x 反转后返回
        def test_strings(x):
            # type: (str) -> str
            # 初始化一个空字符串 reverse 用于存储反转后的结果
            reverse = ""
            # 遍历字符串 x 中的每个字符，并将其逐个拼接到 reverse 的前面
            for c in x:
                reverse = c + reverse
            return reverse

        # 调用 self.checkScript 方法检查 test_strings 函数的行为，传入参数 "hello" 和 ""
        self.checkScript(test_strings, ("hello",))
        self.checkScript(test_strings, ("",))

        # 定义一个函数 test_list_strings 用于将输入列表 x 中的所有字符串连接成一个结果字符串后返回
        def test_list_strings(x):
            # type: (List[str]) -> str
            # 初始化一个空字符串 result 用于存储连接后的结果
            result = ""
            # 遍历列表 x 中的每个子字符串 sub_str，并将其逐个拼接到 result 中
            for sub_str in x:
                result += sub_str
            return result

        # 调用 self.checkScript 方法检查 test_list_strings 函数的行为，传入参数 ["hello", "world"] 和 ["hello", " ", "world", ""]
        self.checkScript(test_list_strings, (["hello", "world"],))
        self.checkScript(test_list_strings, (["hello", " ", "world", ""],))

    def test_for_in_dict(self):
        # 定义一个函数 test_dicts 用于计算字典 x 中所有值的总和并返回
        def test_dicts(x):
            # type: (Dict[str, int]) -> int
            # 初始化一个变量 sum 用于存储总和
            sum = 0
            # 遍历字典 x 中的每个键 key，并将对应值 x[key] 累加到 sum 中
            for key in x:
                sum += x[key]
            return sum

        # 调用 self.checkScript 方法检查 test_dicts 函数的行为，传入参数 {"a": 1, "b": 2, "c": 3}
        self.checkScript(test_dicts, ({"a": 1, "b": 2, "c": 3},))

        # 定义一个函数 test_dict_keys_values 用于分别连接字典 x 的所有键和计算所有值的总和后返回
        def test_dict_keys_values(x):
            # type: (Dict[str, int]) -> Tuple[str, int]
            # 初始化一个空字符串 key_str 用于存储所有键连接后的结果
            key_str = ""
            # 初始化一个变量 sum 用于存储总和
            sum = 0
            # 遍历字典 x 中的所有键 key，并将其逐个拼接到 key_str 中
            for key in x.keys():
                key_str += key
            # 遍历字典 x 中的所有值 val，并累加到 sum 中
            for val in x.values():
                sum += val
            return key_str, sum

        # 调用 self.checkScript 方法检查 test_dict_keys_values 函数的行为，传入参数 {"a": 1, "b": 2, "c": 3}
        self.checkScript(test_dicts, ({"a": 1, "b": 2, "c": 3},))

    def test_for_tuple_unpack(self):
        # 定义一个函数 for_tuple_unpack 用于依次将输入列表 x 中的每个子列表 [3, 4], [5, 6], [7, 8] 中的元素拆解并累加到 x 和 y 中，最后返回
        def for_tuple_unpack(x, y):
            for i, j in [[3, 4], [5, 6], [7, 8]]:
                x += i
                y += j
            return x, y

        # 调用 self.checkScript 方法检查 for_tuple_unpack 函数的行为，传入参数 torch.tensor(3), torch.tensor(5)

        # 定义一个函数 nested_tuple_unpack 用于同时遍历两个列表 x 和 y，将其元素进行拆解和累加后返回总和
        def nested_tuple_unpack(x, y):
            # type: (List[int], List[int]) -> int
            # 初始化一个变量 sum 用于存储总和
            sum = 0
            # 使用 zip 同时遍历列表 x 和 y，以及 x 的枚举值，并将每个元素进行拆解和累加后加入 sum 中
            for i, (j, k), v in zip(x, enumerate(x), y):
                sum += i + j + k + v
            return sum

        # 调用 self.checkScript 方法检查 nested_tuple_unpack 函数的行为，传入参数 ([1, 3, 5], [2, 4, 6])
    def test_dict_comprehension(self):
        # 定义函数 fn，使用字典推导式生成一个从整数到字符的映射字典，范围是 [0, 3]
        def fn():
            return {i: chr(i + 65) for i in range(4)}

        # 使用 self.checkScript 方法验证 fn 函数，不传入任何参数
        self.checkScript(fn, ())

    def test_dict_comprehension_with_type_annotation(self):
        # 定义函数 fn，使用带类型注解的字典推导式生成一个从整数到字符的映射字典，范围是 [0, 3]
        def fn():
            d: Dict[int, str] = {i: chr(i + 65) for i in range(4)}
            return d

        # 使用 self.checkScript 方法验证 fn 函数，不传入任何参数
        self.checkScript(fn, ())

        # 使用 assertRaisesRegex 检测运行时错误，并在错误信息中包含特定字符串
        with self.assertRaisesRegex(RuntimeError, ""):
            # 使用 assertRaisesRegex 检测断言错误，并在错误信息中包含特定字符串
            with self.assertRaisesRegex(
                AssertionError,
                "Expected Dict "
                "type annotation for dict "
                "comprehension, found "
                "Tuple[int, str]",
            ):
                # 使用 torch.jit.script 装饰器，定义一个脚本函数 fn，其类型注解不正确
                @torch.jit.script
                def fn():
                    d: Tuple[int, str] = {i: chr(i + 65) for i in range(4)}
                    return d

    def test_dict_comprehension_scope(self):
        # 定义函数 comprehension_can_access_outer_scope_variables
        def comprehension_can_access_outer_scope_variables():
            # 定义列表 lst
            lst = ["foo", "bar", "baz"]
            # 使用字典推导式生成一个从字符串到其长度的映射字典
            return {l: len(l) for l in lst}

        # 使用 self.checkScript 方法验证 comprehension_can_access_outer_scope_variables 函数，不传入任何参数
        self.checkScript(comprehension_can_access_outer_scope_variables, ())

        # 使用 assertRaisesRegex 检测运行时错误，并在错误信息中包含特定字符串
        with self.assertRaisesRegex(RuntimeError, "undefined value i"):
            # 使用 torch.jit.script 装饰器，定义一个脚本函数 outer_scope_cannot_access_comprehension_variables
            def outer_scope_cannot_access_comprehension_variables():
                # 使用字典推导式生成一个从整数到字符的映射字典
                d = {i: chr(i + 65) for i in range(4)}
                # 尝试访问未定义的变量 i
                i = i + 1  # noqa: F821

    def test_for_tuple_assign(self):
        # 定义函数 test_simple_assign，参数为元组 (x)
        def test_simple_assign(x):
            # type: (Tuple[int, float]) -> float
            # 初始化 sum 变量为浮点数 0.0
            sum = 0.0
            # 对元组 x 进行遍历，将每个元素转换为浮点数并累加到 sum 中
            for a in x:
                sum += float(a)
            return sum

        # 使用 self.checkScript 方法验证 test_simple_assign 函数，传入参数 (1, 2.5)

        self.checkScript(test_simple_assign, ((1, 2.5),))

        # 定义函数 test_tuple_assign，参数为元组 (x)
        def test_tuple_assign(x):
            # type: (Tuple[Tuple[int, int], Tuple[int, int]]) -> int
            # 初始化 sum 变量为整数 0
            sum = 0
            # 对元组 x 进行遍历，每次取出内部元组的第一个和第二个元素，累加到 sum 中
            for a in x:
                sum += a[0]
                sum += a[1]
            return sum

        # 使用 self.checkScript 方法验证 test_tuple_assign 函数，传入参数 (((1, 2), (4, 7)),)

        self.checkScript(test_tuple_assign, (((1, 2), (4, 7)),))

        # 定义函数 test_single_starred_lhs
        def test_single_starred_lhs():
            # 使用 assertRaisesRegex 检测运行时错误，并在错误信息中包含特定字符串
            with self.assertRaisesRegex(
                RuntimeError,
                "A Starred expression may only appear on the lhs within the presence"
                " of another non-starred expression",
            ):
                # 使用 torch.jit.CompilationUnit 构造一个 CompilationUnit 对象 cu
                cu = torch.jit.CompilationUnit(
                    """
                def single_starred_lhs(x):
                    a = (x, x, x)
                    *b, = a
                    return b
                """
                )

    def test_singleton_tuple_unpack(self):
        # 定义函数 foo，参数为元组 (a)
        def foo(a):
            # 对元组 (a,) 进行解包，将元素赋值给变量 b
            (b,) = (a,)
            return b + 1

        # 使用 self.checkScript 方法验证 foo 函数，传入参数 (torch.rand(3),)

        self.checkScript(foo, (torch.rand(3),))
    def test_tuple_assignments(self):
        def var_tuple_assign(x, y):
            # type: (Tuple[Tensor, Tensor], Tensor) -> Tensor
            # 解构元组 x 和 y，将元素分别赋值给 (a, b) 和 c
            (a, b), c = x, y
            return a + b + c

        tuple_inputs = (torch.randn(1, 4), torch.randn(3, 4))
        # 调用 self.checkScript 方法，验证 var_tuple_assign 函数的脚本化版本
        self.checkScript(var_tuple_assign, (tuple_inputs, torch.randn(3, 4)))

        def nested_tuple_assign(x, y, z):
            # type: (int, Tuple[int, Tuple[int, int]], Tuple[int, int]) -> int
            # 嵌套解构元组，将 x 赋值给 a，y 赋值给 (b, (c, d))，z 赋值给 (e, f)
            a, (b, (c, d)), (e, f) = x, y, z
            return a + b + c + d + e + f

        # 调用 self.checkScript 方法，验证 nested_tuple_assign 函数的脚本化版本
        self.checkScript(nested_tuple_assign, ((1, (2, (3, 4)), (5, 6))))

        def subscript_tuple_assign(a, x, i):
            # type: (List[int], Tensor, int) -> Tuple[int, Tensor, int]
            # 元组和列表的下标赋值，将 1 赋值给 a[i]，(2, 3) 赋值给 (x[i], b)
            a[i], (x[i], b) = 1, (2, 3)
            return a[i] + 1, x + 5, b

        self.checkScript(
            subscript_tuple_assign, ([12, 7, 9, 11], torch.tensor((3, 13, 17)), 0)
        )

        def star_tuple_assign():
            # type: () -> Tuple[int, int, Tuple[int, int], Tuple[int, int]]
            # 使用星号解构元组，将 1 赋值给 a，(2, 3, 4) 赋值给 (b, *c)，(5, 6) 赋值给 *d
            a, (b, *c), *d = 1, (2, 3, 4), 5, 6
            return a, b, c, d

        # 调用 self.checkScript 方法，验证 star_tuple_assign 函数的脚本化版本
        self.checkScript(star_tuple_assign, ())

        def subscript_tuple_augmented_assign(a):
            # type: (Tuple[int, int]) -> Tuple[int, int]
            # 元组的下标增强赋值操作，这里会引发 RuntimeError
            a[0] += 1
            return a

        # 使用 self.assertRaisesRegex 检查是否引发了 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "does not support augmented assign"):
            scripted_aug_assign = torch.jit.script(subscript_tuple_augmented_assign)

    def test_multiple_assign(self):
        def test():
            # 多重赋值，将 d 赋值给 b，将 (1, 1) 赋值给 a 和 f
            a = b, c = d, f = (1, 1)

            # 副作用，对 ten 进行张量操作
            ten = torch.tensor(1)
            ten1 = ten2 = ten.add_(1)

            # 顺序赋值，交换 x 和 y 的值
            x = 1
            y = 3
            x, y = y, x + y

            return a, b, c, d, f, ten, ten1, ten2, x, y

        # 调用 self.checkScript 方法，验证 test 函数的脚本化版本
        self.checkScript(test, ())

    def test_opt_opt_refinement(self):
        @torch.jit.script
        def test_unify(weight, bias):
            # type: (Optional[int], Optional[int]) -> Optional[int]
            # 使用 Optional 类型注解，根据 weight 和 bias 的值进行条件赋值
            if weight is not None:
                opt = None
            else:
                if bias is not None:
                    opt = 1
                else:
                    opt = None

            return opt

    def test_optional_refinement(self):
        @torch.jit.script
        def test_if_none_assignment(x):
            # type: (Optional[int]) -> int
            # 对可选类型进行赋值，如果 x 为 None，则将其赋值为 1
            if x is None:
                x = 1
            return x + 1

        # 调用 self.assertEqual 方法，验证 test_if_none_assignment 函数的返回值是否符合预期
        self.assertEqual(test_if_none_assignment(1), 2)
    def test_optional_conversion(self):
        @torch.jit.script
        def other_fn(x=None):
            # type: (Optional[int]) -> int
            # 如果输入参数 x 是可选的整数类型，返回其非空值
            return torch.jit._unwrap_optional(x)

        @torch.jit.script
        def fn(x):
            # type: (int) -> int
            # 调用 other_fn 函数处理输入参数 x
            return other_fn(x)

        # 断言 fn(2) 的返回值为 2
        self.assertEqual(fn(2), 2)

        @torch.jit.script
        def unify_to_optional(x):
            # type: (bool) -> Optional[int]
            # 根据布尔值 x 返回一个可选的整数值
            if x:
                a = None
            else:
                a = 2
            return a

        # 断言 unify_to_optional(True) 的返回值为 None
        self.assertEqual(unify_to_optional(True), None)
        # 断言 unify_to_optional(False) 的返回值为 2
        self.assertEqual(unify_to_optional(False), 2)

        @torch.jit.script
        def opt_list(x):
            # type: (Optional[List[float]]) -> int
            # 输入参数 x 是可选的浮点数列表，返回整数值 2
            return 2

        @torch.jit.script
        def broadcast_opt_list(x):
            # type: (Optional[BroadcastingList2[float]]) -> int
            # 输入参数 x 是可选的广播列表类型，返回整数值 2
            return 2

        @torch.jit.script
        def opt_list_tuple_caller(x):
            # type: (Tuple[float, float]) -> int
            # 调用 opt_list 和 broadcast_opt_list 处理输入参数 x
            return opt_list(x) + broadcast_opt_list(x)

        # 断言 opt_list_tuple_caller((2.0, 3.0)) 的返回值为 4
        self.assertEqual(opt_list_tuple_caller((2.0, 3.0)), 4)

    def test_optional_tuple(self):
        def fn(x=None):
            # type: (Optional[Tuple[int, int]]) -> Tuple[int, int]
            # 如果输入参数 x 是可选的整数对元组，返回该元组；否则返回默认的 (1, 2)
            if x is None:
                new_x = (1, 2)
            else:
                new_x = x
            return new_x

        # 调用 checkScript 函数检查 fn 函数在给定参数 (3, 4) 下的运行情况
        self.checkScript(fn, ((3, 4),))
        # 调用 checkScript 函数检查 fn 函数在没有参数的情况下的运行情况
        self.checkScript(fn, ())

    def test_namedtuple_redefine(self):
        global _1, _2
        # 定义全局变量 _1 和 _2 为两个具名元组类型
        _1 = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
        _2 = namedtuple("GoogLeNetOutputs", ["different"])

        # 使用 assertRaisesRegex 确保在以下代码块中会引发 RuntimeError 异常，异常信息中包含 "redefine"
        with self.assertRaisesRegex(RuntimeError, r"redefine"):

            @torch.jit.script
            def foo(x, y):
                # type: (_1, _2) -> _1
                # 函数 foo 接受两个具名元组类型参数 x 和 y，并返回第一个参数 x
                return x

    def test_namedtuple_py2(self):
        global _GoogLeNetOutputs  # see [local resolution in python]
        # 定义全局变量 _GoogLeNetOutputs 为一个具名元组类型
        _GoogLeNetOutputs = namedtuple(
            "GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"]
        )

        @torch.jit.script
        def foo(x):
            # type: (_GoogLeNetOutputs) -> _GoogLeNetOutputs
            # 函数 foo 接受一个具名元组类型参数 x，并返回该参数 x
            return x

        # 生成随机张量值作为具名元组的字段值
        vals = torch.rand(3), torch.rand(4), torch.rand(5)
        # 调用 foo 函数，传入具名元组作为参数，并断言返回值中各字段与 vals 中对应的值相等
        out = foo(
            _GoogLeNetOutputs(logits=vals[0], aux_logits2=vals[1], aux_logits1=vals[2])
        )
        self.assertEqual(out.logits, vals[0])
        self.assertEqual(out.aux_logits2, vals[1])
        self.assertEqual(out.aux_logits1, vals[2])
    # 定义一个全局变量 _GoogLeNetOutputs，用于在当前模块中共享命名元组
    global _GoogLeNetOutputs  # see [local resolution in python]
    # 创建一个命名元组 _GoogLeNetOutputs，包含三个字段 logits, aux_logits2, aux_logits1
    _GoogLeNetOutputs = namedtuple(
        "GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"]
    )

    # 使用 torch.jit.script 装饰器定义一个脚本函数 foo，接受一个 _GoogLeNetOutputs 类型参数并返回同样类型
    @torch.jit.script
    def foo(x):
        # type: (_GoogLeNetOutputs) -> _GoogLeNetOutputs
        return x

    # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并检查异常信息是否包含特定文本
    with self.assertRaisesRegex(
        RuntimeError, r"aka NamedTuple\(logits, aux_logits2, aux_logits1\)"
    ):
        # 调用 foo 函数并传入一个 _GoogLeNetOutputs 命名元组对象作为参数
        out = foo(_GoogLeNetOutputs(logits="3", aux_logits2="4", aux_logits1="5"))

    # 定义一个名为 _NamedTupleBadMemberType 的命名元组类，其中 f1 字段是 torch.Tensor 类型，f2 是 ABadForwardRefType 类型
    class _NamedTupleBadMemberType(NamedTuple):
        f1: torch.Tensor
        f2: "ABadForwardRefType"  # noqa: F821

    # 将 _NamedTupleBadMemberType 类型注册为全局类型，使其在脚本化时可用
    make_global(_NamedTupleBadMemberType)  # see [local resolution in python]

    # 定义一个接受 _NamedTupleBadMemberType 类型参数并返回 torch.Tensor 类型的函数 fn
    def fn(x: _NamedTupleBadMemberType) -> torch.Tensor:
        # 调用 x 对象的 f1 成员，并对其执行 relu 操作
        return x.f1.relu()

    # 使用 assertRaisesRegex 断言捕获 ValueError 异常，并检查异常信息是否包含特定文本 "at +File"
    with self.assertRaisesRegex(ValueError, "at +File"):
        # 尝试对函数 fn 进行脚本化处理
        torch.jit.script(fn)

    # 定义一个名为 BaseModule 的类，继承自 torch.nn.Module 类
    class BaseModule(torch.nn.Module):
        # 类属性 state 定义为 List[int] 类型的列表
        state: List[int]

        # 定义一个 forward 方法，接受一个参数 x，不进行任何操作
        def forward(self, x):
            pass

    # 定义一个接受 List[int] 类型参数 x 并返回最后一个元素或默认值 5 的函数 do_something_with_list
    def do_something_with_list(x: List[int]):
        if x:
            return x[-1]
        return 5

    # 定义一个名为 Submodule 的类，继承自 BaseModule 类
    class Submodule(BaseModule):
        # 构造函数初始化方法，接受 self_x_value 参数
        def __init__(self, self_x_value):
            super().__init__()
            # 设置实例变量 self.x 为 self_x_value
            self.x = self_x_value
            # 初始化实例变量 self.state 为空列表
            self.state = []

        # 重写基类的 forward 方法，接受一个参数 x，返回 self.x + x + do_something_with_list(self.state) 的结果
        def forward(self, x):
            return self.x + x + do_something_with_list(self.state)

    # 定义一个名为 LowestModule 的类，继承自 Submodule 类
    class LowestModule(Submodule):
        # 构造函数初始化方法，无参数
        def __init__(self):
            # 调用父类 Submodule 的构造函数，并传入参数 123
            super().__init__(123)

    # 创建一个 LowestModule 类的实例 mod 和 mod2
    mod = LowestModule()
    mod2 = LowestModule()

    # 对 mod 和 mod2 进行 torch.jit.script 脚本化处理，返回脚本化后的模型对象 mod_s 和 mod2_s
    mod_s = torch.jit.script(mod)
    mod2_s = torch.jit.script(mod2)
```