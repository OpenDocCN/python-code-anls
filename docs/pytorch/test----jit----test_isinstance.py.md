# `.\pytorch\test\jit\test_isinstance.py`

```py
# Owner(s): ["oncall: jit"]

# 引入必要的库和模块
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch  # 导入PyTorch库

# Make the helper files in test/ importable
# 将test目录下的辅助文件设置为可导入状态
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase  # 导入测试工具类JitTestCase

# 如果脚本直接运行，抛出运行时错误，提示不应直接运行该测试文件
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# Tests for torch.jit.isinstance
# 测试 torch.jit.isinstance 函数

class TestIsinstance(JitTestCase):  # 定义测试类 TestIsinstance，继承自 JitTestCase

    def test_int(self):
        # 测试整数类型
        def int_test(x: Any):
            assert torch.jit.isinstance(x, int)  # 断言 x 是否为整数类型
            assert not torch.jit.isinstance(x, float)  # 断言 x 是否不为浮点数类型

        x = 1  # 设置整数 x
        self.checkScript(int_test, (x,))  # 调用 JitTestCase 的 checkScript 方法进行脚本测试

    def test_float(self):
        # 测试浮点数类型
        def float_test(x: Any):
            assert torch.jit.isinstance(x, float)  # 断言 x 是否为浮点数类型
            assert not torch.jit.isinstance(x, int)  # 断言 x 是否不为整数类型

        x = 1.0  # 设置浮点数 x
        self.checkScript(float_test, (x,))  # 调用 JitTestCase 的 checkScript 方法进行脚本测试

    def test_bool(self):
        # 测试布尔类型
        def bool_test(x: Any):
            assert torch.jit.isinstance(x, bool)  # 断言 x 是否为布尔类型
            assert not torch.jit.isinstance(x, float)  # 断言 x 是否不为浮点数类型

        x = False  # 设置布尔值 x
        self.checkScript(bool_test, (x,))  # 调用 JitTestCase 的 checkScript 方法进行脚本测试

    def test_list(self):
        # 测试字符串列表类型
        def list_str_test(x: Any):
            assert torch.jit.isinstance(x, List[str])  # 断言 x 是否为字符串列表类型
            assert not torch.jit.isinstance(x, List[int])  # 断言 x 是否不为整数列表类型
            assert not torch.jit.isinstance(x, Tuple[int])  # 断言 x 是否不为整数元组类型

        x = ["1", "2", "3"]  # 设置字符串列表 x
        self.checkScript(list_str_test, (x,))  # 调用 JitTestCase 的 checkScript 方法进行脚本测试

    def test_list_tensor(self):
        # 测试张量列表类型
        def list_tensor_test(x: Any):
            assert torch.jit.isinstance(x, List[torch.Tensor])  # 断言 x 是否为张量列表类型
            assert not torch.jit.isinstance(x, Tuple[int])  # 断言 x 是否不为整数元组类型

        x = [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]  # 设置张量列表 x
        self.checkScript(list_tensor_test, (x,))  # 调用 JitTestCase 的 checkScript 方法进行脚本测试

    def test_dict(self):
        # 测试字符串到整数字典类型
        def dict_str_int_test(x: Any):
            assert torch.jit.isinstance(x, Dict[str, int])  # 断言 x 是否为字符串到整数字典类型
            assert not torch.jit.isinstance(x, Dict[int, str])  # 断言 x 是否不为整数到字符串字典类型
            assert not torch.jit.isinstance(x, Dict[str, str])  # 断言 x 是否不为字符串到字符串字典类型

        x = {"a": 1, "b": 2}  # 设置字符串到整数字典 x
        self.checkScript(dict_str_int_test, (x,))  # 调用 JitTestCase 的 checkScript 方法进行脚本测试

    def test_dict_tensor(self):
        # 测试整数到张量字典类型
        def dict_int_tensor_test(x: Any):
            assert torch.jit.isinstance(x, Dict[int, torch.Tensor])  # 断言 x 是否为整数到张量字典类型

        x = {2: torch.tensor([2])}  # 设置整数到张量字典 x
        self.checkScript(dict_int_tensor_test, (x,))  # 调用 JitTestCase 的 checkScript 方法进行脚本测试

    def test_tuple(self):
        # 测试字符串、整数、字符串元组类型
        def tuple_test(x: Any):
            assert torch.jit.isinstance(x, Tuple[str, int, str])  # 断言 x 是否为字符串、整数、字符串元组类型
            assert not torch.jit.isinstance(x, Tuple[int, str, str])  # 断言 x 是否不为整数、字符串、字符串元组类型
            assert not torch.jit.isinstance(x, Tuple[str])  # 断言 x 是否不为仅包含一个字符串的元组类型

        x = ("a", 1, "b")  # 设置字符串、整数、字符串元组 x
        self.checkScript(tuple_test, (x,))  # 调用 JitTestCase 的 checkScript 方法进行脚本测试
    # 定义一个测试函数，用于验证输入参数 x 是否为 Tuple[torch.Tensor, torch.Tensor] 类型
    def test_tuple_tensor(self):
        # 定义内部函数 tuple_tensor_test，用于检查 x 是否符合 Tuple[torch.Tensor, torch.Tensor] 类型
        def tuple_tensor_test(x: Any):
            assert torch.jit.isinstance(x, Tuple[torch.Tensor, torch.Tensor])

        # 定义输入变量 x，其类型为 Tuple[torch.Tensor, torch.Tensor]
        x = (torch.tensor([1]), torch.tensor([[2], [3]]))
        # 调用 self.checkScript 方法，验证 tuple_tensor_test 函数
        self.checkScript(tuple_tensor_test, (x,))

    # 定义一个测试函数，用于验证输入参数 x 是否为 Optional[torch.Tensor] 类型
    def test_optional(self):
        # 定义内部函数 optional_test，用于检查 x 是否符合 Optional[torch.Tensor] 类型
        def optional_test(x: Any):
            assert torch.jit.isinstance(x, Optional[torch.Tensor])
            assert not torch.jit.isinstance(x, Optional[str])

        # 定义输入变量 x，其类型为 torch.Tensor
        x = torch.ones(3, 3)
        # 调用 self.checkScript 方法，验证 optional_test 函数
        self.checkScript(optional_test, (x,))

    # 定义一个测试函数，用于验证输入参数 x 是否为 Optional[torch.Tensor] 类型，且为 None 值
    def test_optional_none(self):
        # 定义内部函数 optional_test_none，用于检查 x 是否符合 Optional[torch.Tensor] 类型
        # TODO: 上述行在急切模式下将评估为 True，而在 TS 解释器中将评估为 False，因为第一个 torch.jit.isinstance 精细化了 'None' 类型
        def optional_test_none(x: Any):
            assert torch.jit.isinstance(x, Optional[torch.Tensor])
            # assert torch.jit.isinstance(x, Optional[str])

        # 定义输入变量 x 为 None
        x = None
        # 调用 self.checkScript 方法，验证 optional_test_none 函数
        self.checkScript(optional_test_none, (x,))

    # 定义一个测试函数，用于验证输入参数 x 是否为 List[Dict[str, int]] 类型
    def test_list_nested(self):
        # 定义内部函数 list_nested，用于检查 x 是否符合 List[Dict[str, int]] 类型
        def list_nested(x: Any):
            assert torch.jit.isinstance(x, List[Dict[str, int]])
            assert not torch.jit.isinstance(x, List[List[str]])

        # 定义输入变量 x，其类型为 List 包含两个字典元素
        x = [{"a": 1, "b": 2}, {"aa": 11, "bb": 22}]
        # 调用 self.checkScript 方法，验证 list_nested 函数
        self.checkScript(list_nested, (x,))

    # 定义一个测试函数，用于验证输入参数 x 是否为 Dict[str, Tuple[str, str, str]] 类型
    def test_dict_nested(self):
        # 定义内部函数 dict_nested，用于检查 x 是否符合 Dict[str, Tuple[str, str, str]] 类型
        def dict_nested(x: Any):
            assert torch.jit.isinstance(x, Dict[str, Tuple[str, str, str]])
            assert not torch.jit.isinstance(x, Dict[str, Tuple[int, int, int]])

        # 定义输入变量 x，其类型为包含两个键值对的字典
        x = {"a": ("aa", "aa", "aa"), "b": ("bb", "bb", "bb")}
        # 调用 self.checkScript 方法，验证 dict_nested 函数
        self.checkScript(dict_nested, (x,))

    # 定义一个测试函数，用于验证输入参数 x 是否为 Tuple[Dict[str, Tuple[str, str, str]], List[bool], Optional[str]] 类型
    def test_tuple_nested(self):
        # 定义内部函数 tuple_nested，用于检查 x 是否符合 Tuple[Dict[str, Tuple[str, str, str]], List[bool], Optional[str]] 类型
        def tuple_nested(x: Any):
            assert torch.jit.isinstance(
                x, Tuple[Dict[str, Tuple[str, str, str]], List[bool], Optional[str]]
            )
            assert not torch.jit.isinstance(x, Dict[str, Tuple[int, int, int]])
            assert not torch.jit.isinstance(x, Tuple[str])
            assert not torch.jit.isinstance(x, Tuple[List[bool], List[str], List[int]])

        # 定义输入变量 x，其类型为包含一个字典、一个布尔列表和一个空值的元组
        x = (
            {"a": ("aa", "aa", "aa"), "b": ("bb", "bb", "bb")},
            [True, False, True],
            None,
        )
        # 调用 self.checkScript 方法，验证 tuple_nested 函数
        self.checkScript(tuple_nested, (x,))

    # 定义一个测试函数，用于验证输入参数 x 是否为 Optional[List[str]] 类型
    def test_optional_nested(self):
        # 定义内部函数 optional_nested，用于检查 x 是否符合 Optional[List[str]] 类型
        def optional_nested(x: Any):
            assert torch.jit.isinstance(x, Optional[List[str]])

        # 定义输入变量 x，其类型为包含三个字符串的列表
        x = ["a", "b", "c"]
        # 调用 self.checkScript 方法，验证 optional_nested 函数
        self.checkScript(optional_nested, (x,))

    # 定义一个测试函数，用于验证输入参数 x 是否为 List[torch.Tensor] 类型
    def test_list_tensor_type_true(self):
        # 定义内部函数 list_tensor_type_true，用于检查 x 是否符合 List[torch.Tensor] 类型
        def list_tensor_type_true(x: Any):
            assert torch.jit.isinstance(x, List[torch.Tensor])

        # 定义输入变量 x，其类型为包含两个 torch.Tensor 元素的列表
        x = [torch.rand(3, 3), torch.rand(4, 3)]
        # 调用 self.checkScript 方法，验证 list_tensor_type_true 函数
        self.checkScript(list_tensor_type_true, (x,))

    # 定义一个测试函数，用于验证输入参数 x 是否为 List[torch.Tensor] 类型
    def test_tensor_type_false(self):
        # 定义内部函数 list_tensor_type_false，用于检查 x 是否不符合 List[torch.Tensor] 类型
        def list_tensor_type_false(x: Any):
            assert not torch.jit.isinstance(x, List[torch.Tensor])

        # 定义输入变量 x，其类型为包含三个整数的列表
        x = [1, 2, 3]
        # 调用 self.checkScript 方法，验证 list_tensor_type_false 函数
        self.checkScript(list_tensor_type_false, (x,))
    def test_in_if(self):
        # 定义一个函数 list_in_if，参数 x 的类型可以是任意类型
        def list_in_if(x: Any):
            # 如果 x 是 List[int] 类型
            if torch.jit.isinstance(x, List[int]):
                # 断言为真
                assert True
            # 如果 x 是 List[str] 类型
            if torch.jit.isinstance(x, List[str]):
                # 断言为假
                assert not True

        # 设定 x 为列表 [1, 2, 3]
        x = [1, 2, 3]
        # 使用 self.checkScript 函数测试 list_in_if 函数
        self.checkScript(list_in_if, (x,))

    def test_if_else(self):
        # 定义一个函数 list_in_if_else，参数 x 的类型可以是任意类型
        def list_in_if_else(x: Any):
            # 如果 x 是 Tuple[str, str, str] 类型
            if torch.jit.isinstance(x, Tuple[str, str, str]):
                # 断言为真
                assert True
            # 否则
            else:
                # 断言为假
                assert not True

        # 设定 x 为元组 ("a", "b", "c")
        x = ("a", "b", "c")
        # 使用 self.checkScript 函数测试 list_in_if_else 函数
        self.checkScript(list_in_if_else, (x,))

    def test_in_while_loop(self):
        # 定义一个函数 list_in_while_loop，参数 x 的类型可以是任意类型
        def list_in_while_loop(x: Any):
            # 初始化计数器 count 为 0
            count = 0
            # 当 x 是 List[Dict[str, int]] 类型并且 count 小于等于 0 时执行循环
            while torch.jit.isinstance(x, List[Dict[str, int]]) and count <= 0:
                # 计数器加一
                count = count + 1
            # 断言 count 等于 1
            assert count == 1

        # 设定 x 为字典列表
        x = [{"a": 1, "b": 2}, {"aa": 11, "bb": 22}]
        # 使用 self.checkScript 函数测试 list_in_while_loop 函数
        self.checkScript(list_in_while_loop, (x,))

    def test_type_refinement(self):
        # 定义一个函数 type_refinement，参数 obj 的类型可以是任意类型
        def type_refinement(obj: Any):
            # 初始化标志 hit 为 False
            hit = False
            # 如果 obj 是 List[torch.Tensor] 类型
            if torch.jit.isinstance(obj, List[torch.Tensor]):
                # 将 hit 取反
                hit = not hit
                # 遍历列表 obj 中的每个元素 el
                for el in obj:
                    # 对每个张量 el 执行 clamp 操作
                    y = el.clamp(0, 0.5)
            # 如果 obj 是 Dict[str, str] 类型
            if torch.jit.isinstance(obj, Dict[str, str]):
                # 将 hit 取反
                hit = not hit
                # 初始化空字符串 str_cat
                str_cat = ""
                # 遍历字典 obj 的所有值 val
                for val in obj.values():
                    # 将每个值 val 添加到 str_cat 中
                    str_cat = str_cat + val
                # 断言拼接后的字符串与 "111222" 相等
                assert "111222" == str_cat
            # 断言 hit 为真
            assert hit

        # 设定 x 为包含张量的列表
        x = [torch.rand(3, 3), torch.rand(4, 3)]
        # 使用 self.checkScript 函数测试 type_refinement 函数
        self.checkScript(type_refinement, (x,))
        # 设定 x 为字符串键值对的字典
        x = {"1": "111", "2": "222"}
        # 使用 self.checkScript 函数测试 type_refinement 函数
        self.checkScript(type_refinement, (x,))

    def test_list_no_contained_type(self):
        # 定义一个函数 list_no_contained_type，参数 x 的类型可以是任意类型
        def list_no_contained_type(x: Any):
            # 断言 x 是 List 类型但未指定包含的具体类型
            assert torch.jit.isinstance(x, List)

        # 设定 x 为包含字符串的列表
        x = ["1", "2", "3"]

        # 错误消息字符串，指出 List 类型需要指定包含的具体类型
        err_msg = (
            "Attempted to use List without a contained type. "
            r"Please add a contained type, e.g. List\[int\]"
        )

        # 使用 self.assertRaisesRegex 检查运行时异常并验证错误消息
        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            # 尝试将 list_no_contained_type 函数编译为 Torch 脚本
            torch.jit.script(list_no_contained_type)
        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            # 直接调用 list_no_contained_type 函数，检查是否会引发相同异常
            list_no_contained_type(x)

    def test_tuple_no_contained_type(self):
        # 定义一个函数 tuple_no_contained_type，参数 x 的类型可以是任意类型
        def tuple_no_contained_type(x: Any):
            # 断言 x 是 Tuple 类型但未指定包含的具体类型
            assert torch.jit.isinstance(x, Tuple)

        # 设定 x 为包含字符串的元组
        x = ("1", "2", "3")

        # 错误消息字符串，指出 Tuple 类型需要指定包含的具体类型
        err_msg = (
            "Attempted to use Tuple without a contained type. "
            r"Please add a contained type, e.g. Tuple\[int\]"
        )

        # 使用 self.assertRaisesRegex 检查运行时异常并验证错误消息
        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            # 尝试将 tuple_no_contained_type 函数编译为 Torch 脚本
            torch.jit.script(tuple_no_contained_type)
        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            # 直接调用 tuple_no_contained_type 函数，检查是否会引发相同异常
            tuple_no_contained_type(x)
    def test_optional_no_contained_type(self):
        # 定义一个测试函数，用于检查使用 Optional 类型时未指定包含的具体类型的情况
        def optional_no_contained_type(x: Any):
            # 断言 x 是否为 Optional 类型
            assert torch.jit.isinstance(x, Optional)

        # 准备一个元组作为测试参数
        x = ("1", "2", "3")

        # 准备错误信息字符串
        err_msg = (
            "Attempted to use Optional without a contained type. "
            r"Please add a contained type, e.g. Optional\[int\]"
        )

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并验证错误信息是否匹配
        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            # 调用 torch.jit.script 对 optional_no_contained_type 进行脚本化，预期会触发异常
            torch.jit.script(optional_no_contained_type)
        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            # 调用 optional_no_contained_type 函数，传入参数 x，预期会触发异常
            optional_no_contained_type(x)

    def test_dict_no_contained_type(self):
        # 定义一个测试函数，用于检查使用 Dict 类型时未指定包含的具体类型的情况
        def dict_no_contained_type(x: Any):
            # 断言 x 是否为 Dict 类型
            assert torch.jit.isinstance(x, Dict)

        # 准备一个字典作为测试参数
        x = {"a": "aa"}

        # 准备错误信息字符串
        err_msg = (
            "Attempted to use Dict without contained types. "
            r"Please add contained type, e.g. Dict\[int, int\]"
        )

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并验证错误信息是否匹配
        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            # 调用 torch.jit.script 对 dict_no_contained_type 进行脚本化，预期会触发异常
            torch.jit.script(dict_no_contained_type)
        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            # 调用 dict_no_contained_type 函数，传入参数 x，预期会触发异常
            dict_no_contained_type(x)

    def test_tuple_rhs(self):
        # 定义一个测试函数，检查在类型注解中使用元组进行类型检查
        def fn(x: Any):
            # 断言 x 是否为 int 或者 List[str] 类型
            assert torch.jit.isinstance(x, (int, List[str]))
            # 断言 x 不是 List[float] 或 Tuple[int, str] 类型
            assert not torch.jit.isinstance(x, (List[float], Tuple[int, str]))
            # 断言 x 不是 List[float] 或 str 类型
            assert not torch.jit.isinstance(x, (List[float], str))

        # 调用 self.checkScript 进行函数 fn 的脚本化测试，分别使用不同类型的参数进行测试
        self.checkScript(fn, (2,))
        self.checkScript(fn, (["foo", "bar", "baz"],))

    def test_nontuple_container_rhs_throws_in_eager(self):
        # 定义一个测试函数，检查在类型注解中使用非元组类型作为容器时是否会触发异常
        def fn1(x: Any):
            # 断言 x 是否为 [int, List[str]] 类型，预期会触发异常
            assert torch.jit.isinstance(x, [int, List[str]])

        def fn2(x: Any):
            # 断言 x 不是 {List[str], Tuple[int, str]} 类型，预期会触发异常
            assert not torch.jit.isinstance(x, {List[str], Tuple[int, str]})

        # 准备错误信息字符串
        err_highlight = "must be a type or a tuple of types"

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并验证错误信息是否匹配
        with self.assertRaisesRegex(RuntimeError, err_highlight):
            fn1(2)

        with self.assertRaisesRegex(RuntimeError, err_highlight):
            fn2(2)

    def test_empty_container_throws_warning_in_eager(self):
        # 定义一个测试函数，检查在类型注解中使用空容器时是否会触发警告
        def fn(x: Any):
            # 断言 x 是否为 List[int] 类型
            torch.jit.isinstance(x, List[int])

        # 使用 warnings.catch_warnings 捕获警告，并验证警告数量
        with warnings.catch_warnings(record=True) as w:
            # 准备一个空的 List[int] 作为测试参数
            x: List[int] = []
            fn(x)
            # 验证捕获的警告数量是否为 1
            self.assertEqual(len(w), 1)

        with warnings.catch_warnings(record=True) as w:
            # 准备一个 int 类型作为测试参数
            x: int = 2
            fn(x)
            # 验证捕获的警告数量是否为 0
            self.assertEqual(len(w), 0)

    def test_empty_container_special_cases(self):
        # 测试函数，检查特殊情况下使用空容器时是否会触发特定错误
        # 不应该触发 "Boolean value of Tensor with no values is ambiguous" 错误
        torch._jit_internal.check_empty_containers(torch.Tensor([]))

        # 不应该触发 "Boolean value of Tensor with more than one value is ambiguous" 错误
        torch._jit_internal.check_empty_containers(torch.rand(2, 3))
```