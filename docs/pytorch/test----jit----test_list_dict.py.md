# `.\pytorch\test\jit\test_list_dict.py`

```
# Owner(s): ["oncall: jit"]

# 导入必要的模块和库
import inspect  # 导入用于获取对象信息的 inspect 模块
import os  # 导入操作系统功能的 os 模块
import sys  # 导入系统相关的功能的 sys 模块
import types  # 导入处理 Python 类型的 types 模块
import unittest  # 导入用于编写和运行单元测试的 unittest 模块
from collections import defaultdict, OrderedDict  # 导入用于创建特定类型字典和有序字典的模块
from textwrap import dedent  # 导入用于移除代码块开头空白的 dedent 函数
from typing import Any, Dict, List, NamedTuple, Optional, Tuple  # 导入用于类型注解的模块

import torch  # 导入 PyTorch 深度学习框架
import torch.nn as nn  # 导入 PyTorch 的神经网络模块

from torch import Tensor  # 从 torch 模块导入 Tensor 类型
from torch.testing import FileCheck  # 导入用于测试文件的检查工具

# 将 test/ 目录下的辅助文件设置为可导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import skipIfTorchDynamo, TEST_CUDA  # 导入用于测试的辅助函数和装饰器
from torch.testing._internal.jit_utils import JitTestCase, make_global  # 导入用于 JIT 测试的辅助工具

if __name__ == "__main__":
    # 如果作为主程序执行，则抛出运行时错误，提示正确的使用方式
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestList(JitTestCase):
    def test_list_bool_conversion(self):
        # 测试列表到布尔值的转换函数
        def if_predicate(l: List[int]):
            if l:
                s = 0
                # 如果列表非空，计算列表中所有元素的和
                for n in l:
                    s += n
                return s
            else:
                # 如果列表为空，返回 -1
                return -1

        # 对 if_predicate 函数进行 JIT 脚本化检查，验证其在输入列表为 [1, 2, 3] 和 [] 时的行为
        self.checkScript(if_predicate, ([1, 2, 3],))
        self.checkScript(if_predicate, ([],))

        # 测试列表到布尔值的转换函数（使用 while 循环）
        def while_predicate(l: List[int]):
            s = 0
            # 当列表非空时，不断从列表中弹出元素并累加到 s 中
            while l:
                s += l.pop()

        # 对 while_predicate 函数进行 JIT 脚本化检查，验证其在输入列表为 [1, 2, 3] 和 [] 时的行为
        self.checkScript(while_predicate, ([1, 2, 3],))
        self.checkScript(while_predicate, ([],))

        # 测试列表到字符串的转换函数（使用三元表达式）
        def ternary_predicate(l: List[int]):
            return "non-empty" if l else "empty"

        # 对 ternary_predicate 函数进行 JIT 脚本化检查，验证其在输入列表为 [1, 2, 3] 和 [] 时的行为
        self.checkScript(ternary_predicate, ([1, 2, 3],))
        self.checkScript(ternary_predicate, ([],))

    def test_in_check(self):
        # 检查整数是否存在于列表中的函数
        def int_in(x: List[int]) -> bool:
            return 2 in x

        # 对 int_in 函数进行 JIT 脚本化检查，验证其在输入列表为 [1, 2, 3] 和 [1, 3, 3] 时的行为
        self.checkScript(int_in, ([1, 2, 3],))
        self.checkScript(int_in, ([1, 3, 3],))

        # 检查浮点数是否存在于列表中的函数
        def float_in(x: List[float]) -> bool:
            return 2.0 in x

        # 对 float_in 函数进行 JIT 脚本化检查，验证其在输入列表为 [1.0, 2.0, 3.0] 和 [1.0, 3.0, 3.0] 时的行为
        self.checkScript(float_in, ([1.0, 2.0, 3.0],))
        self.checkScript(float_in, ([1.0, 3.0, 3.0],))

        # 检查字符串是否存在于列表中的函数
        def str_in(x: List[str]) -> bool:
            return "hi" in x

        # 对 str_in 函数进行 JIT 脚本化检查，验证其在输入列表为 ["not", "here"] 和 ["hi", "bye"] 时的行为
        self.checkScript(str_in, (["not", "here"],))
        self.checkScript(str_in, (["hi", "bye"],))
        self.checkScript(str_in, ([],))
    # 定义一个测试方法，测试列表字面值的重新赋值行为
    def test_list_literal(self):
        
        # 定义重新赋值函数 reassign
        def reassign():
            # 初始化列表 x 包含元素 1
            x = [1]
            # 如果条件满足，则重新赋值 x 为包含元素 2 和 3 的新列表
            if 1 == 1:
                x = [2, 3]
            return
        
        # 使用 self.checkScript 方法检查 reassign 函数的行为，禁用优化
        self.checkScript(reassign, (), optimize=False)

        # 定义重新赋值函数 reassign_arity_change
        def reassign_arity_change():
            # 初始化列表 x 包含元素 1
            x = [1]
            # 如果条件满足，则重新赋值 x 为包含元素 1、2 和 3 的新列表
            if 1 == 1:
                x = [1, 2, 3]
            return
        
        # 使用 self.checkScript 方法检查 reassign_arity_change 函数的行为，禁用优化
        self.checkScript(reassign_arity_change, (), optimize=False)

        # 定义从空列表字面值重新赋值函数 reassign_from_empty_literal
        def reassign_from_empty_literal():
            # 初始化空列表 x
            x = []
            # 如果条件满足，则重新赋值 x 为包含元素 1、2 和 3 的新列表
            if 1 == 1:
                x = [1, 2, 3]
            return
        
        # 使用 self.assertRaisesRegexWithHighlight 方法验证 reassign_from_empty_literal 函数会抛出特定异常
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"previously had type List\[Tensor\]", "x"
        ):
            # 使用 self.checkScript 方法检查 reassign_from_empty_literal 函数的行为，禁用优化
            self.checkScript(reassign_from_empty_literal, (), optimize=False)

        # 定义从空的内建类型注释列表重新赋值函数 reassign_from_empty_builtin
        def reassign_from_empty_builtin():
            # 使用 torch.jit.annotate 标注类型为 List[int] 的空列表 x
            x = torch.jit.annotate(List[int], [])
            # 如果条件满足，则重新赋值 x 为包含元素 1、2 和 3 的新列表
            if 1 == 1:
                x = [1, 2, 3]
            
            # 使用 torch.jit.annotate 标注类型为 List[float] 的空列表 y
            y = torch.jit.annotate(List[float], [])
            # 如果条件满足，则重新赋值 y 为包含元素 1.0、2.0 和 3.0 的新列表
            if 1 == 1:
                y = [1.0, 2.0, 3.0]
            
            # 初始化空列表 z
            z = []
            # 如果条件满足，则重新赋值 z 为包含一个元素为 torch.randn([1]) 的新列表
            if 1 == 1:
                z = [torch.randn([1])]
            return
        
        # 使用 self.checkScript 方法检查 reassign_from_empty_builtin 函数的行为，禁用优化
        self.checkScript(reassign_from_empty_builtin, (), optimize=False)

        # 定义类型不匹配的重新赋值函数 reassign_bad_type
        def reassign_bad_type():
            # 初始化列表 x 包含元素 1
            x = [1]
            # 如果条件满足，则尝试将 x 重新赋值为包含元素 1.0 的新列表，类型不匹配
            if 1 == 1:
                x = [1.0]
            return
        
        # 使用 self.assertRaisesRegexWithHighlight 方法验证 reassign_bad_type 函数会抛出特定异常
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "previously had type", "x"
        ):
            # 使用 self.checkScript 方法检查 reassign_bad_type 函数的行为，禁用优化
            self.checkScript(reassign_bad_type, (), optimize=False)

        # 定义嵌套重新赋值函数 reassign_nested
        def reassign_nested():
            # 使用 torch.jit.annotate 标注类型为 List[int] 的空列表 x
            x = torch.jit.annotate(List[int], [])
            # 如果条件满足，则重新赋值 x 为包含元素 1、2 和 3 的新列表
            if 1 == 1:
                x = [1, 2, 3]
                # 在条件满足的情况下，尝试将 x 重新赋值为包含元素 1.0 的新列表，类型不匹配
                if 1 == 1:
                    x = [1.0]
            return
        
        # 使用 self.assertRaisesRegexWithHighlight 方法验证 reassign_nested 函数会抛出特定异常
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "previously had type", "x"
        ):
            # 使用 self.checkScript 方法检查 reassign_nested 函数的行为，禁用优化
            self.checkScript(reassign_nested, (), optimize=False)
    def test_list_variance(self):
        """
        `List[T1]` is not a subtype of `List[T2]`, even if `T1` is a
        subtype of `T2`. However, if we have a temporary list object
        (that is, a list comprehension or a list literal) on the rhs of
        an assignment statement, we want to ignore the inferred type of
        the rhs if we can prove that: 1) both the lhs and the rhs are
        lists, and 2) the inner type of the lhs list is a subtype of the
        inner type of the rhs list.

        # This should pass
        x: List[Optional[int]] = [None, None, None]

        # This should fail
        y: List[None] = [None, None, None]
        x: List[Optional[int]] = y
        """

        # 定义一个函数，验证列表字面值是否从注解中推断出类型
        def test_listliteral_is_typed_from_annotation():
            # 声明一个列表 x，其元素类型为 Optional[int]
            x: List[Optional[int]] = [None, None, None]
            return x

        # 调用检查函数，验证其结果
        self.checkScript(test_listliteral_is_typed_from_annotation, ())

        # 定义一个函数，验证列表推导是否从注解中推断出类型
        def test_listcomprehension_is_typed_from_annotation():
            # 声明一个列表 x，其元素类型为 Optional[int]
            x: List[Optional[int]] = [None for _ in range(3)]
            return x

        # 调用检查函数，验证其结果
        self.checkScript(test_listcomprehension_is_typed_from_annotation, ())

        # 定义一个函数，验证具有不同内部类型的列表是不变的
        def test_lists_with_different_internal_types_are_invariant(self):
            # 声明一个列表 x，其元素类型为 int
            x: List[int] = [1, 2, 3]
            # 将 x 赋给类型为 List[Optional[int]] 的变量 y，这里应该会失败
            y: List[Optional[int]] = x
            return x

        # 使用断言捕获运行时错误，验证类型不变性的预期行为
        with self.assertRaisesRegex(
            RuntimeError,
            "Variable 'y' is "
            "annotated with type "
            r"List\[Optional\[int\]\] but is "
            "being assigned to a value of type "
            r"List\[int\]",
        ):
            torch.jit.script(test_lists_with_different_internal_types_are_invariant)

        # 定义一个函数，验证具有不同内部类型的列表是不变的（递归情况）
        def test_lists_with_different_internal_types_are_invariant_recursive(self):
            # 声明一个列表 x，其元素类型为 List[int]
            x: List[List[int]] = [[1, 2], [3]]
            # 将 x 赋给类型为 List[List[Optional[int]]] 的变量 y，这里应该会失败
            y: List[List[Optional[int]]] = x
            return x

        # 使用断言捕获运行时错误，验证类型不变性的预期行为（递归情况）
        with self.assertRaisesRegex(
            RuntimeError,
            "Variable 'y' is "
            "annotated with type "
            r"List\[List\[Optional\[int\]\]\] "
            "but is being assigned to a value "
            r"of type List\[List\[int\]\]",
        ):
            torch.jit.script(
                test_lists_with_different_internal_types_are_invariant_recursive
            )
    def test_del(self):
        # 定义一个内部函数inputs，返回一个包含整数的列表
        def inputs():
            return [1, 2, 3, 4]

        # 定义一个函数fn，接受一个整数列表参数x，并返回删除索引1处元素后的列表x
        def fn(x: List[int]) -> List[int]:
            del x[1]
            return x

        # 调用fn函数，并将结果存储在python_out变量中
        python_out = fn(inputs())
        # 创建一个torch脚本编译单元cu
        cu = torch.jit.CompilationUnit()
        # 定义cu的fn方法，将fn函数的源码添加到cu中
        cu.define(dedent(inspect.getsource(fn)))
        # 使用torch.jit.script将fn编译为脚本，并比较其结果与python_out是否相等
        self.assertEqual(cu.fn(inputs()), python_out)
        # 同上，使用torch.jit.script将fn编译为脚本，并比较其结果与python_out是否相等
        self.assertEqual(torch.jit.script(fn)(inputs()), python_out)

        # 使用torch.jit.script装饰器定义一个名为fn2的脚本函数，接受一个整数列表参数x，并删除索引100处的元素
        @torch.jit.script
        def fn2(x: List[int]) -> List[int]:
            del x[100]
            return x

        # 断言调用fn2函数时会抛出RuntimeError异常，异常信息包含"out of range"和"x[100]"关键字
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "out of range", "x[100]"
        ):
            fn2([])

        # 断言使用torch.jit.script装饰器定义的fn函数时会抛出RuntimeError异常，异常信息包含"deletion at a single index"和"x[1:3]"关键字
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "deletion at a single index", "x[1:3]"
        ):
            # 使用torch.jit.script装饰器定义一个名为fn的脚本函数，接受一个整数列表参数x，并尝试删除索引1到2之间的元素
            @torch.jit.script
            def fn(x: List[int]) -> List[int]:
                del x[1:3]
                return x

    def test_list_keyword(self):
        # 定义一个内部函数foo，返回包含多个列表的元组
        def foo():
            return (
                list([1, 2, 3]),  # noqa: C410
                list(("a", "b")),  # noqa: C410
                list(range(5)),
                list("abcdefg"),
            )

        # 调用self.checkScript函数检查foo函数的脚本化结果，不传入任何参数
        self.checkScript(foo, ())

        # 定义一个内部函数foo2，初始化一个整数列表x，追加元素后返回包含x的元组
        def foo2():
            x: List[int] = list()
            x.append(1)
            return (x,)

        # 调用self.checkScript函数检查foo2函数的脚本化结果，不传入任何参数
        self.checkScript(foo2, ())

        # 定义一个内部函数foo3，返回将字符串"abc"转换为列表后的结果
        def foo3():
            return list(list("abc"))  # noqa: C414

        # 调用self.checkScript函数检查foo3函数的脚本化结果，不传入任何参数
        FileCheck().check_count("aten::list", 2, exactly=True).run(
            torch.jit.script(foo3).graph
        )

    def test_dict_keyword_with_kwargs(self):
        # 定义一个内部函数fn，返回带有关键字参数的字典
        def fn():
            return dict(foo=1, bar=2, baz=3)

        # 调用self.checkScript函数检查fn函数的脚本化结果，不传入任何参数
        self.checkScript(fn, ())

    def test_dict_keyword_with_kwargs_using_container_values(self):
        # 定义一个内部函数fn，返回使用容器值作为值的字典
        def fn():
            return dict(foo=[1, 2, 3], bar=[4, 5, 6], baz=[7, 8, 9])

        # 调用self.checkScript函数检查fn函数的脚本化结果，不传入任何参数
        self.checkScript(fn, ())

    def test_dict_keyword_with_iterable(self):
        # 定义一个内部函数fn，返回使用可迭代对象创建的字典
        def fn():
            return dict([("foo", 1), ("bar", 2), ("baz", 3)])  # noqa: C406

        # 调用self.checkScript函数检查fn函数的脚本化结果，不传入任何参数
        self.checkScript(fn, ())

    def test_dict_keyword_with_empty_iterable(self):
        # 定义一个内部函数fn，返回一个空字典
        def fn():
            return dict([])  # noqa: C406

        # 调用self.checkScript函数检查fn函数的脚本化结果，不传入任何参数
        self.checkScript(fn, ())

    def test_dict_keyword_with_internal_aggregate_function(self):
        # 定义一个内部函数fn，返回使用zip函数创建的字典
        def fn():
            return dict(zip(["foo", "baz", "bar"], [1, 2, 3]))

        # 调用self.checkScript函数检查fn函数的脚本化结果，不传入任何参数
        self.checkScript(fn, ())

    def test_dict_keyword_with_mapping(self):
        # 定义一个内部函数fn，返回一个包含映射关系的字典
        def fn():
            return {"foo": 1, "bar": 2, "baz": 3}

        # 调用self.checkScript函数检查fn函数的脚本化结果，不传入任何参数
        self.checkScript(fn, ())

    def test_dict_keyword_with_mapping_and_kwargs(self):
        # 定义一个内部函数fn，返回一个包含映射关系和关键字参数的字典
        def fn():
            return dict({"foo": 1, "bar": 2}, baz=3)

        # 调用self.checkScript函数检查fn函数的脚本化结果，不传入任何参数
        self.checkScript(fn, ())

    def test_dict_keyword_with_dict_comprehension(self):
        # 定义一个内部函数fn，返回一个字典推导式的结果
        def fn():
            return {i: chr(i + 65) for i in range(4)}

        # 调用self.checkScript函数检查fn函数的脚本化结果，不传入任何参数
        self.checkScript(fn, ())
    # 定义一个测试方法，测试使用字典推导式和关键字参数创建字典
    def test_dict_keyword_with_dict_comprehension_and_kwargs(self):
        # 定义一个内部函数 fn，使用字典推导式创建一个字典，同时添加一个关键字参数 'foo'
        def fn():
            return dict({chr(65 + i): i for i in range(4)}, foo=2)
        
        # 调用外部方法 checkScript 来测试 fn 的行为，传入空元组作为参数
        self.checkScript(fn, ())

    # 定义一个测试方法，测试空字典的创建
    def test_dict_keyword_with_empty_dict_comprehension(self):
        # 定义一个内部函数 fn，直接返回一个空字典
        def fn():
            return {}
        
        # 调用外部方法 checkScript 来测试 fn 的行为，传入空元组作为参数
        self.checkScript(fn, ())

    # 定义一个测试方法，验证字典的类型注解
    def test_dict_keyword_is_correctly_typed(self):
        # 定义一个内部函数 fn，创建一个带有类型注解的字典，并添加一个键值对
        def fn():
            x: Dict[str, int] = dict()
            x["foo"] = 1
            return x
        
        # 调用外部方法 checkScript 来测试 fn 的行为，传入空元组作为参数
        self.checkScript(fn, ())

    # 定义一个测试方法，测试使用错误类型注解创建字典时的异常情况
    def test_dict_keyword_with_mismatched_annotations(self):
        # 定义一个错误消息字符串，用于验证异常信息
        err_msg = (
            r"Dict type annotation `Dict\[int, str\]` did not "
            "match the type of an actual key type `str`"
        )
        # 使用 assertRaisesRegex 确保在运行时捕获到指定的 RuntimeError 异常并且异常信息符合预期
        with self.assertRaisesRegex(RuntimeError, err_msg):
            # 定义一个 Torch 脚本函数 fn，尝试使用错误的类型注解创建字典
            @torch.jit.script
            def fn():
                x: Dict[int, str] = dict(  # noqa: C406
                    [("foo", 1), ("bar", 2), ("baz", 3)]
                )
                return x

    # 定义一个测试方法，测试在字典嵌套调用时的行为
    def test_dict_keyword_with_nested_call(self):
        # 定义一个内部函数 fn，使用嵌套字典创建方式返回一个字典
        def fn():
            return dict(dict(foo=1, bar=2, baz=3))
        
        # 调用外部方法 checkScript 来测试 fn 的行为，传入空元组作为参数
        self.checkScript(fn, ())

    # 定义一个测试方法，测试使用先前声明的变量创建字典
    def test_dict_keyword_with_previously_declared_variable(self):
        # 定义一个内部函数 fn，使用先前声明的字典 d 创建一个新的字典并返回
        def fn():
            d = {"foo": 1, "bar": 2}
            return dict(d)
        
        # 调用外部方法 checkScript 来测试 fn 的行为，传入空元组作为参数
        self.checkScript(fn, ())

    # 定义一个测试方法，测试使用先前声明的变量和关键字参数创建字典
    def test_dict_keyword_with_previously_declared_variable_and_kwargs(self):
        # 定义一个内部函数 fn，使用先前声明的字典 d 创建一个新的字典，并添加一个关键字参数
        def fn():
            d = {"foo": 1, "bar": 2}
            return dict(d, baz=3)
        
        # 调用外部方法 checkScript 来测试 fn 的行为，传入空元组作为参数
        self.checkScript(fn, ())

    # 定义一个测试方法，测试 min 函数在布尔列表上的行为
    def test_min_bool_list(self):
        # 定义一个内部函数 jit_min_list，接受两个布尔列表作为参数，并返回它们的最小值
        def jit_min_list(a: List[bool], b: List[bool]) -> List[bool]:
            return min(a, b)
        
        # 调用外部方法 checkScript 来测试 jit_min_list 的行为，传入两个布尔列表作为参数
        self.checkScript(jit_min_list, ([True, False], [False, True]))
    # 定义一个测试函数，用于测试最小值列表操作
    def test_min_max_list(self):
        # 定义一个函数，接受两个整数列表，返回它们的最小值列表
        def jit_min_list(a: List[int], b: List[int]) -> List[int]:
            return min(a, b)

        # 定义一个函数，接受两个浮点数列表，返回它们的最小值列表
        def jit_min_list_float(a: List[float], b: List[float]) -> List[float]:
            return min(a, b)

        # 定义一个函数，接受两个布尔值列表，返回它们的最小值列表
        def jit_min_list_bool(a: List[bool], b: List[bool]) -> List[bool]:
            return min(a, b)

        # 定义一个运行测试函数，接受一个函数和两个列表参数，依次检查每对参数
        def run_tests(func, a, b):
            for t in zip(a, b):
                self.checkScript(func, t)

        # 定义整数类型的左侧参数列表
        args_left_int = [[1, 8, 8], [2, 1, 1], [], [2], [1], [1, 2, 3]]
        # 定义整数类型的右侧参数列表
        args_right_int = [[2, 1, 1], [1, 8, 8], [], [1], [], [1, 2]]
        # 运行整数类型的最小值列表测试
        run_tests(jit_min_list, args_left_int, args_right_int)

        # 定义浮点数类型的左侧参数列表
        args_left_float = [
            [1.0, 8.0, 8.0],
            [2.0, 1.0, 1.0],
            [],
            [2.0],
            [1.0],
            [1.0, 2.0, 3.0],
        ]
        # 定义浮点数类型的右侧参数列表
        args_right_float = [[2.0, 1.0, 1.0], [1.0, 8.0, 8.0], [], [1.0], [], [1.0, 2.0]]
        # 运行浮点数类型的最小值列表测试
        run_tests(jit_min_list_float, args_left_float, args_right_float)

        # 定义布尔值类型的左侧参数列表
        args_left_bool = [
            [],
            [],
            [],
            [False],
            [True],
            [False, True],
            [True, True],
            [False, False, False],
            [False, False, True],
        ]
        # 定义布尔值类型的右侧参数列表
        args_right_bool = [
            [],
            [False],
            [True],
            [True],
            [False],
            [True, True],
            [False, True],
            [False, False, True],
            [False, False, False],
        ]
        # 运行布尔值类型的最小值列表测试
        run_tests(jit_min_list_bool, args_left_bool, args_right_bool)

        # 定义一个函数，接受两个整数列表，返回它们的最大值列表
        def jit_max_list(a: List[int], b: List[int]) -> List[int]:
            return max(a, b)

        # 定义一个函数，接受两个浮点数列表，返回它们的最大值列表
        def jit_max_list_float(a: List[float], b: List[float]) -> List[float]:
            return max(a, b)

        # 定义一个函数，接受两个布尔值列表，返回它们的最大值列表
        def jit_max_list_bool(a: List[bool], b: List[bool]) -> List[bool]:
            return max(a, b)

        # 重新定义整数类型的左侧参数列表
        args_left_int = [[1, 8, 8], [8, 1, 1], [], [1], [], [1, 2]]
        # 重新定义整数类型的右侧参数列表
        args_right_int = [[8, 1, 1], [1, 8, 8], [], [2], [1], [1, 2, 3]]
        # 运行整数类型的最大值列表测试
        run_tests(jit_max_list, args_left_int, args_right_int)

        # 重新定义浮点数类型的左侧参数列表
        args_left_float = [[1.0, 8.0, 8.0], [8.0, 1.0, 1.0], [], [1.0], [], [1.0, 2.0]]
        # 重新定义浮点数类型的右侧参数列表
        args_right_float = [
            [8.0, 1.0, 1.0],
            [1.0, 8.0, 8.0],
            [],
            [2.0],
            [1.0],
            [1.0, 2.0, 3.0],
        ]
        # 运行浮点数类型的最大值列表测试
        run_tests(jit_max_list_float, args_left_float, args_right_float)

        # 运行布尔值类型的最大值列表测试
        run_tests(jit_max_list_bool, args_left_bool, args_right_bool)
    # 定义测试方法 test_list_gather，用于测试列表的索引操作

        # 定义内部函数 index
        def index():
            # 创建列表 a 包含元素 [1, 2, 3]
            a = [1, 2, 3]
            # 返回列表 a 的第二个元素（索引为 1）
            return a[1]

        # 调用自定义方法 checkScript 来验证 index 函数的行为，不传入参数
        self.checkScript(index, ())

        # 定义内部函数 negative_index
        def negative_index():
            # 创建列表 a 包含元素 [1, 2, 3]
            a = [1, 2, 3]
            # 返回列表 a 的倒数第一个元素（索引为 -1）
            return a[-1]

        # 调用自定义方法 checkScript 来验证 negative_index 函数的行为，不传入参数
        self.checkScript(negative_index, ())

        # 定义内部函数 bad_index
        def bad_index():
            # 创建列表 a 包含元素 [1, 2, 3]
            a = [1, 2, 3]
            # 尝试访问列表 a 中索引为 4 的元素，将引发 IndexError 异常
            return a[4]

        # 使用自定义方法 checkScriptRaisesRegex 来验证 bad_index 函数，预期会抛出 IndexError 异常，且异常信息为 "list index out of range"
        self.checkScriptRaisesRegex(bad_index, (), Exception, "list index out of range")

        # 定义内部函数 bad_negative_index
        def bad_negative_index():
            # 创建列表 a 包含元素 [1, 2, 3]
            a = [1, 2, 3]
            # 尝试访问列表 a 中索引为 -5 的元素，将引发 IndexError 异常
            return a[-5]

        # 使用自定义方法 checkScriptRaisesRegex 来验证 bad_negative_index 函数，预期会抛出 IndexError 异常，且异常信息为 "list index out of range"
        self.checkScriptRaisesRegex(
            bad_negative_index, (), Exception, "list index out of range"
        )

    # 定义测试方法 test_list_len，用于测试列表的长度操作
    def test_list_len(self):

        # 定义内部函数 func
        def func():
            # 创建列表 a 包含元素 [1, 2, 3]
            a = [1, 2, 3]
            # 返回判断列表 a 的长度是否为 3 的布尔值
            return len(a) == 3

        # 调用自定义方法 checkScript 来验证 func 函数的行为，不传入参数
        self.checkScript(func, ())

        # 定义内部函数 func2
        def func2():
            # 创建空列表 a
            a = []
            # 返回判断列表 a 的长度是否为 0 的布尔值
            return len(a) == 0

        # 调用自定义方法 checkScript 来验证 func2 函数的行为，不传入参数
        self.checkScript(func2, ())

    # 使用装饰器 @skipIfTorchDynamo，指示在 TorchDynamo 环境下跳过以下测试
    @skipIfTorchDynamo(
        "TorchDynamo fails to raise on this checkScriptRaisesRegex, because we trace it properly now"
    )
    # 定义测试方法 test_list_ops，用于测试不同列表操作的函数
    def test_list_ops(self):
        
        # 定义测试函数 test_equality，测试列表的相等性
        def test_equality():
            a = [1, 2, 3]
            b = [1, 2, 3]
            return a == b
        
        # 调用 self.checkScript 方法，测试 test_equality 函数，优化开启
        self.checkScript(test_equality, (), optimize=True)
        
        # 定义测试函数 test_equality_str，测试字符串列表的相等性
        def test_equality_str():
            a = ["foo", "bar"]
            b = ["foo", "bar"]
            return a == b
        
        # 调用 self.checkScript 方法，测试 test_equality_str 函数，优化开启
        self.checkScript(test_equality_str, (), optimize=True)
        
        # 定义测试函数 test_inequality，测试列表的不等性
        def test_inequality():
            a = [1, 2, 3]
            b = [1, 2, 3]
            return a != b
        
        # 调用 self.checkScript 方法，测试 test_inequality 函数，优化开启
        self.checkScript(test_inequality, (), optimize=True)
        
        # 定义测试函数 test_inequality_str，测试字符串列表的不等性
        def test_inequality_str():
            a = ["foo", "bar"]
            b = ["foo", "bar", "food"]
            return a != b
        
        # 调用 self.checkScript 方法，测试 test_inequality_str 函数，优化开启
        self.checkScript(test_inequality_str, (), optimize=True)
        
        # 定义测试函数 test_non_equality，测试不相等的列表
        def test_non_equality():
            a = [1, 2, 3]
            b = [3]
            return a == b
        
        # 调用 self.checkScript 方法，测试 test_non_equality 函数，优化开启
        self.checkScript(test_non_equality, (), optimize=True)
        
        # 定义测试函数 test_non_inequality，测试不相等的列表
        def test_non_inequality():
            a = [1, 2, 3]
            b = [3]
            return a != b
        
        # 调用 self.checkScript 方法，测试 test_non_inequality 函数，优化开启
        self.checkScript(test_non_inequality, (), optimize=True)
        
        # 定义测试函数 test_list_equality_as_cond，测试列表相等作为条件
        def test_list_equality_as_cond():
            a = [1, 2, 3]
            b = [3]
            if a == b:
                c = 1
            else:
                c = 2
            return c
        
        # 调用 self.checkScript 方法，测试 test_list_equality_as_cond 函数，优化开启
        self.checkScript(test_list_equality_as_cond, (), optimize=True)
        
        # 定义测试函数 test_list_add，测试列表的加法操作
        def test_list_add():
            a = [1, 2, 3]
            b = [2]
            c = a + b
            return c == [1, 2, 3, 2]
        
        # 调用 self.checkScript 方法，测试 test_list_add 函数，优化开启
        self.checkScript(test_list_add, (), optimize=True)
        
        # 定义测试函数 test_list_add_empty，测试将空列表加到列表中
        def test_list_add_empty():
            a = [1, 2, 3]
            b = torch.jit.annotate(List[int], [])
            c = a + b
            return c == [1, 2, 3]
        
        # 调用 self.checkScript 方法，测试 test_list_add_empty 函数，优化开启
        self.checkScript(test_list_add_empty, (), optimize=True)
        
        # 定义测试函数 test_tensor_list_equality，测试张量列表的相等性
        def test_tensor_list_equality():
            t1 = torch.ones([1, 1])
            t2 = torch.ones([1, 1])
            x = [t1, t2]
            y = [t2, t1]
            return x == y
        
        # 调用 self.checkScript 方法，测试 test_tensor_list_equality 函数，优化开启
        self.checkScript(test_tensor_list_equality, (), optimize=True)
        
        # 定义测试函数 test_invalid_list_equality，测试无效的张量列表相等性
        def test_invalid_list_equality():
            t1 = torch.ones([2, 2])
            t2 = torch.ones([2, 2])
            x = [t1, t2]
            y = [t2, t1]
            # 由于张量具有多个元素，将抛出异常
            return x == y
        
        # 调用 self.checkScriptRaisesRegex 方法，测试 test_invalid_list_equality 函数抛出指定异常
        self.checkScriptRaisesRegex(
            test_invalid_list_equality, (), RuntimeError, "Boolean value of Tensor"
        )
    def test_list_sort(self):
        # 定义一个模板字符串，用于动态生成测试函数代码
        template = dedent(
            """
        def func():
            # 创建三个列表，并使用动态传入的列表创建语句填充它们
            li_1 = {list_create}
            li_2 = {list_create}
            li_3 = {list_create}
            # 对第一个列表进行升序排序
            li_1.sort()
            # 对第二个列表进行降序排序
            li_2.sort(reverse=True)
            # 对第三个列表进行排序，并将结果保存到一个新列表中
            li_4 = sorted(li_3)
            return li_1, li_2, li_3, li_4
        """
        )

        # 准备多个不同的列表初始化语句作为测试数据
        lists = [
            "[]",
            "[1, 3, 2]",
            "[True, False, True]",
            "[1.2, .2, 3.2]",
            "[torch.tensor(1.0), torch.tensor(0.2), torch.tensor(0.5)]",
            "[torch.tensor(5), torch.tensor(-2), torch.tensor(4)]",
        ]
        # 对每个列表初始化语句进行测试
        for li in lists:
            # 使用模板字符串填充具体的列表初始化语句，生成完整的测试代码字符串
            code = template.format(list_create=li)
            # 创建一个空的命名空间用于执行动态生成的代码
            scope = {}
            # 在全局命名空间和空的局部命名空间中执行动态生成的代码，将结果保存到scope中
            exec(code, globals(), scope)
            # 使用torch.jit.CompilationUnit将代码字符串编译成可执行单元
            cu = torch.jit.CompilationUnit(code)
            # 调用编译单元中的func函数，获取返回值t1
            t1 = cu.func()
            # 调用scope中的func函数，获取返回值t2
            t2 = scope["func"]()
            # 断言t1和t2的值相等
            self.assertEqual(t1, t2)

        # 定义一个测试函数，对包含Tensor的列表进行排序，并验证是否会抛出异常
        def test_fail(x: List[Tensor]) -> List[Tensor]:
            # 对传入的列表进行排序操作
            x.sort()
            return x

        # 使用self.checkScriptRaisesRegex方法测试test_fail函数的异常情况
        self.checkScriptRaisesRegex(
            test_fail,
            (([torch.zeros([2]), torch.zeros([2])],)),
            Exception,
            "Boolean value of Tensor with more than one value",
        )

        # 使用torch.jit.script装饰器定义一个测试函数，测试列表的变异操作
        @torch.jit.script
        def test_mutation():
            # 创建一个列表并对其进行排序操作
            a = [1, 2, 3]
            a.sort()
            return a

        # 调用test_mutation函数
        test_mutation()
        # 使用FileCheck检查test_mutation函数的计算图中是否包含aten::sort操作
        FileCheck().check("aten::sort").run(test_mutation.graph_for())

        # 定义一个测试函数，测试对包含Tensor的列表进行sorted操作后的结果
        def test_sorted_copy():
            a = [torch.tensor(2), torch.tensor(0), torch.tensor(1)]
            # 使用sorted函数对列表a进行排序，结果保存到b中
            b = sorted(a)
            # 修改列表a中的第一个元素
            a[0] = torch.tensor(10)
            return a, b

        # 使用self.checkScript方法测试test_sorted_copy函数
        self.checkScript(test_sorted_copy, ())

    def test_list_slice(self):
        # 定义一个测试正常切片的函数
        def test_regular_slice():
            # 创建一个包含整数的列表a，并返回切片结果是否为预期列表
            a = [0, 1, 2, 3, 4]
            return a[2:3] == [2]

        # 使用self.checkScript方法测试test_regular_slice函数
        self.checkScript(test_regular_slice, ())

        # 定义一个测试开放式结束切片的函数
        def test_open_ended_slice():
            # 创建一个包含整数的列表a，并返回切片结果是否为预期列表
            a = [0, 1, 2, 3, 4]
            return a[2:] == [2, 3, 4]

        # 使用self.checkScript方法测试test_open_ended_slice函数
        self.checkScript(test_open_ended_slice, ())

        # 定义另一个测试开放式开始切片的函数
        def test_open_ended_slice2():
            # 创建一个包含整数的列表a，并返回切片结果是否为预期列表
            a = [0, 1, 2, 3, 4]
            return a[:2] == [0, 1]

        # 使用self.checkScript方法测试test_open_ended_slice2函数
        self.checkScript(test_open_ended_slice2, ())

        # 定义一个测试负数索引切片的函数
        def test_negative_slice():
            # 创建一个包含整数的列表a，并返回切片结果是否为预期列表
            a = [0, 1, 2, 3, 4]
            return a[:-1] == [0, 1, 2, 3]

        # 使用self.checkScript方法测试test_negative_slice函数
        self.checkScript(test_negative_slice, ())

        # 定义另一个测试负数索引切片的函数
        def test_negative_slice2():
            # 创建一个包含整数的列表a，并返回切片结果是否为预期列表
            a = [0, 1, 2, 3, 4]
            return a[-3:-1] == [2, 3]

        # 使用self.checkScript方法测试test_negative_slice2函数
        self.checkScript(test_negative_slice2, ())

        # 定义一个测试逆向切片的函数
        def test_backward_slice():
            # 创建一个包含整数的列表a，并返回切片结果是否为空列表
            a = [0, 1, 2, 3, 4]
            return a[3:2] == torch.jit.annotate(List[int], [])

        # 使用self.checkScript方法测试test_backward_slice函数
        self.checkScript(test_backward_slice, ())

        # 定义一个测试超出索引切片的函数
        def test_over_slice():
            # 创建一个包含整数的列表a，并返回切片结果是否为预期列表
            a = [0, 1, 2, 3, 4]
            return a[3:10] == [3, 4]

        # 使用self.checkScript方法测试test_over_slice函数
        self.checkScript(test_over_slice, ())
    def test_slice_index(self):
        a = torch.tensor(
            [
                [[1, 11], [2, 22]],
                [[3, 33], [4, 44]],
                [[5, 55], [6, 66]],
            ]
        )

        # 定义测试函数 test_index_slice1，对输入张量进行切片操作，保留所有维度的前两个元素
        def test_index_slice1(x):
            x = x[:, :, [0, 1]]
            return x

        self.checkScript(test_index_slice1, (a,))

        # 定义测试函数 test_index_slice2，对输入张量进行索引操作，交换第一维度的顺序
        def test_index_slice2(x):
            x = x[[2, 1, 0], :, :]
            return x

        self.checkScript(test_index_slice2, (a,))

        # 定义测试函数 test_index_slice3，对输入张量进行混合的切片和索引操作
        def test_index_slice3(x):
            x = x[[0, 1], :, [1]]
            return x

        self.checkScript(test_index_slice3, (a,))

        # 定义测试函数 test_index_slice_empty_list，使用空列表对输入张量进行索引操作
        def test_index_slice_empty_list(x):
            empty_list: List[int] = []
            x = x[empty_list, :, :]
            return x

        self.checkScript(test_index_slice_empty_list, (a,))

        # 定义测试函数 test_index_slice_out_of_bounds_index，测试超出边界索引的情况
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "index 4 is out of bounds for dimension 0 with size 3",
            "x[[4], :, :]",
        ):
            self.checkScript(test_index_slice_out_of_bounds_index, (a,))

    def test_mutable_list_append(self):
        # 定义测试函数 test_append，测试列表的追加操作
        def test_append():
            a = [0, 1]
            a.append(2)
            a.append(3)
            return a == [0, 1, 2, 3]

        self.checkScript(test_append, ())

    def test_comprehensions_basic(self):
        # 定义测试函数 comp，使用列表推导式对整数列表进行操作
        def comp(l: List[int]) -> List[int]:
            n = [x * 3 for x in l]
            return n

        comp([1, 2, 3])
        self.checkScript(comp, ([1, 2, 3],))

    def test_comprehensions_basic_float(self):
        # 定义测试函数 comp，使用列表推导式对浮点数列表进行操作
        def comp(l: List[float]) -> List[float]:
            n = [x * 3 for x in l]
            return n

        self.checkScript(comp, ([1.0, 2.0, 3.0],))

    def test_comprehensions_two_comps(self):
        # 定义 Torch 脚本函数 comp，同时对两个整数列表进行列表推导式操作
        @torch.jit.script
        def comp(l1: List[int], l2: List[int]) -> List[int]:
            n = [x * 3 for x in l1]
            n2 = [x + 2 for x in l2]
            return n + n2

        self.assertEqual(comp([1, 2, 3], [4, 5]), [3, 6, 9, 6, 7])

    def test_comprehension_out_type_not_in_type(self):
        # 定义测试函数 list_cast，测试列表推导式中对张量列表进行强制类型转换
        def list_cast() -> int:
            li = [int(i) for i in [torch.tensor(0), torch.tensor(1), torch.tensor(2)]]
            return li[0] + li[1] + li[2]

        self.checkScript(list_cast, ())
    def test_comprehension_iterable(self):
        # 定义一个函数，用于测试给定函数在Torch脚本中的行为是否与原始函数一致
        def test_func(fn, inputs):
            self.assertEqual(fn(*inputs), torch.jit.script(fn)(*inputs))

        # 定义一个函数，接受两个列表参数，返回由元组组成的列表
        def foo(names: List[int], results: List[int]) -> List[Tuple[int, int]]:
            return [(k + 5, v - 2) for k, v in zip(names, results)]

        # 测试foo函数的行为
        test_func(foo, ([1, 2, 4], [4, 7, 9]))
        test_func(foo, ([5], [4, 7, 9]))

        # 定义一个函数，接受一个整数参数，返回由0到x-1的整数组成的列表
        def fn(x: int) -> List[int]:
            return [i for i in range(x)]  # noqa: C416

        # 测试fn函数的行为
        test_func(fn, (9,))
        test_func(fn, (0,))
        test_func(fn, (-1,))

        # 定义一个函数，创建三个列表，并将整数列表转换为浮点数列表
        def changes_type():
            a = [float(i) for i in range(5)]
            b = [float(i) for i in [1, 2, 3, 4]]
            c = [(float(i), j) for i, j in enumerate([1, 2, 3, 8])]
            return a, b, c

        # 测试changes_type函数的行为
        test_func(changes_type, ())

        # 定义一个函数，返回一个空列表，使用空字符串迭代，预期结果为空列表
        def test_zero_iter():
            return [str(i) for i, j in zip("", "")]

        # 测试test_zero_iter函数的行为
        test_func(test_zero_iter, ())



    def test_mutable_list_append_2(self):
        # 定义一个测试函数，验证列表的append方法和重新赋值的行为
        def test_append_2():
            a = [0, 1]
            a.append(2)  # 向列表a中添加元素2
            a = [1]  # 重新赋值a为列表[1]
            a.append(4)  # 向列表a中添加元素4
            return a == [1, 4]  # 预期a等于[1, 4]

        self.checkScript(test_append_2, ())



    def test_mutable_list_append_if(self):
        # 定义一个测试函数，验证在条件成立时向列表中添加元素的行为
        def test_append_if():
            a = [1]
            if 1 == 1:
                a.append(4)  # 如果条件成立，向列表a中添加元素4
            return a == [1, 4]  # 预期a等于[1, 4]

        self.checkScript(test_append_if, ())



    def test_mutable_list_append_if_else(self):
        # 定义一个测试函数，验证根据条件向列表中添加不同元素的行为
        def test_append_if_else():
            a = [1]
            if 1 == 2:
                a.append(4)
            else:
                a.append(10)  # 如果条件不成立，向列表a中添加元素10
            return a == [1, 10]  # 预期a等于[1, 10]

        self.checkScript(test_append_if_else, ())



    def test_mutable_list_append_loop(self):
        # 定义一个测试函数，使用循环向列表中添加元素
        def test_append_loop():
            a = torch.jit.annotate(List[int], [])
            for i in range(5):
                a.append(i)  # 向列表a中依次添加0到4这五个整数

            return a == [0, 1, 2, 3, 4]  # 预期a等于[0, 1, 2, 3, 4]

        self.checkScript(test_append_loop, ())



    def test_mutable_list_append_loop_if(self):
        # 定义一个测试函数，使用循环和条件语句向列表中添加元素
        def test_append_loop_if():
            a = torch.jit.annotate(List[int], [])
            for i in range(5):
                if i > 3:
                    a.append(i)  # 如果i大于3，向列表a中添加i
                else:
                    a.append(0)  # 否则向列表a中添加0

            return a == [0, 0, 0, 0, 4]  # 预期a等于[0, 0, 0, 0, 4]

        self.checkScript(test_append_loop_if, ())



    def test_mutable_list_nested_loop(self):
        # 定义一个测试函数，使用嵌套循环向列表中添加元素
        def test_nested_loop():
            a = torch.jit.annotate(List[int], [])
            for i in range(2):
                for j in range(2):
                    a.append(i + j)  # 向列表a中添加i+j的结果

            return a == [0, 1, 1, 2]  # 预期a等于[0, 1, 1, 2]

        self.checkScript(test_nested_loop, ())



    def test_mutable_list_function_inline(self):
        # 使用Torch脚本定义一个函数bar，向传入的列表参数中添加元素4
        @torch.jit.script
        def bar(y: List[int]) -> None:
            y.append(4)

        # 使用Torch脚本定义一个函数foo，创建一个列表x，调用bar函数向x中添加元素，并返回x
        @torch.jit.script
        def foo():
            x = [1, 2, 3]
            bar(x)
            return x

        # 验证调用foo函数的结果是否为预期的[1, 2, 3, 4]
        self.assertEqual(foo(), [1, 2, 3, 4])
    # 定义测试方法：测试对空列表进行反转操作
    def test_mutable_list_reverse_empty(self):
        # 定义内部测试函数：测试空列表的反转
        def test_reverse_empty():
            # 创建空列表
            a = []
            # 对列表进行反转操作
            a.reverse()

            # 返回反转后的列表是否为空列表的布尔结果
            return a == []

        # 调用测试框架中的检查函数，验证 test_reverse_empty 函数的行为
        self.checkScript(test_reverse_empty, ())

    # 定义测试方法：测试对非空列表进行反转操作
    def test_mutable_list_reverse(self):
        # 定义内部测试函数：测试非空列表的反转
        def test_reverse():
            # 创建包含元素的列表
            a = [1, 2, 3, 4]
            # 对列表进行反转操作
            a.reverse()

            # 返回反转后的列表是否与预期的倒序列表相同的布尔结果
            return a == [4, 3, 2, 1]

        # 调用测试框架中的检查函数，验证 test_reverse 函数的行为
        self.checkScript(test_reverse, ())

    # 定义测试方法：测试包含张量元素的列表进行反转操作
    def test_mutable_tensor_list_reverse(self):
        # 定义内部测试函数：测试包含张量元素的列表的反转
        def test_tensor_reverse():
            # 创建包含张量的列表
            a = [torch.tensor(1), torch.tensor(2)]
            # 对列表进行反转操作
            a.reverse()

            # 返回反转后的列表是否与预期的张量倒序列表相同的布尔结果
            return a == [torch.tensor(2), torch.tensor(1)]

        # 调用测试框架中的检查函数，验证 test_tensor_reverse 函数的行为
        self.checkScript(test_tensor_reverse, ())

    # 定义测试方法：测试从空列表中弹出元素的行为
    def test_mutable_list_pop_empty(self):
        # 使用 Torch 脚本装饰器定义函数：测试从空列表中弹出元素
        @torch.jit.script
        def test_pop_empty():
            # 创建空整数列表
            a = torch.jit.annotate(List[int], [])
            # 尝试从空列表中弹出元素
            return a.pop()

        # 使用断言检查是否抛出预期的运行时错误异常
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "pop from empty list", "a.pop"
        ):
            # 调用测试弹出空列表函数，期望捕获到运行时错误异常
            test_pop_empty()

    # 定义测试方法：测试从非空列表中弹出元素的行为
    def test_mutable_list_pop(self):
        # 定义内部测试函数：测试从非空列表中弹出元素
        def test_pop():
            # 创建包含元素的列表
            a = [1, 2, 3, 4]
            # 弹出列表的最后一个元素
            b = a.pop()

            # 返回被弹出的元素是否等于预期的值的布尔结果
            return b == 4

        # 调用测试框架中的检查函数，验证 test_pop 函数的行为
        self.checkScript(test_pop, ())

    # 定义测试方法：测试从非空列表中弹出元素后列表长度的变化
    def test_mutable_list_pop2(self):
        # 定义内部测试函数：测试从非空列表中弹出元素后列表长度的变化
        def test_pop2():
            # 创建包含元素的列表
            a = [1, 2, 3, 4]
            # 弹出列表的最后一个元素
            b = a.pop()

            # 返回弹出元素后列表的长度是否等于预期的值的布尔结果
            return len(a) == 3

        # 调用测试框架中的检查函数，验证 test_pop2 函数的行为
        self.checkScript(test_pop2, ())

    # 定义测试方法：测试从非空列表中指定位置弹出元素的行为
    def test_mutable_list_pop_at(self):
        # 定义内部测试函数：测试从非空列表中指定位置弹出元素
        def test_pop_at():
            # 创建包含元素的列表
            a = [1, 2, 3, 4]
            # 弹出列表指定位置的元素
            b = a.pop(1)

            # 返回被弹出的元素是否等于预期的值的布尔结果
            return b == 2

        # 调用测试框架中的检查函数，验证 test_pop_at 函数的行为
        self.checkScript(test_pop_at, ())

    # 定义测试方法：测试从非空列表中指定位置弹出元素后列表长度的变化
    def test_mutable_list_pop_at2(self):
        # 定义内部测试函数：测试从非空列表中指定位置弹出元素后列表长度的变化
        def test_pop_at2():
            # 创建包含元素的列表
            a = [1, 2, 3, 4]
            # 弹出列表指定位置的元素
            b = a.pop(1)

            # 返回弹出元素后列表的长度是否等于预期的值的布尔结果
            return len(a) == 3

        # 调用测试框架中的检查函数，验证 test_pop_at2 函数的行为
        self.checkScript(test_pop_at2, ())

    # 定义测试方法：测试从非空列表中使用负数索引弹出元素的行为
    def test_mutable_list_pop_at_negative(self):
        # 定义内部测试函数：测试从非空列表中使用负数索引弹出元素
        def test_pop_at_negative():
            # 创建包含元素的列表
            a = [1, 2, 3, 4]
            # 弹出列表指定负数索引位置的元素
            b = a.pop(-2)

            # 返回被弹出的元素是否等于预期的值的布尔结果
            return b == 3

        # 调用测试框架中的检查函数，验证 test_pop_at_negative 函数的行为
        self.checkScript(test_pop_at_negative, ())

    # 定义测试方法：测试从非空列表中使用负数索引弹出元素后列表长度的变化
    def test_mutable_list_pop_at_negative2(self):
        # 定义内部测试函数：测试从非空列表中使用负数索引弹出元素后列表长度的变化
        def test_pop_at_negative2():
            # 创建包含元素的列表
            a = [1, 2, 3, 4]
            # 弹出列表指定负数索引位置的元素
            b = a.pop(-2)

            # 返回弹出元素后列表的长度是否等于预期的值的布尔结果
            return len(a) == 3

        # 调用测试框架中的检查函数，验证 test_pop_at_negative2 函数的行为
        self.checkScript(test_pop_at_negative2, ())

    # 定义测试方法：测试从非空列表中使用切片删除元素的行为
    def test_mutable_list_pop_slice(self):
        # 定义内部测试函数：测试从非空列表中使用切片删除元素
        def test_pop_slice():
            # 创建两个相同的包含元素的列表
            a = [1, 2, 3, 4]
            b = [1, 2, 3, 4]

            # 使用 pop 方法删除列表的最后一个元素
            a.pop()
            # 使用切片删除列表的最后一个元素
            b = b[:-1]

            # 返回删除元素后的两个列表是否相等的布尔结果
            return a == b

        # 调用测试框架中的检查函数，验证 test_pop_slice 函数的行为
        self.checkScript(test_pop_slice, ())

    # 定义测试方法：测试清空空列表的行为
    def test_mutable_list_clear_empty(self):
        # 定义内部测试函数：测试清空空列表
    # 定义测试函数，验证可变列表的 insert 方法的行为
    def test_mutable_list_insert(self):
        # 定义内部测试函数，测试列表的插入操作
        def test_list_insert():
            # 创建包含整数的列表
            a = [1, 2, 3, 4]
            # 在索引 2 处插入整数 5
            a.insert(2, 5)

            # 检查插入操作后列表是否符合预期
            return a == [1, 2, 5, 3, 4]

        # 调用自定义的检查脚本方法，验证 test_list_insert 函数的行为
        self.checkScript(test_list_insert, ())

    # 定义测试函数，验证可变列表的 insert 方法在负索引情况下的行为
    def test_mutable_list_insert_negative(self):
        # 定义内部测试函数，测试负索引插入操作
        def test_list_insert_negative():
            # 创建包含整数的列表
            a = [1, 2, 3, 4]
            # 在倒数第一个位置（索引 -1）插入整数 5
            a.insert(-1, 5)

            # 检查插入操作后列表是否符合预期
            return a == [1, 2, 3, 5, 4]

        # 调用自定义的检查脚本方法，验证 test_list_insert_negative 函数的行为
        self.checkScript(test_list_insert_negative, ())

    # 定义测试函数，验证可变列表的 insert 方法在超出负索引范围时的行为
    def test_mutable_list_insert_neg_out_of_bounds(self):
        # 定义内部测试函数，测试超出负索引范围的插入操作
        def test_list_insert_neg_out_of_bounds():
            # 创建包含整数的列表
            a = [1, 2, 3, 4]
            # 在超出列表长度的负索引位置（索引 -10）插入整数 5
            a.insert(-10, 5)

            # 检查插入操作后列表是否符合预期
            return a == [5, 1, 2, 3, 4]

        # 调用自定义的检查脚本方法，验证 test_list_insert_neg_out_of_bounds 函数的行为
        self.checkScript(test_list_insert_neg_out_of_bounds, ())

    # 定义测试函数，验证可变列表的 insert 方法在超出正索引范围时的行为
    def test_mutable_list_insert_out_of_bounds(self):
        # 定义内部测试函数，测试超出正索引范围的插入操作
        def test_list_insert_out_of_bounds():
            # 创建包含整数的列表
            a = [1, 2, 3, 4]
            # 在超出列表长度的正索引位置（索引 10）插入整数 5
            a.insert(10, 5)

            # 检查插入操作后列表是否符合预期
            return a == [1, 2, 3, 4, 5]

        # 调用自定义的检查脚本方法，验证 test_list_insert_out_of_bounds 函数的行为
        self.checkScript(test_list_insert_out_of_bounds, ())

    # 定义测试函数，验证可变列表的 remove 方法在移除不存在元素时的行为
    def test_mutable_list_remove_not_existing(self):
        # 使用 Torch Script 标记的内部测试函数，测试移除不存在元素的操作
        @torch.jit.script
        def test_list_remove_not_existing():
            # 创建包含整数的列表
            a = [1, 2, 3, 4]
            # 尝试移除不存在的整数 5
            a.remove(5)

            # 返回移除操作后的列表
            return a

        # 使用断言验证是否抛出期望的异常信息
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "x not in list", "a.remove"
        ):
            # 调用 test_list_remove_not_existing 函数，期望抛出运行时异常
            test_list_remove_not_existing()

    # 定义测试函数，验证可变列表的 remove 方法在正常移除元素时的行为
    def test_mutable_list_remove(self):
        # 定义内部测试函数，测试正常移除元素的操作
        def test_list_remove():
            # 创建包含整数的列表
            a = [1, 2, 3, 4]
            # 移除整数 3
            a.remove(3)

            # 检查移除操作后列表是否符合预期
            return a == [1, 2, 4]

        # 调用自定义的检查脚本方法，验证 test_list_remove 函数的行为
        self.checkScript(test_list_remove, ())

        # 定义另一个内部测试函数，测试移除字符串元素的操作
        def test_str_list_remove():
            # 创建包含字符串的列表
            a = ["foo", "bar"]
            # 移除字符串 "foo"
            a.remove("foo")

            # 检查移除操作后列表是否符合预期
            return a == ["bar"]

        # 调用自定义的检查脚本方法，验证 test_str_list_remove 函数的行为
        self.checkScript(test_str_list_remove, ())

    # 定义使用 Torch Script 标记的测试函数，验证列表的 index 方法在未找到元素时的行为
    def test_list_index_not_existing(self):
        @torch.jit.script
        def list_index_not_existing():
            # 创建包含整数的列表
            a = [4, 1, 3, 2]
            # 尝试查找整数 5 的索引
            i = a.index(5)

            # 返回查找结果（预期会抛出异常）
            return i

        # 使用断言验证是否抛出期望的异常信息
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "'5' is not in list", "a.index"
        ):
            # 调用 list_index_not_existing 函数，期望抛出运行时异常
            list_index_not_existing()

    # 定义测试函数，验证列表的 index 方法在正常情况下的行为
    def test_list_index(self):
        # 定义内部测试函数，测试正常查找元素的索引操作
        def list_index():
            # 创建包含整数的列表
            a = [4, 1, 3, 2]
            # 查找整数 3 的索引
            i = a.index(3)

            # 检查索引操作返回的结果是否符合预期
            return i == 2

        # 调用自定义的检查脚本方法，验证 list_index 函数的行为
        self.checkScript(list_index, ())

        # 定义另一个内部测试函数，测试查找字符串元素的索引操作
        def list_str_index():
            # 创建包含字符串的列表
            a = ["foo", "bar"]
            # 查找字符串 "bar" 的索引
            i = a.index("bar")

            # 检查索引操作返回的结果是否符合预期
            return i == 1

        # 调用自定义的检查脚本方法，验证 list_str_index 函数的行为
        self.checkScript(list_str_index, ())

    # 定义测试函数，验证列表中包含张量元素时的 index 方法行为
    def test_tensor_list_index(self):
        # 定义内部测试函数，测试包含张量元素时的查找索引操作
        def tensor_list_index():
            # 创建包含张量的列表
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(3), torch.tensor(2)]
            # 查找张量 torch.tensor(3) 的索引
            i = a.index(torch.tensor(3))

            # 检查索引操作返回的结果是否符合预期
            return i == 2

        # 调用自定义的检查脚本方法，验证 tensor_list_index 函数的行为
        self.checkScript(tensor_list_index, ())
    def test_tensor_list_index_not_existing(self):
        @torch.jit.script
        def tensor_list_index_not_existing():
            # 创建包含张量的列表
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(3), torch.tensor(2)]
            # 尝试查找张量 5 在列表中的索引
            i = a.index(torch.tensor(5))

            return i

        # 断言捕获 RuntimeError 异常，其中包含 "is not in list" 和 "a.index"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "is not in list", "a.index"
        ):
            tensor_list_index_not_existing()

    def test_list_count(self):
        def list_count():
            # 创建包含整数的列表
            a = [4, 1, 4, 2, 4]
            # 统计数字 4 在列表中出现的次数
            i = a.count(4)

            return i == 3

        # 检查函数 list_count 是否符合 Torch 脚本
        self.checkScript(list_count, ())

        def list_str_count():
            # 创建包含字符串的列表
            a = ["foo", "bar", "foo"]
            # 统计字符串 "foo" 在列表中出现的次数
            i = a.count("foo")

            return i == 2

        # 检查函数 list_str_count 是否符合 Torch 脚本
        self.checkScript(list_str_count, ())

    def test_list_count_not_existing(self):
        def list_count_not_existing():
            # 创建包含整数的列表
            a = [4, 1, 4, 2, 4]
            # 统计数字 5 在列表中出现的次数
            i = a.count(5)

            return i == 0

        # 检查函数 list_count_not_existing 是否符合 Torch 脚本
        self.checkScript(list_count_not_existing, ())

    def test_tensor_list_count(self):
        def tensor_list_count():
            # 创建包含张量的列表
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(4), torch.tensor(4)]
            # 统计张量 torch.tensor(4) 在列表中出现的次数
            i = a.count(torch.tensor(4))

            return i == 3

        # 检查函数 tensor_list_count 是否符合 Torch 脚本
        self.checkScript(tensor_list_count, ())

    def test_tensor_list_count_not_existing(self):
        def tensor_list_count_not_existing():
            # 创建包含张量的列表
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(4), torch.tensor(4)]
            # 统计张量 torch.tensor(5) 在列表中出现的次数
            i = a.count(torch.tensor(5))

            return i == 0

        # 检查函数 tensor_list_count_not_existing 是否符合 Torch 脚本
        self.checkScript(tensor_list_count_not_existing, ())

    def test_mutable_list_remove_tensor(self):
        def test_list_remove_tensor():
            # 创建包含张量的列表
            a = [torch.ones(1), torch.zeros(1), torch.ones(2)]
            # 移除列表中的 torch.zeros(1) 张量
            a.remove(torch.zeros(1))

            return len(a) == 2

        # 检查函数 test_list_remove_tensor 是否符合 Torch 脚本
        self.checkScript(test_list_remove_tensor, ())

    def test_mutable_list_remove2(self):
        def test_list_remove2():
            # 创建包含整数的列表
            a = [1]
            # 移除列表中的整数 1
            a.remove(1)

            return len(a) == 0

        # 检查函数 test_list_remove2 是否符合 Torch 脚本
        self.checkScript(test_list_remove2, ())

    def test_extend_list_mutable(self):
        @torch.jit.script
        def extend_list(a: List[Tensor], b: List[Tensor]) -> List[Tensor]:
            # 将列表 b 扩展到列表 a 中
            a.extend(b)
            return a

        # 对多个输入列表进行扩展操作，检查结果是否符合预期
        for l in [[], [torch.rand(2)], [torch.rand(2), torch.rand(2), torch.rand(2)]]:
            for r in [
                [],
                [torch.rand(2)],
                [torch.rand(2), torch.rand(2), torch.rand(2)],
            ]:
                self.assertEqual(extend_list(l, r), l + r)

    def test_extend_list_immutable(self):
        @torch.jit.script
        def extend_list(a: List[int], b: List[int]) -> List[int]:
            # 将列表 b 扩展到列表 a 中
            a.extend(b)
            return a

        # 对多个输入列表进行扩展操作，检查结果是否符合预期
        for l in [[], [1], [1, 2, 3]]:
            for r in [[], [1], [1, 2, 3]]:
                self.assertEqual(extend_list(l, r), l + r)
    def test_copy_list_mutable(self):
        @torch.jit.script
        # 定义一个 Torch 脚本函数，用于复制可变列表
        def copy_list(a: List[Tensor]) -> List[Tensor]:
            return a.copy()

        for l in [[], [torch.rand(2)], [torch.rand(2), torch.rand(2), torch.rand(2)]]:
            # 断言复制前后列表相等
            self.assertEqual(copy_list(l), l)

    def test_copy_list_immutable(self):
        @torch.jit.script
        # 定义一个 Torch 脚本函数，用于复制不可变列表
        def copy_list(a: List[int]) -> List[int]:
            return a.copy()

        for l in [[], [1], [1, 2, 3]]:
            # 断言复制前后列表相等
            self.assertEqual(copy_list(l), l)

    def test_min_max_single_list(self):
        # 定义计算列表中最小值和最大值的函数，针对整数、布尔值和浮点数列表
        def min_intlist(li: List[int]) -> int:
            return min(li)

        def max_intlist(li: List[int]) -> int:
            return max(li)

        def min_boollist(li: List[bool]) -> bool:
            return min(li)

        def max_boollist(li: List[bool]) -> bool:
            return max(li)

        def min_floatlist(li: List[float]) -> float:
            return min(li)

        def max_floatlist(li: List[float]) -> float:
            return max(li)

        int_lists = [1], [2, 1, 2], [-3, 4, 2], [-2, -7, 1, 4], [2, 1, 0, 4], []

        # 定义一个检查计算函数（最小值或最大值）的结果是否符合预期的辅助函数
        def check_list(fn, li):
            if len(li) == 0:
                # 如果列表为空，检查脚本是否会引发异常
                self.checkScriptRaisesRegex(fn, (li,), Exception, "empty")
            else:
                # 否则，检查脚本函数的执行结果
                self.checkScript(fn, (li,))

        for int_list in int_lists:
            # 对每个整数列表进行最小值和最大值的检查
            check_list(min_intlist, int_list)
            check_list(max_intlist, int_list)

            # 将整数列表转换为布尔列表并进行最小值和最大值的检查
            bool_li = [bool(x) for x in int_list]
            check_list(min_boollist, bool_li)
            check_list(max_boollist, bool_li)

            # 将整数列表转换为浮点数列表并进行最小值和最大值的检查
            float_li = [float(x) for x in int_list]
            check_list(min_floatlist, float_li)
            check_list(max_floatlist, float_li)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_to_list_gpu(self):
        """GPU tests for Tensor.tolist() function."""

        # 定义将 GPU 上的 Tensor 转换为列表的函数，针对布尔值、整数和浮点数
        def to_list_bool_1D(x: torch.Tensor) -> List[bool]:
            li = torch.jit.annotate(List[bool], x.tolist())
            return li

        def to_list_int_1D(x: torch.Tensor) -> List[int]:
            li = torch.jit.annotate(List[int], x.tolist())
            return li

        def to_list_float_1D(x: torch.Tensor) -> List[float]:
            li = torch.jit.annotate(List[float], x.tolist())
            return li

        # 对每个函数进行 Torch 脚本的检查，输入为相应类型的 Tensor
        self.checkScript(
            to_list_bool_1D,
            (torch.tensor([True, False, True, False], dtype=torch.bool).cuda(),),
        )
        self.checkScript(
            to_list_int_1D, (torch.tensor([1, 2, 3, 4], dtype=torch.long).cuda(),)
        )
        self.checkScript(to_list_float_1D, (torch.randn(5, dtype=torch.double).cuda(),))
    # 定义测试函数，测试在没有元素类型注释的情况下的行为
    def test_no_element_type_annotation(self):
        # 定义带有注释的函数，参数 x 应为 torch.Tensor 类型，返回值类型为 List
        def fn_with_comment(x: torch.Tensor) -> List:
            # 将输入张量 x 转换为 Python 列表 a
            a: List = x.tolist()
            return a

        # 定义带有类型注释的函数，参数 x 应为 torch.Tensor 类型，返回值类型为 List
        def annotated_fn(x: torch.Tensor) -> List:
            # 将输入张量 x 转换为 Python 列表 a
            a: List = x.tolist()
            return a

        # 使用 torch.jit.CompilationUnit() 创建一个编译单元 cu，并定义 fn_with_comment 函数
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        # 使用 torch.jit.CompilationUnit() 创建一个编译单元 cu，并定义 annotated_fn 函数
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        # 使用 torch.jit.script 尝试将 fn_with_comment 函数编译为 Torch 脚本
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            torch.jit.script(fn_with_comment)

        # 使用 torch.jit.script 尝试将 annotated_fn 函数编译为 Torch 脚本
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            torch.jit.script(annotated_fn)

    # 定义测试函数，测试在创建 ListType 时使用 None 类型的行为
    def test_list_none(self):
        with self.assertRaisesRegex(
            RuntimeError, "Can not create ListType with None type"
        ):
            # 尝试创建一个 ListType，其类型为 None
            x = torch._C.ListType(None)

    # 定义测试函数，测试在函数注解中使用 List 时未提供正确的类型提示的行为
    def test_list_unification_hint(self):
        with self.assertRaisesRegex(
            RuntimeError, "Expected an annotation of type List"
        ):
            # 使用 Torch 脚本装饰器 torch.jit.script 注释函数 x
            @torch.jit.script
            def x():
                # 声明变量 b 为 int 类型，但赋值为列表 [2, 3]
                b: int = [2, 3]
                return b
class TestDict(JitTestCase):
    # 定义一个测试类 TestDict，继承自 JitTestCase
    
    def dict(self):
        # 定义一个返回字典的方法 dict，包含键 'a' 到 'c'，值为 torch.Tensor 对象
        return {"a": torch.ones(1), "b": torch.ones(1) + 1, "c": torch.ones(1) + 2}

    def dict2(self):
        # 定义一个返回字典的方法 dict2，包含键 'x' 到 'z'，值为 torch.Tensor 对象
        return {
            "x": torch.ones(1) + 100,
            "y": torch.ones(1) + 101,
            "z": torch.ones(1) + 102,
        }

    def dict_bool(self):
        # 定义一个返回包含 True 键的字典方法 dict_bool
        return {True: 1}

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_dict_bool_conversion(self):
        # 定义测试方法 test_dict_bool_conversion，检验字典转换为布尔值的行为
        def if_predicate(d: Dict[int, int]):
            # 定义条件判断函数 if_predicate，输入参数 d 是一个字典类型
            if d:
                # 如果字典 d 不为空
                s, t = 0, 0
                # 初始化两个计数器 s 和 t
                for k, v in d.items():
                    # 遍历字典 d 的键值对
                    s += k
                    t += v

                return s, t
                # 返回计算后的两个值
            else:
                # 如果字典 d 为空
                return -1, -1
                # 返回默认值 -1, -1

        self.checkScript(if_predicate, ({1: 2, 3: 5},))
        # 调用 checkScript 方法测试 if_predicate 函数行为
        self.checkScript(if_predicate, ({},))
        # 再次调用 checkScript 方法测试 if_predicate 函数行为

        def while_predicate(d: Dict[int, int]):
            # 定义条件判断函数 while_predicate，输入参数 d 是一个字典类型
            while d:
                # 当字典 d 不为空时
                d.clear()
                # 清空字典 d

        self.checkScript(while_predicate, ({1: 2, 3: 5},))
        # 调用 checkScript 方法测试 while_predicate 函数行为
        self.checkScript(while_predicate, ({},))
        # 再次调用 checkScript 方法测试 while_predicate 函数行为

        def ternary_predicate(d: Dict[int, int]):
            # 定义条件判断函数 ternary_predicate，输入参数 d 是一个字典类型
            return "non-empty" if d else "empty"
            # 如果字典 d 不为空，返回 "non-empty"，否则返回 "empty"

        self.checkScript(ternary_predicate, ({1: 2, 3: 5},))
        # 调用 checkScript 方法测试 ternary_predicate 函数行为
        self.checkScript(ternary_predicate, ({},))
        # 再次调用 checkScript 方法测试 ternary_predicate 函数行为

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_del(self):
        # 定义测试方法 test_del，测试删除字典中键的行为
        def inputs():
            # 定义返回字典的函数 inputs
            return {"hi": 2, "bye": 3}

        def fn(x: Dict[str, int]) -> Dict[str, int]:
            # 定义接受字典参数 x，返回字典的函数 fn
            del x["hi"]
            # 删除字典 x 中键为 'hi' 的项
            return x
            # 返回修改后的字典 x

        python_out = fn(inputs())
        # 调用 fn 函数处理 inputs 返回的字典
        # checkScript 方法会重用相同的对象，但这里发生了变异，所以手动处理
        cu = torch.jit.CompilationUnit()
        cu.define(dedent(inspect.getsource(fn)))
        self.assertEqual(cu.fn(inputs()), python_out)
        # 断言使用 CompilationUnit 编译的 fn 函数与 python_out 相等
        self.assertEqual(torch.jit.script(fn)(inputs()), python_out)
        # 断言使用 Torch 的 jit.script 编译的 fn 函数与 python_out 相等
        with self.assertRaisesRegexWithHighlight(RuntimeError, "KeyError", 'x["hi"]'):
            self.checkScript(fn, [{}])
            # 断言在空字典上调用 fn 会抛出 KeyError 异常

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    # 如果 TorchDynamo 在此测试中失败，跳过执行以下方法
    def test_dict_variance(self):
        """
        `Dict[T1, _]` is not a subtype of `Dict[T2, _]`, even if `T1` is
        a subtype of `T2`; similarly `Dict[_, T1]` would not be a
        subtype of `Dict[_, T2]`.

        However, if we have a temporary dict object (that is, a dict
        comprehension or a dict literal) on the rhs of an assignment
        statement, we want to ignore the inferred type of the rhs if we
        can prove that: 1) both the lhs and the rhs are dicts with the
        same key types (TorchScript has a restricted set of allowed key
        types, so we don't need to worry about subtyping relationships
        here), and 2) the value type of the dict is a subtype of the
        value type of the rhs dict.
        """

        def test_dictliteral_is_typed_from_annotation():
            # 创建一个字典，键为字符串，值为可选的整数，注解表明这些值可以为 None
            x: Dict[str, Optional[int]] = {"foo": None, "bar": None, "baz": None}
            return x

        self.checkScript(test_dictliteral_is_typed_from_annotation, ())

        def test_dictcomprehension_is_typed_from_annotation():
            # 使用字典推导式创建一个字典，键为字符串，值为可选的整数，所有值初始化为 None
            metasyntactics = ["foo", "bar", "baz"]
            x: Dict[str, Optional[int]] = {
                word: None for word in metasyntactics
            }  # noqa: RUF025
            return x

        self.checkScript(test_dictcomprehension_is_typed_from_annotation, ())

        def test_dicts_with_different_value_types_are_invariant(self):
            # 创建一个字典，键为字符串，值为整数
            x: Dict[str, int] = {"foo": 1, "bar": 2, "baz": 3}
            # 将值类型为整数的字典赋给值类型为可选整数的字典，应触发运行时错误
            y: Dict[str, Optional[int]] = x
            return x

        with self.assertRaisesRegex(
            RuntimeError,
            "Variable 'y' is "
            "annotated with type "
            r"Dict\[str, Optional\[int\]\] but "
            "is being assigned to a value of "
            r"type Dict\[str, int\]",
        ):
            torch.jit.script(test_dicts_with_different_value_types_are_invariant)

        def test_dicts_with_different_value_types_are_invariant_recursive(self):
            # 创建一个字典，键为字符串，值为整数
            x: Dict[str, int] = {"foo": 1, "bar": 2, "baz": 3}
            # 创建一个字典，键为字符串，值为字典（键为字符串，值为整数），将其赋给值类型为可选整数的字典，应触发运行时错误
            y: Dict[str, Dict[str, int]] = {"foo": x, "bar": x, "baz": x}
            z: Dict[str, Dict[str, Optional[int]]] = y
            return x

        with self.assertRaisesRegex(
            RuntimeError,
            "Variable 'z' is "
            "annotated with type "
            r"Dict\[str, Dict\[str, Optional"
            r"\[int\]\]\] but is being assigned"
            r" to a value of type Dict\[str, "
            r"Dict\[str, int\]\]",
        ):
            torch.jit.script(
                test_dicts_with_different_value_types_are_invariant_recursive
            )

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    `
        # 定义一个测试方法，测试字典的键
        def test_keys(self):
            # 使用 TorchScript 对键的提取进行编译
            @torch.jit.script
            def keys(x: Dict[str, Tensor]) -> List[str]:
                # 返回字典的键列表
                return list(x.keys())
    
            # 验证编译后的函数返回的键集合与原字典的键集合是否相同
            self.assertEqual(set(keys(self.dict())), set(self.dict().keys()))
    
            # 使用 TorchScript 对特定列表进行编译，测试其键的行为
            @torch.jit.script
            def specialized_list():
                li = {1: 1, 2: 2}.keys()
                li.append(3)  # 尝试在键集合中添加元素
                return li
    
            # 验证编译后的函数返回的键集合是否包含 1, 2, 和 3
            self.assertTrue(set(specialized_list()) == {1, 2, 3})
    
        # 跳过因 TorchDynamo 问题而无法运行的测试
        @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
        def test_values(self):
            # 使用 TorchScript 对值的提取进行编译
            @torch.jit.script
            def values(x: Dict[str, Tensor]) -> List[Tensor]:
                # 返回字典的值列表
                return list(x.values())
    
            # 获取测试字典
            the_dict = self.dict()
            # 验证编译后的函数返回的值集合与原字典的值集合是否相同
            self.assertEqual(set(values(the_dict)), set(the_dict.values()))
    
        # 跳过因 TorchDynamo 问题而无法运行的测试
        @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
        def test_len(self):
            # 定义一个函数，返回字典的长度
            def length(x: Dict[str, Tensor]) -> int:
                return len(x)
    
            # 使用 checkScript 验证函数是否可以被编译
            self.checkScript(length, (self.dict(),))
    
        # 跳过因 TorchDynamo 问题而无法运行的测试
        @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
        def test_copy(self):
            # 定义一个函数，复制字典
            def func(x: Dict[str, Tensor]) -> Dict[str, Tensor]:
                return x.copy()
    
            # 使用 checkScript 验证函数是否可以被编译
            self.checkScript(func, (self.dict(),))
    
        # 跳过因 TorchDynamo 问题而无法运行的测试
        @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
        def test_items(self):
            # 定义一个函数，返回字典的项
            def func(x: Dict[str, Tensor]) -> List[Tuple[str, Tensor]]:
                return x.items()
    
            # 将函数编译成 TorchScript
            scripted_func = torch.jit.script(func)
    
            # 获取原始函数和编译后的函数的输出
            eager_out = func(self.dict())
            script_out = scripted_func(self.dict())
    
            # 验证编译前后的输出长度是否相同
            self.assertEqual(len(eager_out), len(script_out))
            # 验证编译前后的输出项是否一致
            for item in eager_out:
                self.assertTrue(item in script_out)
    
        # 跳过因 TorchDynamo 问题而无法运行的测试
        @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
        def test_pop(self):
            # 定义一个函数，弹出字典中的指定键，并返回弹出值和更新后的字典
            def pop(x: Dict[str, Tensor], key: str) -> Tuple[Tensor, Dict[str, Tensor]]:
                return x.pop(key), x
    
            # 定义一个测试函数，比较 eager 和 scripted 函数的输出
            def tester(fn, *args):
                eager_out = fn(self.dict(), *args)
                script_out = torch.jit.script(fn)(self.dict(), *args)
                self.assertEqual(eager_out, script_out)
    
            # 测试 pop 函数的行为
            tester(pop, "a")
    
            # 验证在尝试弹出不存在的键时抛出 KeyError 异常
            with self.assertRaisesRegexWithHighlight(RuntimeError, "KeyError", "x.pop"):
                torch.jit.script(pop)(self.dict(), "x")
    
            # 定义一个函数，带有默认值的弹出操作
            def default_pop(
                x: Dict[str, Tensor], key: str, default: Tensor
            ) -> Tuple[Tensor, Dict[str, Tensor]]:
                return x.pop(key, default), x
    
            # 测试带有默认值的弹出操作
            tester(default_pop, "a", torch.randn(2, 2))
            tester(default_pop, "x", torch.randn(2, 2))
    
        # 跳过因 TorchDynamo 问题而无法运行的测试
        @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_setdefault(self):
        # 定义一个函数 setdefault，用于向字典中指定键设置默认值，并返回更新后的字典
        def setdefault(
            x: Dict[str, Tensor], key: str, default: Tensor
        ) -> Dict[str, Tensor]:
            x.setdefault(key, default)
            return x

        # 对 setdefault 函数进行脚本化检查，使用 self.dict() 作为初始字典，"a" 作为键，torch.randn(2, 2) 作为默认值
        self.checkScript(setdefault, (self.dict(), "a", torch.randn(2, 2)))
        # 对 setdefault 函数进行脚本化检查，使用 self.dict() 作为初始字典，"nonexistant" 作为键，torch.randn(2, 2) 作为默认值

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_update(self):
        # 定义一个函数 update，用于将字典 b 的所有项更新到字典 a 中，并返回更新后的字典 a 和字典 b
        def update(
            a: Dict[str, Tensor], b: Dict[str, Tensor]
        ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
            a.update(b)
            return a, b

        # 对 update 函数进行脚本化检查，使用 self.dict() 作为初始字典，以及另一个空字典作为更新的源字典
        self.checkScript(update, (self.dict(), self.dict()))
        # 对 update 函数进行脚本化检查，使用 self.dict() 作为初始字典，以及 self.dict2() 函数返回的字典作为更新的源字典

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_update_existing_key(self):
        # 定义一个函数 foo，返回一个字典，该字典中键 "a" 在循环中被更新为 0、1、2
        def foo() -> Dict[str, int]:
            a: Dict[str, int] = {}
            for i in range(3):
                a.update({"a": i})
            return a

        # 对 foo 函数进行脚本化检查，不传入任何参数
        self.checkScript(foo, ())

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_aug_assign(self):
        # 定义一个函数 aug_assign_dict_tensor，对字典中键 "a"、"b"、"c" 的值进行增减乘除取余操作，并返回更新后的字典
        def aug_assign_dict_tensor(a: Dict[str, Tensor]) -> Dict[str, Tensor]:
            a["a"] += 1
            a["b"] -= 12
            a["c"] *= 122
            a["c"] /= 2
            a["c"] %= 2
            return a

        # 定义一个函数 aug_assign_dict_prim，对字典中键 "a"、"b"、"c" 的值进行增减乘除取余操作，并返回更新后的字典
        def aug_assign_dict_prim(a: Dict[str, float]) -> Dict[str, float]:
            a["a"] += 3.4
            a["b"] -= 2.4
            a["c"] *= 3.0
            a["c"] /= 2.0
            a["c"] %= 2.0
            return a

        # 对 aug_assign_dict_tensor 函数进行脚本化检查，使用 self.dict() 作为初始字典
        self.checkScript(aug_assign_dict_tensor, (self.dict(),))
        # 对 aug_assign_dict_prim 函数进行脚本化检查，使用 {"a": 3.0, "b": 2.0, "c": 4.0} 字典作为初始字典

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_popitem(self):
        # 使用 Torch Script 标记定义一个函数 popitem，从字典中弹出一个项，并返回被弹出的项和更新后的字典
        @torch.jit.script
        def popitem(
            x: Dict[str, Tensor]
        ) -> Tuple[Tuple[str, Tensor], Dict[str, Tensor]]:
            item = x.popitem()
            return item, x

        # Python 中 popitem 方法返回的值是随机的，因此无法进行 checkScript 检查

        # 创建一个普通字典的副本，并弹出一个项
        eager_in = self.dict()
        eager_out = (eager_in.popitem(), eager_in)

        # 使用 Torch Script 脚本运行 popitem 函数，并比较结果
        script_out = popitem(self.dict())

        # 检查 pop 后字典的长度是否相等
        self.assertEqual(len(eager_out[1]), len(script_out[1]))

        # 检查弹出的项是否具有正确的类型
        self.assertTrue(isinstance(script_out[0][0], str))
        self.assertTrue(isinstance(script_out[0][1], torch.Tensor))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_clear(self):
        # 定义一个函数 clear，清空字典并返回空的字典
        def clear(x: Dict[str, Tensor]) -> Dict[str, Tensor]:
            x.clear()
            return x

        # 对 clear 函数进行脚本化检查，使用 self.dict() 作为初始字典
        self.checkScript(clear, (self.dict(),))
    # 定义一个测试方法，用于测试获取字典中指定键的值的函数 get
    def test_get(self):
        # 定义函数 get，接受一个字典 x 和一个键 key，返回键对应的值或 None
        def get(x: Dict[str, Tensor], key: str) -> Optional[Tensor]:
            return x.get(key)

        # 调用 self.checkScript 方法，验证 get 函数的行为，期望返回指定键的值
        self.checkScript(get, (self.dict(), "a"))
        # 再次调用 self.checkScript 方法，验证 get 函数对于不存在的键的行为，期望返回 None
        self.checkScript(get, (self.dict(), "doesn't exist"))

        # 定义函数 get_default，类似于 get，但是如果键不存在，则返回一个随机生成的 2x2 的张量
        def get_default(x: Dict[str, Tensor], key: str) -> Optional[Tensor]:
            return x.get(key, torch.randn(2, 2))

        # 调用 self.checkScript 方法，验证 get_default 函数的行为，期望返回指定键的值或随机生成的张量
        self.checkScript(get, (self.dict(), "a"))
        # 再次调用 self.checkScript 方法，验证 get_default 函数对于不存在的键的行为，期望返回随机生成的张量
        self.checkScript(get, (self.dict(), "doesn't exist"))

    # 跳过由于 TorchDynamo 未知原因导致的测试失败
    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    # 定义测试方法，测试处理布尔键的函数 get_boolkey
    def test_get_boolkey(self):
        # 定义函数 get，接受一个字典 x 和一个布尔键 key，返回键对应的值或 None
        def get(x: Dict[bool, int], key: bool) -> Optional[int]:
            return x.get(key)

        # 调用 self.checkScript 方法，验证 get 函数的行为，期望返回指定键的值
        self.checkScript(get, (self.dict_bool(), True))
        # 再次调用 self.checkScript 方法，验证 get 函数对于不存在的键的行为，期望返回 None
        self.checkScript(get, (self.dict_bool(), False))

        # 定义函数 get_default，类似于 get，但是如果键不存在，则返回默认值 42
        def get_default(x: Dict[bool, int], key: bool) -> int:
            return x.get(key, 42)

        # 调用 self.checkScript 方法，验证 get_default 函数的行为，期望返回指定键的值或默认值 42
        self.checkScript(get_default, (self.dict_bool(), True))
        # 再次调用 self.checkScript 方法，验证 get_default 函数对于不存在的键的行为，期望返回默认值 42
        self.checkScript(get_default, (self.dict_bool(), False))

    # 跳过由于 TorchDynamo 未知原因导致的测试失败
    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    # 定义基础测试方法，测试简单字典操作的函数 simple
    def test_basic(self):
        # 定义函数 simple，接受一个字典 x，直接返回该字典
        def simple(x: Dict[str, int]) -> Dict[str, int]:
            return x

        # 调用 self.checkScript 方法，验证 simple 函数的行为，期望返回输入的字典本身
        self.checkScript(simple, ({"item": 20, "other_item": 120},))

        # 定义函数 index，接受一个字典 x，返回指定键 "item" 对应的值
        def index(x: Dict[str, int]) -> int:
            return x["item"]

        # 调用 self.checkScript 方法，验证 index 函数的行为，期望返回键 "item" 对应的值 20
        self.checkScript(index, ({"item": 20, "other_item": 120},))

        # 定义函数 type_default，返回一个空字典
        def type_default() -> Dict[str, Tensor]:
            return {}

        # 调用 self.checkScript 方法，验证 type_default 函数的行为，期望返回空字典
        self.checkScript(type_default, ())

        # 使用 torch.jit.script 装饰器定义函数 missing_index，尝试访问不存在的键 "dne"，触发 KeyError
        @torch.jit.script
        def missing_index(x: Dict[str, int]) -> int:
            return x["dne"]

        # 使用 self.assertRaisesRegexWithHighlight 断言捕获 RuntimeError，并检查是否包含 "KeyError" 和 'x["dne"'
        with self.assertRaisesRegexWithHighlight(RuntimeError, "KeyError", 'x["dne"'):
            missing_index({"item": 20, "other_item": 120})

        # 使用 dedent 方法定义多行字符串 code，包含两个函数 literal1 和 literal2 的定义
        code = dedent(
            """
            def literal1():
                return torch.jit.annotate(Dict[int, float], {})
            def literal2():
                return torch.jit.annotate(Dict[int, float], {10: 1.2})
        """
        )
        # 使用 torch.jit.CompilationUnit 构建代码单元 cu，并验证 literal1 函数返回空字典
        cu = torch.jit.CompilationUnit(code)
        self.assertEqual({}, cu.literal1())
        # 验证 literal2 函数返回 {10: 1.2} 的字典
        self.assertEqual({10: 1.2}, cu.literal2())

        # 使用 dedent 方法定义多行字符串 code，包含函数 literal3 的定义
        cu = torch.jit.CompilationUnit(
            dedent(
                """
            def literal3():
                return torch.jit.annotate(Dict[int, float], {10: 1.2, 11: 1.3})
        """
            )
        )
        # 验证 literal3 函数返回 {10: 1.2, 11: 1.3} 的字典
        self.assertEqual({10: 1.2, 11: 1.3}, cu.literal3())

        # 定义函数 list_of_dicts，返回一个列表，包含两个字典，每个字典包含一个张量值
        def list_of_dicts() -> List[Dict[str, Tensor]]:
            return [{"word": torch.ones(2) + 3}, {"other word": torch.ones(1) + 2}]

        # 调用 self.checkScript 方法，验证 list_of_dicts 函数的行为，期望返回包含两个字典的列表
        self.checkScript(list_of_dicts, ())

    # 跳过由于 TorchDynamo 未知原因导致的测试失败
    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    # 定义测试方法，测试字典的可变性
    def test_mutability(self):
        # 使用 torch.jit.script 装饰器定义函数 fn，返回一个字典，包含键 "ok" 和值 10
        @torch.jit.script
        def fn() -> Dict[str, int]:
            a = torch.jit.annotate(Dict[str, int], {})
            a["ok"] = 10
            return a

        # 调用 self.assertEqual 方法，验证 fn 函数返回的字典是否包含 {"ok": 10}
        self.assertEqual(fn(), {"ok": 10})
    # 装饰器：如果 TorchDynamo 对此测试失败，则跳过执行
    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    # 定义测试函数：验证键的类型
    def test_key_type(self):
        # 断言语句开始：期望捕获到 RuntimeError 异常，其中包含 "but instead found type"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "but instead found type", "a[None]"
        ):
            # 使用 Torch 的脚本模式装饰器定义函数 fn，参数为一个字典，键类型为 str，值类型为 int，返回值为 int
            @torch.jit.script
            def fn(a: Dict[str, int]) -> int:
                # 返回字典中 None 键对应的值
                return a[None]

    # 装饰器：如果 TorchDynamo 对此测试失败，则跳过执行
    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    # 定义测试函数：验证循环
    def test_loop(self):
        # 使用 Torch 的脚本模式装饰器定义函数 fn，参数为一个整数 x，返回一个键为 str 类型，值为 int 类型的字典
        @torch.jit.script
        def fn(x: int) -> Dict[str, int]:
            # 声明一个空的字典 a，键为 str 类型，值为 int 类型
            a = torch.jit.annotate(Dict[str, int], {})
            # 循环 x 次，向字典中添加键为 "ok"，值为 i
            for i in range(x):
                a["ok"] = i
            return a

        # 断言：调用 fn(10) 应该返回 {"ok": 9}
        self.assertEqual(fn(10), {"ok": 9})

    # 装饰器：如果 TorchDynamo 对此测试失败，则跳过执行
    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    # 定义测试函数：验证视图操作
    def test_view(self):
        # 定义函数 fn，参数为 x 和 y
        def fn(x, y):
            # 创建一个包含键 "a"，值为 x 的字典 l
            l = {"a": x}
            # 获取 l 中键为 "a" 的值，并赋给 x_view
            x_view = l["a"]
            # 计算 a 的值为 x + x
            a = x + x
            # 在 x_view 上执行 inplace 加法操作 y
            x_view.add_(y)
            # 计算 b 的值为 x + x
            b = x + x
            # 返回 a 是否等于 b
            return a == b

        # 使用给定的随机数生成的数据调用 self.checkScript(fn, (torch.rand(2, 3), torch.rand(2, 3)))
        self.checkScript(fn, (torch.rand(2, 3), torch.rand(2, 3)))

    # 装饰器：如果 TorchDynamo 对此测试失败，则跳过执行
    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    # 定义测试函数：验证成员检查
    def test_membership(self):
        # 定义函数 fn，参数为一个字典 x 和一个整数 y，返回一个整数
        def fn(x: Dict[int, int], y: int) -> int:
            # 返回字典 x 中键为 y 的值，如果不存在则返回默认值 3
            return x.get(y, 3)

        # 创建字典 d
        d = {1: 2, 3: 4}
        # 使用给定的参数调用 self.checkScript(fn, (d, 3))
        self.checkScript(fn, (d, 3))
        # 使用给定的参数调用 self.checkScript(fn, (d, 2))
        self.checkScript(fn, (d, 2))

        # 定义函数 optional，参数为一个字典 x 和一个整数 y，返回一个布尔值
        def optional(x: Dict[int, int], y: int) -> bool:
            # 获取字典 x 中键为 y 的值
            res = x.get(y)
            # 如果 res 是 None，则返回 True，否则返回 False
            return res is None

        # 使用给定的参数调用 self.checkScript(fn, (d, 3))
        self.checkScript(fn, (d, 3))
        # 使用给定的参数调用 self.checkScript(fn, (d, 2))

        # 断言语句开始：期望捕获到 RuntimeError 异常，其中包含 "is actually of type Optional"
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "is actually of type Optional", "return x.get(y"
        ):
            # 使用 Torch 的脚本模式装饰器定义函数 bad_types，参数为一个字典 x 和一个整数 y，返回一个整数
            @torch.jit.script
            def bad_types(x: Dict[int, int], y: int) -> int:
                # 返回字典 x 中键为 y 的值，但是忽略类型检查 T484
                return x.get(y)  # noqa: T484

    # 装饰器：如果 TorchDynamo 对此测试失败，则跳过执行
    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    # 定义测试函数：验证将字典转换为 Python
    def test_dict_to_python(self):
        # 定义函数 python_lookup，使用 torch.jit.ignore 忽略其 Torch 脚本模式转换，参数为一个字典 my_dict 和一个键列表 keys，返回一个值列表
        @torch.jit.ignore
        def python_lookup(my_dict: Dict[str, int], keys: List[str]) -> List[int]:
            # 返回由 my_dict 中每个键对应的值组成的列表，顺序与 keys 中的顺序一致
            return [my_dict[k] for k in keys]

        # 定义函数 fn，参数为一个字典 my_dict 和一个键列表 keys，返回一个值列表
        def fn(my_dict: Dict[str, int], keys: List[str]) -> List[int]:
            # 调用 python_lookup 函数，传入参数 my_dict 和 keys，返回其结果
            return python_lookup(my_dict, keys)

        # 创建一个字典 a_dict
        a_dict = {"a": torch.ones(1), "b": torch.ones(1) + 1, "c": torch.ones(1) + 2}
        # 使用给定的参数调用 self.checkScript(fn, (a_dict, ("a", "c")))
        self.checkScript(fn, (a_dict, ("a", "c")))
    # 定义一个测试函数 test_ordered_dict
    def test_ordered_dict(self):
        # 定义一个测试功能函数 test_func，用于验证通过 JIT 脚本化后的输出是否与原始函数的输出一致
        def test_func(fn, inputs):
            self.assertEqual(fn(*inputs), torch.jit.script(fn)(*inputs))

        # 定义一个返回有重复键的 OrderedDict 的函数
        def repeated_key():
            return OrderedDict([(1, 2), (2, 3), (1, 4)])

        # 测试 repeated_key 函数
        test_func(repeated_key, ())

        # 定义一个没有参数的函数
        def no_args():
            # 创建一个空的 OrderedDict 对象 a，向其中添加两个键值对
            a = OrderedDict()
            a["one"] = torch.tensor(1)
            a["two"] = torch.tensor(2)

        # 测试 no_args 函数
        test_func(no_args, ())

        # 定义一个测试字典构造函数的函数
        def test_dict_constructor():
            # 创建一个普通的字典对象 a，向其中添加一个键值对
            a = dict()
            a["one"] = torch.tensor(1)
            # 返回一个字典和一个含有重复键的字典
            return a, dict([(1, 2), (2, 3), (1, 4)])  # noqa: C406

        # 测试 test_dict_constructor 函数
        test_func(test_dict_constructor, ())

        # 定义一个测试字典初始化列表的函数
        def test_dict_initializer_list():
            # 创建一个字典 a，并初始化其中两个键值对
            a = {"1": torch.tensor(1), "2": torch.tensor(2)}
            output_order = []
            # 遍历字典 a 中的键，并按顺序将对应值添加到列表 output_order 中
            for key in a:
                output_order.append(a[key])
            # 返回按顺序排列的值的列表
            return output_order

        # 测试 test_dict_initializer_list 函数
        test_func(test_dict_initializer_list, ())

        # 定义一个测试字典错误情况的函数
        def test_dict_error():
            # 创建一个普通的字典对象 a，向其中添加一个键值对，但键为整数
            a = dict()
            a[1] = 2
            return a

        # 测试 test_dict_error 函数，在运行时应该捕获到特定的异常
        with self.assertRaisesRegexWithHighlight(
            Exception, "Arguments for call are not", "a[1] = 2"
        ):
            # 对 test_dict_error 函数进行 JIT 脚本化
            torch.jit.script(test_dict_error)

    # 跳过 TorchDynamo 的测试，因为其对此测试的支持存在未知问题
    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_type_annotation_missing_contained_type(self):
        """
        Test that the use of a Dict type annotation without contained
        key and value types produces an error.
        """

        # 定义一个带有类型注释的函数，用于测试
        def fn_with_comment(input: Dict) -> Any:
            return input

        # 定义一个使用 Python3 风格类型注释的函数，用于测试
        def annotated_fn(input: Dict) -> Any:
            return input

        # 在 CompilationUnit 中定义 fn_with_comment 函数时，应该捕获到特定的运行时异常
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Dict without contained types"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        # 在 CompilationUnit 中定义 annotated_fn 函数时，应该捕获到特定的运行时异常
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Dict without contained types"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        # 对 fn_with_comment 函数进行 JIT 脚本化时，应该捕获到特定的运行时异常
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Dict without contained types"
        ):
            m = torch.jit.script(fn_with_comment)

        # 对 annotated_fn 函数进行 JIT 脚本化时，应该捕获到特定的运行时异常
        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Dict without contained types"
        ):
            m = torch.jit.script(annotated_fn)
    # 定义一个测试方法，用于验证字典保持顺序不变
    def test_dict_preserves_order(self):
        # 定义内部函数 dict_ordering，返回一个包含1000个键值对的字典
        def dict_ordering():
            a: Dict[int, int] = {}
            for i in range(1000):
                a[i] = i + 1
            return a

        # 使用 checkScript 方法检查 dict_ordering 函数的脚本化版本
        self.checkScript(dict_ordering, ())
        # 对 dict_ordering 进行脚本化，并调用获取结果
        di = torch.jit.script(dict_ordering)()
        # 将脚本化字典转换为列表形式
        res = list(di.items())
        # 遍历列表中的每个元素，验证键值对的正确性
        for i in range(1000):
            key, value = res[i]
            # 使用 assertTrue 断言键和值分别等于 i 和 i + 1
            self.assertTrue(key == i and value == i + 1)

    # 如果 TorchDynamo 对于这个测试失败且原因不明，则跳过此测试
    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    # 定义一个测试方法，用于测试可选字典的构造
    def test_optional_dict_construct(self):
        # 定义一个内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 定义一个方法 use，接受一个字典 buffer，返回 "prev_key" 对应的值
            def use(self, buffer: Dict[str, Optional[torch.Tensor]]):
                return buffer["prev_key"]

            # 定义前向传播方法 forward，接受输入 x
            def forward(self, x):
                # 创建一个形状为 (2, 3) 的随机张量 prev_key
                prev_key = torch.rand(2, 3)
                # 创建一个形状为 (2, 3) 的随机张量 next_key
                next_key = torch.rand(2, 3)
                # 定义一个字典 saved_state，包含 "prev_key" 和 "next_key" 两个键值对
                saved_state: Dict[str, Optional[torch.Tensor]] = {
                    "prev_key": prev_key,
                    "next_key": next_key,
                }

                # 调用 self.use 方法，传入 saved_state 字典并返回结果
                return self.use(saved_state)

        # 使用 checkModule 方法验证 M 类的实例在输入为 torch.rand(2, 2) 时的行为
        self.checkModule(M(), (torch.rand(2, 2),))
class TestNamedTuple(JitTestCase):
    def test_namedtuple(self):
        # 定义名为 FeatureVector 的命名元组类，包含 float_features (浮点特征)、sequence_features (序列特征列表) 和 time_since_first (自第一次以来的时间)
        class FeatureVector(NamedTuple):
            float_features: float
            sequence_features: List[float]
            time_since_first: float

        @torch.jit.script
        def foo(x) -> float:
            # 创建 FeatureVector 类的实例 fv，初始化 float_features 为 3.0，sequence_features 为 [3.0]，time_since_first 为 3.0
            fv = FeatureVector(3.0, [3.0], 3.0)
            # 从 fv 中获取 float_features 的值并赋给 rv
            rv = fv.float_features
            # 遍历 fv 中的 sequence_features，将每个值加到 rv 上
            for val in fv.sequence_features:
                rv += val
            # 将 rv 乘以 fv 的 time_since_first
            rv *= fv.time_since_first
            # 返回 rv
            return rv

        # 断言调用 foo 函数并传入参数 torch.rand(3, 4) 后的返回值为 18.0
        self.assertEqual(foo(torch.rand(3, 4)), 18.0)

    def test_namedtuple_constant(self):
        # 定义名为 Tup 的命名元组类，包含属性 a 和 b，类型均为 int
        class Tup(NamedTuple):
            a: int
            b: int

        @torch.jit.script
        def foo():
            # 返回 Tup 的实例，属性值为 (1, 2)
            return Tup(1, 2)

        # 断言调用 foo 函数后的返回值为 Tup(1, 2)
        self.assertEqual(foo(), Tup(1, 2))

    def test_return_named_tuple(self):
        # 定义名为 FeatureVector 的命名元组类，与之前相同
        class FeatureVector(NamedTuple):
            float_features: float
            sequence_features: List[float]
            time_since_first: float

        @torch.jit.script
        def foo(x):
            # 创建 FeatureVector 的实例 fv，初始化 float_features 为 3.0，sequence_features 为 [3.0]，time_since_first 为 3.0
            fv = FeatureVector(3.0, [3.0], 3.0)
            # 返回 fv
            return fv

        # 两次调用 foo 函数，并分别用 out 变量接收结果
        out = foo(torch.rand(3, 4))
        out = foo(torch.rand(3, 4))
        # 断言 out 的 float_features 为 3.0，sequence_features 为 [3.0]，time_since_first 为 3.0
        self.assertEqual(out.float_features, 3.0)
        self.assertEqual(out.sequence_features, [3.0])
        self.assertEqual(out.time_since_first, 3.0)

    def test_namedtuple_as_attr(self):
        # 定义名为 Config 的命名元组类，包含属性 size，类型为 int
        class Config(NamedTuple):
            size: int

        # 定义名为 MyMod 的 nn.Module 类
        class MyMod(nn.Module):
            # 包含 configs 属性，类型为 Dict[int, Config]
            configs: Dict[int, Config]

            def __init__(self, configs):
                super().__init__()
                self.configs = configs

            def forward(self, x):
                # 遍历 self.configs 的值并将每个 config.size 加到 x 上
                for config in self.configs.values():
                    x += config.size
                return x

        # 用 torch.jit.script 对 MyMod 类进行脚本化，传入一个包含 {0: Config(size=16)} 的字典作为参数
        s = torch.jit.script(MyMod({0: Config(size=16)}))

    def test_namedtuple_resolution(self):
        # 定义名为 TheType 的命名元组类，包含属性 t，类型为 int
        class TheType(NamedTuple):
            t: int

        # 定义名为 MyModule 的 types.ModuleType 类
        class MyModule(types.ModuleType):
            def __init__(self):
                super().__init__("MyModule")

            def __getattr__(self, attr):
                # 返回 TheType 类
                return TheType

        # 创建 MyModule 的实例 some_module
        some_module = MyModule()

        # 定义函数 fn，其返回类型为 some_module.Type
        def fn() -> some_module.Type:
            # 返回 some_module.Type 的实例，属性值为 1
            return some_module.Type(1)

        # 使用 self.checkScript 对 fn 函数进行脚本化，并传入空列表作为参数
        self.checkScript(fn, [])

    def test_namedtuple_slice_unpack(self):
        # 定义名为 MyCoolNamedTuple 的命名元组类，包含属性 a (int)、b (float) 和 c (List[int])
        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        @torch.jit.script
        def foo(a: int, b: float, c: List[int]):
            # 创建 MyCoolNamedTuple 的实例 tup，属性值分别为 a、b 和 c
            tup = MyCoolNamedTuple(a, b, c)
            # 将 tup 解包，分别赋值给 my_a、my_b 和 my_c
            my_a, my_b, my_c = tup
            # 返回 tup 的前 1 个元素、my_a 和 my_c
            return tup[:1], my_a, my_c

        # 断言调用 foo 函数后的返回值为 ((3,), 3, [6])
        self.assertEqual(foo(3, 3.5, [6]), ((3,), 3, [6]))
    # 定义一个测试方法，用于测试自定义的命名元组在降级过程中的行为
    def test_namedtuple_lower(self):
        # 定义一个继承自NamedTuple的自定义命名元组类型，包含整型a，浮点型b，和整型列表c
        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        # 使用torch.jit.script装饰器将函数foo转换为Torch脚本
        @torch.jit.script
        def foo(a: int):
            # 创建一个MyCoolNamedTuple类型的命名元组tup，包含给定的整数a，浮点数3.14，和整数列表[9]
            tup = MyCoolNamedTuple(a, 3.14, [9])
            return tup

        # 创建一个FileCheck对象，并在foo的计算图中检查是否存在TupleConstruct的标记
        FileCheck().check("TupleConstruct").run(foo.graph)
        # 将foo的计算图中所有的命名元组降级为普通的元组
        torch._C._jit_pass_lower_all_tuples(foo.graph)
        # 再次使用FileCheck对象，在降级后的foo计算图中检查是否不存在TupleConstruct的标记
        FileCheck().check_not("TupleConstruct").run(foo.graph)

    # 定义一个测试方法，用于验证命名元组的类型注解行为
    def test_namedtuple_type_annotation(self):
        # 声明全局变量MyCoolNamedTuple，用于解决Python中的局部解析问题
        global MyCoolNamedTuple  # see [local resolution in python]

        # 定义一个继承自NamedTuple的自定义命名元组类型，包含整型a，浮点型b，和整型列表c
        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        # 使用torch.jit.script装饰器将函数foo转换为Torch脚本
        @torch.jit.script
        def foo(x: MyCoolNamedTuple) -> MyCoolNamedTuple:
            # 直接返回输入的命名元组x
            return x

        # 创建一个MyCoolNamedTuple类型的实例mnt，包含整数42，浮点数420.0，和整数列表[666]
        mnt = MyCoolNamedTuple(42, 420.0, [666])
        # 断言调用foo函数返回的结果与mnt相等
        self.assertEqual(foo(mnt), mnt)

    # 定义一个测试方法，用于验证错误类型的命名元组构造行为
    def test_namedtuple_wrong_types(self):
        # 定义一个继承自NamedTuple的自定义命名元组类型，包含整型a，浮点型b，和整型列表c
        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        # 使用self.assertRaisesRegex断言捕获运行时错误，并检查错误消息的内容
        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a value of type 'int' for argument 'a'"
            " but instead found type 'str'",
        ):
            # 使用torch.jit.script装饰器将函数foo转换为Torch脚本
            @torch.jit.script
            def foo():
                # 尝试用字符串"foo"、"bar"和"baz"创建一个MyCoolNamedTuple类型的命名元组tup
                tup = MyCoolNamedTuple("foo", "bar", "baz")
                return tup

    # 定义一个测试方法，用于验证命名元组的关键字参数构造行为
    def test_namedtuple_kwarg_construct(self):
        # 定义一个继承自NamedTuple的自定义命名元组类型，包含整型a，浮点型b，和整型列表c
        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        # 使用torch.jit.script装饰器将函数foo转换为Torch脚本
        @torch.jit.script
        def foo():
            # 使用关键字参数构造一个MyCoolNamedTuple类型的命名元组tup，分别指定c=[1, 2, 3]，b=3.5，a=9
            tup = MyCoolNamedTuple(c=[1, 2, 3], b=3.5, a=9)
            return tup

        # 调用foo函数并获取返回值tup
        tup = foo()
        # 使用self.assertEqual断言验证tup的属性a等于9，b等于3.5，c等于[1, 2, 3]
        self.assertEqual(tup.a, 9)
        self.assertEqual(tup.b, 3.5)
        self.assertEqual(tup.c, [1, 2, 3])

    # 定义一个被标记为跳过的测试方法，用于验证命名元组的序列化行为
    @unittest.skipIf(True, "broken while these tests were not in CI")
    def test_namedtuple_serialization(self):
        # 定义一个继承自NamedTuple的自定义命名元组类型，包含整型a，浮点型b，和整型列表c
        class MyCoolNamedTuple(NamedTuple):
            a: int
            b: float
            c: List[int]

        # 定义一个继承自torch.jit.ScriptModule的自定义模块类MyMod
        class MyMod(torch.jit.ScriptModule):
            # 定义一个脚本方法forward
            @torch.jit.script_method
            def forward(self):
                # 直接返回一个MyCoolNamedTuple类型的命名元组，包含整数3，浮点数3.5，和整数列表[3, 4, 5]
                return MyCoolNamedTuple(3, 3.5, [3, 4, 5])

        # 创建MyMod类的实例mm
        mm = MyMod()
        # 将mm保存到名为"foo.zip"的文件中
        mm.save("foo.zip")
        # 清空类注册表中的所有类信息，用于测试目的
        torch.testing._internal.jit_utils.clear_class_registry()
        # 从文件"foo.zip"中加载模型，并返回加载后的模型对象loaded
        loaded = torch.jit.load("foo.zip")

        # 分别调用mm和loaded，获取它们的返回结果out和out_loaded
        out = mm()
        out_loaded = loaded()

        # 遍历命名元组的属性名称["a", "b", "c"]，并使用self.assertEqual断言验证loaded和mm返回结果中对应属性的值相等
        for name in ["a", "b", "c"]:
            self.assertEqual(getattr(out_loaded, name), getattr(out, name))
    # 测试一个包含向前引用的命名元组在内部定义的情况
    def test_namedtuple_inside_forwardref(self):
        # 定义一个命名元组 FeatureVector，包含三个字段：float_features 是 float 类型，sequence_features 是 List[float] 类型，time_since_first 是 float 类型
        class FeatureVector(NamedTuple):
            float_features: "float"
            sequence_features: "List[float]"
            time_since_first: "float"

        # 使用 Torch 的脚本模式装饰器
        @torch.jit.script
        # 定义函数 foo，接受参数 x，并返回 float 类型
        def foo(x) -> float:
            # 创建 FeatureVector 实例 fv，初始化 float_features 为 3.0，sequence_features 包含 [3.0]，time_since_first 为 3.0
            fv = FeatureVector(3.0, [3.0], 3.0)
            # 获取 fv 的 float_features 值赋给 rv
            rv = fv.float_features
            # 遍历 fv 的 sequence_features，将其元素加到 rv 上
            for val in fv.sequence_features:
                rv += val
            # 将 rv 乘以 fv 的 time_since_first
            rv *= fv.time_since_first
            # 返回计算结果 rv
            return rv

        # 断言 foo 函数对于给定的 torch.rand(3, 4) 参数返回 18.0
        self.assertEqual(foo(torch.rand(3, 4)), 18.0)

    # 测试命名元组内部包含向前引用的情况
    def test_namedtuple_input_forwardref(self):
        # 定义一个命名元组 MyNamedTuple，包含三个字段：a 是 int 类型，b 是 float 类型，c 是 torch.Tensor 类型
        class MyNamedTuple(NamedTuple):
            a: "int"
            b: "float"
            c: "torch.Tensor"

        # 将 MyNamedTuple 类型注册为全局类型
        make_global(MyNamedTuple)

        # 创建 MyNamedTuple 的实例 nt，初始化为 (4, 2.5, torch.rand((2, 2)))
        nt = MyNamedTuple(4, 2.5, torch.rand((2, 2)))

        # 定义函数 fn，接受一个参数 obj，类型为 MyNamedTuple，返回一个数学运算的结果
        def fn(obj: MyNamedTuple):
            return ((obj.c + obj.b) ** obj.a).sin()

        # 计算预期的结果
        expected = fn(nt)
        # 使用 Torch 的脚本模式装饰器，将 fn 函数编译为 Torch 脚本
        fn_s = torch.jit.script(fn)
        # 计算编译后的实际结果
        actual = fn_s(nt)
        # 断言预期结果与实际结果相等
        self.assertEqual(expected, actual)

    # 见 issue #95858，测试命名元组解析中的向前引用情况
    @unittest.expectedFailure
    def test_namedtuple_resolution_forwardref(self):
        # 定义一个命名元组 TheType，包含一个字段 t，类型为 int
        class TheType(NamedTuple):
            t: "int"

        # 定义一个自定义模块 MyModule，继承自 types.ModuleType
        class MyModule(types.ModuleType):
            def __init__(self):
                super().__init__("MyModule")

            # 自定义 __getattr__ 方法，返回 TheType 类型
            def __getattr__(self, attr):
                return TheType

        # 创建 MyModule 的实例 some_module
        some_module = MyModule()

        # 定义函数 fn，返回类型为 some_module.Type
        def fn() -> some_module.Type:
            return some_module.Type(1)

        # 检查 fn 函数在空参数列表下的 Torch 脚本化结果
        self.checkScript(fn, [])
    def test_repr(self):
        """
        Test the __repr__ method.
        """
        # 调用 _compare_eager_and_script 方法，比较传入 lambda 表达式对字典的 __repr__ 方法的调用结果
        self._compare_eager_and_script(lambda d: repr(d), {1: 2})
    def test_bool(self):
        """
        Test the __bool__ method. This should return True
        if the dictionary is non-empty and False otherwise.
        """
        # 调用 _compare_eager_and_script 方法测试 lambda 函数 bool(d) 在非空字典上返回 True
        self._compare_eager_and_script(lambda d: bool(d), {1: 2})
        # 调用 _compare_eager_and_script 方法测试 lambda 函数 bool(d) 在空字典上返回 False
        self._compare_eager_and_script(lambda d: bool(d), {})

    def test_iter(self):
        """
        Test iteration over a dictionary's keys.
        """

        def sum_keys(input_dict):
            s = 0
            # 遍历字典的键，累加键的值到 s
            for k in input_dict:
                s += k

            return s

        # 调用 _compare_eager_and_script 方法测试 sum_keys 函数
        self._compare_eager_and_script(sum_keys, {1: 2, 3: 4})

    def test_items(self):
        """
        Test .items().
        """

        def sum_pair_product(input_dict):
            s = 0
            # 遍历字典的键值对，累加键乘以值到 s
            for k, v in input_dict.items():
                s += k * v

            return s

        # 调用 _compare_eager_and_script 方法测试 sum_pair_product 函数
        self._compare_eager_and_script(sum_pair_product, {1: 2, 3: 4})

    def test_getitem(self):
        """
        Test accessing dictionary values using the [] operator.
        """
        data = {1: 2, 3: 4}
        # 测试 lambda 函数通过索引访问字典值 d[1]
        self._compare_eager_and_script(lambda d: d[1], data)
        # 测试 lambda 函数通过索引访问字典中不存在的键 d[4]
        self._compare_eager_and_script(lambda d: d[4], data)
        # 测试 lambda 函数通过索引访问字典中不存在的键 d[2]
        self._compare_eager_and_script(lambda d: d[2], data)
        # 测试 lambda 函数通过字符串键访问字典值 d["key"]
        self._compare_eager_and_script(lambda d: d["key"], data)

    def test_setitem(self):
        """
        Test setting dictionary values using the [] operator.
        """
        data = {1: 2, 3: 4}

        def fn(input_dict):
            # 修改字典中键 1 的值为 10
            input_dict[1] = 10
            # 修改字典中键 3 的值为 11
            input_dict[3] = 11

        # 调用 _compare_eager_and_script 方法测试 fn 函数
        self._compare_eager_and_script(fn, data)

        # 检查使用错误类型的键和值会抛出 TypeError
        # 由于 _compare_eager_and_script 不能用于此处，因此使用 torch.jit.script 将 data 转换为脚本化的数据
        script_data = torch.jit.script(data)

        with self.assertRaises(TypeError):
            # 尝试使用字符串作为键，预期抛出 TypeError
            script_data["str"] = 3

        with self.assertRaises(TypeError):
            # 尝试使用字符串作为值，预期抛出 TypeError
            script_data[3] = "str"

    def test_contains(self):
        """
        Test membership checks (x in y, x not in y).
        """
        data = {1: 2, 3: 4}

        def fn(input_dict):
            # 返回四个成员检查的结果：1 在字典中、2 不在字典中、3 在字典中、4 不在字典中
            return (
                1 in input_dict,
                2 not in input_dict,
                3 in input_dict,
                4 not in input_dict,
            )

        # 调用 _compare_eager_and_script 方法测试 fn 函数
        self._compare_eager_and_script(fn, data)

        # 检查使用错误类型的键会抛出 KeyError
        script_data = torch.jit.script(data)

        with self.assertRaises(KeyError):
            # 尝试使用字符串作为键，预期抛出 KeyError
            a = "str" in script_data
    # 定义一个单元测试方法，用于测试删除操作。
    def test_delitem(self):
        """
        Test deletion.
        """
        # 创建一个普通字典作为测试数据
        data = {1: 2, 3: 4}

        # 定义一个删除指定键的函数
        def del_fn(input_dict):
            del input_dict[1]

        # 定义一个尝试删除不存在键的函数
        def del_fn_raises(input_dict):
            del input_dict[10]

        # 测试删除函数对普通字典的影响
        self._compare_eager_and_script(del_fn, data)
        self._compare_eager_and_script(del_fn_raises, data)

        # 使用 TorchScript 将普通字典转换为脚本化数据
        script_data = torch.jit.script(data)

        # 检查尝试使用错误类型的键进行删除是否会抛出 TypeError 异常
        with self.assertRaises(TypeError):
            del script_data["str"]

    # 定义一个单元测试方法，用于测试内置函数 len()
    def test_len(self):
        """
        Test len() builtin function.
        """
        # 测试空字典和非空字典的长度计算是否正确
        self._compare_eager_and_script(lambda d: len(d), {1: 2})
        self._compare_eager_and_script(lambda d: len(d), {})

    # 标记为跳过的单元测试，因为所有从 TorchScript 返回的字典都必须是 ScriptDicts
    @unittest.skip(
        "Cannot pass until all dicts returned from TorchScript are ScriptDicts"
    )
    # 定义一个单元测试方法，用于测试嵌套的 ScriptDict 在 TorchScript 中的引用语义
    def test_nested(self):
        """
        Test that reference semantics are honoured when the ScriptDict that is
        mutated using TorchScript is inside another.
        """
        # 使用 TorchScript 将嵌套字典转换为脚本化数据
        nested = torch.jit.script(
            {1: {1: 2}, 2: {3: 4}}, type_hint=Dict[int, Dict[int, int]]
        )

        # 获取嵌套字典中的子字典
        one = nested[1]
        two = nested[2]

        # 在子字典中添加新的键值对
        self._script_dict_add(one, 9, 10)
        self._script_dict_add(two, 11, 12)

        # 检查修改后的子字典对原始嵌套字典的影响
        self.assertEqual(len(one), 2)
        self.assertEqual(len(two), 2)
        self.assertEqual(len(nested[1]), 2)
        self.assertEqual(len(nested[2]), 2)

    # 定义一个单元测试方法，用于测试在 TorchScript 中修改 ScriptDict 的引用语义
    def test_reference_semantics(self):
        """
        Test that reference semantics are honoured; that modifications made
        to a ScriptDict in TorchScript are visible in Python.
        """
        # 使用 TorchScript 将普通字典转换为脚本化数据
        data = torch.jit.script({1: 2})
        # 在脚本化数据中添加新的键值对
        self._script_dict_add(data, 3, 4)

        # 检查修改后的脚本化数据对原始字典的影响
        self.assertEqual(len(data), 2)
        self.assertTrue(3 in data)
        self.assertEqual(data[3], 4)
    def test_repr(self):
        """
        Test the __repr__ method.
        """
        # 使用 _compare_eager_and_script 方法比较 __repr__ 方法在普通列表和脚本化列表上的行为
        self._compare_eager_and_script(lambda l: repr(l), [1])
    def test_bool(self):
        """
        Test the __bool__ method. This should return True
        if the list is non-empty and False otherwise.
        """
        # 测试 __bool__ 方法，应当在列表非空时返回 True，否则返回 False
        self._compare_eager_and_script(lambda l: bool(l), [1])
        self._compare_eager_and_script(lambda l: bool(l), [])

    def test_iter(self):
        """
        Test iteration over a list's elements.
        """

        def sum_elements(input_list):
            # 计算列表中所有元素的和
            s = 0
            for k in input_list:
                s += k

            return s

        # 测试对列表元素进行迭代
        self._compare_eager_and_script(sum_elements, [1, 2, 3, 4])

    def test_getitem(self):
        """
        Test accessing list elements using the [] operator.
        """
        data = [1, 2, 3, 4]

        # 测试常规索引
        self._compare_eager_and_script(lambda l: l[1], data)
        self._compare_eager_and_script(lambda l: l[3], data)
        self._compare_eager_and_script(lambda l: l[-1], data)

        # 测试切片操作
        self._compare_eager_and_script(lambda l: l[1:3], data)
        self._compare_eager_and_script(lambda l: l[:], data)
        self._compare_eager_and_script(lambda l: l[1:], data)
        self._compare_eager_and_script(lambda l: l[:2], data)
        self._compare_eager_and_script(lambda l: l[-1], data)
        self._compare_eager_and_script(lambda l: l[-1::-1], data)

        # 测试错误情况
        self._compare_eager_and_script(lambda l: l[5], data)
        self._compare_eager_and_script(lambda l: l[-7], data)
        self._compare_eager_and_script(lambda l: l["key"], data)
    def test_setitem(self):
        """
        Test setting list elements using the [] operator.
        """
        data = [1, 2, 3, 4]

        # Test regular assignment.
        # 定义函数，测试普通赋值操作
        def setitem(input_list):
            input_list[1] = 10  # 将索引为1的元素设置为10
            input_list[3] = 11  # 将索引为3的元素设置为11
            input_list[-1] = 12  # 将倒数第一个元素设置为12

        self._compare_eager_and_script(setitem, data.copy())

        # Test slice assignment.
        # 测试切片赋值操作
        # TODO: Something like input_list[:1] = [1, 2, 3, 4, 5]
        # is allowed in Python, but pybind11/stl_bind.h does not
        # allow it. Should we?
        def setitem_slice(input_list):
            input_list[:4:2] = [10, 11]  # 将索引0到3之间，步长为2的元素设置为[10, 11]
            input_list[-2:] = [15, 16]  # 将倒数第二个和倒数第一个元素设置为[15, 16]

        self._compare_eager_and_script(setitem_slice, data)

        # Test errors.
        # 测试错误情况
        def out_of_range(input_list):
            input_list[11] = 3  # 尝试设置索引为11的元素，超出列表长度

        def out_of_range_negative(input_list):
            input_list[-11] = 3  # 尝试设置索引为-11的元素，超出列表长度

        def wrong_index_type(input_list):
            input_list["str"] = 3  # 尝试使用字符串作为索引，应该抛出TypeError

        self._compare_eager_and_script(out_of_range, data)
        self._compare_eager_and_script(out_of_range_negative, data)
        self._compare_eager_and_script(wrong_index_type, data)

        # Check that using value of an incorrect type throws TypeError.
        # 检查使用错误类型的值会抛出TypeError异常
        # _compare_eager_and_script在此处不能使用，因为
        # 下面的__setitem__使用在Python中是有效的。
        script_data = torch.jit.script(data)

        with self.assertRaises(TypeError):
            script_data[0] = "str"

    def test_contains(self):
        """
        Test membership checks (x in y, x not in y).
        """
        data = [1, 2, 3, 4]

        def fn(input_list):
            return (
                1 in input_list,   # 检查1是否在列表中
                2 not in input_list,  # 检查2是否不在列表中
                3 in input_list,   # 检查3是否在列表中
                4 not in input_list,  # 检查4是否不在列表中
            )

        self._compare_eager_and_script(fn, data)

        # Check that using a value of an incorrect type throws a TypeError.
        # 检查使用错误类型的值会抛出TypeError异常
        script_data = torch.jit.script(data)

        with self.assertRaises(TypeError):
            a = "str" in script_data

    def test_delitem(self):
        """
        Test deletion.
        """
        data = [1, 2, 3, 4]

        def del_fn(input_list):
            del input_list[1]  # 删除索引为1的元素

        def del_fn_out_of_range(input_list):
            del input_list[10]  # 尝试删除超出列表长度的索引

        def del_fn_wrong_type(input_list):
            del input_list["str"]  # 尝试使用字符串作为索引，应该抛出TypeError

        self._compare_eager_and_script(del_fn, data.copy())
        self._compare_eager_and_script(del_fn_out_of_range, data)
        self._compare_eager_and_script(del_fn_wrong_type, data)

    def test_len(self):
        """
        Test len() builtin function.
        """
        self._compare_eager_and_script(lambda l: len(l), [1, 2, 3, 4])  # 测试len()函数对有元素的列表
        self._compare_eager_and_script(lambda l: len(l), [])  # 测试len()函数对空列表
    def test_count(self):
        """
        Test count method.
        """
        # 使用 _compare_eager_and_script 方法比较 lambda 表达式计算的 count(3) 结果，预期结果是 [1, 2, 3, 3] 中 3 的个数
        self._compare_eager_and_script(lambda l: l.count(3), [1, 2, 3, 3])

        # 检查使用错误类型的值会抛出 TypeError 异常
        script_data = torch.jit.script([1])

        with self.assertRaises(TypeError):
            # 尝试在列表中查找字符串 "str"
            script_data.count("str")

    def test_remove(self):
        """
        Test remove method.
        """
        # 使用 _compare_eager_and_script 方法比较 lambda 表达式调用 remove(1) 的效果，预期结果是移除列表中的第一个 1
        self._compare_eager_and_script(lambda l: l.remove(1), [1, 2, 3])
        # 测试 remove 方法尝试移除列表中不存在的元素 10，预期不会发生任何变化
        self._compare_eager_and_script(lambda l: l.remove(10), [1, 2, 3])

        # 检查使用错误类型的值会抛出 TypeError 异常
        script_data = torch.jit.script([1])

        with self.assertRaises(TypeError):
            # 尝试在列表中移除字符串 "str"
            script_data.remove("str")

    def test_append(self):
        """
        Test append method.
        """
        # 使用 _compare_eager_and_script 方法比较 lambda 表达式调用 append(1) 的效果，预期结果是在列表末尾添加元素 1
        self._compare_eager_and_script(lambda l: l.append(1), [4, 3, 2])

        # 检查使用错误类型的值会抛出 TypeError 异常
        script_data = torch.jit.script([1])

        with self.assertRaises(TypeError):
            # 尝试向列表中添加字符串 "str"
            script_data.append("str")

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1991")
    def test_clear(self):
        """
        Test clear.
        """
        # 使用 _compare_eager_and_script 方法比较 lambda 表达式调用 clear() 的效果，预期结果是清空列表
        self._compare_eager_and_script(lambda l: l.clear(), [4, 3, 2])

    def test_extend(self):
        """
        Test extend.
        """

        class Iterable:
            def __init__(self, limit: int):
                self.limit = limit
                self.value = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.value == limit:  # noqa: F821
                    raise StopIteration

                ret = self.value
                self.value += 1
                return ret

        data = [1, 2, 3]

        def extend_list(input_list):
            # 将 [4, 5, 6] 扩展到输入列表中
            input_list.extend([4, 5, 6])

        def extend_dict(input_list):
            # 尝试将 {4: 10, 5: 11, 6: 12} 扩展到输入列表中，预期会抛出 TypeError 异常
            input_list.extend({4: 10, 5: 11, 6: 12})

        def extend_iterable(input_list):
            # 将 Iterable(3) 对象中的元素扩展到输入列表中
            input_list.extend(Iterable(3))

        # 分别比较上述扩展方法在数据副本上的效果
        self._compare_eager_and_script(extend_list, data.copy())
        self._compare_eager_and_script(extend_dict, data.copy())
        self._compare_eager_and_script(extend_iterable, data)

        # 检查使用错误类型的值会抛出 TypeError 异常
        script_data = torch.jit.script([1])

        with self.assertRaises(TypeError):
            # 尝试使用字符串数组扩展列表
            script_data.extend(["a"])

        with self.assertRaises(TypeError):
            # 尝试使用字典扩展列表
            script_data.extend({"a": 1})
    # 定义一个测试函数 test_insert，用于测试列表的 insert 方法。
    def test_insert(self):
        """
        Test insert.
        """
        # 创建一个列表 data，包含元素 [1, 2, 4]
        data = [1, 2, 4]

        # 调用 self._compare_eager_and_script 方法比较在执行 lambda 函数插入元素 3 后的效果
        self._compare_eager_and_script(lambda l: l.insert(3, 3), data.copy())
        # 同上，但在列表头插入元素 3
        self._compare_eager_and_script(lambda l: l.insert(0, 3), data.copy())
        # 同上，但在倒数第二个位置（索引为 -2）插入元素 3
        self._compare_eager_and_script(lambda l: l.insert(-2, 3), data)

        # 检查使用错误类型的值（元组）插入时是否会抛出 TypeError 异常
        script_data = torch.jit.script([1])

        with self.assertRaises(TypeError):
            script_data.insert((0, "str"))

    # 定义一个测试函数 test_pop，用于测试列表的 pop 方法。
    def test_pop(self):
        """
        Test pop.
        """
        # 创建一个列表 data，包含元素 [1, 2, 3, 4, 5]
        data = [1, 2, 3, 4, 5]

        # 测试 pop 方法的正常情况，从列表尾部弹出一个元素
        self._compare_eager_and_script(lambda l: l.pop(), data.copy())
        # 同上，但从索引为 2 的位置弹出元素
        self._compare_eager_and_script(lambda l: l.pop(2), data.copy())
        # 同上，但从倒数第三个位置（索引为 -3）弹出元素
        self._compare_eager_and_script(lambda l: l.pop(-3), data.copy())

        # 测试 pop 方法的错误情况，尝试从超出列表长度的位置（索引 10）弹出元素
        self._compare_eager_and_script(lambda l: l.pop(10), data)

    # 定义一个测试函数 test_nested，测试 TorchScript 中的嵌套列表的引用语义是否正确处理。
    @unittest.skip(
        "Cannot pass until all list returned from TorchScript are ScriptLists"
    )
    def test_nested(self):
        """
        Test that reference semantics are honoured when the ScriptList that is
        mutated using TorchScript is inside another.
        """
        # 使用 TorchScript 创建一个嵌套列表 nested，包含两个子列表 [1] 和 [2]
        nested = torch.jit.script([[1], [2]], List[List[int]])

        # 从 nested 中获取第一个子列表和第二个子列表
        one = nested[0]
        two = nested[1]

        # 在第一个子列表 one 中添加元素 3
        self._script_list_add(one, 3)
        # 在第二个子列表 two 中添加元素 4
        self._script_list_add(two, 4)

        # 检查修改后的元素是否在原始列表 nested 中可见
        self.assertEqual(len(one), 2)
        self.assertEqual(len(two), 2)
        self.assertEqual(one[len(one) - 1], 3)
        self.assertEqual(two[len(one) - 1], 4)
        self.assertEqual(len(nested[0]), 2)
        self.assertEqual(len(nested[1]), 2)

    # 定义一个测试函数 test_reference_semantics，测试 TorchScript 中的列表在 Python 中的引用语义是否正确处理。
    def test_reference_semantics(self):
        """
        Test that reference semantics are honoured; that modifications made
        to a ScriptList in TorchScript are visible in Python.
        """
        # 使用 TorchScript 创建一个列表 l，包含元素 [1, 2]
        l = torch.jit.script([1, 2])
        # 在列表 l 中添加元素 3
        self._script_list_add(l, 3)

        # 检查列表长度是否为 3，元素 3 是否在列表中，以及列表索引为 2 的位置是否为 3
        self.assertEqual(len(l), 3)
        self.assertTrue(3 in l)
        self.assertEqual(l[2], 3)
    def test_defaultdict(self):
        # 定义内部函数，返回一个使用defaultdict创建的空列表的字典
        def get_dict():
            test_dict = defaultdict(list)
            return test_dict

        # 定义一个继承自torch.nn.Module的测试类
        class Test(torch.nn.Module):
            # 类变量：segments_groupby_col 是一个从字符串映射到字符串列表的字典
            segments_groupby_col: Dict[str, List[str]]

            # 类的初始化函数
            def __init__(self):
                super().__init__()
                # 初始化segments_groupby_col为一个使用defaultdict创建的空列表的字典
                self.segments_groupby_col = get_dict()
                # 初始化类的其他成员变量
                self.col1 = "a"
                self.col2 = "b"

            # 类的前向传播函数
            def forward(self):
                # 检查self.col1是否在segments_groupby_col的键集合中
                if self.col1 in self.segments_groupby_col.keys():
                    return 1
                else:
                    return 2

        # 创建Test类的一个实例对象
        test = Test()
        # 对test对象进行脚本化（转换为Torch脚本）
        test_script = torch.jit.script(test)
        # 访问test_script对象的segments_groupby_col属性

        # 对代码进行冒烟测试，用来检测稳定性。大约需要2秒钟。
        for i in range(300):
            # 创建Test类的新实例对象
            test = Test()
            # 对新的test对象进行脚本化（转换为Torch脚本）
            test_script = torch.jit.script(test)
```