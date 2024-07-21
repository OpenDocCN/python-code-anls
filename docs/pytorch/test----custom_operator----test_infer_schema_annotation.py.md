# `.\pytorch\test\custom_operator\test_infer_schema_annotation.py`

```
# Owner(s): ["module: pt2-dispatcher"]
# 导入未来的注解支持
from __future__ import annotations

# 导入类型相关的模块和类型注解
import typing
from typing import List, Optional, Sequence, Union  # noqa: F401

# 导入 PyTorch 相关模块
import torch
import torch._custom_op.impl
from torch import Tensor, types
from torch.testing._internal.common_utils import run_tests, TestCase

# 创建一个空字典用于存储可能修改的参数信息
mutates_args = {}

# 定义一个测试类 TestInferSchemaWithAnnotation 继承自 TestCase 类
class TestInferSchemaWithAnnotation(TestCase):

    # 定义测试方法 test_tensor
    def test_tensor(self):
        # 定义一个函数 foo_op，接收一个 torch.Tensor 类型参数并返回一个 torch.Tensor 类型结果
        def foo_op(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        # 调用 torch._custom_op.impl.infer_schema 函数推断 foo_op 的模式，并断言结果符合预期
        result = torch._custom_op.impl.infer_schema(foo_op, mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        # 定义另一个函数 foo_op_2，接收两个 torch.Tensor 类型参数并返回一个 torch.Tensor 类型结果
        def foo_op_2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x.clone() + y

        # 调用 torch._custom_op.impl.infer_schema 函数推断 foo_op_2 的模式，并断言结果符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_2, mutates_args)
        self.assertEqual(result, "(Tensor x, Tensor y) -> Tensor")

    # 定义测试方法 test_native_types
    def test_native_types(self):
        # 定义一个函数 foo_op，接收一个 int 类型参数并返回一个 int 类型结果
        def foo_op(x: int) -> int:
            return x

        # 调用 torch._custom_op.impl.infer_schema 函数推断 foo_op 的模式，并断言结果符合预期
        result = torch._custom_op.impl.infer_schema(foo_op, mutates_args)
        self.assertEqual(result, "(SymInt x) -> SymInt")

        # 定义一个函数 foo_op_2，接收一个 bool 类型参数并返回一个 bool 类型结果
        def foo_op_2(x: bool) -> bool:
            return x

        # 调用 torch._custom_op.impl.infer_schema 函数推断 foo_op_2 的模式，并断言结果符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_2, mutates_args)
        self.assertEqual(result, "(bool x) -> bool")

        # 定义一个函数 foo_op_3，接收一个 str 类型参数并返回一个 int 类型结果
        def foo_op_3(x: str) -> int:
            return 1

        # 调用 torch._custom_op.impl.infer_schema 函数推断 foo_op_3 的模式，并断言结果符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_3, mutates_args)
        self.assertEqual(result, "(str x) -> SymInt")

        # 定义一个函数 foo_op_4，接收一个 float 类型参数并返回一个 float 类型结果
        def foo_op_4(x: float) -> float:
            return x

        # 调用 torch._custom_op.impl.infer_schema 函数推断 foo_op_4 的模式，并断言结果符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_4, mutates_args)
        self.assertEqual(result, "(float x) -> float")

    # 定义测试方法 test_torch_types
    def test_torch_types(self):
        # 定义一个函数 foo_op_1，接收一个 torch.types.Number 类型参数并返回一个 torch.types.Number 类型结果
        def foo_op_1(x: torch.types.Number) -> torch.types.Number:
            return x

        # 调用 torch._custom_op.impl.infer_schema 函数推断 foo_op_1 的模式，并断言结果符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_1, mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

        # 定义一个函数 foo_op_2，接收一个 torch.dtype 类型参数并返回一个 int 类型结果
        def foo_op_2(x: torch.dtype) -> int:
            return 1

        # 调用 torch._custom_op.impl.infer_schema 函数推断 foo_op_2 的模式，并断言结果符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_2, mutates_args)
        self.assertEqual(result, "(ScalarType x) -> SymInt")

        # 定义一个函数 foo_op_3，接收一个 torch.device 类型参数并返回一个 int 类型结果
        def foo_op_3(x: torch.device) -> int:
            return 1

        # 调用 torch._custom_op.impl.infer_schema 函数推断 foo_op_3 的模式，并断言结果符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_3, mutates_args)
        self.assertEqual(result, "(Device x) -> SymInt")
    # 定义测试方法 test_type_variants，用于测试不同类型的函数参数和返回类型
    def test_type_variants(self):
        
        # 定义带有 Optional[int] 类型注解和 int 返回类型的函数 foo_op_1
        def foo_op_1(x: typing.Optional[int]) -> int:
            return 1
        
        # 调用推断函数 infer_schema 分析 foo_op_1 的类型，并断言其结果
        result = torch._custom_op.impl.infer_schema(foo_op_1, mutates_args)
        self.assertEqual(result, "(SymInt? x) -> SymInt")
        
        # 定义带有 Sequence[int] 类型注解和 int 返回类型的函数 foo_op_2
        def foo_op_2(x: typing.Sequence[int]) -> int:
            return 1
        
        # 调用推断函数 infer_schema 分析 foo_op_2 的类型，并断言其结果
        result = torch._custom_op.impl.infer_schema(foo_op_2, mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")
        
        # 定义带有 List[int] 类型注解和 int 返回类型的函数 foo_op_3
        def foo_op_3(x: typing.List[int]) -> int:
            return 1
        
        # 调用推断函数 infer_schema 分析 foo_op_3 的类型，并断言其结果
        result = torch._custom_op.impl.infer_schema(foo_op_3, mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")
        
        # 定义带有 Optional[Sequence[int]] 类型注解和 int 返回类型的函数 foo_op_4
        def foo_op_4(x: typing.Optional[typing.Sequence[int]]) -> int:
            return 1
        
        # 调用推断函数 infer_schema 分析 foo_op_4 的类型，并断言其结果
        result = torch._custom_op.impl.infer_schema(foo_op_4, mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")
        
        # 定义带有 Optional[List[int]] 类型注解和 int 返回类型的函数 foo_op_5
        def foo_op_5(x: typing.Optional[typing.List[int]]) -> int:
            return 1
        
        # 调用推断函数 infer_schema 分析 foo_op_5 的类型，并断言其结果
        result = torch._custom_op.impl.infer_schema(foo_op_5, mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")
        
        # 定义带有 Union[int, float, bool] 类型注解和 Number 返回类型的函数 foo_op_6
        def foo_op_6(x: typing.Union[int, float, bool]) -> types.Number:
            return x
        
        # 调用推断函数 infer_schema 分析 foo_op_6 的类型，并断言其结果
        result = torch._custom_op.impl.infer_schema(foo_op_6, mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")
        
        # 定义带有 Union[int, bool, float] 类型注解和 Number 返回类型的函数 foo_op_7
        def foo_op_7(x: typing.Union[int, bool, float]) -> types.Number:
            return x
        
        # 调用推断函数 infer_schema 分析 foo_op_7 的类型，并断言其结果
        result = torch._custom_op.impl.infer_schema(foo_op_7, mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")
    # 定义一个测试函数，测试在没有使用库前缀的情况下的自定义操作
    def test_no_library_prefix(self):
        # 定义一个函数 foo_op，接受一个 Tensor 类型的参数 x，并返回一个 Tensor 类型的结果
        def foo_op(x: Tensor) -> Tensor:
            return x.clone()

        # 调用 torch._custom_op.impl.infer_schema 函数，推断 foo_op 的签名并检查是否符合预期
        result = torch._custom_op.impl.infer_schema(foo_op, mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        # 定义另一个函数 foo_op_2，接受一个 Tensor 类型的参数 x，并返回一个 Tensor 类型的结果
        def foo_op_2(x: Tensor) -> torch.Tensor:
            return x.clone()

        # 调用 torch._custom_op.impl.infer_schema 函数，推断 foo_op_2 的签名并检查是否符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_2, mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        # 定义函数 foo_op_3，接受一个 torch.Tensor 类型的参数 x，并返回一个 Tensor 类型的结果
        def foo_op_3(x: torch.Tensor) -> Tensor:
            return x.clone()

        # 调用 torch._custom_op.impl.infer_schema 函数，推断 foo_op_3 的签名并检查是否符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_3, mutates_args)
        self.assertEqual(result, "(Tensor x) -> Tensor")

        # 定义函数 foo_op_4，接受一个 List[int] 类型的参数 x，并返回一个 types.Number 类型的结果
        def foo_op_4(x: List[int]) -> types.Number:
            return x[0]

        # 调用 torch._custom_op.impl.infer_schema 函数，推断 foo_op_4 的签名并检查是否符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_4, mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> Scalar")

        # 定义函数 foo_op_5，接受一个 Optional[int] 类型的参数 x，并返回一个 int 类型的结果
        def foo_op_5(x: Optional[int]) -> int:
            return 1

        # 调用 torch._custom_op.impl.infer_schema 函数，推断 foo_op_5 的签名并检查是否符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_5, mutates_args)
        self.assertEqual(result, "(SymInt? x) -> SymInt")

        # 定义函数 foo_op_6，接受一个 Sequence[int] 类型的参数 x，并返回一个 int 类型的结果
        def foo_op_6(x: Sequence[int]) -> int:
            return 1

        # 调用 torch._custom_op.impl.infer_schema 函数，推断 foo_op_6 的签名并检查是否符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_6, mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")

        # 定义函数 foo_op_7，接受一个 List[int] 类型的参数 x，并返回一个 int 类型的结果
        def foo_op_7(x: List[int]) -> int:
            return 1

        # 调用 torch._custom_op.impl.infer_schema 函数，推断 foo_op_7 的签名并检查是否符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_7, mutates_args)
        self.assertEqual(result, "(SymInt[] x) -> SymInt")

        # 定义函数 foo_op_8，接受一个 Optional[Sequence[int]] 类型的参数 x，并返回一个 int 类型的结果
        def foo_op_8(x: Optional[Sequence[int]]) -> int:
            return 1

        # 调用 torch._custom_op.impl.infer_schema 函数，推断 foo_op_8 的签名并检查是否符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_8, mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")

        # 定义函数 foo_op_9，接受一个 Optional[List[int]] 类型的参数 x，并返回一个 int 类型的结果
        def foo_op_9(x: Optional[List[int]]) -> int:
            return 1

        # 调用 torch._custom_op.impl.infer_schema 函数，推断 foo_op_9 的签名并检查是否符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_9, mutates_args)
        self.assertEqual(result, "(SymInt[]? x) -> SymInt")

        # 定义函数 foo_op_10，接受一个 Union[int, float, bool] 类型的参数 x，并返回一个 types.Number 类型的结果
        def foo_op_10(x: Union[int, float, bool]) -> types.Number:
            return x

        # 调用 torch._custom_op.impl.infer_schema 函数，推断 foo_op_10 的签名并检查是否符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_10, mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")

        # 定义函数 foo_op_11，接受一个 Union[int, bool, float] 类型的参数 x，并返回一个 types.Number 类型的结果
        def foo_op_11(x: Union[int, bool, float]) -> types.Number:
            return x

        # 调用 torch._custom_op.impl.infer_schema 函数，推断 foo_op_11 的签名并检查是否符合预期
        result = torch._custom_op.impl.infer_schema(foo_op_11, mutates_args)
        self.assertEqual(result, "(Scalar x) -> Scalar")
    # 测试不支持的注解是否引发异常
    def test_unsupported_annotation(self):
        # 使用断言上下文管理器检查是否抛出 ValueError 异常，并验证异常消息是否包含特定字符串
        with self.assertRaisesRegex(
            ValueError,
            r"Unsupported type annotation D. It is not a type.",
        ):
            # 定义一个函数 foo_op，其参数 x 带有注解 D（此处忽略 F821 错误）
            def foo_op(x: D) -> Tensor:
                # 返回一个 Tensor 对象，将输入 x 转换为 Tensor
                return torch.Tensor(x)

            # 调用 Torch 自定义操作的推断模式方法，传递 foo_op 函数和一个标志 mutates_args
            torch._custom_op.impl.infer_schema(foo_op, mutates_args)

        # 使用断言上下文管理器检查是否抛出 ValueError 异常，并验证异常消息是否包含特定字符串
        with self.assertRaisesRegex(
            ValueError,
            r"Unsupported type annotation E. It is not a type.",
        ):
            # 定义另一个函数 foo_op_2，其参数 x 带有注解 Tensor，返回类型注解为 E（此处忽略 F821 错误）
            def foo_op_2(x: Tensor) -> E:
                # 返回输入 x，不进行任何转换
                return x

            # 调用 Torch 自定义操作的推断模式方法，传递 foo_op_2 函数和一个标志 mutates_args
            torch._custom_op.impl.infer_schema(foo_op_2, mutates_args)
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 调用函数 run_tests() 来执行测试
    run_tests()
```