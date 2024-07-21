# `.\pytorch\tools\test\test_executorch_types.py`

```py
# 导入单元测试模块
import unittest

# 从torchgen模块导入local子模块
from torchgen import local

# 从torchgen.api.types模块导入多个类型定义
from torchgen.api.types import (
    BaseCType,
    ConstRefCType,
    CType,
    longT,
    MutRefCType,
    NamedCType,
    OptionalCType,
    TupleCType,
    VectorCType,
    voidT,
)

# 从torchgen.executorch.api.et_cpp模块导入三个函数
from torchgen.executorch.api.et_cpp import argument_type, return_type, returns_type

# 从torchgen.executorch.api.types模块导入ArrayRefCType, scalarT, tensorListT, tensorT类型
from torchgen.executorch.api.types import ArrayRefCType, scalarT, tensorListT, tensorT

# 从torchgen.model模块导入Argument, FunctionSchema, Return类
from torchgen.model import Argument, FunctionSchema, Return


# 定义单元测试类ExecutorchCppTest，继承自unittest.TestCase
class ExecutorchCppTest(unittest.TestCase):
    """
    Test torchgen.executorch.api.cpp
    """

    # 定义测试函数_test_argumenttype_type，接收参数arg_str和expected
    def _test_argumenttype_type(self, arg_str: str, expected: NamedCType) -> None:
        # 解析参数字符串arg_str生成Argument对象
        arg = Argument.parse(arg_str)
        # 断言argument_type函数对Argument对象arg的绑定结果等于预期的expected字符串表示
        self.assertEqual(str(argument_type(arg, binds=arg.name)), str(expected))

    # 使用local.parametrize装饰器定义测试函数test_argumenttype_type，无参数
    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    # 定义测试函数test_argumenttype_type，无参数
    def test_argumenttype_type(self) -> None:
        # 定义测试数据列表data，每个元素为一个元组，包含参数字符串和预期的NamedCType对象
        data = [
            ("Tensor self", NamedCType("self", ConstRefCType(BaseCType(tensorT)))),
            ("Tensor(a!) out", NamedCType("out", MutRefCType(BaseCType(tensorT)))),
            (
                "Tensor? opt",
                NamedCType("opt", ConstRefCType(OptionalCType(BaseCType(tensorT)))),
            ),
            ("Scalar scalar", NamedCType("scalar", ConstRefCType(BaseCType(scalarT)))),
            (
                "Scalar? scalar",
                NamedCType("scalar", ConstRefCType(OptionalCType(BaseCType(scalarT)))),
            ),
            ("int[] size", NamedCType("size", ArrayRefCType(BaseCType(longT)))),
            ("int? dim", NamedCType("dim", OptionalCType(BaseCType(longT)))),
            ("Tensor[] weight", NamedCType("weight", BaseCType(tensorListT))),
            (
                "Scalar[] spacing",
                NamedCType("spacing", ArrayRefCType(ConstRefCType(BaseCType(scalarT)))),
            ),
            (
                "Tensor?[] weight",
                NamedCType("weight", ArrayRefCType(OptionalCType(BaseCType(tensorT)))),
            ),
            (
                "SymInt[]? output_size",
                NamedCType(
                    "output_size", OptionalCType(ArrayRefCType(BaseCType(longT)))
                ),
            ),
            (
                "int[]? dims",
                NamedCType("dims", OptionalCType(ArrayRefCType(BaseCType(longT)))),
            ),
        ]
        # 遍历测试数据data，对每个元组调用self._test_argumenttype_type方法进行测试
        for d in data:
            self._test_argumenttype_type(*d)

    # 定义测试函数_test_returntype_type，接收参数ret_str和expected
    def _test_returntype_type(self, ret_str: str, expected: CType) -> None:
        # 解析返回字符串ret_str生成Return对象
        ret = Return.parse(ret_str)
        # 断言return_type函数对Return对象ret的结果等于预期的expected字符串表示
        self.assertEqual(str(return_type(ret)), str(expected))

    # 使用local.parametrize装饰器定义测试函数test_returntype_type，无参数
    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    # 定义测试函数test_returntype_type，无参数
    def test_returntype_type(self) -> None:
        # 测试数据列表data包含两个元组，每个元组包含返回字符串和预期的CType对象
        data = [
            ("Tensor", BaseCType(tensorT)),
            ("Scalar?", OptionalCType(BaseCType(scalarT))),
            ("int[]", ArrayRefCType(BaseCType(longT))),
            ("Tensor[]", BaseCType(tensorListT)),
            (
                "Scalar[]",
                ArrayRefCType(ConstRefCType(BaseCType(scalarT)))),
            (
                "Tensor?[]", ArrayRefCType(OptionalCType(BaseCType(tensorT)))),
            (
                "SymInt[]?",
                OptionalCType(ArrayRefCType(BaseCType(longT)))),
            (
                "int[]?",
                OptionalCType(ArrayRefCType(BaseCType(longT))))```
        ]
        # 遍历测试数据data，对每个元组调用self._test_returntype_type方法进行测试
        for d in data:
            self._test_returntype_type(*d)
    # 定义一个测试方法，验证返回类型
    def test_returntype_type(self) -> None:
        # 准备测试数据，每个元素是一个元组，包含类型描述和相应的类型对象
        data = [
            ("Tensor", BaseCType(tensorT)),
            ("Tensor(a!)", MutRefCType(BaseCType(tensorT))),
            ("Tensor[]", VectorCType(BaseCType(tensorT))),
        ]
        # 对每个数据项执行测试方法 _test_returntype_type
        for d in data:
            self._test_returntype_type(*d)

    # 使用本地参数化装饰器，定义测试返回类型的方法
    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    # 定义一个测试方法，验证返回类型解析是否正确
    def test_returns_type(self) -> None:
        # 解析给定函数的函数模式，例如 "min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)"
        func = FunctionSchema.parse(
            "min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)"
        )
        # 期望的返回类型是一个元组类型，包含两个 BaseCType(tensorT) 类型对象
        expected = TupleCType([BaseCType(tensorT), BaseCType(tensorT)])
        # 断言返回类型解析结果与预期值相等
        self.assertEqual(str(returns_type(func.returns)), str(expected))

    # 使用本地参数化装饰器，定义测试空返回类型的方法
    @local.parametrize(
        use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False
    )
    # 定义一个测试方法，验证空返回类型解析是否正确
    def test_void_return_type(self) -> None:
        # 解析给定函数的函数模式，例如 "_foreach_add_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()"
        func = FunctionSchema.parse(
            "_foreach_add_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()"
        )
        # 期望的返回类型是一个 BaseCType(voidT) 类型对象
        expected = BaseCType(voidT)
        # 断言返回类型解析结果与预期值相等
        self.assertEqual(str(returns_type(func.returns)), str(expected))
# 如果当前脚本被直接执行（而不是作为模块导入），则执行单元测试
if __name__ == "__main__":
    # 运行单元测试主程序，执行测试用例
    unittest.main()
```