# `.\pytorch\test\test_python_dispatch.py`

```
# Owner(s): ["module: __torch_dispatch__"]

# 导入必要的模块和函数
import tempfile
import unittest
from copy import deepcopy

import torch
from torch import SymInt  # 导入 SymInt 类
from torch._subclasses.fake_tensor import FakeTensorMode  # 导入 FakeTensorMode 类
from torch.cuda.jiterator import _create_jit_fn  # 导入 _create_jit_fn 函数
from torch.fx.experimental.symbolic_shapes import ShapeEnv  # 导入 ShapeEnv 类
from torch.library import _scoped_library, fallthrough_kernel, impl, Library  # 导入若干函数和类
from torch.testing._internal.common_utils import *  # 导入测试工具函数，忽略 F403 错误
import logging  # 导入 logging 模块
import sys  # 导入 sys 模块

# 导入特定模块和函数
import torch._dynamo
from torch._C import DispatchKey, DispatchKeySet  # 导入 DispatchKey 和 DispatchKeySet
from torch._custom_op.functional import register_functional_op  # 导入 register_functional_op 函数
from torch.fx.experimental.proxy_tensor import make_fx  # 导入 make_fx 函数
from torch.multiprocessing.reductions import StorageWeakRef  # 导入 StorageWeakRef 类
from torch.testing._internal.common_device_type import (  # 导入测试设备类型相关函数
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db  # 导入 op_db 函数
from torch.testing._internal.custom_op_db import custom_op_db  # 导入 custom_op_db 函数
from torch.testing._internal.logging_tensor import (  # 导入用于日志记录的类和函数
    capture_logs,
    capture_logs_with_logging_tensor_mode,
    log_input,
    LoggingTensor,
    LoggingTensorMode,
    LoggingTensorReentrant,
)
from torch.testing._internal.two_tensor import TwoTensor  # 导入 TwoTensor 类
from torch.utils import _pytree as pytree  # 导入 _pytree 模块并重命名为 pytree
from torch.utils._mode_utils import all_same_mode, no_dispatch  # 导入 all_same_mode 和 no_dispatch 函数
from torch.utils._python_dispatch import (  # 导入 Python 调度相关函数和类
    _get_current_dispatch_mode,
    _get_current_dispatch_mode_stack,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_map, tree_map_only  # 导入 tree_map 和 tree_map_only 函数


# 在 DataLoader 的 collate_fn 中使用的函数，避免尝试 pickle lambda 函数
def _identity(x):
    return x


# 测试用例类，继承自 unittest.TestCase
class TestDispatcherPythonBindings(TestCase):

    # 测试调用 boxed 函数的方法
    def test_call_boxed(self) -> None:
        # 获取 "aten::sin" 操作的分发模式
        sin = torch._C._dispatch_find_schema_or_throw("aten::sin", "")
        # 创建一个随机张量
        x = torch.randn(3)
        # 调用 boxed 函数进行分发调用
        y = torch._C._dispatch_call_boxed(sin, x)
        # 断言 y 等于 x 的 sin 函数结果
        self.assertEqual(y, x.sin())


# 测试 Python 注册功能的类
class TestPythonRegistration(TestCase):

    # 测试结束时的清理函数，删除测试用的命名空间
    def tearDown(self):
        if hasattr(torch.ops, self.test_ns):
            del torch.ops._test_python_registration
    # 定义测试函数，用于验证覆盖多个库中的 ATen 操作
    def test_override_aten_ops_with_multiple_libraries(self) -> None:
        # 创建一个包含两个元素的张量 x
        x = torch.tensor([1, 2])
        
        # 使用 _scoped_library 上下文管理器创建 my_lib2，覆盖 aten 命名空间中的 IMPL 库
        with _scoped_library("aten", "IMPL") as my_lib2:
            # 使用 _scoped_library 上下文管理器创建 my_lib1，同样覆盖 aten 命名空间中的 IMPL 库
            with _scoped_library("aten", "IMPL") as my_lib1:
                # Example 1: 定义自定义的 neg 操作
                def my_neg(*args, **kwargs):
                    return args[0]._neg_view()

                # 将自定义的 my_neg 实现注册到 my_lib1 中的 neg 操作上，用于 AutogradCPU
                my_lib1.impl("neg", my_neg, "AutogradCPU")

                # 验证 torch.neg(x) 是否为负数张量
                self.assertTrue(torch.neg(x).is_neg())

                # 使用错误示例，尝试在 foo 命名空间下定义 neg 操作，预期会抛出 RuntimeError
                with self.assertRaisesRegex(
                    RuntimeError, "operator name does not match namespace"
                ):
                    with _scoped_library("foo", "DEF") as my_lib3:
                        my_lib3.define("neg(Tensor self) -> Tensor")
                        my_lib3.impl(torch.ops.aten.neg.default, my_neg, "AutogradCPU")

                # Example 2: 定义自定义的 mul 操作
                def my_mul(*args, **kwargs):
                    return torch.zeros_like(args[0])

                # 将自定义的 my_mul 实现注册到 my_lib2 中的 aten::mul.Tensor 操作上，用于 ZeroTensor
                my_lib2.impl("aten::mul.Tensor", my_mul, "ZeroTensor")

                # 创建一个零张量 y，并验证 torch.mul(x, y) 是否为零张量
                y = torch._efficientzerotensor(2)
                self.assertFalse(torch.mul(x, y)._is_zerotensor())

                # 验证不能重写已有 (namespace, op, dispatch_key) 组合的行为，预期会抛出 RuntimeError
                with self.assertRaisesRegex(
                    RuntimeError, "already a kernel registered from python"
                ):
                    my_lib2.impl(torch.ops.aten.mul.Tensor, my_mul, "ZeroTensor")

            # 验证移除 my_lib1 后 my_lib2 的状态未受影响
            self.assertFalse(torch.mul(x, y)._is_zerotensor())

        # 验证在退出所有上下文管理器后，neg 和 mul 的行为恢复为默认行为
        self.assertFalse(torch.neg(x).is_neg())
        self.assertTrue(torch.mul(x, y)._is_zerotensor())

    # 验证如果函数不可调用，则会引发 TypeError
    def test_error_if_fn_not_callable(self):
        with self.assertRaisesRegex(
            TypeError, "Input function is required to be a callable"
        ):
            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl(torch.ops.aten.neg.default, [], "AutogradCPU")
    def test_finalizer(self):
        # 获取 torch.library._impls 的引用计数
        impls_refcnt = sys.getrefcount(torch.library._impls)
        # 创建一个 Library 对象，命名空间为 self.test_ns，类型为 "FRAGMENT"
        lib = Library(self.test_ns, "FRAGMENT")  # noqa: TOR901
        # 定义一个函数签名 "foo123(Tensor x) -> Tensor"
        lib.define("foo123(Tensor x) -> Tensor")

        # 断言 lib 的引用计数为 2（1 为 lib 本身，1 为 sys.getrefcount 的引用）
        self.assertEqual(sys.getrefcount(lib), 2)
        # 当 finalizer 运行时，我们会多得到一个引用，该引用在 finalizer 运行时会被清除
        self.assertEqual(sys.getrefcount(torch.library._impls), impls_refcnt + 1)
        # 断言 lib._op_impls 的引用计数为 3
        self.assertEqual(sys.getrefcount(lib._op_impls), 3)

        def foo123(x):
            pass

        # 将 foo123 函数注册到 lib 中
        lib.impl(f"{self.test_ns}::foo123", foo123, "CPU")
        key = f"{self.test_ns}/foo123/CPU"
        # 断言 key 是否存在于 torch.library._impls 中
        self.assertTrue(key in torch.library._impls)

        saved_op_impls = lib._op_impls

        # 删除 lib 对象，确保 del 操作可以成功执行
        del lib

        # 断言 saved_op_impls 的引用计数为 2（1 为 saved_op_impls 本身，1 为 sys.getrefcount 的引用）
        self.assertEqual(sys.getrefcount(saved_op_impls), 2)

        # 断言 key 不再存在于 torch.library._impls 中
        self.assertTrue(key not in torch.library._impls)

        # 断言 torch.library._impls 的引用计数恢复到原始值
        self.assertEqual(sys.getrefcount(torch.library._impls), impls_refcnt)

    def test_override_cpu_sum(self) -> None:
        # 创建一个标志位列表 run，用于记录函数 my_sum 是否被调用过
        run = [False]

        # 定义一个函数 my_sum，用于替换 torch.sum 的实现
        def my_sum(*args, **kwargs):
            run[0] = True
            return args[0].clone()

        # 在特定的作用域内，使用自定义的 my_sum 函数来替换 "aten::sum" 的实现
        with _scoped_library("aten", "IMPL") as my_lib1:
            my_lib1.impl("aten::sum", my_sum, "CPU")
            x = torch.tensor([1, 2])
            # 断言 torch.sum(x) 的结果等于 x 本身
            self.assertEqual(torch.sum(x), x)
            # 断言 my_sum 函数已经被调用过
            self.assertTrue(run[0])
        # 验证 torch.sum 恢复到原始的行为
        self.assertEqual(torch.sum(x), torch.tensor(3))

    def test_extend_library_with_dispatch_key_arg(self):
        # 定义一个函数 my_sum，用于在自定义的作用域内使用
        def my_sum(*args, **kwargs):
            return args[0].clone()

        # 在特定的作用域内，使用自定义的 dispatch_key="CPU" 创建 _scoped_library 对象 my_lib1
        with _scoped_library("aten", "IMPL", dispatch_key="CPU") as my_lib1:
            # 期望抛出 RuntimeError，并包含 "inconsistent with the dispatch key" 的错误信息
            with self.assertRaisesRegex(
                RuntimeError, "inconsistent with the dispatch key"
            ):
                # 尝试使用 "Conjugate" 的 dispatch key 来实现 "sum" 函数
                my_lib1.impl("sum", my_sum, "Conjugate")
            # 使用默认的 dispatch key 创建 "aten::sum" 的实现
            my_lib1.impl("aten::sum", my_sum)
            x = torch.tensor([1, 2])
            # 断言 torch.sum(x) 的结果等于 x 本身
            self.assertEqual(torch.sum(x), x)
    def test_create_new_library(self) -> None:
        # 使用 _scoped_library 上下文管理器创建名为 "DEF" 的新库 my_lib1
        with _scoped_library(self.test_ns, "DEF") as my_lib1:
            # 在 my_lib1 中定义一个函数 "sum(Tensor self) -> Tensor"
            my_lib1.define("sum(Tensor self) -> Tensor")

            # Example 1
            # 在 my_lib1 中实现名为 "sum" 的函数，部署到 "CPU"
            @torch.library.impl(my_lib1, "sum", "CPU")
            def my_sum(*args, **kwargs):
                # 返回第一个参数的克隆版本
                return args[0].clone()

            # 创建一个张量 x
            x = torch.tensor([1, 2])
            # 获取 torch.ops 中 self.test_ns 命名空间下的 sum 操作符
            op = getattr(torch.ops, self.test_ns).sum
            # 断言 sum(x) 的结果与 x 相等
            self.assertEqual(op(x), x)

            # 创建名为 "IMPL" 的新库 my_lib2
            with _scoped_library(self.test_ns, "IMPL") as my_lib2:
                # Example 2
                # 在 my_lib2 中实现与 op.default 相关的操作，部署到 "ZeroTensor"
                @torch.library.impl(my_lib2, op.default, "ZeroTensor")
                def my_sum_zt(*args, **kwargs):
                    # 如果第一个参数是零张量，则返回一个有效的零张量
                    if args[0]._is_zerotensor():
                        return torch._efficientzerotensor(args[0].shape)
                    else:
                        # 否则返回第一个参数的克隆版本
                        return args[0].clone()

                # 创建一个零张量 y
                y = torch._efficientzerotensor(3)
                # 断言对零张量 y 执行操作 op(y) 后结果是零张量
                self.assertTrue(op(y)._is_zerotensor())
                # 再次断言 sum(x) 的结果与 x 相等
                self.assertEqual(op(x), x)

    def test_create_new_library_fragment_no_existing(self):
        # 使用 _scoped_library 上下文管理器创建名为 "FRAGMENT" 的新库 my_lib
        with _scoped_library(self.test_ns, "FRAGMENT") as my_lib:
            # 在 my_lib 中定义一个函数 "sum2(Tensor self) -> Tensor"
            my_lib.define("sum2(Tensor self) -> Tensor")

            # 在 my_lib 中实现名为 "sum2" 的函数，部署到 "CPU"
            @torch.library.impl(my_lib, "sum2", "CPU")
            def my_sum(*args, **kwargs):
                # 返回第一个参数本身
                return args[0]

            # 创建一个张量 x
            x = torch.tensor([1, 2])
            # 断言调用 torch.ops 中 self.test_ns 命名空间下的 sum2 操作符后结果与 x 相等
            self.assertEqual(getattr(torch.ops, self.test_ns).sum2(x), x)

    def test_create_new_library_fragment_with_existing(self):
        # 使用 _scoped_library 上下文管理器创建名为 "DEF" 的新库 my_lib1
        with _scoped_library(self.test_ns, "DEF") as my_lib1:
            # 创建一个片段
            # 使用 _scoped_library 上下文管理器创建名为 "FRAGMENT" 的新库 my_lib2
            with _scoped_library(self.test_ns, "FRAGMENT") as my_lib2:
                # 在 my_lib2 中定义一个函数 "sum4(Tensor self) -> Tensor"
                my_lib2.define("sum4(Tensor self) -> Tensor")

                # 在 my_lib2 中实现名为 "sum4" 的函数，部署到 "CPU"
                @torch.library.impl(my_lib2, "sum4", "CPU")
                def my_sum4(*args, **kwargs):
                    # 返回第一个参数本身
                    return args[0]

                # 创建一个张量 x
                x = torch.tensor([1, 2])
                # 断言调用 torch.ops 中 self.test_ns 命名空间下的 sum4 操作符后结果与 x 相等
                self.assertEqual(getattr(torch.ops, self.test_ns).sum4(x), x)

                # 创建另一个片段
                # 使用 _scoped_library 上下文管理器创建名为 "FRAGMENT" 的新库 my_lib3
                with _scoped_library(self.test_ns, "FRAGMENT") as my_lib3:
                    # 在 my_lib3 中定义一个函数 "sum3(Tensor self) -> Tensor"
                    my_lib3.define("sum3(Tensor self) -> Tensor")

                    # 在 my_lib3 中实现名为 "sum3" 的函数，部署到 "CPU"
                    @torch.library.impl(my_lib3, "sum3", "CPU")
                    def my_sum3(*args, **kwargs):
                        # 返回第一个参数本身
                        return args[0]

                    # 创建一个张量 x
                    x = torch.tensor([1, 2])
                    # 断言调用 torch.ops 中 self.test_ns 命名空间下的 sum3 操作符后结果与 x 相等
                    self.assertEqual(getattr(torch.ops, self.test_ns).sum3(x), x)
    # 定义测试函数 test_alias_analysis，用于测试别名分析功能
    def test_alias_analysis(self):
        
        # 定义测试辅助函数 test_helper，用于设置测试环境并执行测试
        def test_helper(alias_analysis=""):
            # 创建名为 my_lib1 的 Library 对象，使用 self.test_ns 和 "DEF" 作为参数
            my_lib1 = Library(self.test_ns, "DEF")  # noqa: TOR901
            
            # 初始化调用计数器列表
            called = [0]
            
            # 定义名为 _op 的函数作为 torch.library.define 的装饰器函数
            @torch.library.define(
                my_lib1, "_op() -> None", alias_analysis=alias_analysis
            )
            def _op(*args, **kwargs):
                called[0] += 1
            
            # 定义名为 _test 的 Torch 脚本函数
            @torch.jit.script
            def _test():
                torch.ops._test_python_registration._op()
            
            # 断言 "_test_python_registration::_op" 是否在 _test 的计算图中
            assert "_test_python_registration::_op" in str(_test.graph)
        
        # 使用断言检查是否会引发 AssertionError 异常
        with self.assertRaises(AssertionError):
            test_helper("")  # alias_analysis="FROM_SCHEMA"
        
        # 调用 test_helper 函数，设置 alias_analysis 参数为 "CONSERVATIVE"
        test_helper("CONSERVATIVE")

    # 定义测试函数 test_error_for_unsupported_ns_or_kind，用于测试不支持的命名空间或类型错误情况
    def test_error_for_unsupported_ns_or_kind(self) -> None:
        
        # 使用断言检查是否会引发 ValueError 异常，并匹配指定的错误信息
        with self.assertRaisesRegex(ValueError, "Unsupported kind"):
            # 创建名为 my_lib1 的 Library 对象，使用 "myns" 和 "BLA" 作为参数
            my_lib1 = Library("myns", "BLA")  # noqa: TOR901
        
        # 遍历种类列表 ("DEF", "FRAGMENT")
        for kind in ("DEF", "FRAGMENT"):
            # 使用断言检查是否会引发 ValueError 异常，并匹配指定的错误信息
            with self.assertRaisesRegex(ValueError, "reserved namespace"):
                # 创建名为 my_lib1 的 Library 对象，使用 "prim" 和当前种类作为参数
                my_lib1 = Library("prim", kind)  # noqa: TOR901

    # 定义测试函数 test_returning_symint，用于测试返回 SymInt 类型的功能
    def test_returning_symint(self) -> None:
        # 创建 ShapeEnv 对象和 FakeTensorMode 对象
        shape_env = ShapeEnv()
        fake_tensor_mode = FakeTensorMode(shape_env=shape_env)
        
        # 使用 fake_tensor_mode 创建一个假的 Tensor 对象 ft
        ft = fake_tensor_mode.from_tensor(torch.rand(2, 3))
        
        # 解包 ft 的形状 s0 和 s1
        s0, s1 = ft.shape
        
        # 使用 _scoped_library 设置测试环境并定义函数 "sqsum"
        with _scoped_library(self.test_ns, "DEF") as tlib:
            tlib.define("sqsum(SymInt a, SymInt b) -> SymInt")
            
            # 使用装饰器函数 impl 注册函数 sqsum
            @impl(tlib, "sqsum", "CompositeExplicitAutograd")
            def sqsum(a: SymInt, b: SymInt):
                return a * a + b * b
            
            # 调用通过 torch.ops 获取的函数 sqsum，并返回结果 out
            out = getattr(torch.ops, self.test_ns).sqsum.default(s0, s1)
            
            # 使用 shape_env 对象评估 out 节点的表达式并获取结果 out_val
            out_val = shape_env.evaluate_expr(out.node.expr)
        
        # 使用断言检查 out_val 是否等于 13
        self.assertEqual(out_val, 13)

    # 定义测试函数 test_register_functional_op_error_cases，用于测试注册函数操作的错误情况
    def test_register_functional_op_error_cases(self):
        
        # 使用 _scoped_library 设置测试环境并创建 lib 对象
        with _scoped_library(self.test_ns, "FRAGMENT") as lib:
            
            # 使用断言检查是否会引发 TypeError 异常，并匹配指定的错误信息
            with self.assertRaisesRegex(TypeError, "instance of OpOverload"):
                # 尝试注册函数 "abs" 使用 torch.ops.aten.abs_ 函数
                register_functional_op(lib, "abs", torch.ops.aten.abs_)
            
            # 使用断言检查是否会引发 RuntimeError 异常，并匹配指定的错误信息
            with self.assertRaisesRegex(RuntimeError, "Expected op to be mutable"):
                # 尝试注册函数 "abs" 使用 torch.ops.aten.abs_.default 函数
                register_functional_op(lib, "abs", torch.ops.aten.abs_.default)
            
            # 使用断言检查是否会引发 RuntimeError 异常，并匹配指定的错误信息
            with self.assertRaisesRegex(RuntimeError, "Expected op to be mutable"):
                # 尝试注册函数 "abs" 使用 torch.ops.aten.abs.out 函数
                register_functional_op(lib, "abs", torch.ops.aten.abs.out)
            
            # 定义 schemas 列表，包含多个函数签名字符串
            schemas = [
                "foo(Tensor x, Tensor(a!)[] y) -> ()",
                "foo(Tensor x, Tensor(a!) y, Tensor(b) z) -> Tensor(b)",
                "foo(Tensor x, Tensor(a!) y) -> (Tensor, Tensor(a))",
            ]
        
        # 遍历 schemas 列表
        for schema in schemas:
            # 使用 _scoped_library 设置测试环境并创建 lib 对象
            with _scoped_library(self.test_ns, "FRAGMENT") as lib:
                # 使用 lib 对象定义给定的 schema 函数签名
                lib.define(schema)
                
                # 使用断言检查是否会引发 RuntimeError 异常，并匹配指定的错误信息
                with self.assertRaisesRegex(RuntimeError, "NYI"):
                    # 尝试注册函数 "foo_functional" 使用 getattr(torch.ops, self.test_ns).foo.default 函数
                    register_functional_op(
                        lib,
                        "foo_functional",
                        getattr(torch.ops, self.test_ns).foo.default,
                    )
    # 检查是否是函数变体
    def _check_is_functional_variant(self, mutable_op, functional_op, args):
        # functional op should not mutate
        # 克隆参数以防止函数操作对原参数进行修改
        cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
        # 调用函数式操作并获取其结果
        functional_result = functional_op(*cloned_args)
        # 断言克隆后的参数与原参数相等，以确保函数操作未修改参数
        self.assertEqual(cloned_args, args)

        # 检查函数式操作的结果是否包含可变操作的结果
        # 调用可变操作获取其结果
        mutable_result = mutable_op(*cloned_args)
        # 如果可变操作的结果为空，则初始化为空列表
        if mutable_result is None:
            flat_mutable_result = []
        else:
            # 将可变操作结果扁平化为列表
            flat_mutable_result = pytree.tree_leaves(mutable_result)
        # 将函数式操作结果扁平化为列表
        flat_functional_result = pytree.tree_leaves(functional_result)
        # 断言函数式操作结果的长度大于可变操作结果的长度
        assert len(flat_functional_result) > len(flat_mutable_result)
        # 断言函数式操作结果的前部分与可变操作结果相等
        self.assertEqual(
            flat_functional_result[: len(flat_mutable_result)], flat_mutable_result
        )

        # 检查函数式操作结果的剩余部分是否为被修改过的参数
        # 通过比较克隆后的参数与原参数，找出被修改过的参数
        mutated_args = [
            maybe_mutated_arg
            for maybe_mutated_arg, arg in zip(cloned_args, args)
            if not (
                maybe_mutated_arg is not None
                and arg is not None
                and torch.allclose(maybe_mutated_arg, arg)
            )
        ]
        # 断言函数式操作结果的剩余部分与被修改过的参数相等
        self.assertEqual(
            flat_functional_result[len(flat_mutable_result) :], mutated_args
        )

        # 检查函数化内核确实已经注册
        # 定义一个函数，该函数内部调用可变操作并返回克隆后的参数
        def fn(*args):
            cloned_args = pytree.tree_map_only(torch.Tensor, torch.clone, args)
            mutable_op(*cloned_args)
            return cloned_args

        # 使用 `make_fx` 将函数化后的函数应用于参数，构建计算图
        gm = make_fx(torch.func.functionalize(fn))(*args)
        # 初始化函数式操作标志位为假
        has_functional_op = False
        # 遍历计算图的节点
        for node in gm.graph.nodes:
            # 断言节点的目标不是可变操作
            self.assertFalse(node.target is mutable_op)
            # 如果节点的目标是函数式操作，则设置函数式操作标志位为真
            if node.target is functional_op:
                has_functional_op = True
        # 断言函数式操作标志位为真
        self.assertTrue(has_functional_op)

    # 测试注册无返回值的函数式操作
    def test_register_functional_op_no_returns(self):
        # 使用 `_scoped_library` 上下文管理器设置命名空间和库名称
        with _scoped_library(self.test_ns, "FRAGMENT") as lib:
            # 定义一个函数签名并注册到库中
            lib.define("foo(Tensor x, Tensor(a!) y, Tensor z, Tensor(b!) w) -> ()")

            # 实现函数 `foo` 的具体逻辑
            def foo_impl(x, y, z, w):
                y.fill_(3.14)
                w.fill_(2.71)

            # 将函数实现注册到库中的 `foo` 函数
            lib.impl("foo", foo_impl, "CPU")

            # 注册函数式操作，将其关联到库中的 `foo` 函数的默认实现
            register_functional_op(
                lib, "foo_functional", getattr(torch.ops, self.test_ns).foo.default
            )

            # 初始化张量 `x`, `y`, `z`, `w`，并调用函数 `_check_is_functional_variant` 进行检查
            x = torch.randn([])
            y = torch.randn([])
            z = torch.randn([])
            w = torch.randn([])
            self._check_is_functional_variant(
                getattr(torch.ops, self.test_ns).foo.default,
                getattr(torch.ops, self.test_ns).foo_functional.default,
                (x, y, z, w),
            )
    # 定义测试函数：注册带有可选参数的功能操作
    def test_register_functional_op_with_optional(self):
        # 在测试命名空间下创建一个临时库
        with _scoped_library(self.test_ns, "FRAGMENT") as lib:
            # 定义函数签名和参数列表，包括可选参数
            lib.define(
                "foo(Tensor x, Tensor(a!) y, Tensor (b!) z, Tensor(c!)? w) -> ()"
            )

            # 实现函数 foo_impl，对输入的张量进行填充操作
            def foo_impl(x, y, z, w):
                y.fill_(3.14)
                z.fill_(2.71)
                if w is not None:
                    w.fill_(1.618)

            # 将 foo_impl 注册为库中的 foo 函数的实现，限定在 CPU 上执行
            lib.impl("foo", foo_impl, "CPU")

            # 将 torch 操作的 foo.default 注册为功能操作 foo_functional
            register_functional_op(
                lib, "foo_functional", getattr(torch.ops, self.test_ns).foo.default
            )

            # 创建随机张量作为参数
            x = torch.randn([])
            y = torch.randn([])
            z = torch.randn([])
            w = torch.randn([])

            # 检查注册的功能操作是否是预期的变体，参数包括 w 为非空和空的两种情况
            self._check_is_functional_variant(
                getattr(torch.ops, self.test_ns).foo.default,
                getattr(torch.ops, self.test_ns).foo_functional.default,
                (x, y, z, w),
            )
            self._check_is_functional_variant(
                getattr(torch.ops, self.test_ns).foo.default,
                getattr(torch.ops, self.test_ns).foo_functional.default,
                (x, y, z, None),
            )

    # 定义测试函数：注册只返回一个张量的功能操作
    def test_register_functional_op_one_return(self):
        # 在测试命名空间下创建一个临时库
        with _scoped_library(self.test_ns, "FRAGMENT") as lib:
            # 定义函数签名和参数列表，只返回一个张量
            lib.define(
                "foo(Tensor x, Tensor(a!) y, Tensor(c!) z, Tensor(b!) w) -> Tensor"
            )

            # 实现函数 foo_impl，对输入的张量进行填充操作，并返回克隆的张量 x
            def foo_impl(x, y, z, w):
                y.fill_(3.14)
                w.fill_(2.71)
                z.fill_(0.99)
                return x.clone()

            # 将 foo_impl 注册为库中的 foo 函数的实现，限定在 CPU 上执行
            lib.impl("foo", foo_impl, "CPU")

            # 将 torch 操作的 foo.default 注册为功能操作 foo_functional
            register_functional_op(
                lib, "foo_functional", getattr(torch.ops, self.test_ns).foo.default
            )

            # 创建随机张量作为参数
            x = torch.randn([])
            y = torch.randn([])
            z = torch.randn([])
            w = torch.randn([])

            # 检查注册的功能操作是否是预期的变体，包括参数 (x, y, z, w)
            self._check_is_functional_variant(
                getattr(torch.ops, self.test_ns).foo.default,
                getattr(torch.ops, self.test_ns).foo_functional.default,
                (x, y, z, w),
            )
    # 测试函数：测试注册多返回值的函数操作
    def test_register_functional_op_multiple_returns(self):
        # 使用指定命名空间和"FRAGMENT"作用域创建库对象
        with _scoped_library(self.test_ns, "FRAGMENT") as lib:
            # 定义函数签名为"foo(Tensor x, Tensor(a!) y, Tensor z, Tensor(b!) w) -> (Tensor, Tensor)"
            lib.define(
                "foo(Tensor x, Tensor(a!) y, Tensor z, Tensor(b!) w) -> (Tensor, Tensor)"
            )

            # 实现名为foo的函数，对参数y和w进行填充，然后返回x和z的克隆
            def foo_impl(x, y, z, w):
                y.fill_(3.14)
                w.fill_(2.71)
                return x.clone(), z.clone()

            # 将foo_impl函数注册为"foo"操作的实现，指定在CPU上执行
            lib.impl("foo", foo_impl, "CPU")

            # 注册一个功能性变体"foo_functional"，调用torch.ops中self.test_ns命名空间下foo的默认实现
            register_functional_op(
                lib, "foo_functional", getattr(torch.ops, self.test_ns).foo.default
            )

            # 创建四个随机张量x, y, z, w
            x = torch.randn([])
            y = torch.randn([])
            z = torch.randn([])
            w = torch.randn([])

            # 调用辅助函数，检查注册的函数操作的功能性变体
            self._check_is_functional_variant(
                getattr(torch.ops, self.test_ns).foo.default,
                getattr(torch.ops, self.test_ns).foo_functional.default,
                (x, y, z, w),
            )

    # 测试函数：测试注册fallthrough行为
    def test_register_fallthrough(self):
        # 使用"aten"命名空间和"IMPL"作用域创建库对象my_lib
        with _scoped_library("aten", "IMPL") as my_lib:
            # 将mm操作注册为fallthrough_kernel，使用"AutocastCPU"执行
            my_lib.impl("mm", fallthrough_kernel, "AutocastCPU")

            # 创建两个随机张量a, b，指定device为CPU，dtype为float32
            a = torch.randn(2, 3, device="cpu", dtype=torch.float32)
            b = torch.randn(3, 2, device="cpu", dtype=torch.float32)

            # 在autocast模式下，设定device_type为CPU，dtype为bfloat16
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                # 检查torch.mm(a, b)的dtype应为float32，因为我们注册了fallthrough
                self.assertEqual(torch.mm(a, b).dtype, torch.float32)
                # 没有注册fallthrough的操作如torch.matmul应不受影响，其dtype应为bfloat16
                self.assertEqual(torch.matmul(a, b).dtype, torch.bfloat16)

        # 恢复默认autocast行为，验证torch.mm(a, b)的dtype应为bfloat16
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            self.assertEqual(torch.mm(a, b).dtype, torch.bfloat16)
class TestPythonDispatch(TestCase):
    # 定义测试类 TestPythonDispatch，继承自 TestCase

    def test_basic(self) -> None:
        # 定义测试方法 test_basic，返回类型为 None

        with capture_logs() as logs:
            # 使用 capture_logs() 进行日志捕获，并将结果保存到 logs 变量中
            x = LoggingTensor(torch.tensor([3.0]), requires_grad=True)
            # 创建一个 LoggingTensor 对象 x，包含值为 3.0 的张量，要求梯度计算
            log_input("x", x)
            # 记录输入 x 到日志中
            y = x * x
            # 计算 x 的平方，并赋值给 y
            saved_x = y.grad_fn._saved_self
            # 从 y 的梯度函数中获取 _saved_self，并赋值给 saved_x
            grad_y = LoggingTensor(torch.tensor([1.0]))
            # 创建一个 LoggingTensor 对象 grad_y，包含值为 1.0 的张量
            log_input("grad_y", grad_y)
            # 记录输入 grad_y 到日志中
            (g,) = torch.autograd.grad((y,), (x,), (grad_y,))
            # 计算 y 对 x 的梯度，并将结果保存到 g 中

        self.assertEqual(g.elem, torch.tensor([6.0]))
        # 断言 g 的元素与 torch.tensor([6.0]) 相等
        with torch.no_grad():
            # 使用 torch.no_grad() 上下文管理器
            self.assertEqual(saved_x, x)
            # 断言 saved_x 等于 x
            self.assertEqual(saved_x._version, x._version)
            # 断言 saved_x 的版本号等于 x 的版本号
            x.add_(2)
            # 将 x 增加 2
            self.assertEqual(saved_x, x)
            # 再次断言 saved_x 等于 x
            # TODO: figure out why broken
            # self.assertEqual(saved_x._version, x._version)
            # 断言 saved_x 的版本号等于 x 的版本号（注释掉的部分）

        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = torch._ops.aten.mul.Tensor($0, $0)
$2: f32[1] = input('grad_y')
$3: f32[1] = torch._ops.aten.mul.Tensor($2, $0)
$4: f32[1] = torch._ops.aten.mul.Tensor($2, $0)
$5: f32[1] = torch._ops.aten.add.Tensor($4, $3)""",
        )
        # 断言日志 logs 的内容与预期的字符串相等

    def test_out(self) -> None:
        # 定义测试方法 test_out，返回类型为 None

        with capture_logs() as logs:
            # 使用 capture_logs() 进行日志捕获，并将结果保存到 logs 变量中
            x = LoggingTensor(torch.ones(1))
            # 创建一个 LoggingTensor 对象 x，包含值为 1.0 的张量
            y = LoggingTensor(torch.zeros(1))
            # 创建一个 LoggingTensor 对象 y，包含值为 0.0 的张量
            log_input("x", x)
            # 记录输入 x 到日志中
            log_input("y", y)
            # 记录输入 y 到日志中
            torch.abs(x, out=y)
            # 计算 x 的绝对值，将结果存储到 y 中

        self.assertEqual(y.elem, torch.ones(1))
        # 断言 y 的元素为 torch.ones(1)
        # TODO: arguably this shouldn't pass and we should complain
        # that out isn't a kwarg
        # 断言日志 logs 的内容与预期的字符串相等

    def test_kwarg_only(self) -> None:
        # 定义测试方法 test_kwarg_only，返回类型为 None

        with capture_logs() as logs:
            # 使用 capture_logs() 进行日志捕获，并将结果保存到 logs 变量中
            x = LoggingTensor(torch.ones(1))
            # 创建一个 LoggingTensor 对象 x，包含值为 1.0 的张量
            y = LoggingTensor(torch.ones(1, 1))
            # 创建一个 LoggingTensor 对象 y，包含值为 1.0 的 1x1 张量
            z = LoggingTensor(torch.ones(1))
            # 创建一个 LoggingTensor 对象 z，包含值为 1.0 的张量
            log_input("x", x)
            # 记录输入 x 到日志中
            log_input("y", y)
            # 记录输入 y 到日志中
            log_input("z", z)
            # 记录输入 z 到日志中
            torch.addmv(x, y, z)
            # 执行 torch.addmv 操作，参数为 x, y, z
            torch.addmv(x, y, z, beta=1)
            # 执行 torch.addmv 操作，参数为 x, y, z, beta=1
            torch.addmv(x, y, z, beta=2)
            # 执行 torch.addmv 操作，参数为 x, y, z, beta=2
            torch.addmv(x, y, z, alpha=2)
            # 执行 torch.addmv 操作，参数为 x, y, z, alpha=2
            torch.addmv(x, y, z, beta=2, alpha=2)
            # 执行 torch.addmv 操作，参数为 x, y, z, beta=2, alpha=2

        # The expectation is that beta/alpha don't show up when they're
        # defaulted.  This is even if the user explicitly specified it.
        # 断言日志 logs 的内容与预期的字符串相等
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1, 1] = input('y')
$2: f32[1] = input('z')
$3: f32[1] = torch._ops.aten.addmv.default($0, $1, $2)
$4: f32[1] = torch._ops.aten.addmv.default($0, $1, $2)
$5: f32[1] = torch._ops.aten.addmv.default($0, $1, $2, beta=2)
$6: f32[1] = torch._ops.aten.addmv.default($0, $1, $2, alpha=2)
$7: f32[1] = torch._ops.aten.addmv.default($0, $1, $2, beta=2, alpha=2)""",
        )
    # 定义一个测试方法，测试关键字参数和位置默认参数
    def test_kwarg_only_and_positional_default(self) -> None:
        # 使用 capture_logs 上下文管理器捕获日志
        with capture_logs() as logs:
            # 创建 LoggingTensor 对象 x，并记录日志
            x = LoggingTensor(torch.ones(1))
            log_input("x", x)
            # 调用 torch.ops.aten._foobar(x)，无任何额外参数
            torch.ops.aten._foobar(x)
            # 调用 torch.ops.aten._foobar(x) 设置第二个位置参数为 False
            torch.ops.aten._foobar(x, False)
            # 调用 torch.ops.aten._foobar(x) 设置关键字参数 arg3 为 False
            torch.ops.aten._foobar(x, arg3=False)
            # 调用 torch.ops.aten._foobar(x) 设置第二个位置参数为 False，关键字参数 arg3 为 False
            torch.ops.aten._foobar(x, False, arg3=False)

        # 断言验证测试结果是否符合预期
        # 在这里我们测试的是，即使设置了关键字参数，如果第二个位置参数是默认的，我们会忽略它
        self.assertExpectedInline(
            "\n".join(logs),
            """\
# 从输入获取一个单元素的浮点数张量 'x'
$0: f32[1] = input('x')

# 使用默认选项调用 torch._ops.aten._foobar.default 函数，并传入 $0 作为参数
$1: f32[1] = torch._ops.aten._foobar.default($0)

# 调用 torch._ops.aten._foobar.default 函数，显式指定第二个参数为 False
$2: f32[1] = torch._ops.aten._foobar.default($0, False)

# 调用 torch._ops.aten._foobar.default 函数，指定关键字参数 arg3 为 False
$3: f32[1] = torch._ops.aten._foobar.default($0, arg3=False)

# 调用 torch._ops.aten._foobar.default 函数，同时显式指定第二个参数和关键字参数 arg3 为 False
$4: f32[1] = torch._ops.aten._foobar.default($0, False, arg3=False)
    # 定义测试函数，用于测试当返回值无效时是否会得到合理的错误消息
    def test_invalid_ret(self) -> None:
        # 定义一个继承自 torch.Tensor 的子类 A
        class A(torch.Tensor):
            # 重载静态方法 __new__，返回一个带有 requires_grad 属性的子类实例
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            # 定义类方法 __torch_dispatch__，返回字符串 "arf"
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return "arf"

        # 使用 assertRaisesRegex 确保调用 lambda 函数时会抛出 RuntimeError，并且错误消息包含 "Unable to cast"
        self.assertRaisesRegex(
            RuntimeError,
            "Unable to cast",
            lambda: A(torch.zeros(1)).neg(),  # 对创建的 A 类对象调用 neg 方法
        )
        # 再次使用 assertRaisesRegex 确保调用 lambda 函数时会抛出 RuntimeError，并且错误消息包含 "Unable to cast"
        self.assertRaisesRegex(
            RuntimeError,
            "Unable to cast",
            lambda: A(torch.zeros(1)).detach(),  # 对创建的 A 类对象调用 detach 方法
        )

    # 定义测试函数，测试调用 detach 方法一次时是否正确记录日志
    def test_detach_appears_twice_when_called_once(self) -> None:
        # 使用 capture_logs() 上下文管理器捕获日志
        with capture_logs() as logs:
            # 创建一个 LoggingTensor 对象 x，设置 requires_grad=True
            x = LoggingTensor(torch.tensor([3.0]), requires_grad=True)
            # 记录输入日志，标识为 "x"，记录 x 的信息
            log_input("x", x)
            # 调用 x 的 detach 方法
            x.detach()
        # FIXME: 实际上我们希望这里只记录一个 detach。然而，当前情况下会记录两次，
        # 原因不明确。保留这个测试用例以确保我们不会进一步倒退（如果调用一次 detach() 会导致三次或更多次 detach() 就很糟糕）。
        # 使用 assertExpectedInline 断言捕获的日志和预期的字符串一致
        self.assertExpectedInline(
            "\n".join(logs),
            """\
# 定义一个输入节点，将用户输入的值赋给 f32 类型的张量 $0
$0: f32[1] = input('x')
# 使用 torch._ops.aten.detach.default 操作将 $0 张量从计算图中分离，生成新的张量 $1
$1: f32[1] = torch._ops.aten.detach.default($0)
# 同样使用 detach 操作将 $1 张量从计算图中分离，生成最终的张量 $2
$2: f32[1] = torch._ops.aten.detach.default($1)
    # 定义一个测试方法，用于自定义自动求导的功能测试
    def test_custom_autograd(self) -> None:
        # 定义一个逃逸变量列表，用于保存对象以测试自动求导函数的反向传播
        escape = [None]

        # 定义一个自定义的平方函数，继承自torch.autograd.Function
        class Square(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # 前向传播函数：计算输入张量的平方
                y = x**2
                # 保存上下文信息，以便在反向传播时使用
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                # 反向传播函数：计算梯度
                assert isinstance(grad_output, LoggingTensor)
                # 获取保存的张量 x
                (x,) = ctx.saved_tensors
                assert isinstance(x, LoggingTensor)
                # 将 x 存入逃逸变量列表，用于后续测试
                escape[0] = x
                return grad_output * 2 * x

        # 使用 capture_logs() 上下文管理器捕获日志输出
        with capture_logs() as logs:
            # 创建一个记录张量 x，并设置 requires_grad=True 启用梯度计算
            x = LoggingTensor(torch.ones(1), requires_grad=True)
            log_input("x", x)
            # 创建一个记录张量 x.grad，并初始化为零
            x.grad = LoggingTensor(torch.zeros(1))
            log_input("x.grad", x.grad)
            # 调用自定义的 Square 函数计算 y = x^2
            y = Square.apply(x)
            # 创建一个记录张量 grad_output，并初始化为全为1
            grad_output = LoggingTensor(torch.ones(1))
            log_input("grad_output", grad_output)
            # 对 y 进行反向传播，计算梯度
            y.backward(grad_output)

        # 使用 torch.no_grad() 上下文管理器禁用梯度计算，进行断言测试
        with torch.no_grad():
            # 断言逃逸变量中的张量与原始输入张量 x 相同
            self.assertEqual(escape[0], x)
            # 断言逃逸变量中的张量版本与原始输入张量 x 的版本相同
            self.assertEqual(escape[0]._version, x._version)
            # TODO: figure out why x.requires_grad = False doesn't
            # trigger an error for LoggingTensor
            # 对张量 x 添加常数2，继续进行断言测试
            x.add_(2)
            # 断言逃逸变量中的张量与修改后的张量 x 相同
            self.assertEqual(escape[0], x)
            # TODO: figure out why this is broken
            # self.assertEqual(escape[0]._version, x._version)

        # 断言测试方法返回的内联日志与预期的日志相同
        self.assertExpectedInline(
            "\n".join(logs),
            """\
# 定义变量 $0，并从用户输入中获取值，此处用于表示输入张量 x
$0: f32[1] = input('x')

# 定义变量 $1，并从用户输入中获取值，用于表示 x 的梯度
$1: f32[1] = input('x.grad')

# 使用 Torch 的底层操作计算 $0 的平方，结果存储在 $2 中
$2: f32[1] = torch._ops.aten.pow.Tensor_Scalar($0, 2)

# 定义变量 $3，并从用户输入中获取值，表示梯度输出
$3: f32[1] = input('grad_output')

# 使用 Torch 的底层操作将 $3 乘以 2，结果存储在 $4 中
$4: f32[1] = torch._ops.aten.mul.Tensor($3, 2)

# 使用 Torch 的底层操作将 $4 乘以 $0，结果存储在 $5 中
$5: f32[1] = torch._ops.aten.mul.Tensor($4, $0)

# 使用 Torch 的底层操作将 $1 和 $5 相加，并将结果存储回 $1 中
$6: f32[1] = torch._ops.aten.add_.Tensor($1, $5)
    def test_make_fx_with_subclass(self) -> None:
        # 定义一个函数 f，接受两个参数 x 和 y，返回两个张量的乘积和两倍的 y
        def f(x, y):
            # Returns (TwoTensor, Tensor)
            return x * y, y + y

        # 创建两个零张量 x_a 和 x_b，每个张量有四个元素
        x_a = torch.zeros(4)
        x_b = torch.zeros(4)
        # 创建一个全为 1 的张量 y，有四个元素
        y = torch.ones(4)

        # 在 make_fx() 中，不负责展开张量子类的输入，
        # 因此我们在这里手动进行展开。
        # 为什么要这样做？一般来说，make_fx(f)(*args) 承诺返回的图形具有与 f(*args) 相同的调用约定。
        # 展开张量子类的输入可能会改变图形的输入参数数量，从而破坏该假设。
        def f_to_trace(x_a, x_b, y):
            # 使用 TwoTensor 类将 x_a 和 x_b 封装成一个新的张量 x
            x = TwoTensor(x_a, x_b)
            # 调用函数 f 处理 x 和 y，得到两个输出 out1 和 out2
            out1, out2 = f(x, y)
            # 获取 out1 的展开属性，并使用这些属性从 out1 中提取对应的值
            out1_unwrapped_attrs, _ = out1.__tensor_flatten__()
            return (*[getattr(out1, attr) for attr in out1_unwrapped_attrs], out2)

        # 使用 make_fx() 将 f_to_trace 转换成图形 fx_g，并传入 x_a, x_b, y 作为参数
        fx_g = make_fx(f_to_trace, tracing_mode="fake")(x_a, x_b, y)
        # 断言 fx_g 的代码与预期的内联代码一致
        self.assertExpectedInline(
            fx_g.code,
            """\
    def forward(self, x_a_1, x_b_1, y_1):
        # 使用 torch.ops.aten.mul.Tensor 计算 x_a_1 和 y_1 的乘积，并将结果赋给 mul，然后将 x_a_1 置为 None
        mul = torch.ops.aten.mul.Tensor(x_a_1, y_1);  x_a_1 = None
        # 使用 torch.ops.aten.mul.Tensor 计算 x_b_1 和 y_1 的乘积，并将结果赋给 mul_1，然后将 x_b_1 置为 None
        mul_1 = torch.ops.aten.mul.Tensor(x_b_1, y_1);  x_b_1 = None
        # 使用 torch.ops.aten.add.Tensor 计算 y_1 和 y_1 的加法，并将结果赋给 add，然后将 y_1 置为 None
        add = torch.ops.aten.add.Tensor(y_1, y_1);  y_1 = None
        # 返回三个计算结果作为元组 (mul, mul_1, add)
        return (mul, mul_1, add)
    """
    ),
    # 此处是一个测试方法，用于验证 GitHub 上的一个问题
    def test_return_and_correct_aliasing_gives_correct_stride(self):
        t = TwoTensor(torch.randn(2, 2), torch.randn(2, 2))
        x = torch.randn(2, 2)
        # 断言切片操作后 TwoTensor 对象的步长与密集张量 x 的步长相同
        self.assertEqual(t[:, 0].stride(), x[:, 0].stride())

    # 此处是一个测试方法，用于验证子类包装的元数据传播功能
    def test_make_wrapper_subclass_propagates_metadata(self) -> None:
        class WrapperTensor(torch.Tensor):
            elem: torch.Tensor

            __slots__ = ["elem"]

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                # 使用 torch.Tensor._make_wrapper_subclass 创建包装子类对象，并传播元数据
                r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                    strides=elem.stride(),
                    storage_offset=elem.storage_offset(),
                )
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                raise RuntimeError("NYI")

        # 创建一个非连续步长和非零存储偏移的张量 x
        x = torch.randn(4, 6).t().diagonal(offset=2)
        y = WrapperTensor(x)
        # 断言包装张量 y 的大小与张量 x 的大小相同
        self.assertEqual(y.size(), x.size())
        # 断言包装张量 y 的步长与张量 x 的步长相同
        self.assertEqual(y.stride(), x.stride())
        # 断言包装张量 y 的存储偏移与张量 x 的存储偏移相同
        self.assertEqual(y.storage_offset(), x.storage_offset())

    # 此处是一个测试方法，用于验证子类包装的序列化功能
    def test_wrapper_subclass_serializes(self) -> None:
        with tempfile.TemporaryFile() as f:
            # 使用 LoggingTensor 创建一个张量 x，并使用 torch.save 将其保存到临时文件中
            # 用于测试非默认 dtype 的情况（此处使用 int64）
            x = LoggingTensor(torch.randperm(3))
            torch.save(x, f)
            f.seek(0)
            # 使用 torch.load 从临时文件中加载张量 x，并断言其类型与原始张量 x 的类型相同
            x_loaded = torch.load(f)
            self.assertTrue(type(x_loaded) is type(x))
            # 断言张量 x_loaded 与张量 x 的值相等
            self.assertEqual(x, x_loaded)
            # 断言张量 x_loaded 的 elem 属性与张量 x 的 elem 属性相等
            self.assertEqual(x.elem, x_loaded.elem)
            # 断言张量 x 和 x_loaded 不是同一个对象
            self.assertFalse(x is x_loaded)

    # 此处是一个测试方法，用于验证深拷贝子类包装的功能
    def test_deepcopy_wrapper_subclass(self) -> None:
        # 使用 LoggingTensor 创建一个张量 x，并对其进行深拷贝
        x = LoggingTensor(torch.randperm(3))
        x_copy = deepcopy(x)
        # 断言拷贝后的张量 x_copy 的类型与原始张量 x 的类型相同
        self.assertTrue(type(x_copy) is type(x))
        # 断言张量 x_copy 与张量 x 的值相等
        self.assertEqual(x, x_copy)
        # 断言张量 x_copy 的 elem 属性与张量 x 的 elem 属性相等
        self.assertEqual(x.elem, x_copy.elem)
        # 断言张量 x 和 x_copy 不是同一个对象
    ) -> None:
        # 定义一个名为 MyWrapperTensor 的子类，继承自 torch.Tensor
        class MyWrapperTensor(torch.Tensor):
            # 声明一个名为 elem 的成员变量
            elem: torch.Tensor

            # 限定只允许存在 elem 这一个成员变量
            __slots__ = ["elem"]

            # 静态方法，用来创建 MyWrapperTensor 的新实例
            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                # 调用 torch.Tensor._make_wrapper_subclass 方法创建一个包装子类
                r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                    strides=elem.stride(),
                    storage_offset=elem.storage_offset(),
                )
                # 将传入的 elem 赋值给新创建的实例的 elem 属性
                r.elem = elem
                return r

            # 类方法，处理 torch 分发机制
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # 如果调用的函数是 clone()
                if func.overloadpacket.__name__ == "clone":
                    # 返回 elem 属性的克隆副本
                    return args[0].elem.clone()
                # 抛出运行时异常，表示该功能尚未实现
                raise RuntimeError("NYI")

            # 注意事项：默认的 Tensor.__torch_function__ 实现在深拷贝（clone()）时会禁用 __torch_function__，
            # 所以不需要显式地为这个子类禁用 __torch_function__。

        # 创建 MyWrapperTensor 类的实例 x，传入一个随机生成的张量
        x = MyWrapperTensor(torch.randn(3))
        
        # 使用 assertRaisesRegex 断言，捕获 RuntimeError 异常，检查其错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            "for which cloning returns another instance of the same subclass",
        ):
            # 对 x 进行深拷贝操作
            x_copy = deepcopy(x)

    # 定义一个测试函数，用于测试非包装子类的深拷贝操作
    def test_deepcopy_non_wrapper_subclass(self) -> None:
        # 确保常见错误情况下会抛出正确的错误信息
        # 定义一个错误的子类，未实现 new_empty() 方法
        class SubTensorError1(torch.Tensor):
            # 默认实现的 new_empty() 返回一个普通张量
            pass

        # 定义另一个错误的子类，new_empty() 方法返回错误类型（普通张量）
        class SubTensorError2(torch.Tensor):
            def new_empty(self, shape):
                return torch.Tensor(shape)

        # 遍历两种错误的子类情况
        for error_cls in [SubTensorError1, SubTensorError2]:
            # 创建 error_cls 的实例 x
            x = error_cls(3)
            # 使用 assertRaisesRegex 断言，捕获 RuntimeError 异常，检查其错误信息
            with self.assertRaisesRegex(
                RuntimeError,
                "for which that function returns another instance of the same subclass",
            ):
                # 对 x 进行深拷贝操作
                x_copy = deepcopy(x)

        # 确保正确实现 new_empty() 方法能使深拷贝正常工作
        # 定义一个成功实现 new_empty() 方法的子类
        class SubTensorSuccess(torch.Tensor):
            def new_empty(self, shape):
                # 返回当前类型的新实例
                return type(self)(shape)

        # 创建 SubTensorSuccess 的实例 x
        x = SubTensorSuccess(3)
        # 对 x 进行深拷贝操作
        x_copy = deepcopy(x)
        # 使用 assertIs 断言，检查深拷贝后的类型与原始类型相同
        self.assertIs(type(x_copy), type(x))
    def test_wrapper_subclass_extra_dispatch_keys(self) -> None:
        class ExtraKeysTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                # NB: only the non-kwarg overload of _make_wrapper_subclass supports
                #     extra dispatch keys. We probably want to unify the two APIs
                #     in the future.
                # 警告：只有不支持额外分派键的 _make_wrapper_subclass 的非关键字参数重载。我们可能希望在未来统一这两个 API。
                r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                    cls,
                    elem.size(),                    # 获取张量的尺寸
                    elem.stride(),                  # 获取张量的步幅
                    elem.storage_offset(),          # 获取张量的存储偏移量
                    torch.contiguous_format,        # 使用连续格式
                    elem.dtype,                     # 获取张量的数据类型
                    elem.layout,                    # 获取张量的布局
                    elem.device,                    # 获取张量的设备
                    False,                          # 不需要梯度
                    False,                          # 不是复制的张量
                    None,                           # 无需复制元数据
                    False,                          # 不是用于自动微分
                    False,                          # 不需要计算梯度
                    DispatchKeySet(DispatchKey.NestedTensor),  # 使用嵌套张量的分派键集合
                )
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # 静态方法：当调用 torch_dispatch 方法时，记录调用的函数，并返回 MyTensor 包装的张量
                called_funcs.append(func)
                return MyTensor(torch.tensor(3))

        x = ExtraKeysTensor(torch.randn(3))                    # 创建 ExtraKeysTensor 类的实例 x
        self.assertTrue(torch._C._dispatch_keys(x).has(DispatchKey.NestedTensor))  # 断言：x 使用 NestedTensor 分派键
        self.assertFalse(
            torch._C._dispatch_keys(x).has(DispatchKey.AutogradNestedTensor)
        )                                                       # 断言：x 不使用 AutogradNestedTensor 分派键

    def test_wrapper_subclass_multiprocessing_preserves_dtype(self):
        # a and b have dtype of int64, which is purposefully different from the default
        # assumed by _make_wrapper_subclass().
        a = torch.randperm(5)                                   # 创建一个随机排列的张量 a
        b = torch.randperm(5)                                   # 创建一个随机排列的张量 b
        data = TwoTensor(a, b)                                  # 创建 TwoTensor 对象 data，包含张量 a 和 b
        expected_dtype = data.dtype                             # 获取 data 的数据类型

        loader = torch.utils.data.DataLoader(
            [data, data],                                       # 加载数据为包含两个 data 对象的列表
            batch_size=2,                                       # 指定批次大小为 2
            num_workers=2,                                      # 指定使用的工作线程数为 2
            collate_fn=_identity,                               # 指定用于批处理的函数为 _identity
        )
        for batch in loader:                                    # 迭代加载器中的每个批次
            self.assertEqual(batch[0].dtype, expected_dtype)    # 断言：批次中第一个张量的数据类型与预期的数据类型相等

    def test_index_put_where_only_index_is_subclass(self) -> None:
        called_funcs = []                                       # 初始化空列表 called_funcs

        class MyTensor(torch.Tensor):                           # 定义 MyTensor 类，继承自 torch.Tensor
            elem: torch.Tensor                                  # 类型注解：elem 是 torch.Tensor 类型
            __slots__ = ["elem"]                                # 限制 MyTensor 类的实例只能有一个属性 elem

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                r = torch.Tensor._make_wrapper_subclass(        # 创建 MyTensor 的新实例 r
                    cls,
                    elem.size(),                                # 获取张量 elem 的尺寸
                    dtype=elem.dtype,                           # 获取张量 elem 的数据类型
                    layout=elem.layout,                         # 获取张量 elem 的布局
                    device=elem.device,                         # 获取张量 elem 的设备
                    requires_grad=elem.requires_grad,           # 获取张量 elem 是否需要梯度
                )
                r.elem = elem                                   # 将 elem 赋值给实例的 elem 属性
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # 类方法：当调用 torch_dispatch 方法时，记录调用的函数，并返回包含张量 3 的 MyTensor 对象
                called_funcs.append(func)
                return MyTensor(torch.tensor(3))

        x = torch.randn(3, 3)                                   # 创建一个形状为 (3, 3) 的随机张量 x
        idxs = (MyTensor(torch.tensor(0)),)                    # 创建包含 MyTensor 对象的索引元组 idxs
        v = torch.randn(1)                                      # 创建一个随机张量 v
        res = x.index_put_(idxs, v)                             # 使用索引 idxs 将张量 v 放入张量 x 中
        self.assertEqual(called_funcs, [torch.ops.aten.index_put_.default])
    # 定义测试方法 test_torch_dispatch_mode_basic，不返回任何值
    def test_torch_dispatch_mode_basic(self) -> None:
        # 使用 capture_logs(is_mode=True) 开始捕获日志，标志为模式日志
        with capture_logs(is_mode=True) as logs:
            # 进入 LoggingTensorMode 上下文，记录张量操作日志
            with LoggingTensorMode():
                # 创建一个空的 PyTorch 张量
                torch.empty([])
        # 使用 self.assertExpectedInline 断言捕获的日志内容符合预期
        self.assertExpectedInline(
            "\n".join(logs),
            """\
# 定义测试类，用于测试不相关张量的 Torch 分发模式
class TestLoggingTensorMode(TestCase):

    # 测试当不相关张量时的 Torch 分发模式
    def test_torch_dispatch_mode_unrelated_tensors(self) -> None:
        # 创建随机张量 x
        x = torch.randn([])
        # 创建随机张量 y
        y = torch.randn([])
        # 捕获日志并进入模式
        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode():
                # 执行张量加法操作
                x + y
        # 断言日志输出
        self.assertExpectedInline(
            "\n".join(logs), """$2: f32[] = torch._ops.aten.add.Tensor($0, $1)"""
        )

    # 测试嵌套推入 LoggingTensorMode 的情况
    def test_nested_push_logging_tensor_mode(self):
        # 创建随机张量 x
        x = torch.randn([])
        # 创建随机张量 y
        y = torch.randn([])
        # 捕获日志并进入模式
        with capture_logs(is_mode=True) as logs:
            with LoggingTensorMode():
                with LoggingTensorMode():
                    # 创建空张量
                    torch.empty([])
                    # 执行张量加法操作
                    x + y

        # 断言日志输出
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$3: f32[] = torch._ops.aten.add.Tensor($1, $2)
$3: f32[] = torch._ops.aten.add.Tensor($1, $2)""",
        )

    # 测试在 Torch 分发模式下捕获日志
    def test_capture_logs_with_torch_dispatch_mode(self):
        # 创建随机张量 x
        x = torch.randn([])
        # 创建随机张量 y
        y = torch.randn([])
        # 捕获带 LoggingTensorMode 的日志
        with capture_logs_with_logging_tensor_mode() as logs:
            # 创建空张量
            torch.empty([])
            # 执行张量加法操作
            x + y
        # 断言日志输出
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$3: f32[] = torch._ops.aten.add.Tensor($1, $2)""",
        )

        # 创建随机张量 x
        x = torch.randn([])
        # 创建随机张量 y
        y = torch.randn([])

        # 捕获带 LoggingTensorMode 的日志
        with capture_logs_with_logging_tensor_mode() as logs1:
            with capture_logs_with_logging_tensor_mode() as logs2:
                # 创建空张量
                torch.empty([])
                # 执行张量加法操作
                x + y

        # 断言日志输出
        self.assertExpectedInline(
            "\n".join(logs2),
            """\
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
$3: f32[] = torch._ops.aten.add.Tensor($1, $2)
$3: f32[] = torch._ops.aten.add.Tensor($1, $2)""",
        )

        # 断言 logs1 和 logs2 相等
        self.assertEqual(logs1, logs2)
    def test_torch_dispatch_mode_subclass_priority(self) -> None:
        class ErrorA(RuntimeError):
            pass

        class ErrorB(RuntimeError):
            pass

        class A(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                # 使用 Tensor 的子类方法创建新的 A 类对象
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # 在 A 类中的 torch_dispatch 方法，抛出 ErrorA 异常
                with AMode():
                    raise ErrorA

        class B(A):
            @staticmethod
            def __new__(cls, elem):
                # 使用 Tensor 的子类方法创建新的 B 类对象
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # 在 B 类中的 torch_dispatch 方法，执行传入的函数
                with BMode():
                    func(*args, **kwargs)

        class AMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # AMode 中的 torch_dispatch 方法，抛出 ErrorA 异常
                raise ErrorA

        class BMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # BMode 中的 torch_dispatch 方法，抛出 ErrorB 异常
                raise ErrorB

        # 创建 A 类对象 a 和 B 类对象 b，各自带有空的 Tensor
        a = A(torch.empty(1))
        b = B(torch.empty(1))

        # 断言：a + a 操作应抛出 ErrorA 异常
        with self.assertRaises(ErrorA):
            a + a
        # 断言：a + b 操作应抛出 ErrorB 异常
        with self.assertRaises(ErrorB):
            a + b

        # 断言：在 AMode 下，b + b 操作应抛出 ErrorA 异常
        with self.assertRaises(ErrorA):
            with AMode():
                b + b
        # 断言：在 BMode 下，a + a 操作应抛出 ErrorB 异常
        with self.assertRaises(ErrorB):
            with BMode():
                a + a
        # 断言：在 BMode 下，a + b 操作应抛出 ErrorB 异常
        with self.assertRaises(ErrorB):
            with BMode():
                a + b

    def test_mode_with_make_subclass(self):
        class SubTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                # 使用 Tensor 的子类方法创建新的 SubTensor 类对象
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

        class BasicMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # BasicMode 中的 torch_dispatch 方法，执行传入的函数
                return func(*args, **kwargs)

        # 创建一个标准的 Tensor 对象 x
        x = torch.randn(3)
        # 在 BasicMode 下创建 SubTensor 对象 y
        with BasicMode():
            y = SubTensor(x)
        # 断言：y 应该是 SubTensor 类的实例
        self.assertIsInstance(y, SubTensor)

    def test_torch_dispatch_mode_respects_no_dispatch(self) -> None:
        with capture_logs(is_mode=True) as logs1:
            with LoggingTensorMode():
                # 使用 LoggingTensorMode 记录日志，创建全为 1 的 Tensor
                torch.ones([2, 3])
                # 使用 no_dispatch 上下文管理器，创建全为 1 的 Tensor
                with no_dispatch():
                    torch.ones([2, 3])
        with capture_logs(is_mode=True) as logs2:
            with LoggingTensorMode():
                # 使用 LoggingTensorMode 记录日志，创建全为 1 的 Tensor
                torch.ones([2, 3])
        # 断言：logs1 和 logs2 应该相等
        self.assertEqual(logs1, logs2)
    # 定义一个测试方法，验证浅拷贝和分离操作
    def test_shallow_copy_and_detach(self) -> None:
        # 初始化一个空集合，用于记录已见过的对象
        seen = set()
        # 将当前测试实例赋值给 test_case 变量
        test_case = self

        # 定义一个自定义的 TorchDispatchMode 类 TestMode
        class TestMode(TorchDispatchMode):
            # 重写 __torch_dispatch__ 方法
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # 对 torch.Tensor 类型的参数进行树状映射，检查其是否在 seen 集合中
                tree_map_only(
                    torch.Tensor, lambda t: test_case.assertIn(t, seen), (args, kwargs)
                )
                # 如果 kwargs 为 None，则初始化为空字典
                if kwargs is None:
                    kwargs = {}
                # 执行 func 函数，传入 args 和 kwargs，并接收返回值 r
                r = func(*args, **kwargs)
                # 对返回值 r 中的 torch.Tensor 类型对象进行树状映射，将其添加到 seen 集合中
                tree_map_only(torch.Tensor, lambda t: seen.add(t), r)
                # 返回函数执行结果 r
                return r

        # 使用 TestMode 类进行上下文管理
        with TestMode():
            # 创建一个形状为 (3,) 的张量 x，要求计算梯度
            x = torch.randn(3, requires_grad=True)
            # 计算损失函数，这里是 x 的平方和
            loss = (x * x).sum()
            # 反向传播求梯度
            loss.backward()

    # 定义一个测试方法，验证异常处理
    def test_exception_handling(self):
        # 定义一个继承自 torch.Tensor 的子类 A
        class A(torch.Tensor):
            # 静态方法，根据 elem 创建 A 类对象
            @staticmethod
            def __new__(cls, elem):
                # 调用 torch.Tensor 的 _make_subclass 方法创建子类对象，要求 elem 的梯度需求与 elem 一致
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

        # 定义一个自定义的 TorchDispatchMode 类 AMode
        class AMode(TorchDispatchMode):
            # 重写 __torch_dispatch__ 方法
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # 如果 func 函数名为 "randn.default"，则抛出 RuntimeError 异常
                if func.__name__ == "randn.default":
                    raise RuntimeError
                # 否则返回一个 A 类的对象，该对象值为零的标量张量
                return A(torch.zeros(()))

        # 使用 AMode 类进行上下文管理
        with AMode():
            # 尝试执行 torch.randn(()) 函数，捕获 RuntimeError 异常并忽略
            try:
                torch.randn(())
            except RuntimeError:
                pass
            # 断言 torch.zeros(()) 返回的对象是 A 类的实例
            self.assertTrue(isinstance(torch.zeros(()), A))

    # 定义一个测试方法，验证独立创建的模式对象
    def test_with_mode_created_separately(self):
        # 定义一个继承自 RuntimeError 的异常类 ErrorA
        class ErrorA(RuntimeError):
            pass

        # 定义一个继承自 TorchDispatchMode 的子类 A
        class A(TorchDispatchMode):
            # 重写 __torch_dispatch__ 方法
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # 抛出 ErrorA 异常
                raise ErrorA

        # 创建 A 类的实例 x
        x = A()
        # 使用 self.assertRaises 捕获 ErrorA 异常
        with self.assertRaises(ErrorA):
            # 在 x 的上下文中执行 torch.empty([]) 函数调用
            with x:
                torch.empty([])

    # 定义一个测试方法，验证嵌套模式的行为
    def test_with_nested_modes(self):
        # 定义一个继承自 RuntimeError 的异常类 ErrorA，带有消息参数
        class ErrorA(RuntimeError):
            def __init__(self, msg):
                super().__init__(msg)

        # 定义一个继承自 TorchDispatchMode 的子类 A
        class A(TorchDispatchMode):
            # 构造方法，接收消息参数 msg
            def __init__(self, msg):
                self.msg = msg

            # 重写 __torch_dispatch__ 方法
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # 抛出 ErrorA 异常，异常消息为实例化时传入的 msg
                raise ErrorA(self.msg)

        # 使用 self.assertRaisesRegex 捕获 ErrorA 异常，并验证异常消息包含 "layer2"
        with self.assertRaisesRegex(ErrorA, "layer2"):
            # 在 A("layer1") 上下文中执行以下代码块
            with A("layer1"):
                # 在 A("layer2") 上下文中执行以下代码块
                with A("layer2"):
                    # 执行 torch.empty([]) 函数调用，引发 ErrorA 异常
                    torch.empty([])
    def test_make_subclass_with_modes(self):
        # 定义一个测试函数，用于测试在不同模式下创建子类张量的行为

        class ModeTensor(torch.Tensor):
            def __new__(cls, elem, mode):
                # 自定义张量子类的构造方法，基于父类的_make_subclass方法创建子类张量
                r = torch.Tensor._make_subclass(cls, elem, elem.requires_grad)
                r.elem = elem
                r.mode = mode
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # 类方法，用于处理张量的分发调度，如果调用到这里，抛出未实现的错误
                raise NotImplementedError("Shouldn't be here")

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # 定义模式类的分发方法，包装和解包张量
                def unwrap(e):
                    if isinstance(e, ModeTensor):
                        return e.elem
                    else:
                        return e

                def wrap(t):
                    if isinstance(t, torch.Tensor):
                        return ModeTensor(t, self)
                    else:
                        return t

                return wrap(func(*tuple(unwrap(a) for a in args), **kwargs))

        class BasicMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # 基础模式类的分发方法，直接调用给定的函数
                return func(*args, **kwargs)

        x = torch.tensor(4.0)
        # 创建一个浮点张量 x

        with Mode():
            # 在 Mode 模式下执行以下操作
            y = x + x
            # 对 x 执行加法操作得到 y
            z = y + y
            # 对 y 执行加法操作得到 z
        self.assertIsInstance(y, ModeTensor)
        # 断言 y 是 ModeTensor 的实例
        self.assertIsInstance(z, ModeTensor)
        # 断言 z 是 ModeTensor 的实例

        with Mode():
            # 再次进入 Mode 模式
            with BasicMode():
                # 嵌套进入 BasicMode，因为 make_subclass 只接受普通张量，所以不能嵌套两个调用 make_subclass 的模式
                y = x + x
                # 对 x 执行加法操作得到 y
                z = y + y
                # 对 y 执行加法操作得到 z
        self.assertIsInstance(y, ModeTensor)
        # 断言 y 是 ModeTensor 的实例
        self.assertIsInstance(z, ModeTensor)
        # 断言 z 是 ModeTensor 的实例

        assert self.assertRaisesRegex(
            RuntimeError,
            "subclass Mode but.* associated to a python object of type Mode",
        )
        # 使用断言检查是否抛出了预期的 RuntimeError 异常信息
    def test_nesting_same_mode(self):
        # 如果推入的模式与当前模式是同一个实例，我们允许推入一个已经处于活动状态的模式。

        # 使用 capture_logs 进行日志捕获，设置 is_mode 标志为 True
        with capture_logs(is_mode=True) as logs:
            # 进入 LoggingTensorMode 上下文，并将 reenabled 实例化为当前模式
            with LoggingTensorMode() as reenabled:
                # 在 reenabled 模式下，再次进入同一个模式上下文
                with reenabled:
                    # 调用 torch.empty([]) 生成一个空张量
                    torch.empty([])
            # 使用 assertExpectedInline 方法断言日志内容符合预期
            self.assertExpectedInline(
                "\n".join(logs),
                """\
# 创建一个空的 torch.Tensor，数据类型为 f32（32位浮点数数组）
# 该 Tensor 没有任何元素（空列表 []），并且内存格式为空
# 使用给定的设备（此处为 CPU 设备）来分配内存
# 不将数据固定在内存中（pin_memory=False）
$0: f32[] = torch._ops.aten.empty.memory_format([], device=device(type='cpu'), pin_memory=False)
    # 定义测试方法 test_record_stream，用于测试记录流操作
    def test_record_stream(self) -> None:
        # 定义 TestMode 类，继承自 TorchDispatchMode 类
        class TestMode(TorchDispatchMode):
            # 初始化方法，接收一个 testcase 参数
            def __init__(self, testcase):
                self.testcase = testcase

            # 实现 __torch_dispatch__ 方法，处理 Torch 函数调用
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # 断言函数名为 "aten::record_stream"
                self.testcase.assertEqual(func.name(), "aten::record_stream")
                # 断言第一个参数是 torch.Tensor 类型
                self.testcase.assertIsInstance(args[0], torch.Tensor)
                # 断言第二个参数是 torch.Stream 类型
                self.testcase.assertIsInstance(args[1], torch.Stream)
                # 断言第二个参数的 stream_id 属性为 1
                self.testcase.assertEqual(args[1].stream_id, 1)
                # 断言第二个参数的 device_index 属性为 2
                self.testcase.assertEqual(args[1].device_index, 2)
                # 断言第二个参数的 device_type 属性为 3
                self.testcase.assertEqual(args[1].device_type, 3)

        # 创建一个 tensor t，值为 5.0
        t = torch.tensor(5.0)
        # 创建一个流对象 s，设置 stream_id 为 1，device_index 为 2，device_type 为 3
        s = torch.Stream(stream_id=1, device_index=2, device_type=3)
        
        # 使用 TestMode 类进行测试环境的上下文管理
        with TestMode(self):
            # 对 tensor t 记录流 s
            t.record_stream(s)

    # 定义测试方法 test_return_stream，用于测试返回流操作
    def test_return_stream(self) -> None:
        # 使用 _scoped_library 方法创建测试库 "test_return_stream"，定义 l_def 上下文
        with _scoped_library("test_return_stream", "DEF") as l_def:
            # 在 l_def 上下文中定义 "return_stream(Tensor self) -> Stream"
            l_def.define("return_stream(Tensor self) -> Stream")
            
            # 使用 _scoped_library 方法创建测试库 "test_return_stream"，定义 l_impl 上下文
            with _scoped_library("test_return_stream", "IMPL", "CPU") as l_impl:
                # 在 l_impl 上下文中实现 "return_stream" 方法，返回一个流对象
                l_impl.impl(
                    "return_stream",
                    lambda _: torch.Stream(stream_id=0, device_index=1, device_type=2),
                )

                # 定义 TestMode 类，继承自 TorchDispatchMode 类
                class TestMode(TorchDispatchMode):
                    # 实现 __torch_dispatch__ 方法，返回一个流对象
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        return torch.Stream(stream_id=1, device_index=2, device_type=3)

                # 创建一个 tensor t，值为 5.0
                t = torch.tensor(5.0)
                # 调用 torch.ops.test_return_stream.return_stream 方法，返回流对象 s
                s = torch.ops.test_return_stream.return_stream(t)
                # 断言 s 是 torch.Stream 类型
                self.assertIsInstance(s, torch.Stream)
                # 断言 s 的 stream_id 属性为 0
                self.assertEqual(s.stream_id, 0)
                # 断言 s 的 device_index 属性为 1
                self.assertEqual(s.device_index, 1)
                # 断言 s 的 device_type 属性为 2

                # 使用 TestMode 类进行测试环境的上下文管理
                with TestMode():
                    # 再次调用 torch.ops.test_return_stream.return_stream 方法，返回流对象 s
                    s = torch.ops.test_return_stream.return_stream(t)
                # 断言 s 是 torch.Stream 类型
                self.assertIsInstance(s, torch.Stream)
                # 断言 s 的 stream_id 属性为 1
                self.assertEqual(s.stream_id, 1)
                # 断言 s 的 device_index 属性为 2
                self.assertEqual(s.device_index, 2)
                # 断言 s 的 device_type 属性为 3
    # 定义一个测试方法，用于检查子类的自动梯度设备
    def test_subclass_autograd_device_check(self) -> None:
        # 定义一个继承自 torch.Tensor 的非包装子类
        class NonWrapperSubclass(torch.Tensor):
            # 定义一个名为 elem 的 torch.Tensor 类型的属性
            elem: torch.Tensor

            # 明确声明只允许存在 elem 这一个实例变量
            __slots__ = ["elem"]

            # 静态方法，用于创建新的 NonWrapperSubclass 实例
            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                # 输出错误信息，表明此处设备选择错误
                r = torch.Tensor._make_subclass(
                    cls, elem.to("meta"), elem.requires_grad
                )
                # 将 elem 作为 r 的属性进行保存
                r.elem = elem
                return r

            # 类方法，用于 torch 分发
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # 定义用于解包的函数 unwrap
                def unwrap(e):
                    return e.elem if isinstance(e, NonWrapperSubclass) else e

                # 定义用于包装的函数 wrap
                def wrap(e):
                    return NonWrapperSubclass(e) if isinstance(e, torch.Tensor) else e

                # 对 args 和 kwargs 中的元素进行递归映射处理
                rs = tree_map(
                    wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
                )
                # 记录日志信息到 "NonWrapperSubclass" 日志器中
                logging.getLogger("NonWrapperSubclass").info(
                    f"{func.__module__}.{func.__name__}", args, kwargs, rs  # noqa: G004
                )
                return rs

        # 创建 NonWrapperSubclass 的实例 x，传入一个 torch.Tensor 对象
        x = NonWrapperSubclass(torch.tensor([3.0, 4.0], requires_grad=True))
        # 创建一个随机的 torch.Tensor 对象 y，同时指定需要计算梯度
        y = torch.randn(2, requires_grad=True)
        # 对 x 和 y 进行乘法运算，得到 z，确保 z 是 NonWrapperSubclass 的实例
        z = x * y
        # 断言 z 是 NonWrapperSubclass 的实例
        self.assertIsInstance(z, NonWrapperSubclass)
        # 对 z 的所有元素求和并反向传播梯度，梯度起点为 torch.tensor(1)
        z.sum().backward(torch.tensor(1))
        # 断言 x 的梯度应与 y 的值相等
        self.assertEqual(x.grad, y)
        # 断言 y 的梯度应与 x 的值相等
        self.assertEqual(y.grad, x)
    def test_none_wrapping(self):
        # 定义一个返回 None 的 Tensor 子类，用于在执行加法时返回 None
        # 更多信息请参考上面的 LoggingTensor 类
        class SubclassWithNone(torch.Tensor):
            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                # 创建一个包装的子类，继承自 torch.Tensor
                r = torch.Tensor._make_wrapper_subclass(
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                )
                # 将原始的元素保存在 r.elem 中
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # 解包装函数，如果是 SubclassWithNone 类型，则返回其 elem 属性
                def unwrap(e):
                    return e.elem if isinstance(e, SubclassWithNone) else e

                # 包装函数，如果是 torch.Tensor 类型，则封装为 SubclassWithNone
                def wrap(e):
                    return SubclassWithNone(e) if isinstance(e, torch.Tensor) else e

                # 对参数和关键字参数进行解包装和封装操作
                rs = tree_map(
                    wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
                )
                # 如果函数名为 "add"，则返回 None，否则返回 rs
                if func.overloadpacket.__name__ == "add":
                    return None
                else:
                    return rs

        # 创建一个 SubclassWithNone 实例 x，传入一个随机的 torch.Tensor
        x = SubclassWithNone(torch.rand(2))
        # 确保乘法运算不会出错
        self.assertIsInstance(x * 2, SubclassWithNone)
        # 确保加法运算返回 None
        self.assertIsNone(x + 2)

        # 设置 x 的 requires_grad 为 True
        x.requires_grad_()
        out = x.acos().sum()

        # 对 acos 的反向传播，确保用户代码生成的未定义 Tensor 能够得到良好处理
        with self.assertRaisesRegex(RuntimeError, "but got None"):
            out.backward()

    def test_storage_can_be_converted_to_python_object(self):
        # 创建一个空的 torch.Storage 实例 s
        s = torch.Storage()
        # 创建一个 LoggingTensor 实例 z，传入一个空的 torch.Tensor
        z = LoggingTensor(torch.empty([]))
        # 将 s 设置为 z 的存储
        z.set_(s)

    def test_autograd_in_attr(self):
        # 创建一个需要梯度的随机 Tensor true_t
        true_t = torch.rand(2, requires_grad=True)
        # 创建一个 LoggingTensorReentrant 实例 t，传入 true_t
        t = LoggingTensorReentrant(true_t)

        # 对 t 加 2
        out = t + 2

        # 确保 out 不需要梯度，且没有梯度函数
        self.assertFalse(out.requires_grad)
        self.assertIsNone(out.grad_fn)

        # 确保 out.elem 需要梯度，且有梯度函数
        self.assertTrue(out.elem.requires_grad)
        self.assertIsNotNone(out.elem.grad_fn)

        # 对 out.sum() 执行反向传播，预期会抛出 RuntimeError，说明不需要梯度
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            out.sum().backward()

        # 对 out.elem.sum() 执行反向传播
        out.elem.sum().backward()

        # 确保 t 的梯度为 None，但 t.elem 的梯度不为 None
        self.assertIsNone(t.grad)
        self.assertIsNotNone(t.elem.grad)
    def test_dispatch_super_call(self):
        called = []

        class SubTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                called.append(func)
                # 调用 __torch_dispatch__ 方法时记录调用的函数，并调用父类的同名方法
                return super().__torch_dispatch__(func, types, args, kwargs)

        x = torch.randn(2)
        y = torch.randn(2)
        self.assertEqual(SubTensor(x) + SubTensor(y), x + y)
        # 断言调用记录是否为 torch.ops.aten.add.Tensor
        self.assertEqual(called, [torch.ops.aten.add.Tensor])

    def test_dispatch_super_call_list_arg(self):
        called = []

        class SubTensorWithListArg(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                called.append(func)
                # 调用 __torch_dispatch__ 方法时记录调用的函数，并转换 args 为列表后调用父类的同名方法
                return super().__torch_dispatch__(func, types, list(args), kwargs)

        x = torch.randn(2)
        self.assertEqual(SubTensorWithListArg(x).neg(), x.neg())
        # 断言调用记录是否为 torch.ops.aten.neg.default
        self.assertEqual(called, [torch.ops.aten.neg.default])

    def test_dispatch_super_dont_autograd(self):
        called = []

        class SubTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                called.append(func)
                # 断言第一个参数是否需要梯度
                self.assertTrue(args[0].requires_grad)
                r = super().__torch_dispatch__(func, types, args, kwargs)
                # 断言返回的结果是否不需要梯度
                self.assertFalse(r.requires_grad)
                return r

        x = SubTensor(torch.randn(2, requires_grad=True))
        x.neg()
        # 断言调用记录是否为 torch.ops.aten.neg.default
        self.assertEqual(called, [torch.ops.aten.neg.default])

    def test_set_data(self):
        called = 0

        class SubTensor(torch.Tensor):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                nonlocal called
                called += 1
                return super().__torch_dispatch__(func, types, args, kwargs)

        x = SubTensor(torch.empty(2))
        x.data
        self.assertEqual(called, 1)
        x.data = torch.empty(2)
        self.assertEqual(called, 1)
        x.data
        self.assertEqual(called, 2)
        self.assertIs(type(x), SubTensor)
        x.set_(torch.empty(2))
        self.assertEqual(called, 3)
        x.data
        self.assertEqual(called, 4)
        self.assertIs(type(x), SubTensor)
    # 定义测试方法，构造一个整型张量子类，并确保不会失败
    def test_construct_int_tensor(self):
        # 定义一个继承自 torch.Tensor 的子类 SubTensor
        class SubTensor(torch.Tensor):
            pass

        # 调用 SubTensor 类，传入一个全零的整型张量作为参数，应该不会出错
        SubTensor(torch.zeros(2, dtype=torch.int))

    # 测试多个操作的子类
    def test_multiple_ops_subclass(self):
        # 定义一个直接子类 MySubclass，继承自 torch.Tensor
        # 注意：这种直接继承不建议使用
        class MySubclass(torch.Tensor):
            # 使用静态方法重写 __new__ 方法
            @staticmethod
            def __new__(cls, elem):
                # 调用 torch.Tensor._make_subclass 方法创建子类对象
                r = torch.Tensor._make_subclass(cls, elem)
                return r

            # 使用类方法重写 __torch_dispatch__ 方法
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # 使用 no_dispatch 上下文管理器，禁用分发
                with no_dispatch():
                    return func(*args, **kwargs)

        # 创建 MySubclass 的实例 x，传入一个随机复数张量作为参数
        x = MySubclass(torch.rand(2, 2, dtype=torch.complex64))
        # 对 x 进行共轭操作，得到实例 y
        y = x.conj()

        # 进行 bug 测试的详细说明：
        # 在这里，y 的分发键是: {PythonTLSSnapshot, AutogradCPU, Conjugate, Python, CPU}
        # 这里会发生几次分发调用：
        #  - 调用 exp: 用户在 y 上调用 exp
        #    - PythonTLSSnapshot: 记录进入时的 TLS 并重新分发
        #    - AutogradCPU: 没有输入需要梯度，所以不执行任何操作并重新分发
        #    - Conjugate: exp 没有特殊实现，使用回退策略，首先克隆张量（以生成共轭），然后重新分发
        #      - 调用 clone: 共轭回退调用 y 的 clone
        #        - PythonTLSSnapshot: 记录进入时的 TLS 并重新分发
        #        - (AutogradCPU: 被跳过，因为 autograd 在上面将自身添加到排除集中)
        #        - Conjugate: clone 的特殊实现，直接跳过此键
        #        - Python: 根据上述快照重置 TLS 并调用用户实现（实际上再次调用分发器，但由于我们在此之前
        #                  禁用了两个键，这里不详细展示）
        #        - 退出 Python: 恢复 TLS 并退出
        #        - 退出 Conjugate: 没有原地操作，所以直接退出
        #        - 退出 PythonTLSSnapshot: 完成此调用，将保存的 TLS 重置为空
        #    - Python: 根据快照再次重置 TLS。 <- 这里曾经失败过
        #    - 更多步骤....
        y.exp()

    # 静态方法，辅助创建子类对象
    @staticmethod
    def subclass_helper(cls, data, use_wrapper_subclass, **kwargs):
        if use_wrapper_subclass:
            # 如果使用包装器子类，设置一些参数并调用 torch.Tensor._make_wrapper_subclass 方法
            kwargs["device"] = data.device
            kwargs["dtype"] = data.dtype
            kwargs["layout"] = data.layout
            kwargs["requires_grad"] = True
            return torch.Tensor._make_wrapper_subclass(cls, data.size(), **kwargs)  # type: ignore[attr-defined]
        else:
            # 否则调用 torch.Tensor._make_subclass 方法创建子类对象
            return torch.Tensor._make_subclass(cls, data, True, **kwargs)
    # 定义一个测试函数，用于测试非连续张量的多分派处理
    def test_is_contiguous_slow_path(self):
        # 创建一个 3x3 的随机张量
        data = torch.randn(3, 3)
        # 克隆数据以获得连续的张量
        contiguous_data = data.clone()
        # 使用 torch.as_strided 创建一个非连续的张量
        not_contiguous_data = torch.as_strided(data.clone(), (2, 2), (1, 2))

        # 针对是否使用子类包装的标志进行循环测试
        for use_wrapper_subclass in [True, False]:

            # 定义示例张量子类 ExampleTensor1
            class ExampleTensor1(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    # 使用辅助函数 subclass_helper 创建新的张量实例
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 当调用函数是 torch.ops.aten.is_contiguous 时返回未实现
                    return NotImplemented

            # 定义示例张量子类 ExampleTensor2
            class ExampleTensor2(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    # 使用辅助函数 subclass_helper 创建新的张量实例
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 当调用函数是 torch.ops.aten.is_contiguous 时返回连续数据的判断结果
                    if func.overloadpacket == torch.ops.aten.is_contiguous:
                        return contiguous_data.is_contiguous()
                    return NotImplemented

            # 定义示例张量子类 ExampleTensor3
            class ExampleTensor3(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    # 使用辅助函数 subclass_helper 创建新的张量实例
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 当调用函数是 torch.ops.aten.is_contiguous 时返回非连续数据的判断结果
                    if func.overloadpacket == torch.ops.aten.is_contiguous:
                        return not_contiguous_data.is_contiguous()
                    return NotImplemented

            # 定义错误消息
            err_msg = "Multiple dispatch failed for 'torch.ops.aten.is_contiguous'"
            
            # 创建 ExampleTensor1 实例，并使用断言测试 is_contiguous 方法和 contiguous 方法抛出的错误消息
            e = ExampleTensor1(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.is_contiguous()
            with self.assertRaisesRegex(TypeError, err_msg):
                e.contiguous()

            # 创建 ExampleTensor2 实例，并使用断言测试 is_contiguous 方法的返回结果
            e = ExampleTensor2(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.is_contiguous(), True)
            e.contiguous()  # 这里只会返回原始的 TensorImpl，因为 is_contiguous = True

            # 更新错误消息
            err_msg = "Multiple dispatch failed for"
            
            # 创建 ExampleTensor3 实例，并使用断言测试 is_contiguous 方法的返回结果和 contiguous 方法抛出的错误消息
            e = ExampleTensor3(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.is_contiguous(), False)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.contiguous()
    # 定义一个测试方法，用于测试自定义张量类的特殊步幅功能
    def test_fancy_strides(self):
        # 初始化一个空列表，用于记录调用过的函数和它们的参数
        calls = []

        # 定义一个继承自torch.Tensor的示例张量类ExampleTensor
        class ExampleTensor(torch.Tensor):
            # 静态方法，用于创建新的ExampleTensor实例
            @staticmethod
            def __new__(cls, data):
                # 调用辅助方法subclass_helper来创建新实例
                return TestPythonDispatch.subclass_helper(
                    cls, data, False, dispatch_sizes_strides_policy="strides"
                )

            # 类方法，处理torch分发逻辑
            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                # 如果func是以下函数之一，则记录其调用和参数到calls列表中，并返回None
                if func in [
                    torch.ops.aten.is_contiguous.default,
                    torch.ops.aten.is_contiguous.memory_format,
                    torch.ops.aten.is_strides_like_format.default,
                    torch.ops.aten.is_non_overlapping_and_dense.default,
                    torch.ops.aten.stride.default,
                ]:
                    calls.append((func, list(args)[1:]))
                    return None
                # 否则，使用no_dispatch上下文管理器，调用func并返回其结果
                with no_dispatch():
                    return func(*args, **kwargs)

        # 创建一个ExampleTensor实例e，传入一个2x2的随机张量
        e = ExampleTensor(torch.randn(2, 2))
        
        # 断言e不是按照指定的内存格式（torch.channels_last）连续的
        self.assertFalse(e.is_contiguous(memory_format=torch.channels_last))
        
        # 断言调用列表calls等于指定的调用列表，检查是否正确记录了函数调用和参数
        self.assertEqual(
            calls, [(torch.ops.aten.is_contiguous.memory_format, [torch.channels_last])]
        )
        calls.clear()  # 清空calls列表
        
        # 断言e不是按照指定的步幅格式（torch.channels_last）的
        self.assertFalse(
            torch.ops.aten.is_strides_like_format.default(e, torch.channels_last)
        )
        
        # 断言调用列表calls等于指定的调用列表，检查是否正确记录了函数调用和参数
        self.assertEqual(
            calls,
            [(torch.ops.aten.is_strides_like_format.default, [torch.channels_last])],
        )
        calls.clear()  # 清空calls列表
        
        # 断言e是非重叠且密集的
        self.assertTrue(torch.ops.aten.is_non_overlapping_and_dense.default(e))
        
        # 断言调用列表calls等于指定的调用列表，检查是否正确记录了函数调用和参数
        self.assertEqual(
            calls, [(torch.ops.aten.is_non_overlapping_and_dense.default, [])]
        )
    # 定义测试方法 test_device_slowpath，用于测试特定情况下的设备分发逻辑
    def test_device_slowpath(self):
        # 循环遍历布尔值列表，仅包含一个元素 True
        for use_wrapper_subclass in [True]:

            # 定义 ExampleTensor1 类，继承自 torch.Tensor
            class ExampleTensor1(torch.Tensor):
                # 重载静态方法 __new__，用于创建实例
                @staticmethod
                def __new__(cls, data, wrapper):
                    # 调用辅助方法 subclass_helper 进行实例化
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_device=True
                    )

                # 类方法 __torch_dispatch__，处理 Torch 分发调用
                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 如果 func 的重载包是 torch.ops.prim.device，则返回元数据设备
                    return NotImplemented

            # 定义 ExampleTensor2 类，继承自 torch.Tensor
            class ExampleTensor2(torch.Tensor):
                # 重载静态方法 __new__，用于创建实例
                @staticmethod
                def __new__(cls, data, wrapper):
                    # 调用辅助方法 subclass_helper 进行实例化
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_device=True
                    )

                # 类方法 __torch_dispatch__，处理 Torch 分发调用
                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 如果 func 的重载包是 torch.ops.prim.device，则返回元数据设备
                    if func.overloadpacket == torch.ops.prim.device:
                        return torch.device("meta")
                    return NotImplemented

            # 定义 ExampleTensor3 类，继承自 torch.Tensor
            class ExampleTensor3(torch.Tensor):
                # 重载静态方法 __new__，用于创建实例
                @staticmethod
                def __new__(cls, data, wrapper):
                    # 调用辅助方法 subclass_helper 进行实例化
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_device=True
                    )

                # 类方法 __torch_dispatch__，处理 Torch 分发调用
                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 如果 func 的重载包是 torch.ops.prim.device，则返回元数据设备
                    if func.overloadpacket == torch.ops.prim.device:
                        return torch.device("meta")
                    return NotImplemented

            # 错误消息，用于断言验证多分发是否失败于 'torch.ops.prim.device'
            err_msg = "Multiple dispatch failed for 'torch.ops.prim.device'"
            # 使用断言检测是否引发 TypeError，并验证错误消息是否匹配
            with self.assertRaisesRegex(TypeError, err_msg):
                # 创建 ExampleTensor1 实例并调用 device 方法，预期引发异常
                e = ExampleTensor1(torch.randn(3, 3), use_wrapper_subclass)
                e.device()

            # 创建一个 tensor 实例
            ten = torch.rand([1])
            # 创建 ExampleTensor2 实例，并验证其设备类型为 "meta"
            e = ExampleTensor2(torch.randn(3, 3, device="cpu"), use_wrapper_subclass)
            self.assertEqual(e.device.type, "meta")
            # 使用 type_as 方法将 ten 的类型调整为 e 的类型，并验证设备类型为 "meta"
            self.assertEqual(ten.type_as(e).device.type, "meta")

            # 创建 ExampleTensor3 实例，并验证其设备类型为 "meta"
            e = ExampleTensor3(torch.randn(3, 3, device="cpu"), use_wrapper_subclass)
            self.assertEqual(e.device.type, "meta")
            # 使用 type_as 方法将 ten 的类型调整为 e 的类型，并验证设备类型为 "meta"
            self.assertEqual(ten.type_as(e).device.type, "meta")
    # 定义一个测试函数，测试处理维度的慢路径
    def test_dim_slowpath(self):
        # 创建一个3x3的随机张量
        data = torch.randn(3, 3)

        # 针对是否使用包装子类进行循环测试
        for use_wrapper_subclass in [True, False]:

            # 定义一个未实现维度操作的张量子类
            class DimNotImplementedTensor(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    # 调用辅助函数创建子类实例
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 返回未实现，表示不支持此操作
                    return NotImplemented

            # 定义一个已实现维度操作的张量子类
            class DimImplementedTensor(torch.Tensor):
                @staticmethod
                def __new__(cls, data, wrapper):
                    # 调用辅助函数创建子类实例
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                    )

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 如果是获取维度的操作，则返回数据张量的维度
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    # 否则返回未实现，表示不支持此操作
                    return NotImplemented

            # 定义错误信息
            err_msg = "Multiple dispatch failed for 'torch.ops.aten.dim'"

            # 创建未实现维度操作的张量实例
            e = DimNotImplementedTensor(torch.randn(3, 3), use_wrapper_subclass)

            # 断言调用维度操作会引发类型错误，并包含指定错误信息
            with self.assertRaisesRegex(TypeError, err_msg):
                e.dim()

            # 创建已实现维度操作的张量实例
            t = DimImplementedTensor(torch.randn(3, 3), use_wrapper_subclass)

            # 断言调用维度操作返回的维度为2
            self.assertEqual(t.dim(), 2)

    # 测试可能存在的元组错误
    def test_maybe_tuple_bug(self):
        # 定义一个简单的张量子类，不实现任何新功能
        class T(torch.Tensor):
            @classmethod
            def __torch_function__(cls, *args, **kwargs):
                pass

        # 创建一个长度为3的随机张量
        a = torch.rand(3)

        # 访问张量的两个索引位置，使用未实现任何新功能的张量子类
        a[[T(), T()]]

    # 测试标准的张量是否不是子类
    def test_standard_is_not_subclass(self):
        # 断言标准空张量不是张量子类的子类
        self.assertFalse(torch._C._dispatch_isTensorSubclassLike(torch.empty(0)))
    # 定义一个名为 test_sym_sizes_strides_slow_path 的测试方法
    def test_sym_sizes_strides_slow_path(self):
        # 定义一个名为 TestTensor 的内部类，继承自 torch.Tensor
        class TestTensor(torch.Tensor):
            # 静态方法，用于创建一个新的 TestTensor 实例
            @staticmethod
            def __new__(cls, *args, **kwargs):
                # 调用 torch.Tensor._make_wrapper_subclass 方法创建子类实例
                r = torch.Tensor._make_wrapper_subclass(
                    cls, (0,), dispatch_sizes_strides_policy="sizes"
                )
                return r

            # 类方法，处理 torch 派发时的特殊方法
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # 如果 func 是 torch.ops.aten.sym_size.default 或 torch.ops.aten.sym_stride.default
                if func in (
                    torch.ops.aten.sym_size.default,
                    torch.ops.aten.sym_stride.default,
                ):
                    # 导入所需的模块和类
                    from torch._dynamo.source import ConstantSource
                    from torch.fx.experimental.symbolic_shapes import (
                        DimDynamic,
                        ShapeEnv,
                    )

                    # 创建一个符号形状的环境
                    shape_env = ShapeEnv()
                    # 创建一个符号整数节点
                    si = shape_env.create_symintnode(
                        shape_env.create_symbol(
                            123,
                            source=ConstantSource("abc"),
                            dynamic_dim=DimDynamic.DUCK,
                            constraint_dim=None,
                        ),
                        hint=123,
                    )
                    # 返回一个包含符号整数节点的元组
                    return (si,)

        # 创建 TestTensor 的实例 t
        t = TestTensor()
        # 获取 t 的大小（size）的第一个元素
        si = t.size()[0]
        # 断言 si 的类型是 torch.SymInt
        self.assertIsInstance(si, torch.SymInt)
        # 获取 t 的步幅（stride）的第一个元素
        si = t.stride()[0]
        # 断言 si 的类型是 torch.SymInt
        self.assertIsInstance(si, torch.SymInt)
    # 定义一个测试方法，用于测试不同情况下的张量子类
    def test_strides_slow_path(self):
        # 遍历是否使用包装子类的布尔值列表
        for use_wrapper_subclass in [True, False]:

            # 定义一个未实现 strides 方法的张量子类
            class StridesNotImplemented(torch.Tensor):
                @staticmethod
                # 重写 __new__ 方法，调用辅助方法创建子类实例
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                # 实现 __torch_dispatch__ 方法，处理函数调度逻辑
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    return NotImplemented

            # 定义一个返回自定义 strides 结果的张量子类
            class StridesCustomReturn(torch.Tensor):
                @staticmethod
                # 重写 __new__ 方法，调用辅助方法创建子类实例
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                # 实现 __torch_dispatch__ 方法，处理函数调度逻辑
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 如果调用的是 torch.ops.aten.sym_stride.default 函数，则返回 (4, 2)
                    if func == torch.ops.aten.sym_stride.default:
                        return (4, 2)
                    return NotImplemented

            # 定义一个返回默认 strides 结果的张量子类
            class StridesDefaultReturn(torch.Tensor):
                @staticmethod
                # 重写 __new__ 方法，调用辅助方法创建子类实例
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="strides"
                    )

                @classmethod
                # 实现 __torch_dispatch__ 方法，处理函数调度逻辑
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 如果调用的是 torch.ops.aten.sym_stride.default 函数，则返回 None
                    if func == torch.ops.aten.sym_stride.default:
                        return None
                    return NotImplemented

            # 定义错误消息字符串，用于断言多重调度失败
            err_msg = "Multiple dispatch failed for 'torch.ops.aten.sym_stride'"
            
            # 创建一个未实现 strides 方法的实例，并断言会抛出 TypeError 异常
            e = StridesNotImplemented(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.stride()

            # 创建一个自定义返回 strides 结果的实例，并断言其返回值为 (4, 2)
            e = StridesCustomReturn(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.stride(), (4, 2))

            # 创建一个默认返回 strides 结果的实例，并断言其返回值为 (2, 1)
            e = StridesDefaultReturn(torch.randn(6, 2), use_wrapper_subclass)
            self.assertEqual(e.stride(), (2, 1))
    # 定义测试函数 test_sizes_slow_path，用于测试不同情况下的多分派功能
    def test_sizes_slow_path(self):
        # 对于每种使用子类包装器的情况进行迭代测试
        for use_wrapper_subclass in [True, False]:
            # 创建一个形状为 (6, 2) 的随机张量数据
            data = torch.randn(6, 2)

            # 定义未实现 sym_size 方法的 SizesNotImplemented 子类
            class SizesNotImplemented(torch.Tensor):
                @staticmethod
                # 定义 __new__ 静态方法，用于创建新对象
                def __new__(cls, data, wrapper):
                    # 调用 helper 函数创建子类对象，设置 dispatch_sizes_strides_policy 为 "sizes"
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                    )

                @classmethod
                # 定义 __torch_dispatch__ 类方法，处理多分派调度
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 如果调用的函数是 torch.ops.aten.dim，则返回张量的维度
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    # 否则返回未实现
                    return NotImplemented

            # 定义实现了自定义 sym_size 方法返回的 SizesCustomReturn 子类
            class SizesCustomReturn(torch.Tensor):
                @staticmethod
                # 定义 __new__ 静态方法，用于创建新对象
                def __new__(cls, data, wrapper):
                    # 调用 helper 函数创建子类对象，设置 dispatch_sizes_strides_policy 为 "sizes"
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                    )

                @classmethod
                # 定义 __torch_dispatch__ 类方法，处理多分派调度
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 如果调用的函数是 torch.ops.aten.dim，则返回张量的维度
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    # 如果调用的函数是 torch.ops.aten.sym_size，则返回固定的大小 (5, 3)
                    if func.overloadpacket == torch.ops.aten.sym_size:
                        return (5, 3)
                    # 否则返回未实现
                    return NotImplemented

            # 定义默认情况下 sym_size 方法返回 None 的 SizesDefaultReturn 子类
            class SizesDefaultReturn(torch.Tensor):
                @staticmethod
                # 定义 __new__ 静态方法，用于创建新对象
                def __new__(cls, data, wrapper):
                    # 调用 helper 函数创建子类对象，设置 dispatch_sizes_strides_policy 为 "sizes"
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_sizes_strides_policy="sizes"
                    )

                @classmethod
                # 定义 __torch_dispatch__ 类方法，处理多分派调度
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 如果调用的函数是 torch.ops.aten.dim，则返回张量的维度
                    if func.overloadpacket == torch.ops.aten.dim:
                        return data.dim()
                    # 如果调用的函数是 torch.ops.aten.sym_size，则返回 None
                    if func.overloadpacket == torch.ops.aten.sym_size:
                        return None
                    # 否则返回未实现
                    return NotImplemented

            # 错误消息，用于断言测试失败时的异常信息
            err_msg = "Multiple dispatch failed for 'torch.ops.aten.sym_size'"

            # 创建 SizesNotImplemented 实例 e，使用断言检测是否抛出 TypeError 异常
            e = SizesNotImplemented(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.size()

            # 创建 SizesCustomReturn 实例 e，使用断言检测其 size 方法返回是否为 (5, 3)
            e = SizesCustomReturn(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.size(), (5, 3))

            # 创建 SizesDefaultReturn 实例 e，使用断言检测其 size 方法返回是否为 (4, 2)
            e = SizesDefaultReturn(torch.randn(4, 2), use_wrapper_subclass)
            self.assertEqual(e.size(), (4, 2))
        def test_custom_size_policy_dynamic_shapes(self):
            # 创建一个形状为 (6, 2) 的随机张量数据
            data = torch.randn(6, 2)

            # 定义一个继承自 torch.Tensor 的自定义张量类 CustomSizeDynamicShapesTensor
            class CustomSizeDynamicShapesTensor(torch.Tensor):
                @staticmethod
                def __new__(cls, inner):
                    # 使用 _make_wrapper_subclass 方法创建子类的实例
                    return torch.Tensor._make_wrapper_subclass(
                        # 传递给 _make_wrapper_subclass 的参数，包括形状、步幅、dtype、布局和设备等
                        cls,
                        inner.size(),  # 获取张量 inner 的形状作为参数传递
                        inner.stride(),  # 获取张量 inner 的步幅作为参数传递
                        None,
                        None,
                        inner.dtype,
                        inner.layout,
                        inner.device,
                        False,
                        inner.requires_grad,
                        "sizes",  # 指定额外的参数 "sizes"
                    )

                def __init__(self, inner):
                    self.inner = inner  # 初始化内部张量数据

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 处理 torch 函数调用的分发逻辑
                    if func == torch.ops.aten.sym_size.default:
                        return args[0].inner.shape  # 返回内部张量的形状
                    if func == torch.ops.aten.sym_stride.default:
                        return args[0].inner.shape  # 返回内部张量的步幅
                    return NotImplemented  # 如果不匹配预期的函数，返回未实现提示

            x = torch.ones(2, 2)  # 创建一个形状为 (2, 2) 的全 1 张量 x

            def trace_fn(x):
                x_wrapper = CustomSizeDynamicShapesTensor(x)  # 使用自定义张量类包装张量 x
                return x_wrapper.size(), x_wrapper.stride()  # 返回包装后张量的形状和步幅信息

            fx_g = make_fx(trace_fn, tracing_mode="symbolic")(x)  # 使用 make_fx 进行函数追踪
            self.assertExpectedInline(
                fx_g.code.strip(),
                """\
    def forward(self, x_1):
        # 调用 torch 库的自定义操作，获取输入张量 x_1 的第一个维度大小
        sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
        # 调用 torch 库的自定义操作，获取输入张量 x_1 的第二个维度大小，并清空 x_1 引用
        sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1);  x_1 = None
        # 返回一个包含两个相同元组的元组，每个元组包含输入张量 x_1 的两个维度大小
        return ((sym_size_int, sym_size_int_1), (sym_size_int, sym_size_int_1))
    def test_layout_slow_path(self):
        # 循环测试是否使用包装子类
        for use_wrapper_subclass in [True, False]:
            # 创建一个 6x2 的张量数据
            data = torch.randn(6, 2)

            # 定义一个未实现布局的张量子类
            class LayoutNotImplemented(torch.Tensor):
                @staticmethod
                # 重载 __new__ 方法以支持多态分发
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_layout=True
                    )

                @classmethod
                # 实现多态分发方法 __torch_dispatch__
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 如果调用的函数是 torch.ops.prim.layout，则返回未实现
                    return NotImplemented

            # 定义一个自定义返回的布局张量子类
            class LayoutCustomReturn(torch.Tensor):
                @staticmethod
                # 重载 __new__ 方法以支持多态分发
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_layout=True
                    )

                @classmethod
                # 实现多态分发方法 __torch_dispatch__
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 如果调用的函数是 torch.ops.prim.layout，则返回稀疏 CSR 张量
                    if func.overloadpacket == torch.ops.prim.layout:
                        return torch.sparse_csr
                    return NotImplemented

            # 定义一个默认返回的布局张量子类
            class LayoutDefaultReturn(torch.Tensor):
                @staticmethod
                # 重载 __new__ 方法以支持多态分发
                def __new__(cls, data, wrapper):
                    return TestPythonDispatch.subclass_helper(
                        cls, data, wrapper, dispatch_layout=True
                    )

                @classmethod
                # 实现多态分发方法 __torch_dispatch__
                def __torch_dispatch__(cls, func, types, args, kwargs):
                    # 如果调用的函数是 torch.ops.prim.layout，则返回数据的布局信息
                    if func.overloadpacket == torch.ops.prim.layout:
                        return data.layout
                    return NotImplemented

            # 错误信息，用于断言多重分发失败于 'torch.ops.prim.layout'
            err_msg = "Multiple dispatch failed for 'torch.ops.prim.layout'"
            
            # 测试 LayoutNotImplemented 类的 layout 方法，预期抛出 TypeError 异常
            e = LayoutNotImplemented(torch.randn(3, 3), use_wrapper_subclass)
            with self.assertRaisesRegex(TypeError, err_msg):
                e.layout

            # 测试 LayoutCustomReturn 类的 layout 方法，预期返回稀疏 CSR 张量
            e = LayoutCustomReturn(torch.randn(3, 3), use_wrapper_subclass)
            self.assertEqual(e.layout, torch.sparse_csr)

            # 测试 LayoutDefaultReturn 类的 layout 方法，预期返回普通张量的布局信息
            e = LayoutDefaultReturn(torch.randn(4, 2), use_wrapper_subclass)
            self.assertEqual(e.layout, torch.strided)
class TestPythonDispatcher(TestCase):
    def test_basic(self):
        # 创建一个形状为 (2,) 的张量 x，允许计算梯度
        x = torch.randn(2, requires_grad=True)
        # 启用 Python 调度器的 C++ 实现
        r = torch._C._EnablePythonDispatcher()
        # 对张量 x 执行加法操作并返回结果

    def test_lstsq(self):
        # 创建形状为 (4, 3) 的随机张量 a 和 b
        a = torch.randn(4, 3)
        b = torch.rand(4, 3)
        # 计算使用最小二乘法求解 a @ x = b 的期望解的形状
        expected_shape = torch.linalg.lstsq(a, b).solution.shape
        # 启用 Python 调度器的 C++ 实现
        r = torch._C._EnablePythonDispatcher()
        # 计算使用最小二乘法求解 a @ x = b 的解的形状
        python_disp_shape = torch.linalg.lstsq(a, b).solution.shape
        # 断言期望解的形状与使用 Python 调度器后的解的形状相等

class TestWrapperSubclassAliasing(TestCase):
    def _test_wrapper_subclass_aliasing(self, op, args, kwargs):
        # 将输入张量 t 转换为 TwoTensor 类的实例，其中包含 t 和 t 克隆的张量
        def to_subclass(t: torch.Tensor):
            return TwoTensor(t, t.clone())

        # 执行操作 op，使用给定的参数 args 和 kwargs，并获取参考结果
        result_ref = op(*args, **kwargs)

        # 将输入参数 args 转换为 WrapperSubclass 实例的树形结构
        args_subclass = pytree.tree_map_only(torch.Tensor, to_subclass, args)
        # 将输入参数 kwargs 转换为 WrapperSubclass 实例的树形结构
        kwargs_subclass = pytree.tree_map_only(torch.Tensor, to_subclass, kwargs)

        # 执行操作 op，使用转换后的参数 args_subclass 和 kwargs_subclass，并获取测试结果
        result_test = op(*args_subclass, **kwargs_subclass)

        # 从输入参数 args 和 kwargs 中提取所有的张量，并扁平化成列表
        args_ref_flat = pytree.arg_tree_leaves(*args, **kwargs)
        # 从转换后的参数 args_subclass 和 kwargs_subclass 中提取所有的张量，并扁平化成列表
        args_test_flat = pytree.tree_leaves((args_subclass, kwargs_subclass))

        # 从参考结果 result_ref 中提取所有的张量，并扁平化成列表
        result_ref_flat = pytree.tree_leaves(result_ref)
        # 从测试结果 result_test 中提取所有的张量，并扁平化成列表
        result_test_flat = pytree.tree_leaves(result_test)

        # 从所有扁平化的参考结果中筛选出张量
        args_ref_flat_tensors = [
            x for x in args_ref_flat if isinstance(x, torch.Tensor)
        ]
        # 从所有扁平化的测试结果中筛选出张量
        args_test_flat_tensors = [
            x for x in args_test_flat if isinstance(x, torch.Tensor)
        ]
        # 从所有扁平化的参考结果中筛选出张量
        result_ref_flat_tensors = [
            x for x in result_ref_flat if isinstance(x, torch.Tensor)
        ]
        # 从所有扁平化的测试结果中筛选出张量
        result_test_flat_tensors = [
            x for x in result_test_flat if isinstance(x, torch.Tensor)
        ]

        # 检查每对参考结果和测试结果的张量
        for o_ref, o_test in zip(result_ref_flat_tensors, result_test_flat_tensors):
            for a_ref, a_test in zip(args_ref_flat_tensors, args_test_flat_tensors):
                # 检查测试结果张量是否与参考结果张量相同对象
                out_is_inpt = o_ref is a_ref
                if out_is_inpt:
                    self.assertTrue(o_test is a_test)

                # 检查测试结果张量是否与参考结果张量共享相同的存储引用
                out_aliases_inpt = StorageWeakRef(
                    o_ref.untyped_storage()
                ) == StorageWeakRef(a_ref.untyped_storage())
                if out_aliases_inpt:
                    self.assertTrue(
                        StorageWeakRef(o_test.untyped_storage())
                        == StorageWeakRef(a_test.untyped_storage())
                    )
                else:
                    self.assertFalse(
                        StorageWeakRef(o_test.untyped_storage())
                        == StorageWeakRef(a_test.untyped_storage())
                    )

    # This tests the correctness of `torch.utils._python_dispatch.return_and_correct_aliasing`,
    # a util for wrapper subclasses to promise correct aliasing behavior.
    # It's probably overkill to test every OpInfo,
    # so I picked a sampling of ops with representative schemas.
    # 测试 `torch.utils._python_dispatch.return_and_correct_aliasing` 的正确性，
    # 用于确保包装子类的正确别名行为。
    # 可能不需要测试每个 OpInfo，所以选择了一些具有代表性的操作模式。
    # 定义装饰器函数 ops，用于测试特定操作
    @ops(
        [
            op  # 遍历操作数据库中的操作
            for op in op_db
            if op.name
            in [
                "mul",  # 非原地操作 (out-of-place)
                "cat",  # 非原地操作，输入为 TensorList (TensorList input)
                "index",  # 非原地操作，可选的 TensorList 输入 (Optional TensorList input)
                "mul_",  # 原地操作 (inplace)
                "view",  # 视图操作 (view)
                "t_",  # 原地视图操作 (inplace-view)
                "split",  # 视图操作，多返回值 (view, multi-return)
                "native_batch_norm",  # 可变操作，返回输出并改变一些输入 (mutable op, returns outputs and mutates some inputs)
            ]
        ],
        allowed_dtypes=(torch.float,),  # 允许的数据类型为 torch.float
    )
    # 定义测试函数 test_wrapper_subclass_aliasing，用于测试操作的子类别别名
    def test_wrapper_subclass_aliasing(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)  # 根据设备和数据类型生成操作的样本输入
        sample = first_sample(self, samples)  # 获取第一个样本
        args = (sample.input, *sample.args)  # 构造参数元组
        kwargs = sample.kwargs  # 获取关键字参数
        self._test_wrapper_subclass_aliasing(op, args, kwargs)  # 调用测试子类别别名的私有方法
    
    # 定义装饰器函数 ops，用于测试自定义操作数据库中的操作
    @ops(custom_op_db, allowed_dtypes=(torch.float,))
    # 定义测试函数 test_wrapper_subclass_aliasing_custom，用于测试自定义操作的子类别别名
    def test_wrapper_subclass_aliasing_custom(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)  # 根据设备和数据类型生成操作的样本输入
        sample = first_sample(self, samples)  # 获取第一个样本
        args = (sample.input, *sample.args)  # 构造参数元组
        kwargs = sample.kwargs  # 获取关键字参数
        self._test_wrapper_subclass_aliasing(op, args, kwargs)  # 调用测试子类别别名的私有方法
    
    # 定义测试函数 test_wrapper_subclass_aliasing_conv2d，用于测试 conv2d 操作的子类别别名
    def test_wrapper_subclass_aliasing_conv2d(self, device):
        args = (torch.randn(4, 4, 4, 4), torch.randn(4, 4, 4, 4))  # 构造输入参数
        kwargs = {}
        # conv2d 操作默认参数 'int[2] strides=0'，在 torchscript 中展开为 'int[2] strides=[0, 0]'
        # 确保 _return_and_correct_aliasing 能处理这种情况
        # （使用 inference_mode 确保 conv2d 不会分解并且直接进入 torch_dispatch）
        with torch.inference_mode():
            self._test_wrapper_subclass_aliasing(
                torch.ops.aten.conv2d.default, args, kwargs
            )  # 调用测试子类别别名的私有方法
    
    # 定义测试函数 test_wrapper_subclass_aliasing_out_op，用于测试带有可变张量关键字参数的操作
    def test_wrapper_subclass_aliasing_out_op(self, device):
        args = (torch.ones(4), torch.ones(4))  # 构造输入参数
        kwargs = {"out": torch.empty(4)}  # 设置关键字参数 out 为一个空张量
        self._test_wrapper_subclass_aliasing(torch.ops.aten.add.out, args, kwargs)  # 调用测试子类别别名的私有方法
# 实例化设备类型测试，使用给定的测试包装子类别别名和全局变量
instantiate_device_type_tests(TestWrapperSubclassAliasing, globals())

# 如果当前脚本作为主程序执行，则运行测试
if __name__ == "__main__":
    run_tests()
```