# `.\pytorch\test\export\test_passes.py`

```
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_functionalization_with_native_python_assertion)
"""

# 导入所需的模块和库
import math  # 导入数学模块
import operator  # 导入运算符模块
import unittest  # 导入单元测试模块
from re import escape  # 导入正则表达式中的转义函数
from typing import List, Set  # 导入类型提示相关的 List 和 Set 类型

import torch  # 导入 PyTorch 库

# 导入 Functorch 库中的条件语句控制模块
from functorch.experimental.control_flow import cond

# 导入 Torch 的内部评估框架中的相关函数和类
from torch._dynamo.eval_frame import is_dynamo_supported

# 导入 Torch 的非严格模式下实用的工具函数
from torch._export.non_strict_utils import (
    _fakify_script_objects,
    _gather_constant_attrs,
)

# 导入 Torch 的导出基类，已过时，不建议使用
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse

# 导入替换 set_grad_enabled 调用为 hop 的相关函数
from torch._export.passes.replace_set_grad_with_hop_pass import (
    _is_set_grad_enabled_node,
    _is_set_grad_enabled_sub_mod,
)

# 导入替换视图操作为视图复制操作的相关函数和类
from torch._export.passes.replace_view_ops_with_view_copy_ops_pass import (
    get_view_copy_of_view_op,
    is_view_op,
    ReplaceViewOpsWithViewCopyOpsPass,
)

# 导入 Torch 的导出工具函数
from torch._export.utils import (
    node_inline_,
    nodes_count,
    nodes_filter,
    nodes_map,
    sequential_split,
)

# 导入 Torch 的自动函数化模块
from torch._higher_order_ops.auto_functionalize import auto_functionalized

# 导入 Torch 的虚拟张量模式
from torch._subclasses.fake_tensor import FakeTensorMode

# 导入 Torch 的导出模块和函数
from torch.export import export

# 导入移除自动函数化的相关函数
from torch.export._remove_auto_functionalized_pass import (
    unsafe_remove_auto_functionalized_pass,
)

# 导入移除效果标记的函数
from torch.export._remove_effect_tokens_pass import _remove_effect_tokens

# 导入 Torch 的符号形状实验模块
from torch.fx.experimental.symbolic_shapes import ShapeEnv

# 导入 Torch 的图分区模块
from torch.fx.passes.infra.partitioner import Partition

# 导入 Torch 的操作支持模块
from torch.fx.passes.operator_support import OperatorSupport

# 导入 Torch 的库实现模块
from torch.library import _scoped_library, impl

# 导入 Torch 的内部测试工具函数
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)

# 导入 Torch 的 Torchbind 实现模块
from torch.testing._internal.torchbind_impls import init_torchbind_implementations

# 导入 Torch 的 Pytree 模块
from torch.utils import _pytree as pytree


def count_call_function(graph: torch.fx.Graph, target: torch.ops.OpOverload) -> int:
    # 初始化计数器
    count = 0
    # 遍历图中的每个节点
    for node in graph.nodes:
        # 检查节点是否为函数调用节点且目标与给定目标相同
        if node.op == "call_function" and node.target == target:
            # 如果是，则增加计数
            count += 1
    # 返回计数结果
    return count


class _AddOperatorSupport(OperatorSupport):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # 判断节点是否为调用函数操作，并且目标是加法运算符
        return node.op == "call_function" and node.target in {operator.add}


class _AtenAddOperatorSupport(OperatorSupport):
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # 判断节点是否为调用函数操作，并且目标是 Torch 的 Aten 模块中的张量加法操作
        return node.op == "call_function" and node.target in {torch.ops.aten.add.Tensor}


def _to_partition_names(partitions: List[Partition]) -> List[Set[str]]:
    # 返回每个分区中节点名称的集合列表
    return [{n.name for n in p.nodes} for p in partitions]


def _get_output_names(gm: torch.fx.GraphModule) -> List[str]:
    # 查找输出节点
    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    # 提取输出节点的参数
    args = pytree.tree_leaves(output_node.args)
    # 返回参数的字符串表示形式列表
    return [str(arg) for arg in args]


class ModelsWithScriptObjectAttr:
    # 定义一个简单的神经网络模块类 Simple
    class Simple(torch.nn.Module):
        # 初始化方法
        def __init__(self):
            # 调用父类的初始化方法
            super().__init__()
            # 创建一个 TorchScriptTesting 库中的 _Foo 对象实例，并将其赋值给实例属性 attr
            self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

    # 定义一个包含属性的神经网络模块类 SimpleWithAttrInContainer
    class SimpleWithAttrInContainer(torch.nn.Module):
        # 初始化方法
        def __init__(self):
            # 调用父类的初始化方法
            super().__init__()
            # 创建一个 TorchScriptTesting 库中的 _Foo 对象实例，并将其赋值给实例属性 attr
            self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)
            # 创建一个包含多个元素的列表属性 pytree_attr2
            self.pytree_attr2 = [
                torch.classes._TorchScriptTesting._Foo(1, 2),  # 创建一个 _Foo 对象实例
                {torch.classes._TorchScriptTesting._Foo(3, 4)},  # 创建一个包含 _Foo 对象的集合
                {"foo": torch.classes._TorchScriptTesting._Foo(5, 6)},  # 创建一个包含 _Foo 对象的字典
            ]

    # 定义一个嵌套属性的神经网络模块类 NestedWithAttrInContainer
    class NestedWithAttrInContainer(torch.nn.Module):
        # 初始化方法
        def __init__(self):
            # 调用父类的初始化方法
            super().__init__()
            # 创建一个 TorchScriptTesting 库中的 _Foo 对象实例，并将其赋值给实例属性 attr
            self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)
            # 创建一个包含多个元素的列表属性 pytree_attr2
            self.pytree_attr2 = [
                torch.classes._TorchScriptTesting._Foo(1, 2),  # 创建一个 _Foo 对象实例
                {torch.classes._TorchScriptTesting._Foo(3, 4)},  # 创建一个包含 _Foo 对象的集合
                {"foo": torch.classes._TorchScriptTesting._Foo(5, 6)},  # 创建一个包含 _Foo 对象的字典
            ]
            # 创建一个 Simple 类的实例，并将其赋值给实例属性 sub_mod
            self.sub_mod = ModelsWithScriptObjectAttr.Simple()
            # 创建一个 SimpleWithAttrInContainer 类的实例，并将其赋值给实例属性 sub_mod2
            self.sub_mod2 = ModelsWithScriptObjectAttr.SimpleWithAttrInContainer()

    # 定义一个更深层次嵌套属性的神经网络模块类 MoreNestedWithAttrInContainer
    class MoreNestedWithAttrInContainer(torch.nn.Module):
        # 初始化方法
        def __init__(self):
            # 调用父类的初始化方法
            super().__init__()
            # 创建一个 TorchScriptTesting 库中的 _Foo 对象实例，并将其赋值给实例属性 attr
            self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)
            # 创建一个包含多个元素的列表属性 pytree_attr2
            self.pytree_attr2 = [
                torch.classes._TorchScriptTesting._Foo(1, 2),  # 创建一个 _Foo 对象实例
                {torch.classes._TorchScriptTesting._Foo(3, 4)},  # 创建一个包含 _Foo 对象的集合
                {"foo": torch.classes._TorchScriptTesting._Foo(5, 6)},  # 创建一个包含 _Foo 对象的字典
            ]
            # 创建一个 Simple 类的实例，并将其赋值给实例属性 sub_mod
            self.sub_mod = ModelsWithScriptObjectAttr.Simple()
            # 创建一个 NestedWithAttrInContainer 类的实例，并将其赋值给实例属性 sub_mod2
            self.sub_mod2 = ModelsWithScriptObjectAttr.NestedWithAttrInContainer()
# 定义一个函数，用于测试设置梯度开启状态的场景
def _set_grad_enabled_tests():
    # 导入 _export 函数
    from torch.export._trace import _export

    # 定义一个简单的 Module 类，用于测试直接设置梯度开启和关闭
    class SetGradOp(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            # 设置梯度开启
            torch._C._set_grad_enabled(True)
            c = x.sin().sum()
            # 设置梯度关闭
            torch._C._set_grad_enabled(False)
            d = c + 1
            # 再次设置梯度开启
            torch._C._set_grad_enabled(True)
            e = d - 1
            return d, e

    # 定义一个使用上下文管理器的 Module 类，测试使用 torch.enable_grad() 和 torch.no_grad() 的效果
    class SetGradCtxManager(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            with torch.enable_grad():
                c = x.sin().sum()
            with torch.no_grad():
                d = c + 1
            with torch.enable_grad():
                e = d - 1
            return d, e

    # 定义一个复杂的 Module 类，测试多个依赖的情况下的梯度设置
    class SetGradCtxManagerMultiDep(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            with torch.enable_grad():
                c1 = x.sin().sum()
                c2 = x.cos().sum()
            with torch.no_grad():
                d1 = c1 + 1
                d2 = c2 + 1
            with torch.enable_grad():
                e1 = d1 - 1
                e2 = d2 - 1
            return d1, d2, e1, e2

    # 生成一个随机输入张量
    x = torch.randn(2, 2)

    # 定义一个函数，用于获取预分派模块，并测试梯度开启状态
    def _get_predispatch_module(mod, args, ambient_grad_enabled=True):
        with torch.set_grad_enabled(ambient_grad_enabled):
            return _export(mod, args, pre_dispatch=True).module()

    # 返回一个字典，包含不同设置下的模块实例及其输入
    return {
        "ctx_manager": (_get_predispatch_module(SetGradCtxManager(), (x,)), (x,)),
        "ctx_manager_under_no_grad": (
            _get_predispatch_module(SetGradCtxManager(), (x,), False),
            (x,),
        ),
        "ctx_manager_multi_dep": (
            _get_predispatch_module(SetGradCtxManagerMultiDep(), (x,)),
            (x,),
        ),
        "ctx_manager_multi_dep_no_grad": (
            _get_predispatch_module(SetGradCtxManagerMultiDep(), (x,), False),
            (x,),
        ),
        "op": (_get_predispatch_module(SetGradOp(), (x,)), (x,)),
        "op_under_no_grad": (_get_predispatch_module(SetGradOp(), (x,), False), (x,)),
    }
    # 定义一个名为 _insert_dilimiter_nodes 的函数，用于在图模块中插入分隔符节点
    def _insert_dilimiter_nodes(gm: torch.fx.GraphModule, step: int = 1):
        # 初始化插入位置的列表
        insert_locs = []
        # 遍历图中所有操作为 "call_function" 的节点
        for i, node in enumerate(
            nodes_filter(gm.graph.nodes, lambda n: n.op == "call_function")
        ):
            # 如果节点索引能被 step 整除，则将该节点添加到插入位置列表中
            if i % step == 0:
                insert_locs.append(node)

        # 遍历插入位置列表中的节点
        for i, node in enumerate(insert_locs):
            # 在节点之前插入一个新节点
            with gm.graph.inserting_before(node):
                # 根据条件调用 torch._C._set_grad_enabled 函数，使得每两个节点之间的调用使得梯度开启或关闭
                gm.graph.call_function(
                    torch._C._set_grad_enabled, (True if i % 2 == 0 else False,), {}
                )
        
        # 返回修改后的图模块
        return gm

    # 创建一个 2x2 的随机张量 x
    x = torch.randn(2, 2)
    # 调用 _get_predispatch_module 函数，获取处理前模块 Simple 的实例 simple，并传入张量 x
    simple = _get_predispatch_module(Simple(), (x,))
    # 再次调用 _get_predispatch_module 函数，获取处理前模块 Simple 的另一个实例 simple1，并传入张量 x
    simple1 = _get_predispatch_module(Simple(), (x,))
    # 调用 _get_predispatch_module 函数，获取处理前模块 MultiDep 的实例 multi_dep，并传入张量 x 和 x.sin() 的结果
    multi_dep = _get_predispatch_module(MultiDep(), (x, x.sin()))
    # 再次调用 _get_predispatch_module 函数，获取处理前模块 MultiDep 的另一个实例 multi_dep1，并传入张量 x 和 x.sin() 的结果
    multi_dep1 = _get_predispatch_module(MultiDep(), (x, x.sin()))
    
    # 返回一个字典，包含不同模块经过 _insert_dilimiter_nodes 处理后的结果及输入参数
    return {
        "simple_step1": (_insert_dilimiter_nodes(simple1, 1), (x,)),
        "simple_step2": (_insert_dilimiter_nodes(simple, 2), (x,)),
        "multi_dep_step2": (_insert_dilimiter_nodes(multi_dep, 2), (x, x.sin())),
        "multi_dep_step3": (_insert_dilimiter_nodes(multi_dep1, 3), (x, x.sin())),
    }
# 根据特定条件跳过 TorchDynamo 测试，如果不支持 Dynamo 则跳过整个测试类
@skipIfTorchDynamo("recursively running dynamo on export is unlikely")
@unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
class TestPasses(TestCase):
    # 在每个测试方法运行前执行的设置操作
    def setUp(self):
        super().setUp()
        # 初始化内联测试分割序列和梯度设置测试集合
        self.SEQUENTIAL_SPLIT_INLINE_TESTS = _sequential_split_inline_tests()
        self.SET_GRAD_ENABLED_TESTS = _set_grad_enabled_tests()

        # 初始化 TorchBind 实现
        init_torchbind_implementations()

    # 在每个测试方法运行后执行的清理操作
    def tearDown(self):
        # 清空内联测试分割序列和梯度设置测试集合
        self.SEQUENTIAL_SPLIT_INLINE_TESTS.clear()
        self.SET_GRAD_ENABLED_TESTS.clear()
        super().tearDown()

    # 测试单维度运行时断言
    def test_runtime_assert_one_dim(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.cos()

        x = torch.zeros(2, 2, 3)

        # 定义导出时的动态维度约束
        dim1_x = torch.export.Dim("dim1_x", min=2, max=6)
        ep = torch.export.export(M(), (x,), dynamic_shapes={"x": {1: dim1_x}})

        # 断言捕获运行时错误，并检查错误消息格式化
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[0].shape[1] to be <= 6, but got 7"),
        ):
            ep.module()(torch.zeros(2, 7, 3))

        # 断言模型预测结果与预期结果一致
        self.assertEqual(
            ep.module()(torch.ones(2, 4, 3)), M().forward(torch.ones(2, 4, 3))
        )

    # 测试多维度运行时断言
    def test_runtime_assert_multiple_dims(self) -> None:
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x.cos().sum() + y.sin().sum()

        x = torch.zeros(4, 2, 3)
        y = torch.zeros(5, 5, 5)

        # 定义导出时的动态维度约束
        dim1_x = torch.export.Dim("dim1_x", min=2, max=6)
        dim0_x, dim0_y = torch.export.dims("dim0_x", "dim0_y", min=3)

        ep = torch.export.export(
            M(), (x, y), dynamic_shapes={"x": {0: dim0_x, 1: dim1_x}, "y": {0: dim0_y}}
        )

        # 断言捕获运行时错误，并检查错误消息格式化
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[0].shape[1] to be <= 6, but got 7"),
        ):
            ep.module()(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[1].shape[0] to be >= 3, but got 2"),
        ):
            ep.module()(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))
    def test_runtime_assert_some_dims_not_specified(self) -> None:
        # 定义一个测试函数，验证当某些维度未指定时的运行时断言
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                # 模型的前向传播，对输入 x 和 y 进行操作
                return x.cos().sum() + y.sin().sum()

        # 创建两个张量作为输入
        x = torch.zeros(4, 2, 3)
        y = torch.zeros(5, 5, 5)

        # 定义动态维度约束对象
        dim1_x = torch.export.Dim("dim1_x", min=2, max=6)
        dim0_x = torch.export.Dim("dim0_x", min=3)

        # 导出模型并应用动态形状约束
        ep = torch.export.export(
            M(), (x, y), dynamic_shapes={"x": {0: dim0_x, 1: dim1_x}, "y": None}
        )

        # 断言捕获运行时错误，并验证错误信息中的预期输出
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[0].shape[1] to be <= 6, but got 7"),
        ):
            ep.module()(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        # 对于 y 的特化维度为 5，验证对应错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[1].shape[0] to be equal to 5, but got 2"),
        ):
            ep.module()(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))

        # 因为没有插入 x[1] >= 2 的约束，所以对于 x[1] == 1 的情况应该可以工作
        gm_result_for_1_size = ep.module()(torch.ones(3, 1, 3), torch.ones(5, 5, 5))
        eager_result_for_1_size = M().forward(torch.ones(3, 1, 3), torch.ones(5, 5, 5))

        # 断言结果相等
        self.assertEqual(gm_result_for_1_size, eager_result_for_1_size)

    def test_runtime_assert_some_inps_not_used(self) -> None:
        # 定义一个测试函数，验证当某些输入未使用时的运行时断言
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                # 模型的前向传播，只对输入 y 进行操作
                return y.cos().sum()

        # 创建两个张量作为输入
        x = torch.zeros(4, 2, 3)
        y = torch.zeros(5, 5, 5)

        # 定义动态维度约束对象
        dim1_y = torch.export.Dim("dim1_y", min=3, max=6)

        # 导出模型并应用动态形状约束
        ep = torch.export.export(
            M(), (x, y), dynamic_shapes={"x": None, "y": {1: dim1_y}}
        )

        # 断言捕获运行时错误，并验证错误信息中的预期输出
        with self.assertRaisesRegex(RuntimeError, escape("shape[1] to be equal to 2")):
            ep.module()(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        # 对于 y 的特化维度为 5，验证对应错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[1].shape[0] to be equal to 5, but got 2"),
        ):
            ep.module()(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))

        # 因为没有插入 x[1] >= 2 的约束，所以对于 x[1] == 1 的情况应该可以工作
        gm_result_for_1_size = ep.module()(torch.zeros(4, 2, 3), torch.ones(5, 5, 5))
        eager_result_for_1_size = M().forward(torch.zeros(4, 2, 3), torch.ones(5, 5, 5))

        # 断言结果相等
        self.assertEqual(gm_result_for_1_size, eager_result_for_1_size)
    # 定义一个测试方法，验证视图到视图拷贝的转换
    def test_view_to_view_copy(self) -> None:
        # 定义一个继承自torch.nn.Module的子类M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 前向传播方法，对输入张量x进行处理
            def forward(self, x):
                # 创建一个视图z，形状与x相同
                z = x.view(x.shape)
                # 对z中的每个元素进行余弦运算，并返回所有元素的和
                return z.cos().sum()

        # 创建一个形状为(4, 2, 3)的全零张量x
        x = torch.zeros(4, 2, 3)

        # 对模型M进行导出，获取其图形表示ep
        ep = export(M(), (x,))
        # 断言在ep的计算图中调用torch.ops.aten.view.default的次数为1
        self.assertEqual(count_call_function(ep.graph, torch.ops.aten.view.default), 1)

        # 使用ReplaceViewOpsWithViewCopyOpsPass转换ep
        ep = ep._transform_do_not_use(ReplaceViewOpsWithViewCopyOpsPass())
        # 断言在转换后的ep的计算图中不再调用torch.ops.aten.view.default
        self.assertEqual(count_call_function(ep.graph, torch.ops.aten.view.default), 0)

    # 定义一个测试方法，验证使用视图拷贝进行功能化处理
    def test_functionalization_with_view_copy(self) -> None:
        # 定义一个继承自torch.nn.Module的子类Module
        class Module(torch.nn.Module):
            # 前向传播方法，对输入张量x进行处理
            def forward(self, x):
                # 对输入张量x加上4
                y = x + 4
                # 在y上加上4（就地操作）
                y.add_(4)
                # 创建一个视图z，形状与y相同
                z = y.view(y.shape)
                # 返回x的余弦和z的余弦之和
                return x.cos() + z.cos()

        # 创建一个形状为(4, 2, 3)的全零张量x
        x = torch.zeros(4, 2, 3)
        # 创建Module类的实例foo
        foo = Module()
        # 对模型foo进行导出，获取其图形表示ep，并应用视图拷贝的转换
        ep = export(foo, (x,))._transform_do_not_use(
            ReplaceViewOpsWithViewCopyOpsPass()
        )
        # 断言在转换后的ep的计算图中不再调用torch.ops.aten.view.default
        self.assertTrue(count_call_function(ep.graph, torch.ops.aten.view.default) == 0)
        # 断言在转换后的ep的计算图中调用torch.ops.aten.view_copy.default至少一次
        self.assertTrue(
            count_call_function(ep.graph, torch.ops.aten.view_copy.default) > 0
        )

    # 定义一个测试方法，验证具有视图拷贝的视图操作
    def test_views_op_having_view_copy(self) -> None:
        # 获取所有的分发键对应的注册模式
        schemas = torch._C._dispatch_get_registrations_for_dispatch_key("")
        # 从所有注册模式中筛选出以"aten::"开头的模式
        aten_schemas = [s[6:] for s in schemas if s.startswith("aten::")]

        # 遍历每个符合条件的aten_schema
        for aten_schema in aten_schemas:
            # 将aten_schema按"."分割成name和overload
            val = aten_schema.split(".")
            # 确保分割结果不超过两个元素
            assert len(val) <= 2
            name = ""
            overload = ""
            # 根据分割结果确定name和overload的值
            if len(val) == 1:
                name = val[0]
                overload = "default"
            else:
                name, overload = val[0], val[1]

            # 获取指定名称和重载的操作函数
            op_overload = getattr(getattr(torch.ops.aten, name), overload)
            # 如果操作函数标记为核心，并且是视图操作
            if torch.Tag.core in op_overload.tags and is_view_op(op_overload._schema):
                # 确保能找到对应视图操作的视图拷贝
                self.assertIsNotNone(get_view_copy_of_view_op(op_overload._schema))
    def test_custom_obj_tuple_out(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个自定义对象 _Foo 的实例，并赋值给属性 attr
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # 调用 TorchScript 函数 takes_foo_tuple_return 处理 self.attr 和输入 x，返回结果保存在 a 中
                a = torch.ops._TorchScriptTesting.takes_foo_tuple_return(self.attr, x)
                # 对结果 a 中的两个元素求和，保存在 y 中
                y = a[0] + a[1]
                # 调用 TorchScript 函数 takes_foo 处理 self.attr 和 y，返回结果保存在 b 中
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                # 返回结果 b
                return b

        m = MyModule()
        inputs = (torch.ones(2, 3),)
        # 导出模型 m，使用输入 inputs，允许非严格模式（strict=False）
        ep = torch.export.export(m, inputs, strict=False)

        inp = torch.randn(2, 3)
        # 使用原始模型 m 处理输入 inp，保存结果到 orig_res
        orig_res = m(inp)
        # 使用导出模型 ep 处理输入 inp，保存结果到 ep_res
        ep_res = ep.module()(inp)

        # 移除导出模型中的效果标记
        without_token_ep = _remove_effect_tokens(ep)
        # 对移除效果标记后的模型进行验证
        without_token_ep.verifier().check(without_token_ep)
        # 使用移除效果标记后的模型处理输入 inp，保存结果到 without_token_res
        without_token_res = without_token_ep.module()(inp)

        # 断言：原始模型结果 orig_res 与导出模型结果 ep_res 在数值上全部接近
        self.assertTrue(torch.allclose(orig_res, ep_res))
        # 断言：原始模型结果 orig_res 与移除效果标记后模型结果 without_token_res 在数值上全部接近
        self.assertTrue(torch.allclose(orig_res, without_token_res))

    def test_fakify_script_objects(self):
        # 循环遍历多个模型对象 m
        for m in [
            ModelsWithScriptObjectAttr.Simple(),
            ModelsWithScriptObjectAttr.SimpleWithAttrInContainer(),
            ModelsWithScriptObjectAttr.NestedWithAttrInContainer(),
            ModelsWithScriptObjectAttr.MoreNestedWithAttrInContainer(),
        ]:
            # 收集模型 m 的常量属性
            constant_attrs = _gather_constant_attrs(m)
            # 创建 FakeTensorMode 实例 fake_mode
            fake_mode = FakeTensorMode(
                shape_env=ShapeEnv(tracked_fakes=[]),
                allow_non_fake_inputs=True,
            )
            # 使用 _fakify_script_objects 函数模拟脚本对象 m
            with _fakify_script_objects(m, tuple(), {}, fake_mode) as (
                patched_mod,
                _,
                _,
                fake_constant_attrs,
                fake_to_real,
            ):
                # 断言：模拟的常量属性数量与原始模型的常量属性数量相同
                self.assertEqual(len(fake_constant_attrs), len(constant_attrs))
                # 遍历模拟的常量属性和其对应的全限定名，进行断言
                for fake_obj, fqn in fake_constant_attrs.items():
                    self.assertEqual(constant_attrs[fake_to_real[fake_obj]], fqn)

    # TODO: _gather_constants doesn't recursively look into the pytree containers.
    @unittest.expectedFailure
    def test_fakify_script_objects_properly_handle_containers(self):
        # 创建模型对象 m
        m = ModelsWithScriptObjectAttr.SimpleWithAttrInContainer()
        # 收集模型 m 的常量属性
        constant_attrs = _gather_constant_attrs(m)
        # 创建 FakeTensorMode 实例 fake_mode
        fake_mode = FakeTensorMode(
            shape_env=ShapeEnv(tracked_fakes=[]),
            allow_non_fake_inputs=True,
        )
        # 使用 _fakify_script_objects 函数模拟脚本对象 m
        with _fakify_script_objects(m, tuple(), {}, fake_mode) as (
            patched_mod,
            _,
            _,
            fake_constant_attrs,
            fake_to_real,
        ):
            # 断言：模拟的常量属性中包含 "attr" 和 "pytree_attr2" 这两个值
            self.assertTrue("attr" in fake_constant_attrs.values())
            self.assertTrue("pytree_attr2" in fake_constant_attrs.values())
    def test_runtime_assert_inline_constraints_for_item(self) -> None:
        # 定义一个测试函数，验证在模型中对单个项的运行时约束条件
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 获取张量 x 的单个元素，并赋值给变量 b
                b = x.item()
                # 断言 b 的值大于等于 2
                torch._check(b >= 2)
                # 断言 b 的值小于等于 5
                torch._check(b <= 5)
                # 返回 b
                return b

        x = torch.tensor([2])
        # 实例化 M 类
        mod = M()
        # 导出模型的推理图谱
        ep = export(mod, (x,))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Invalid value range for 6 between \[2, 5\]",
        ):
            # 调用导出的模型并传入超出范围的输入张量
            ep.module()(torch.tensor([6]))

        new_inp = torch.tensor([5])
        # 断言模型在给定新输入时的输出与预期输出相等
        self.assertEqual(mod(new_inp), ep.module()(new_inp))

    def test_runtime_assert_inline_constraints_for_nonzero(self) -> None:
        # 定义一个测试函数，验证在模型中对非零元素的运行时约束条件
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 获取张量 x 的非零元素的索引，并赋值给变量 b
                b = x.nonzero()
                # 断言 b 的形状的第一个维度大于等于 3
                torch._check(b.shape[0] >= 3)
                # 断言 b 的形状的第一个维度小于等于 5
                torch._check(b.shape[0] <= 5)
                # 返回 b
                return b

        x = torch.tensor([2, 1, 2, 3, 5, 0])

        mod = M()
        dim0_x = torch.export.Dim("dim0_x")
        # 导出模型，并指定动态维度参数
        ep = torch.export.export(mod, (x,), dynamic_shapes={"x": {0: dim0_x}})

        num_assert = count_call_function(
            ep.graph, torch.ops.aten._assert_scalar.default
        )

        # 断言导出的模型中调用 _assert_scalar.default 函数的次数为 2
        self.assertEqual(num_assert, 2)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Invalid value range for",
        ):
            # 调用导出的模型并传入超出范围的输入张量
            ep.module()(torch.tensor([1, 1, 0, 0, 0]))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Invalid value range for",
        ):
            # 调用导出的模型并传入全为 1 的张量
            ep.module()(torch.ones(6))

        new_inp = torch.tensor([1, 1, 1, 1])
        # 断言模型在给定新输入时的输出与预期输出相等
        self.assertEqual(mod(new_inp), ep.module()(new_inp))

    @unittest.skipIf(IS_WINDOWS, "Windows not supported")
    def test_runtime_assert_inline_constraints_for_cond(self) -> None:
        # 定义一个测试函数，验证在模型中对条件运算的运行时约束条件
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, pred, x, y):
                # 定义一个真值分支函数
                def true_fn(x, y):
                    # 获取张量 x 的单个元素，并赋值给变量 b
                    b = x.item()
                    # 断言 b 的值大于等于 2
                    torch._check(b >= 2)
                    # 断言 b 的值小于等于 5
                    torch._check(b <= 5)
                    # 返回 x 减去 b 的结果
                    return x - b

                # 定义一个假值分支函数
                def false_fn(x, y):
                    # 获取张量 y 的单个元素，并赋值给变量 c
                    c = y.item()
                    # 断言 c 的值大于等于 2
                    torch._check(c >= 2)
                    # 断言 c 的值小于等于 5
                    torch._check(c <= 5)
                    # 返回 y 减去 c 的结果
                    return y - c

                # 根据 pred 的值选择执行 true_fn 或 false_fn
                ret = cond(pred, true_fn, false_fn, [x, y])
                # 返回结果 ret
                return ret

        x = torch.tensor([2])
        y = torch.tensor([5])
        mod = M()
        # 导出模型的推理图谱
        ep = export(mod, (torch.tensor(True), x, y))

        with self.assertRaisesRegex(
            RuntimeError, "is outside of inline constraint \\[2, 5\\]."
        ):
            # 调用导出的模型并传入不满足条件的输入张量
            ep.module()(torch.tensor(False), torch.tensor([6]), torch.tensor([6]))
    def test_math_ops(self):
        # 定义一个内嵌的 PyTorch 模块类 Module，继承自 torch.nn.Module
        class Module(torch.nn.Module):
            # 定义模块的前向传播方法
            def forward(self, x):
                # 返回 x 的向上取整和向下取整后的张量
                return (
                    torch.tensor([math.ceil(x.item())]),  # 向上取整的张量
                    torch.tensor([math.floor(x.item())]),  # 向下取整的张量
                )

        # 创建 Module 类的实例 func
        func = Module()
        # 生成一个随机张量 x，数据类型为 torch.float32
        x = torch.randn(1, dtype=torch.float32)
        # 使用 torch.export.export 将 func 模块导出为一个表示执行计算的对象 ep
        ep = torch.export.export(func, args=(x,))
        # 调用 _ExportPassBaseDeprecatedDoNotUse 类的实例的方法，传入 ep.graph_module 进行处理
        _ExportPassBaseDeprecatedDoNotUse()(ep.graph_module)

    def test_predispatceh_set_grad(self):
        # 从 self.SET_GRAD_ENABLED_TESTS["op"] 中获取 mod 和 args
        mod, args = self.SET_GRAD_ENABLED_TESTS["op"]
        # 断言 mod.code.strip("\n") 与下面的字符串相等
        self.assertExpectedInline(
            mod.code.strip("\n"),
            """\
# 定义一个方法 forward，用于神经网络模块的前向传播
def forward(self, x):
    # 将输入 x 扁平化并符合指定规范，以便后续处理
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    # 使用 Torch 的 ATen 操作，对 x 加 1
    add = torch.ops.aten.add.Tensor(x, 1);  x = None
    # 对 add 执行正弦函数操作
    sin = torch.ops.aten.sin.default(add);  add = None
    # 对 sin 操作结果求和
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    # 获取当前模块下的 submod_2 属性，赋值给 submod_4
    submod_4 = self.submod_2
    # 使用 Torch 的高阶操作，禁用梯度追踪，并应用于 submod_4 和 sum_1
    add_1 = torch._higher_order_ops.wrap.wrap_with_set_grad_enabled(False, submod_4, sum_1);  submod_4 = sum_1 = None
    # 对 add_1 执行减法操作
    sub = torch.ops.aten.sub.Tensor(add_1, 1)
    # 将结果重新构建成树结构，并符合输出规范，返回结果
    return pytree.tree_unflatten((add_1, sub), self._out_spec)
    cos = torch.ops.aten.cos.default(add);  add = None
    # 计算给定张量的余弦值，结果存储在 cos 中；add 变量置为 None

    sum_2 = torch.ops.aten.sum.default(cos);  cos = None
    # 对 cos 张量进行求和操作，结果存储在 sum_2 中；cos 变量置为 None

    submod_3 = self.submod_1
    # 将 self 对象的 submod_1 属性赋值给 submod_3 变量

    wrap_with_set_grad_enabled = torch._higher_order_ops.wrap.wrap_with_set_grad_enabled(False, submod_3, sum_1, sum_2);  submod_3 = sum_1 = sum_2 = None
    # 使用 wrap_with_set_grad_enabled 函数将上下文管理器包装在计算图设置为不启用梯度的环境中；
    # 设置 submod_3、sum_1 和 sum_2 变量为 None

    add_1 = wrap_with_set_grad_enabled[0]
    # 从 wrap_with_set_grad_enabled 返回的结果中获取第一个元素，存储在 add_1 中

    add_2 = wrap_with_set_grad_enabled[1];  wrap_with_set_grad_enabled = None
    # 从 wrap_with_set_grad_enabled 返回的结果中获取第二个元素，存储在 add_2 中；
    # wrap_with_set_grad_enabled 变量置为 None

    sub = torch.ops.aten.sub.Tensor(add_1, 1)
    # 创建一个张量，该张量是 add_1 减去标量 1 的结果，存储在 sub 中

    sub_1 = torch.ops.aten.sub.Tensor(add_2, 1)
    # 创建一个张量，该张量是 add_2 减去标量 1 的结果，存储在 sub_1 中

    return pytree.tree_unflatten((add_1, add_2, sub, sub_1), self._out_spec)
    # 使用 pytree.tree_unflatten 将 (add_1, add_2, sub, sub_1) 打包成一个树结构，
    # 返回结果作为方法调用的返回值，self._out_spec 作为参数传递给 tree_unflatten 函数
def forward(self, x):
    # 将输入 x 打包成 pytree，并根据输入规范展开
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    # 使用 Torch 的 aten.add 操作，给输入 x 加上常数 1
    add = torch.ops.aten.add.Tensor(x, 1);  x = None
    # 将 self.submod_1 赋给 submod_5
    submod_5 = self.submod_1
    # 使用 wrap_with_set_grad_enabled 包装 set_grad_enabled(True) 和 add 操作
    wrap_with_set_grad_enabled = torch._higher_order_ops.wrap.wrap_with_set_grad_enabled(True, submod_5, add);  submod_5 = add = None
    # 获取 wrap_with_set_grad_enabled 返回的第一个元素，即加法的结果 sum_1
    sum_1 = wrap_with_set_grad_enabled[0]
    # 获取 wrap_with_set_grad_enabled 返回的第二个元素，即加法的结果 sum_2；然后清空 wrap_with_set_grad_enabled
    sum_2 = wrap_with_set_grad_enabled[1];  wrap_with_set_grad_enabled = None
    # 对 sum_1 再次执行 aten.add 操作，加上常数 1
    add_1 = torch.ops.aten.add.Tensor(sum_1, 1);  sum_1 = None
    # 对 sum_2 再次执行 aten.add 操作，加上常数 1
    add_2 = torch.ops.aten.add.Tensor(sum_2, 1);  sum_2 = None
    # 将 self.submod_3 赋给 submod_6
    submod_6 = self.submod_3
    # 使用 wrap_with_set_grad_enabled 包装 set_grad_enabled(True)、add_1 和 add_2 操作
    wrap_with_set_grad_enabled_1 = torch._higher_order_ops.wrap.wrap_with_set_grad_enabled(True, submod_6, add_1, add_2);  submod_6 = None
    # 获取 wrap_with_set_grad_enabled_1 返回的第一个元素，即减法的结果 sub
    sub = wrap_with_set_grad_enabled_1[0]
    # 获取 wrap_with_set_grad_enabled_1 返回的第二个元素，即减法的结果 sub_1；然后清空 wrap_with_set_grad_enabled_1
    sub_1 = wrap_with_set_grad_enabled_1[1];  wrap_with_set_grad_enabled_1 = None
    # 使用 pytree.tree_unflatten 将 add_1、add_2、sub 和 sub_1 打包成 pytree，根据输出规范进行展开并返回结果
    return pytree.tree_unflatten((add_1, add_2, sub, sub_1), self._out_spec)
    def forward(self, add, add_1):
        # 禁用梯度跟踪
        _set_grad_enabled_1 = torch._C._set_grad_enabled(False)
        # 计算第一个输入张量的正弦值
        sin = torch.ops.aten.sin.default(add);  add = None
        # 计算第二个输入张量的余弦值
        cos = torch.ops.aten.cos.default(add_1);  add_1 = None
        # 返回正弦值和余弦值
        return (sin, cos)
    """
        )

        # 断言预期的内联结果
        self.assertExpectedInline(
            new_gm.submod_3.code.strip("\n"),
            """\
def forward(self, sin, cos):
    # 启用梯度跟踪
    _set_grad_enabled_2 = torch._C._set_grad_enabled(True)
    # 将正弦值张量加1
    add_2 = torch.ops.aten.add.Tensor(sin, 1);  sin = None
    # 将余弦值张量加1
    add_3 = torch.ops.aten.add.Tensor(cos, 1);  cos = None
    # 返回加1后的张量
    return (add_2, add_3)
    """
        )

    def test_inline_(self):
        # 遍历顺序分割内联测试值
        for gm, args in self.SEQUENTIAL_SPLIT_INLINE_TESTS.values():
            # 获取转换前的可读字符串
            before_str = gm.print_readable(print_output=False)
            # 进行顺序分割转换
            new_gm = sequential_split(gm, _is_set_grad_enabled_node)
            # 映射节点内联
            nodes_map(
                new_gm.graph.nodes,
                lambda node: node_inline_(node) if node.op == "call_module" else node,
            )
            # 获取内联后的可读字符串
            after_inline_str = new_gm.print_readable(print_output=False)
            # 断言转换前后字符串一致
            self.assertEqual(before_str, after_inline_str)
            # 断言转换前后模型输出一致
            self.assertEqual(gm(*args), new_gm(*args))

    def test_remove_auto_functionalized_pass(self) -> None:
        # 使用测试专用库定义函数
        with _scoped_library("DO_NOT_USE_TEST_ONLY", "DEF") as lib:
            lib.define("custom_mutator(Tensor x, Tensor(a!) y) -> Tensor")

            @impl(lib, "custom_mutator", "Meta")
            def custom_mutator_meta(
                x: torch.Tensor,
                y: torch.Tensor,
            ) -> torch.Tensor:
                return torch.empty_like(x)

            @impl(lib, "custom_mutator", "CompositeExplicitAutograd")
            def custom_mutator(
                x: torch.Tensor,
                y: torch.Tensor,
            ) -> torch.Tensor:
                return x + y.add_(1)

            # 定义测试模型类
            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("state", torch.zeros(1))

                def forward(self, x):
                    # 调用自定义的变异器函数
                    return torch.ops.DO_NOT_USE_TEST_ONLY.custom_mutator(x, self.state)

            # 实例化测试模型
            mod = M()
            # 生成随机输入张量
            x = torch.randn([3, 3])
            # 导出模型
            ep = export(mod, (x,))
            # 执行不安全的自动功能化删除
            inplace_ep = unsafe_remove_auto_functionalized_pass(ep)
            # 获取图中的节点
            nodes = inplace_ep.graph.nodes
            # 断言每个调用函数节点的目标不是自动功能化或者getitem
            for node in nodes:
                if node.op == "call_function":
                    self.assertFalse(node.target is auto_functionalized)
                    self.assertFalse(node.target is operator.getitem)

            # 断言输出签名中没有"getitem"
            for spec in inplace_ep.graph_signature.output_specs:
                self.assertFalse("getitem" in spec.arg.name)
    def test_remove_auto_functionalized_pass_tuple(self) -> None:
        # 使用特定库进行临时上下文管理
        with _scoped_library("DO_NOT_USE_TEST_ONLY", "DEF") as lib:
            # 在库中定义一个名为custom_mutator_tuple的函数
            lib.define(
                "custom_mutator_tuple(Tensor x, Tensor(a!) y) -> (Tensor, Tensor)"
            )

            # 实现custom_mutator_tuple函数的Meta版本，返回两个空张量
            @impl(lib, "custom_mutator_tuple", "Meta")
            def custom_mutator_tuple_meta(
                x: torch.Tensor,
                y: torch.Tensor,
            ):
                return (torch.empty_like(x), torch.empty_like(x))

            # 实现custom_mutator_tuple函数的CompositeExplicitAutograd版本，返回(x, x + y.add_(1))
            @impl(lib, "custom_mutator_tuple", "CompositeExplicitAutograd")
            def custom_mutator_tuple(
                x: torch.Tensor,
                y: torch.Tensor,
            ):
                return (x, x + y.add_(1))

            # 定义一个继承自torch.nn.Module的类M
            class M(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("state", torch.zeros(1))

                # 实现Module的forward方法，调用自定义函数custom_mutator_tuple
                def forward(self, x):
                    return torch.ops.DO_NOT_USE_TEST_ONLY.custom_mutator_tuple(
                        x, self.state
                    )

            # 创建M类的实例mod
            mod = M()
            # 生成一个3x3大小的随机张量x
            x = torch.randn([3, 3])
            # 对M类实例mod进行导出操作，传入参数(x,)，并得到输出ep
            ep = export(mod, (x,))
            # 在原地移除自动功能化传递的输出，得到inplace_ep
            inplace_ep = unsafe_remove_auto_functionalized_pass(ep)

            # 获取图中的节点列表
            nodes = inplace_ep.graph.nodes
            getitems = 0
            # 遍历节点列表
            for node in nodes:
                # 如果节点操作为"call_function"
                if node.op == "call_function":
                    # 断言节点目标不是自动功能化
                    self.assertFalse(node.target is auto_functionalized)
                    # 如果节点目标是operator.getitem，则增加getitems计数
                    if node.target is operator.getitem:
                        getitems += 1
            # 断言getitems的计数为2，即元组返回长度为2
            self.assertEqual(getitems, 2)

            # 获取输出签名的规范
            out_specs = inplace_ep.graph_signature.output_specs
            # 断言第一个输出规范的参数名为"b_state"，对应state
            self.assertEqual(out_specs[0].arg.name, "b_state")
            # 断言第二个输出规范的参数名为"getitem"，对应元组返回的第一个项目
            self.assertEqual(out_specs[1].arg.name, "getitem")
            # 断言第三个输出规范的参数名为"getitem_1"，对应元组返回的第二个项目
            self.assertEqual(out_specs[2].arg.name, "getitem_1")
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 调用函数 run_tests()，用于执行测试代码或者功能测试
    run_tests()
```