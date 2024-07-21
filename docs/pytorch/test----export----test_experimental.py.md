# `.\pytorch\test\export\test_experimental.py`

```
# Owner(s): ["oncall: export"]
# flake8: noqa

# 导入单元测试框架
import unittest

# 导入类型提示
from typing import Dict, List, Tuple

# 导入PyTorch库
import torch
import torch._dynamo
from torch._dynamo.test_case import run_tests, TestCase
from torch._export.wrappers import _mark_strict_experimental

# 导入功能模块
from torch._functorch.aot_autograd import aot_export_module
from torch.export._trace import _convert_ts_to_export_experimental
from torch.export.experimental import _export_forward_backward

# 导入文件检查工具
from torch.testing import FileCheck

# 如果当前环境不支持动态图功能，则跳过测试
@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't supported")
class TestExperiment(TestCase):
    # 定义测试方法，测试包含缓冲区的子模块
    def test_with_buffer_as_submodule(self):
        # 使用严格实验性标记修饰类
        @_mark_strict_experimental
        class B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个名为buffer1的缓冲区，值为全1的张量
                self.register_buffer("buffer1", torch.ones(3))

            def forward(self, x):
                # 对输入张量执行操作，并将结果保存到y中
                y = x + 2
                # 在y上执行原地加法操作
                y.add_(4)
                # 试图对缓冲区buffer1执行原地加法，但是当前不支持HOO
                # self.buffer1.add_(6)
                # 计算更新后的buffer1
                buffer_updated = self.buffer1 + 6
                # 返回输入张量x、y的和以及buffer_updated的和
                return x.sum() + y.sum() + buffer_updated.sum()

        # 定义包含子模块B的主模块M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submodule = B()

            def forward(self, x):
                # 对输入张量执行sin函数操作
                x_v2 = x.sin()
                # 返回子模块的输出和x加3的结果
                return (self.submodule(x_v2), x + 3)

        # 创建一个随机张量作为输入
        inp = torch.randn(3)
        # 对模块M进行导出，传入输入张量inp，并设置严格模式为False
        ep = torch.export.export(M(), (inp,), strict=False)

        # 断言导出的代码与预期的内联代码匹配
        self.assertExpectedInline(
            str(ep.graph_module.code.strip()),
            """\
def forward(self, b_submodule_buffer1, x):
    sin = torch.ops.aten.sin.default(x)
    strict_graph_0 = self.strict_graph_0
    strict_mode = torch.ops.higher_order.strict_mode(strict_graph_0, (sin, b_submodule_buffer1));  strict_graph_0 = sin = b_submodule_buffer1 = None
    getitem_2 = strict_mode[0];  strict_mode = None
    add = torch.ops.aten.add.Tensor(x, 3);  x = None
    return (getitem_2, add)""",
        )

        # 断言导出的严格图代码与预期的内联代码匹配
        self.assertExpectedInline(
            str(ep.graph_module.strict_graph_0.code.strip()),
            """\
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg0_1, 2)
    add_1 = torch.ops.aten.add.Tensor(add, 4);  add = None
    add_2 = torch.ops.aten.add.Tensor(arg1_1, 6);  arg1_1 = None
    sum_1 = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
    sum_2 = torch.ops.aten.sum.default(add_1);  add_1 = None
    add_3 = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
    sum_3 = torch.ops.aten.sum.default(add_2);  add_2 = None
    add_4 = torch.ops.aten.add.Tensor(add_3, sum_3);  add_3 = sum_3 = None
""",
        )
    return (add_4,)""",

返回一个包含单个元素 `(add_4,)` 的元组。


        )

        eager_mod = M()

创建一个新的 `M` 类的实例 `eager_mod`。


        ep = torch.export.export(eager_mod, (inp,), strict=True)

使用 `torch.export.export` 导出 `eager_mod` 模型，并且传入输入 `(inp,)` 和严格模式 `strict=True`。


        graph_res_1, graph_res_2 = ep.module()(inp)
        eager_res_1, eager_res_2 = eager_mod(inp)

调用 `ep.module()(inp)` 和 `eager_mod(inp)`，分别得到 `graph_res_1, graph_res_2` 和 `eager_res_1, eager_res_2`。


        self.assertTrue(torch.allclose(graph_res_2, eager_res_2))
        self.assertTrue(torch.allclose(graph_res_1, eager_res_1))

断言 `graph_res_2` 与 `eager_res_2` 以及 `graph_res_1` 与 `eager_res_1` 的所有元素都近似相等。


        graph_res_1, graph_res_2 = ep.module()(inp)
        eager_res_1, eager_res_2 = eager_mod(inp)

再次调用 `ep.module()(inp)` 和 `eager_mod(inp)`，重新得到 `graph_res_1, graph_res_2` 和 `eager_res_1, eager_res_2`。


        self.assertTrue(torch.allclose(graph_res_2, eager_res_2))
        self.assertTrue(torch.allclose(graph_res_1, eager_res_1))

再次断言 `graph_res_2` 与 `eager_res_2` 以及 `graph_res_1` 与 `eager_res_1` 的所有元素都近似相等。

    def test_mark_strict_with_container_type(self):

定义一个测试方法 `test_mark_strict_with_container_type`。


        @_mark_strict_experimental

使用 `_mark_strict_experimental` 装饰器，该装饰器可能标记一个类或函数为实验性严格模式。


        class B(torch.nn.Module):

定义一个继承自 `torch.nn.Module` 的类 `B`。


            def __init__(self):
                super().__init__()

`B` 类的构造函数，调用父类 `torch.nn.Module` 的构造函数。


            def forward(self, x):
                x0 = x[0][0]
                return x0.sum()

`B` 类的前向传播函数 `forward`，接收输入 `x`，取其第一层嵌套的第一个元素 `x[0][0]`，并返回其求和。


        class M(torch.nn.Module):

定义一个继承自 `torch.nn.Module` 的类 `M`。


            def __init__(self):
                super().__init__()
                self.submodule = B()

`M` 类的构造函数，调用父类 `torch.nn.Module` 的构造函数，并初始化一个 `B` 类的实例 `self.submodule`。


            def forward(self, x):
                return self.submodule(x)

`M` 类的前向传播函数 `forward`，将输入 `x` 传递给 `self.submodule` 的前向传播函数。


        inp = ((torch.randn(3),),)

定义一个包含一个元组的元组 `inp`，元组中包含一个包含一个形状为 `(3,)` 的随机张量的元组。


        with self.assertRaisesRegex(
            RuntimeError, "strict_mode HOO doesn't work unless"
        ):

使用 `self.assertRaisesRegex` 上下文管理器，断言在运行时抛出 `RuntimeError` 异常，并且异常信息包含字符串 `"strict_mode HOO doesn't work unless"`。


            ep = torch.export.export(M(), inp, strict=False)

在 `with` 块中，调用 `torch.export.export` 导出 `M` 类的实例，并传入输入 `inp`，严格模式为 `strict=False`。

    def test_torchscript_module_export(self):

定义一个测试方法 `test_torchscript_module_export`。


        class M(torch.nn.Module):

定义一个继承自 `torch.nn.Module` 的类 `M`。


            def forward(self, x):
                return x.cos() + x.sin()

`M` 类的前向传播函数 `forward`，对输入 `x` 执行余弦和正弦操作，并返回结果。


        model_to_trace = M()

创建一个 `M` 类的实例 `model_to_trace`。


        inps = (torch.randn(4, 4),)

定义一个包含一个形状为 `(4, 4)` 的随机张量的元组 `inps`。


        traced_module_by_torchscript = torch.jit.trace(M(), example_inputs=inps)

使用 `torch.jit.trace` 对 `M` 类进行追踪转换，传入示例输入 `inps`。


        exported_module = _convert_ts_to_export_experimental(
            traced_module_by_torchscript, inps
        )

调用 `_convert_ts_to_export_experimental` 函数，将追踪后的 `traced_module_by_torchscript` 和输入 `inps` 转换为导出的模块 `exported_module`。


        self.assertTrue(torch.allclose(exported_module(*inps), model_to_trace(*inps)))

断言调用 `exported_module` 和 `model_to_trace` 传入 `inps` 后的结果近似相等。

    def test_torchscript_module_export_single_input(self):

定义一个测试方法 `test_torchscript_module_export_single_input`。


        class M(torch.nn.Module):

定义一个继承自 `torch.nn.Module` 的类 `M`。


            def forward(self, x):
                return x.cos() + x.sin()

`M` 类的前向传播函数 `forward`，对输入 `x` 执行余弦和正弦操作，并返回结果。


        model_to_trace = M()

创建一个 `M` 类的实例 `model_to_trace`。


        inps = torch.randn(4, 4)

定义一个形状为 `(4, 4)` 的随机张量 `inps`。


        traced_module_by_torchscript = torch.jit.trace(M(), example_inputs=inps)

使用 `torch.jit.trace` 对 `M` 类进行追踪转换，传入示例输入 `inps`。


        exported_module = _convert_ts_to_export_experimental(
            traced_module_by_torchscript, inps
        )

调用 `_convert_ts_to_export_experimental` 函数，将追踪后的 `traced_module_by_torchscript` 和输入 `inps` 转换为导出的模块 `exported_module`。


        self.assertTrue(torch.allclose(exported_module(inps), model_to_trace(inps)))

断言调用 `exported_module` 和 `model_to_trace` 传入 `inps` 后的结果近似相等。
    def test_torchscript_module_export_various_inputs_with_annotated_input_names(self):
        def _check_equality_and_annotations(m_func, inps):
            # 原始模块。
            model_to_trace = m_func()

            # 使用 TorchScript 对模块进行跟踪。
            traced_module_by_torchscript = torch.jit.trace(
                m_func(), example_inputs=inps
            )

            # 将 TorchScript 模块转换为 ExportedProgram。
            exported_module = _convert_ts_to_export_experimental(
                traced_module_by_torchscript, inps
            )

            # 从原始模块导出的 ExportedProgram。
            original_exported_module = torch.export.export(m_func(), inps)

            # 检查输入注释是否与跟踪原始模块时相同。
            orig_ph_name_list = [
                n.name
                for n in original_exported_module.graph.nodes
                if n.op == "placeholder"
            ]
            ph_name_list = [
                n.name for n in exported_module.graph.nodes if n.op == "placeholder"
            ]
            self.assertEqual(orig_ph_name_list, ph_name_list)

            # 检查结果的相等性。
            self.assertTrue(
                torch.allclose(exported_module(*inps), model_to_trace(*inps))
            )

        # Tuple
        class MTuple(torch.nn.Module):
            def forward(self, x: Tuple[torch.Tensor]):
                return x[0] + x[1]

        _check_equality_and_annotations(MTuple, ((torch.randn(4), torch.randn(4)),))

        # List
        class MList(torch.nn.Module):
            def forward(self, x: List[torch.Tensor]):
                return x[0] + x[1]

        _check_equality_and_annotations(MList, ([torch.randn(4), torch.randn(4)],))

        # Dict
        class MDict(torch.nn.Module):
            def forward(self, x: Dict[str, torch.Tensor]):
                return x["0"] + x["1"]

        _check_equality_and_annotations(
            MDict, ({"0": torch.randn(4), "1": torch.randn(4)},)
        )

    def test_joint_dynamic(self) -> None:
        from torch.export import Dim

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.y = torch.nn.Parameter(torch.randn(3))

            def forward(self, x):
                x = torch.ones(x.shape[0], 3)
                return (self.y + x).sum()

        m = Module()
        example_inputs = (torch.randn(3),)
        m(*example_inputs)
        # 使用预分发和动态形状导出模块。
        ep = torch.export._trace._export(
            m, example_inputs, pre_dispatch=True, dynamic_shapes={"x": {0: Dim("x0")}}
        )
        # 导出前向和反向传播。
        joint_ep = _export_forward_backward(ep)
# 如果当前脚本被直接执行（而不是被导入到其他模块中），则执行以下代码块
if __name__ == "__main__":
    # 调用运行测试的函数
    run_tests()
```