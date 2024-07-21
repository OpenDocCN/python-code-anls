# `.\pytorch\test\onnx\test_pytorch_onnx_no_runtime.py`

```
# Owner(s): ["module: onnx"]

"""Tests for onnx export that don't run the exported model."""

from __future__ import annotations

import contextlib
import io
import itertools
import unittest
import unittest.mock
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import onnx  # 导入 ONNX 库
import onnx.numpy_helper  # 导入 ONNX 中用于处理 NumPy 的辅助函数
import pytorch_test_common  # 导入 PyTorch 测试常用函数

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块
from torch import Tensor  # 导入 PyTorch 中的张量类
from torch.onnx import symbolic_helper, utils  # 导入 PyTorch ONNX 导出相关的辅助函数和工具
from torch.onnx._internal import registration  # 导入 PyTorch ONNX 内部注册模块
from torch.testing._internal import common_quantization, common_utils, jit_utils  # 导入 PyTorch 内部测试相关模块


def export_to_onnx(
    model: Union[torch.nn.Module, torch.jit.ScriptFunction],  # 定义函数参数，可以是模块或脚本函数
    input: Union[torch.Tensor, Tuple[torch.Tensor]],  # 定义输入参数，可以是张量或张量元组
    custom_ops: Optional[  # 可选参数，自定义操作的上下文管理器列表
        Iterable[Union[contextlib.AbstractContextManager, contextlib.ContextDecorator]]
    ] = None,
    mocks: Optional[Iterable] = None,  # 可选参数，用于模拟的列表
    operator_export_type: torch.onnx.OperatorExportTypes = torch.onnx.OperatorExportTypes.ONNX,  # 导出类型，默认为 ONNX
    opset_version: int = 17,  # ONNX opset 版本，默认为 17
    **torch_onnx_export_kwargs,  # 其他传递给 torch.onnx.export 的关键字参数
) -> onnx.ModelProto:
    """Exports `model(input)` to ONNX and returns it.

    Custom operators and/or unittest patches can be used help reproducing specific behaviors.

    Args:
        model: model to export  # 待导出的模型
        input: model input with same format as `torch.onnx.export(..,args,...)`  # 与 torch.onnx.export 函数格式相同的模型输入
        custom_ops: list of custom operators to use during export  # 在导出过程中使用的自定义操作列表
        mocks: list of mocks to use during export  # 在导出过程中使用的模拟列表
        operator_export_type: export type as described by `torch.onnx.export(...operator_export_type,...)`  # 导出类型，如 torch.onnx.export 中描述的
        opset_version: ONNX opset version as described by `torch.onnx.export(...opset_version,...)`  # ONNX opset 版本，如 torch.onnx.export 中描述的
        torch_onnx_export_kwargs: extra torch.onnx.export kwargs arguments  # 额外的 torch.onnx.export 关键字参数
    Returns:
        A valid ONNX model (`onnx.ModelProto`)  # 一个有效的 ONNX 模型（onnx.ModelProto）
    """
    custom_ops = custom_ops or []  # 如果 custom_ops 为 None，则设为空列表
    mocks = mocks or []  # 如果 mocks 为 None，则设为空列表
    with contextlib.ExitStack() as stack:  # 使用上下文管理器 ExitStack
        for ctx in itertools.chain(custom_ops, mocks):  # 遍历自定义操作和模拟列表
            stack.enter_context(ctx)  # 进入上下文

        f = io.BytesIO()  # 创建一个字节流对象
        torch.onnx.export(  # 使用 torch.onnx.export 导出模型
            model,
            input,
            f,
            operator_export_type=operator_export_type,
            opset_version=opset_version,
            **torch_onnx_export_kwargs,
        )

    # Validate ONNX graph before returning it  # 在返回之前验证 ONNX 图
    onnx_model = onnx.load_from_string(f.getvalue())  # 从字节流加载 ONNX 模型
    onnx.checker.check_model(onnx_model)  # 检查 ONNX 模型的有效性
    return onnx_model  # 返回 ONNX 模型


@common_utils.instantiate_parametrized_tests  # 使用 common_utils 中的装饰器实例化参数化测试
class TestONNXExport(pytorch_test_common.ExportTestCase):  # 定义测试类，继承自 ExportTestCase
    def test_fuse_addmm(self):  # 定义测试方法 test_fuse_addmm
        class AddmmModel(torch.nn.Module):  # 定义一个简单的 PyTorch 模型类
            def forward(self, x):  # 定义模型的前向传播方法
                return torch.mm(x, x) + x  # 返回矩阵乘法结果加上输入张量 x

        x = torch.ones(3, 3)  # 创建一个 3x3 全为 1 的张量
        f = io.BytesIO()  # 创建一个字节流对象
        torch.onnx.export(AddmmModel(), x, f, verbose=False)  # 使用 torch.onnx.export 导出 AddmmModel 模型
    def test_onnx_transpose_incomplete_tensor_type(self):
        # 对不完整的 TensorType 进行转置操作的烟雾测试
        # 其中输入的 TensorType 没有大小信息。以前是无法工作的，
        # 因为我们会获取输入的大小并使用其大小的长度作为置换的维度数。
        class Foo(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.contiguous().transpose(0, 1).sum()

        class TraceMe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = Foo()

            def forward(self, x):
                return self.foo(x)

        # 创建 TraceMe 实例
        tm = TraceMe()
        # 对 TraceMe 进行跟踪
        tm = torch.jit.trace(tm, torch.rand(3, 4))
        # 创建字节流对象
        f = io.BytesIO()
        # 导出 ONNX 模型到字节流中
        torch.onnx.export(tm, (torch.rand(3, 4),), f)

    def test_export_tensoroption_to(self):
        def foo(x):
            return x[0].clone().detach().cpu() + x

        # 对 foo 函数进行跟踪
        traced = torch.jit.trace(foo, (torch.rand([2])))

        # 导出为 ONNX 格式的字符串
        torch.onnx.export_to_pretty_string(traced, (torch.rand([2]),))

    def test_onnx_export_script_module(self):
        class ModuleToExport(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                y = x - x
                return x + x

        # 创建 ModuleToExport 实例
        mte = ModuleToExport()
        # 导出为 ONNX 格式的字符串
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3),), verbose=False)

    @common_utils.suppress_warnings
    def test_onnx_export_func_with_warnings(self):
        @torch.jit.script
        def func_with_warning(inp):
            return torch.nn.functional.sigmoid(inp)  # 触发一个弃用警告

        class WarningTest(torch.nn.Module):
            def forward(self, x):
                return func_with_warning(x)

        # 没有异常
        torch.onnx.export_to_pretty_string(
            WarningTest(), torch.randn(42), verbose=False
        )

    def test_onnx_export_script_python_fail(self):
        class PythonModule(torch.jit.ScriptModule):
            @torch.jit.ignore
            def forward(self, x):
                return torch.neg(x)

        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.mod = PythonModule()

            @torch.jit.script_method
            def forward(self, x):
                y = self.mod(x)
                return y + y

        # 创建 ModuleToExport 实例
        mte = ModuleToExport()
        # 创建字节流对象
        f = io.BytesIO()
        # 用断言检测是否能导出 Python 失败
        with self.assertRaisesRegex(RuntimeError, "Couldn't export Python"):
            torch.onnx.export(mte, (torch.zeros(1, 2, 3),), f, verbose=False)
    # 定义一个测试方法，用于在进行 ONNX 导出时内联跟踪模块的情况
    def test_onnx_export_script_inline_trace(self):
        # 定义一个内联模块，继承自 torch.nn.Module
        class ModuleToInline(torch.nn.Module):
            # 实现前向传播方法，对输入张量进行取反操作
            def forward(self, x):
                return torch.neg(x)

        # 定义一个导出模块，继承自 torch.jit.ScriptModule
        class ModuleToExport(torch.jit.ScriptModule):
            # 构造方法初始化
            def __init__(self):
                super().__init__()
                # 使用 torch.jit.trace 方法对 ModuleToInline 进行跟踪，生成可追踪对象 self.mod
                self.mod = torch.jit.trace(ModuleToInline(), torch.zeros(1, 2, 3))

            # 使用 torch.jit.script_method 装饰器声明脚本方法 forward
            def forward(self, x):
                # 调用 self.mod 进行前向传播计算
                y = self.mod(x)
                # 返回 y 加上自身的结果
                return y + y

        # 创建 ModuleToExport 实例
        mte = ModuleToExport()
        # 使用 torch.onnx.export_to_pretty_string 导出模块到 ONNX 格式的字符串表示，关闭详细输出
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3),), verbose=False)

    # 定义一个测试方法，用于在进行 ONNX 导出时内联脚本模块的情况
    def test_onnx_export_script_inline_script(self):
        # 定义一个内联脚本模块，继承自 torch.jit.ScriptModule
        class ModuleToInline(torch.jit.ScriptModule):
            # 使用 torch.jit.script_method 装饰器声明脚本方法 forward
            def forward(self, x):
                # 对输入张量进行取反操作并返回结果
                return torch.neg(x)

        # 定义一个导出模块，继承自 torch.jit.ScriptModule
        class ModuleToExport(torch.jit.ScriptModule):
            # 构造方法初始化
            def __init__(self):
                super().__init__()
                # 创建 ModuleToInline 实例作为成员 self.mod
                self.mod = ModuleToInline()

            # 使用 torch.jit.script_method 装饰器声明脚本方法 forward
            def forward(self, x):
                # 调用 self.mod 的 forward 方法计算结果
                y = self.mod(x)
                # 返回 y 加上自身的结果
                return y + y

        # 创建 ModuleToExport 实例
        mte = ModuleToExport()
        # 使用 torch.onnx.export_to_pretty_string 导出模块到 ONNX 格式的字符串表示，关闭详细输出
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3),), verbose=False)

    # 定义一个测试方法，用于在进行 ONNX 导出时包含循环结构的情况
    def test_onnx_export_script_module_loop(self):
        # 定义一个导出模块，继承自 torch.jit.ScriptModule
        class ModuleToExport(torch.jit.ScriptModule):
            # 使用 torch.jit.script_method 装饰器声明脚本方法 forward
            def forward(self, x):
                # 在 forward 方法中包含循环结构，测试是否支持端到端的 ONNX 导出
                for _ in range(5):
                    for i in range(3):
                        x = x + i
                return x

        # 创建 ModuleToExport 实例
        mte = ModuleToExport()
        # 使用 torch.onnx.export_to_pretty_string 导出模块到 ONNX 格式的字符串表示，关闭详细输出
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3),), verbose=False)

    # 使用 common_utils.suppress_warnings 装饰器，定义一个测试方法，用于在进行 ONNX 导出时涉及真除法的情况
    @common_utils.suppress_warnings
    def test_onnx_export_script_truediv(self):
        # 定义一个导出模块，继承自 torch.jit.ScriptModule
        class ModuleToExport(torch.jit.ScriptModule):
            # 使用 torch.jit.script_method 装饰器声明脚本方法 forward
            def forward(self, x):
                # 对输入张量的大小进行真除以 2，然后加到输入张量上并返回结果
                z = x.size(0) / 2
                return x + z

        # 创建 ModuleToExport 实例
        mte = ModuleToExport()

        # 使用 torch.onnx.export_to_pretty_string 导出模块到 ONNX 格式的字符串表示，关闭详细输出
        torch.onnx.export_to_pretty_string(
            mte, (torch.zeros(1, 2, 3, dtype=torch.float),), verbose=False
        )

    # 定义一个测试方法，用于在进行 ONNX 导出时涉及非字母字符的加减法操作的情况
    def test_onnx_export_script_non_alpha_add_sub(self):
        # 定义一个导出模块，继承自 torch.jit.ScriptModule
        class ModuleToExport(torch.jit.ScriptModule):
            # 使用 torch.jit.script_method 装饰器声明脚本方法 forward
            def forward(self, x):
                # 计算输入张量大小的行数加 1，然后减 1 并返回结果
                bs = x.size(0) + 1
                return bs - 1

        # 创建 ModuleToExport 实例
        mte = ModuleToExport()
        # 使用 torch.onnx.export_to_pretty_string 导出模块到 ONNX 格式的字符串表示，关闭详细输出
        torch.onnx.export_to_pretty_string(mte, (torch.rand(3, 4),), verbose=False)
    def test_onnx_export_script_module_if(self):
        # 定义一个继承自 torch.jit.ScriptModule 的模块，用于导出到 ONNX
        class ModuleToExport(torch.jit.ScriptModule):
            # 定义一个脚本方法 forward，接受输入 x
            @torch.jit.script_method
            def forward(self, x):
                # 如果 x 元素的和大于 0，则对 x 取负
                if bool(torch.sum(x) > 0):
                    x = torch.neg(x)
                return x

        # 创建 ModuleToExport 类的实例
        mte = ModuleToExport()
        # 将 mte 模块导出为 ONNX 格式的字符串，输入为 torch.zeros(1, 2, 3)
        torch.onnx.export_to_pretty_string(mte, (torch.zeros(1, 2, 3),), verbose=False)

    def test_onnx_export_script_inline_params(self):
        # 定义一个内联参数的脚本模块
        class ModuleToInline(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 定义一个形状为 (3, 3) 的参数矩阵 m
                self.m = torch.nn.Parameter(torch.ones(3, 3))
                # 定义一个未使用的参数，形状为 (1, 2, 3)
                self.unused = torch.nn.Parameter(torch.ones(1, 2, 3))

            # 定义一个脚本方法 forward，接受输入 x
            @torch.jit.script_method
            def forward(self, x):
                # 返回 x 和 self.m 的矩阵乘积
                return torch.mm(x, self.m)

        # 定义一个包含内联模块 ModuleToInline 的模块
        class ModuleToExport(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # 实例化 ModuleToInline 类
                self.mod = ModuleToInline()
                # 定义一个形状为 (3, 4) 的参数矩阵 param
                self.param = torch.nn.Parameter(torch.ones(3, 4))

            # 定义一个脚本方法 forward，接受输入 x
            @torch.jit.script_method
            def forward(self, x):
                # 调用 self.mod 的 forward 方法，输入 x，得到输出 y
                y = self.mod(x)
                # 返回 y 和 self.param 的矩阵乘积
                return torch.mm(y, self.param)

        # 创建 ModuleToExport 类的实例
        mte = ModuleToExport()
        # 对 mte 模块执行前向传播，输入为 torch.zeros(2, 3)
        result = mte(torch.zeros(2, 3))
        # 生成参考结果，为两次矩阵乘积的结果
        reference = torch.mm(
            torch.mm(torch.zeros(2, 3), torch.ones(3, 3)), torch.ones(3, 4)
        )
        # 断言模块执行的结果与参考结果相等
        self.assertEqual(result, reference)
        # 将 mte 模块导出为 ONNX 格式的字符串，输入为 torch.ones(2, 3)
        torch.onnx.export_to_pretty_string(mte, (torch.ones(2, 3),), verbose=False)

    def test_onnx_export_speculate(self):
        # 定义一个继承自 torch.jit.ScriptModule 的模块 Foo
        class Foo(torch.jit.ScriptModule):
            def __init__(self, m):
                super().__init__()
                self.m = m

            # 定义一个脚本方法 forward，接受输入 x
            @torch.jit.script_method
            def forward(self, x):
                # 将 x 加倍
                x += x
                # 检查 torch.sum(x) 是否大于 4，并将结果存储在变量 c 中
                c = torch.sum(x) > 4
                # 如果 c 为真，则执行以下代码块
                if bool(c):
                    if bool(c):
                        # 调用 self.m(x) 并将结果赋给 y
                        y = self.m(x)
                    else:
                        # 否则，同样调用 self.m(x) 并将结果赋给 y
                        y = self.m(x)
                else:
                    # 如果 c 为假，则直接调用 self.m(x) 并将结果赋给 y
                    y = self.m(x)
                return y

        # 使用 torch.jit.trace 将 torch.nn.Linear(10, 20) 模块转换为脚本
        linear = torch.jit.trace(
            torch.nn.Linear(10, 20).float(), torch.zeros(1, 10, dtype=torch.float)
        )

        # 定义一个脚本函数 transpose，将输入 x 进行转置操作
        @torch.jit.script
        def transpose(x):
            return x.t()

        # 创建 Foo 类的两个实例 f1 和 f2
        f1 = Foo(transpose)
        f2 = Foo(linear)

        # 将 f1 模块导出为 ONNX 格式的字符串，输入为 torch.ones(1, 10, dtype=torch.float)
        torch.onnx.export_to_pretty_string(f1, (torch.ones(1, 10, dtype=torch.float),))
        # 将 f2 模块导出为 ONNX 格式的字符串，输入为 torch.ones(1, 10, dtype=torch.float)
        torch.onnx.export_to_pretty_string(f2, (torch.ones(1, 10, dtype=torch.float),))
    def test_onnx_export_shape_reshape(self):
        # 定义一个简单的神经网络模块 Foo，用于 ONNX 导出测试
        class Foo(torch.nn.Module):
            def forward(self, x):
                import torch.onnx.operators

                # 将输入张量 x 在第0维度上重复5次，其他维度不变
                x = x.repeat(5, 1, 1)
                # 使用 torch.onnx.operators.shape_as_tensor 获取张量 x 的形状
                shape = torch.onnx.operators.shape_as_tensor(x)
                # 根据给定的形状重新整形张量 x
                reshaped = torch.onnx.operators.reshape_from_tensor_shape(x, shape)
                return reshaped

        # 使用 torch.jit.trace 对 Foo 模块进行跟踪，生成 TorchScript 图形
        foo = torch.jit.trace(Foo(), torch.zeros(1, 2, 3))
        # 将生成的 TorchScript 图形导出为格式化字符串
        torch.onnx.export_to_pretty_string(foo, (torch.zeros(1, 2, 3)))

    def test_listconstruct_erasure(self):
        # 定义一个简单的神经网络模块 FooMod，用于 ONNX 导出测试
        class FooMod(torch.nn.Module):
            def forward(self, x):
                # 创建一个布尔掩码，标记 x 中小于 0.0 的元素
                mask = x < 0.0
                # 返回 x 中满足掩码条件的元素
                return x[mask]

        # 将 FooMod 模块导出为格式化字符串，不添加节点名称，不执行常数折叠
        # 使用 torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK 导出运算符
        torch.onnx.export_to_pretty_string(
            FooMod(),
            (torch.rand(3, 4),),
            add_node_names=False,
            do_constant_folding=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )

    def test_export_dynamic_slice(self):
        # 定义一个动态切片导出模块 DynamicSliceExportMod，继承自 torch.jit.ScriptModule
        class DynamicSliceExportMod(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                retval = x[0]
                for i in range(x.size(1)):
                    # 将 x 在第0维度上从0到i的切片进行求和，dim=0 表示在第0维度上求和
                    retval += torch.sum(x[0:i], dim=0)
                return retval

        mod = DynamicSliceExportMod()

        input = torch.rand(3, 4, 5)

        # 将 DynamicSliceExportMod 模块导出为格式化字符串，设置 opset_version=10
        torch.onnx.export_to_pretty_string(
            DynamicSliceExportMod(), (input,), opset_version=10
        )

    def test_export_dict(self):
        # 定义一个返回字典类型的神经网络模块 DictModule
        class DictModule(torch.nn.Module):
            def forward(self, x_in: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"test_key_out": x_in}

        x_in = torch.tensor(1)
        mod = DictModule()
        mod.train(False)

        # 将 DictModule 模块导出为格式化字符串
        torch.onnx.export_to_pretty_string(mod, (x_in,))

        # 测试运行时错误，确保不支持导出字典构造的模块
        with self.assertRaisesRegex(RuntimeError, r"DictConstruct.+is not supported."):
            torch.onnx.export_to_pretty_string(torch.jit.script(mod), (x_in,))

    def test_source_range_propagation(self):
        # 定义一个扩展模块 ExpandingModule，继承自 torch.nn.Module
        class ExpandingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 将 LayerNorm 应用到模块中的输入
                self.ln = torch.nn.LayerNorm([1])

            def forward(self, input):
                # 返回输入 input 经 LayerNorm 处理后的结果
                return self.ln(input)

        mod = ExpandingModule()

        # 将模块 mod 转换为 TorchScript 图形，并获取导出的操作类型
        graph, _, _ = utils._model_to_graph(
            mod,
            (torch.zeros(1),),
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        )

        # 确保图中的每个节点都有有效的源范围
        for node in graph.nodes():
            self.assertTrue(node.sourceRange())
    def test_clip_aten_fallback_due_exception(self):
        # 定义一个返回特定异常信息的函数，模拟出现异常时的行为
        def bad_clamp(g, self, min, max):
            return symbolic_helper._onnx_unsupported("Bad boy!")

        # 定义一个继承自 torch.nn.Module 的类 MyClip，实现了 forward 方法
        class MyClip(torch.nn.Module):
            def forward(self, x):
                # 使用 torch.clamp 对输入张量 x 进行数值范围的限制
                return torch.clamp(x, min=-0.5, max=0.5)

        # 将 MyClip 模型导出为 ONNX 格式的模型
        onnx_model = export_to_onnx(
            MyClip(),
            torch.randn(3, 4, requires_grad=True),
            # 自定义操作，替换 torch.clamp 的行为为 bad_clamp 函数
            custom_ops=[common_utils.custom_op("aten::clamp", bad_clamp, 17)],
            # 使用 ATen 运算的回退机制导出模型
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        # 断言导出的 ONNX 模型中包含 "clamp" 操作
        self.assertAtenOp(onnx_model, "clamp", "Tensor")

    def test_clip_aten_fallback_explicit_request(self):
        # 定义一个继承自 torch.nn.Module 的类 MyClip，实现了 forward 方法
        class MyClip(torch.nn.Module):
            def forward(self, x):
                # 使用 torch.clamp 对输入张量 x 进行数值范围的限制
                return torch.clamp(x, min=-0.5, max=0.5)

        # 保存原始的注册方法，避免在运行时陷入无限递归
        original_get_function_group = registration.registry.get_function_group

        # 定义一个 mock 函数，用于模拟注册过程中的缺失符号
        def break_is_registered_op_api(name):
            fake_missing_symbolics = {"aten::clamp"}
            if name in fake_missing_symbolics:
                return None
            return original_get_function_group(name)

        # 导出 MyClip 模型为 ONNX 格式的模型，使用自定义的注册方法
        onnx_model = export_to_onnx(
            MyClip(),
            torch.randn(3, 4, requires_grad=True),
            mocks=[
                unittest.mock.patch(
                    "torch.onnx._internal.registration.registry.get_function_group",
                    side_effect=break_is_registered_op_api,
                )
            ],
            # 使用 ATen 运算的回退机制导出模型
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        # 断言导出的 ONNX 模型中包含 "clamp" 操作
        self.assertAtenOp(onnx_model, "clamp", "Tensor")

    def _helper_test_to_(self, cast_fn: Callable[[torch.Tensor], torch.Tensor]):
        """Helper to test aten::to(device) variants.

        `cast_fn` is converted into a `torch.jit.script`. It wraps `aten::to`
        during export to preventing the devices to be hard-coded.

        Needed by detectron2 after https://github.com/facebookresearch/detectron2/pull/4132/
        """
        # 将传入的 cast_fn 函数转换为 torch.jit.script，用于在导出过程中包装 aten::to 操作
        cast_fn = torch.jit.script(cast_fn)
        # 导出 cast_fn 模型为 ONNX 格式的模型
        onnx_model = export_to_onnx(cast_fn, torch.zeros([1, 3, 32, 32]))
        # 遍历 ONNX 模型的节点，断言不包含 "To" 或 "Cast" 操作
        for n in onnx_model.graph.node:
            self.assertNotEqual(n.op_type, "To")
            self.assertNotEqual(n.op_type, "Cast")

    def test_to__cpu_string(self):
        # 定义一个将输入张量转移到 CPU 设备的函数
        def cast_cpu_string(src: torch.Tensor) -> torch.Tensor:
            return src.to("cpu")

        # 调用 _helper_test_to_ 辅助方法，测试 cast_cpu_string 函数
        self._helper_test_to_(cast_cpu_string)

    def test_to__device_cpu_string(self):
        # 定义一个将输入张量转移到 CPU 设备的函数（通过设备名称指定）
        def cast_device_cpu_string(src: torch.Tensor) -> torch.Tensor:
            return src.to(device="cpu")

        # 调用 _helper_test_to_ 辅助方法，测试 cast_device_cpu_string 函数
        self._helper_test_to_(cast_device_cpu_string)
    def test_initializer_sequence(self):
        # 定义一个名为 MyModule 的自定义神经网络模型类，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 初始化方法，接受输入大小 input_size、隐藏层大小 hidden_size、类别数 num_classes 作为参数
            def __init__(self, input_size, hidden_size, num_classes):
                super().__init__()
                # 定义第一个全连接层，输入大小为 input_size，输出大小为 hidden_size
                self.fc1 = torch.nn.Linear(input_size, hidden_size)
                # 定义激活函数 ReLU
                self.relu = torch.nn.ReLU()
                # 定义第二个全连接层，输入大小为 hidden_size，输出大小为 num_classes
                self.fc2 = torch.nn.Linear(hidden_size, num_classes)

            # 前向传播方法，接受输入张量 x 作为参数
            def forward(self, x):
                # 第一层全连接层计算
                out = self.fc1(x)
                # 应用 ReLU 激活函数
                out = self.relu(out)
                # 第二层全连接层计算
                out = self.fc2(out)
                # 返回前向传播的结果张量
                return out

        # 创建 MyModule 类的实例 test_model，输入大小为 3，隐藏层大小为 4，类别数为 10
        test_model = MyModule(3, 4, 10)
        # 获取模型状态字典中所有键的列表
        state_dict_list = [k for (k, v) in test_model.state_dict().items()]
        # 获取模型命名参数中所有键的列表
        named_params_list = [k for (k, v) in test_model.named_parameters()]

        # 创建一个形状为 [32, 3] 的随机张量 x
        x = torch.randn(32, 3)
        # 创建一个字节流对象 f
        f = io.BytesIO()
        # 将模型 test_model 导出为 ONNX 格式，关闭常量折叠
        torch.onnx.export(test_model, (x,), f, do_constant_folding=False)
        # 从字节流中加载 ONNX 模型
        loaded_model = onnx.load_from_string(f.getvalue())

        # 获取加载的 ONNX 模型中初始化器的名称列表
        actual_list = [p.name for p in loaded_model.graph.initializer]
        # 断言加载的 ONNX 模型中的初始化器顺序与模型状态字典中的键顺序相同
        assert actual_list == state_dict_list, (
            "Initializers' sequence is not as same as state_dict(). Expected: ("
            + ", ".join(state_dict_list)
            + "). Actual:("
            + ", ".join(actual_list)
            + ")."
        )
        # 断言加载的 ONNX 模型中的初始化器顺序与模型命名参数中的键顺序相同
        assert actual_list == named_params_list, (
            "Initializers' sequence is not as same as named_parameters(). Expected: ("
            + ", ".join(named_params_list)
            + "). Actual:("
            + ", ".join(actual_list)
            + ")."
        )
    # 定义一个测试函数，用于验证初始化顺序是否与 state_dict() 方法返回的初始化顺序相同
    def test_initializer_sequence_script_model(self):
        
        # 定义一个辅助函数，用于检查短列表中的元素是否依次出现在长列表中对应位置上
        def list_is_expected(short_list, long_list) -> bool:
            # 如果短列表比长列表长，则返回 False
            if len(short_list) > len(long_list):
                return False
            
            # 遍历短列表
            for i in range(len(short_list)):
                # 如果短列表当前位置的元素不在长列表对应位置的元素中，则返回 False
                if short_list[i] not in long_list[i]:
                    return False
            
            # 如果所有条件都满足，则返回 True
            return True
        
        # 定义一个简单的循环函数，用于将 x 和 y 参数相加多次
        def loop(x, y):
            # 循环 y 次，将每次循环的索引 i 累加到 x 上
            for i in range(int(y)):
                x = x + i
            return x
        
        # 定义一个继承自 torch.nn.Module 的模型类
        class MyModule(torch.nn.Module):
            # 初始化方法，接收输入大小 input_size、隐藏层大小 hidden_size 和类别数 num_classes 作为参数
            def __init__(self, input_size, hidden_size, num_classes):
                super().__init__()
                # 创建第一个全连接层，将输入大小映射到隐藏层大小
                self.fc1 = torch.nn.Linear(input_size, hidden_size)
                # 创建 ReLU 激活函数实例
                self.relu = torch.nn.ReLU()
                # 创建第二个全连接层，将隐藏层输出映射到类别数
                self.fc2 = torch.nn.Linear(hidden_size, num_classes)
            
            # 前向传播方法，接收输入张量 x 和标量 y 作为参数
            def forward(self, x, y):
                # 调用外部定义的 loop 函数处理输入 x 和 y
                x = loop(x, y)
                # 将处理后的 x 输入到第一个全连接层中
                out = self.fc1(x)
                # 将第一个全连接层的输出应用 ReLU 激活函数
                out = self.relu(out)
                # 将激活后的输出输入到第二个全连接层中
                out = self.fc2(out)
                # 返回最终的输出结果
                return out
        
        # 使用 torch.jit.script 方法将 MyModule 类实例化为一个脚本化的模型
        test_model = torch.jit.script(MyModule(3, 4, 10))
        
        # 获取 test_model 的状态字典的键值列表
        state_dict_list = [k for (k, v) in test_model.state_dict().items()]
        # 获取 test_model 的命名参数列表
        named_params_list = [k for (k, v) in test_model.named_parameters()]
        
        # 创建一个全为 1 的浮点型张量 x，形状为 (2, 3)
        x = torch.ones(2, 3, dtype=torch.float)
        # 创建一个值为 5 的长整型张量 y
        y = torch.tensor(5, dtype=torch.long)
        # 创建一个空的字节流对象 f
        f = io.BytesIO()
        
        # 使用 torch.onnx.export 方法将 test_model 导出为 ONNX 格式，存储在字节流对象 f 中
        torch.onnx.export(test_model, (x, y), f, do_constant_folding=False)
        # 从字节流对象 f 中加载模型定义并存储在 loaded_model 中
        loaded_model = onnx.load_from_string(f.getvalue())
        
        # 获取 loaded_model 的图初始化器列表中的名称列表
        actual_list = [p.name for p in loaded_model.graph.initializer]
        
        # 断言 loaded_model 的初始化器列表与 state_dict() 方法返回的列表顺序是否一致
        assert list_is_expected(state_dict_list, actual_list), (
            "ScriptModel - Initializers' sequence is not as same as state_dict(). Expected: ("
            + ", ".join(state_dict_list)
            + "). Actual:("
            + ", ".join(actual_list)
            + ")."
        )
        
        # 断言 loaded_model 的初始化器列表与 named_parameters() 方法返回的列表顺序是否一致
        assert list_is_expected(named_params_list, actual_list), (
            "ScriptModel - Initializers' sequence is not as same as named_parameters(). Expected: ("
            + ", ".join(named_params_list)
            + "). Actual:("
            + ", ".join(actual_list)
            + ")."
        )
    # 定义一个测试方法，用于测试 ONNX 检查器对无效图形的处理
    def test_onnx_checker_invalid_graph(self):
        # 定义一个自定义的 Torch 模块，实现加法操作
        class CustomAddModule(torch.nn.Module):
            def forward(self, x, y):
                return torch.add(x, y)

        # 定义一个符号化的自定义无效加法操作，用于 ONNX 导出
        def symbolic_custom_invalid_add(g, input, other, alpha=None):
            return g.op("Add", input, other, invalid_attr_i=1)

        # 注册自定义的符号化操作到 ONNX
        torch.onnx.register_custom_op_symbolic(
            "::add", symbolic_custom_invalid_add, opset_version=9
        )

        # 创建输入张量 x 和 y
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)

        # 实例化自定义的加法模块
        test_model = CustomAddModule()

        # 创建一个字节流对象
        f = io.BytesIO()

        try:
            # 使用 assertRaises 检查器错误，确保导出时出现错误
            with self.assertRaises(torch.onnx.errors.CheckerError):
                torch.onnx.export(test_model, (x, y), f, opset_version=9)
        finally:
            # 最终解除注册的自定义符号化操作
            torch.onnx.unregister_custom_op_symbolic("::add", 9)

        # 断言字节流对象 f 内容不为空，表示 ONNX 图未成功导出
        self.assertTrue(f.getvalue(), "ONNX graph was not exported.")

        # 加载通过字节流 f 导出的 ONNX 模型
        loaded_model = onnx.load_from_string(f.getvalue())

    # 定义一个测试方法，用于测试 ONNX 导出时的形状值映射
    def test_shape_value_map(self):
        # 定义一个包含 RSoftMax 模块的 Torch 模型
        class RSoftMax(torch.nn.Module):
            def __init__(self, radix, cardinality):
                super().__init__()
                self.radix = radix
                self.cardinality = cardinality

            def forward(self, x):
                batch = x.size(0)
                x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
                x = F.softmax(x, dim=1)
                x = x.reshape(batch, -1)
                return x

        # 定义模型的参数 radix 和 cardinality
        radix = 2
        cardinality = 1

        # 创建输入张量 x
        x = torch.randn(10, 1, 128, 1)

        # 创建一个字节流对象
        f = io.BytesIO()

        # 使用 Torch 导出 RSoftMax 模型到 ONNX 格式
        torch.onnx.export(
            RSoftMax(radix, cardinality),
            (x,),
            f,
            input_names=["x"],
            dynamic_axes={"x": [0]},
        )

        # 加载通过字节流 f 导出的 ONNX 模型
        loaded_model = onnx.load_from_string(f.getvalue())

        # 断言加载的模型输出维度的第二个维度长度为 128
        self.assertEqual(
            loaded_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value, 128
        )

    # 定义一个测试方法，用于测试 ONNX 协议检查器
    def test_onnx_proto_checker(self):
        # 定义一个简单的 Torch 模型，实现输入张量的乘法操作
        class Model(torch.nn.Module):
            def forward(self, x):
                return 2 * x

        # 创建输入张量 x
        x = torch.randn(1, 2, 3, requires_grad=True)

        # 创建一个字节流对象
        f = io.BytesIO()

        # 使用 Torch 导出模型到 ONNX 格式
        torch.onnx.export(Model(), x, f)

        # 从字节流 f 加载 ONNX 模型
        model = onnx.load(f)

        # 设置模型的 IR 版本为 0
        model.ir_version = 0

        # 定义一个函数检查 ONNX 协议
        def check_proto():
            torch._C._check_onnx_proto(model.SerializeToString())

        # 使用 assertRaises 检查 RuntimeError 是否被抛出，表示 ONNX 协议不符合预期
        self.assertRaises(RuntimeError, check_proto)
    def test_maintain_dynamic_shapes_of_unreliable_nodes(self):
        # 定义一个符号化 Python 操作的函数，用于自定义 ONNX 符号操作
        def symbolic_pythonop(ctx: torch.onnx.SymbolicContext, g, *args, **kwargs):
            return g.op("com.microsoft::PythonOp")

        # 注册自定义的 ONNX 符号操作 "prim::PythonOp"
        torch.onnx.register_custom_op_symbolic("prim::PythonOp", symbolic_pythonop, 1)
        # 在测试完成后取消注册自定义的 ONNX 符号操作 "prim::PythonOp"
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "prim::PythonOp", 1)

        # Transformer embeddings 所需的参数
        hidden_size = 48
        max_position_embeddings = 32
        batch_size = 2

        # 发现问题：autograd.function 使得下游节点不可靠，但形状是静态的。
        # 首次发现问题是在 Transformers 中使用 Apex FusedLayerNorm 时。
        class CustomLayerNorm(torch.autograd.Function):
            @staticmethod
            def forward(ctx, embedding):
                # 创建 LayerNorm 实例
                layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
                # 应用 LayerNorm 到输入 embedding
                return layer_norm(embedding)

        # 定义一个包含自定义层 CustomLayerNorm 的模块
        class EmbeddingModule(torch.nn.Module):
            def forward(
                self,
                embeddings=None,
            ):
                # 应用 CustomLayerNorm 到输入 embeddings
                embedding_output = CustomLayerNorm.apply(embeddings)
                # 调整 query 的维度顺序
                query = embedding_output.transpose(0, 1)
                # 获取 query 的维度信息
                target_len, batch_size, embedding_dim = query.size()
                # 使用 reshape 操作以消耗 batch_size，如果 batch_size 是静态的，
                # 这将在图中生成一个常量节点
                query = query.reshape(target_len, batch_size, embedding_dim)
                return query

        # 创建随机的 embeddings，形状为 (batch_size, max_position_embeddings, hidden_size)
        embeddings = torch.randn(batch_size, max_position_embeddings, hidden_size)

        # 导出 EmbeddingModule 模块到 ONNX 格式
        f = io.BytesIO()
        torch.onnx.export(
            EmbeddingModule().eval(),
            (embeddings,),
            f,
            input_names=["embeddings"],
            dynamic_axes={
                "embeddings": {
                    0: "batch_size",
                    1: "max_position_embeddings",
                    2: "hidden_size",
                }
            },
            custom_opsets={"com.microsoft": 1},
        )
        # 加载导出的 ONNX 模型
        model = onnx.load(io.BytesIO(f.getvalue()))

        # 如果模型图中存在一个 op_type 为 "Constant" 的常量节点，并且其形状为 [max_position_embeddings, batch_size, hidden_size]，
        # 这表示形状已经变成了静态的。通常情况下，对于动态的 batch_size，这样的常量节点不应该存在。
        const_node = [n for n in model.graph.node if n.op_type == "Constant"]
        self.assertNotEqual(len(const_node), 0)
        for node in const_node:
            for a in node.attribute:
                if a.name == "value":
                    shape = onnx.numpy_helper.to_array(a.t)
                    self.assertNotEqual(
                        shape.tolist(),
                        [max_position_embeddings, batch_size, hidden_size],
                    )
    # 定义一个测试函数，用于测试是否为C_TypeList
    def test_is_fp_for_C_TypeList(self):
        # 定义一个继承自torch.nn.Module的类M
        class M(torch.nn.Module):
            # 定义forward方法
            def forward(self, x):
                # 压缩x的第一个维度
                x = x.squeeze(1)
                # 获取x的第三个维度的大小
                w = x.shape[2]
                # 将x变形为2行，-1列的矩阵，找到每行最大值的索引
                pos = x.view(2, -1).argmax(1)
                # 计算x_int和y_int
                x_int = pos % w
                y_int = (pos - x_int) // w
                return y_int, x_int

        # 使用torch.jit.script对M类进行脚本化
        model = torch.jit.script(M())
        # 生成一个2x4x6的随机输入
        inputs = torch.randn(2, 4, 6)
        # 创建一个BytesIO对象
        f = io.BytesIO()
        # 导出模型到ONNX格式
        torch.onnx.export(
            model, inputs, f, dynamic_axes={"x": [0, 1]}, input_names=["x"]
        )

    # 测试dropout脚本化
    def test_dropout_script(self):
        # 创建一个requires_grad为True的1x2x3的全零张量
        eg = torch.zeros(1, 2, 3, requires_grad=True)

        # 使用jit_utils._trace对函数进行追踪
        @jit_utils._trace(eg)
        def foo(x):
            # 对x取负
            x = torch.neg(x)
            # 对x进行dropout
            return F.dropout(x)

        # 定义一个MyDrop类继承自torch.nn.Module
        class MyDrop(torch.nn.Module):
            def forward(self, x):
                return foo(x)

        # 创建一个BytesIO对象
        f = io.BytesIO()
        # 导出MyDrop模型到ONNX格式
        with warnings.catch_warnings(record=True):
            torch.onnx.export(MyDrop(), (eg,), f, verbose=False)

    # 测试pack_padded_sequence和pad_packed_sequence的追踪
    def test_pack_padded_pad_packed_trace(self):
        # 导入pack_padded_sequence和pad_packed_sequence函数
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        T, B, C = 3, 5, 7

        # 定义一个PadPackedWrapper类继承自torch.nn.Module
        class PadPackedWrapper(torch.nn.Module):
            def forward(self, x, seq_lens):
                # 对输入进行pack_padded_sequence
                x = pack_padded_sequence(x, seq_lens)
                # 对输入进行pad_packed_sequence
                x, _ = pad_packed_sequence(x)
                return x

        # 创建一个TxBxC的全1张量x和长度为[3, 3, 2, 2, 1]的seq_lens
        x = np.ones((T, B, C))
        seq_lens = np.array([3, 3, 2, 2, 1], dtype=np.int32)
        # 设置填充值以便测试等价性
        for b in range(B):
            if seq_lens[b] < T:
                x[seq_lens[b] :, b, :] = 0
        seq_lens = torch.from_numpy(seq_lens)
        x = torch.autograd.Variable(torch.from_numpy(x), requires_grad=True)

        # 创建PadPackedWrapper实例m
        m = PadPackedWrapper()
        # 对m进行追踪
        m_traced = torch.jit.trace(
            m,
            (
                x,
                seq_lens,
            ),
        )

        # 对m进行前向传播
        y = m(x, seq_lens)
        loss = torch.sum(y)
        loss.backward()
        grad = x.grad.clone()
        x.grad.zero_()

        # 对追踪后的m进行前向传播
        y_traced = m_traced(x, seq_lens)
        loss_traced = torch.sum(y_traced)
        loss_traced.backward()
        grad_traced = x.grad.clone()

        # 断言y_traced和x相等
        self.assertEqual(y_traced, x)
        # 断言y_traced和y相等
        self.assertEqual(y_traced, y)
        # 断言grad和grad_traced相等
        self.assertEqual(grad, grad_traced)

        # 创建一个BytesIO对象
        f = io.BytesIO()
        # 导出m模型到ONNX格式
        torch.onnx.export(m, (x, seq_lens), f, verbose=False)

    # ONNX在导出RNN时会发出警告，因为可能存在批大小不匹配的情况
    @common_utils.suppress_warnings
    def test_rnn_trace_override(self):
        # 导入需要的函数和模块
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        
        # 定义 RNN 的层数、时间步长 T、批次大小 B、特征数 C
        num_layers = 3
        T, B, C = 11, 5, 7

        # 定义 RNN 的包装类 RNNTraceWrapper
        class RNNTraceWrapper(torch.nn.Module):
            def __init__(self, cell_type):
                super().__init__()
                # 根据不同的 cell_type 初始化不同的 RNN 类型
                if cell_type == "RNN":
                    self.rnn = torch.nn.RNN(
                        input_size=C, hidden_size=C, num_layers=num_layers
                    )
                elif cell_type == "LSTM":
                    self.rnn = torch.nn.LSTM(
                        input_size=C, hidden_size=C, num_layers=num_layers
                    )
                elif cell_type == "GRU":
                    self.rnn = torch.nn.GRU(
                        input_size=C, hidden_size=C, num_layers=num_layers
                    )

            def forward(self, x, seq_lens):
                # 对输入 x 按照 seq_lens 进行填充和压缩
                x = pack_padded_sequence(x, seq_lens)
                # 将 x 输入 RNN 模型进行前向计算
                x, _ = self.rnn(x)
                # 对输出 x 进行填充解压缩
                x, _ = pad_packed_sequence(x)
                return x

        # 针对每种 RNN 类型进行测试
        for cell_type in ["RNN", "LSTM", "GRU"]:
            # 创建输入数据 x 和 seq_lens
            x = torch.ones(T, B, C, requires_grad=True)
            seq_lens = torch.from_numpy(np.array([11, 3, 2, 2, 1], dtype=np.int32))

            # 实例化 RNNTraceWrapper 类
            m = RNNTraceWrapper(cell_type)
            # 对模型进行追踪
            m_traced = torch.jit.trace(
                m,
                (
                    x,
                    seq_lens,
                ),
            )

            # 使用原始模型计算输出 y 和损失 loss
            y = m(x, seq_lens)
            loss = torch.sum(y)
            # 反向传播计算梯度
            loss.backward()
            grad = x.grad.clone()
            x.grad.zero_()

            # 使用追踪模型计算输出 y_traced 和损失 loss_traced
            y_traced = m_traced(x, seq_lens)
            loss_traced = torch.sum(y_traced)
            # 反向传播计算追踪模型的梯度
            loss_traced.backward()
            grad_traced = x.grad.clone()

            # 断言追踪模型的输出与原始模型的输出一致
            self.assertEqual(y_traced, y)
            # 断言原始模型和追踪模型的梯度一致
            self.assertEqual(grad, grad_traced)

            # 将模型导出为 ONNX 格式并保存在字节流中
            f = io.BytesIO()
            torch.onnx.export(m, (x, seq_lens), f, verbose=False)
    def test_pushpackingpastrnn_in_peephole_create_own_gather_input(self):
        # 导入需要的模块和函数
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        # 定义变量：LSTM 层的层数、时间步长 T、批量大小 B、输入特征维度 C、掩码起始点
        num_layers = 3
        T, B, C = 11, 5, 7
        mask_start_point = 0

        # 定义一个自定义的 LSTM 封装类
        class LSTMTraceWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()

                # 在初始化方法中定义一个 LSTM 层
                self.rnn = torch.nn.LSTM(
                    input_size=C, hidden_size=C, num_layers=num_layers
                )

            def forward(self, x, seq_lens):
                # 创建一个掩码，从输入中选择有效的序列长度
                mask = torch.arange(mask_start_point, x.shape[1])
                seq_lens = seq_lens[mask]
                # 对输入序列进行填充并打包
                x = pack_padded_sequence(x, seq_lens)
                # 准备用于传递给 LSTM 初始状态的零缓冲区视图
                max_batch_size = x.batch_sizes[0]
                hx = torch.randn(num_layers, max_batch_size, C)
                cx = torch.randn(num_layers, max_batch_size, C)
                # 在 LSTM 层上执行前向传播
                x, _ = self.rnn(x, (hx, cx))
                # 解压填充后的序列
                x, _ = pad_packed_sequence(x)
                return x

        # 创建输入张量 x 和序列长度张量 seq_lens
        x = torch.ones(T, B, C)
        seq_lens = torch.from_numpy(np.array([11, 3, 2, 2, 1], dtype=np.int32))
        # 创建 LSTMTraceWrapper 实例
        m = LSTMTraceWrapper()

        # 准备用于 ONNX 导出的字节流对象
        f = io.BytesIO()
        # 使用 torch.onnx.export 导出模型到 ONNX 格式
        torch.onnx.export(
            m,
            (x, seq_lens),
            f,
            verbose=True,
            input_names=["input", "seq_len"],
            dynamic_axes={"input": {1: "B"}},
        )
        # 从导出的 ONNX 模型字符串加载模型
        onnx_proto = onnx.load_model_from_string(f.getvalue())

        # 查找 ONNX 图中的常量节点和 Range 操作的第一个输入
        const_node = []
        constant_input_name = None
        for n in onnx_proto.graph.node:
            if n.op_type == "Constant":
                const_node.append(n)
            elif n.op_type == "Range":
                constant_input_name = n.input[0]
        # 断言确保找到了常量节点和 Range 操作的第一个输入
        self.assertNotEqual(constant_input_name, None)
        self.assertNotEqual(len(const_node), 0)

        # 从常量节点中获取特定名称的输出值
        value = None
        for n in const_node:
            if n.output[0] == constant_input_name:
                value = np.frombuffer(n.attribute[0].t.raw_data, dtype=np.int64)
        # 断言确保值等于 0
        self.assertEqual(value, 0)

    def test_trace_fork_wait_inline_onnx(self):
        # 定义一个函数 fork_body，对输入张量执行负值操作
        def fork_body(x):
            return torch.neg(x), torch.neg(x)

        # 定义一个简单的 Module 类 MyMod，实现 fork 和 wait 操作
        class MyMod(torch.nn.Module):
            def forward(self, x):
                fut = torch.jit._fork(fork_body, x)
                val = torch.jit._wait(fut)
                return val[1]

        # 对 MyMod 模型进行 ONNX 导出的简单测试
        f = io.BytesIO()
        torch.onnx.export(MyMod(), (torch.rand(3, 4),), f)

    def test_trace_detach_onnx_erase(self):
        # 定义一个简单的 Module 类 Mod，执行输入张量与权重张量的矩阵乘法，并将结果进行 detach 处理
        class Mod(torch.nn.Module):
            def forward(self, x, w):
                return torch.matmul(x, w).detach()

        # 使用 torch.onnx.export_to_pretty_string 导出 Mod 模型到 ONNX 格式的字符串
        torch.onnx.export_to_pretty_string(Mod(), (torch.rand(3, 4), torch.rand(4, 5)))
    # 定义一个测试方法，用于验证在 ATen 操作无法导出为 ONNX 时必须进行回退的情况
    def test_aten_fallback_must_fallback(self):
        # 定义一个不包含 ONNX 可导出操作的模型类
        class ModelWithAtenNotONNXOp(torch.nn.Module):
            def forward(self, x, y):
                # 执行张量加法操作
                abcd = x + y
                # 执行 torch.linalg.qr() 操作
                defg = torch.linalg.qr(abcd)
                return defg

        # 创建随机张量作为输入
        x = torch.rand(3, 4)
        y = torch.rand(3, 4)
        # 创建一个字节流对象
        f = io.BytesIO()
        # 将模型导出为 ONNX 格式，指定 ATen 回退策略
        torch.onnx.export(
            ModelWithAtenNotONNXOp(),
            (x, y),
            f,
            do_constant_folding=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            # 指定 opset 版本为 9，因为 linalg.qr 在较高版本的 opset 中支持
            opset_version=9,
        )
        # 加载导出的 ONNX 模型
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        # 断言是否存在特定的 ATen 操作节点
        self.assertAtenOp(onnx_model, "linalg_qr")

    # 定义一个测试方法，用于验证在 ATen 操作可导出为 ONNX 时的情况
    def test_onnx_aten(self):
        # 定义一个包含 ONNX 可导出操作的模型类
        class ModelWithAtenFmod(torch.nn.Module):
            def forward(self, x, y):
                # 执行 torch.fmod() 操作
                return torch.fmod(x, y)

        # 创建随机浮点数张量作为输入
        x = torch.randn(3, 4, dtype=torch.float32)
        y = torch.randn(3, 4, dtype=torch.float32)
        # 创建一个字节流对象
        f = io.BytesIO()
        # 将模型导出为 ONNX 格式，指定 ATen 导出策略
        torch.onnx.export(
            ModelWithAtenFmod(),
            (x, y),
            f,
            do_constant_folding=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN,
        )
        # 加载导出的 ONNX 模型
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        # 断言是否存在特定的 ATen 操作节点
        self.assertAtenOp(onnx_model, "fmod", "Tensor")

    # 定义一个测试方法，用于验证在 ATen 操作无法回退为 ONNX 时不进行回退的情况
    def test_onnx_aten_fallback_must_not_fallback(self):
        # 定义一个可以导出为 ONNX 的模型类
        class ONNXExportable(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.fc1 = torch.nn.Linear(12, 8)
                self.fc2 = torch.nn.Linear(8, 4)
                self.fc3 = torch.nn.Linear(4, 6)
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                # 执行量化和反量化操作
                x = self.quant(x)
                x = x.view((-1, 12))
                h = F.relu(self.fc1(x))
                h = F.relu(self.fc2(h))
                h = F.relu(self.fc3(h))
                h = self.dequant(h)
                return h

        # 创建一个随机输入张量
        dummy_input = torch.randn(12)
        # 创建一个字节流对象
        f = io.BytesIO()
        # 将模型导出为 ONNX 格式，指定 ATen 回退策略
        torch.onnx.export(
            ONNXExportable(),
            (dummy_input,),
            f,
            do_constant_folding=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        # 加载导出的 ONNX 模型
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        # 获取所有的 ATen 节点，并断言其数量为零
        all_aten_nodes = [
            p
            for p in onnx_model.graph.node
            if p.op_type == "ATen" and p.domain == "org.pytorch.aten"
        ]
        self.assertEqual(len(all_aten_nodes), 0)
    # 定义一个测试方法，用于测试在空张量情况下的 torch.cat 操作
    def test_cat_with_empty_tensor(self):
        # 定义一个简单的 PyTorch 模块，将输入张量与空张量进行连接
        class NoopConcat(torch.nn.Module):
            def forward(self, x):
                return torch.cat((torch.Tensor([]), x))

        # 创建一个形状为 (4, 5, 6) 的随机张量 x
        x = torch.randn(4, 5, 6)
        # 循环遍历 opset_version 参数集合 {9, 11}
        for opset_version in {9, 11}:
            # 创建一个 BytesIO 对象 f 用于存储导出的 ONNX 模型
            f = io.BytesIO()
            # 将 NoopConcat 模块导出为 ONNX 模型，并写入 BytesIO 对象 f
            torch.onnx.export(NoopConcat(), (x,), f, opset_version=opset_version)
            # 从导出的字节数据中加载 ONNX 模型
            loaded_model = onnx.load_from_string(f.getvalue())
            # 断言加载的 ONNX 模型输出的张量形状维度为 3
            self.assertEqual(
                len(loaded_model.graph.output[0].type.tensor_type.shape.dim), 3
            )
            # 断言加载的 ONNX 模型输出的每个维度与原始张量 x 的对应维度相等
            for idx, dim in enumerate(x.shape):
                self.assertEqual(
                    loaded_model.graph.output[0]
                    .type.tensor_type.shape.dim[idx]
                    .dim_value,
                    dim,
                )

    # 定义一个测试方法，用于测试 torch.nn.Unfold 模块的使用情况
    def test_col2im(self):
        # 定义一个随机批量的 RGB 图像输入张量，形状为 (64, 3, 32, 32)
        original_image_inputs = torch.randn((64, 3, 32, 32))
        # 定义输出大小为原始图像输入的大小 (32, 32)
        output_size = tuple(original_image_inputs.shape[2:])
        # 定义 Unfold 模块，用于将输入图像块化
        kernel_size = (1, 2)
        dilation = 3
        padding = 2
        stride = 1
        model_im2col = torch.nn.Unfold(
            kernel_size, dilation=dilation, padding=padding, stride=stride
        )
        # 对原始图像输入应用 Unfold 模块，得到图像块
        blocks = model_im2col(original_image_inputs)

        # 定义 Fold 模块，用于将图像块重构为原始图像
        model = torch.nn.Fold(
            output_size=output_size,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        f = io.BytesIO()
        # 将 Fold 模块导出为 ONNX 模型，并写入 BytesIO 对象 f
        torch.onnx.export(model, (blocks,), f, opset_version=18)

        # 加载导出的 ONNX 模型
        onnx_model = onnx.load(io.BytesIO(f.getvalue()))
        # 断言 ONNX 模型最后一个节点的操作类型为 "Col2Im"
        self.assertEqual(onnx_model.graph.node[-1].op_type, "Col2Im")
        # 断言 ONNX 模型最后一个节点的域为空字符串
        self.assertEqual(onnx_model.graph.node[-1].domain, "")
        # 断言 ONNX 模型最后一个节点的输入张量数为 3
        self.assertEqual(len(onnx_model.graph.node[-1].input), 3)
        # 断言 ONNX 模型最后一个节点的属性中包含 "dilations" 属性
        self.assertEqual(onnx_model.graph.node[-1].attribute[0].name, "dilations")
        # 断言 ONNX 模型最后一个节点的属性中包含 "pads" 属性
        self.assertEqual(onnx_model.graph.node[-1].attribute[1].name, "pads")
        # 断言 ONNX 模型最后一个节点的属性中包含 "strides" 属性
        self.assertEqual(onnx_model.graph.node[-1].attribute[2].name, "strides")

    # 使用 unittest 装饰器跳过测试，如果未安装 torch_scatter 模块
    @unittest.skipIf(
        not torch.hub._check_module_exists("torch_scatter"),
        "torch_scatter not installed.",
    )
    @common_utils.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    # 使用 common_utils.parametrize 装饰器，定义了一个参数化测试函数，用于测试不同的 fp8_dtype 类型
    def test_fp8_export(self, fp8_dtype: torch.dtype):
        # 定义了一个测试类方法，用于测试将 fp8_dtype 类型的数据导出为 ONNX 格式
        class Model(torch.nn.Module):
            def forward(self, x):
                return x.to(torch.float32)

        # 创建一个 fp8_dtype 类型的输入张量 x
        x = torch.randn(2, 3).to(fp8_dtype)

        # 创建一个字节流对象 f
        f = io.BytesIO()
        # 将 Model 类实例化后的模型以 ONNX 格式导出到字节流 f，使用 opset_version=19
        torch.onnx.export(Model(), x, f, opset_version=19)
        # 对导出的 ONNX 模型进行模型检查
        onnx.checker.check_model(f.getvalue())

        # 定义了一个字典 onnx_type，将 fp8_dtype 映射到相应的 ONNX 数据类型
        onnx_type = {
            torch.float8_e4m3fn: 17,
            torch.float8_e5m2: 19,
        }  # From https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3#L512-L521
        # 从字节流 f 中加载 ONNX 模型
        loaded_model = onnx.load_from_string(f.getvalue())
        # 断言加载的 ONNX 模型的输入张量类型与对应 fp8_dtype 类型的映射相符
        self.assertEqual(
            loaded_model.graph.input[0].type.tensor_type.elem_type, onnx_type[fp8_dtype]
        )
# 定义一个测试类 TestQuantizeEagerONNXExport，继承自 common_utils.TestCase
class TestQuantizeEagerONNXExport(common_utils.TestCase):

    # 定义测试方法 _test_lower_graph_impl，接受模型和数据作为参数
    def _test_lower_graph_impl(self, model, data):
        # 设置模型的量化配置为默认配置
        model.qconfig = torch.ao.quantization.default_qconfig
        # 准备模型以便量化
        model = torch.ao.quantization.prepare(model)
        # 将模型转换为量化版本
        model = torch.ao.quantization.convert(model)

        # 对模型进行推断
        _ = model(data)
        # 设置输入的名称列表
        input_names = ["x"]

        # 定义函数 _export_to_onnx，接受模型、输入和输入名称作为参数
        def _export_to_onnx(model, input, input_names):
            # 对模型进行追踪以获取 TorchScript 表示
            traced = torch.jit.trace(model, input)
            # 创建一个字节流对象
            buf = io.BytesIO()
            # 将追踪到的模型保存到字节流中
            torch.jit.save(traced, buf)
            buf.seek(0)

            # 从字节流中加载模型
            model = torch.jit.load(buf)
            # 创建另一个字节流对象
            f = io.BytesIO()
            # 导出模型到 ONNX 格式
            torch.onnx.export(
                model,
                input,
                f,
                input_names=input_names,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                opset_version=9,
            )

        # 调用 _export_to_onnx 函数，导出模型到 ONNX 格式
        _export_to_onnx(model, data, input_names)

    # 标记装饰器，如果没有 FBGEMM 支持则跳过测试
    @common_quantization.skipIfNoFBGEMM
    @unittest.skip(
        "onnx opset9 does not support quantize_per_tensor and caffe2 \
    does not support conv3d"
    )
    # 定义测试方法 test_lower_graph_conv3d，测试量化下的 Conv3d 操作
    def test_lower_graph_conv3d(self):
        # 创建一个量化封装的 Conv3d 模型
        model = torch.ao.quantization.QuantWrapper(
            torch.nn.Conv3d(3, 5, 2, bias=True)
        ).to(dtype=torch.float)
        # 创建随机数据作为输入
        data_numpy = np.random.rand(1, 3, 6, 6, 6).astype(np.float32)
        data = torch.from_numpy(data_numpy).to(dtype=torch.float)
        # 执行具体的下降图测试实现
        self._test_lower_graph_impl(model, data)

    # 标记装饰器，如果没有 CUDA 支持则跳过测试
    @pytorch_test_common.skipIfNoCuda
    # 定义测试方法 test_composed_layer_norm_small_eps_fp16_keep_double，测试组合层归一化
    def test_composed_layer_norm_small_eps_fp16_keep_double(self):
        # 定义一个简单的神经网络模型，包含层归一化
        class Net(torch.nn.Module):
            def __init__(self, C):
                super().__init__()
                self.layer_norm = torch.nn.LayerNorm(C, eps=1e-8)

            def forward(self, x):
                return self.layer_norm(x)

        # 设置 N 和 C 的值
        N, C = 8, 4
        # 创建具有 CUDA 和半精度数据类型的模型
        model = Net(C).cuda().half()
        # 创建 CUDA 和半精度数据类型的输入
        x = torch.randn(N, C).cuda().half()
        # 创建一个字节流对象
        f = io.BytesIO()
        # 将模型导出为 ONNX 格式
        torch.onnx.export(model, x, f, opset_version=14)
        # 从字节流中加载 ONNX 模型
        onnx_model = onnx.load_from_string(f.getvalue())
        # 获取 ONNX 图中的常量节点
        const_node = [n for n in onnx_model.graph.node if n.op_type == "Constant"]
        # 断言：常量节点的数量不为零
        self.assertNotEqual(len(const_node), 0)
        # 计数双精度类型常量的数量
        double_type_count = 0
        for node in const_node:
            for a in node.attribute:
                # 断言：EPS 常量应该是双精度类型
                if a.name == "value" and a.t.data_type == 11:
                    double_type_count += 1
        # 断言：双精度类型常量的数量不为零
        self.assertNotEqual(double_type_count, 0)

    # 标记装饰器，如果没有 CUDA 支持则跳过测试
    @pytorch_test_common.skipIfNoCuda
    def test_aten_device_with_index(self):
        # 导入需要的库和模型
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # 加载预训练的tokenizer和Seq2Seq模型
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        
        # 将模型编译为ONNX格式，并选择后端为onnxruntime
        model = torch.compile(model, backend="onnxrt")
        
        # 设置模型为评估模式（不计算梯度）
        model = model.eval()
        
        # 将模型移动到指定的GPU设备上（cuda:0）
        device = "cuda:0"
        model = model.to(device)
        
        # 使用tokenizer编码输入文本，并将其转移到GPU设备上
        ids = tokenizer.batch_encode_plus(["This is a test"], return_tensors="pt").to(
            device
        )

        # 在不计算梯度的上下文中，使用模型进行推理
        with torch.no_grad():
            _ = model(
                input_ids=ids["input_ids"],
                attention_mask=ids["attention_mask"],
                decoder_input_ids=ids["input_ids"],
                decoder_attention_mask=ids["attention_mask"],
            )

    def test_aten_linalg_vector_norm_with_reducel2(self):
        # 定义一个简单的神经网络模型
        class Net(torch.nn.Module):
            def forward(self, x):
                # 对输入进行向量归一化操作
                x = F.normalize(x)
                return x

        # 创建一个字节流对象
        f = io.BytesIO()
        
        # 将模型导出为ONNX格式并保存到字节流中
        torch.onnx.export(Net(), (torch.randn(1, 2, 2),), f)
        
        # 从字节流中加载ONNX模型
        onnx_model = onnx.load_from_string(f.getvalue())
        
        # 获取ONNX模型中所有节点的操作类型
        onnx_nodes = [n.op_type for n in onnx_model.graph.node]
        
        # 断言"ReduceL2"操作类型存在于ONNX模型的节点中
        self.assertIn("ReduceL2", onnx_nodes)
# 如果当前脚本作为主程序执行（而不是被导入到其他脚本中执行），则执行下面的代码块
if __name__ == "__main__":
    # 调用通用工具模块中的运行测试函数
    common_utils.run_tests()
```