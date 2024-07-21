# `.\pytorch\test\onnx\test_verification.py`

```py
# 导入所需的库和模块
import contextlib
import io
import tempfile
import unittest

import numpy as np
import onnx
import parameterized
import pytorch_test_common
from packaging import version

import torch
from torch.onnx import _constants, _experimental, verification
from torch.testing._internal import common_utils

# 定义测试类 TestVerification，继承自 pytorch_test_common.ExportTestCase
class TestVerification(pytorch_test_common.ExportTestCase):

    # 测试不可导出模型在常量不匹配时返回差异
    def test_check_export_model_diff_returns_diff_when_constant_mismatch(self):
        # 定义不可导出的模型 UnexportableModel
        class UnexportableModel(torch.nn.Module):
            # 模型的前向传播函数
            def forward(self, x, y):
                # tensor.data() 将被导出为常量，
                # 导致在不同输入下模型输出错误。
                return x + y.data

        # 定义测试输入组
        test_input_groups = [
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
        ]

        # 调用 verification 模块的 check_export_model_diff 函数进行验证
        results = verification.check_export_model_diff(
            UnexportableModel(), test_input_groups
        )
        # 断言结果匹配特定的正则表达式模式
        self.assertRegex(
            results,
            r"Graph diff:(.|\n)*"
            r"First diverging operator:(.|\n)*"
            r"prim::Constant(.|\n)*"
            r"Former source location:(.|\n)*"
            r"Latter source location:",
        )

    # 测试不可导出模型在动态控制流不匹配时返回差异
    def test_check_export_model_diff_returns_diff_when_dynamic_controlflow_mismatch(
        self,
    ):
        # 定义不可导出的模型 UnexportableModel
        class UnexportableModel(torch.nn.Module):
            # 模型的前向传播函数
            def forward(self, x, y):
                for i in range(x.size(0)):
                    y = x[i] + y
                return y

        # 定义测试输入组
        test_input_groups = [
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
            ((torch.randn(4, 3), torch.randn(2, 3)), {}),
        ]

        # 定义导出选项 export_options
        export_options = _experimental.ExportOptions(
            input_names=["x", "y"], dynamic_axes={"x": [0]}
        )

        # 调用 verification 模块的 check_export_model_diff 函数进行验证
        results = verification.check_export_model_diff(
            UnexportableModel(), test_input_groups, export_options
        )
        # 断言结果匹配特定的正则表达式模式
        self.assertRegex(
            results,
            r"Graph diff:(.|\n)*"
            r"First diverging operator:(.|\n)*"
            r"prim::Constant(.|\n)*"
            r"Latter source location:(.|\n)*",
        )

    # 测试可导出模型在正确导出时返回空字符串
    def test_check_export_model_diff_returns_empty_when_correct_export(self):
        # 定义支持的模型 SupportedModel
        class SupportedModel(torch.nn.Module):
            # 模型的前向传播函数
            def forward(self, x, y):
                return x + y

        # 定义测试输入组
        test_input_groups = [
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
            ((torch.randn(2, 3), torch.randn(2, 3)), {}),
        ]

        # 调用 verification 模块的 check_export_model_diff 函数进行验证
        results = verification.check_export_model_diff(
            SupportedModel(), test_input_groups
        )
        # 断言结果为空字符串
        self.assertEqual(results, "")

    # 测试 ORT 和 PyTorch 输出比较在可接受误差范围内不引发异常
    def test_compare_ort_pytorch_outputs_no_raise_with_acceptable_error_percentage(
        self,
    ):
        # 定义 ONNX 和 PyTorch 的输出结果作为 NumPy 数组和 Torch 张量
        ort_outs = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        pytorch_outs = [torch.tensor([[1.0, 2.0], [3.0, 1.0]])]
        
        # 设置用于验证的选项，包括相对误差容忍度（rtol）、绝对误差容忍度（atol）、形状检查（check_shape）、数据类型检查（check_dtype）、忽略 None 值（ignore_none）、可接受误差百分比（acceptable_error_percentage）
        options = verification.VerificationOptions(
            rtol=1e-5,
            atol=1e-6,
            check_shape=True,
            check_dtype=False,
            ignore_none=True,
            acceptable_error_percentage=0.3,
        )
        
        # 调用函数验证 ONNX 和 PyTorch 的输出是否一致，使用设定的选项
        verification._compare_onnx_pytorch_outputs(
            ort_outs,
            pytorch_outs,
            options,
        )

    def test_compare_ort_pytorch_outputs_raise_without_acceptable_error_percentage(
        self,
    ):
        # 定义 ONNX 和 PyTorch 的输出结果作为 NumPy 数组和 Torch 张量
        ort_outs = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        pytorch_outs = [torch.tensor([[1.0, 2.0], [3.0, 1.0]])]
        
        # 设置用于验证的选项，但这次不定义可接受的误差百分比
        options = verification.VerificationOptions(
            rtol=1e-5,
            atol=1e-6,
            check_shape=True,
            check_dtype=False,
            ignore_none=True,
            acceptable_error_percentage=None,
        )
        
        # 使用断言来验证函数是否会引发 AssertionError，因为没有定义可接受的误差百分比
        with self.assertRaises(AssertionError):
            verification._compare_onnx_pytorch_outputs(
                ort_outs,
                pytorch_outs,
                options,
            )
@common_utils.instantiate_parametrized_tests
class TestVerificationOnWrongExport(pytorch_test_common.ExportTestCase):
    opset_version: int  # 定义测试中使用的ONNX操作集版本号

    def setUp(self):
        super().setUp()

        # 定义一个错误的自定义符号函数，始终返回self
        def incorrect_add_symbolic_function(g, self, other, alpha):
            return self

        self.opset_version = _constants.ONNX_DEFAULT_OPSET  # 设置ONNX操作集版本为默认版本
        # 注册自定义的ONNX操作符号函数
        torch.onnx.register_custom_op_symbolic(
            "aten::add",
            incorrect_add_symbolic_function,
            opset_version=self.opset_version,
        )

    def tearDown(self):
        super().tearDown()
        # 取消注册自定义的ONNX操作符号函数
        torch.onnx.unregister_custom_op_symbolic(
            "aten::add", opset_version=self.opset_version
        )

    @common_utils.parametrize(
        "onnx_backend",
        [
            common_utils.subtest(
                verification.OnnxBackend.REFERENCE,
                decorators=[
                    unittest.skipIf(
                        version.Version(onnx.__version__) < version.Version("1.13"),
                        reason="Reference Python runtime was introduced in 'onnx' 1.13.",
                    )
                ],
            ),
            verification.OnnxBackend.ONNX_RUNTIME_CPU,  # 使用ONNX运行时的CPU后端
        ],
    )
    def test_verify_found_mismatch_when_export_is_wrong(
        self, onnx_backend: verification.OnnxBackend
    ):
        # 定义一个简单的模型，前向传播时对输入张量加1
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        with self.assertRaisesRegex(AssertionError, ".*Tensor-likes are not close!.*"):
            # 进行模型导出验证，预期会出现断言错误异常
            verification.verify(
                Model(),
                (torch.randn(2, 3),),  # 随机生成2x3大小的张量作为输入
                opset_version=self.opset_version,
                options=verification.VerificationOptions(backend=onnx_backend),
            )


@parameterized.parameterized_class(
    [
        # TODO: enable this when ONNX submodule catches up to >= 1.13.
        # {"onnx_backend": verification.OnnxBackend.ONNX},
        {"onnx_backend": verification.OnnxBackend.ONNX_RUNTIME_CPU},  # 使用ONNX运行时的CPU后端
    ],
    class_name_func=lambda cls, idx, input_dicts: f"{cls.__name__}_{input_dicts['onnx_backend'].name}",
)
class TestFindMismatch(pytorch_test_common.ExportTestCase):
    onnx_backend: verification.OnnxBackend
    opset_version: int  # 定义测试中使用的ONNX操作集版本号
    graph_info: verification.GraphInfo  # 图信息用于验证
    # 在每个测试方法执行前设置环境
    def setUp(self):
        super().setUp()
        self.opset_version = _constants.ONNX_DEFAULT_OPSET

        # 定义一个错误的 ReLU 符号函数，用于注册自定义的 ONNX 操作符符号
        def incorrect_relu_symbolic_function(g, self):
            return g.op("Add", self, g.op("Constant", value_t=torch.tensor(1.0)))

        # 注册自定义的操作符符号 "aten::relu"
        torch.onnx.register_custom_op_symbolic(
            "aten::relu",
            incorrect_relu_symbolic_function,
            opset_version=self.opset_version,
        )

        # 定义一个简单的神经网络模型
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(3, 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(4, 5),
                    torch.nn.ReLU(),
                    torch.nn.Linear(5, 6),
                )

            def forward(self, x):
                return self.layers(x)

        # 执行模型验证，获取图信息，用于后续的测试
        self.graph_info = verification.find_mismatch(
            Model(),
            (torch.randn(2, 3),),
            opset_version=self.opset_version,
            options=verification.VerificationOptions(backend=self.onnx_backend),
        )

    # 在每个测试方法执行后清理环境
    def tearDown(self):
        super().tearDown()
        # 注销自定义的操作符符号 "aten::relu"
        torch.onnx.unregister_custom_op_symbolic(
            "aten::relu", opset_version=self.opset_version
        )
        # 删除属性，清理环境
        delattr(self, "opset_version")
        delattr(self, "graph_info")

    # 测试方法：验证图形化输出是否符合预期
    def test_pretty_print_tree_visualizes_mismatch(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.graph_info.pretty_print_tree()
        # 断言输出的内容符合预期
        self.assertExpected(f.getvalue())

    # 测试方法：验证错误源位置是否保留在输出中
    def test_preserve_mismatch_source_location(self):
        # 获取所有不匹配叶子节点的信息
        mismatch_leaves = self.graph_info.all_mismatch_leaf_graph_info()

        # 断言至少存在一个不匹配的叶子节点
        self.assertTrue(len(mismatch_leaves) > 0)

        # 遍历每个不匹配的叶子节点信息
        for leaf_info in mismatch_leaves:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                # 打印该叶子节点的不匹配信息（包括图形化）
                leaf_info.pretty_print_mismatch(graph=True)
            # 使用正则表达式断言输出中包含 "aten::relu" 函数的源代码位置信息
            self.assertRegex(
                f.getvalue(),
                r"(.|\n)*" r"aten::relu.*/torch/nn/functional.py:[0-9]+(.|\n)*",
            )

    # 测试方法：验证所有不匹配操作符的数量是否正确
    def test_find_all_mismatch_operators(self):
        # 获取所有不匹配叶子节点的信息
        mismatch_leaves = self.graph_info.all_mismatch_leaf_graph_info()

        # 断言不匹配叶子节点的数量为 2
        self.assertEqual(len(mismatch_leaves), 2)

        # 遍历每个不匹配的叶子节点信息
        for leaf_info in mismatch_leaves:
            # 断言每个叶子节点的关键节点数量为 1
            self.assertEqual(leaf_info.essential_node_count(), 1)
            # 断言每个叶子节点的关键节点类型为 "aten::relu"
            self.assertEqual(leaf_info.essential_node_kinds(), {"aten::relu"})
    # 定义一个测试方法，用于验证在没有不匹配时打印正确信息的情况
    def test_find_mismatch_prints_correct_info_when_no_mismatch(self):
        # 设置最大差异为无限制
        self.maxDiff = None

        # 定义一个简单的模型类，继承自torch.nn.Module
        class Model(torch.nn.Module):
            # 模型的前向传播方法，简单地返回输入加1的结果
            def forward(self, x):
                return x + 1

        # 创建一个StringIO对象，用于捕获输出
        f = io.StringIO()

        # 将标准输出重定向到StringIO对象f
        with contextlib.redirect_stdout(f):
            # 调用verification.find_mismatch函数，并将输出结果写入f
            verification.find_mismatch(
                Model(),  # 使用定义的模型对象
                (torch.randn(2, 3),),  # 传入一个随机张量作为输入
                opset_version=self.opset_version,  # 指定操作集版本
                options=verification.VerificationOptions(backend=self.onnx_backend),  # 设置验证选项
            )

        # 使用self.assertExpected方法验证f.getvalue()的输出是否符合预期

    # 定义一个测试方法，用于验证导出不匹配的重现信息
    def test_export_repro_for_mismatch(self):
        # 获取所有不匹配叶子图信息
        mismatch_leaves = self.graph_info.all_mismatch_leaf_graph_info()

        # 断言至少存在一个不匹配的叶子图信息
        self.assertTrue(len(mismatch_leaves) > 0)

        # 获取第一个不匹配叶子图信息
        leaf_info = mismatch_leaves[0]

        # 使用临时目录创建一个TemporaryDirectory对象
        with tempfile.TemporaryDirectory() as temp_dir:
            # 导出不匹配重现信息到临时目录，返回导出的重现信息目录
            repro_dir = leaf_info.export_repro(temp_dir)

            # 使用assertRaisesRegex断言抛出AssertionError，并检查错误信息中是否包含特定字符串
            with self.assertRaisesRegex(AssertionError, "Tensor-likes are not close!"):
                # 创建验证选项对象
                options = verification.VerificationOptions(backend=self.onnx_backend)

                # 创建OnnxTestCaseRepro对象，并使用导出的重现信息目录进行验证
                verification.OnnxTestCaseRepro(repro_dir).validate(options)
# 如果当前脚本被直接执行（而不是被导入到其他脚本中执行），则执行以下代码块
if __name__ == "__main__":
    # 调用 common_utils 模块中的 run_tests 函数，通常用于执行测试用例
    common_utils.run_tests()
```