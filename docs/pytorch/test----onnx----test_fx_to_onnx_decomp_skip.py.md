# `.\pytorch\test\onnx\test_fx_to_onnx_decomp_skip.py`

```py
# Owner(s): ["module: onnx"]
# 导入从未来导入的特性，支持类型注解
from __future__ import annotations

# 导入 ONNX 相关模块
import onnx
import onnx.inliner
import pytorch_test_common

# 导入 PyTorch 相关模块
import torch
from torch.testing._internal import common_utils

# 定义一个函数，用于检查指定操作类型是否存在于给定的 ONNX 模型中
def assert_op_in_onnx_model(model: onnx.ModelProto, op_type: str):
    # 内联模型中的局部函数
    inlined = onnx.inliner.inline_local_functions(model)
    # 遍历模型中的每个节点
    for node in inlined.graph.node:
        # 如果节点的操作类型等于给定的操作类型，则直接返回
        if node.op_type == op_type:
            return
    # 如果未找到指定的操作类型，则引发断言错误
    raise AssertionError(f"Op {op_type} not found in model")

# 定义一个测试类，继承自 ExportTestCase
class TestDynamoExportDecompSkip(pytorch_test_common.ExportTestCase):
    # 测试导出的程序是否强制分解
    def _test_exported_program_forces_decomposition(self, model, input, op_type):
        # 导出 Torch 模型
        ep = torch.export.export(model, input)
        # 使用 Torch ONNX 动态导出
        onnx_program = torch.onnx.dynamo_export(ep, *input)
        # 断言 ONNX 模型中存在指定的操作类型
        with self.assertRaises(AssertionError):
            assert_op_in_onnx_model(onnx_program.model_proto, op_type)

    # 测试双线性 2D 上采样操作
    def test_upsample_bilinear2d(self):
        # 定义一个测试用的模型
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

            def forward(self, x):
                return self.upsample(x)

        # 使用 Torch ONNX 动态导出
        onnx_program = torch.onnx.dynamo_export(TestModel(), torch.randn(1, 1, 2, 2))
        # 断言 ONNX 模型中存在指定的操作类型
        assert_op_in_onnx_model(onnx_program.model_proto, "Resize")
        # 测试导出的程序是否强制分解
        self._test_exported_program_forces_decomposition(
            TestModel(), (torch.randn(1, 1, 2, 2),), "Resize"
        )

    # 测试双线性 2D 上采样操作（指定输出大小）
    def test_upsample_bilinear2d_output_size(self):
        # 定义一个函数
        def func(x: torch.Tensor):
            return torch.nn.functional.interpolate(x, size=(4, 4), mode="bilinear")

        # 使用 Torch ONNX 动态导出
        onnx_program = torch.onnx.dynamo_export(func, torch.randn(1, 1, 2, 2))
        # 断言 ONNX 模型中存在指定的操作类型
        assert_op_in_onnx_model(onnx_program.model_proto, "Resize")

    # 测试三线性 3D 上采样操作
    def test_upsample_trilinear3d(self):
        # 定义一个测试用的模型
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.upsample = torch.nn.Upsample(scale_factor=2, mode="trilinear")

            def forward(self, x):
                return self.upsample(x)

        # 使用 Torch ONNX 动态导出
        onnx_program = torch.onnx.dynamo_export(TestModel(), torch.randn(1, 1, 2, 2, 3))
        # 断言 ONNX 模型中存在指定的操作类型
        assert_op_in_onnx_model(onnx_program.model_proto, "Resize")
        # 测试导出的程序是否强制分解
        self._test_exported_program_forces_decomposition(
            TestModel(), (torch.randn(1, 1, 2, 2, 3),), "Resize"
        )
    # 定义一个测试函数，用于测试 trilinear 三维插值的输出大小
    def test_upsample_trilinear3d_output_size(self):
        # 定义一个内部函数 func，接受一个 torch.Tensor x 作为输入，使用 trilinear 插值将其大小调整为 (4, 4, 4)
        def func(x: torch.Tensor):
            return torch.nn.functional.interpolate(x, size=(4, 4, 4), mode="trilinear")

        # 使用 torch.onnx.dynamo_export 将 func 函数导出为 ONNX 程序
        onnx_program = torch.onnx.dynamo_export(func, torch.randn(1, 1, 2, 2, 3))
        # 断言在导出的 ONNX 模型中存在 Resize 操作
        assert_op_in_onnx_model(onnx_program.model_proto, "Resize")

    # 定义一个测试函数，测试实例标准化（instance normalization）
    def test_instance_norm(self):
        # 定义一个继承自 torch.nn.Module 的测试模型 TestModel
        class TestModel(torch.nn.Module):
            # 定义模型的前向传播函数 forward，对输入 x 进行实例标准化处理
            def forward(self, x):
                return torch.nn.functional.instance_norm(x)

        # 使用 torch.onnx.dynamo_export 将 TestModel 实例导出为 ONNX 程序
        onnx_program = torch.onnx.dynamo_export(TestModel(), torch.randn(1, 1, 2, 2))
        # 断言在导出的 ONNX 模型中存在 InstanceNormalization 操作
        # 如果跳过了分解（decomposition），模型将包含 InstanceNormalization 操作而非带有 training=True 的 BatchNormalization 操作
        assert_op_in_onnx_model(onnx_program.model_proto, "InstanceNormalization")
        
        # 调用 _test_exported_program_forces_decomposition 方法，验证导出的程序是否强制进行了分解
        self._test_exported_program_forces_decomposition(
            TestModel(), (torch.randn(1, 1, 2, 2),), "InstanceNormalization"
        )
# 如果当前脚本作为主程序执行（而不是被导入到其他模块），则执行下面的代码块
if __name__ == "__main__":
    # 调用公共工具库中的运行测试函数，用于执行测试用例
    common_utils.run_tests()
```