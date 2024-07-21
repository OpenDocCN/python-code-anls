# `.\pytorch\test\onnx\test_export_modes.py`

```
# Owner(s): ["module: onnx"]

# 引入所需的模块和库
import io  # 导入用于处理字节流的模块
import os  # 导入用于处理操作系统相关功能的模块
import shutil  # 导入用于文件和目录操作的模块
import sys  # 导入系统相关的参数和函数
import tempfile  # 导入创建临时文件和目录的模块

import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入神经网络模块
from torch.autograd import Variable  # 导入变量自动求导模块
from torch.onnx import OperatorExportTypes  # 导入ONNX模型导出类型

# 将test/目录下的helper文件导入路径中，使其可被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
import pytorch_test_common  # 导入PyTorch测试的常用模块

from torch.testing._internal import common_utils  # 导入PyTorch内部测试的常用工具模块


# 对导出方法进行烟雾测试
class TestExportModes(pytorch_test_common.ExportTestCase):

    # 定义一个简单的模型用于测试
    class MyModel(nn.Module):
        def __init__(self):
            super(TestExportModes.MyModel, self).__init__()

        def forward(self, x):
            return x.transpose(0, 1)

    # 测试将模型导出为Protobuf格式
    def test_protobuf(self):
        torch_model = TestExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(
            torch_model,
            (fake_input),
            f,
            verbose=False,
            export_type=torch.onnx.ExportTypes.PROTOBUF_FILE,
        )

    # 测试将模型导出为ZIP文件格式
    def test_zipfile(self):
        torch_model = TestExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(
            torch_model,
            (fake_input),
            f,
            verbose=False,
            export_type=torch.onnx.ExportTypes.ZIP_ARCHIVE,
        )

    # 测试将模型导出为压缩的ZIP文件格式
    def test_compressed_zipfile(self):
        torch_model = TestExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(
            torch_model,
            (fake_input),
            f,
            verbose=False,
            export_type=torch.onnx.ExportTypes.COMPRESSED_ZIP_ARCHIVE,
        )

    # 测试将模型导出到指定目录
    def test_directory(self):
        torch_model = TestExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        d = tempfile.mkdtemp()  # 创建临时目录
        torch.onnx._export(
            torch_model,
            (fake_input),
            d,
            verbose=False,
            export_type=torch.onnx.ExportTypes.DIRECTORY,
        )
        shutil.rmtree(d)  # 删除临时目录及其内容

    # 测试带有多个返回值的ONNX模型导出
    def test_onnx_multiple_return(self):
        @torch.jit.script
        def foo(a):
            return (a, a)

        f = io.BytesIO()
        x = torch.ones(3)
        torch.onnx.export(foo, (x,), f)

    @common_utils.skipIfNoLapack  # 如果没有LAPACK库则跳过测试
    # 定义一个测试方法，用于测试 ATen 回退（fallback）到 ONNX 的情况
    def test_aten_fallback(self):
        # 定义一个继承自 nn.Module 的类，模拟不支持 ONNX 操作的模型
        class ModelWithAtenNotONNXOp(nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x, y):
                # 计算 x 和 y 的和
                abcd = x + y
                # 使用 torch.linalg.qr 函数对 abcd 进行 QR 分解
                defg = torch.linalg.qr(abcd)
                return defg

        # 创建输入张量 x 和 y，形状为 (3, 4)，元素值为随机生成的浮点数
        x = torch.rand(3, 4)
        y = torch.rand(3, 4)
        # 导出模型到 ONNX 格式的字符串，设置不添加节点名称、不进行常量折叠，使用 ATen 回退操作导出
        torch.onnx.export_to_pretty_string(
            ModelWithAtenNotONNXOp(),
            (x, y),
            add_node_names=False,
            do_constant_folding=False,
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
            # 指定 opset 版本为 9，因为 linalg.qr 支持需要较高的 opset 版本
            opset_version=9,
        )

    # 定义一个测试方法，用于测试 ATen 操作导出到 ONNX 的情况
    def test_onnx_aten(self):
        # 定义一个继承自 nn.Module 的类，模拟包含 torch.fmod 操作的模型
        class ModelWithAtenFmod(nn.Module):
            # 定义模型的前向传播方法
            def forward(self, x, y):
                # 使用 torch.fmod 函数计算 x 除以 y 的余数
                return torch.fmod(x, y)

        # 创建输入张量 x 和 y，形状为 (3, 4)，元素值为随机生成的浮点数
        x = torch.randn(3, 4, dtype=torch.float32)
        y = torch.randn(3, 4, dtype=torch.float32)
        # 导出模型到 ONNX 格式的字符串，设置不添加节点名称、不进行常量折叠，使用标准 ATen 操作导出
        torch.onnx.export_to_pretty_string(
            ModelWithAtenFmod(),
            (x, y),
            add_node_names=False,
            do_constant_folding=False,
            operator_export_type=OperatorExportTypes.ONNX_ATEN,
        )
# 如果当前脚本作为主程序执行（而不是被导入到其他模块中执行），则执行以下代码块
if __name__ == "__main__":
    # 调用common_utils模块中的run_tests函数，用于执行程序中定义的测试用例
    common_utils.run_tests()
```