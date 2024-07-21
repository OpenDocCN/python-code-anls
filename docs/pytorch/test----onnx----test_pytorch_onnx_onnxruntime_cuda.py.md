# `.\pytorch\test\onnx\test_pytorch_onnx_onnxruntime_cuda.py`

```py
# Owner(s): ["module: onnx"]

# 导入单元测试模块
import unittest

# 导入 ONNX 测试常用功能模块
import onnx_test_common

# 导入 ONNX Runtime，并标记为不使用的引用，确保其引入但不直接使用
import onnxruntime  # noqa: F401

# 导入参数化测试模块
import parameterized

# 导入需要的函数和类
from onnx_test_common import MAX_ONNX_OPSET_VERSION, MIN_ONNX_OPSET_VERSION
from pytorch_test_common import (
    skipIfNoBFloat16Cuda,
    skipIfNoCuda,
    skipIfUnsupportedMinOpsetVersion,
    skipScriptTest,
)

# 导入 PyTorch 相关模块
import torch
from torch.cuda.amp import autocast
from torch.testing._internal import common_utils

# 使用参数化测试类，传递最小和最大的 ONNX Opset 版本，并设置类名的生成函数
@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(
        MIN_ONNX_OPSET_VERSION, MAX_ONNX_OPSET_VERSION
    ),
    class_name_func=onnx_test_common.parameterize_class_name,
)
# 继承自 onnx_test_common._TestONNXRuntime 的测试类
class TestONNXRuntime_cuda(onnx_test_common._TestONNXRuntime):

    # 跳过不支持的最小 Opset 版本为 9 的测试
    @skipIfUnsupportedMinOpsetVersion(9)
    # 跳过非 CUDA 环境的测试
    @skipIfNoCuda
    def test_gelu_fp16(self):
        # 定义一个 GeluModel 类，继承自 torch.nn.Module
        class GeluModel(torch.nn.Module):
            def forward(self, x):
                # 使用 torch.nn.functional.gelu 函数计算 gelu
                return torch.nn.functional.gelu(x)

        # 生成一个随机的 float16 类型的 CUDA 张量作为输入
        x = torch.randn(
            2,
            4,
            5,
            6,
            requires_grad=True,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
        # 运行测试，验证 GeluModel 的输出
        self.run_test(GeluModel(), x, rtol=1e-3, atol=1e-5)

    # 跳过不支持的最小 Opset 版本为 9 的测试
    @skipIfUnsupportedMinOpsetVersion(9)
    # 跳过非 CUDA 环境的测试
    @skipIfNoCuda
    # 标记为脚本测试
    @skipScriptTest()
    def test_layer_norm_fp16(self):
        # 定义一个 LayerNormModel 类，继承自 torch.nn.Module
        class LayerNormModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_norm = torch.nn.LayerNorm([10, 10])

            @autocast()
            def forward(self, x):
                # 执行 LayerNorm 操作
                return self.layer_norm(x)

        # 生成一个随机的 float16 类型的 CUDA 张量作为输入
        x = torch.randn(
            20,
            5,
            10,
            10,
            requires_grad=True,
            dtype=torch.float16,
            device=torch.device("cuda"),
        )
        # 运行测试，验证 LayerNormModel 的输出
        self.run_test(LayerNormModel().cuda(), x, rtol=1e-3, atol=1e-5)

    # 跳过不支持的最小 Opset 版本为 12 的测试
    @skipIfUnsupportedMinOpsetVersion(12)
    # 跳过非 CUDA 环境的测试
    @skipIfNoCuda
    # 标记为脚本测试
    @skipScriptTest()
    def test_softmaxCrossEntropy_fusion_fp16(self):
        # 定义一个 FusionModel 类，继承自 torch.nn.Module
        class FusionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction="none")
                self.m = torch.nn.LogSoftmax(dim=1)

            @autocast()
            def forward(self, input, target):
                # 执行 LogSoftmax 和 NLLLoss 操作
                output = self.loss(self.m(2 * input), target)
                return output

        N, C = 5, 4
        # 生成一个随机的 float16 类型的 CUDA 张量作为输入和一个随机的目标张量
        input = torch.randn(N, 16, dtype=torch.float16, device=torch.device("cuda"))
        target = torch.empty(N, dtype=torch.long, device=torch.device("cuda")).random_(
            0, C
        )

        # 将目标张量中为 1 的元素设置为 -100，用作测试数据中的默认 ignore_index
        target[target == 1] = -100
        # 运行测试，验证 FusionModel 的输出
        self.run_test(FusionModel(), (input, target))

    # 跳过非 CUDA 环境的测试
    @skipIfNoCuda
    # 标记为脚本测试
    @skipScriptTest()
    # 定义一个测试方法，用于测试使用 Apex 加速训练的情况
    def test_apex_o2(self):
        # 定义一个简单的线性模型类，继承自 torch.nn.Module
        class LinearModel(torch.nn.Module):
            # 模型初始化方法
            def __init__(self):
                super().__init__()
                # 定义一个线性层，输入维度为3，输出维度为5
                self.linear = torch.nn.Linear(3, 5)

            # 前向传播方法
            def forward(self, x):
                return self.linear(x)

        # 尝试导入 Apex 模块，如果导入失败则抛出跳过测试的异常
        try:
            from apex import amp
        except Exception as e:
            raise unittest.SkipTest("Apex is not available") from e
        
        # 生成一个随机输入张量，大小为3x3，放在 CUDA 设备上
        input = torch.randn(3, 3, device=torch.device("cuda"))
        # 使用 Apex 的 amp.initialize 方法对模型进行初始化，优化级别为 "O2"
        model = amp.initialize(LinearModel(), opt_level="O2")
        # 运行测试方法，将模型和输入作为参数传入
        self.run_test(model, input)

    # 如果 ONNX 的操作集版本 >= 13，支持 bfloat16 类型
    # Add、Sub 和 Mul 操作不支持 bfloat16 在 onnxruntime 上运行在 CPU 上。
    @skipIfUnsupportedMinOpsetVersion(13)
    @skipIfNoBFloat16Cuda
    def test_arithmetic_bfp16(self):
        # 定义一个简单的神经网络模块类
        class MyModule(torch.nn.Module):
            # 前向传播方法
            def forward(self, x):
                # 生成一个大小为3x4的全1张量，数据类型为 bfloat16，在 CUDA 设备上
                y = torch.ones(3, 4, dtype=torch.bfloat16, device=torch.device("cuda"))
                # 将输入张量 x 转换为和 y 相同的数据类型
                x = x.type_as(y)
                # 返回乘积结果和减法结果的乘积，最终转换为 float16 数据类型
                return torch.mul(torch.add(x, y), torch.sub(x, y)).to(
                    dtype=torch.float16
                )

        # 生成一个全1张量，大小为3x4，需要梯度计算，数据类型为 float16，在 CUDA 设备上
        x = torch.ones(
            3, 4, requires_grad=True, dtype=torch.float16, device=torch.device("cuda")
        )
        # 运行测试方法，将模块实例和输入作为参数传入，设置相对和绝对容差
        self.run_test(MyModule(), x, rtol=1e-3, atol=1e-5)

    # 如果没有 CUDA 设备，跳过测试
    def test_deduplicate_initializers_diff_devices(self):
        # 定义一个简单的神经网络模型类
        class Model(torch.nn.Module):
            # 模型初始化方法
            def __init__(self):
                super().__init__()
                # 定义一个维度为2x3的参数 w，放在 CPU 设备上
                self.w = torch.nn.Parameter(
                    torch.ones(2, 3, device=torch.device("cpu"))
                )
                # 定义一个维度为3的参数 b，放在 CUDA 设备上
                self.b = torch.nn.Parameter(torch.ones(3, device=torch.device("cuda")))

            # 前向传播方法，接受两个输入 x 和 y
            def forward(self, x, y):
                # 返回 w 和 x 的矩阵乘积，以及 y 加上参数 b
                return torch.matmul(self.w, x), y + self.b

        # 生成一个随机输入张量 x，大小为3x3，放在 CPU 设备上
        x = torch.randn(3, 3, device=torch.device("cpu"))
        # 生成一个随机输入张量 y，大小为3x3，放在 CUDA 设备上
        y = torch.randn(3, 3, device=torch.device("cuda"))
        # 运行测试方法，将模型实例和输入作为参数传入
        self.run_test(Model(), (x, y))
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 调用 common_utils 模块中的 run_tests 函数，用于运行测试
    common_utils.run_tests()
```