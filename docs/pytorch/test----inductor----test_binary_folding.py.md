# `.\pytorch\test\inductor\test_binary_folding.py`

```
# 导入必要的模块和库
import functools  # 导入 functools 模块
import importlib  # 导入 importlib 模块
import itertools  # 导入 itertools 模块
import os  # 导入 os 模块
import sys  # 导入 sys 模块
import unittest  # 导入 unittest 模块

import torch  # 导入 PyTorch 库
from torch import nn  # 从 torch 中导入 nn 模块
from torch._inductor import config as inductor_config  # 导入 torch._inductor 的 config 模块
from torch.testing._internal.common_cuda import TEST_CUDNN  # 导入 torch.testing._internal.common_cuda 的 TEST_CUDNN 变量

# 让 test/ 中的辅助文件可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS, TEST_WITH_ASAN  # 导入测试中使用的常用工具函数和变量
from torch.testing._internal.inductor_utils import skipCUDAIf  # 导入 skipCUDAIf 函数用于条件跳过测试

# 在 Windows CI 环境下，如果是 CI 测试，输出相应的提示信息，并跳过测试
if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

# 导入测试所需的模块和函数
from inductor.test_inductor_freezing import TestCase  # 导入测试冻结功能的 TestCase 类
from inductor.test_torchinductor import check_model, check_model_gpu, copy_tests  # 导入测试 Torch Inductor 功能的相关函数

importlib.import_module("functorch")  # 动态导入 functorch 模块
importlib.import_module("filelock")  # 动态导入 filelock 模块

from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU  # 导入测试中使用的 GPU_TYPE、HAS_CPU、HAS_GPU 变量

aten = torch.ops.aten  # 设置 aten 为 torch 的 aten 操作模块

# 定义二进制折叠模板类，继承自 TestCase
class BinaryFoldingTemplate(TestCase):
    @skipCUDAIf(TEST_CUDNN, "CUDNN has accuracy issues for this test")  # 根据 TEST_CUDNN 变量条件跳过 CUDA 相关测试
    @inductor_config.patch({"freezing": True})  # 使用 inductor_config.patch 方法设置 freezing 参数为 True，用于配置测试环境
    def test_conv_bn_folding(self):
        @torch.no_grad()
        def test_conv_fusion(use_bias, module, expect_success):
            # 定义一个模拟的卷积和批归一化融合操作的测试函数
            class ConvOp(nn.Module):
                def __init__(self, in_channels, out_channels, device, **kwargs):
                    super().__init__()
                    # 初始化卷积层和批归一化层，并将它们移到指定设备上
                    self.conv = module[0](
                        in_channels, out_channels, bias=use_bias, **kwargs
                    ).to(device)
                    self.bn = module[1](out_channels).to(device)

                def forward(self, x):
                    # 执行卷积操作
                    x = self.conv(x)
                    # 执行批归一化操作
                    return self.bn(x)

            # 导入编译相关的库
            from torch._inductor.compile_fx import compile_fx, compile_fx_inner

            # 定义需要优化的运算类型
            aten_binary = [
                aten.add.Tensor,
                aten.sub.Tensor,
                aten.mul.Tensor,
                aten.div.Tensor,
            ]
            n_binary_ops = 0

            # 自定义的编译函数，用于内部优化
            def my_inner_compile(gm, example_inputs, *args, **kwargs):
                # 调用内部编译函数进行优化
                out = compile_fx_inner(gm, example_inputs, *args, **kwargs)
                nonlocal n_binary_ops
                # 统计优化后图中的二元运算数量
                binary_ops = [n for n in gm.graph.nodes if n.target in aten_binary]
                n_binary_ops += len(binary_ops)
                return out

            # 重置 Torch 动态图
            torch._dynamo.reset()
            # 创建一个评估模式下的 ConvOp 实例
            mod_eager = ConvOp(3, 32, self.device, kernel_size=3, stride=2).eval()
            # 使用自定义编译函数对模型进行优化编译
            out_optimized = torch.compile(
                mod_eager,
                backend=functools.partial(compile_fx, inner_compile=my_inner_compile),
            )

            # 定义输入的形状
            inps = [4, 3, 4]
            # 根据卷积层的类型决定是否添加额外的输入维度
            if module[0] == nn.Conv2d:
                inps.append(inps[-1])
            if module[0] == nn.Conv3d:
                inps.append(inps[-1])
                inps.append(inps[-1])

            # 创建随机输入数据，并移到指定设备上
            inp = torch.rand(inps).to(self.device)
            # 分别使用原始模型和优化后的模型进行推理
            out_eager = mod_eager(inp)
            out_optimized = out_optimized(inp)
            # 断言优化后的输出与原始输出在一定容差范围内一致
            self.assertEqual(out_optimized, out_eager, atol=2e-04, rtol=1e-5)
            # 根据预期结果判断是否成功优化掉了二元运算
            if expect_success:
                self.assertTrue(n_binary_ops == 0)
            else:
                self.assertTrue(n_binary_ops > 1)

        # 定义卷积层是否包含偏置的选项
        conv_bias = [True, False]
        # 定义不同维度下的卷积和批归一化层组合
        modules = [
            (nn.Conv1d, nn.BatchNorm1d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv3d, nn.BatchNorm3d),
        ]
        # 对每一种组合进行测试
        for use_bias, module in itertools.product(conv_bias, modules):
            test_conv_fusion(
                use_bias,
                module,
                expect_success=True,
            )
# 如果有 CPU 并且不支持多进程并行策略

    class FreezingCpuTests(TestCase):
        common = check_model  # 设置公共属性为 check_model
        device = "cpu"  # 设置设备为 CPU
        autocast = torch.cpu.amp.autocast  # 设置自动混合精度模式为 CPU

# 如果有 GPU 并且不使用 AddressSanitizer（内存错误检测工具）

    class FreezingGpuTests(TestCase):
        common = check_model_gpu  # 设置公共属性为 check_model_gpu
        device = GPU_TYPE  # 设置设备为 GPU_TYPE，通常为具体的 GPU 类型
        autocast = torch.amp.autocast(device_type=GPU_TYPE)  # 设置自动混合精度模式为指定 GPU 类型

    copy_tests(BinaryFoldingTemplate, FreezingGpuTests, GPU_TYPE)  # 复制 BinaryFoldingTemplate 的测试到 FreezingGpuTests，并指定 GPU 类型

# 删除 BinaryFoldingTemplate 类

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests  # 导入 run_tests 函数

    if HAS_CPU or HAS_GPU:  # 如果有 CPU 或 GPU
        run_tests(needs="filelock")  # 运行测试，需要 filelock（文件锁定功能）
```