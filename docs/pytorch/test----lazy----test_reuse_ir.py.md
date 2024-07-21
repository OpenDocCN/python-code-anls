# `.\pytorch\test\lazy\test_reuse_ir.py`

```py
# Owner(s): ["oncall: jit"]

# 导入必要的库和模块
import os
import unittest

import torch
import torch._lazy  # 导入懒加载模块
import torch._lazy.config  # 导入懒加载配置模块
import torch._lazy.ir_cache  # 导入懒加载 IR 缓存模块
import torch._lazy.metrics as metrics  # 导入懒加载指标模块
import torch._lazy.ts_backend  # 导入懒加载 TS 后端模块
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase  # 导入测试相关的工具和类

# 初始化懒加载 TS 后端
torch._lazy.ts_backend.init()
# 设置 IR 重用策略为 True
torch._lazy.config.set_reuse_ir(True)


def get_test_device():
    # 根据环境变量判断测试设备为 "cuda" 或 "cpu"
    return "cuda" if "LTC_TS_CUDA" in os.environ else "cpu"


# 标记在 Windows 平台下需要修复的测试用例
@unittest.skipIf(IS_WINDOWS, "To be fixed")
class TestLazyReuseIr(TestCase):
    def testAdd(self):
        device = get_test_device()
        # 在指定设备上创建随机张量 x, y 和全零张量 z
        x = torch.randn(2, 3, 4, device=device)
        y = torch.randn(2, 3, 4, device=device)
        z = torch.zeros(2, 3, 4, device=device)

        device = "lazy"  # 设置懒加载设备
        # 将 x, y, z 分别创建为懒加载张量 x_lazy, y_lazy, z_lazy
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        # 执行十次加法操作
        for i in range(10):
            z += x + y

        # 执行十次懒加载加法操作，并标记每一步
        for i in range(10):
            z_lazy += x_lazy + y_lazy
            torch._lazy.mark_step()

        # 断言 z 和 z_lazy 在 CPU 上的结果接近
        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        # 断言 IR 节点重用的计数至少为 14
        assert metrics.counter_value("IrNodeReused_torch::lazy::AddTensor") >= 14
        # 重置指标
        metrics.reset()
        # 重置 IR 缓存
        torch._lazy.ir_cache.reset()

    def testAddSub(self):
        device = get_test_device()
        # 在指定设备上创建随机张量 x, y 和全零张量 z
        x = torch.randn(2, 3, 4, device=device)
        y = torch.randn(2, 3, 4, device=device)
        z = torch.zeros(2, 3, 4, device=device)

        device = "lazy"  # 设置懒加载设备
        # 将 x, y, z 分别创建为懒加载张量 x_lazy, y_lazy, z_lazy
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        # 执行十次条件加法或减法操作
        for i in range(10):
            if i < 5:
                z += x + y
            else:
                z += x - y

        # 执行十次条件懒加载加法或减法操作，并标记每一步
        for i in range(10):
            if i < 5:
                z_lazy += x_lazy + y_lazy
            else:
                z_lazy += x_lazy - y_lazy
            torch._lazy.mark_step()

        # 断言 z 和 z_lazy 在 CPU 上的结果接近
        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        # 断言 IR 节点重用的计数至少为 8
        assert metrics.counter_value("IrNodeReused_torch::lazy::AddTensor") >= 8
        # 重置指标
        metrics.reset()
        # 重置 IR 缓存
        torch._lazy.ir_cache.reset()
    def testAddSubFallback(self):
        # 设置强制回退到使用 "aten::sub" 操作
        torch._lazy.config.set_force_fallback("aten::sub")
        # 获取测试设备
        device = get_test_device()
        # 创建随机张量 x, y, z，并指定设备
        x = torch.randn(2, 3, 4, device=device)
        y = torch.randn(2, 3, 4, device=device)
        z = torch.zeros(2, 3, 4, device=device)

        # 将设备切换为 "lazy" 模式，创建 x_lazy, y_lazy, z_lazy 的懒惰拷贝
        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        y_lazy = y.detach().clone().to(device=device)
        z_lazy = z.detach().clone().to(device=device)

        # 循环执行加法和减法操作
        for i in range(10):
            if i < 5:
                z += x + y
            else:
                z += x - y

        # 在 lazy 模式下，循环执行加法和减法操作，每次操作后标记步骤
        for i in range(10):
            if i < 5:
                z_lazy += x_lazy + y_lazy
            else:
                z_lazy += x_lazy - y_lazy
            torch._lazy.mark_step()

        # 断言 z 和 z_lazy 在 CPU 上的值接近
        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        # 断言计算的节点重用次数符合预期
        assert metrics.counter_value("IrNodeReused_torch::lazy::AddTensor") >= 8
        # 重置度量指标
        metrics.reset()
        # 重置 IR 缓存
        torch._lazy.ir_cache.reset()
        # 清空强制回退设置
        torch._lazy.config.set_force_fallback("")

    def testBatchNorm(self):
        # 获取测试设备
        device = get_test_device()
        # 创建随机输入张量 x, 权重 weight, 偏置 bias，并指定设备
        x = torch.randn(16, 3, 224, 224, device=device)
        weight = torch.randn(3, device=device)
        bias = torch.randn(3, device=device)

        # 循环执行 BatchNorm2d 操作，通过调用 `torch.ops.aten.native_batch_norm` 来绕过维度检查
        for i in range(10):
            # z 是 BatchNorm2d 操作的结果，_ 和 _ 是无用返回值
            z, _, _ = torch.ops.aten.native_batch_norm(
                x, weight, bias, None, None, True, 0.1, 1e-5
            )
            # z_legit 是另一种 BatchNorm2d 操作的结果，通过 `torch.ops.aten._native_batch_norm_legit` 调用
            z_legit, _, _ = torch.ops.aten._native_batch_norm_legit(
                x, weight, bias, True, 0.1, 1e-5
            )

        # 将设备切换为 "lazy" 模式，创建懒惰拷贝 x_lazy, weight_lazy, bias_lazy
        device = "lazy"
        x_lazy = x.detach().clone().to(device=device)
        weight_lazy = weight.detach().clone().to(device=device)
        bias_lazy = bias.detach().clone().to(device=device)

        # 在 lazy 模式下，循环执行 BatchNorm2d 操作，每次操作后标记步骤
        for i in range(10):
            z_lazy, _, _ = torch.ops.aten.native_batch_norm(
                x_lazy, weight_lazy, bias_lazy, None, None, True, 0.1, 1e-5
            )
            z_legit_lazy, _, _ = torch.ops.aten._native_batch_norm_legit(
                x_lazy, weight_lazy, bias_lazy, True, 0.1, 1e-5
            )
            torch._lazy.mark_step()

        # 断言 z 和 z_lazy 在 CPU 上的值接近
        torch.testing.assert_close(z.cpu(), z_lazy.cpu())
        # 断言 z_legit 和 z_legit_lazy 在 CPU 上的值接近
        torch.testing.assert_close(z_legit.cpu(), z_legit_lazy.cpu())
        # 断言计算的节点重用次数符合预期
        assert metrics.counter_value("IrNodeReused_torch::lazy::NativeBatchNorm") >= 7
        # 重置度量指标
        metrics.reset()
        # 重置 IR 缓存
        torch._lazy.ir_cache.reset()
# 如果当前脚本作为主程序运行（而不是被导入到其他脚本中），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```