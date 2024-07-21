# `.\pytorch\test\inductor\test_loop_ordering.py`

```py
# Owner(s): ["module: inductor"]

# 导入 PyTorch 库
import torch
# 导入测试相关的随机生成函数
from torch._dynamo.testing import rand_strided
# 导入用于比较函数
from torch._dynamo.utils import same
# 导入 Inductor 配置和度量模块
from torch._inductor import config as inductor_config, metrics
# 导入测试用例基类
from torch._inductor.test_case import run_tests, TestCase
# 导入 Inductor 工具 CUDA 是否可用的标志
from torch.testing._internal.inductor_utils import HAS_CUDA

# 如果 CUDA 可用，设置默认使用 CUDA 设备
if HAS_CUDA:
    torch.set_default_device("cuda")

# 使用装饰器设置 Inductor 相关配置参数
@inductor_config.patch(
    {
        "benchmark_kernel": True,
        "triton.unique_kernel_names": True,
    }
)
# 定义测试类 LoopOrderingTest，继承自 TestCase
class LoopOrderingTest(TestCase):
    # 定义测试辅助函数 do_acc_test，用于测试函数 f 的精度
    def do_acc_test(self, f, *args):
        # 调用函数 f 计算期望结果
        expect = f(*args)
        # 使用 Torch 编译后调用函数 f 计算实际结果
        actual = torch.compile(f)(*args)
        # 断言期望结果与实际结果近似相等，精度为 1e-3
        self.assertTrue(same(expect, actual, tol=1e-3))

    # 定义测试函数 test_for_reordering_reindex，用于测试函数重排序和重索引
    def test_for_reordering_reindex(self):
        """
        ComputedBuffer.iter_reoredering_reindex can cause some fusion
        opportunitiies being skipped.

        In this test case, Inductor generates 2 triton kernels before.
        By removing ComputedBuffer.iter_reoredering_reindex, we can fuse those
        two kernels into a single one.
        """
        # 定义函数 f，执行矩阵乘法操作，并强制输出布局
        def f(x, y):
            """
            Add a matmul since inductor may force layout for output.
            """
            return (x.sum(dim=-1) + 1) @ y

        A, B = 20, 30
        # 创建 CUDA 设备上的随机张量 x，形状为 [A, A, B]
        # 意图是使前两个维度无法合并，以验证 ComputedBuffer.iter_reoredering_reindex 是否会更新
        x = rand_strided([A, A, B], [B, B * A + 300, 1], device="cuda")
        # 创建形状为 [A, A] 的随机张量 y
        y = torch.randn(A, A)

        # 执行精度测试，验证函数 f 的正确性
        self.do_acc_test(f, x, y)
        # 断言生成的 triton 内核数量为 1
        self.assertEqual(1, metrics.generated_kernel_count)
        # 预期访问的总字节数计算
        expected_num_bytes = 0
        expected_num_bytes += A * A * B + A * A  # for the fused reduction
        expected_num_bytes += A * A * 3  # for matmul
        expected_num_bytes *= x.itemsize
        # 断言实际访问的总字节数与预期相等
        self.assertEqual(expected_num_bytes, metrics.num_bytes_accessed)

# 如果作为主程序运行，并且 CUDA 可用，执行测试
if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
```