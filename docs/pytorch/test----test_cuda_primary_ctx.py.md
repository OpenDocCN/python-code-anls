# `.\pytorch\test\test_cuda_primary_ctx.py`

```py
# Owner(s): ["module: cuda"]

# 导入系统模块和单元测试模块
import sys
import unittest

# 导入 torch 库及相关 CUDA 测试标记和工具函数
import torch
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_utils import (
    NoTest,
    run_tests,
    skipIfRocmVersionLessThan,
    TestCase,
)

# NOTE: this needs to be run in a brand new process
# 注意: 这段代码需要在全新的进程中运行

# 如果 CUDA 不可用，则跳过测试，并将 TestCase 设置为 NoTest
if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

# 定义一个测试类，标记为 DynamoStrict 测试
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCudaPrimaryCtx(TestCase):

    # 定义一个错误信息常量，用于指示必须在不创建 CUDA 上下文的进程中运行测试
    CTX_ALREADY_CREATED_ERR_MSG = (
        "Tests defined in test_cuda_primary_ctx.py must be run in a process "
        "where CUDA contexts are never created. Use either run_test.py or add "
        "--subprocess to run each test in a different subprocess."
    )

    # 在每个测试方法运行前执行的设置方法
    @skipIfRocmVersionLessThan((4, 4, 21504))
    def setUp(self):
        # 遍历所有 CUDA 设备
        for device in range(torch.cuda.device_count()):
            # 确保在测试开始前，CUDA 主上下文尚未创建
            self.assertFalse(
                torch._C._cuda_hasPrimaryContext(device),
                TestCudaPrimaryCtx.CTX_ALREADY_CREATED_ERR_MSG,
            )

    # 如果只有一个 GPU 可用，则跳过测试
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_str_repr(self):
        # 在 'cuda:1' 设备上生成一个随机张量
        x = torch.randn(1, device="cuda:1")

        # 断言在 'cuda:1' 设备上已创建主上下文，而在 'cuda:0' 设备上未创建主上下文
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        # 对张量进行 str 和 repr 操作
        str(x)
        repr(x)

        # 再次断言在 'cuda:1' 设备上已创建主上下文，而在 'cuda:0' 设备上未创建主上下文
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

    # 如果只有一个 GPU 可用，则跳过测试
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_copy(self):
        # 在 'cuda:1' 设备上生成一个随机张量
        x = torch.randn(1, device="cuda:1")

        # 断言在 'cuda:1' 设备上已创建主上下文，而在 'cuda:0' 设备上未创建主上下文
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        # 在 CPU 设备上生成一个随机张量，并将 'cuda:1' 的张量复制到 CPU 张量上
        y = torch.randn(1, device="cpu")
        y.copy_(x)

        # 再次断言在 'cuda:1' 设备上已创建主上下文，而在 'cuda:0' 设备上未创建主上下文
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

    # 如果只有一个 GPU 可用，则跳过测试
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    # 定义一个测试方法，用于测试 pin_memory 功能
    def test_pin_memory(self):
        # 在 'cuda:1' 设备上生成一个随机张量
        x = torch.randn(1, device="cuda:1")

        # 断言：应该只在 'cuda:1' 上创建了主上下文
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        # 断言：张量 x 不是被固定在内存中
        self.assertFalse(x.is_pinned())

        # 断言：应该仍然只在 'cuda:1' 上创建了主上下文
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        # 在 'cpu' 设备上生成一个随机张量，并将其固定在内存中
        x = torch.randn(3, device="cpu").pin_memory()

        # 断言：应该仍然只在 'cuda:1' 上创建了主上下文
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        # 断言：张量 x 已经被固定在内存中
        self.assertTrue(x.is_pinned())

        # 断言：应该仍然只在 'cuda:1' 上创建了主上下文
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        # 在 'cpu' 设备上生成一个随机张量，并在创建时将其固定在内存中
        x = torch.randn(3, device="cpu", pin_memory=True)

        # 断言：应该仍然只在 'cuda:1' 上创建了主上下文
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        # 用零初始化一个张量，并将其固定在内存中
        x = torch.zeros(3, device="cpu", pin_memory=True)

        # 断言：应该仍然只在 'cuda:1' 上创建了主上下文
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        # 用未初始化的数据创建一个张量，并将其固定在内存中
        x = torch.empty(3, device="cpu", pin_memory=True)

        # 断言：应该仍然只在 'cuda:1' 上创建了主上下文
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        # 将张量 x 固定在内存中
        x = x.pin_memory()

        # 断言：应该仍然只在 'cuda:1' 上创建了主上下文
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```