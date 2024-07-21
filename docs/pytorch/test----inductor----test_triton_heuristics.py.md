# `.\pytorch\test\inductor\test_triton_heuristics.py`

```
# Owner(s): ["module: inductor"]

# 导入必要的模块和库
import sys
import unittest

import torch

# 导入内部测试和工具函数
from torch.testing._internal.common_utils import IS_LINUX, skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU

# 尝试导入 triton，如果失败则根据条件退出或跳过测试
try:
    import triton  # noqa: F401
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904

# 导入内部配置和相关功能模块
from torch._inductor import config
from torch._inductor.runtime.hints import TRITON_MAX_BLOCK
from torch._inductor.runtime.triton_heuristics import triton_config
from torch._inductor.test_case import run_tests, TestCase


class TestTritonHeuristics(TestCase):
    device_type = GPU_TYPE

    def test_triton_config(self):
        """
        确保块大小不超过在inductor配置中定义的最大值。
        """
        # 获取 triton 的配置参数
        cfg = triton_config([2048, 2], 64, 64)
        # 遍历 "XYZ"，检查是否在配置中定义块大小，如果没有则跳过
        for label in "XYZ":
            key = f"{label}BLOCK"
            if key not in cfg.kwargs:
                continue
            # 断言块大小不超过预定义的最大块大小
            self.assertTrue(cfg.kwargs[key] <= TRITON_MAX_BLOCK[label])

    def _test_artificial_zgrid(self):
        """
        内部方法，测试处理人工生成的 zgrid 的运算。
        """
        def forward(primals_1, primals_2, primals_5):
            # 对输入进行形状重塑操作
            view = torch.ops.aten.reshape.default(primals_5, [-1, 2, 4])
            primals_5 = None
            # 对重塑后的张量进行排列操作
            permute = torch.ops.aten.permute.default(view, [0, 2, 1])
            clone = torch.ops.aten.clone.default(
                permute, memory_format=torch.contiguous_format
            )
            permute = None
            # 对克隆的张量再次进行形状重塑
            view_1 = torch.ops.aten.reshape.default(clone, [-1, 4])
            clone = None
            # 对输入张量进行排列操作
            permute_1 = torch.ops.aten.permute.default(primals_1, [1, 0])
            primals_1 = None
            # 执行矩阵相乘操作
            addmm = torch.ops.aten.addmm.default(primals_2, view_1, permute_1)
            primals_2 = None
            return addmm

        s0 = 16777472
        s1 = 8

        # 定义测试用例的输入参数
        args = [
            torch.rand([2, 4], device=GPU_TYPE),
            torch.rand([2], device=GPU_TYPE),
            torch.rand([s0, s1], device=GPU_TYPE),
        ]
        # 标记最后一个参数为动态参数
        torch._dynamo.mark_dynamic(args[-1], 0)
        # 编译 forward 函数
        foo_c = torch.compile(forward)

        # 断言直接调用和编译后调用的结果相等
        self.assertEqual(forward(*args), foo_c(*args))

        # 重新定义参数并再次测试
        args = [
            torch.rand([2, 4], device=GPU_TYPE),
            torch.rand([2], device=GPU_TYPE),
            torch.rand([s0, s1], device=GPU_TYPE),
        ]
        self.assertEqual(forward(*args), foo_c(*args))

    @skipIfXpu
    def test_artificial_zgrid(self):
        """
        测试处理人工生成的 zgrid 的运算。
        """
        self._test_artificial_zgrid()

    @skipIfXpu
    @config.patch("cpp_wrapper", True)
    def test_artificial_grid_cpp_wrapper(self):
        """
        测试使用 cpp_wrapper 标记进行编译的人工生成的 zgrid 的运算。
        """
        self._test_artificial_zgrid()


if __name__ == "__main__":
    # 如果在 Linux 并且有 GPU，运行测试
    if IS_LINUX and HAS_GPU:
        run_tests()
```