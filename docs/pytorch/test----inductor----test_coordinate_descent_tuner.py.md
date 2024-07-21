# `.\pytorch\test\inductor\test_coordinate_descent_tuner.py`

```
# Owner(s): ["module: inductor"]

# 导入必要的模块和库
import sys
import unittest
from unittest import mock

import torch
from torch._inductor.runtime.hints import TRITON_MAX_BLOCK

# 导入测试相关的模块和函数
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU

# 尝试导入 triton 库，如果失败则根据条件决定是否退出或跳过测试
try:
    import triton
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904

# 导入配置和 CoordescTuner 类
from torch._inductor import config
from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner

# 设置配置选项
config.benchmark_kernel = True
config.coordinate_descent_tuning = True

# 保存原始的 compare_config 方法
orig_compare_config = CoordescTuner.compare_config

# 定义用于模拟 compare_config 的函数，优先考虑较大的 XBLOCK 值
def mock_compare_config_prefer_larger_XBLOCK(
    self, func, candidate_config, best_config, best_timing
):
    """
    self is the CoordescTuner object
    """
    if "XBLOCK" in candidate_config.kwargs:
        assert "XBLOCK" in best_config.kwargs
        # 如果候选配置的 XBLOCK 小于最佳配置的 XBLOCK，则选择候选配置
        if candidate_config.kwargs["XBLOCK"] < best_config.kwargs["XBLOCK"]:
            func(candidate_config)  # 运行 func 以创建启动器
            return False, best_timing * 1.1
        # 如果候选配置的 XBLOCK 大于最佳配置的 XBLOCK，则选择最佳配置
        elif candidate_config.kwargs["XBLOCK"] > best_config.kwargs["XBLOCK"]:
            func(candidate_config)
            return True, best_timing * 0.9

    # 使用原始的 compare_config 方法进行比较
    return orig_compare_config(self, func, candidate_config, best_config, best_timing)


class TestCoordinateDescentTuner(TestCase):
    def test_abs_function(self):
        """
        The benchmark result is simply abs(XBLOCK - 15)
        """
        # 创建 CoordescTuner 对象用于测试
        tuner = CoordescTuner()
        # 定义基准配置
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        # 定义测试函数 func，计算 abs(XBLOCK - 15) 的值
        def func(config):
            return abs(config.kwargs["XBLOCK"] - 15)

        # 进行自动调优，找到使得 abs(XBLOCK - 15) 最小的配置
        best_config = tuner.autotune(func, baseline_config)
        self.assertTrue(best_config.kwargs.get("XBLOCK") == 16, str(best_config))

    def test_no_neighbors(self):
        """
        Test the case that there is no available neighbor values for a field.
        """
        # 创建 CoordescTuner 对象，设置 size_hints=[1] 限制 XBLOCK 的最大值为 1
        tuner = CoordescTuner(size_hints=[1])
        # 定义基准配置
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        # 定义测试函数 func，计算 abs(XBLOCK - 15) 的值
        def func(config):
            return abs(config.kwargs["XBLOCK"] - 15)

        # 进行自动调优，由于只有 XBLOCK=1 可供选择，因此最佳配置仍然是 XBLOCK=1
        best_config = tuner.autotune(func, baseline_config)
        self.assertTrue(best_config.kwargs.get("XBLOCK") == 1, str(best_config))

    def test_get_neighbour_values(self):
        # 创建 CoordescTuner 对象用于测试
        tuner = CoordescTuner()

        # 测试获取 num_stages 字段的邻居值，半径为 2
        neighbours = tuner.get_neighbour_values("num_stages", 2, radius=2)
        self.assertEqual(set(neighbours), {1, 3, 4})

        # 测试获取 num_warps 字段的邻居值，半径为 2
        neighbours = tuner.get_neighbour_values("num_warps", 2, radius=2)
        self.assertEqual(set(neighbours), {1, 4, 8})
    def test_persistent_reduction(self):
        # 定义一个函数 f，用于计算张量 x 按最后一个维度进行归一化后的结果
        def f(x):
            return x / x.sum(dim=-1, keepdim=True)

        # 使用 mock.patch.object 来模拟 CoordescTuner 类中的 compare_config 方法
        # 替换为 mock_compare_config_prefer_larger_XBLOCK 方法
        with mock.patch.object(
            CoordescTuner, "compare_config", mock_compare_config_prefer_larger_XBLOCK
        ):
            # 创建一个大小为 (2, 256) 的全一张量，并移动到 GPU 上
            x = torch.ones(2, 256).to(GPU_TYPE)
            # 计算使用函数 f 对 x 的预期结果
            expected = f(x)
            # 第一次调用 torch.compile(f)(x)，可能由于缓存未命中而得到正确结果，原因尚不清楚
            _ = torch.compile(f)(x)
            # 第二次调用 torch.compile(f)(x)，获取实际结果
            actual = torch.compile(f)(x)
            # 断言预期结果与实际结果的接近程度，使用绝对误差和相对误差作为容忍度
            self.assertTrue(
                torch.allclose(expected, actual, atol=1e-4, rtol=1e-4),
                f"Expected:\n{expected}\nActual:\n{actual}",
            )

    def test_value_too_large(self):
        # 定义一个大小提示列表，每个值为 2 的 20 次方
        size_hints = [2**20, 2**20]

        # 创建一个 CoordescTuner 对象，使用 size_hints 作为大小提示
        tuner = CoordescTuner(size_hints=size_hints)

        # 获取 TRITON_MAX_BLOCK 中的最大 XBLOCK 大小
        max_block = TRITON_MAX_BLOCK
        # 断言 XBLOCK 的大小不超过最大值 max_block["X"]
        self.assertFalse(tuner.value_too_large("XBLOCK", max_block["X"]))
        # 断言 2 倍于 XBLOCK 的大小超过了最大值 max_block["X"]
        self.assertTrue(tuner.value_too_large("XBLOCK", max_block["X"] * 2))
        # 断言 RBLOCK 的大小不超过最大值 max_block["R"]
        self.assertFalse(tuner.value_too_large("RBLOCK", max_block["R"]))
        # 断言 2 倍于 RBLOCK 的大小超过了最大值 max_block["R"]
        self.assertTrue(tuner.value_too_large("RBLOCK", max_block["R"] * 2))
# 如果当前脚本作为主程序运行（而不是被导入），则执行以下代码块
if __name__ == "__main__":
    # 如果操作系统是 Linux 并且有 GPU 资源可用
    if IS_LINUX and HAS_GPU:
        # 运行测试函数
        run_tests()
```