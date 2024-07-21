# `.\pytorch\test\test_compile_benchmark_util.py`

```py
# Owner(s): ["module: dynamo"]

# 引入 unittest 模块，用于编写和运行测试
import unittest

# 引入 torch 库
import torch
# 引入 torch._dynamo 模块，这是一个私有的 Dynamo 模块
import torch._dynamo as torchdynamo
# 从 torch.testing._internal.common_utils 模块中引入 run_tests、TEST_CUDA 和 TestCase
from torch.testing._internal.common_utils import run_tests, TEST_CUDA, TestCase

try:
    # 尝试引入 tabulate 库，用于表格化数据显示
    import tabulate  # noqa: F401  # type: ignore[import]
    # 从 torch.utils.benchmark.utils.compile 模块中引入 bench_all 函数
    from torch.utils.benchmark.utils.compile import bench_all
    # 标记是否成功引入 tabulate 库
    HAS_TABULATE = True
except ImportError:
    # 如果引入失败，则将 HAS_TABULATE 标记为 False
    HAS_TABULATE = False


# 跳过测试，如果 CUDA 不可用或者 tabulate 不可用
@unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
@unittest.skipIf(not HAS_TABULATE, "tabulate not available")
class TestCompileBenchmarkUtil(TestCase):
    # 定义测试用例类 TestCompileBenchmarkUtil，继承自 unittest.TestCase
    def test_training_and_inference(self):
        # 定义一个简单的神经网络模型 ToyModel
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.Tensor(2, 2))

            def forward(self, x):
                return x * self.weight

        # 重置 torchdynamo 状态
        torchdynamo.reset()
        # 创建 ToyModel 的实例，并将其移动到 CUDA 设备上
        model = ToyModel().cuda()

        # 执行推断性能测试，并生成推断性能表格
        inference_table = bench_all(model, torch.ones(1024, 2, 2).cuda(), 5)
        # 断言表格中包含 "Inference"、"Eager" 和 "-"，表明测试通过
        self.assertTrue(
            "Inference" in inference_table
            and "Eager" in inference_table
            and "-" in inference_table
        )

        # 执行训练性能测试，并生成训练性能表格
        training_table = bench_all(
            model,
            torch.ones(1024, 2, 2).cuda(),
            5,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        )
        # 断言表格中包含 "Train"、"Eager" 和 "-"，表明测试通过
        self.assertTrue(
            "Train" in training_table
            and "Eager" in training_table
            and "-" in training_table
        )


# 如果当前脚本作为主程序运行，则执行所有测试
if __name__ == "__main__":
    run_tests()
```