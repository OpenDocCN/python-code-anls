# `.\pytorch\test\test_import_stats.py`

```
# Owner(s): ["module: ci"]

# 从 torch.testing._internal.common_utils 导入 TestCase 类和 run_tests 函数
from torch.testing._internal.common_utils import TestCase, run_tests

# 这些测试最终可以更改为在导入/初始化时间超过某个阈值时失败，
# 但目前我们只是用它们来跟踪 `import torch` 的持续时间。

# 定义一个测试类 TestImportTime，继承自 TestCase 类
class TestImportTime(TestCase):
    
    # 测试导入 torch 所需的时间
    def test_time_import_torch(self):
        # 使用 TestCase 类的方法运行带有 PyTorch API 使用情况标准错误输出的代码
        TestCase.runWithPytorchAPIUsageStderr("import torch")

    # 测试获取 CUDA 设备数量所需的时间
    def test_time_cuda_device_count(self):
        # 使用 TestCase 类的方法运行带有 PyTorch API 使用情况标准错误输出的代码
        TestCase.runWithPytorchAPIUsageStderr(
            "import torch; torch.cuda.device_count()",
        )

# 如果该脚本作为主程序运行，则执行所有测试
if __name__ == "__main__":
    run_tests()
```