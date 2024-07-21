# `.\pytorch\test\test_logging.py`

```py
# Owner(s): ["module: unknown"]

# 导入 torch 库
import torch
# 从 torch.testing._internal.common_utils 导入 run_tests 和 TestCase 类
from torch.testing._internal.common_utils import run_tests, TestCase

# 定义 LoggingTest 类，继承自 TestCase
class LoggingTest(TestCase):
    # 定义 testApiUsage 方法，用于测试 API 使用情况
    def testApiUsage(self):
        """
        This test verifies that api usage logging is not triggered via static
        initialization. Since it's triggered at first invocation only - we just
        subprocess
        """
        # 运行带有 PyTorch API 使用日志的标准错误输出的测试，验证 import torch 语句
        s = TestCase.runWithPytorchAPIUsageStderr("import torch")
        # 断言 s 中应包含 "PYTORCH_API_USAGE" 和 "import" 字符串
        self.assertRegex(s, "PYTORCH_API_USAGE.*import")
        
        # 直接导入共享库（shared library） - 触发静态初始化但不调用任何内容
        s = TestCase.runWithPytorchAPIUsageStderr(
            f"from ctypes import CDLL; CDLL('{torch._C.__file__}')"
        )
        # 断言 s 中不应包含 "PYTORCH_API_USAGE" 字符串
        self.assertNotRegex(s, "PYTORCH_API_USAGE")

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```