# `.\pytorch\tools\test\test_upload_test_stats.py`

```
import os  # 导入操作系统模块
import unittest  # 导入单元测试模块

from tools.stats.upload_test_stats import get_tests, summarize_test_cases  # 从工具包中导入测试统计相关函数


IN_CI = os.environ.get("CI")  # 获取环境变量 CI 的值


class TestUploadTestStats(unittest.TestCase):  # 定义测试类 TestUploadTestStats，继承自 unittest.TestCase
    @unittest.skipIf(  # 使用 unittest 装饰器，条件跳过测试
        IN_CI,
        "don't run in CI as this does a lot of network calls and uses up GH API rate limit",
    )
    def test_existing_job(self) -> None:  # 定义测试方法 test_existing_job，返回 None
        """Run on a known-good job and make sure we don't error and get basically okay results."""
        test_cases = get_tests(2561394934, 1)  # 调用 get_tests 函数获取测试用例
        self.assertEqual(len(test_cases), 609873)  # 断言获取的测试用例数量为 609873
        summary = summarize_test_cases(test_cases)  # 调用 summarize_test_cases 函数对测试用例进行总结
        self.assertEqual(len(summary), 5068)  # 断言总结结果的长度为 5068


if __name__ == "__main__":  # 判断当前脚本是否为主程序入口
    unittest.main()  # 运行单元测试
```