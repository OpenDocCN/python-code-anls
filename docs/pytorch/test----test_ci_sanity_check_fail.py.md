# `.\pytorch\test\test_ci_sanity_check_fail.py`

```py
# Owner(s): ["module: ci"]
# Sanity check for CI setup in GHA.  This file is expected to fail so it can trigger reruns

import os  # 导入操作系统接口模块

from torch.testing._internal.common_utils import run_tests, slowTest, TestCase  # 导入测试运行函数和装饰器，以及测试用例基类


class TestCISanityCheck(TestCase):
    def test_env_vars_exist(self):
        # This check should fail and trigger reruns.  If it passes, something is wrong
        # 检查环境变量 "CI" 是否为 None，预期结果是失败以触发重新运行测试；如果通过了，说明有问题
        self.assertTrue(os.environ.get("CI") is None)

    @slowTest
    def test_env_vars_exist_slow(self):
        # Same as the above, but for the slow suite
        # 与上面的测试相同，但用于慢速测试套件
        self.assertTrue(os.environ.get("CI") is None)


if __name__ == "__main__":
    run_tests()  # 运行测试
```