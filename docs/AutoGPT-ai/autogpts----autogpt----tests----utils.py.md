# `.\AutoGPT\autogpts\autogpt\tests\utils.py`

```py
# 导入 os 模块
import os

# 导入 pytest 模块
import pytest

# 定义装饰器函数，用于在 CI 环境下跳过测试函数
def skip_in_ci(test_function):
    # 使用 pytest.mark.skipif 装饰器，判断是否在 CI 环境下
    return pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="This test doesn't work on GitHub Actions.",
    )(test_function)

# 获取工作空间中指定文件的路径
def get_workspace_file_path(workspace, file_name):
    # 将文件名与工作空间路径拼接成完整的文件路径，并返回字符串类型
    return str(workspace.get_path(file_name))
```