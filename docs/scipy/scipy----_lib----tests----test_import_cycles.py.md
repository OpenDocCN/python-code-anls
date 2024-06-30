# `D:\src\scipysrc\scipy\scipy\_lib\tests\test_import_cycles.py`

```
# 导入 pytest 模块，用于编写和运行测试用例
import pytest
# 导入 sys 模块，用于与 Python 解释器交互，例如获取解释器路径
import sys
# 导入 subprocess 模块，用于在新的进程中执行命令或脚本
import subprocess

# 从当前目录的 test_public_api 模块中导入 PUBLIC_MODULES 常量
from .test_public_api import PUBLIC_MODULES

# Regression tests for gh-6793.
# 对 gh-6793 的回归测试。
# Check that all modules are importable in a new Python process.
# 检查所有模块是否可以在新的 Python 进程中导入。
# This is not necessarily true if there are import cycles present.
# 如果存在导入循环，这不一定是真实情况。

# 使用 pytest.mark.fail_slow(40) 和 pytest.mark.slow 装饰器标记测试用例为失败慢和慢速测试
@pytest.mark.fail_slow(40)
@pytest.mark.slow
def test_public_modules_importable():
    # 创建一个子进程列表，每个子进程尝试导入 PUBLIC_MODULES 中的一个模块
    pids = [subprocess.Popen([sys.executable, '-c', f'import {module}'])
            for module in PUBLIC_MODULES]
    # 遍历子进程列表
    for i, pid in enumerate(pids):
        # 等待子进程结束，并检查其返回码是否为 0（导入成功）
        assert pid.wait() == 0, f'Failed to import {PUBLIC_MODULES[i]}'
```