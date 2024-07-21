# `.\pytorch\test\test_jit_profiling.py`

```py
# 引入 sys 模块，用于访问系统相关功能
import sys

# 向命令行参数列表中添加一个选项，指定 JIT 执行器为 profiling 模式
sys.argv.append("--jit-executor=profiling")

# 从 test_jit 模块中导入所有内容，禁止 Flake8 F403 错误的警告
from test_jit import *  # noqa: F403

# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 运行测试函数
    run_tests()
```